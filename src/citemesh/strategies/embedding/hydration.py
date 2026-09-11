"""Corpus cache hydration for the embedding graph builder.

Owns the full-corpus and exact-slice hydration passes over the arXiv dataset:
resume of an interrupted cache, newest-revision row selection, int8 calibration
sampling, batched metadata encoding, and the cached-corpus search that backs
retrieval. ``HYDRATION_FLUSH_SIZE`` lives here, next to the loops that read it.
"""

from __future__ import annotations

import logging
import os
import random
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import islice
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from citemesh.data.embedding_cache import EMBEDDING_DATASET_CHUNK_ROWS
from citemesh.progress import progress_task
from citemesh.strategies.base import deterministic_sort_key

from . import deps
from .config import (
    CALIBRATION_RESERVOIR_SEED,
    CANDIDATE_MULTIPLIER,
)
from .records import (
    _arxiv_id_chronology_key,
    _extract_dataset_paper_metadata,
    _HydrationSourceSliceResult,
    _newest_records_by_arxiv_id,
)

logger = logging.getLogger(__name__)

HYDRATION_FLUSH_SIZE = EMBEDDING_DATASET_CHUNK_ROWS


class _CorpusHydrationMixin:
    """Dataset hydration, calibration and cache-native candidate search.

    Mixed into :class:`~citemesh.strategies.embedding.builder.EmbeddingGraphBuilder`.

    Requires the host to provide: ``model``, ``model_profile``, ``client``,
    ``embedding_cache``, ``dataset_source``, ``dataset_split``,
    ``semantic_source``, ``storage_precision``, ``max_corpus_papers``,
    ``calibration_sample_size``, ``encode_batch_size``, ``candidate_pool_size``,
    ``_cache_lock``, ``_cache_hydrated``, ``_hydrated_cache_spec``,
    ``_last_search_used_binary_prefilter``, ``_load_model``, ``_encode_texts``,
    ``_get_model_for_encoding``, ``_ensure_cache_model_fingerprint``,
    ``_embedding_runtime_metadata`` and ``_format_retrieval_document_metadata``.
    """

    def _search_cache_candidates(
        self, seed_embedding: np.ndarray
    ) -> List[Tuple[str, Dict, np.ndarray]]:
        """Run cache-native retrieval and map results to candidate tuples.

        :param np.ndarray seed_embedding: Normalized seed embedding vector.
        :return List[Tuple[str, Dict, np.ndarray]]: Candidate tuples ordered by score.
        """
        top_k = max(self.max_papers * CANDIDATE_MULTIPLIER, self.max_papers)
        search_results = self.embedding_cache.search(
            query_embedding=np.asarray(seed_embedding, dtype=np.float32),
            top_k=top_k,
            binary_prefilter=self.binary_prefilter,
            binary_rescore_multiplier=self.binary_rescore_multiplier,
        )
        self._last_search_used_binary_prefilter = (
            self.embedding_cache.last_search_used_binary_prefilter
        )

        scored_candidates = [
            (
                float(result.score),
                str(result.paper_id),
                dict(result.metadata),
                np.asarray(result.embedding, dtype=np.float32),
                idx,
            )
            for idx, result in enumerate(search_results)
        ]
        scored_candidates.sort(
            key=lambda item: deterministic_sort_key(
                item[0], item[1], stable_index=item[4]
            )
        )
        limited = min(top_k, len(scored_candidates))
        compared_embeddings = getattr(
            self.embedding_cache, "last_search_total_embeddings", None
        )
        rescored_embeddings = getattr(
            self.embedding_cache, "last_search_rescored_embeddings", None
        )
        if compared_embeddings is not None:
            prefilter_used = (
                self._last_search_used_binary_prefilter
                if self._last_search_used_binary_prefilter is not None
                else False
            )
            compared_label = f"{int(compared_embeddings):,}"
            rescored_label = (
                f"{int(rescored_embeddings):,}"
                if rescored_embeddings is not None
                else "unknown"
            )
            logger.info(
                "Semantic cache search compared against %s embeddings "
                "(rescored=%s, prefilter=%s).",
                compared_label,
                rescored_label,
                "on" if prefilter_used else "off",
            )

        return [
            (paper_id, metadata, embedding)
            for _, paper_id, metadata, embedding, _ in scored_candidates[:limited]
        ]

    def _ensure_cache_hydrated(self, use_streaming: bool) -> None:
        """Ensure cache contains hydrated corpus embeddings for current split/cap.

        :param bool use_streaming: Whether to use streaming dataset hydration.
        :return None: Mutates cache state in-place when hydration is required.
        """
        self._ensure_cache_model_fingerprint()
        with self.embedding_cache.hydration_operation_lock():
            self._ensure_cache_hydrated_locked(use_streaming=use_streaming)

    def _ensure_cache_hydrated_locked(self, use_streaming: bool) -> None:
        """Hydrate the active corpus while its operation lock is held.

        :param bool use_streaming: Whether to use streaming dataset hydration.
        :return None: Mutates cache state in-place when hydration is required.
        """
        cached_dataset_source = self.embedding_cache.get_hydrated_dataset_source()
        if self.embedding_cache.is_hydrated(
            self.dataset_split,
            self.corpus_size,
            dataset_source=self.dataset_source,
        ):
            self._refresh_cached_corpus_metadata(cached_dataset_source, use_streaming)
            self._refresh_hydrated_full_corpus_cache(
                use_streaming=use_streaming,
                cached_dataset_source=cached_dataset_source,
            )
            cached_dataset_source = self.embedding_cache.get_hydrated_dataset_source()
            if self.embedding_cache.is_hydrated(
                self.dataset_split,
                self.corpus_size,
                dataset_source=self.dataset_source,
            ):
                logger.debug(
                    "Embedding cache already hydrated for split=%s corpus_size=%s source=%s; "
                    "skipping dataset load.",
                    self.dataset_split,
                    "all" if self.corpus_size is None else self.corpus_size,
                    cached_dataset_source or "unknown",
                )
                return
            logger.warning(
                "Hydrated cache revalidation invalidated source=%s for split=%s corpus_size=%s; "
                "performing full source revalidation.",
                cached_dataset_source or "unknown",
                self.dataset_split,
                "all" if self.corpus_size is None else self.corpus_size,
            )

        if self._resume_incomplete_full_corpus_cache(
            use_streaming=use_streaming,
            cached_dataset_source=cached_dataset_source,
        ):
            self._refresh_cached_corpus_metadata(cached_dataset_source, use_streaming)
            return

        dataset_source: Optional[str]
        dataset: Iterable[Dict[str, Any]]

        if cached_dataset_source or self.embedding_cache.has_cached_payload():
            logger.warning(
                "REBUILDING EMBEDDING CACHE — please hang tight. "
                "Embeddings will be regenerated automatically; this may take a while. "
                "No action is needed. "
                "Reason: the cached corpus no longer matches the requested source, "
                "split, size, or hydration state. Preparing the dataset first."
            )

        try:
            dataset_source, dataset = self._load_dataset_for_hydration(
                use_streaming=use_streaming,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to resolve hydration dataset source {self.dataset_source!r}; refusing to reuse "
                "existing hydrated cache without source revalidation."
            ) from exc

        if self.embedding_cache.is_hydrated(
            self.dataset_split,
            self.corpus_size,
            dataset_source=dataset_source,
        ):
            self._refresh_cached_corpus_metadata(dataset_source, use_streaming)
            return

        logger.info(
            "Hydrating embedding cache for split=%s corpus_size=%s (streaming=%s).",
            self.dataset_split,
            "all" if self.corpus_size is None else self.corpus_size,
            use_streaming,
        )
        if self.corpus_size is not None:
            logger.info(
                "Capped corpus hydration selects the %d most recently "
                "submitted papers (by arXiv ID chronology). Use --all-corpus "
                "for full coverage.",
                int(self.corpus_size),
            )
        retained_fingerprint = (
            str(self._resolved_model_fingerprint).strip()
            if self._resolved_model_fingerprint is not None
            else ""
        )
        if not retained_fingerprint:
            fallback_fingerprint = self.embedding_cache.get_model_fingerprint()
            retained_fingerprint = (
                str(fallback_fingerprint).strip()
                if fallback_fingerprint is not None
                else ""
            )
        self._clear_embedding_cache(
            "hydration metadata mismatch requires rebuild "
            f"(requested_split={self.dataset_split}, "
            f"requested_corpus={'all' if self.corpus_size is None else self.corpus_size}, "
            f"resolved_source={dataset_source}, cached_source={cached_dataset_source or 'unknown'})"
        )
        if retained_fingerprint:
            self.embedding_cache.set_model_fingerprint(retained_fingerprint)
        self.embedding_cache.mark_hydrated(
            dataset_source=dataset_source,
            dataset_split=self.dataset_split,
            corpus_size=self.corpus_size,
            complete=False,
        )
        # The cleared namespace contains only rows from the current adapter,
        # including when this hydration is interrupted and resumed later.
        self.embedding_cache.mark_corpus_metadata_current()

        progress_total = self._resolve_hydration_progress_total(
            dataset,
            use_streaming=use_streaming,
        )
        self._ensure_int8_calibration_ranges(
            use_streaming=use_streaming,
            dataset_source=dataset_source,
            selected_dataset=(
                dataset if not use_streaming or isinstance(dataset, list) else None
            ),
        )

        hydrated_records = self._hydrate_dataset_records(
            dataset=dataset,
            progress_total=progress_total,
            progress_label="Hydrating dataset",
        )

        if hydrated_records == 0:
            logger.warning(
                "Hydration produced zero records for split=%s corpus_size=%s; "
                "cache remains incomplete.",
                self.dataset_split,
                "all" if self.corpus_size is None else self.corpus_size,
            )
            return

        if self.corpus_size is None and ":" not in str(self.dataset_split):
            upstream_rows = self._resolve_dataset_split_row_count(dataset_source)
            updated_rows = self._cached_payload_row_count()
            rows_reconciled = self._finalize_full_corpus_hydration_rows(
                source=dataset_source,
                updated_rows=updated_rows,
                upstream_rows=upstream_rows,
                mark_complete=True,
            )
            if not rows_reconciled:
                logger.info(
                    "Initial full-corpus hydration for %s/%s completed with "
                    "cache_rows=%d and upstream_rows=%d; recording the expected "
                    "duplicate-ID row-count deficit.",
                    dataset_source,
                    self.dataset_split,
                    updated_rows,
                    upstream_rows,
                )
            return

        self.embedding_cache.mark_hydrated(
            dataset_source=dataset_source,
            dataset_split=self.dataset_split,
            corpus_size=self.corpus_size,
            complete=True,
        )

    def _refresh_cached_corpus_metadata(
        self, source: Optional[str], use_streaming: bool
    ) -> None:
        """Backfill corpus years and DOIs without changing persisted vectors.

        :param Optional[str] source: Dataset recorded on the matching cache.
        :param bool use_streaming: Whether source rows should be streamed.
        :return None: Refreshes existing SQLite rows once after an adapter change.
        """
        cache = self.embedding_cache
        if (
            not source
            or cache.has_current_corpus_metadata()
            or not cache.has_cached_payload()
        ):
            return

        logger.info("Refreshing cached publication years and DOIs from %s.", source)
        # A capped cache retains its original paper selection. Inspect the full
        # selected split so older cached papers can still receive metadata fixes.
        dataset = deps._import_datasets_module().load_dataset(
            source,
            split=self.dataset_split,
            streaming=use_streaming,
            num_proc=None if use_streaming else max(1, (os.cpu_count() or 1) // 2),
        )
        batch: List[Dict] = []
        for index, record in enumerate(dataset):
            batch.append(_extract_dataset_paper_metadata(record, index))
            if len(batch) >= HYDRATION_FLUSH_SIZE:
                cache.update_corpus_metadata(batch)
                batch = []
        if batch:
            cache.update_corpus_metadata(batch)
        # Older resume code could memoize a deficit without reconciling IDs.
        cache.clear_hydration_rowcount_reconciliation()
        cache.mark_corpus_metadata_current()

    def _resume_incomplete_full_corpus_cache(
        self,
        *,
        use_streaming: bool,
        cached_dataset_source: Optional[str],
    ) -> bool:
        """Resume an incomplete corpus hydration when cached rows are reusable.

        :param bool use_streaming: Whether hydration mode is streaming.
        :param Optional[str] cached_dataset_source: Dataset source recorded on the
            incomplete cache attempt.
        :return bool: ``True`` when the incomplete cache was resumed or safely
            retained without requiring a full namespace clear.
        """
        source = str(cached_dataset_source or "").strip()
        if source != self.dataset_source:
            return False

        stats = self.embedding_cache.payload_stats()
        if stats.hydration_complete:
            return False
        if stats.hydration_split != self.dataset_split:
            return False
        expected_corpus_size = (
            "all" if self.corpus_size is None else f"newest:{int(self.corpus_size)}"
        )
        if stats.hydration_corpus_size != expected_corpus_size:
            return False
        if stats.hydration_dataset_source != source:
            return False
        if stats.sqlite_rows < 1 or stats.embedding_rows < 1:
            return False
        if stats.sqlite_rows != stats.embedding_rows:
            logger.warning(
                "Incomplete full-corpus cache rows diverged for %s/%s "
                "(sqlite_rows=%d, embedding_rows=%d); performing full rebuild.",
                source,
                self.dataset_split,
                stats.sqlite_rows,
                stats.embedding_rows,
            )
            return False
        if (
            self.storage_precision == "int8"
            and not self.embedding_cache.has_calibration_ranges()
        ):
            logger.warning(
                "Incomplete full-corpus cache for %s/%s is missing int8 calibration "
                "ranges; performing full rebuild.",
                source,
                self.dataset_split,
            )
            return False

        if self.corpus_size is not None or ":" in str(self.dataset_split):
            cached_paper_ids = self.embedding_cache.get_cached_paper_ids()
            logger.info(
                "Resuming incomplete selected-corpus cache for %s/%s from "
                "cached_rows=%d.",
                source,
                self.dataset_split,
                stats.sqlite_rows,
            )
            self._ensure_int8_calibration_ranges(
                use_streaming=use_streaming,
                dataset_source=source,
            )
            resume_result = self._hydrate_exact_hydration_source_slice(
                use_streaming=use_streaming,
                source=source,
                progress_total=self.corpus_size,
                progress_label="Resuming dataset",
                operation="Incomplete hydration resume",
                existing_paper_ids=cached_paper_ids,
            )
            if not resume_result.source_exhausted:
                logger.warning(
                    "Incomplete selected-corpus resume for %s/%s did not exhaust "
                    "its source; performing full rebuild.",
                    source,
                    self.dataset_split,
                )
                return False
            updated_rows = self._cached_payload_row_count()
            if self.corpus_size is not None and updated_rows > self.corpus_size:
                logger.warning(
                    "Resumed capped corpus has %d cached rows, exceeding "
                    "--corpus-size %d after the source selection changed. "
                    "Retained existing vectors; rebuild the corpus to apply "
                    "the cap exactly.",
                    updated_rows,
                    self.corpus_size,
                )
            self.embedding_cache.mark_hydrated(
                dataset_source=source,
                dataset_split=self.dataset_split,
                corpus_size=self.corpus_size,
                complete=True,
            )
            logger.info(
                "Resumed incomplete selected-corpus cache for %s/%s "
                "(source_rows=%d, added=%d, cache_rows=%d).",
                source,
                self.dataset_split,
                resume_result.source_rows_consumed,
                resume_result.hydrated_records,
                updated_rows,
            )
            return True

        cached_rows = int(stats.sqlite_rows)
        upstream_rows = self._resolve_dataset_split_row_count(source)
        if upstream_rows is not None:
            if upstream_rows < cached_rows:
                logger.warning(
                    "Incomplete full-corpus cache rows (%d) exceed upstream split rows "
                    "(%d) for %s/%s; performing full rebuild.",
                    cached_rows,
                    upstream_rows,
                    source,
                    self.dataset_split,
                )
                return False
            if upstream_rows == cached_rows:
                logger.info(
                    "Incomplete full-corpus cache for %s/%s already matches upstream "
                    "row count (%d); marking hydration complete.",
                    source,
                    self.dataset_split,
                    cached_rows,
                )
                self._finalize_full_corpus_hydration_rows(
                    source=source,
                    updated_rows=cached_rows,
                    upstream_rows=upstream_rows,
                    mark_complete=True,
                )
                return True

        row_limit = None if upstream_rows is None else upstream_rows - cached_rows
        logger.info(
            "Resuming incomplete full-corpus cache for %s/%s from cached_rows=%d%s.",
            source,
            self.dataset_split,
            cached_rows,
            "" if upstream_rows is None else f" toward upstream_rows={upstream_rows}",
        )
        self._ensure_int8_calibration_ranges(
            use_streaming=use_streaming,
            dataset_source=source,
        )
        resume_result = self._hydrate_exact_hydration_source_slice(
            use_streaming=use_streaming,
            source=source,
            row_limit=row_limit,
            row_offset=cached_rows,
            progress_total=row_limit,
            progress_label="Resuming dataset",
            operation="Incomplete hydration resume",
        )
        resumed_records = resume_result.hydrated_records
        updated_rows = self._cached_payload_row_count()
        if updated_rows < cached_rows:
            raise RuntimeError(
                "Incomplete hydration resume reduced cached row count unexpectedly "
                f"({updated_rows} < {cached_rows})."
            )

        if upstream_rows is not None:
            expected_source_rows = int(row_limit or 0)
            if (
                not resume_result.source_exhausted
                or resume_result.source_rows_consumed != expected_source_rows
            ):
                logger.warning(
                    "Incomplete full-corpus resume for %s/%s consumed %d of %d "
                    "expected source rows (slice_exhausted=%s); performing full "
                    "rebuild.",
                    source,
                    self.dataset_split,
                    resume_result.source_rows_consumed,
                    expected_source_rows,
                    resume_result.source_exhausted,
                )
                return False
        elif not resume_result.source_exhausted:
            logger.warning(
                "Incomplete full-corpus resume for %s/%s did not exhaust its "
                "unknown-cardinality source slice; performing full revalidation.",
                source,
                self.dataset_split,
            )
            return False

        # Cached unique IDs are not a source offset after reordered growth.
        # Exhausting the tail alone cannot establish a duplicate-ID deficit.
        if upstream_rows is None or updated_rows < upstream_rows:
            reconciled = self._hydrate_exact_hydration_source_slice(
                use_streaming=use_streaming,
                source=source,
                progress_total=upstream_rows,
                progress_label="Reconciling dataset",
                operation="Resume missing-ID reconciliation",
                existing_paper_ids=self.embedding_cache.get_cached_paper_ids(),
            )
            resumed_records += reconciled.hydrated_records
            updated_rows = self._cached_payload_row_count()

        rows_reconciled = self._finalize_full_corpus_hydration_rows(
            source=source,
            updated_rows=updated_rows,
            upstream_rows=upstream_rows,
            mark_complete=True,
        )
        if not rows_reconciled:
            logger.info(
                "Incomplete full-corpus resume for %s/%s exhausted its expected "
                "source and reconciled missing IDs with cache_rows=%d and upstream_rows=%d; recording "
                "the duplicate/invalid-ID row-count deficit.",
                source,
                self.dataset_split,
                updated_rows,
                upstream_rows,
            )
        logger.info(
            "Resumed incomplete full-corpus cache for %s/%s "
            "(source_rows=%d, added=%d, cache_rows=%d).",
            source,
            self.dataset_split,
            resume_result.source_rows_consumed,
            resumed_records,
            updated_rows,
        )
        return True

    def _load_exact_hydration_source_slice(
        self,
        *,
        use_streaming: bool,
        source: str,
        operation: str,
        row_limit: Optional[int] = None,
        row_offset: Optional[int] = None,
    ) -> Iterable[Dict[str, Any]]:
        """Load a hydration slice while requiring the recorded source exactly.

        :param bool use_streaming: Whether to load a streaming dataset iterator.
        :param str source: Previously recorded dataset source to load.
        :param str operation: Caller-facing operation name for mismatch errors.
        :param Optional[int] row_limit: Optional number of rows to load.
        :param Optional[int] row_offset: Optional source row offset.
        :return Iterable[Dict[str, Any]]: The exact-source dataset slice.
        :raises RuntimeError: If the loader resolves a different source.
        """
        resolved_source, dataset = self._load_dataset_for_hydration(
            use_streaming=use_streaming,
            row_limit=row_limit,
            row_offset=row_offset,
        )
        if resolved_source != source:
            raise RuntimeError(
                f"{operation} resolved unexpected dataset source {resolved_source!r} "
                f"(expected {source!r})."
            )
        return dataset

    def _hydrate_exact_hydration_source_slice(
        self,
        *,
        use_streaming: bool,
        source: str,
        progress_total: Optional[int],
        progress_label: str,
        operation: str,
        row_limit: Optional[int] = None,
        row_offset: Optional[int] = None,
        existing_paper_ids: Optional[Set[str]] = None,
        max_new_records: Optional[int] = None,
    ) -> _HydrationSourceSliceResult:
        """Load an exact-source slice and report cache and source progress.

        Exceptions from source iteration or cache writes propagate before a
        result is returned, so callers cannot mistake a failed pass for clean EOF.

        :param bool use_streaming: Whether to load a streaming dataset iterator.
        :param str source: Previously recorded dataset source to load.
        :param Optional[int] progress_total: Expected row count for progress display.
        :param str progress_label: Progress-bar description label.
        :param str operation: Caller-facing operation name for mismatch errors.
        :param Optional[int] row_limit: Optional number of rows to load.
        :param Optional[int] row_offset: Optional source row offset.
        :param Optional[Set[str]] existing_paper_ids: IDs to skip during reconciliation.
        :param Optional[int] max_new_records: Optional cap on newly hydrated records.
        :return _HydrationSourceSliceResult: Cache writes and source-consumption state.
        """
        dataset = self._load_exact_hydration_source_slice(
            use_streaming=use_streaming,
            source=source,
            operation=operation,
            row_limit=row_limit,
            row_offset=row_offset,
        )
        source_rows_consumed = 0
        source_exhausted = False

        def tracked_dataset() -> Iterable[Dict[str, Any]]:
            """Yield source rows while recording clean iterator exhaustion.

            :return Iterable[Dict[str, Any]]: Tracked source records.
            """
            nonlocal source_rows_consumed, source_exhausted
            iterator = iter(dataset)
            while True:
                try:
                    raw_record = next(iterator)
                except StopIteration:
                    source_exhausted = True
                    return
                source_rows_consumed += 1
                yield raw_record

        hydrated_records = self._hydrate_dataset_records(
            dataset=tracked_dataset(),
            progress_total=progress_total,
            progress_label=progress_label,
            existing_paper_ids=existing_paper_ids,
            max_new_records=max_new_records,
            fallback_index_offset=int(row_offset or 0),
        )
        return _HydrationSourceSliceResult(
            hydrated_records=hydrated_records,
            source_rows_consumed=source_rows_consumed,
            source_exhausted=source_exhausted,
        )

    def _finalize_full_corpus_hydration_rows(
        self,
        *,
        source: str,
        updated_rows: int,
        upstream_rows: Optional[int],
        mark_complete: bool,
    ) -> bool:
        """Finalize hydration completion and row-count reconciliation metadata.

        :param str source: Exact dataset source used for hydration.
        :param int updated_rows: Current cached payload row count.
        :param Optional[int] upstream_rows: Upstream split row count, if known.
        :param bool mark_complete: Whether this path has completed hydration.
        :return bool: ``True`` when row counts match or cannot be compared.
        """
        if mark_complete:
            self.embedding_cache.mark_hydrated(
                dataset_source=source,
                dataset_split=self.dataset_split,
                corpus_size=self.corpus_size,
                complete=True,
            )
        if upstream_rows is not None and updated_rows < upstream_rows:
            self.embedding_cache.set_hydration_rowcount_reconciliation(
                upstream_rows=upstream_rows,
                cached_rows=updated_rows,
            )
            return False
        self.embedding_cache.clear_hydration_rowcount_reconciliation()
        return True

    def _resolve_hydration_progress_total(
        self, dataset: Iterable[Dict[str, Any]], *, use_streaming: bool
    ) -> Optional[int]:
        """Resolve best-effort progress totals for hydration-related passes.

        :param Iterable[Dict[str, Any]] dataset: Dataset iterable used by the pass.
        :param bool use_streaming: Whether the iterable came from streaming mode.
        :return Optional[int]: Progress-bar total when it can be inferred.
        """
        progress_total = self.corpus_size if self.corpus_size else None
        if not use_streaming and self.corpus_size is None:
            try:
                progress_total = len(dataset)
            except TypeError:  # pragma: no cover - defensive for dataset APIs
                progress_total = None
        return progress_total

    def _needs_explicit_int8_calibration(self) -> bool:
        """Return whether int8 hydration must initialize persisted ranges first.

        :return bool: ``True`` when int8 cache writes would otherwise fail closed.
        """
        return (
            self.storage_precision == "int8"
            and not self.embedding_cache.has_calibration_ranges()
        )

    def _sample_calibration_records(
        self,
        dataset: Iterable[Dict[str, Any]],
        *,
        progress_total: Optional[int],
        progress_label: str,
    ) -> List[Dict]:
        """Reservoir-sample representative metadata records for int8 calibration.

        The sample is deterministic so cache bootstrap remains reproducible under
        tests and across repeated local runs given the same corpus slice/order.

        :param Iterable[Dict[str, Any]] dataset: Dataset records to sample.
        :param Optional[int] progress_total: Optional progress-bar total.
        :param str progress_label: Progress-bar description label.
        :return List[Dict]: Reservoir-sampled metadata records.
        """
        rng = random.Random(CALIBRATION_RESERVOIR_SEED)
        sampled_records: List[Dict] = []

        with progress_task(
            total=progress_total,
            description=progress_label,
            unit="papers",
        ) as progress:
            for idx, raw_record in enumerate(dataset):
                if self.corpus_size is not None and idx >= self.corpus_size:
                    break

                metadata = _extract_dataset_paper_metadata(raw_record, idx)
                if len(sampled_records) < self.calibration_sample_size:
                    sampled_records.append(metadata)
                else:
                    replace_idx = rng.randint(0, idx)
                    if replace_idx < self.calibration_sample_size:
                        sampled_records[replace_idx] = metadata
                progress.update(1)

            if progress_total is None:
                progress.set_postfix_str(f"processed {progress.n}")

        return sampled_records

    def _ensure_int8_calibration_ranges(
        self,
        *,
        use_streaming: bool,
        dataset_source: str,
        selected_dataset: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Initialize representative int8 calibration ranges before hydration writes.

        :param bool use_streaming: Whether the hydration source streams records.
        :param str dataset_source: Resolved dataset source token for hydration.
        :param Optional[Sequence[Dict[str, Any]]] selected_dataset: Already
            loaded, repeatable hydration rows to sample instead of reloading the
            source. This includes non-streaming datasets and capped streaming
            selections that have fully drained the remote stream once.
        :return None: Persists calibration ranges in cache when required.
        :raises RuntimeError: If calibration source resolution or sampling fails.
        """
        if not self._needs_explicit_int8_calibration():
            return

        logger.info(
            "Initializing representative int8 calibration ranges from %s (sample_size=%d).",
            dataset_source,
            self.calibration_sample_size,
        )
        if selected_dataset is not None:
            calibration_dataset: Iterable[Dict[str, Any]] = selected_dataset
            progress_total: Optional[int] = len(selected_dataset)
        else:
            calibration_dataset = self._load_exact_hydration_source_slice(
                use_streaming=use_streaming,
                source=dataset_source,
                operation="Calibration prepass",
            )
            progress_total = self._resolve_hydration_progress_total(
                calibration_dataset,
                use_streaming=use_streaming,
            )

        calibration_records = self._sample_calibration_records(
            calibration_dataset,
            progress_total=progress_total,
            progress_label="Calibrating dataset",
        )
        if not calibration_records:
            logger.warning(
                "Representative int8 calibration prepass produced zero records for %s; "
                "hydration will remain incomplete until a non-empty source is available.",
                dataset_source,
            )
            return
        self._initialize_calibration_ranges(calibration_records)

    def _hydrate_dataset_records(
        self,
        dataset: Iterable[Dict[str, Any]],
        *,
        progress_total: Optional[int],
        progress_label: str,
        existing_paper_ids: Optional[Set[str]] = None,
        max_new_records: Optional[int] = None,
        fallback_index_offset: int = 0,
    ) -> int:
        """Hydrate cache records from dataset iterator without clearing namespace.

        :param Iterable[Dict[str, Any]] dataset: Dataset records to process.
        :param Optional[int] progress_total: Optional progress-bar total.
        :param str progress_label: Progress-bar description label.
        :param Optional[Set[str]] existing_paper_ids: Optional set used to skip
            already-cached paper IDs while hydrating.
        :param Optional[int] max_new_records: Optional cap on newly selected records.
        :param int fallback_index_offset: Source offset for synthetic paper IDs.
        :return int: Number of records routed into cache batching.
        """
        if max_new_records is not None and int(max_new_records) < 1:
            raise ValueError("max_new_records must be at least 1 when provided")

        hydrated_records = 0
        selected_records = 0

        with progress_task(
            total=progress_total,
            description=progress_label,
            unit="papers",
        ) as progress:
            with ThreadPoolExecutor(max_workers=1) as executor:
                batch: List[Dict] = []
                pending_write: Optional[Future[int]] = None
                pending_batch_size = 0
                for local_idx, raw_record in enumerate(dataset):
                    if self.corpus_size is not None and local_idx >= self.corpus_size:
                        break

                    metadata = _extract_dataset_paper_metadata(
                        raw_record,
                        fallback_index_offset + local_idx,
                    )
                    if existing_paper_ids is not None:
                        paper_id = str(metadata.get("paper_id", "")).strip()
                        if not paper_id or paper_id in existing_paper_ids:
                            progress.update(1)
                            continue
                        existing_paper_ids.add(paper_id)

                    selected_records += 1
                    batch.append(metadata)
                    if len(batch) >= HYDRATION_FLUSH_SIZE:
                        if pending_write is not None:
                            hydrated_records += pending_write.result()
                            progress.update(pending_batch_size)
                        pending_batch_size = len(batch)
                        pending_write = executor.submit(
                            self._cache_metadata_batch, batch
                        )
                        batch = []
                    if max_new_records is not None and selected_records >= int(
                        max_new_records
                    ):
                        break

                if batch:
                    if pending_write is not None:
                        hydrated_records += pending_write.result()
                        progress.update(pending_batch_size)
                    pending_batch_size = len(batch)
                    pending_write = executor.submit(self._cache_metadata_batch, batch)

                if pending_write is not None:
                    hydrated_records += pending_write.result()
                    progress.update(pending_batch_size)

            if progress_total is None:
                progress.set_postfix_str(f"processed {progress.n}")

        return hydrated_records

    def _cached_payload_row_count(self) -> int:
        """Return the inspected hydrated payload row count for this namespace.

        :return int: Maximum of SQLite and HDF5 embedding row counts.
        :raises RuntimeError: If either stored row count cannot be read.
        """
        stats = self.embedding_cache.payload_stats()
        return max(int(stats.sqlite_rows), int(stats.embedding_rows))

    def _select_newest_corpus_rows(
        self, dataset: Iterable[Dict[str, Any]], dataset_source: str
    ) -> Iterable[Dict[str, Any]]:
        """Select the ``corpus_size`` most recently submitted rows by arXiv ID.

        Snapshot datasets are not ordered by submission time (the arXiv
        snapshot ships newest-``update_date`` first, with pre-2007 IDs at the
        tail), so positional slicing cannot express "newest papers". Rows are
        instead ranked by the submission chronology encoded in their arXiv
        IDs. Any shortfall is filled from rows without parseable IDs in source
        order, with a warning.

        :param Iterable[Dict[str, Any]] dataset: Loaded dataset or record stream.
        :param str dataset_source: Dataset source identifier (for logging).
        :return Iterable[Dict[str, Any]]: Selected rows (or the original
            iterable when no arXiv IDs are parseable).
        """
        limit = int(self.corpus_size)
        column_names = getattr(dataset, "column_names", None)
        select_by_index = column_names is not None and hasattr(dataset, "select")
        if select_by_index:
            if "id" not in column_names:
                logger.warning(
                    "Dataset %s has no 'id' column; capped hydration takes "
                    "the first %d rows instead of the newest.",
                    dataset_source,
                    limit,
                )
                return dataset
            selected = _newest_records_by_arxiv_id(
                (
                    {"id": raw_id, "source_index": idx}
                    for idx, raw_id in enumerate(dataset["id"])
                ),
                limit,
            )
        else:
            logger.warning(
                "Newest-first selection must scan the entire %s stream to rank "
                "submissions before hydration begins; use --no-streaming or an "
                "explicit --dataset-split slice to avoid the full pass.",
                dataset_source,
            )
            selected = _newest_records_by_arxiv_id(dataset, limit)

        chronology_keys = [
            key
            for record in selected
            if (key := _arxiv_id_chronology_key(record.get("id"))) is not None
        ]
        if not chronology_keys:
            logger.warning(
                "No parseable arXiv IDs in %s; capped hydration takes the "
                "first %d rows instead of the newest.",
                dataset_source,
                limit,
            )
            return dataset if select_by_index else selected
        else:
            if len(chronology_keys) < len(selected):
                logger.warning(
                    "Only %d rows in %s have parseable arXiv IDs; filling "
                    "the corpus cap with %d rows in source order.",
                    len(chronology_keys),
                    dataset_source,
                    len(selected) - len(chronology_keys),
                )
            logger.info(
                "Selected the %d most recently submitted rows from %s by "
                "arXiv ID chronology (submission window %04d-%02d..%04d-%02d).",
                len(chronology_keys),
                dataset_source,
                chronology_keys[0][0],
                chronology_keys[0][1],
                chronology_keys[-1][0],
                chronology_keys[-1][1],
            )
        if select_by_index:
            return dataset.select(
                sorted(int(record["source_index"]) for record in selected)
            )
        return selected

    def _resolve_dataset_split_row_count(self, dataset_source: str) -> Optional[int]:
        """Resolve dataset split row count from HuggingFace metadata when available.

        :param str dataset_source: Dataset source identifier.
        :return Optional[int]: Split row count, or ``None`` when unavailable.
        """
        if ":" in str(self.dataset_split):
            return None

        load_dataset_builder = deps._import_datasets_module().load_dataset_builder

        try:
            builder = load_dataset_builder(dataset_source)
            splits = getattr(getattr(builder, "info", None), "splits", None)
            if splits is None:
                return None
            if hasattr(splits, "get"):
                split_info = splits.get(self.dataset_split)
            elif self.dataset_split in splits:
                split_info = splits[self.dataset_split]
            else:
                split_info = None
            if split_info is None:
                return None
            num_examples = getattr(split_info, "num_examples", None)
            if num_examples is None:
                return None
            parsed = int(num_examples)
            return parsed if parsed >= 0 else None
        except Exception as exc:  # pragma: no cover - source/network dependent
            logger.warning(
                "Could not resolve split row count for %s/%s: %s",
                dataset_source,
                self.dataset_split,
                exc,
            )
            return None

    def _refresh_hydrated_full_corpus_cache(
        self, *, use_streaming: bool, cached_dataset_source: Optional[str]
    ) -> None:
        """Incrementally refresh hydrated full-corpus cache when source row count grows.

        This avoids clearing/re-encoding existing payload when a source only appends
        new records.

        :param bool use_streaming: Whether hydration mode is streaming.
        :param Optional[str] cached_dataset_source: Hydrated dataset source token.
        :return None: Mutates cache in-place when incremental refresh is required.
        """
        if self.corpus_size is not None:
            return
        if ":" in str(self.dataset_split):
            return
        source = str(cached_dataset_source or "").strip()
        if not source:
            return

        cached_rows = self._cached_payload_row_count()
        if cached_rows < 1:
            return

        upstream_rows = self._resolve_dataset_split_row_count(source)
        if upstream_rows is None:
            return
        if upstream_rows <= cached_rows:
            if upstream_rows < cached_rows:
                logger.warning(
                    "Cached embedding payload rows (%d) exceed upstream split rows (%d) "
                    "for %s/%s; marking hydration incomplete for full source revalidation.",
                    cached_rows,
                    upstream_rows,
                    source,
                    self.dataset_split,
                )
                self.embedding_cache.mark_hydrated(
                    dataset_source=source,
                    dataset_split=self.dataset_split,
                    corpus_size=self.corpus_size,
                    complete=False,
                )
            self.embedding_cache.clear_hydration_rowcount_reconciliation()
            return

        previous_reconciliation = (
            self.embedding_cache.get_hydration_rowcount_reconciliation()
        )
        if previous_reconciliation == (upstream_rows, cached_rows):
            logger.info(
                "Skipping incremental refresh for %s/%s: prior reconciliation already "
                "verified this row-count delta (cache_rows=%d, upstream_rows=%d).",
                source,
                self.dataset_split,
                cached_rows,
                upstream_rows,
            )
            return

        self.embedding_cache.mark_hydrated(
            dataset_source=source,
            dataset_split=self.dataset_split,
            corpus_size=self.corpus_size,
            complete=False,
        )
        self._ensure_int8_calibration_ranges(
            use_streaming=use_streaming,
            dataset_source=source,
        )

        delta_rows = upstream_rows - cached_rows
        logger.info(
            "Detected %d new dataset rows for %s/%s (cached=%d, upstream=%d). "
            "Running incremental cache refresh.",
            delta_rows,
            source,
            self.dataset_split,
            cached_rows,
            upstream_rows,
        )
        tail_refreshed_records = self._hydrate_exact_hydration_source_slice(
            use_streaming=use_streaming,
            source=source,
            row_limit=delta_rows,
            row_offset=cached_rows,
            progress_total=delta_rows,
            progress_label="Refreshing dataset",
            operation="Incremental refresh",
        ).hydrated_records
        updated_rows = self._cached_payload_row_count()
        reconciled_records = [0, 0]

        if updated_rows < upstream_rows:
            cached_paper_ids = self.embedding_cache.get_cached_paper_ids()
            reconciliation_passes = (
                (
                    "Head-slice reconciliation",
                    delta_rows,
                    0,
                    delta_rows,
                    "head",
                ),
                (
                    "Full-split reconciliation",
                    None,
                    None,
                    upstream_rows,
                    "full",
                ),
            )
            previous_operation = "Tail delta refresh"
            for pass_index, (
                operation,
                row_limit,
                row_offset,
                progress_total,
                progress_scope,
            ) in enumerate(reconciliation_passes):
                if updated_rows >= upstream_rows:
                    break
                remaining_rows = upstream_rows - updated_rows
                logger.warning(
                    "%s left %d unresolved rows for %s/%s "
                    "(cache_rows=%d, upstream=%d). Running %s.",
                    previous_operation,
                    remaining_rows,
                    source,
                    self.dataset_split,
                    updated_rows,
                    upstream_rows,
                    operation.lower().replace(
                        "reconciliation", "missing-ID reconciliation"
                    ),
                )
                reconciled_records[pass_index] = (
                    self._hydrate_exact_hydration_source_slice(
                        use_streaming=use_streaming,
                        source=source,
                        row_limit=row_limit,
                        row_offset=row_offset,
                        progress_total=progress_total,
                        progress_label=f"Reconciling {progress_scope}",
                        operation=operation,
                        existing_paper_ids=cached_paper_ids,
                        max_new_records=(remaining_rows if pass_index == 0 else None),
                    ).hydrated_records
                )
                updated_rows = self._cached_payload_row_count()
                previous_operation = operation

        logger.info(
            "Incremental refresh processed tail=%d head=%d full=%d rows "
            "for %s/%s (cache_rows=%d, upstream_rows=%d).",
            tail_refreshed_records,
            *reconciled_records,
            source,
            self.dataset_split,
            updated_rows,
            upstream_rows,
        )
        rows_reconciled = self._finalize_full_corpus_hydration_rows(
            source=source,
            updated_rows=updated_rows,
            upstream_rows=upstream_rows,
            mark_complete=updated_rows > 0,
        )
        if not rows_reconciled:
            logger.info(
                "Full-split reconciliation completed for %s/%s with cache_rows=%d "
                "and upstream_rows=%d. Remaining row-count delta likely reflects "
                "duplicate upstream paper IDs; this state is memoized to skip "
                "repeat full-split scans until row counts change.",
                source,
                self.dataset_split,
                updated_rows,
                upstream_rows,
            )

    def _load_dataset_for_hydration(
        self,
        use_streaming: bool,
        row_limit: Optional[int] = None,
        row_offset: Optional[int] = None,
    ) -> Tuple[str, Iterable[Dict[str, Any]]]:
        """Load the configured arXiv metadata dataset for hydration.

        :param bool use_streaming: Whether to load streaming dataset iterator.
        :param Optional[int] row_limit: Optional row cap override for dataset loading.
        :param Optional[int] row_offset: Optional row offset for delta refresh loading.
        :return Tuple[str, Iterable[Dict[str, Any]]]: Dataset source name and iterable.
        """
        load_dataset = deps._import_datasets_module().load_dataset

        parsed_row_limit: Optional[int] = None
        if row_limit is not None:
            parsed_row_limit = int(row_limit)
            if parsed_row_limit < 1:
                raise ValueError("row_limit must be at least 1 when provided")

        parsed_row_offset = 0 if row_offset is None else int(row_offset)
        if parsed_row_offset < 0:
            raise ValueError("row_offset must be at least 0 when provided")

        split_for_load = self.dataset_split
        if not use_streaming and parsed_row_offset > 0 and ":" in split_for_load:
            raise ValueError(
                "row_offset requires a non-sliced dataset_split in non-streaming mode"
            )
        if (
            not use_streaming
            and ":" not in split_for_load
            and (parsed_row_limit is not None or parsed_row_offset > 0)
        ):
            start_idx = str(parsed_row_offset) if parsed_row_offset else ""
            stop_idx = (
                ""
                if parsed_row_limit is None
                else str(parsed_row_offset + parsed_row_limit)
            )
            split_for_load = f"{split_for_load}[{start_idx}:{stop_idx}]"
        dataset = load_dataset(
            self.dataset_source,
            split=split_for_load,
            streaming=use_streaming,
            num_proc=None if use_streaming else max(1, (os.cpu_count() or 1) // 2),
        )
        if use_streaming and (parsed_row_limit is not None or parsed_row_offset > 0):
            stop_idx = (
                None
                if parsed_row_limit is None
                else parsed_row_offset + parsed_row_limit
            )
            dataset = islice(dataset, parsed_row_offset, stop_idx)
        elif (
            self.corpus_size is not None
            and parsed_row_limit is None
            and parsed_row_offset == 0
        ):
            # Snapshot row order does not track submission time.
            dataset = self._select_newest_corpus_rows(dataset, self.dataset_source)
        logger.debug(
            "Hydration dataset selected: %s (split=%s, streaming=%s, row_limit=%s, row_offset=%s).",
            self.dataset_source,
            split_for_load,
            use_streaming,
            "none" if parsed_row_limit is None else parsed_row_limit,
            parsed_row_offset,
        )
        return self.dataset_source, dataset

    def _initialize_calibration_ranges(self, records: List[Dict]) -> None:
        """Compute and persist int8 calibration ranges from metadata records.

        :param List[Dict] records: Records used for calibration embedding sample.
        :return None: Persists calibration ranges in cache.
        """
        if self.storage_precision != "int8":
            return
        if self.embedding_cache.has_calibration_ranges():
            return
        if not records:
            return

        sample_texts = [
            self._format_retrieval_document_metadata(metadata) for metadata in records
        ]
        sample_embeddings = self._encode_texts(
            sample_texts,
            batch_size=self.encode_batch_size,
            show_progress_bar=False,
        )
        ranges = np.stack(
            (sample_embeddings.min(axis=0), sample_embeddings.max(axis=0))
        ).astype(np.float32)
        self.embedding_cache.set_calibration_ranges(
            ranges=ranges,
            embedding_dim=int(sample_embeddings.shape[1]),
        )

    def _cache_metadata_batch(self, batch: List[Dict]) -> int:
        """Encode/cache a batch of metadata records.

        :param List[Dict] batch: Metadata records including ``paper_id``.
        :return int: Number of paper IDs routed into cache encoding.
        """
        if not batch:
            return 0

        metadata_map: Dict[str, Dict] = {}
        for metadata in batch:
            paper_id = str(metadata.get("paper_id", "")).strip()
            if not paper_id:
                continue
            payload = dict(metadata)
            metadata_map[paper_id] = payload

        if not metadata_map:
            return 0

        self.embedding_cache.upsert_embeddings(
            metadata_map,
            self._get_model_for_encoding(),
            batch_size=min(self.encode_batch_size, len(metadata_map)),
            show_progress=False,
            text_builder=self._format_retrieval_document_metadata,
        )
        return len(metadata_map)
