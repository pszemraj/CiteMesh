"""Explicit, pinned reconciliation of a manually managed corpus cache."""

from __future__ import annotations

import logging
import re
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np

from citemesh.data.embedding_contract import (
    INT8_QUANTIZATION,
    CorpusSource,
    CorpusSourceData,
    EmbeddingContract,
    ExistingCacheSelection,
)
from citemesh.data.model_profiles import embedding_model_profile_key

from . import corpus_source
from .hydration import HYDRATION_FLUSH_SIZE

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from .builder import EmbeddingBuilderRuntimeOptions, EmbeddingGraphBuilder

logger = logging.getLogger(__name__)


class _UnchangedRefreshResult(TypedDict):
    """Refresh result when the pinned source already matches the cache."""

    unchanged: Literal[True]
    source: CorpusSourceData


class _UpdatedRefreshResult(TypedDict):
    """Refresh result with counts for the reconciled source."""

    unchanged: Literal[False]
    source: CorpusSourceData
    source_records_examined: int
    active_rows: int
    encoded_rows: int


RefreshResult = _UnchangedRefreshResult | _UpdatedRefreshResult


class _LazyEncoder:
    """Load a builder's model on the first cache miss during refresh."""

    def __init__(self, builder: EmbeddingGraphBuilder) -> None:
        """Wrap the builder and initialize the encoded-row counter.

        :param builder: Builder that owns the active model and cache namespace.
        """
        self.builder = builder
        self.encoded_count = 0

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_tensor: Literal[False],
        normalize_embeddings: Literal[True],
        show_progress_bar: bool,
    ) -> np.ndarray:
        """Encode one cache-miss batch with the builder's active model.

        :param sentences: Retrieval texts to encode.
        :param batch_size: Maximum number of texts in an encode batch.
        :param convert_to_tensor: Whether the encoder should return a tensor.
        :param normalize_embeddings: Whether the encoder should normalize vectors.
        :param show_progress_bar: Whether the encoder may show a progress bar.
        :return: Encoded vectors for the supplied texts.
        """
        self.builder._load_model()
        self.builder._ensure_cache_model_fingerprint()
        result = self.builder._get_model_for_encoding().encode(
            sentences,
            batch_size=batch_size,
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )
        self.encoded_count += len(sentences)
        return result


def _builder_contract(
    builder: EmbeddingGraphBuilder, *, fingerprint: str | None = None
) -> EmbeddingContract:
    """Build the persisted vector contract for a builder configuration.

    :param builder: Builder whose model, formatter, and storage settings define the contract.
    :param fingerprint: Resolved model identity to use when one is already available.
    :return: Contract represented by the cache namespace.
    :raises ValueError: If immutable model identity or embedding dimension is unavailable.
    """
    fingerprint = fingerprint or builder._resolve_model_fingerprint()
    prefix = f"hf::{builder.model_name}::"
    revision = fingerprint.removeprefix(prefix)
    if not fingerprint.startswith(prefix) or not re.fullmatch(
        r"[0-9a-fA-F]{40}", revision
    ):
        raise ValueError(
            "Cannot establish immutable model_revision for corpus certification"
        )
    if builder.truncate_dim is None:
        raise ValueError(
            "Cannot establish embedding dimension; supply truncate_dim explicitly"
        )
    return EmbeddingContract(
        model_id=builder.model_name,
        model_revision=revision,
        model_artifact_identity=fingerprint,
        profile_key=embedding_model_profile_key(builder.model_profile),
        profile_schema=builder.model_profile.schema_token,
        representation="retrieval-document-v1",
        formatter_identity=builder._document_formatter_fingerprint,
        normalization="l2-v1",
        dimension=builder.truncate_dim,
        stored_dtype=builder.storage_precision,
        quantization=INT8_QUANTIZATION if builder.storage_precision == "int8" else None,
        calibration_sample_size=builder.calibration_sample_size
        if builder.storage_precision == "int8"
        else None,
    )


def refresh_builder_corpus(
    builder: EmbeddingGraphBuilder, revision: str | None = None
) -> RefreshResult:
    """Reconcile one pinned source target, loading weights only for cache misses.

    :param builder: Builder that owns the cache and corpus selectors.
    :param revision: Optional dataset revision to reconcile.
    :return: Source identity and either an unchanged marker or refresh counts.
    """
    contract = (
        builder._cache_selection.contract
        if builder._cache_selection
        else _builder_contract(builder)
    )
    selectors = builder._namespace_selectors(
        artifact_identity=contract.model_artifact_identity
    )
    if builder._cache_selection is None:
        from citemesh.data import get_cache_dir
        from citemesh.data.embedding_cache import (
            open_existing_cache,
            select_existing_cache,
        )

        namespace = builder._embedding_cache_namespace(
            artifact_identity=contract.model_artifact_identity
        )
        namespace_id = sha256(namespace.encode()).hexdigest()[:12]
        directory = get_cache_dir(create=False) / "embeddings"
        occupied = any(
            (directory / name).exists()
            for name in (f"metadata_{namespace_id}.db", f"embeddings_{namespace_id}.h5")
        )
        if occupied:
            builder._cache_selection = select_existing_cache(
                namespace_id, require_ready=False
            )
            builder.embedding_cache = open_existing_cache(
                builder._cache_selection, require_ready=False
            )
    source = CorpusSource(
        builder.dataset_source,
        builder._cache_selection.source_config_name
        if builder._cache_selection
        else None,
        builder.dataset_split,
        "",
        "all" if builder.corpus_size is None else "newest",
        builder.corpus_size,
    )
    if builder._cache_selection is None:
        source = corpus_source.resolve_source_snapshot(source, revision or "main")
        builder._cache_selection = ExistingCacheSelection(
            namespace_id,
            namespace,
            contract,
            builder.dataset_source,
            None,
            builder.dataset_split,
        )
        builder.embedding_cache = builder._create_embedding_cache(namespace)
        builder.embedding_cache.set_model_fingerprint(contract.model_artifact_identity)
    cache = builder.embedding_cache
    with cache.hydration_operation_lock():
        state = cache.corpus_state()
        if state:
            selectors = state.get("namespace_selectors", selectors)
        pending = state.get("pending") if state else None
        if pending:
            target = CorpusSource(**pending["target_source"])
            if revision is not None and revision != target.revision:
                requested = corpus_source.resolve_source_snapshot(source, revision)
                if requested.revision != target.revision:
                    raise ValueError(
                        f"Refresh is pending at {target.revision}; resume that revision before advancing"
                    )
        else:
            target = (
                source
                if source.revision
                else corpus_source.resolve_source_snapshot(source, revision or "main")
            )
            if (
                state
                and state.get("membership_complete")
                and state.get("source") == target.to_dict()
            ):
                return {"unchanged": True, "source": target.to_dict()}
        builder.model_revision = contract.model_revision
        logger.info(
            "Refreshing corpus %s at source revision %s",
            target.repo_id,
            target.revision,
        )
        with cache.operation_session(
            kind="refresh",
            identity={"source": target.to_dict()},
            target_source=target,
            contract=contract,
            namespace_selectors=selectors,
        ) as session:
            target = session.target_source
            if (
                builder.storage_precision == "int8"
                and not cache.has_calibration_ranges()
            ):
                if cache.has_cached_payload():
                    raise ValueError(
                        "Populated int8 cache has no calibration ranges; cannot adopt it"
                    )
                sample = builder._sample_calibration_records(
                    corpus_source.iter_source_records(target),
                    progress_total=None,
                    progress_label="Calibrating source",
                    corpus_size=None,
                )
                builder._load_model()
                builder._ensure_cache_model_fingerprint()
                builder._initialize_calibration_ranges(sample)
            scan_stats: corpus_source.SourceScanStats = {"source_records_examined": 0}
            encoder = _LazyEncoder(builder)
            for records in corpus_source.iter_source_batches(
                target, HYDRATION_FLUSH_SIZE, scan_stats
            ):
                records = session.validate_records(records)
                cache.upsert_embeddings(
                    {row["paper_id"]: row for row in records},
                    encoder,
                    batch_size=builder.encode_batch_size,
                    show_progress=False,
                    text_builder=builder._format_retrieval_document_metadata,
                )
                session.record_membership(records)
            source_rows = scan_stats["source_records_examined"]
            active_rows = session.membership_count
            session.finish(
                expected_rows=active_rows,
                source_records_examined=source_rows,
            )
        builder._completed_source_revision = target.revision
        logger.info(
            "Corpus refresh complete: %s source rows, %s active papers, %s encoded at %s",
            f"{source_rows:,}",
            f"{active_rows:,}",
            f"{encoder.encoded_count:,}",
            target.revision,
        )
        return {
            "unchanged": False,
            "source": target.to_dict(),
            "source_records_examined": source_rows,
            "active_rows": active_rows,
            "encoded_rows": encoder.encoded_count,
        }


def refresh_cache_selection(
    selection: ExistingCacheSelection,
    revision: str | None = None,
    **runtime_options: Unpack[EmbeddingBuilderRuntimeOptions],
) -> RefreshResult:
    """Refresh a selected native cache, including interrupted work.

    :param selection: Resolved namespace contract and source selectors.
    :param revision: Optional dataset revision to reconcile.
    :param runtime_options: Optional arguments passed to ``EmbeddingGraphBuilder``.
    :return: Source identity and either an unchanged marker or refresh counts.
    """
    from citemesh.data.embedding_cache import open_existing_cache

    from .builder import EmbeddingGraphBuilder

    cache = open_existing_cache(selection, require_ready=False)
    state = cache.corpus_state()
    source = state.get("source") or (state.get("pending") or {}).get("target_source")
    corpus_size = source.get("limit") if source else None
    builder = EmbeddingGraphBuilder(
        cache_selection=selection, corpus_size=corpus_size, **runtime_options
    )
    builder.embedding_cache = cache
    return refresh_builder_corpus(builder, revision)


def adopt_legacy_cache(
    builder: EmbeddingGraphBuilder,
    namespace_id: str,
    *,
    cache_dir: Path | None = None,
    revision: str | None = None,
) -> RefreshResult:
    """Reconstruct a legacy cache contract before refreshing its source.

    :param builder: Builder configured with the source selectors for the cache.
    :param namespace_id: Legacy cache namespace to open.
    :param cache_dir: Cache directory containing the legacy namespace.
    :param revision: Optional dataset revision to reconcile.
    :return: Source identity and either an unchanged marker or refresh counts.
    """
    from citemesh.data.embedding_cache import constants, open_existing_cache
    from citemesh.data.embedding_cache.lifecycle import read_existing_cache_metadata

    metadata = read_existing_cache_metadata(namespace_id, cache_dir)
    fingerprint = metadata.get(constants.MODEL_FINGERPRINT_KEY)
    if not fingerprint:
        raise ValueError(
            "Cannot establish legacy model_artifact_identity: stored fingerprint is missing"
        )
    contract = _builder_contract(builder, fingerprint=fingerprint)
    builder._resolved_model_fingerprint = fingerprint
    for key, supplied in (
        (constants.HYDRATION_DATASET_SOURCE_KEY, builder.dataset_source),
        (constants.HYDRATION_SPLIT_KEY, builder.dataset_split),
    ):
        if metadata.get(key) and metadata[key] != supplied:
            raise ValueError(f"Legacy {key} differs from the supplied source selectors")
    coverage = metadata.get(constants.HYDRATION_CORPUS_SIZE_KEY)
    if coverage is not None:
        builder.corpus_size = constants._corpus_size_from_token(coverage)
    namespace = builder._embedding_cache_namespace(
        artifact_identity=contract.model_artifact_identity
    )
    selection = ExistingCacheSelection(
        namespace_id,
        namespace,
        contract,
        builder.dataset_source,
        None,
        builder.dataset_split,
    )
    cache = open_existing_cache(
        selection,
        cache_dir,
        require_ready=False,
        allow_legacy=True,
    )
    builder._cache_selection = selection
    builder.embedding_cache = cache
    return refresh_builder_corpus(builder, revision)
