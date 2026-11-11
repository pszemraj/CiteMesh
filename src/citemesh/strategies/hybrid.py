"""
Hybrid graph building strategy.

Discovery pools every reference and citer that Semantic Scholar returns for
the seed, 200 of its recommendations (``candidates`` and ``both``) and, with a
local corpus (``arxiv-corpus`` and ``both``), the seed's nearest corpus rows in
the retrieval-query and retrieval-document roles. The deduplicated pool is
ranked by reciprocal rank fusion (:mod:`citemesh.strategies.discovery_ranking`)
and the best ``max_papers - 1`` papers join the seed. Graph edges between them
combine embedding similarity with bibliographic coupling, weighted by whether
each paper came from citations, semantic search or both.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from contextlib import nullcontext
from dataclasses import replace
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from citemesh.core.models import PaperGraph
import numpy as np
from numpy.typing import NDArray

from citemesh.core import (
    EMBEDDING_CONFIG,
    EMBEDDING_STORAGE_CONFIG,
    HYBRID_CONFIG,
    Paper,
)
from citemesh.core.choices import HYBRID_SEMANTIC_SOURCE_CHOICES
from citemesh.core.paper_ids import is_local_corpus_paper_id, normalize_paper_id
from citemesh.core.text_batching import l2_normalize_embeddings
from citemesh.data import DEFAULT_EMBEDDING_MODEL_NAME
from citemesh.data.embedding_contract import ExistingCacheSelection
from citemesh.services import get_client
from citemesh.strategies.base import (
    GraphBuilderStrategy,
    GraphCollectionOptions,
    build_capped_undirected_graph,
    validate_embedding_vectors,
)
from citemesh.strategies.candidates import (
    CandidateSourceResult,
    CandidateSourceState,
    candidate_identity_keys,
    candidate_records_match,
    fetch_candidate_source,
    fetch_seed_references,
    provider_lookup_identifier,
    require_available_candidate_source,
    scope_candidate_collection,
)
from citemesh.strategies.discovery_pool import DiscoveryPool
from citemesh.strategies.discovery_ranking import (
    DiscoveryCandidate,
    RankedCandidate,
    rank_candidates,
)
from citemesh.strategies.embedding import (
    DEFAULT_DATASET_SOURCE,
    ENCODE_BATCH_SIZE,
    EmbeddingGraphBuilder,
    EmbeddingTask,
    _check_embedding_deps,
    format_paper_for_embedding,
)
from citemesh.strategies.reference_hydration import ReferenceHydrator
from citemesh.strategies.similarity import cosine_similarity_to_query

if TYPE_CHECKING:
    from citemesh.services.semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)
HYBRID_DEFAULT_MAX_PAPERS = 45
HYBRID_DEFAULT_MAX_CITATIONS = 1000
HYBRID_DEFAULT_MAX_REFERENCES = 1000
HYBRID_RECOMMENDATION_LIMIT = 200
HYBRID_CORPUS_NEIGHBOURS = 180
_REFERENCE_RELATIONS = frozenset({"referenced_by_seed", "overlap"})
_CITER_RELATIONS = frozenset({"cites_seed", "overlap"})


class EmbeddingInferenceError(RuntimeError):
    """Semantic inference could not produce a complete hybrid ranking space."""


class HybridGraphBuilder(GraphBuilderStrategy):
    """Rank a wide citation and semantic candidate pool, then score edges."""

    strategy_name = "hybrid"

    def __init__(
        self,
        max_papers: int = HYBRID_DEFAULT_MAX_PAPERS,
        max_citations: int = HYBRID_DEFAULT_MAX_CITATIONS,
        max_references: int = HYBRID_DEFAULT_MAX_REFERENCES,
        fetch_references: bool = True,
        refresh_reference_cache: bool = False,
        model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        model_profile: str = "auto",
        model_revision: str | None = None,
        dataset_split: str = "train",  # Full snapshot split; use corpus_size in embedding strategy to bound runtime.
        corpus_size: int | None = None,
        truncate_dim: int | None = None,
        use_streaming: bool = False,
        force_rebuild_cache: bool = False,
        force_rebuild_reason: str | None = None,
        storage_precision: str = EMBEDDING_STORAGE_CONFIG.storage_precision,
        binary_prefilter: bool | None = None,
        binary_rescore_multiplier: int | None = None,
        calibration_sample_size: int = EMBEDDING_STORAGE_CONFIG.calibration_sample_size,
        cache_compression: str = EMBEDDING_STORAGE_CONFIG.compression,
        cache_compression_level: int = EMBEDDING_STORAGE_CONFIG.compression_level,
        encode_batch_size: int = ENCODE_BATCH_SIZE,
        enable_torch_compile: bool = False,
        device: str | None = None,
        semantic_source: str = "candidates",
        client: SemanticScholarClient | None = None,
        min_semantic_similarity: float = EMBEDDING_CONFIG.min_semantic_similarity,
        dataset_source: str = DEFAULT_DATASET_SOURCE,
        cache_selection: ExistingCacheSelection | None = None,
    ) -> None:
        """
        Initialize hybrid graph builder.

        :param int max_papers: Maximum total papers, including the seed.
        :param int max_citations: Provider fetch limit for citing papers.
        :param int max_references: Provider fetch limit for referenced papers.
        :param bool fetch_references: Whether selected papers' reference lists are
            fetched for bibliographic coupling.
        :param bool refresh_reference_cache: Whether reference lookups bypass
            persisted cache reads.
        :param str model_name: Embedding model name
        :param str model_profile: Model task/runtime profile override.
        :param Optional[str] model_revision: Optional model revision token for hub-backed models.
        :param str dataset_split: ArXiv dataset split
        :param str dataset_source: HuggingFace dataset repository with arXiv metadata fields.
        :param Optional[int] corpus_size: Optional cap on papers loaded for semantic search
            (default ``None`` = full selected split).
        :param Optional[int] truncate_dim: Optional embedding dimension truncation.
        :param float min_semantic_similarity: Minimum semantic cosine for graph edges.
        :param bool use_streaming: Whether to stream the embedding corpus.
        :param bool force_rebuild_cache: Whether to clear embedding cache before semantic enrichment.
        :param force_rebuild_reason: Optional message recorded when
            ``force_rebuild_cache`` clears embedding namespace state.
        :param str storage_precision: Persistent cache precision for semantic branch embeddings.
        :param Optional[bool] binary_prefilter: Whether corpus search uses binary
            prefiltering. When ``None``, defaults are selected by embedding precision.
        :param Optional[int] binary_rescore_multiplier: Candidate oversampling factor
            for corpus search. When ``None``, defaults are selected by
            embedding precision.
        :param int calibration_sample_size: Calibration sample size for int8 storage ranges.
        :param str cache_compression: HDF5 compression filter for semantic branch cache.
        :param int cache_compression_level: HDF5 compression level for semantic branch cache.
        :param int encode_batch_size: Batch size used for embedding encodes.
        :param bool enable_torch_compile: Whether the embedding runtime may use
            best-effort inner-model ``torch.compile`` optimization.
        :param Optional[str] device: Requested compute device token
            (``auto``/``cuda``/``mps``/``cpu``).
        :param str semantic_source: ``candidates`` (default; references, citers
            and S2 recommendations), ``arxiv-corpus`` (references, citers and
            local corpus neighbours) or ``both`` (all of them).
        :param Optional[SemanticScholarClient] client: Optional injected S2 client.
        :param cache_selection: Existing corpus cache selected for semantic retrieval.
        :raises ValueError: If a fetch limit is negative or the source is unknown.
        """
        normalized_semantic_source = str(semantic_source).strip().lower()
        if cache_selection is not None and normalized_semantic_source != "both":
            normalized_semantic_source = "arxiv-corpus"
        if normalized_semantic_source not in HYBRID_SEMANTIC_SOURCE_CHOICES:
            formatted = ", ".join(HYBRID_SEMANTIC_SOURCE_CHOICES)
            raise ValueError(f"semantic_source must be one of: {formatted}")
        if max_citations < 0 or max_references < 0:
            raise ValueError("max_citations and max_references must be non-negative")

        super().__init__(max_papers)
        self.client = client or get_client()
        self.max_citations = int(max_citations)
        self.max_references = int(max_references)
        self.semantic_source = normalized_semantic_source
        self._uses_corpus = normalized_semantic_source != "candidates"

        _check_embedding_deps(
            require_corpus=self._uses_corpus and cache_selection is None
        )
        self.embedding_builder = EmbeddingGraphBuilder(
            max_papers=max_papers,
            model_name=model_name,
            model_profile=model_profile,
            model_revision=model_revision,
            dataset_source=dataset_source,
            dataset_split=dataset_split,
            corpus_size=corpus_size,
            truncate_dim=truncate_dim,
            min_semantic_similarity=min_semantic_similarity,
            use_streaming=use_streaming,
            force_rebuild_cache=force_rebuild_cache,
            force_rebuild_reason=force_rebuild_reason,
            storage_precision=storage_precision,
            binary_prefilter=binary_prefilter,
            binary_rescore_multiplier=binary_rescore_multiplier,
            calibration_sample_size=calibration_sample_size,
            cache_compression=cache_compression,
            cache_compression_level=cache_compression_level,
            encode_batch_size=encode_batch_size,
            enable_torch_compile=enable_torch_compile,
            device=device,
            semantic_source="arxiv-corpus" if self._uses_corpus else "candidates",
            client=self.client,
            cache_selection=cache_selection,
        )
        # Fetches the selected papers' reference lists for bibliographic coupling.
        self.reference_hydrator = ReferenceHydrator(
            self.client,
            fetch_references=fetch_references,
            refresh_reference_cache=refresh_reference_cache,
        )

        self.paper_sources: dict[str, str] = {}  # paper_id -> citation|semantic|both
        self.seed_relations: dict[str, str] = {}
        self.candidate_source_status: dict[str, str] = {}
        self.discovery_pool: DiscoveryPool | None = None
        self.discovery_ranking: list[RankedCandidate] = []

    @scope_candidate_collection
    def collect_papers(
        self, seed_id: str, **kwargs: Unpack[GraphCollectionOptions]
    ) -> dict[str, Paper]:
        """Collect papers while retaining any corpus lifecycle operation lock.

        :param str seed_id: Seed paper identifier.
        :param kwargs: Extra options are ignored.
        :return Dict[str, Paper]: The seed and the selected papers.
        """
        operation_lock = (
            self.embedding_builder._corpus_operation_lock()
            if self._uses_corpus
            else nullcontext()
        )
        with operation_lock:
            return self._collect_papers_under_operation_lock(seed_id)

    def _collect_papers_under_operation_lock(self, seed_id: str) -> dict[str, Paper]:
        """Pool, rank and select candidates, then enrich the selected papers.

        :param str seed_id: Seed paper identifier.
        :return Dict[str, Paper]: The seed and the selected papers.
        """
        embedding = self.embedding_builder
        self.paper_sources = {}
        self.seed_relations = {}
        self.candidate_source_status = {}
        self.discovery_pool = None
        self.discovery_ranking = []
        embedding.retrieval_embeddings = {}
        embedding.embeddings = {}
        self.reference_hydrator.reset()
        timings: dict[str, float] = {}

        started = perf_counter()
        seed, corpus_seed = self._resolve_seed(seed_id)
        timings["prepare"] = perf_counter() - started
        self.seed_relations[seed.paper_id] = "seed"
        self.paper_sources[seed.paper_id] = (
            "semantic"
            if corpus_seed and provider_lookup_identifier(seed.paper_id, seed) is None
            else "citation"
        )
        pool = DiscoveryPool(seed, self.seed_relations)
        self.discovery_pool = pool

        started = perf_counter()
        logger.info("Collecting related papers...")
        results = self._fetch_provider_sources(seed, seed_id)
        timings["provider"] = perf_counter() - started
        if not self._uses_corpus:
            # Without a corpus search to wait for, a total outage fails here,
            # before the model loads.
            self._require_available_sources(results, seed)

        started = perf_counter()
        provider_papers = [paper for result in results for paper in result.papers]
        corpus_rows = self._corpus_rows(provider_papers) if self._uses_corpus else {}
        for result in results:
            self._ingest(pool, result, corpus_rows)
        timings["corpus"] = perf_counter() - started

        started = perf_counter()
        embedding._load_model()
        timings["model"] = perf_counter() - started
        started = perf_counter()
        try:
            seed_query = embedding._encode_seed_embedding(seed)
        except Exception as exc:
            raise EmbeddingInferenceError(
                f"Hybrid seed embedding failed: {type(exc).__name__}: {exc}"
            ) from exc
        seed_row = self._seed_corpus_row(seed, corpus_seed)
        seed_vectors = (
            embedding.cached_corpus_vectors([seed_row]) if seed_row is not None else {}
        )
        seed_document = seed_vectors.get(seed_row or "")
        if seed_document is None:
            seed_document = self._embed_documents({seed.paper_id: seed})[seed.paper_id]
        timings["encoding"] = perf_counter() - started

        if self._uses_corpus:
            started = perf_counter()
            logger.info("Searching the arXiv corpus...")
            results.append(self._search_corpus(pool, seed_query, seed_document))
            timings["corpus"] += perf_counter() - started
            self._require_available_sources(results, seed)

        folded = pool.reconcile_publication_versions()
        if folded:
            logger.debug("Folded %d publication records into corpus preprints.", folded)

        started = perf_counter()
        vectors = (
            embedding.cached_corpus_vectors(
                [
                    paper_id
                    for paper_id, paper in pool.papers.items()
                    if paper.is_local_corpus
                ]
            )
            if self._uses_corpus
            else {}
        )
        timings["corpus"] += perf_counter() - started
        started = perf_counter()
        vectors.update(self._embed_pool_documents(pool, set(vectors)))
        timings["encoding"] += perf_counter() - started

        started = perf_counter()
        ranked = self._rank_pool(pool, vectors, seed_document, seed_query)
        timings["ranking"] = perf_counter() - started
        self.discovery_ranking = ranked
        selected = [item.paper_id for item in ranked[: self.max_papers - 1]]
        logger.info(
            "Ranked %d candidate%s; keeping %d.",
            len(ranked),
            "" if len(ranked) == 1 else "s",
            len(selected),
        )

        papers: dict[str, Paper] = {seed.paper_id: seed}
        for paper_id in selected:
            papers[paper_id] = pool.papers[paper_id]
            tags = pool.sources[paper_id]
            self.paper_sources[paper_id] = "both" if len(tags) > 1 else next(iter(tags))

        started = perf_counter()
        self._enrich_selected(papers, pool, corpus_seed, results)
        timings["enrichment"] = perf_counter() - started
        logger.debug(
            "Hybrid collection timings (s): %s.",
            ", ".join(f"{phase}={seconds:.2f}" for phase, seconds in timings.items()),
        )
        return papers

    def _require_available_sources(
        self, results: Sequence[CandidateSourceResult], seed: Paper
    ) -> None:
        """Record every source outcome and fail when no source could be read.

        :param Sequence[CandidateSourceResult] results: Every attempted source.
        :param Paper seed: Resolved seed.
        :return None: Updates ``candidate_source_status``.
        :raises CandidateAcquisitionError: If every source was unavailable.
        """
        self.candidate_source_status = {
            result.source: result.state.value for result in results
        }
        acquisition = {
            "both": "combined",
            "arxiv-corpus": "corpus",
            "candidates": "candidate",
        }[self.semantic_source]
        require_available_candidate_source(
            results,
            context=f"hybrid {acquisition} acquisition for {seed.paper_id}",
        )

    def _resolve_seed(self, seed_id: str) -> tuple[Paper, bool]:
        """Resolve the seed from the prepared corpus, else from Semantic Scholar.

        :param str seed_id: Seed paper identifier.
        :return tuple[Paper, bool]: Seed carrying the build seed role, and whether
            it came from the corpus.
        :raises RuntimeError: If corpus preparation fails.
        :raises ValueError: If the seed is unknown, or a local corpus ID is used
            without a corpus.
        """
        cached: Paper | None = None
        if self._uses_corpus:
            try:
                self.embedding_builder._prepare_corpus_for_build()
                cached = self.embedding_builder.resolve_cached_corpus_seed(
                    seed_id, _corpus_prepared=True
                )
            except Exception as exc:
                raise RuntimeError(f"Semantic enrichment failed: {exc}") from exc
        elif is_local_corpus_paper_id(seed_id):
            raise ValueError(
                f"Local corpus seed ID '{seed_id}' requires a corpus. Re-run the "
                "hybrid build with --semantic-source arxiv-corpus or both and the "
                "same embedding settings used for local search."
            )
        seed = cached or self.client.get_paper(seed_id, raise_on_unavailable=True)
        if seed is None:
            raise ValueError(
                f"Seed paper not found: {seed_id} (Semantic Scholar does not "
                "know this identifier; check the DOI/arXiv/S2 ID)."
            )
        # Seed role belongs to this build, not the client-owned metadata record.
        return replace(seed, is_seed=True), cached is not None

    def _fetch_provider_sources(
        self, seed: Paper, seed_id: str
    ) -> list[CandidateSourceResult]:
        """Fetch references, citers and, outside corpus-only mode, recommendations.

        :param Paper seed: Resolved seed.
        :param str seed_id: Original input, retaining an arXiv version.
        :return list[CandidateSourceResult]: One result per attempted source.
        """
        provider_id = provider_lookup_identifier(seed.paper_id, seed)
        results: list[CandidateSourceResult] = []
        if self.max_references > 0:
            corpus = self.embedding_builder if self._uses_corpus else None
            results.extend(
                fetch_seed_references(
                    self.client,
                    seed,
                    self.max_references,
                    seed_identifier=seed_id,
                    local_lookup=corpus.cached_reference_metadata if corpus else None,
                    local_bibliography_lookup=(
                        corpus.cached_bibliography_metadata if corpus else None
                    ),
                )
            )
        if self.max_citations > 0 and provider_id is not None:
            citer_lookup_id = provider_id
            results.append(
                fetch_candidate_source(
                    "citations",
                    lambda: self.client.get_paper_citations(
                        citer_lookup_id,
                        limit=self.max_citations,
                        raise_on_unavailable=True,
                    ),
                )
            )
        if self.semantic_source != "arxiv-corpus":
            results.append(self._fetch_recommendations(provider_id))
        logger.debug(
            "Provider candidates: %s.",
            ", ".join(f"{result.source}={len(result.papers)}" for result in results),
        )
        return results

    def _fetch_recommendations(self, provider_id: str | None) -> CandidateSourceResult:
        """Fetch the seed's ``all-cs`` recommendations.

        :param Optional[str] provider_id: Provider-safe seed identifier.
        :return CandidateSourceResult: Recommendations in provider order.
        """
        if provider_id is None:
            return CandidateSourceResult(
                "recommendations",
                CandidateSourceState.UNAVAILABLE,
                error="The corpus seed has no Semantic Scholar lookup identifier.",
            )
        return fetch_candidate_source(
            "recommendations",
            lambda: self.client.get_recommended_papers(
                provider_id,
                limit=HYBRID_RECOMMENDATION_LIMIT,
                raise_on_unavailable=True,
                pool="all-cs",
            ),
        )

    def _corpus_rows(self, papers: Sequence[Paper]) -> dict[str, Paper]:
        """Find the corpus row each provider record identifies.

        A row is used only when the record's arXiv and DOI identifiers lead to
        exactly one row and that row does not contradict the record.

        :param Sequence[Paper] papers: Provider records.
        :return dict[str, Paper]: Corpus rows keyed by provider record ID.
        """
        lookups: dict[str, list[str]] = {}
        for paper in papers:
            if paper.is_local_corpus or paper.paper_id in lookups:
                continue
            identifiers = [
                normalize_paper_id(identifier).lower()
                for _namespace, identifier in candidate_identity_keys(paper)
            ]
            if identifiers:
                lookups[paper.paper_id] = identifiers
        if not lookups:
            return {}
        found = self.embedding_builder.cached_reference_metadata(
            sorted({identifier for ids in lookups.values() for identifier in ids})
        )
        records = {paper.paper_id: paper for paper in papers}
        rows: dict[str, Paper] = {}
        for paper_id, identifiers in lookups.items():
            matches = {
                found[identifier].paper_id: found[identifier]
                for identifier in identifiers
                if identifier in found
            }
            if len(matches) == 1:
                row = next(iter(matches.values()))
                if candidate_records_match(row, records[paper_id]):
                    rows[paper_id] = row
        logger.debug(
            "Matched %d of %d provider records to corpus rows.",
            len(rows),
            len(lookups),
        )
        return rows

    def _seed_corpus_row(self, seed: Paper, corpus_seed: bool) -> str | None:
        """Return the corpus row holding the seed's stored document vector.

        :param Paper seed: Resolved seed.
        :param bool corpus_seed: Whether the seed was read from the corpus.
        :return Optional[str]: Corpus row ID, or ``None`` without one.
        """
        if not self._uses_corpus:
            return None
        if corpus_seed:
            return seed.paper_id
        row = self._corpus_rows([seed]).get(seed.paper_id)
        return row.paper_id if row is not None else None

    @staticmethod
    def _ingest(
        pool: DiscoveryPool,
        result: CandidateSourceResult,
        corpus_rows: dict[str, Paper],
    ) -> None:
        """Add one provider list to the pool in provider order.

        A record's corpus row joins first, so the record merges into it and the
        member keeps the corpus identity and stored vector.

        :param DiscoveryPool pool: Discovery pool.
        :param CandidateSourceResult result: Provider list.
        :param dict[str, Paper] corpus_rows: Corpus rows by provider record ID.
        :return None: Updates the pool in place.
        """
        source, relation = {
            "references": ("citation", "referenced_by_seed"),
            "arxiv_references": ("citation", "referenced_by_seed"),
            "citations": ("citation", "cites_seed"),
            "recommendations": ("semantic", "semantic_only"),
        }[result.source]
        for rank, paper in enumerate(result.papers, start=1):
            row = corpus_rows.get(paper.paper_id)
            if row is not None:
                pool.add(row, source=source, relation=relation)
            member_id = pool.add(paper, source=source, relation=relation)
            if member_id is not None and result.source == "recommendations":
                pool.record_recommendation(member_id, rank)

    def _search_corpus(
        self,
        pool: DiscoveryPool,
        seed_query: NDArray[np.float32],
        seed_document: NDArray[np.float32],
    ) -> CandidateSourceResult:
        """Add the seed's nearest corpus rows in both retrieval roles.

        :param DiscoveryPool pool: Discovery pool.
        :param NDArray[np.float32] seed_query: Seed in the retrieval-query role.
        :param NDArray[np.float32] seed_document: Seed in the retrieval-document role.
        :return CandidateSourceResult: Corpus outcome, empty without neighbours.
        :raises RuntimeError: If the corpus search fails.
        """
        added = 0
        for vector in (seed_query, seed_document):
            try:
                neighbours = self.embedding_builder.search_prepared_corpus(
                    vector, HYBRID_CORPUS_NEIGHBOURS
                )
            except Exception as exc:
                raise RuntimeError(f"Semantic enrichment failed: {exc}") from exc
            for paper in neighbours:
                if (
                    pool.add(paper, source="semantic", relation="semantic_only")
                    is not None
                ):
                    added += 1
        logger.debug("Corpus neighbours: %d across both seed roles.", added)
        return CandidateSourceResult(
            "arxiv_corpus",
            CandidateSourceState.COMPLETE if added else CandidateSourceState.EMPTY,
        )

    def _embed_documents(
        self, papers: dict[str, Paper]
    ) -> dict[str, NDArray[np.float32]]:
        """Return retrieval-document vectors for papers without a corpus vector.

        Candidates mode reads and fills the candidate embedding cache, so a
        rebuild of the same seed encodes no candidate again. A corpus mode's cache
        is the corpus itself, which must not take provider rows, so those modes
        encode in memory, with the same document text.

        :param Dict[str, Paper] papers: Papers keyed by member ID.
        :return dict[str, NDArray[np.float32]]: One vector per paper.
        :raises EmbeddingInferenceError: If encoding or the cache fails.
        """
        if not papers:
            return {}
        embedding = self.embedding_builder
        try:
            if not self._uses_corpus:
                return embedding.embed_papers(papers)
            texts = [
                format_paper_for_embedding(
                    profile=embedding.model_profile,
                    paper=paper,
                    task=EmbeddingTask.RETRIEVAL_DOCUMENT,
                )
                for paper in papers.values()
            ]
            encoded = embedding._encode_texts(
                texts,
                batch_size=min(embedding.encode_batch_size, len(texts)),
                show_progress_bar=False,
            )
            return dict(zip(papers, encoded, strict=True))
        except Exception as exc:
            raise EmbeddingInferenceError(
                f"Hybrid candidate embedding failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _embed_pool_documents(
        self, pool: DiscoveryPool, stored: Collection[str]
    ) -> dict[str, NDArray[np.float32]]:
        """Embed every member without a stored corpus vector.

        :param DiscoveryPool pool: Discovery pool.
        :param Collection[str] stored: Members that have stored corpus vectors.
        :return dict[str, NDArray[np.float32]]: Vectors keyed by member ID.
        """
        missing = sorted(paper_id for paper_id in pool.papers if paper_id not in stored)
        if missing:
            logger.debug("Embedding %d candidate documents.", len(missing))
        return self._embed_documents(
            {paper_id: pool.papers[paper_id] for paper_id in missing}
        )

    @staticmethod
    def _rank_pool(
        pool: DiscoveryPool,
        vectors: dict[str, NDArray[np.float32]],
        seed_document: NDArray[np.float32],
        seed_query: NDArray[np.float32],
    ) -> list[RankedCandidate]:
        """Rank every pool member by reciprocal rank fusion.

        :param DiscoveryPool pool: Discovery pool.
        :param dict[str, NDArray[np.float32]] vectors: Member document vectors.
        :param NDArray[np.float32] seed_document: Seed document vector.
        :param NDArray[np.float32] seed_query: Seed query-role vector.
        :return list[RankedCandidate]: Every member, best first.
        :raises EmbeddingInferenceError: If a vector is missing, malformed,
            non-finite, zero, or of another dimension than the seed's.
        """
        seed_id = pool.seed.paper_id
        # Members carry provider or corpus IDs, which never contain spaces, so
        # this key cannot collide with one.
        query_key = f"{seed_id} (query role)"
        # The one validity gate for ranking inputs; the ranker trusts them.
        validated = validate_embedding_vectors(
            [seed_id, query_key, *pool.papers],
            {**vectors, seed_id: seed_document, query_key: seed_query},
            context="Hybrid ranking",
            vector_label="embedding",
            error_factory=EmbeddingInferenceError,
        )
        candidates = []
        for paper_id in pool.papers:
            relation = pool.seed_relations.get(paper_id, "")
            candidates.append(
                DiscoveryCandidate(
                    paper_id=paper_id,
                    vector=validated[paper_id],
                    is_reference=relation in _REFERENCE_RELATIONS,
                    cites_seed=relation in _CITER_RELATIONS,
                    recommendation_rank=pool.recommendation_ranks.get(paper_id),
                )
            )
        ranked = rank_candidates(
            candidates,
            seed_document=validated[seed_id],
            seed_query=validated[query_key],
        )
        for item in ranked[:10]:
            logger.debug(
                "Rank %d %s: score=%.5f lists=%s cos_dd=%.4f.",
                item.rank,
                item.paper_id,
                item.score,
                dict(item.list_ranks),
                item.features.cos_dd,
            )
        return ranked

    def _enrich_selected(
        self,
        papers: dict[str, Paper],
        pool: DiscoveryPool,
        corpus_seed: bool,
        results: Sequence[CandidateSourceResult],
    ) -> None:
        """Fill selected papers' provider metadata and citation reference lists.

        :param dict[str, Paper] papers: The seed and the selected papers.
        :param DiscoveryPool pool: Discovery pool, for provider evidence.
        :param bool corpus_seed: Whether the seed was read from the corpus.
        :param Sequence[CandidateSourceResult] results: Attempted sources.
        :return None: Updates papers in place.
        """
        seed = pool.seed
        embedding = self.embedding_builder
        if corpus_seed:
            embedding.enrich_cached_corpus_seed(seed)
        # Members that absorbed a provider record already carry its metadata.
        provider_backed = set(pool.provider_aliases.values())
        corpus_only = {
            paper_id: paper
            for paper_id, paper in papers.items()
            if paper.is_local_corpus
            and not paper.is_seed
            and paper_id not in provider_backed
        }
        if corpus_only:
            embedding._enrich_corpus_papers(corpus_only)
        if any(
            result.source == "references" and result.state is CandidateSourceState.EMPTY
            for result in results
        ):
            self.reference_hydrator.withhold_seed_references(seed)
        # Reference lists, the coupling input, are fetched for citation-sourced papers.
        self.reference_hydrator.hydrate(
            {
                paper_id: paper
                for paper_id, paper in papers.items()
                if "citation" in pool.sources.get(paper_id, ())
            },
            seed,
        )

    def prepare_graph_scoring(self, papers: dict[str, Paper]) -> None:
        """Build the final symmetric vector space used by hybrid graph edges.

        :param Dict[str, Paper] papers: Final selected papers.
        :return None: Populates the embedding builder's graph-vector map.
        :raises EmbeddingInferenceError: If symmetric inference cannot complete.
        """
        try:
            self.embedding_builder.materialize_graph_embeddings(papers)
        except Exception as exc:
            raise EmbeddingInferenceError(
                f"Hybrid graph-similarity embedding failed: {type(exc).__name__}: {exc}"
            ) from exc

    def compute_similarity(self, paper1: Paper, paper2: Paper) -> float:
        """
        Compute similarity using adaptive weights.

        :param Paper paper1: First paper
        :param Paper paper2: Second paper
        :return float: Similarity score (0.0 to 1.0)
        """
        source1 = self.paper_sources.get(paper1.paper_id, "citation")
        source2 = self.paper_sources.get(paper2.paper_id, "citation")

        # Get component similarities
        temporal_sim = self.temporal_similarity(paper1, paper2)
        citation_sim = self.citation_similarity(paper1, paper2)
        biblio_coupling = self.bibliographic_coupling(paper1, paper2)

        # Compute embedding similarity if available
        embed_sim = 0.0
        if (
            paper1.paper_id in self.embedding_builder.embeddings
            and paper2.paper_id in self.embedding_builder.embeddings
        ):
            emb1 = l2_normalize_embeddings(
                self.embedding_builder.embeddings[paper1.paper_id]
            )
            embed_sim = cosine_similarity_to_query(
                emb1, self.embedding_builder.embeddings[paper2.paper_id]
            )

        if (
            embed_sim < self.embedding_builder.min_semantic_similarity
            and biblio_coupling <= 0.0
        ):
            return 0.0

        source1_has_semantic = source1 in {"semantic", "both"}
        source2_has_semantic = source2 in {"semantic", "both"}
        source1_has_citation = source1 in {"citation", "both"}
        source2_has_citation = source2 in {"citation", "both"}

        # Adaptive weighting by relationship provenance.
        if (
            source1_has_semantic
            and source2_has_semantic
            and not (source1_has_citation and source2_has_citation)
        ):
            # Both semantic: emphasize embeddings.
            weights = HYBRID_CONFIG.semantic_semantic_weights
        elif (
            source1_has_citation
            and source2_has_citation
            and not (source1_has_semantic and source2_has_semantic)
        ):
            # Both citation: emphasize bibliographic coupling.
            weights = HYBRID_CONFIG.citation_citation_weights
        else:
            # Mixed: balanced approach.
            weights = HYBRID_CONFIG.mixed_weights

        similarity = (
            weights[0] * embed_sim
            + weights[1] * temporal_sim
            + weights[2] * citation_sim
            + weights[3] * biblio_coupling
        )

        return min(similarity, 1.0)  # Cap at 1.0

    def build_graph(
        self, seed_id: str, **kwargs: Unpack[GraphCollectionOptions]
    ) -> tuple[PaperGraph, str]:
        """
        Build graph and enforce per-node edge limits for readability.

        :param str seed_id: Seed paper identifier
        :param kwargs: Extra options are ignored by collection.
        :return tuple[PaperGraph, str]: Built graph and canonical seed ID.
        """
        graph, actual_seed_id = super().build_graph(seed_id, **kwargs)
        graph.graph["strategy"] = self._resolved_strategy_name()
        graph.graph["embedding_runtime"] = (
            self.embedding_builder._embedding_runtime_metadata()
        )
        graph.graph["paper_sources"] = {
            str(paper_id): str(source)
            for paper_id, source in sorted(
                self.paper_sources.items(), key=lambda item: item[0]
            )
        }
        graph.graph["seed_relations"] = {
            str(paper_id): str(relation)
            for paper_id, relation in sorted(
                self.seed_relations.items(), key=lambda item: item[0]
            )
            if str(paper_id) in graph.nodes
        }
        graph.graph["candidate_source_status"] = dict(
            sorted(self.candidate_source_status.items())
        )

        max_edges = HYBRID_CONFIG.max_edges_per_node
        if not max_edges or max_edges <= 0:
            logger.debug(
                "Graph complete: %s nodes, %s edges",
                graph.number_of_nodes(),
                graph.number_of_edges(),
            )
            return graph, actual_seed_id

        filtered_graph = build_capped_undirected_graph(
            graph, max_edges, seed_id=actual_seed_id
        )

        logger.debug(
            "Hybrid edge cap applied: %s -> %s edges",
            graph.number_of_edges(),
            filtered_graph.number_of_edges(),
        )
        logger.debug(
            "Graph complete: %s nodes, %s edges",
            filtered_graph.number_of_nodes(),
            filtered_graph.number_of_edges(),
        )
        return filtered_graph, actual_seed_id

    def should_create_edge(
        self, paper1: Paper, paper2: Paper, similarity: float
    ) -> bool:
        """
        Create edges with per-node limits.

        :param Paper paper1: First paper
        :param Paper paper2: Second paper
        :param float similarity: Computed similarity
        :return bool: True if edge should be created
        """
        # Basic threshold
        if similarity < 0.2:
            return False

        # Seed paper: always connect if above threshold
        if paper1.is_seed or paper2.is_seed:
            return similarity > 0.4

        # Higher threshold for non-seed
        return similarity > 0.5
