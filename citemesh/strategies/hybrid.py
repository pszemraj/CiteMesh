"""
Hybrid graph building strategy.

Combines citation relationships with semantic similarity for
comprehensive paper discovery.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from citemesh.core import EMBEDDING_STORAGE_CONFIG, HYBRID_CONFIG, Paper
from citemesh.data import DEFAULT_EMBEDDING_MODEL_NAME
from citemesh.services import get_client
from citemesh.strategies.base import (
    GraphBuilderStrategy,
    build_capped_undirected_graph,
    deterministic_sort_key,
    validate_embedding_vectors,
)
from citemesh.strategies.candidates import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    SEMANTIC_SOURCE_CHOICES,
    CandidateAcquisitionError,
    IdentityRegistry,
    fetch_candidate_source,
    merge_seed_relation,
    reconcile_paper_identity,
    register_aliases,
    require_available_candidate_source,
)
from citemesh.strategies.citation import CitationGraphBuilder
from citemesh.strategies.embedding import (
    ENCODE_BATCH_SIZE,
    EmbeddingGraphBuilder,
    EmbeddingTask,
    _check_embedding_deps,
    format_paper_for_embedding,
)
from citemesh.text_batching import l2_normalize_embeddings

if TYPE_CHECKING:
    from citemesh.services.semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)
HYBRID_DEFAULT_MAX_PAPERS = 45
HYBRID_DEFAULT_MAX_CITATIONS = 45
HYBRID_DEFAULT_MAX_REFERENCES = 12
DEFAULT_MAX_SEMANTIC = 20
HYBRID_SEMANTIC_CANDIDATE_MULTIPLIER = 3
HYBRID_SEED_RERANK_WEIGHTS = (0.62, 0.16, 0.14, 0.08)
HYBRID_SOURCE_OVERLAP_BONUS = 0.10
HYBRID_CITATION_SOURCE_BONUS = 0.02


class EmbeddingInferenceError(RuntimeError):
    """Semantic inference could not produce a complete hybrid ranking space."""


def require_complete_embeddings(
    *,
    seed_id: str,
    candidate_ids: List[str],
    embeddings: Dict[str, np.ndarray],
) -> np.ndarray:
    """Validate that hybrid reranking has one usable vector per required paper.

    :param str seed_id: Seed paper identifier.
    :param List[str] candidate_ids: Candidate identifiers admitted to reranking.
    :param Dict[str, np.ndarray] embeddings: Materialized retrieval embeddings.
    :return np.ndarray: Validated seed embedding.
    :raises EmbeddingInferenceError: If vectors are missing, malformed, non-finite,
        zero length, or dimensionally inconsistent.
    """
    vectors = validate_embedding_vectors(
        [seed_id, *candidate_ids],
        embeddings,
        context="Semantic reranking",
        vector_label="embedding",
        error_factory=EmbeddingInferenceError,
    )
    return vectors[seed_id]


class HybridGraphBuilder(GraphBuilderStrategy):
    """
    Hybrid strategy combining citations and embeddings.

    This strategy:
    1. Collects papers via citations (ground truth relationships)
    2. Enriches with semantically similar papers from corpus
    3. Uses adaptive similarity computation based on relationship type
    """

    strategy_name = "hybrid"

    def __init__(
        self,
        max_papers: int = HYBRID_DEFAULT_MAX_PAPERS,
        max_citations: int = HYBRID_DEFAULT_MAX_CITATIONS,
        max_references: int = HYBRID_DEFAULT_MAX_REFERENCES,
        fetch_references: bool = True,
        refresh_reference_cache: bool = False,
        max_semantic: Optional[int] = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        model_profile: str = "auto",
        model_revision: Optional[str] = None,
        dataset_split: str = "train",  # Full snapshot split; use corpus_size in embedding strategy to bound runtime.
        corpus_size: Optional[int] = 50000,
        truncate_dim: Optional[int] = None,
        use_streaming: bool = False,
        force_rebuild_cache: bool = False,
        force_rebuild_reason: Optional[str] = None,
        storage_precision: str = EMBEDDING_STORAGE_CONFIG.storage_precision,
        binary_prefilter: Optional[bool] = None,
        binary_rescore_multiplier: Optional[int] = None,
        calibration_sample_size: int = EMBEDDING_STORAGE_CONFIG.calibration_sample_size,
        cache_compression: str = EMBEDDING_STORAGE_CONFIG.compression,
        cache_compression_level: int = EMBEDDING_STORAGE_CONFIG.compression_level,
        encode_batch_size: int = ENCODE_BATCH_SIZE,
        enable_torch_compile: bool = False,
        device: Optional[str] = None,
        semantic_source: str = "candidates",
        candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
        client: Optional[SemanticScholarClient] = None,
    ):
        """
        Initialize hybrid graph builder.

        :param int max_papers: Maximum total papers
        :param int max_citations: Maximum citing papers from S2
        :param int max_references: Maximum referenced papers from S2
        :param bool fetch_references: Whether citation branch fetches reference lists.
        :param bool refresh_reference_cache: Whether citation/reference lookups bypass persisted cache reads.
        :param Optional[int] max_semantic: Maximum non-seed semantic papers added during
            enrichment. Values must satisfy ``0 <= max_semantic <= max_papers - 1``.
            When omitted, defaults to ``min(20, max_papers - 1)``.
        :param str model_name: Embedding model name
        :param str model_profile: Model task/runtime profile override.
        :param Optional[str] model_revision: Optional model revision token for hub-backed models.
        :param str dataset_split: ArXiv dataset split
        :param Optional[int] corpus_size: Maximum papers loaded for semantic search.
        :param Optional[int] truncate_dim: Optional embedding dimension truncation.
        :param bool use_streaming: Whether to stream the embedding corpus.
        :param bool force_rebuild_cache: Whether to clear embedding cache before semantic enrichment.
        :param Optional[str] force_rebuild_reason: Optional operator rationale logged
            when ``force_rebuild_cache`` clears embedding namespace state.
        :param str storage_precision: Persistent cache precision for semantic branch embeddings.
        :param Optional[bool] binary_prefilter: Whether semantic branch uses binary
            prefiltering. When ``None``, defaults are selected by embedding precision.
        :param Optional[int] binary_rescore_multiplier: Candidate oversampling factor
            for semantic branch search. When ``None``, defaults are selected by
            embedding precision.
        :param int calibration_sample_size: Calibration sample size for int8 storage ranges.
        :param str cache_compression: HDF5 compression filter for semantic branch cache.
        :param int cache_compression_level: HDF5 compression level for semantic branch cache.
        :param int encode_batch_size: Batch size used for semantic branch embedding encodes.
        :param bool enable_torch_compile: Whether semantic branch may use
            best-effort inner-model ``torch.compile`` optimization.
        :param Optional[str] device: Requested compute device token for the
            semantic branch (``auto``/``cuda``/``mps``/``cpu``).
        :param str semantic_source: Semantic candidate sourcing mode:
            ``candidates`` (default; S2 recommendations, no local corpus) or
            ``arxiv-corpus`` (hydrated arXiv corpus search).
        :param int candidate_pool_size: Maximum S2 candidate pool size used by
            the embedding builder in ``candidates`` mode.
        :param Optional[SemanticScholarClient] client: Optional injected S2 client.
        """
        normalized_semantic_source = str(semantic_source).strip().lower()
        if normalized_semantic_source not in SEMANTIC_SOURCE_CHOICES:
            formatted = ", ".join(SEMANTIC_SOURCE_CHOICES)
            raise ValueError(f"semantic_source must be one of: {formatted}")

        if max_semantic is None:
            resolved_max_semantic = max(0, min(DEFAULT_MAX_SEMANTIC, max_papers - 1))
        else:
            resolved_max_semantic = max_semantic

        if resolved_max_semantic < 0:
            raise ValueError("max_semantic must be non-negative")
        if resolved_max_semantic >= max_papers:
            raise ValueError(
                "max_semantic must be between 0 and max_papers - 1 "
                f"(got max_semantic={resolved_max_semantic}, max_papers={max_papers})"
            )

        super().__init__(max_papers)
        self.client = client or get_client()
        self.max_semantic = resolved_max_semantic
        self.semantic_source = normalized_semantic_source
        self._semantic_candidate_cap = 0

        if self.max_semantic > 0:
            _check_embedding_deps(
                require_corpus=normalized_semantic_source == "arxiv-corpus"
            )
            self._semantic_candidate_cap = max(
                self.max_semantic,
                min(
                    max_papers - 1,
                    self.max_semantic * HYBRID_SEMANTIC_CANDIDATE_MULTIPLIER,
                ),
            )
            self.embedding_builder = EmbeddingGraphBuilder(
                # Embedding strategy budgets include the seed node; hybrid's
                # max_semantic contract counts only added non-seed neighbors.
                max_papers=self._semantic_candidate_cap + 1,
                model_name=model_name,
                model_profile=model_profile,
                model_revision=model_revision,
                dataset_split=dataset_split,
                corpus_size=corpus_size,
                truncate_dim=truncate_dim,
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
                semantic_source=normalized_semantic_source,
                candidate_pool_size=candidate_pool_size,
                client=self.client,
            )
        else:
            self.embedding_builder = None

        citation_candidates = (
            max_papers
            if self.max_semantic <= 0
            else 1 + int(max_references) + int(max_citations)
        )

        self.citation_builder = CitationGraphBuilder(
            max_papers=citation_candidates,
            max_citations=max_citations,
            max_references=max_references,
            fetch_references=fetch_references,
            refresh_reference_cache=refresh_reference_cache,
            client=self.client,
        )

        # Track paper sources for adaptive similarity
        self.paper_sources: Dict[str, str] = {}  # paper_id -> citation|semantic|both
        self.seed_relations: Dict[str, str] = {}
        self.candidate_source_status: Dict[str, str] = {}

    def _ingest_candidate(
        self,
        aliases: IdentityRegistry,
        seed: Paper,
        candidates: Dict[str, Paper],
        candidate_sources: Dict[str, Set[str]],
        incoming: Paper,
        *,
        source: str,
        relation: str,
    ) -> Optional[str]:
        """Reconcile and tag one pre-ranking candidate.

        :param IdentityRegistry aliases: Identity registry.
        :param Paper seed: Canonical seed paper.
        :param Dict[str, Paper] candidates: Pre-ranking candidate map.
        :param Dict[str, Set[str]] candidate_sources: Candidate provenance map.
        :param Paper incoming: Newly observed candidate payload.
        :param str source: Candidate source tag.
        :param str relation: Relation-to-seed label.
        :return Optional[str]: Candidate survivor, or ``None`` for a seed match.
        """
        reconciliation = reconcile_paper_identity(aliases, seed, candidates, incoming)
        if reconciliation.seed_matched:
            for paper_id in reconciliation.collapsed_ids:
                candidate_sources.pop(paper_id, None)
                self.seed_relations.pop(paper_id, None)
                self.paper_sources.pop(paper_id, None)
            return None

        canonical_id = reconciliation.canonical_id
        if canonical_id is None:
            canonical_id = str(incoming.paper_id)
            candidates[canonical_id] = incoming
            register_aliases(aliases, canonical_id, incoming)

        for paper_id in reconciliation.collapsed_ids:
            candidate_sources.setdefault(canonical_id, set()).update(
                candidate_sources.pop(paper_id, set())
            )
            merged_relation = merge_seed_relation(
                self.seed_relations.get(canonical_id, ""),
                self.seed_relations.pop(paper_id, ""),
            )
            if merged_relation:
                self.seed_relations[canonical_id] = merged_relation
            if paper_id in self.paper_sources:
                self.paper_sources[canonical_id] = self.paper_sources.pop(paper_id)

        candidate_sources.setdefault(canonical_id, set()).add(source)
        merged_relation = merge_seed_relation(
            self.seed_relations.get(canonical_id, ""), relation
        )
        if merged_relation:
            self.seed_relations[canonical_id] = merged_relation
        return canonical_id

    def _embed_candidates(
        self,
        candidate_ids: List[str],
        candidates: Dict[str, Paper],
        embeddings_map: Dict[str, np.ndarray],
    ) -> None:
        """Embed rerank candidates for seed-relevance scoring.

        Candidate mode persists vectors through the embedding cache (its
        namespace is candidate-scoped). Corpus mode encodes in memory only, so
        S2 candidate rows never distort corpus-hydration row counts.

        :param List[str] candidate_ids: Candidate IDs needing embeddings.
        :param Dict[str, Paper] candidates: Candidate paper payloads.
        :param Dict[str, np.ndarray] embeddings_map: In-memory embedding map to update.
        :return None: Mutates ``embeddings_map`` in place.
        :raises EmbeddingInferenceError: If candidate embeddings cannot be completed.
        """
        if not candidate_ids or self.embedding_builder is None:
            return

        if self.semantic_source != "arxiv-corpus":
            subset = {
                paper_id: candidates[paper_id]
                for paper_id in candidate_ids
                if paper_id in candidates
            }
            if not subset:
                return
            try:
                encoded_map = self.embedding_builder.embed_papers(subset)
            except Exception as exc:
                raise EmbeddingInferenceError(
                    "Hybrid candidate embedding failed during semantic reranking: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            embeddings_map.update(encoded_map)
            return

        model_profile = getattr(self.embedding_builder, "model_profile", None)
        encode_texts = getattr(self.embedding_builder, "_encode_texts", None)
        if model_profile is None or not callable(
            getattr(model_profile, "format_document", None)
        ):
            raise EmbeddingInferenceError(
                "Hybrid semantic reranking has no document formatter."
            )
        if not callable(encode_texts):
            raise EmbeddingInferenceError(
                "Hybrid semantic reranking has no embedding encoder."
            )

        texts: List[str] = []
        ordered_candidate_ids: List[str] = []
        for paper_id in candidate_ids:
            paper = candidates.get(paper_id)
            if paper is None:
                continue

            document_text = format_paper_for_embedding(
                profile=model_profile,
                paper=paper,
                task=EmbeddingTask.RETRIEVAL_DOCUMENT,
            ).strip()
            texts.append(document_text)
            ordered_candidate_ids.append(paper_id)

        if not ordered_candidate_ids:
            return

        batch_size = min(
            int(getattr(self.embedding_builder, "encode_batch_size", 32)),
            len(ordered_candidate_ids),
        )
        try:
            encoded = encode_texts(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingInferenceError(
                "Hybrid in-memory candidate embedding failed during semantic "
                f"reranking: {type(exc).__name__}: {exc}"
            ) from exc

        encoded_array = np.asarray(encoded, dtype=np.float32)
        if encoded_array.ndim == 1:
            encoded_array = encoded_array.reshape(1, -1)
        if encoded_array.shape[0] != len(ordered_candidate_ids):
            raise EmbeddingInferenceError(
                "Hybrid candidate encoding returned "
                f"{int(encoded_array.shape[0])} row(s) for "
                f"{len(ordered_candidate_ids)} candidate(s)."
            )

        for idx, paper_id in enumerate(ordered_candidate_ids):
            embeddings_map[paper_id] = encoded_array[idx]

    def _ensure_candidate_embeddings(
        self, seed_paper: Paper, candidates: Dict[str, Paper]
    ) -> Optional[np.ndarray]:
        """Ensure seed/candidate embeddings are materialized for reranking.

        :param Paper seed_paper: Seed paper.
        :param Dict[str, Paper] candidates: Candidate paper pool.
        :return Optional[np.ndarray]: Seed embedding, or ``None`` only when semantic
            reranking was explicitly disabled.
        :raises EmbeddingInferenceError: If required semantic vectors cannot be
            materialized or validated.
        """
        if self.embedding_builder is None:
            return None

        embeddings_map = getattr(self.embedding_builder, "retrieval_embeddings", None)
        if not isinstance(embeddings_map, dict):
            return None

        seed_embedding = embeddings_map.get(seed_paper.paper_id)
        if seed_embedding is None:
            try:
                seed_text = format_paper_for_embedding(
                    profile=self.embedding_builder.model_profile,
                    paper=seed_paper,
                    task=EmbeddingTask.RETRIEVAL_QUERY,
                )
                seed_embedding = self.embedding_builder._encode_texts(
                    [seed_text], show_progress_bar=False
                )[0]
            except Exception as exc:
                raise EmbeddingInferenceError(
                    "Hybrid seed embedding failed during semantic reranking: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            embeddings_map[seed_paper.paper_id] = seed_embedding

        missing_ids = [
            paper_id for paper_id in candidates if paper_id not in embeddings_map
        ]
        if missing_ids:
            self._embed_candidates(
                candidate_ids=missing_ids,
                candidates=candidates,
                embeddings_map=embeddings_map,
            )

        return require_complete_embeddings(
            seed_id=seed_paper.paper_id,
            candidate_ids=list(candidates),
            embeddings=embeddings_map,
        )

    def _seed_relevance_score(
        self,
        seed_embedding: Optional[np.ndarray],
        seed_paper: Paper,
        candidate: Paper,
        source_tags: Set[str],
        max_citation_count: int,
    ) -> float:
        """Score candidate relevance to seed for hybrid adjudication.

        :param Optional[np.ndarray] seed_embedding: Seed embedding vector when available.
        :param Paper seed_paper: Seed paper.
        :param Paper candidate: Candidate paper.
        :param Set[str] source_tags: Candidate provenance tags.
        :param int max_citation_count: Max citation count in candidate pool.
        :return float: Relevance score in ``[0, 1]``.
        """
        semantic_score = 0.0
        if (
            seed_embedding is not None
            and self.embedding_builder is not None
            and isinstance(
                getattr(self.embedding_builder, "retrieval_embeddings", None), dict
            )
            and candidate.paper_id in self.embedding_builder.retrieval_embeddings
        ):
            candidate_embedding = np.asarray(
                self.embedding_builder.retrieval_embeddings[candidate.paper_id],
                dtype=np.float32,
            )
            cosine = float(
                np.clip(np.dot(seed_embedding, candidate_embedding), -1.0, 1.0)
            )
            semantic_score = 0.5 * (cosine + 1.0)

        temporal_score = self.temporal_similarity(seed_paper, candidate)
        citation_denominator = max(max_citation_count, 1)
        citation_score = float(
            np.log1p(max(candidate.citation_count, 0))
            / np.log1p(citation_denominator + 1)
        )
        biblio_score = self.bibliographic_coupling(seed_paper, candidate)

        semantic_w, temporal_w, citation_w, biblio_w = HYBRID_SEED_RERANK_WEIGHTS
        score = (
            semantic_w * semantic_score
            + temporal_w * temporal_score
            + citation_w * citation_score
            + biblio_w * biblio_score
        )
        if "citation" in source_tags and "semantic" in source_tags:
            score += HYBRID_SOURCE_OVERLAP_BONUS
        elif "citation" in source_tags:
            score += HYBRID_CITATION_SOURCE_BONUS
        return float(min(score, 1.0))

    def _rank_candidates(
        self,
        seed_paper: Paper,
        candidates: Dict[str, Paper],
        candidate_sources: Dict[str, Set[str]],
    ) -> List[str]:
        """Return candidate IDs ranked by seed-centric hybrid relevance.

        :param Paper seed_paper: Seed paper.
        :param Dict[str, Paper] candidates: Candidate paper pool.
        :param Dict[str, Set[str]] candidate_sources: Candidate provenance mapping.
        :return List[str]: Ranked candidate IDs.
        """
        if not candidates:
            return []

        seed_embedding = self._ensure_candidate_embeddings(seed_paper, candidates)
        max_citation_count = max(
            (paper.citation_count for paper in candidates.values()), default=0
        )
        scored: List[Tuple[float, str, int]] = []
        for idx, (paper_id, paper) in enumerate(candidates.items()):
            score = self._seed_relevance_score(
                seed_embedding=seed_embedding,
                seed_paper=seed_paper,
                candidate=paper,
                source_tags=candidate_sources.get(paper_id, {"citation"}),
                max_citation_count=max_citation_count,
            )
            scored.append((score, paper_id, idx))
        scored.sort(
            key=lambda item: deterministic_sort_key(
                item[0], item[1], stable_index=item[2]
            )
        )
        return [paper_id for _, paper_id, _ in scored]

    def collect_papers(self, seed_id: str, **kwargs: Any) -> Dict[str, Paper]:
        """
        Collect papers from both citation and semantic sources.

        :param str seed_id: Seed paper identifier
        :param Any kwargs: Strategy-specific options (currently unused).
        :return Dict[str, Paper]: Combined dictionary of papers
        """
        papers: Dict[str, Paper] = {}
        self.paper_sources = {}
        self.seed_relations = {}
        self.candidate_source_status = {}
        if self.embedding_builder is not None:
            self.embedding_builder.retrieval_embeddings = {}
            self.embedding_builder.embeddings = {}
        alias_map = IdentityRegistry()

        # Step 1: Collect from citations
        logger.debug("Collecting papers via citations...")
        if self.embedding_builder is not None and self.semantic_source == "candidates":
            citation_papers = self.citation_builder.collect_papers(
                seed_id,
                validate_source_availability=False,
            )
        else:
            citation_papers = self.citation_builder.collect_papers(seed_id)
        citation_source_status = getattr(
            self.citation_builder, "candidate_source_status", {}
        )
        if isinstance(citation_source_status, dict):
            self.candidate_source_status.update(citation_source_status)
        seed_paper = next(
            (paper for paper in citation_papers.values() if paper.is_seed), None
        )
        if seed_paper is None:
            raise RuntimeError(
                "Hybrid collection failed: citation branch returned no seed"
            )
        papers[seed_paper.paper_id] = seed_paper
        self.paper_sources[seed_paper.paper_id] = "citation"
        self.seed_relations[seed_paper.paper_id] = "seed"
        register_aliases(alias_map, seed_paper.paper_id, seed_paper)
        citation_seed_relations = getattr(self.citation_builder, "seed_relations", {})

        candidate_pool: Dict[str, Paper] = {}
        candidate_sources: Dict[str, Set[str]] = {}
        for paper in citation_papers.values():
            if paper.paper_id == seed_paper.paper_id or paper.is_seed:
                continue
            self._ingest_candidate(
                alias_map,
                seed_paper,
                candidate_pool,
                candidate_sources,
                paper,
                source="citation",
                relation=str(citation_seed_relations.get(paper.paper_id, "citation")),
            )

        if self.embedding_builder is None:
            for paper_id, paper in candidate_pool.items():
                if len(papers) >= self.max_papers:
                    break
                papers[paper_id] = paper
                self.paper_sources[paper_id] = "citation"
                self.seed_relations.setdefault(paper_id, "citation")
            return papers

        # Step 2: Enrich with semantic matches
        semantic_budget = min(self.max_semantic, max(0, self.max_papers - 1))
        if semantic_budget <= 0:
            ranked_ids = self._rank_candidates(
                seed_paper, candidate_pool, candidate_sources
            )
            for paper_id in ranked_ids:
                if len(papers) >= self.max_papers:
                    break
                papers[paper_id] = candidate_pool[paper_id]
                self.paper_sources[paper_id] = "citation"
                self.seed_relations.setdefault(paper_id, "citation")
            return papers

        logger.info("Enriching with up to %s semantic matches...", semantic_budget)

        from citemesh.services import SemanticScholarUnavailableError

        try:
            if self.semantic_source == "arxiv-corpus":
                semantic_papers = self.embedding_builder.collect_papers(
                    seed_id,
                    seed_paper=seed_paper,
                )
            else:
                # Candidate mode: the citation branch already covers references
                # and citations, so the semantic branch adds recommendations only.
                recommendation_result = fetch_candidate_source(
                    "recommendations",
                    lambda: self.client.get_recommended_papers(
                        seed_paper.paper_id,
                        limit=min(
                            self._semantic_candidate_cap,
                            self.embedding_builder.candidate_pool_size,
                        ),
                        raise_on_unavailable=True,
                    ),
                )
                self.candidate_source_status["recommendations"] = (
                    recommendation_result.state.value
                )
                require_available_candidate_source(
                    [
                        *self.citation_builder.candidate_source_results,
                        recommendation_result,
                    ],
                    context=f"hybrid candidate acquisition for {seed_paper.paper_id}",
                )
                self.embedding_builder._load_model()
                semantic_papers = {
                    paper.paper_id: paper for paper in recommendation_result.papers
                }
        except SemanticScholarUnavailableError:
            raise
        except CandidateAcquisitionError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Semantic enrichment failed: {exc}") from exc

        for paper in semantic_papers.values():
            if paper.is_seed:
                continue
            self._ingest_candidate(
                alias_map,
                seed_paper,
                candidate_pool,
                candidate_sources,
                paper,
                source="semantic",
                relation="semantic_only",
            )

        ranked_ids = self._rank_candidates(
            seed_paper, candidate_pool, candidate_sources
        )
        added_semantic = 0
        for paper_id in ranked_ids:
            if len(papers) >= self.max_papers:
                break
            tags = candidate_sources.get(paper_id, {"citation"})
            semantic_only = "semantic" in tags and "citation" not in tags
            if semantic_only and added_semantic >= semantic_budget:
                continue
            papers[paper_id] = candidate_pool[paper_id]
            if semantic_only:
                added_semantic += 1
                self.seed_relations[paper_id] = merge_seed_relation(
                    self.seed_relations.get(paper_id, ""),
                    "semantic_only",
                )
            self.paper_sources[paper_id] = "both" if len(tags) > 1 else next(iter(tags))
            if "citation" in tags:
                self.seed_relations.setdefault(paper_id, "citation")

        logger.info("Added %s semantic papers", added_semantic)

        return papers

    def prepare_graph_scoring(self, papers: Dict[str, Paper]) -> None:
        """Build the final symmetric vector space used by hybrid graph edges.

        :param Dict[str, Paper] papers: Final selected papers.
        :return None: Populates the embedding builder's graph-vector map.
        :raises EmbeddingInferenceError: If symmetric inference cannot complete.
        """
        if self.embedding_builder is None:
            return
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
            self.embedding_builder is not None
            and paper1.paper_id in self.embedding_builder.embeddings
            and paper2.paper_id in self.embedding_builder.embeddings
        ):
            emb1 = l2_normalize_embeddings(
                self.embedding_builder.embeddings[paper1.paper_id]
            )
            emb2 = l2_normalize_embeddings(
                self.embedding_builder.embeddings[paper2.paper_id]
            )
            embed_sim = float(np.clip(np.dot(emb1, emb2), -1.0, 1.0))

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

        # Add co-citation boost if papers are from same era
        if (
            paper1.year is not None
            and paper2.year is not None
            and abs(paper1.year - paper2.year) < 2
        ):
            similarity += HYBRID_CONFIG.co_citation_boost

        return min(similarity, 1.0)  # Cap at 1.0

    def build_graph(self, seed_id: str, **kwargs: Any) -> Tuple[nx.Graph, str]:
        """
        Build graph and enforce per-node edge limits for readability.

        :param str seed_id: Seed paper identifier
        :param Any kwargs: Strategy-specific options (currently unused).
        :return Tuple[nx.Graph, str]: Tuple of (NetworkX graph, seed_id).
        """
        graph, actual_seed_id = super().build_graph(seed_id, **kwargs)
        graph.graph["strategy"] = self._resolved_strategy_name()
        if self.embedding_builder is not None:
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
            return graph, actual_seed_id

        filtered_graph = build_capped_undirected_graph(graph, max_edges)

        logger.info(
            "Hybrid edge cap applied: %s -> %s edges",
            graph.number_of_edges(),
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
