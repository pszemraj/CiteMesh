"""
Hybrid graph building strategy.

Combines citation relationships with semantic similarity for
comprehensive paper discovery.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from citemesh.core import EMBEDDING_STORAGE_CONFIG, HYBRID_CONFIG, Paper
from citemesh.data import DEFAULT_EMBEDDING_MODEL_NAME
from citemesh.data.model_profiles import compose_title_abstract_text
from citemesh.paper_ids import normalize_paper_id
from citemesh.services import get_client
from citemesh.strategies.base import (
    GraphBuilderStrategy,
    deterministic_sort_key,
    select_capped_undirected_edges,
)
from citemesh.strategies.citation import CitationGraphBuilder
from citemesh.strategies.embedding import (
    ENCODE_BATCH_SIZE,
    EmbeddingGraphBuilder,
    _check_embedding_deps,
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
        :param Optional[SemanticScholarClient] client: Optional injected S2 client.
        """
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
        self._semantic_candidate_cap = 0

        if self.max_semantic > 0:
            _check_embedding_deps()
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

    @staticmethod
    def _normalize_identity_text(raw_text: str) -> str:
        """Normalize free-form text for deterministic paper identity matching.

        :param str raw_text: Raw user/content text.
        :return str: Lowercased alphanumeric text with compact spacing.
        """
        compact = re.sub(r"[^0-9a-z]+", " ", str(raw_text).strip().lower())
        return " ".join(compact.split())

    @classmethod
    def _paper_identity_aliases(cls, paper: Paper) -> List[str]:
        """Return deterministic alias keys used to deduplicate equivalent papers.

        :param Paper paper: Paper candidate to alias.
        :return List[str]: Stable sorted alias keys.
        """
        aliases: Set[str] = set()
        raw_id = str(paper.paper_id).strip()
        if raw_id:
            aliases.add(f"id:{raw_id.lower()}")
            try:
                aliases.add(f"id:{normalize_paper_id(raw_id).lower()}")
            except ValueError:
                pass

        normalized_title = cls._normalize_identity_text(paper.title or "")
        if normalized_title:
            year_token = (
                str(int(paper.year))
                if isinstance(paper.year, int) and paper.year > 0
                else "n.d."
            )
            aliases.add(f"meta:{normalized_title}|{year_token}")
            normalized_abstract = cls._normalize_identity_text(paper.abstract or "")
            if normalized_abstract:
                aliases.add(f"meta:{normalized_title}|abs:{normalized_abstract[:256]}")
            author_tokens = [
                cls._normalize_identity_text(author.name)
                for author in paper.authors[:3]
                if getattr(author, "name", None)
            ]
            compact_authors = "|".join(token for token in author_tokens if token)
            if compact_authors:
                aliases.add(f"meta:{normalized_title}|{year_token}|{compact_authors}")

        return sorted(aliases)

    def _resolve_alias(self, aliases: Dict[str, str], paper: Paper) -> Optional[str]:
        """Resolve an existing canonical paper ID from alias map.

        :param Dict[str, str] aliases: Alias-to-canonical map.
        :param Paper paper: Incoming paper payload.
        :return Optional[str]: Canonical paper ID when already known.
        """
        for alias in self._paper_identity_aliases(paper):
            canonical_id = aliases.get(alias)
            if canonical_id is not None:
                return canonical_id
        return None

    def _register_aliases(
        self, aliases: Dict[str, str], canonical_id: str, paper: Paper
    ) -> None:
        """Register identity aliases for a canonical paper ID.

        :param Dict[str, str] aliases: Alias-to-canonical map to mutate.
        :param str canonical_id: Canonical paper identifier.
        :param Paper paper: Paper payload providing alias candidates.
        :return None: Alias map is mutated in place.
        """
        for alias in self._paper_identity_aliases(paper):
            aliases.setdefault(alias, canonical_id)

    @staticmethod
    def _merge_seed_relation(existing: str, incoming: str) -> str:
        """Merge two seed-relation labels conservatively.

        :param str existing: Existing relation label.
        :param str incoming: Incoming relation label.
        :return str: Merged relation label.
        """
        normalized_existing = str(existing or "").strip().lower()
        normalized_incoming = str(incoming or "").strip().lower()
        if not normalized_existing:
            return normalized_incoming
        if (
            not normalized_incoming
            or normalized_existing == normalized_incoming
            or normalized_existing == "seed"
        ):
            return normalized_existing
        if normalized_existing == "overlap" or normalized_incoming == "overlap":
            return "overlap"
        if {
            normalized_existing,
            normalized_incoming,
        } == {"referenced_by_seed", "cites_seed"}:
            return "overlap"
        if normalized_existing == "semantic_only":
            return normalized_incoming
        if normalized_incoming == "semantic_only":
            return normalized_existing
        return normalized_existing

    def _merge_paper_metadata(self, preferred: Paper, incoming: Paper) -> Paper:
        """Merge supplemental metadata from an alternate source into ``preferred``.

        :param Paper preferred: Canonical paper record to retain.
        :param Paper incoming: Supplemental paper record to merge.
        :return Paper: ``preferred`` with missing metadata hydrated.
        """
        if not preferred.abstract and incoming.abstract:
            preferred.abstract = incoming.abstract
        if preferred.year is None and incoming.year is not None:
            preferred.year = incoming.year
        if (not preferred.authors) and incoming.authors:
            preferred.authors = incoming.authors
        if preferred.citation_count <= 0 and incoming.citation_count > 0:
            preferred.citation_count = incoming.citation_count
        if (not preferred.venue) and incoming.venue:
            preferred.venue = incoming.venue
        if (not preferred.arxiv_id) and incoming.arxiv_id:
            preferred.arxiv_id = incoming.arxiv_id
        if (not preferred.doi) and incoming.doi:
            preferred.doi = incoming.doi
        if (not preferred.categories) and incoming.categories:
            preferred.categories = incoming.categories
        if (not preferred.references) and incoming.references:
            preferred.references = incoming.references
        return preferred

    def _paper_embedding_metadata(self, paper: Paper) -> Dict[str, object]:
        """Build embedding-cache metadata payload for a paper.

        :param Paper paper: Paper to normalize.
        :return Dict[str, object]: Metadata payload accepted by embedding cache.
        """
        return {
            "title": paper.title or "",
            "abstract": paper.abstract or "",
            "year": paper.year,
            "authors": [author.name for author in paper.authors],
            "venue": paper.venue or "",
            "arxiv_id": paper.arxiv_id or "",
            "doi": paper.doi or "",
            "categories": list(paper.categories or []),
        }

    def _seed_query_text(self, seed_paper: Paper) -> str:
        """Build query text used to embed the hybrid seed paper.

        :param Paper seed_paper: Seed paper record.
        :return str: Query text payload used for encoding.
        """
        text = compose_title_abstract_text(
            {
                "title": seed_paper.title,
                "abstract": seed_paper.abstract,
            }
        )
        return text or str(seed_paper.paper_id)

    def _embed_candidates_in_memory(
        self,
        candidate_ids: List[str],
        candidates: Dict[str, Paper],
        embeddings_map: Dict[str, np.ndarray],
    ) -> None:
        """Embed rerank candidates without mutating the persistent corpus cache.

        :param List[str] candidate_ids: Candidate IDs needing embeddings.
        :param Dict[str, Paper] candidates: Candidate paper payloads.
        :param Dict[str, np.ndarray] embeddings_map: In-memory embedding map to update.
        :return None: Mutates ``embeddings_map`` in place when encoding succeeds.
        """
        if not candidate_ids or self.embedding_builder is None:
            return

        model_profile = getattr(self.embedding_builder, "model_profile", None)
        encode_texts = getattr(self.embedding_builder, "_encode_texts", None)
        if model_profile is None or not callable(
            getattr(model_profile, "format_document", None)
        ):
            return
        if not callable(encode_texts):
            return

        texts: List[str] = []
        ordered_candidate_ids: List[str] = []
        for paper_id in candidate_ids:
            paper = candidates.get(paper_id)
            if paper is None:
                continue

            document_text = str(
                model_profile.format_document(self._paper_embedding_metadata(paper))
            ).strip()
            if not document_text:
                document_text = str(paper.paper_id)
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
            logger.debug(
                "Hybrid in-memory candidate embedding encode failed; rerank falls back "
                "to non-semantic scoring (%s: %s).",
                type(exc).__name__,
                exc,
            )
            return

        encoded_array = np.asarray(encoded, dtype=np.float32)
        if encoded_array.ndim == 1:
            encoded_array = encoded_array.reshape(1, -1)
        if encoded_array.shape[0] != len(ordered_candidate_ids):
            logger.warning(
                "Hybrid candidate encode returned %d rows for %d candidates; "
                "skipping semantic rerank enrichment for this batch.",
                int(encoded_array.shape[0]),
                len(ordered_candidate_ids),
            )
            return

        for idx, paper_id in enumerate(ordered_candidate_ids):
            embeddings_map[paper_id] = encoded_array[idx]

    def _ensure_candidate_embeddings(
        self, seed_paper: Paper, candidates: Dict[str, Paper]
    ) -> Optional[np.ndarray]:
        """Ensure seed/candidate embeddings are materialized for reranking.

        :param Paper seed_paper: Seed paper.
        :param Dict[str, Paper] candidates: Candidate paper pool.
        :return Optional[np.ndarray]: Seed embedding when available.
        """
        if self.embedding_builder is None:
            return None

        embeddings_map = getattr(self.embedding_builder, "embeddings", None)
        if not isinstance(embeddings_map, dict):
            return None

        seed_embedding = embeddings_map.get(seed_paper.paper_id)
        if seed_embedding is None:
            try:
                seed_text = self.embedding_builder._format_seed_for_embedding(
                    seed_text=self._seed_query_text(seed_paper),
                    seed_metadata={
                        "title": seed_paper.title or "",
                        "abstract": seed_paper.abstract or "",
                    },
                    seed_is_free_text_query=str(seed_paper.paper_id).startswith(
                        "query:"
                    ),
                )
                seed_embedding = self.embedding_builder._encode_texts(
                    [seed_text], show_progress_bar=False
                )[0]
            except Exception as exc:
                logger.debug(
                    "Hybrid seed embedding unavailable for %s; rerank falls back to "
                    "non-semantic seed scoring (%s: %s).",
                    seed_paper.paper_id,
                    type(exc).__name__,
                    exc,
                )
                seed_embedding = None
            if seed_embedding is not None:
                embeddings_map[seed_paper.paper_id] = seed_embedding

        missing_ids = [
            paper_id for paper_id in candidates if paper_id not in embeddings_map
        ]
        if missing_ids and seed_embedding is not None:
            self._embed_candidates_in_memory(
                candidate_ids=missing_ids,
                candidates=candidates,
                embeddings_map=embeddings_map,
            )

        # Transient seed-encode failures should degrade to non-semantic reranking.
        if seed_embedding is None:
            return None
        return np.asarray(seed_embedding, dtype=np.float32)

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
            and isinstance(getattr(self.embedding_builder, "embeddings", None), dict)
            and candidate.paper_id in self.embedding_builder.embeddings
        ):
            candidate_embedding = np.asarray(
                self.embedding_builder.embeddings[candidate.paper_id], dtype=np.float32
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
        alias_map: Dict[str, str] = {}

        # Step 1: Collect from citations
        logger.debug("Collecting papers via citations...")
        citation_papers = self.citation_builder.collect_papers(seed_id)
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
        self._register_aliases(alias_map, seed_paper.paper_id, seed_paper)
        citation_seed_relations = getattr(self.citation_builder, "seed_relations", {})

        candidate_pool: Dict[str, Paper] = {}
        candidate_sources: Dict[str, Set[str]] = {}
        for paper in citation_papers.values():
            if paper.paper_id == seed_paper.paper_id or paper.is_seed:
                continue
            resolved = self._resolve_alias(alias_map, paper)
            if resolved == seed_paper.paper_id:
                papers[seed_paper.paper_id] = self._merge_paper_metadata(
                    papers[seed_paper.paper_id], paper
                )
                self._register_aliases(alias_map, seed_paper.paper_id, paper)
                continue
            if resolved is not None:
                candidate_pool[resolved] = self._merge_paper_metadata(
                    candidate_pool[resolved], paper
                )
                candidate_sources.setdefault(resolved, set()).add("citation")
                relation = self._merge_seed_relation(
                    self.seed_relations.get(resolved, ""),
                    str(citation_seed_relations.get(paper.paper_id, "citation")),
                )
                if relation:
                    self.seed_relations[resolved] = relation
                self._register_aliases(alias_map, resolved, paper)
                continue

            canonical_id = str(paper.paper_id)
            candidate_pool[canonical_id] = paper
            candidate_sources[canonical_id] = {"citation"}
            relation = str(citation_seed_relations.get(paper.paper_id, "citation"))
            if relation:
                self.seed_relations[canonical_id] = relation
            self._register_aliases(alias_map, canonical_id, paper)

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

        try:
            semantic_papers = self.embedding_builder.collect_papers(seed_id)
        except Exception as exc:
            raise RuntimeError(f"Semantic enrichment failed: {exc}") from exc

        for paper in semantic_papers.values():
            if paper.is_seed:
                continue
            resolved = self._resolve_alias(alias_map, paper)
            if resolved == seed_paper.paper_id:
                papers[seed_paper.paper_id] = self._merge_paper_metadata(
                    papers[seed_paper.paper_id], paper
                )
                self._register_aliases(alias_map, seed_paper.paper_id, paper)
                continue
            if resolved is not None:
                candidate_pool[resolved] = self._merge_paper_metadata(
                    candidate_pool[resolved], paper
                )
                candidate_sources.setdefault(resolved, set()).add("semantic")
                self.seed_relations.setdefault(resolved, "semantic_only")
                self._register_aliases(alias_map, resolved, paper)
                continue

            canonical_id = str(paper.paper_id)
            candidate_pool[canonical_id] = paper
            candidate_sources[canonical_id] = {"semantic"}
            self.seed_relations.setdefault(canonical_id, "semantic_only")
            self._register_aliases(alias_map, canonical_id, paper)

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
                self.seed_relations[paper_id] = self._merge_seed_relation(
                    self.seed_relations.get(paper_id, ""),
                    "semantic_only",
                )
            self.paper_sources[paper_id] = "both" if len(tags) > 1 else next(iter(tags))
            if "citation" in tags:
                self.seed_relations.setdefault(paper_id, "citation")

        logger.info("Added %s semantic papers", added_semantic)

        return papers

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
        }

        max_edges = HYBRID_CONFIG.max_edges_per_node
        if not max_edges or max_edges <= 0:
            return graph, actual_seed_id

        filtered_graph = nx.Graph()
        filtered_graph.graph.update(graph.graph)
        filtered_graph.add_nodes_from(graph.nodes(data=True))

        for u, v, weight in select_capped_undirected_edges(
            graph.edges(data=True), max_edges
        ):
            filtered_graph.add_edge(u, v, weight=weight)

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
