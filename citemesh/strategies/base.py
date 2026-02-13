"""
Abstract base class for graph building strategies.

This module defines the interface that all graph building strategies must implement,
enabling the Strategy pattern for different similarity computation approaches.
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import numpy as np

from citemesh.core import TEMPORAL_CONFIG, Paper

logger = logging.getLogger(__name__)


class GraphBuilderStrategy(ABC):
    """
    Abstract base class for all paper graph building strategies.

    This class defines the template method pattern for building similarity graphs.
    Subclasses must implement the abstract methods for collecting papers and
    computing similarity, while the base class handles common graph construction logic.
    """

    def __init__(self, max_papers: int = 40, random_seed: Optional[int] = None):
        """
        Initialize the graph builder.

        :param int max_papers: Maximum number of papers to include in graph
        :param Optional[int] random_seed: Reserved random seed parameter kept for compatibility.
        """
        self.max_papers = max_papers
        self.random_seed = random_seed
        self.papers: Dict[str, Paper] = {}  # paper_id -> Paper object
        self._collection_summary: Optional[str] = None

    @abstractmethod
    def collect_papers(self, seed_id: str, **kwargs: Any) -> Dict[str, Paper]:
        """
        Collect papers for the graph using strategy-specific method.

        :param str seed_id: The seed paper identifier (DOI, arXiv ID, or S2 ID)
        :param kwargs: Strategy-specific parameters
        :return Dict[str, Paper]: Dictionary mapping paper IDs to Paper objects
        :raises ValueError: If seed paper cannot be found
        """
        pass

    @abstractmethod
    def compute_similarity(self, paper1: Paper, paper2: Paper) -> float:
        """
        Compute similarity between two papers using strategy-specific method.

        :param Paper paper1: First paper
        :param Paper paper2: Second paper
        :return float: Similarity score from 0.0 (not similar) to 1.0 (identical)
        """
        pass

    def should_create_edge(
        self, paper1: Paper, paper2: Paper, similarity: float
    ) -> bool:
        """
        Decide whether to create an edge based on similarity and paper properties.

        :param Paper paper1: First paper
        :param Paper paper2: Second paper
        :param float similarity: Computed similarity score
        :return bool: True if edge should be created
        """
        return similarity > 0.0

    def get_collection_summary(self) -> Optional[str]:
        """
        Optional one-line summary describing collected papers.

        Subclasses can set this to surface additional detail (e.g., reference counts)
        that should be shown to users on stdout.

        :return Optional[str]: Optional summary string to display after collection.
        """
        return self._collection_summary

    def _set_collection_summary(self, summary: str) -> None:
        """Allow subclasses to provide a collection summary."""
        self._collection_summary = summary

    def build_graph(self, seed_id: str, **kwargs: Any) -> Tuple[nx.Graph, str]:
        """
        Build the complete similarity graph.

        :param str seed_id: The seed paper identifier
        :param kwargs: Strategy-specific parameters passed to collect_papers.
        :return Tuple[nx.Graph, str]: Tuple of (NetworkX graph, seed paper ID)
        """
        # Step 1: Collect papers
        logger.info("Collecting papers using %s...", self.__class__.__name__)
        self.papers = self.collect_papers(seed_id, **kwargs)

        if not self.papers:
            raise ValueError("No papers collected")

        # Find actual seed paper ID (may have been normalized)
        seed_paper = next((p for p in self.papers.values() if p.is_seed), None)
        if not seed_paper:
            raise ValueError("Seed paper not found in collected papers")

        actual_seed_id = seed_paper.paper_id

        summary = self.get_collection_summary()
        if summary:
            logger.info(summary)
        else:
            logger.info("Collected %s papers", len(self.papers))
        logger.info("Seed paper: %s...", seed_paper.title[:50])

        # Step 2: Create graph with nodes
        graph = nx.Graph()
        for paper in self.papers.values():
            graph.add_node(
                paper.paper_id,
                paper=paper,  # Store full paper object
                # Also store individual attributes for backward compatibility
                title=paper.title,
                year=paper.year,
                authors=[a.name for a in paper.authors[:3]],
                citation_count=paper.citation_count,
                is_seed=paper.is_seed,
            )

        # Step 3: Compute similarities and create edges
        logger.info("Computing similarities and creating edges...")
        paper_list = list(self.papers.values())
        edges_created = 0

        for i, p1 in enumerate(paper_list):
            for j in range(i + 1, len(paper_list)):
                p2 = paper_list[j]

                # Compute similarity
                similarity = self.compute_similarity(p1, p2)

                # Decide whether to create edge
                if self.should_create_edge(p1, p2, similarity):
                    graph.add_edge(p1.paper_id, p2.paper_id, weight=similarity)
                    edges_created += 1

        logger.info(
            "Graph complete: %s nodes, %s edges",
            graph.number_of_nodes(),
            edges_created,
        )
        return graph, actual_seed_id

    # Utility methods for common similarity computations

    @staticmethod
    def temporal_similarity(paper1: Paper, paper2: Paper) -> float:
        """
        Compute temporal similarity based on publication year difference.

        :param Paper paper1: First paper
        :param Paper paper2: Second paper
        :return float: Temporal similarity score (0.0 to 1.0)
        """
        if paper1.year is None or paper2.year is None:
            return 0.5
        year_diff = abs(paper1.year - paper2.year)
        return TEMPORAL_CONFIG.year_similarity(year_diff)

    @staticmethod
    def citation_similarity(paper1: Paper, paper2: Paper) -> float:
        """
        Compute similarity based on citation counts (log scale).

        :param Paper paper1: First paper
        :param Paper paper2: Second paper
        :return float: Citation similarity score (0.0 to 1.0)
        """
        cit1 = paper1.citation_count
        cit2 = paper2.citation_count

        if cit1 > 0 and cit2 > 0:
            log_cit1 = np.log10(cit1 + 1)
            log_cit2 = np.log10(cit2 + 1)
            max_log = max(log_cit1, log_cit2)
            if max_log > 0:
                return 1.0 - abs(log_cit1 - log_cit2) / max_log
        return 0.3  # Default for papers without citations

    @staticmethod
    def bibliographic_coupling(paper1: Paper, paper2: Paper) -> float:
        """
        Compute bibliographic coupling strength.

        :param Paper paper1: First paper
        :param Paper paper2: Second paper
        :return float: Bibliographic coupling coefficient (0.0 to 1.0)
        """
        return paper1.reference_overlap(paper2)

    @staticmethod
    def exponential_temporal_decay(
        paper1: Paper, paper2: Paper, decay_factor: float = 8.0
    ) -> float:
        """
        Compute exponential temporal similarity decay.

        :param Paper paper1: First paper
        :param Paper paper2: Second paper
        :param float decay_factor: Controls decay rate (higher = slower decay)
        :return float: Similarity score (0.0 to 1.0)
        """
        if paper1.year is None or paper2.year is None:
            return 0.5
        year_diff = abs(paper1.year - paper2.year)
        return math.exp(-year_diff / decay_factor)
