"""
Abstract base class for graph building strategies.

This module defines the interface that all graph building strategies must implement,
enabling the Strategy pattern for different similarity computation approaches.
"""

import math
import random
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import torch

from citemesh.config import TEMPORAL_CONFIG
from citemesh.models import Paper


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

        Args:
            max_papers: Maximum number of papers to include in graph
            random_seed: Random seed for reproducibility (None = non-deterministic)
        """
        self.max_papers = max_papers
        self.random_seed = random_seed
        self.papers: Dict[str, Paper] = {}  # paper_id -> Paper object

        # Set random seeds for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

    @abstractmethod
    def collect_papers(self, seed_id: str, **kwargs) -> Dict[str, Paper]:
        """
        Collect papers for the graph using strategy-specific method.

        Args:
            seed_id: The seed paper identifier (DOI, arXiv ID, or S2 ID)
            **kwargs: Strategy-specific parameters

        Returns:
            Dictionary mapping paper IDs to Paper objects

        Raises:
            ValueError: If seed paper cannot be found
        """
        pass

    @abstractmethod
    def compute_similarity(self, paper1: Paper, paper2: Paper) -> float:
        """
        Compute similarity between two papers using strategy-specific method.

        Args:
            paper1: First paper
            paper2: Second paper

        Returns:
            Similarity score from 0.0 (not similar) to 1.0 (identical)
        """
        pass

    def should_create_edge(
        self, paper1: Paper, paper2: Paper, similarity: float
    ) -> bool:
        """
        Decide whether to create an edge based on similarity and paper properties.

        Default implementation: always create edge if similarity > 0.
        Subclasses can override for more sophisticated logic.

        Args:
            paper1: First paper
            paper2: Second paper
            similarity: Computed similarity score

        Returns:
            True if edge should be created
        """
        return similarity > 0.0

    def build_graph(self, seed_id: str, **kwargs) -> Tuple[nx.Graph, str]:
        """
        Build the complete similarity graph.

        This is the template method that orchestrates the graph building process.

        Args:
            seed_id: The seed paper identifier
            **kwargs: Strategy-specific parameters passed to collect_papers

        Returns:
            Tuple of (NetworkX graph, seed paper ID)
        """
        # Step 1: Collect papers
        print(f"Collecting papers using {self.__class__.__name__}...")
        self.papers = self.collect_papers(seed_id, **kwargs)

        if not self.papers:
            raise ValueError("No papers collected")

        # Find actual seed paper ID (may have been normalized)
        seed_paper = next((p for p in self.papers.values() if p.is_seed), None)
        if not seed_paper:
            raise ValueError("Seed paper not found in collected papers")

        actual_seed_id = seed_paper.paper_id

        print(f"Collected {len(self.papers)} papers")
        print(f"Seed paper: {seed_paper.title[:50]}...")

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
        print("Computing similarities and creating edges...")
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

        print(f"Graph complete: {graph.number_of_nodes()} nodes, {edges_created} edges")
        return graph, actual_seed_id

    # Utility methods for common similarity computations

    @staticmethod
    def temporal_similarity(paper1: Paper, paper2: Paper) -> float:
        """
        Compute temporal similarity based on publication year difference.

        Uses configuration from TemporalConfig.

        Args:
            paper1: First paper
            paper2: Second paper

        Returns:
            Temporal similarity score (0.0 to 1.0)
        """
        year_diff = abs(paper1.year - paper2.year)
        return TEMPORAL_CONFIG.year_similarity(year_diff)

    @staticmethod
    def citation_similarity(paper1: Paper, paper2: Paper) -> float:
        """
        Compute similarity based on citation counts (log scale).

        Papers with similar impact (citation counts) are considered more similar.

        Args:
            paper1: First paper
            paper2: Second paper

        Returns:
            Citation similarity score (0.0 to 1.0)
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

        Uses the Kessler (1963) formula:
            coupling = |shared_refs| / sqrt(|refs1| * |refs2|)

        Args:
            paper1: First paper
            paper2: Second paper

        Returns:
            Bibliographic coupling coefficient (0.0 to 1.0)
        """
        return paper1.reference_overlap(paper2)

    @staticmethod
    def exponential_temporal_decay(
        paper1: Paper, paper2: Paper, decay_factor: float = 8.0
    ) -> float:
        """
        Compute exponential temporal similarity decay.

        Args:
            paper1: First paper
            paper2: Second paper
            decay_factor: Controls decay rate (higher = slower decay)

        Returns:
            Similarity score (0.0 to 1.0)
        """
        year_diff = abs(paper1.year - paper2.year)
        return math.exp(-year_diff / decay_factor)
