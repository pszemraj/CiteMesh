#!/usr/bin/env python3
"""
Citation Network Graph Builder

A tool for building and visualizing citation networks from academic papers
using Semantic Scholar's API.

Author: Research Tools
License: MIT
"""

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from pyvis.network import Network
from semanticscholar import SemanticScholar
from semanticscholar.Paper import Paper


class EdgeType(Enum):
    """Types of edges in the citation graph."""

    CITES = "red"
    CITED_BY = "blue"


@dataclass
class GraphConfig:
    """Configuration for graph visualization."""

    height: str = "750px"
    width: str = "100%"
    bgcolor: str = "#222222"
    font_color: str = "white"
    gravity: int = -8000
    central_gravity: float = 0.3
    spring_length: int = 100
    max_title_length: int = 50


@dataclass
class PaperNode:
    """Represents a paper node in the citation graph."""

    id: str
    title: str
    year: Optional[int] = None
    citation_count: Optional[int] = None
    authors: List[str] = field(default_factory=list)
    abstract: Optional[str] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None


@dataclass
class GraphStatistics:
    """Statistics about the citation graph."""

    num_nodes: int
    num_edges: int
    num_citations: int
    num_references: int
    avg_degree: float
    max_in_degree: int
    max_out_degree: int
    density: float
    num_components: int
    largest_component_size: int
    diameter: Optional[int] = None
    avg_clustering: float = 0.0
    most_cited_papers: List[Tuple[str, int]] = field(default_factory=list)
    most_citing_papers: List[Tuple[str, int]] = field(default_factory=list)
    year_distribution: Dict[int, int] = field(default_factory=dict)


class CitationGraphBuilder:
    """
    Builds citation networks from academic papers.

    Uses Semantic Scholar's API to fetch paper metadata and citation
    relationships, then constructs a directed graph representation.
    """

    def __init__(self, config: Optional[GraphConfig] = None, rate_limit: float = 0.5):
        """
        Initialize the citation graph builder.

        Args:
            config: Graph visualization configuration
            rate_limit: Seconds to wait between API calls (min 0.1 for API limits)
        """
        self.client = SemanticScholar()
        self.config = config or GraphConfig()
        self.rate_limit = max(0.1, rate_limit)  # Enforce minimum rate limit
        self.logger = logging.getLogger(__name__)
        self._paper_cache: Dict[str, PaperNode] = {}

    def _truncate_title(self, title: str) -> str:
        """Truncate title to maximum length if needed."""
        if not title:
            return "Unknown"
        max_len = self.config.max_title_length
        return f"{title[:max_len]}..." if len(title) > max_len else title

    def _create_node_from_paper(self, paper: Paper) -> PaperNode:
        """Create a PaperNode from a Semantic Scholar Paper object."""
        # Cache check
        if paper.paperId in self._paper_cache:
            return self._paper_cache[paper.paperId]

        authors = []
        if paper.authors:
            authors = [a.name for a in paper.authors[:5] if a.name]

        node = PaperNode(
            id=paper.paperId,
            title=paper.title or "Unknown Title",
            year=paper.year,
            citation_count=paper.citationCount,
            authors=authors,
            abstract=paper.abstract[:500] if paper.abstract else None,
            venue=paper.venue,
            doi=paper.doi,
            url=paper.url,
        )

        self._paper_cache[paper.paperId] = node
        return node

    def build(
        self,
        paper_id: str,
        depth: int = 1,
        max_citations: int = 20,
        max_references: int = 20,
    ) -> nx.DiGraph:
        """
        Build the citation graph starting from a root paper.

        Args:
            paper_id: Root paper identifier (DOI, arXiv ID, or S2 ID)
            depth: Maximum depth to traverse
            max_citations: Maximum citations per paper
            max_references: Maximum references per paper

        Returns:
            NetworkX directed graph with paper nodes and citation edges

        Raises:
            ValueError: If the paper cannot be found
            RuntimeError: If the API is unreachable
        """
        graph = nx.DiGraph()
        visited: Set[str] = set()

        def fetch_and_add(pid: str, current_depth: int = 0) -> Optional[str]:
            """Recursively fetch papers and add to graph."""
            if pid in visited or current_depth > depth:
                return None

            visited.add(pid)

            try:
                # Fetch main paper
                paper = self.client.get_paper(pid)
                if not paper:
                    self.logger.warning(f"Paper not found: {pid}")
                    return None

                node = self._create_node_from_paper(paper)

                # Add node with all attributes
                graph.add_node(node.id, **asdict(node))

                self.logger.info(
                    f"[Depth {current_depth}] Processing: {self._truncate_title(node.title)} "
                    f"({node.year}) - {node.citation_count or 0} citations"
                )

                # Fetch citations
                if max_citations > 0 and current_depth < depth:
                    citations = self.client.get_paper_citations(
                        paper.paperId, limit=max_citations
                    )

                    for i, citation in enumerate(citations):
                        if citation.citingPaper:
                            citing_node = self._create_node_from_paper(
                                citation.citingPaper
                            )
                            graph.add_node(citing_node.id, **asdict(citing_node))
                            graph.add_edge(
                                citing_node.id,
                                node.id,
                                type=EdgeType.CITED_BY,
                                color=EdgeType.CITED_BY.value,
                                weight=1,
                            )

                            # Recursive fetch for important papers
                            if current_depth < depth - 1 and i < 5:
                                fetch_and_add(citing_node.id, current_depth + 1)

                # Fetch references
                if max_references > 0:
                    references = self.client.get_paper_references(
                        paper.paperId, limit=max_references
                    )

                    for i, reference in enumerate(references):
                        if reference.citedPaper:
                            cited_node = self._create_node_from_paper(
                                reference.citedPaper
                            )
                            graph.add_node(cited_node.id, **asdict(cited_node))
                            graph.add_edge(
                                node.id,
                                cited_node.id,
                                type=EdgeType.CITES,
                                color=EdgeType.CITES.value,
                                weight=1,
                            )

                            # Recursive fetch for important papers
                            if current_depth < depth - 1 and i < 5:
                                fetch_and_add(cited_node.id, current_depth + 1)

                return node.id

            except Exception as e:
                self.logger.error(f"Error processing {pid}: {e}")
                if current_depth == 0:
                    raise RuntimeError(f"Failed to fetch root paper: {e}")
                return None

        # Start building
        root_id = fetch_and_add(paper_id)
        if not root_id:
            raise ValueError(f"Could not find paper: {paper_id}")

        self.logger.info(
            f"Graph complete: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )

        return graph

    def compute_statistics(self, graph: nx.DiGraph) -> GraphStatistics:
        """
        Compute statistics about the citation graph.

        Args:
            graph: NetworkX directed graph

        Returns:
            GraphStatistics object with computed metrics
        """
        stats = GraphStatistics(
            num_nodes=graph.number_of_nodes(),
            num_edges=graph.number_of_edges(),
            num_citations=len(
                [
                    e
                    for e in graph.edges(data=True)
                    if e[2].get("type") == EdgeType.CITED_BY
                ]
            ),
            num_references=len(
                [
                    e
                    for e in graph.edges(data=True)
                    if e[2].get("type") == EdgeType.CITES
                ]
            ),
            avg_degree=sum(dict(graph.degree()).values())
            / max(graph.number_of_nodes(), 1),
            max_in_degree=max(dict(graph.in_degree()).values()) if graph.nodes() else 0,
            max_out_degree=max(dict(graph.out_degree()).values())
            if graph.nodes()
            else 0,
            density=nx.density(graph),
            num_components=nx.number_weakly_connected_components(graph),
            largest_component_size=len(
                max(nx.weakly_connected_components(graph), key=len)
            )
            if graph.nodes()
            else 0,
        )

        # Clustering coefficient
        try:
            stats.avg_clustering = nx.average_clustering(graph.to_undirected())
        except:
            stats.avg_clustering = 0.0

        # Diameter (only for strongly connected component)
        try:
            largest_cc = max(nx.weakly_connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_cc)
            if nx.is_strongly_connected(subgraph):
                stats.diameter = nx.diameter(subgraph)
        except:
            pass

        # Most cited papers (highest in-degree)
        in_degrees = graph.in_degree()
        stats.most_cited_papers = [
            (graph.nodes[node]["title"], degree)
            for node, degree in sorted(in_degrees, key=lambda x: x[1], reverse=True)[:5]
        ]

        # Most citing papers (highest out-degree)
        out_degrees = graph.out_degree()
        stats.most_citing_papers = [
            (graph.nodes[node]["title"], degree)
            for node, degree in sorted(out_degrees, key=lambda x: x[1], reverse=True)[
                :5
            ]
        ]

        # Year distribution
        years = [
            data.get("year") for _, data in graph.nodes(data=True) if data.get("year")
        ]
        stats.year_distribution = dict(Counter(years))

        return stats

    def visualize(self, graph: nx.DiGraph, output_path: Path) -> None:
        """
        Create an interactive HTML visualization of the graph.

        Args:
            graph: NetworkX directed graph to visualize
            output_path: Path to save the HTML file
        """
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot visualize empty graph")

        net = Network(
            height=self.config.height,
            width=self.config.width,
            bgcolor=self.config.bgcolor,
            font_color=self.config.font_color,
            directed=True,
        )

        # Add nodes with size based on centrality
        centrality = (
            nx.pagerank(graph) if graph.edges() else {n: 1 for n in graph.nodes()}
        )

        for node_id, attrs in graph.nodes(data=True):
            # Create label
            title = self._truncate_title(attrs.get("title", "Unknown"))
            year = attrs.get("year", "N/A")
            citations = attrs.get("citation_count", 0)

            label = f"{title}\n({year})"
            if citations:
                label += f"\n[{citations} citations]"

            # Create hover text
            hover_parts = [
                f"<b>{attrs.get('title', 'Unknown')}</b>",
                f"Year: {year}",
                f"Citations: {citations}",
                f"Authors: {', '.join(attrs.get('authors', []))[:100]}",
            ]

            if attrs.get("venue"):
                hover_parts.append(f"Venue: {attrs['venue']}")

            if attrs.get("abstract"):
                hover_parts.append(f"<br><i>{attrs['abstract'][:200]}...</i>")

            # Node size based on PageRank centrality
            size = 10 + centrality.get(node_id, 0) * 100

            net.add_node(
                node_id,
                label=label,
                title="<br>".join(hover_parts),
                size=size,
                color="gold" if graph.in_degree(node_id) > 5 else "lightblue",
            )

        # Add edges
        for source, target, attrs in graph.edges(data=True):
            net.add_edge(
                source,
                target,
                color=attrs.get("color", "gray"),
                arrows={"to": {"enabled": True, "scaleFactor": 0.5}},
            )

        # Configure physics
        net.barnes_hut(
            gravity=self.config.gravity,
            central_gravity=self.config.central_gravity,
            spring_length=self.config.spring_length,
        )

        # Enable navigation buttons
        net.show_buttons(filter_=["physics", "layout", "interaction"])

        # Save
        net.save_graph(str(output_path))
        self.logger.info(f"Visualization saved to {output_path}")

    def export_formats(self, graph: nx.DiGraph, base_path: Path) -> None:
        """
        Export graph in multiple formats.

        Args:
            graph: NetworkX directed graph
            base_path: Base path for output files (without extension)
        """
        # GraphML
        graphml_path = base_path.with_suffix(".graphml")
        nx.write_graphml(graph, graphml_path)
        self.logger.info(f"Exported GraphML to {graphml_path}")

        # GEXF (for Gephi)
        gexf_path = base_path.with_suffix(".gexf")
        nx.write_gexf(graph, gexf_path)
        self.logger.info(f"Exported GEXF to {gexf_path}")

        # JSON
        json_path = base_path.with_suffix(".json")
        data = nx.node_link_data(graph)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        self.logger.info(f"Exported JSON to {json_path}")


def setup_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level."""
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(verbosity, len(levels) - 1)]

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build and visualize citation networks from academic papers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "paper_id", help="Paper identifier (DOI, arXiv ID, or Semantic Scholar ID)"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("citation_graph.html"),
        help="Output file path for visualization",
    )

    parser.add_argument(
        "-d",
        "--depth",
        type=int,
        default=1,
        help="Maximum traversal depth in the citation network",
    )

    parser.add_argument(
        "-c",
        "--max-citations",
        type=int,
        default=20,
        help="Maximum number of citations to fetch per paper",
    )

    parser.add_argument(
        "-r",
        "--max-references",
        type=int,
        default=20,
        help="Maximum number of references to fetch per paper",
    )

    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export graph in multiple formats (GraphML, GEXF, JSON)",
    )

    parser.add_argument(
        "--stats", action="store_true", help="Print detailed graph statistics"
    )

    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Seconds to wait between API calls",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be repeated)",
    )

    return parser.parse_args()


def print_statistics(stats: GraphStatistics) -> None:
    """Print graph statistics in a formatted manner."""
    print("\n" + "=" * 60)
    print("CITATION GRAPH STATISTICS")
    print("=" * 60)

    print("\nGraph Structure:")
    print(f"  Nodes: {stats.num_nodes}")
    print(f"  Edges: {stats.num_edges}")
    print(f"  Citations: {stats.num_citations}")
    print(f"  References: {stats.num_references}")
    print(f"  Density: {stats.density:.4f}")
    print(f"  Components: {stats.num_components}")
    print(f"  Largest Component: {stats.largest_component_size} nodes")

    if stats.diameter:
        print(f"  Diameter: {stats.diameter}")

    print("\nDegree Statistics:")
    print(f"  Average Degree: {stats.avg_degree:.2f}")
    print(f"  Max In-Degree: {stats.max_in_degree}")
    print(f"  Max Out-Degree: {stats.max_out_degree}")
    print(f"  Clustering Coefficient: {stats.avg_clustering:.4f}")

    if stats.most_cited_papers:
        print("\nMost Cited Papers:")
        for title, count in stats.most_cited_papers:
            print(f"  - {title}: {count} citations")

    if stats.most_citing_papers:
        print("\nPapers with Most References:")
        for title, count in stats.most_citing_papers:
            print(f"  - {title}: {count} references")

    if stats.year_distribution:
        print("\nYear Distribution:")
        for year in sorted(stats.year_distribution.keys())[-10:]:
            print(f"  {year}: {stats.year_distribution[year]} papers")

    print("=" * 60 + "\n")


def main() -> int:
    """Main entry point."""
    args = parse_arguments()
    setup_logging(args.verbose)

    try:
        # Build graph
        builder = CitationGraphBuilder(rate_limit=args.rate_limit)

        graph = builder.build(
            paper_id=args.paper_id,
            depth=args.depth,
            max_citations=args.max_citations,
            max_references=args.max_references,
        )

        # Compute and print statistics
        if args.stats:
            stats = builder.compute_statistics(graph)
            print_statistics(stats)

        # Visualize
        builder.visualize(graph, args.output)

        # Export additional formats
        if args.export_all:
            base_path = args.output.with_suffix("")
            builder.export_formats(graph, base_path)

        return 0

    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
