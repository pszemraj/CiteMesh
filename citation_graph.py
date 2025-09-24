#!/usr/bin/env python3
"""
Citation Network Graph Builder with Connected Papers-style Visualization

A tool for building and visualizing citation networks from academic papers
using Semantic Scholar's API. Supports both traditional citation graphs
and similarity-based layouts inspired by Connected Papers.

Author: Research Tools
License: MIT
"""

import argparse
import json
import logging
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from pyvis.network import Network
from semanticscholar import SemanticScholar
from sklearn.manifold import MDS, TSNE
from tqdm import tqdm


class LayoutStyle(Enum):
    """Available layout algorithms."""

    FORCE = "force"  # Traditional force-directed
    SIMILARITY = "similarity"  # Connected Papers style


@dataclass
class GraphConfig:
    """Configuration for graph visualization."""

    height: str = "750px"
    width: str = "100%"
    bgcolor: str = "#fafafa"
    font_color: str = "#2d3748"
    gravity: int = -5000
    central_gravity: float = 0.3
    spring_length: int = 100
    max_title_length: int = 50
    layout_style: LayoutStyle = LayoutStyle.SIMILARITY


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
    references: List[str] = field(default_factory=list)  # Papers this cites
    citations: List[str] = field(default_factory=list)  # Papers citing this


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
    Builds citation networks from academic papers with multiple visualization options.

    Supports both traditional force-directed citation graphs and Connected Papers-style
    similarity-based layouts using bibliographic coupling and co-citation analysis.
    """

    def __init__(self, config: Optional[GraphConfig] = None, rate_limit: float = 0.5):
        """
        Initialize the citation graph builder.

        Args:
            config: Graph visualization configuration
            rate_limit: Seconds to wait between API calls
        """
        self.client = SemanticScholar()
        self.config = config or GraphConfig()
        self.rate_limit = max(0.0, rate_limit)
        self.logger = logging.getLogger(__name__)
        self._paper_cache: Dict[str, PaperNode] = {}

    def _truncate_title(self, title: str) -> str:
        """Truncate title to maximum length if needed."""
        if not title:
            return "Unknown"
        max_len = self.config.max_title_length
        return f"{title[:max_len]}..." if len(title) > max_len else title

    def _create_node_from_paper(self, paper) -> PaperNode:
        """Create a PaperNode from a Semantic Scholar Paper object."""
        if paper.paperId in self._paper_cache:
            return self._paper_cache[paper.paperId]

        authors = []
        if paper.authors:
            authors = [a.name for a in paper.authors[:5] if a.name]

        # Extract DOI from externalIds if available
        doi = None
        if hasattr(paper, "externalIds") and paper.externalIds:
            doi = paper.externalIds.get("DOI")

        node = PaperNode(
            id=paper.paperId,
            title=paper.title or "Unknown Title",
            year=paper.year,
            citation_count=paper.citationCount,
            authors=authors,
            abstract=paper.abstract[:500] if paper.abstract else None,
            venue=paper.venue,
            doi=doi,
            url=paper.url,
        )

        self._paper_cache[paper.paperId] = node
        return node

    def _create_node_from_dict(self, paper_dict: dict) -> PaperNode:
        """Create a PaperNode from a dictionary (e.g., from Citation/Reference objects)."""
        paper_id = paper_dict.get("paperId")

        if paper_id and paper_id in self._paper_cache:
            return self._paper_cache[paper_id]

        authors = []
        if "authors" in paper_dict and paper_dict["authors"]:
            authors_data = paper_dict["authors"][:5]
            for a in authors_data:
                if isinstance(a, dict):
                    if a.get("name"):
                        authors.append(a["name"])
                elif hasattr(a, "name") and a.name:
                    authors.append(a.name)

        doi = None
        if "externalIds" in paper_dict and paper_dict["externalIds"]:
            doi = paper_dict["externalIds"].get("DOI")

        node = PaperNode(
            id=paper_id,
            title=paper_dict.get("title", "Unknown Title"),
            year=paper_dict.get("year"),
            citation_count=paper_dict.get("citationCount"),
            authors=authors,
            abstract=paper_dict.get("abstract", "")[:500]
            if paper_dict.get("abstract")
            else None,
            venue=paper_dict.get("venue"),
            doi=doi,
            url=paper_dict.get("url"),
        )

        if paper_id:
            self._paper_cache[paper_id] = node
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

                # Store references and citations in node for similarity calculation
                if max_citations > 0:
                    citations = self.client.get_paper_citations(
                        paper.paperId, limit=max_citations
                    )
                    citation_list = list(citations)
                    for cit in citation_list:
                        try:
                            if cit["citingPaper"] and cit["citingPaper"].get("paperId"):
                                node.citations.append(cit["citingPaper"]["paperId"])
                        except (KeyError, TypeError):
                            continue

                if max_references > 0:
                    references = self.client.get_paper_references(
                        paper.paperId, limit=max_references
                    )
                    reference_list = list(references)
                    for ref in reference_list:
                        try:
                            if ref["citedPaper"] and ref["citedPaper"].get("paperId"):
                                node.references.append(ref["citedPaper"]["paperId"])
                        except (KeyError, TypeError):
                            continue

                # Add node with all attributes
                graph.add_node(node.id, **asdict(node))

                self.logger.info(
                    f"[Depth {current_depth}] Processing: {self._truncate_title(node.title)} "
                    f"({node.year}) - {node.citation_count or 0} citations"
                )

                # Process citations
                if max_citations > 0 and current_depth < depth:
                    citations = self.client.get_paper_citations(
                        paper.paperId, limit=max_citations
                    )

                    citation_list = list(citations)
                    for i, citation in enumerate(
                        tqdm(citation_list, desc="Processing citations", leave=False)
                    ):
                        try:
                            citing_paper_data = citation["citingPaper"]
                            if citing_paper_data and citing_paper_data.get("paperId"):
                                citing_node = self._create_node_from_dict(
                                    citing_paper_data
                                )
                                graph.add_node(citing_node.id, **asdict(citing_node))
                                graph.add_edge(
                                    citing_node.id,
                                    node.id,
                                    type="cited_by",
                                    color="#3b82f6",
                                    weight=1,
                                )

                                # Recursive fetch for important papers
                                if current_depth < depth - 1 and i < 5:
                                    fetch_and_add(citing_node.id, current_depth + 1)
                        except (KeyError, TypeError):
                            self.logger.debug("Citation missing citingPaper data")

                # Process references
                if max_references > 0:
                    references = self.client.get_paper_references(
                        paper.paperId, limit=max_references
                    )

                    reference_list = list(references)
                    for i, reference in enumerate(
                        tqdm(reference_list, desc="Processing references", leave=False)
                    ):
                        try:
                            cited_paper_data = reference["citedPaper"]
                            if cited_paper_data and cited_paper_data.get("paperId"):
                                cited_node = self._create_node_from_dict(
                                    cited_paper_data
                                )
                                graph.add_node(cited_node.id, **asdict(cited_node))
                                graph.add_edge(
                                    node.id,
                                    cited_node.id,
                                    type="cites",
                                    color="#ef4444",
                                    weight=1,
                                )

                                # Recursive fetch for important papers
                                if current_depth < depth - 1 and i < 5:
                                    fetch_and_add(cited_node.id, current_depth + 1)
                        except (KeyError, TypeError):
                            self.logger.debug("Reference missing citedPaper data")

                return node.id

            except Exception as e:
                self.logger.error(f"Error processing {pid}: {e}")
                if current_depth == 0:
                    raise RuntimeError(f"Failed to fetch root paper: {e}")
                return None

        # Start building
        print(f"Building citation graph for {paper_id}...")
        print(
            f"Settings: depth={depth}, max_citations={max_citations}, max_references={max_references}"
        )

        root_id = fetch_and_add(paper_id)
        if not root_id:
            raise ValueError(f"Could not find paper: {paper_id}")

        self.logger.info(
            f"Graph complete: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )

        return graph

    def compute_similarity_matrix(self, graph: nx.DiGraph) -> np.ndarray:
        """
        Compute pairwise similarity between papers based on citation relationships.

        Uses bibliographic coupling (shared references) and co-citation (shared citations).
        """
        nodes = list(graph.nodes())
        n = len(nodes)
        similarity = np.zeros((n, n))

        # Calculate similarity for each pair
        for i, node1 in enumerate(nodes):
            data1 = graph.nodes[node1]
            refs1 = set(data1.get("references", []))
            cits1 = set(data1.get("citations", []))

            for j, node2 in enumerate(nodes[i + 1 :], start=i + 1):
                data2 = graph.nodes[node2]
                refs2 = set(data2.get("references", []))
                cits2 = set(data2.get("citations", []))

                # Bibliographic coupling (shared references)
                if refs1 and refs2:
                    shared_refs = refs1 & refs2
                    bc_score = len(shared_refs) / len(refs1 | refs2)
                else:
                    bc_score = 0

                # Co-citation (shared citations)
                if cits1 and cits2:
                    shared_cits = cits1 & cits2
                    cc_score = len(shared_cits) / len(cits1 | cits2)
                else:
                    cc_score = 0

                # Combined similarity
                sim = 0.6 * bc_score + 0.4 * cc_score

                # Adjust for temporal proximity
                year1 = data1.get("year")
                year2 = data2.get("year")
                if year1 and year2:
                    year_diff = abs(year1 - year2)
                    time_factor = 1.0 / (1.0 + year_diff / 10.0)
                    sim *= time_factor

                similarity[i, j] = sim
                similarity[j, i] = sim

        np.fill_diagonal(similarity, 1.0)
        return similarity

    def compute_layout(
        self, graph: nx.DiGraph, layout_style: LayoutStyle = None
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute 2D layout for the graph nodes.

        Args:
            graph: NetworkX graph
            layout_style: Layout algorithm to use

        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        if layout_style is None:
            layout_style = self.config.layout_style

        if layout_style == LayoutStyle.SIMILARITY:
            # Use similarity-based layout (Connected Papers style)
            return self._compute_similarity_layout(graph)
        else:
            # Use traditional force-directed layout
            return self._compute_force_layout(graph)

    def _compute_force_layout(
        self, graph: nx.DiGraph
    ) -> Dict[str, Tuple[float, float]]:
        """Compute traditional force-directed layout."""
        pos = nx.spring_layout(
            graph, k=2 / math.sqrt(graph.number_of_nodes()), iterations=50, seed=42
        )

        # Scale to viewport
        return self._scale_positions(pos, width=800, height=600, padding=50)

    def _compute_similarity_layout(
        self, graph: nx.DiGraph
    ) -> Dict[str, Tuple[float, float]]:
        """Compute similarity-based layout using dimensionality reduction."""
        n = graph.number_of_nodes()
        if n <= 2:
            # Simple layout for very few nodes
            nodes = list(graph.nodes())
            if n == 1:
                return {nodes[0]: (400, 300)}
            return {nodes[0]: (300, 300), nodes[1]: (500, 300)}

        # Compute similarity matrix
        similarity = self.compute_similarity_matrix(graph)
        distance = 1 - similarity

        # Use dimensionality reduction
        if n > 30:
            # t-SNE for many nodes
            tsne = TSNE(
                n_components=2,
                metric="precomputed",
                init="random",
                perplexity=min(30, n - 1),
                max_iter=1000,
                random_state=42,
            )
            coords = tsne.fit_transform(distance)
        else:
            # MDS for fewer nodes (preserves distances better)
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
            coords = mds.fit_transform(distance)

        # Convert to position dictionary
        nodes = list(graph.nodes())
        positions = {nodes[i]: (coords[i, 0], coords[i, 1]) for i in range(len(nodes))}

        return self._scale_positions(positions, width=800, height=600, padding=50)

    def _scale_positions(
        self, positions: Dict, width: int, height: int, padding: int
    ) -> Dict[str, Tuple[float, float]]:
        """Scale positions to fit within viewport."""
        if not positions:
            return {}

        # Get coordinate ranges
        x_values = [p[0] for p in positions.values()]
        y_values = [p[1] for p in positions.values()]

        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1

        # Scale to viewport
        scaled = {}
        for node, (x, y) in positions.items():
            scaled_x = padding + (width - 2 * padding) * (x - x_min) / x_range
            scaled_y = padding + (height - 2 * padding) * (y - y_min) / y_range
            scaled[node] = (scaled_x, scaled_y)

        return scaled

    def compute_statistics(self, graph: nx.DiGraph) -> GraphStatistics:
        """Compute statistics about the citation graph."""
        stats = GraphStatistics(
            num_nodes=graph.number_of_nodes(),
            num_edges=graph.number_of_edges(),
            num_citations=len(
                [e for e in graph.edges(data=True) if e[2].get("type") == "cited_by"]
            ),
            num_references=len(
                [e for e in graph.edges(data=True) if e[2].get("type") == "cites"]
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
        except Exception:
            stats.avg_clustering = 0.0

        # Diameter (only for strongly connected component)
        try:
            largest_cc = max(nx.weakly_connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_cc)
            if nx.is_strongly_connected(subgraph):
                stats.diameter = nx.diameter(subgraph)
        except Exception:
            pass

        # Most cited/citing papers
        in_degrees = graph.in_degree()
        stats.most_cited_papers = [
            (graph.nodes[node]["title"], degree)
            for node, degree in sorted(in_degrees, key=lambda x: x[1], reverse=True)[:5]
        ]

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

    def visualize(
        self,
        graph: nx.DiGraph,
        output_path: Path,
        layout_style: LayoutStyle = None,
    ) -> None:
        """
        Create an interactive HTML visualization of the graph.

        Args:
            graph: NetworkX directed graph to visualize
            output_path: Path to save the HTML file
            layout_style: Layout algorithm to use
        """
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot visualize empty graph")

        if layout_style is None:
            layout_style = self.config.layout_style

        # Compute layout
        positions = self.compute_layout(graph, layout_style)

        # Get visual scales
        years = [d.get("year", 2020) for _, d in graph.nodes(data=True)]
        min_year = min(years) if years else 2020
        max_year = max(years) if years else 2024

        citations = [d.get("citation_count", 0) for _, d in graph.nodes(data=True)]
        citation_percentiles = (
            np.percentile(citations, [50, 75, 90, 95])
            if citations
            else [10, 20, 30, 40]
        )

        # Create network
        net = Network(
            height=self.config.height,
            width=self.config.width,
            bgcolor=self.config.bgcolor,
            font_color=self.config.font_color,
            directed=True,
        )

        # Add nodes
        for node_id, attrs in graph.nodes(data=True):
            x, y = positions[node_id]

            # Size based on citations
            cit_count = attrs.get("citation_count", 0)
            if cit_count == 0:
                size = 8
            elif cit_count <= citation_percentiles[0]:
                size = 12
            elif cit_count <= citation_percentiles[1]:
                size = 18
            elif cit_count <= citation_percentiles[2]:
                size = 24
            else:
                size = 30

            # Color based on year (blue gradient)
            if attrs.get("year"):
                year_norm = (attrs["year"] - min_year) / max(max_year - min_year, 1)
                lightness = 75 - year_norm * 35  # 75% to 40%
                color = f"hsl(210, 60%, {lightness}%)"
            else:
                color = "hsl(210, 30%, 60%)"

            # Create hover text
            hover = f"""<div style='max-width: 300px'>
                <b>{attrs.get("title", "Unknown")}</b><br>
                Year: {attrs.get("year", "N/A")}<br>
                Citations: {attrs.get("citation_count", 0)}<br>
                Authors: {", ".join(attrs.get("authors", [])[:3])}
            </div>"""

            net.add_node(
                node_id,
                label=self._truncate_title(attrs.get("title", "Unknown")),
                title=hover,
                size=size,
                color=color,
                x=x,
                y=y,
                physics=False if layout_style == LayoutStyle.SIMILARITY else True,
            )

        # Add edges
        for source, target, attrs in graph.edges(data=True):
            edge_type = attrs.get("type", "cites")
            color = "#3b82f6" if edge_type == "cited_by" else "#ef4444"
            net.add_edge(
                source,
                target,
                color={"color": color, "opacity": 0.5},
                arrows={"to": {"enabled": True, "scaleFactor": 0.5}},
            )

        # Configure physics
        if layout_style == LayoutStyle.FORCE:
            net.barnes_hut(
                gravity=self.config.gravity,
                central_gravity=self.config.central_gravity,
                spring_length=self.config.spring_length,
            )
        else:
            # Minimal physics for similarity layout
            net.set_options("""
            {
                "physics": {
                    "enabled": false
                },
                "interaction": {
                    "dragNodes": true,
                    "dragView": true,
                    "zoomView": true
                }
            }
            """)

        # Enable navigation buttons
        net.show_buttons(filter_=["physics", "layout", "interaction"])

        # Save
        net.save_graph(str(output_path))
        self.logger.info(f"Visualization saved to {output_path}")

    def export_formats(self, graph: nx.DiGraph, base_path: Path) -> None:
        """Export graph in multiple formats."""
        # GraphML
        graphml_path = base_path.with_suffix(".graphml")
        nx.write_graphml(graph, graphml_path)
        self.logger.info(f"Exported GraphML to {graphml_path}")

        # GEXF
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
        help="Maximum traversal depth in the citation network (default: 1 for performance)",
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
        "--layout",
        type=str,
        choices=["force", "similarity"],
        default="similarity",
        help="Layout algorithm: 'force' for traditional, 'similarity' for Connected Papers style",
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
        config = GraphConfig(
            layout_style=LayoutStyle.SIMILARITY
            if args.layout == "similarity"
            else LayoutStyle.FORCE
        )
        builder = CitationGraphBuilder(config=config, rate_limit=args.rate_limit)

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
        builder.visualize(graph, args.output, layout_style=config.layout_style)

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
