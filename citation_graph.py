#!/usr/bin/env python3
"""
Citation graph generator (backward compatibility wrapper).

This script maintains backward compatibility with the original citation_graph.py
interface while using the new unified CiteMesh architecture.

For new projects, use the unified CLI: python citemesh.py build --strategy citation
"""

import argparse
import logging
from pathlib import Path

from citemesh.strategies.citation import CitationGraphBuilder
from citemesh.visualization import generate_output_path, visualize_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point preserving original CLI interface."""
    parser = argparse.ArgumentParser(
        description="Citation-based graph visualization (Connected Papers style)"
    )

    parser.add_argument("paper_id", type=str, help="Paper ID (DOI, arXiv ID, or S2 ID)")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output PNG file (auto-named if not specified)",
    )

    parser.add_argument(
        "-p", "--max-papers", type=int, default=40, help="Maximum papers (default: 40)"
    )

    parser.add_argument(
        "-c",
        "--max-citations",
        type=int,
        default=20,
        help="Maximum citations to fetch (default: 20)",
    )

    parser.add_argument(
        "-r",
        "--max-references",
        type=int,
        default=20,
        help="Maximum references to fetch (default: 20)",
    )

    parser.add_argument(
        "-s",
        "--similarity-threshold",
        type=float,
        default=0.2,
        help="Minimum similarity for edges (default: 0.2)",
    )

    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=100,
        help="Layout iterations (default: 100)",
    )

    parser.add_argument(
        "-d", "--dpi", type=int, default=150, help="Output DPI (default: 150)"
    )

    args = parser.parse_args()

    try:
        # Build graph using new unified architecture
        logger.info("Building citation graph...")
        builder = CitationGraphBuilder(
            max_papers=args.max_papers,
            max_citations=args.max_citations,
            max_references=args.max_references,
            similarity_threshold=args.similarity_threshold,
            fetch_references=True,  # Enable real bibliographic coupling
        )

        graph, seed_id = builder.build_graph(args.paper_id)

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = generate_output_path(graph, seed_id)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Visualize using unified visualization
        logger.info("Creating visualization...")
        visualize_graph(
            graph, seed_id, output_path, iterations=args.iterations, dpi=args.dpi
        )

        logger.info(f"✓ Visualization saved to {output_path}")
        print(
            f"\nGraph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        print(f"Output: {output_path}")

    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
