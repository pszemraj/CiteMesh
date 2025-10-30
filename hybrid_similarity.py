#!/usr/bin/env python3
"""
Hybrid graph generator (backward compatibility wrapper).

This script maintains backward compatibility with the original hybrid_similarity.py
interface while using the new unified CiteMesh architecture.

For new projects, use the unified CLI: python citemesh.py build --strategy hybrid
"""

import argparse
import logging
from pathlib import Path

from citemesh.strategies.hybrid import HybridGraphBuilder
from citemesh.visualization import generate_output_path, visualize_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point preserving original CLI interface."""
    parser = argparse.ArgumentParser(
        description="Hybrid citation + semantic similarity graph visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "paper_id", type=str, help="Paper identifier (DOI, arXiv ID, or S2 ID)"
    )

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
        "--corpus-size",
        type=int,
        default=None,
        help="Max papers to load from ArXiv corpus (None = all in split)",
    )

    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split specification (default: train = full ~117k papers)",
    )

    parser.add_argument(
        "-s",
        "--similarity-threshold",
        type=float,
        default=0.35,
        help="Minimum similarity for edges (default: 0.35)",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="google/embeddinggemma-300m",
        help="Sentence transformer model (default: embeddinggemma-300m)",
    )

    parser.add_argument(
        "--max-semantic",
        type=int,
        default=10,
        help="Max papers to add from semantic similarity (default: 10)",
    )

    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=100,
        help="Layout iterations (default: 100)",
    )

    parser.add_argument(
        "--dpi", type=int, default=150, help="Output DPI (default: 150)"
    )

    args = parser.parse_args()

    try:
        # Build graph using new unified architecture
        logger.info("Building hybrid citation + semantic graph...")
        builder = HybridGraphBuilder(
            max_papers=args.max_papers,
            model_name=args.model,
            dataset_split=args.dataset_split,
            corpus_size=args.corpus_size,
            max_semantic_papers=args.max_semantic,
            similarity_threshold=args.similarity_threshold,
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
