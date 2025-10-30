"""
CiteMesh CLI entry point for module execution.

This allows running: python -m citemesh build "arxiv:123" --strategy citation
"""

import argparse
import logging
import sys
from pathlib import Path

from citemesh.strategies.citation import CitationGraphBuilder
from citemesh.strategies.embedding import EmbeddingGraphBuilder
from citemesh.strategies.hybrid import HybridGraphBuilder
from citemesh.visualization import generate_output_path, visualize_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_citation_graph(args):
    """Build graph using citation strategy."""
    builder = CitationGraphBuilder(
        max_papers=args.max_papers,
        max_citations=args.max_citations,
        max_references=args.max_references,
        similarity_threshold=args.similarity_threshold,
        fetch_references=not args.no_references,
        random_seed=args.seed,
    )

    graph, seed_id = builder.build_graph(args.paper_id)
    return graph, seed_id


def build_embedding_graph(args):
    """Build graph using embedding strategy."""
    builder = EmbeddingGraphBuilder(
        max_papers=args.max_papers,
        model_name=args.model,
        dataset_split=args.dataset_split,
        corpus_size=args.corpus_size,
        top_k=args.top_k,
        random_seed=args.seed,
    )

    graph, seed_id = builder.build_graph(args.paper_id)
    return graph, seed_id


def build_hybrid_graph(args):
    """Build graph using hybrid strategy."""
    builder = HybridGraphBuilder(
        max_papers=args.max_papers,
        max_citations=args.max_citations,
        max_references=args.max_references,
        max_semantic=args.max_semantic,
        model_name=args.model,
        dataset_split=args.dataset_split,
        random_seed=args.seed,
    )

    graph, seed_id = builder.build_graph(args.paper_id)
    return graph, seed_id


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CiteMesh: Create Connected Papers-style citation graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Citation-based graph (fast, uses S2 API)
  citemesh build "arxiv:1706.03762" --strategy citation

  # Embedding-based graph (semantic similarity)
  citemesh build "arxiv:1706.03762" --strategy embedding

  # Hybrid approach (combines both)
  citemesh build "arxiv:1706.03762" --strategy hybrid

  # Custom output path
  citemesh build "10.1038/nature14539" -o my_graph.png

  # Quick test with fewer papers
  citemesh build "arxiv:1810.04805" -p 20 --strategy citation
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Build command
    build_parser = subparsers.add_parser(
        "build", help="Build and visualize paper graph"
    )

    # Required arguments
    build_parser.add_argument(
        "paper_id", type=str, help="Paper identifier (DOI, arXiv ID, or S2 ID)"
    )

    # Strategy selection
    build_parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        choices=["citation", "embedding", "hybrid"],
        default="citation",
        help="Graph building strategy (default: citation)",
    )

    # Common arguments
    build_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output PNG file path (auto-named if not specified)",
    )

    build_parser.add_argument(
        "--max-papers",
        "-p",
        type=int,
        default=40,
        help="Maximum papers to include (default: 40)",
    )

    build_parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=100,
        help="Layout iterations for quality (default: 100)",
    )

    build_parser.add_argument(
        "--dpi",
        "-d",
        type=int,
        default=150,
        help="Output image resolution (default: 150)",
    )

    build_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None = non-deterministic)",
    )

    # Citation strategy arguments
    citation_group = build_parser.add_argument_group("citation strategy options")
    citation_group.add_argument(
        "--max-citations",
        "-c",
        type=int,
        default=20,
        help="Maximum citing papers to fetch (default: 20)",
    )

    citation_group.add_argument(
        "--max-references",
        "-r",
        type=int,
        default=20,
        help="Maximum referenced papers to fetch (default: 20)",
    )

    citation_group.add_argument(
        "--similarity-threshold",
        "-t",
        type=float,
        default=0.2,
        help="Minimum similarity for edges (default: 0.2)",
    )

    citation_group.add_argument(
        "--no-references",
        action="store_true",
        help="Disable fetching reference lists (faster but no real bibliographic coupling)",
    )

    # Embedding strategy arguments
    embedding_group = build_parser.add_argument_group("embedding strategy options")
    embedding_group.add_argument(
        "--model",
        "-m",
        type=str,
        default="google/embeddinggemma-300m",
        help="Sentence transformer model name",
    )

    embedding_group.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="ArXiv dataset split (default: train = full ~117k papers)",
    )

    embedding_group.add_argument(
        "--corpus-size",
        type=int,
        default=None,
        help="Maximum papers to load from corpus (default: all in split)",
    )

    embedding_group.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=2,
        help="Top-k neighbors per node (default: 2)",
    )

    # Hybrid strategy arguments
    hybrid_group = build_parser.add_argument_group("hybrid strategy options")
    hybrid_group.add_argument(
        "--max-semantic",
        type=int,
        default=10,
        help="Maximum papers from semantic search (default: 10)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "build":
        try:
            # Build graph based on strategy
            logger.info(f"Building graph using {args.strategy} strategy...")

            if args.strategy == "citation":
                graph, seed_id = build_citation_graph(args)
            elif args.strategy == "embedding":
                graph, seed_id = build_embedding_graph(args)
            elif args.strategy == "hybrid":
                graph, seed_id = build_hybrid_graph(args)
            else:
                logger.error(f"Unknown strategy: {args.strategy}")
                sys.exit(1)

            # Determine output path
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = generate_output_path(graph, seed_id)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Visualize
            logger.info("Creating visualization...")
            visualize_graph(
                graph, seed_id, output_path, iterations=args.iterations, dpi=args.dpi
            )

            logger.info(f"✓ Graph saved to {output_path}")
            logger.info(
                f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}"
            )

        except Exception as e:
            logger.error(f"Failed to build graph: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
