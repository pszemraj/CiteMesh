#!/usr/bin/env python3
"""
CiteMesh: Unified CLI for CiteMesh visualizations.

This is the main entry point for the refactored CiteMesh package,
providing a single interface to all graph building strategies.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import networkx as nx
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from citemesh.services import get_client
from citemesh.services.semantic_scholar import normalize_paper_id
from citemesh.strategies.citation import CitationGraphBuilder
from citemesh.strategies.embedding import EmbeddingGraphBuilder
from citemesh.strategies.hybrid import HybridGraphBuilder
from citemesh.strategies.recommendation import RecommendationGraphBuilder
from citemesh.visualization import (
    GraphExporter,
    generate_output_path,
    visualize_graph,
)

console = Console(stderr=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=console,
            show_time=False,
            show_path=False,
            rich_tracebacks=False,
            markup=True,
        )
    ],
)
logger = logging.getLogger(__name__)

EXPORT_FORMATS = ("png", "html", "plotly", "json", "graphml")
EXPORT_EXTENSIONS: Dict[str, str] = {
    "png": ".png",
    "html": ".html",
    "plotly": ".plotly.html",
    "json": ".json",
    "graphml": ".graphml",
}
KNOWN_EXPORT_SUFFIXES: List[str] = sorted(
    EXPORT_EXTENSIONS.values(), key=len, reverse=True
)


def resolve_output_paths(
    base_output_path: Path, selected_formats: List[str], explicit_output: bool
) -> Dict[str, Path]:
    """
    Resolve final output paths for selected export formats.

    :param Path base_output_path: Path provided by the user or auto-generated filename.
    :param List[str] selected_formats: Export formats selected for this run.
    :param bool explicit_output: True when the user provided ``--output``.
    :return Dict[str, Path]: Mapping of export format -> resolved output path.
    """
    base_str = str(base_output_path)
    matched_suffix = next(
        (ext for ext in KNOWN_EXPORT_SUFFIXES if base_str.lower().endswith(ext)),
        None,
    )

    output_paths: Dict[str, Path] = {}
    if explicit_output and len(selected_formats) == 1:
        fmt = selected_formats[0]
        desired_ext = EXPORT_EXTENSIONS[fmt]

        if matched_suffix == desired_ext:
            output_paths[fmt] = base_output_path
            return output_paths

        if matched_suffix:
            output_paths[fmt] = Path(base_str[: -len(matched_suffix)] + desired_ext)
            return output_paths

        output_paths[fmt] = Path(base_str + desired_ext)
        return output_paths

    if matched_suffix:
        output_base = base_str[: -len(matched_suffix)]
    else:
        output_base = base_str

    for fmt in selected_formats:
        output_paths[fmt] = Path(output_base + EXPORT_EXTENSIONS[fmt])

    return output_paths


def canonicalize_paper_id_for_metadata(paper_id: str) -> str:
    """
    Best-effort canonical paper ID for output metadata display.

    :param str paper_id: Raw CLI paper identifier.
    :return str: Canonicalized identifier when possible; otherwise original input.
    """
    try:
        return normalize_paper_id(paper_id)
    except ValueError:
        return paper_id


def build_citation_graph(args: argparse.Namespace) -> tuple[nx.Graph, str]:
    """
    Build graph using citation strategy.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return tuple[nx.Graph, str]: Tuple of graph and normalized seed paper ID.
    """
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


def build_recommendation_graph(args: argparse.Namespace) -> tuple[nx.Graph, str]:
    """
    Build graph using recommendation strategy.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return tuple[nx.Graph, str]: Tuple of graph and normalized seed paper ID.
    """
    builder = RecommendationGraphBuilder(
        max_papers=args.max_papers,
        fetch_references=not args.no_references,
        similarity_threshold=args.similarity_threshold,
        random_seed=args.seed,
    )

    graph, seed_id = builder.build_graph(args.paper_id)
    return graph, seed_id


def build_embedding_graph(args: argparse.Namespace) -> tuple[nx.Graph, str]:
    """
    Build graph using embedding strategy.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return tuple[nx.Graph, str]: Tuple of graph and normalized seed paper ID.
    """
    builder = EmbeddingGraphBuilder(
        max_papers=args.max_papers,
        model_name=args.model,
        dataset_split=args.dataset_split,
        corpus_size=args.corpus_size,
        top_k=args.top_k,
        random_seed=args.seed,
        use_streaming=args.streaming,
    )

    graph, seed_id = builder.build_graph(args.paper_id)
    return graph, seed_id


def build_hybrid_graph(args: argparse.Namespace) -> tuple[nx.Graph, str]:
    """
    Build graph using hybrid strategy.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return tuple[nx.Graph, str]: Tuple of graph and normalized seed paper ID.
    """
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


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CiteMesh: Create citation graph visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Citation-based graph (fast, uses S2 API)
  citemesh build "arxiv:1706.03762" --strategy citation

  # Recommendation graph (semantic-aware by default)
  citemesh build "arxiv:1706.03762"

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
        choices=["recommendation", "citation", "embedding", "hybrid"],
        default="recommendation",
        help="Graph building strategy (default: recommendation)",
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
        "--export",
        "-e",
        choices=["png", "html", "plotly", "json", "graphml", "all"],
        default="png",
        help="Export format (default: png)",
    )

    build_parser.add_argument(
        "--theme",
        choices=["light", "dark", "solarized", "auto"],
        default="light",
        help="Visualization theme to use",
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

    embedding_group.add_argument(
        "--streaming",
        action="store_true",
        help="Stream HuggingFace dataset instead of loading it into memory",
    )

    # Hybrid strategy arguments
    hybrid_group = build_parser.add_argument_group("hybrid strategy options")
    hybrid_group.add_argument(
        "--max-semantic",
        type=int,
        default=10,
        help="Maximum papers from semantic search (default: 10)",
    )

    # Search subcommand
    search_parser = subparsers.add_parser(
        "search", help="Search papers by title or keyword"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=10,
        help="Maximum results (default: 10)",
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
            elif args.strategy == "recommendation":
                graph, seed_id = build_recommendation_graph(args)
            elif args.strategy == "embedding":
                graph, seed_id = build_embedding_graph(args)
            elif args.strategy == "hybrid":
                graph, seed_id = build_hybrid_graph(args)
            else:
                logger.error(f"Unknown strategy: {args.strategy}")
                sys.exit(1)

            # Determine output paths
            if args.output:
                base_output_path = Path(args.output)
            else:
                base_output_path = generate_output_path(
                    graph, seed_id, strategy=args.strategy
                )

            output_dir = base_output_path.parent
            if output_dir and not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)

            selected_formats = (
                list(EXPORT_FORMATS) if args.export == "all" else [args.export]
            )
            output_paths = resolve_output_paths(
                base_output_path=base_output_path,
                selected_formats=selected_formats,
                explicit_output=bool(args.output),
            )

            # Visualize / export
            logger.info("Creating visualization...")
            metadata = {
                "paper_id": canonicalize_paper_id_for_metadata(args.paper_id),
                "seed_id": seed_id,
                "strategy": args.strategy,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "theme": args.theme,
            }

            exporter = GraphExporter(
                graph,
                seed_id,
                metadata=metadata,
                theme_name=args.theme,
            )

            if "png" in output_paths:
                visualize_graph(
                    graph,
                    seed_id,
                    output_paths["png"],
                    iterations=args.iterations,
                    dpi=args.dpi,
                    metadata=metadata,
                    theme_name=args.theme,
                )
                logger.info(f"✓ PNG saved to {output_paths['png']}")

            if "html" in output_paths:
                exporter.to_interactive_html(output_paths["html"], theme=args.theme)
                logger.info(f"✓ Interactive HTML saved to {output_paths['html']}")

            if "plotly" in output_paths:
                exporter.to_plotly_html(output_paths["plotly"], theme=args.theme)
                logger.info(f"✓ Plotly HTML saved to {output_paths['plotly']}")

            if "json" in output_paths:
                exporter.to_json(output_paths["json"])
                logger.info(f"✓ Graph JSON saved to {output_paths['json']}")

            if "graphml" in output_paths:
                exporter.to_graphml(output_paths["graphml"])
                logger.info(f"✓ GraphML saved to {output_paths['graphml']}")

            logger.info(
                f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}"
            )

        except Exception as e:
            message = f"Failed to build graph: {e}"
            logger.error(message)
            print(message, file=sys.stderr)
            # Only show full traceback in debug mode
            if logging.getLogger().level == logging.DEBUG:
                import traceback

                traceback.print_exc()
            sys.exit(1)
    elif args.command == "search":
        try:
            client = get_client()
            logger.info(f"Searching for: {args.query}")
            results = client.search_papers(args.query, limit=args.limit)

            if not results:
                logger.error("No results found.")
                sys.exit(1)

            table = Table(title=f"Search results for '{args.query}'")
            table.add_column("#", style="dim", width=3)
            table.add_column("ID", style="cyan", max_width=20)
            table.add_column("Title", max_width=50)
            table.add_column("Year", justify="right", width=6)
            table.add_column("Citations", justify="right", width=10)
            table.add_column("Authors", max_width=30)

            for i, paper in enumerate(results, 1):
                authors_str = ", ".join(a.name for a in paper.authors[:2])
                if len(paper.authors) > 2:
                    authors_str += " et al."

                table.add_row(
                    str(i),
                    paper.paper_id[:18] + "..."
                    if len(paper.paper_id) > 18
                    else paper.paper_id,
                    paper.title[:48] + "..." if len(paper.title) > 48 else paper.title,
                    str(paper.year) if paper.year is not None else "",
                    f"{paper.citation_count:,}",
                    authors_str,
                )

            console.print(table)
            console.print(
                "\n[dim]Use the paper ID with:[/dim] "
                'citemesh build "<ID>" --strategy recommendation'
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
