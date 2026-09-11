"""Output-path and filename derivation for graph artifacts.

Pure path/string logic, deliberately free of matplotlib and numpy so callers
that only need to resolve where artifacts go (the CLI's output resolver) do not
pay for the rendering stack. :mod:`citemesh.visualization.render` re-exports
these names so ``render.generate_output_path`` stays a valid import path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

MAX_TITLE_CHARS = 40

__all__ = ["MAX_TITLE_CHARS", "generate_output_path"]


def _filename_safe(text: str, max_chars: int = MAX_TITLE_CHARS) -> str:
    """Create a filesystem-safe slug from input text.

    :param str text: Raw text value.
    :param int max_chars: Maximum slug length.
    :return str: Safe slug using lowercase alnum/hyphen tokens.
    """
    normalized = text.lower()
    normalized = "".join(c if c.isalnum() or c in " -" else "" for c in normalized)
    slug = "-".join(normalized.split())[:max_chars].strip("-")
    return slug or "graph"


def _seed_suffix(seed_id: str, length: int = 8) -> str:
    """Build a short, stable suffix from the seed identifier.

    :param str seed_id: Seed paper identifier.
    :param int length: Number of digest characters to keep.
    :return str: Stable hex suffix used in output directory naming.
    """
    return hashlib.sha256(seed_id.encode("utf-8")).hexdigest()[:length]


def _output_dir_name(label: str, seed_id: str, max_chars: int = MAX_TITLE_CHARS) -> str:
    """Build output directory name with stable seed suffix under truncation.

    :param str label: Paper title or fallback seed ID used for the slug prefix.
    :param str seed_id: Canonical seed identifier used for stable hash suffix.
    :param int max_chars: Maximum total directory-name length.
    :return str: Filesystem-safe directory name containing a slug and hash suffix.
    """
    suffix = f"-{_seed_suffix(seed_id)}"
    label_budget = max_chars - len(suffix)
    label_budget = max(1, label_budget)
    return f"{_filename_safe(label, max_chars=label_budget)}{suffix}"


def generate_output_path(
    graph: nx.Graph, seed_id: str, output_dir: Path = Path("out"), strategy: str = ""
) -> Path:
    """
    Generate a title-based output path, reusing an existing seed directory.

    :param nx.Graph graph: NetworkX graph containing the seed paper title.
    :param str seed_id: ID of seed paper
    :param Path output_dir: Output directory
    :param str strategy: Optional strategy suffix used in filename.
    :return Path: Path object for output file
    """
    existing_dirs = sorted(
        path for path in output_dir.glob(f"*-{_seed_suffix(seed_id)}") if path.is_dir()
    )
    if existing_dirs:
        paper_dir = existing_dirs[0]
    else:
        seed_attrs = graph.nodes[seed_id] if seed_id in graph else {}
        title = seed_attrs.get("title") or seed_id
        paper_dir = output_dir / _output_dir_name(label=title, seed_id=seed_id)
    paper_dir.mkdir(parents=True, exist_ok=True)

    basename = _filename_safe(strategy, max_chars=32) if strategy else "graph"
    return paper_dir / f"{basename}.png"
