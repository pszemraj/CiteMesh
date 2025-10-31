# CiteMesh Architecture

The CLI orchestrates a consistent pipeline regardless of the strategy you choose (citation, embedding, or hybrid). This document describes the major components, how data flows between them, and where to hook in new functionality.

## Execution Flow

```
CLI (citemesh/cli.py)
    ├── Parse arguments / resolve output paths & themes
    ├── Instantiate selected GraphBuilderStrategy
    └── Invoke build_graph(seed_id, **kwargs)
            ↓
GraphBuilderStrategy (base class)
    ├── collect_papers(seed_id, **kwargs)  → Dict[str, Paper]
    ├── compute_similarity(paper_a, paper_b) → float
    └── build_graph(..) → (networkx.Graph, seed_id)
            ↓
Visualization + Export
    ├── visualization.visualize_graph → PNG
    └── export.GraphExporter → HTML / JSON / GraphML
```

Every node added to the NetworkX graph carries the same attributes (`paper`, `title`, `year`, `authors`, `citation_count`, `is_seed`). This uniform payload allows the visualization and exporter layers to remain strategy-agnostic.

## Module Overview

### `citemesh/cli.py`

- Defines the `citemesh` console entry point.
- Adds strategy-specific arguments (e.g., `--max-citations`, `--top-k`, `--max-semantic`).
- Resolves output filenames for each requested export format.
- Builds a metadata dictionary (paper id, strategy, node/edge counts, timestamp) passed to both Matplotlib and HTML exporters.

### `citemesh/strategies/base.py`

- Implements the template method shared by all strategies.
- Handles reproducibility by seeding NumPy, Python `random`, and Torch when `random_seed` is provided.
- Provides `collect_papers` and `compute_similarity` hooks the concrete strategies override.

### Strategy Implementations

| Strategy    | Responsibilities                                               | Highlights                                                                         |
| ----------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `citation`  | Pulls seed, references, and citations from Semantic Scholar    | Uses temporal, citation-impact, and real bibliographic coupling scores             |
| `embedding` | Loads/streams HuggingFace ML-ArXiv corpus, computes embeddings | Persistent embedding cache (SQLite + HDF5), multi-factor similarity, top-k pruning |
| `hybrid`    | Starts with citation graph, enriches with semantic matches     | Adjusts weightings based on relationship type, caps edges per node                 |

Each strategy can surface helpful logging by calling `_set_collection_summary`, which the CLI prints after graph construction.

### `citemesh/models.py`

- `Paper` and `Author` dataclasses encapsulate metadata and validation.
- Provides helpers such as `.label`, `.first_author_surname`, `.category_overlap`, and `.shares_authors_with`.
- Makes node attributes both ergonomic (full object) and serialisable (primitive fields).

### Visualization (`citemesh/visualization.py`)

- Computes layouts (Kamada-Kawai with spring fallback), node sizes, colors, labels, and metadata placement.
- Uses theme-driven colors retrieved from `citemesh/themes.py`.
- Writes to PNG via Matplotlib, applying the selected theme's background and text colors.

### Themes (`citemesh/themes.py`)

- Declares immutable `Theme` objects for `light`, `dark`, `solarized`, plus an `auto` detector.
- Provides RGB interpolation for smooth gradients that both Matplotlib and HTML exporters share.

### Exporter (`citemesh/export.py`)

- `GraphExporter` emits:
  - JSON snapshots of nodes/edges/metadata
  - GraphML files for tools like Gephi or Cytoscape
  - Pyvis HTML (vis.js) with physics controls and detailed tooltips
  - Plotly HTML for dashboard embedding
- Reuses computed layouts, sizes, and colors for consistent styling across formats.
- Normalises node attributes for serialization (e.g., flattening author lists for GraphML).

### Caching Support

- `citemesh/cache_utils.py` picks a cross-platform cache directory (`~/.cache/citemesh`, `%LOCALAPPDATA%\CiteMesh`, etc.) and honours `CITEMESH_CACHE_DIR`.
- `citemesh/embedding_cache.py` stores embeddings in SQLite (metadata) + HDF5 (vectors), keyed by model hash and content checksum to avoid stale results.
- Joblib caches for HuggingFace corpora point to the same cache root, keeping the repository workspace clean.

## External Dependencies

- **Semantic Scholar API**: citation and hybrid strategies fetch paper metadata, references, and citations.
- **HuggingFace Datasets**: embedding strategy loads the ML ArXiv corpus (offline after first download).
- **SentenceTransformers**: embedding/hybrid strategies load configurable models via `SentenceTransformer`.

## Output Handling

- Default target directory is `out/`; the CLI auto-sanitises seed titles for filenames.
- When multiple export formats are requested, filenames receive distinct extensions without overwriting each other.
- Integration tests point exports to temporary directories to avoid polluting working graphs.

## Extensibility

- **New strategy**: subclass `GraphBuilderStrategy`, register it in the CLI, and implement `collect_papers` / `compute_similarity`.
- **New export format**: extend `GraphExporter` and add a case in the CLI export switchboard.
- **Custom themes**: add entries to `THEMES`; they become immediately available through `--theme`.
- **Alternative corpora**: adapt `EmbeddingGraphBuilder._load_corpus` and continue using `EmbeddingCache` for persistence.

This modular structure keeps the user workflow stable while making space for experimentation with similarity metrics, visual styles, or data sources.
