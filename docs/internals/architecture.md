# CiteMesh Architecture

The CLI orchestrates a consistent pipeline regardless of the strategy you choose (`recommendation`, `citation`, `embedding`, or `hybrid`). This document describes the major components, how data flows between them, and where to hook in new functionality.

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
    └── export.GraphExporter → HTML / Plotly / JSON / GraphML
```

Every node added to the NetworkX graph carries the same attributes (`paper`, `title`, `year`, `authors`, `citation_count`, `is_seed`). This uniform payload allows the visualization and exporter layers to remain strategy-agnostic.

## Module Overview

### `citemesh/cli.py`

- Defines the `citemesh` console entry point.
- Parses and validates all CLI arguments.
- Resolves output filenames for requested export formats.
- Computes one shared layout per build run (optionally seeded via `--seed`) and reuses it across PNG/HTML/Plotly outputs.
- Builds a metadata dictionary (paper id, strategy, node/edge counts, timestamp) passed to both Matplotlib and HTML exporters.

Authoritative flag/identifier behavior lives in [CLI Usage](../guides/cli.md).

### `citemesh/strategies/base.py`

- Implements the template method shared by all strategies.
- Provides `collect_papers` and `compute_similarity` hooks the concrete strategies override.

### Strategy Implementations

| Strategy    | Responsibilities                                               | Highlights                                                                         |
| ----------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `citation`  | Pulls seed, references, and citations from Semantic Scholar    | Uses abstract similarity, temporal signal, citation impact, and bibliographic coupling |
| `embedding` | Loads/streams the HuggingFace arXiv metadata snapshot, computes embeddings | Persistent embedding cache (SQLite + HDF5), multi-factor similarity, top-k pruning |
| `recommendation` | Uses Semantic Scholar recommendations as the primary neighborhood signal  | Fast topical discovery, deterministic thresholded edge filtering                  |
| `hybrid`    | Starts with citation graph, enriches with semantic matches     | Adjusts weightings based on relationship type, caps edges per node                 |

Each strategy can surface helpful logging by calling `_set_collection_summary`, which the CLI prints after graph construction.

### `citemesh/core/models.py`

- `Paper` and `Author` dataclasses encapsulate metadata and validation.
- Provides helpers such as `.label`, `.first_author_surname`, `.category_overlap`, and `.shares_authors_with`.
- Makes node attributes both ergonomic (full object) and serialisable (primitive fields).

### Visualization (`citemesh/visualization/render.py`)

- Computes layouts (Kamada-Kawai with spring fallback), node sizes, colors, labels, and metadata placement.
- Uses theme-driven colors retrieved from `citemesh/visualization/themes.py`.
- Writes to PNG via Matplotlib, applying the selected theme's background and text colors.

### Themes (`citemesh/visualization/themes.py`)

- Declares immutable `Theme` objects for `light`, `dark`, `solarized`, plus an `auto` detector.
- Provides RGB interpolation for smooth gradients that both Matplotlib and HTML exporters share.

### Exporter (`citemesh/visualization/export.py`)

- `GraphExporter` emits:
  - JSON snapshots of nodes/edges/metadata
  - GraphML files for tools like Gephi or Cytoscape
  - Pyvis HTML (vis.js) with physics controls and detailed tooltips
  - Plotly HTML for dashboard embedding
- Reuses computed layouts, sizes, and colors for consistent styling across formats.
- Normalises node attributes for serialization (e.g., flattening author lists for GraphML).

### Caching Support

- `citemesh/data/cache.py` and `citemesh/data/embedding_cache.py` configure user-scoped storage for persistent graph artifacts and embedding vectors.
- `citemesh/data/model_profiles.py` captures per-model metadata (for example, prompts and dtype hints) consumed by embedding strategies.
- See [Caching Guide](../guides/caching.md) for full cache layout, invalidation rules, and cleanup commands.

### Service Clients (`citemesh/services/semantic_scholar.py`)

- Wraps the Semantic Scholar API with retries, rate limiting, and local caching of reference lists.
- Exposes `get_client()` for strategies and re-exports the client in `citemesh.services` for convenience.

## External Dependencies

- **Semantic Scholar API**: citation and hybrid strategies fetch paper metadata, references, and citations.
- **HuggingFace Datasets**: embedding strategy loads the arXiv metadata snapshot corpus (offline after first download).
- **SentenceTransformers**: embedding/hybrid strategies load configurable models via `SentenceTransformer`.

## Output Handling

Output naming and extension-resolution rules are defined in [CLI Usage](../guides/cli.md). This architecture doc intentionally stays at component-level behavior.

## Extensibility

- **New strategy**: subclass `GraphBuilderStrategy`, register it in the CLI, and implement `collect_papers` / `compute_similarity`.
- **New export format**: extend `GraphExporter` and add a case in the CLI export switchboard.
- **Custom themes**: add entries to `THEMES`; they become immediately available through `--theme`.
- **Alternative corpora**: adapt `EmbeddingGraphBuilder._load_corpus` and continue using `EmbeddingCache` for persistence.

This modular structure keeps the user workflow stable while making space for experimentation with similarity metrics, visual styles, or data sources.
