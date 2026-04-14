# CiteMesh Architecture

The CLI orchestrates the same high-level pipeline for `recommendation`, `citation`, `embedding`, and `hybrid`, then hands strategy-agnostic graph data to the visualization and export layers.

Related docs:

- CLI flags and command examples: [CLI Usage](../guides/cli.md)
- Cache layout and hydration details: [Caching & Data](../guides/caching.md)
- Environment variables: [Environment Variables](../reference/environment.md)
- Export/sidecar file contracts: [Output Artifacts](../reference/output-artifacts.md)

## Execution Flow

```text
CLI (citemesh/cli.py)
    ├── Parse arguments and resolve output/export targets
    ├── Instantiate selected GraphBuilderStrategy
    └── Invoke build_graph(seed_id, **kwargs)
            ↓
GraphBuilderStrategy (base class)
    ├── collect_papers(seed_id, **kwargs) -> Dict[str, Paper]
    ├── compute_similarity(paper_a, paper_b) -> float
    └── build_graph(..) -> (networkx.Graph, seed_id)
            ↓
Visualization + Export
    ├── visualization.visualize_graph -> PNG
    ├── export.GraphExporter -> HTML / Plotly / Dashboard / JSON / CSV / BibTeX / GraphML
    └── cli sidecar writer -> *.config.json (rebuild params + run metadata)
```

Each graph node carries a shared attribute payload (`paper`, `title`, `year`, `authors`, `citation_count`, `venue`, `arxiv_id`, `doi`, `is_seed`) so visualization and export layers remain strategy-agnostic.

## Module Overview

### `citemesh/cli.py`

- Defines `citemesh` entry point and command dispatch.
- Parses validated arguments and resolves output paths.
- Selects strategy implementations and triggers graph construction.
- Coordinates render/export steps, writes sidecar config artifacts, and passes run metadata downstream.

Command-line behavior is documented in [CLI Usage](../guides/cli.md).

### `citemesh/strategies/base.py`

- Provides the shared template method for graph construction.
- Defines extension hooks (`collect_papers`, `compute_similarity`).
- Handles edge selection/pruning flow used by concrete strategies.

### Strategy Implementations

- `citation.py`, `recommendation.py`, `embedding.py`, and `hybrid.py` implement
  `GraphBuilderStrategy`.
- Strategies may emit collection summaries through `_set_collection_summary` for
  consistent logging.
- Selection guidance, tradeoffs, and user-facing strategy behavior are covered in
  [Guides: Strategies](../guides/strategies.md).

### `citemesh/core/models.py`

- `Paper` and `Author` dataclasses encapsulate validated metadata.
- Utility helpers support label generation and overlap checks.
- Model payloads are designed for both graph operations and export serialization.

### Visualization (`citemesh/visualization/render.py`)

- Computes layouts, node sizes/colors, labels, and metadata overlays.
- Applies selected theme values from `themes.py`.
- Produces static PNG output via Matplotlib.

### Themes (`citemesh/visualization/themes.py`)

- Defines immutable theme objects (`light`, `dark`, `solarized`, `auto`).
- Provides shared interpolation utilities reused by static and interactive outputs.

### Exporter (`citemesh/visualization/export.py`)

- `GraphExporter` writes interactive and structured output formats from one graph object.
- Reuses computed layout and style values for cross-format consistency.
- Normalizes node attributes for serializer compatibility (for example GraphML-safe fields).
- Artifact-level format details and sidecar schema are documented in
  [Output Artifacts](../reference/output-artifacts.md).

### Caching Support

- `citemesh/data/cache.py` resolves user-scoped cache roots.
- `citemesh/data/embedding_cache.py` manages quantized SQLite/HDF5 embedding cache state (`int8` matrix, calibration ranges, optional binary index, hydration metadata).
- `citemesh/data/model_profiles.py` stores model-specific runtime profile metadata.

On-disk layout and invalidation behavior are documented in [Caching & Data](../guides/caching.md).

### Service Client (`citemesh/services/semantic_scholar.py`)

- Wraps Semantic Scholar API calls with retries and rate-limit handling.
- Handles reference-list caching integration.
- Exposes `get_client()` for strategy use.

## External Dependencies

- **Semantic Scholar API** for citation/recommendation data.
- **HuggingFace Datasets** for embedding corpus sources.
- **SentenceTransformers** for semantic embeddings.
- **Optional `.[viz]` extras** for interactive HTML exporters.

## Extensibility

- **Add a strategy**: subclass `GraphBuilderStrategy`, implement hooks, register in CLI dispatch.
- **Add an export format**: extend `GraphExporter` and wire CLI export routing.
- **Add themes**: extend `THEMES` definitions and renderer/export color lookups.
- **Add corpus source**: adapt embedding-corpus loading while preserving shared cache/graph contracts.

The user-facing workflow stays stable while ranking, similarity, visualization,
and data-source internals evolve behind the strategy/export interfaces.
