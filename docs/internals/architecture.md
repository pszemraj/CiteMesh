# CiteMesh Architecture

The CLI orchestrates a consistent pipeline regardless of strategy (`recommendation`, `citation`, `embedding`, `hybrid`). This document describes major components, data flow, and extension points.

Related docs:

- CLI flags and command examples: [CLI Usage](https://github.com/pszemraj/CiteMesh/blob/main/docs/guides/cli.md)
- Cache layout and hydration details: [Caching & Data](https://github.com/pszemraj/CiteMesh/blob/main/docs/guides/caching.md)
- Environment variables: [Environment Variables](https://github.com/pszemraj/CiteMesh/blob/main/docs/reference/environment.md)
- Export/sidecar file contracts: [Output Artifacts](https://github.com/pszemraj/CiteMesh/blob/main/docs/reference/output-artifacts.md)
- Docs index: [Documentation](https://github.com/pszemraj/CiteMesh/blob/main/docs/README.md)

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
    ├── export.GraphExporter -> HTML / Plotly / Dashboard / JSON / GraphML
    └── cli sidecar writer -> *.config.json (rebuild params + run metadata)
```

Each graph node carries a shared attribute payload (`paper`, `title`, `year`, `authors`, `citation_count`, `is_seed`) so visualization and export layers remain strategy-agnostic.

## Module Overview

### `citemesh/cli.py`

- Defines `citemesh` entry point and command dispatch.
- Parses validated arguments and resolves output paths.
- Selects strategy implementations and triggers graph construction.
- Coordinates render/export steps, writes sidecar config artifacts, and passes run metadata downstream.

Command-line behavior is documented in [CLI Usage](https://github.com/pszemraj/CiteMesh/blob/main/docs/guides/cli.md).

### `citemesh/strategies/base.py`

- Provides the shared template method for graph construction.
- Defines extension hooks (`collect_papers`, `compute_similarity`).
- Handles edge selection/pruning flow used by concrete strategies.

### Strategy Implementations

| Strategy | Responsibilities | Highlights |
| --- | --- | --- |
| `citation` | Pull seed, references, and citations from Semantic Scholar | Similarity from bibliographic and metadata features |
| `embedding` | Hydrate/query quantized cache and compute semantic neighbors | Int8/binary cache-native retrieval, metadata-authoritative warm runs, capped edge pruning |
| `recommendation` | Use Semantic Scholar recommendations as primary neighborhood signal | Fast topical discovery path |
| `hybrid` | Merge citation and semantic candidates, then rerank | Seed-relevance rerank with semantic-only cap enforcement |

Strategies may emit collection summaries through `_set_collection_summary` for consistent logging.
Strategy behavior and selection guidance are documented in [Guides: Strategies](https://github.com/pszemraj/CiteMesh/blob/main/docs/guides/strategies.md).

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

- `GraphExporter` writes JSON, GraphML, Pyvis HTML, Plotly HTML, and dashboard HTML.
- Reuses computed layout and style values for cross-format consistency.
- Normalizes node attributes for serializer compatibility (for example GraphML-safe fields).

Artifact-level format details and sidecar schema are documented in
[Output Artifacts](https://github.com/pszemraj/CiteMesh/blob/main/docs/reference/output-artifacts.md).

### Caching Support

- `citemesh/data/cache.py` resolves user-scoped cache roots.
- `citemesh/data/embedding_cache.py` manages quantized SQLite/HDF5 embedding cache state (`int8` matrix, calibration ranges, optional binary index, hydration metadata).
- `citemesh/data/model_profiles.py` stores model-specific runtime profile metadata.

On-disk layout and invalidation behavior are documented in [Caching & Data](https://github.com/pszemraj/CiteMesh/blob/main/docs/guides/caching.md).

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

This structure keeps the user-facing workflow stable while allowing iteration on ranking, similarity, visualization, and data-source internals.
