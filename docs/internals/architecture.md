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
    │       └── strategies/candidates.py: candidate-source fetch, availability
    │           policy, identity reconciliation, and pool budgets
    ├── compute_similarity(paper_a, paper_b) -> float
    └── build_graph(..) -> (networkx.Graph, seed_id)
            ↓
Visualization + Export
    ├── visualization.visualize_graph -> PNG
    ├── export.GraphExporter -> HTML / Plotly / Dashboard / JSON / CSV / BibTeX / GraphML
    ├── dashboard collection writer -> dashboard.citemesh.json + dashboard.html
    └── per-seed exports -> <slug>-<hash>/*.json + *.config.json (also for dashboards)
```

Each graph node carries a shared attribute payload (`paper`, `title`, `year`, `authors`, `citation_count`, `venue`, `arxiv_id`, `doi`, `is_seed`) so visualization and export layers remain strategy-agnostic.

## Module Overview

### `citemesh/cli.py`

- Defines `citemesh` entry point and command dispatch.
- Parses validated arguments and resolves output paths.
- Selects strategy implementations and triggers graph construction.
- Coordinates render/export steps, writes sidecar config artifacts, upserts dashboard collection packages, and passes run metadata downstream.

Command-line behavior is documented in [CLI Usage](../guides/cli.md).

### `citemesh/strategies/base.py`

- Provides the shared template method for graph construction.
- Defines extension hooks (`collect_papers`, `compute_similarity`).
- Handles edge selection/pruning flow used by concrete strategies.

### Strategy Implementations

- `citation.py`, `recommendation.py`, `embedding.py`, and `hybrid.py` implement `GraphBuilderStrategy`.
- Strategies may emit collection summaries through `_set_collection_summary` for consistent logging.
- Selection guidance, tradeoffs, and user-facing strategy behavior are covered in [Guides: Strategies](../guides/strategies.md).

### `citemesh/strategies/candidates.py`

- Shared candidate-acquisition layer used by every strategy: `fetch_candidate_source` tracks per-source outcomes (`complete`/`empty`/`unavailable`), and `require_available_candidate_source` implements the partial-outage-versus-fail policy recorded in export metadata.
- `IdentityRegistry` and `reconcile_paper_identity` de-duplicate papers across requested IDs and Semantic Scholar/arXiv/DOI aliases; `merge_seed_relation` folds seed relations when identities merge.
- `fetch_candidate_pool` implements the reference/citation/recommendation budget split and the free-text `query:` seed proxy.

### `citemesh/similarity.py` and `citemesh/dashboard_contracts.py`

- `similarity.py` provides `AbstractSimilarityIndex`, the TF-IDF scorer behind citation and recommendation topical similarity.
- `dashboard_contracts.py` holds the `kind`/`schema_version` identities shared by the export producers and the dashboard viewer.

### `citemesh/core/models.py`

- `Paper` and `Author` dataclasses encapsulate validated metadata.
- Utility helpers support label generation and overlap checks.
- Model payloads are designed for both graph operations and export serialization.

### `citemesh/core/user_config.py`

- Loads, validates, and rewrites the persistent `config.toml` at the cache root.
- Whitelists `[defaults]` build-flag keys and `[api] s2_api_key` with per-key casters; invalid entries are ignored with warnings so a bad config never blocks CLI usage.
- The CLI applies these values after argument parsing; precedence and supported
  keys are described in [User Configuration](../guides/configuration.md).

### Visualization (`citemesh/visualization/render.py`)

- Computes layouts, node sizes/colors, labels, and metadata overlays.
- Applies selected theme values from `themes.py`.
- Produces static PNG output via Matplotlib.

### Themes (`citemesh/visualization/themes.py`)

- Defines the immutable `light`, `dark`, and `solarized` palettes.
- Resolves `auto` from environment and host appearance signals.

### Exporter (`citemesh/visualization/export.py`)

- `GraphExporter` writes interactive and structured output formats from one graph object.
- Reuses computed layout and style values for cross-format consistency.
- Normalizes node attributes for serializer compatibility (for example GraphML-safe fields).
- `visualization/years.py` centralizes publication-year coercion (including the optional-bounds split behind `meta.year_range: null`); `visualization/ordering.py` provides the deterministic node/edge ordering every export relies on.
- Artifact-level format details and sidecar schema are documented in [Output Artifacts](../reference/output-artifacts.md).

### Dashboard Collections

- `citemesh/cli.py` validates, locks, and atomically upserts collection packages.
- `GraphExporter` embeds the selected collection snapshot in the reusable viewer.
- Every collection build also writes the current seed's graph JSON and build
  sidecar to its own output directory, independently of the shared package.

File placement, schemas, migration, browser imports, and standalone-dashboard
behavior are described in [Output Artifacts](../reference/output-artifacts.md).

### Caching Support

- `citemesh/data/cache.py` resolves user-scoped cache roots.
- `citemesh/data/embedding_cache.py` manages SQLite metadata and HDF5 embedding
  datasets.
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
- **`.[recommended]` extra** for the primary embedding + interactive-export runtime bundle.
- **Optional `.[viz]` extra** when a base/dev install only needs interactive HTML exporters.

## Extensibility

- **Add a strategy**: subclass `GraphBuilderStrategy`, implement hooks, register in CLI dispatch.
- **Add an export format**: extend `GraphExporter` and wire CLI export routing.
- **Add themes**: extend `THEMES` definitions and renderer/export color lookups.
- **Add corpus source**: adapt embedding-corpus loading while preserving shared cache/graph contracts.

The user-facing workflow stays stable while ranking, similarity, visualization, and data-source internals evolve behind the strategy/export interfaces.
