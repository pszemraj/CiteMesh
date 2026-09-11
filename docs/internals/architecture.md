# CiteMesh Architecture

Package structure and the conventions that hold it together. For what the pipeline actually *does* at each stage, read [How CiteMesh builds a graph](../guides/how-it-works.md) instead — this page is about where code lives and how to extend it.

## Execution flow

```text
CLI (src/citemesh/cli/)
    ├── Parse arguments, apply config.toml defaults, resolve output/export targets
    ├── Instantiate the selected GraphBuilderStrategy
    └── Invoke build_graph(seed_id, **kwargs)
            ↓
GraphBuilderStrategy (strategies/base.py)
    ├── collect_papers(seed_id, **kwargs) -> Dict[str, Paper]
    │       └── strategies/candidates.py: source fetch, availability policy,
    │           identity reconciliation, pool budgets
    ├── prepare_graph_scoring(papers)          # build the scoring space
    ├── compute_similarity(paper_a, paper_b)   # per-pair score
    └── returns (networkx.Graph, resolved_seed_id)
            ↓
Visualization + Export (src/citemesh/visualization/)
    ├── render.compute_layout -> shared geometry
    ├── render.visualize_graph -> PNG
    ├── export.GraphExporter -> HTML / Plotly / Dashboard / JSON / CSV / BibTeX / GraphML
    └── dashboard/package.py -> dashboard.citemesh.json + dashboard.html
```

Each graph node carries a shared attribute payload (`paper`, `title`, `year`, `authors`, `citation_count`, `venue`, `arxiv_id`, `doi`, `is_seed`), so the visualization and export layers stay strategy-agnostic.

## Package map

### `core/` — data model and constants, no I/O

| Module | Contents |
| --- | --- |
| `config.py` | the in-code default dataclasses and their singletons: `TemporalConfig`, `EmbeddingSimilarityConfig`, `EmbeddingStorageConfig`, `HybridSimilarityConfig`, `VisualizationConfig`, `APIConfig`. Weight sets validate on import. |
| `models.py` | `Paper` and `Author`, plus the overlap helpers (`reference_overlap`, `category_overlap`, `shares_authors_with`) the scorers rely on. |
| `paper_fields.py` | tolerant coercion of venue, author, and category fields out of inconsistent upstream payloads. |
| `paper_ids.py` | identifier normalization and alias derivation (`normalize_paper_id`, `recognize_arxiv_identifier`, `paper_identifier_aliases`, `external_ids_from_canonical_paper_id`). |
| `values.py`, `validation.py` | shared value coercion and input validation primitives. |

### `data/` — persistence

| Module | Contents |
| --- | --- |
| `cache.py` | cache-root resolution across platforms, atomic text/JSON writers, the `atomic_output_path` context manager, and the cache-root `ReadWriteLock` coordination. |
| `user_config.py` | loads, validates, and rewrites `config.toml`. Whitelists `[defaults]` build-flag keys and `[api] s2_api_key` with per-key casters; invalid entries are warned and ignored so a bad config never blocks the CLI. |
| `model_profiles.py` | the embedding model profile registry: the EmbeddingGemma profile, its three prompt formatters, truncate-dim policy, attention and compile eligibility, and the default fallback chain. |
| `embedding_cache/` | package. `store.py` composes `EmbeddingCache` from four mixins — `ingest` (write pipeline), `layout` (SQLite/HDF5 schema and the runtime contract), `recovery` (locks, journal, truncation), `search` (scoring kernels) — over `constants`, `sql`, `models`, `quantization`. See [Embedding Cache Internals](embedding-cache.md). |

### `services/semantic_scholar/` — the only network boundary

`errors.py` (failure taxonomy and the per-capability `_FailureDomain` budgets) · `retry.py` (Tenacity backoff policy and `Retry-After` handling) · `disk_cache.py` (persisted paper-metadata and reference-ID caches) · `payloads.py` (response parsing into `Paper`) · `endpoints.py` (one method per S2 capability) · `client.py` (transport, rate limiting, `candidate_operation_scope`, `get_client`).

### `strategies/` — candidate acquisition and scoring

| Module | Contents |
| --- | --- |
| `base.py` | `GraphBuilderStrategy`: the template method, the `collect_papers` / `compute_similarity` / `should_create_edge` / `prepare_graph_scoring` hooks, the shared temporal / citation / bibliographic scorers, `deterministic_sort_key`, and edge capping (`select_capped_undirected_edges`, `build_capped_undirected_graph`). |
| `candidates.py` | the shared acquisition layer: pool budgets, `fetch_candidate_source` and its `complete`/`empty`/`unavailable` vocabulary, `require_available_candidate_source`, `IdentityRegistry`, `reconcile_paper_identity`, `merge_seed_relation`, `scope_candidate_collection`. |
| `similarity.py` | `AbstractSimilarityIndex`, the TF-IDF scorer behind citation and recommendation topical similarity. |
| `citation.py`, `recommendation.py`, `hybrid.py` | concrete strategies. |
| `embedding/` | package: `deps` (lazy optional-dependency guards) · `runtime` (device resolution, backend probes) · `model_runtime` (load, fallback chain, precision validation, TF32 and compile guards) · `precision` (`_PrecisionEncodeProxy`) · `text` (`EmbeddingTask`, formatters) · `records` (cached-metadata ↔ `Paper` conversion, `_query_seed_id`) · `config` (encode constants) · `fingerprint` (artifact and formatter digests) · `hydration` (corpus selection, calibration prepass, resume) · `builder` (`EmbeddingGraphBuilder`). |

### `visualization/` — layout, render, export

| Module | Contents |
| --- | --- |
| `render.py` | `compute_layout` and its whole chain (community detection, spreading, packing, orientation, normalization), `compute_node_sizes`, `visualize_graph` (Matplotlib PNG), `generate_output_path`. |
| `themes.py` | the immutable `light`/`dark`/`solarized` palettes and `auto` resolution from environment and host appearance. |
| `years.py`, `ordering.py` | publication-year coercion and the deterministic node/edge ordering every export depends on. |
| `export/` | package: `__init__` (`GraphExporter` and the `to_*` writers) · `nodes` (payload enrichment, seed-relevance PageRank) · `geometry` (layout/color/label primitives shared with the dashboard) · `plotly_figure` (deterministic Plotly figure assembly) · `links`, `keys`, `loaders`, `bibtex`, `graphml`, `csv_`. |
| `dashboard/` | `contracts.py` (`kind` / `schema_version` identities), `payload.py` (collection bundle assembly), `package.py` (locking, staging, upsert, rollback), `assets/{template.html,dashboard.css,dashboard.js}`. |

### `cli/` and top-level modules

`cli/__init__.py` (entry point and dispatch) · `parser.py` · `console.py` (Rich console and logging) · `build_options.py` and `build_contract.py` (strategy-scoped option validation and builder selection) · `outputs.py` (output-path resolution) · `graph_config.py` (the `*.config.json` sidecar) · `cache_ops.py` · `commands/{build,search,view,config,cache}.py`.

Top-level: `progress.py` (phase progress reporting), `_runtime.py` (process-level runtime setup), `text_batching.py` (length-bucketed encode batching and `l2_normalize_embeddings`), `_version.py`.

## Rules

**Dependency direction is one-way**: `core` → `data` → `services` → `strategies` → `visualization` → `cli`. A module never imports from a layer to its right. In particular `visualization/` never imports `cli` — the exporters must be usable without an argument parser, and the CLI is what wires paths and options into them.

**Optional dependencies are lazily imported.** torch, sentence-transformers, datasets, plotly, and pyvis must never be imported at module scope in a path the base CLI can reach. A bare `pip install citemesh` has to run `citemesh build --strategy recommendation` and `citemesh --help` with none of them installed. Guards live in `strategies/embedding/deps.py` and `visualization/export/loaders.py`; follow those patterns rather than adding a top-level `import torch`.

**The test suite is white-box.** Patch the name where it is *used* — the binding in the module under test — not where it was originally defined. `patch("citemesh.strategies.recommendation.get_client")` works; patching `citemesh.services.get_client` after the strategy module has already imported the name does nothing. Tests are network-free by default and isolate the cache root per test through `tests/conftest.py`.

**Duplicated choice lists are pinned by tests.** `data/user_config.py` declares its own literal choice tuples (`STRATEGY_CHOICES`, `THEME_CHOICES`, `DEVICE_CHOICES`, `MODEL_PROFILE_CHOICES`, `SEMANTIC_SOURCE_CHOICES`, `STORAGE_PRECISION_CHOICES`, `SEARCH_MODE_CHOICES`, `EXPORT_CHOICES`) rather than importing them from the parser, so loading configuration never drags in strategy or visualization modules. The contract tests in `tests/test_user_config.py` are what keep them in sync with `cli/parser.py` — change both together.

## Python ↔ JavaScript duplication

The dashboard viewer rebuilds its Plotly figure client-side from the embedded graph payload, so a set of algorithms is implemented twice on purpose: `stableCurveDirection`, `selectDashboardLabelIds`, `normalizeDashboardEdgeStrengths`, `dashboardHoverText`, `buildFigureSpecFromPayload` (mirroring `_build_plotly_figure` and its label/halo geometry), `csvGuard` / `csvEscape` and the CSV column order, `seedSlug`, `dashboardNodeLabel`, `safeExternalUrl`, `normalizeCollectionPackage` / `hasCompleteDashboardGeometry`, and `upsertCollectionEntries`. Their Python counterparts live in `visualization/export/{geometry,plotly_figure,links,csv_,nodes}.py`, `visualization/render.py`, and `visualization/dashboard/{package,payload}.py`.

This is pinned by tests in `tests/test_visualization.py`, not by convention:

- `test_exporter_dashboard_runtime_script_contracts` — asserts the emitted script contains each JS function definition plus formula fragments interpolated from the Python constants, then executes `csvGuard`, `csvEscape`, `seedSlug`, `dashboardNodeLabel`, and `escapeHtml` in a Node subprocess and checks exact outputs.
- `test_dashboard_labels_clear_selection_halos_after_import` — runs `buildFigureSpecFromPayload` in Node and checks label offsets against the Python halo math.
- `test_dashboard_highlights_fallback_to_path_order_and_use_theme_styles` — pins the JS color and opacity math against the Python theme values.
- `test_dashboard_invalid_embedded_collection_keeps_current_graph_in_node` and `test_dashboard_snapshot_loads_latest_same_result_in_node` — pin the client-side package validation against the server-side contract.

These skip when Node is not installed, so run them on a machine with `node` on `PATH` before changing either side.

## Extension points

**Add a strategy.** Subclass `GraphBuilderStrategy` in a new `strategies/<name>.py`. Implement `collect_papers` (fetch through `strategies/candidates.py` rather than calling the client directly, so you inherit budget splitting, identity reconciliation, and the outage policy) and `compute_similarity`. Override `prepare_graph_scoring` if you need to build an index over the selected papers first, `should_create_edge` for a custom admission rule, and `build_graph` if you need per-strategy edge capping or graph metadata. Then register the class in `strategies/__init__.py`'s lazy export map and `citemesh/__init__.py`'s, add the name to the parser's `--strategy` choices and to `STRATEGY_CHOICES` in `data/user_config.py`, and wire it into `cli/build_contract.py`.

**Add an export format.** Add a writer module under `visualization/export/`, expose a `to_<format>(path, ...)` method on `GraphExporter` that reads from `graph_payload()` rather than walking the graph itself, register the extension in the CLI's export routing and `EXPORT_CHOICES`, and add it to the optional-format cleanup list in `visualization/dashboard/package.py` so switching formats between runs removes stale siblings. Use `atomic_output_path` so a failed write never replaces a good file.

**Add a corpus source.** The arXiv corpus adapter lives in `strategies/embedding/hydration.py`. A new source must yield the same record shape (`id`, `title`, `abstract`, optionally `authors`, `categories`, `year`, `doi`, `venue`) and preserve the cache contracts: hydration metadata (source, split, cap, completion flag), the newest-first selection semantics, and resumability. Custom column mappings are deliberately unsupported — adapt at the source boundary.

**Add a model profile.** Append an `EmbeddingModelProfile` to `EMBEDDING_MODEL_PROFILES` in `data/model_profiles.py` with its own `schema_token`, prompt formatters, truncate-dim policy, autocast devices, attention preference, and compile eligibility; add the key to `_PROFILE_BY_KEY` and `MODEL_PROFILE_CHOICES`. The `schema_token` is part of the cache namespace, so bump it whenever a change alters what a vector means. Auto-detection for local checkpoints is `_local_embeddinggemma_evidence`-style architecture and prompt inspection; a profile with no detection evidence still works through an explicit `--model-profile`.

**Add a theme.** Extend the `THEMES` mapping in `visualization/themes.py`; the renderer, Plotly figure, and dashboard all read from it, and the dashboard CSS consumes the same values.

## External dependencies

- **Semantic Scholar API** for citation and recommendation data (the only network dependency of the core CLI).
- **HuggingFace Datasets** for arXiv corpus sources, and **huggingface_hub** for checkpoint resolution.
- **SentenceTransformers / Transformers / torch** for embeddings (`embeddings` extra).
- **Plotly and pyvis** for interactive exports (`viz` extra); `recommended` bundles both groups.
- **h5py, NetworkX, Matplotlib, scikit-learn, Rich** in the base install.
