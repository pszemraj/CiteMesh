# CiteMesh architecture

CiteMesh separates argument handling, paper acquisition, scoring, persistence, and visualization. The [pipeline walkthrough](../guides/how-it-works.md) follows a complete build.

## Execution flow

`cli/` parses arguments, applies `config.toml` defaults, validates the effective options in `build_contract.py`, resolves export targets, and calls `build_graph` on the selected `GraphBuilderStrategy`. The strategy collects papers through `strategies/candidates.py` (source fetch, availability policy, identity reconciliation, pool budgets), builds its scoring space in `prepare_graph_scoring`, scores pairs in `compute_similarity`, and returns a `networkx.Graph`. `visualization/` computes one shared layout for `render.visualize_graph`, `export.GraphExporter`, and `dashboard/package.py`.

Every node carries the same attribute payload (`paper`, `title`, `year`, `authors`, `citation_count`, `venue`, `arxiv_id`, `doi`, `is_seed`, `is_local_corpus`), which is what keeps visualization and export strategy-agnostic.

## Package map

### `core/` - data model and constants, no I/O

- [choices.py](../../src/citemesh/core/choices.py) - shared CLI, config, and runtime option vocabularies
- `config.py` - default dataclasses and their singletons (`TemporalConfig`, `EmbeddingSimilarityConfig`, `EmbeddingStorageConfig`, `HybridSimilarityConfig`, `VisualizationConfig`, `APIConfig`); embedding weight and storage settings validate on import
- `models.py` - `Paper`, `Author`, and the overlap helpers the scorers rely on
- `paper_fields.py` - tolerant coercion of venue, author, and category fields out of inconsistent upstream payloads
- `paper_ids.py` - identifier normalization and alias derivation
- `values.py`, `validation.py` - shared value coercion and input validation primitives
- `text_batching.py` - length-bucketed encode batching and `l2_normalize_embeddings`; it sits here rather than under `strategies/embedding/` because `data/embedding_cache/` encodes through it too

### `data/` - persistence

- `cache.py` - cache-root resolution across platforms, atomic text/JSON writers, `atomic_output_path`, the cache-root `ReadWriteLock`
- `user_config.py` - loads, validates, and rewrites `config.toml`, whitelisting `[defaults]` keys and `[api] s2_api_key` with per-key casters; invalid entries are ignored so a bad config never blocks the CLI
- `model_profiles.py` - the embedding profile registry: EmbeddingGemma, its three prompt formatters, truncate-dim policy, attention and compile eligibility, and the fallback chain
- `embedding_cache/` - `store.py` composes `EmbeddingCache` from the `ingest`, `layout`, `recovery`, and `search` mixins over `constants`, `sql`, `models`, `quantization`; see [Embedding Cache Internals](embedding-cache.md)

### `services/semantic_scholar/` - Semantic Scholar transport

`errors.py` (failure taxonomy, per-capability `_FailureDomain` budgets) · `retry.py` (Tenacity backoff, `Retry-After`) · `disk_cache.py` (persisted paper and reference-ID caches) · `payloads.py` (parsing into `Paper`) · `endpoints.py` (one method per capability) · `client.py` (transport, rate limiting, `candidate_operation_scope`, `get_client`).

### `strategies/` - candidate acquisition and scoring

- `base.py` - `GraphBuilderStrategy`: the template method and its four hooks, the shared temporal/citation/bibliographic scorers, `deterministic_sort_key`, edge capping
- `candidates.py` - pool budgets, `fetch_candidate_source` and its `complete`/`empty`/`unavailable` vocabulary, `IdentityRegistry`, `reconcile_paper_identity`, `scope_candidate_collection`
- `similarity.py` - `AbstractSimilarityIndex`, the TF-IDF scorer behind citation and recommendation topical similarity
- `citation.py`, `recommendation.py`, `hybrid.py` - the concrete strategies
- `embedding/` - `deps` (lazy dependency guards) · `runtime` (device resolution, probes) · `model_runtime` (load, fallback, precision validation, TF32 and compile guards) · `precision` · `text` (`EmbeddingTask`, formatters) · `records` · `config` · `fingerprint` · `hydration` (corpus selection, calibration, resume) · `builder`

### `visualization/` - layout, render, export

- `render.py` - `compute_layout` and its chain (community detection, spreading, packing, orientation, normalization), `compute_node_sizes`, `visualize_graph`
- `paths.py` - output-path and filename derivation (`generate_output_path`), free of matplotlib and numpy so the CLI resolver skips the rendering stack; `render` re-exports it.
- `themes.py` - the immutable `light`/`dark`/`solarized` palettes and `auto` resolution
- `node_data.py` - [Paper/scalar metadata precedence](../reference/output-artifacts.md#graph-input) shared by exporters, rendering, and filenames
- `years.py`, `ordering.py` - year coercion and deterministic node/edge ordering
- `export/` - `__init__` (`GraphExporter`, the `to_*` writers) · `nodes` (enrichment, seed-relevance PageRank) · `geometry` (primitives shared with the dashboard) · `plotly_figure`, `links`, `keys`, `loaders`, `bibtex`, `graphml`, `csv_`
- `dashboard/` - `contracts.py` (`kind`/`schema_version`), `payload.py` (bundle assembly), `package.py` (locking, staging, upsert, rollback), `assets/`

### `cli/` and top-level modules

`__init__.py` (entry point and dispatch) · `parser.py` · `console.py` (Rich console and logging) · `build_options.py` and `build_contract.py` (strategy-scoped option validation and builder selection) · `outputs.py` · `graph_config.py` (the `*.config.json` sidecar) · `cache_ops.py` · `commands/{build,search,view,config,cache}.py`.

Top-level: `__main__.py` (`python -m citemesh`), `_lazy.py` (the shared PEP 562 lazy-export plumbing every `__init__` uses), `progress.py`, `_runtime.py` (process-level runtime setup), `_version.py`.

## Rules

**Dependency direction is one-way**: `core` → `data` → `services` → `strategies` → `visualization` → `cli`; a module never imports from a layer to its right, and `visualization/` never imports `cli`, so exporters stay usable without an argument parser.

**Optional dependencies stay lazily imported.** torch, sentence-transformers, datasets, plotly, and pyvis must never be imported at module scope on a path the base CLI reaches: a bare `pip install citemesh` has to run `citemesh build --strategy recommendation` and `citemesh --help` without them. Follow the guards in `strategies/embedding/deps.py` and `visualization/export/loaders.py`.

Option vocabularies in `core/choices.py` have no runtime dependencies. The parser, persisted configuration, and relevant runtimes import them directly; changing a vocabulary does not require a second literal list or a test comparing copies.

Test and code conventions are in [Contributing](../../CONTRIBUTING.md#code-conventions).

## Python ↔ JavaScript duplication

The viewer rebuilds its Plotly figure client-side from the embedded payload, so a dozen algorithms exist in both languages on purpose: `stableCurveDirection`, `selectDashboardLabelIds`, `normalizeDashboardEdgeStrengths`, `dashboardHoverText`, `buildFigureSpecFromPayload` (mirroring `_build_plotly_figure` and its label/halo geometry), `csvGuard`/`csvEscape` and the CSV column order, `seedSlug`, `dashboardNodeLabel`, `safeExternalUrl`, `normalizeCollectionPackage`/`hasCompleteDashboardGeometry`, and `upsertCollectionEntries`. The Python counterparts live in `visualization/export/{geometry,plotly_figure,links,csv_,nodes}.py`, `visualization/render.py`, and `visualization/dashboard/{package,payload}.py`.

`tests/test_visualization.py` pins both sides: it checks the emitted script against formula fragments interpolated from the Python constants, then runs those functions in a Node subprocess and compares outputs, label offsets, color math, and package validation. It skips without Node, so run it with `node` on `PATH` before touching either side.

## Extension points

**A strategy** subclasses `GraphBuilderStrategy` in `strategies/<name>.py` with `collect_papers` (fetching through `candidates.py`, not the client, to inherit budget splitting, identity reconciliation, and the outage policy) and `compute_similarity`. Register it in the lazy export maps of `strategies/__init__.py` and `citemesh/__init__.py`, in `core/choices.py:STRATEGY_CHOICES`, and in `cli/build_options.py`'s factory dispatch and option-support tables. Add any strategy-specific options and validation to the parser and build contract.

**An export format** adds a writer and a `to_<format>(path, ...)` method on `GraphExporter`. Reuse node enrichment or `graph_payload()` as the format requires. Add the format to `core/choices.py:EXPORT_FORMATS`, its extension and method to `cli/outputs.py`, and its cleanup entry in `dashboard/package.py`. Publish through the atomic writers in `data/cache.py`.

**A corpus source** feeds the [supported record adapter](../guides/caching.md#corpus-records) in `strategies/embedding/records.py` and the selection/resume machinery in `hydration.py`. Preserve stable source IDs and hydration metadata. Parseable arXiv IDs enable chronology-based ranking; other IDs and anonymous records use the documented fallback selection.

**A model profile** appends an `EmbeddingModelProfile` to `EMBEDDING_MODEL_PROFILES` in `data/model_profiles.py` and its key to `_PROFILE_BY_KEY` and `core/choices.py:MODEL_PROFILE_CHOICES`. Its `schema_token` is part of the cache namespace, so bump it whenever a change alters what a vector means. A profile with no detection evidence still works through an explicit `--model-profile`.

**A theme** extends `THEMES` in `visualization/themes.py` and `core/choices.py:THEME_CHOICES`; renderer, Plotly, dashboard, and CSS consume the palette.

## External dependencies

The Semantic Scholar API is the only network dependency of the core CLI. Everything torch- or Plotly-shaped lives behind the `embeddings` and `viz` extras; h5py, NetworkX, Matplotlib, scikit-learn, and Rich ship in the base install.
