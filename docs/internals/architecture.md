# CiteMesh architecture

CiteMesh separates argument handling, paper acquisition, scoring, persistence, and visualization. The [pipeline walkthrough](../guides/how-it-works.md) follows a complete build.

## Execution flow

`cli/` parses arguments, applies `config.toml` defaults, validates the effective options in `build_contract.py`, resolves export targets, and calls `build_graph` on the selected `GraphBuilderStrategy`. The strategy collects papers through `strategies/candidates.py` (source fetch, availability policy, identity reconciliation, pool budgets), builds its scoring space in `prepare_graph_scoring`, scores pairs in `compute_similarity`, and returns a `networkx.Graph`. `visualization/` computes one shared layout for `render.visualize_graph`, `export.GraphExporter`, and `dashboard/package.py`.

Every node carries a consistent attribute payload, which keeps visualization and export strategy-agnostic; see the [graph-input contract](../reference/output-artifacts.md#graph-input).

## Package map

- `core/` owns the data model, validation, shared choices, and configuration without I/O.
- `data/` owns user configuration, cache roots, model profiles, and persistent embedding storage.
- `services/` owns Semantic Scholar transport/cache behavior and conditional arXiv reference recovery.
- `strategies/` owns candidate acquisition, graph construction, scoring, and embedding runtime behavior.
- `visualization/` owns layouts, exporters, dashboard payloads, and browser assets.
- `cli/` owns parsing, effective-option validation, command dispatch, logging, and output paths.

The [embedding-cache internals](embedding-cache.md), [arXiv recovery guide](../guides/cli.md#missing-semantic-scholar-references), and [output-artifact reference](../reference/output-artifacts.md) document the boundaries that need more detail.

## Rules

**Dependency direction is one-way**: `core` → `data` → `services` → `strategies` → `visualization` → `cli`; a module never imports from a layer to its right, and `visualization/` never imports `cli`, so exporters stay usable without an argument parser.

**Optional dependencies stay lazily imported.** torch, sentence-transformers, datasets, plotly, and pyvis must never be imported at module scope on a path the base CLI reaches: a bare `pip install citemesh` has to run `citemesh build --strategy recommendation` and `citemesh --help` without them. Follow the guards in `strategies/embedding/deps.py` and `visualization/export/loaders.py`.

Option vocabularies in `core/choices.py` have no runtime dependencies. The parser, persisted configuration, and relevant runtimes import them directly; changing a vocabulary does not require a second literal list or a test comparing copies.

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

The core CLI uses the Semantic Scholar API and, only when seed-reference discovery is empty or unavailable, arXiv HTML and its metadata API. The arXiv fallback uses the standard library and adds no package dependency. Everything torch- or Plotly-shaped lives behind the `embeddings` and `viz` extras; h5py, NetworkX, Matplotlib, scikit-learn, and Rich ship in the base install.
