# CiteMesh architecture

CiteMesh separates argument handling, paper acquisition, scoring, persistence, and visualization. The [pipeline walkthrough](../guides/how-it-works.md) follows a complete build.

## Execution flow

`cli/` parses arguments, applies `config.toml` defaults, validates effective options in `build_contract.py`, and resolves export targets before dispatching to `GraphBuilderStrategy`. Strategies acquire candidates through `strategies/candidates.py`, pool them in `strategies/discovery_pool.py`, prepare scoring in `prepare_graph_scoring`, and return a `networkx.Graph`; `visualization/` lays out and exports it. The [graph-input contract](../reference/output-artifacts.md#graph-input) defines node attributes.

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

The viewer rebuilds its Plotly figure client-side from the embedded payload, so a dozen algorithms exist in both languages on purpose: `stableCurveDirection`, `selectDashboardLabelIds`, `normalizeDashboardEdgeStrengths`, `dashboardHoverText`, `buildFigureSpecFromPayload` (mirroring `_build_plotly_figure` and its label/halo geometry), `csvGuard`/`csvEscape` and the CSV column order, `focalSlug`, `dashboardNodeLabel`, `safeExternalUrl`, `normalizeCollectionPackage`/`hasCompleteDashboardGeometry`, and `upsertCollectionEntries`. The Python counterparts live in `visualization/export/{geometry,plotly_figure,links,csv_,nodes}.py`, `visualization/render.py`, and `visualization/dashboard/{package,payload}.py`.

`tests/test_visualization.py` pins both sides: it checks the emitted script against formula fragments interpolated from the Python constants, then runs those functions in a Node subprocess and compares outputs, label offsets, color math, and package validation. It skips without Node, so run it with `node` on `PATH` before touching either side.

## Extension points

**A strategy** subclasses `GraphBuilderStrategy` in `strategies/<name>.py` with `collect_papers` (fetching through the source helpers in `candidates.py` and pooling with `DiscoveryPool` or `fetch_candidate_pool` from `discovery_pool.py`, not the client directly, to inherit the outage policy, identity reconciliation, and reference recovery) and `compute_similarity`. Register it in the lazy export maps of `strategies/__init__.py` and `citemesh/__init__.py`, in `core/choices.py:STRATEGY_CHOICES`, and in `cli/build_options.py`'s factory dispatch and option-support tables. Add any strategy-specific options and validation to the parser and build contract.

**An export format** adds a writer, usually through a `to_<format>(path, ...)` method on `GraphExporter`; `png` uses `visualize_graph`. Reuse node enrichment or `graph_payload()` as the format requires. Register it in `core/choices.py:EXPORT_FORMATS` and `cli/outputs.py`'s `EXPORT_EXTENSIONS` and `_EXPORTER_METHOD` tables. Collection cleanup derives optional result paths from `EXPORT_FORMATS` and `EXPORT_EXTENSIONS`. Publish through the atomic writers in `data/cache.py`.

**A corpus source** feeds the [supported record adapter](../guides/caching.md#corpus-records) in `strategies/embedding/records.py` and the selection/resume machinery in `hydration.py`. Preserve stable source IDs and hydration metadata. Parseable arXiv IDs enable chronology-based ranking; other IDs and anonymous records use the documented fallback selection.

**A model profile** appends an `EmbeddingModelProfile` to `EMBEDDING_MODEL_PROFILES` in `data/model_profiles.py` and its key to `_PROFILE_BY_KEY` and `core/choices.py:MODEL_PROFILE_CHOICES`. Its `schema_token` is part of the cache namespace, so bump it whenever a change alters what a vector means. A profile with no detection evidence still works through an explicit `--model-profile`.

**A theme** extends `THEMES` in `visualization/themes.py` and `core/choices.py:THEME_CHOICES`; renderer, Plotly, dashboard, and CSS consume the palette.

## External dependencies

The core CLI uses the Semantic Scholar API and, only when reference discovery for the focal paper is empty or unavailable, arXiv HTML and its metadata API. The arXiv fallback uses the standard library and adds no package dependency. Everything torch- or Plotly-shaped lives behind the `embeddings` and `viz` extras; h5py, NetworkX, Matplotlib, scikit-learn, and Rich ship in the base install.

## Hosted demo

The [Pages workflow](../../.github/workflows/pages.yml) publishes `assets/examples/megalodon/` on each push to `main`. The saved `dashboard.html` embeds its scripts and graph data for offline use. [`scripts/stage_pages_demo.py`](../../scripts/stage_pages_demo.py) extracts those scripts and data into `dashboard.js` for hosting, leaving the branded metadata and interface markup in `index.html`. Pages also publishes the collection package, favicon, and Open Graph image.

**Working link previews are a required outcome of a demo refresh.** Keep the staged HTML below 512 KiB (524,288 bytes), the limit reported by the preview scanner. Copying the standalone dashboard directly to `index.html` exceeds that limit because it embeds Plotly and the graph data. The image's compressed size is separate from the HTML response limit.

After refreshing the saved dashboard, preserve its branded title, description, canonical URL, favicon, Open Graph tags, and Twitter card tags; the generic dashboard exporter does not supply them. When replacing `og-image.png`, update its declared width and height to match the image. The [curated demo notes](../../AGENTS.md#curated-demo) identify the focal paper used for the saved example.

From the repository root, reproduce Pages staging and serve it locally:

```bash
python scripts/stage_pages_demo.py assets/examples/megalodon/dashboard.html out/pages-demo
cp assets/examples/megalodon/{dashboard.citemesh.json,favicon.ico,og-image.png} out/pages-demo/
wc -c out/pages-demo/index.html
python -m http.server 8000 --directory out/pages-demo
```

Open `http://localhost:8000/` and check the graph, search, and paper selection. The staging script uses only Python's standard library. Staged files belong in the ignored output directory; commit the source assets and staging code.

After deployment, check the [public demo](https://pszemraj.github.io/CiteMesh/) again:

- Inspect the initial HTML response for the size limit and complete Open Graph and Twitter metadata, without relying on JavaScript to add it.
- Fetch the image URL from the metadata and confirm HTTP 200, the image content type, dimensions, and compressed file size.
- Rescan the deployed URL with the preview checker and confirm that the title, description, and image appear.

Report live preview verification as pending until the deployed page has passed these checks. A local browser render alone does not verify the public preview.
