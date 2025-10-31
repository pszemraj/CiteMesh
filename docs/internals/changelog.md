# Changelog & Key Improvements

This living document summarizes noteworthy changes from the initial script-based prototypes to the current CiteMesh implementation.

## Export & Visualization

- Added the `GraphExporter` abstraction to generate PNG, Pyvis HTML, Plotly HTML, JSON, and GraphML from a single graph object.
- Centralized theming with reusable palettes (light, dark, solarized, auto) for static and interactive outputs.
- Auto-positioned metadata callout now adapts text/background contrast based on the active theme.
- Node colors come from continuous gradients rather than ad-hoc per-output palettes, keeping year-based styling consistent.

## Graph Rendering

- All strategies feed a unified visualization pipeline: shared layout (Kamada-Kawai with spring fallback), node sizing tiers, and author-year labels.
- Edge opacity and width scale with normalized weights, maintaining clarity even with hybrid similarity signals.
- Integration tests route outputs to temporary paths so development runs don't get confused with fixtures.

## Persistent Caching

- Introduced `EmbeddingCache` (SQLite metadata + HDF5 vectors) keyed by model hash and input checksum, dramatically reducing repeat embedding costs.
- Joblib caches now live under the same user cache root, avoiding `./cache` clutter within the repo.
- Added `CITEMESH_CACHE_DIR` environment override for cluster or container deployments.
- Embedding strategy hits cached HuggingFace datasets by default; streaming is opt-in via `--streaming`.

## Strategy Enhancements

- **Citation**: uses real bibliographic coupling (shared reference lists) and temporal penalties; optional reference fetching skip for faster runs.
- **Embedding**: combines semantic similarity with temporal/category/author factors; fetches citation counts for top matches to balance node sizing.
- **Hybrid**: builds on the citation graph, injects semantic neighbors, and adjusts weights based on relationship provenance while capping per-node edges.

## CLI & Developer Experience

- `--export all` simplifies multi-format workflows; individual options remain for targeted runs.
- Strategy-specific flags (e.g., `--max-semantic`, `--dataset-split`) surface directly in `citemesh build --help`.
- Tests verify CLI ergonomics (help text, invalid args) and run end-to-end builds with ephemeral outputs.

## Future Opportunities

- More robust batching for Semantic Scholar reference fetching to increase bibliographic coupling coverage.
- Surface co-citation analytics or shared-neighbor stats in exports for richer post-processing.
- Provide ready-made Plotly templates / Dash layouts for embedding graphs in analytical notebooks or dashboards.
