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
- Introduced model profiles (starting with EmbeddingGemma) to add recommended prompts and dtype notes without hard-coding logic in strategies.

## Strategy Enhancements

- **Citation**: uses real bibliographic coupling (shared reference lists) and temporal penalties; optional reference fetching skip for faster runs.
- **Embedding**: combines semantic similarity with temporal/category/author factors; fetches citation counts for top matches to balance node sizing.
- **Hybrid**: builds on the citation graph, injects semantic neighbors, and adjusts weights based on relationship provenance while capping per-node edges.
- **Recommendation**: adds Semantic Scholar recommendation-based discovery with direct search endpoint and API-key-aware rate-limit handling.

## CLI & Developer Experience

- `--export all` simplifies multi-format workflows; individual options remain for targeted runs.
- Strategy-specific flags (e.g., `--max-semantic`, `--dataset-split`) surface directly in `citemesh build --help`.
- Tests verify CLI ergonomics (help text, invalid args) and run end-to-end builds with ephemeral outputs.
- Package layout uses canonical modules (`core/`, `data/`, `services/`, `visualization/`); pre-release legacy import shims were removed.
- Paper ID normalization now accepts arXiv/DOI URLs directly (for example `https://arxiv.org/abs/...`) and converts them to canonical IDs before API calls.
- Explicit output basenames with dots are now preserved across multi-export runs (for example `-o out/arxiv-2508.14040-example --export all`).
- Embedding `--top-k` now enforces strict per-node edge caps during pruning.
- CLI computes a single shared layout per run and reuses it across static/interactive exports; `--seed` now consistently controls that shared layout path.
- Default auto-generated outputs are grouped under per-paper folders (`out/<safe-seed-title>/`) instead of a flat `out/` namespace.

## Maintenance Consolidation

- Removed dead configuration and helper surface that was no longer referenced (unused config fields/methods and stale similarity helper property).
- Deduplicated Semantic Scholar citation/reference fetch loops via shared retry/convert logic.
- Tightened test suite by removing flaky pass-without-assert slow branches and keeping one explicit slow smoke test.

## Future Opportunities

- More robust batching for Semantic Scholar reference fetching to increase bibliographic coupling coverage.
- Surface co-citation analytics or shared-neighbor stats in exports for richer post-processing.
- Provide ready-made Plotly templates / Dash layouts for embedding graphs in analytical notebooks or dashboards.
