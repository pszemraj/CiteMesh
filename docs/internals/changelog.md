# Changelog & Key Improvements

This is a historical changelog.

This living document summarizes noteworthy changes from the initial script-based prototypes to the current CiteMesh implementation.

For current operational behavior, use the canonical guides:
[CLI Usage](../guides/cli.md) and [Caching & Data](../guides/caching.md).
Treat this page as historical context, not the normative behavior spec.

## Export & Visualization

- Added the `GraphExporter` abstraction to generate PNG, Pyvis HTML, Plotly HTML, JSON, and GraphML from a single graph object.
- Centralized theming with reusable palettes (light, dark, solarized, auto) for static and interactive outputs.
- Auto-positioned metadata callout now adapts text/background contrast based on the active theme.
- Node colors come from continuous gradients rather than ad-hoc per-output palettes, keeping year-based styling consistent.

## Graph Rendering

- Layout-based renders now share a unified pipeline: shared layout (Kamada-Kawai with spring fallback), node sizing tiers, and author-year labels.
- Edge opacity and width scale with normalized weights, maintaining clarity even with hybrid similarity signals.
- Integration tests route outputs to temporary paths so development runs don't get confused with fixtures.
- Layout computation now canonicalizes node/edge insertion order, and node-size ranking now uses deterministic tie-breaking (`citation_count`, then node ID).

## Persistent Caching

- Introduced `EmbeddingCache` (SQLite metadata + HDF5 vectors) keyed by model hash and input checksum, dramatically reducing repeat embedding costs.
- Joblib caches now live under the same user cache root, avoiding `./cache` clutter within the repo.
- Added `CITEMESH_CACHE_DIR` environment override for cluster or container deployments.
- Embedding strategy hits cached HuggingFace datasets by default; streaming is opt-in via `--streaming`.
- Introduced model profiles (starting with EmbeddingGemma) to add recommended prompts and explicit precision policies without hard-coding logic in strategies.

## Strategy Enhancements

- **Citation**: uses real bibliographic coupling (shared reference lists) and temporal penalties; optional reference fetching skip for faster runs.
- **Embedding**: combines semantic similarity with temporal/category/author factors; fetches citation counts for top matches to balance node sizing. EmbeddingGemma now defaults to 256d Matryoshka embeddings (with clear runtime logs and dim-specific cache names).
- **Embedding**: EmbeddingGemma now attempts a best-effort `torch.compile` on the inner HF model (`model[0].auto_model`) while leaving the SentenceTransformer wrapper uncompiled; failures fall back safely.
- **Hybrid**: builds on the citation graph, injects semantic neighbors, and adjusts weights based on relationship provenance while capping per-node edges.
- **Recommendation**: adds Semantic Scholar recommendation-based discovery with direct search endpoint and API-key-aware rate-limit handling. Reference-aware mode now requests reference payloads directly (with fallback hydration) so bibliographic coupling can contribute to scoring without unnecessary per-paper calls.
- Citation and recommendation edge creation now use `--similarity-threshold` as the sole minimum edge gate.
- Embedding streaming mode now fails fast on sliced split expressions (for example `train[:5%]`) with explicit guidance to use `--corpus-size`.

## CLI & Developer Experience

- `--export all` simplifies multi-format workflows; individual options remain for targeted runs.
- CLI help and parser validation now surface strategy-specific controls clearly, and tests cover invalid-argument ergonomics plus end-to-end smoke paths with ephemeral outputs.
- Package layout uses canonical modules (`core/`, `data/`, `services/`, `visualization/`); pre-release legacy import shims were removed.
- Paper ID normalization now accepts arXiv/DOI URL forms and strips arXiv version suffixes before API calls.
- Explicit output basenames with dots are preserved across multi-export runs.
- Embedding `--top-k` enforces strict per-node edge caps during pruning.
- CLI computes one shared layout per run for layout-consuming exports and reuses it across PNG/Plotly outputs.
- Embedding/hybrid CLI flows default to a bounded semantic corpus with explicit uncapped opt-in.
- Recommendation strategy requests reference payloads directly when enabled, reducing per-paper reference hydration calls.
- Metadata timestamps are now opt-in via `--include-timestamp`, improving deterministic artifact generation by default.
- Default auto-generated outputs are grouped under per-paper folders with stable seed suffixes (`out/<safe-seed-title>-<seed-hash8>/`) to reduce collisions.
- Interactive exporters (`pyvis`, `plotly`) moved to optional `.[viz]` dependencies.

For exact current flag semantics and defaults, see the canonical [CLI Usage](../guides/cli.md) guide.

## Maintenance Consolidation

- Removed dead configuration and helper surface that was no longer referenced (unused config fields/methods and stale similarity helper property).
- Deduplicated Semantic Scholar citation/reference fetch loops via shared retry/convert logic.
- Tightened test suite by removing flaky pass-without-assert slow branches and keeping one explicit slow smoke test.

## Future Opportunities

- More robust batching for Semantic Scholar reference fetching to increase bibliographic coupling coverage.
- Surface co-citation analytics or shared-neighbor stats in exports for richer post-processing.
- Provide ready-made Plotly templates / Dash layouts for embedding graphs in analytical notebooks or dashboards.
