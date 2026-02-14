# Changelog & Key Improvements

## Scope

This page is historical context, not a normative behavior specification.

- Use [CLI Usage](../guides/cli.md) for current command/flag behavior.
- Use [Caching & Data](../guides/caching.md) for current cache behavior.

This changelog summarizes notable changes from early script-based prototypes to the current package architecture.

## Export & Visualization

- Added `GraphExporter` to emit PNG, Pyvis HTML, Plotly HTML, JSON, and GraphML from one graph object.
- Centralized theming (light, dark, solarized, auto) for static and interactive exporters.
- Added adaptive metadata callout contrast based on active theme.
- Switched node coloring to continuous gradients for consistent year-based styling.

## Graph Rendering

- Unified layout-oriented rendering pipeline across outputs.
- Added normalized edge opacity/width scaling for weight readability.
- Routed integration-test outputs to temporary paths to reduce fixture confusion.
- Canonicalized node/edge ordering and deterministic tie-breaks for stable layout artifacts.

## Persistent Caching

- Introduced `EmbeddingCache` with SQLite metadata + HDF5 matrix storage.
- Moved joblib caches under a user-scoped cache root.
- Added `CITEMESH_CACHE_DIR` override for custom deployments.
- Added streaming mode for embedding corpus ingestion as an explicit opt-in.
- Added model profiles (starting with EmbeddingGemma) for prompts/precision policy.

## Strategy Evolution

- **Citation**: moved to real bibliographic coupling with reference-aware similarity factors.
- **Embedding**: expanded multi-factor similarity and added citation-count hydration for top matches.
- **Embedding**: added best-effort `torch.compile` path for EmbeddingGemma internals with safe fallback.
- **Hybrid**: formalized citation-first enrichment with semantic additions and capped edges.
- **Recommendation**: added recommendation-based discovery with direct endpoint handling and rate-limit-aware behavior.
- Unified edge gating for citation/recommendation under `--similarity-threshold`.
- Added explicit streaming split validation for embedding mode.

## CLI & Developer Experience

- Added `--export all` for multi-format runs.
- Improved parser validation and test coverage around CLI ergonomics.
- Removed legacy module shims after package consolidation.
- Expanded paper-ID normalization for DOI/arXiv URL forms.
- Preserved dotted custom output basenames across multi-export workflows.
- Enforced strict embedding `--top-k` per-node edge caps.
- Reused a single layout per run across layout-consuming exporters.
- Made metadata timestamps opt-in (`--include-timestamp`) for deterministic outputs by default.
- Grouped auto outputs into stable per-paper folder naming.
- Moved interactive exporters to optional `.[viz]` extras.

## Maintenance Consolidation

- Removed dead configuration/helpers that were no longer referenced.
- Deduplicated Semantic Scholar citation/reference fetch loop logic.
- Tightened slow-test policy to keep one explicit smoke path.

## Future Opportunities

- Improve batching strategies for reference fetching coverage.
- Add co-citation/shared-neighbor analytics in structured exports.
- Provide ready-made Plotly/Dash templates for downstream analysis.
