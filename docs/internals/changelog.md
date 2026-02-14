# Changelog & Key Improvements

## Scope

This page is historical context, not a normative behavior specification.

- Use [CLI Usage](../guides/cli.md) for current command/flag behavior.
- Use [Caching & Data](../guides/caching.md) for current cache behavior.
- Documentation ownership map: [Documentation Index](../README.md).

This changelog summarizes notable changes from early script-based prototypes to the current package architecture.
Historical bullets below may describe superseded behavior; canonical current behavior remains in the guides above.

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
- Switched embedding cache defaults to quantized storage (`int8` + calibration ranges) with optional binary prefilter index for large-corpus retrieval.
- Added cache-native search API with binary Hamming prefilter + float query rescoring.
- Persisted authors/categories metadata in cache so warm-cache retrieval can skip corpus reloads.
- Added hydration metadata gating so matching split/corpus-cap runs query directly from cache.
- Added precision-aware cache namespaces (`storage precision`, `binary mode`, `source dtype`) to isolate incompatible cache layouts.
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
- Added embedding/hybrid cache controls: `--storage-precision`, binary prefilter toggles, binary rescore multiplier, calibration sample size, and cache compression knobs.
- Removed legacy module shims after package consolidation.
- Expanded paper-ID normalization for DOI/arXiv URL forms.
- Switched multi-export explicit `--output` handling to directory-based exports with strategy-named files.
- Enforced strict embedding `--top-k` per-node edge caps.
- Reused a single layout per run across layout-consuming exporters.
- Made metadata timestamps opt-in (`--include-timestamp`) for deterministic outputs by default.
- Grouped auto outputs into stable per-paper title folders.
- Removed title truncation in runtime seed logs and static/Plotly chart titles.
- Moved interactive exporters to optional `.[viz]` extras.

## Maintenance Consolidation

- Removed dead configuration/helpers that were no longer referenced.
- Deduplicated Semantic Scholar citation/reference fetch loop logic.
- Tightened slow-test policy to keep one explicit smoke path.

## Future Opportunities

- Improve batching strategies for reference fetching coverage.
- Add co-citation/shared-neighbor analytics in structured exports.
- Provide ready-made Plotly/Dash templates for downstream analysis.
