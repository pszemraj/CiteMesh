# Changelog & Key Improvements

This changelog summarizes notable changes from early script-based prototypes to the current package architecture.
For current usage details, see:

- [CLI Usage](https://github.com/pszemraj/CiteMesh/blob/main/docs/guides/cli.md)
- [Caching & Data](https://github.com/pszemraj/CiteMesh/blob/main/docs/guides/caching.md)
- [Environment Variables](https://github.com/pszemraj/CiteMesh/blob/main/docs/reference/environment.md)
- [Embedding Runtime](https://github.com/pszemraj/CiteMesh/blob/main/docs/reference/embedding-runtime.md)

## Breaking Changes

- Removed deprecated `GraphBuilderStrategy.exponential_temporal_decay`.
- Removed `random_seed` constructor arguments from strategy builders.
- Clarified `--seed` CLI semantics as layout/export determinism only.

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
- Added precision-aware cache namespaces (`storage precision`, effective `binary mode`, source dtype) to isolate incompatible cache layouts.
- Added `CITEMESH_CACHE_DIR` override for custom deployments.
- Added streaming mode for embedding corpus ingestion as an explicit opt-in.
- Added model profiles (starting with EmbeddingGemma) for prompts/precision policy.
- Added reason-tagged embedding namespace clear logs with payload-size/row summaries for cache invalidation transparency.
- Added incremental full-corpus hydration growth checks so upstream row-count increases append only delta records instead of forcing full namespace rebuilds.
- Strengthened full-corpus incremental hydration with staged tail/head delta checks plus missing-ID reconciliation fallback, and memoized duplicate-only row-count deltas to avoid repeated full-split rescans.
- Fixed incremental full-corpus hydration to fetch tail slices (`offset=cached_rows`) so delta refreshes hydrate newly appended records instead of reloading leading rows.
- Hardened reference-cache reuse against unreadable/non-object JSON entries and improved atomic write durability with parent-directory fsync.

## Strategy Evolution

- **Citation**: moved to real bibliographic coupling with reference-aware similarity factors.
- **Embedding**: expanded multi-factor similarity and added citation-count hydration for top matches.
- **Embedding**: added best-effort `torch.compile` path for EmbeddingGemma internals with safe fallback.
- **Embedding**: switched eager TF32 runtime configuration to the PyTorch 2.9+ `torch.backends.fp32_precision` API.
- **Embedding**: for torch 2.9/2.10 CUDA compile paths, switched TF32 control to `torch.set_float32_matmul_precision("high")` so `torch.compile` remains available without tripping the mixed TF32 API conflict in release-branch Inductor.
- **Embedding**: changed default checkpoint to `unsloth/embeddinggemma-300m` (ungated) and added automatic fallback to `google/embeddinggemma-300m` for default-revision loads.
- **Embedding**: citation-count enrichment logs now show bounded target counts and render a visible progress bar on TTY runs.
- **Embedding**: cache fingerprint enforcement now follows the runtime-active checkpoint identity after model fallback selection, preventing stale cross-checkpoint reuse in shared namespaces.
- **Hybrid**: moved from citation-first semantic add-on behavior to merged citation+semantic candidate reranking with semantic-only cap enforcement.
- **Hybrid**: default depth targets were raised to `25/25/25` (references/citations/semantic cap) after the February 2026 sweep to improve foundational-paper recovery while keeping recent-paper quality high.
- **Recommendation**: added recommendation-based discovery with direct endpoint handling and rate-limit-aware behavior.
- Unified edge gating for citation/recommendation under `--similarity-threshold`.
- Added explicit streaming split validation for embedding mode.

## CLI & Developer Experience

- Added `--export all` for multi-format runs.
- Improved parser validation and test coverage around CLI ergonomics.
- Added embedding/hybrid cache controls: `--storage-precision`, binary prefilter toggles, binary rescore multiplier, calibration sample size, and cache compression knobs.
- Added explicit overwrite acknowledgement for embedding cache rebuilds (`--force-rebuild-cache` + `--overwrite-cache`) with default confirmation prompts.
- Added optional cache-clear rationale flags (`--cache-overwrite-reason`, `cache clear --reason`) and standardized destructive-clear logs with file/size snapshots and large-cache warnings.
- Removed legacy module shims after package consolidation.
- Expanded paper-ID normalization for DOI/arXiv URL forms.
- Switched multi-export explicit `--output` handling to directory-based exports with strategy-named files.
- Enforced strict embedding `--top-k` per-node edge caps.
- Reused a single layout per run across layout-consuming exporters.
- Made metadata timestamps opt-in (`--include-timestamp`) for deterministic outputs by default.
- Added per-run `*.config.json` sidecar exports containing rebuild parameters and run metadata.
- Simplified export completion logging to one summary line per run.
- Grouped auto outputs into stable per-paper title folders.
- Removed title truncation in runtime seed logs and static/Plotly chart titles.
- Reclassified non-empty reference-cache payloads with zero valid IDs as invalid so corrupted payloads are rebuilt instead of silently suppressing references.
- Fixed hybrid CLI validation to resolve effective default `--max-semantic` before rejecting embedding-only flags.
- Fixed cache-subcommand CLI parsing so shared logging flags are accepted after `cache scan/clear` tokens (for example `citemesh cache scan --log-level debug`).
- Removed unsupported `szip` embedding-cache compression mode from CLI/validation.
- Moved interactive exporters to optional `.[viz]` extras.

## Maintenance Consolidation

- Removed dead configuration/helpers that were no longer referenced.
- Deduplicated Semantic Scholar citation/reference fetch loop logic.
- Tightened slow-test policy to keep one explicit smoke path.

## Future Opportunities

- Improve batching strategies for reference fetching coverage.
- Add co-citation/shared-neighbor analytics in structured exports.
- Provide ready-made Plotly/Dash templates for downstream analysis.

### Dashboard Backlog (Tracked TODOs)

- [TODO-dashboard] Add cluster-level labels/hulls in dashboard graph view (topic keyword extraction per cluster).
- [TODO-dashboard] Add bridge-paper quick lens (betweenness/connector score + dedicated list mode).
- [TODO-dashboard] Add graph-pane matrix toggle (adjacency heatmap ordered by cluster/relevance).
- [TODO-dashboard] Add explicit "obscure gems" ranking lens (semantic relevance + citation-age normalization).
- [TODO-dashboard] Expand seed-relation facets for recommendation/embedding graphs (directed relation metadata beyond provenance/year heuristics).
- [TODO-dashboard] Add reading-queue workflow (save/reject/note with `localStorage` export/import).
- [TODO-dashboard] Add optional local PDF download + inline viewer workflow (`--download-pdfs` style export mode).
