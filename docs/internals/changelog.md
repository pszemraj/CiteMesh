# Changelog & Key Improvements

Major changes from the early script-based prototypes to the current pre-1.0
package. Current behavior is described in the linked guides and references:

- [CLI Usage](../guides/cli.md)
- [Caching & Data](../guides/caching.md)
- [User Configuration](../guides/configuration.md)
- [Output Artifacts](../reference/output-artifacts.md)
- [Embedding Runtime](../reference/embedding-runtime.md)

## Public Release Readiness

- Added intentionally small GitHub Actions coverage for lint/format,
  representative Linux and macOS tests, and a no-extras CLI smoke. The
  package-installing jobs fetch Git tags so `setuptools-scm` resolves an accurate
  version.
- Added CONTRIBUTING.md, AGENTS.md, issue templates, and a pull request template.
- Added authenticated Semantic Scholar pacing, a key-less usage notice, and
  full-jitter retry handling that honors `Retry-After`.
- Distinguished unknown paper identifiers from unavailable or rate-limited
  Semantic Scholar requests across search and graph strategies.
- Preserved partial candidate-source success while recording source outcomes in
  exports; total source outages now fail instead of producing plausible
  seed-only graphs.
- Broadened recommendation fallback for successful empty responses without
  doubling retry traffic when the primary service is unavailable.

## Configuration and Search

- Added persistent `config.toml` defaults and API credentials through
  `citemesh config`; see [User Configuration](../guides/configuration.md).
- Made invalid TOML and non-UTF-8 config files non-blocking for ordinary commands,
  while config mutations fail closed instead of overwriting unreadable content.
- Unified the default macOS and Linux cache root under `~/.cache/citemesh` and
  preserved `config.toml` during `citemesh cache clear`.
- Added local semantic search over the active retrieval cache and the
  `auto`, `local`, and `s2` search modes documented in
  [CLI Usage](../guides/cli.md).
- Added full DOI and arXiv URL normalization, including arXiv version stripping.

## Semantic Discovery

- Made Semantic Scholar candidate sourcing the default corpus-free path for
  embedding and hybrid builds. The arXiv corpus remains opt-in.
- Added explicit candidate-source status tracking and conservative paper identity
  reconciliation across DOI, arXiv, Semantic Scholar, and metadata evidence.
- Split asymmetric retrieval vectors from symmetric graph-similarity vectors and
  gave each representation its own formatter and cache identity; see
  [Embedding Runtime](../reference/embedding-runtime.md).
- Resolved embedding model profiles from local checkpoint metadata and runtime-active fallbacks, with an explicit override and profile-schema cache partitioning.
- Persisted hybrid candidate vectors and changed hybrid selection from a
  citation-first add-on to merged candidate reranking with an overlap boost.
- Selected capped arXiv corpora by submission chronology encoded in paper IDs
  instead of snapshot row position.
- Updated hybrid depth defaults after multi-seed evaluation and follow-up manual
  review; the measurements and decisions are recorded in
  [Defaults Tuning Study](../reference/defaults-tuning-study.md).

## Runtime and Caching

- Added explicit CUDA, MPS, and CPU device resolution, verified bfloat16 autocast
  for supported accelerator profiles, and float32 fallback for unsupported
  runtimes.
- Split torch dependency floors by platform and made `torch.compile` an opt-in,
  profile-gated optimization.
- Hardened automatic checkpoint loading with live dtype verification, native-only
  CUDA bfloat16 gating, profile-scoped attention selection, encode-scoped TF32,
  and a real eager retry for lazy compile failures.
- Required Transformers 4.57 or newer for EmbeddingGemma, enforced the profile-specific floor before model loading, and invalidated caches created before bidirectional attention was guaranteed.
- Added immutable model-artifact fingerprints and runtime-active checkpoint
  identity to persistent cache namespaces.
- Moved embedding storage to SQLite metadata plus HDF5 vector datasets with
  `int8` and `float32` storage, optional binary prefiltering, and fail-closed
  calibration and integrity checks.
- Added resumable full-corpus hydration, upstream growth reconciliation, and
  cache-native search. Detailed storage and hydration behavior is in
  [Caching & Data](../guides/caching.md).
- Hardened reference-cache validation and atomic replacement, and reduced
  repetitive empty-hit and quantization-warning logs.

## Export and Visualization

- Added one export pipeline: static PNG rendering plus Pyvis HTML, Plotly HTML,
  dashboard, JSON, CSV, BibTeX, and GraphML through `GraphExporter`.
- Replaced per-result dashboard directories with reusable two-file collections and
  portable versioned graph packages; see
  [Output Artifacts](../reference/output-artifacts.md).
- Added browser-side multi-file import, collection export, reading-list actions,
  and non-destructive migration from the former dashboard manifest layout.
- Unified themes and year color scales across static and interactive renderers,
  normalized edge styling, and aligned dashboard re-render behavior with the
  initial Python figure.
- Reworked disconnected-component packing, label selection, backend-independent
  collision handling, and viewport fitting for static exports.
- Preserved application-selected Matplotlib backends and isolated MacOSX static
  rendering on an Agg canvas without changing global plotting state.
- Added `darkreader-lock` and `color-scheme` metadata to HTML exports.

## CLI and Packaging

- Added multi-export and `--export all`, deterministic output grouping, and
  per-run config sidecars.
- Added strategy-aware flag validation, explicit cache-overwrite acknowledgement,
  plain-text log files, and deterministic layout reuse across compatible exports.
- Moved embedding and interactive visualization dependencies behind optional
  extras while keeping the core CLI importable without them.
- Made timestamps opt-in and clarified `--seed` as a layout/export setting.

## Breaking and Maintenance Changes

- Removed deprecated `GraphBuilderStrategy.exponential_temporal_decay`, strategy
  `random_seed` constructor arguments, and legacy module shims.
- Removed unsupported float16 storage and `szip` cache compression.
- Consolidated duplicated configuration, paper-identity, availability, similarity,
  formatting, cache-identity, atomic-write, and visualization contracts.
- Removed dead cache observability fields, stale SQLite indexes, shadow test
  serializers, redundant test cases, and the network-dependent integration smoke.
