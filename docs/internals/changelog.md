# Changelog & Key Improvements

Notable changes from the early script-based prototypes to the current package layout.
For current usage details, see:

- [CLI Usage](../guides/cli.md)
- [Caching & Data](../guides/caching.md)
- [User Configuration](../guides/configuration.md)
- [Environment Variables](../reference/environment.md)
- [Embedding Runtime](../reference/embedding-runtime.md)

## User Configuration

- Added a persistent user config system: `config.toml` at the cache root plus
  a `citemesh config` subcommand (`list`/`get`/`set`/`unset`/`path`). A
  whitelisted `[defaults]` table overrides built-in defaults for most build
  flags (for example `semantic_source = "arxiv-corpus"` to restore the corpus
  default), and `[api] s2_api_key` supplies a Semantic Scholar key when the
  `S2_API_KEY` environment variable is absent. Precedence: explicit CLI flag >
  environment variable > config.toml > built-in default.
- Config-supplied defaults outrank tuned implicit defaults (hybrid budget
  knobs) but never count as explicit flags for strategy gating or corpus-mode
  implication, so a global default cannot break unrelated strategies.
- Unified the macOS cache root with Linux: `~/.cache/citemesh`
  (HuggingFace-style) instead of `~/Library/Caches/citemesh` (pre-release
  breaking change; `citemesh cache scan` hints when the legacy directory still
  exists).
- `citemesh cache clear` now preserves `config.toml` while deleting cache
  payloads.

## Semantic Sources

- Candidate-only ("corpus-free") semantic sourcing is now the DEFAULT for the
  embedding and hybrid strategies (`--semantic-source candidates`): candidates
  come from S2 seed neighbors (references/citations/recommendations, capped by
  `--candidate-pool-size`) and only those abstracts are embedded locally, with
  vectors persisted incrementally in a candidate-scoped cache namespace.
- The arXiv corpus path remains available via `--semantic-source arxiv-corpus`;
  corpus-only flags imply it when `--semantic-source` is omitted, so existing
  invocations keep working.
- Candidate mode stores embeddings as float32 (int8 calibration is computed
  during corpus hydration only); explicit `--storage-precision int8` now
  requires `arxiv-corpus`.
- Hybrid candidate rerank vectors now persist through the embedding cache in
  candidate mode instead of being re-encoded every run (corpus mode keeps the
  in-memory path so corpus row counts stay undistorted).
- Capped corpus hydration now warns that it takes the first `--corpus-size`
  rows of the split (typically the oldest arXiv records) and that interrupted
  capped hydrations restart from zero; a newest-slice policy and capped-resume
  support remain open follow-ups.
- The `datasets` dependency is now only required for `arxiv-corpus` mode.

## Runtime & Devices

- Added explicit device resolution with a `--device {auto,cuda,mps,cpu}` flag for
  embedding/hybrid strategies: `auto` prefers CUDA, then MPS (Apple Silicon),
  then CPU; explicit unavailable devices fail fast at parse time.
- Added first-class MPS support: EmbeddingGemma loads bf16 weights on MPS
  (torch >= 2.13) with `sdpa` attention and autocast off; float16-safe profiles
  run fp16. CPU stays float32.
- Cache namespaces track compute dtype (not device), so bf16 caches built on
  CUDA and MPS interoperate; warm-on-GPU-then-copy-to-Mac now works.
- TF32 configuration is now gated on the resolved device: `--device cpu` on a
  CUDA host no longer flips global TF32 backend state (bug fix).
- `--torch-compile` is now declined on CPU (pure warm-up cost for CLI runs) and
  marked experimental on MPS (Inductor/Metal, eager fallback on failure).
- Split the torch dependency floor by platform: `>=2.9` on Linux/Windows,
  `>=2.13` on macOS. Declared the previously transitive `huggingface_hub`
  dependency explicitly.
- Pinned a headless matplotlib backend (`Agg`) for static exports unless
  `MPLBACKEND` is set, avoiding the main-thread-only MacOSX GUI backend.

## Breaking Changes

- Removed deprecated `GraphBuilderStrategy.exponential_temporal_decay`.
- Removed `random_seed` constructor arguments from strategy builders.
- Clarified `--seed` CLI semantics as layout/export determinism only.

## Export & Visualization

- Added `GraphExporter` to emit PNG, Pyvis HTML, Plotly HTML, Dashboard HTML, JSON, CSV, BibTeX, and GraphML from one graph object.
- Centralized theming (light, dark, solarized, auto) for static and interactive exporters.
- Added adaptive metadata callout contrast based on active theme.
- Switched node coloring to continuous gradients for consistent year-based styling.
- Restricted dashboard collection bundle payload reads to manifest paths that stay under the collection root.

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
- Resumed interrupted full-corpus hydration runs from the cached row boundary when SQLite/HDF5 payload rows and hydration metadata still match the requested source/split, instead of immediately discarding partial progress.
- Hardened reference-cache reuse against unreadable/non-object JSON entries and improved atomic write durability with parent-directory fsync.
- Clarified embedding-cache clear logs to label replaced payload metadata as `cached_*`, avoiding confusion during capped-to-full corpus rebuilds.
- Suppressed repeated per-batch int8 saturation warnings after the first warning in a run while continuing to persist cumulative clipping stats.
- Suppressed empty-reference cache hit debug spam so long citation/hybrid runs no longer emit one `Loaded 0 cached references` line per paper.

## Strategy Evolution

- **Citation**: moved to real bibliographic coupling with reference-aware similarity factors.
- **Embedding**: expanded multi-factor similarity and added citation-count hydration for top matches.
- **Embedding**: added best-effort `torch.compile` path for EmbeddingGemma internals with safe fallback.
- **Embedding**: switched eager TF32 runtime configuration to the PyTorch 2.9+ `torch.backends.fp32_precision` API.
- **Embedding**: for torch 2.9/2.10 CUDA compile paths, switched TF32 control to `torch.set_float32_matmul_precision("high")` so `torch.compile` remains available without tripping the mixed TF32 API conflict in release-branch Inductor.
- **Embedding**: changed default checkpoint to `unsloth/embeddinggemma-300m` (ungated) and added automatic fallback to `google/embeddinggemma-300m` for default-revision loads.
- **Embedding**: citation-count enrichment logs now show bounded target counts and render a visible progress bar on TTY runs.
- **Embedding**: citation-count enrichment now batches Semantic Scholar paper lookups before falling back to single-paper retries for unresolved IDs.
- **Embedding**: cache fingerprint enforcement now follows the runtime-active checkpoint identity after model fallback selection, preventing stale cross-checkpoint reuse in shared namespaces.
- **Embedding/Hybrid**: semantic enrichment reuses the citation branch seed metadata when available, avoiding a second Semantic Scholar fetch for the same seed paper during hybrid runs.
- **Hybrid**: moved from citation-first semantic add-on behavior to merged citation+semantic candidate reranking with semantic-only cap enforcement.
- **Hybrid**: default depth targets were raised to `25/25/25` (references/citations/semantic cap) after the February 2026 sweep to improve foundational-paper recovery while keeping recent-paper quality high.
- **Hybrid**: updated omitted-budget defaults again after the February 2026 follow-up review; see [Defaults Tuning Study](../reference/defaults-tuning-study.md) for the current values and rationale.
- **Recommendation**: added recommendation-based discovery with direct endpoint handling and rate-limit-aware behavior.
- Unified edge gating for citation/recommendation under `--similarity-threshold`.
- Added explicit streaming split validation for embedding mode.

## CLI & Developer Experience

- Added `--export all` and multi-export (`-e json -e dashboard`) for selective format combinations.
- Improved parser validation and test coverage around CLI ergonomics.
- Added embedding/hybrid cache controls: `--storage-precision`, binary prefilter toggles, binary rescore multiplier, calibration sample size, and cache compression knobs.
- Added explicit overwrite acknowledgement for embedding cache rebuilds (`--force-rebuild-cache` + `--overwrite-cache`) with default confirmation prompts.
- Added optional cache-clear rationale flags (`--cache-overwrite-reason`, `cache clear --reason`) and standardized destructive-clear logs with file/size snapshots and large-cache warnings.
- Switched default Rich CLI width selection to auto-size on TTYs while keeping a fixed fallback for redirected output, preventing double-wrapped local terminal logs.
- Added shared `--log-file` CLI support for build/search/cache commands so verbose runs can capture plain-text diagnostics without shell redirection, while keeping debug-only chatter out of the Rich console.
- Clamped noisy dependency debug loggers (`filelock`, `urllib3`, `matplotlib`, `semanticscholar`, and similar) to `WARNING` so `--log-file` traces stay focused on CiteMesh internals.
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
