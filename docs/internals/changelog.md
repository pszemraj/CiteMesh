# Changelog & Key Improvements

Notable changes from the early script-based prototypes to the current package layout. For current usage details, see:

- [CLI Usage](../guides/cli.md)
- [Caching & Data](../guides/caching.md)
- [User Configuration](../guides/configuration.md)
- [Environment Variables](../reference/environment.md)
- [Embedding Runtime](../reference/embedding-runtime.md)

## Public Release Readiness

- Added intentionally small GitHub Actions CI: ruff lint/format, representative Ubuntu + macOS tests on Python 3.10/3.13 (CPU-only torch on Linux and the platform wheel on macOS), and a no-extras install smoke that locks in the lazy-import contract for the core CLI. Package-installing jobs fetch tags for accurate `setuptools-scm` versions; lint-only checkout stays shallow.
- Key-aware Semantic Scholar rate limiting: authenticated clients pace at 1 request/second, anonymous clients stay at 0.5. A one-time INFO notice on key-less runs points at the free API key signup.
- Smarter retry backoff (tenacity): direct REST calls (search, recommendations) now retry with full-jitter exponential backoff floored at the server's `Retry-After` and capped at 60s, instead of sleeping a flat `Retry-After: 2` on every attempt. The library-mediated call wrapper shares the same policy.
- Fixed empty recommendation results for classic seed papers: the S2 recommendations endpoint's default "recent" candidate pool returns nothing for older landmark papers (e.g. arXiv:1706.03762), which silently emptied the recommendation strategy and the hybrid/embedding candidate pools. CiteMesh retries the broader `all-cs` pool only after a successful empty response; an unavailable primary request does not trigger a second retry cycle.
- Seed-paper fetch failures now distinguish "identifier unknown to Semantic Scholar" (`ValueError`) from "API rate-limited/unreachable after retries" (`SemanticScholarUnavailableError`) for citation, recommendation, and standalone embedding seeds.
- Paper search gets the same treatment: `citemesh search` and free-text query seeds no longer report "No results found" when the search API was actually rate-limited or unreachable — exhausted retries now surface as a `SemanticScholarUnavailableError` with the free-key pointer.
- Added local semantic search to `citemesh search`: mode `local` searches the locally cached embeddings (candidate vectors accumulated across builds, or a hydrated corpus) via the cache-native search API. The query is encoded in the model's query prompt space and ranked with cosine scores; no Semantic Scholar traffic. Targets the same cache namespace a flagless build writes to, honoring `config.toml` defaults, with `--model`/`--device` overrides (which imply local mode).
- Made `citemesh search` mode-aware: `--mode {auto,local,s2}` with `auto` as the default — local semantic search when the cache has vectors, Semantic Scholar keyword search otherwise, logging which backend ran. The preference persists via `citemesh config set defaults.search_mode <mode>` (explicit flag wins). Explicitly requested local mode fails with build guidance on an empty cache instead of silently falling back.
- Rewrote the README for public beta (reference tool comparison, macOS/MPS support statement, S2 key guidance) and added CONTRIBUTING.md, AGENTS.md, and issue/PR templates.
- Scoped the stray run-config ignore to root-level `/*.yaml`, so nested YAML files are never hidden from contributors.

## User Configuration

- Added a persistent user config system: `config.toml` at the cache root plus a `citemesh config` subcommand (`list`/`get`/`set`/`unset`/`path`). A whitelisted `[defaults]` table overrides built-in defaults for most build flags (for example `semantic_source = "arxiv-corpus"` to restore the corpus default), and `[api] s2_api_key` supplies a Semantic Scholar key when the `S2_API_KEY` environment variable is absent. Precedence: explicit CLI flag > environment variable > config.toml > built-in default.
- Config-supplied defaults outrank tuned implicit defaults (hybrid budget knobs) but never count as explicit flags for strategy gating. Explicit corpus-only and candidate-only flags symmetrically imply their source mode over a conflicting config default, while settings for the inactive mode stay inert.
- Malformed TOML and invalid UTF-8 warn and yield an empty config for ordinary commands; config mutations fail closed rather than overwriting unreadable content.
- Added `--no-streaming` so an enabled `defaults.streaming` preference can be disabled for one invocation while preserving explicit CLI-over-config precedence.
- Unified the macOS cache root with Linux: `~/.cache/citemesh` (HuggingFace-style) instead of `~/Library/Caches/citemesh` (pre-release breaking change; `citemesh cache scan` hints when the legacy directory still exists).
- `citemesh cache clear` now preserves `config.toml` while deleting cache payloads.

## Semantic Sources

- Candidate-only ("corpus-free") semantic sourcing is now the DEFAULT for the embedding and hybrid strategies (`--semantic-source candidates`): candidates come from S2 seed neighbors (references/citations/recommendations, capped by `--candidate-pool-size`) and only those abstracts are embedded locally, with vectors persisted incrementally in a candidate-scoped cache namespace.
- The arXiv corpus path remains available via `--semantic-source arxiv-corpus`; corpus-only flags imply it when `--semantic-source` is omitted, so existing invocations keep working.
- Candidate mode stores embeddings as float32 (int8 calibration is computed during corpus hydration only); explicit `--storage-precision int8` now requires `arxiv-corpus`.
- Hybrid candidate rerank vectors now persist through the embedding cache in candidate mode instead of being re-encoded every run (corpus mode keeps the in-memory path so corpus row counts stay undistorted).
- Hybrid candidate mode now applies `--candidate-pool-size` to its recommendation-only semantic request instead of bypassing the configured pool cap.
- Capped corpus hydration now selects the `--corpus-size` most recently SUBMITTED papers, ranked by the submission chronology encoded in each arXiv ID (new-style `YYMM.NNNNN` and old-style `archive/YYMMNNN`). Positional slicing cannot express this: the arXiv snapshot ships ordered by `update_date` descending (head = recently revised papers of any age, tail = the pre-2007 ID block), so neither end of the split is "the newest papers". Selection is a bounded top-N pass over the ID column (or record stream when streaming); sources without parseable arXiv IDs fall back to the head of the split with a warning. The int8 calibration prepass samples the same newest window. The hydration corpus-size token now encodes the slice policy (`newest:N`), so caches hydrated under the old positional policy rehydrate instead of silently serving stale windows. Interrupted capped hydrations still restart from zero; capped-resume support remains an open follow-up.
- The `datasets` dependency is now only required for `arxiv-corpus` mode.

## Runtime & Devices

- Added explicit device resolution with a `--device {auto,cuda,mps,cpu}` flag for embedding/hybrid strategies: `auto` prefers CUDA, then MPS (Apple Silicon), then CPU; explicit unavailable devices fail fast at parse time.
- Added first-class MPS support: EmbeddingGemma uses bf16 autocast on verified MPS runtimes (torch >= 2.13) with `sdpa` attention; unsupported runtimes and CPU stay float32.
- Cache namespaces track compute dtype (not device), so bf16 caches built on CUDA and MPS interoperate; warm-on-GPU-then-copy-to-Mac now works.
- TF32 configuration is now gated on the resolved device: `--device cpu` on a CUDA host no longer flips global TF32 backend state (bug fix).
- `--torch-compile` is now declined on CPU (pure warm-up cost for CLI runs) and marked experimental on MPS (Inductor/Metal, eager fallback on failure).
- Candidate-mode `--torch-compile` now runs without waiting for corpus hydration metadata; cold-cache deferral remains scoped to arXiv corpus hydration.
- Split the torch dependency floor by platform: `>=2.9` on Linux/Windows, `>=2.13` on macOS. Declared the previously transitive `huggingface_hub` dependency explicitly.
- Static exports preserve the application's selected Matplotlib backend. When the active backend is the main-thread-only MacOSX GUI backend, CiteMesh attaches an Agg canvas only to its own static figure instead of mutating global plotting state.

## Breaking Changes

- Removed deprecated `GraphBuilderStrategy.exponential_temporal_decay`.
- Removed `random_seed` constructor arguments from strategy builders.
- Clarified `--seed` CLI semantics as layout/export determinism only.

## Export & Visualization

- Replaced per-result dashboard shells and the path-based dashboard manifest with a reusable two-file collection: `dashboard.html` plus the versioned, portable `dashboard.citemesh.json` package. Dashboard-only builds now create exactly those two root files, and repeated builds upsert the `(strategy, seed_id)` slot; explicitly requested other formats keep their per-seed artifacts and sidecar.
- Added explicit collection/graph contracts (`kind: "citemesh-dashboard-collection"` and `kind: "citemesh-graph"`, both schema v1), portable per-result build settings, non-destructive migration from valid legacy `dashboard.manifest.json` entries on the next build, browser multi-file **Add Results**, and **Export Collection**.
- Kept collection delivery intentionally local and self-contained: the viewer embeds the current package snapshot for reliable `file://` use, with no database, server, service worker, ZIP layer, or additional CI job.
- Made dark the built-in visualization default and taught `--theme auto` to read the active macOS interface appearance before terminal fallbacks; responsive dashboards now protect edge labels and footer overlays, keep detail abstracts readable with expanded filters, and let the tall wrapped toolbar scroll away at phone widths.
- Added `GraphExporter` to emit PNG, Pyvis HTML, Plotly HTML, Dashboard HTML, JSON, CSV, BibTeX, and GraphML from one graph object.
- Centralized theming (light, dark, solarized, auto) for static and interactive exporters.
- Added adaptive metadata callout contrast based on active theme.
- Switched node coloring to continuous gradients for consistent year-based styling.
- Restricted legacy dashboard-manifest migration reads to payload paths that stay under the collection root.
- HTML exports (dashboard, Plotly, pyvis) now carry a `darkreader-lock` meta tag plus a transparent-overlay CSS guard: the Dark Reader browser extension repaints Plotly's transparent overlay SVGs with an opaque background, which hid the entire dashboard graph. Verified live in Chrome with Dark Reader installed — the lock disengages the extension and the graph renders in the native theme. Also fixed the paper-count label pluralization ("1 paper").
- HTML exports additionally declare the standards-based `color-scheme` meta + CSS property (dark or light per theme) so Chrome's Auto Dark Mode and other well-behaved darkening extensions skip repainting; this complements the Dark Reader-specific lock (Plotly/vis.js offer no such signal themselves — the page-level declaration is the mechanism).
- Dashboard hover tooltips are now theme-styled glance cards (panel background and border instead of raw marker color), wrap long titles instead of spanning the full pane, and add venue plus a relation line ("referenced by seed", "cites seed", "semantic match") so hover answers "should I look at this paper". Collection selection and imported-result rebuilds retain the same hover contract.
- Fixed a misleading dashboard legend that claimed nodes were colored by provenance (seed/citation/semantic/both) while they are colored by a publication-year gradient; the legend now shows the real encodings (seed ring, older-to-newer gradient, size = citations) and the year timeline reuses the exact node colorscale.
- Dashboard edge opacity/width is now min-max normalized per graph (raw hybrid weights cluster in 0.55-0.95 and previously clamped to a flat 0.6 alpha, rendering every edge identically); collection selection and imported-result rebuilds use the same scale and viewport padding as the initial Python figure.
- Added a saved-papers reading list to the dashboard: star toggles on list rows and in the details panel, a Saved filter chip, localStorage persistence per strategy and seed, plus "Saved BibTeX" download and "Copy Saved Links" (markdown list). Saved IDs are intersected with the active payload before badge, filter, or export use.
- The seed's "Why This Paper" panel no longer prints a meaningless self-relevance score and self-path; it states the node is the seed and keeps its strongest links.

## Graph Rendering

- Unified layout-oriented rendering pipeline across outputs.
- Static layouts now scaffold and size-pack disconnected groups, wrap singleton/small components across deterministic rows, place the largest component first, orient long extents horizontally, cap non-seed labels by citation priority, suppress text overlaps through a backend-independent renderer, and fit a landscape viewport to the graph so dense exports use the canvas and remain legible.
- Dashboard graphs now use spatially distributed priority labels, larger nodes, tighter plot bounds, quieter background edges, and collapsed-by-default filters so dense interactive exports remain readable in three-pane browser layouts.
- Added normalized edge opacity/width scaling for weight readability.
- Routed integration-test outputs to temporary paths to reduce fixture confusion.
- Canonicalized node/edge ordering and deterministic tie-breaks for stable layout artifacts.

## Persistent Caching

- Introduced `EmbeddingCache` with SQLite metadata + HDF5 matrix storage.
- Restricted persistent embedding storage to `int8` or `float32`; float16 is no longer accepted by cache, CLI, or user-config storage settings.
- Changed embedding precision policy to automatic checkpoint weight-dtype loading, with bf16 compute available only through verified autocast on supported CUDA/MPS runtimes; unsupported or rejected autocast falls back to fp32 compute.
- Existing embedding caches now retain their physical HDF5 compression filter and level when reopened with different compression flags; requested compression applies to new or explicitly rebuilt payloads without changing semantic cache identity.
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
- **Embedding**: cache fingerprint enforcement now follows the runtime-active checkpoint identity after model fallback selection and runs before corpus, candidate, and local-search cache access, preventing stale cross-checkpoint reuse in shared namespaces.
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
