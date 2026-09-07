# CLI Usage Guide

Build paper graphs with `recommendation`, `citation`, `embedding`, or `hybrid`.

Related docs:

- Cache layout and hydration: [Caching & Data](caching.md)
- Persistent user defaults: [User Configuration](configuration.md)
- Output files and sidecar schema: [Output Artifacts](../reference/output-artifacts.md)
- Embedding runtime policy: [Embedding Runtime](../reference/embedding-runtime.md)
- Defaults parameter study: [Defaults Tuning Study](../reference/defaults-tuning-study.md)
- Environment variables: [Environment Variables](../reference/environment.md)

Installation and optional extras are covered in [README](../../README.md).

## Help and Console Output

Every command supports `-h` / `--help`, including nested commands:

```bash
citemesh --help
citemesh build --help
citemesh cache --help
citemesh cache clear --help
citemesh config set --help
```

Help uses Rich styling, compact usage lines, grouped options, and examples.
Build settings are grouped into graph, output, citations/references, semantic
discovery, embedding runtime, arXiv corpus, embedding cache, and hybrid expansion.
Help describes built-in defaults; use `citemesh config list` to see saved overrides.

Help adapts to terminal width, up to 110 columns. Redirected help is plain text
without ANSI styling under normal terminal detection. Set `NO_COLOR=1` to disable
colors in an interactive terminal (emphasis such as bold may remain).
`--log-width` controls result tables and logs; help uses the terminal width.

Cache and config listings use compact tables. Search results show titles with
authors underneath, followed by full paper IDs for copying into a build command.
Queries, paper titles, and config values are displayed literally, including square
brackets. `config get` and `config path` return raw, unwrapped values on stdout for
shell substitution; operational logs go to stderr.

## Basic Invocation

```bash
citemesh build "<paper-id>" [options]
```

Semantic Scholar operations allow up to **30 total attempts**, including the initial request. Both SDK and direct HTTP calls use Tenacity with exponential full jitter: the random wait ceiling doubles from 2 seconds (4 seconds for HTTP 429) up to 60 seconds. A numeric `Retry-After` header sets the minimum wait beyond the jitter ceiling, honored up to 300 seconds per delay. This budget applies per operation, not to the whole build. There is deliberately no elapsed-time deadline: long retries favor finishing resumable builds, and the caps bound each delay rather than total waiting time. A prolonged outage can take many minutes before it is reported; any single wait of 30 seconds or longer is announced at the default log level. Invalid request parameters and rejected credentials fail immediately. Ctrl+C interrupts retries. Full retry details appear at `--log-level debug`.

Other command groups:

```bash
# Find seed paper IDs (auto: local semantic search when your cache has
# embeddings, otherwise Semantic Scholar keyword search)
citemesh search "<query>" [--limit N|-n N]

# Force a mode: local cache semantic search or S2 keyword search
citemesh search "<query>" --mode local [--model M] [--model-profile P] [--device D]
citemesh search "<query>" --mode s2

# Cache management commands
citemesh cache scan
citemesh cache scan --log-level debug
citemesh build "arxiv:1706.03762" --strategy hybrid --log-level debug --log-file out/run.log
citemesh cache clear [--yes] [--reason "<text>"]

# Persistent user configuration (config.toml at the cache root)
citemesh config list
citemesh config set defaults.semantic_source arxiv-corpus
citemesh config get defaults.semantic_source
citemesh config unset defaults.semantic_source
citemesh config path
```

For cache path/layout/hydration details, see [Caching & Data](caching.md). In non-interactive shells, `citemesh cache clear` requires `--yes`. `citemesh cache clear` never deletes `config.toml`. In non-interactive embedding/hybrid runs, `--force-rebuild-cache` requires `--overwrite-cache`.

Persistent defaults for most build flags can be stored with `citemesh config`.
Supported keys, value forms, and precedence are documented in
[User Configuration](configuration.md).

## Accepted Identifiers

- DOI (`10.1038/nature14539`)
- DOI with prefix (`doi:10.1038/nature14539`)
- DOI URL (`https://doi.org/10.1038/nature14539`)
- arXiv ID (`arxiv:1706.03762`; version suffixes like `v5` are normalized away)
- bare arXiv-like IDs (for example `1706.03762`) may work when Semantic Scholar resolves them
- arXiv URL (`https://arxiv.org/abs/1706.03762`, `https://arxiv.org/abs/arXiv:1706.03762`, `https://arxiv.org/pdf/1706.03762.pdf`; `vN` suffixes are normalized away)
- Semantic Scholar paper ID
- Free-form text query for the embedding strategy when Semantic Scholar reports
  that the input is not a known paper. Service outages remain errors.

## Core Options

| Flag | Description | Default |
| --- | --- | --- |
| `--strategy`, `-s` | `recommendation`, `citation`, `embedding`, or `hybrid` | `recommendation` |
| `--max-papers`, `-p` | Maximum nodes in final graph (seed included) | `40` (`hybrid`: implicit `45` when omitted) |
| `--refresh-paper-cache` | Fetch fresh Semantic Scholar paper metadata and citation counts without rebuilding embeddings; failed refreshes retain cached metadata | disabled |
| `--spring-iterations`, `-i` | Iterations used only for spring-layout fallback | `100` |
| `--dpi`, `-d` | PNG output resolution | `150` |
| `--seed` | Seed for layout computation used by layout-based exports (`png`, `plotly`, `dashboard`, `json`) | deterministic built-in seed |
| `--include-timestamp` | Include generation time in output metadata | disabled |
| `--export`, `-e` | `png`, `html`, `plotly`, `dashboard`, `json`, `csv`, `bibtex`, `graphml`, or `all`; repeat flag for multiple (e.g. `-e json -e dashboard`) | `png` |
| `--theme` | `light`, `dark`, `solarized`, `auto`; `auto` checks explicit environment hints before macOS appearance | `dark` |
| `--output`, `-o` | Output path, or collection root for normal dashboard exports; an explicit `*.dashboard.html` path requests standalone mode | auto-generated `out/` collection root for dashboard, otherwise a per-paper folder |
| `--log-level` | Console logging level (`debug`, `info`, `warning`, `error`) | `info` |
| `--log-width` | Rich console wrap width in columns (`0` uses terminal width on TTYs and a stable redirected fallback) | `0` |
| `--log-file` | Optional plain-text log file path (overwrites existing file) | disabled |

Output-path normalization, file naming, and sidecar placement are defined in [Output Artifacts](../reference/output-artifacts.md).

For repository-local runs, omit `--output` to use the ignored `out/` directory.
Dashboard builds maintain `out/dashboard.html` and `out/dashboard.citemesh.json`,
and always save `<strategy>.json` plus `<strategy>.config.json` under
`out/<slug>-<hash>/`. Different seeds retain separate files and accumulate in the
collection; rerunning the same seed and strategy replaces that result. Other
requested formats go alongside the per-paper JSON.

`--log-level`, `--log-width`, and `--log-file` are shared command options and are accepted for `build`, `search`, `cache`, and `config` command trees (including `cache scan` / `cache clear`).

`--seed` controls shared layout generation for `png`, `plotly`, `dashboard`, and `json` exports. Pyvis `html` exports use vis.js browser physics and do not consume this precomputed layout.

Numeric validation:

- `build <paper-id>` and `search <query>` require non-empty strings.
- `--max-papers`, `--spring-iterations`, `--dpi`, `--corpus-size`, `--top-k`, `--truncate-dim`, `--binary-rescore-multiplier`, `--calibration-sample-size`, and `search --limit` must be at least `1`.
- `--max-citations` and `--max-references` must be at least `0`.
- `--calibration-sample-size` is valid only with `--storage-precision int8`.
- `--cache-compression-level` must be at least `0` and is valid only with `--cache-compression gzip`.
- `--max-semantic` must satisfy `0 <= max-semantic <= max-papers - 1` (hybrid strategy).
- `--similarity-threshold` and `--min-semantic-similarity` must be finite floats between `0.0` and `1.0`.

## Strategy-Specific Flags

Strategy behavior and tradeoffs are described in [Strategies Guide](strategies.md). Flag contracts are listed here.

Build command options are strategy-scoped. If you pass a flag that is not supported for the selected `--strategy`, CiteMesh exits with a CLI error instead of silently ignoring it.

### Cross-Strategy Behavior

- `--similarity-threshold` applies to `recommendation` and `citation` strategies as the minimum edge similarity threshold (default `0.2`).
- `--no-references` applies to `recommendation`, `citation`, and the citation branch of `hybrid`.
- `--refresh-reference-cache` applies to `recommendation`, `citation`, and the citation branch of `hybrid` to bypass persisted reference-cache reads.
- `embedding` uses embedding-specific controls (`--dataset-split`, `--corpus-size`, `--all-corpus`, `--truncate-dim`, `--streaming`, `--top-k`, storage/cache flags below).
- The embedding branch of `hybrid` reuses embedding controls except `--top-k` (hybrid edge pruning follows its own policy).

### Citation Strategy

- `--max-citations`, `-c`: limit number of citing papers (default `25`; hybrid implicit default `45`)
- `--max-references`, `-r`: limit number of referenced papers (default `25`; hybrid implicit default `12`)

### Embedding Strategy

- `--model`, `-m`: sentence-transformer model name (default `unsloth/embeddinggemma-300m`; alternate current checkpoint: `google/embeddinggemma-300m`)
- `--model-profile {auto,default,embeddinggemma}`: task/runtime contract selection. `auto` recognizes known Hub aliases and compatible local checkpoint metadata; use an explicit profile for stripped local fine-tune exports.
- `--model-revision`: optional model revision token (branch/tag/commit) for hub-backed models
- `--dataset-split`: HuggingFace split (default `train`; sliced forms like `train[:5%]` are supported in non-streaming mode and bound the rows exposed to CiteMesh after dataset preparation)
- `--corpus-size`: maximum papers to embed/cache after scanning the selected split to select the newest submissions by arXiv ID (default `50000`); it does not cap that selection scan
- `--all-corpus`: remove corpus-size cap and process the full selected split
- `--all-corpus` applies within the selected `--dataset-split`; `--dataset-split train --all-corpus` means "all of `train`", not "every split published by the dataset"
- If a same-model cache namespace was previously hydrated with a capped corpus, `--all-corpus` rebuilds that namespace; cache-clear logs label the replaced payload as `cached_*` to distinguish it from the new target.
- `--all-corpus` cannot be combined with an explicit `--corpus-size` value
- `--top-k`, `-k`: strict per-node edge cap during embedding-graph pruning (default `4`)
- `--min-semantic-similarity`: semantic cosine required for embedding/hybrid graph edges (default `0.74`). Hybrid can also admit pairs with shared references. Set a persistent override with `citemesh config set defaults.min_semantic_similarity VALUE`. This setting changes graph eligibility without rebuilding embeddings.
- `--truncate-dim`: optional embedding output-dimension truncation (for
  EmbeddingGemma: `768`, `512`, `256`, `128`; omitted uses the model profile's
  recommendation, **`512` for EmbeddingGemma**). See the
  [dimension study](../reference/defaults-tuning-study.md#embedding-dimensions-september-2026)
  for the retrieval, storage, and search-time tradeoffs.
- `--streaming` / `--no-streaming`: stream the HuggingFace dataset or load cached shards. Streaming requires a non-sliced split (for example `train`); the negative form overrides an enabled `defaults.streaming` config value for one run.
- `--force-rebuild-cache`: clear and rebuild embedding cache for this model before running (requires confirmation by default)
- `--overwrite-cache`: acknowledge destructive overwrite for `--force-rebuild-cache` and skip interactive confirmation (required for non-interactive/scripting workflows)
- `--cache-overwrite-reason`: optional rationale string logged when `--force-rebuild-cache` clears embedding cache state
- `--storage-precision {int8,float32}`: persistent embedding-cache precision. Corpus mode defaults to `int8`; the default candidates mode normalizes an implicit `int8` setting to `float32` because corpus calibration is unavailable.
- `--binary-prefilter` / `--no-binary-prefilter`: enable/disable binary Hamming prefilter for quantized search. It defaults on for int8 corpus mode and is normalized off in candidates mode. Explicit `--binary-prefilter` requires `--storage-precision int8`.
- `--binary-rescore-multiplier`: oversampling factor for binary-prefilter rescoring.
  The int8 default is `8`; float32 and candidate modes normalize it to an unused
  effective value of `1`. Explicit use requires `--storage-precision int8`.
- `--calibration-sample-size`: calibration sample size used to compute int8 ranges (default `2000`; explicit use requires `--storage-precision int8`)
- `--encode-batch-size`: embedding-model encode batch size used during hydration/search (default `32`)
- `--cache-compression`: HDF5 compression filter for cache datasets (`gzip`, `lzf`; default `gzip`)
- `--cache-compression-level`: HDF5 compression level for cache datasets (gzip
  default `1`). Selecting `lzf` without an explicit level normalizes the level to
  `0`; combining `lzf` with an explicit level is rejected.
- `--torch-compile` / `--no-torch-compile`: enable/disable best-effort inner-model `torch.compile` for supported profiles (default disabled). CUDA and CPU honor the flag during cold or resumed corpus hydration using dynamic shapes; CPU also freezes inference weights and autotunes GEMMs. MPS defers compilation until the corpus cache is hydrated. The first encode pays compilation and autotuning warm-up cost where applicable.
- `--device {auto,cuda,mps,cpu}`: compute device for embedding model runs (default `auto`, which prefers CUDA, then MPS on Apple Silicon, then CPU). Explicit unavailable devices fail fast. Shared with hybrid.
- `--semantic-source {candidates,arxiv-corpus}`: semantic candidate sourcing
  (default `candidates`, which uses an S2-derived pool without a corpus download).
  Explicit corpus-only flags (`--dataset-split`, `--corpus-size`, `--all-corpus`,
  `--streaming`) imply `arxiv-corpus` when `--semantic-source` is omitted; the
  candidate-only `--candidate-pool-size` flag likewise implies `candidates`. An
  explicit source that conflicts with a mode-only flag is rejected. Candidate mode
  stores vectors as float32 (`--storage-precision int8` requires `arxiv-corpus`).
- `--candidate-pool-size`: S2 source-fetch budget for known-paper seeds in
  candidates mode (default `400`; candidates mode only). A free-text seed first
  adds up to 20 keyword-search results, then applies the recommendation budget to
  its top anchor. As an explicit candidate-mode request, this flag overrides a
  configured `defaults.semantic_source = "arxiv-corpus"` for that run.
- Runtime defaults and execution policy details (default checkpoint chain, precision policy, and compile guard behavior) are documented in [Embedding Runtime](../reference/embedding-runtime.md).
- Default-value tuning context for recent-paper workflows is summarized in [Defaults Tuning Study](../reference/defaults-tuning-study.md).
- Use `--log-level debug --log-file out/run.log` when you want detailed embedding/cache diagnostics in both the Rich console and a shareable plain-text file. `*.log` is ignored by git in this repo.
- `info` keeps user-facing phase progress and one-time runtime summaries. Detailed
  option routing, effective embedding configuration, retry attempts, model
  provenance, and cache namespace diagnostics appear at `debug`. Warnings are
  reserved for exhausted/degraded operations, recovery, and material cache clears.

### Hybrid Strategy

- Inherits citation flags for collection, including `--no-references` and `--refresh-reference-cache`.
- Reuses all embedding controls except `--top-k`.
- When omitted, hybrid applies tuned seed-discovery defaults for collection depth:
  - `--max-papers`: `45`
  - `--max-citations`: `45`
  - `--max-references`: `12`
- `--max-semantic`: maximum non-seed semantic neighbors to add after hybrid reranking. Valid values are `0` through `max-papers - 1`. If omitted, hybrid defaults to `min(20, max-papers - 1)`.
- Setting `--max-semantic 0` disables semantic enrichment; embedding-only flags are rejected to avoid no-op configuration.
- If `--max-semantic` is omitted and `--max-papers` is `1`, the effective default is also `0`; embedding-only hybrid flags are rejected in that configuration for the same reason.

## Export Formats

Format-specific files, dashboard collection and standalone behavior, package
schemas, `*.config.json` sidecars, and determinism notes are covered in
[Output Artifacts](../reference/output-artifacts.md).

Interactive exports require viz dependencies. The `recommended` extra already includes them; otherwise install `viz` explicitly:

```bash
pip install -e ".[viz]"
```

## Examples

```bash
# Minimal citation graph
citemesh build "arxiv:1706.03762" --strategy citation -p 20

# Hybrid graph with all export formats
citemesh build "arxiv:1706.03762" --strategy hybrid --export all --theme dark

# Dashboard collection plus per-paper JSON and settings under ./out (the default)
citemesh build "arxiv:1706.03762" --strategy hybrid -e dashboard --theme dark

# Add another selectable result to that same dashboard/package
citemesh build "arxiv:1810.04805" --strategy recommendation -e dashboard --theme dark

# Add CSV alongside the automatically saved graph JSON and build sidecar
citemesh build "arxiv:1706.03762" --strategy hybrid -e dashboard -e csv --theme dark

# Standalone dashboard file for a one-off result
citemesh build "arxiv:1706.03762" --strategy hybrid -e dashboard -o out/report.dashboard.html --theme dark

# Embedding graph with a small dataset slice
citemesh build "arxiv:1810.04805" \
  --strategy embedding \
  -m unsloth/embeddinggemma-300m \
  --dataset-split "train[:2%]" \
  --export plotly

# Full selected split with a separate debug trace file
citemesh build "arxiv:2404.08801" \
  --strategy hybrid \
  --dataset-split train \
  --all-corpus \
  --export all \
  --theme dark \
  --log-level debug \
  --log-file out/full-corpus.log

# Search then build from selected ID
citemesh search "attention mechanism transformers" --limit 5
citemesh build "<paper-id-from-search>" --strategy recommendation
```

`citemesh search` has two backends selected by `--mode`. Mode `local` is semantic search over the embeddings already persisted in your local cache: candidate vectors accumulate across embedding/hybrid builds (every build grows your searchable library), or the full hydrated corpus in `arxiv-corpus` mode. The query is encoded locally in the model's query prompt space and ranked against every cached vector - no Semantic Scholar traffic, works offline once the model is downloaded. Results include cosine scores and full paper IDs ready for `citemesh build`. Mode `s2` is remote keyword search on the Semantic Scholar API - a convenience for finding seed paper IDs. It shares the anonymous S2 rate-limit pool (the endpoint most prone to 429s) unless `S2_API_KEY` is set; when the pool is saturated the command reports the rate limit honestly instead of pretending there were no results.

The default mode is `auto`: local search when your cache has embeddings, S2 keyword search otherwise, with a log line saying which backend ran and why. Persist a preference with `citemesh config set defaults.search_mode <auto|local|s2>` (explicit `--mode` still wins). Passing `--model`, `--model-profile`, or `--device` implies local mode. Explicitly requesting `local` (flag or config) with an empty cache is an error with guidance rather than a silent fallback.

Local search targets the same retrieval-document cache namespace a flagless build writes to (honoring `config.toml` defaults), so it finds your vectors automatically in the common case. It does not search the separate graph-similarity cache. Namespaces are keyed by the runtime-active model artifact, requested revision, model profile, representation contract, formatter, dimensions, and compute dtype. Pass `--model` for a non-default model and `--model-profile` when the build used an explicit profile override; float32 and bfloat16 runs use distinct namespaces, while CPU, CUDA, and MPS share a namespace when their effective compute dtype and other contracts match.

## Troubleshooting

- **No results / paper not found**: confirm identifier format and Semantic Scholar availability.
- **Partial Semantic Scholar outage**: CiteMesh continues only when at least one
  requested neighborhood source completed. Export metadata records each attempted
  source as `complete`, `empty`, or `unavailable`; a total source outage exits
  nonzero instead of producing a plausible seed-only graph.
- **Slow first embedding run**: see [Caching & Data](caching.md) for hydration behavior, cache reuse, and tuning guidance.
- **Full-corpus run still mentions `50000`**: that usually means CiteMesh is replacing an older capped namespace before hydrating the requested full selected split. Check the compact config log line for `split=...` and `corpus=all`.
- **Missing exports**: verify `--export` values; unknown strings are rejected by argparse.
- **API limits**: configure `S2_API_KEY`; see [Environment Variables](../reference/environment.md).
