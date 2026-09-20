# CLI usage guide

`citemesh build` creates a graph; `search` finds seeds, `view` opens results, `cache` manages persisted data, and `config` saves defaults. See [installation](../../README.md#quick-start) and the [build pipeline](how-it-works.md).

## Common workflows

### Build a graph

```bash
# Minimal citation graph
citemesh build "arxiv:1706.03762" --strategy citation -p 20

# Dashboard collection in ./out (default); a second build adds to it
citemesh build "arxiv:1706.03762" -s hybrid -e dashboard -e csv --theme dark
citemesh build "arxiv:1810.04805" -s recommendation -e dashboard --theme dark

# Standalone single-file dashboard
citemesh build "arxiv:1706.03762" -s hybrid -e dashboard -o out/report.dashboard.html

# Embedding graph over a corpus slice, with a debug trace
citemesh build "arxiv:1810.04805" -s embedding --dataset-split "train[:2%]" \
  -e plotly --verbose --log-file out/run.log
```

Output directories, standalone files, and collection updates follow the [output-location rules](../reference/output-artifacts.md#output-location).

### Accepted identifiers

- DOI, bare or with a `doi:` prefix or `https://doi.org/` URL, canonicalized to lowercase (DOI names are case-insensitive)
- arXiv ID or URL (`arxiv:1706.03762`, `https://arxiv.org/abs/1706.03762`, `.../pdf/1706.03762.pdf`), `vN` suffixes normalized away; bare `1706.03762` works when S2 resolves it
- Semantic Scholar paper ID
- Free-form text under `--strategy embedding`, when S2 says the input is not a known paper (outages stay errors)

### Find a seed paper

```bash
citemesh search "attention mechanism transformers" --limit 5
citemesh build "<paper-id-from-search>" --strategy recommendation
# search what an arXiv-corpus build hydrated
citemesh search "long-context attention" --semantic-source arxiv-corpus
```

Search modes:

- `local` encodes the query and ranks it against cached retrieval-document vectors, returning cosine scores and full paper IDs. It fetches no paper metadata and fails if the selected cache is empty.
- `s2` performs Semantic Scholar keyword search and returns citation counts. Requests above 100 results are paginated, up to the service's 1,000-result relevance limit. Local search accepts any positive limit.
- `auto` tries local search first and falls back to S2 when the cache is empty or unavailable, logging the reason.

Local search derives its namespace from the [saved embedding settings](configuration.md#supported-keys). Override them with `--model`, `--model-profile`, `--model-revision`, `--device`, `--semantic-source`, `--dataset-source`, `--truncate-dim`, `--storage-precision`, or `--calibration-sample-size`. A selector implies local mode unless `--mode auto` is explicit; selectors conflict with `--mode s2`. Invalid explicit option combinations fail as usage errors instead of triggering fallback.

Match the [namespace used by the build](caching.md#embedding-namespaces). `--dataset-source` implies `arxiv-corpus` unless paired with a conflicting explicit source. Local results include the searched count and a copyable build command using the active model and recorded corpus scope; the command is omitted with a warning if the cache has no recorded split. [API-key configuration](configuration.md#api-key) and [retries](#appendix-b-troubleshooting) apply whenever search uses S2; `--s2-retry-budget` overrides the recovery-time cap for an S2 search or `auto` fallback.

### View a saved dashboard

`citemesh view [PATH]` opens a saved HTML export in a browser without rebuilding or starting a server. `PATH` defaults to `out/dashboard.html`; a directory resolves to its `dashboard.html`, and an explicit `.html`/`.htm` file works. A missing dashboard, non-HTML input, or failed launch exits `1`.

```bash
citemesh view
citemesh view out/my-collection
# pick a browser
citemesh view out/report.dashboard.html --browser google-chrome
```

### Save defaults, inspect the cache

```bash
citemesh config set defaults.semantic_source arxiv-corpus
citemesh cache scan
```

See [User Configuration](configuration.md) for saved defaults and [cache maintenance](caching.md#inspecting-and-clearing) for scan and reset behavior.

### Help and console output

Every command takes `-h`/`--help`, listing built-in defaults; [user configuration](configuration.md) shows how to inspect saved overrides. `--log-width` sizes result tables and logs but never help; [environment variables](../reference/environment.md#other-respected-variables) control color. Logs go to stderr.

`--verbose` is shorthand for `--log-level debug`; it works before or after a subcommand. When several verbosity options are present, the last one wins. `--log-level debug --log-file out/run.log` adds option routing, cache and provider details, runtime configuration, retry attempts, and namespace decisions.

| Level | Output |
| --- | --- |
| `error` | The requested operation failed. |
| `warning` | A result was materially degraded or an explicitly requested runtime feature could not be used. |
| `info` | Brief build phases, useful outcomes, and saved output locations. |
| `debug` | Cache, provider, runtime, and scoring diagnostics for development. |

Progress bars appear only on an interactive terminal at `info` or `debug`.

## Flag reference

Build options are strategy-scoped: an explicit flag unsupported by the selected strategy is a CLI error. [Configured defaults](configuration.md#precedence) follow different applicability rules.

### Core options

| Flag | Description | Default |
| --- | --- | --- |
| `--strategy`, `-s` | `recommendation`, `citation`, `embedding`, or `hybrid` | `recommendation` |
| `--max-papers`, `-p` | Maximum nodes in the final graph, seed included | `40` (`hybrid`: implicit `45`) |
| `--refresh-paper-cache` | Bypass [persisted paper metadata](caching.md#paper-metadata-and-reference-ids) for this run | disabled |
| `--s2-retry-budget SECONDS` | Override the [Semantic Scholar recovery budget](#appendix-b-troubleshooting) | `90` seconds anonymously; no elapsed cap with an API key |
| `--spring-iterations`, `-i` | Iterations for the spring-layout fallback only | `100` |
| `--dpi`, `-d` | PNG output resolution | `150` |
| `--seed` | Seed for the layout shared by `png`, `plotly`, `dashboard`, `json` | deterministic built-in seed |
| `--include-timestamp` | Include generation time in output metadata | disabled |
| `--export`, `-e` | Select an [export format](../reference/output-artifacts.md); repeat for multiple, or use `all` | `png` |
| `--theme` | `light`, `dark`, `solarized`, `auto`; `auto` checks environment hints before macOS appearance | `dark` |
| `--output`, `-o` | File or directory, following the [output-location rules](../reference/output-artifacts.md#output-location) | automatic |
| `--log-level` | `debug`, `info`, `warning`, `error` | `info` |
| `--verbose` | Shorthand for `--log-level debug` | disabled |
| `--log-width` | Console wrap width in columns; `0` means terminal width on a TTY, a stable fallback when redirected | `0` |
| `--log-file` | Plain-text log file path, overwriting an existing file | disabled |

The logging flags work on every command, including nested `cache` and `config` subcommands.

### Cross-strategy behavior

- `--similarity-threshold`, `-t` sets minimum edge similarity for `recommendation` and `citation` (default `0.2`).
- `--no-references` and `--refresh-reference-cache` apply to `recommendation`, `citation`, and hybrid's citation branch.

### Citation strategy

- `--max-citations`, `-c`: citing papers to fetch (default `25`; hybrid implicit default `45`)
- `--max-references`, `-r`: referenced papers to fetch (default `25`; hybrid implicit default `12`)

### Embedding strategy

#### Model and device

- `--model`, `-m`: checkpoint or local path; see [model selection and fallback](../reference/embedding-runtime.md#model-selection-and-fallback) for the default.
- `--model-profile {auto,default,embeddinggemma}`: task and runtime contract. `auto` recognizes known Hub aliases and compatible local metadata; name one explicitly for stripped local fine-tune exports.
- `--model-revision`: branch, tag, or commit for hub models
- `--truncate-dim`: output dimensions, overriding the [model profile](../reference/embedding-runtime.md#embeddinggemma-profile).
- `--batch-size`, `-bs`: encode batch size for hydration and search (default `32`)
- `--device {auto,cuda,mps,cpu}`: compute device; see [device selection](../reference/embedding-runtime.md#device-selection).
- `--torch-compile` / `--no-torch-compile`: enable or disable the [compile policy](../reference/embedding-runtime.md#compile-policy).

#### Candidate sourcing

- `--semantic-source {candidates,arxiv-corpus}`: default `candidates`, an S2-derived pool with no corpus download. With the source omitted, the corpus-only flags below imply `arxiv-corpus` and `--candidate-pool-size` implies `candidates`; mixing the two families, or naming a source that conflicts with a mode-only flag, is rejected.
- `--candidate-pool-size`: S2 fetch budget in candidates mode (default `400`); allocation is described under [candidate acquisition](how-it-works.md#2-candidate-acquisition).
- `--dataset-source`: Hugging Face metadata repository (default `librarian-bots/arxiv-metadata-snapshot`), using the [supported corpus fields](caching.md#corpus-records).
- `--dataset-split`: split within that source (default `train`). Non-streaming slices such as `train[:5%]` bound the rows exposed to CiteMesh after dataset preparation.
- `--corpus-size`: cap the selected corpus at N papers; omitted means the full split. [Hydration and resume](caching.md#corpus-hydration-and-resume) explain selection order, scanning, and later growth.
- `--all-corpus`: use the full selected split, overriding a configured cap; rejects an explicit `--corpus-size`.
- `--streaming` / `--no-streaming`: stream the dataset or load cached shards. Streaming requires a non-sliced split; the negative form overrides `defaults.streaming` for one run.

#### Graph edges and cache storage

- `--top-k`, `-k`: strict per-node edge cap during embedding-graph pruning (default `4`)
- `--min-semantic-similarity`: cosine required for embedding/hybrid graph edges (default `0.74`, calibrated for EmbeddingGemma at 512 dimensions); hybrid can also admit pairs with shared references. It does not re-encode anything.
- `--storage-precision {int8,float32}`: persistent cache precision, `int8` in corpus mode and `float32` in candidates mode, which has no calibration data - so explicit `int8` requires `arxiv-corpus`.
- `--binary-prefilter` / `--no-binary-prefilter`: binary Hamming prefilter, on for int8 corpus mode and normalized off in candidates mode
- `--binary-rescore-multiplier`: oversampling factor for prefilter rescoring (int8 default `8`; elsewhere normalized to an unused `1`)
- `--calibration-sample-size`: sample size for int8 quantization ranges (default `2000`)
- `--cache-compression` / `--cache-compression-level`: HDF5 filter for cache datasets (`gzip` or `lzf`, default `gzip` at level `1`). `lzf` normalizes the level to `0` and rejects an explicit one.
- `--force-rebuild-cache`, `--overwrite-cache`, `--cache-overwrite-reason`: request, approve, and annotate a [forced rebuild](caching.md#inspecting-and-clearing). The latter two require `--force-rebuild-cache`.

The prefilter, rescore-multiplier, and calibration flags are int8-only: passing any explicitly requires `--storage-precision int8`. The sweeps behind these numbers are in [Defaults Tuning Study](../reference/defaults-tuning-study.md).

### Hybrid strategy

- Inherits the citation collection flags and every embedding control except `--top-k`.
- `--max-semantic`: semantic-only non-seed additions after reranking, from `0` through `max-papers - 1`, defaulting to `min(20, max-papers - 1)`. At an effective `0` - explicit, or implied by `--max-papers 1` - semantic enrichment is off and embedding-only flags are rejected.

### Export formats

Formats, dashboard collection and standalone behavior, package schemas, sidecars, and determinism notes: [Output Artifacts](../reference/output-artifacts.md). Interactive exports need the [viz extra](../../README.md#quick-start).

![CiteMesh dashboard built with --theme light, with a paper selected](../../assets/ui-dashboard-light-theme.png)

_`--theme light` with a paper selected._

## Appendix A: Validation rules

- `build <paper-id>` and `search <query>` require non-empty strings.
- At least `1`: `--max-papers`, `--spring-iterations`, `--dpi`, `--corpus-size`, `--top-k`, `--truncate-dim`, `--binary-rescore-multiplier`, `--calibration-sample-size`, `--batch-size` / `-bs`, `--candidate-pool-size`, `search --limit`.
- At least `0`: `--max-citations`, `--max-references`, `--cache-compression-level` (valid only with `--cache-compression gzip`).
- `--s2-retry-budget` must be a finite float at least `0`.
- `--similarity-threshold` and `--min-semantic-similarity` must be finite floats in `[0.0, 1.0]`.

## Appendix B: Troubleshooting

Semantic Scholar calls retry with exponential full jitter: up to 30 attempts per HTTP request, the wait ceiling doubling from 2 seconds (4 for HTTP 429) to 60, and a numeric `Retry-After` honored as a floor up to 300 seconds per delay. Pagination retries only the failed page. Anonymous builds have a shared 90-second recovery budget for a candidate collection; failed requests, retry attempts, and recovery waits consume it, while healthy first requests, local embedding work, and rendering do not. Authenticated builds retain the attempt limit without an elapsed cap unless `--s2-retry-budget SECONDS` overrides it; pass `0` to remove the elapsed cap while keeping 30 attempts. An in-flight HTTP request cannot be cancelled at an exact recovery-budget boundary. HTTP 408 and server errors retry; bad parameters and rejected credentials fail immediately.

- **No results / paper not found**: check the identifier format and S2 availability.
- **Discovery check unavailable**: required acquisition failure exits nonzero without replacing prior outputs; retry later, [configure an API key](configuration.md#api-key), or use the local `arxiv-corpus` semantic source, which requires the embeddings extras and a downloaded, embedded corpus. The error identifies the operation, HTTP failure when available, attempts, recovery time, and stopping reason.
- **Partial outage**: citation and hybrid builds can continue while at least one requested source completes or returns a valid empty result. A failed source does not by itself prevent checking another source. The recovery budget is shared across all requests: when it is spent, further uncached S2 calls stop for that collection. Export metadata marks every source `complete`, `empty`, or `unavailable`, and a total outage exits nonzero rather than emitting a seed-only graph. Optional reference enrichment stops further network lookups after an outage while still reusing available cached reference lists.
- **Slow first embedding run or an older corpus cap in the logs**: see [hydration and resume](caching.md#corpus-hydration-and-resume).
