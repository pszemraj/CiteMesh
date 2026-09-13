# CLI Usage Guide

`citemesh build` turns a seed paper into a graph with one of four strategies — `recommendation`, `citation`, `embedding`, `hybrid` — and `search`, `view`, `cache`, and `config` support it. Install notes: [README](../../README.md); mechanism: [How CiteMesh builds a graph](how-it-works.md).

[Common workflows](#common-workflows) is what you type, [Flag reference](#flag-reference) the contract of every option, plus appendices on [validation](#appendix-a-validation-rules) and [troubleshooting](#appendix-b-troubleshooting).

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
  -e plotly --log-level debug --log-file out/run.log
```

Without `--output`, results land in `out/` (gitignored in a source checkout). A dashboard build maintains `out/dashboard.html` and `out/dashboard.citemesh.json`, and always writes `<strategy>.json` plus `<strategy>.config.json` into `out/<title-slug>-<hash>/`. The hash comes from the canonical seed ID, so a corrected title does not move the exports, seeds accumulate side by side, and rerunning one replaces only its own result. Naming and path rules: [Output Artifacts](../reference/output-artifacts.md).

### Accepted identifiers

- DOI, bare or with a `doi:` prefix or `https://doi.org/` URL, canonicalized to lowercase (DOI names are case-insensitive)
- arXiv ID or URL (`arxiv:1706.03762`, `https://arxiv.org/abs/1706.03762`, `.../pdf/1706.03762.pdf`), `vN` suffixes normalized away; bare `1706.03762` works when S2 resolves it
- Semantic Scholar paper ID
- Free-form text under `--strategy embedding`, when S2 says the input is not a known paper (outages stay errors)

### Find a seed paper

```bash
citemesh search "attention mechanism transformers" --limit 5
citemesh build "<paper-id-from-search>" --strategy recommendation
```

`--mode local` searches the vectors already in your cache — the query encoded in the model's query prompt space and ranked against every cached retrieval-document vector, offline once the model is downloaded, returning cosine scores and IDs ready for `build`. Every embedding or hybrid build grows that library. `--mode s2` is Semantic Scholar keyword search, convenient from a cold start, but it shares the anonymous rate-limit pool unless `S2_API_KEY` or `api.s2_api_key` is set, and reports a 429 honestly rather than as "no results".

The default `auto` picks local when the cache has vectors and S2 otherwise, logging which and why. `--model`, `--model-profile`, and `--device` imply local mode and are rejected alongside `--mode s2`; local against an empty cache is an error naming whatever asked for it, reported with the resolved device and compute dtype. Local search reads the namespace a flagless build writes to, honoring `config.toml`, never the graph-similarity cache — so match any non-default build settings here or in config ([Caching & Data](caching.md)).

### View a saved dashboard

`citemesh view [PATH]` opens a saved dashboard in a browser — no rebuild, no server. `PATH` defaults to `out/dashboard.html`; a directory resolves to its `dashboard.html`, and an explicit `.html`/`.htm` file works. A missing dashboard, non-HTML input, or failed launch exits `1`.

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
citemesh cache clear [--yes|-y] [--reason "<text>"]
```

Most build flags persist in `config.toml` at the cache root ([User Configuration](configuration.md)). A non-interactive shell needs `--yes` for `cache clear`, which never deletes `config.toml`, and `--overwrite-cache` alongside `--force-rebuild-cache`. Cache layout and hydration policy: [Caching & Data](caching.md).

### Help and console output

Every command takes `-h`/`--help`, listing built-in defaults; `citemesh config list` shows your saved overrides. `--log-width` sizes result tables and logs but never help, `NO_COLOR=1` drops color, and `config get`/`config path` write raw values to stdout for shell substitution while logs go to stderr.

`--log-level debug --log-file out/run.log` adds option routing, effective embedding configuration, retry attempts, model provenance, and namespace decisions. `info` keeps phase progress and one-time runtime summaries; warnings mark degraded operations, recovery, and material cache clears.

## Flag reference

Build options are strategy-scoped: a flag the selected `--strategy` does not support is a CLI error, not a silent no-op. Strategy tradeoffs are in the [Strategy Guide](strategies.md); the contracts are here.

### Core options

| Flag | Description | Default |
| --- | --- | --- |
| `--strategy`, `-s` | `recommendation`, `citation`, `embedding`, or `hybrid` | `recommendation` |
| `--max-papers`, `-p` | Maximum nodes in the final graph, seed included | `40` (`hybrid`: implicit `45`) |
| `--refresh-paper-cache` | Fetch fresh S2 metadata and citation counts without re-encoding; a failed refresh keeps the cached entry | disabled |
| `--spring-iterations`, `-i` | Iterations for the spring-layout fallback only | `100` |
| `--dpi`, `-d` | PNG output resolution | `150` |
| `--seed` | Seed for the layout shared by `png`, `plotly`, `dashboard`, `json` | deterministic built-in seed |
| `--include-timestamp` | Include generation time in output metadata | disabled |
| `--export`, `-e` | `png`, `html`, `plotly`, `dashboard`, `json`, `csv`, `bibtex`, `graphml`, `all`; repeat for multiple | `png` |
| `--theme` | `light`, `dark`, `solarized`, `auto`; `auto` checks environment hints before macOS appearance | `dark` |
| `--output`, `-o` | Output path, or collection root for dashboard exports. An existing directory receives artifacts inside it; an explicit `*.dashboard.html` path requests standalone mode | `out/` collection root for dashboard, else a per-paper folder |
| `--log-level` | `debug`, `info`, `warning`, `error` | `info` |
| `--log-width` | Console wrap width in columns; `0` means terminal width on a TTY, a stable fallback when redirected | `0` |
| `--log-file` | Plain-text log file path, overwriting an existing file | disabled |

The logging flags work on `build`, `search`, `cache`, and `config`, subcommands included. Pyvis `html` exports run vis.js physics and ignore the precomputed `--seed` layout.

### Cross-strategy behavior

- `--similarity-threshold`, `-t` sets minimum edge similarity for `recommendation` and `citation` (default `0.2`).
- `recommendation` and `citation` cap each paper at 3 edges, fixed; `--top-k` applies only to `embedding`, and `hybrid` caps at 5.
- `--no-references` and `--refresh-reference-cache` apply to `recommendation`, `citation`, and hybrid's citation branch.

### Citation strategy

- `--max-citations`, `-c`: citing papers to fetch (default `25`; hybrid implicit default `45`)
- `--max-references`, `-r`: referenced papers to fetch (default `25`; hybrid implicit default `12`)

### Embedding strategy

**Model and device**

- `--model`, `-m`: sentence-transformer checkpoint or local path (default `unsloth/embeddinggemma-300m`, falling back to `google/embeddinggemma-300m`)
- `--model-profile {auto,default,embeddinggemma}`: task and runtime contract. `auto` recognizes known Hub aliases and compatible local metadata; name one explicitly for stripped local fine-tune exports.
- `--model-revision`: branch, tag, or commit for hub models
- `--truncate-dim`: output-dimension truncation (EmbeddingGemma: `768`, `512`, `256`, `128`; omitted takes the profile default, **`512`** there). Tradeoffs: [dimension study](../reference/defaults-tuning-study.md#embedding-dimensions-september-2026).
- `--batch-size`, `-bs`: encode batch size for hydration and search (default `32`)
- `--device {auto,cuda,mps,cpu}`: `auto` prefers CUDA, then MPS on Apple Silicon, then CPU. An explicitly requested unavailable device fails fast.
- `--torch-compile` / `--no-torch-compile`: best-effort inner-model `torch.compile`, default off; per-device behavior and warm-up cost are in [Embedding Runtime](../reference/embedding-runtime.md).

**Candidate sourcing**

- `--semantic-source {candidates,arxiv-corpus}`: default `candidates`, an S2-derived pool with no corpus download. With the source omitted, the corpus-only flags below imply `arxiv-corpus` and `--candidate-pool-size` implies `candidates`; mixing the two families, or naming a source that conflicts with a mode-only flag, is rejected.
- `--candidate-pool-size`: S2 fetch budget for a known-paper seed in candidates mode (default `400`). A free-text seed first takes up to 20 keyword-search results, then spends the budget on its top anchor.
- `--dataset-source`: HuggingFace arXiv metadata repository (default `librarian-bots/arxiv-metadata-snapshot`). It must supply `id` (or `paper_id` / `paperId`), `title`, and `abstract` (or `summary`); `authors`, `categories`, `year`, `doi`, and `venue` / `journal_ref` / `journal` are used when present.
- `--dataset-split`: split within that source (default `train`). Non-streaming slices such as `train[:5%]` bound the rows exposed to CiteMesh after dataset preparation.
- `--corpus-size`: embed and cache the N newest submissions by arXiv ID from the selected split; it does not cap the scan establishing that order. It sizes a cold build rather than bounding the namespace for life — a later run can extend or refresh it past N ([Caching & Data](caching.md)). Without it, CiteMesh hydrates the full split.
- `--all-corpus`: the full selected split, overriding a configured cap — "all of `--dataset-split`", not every split the dataset publishes. It rejects an explicit `--corpus-size`, and extends a namespace previously hydrated with a cap rather than re-encoding it.
- `--streaming` / `--no-streaming`: stream the dataset or load cached shards. Streaming requires a non-sliced split; the negative form overrides `defaults.streaming` for one run.

**Graph edges and cache storage**

- `--top-k`, `-k`: strict per-node edge cap during embedding-graph pruning (default `4`)
- `--min-semantic-similarity`: cosine required for embedding/hybrid graph edges (default `0.74`, calibrated for EmbeddingGemma at 512 dimensions); hybrid can also admit pairs with shared references. It does not re-encode anything.
- `--storage-precision {int8,float32}`: persistent cache precision, `int8` in corpus mode and `float32` in candidates mode, which has no calibration data — so explicit `int8` requires `arxiv-corpus`.
- `--binary-prefilter` / `--no-binary-prefilter`: binary Hamming prefilter, on for int8 corpus mode and normalized off in candidates mode
- `--binary-rescore-multiplier`: oversampling factor for prefilter rescoring (int8 default `8`; elsewhere normalized to an unused `1`)
- `--calibration-sample-size`: sample size for int8 quantization ranges (default `2000`)
- `--cache-compression` / `--cache-compression-level`: HDF5 filter for cache datasets (`gzip` or `lzf`, default `gzip` at level `1`). `lzf` normalizes the level to `0` and rejects an explicit one.
- `--force-rebuild-cache`, `--overwrite-cache`, `--cache-overwrite-reason`: clear and re-encode this model's namespaces before running. The rebuild prompts unless `--overwrite-cache` is passed; the other two are rejected without it.

The prefilter, rescore-multiplier, and calibration flags are int8-only: passing any explicitly requires `--storage-precision int8`. The sweeps behind these numbers are in [Defaults Tuning Study](../reference/defaults-tuning-study.md).

### Hybrid strategy

- Inherits the citation collection flags and every embedding control except `--top-k`.
- Omitted budget knobs take tuned seed-discovery defaults: `--max-papers` `45`, `--max-citations` `45`, `--max-references` `12`.
- `--max-semantic`: non-seed semantic neighbors added after reranking, from `0` through `max-papers - 1`, defaulting to `min(20, max-papers - 1)`. At an effective `0` — explicit, or implied by `--max-papers 1` — semantic enrichment is off and embedding-only flags are rejected.

### Export formats

Formats, dashboard collection and standalone behavior, package schemas, sidecars, and determinism notes: [Output Artifacts](../reference/output-artifacts.md). Interactive exports need the viz dependencies — in the `recommended` extra, otherwise `pip install -e ".[viz]"`.

![CiteMesh dashboard built with --theme light, with a paper selected](../../assets/ui-dashboard-light-theme.png)

_`--theme light` with a paper selected. The theme is applied when the export is written; the dashboard has no toggle._

## Appendix A: Validation rules

- `build <paper-id>` and `search <query>` require non-empty strings.
- At least `1`: `--max-papers`, `--spring-iterations`, `--dpi`, `--corpus-size`, `--top-k`, `--truncate-dim`, `--binary-rescore-multiplier`, `--calibration-sample-size`, `--batch-size` / `-bs`, `--candidate-pool-size`, `search --limit`.
- At least `0`: `--max-citations`, `--max-references`, `--cache-compression-level` (valid only with `--cache-compression gzip`).
- `--max-semantic` must satisfy `0 <= max-semantic <= max-papers - 1` (hybrid only).
- `--similarity-threshold` and `--min-semantic-similarity` must be finite floats in `[0.0, 1.0]`.

## Appendix B: Troubleshooting

Semantic Scholar calls retry with exponential full jitter: up to 30 attempts per operation, the wait ceiling doubling from 2 seconds (4 for HTTP 429) to 60, a numeric `Retry-After` honored as a floor up to 300 seconds per delay, and no elapsed-time deadline, since long retries favor finishing a resumable build. An outage can therefore take minutes to surface, though any wait of 30 seconds or more is logged and Ctrl+C interrupts. HTTP 408 and server errors retry; bad parameters and rejected credentials fail immediately.

- **No results / paper not found**: check the identifier format and S2 availability.
- **Partial outage**: a build continues while at least one requested source completes or returns a valid empty result, skipping an exhausted capability for the rest of that collection and starting fresh on the next. Reference-ID and full-reference lookups share a `references` budget; citations, recommendations, search, and metadata each have their own. Export metadata marks every source `complete`, `empty`, or `unavailable`, and a total outage exits nonzero rather than emitting a seed-only graph.
- **Slow first embedding run**: the cold path downloads the checkpoint and encodes every candidate or corpus paper; later runs read the cache.
- **A full-split run still mentions `50000`**: it is replacing an older capped namespace, not capping the new one; check the config log line for `split=...` and `corpus=all`.
- **Missing exports**: unknown `--export` values are rejected.
- **Rate limits**: configure `S2_API_KEY` ([Environment Variables](../reference/environment.md)). Retry detail appears at `--log-level debug`.
