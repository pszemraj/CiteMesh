# CLI Usage Guide

The `citemesh` command builds paper graphs with one of four strategies: `recommendation`, `citation`, `embedding`, or `hybrid`.

Related docs:

- Cache layout and hydration: [Caching & Data](caching.md)
- Environment variables: [Environment Variables](../reference/environment.md)
- Output files and sidecar schema: [Output Artifacts](../reference/output-artifacts.md)
- Embedding runtime policy: [Embedding Runtime](../reference/embedding-runtime.md)
- Defaults parameter study: [Defaults Tuning Study](../reference/defaults-tuning-study.md)
- Docs index: [Documentation](../README.md)

## Basic Invocation

```bash
citemesh build "<paper-id>" [options]
```

Other command groups:

```bash
# Search by keyword/title
citemesh search "<query>" [--limit N|-n N]

# Cache management commands
citemesh cache scan
citemesh cache scan --log-level debug
citemesh cache clear [--yes] [--reason "<text>"]
```

For cache path/layout/hydration details, see [Caching & Data](caching.md).
In non-interactive shells, `citemesh cache clear` requires `--yes`.
In non-interactive embedding/hybrid runs, `--force-rebuild-cache` requires `--overwrite-cache`.

## Accepted Identifiers

- DOI (`10.1038/nature14539`)
- DOI with prefix (`doi:10.1038/nature14539`)
- DOI URL (`https://doi.org/10.1038/nature14539`)
- arXiv ID (`arxiv:1706.03762`; version suffixes like `v5` are normalized away)
- bare arXiv-like IDs (for example `1706.03762`) may work when Semantic Scholar resolves them
- arXiv URL (`https://arxiv.org/abs/1706.03762`, `https://arxiv.org/abs/arXiv:1706.03762`, `https://arxiv.org/pdf/1706.03762.pdf`; `vN` suffixes are normalized away)
- Semantic Scholar paper ID
- Free-form text query (embedding strategy treats it as text seed when S2 lookup fails)

## Core Options

| Flag | Description | Default |
| --- | --- | --- |
| `--strategy`, `-s` | `recommendation`, `citation`, `embedding`, or `hybrid` | `recommendation` |
| `--max-papers`, `-p` | Maximum nodes in final graph (seed included) | `40` (`hybrid`: implicit `45` when omitted) |
| `--spring-iterations`, `-i` | Iterations used only for spring-layout fallback | `100` |
| `--dpi`, `-d` | PNG output resolution | `150` |
| `--seed` | Seed for layout computation used by layout-based exports (`png`, `plotly`, `dashboard`) | deterministic built-in seed |
| `--include-timestamp` | Include generation time in output metadata | disabled |
| `--export`, `-e` | `png`, `html`, `plotly`, `dashboard`, `json`, `csv`, `bibtex`, `graphml`, or `all`; repeat flag for multiple (e.g. `-e json -e dashboard`) | `png` |
| `--theme` | `light`, `dark`, `solarized`, `auto` | `light` |
| `--output`, `-o` | Output path (single export) or output directory base (multi-export) | auto-generated per-paper folder |
| `--log-level` | Console logging level (`debug`, `info`, `warning`, `error`) | `info` |
| `--log-width` | Rich console wrap width in columns (`0` uses terminal width) | `140` |

Output-path normalization, file naming, and sidecar placement are defined in
[Output Artifacts](../reference/output-artifacts.md).
Use that reference as the canonical source for `--output` behavior in single-export
and multi-export runs.

`--log-level` and `--log-width` are shared command options and are accepted for
`build`, `search`, and `cache` command trees (including `cache scan` / `cache clear`).

`--seed` controls shared layout generation for `png`, `plotly`, and `dashboard` exports. Pyvis
`html` exports use vis.js browser physics and do not consume this precomputed layout.

Numeric validation:

- `build <paper-id>` and `search <query>` require non-empty strings.
- `--max-papers`, `--spring-iterations`, `--dpi`, `--corpus-size`, `--top-k`, `--truncate-dim`, `--binary-rescore-multiplier`, `--calibration-sample-size`, and `search --limit` must be at least `1`.
- `--max-citations` and `--max-references` must be at least `0`.
- `--calibration-sample-size` is valid only with `--storage-precision int8`.
- `--cache-compression-level` must be at least `0` and is valid only with `--cache-compression gzip`.
- `--max-semantic` must satisfy `0 <= max-semantic <= max-papers - 1` (hybrid strategy).
- `--similarity-threshold` must be a finite float between `0.0` and `1.0`.

## Strategy-Specific Flags

Strategy behavior and tradeoffs are described in [Strategies Guide](strategies.md). Flag contracts are listed here.

Build command options are strategy-scoped. If you pass a flag that is not supported
for the selected `--strategy`, CiteMesh exits with a CLI error instead of silently
ignoring it.

### Cross-Strategy Behavior

- `--similarity-threshold` applies to `recommendation` and `citation` strategies as the minimum edge similarity threshold (default `0.2`).
- `--no-references` applies to `recommendation`, `citation`, and the citation branch of `hybrid`.
- `--refresh-reference-cache` applies to `recommendation`, `citation`, and the citation branch of `hybrid` to bypass persisted reference-cache reads.
- `embedding` uses embedding-specific controls (`--dataset-split`, `--corpus-size`, `--all-corpus`, `--truncate-dim`, `--streaming`, `--top-k`, storage/cache flags below).
- The embedding branch of `hybrid` reuses embedding controls except `--top-k` (hybrid edge pruning follows its own policy).

### Recommendation Strategy

- Uses Semantic Scholar recommendations as the primary neighborhood signal.
- Uses cross-strategy controls above (`--similarity-threshold`, `--no-references`, `--refresh-reference-cache`).

### Citation Strategy

- `--max-citations`, `-c`: limit number of citing papers (default `25`; hybrid implicit default `45`)
- `--max-references`, `-r`: limit number of referenced papers (default `25`; hybrid implicit default `12`)
- `--similarity-threshold`, `-t`: minimum edge similarity threshold (`0.0` to `1.0`, default `0.2`)
- `--no-references`: skip reference-list fetching (faster, no bibliographic coupling)
- `--refresh-reference-cache`: bypass persisted reference-cache reads and fetch fresh reference IDs

### Embedding Strategy

- `--model`, `-m`: sentence-transformer model name (default `unsloth/embeddinggemma-300m`; examples: `all-MiniLM-L6-v2`, `google/embeddinggemma-300m`)
- `--model-revision`: optional model revision token (branch/tag/commit) for hub-backed models
- `--dataset-split`: HuggingFace split (default `train`; sliced forms like `train[:5%]` are supported in non-streaming mode)
- `--corpus-size`: maximum papers to load from corpus (default `50000`)
- With non-streaming unsliced splits, CiteMesh loads `split[:corpus_size]` directly (it does not download/process the full split just to stop after `corpus_size` rows).
- For the default `librarian-bots/arxiv-metadata-snapshot` source, current ordering places the newest `update_date` rows first, so the default cap targets recent updates.
- `--all-corpus`: remove corpus-size cap and process the full selected split
- `--all-corpus` cannot be combined with an explicit `--corpus-size` value
- `--top-k`, `-k`: strict per-node edge cap during embedding-graph pruning (default `4`)
- `--truncate-dim`: optional embedding output-dimension truncation (for EmbeddingGemma: `768`, `512`, `256`, `128`)
- `--streaming`: stream HuggingFace dataset instead of loading cached shards. Streaming requires a non-sliced split (for example `train`).
- `--force-rebuild-cache`: clear and rebuild embedding cache for this model before running (requires confirmation by default)
- `--overwrite-cache`: acknowledge destructive overwrite for `--force-rebuild-cache` and skip interactive confirmation (required for non-interactive/scripting workflows)
- `--cache-overwrite-reason`: optional rationale string logged when `--force-rebuild-cache` clears embedding cache state
- `--storage-precision {int8,float16,float32}`: persistent embedding-cache precision (default `int8`)
- `--binary-prefilter` / `--no-binary-prefilter`: enable/disable binary Hamming prefilter for quantized search (default enabled). Explicit `--binary-prefilter` requires `--storage-precision int8`.
- `--binary-rescore-multiplier`: oversampling factor for binary prefilter candidate rescoring (default `8`). Explicit use requires `--storage-precision int8`.
- `--calibration-sample-size`: calibration sample size used to compute int8 ranges (default `2000`; explicit use requires `--storage-precision int8`)
- `--encode-batch-size`: embedding-model encode batch size used during hydration/search (default `32`)
- `--cache-compression`: HDF5 compression filter for cache datasets (`gzip`, `lzf`; default `gzip`)
- `--cache-compression-level`: HDF5 compression level for cache datasets (default `1`; unsupported with `--cache-compression lzf`)
- `--torch-compile` / `--no-torch-compile`: enable/disable best-effort inner-model `torch.compile` for supported profiles (default enabled). Compile is deferred on cold-cache hydration runs and applied on warm-cache runs.
- Runtime defaults and execution policy details (default checkpoint chain, precision policy, and compile guard behavior) are documented in [Embedding Runtime](../reference/embedding-runtime.md).
- Default-value tuning context for recent-paper workflows is summarized in [Defaults Tuning Study](../reference/defaults-tuning-study.md).

Execution transparency:

- Embedding/hybrid runs print a compact config summary (model, split, corpus cap, streaming mode, storage precision).
- Retrieval/caching internals (prefilter semantics, compared/rescored counts, compile guard behavior, hydration/lock policy) are defined in:
  - [Embedding Runtime](../reference/embedding-runtime.md)
  - [Caching & Data](caching.md)
- Use `--log-level debug` when you want detailed internals (cache selection, compile skip reasons, dataset-source selection, and similar diagnostics).

### Hybrid Strategy

- Inherits citation flags for collection, including `--no-references` and `--refresh-reference-cache`.
- Reuses embedding corpus/model/cache controls (`--model-revision`, `--dataset-split`, `--corpus-size`, `--all-corpus`, `--truncate-dim`, `--streaming`, `--storage-precision`, binary prefilter/rescore flags, calibration/compression flags).
- When omitted, hybrid applies tuned seed-discovery defaults for collection depth:
  - `--max-papers`: `45`
  - `--max-citations`: `45`
  - `--max-references`: `12`
- `--max-semantic`: maximum non-seed semantic neighbors to add after hybrid reranking.
  Valid values are `0` through `max-papers - 1`.
  If omitted, hybrid defaults to `min(20, max-papers - 1)`.
- Hybrid adjudication policy:
  - fetches full citation candidates up to `max_references + max_citations`
  - fetches semantic candidates (expanded pool) and merges duplicates
  - reranks the union by seed relevance (semantic/citation/temporal/bibliographic signals) with a boost for papers found by both branches
  - applies the `max_semantic` cap only to semantic-only additions
- Setting `--max-semantic 0` disables semantic enrichment; embedding-only flags are rejected to avoid no-op configuration.
- If `--max-semantic` is omitted and `--max-papers` is `1`, the effective default is also `0`; embedding-only hybrid flags are rejected in that configuration for the same reason.
- Hybrid runs fail closed if semantic enrichment fails; CiteMesh does not silently downgrade to citation-only output.

## Export Formats

- `png`: Matplotlib static render with theme-aware background and labels
- `html` (Pyvis): vis.js network with hover tooltips and in-browser physics
- `plotly`: interactive Plotly graph (HTML), written with `.plotly.html` suffix
- `dashboard`: tri-pane research dashboard shell (list + graph + detail). In normal collection flows it is written as `dashboard.html` at the output root, embeds the saved-result collection so you can switch runs from one UI, and keeps an explicit single-file `*.dashboard.html` mode for one-off exports.
- `json`: structured graph data payload (nodes/edges)
- `csv`: flat paper table (one row per paper) for pandas/spreadsheet import
- `bibtex`: combined BibTeX entries for all papers, ready for reference managers or LaTeX
- `graphml`: exchange format for Gephi, Cytoscape, and similar tools
- `*.config.json`: run config + metadata sidecar

For field-level JSON/sidecar schema and determinism details, see
[Output Artifacts](../reference/output-artifacts.md).

Interactive exports require optional viz dependencies:

```bash
pip install -e ".[viz]"
```

## Examples

```bash
# Minimal citation graph
citemesh build "arxiv:1706.03762" --strategy citation -p 20

# Hybrid graph with all export formats
citemesh build "arxiv:1706.03762" --strategy hybrid --export all --theme dark

# Shared dashboard shell + per-run JSON payloads under ./research
citemesh build "arxiv:1706.03762" --strategy hybrid -e dashboard -e json -o research --theme dark

# Explicit standalone dashboard file for a one-off result
citemesh build "arxiv:1706.03762" --strategy hybrid -e dashboard -o report.dashboard.html --theme dark

In collection mode, each build refreshes the shared `dashboard.html` so the
saved-result selector stays in sync with `dashboard.manifest.json` and the
embedded JSON payloads available under that output root.

# Embedding graph with a small dataset slice
citemesh build "arxiv:1810.04805" \
  --strategy embedding \
  -m all-MiniLM-L6-v2 \
  --dataset-split "train[:2%]" \
  --export plotly

# Search then build from selected ID
citemesh search "attention mechanism transformers" --limit 5
citemesh build "<paper-id-from-search>" --strategy recommendation

# Build directly from an arXiv URL
citemesh build "https://arxiv.org/abs/1706.03762" --strategy recommendation --export all
```

## Troubleshooting

- **No results / paper not found**: confirm identifier format and Semantic Scholar availability.
- **Slow first embedding run**: see [Caching & Data](caching.md) for hydration behavior, cache reuse, and tuning guidance.
- **Missing exports**: verify `--export` values; unknown strings are rejected by argparse.
- **API limits**: configure `S2_API_KEY`; see [Environment Variables](../reference/environment.md).
