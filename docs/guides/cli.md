# CLI Usage Guide

The `citemesh` command builds paper graphs with one of four strategies: `recommendation`, `citation`, `embedding`, or `hybrid`.

## Scope

This is the canonical CLI behavior specification.

- Normative here: commands, flags, defaults, validation, identifier normalization, output naming, and export semantics.
- Non-normative here: cache storage internals and on-disk layout. See [Caching & Data](caching.md).
- Runtime environment-variable definitions are canonical in [Environment Variables](../reference/environment.md).
- Documentation ownership map: [Documentation Index](../README.md).

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
citemesh cache clear [--yes]
```

For cache path/layout/hydration details, see [Caching & Data](caching.md).
In non-interactive shells, `citemesh cache clear` requires `--yes`.

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
| `--max-papers`, `-p` | Maximum nodes in final graph (seed included) | `40` |
| `--spring-iterations`, `-i` | Iterations used only for spring-layout fallback | `100` |
| `--dpi`, `-d` | PNG output resolution | `150` |
| `--seed` | Seed for layout computation used by layout-based exports (`png`, `plotly`) | deterministic built-in seed |
| `--include-timestamp` | Include generation time in output metadata | disabled |
| `--export`, `-e` | One of `png`, `html`, `plotly`, `json`, `graphml`, or `all` | `png` |
| `--theme` | `light`, `dark`, `solarized`, `auto` | `light` |
| `--output`, `-o` | Output path (single export) or output directory base (multi-export) | auto-generated per-paper folder |

When `--output` is omitted, CiteMesh writes to `out/<slug>-<hash>/<strategy>.<ext>`.

`<slug>` is a filesystem-safe version of the seed title, `<hash>` is the first 8 hex characters of `sha256(seed_id)`, and the combined directory name is capped at 40 characters.

When `--export all` is used, CiteMesh writes every supported format using consistent styling into a directory. With default naming this is `out/<slug>-<hash>/`; with explicit output it uses the directory base derived from `-o`.

If explicit `-o` ends with a known export suffix (for example `-o out/my-run.json`), the suffix is stripped and the remaining path is treated as the directory base for multi-export runs.
If explicit `-o` has no known suffix, it is treated as the directory base for multi-export runs.
Example: `citemesh build "<paper-id>" --strategy hybrid --export all -o out.png`
normalizes to directory `out/` and writes files like `out/hybrid.png`, `out/hybrid.html`,
`out/hybrid.plotly.html`, `out/hybrid.json`, and `out/hybrid.graphml`.

For multi-export runs, files are named `<strategy>.<ext>` inside the selected directory. Example:

`citemesh build "<paper-id>" --strategy hybrid --export all -o out/arxiv-2508.14040-hybrid-full`

writes:

- `out/arxiv-2508.14040-hybrid-full/hybrid.png`
- `out/arxiv-2508.14040-hybrid-full/hybrid.html`
- `out/arxiv-2508.14040-hybrid-full/hybrid.plotly.html`
- `out/arxiv-2508.14040-hybrid-full/hybrid.json`
- `out/arxiv-2508.14040-hybrid-full/hybrid.graphml`

For single-export runs, explicit `-o` remains file-style and preserves extension replacement/appending behavior.

`--seed` controls shared layout generation for `png` and `plotly` exports. Pyvis `html` exports use vis.js browser physics and do not consume this precomputed layout.

Numeric validation:

- `build <paper-id>` and `search <query>` require non-empty strings.
- `--max-papers`, `--spring-iterations`, `--dpi`, `--corpus-size`, `--top-k`, `--truncate-dim`, `--binary-rescore-multiplier`, `--calibration-sample-size`, and `search --limit` must be at least `1`.
- `--max-citations` and `--max-references` must be at least `0`.
- `--cache-compression-level` must be at least `0`.
- `--max-semantic` must satisfy `0 <= max-semantic <= max-papers - 1` (hybrid strategy).
- `--similarity-threshold` must be a finite float between `0.0` and `1.0`.

## Strategy-Specific Flags

Strategy behavior and tradeoffs are canonical in [Strategies Guide](strategies.md). Flag contracts live here.

Build command options are strategy-scoped. If you pass a flag that is not supported
for the selected `--strategy`, CiteMesh exits with a CLI error instead of silently
ignoring it.

### Cross-Strategy Scope

- `--similarity-threshold` applies to `recommendation` and `citation` strategies as the minimum edge similarity threshold (default `0.2`).
- `--no-references` applies to `recommendation`, `citation`, and the citation branch of `hybrid`.
- `--refresh-reference-cache` applies to `recommendation`, `citation`, and the citation branch of `hybrid` to bypass persisted reference-cache reads.
- `embedding` uses embedding-specific controls (`--dataset-split`, `--corpus-size`, `--all-corpus`, `--truncate-dim`, `--streaming`, `--top-k`, storage/cache flags below).
- The embedding branch of `hybrid` reuses embedding controls except `--top-k` (hybrid edge pruning follows its own policy).

### Recommendation Strategy

- Uses Semantic Scholar recommendations as the primary neighborhood signal.
- Uses cross-strategy controls above (`--similarity-threshold`, `--no-references`, `--refresh-reference-cache`).

### Citation Strategy

- `--max-citations`, `-c`: limit number of citing papers (default `20`)
- `--max-references`, `-r`: limit number of referenced papers (default `20`)
- `--similarity-threshold`, `-t`: minimum edge similarity threshold (`0.0` to `1.0`, default `0.2`)
- `--no-references`: skip reference-list fetching (faster, no bibliographic coupling)
- `--refresh-reference-cache`: bypass persisted reference-cache reads and fetch fresh reference IDs

### Embedding Strategy

- `--model`, `-m`: sentence-transformer model name (for example `all-MiniLM-L6-v2`, `google/embeddinggemma-300m`)
- `--model-revision`: optional model revision token (branch/tag/commit) for hub-backed models
- `--dataset-split`: HuggingFace split (default `train`; sliced forms like `train[:5%]` are supported in non-streaming mode)
- `--corpus-size`: maximum papers to load from corpus (default `50000`)
- `--all-corpus`: remove corpus-size cap and process the full selected split
- `--all-corpus` cannot be combined with an explicit `--corpus-size` value
- `--top-k`, `-k`: strict per-node edge cap during embedding-graph pruning (default `2`)
- `--truncate-dim`: optional embedding output-dimension truncation (for EmbeddingGemma: `768`, `512`, `256`, `128`)
- `--streaming`: stream HuggingFace dataset instead of loading cached shards. Streaming requires a non-sliced split (for example `train`).
- `--force-rebuild-cache`: clear and rebuild embedding cache for this model before running
- `--storage-precision {int8,float16,float32}`: persistent embedding-cache precision (default `int8`)
- `--binary-prefilter` / `--no-binary-prefilter`: enable/disable binary Hamming prefilter for quantized search (default enabled). Explicit `--binary-prefilter` requires `--storage-precision int8`.
- `--binary-rescore-multiplier`: oversampling factor for binary prefilter candidate rescoring (default `8`). Explicit use requires `--storage-precision int8`.
- `--calibration-sample-size`: calibration sample size used to compute int8 ranges (default `2000`)
- `--cache-compression`: HDF5 compression filter for cache datasets (`gzip`, `lzf`; default `gzip`)
- `--cache-compression-level`: HDF5 compression level for cache datasets (default `1`)
- `--torch-compile` / `--no-torch-compile`: enable/disable best-effort inner-model `torch.compile` for supported profiles (default enabled)
- On `torch==2.9` with Ampere+ CUDA TF32 enabled, CiteMesh skips `torch.compile` for embedding models to avoid a known upstream TF32 API conflict path in TorchInductor.

Runtime precision policy:

- TF32 kernels are auto-enabled on supported Ampere+ CUDA runtimes for embedding inference via `torch.backends.fp32_precision = "tf32"` (new API only; requires `torch>=2.9.0`). This behavior is intentional and currently does not expose a CLI toggle.

Execution transparency:

- Before embedding/hybrid execution, CLI logs a preflight contract describing expected side effects (model/dataset artifact download risk and embedding-cache mutation scope).
- With non-int8 precision, implicit binary-prefilter defaults are normalized to effective runtime values (`binary_prefilter=false`, `binary_rescore_multiplier=1`) to avoid no-op ambiguity.

### Hybrid Strategy

- Inherits citation flags for collection, including `--no-references` and `--refresh-reference-cache`.
- Reuses embedding corpus/model/cache controls (`--model-revision`, `--dataset-split`, `--corpus-size`, `--all-corpus`, `--truncate-dim`, `--streaming`, `--storage-precision`, binary prefilter/rescore flags, calibration/compression flags).
- `--max-semantic`: maximum non-seed semantic neighbors to add when enriching the citation graph.
  Hybrid reserves this capacity from citation collection (`citation_budget = max_papers - max_semantic`), so valid values are `0` through `max-papers - 1`.
  If omitted, hybrid defaults to `min(10, max-papers - 1)`.
- Setting `--max-semantic 0` disables semantic enrichment; embedding-only flags are rejected to avoid no-op configuration.
- If `--max-semantic` is omitted and `--max-papers` is `1`, the effective default is also `0`; embedding-only hybrid flags are rejected in that configuration for the same reason.
- Hybrid runs fail closed if semantic enrichment fails; CiteMesh does not silently downgrade to citation-only output.

## Export Formats

- `png`: Matplotlib static render with theme-aware background and labels
- `html` (Pyvis): vis.js network with hover tooltips and in-browser physics
- `plotly`: interactive Plotly graph (HTML), written with `.plotly.html` suffix
- `json`: structured graph data with nodes, edges, metadata
- `graphml`: exchange format for Gephi, Cytoscape, and similar tools

For `embedding` and `hybrid` strategies, export metadata includes embedding provenance fields (`effective_vector_dtype`, `storage_precision`, configured binary-prefilter state, and `binary_prefilter_used_for_query` when runtime retrieval metadata is available).
All strategies include a `score_contract` object in export metadata describing score semantics (`score_type`) and explicitly marking scores as non-comparable across strategies.
Hybrid exports also include `score_contract.adjudication_policy` describing citation/semantic merge behavior.

Determinism notes:

- `json` exports are deterministic (stable key order and indentation).
- `graphml` export is deterministic on NetworkX versions with stable writer ordering (`>=2.8` uses strict node/edge sorting metadata).
- `png` is deterministic for the same input graph and `--seed`.
- `plotly` is deterministic for the same input graph and `--seed` when Plotly supports `write_html(div_id=...)` (CiteMesh fails fast if unavailable).
- Pyvis `html` export has deterministic serialized ordering, but runtime browser physics remain non-deterministic.

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
- **API limits**: configure `S2_API_KEY`; variable contract is canonical in [Environment Variables](../reference/environment.md).

Related canonical docs:

- Docs ownership map: [Documentation Index](../README.md)
- Cache behavior: [Caching & Data](caching.md)
- Runtime variables: [Environment Variables](../reference/environment.md)
- Component architecture: [Architecture](../internals/architecture.md)
