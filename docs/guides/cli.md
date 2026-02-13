# CLI Usage Guide

The `citemesh` command builds paper graphs using one of four strategies (`recommendation`, `citation`, `embedding`, `hybrid`). This guide walks through common flags and workflows.

This document is the canonical CLI reference for identifiers, flags, output naming, and export behavior. Other docs should link here for CLI specifics instead of restating them.

## Basic Invocation

```bash
citemesh build "<paper-id>" [options]
```

You can also discover papers by keyword/title with:

```bash
citemesh search "<query>" [--limit N]
```

Accepted identifiers:

- DOI (`10.1038/nature14539`)
- DOI with prefix (`doi:10.1038/nature14539`)
- DOI URL (`https://doi.org/10.1038/nature14539`)
- arXiv ID (`arxiv:1706.03762`; version suffixes like `v5` are normalized away)
- bare arXiv-like IDs (for example `1706.03762`) may work when Semantic Scholar resolves them
- arXiv URL (`https://arxiv.org/abs/1706.03762`, `https://arxiv.org/pdf/1706.03762.pdf`; `vN` suffixes are normalized away)
- Semantic Scholar Paper ID
- Free-form text query (embedding strategy treats it as a search query)

## Core Options

| Flag                 | Description                                                      | Default                        |
| -------------------- | ---------------------------------------------------------------- | ------------------------------ |
| `--strategy`, `-s`   | `recommendation`, `citation`, `embedding`, or `hybrid`           | `recommendation`               |
| `--max-papers`, `-p` | Maximum nodes in final graph                                     | `40`                           |
| `--spring-iterations`, `-i` | Iterations used only for spring-layout fallback                    | `100`                          |
| `--dpi`, `-d`        | PNG output resolution                                            | `150`                          |
| `--seed`             | Seed for layout computation used by layout-based exports (`png`, `plotly`) | deterministic built-in seed    |
| `--include-timestamp`| Include generation time in output metadata                       | disabled                       |
| `--export`, `-e`     | One of `png`, `html`, `plotly`, `json`, `graphml`, or `all`     | `png`                          |
| `--theme`            | `light`, `dark`, `solarized`, `auto`                             | `light`                        |
| `--output`, `-o`     | Base filename for exports                                        | auto-generated in paper folder |

When `--output` is omitted, CiteMesh writes to `out/<safe_seed_title[:50]>-<seed_hash8>/<strategy>.<ext>`.
When `--export all` is used, CiteMesh writes every supported format using consistent styling. If you specify a custom output path, the CLI appends the correct extension for each exported format.
Custom basenames containing dots (for example `-o out/arxiv-2508.14040-example`) are preserved; format extensions are appended without truncating the basename.

`--seed` controls the shared layout path for `png` and `plotly` exports. Pyvis `html` exports use vis.js physics and do not consume this layout.
Metadata timestamps are omitted by default for deterministic artifacts; use `--include-timestamp` to opt in.
Numeric validation:
- `--max-papers`, `--spring-iterations`, `--dpi`, `--corpus-size`, `--top-k`, and `search --limit` must be at least `1`.
- `--max-citations`, `--max-references`, and `--max-semantic` must be at least `0`.

## Strategy-Specific Flags

### Cross-Strategy Scope

- `--similarity-threshold` applies to `recommendation` and `citation` strategies as the minimum edge similarity threshold (`0.0` to `1.0`).
- `--no-references` applies to `recommendation`, `citation`, and the citation branch of `hybrid`.
- `embedding` and the embedding branch of `hybrid` use embedding-specific controls (`--dataset-split`, `--corpus-size`, `--all-corpus`, `--truncate-dim`, `--streaming`, `--top-k`).

### Recommendation Strategy

- Recommended default.
- Uses Semantic Scholar recommendations for fast, high-signal topical seeds.
- Reuses:
  - `--similarity-threshold`, `-t` as the minimum edge similarity threshold.
  - `--no-references` to skip fetching references for the seed and recommendation neighbors.

### Citation Strategy

- `--max-citations`, `-c`: limit number of citing papers
- `--max-references`, `-r`: limit number of referenced papers
- `--similarity-threshold`, `-t`: minimum edge similarity threshold (`0.0` to `1.0`)
- `--no-references`: skip fetching reference lists (speeds up runs, removes true bibliographic coupling)

### Embedding Strategy

- `--model`, `-m`: sentence-transformer model name (e.g., `all-MiniLM-L6-v2`, `google/embeddinggemma-300m`)
- `--dataset-split`: HuggingFace split (`train`, `train[:5%]`, etc.)
- `--corpus-size`: maximum number of papers to load from dataset (default `50000`)
- `--all-corpus`: remove the default cap and process the full selected split
- `--top-k`, `-k`: strict per-node edge cap applied during embedding graph pruning
- `--truncate-dim`: optional embedding output dimension truncation (for EmbeddingGemma: `768`, `512`, `256`, `128`)
- `--streaming`: stream HuggingFace dataset instead of loading cached shards. Streaming requires a non-sliced split (for example `train`); use `--corpus-size` to cap runtime in streaming mode.
  
  _Note_: When using EmbeddingGemma, CiteMesh automatically applies the model card’s recommended query/document prompts, defaults to `256d` Matryoshka embeddings (available: `768/512/256/128`), and logs the selected dimension at model load. It also prefers `bfloat16` model loading with CUDA autocast; if BF16 is unavailable, it falls back to float32.

### Hybrid Strategy

- Inherits citation flags for paper collection, including `--no-references`
- Reuses embedding corpus/model knobs (`--dataset-split`, `--corpus-size`, `--all-corpus`, `--truncate-dim`, `--streaming`)
- `--max-semantic`: number of semantic neighbors to add when enriching the citation graph

## Export Formats

- `png`: Matplotlib static render with theme-aware background/labels.
- `html` (Pyvis): vis.js network with hover tooltips and force-physics enabled by default.
- `plotly`: interactive Plotly graph (HTML) suitable for notebook/dashboard embedding.
- `json`: structured graph data with nodes, edges, metadata.
- `graphml`: exchange format for Gephi, Cytoscape, and similar tools.

Determinism notes:

- `json` and `graphml` exports are deterministic by default.
- `png` and `plotly` are deterministic when using the same input graph and `--seed`.
- Pyvis `html` export uses deterministic node/edge ordering in the generated file, but runtime force physics remain non-deterministic in-browser.

Interactive exports require optional viz dependencies:

```bash
pip install -e ".[viz]"
```

## Examples

```bash
# Minimal citation graph with 20 nodes
citemesh build "arxiv:1706.03762" --strategy citation -p 20

# Hybrid graph with all export formats and dark mode
citemesh build "arxiv:1706.03762" --strategy hybrid --export all --theme dark

# Embedding graph with a fast model and small dataset slice
citemesh build "arxiv:1810.04805" \
  --strategy embedding \
  -m all-MiniLM-L6-v2 \
  --dataset-split "train[:2%]" \
  --export plotly

# Use a DOI and write outputs to a custom location
citemesh build "10.1145/3133956.3134029" \
  --strategy citation \
  --export png \
  -o reports/attention-visualization.png

# Search for relevant recent papers and build a recommendation graph
citemesh search "attention mechanism transformers" --limit 5
citemesh build "<paper-id-from-search>" --strategy recommendation

# Build directly from an arXiv URL
citemesh build "https://arxiv.org/abs/1706.03762" --strategy recommendation --export all
```

## Troubleshooting Tips

- **No results / paper not found**: confirm the identifier format and availability in Semantic Scholar. For embedding-only runs, free-form text can succeed even if the paper lacks metadata.
- **Slow embedding runs on first attempt**: the initial execution downloads HuggingFace data and computes embeddings. Subsequent runs reuse cached corpora and vectors.
- **Missing exports**: double-check `--export` values; unknown strings are rejected by argparse.
- **API limits**: set `S2_API_KEY` for higher Semantic Scholar rate limits, especially for recommendation and search heavy workflows.

```bash
export S2_API_KEY="your-key-here"
```

For cache behavior, use the canonical cache guide:
[Caching & Data](caching.md).

For architecture details and extension points:
[Architecture](../internals/architecture.md).
