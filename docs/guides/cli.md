# CLI Usage Guide

The `citemesh` command builds paper graphs using one of four strategies (`recommendation`, `citation`, `embedding`, `hybrid`). This guide walks through common flags and workflows.

This document is the canonical CLI reference for identifiers, flags, output naming, and export behavior.

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
- DOI URL (`https://doi.org/10.1038/nature14539`)
- arXiv ID (`arxiv:1706.03762`, `1706.03762`)
- arXiv URL (`https://arxiv.org/abs/1706.03762`, `https://arxiv.org/pdf/1706.03762.pdf`)
- Semantic Scholar Paper ID
- Free-form text query (embedding strategy treats it as a search query)

## Core Options

| Flag                 | Description                                                      | Default                        |
| -------------------- | ---------------------------------------------------------------- | ------------------------------ |
| `--strategy`, `-s`   | `recommendation`, `citation`, `embedding`, or `hybrid`           | `recommendation`               |
| `--max-papers`, `-p` | Maximum nodes in final graph                                     | `40`                           |
| `--iterations`, `-i` | Layout iterations (higher = smoother)                            | `100`                          |
| `--dpi`, `-d`        | PNG output resolution                                            | `150`                          |
| `--seed`             | Layout seed reused across exporters for reproducible positioning | none (uses default behavior)   |
| `--export`, `-e`     | One of `png`, `html`, `plotly`, `json`, `graphml`, or `all`     | `png`                          |
| `--theme`            | `light`, `dark`, `solarized`, `auto`                             | `light`                        |
| `--output`, `-o`     | Base filename for exports                                        | auto-generated in paper folder |

When `--output` is omitted, CiteMesh writes to `out/<safe_seed_title[:50]>/<strategy>.<ext>`.
When `--export all` is used, CiteMesh writes every supported format using consistent styling. If you specify a custom output path, the CLI appends the correct extension for each exported format.
Custom basenames containing dots (for example `-o out/arxiv-2508.14040-example`) are preserved; format extensions are appended without truncating the basename.

## Strategy-Specific Flags

### Recommendation Strategy

- Recommended default.
- Uses Semantic Scholar recommendations for fast, high-signal topical seeds.
- Reuses:
  - `--similarity-threshold`, `-t` for edge filtering.
  - `--no-references` to skip fetching references for the seed and recommendation neighbors.

### Citation Strategy

- `--max-citations`, `-c`: limit number of citing papers
- `--max-references`, `-r`: limit number of referenced papers
- `--similarity-threshold`, `-t`: minimum similarity score for edges
- `--no-references`: skip fetching reference lists (speeds up runs, removes true bibliographic coupling)

### Embedding Strategy

- `--model`, `-m`: sentence-transformer model name (e.g., `all-MiniLM-L6-v2`, `google/embeddinggemma-300m`)
- `--dataset-split`: HuggingFace split (`train`, `train[:5%]`, etc.)
- `--corpus-size`: maximum number of papers to load from dataset
- `--top-k`, `-k`: strict per-node edge cap applied during embedding graph pruning
- `--streaming`: stream HuggingFace dataset instead of loading cached shards (disabled by default)
  
  _Note_: When using EmbeddingGemma, CiteMesh automatically applies the model card’s recommended query/document prompts, defaults to `256d` Matryoshka embeddings (available: `768/512/256/128`), and logs the selected dimension at model load. It also prefers `bfloat16` model loading with CUDA autocast; if BF16 is unavailable, it falls back to float32.

### Hybrid Strategy

- Inherits citation flags for paper collection
- `--max-semantic`: number of semantic neighbors to add when enriching the citation graph

## Export Formats

- `png`: Matplotlib static render with theme-aware background/labels.
- `html` (Pyvis): vis.js network with hover tooltips and optional physics.
- `plotly`: interactive Plotly graph (HTML) suitable for notebook/dashboard embedding.
- `json`: structured graph data with nodes, edges, metadata.
- `graphml`: exchange format for Gephi, Cytoscape, and similar tools.

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
