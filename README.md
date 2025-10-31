# Paper Graph Visualizer (CiteMesh)

CiteMesh builds exploration-ready citation graphs from a single paper identifier (DOI, arXiv ID, Semantic Scholar ID, or even a free-form query for the embedding workflow). Three strategies – citation, embedding, and hybrid – share a unified CLI, visualization pipeline, and export system so you can jump between approaches without changing tooling.

## Highlights

- **Single CLI** with interchangeable strategies (`citation`, `embedding`, `hybrid`)
- **Multi-format exports**: high-res PNG, interactive Pyvis HTML, Plotly dashboards, JSON and GraphML raw data
- **Theme system**: light, dark, solarized, or auto-detected visuals across static and interactive outputs
- **True bibliographic coupling** using real reference lists, plus semantic similarity and author/category signals
- **Persistent caching** with user-level storage (`~/.cache/citemesh` or OS equivalent) for HuggingFace corpora, embeddings, and joblib results
- **Type-safe core**: dataclass-based `Paper`/`Author` models and strategy abstractions

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/paper-graph-vis.git
cd paper-graph-vis

# Install in editable mode (installs optional exporters + caching deps)
pip install -e .

# Build a citation graph with every export and a dark theme
citemesh build "arxiv:1706.03762" \
  --strategy hybrid \
  --export all \
  --theme dark
```

Exports land in the `out/` directory by default (one file per requested format). Embedding and dataset caches live under the user cache root (override with `CITEMESH_CACHE_DIR` if needed).

## Usage

```bash
# Citation strategy with fewer nodes
citemesh build "arxiv:1706.03762" --strategy citation -p 25

# Embedding strategy with a fast model and small slice
citemesh build "arxiv:1706.03762" --strategy embedding \
  -m all-MiniLM-L6-v2 \
  --dataset-split "train[:5%]"

# Hybrid strategy (default) with custom output filename
citemesh build "arxiv:1706.03762" --strategy hybrid \
  --export png \
  -o out/transformer-hybrid.png

# Interactive exports only
citemesh build "10.1038/nature14539" --strategy embedding \
  --export html plotly \
  --theme solarized

# Detailed help
citemesh build --help
```

Key flags:

- `--export` / `-e`: one or more of `png`, `html` (Pyvis), `plotly`, `json`, `graphml`, or `all`
- `--theme`: `light`, `dark`, `solarized`, or `auto`
- `--streaming`: opt-in for streaming the full HuggingFace corpus (defaults to cached on-disk loads)
- Strategy-specific controls (`--max-citations`, `--top-k`, `--max-semantic`, etc.) surface in help output

## What Each Strategy Does

| Strategy   | Data Sources                                               | Similarity Signals                                                     | Notes |
|------------|------------------------------------------------------------|------------------------------------------------------------------------|-------|
| citation   | Semantic Scholar seed + citations + references             | Temporal proximity, citation impact (log scale), true bibliographic coupling | Favors historically grounded context |
| embedding  | HuggingFace ML ArXiv corpus + sentence transformers        | Normalized embedding cosine, temporal factor, category overlap, author overlap bonus; optional citation enrichment | Works offline once corpus cached |
| hybrid     | Everything above, plus semantic enrichment for citation graph | Mixes citation-derived weights with semantic neighbors, enforces per-node edge caps | Recommended default |

All strategies feed into the same visualization + export pipeline, so sizing, coloring, and metadata alignment are consistent.

## Visualization & Exports

- **Layout**: Kamada-Kawai with light perturbations for organic clusters
- **Node size**: tiered citation ranking with logarithmic boosts; seeds get special treatment
- **Node color**: theme-aware continuous gradient by publication year; seeds use explicit theme colors
- **Edges**: weight-proportional opacity/thickness
- **Metadata block**: auto-positioned subtitle showing strategy, counts, timestamps

Exports reuse these encodings:

- **PNG**: Matplotlib static render
- **HTML (Pyvis)**: vis.js interactive network with hover tooltips and physics controls
- **HTML (Plotly)**: inspector-ready layout for dashboards/notebooks
- **JSON**: graph structure + metadata for programmatic reuse
- **GraphML**: plug directly into tools like Gephi or Cytoscape

## Caching & Performance

- **Embedding cache**: SQLite metadata + HDF5 vectors stored at `~/.cache/citemesh/embeddings/<model-hash>`
- **Joblib dataset cache**: `~/.cache/citemesh/joblib`
- **HuggingFace datasets**: respect the standard HF cache (`~/.cache/huggingface`)
- **CLI defaults** avoid streaming; re-runs pull embeddings from cache in milliseconds once computed

Override the base cache directory by setting `CITEMESH_CACHE_DIR=/custom/path`.

## Architecture Overview

- `citemesh/cli.py`: argparse-powered entry point with per-strategy builders
- `citemesh/strategies/`: concrete `GraphBuilderStrategy` implementations (`citation`, `embedding`, `hybrid`)
- `citemesh/visualization.py`: layout, sizing, labeling, and metadata rendering
- `citemesh/themes.py`: theme registry and helpers used by both static and interactive outputs
- `citemesh/export.py`: multi-format `GraphExporter`
- `citemesh/embedding_cache.py`: persistent embedding cache service
- `citemesh/cache_utils.py`: platform-aware cache directory resolution

Browse the [docs index](docs/README.md) for usage guides, caching details, a deeper architecture dive, and the changelog.

## Development

```bash
pip install -e .[dev]
pytest
```

Integration tests run the CLI with low-cost configurations; outputs are written to temporary directories to keep the `out/` folder clean.

## License

MIT License – see [LICENSE](LICENSE) for details.
