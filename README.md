# CiteMesh

Build exploration-friendly citation graphs from a single paper or query using citation, embedding, or hybrid strategies. CiteMesh ships as a single CLI with consistent visuals and export formats so you can jump between approaches without changing tools.

## Quick Start

```bash
git clone https://github.com/yourusername/paper-graph-vis.git
cd paper-graph-vis
pip install -e .
```

Example: hybrid graph with all exports in dark mode:

```bash
# Example: hybrid graph with all exports in dark mode
citemesh build "arxiv:1706.03762" --strategy hybrid --export all --theme dark
```

Search by title/keyword first:

```bash
citemesh search "attention mechanism"
```

## Why CiteMesh

- One CLI for citation, embedding, and hybrid graphs.
- Recommendation strategy is now the default discovery path and uses Semantic Scholar recommendations.
- Multi-format outputs: PNG, Pyvis HTML, Plotly HTML, JSON, GraphML.
- Theme-aware visuals (light, dark, solarized, auto) shared across exporters.
- Persistent user-level caching for embeddings and corpora; re-runs are fast.
- Typed models and strategy abstraction make extensions straightforward.

## Essentials

- `citemesh build "<paper-id>" --strategy <recommendation|citation|embedding|hybrid> [options]`
- Identifiers: DOI (`10.1038/...`) or DOI URL (`https://doi.org/...`), arXiv (`arxiv:1706.03762`, `1706.03762`, or `https://arxiv.org/abs/...`), Semantic Scholar Paper ID, or free-form text (embedding strategy).
- `--export` accepts any combination of `png`, `html`, `plotly`, `json`, `graphml`, or `all`.
- `--theme` selects a colour palette for both static and interactive outputs.

## API Key

Set `S2_API_KEY` for higher Semantic Scholar rate limits:

```bash
export S2_API_KEY="your-semantic-scholar-key"
```

- Without a key: lower request limits (historically ~100 requests / 5 minutes).
- With a key: much higher throughput for discovery workflows (historically ~100 requests / second).

## Documentation

See [docs/README.md](docs/README.md) for:

- CLI usage details and strategy-specific options.
- Cache locations and management tips.
- Architecture internals and changelog.

## Development

install with dev dependencies and run tests:

```bash
pip install -e .[dev]
pytest
```

MIT License - see [LICENSE](LICENSE).
