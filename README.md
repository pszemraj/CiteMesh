# CiteMesh

Build exploration-friendly citation graphs from a single paper or query using citation, embedding, or hybrid strategies. CiteMesh ships as a single CLI with consistent visuals and export formats so you can jump between approaches without changing tools.

## Quick Start

```bash
git clone <repository-url>
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
- See [CLI Usage](docs/guides/cli.md) for supported identifiers, full flag reference, and export behavior.
- `--theme` selects a colour palette for both static and interactive outputs.

## API Key

Set `S2_API_KEY` for higher Semantic Scholar rate limits:

```bash
export S2_API_KEY="your-semantic-scholar-key"
```

- Without a key: conservative throttling is applied by default.
- With a key: Semantic Scholar may grant a higher quota and better throughput.

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
