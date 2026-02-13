# CiteMesh

Build exploration-friendly paper graphs from a single paper or query using recommendation, citation, embedding, or hybrid strategies. CiteMesh ships as a single CLI with consistent visuals and export formats so you can switch approaches without changing tools.

Documentation is organized with canonical sources per topic. Start from [docs/README.md](docs/README.md).

## Quick Start

### Install

Install from GitHub:

```bash
pip install "git+https://github.com/pszemraj/CiteMesh.git"
```

For development:

```bash
git clone https://github.com/pszemraj/CiteMesh.git && cd CiteMesh
pip install -e ".[dev]"
```

Install embedding dependencies (needed for `--strategy embedding` and semantic enrichment in `--strategy hybrid`):

```bash
pip install -e ".[embeddings]"
```

Install interactive visualization dependencies (needed for `--export html` and `--export plotly`):

```bash
pip install -e ".[viz]"
```

### Use the CLI

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

- One CLI for recommendation, citation, embedding, and hybrid graphs.
- Recommendation strategy is now the default discovery path and uses Semantic Scholar recommendations.
- Multi-format outputs: PNG, Pyvis HTML, Plotly HTML, JSON, GraphML.
- Theme-aware visuals (light, dark, solarized, auto) shared across exporters.
- Persistent user-level caching for embeddings and corpora; re-runs are fast.
- Typed models and strategy abstraction make extensions straightforward.

## Canonical References

- CLI behavior (identifiers, flags, output naming, export behavior): [docs/guides/cli.md](docs/guides/cli.md)
- Cache and storage behavior: [docs/guides/caching.md](docs/guides/caching.md)
- Architecture and extension points: [docs/internals/architecture.md](docs/internals/architecture.md)

## API Key

Set `S2_API_KEY` for higher Semantic Scholar rate limits:

```bash
export S2_API_KEY="your-semantic-scholar-key"
```

For API-related CLI behavior (retries, limits, identifier normalization), see:
[docs/guides/cli.md](docs/guides/cli.md).

## Documentation

Use [docs/README.md](docs/README.md) as the documentation index and source-of-truth map.

## Development

Install dev dependencies and run tests:

```bash
pip install -e .[dev]
pytest
```

MIT License - see [LICENSE](LICENSE).
