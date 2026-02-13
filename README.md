# CiteMesh

Build exploration-friendly paper graphs from a single paper or query using recommendation, citation, embedding, or hybrid strategies. CiteMesh ships as a single CLI with consistent visuals and export formats so you can switch approaches without changing tools.

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

## Essentials

- Core command:
  `citemesh build "<paper-id>" --strategy <recommendation|citation|embedding|hybrid> [options]`
- Canonical CLI reference (identifiers, flags, output naming, export behavior):
  [docs/guides/cli.md](docs/guides/cli.md)
- Cache/storage behavior:
  [docs/guides/caching.md](docs/guides/caching.md)

## API Key

Set `S2_API_KEY` for higher Semantic Scholar rate limits:

```bash
export S2_API_KEY="your-semantic-scholar-key"
```

For full API-behavior notes, see the canonical CLI guide:
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
