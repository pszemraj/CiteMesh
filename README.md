# CiteMesh

Build exploration-friendly paper graphs from a single paper or query using recommendation, citation, embedding, or hybrid strategies. CiteMesh ships as a single CLI with consistent visuals and export formats so you can switch approaches without changing tools.

## Documentation

- Documentation index: [docs/README.md](./docs/README.md)
- CLI guide: [docs/guides/cli.md](./docs/guides/cli.md)
- Strategy guide: [docs/guides/strategies.md](./docs/guides/strategies.md)
- Caching and data: [docs/guides/caching.md](./docs/guides/caching.md)
- Environment variables: [docs/reference/environment.md](./docs/reference/environment.md)
- Embedding runtime behavior: [docs/reference/embedding-runtime.md](./docs/reference/embedding-runtime.md)

## Quick Start

### Install

Install from GitHub:

```bash
pip install "git+https://github.com/pszemraj/CiteMesh.git"
```

For local development:

```bash
git clone https://github.com/pszemraj/CiteMesh.git && cd CiteMesh
pip install -e ".[dev]"
```

Optional extras:

```bash
# Embedding strategy + semantic enrichment support
pip install -e ".[embeddings]"

# Interactive HTML/Plotly exports
pip install -e ".[viz]"

# all
pip install -e ".[all]"
```

### Run One Graph

```bash
citemesh build "arxiv:1706.03762" --strategy hybrid --export all --theme dark
```

For complete command behavior and examples, use [CLI Usage](./docs/guides/cli.md).

## Why CiteMesh

- One CLI for recommendation, citation, embedding, and hybrid graphs.
- Multi-format outputs: PNG, Pyvis HTML, Plotly HTML, JSON, GraphML.
- Theme-aware visuals shared across exporters.
- Persistent user-level caching for embeddings and corpus data.
- Typed, modular architecture that is straightforward to extend.

## License

MIT License. See [LICENSE](LICENSE).
