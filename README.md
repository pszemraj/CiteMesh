# CiteMesh

Build exploration-friendly paper graphs from a single paper or query using recommendation, citation, embedding, or hybrid strategies. CiteMesh ships as a single CLI with consistent visuals and export formats so you can switch approaches without changing tools.

## Scope

This README is intentionally high-level.

- Canonical CLI behavior (flags, defaults, identifier normalization, outputs): [CLI Usage](./docs/guides/cli.md)
- Strategy behavior and tradeoffs: [Strategies Guide](./docs/guides/strategies.md)
- Canonical cache behavior (paths, layout, hydration/invalidation, cleanup): [Caching & Data](./docs/guides/caching.md)
- Full docs ownership map: [Documentation Index](./docs/README.md)
- This file intentionally avoids duplicating CLI/cache contracts; those remain centralized in the canonical docs above.

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

## Documentation

Use [Documentation Index](./docs/README.md) as the source-of-truth map. It defines ownership for each behavior topic.

## License

MIT License. See [LICENSE](LICENSE).
