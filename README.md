# CiteMesh

Build exploration-friendly paper graphs from a single paper or query using recommendation, citation, embedding, or hybrid strategies. CiteMesh ships as a single CLI with consistent visuals and export formats so you can switch approaches without changing tools.

![CiteMesh UI](assets/ui.png)

## Core Use Case

Start from one paper you already know, then quickly discover:

- newer papers that are genuinely related to that paper's core topic
- older, high-quality foundational papers that matter for understanding the same area

The CLI defaults to the `recommendation` strategy for the fastest topical pass.
Use `--strategy hybrid` when you want the tuned citation-plus-semantic workflow for this discovery pattern.

## Project Status

CiteMesh is currently a private, fast-moving pre-release tool. The project
optimizes for discovery quality, correctness, and simpler internals over
backward compatibility. Older exports, checkpoints, or intermediate artifacts
may stop working between revisions unless explicitly documented otherwise.

## Documentation

[Documentation index](docs/README.md)

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

For command syntax and operational details, use:

- [CLI Usage](docs/guides/cli.md)
- [Strategy Guide](docs/guides/strategies.md)
- [Caching & Data](docs/guides/caching.md)

## Why CiteMesh

- One CLI for recommendation, citation, embedding, and hybrid graphs.
- Built for seed-paper-driven discovery of both recent follow-up work and foundational prior work.
- Multi-format outputs with one shared run contract across static, interactive, and structured exports.
- Persistent user-level caching for embeddings and reference expansion.
- Typed, modular internals that are straightforward to extend.

## License

MIT License. See [LICENSE](LICENSE).
