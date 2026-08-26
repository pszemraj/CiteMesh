# CiteMesh

Build exploration-friendly paper graphs from a single paper or query using recommendation, citation, embedding, or hybrid strategies. CiteMesh ships as a single CLI with consistent visuals and export formats so you can switch approaches without changing tools — an open-source, embeddings-powered take on the reference tool workflow.

![CiteMesh UI](assets/ui.png)

## Core Use Case

Start from one paper you already know, then quickly discover:

- newer papers that are genuinely related to that paper's core topic
- older, high-quality foundational papers that matter for understanding the same area

The CLI defaults to the `recommendation` strategy for the fastest topical pass. Use `--strategy hybrid` when you want the tuned citation-plus-semantic workflow for this discovery pattern.

## Why CiteMesh (vs. reference tool and similar tools)

| | CiteMesh | Typical hosted graph tools |
| --- | --- | --- |
| Open source | MIT, self-hosted CLI | Closed, web-only |
| Graph quota | None — build as many as you like | Limited free graphs |
| Strategies | Recommendation, citation, embedding, hybrid | Single fixed algorithm |
| Semantic similarity | Your embeddings, computed locally (CUDA / Apple Silicon MPS / CPU) | Opaque server-side |
| Outputs | PNG, interactive HTML/Plotly, self-contained dashboard, JSON, CSV, BibTeX, GraphML | Screenshot or share link |
| Automation | Scriptable CLI with deterministic exports and JSON sidecars | Manual browsing |

By default the embedding and hybrid strategies are **corpus-free**: they embed only the seed's Semantic Scholar neighbors (references/citations/ recommendations), so a laptop builds a graph in seconds — no multi-gigabyte corpus download required. An opt-in local arXiv corpus mode (`--semantic-source arxiv-corpus`) is available for corpus-scale retrieval.

## Project Status

Public beta, pre-1.0. CiteMesh optimizes for discovery quality, correctness, and simple internals over backward compatibility: export formats, cache layouts, and defaults may change between revisions unless explicitly documented otherwise. macOS (Apple Silicon, MPS), Linux, and Windows are supported; CI covers Linux and macOS.

## Documentation

[Documentation index](docs/README.md)

## Quick Start

### Install

Install from GitHub:

```bash
pip install "citemesh[recommended] @ git+https://github.com/pszemraj/CiteMesh.git"
```

For local development:

```bash
git clone https://github.com/pszemraj/CiteMesh.git && cd CiteMesh
pip install -e ".[dev,viz]"
```

Optional extras:

```bash
# Minimal citation/recommendation CLI only
pip install "git+https://github.com/pszemraj/CiteMesh.git"

# Recommended runtime bundle: embeddings + interactive exports
pip install -e ".[recommended]"

# Embedding strategy + semantic enrichment support
pip install -e ".[embeddings]"

# Interactive HTML/Plotly exports
pip install -e ".[viz]"

# Everything currently defined by the project, including dev tools
pip install -e ".[all]"
```

On macOS the `embeddings` extra requires torch >= 2.13 (installed automatically) and runs on the MPS backend with bfloat16 by default; see [Embedding Runtime](docs/reference/embedding-runtime.md).

### Run One Graph

```bash
citemesh build "arxiv:1706.03762" --strategy hybrid --export all --theme dark
```

For command syntax and operational details, use:

- [CLI Usage](docs/guides/cli.md)
- [Strategy Guide](docs/guides/strategies.md)
- [Caching & Data](docs/guides/caching.md)
- [User Configuration](docs/guides/configuration.md)

### Semantic Scholar API key (recommended)

CiteMesh works without credentials using Semantic Scholar's shared anonymous pool, but that pool is small and 429 rate-limit errors are common. A free API key gives you a dedicated 1 request/second budget:

1. Request a key at <https://www.semanticscholar.org/product/api>
2. Provide it via the environment (`export S2_API_KEY=...`) or persist it:

   ```bash
   citemesh config set api.s2_api_key YOUR_KEY
   ```

## Highlights

- One CLI for recommendation, citation, embedding, and hybrid graphs.
- Built for seed-paper-driven discovery of both recent follow-up work and foundational prior work.
- Local embeddings with first-class device support: CUDA, Apple Silicon (MPS, bf16), CPU.
- Multi-format outputs with one shared run contract across static, interactive, and structured exports.
- Persistent user-level caching for embeddings and reference expansion; caches are portable across machines at matching compute dtype.
- Persistent personal defaults via `citemesh config` (`config.toml`).
- Typed, modular internals that are straightforward to extend.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and feature requests are welcome via [issues](https://github.com/pszemraj/CiteMesh/issues).

## License

MIT License. See [LICENSE](LICENSE).
