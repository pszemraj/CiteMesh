# CiteMesh

Build exploration-friendly paper graphs from a known paper with recommendation,
citation, embedding, or hybrid strategies, or start the embedding strategy from a
free-text query. CiteMesh ships as a single CLI with consistent visuals and export
formats so you can switch approaches without changing tools -- an open-source,
embeddings-powered alternative to hosted literature-mapping services.

![CiteMesh dashboard showing a hybrid graph for Attention is All you Need](assets/ui.png)

_Current dashboard rendered from a repository-local hybrid run for “Attention is
All you Need” (45 papers, 107 links)._

## Core Use Case

Start from one paper you already know, then quickly discover:

- newer papers that are genuinely related to that paper's core topic
- older, high-quality foundational papers that matter for understanding the same area

The CLI defaults to the `recommendation` strategy for the fastest topical pass. Use `--strategy hybrid` when you want the tuned citation-plus-semantic workflow for this discovery pattern.

## Why CiteMesh (vs. hosted literature-mapping tools)

| | CiteMesh | Typical hosted graph tools |
| --- | --- | --- |
| Open source | MIT, self-hosted CLI | Closed, web-only |
| Graph quota | None - build as many as you like | Limited free graphs |
| Strategies | Recommendation, citation, embedding, hybrid | Single fixed algorithm |
| Semantic similarity | Your embeddings, computed locally (CUDA / Apple Silicon MPS / CPU) | Opaque server-side |
| Outputs | PNG, interactive HTML/Plotly, reusable dashboard collections, JSON, CSV, BibTeX, GraphML | Screenshot or share link |
| Automation | Scriptable CLI with deterministic exports and JSON sidecars | Manual browsing |

By default the embedding and hybrid strategies are **corpus-free**. For known-paper
seeds they embed Semantic Scholar neighbors instead of downloading a local corpus.
An opt-in arXiv mode (`--semantic-source arxiv-corpus`) is available for
corpus-scale retrieval.

## Project Status

Public beta, pre-1.0. CiteMesh optimizes for discovery quality, correctness, and simple internals over backward compatibility: export formats, cache layouts, and defaults may change between revisions unless explicitly documented otherwise. macOS (Apple Silicon, MPS), Linux, and Windows are supported.

## Documentation

[Documentation index](docs/README.md)

## Quick Start

### Install

Before installing the recommended or embeddings extras, install PyTorch for your
hardware using the [official installation selector](https://pytorch.org/get-started/locally/).
Otherwise, pip may install a build you did not intend to use. Then install from GitHub:

```bash
pip install "citemesh[recommended] @ git+https://github.com/pszemraj/CiteMesh.git"
```

Optional extras:

```bash
# Minimal citation/recommendation CLI only
pip install "citemesh @ git+https://github.com/pszemraj/CiteMesh.git"

# Embedding strategy + semantic enrichment support
pip install "citemesh[embeddings] @ git+https://github.com/pszemraj/CiteMesh.git"

# Interactive HTML/Plotly exports
pip install "citemesh[viz] @ git+https://github.com/pszemraj/CiteMesh.git"
```

For an editable development install, follow [Contributing](CONTRIBUTING.md).

On macOS the `embeddings` extra requires torch >= 2.13 and runs on the MPS backend with bfloat16 autocast when supported, falling back to float32; see [Embedding Runtime](docs/reference/embedding-runtime.md).

### Run One Graph

```bash
citemesh build "arxiv:1706.03762" --strategy hybrid --export all --theme dark
```

Omit `--output` to keep generated files under the repository's ignored `out/`
directory. Repeated dashboard builds share an offline collection there:

```bash
citemesh build "arxiv:1706.03762" --strategy hybrid --export dashboard
citemesh build "arxiv:1810.04805" --strategy recommendation --export dashboard
```

Each build saves its graph and settings as `out/<paper-slug>-<hash>/<strategy>.json`
and `<strategy>.config.json`. The shared `out/dashboard.html` viewer and
`out/dashboard.citemesh.json` package collect those results for browsing and sharing.
Different seeds add results; rebuilding the same seed with the same strategy
replaces its result. Use the dashboard's graph selector to switch papers.

See [Output Artifacts](docs/reference/output-artifacts.md) for collection updates,
standalone dashboard files, and the portable package format.

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
- Local embeddings with first-class device support: CUDA, Apple Silicon (MPS), and CPU, using bf16 autocast on supported runtimes.
- Multi-format outputs with one shared run contract across static, interactive, and structured exports; dashboard collections retain separate graph files per seed.
- Persistent user-level caching for embeddings and reference expansion; caches are portable across machines at matching compute dtype.
- Mode-aware `citemesh search`: offline semantic search over the active retrieval
  cache, used automatically when that namespace has vectors, with Semantic Scholar
  keyword search as the fallback (`--mode local|s2|auto`).
- Persistent personal defaults via `citemesh config` (`config.toml`).
- Typed, modular internals that are straightforward to extend.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and feature requests are welcome via [issues](https://github.com/pszemraj/CiteMesh/issues).

## License

MIT License. See [LICENSE](LICENSE).
