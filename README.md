# CiteMesh

Build exploration-friendly paper graphs from a known paper with recommendation, citation, embedding, or hybrid strategies, or start the embedding strategy from a free-text query. CiteMesh ships as a single CLI with consistent visuals and export formats so you can switch approaches without changing tools -- an open-source alternative to hosted literature-mapping services.

**By default, CiteMesh uses Semantic Scholar recommendations.** Local embeddings are opt-in: choose `--strategy embedding` or `--strategy hybrid`. Both use Semantic Scholar candidates by default; to download and search a local arXiv corpus, also set `--semantic-source arxiv-corpus`. You can save these choices as personal defaults with [user configuration](docs/guides/configuration.md).

![CiteMesh dashboard showing a Megalodon hybrid graph with LM-Infinite selected](assets/ui.png)

_A repository-local hybrid graph seeded by “Megalodon,” with “LM-Infinite” selected to expose its semantic relation and shortest path to the seed (45 papers, 108 links)._

## Core Use Case

Start from one paper you already know, then quickly discover:

- newer papers that are genuinely related to that paper's core topic
- older, high-quality foundational papers that matter for understanding the same area

Use `--strategy hybrid` to combine citation links and local semantic ranking for this discovery pattern.

## Why CiteMesh (vs. hosted literature-mapping tools)

| | CiteMesh | Typical hosted graph tools |
| --- | --- | --- |
| Licensing | MIT, self-hosted CLI | Closed, web-only |
| Graph quota | None | Limited free graphs |
| Strategies | Four, switchable per run | One fixed algorithm |
| Semantic similarity | Your embeddings, computed locally (CUDA / MPS / CPU) | Opaque server-side |
| Outputs | PNG, HTML/Plotly, dashboard collections, JSON, CSV, BibTeX, GraphML | Screenshot or share link |
| Automation | Scriptable CLI, deterministic exports, JSON sidecars | Manual browsing |

## Project Status

Public beta, pre-1.0. CiteMesh optimizes for discovery quality, correctness, and simple internals over backward compatibility: export formats, cache layouts, and defaults may change between revisions unless explicitly documented otherwise. macOS (Apple Silicon, MPS), Linux, and Windows are supported.

## Documentation

[Documentation index](docs/README.md) — start with the [CLI guide](docs/guides/cli.md) and [How CiteMesh builds a graph](docs/guides/how-it-works.md).

## Quick Start

### Install

Install PyTorch for your hardware first, using the [official installation selector](https://pytorch.org/get-started/locally/) — otherwise pip may resolve a build you did not intend. Then install from GitHub (no PyPI release yet):

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

Python >= 3.10. The `embeddings` extra requires torch `>=2.9` on Linux and Windows and `>=2.13` on macOS — the macOS floor is the release verified for reliable MPS bfloat16 execution, and CiteMesh falls back to float32 where bf16 is unavailable or unverified. Everything except `embeddings` runs without torch. Details: [Embedding Runtime](docs/reference/embedding-runtime.md).

For an editable development install, follow [Contributing](CONTRIBUTING.md).

### Semantic Scholar API key (recommended)

CiteMesh works without credentials using Semantic Scholar's shared anonymous pool, but that pool is small and 429 rate-limit errors are common — an unkeyed first run can spend most of its time waiting out retries. A free API key gives you a dedicated 1 request/second budget:

1. Request a key at <https://www.semanticscholar.org/product/api>
2. Provide it via the environment (`export S2_API_KEY=...`) or persist it:

   ```bash
   citemesh config set api.s2_api_key YOUR_KEY
   ```

### Run One Graph

```bash
# Default: Semantic Scholar recommendations
citemesh build "arxiv:1706.03762" --export all --theme dark

# Opt in to local embeddings over Semantic Scholar candidates, plus citation links
citemesh build "arxiv:1706.03762" --strategy hybrid --export all --theme dark

# Opt in to embedding search over a downloaded arXiv corpus
citemesh build "arxiv:1706.03762" --strategy embedding --semantic-source arxiv-corpus
```

The default recommendation strategy does not download an embedding model or corpus. The first embedding or hybrid run downloads the `unsloth/embeddinggemma-300m` checkpoint (~300M parameters). With the default Semantic Scholar candidate source, it encodes up to 400 candidate abstracts. When you opt into arXiv corpus mode, CiteMesh hydrates the full selected split by default; add `--corpus-size N` to choose a smaller newest-first corpus. Later runs reuse the persistent embedding cache. See [CLI Usage](docs/guides/cli.md) for corpus size and loading options.

Omit `--output` and generated files land in `out/` under the current working directory (a source checkout already gitignores that path). Each build saves its graph under `out/<title-slug>-<hash>/`, and repeated dashboard builds accumulate into a shared `out/dashboard.html` viewer — see [Output Artifacts](docs/reference/output-artifacts.md) for the collection semantics, standalone dashboard files, and the portable package format.

```bash
citemesh build "arxiv:1706.03762" --strategy hybrid --export dashboard
citemesh build "arxiv:1810.04805" --strategy recommendation --export dashboard
```

Open the saved default viewer with `citemesh view`, a named collection with `citemesh view out/my-collection`, or an explicit HTML file with `citemesh view out/report.dashboard.html --browser google-chrome`.

## How it works

Every strategy runs the same eight-stage pipeline; they differ only in where candidates come from and how pairs are scored.

1. **Seed resolution** — your DOI / arXiv ID / URL / S2 ID is normalized to a canonical form and resolved against Semantic Scholar. For `--strategy embedding`, an identifier S2 cannot resolve is reinterpreted as a free-text query.
2. **Candidate acquisition** — references, citations, and recommendations are fetched within a `--candidate-pool-size` budget (default 400, split roughly 1:2:1) and de-duplicated across S2 / arXiv / DOI identities.
3. **Embedding** — EmbeddingGemma encodes the seed as a retrieval *query* and candidates as retrieval *documents*, at 512 dimensions in float32.
4. **Caching** — vectors persist in a SQLite + HDF5 cache keyed by a namespace fingerprint, so later runs skip encoding.
5. **Ranking and selection** — candidates are ranked against the seed and cut down to `--max-papers` (default 40; hybrid 45).
6. **Edge scoring** — selected papers are re-encoded with a *symmetric* prompt; pairs need cosine >= 0.74 (`--min-semantic-similarity`) before temporal, category, and shared-author signals adjust the weight, then per-node edge caps prune the graph.
7. **Layout** — one deterministic layout (`--seed`) is computed in Python and shared by every layout-based export.
8. **Export** — a single graph payload is rendered to PNG, Plotly, dashboard, JSON, CSV, BibTeX, and GraphML.

The full walkthrough, with every threshold, budget, and knob: [How CiteMesh builds a graph](docs/guides/how-it-works.md).

For command syntax and operational details, use:

- [CLI Usage](docs/guides/cli.md)
- [Strategy Guide](docs/guides/strategies.md)
- [Caching & Data](docs/guides/caching.md)
- [User Configuration](docs/guides/configuration.md)

## Highlights

- One CLI for recommendation, citation, embedding, and hybrid graphs.
- Built for seed-paper-driven discovery of both recent follow-up work and foundational prior work.
- Local embeddings with first-class device support: CUDA, Apple Silicon (MPS), and CPU, using bf16 autocast on supported runtimes.
- Multi-format outputs with one shared run contract across static, interactive, and structured exports; dashboard collections retain separate graph files per seed.
- Persistent user-level caching for embeddings and reference expansion; caches are portable across machines at matching compute dtype.
- Mode-aware `citemesh search`: offline semantic search over the active retrieval cache, used automatically when that namespace has vectors, with Semantic Scholar keyword search as the fallback (`--mode local|s2|auto`).
- Persistent personal defaults via `citemesh config` (`config.toml`).
- Typed, modular internals that are straightforward to extend.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and feature requests are welcome via [issues](https://github.com/pszemraj/CiteMesh/issues).

## License

MIT License. See [LICENSE](LICENSE).
