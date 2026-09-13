# CiteMesh

CiteMesh turns one paper you already know into a graph of the research around it: the newer papers that follow up on it, and the older foundational work it rests on.

It runs locally from a single CLI with four switchable strategies — recommendation, citation, embedding, and hybrid — and exports the same graph to every format you might want.

![CiteMesh dashboard showing a Megalodon hybrid graph with LM-Infinite selected](assets/ui.png)

_A hybrid graph seeded by “Megalodon” over a full arXiv corpus index, with “LM-Infinite” selected to expose its semantic relation and shortest path to the seed (45 papers, 108 links)._

Try the same graph without running the pipeline: after cloning this repository, open [assets/examples/megalodon/dashboard.html](assets/examples/megalodon/dashboard.html) in your browser. The saved example works offline with no installation, API key, or model download. To practice importing results, click **Add Results** and select [dashboard.citemesh.json](assets/examples/megalodon/dashboard.citemesh.json) from the same folder; importing this example into its own dashboard refreshes the existing result rather than adding a duplicate. Use **Export Collection** to save results imported into your browser session.

## What you get

| | CiteMesh | Typical hosted graph tool |
| --- | --- | --- |
| Licensing | MIT, self-hosted CLI | Closed, web-only |
| Graph quota | None | Limited free graphs |
| Strategies | Four, switchable per run | One fixed algorithm |
| Semantic similarity | Local embeddings (CUDA / MPS / CPU) | Server-side, opaque |
| Outputs | PNG, HTML/Plotly, dashboard collections, JSON, CSV, BibTeX, GraphML | Screenshot or share link |
| Automation | Scriptable CLI, deterministic exports, JSON sidecars | Manual browsing |

Public beta, pre-1.0: export formats, cache layouts, and defaults may change between revisions unless documented otherwise. Runs on macOS (Apple Silicon, MPS), Linux, and Windows.

## Quick Start

Install PyTorch for your hardware first with the [official selector](https://pytorch.org/get-started/locally/), or pip may resolve a build you did not intend. Then install from GitHub (no PyPI release yet):

```bash
pip install "citemesh[recommended] @ git+https://github.com/pszemraj/CiteMesh.git"
```

Extras: `embeddings` (embedding and hybrid strategies), `viz` (HTML/Plotly exports), `recommended` (both), `all` (adds dev tooling). Omit the extra for the citation/recommendation-only CLI. Editable installs: [Contributing](CONTRIBUTING.md).

Python >= 3.10. Only the `embeddings` extra needs torch: `>=2.9` on Linux and Windows, `>=2.13` on macOS (the release verified for MPS bfloat16; float32 is the fallback wherever bf16 is unavailable). Details: [Embedding Runtime](docs/reference/embedding-runtime.md).

### Semantic Scholar API key (recommended)

CiteMesh runs without credentials on Semantic Scholar's shared anonymous pool, but an unkeyed first run can spend most of its time waiting out 429 retries. A free key from <https://www.semanticscholar.org/product/api> buys a dedicated 1 request/second budget — pass it as `export S2_API_KEY=...`, or persist it:

```bash
citemesh config set api.s2_api_key YOUR_KEY
```

### Run one graph

Builds default to Semantic Scholar recommendations. Local embeddings are opt-in with `--strategy embedding` or `--strategy hybrid`; add `--semantic-source arxiv-corpus` to swap the S2 candidates for a downloaded arXiv corpus. Any of these can be saved as personal defaults ([user configuration](docs/guides/configuration.md)).

```bash
# Default: Semantic Scholar recommendations
citemesh build "arxiv:1706.03762"

# Local embeddings over S2 candidates, plus citation links
citemesh build "arxiv:1706.03762" --strategy hybrid --export all

# Embedding search over a downloaded arXiv corpus
citemesh build "arxiv:1706.03762" --strategy embedding --semantic-source arxiv-corpus
```

The first embedding or hybrid run downloads `unsloth/embeddinggemma-300m` and encodes up to `--candidate-pool-size` abstracts (default 400); `arxiv-corpus` mode hydrates the full selected split unless capped with `--corpus-size N` ([CLI guide](docs/guides/cli.md)).

Omit `--output` and each build saves under `out/<title-slug>-<hash>/`, while repeated dashboard builds accumulate into a shared `out/dashboard.html` collection. Open it with `citemesh view`, or name a collection or file (`citemesh view out/my-collection`) — see [Output Artifacts](docs/reference/output-artifacts.md).

## How it works

Every strategy runs the same eight-stage pipeline; they differ only in where candidates come from and how pairs are scored.

![The eight stages of a CiteMesh build: seed resolution, candidate acquisition, embedding, caching, ranking, edge scoring, layout, and export](assets/how-it-works.png)

_Defaults for a hybrid build. Recommendation and citation builds skip stages 3 and 4: they never load the model and score pairs from TF-IDF instead._

Four flags move most of the outcome. `--candidate-pool-size` sets how many papers are fetched and encoded, `--max-papers` caps how many survive ranking, `--min-semantic-similarity` is the cosine gate a pair must clear before temporal, category, and shared-author signals adjust its weight, and `--seed` fixes the layout so every export of one graph lines up.

The full walkthrough, with the numbers that matter at each stage: [How CiteMesh builds a graph](docs/guides/how-it-works.md). For syntax and operational detail: [CLI guide](docs/guides/cli.md), [Strategy guide](docs/guides/strategies.md), [Caching & Data](docs/guides/caching.md), [User configuration](docs/guides/configuration.md), and the [documentation index](docs/README.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and feature requests are welcome via [issues](https://github.com/pszemraj/CiteMesh/issues).

## License

MIT License. See [LICENSE](LICENSE).
