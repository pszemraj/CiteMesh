# Python API

**Status: unstable, pre-1.0.** The CLI is the supported interface. These imports exist because the internals are already typed and modular, and driving them from a notebook or a script is often more convenient than shelling out — but they are not a frozen contract. Signatures, keyword names, and payload shapes can change in any release, and only the names listed below are intended for import at all. Anything reached through a leading underscore, or through a submodule not named here, is internal.

## What is exported

`citemesh.__all__` (strategy builders resolve lazily on first attribute access, so importing `citemesh` does not pull in torch):

| Name | Defined in | Purpose |
| --- | --- | --- |
| `Paper` | `citemesh.core.models` | validated paper metadata; graph nodes carry one under the `paper` attribute |
| `Author` | `citemesh.core.models` | author record attached to a `Paper` |
| `GraphBuilderStrategy` | `citemesh.strategies.base` | the template base class; subclass it to add a strategy |
| `CitationGraphBuilder` | `citemesh.strategies.citation` | references + citations + bibliographic coupling |
| `RecommendationGraphBuilder` | `citemesh.strategies.recommendation` | Semantic Scholar recommendations |
| `EmbeddingGraphBuilder` | `citemesh.strategies.embedding` | dense vector similarity |
| `HybridGraphBuilder` | `citemesh.strategies.hybrid` | citation candidates plus semantic enrichment |

`citemesh.visualization` additionally exports `GraphExporter`, `compute_layout`, `visualize_graph`, `generate_output_path`, `get_theme`, and `THEMES`.

Every builder implements the same two-method contract from `GraphBuilderStrategy`:

```python
build_graph(seed_id: str, **kwargs) -> tuple[networkx.Graph, str]   # returns (graph, resolved_seed_id)
collect_papers(seed_id: str, **kwargs) -> dict[str, Paper]
```

`build_graph` returns the *resolved* seed ID, which may differ from what you passed in — Semantic Scholar can answer an arXiv ID with its own paper ID, and the free-text embedding path returns a synthesized `query:<digest>` ID. Always use the returned value downstream; that is what `GraphExporter` and the layout helpers expect.

## Exporting

```python
GraphExporter(graph, seed_id, metadata=None, theme_name="dark", layout=None)
```

| Method | Writes |
| --- | --- |
| `graph_payload() -> dict` | the canonical payload every format is derived from |
| `to_json(path)` | graph JSON, including dashboard geometry |
| `to_csv(path)` | flat paper table |
| `to_bibtex(path)` | combined BibTeX entries |
| `to_graphml(path)` | GraphML for Gephi / Cytoscape |
| `to_plotly_html(path, theme=None)` | interactive Plotly page |
| `to_dashboard_html(path, theme=None)` | standalone dashboard viewer |
| `to_interactive_html(path, theme=None, physics=True)` | Pyvis network (browser physics; ignores `layout`) |

Every writer takes a `pathlib.Path` and creates nothing above it — make the parent directory yourself. Passing a `layout` is optional; omit it and the exporter computes one lazily on first use. Pass one when you want several exporters, or several runs, to share identical geometry.

The collection-package machinery behind `out/dashboard.html` lives in `citemesh.visualization.dashboard.package` and is **not** part of this surface. From Python, write standalone dashboards; use the CLI when you want a collection.

## Worked example

```python
from pathlib import Path

from citemesh import HybridGraphBuilder
from citemesh.visualization import GraphExporter, compute_layout

builder = HybridGraphBuilder(
    max_papers=25,
    max_semantic=10,
    device="mps",  # "auto" (default), "cuda", "mps", or "cpu"
    truncate_dim=512,
)

graph, seed_id = builder.build_graph("arxiv:1706.03762")
print(
    f"{graph.number_of_nodes()} papers, {graph.number_of_edges()} links, seed={seed_id}"
)

# Compute the layout once so both exports share identical geometry.
layout = compute_layout(graph, layout_seed=1234)

exporter = GraphExporter(graph, seed_id, theme_name="dark", layout=layout)

out_dir = Path("out/api-demo")
out_dir.mkdir(parents=True, exist_ok=True)
exporter.to_json(out_dir / "hybrid.json")
exporter.to_dashboard_html(out_dir / "hybrid.dashboard.html")
```

`RecommendationGraphBuilder` is the lighter substitute — it never loads the embedding model, so it needs no `embeddings` extra and no GPU:

```python
from citemesh import RecommendationGraphBuilder

builder = RecommendationGraphBuilder(max_papers=40, similarity_threshold=0.2)
graph, seed_id = builder.build_graph("arxiv:1706.03762")
```

A static PNG instead of the HTML exports:

```python
from citemesh.visualization import visualize_graph

visualize_graph(
    graph, seed_id, out_dir / "hybrid.png", dpi=150, theme_name="dark", layout=layout
)
```

## Things to know before you build on this

- **Network and credentials.** Every builder calls Semantic Scholar. Set `S2_API_KEY` in the environment, or persist it with `citemesh config set api.s2_api_key`, before running any of this at volume — the anonymous pool rate-limits aggressively and the retry policy has no overall deadline.
- **Caches are shared with the CLI.** A builder constructed in Python reads and writes the same cache root, embedding namespaces, and `config.toml` as `citemesh build`. That is usually what you want; be aware of it if you are benchmarking.
- **Constructor keywords mirror CLI flags.** `HybridGraphBuilder(max_semantic=...)` is `--max-semantic`, `truncate_dim=` is `--truncate-dim`, and so on. The CLI's cross-flag validation does **not** run here, so Python callers can construct combinations the CLI would reject. The builders validate their own invariants (`max_semantic` must satisfy `0 <= max_semantic <= max_papers - 1`, `top_k >= 1`, `max_papers >= 1`) and raise `ValueError` otherwise.
- **Failures are exceptions, not empty graphs.** A total candidate-source outage raises `CandidateAcquisitionError`; semantic inference that cannot produce a complete ranking space raises `EmbeddingInferenceError`; an unavailable explicit device raises `ValueError` at construction. Hybrid never silently downgrades to citation-only output.
- **Everything is deterministic given a seed.** Ordering, tie-breaking, and layout are seeded; pass `layout_seed` for reproducible geometry across runs.

Flag semantics and defaults for every constructor keyword are documented once in the [CLI guide](cli.md#flag-reference); the mechanism behind each stage is in [How CiteMesh builds a graph](how-it-works.md).
