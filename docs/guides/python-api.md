# Python API

Import the builders and exporters directly when a notebook or script beats shelling out to the CLI.

> [!IMPORTANT]
> **Unstable, pre-1.0.** The CLI is the supported interface; signatures, keyword names, and payload shapes here can change in any release. Only the names below are meant for import - anything behind a leading underscore or in an unlisted submodule is internal.

## What you can import

`citemesh.__all__` gives you `Paper` and `Author` from `citemesh.core.models` (graph nodes carry a `Paper` under the `paper` attribute), the `GraphBuilderStrategy` base class to subclass, and the four builders: `RecommendationGraphBuilder`, `CitationGraphBuilder`, `EmbeddingGraphBuilder`, `HybridGraphBuilder`. Builders resolve lazily on first attribute access, so `import citemesh` does not pull in torch. `citemesh.visualization` adds `GraphExporter`, `compute_layout`, `visualize_graph`, `generate_output_path`, `get_theme`, and `THEMES`.

Every builder implements the same two methods:

```python
build_graph(seed_id: str, **kwargs) -> tuple[networkx.Graph, str]  # (graph, resolved_seed_id)
collect_papers(seed_id: str, **kwargs) -> dict[str, Paper]
```

`build_graph` returns the *resolved* seed ID, which may differ from what you passed: S2 can answer an arXiv ID with its own paper ID, and the free-text embedding path returns a synthesized `query:<digest>`. Always use it downstream - that is what `GraphExporter` and the layout helpers expect.

## Exporting

`GraphExporter(graph, seed_id, metadata=None, theme_name="dark", layout=None)` exposes `graph_payload()`, the versioned graph dictionary used by JSON and dashboard consumers, plus `to_json`, `to_csv`, `to_bibtex`, `to_graphml`, `to_plotly_html`, `to_dashboard_html`, and `to_interactive_html`. Each writer takes a `pathlib.Path` and creates missing parent directories before publishing the artifact by atomic replacement. `layout` is optional: omit it and the exporter computes one lazily, or pass one to make several exporters or runs share identical geometry. `to_interactive_html` is Pyvis and ignores `layout`. Format contents are in [Output Artifacts](../reference/output-artifacts.md).

Graphs supplied by callers must follow the [graph-input contract](../reference/output-artifacts.md#graph-input).

Because graph JSON is designed for dashboard import, `graph_payload()` and `to_json()` require `seed_id` to identify a node present in the graph. Empty graphs remain valid for the tabular `to_csv()` and bibliography `to_bibtex()` writers.

The collection-package machinery behind `out/dashboard.html` is **not** part of this surface: from Python, write standalone dashboards; use the CLI when you want a collection.

## Worked example

```python
from pathlib import Path

from citemesh import HybridGraphBuilder
from citemesh.visualization import GraphExporter, compute_layout, visualize_graph

builder = HybridGraphBuilder(
    max_papers=25,
    max_semantic=10,
    device="auto",
    truncate_dim=512,
)
graph, seed_id = builder.build_graph("arxiv:1706.03762")

# Compute the layout once so every export shares identical geometry.
layout = compute_layout(graph, layout_seed=1234)
exporter = GraphExporter(graph, seed_id, theme_name="dark", layout=layout)

out_dir = Path("out/api-demo")
out_dir.mkdir(parents=True, exist_ok=True)
exporter.to_json(out_dir / "hybrid.json")
exporter.to_dashboard_html(out_dir / "hybrid.dashboard.html")
visualize_graph(
    graph, seed_id, out_dir / "hybrid.png", dpi=150, theme_name="dark", layout=layout
)
```

`RecommendationGraphBuilder(max_papers=40, similarity_threshold=0.2)` is the lighter substitute: same two methods, no embedding model, so no `embeddings` extra and no GPU.

## Before you build on this

- **Credentials.** Set `S2_API_KEY` before the first builder creates the shared API client. Saved `api.s2_api_key` values are applied by the CLI, not by direct builder construction. The [API request policy](cli.md#appendix-b-troubleshooting) also applies to Python calls.
- **Configuration and caches.** Builders take constructor arguments and built-in defaults; they do not load `[defaults]` from `config.toml`. They share the CLI's [cache root and namespaces](caching.md), so Python runs can reuse or extend those caches.
- **Constructor keywords mirror CLI flags** (`max_semantic=` is `--max-semantic`, `truncate_dim=` is `--truncate-dim`), but the CLI's cross-flag validation does not run here, so callers can construct combinations the CLI would reject. Builders still check their own invariants - `0 <= max_semantic <= max_papers - 1`, `top_k >= 1`, `max_papers >= 1` - and raise `ValueError`.
- **Failures are exceptions, not empty graphs.** A total candidate-source outage raises `CandidateAcquisitionError`, semantic inference that cannot produce a complete ranking space raises `EmbeddingInferenceError`, and an unavailable explicit device raises `ValueError`.

Ordering and layout are deterministic for the same graph inputs; API responses, corpus updates, and runtime numerics can still change results. Pass `layout_seed` to `compute_layout` or `visualize_graph` to choose the layout seed. The [CLI guide](cli.md#flag-reference) describes corresponding controls; constructor names can differ, such as `enable_torch_compile` for `--torch-compile`.
