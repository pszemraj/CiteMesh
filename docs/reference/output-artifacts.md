# Output Artifacts

CiteMesh writes graph outputs by format plus a run-configuration sidecar.

Related docs:

- CLI flags and output-path behavior: [CLI Usage](https://github.com/pszemraj/CiteMesh/blob/main/docs/guides/cli.md)
- Strategy behavior and score semantics: [Strategy Guide](https://github.com/pszemraj/CiteMesh/blob/main/docs/guides/strategies.md)
- Embedding runtime metadata terms: [Embedding Runtime](https://github.com/pszemraj/CiteMesh/blob/main/docs/reference/embedding-runtime.md)

## Artifact Set

When `--export all` is used, CiteMesh writes:

- `<strategy>.png` (static Matplotlib render)
- `<strategy>.html` (Pyvis interactive network)
- `<strategy>.plotly.html` (Plotly interactive graph)
- `<strategy>.dashboard.html` (standalone tri-pane research dashboard)
- `<strategy>.json` (graph data payload)
- `<strategy>.graphml` (exchange format for Gephi/Cytoscape)
- `<strategy>.config.json` (run config + metadata sidecar)

For single-export runs, only the requested format is written.
`*.config.json` is still written as the run sidecar.

## Output Location

Path components:

- `<slug>` is a filesystem-safe version of the seed title
- `<hash>` is the first 8 chars of `sha256(seed_id)`

Canonical path-normalization rules:

- `--output` omitted:
  - artifacts are written under `out/<slug>-<hash>/` as `<strategy>.<ext>`
- single-export run (`--export <one-format>`) with explicit `--output`:
  - if `--output` ends with the target format suffix, it is used as-is
  - if `--output` ends with a different known export suffix, that suffix is replaced
  - if `--output` has no known export suffix, the target suffix is appended
- multi-export run (`--export all` or multiple formats) with explicit `--output`:
  - if `--output` ends with a known export suffix (for example `out.png`), that suffix
    is stripped and the remainder is treated as directory base
  - if `--output` has no known suffix, it is treated directly as directory base
  - each format is written as `<directory-base>/<strategy>.<ext>`

Examples:

- `citemesh build "<paper-id>" --strategy hybrid --export all -o out.png`
  writes `out/hybrid.png`, `out/hybrid.html`, `out/hybrid.plotly.html`,
  `out/hybrid.dashboard.html`, `out/hybrid.json`, `out/hybrid.graphml`,
  `out/hybrid.config.json`
- `citemesh build "<paper-id>" --strategy citation --export json -o report.graphml`
  writes `report.json`

## JSON vs Sidecar

`<strategy>.json` and `<strategy>.config.json` serve different purposes:

- `<strategy>.json`: graph payload (`nodes`, `edges`, basic summary) for downstream graph/data work.
- `<strategy>.config.json`: run contract (CLI parameters, resolved outputs, and metadata) for reproducibility and audit trails.

Sidecar path contract:

- for strategy-named outputs, sidecar is `<strategy>.config.json` in the same directory
- for explicit single-file outputs, sidecar uses the resolved output stem
  (for example `report.json` -> `report.config.json`)

### Graph JSON (`<strategy>.json`)

Top-level fields:

- `seed_id`
- `summary` (`nodes`, `edges`)
- `nodes` (id/title/year/authors/abstract/categories/is_seed/citation_count)
- `nodes` (id/title/year/authors/abstract/venue/arxiv_id/doi/categories/is_seed/citation_count)
- `edges` (`source`, `target`, `weight`, plus readable source/target title/label fields)

`source`/`target` are canonical node IDs for unambiguous graph processing.
The extra `*_title` and `*_label` fields are provided for readable inspection.

### Dashboard HTML Payload Notes

The embedded dashboard payload (inside `<strategy>.dashboard.html`) includes
derived interactive fields used by the static client UI, including:

- `seed_relevance` (seed-centric personalized PageRank score)
- `provenance` / `provenance_base`
- `seed_relation` (when strategy metadata is available)
- `links` (arXiv/DOI/S2 URLs derived from node IDs and external IDs)
- `bibtex` (deterministic inline entry for copy/download actions)

### Sidecar (`<strategy>.config.json`)

Top-level fields:

- `schema_version`
- `build` (resolved build parameters)
- `outputs` (resolved artifact paths)
- `metadata` (run metadata captured during graph build/export)

`build` includes strategy-specific sections:

- `citation` for citation/recommendation/hybrid collection knobs.
- `hybrid` for resolved `max_semantic`.
- `embedding` for embedding/hybrid semantic settings.

`metadata` includes:

- common run metadata (`paper_id`, `seed_id`, `nodes`, `edges`, `theme`, `strategy`)
- strategy score semantics (`score_contract`)
- embedding/hybrid runtime retrieval metadata when available

See embedding metadata term definitions in
[Embedding Runtime](https://github.com/pszemraj/CiteMesh/blob/main/docs/reference/embedding-runtime.md).

## Determinism Notes

- `json`: deterministic key order + indentation.
- `graphml`: deterministic ordering on supported NetworkX versions.
- `png`: deterministic for same input graph and `--seed`.
- `plotly`: deterministic for same input graph and `--seed` when Plotly supports `write_html(div_id=...)`.
- `html` (Pyvis): deterministic serialized structure, but browser physics are runtime-driven.
