# Output Artifacts

CiteMesh writes graph outputs by format plus a run-configuration sidecar.

Related docs:

- CLI flags and output-path behavior: [CLI Usage](../guides/cli.md)
- Strategy behavior and score semantics: [Strategy Guide](../guides/strategies.md)
- Embedding runtime metadata terms: [Embedding Runtime](embedding-runtime.md)

## Artifact Set

When `--export all` is used, CiteMesh writes:

- `<strategy>.png` (static Matplotlib render)
- `<strategy>.html` (Pyvis interactive network)
- `<strategy>.plotly.html` (Plotly interactive graph)
- `dashboard.html` (shared tri-pane research dashboard shell when dashboard export is part of a normal collection flow)
- `dashboard.manifest.json` (shared collection index for dashboard result payloads)
- `<strategy>.json` (enriched graph data payload)
- `<strategy>.csv` (flat paper table for pandas/spreadsheets)
- `<strategy>.bib` (combined BibTeX entries for all papers)
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
  - non-dashboard artifacts are written under `out/<slug>-<hash>/` as `<strategy>.<ext>`
  - when `dashboard` export is selected, the shared shell is written to `out/dashboard.html`, the collection index to `out/dashboard.manifest.json`, and the shared shell is refreshed with the current saved-result bundle
- single-export run (`--export <one-format>`) with explicit `--output`:
  - `--export dashboard -o report.dashboard.html` keeps the legacy standalone one-file behavior and writes exactly `report.dashboard.html`
  - if `--output` ends with the target format suffix, it is used as-is
  - if `--output` ends with a different known export suffix, that suffix is replaced
  - if `--output` has no known export suffix, the target suffix is appended
- multi-export run (`--export all` or multiple formats) with explicit `--output`:
  - if `--output` ends with a known export suffix (for example `out.png`), that suffix
    is stripped and the remainder is treated as directory base
  - if `--output` has no known suffix, it is treated directly as directory base
  - each format is written as `<directory-base>/<strategy>.<ext>`
  - if `dashboard` is among the selected formats, the shared shell is written as `<directory-base>/dashboard.html`, the collection index as `<directory-base>/dashboard.manifest.json`, the shell embeds the currently available saved-result payloads from that collection, and the run-specific data/config artifacts are written under `<directory-base>/<slug>-<hash>/`
  - re-running the same `seed_id` with the same `strategy` refreshes that collection slot instead of creating a second entry, because the JSON/config artifact path for that seed is stable

Examples:

- `citemesh build "<paper-id>" --strategy hybrid --export all -o out.png`
  writes `out/dashboard.html`, `out/dashboard.manifest.json`,
  `out/<slug>-<hash>/hybrid.png`, `out/<slug>-<hash>/hybrid.html`,
  `out/<slug>-<hash>/hybrid.plotly.html`, `out/<slug>-<hash>/hybrid.json`,
  `out/<slug>-<hash>/hybrid.graphml`, `out/<slug>-<hash>/hybrid.config.json`
- `citemesh build "<paper-id>" --strategy citation --export json -o report.graphml`
  writes `report.json`
- `citemesh build "<paper-id>" --strategy recommendation --export dashboard -o report.dashboard.html`
  writes the standalone dashboard file `report.dashboard.html`

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
- `meta` (`strategy`, `year_range`)
- `summary` (`nodes`, `edges`)
- `nodes` — enriched per-paper objects (see below)
- `edges` (`source`, `target`, `weight`, plus readable source/target title/label fields)

`source`/`target` are canonical node IDs for unambiguous graph processing.
The extra `*_title` and `*_label` fields are provided for readable inspection.

Each node includes:

- core fields: `id`, `title`, `year`, `authors`, `abstract`, `citation_count`, `venue`, `arxiv_id`, `doi`, `categories`, `is_seed`
- analysis fields: `provenance` (seed/citation/semantic/both), `provenance_base`, `seed_relation` (cites_seed/referenced_by_seed/semantic_only/overlap/seed), `seed_relevance` (personalized PageRank score)
- external: `links` (arXiv abs/pdf URLs, DOI URL, Semantic Scholar URL)
- `bibtex` (deterministic BibTeX entry)

The JSON and dashboard formats share the same enriched node schema, and the JSON
payload also carries dashboard render metadata (`dashboard.meta.plotly_*`) so
current-version CiteMesh JSON files can be loaded back into the dashboard via
the **Load Results** button without losing graph geometry.

### CSV (`<strategy>.csv`)

Flat table with one row per paper. Columns: `id`, `title`, `year`, `authors`
(semicolon-separated), `citation_count`, `venue`, `arxiv_id`, `doi`,
`categories` (semicolon-separated), `is_seed`, `provenance`, `seed_relation`,
`seed_relevance`, `arxiv_url`, `doi_url`, `semantic_scholar_url`, `abstract`.

### BibTeX (`<strategy>.bib`)

Combined BibTeX entries for all papers in the graph, one `@article` per paper.
Ready for direct import into reference managers or LaTeX projects.

### Dashboard HTML

The dashboard shell (`dashboard.html` in collection mode, or
`<name>.dashboard.html` for explicit standalone output) is a tri-pane research
interface with embedded Plotly graph, paper list, and detail panel.

**Toolbar data actions:**

- **Export JSON** — downloads the embedded enriched payload as a standalone `.json` file
- **Export CSV** — generates a CSV table client-side from the current dataset
- **All BibTeX** — downloads all papers' BibTeX entries as a single `.bib` file
- **Saved Results selector** — switches between JSON payload slots already tracked in the collection shell, with no extra file picking
- **Load Results** — file picker that accepts current-version CiteMesh JSON files plus dashboard HTML exports. Current-version JSON keeps stored graph geometry. Legacy dashboard HTML can recover embedded geometry from the exported Plotly figure. Legacy JSON without stored dashboard geometry still loads, but CiteMesh reconstructs a deterministic fallback layout and shows a warning.

In collection mode, this means you can keep one `dashboard.html` open and move
between saved result slots from the built-in selector, or load any other
compatible JSON payload manually, rather than opening a separate dashboard HTML
file for each paper. Re-running the same seed with the same strategy refreshes
that slot instead of adding another selector entry.

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
[Embedding Runtime](embedding-runtime.md).

## Determinism Notes

- `json`: deterministic key order + indentation.
- `csv`: deterministic column order and row order (same as JSON node order).
- `bibtex`: deterministic entry order (same as JSON node order).
- `graphml`: deterministic ordering on supported NetworkX versions.
- `png`: deterministic for same input graph and `--seed`.
- `plotly`: deterministic for same input graph and `--seed` when Plotly supports `write_html(div_id=...)`.
- `html` (Pyvis): deterministic serialized structure, but browser physics are runtime-driven.
