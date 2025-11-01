# Hybrid Strategy Notes

## Default Collection Limits

By default the hybrid builder collects a small, fast slice of the graph:

- up to 20 references from the seed paper
- up to 9 citing papers
- up to 10 semantic neighbors sourced from the embedding corpus

These limits are chosen so the command finishes quickly, but the knobs are
exposed on the CLI. Use flags such as `--max-references`, `--max-citations`, and
`--max-semantic` to raise the limits when you want denser graphs. All exports use
this same configuration, so saved bundles will reflect the options you chose.

## How Embeddings Contribute

Hybrid mode is not just a citation graph. The pipeline is:

1. Fetch references/citations from Semantic Scholar for the seed paper.
2. Query the HuggingFace corpus to add semantic neighbors (up to
   `--max-semantic`, cache-backed for speed).
3. Build the graph with adaptive weighting:
   - embedding similarity dominates for pairs of semantic neighbors,
   - bibliographic coupling takes over for citation-only pairs,
   - mixed pairs blend the signals.
4. Semantic neighbors also get citation counts/metadata from Semantic Scholar
   where available, so "Unknown" author clusters disappear once enrichment runs.

This means the hybrid outputs differ from both pure strategies: you see the
citation structure anchored by the seed plus fresh semantic additions that would
otherwise be absent.

## Output Bundles

By default each run saves into `out/citemesh-<slug>/`. Inside that directory you
get every requested export (PNG, HTML, Plotly, JSON, GraphML) and a
`parameters.json` file that records the CLI options used (limits, model name,
seed, theme, etc.). Bump the collection knobs and re-run to compare—you'll have
one bundle per experiment.
