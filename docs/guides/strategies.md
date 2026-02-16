# Strategy Guide

The `citemesh` CLI can build graphs with four strategies: `recommendation`, `citation`, `embedding`, and `hybrid`. This guide explains how they differ and when to use each.

Related docs:

- CLI flags and defaults: [CLI Usage](cli.md)
- Cache behavior: [Caching & Data](caching.md)
- Environment variables: [Environment Variables](../reference/environment.md)
- Embedding runtime defaults and precision policy: [Embedding Runtime](../reference/embedding-runtime.md)
- Docs index: [Documentation](../README.md)

## At a Glance

| Strategy | Primary signal | Data source | Best for |
| --- | --- | --- | --- |
| `recommendation` | Semantic Scholar recommendations | Semantic Scholar API | Fast topical exploration |
| `citation` | References + citations + bibliographic coupling | Semantic Scholar API | Relationship-grounded neighborhoods |
| `embedding` | Dense vector similarity | HuggingFace corpus + embedding model | Conceptual similarity beyond citations |
| `hybrid` | Citation graph + semantic enrichment | Semantic Scholar API + HuggingFace corpus | Balanced recall with grounded edges |

## Recommendation Strategy

- Data source: Semantic Scholar recommendation endpoints.
- Strengths: fast, low setup, good for topical exploration.
- Limitations: depends on Semantic Scholar availability and coverage.
- Typical use: quick graphing from a known paper ID or query.

## Citation Strategy

- Data source: Semantic Scholar references and citations.
- Strengths: edges follow explicit citation relationships.
- Limitations: coverage varies by paper and field; requires reference lists for full bibliographic coupling.
- Typical use: grounded citation neighborhoods and reference-aware similarity.

## Embedding Strategy

- Data source: HuggingFace ArXiv-like corpus + sentence-transformer model.
- Strengths: captures semantic similarity even when citations are missing.
- Limitations: first run hydrates a cache for the selected corpus spec; requires optional embedding dependencies.
- Typical use: semantic exploration and discovery beyond citation graphs.

Embedding cache behavior, hydration, and precision controls are defined in [Caching & Data](caching.md).
Embedding model defaults/fallbacks and compile policy are defined in [Embedding Runtime](../reference/embedding-runtime.md).

## Hybrid Strategy

- Data source: citation collection plus semantic enrichment.
- Strengths: combines grounded citation edges with semantic reranking of the whole candidate pool.
- Limitations: inherits dependency and cache requirements from the embedding path.
- Default behavior builds citation and semantic candidate pools, then reranks by seed relevance with a boost for overlap papers discovered by both branches.
- `max_semantic` limits semantic-only additions, not overlap papers that also appear in citation candidates.
- Typical use: balanced graphs when you want citation structure plus semantic recall.

## Choosing a Strategy

- Start with `recommendation` for fast topical graphs.
- Use `citation` when explicit reference structure is most important.
- Use `embedding` to surface conceptually similar papers without citation dependency.
- Use `hybrid` when you want citation grounding plus semantic expansion.

## Related Docs

- [CLI Usage](cli.md)
- [Caching & Data](caching.md)
- [Environment Variables](../reference/environment.md)
- [Architecture](../internals/architecture.md)
