# Strategy Guide

Use this guide to choose between `recommendation`, `citation`, `embedding`, and `hybrid`.

Related docs:

- CLI flags and defaults: [CLI Usage](cli.md)
- Cache behavior: [Caching & Data](caching.md)
- Embedding runtime defaults and precision policy: [Embedding Runtime](../reference/embedding-runtime.md)
- Default-parameter sweep rationale: [Defaults Tuning Study](../reference/defaults-tuning-study.md)
- Environment variables: [Environment Variables](../reference/environment.md)

## At a Glance

| Strategy | Primary signal | Data source | Best for |
| --- | --- | --- | --- |
| `recommendation` | Semantic Scholar recommendations | Semantic Scholar API | Fast topical exploration |
| `citation` | References + citations + bibliographic coupling | Semantic Scholar API | Relationship-grounded neighborhoods |
| `embedding` | Dense vector similarity | S2 candidate pool + embedding model (default) or HuggingFace corpus | Conceptual similarity beyond citations |
| `hybrid` | Citation graph + semantic enrichment | Semantic Scholar API + embedding model | Balanced recall with grounded edges |

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

Two semantic sources, selected with `--semantic-source`:

- `candidates` (default): fetches the seed's Semantic Scholar neighbors (references, citations, and recommendations, up to `--candidate-pool-size`), embeds only those abstracts locally, and ranks them by cosine similarity to the seed. Fast (seconds after the one-time model download), covers all venues S2 indexes, and needs no corpus download. Candidate vectors persist incrementally in a candidate-scoped cache namespace.
- `arxiv-corpus` (opt-in): hydrates a local arXiv abstract corpus from HuggingFace and searches it. This can surface papers with no citation path to the seed, but cold hydration encodes the full corpus cap and is only practical on strong accelerators for large caps. A capped hydration selects the `--corpus-size` most recently submitted papers, ranked by the submission date encoded in each arXiv ID (snapshot row order does not track submission time); use `--all-corpus` (resumable) for full coverage. Requires the `datasets` dependency.

Providing corpus flags (`--dataset-split`, `--corpus-size`, `--all-corpus`, `--streaming`) without `--semantic-source` implies `arxiv-corpus` for backwards compatibility.

- Strengths: captures semantic similarity even when citations are missing.
- Typical use: semantic exploration and discovery beyond citation graphs.

Embedding cache behavior, hydration, and precision controls are defined in [Caching & Data](caching.md). Embedding model defaults/fallbacks and compile policy are defined in [Embedding Runtime](../reference/embedding-runtime.md).

## Hybrid Strategy

- Data source: citation collection plus semantic enrichment. In the default `candidates` mode the semantic branch pulls S2 recommendations and embeds the merged candidate pool locally; `arxiv-corpus` mode searches the hydrated corpus instead.
- Strengths: combines grounded citation edges with semantic reranking of the whole candidate pool.
- Limitations: inherits dependency and cache requirements from the embedding path.
- Default behavior builds citation and semantic candidate pools, then reranks by seed relevance with a boost for overlap papers discovered by both branches.
- `max_semantic` limits semantic-only additions, not overlap papers that also appear in citation candidates.
- Hybrid defaults are tuned for the seed-paper discovery workflow (recent follow-up + foundational prior work), with current depth targets documented in the defaults study.
- Typical use: balanced graphs when you want citation structure plus semantic recall.

## Choosing a Strategy

- Start with `recommendation` for fast topical graphs.
- Use `citation` when explicit reference structure is most important.
- Use `embedding` to surface conceptually similar papers without citation dependency.
- Use `hybrid` when you want citation grounding plus semantic expansion.
