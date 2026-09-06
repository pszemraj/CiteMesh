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
| `citation` | References + citations + bibliographic coupling | Semantic Scholar API | Citation-derived neighborhoods |
| `embedding` | Dense vector similarity | S2 candidate pool + embedding model (default) or HuggingFace corpus | Conceptual similarity beyond citations |
| `hybrid` | Citation-derived candidates + semantic enrichment | Semantic Scholar API + embedding model | Balanced grounded and semantic recall |

## Recommendation Strategy

- Data source: Semantic Scholar recommendation endpoints.
- Strengths: fast, low setup, good for topical exploration.
- Limitations: depends on Semantic Scholar availability and coverage.
- Typical use: quick graphing from a known paper ID.
- Edges require positive topical similarity or shared-reference evidence and meet
  the configured similarity threshold. At most three edges touch each paper;
  the seed's strongest eligible edges are reserved first, then remaining edges
  are selected by strength within each paper's cap. The seed retains up to three
  existing neighbors; capping never invents unsupported edges.

## Citation Strategy

- Data source: Semantic Scholar references and citations.
- Strengths: candidates come from explicit reference and citation relationships;
  graph edges combine topical, temporal, citation-impact, and bibliographic
  evidence.
- Limitations: coverage varies by paper and field; requires reference lists for full bibliographic coupling.
- Typical use: citation-derived neighborhoods and reference-aware similarity.
- As with recommendation graphs, dates and citation popularity alone cannot
  create an edge, and each paper has at most three edges. The default 40-paper
  graph therefore has at most 60 edges. The seed's strongest eligible edges are
  reserved before selecting the remaining edges.

Both strategies warn when reference hydration exhausts its retries, then skip
further reference hydration for that collection. Existing reference lists remain
available for scoring; a subsequent collection tries the source again.

## Embedding Strategy

Two semantic sources, selected with `--semantic-source`:

- `candidates` (default): for a known-paper seed, fetches Semantic Scholar
  references, citations, and recommendations within `--candidate-pool-size`, then
  embeds title/abstract text locally and ranks it against the seed. A free-text
  embedding seed instead starts with up to 20 S2 keyword results and expands
  recommendations from the top anchor. Candidate vectors persist incrementally;
  no local corpus is downloaded.
  Known-paper budgets are split approximately 1:2:1 across references, citations,
  and recommendations, with references and recommendations each capped at 100.
  Pools of at least three include all three sources; a one-paper budget requests
  recommendations, and a two-paper budget also requests one reference. Candidate
  metadata already includes citation counts, so this mode does not refetch them.
- `arxiv-corpus` (opt-in): hydrates and searches a local arXiv abstract corpus from
  HuggingFace. This can surface papers with no citation path to the seed but needs
  the `datasets` dependency and substantially more cold-cache work. Selection,
  resumption, and storage behavior are described in [Caching & Data](caching.md).
  Citation-count enrichment is optional: a rejected Semantic Scholar batch warns
  and retains the selected papers with their existing counts. Invalid individual
  batch rows are skipped so valid rows can still enrich their selected papers.

- Strengths: captures semantic similarity even when citations are missing.
- Edge selection reserves the seed's strongest eligible neighbors before the
  remaining edges, while preserving the per-paper `top_k` cap.
- Typical use: semantic exploration and discovery beyond citation graphs.

Embedding cache behavior, hydration, and precision controls are defined in [Caching & Data](caching.md). Embedding model defaults/fallbacks and compile policy are defined in [Embedding Runtime](../reference/embedding-runtime.md).

## Hybrid Strategy

- Data source: citation collection plus semantic enrichment. In the default `candidates` mode the semantic branch pulls S2 recommendations and embeds the merged candidate pool locally; `arxiv-corpus` mode searches the hydrated corpus instead.
- Strengths: combines citation-derived evidence with semantic reranking of the
  whole candidate pool.
- Limitations: inherits dependency and cache requirements from the embedding path.
- Default behavior builds citation and semantic candidate pools, then reranks by seed relevance with a boost for overlap papers discovered by both branches.
- `max_semantic` limits semantic-only additions, not overlap papers that also appear in citation candidates.
- Semantic enrichment failures stop the build; hybrid does not silently downgrade
  to citation-only output.
- If citation/reference endpoints are unavailable after the seed resolves,
  semantic enrichment can still proceed from recommendations or the local arXiv
  corpus. Source availability is recorded in the graph metadata.
- Hybrid defaults are tuned for the seed-paper discovery workflow (recent
  follow-up + foundational prior work); the evaluation rationale is documented in
  the defaults study.
- Typical use: balanced graphs when you want citation evidence plus semantic recall.
- Edge selection reserves the seed's strongest eligible neighbors before the
  remaining edges, while preserving the configured per-paper degree cap.

## Choosing a Strategy

All strategies warn when no selected paper pair meets the edge criteria and the
resulting graph contains no edges.

- Start with `recommendation` for fast topical graphs.
- Use `citation` when a citation-derived neighborhood is most important.
- Use `embedding` to surface conceptually similar papers without citation dependency.
- Use `hybrid` when you want citation grounding plus semantic expansion.
