# Strategy guide

Choose based on the evidence you want to explore and the cost of collecting it.

| Strategy | Candidate source | Local model | Best for |
| --- | --- | --- | --- |
| `recommendation` | Semantic Scholar recommendations | no | quick topical exploration from a known paper |
| `citation` | references and citations | no | a bibliographic neighborhood |
| `embedding` | S2 candidates or an arXiv corpus | yes | conceptual similarity and free-text queries |
| `hybrid` | citation-derived and semantic candidates | when enrichment is enabled | discovery combining citation evidence and semantic relevance |

## Recommendation and citation

Recommendation starts from Semantic Scholar's recommendations, so sparse provider coverage can produce thin graphs. Citation starts from the seed's references and citations; their topics may diverge. Reference hydration adds coupling evidence to both and costs extra requests.

See [candidate acquisition](how-it-works.md#2-candidate-acquisition) and [edge scoring](how-it-works.md#6-edge-scoring) for budgets, source failures, and pruning.

## Embedding

Use `--semantic-source candidates` for a small S2-derived pool, or `--semantic-source arxiv-corpus` to search a downloaded abstract corpus. Corpus mode can reach work outside the seed's citation neighborhood, with substantially more initial encoding and storage. [Caching and data](caching.md#corpus-hydration-and-resume) describes that cost and subsequent reuse.

Embedding is the only strategy that accepts [free-text seeds](how-it-works.md#1-seed-resolution). Its [runtime requirements](../reference/embedding-runtime.md) apply to both source modes.

## Hybrid

Hybrid combines citation and semantic candidates, with adaptive reranking and edge scoring. See [ranking](how-it-works.md#5-ranking-and-selection) and [edge scoring](how-it-works.md#6-edge-scoring) for how each source affects selection and connections.

Use `--semantic-source both` to add Semantic Scholar's `all-cs` recommendations
to local corpus retrieval. It requires the same embeddings and corpus setup as
`arxiv-corpus` and makes additional S2 requests. For an installed index:

```bash
citemesh build arxiv:2410.05192 --strategy hybrid --semantic-source both \
  --cache-namespace NAMESPACE --max-papers 45
```

The combined mode searches beyond S2's default `recent` recommendation pool.
It keeps the existing relevance ranking, paper budget, and edge criteria.
`--candidate-pool-size` bounds its recommendation fetch; `--max-semantic` caps
the combined semantic-only selections from both sources. Source positions
survive deduplication and are available in [graph JSON](../reference/output-artifacts.md#graph-json-strategyjson).
Unavailable recommendations are recorded in source status while usable corpus
results can continue; a failed corpus load still stops the build.

A heavily cited seed can fill the graph before semantic-only papers qualify. Lower the citation budget or use corpus sourcing when semantic recall matters.

The [hybrid tuning study](../reference/defaults-tuning-study.md#hybrid-defaults-february-2026) explains the discovery tradeoff behind its defaults. For overrides, see the [CLI controls](cli.md#option-reference).
