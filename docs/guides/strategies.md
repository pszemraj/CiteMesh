# Strategy guide

Choose based on the evidence you want to explore and the cost of collecting it.

| Strategy | Candidate source | Local model | Best for |
| --- | --- | --- | --- |
| `recommendation` | Semantic Scholar recommendations | no | quick topical exploration from a known paper |
| `citation` | references and citations | no | a bibliographic neighborhood |
| `embedding` | S2 candidates or an arXiv corpus | yes | conceptual similarity and free-text queries |
| `hybrid` | citation-derived and semantic candidates | when enrichment is enabled | discovery combining citation evidence and semantic relevance |

## Recommendation and citation

Both use TF-IDF topical similarity with temporal and bibliographic evidence. Recommendation depends on Semantic Scholar's coverage; papers with few recommendations can produce thin graphs. Citation candidates have a direct bibliographic relationship with the seed, though their topics may diverge. Reference hydration adds coupling evidence to both; disabling it trades that evidence for fewer requests.

See [candidate acquisition](how-it-works.md#2-candidate-acquisition) and [edge scoring](how-it-works.md#6-edge-scoring) for budgets, source failures, and pruning.

## Embedding

Use `--semantic-source candidates` for a small S2-derived pool, or `--semantic-source arxiv-corpus` to search a downloaded abstract corpus. Corpus mode can reach work outside the seed's citation neighborhood, with substantially more initial encoding and storage. [Caching and data](caching.md#corpus-hydration-and-resume) describes that cost and subsequent reuse.

Embedding is the only strategy that accepts [free-text seeds](how-it-works.md#1-seed-resolution). Its [runtime requirements](../reference/embedding-runtime.md) apply to both source modes.

## Hybrid

Hybrid reranks the union of citation-derived and semantic candidates. Citation evidence can admit an edge that would fail a purely semantic gate; [ranking](how-it-works.md#5-ranking-and-selection) and [edge scoring](how-it-works.md#6-edge-scoring) define the bonuses, limits, and thresholds.

A heavily cited seed can fill the graph before semantic-only papers qualify. Lower the citation budget or use corpus sourcing when semantic recall matters. Disabling semantic enrichment uses citation scoring while retaining hybrid's edge cap. When enrichment is enabled, inference failures stop the build; unavailable citation endpoints can still leave recommendations or the local corpus usable.

The [hybrid tuning study](../reference/defaults-tuning-study.md#hybrid-defaults-february-2026) explains the discovery tradeoff behind its defaults. For overrides, see the [CLI controls](cli.md#hybrid-strategy).
