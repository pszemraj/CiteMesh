# Strategy guide

Choose based on the evidence you want to explore and the cost of collecting it.

| Strategy | Candidate source | Local model | Best for |
| --- | --- | --- | --- |
| `recommendation` | Semantic Scholar recommendations | no | quick topical exploration from a known paper |
| `citation` | references and citations | no | a bibliographic neighborhood |
| `embedding` | S2 candidates or an arXiv corpus | yes | conceptual similarity and free-text queries |
| `hybrid` | the focal paper's references and citers, plus S2 recommendations or corpus neighbors | yes | discovery combining citation evidence and semantic relevance |

## Recommendation and citation

Sparse recommendation coverage can produce thin graphs. Citation graphs cover direct links whose topics may diverge. [Reference hydration](caching.md#paper-metadata-and-reference-ids) adds coupling evidence to both.

See [candidate acquisition](how-it-works.md#2-candidate-acquisition) and [edge scoring](how-it-works.md#6-edge-scoring) for budgets, source failures, and pruning.

## Embedding

Use `--semantic-source candidates` for a small S2-derived pool, or `--semantic-source arxiv-corpus` to search a downloaded abstract corpus. Corpus mode can reach work outside the focal paper's citation neighborhood, with substantially more initial encoding and storage. [Caching and data](caching.md#corpus-hydration-and-resume) describes that cost and subsequent reuse.

Embedding is the only strategy that accepts [free-text inputs](how-it-works.md#1-focal-paper-resolution). Its [runtime requirements](../reference/embedding-runtime.md) apply to both source modes.

## Hybrid

Hybrid always needs the embeddings extra and the embedding model. Choose which candidates to add to the focal paper's references and citers with `--semantic-source`.

- `candidates` (default) adds Semantic Scholar recommendations.
- `arxiv-corpus` adds neighbors from the local corpus.
- `both` combines recommendations and corpus neighbors, with the same index setup as `arxiv-corpus`.

For an installed index:

```bash
citemesh build arxiv:2410.05192 --strategy hybrid --semantic-source both \
  --cache-namespace NAMESPACE --max-papers 45
```

With `--cache-namespace`, hybrid keeps `both` only when `--semantic-source both` was passed or `both` is the configured default; otherwise the installed index selects `arxiv-corpus`, so no provider recommendations are requested.

A failed corpus load stops hybrid. In `candidates` mode, an outage across every requested source stops the build before model loading. See [partial outage behavior](cli.md#troubleshooting).

See [candidate acquisition](how-it-works.md#2-candidate-acquisition), [ranking](how-it-works.md#5-ranking-and-selection), and [edge scoring](how-it-works.md#6-edge-scoring) for what hybrid does with the pool. The [hybrid discovery evaluation](../internals/hybrid-discovery-status.md) reports which papers the resulting rankings surfaced. For overrides, see the [CLI controls](cli.md#option-reference).
