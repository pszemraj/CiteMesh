# Strategy guide

Choose based on the evidence you want to explore and the cost of collecting it.

| Strategy | Candidate source | Local model | Best for |
| --- | --- | --- | --- |
| `recommendation` | Semantic Scholar recommendations | no | quick topical exploration from a known paper |
| `citation` | references and citations | no | a bibliographic neighborhood |
| `embedding` | S2 candidates or an arXiv corpus | yes | conceptual similarity and free-text queries |
| `hybrid` | the focal paper's references and citers, plus S2 recommendations or corpus neighbors | yes | discovery combining citation evidence and semantic relevance |

## Recommendation and citation

Recommendation starts from Semantic Scholar's recommendations, so sparse provider coverage can produce thin graphs. Citation starts from the focal paper's references and citations; their topics may diverge. Reference hydration adds coupling evidence to both and costs extra requests.

See [candidate acquisition](how-it-works.md#2-candidate-acquisition) and [edge scoring](how-it-works.md#6-edge-scoring) for budgets, source failures, and pruning.

## Embedding

Use `--semantic-source candidates` for a small S2-derived pool, or `--semantic-source arxiv-corpus` to search a downloaded abstract corpus. Corpus mode can reach work outside the focal paper's citation neighborhood, with substantially more initial encoding and storage. [Caching and data](caching.md#corpus-hydration-and-resume) describes that cost and subsequent reuse.

Embedding is the only strategy that accepts [free-text inputs](how-it-works.md#1-focal-paper-resolution). Its [runtime requirements](../reference/embedding-runtime.md) apply to both source modes.

## Hybrid

Hybrid pools every reference and every citer that Semantic Scholar returns for the focal paper (up to `--max-references` and `--max-citations`, 1000 each by default), adds candidates from the source chosen by `--semantic-source`, and ranks the whole pool by reciprocal rank fusion. It always needs the embeddings extra and the embedding model; `--strategy citation` is the citation-only option. See [candidate acquisition](how-it-works.md#2-candidate-acquisition), [ranking](how-it-works.md#5-ranking-and-selection), and [edge scoring](how-it-works.md#6-edge-scoring) for the details.

- `candidates` (default) adds 200 of Semantic Scholar's `all-cs` recommendations.
- `arxiv-corpus` adds the 180 nearest corpus rows to the focal paper in each of two embedding roles, the focal paper as a retrieval query and as a retrieval document. References and citers still come from Semantic Scholar.
- `both` adds both sets. It requires the same embeddings and corpus setup as `arxiv-corpus` and makes the additional recommendation request.

For an installed index:

```bash
citemesh build arxiv:2410.05192 --strategy hybrid --semantic-source both \
  --cache-namespace NAMESPACE --max-papers 45
```

With `--cache-namespace`, hybrid keeps `both` only when `--semantic-source both` was passed or `both` is the configured default; otherwise the installed index selects `arxiv-corpus`, so no provider recommendations are requested.

What hybrid does with the pool:

- **Ranks it.** Every candidate gets a retrieval-document vector, and up to four ranked lists are fused by reciprocal rank fusion: similarity to the focal paper, similarity to the focal paper's bibliography, the directly linked candidates, and the provider's recommendations. The best `max_papers - 1` candidates join the focal paper. There is no year term and no citation-count term: every candidate competes on the same fused score, whatever its source. The lists and formula are in [ranking and selection](how-it-works.md#5-ranking-and-selection); where the vectors are stored is in [Discovery checks](caching.md#discovery-checks).
- **Merges duplicates first.** Publication and preprint versions of one work are reconciled across the whole pool before ranking, so a pair recognized as one work never takes two slots ([candidate acquisition](how-it-works.md#2-candidate-acquisition)).
- **Gives the focal paper both kinds of neighbor.** Each paper keeps at most 5 edges, and the focal paper's slots alternate between its own references and citers and its closest other neighbors ([edge scoring](how-it-works.md#6-edge-scoring)).
- **Fetches reference lists last.** They are fetched after selection, for the selected citation-sourced papers only, and feed bibliographic coupling in edge scoring.
- **Exports its evidence.** Each related paper's ranking evidence (its rank, fused score, similarity to the focal paper and its bibliography, direct-citation flag, recommendation rank, and rank in each fused list) is exported in [graph JSON](../reference/output-artifacts.md#graph-json-strategyjson) and shown in the dashboard; it is separate from `seed_relevance`, which is centrality over the final edges.
- **Survives partial outages.** An unavailable source is recorded in source status while the others continue; a failed corpus load, or an outage of every source, still stops the build, in `candidates` mode before the model loads.

`--candidate-pool-size` applies only to the embedding strategy's `candidates` mode. The [hybrid discovery status note](../internals/hybrid-discovery-status.md) summarizes how the current ranking was evaluated. For overrides, see the [CLI controls](cli.md#option-reference).
