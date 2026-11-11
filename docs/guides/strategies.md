# Strategy guide

Choose based on the evidence you want to explore and the cost of collecting it.

| Strategy | Candidate source | Local model | Best for |
| --- | --- | --- | --- |
| `recommendation` | Semantic Scholar recommendations | no | quick topical exploration from a known paper |
| `citation` | references and citations | no | a bibliographic neighborhood |
| `embedding` | S2 candidates or an arXiv corpus | yes | conceptual similarity and free-text queries |
| `hybrid` | seed references and citers, plus S2 recommendations or corpus neighbors | yes | discovery combining citation evidence and semantic relevance |

## Recommendation and citation

Recommendation starts from Semantic Scholar's recommendations, so sparse provider coverage can produce thin graphs. Citation starts from the seed's references and citations; their topics may diverge. Reference hydration adds coupling evidence to both and costs extra requests.

See [candidate acquisition](how-it-works.md#2-candidate-acquisition) and [edge scoring](how-it-works.md#6-edge-scoring) for budgets, source failures, and pruning.

## Embedding

Use `--semantic-source candidates` for a small S2-derived pool, or `--semantic-source arxiv-corpus` to search a downloaded abstract corpus. Corpus mode can reach work outside the seed's citation neighborhood, with substantially more initial encoding and storage. [Caching and data](caching.md#corpus-hydration-and-resume) describes that cost and subsequent reuse.

Embedding is the only strategy that accepts [free-text seeds](how-it-works.md#1-seed-resolution). Its [runtime requirements](../reference/embedding-runtime.md) apply to both source modes.

## Hybrid

Hybrid pools every reference and every citer that Semantic Scholar returns for the seed (up to `--max-references` and `--max-citations`, 1000 each by default), adds candidates from the source chosen by `--semantic-source`, and ranks the whole pool by reciprocal rank fusion. It always needs the embeddings extra and the embedding model; `--strategy citation` is the citation-only option. See [candidate acquisition](how-it-works.md#2-candidate-acquisition), [ranking](how-it-works.md#5-ranking-and-selection), and [edge scoring](how-it-works.md#6-edge-scoring) for the details.

- `candidates` (default) adds 200 of Semantic Scholar's `all-cs` recommendations.
- `arxiv-corpus` adds the 180 nearest corpus rows to the seed in each of two embedding roles, the seed as a retrieval query and as a retrieval document. References and citers still come from Semantic Scholar.
- `both` adds both sets. It requires the same embeddings and corpus setup as `arxiv-corpus` and makes the additional recommendation request.

For an installed index:

```bash
citemesh build arxiv:2410.05192 --strategy hybrid --semantic-source both \
  --cache-namespace NAMESPACE --max-papers 45
```

Every candidate gets a retrieval-document vector: its stored corpus vector when it is a corpus row, otherwise an in-memory encoding of its title and abstract. Publication and preprint versions of one work are reconciled across the whole pool before ranking, so a duplicate never takes two slots. Up to four ranked lists are then fused, each candidate scoring the sum of `1 / (60 + rank)` over the lists that contain it:

- similarity to the seed, from the document-role cosine
- similarity to the seed's bibliography: the mean of the candidate's five highest cosines to the seed's references, excluding itself, absent for a candidate when fewer than five other references exist
- the directly linked candidates (cited by the seed or citing it), ordered by similarity to the seed
- the recommended candidates in provider order, only in modes that fetch recommendations

The best `max_papers - 1` candidates join the seed. There is no year term, no citation-count term, no source quota and no cap on semantic-only papers: every candidate competes on the same fused score, whatever its source. Reference lists are fetched after selection, for the selected citation-sourced papers only, and feed bibliographic coupling in edge scoring, which is unchanged.

`--max-semantic` and the `max_semantic` configuration key no longer exist, and `--candidate-pool-size` applies only to the embedding strategy's `candidates` mode. Each related paper's ranking evidence (its rank, fused score, seed and bibliography similarity, direct-citation flag, recommendation rank, and rank in each fused list) is exported in [graph JSON](../reference/output-artifacts.md#graph-json-strategyjson) and shown in the dashboard; it is separate from `seed_relevance`, which is centrality over the final edges. An unavailable source is recorded in source status while the others continue; a failed corpus load, or an outage of every source, still stops the build, in `candidates` mode before the model loads.

The [hybrid tuning study](../reference/defaults-tuning-study.md#hybrid-defaults-february-2026) records the earlier budget sweep and is historical. The [hybrid discovery status note](../internals/hybrid-discovery-status.md) summarizes how the current ranking was evaluated. For overrides, see the [CLI controls](cli.md#option-reference).
