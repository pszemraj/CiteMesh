# Hybrid discovery status

This note records what the `hybrid` strategy does after the discovery rework and the evidence behind the design. It addresses the discovery-precision concern raised in issues #14, #16 and #29.

## What hybrid does now

Hybrid pools every seed reference and citer that Semantic Scholar returns (fetch limits `-r` and `-c`, 1000 each by default), 200 provider recommendations in `candidates` and `both` modes, and, in `arxiv-corpus` and `both` modes, the 180 nearest corpus rows to the seed in each of two embedding roles. Every candidate gets a document-role vector, publication and preprint versions of one work are reconciled across the pool, and candidates are ranked by reciprocal rank fusion (k = 60) over similarity to the seed, similarity to the seed's bibliography, direct citation links, and recommendation order. The best `max_papers - 1` are kept, with no year term, citation-count term, source quota or semantic-only ceiling, and edge scoring is unchanged.

## How it was evaluated

Related papers were graded 0 (tangent), 1 (useful) or 2 (important) from title and abstract by a blind LLM judge that did not see which system produced a paper. The same judge graded every arm, 2,388 seed and paper pairs in all. The ranking design was chosen on an offline replay of the policy; every number below for the current hybrid comes from builds through the CLI, with each selected paper graded. Regrading hidden duplicates gave the identical grade about 85 to 96 percent of the time and never differed by two grades, and two judge models agreed on 91.5 percent of 200 pairs. An earlier set of grades from primary-source reading was about one grade more lenient and is not used for these numbers. The evaluation data lives in the ignored `local-scratch/hybrid-discovery/v2/` directory: it is machine-local and not part of the repository.

## Results on the development seeds

Five development seeds, at the list sizes of a hosted literature-mapping tool's exports for the same seeds (194 slots in total). Useful counts papers graded 1 or 2; important counts papers graded 2; tangent counts papers graded 0. Every system is scored on the same 194 slots, so each count is a share of the returned list. The ranking design was chosen on these seeds.

| System | Useful | Important | Tangent |
| --- | ---: | ---: | ---: |
| Hosted literature-mapping tool | 62 (32%) | 27 (14%) | 132 (68%) |
| Previous hybrid | 92 (47%) | 47 (24%) | 102 (53%) |
| Current hybrid, `both` (local corpus index and recommendations) | 138 (71%) | 66 (34%) | 56 (29%) |
| Current hybrid, default `candidates` mode | 136 (70%) | 59 (30%) | 58 (30%) |
| Current hybrid, `arxiv-corpus` | 128 (66%) | 64 (33%) | 66 (34%) |

## Results on held-out seeds

Eleven held-out seeds that played no part in the design, 44 related papers each.

| System | Useful | Important | Tangent | Papers graded |
| --- | ---: | ---: | ---: | ---: |
| Previous hybrid | 263 (54%) | 140 (29%) | 220 (46%) | 483 |
| Current hybrid, `both` | 345 (71%) | 180 (37%) | 139 (29%) | 484 |
| Current hybrid, `candidates` | 343 (71%) | 174 (36%) | 140 (29%) | 483 |
| Current hybrid, `arxiv-corpus` | 328 (68%) | 150 (31%) | 156 (32%) | 484 |

In `both` mode nine seeds gained useful papers, one tied and one lost one useful paper, within the judge's repeat noise. In `candidates` mode nine gained and two tied on useful papers while losing one to three important ones. `both` is the strongest mode: against `arxiv-corpus` the recommendation list adds 17 useful and 30 important papers on these seeds.

## Precision by list length

The list length is set by `-p` and did not change: the wider pool feeds the same 44 slots the previous hybrid filled, so the counts above are precision, not recall. Share of the first k papers graded useful / important on the held-out seeds:

| Mode | First 10 | First 20 | First 30 | All 44 |
| --- | ---: | ---: | ---: | ---: |
| `both` | 83% / 56% | 81% / 50% | 77% / 44% | 71% / 37% |
| `candidates` | 82% / 55% | 79% / 47% | 76% / 42% | 71% / 36% |
| `arxiv-corpus` | 84% / 52% | 77% / 42% | 73% / 36% | 68% / 31% |

Precision falls gradually with rank, with no sharp knee: in `both` mode ranks 31 to 44 are 60 percent useful and 23 percent important. The list is always filled to the requested size, so a seed with few relevant candidates is padded with tangents. Tangents per 44-paper graph in `both` mode ran from 5 to 27 across the held-out seeds, median 11; for the previous hybrid they ran from 14 to 28, median 19.

## Limits

- One judge family, reading abstracts only.
- The development seeds are not independent evidence, because the ranking design was chosen on them.
- Recall against the whole literature is not measured.
- For seeds with more than 1,000 citers the provider returns only the most recent 1,000, so older follow-up work reaches the list only through recommendations or corpus neighbors.
- Edges and their scoring were not evaluated or changed.
- CLI builds reproduce the offline replay closely in the corpus modes (97 percent of selected papers shared in `both`, 96 percent in `arxiv-corpus`). In `candidates` mode a build encodes the provider's title and abstract where the replay used stored index vectors, so it shares about 81 percent of the replay's selections; the graded result above is the measurement, not the replay.
- The seed's reference list comes from the provider, and the arXiv bibliography is read only when the provider returns none. The provider's list can omit a few references (one to five on ten of the sixteen seeds), which then rank without their direct-citation evidence. Without the local index, bibliography recovery resolves only entries that carry an explicit identifier.
- A publication and its preprint are folded when they share an identifier, or when title and first author agree and the publication year is the preprint's or the next. A version republished under a different title is not recognized and can take a second slot; one such pair appeared across the 48 validation builds (16 seeds in three modes).

The supported behavior is described in [strategies](../guides/strategies.md#hybrid) and [how it works](../guides/how-it-works.md#5-ranking-and-selection).
