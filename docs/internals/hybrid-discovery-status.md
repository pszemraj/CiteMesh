# Hybrid discovery status

This note records what the `hybrid` strategy does after the discovery rework and the evidence behind the design. It addresses the discovery-precision concern raised in issues #14, #16 and #29.

## What hybrid does now

Hybrid pools every seed reference and citer that Semantic Scholar returns (fetch limits `-r` and `-c`, 1000 each by default), 200 provider recommendations in `candidates` and `both` modes, and, in `arxiv-corpus` and `both` modes, the 180 nearest corpus rows to the seed in each of two embedding roles. Every candidate gets a document-role vector, publication and preprint versions of one work are reconciled across the pool, and candidates are ranked by reciprocal rank fusion (k = 60) over similarity to the seed, similarity to the seed's bibliography, direct citation links, and recommendation order. The best `max_papers - 1` are kept, with no year term, citation-count term, source quota or semantic-only ceiling, and edge scoring is unchanged.

## How it was evaluated

Related papers were graded 0 (tangent), 1 (useful) or 2 (important) from title and abstract by a blind LLM judge that did not see which system produced a paper. The same judge graded every arm, 2,306 seed and paper pairs in all. Regrading hidden duplicates gave the identical grade about 85 to 96 percent of the time and never differed by two grades, and two judge models agreed on 91.5 percent of 200 pairs. An earlier set of grades from primary-source reading was about one grade more lenient and is not used for these numbers. The evaluation data lives in the ignored `local-scratch/hybrid-discovery/v2/` directory: it is machine-local and not part of the repository.

## Results on the development seeds

Five development seeds, at the list sizes of a hosted literature-mapping tool's exports for the same seeds (194 slots in total). Useful counts papers graded 1 or 2; important counts papers graded 2. The ranking design was chosen on these seeds.

| System | Useful | Important |
| --- | ---: | ---: |
| Hosted literature-mapping tool | 62 | 27 |
| Previous hybrid | 92 | 47 |
| Current hybrid, local corpus index (`both`) | 143 | 68 |
| Current hybrid, default `candidates` mode | 135 | 64 |

## Results on held-out seeds

Eleven held-out seeds that played no part in the design, 44 related papers each.

| System | Useful | Important | Papers graded |
| --- | ---: | ---: | ---: |
| Previous hybrid | 263 | 140 | 483 |
| Current hybrid, `both` | 345 | 182 | 484 |
| Current hybrid, `candidates` | 347 | 177 | 484 |

Nine seeds improved, one tied and one was one paper lower, within the judge's repeat noise. Precision falls with rank, so the first twenty papers are the most reliable.

## Limits

- One judge family, reading abstracts only.
- The development seeds are not independent evidence, because the ranking design was chosen on them.
- Recall against the whole literature is not measured.
- For seeds with more than 1,000 citers the provider returns only the most recent 1,000, so older follow-up work reaches the list only through recommendations or corpus neighbors.
- Edges and their scoring were not evaluated.
- These numbers come from an offline replay of the policy; a live build matched the replayed list on 43 of 44 papers for the one seed checked so far.

The supported behavior is described in [strategies](../guides/strategies.md#hybrid) and [how it works](../guides/how-it-works.md#5-ranking-and-selection).
