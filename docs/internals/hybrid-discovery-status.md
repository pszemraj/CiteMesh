# Hybrid discovery status

This note records what the `hybrid` strategy does after the discovery rework and the evidence behind the design. It addresses the discovery-precision concern raised in issues #14, #16 and #29.

## What hybrid does now

Hybrid pools every reference and citer of the focal paper that Semantic Scholar returns (fetch limits `-r` and `-c`, 1000 each by default), 200 provider recommendations in `candidates` and `both` modes, and, in `arxiv-corpus` and `both` modes, the 180 nearest corpus rows to the focal paper in each of two embedding roles. Every candidate gets a document-role vector, publication and preprint versions of one work are reconciled across the pool, and candidates are ranked by reciprocal rank fusion (k = 60) over similarity to the focal paper, similarity to the focal paper's bibliography, direct citation links, and recommendation order. The best `max_papers - 1` are kept, with no year term and no citation-count term, whatever a candidate's source. Edges follow [edge scoring](../guides/how-it-works.md#6-edge-scoring), where the focal paper's own references and citers take every other one of the focal paper's edge slots.

## How it was evaluated

Related papers were graded 0 (tangent), 1 (useful) or 2 (important) from title and abstract by a blind LLM judge that did not see which system produced a paper. The same judge graded every arm, 2,388 pairs of focal and related papers in all. The ranking design was chosen on an offline replay of the policy; every number below for the current hybrid comes from builds through the CLI, with each selected paper graded. Regrading hidden duplicates gave the identical grade about 85 to 96 percent of the time and never differed by two grades, and two judge models agreed on 91.5 percent of 200 pairs. An earlier set of grades from primary-source reading was about one grade more lenient and is not used for these numbers. The evaluation data lives in the ignored `local-scratch/hybrid-discovery/v2/` directory: it is machine-local and not part of the repository.

The tables below were measured before embedding text was normalized with `rehuman` and the public arXiv index was rebuilt on 2026-11-12. Rerunning `both` mode on the rebuilt index gave 358 useful (74%) and 184 important (38%) of the 483 held-out papers, and 137 useful (71%) and 62 important (32%) of the 194 development slots. Twelve of those papers were new to the label set and were graded by the same model through its API. The other two modes were not rerun.

## Results on the development focal papers

Five development focal papers, at the list sizes of a hosted literature-mapping tool's exports for the same focal papers (194 slots in total). Useful counts papers graded 1 or 2; important counts papers graded 2; tangent counts papers graded 0. Every system is scored on the same 194 slots, so each count is a share of the returned list. The ranking design was chosen on these focal papers.

| System | Useful | Important | Tangent |
| --- | ---: | ---: | ---: |
| Hosted literature-mapping tool | 62 (32%) | 27 (14%) | 132 (68%) |
| Previous hybrid | 92 (47%) | 47 (24%) | 102 (53%) |
| Current hybrid, `both` (local corpus index and recommendations) | 138 (71%) | 66 (34%) | 56 (29%) |
| Current hybrid, default `candidates` mode | 136 (70%) | 59 (30%) | 58 (30%) |
| Current hybrid, `arxiv-corpus` | 128 (66%) | 64 (33%) | 66 (34%) |

## Results on held-out focal papers

Eleven held-out focal papers that played no part in the design, 44 related papers each.

| System | Useful | Important | Tangent | Papers graded |
| --- | ---: | ---: | ---: | ---: |
| Previous hybrid | 263 (54%) | 140 (29%) | 220 (46%) | 483 |
| Current hybrid, `both` | 345 (71%) | 180 (37%) | 139 (29%) | 484 |
| Current hybrid, `candidates` | 343 (71%) | 174 (36%) | 140 (29%) | 483 |
| Current hybrid, `arxiv-corpus` | 328 (68%) | 150 (31%) | 156 (32%) | 484 |

In `both` mode nine focal papers gained useful papers, one tied and one lost one useful paper, within the judge's repeat noise. In `candidates` mode nine gained and two tied on useful papers while losing one to three important ones. `both` is the strongest mode: against `arxiv-corpus` the recommendation list adds 17 useful and 30 important papers on these focal papers.

## Precision by list length

The list length is set by `-p` and did not change: the wider pool feeds the same 44 slots the previous hybrid filled, so the counts above are precision, not recall. Share of the first k papers graded useful / important on the held-out focal papers:

| Mode | First 10 | First 20 | First 30 | All 44 |
| --- | ---: | ---: | ---: | ---: |
| `both` | 83% / 56% | 81% / 50% | 77% / 44% | 71% / 37% |
| `candidates` | 82% / 55% | 79% / 47% | 76% / 42% | 71% / 36% |
| `arxiv-corpus` | 84% / 52% | 77% / 42% | 73% / 36% | 68% / 31% |

Precision falls gradually with rank, with no sharp knee: in `both` mode ranks 31 to 44 are 60 percent useful and 23 percent important. The dashboard opens hybrid graphs in this order; in its previous default order (relevance over the final graph) the first 10 papers in `both` mode were 76 percent useful and 49 percent important. The list is always filled to the requested size, so a focal paper with few relevant candidates is padded with tangents. Tangents per 44-paper graph in `both` mode ran from 5 to 27 across the held-out focal papers, median 11; for the previous hybrid they ran from 14 to 28, median 19.

## Follow-up checks on ranking and recommendation sources

Two alternatives to the direct-citation vote were replayed on the eleven held-out machine-learning papers. In the first ten slots per focal paper, the current candidate-only ranking returned 93 useful papers, including 63 important papers, out of 110. Giving every direct citation equal credit yielded 89 useful and 60 important. Halving the overlapping semantic credit yielded 88 and 61. Both variants also lost important papers when reranking the original CLI selections. The default fusion weights were retained.

For two focal papers outside computer science, we held references and citers fixed and changed only the recommendation pool, using the pinned EmbeddingGemma model on CUDA. A blind judge graded the union of each pair of top tens. For the [repeating fast radio burst in M81](https://arxiv.org/abs/2105.11445), the two pools produced the same ten records, all graded useful or important. Two were versions of the same paper. For the [Chicxulub impact](https://doi.org/10.1126/science.1177265), the provider's `recent` pool returned five useful records including four important ones, compared with four and three for `all-cs`. A study of Chicxulub ejecta entering at rank nine accounts for the gain. The provider supplied 50 citers for this comparison, no references, and no abstracts for the focal paper or either top ten, so the judgment rests on titles alone. All sixteen focal papers in the machine-learning evaluation carry a computing subject in both the provider and indexed builds and continue to use `all-cs`.

The M81 case was also run through the CLI on CUDA with and without the full 3.2-million-paper index. Five specific papers were chosen before retrieval. Both 40-paper graphs found the same two in the first ten and missed the other three. Of the 39 related slots, the two graphs shared 28 distinct works. The candidate-only run surfaced a white dwarf merger study at rank 28, while the indexed run surfaced a separate sub-100 ns observation of the same source at rank 35. Both graphs contained a preprint and its published version under different titles. The indexed graph contained a second version pair.

## Limits

- Related papers are graded from title and abstract. On the development focal papers, a second judge family (GPT-6.1 Sol) put CiteMesh ahead of the hosted tool on every focal paper, and giving the judge the focal paper's full text on three of them raised the hosted lists by 6 to 8 useful papers of 115 and CiteMesh's by at most 1.
- The development focal papers are not independent evidence, because the ranking design was chosen on them.
- Recall against the whole literature is not measured.
- For focal papers with more than 1,000 citers the provider returns only the most recent 1,000, so older follow-up work reaches the list only through recommendations or corpus neighbors.
- Edges were not evaluated. Giving the focal paper's own references and citers its edge slots changes edges only, not which papers are selected.
- CLI builds reproduce the offline replay closely in the corpus modes (97 percent of selected papers shared in `both`, 96 percent in `arxiv-corpus`). In `candidates` mode a build encodes the provider's title and abstract where the replay used stored index vectors, so it shares about 81 percent of the replay's selections; the graded result above is the measurement, not the replay.
- The focal paper's reference list comes from the provider, and the arXiv bibliography is read only when the provider returns none. The provider's list can omit a few references (one to five on ten of the sixteen focal papers), which then rank without their direct-citation evidence. Without the local index, bibliography recovery resolves only entries that carry an explicit identifier.
- A publication and its preprint are folded when they share an identifier or meet the [bibliographic matching rules](../guides/how-it-works.md#2-candidate-acquisition). A version republished under a different title can take a second slot. One such pair appeared across the 48 validation builds (16 focal papers in three modes).

## Possible follow-ups

- Compare with the hosted literature-mapping tool on held-out focal papers. Its exports exist only for the five development focal papers, so the comparison above rests on focal papers the design was chosen on. Exporting its graphs for some of the eleven held-out focal papers and grading them with the same judge would show whether the lead holds on focal papers that played no part in the design.

The supported behavior is described in [strategies](../guides/strategies.md#hybrid) and [how it works](../guides/how-it-works.md#5-ranking-and-selection).
