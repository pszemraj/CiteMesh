# Hybrid discovery status

These evaluations compare changes to [hybrid discovery and ranking](../guides/how-it-works.md#5-ranking-and-selection) in response to the precision concerns in issues #14, #16, and #29.

## How it was evaluated

Related papers were graded 0 (tangent), 1 (useful) or 2 (important) from title and abstract by a blind LLM judge that did not see which system produced a paper. The same judge graded every arm, 2,388 pairs of focal and related papers in all. The ranking design was chosen on an offline replay of the policy; the development, held-out and list-length tables below report CLI builds of the four-list ranker, with each selected paper graded. The later focal-reference experiment reports its paired results separately. Regrading hidden duplicates gave the identical grade about 85 to 96 percent of the time and never differed by two grades, and two judge models agreed on 91.5 percent of 200 pairs. An earlier set of grades from primary-source reading was about one grade more lenient and is not used for these numbers. The evaluation data lives in the ignored `local-scratch/hybrid-discovery/v2/` directory: it is machine-local and not part of the repository.

The tables below were measured before embedding text was normalized with `rehuman` and the public arXiv index was rebuilt on 2026-11-12. Rerunning `both` mode on the rebuilt index gave 358 useful (74%) and 184 important (38%) of the 483 held-out papers, and 137 useful (71%) and 62 important (32%) of the 194 development slots. Twelve of those papers were new to the label set and were graded by the same model through its API. The other two modes were not rerun.

## Four-list results on the development focal papers

Five development focal papers, at the list sizes of a hosted literature-mapping tool's exports for the same focal papers (194 slots in total). Useful counts papers graded 1 or 2; important counts papers graded 2; tangent counts papers graded 0. Every system is scored on the same 194 slots, so each count is a share of the returned list. The ranking design was chosen on these focal papers.

| System | Useful | Important | Tangent |
| --- | ---: | ---: | ---: |
| Hosted literature-mapping tool | 62 (32%) | 27 (14%) | 132 (68%) |
| Hybrid before rank fusion | 92 (47%) | 47 (24%) | 102 (53%) |
| Four-list hybrid, `both` | 138 (71%) | 66 (34%) | 56 (29%) |
| Four-list hybrid, `candidates` | 136 (70%) | 59 (30%) | 58 (30%) |
| Four-list hybrid, `arxiv-corpus` | 128 (66%) | 64 (33%) | 66 (34%) |

## Four-list results on held-out focal papers

Eleven held-out focal papers played no part in the design. Each build requested 44 related papers.

| System | Useful | Important | Tangent | Papers graded |
| --- | ---: | ---: | ---: | ---: |
| Hybrid before rank fusion | 263 (54%) | 140 (29%) | 220 (46%) | 483 |
| Four-list hybrid, `both` | 345 (71%) | 180 (37%) | 139 (29%) | 484 |
| Four-list hybrid, `candidates` | 343 (71%) | 174 (36%) | 140 (29%) | 483 |
| Four-list hybrid, `arxiv-corpus` | 328 (68%) | 150 (31%) | 156 (32%) | 484 |

In `both` mode nine focal papers gained useful papers, one tied and one lost one useful paper, within the judge's repeat noise. In `candidates` mode nine gained and two tied on useful papers while losing one to three important ones. `both` had the highest full-list counts for this cohort, with 17 more useful papers and 30 more important papers than `arxiv-corpus`.

## Precision by list length

Every comparison used the same `-p` limit of 44 related papers. These figures measure precision among returned papers. The table shows the share of the first k papers graded useful and important on the held-out focal papers.

| Mode | First 10 | First 20 | First 30 | All 44 |
| --- | ---: | ---: | ---: | ---: |
| `both` | 83% / 56% | 81% / 50% | 77% / 44% | 71% / 37% |
| `candidates` | 82% / 55% | 79% / 47% | 76% / 42% | 71% / 36% |
| `arxiv-corpus` | 84% / 52% | 77% / 42% | 73% / 36% | 68% / 31% |

Precision fell gradually with rank. In `both` mode ranks 31 to 44 were 60 percent useful and 23 percent important. Ordering the same graphs by the earlier dashboard default (**Graph relevance**) put 76 percent useful and 49 percent important papers in the first ten. When enough candidates are available, ranking can fill the budget with tangents after the relevant papers run out. Tangents per 44-paper graph in `both` mode ranged from 5 to 27 across these focal papers, with a median of 11. Before rank fusion the range was 14 to 28, with a median of 19.

## Follow-up checks on ranking and recommendation sources

Two alternatives to the direct-citation vote were replayed on the eleven held-out machine-learning papers. In the first ten slots per focal paper, the four-list `candidates` ranking returned 93 useful papers, including 63 important papers, out of 110. Giving every direct citation equal credit yielded 89 useful and 60 important. Halving the overlapping semantic credit yielded 88 and 61. Both variants also lost important papers when reranking the original CLI selections. The direct-citation list kept its original weight.

For two focal papers outside computer science, we held references and citers fixed and changed only the recommendation pool, using the pinned EmbeddingGemma model on CUDA. A blind judge graded the union of each pair of top tens. For the [repeating fast radio burst in M81](https://arxiv.org/abs/2105.11445), the two pools produced the same ten records, all graded useful or important. Two were versions of the same paper. For the [Chicxulub impact](https://doi.org/10.1126/science.1177265), the provider's `recent` pool returned five useful records including four important ones, compared with four and three for `all-cs`. A study of Chicxulub ejecta entering at rank nine accounts for the gain. The provider supplied 50 citers for this comparison, no references, and no abstracts for the focal paper or either top ten, so the judgment rests on titles alone. All sixteen focal papers in the machine-learning evaluation carry a computing subject in both the provider and indexed builds and continue to use `all-cs`.

The M81 case was also run through the CLI on CUDA with and without the full 3.2-million-paper index. Five specific papers were chosen before retrieval. Both 40-paper graphs found the same two in the first ten and missed the other three. Of the 39 related slots, the two graphs shared 28 distinct works. The candidate-only run surfaced a white dwarf merger study at rank 28, while the indexed run surfaced a separate sub-100 ns observation of the same source at rank 35. Both graphs contained a preprint and its published version under different titles. The indexed graph contained a second version pair.

## Citation use in the focal paper

A reference discussed in several passages can carry a central definition or result regardless of title similarity. This comparison added the [focal reference vote](../guides/how-it-works.md#5-ranking-and-selection) while holding candidate embeddings and graph size fixed.

For [*Navier-Stokes lost in translation*](https://arxiv.org/abs/2610.08144), ten specific works were identified before examining any recommendations. The following ranks come from complete local EmbeddingGemma builds with 40 related papers. Each pair shows the original and revised rank.

| Previously identified work | Candidate embeddings | With full arXiv index |
| --- | ---: | ---: |
| *Formalizing Mathematics at Scale* | 31 → 4 | 69 → 21 |
| *Computing Spectra* (listed under its earlier title) | 75 → 12 | 52 → 9 |
| Jones, *Universal diophantine equation* | 99 → 30 | 74 → 22 |

Two further works, *The Faithfulness Gap* and *Reliable Evaluation and Benchmarks for Statement Autoformalization*, appeared at positions 11 and 28 in the index query. The dashboard links both under **More papers**. Five of the ten specified works were missing from the indexed candidate pool, including papers on the particular fluid construction.

Two source-blinded reviewers graded each work entering or leaving the main reading list. Both counted four more important works in the candidate-only graph and five more in the indexed graph. On identical frozen pools for eleven earlier focal papers, the revision increased important works in the first ten from 63 to 68 with candidate embeddings and from 63 to 67 with the index, out of 110 slots each. The candidate-only Chinchilla graph lost three important works at its full reading budget. On the five development focal papers at their original graph sizes, the half-weight vote gained one important work without the index and two with it. The total graded useful, which includes the important works, fell by one and two respectively.

A separate identity check used all five original reference exports at matched sizes, totaling 194 related-paper slots per mode. For each new build we reconstructed the previous ranking on exactly the same candidates. The selected reference works were identical before and after on all five focal papers, totaling 26 matches with candidate embeddings and 28 with the index. Five of the ten builds reused provider responses after rate limiting, including archived recommendations for Rethinking Pre-pretraining. Each mode used its normal local embedding path. One damaged reference record could not be identified. Some incoming and outgoing works had no relevance grade, so the reference overlap cannot settle their reading value.

PolyCom illustrates a retrieval limit. Each mode acquired four of its 38 identifiable reference works. Of seven previously judged important, only *Rational Neural Networks* entered either pool. Both rankings selected it. Additional candidate sources or queries would be needed to reach the others.

## Limits

- On the development focal papers, a second judge family (GPT-6.1 Sol) put CiteMesh ahead of the hosted tool on every focal paper, and giving the judge the focal paper's full text on three of them raised the hosted lists by 6 to 8 useful papers of 115 and CiteMesh's by at most 1.
- Recall against the whole literature is not measured.
- The builds capped citing papers at 1,000. Follow-up work beyond those provider results could still enter through recommendations or corpus neighbors.
- Edges were not evaluated. Giving the focal paper's own references and citers its edge slots changes edges only, not which papers are selected.
- CLI builds reproduce the offline replay closely in the corpus modes (97 percent of selected papers shared in `both`, 96 percent in `arxiv-corpus`). In `candidates` mode a build encodes the provider's title and abstract where the replay used stored index vectors, so it shares about 81 percent of the replay's selections; the graded result above is the measurement, not the replay.
- The provider's lists omitted one to five references on ten of the sixteen focal papers. Those references can rank without their direct-citation evidence. See [when arXiv bibliography recovery runs](../guides/cli.md#missing-semantic-scholar-references).
- A publication and its preprint are folded when they share an identifier or meet the [bibliographic matching rules](../guides/how-it-works.md#2-candidate-acquisition). A version republished under a different title can take a second slot. One such pair appeared across the 48 validation builds (16 focal papers in three modes).

## Possible follow-ups

- Compare reference exports for some of the eleven held-out focal papers. The five development exports helped shape the original ranker. The later Navier-Stokes case adds a different subject and provides one additional focal paper for a matched comparison.
