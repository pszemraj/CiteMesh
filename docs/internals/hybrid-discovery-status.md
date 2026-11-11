# Hybrid discovery status

The `feat/hybrid-discovery` branch improves candidate admission and paper
identity handling. It does **not** establish a general discovery-precision
improvement or superiority over hosted literature-mapping tools. The precision investigation
was stopped at the user's request without promoting an experimental ranker or
relevance filter.

## Retained implementation

- Opt-in `semantic_source="both"` combines corpus retrieval and Semantic Scholar
  recommendations, preserving source positions and availability through
  reconciliation and export.
- Healthy provider references are ranked before their admission cap. Verified
  reference evidence survives admission limits; empty-provider arXiv fallback
  can recover unambiguous bibliography identities from the prepared corpus.
- Omitted `max_semantic` allows skipped semantic candidates to fill vacancies;
  an explicit value remains a hard ceiling. This can increase reading volume
  and should not be interpreted as improved precision.
- Identifier reconciliation recognizes arXiv DOIs and explicit arXiv journal
  volumes. Selected publication/preprint duplicates can reconcile through
  mutually unique title, first-author and publication-year matches. Both
  versions must already be selected; cleanup preserves the earlier position
  and evidence without filling the freed slot.

The final ranking weights and designated embedding model remain unchanged. No
additional inference model, experimental filter, graph ranker, wider reference
cap or density correction is included. See [strategies](../guides/strategies.md)
and [how it works](../guides/how-it-works.md) for supported behavior.

## Evidence and remaining work

Saved-pool experiments examined filtering, native-score calibration, citation
neighborhoods, candidate expansion, bibliography recovery, representation
changes and query-density effects. Some improved individual seeds while losing
useful or important papers elsewhere; conservative filters left outputs
unchanged. None supplied sufficient evidence to replace the retained policy.
These failures apply to the tested rules, not to every possible use of those
signals.

The comparison used source-blinded, model-assisted reading of abstracts and
pertinent primary contents, with unresolved evidence preserved. Individual
connections generally have one primary judgment, not reviewer consensus.
Corrected judgments supersede earlier claims of perfect useful-paper retention,
an overall CiteMesh lead, or a general advantage on recent papers. Re-scoring
saved outputs is not a software improvement. The frozen inputs also mix live
captures and admission replays; they are not all fresh outputs of this branch.

The detailed machine-local handoff is
`local-scratch/hybrid-discovery/HANDOFF.md`. It identifies the retained commits,
corrected per-seed comparisons, failed experiments, evidence files and unfinished
checks. That directory is ignored: generated outputs and review packets are not
part of the repository. The handoff is needed to continue the local research;
this note alone does not reproduce its empirical results.

Issues #14, the semantic-slot portion of #16, and #29 motivated this work.
Discovery precision and important omissions remain unresolved; #29 should stay
open. No release, push, model replacement or further experiment is implied by
this status note.
