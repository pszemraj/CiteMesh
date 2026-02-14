# Developer Notes

This document captures implementation notes and deferred refactor work for future passes.
It is not a normative behavior spec for CLI, strategies, or cache contracts.

## Deferred Riskier Consolidations

The following items were intentionally deferred during the easy-win + breaking API cleanup pass:

- `#5` Retry/backoff consolidation in `citemesh/services/semantic_scholar.py`
  - Scope: unify retry scaffolding used by `_request_json`, `get_paper`, `_get_related_papers`, and `get_reference_ids`.
  - Deferred because endpoint-specific exception handling and fallback return policies differ and are easy to regress.
- `#7` Title+abstract formatter unification across `citemesh/data/model_profiles.py` and `citemesh/data/embedding_cache.py`
  - Deferred because formatter changes can alter embedding text hashes and trigger broad cache invalidation/rebuilds.
- `#8` Year-scale normalization unification across `citemesh/visualization/render.py` and `citemesh/visualization/export.py`
  - Deferred because current render/export paths intentionally handle missing years with different semantics.
- `#9` arXiv canonicalization unification across `citemesh/strategies/embedding.py` and `citemesh/services/semantic_scholar.py`
  - Deferred because these layers currently normalize identifiers for different bounded contexts (dataset vs API inputs).
- Test overlap trims `#2` and `#4`
  - Scope: reduce duplicated normalization assertions and overlapping deterministic-ordering checks.
  - Deferred to avoid accidental loss of edge-case coverage before targeted replacement tests are added.
- `#10` Embedding cache two-phase lock refactor (`check -> unlock -> encode -> relock -> commit`)
  - Scope: reduce lock hold duration during long model encode calls by moving compute outside the namespace lock with safe re-check/commit semantics.
  - Deferred because this touches cache coherence across SQLite/HDF5 writes and needs dedicated race characterization tests.
- `#11` CLI/API validation parity for strategy option contracts
  - Scope: expose a shared validator callable from both CLI parsing and direct builder/API entrypoints so invalid combinations are rejected consistently outside CLI.
  - Deferred because current strategy constructors already guard high-risk invariants, and introducing a shared contract layer needs a stable public API boundary decision.
- `#12` Cross-strategy score taxonomy harmonization
  - Scope: define optional calibrated score bands/labels that can be consumed uniformly across citation/recommendation/embedding/hybrid outputs.
  - Deferred because current workflows intentionally use strategy-specific scoring math and need a calibration design pass before claiming comparability.

## Follow-up Conditions

Promote deferred items into an implementation pass only when:

- characterization tests pin current behavior for each impacted path, and
- acceptance criteria explicitly preserve endpoint-specific error/return contracts and missing-data semantics.
