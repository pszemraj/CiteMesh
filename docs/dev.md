# Developer Notes

This document captures implementation notes and deferred refactor work for future passes.
For user-facing behavior and runtime details, use the guides/reference docs in [docs/README.md](README.md).

## Deferred Riskier Consolidations

The following items were intentionally deferred during the easy-win + breaking API cleanup pass:

- `#5` Retry/backoff consolidation in `citemesh/services/semantic_scholar.py`
  - Goal: unify retry scaffolding used by `_request_json`, `get_paper`, `_get_related_papers`, and `get_reference_ids`.
  - Deferred because endpoint-specific exception handling and fallback return policies differ and are easy to regress.
- `#7` Title+abstract formatter unification across `citemesh/data/model_profiles.py` and `citemesh/data/embedding_cache.py`
  - Deferred because formatter changes can alter embedding text hashes and trigger broad cache invalidation/rebuilds.
- `#8` Year-scale normalization unification across `citemesh/visualization/render.py` and `citemesh/visualization/export.py`
  - Deferred because current render/export paths intentionally handle missing years with different semantics.
- `#9` arXiv canonicalization unification across `citemesh/strategies/embedding.py` and `citemesh/services/semantic_scholar.py`
  - Deferred because these layers currently normalize identifiers for different bounded contexts (dataset vs API inputs).
- Test overlap trims `#2` and `#4`
  - Goal: reduce duplicated normalization assertions and overlapping deterministic-ordering checks.
  - Deferred to avoid accidental loss of edge-case coverage before targeted replacement tests are added.
- `#12` Cross-strategy score taxonomy harmonization
  - Goal: define optional calibrated score bands/labels that can be consumed uniformly across citation/recommendation/embedding/hybrid outputs.
  - Deferred because current workflows intentionally use strategy-specific scoring math and need a calibration design pass before claiming comparability.
- `#13` Hydration dataset identity hardening beyond source-name checks
  - Goal: persist and verify immutable dataset revision/fingerprint metadata so warm-cache fast paths can prove equivalence to a fresh hydration when upstream dataset aliases change.
  - Deferred because current behavior relies on source-name/split/corpus boundaries plus formatter/model provenance; robust revision checks need stable identity contracts across streaming and non-streaming loaders.
- `#14` ANN retrieval backend for hydrated embedding corpora
  - Goal: keep SQLite/HDF5 for metadata/cold storage, but move hot retrieval onto a real vector index (FAISS or USEARCH) instead of refining the current HDF5 scan path indefinitely.
  - Deferred because this needs an index lifecycle design (build/update/rebuild semantics, row-id mapping, and cache invalidation boundaries) plus benchmark-driven acceptance criteria.
- `#15` Embedding runtime backend/attention policy expansion
  - Goal: broaden runtime selection beyond the current BF16-or-float32 torch policy to include explicit attention-kernel selection plus CPU-focused ONNX/OpenVINO backends where they materially help.
  - Deferred because it needs a tested runtime matrix and feature-detection policy, not just more branches in `_resolve_model_kwargs()`.
- `#17` Separate cache upsert API from embedding retrieval return path
  - Goal: let hydration/write-heavy callers persist embeddings without paying the extra dict/materialization path used by `EmbeddingCache.get_embeddings(...)` return values.
  - Deferred because the new two-phase cache flow should settle first; then the write-only API can be introduced with cleaner call sites and without duplicating persistence logic.

## Follow-up Conditions

Promote deferred items into an implementation pass only when:

- characterization tests pin current behavior for each impacted path, and
- acceptance criteria explicitly preserve endpoint-specific error/return contracts and missing-data semantics.
