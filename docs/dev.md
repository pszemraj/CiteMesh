# Developer Notes

Deferred implementation work:

- Retry/backoff scaffolding in `citemesh/services/semantic_scholar.py`: unify `_request_json`, `get_paper`, `_get_related_papers`, and `get_reference_ids` only after characterization tests pin each endpoint's return and error contract.
- Title/abstract formatter unification across `citemesh/data/model_profiles.py` and `citemesh/data/embedding_cache.py`: formatter changes alter embedding text hashes and can invalidate existing caches.
- Year-scale normalization across `citemesh/visualization/render.py` and `citemesh/visualization/export.py`: render and export paths currently treat missing years differently.
- arXiv canonicalization across `citemesh/strategies/embedding.py` and `citemesh/services/semantic_scholar.py`: dataset ingestion and API input normalization operate under different assumptions.
- Test overlap trims for normalization and deterministic-ordering assertions: replace duplicates only after targeted edge-case coverage exists.
- Cross-strategy score taxonomy harmonization: current scoring is intentionally strategy-specific and is not calibrated for cross-strategy comparison.
- Hydration dataset identity hardening beyond source-name checks: warm-cache reuse still depends on source/split/corpus metadata rather than immutable upstream dataset revisions.
- ANN retrieval backend for hydrated embedding corpora: keep SQLite/HDF5 for metadata and cold storage, but move hot retrieval onto a vector index once index lifecycle and rebuild rules are defined.
- Separate cache upsert API from embedding retrieval return path: add a write-only hydration path after the two-phase cache flow settles.

## Follow-up Conditions

Promote deferred items into an implementation pass only when:

- characterization tests pin current behavior for each impacted path, and
- acceptance criteria explicitly preserve endpoint-specific error/return contracts and missing-data semantics.
