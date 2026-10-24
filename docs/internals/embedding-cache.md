# Embedding cache internals

`EmbeddingCache` coordinates SQLite metadata and HDF5 vectors so interrupted writes can be recovered. For normal operation, see [cache reuse and hydration](../guides/caching.md).

## Store composition

`EmbeddingCache` (`store.py`) is one class assembled from five mixins, in MRO order:

```python
class EmbeddingCache(
    _LifecycleMixin, _IngestMixin, _H5LayoutMixin, _RecoveryMixin, _SearchMixin
):
```

- `lifecycle.py` (`_LifecycleMixin`) - strict existing-cache inspection, certified corpus state, membership, read-only vector access, and owned import/refresh sessions.
- `ingest.py` (`_IngestMixin`) - the three-phase write pipeline: `_process_embeddings_locked` scans for hits under the lock, encodes **outside** it, then commits under it again (`_scan_cache_for_hits`, `_encode_pending_texts`, `_commit_encoded_embeddings`, `_apply_replacement_rows`, `_append_new_rows`).
- `layout.py` (`_H5LayoutMixin`) - SQLite schema creation and migration (`_init_db`), cache-metadata get/set, the runtime-contract stamp and check (`_assert_runtime_cache_consistency`), HDF5 dataset creation and validation, and the open-time repair pass `_ensure_h5_layout`.
- `recovery.py` (`_RecoveryMixin`) - locking (`_cache_lock`, `_cache_operation_lock`, `hydration_operation_lock`), the SQLite connection context, the replacement journal and its replay, trailing-row truncation (`_recover_trailing_rows`), and the durability barrier `_flush_h5_file`.
- `search.py` (`_SearchMixin`) - query-time scoring: the calibration gate, the Hamming prefilter, the chunked scoring kernels (`_score_int8_rows`, `_score_float_rows`, `_select_top_k`), the finiteness guard, and the row-to-metadata join.

Supporting modules: `constants.py` (metadata keys, schema version, chunk sizes, lock timeout), `sql.py` (DDL and upserts), `models.py` (result dataclasses), `quantization.py` (int8 and binary packing). `data/embedding_contract.py` owns the pure namespace serializer and fixed embedding/source selection types; it does not load models. `lifecycle.py` owns managed corpus state, strict opening, membership, and import/refresh sessions. `portable.py` streams the interchange format through lazy Arrow imports.

## Physical layout

All namespaces share one `embeddings/` directory, separated by a filename suffix rather than a subdirectory: `sha256(model_name)[:12]` gives `metadata_<hash>.db`, `embeddings_<hash>.h5`, `cache_<hash>.lock`, and `hydration_<hash>.lock`. SQLite holds `papers` (keyed by `paper_id`, carrying `text_hash`, `row_idx`, normalized metadata, and a nullable packed arXiv `chronology_key`), `cache_metadata` (key/value), `replacement_journal` (the undo log below), and `corpus_membership` (active IDs and source ordinals). `papers` carries two indexes, `idx_papers_row_idx` and `idx_papers_chronology_key`; corpus selection and export read chronology directly. HDF5 holds `embeddings` (resizable `N x dim`, chunked at 2048 rows), the optional `binary_index` (`N x ceil(dim/8)` `uint8`, tagged `int8-midpoint-sign-v1`), and for int8 namespaces the fixed `calibration_ranges` (`2 x dim` float32). The runtime contract lives in HDF5 root attrs and is re-asserted on every open.

`sqlite3` uses its default rollback journal and durability settings. The HDF5 flush barrier below orders writes across the two stores.

## Write ordering and the replacement journal

Each embedding commit flushes at the end of its second locked phase: **HDF5 data is flushed and fsynced before the SQLite rows that point at it are committed**, so a durable mapping never outlives the vector it references. The opposite order would leave committed mappings recovery could only fix by discarding good rows.

Existing-row replacements use a durable SQLite undo journal under the namespace lock: before overwriting a vector, CiteMesh commits the previous vector and binary-index row to `replacement_journal`, and deletes those entries in the same transaction that commits the new paper metadata. If that commit fails or the process stops, the next access restores the journaled rows and removes uncommitted trailing appends before reading vectors. Failed restoration stops access and keeps the journal for another attempt.

During cache-native search, a scored row with no metadata mapping fails closed with an integrity error rather than returning partial top-k results.

## What a crash leaves, and how it is repaired

- **Orphan trailing HDF5 rows** - an append that resized and wrote the matrix but never committed its SQLite mappings. `_recover_trailing_rows` truncates matrix and binary index back to the contiguous prefix shared by SQLite and HDF5 and marks hydration incomplete; the inverse case, mappings whose vectors were lost, deletes those mappings and does the same. Truncation happens only when the committed SQLite prefix is dense (`0..n-1`); an ambiguous mapping is left alone and surfaces as an integrity error rather than being silently "repaired".
- **A half-applied replacement** - the journal replay above. Row bounds, embedding width, and payload byte size are validated before a restore; a mismatch raises rather than guessing, and existing files are preserved.
- **An interrupted first encode** - a file with no datasets and no schema attributes. That is an *empty* namespace, not a proven mismatch: reopening discards mappings whose vectors were never persisted, marks hydration incomplete, keeps the source/split/corpus-cap and corpus-metadata markers, and writes into it without a wipe warning.
- **A calibration-only file** written before the first embedding write survives for resume while its runtime and calibration contract matches; a changed formatter, storage precision, or calibration-sample setting invalidates those ranges.
- **A stale binary index** whose encoding marker or row count disagrees with the embeddings matrix is rebuilt from the persisted int8 rows - no model encoding, no loss of vectors, metadata, or hydration state. A genuinely incompatible index is dropped.

Runtime consistency keys are seeded once, compared on every reopen, and restamped only after that check passes, the namespace is rebuilt, or it holds no vectors. Ambiguous mappings and file-open, lock, or IO failures propagate without clearing the namespace; proven incompatible layouts or invalid calibration metadata do reset it, with the mismatch named in the warning, and a missing HDF5 file clears its stale SQLite mappings.

Those automatic repairs apply to ordinary uncertified caches. Managed corpus
caches use strict inspection and preserve incompatible or missing-half storage
until an explicit reset.

Schema **4** requires rebuilding older namespaces whose rows lack `chronology_key`, which capped-corpus growth checks need as a recency watermark.

## Locking

Three locks use the [embedding-cache timeout](../reference/environment.md#citemesh-variables):

- the per-namespace `cache_<hash>.lock`, wrapping only the short scan and commit phases
- the reentrant `hydration_<hash>.lock`, held across the complete-check, resume or rebuild, and the search that consumes hydrated rows
- the shared cache-root `ReadWriteLock` at `.locks/cache-operations.db`, taken on the read side by every live cache operation and on the write side (non-blocking) by `cache clear`

Exceeding a timeout raises with the lock path and a pointer at the env var or an isolated `CITEMESH_CACHE_DIR`.

## Certified corpus lifecycle

Portable indexes add a `corpus_membership` table keyed by paper ID with the
original source ordinal. This additive change does not invalidate schema-4
vectors. An empty new membership table means unknown coverage until an
operation certifies it. Physical rows can outlive source membership.

Structured cache metadata records the vector contract, original namespace
serialization, imported publication origin, completed source descriptor,
membership certification, and pending operation target. See [portable index
selection](../guides/caching.md#portable-corpus-indexes) for user-facing
namespace and source-revision behavior.

Strict opening holds the namespace lock and opens the existing SQLite file in
read-write mode so SQLite can roll back a hot journal after a crash. It opens
HDF5 read-only. Inspection performs no schema migration or layout repair;
explicit legacy adoption initializes missing native tables after validation.
Missing halves, incompatible contracts, and ambiguous occupied storage fail
without clearing payload. Consumer readiness additionally requires
completed source state and certified membership; refresh can open its own
pending state. Ordinary namespace selection also protects managed storage
before repair. Explicit reset is separate from import, which never replaces an
occupied incompatible cache.

Import and refresh hold root and hydration operation locks, acquiring the short
namespace lock for each mutation. Pending identity is durable before payload
writes. Import appends stored values directly and commits metadata/membership
only after the HDF5 flush barrier. Recovery determines the committed dense
prefix from native mappings. Binary reconstruction runs once at finalization;
it never rewrites primary int8 bytes. Completion promotes source, membership,
and readiness in one SQLite transaction after flushing HDF5.

Refresh rescans one pinned source after interruption, reuses the existing
formatted-text hit/miss comparison and replacement journal, and acquires an
encoder only for misses. Metadata hits update their complete normalized
record. Membership reflects the selected revision at clean EOF; removed
papers' stored rows remain inactive. Failure after replacement leaves the
operation incomplete, since the old source no longer exists intact.

Full and newest-N refreshes share a temporary SQLite staging pass, resolving
canonical IDs across the entire pinned split before calibration or encoding.
Duplicate IDs follow the [source precedence rules](../guides/caching.md#portable-corpus-indexes). Staging
bounds memory by the record batch and uses temporary disk space proportional to
source metadata.
Refresh retries rebuild membership from these winners while preserving native
vectors and calibration; import retries retain their committed prefix.

Ordinary ingest and corpus metadata writes reject a managed corpus outside
its owned operation. Both locked ingest phases check ownership, so an encoder
started before certification cannot commit afterwards. Read-only vector lookup
and the separate symmetric graph cache serve consumers.

Native search intersects every requested view with certified membership before
Hamming candidate selection or int8/float32 scoring. Eligibility is indexed by
physical row; absolute native positions remain the cutoff tie key, and excluded
rows consume no candidate slots. Newest-N views use the
[active membership ordering](../guides/caching.md#portable-corpus-indexes).

## Portable format

Each configuration has `index.json`, ordered Zstandard Parquet shards, and,
for int8 only, the exact float32 `calibration_ranges.npy`. The manifest records
format/snapshot identity, vector contract, original native selectors, source
revision and coverage, producer provenance, and each shard's row interval.
Float32 and int8 vectors support the manifest's dimension without a fixed
512-coordinate assumption.

Producer provenance preserves the original compute and vector dtypes.
`export_runtime_versions` records Python, NumPy, h5py, and PyArrow versions for
the current export; these describe the exporter, not the historical encoder.
Re-exporting an installed snapshot updates this field while retaining its
creation provenance.

Parquet stores dense `row_number`, stable `paper_id`, `arxiv_id`, `doi`, title,
abstract, nullable year, authors/categories lists, venue, formatted `text_hash`,
nullable `chronology_key`, `source_ordinal`, and a non-null fixed-size embedding
list of the declared scalar type. Export reads active rows in ascending native
row order, preserving search tie behavior despite gaps from inactive rows.
Metadata and raw vectors are streamed together under the snapshot lock.

Import validates schema, contiguous portable row intervals, unique paper IDs,
vector width/type, non-null coordinates, finite float32 values, and the native
int8 range shape/type. No normalization or quantization occurs. Defaults are
65,536 rows per export shard and 8,192 rows per Arrow batch; the complete matrix
is never constructed as Python lists in memory. A fresh export directory gains
its manifest only after every artifact closes successfully.

Bucket transport reads the catalog, then downloads the selected manifest and
exactly its declared artifacts. Snapshots live at
`variants/<config>/<snapshot>/index.json`, with payloads beside the manifest.
Each configuration owns its source descriptor. Import origin records
`bucket_id`, `config_name`, and `snapshot_id`, independently of the upstream
source revision. Complete downloads stay in the Hugging Face cache for native
import retries; see the [portable index guide](../guides/caching.md#portable-corpus-indexes)
for resume behavior.

The publisher uploads payloads, then the manifest, updates the catalog's current
pointer, and prunes older files for that configuration. The
[portable index guide](../guides/caching.md#portable-corpus-indexes) describes
the publisher command and retention policy.
