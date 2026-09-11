# Embedding Cache Internals

Crash-safety invariants and the internal structure of `src/citemesh/data/embedding_cache/`. You do not need this page to use CiteMesh — the on-disk contract, namespace identity, hydration policy, and the flags that control them are in [Caching & Data](../guides/caching.md).

## Store composition

`EmbeddingCache` (`store.py`) is one class assembled from four mixins, in MRO order:

```python
class EmbeddingCache(_IngestMixin, _H5LayoutMixin, _RecoveryMixin, _SearchMixin):
```

| Module | Mixin | Responsibility |
| --- | --- | --- |
| `ingest.py` | `_IngestMixin` | The three-phase write pipeline: `_process_embeddings_locked` scans for hits under the lock, encodes **outside** it, then commits under it again (`_scan_cache_for_hits`, `_encode_pending_texts`, `_prepare_storage_embeddings`, `_commit_encoded_embeddings`, `_plan_vector_writes`, `_apply_replacement_rows`, `_append_new_rows`, `_commit_row_mappings`). |
| `layout.py` | `_H5LayoutMixin` | SQLite schema creation and migration (`_init_db`), cache-metadata get/set, the runtime-contract stamp and consistency check (`_runtime_contract_values`, `_assert_runtime_cache_consistency`), HDF5 dataset creation/validation (`_ensure_embeddings_dataset`, `_ensure_binary_dataset`, `_rebuild_binary_dataset`), and the open-time repair pass `_ensure_h5_layout`. |
| `recovery.py` | `_RecoveryMixin` | Locking (`_cache_lock`, `_cache_operation_lock`, `hydration_operation_lock`), the SQLite connection context (`_connect_db`), the replacement journal (`_persist_replacement_journal` and its replay), trailing-row truncation (`_recover_trailing_rows`), and the durability barrier `_flush_h5_file`. |
| `search.py` | `_SearchMixin` | Query-time scoring: the calibration gate (`_require_calibration_ranges`), the Hamming prefilter (`_binary_prefilter_rows`), the chunked scoring kernels (`_score_int8_rows`, `_score_float_rows`, `_score_chunked_rows`, `_select_top_k`), the finiteness guard, and the row-to-metadata join. |

Supporting modules: `constants.py` (every metadata key, the schema version, chunk sizes, and the lock timeout), `sql.py` (table DDL and the upsert statements), `models.py` (result dataclasses), `quantization.py` (int8 and binary packing).

## Physical layout

All namespaces share one `embeddings/` directory; the namespace is separated by a filename suffix, not by a subdirectory:

```python
model_hash = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
```

giving `metadata_<hash>.db`, `embeddings_<hash>.h5`, `cache_<hash>.lock`, and `hydration_<hash>.lock`. The SQLite side has three tables — `papers` (keyed by `paper_id`, carrying `text_hash`, `row_idx`, and the metadata authority fields), `cache_metadata` (key/value), and `replacement_journal` (the undo log below). The HDF5 side holds `embeddings` (resizable `N x dim`, chunked at 2048 rows), the optional `binary_index` (`N x ceil(dim/8)` `uint8`, tagged with the encoding marker `int8-midpoint-sign-v1`), and, for int8 namespaces, the fixed `calibration_ranges` (`2 x dim` float32). The runtime contract lives in HDF5 root attrs and is re-asserted on every open.

`sqlite3` runs in its default rollback-journal mode; no WAL or `synchronous` pragma is configured. Durability comes from the flush ordering below, not from SQLite settings.

## Write ordering and the replacement journal

Every write flushes exactly once, at the end of the second locked phase, and the order is deliberate: **HDF5 data is flushed and fsynced before the SQLite rows that point at it are committed.** A durable row mapping therefore never outlives the vector it references; the opposite ordering would leave committed mappings that recovery could only fix by discarding good rows.

Existing-row replacements use a durable SQLite undo journal under the namespace lock. Before overwriting a vector, CiteMesh commits its previous stored vector and binary-index row to `replacement_journal`. Journal entries are deleted in the same transaction that commits the new paper metadata. If that commit fails or the process stops, the next access restores the journaled rows and removes uncommitted trailing appends before reading vectors or checking their mappings. Failed restoration stops access and retains the journal for another attempt.

During cache-native search, scored embedding rows must map to metadata rows. A missing metadata row mapping fails closed with an integrity error instead of returning partial top-k results.

## What a crash leaves, and how it is repaired

- **Orphan trailing HDF5 rows** — an append that resized and wrote the matrix but never committed its SQLite mappings. `_recover_trailing_rows` truncates the matrix (and the binary index) back to the contiguous row prefix shared by SQLite and HDF5, and marks hydration incomplete. The inverse case (SQLite mappings whose vectors were lost) deletes those mappings and likewise marks hydration incomplete. Truncation is attempted only when the committed SQLite prefix is dense (`0..n-1`, no gaps or duplicates); an ambiguous mapping is left alone and surfaces as an integrity error instead of being silently "repaired".
- **A half-applied replacement** — the journal replay above. Row bounds, embedding width, and payload byte size are all validated before a restore; a mismatch raises rather than guessing, and existing cache files are preserved.
- **An interrupted first encode** — a file holding no datasets and no schema attributes. This is an *empty* namespace, not a proven mismatch: reopening discards row mappings whose vectors were never persisted, marks hydration incomplete, keeps the dataset-source/split/corpus-cap and corpus-metadata markers, and writes into it without an incompatible-schema wipe warning.
- **A calibration-only file** written before the first embedding write is preserved for resume when its runtime and calibration contract still matches; changed formatter, dtype, or calibration sample settings invalidate those old ranges.
- **A stale binary index** whose encoding marker or row count disagrees with the embeddings matrix is rebuilt from the persisted int8 rows — no model encoding, and vectors, paper metadata, and hydration state are preserved. A genuinely incompatible index is dropped.

Stored runtime consistency keys are seeded once and then compared on every reopen, and are restamped only after that check passes, the namespace is rebuilt, or the namespace holds no vectors yet. Ambiguous row mappings and file-open, lock, or IO failures propagate without clearing the namespace. Proven incompatible layouts or invalid calibration metadata still reset that namespace, with the specific mismatch included in the warning; a missing HDF5 file clears its stale SQLite mappings.

Embedding cache schema **3** requires one-time re-encoding of older namespaces when they are next opened. Earlier in-place updates could leave old text paired with a replacement vector after a failed commit, and row counts cannot identify those entries. The existing schema mismatch reset therefore rebuilds each old namespace instead of reusing potentially inconsistent vectors.

## Locking

Three locks, all governed by the same timeout (`CITEMESH_EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS`, default 900 seconds):

- the per-namespace `cache_<hash>.lock`, wrapping only the short scan and commit phases;
- the reentrant `hydration_<hash>.lock`, held across the complete-check, resume or rebuild, and the search that consumes hydrated rows;
- the shared cache-root `ReadWriteLock` at `.locks/cache-operations.db`, taken on the read side by every live cache operation and on the write side (non-blocking) by `cache clear`.

Exceeding a timeout raises with the lock path and a pointer at the env var or an isolated `CITEMESH_CACHE_DIR`.
