# Embedding Cache Internals

`EmbeddingCache` coordinates SQLite metadata and HDF5 vectors so interrupted writes can be recovered. For normal operation, see [cache reuse and hydration](../guides/caching.md).

## Store composition

`EmbeddingCache` (`store.py`) is one class assembled from four mixins, in MRO order:

```python
class EmbeddingCache(_IngestMixin, _H5LayoutMixin, _RecoveryMixin, _SearchMixin):
```

- `ingest.py` (`_IngestMixin`) — the three-phase write pipeline: `_process_embeddings_locked` scans for hits under the lock, encodes **outside** it, then commits under it again (`_scan_cache_for_hits`, `_encode_pending_texts`, `_commit_encoded_embeddings`, `_apply_replacement_rows`, `_append_new_rows`).
- `layout.py` (`_H5LayoutMixin`) — SQLite schema creation and migration (`_init_db`), cache-metadata get/set, the runtime-contract stamp and check (`_assert_runtime_cache_consistency`), HDF5 dataset creation and validation, and the open-time repair pass `_ensure_h5_layout`.
- `recovery.py` (`_RecoveryMixin`) — locking (`_cache_lock`, `_cache_operation_lock`, `hydration_operation_lock`), the SQLite connection context, the replacement journal and its replay, trailing-row truncation (`_recover_trailing_rows`), and the durability barrier `_flush_h5_file`.
- `search.py` (`_SearchMixin`) — query-time scoring: the calibration gate, the Hamming prefilter, the chunked scoring kernels (`_score_int8_rows`, `_score_float_rows`, `_select_top_k`), the finiteness guard, and the row-to-metadata join.

Supporting modules: `constants.py` (metadata keys, schema version, chunk sizes, lock timeout), `sql.py` (DDL and upserts), `models.py` (result dataclasses), `quantization.py` (int8 and binary packing).

## Physical layout

All namespaces share one `embeddings/` directory, separated by a filename suffix rather than a subdirectory: `sha256(model_name)[:12]` gives `metadata_<hash>.db`, `embeddings_<hash>.h5`, `cache_<hash>.lock`, and `hydration_<hash>.lock`. SQLite holds three tables — `papers` (keyed by `paper_id`, carrying `text_hash`, `row_idx`, the metadata fields, and a nullable `chronology_key`: the packed arXiv submission date, kept off the row-read path because only the aggregate recency watermark reads it), `cache_metadata` (key/value), and `replacement_journal` (the undo log below). `papers` carries two indexes, `idx_papers_row_idx` and `idx_papers_chronology_key`. HDF5 holds `embeddings` (resizable `N x dim`, chunked at 2048 rows), the optional `binary_index` (`N x ceil(dim/8)` `uint8`, tagged `int8-midpoint-sign-v1`), and for int8 namespaces the fixed `calibration_ranges` (`2 x dim` float32). The runtime contract lives in HDF5 root attrs and is re-asserted on every open.

`sqlite3` runs in its default rollback-journal mode, with no WAL or `synchronous` pragma; durability comes from the flush ordering below, not from SQLite settings.

## Write ordering and the replacement journal

Every write flushes exactly once, at the end of the second locked phase: **HDF5 data is flushed and fsynced before the SQLite rows that point at it are committed**, so a durable mapping never outlives the vector it references. The opposite order would leave committed mappings recovery could only fix by discarding good rows.

Existing-row replacements use a durable SQLite undo journal under the namespace lock: before overwriting a vector, CiteMesh commits the previous vector and binary-index row to `replacement_journal`, and deletes those entries in the same transaction that commits the new paper metadata. If that commit fails or the process stops, the next access restores the journaled rows and removes uncommitted trailing appends before reading vectors. Failed restoration stops access and keeps the journal for another attempt.

During cache-native search, a scored row with no metadata mapping fails closed with an integrity error rather than returning partial top-k results.

## What a crash leaves, and how it is repaired

- **Orphan trailing HDF5 rows** — an append that resized and wrote the matrix but never committed its SQLite mappings. `_recover_trailing_rows` truncates matrix and binary index back to the contiguous prefix shared by SQLite and HDF5 and marks hydration incomplete; the inverse case, mappings whose vectors were lost, deletes those mappings and does the same. Truncation happens only when the committed SQLite prefix is dense (`0..n-1`); an ambiguous mapping is left alone and surfaces as an integrity error rather than being silently "repaired".
- **A half-applied replacement** — the journal replay above. Row bounds, embedding width, and payload byte size are validated before a restore; a mismatch raises rather than guessing, and existing files are preserved.
- **An interrupted first encode** — a file with no datasets and no schema attributes. That is an *empty* namespace, not a proven mismatch: reopening discards mappings whose vectors were never persisted, marks hydration incomplete, keeps the source/split/corpus-cap and corpus-metadata markers, and writes into it without a wipe warning.
- **A calibration-only file** written before the first embedding write survives for resume while its runtime and calibration contract matches; a changed formatter, dtype, or calibration-sample setting invalidates those ranges.
- **A stale binary index** whose encoding marker or row count disagrees with the embeddings matrix is rebuilt from the persisted int8 rows - no model encoding, no loss of vectors, metadata, or hydration state. A genuinely incompatible index is dropped.

Runtime consistency keys are seeded once, compared on every reopen, and restamped only after that check passes, the namespace is rebuilt, or it holds no vectors. Ambiguous mappings and file-open, lock, or IO failures propagate without clearing the namespace; proven incompatible layouts or invalid calibration metadata do reset it, with the mismatch named in the warning, and a missing HDF5 file clears its stale SQLite mappings.

Schema **4** requires one-time re-encoding of older namespaces when they are next opened: rows written before it carry no `chronology_key`, leaving a capped corpus with no trustworthy recency watermark and a growth check that would silently decline. Row counts cannot identify those rows, so the schema-mismatch reset rebuilds each old namespace instead of reusing vectors it cannot vouch for.

## Locking

Three locks use the [embedding-cache timeout](../reference/environment.md#citemesh-variables):

- the per-namespace `cache_<hash>.lock`, wrapping only the short scan and commit phases
- the reentrant `hydration_<hash>.lock`, held across the complete-check, resume or rebuild, and the search that consumes hydrated rows
- the shared cache-root `ReadWriteLock` at `.locks/cache-operations.db`, taken on the read side by every live cache operation and on the write side (non-blocking) by `cache clear`

Exceeding a timeout raises with the lock path and a pointer at the env var or an isolated `CITEMESH_CACHE_DIR`.
