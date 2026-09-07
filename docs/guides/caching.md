# Caching & Data Storage

CiteMesh uses persistent caches to reuse paper metadata, reference IDs, datasets, and embeddings.

Related docs:

- CLI command usage: [CLI Usage](cli.md)
- Persistent user defaults: [User Configuration](configuration.md)
- Environment variables: [Environment Variables](../reference/environment.md)
- Embedding runtime policy: [Embedding Runtime](../reference/embedding-runtime.md)

## Cache Root

By default, project caches are stored under:

- **Linux and macOS**: `${XDG_CACHE_HOME:-~/.cache}/citemesh`
- **Windows**: `%LOCALAPPDATA%\\CiteMesh` (or `%APPDATA%\\CiteMesh` if `LOCALAPPDATA` is unset)

macOS previously used `~/Library/Caches/citemesh`; the root is now unified with Linux (HuggingFace-style `~/.cache/citemesh`) so cache paths and `config.toml` are predictable across machines. `citemesh cache scan` prints a migration hint if the legacy macOS directory still exists - move or delete it to reclaim space.

Override the root with:

```bash
export CITEMESH_CACHE_DIR=/path/to/custom/cache
```

Variable details are documented in [Environment Variables](../reference/environment.md).

## Directory Layout

```text
citemesh cache root
├── config.toml                    # Persistent user configuration (see Configuration guide)
├── config.toml.lock               # Coordination for configuration writes and cache clearing
├── embeddings/
│   ├── metadata_<model-hash>.db   # SQLite metadata (paper ids, text hashes, row_idx, authors/categories JSON, hydration state)
│   ├── embeddings_<model-hash>.h5 # HDF5 matrix datasets (int8/float32 + optional binary index + calibration ranges)
│   ├── cache_<model-hash>.lock    # Inter-process lock for cache mutation
│   └── hydration_<model-hash>.lock # Corpus hydration and consuming-search lock
├── papers/
│   └── <sha1>.json                # Semantic Scholar paper metadata by requested ID and known aliases
└── references/
    └── <sha1>.json                # Semantic Scholar reference ID cache entries
```

`config.toml` is configuration, not cache: it is documented in [User Configuration](configuration.md) and survives `citemesh cache clear`. Atomic text writes preserve an existing file's permission bits (new files follow the process umask); `config.toml` is the exception and is always written `0600` because it can hold `api.s2_api_key`.

Model hashes are the first 12 characters of `sha256(<namespace>)`. Every embedding namespace binds the runtime-active model (including a fallback checkpoint), requested revision, immutable resolved artifact fingerprint, representation role, normalization contract, resolved truncate dimension, storage precision, effective binary-prefilter mode, resolved source torch dtype, and task-formatter fingerprint; `int8` namespaces also include calibration sample size. Candidate mode (`--semantic-source candidates`, the default) adds `mode=candidates` to the retrieval-document namespace so incrementally embedded S2 candidates never mix with corpus hydrations. The graph-similarity namespace is source-mode independent because it contains only selected papers encoded under the same symmetric task contract. Namespaces intentionally carry no device token: CPU, CUDA, and MPS share a namespace whenever compute dtype and the other contracts match, whether bf16 or float32.

EmbeddingGemma now defaults to 512 dimensions. Because the resolved dimension is
part of both retrieval and graph-similarity cache identities, default runs use
separate namespaces from older 256-dimensional runs. Existing 256-dimensional
caches are preserved; `--truncate-dim 256` selects them when the other cache
settings match. A configured `defaults.truncate_dim = 256` also keeps that choice
for builds and local search; unset it to adopt the profile default. The change
does not clear the independent paper-metadata or reference caches.

## Paper Metadata Cache

Successful `get_paper` and `get_papers` lookups persist metadata under `papers/`,
independently of the embedding model and encoding batch size. Lookups check this
cache before contacting Semantic Scholar, including seed resolution at the start
of a rerun. Batch requests send only missing IDs. Entries are shared across the
requested identifier and known Semantic Scholar, arXiv, and DOI aliases.

A null entry in a successful batch response means that requested paper is absent;
CiteMesh skips an individual follow-up lookup and does not cache the miss.

Metadata has no TTL and is reused until manually refreshed or cleared. Failed, missing, and
malformed API responses are not cached. Reference IDs remain in their separate
cache and are loaded or fetched when requested; cached metadata alone does not
mean references have been fetched. Graph-specific seed flags are not persisted.

Runs made before this cache was introduced require one successful paper lookup
to populate it. Citation/reference paper lists, recommendation results, and
keyword search results still require live requests; cached seed metadata does
not make an entire graph build offline. Use `build --refresh-paper-cache` to
bypass persisted paper-metadata reads for that run. Successful fresh
responses replace cached metadata, including citation counts; failures preserve
the previous cache entries. Refreshing metadata leaves embedding caches intact
and does not by itself re-encode unchanged paper text. Reference IDs keep their
separate `--refresh-reference-cache` control.

## Embedding Cache Behavior

`EmbeddingCache` stores each paper embedding once per namespace. Vectors are kept in a resizable HDF5 matrix, while SQLite tracks metadata and `row_idx` mappings.

### What you need to know

- Embedding vectors persist across runs in two independent namespaces per model contract: a retrieval-document cache (candidate or corpus papers, what local `citemesh search` reads) and a graph-similarity cache (selected graph papers only).
- Default storage precision is float32 in candidate mode and int8 in arXiv-corpus mode. Changing model, revision, profile, dimension, or precision selects a different namespace rather than invalidating the old one.
- A cached vector is re-encoded only when the paper is new to the namespace or its embedding input text changed; metadata-only updates never trigger re-encoding.
- `--force-rebuild-cache` (plus `--overwrite-cache` in scripts) is the supported way to re-encode a namespace, for example after persistent calibration-clipping warnings.

The rest of this section documents the exact storage contracts; routine use does not require it.

### Storage layout and behavior

Embedding and hybrid builds use two physical cache roles. Prompt routing and
formatter behavior are described in [Embedding Runtime](../reference/embedding-runtime.md):

- The retrieval-document cache stores candidate or corpus papers for seed-to-paper
  ranking. Query vectors are transient and are not persisted as documents.
- The graph-similarity cache stores only selected graph papers. It is always
  float32 with no binary prefilter and is the sole vector source for
  paper-to-paper edges.

The roles have different representation and formatter identities, so dimensions
alone cannot make their vectors interchangeable. Local semantic search reads only
retrieval-document caches.

Local-search discovery checks for existing vector files without creating a
provisional namespace. If files exist, it resolves the active model before
opening its cache; an empty cache root stays empty when search falls back to
Semantic Scholar.

With built-in CLI settings:

- Candidate mode uses float32 retrieval storage with no binary prefilter.
- arXiv corpus mode uses int8 retrieval storage with the binary prefilter enabled.

Explicit corpus-mode flags can select float32 storage or disable the prefilter. An
int8 retrieval cache may contain:

- `embeddings`: `int8` matrix (`N x dim`)
- `calibration_ranges`: float32 per-dimension min/max (`2 x dim`)
- `binary_index`: packed `uint8` matrix (`N x ceil(dim/8)`) used for Hamming prefiltering

The `binary_index` is an auxiliary retrieval index, not the primary embedding store. Final ranking still uses the cached `int8` or `float32` vectors. For `int8`, calibration ranges must already exist before cache writes begin. Hydration-managed embedding workflows create and persist those ranges before the first int8 cache write; raw `EmbeddingCache` int8 writes now fail closed instead of bootstrapping ranges from an arbitrary request batch. Hydration no longer takes the first-N records for calibration. Instead, it runs a separate representative reservoir-sampling prepass over the active hydration slice and persists ranges before the main cache-write pass begins. When a capped streaming selection has already materialized its rows, the prepass samples those rows directly instead of re-reading the source.

New ranges use each dimension's minimum and maximum over that sample, following
the [Sentence Transformers scalar quantization guidance](https://www.sbert.net/examples/sentence_transformer/applications/embedding-quantization/README.html#scalar-int8-quantization).
They cover the entire calibration sample, though later embeddings can still fall
outside the sampled bounds. Resumes reuse the persisted ranges, including ranges
created by the earlier 0.1–99.9 percentile policy; updating CiteMesh does not clear
or recalibrate an existing namespace. Ranges cannot be replaced once int8 rows
exist, because those same ranges are required to decode the stored bytes.
The quantizer floors into 256 unsigned buckets, clips out-of-range values, and
then applies the signed int8 offset. Decoding reconstructs each bucket's centre,
clamped at the calibration maximum, to avoid the systematic low bias of decoding
its lower boundary. Existing rows remain readable without rewriting files;
older rows written with rounding now receive the same approximate midpoint
decoding, so their reconstructed values and similarity scores can change.
The binary prefilter is derived from decoded persisted vectors on both writes
and index rebuilds, keeping their sign-bit representation consistent. Existing
indexes without the current midpoint-sign encoding marker are rebuilt once when
the namespace is reopened. That auxiliary rebuild reads persisted int8 rows;
it preserves vectors, paper metadata, and hydration state without model encoding.
Opening a namespace with its binary prefilter disabled drops the auxiliary
index. Re-enabling it rebuilds from persisted vectors, so updates made while
disabled cannot leave stale sign bits in later searches.
Newly encoded `int8` vectors are returned only after that same store-and-decode round trip, so a paper's returned vector is identical on the run that encodes it and on every later cache hit; `float32` namespaces return the model output unchanged.

Persistent storage supports only `int8` and `float32` via `--storage-precision`; model runtime compute dtype is configured independently. CLI-managed compression filters are `gzip` and `lzf` (`szip` is intentionally rejected). `lzf` does not support configurable levels; CiteMesh normalizes level to `0`. Runtime availability still depends on your `h5py` build. Compression is a physical HDF5 layout choice, not part of embedding semantics: an existing valid cache keeps its stored codec and level when reopened, while `--cache-compression` and `--cache-compression-level` apply when a cache is first created or explicitly rebuilt.

SQLite stores metadata authority fields used for warm-cache retrieval:

- `title`, `abstract`, `year`, `venue`, `arxiv_id`, `doi`
- `authors_json`, `categories_json`
- runtime cache consistency keys (`storage_precision`, source torch dtype, effective embedding vector dtype, text-formatter fingerprint, binary-prefilter mode, physical compression filter/level, and `int8` calibration sample size)
- hydration metadata keys (`dataset source`, `split`, `corpus cap`, completion flag)
- `model_fingerprint` (active model identity guard for namespace reuse)

A vector is recomputed when:

- The paper is missing from cache, or
- The embedding invalidation hash changed (derived from the composed embedding input text).

Metadata-only changes (`year`, `authors`, `categories`, or other stored fields that do not alter embedding input text) refresh SQLite metadata rows without re-encoding vectors.

Corpus metadata retains the source DOI alongside its arXiv ID. Years come from an explicit publication year when available, otherwise the initial submission year encoded in the arXiv ID; `update_date` is never used as a publication date.

On the next corpus build, caches created before this adapter correction receive a one-time year/DOI backfill from their recorded source and selected split. The pass updates only existing SQLite rows: it preserves the original corpus selection, stored text, and HDF5 vectors, including INT8 calibration. For legacy caches, the `corpus_metadata_version` marker is saved only after the source pass succeeds, so interrupted backfills retry on the next build. Fresh namespaces receive this marker before hydration starts because every row uses the current adapter; interrupted hydration therefore resumes without a redundant metadata backfill. Source fields containing multiple DOIs retain the first whitespace-, comma-, or semicolon-separated DOI. Local search remains offline and uses persisted metadata; run a corpus build to apply the backfill before searching an older cache.

Rewriting compressed HDF5 rows can leave unused space in the file, so its byte
size may grow while the embedding row count stays fixed. CiteMesh does not
compact files automatically; this allocation behavior does not indicate lost
vectors or duplicate cache rows.

Cache writes are serialized via per-model lock files (`cache_<model-hash>.lock`) to avoid multi-process HDF5 write races. The expensive encode step runs outside that lock; the lock only wraps short lookup and commit phases, and the commit phase re-checks cache misses before assigning final rows. Lock acquisition timeout defaults to `900` seconds and can be overridden with `CITEMESH_EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS` (details: [Environment Variables](../reference/environment.md)).

Corpus hydration also holds `hydration_<model-hash>.lock` across the complete
check, resume or rebuild, and the cache search that consumes the hydrated rows.
Concurrent builds targeting different source, split, or corpus-size settings
therefore finish one at a time instead of mixing rows under one hydration marker.
Explicit corpus-cache rebuilds wait for the same operation lock. The operation
lock also makes local corpus searches wait for an ongoing hydration to finish. It
uses the same timeout setting while the short mutation lock continues to protect
each SQLite/HDF5 commit.

Opening a cache performs exhaustive row-coverage validation and interrupted-append
recovery. Normal lookups, writes, and searches check runtime settings and indexed
row bounds without recounting the entire corpus. Search also requires metadata
for every returned vector, and accessed rows must have a unique paper mapping.
Interior mapping corruption outside the accessed rows
is detected by the full validation on the next cache opening.

Hydration write policy:

- Encoding uses conservative model micro-batches by default (`32`) for runtime stability, configurable via `--encode-batch-size`.
- Cache persistence flushes metadata/embedding appends in larger bursts (`2048` records, matching the HDF5 dataset chunk size) to reduce SQLite/HDF5 lock and resize overhead during long corpus hydration.

For capped hydration, `--corpus-size` limits how many of the newest submissions are embedded and cached, not how many selected-split rows are inspected to determine that ordering. In streaming mode CiteMesh warns that newest-first selection must drain the whole stream before hydration starts; `--no-streaming` or an explicit `--dataset-split` slice avoids that pass. If too few records have parseable submission IDs, CiteMesh fills the remaining budget from records with unparseable IDs in source order and emits a warning. A non-streaming `--dataset-split` slice bounds the rows exposed to CiteMesh, although a cold Hugging Face dataset builder may still prepare its complete underlying Arrow split before applying the slice.

Previously completed capped caches are reused as recorded. To replace an older
cache that under-filled because some IDs were unparseable, use the explicit
rebuild workflow below.

Embedding/hybrid workflows can trigger a namespace rebuild using `--force-rebuild-cache` (see [CLI Usage](cli.md)). The rebuild clears both the retrieval-document and graph-similarity namespaces for the resolved model contract. By default, CiteMesh asks for confirmation before applying this destructive rebuild. Use `--overwrite-cache` to skip the prompt (required for non-interactive scripts). Use `--cache-overwrite-reason "<text>"` to attach a human-readable rationale to rebuild logs and config metadata; namespace rebuild/clear logs record `reason=unspecified` when no rationale is supplied.

Before persistent cache access, CiteMesh resolves an immutable artifact fingerprint and makes it part of the physical namespace. Hugging Face repositories use the resolved commit SHA when available; an explicitly requested 40-character commit is already immutable and works offline. A standard local Hugging Face snapshot also exposes its commit SHA without an API request. If a cached snapshot does not expose a SHA, CiteMesh hashes its complete inference-relevant artifact manifest. The same manifest policy applies to arbitrary local model paths and covers weights and referenced shards, tokenizer inputs, SentenceTransformers module definitions and numbered module configuration (including pooling), and custom model code. Documentation and training-only files are excluded.

The namespace also records a stable model-profile schema token. Automatic profile selection inspects compatible local SentenceTransformers and transformer metadata, so a local EmbeddingGemma checkpoint uses the same retrieval-query, retrieval-document, STS, truncation, and runtime contract as its Hub alias. Explicit `--model-profile` overrides select a separate matching namespace. The EmbeddingGemma v2 profile token prevents reuse of vectors created before CiteMesh guaranteed a Transformers backend with bidirectional-attention support.

Model loading fallback is resolved before persistent vectors are read or written, so the active checkpoint, not merely the requested model token, selects the namespace. Changing a local artifact in place or moving a mutable Hub revision to new contents selects a different cache while preserving the old one; switching revision A to B and back to A therefore reopens A's prior cache. If no reliable commit or complete local artifact identity can be established, CiteMesh refuses persistent cache access. It never adopts an unidentified legacy payload or an `offline-unverified` assumption. A fingerprint mismatch inside an identified namespace is treated as corruption and cleared before use.

When hydration metadata matches the requested split/corpus cap, records a non-empty dataset source, and points to a queryable embedding+metadata row mapping, embedding retrieval runs from cache and skips HuggingFace corpus loading after any required one-time metadata backfill.

Hydration and resume inspect cache state strictly. A failed filesystem, SQLite,
or HDF5 inspection
stops the operation with the cache paths and original error; it does not count
as a metadata mismatch or an empty cache, and existing files are preserved.
Retry after resolving the storage error. Best-effort counts are used only to
report the impact of a clear that has already been requested.

For hydrated full-corpus runs (`--all-corpus`), CiteMesh performs an incremental growth check using upstream split row counts. When upstream rows increased, it uses a staged reconciliation flow:

`--all-corpus` means "the full selected `--dataset-split`". For example, `--dataset-split train --all-corpus` hydrates the full `train` split; it does not merge `train`, `validation`, and `test` into one cache namespace.

- tail delta slice (`cached_rows:upstream_rows`)
- head delta slice (`0:delta_rows`) if tail under-fills
- full-split missing-ID reconciliation only when needed

All reconciliation steps are ID-aware and append only uncached paper IDs. Before an incremental growth refresh writes data, CiteMesh marks hydration incomplete; a source or write failure propagates while preserving completed rows for the next resume instead of serving the partial refresh or clearing it. If a prior full-corpus hydration was interrupted but the cached SQLite/HDF5 row counts still match each other and the hydration metadata still matches the requested source/split, CiteMesh first resumes from `cached_rows` instead of clearing the namespace and starting from zero again. If the tail leaves a known row-count shortfall or the upstream row count is unavailable, resume scans the full split for missing IDs before marking hydration complete or memoizing a duplicate-ID deficit; exhausting a tail slice does not prove full source coverage. If upstream split row counts shrink below cached payload size, CiteMesh marks the namespace hydration state incomplete and forces full source revalidation instead of serving stale over-cap rows from the prior cache snapshot. If full reconciliation confirms no uncached IDs while row-count delta remains, CiteMesh treats that as duplicate-ID upstream growth (not a cache failure), records the reconciled row-count state, and skips repeated full-split scans until row counts change again.

When switching a namespace from a capped corpus (for example `--corpus-size 50000`) to `--all-corpus`, cache-clear logs report both the requested target and the replaced cached payload. Seeing `requested_corpus=all` alongside `cached_corpus=newest:50000` means CiteMesh is replacing the old capped namespace before hydrating the full split; it does not mean the new run is silently limited to `50000`.

When at least 0.5% of a write's embedding coordinates fall outside the calibration
ranges, CiteMesh warns once per cache instance/run and continues accumulating
clipping statistics in cache metadata. This is a coordinate clipping rate, not a
measurement of lost retrieval recall or a cache corruption error. A warning alone
does not require stopping or rebuilding the run. If retrieval quality warrants
new calibration, use `--force-rebuild-cache` to re-encode the namespace, optionally
with a larger `--calibration-sample-size`; replacing only the ranges would
reinterpret existing rows incorrectly. Increasing the sample size changes the
namespace and starts a separate cache rather than resuming existing rows.

Current limitation: hydration compatibility is keyed to dataset source/split/corpus metadata, not an immutable upstream dataset revision fingerprint. If a dataset alias mutates upstream without changing source name, treat cache reuse as a performance optimization rather than a strict reproducibility guarantee.

### Durability and recovery internals

You do not need this section to use CiteMesh; it documents crash-safety invariants.

During cache-native search, scored embedding rows must map to metadata rows. Missing metadata row mappings now fail closed with an integrity error instead of returning partial top-k results.

Existing-row replacements use a durable SQLite undo journal under the namespace lock. Before overwriting a vector, CiteMesh commits its previous stored vector and binary-index row to the journal. Every write that maps rows, appends included, flushes and synchronizes its HDF5 data before atomically committing the new paper metadata and removing any journal entries, so a durable row mapping never outlives the vector it points at. If that commit fails or the process stops, the next access restores the journaled rows and removes uncommitted trailing appends before reading vectors or checking their mappings. Failed restoration stops access and retains the journal for another attempt.

Embedding cache schema **3** requires one-time re-encoding of older namespaces
when they are next opened. Earlier in-place updates could leave old text paired
with a replacement vector after a failed commit; row counts cannot identify
those entries. The existing schema mismatch reset therefore rebuilds each old
namespace instead of reusing potentially inconsistent vectors.

An interrupted append preserves the contiguous row prefix shared by SQLite and HDF5. Extra HDF5 rows are truncated; extra SQLite mappings whose vectors were lost are removed and hydration is marked incomplete. A calibration-only file before the first embedding write is preserved for resume when its runtime and calibration contract still matches; changed formatter, dtype, or calibration sample settings invalidate those old ranges. A file holding no datasets and no schema attributes, which is what an interrupted first encode leaves behind, is an empty namespace rather than a proven mismatch: reopening discards any row mappings whose vectors were never persisted, marks hydration incomplete, keeps the dataset-source/split/corpus-cap and corpus-metadata markers, and writes into it without an incompatible-schema wipe warning. Stored runtime consistency keys are seeded once and then compared on every reopen, and are restamped only after that check passes, the namespace is rebuilt, or the namespace holds no vectors yet. Ambiguous row mappings and file-open, lock, or IO failures propagate without clearing the namespace. Proven incompatible layouts or invalid calibration metadata still reset that namespace, with the specific mismatch included in the warning; a missing HDF5 file clears its stale SQLite mappings.

## Semantic Scholar Reference Cache

When reference expansion is enabled, reference-ID lookups are cached under `references/` using hashed filenames.

- Default policy is no TTL: version-matched cache entries are reused until manually cleared or refreshed.
- `--refresh-reference-cache` bypasses persisted reference-cache reads and fetches fresh reference IDs from the API (write-through cache update).
- Successful empty reference responses are cached as explicit empty lists to avoid repeated API calls for papers with no references. A `paper not found` response is likewise cached as an empty reference list (unlike the paper-metadata cache, which does not cache misses).
- Successful reference pages containing only unresolved `paperId: null` records warn and cache an empty list.
- Empty cached reference hits are reused silently; debug logging emits cache-hit lines only for non-empty reference lists so long runs do not spam one zero-count line per paper.
- Non-empty cached payloads that contain no valid reference IDs are treated as invalid and rebuilt from API data instead of being reused as implicit empties.
- Corrupt/unreadable JSON cache entries (including non-object payloads) are treated as invalid and rebuilt from API data.
- Repeated reference-fetch failures now raise a runtime error after retries instead of silently returning an empty list.
- Reference cache directory resolution occurs at call time, so cache-root policy (`CITEMESH_CACHE_DIR`) changes are honored for new lookups.

## HuggingFace Default Cache

HuggingFace libraries keep their own cache (commonly `~/.cache/huggingface`, relocatable via `HF_HOME`). CiteMesh does not override that location; model checkpoints and corpus datasets live there, not under the CiteMesh cache root.

## Cleaning and Inspecting Cache

Inspect cache usage:

```bash
citemesh cache scan
```

Clear cached data under the CiteMesh cache root:

```bash
citemesh cache clear --yes --reason "manual local reset"
```

Omit `--yes` for interactive confirmation. `cache clear` deletes every cache-root entry (embeddings, papers, references, and anything else present) except `config.toml` and its coordination lock, `config.toml.lock`. Clearing acquires the same lock as configuration writes, waiting up to 10 seconds before failing if another process holds it. The cache root remains available for locking even when no configuration file exists.

For command syntax and defaults, see [CLI Usage](cli.md); this section focuses on cache maintenance workflows.

To remove artifacts for one namespace, delete matching `.db` and `.h5` files in `embeddings/`.

Manual full reset examples (note: unlike `citemesh cache clear`, these also delete `config.toml`):

```bash
# Linux / macOS
rm -rf "${XDG_CACHE_HOME:-$HOME/.cache}/citemesh"

# macOS legacy pre-unification root (if it still exists)
rm -rf "$HOME/Library/Caches/citemesh"

# custom root
rm -rf "$CITEMESH_CACHE_DIR"
```

On Windows, remove the cache directory via Explorer or PowerShell.

## Best Practices

- Keep cache data on fast local storage (SSD) for embedding-heavy runs.
- Periodically prune unused model caches if disk usage grows.
- In shared environments, set `CITEMESH_CACHE_DIR` per user or per project.
