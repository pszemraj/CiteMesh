# Caching & Data Storage

CiteMesh uses persistent caches to avoid recomputing expensive datasets and embeddings.

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
├── embeddings/
│   ├── metadata_<model-hash>.db   # SQLite metadata (paper ids, text hashes, row_idx, authors/categories JSON, hydration state)
│   ├── embeddings_<model-hash>.h5 # HDF5 matrix datasets (int8/float32 + optional binary index + calibration ranges)
│   └── cache_<model-hash>.lock    # Inter-process lock for cache mutation
└── references/
    └── <sha1>.json                # Semantic Scholar reference ID cache entries
```

`config.toml` is configuration, not cache: it is documented in [User Configuration](configuration.md) and survives `citemesh cache clear`.

Model hashes are the first 12 characters of `sha256(<namespace>)`. Every embedding namespace binds the runtime-active model (including a fallback checkpoint), requested revision, immutable resolved artifact fingerprint, representation role, normalization contract, resolved truncate dimension, storage precision, effective binary-prefilter mode, resolved source torch dtype, and task-formatter fingerprint; `int8` namespaces also include calibration sample size. Candidate mode (`--semantic-source candidates`, the default) adds `mode=candidates` to the retrieval-document namespace so incrementally embedded S2 candidates never mix with corpus hydrations. The graph-similarity namespace is source-mode independent because it contains only selected papers encoded under the same symmetric task contract. Namespaces intentionally carry no device token: matching contracts share a namespace whenever compute dtype also matches, including CUDA/MPS at bf16 or CPU/accelerator runtimes at float32.

## Embedding Cache Behavior

`EmbeddingCache` stores each paper embedding once per namespace. Vectors are kept in a resizable HDF5 matrix, while SQLite tracks metadata and `row_idx` mappings.

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

With built-in CLI settings:

- Candidate mode uses float32 retrieval storage with no binary prefilter.
- arXiv corpus mode uses int8 retrieval storage with the binary prefilter enabled.

Explicit corpus-mode flags can select float32 storage or disable the prefilter. An
int8 retrieval cache may contain:

- `embeddings`: `int8` matrix (`N x dim`)
- `calibration_ranges`: float32 per-dimension min/max (`2 x dim`)
- `binary_index`: packed `uint8` matrix (`N x ceil(dim/8)`) used for Hamming prefiltering

The `binary_index` is an auxiliary retrieval index, not the primary embedding store. Final ranking still uses the cached `int8` or `float32` vectors. For `int8`, calibration ranges must already exist before cache writes begin. Hydration-managed embedding workflows create and persist those ranges before the first int8 cache write; raw `EmbeddingCache` int8 writes now fail closed instead of bootstrapping ranges from an arbitrary request batch. Hydration no longer takes the first-N records for calibration. Instead, it runs a separate representative reservoir-sampling prepass over the active hydration slice and persists ranges before the main cache-write pass begins.

Persistent storage supports only `int8` and `float32` via `--storage-precision`; model runtime compute dtype is configured independently. CLI-managed compression filters are `gzip` and `lzf` (`szip` is intentionally rejected). `lzf` does not support configurable levels; CiteMesh normalizes level to `0`. Runtime availability still depends on your `h5py` build. Compression is a physical HDF5 layout choice, not part of embedding semantics: an existing valid cache keeps its stored codec and level when reopened, while `--cache-compression` and `--cache-compression-level` apply when a cache is first created or explicitly rebuilt.

SQLite stores metadata authority fields used for warm-cache retrieval:

- `title`, `abstract`, `year`
- `authors_json`, `categories_json`
- runtime cache consistency keys (`storage_precision`, source torch dtype, effective embedding vector dtype, text-formatter fingerprint, binary-prefilter mode, physical compression filter/level, and `int8` calibration sample size)
- hydration metadata keys (`dataset source`, `split`, `corpus cap`, completion flag)
- `model_fingerprint` (active model identity guard for namespace reuse)

A vector is recomputed when:

- The paper is missing from cache, or
- The embedding invalidation hash changed (derived from the composed embedding input text).

Metadata-only changes (`year`, `authors`, `categories`, or other stored fields that do not alter embedding input text) refresh SQLite metadata rows without re-encoding vectors.

Cache writes are serialized via per-model lock files (`cache_<model-hash>.lock`) to avoid multi-process HDF5 write races. The expensive encode step runs outside that lock; the lock only wraps short lookup and commit phases, and the commit phase re-checks cache misses before assigning final rows. Lock acquisition timeout defaults to `900` seconds and can be overridden with `CITEMESH_EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS` (details: [Environment Variables](../reference/environment.md)).

Hydration write policy:

- Encoding uses conservative model micro-batches by default (`32`) for runtime stability, configurable via `--encode-batch-size`.
- Cache persistence flushes metadata/embedding appends in larger bursts (`2048` records, matching the HDF5 dataset chunk size) to reduce SQLite/HDF5 lock and resize overhead during long corpus hydration.

For capped hydration, `--corpus-size` limits how many of the newest submissions are embedded and cached, not how many selected-split rows are inspected to determine that ordering. A non-streaming `--dataset-split` slice bounds the rows exposed to CiteMesh, although a cold Hugging Face dataset builder may still prepare its complete underlying Arrow split before applying the slice.

Embedding/hybrid workflows can trigger a namespace rebuild using `--force-rebuild-cache` (see [CLI Usage](cli.md)). The rebuild clears both the retrieval-document and graph-similarity namespaces for the resolved model contract. By default, CiteMesh asks for confirmation before applying this destructive rebuild. Use `--overwrite-cache` to skip the prompt (required for non-interactive scripts). Use `--cache-overwrite-reason "<text>"` to attach a human-readable rationale to rebuild logs and config metadata.

Before persistent cache access, CiteMesh resolves an immutable artifact fingerprint and makes it part of the physical namespace. Hugging Face repositories use the resolved commit SHA when available; an explicitly requested 40-character commit is already immutable and works offline. A standard local Hugging Face snapshot also exposes its commit SHA without an API request. If a cached snapshot does not expose a SHA, CiteMesh hashes its complete inference-relevant artifact manifest. The same manifest policy applies to arbitrary local model paths and covers weights and referenced shards, tokenizer inputs, SentenceTransformers module definitions and numbered module configuration (including pooling), and custom model code. Documentation and training-only files are excluded.

The namespace also records a stable model-profile schema token. Automatic profile selection inspects compatible local SentenceTransformers and transformer metadata, so a local EmbeddingGemma checkpoint uses the same retrieval-query, retrieval-document, STS, truncation, and runtime contract as its Hub alias. Explicit `--model-profile` overrides select a separate matching namespace. The EmbeddingGemma v2 profile token prevents reuse of vectors created before CiteMesh guaranteed a Transformers backend with bidirectional-attention support.

Model loading fallback is resolved before persistent vectors are read or written, so the active checkpoint, not merely the requested model token, selects the namespace. Changing a local artifact in place or moving a mutable Hub revision to new contents selects a different cache while preserving the old one; switching revision A to B and back to A therefore reopens A's prior cache. If no reliable commit or complete local artifact identity can be established, CiteMesh refuses persistent cache access. It never adopts an unidentified legacy payload or an `offline-unverified` assumption. A fingerprint mismatch inside an identified namespace is treated as corruption and cleared before use.

When hydration metadata matches the requested split/corpus cap, records a non-empty dataset source, and points to a queryable embedding+metadata row mapping, embedding retrieval runs fully from cache and skips HuggingFace corpus loading.

For hydrated full-corpus runs (`--all-corpus`), CiteMesh performs an incremental growth check using upstream split row counts. When upstream rows increased, it uses a staged reconciliation flow:

`--all-corpus` means "the full selected `--dataset-split`". For example, `--dataset-split train --all-corpus` hydrates the full `train` split; it does not merge `train`, `validation`, and `test` into one cache namespace.

- tail delta slice (`cached_rows:upstream_rows`)
- head delta slice (`0:delta_rows`) if tail under-fills
- full-split missing-ID reconciliation only when needed

All reconciliation steps are ID-aware and append only uncached paper IDs. Before an incremental growth refresh writes data, CiteMesh marks hydration incomplete; a source or write failure propagates while preserving completed rows for the next resume instead of serving the partial refresh or clearing it. If a prior full-corpus hydration was interrupted but the cached SQLite/HDF5 row counts still match each other and the hydration metadata still matches the requested source/split, CiteMesh resumes from `cached_rows` instead of clearing the namespace and starting from zero again. If upstream split row counts shrink below cached payload size, CiteMesh marks the namespace hydration state incomplete and forces full source revalidation instead of serving stale over-cap rows from the prior cache snapshot. If full reconciliation confirms no uncached IDs while row-count delta remains, CiteMesh treats that as duplicate-ID upstream growth (not a cache failure), records the reconciled row-count state, and skips repeated full-split scans until row counts change again.

When switching a namespace from a capped corpus (for example `--corpus-size 50000`) to `--all-corpus`, cache-clear logs report both the requested target and the replaced cached payload. Seeing `requested_corpus=all` alongside `cached_corpus=50000` means CiteMesh is replacing the old capped namespace before hydrating the full split; it does not mean the new run is silently limited to `50000`.

When int8 calibration clipping is detected during a hydration run, CiteMesh emits
that warning once per cache instance/run and keeps accumulating the underlying
saturation stats in cache metadata instead of repeating the same warning every
flush window.

Current limitation: hydration compatibility is keyed to dataset source/split/corpus metadata, not an immutable upstream dataset revision fingerprint. If a dataset alias mutates upstream without changing source name, treat cache reuse as a performance optimization rather than a strict reproducibility guarantee.

During cache-native search, scored embedding rows must map to metadata rows. Missing metadata row mappings now fail closed with an integrity error instead of returning partial top-k results.

If cache payload files become inconsistent (for example missing matrix file, incompatible layout, or invalid calibration metadata), CiteMesh resets that namespace state and rebuilds on the next hydration run.

## Semantic Scholar Reference Cache

When reference expansion is enabled, reference-ID lookups are cached under `references/` using hashed filenames.

- Default policy is no TTL: version-matched cache entries are reused until manually cleared or refreshed.
- `--refresh-reference-cache` bypasses persisted reference-cache reads and fetches fresh reference IDs from the API (write-through cache update).
- Successful empty reference responses are cached as explicit empty lists to avoid repeated API calls for papers with no references.
- Empty cached reference hits are reused silently; debug logging emits cache-hit lines only for non-empty reference lists so long runs do not spam one zero-count line per paper.
- Non-empty cached payloads that contain no valid reference IDs are treated as invalid and rebuilt from API data instead of being reused as implicit empties.
- Corrupt/unreadable JSON cache entries (including non-object payloads) are treated as invalid and rebuilt from API data.
- Repeated reference-fetch failures now raise a runtime error after retries instead of silently returning an empty list.
- Reference cache directory resolution occurs at call time, so cache-root policy (`CITEMESH_CACHE_DIR`) changes are honored for new lookups.

## HuggingFace Default Cache

HuggingFace libraries keep their own cache (commonly `~/.cache/huggingface`). CiteMesh does not override that location.

## Cleaning and Inspecting Cache

Inspect cache usage:

```bash
citemesh cache scan
```

Clear cached data under the CiteMesh cache root:

```bash
citemesh cache clear --yes --reason "manual local reset"
```

Omit `--yes` for interactive confirmation. `cache clear` deletes cache payloads (embeddings, references) but always preserves `config.toml`.

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
