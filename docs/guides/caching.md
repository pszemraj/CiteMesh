# Caching & Data Storage

CiteMesh uses persistent caches to avoid recomputing expensive datasets and embeddings.

Related docs:

- CLI command usage: [CLI Usage](cli.md)
- Environment variables: [Environment Variables](../reference/environment.md)
- Embedding runtime policy: [Embedding Runtime](../reference/embedding-runtime.md)
- Docs index: [Documentation](../README.md)

## Cache Root

By default, project caches are stored under:

- **Linux**: `${XDG_CACHE_HOME:-~/.cache}/citemesh`
- **macOS**: `~/Library/Caches/citemesh`
- **Windows**: `%LOCALAPPDATA%\\CiteMesh` (or `%APPDATA%\\CiteMesh` if `LOCALAPPDATA` is unset)

Override the root with:

```bash
export CITEMESH_CACHE_DIR=/path/to/custom/cache
```

Variable details are documented in [Environment Variables](../reference/environment.md).

## Directory Layout

```text
citemesh cache root
├── embeddings/
│   ├── metadata_<model-hash>.db   # SQLite metadata (paper ids, text hashes, row_idx, authors/categories JSON, hydration state)
│   ├── embeddings_<model-hash>.h5 # Quantized HDF5 matrix datasets (int8/f16/f32 + optional binary index + calibration ranges)
│   └── cache_<model-hash>.lock    # Inter-process lock for cache mutation
└── references/
    └── <sha1>.json                # Semantic Scholar reference ID cache entries
```

Model hashes are the first 12 characters of `sha256(<namespace>)`. The embedding namespace string includes model + resolved truncate dim + storage precision + effective binary prefilter mode + resolved source torch dtype + document-formatter fingerprint, and adds calibration sample size in `int8` mode.

## Embedding Cache Behavior

`EmbeddingCache` stores each paper embedding once per namespace. Vectors are kept in a resizable HDF5 matrix, while SQLite tracks metadata and `row_idx` mappings.

Default storage mode is quantized:

- `embeddings`: `int8` matrix (`N x dim`)
- `calibration_ranges`: float32 per-dimension min/max (`2 x dim`)
- `binary_index`: packed `uint8` matrix (`N x ceil(dim/8)`) used for Hamming prefiltering

Non-int8 modes (`float16`, `float32`) are supported via `--storage-precision`.
CLI-managed compression filters are `gzip` and `lzf` (`szip` is intentionally rejected).
Runtime availability still depends on your `h5py` build.

SQLite stores metadata authority fields used for warm-cache retrieval:

- `title`, `abstract`, `year`
- `authors_json`, `categories_json`
- runtime cache consistency keys (`storage_precision`, source torch dtype, effective embedding vector dtype, text-formatter fingerprint, binary-prefilter mode, and `int8` calibration sample size)
- hydration metadata keys (`dataset source`, `split`, `corpus cap`, completion flag)
- `model_fingerprint` (active model identity guard for namespace reuse)

A vector is recomputed when:

- The paper is missing from cache, or
- The embedding invalidation hash changed (derived from the composed embedding input text).

Metadata-only changes (`year`, `authors`, `categories`, or other stored fields that do
not alter embedding input text) refresh SQLite metadata rows without re-encoding vectors.

Cache writes are serialized via per-model lock files (`cache_<model-hash>.lock`) to avoid multi-process HDF5 write races.
Lock acquisition timeout defaults to `60` seconds and can be overridden with
`CITEMESH_EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS` (details: [Environment Variables](../reference/environment.md)).

Embedding/hybrid workflows can trigger a namespace rebuild using `--force-rebuild-cache` (see [CLI Usage](cli.md)).

For Hugging Face repo IDs, hydration resolves and stores a model fingerprint.
CiteMesh first attempts commit-SHA resolution (online API, then local snapshot SHA).
If SHA resolution is unavailable, CiteMesh falls back to hashing two local artifact
files when present: `config.json` and `model.safetensors`.
If neither strong SHA nor local artifact hashes are available, CiteMesh uses a
deterministic offline identity token (`...::offline-unverified`) and emits warnings
that cache reuse is based on assumptions rather than full verification.
In that mode, CiteMesh records the assumed fingerprint in cache metadata so future
offline checks are explicit and traceable.
When requested revision is `main` and only a legacy cached SHA is available, reuse
is still allowed with a warning because `main` cannot be proven offline.
Set `CITEMESH_STRICT_OFFLINE_FINGERPRINT=1` to disable that legacy `main` reuse
assumption and force namespace clear/rebuild when identity cannot be verified
(details: [Environment Variables](../reference/environment.md)).
If compatibility checks fail (for example unresolved revision mismatch), CiteMesh clears
and rebuilds that namespace before reuse to avoid stale model-version mixing.

When hydration metadata matches the requested split/corpus cap, records a non-empty dataset source, and points to a queryable embedding+metadata row mapping, embedding retrieval runs fully from cache and skips HuggingFace corpus loading.

Current limitation: hydration compatibility is keyed to dataset source/split/corpus
metadata, not an immutable upstream dataset revision fingerprint. If a dataset alias
mutates upstream without changing source name, treat cache reuse as a performance
optimization rather than a strict reproducibility guarantee.

During cache-native search, scored embedding rows must map to metadata rows. Missing
metadata row mappings now fail closed with an integrity error instead of returning
partial top-k results.

If cache payload files become inconsistent (for example missing matrix file, incompatible layout, or invalid calibration metadata), CiteMesh resets that namespace state and rebuilds on the next hydration run.

## Semantic Scholar Reference Cache

When reference expansion is enabled, reference-ID lookups are cached under `references/`
using hashed filenames.

- Default policy is no TTL: version-matched cache entries are reused until manually
  cleared or refreshed.
- `--refresh-reference-cache` bypasses persisted reference-cache reads and fetches
  fresh reference IDs from the API (write-through cache update).
- Successful empty reference responses are cached as explicit empty lists to avoid
  repeated API calls for papers with no references.
- Repeated reference-fetch failures now raise a runtime error after retries instead
  of silently returning an empty list.
- Reference cache directory resolution occurs at call time, so cache-root policy
  (`CITEMESH_CACHE_DIR`) changes are honored for new lookups.

## HuggingFace Default Cache

HuggingFace libraries keep their own cache (commonly `~/.cache/huggingface`). CiteMesh does not override that location.

## Cleaning and Inspecting Cache

Inspect cache usage:

```bash
citemesh cache scan
```

Clear entire CiteMesh cache root:

```bash
citemesh cache clear --yes
```

Omit `--yes` for interactive confirmation.

For command syntax and defaults, see [CLI Usage](cli.md); this section focuses on cache maintenance workflows.

To remove artifacts for one namespace, delete matching `.db` and `.h5` files in `embeddings/`.

Manual full reset examples:

```bash
# Linux
rm -rf "${XDG_CACHE_HOME:-$HOME/.cache}/citemesh"

# macOS
rm -rf "$HOME/Library/Caches/citemesh"

# custom root
rm -rf "$CITEMESH_CACHE_DIR"
```

On Windows, remove the cache directory via Explorer or PowerShell.

## Best Practices

- Keep cache data on fast local storage (SSD) for embedding-heavy runs.
- Periodically prune unused model caches if disk usage grows.
- In shared environments, set `CITEMESH_CACHE_DIR` per user or per project.
