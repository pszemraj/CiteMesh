# Caching & Data Storage

CiteMesh uses persistent caches to avoid recomputing expensive datasets and embeddings.

## Scope

This is the canonical cache behavior specification.

- Normative here: cache root resolution, directory layout, quantized embedding cache behavior, and cleanup guidance.
- Non-normative here: broader CLI command semantics. See [CLI Usage](cli.md) for command contracts.
- Documentation ownership map: [Documentation Index](../README.md).

## Cache Root

By default, project caches are stored under:

- **Linux**: `${XDG_CACHE_HOME:-~/.cache}/citemesh`
- **macOS**: `~/Library/Caches/citemesh`
- **Windows**: `%LOCALAPPDATA%\\CiteMesh` (or `%APPDATA%\\CiteMesh` if `LOCALAPPDATA` is unset)

Override the root with:

```bash
export CITEMESH_CACHE_DIR=/path/to/custom/cache
```

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

Model hashes are the first 12 characters of `sha256(<namespace>)`. The embedding namespace string includes model + resolved truncate dim + storage precision + effective binary prefilter mode + resolved source torch dtype, and adds calibration sample size in `int8` mode.

## Embedding Cache Behavior

`EmbeddingCache` stores each paper embedding once per namespace. Vectors are kept in a resizable HDF5 matrix, while SQLite tracks metadata and `row_idx` mappings.

Default storage mode is quantized:

- `embeddings`: `int8` matrix (`N x dim`)
- `calibration_ranges`: float32 per-dimension min/max (`2 x dim`)
- `binary_index`: packed `uint8` matrix (`N x ceil(dim/8)`) used for Hamming prefiltering

Non-int8 modes (`float16`, `float32`) are supported via `--storage-precision`.

SQLite stores metadata authority fields used for warm-cache retrieval:

- `title`, `abstract`, `year`
- `authors_json`, `categories_json`
- runtime cache consistency keys (`storage_precision`, source torch dtype, effective embedding vector dtype, binary-prefilter mode, and `int8` calibration sample size)
- hydration metadata keys (`dataset source`, `split`, `corpus cap`, completion flag)
- `model_fingerprint` (active model identity guard for namespace reuse)

A vector is recomputed when:

- The paper is missing from cache, or
- The embedding invalidation hash changed (derived from `title`, `abstract`, `year`,
  `authors`, `categories`, and composed embedding text).

Cache writes are serialized via per-model lock files (`cache_<model-hash>.lock`) to avoid multi-process HDF5 write races.

Embedding/hybrid workflows can trigger a namespace rebuild using `--force-rebuild-cache` (flag semantics are canonical in [CLI Usage](cli.md)).

For Hugging Face repo IDs, hydration resolves and stores a commit-SHA fingerprint.
If the active fingerprint cannot be resolved while a cached fingerprint exists, CiteMesh
reuses the cached namespace with a warning to avoid unnecessary failures in offline
or rate-limited environments. If a stored fingerprint differs from a successfully
resolved active fingerprint, CiteMesh clears and rebuilds that namespace before reuse.

When hydration metadata matches the requested split/corpus cap, records a non-empty dataset source, and points to a queryable embedding+metadata row mapping, embedding retrieval runs fully from cache and skips HuggingFace corpus loading.

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

Command syntax/defaults remain canonical in [CLI Usage](cli.md); this section documents cache maintenance workflows.

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
