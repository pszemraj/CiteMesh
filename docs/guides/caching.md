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

Model hashes are the first 12 characters of `sha256(model_name)`. For embedding strategy caches, the namespace string includes model + truncate dim + storage precision + binary prefilter mode + source dtype hint, so incompatible precision modes are isolated by design.

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
- hydration metadata keys (`dataset source`, `split`, `corpus cap`, completion flag)

A vector is recomputed when:

- The paper is missing from cache, or
- The composed text (`title + abstract`) hash changed.

Cache writes are serialized via per-model lock files (`cache_<model-hash>.lock`) to avoid multi-process HDF5 write races.

Embedding/hybrid workflows can trigger a namespace rebuild using `--force-rebuild-cache` (flag semantics are canonical in [CLI Usage](cli.md)).

When hydration metadata matches the requested split/corpus cap and a queryable embedding matrix exists, embedding retrieval runs fully from cache and skips HuggingFace corpus loading.

## Semantic Scholar Reference Cache

When reference expansion is enabled, reference-ID lookups are cached under `references/` using hashed filenames.

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
