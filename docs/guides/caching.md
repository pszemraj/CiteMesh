# Caching & Data Storage

CiteMesh uses persistent caches to avoid recomputing expensive datasets and embeddings.

## Scope

This is the canonical cache behavior specification.

- Normative here: cache root resolution, directory layout, cache migration/backup behavior, and cleanup guidance.
- Non-normative here: broader CLI command semantics. See [CLI Usage](cli.md) for command contracts.

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
│   ├── metadata_<model-hash>.db   # SQLite metadata (paper ids, hashes, dims, row_idx)
│   ├── embeddings_<model-hash>.h5 # HDF5 matrix dataset: embeddings[row_idx] -> vector
│   └── cache_<model-hash>.lock    # Inter-process lock for cache mutation
├── joblib/
│   └── ...                        # Normalized corpus payloads cached via joblib
└── references/
    └── <sha1>.json                # Semantic Scholar reference ID cache entries
```

Model hashes are the first 12 characters of `sha256(model_name)` so cache artifacts remain isolated per model.

## Embedding Cache Behavior

`EmbeddingCache` stores each paper embedding once per model. Vectors are kept in a single resizable HDF5 matrix, while SQLite tracks metadata and `row_idx` mappings.

A vector is recomputed when:

- The paper is missing from cache, or
- The composed text (`title + abstract`) hash changed.

Legacy per-paper HDF5 layouts are moved to timestamped `.bak...` files during migration. `clear()` also preserves prior cache bytes by backing up existing files (`.bak.<state>.<timestamp>`).

Cache writes are serialized via per-model lock files (`cache_<model-hash>.lock`) to avoid multi-process HDF5 write races.

Embedding/hybrid workflows can trigger a model-specific rebuild using `--force-rebuild-cache` (flag semantics are canonical in [CLI Usage](cli.md)).

## Joblib Dataset Cache

ArXiv corpus normalization is cached under `joblib/` so repeated runs can skip rebuilding the same in-process corpus mapping.

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

To remove artifacts for one model, delete matching `.db` and `.h5` files in `embeddings/`.

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
