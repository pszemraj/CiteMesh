# Caching & Data Storage

CiteMesh uses persistent caches to avoid recomputing expensive datasets and embeddings. This is the canonical cache/storage reference for the repository.

## Cache Root

By default, all project-specific caches live under:

- **Linux**: `${XDG_CACHE_HOME:-~/.cache}/citemesh`
- **macOS**: `~/Library/Caches/citemesh`
- **Windows**: `%LOCALAPPDATA%\CiteMesh` (or `%APPDATA%` if `LOCALAPPDATA` is unset)

Override the root path by setting an environment variable before running the CLI:

```bash
export CITEMESH_CACHE_DIR=/path/to/custom/cache
```

## Directory Layout

```
citemesh cache root
├── embeddings/
│   ├── metadata_<model-hash>.db   # SQLite metadata (paper ids, hashes, dims, row_idx)
│   ├── embeddings_<model-hash>.h5 # HDF5 matrix dataset: embeddings[row_idx] -> vector
│   └── cache_<model-hash>.lock    # Inter-process file lock for cache mutation
├── joblib/
│   └── ...                        # Normalized corpus payloads cached via joblib
└── references/
    └── <sha1>.json                # Semantic Scholar reference ID cache entries
```

> Model hashes are the first **12** characters of the SHA-256 digest of the model name, ensuring caches stay isolated when you switch between sentence-transformer checkpoints.

## Embedding Cache

`EmbeddingCache` stores each paper's embedding only once per model. Vectors are appended to a single resizable HDF5 matrix dataset, and SQLite tracks each paper's matrix row via `row_idx`.

A new vector is computed when:

- The paper is not yet in the cache, or
- The combined text (`title + abstract`) has changed (detected via SHA-256 hash).

Older per-paper HDF5 cache layouts are renamed to a `.bak...` backup on first use, so existing data can be manually recovered before matrix storage is rebuilt.

`clear()` and migration operations now preserve old cache bytes by moving them to timestamped backups (`.bak.<state>.<timestamp>`). Use this to recover from unexpected behavior before removing those backups manually.

Embedding-cache writes are serialized with a per-model lock file (`cache_<model-hash>.lock`). This prevents concurrent processes from opening/writing the same HDF5 file at the same time, eliminating lock/open races under parallel workloads.

To force immediate rebuilds from scratch, pass `--force-rebuild-cache` with the `embedding` or `hybrid` strategy.

This makes iterative runs fast: after the first run, loading vectors becomes a disk-read operation even for large corpora.

For CLI-level cache invalidation controls and flags, refer to [CLI Usage](../guides/cli.md).

## Joblib Dataset Cache

ArXiv corpus loading is wrapped with joblib caching. The normalized corpus mapping (for a given split and paper cap) is cached under `joblib/`, so repeated runs can skip rebuilding that in-process structure.

## Semantic Scholar Reference Cache

When reference expansion is enabled, CiteMesh caches reference-id lookups under `references/` using hashed filenames. This reduces repeated API calls for the same paper IDs across runs.

## HuggingFace Default Cache

The HuggingFace library also maintains its own cache (usually `~/.cache/huggingface`). CiteMesh does not alter that setting; you can relocate it by configuring HuggingFace environment variables if desired.

## Cleaning the Cache

For a full cache reset via CLI:

```bash
citemesh cache clear --yes
```

Omit `--yes` to get an interactive confirmation prompt.

To remove embeddings for a given model, delete the corresponding `.db` and `.h5` files inside `embeddings/`. You can safely regenerate them on the next run. For a full reset on Linux/macOS:

```bash
# Linux
rm -rf "${XDG_CACHE_HOME:-$HOME/.cache}/citemesh"

# macOS
rm -rf "$HOME/Library/Caches/citemesh"
```

Or with a custom root:

```bash
rm -rf "$CITEMESH_CACHE_DIR"
```

On Windows, remove the cache directory in Explorer or PowerShell instead of `rm -rf`.

## Best Practices

- Keep the cache on fast local storage (SSD) for best performance during embedding-heavy workflows.
- Consider pruning unused model caches occasionally if disk space is a concern.
- For shared environments, set `CITEMESH_CACHE_DIR` to a path scoped to the user or project to avoid collisions.
