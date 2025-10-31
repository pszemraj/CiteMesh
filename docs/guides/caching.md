# Caching & Data Storage

CiteMesh uses persistent caches to avoid recomputing expensive datasets and embeddings. This guide explains what gets cached, where it lives by default, and how to override the locations.

## Cache Root

By default, all project-specific caches live under:

- **Linux / macOS**: `~/.cache/citemesh`
- **Windows**: `%LOCALAPPDATA%\CiteMesh` (or `%APPDATA%` if `LOCALAPPDATA` is unset)

Override the root path by setting an environment variable before running the CLI:

```bash
export CITEMESH_CACHE_DIR=/path/to/custom/cache
```

## Directory Layout

```
citemesh cache root
├── embeddings/
│   ├── metadata_<model-hash>.db   # SQLite metadata (paper ids, hashes, dims)
│   └── embeddings_<model-hash>.h5 # HDF5 vectors
├── joblib/
│   └── ...                        # HuggingFace dataset shards cached via joblib
└── logs/ (future use)
```

> Model hashes are the first eight characters of the SHA-256 digest of the model name, ensuring caches stay isolated when you switch between sentence-transformer checkpoints.

## Embedding Cache

`EmbeddingCache` stores each paper's embedding only once per model. A new vector is computed when:

- The paper is not yet in the cache, or
- The combined text (`title + abstract`) has changed (detected via SHA-256 hash).

This makes iterative runs fast: after the first run, loading vectors becomes a disk-read operation even for large corpora.

## Joblib Dataset Cache

`datasets.load_dataset` is wrapped with joblib caching. When you request a split like `train[:5%]`, the underlying HuggingFace dataset is stored in `joblib/` so subsequent runs reuse the on-disk Arrow shards instead of re-downloading.

## HuggingFace Default Cache

The HuggingFace library also maintains its own cache (usually `~/.cache/huggingface`). CiteMesh does not alter that setting; you can relocate it by configuring HuggingFace environment variables if desired.

## Cleaning the Cache

To remove embeddings for a given model, delete the corresponding `.db` and `.h5` files inside `embeddings/`. You can safely regenerate them on the next run. For a full reset:

```bash
rm -rf ~/.cache/citemesh
```

Or with a custom root:

```bash
rm -rf "$CITEMESH_CACHE_DIR"
```

## Best Practices

- Keep the cache on fast local storage (SSD) for best performance during embedding-heavy workflows.
- Consider pruning unused model caches occasionally if disk space is a concern.
- For shared environments, set `CITEMESH_CACHE_DIR` to a path scoped to the user or project to avoid collisions.
