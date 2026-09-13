# Caching & Data

CiteMesh caches paper metadata, reference lists, and embeddings on disk so a rerun costs a fraction of the first run. This page is the map: where those files live, what makes CiteMesh reuse or re-encode them, and how to reset the lot.

## Where it lives

The default cache root is `${XDG_CACHE_HOME:-~/.cache}/citemesh` on Linux and macOS, `%LOCALAPPDATA%\CiteMesh` on Windows (falling back to `%APPDATA%`, then `%USERPROFILE%\AppData\Local`). `CITEMESH_CACHE_DIR` overrides it — one root per user or project on a shared machine. macOS used `~/Library/Caches/citemesh` before the root was unified with Linux, and `citemesh cache scan` hints at migration while it exists.

```text
citemesh cache root
├── config.toml                # persistent user configuration
├── config.toml.lock           # config write coordination
├── .locks/
│   └── cache-operations.db    # SQLite operation/clear coordination
├── embeddings/
│   ├── metadata_<hash>.db     # SQLite: ids, text hashes, row_idx, metadata, hydration
│   ├── embeddings_<hash>.h5   # HDF5: vectors, int8 calibration ranges, binary index
│   ├── cache_<hash>.lock      # namespace mutation lock
│   └── hydration_<hash>.lock  # hydration and consuming-search lock
├── papers/
│   └── <sha1>.json            # S2 paper metadata
└── references/
    └── <sha1>.json            # S2 reference IDs
```

`config.toml` is configuration, not cache: it survives `citemesh cache clear` along with its lock and `.locks/` ([User Configuration](configuration.md)). Dashboard collection locks live beside their output packages, not here. HuggingFace keeps checkpoints and datasets in its own cache (`~/.cache/huggingface`, relocatable with `HF_HOME`), which CiteMesh does not touch.

## Paper metadata and reference IDs

Every successful Semantic Scholar lookup — paper fetches, citations, references, recommendations, search — persists under `papers/`, keyed by the requested ID and every known S2, arXiv, and DOI alias. Later lookups check disk first, seed resolution included, and batch requests send only the IDs still missing. Failed and malformed responses are not cached.

There is no TTL: entries are reused until you clear them or pass `--refresh-paper-cache`, which bypasses persisted reads for that run and replaces entries — citation counts included — from fresh responses, keeping the old entry if the fetch fails. Refreshing does not re-encode unchanged title and abstract text, so embedding caches are untouched. Entries carry schema version 2; anything older reads as a miss.

Reference IDs live in `references/` under the same no-TTL policy with their own `--refresh-reference-cache`, since cached metadata does not imply its references were fetched. Empty reference lists are cached explicitly so those papers stop costing requests; a first-page `paper not found` returns empty *without* caching, so a later lookup can recover. Corrupt or unusable payloads are rebuilt from the API.

Caching does not make a build offline: citation, reference, recommendation, and search calls still need the network.

## Embedding namespaces

Vectors are stored once per namespace in a resizable HDF5 matrix, with SQLite tracking metadata and row mappings; filenames carry the first 12 characters of `sha256(<namespace>)`. The namespace binds everything that could change what a vector means, from the resolved model artifact down to the text formatter — [How CiteMesh builds a graph](how-it-works.md) walks through that fingerprint. What it buys you here:

- Two namespaces per model contract: a retrieval-document cache (candidate or corpus papers, what local `citemesh search` reads) and a graph-similarity cache (selected graph papers, always float32, the only source for paper-to-paper edges). Matching dimensions do not make their vectors interchangeable, and candidate mode further tags its retrieval namespace `mode=candidates` so S2 candidates never mix with a corpus hydration.
- No device or compute-dtype token: CPU, CUDA, and MPS share one namespace whenever the other contracts match, so a corpus built on a bf16 GPU is read directly by an fp32 host instead of being re-encoded. Compute dtype is recorded as provenance rather than bound into identity — like the attention backend, TF32, and `--torch-compile`, it shifts numerics slightly without changing what a vector means.
- Changing model, revision, profile, dimension, or precision selects a *different* namespace rather than invalidating the old one, so switching back reopens the original vectors. EmbeddingGemma now defaults to 512 dimensions; 256-dimensional caches survive, and `--truncate-dim 256` still selects them.
- The binary prefilter is not part of the namespace: toggling it reuses the same vectors, ranges, and hydration state, rebuilding or dropping only the derived index.
- A vector is re-encoded only when the paper is new to the namespace or its input text changed. Metadata-only updates refresh the SQLite row, not the vector.

Crash safety, the replacement journal, flush ordering, and locking are in [Embedding Cache Internals](../internals/embedding-cache.md).

## Corpus hydration and resume

arXiv-corpus mode hydrates the full selected `--dataset-split` by default — "all of `train`", not every split the dataset publishes. `--corpus-size N` opts into the N newest submissions by arXiv ID; the cap bounds what gets embedded, not how many rows are scanned to establish that order. Under `--streaming` that selection must drain the whole stream first, and CiteMesh warns about it; `--no-streaming` or a slice such as `train[:2%]` avoids the pass.

Hydration is resumable. An interrupted run resumes from its cached rows whenever the recorded source, split, and cap still match, encoding only the missing paper IDs. A full-split cache also gets an incremental growth check against upstream row counts, appending only uncached IDs; same-count replacements and revised abstracts go undetected, so rebuild for those ([issue #13](https://github.com/pszemraj/CiteMesh/issues/13)).

A completed capped cache keeps its original selection, so `--corpus-size` does not roll forward as the dataset grows; rebuild to reselect. A resume can exceed the cap once the upstream newest selection has moved, and CiteMesh reports the actual count.

Changing the cap between runs never costs you the vectors you already have. The recorded cap describes what is cached, so a larger `--corpus-size` — or `--all-corpus` — extends the same namespace in place, encoding only the newly selected papers, and a smaller one reuses the existing rows as-is: CiteMesh warns that results come from the larger cached corpus and leaves the recorded cap where the vectors actually are. Reach for `citemesh cache clear` or `--force-rebuild-cache` when you want a namespace holding exactly the requested size.

Changing `--dataset-source` replaces the corpus in the same namespace through that rebuild path; a failed load stops the build before anything is replaced, and local search refuses a source mismatch outright. A failed storage inspection surfaces the paths and the original error rather than reading as an empty cache.

An int8 write whose coordinates fall outside the persisted calibration ranges warns once per run; persistent warnings are the one signal worth acting on. Ranges cannot be replaced in place, since they also decode existing rows, so recalibrating means `--force-rebuild-cache` — and a larger `--calibration-sample-size` starts a fresh namespace.

One caveat worth internalizing: hydration compatibility is keyed to dataset source, split, and cap rather than an immutable upstream revision. If an alias mutates under the same name, treat cache reuse as a performance optimization, not a reproducibility guarantee.

## Inspecting and clearing

```bash
# usage by section
citemesh cache scan
citemesh cache clear --yes --reason "manual local reset"
```

`cache clear` deletes every entry under the cache root except `config.toml`, its lock, and `.locks/`; omit `--yes` for an interactive prompt. It fails without deleting anything while a cache operation is live, and it cannot interrupt a pending config write. To drop a single namespace instead, delete the matching `.db` and `.h5` in `embeddings/`.

> [!CAUTION]
> `cache clear` and `--force-rebuild-cache` are irreversible. A rebuild clears both namespaces for the resolved model contract and re-encodes from scratch, which on a hydrated corpus means hours of GPU time. Both prompt interactively; `--yes` and `--overwrite-cache` skip the prompt for scripts, and `--cache-overwrite-reason "<text>"` records a rationale that otherwise logs as `reason=unspecified`.

A rewritten HDF5 file can grow in bytes while its row count stays fixed: compressed rows leave holes and CiteMesh does not compact automatically. That is not lost vectors or duplicated rows.

For a full reset including `config.toml`, remove the root itself — `rm -rf "${XDG_CACHE_HOME:-$HOME/.cache}/citemesh"`, or `$CITEMESH_CACHE_DIR` if you set one, plus the legacy `~/Library/Caches/citemesh` on macOS.

Flag syntax and defaults: [CLI Usage](cli.md). Variables: [Environment Variables](../reference/environment.md).
