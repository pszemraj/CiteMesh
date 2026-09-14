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
│   ├── metadata_<hash>.db     # SQLite: ids, text hashes, row_idx, metadata, chronology, hydration
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

There is no TTL: entries are reused until you clear them or pass `--refresh-paper-cache`, which bypasses persisted reads for that run and replaces entries — citation counts included — from fresh responses, keeping the old entry if the fetch fails. Refreshing does not re-encode unchanged title and abstract text, so embedding caches are untouched. Entries carry paper-cache schema version 2 — the embedding cache further down keeps its own, unrelated number — and anything older reads as a miss.

Reference IDs live in `references/` under the same no-TTL policy with their own `--refresh-reference-cache`, since cached metadata does not imply its references were fetched. Empty reference lists are cached explicitly so those papers stop costing requests; a first-page `paper not found` returns empty *without* caching, so a later lookup can recover. Corrupt or unusable payloads are rebuilt from the API.

Caching does not make a build offline: citation, reference, recommendation, and search calls still need the network.

## Embedding namespaces

Vectors are stored once per namespace in a resizable HDF5 matrix, with SQLite tracking metadata and row mappings; filenames carry the first 12 characters of `sha256(<namespace>)`. The namespace binds everything that could change what a vector means, from the resolved model artifact down to the text formatter — [How CiteMesh builds a graph](how-it-works.md) walks through that fingerprint. What it buys you here:

- Two namespaces per model contract: a retrieval-document cache (candidate or corpus papers, what local `citemesh search` reads) and a graph-similarity cache (selected graph papers, always float32, the only source for paper-to-paper edges). Matching dimensions do not make their vectors interchangeable, and candidate mode further tags its retrieval namespace `mode=candidates` so S2 candidates never mix with a corpus hydration.
- No device or compute-dtype token in the namespace: CPU, CUDA, and MPS resolve the same one whenever the other contracts match, so a corpus built on a bf16 GPU is read directly by an fp32 host rather than re-encoded. An auto-resolved compute dtype is provenance rather than identity — like the attention backend, TF32, and `--torch-compile`, it shifts numerics slightly without changing what a vector means. The dtype that created a namespace is still recorded, but it does not decide compatibility on reopen.
- Changing model, revision, profile, dimension, or precision selects a *different* namespace rather than invalidating the old one, so switching back reopens the original vectors. EmbeddingGemma now defaults to 512 dimensions; 256-dimensional caches survive, and `--truncate-dim 256` still selects them.
- The binary prefilter is not part of the namespace: toggling it reuses the same vectors, ranges, and hydration state, rebuilding or dropping only the derived index.
- A vector is re-encoded only when the paper is new to the namespace or its input text changed. Metadata-only updates refresh the SQLite row, not the vector.

The migration that removed compute dtype from namespace names deliberately leaves older dtype-keyed `.db` and `.h5` pairs untouched: their filename hashes cannot be retargeted safely. Upgrading rehydrates a new shared namespace while the old vectors remain on disk. `citemesh cache scan` lists every embedding namespace separately and marks these schema-3 pairs as reclaimable; after confirming the replacement cache works, delete the matching `metadata_<namespace>.db` and `embeddings_<namespace>.h5` files to reclaim their space. `citemesh cache clear` also reclaims them, but deletes every namespace.

Crash safety, the replacement journal, flush ordering, and locking are in [Embedding Cache Internals](../internals/embedding-cache.md).

## Corpus hydration and resume

arXiv-corpus mode hydrates the full selected `--dataset-split` by default — "all of `train`", not every split the dataset publishes. `--corpus-size N` opts into the N newest submissions by arXiv ID; on a cold build the cap bounds what gets embedded, not how many rows are scanned to establish that order. Under `--streaming` that ranking must drain the whole stream before hydration starts, and CiteMesh warns about it; `--no-streaming` skips the drain but still reads the `id` of every row, and a slice such as `train[:2%]` shrinks the population ranked rather than skipping the ranking.

Hydration is resumable across cap changes: a smaller request finishes the recorded selection, and a larger request expands it. An interrupted run scans the selected source IDs and encodes only missing papers, provided the recorded source and split match, the recorded cap is compatible, and the namespace is self-consistent — equal SQLite and embedding row counts, at least one row, and persisted calibration ranges under `--storage-precision int8`. Anything else falls through to a full rebuild.

A hydrated cache on a non-sliced split also tracks upstream growth, and an unchanged upstream row count is answered from memoized cache metadata without re-scanning the corpus. A full-split cache uses a changed source count to trigger an ID scan and appends every uncached paper, including when net growth combines additions and removals; a capped one checks the upstream newest-N selection against its cached paper IDs, including backfilled submissions that leave the newest timestamp unchanged and fallback selections without parseable submission dates. Rows are only added, never evicted, so an ordinary rerun can leave a capped cache holding more than `--corpus-size`; CiteMesh warns and reports the actual count. When the upstream shrinks, CiteMesh scans its IDs immediately, preserves historical vectors, and records the verified source count separately from the cached vector count. Repeated runs reuse that reconciliation; a changed source count triggers another scan, including when regrowth happens to match the number of stored vectors. Same-count replacements and revised abstracts still go undetected, so rebuild for those ([issue #13](https://github.com/pszemraj/CiteMesh/issues/13)).

Changing the cap never costs you vectors you already have: a larger `--corpus-size` — or `--all-corpus` — extends the same namespace in place, encoding only the newly selected papers; a smaller one reuses the existing rows as-is, warning that results come from the larger cached corpus and leaving the recorded cap where the vectors actually are. An interrupted extension keeps the cache complete at its recorded size rather than discarding it. That recorded cap is a floor, not an inventory — a recency refresh or upstream shrink can retain historical rows past the current selection — so use `citemesh cache clear` or `--force-rebuild-cache` when you want exactly the requested size and current upstream membership.

Of the corpus flags, only a changed `--dataset-source` or `--dataset-split` still clears, and that automatic rebuild empties only the retrieval namespace; `--force-rebuild-cache` is the path that also discards graph-similarity vectors. A failed load stops the build before anything is replaced, and local search refuses a source mismatch outright. A failed storage inspection surfaces the paths and the original error rather than reading as an empty cache.

A model-fingerprint mismatch is not one of those conditions. When a hydrated corpus was built under a different fingerprint than the active model's, CiteMesh stops and names both fingerprints, the rows and on-disk size at stake, and the remedy — `--force-rebuild-cache`, plus `--overwrite-cache` for scripts, or `citemesh cache clear`. Candidate-pool and graph-similarity namespaces re-encode in seconds and still clear silently; corpus hydration metadata is what earns a namespace the protection.

An int8 write whose coordinates fall outside the persisted calibration ranges warns once per run; persistent warnings are the one signal worth acting on. Ranges cannot be replaced in place, since they also decode existing rows, so recalibrating means `--force-rebuild-cache` — and a larger `--calibration-sample-size` starts a fresh namespace.

One caveat worth internalizing: hydration compatibility is keyed to dataset source, split, and cap rather than an immutable upstream revision, and a capped corpus deliberately chases upstream growth even when the alias never changes. Treat cache reuse as a performance optimization, not a reproducibility guarantee; a sliced `--dataset-split` opts out of both growth checks when you need a fixed population.

## Inspecting and clearing

```bash
# usage by section
citemesh cache scan
citemesh cache clear --yes --reason "manual local reset"
```

`cache scan` reports section totals and then each standard embedding namespace's ID, files, bytes, and schema status. `cache clear` deletes every entry under the cache root except `config.toml`, its lock, and `.locks/`. It fails without deleting anything while a cache operation is live, and it cannot interrupt a pending config write. To drop a single namespace instead, delete the matching `.db` and `.h5` in `embeddings/`.

> [!CAUTION]
> `cache clear` and `--force-rebuild-cache` are irreversible. A forced rebuild clears both namespaces for the resolved model contract and re-encodes from scratch, which on a hydrated corpus means hours of GPU time. Both prompt interactively on a TTY and refuse outright without one, so `--yes` and `--overwrite-cache` are how a script approves them; `--cache-overwrite-reason "<text>"` records a rationale that otherwise logs as `reason=unspecified`.

A rewritten HDF5 file can grow in bytes while its row count stays fixed: compressed rows leave holes and CiteMesh does not compact automatically. That is not lost vectors or duplicated rows.

For a full reset including `config.toml`, remove the root itself — `rm -rf "${XDG_CACHE_HOME:-$HOME/.cache}/citemesh"`, or `$CITEMESH_CACHE_DIR` if you set one, plus the legacy `~/Library/Caches/citemesh` on macOS.

Flag syntax and defaults: [CLI Usage](cli.md). Variables: [Environment Variables](../reference/environment.md).
