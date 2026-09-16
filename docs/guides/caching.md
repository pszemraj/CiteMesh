# Caching and data

CiteMesh stores paper metadata, reference lists, and embeddings on disk for reuse between runs.

## Where it lives

The [environment settings](../reference/environment.md#platform-variables-used-for-cache-root-resolution) determine the cache root. `citemesh cache scan` prints its path and usage, including a migration hint if the older macOS cache still exists.

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
├── discovery/
│   └── <sha1>.json            # successful ordered S2 discovery IDs
└── references/
    └── <sha1>.json            # S2 reference IDs
```

`config.toml` is configuration, not cache: it survives `citemesh cache clear` along with its lock and `.locks/` ([User Configuration](configuration.md)). Dashboard collection locks live beside their output packages, not here. HuggingFace keeps checkpoints and datasets in its own cache (`~/.cache/huggingface`, relocatable with `HF_HOME`), which CiteMesh does not touch.

## Paper metadata and reference IDs

Successful Semantic Scholar lookups persist paper metadata under `papers/`, keyed by the requested ID and known S2, arXiv, and DOI aliases. Later lookups check disk first, seed resolution included, and batch requests send only the IDs still missing. Failed and malformed responses are not cached.

Paper entries have no TTL. `--refresh-paper-cache` bypasses persisted reads and replaces metadata, including citation counts, from successful responses; a failed fetch preserves the old entry. Unchanged title and abstract text reuses its embedding. Paper-cache schema version 2 is required; older entries are treated as misses.

Reference IDs live in `references/` under the same no-TTL policy with their own `--refresh-reference-cache`, since cached metadata does not imply its references were fetched. Empty reference lists are cached explicitly so those papers stop costing requests; a first-page `paper not found` returns empty *without* caching, so a later lookup can recover. Corrupt or unusable payloads are rebuilt from the API.

Caching does not make a build offline: citation, reference, recommendation, and search calls still need the network.

## Discovery checks

Seeded discovery has its own cache records under `discovery/`, separate from complete paper metadata and reference-ID enrichment. A normal build always rechecks the ordered upstream IDs for the requested recommendations, citations, or references with the minimal `paperId` projection; immediate reruns behave the same way. Successful checks update their snapshot, keyed by the normalized seed, endpoint, effective limit, and recommendation pool, and record when that check completed. The on-disk snapshot is diagnostic: CiteMesh reads it only to report whether membership or ordering changed since the previous successful check. It is never used to serve candidate IDs, skip the next upstream check, make discovery offline, or provide a stale fallback. Repeated requests within one candidate collection reuse the freshly checked in-memory IDs instead.

The check preserves upstream membership and ordering among resolvable papers. CiteMesh rebuilds candidates in that order from `papers/`, batch-fetching only new IDs in batches of at most 500. Cached full records avoid metadata fetches when every returned ID is already present. If the metadata service returns a null or malformed record for one discovered ID, CiteMesh warns and omits that ID from the result and successful snapshot while retaining the other papers. IDs removed upstream leave the current candidate list but remain in the paper cache for other builds.

Only successful discovery responses are stored. A failed page, failed required metadata batch, or failed `all-cs` fallback preserves the prior successful snapshot and stops the required acquisition rather than treating the failure as an empty result. When the recent recommendation pool returns a valid empty list, CiteMesh checks `all-cs`; only two successful empty checks mean there are no recommendations. A bounded reference-discovery response never overwrites the complete reference-ID enrichment cache.

Normal build output reports that discovery IDs were rechecked upstream separately from reused paper metadata and reference IDs. `--refresh-paper-cache` and `--refresh-reference-cache` still control those two enrichment caches; neither suppresses the discovery check.

## Embedding namespaces

A namespace combines model, artifact identity, representation, dimension, formatter, and storage settings. The artifact identity is a resolved commit SHA or a digest of the inference-artifact manifest, so changing a checkpoint selects a different cache. SQLite and HDF5 store each namespace as a pair; [physical layout](../internals/embedding-cache.md#physical-layout) describes the files.

Local custom model fingerprints include declared Python modules, their package initializers, and transitive relative imports. Changing those files selects a new namespace. Fingerprinting parses the code without executing it.

- The [retrieval-document and graph-similarity roles](../reference/embedding-runtime.md#task-specific-vector-spaces) have separate namespaces. Candidate mode additionally tags its retrieval namespace `mode=candidates` so S2 candidates never mix with corpus hydration.
- No device or compute-dtype token in the namespace: CPU, CUDA, and MPS resolve the same one whenever the other contracts match, so a corpus built on a bf16 GPU is read directly by an fp32 host rather than re-encoded. An auto-resolved compute dtype is provenance rather than identity - like the attention backend, TF32, and `--torch-compile`, it shifts numerics slightly without changing what a vector means. The dtype that created a namespace is still recorded, but it does not decide compatibility on reopen.
- Changing model, revision, profile, dimension, storage precision, or int8 calibration size selects a different namespace. Switching back to the earlier settings reopens the original vectors.
- The binary prefilter is not part of the namespace: toggling it reuses the same vectors, ranges, and hydration state, rebuilding or dropping only the derived index.
- A vector is re-encoded only when the paper is new to the namespace or its input text changed. Metadata-only updates refresh the SQLite row, not the vector.

The migration that removed compute dtype from namespace names deliberately leaves older dtype-keyed `.db` and `.h5` pairs untouched: their filename hashes cannot be retargeted safely. Upgrading rehydrates a new shared namespace while the old vectors remain on disk. `citemesh cache scan` lists every embedding namespace separately and marks these schema-3 pairs as reclaimable; after confirming the replacement cache works, delete the matching `metadata_<namespace>.db` and `embeddings_<namespace>.h5` files to reclaim their space. `citemesh cache clear` also reclaims them, but deletes every namespace.

Crash safety, the replacement journal, flush ordering, and locking are in [Embedding Cache Internals](../internals/embedding-cache.md).

## Corpus records

The dataset adapter accepts `id`, `paper_id`, or `paperId`, taking the first non-empty identifier. Stable source IDs preserve cache identity across updates. Without an ID, usable title or abstract text produces a content-derived ID; a row with neither fails and reports its source-row index.

`title` supplies the paper name. `abstract` supplies the text, with `summary` used when the abstract is empty or unusable. Optional fields include `authors`, `authors_parsed`, `categories`, `year`, `doi`, and `venue` / `journal_ref` / `journal`. Structured `authors_parsed` entries take precedence over free-text authors. Missing titles become `Unknown`; years fall back to the arXiv submission year when available. Arbitrary column mappings are unsupported.

## Corpus hydration and resume

arXiv-corpus mode hydrates the full selected `--dataset-split` by default - "all of `train`", not every split the dataset publishes. `--corpus-size N` opts into the N newest submissions by arXiv ID; on a cold build the cap bounds what gets embedded, not how many rows are scanned to establish that order. Streaming ranking drains the source before encoding begins. An indexable non-streaming dataset instead scans its identifier columns. A slice such as `train[:2%]` limits the population ranked. Rows without parseable arXiv IDs fill any shortfall in source order, with a warning; when none can be parsed, the cap selects the first N rows.

An interrupted run scans the selected source IDs and encodes only missing papers, provided the recorded source and split match, the recorded cap is compatible, and the namespace is self-consistent - equal SQLite and embedding row counts, at least one row, and persisted calibration ranges under `--storage-precision int8`. Anything else falls through to a full rebuild.

Versioned metadata migrations rescan the selected source split and update cached publication years, DOIs, and venues in place while preserving vectors and corpus membership. The summary adapter upgrade also restores usable `summary` text that an empty `abstract` previously hid, re-encoding only affected rows. Older positional IDs remain unchanged and also match their equivalent content identity; source membership is reconciled once without rewriting unchanged vectors. Summary restoration matches the old adapter's exact output before assigning an updated identity; if discarded summaries made several distinct anonymous papers indistinguishable, the upgrade reports that a rebuild or stable source IDs are required.

A hydrated cache on a non-sliced split checks upstream growth:

| Upstream state | Reuse or update |
| --- | --- |
| Published row count matches the last verified count | Reuse memoized metadata without scanning source IDs. |
| Row count changed | Scan current IDs. Full-split caches append missing papers; capped caches compare the newest-N selection, including backfilled submissions. |
| No published row count | Scan IDs each build. If this optional check cannot read the source, keep the complete cache usable and retry next time. |
| Rows removed upstream | Preserve historical vectors and record the verified source count separately from stored vector count. |

Rows are added, never evicted automatically. A capped cache can therefore exceed `--corpus-size`; the run warns and reports the actual count. A later count change triggers another scan even if it equals the number of stored vectors. Explicit source IDs remain authoritative, so rebuild after ID remapping. Same-count replacements and revised abstracts are not detected by the count check; rebuild for those ([issue #13](https://github.com/pszemraj/CiteMesh/issues/13)).

Changing the cap never costs you vectors you already have: a larger `--corpus-size` - or `--all-corpus` - extends the same namespace in place, encoding only the newly selected papers; a smaller one reuses the existing rows as-is, warning that results come from the larger cached corpus and leaving the recorded cap where the vectors actually are. An interrupted extension keeps the cache complete at its recorded size rather than discarding it. That recorded cap is a floor, not an inventory - a recency refresh or upstream shrink can retain historical rows past the current selection - so use `citemesh cache clear` or `--force-rebuild-cache` when you want exactly the requested size and current upstream membership.

Of the corpus flags, only a changed `--dataset-source` or `--dataset-split` still clears, and that automatic rebuild empties only the retrieval namespace; `--force-rebuild-cache` is the path that also discards graph-similarity vectors. A failed load stops the build before anything is replaced, and local search refuses a source mismatch outright. A failed storage inspection surfaces the paths and the original error rather than reading as an empty cache.

A model-fingerprint mismatch is not one of those conditions. When a hydrated corpus was built under a different fingerprint than the active model's, CiteMesh stops and names both fingerprints, the rows and on-disk size at stake, and the remedy - `--force-rebuild-cache`, plus `--overwrite-cache` for scripts, or `citemesh cache clear`. Candidate-pool and graph-similarity namespaces re-encode in seconds and still clear silently; corpus hydration metadata is what earns a namespace the protection.

Hydration compatibility uses dataset source, split, and cap rather than an immutable upstream revision. A sliced `--dataset-split` skips growth checks, but the same slice expression can refer to different records after an upstream update. Neither slicing nor cache reuse guarantees an immutable corpus.

### Quantization

Corpus storage uses per-dimension affine int8 values. The optional Hamming prefilter keeps `top_k * binary_rescore_multiplier` rows for exact vector rescoring. Calibration ranges come from a reservoir sample and are persisted before the first int8 write.

An int8 write outside those ranges warns once per run. Existing rows need the original ranges for decoding, so recalibration requires a forced rebuild; changing the calibration sample size selects a new namespace. Candidates use float32 without calibration. Flag defaults are in the [CLI storage options](cli.md#graph-edges-and-cache-storage).


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

For a full reset including configuration, stop active CiteMesh processes and remove the root printed by `citemesh cache scan`. Remove any reported legacy cache separately.

Flag syntax and defaults: [CLI Usage](cli.md). Variables: [Environment Variables](../reference/environment.md).
