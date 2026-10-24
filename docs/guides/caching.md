# Caching and data

CiteMesh stores paper metadata, reference lists, and embeddings on disk for reuse between runs.

## Where it lives

The [environment settings](../reference/environment.md#platform-variables-used-for-cache-root-resolution) determine the cache root. `citemesh cache scan` prints its path and usage, including a migration hint if the older macOS cache still exists.

```text
<cache_root>/
├── config.toml                # persistent user configuration
├── config.toml.lock           # config write coordination
├── .locks/
│   └── cache-operations.db    # SQLite operation/clear coordination
├── embeddings/                # native embedding namespaces
├── papers/
│   └── <sha1>.json            # canonical S2 metadata and identifier lookups
└── references/
    └── <sha1>.json            # S2 reference IDs
```

For embedding files, see [physical layout](../internals/embedding-cache.md#physical-layout). Dashboard collection locks live beside their output packages. Hugging Face downloads stay outside CiteMesh's cache ([`HF_HOME`](../reference/environment.md#other-respected-variables)).

## Paper metadata and reference IDs

Successful Semantic Scholar lookups persist one full metadata record under `papers/`, keyed by the returned S2 `paperId`. Requested identifiers and known arXiv/DOI identifiers point directly to that record. Refreshing metadata updates the canonical record; identifiers do not hold separate metadata copies. Later lookups check disk first, seed resolution included, and batch requests send only the IDs still missing. Failed and malformed responses are not cached.

Paper entries have no TTL. `--refresh-paper-cache` bypasses persisted reads and replaces metadata, including citation counts, from successful responses; a failed fetch preserves the old entry. Unchanged title and abstract text reuses its embedding. Version 2 paper caches remain readable: an older alias copy supplies its canonical paper ID, and lookups read the canonical record. No cache rebuild is required.

Reference IDs live in `references/` under the same no-TTL policy with their own `--refresh-reference-cache`, since cached metadata does not imply its references were fetched. Empty reference lists are cached explicitly so those papers stop costing requests; a first-page `paper not found` returns empty *without* caching, so a later lookup can recover. Corrupt or unusable payloads are rebuilt from the API.

Caching does not make a build offline: citation, reference, recommendation, and search calls still need the network.

## Discovery checks

A normal build always checks the ordered upstream IDs for the requested recommendations, citations, or references with the minimal `paperId` projection; immediate reruns behave the same way. Discovery lists are not cached. Each endpoint invocation checks upstream and uses the returned order directly.

The check preserves upstream membership and ordering among resolvable papers. CiteMesh rebuilds candidates in that order from `papers/`, batch-fetching only missing IDs in batches of at most 500. Cached full records avoid metadata fetches when every returned ID is already present. If the metadata service returns a null or malformed record for one discovered ID, CiteMesh warns and omits that ID while retaining the other papers. IDs removed upstream leave the current candidate list but remain in the paper cache for other builds.

A failed page, failed required metadata batch, or failed `all-cs` fallback stops the required acquisition without replacing existing outputs. When the recent recommendation pool yields no resolvable papers, CiteMesh checks `all-cs`; both checks must succeed before returning an empty recommendation list. A bounded reference-discovery response never overwrites the complete reference-ID enrichment cache.

Normal build output reports that discovery IDs were rechecked upstream separately from reused paper metadata and reference IDs. `--refresh-paper-cache` and `--refresh-reference-cache` still control those two enrichment caches; neither suppresses the discovery check.

## Embedding namespaces

A namespace combines model, artifact identity, [representation](../reference/embedding-runtime.md#task-specific-vector-spaces), dimension, formatter, and storage settings. The artifact identity is a resolved commit SHA or a digest of the inference-artifact manifest, so changing a checkpoint selects a different cache. SQLite and HDF5 store each namespace as a pair; [physical layout](../internals/embedding-cache.md#physical-layout) describes the files.

Local custom model fingerprints include declared Python modules, their package initializers, and transitive relative imports. Changing those files selects a new namespace. Fingerprinting parses the code without executing it.

- Candidate mode additionally tags its retrieval namespace `mode=candidates` so S2 candidates never mix with corpus hydration.
- No device or compute-dtype token in the namespace: CPU, CUDA, and MPS resolve the same one whenever the other contracts match, so a corpus built on a bf16 GPU is read directly by an fp32 host rather than re-encoded. An auto-resolved compute dtype is provenance rather than identity - like the attention backend, TF32, and `--torch-compile`, it shifts numerics slightly without changing what a vector means. The dtype that created a namespace is still recorded, but it does not decide compatibility on reopen.
- Changing model, revision, profile, dimension, storage precision, or int8 calibration size selects a different namespace. Switching back to the earlier settings reopens the original vectors.
- The binary prefilter is not part of the namespace: toggling it reuses the same vectors, ranges, and hydration state, rebuilding or dropping only the derived index.
- A vector is re-encoded only when the paper is new to the namespace or its input text changed. Metadata-only updates refresh the SQLite row, not the vector.

Older dtype-keyed `.db` and `.h5` pairs remain separate and appear as reclaimable schema-3 namespaces in `citemesh cache scan`. After confirming the replacement cache works, delete the matching `metadata_<namespace>.db` and `embeddings_<namespace>.h5` files to reclaim their space. `citemesh cache clear` also reclaims them, but deletes every namespace.

Crash safety, the replacement journal, flush ordering, and locking are in [Embedding Cache Internals](../internals/embedding-cache.md).

## Portable corpus indexes

A portable index installs precomputed retrieval-document embeddings into the
native cache. Install the `embeddings` extra first. Import and export copy the
stored int8 or float32 values without loading an encoder; query encoding and
the selected graph papers' symmetric embeddings still use the recorded model.

```bash
# Import a local export, or one configuration from a Storage Bucket.
citemesh cache import --path ./arxiv-index
citemesh cache import --bucket OWNER/INDEX --config embeddinggemma-512-int8 \
  --snapshot SNAPSHOT_ID
citemesh cache scan

# Use the namespace printed by import or scan.
citemesh search "long-context language models" --mode local \
  --cache-namespace NAMESPACE --corpus-size 50000
citemesh build arxiv:2404.08801 --strategy embedding \
  --cache-namespace NAMESPACE --all-corpus

# Export a completed corpus into a new directory.
citemesh cache export --namespace NAMESPACE --output ./arxiv-index
```

`--path` and `--bucket` are alternatives. Bucket imports use the selected
configuration, or the catalog's default, and download only its declared files.
Omit `--snapshot` to install the catalog's current snapshot; an explicit
snapshot requires `--config`. Snapshot identity identifies the portable files;
the manifest separately records the immutable model and upstream dataset
revisions. The upstream source revision is provenance, not namespace identity.
Each configuration can have different coverage and freshness.

Private buckets require a Hugging Face token with read access for import and
write access for publication. Set [`HF_TOKEN`](../reference/environment.md#other-respected-variables)
or save a token with `hf auth login`; public bucket reads need no token.

Explicit `--cache-namespace` selects existing files and pins model, profile,
document formatter, dimension, storage precision, dataset, and split from their
stored contract. Conflicting explicit flags fail. Device, batch size, scoring,
and result limits remain configurable. Search failures stay errors even in
`--mode auto`. Import does not change saved defaults.

Installed indexes are manually refreshed. Build and search use their completed
source revision without checking or scanning the upstream dataset:

```bash
citemesh cache refresh --namespace NAMESPACE
citemesh cache refresh --namespace NAMESPACE --dataset-revision SOURCE_COMMIT
```

A new refresh resolves its source revision once, compares full normalized
records, and encodes only new or changed document text. Metadata-only changes
retain their vectors and need no encoder. Existing int8 calibration ranges stay
fixed. Removed source papers keep their physical vectors but leave active
membership, retrieval, and export. Refresh scans source metadata; it does not
promise a delta download.

Repeated source IDs are resolved before calibration, encoding, or newest-N
selection. The most recent valid `update_date` wins, followed by the highest
arXiv version number and its creation time. When conflicting records have equal
or unavailable revision information, a stable ordering of their normalized
content breaks the tie and CiteMesh warns once with the number of ambiguous
comparisons. This tie-break is deterministic; it does not establish freshness.
Unparseable timestamps rank below valid ones, which are compared in UTC. The
winning record retains its original source position. Equally recent, identical
records retain the first occurrence.

For an installed index, `--corpus-size N` ranks only the newest N active members
at the completed revision; `--all-corpus` uses every installed active member. Chronology ties
prefer later source rows; undated papers fill remaining places in source
order. A smaller view changes neither stored coverage nor membership. A capped
index cannot supply a larger view or claim full coverage. Seed lookup is
independent of neighbor eligibility, so an older seed can search newer papers.
The same membership and view rules apply when ordinary model settings resolve
to a certified namespace. Explicit local-search corpus-view flags require
certified membership; uncertified caches reject them.

An interrupted import or refresh remains unavailable to consumers. Resume an
import with the same local snapshot or the same bucket configuration and
snapshot ID. A conflicting-import error prints the pinned resume command.
Complete bucket downloads remain in the Hugging Face cache, so
native import can resume from that copy even after the publisher retires the
remote snapshot. An unfinished transfer may need to restart from the current
snapshot if its files have been retired. Resume refresh against its recorded
source target; a moving upstream branch cannot change that pending target.
The cache recovers committed vector writes and reuses them on retry.

Import accepts a fresh destination, the same incomplete import, or an identical
completed import. An occupied incompatible destination is preserved and
refused, including a bootstrap snapshot whose local copy has since been
refreshed. Use another cache root for a different source or calibration lineage
that would otherwise share a native address. Existing legacy caches need an
explicit reconciliation before they can be exported with certified coverage:

```bash
# Supply the original model/storage/source settings when they differ from defaults.
citemesh cache refresh --namespace NAMESPACE --adopt \
  --model-revision main --dataset-revision SOURCE_COMMIT
```

Adoption checks the supplied historical selectors and stored model fingerprint;
it never labels old vectors with the current moving model revision. A namespace
whose original settings cannot be established is preserved and refused.

The manual publisher uses these same refresh and export operations. Create a
private Storage Bucket first, then run from a source checkout:

```bash
hf buckets create OWNER/PRIVATE_INDEX --private
python scripts/publish_arxiv_index.py OWNER/PRIVATE_INDEX ./arxiv-index \
  --adopt-namespace NAMESPACE --device cuda
```

Use `--namespace` for an already certified corpus, or omit both namespace flags
to create a new configuration. Optional `--config NAME` overrides the published
configuration name; otherwise export derives `<profile>-<dimension>-<dtype>`
from the stored contract, such as `embeddinggemma-512-int8`. The output directory
must be new. The script resolves and pins the source, refreshes, exports, and
publishes one complete configuration. For upload ordering, see
[Embedding Cache Internals](../internals/embedding-cache.md#portable-format).
Use one publisher at a time. The bucket must already exist and be private.

Only the current snapshot of each configuration is retained remotely, so old
full indexes do not accumulate as repository history. Keep a local export if
you need an older snapshot later. Downloads and installed native caches remain
on each consumer until removed there. Int8 ranges are established once for a
new configuration and reused on refresh; a different calibration lineage needs
a different configuration and native destination.

## Corpus records

The dataset adapter accepts `id`, `paper_id`, or `paperId`, taking the first non-empty identifier. Stable source IDs preserve cache identity across updates. Without an ID, usable title or abstract text produces a content-derived ID; a row with neither fails and reports its source-row index.

`title` supplies the paper name. `abstract` supplies the text, with `summary` used when the abstract is empty or unusable. Optional fields include `authors`, `authors_parsed`, `categories`, `year`, `doi`, and `venue` / `journal_ref` / `journal`. Structured `authors_parsed` entries take precedence over free-text authors. Missing titles become `Unknown`; years fall back to the arXiv submission year when available. Arbitrary column mappings are unsupported.

## Corpus hydration and resume

The automatic hydration rules below apply to ordinary legacy corpus caches.
Certified portable indexes follow the explicit refresh and membership rules
above.

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

An int8 write outside those ranges warns once per run. Existing rows need the original ranges for decoding, so recalibration requires a forced rebuild. Candidates use float32 without calibration. Run `citemesh build --help` for the current storage flags and defaults.

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
