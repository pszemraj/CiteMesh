# Caching and data

CiteMesh stores paper metadata, reference lists, and embeddings on disk for reuse between runs.

## Where it lives

The [environment settings](../reference/environment.md#platform-variables-used-for-cache-root-resolution) determine the cache root. `citemesh cache scan` prints its path and usage, including a migration hint if a legacy macOS cache exists.

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

Successful Semantic Scholar lookups persist one full metadata record under `papers/`, keyed by the returned S2 `paperId`. Requested identifiers and known arXiv/DOI identifiers point directly to that record. Refreshing metadata updates the canonical record; identifiers do not hold separate metadata copies. Later lookups check disk first, focal paper resolution included, and batch requests send only the IDs still missing. Failed and malformed responses are not cached.

Discovery listings are the exception: reference, citer, and recommendation responses carry each related paper's metadata inline, so they make no batch request and write nothing to `papers/`. The arXiv bibliography fallback, used when a focal paper's reference listing is empty or unavailable, still resolves its entries through the batch endpoint and the paper cache.

Paper entries have no TTL. `--refresh-paper-cache` bypasses persisted reads and replaces metadata, including citation counts, from successful responses; a failed fetch preserves the old entry. Unchanged title and abstract text reuses its embedding.

Reference IDs live in `references/` under the same no-TTL policy with their own `--refresh-reference-cache`, since cached metadata does not imply its references were fetched. Empty reference lists are cached explicitly so those papers stop costing requests; a first-page `paper not found` returns empty *without* caching, so a later lookup can recover. Corrupt or unusable payloads are rebuilt from the API. A focal paper's reference listing fetched for discovery refreshes its entry here only when it is complete, meaning the provider ran out of references before the fetch limit.

Local semantic search over a ready namespace uses cached vectors and metadata without fetching the corpus, though query encoding still uses the recorded model, which may need downloading. Uncached Semantic Scholar search and citation, reference, or recommendation discovery need network access; graph builds may also request provider metadata.

## Discovery checks

A normal build always checks upstream for the requested recommendations, citations, or references; immediate reruns behave the same way. Discovery lists are not cached. Each endpoint invocation checks upstream and uses the returned order directly. Reference, citer, and recommendation requests ask for the default paper fields of every row; only the later reference-ID hydration of selected papers requests the minimal `paperId` projection.

The check preserves upstream membership and ordering among resolvable papers. Candidates come straight from the response rows, with the first row kept for each paper. A row without an ID or usable metadata is omitted while the other papers are retained; each listing logs its dropped rows once, as a warning only when more than half of that listing's rows are dropped and at debug level otherwise.

A failed page or failed `all-cs` fallback stops the required acquisition without replacing existing outputs. A request that starts with the `recent` recommendation pool checks `all-cs` if the first response contains no resolvable papers. Both checks must succeed before returning an empty recommendation list. Hybrid's [candidate acquisition](how-it-works.md#2-candidate-acquisition) rules select which provider pool to query first. A reference listing that reached the fetch limit never overwrites the complete reference-ID cache.

Hybrid needs a retrieval-document vector for every pool member:

- In `candidates` mode the pool is the focal paper's references and citers plus up to 200 recommendations, and their vectors are kept in the embedding cache's candidate namespace (see [Embedding namespaces](#embedding-namespaces)). A warm rebuild encodes only the focal paper's query-role vector. The first build encodes the whole pool, at most about 2,200 title-and-abstract texts at the default 1000-paper limits, dominated by the citers of a heavily cited focal paper.
- In the corpus modes the namespace is the corpus itself and takes no provider rows. Stored corpus vectors are read for corpus rows; the references, citers, and recommendations outside the installed index are encoded in memory on every build and never stored, so they do not change the corpus namespace or its membership.
- In every mode, the selected papers' symmetric graph-similarity vectors are written to their own cache.

Debug-level output (`--verbose` or `--log-level debug`) reports that discovery IDs were rechecked upstream separately from reused paper metadata and reference IDs; normal output does not. `--refresh-paper-cache` and `--refresh-reference-cache` control those two enrichment caches; neither suppresses the discovery check.

## Embedding namespaces

A namespace combines model, artifact identity, [representation](../reference/embedding-runtime.md#task-specific-vector-spaces), dimension, formatter, and storage settings. The formatter includes the [text normalization](../reference/embedding-runtime.md#task-specific-vector-spaces) applied to titles, abstracts, and queries. The artifact identity is a resolved commit SHA or a digest of the inference-artifact manifest, so changing a checkpoint selects a different cache. SQLite and HDF5 store each namespace as a pair; [physical layout](../internals/embedding-cache.md#physical-layout) describes the files.

Local custom model fingerprints include declared Python modules, their package initializers, and transitive relative imports. Changing those files selects a new namespace. Fingerprinting parses the code without executing it.

- Candidate mode additionally tags its retrieval namespace `mode=candidates` so S2 candidates never mix with corpus hydration.
- No device or compute-dtype token in the namespace: CPU, CUDA, and MPS resolve the same one whenever the other contracts match, so a corpus built on a bf16 GPU is read directly by an fp32 host rather than re-encoded. An auto-resolved compute dtype is provenance rather than identity - like the attention backend, TF32, and `--torch-compile`, it shifts numerics slightly without changing what a vector means. The dtype that created a namespace is still recorded, but it does not decide compatibility on reopen.
- Changing model, revision, profile, dimension, storage precision, int8 calibration size, or the text formatter selects a different namespace. Switching back to the earlier settings reopens the original vectors.
- The binary prefilter is not part of the namespace: toggling it reuses the same vectors, ranges, and hydration state, rebuilding or dropping only the derived index.
- A vector is re-encoded only when the paper is new to the namespace or its input text changed. Metadata-only updates refresh the SQLite row, not the vector.

Crash safety, the replacement journal, flush ordering, and locking are in [Embedding Cache Internals](../internals/embedding-cache.md).

## Portable corpus indexes

### Import and selection

A portable index installs precomputed retrieval-document embeddings into the native cache. Install the `embeddings` extra first. Import and export copy the stored int8 or float32 values without loading an encoder; query encoding and the selected graph papers' symmetric embeddings still use the recorded model.

Before starting a compatible arXiv corpus from an empty cache, builds show one INFO reminder with the public index import command. After import, rerun the same command. Reused caches and resumed hydration skip the reminder. The reminder leaves the build running and requires no network check.

```bash
# Import a local export, or one configuration from a Storage Bucket.
citemesh cache import --path ./arxiv-index
citemesh cache import --bucket pszemraj/citemesh-arxiv-indexes --config embeddinggemma-512-int8
citemesh cache scan

# Default model settings select the public index automatically.
citemesh search "long-context language models" --semantic-source arxiv-corpus
citemesh build arxiv:2404.08801 --strategy hybrid --semantic-source both --export dashboard
citemesh config set defaults.semantic_source both

# Select a different installed index by the namespace printed by import or scan.
citemesh search "long-context language models" --cache-namespace NAMESPACE --corpus-size 50000

# Export a completed corpus into a new directory.
citemesh cache export --namespace NAMESPACE --output ./arxiv-index
```

Import prints the installed namespace and copyable search and build commands. It resolves your settings the way those commands will: built-in defaults overlaid with `config.toml`. When those settings do not select the index, for example because of a different model, dimension, or dataset, the printed commands add `--cache-namespace`.

Where the index comes from:

- `--path` and `--bucket` are alternatives.
- Bucket imports use the selected configuration, or the catalog's default, and download only its declared files.
- Omit `--snapshot` to install the catalog's current snapshot; an explicit snapshot requires `--config`.
- Snapshot identity identifies the portable files; the manifest separately records the immutable model and upstream dataset revisions. The upstream source revision is provenance, not namespace identity. Each configuration can have different coverage and freshness.
- Publication needs a Hugging Face token with write access to the bucket, and import from a private bucket needs one with read access. Set [`HF_TOKEN`](../reference/environment.md#other-respected-variables) or save a token with `hf auth login`; public bucket reads need no token.

An installed index whose embedding contract differs from the running version's, for example a different model revision or text formatter, is refused rather than silently re-encoded. A republished index with the new contract installs beside it as a new namespace; remove the old one as described in [Inspecting and clearing](#inspecting-and-clearing).

How `--cache-namespace` behaves:

- It selects existing files and pins model, profile, formatter, dimension, storage precision, calibration sample size, dataset, and split. Conflicting flags fail; device, batch size, scoring, and result limits remain configurable.
- In `--mode auto`, installed-index readiness/compatibility errors stay errors under explicit or ordinary namespace selection.
- Import does not change saved defaults.
- Hybrid's handling of `--semantic-source both` under `--cache-namespace` is described in [Strategies](strategies.md#hybrid).

Corpus views of an installed index:

- `--corpus-size N` ranks only the newest N active members at the completed revision; `--all-corpus` uses every installed active member.
- Chronology ties prefer later source rows; undated papers fill remaining places in source order.
- A smaller view changes neither stored coverage nor membership. A capped index cannot supply a larger view or claim full coverage.
- Focal paper lookup is independent of neighbor eligibility, so an older focal paper can search newer papers.
- The same membership and view rules apply when ordinary model settings resolve to a certified namespace. Explicit local-search corpus-view flags require certified membership; uncertified caches reject them.

### Refresh

Installed indexes are manually refreshed. Build and search use their completed source revision without checking or scanning the upstream dataset:

```bash
citemesh cache refresh --namespace NAMESPACE
citemesh cache refresh --namespace NAMESPACE --dataset-revision SOURCE_COMMIT
```

A new refresh resolves its source revision once, compares full normalized records, and encodes only new or changed document text:

- Metadata-only changes retain their vectors and need no encoder.
- Existing int8 calibration ranges stay fixed.
- Removed source papers keep their physical vectors but leave active membership, retrieval, and export.

How refresh reads the source depends on what the namespace holds:

- Reconciling a complete corpus with a newer revision, such as an index imported from a bucket, streams the source once. It still reads every row; only the encoding is incremental.
- Every pass that has to process the whole source downloads the pinned revision instead: a namespace without a complete corpus, a pending or resumed refresh, first int8 calibration, and the first `--adopt` of a namespace. `datasets` prepares the revision as Arrow in its own cache, and CiteMesh normalizes the rows in parallel on half the logical CPUs once a source has 50,000 rows. For the full arXiv snapshot, expect about 5 GB of prepared Arrow per revision in the `datasets` cache, which later passes reuse. The normalized rows, about 5 GB more, go to a temporary directory under the CiteMesh cache root that is removed when the pass ends, so every pass normalizes with the running code.
- One pass serves calibration and encoding: the int8 calibration sample is drawn from the prepared rows with the same seeded reservoir build hydration uses.

Repeated source IDs are resolved the same way whether the source is streamed or downloaded, before calibration, encoding, or newest-N selection:

- The most recent valid `update_date` wins, followed by the highest arXiv version number and its creation time.
- When conflicting records have equal or unavailable revision information, a stable ordering of their normalized content breaks the tie and CiteMesh warns once with the number of ambiguous comparisons. This tie-break is deterministic; it does not establish freshness.
- Unparseable timestamps rank below valid ones, which are compared in UTC.
- The winning record retains its original source position. Equally recent, identical records retain the first occurrence.

### Resume and adoption

An interrupted import or refresh remains unavailable to consumers.

- Resume an import with the same local snapshot or the same bucket configuration and snapshot ID. A conflicting-import error prints the pinned resume command.
- Complete bucket downloads remain in the Hugging Face cache, so native import can resume from that copy even after the publisher retires the remote snapshot. An unfinished transfer may need to restart from the current snapshot if its files have been retired.
- Resume refresh against its recorded source target; a moving upstream branch cannot change that pending target.
- The cache recovers committed vector writes and reuses them on retry.
- A resumed refresh keeps the membership it recorded and skips those papers without re-checking their text. A member is processed again only when its source position no longer holds the selected row for that paper, or when recovery found its vector missing.

Import accepts a fresh destination, the same incomplete import, or an identical completed import. An occupied incompatible destination is preserved and refused, including a bootstrap snapshot whose local copy has since been refreshed. Use another cache root for a different source or calibration lineage that would otherwise share a native address. Existing legacy caches need an explicit reconciliation before they can be exported with certified coverage:

```bash
# Supply the original model/storage/source settings when they differ from defaults.
citemesh cache refresh --namespace NAMESPACE --adopt \
  --model-revision main --dataset-revision SOURCE_COMMIT
```

Adoption checks the supplied historical selectors and stored model fingerprint; it never labels old vectors with the current moving model revision. Without `--dataset-revision`, a namespace whose build hydration recorded a source commit is adopted at that commit, and an older namespace at the current `main`. A namespace whose original settings cannot be established is preserved and refused.

### Publishing

The manual publisher uses these same refresh and export operations. Create a Storage Bucket first (add `--private` if only you should import from it), then run from a source checkout:

```bash
hf buckets create OWNER/INDEX
python scripts/publish_arxiv_index.py OWNER/INDEX ./arxiv-index \
  --adopt-namespace NAMESPACE --device cuda
```

Use `--namespace` for an already certified corpus, or omit both namespace flags to encode a new one. Optional `--config NAME` overrides the published configuration name; otherwise export derives `<profile>-<dimension>-<dtype>` from the stored contract, such as `embeddinggemma-512-int8`. The output directory must be new. The script resolves and pins the source, refreshes, exports, and publishes one complete configuration. For upload ordering, see [Embedding Cache Internals](../internals/embedding-cache.md#portable-format). Use one publisher at a time.

Publishing to an existing configuration name replaces its snapshot, including when the embedding contract or int8 calibration changed, as after a formatter change. Re-run `citemesh cache import` to install the replacement. A changed contract maps to a new native namespace beside the old one; a replacement with the same contract maps to the installed namespace, which import never overwrites, so delete that namespace's `.db` and `.h5` first ([Inspecting and clearing](#inspecting-and-clearing)).

The publisher writes `README.md` when it is missing and regenerates it on later publications while it still matches the generated overview. Once you edit it, adding links, images, or diagrams, later publications preserve it, along with assets outside `variants/`. Imports read the catalog, selected manifest, and its declared data files, without reading or validating documentation.

Only the current snapshot of each configuration is retained remotely, so old full indexes do not accumulate as repository history. Keep a local export if you need an older snapshot later. Downloads and installed native caches remain on each consumer until removed there. Int8 ranges are established once per native namespace and reused on refresh.

## Corpus records

The dataset adapter accepts `id`, `paper_id`, or `paperId`, taking the first non-empty identifier. Stable source IDs preserve cache identity across updates. Without an ID, usable title or abstract text produces a content-derived ID; a row with neither fails and reports its source-row index.

`title` supplies the paper name. `abstract` supplies the text, with `summary` used when the abstract is empty or unusable. Optional fields include `authors`, `authors_parsed`, `categories`, `year`, `doi`, and `venue` / `journal_ref` / `journal-ref` / `journal`. Structured `authors_parsed` entries take precedence over free-text authors. Missing titles become `Unknown`; years fall back to the arXiv submission year when available. Arbitrary column mappings are unsupported.

The `doi` field accepts a bare DOI, a `doi:` prefix, or a DOI URL. It uses the first token if valid, then falls back to a DOI-based paper ID.

## Corpus hydration and resume

The automatic hydration rules below apply to ordinary legacy corpus caches. Certified portable indexes follow the explicit refresh and membership rules above.

arXiv-corpus mode hydrates the full selected `--dataset-split` by default - "all of `train`", not every split the dataset publishes. `--corpus-size N` opts into the N newest submissions by arXiv ID; on a cold build the cap bounds what gets embedded, not how many rows are scanned to establish that order. Streaming ranking drains the source before encoding begins. An indexable non-streaming dataset instead scans its identifier columns. A slice such as `train[:2%]` limits the population ranked. Rows without parseable arXiv IDs fill any shortfall in source order, with a warning; when none can be parsed, the cap selects the first N rows.

A hydration from an empty namespace, including one emptied by `--force-rebuild-cache`, resolves `--dataset-source` to its current commit once and records that commit in the namespace before writing the first row; every read of the source during that hydration, its int8 calibration sample and row count included, uses the recorded commit.

An interrupted run scans the selected source IDs at the recorded commit and encodes only missing papers, provided the recorded source and split match, the recorded cap is compatible, and the namespace is self-consistent - equal SQLite and embedding row counts, at least one row, and persisted calibration ranges under `--storage-precision int8`. Anything else falls through to a full rebuild. Upstream commits published in the meantime never reach a resumed run, so one hydration never mixes rows from two revisions. An interrupted cache whose recorded commit can no longer be loaded stops the build with an error instead of resuming, and so does an unpinned cache from an older version that is left incomplete (an interrupted older build, or vectors lost in a crash), since it records no commit to resume at. Either can be finished at the current upstream commit and its recorded size with `citemesh cache refresh --adopt --namespace NAMESPACE --dataset-revision main` plus the model and storage options the build used (the error prints the full command), which compares full records and encodes only papers whose text is missing or changed; `--force-rebuild-cache --overwrite-cache` rebuilds it from scratch instead.

Metadata migrations of older corpus caches are described under [Older caches](#older-caches).

A complete cache stays at its recorded commit: builds reuse it without checking upstream, and moving it to a newer revision is the job of [`citemesh cache refresh`](#refresh), which records the new commit when it completes. A cache that a build hydrated is not yet certified ([#31](https://github.com/pszemraj/CiteMesh/issues/31)), so its first move is `citemesh cache refresh --adopt --namespace NAMESPACE --dataset-revision main`, which certifies it and advances it in one pass. A complete cache hydrated before commits were recorded has no pin, so on a non-sliced split it still checks upstream growth as below; a full-split rescan after the published row count changes, and the one-time identity migration of older positional IDs, record the commit they read before they start, after which the cache is pinned like any other. Without a published row count the scan reads the latest revision and records nothing, so the cache stays unpinned:

| Upstream state | Reuse or update |
| --- | --- |
| Published row count matches the last verified count | Reuse memoized metadata without scanning source IDs. |
| Row count changed | Scan current IDs. Full-split caches append missing papers; capped caches compare the newest-N selection, including backfilled submissions. |
| No published row count | Scan IDs each build. If this optional check cannot read the source, keep the complete cache usable and retry next time. |
| Rows removed upstream | Preserve historical vectors, warn with the number of cached papers the source no longer lists, and record the verified source count separately from stored vector count. |

Rows are added, never evicted automatically:

- A capped cache can therefore store more than `--corpus-size` papers; a build's search view still uses only the newest N eligible cached papers.
- A later count change triggers another scan even if it equals the number of stored vectors.
- Explicit source IDs remain authoritative, so rebuild after ID remapping.
- Same-count replacements and revised abstracts are not detected by the count check; rebuild for those ([issue #13](https://github.com/pszemraj/CiteMesh/issues/13)).

Changing the cap never costs you vectors you already have:

- A larger `--corpus-size` - or `--all-corpus` - extends the same namespace in place, encoding only the newly selected papers.
- A smaller one limits retrieval to the newest N cached papers before scoring, while reusing the existing rows and leaving the recorded hydration cap unchanged. Chronology ties prefer later stored rows; undated papers fill any shortfall in stored order.
- An interrupted extension keeps the cache complete at its recorded size rather than discarding it.
- The recorded cap is a floor, not an inventory - a recency refresh or upstream shrink can retain historical rows past the current selection. Clearing or rebuilding is needed only when you want the stored rows themselves to match that selection, not to limit a build's search view.

Of the corpus flags, only a changed `--dataset-source` or `--dataset-split` clears, and that automatic rebuild empties only the retrieval namespace; `--force-rebuild-cache` is the path that also discards graph-similarity vectors. A failed load stops the build before anything is replaced, and local search refuses a source mismatch outright. A failed storage inspection surfaces the paths and the original error rather than reading as an empty cache.

A model-fingerprint mismatch is not one of those conditions. When a hydrated corpus was built under a different fingerprint than the active model's, CiteMesh stops and names both fingerprints, the rows and on-disk size at stake, and the remedy - `--force-rebuild-cache`, plus `--overwrite-cache` for scripts, or `citemesh cache clear`. Candidate-pool and graph-similarity namespaces re-encode in seconds and still clear silently; corpus hydration metadata is what earns a namespace the protection.

Hydration compatibility uses dataset source, split, and cap, not the recorded commit; the commit decides which revision a resume, an in-place cap extension, a metadata migration, or an adoption without `--dataset-revision` reads, and whether builds check upstream growth. A sliced `--dataset-split` skips growth checks, and the same slice expression refers to the same records only at the same commit.

### Quantization

Corpus storage uses per-dimension affine int8 values. The optional Hamming prefilter keeps `top_k * binary_rescore_multiplier` rows, and at least 512, for exact vector rescoring, so a search with a small `--limit` still rescores 512 rows. Calibration ranges come from a reservoir sample and are persisted before the first int8 write.

An int8 write outside those ranges warns once per run. Existing rows need the original ranges for decoding, so recalibration requires a forced rebuild. Candidates use float32 without calibration. Run `citemesh build --help` for the current storage flags and defaults.

## Older caches

Caches written by earlier layouts are read or reported as follows, so none needs a manual rebuild to keep working.

- **Version 2 paper caches** stay readable: an alias copy written under a requested identifier supplies its canonical paper ID, and lookups read the canonical record.
- **Dtype-keyed embedding namespaces** (older `.db` and `.h5` pairs) remain separate and appear as reclaimable schema-3 namespaces in `citemesh cache scan`. After confirming the replacement cache works, delete the matching `metadata_<namespace>.db` and `embeddings_<namespace>.h5` files to reclaim their space. `citemesh cache clear` also reclaims them, but deletes every namespace.
- **Legacy macOS cache**: `citemesh cache scan` reports it with a migration hint. For a full reset, remove it separately from the current cache root.
- **Legacy corpus caches** need an explicit [adoption](#resume-and-adoption) before they can be exported with certified coverage.
- **Corpus metadata migrations** are versioned. Each rescans the selected source split and updates cached publication years, DOIs, and venues in place while preserving vectors and corpus membership.
- **Summary adapter migration**: it restores usable `summary` text that an empty `abstract` previously hid, re-encoding only affected rows. Older positional IDs remain unchanged and also match their equivalent content identity; source membership is reconciled once without rewriting unchanged vectors. Summary restoration matches the old adapter's exact output before assigning an updated identity; if discarded summaries made several distinct anonymous papers indistinguishable, the migration reports that a rebuild or stable source IDs are required.

## Inspecting and clearing

```bash
# usage by section
citemesh cache scan
citemesh cache clear --yes --reason "manual local reset"
```

`cache scan` reports section totals and then each standard embedding namespace's ID, files, bytes, and status, reading SQLite metadata without loading a model. An installed corpus's status lists its dimension and stored dtype, active and physical row counts, source revision, coverage, and origin when recorded, ending in its state:

- `ready`: the import or refresh finished and membership is certified, so search, builds, and export accept it.
- `pending import` or `pending refresh`: an interrupted operation that blocks search, builds, and export until you [resume it](#resume-and-adoption).
- `membership uncertified`: source membership is not yet certified, so an explicit `cache refresh` (or [adoption](#resume-and-adoption) for a legacy cache) must finish before the corpus can be searched, viewed, or exported.
- `Current schema`: a namespace with no installed-corpus state; this reports schema reachability, not vector health. Older layouts show as `Legacy dtype-keyed namespace` or `Legacy schema vN`.
- `Metadata busy` or `SQLite recovery pending`: another operation holds the namespace or an interrupted one needs write access to recover, so scan again or resume the import or refresh.
- `Metadata unavailable`, `Corpus metadata unavailable`, or `Orphaned artifact (metadata database missing)`: inspect the files before removing them.

`cache clear` deletes every entry under the cache root except `config.toml`, its lock, and `.locks/`. It fails without deleting anything while a cache operation is live, and it cannot interrupt a pending config write. To drop a single namespace instead, delete the matching `.db` and `.h5` in `embeddings/`.

> [!CAUTION]
> `cache clear` and `--force-rebuild-cache` are irreversible. A forced rebuild clears both namespaces for the resolved model contract and re-encodes from scratch, which on a hydrated corpus is a long GPU job. Both prompt interactively on a TTY and refuse outright without one, so `--yes` and `--overwrite-cache` are how a script approves them; `--cache-overwrite-reason "<text>"` records a rationale that otherwise logs as `reason=unspecified`.

A rewritten HDF5 file can grow in bytes while its row count stays fixed: compressed rows leave holes and CiteMesh does not compact automatically. That is not lost vectors or duplicated rows.

For a full reset including configuration, stop active CiteMesh processes and remove the root printed by `citemesh cache scan`. Remove any reported legacy cache separately.

Flag syntax and defaults: [CLI Usage](cli.md). Variables: [Environment Variables](../reference/environment.md).
