# How CiteMesh builds a graph

Every `citemesh build` runs the same eight stages. Strategies differ only in where candidates come from and how a pair of papers is scored; seed resolution, selection, layout, and export are shared. This page walks the pipeline end to end with the numbers that are actually in the code.

```text
  "arxiv:1706.03762"
          │
  ┌───────▼────────┐
  │ 1. SEED        │  core/paper_ids.py — normalize, resolve against S2
  │    RESOLUTION  │  404 + embedding strategy ⇒ free-text "query:" seed
  └───────┬────────┘
  ┌───────▼────────┐
  │ 2. CANDIDATE   │  strategies/candidates.py — references / citations /
  │    ACQUISITION │  recommendations on a ~1:2:1 budget, identity-merged
  └───────┬────────┘
  ┌───────▼────────┐
  │ 3. EMBEDDING   │  EmbeddingGemma. seed = retrieval query,
  │                │  candidates = retrieval documents. 512-d, fp32
  └───────┬────────┘
  ┌───────▼────────┐
  │ 4. CACHE       │  data/embedding_cache/ — SQLite metadata + HDF5 vectors,
  │                │  keyed by a namespace fingerprint. Warm runs skip encoding
  └───────┬────────┘
  ┌───────▼────────┐
  │ 5. RANKING &   │  rank candidates against the seed, cut to --max-papers
  │    SELECTION   │  (hybrid additionally reranks across both branches)
  └───────┬────────┘
  ┌───────▼────────┐
  │ 6. EDGE        │  re-encode the survivors with a SYMMETRIC prompt.
  │    SCORING     │  cosine gate → composite score → per-node cap
  └───────┬────────┘
  ┌───────▼────────┐
  │ 7. LAYOUT      │  visualization/render.py — Kamada-Kawai, community
  │                │  spreading, component packing, orientation, normalize
  └───────┬────────┘
  ┌───────▼────────┐
  │ 8. EXPORT      │  one graph payload → PNG / Plotly / dashboard /
  │                │  JSON / CSV / BibTeX / GraphML
  └────────────────┘
```

Stages 3, 4, and 6 only run for `--strategy embedding` and `--strategy hybrid`. The `recommendation` and `citation` strategies substitute a TF-IDF topical score (stage 5) and skip the model entirely.

## 1. Seed resolution

`normalize_paper_id` canonicalizes whatever you typed before any network call:

| Input | Canonical form |
| --- | --- |
| `doi:10.1038/Nature14539`, `https://doi.org/10.1038/Nature14539` | `10.1038/nature14539` — the `doi:` prefix is dropped and the name is lowercased, because DOI names are case-insensitive |
| `arxiv:1706.03762v5` | `arxiv:1706.03762` — a trailing `v<digits>` is stripped by `strip_arxiv_version` |
| `https://arxiv.org/abs/1706.03762`, `.../pdf/1706.03762.pdf` | `arxiv:1706.03762` — the `/abs/` or `/pdf/` segment and any `.pdf` suffix are peeled off first |
| `1706.03762`, a Semantic Scholar 40-hex ID | passed through unchanged; Semantic Scholar resolves it |

A URL is only rewritten when its host matches `arxiv.org` or `doi.org` (exact match or a subdomain). A value with no scheme but containing `/` is retried as `https://<value>` so `doi.org/10.x/y` still works. Everything else is returned trimmed and unchanged — CiteMesh does not guess.

`paper_identifier_aliases` then builds the alias set (raw ID, normalized ID, bare arXiv form, lowercased DOI) that the metadata cache and the identity registry in stage 2 both key on, so one paper reached through three different identifiers is one cache entry and one node.

**Free-text seeds.** The embedding strategy calls `client.get_paper(seed_id, raise_on_unavailable=True)`. A genuine HTTP 404 — Semantic Scholar has no such paper — returns `None`, and the builder logs `Using '<seed>' as text query` and synthesizes a seed node whose ID is `query:` plus the first 8 hex characters of `sha1(query_text)`. That is the only fallback. If instead the endpoint is *unavailable* (retries exhausted), `SemanticScholarUnavailableError` propagates and the build fails: an outage must never be silently reinterpreted as "you meant to search for that string". The other three strategies have no query path at all and fail on an unresolvable seed.

That `query:` digest is unrelated to the `<hash>` in output directory names, which is the first 8 characters of `sha256(seed_id)`.

**Knobs**

- the positional `PAPER` argument — any form in the table above, or free text with `--strategy embedding`
- `--refresh-paper-cache` — bypass persisted paper metadata for this run; a failed refresh keeps the cached entry

**Read the code**: `src/citemesh/core/paper_ids.py` (`normalize_paper_id`, `recognize_arxiv_identifier`, `strip_arxiv_version`, `paper_identifier_aliases`, `external_ids_from_canonical_paper_id`); `src/citemesh/strategies/embedding/builder.py` (`EmbeddingGraphBuilder.collect_papers`); `src/citemesh/strategies/embedding/records.py` (`_query_seed_id`).

## 2. Candidate acquisition

`strategies/candidates.py` is the shared acquisition layer. Every strategy fetches through it, so budget splitting, identity reconciliation, and outage policy behave identically regardless of which builder you chose.

**The budget split.** In candidates mode the pool size (`--candidate-pool-size`, default `400`) is divided by `_candidate_pool_budgets`:

```python
max_recommendations = min(100, max(1, pool_size // 4))
remaining = max(0, pool_size - max_recommendations)
max_references = min(100, (remaining + 2) // 3)
max_citations = max(0, remaining - max_references)
```

At the default 400 this is recommendations `100` (hitting the cap), references `100` (also capped), citations `200` — the ~1:2:1 split, weighted toward citations because forward citations are where newer follow-up work lives. The 100 caps on references and recommendations match Semantic Scholar's own per-request endpoint limits.

The arithmetic degrades sensibly at tiny budgets: a budget of `1` requests recommendations only `(0, 0, 1)`; `2` adds one reference `(1, 0, 1)`; `3` is the smallest budget that touches all three sources `(1, 1, 1)`. Recommendations are floored at 1 so the pool is never empty.

A free-text `query:` seed takes a different route: it runs one keyword search capped at `QUERY_SEED_SEARCH_LIMIT = 20`, then applies the recommendation budget to the single top hit as its anchor. References and citations are skipped — a synthetic seed has neither.

**Identity reconciliation.** Semantic Scholar, arXiv, and Crossref disagree about identity constantly, and S2 itself sometimes issues two records for one work. `IdentityRegistry` maps every alias to the set of canonical IDs claiming it, and `reconcile_paper_identity` collapses duplicates as they arrive. Three strong namespaces are tracked — `arxiv`, `doi`, `s2` — plus a weak `meta:` key built from title, year, and up to three author surnames, which is only formed when all three are present and the title is not a placeholder (`"unknown"`, `"untitled"`, `"n a"`, and similar are filtered).

The conflict rule matters: two records conflict when any namespace present on *both* sides has disjoint values — **except** when the sole disagreement is `s2` and the DOI or arXiv ID agree. An agreeing external identifier outranks a disagreeing opaque S2 hash, because that is the duplicate-record case. A DOI or arXiv disagreement is always irreconcilable. When a merge happens, the seed always wins; otherwise the first-inserted candidate survives and the later record's sources, seed relations, and aliases are folded into it.

**Outage policy.** Each source records one of three states — `complete` (returned papers), `empty` (succeeded, returned nothing), `unavailable` (retries exhausted) — into the pool's `source_status`, which reaches the exported graph as `candidate_source_status`. `require_available_candidate_source` raises `CandidateAcquisitionError` only when *every* attempted source is unavailable; a partial outage logs one warning and continues, because a graph built from citations alone is still a real graph, while a seed-only graph produced during a total outage would look plausible and be worthless.

Retry budgets are per capability, not global. `_FailureDomain` defines five independent budgets — `references`, `citations`, `recommendations`, `search`, `paper_metadata` — scoped to one collection. Reference-ID and full-reference retrieval share the `references` budget; the rest are separate. Once a domain exhausts its budget, later calls to that domain in the same collection are skipped immediately rather than retried again, while healthy domains continue untouched. A new collection starts with fresh budgets.

Each budget is 30 total attempts with exponential full jitter: the wait is `random.uniform(0, cap)` where `cap = 2.0 * 2^(attempt-1)`, clamped at 60 seconds. HTTP 429 doubles the base to `4.0`, and a numeric `Retry-After` header raises the *floor* of the wait to `min(retry_after, 300)` seconds. Any single wait of 30 seconds or longer is announced at the default log level. There is deliberately no overall deadline — the caps bound each delay, not total waiting time, because long retries let a resumable build finish. 404 returns `None` without retrying (except at a pagination offset above zero, where a 404 is treated as transient service inconsistency); 408, 5xx, and JSON decode failures retry; other 4xx, including rejected credentials, fail immediately.

**Knobs**

- `--candidate-pool-size` (candidates mode; default `400`)
- `--max-references` / `--max-citations` (citation and hybrid collection; defaults `25`/`25`, hybrid `12`/`45`)
- `--no-references` — skip reference lists entirely, at the cost of real bibliographic coupling
- `--refresh-reference-cache` — bypass persisted reference IDs for this run

**Read the code**: `src/citemesh/strategies/candidates.py` (`fetch_candidate_pool`, `fetch_candidate_source`, `require_available_candidate_source`, `IdentityRegistry`, `reconcile_paper_identity`, `CandidateSourceState`); `src/citemesh/strategies/embedding/builder.py` (`_candidate_pool_budgets`); `src/citemesh/services/semantic_scholar/retry.py` (`_jittered_backoff`) and `errors.py` (`_FailureDomain`).

## 3. Embedding

The default checkpoint is `unsloth/embeddinggemma-300m`, with `google/embeddinggemma-300m` as a license-gated fallback used only on default-revision loads. Both map to the same profile (`schema_token="embeddinggemma-v2"`), which requires Transformers >= 5.2 and declares `recommended_truncate_dim=512` out of `(768, 512, 256, 128)`.

**Three prompt roles, deliberately kept apart.** EmbeddingGemma is prompt-conditioned, so the same abstract encoded under different task prompts lands in different regions of the space. CiteMesh uses exactly three, defined by `EmbeddingTask`:

| Role | Prompt | Used for |
| --- | --- | --- |
| `retrieval-query` | `task: search result \| query: {text}` | the seed paper, and a free-text `citemesh search` query |
| `retrieval-document` | `title: {title} \| text: {abstract}` | candidate and corpus papers, for seed-to-paper ranking and local search |
| `graph-similarity` | `task: sentence similarity \| query: {text}` | the papers selected for the final graph, for paper-to-paper edge scores |

Retrieval is *asymmetric*: a query vector and a document vector are trained to be close when the document answers the query, which is not the same relation as "these two papers are about the same thing". Edge scoring needs the symmetric relation, so the selected papers are re-encoded under the STS prompt. To make it impossible to accidentally mix them, the two live in separate cache namespaces: the namespace string carries a `representation=` token (`retrieval-document-v1` vs `graph-similarity-v1`) and a distinct `formatter=` fingerprint, and the graph namespace is forced to float32 with no prefilter. Matching dimensions alone can therefore never make retrieval vectors eligible for symmetric scoring.

**Precision.** Weights load with `model_kwargs={"dtype": "auto"}` — the checkpoint's own dtype, neither forced to fp32 nor coerced down. Reduced-precision compute is bf16 only, and bf16 is *verified* rather than assumed: `_bf16_autocast_allowed` checks the profile's autocast device list, then native hardware support (CUDA `is_bf16_supported(including_emulation=False)`, CPU via `torch.cpu.get_capabilities()` for `avx512_bf16`/`amx_bf16`/`sve_bf16`, MPS requiring torch >= 2.13), and then actually enters the autocast context inside a `try`/`except` as a live smoke test. Anything that fails falls back to fp32. After loading, `_validate_loaded_model_precision` inspects live parameter and buffer dtypes: float16 is rejected outright everywhere, bfloat16 is rejected unless the verified bf16 path was the one selected, and any other dtype is rejected. fp16 is never used for compute, output, storage, or calibration.

**Dimensions and normalization, in that order.** The forward pass output is sliced to `truncate_dim` while still in compute dtype, cast to fp32, moved to numpy, and only then L2-normalized (with the norm floored at `1e-12`). SentenceTransformers' own encode-time normalization is disabled so that division never happens in bf16 — rounding a normalization in reduced precision is exactly how a cosine gate drifts.

**Device.** `resolve_embedding_device` resolves `auto` as cuda → mps → cpu. An explicit `--device cuda` or `--device mps` on a host without that backend raises rather than silently downgrading, and the MPS diagnostic distinguishes a torch build compiled without MPS from a built backend unavailable on this machine.

**Token window.** There is no hardcoded limit; the window is the loaded model's own `max_seq_length`. Before encoding, `warn_on_truncated_inputs` tokenizes with `truncation=False` and warns once per encode call when any input — prompt and special tokens included — exceeds it. The encoder still truncates; the warning exists so you know those embeddings represent only part of the text. Cached vectors reused without encoding do not repeat it.

**The encode proxy.** `_PrecisionEncodeProxy` wraps the SentenceTransformer and is constructed only when it has something to do. It runs every encode inside the combined TF32 + autocast context; it normalizes in fp32 afterward; on any exception it restores the eager inner transformer and retries the batch once (the lazy-compile recovery path); and on CUDA only, it tokenizes the next batch on a worker thread while the GPU runs the current one. Underneath, `text_batching.py` sorts inputs by an estimated length (`max(whitespace_tokens, len(text) // 4)`) into `--batch-size` chunks so similarly sized texts pad together, then restores the original order. Below one batch worth of texts, no bucketing happens.

**`--torch-compile`** compiles the *inner* Hugging Face transformer (`model[0].model`, or the legacy `auto_model` alias) — never the SentenceTransformer wrapper, which is not compile-compatible. CUDA and CPU pass `dynamic=True` for variable-length batches; CPU additionally enables `max_autotune` for GEMMs and scopes Inductor weight freezing around each encode call (Dynamo has to see freezing while capturing parameters — setting it as a backend option alone is too late). MPS gets neither, and defers compilation entirely while an arXiv corpus cache is still hydrating, resuming on warm-cache runs. Failures are non-fatal in two places: a wrap-time failure restores the eager module and logs why, and a call-time failure restores eager and retries the batch once before giving up. It is off by default; enable it when you will amortize the warm-up.

**Knobs**

- `--device` (`auto`/`cuda`/`mps`/`cpu`)
- `--truncate-dim` (`768`/`512`/`256`/`128`; omitted uses the profile's `512`)
- `--batch-size` / `-bs` (default `32`)
- `--torch-compile` / `--no-torch-compile`

**Read the code**: `src/citemesh/data/model_profiles.py` (the profile registry and the three formatter functions); `src/citemesh/strategies/embedding/text.py` (`EmbeddingTask`, `format_paper_for_embedding`); `precision.py` (`_PrecisionEncodeProxy`, `_model_floating_dtype_names`); `model_runtime.py` (`_bf16_autocast_allowed`, `_validate_loaded_model_precision`, `_maybe_compile_inner_transformer`); `runtime.py` (`resolve_embedding_device`); `src/citemesh/text_batching.py` (`encode_texts`, `l2_normalize_embeddings`, `warn_on_truncated_inputs`).

## 4. The cache

Encoding is the expensive part, so every vector is persisted. `data/embedding_cache/` stores metadata in SQLite and vectors in a resizable HDF5 matrix, one `metadata_<hash>.db` / `embeddings_<hash>.h5` pair per namespace, where `<hash>` is the first 12 characters of `sha256(namespace)`.

**The namespace fingerprint** is what makes cache reuse safe. It is a `::`-joined string of everything that could change what a vector *means*:

`model` · `revision` · `artifact` · `profile` · `representation` · `normalization` · `truncate_dim` · `storage_precision` · `binary_prefilter` · `calibration_sample_size` (int8 only) · `source_dtype` · `formatter` · `mode` (retrieval-document in candidates mode only)

Two of those deserve a note. `artifact` is an *immutable* checkpoint identity, not the model name you typed: a resolved 40-hex Hugging Face commit SHA where one is available, otherwise a sha256 digest over the selected inference-artifact manifest (config, tokenizer files, SentenceTransformers module definitions and their code, and exactly one active weight layout). Mutating a local checkpoint in place, or moving a branch to new contents, therefore selects a different cache while preserving the old one — flipping revision A → B → A reopens A's original vectors. If no reliable identity can be established at all, CiteMesh refuses persistent cache access rather than adopting an unidentified payload. `formatter` is a 16-hex digest over the profile name, the formatter's module and qualname, and its output against four fixed probe inputs, so changing a prompt template invalidates exactly the namespaces that used it.

There is no device token. CPU, CUDA, and MPS share a namespace whenever the compute dtype and every other contract match, which is what makes a cache portable between machines; a bf16 run and an fp32 run do not, because `source_dtype` differs.

**Two storage modes.** Candidates mode stores float32 with no prefilter — the pool is a few hundred vectors and calibration data does not exist yet. arXiv corpus mode stores int8 with a binary Hamming prefilter, because a hydrated split is millions of rows.

The int8 scheme is per-dimension asymmetric affine quantization, not a single global scale: with `starts = ranges[0]` and `steps = (ranges[1] - ranges[0]) / 255`, a value becomes `clip(floor((x - starts) / steps), 0, 255) - 128`. Decoding reconstructs each bucket's *centre* (`+128.5`), clamped at the calibration maximum, rather than its lower boundary — decoding the boundary would give every value a systematic low bias.

Calibration ranges are per-dimension min/max over a reservoir sample of `--calibration-sample-size` rows (default `2000`, seeded deterministically), computed in a separate prepass and persisted *before* the first int8 write. Raw writes into an int8 namespace with no ranges fail closed rather than bootstrapping ranges from whatever batch happened to arrive first. Ranges can never be replaced once int8 rows exist, because those same ranges are required to decode the stored bytes — that is why re-calibrating means `--force-rebuild-cache`, not an in-place fix. When more than 0.5% of a write's coordinates fall outside the ranges, CiteMesh warns once per run; that is a coordinate clipping rate, not a recall measurement, and it does not by itself require a rebuild.

The binary index is one bit per dimension (the sign, strictly `> 0`), packed into `ceil(dim/8)` bytes. A prefiltered search scans it in 65,536-row chunks, keeps the `top_k × --binary-rescore-multiplier` (default `8`) nearest rows by Hamming distance using a 256-entry popcount table, then exactly rescores *only those rows* against the dequantized int8 vectors. Final ranking always comes from the real vectors; the binary index is an accelerator, never the answer. float32 and un-prefiltered int8 searches score every row through the same chunked kernels.

**Incremental persistence.** Hydration encodes in `--batch-size` model micro-batches (default `32`) but flushes cache appends in bursts of 2048 records, matching the HDF5 chunk size, so long corpus runs are not dominated by SQLite and HDF5 resize overhead. Within a write, HDF5 data is flushed and fsynced *before* the SQLite rows that point at it are committed, so a durable row mapping never outlives its vector; an interrupted run is resumable rather than corrupt.

The on-disk contract, hydration and resume policy, cache-root layout, and maintenance commands are in [Caching & Data](caching.md). The crash-recovery invariants and the mixin structure are in [Embedding Cache Internals](../internals/embedding-cache.md).

**Knobs**

- `--storage-precision` (`int8` corpus / `float32` candidates)
- `--binary-prefilter` / `--no-binary-prefilter` and `--binary-rescore-multiplier`
- `--calibration-sample-size` (int8 only; changing it selects a new namespace)
- `--force-rebuild-cache` (plus `--overwrite-cache` in scripts) — the only supported way to re-encode a namespace

**Read the code**: `src/citemesh/data/embedding_cache/store.py` (`EmbeddingCache`, `search`), `quantization.py`, `search.py`, `constants.py`; `src/citemesh/strategies/embedding/fingerprint.py` (`_resolve_model_fingerprint`, `_resolve_formatter_fingerprint`); `src/citemesh/strategies/embedding/builder.py` (`_embedding_cache_namespace`).

## 5. Ranking and selection

The pipeline so far produced far more candidates than the graph will hold. Selection cuts the pool down to `--max-papers` (default `40`; hybrid `45`), seed included.

**Embedding strategy.** The seed is encoded once as a retrieval query. In candidates mode the whole pool is encoded as retrieval documents and ranked by raw cosine against that query vector; in corpus mode the query goes straight to the cache's top-k search over the hydrated corpus. Either way, ranked candidates are admitted in order until `max_papers` is reached, skipping anything that reconciles to the seed itself. Ties break deterministically by `(-score, paper_id, insertion_index)`, so the same inputs always produce the same graph.

**Hybrid strategy.** Hybrid collects two pools — citation-derived and semantic — and reranks the union against the seed with `_seed_relevance_score`:

```text
score = 0.62 · semantic + 0.16 · temporal + 0.14 · citation + 0.08 · bibliographic
```

where `semantic = 0.5 · (cosine + 1)` maps cosine into `[0, 1]`, and `citation = log1p(citations) / log1p(max_citations_in_pool + 1)` compresses the citation-count long tail. Two provenance bonuses then apply: **+0.10** when a candidate was discovered by *both* branches (agreement between an explicit citation link and semantic similarity is the strongest evidence available), and **+0.02** when it is citation-derived rather than semantic-only (a small thumb on the scale toward grounded evidence). The result is capped at 1.0.

`--max-semantic` (default `min(20, max_papers - 1)`) limits how many **semantic-only** papers may be added; overlap papers, being citation-derived too, do not count against it. Internally hybrid asks the embedding branch for up to `max(max_semantic, min(max_papers - 1, max_semantic × 3))` candidates so the reranker has room to discriminate. Setting `--max-semantic 0` disables the semantic branch entirely, at which point hybrid falls back to the citation strategy's scorer and threshold while keeping its own degree cap.

**Recommendation and citation strategies** never load the model. They build an `AbstractSimilarityIndex` over the selected papers in `prepare_graph_scoring` — a TF-IDF vectorizer with unigrams and bigrams, English stop words, `max_features=5000`, and sublinear term frequency — and use the cosine of the L2-normalized TF-IDF rows as the topical component of their composite score, alongside temporal, citation-impact, and bibliographic-coupling signals. Fewer than two papers with usable text produces an empty index and a topical score of 0.0 everywhere.

**Knobs**

- `--max-papers` / `-p`
- `--max-semantic` (hybrid only)
- `--similarity-threshold` / `-t` (recommendation and citation; default `0.2`)

**Read the code**: `src/citemesh/strategies/embedding/builder.py` (`_select_candidates_from_pool`, `_collect_corpus_cache_papers`); `src/citemesh/strategies/hybrid.py` (`_rank_candidates`, `_seed_relevance_score`, `HYBRID_SEED_RERANK_WEIGHTS`); `src/citemesh/strategies/similarity.py` (`AbstractSimilarityIndex`); `src/citemesh/strategies/base.py` (`deterministic_sort_key`).

## 6. Edge scoring

Selection decided which papers appear. Edge scoring decides which pairs are connected — a separate question, answered with a separate vector space.

**The 0.74 gate.** Every selected paper is re-encoded under the symmetric STS prompt, and a pair whose cosine falls below `--min-semantic-similarity` (default `0.74`) scores exactly zero in the embedding strategy. Be clear about what this number is not: it is not a probability, not a general relevance scale, and not transferable across models or dimensions. It was selected on 14 full real abstracts and validated on 10 independent ones, at EmbeddingGemma with STS formatting at 512 dimensions. On the combined fixtures it retains 17 of 21 related pairs while admitting 1 of 105 unrelated pairs; 0.72 retains 18 and admits 5. It buys precision and does miss real architectural relationships — Transformer/BERT among them. CiteMesh warns when the active profile is not EmbeddingGemma or the dimension is not 512, because the boundary is uncalibrated there. Evaluate labeled pairs before overriding it; a smaller dimension does not preserve the same boundary.

**The composite score.** Once a pair clears the gate, the embedding strategy combines four signals from `EmbeddingSimilarityConfig`:

```text
similarity = (0.5 · semantic + 0.2 · temporal + 0.2 · category) × author_factor      # capped at 1.0
author_factor = 1.5 if the papers share an author, else 1.0
```

The weights sum to 0.9 rather than 1.0, leaving headroom for the shared-author multiplier; `validate()` rejects any weight set totalling outside `(0.0, 1.0]`. `temporal` comes from `TemporalConfig.year_similarity`, a linear decay with a cliff: `1.0 - (Δyears / 5) × 0.8` up to five years apart, then a flat `0.1` beyond — so a same-year pair scores 1.0, a five-year gap scores 0.2, and anything older drops off a cliff to 0.1 rather than decaying smoothly. A missing year on either side scores 0.5. `category` is arXiv category overlap. Note the ordering: dates, categories, and shared authorship can only *modify* a score that already cleared the semantic gate. They can never create an edge on their own.

A final `should_create_edge` check requires `similarity > 0.1` before the pair reaches capping.

**Hybrid scores differently**, because it has provenance information the embedding strategy does not. Its gate is a disjunction — cosine >= `min_semantic_similarity` **or** non-zero bibliographic coupling — so two papers with a shared reference list can connect even when the model does not see them as similar. Weights then adapt to where the two papers came from, over `(embedding, temporal, citation, bibliographic)`:

| Both papers from | Weights | Rationale |
| --- | --- | --- |
| semantic only | `(0.6, 0.2, 0.1, 0.1)` | no citation evidence to lean on; trust the vectors |
| citation only | `(0.3, 0.3, 0.2, 0.2)` | bibliographic coupling is real evidence; spread the weight |
| mixed | `(0.4, 0.3, 0.2, 0.1)` | balanced |

Hybrid then applies asymmetric floors: any pair below **0.2** is rejected outright; a seed-incident pair needs **> 0.4**; every other pair needs **> 0.5**. The seed gets a lower bar because a graph whose seed is isolated is useless, while a weakly-supported edge between two peripheral papers is just noise. Between them, the disjunctive gate and the floors mean a shared-reference pair with weak topical and temporal evidence is still rejected.

**Per-node caps and seed-first reservation.** Dense graphs are unreadable, so each strategy caps node degree: `3` for recommendation and citation (fixed), `--top-k` (default `4`) for embedding, and `5` for hybrid. A 40-paper citation graph therefore has at most 60 edges.

`select_capped_undirected_edges` does the pruning. It canonicalizes each undirected edge to a sorted `(left, right)` key keeping the strongest duplicate, sorts, then greedily keeps an edge only when *both* endpoints are still under the cap. The sort key is what makes the seed behave well: it prepends a boolean that is `False` for seed-incident edges, so every edge touching the seed sorts ahead of every edge that does not, regardless of weight. The seed's strongest eligible neighbors are locked in before any other node competes for its budget. Remaining ties break on descending weight and then string comparison, so the pruning is fully deterministic. Capping only ever removes edges — it never invents an unsupported one, and a seed with only two eligible neighbors keeps two.

**Knobs**

- `--min-semantic-similarity` (embedding and hybrid; default `0.74`)
- `--top-k` / `-k` (embedding only; default `4`)
- `--similarity-threshold` / `-t` (recommendation and citation; default `0.2`)

**Read the code**: `src/citemesh/core/config.py` (`EmbeddingSimilarityConfig`, `TemporalConfig`, `HybridSimilarityConfig`); `src/citemesh/strategies/embedding/builder.py` (`compute_similarity`, `should_create_edge`); `src/citemesh/strategies/hybrid.py` (`compute_similarity`, `should_create_edge`); `src/citemesh/strategies/base.py` (`select_capped_undirected_edges`, `build_capped_undirected_graph`, `temporal_similarity`, `bibliographic_coupling`).

## 7. Layout and rendering

One layout is computed in Python and shared by every layout-based export, so the PNG, the Plotly HTML, the dashboard, and the JSON geometry all agree. `compute_layout(graph, iterations, layout_seed)` runs six stages.

**1. Canonicalize.** The graph is rebuilt with nodes and edges inserted in sorted order, so layout does not depend on dictionary insertion order.

**2. Detect communities.** `greedy_modularity_communities` (Clauset-Newman-Moore greedy modularity maximization, weighted) — not Louvain. A graph with no edges becomes one singleton community per node; a failure falls back to a single community containing everything. Communities are sorted by `(-size, member names)` for determinism.

**3. Convert similarity to distance.** Kamada-Kawai wants path lengths, not similarities, so each edge gets `layout_distance = 1 / (1e-6 + max(weight, 0))` — stronger similarity, shorter distance — then multiplied by `1.05` within a community and `1.42` across communities. That asymmetry is the anti-hairball term.

**4. Lay out.** The primary algorithm is **`networkx.kamada_kawai_layout`** on `layout_distance`, with `scale=0.9` and `center=(0.5, 0.5)`. Kamada-Kawai is stress majorization: it places nodes so Euclidean distances match the requested graph distances as closely as possible, and it is deterministic — no RNG. `nx.spring_layout` is used **only if Kamada-Kawai raises**, with `k = 0.8 / sqrt(N)` and `iterations` from `--spring-iterations`. That is the *only* place `--spring-iterations` is read; on the normal path the flag does nothing.

**5. Spread by community.** With two or more communities, a meta-graph of communities is built with summed inter-community weights. If that meta-graph is itself disconnected, scaffold edges (weight `0.15`) tie each isolated component to the largest one — without them, isolated groups drift to arbitrary extremes and collapse the useful graph area once the viewport is normalized. Community anchors come from a second spring layout over the meta-graph (`iterations = max(80, min(220, 60 × n_communities))`), normalized with padding `0.22` and clamped to radius `0.69`. Each community is then translated by `anchor × separation_scale − centroid`, where `separation_scale = 0.92 + min(0.32, 0.06 × (n_communities − 1))` — more communities, more separation, bounded.

**6. Perturb, pack, orient, normalize.** Every node gets a small Gaussian jitter (`σ = 0.02`) so the result looks organic rather than lattice-like. Disconnected components are then packed: each is normalized, oriented, scaled by `max(0.32, sqrt(size / largest_size))` with its bounding box floored at `0.28`, and laid out by greedy left-to-right shelf packing with `0.18` gaps against a target row width of `max(largest_width, sqrt(padded_area × 1.6))`. Orientation is a bounding-box heuristic, not PCA: if the vertical span exceeds the horizontal one, rotate 90° via `(x, y) → (y, −x)`, which suits wide screens and wide figures. Finally `normalize_layout_positions` centers the result and scales it uniformly (aspect-preserving) to roughly ±0.9 on the longer axis. For the static PNG, `_layout_viewport_limits` then fits the axis limits to the figure's physical aspect ratio, *expanding* whichever axis is proportionally too tight — it never crops.

**Determinism and `--seed`.** Kamada-Kawai, community detection, packing, orientation, and normalization consume no randomness. Three stages do, each with a built-in default so an unseeded run is still reproducible: the community-anchor spring layout (default `17`), the per-node perturbation (default `0`), and the spring fallback (default `42`). `--seed` overrides all three. Note that the community-anchor layout and the perturbation run on essentially every graph, so `--seed` changes the picture even though the primary algorithm is deterministic.

**Node sizes** come from `VisualizationConfig`, ranked by citation count (ties broken by node ID):

| Rank | Size |
| --- | --- |
| seed, ranked in the top 3 | `2500` (`seed_size`) |
| seed, ranked below 3 | `1000` — visible, not dominant |
| highest-cited non-seed | `2200` (`max_non_seed_size`) |
| non-seed ranks 1-2 | `1200 + (3 − rank) × 200` (`size_tiers["top_3"]`) |
| non-seed ranks 3-7 | `500 + (8 − rank) × 80` (`size_tiers["top_8"]`) |
| non-seed ranks 8-14 | `250 + (15 − rank) × 30` (`size_tiers["top_15"]`) |
| non-seed rank 15+ | `100` (`min_size`) |

Every node then adds `log10(citations + 1) × 100` and is clamped to `2500` (seed) or `2200` (non-seed). Edges map weight to appearance as `alpha = clamp(weight × 0.6, 0.3, 0.6)` and `width = max(0.3, weight × 1.5)`.

**Pyvis is the exception.** The `html` export hands nodes to vis.js with no coordinates at all and enables browser physics (`forceAtlas2Based`, `gravitationalConstant −50`, `springLength 100`, `timestep 0.35`, 150 stabilization iterations). It ignores the precomputed layout entirely and settles in the browser, which is why `--seed` does not affect it.

**Knobs**

- `--seed` — deterministic layout across runs
- `--spring-iterations` / `-i` — only reaches the spring fallback
- `--dpi` and `--theme` — PNG resolution and palette

**Read the code**: `src/citemesh/visualization/render.py` (`compute_layout`, `_detect_communities`, `_spread_layout_by_communities`, `_pack_disconnected_components`, `_orient_layout_horizontally`, `normalize_layout_positions`, `compute_node_sizes`, `visualize_graph`); `src/citemesh/visualization/ordering.py` (`canonicalize_graph_for_layout`); `src/citemesh/core/config.py` (`VisualizationConfig`).

## 8. Exports and the dashboard collection

**One payload feeds every format.** `GraphExporter.graph_payload()` builds a single versioned dict (`kind: "citemesh-graph"`, `schema_version: 1`) with `seed_id`, `meta`, `summary`, `nodes`, `edges`, and a `dashboard` block carrying the layout geometry, node order, and sizes. The layout is always computed, so even a bare `--export json` embeds dashboard geometry and that file can later be dropped into a dashboard through **Add Results**. Node enrichment adds external links, provenance, seed relation, and a seed-relevance score computed by personalized PageRank (`alpha=0.85`, personalized on the seed, edge weights honored). `to_json`, `to_csv`, `to_bibtex`, `to_graphml`, `to_plotly_html`, `to_dashboard_html`, and `to_interactive_html` all read from that same payload, which is why the formats never disagree about a node.

**The dashboard is a viewer, not a render.** `dashboard.html` embeds the payload and rebuilds the Plotly figure *client-side in JavaScript*, so the page can filter, select, re-highlight, import another result, and export CSV without a Python round trip. That design has a real cost: roughly a dozen algorithms exist in both languages by construction — `stableCurveDirection`, `selectDashboardLabelIds`, `normalizeDashboardEdgeStrengths`, `dashboardHoverText`, `buildFigureSpecFromPayload` (mirroring `_build_plotly_figure` including curved-edge shapes, marker `sizeref`, selection-halo geometry, and label clearance), `csvGuard`/`csvEscape` and the CSV column order, `seedSlug`, `dashboardNodeLabel`, `safeExternalUrl`, `normalizeCollectionPackage`/`hasCompleteDashboardGeometry`, and `upsertCollectionEntries`.

This duplication is deliberate and it is pinned by tests, not by discipline. `tests/test_visualization.py::test_exporter_dashboard_runtime_script_contracts` asserts that the emitted script contains each JS function definition and specific formula fragments interpolated from the Python constants, then executes `csvGuard`, `csvEscape`, `seedSlug`, `dashboardNodeLabel`, and `escapeHtml` in a real Node subprocess and checks exact outputs. `test_dashboard_labels_clear_selection_halos_after_import` runs `buildFigureSpecFromPayload` in Node and checks the label offsets against the Python halo math; `test_dashboard_highlights_fallback_to_path_order_and_use_theme_styles` pins the JS color and opacity math against the Python theme values; `test_dashboard_invalid_embedded_collection_keeps_current_graph_in_node` and `test_dashboard_snapshot_loads_latest_same_result_in_node` pin the client-side package validation against the server-side contract. Change one side without the other and these fail. (They skip when Node is not installed.)

**Collections.** Dashboard builds maintain two shared files at the collection root — the reusable `dashboard.html` viewer and the authoritative `dashboard.citemesh.json` package — plus one directory per seed at `out/<title-slug>-<hash>/`, where `<hash>` is the first 8 characters of `sha256(seed_id)`. Later builds reuse a directory by that hash, so correcting a paper's title does not move its exports. Results inside the package are keyed by `"<strategy>:<seed_id>"`: the matching entry is dropped and the new one prepended, so a different seed adds a result, the same seed with a different strategy adds a separate result, and the same seed and strategy replaces its slot. Writes are staged, backed up, published, and rolled back on failure under a 60-second file lock that lives beside the package rather than in the cache root, so changing `CITEMESH_CACHE_DIR` cannot break coordination between collection writers.

Standalone mode (`--export dashboard -o out/report.dashboard.html`) bypasses the collection entirely and writes one self-contained file with an empty collection bundle.

**Knobs**

- `--export` / `-e` — repeat for multiple formats, or `all`
- `--output` / `-o` — collection root, or an explicit `*.dashboard.html` for standalone mode
- `--theme` and `--include-timestamp`

**Read the code**: `src/citemesh/visualization/export/__init__.py` (`GraphExporter` and its `to_*` methods); `nodes.py` (`_enriched_nodes`, `_seed_relevance_scores`); `geometry.py`, `plotly_figure.py`, `links.py`, `keys.py`, `csv_.py`; `src/citemesh/visualization/dashboard/` (`contracts.py`, `payload.py`, `package.py`, `assets/dashboard.js`).

Full file naming, path-normalization rules, package schema, and sidecar contents are in [Output Artifacts](../reference/output-artifacts.md).
