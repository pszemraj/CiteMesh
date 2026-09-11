# How CiteMesh builds a graph

Every `citemesh build` runs the same eight stages, and this page walks them end to end with the numbers that are in the code. Strategies differ only in where candidates come from and how a pair is scored; the rest is shared. Flag contracts live in the [CLI guide](cli.md#flag-reference) — this is the mechanism behind them.

```text
  seed ID, or free text with --strategy embedding
        │
        ▼
  1. Seed resolution → 2. Candidate acquisition → 3. Embedding → 4. Cache
                                                                     │
  8. Export ← 7. Layout ← 6. Edge scoring ← 5. Ranking & selection ◄──┘
```

Stages 3 and 4 run only for `--strategy embedding` and `--strategy hybrid`; the other two strategies never load the model and score edges from TF-IDF instead of embeddings.

## 1. Seed resolution

`normalize_paper_id` canonicalizes what you typed before any network call: a `doi:` prefix is dropped and the name lowercased (DOI names are case-insensitive), an arXiv URL or trailing `v5` collapses to `arxiv:1706.03762`, and a bare arXiv number or 40-hex S2 ID passes through for S2 to resolve. URLs are rewritten only on `arxiv.org` and `doi.org` hosts; anything else comes back trimmed and unchanged, because CiteMesh does not guess. `paper_identifier_aliases` then builds the alias set the metadata cache and stage 2's identity registry both key on, so one paper reached through three identifiers stays one node.

The embedding strategy is the only one with a free-text path. A genuine HTTP 404 means S2 has no such paper, so the builder synthesizes a seed whose ID is `query:` plus the first 8 hex of `sha1(query_text)`. An *unavailable* endpoint (retries exhausted) raises instead — an outage is never silently reinterpreted as a search.

**Knobs:** the positional `PAPER`; `--refresh-paper-cache`.

**Read the code:** `src/citemesh/core/paper_ids.py`.

## 2. Candidate acquisition

Every strategy fetches through `strategies/candidates.py`, so budgets, identity reconciliation, and outage policy are identical across builders. In candidates mode the pool (`--candidate-pool-size`, default `400`) splits roughly 1:2:1: recommendations `100`, references `100`, citations `200` at the default. The two 100s are Semantic Scholar's own per-request endpoint caps, and the remainder goes to citations, where newer follow-up work lives. A free-text `query:` seed instead runs one keyword search capped at 20 hits and applies the recommendation budget to the top hit.

S2, arXiv, and Crossref disagree about identity constantly, and S2 sometimes issues two records for one work, so `IdentityRegistry` collapses duplicates as they arrive across `arxiv`, `doi`, `s2`, and a weak title/year/author key. Two records conflict when a namespace on *both* sides has disjoint values — except when the sole disagreement is `s2` and the DOI or arXiv ID agree, which is the duplicate-record case. On a merge the seed wins.

Each source ends `complete`, `empty`, or `unavailable`, which reaches the exported graph as `candidate_source_status`. The build fails only when *every* attempted source is unavailable; a partial outage warns and continues, because a graph from citations alone is still real while a seed-only graph would look plausible and be worthless. Retry budgets are per capability and per collection: 30 attempts each, full-jitter backoff, no overall deadline, because long retries let a resumable build finish.

**Knobs:** `--candidate-pool-size`; `--max-references` / `--max-citations` (defaults `25`/`25`, hybrid `12`/`45`); `--no-references`; `--refresh-reference-cache`.

**Read the code:** `src/citemesh/strategies/candidates.py`.

## 3. Embedding

The default checkpoint is `unsloth/embeddinggemma-300m` (`google/embeddinggemma-300m` is the license-gated fallback), at 512 dimensions out of `(768, 512, 256, 128)`.

EmbeddingGemma is prompt-conditioned: the same abstract under different task prompts lands in a different region of the space. CiteMesh keeps three roles apart — `retrieval-query` for the seed, `retrieval-document` for candidate and corpus papers, and symmetric STS for graph edges — because retrieval is *asymmetric*: "does this document answer this query" is not the relation "are these two papers about the same thing". Retrieval and graph vectors therefore get separate cache namespaces, so matching dimensions alone can never make retrieval vectors eligible for symmetric scoring.

Precision is verified rather than assumed: bf16 compute is used only after the profile, the hardware, and a live autocast probe all agree, fp16 is rejected outright, and normalization runs in fp32 after truncation — rounding a normalization in reduced precision is how a cosine gate drifts. The per-device matrix, device resolution, attention selection, and the `--torch-compile` path are in [Embedding Runtime](../reference/embedding-runtime.md).

**Knobs:** `--device`, `--truncate-dim`, `--batch-size`, `--torch-compile`.

**Read the code:** `src/citemesh/strategies/embedding/precision.py` and `model_runtime.py`.

## 4. The cache

Encoding is the expensive part, so every vector is persisted: `data/embedding_cache/` holds metadata in SQLite and vectors in a resizable HDF5 matrix, one pair per namespace.

The namespace fingerprint is what makes reuse safe: it joins every token that could change what a vector *means*, from the model and its `artifact` identity through representation, truncate dim, source dtype, formatter, and storage contract. `artifact` is immutable — a resolved commit SHA, else a digest over the inference-artifact manifest — so mutating a checkpoint in place selects a different cache instead of poisoning the old one. There is no device token: CPU, CUDA, and MPS share a namespace when every other contract matches, which makes a cache portable — but a bf16 and an fp32 run differ in `source_dtype`.

Candidates mode stores float32 unfiltered: a few hundred vectors, no calibration data yet. Corpus mode stores per-dimension affine int8 plus a Hamming prefilter that keeps `top_k × --binary-rescore-multiplier` (default `8`) nearest rows and exactly rescores only those, so final ranking still comes from real vectors. Its calibration ranges (a reservoir sample of `--calibration-sample-size` rows, default `2000`) are persisted before the first int8 write and can never be replaced while int8 rows exist, which is why re-calibrating means `--force-rebuild-cache`.

Hydration, the on-disk contract, and maintenance commands are in [Caching & Data](caching.md); crash-recovery invariants in [Embedding Cache Internals](../internals/embedding-cache.md).

**Knobs:** `--storage-precision`, `--binary-prefilter` / `--binary-rescore-multiplier`, `--calibration-sample-size`, `--force-rebuild-cache`.

**Read the code:** `src/citemesh/data/embedding_cache/` and `src/citemesh/strategies/embedding/fingerprint.py`.

## 5. Ranking and selection

Selection cuts the pool to `--max-papers` (default `40`; hybrid `45`), seed included.

The embedding strategy encodes the seed as a retrieval query and ranks by raw cosine — over the encoded pool in candidates mode, through the cache's top-k search in corpus mode — admitting candidates in order until the budget is full. Ties break on `(-score, paper_id, insertion_index)`, so the same inputs always produce the same graph.

Hybrid collects two pools, citation-derived and semantic, and reranks the union against the seed:

```text
score = 0.62 · semantic + 0.16 · temporal + 0.14 · citation + 0.08 · bibliographic
```

where `semantic = 0.5 · (cosine + 1)` maps cosine into `[0, 1]` and `citation = log1p(citations) / log1p(max_citations_in_pool + 1)` compresses the long tail. Then **+0.10** when both branches found the candidate, since citation-plus-semantic agreement is the strongest evidence available, and **+0.02** for citation-derived candidates. `--max-semantic` (default `min(20, max_papers - 1)`) caps semantic-*only* additions; overlap papers do not count against it.

Recommendation and citation instead build a TF-IDF index over the selected papers (unigrams and bigrams, `max_features=5000`, sublinear tf) and use its cosine as the topical component alongside temporal, citation-impact, and bibliographic-coupling signals.

**Knobs:** `--max-papers`, `--max-semantic`, `--similarity-threshold` (default `0.2`).

**Read the code:** `src/citemesh/strategies/hybrid.py` (`_rank_candidates`, `_seed_relevance_score`).

## 6. Edge scoring

Selection decided which papers appear; edge scoring decides which pairs connect, in a different vector space.

The gate is `--min-semantic-similarity`, default `0.74`, on the symmetric cosine — not a probability, not a relevance scale, not transferable across models or dimensions. Calibrated at EmbeddingGemma / STS / 512 dimensions, it retains 17 of 21 related fixture pairs and admits 1 of 105 unrelated ones: precision bought at the cost of real relationships, Transformer/BERT among them. CiteMesh warns when the profile or dimension differs, since the boundary is uncalibrated there; method and label exclusions are in the [threshold study](../reference/defaults-tuning-study.md#semantic-edge-threshold-september-2026).

Past the gate, the embedding strategy combines four signals:

```text
similarity = (0.5 · semantic + 0.2 · temporal + 0.2 · category) × author_factor   # capped at 1.0
author_factor = 1.5 if the papers share an author, else 1.0
```

The weights stop at 0.9 to leave headroom for the multiplier; temporal similarity decays linearly to five years (`1.0 - (Δyears / 5) × 0.8`), then flattens at `0.1`. Note the ordering: dates, categories, and shared authorship only *modify* a score that already cleared the gate — they never create an edge.

Hybrid has provenance the embedding strategy lacks, so it gates on a disjunction — cosine over the threshold **or** non-zero bibliographic coupling — letting papers that share a reference list connect even when the model does not see them as similar. Weights adapt over `(embedding, temporal, citation, bibliographic)` — `(0.6, 0.2, 0.1, 0.1)` semantic-only, `(0.3, 0.3, 0.2, 0.2)` citation-derived, `(0.4, 0.3, 0.2, 0.1)` mixed — then asymmetric floors: below `0.2` rejected, a seed-incident pair needs `> 0.4`, everything else `> 0.5`. The seed gets the lower bar because an isolated seed is useless, while a weak peripheral edge is noise.

Degree is capped last: `3` for recommendation and citation, `--top-k` (default `4`) for embedding, `5` for hybrid — a 40-paper citation graph holds at most 60 edges. `select_capped_undirected_edges` sorts seed-incident edges ahead of everything else regardless of weight, so the seed's strongest neighbors are locked in before other nodes compete; an edge survives only when *both* endpoints are under the cap. Capping only removes edges.

**Knobs:** `--min-semantic-similarity`, `--top-k`, `--similarity-threshold`.

**Read the code:** `src/citemesh/strategies/base.py` (`select_capped_undirected_edges`) and `src/citemesh/core/config.py`.

## 7. Layout and rendering

One layout is computed in Python and shared by every layout-based export, so the PNG, Plotly page, dashboard, and JSON geometry all agree.

`compute_layout` rebuilds the graph in sorted insertion order, detects communities with weighted greedy modularity (Clauset-Newman-Moore, not Louvain), and converts similarity to the path lengths Kamada-Kawai wants: `1 / (1e-6 + max(weight, 0))`, scaled `1.05` within a community and `1.42` across one — that asymmetry is the anti-hairball term. The layout is `networkx.kamada_kawai_layout` at `scale=0.9`: stress majorization, no RNG. `nx.spring_layout` runs **only if** it raises, the one place `--spring-iterations` is read; on the normal path that flag does nothing.

Communities are then spread by anchors from a spring layout over a community meta-graph, every node gets a σ `0.02` Gaussian jitter, disconnected components are shelf-packed, a taller-than-wide result is rotated 90°, and the whole is centered and uniformly scaled. The static PNG viewport *expands* whichever axis is too tight rather than cropping.

Three stages consume randomness, each with a default so an unseeded run still reproduces: the community-anchor layout (`17`), the perturbation (`0`), and the spring fallback (`42`). `--seed` overrides all three and does change the picture, since the first two run on every graph. The Pyvis `html` export ships no coordinates and settles under browser physics, so `--seed` does not reach it.

**Knobs:** `--seed`, `--spring-iterations`, `--dpi`, `--theme`.

**Read the code:** `src/citemesh/visualization/render.py`; node and edge sizing live in `VisualizationConfig`.

## 8. Exports and the dashboard collection

`GraphExporter.graph_payload()` builds one versioned dict (`kind: "citemesh-graph"`, `schema_version: 1`) that every writer reads from, which is why the formats never disagree about a node. The layout is always computed, so even a bare `--export json` embeds dashboard geometry and can be dropped into a dashboard later through **Add Results**. Node enrichment adds external links, provenance, seed relation, and a seed-relevance score from personalized PageRank (`alpha=0.85`, weighted).

The dashboard is a viewer, not a render: it embeds that payload and rebuilds the Plotly figure *client-side*, so filtering, selection, import, and CSV export need no Python round trip. The cost is roughly a dozen algorithms existing in both languages, pinned by tests that run the emitted JavaScript in a Node subprocess against the Python constants.

A dashboard build maintains two shared files at the collection root — the `dashboard.html` viewer and the authoritative `dashboard.citemesh.json` package — plus one directory per seed at `out/<title-slug>-<hash>/`, `<hash>` being the first 8 hex of `sha256(seed_id)`. Builds reuse a directory by that hash, so correcting a title does not move its exports, and results keyed `"<strategy>:<seed_id>"` mean a rerun replaces its own slot while a different strategy adds one. Standalone mode (`-o out/report.dashboard.html`) skips the collection.

File naming, path rules, the package schema, and sidecar contents are in [Output Artifacts](../reference/output-artifacts.md).

**Knobs:** `--export`, `--output`, `--theme`, `--include-timestamp`.

**Read the code:** `src/citemesh/visualization/export/` and `src/citemesh/visualization/dashboard/`.
