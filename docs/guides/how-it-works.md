# How CiteMesh builds a graph

A build resolves its seed, collects candidates, scores papers and edges, computes a layout, and exports the graph. The [strategy](strategies.md) determines candidate sources and scoring; the [CLI guide](cli.md#option-reference) points to the current controls.

```text
  seed ID, or free text with --strategy embedding
        │
        ▼
  1. Seed resolution → 2. Candidate acquisition → 3. Embedding → 4. Cache
                                                                     │
  8. Export ← 7. Layout ← 6. Edge scoring ← 5. Ranking & selection ◄──┘
```

The embedding and vector-cache stages run for `--strategy embedding` and for hybrid with semantic enrichment enabled. Recommendation, citation, and hybrid with enrichment disabled use TF-IDF scoring. All strategies can reuse paper metadata.

## 1. Seed resolution

`normalize_paper_id` normalizes the [accepted identifiers](cli.md#accepted-identifiers) before lookup. S2-backed records use the returned S2 `paperId` throughout collection and graph construction. Input arXiv IDs and DOIs resolve to that record in the metadata cache. Local corpus records retain their source IDs, including when S2 supplies supplemental metadata.

The embedding strategy is the only one with a free-text path. A genuine HTTP 404 means S2 has no such paper, so the builder synthesizes a seed whose ID is `query:` plus the first 8 hex of `sha1(query_text)`. An *unavailable* endpoint (retries exhausted) raises instead - an outage is never silently reinterpreted as a search.

Implementation: [paper_ids.py](../../src/citemesh/core/paper_ids.py).

## 2. Candidate acquisition

S2 candidate acquisition shares availability and identity handling in `strategies/candidates.py`. An embedding candidate pool allocates roughly a quarter to recommendations and a quarter to references, each capped at 100; citations receive the remainder. Hybrid's citation branch uses its own fetch budgets, while its semantic branch adds recommendations up to the smallest of 100, `max(max_semantic, min(max_papers - 1, 3 * max_semantic))`, and `--candidate-pool-size`.

When S2 returns no usable seed references and the seed has an arXiv ID, the shared acquisition path can recover explicit arXiv IDs and DOIs from an available arXiv HTML bibliography. [CLI Usage](cli.md#missing-semantic-scholar-references) defines the fallback's resolution order and limits.

A free-text seed starts with keyword search capped at 20 hits and the total source budget. If slots remain, recommendations expand only the top hit. Corpus mode instead follows [hydration and selection](caching.md#corpus-hydration-and-resume).

S2 candidate collection reconciles exact paper IDs or a unique, conflict-free normalized DOI/arXiv match; ambiguous matches remain separate. Joining a local corpus record with an S2 record requires the same explicit identifier agreement, and the corpus ID remains stable. Similar titles, years, authors, or abstracts do not establish identity; records without a shared explicit identifier remain separate.

Acquisition follows the [source-failure and retry policy](cli.md#appendix-b-troubleshooting), recording availability in exported metadata.

Implementation: [candidates.py](../../src/citemesh/strategies/candidates.py).

## 3. Embedding

CiteMesh ranks candidates in a retrieval space and scores selected pairs in a symmetric space, using [task-specific prompts](../reference/embedding-runtime.md#task-specific-vector-spaces).

Implementation: [precision.py](../../src/citemesh/strategies/embedding/precision.py) and [model_runtime.py](../../src/citemesh/strategies/embedding/model_runtime.py).

## 4. The cache

The build reuses [cached vectors](caching.md#embedding-namespaces) or [hydrates the corpus](caching.md#corpus-hydration-and-resume) before ranking.

Implementation: [embedding_cache](../../src/citemesh/data/embedding_cache/) and [fingerprint.py](../../src/citemesh/strategies/embedding/fingerprint.py).

## 5. Ranking and selection

Selection cuts the pool to `--max-papers`, seed included.

The embedding strategy encodes the seed as a retrieval query and ranks by raw cosine - over the encoded pool in candidates mode, through the cache's top-k search in corpus mode - admitting candidates in order until the budget is full. Ties break on `(-score, paper_id, insertion_index)`, so the same inputs always produce the same graph.

Hybrid collects two pools, citation-derived and semantic, and reranks the union against the seed:

```text
score = 0.62 · semantic + 0.16 · temporal + 0.14 · citation + 0.08 · bibliographic
```

where `semantic = 0.5 · (cosine + 1)` maps cosine into `[0, 1]` and `citation = log1p(citations) / log1p(max_citations_in_pool + 1)` compresses the long tail. Then **+0.10** when both branches found the candidate, since citation-plus-semantic agreement is the strongest evidence available, and **+0.02** for citation-derived candidates. `--max-semantic` caps semantic-*only* additions; overlap papers do not count against it. It is a ceiling, not a reservation: if citation-derived candidates fill `--max-papers` first, no semantic-only paper is added.

Recommendation and citation instead build a TF-IDF index over the selected papers (unigrams and bigrams, `max_features=5000`, sublinear tf) and use its cosine as the topical component alongside temporal, citation-impact, and bibliographic-coupling signals.

Implementation: [hybrid.py](../../src/citemesh/strategies/hybrid.py), `_rank_candidates` and `_seed_relevance_score`.

## 6. Edge scoring

Selection decided which papers appear; edge scoring decides which pairs connect, in a different vector space.

`--min-semantic-similarity` gates the symmetric cosine before composite scoring. Its [calibration study](../reference/defaults-tuning-study.md#semantic-edge-threshold-september-2026) describes the related pairs retained, false positives, and limits across models and dimensions.

Past the gate, the embedding strategy combines four signals:

```text
similarity = (0.5 · semantic + 0.2 · temporal + 0.2 · category) × author_factor   # capped at 1.0
author_factor = 1.5 if the papers share an author, else 1.0
```

The weights stop at 0.9 to leave headroom for the multiplier; temporal similarity decays linearly to five years (`1.0 - (Δyears / 5) × 0.8`), then flattens at `0.1`. Note the ordering: dates, categories, and shared authorship only *modify* a score that already cleared the gate - they never create an edge.

Hybrid has provenance the embedding strategy lacks, so it gates on a disjunction - cosine at or above the threshold **or** non-zero bibliographic coupling - letting papers that share a reference list connect even when the model does not see them as similar. Weights adapt over `(embedding, temporal, citation, bibliographic)` - `(0.6, 0.2, 0.1, 0.1)` semantic-only, `(0.3, 0.3, 0.2, 0.2)` citation-derived, `(0.4, 0.3, 0.2, 0.1)` mixed - then asymmetric floors: below `0.2` rejected, a seed-incident pair needs `> 0.4`, everything else `> 0.5`. The seed gets the lower bar because an isolated seed is useless, while a weak peripheral edge is noise.

Degree is capped last: `3` for recommendation and citation, `--top-k` for embedding, `5` for hybrid - a 40-paper citation graph holds at most 60 edges. `select_capped_undirected_edges` sorts seed-incident edges ahead of everything else regardless of weight, so the seed's strongest neighbors are locked in before other nodes compete; an edge survives only when *both* endpoints are under the cap. Capping only removes edges.

Implementation: [base.py](../../src/citemesh/strategies/base.py), `select_capped_undirected_edges`, and [config.py](../../src/citemesh/core/config.py).

## 7. Layout and rendering

One layout is computed in Python and shared by every layout-based export, so the PNG, Plotly page, dashboard, and JSON geometry all agree.

`compute_layout` rebuilds the graph in sorted insertion order, detects communities with weighted greedy modularity (Clauset-Newman-Moore, not Louvain), and converts similarity to the path lengths Kamada-Kawai wants: `1 / (1e-6 + max(weight, 0))`, scaled `1.05` within a community and `1.42` across one - that asymmetry is the anti-hairball term. The layout is `networkx.kamada_kawai_layout` at `scale=0.9`, minimizing path-length error without a random seed. `nx.spring_layout` runs **only if** it raises, the one place `--spring-iterations` is read; on the normal path that flag does nothing.

Communities are then spread by anchors from a spring layout over a community meta-graph, every node gets a σ `0.02` Gaussian jitter, disconnected components are shelf-packed, a taller-than-wide result is rotated 90°, and the whole is centered and uniformly scaled. The static PNG viewport *expands* whichever axis is too tight rather than cropping.

Three stages consume randomness, each with a default so an unseeded run still reproduces: the community-anchor layout (`17`), the perturbation (`0`), and the spring fallback (`42`). `--seed` overrides all three and does change the picture, when the relevant stages run. A single-community graph needs no community-anchor layout. The Pyvis `html` export ships no coordinates and settles under browser physics, so `--seed` does not reach it.

Implementation: [render.py](../../src/citemesh/visualization/render.py); node and edge sizing use `VisualizationConfig`.

## 8. Exports and the dashboard collection

Exports enrich nodes with external links, provenance, seed relation, and personalized PageRank (`alpha=0.85`, weighted). PageRank measures relevance within the final graph; it is separate from the candidate-ranking score.

The dashboard embeds graph data and rebuilds its figure in the browser, so selection, filtering, and imports work offline. The [Python/JavaScript boundary](../internals/architecture.md#python--javascript-duplication) explains how both implementations are tested. [Output Artifacts](../reference/output-artifacts.md) describes formats, schemas, naming, and collection updates.

Implementation: [export](../../src/citemesh/visualization/export/) and [dashboard](../../src/citemesh/visualization/dashboard/).
