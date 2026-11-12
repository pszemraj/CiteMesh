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

The embedding and vector-cache stages run for `--strategy embedding` and `--strategy hybrid`. Recommendation and citation use TF-IDF scoring. All strategies can reuse paper metadata.

## 1. Seed resolution

`normalize_paper_id` normalizes the [accepted identifiers](cli.md#accepted-identifiers) before lookup. S2-backed records use the returned S2 `paperId` throughout collection and graph construction. Input arXiv IDs and DOIs resolve to that record in the metadata cache. Local corpus records retain their source IDs, including when S2 supplies supplemental metadata.

The embedding strategy is the only one with a free-text path. A genuine HTTP 404 means S2 has no such paper, so the builder synthesizes a seed whose ID is `query:` plus the first 8 hex of `sha1(query_text)`. An *unavailable* endpoint (retries exhausted) raises instead - an outage is never silently reinterpreted as a search.

Implementation: [paper_ids.py](../../src/citemesh/core/paper_ids.py).

## 2. Candidate acquisition

S2 candidate acquisition shares availability and identity handling in `strategies/candidates.py`. The embedding candidate mode allocates `min(100, max(1, candidate_pool_size // 4))` recommendations, then `min(100, ceil(remaining / 3))` references; citations get the remainder. Hybrid takes no pool size and applies no quota. It requests up to `--max-references` references and `--max-citations` citers (1000 each by default) and keeps every record the provider returns with usable metadata. In `candidates` and `both` modes it also requests 200 of S2's `all-cs` recommendations, and in `arxiv-corpus` and `both` modes it searches the corpus for the 180 nearest rows to the seed in each of two roles: the seed encoded as a retrieval query and as a retrieval document. Each paper joins the pool once and carries a citation tag (references, citers), a semantic tag (recommendations, corpus neighbors), or both; the tags set its provenance and its edge-scoring weights.

When S2 returns no usable seed references for an arXiv seed, the build can use the [arXiv bibliography fallback](cli.md#missing-semantic-scholar-references). Recovered references enter the pool like provider references, with the same seed relation, up to the reference limit; a bibliography that recovers more references than the limit is cut by title and abstract similarity to the seed, then citation count.

A free-text seed starts with keyword search capped at 20 hits and the total source budget. If slots remain, recommendations expand only the top hit. Corpus mode instead follows [hydration and selection](caching.md#corpus-hydration-and-resume).

During candidate collection, S2 records reconcile by exact paper ID or a unique, conflict-free normalized DOI/arXiv match; ambiguous matches remain separate. Joining a local corpus record with an S2 record uses the same explicit identifier agreement, and the corpus ID remains stable. The bibliography fallback separately resolves identifier-free entries against a prepared corpus using the exact normalized title, full first author, and publication-year rules described in the [CLI guide](cli.md#missing-semantic-scholar-references).

Before hybrid ranking, a preprint and a DOI publication of one work also reconcile by exact normalized title and full first-author name, with publication in the same or following year. The preprint is the corpus row when one of the two is, and otherwise the record with an arXiv ID. Both directions must match uniquely across the whole pool, and conflicting identifiers prevent reconciliation. The preprint's ID survives with merged metadata, merged source tags and seed relation, and the earlier recommendation position. Because this happens before ranking, a reconciled pair never takes a second slot; versions with different titles or first-author names stay separate.

The pool retains reconciled S2 IDs as aliases, so a later list with sparser metadata still joins its existing member. Only the recommendation order is kept as a position, because it is the one source order that feeds the ranking. What the ranker saw for each selected paper is exported as [discovery evidence](../reference/output-artifacts.md#graph-json-strategyjson), not as source positions.

Acquisition follows the [source-failure and retry policy](cli.md#troubleshooting), recording availability in exported metadata.

Implementation: [candidates.py](../../src/citemesh/strategies/candidates.py) and [discovery_pool.py](../../src/citemesh/strategies/discovery_pool.py), the identity-indexed pool that hybrid, the citation and recommendation strategies, and the embedding strategy's candidates mode collect into.

## 3. Embedding

CiteMesh ranks candidates in a retrieval space and scores selected pairs in a symmetric space, using [task-specific prompts](../reference/embedding-runtime.md#task-specific-vector-spaces).

Hybrid ranks every pool member by a retrieval-document vector: the stored corpus vector when the member is a corpus row, otherwise an encoding of its title and abstract. In `candidates` mode those encodings are stored in the embedding cache, so a warm rebuild encodes no candidate again. In the corpus modes the cache is the corpus itself, which takes no provider rows, so each build encodes the provider candidates that are not corpus rows in memory. The seed also gets a retrieval-query vector, which is encoded on every build.

Implementation: [precision.py](../../src/citemesh/strategies/embedding/precision.py) and [model_runtime.py](../../src/citemesh/strategies/embedding/model_runtime.py).

## 4. The cache

The build reuses [cached vectors](caching.md#embedding-namespaces) or [hydrates the corpus](caching.md#corpus-hydration-and-resume) before ranking.

Implementation: [embedding_cache](../../src/citemesh/data/embedding_cache/) and [fingerprint.py](../../src/citemesh/strategies/embedding/fingerprint.py).

## 5. Ranking and selection

Selection cuts the pool to `--max-papers`, seed included.

The embedding strategy encodes the seed as a retrieval query and ranks by raw cosine - over the encoded pool in candidates mode, through the cache's top-k search in corpus mode - admitting candidates in order until the budget is full. Ties break on `(-score, paper_id, insertion_index)`, so the same inputs always produce the same graph.

Hybrid ranks the whole deduplicated pool by reciprocal rank fusion (RRF). Each candidate has a document-role vector, and up to four ranked lists are built from it:

- **similarity to the seed**: every candidate, by the cosine between its vector and the seed's document vector, with ties broken on the cosine with the seed's query vector, then paper ID
- **similarity to the seed's bibliography**: candidates with at least five seed references other than themselves, by the mean of their five highest cosines to those references, with the same tie-breaks
- **direct links**: candidates the seed cites or that cite the seed, by the same cosine as the first list
- **recommendations**: the provider's recommended candidates in provider order, present only in `candidates` and `both` modes

```text
score = Σ 1 / (60 + rank)   over the lists that contain the candidate, ranks starting at 1
```

Candidates sort by score, then paper ID, and the first `max_papers - 1` join the seed. Nothing is fitted: there is no year term, no citation-count term, no source quota and no semantic-only ceiling, and a candidate absent from a list simply adds nothing for it. Ranking uses only embedding cosines, seed links and recommendation order, so it runs before any selected paper's reference list is fetched; those lists come afterwards, for the selected citation-sourced papers only, and feed the bibliographic-coupling term of [edge scoring](#6-edge-scoring). Each kept paper's rank, fused score, features, and rank in every list are exported as [discovery evidence](../reference/output-artifacts.md#graph-json-strategyjson); it is separate from `seed_relevance`, which is computed later over the final graph.

Recommendation and citation instead build a TF-IDF index over the selected papers (unigrams and bigrams, `max_features=5000`, sublinear tf) and use its cosine as the topical component alongside temporal, citation-impact, and bibliographic-coupling signals.

Implementation: [discovery_ranking.py](../../src/citemesh/strategies/discovery_ranking.py) (`rank_candidates`), [discovery_pool.py](../../src/citemesh/strategies/discovery_pool.py) (`DiscoveryPool`), and `HybridGraphBuilder._rank_pool` in [hybrid.py](../../src/citemesh/strategies/hybrid.py).

## 6. Edge scoring

Selection decided which papers appear; edge scoring decides which pairs connect, in a different vector space.

`--min-semantic-similarity` gates the symmetric cosine before composite scoring. Its [calibration study](../reference/defaults-tuning-study.md#semantic-edge-threshold-september-2026) describes the related pairs retained, false positives, and limits across models and dimensions.

Past the gate, the embedding strategy combines four signals:

```text
similarity = (0.5 · semantic + 0.2 · temporal + 0.2 · category) × author_factor   # capped at 1.0
author_factor = 1.5 if the papers share an author, else 1.0
```

The weights stop at 0.9 to leave headroom for the multiplier; temporal similarity decays linearly to five years (`1.0 - (Δyears / 5) × 0.8`), scoring `0.2` at exactly five years and flattening at `0.1` for greater differences. Note the ordering: dates, categories, and shared authorship only *modify* a score that already cleared the gate - they never create an edge.

Hybrid has provenance the embedding strategy lacks, so it gates on a disjunction - cosine at or above the threshold **or** non-zero bibliographic coupling - letting papers that share a reference list connect even when the model does not see them as similar. Weights adapt over `(embedding, temporal, citation, bibliographic)` - `(0.6, 0.2, 0.1, 0.1)` semantic-only, `(0.3, 0.3, 0.2, 0.2)` citation-derived, `(0.4, 0.3, 0.2, 0.1)` mixed - then asymmetric floors: below `0.2` rejected, a seed-incident pair needs `> 0.4`, everything else `> 0.5`. The seed gets the lower bar because an isolated seed is useless, while a weak peripheral edge is noise.

Degree is capped last: `3` for recommendation and citation, `--top-k` for embedding, `5` for hybrid - a 40-paper citation graph holds at most 60 edges. `select_capped_undirected_edges` sorts seed-incident edges ahead of everything else regardless of weight, so the seed's strongest neighbors are locked in before other nodes compete; an edge survives only when *both* endpoints are under the cap. Capping only removes edges.

Implementation: [base.py](../../src/citemesh/strategies/base.py), `select_capped_undirected_edges`, and [config.py](../../src/citemesh/core/config.py); the reference lists that bibliographic coupling compares come from `ReferenceHydrator` in [reference_hydration.py](../../src/citemesh/strategies/reference_hydration.py), shared by the citation, recommendation, and hybrid strategies.

## 7. Layout and rendering

One layout is computed in Python and shared by every layout-based export, so the PNG, Plotly page, dashboard, and JSON geometry all agree.

`compute_layout` rebuilds the graph in sorted insertion order, detects communities with weighted greedy modularity (Clauset-Newman-Moore, not Louvain), and converts similarity to the path lengths Kamada-Kawai wants: `1 / (1e-6 + max(weight, 0))`, scaled `1.05` within a community and `1.42` across one - that asymmetry is the anti-hairball term. The layout is `networkx.kamada_kawai_layout` at `scale=0.9`, minimizing path-length error without a random seed. `nx.spring_layout` runs **only if** it raises, the one place `--spring-iterations` is read; on the normal path that flag does nothing.

Communities are then spread by anchors from a spring layout over a community meta-graph, every node gets a σ `0.02` Gaussian jitter, disconnected components are shelf-packed, a taller-than-wide result is rotated 90°, and the whole is centered and uniformly scaled. The static PNG viewport *expands* whichever axis is too tight rather than cropping.

Three stages consume randomness, each with a default so an unseeded run still reproduces: the community-anchor layout (`17`), the perturbation (`0`), and the spring fallback (`42`). `--seed` overrides all three and does change the picture, when the relevant stages run. A single-community graph needs no community-anchor layout. The Pyvis `html` export ships no coordinates and settles under browser physics, so `--seed` does not reach it.

Implementation: [render.py](../../src/citemesh/visualization/render.py); node and edge sizing use `VisualizationConfig`.

## 8. Exports and the dashboard collection

Exports enrich nodes with external links, provenance, seed relation, and personalized PageRank (`alpha=0.85`, weighted). PageRank measures relevance within the final graph; it is separate from the candidate-ranking score.

The dashboard embeds graph data and rebuilds its figure in the browser. The [Python/JavaScript boundary](../internals/architecture.md#python--javascript-duplication) explains how both implementations are tested; [Output Artifacts](../reference/output-artifacts.md#dashboard-html) describes dashboard behavior.

Implementation: [export](../../src/citemesh/visualization/export/) and [dashboard](../../src/citemesh/visualization/dashboard/).
