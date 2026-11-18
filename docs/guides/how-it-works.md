# How CiteMesh builds a graph

A build resolves its focal paper, collects candidates, scores papers and edges, computes a layout, and exports the graph. The [strategy](strategies.md) determines candidate sources and scoring; the [CLI guide](cli.md#option-reference) points to the current controls.

```text
  focal paper ID, or free text with --strategy embedding
        │
        ▼
  1. Focal paper resolution → 2. Candidate acquisition → 3. Embedding → 4. Cache
                                                                           │
  8. Export ← 7. Layout ← 6. Edge scoring ← 5. Ranking & selection ◄───────┘
```

The embedding and vector-cache stages run for `--strategy embedding` and `--strategy hybrid`. Recommendation and citation use TF-IDF scoring. All strategies can reuse paper metadata.

## 1. Focal paper resolution

`normalize_paper_id` normalizes the [accepted identifiers](cli.md#accepted-identifiers) before lookup. S2-backed records use the returned S2 `paperId` throughout collection and graph construction. Input arXiv IDs and DOIs resolve to that record in the metadata cache. Local corpus records retain their source IDs, including when S2 supplies supplemental metadata.

The embedding strategy is the only one with a free-text path. A genuine HTTP 404 means S2 has no such paper, so the builder synthesizes a focal record whose ID is `query:` plus the first 8 hex of `sha1(query_text)`. An *unavailable* endpoint (retries exhausted) raises instead - an outage is never silently reinterpreted as a search.

Implementation: [paper_ids.py](../../src/citemesh/core/paper_ids.py).

## 2. Candidate acquisition

S2 candidate acquisition shares availability and identity handling in `strategies/candidates.py`. What each strategy collects:

- **Embedding, `candidates` mode**: allocates `min(100, max(1, candidate_pool_size // 4))` recommendations, then `min(100, ceil(remaining / 3))` references; citations get the remainder.
- **Hybrid** takes no pool size. It requests up to `--max-references` references and `--max-citations` citers (1000 each by default) and keeps every record the provider returns with usable metadata. In `candidates` and `both` modes it also requests up to 200 Semantic Scholar recommendations. In `arxiv-corpus` and `both` modes it searches the corpus for the 180 nearest rows to the focal paper in each of two roles: the focal paper encoded as a retrieval query and as a retrieval document.
- **Provenance**: each paper joins the pool once and carries a citation tag (references, citers), a semantic tag (recommendations, corpus neighbors), or both; the tags set its provenance and its edge-scoring weights.

Hybrid chooses the recommendation pool using the subject labels already recorded on the focal paper. Any computing label (`Computer Science`, `cs`, `cs.*`, `stat.ML`, or legacy `cmp-lg`) selects `all-cs`, including on a paper with multiple subjects. Missing subjects or a sole `Unknown` label also select `all-cs`. A paper with another recorded subject and no computing label uses `recent` first.

When S2 returns no usable references for an arXiv focal paper, the build can use the [arXiv bibliography fallback](cli.md#missing-semantic-scholar-references). Recovered references enter the pool like provider references, with the same relation to the focal paper, up to the reference limit; a bibliography that recovers more references than the limit is cut by title and abstract similarity to the focal paper, then citation count.

A free-text input starts with keyword search capped at 20 hits and the total source budget. If slots remain, recommendations expand only the top hit. Corpus mode instead follows [hydration and selection](caching.md#corpus-hydration-and-resume).

During candidate collection, S2 records reconcile by exact paper ID or a unique, conflict-free normalized DOI/arXiv match; ambiguous matches remain separate. Joining a local corpus record with an S2 record uses the same explicit identifier agreement, and the corpus ID remains stable. The bibliography fallback separately resolves identifier-free entries against a prepared corpus using the exact normalized title, full first author, and publication-year rules described in the [CLI guide](cli.md#missing-semantic-scholar-references).

Before hybrid ranking, a preprint and a DOI publication of one work also reconcile by exact normalized title and full first-author name, with publication in the same or following year. When the publication abbreviates the first author's given names to initials, those must agree in order and the remaining surname tokens must match, including particles and suffixes. Both records must also have the same normalized abstract of at least 20 words. The preprint is the corpus row when one of the two is, and otherwise the record with an arXiv ID. Both directions must match uniquely across the whole pool, and conflicting identifiers prevent reconciliation. The preprint's ID survives with merged metadata, source tags, relation to the focal paper, and the earlier recommendation position.

The pool retains reconciled S2 IDs as aliases, so a later list with sparser metadata still joins its existing member. Only the recommendation order is kept as a position, because it is the one source order that feeds the ranking. What the ranker saw for each selected paper is exported as [discovery evidence](../reference/output-artifacts.md#graph-json-strategyjson), not as source positions.

Acquisition follows the [source-failure and retry policy](cli.md#troubleshooting), recording availability in exported metadata.

Implementation: [candidates.py](../../src/citemesh/strategies/candidates.py) and [discovery_pool.py](../../src/citemesh/strategies/discovery_pool.py), the identity-indexed pool that hybrid, the citation and recommendation strategies, and the embedding strategy's candidates mode collect into.

## 3. Embedding

CiteMesh ranks candidates in a retrieval space and scores selected pairs in a symmetric space, using [task-specific prompts](../reference/embedding-runtime.md#task-specific-vector-spaces). Every title, abstract, and free-text query is normalized before any role formats it: invisible characters are removed, typographic punctuation is folded, and line breaks and whitespace runs collapse, so arXiv's hard-wrapped metadata and a provider's clean copy of the same paper encode identically.

Hybrid ranks every pool member by a retrieval-document vector: the stored corpus vector when the member is a corpus row, otherwise an encoding of its title and abstract. In `candidates` mode those encodings are stored in the embedding cache, so a warm rebuild encodes no candidate again. In the corpus modes the cache is the corpus itself, which takes no provider rows, so each build encodes the provider candidates that are not corpus rows in memory. The focal paper also gets a retrieval-query vector, which is encoded on every build.

Implementation: [precision.py](../../src/citemesh/strategies/embedding/precision.py) and [model_runtime.py](../../src/citemesh/strategies/embedding/model_runtime.py).

## 4. The cache

The build reuses [cached vectors](caching.md#embedding-namespaces) or [hydrates the corpus](caching.md#corpus-hydration-and-resume) before ranking.

Implementation: [embedding_cache](../../src/citemesh/data/embedding_cache/) and [fingerprint.py](../../src/citemesh/strategies/embedding/fingerprint.py).

## 5. Ranking and selection

Selection cuts the pool to `--max-papers`, focal paper included.

The embedding strategy encodes the focal paper as a retrieval query and ranks by raw cosine - over the encoded pool in candidates mode, through the cache's top-k search in corpus mode - admitting candidates in order until the budget is full. Ties break on `(-score, paper_id, insertion_index)`, so the same inputs always produce the same graph.

Hybrid ranks the whole deduplicated pool by reciprocal rank fusion (RRF). Each candidate has a document-role vector. The available evidence determines which of five ranked lists contribute.

- **similarity to the focal paper**: every candidate, by the cosine between its vector and the focal paper's document vector, with ties broken on the cosine with the focal paper's query vector, then paper ID
- **similarity to the focal paper's bibliography**: candidates with at least five references from the focal paper other than themselves, by the mean of their five highest cosines to those references, with ties broken by query cosine, then paper ID
- **direct links**: candidates the focal paper cites or that cite the focal paper, by document cosine to the focal paper, then paper ID
- **recommendations**: the provider's recommended candidates in provider order after deduplication, present only in `candidates` and `both` modes
- **focal reference evidence**: references cited in two or more distinct passages of the focal paper or marked influential by Semantic Scholar. Higher passage counts come first, followed by the influence flag and the focal similarity order. This list has weight `0.5`.

```text
score = Σ weight / (60 + rank)   over the lists that contain the candidate, ranks starting at 1
```

The other lists have weight `1`. Candidates sort by score, then paper ID, and the first `max_papers - 1` join the focal paper. A list contributes only for candidates it contains. With no qualifying focal reference evidence, hybrid fuses the other available lists. Ranking runs before optional reference hydration for selected citation-sourced papers, which supplies bibliographic coupling during [edge scoring](#6-edge-scoring). Each kept paper's rank, fused score, features, and list ranks are exported as [discovery evidence](../reference/output-artifacts.md#graph-json-strategyjson). The `seed_relevance` value is computed later over the final graph.

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

Hybrid has provenance the embedding strategy lacks, so it scores differently:

- **Gate**: a disjunction - cosine at or above the threshold **or** non-zero bibliographic coupling - letting papers that share a reference list connect even when the model does not see them as similar.
- **Weights** adapt over `(embedding, temporal, citation, bibliographic)`: `(0.6, 0.2, 0.1, 0.1)` semantic-only, `(0.3, 0.3, 0.2, 0.2)` citation-derived, `(0.4, 0.3, 0.2, 0.1)` mixed.
- **Floors** are asymmetric: below `0.2` rejected, a pair incident to the focal paper needs `> 0.4`, everything else `> 0.5`. The focal paper gets the lower bar because an isolated focal paper is useless, while a weak peripheral edge is noise.

Degree is capped last: `3` for recommendation and citation, `--top-k` for embedding, `5` for hybrid - a 40-paper citation graph holds at most 60 edges. `select_capped_undirected_edges` sorts edges incident to the focal paper ahead of everything else regardless of weight, so the focal paper's edges are locked in before other nodes compete; an edge survives only when *both* endpoints are under the cap. Capping only removes edges.

In hybrid the focal paper's slots alternate between its direct references and citers and its other neighbors, strongest first within each group and starting with a direct link, so the focal paper keeps both its cited neighborhood and its closest discoveries. Both groups compete only among edges that cleared the gate and the floor for the focal paper, and a group with nothing left to offer yields its turns to the other.

Implementation: [base.py](../../src/citemesh/strategies/base.py), `select_capped_undirected_edges`, and [config.py](../../src/citemesh/core/config.py); the reference lists that bibliographic coupling compares come from `ReferenceHydrator` in [reference_hydration.py](../../src/citemesh/strategies/reference_hydration.py), shared by the citation, recommendation, and hybrid strategies.

## 7. Layout and rendering

One layout is computed in Python and shared by every layout-based export, so the PNG, Plotly page, dashboard, and JSON geometry all agree.

`compute_layout` rebuilds the graph in sorted insertion order, detects communities with weighted greedy modularity (Clauset-Newman-Moore, not Louvain), and converts similarity to the path lengths Kamada-Kawai wants: `1 / (1e-6 + max(weight, 0))`, scaled `1.05` within a community and `1.42` across one - that asymmetry is the anti-hairball term. The layout is `networkx.kamada_kawai_layout` at `scale=0.9`, minimizing path-length error without a random seed. `nx.spring_layout` runs **only if** it raises, the one place `--spring-iterations` is read; on the normal path that flag does nothing.

Communities are then spread by anchors from a spring layout over a community meta-graph, every node gets a σ `0.02` Gaussian jitter, disconnected components are shelf-packed, a taller-than-wide result is rotated 90°, and the whole is centered and uniformly scaled. The static PNG viewport *expands* whichever axis is too tight rather than cropping.

Three stages consume randomness, each with a default so an unseeded run still reproduces: the community-anchor layout (`17`), the perturbation (`0`), and the spring fallback (`42`). `--seed` overrides all three and does change the picture, when the relevant stages run. A single-community graph needs no community-anchor layout. The Pyvis `html` export ships no coordinates and settles under browser physics, so `--seed` does not reach it.

Implementation: [render.py](../../src/citemesh/visualization/render.py); node and edge sizing use `VisualizationConfig`.

## 8. Exports and the dashboard collection

Exports enrich nodes with external links, provenance, relation to the focal paper, and personalized PageRank (`alpha=0.85`, weighted). PageRank measures relevance within the final graph; it is separate from the candidate-ranking score.

The dashboard embeds graph data and rebuilds its figure in the browser. The [Python/JavaScript boundary](../internals/architecture.md#python--javascript-duplication) explains how both implementations are tested; [Output Artifacts](../reference/output-artifacts.md#dashboard-html) describes dashboard behavior.

Implementation: [export](../../src/citemesh/visualization/export/) and [dashboard](../../src/citemesh/visualization/dashboard/).
