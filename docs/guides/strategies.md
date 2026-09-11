# Strategy Guide

Pick a strategy here. `recommendation`, `citation`, `embedding`, and `hybrid` differ in where candidates come from and how a pair of papers is scored into an edge; everything else about a build is shared. The mechanism behind each stage is in [How CiteMesh builds a graph](how-it-works.md), and the flag contracts are in the [CLI guide](cli.md#flag-reference).

| Strategy | Primary signal | Loads the model | Best for |
| --- | --- | --- | --- |
| `recommendation` | Semantic Scholar recommendations | no | fast topical exploration from a known paper |
| `citation` | references, citations, bibliographic coupling | no | citation-derived neighborhoods |
| `embedding` | dense vector similarity | yes | conceptual similarity beyond citations |
| `hybrid` | citation candidates plus semantic reranking | yes | balanced grounded and semantic recall |

All four cap node degree, reserve the seed's strongest edges before any other node competes for its budget, and warn when no selected pair clears the edge criteria and the graph comes out with no edges at all.

## recommendation

Candidates come from Semantic Scholar's recommendation endpoints, which makes this the shortest path from a known paper ID to a graph and the one with the least setup. Edges score TF-IDF topical similarity against temporal and bibliographic-coupling evidence and must clear `--similarity-threshold` (default `0.2`); a pair with no topical overlap and no shared references scores zero, so publication era and citation popularity alone never connect two papers. At most three edges touch any paper.

Coverage is entirely Semantic Scholar's. A paper the service knows little about gives a thin graph, and there is no fallback signal to make up for it.

## citation

Candidates come from the seed's references and citations, so every node has an explicit bibliographic link into the neighborhood even where topics diverge. Edges use the same `0.2` threshold and the same three-edge cap as `recommendation` — a default 40-paper graph therefore holds at most 60 edges — but weight bibliographic coupling much more heavily when reference lists are available on both sides. `--no-references` skips fetching them, which is faster and gives up real coupling.

Both this and `recommendation` warn when reference hydration exhausts its retries, then stop hydrating for that collection; reference lists already fetched still score, and a later collection retries the source.

## embedding

Title and abstract text is encoded locally and ranked against the seed by cosine. This is the only strategy that reaches papers with no citation path to the seed, and the only one that accepts free text instead of an ID. `--semantic-source` decides where the candidates come from:

- `candidates` (default) fetches S2 references, citations, and recommendations within `--candidate-pool-size` and embeds that pool. Nothing else is downloaded, and candidate metadata already carries citation counts.
- `arxiv-corpus` hydrates a local arXiv abstract corpus from HuggingFace (`librarian-bots/arxiv-metadata-snapshot` by default) and searches that. It surfaces work no citation path would reach, at the cost of the `datasets` extra and substantially more cold-cache time. `--dataset-source` and `--dataset-split` select another repository or split; `--corpus-size N` caps hydration to the N newest submissions after scanning the split. Hydration, resumption, and storage behavior are in [Caching & Data](caching.md).

Edges need a symmetric cosine of at least **0.74** before publication year, category overlap, and shared authors modify the weight, and `--top-k` (default `4`) caps how many neighbors each paper keeps. Model defaults, precision, and compile policy are in [Embedding Runtime](../reference/embedding-runtime.md).

## hybrid

Hybrid builds both pools — citation-derived and semantic — and reranks the union by seed relevance, so one graph carries citation grounding and semantic recall. It inherits the embedding path's dependencies and caches. Three behaviors are worth knowing before you pick it:

- Its edge gate is a disjunction: symmetric cosine over `0.74` **or** shared references. A bibliographic link can therefore carry an edge the model would not, subject to composite floors of `0.4` for seed-incident pairs and `0.5` for everything else, so a shared-reference pair with weak topical and temporal evidence is still rejected.
- `--max-semantic` limits semantic-*only* additions, not overlap papers that both branches found. `--max-semantic 0` falls back to the citation scorer and threshold while keeping hybrid's degree cap.
- Semantic enrichment failures stop the build rather than silently downgrading to citation-only output. If the citation endpoints go down after the seed resolves, enrichment still proceeds from recommendations or the local corpus, and source availability is recorded in the graph metadata.

Hybrid's defaults are tuned for seed-paper discovery — recent follow-up work plus foundational prior work — with the evaluation written up in the [defaults study](../reference/defaults-tuning-study.md#hybrid-defaults-february-2026).

## Tuning the semantic boundary

`0.74` is not a probability and not a model-independent relevance scale: it was calibrated for EmbeddingGemma with STS formatting at 512 dimensions. Override it for one run with `--min-semantic-similarity VALUE`, or persist it with `citemesh config set defaults.min_semantic_similarity VALUE`. CiteMesh warns when the active profile is not EmbeddingGemma or the dimension is not 512, because the boundary is uncalibrated there and a smaller dimension does not preserve it.

Evaluate labeled related and unrelated pairs before choosing an override. The selection method, fixture counts, label exclusions, and known misses are in the [semantic threshold study](../reference/defaults-tuning-study.md#semantic-edge-threshold-september-2026); with the default model cached, `python -m pytest -m slow tests/test_semantic_quality.py` reruns the same fixtures offline.
