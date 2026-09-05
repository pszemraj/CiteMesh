# Defaults Tuning Studies

Measurements and tradeoffs behind CiteMesh's embedding dimension and hybrid
discovery defaults.

Related docs:

- CLI defaults and flags: [CLI Usage](../guides/cli.md)
- Strategy behavior overview: [Strategy Guide](../guides/strategies.md)
- Embedding runtime policy: [Embedding Runtime](embedding-runtime.md)

## Embedding Dimensions (September 2026)

### Decision

Use **512 dimensions** by default for EmbeddingGemma. The previous 256-dimensional
default reduced vector storage, but the paired corpus study below found a
substantial loss of the full model's nearest neighbors. Moving to 512 recovered
more of those neighbors with essentially the same GPU corpus encoding time and
an 11.4% increase in complete cache size. Local search became slower.

The shared profile applies this choice to embedding and hybrid builds, candidate
and arXiv corpus sourcing, local search, and symmetric graph-similarity encoding
on CUDA, MPS, and CPU. It covers the default Unsloth model, the Google fallback,
and recognized local EmbeddingGemma checkpoints. This is one consistent default;
the experiment directly measured CUDA corpus retrieval, not each of those paths.

EmbeddingGemma supports `768`, `512`, `256`, and `128` dimensions. An explicit
`--truncate-dim` or `defaults.truncate_dim` configuration still takes precedence;
other model profiles retain their own dimension policies. Existing 256d caches
remain separate and reusable with matching settings; see
[Caching & Data](../guides/caching.md).

### Paired corpus and runtime

The study completed on September 5, 2026 (UTC), using separate temporary CiteMesh
caches containing the **same 100,000 papers in the same order**. The main user
cache was not modified.

- Dataset: `librarian-bots/arxiv-metadata-snapshot/train`, revision
  `47141d6fd17f52b65424d246665334914cac3011`, last updated August 31, 2026.
- Selection: scan all 3,148,882 records and select the newest 100,000 submissions
  by arXiv ID, from `2605.20790` through `2608.27458`, preserving source order.
  This covers May–August 2026 in that snapshot, not a live September feed or
  recently revised older submissions.
- Model: `unsloth/embeddinggemma-300m`, artifact
  `bfa3c846ac738e62aa61806ef9112d34acb1dc5a`.
- Runtime: RTX 5090, torch `2.13.0+cu130`, sentence-transformers `6.0.0`,
  transformers `5.2.0`; BF16 autocast, Flash Attention 2, `torch.compile`, batch
  size 128, and eight CPU threads. Checkpoint weights used automatic dtype
  selection, embedding outputs were FP32, and no FP16 was used.
- Storage: INT8 with gzip level 1 and a binary sign index, using the same
  deterministic 2,000-record reservoir sample for per-dimension min/max
  calibration in each run.

Each dimension had its own GPU corpus inference run through CiteMesh's normal
calibration and hydration path, with the prepared local rows supplied as input.
During the 512d run, the recorder retained the model's full 768d output before
truncating and normalizing to 512d for the cache. This provided a reference without
a third corpus inference run. The independently encoded 256d vectors agreed with
normalized 256d prefixes of that reference to mean cosine 1.000000 and minimum
0.999992.

### Seed selection and comparison method

Thirty queries used seed titles and abstracts with CiteMesh's retrieval-query
formatting. Twelve established papers covered language modeling, retrieval,
vision, graph learning, and gravitational waves. Eighteen recent seeds were
selected with NumPy RNG seed `20260905`, two per primary category from the
selected corpus. Self matches were removed before comparing the top 20 results.

| Seed group | arXiv IDs |
| --- | --- |
| Established | `2404.08801` (Megalodon), `1706.03762` (Transformer), `2312.00752` (Mamba), `2305.18290` (DPO), `2005.11401` (RAG), `2103.00020` (CLIP), `2006.11239` (DDPM), `2003.08934` (NeRF), `2304.02643` (SAM), `1609.02907` (GCN), `1602.03837` (binary black hole merger), `2303.08774` (GPT-4 report) |
| cs.CL | `2606.08932`, `2606.05494` |
| cs.CV | `2605.26485`, `2608.17110` |
| cs.LG | `2607.20694`, `2605.25304` |
| cs.IR | `2606.29946`, `2608.00816` |
| cs.RO | `2608.25459`, `2607.23384` |
| quant-ph | `2606.24540`, `2606.21675` |
| astro-ph.HE | `2606.25806`, `2606.21691` |
| cond-mat.mtrl-sci | `2606.29456`, `2608.11687` |
| math.PR | `2606.27839`, `2607.24561` |

The reference was exact cosine search over full **768-dimensional FP32 outputs**
from the same model and corpus. Top-20 retention is the fraction of reference
neighbors found in a compared list, averaged equally across the 30 seeds. It is
agreement with the untruncated model, **not relevance accuracy**: that model can
also retrieve poor matches, and a different neighbor may be equally relevant.

Two search budgets were measured:

- **Graph candidates:** the default 40-paper embedding build requests 160
  candidates, shortlists 1,280 via the binary index (8x), and rescores them. The
  comparison takes its first 20 non-self neighbors, before hybrid fusion or graph
  scoring.
- **Local cache search:** request 21 results, shortlist 168 (8x), then remove any
  self match and retain 20. Search latency excludes query encoding.

### Results

| Mean top-20 retention against the full 768d reference | 256d | 512d |
| --- | ---: | ---: |
| Exact FP32 search | 53.7% | 75.0% |
| Exact INT8 search | 53.5% | 75.2% |
| Graph candidate search | 53.3% | 74.7% |
| Local cache search | 46.5% | 69.7% |

512d retained more reference neighbors on **all 30 seeds** in the graph candidate
comparison. Its list overlapped the 256d list by 59.3% on average. For Megalodon,
reference retention increased from 35% to 55%; for RAG, from 45% to 75%; for Mamba,
from 75% to 85%.

| Build and storage measurement | 256d | 512d |
| --- | ---: | ---: |
| Corpus encoding time | 89.6 s | 88.2 s |
| Calibration encoding and compilation warmup | 39.9 s | 32.0 s |
| Total hydration, including calibration and writes | 137.0 s | 127.3 s |
| Vectors, binary index, and calibration | 26.9 MB | 53.8 MB |
| SQLite metadata | 208.2 MB | 208.2 MB |
| Complete cache | 235.2 MB | 262.1 MB |
| Median local cache search | 99.7 ms | 165.8 ms |

MB uses decimal units. Vector storage approximately doubled, while complete
cache size rose only 11.4% because metadata dominated this corpus. Local search
was about 66% slower. These were single paired runs with 781 full 128-record
batches each, not repeated timing trials. The second run reused compiler
artifacts; the warmup difference does not establish that 512d is faster. Corpus
encoding timers include the encoder call and output conversion, but exclude
the separate FP32 reference-file write.

Dimension truncation and binary shortlisting affected neighbors much more than
fresh INT8 quantization:

| Mean top-20 retention against the same dimension | 256d | 512d |
| --- | ---: | ---: |
| Exact INT8 versus exact FP32 | 98.5% | 99.2% |
| Graph candidate search versus exact INT8 | 92.2% | 97.5% |
| Local cache search versus exact INT8 | 64.5% | 82.8% |

Only 0.0964% / 0.0957% of coordinates were clipped in the respective fresh caches.
The smaller local-search shortlist caused additional loss independently of INT8
rounding; this study did not change the shortlist settings.

### Qualitative review and limits

Two GPT-5.6 Sol reviewers each assessed six different established seeds using
blinded A/B top-five lists, titles, and abstracts, ignoring cosine scores as
evidence of relevance. Preferences were **512d: 7, 256d: 4, tie: 1**. These are
model-assisted judgments, not human labels or duplicate independent reviews.
Several preferences depended on one differing result.

- 512d was preferred for Megalodon, Transformer, RAG, CLIP, DDPM, SAM, and the
  binary black hole merger paper. For RAG, it offered more retrieval-method
  coverage and fewer repetitive historical-document/OCR applications.
- 256d was preferred for DPO, NeRF, GCN, and the GPT-4 report; differing results
  were closer to the seed's method or breadth. Mamba was tied.
- Both Megalodon lists still contained tangential papers. Higher dimensions did
  not make every neighbor useful.

Both saved caches reopened with 100,000 rows and reused hydration without loading
the dataset. CiteMesh's actual graph candidate search reproduced all 60 ranked
lists exactly. The evaluation did not measure the full 3.15-million-paper corpus,
Semantic Scholar fusion, graph topology, or real CPU/MPS execution. The 11.4%
complete-cache increase applies to these INT8 corpus caches; it is not an estimate
for FP32 candidate or graph-similarity stores.

The decision favors retention of the full model's retrieval behavior over the
lowest storage and search cost. The measured gain supports 512d as the common
default, while explicit 256d remains available when those costs matter more.

## Hybrid Defaults (February 2026)

The earlier hybrid-default sweep below tuned citation/reference/semantic budgets;
it did not compare embedding dimensions.

### Scope

#### Seeds

- [arXiv:2506.05209](https://arxiv.org/abs/2506.05209)
- [arXiv:2511.22699](https://arxiv.org/abs/2511.22699)
- [arXiv:2509.20354](https://arxiv.org/abs/2509.20354)
- [arXiv:2508.14040](https://arxiv.org/abs/2508.14040)
- [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

#### Common settings

- `--strategy hybrid`
- `--dataset-split train`
- `--corpus-size 100000`
- `--max-papers 40`
- `--encode-batch-size 96`
- `--no-torch-compile`
- `--theme dark`
- isolated temp cache root (`CITEMESH_CACHE_DIR=/tmp/citemesh-agent-cache...`)

### Hybrid Sweep Matrix

Configs tested end-to-end with full CLI execution:

- `h20_20_10`
- `h20_20_12`
- `h20_20_15`
- `h20_20_20`
- `h20_20_25`
- `h20_20_30`
- `h25_25_25`

Naming format is `h<max_refs>_<max_cites>_<max_semantic>`.

### Results

#### Overall (all 5 seeds)

| Config | Quality Score | Mean Runtime (s) | Mean Nodes | Mean Edges | LCR | Median Weight | P25 Weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `h25_25_25` | 0.904825 | 168.047 | 39.4 | 96.0 | 0.930 | 0.894618 | 0.855580 |
| `h20_20_30` | 0.897266 | 165.207 | 40.0 | 98.2 | 0.905 | 0.896479 | 0.860641 |
| `h20_20_25` | 0.893416 | 189.599 | 39.4 | 96.6 | 0.905 | 0.895379 | 0.852805 |
| `h20_20_20` | 0.891471 | 155.504 | 38.4 | 94.2 | 0.905 | 0.892208 | 0.846904 |
| `h20_20_15` | 0.879880 | 66.348 | 36.6 | 89.6 | 0.905 | 0.883747 | 0.829730 |
| `h20_20_12` | 0.869481 | 113.113 | 35.4 | 86.6 | 0.905 | 0.878746 | 0.820961 |
| `h20_20_10` | 0.868924 | 167.070 | 34.6 | 84.6 | 0.905 | 0.875584 | 0.820299 |

#### Recent-paper subset (first 4 seeds)

Recent subset quality was highest for `h20_20_30` (0.928966), with `h25_25_25` nearly tied (0.927900).

#### Legacy-paper subset (`arxiv:1706.03762`)

Legacy connectivity separated candidates clearly:

- `h25_25_25`: largest component ratio `0.65`
- `h20_20_*`: largest component ratio `0.525`

This was the deciding signal within the initial sweep matrix.

#### Runtime note

Per-run elapsed time has heavy-tail behavior driven by network-bound citation-count enrichment. Use median and upper-quantile runtime when comparing configs; means alone are noisy.

### Remaining Work

- Citation-count enrichment is now batched and visible in progress output, but it can still dominate tail latency on some runs; tighter timeout/retry budgets would make end-to-end runtime more predictable.

### Default Decision

The initial sweep favored `h25_25_25` for legacy connectivity. A later fuzzy-match
and abstract-review study favored a more citation-heavy, reference-light
allocation. The active values are listed in
[CLI Usage](../guides/cli.md).

Why:

- The broad multi-seed sweep established a stable baseline but over-selected
  off-goal papers in manual relevance checks.
- Follow-up fuzzy-match and abstract review favored a citation-heavy,
  reference-light allocation for the discovery goal.
- The updated defaults improve practical triage for "recent follow-up + foundational prior work" without forcing users to set branch-specific knobs each run.
