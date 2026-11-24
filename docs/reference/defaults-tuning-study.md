# Defaults tuning studies

Measurements and tradeoffs behind CiteMesh's semantic threshold, embedding dimension, and embedding text.

## Semantic edge threshold (September 2026)

The development set selected a semantic eligibility threshold of **0.74** for EmbeddingGemma's symmetric STS embeddings at 512 dimensions by maximum F1 on a fixed 0.65-0.80 grid in 0.01 increments, breaking ties by precision and then higher threshold. A separate validation set assesses the selected value.

Two independent fixture evaluations used full primary-source arXiv abstracts, with pair labels fixed before inspecting scores. Development has 14 papers from translation, pretrained language models, dense retrieval, graph learning, diffusion, and residual vision networks. Its 81 included pairs contain 13 positives and 68 negatives; 16 negatives deliberately share a nearby ML topic. Ten ambiguous attention/efficiency/representation-lineage pairs are excluded and listed explicitly in the fixture. Transformer/BERT and Transformer/RoBERTa are positives because architectural lineage matters for discovery even across tasks.

Validation has 10 different papers from object detection, semantic segmentation, speech recognition, and policy optimization: 8 positives and 37 negatives. Nine detection-versus-segmentation pairs are its nearest negative boundary. Labels describe the same core research problem or an explicit core methodological continuation, rather than requiring identical methods.

Both evaluations used `unsloth/embeddinggemma-300m`, the graph-similarity formatter, CPU inference with four threads, automatic checkpoint dtype selection, verified BF16 autocast, and normalized FP32 outputs. The [fixtures](../../tests/_semantic_abstracts.py) contain the frozen texts, source links, and labels. The [slow tests](../../tests/test_semantic_quality.py) recompute the development selection and check edge decisions through both embedding and hybrid scorers. Equal years and citation counts isolate the semantic gate.

| Set | Threshold | Related retained | Unrelated admitted | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 0.71 | 12/13 | 3/68 | 0.800 | 0.923 | 0.857 |
| Development | 0.72 | 10/13 | 3/68 | 0.769 | 0.769 | 0.769 |
| Development | **0.74** | **10/13** | **0/68** | **1.000** | **0.769** | **0.870** |
| Validation | 0.71 | 8/8 | 3/37 | 0.727 | 1.000 | 0.842 |
| Validation | 0.72 | 8/8 | 2/37 | 0.800 | 1.000 | 0.889 |
| Validation | **0.74** | **7/8** | **1/37** | **0.875** | **0.875** | **0.875** |

The higher threshold favors precision: combined false edges drop from five to one, at the cost of one additional missed related pair. Validation F1 slightly decreases; 0.74 is a development-selected precision tradeoff, not a universal optimum. In validation, RetinaNet/YOLO (0.7392) becomes a false negative and Faster R-CNN/FCN (0.7445) remains a false positive under these labels. Transformer/BERT (0.7146) remains below the gate. Lowering the threshold to 0.71 recovers it but admits additional unrelated edges. Shared references can still qualify pairs independently in hybrid graphs.

These small, manually labeled fixtures are regressions rather than a representative benchmark of scientific relevance. They do not calibrate other checkpoints or dimensions. Set `--min-semantic-similarity` or [`defaults.min_semantic_similarity`](config/config_reference.toml) after evaluating the intended representation and discovery tradeoff.

With the designated model cached locally:

```bash
pytest -n 0 -m slow tests/test_semantic_quality.py
```

## Embedding dimensions (September 2026)

### Decision

The paired corpus study below compares the previous 256-dimensional default with 512 dimensions. The smaller space reduced vector storage but lost a substantial share of the full model's nearest neighbors. Moving to 512 recovered more of those neighbors with essentially the same GPU corpus encoding time and an 11.4% increase in complete cache size. Local search became slower.

The [EmbeddingGemma profile](embedding-runtime.md#embeddinggemma-profile) controls the dimension default. This experiment directly measured CUDA corpus retrieval; it did not measure every build or runtime path.

### Paired corpus and runtime

The study completed on September 5, 2026 (UTC), using separate temporary CiteMesh caches containing the **same 100,000 papers in the same order**. The main user cache was not modified.

- Dataset: `librarian-bots/arxiv-metadata-snapshot/train`, revision `47141d6fd17f52b65424d246665334914cac3011`, last updated August 31, 2026.
- Selection: scan all 3,148,882 records and select the newest 100,000 submissions by arXiv ID, from `2605.20790` through `2608.27458`, preserving source order. This covers May-August 2026 in that snapshot, not a live September feed or recently revised older submissions.
- Model: `unsloth/embeddinggemma-300m`, artifact `bfa3c846ac738e62aa61806ef9112d34acb1dc5a`.
- Runtime: RTX 5090, torch `2.13.0+cu130`, sentence-transformers `6.0.0`, transformers `5.2.0`; BF16 autocast, Flash Attention 2, `torch.compile`, batch size 128, and eight CPU threads. Checkpoint weights used automatic dtype selection, embedding outputs were FP32, and no FP16 was used.
- Storage: INT8 with gzip level 1 and a binary sign index, using the same deterministic 2,000-record reservoir sample for per-dimension min/max calibration in each run.

Each dimension had its own GPU corpus inference run through CiteMesh's normal calibration and hydration path, with the prepared local rows supplied as input. During the 512d run, the recorder retained the model's full 768d output before truncating and normalizing to 512d for the cache. This provided a reference without a third corpus inference run. The independently encoded 256d vectors agreed with normalized 256d prefixes of that reference to mean cosine 1.000000 and minimum 0.999992.

### Focal paper selection and comparison method

Thirty queries used focal paper titles and abstracts with CiteMesh's retrieval-query formatting. Twelve established papers covered language modeling, retrieval, vision, graph learning, and gravitational waves. Eighteen recent focal papers were selected with NumPy RNG seed `20260905`, two per primary category from the selected corpus. Self matches were removed before comparing the top 20 results.

| Focal paper group | arXiv IDs |
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

The reference was exact cosine search over full **768-dimensional FP32 outputs** from the same model and corpus. Top-20 retention is the fraction of reference neighbors found in a compared list, averaged equally across the 30 focal papers. It is agreement with the untruncated model, **not relevance accuracy**: that model can also retrieve poor matches, and a different neighbor may be equally relevant.

Two search budgets were measured:

- **Graph candidates:** the default 40-paper embedding build requests 160 candidates, shortlists 1,280 via the binary index (8x), and rescores them. The comparison takes its first 20 non-self neighbors, before hybrid fusion or graph scoring.
- **Local cache search:** request 21 results, shortlist 168 (8x), then remove any self match and retain 20. Search latency excludes query encoding.

### Results

| Mean top-20 retention against the full 768d reference | 256d | 512d |
| --- | ---: | ---: |
| Exact FP32 search | 53.7% | 75.0% |
| Exact INT8 search | 53.5% | 75.2% |
| Graph candidate search | 53.3% | 74.7% |
| Local cache search | 46.5% | 69.7% |

512d retained more reference neighbors on **all 30 focal papers** in the graph candidate comparison. Its list overlapped the 256d list by 59.3% on average. For Megalodon, reference retention increased from 35% to 55%; for RAG, from 45% to 75%; for Mamba, from 75% to 85%.

| Build and storage measurement | 256d | 512d |
| --- | ---: | ---: |
| Corpus encoding time | 89.6 s | 88.2 s |
| Calibration encoding and compilation warmup | 39.9 s | 32.0 s |
| Total hydration, including calibration and writes | 137.0 s | 127.3 s |
| Vectors, binary index, and calibration | 26.9 MB | 53.8 MB |
| SQLite metadata | 208.2 MB | 208.2 MB |
| Complete cache | 235.2 MB | 262.1 MB |
| Median local cache search | 99.7 ms | 165.8 ms |

MB uses decimal units. Vector storage approximately doubled, while complete cache size rose only 11.4% because metadata dominated this corpus. Local search was about 66% slower. These were single paired runs with 781 full 128-record batches each, not repeated timing trials. The second run reused compiler artifacts; the warmup difference does not establish that 512d is faster. Corpus encoding timers include the encoder call and output conversion, but exclude the separate FP32 reference-file write.

Dimension truncation and binary shortlisting affected neighbors much more than fresh INT8 quantization:

| Mean top-20 retention against the same dimension | 256d | 512d |
| --- | ---: | ---: |
| Exact INT8 versus exact FP32 | 98.5% | 99.2% |
| Graph candidate search versus exact INT8 | 92.2% | 97.5% |
| Local cache search versus exact INT8 | 64.5% | 82.8% |

Only 0.0964% / 0.0957% of coordinates were clipped in the respective fresh caches. The smaller local-search shortlist caused additional loss independently of INT8 rounding; this study did not change the shortlist settings.

### Qualitative review and limits

Two GPT-5.6 Sol reviewers each assessed six different established focal papers using blinded A/B top-five lists, titles, and abstracts, ignoring cosine scores as evidence of relevance. Preferences were **512d: 7, 256d: 4, tie: 1**. These are model-assisted judgments, not human labels or duplicate independent reviews. Several preferences depended on one differing result.

- 512d was preferred for Megalodon, Transformer, RAG, CLIP, DDPM, SAM, and the binary black hole merger paper. For RAG, it offered more retrieval-method coverage and fewer repetitive historical-document/OCR applications.
- 256d was preferred for DPO, NeRF, GCN, and the GPT-4 report; differing results were closer to the focal paper's method or breadth. Mamba was tied.
- Both Megalodon lists still contained tangential papers. Higher dimensions did not make every neighbor useful.

Both saved caches reopened with 100,000 rows and reused hydration without loading the dataset. CiteMesh's actual graph candidate search reproduced all 60 ranked lists exactly. The evaluation did not measure the full 3.15-million-paper corpus, Semantic Scholar fusion, graph topology, or real CPU/MPS execution. The 11.4% complete-cache increase applies to these INT8 corpus caches; it is not an estimate for FP32 candidate or graph-similarity stores.

The decision favors retention of the full model's retrieval behavior over the lowest storage and search cost. Explicit 256d remains available when those costs matter more.

## Embedding text (November 2026)

### Decision

Retrieval keeps both title and abstract in the [query and document prompts](embedding-runtime.md#task-specific-vector-spaces). Dropping the title returned fewer important papers in the first ten. Title-only retrieval returned fewer useful papers. Combining separate title and abstract encodings returned fewer useful papers and requires two encodes per paper.

### Method

The offline replay used sixteen machine-learning focal papers from 2021 to 2026 and the earlier four-list hybrid ranker (`rrf_dd2_direct_rec`). Each pool held references, citers, up to 200 recommendations, and up to 180 corpus neighbors per focal embedding role. The replay retained 44 papers per focal paper. Each variant re-encoded candidates as documents and the focal paper as both query and document, using normalized arXiv text for corpus rows and Semantic Scholar text for other candidates. A blind Claude Sonnet judge graded titles and abstracts. Useful is 1 or 2, important is 2, and tangent is 0. Ungraded papers count only in the denominator.

- **TA**: title and abstract, as shipped.
- **A**: abstract only (`title: none | text: A`); papers without an abstract keep their TA text.
- **T**: title only (`title: T | text:`); papers without a title keep their TA text.
- **SUM**: one stored vector, the normalized sum of the T and A vectors, at two encodes per paper.
- **PAIR**: two stored vectors per paper; a pair scores the mean of its title-title and abstract-abstract cosines.

### Results

Shares of the returned list over all 16 focal papers (160 slots at k = 10, 320 at k = 20, 704 at k = 44):

| variant | k | useful | important | tangent | ungraded |
|---|---|---|---|---|---|
| TA | 10 | 86.2% | 61.9% | 13.8% | 0.0% |
| TA | 20 | 81.2% | 46.9% | 18.8% | 0.0% |
| TA | 44 | 72.2% | 35.8% | 26.3% | 1.6% |
| A | 10 | 85.0% | 54.4% | 14.4% | 0.6% |
| A | 20 | 79.4% | 45.6% | 18.4% | 2.2% |
| A | 44 | 65.8% | 32.2% | 25.4% | 8.8% |
| T | 10 | 63.1% | 37.5% | 18.1% | 18.8% |
| T | 20 | 53.8% | 26.2% | 18.4% | 27.8% |
| T | 44 | 41.6% | 19.6% | 16.9% | 41.5% |
| SUM | 10 | 83.8% | 52.5% | 15.0% | 1.2% |
| SUM | 20 | 74.4% | 41.6% | 21.6% | 4.1% |
| SUM | 44 | 59.4% | 30.1% | 25.7% | 14.9% |
| PAIR | 10 | 80.6% | 50.6% | 18.8% | 0.6% |
| PAIR | 20 | 76.9% | 43.8% | 19.7% | 3.4% |
| PAIR | 44 | 61.5% | 31.0% | 24.6% | 13.9% |

TA and A returned similar numbers of useful and tangent papers at k = 10 and k = 20. At k = 10, TA returned more useful papers for 3 focal papers and fewer for 4. TA returned more important papers in its top ten for 10 focal papers and fewer for 3 (sign test p = 0.09). For useful papers, TA beat SUM at k = 20 on 10 focal papers and lost on none (p = 0.002), beat PAIR at k = 10 on 7 and lost on 1 (p = 0.07), and beat T at k = 10 on 15 of 16, tying on the other.

The same replay with the stored pre-normalization vectors scores 86.9% useful, 60.0% important and 13.1% tangent at k = 10, and 70.7% useful and 29.3% tangent at k = 44 with nothing ungraded, so normalization matches the old text at the top and returns 3 points fewer tangents over the full list.

### Limits

- Each variant re-selected its corpus neighbours from each focal paper's wide candidate table (2,260 to 20,093 corpus papers per focal paper: the TA neighbourhoods of the focal paper, its references, and its citers), not from the whole index, so a variant could not retrieve a paper TA never reached.
- At k = 44, A leaves 8.8% of its list ungraded. Counting all of those papers as useful raises A to at most 74.6%, compared with TA's 72.2% to 73.8%. The ungraded papers leave the full-list ordering uncertain.
- The judge reads only titles and abstracts, which may favour semantically retrieved papers over structurally relevant ones.
- Graph edges were not compared: the symmetric similarity role keeps `T. A`, and the 0.74 threshold above was calibrated on that text.
