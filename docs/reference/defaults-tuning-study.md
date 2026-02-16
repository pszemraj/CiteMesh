# Defaults Tuning Study (February 2026)

This document summarizes the parameter study used to tune graph-building defaults for recent-paper workflows.

## Scope

- Strategies studied: `hybrid`, `embedding`
- Seeds:
  - [arXiv:2506.05209](https://arxiv.org/abs/2506.05209)
  - [arXiv:2511.22699](https://arxiv.org/abs/2511.22699)
  - [arXiv:2509.20354](https://arxiv.org/abs/2509.20354)
- Corpus sizes:
  - Broad sweep at `50,000`
  - Targeted validation at `100,000`

## Method

- Fixed common build settings:
  - `--max-papers 40`
  - `--dataset-split train`
  - `--theme dark`
- Compile was disabled during sweeps (`--no-torch-compile`) to reduce runtime variance during long cache hydration.
- Runs were executed with isolated cache roots to avoid cross-run/process lock contention.
- Evaluation was based on exported JSON graph structure and edge weights:
  - nodes/edges
  - connected components and largest component ratio
  - seed degree
  - median / 25th percentile edge weight
  - elapsed runtime

## 50k Sweep

Aggregated over three seeds.

### Hybrid

| Config | Nodes | Edges | Components | Largest Component Ratio | Seed Degree | Median Weight | P25 Weight | Avg Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `h_default` (`20/15/10`) | 29.33 | 71.33 | 1.00 | 1.000 | 4.00 | 0.652 | 0.578 | 44.87 |
| `h_citation_heavy` (`24/20/8`) | 29.67 | 71.33 | 1.33 | 0.992 | 4.67 | 0.644 | 0.592 | 43.39 |
| `h_lean_fast` (`16/12/8`) | 26.00 | 62.00 | 1.00 | 1.000 | 4.67 | 0.640 | 0.577 | 59.03 |
| `h_semantic_heavy` (`18/15/12`) | 30.67 | 73.67 | 1.00 | 1.000 | 3.67 | 0.692 | 0.579 | 29.76 |

### Embedding

| Config | Nodes | Edges | Components | Largest Component Ratio | Seed Degree | Median Weight | P25 Weight | Avg Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e_k2` | 40.00 | 39.00 | 8.33 | 0.300 | 0.00 | 0.744 | 0.665 | 25.02 |
| `e_k3` | 40.00 | 59.00 | 1.67 | 0.917 | 1.33 | 0.734 | 0.646 | 24.98 |
| `e_k4` | 40.00 | 78.33 | 2.33 | 0.858 | 1.00 | 0.722 | 0.634 | 45.32 |
| `e_k5` (follow-up check) | 40.00 | 98.67 | 1.67 | 0.900 | 2.67 | 0.719 | 0.632 | n/a |

## 100k Validation

Targeted validation of leading candidates:

- Hybrid `h_semantic_heavy_100k`
- Embedding `e_k3_100k` with `e_k4_100k` and `e_k5_100k` follow-ups

| Config | Nodes | Edges | Components | Largest Component Ratio | Seed Degree | Median Weight | P25 Weight | Avg Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `h_semantic_heavy_100k` | 30.67 | 73.67 | 1.00 | 1.000 | 5.00 | 0.705 | 0.581 | 184.55* |
| `e_k3_100k` | 40.00 | 59.67 | 2.33 | 0.783 | 3.00 | 0.739 | 0.656 | 34.99 |
| `e_k4_100k` | 40.00 | 78.67 | 2.00 | 0.883 | 2.00 | 0.730 | 0.654 | n/a |
| `e_k5_100k` | 40.00 | 99.00 | 1.67 | 0.900 | 4.00 | 0.726 | 0.645 | n/a |

\* First 100k run includes one-time cache hydration cost; warm-cache subsequent runs were substantially faster.

## Hydration Throughput Follow-Up (February 16, 2026)

A focused follow-up benchmark isolated long-hydration throughput effects using:

- model: `unsloth/embeddinggemma-300m`
- split: `train[:80000]`
- storage: `int8`
- compile: disabled (`--no-torch-compile`) to isolate cache-write behavior

Measured variants:

| Config | Encode Batch | Flush Size | Compression | Hydration Rate (papers/s) | H5 Size (MiB) |
| --- | ---: | ---: | --- | ---: | ---: |
| baseline | 32 | 32 | gzip-1 | 266.97 | 333.1 |
| tuned | 32 | 256 | gzip-1 | 398.61 | 57.3 |
| tuned | 32 | 256 | lzf | 401.47 | 62.2 |

Conclusion:

- Throughput regression was dominated by too-frequent cache flushes, not embedding micro-batch size.
- Increasing flush size to `256` while keeping encode batch at `32` improved sustained hydration throughput by about `1.5x` in this setup.

## Outcome

Defaults were adjusted to:

- `--top-k` default: `3` (from `2`)
- Hybrid implicit semantic cap: `min(12, max-papers - 1)` (from `min(10, max-papers - 1)`)
- `--log-width` default: `140` (from `160`) for more readable terminal output
- Hydration cache flush window: `256` records (encode micro-batch remains `32`)

Rationale:

- `top_k=2` produced overly fragmented embedding graphs.
- `top_k=3` gave the best quality/speed balance in this set.
- Hybrid semantic cap `12` improved coverage and similarity quality without harming connectivity.
- Larger hydration flush windows dramatically reduced long-run cache append overhead while preserving stable encode batch sizing.

## Artifacts

Generated outputs from this study are stored under:

- `out/studies/defaults-sweep-v2-50k`
- `out/studies/defaults-sweep-v2-100k`
- `out/studies/recommended-defaults`
- `out/studies/new-defaults`
