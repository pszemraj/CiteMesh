# Defaults Tuning Study (February 2026)

Hybrid-default sweep results used to choose the current discovery-oriented defaults.

Related docs:

- CLI defaults and flags: [CLI Usage](../guides/cli.md)
- Strategy behavior overview: [Strategy Guide](../guides/strategies.md)
- Embedding runtime policy: [Embedding Runtime](embedding-runtime.md)

## Scope

### Seeds

- [arXiv:2506.05209](https://arxiv.org/abs/2506.05209)
- [arXiv:2511.22699](https://arxiv.org/abs/2511.22699)
- [arXiv:2509.20354](https://arxiv.org/abs/2509.20354)
- [arXiv:2508.14040](https://arxiv.org/abs/2508.14040)
- [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

### Common settings

- `--strategy hybrid`
- `--dataset-split train`
- `--corpus-size 100000`
- `--max-papers 40`
- `--encode-batch-size 96`
- `--no-torch-compile`
- `--theme dark`
- isolated temp cache root (`CITEMESH_CACHE_DIR=/tmp/citemesh-agent-cache...`)

## Hybrid Sweep Matrix

Configs tested end-to-end with full CLI execution:

- `h20_20_10`
- `h20_20_12`
- `h20_20_15`
- `h20_20_20`
- `h20_20_25`
- `h20_20_30`
- `h25_25_25`

Naming format is `h<max_refs>_<max_cites>_<max_semantic>`.

## Results

### Overall (all 5 seeds)

| Config | Quality Score | Mean Runtime (s) | Mean Nodes | Mean Edges | LCR | Median Weight | P25 Weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `h25_25_25` | 0.904825 | 168.047 | 39.4 | 96.0 | 0.930 | 0.894618 | 0.855580 |
| `h20_20_30` | 0.897266 | 165.207 | 40.0 | 98.2 | 0.905 | 0.896479 | 0.860641 |
| `h20_20_25` | 0.893416 | 189.599 | 39.4 | 96.6 | 0.905 | 0.895379 | 0.852805 |
| `h20_20_20` | 0.891471 | 155.504 | 38.4 | 94.2 | 0.905 | 0.892208 | 0.846904 |
| `h20_20_15` | 0.879880 | 66.348 | 36.6 | 89.6 | 0.905 | 0.883747 | 0.829730 |
| `h20_20_12` | 0.869481 | 113.113 | 35.4 | 86.6 | 0.905 | 0.878746 | 0.820961 |
| `h20_20_10` | 0.868924 | 167.070 | 34.6 | 84.6 | 0.905 | 0.875584 | 0.820299 |

### Recent-paper subset (first 4 seeds)

Recent subset quality was highest for `h20_20_30` (0.928966), with `h25_25_25` nearly tied (0.927900).

### Legacy-paper subset (`arxiv:1706.03762`)

Legacy connectivity separated candidates clearly:

- `h25_25_25`: largest component ratio `0.65`
- `h20_20_*`: largest component ratio `0.525`

This was the deciding signal for the default choice.

### Runtime note

Per-run elapsed time has heavy-tail behavior driven by network-bound citation-count enrichment. Use median and upper-quantile runtime when comparing configs; means alone are noisy.

## Remaining Work

- Citation-count enrichment is now batched and visible in progress output, but it can still dominate tail latency on some runs; tighter timeout/retry budgets would make end-to-end runtime more predictable.

## Default Decision

Defaults are set to:

- `--max-papers`: `45` (hybrid when omitted)
- `--max-references`: `12` (hybrid when omitted)
- `--max-citations`: `45` (hybrid when omitted)
- hybrid implicit `--max-semantic`: `min(20, max-papers - 1)`

Why:

- Broad multi-seed sweep established a stable baseline but over-selected off-goal papers in manual relevance checks.
- Follow-up fuzzy-match + abstract review study (PolyCom seed, in-process run set) favored citation-heavy / reference-light allocation (`45/12/20`) for the discovery goal.
- The updated defaults improve practical triage for "recent follow-up + foundational prior work" without forcing users to set branch-specific knobs each run.

## Related Outcomes From Earlier Tuning

These remain in effect from earlier study stages:

- embedding `--top-k` default `4`
- hydration flush window `256`
- log width default `140`

## Artifacts

Local raw outputs for this sweep are under:

- `out/studies/defaults-sweep-v4-2026-02-16/`
- `out/studies/defaults-sweep-v4-2026-02-16/runs.csv`
- `out/studies/defaults-sweep-v4-2026-02-16/summary_all.csv`
- `out/studies/defaults-sweep-v4-2026-02-16/summary_recent.csv`
- `out/studies/defaults-sweep-v4-2026-02-16/summary_legacy.csv`
- `out/OLD-polynomial-composition-activati-b68d34de/study-inproc-20260220-224514/guidance.md`
