# Documentation

Start with the [CLI guide](guides/cli.md) for the commands, then [How CiteMesh builds a graph](guides/how-it-works.md) for what happens between a seed paper and a rendered graph. The rest is reference.

## Guides

- [CLI guide](guides/cli.md) — for anyone running `citemesh`: the workflows, the full flag contract, and what to do when a run misbehaves.
- [How CiteMesh builds a graph](guides/how-it-works.md) — for readers who want the mechanism: seed resolution, candidates, embedding, ranking, edge scoring, layout, export.
- [Strategy guide](guides/strategies.md) — for deciding which of the four strategies fits the question you are asking.
- [Caching & Data](guides/caching.md) — for anyone wondering where the disk went, what gets reused between runs, and how to reset it.
- [User configuration](guides/configuration.md) — for people tired of retyping the same flags every run.
- [Python API](guides/python-api.md) — for importing CiteMesh instead of shelling out; unstable before 1.0.

Project overview and installation: [README](../README.md). Change history: [release notes](https://github.com/pszemraj/CiteMesh/releases). Development conventions: [Contributing](../CONTRIBUTING.md).

## Reference

- [Environment variables](reference/environment.md) — for CI and shared machines: every variable CiteMesh reads.
- [Output artifacts](reference/output-artifacts.md) — for consumers of the exports: file formats, the dashboard collection, the sidecar schema.
- [Embedding runtime](reference/embedding-runtime.md) — for tuning the encoder: model defaults, fallback chain, device, precision, and compile policy.
- [Defaults tuning studies](reference/defaults-tuning-study.md) — for anyone who wants to argue with a shipped default: the sweeps behind each one.

## Internals

- [Architecture](internals/architecture.md) — for contributors: package map, dependency rules, test conventions, extension points.
- [Embedding cache internals](internals/embedding-cache.md) — for debugging a cache that crashed: mixin structure, write ordering, recovery invariants.
