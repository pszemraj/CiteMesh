# Documentation

New here? Read the [CLI guide](guides/cli.md) for the commands, then [How CiteMesh builds a graph](guides/how-it-works.md) for what actually happens between a seed paper and a rendered graph. The rest is reference. For development conventions see [Contributing](../CONTRIBUTING.md).

## Guides

- Command usage, flag reference, and troubleshooting: [CLI guide](guides/cli.md)
- The pipeline end to end — seed resolution, candidates, embedding, cache, ranking, edge scoring, layout, export: [How CiteMesh builds a graph](guides/how-it-works.md)
- Choosing between the four strategies: [Strategy guide](guides/strategies.md)
- Cache behavior and maintenance: [Caching guide](guides/caching.md)
- Persistent user defaults (`citemesh config`): [Configuration guide](guides/configuration.md)
- Importing CiteMesh from Python (unstable, pre-1.0): [Python API](guides/python-api.md)

Project overview, installation, and current project status: [README](../README.md). Change history: [release notes](https://github.com/pszemraj/CiteMesh/releases).

## Reference

- Runtime environment variables: [reference/environment.md](reference/environment.md)
- Output formats, the interactive dashboard UI, and sidecar schema: [reference/output-artifacts.md](reference/output-artifacts.md)
- Embedding model defaults, fallback chain, and precision/compile behavior: [reference/embedding-runtime.md](reference/embedding-runtime.md)
- The tuning lab notebook — how the shipped defaults were chosen, with the sweeps, fixtures, and tradeoffs behind each one: [reference/defaults-tuning-study.md](reference/defaults-tuning-study.md)

## Internals

- Package map, dependency rules, test conventions, and extension points: [internals/architecture.md](internals/architecture.md)
- Embedding cache crash-safety invariants and mixin structure: [internals/embedding-cache.md](internals/embedding-cache.md)
