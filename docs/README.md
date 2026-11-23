# Documentation

New here: [install and build a first graph](../README.md#quick-start), then pick a [strategy](guides/strategies.md). Release notes are on the [GitHub releases page](https://github.com/pszemraj/CiteMesh/releases).

## Guides

- [CLI usage](guides/cli.md): commands, flags, and troubleshooting
- [Strategies](guides/strategies.md): choosing a discovery method
- [How CiteMesh builds a graph](guides/how-it-works.md): the pipeline from focal paper to export
- [User configuration](guides/configuration.md): persistent defaults and the API key
- [Caching and data](guides/caching.md): reuse, hydration, portable corpus indexes, and maintenance
- [Python API](guides/python-api.md): builders, installed indexes, and exporters for scripts and notebooks
- [Portable index laptop test](guides/portable-index-laptop-test.md): repeatable import and unseen-paper checks across devices

## Reference

- [Config reference](reference/config/config_reference.toml): every `config.toml` key, with [example.toml](reference/config/example.toml) to start from
- [Environment variables](reference/environment.md)
- [Output artifacts](reference/output-artifacts.md): file formats, schemas, and output locations
- [Embedding runtime](reference/embedding-runtime.md): model selection, precision, and device policy
- [Defaults tuning studies](reference/defaults-tuning-study.md): the experiments behind tuned defaults

## Internals

- [Architecture](internals/architecture.md): package boundaries, extension points, and [hosted demo maintenance](internals/architecture.md#hosted-demo)
- [Embedding cache](internals/embedding-cache.md)
- [Hybrid discovery status](internals/hybrid-discovery-status.md): discovery evidence and evaluation limits
