# Documentation

## Guides

- [CLI usage](guides/cli.md): commands, flags, and troubleshooting
- [How CiteMesh builds a graph](guides/how-it-works.md): the pipeline from seed to export
- [Strategies](guides/strategies.md): choosing a discovery method
- [Caching and data](guides/caching.md): reuse, hydration, and maintenance
- [Portable index laptop test](guides/portable-index-laptop-test.md): repeatable import and unseen-paper checks across devices
- [User configuration](guides/configuration.md): persistent defaults and credentials
- [Python API](guides/python-api.md): builders, installed indexes, and exporters for scripts and notebooks

## Reference

- [Config reference](reference/config/config_reference.toml): every `config.toml` key, with [example.toml](reference/config/example.toml) to start from
- [Environment variables](reference/environment.md)
- [Output artifacts](reference/output-artifacts.md)
- [Embedding runtime](reference/embedding-runtime.md)
- [Defaults tuning studies](reference/defaults-tuning-study.md)

## Internals

- [Architecture](internals/architecture.md)
- [Embedding cache](internals/embedding-cache.md)
- [Hybrid discovery status](internals/hybrid-discovery-status.md): what hybrid does after the discovery rework and how it was evaluated

[Installation](../README.md#quick-start) and [release notes](https://github.com/pszemraj/CiteMesh/releases).
