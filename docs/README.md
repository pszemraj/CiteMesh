# CiteMesh Documentation

Use this index to navigate documentation by topic. Each topic has one canonical reference; other docs should link there instead of restating behavior-level details.

## Source-of-Truth Map

| Topic | Canonical doc |
| --- | --- |
| CLI commands, flags, identifiers, output naming, export behavior | [Guides: CLI Usage](guides/cli.md) |
| Cache layout, overrides, invalidation | [Guides: Caching & Data](guides/caching.md) |
| Component responsibilities and data flow | [Internals: Architecture](internals/architecture.md) |
| Historical changes and release notes | [Internals: Changelog](internals/changelog.md) |

## Guides

- [CLI Usage](guides/cli.md) - canonical command-line reference.
- [Caching & Data](guides/caching.md) - canonical cache/storage reference.

## Internals

- [Architecture](internals/architecture.md) - system structure and extension points.
- [Changelog & Key Improvements](internals/changelog.md) - major updates and future opportunities.

This repository does not currently load `config.yaml` at runtime; configuration is handled via CLI flags and environment variables.
