# CiteMesh Documentation

Use this index to navigate docs by topic and keep behavior definitions centralized.

## Documentation Contract

- Each topic has one canonical document.
- Non-canonical docs should summarize and link to the canonical page instead of repeating detailed behavior.
- If two docs disagree, the canonical doc in the map below is the source of truth and the other doc should be updated.

## Source-of-Truth Map

| Topic | Canonical doc |
| --- | --- |
| CLI commands, flags, identifiers, output naming, export behavior | [Guides: CLI Usage](guides/cli.md) |
| Cache layout, platform paths, overrides, invalidation | [Guides: Caching & Data](guides/caching.md) |
| Component responsibilities and data flow | [Internals: Architecture](internals/architecture.md) |
| Historical changes and release notes | [Internals: Changelog](internals/changelog.md) |

## Quick Navigation

- [CLI Usage](guides/cli.md) - canonical command-line behavior.
- [Caching & Data](guides/caching.md) - canonical cache/storage behavior.
- [Architecture](internals/architecture.md) - system structure and extension points.
- [Changelog & Key Improvements](internals/changelog.md) - historical updates and future opportunities.

## Scope Notes

This repository does not currently load `config.yaml` at runtime; configuration is handled via CLI flags and environment variables.
The top-level `README.md` is intentionally high-level; operational behavior should be maintained in the canonical docs listed above.
