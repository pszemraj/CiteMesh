# CiteMesh Documentation

Use this index to navigate docs by topic and keep behavior definitions centralized.

## Scope

This page is the canonical map of documentation ownership.

- Each topic has one canonical document.
- Non-canonical documents should summarize in 1-2 lines and link to the canonical source.
- If documents disagree, the canonical document listed below wins.
- Use repository-relative Markdown links so references stay clickable on GitHub and in local previews.

## Source-of-Truth Map

| Topic | Canonical doc |
| --- | --- |
| Project overview, install, and quick start | [Repository README](../README.md) |
| CLI commands, flags, identifiers, output naming, export behavior | [Guides: CLI Usage](guides/cli.md) |
| Strategy behavior and tradeoffs | [Guides: Strategies](guides/strategies.md) |
| Cache paths/layout, hydration/invalidation, cleanup | [Guides: Caching & Data](guides/caching.md) |
| Component responsibilities and data flow | [Internals: Architecture](internals/architecture.md) |
| Historical changes and release notes | [Internals: Changelog](internals/changelog.md) |
| Developer backlog notes (non-normative) | [Developer Notes](dev.md) |

## Internal Docs Rule

`docs/internals/architecture.md` and `docs/internals/changelog.md` are not normative for CLI flag contracts or cache behavior details. Those remain canonical in:

- [Guides: CLI Usage](guides/cli.md)
- [Guides: Caching & Data](guides/caching.md)
