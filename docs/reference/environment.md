# Environment Variables

Use this page as the canonical source of truth for runtime environment variables consumed by CiteMesh.

## Scope

- Normative here: variable names, accepted values, defaults, and runtime effect.
- Non-normative here: command syntax and strategy behavior details.
  - CLI contracts: [Guides: CLI Usage](../guides/cli.md)
  - Cache behavior: [Guides: Caching & Data](../guides/caching.md)
  - Documentation ownership map: [Documentation Index](../README.md)

## CiteMesh Variables

| Variable | Default | Accepted values | Runtime effect |
| --- | --- | --- | --- |
| `S2_API_KEY` | unset | non-empty string | Adds Semantic Scholar API key for higher API limits and authenticated requests. |
| `CITEMESH_CACHE_DIR` | platform default cache root | filesystem path | Overrides CiteMesh cache root used for embedding/reference caches. |
| `CITEMESH_EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS` | `60` | positive finite number | Overrides embedding-cache inter-process lock timeout; invalid values fall back to default. |
| `CITEMESH_STRICT_OFFLINE_FINGERPRINT` | disabled | `1`, `true`, `yes`, `on` (case-insensitive) | Disables legacy offline assumption that cached SHA for `main` may still be reused when full fingerprint verification is unavailable. |

Implementation references:

- [Semantic Scholar API key lookup](https://github.com/pszemraj/CiteMesh/blob/main/citemesh/services/semantic_scholar.py)
- [Cache-root override handling](https://github.com/pszemraj/CiteMesh/blob/main/citemesh/data/cache.py)
- [Embedding lock-timeout handling](https://github.com/pszemraj/CiteMesh/blob/main/citemesh/data/embedding_cache.py)
- [Strict offline fingerprint mode](https://github.com/pszemraj/CiteMesh/blob/main/citemesh/strategies/embedding.py)

## Platform Variables Respected by Cache-Root Resolution

These are not CiteMesh-specific, but CiteMesh honors them when `CITEMESH_CACHE_DIR` is unset:

| Variable | Platform | Effect |
| --- | --- | --- |
| `XDG_CACHE_HOME` | Linux/Unix | Base for default cache root (`$XDG_CACHE_HOME/citemesh`). |
| `LOCALAPPDATA` | Windows | Primary base for default cache root (`%LOCALAPPDATA%\\CiteMesh`). |
| `APPDATA` | Windows | Fallback base when `LOCALAPPDATA` is unset. |

Cache-root behavior details remain canonical in [Guides: Caching & Data](../guides/caching.md).
