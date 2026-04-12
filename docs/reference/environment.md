# Environment Variables

Runtime environment variables consumed by CiteMesh.

Related docs:

- CLI usage: [Guides: CLI Usage](../guides/cli.md)
- Cache behavior: [Guides: Caching & Data](../guides/caching.md)
- Docs index: [Documentation](../README.md)

## CiteMesh Variables

| Variable | Default | Accepted values | Runtime effect |
| --- | --- | --- | --- |
| `S2_API_KEY` | unset | non-empty string | Adds Semantic Scholar API key for higher API limits and authenticated requests. |
| `CITEMESH_CACHE_DIR` | platform default cache root | filesystem path | Overrides CiteMesh cache root used for embedding/reference caches. |
| `CITEMESH_EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS` | `900` | positive finite number | Overrides embedding-cache inter-process lock timeout; invalid values fall back to default. |
| `CITEMESH_STRICT_OFFLINE_FINGERPRINT` | disabled | `1`, `true`, `yes`, `on` (case-insensitive) | Disables legacy offline assumption that cached SHA for `main` may still be reused when full fingerprint verification is unavailable. |

Implementation references:

- [Semantic Scholar API key lookup](../../citemesh/services/semantic_scholar.py)
- [Cache-root override handling](../../citemesh/data/cache.py)
- [Embedding lock-timeout handling](../../citemesh/data/embedding_cache.py)
- [Strict offline fingerprint mode](../../citemesh/strategies/embedding.py)

## Platform Variables Respected by Cache-Root Resolution

These are not CiteMesh-specific, but CiteMesh honors them when `CITEMESH_CACHE_DIR` is unset:

| Variable | Platform | Effect |
| --- | --- | --- |
| `XDG_CACHE_HOME` | Linux/Unix | Base for default cache root (`$XDG_CACHE_HOME/citemesh`). |
| `LOCALAPPDATA` | Windows | Primary base for default cache root (`%LOCALAPPDATA%\\CiteMesh`). |
| `APPDATA` | Windows | Fallback base when `LOCALAPPDATA` is unset. |

Cache-root behavior details are covered in [Guides: Caching & Data](../guides/caching.md).
