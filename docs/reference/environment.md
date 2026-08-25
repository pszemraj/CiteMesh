# Environment Variables

Runtime environment variables consumed by CiteMesh.

Related docs:

- CLI usage: [Guides: CLI Usage](../guides/cli.md)
- Cache behavior: [Guides: Caching & Data](../guides/caching.md)

## CiteMesh Variables

| Variable | Default | Accepted values | Runtime effect |
| --- | --- | --- | --- |
| `S2_API_KEY` | unset | string | Adds Semantic Scholar API key for authenticated requests; CiteMesh then paces at 1 request/second instead of the anonymous 0.5. When present (even empty), it wins over a configured `api.s2_api_key` in `config.toml`. |
| `CITEMESH_CACHE_DIR` | platform default cache root | filesystem path | Overrides CiteMesh cache root used for embedding/reference caches and the `config.toml` location. |
| `CITEMESH_EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS` | `900` | positive finite number | Overrides embedding-cache inter-process lock timeout; invalid values fall back to default. |

Implementation references:

- [Semantic Scholar API key lookup](../../citemesh/services/semantic_scholar.py)
- [Cache-root override handling](../../citemesh/data/cache.py)
- [User config precedence handling](../../citemesh/core/user_config.py)
- [Embedding lock-timeout handling](../../citemesh/data/embedding_cache.py)
- [Embedding fingerprint verification](../../citemesh/strategies/embedding.py)

## Platform Variables Respected by Cache-Root Resolution

These are not CiteMesh-specific, but CiteMesh honors them when `CITEMESH_CACHE_DIR` is unset:

| Variable | Platform | Effect |
| --- | --- | --- |
| `XDG_CACHE_HOME` | Linux/macOS | Base for default cache root (`$XDG_CACHE_HOME/citemesh`). |
| `LOCALAPPDATA` | Windows | Primary base for default cache root (`%LOCALAPPDATA%\\CiteMesh`). |
| `APPDATA` | Windows | Fallback base when `LOCALAPPDATA` is unset. |

Cache-root behavior details are covered in [Guides: Caching & Data](../guides/caching.md).

## Other Respected Variables

| Variable | Effect |
| --- | --- |
| `MPLBACKEND` | When set, CiteMesh skips its default headless `Agg` matplotlib backend pin and uses your backend for static exports. |
| `CUDA_VISIBLE_DEVICES` | Honored by torch for CUDA device visibility/selection. |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Honored by torch: falls back to CPU for individual ops missing MPS kernels. |
