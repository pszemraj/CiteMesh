# Environment Variables

Every environment variable CiteMesh reads, plus the platform and library variables it honors.

## CiteMesh variables

| Variable | Default | Accepted values | Runtime effect |
| --- | --- | --- | --- |
| `S2_API_KEY` | unset | string | CiteMesh paces authenticated requests at 1 request/second and anonymous requests at 0.5. Credential selection follows [API-key precedence](../guides/configuration.md#api-key). |
| `CITEMESH_CACHE_DIR` | platform default cache root | filesystem path (`~` expanded) | Overrides CiteMesh cache root used for embedding/reference caches and the `config.toml` location. |
| `CITEMESH_EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS` | `900` | positive finite number | Overrides embedding-cache inter-process lock timeout; invalid values fall back to default. |

Implementation: API-key lookup in `services/semantic_scholar/client.py`, cache-root override (including the Windows `LOCALAPPDATA` / `APPDATA` fallback) in `data/cache.py`, config precedence in `data/user_config.py`, lock timeout in `data/embedding_cache/constants.py`.

## Platform variables used for cache-root resolution

With no override, Linux and macOS use `~/.cache/citemesh`; Windows uses the first available base below. `CITEMESH_CACHE_DIR` overrides all of them.

| Variable | Platform | Effect |
| --- | --- | --- |
| `XDG_CACHE_HOME` | Linux/macOS | Base for default cache root (`$XDG_CACHE_HOME/citemesh`). |
| `LOCALAPPDATA` | Windows | Primary base for default cache root (`%LOCALAPPDATA%\\CiteMesh`). |
| `APPDATA` | Windows | Fallback base when `LOCALAPPDATA` is unset (`%APPDATA%\\CiteMesh`). With both unset, the root falls back to `%USERPROFILE%\\AppData\\Local\\CiteMesh`. |

Cache-root behavior is covered in [Caching & Data](../guides/caching.md). An explicit `CITEMESH_CACHE_DIR` or `XDG_CACHE_HOME` suppresses the legacy macOS cache-location hint.

## Other respected variables

| Variable | Effect |
| --- | --- |
| `MPLBACKEND` | Selects Matplotlib's backend before import. CiteMesh preserves the global selection; a static figure using the main-thread-only MacOSX backend receives its own Agg canvas. |
| `COLORFGBG` | First `--theme auto` hint. The final terminal background-color index selects light or dark. |
| `DARKMODE` | When set to `1`, selects dark mode for `--theme auto` if `COLORFGBG` did not resolve a theme. |
| `CUDA_VISIBLE_DEVICES` | Honored by torch for CUDA device visibility/selection. |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Honored by torch: falls back to CPU for individual ops missing MPS kernels. |
| `NO_COLOR` | Honored by Rich: disables ANSI color in console and help output; non-color emphasis such as bold may remain. |
| `HF_TOKEN` | Honored by huggingface_hub for authenticated model/dataset downloads. Required for the license-gated `google/embeddinggemma-300m` fallback checkpoint. |
| `HF_HOME` | Honored by huggingface_hub: relocates the Hugging Face cache holding downloaded model checkpoints and corpus datasets (separate from the CiteMesh cache root). |
