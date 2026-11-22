# Environment variables

## CiteMesh variables

| Variable | Default | Accepted values | Runtime effect |
| --- | --- | --- | --- |
| `S2_API_KEY` | unset | string | Semantic Scholar credentials; see [API-key precedence](../guides/configuration.md#api-key) and [request policy](../guides/cli.md#troubleshooting). |
| `CITEMESH_CACHE_DIR` | platform default cache root | filesystem path (`~` expanded) | Overrides the root for paper, reference, and embedding caches and `config.toml`. |
| `CITEMESH_EMBEDDING_CACHE_LOCK_TIMEOUT_SECONDS` | `900` | positive finite number | Overrides embedding-cache inter-process lock timeout; invalid values fall back to default. |

Implementation: [API client](../../src/citemesh/services/semantic_scholar/client.py), [cache-root resolution](../../src/citemesh/data/cache.py), [config precedence](../../src/citemesh/cli/build_options.py), and [lock timeout](../../src/citemesh/data/embedding_cache/constants.py).

## Platform variables used for cache-root resolution

When `CITEMESH_CACHE_DIR` is unset or empty, Linux and macOS default to `~/.cache/citemesh` and Windows uses the first available base below.

| Variable | Platform | Effect |
| --- | --- | --- |
| `XDG_CACHE_HOME` | Linux/macOS | Base for default cache root (`$XDG_CACHE_HOME/citemesh`). |
| `LOCALAPPDATA` | Windows | Primary base for default cache root (`%LOCALAPPDATA%\CiteMesh`). |
| `APPDATA` | Windows | Fallback base when `LOCALAPPDATA` is unset or empty (`%APPDATA%\CiteMesh`). With both unset or empty, the root falls back to `%USERPROFILE%\AppData\Local\CiteMesh`. |

A non-empty `CITEMESH_CACHE_DIR` or `XDG_CACHE_HOME` suppresses the legacy macOS cache-location hint. For cache contents and maintenance, see [Caching and Data](../guides/caching.md).

## Other respected variables

| Variable | Effect |
| --- | --- |
| `MPLBACKEND` | Selects Matplotlib's backend before import. CiteMesh preserves the global selection; a static figure using the main-thread-only MacOSX backend receives its own Agg canvas. |
| `COLORFGBG` | First `--theme auto` hint. With at least two semicolon-separated fields, the last parseable integer selects dark below `8` and light otherwise. |
| `DARKMODE` | When set to `1`, selects dark mode for `--theme auto` if `COLORFGBG` did not resolve a theme. Otherwise macOS uses the system appearance and other platforms fall back to light. |
| `CUDA_VISIBLE_DEVICES` | Honored by torch for CUDA device visibility/selection. |
| `TRITON_DEFAULT_FP_FUSION` | Controls Triton's multiply/add fusion; see [compile policy](embedding-runtime.md#compile-policy) for CiteMesh's default and override behavior. |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Honored by torch: falls back to CPU for individual ops missing MPS kernels. |
| `NO_COLOR` | Honored by Rich: disables ANSI color in console and help output; non-color emphasis such as bold may remain. |
| `HF_TOKEN` | Honored by huggingface_hub for authenticated model/dataset downloads and bucket access; see [bucket permissions](../guides/caching.md#import-and-selection) and [model access](embedding-runtime.md#model-selection-and-fallback). |
| `HF_HOME` | Honored by huggingface_hub: relocates the Hugging Face cache holding downloaded model checkpoints and corpus datasets (separate from the CiteMesh cache root). |
