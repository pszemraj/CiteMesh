# User configuration

`citemesh config` stores personal defaults and an optional Semantic Scholar API key in `config.toml` under the [cache root](../reference/environment.md#platform-variables-used-for-cache-root-resolution). The [config reference](../reference/config/config_reference.toml) is loadable TOML with each key's type, default, and interactions. Copy only the keys you want to change, or start from [example.toml](../reference/config/example.toml).

```bash
# configured values + file path
citemesh config list
# one value, raw on stdout
citemesh config get defaults.semantic_source
citemesh config set defaults.semantic_source arxiv-corpus
citemesh config unset defaults.semantic_source
# config file path, raw on stdout
citemesh config path
```

`set` validates the value with the same rules the loader applies, and rejects unknown keys and invalid values instead of writing them. On the command line, booleans accept `true`/`false`, `1/0`, `yes/no`, or `on/off`, and `export` takes a comma-separated list such as `json,dashboard`. `get` exits nonzero when the key is not set.

`list` shows saved settings, not every effective default. Narrow terminals place each value below its full key; wider terminals use a table. Long values wrap rather than being shortened. Use `get` for an unwrapped value in scripts.

## Precedence

The [config reference](../reference/config/config_reference.toml) defines how saved defaults interact with command-line flags, strategy-specific budgets, and installed indexes. Its Graph and Semantic discovery sections explain which values apply to each strategy. The `[api]` section covers credential precedence separately.

Config defaults, including ignored corpus-only settings, are logged at DEBUG with their keys and file path. The build sidecar records only settings applicable to the selected source and precision.

## API key

```bash
citemesh config set api.s2_api_key YOUR_KEY
```

The `api.s2_api_key` entry in the [config reference](../reference/config/config_reference.toml) covers precedence against the `S2_API_KEY` environment variable and how `config list` and `config get` show the key. The CLI passes a saved key directly to its API client; it does not inject that saved value into child-process environments.
Request a key through [Semantic Scholar](https://www.semanticscholar.org/product/api). See the [API request policy](cli.md#troubleshooting) and [Python API](python-api.md#before-you-build-on-this) for direct builders.

## The file itself

- CiteMesh rewrites it with mode `0600` because it can hold `api.s2_api_key`; broader pre-existing permission bits are narrowed on every write.
- Mutations serialize on a file lock, fail after 10 seconds rather than overwrite, and refuse to clobber an unreadable file. Ordinary reads skip what they cannot use with a warning, as the reference header describes, so a broken config never blocks other commands.
- Unknown keys, such as one a later version removed, load with a warning and survive a rewrite; `citemesh config unset` removes one from the file. Comments do not survive, since the TOML round-trip is value-level.
- To reset, use `citemesh config unset` per key, or delete the path printed by `citemesh config path`. `citemesh cache clear` keeps the file; see [cache maintenance](caching.md#inspecting-and-clearing).
