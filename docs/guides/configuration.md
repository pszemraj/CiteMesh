# User configuration

`citemesh config` stores personal defaults in `config.toml` under the [cache root](../reference/environment.md#platform-variables-used-for-cache-root-resolution). The [config reference](../reference/config/config_reference.toml) is loadable TOML with each key's type, default, and interactions. Copy only the keys you want to change, or start from [example.toml](../reference/config/example.toml).

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

The [config reference](../reference/config/config_reference.toml) defines how saved defaults interact with command-line flags, strategy-specific budgets, and installed indexes. Its Graph and Semantic discovery sections explain which values apply to each strategy.

Config defaults, including ignored corpus-only settings, are logged at DEBUG with their keys and file path. The build sidecar records only settings applicable to the selected source and precision.

## API key

CiteMesh reads the Semantic Scholar API key only from the `S2_API_KEY` environment variable, never from `config.toml`. Export it from your shell profile, secret manager, or CI secret store; see the [OWASP Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html) for storing secrets properly.

The CLI also reads `S2_API_KEY` from a `.env` file in the working directory or its parents, and a variable already set in the environment wins, including an empty `S2_API_KEY=`, which selects anonymous access. Keep `.env` out of version control.

Request a key through [Semantic Scholar](https://www.semanticscholar.org/product/api). See the [API request policy](cli.md#troubleshooting) and [Python client setup](python-api.md#semantic-scholar-client) for direct builders.

## The file itself

- Mutations serialize on a file lock, fail after 10 seconds rather than overwrite, and refuse to clobber an unreadable file. Ordinary reads skip what they cannot use with a warning, as the reference header describes, so a broken config never blocks other commands.
- Unknown keys, such as one a later version removed, load with a warning and survive a rewrite; `citemesh config unset` removes one from the file. Comments do not survive, since the TOML round-trip is value-level.
- To reset, use `citemesh config unset` per key, or delete the path printed by `citemesh config path`. `citemesh cache clear` keeps the file; see [cache maintenance](caching.md#inspecting-and-clearing).
