# User configuration

`citemesh config` stores personal defaults and an optional Semantic Scholar API key in `config.toml` under the [cache root](../reference/environment.md#platform-variables-used-for-cache-root-resolution). The [config reference](../reference/config/config_reference.toml) documents every key, with its type, default, and interactions; [example.toml](../reference/config/example.toml) is a small config to start from.

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

## Precedence

Build defaults resolve as explicit CLI flag, then `config.toml`, then the built-in value. Credentials use the separate [API-key precedence](#api-key).

Configured values behave like *your* built-in defaults rather than like flags you typed:

- They never trigger "unsupported option for strategy" errors: `defaults.device` does not break a `--strategy citation` run, it is just unused there.
- They outrank tuned implicit defaults such as hybrid's 45-paper limit and its 1000-record citation and reference fetch limits. The reference's Graph section explains what that means for a cap chosen for citation builds.
- They lose to an explicit flag that implies a different mode, such as `--corpus-size` against a configured `semantic_source = "candidates"`. The reference lists these flags under Semantic discovery.
- A configured `semantic_source = "both"` is never rejected for the strategy in use: the reference's `semantic_source` entry says what each strategy and local search read it as. Only the command-line flag `--semantic-source both` requires `--strategy hybrid`.

Config defaults, including ignored corpus-only settings, are logged at DEBUG with their keys and file path. The build sidecar records only settings applicable to the selected source and precision.

## Supported keys

The [config reference](../reference/config/config_reference.toml) lists every `[defaults]` and `[api]` key. It is a loadable TOML file: uncommented keys hold their built-in defaults, so copy only the keys you want to change.

The former `max_semantic` key was removed along with `--max-semantic`. A config file that still contains it loads, and the key is ignored with an unknown-key warning; `citemesh config unset defaults.max_semantic` removes the stale key from the file.

## API key

```bash
citemesh config set api.s2_api_key YOUR_KEY
```

The `api.s2_api_key` entry in the [config reference](../reference/config/config_reference.toml) covers precedence against the `S2_API_KEY` environment variable and how `config list` and `config get` show the key. The CLI passes a saved key directly to its API client; it does not inject that saved value into child-process environments.

Request a key through [Semantic Scholar](https://www.semanticscholar.org/product/api). Direct-builder behavior is in the [Python API](python-api.md#before-you-build-on-this).

## The file itself

- CiteMesh rewrites it with mode `0600` because it can hold `api.s2_api_key`; broader pre-existing permission bits are narrowed on every write.
- Mutations serialize on a file lock, fail after 10 seconds rather than overwrite, and refuse to clobber an unreadable file. Ordinary reads skip what they cannot use with a warning, as the reference header describes, so a broken config never blocks other commands.
- Unknown keys survive a rewrite; comments do not, since the TOML round-trip is value-level.
- To reset, use `citemesh config unset` per key, or delete the path printed by `citemesh config path`. `citemesh cache clear` keeps the file; see [cache maintenance](caching.md#inspecting-and-clearing).
