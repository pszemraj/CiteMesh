# User Configuration

`citemesh config` saves the flags you would otherwise retype on every run — a preferred theme, corpus-backed semantic sourcing, a Semantic Scholar API key — into a TOML file at `<cache_root>/config.toml`.

```bash
# configured values + file path
citemesh config list
# one value, raw on stdout
citemesh config get defaults.semantic_source
citemesh config set defaults.semantic_source arxiv-corpus
citemesh config unset defaults.semantic_source
citemesh config path
```

`set` rejects unknown keys and invalid values. Booleans accept `true`/`false`, `1/0`, `yes/no`, or `on/off`; `export` takes a comma-separated list such as `json,dashboard`. Other accepted values follow the [CLI flag contracts](cli.md#flag-reference).

## Precedence

Build defaults resolve as explicit CLI flag, then `config.toml`, then the built-in value. Credentials use the separate [API-key precedence](#api-key).

Configured values behave like *your* built-in defaults rather than like flags you typed:

- They never trigger "unsupported option for strategy" errors: `defaults.device` does not break a `--strategy citation` run, it is just unused there.
- They outrank tuned implicit defaults such as hybrid's citation and reference budgets.
- They lose to a flag that implies a mode: `--dataset-source` or `--corpus-size` implies `--semantic-source arxiv-corpus` against a configured `candidates`, and `--candidate-pool-size` implies `candidates` symmetrically, leaving config-only settings for the other mode inert. `--all-corpus` likewise beats a configured `corpus_size`.

Applied config defaults are logged at DEBUG with the keys and the file path. Inert corpus-only defaults are reported at INFO and left out of the build sidecar, which records only settings applicable to the selected source and precision.

## Supported keys

Each `[defaults]` key supplies the corresponding CLI option. `encode_batch_size` maps to `--batch-size`; `search_mode` maps to `search --mode`. [Local search](cli.md#find-a-seed-paper) also uses applicable saved embedding settings.

| Group | Keys |
| --- | --- |
| Graph | `strategy`, `max_papers`, `max_semantic`, `max_citations`, `max_references`, `top_k` |
| Model and runtime | `model`, `model_profile`, `model_revision`, `device`, `truncate_dim`, `encode_batch_size`, `torch_compile` |
| Semantic source | `semantic_source`, `candidate_pool_size`, `min_semantic_similarity` |
| Corpus and storage | `dataset_source`, `dataset_split`, `corpus_size`, `streaming`, `storage_precision`, `binary_prefilter`, `calibration_sample_size` |
| Output | `export`, `theme` |
| Search | `search_mode` |

## API key

For the CLI, `S2_API_KEY` takes precedence over `[api].s2_api_key`, including an explicitly empty environment value, which selects anonymous access. `config list` masks the saved key; `config get api.s2_api_key` prints it in full. The CLI passes the key directly to its API client without adding it to subprocess environments.

```bash
citemesh config set api.s2_api_key YOUR_KEY
```

Request a key through [Semantic Scholar](https://www.semanticscholar.org/product/api). [Direct Python builders](python-api.md#before-you-build-on-this) do not apply saved CLI configuration.

```toml
[defaults]
search_mode = "local"
semantic_source = "arxiv-corpus"
theme = "dark"
export = ["json", "dashboard"]

[api]
s2_api_key = "your-key-here"
```

## The file itself

- [Cache-root settings](../reference/environment.md#platform-variables-used-for-cache-root-resolution) determine its location. [Cache clearing](caching.md#inspecting-and-clearing) preserves it.
- CiteMesh rewrites it with mode `0600` because it can hold `api.s2_api_key`; broader pre-existing permission bits are narrowed on every write.
- Mutations serialize on a file lock, fail after 10 seconds rather than overwrite, and refuse to clobber an unreadable file. Ordinary reads ignore invalid entries, malformed TOML, and non-UTF-8 with a warning, so a broken config never blocks other commands.
- Unknown keys survive a rewrite; comments do not, since the TOML round-trip is value-level.
- To reset, use `citemesh config unset` per key, or delete the path printed by `citemesh config path`.
