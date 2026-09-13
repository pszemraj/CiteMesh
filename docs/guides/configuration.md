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

`set` validates against the same whitelist the CLI uses and rejects a bad key or value with the valid options. Booleans accept `true`/`false` (also `1/0`, `yes/no`, `on/off`); `export` takes a comma-separated list (`json,dashboard`); thresholds are decimals in `[0.0, 1.0]`; integer counts must be at least `1`, except `max_semantic`, `max_citations`, and `max_references`, which also accept `0`.

## Precedence

An explicit CLI flag beats the environment, which beats `config.toml`, which beats the built-in default. `S2_API_KEY` is the only variable in play; its presence wins even when empty, and an empty value selects anonymous access.

Configured values behave like *your* built-in defaults rather than like flags you typed:

- They never trigger "unsupported option for strategy" errors: `defaults.device` does not break a `--strategy citation` run, it is just unused there.
- They outrank tuned implicit defaults such as hybrid's citation and reference budgets.
- They lose to a flag that implies a mode: `--dataset-source` or `--corpus-size` implies `--semantic-source arxiv-corpus` against a configured `candidates`, and `--candidate-pool-size` implies `candidates` symmetrically, leaving config-only settings for the other mode inert. `--all-corpus` likewise beats a configured `corpus_size`.

Applied config defaults are logged at DEBUG with the keys and the file path. Inert corpus-only defaults are reported at INFO and left out of the build sidecar, which records only settings applicable to the selected source and precision.

## Supported keys

Each `[defaults]` key sets the default for the `--flag` of the same name unless noted. `search_mode` applies to `citemesh search` rather than `build`, and local-mode search reads `model`, `model_profile`, `semantic_source`, `truncate_dim`, and the rest of this table to pick which embedding-cache namespace it queries (`device` is read too, but only to encode the query — it is not part of the namespace).

| Key | Notes |
| --- | --- |
| `strategy` | `recommendation`, `citation`, `embedding`, `hybrid` |
| `export` | format list |
| `theme` | `light`, `dark`, `solarized`, `auto` |
| `model` | checkpoint or local path |
| `model_profile` | `auto`, `default`, `embeddinggemma` |
| `model_revision` | branch, tag, or commit |
| `device` | `auto`, `cuda`, `mps`, `cpu` |
| `semantic_source` | `candidates`, `arxiv-corpus` |
| `candidate_pool_size` | |
| `encode_batch_size` | sets `--batch-size` / `-bs` |
| `storage_precision` | `int8`, `float32` |
| `binary_prefilter` | int8 corpus caches only |
| `calibration_sample_size` | int8 only; must match the build for local search to reuse its namespace |
| `max_papers`, `max_semantic`, `max_citations`, `max_references`, `top_k` | |
| `truncate_dim` | unset uses the model profile (`512` for EmbeddingGemma) |
| `min_semantic_similarity` | embedding/hybrid edge eligibility |
| `corpus_size` | unset hydrates the full selected split |
| `dataset_source`, `dataset_split` | HuggingFace arXiv metadata repository and split |
| `streaming`, `torch_compile` | booleans |
| `search_mode` | `auto`, `local`, `s2` for `citemesh search`; `auto` uses the local embedding cache when it has vectors, else Semantic Scholar keyword search |

`[api]` holds one key, `s2_api_key`, used only when `S2_API_KEY` is absent. `config list` masks it; `config get` prints it in full. It goes straight to the API client, never into subprocess environments.

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

- It lives at the cache root, so `CITEMESH_CACHE_DIR` moves it, and it survives `citemesh cache clear` along with its lock ([Caching & Data](caching.md)).
- CiteMesh rewrites it with mode `0600` because it can hold `api.s2_api_key`; broader pre-existing permission bits are narrowed on every write.
- Mutations serialize on a file lock, fail after 10 seconds rather than overwrite, and refuse to clobber an unreadable file. Ordinary reads ignore invalid entries, malformed TOML, and non-UTF-8 with a warning, so a broken config never blocks other commands.
- Unknown keys survive a rewrite; comments do not, since the TOML round-trip is value-level.
- To reset, use `citemesh config unset` per key, or delete the path printed by `citemesh config path`.
