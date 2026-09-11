# User Configuration

CiteMesh stores durable personal defaults in a TOML file at the cache root:

```text
<cache_root>/config.toml
```

Use it for preferences you would otherwise repeat on every invocation - for example always using corpus-backed semantic sourcing, a preferred theme, or a Semantic Scholar API key.

## Precedence

Effective values resolve in this order (first match wins):

1. Explicit CLI flag (`--semantic-source arxiv-corpus`)
2. Environment variable (only `S2_API_KEY` today; presence wins even when empty, and an empty value explicitly selects anonymous access)
3. `config.toml` value
4. Built-in default

Config values behave like personal built-in defaults, not like explicit flags:

- They never trigger "unsupported option for strategy" errors. Setting `defaults.device` does not break `--strategy citation` runs; the value is simply unused there.
- They outrank tuned implicit defaults (for example the hybrid strategy's implicit citation/reference budgets).
- Explicit corpus-only CLI flags (for example `--dataset-source` or `--corpus-size`) still imply `--semantic-source arxiv-corpus`, overriding a configured `defaults.semantic_source = "candidates"` for that run.
- `--all-corpus` uses the full selected split even when `defaults.corpus_size` sets a personal cap.
- Explicit candidate-only CLI flags work symmetrically: `--candidate-pool-size` implies `--semantic-source candidates`, overriding a configured `defaults.semantic_source = "arxiv-corpus"`. Corpus-only settings that came only from config (for example a dataset source plus streaming) stay inert in that candidate run.

When config defaults are applied to a build, CiteMesh logs one DEBUG line listing the applied keys and the config file path.

## Commands

```bash
citemesh config list                                   # show configured values + file path
citemesh config get defaults.semantic_source           # print one value (script-friendly)
citemesh config set defaults.semantic_source arxiv-corpus
citemesh config unset defaults.semantic_source
citemesh config path                                   # print the config file path
```

Value forms for `config set`:

- Booleans: `true` / `false` (also `1/0`, `yes/no`, `on/off`)
- Export lists: comma-separated, e.g. `citemesh config set defaults.export json,dashboard`
- Similarity thresholds: decimal numbers between `0.0` and `1.0`
- Everything else: plain strings/integers (integer counts must be at least `1`, except `max_semantic`, `max_citations`, and `max_references`, which also accept `0`)

Invalid keys and values are rejected at `set` time with the list of valid options. Invalid entries, malformed TOML, and non-UTF-8 files are ignored with a warning during ordinary CLI loads, so a bad config never blocks unrelated commands. Mutating `config set`/`unset` operations fail instead of overwriting an unreadable file, and they serialize on a file lock - a mutation that cannot acquire the lock within 10 seconds fails with an error naming the config path. Unknown keys already in the file are preserved when CiteMesh rewrites it (comments are not - the TOML round-trip is value-level).

## Supported keys

`[defaults]` - whitelisted build-flag defaults. `search_mode` applies to `citemesh search` instead of `build`, and local-mode search also reads the rest of this table to select which embedding-cache namespace it queries (`model`, `model_profile`, `device`, `semantic_source`, `truncate_dim`, and related keys):

| Key | Meaning |
| --- | --- |
| `strategy` | Default `--strategy` (`recommendation`, `citation`, `embedding`, `hybrid`) |
| `export` | Default `--export` format list |
| `theme` | Default `--theme` (`light`, `dark`, `solarized`, `auto`) |
| `model` | Default `--model` checkpoint |
| `model_profile` | Default `--model-profile` (`auto`, `default`, `embeddinggemma`) |
| `model_revision` | Default `--model-revision` |
| `device` | Default `--device` (`auto`, `cuda`, `mps`, `cpu`) |
| `semantic_source` | Default `--semantic-source` (`candidates`, `arxiv-corpus`) |
| `candidate_pool_size` | Default `--candidate-pool-size` |
| `encode_batch_size` | Default `--batch-size` / `-bs` |
| `storage_precision` | Default `--storage-precision` (`int8`, `float32`) |
| `binary_prefilter` | Default `--binary-prefilter` / `--no-binary-prefilter` toggle for int8 corpus caches |
| `calibration_sample_size` | Default `--calibration-sample-size` for int8 corpus caches; ignored when effective storage is `float32` (including candidate mode). Must match the int8 corpus build to reuse its namespace in local search. |
| `max_papers` | Default `--max-papers` |
| `max_semantic` | Default `--max-semantic` |
| `max_citations` | Default `--max-citations` |
| `max_references` | Default `--max-references` |
| `top_k` | Default `--top-k` |
| `truncate_dim` | Default `--truncate-dim`; unset uses the model profile (`512` for EmbeddingGemma) |
| `min_semantic_similarity` | Default `--min-semantic-similarity` for embedding/hybrid edge eligibility; CLI flags override this value |
| `corpus_size` | Optional default `--corpus-size` cap; when unset, arXiv corpus mode hydrates the full selected split |
| `dataset_source` | Default `--dataset-source` (the HuggingFace repository for ArXiv metadata records) |
| `dataset_split` | Default `--dataset-split` |
| `streaming` | Default `--streaming` / `--no-streaming` toggle |
| `torch_compile` | Default `--torch-compile` toggle |
| `search_mode` | Default `citemesh search` mode (`auto`, `local`, `s2`). `auto` searches your local embedding cache when it has vectors and falls back to Semantic Scholar keyword search otherwise. |

`[api]`:

| Key | Meaning |
| --- | --- |
| `s2_api_key` | Semantic Scholar API key used only when `S2_API_KEY` is absent. Masked in `config list` output; `config get` prints the full value. |

The configured key is passed directly to the API client; CiteMesh does not add it to subprocess environments. Keys explicitly supplied through `S2_API_KEY` retain normal environment inheritance.

Corpus-only defaults ignored in candidate mode are reported at INFO level and omitted from the build sidecar's embedding settings. The sidecar records settings applicable to the selected source and storage precision.

Example `config.toml`:

```toml
[defaults]
search_mode = "local"
semantic_source = "arxiv-corpus"
theme = "dark"
export = ["json", "dashboard"]

[api]
s2_api_key = "your-key-here"
```

## Location and lifecycle

- The file lives at the cache root, so `CITEMESH_CACHE_DIR` moves it too.
- CiteMesh rewrites the file with mode `0600` (owner read/write only) because it can hold `api.s2_api_key`; broader pre-existing permission bits are narrowed on every write.
- `citemesh cache clear` deletes cached payloads but **never** `config.toml`. It preserves the configuration lock and cache-operation coordination directory, acquires the exclusive cache lock before deletion, then acquires the configuration lock so clearing cannot interrupt a pending configuration write or let concurrent writers bypass the lock.
- To reset configuration, delete the path printed by `citemesh config path`, or use `citemesh config unset` for individual keys.
