# Embedding Runtime Policy

Embedding behavior that affects model selection, fallback, precision, and compilation.

Related docs:

- CLI flags: [CLI Usage](https://github.com/pszemraj/CiteMesh/blob/main/docs/guides/cli.md)
- Cache layout and hydration: [Caching & Data](https://github.com/pszemraj/CiteMesh/blob/main/docs/guides/caching.md)
- Defaults parameter study: [Defaults Tuning Study](https://github.com/pszemraj/CiteMesh/blob/main/docs/reference/defaults-tuning-study.md)
- Docs index: [Documentation](https://github.com/pszemraj/CiteMesh/blob/main/docs/README.md)

## Default Model Selection

- Default embedding model: `unsloth/embeddinggemma-300m`
- Default fallback chain: `unsloth/embeddinggemma-300m` -> `google/embeddinggemma-300m`

Fallback behavior:

- The fallback chain is used only for default-revision loads (no explicit `--model-revision`).
- If `--model-revision` is set, fallback retries are disabled to preserve deterministic revision pinning.
- If all candidates fail, CiteMesh raises a runtime error with per-candidate failure summaries.

## EmbeddingGemma Profile Mapping

- `unsloth/embeddinggemma-*` and `google/embeddinggemma-*` map to the same EmbeddingGemma runtime profile.
- This means both receive the same prompt formatting, truncate-dim policy, and compile eligibility behavior.

## Precision and Compile Policy

Runtime precision policy:

- EmbeddingGemma prefers `torch_dtype=bfloat16` with CUDA autocast when supported.
- If CUDA or CUDA bfloat16 is unavailable, CiteMesh falls back to float32.
- On Ampere+ CUDA devices in eager mode, TF32 is enabled with the new API: `torch.backends.fp32_precision = "tf32"`.
- On Ampere+ CUDA devices with `torch.compile` enabled on torch `2.9`/`2.10`, CiteMesh uses `torch.set_float32_matmul_precision("high")` and does not touch `torch.backends.*.fp32_precision` to avoid the release-branch Inductor mixed-API conflict.

Compile policy:

- `torch.compile` is best-effort and profile-gated.
- `torch.compile` remains enabled on torch `2.9`/`2.10` CUDA when available; the runtime switches TF32 control to the compile-safe matmul precision bridge above.
- On cold-cache runs that must hydrate embeddings, compile is deferred for that run to avoid Inductor compile/recompile overhead during long corpus hydration.
- On warm-cache runs (matching hydrated cache already present), compile is attempted normally.

## Dependency Floor

- Embedding workflows require `torch>=2.9.0` (plus `sentence-transformers` and `datasets`).

## Equivalence Note for Default/Fallback Checkpoints

Verification timestamp: February 15, 2026.

- `google/embeddinggemma-300m` and `unsloth/embeddinggemma-300m` had matching Hugging Face file metadata across all 19 files.
- At verification time, both had identical `config.json` Git OID (`aca32ef8dacddfbe15dc12c0427a9153d5c2b0f4`) and identical `model.safetensors` LFS OID (`cbf5a78393b6a033e0b8a63a57549964f7ed5c6fbeb4ba0694214f36123f2fd2`).

This is an observational equivalence check, not a permanent guarantee. If either upstream repo changes, rerun equivalence verification before assuming interchangeability.

## Implementation References

- Model defaults, aliases, and fallback chain: [citemesh/data/model_profiles.py](https://github.com/pszemraj/CiteMesh/blob/main/citemesh/data/model_profiles.py)
- Embedding model load and fallback execution: [citemesh/strategies/embedding.py](https://github.com/pszemraj/CiteMesh/blob/main/citemesh/strategies/embedding.py)
- TF32 and compile guard behavior: [citemesh/strategies/embedding.py](https://github.com/pszemraj/CiteMesh/blob/main/citemesh/strategies/embedding.py)
- CLI default model wiring: [citemesh/cli.py](https://github.com/pszemraj/CiteMesh/blob/main/citemesh/cli.py)
