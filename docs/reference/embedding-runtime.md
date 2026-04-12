# Embedding Runtime Policy

Runtime behavior that affects embedding model selection, fallback, precision, and compilation.

Related docs:

- CLI flags: [CLI Usage](../guides/cli.md)
- Cache layout and hydration: [Caching & Data](../guides/caching.md)
- Output metadata fields in exports: [Output Artifacts](output-artifacts.md)
- Defaults parameter study: [Defaults Tuning Study](defaults-tuning-study.md)
- Docs index: [Documentation](../README.md)

## Default Model Selection

- Default embedding model: `unsloth/embeddinggemma-300m`
- Default fallback chain: `unsloth/embeddinggemma-300m` -> `google/embeddinggemma-300m`

Fallback behavior:

- The fallback chain is used only for default-revision loads (no explicit `--model-revision`).
- If `--model-revision` is set, fallback retries are disabled to preserve deterministic revision pinning.
- If all candidates fail, CiteMesh raises a runtime error with per-candidate failure summaries.
- When fallback succeeds, cache fingerprint checks follow the active loaded checkpoint
  identity (not just the originally requested model token).

## EmbeddingGemma Profile Mapping

- `unsloth/embeddinggemma-*` and `google/embeddinggemma-*` map to the same EmbeddingGemma runtime profile.
- This means both receive the same prompt formatting, truncate-dim policy, and compile eligibility behavior.

## Precision and Compile Policy

Runtime precision policy:

- EmbeddingGemma prefers `dtype=bfloat16` with CUDA autocast when supported.
- Other torch-backed profiles use `dtype=float16` on CUDA when the profile marks float16 as safe.
- If CUDA or the preferred reduced-precision path is unavailable, CiteMesh falls back to float32.
- When torch runs on CUDA, CiteMesh now sets an explicit attention implementation:
  - `flash_attention_2` when `flash_attn` is installed
  - otherwise `sdpa`
- On CPU-only hosts, CiteMesh prefers `openvino` when both `openvino` and `optimum.intel` are available, then `onnx` when `onnxruntime` is available, and otherwise falls back to `torch`.
- On Ampere+ CUDA devices in eager mode, TF32 is enabled with the new API: `torch.backends.fp32_precision = "tf32"`.
- On Ampere+ CUDA devices with `torch.compile` enabled on torch `2.9`/`2.10`, CiteMesh uses `torch.set_float32_matmul_precision("high")` and does not touch `torch.backends.*.fp32_precision` to avoid the release-branch Inductor mixed-API conflict.

Compile policy:

- `torch.compile` is best-effort, profile-gated, and disabled by default.
- Enable it explicitly when you want to pay the warm-up cost for a warm-cache or longer-lived run.
- On cold-cache runs that must hydrate embeddings, compile is deferred for that run to avoid Inductor compile/recompile overhead during long corpus hydration.
- When compile is enabled on torch `2.9`/`2.10` CUDA, the runtime switches TF32 control to the compile-safe matmul precision bridge above.

## Int8 Retrieval Pipeline

When `--storage-precision int8` is active, retrieval uses a two-stage path:

- Stage 1 (`binary_prefilter`): approximate shortlist with Hamming distance over bit-packed binary sign sketches.
- Stage 2 (`binary_rescore_multiplier`): exact dot-product rescoring on int8/dequantized vectors for the shortlist.

Important distinction:

- This is not end-to-end "binary embeddings" storage/retrieval in the SBERT sense.
- Primary cache vectors remain `int8` (or `float16`/`float32` by config), and final ranking is computed from those vectors.
- The binary representation is only a prefilter index for candidate pruning before exact rescoring.

Interpretation:

- `binary_prefilter_enabled=true` means the namespace is configured to use Stage 1.
- `binary_prefilter_used_for_query=true` means Stage 1 was actually used for this query (not bypassed due compatibility fallback).
- `binary_rescore_multiplier=8` means CiteMesh rescored `top_k * 8` shortlisted candidates exactly before taking final top-k.

## Dependency Floor

- Embedding workflows require `torch>=2.9.0` (plus `sentence-transformers` and `datasets`).

## Implementation References

- Model defaults, aliases, and fallback chain: [citemesh/data/model_profiles.py](../../citemesh/data/model_profiles.py)
- Embedding model load and fallback execution: [citemesh/strategies/embedding.py](../../citemesh/strategies/embedding.py)
- TF32 and compile guard behavior: [citemesh/strategies/embedding.py](../../citemesh/strategies/embedding.py)
- CLI default model wiring: [citemesh/cli.py](../../citemesh/cli.py)
