# Embedding Runtime Policy

Runtime behavior that affects embedding model selection, fallback, precision, and compilation.

Related docs:

- CLI flags: [CLI Usage](../guides/cli.md)
- Cache layout and hydration: [Caching & Data](../guides/caching.md)
- Output metadata fields in exports: [Output Artifacts](output-artifacts.md)
- Defaults parameter study: [Defaults Tuning Study](defaults-tuning-study.md)

## Default Model Selection

- Default embedding model: `unsloth/embeddinggemma-300m`
- Default fallback chain: `unsloth/embeddinggemma-300m` -> `google/embeddinggemma-300m`

Fallback behavior:

- The fallback chain is used only for default-revision loads (no explicit `--model-revision`).
- If `--model-revision` is set, fallback retries are disabled to preserve deterministic revision pinning.
- If all candidates fail, CiteMesh raises a runtime error with per-candidate failure summaries.
- When fallback succeeds, cache fingerprint checks follow the active loaded checkpoint identity (not just the originally requested model token).

## EmbeddingGemma Profile Mapping

- `unsloth/embeddinggemma-*` and `google/embeddinggemma-*` map to the same EmbeddingGemma runtime profile.
- This means both receive the same prompt formatting, truncate-dim policy, and compile eligibility behavior.

## Device Selection

CiteMesh resolves an explicit compute device before loading any embedding model:

- `--device auto` (default) prefers `cuda`, then `mps` (Apple Silicon Metal), then `cpu`.
- An explicit `--device cuda` or `--device mps` on a host where that backend is unavailable fails fast with a parser error — CiteMesh never silently downgrades an explicit accelerator request to CPU.
- The resolved device is passed directly to `SentenceTransformer(device=...)` and drives every precision, attention, TF32, and compile decision below.
- The effective device and compute dtype are recorded in the run's export metadata (`effective_device`, `effective_compute_dtype`) and config sidecar.

## Precision and Compile Policy

Per-device precision matrix (EmbeddingGemma is the default profile):

| Device | EmbeddingGemma | Other profiles | Autocast |
| --- | --- | --- | --- |
| `cuda` | bfloat16 when `is_bf16_supported()`, otherwise float32 | float32 | bfloat16 only for profiles that opt in |
| `mps` | bfloat16 when torch >= 2.13 and the context is accepted, otherwise float32 | float32 | bfloat16 only for profiles that opt in |
| `cpu` | float32 | float32 | off |

Notes:

- Embedding models load with Transformers automatic dtype resolution (`dtype="auto"` on current Transformers, with the legacy `torch_dtype="auto"` spelling on older supported releases). This preserves the checkpoint/configured weight dtype instead of forcing fp32 or a reduced dtype.
- Reduced-precision compute is bfloat16-only and is entered through `torch.autocast` around encode calls. If the device capability, torch API, version guard, or autocast-context probe rejects bfloat16, CiteMesh uses float32.
- bf16-on-MPS requires torch >= 2.13 (the floor verified on Apple Silicon). Older torch releases fall back to float32.
- CPU stays float32: reduced precision on CPU is slower, not faster.
- Attention implementation: `sdpa` on CUDA and MPS (`flash_attention_2` only via model-profile opt-in plus an importable `flash_attn`, CUDA-only); CPU leaves the transformers default. `flash_attn` is never probed off-CUDA.
- CiteMesh does not select OpenVINO or ONNX backends on top of torch.
- On Ampere+ CUDA devices in eager mode, TF32 is enabled with the new API: `torch.backends.fp32_precision = "tf32"`. TF32 configuration is skipped entirely for non-CUDA devices, including `--device cpu` on a CUDA host.

Compile policy:

- `torch.compile` is best-effort, profile-gated, and disabled by default.
- Compile is only attempted on `cuda` and `mps`. On CPU it is declined: Inductor warm-up for a short-lived CLI run has no payoff.
- Inductor-on-Metal (`mps`) is experimental; failures fall back to eager.
- Enable it explicitly when you want to pay the warm-up cost for a warm-cache or longer-lived run.
- On cold-cache runs that must hydrate embeddings, compile is deferred for that run to avoid Inductor compile/recompile overhead during long corpus hydration.
- Legacy note: on torch `2.9`/`2.10` with compile enabled, the runtime uses `torch.set_float32_matmul_precision("high")` instead of the `fp32_precision` API to avoid a release-branch Inductor mixed-API conflict. Later torch releases use the modern API directly.

## Cache Portability Across Devices

The embedding cache namespace tracks the runtime-active model artifact, requested revision, representation/formatter contract, dimensions, storage settings, and *compute dtype*, but not the device. A cache hydrated with bf16 on a CUDA box and one hydrated with bf16 on MPS share a byte-identical namespace when every semantic input matches: you can warm the cache on a GPU host, copy the cache directory to a Mac, and get full cache hits. CPU (float32) caches live in a separate, deliberately conservative namespace.

## Int8 Retrieval Pipeline

When `--storage-precision int8` is active, retrieval uses a two-stage path:

- Stage 1 (`binary_prefilter`): approximate shortlist with Hamming distance over bit-packed binary sign sketches.
- Stage 2 (`binary_rescore_multiplier`): exact dot-product rescoring on int8/dequantized vectors for the shortlist.

Important distinction:

- This is not end-to-end "binary embeddings" storage/retrieval in the SBERT sense.
- Primary cache vectors remain `int8` (or `float32` by config), and final ranking is computed from those vectors.
- The binary representation is only a prefilter index for candidate pruning before exact rescoring.

Interpretation:

- `binary_prefilter_enabled=true` means the namespace is configured to use Stage 1.
- `binary_prefilter_used_for_query=true` means Stage 1 was actually used for this query (not bypassed due compatibility fallback).
- `binary_rescore_multiplier=8` means CiteMesh rescored `top_k * 8` shortlisted candidates exactly before taking final top-k.

## Dependency Floor

- Embedding workflows require `torch>=2.9.0` on Linux/Windows and `torch>=2.13.0` on macOS (plus `sentence-transformers` and `datasets`). The macOS floor matches the torch release verified for MPS bf16 execution.

## Implementation References

- Model defaults, aliases, and fallback chain: [citemesh/data/model_profiles.py](../../citemesh/data/model_profiles.py)
- Embedding model load and fallback execution: [citemesh/strategies/embedding.py](../../citemesh/strategies/embedding.py)
- TF32 and compile guard behavior: [citemesh/strategies/embedding.py](../../citemesh/strategies/embedding.py)
- CLI default model wiring: [citemesh/cli.py](../../citemesh/cli.py)
