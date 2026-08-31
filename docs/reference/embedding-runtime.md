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
- Local checkpoints are recognized from Gemma 3 architecture plus either the model's `use_bidirectional_attention = true` contract or the SentenceTransformers retrieval/STS task prompts, including plain Transformers exports, direct Hugging Face snapshot paths, and nested Transformer modules.
- A local Gemma 3 artifact with neither form of EmbeddingGemma evidence stays on the default profile and emits a warning instead of silently claiming the EmbeddingGemma contract.
- `--model-profile embeddinggemma` binds the same contract for stripped/custom fine-tune exports; `--model-profile default` explicitly disables family-specific behavior.
- This means both receive the same prompt formatting, truncate-dim policy, and compile eligibility behavior.
- When `--truncate-dim` is omitted, this profile uses its recommended dimension of
  `256`.
- Every fallback load candidate resolves its own profile before loader kwargs, formatter fingerprints, and cache namespaces are computed.
- The profile requires Transformers 4.57 or newer, where Gemma 3 honors the checkpoint's bidirectional-attention setting. CiteMesh checks this before constructing the model and does not treat an incompatible backend as a checkpoint-load failure eligible for fallback.

## Task-Specific Vector Spaces

CiteMesh keeps retrieval ranking and graph topology in separate prompt-conditioned spaces:

- A resolved paper seed and a free-text seed use the model's retrieval-query role.
- Candidate and corpus papers use the retrieval-document role for seed-to-paper ranking and local semantic search.
- Papers selected for the final graph are re-encoded with the symmetric sentence-similarity (STS) role; only those vectors are used for paper-to-paper edge scores.

For EmbeddingGemma, the symmetric formatter is exactly `task: sentence similarity | query: <title/abstract>`. Other profiles default to an identity formatter until they declare a task-specific symmetric contract. Encoder outputs remain normalized float32 before any retrieval-cache quantization.

Retrieval documents and graph-similarity vectors have independent cache namespaces, formatter fingerprints, and storage contracts. The graph namespace is always float32 with no binary prefilter. This prevents a dimension match from making asymmetric retrieval vectors eligible for symmetric graph scoring.

The regression suite locks in prompt routing, cache separation, and fail-closed vector completeness. A small frozen real-EmbeddingGemma MPS smoke additionally checks retrieval Recall/nDCG and a related-versus-unrelated STS margin. Per the project's solo/pre-user CI policy, that real-model check remains opt-in local/release validation (`pytest -m slow`), not another CI job.

## Device Selection

CiteMesh resolves an explicit compute device before loading any embedding model:

- `--device auto` (default) prefers `cuda`, then `mps` (Apple Silicon Metal), then `cpu`.
- An explicit `--device cuda` or `--device mps` on a host where that backend is unavailable fails fast with a CLI usage error; MPS diagnostics distinguish a torch build without MPS from a built backend that is unavailable on the machine. CiteMesh never silently downgrades an explicit accelerator request to CPU.
- The resolved device is passed directly to `SentenceTransformer(device=...)` and drives every precision, attention, TF32, and compile decision below.
- The effective device and compute dtype are recorded in the run's export metadata (`effective_device`, `effective_compute_dtype`) and config sidecar.

## Precision and Compile Policy

Per-device precision matrix (EmbeddingGemma is the default profile):

| Device | EmbeddingGemma | Other profiles | Autocast |
| --- | --- | --- | --- |
| `cuda` | bfloat16 when native support is reported, otherwise float32 | float32 | bfloat16 only for profiles that opt in |
| `mps` | bfloat16 when torch >= 2.13 and the context is accepted, otherwise float32 | float32 | bfloat16 only for profiles that opt in |
| `cpu` | float32 | float32 | off |

Notes:

- Embedding models load with Transformers automatic dtype resolution (`dtype="auto"` on current Transformers, with the legacy `torch_dtype="auto"` spelling on older supported releases). This preserves the checkpoint/configured weight dtype instead of forcing fp32 or a reduced dtype.
- After loading, CiteMesh inspects the live parameter dtypes before binding a cache namespace. Float16 weights are rejected everywhere; bfloat16 weights are rejected when the active device/profile did not select the verified bfloat16 path. This keeps automatic loading without allowing the runtime log or cache provenance to claim float32 for bfloat16 execution.
- Reduced-precision compute is bfloat16-only and is entered through `torch.autocast` around encode calls. If the device capability, torch API, version guard, or autocast-context probe rejects bfloat16, CiteMesh uses float32.
- CUDA capability checks request native support (`is_bf16_supported(including_emulation=False)`), so tensor-level emulation on older GPUs does not enable bfloat16 autocast.
- bf16-on-MPS requires torch >= 2.13 (the floor verified on Apple Silicon). Older torch releases fall back to float32.
- CPU stays float32: reduced precision on CPU is slower, not faster. An automatically loaded bfloat16 checkpoint is rejected on this path rather than silently sharing a float32 cache namespace.
- Attention selection is model-profile-driven. EmbeddingGemma requests `sdpa` on CUDA and MPS; profiles without an explicit preference leave the Transformers backend automatic. `flash_attention_2` remains CUDA-only and requires both a profile opt-in and importable `flash_attn`.
- CiteMesh does not select OpenVINO or ONNX backends on top of torch.
- On Ampere+ CUDA devices in eager mode, TF32 is scoped to the CUDA matmul and cuDNN convolution backends for each encode call, then the prior process settings are restored. TF32 configuration is skipped entirely for non-CUDA devices, including `--device cpu` on a CUDA host.

Compile policy:

- `torch.compile` is best-effort, profile-gated, and disabled by default.
- Compile is only attempted on `cuda` and `mps`. On CPU it is declined: Inductor warm-up for a short-lived CLI run has no payoff.
- Inductor-on-Metal (`mps`) is experimental. Because compilation is lazy, a first-encode failure restores the original eager inner model and retries the complete encode request once; no partial bucket result is persisted.
- Enable it explicitly when you want to pay the warm-up cost for a warm-cache or longer-lived run.
- On cold-cache runs that must hydrate embeddings, compile is deferred for that run to avoid Inductor compile/recompile overhead during long corpus hydration.
- Legacy note: on torch `2.9`/`2.10` with compile enabled, the runtime scopes `torch.set_float32_matmul_precision("high")` to encode calls instead of mixing it with the `fp32_precision` API. Later torch releases scope the CUDA-specific modern controls the same way.

## Cache Storage and Portability

Namespace identity, cross-device reuse, physical storage precision, calibration,
and the optional binary prefilter are described in
[Caching & Data](../guides/caching.md).

## Dependency Floor

- The `embeddings` install extra provides `torch>=2.9.0` on Linux/Windows, `torch>=2.13.0` on macOS, `transformers>=4.57.0`, SentenceTransformers, and Datasets. Candidate mode uses the encoder stack without loading Datasets; `arxiv-corpus` mode also uses Datasets for hydration. The macOS torch floor matches the release verified for MPS bf16 execution, while the Transformers floor is required for EmbeddingGemma's bidirectional attention.

## Implementation References

- Model defaults, aliases, and fallback chain: [citemesh/data/model_profiles.py](../../citemesh/data/model_profiles.py)
- Embedding model load and fallback execution: [citemesh/strategies/embedding.py](../../citemesh/strategies/embedding.py)
- TF32 and compile guard behavior: [citemesh/strategies/embedding.py](../../citemesh/strategies/embedding.py)
- CLI default model wiring: [citemesh/cli.py](../../citemesh/cli.py)
