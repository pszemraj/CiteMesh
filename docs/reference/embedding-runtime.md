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
- The `google/embeddinggemma-300m` fallback is a license-gated repository: it requires an accepted license on Hugging Face plus an `HF_TOKEN`, so without those the fallback fails rather than transparently retrying.
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
  `512`. This applies to embedding/hybrid builds, local search, and symmetric
  graph-similarity encoding on CUDA, MPS, and CPU, including recognized local
  checkpoints and the Google fallback. Explicit CLI or configuration values take
  precedence. Supported dimensions remain `768`, `512`, `256`, and `128`.
- The [September 2026 dimension study](defaults-tuning-study.md#embedding-dimensions-september-2026)
  motivates the default: better retention of full-model neighbors at similar GPU
  encoding cost, with larger vector storage and slower local searches.
- Every fallback load candidate resolves its own profile before loader kwargs, formatter fingerprints, and cache namespaces are computed.
- The profile requires Transformers 5.2 or newer, where `dtype="auto"` is the supported checkpoint-loading API and Gemma 3 honors the checkpoint's bidirectional-attention setting. CiteMesh checks this before constructing the model and does not treat an incompatible backend as a checkpoint-load failure eligible for fallback.

## Task-Specific Vector Spaces

CiteMesh keeps retrieval ranking and graph topology in separate prompt-conditioned spaces:

- A resolved paper seed and a free-text seed use the model's retrieval-query role.
- Candidate and corpus papers use the retrieval-document role for seed-to-paper ranking and local semantic search.
- Papers selected for the final graph are re-encoded with the symmetric sentence-similarity (STS) role; only those vectors are used for paper-to-paper edge scores.

For EmbeddingGemma, the symmetric formatter is exactly `task: sentence similarity | query: <title/abstract>`. Other profiles default to an identity formatter until they declare a task-specific symmetric contract. Encoder outputs remain normalized float32 before any retrieval-cache quantization.

Retrieval documents and graph-similarity vectors have independent cache namespaces, formatter fingerprints, and storage contracts. The graph namespace is always float32 with no binary prefilter. This prevents a dimension match from making asymmetric retrieval vectors eligible for symmetric graph scoring.

Before encoding, CiteMesh warns when text exceeds the loaded model's token window,
including prompts and special tokens. The encoder still truncates those inputs;
their embeddings represent only part of the text. This applies to queries, paper
documents, corpus hydration, calibration, and graph-similarity encoding. Cached
vectors reused without encoding do not repeat the warning.

The regression suite locks in prompt routing, cache separation, and fail-closed vector completeness. A small frozen real-EmbeddingGemma MPS smoke additionally checks retrieval Recall/nDCG and a related-versus-unrelated STS margin. Real-CUDA smokes load the designated model through the installed encoder stack, verify its live dtype/attention/autocast contract, and round-trip a tiny network-free corpus through INT8 hydration and retrieval. These real-model checks remain opt-in local validation (`pytest -m slow`, or `pytest -m "slow and cuda"` for CUDA only) on the relevant hardware.

## Device Selection

CiteMesh resolves an explicit compute device before loading any embedding model:

- `--device auto` (default) prefers `cuda`, then `mps` (Apple Silicon Metal), then `cpu`.
- An explicit `--device cuda` or `--device mps` on a host where that backend is unavailable fails fast with a CLI usage error; MPS diagnostics distinguish a torch build without MPS from a built backend that is unavailable on the machine. CiteMesh never silently downgrades an explicit accelerator request to CPU.
- The resolved device is passed directly to `SentenceTransformer(device=...)` and drives every precision, attention, TF32, and compile decision below.
- The effective device and compute dtype are recorded in the config sidecar's `metadata.embedding` block (`effective_device`, `effective_compute_dtype`). The graph JSON export does not include these runtime fields. The sidecar's `build.embedding.device` records the requested token.

## Precision and Compile Policy

Per-device precision matrix (EmbeddingGemma is the default profile):

| Device | EmbeddingGemma | Other profiles | Autocast |
| --- | --- | --- | --- |
| `cuda` | bfloat16 when native support is reported, otherwise float32 | float32 | bfloat16 only for profiles that opt in |
| `mps` | bfloat16 when torch >= 2.13 and the context is accepted, otherwise float32 | float32 | bfloat16 only for profiles that opt in |
| `cpu` | bfloat16 when native support is reported, otherwise float32 | float32 | bfloat16 only for profiles that opt in |

Notes:

- Embedding models load with Transformers automatic dtype resolution (`dtype="auto"`). This preserves the checkpoint/configured weight dtype instead of forcing fp32 or a reduced dtype.
- After loading, CiteMesh inspects live parameter and buffer dtypes before binding a cache namespace. Float16 tensors are rejected everywhere; bfloat16 tensors are rejected when the active device/profile did not select the verified bfloat16 path; any dtype other than float32/bfloat16 is rejected outright. This keeps automatic loading without allowing the runtime log or cache provenance to claim float32 for bfloat16 execution.
- Reduced-precision compute is bfloat16-only and is entered through `torch.autocast` around encode calls. If the device capability, torch API, version guard, or autocast-context probe rejects bfloat16, CiteMesh uses float32.
- CUDA capability checks request native support (`is_bf16_supported(including_emulation=False)`), so tensor-level emulation on older GPUs does not enable bfloat16 autocast.
- bf16-on-MPS requires torch >= 2.13 (the floor verified on Apple Silicon). Older torch releases fall back to float32.
- CPU BF16 selection checks native x86 or ARM instructions through `torch.cpu.get_capabilities()` where available, or the older x86 BF16 probe. Missing or unverified hardware support keeps CPU compute in float32. When using the precision wrapper, final embedding normalization after dimension truncation runs once in float32 outside autocast; SentenceTransformers' additional encode-time normalization is disabled. The checkpoint's own modules remain intact.
- Attention selection is model-profile-driven. EmbeddingGemma prefers `flash_attention_2` on CUDA when `flash_attn` is installed and BF16 compute is available. Missing FA2 or FP32 compute selects SDPA; an FA2 model-load failure retries the same checkpoint with SDPA. MPS uses SDPA and CPU leaves attention automatic. Profiles without an explicit preference leave the Transformers backend automatic.
- FP32 checkpoint weights are compatible with FA2 when encoding uses BF16 autocast:
  Transformers converts attention inputs to the active autocast dtype before
  calling FA2. For this verified CUDA path only, CiteMesh filters the upstream
  FP32-weight warning during model construction. Other load warnings and runtime
  failures remain visible; weights still load with `dtype="auto"`.
- CiteMesh does not select OpenVINO or ONNX backends on top of torch.
- On Ampere+ CUDA devices in eager mode, TF32 is scoped to the CUDA matmul and cuDNN convolution backends for each encode call, then the prior process settings are restored. TF32 configuration is skipped entirely for non-CUDA devices, including `--device cpu` on a CUDA host.

Compile policy:

- `torch.compile` is best-effort, profile-gated, and disabled by default.
- Compile can be enabled on `cuda`, `cpu`, and `mps`; Inductor-on-Metal (`mps`) remains experimental.
- Because compilation is lazy, a compiled-call failure restores the original eager inner model and retries the affected encode batch once. This applies to direct encoding, cached candidates, graph embeddings, and corpus updates; cache writes occur only after all batches finish successfully.
- Enable it explicitly with `--torch-compile` when you want to pay the warm-up cost. CUDA and CPU honor this flag during corpus hydration, including resumed hydration, and use dynamic shapes for variable-length batches. MPS continues to defer compilation during cold-cache hydration.
- CPU compilation enables GEMM autotuning and scopes Inductor freezing around each encode call. Freezing must be active while Dynamo captures parameters, including retraces; setting only a backend compile option is too late. SentenceTransformers already uses evaluation and inference mode, which satisfy freezing's disabled-gradient requirement. Previous compiler settings are restored after encoding, and eager parameters are retained for fallback. Autotuning may choose oneDNN or a C++ GEMM template depending on the CPU and shapes; it adds first-call compilation cost. See PyTorch's [CPU max-autotune tutorial](https://docs.pytorch.org/tutorials/unstable/max_autotune_on_CPU_tutorial.html).
- CiteMesh replaces SentenceTransformers 6's active `model[0].model` for compilation and eager recovery (with support for the legacy `auto_model` layout). Replacing only the legacy alias can leave the real forward pass uncompiled.
- During compiled FA2 encode calls, CiteMesh scopes a compiler setting that ignores Transformers' autocast-conversion log call while tracing (or defers it on older torch releases). This prevents a logging-induced graph break inside the decoder loop; the prior compiler settings are restored after encoding.
- Legacy note: on torch `2.9`/`2.10` with compile enabled, the runtime scopes `torch.set_float32_matmul_precision("high")` to encode calls instead of mixing it with the `fp32_precision` API. Later torch releases scope the CUDA-specific modern controls the same way.

## Cache Storage and Portability

Namespace identity, cross-device reuse, physical storage precision, calibration,
and the optional binary prefilter are described in
[Caching & Data](../guides/caching.md).

## Dependency Floor

- The `embeddings` install extra provides `torch>=2.9.0` on non-macOS platforms, `torch>=2.13.0` on macOS, `transformers>=5.2.0`, `sentence-transformers>=5.7.0`, `datasets>=2.14.0`, and `huggingface_hub>=0.24.0`. Candidate mode uses the encoder stack without loading Datasets; `arxiv-corpus` mode also uses Datasets for hydration. The macOS torch floor matches the release verified for MPS bf16 execution, while the encoder-stack floors provide the supported automatic-dtype API and EmbeddingGemma's bidirectional attention. The automatic-dtype contract is carried by `transformers>=5.2.0`, which the model loader passes through as `model_kwargs={"dtype": "auto"}`; Sentence Transformers only forwards it, so the floor there is the oldest release verified against this stack rather than a required API.

## Implementation References

- Model defaults, aliases, and fallback chain: [citemesh/data/model_profiles.py](../../citemesh/data/model_profiles.py)
- Embedding model load and fallback execution: [citemesh/strategies/embedding.py](../../citemesh/strategies/embedding.py)
- TF32 and compile guard behavior: [citemesh/strategies/embedding.py](../../citemesh/strategies/embedding.py)
- CLI default model wiring: [citemesh/cli.py](../../citemesh/cli.py)
