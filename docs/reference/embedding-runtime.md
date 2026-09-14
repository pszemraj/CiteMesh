# Embedding Runtime Policy

How CiteMesh resolves an embedding checkpoint, device, precision, attention backend, and compile mode at run time.

## Model selection and fallback

Default model `unsloth/embeddinggemma-300m`, falling back to `google/embeddinggemma-300m`.

- The chain applies only to default-revision loads; `--model-revision` disables it, preserving deterministic pinning.
- `google/embeddinggemma-300m` is license-gated - an accepted Hugging Face license plus `HF_TOKEN` - so without those the fallback fails rather than retrying silently. If every candidate fails, CiteMesh raises with per-candidate summaries.
- Each candidate resolves its own profile before loader kwargs are built, and a successful fallback binds fingerprints and cache namespaces to the loaded checkpoint identity, not the requested token.

## EmbeddingGemma profile

`unsloth/embeddinggemma-*` and `google/embeddinggemma-*` map to one profile, so both get the same prompt formatting, truncate-dim policy, and compile eligibility.

- Local checkpoints are recognized from Gemma 3 architecture plus either `use_bidirectional_attention = true` or the SentenceTransformers retrieval/STS task prompts; a Gemma 3 artifact with neither stays on the default profile and warns.
- `--model-profile embeddinggemma` binds the contract for stripped or custom exports; `--model-profile default` disables family-specific behavior.
- Without `--truncate-dim` the profile uses `512` for builds, local search, and symmetric graph encoding. Supported dimensions are `768`, `512`, `256`, `128`; explicit CLI or config values win. The [September 2026 dimension study](defaults-tuning-study.md#embedding-dimensions-september-2026) motivates that default.
- The profile requires Transformers 5.2+, where `dtype="auto"` is the supported loading API and Gemma 3 honors the checkpoint's bidirectional-attention setting. CiteMesh checks this before constructing the model, and an incompatible backend is not a fallback-eligible load failure.

## Task-specific vector spaces

Retrieval ranking and graph topology live in separate prompt-conditioned spaces:

- paper and free-text seeds use the retrieval-query role
- candidate and corpus papers use the retrieval-document role, for seed-to-paper ranking and local search
- papers selected for the final graph are re-encoded with the symmetric sentence-similarity role, and only those vectors score paper-to-paper edges

For EmbeddingGemma the symmetric formatter is exactly `task: sentence similarity | query: <title/abstract>`; other profiles use an identity formatter until they declare their own. Encoder outputs are normalized float32 before any retrieval-cache quantization.

The roles keep independent cache namespaces, fingerprints, and storage contracts, and the graph namespace is always float32 with no binary prefilter, so a matching dimension cannot make asymmetric vectors eligible for symmetric scoring. Namespace identity, cross-device reuse, storage precision, and calibration are in [Caching & Data](../guides/caching.md).

Inputs exceeding the token window, including prompts and special tokens, are truncated and counted in a warning. The ordinary path reports before encoding; CUDA prefetch reports the completed batch counts after encoding and resets partial counts on a compile retry. Reused cached vectors do not repeat the warning.

## Device selection

CiteMesh resolves an explicit compute device before loading any model - it never silently downgrades an accelerator request to CPU - and that device drives every decision below.

- `--device auto` (default) prefers `cuda`, then `mps`, then `cpu`.
- An explicit `--device cuda` or `--device mps` on a host without that backend fails fast with a CLI usage error; MPS diagnostics distinguish a torch build without MPS from a built backend unavailable on this machine.
- The effective device and compute dtype are recorded in the run sidecar, never in the graph JSON ([Output Artifacts](output-artifacts.md)).

## Precision

EmbeddingGemma compute dtype: bfloat16 on `cuda` and `cpu` when native support is reported, bfloat16 on `mps` when torch >= 2.13, and on every device only once a live `torch.autocast` context is entered without raising; float32 otherwise. Profiles that do not opt into bf16 autocast run float32 everywhere.

- Weights load with `dtype="auto"`. Floating-point parameters and buffers must be float32, or bfloat16 on a verified BF16 runtime. Float16 and other floating dtypes are rejected; inability to inspect them is also an error. CiteMesh does not silently recast an incompatible checkpoint.
- Reduced precision is bfloat16 only, through `torch.autocast` around encode calls; a rejected capability, API, version, or context probe falls back to float32. CUDA asks for native support (`is_bf16_supported(including_emulation=False)`), so emulation does not qualify; MPS requires torch >= 2.13; CPU probes native x86/ARM instructions. Final normalization after dimension truncation then runs once in float32 outside autocast, with SentenceTransformers' own encode-time normalization disabled.
- Attention is profile-driven: EmbeddingGemma prefers `flash_attention_2` on CUDA when `flash_attn` is installed and BF16 is available, while missing FA2, FP32 compute, or an FA2 load failure selects SDPA. MPS uses SDPA, CPU leaves attention automatic. FP32 weights stay FA2-compatible under BF16 autocast, so CiteMesh filters the upstream FP32-weight warning on that verified path only.
- On Ampere+ CUDA in eager mode, TF32 is scoped to the CUDA matmul and cuDNN convolution backends per encode call, then restored; non-CUDA devices skip it entirely, including `--device cpu` on a CUDA host. CiteMesh selects no OpenVINO or ONNX backend.

## Compile policy

`torch.compile` is best-effort, profile-gated, and disabled by default; `--torch-compile` enables it at a warm-up cost.

- Eligible on `cuda`, `cpu`, and `mps`, where Inductor-on-Metal remains experimental.
- Compilation is lazy, so a compiled-call failure restores the eager inner model and retries that encode batch once, for every encode path. Cache writes happen only after all batches finish.
- CUDA and CPU honor the flag during corpus hydration, resumed hydration included, with dynamic shapes for variable-length batches; MPS defers compilation during cold-cache hydration.
- CPU compilation enables GEMM autotuning (first-call cost: PyTorch's [max-autotune tutorial](https://docs.pytorch.org/tutorials/unstable/max_autotune_on_CPU_tutorial.html)) and scopes Inductor freezing around each encode call, restoring prior settings and keeping eager parameters for fallback.
- It compiles SentenceTransformers 6's active `model[0].model`, with legacy `auto_model` support; replacing only the alias can leave the real forward pass uncompiled.

## Dependency floor

The `embeddings` extra provides `torch>=2.9.0` (`>=2.13.0` on macOS, the release verified for MPS bf16), `transformers>=5.2.0`, `sentence-transformers>=5.7.0`, `datasets>=2.14.0`, and `huggingface_hub>=0.24.0`. The automatic-dtype contract comes from Transformers - Sentence Transformers only forwards it, so its floor is the oldest release verified against this stack.

The default suite covers prompt routing, cache separation, and complete vector sets. Run real-model checks as described in [Contributing](../../CONTRIBUTING.md#before-you-open-a-pr).

## Implementation references

Defaults, aliases, and the fallback chain live in [data/model_profiles.py](../../src/citemesh/data/model_profiles.py); load, TF32, and compile guards in [strategies/embedding/model_runtime.py](../../src/citemesh/strategies/embedding/model_runtime.py); device and capability probes in [strategies/embedding/runtime.py](../../src/citemesh/strategies/embedding/runtime.py).
