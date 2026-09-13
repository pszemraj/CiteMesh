# Agent / Contributor Notes

Rules for working on CiteMesh, for humans and coding agents. Setup, the pre-PR checks, and CI live in [CONTRIBUTING.md](CONTRIBUTING.md).

## Scope

- CI is maintained ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)): fix it when it breaks, and add no parallel workflows, release automation, or publishing steps unless asked.
- No compatibility layers, hypothetical platform/user support, or security machinery unless requested or intrinsic to the task; no checksums, verification manifests, or extra reporting artifacts for ordinary work.
- Keep the validation and error handling real failure modes require; do not invent trust boundaries or defensive frameworks.
- Ask before expanding the request's or the repo's natural scope. Use sub-agents when sensible.

## Git workflow

- Commit as you go, at logical increments — never one batch commit of unrelated changes at the end.
- Assume squash-merge; raise it if that seems wrong for the change at hand.
- Never commit generated outputs, comparison JSONs, or caches. The one exception is curated documentation assets under `assets/` — the screenshots and the example dashboard in `assets/examples/` are produced by a build but committed on purpose, refreshed deliberately rather than per run, and linked from README and the docs as the no-install example.
- Ask before destructive git operations. NEVER `git push` without explicit instruction or approval in the prior turn.

## Model and dtype policy (non-negotiable)

> [!IMPORTANT]
> Never use FP16/float16/half precision anywhere in CiteMesh: not for model compute, autocast, embedding output, persistent storage, calibration, tests, benchmarks, or fallbacks. If a selected checkpoint or runtime resolves to FP16, stop and choose a compatible current model instead of proceeding.

- Load weights with the library's automatic dtype selection (`dtype="auto"`, or the supported equivalent): do not force all weights to FP32, and never request FP16 weights.
- The only compute modes are verified BF16 autocast and FP32; if BF16 is unavailable or unverified, fall back to FP32.
- Embedding outputs are FP32. Persistent embedding storage is INT8 or FP32 only.
- No obsolete or legacy embedding models for real inference, validation, benchmarks, or defaults; MiniLM is explicitly disallowed. Use the project-designated model — presently `unsloth/embeddinggemma-300m` — or a newer suitable one, never an older or smaller substitute without user approval.

## Sandboxed execution caveat (macOS)

Inside sandboxed agent shells Metal is not visible: `torch.backends.mps.is_available()` falsely returns `False`, and network may be blocked. Anything touching torch devices, model downloads, or live APIs must run escalated or outside the sandbox. Unit tests are sandbox-safe: fake-torch harness, no real models.

## Environment and commands

The maintainer's dev environment is the conda env `inf` (Python 3.12, torch 2.13+); run every project command through it. Install and setup live once in [CONTRIBUTING.md](CONTRIBUTING.md#getting-set-up), the pre-PR triple once in [its PR section](CONTRIBUTING.md#before-you-open-a-pr) — do not restate either here. Keep the suite green and lint-clean before committing.

```bash
conda run -n inf python -m pytest              # unit suite (slow tests excluded by default)
conda run -n inf python -m pytest -m slow      # real-model smokes (needs escalation on macOS)
```

## Code conventions

- Explicit over clever.
- Docstrings: reST field style (`:param type name:`, `:return type:`) on every function, test helpers included.
- Comments state constraints the code can't, not narration of the change.
- Optional dependencies (torch, sentence-transformers, datasets, plotly, pyvis) stay lazily imported so the core CLI works with no extras.
- The test suite is white-box: patch the name where it is used, in the module under test.
- Argparse choices duplicated in `data/user_config.py` are guarded by `tests/test_user_config.py` — update both together.
- Never hard-wrap Markdown; editors soft-wrap.
- Changed a CLI flag, default, cache layout, or env var? Update the matching page under `docs/`; stale docs are bugs.
- Release notes are the sole change history. Do not create or maintain a separate changelog.

## Runtime data

The cache root (`CITEMESH_CACHE_DIR`, see [Environment Variables](docs/reference/environment.md)) also holds the user config `config.toml` (`citemesh config`). Tests isolate it per test via `tests/conftest.py` and must never depend on network access.
