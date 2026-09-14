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

Apply the [model-loading and precision rules](docs/reference/embedding-runtime.md) to implementation, tests, calibration, validation, benchmarks, and fallbacks. Persistent embeddings follow the [storage contract](docs/guides/cli.md#graph-edges-and-cache-storage).

For real inference, use the [designated model](docs/reference/embedding-runtime.md#model-selection-and-fallback) or a newer suitable one. Never substitute an older or smaller model without user approval; MiniLM is explicitly disallowed.

## Sandboxed execution caveat (macOS)

Inside sandboxed agent shells Metal is not visible: `torch.backends.mps.is_available()` falsely returns `False`, and network may be blocked. Anything touching torch devices, model downloads, or live APIs must run escalated or outside the sandbox. Unit tests are sandbox-safe: fake-torch harness, no real models.

## Environment and commands

The maintainer's dev environment is the conda env `inf` (Python 3.12, torch 2.13+); run every project command through it. Install and setup live once in [CONTRIBUTING.md](CONTRIBUTING.md#getting-set-up), the pre-PR triple once in [its PR section](CONTRIBUTING.md#before-you-open-a-pr) — do not restate either here. Keep the suite green and lint-clean before committing.


## Code conventions

Follow the [contributor conventions](CONTRIBUTING.md#code-conventions).

## Runtime data

The cache root (`CITEMESH_CACHE_DIR`, see [Environment Variables](docs/reference/environment.md)) also holds the user config `config.toml` (`citemesh config`). Tests isolate it per test via `tests/conftest.py` and must never depend on network access.
