# Agent / Contributor Notes

Rules for working on CiteMesh, for humans and coding agents.

## Scope

- Maintain [CI](.github/workflows/ci.yml) as one Linux/Python job running Ruff lint/format and Mypy. Do not restore tests or add build validation, benchmarks, model runs, extra workflows, release automation, or publishing steps without explicit scope and cost approval.
- No compatibility layers, hypothetical platform/user support, or security machinery unless requested or intrinsic to the task; no checksums, verification manifests, or extra reporting artifacts for ordinary work.
- Keep the validation and error handling real failure modes require; do not invent trust boundaries or defensive frameworks.
- Ask before expanding the request's or the repo's natural scope. Use sub-agents when sensible.

## Git workflow

- Commit at logical increments; keep unrelated changes in separate commits.
- Assume squash-merge; raise it if that seems wrong for the change at hand.
- Never commit generated outputs, comparison JSONs, or caches. Curated screenshots and the example dashboard under `assets/` are the exception: refresh these deliberately, not on every build.
- Ask before destructive git operations. NEVER `git push` without explicit instruction or approval in the prior turn.

## Model and dtype policy (non-negotiable)

Apply the [model-loading and precision rules](docs/reference/embedding-runtime.md) to implementation, tests, calibration, validation, benchmarks, and fallbacks. Persistent embeddings follow the [storage contract](docs/guides/caching.md#embedding-namespaces).

For real inference, use the [designated model](docs/reference/embedding-runtime.md#model-selection-and-fallback) or a newer suitable one. Never substitute an older or smaller model without user approval; MiniLM is explicitly disallowed.

## Sandboxed execution caveat (macOS)

Inside sandboxed agent shells Metal is not visible: `torch.backends.mps.is_available()` falsely returns `False`, and network may be blocked. Anything touching torch devices, model downloads, or live APIs must run escalated or outside the sandbox. Unit tests are sandbox-safe: fake-torch harness, no real models.

## Environment and commands

Read the repository-root `LOCAL.md`, if present, for this machine's development
environment, test worker selection, and hardware notes. Keep `LOCAL.md` local and
gitignored; keep shared project instructions in this tracked `AGENTS.md`. If the
Python environment is not specified, ask which one to use.

Install the project for development with `pip install -e ".[all]"`. Versions come
from Git tags via setuptools-scm, so fetch tags when a checkout reports an
unexpected development version.

Before committing, run these checks using the environment and pytest worker
selection from `LOCAL.md` when available:

```bash
python -m pytest -n auto --dist worksteal
ruff check .
ruff format --check .
```

The default suite is network-free and excludes `slow` tests. Run it in parallel
when the machine has sufficient CPU and memory. Without a local worker selection,
`-n auto` starts one worker per physical CPU core using the development extra's
`psutil` dependency. Adjust with `-n N` for the machine's resources and other
workloads; use `-n 0` for serial debugging. Cache-locking, subprocess,
threaded-export, and rendering tests belong in this parallel suite: each test has
isolated cache and temporary paths.

Run opt-in real-model and compilation checks serially with `python -m pytest -n 0 -m slow`. These tests can compete for the same accelerator, CPU inference threads, and compiler resources. For CUDA only, use `python -m pytest -n 0 -m "slow and cuda"`. Some slow tests may download the designated model or fetch live arXiv data; CPU semantic-quality tests require the model already cached.

## Code conventions

- Prefer explicit code and focused changes. Add or update tests for changed behavior.
- Patch names where they are used, not where they were originally defined.
- Keep optional dependencies lazily imported and preserve the dependency direction
  in [Architecture](docs/internals/architecture.md#rules).
- Use INFO for brief phases and outcomes. Use WARNING only when results are
  materially degraded or an explicitly requested capability is unavailable. Put
  cache, provider, runtime, and scoring details at DEBUG.

## Runtime data

The cache root (`CITEMESH_CACHE_DIR`, see [Environment Variables](docs/reference/environment.md)) also holds the user config `config.toml` (`citemesh config`). Tests isolate it per test via `tests/conftest.py` and must never depend on network access.

## Curated demo

The checked-in Megalodon dashboard under `assets/examples/megalodon/` uses
`arxiv:2404.08801`, matching the bare input recorded in its collection package.
Use `arxiv:2404.08801v1` only when intentionally pinning arXiv bibliography
recovery to version 1. The `arxiv:2608.27147` identifiers in recovery tests refer
to Thomson and are not the demo seed.
