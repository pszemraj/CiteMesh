# Agent / Contributor Notes

Rules for working on CiteMesh, for humans and coding agents.

## Scope

- CI is maintained ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)): fix it when it breaks, and add no parallel workflows, release automation, or publishing steps unless asked.
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

Install the project for development with `pip install -e ".[all]"`. Versions come
from Git tags via setuptools-scm, so fetch tags when a checkout reports an
unexpected development version.

Run every project command in the configured development environment. Before
committing, run:

```bash
python -m pytest
ruff check .
ruff format --check .
```

The default suite is network-free and excludes slow tests. Real CUDA checks are
opt-in with `python -m pytest -m "slow and cuda"` and may download the designated
embedding model.

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
