# Agent / Contributor Notes

Practical conventions for working on CiteMesh (humans and coding agents).

## Scope

- CI exists and is maintained: [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs `ruff check .`, `ruff format --check .`, and `python -m pytest` on Python 3.10-3.13 (ubuntu-latest, CPU torch), plus a build job (`python -m build`, `twine check dist/*`, and a wheel-contents assertion for `py.typed` and the dashboard JS asset). Fix it when it breaks; do not add parallel workflows, release automation, or publishing steps unless asked.
- Do not add compatibility layers, support for hypothetical platforms/users, or security/integrity machinery unless requested or intrinsic to the task.
- Do not generate checksums, verification manifests, or extra reporting artifacts for ordinary local work.
- Keep the input validation and error handling that real failure modes require; do not invent trust boundaries or defensive frameworks.
- Ask before expanding the request's or the repo's natural scope.
- Use sub-agents when sensible.

## Git workflow

- Complete work atomically and commit as you go, at logical increments — not one batch commit of unrelated changes at the end.
- Assume squash-merge; raise it if that seems wrong for the change at hand.
- Never commit generated outputs, comparison JSONs, or caches.
- Ask before destructive git operations. NEVER `git push` without explicit instruction or approval in the prior turn.

## Environment

- Python >= 3.10. The maintainer's dev environment is the conda env `inf` (Python 3.12, torch 2.13+); install with `pip install -e ".[all]"`.
- Run project commands through the env, e.g. `conda run -n inf python -m pytest`.
- Install instructions and the platform-split torch floors are in the [README](README.md#install) — that is the single source; do not restate them elsewhere.
- The package uses a `src/` layout; tests import the installed package, so run `pip install -e ".[all]"` again after pulling a change that moves files.

## Model and dtype policy (non-negotiable)

- Never use FP16/float16/half precision anywhere in CiteMesh: not for model compute, autocast, embedding output, persistent storage, calibration, tests, benchmarks, or fallbacks. If a selected checkpoint or runtime resolves to FP16, stop and choose a compatible current model instead of proceeding.
- Load model weights with the library's automatic checkpoint dtype selection (`dtype="auto"`, or the supported compatibility equivalent). Do not force all weights to FP32, and never request FP16 weights.
- The only compute modes are verified BF16 autocast and FP32. If BF16 is unavailable or unverified, fall back to FP32.
- Embedding outputs are FP32. Persistent embedding storage is INT8 or FP32 only.
- Do not use obsolete or legacy embedding models for real inference, validation, benchmarks, or defaults. MiniLM is explicitly disallowed. Use the current project-designated model—presently `unsloth/embeddinggemma-300m`—or a newer suitable model; do not substitute an older/smaller model for convenience without explicit user approval.

## Sandboxed execution caveat (macOS)

Inside sandboxed agent shells (e.g. Claude Code's sandbox), Metal is not visible: `torch.backends.mps.is_available()` falsely returns `False` and network access may be blocked. Anything touching torch devices, model downloads, or live APIs must run escalated/outside the sandbox. Unit tests are sandbox-safe — they use a fake-torch harness and never load real models.

## Commands

The pre-PR command triple (pytest + `ruff check` + `ruff format --check`) is documented once, in [CONTRIBUTING.md](CONTRIBUTING.md#before-you-open-a-pr). Run it through the env:

```bash
conda run -n inf python -m pytest              # unit suite (slow tests excluded by default)
conda run -n inf python -m pytest -m slow      # real-model smoke tests (needs escalation on macOS)
```

The suite must be green and lint/format clean before committing. Real-model and MPS quality smokes remain opt-in local validation on the relevant hardware.

## Code conventions

- Explicit over clever.
- Docstrings: reST field style (`:param type name:`, `:return type:`) on every function, including tests' helpers where present.
- Comments state constraints the code can't, not narration of the change.
- Optional dependencies (torch, sentence-transformers, datasets, plotly, pyvis) must stay lazily imported so the core CLI works with no extras.
- The test suite is white-box: patch the name where it is used (the module under test's own binding), not the module that originally defined it.
- Argparse choices duplicated in `src/citemesh/data/user_config.py` are guarded by sync tests in `tests/test_user_config.py` — update both together.

## Docs rule

CLI flags, defaults, cache layout, or environment variables changed? Update the matching page under `docs/guides/` or `docs/reference/`. The docs are contract-style; stale docs are treated as bugs.

Never hard-wrap Markdown; editors soft-wrap.

Release notes are the sole change history. Do not create or maintain a separate changelog.

## Runtime data

- Cache root: `~/.cache/citemesh` (Linux/macOS) or `%LOCALAPPDATA%\CiteMesh` (Windows, falling back to `%APPDATA%`); override with `CITEMESH_CACHE_DIR`. Tests isolate it per-test via `tests/conftest.py`.
- Persistent user config: `<cache_root>/config.toml` (`citemesh config`).
- Live Semantic Scholar calls need `S2_API_KEY` for a dedicated rate limit; tests must never depend on network access.
