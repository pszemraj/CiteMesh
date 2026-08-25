# Agent / Contributor Notes

Practical conventions for working on CiteMesh (humans and coding agents).

## Environment

- Python >= 3.10. The maintainer's dev environment is the conda env `inf`
  (Python 3.12, torch 2.13+); install with `pip install -e ".[all]"`.
- Run project commands through the env, e.g.
  `conda run -n inf python -m pytest`.
- Torch floors are platform-split: `>=2.9` Linux/Windows, `>=2.13` macOS
  (required for reliable MPS bf16).

## Sandboxed execution caveat (macOS)

Inside sandboxed agent shells (e.g. Claude Code's sandbox), Metal is not
visible: `torch.backends.mps.is_available()` falsely returns `False` and
network access may be blocked. Anything touching torch devices, model
downloads, or live APIs must run escalated/outside the sandbox. Unit tests are
sandbox-safe — they use a fake-torch harness and never load real models.

## Commands

```bash
conda run -n inf python -m pytest              # unit suite (slow tests excluded by default)
conda run -n inf python -m pytest -m slow      # real-model smoke tests (needs escalation on macOS)
conda run -n inf ruff check . && conda run -n inf ruff format .
```

The suite must be green and `ruff check` + `ruff format --check` clean before
committing. CI runs lint plus tests on Linux/macOS across supported Pythons,
and a no-extras install smoke (`pip install .` then `citemesh --help`), so
keep optional deps lazily imported.

## Code conventions

- Docstrings: reST field style (`:param type name:`, `:return type:`) on every
  function, including tests' helpers where present.
- Comments state constraints the code can't, not narration of the change.
- Optional dependencies (torch, sentence-transformers, datasets, plotly,
  pyvis) must stay lazily imported so the core CLI works with no extras.
- Argparse choices duplicated in `citemesh/core/user_config.py` are guarded by
  sync tests in `tests/test_user_config.py` — update both together.

## Docs rule

CLI flags, defaults, cache layout, or environment variables changed? Update
the matching page under `docs/guides/` or `docs/reference/`, and add a line to
`docs/internals/changelog.md`. The docs are contract-style; stale docs are
treated as bugs.

## Runtime data

- Cache root: `~/.cache/citemesh` (Linux/macOS) or `%LOCALAPPDATA%\CiteMesh`
  (Windows); override with `CITEMESH_CACHE_DIR`. Tests isolate it per-test via
  `tests/conftest.py`.
- Persistent user config: `<cache_root>/config.toml` (`citemesh config`).
- Live Semantic Scholar calls need `S2_API_KEY` for a dedicated rate limit;
  tests must never depend on network access.
