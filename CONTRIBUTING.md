# Contributing to CiteMesh

CiteMesh is a pre-1.0 project moving quickly; small, focused pull requests are the easiest to review and land.

## Getting set up

> [!IMPORTANT]
> Install PyTorch for your platform *before* CiteMesh, or pip may resolve a build you did not intend. Only the `embeddings` extra needs torch; the platform-split floors are in the [README](README.md#quick-start).

```bash
git clone https://github.com/pszemraj/CiteMesh.git && cd CiteMesh
pip install -e ".[all]"
```

Python >= 3.10. Versions come from git tags via setuptools-scm, so a shallow clone without tags reports `0.0.post1.devN` — run `git fetch --tags` first. Tests import the installed package from the `src/` layout, so rerun `pip install -e ".[all]"` after pulling a change that moves or renames modules.

The Semantic Scholar SDK is pinned to `>=0.8.0,<0.13` because CiteMesh adapts its requester to preserve HTTP status codes; check the transport and pagination tests before widening that range.

## Before you open a PR

```bash
python -m pytest              # must be green (slow tests excluded by default)
ruff check .
ruff format --check .
```

CI runs those three on Python 3.10-3.13 (`ubuntu-latest`, CPU torch) plus a build job — `python -m build`, `twine check dist/*`, and a wheel check for `citemesh/py.typed` and the dashboard JS ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). Real CUDA embedding checks stay opt-in; they may download the EmbeddingGemma checkpoint and need a GPU: `python -m pytest -m "slow and cuda"`.

Clear stale output before building locally (`rm -rf build dist && python -m build`): setuptools reuses `build/lib`, so a wheel built over an old tree can silently ship modules you deleted.

- Add or update tests for behavior you change. The default suite is network-free and isolates the cache directory per test.
- The suite is white-box: patch the name where it is *used*. `patch("citemesh.strategies.recommendation.get_client")` replaces the binding that strategy calls; patching `citemesh.services.get_client` after the module imported it does nothing.
- Changed a CLI flag, default, cache layout, or environment variable? Update the matching page under `docs/`.
- Release notes are the sole change history; no separate changelog is maintained.
- Follow the existing reST docstring style (`:param type name:`, `:return type:`), and never hard-wrap Markdown.

Agent-assisted development notes live in [AGENTS.md](AGENTS.md).

## Bugs, features, and scope

Use the [issue templates](https://github.com/pszemraj/CiteMesh/issues/new/choose); for bugs, include the full `citemesh` command, log output (ideally `--log-level debug`), OS, Python version, and how you installed CiteMesh. Fixes for live Semantic Scholar behavior (rate limits, payload quirks) are very welcome. New strategies or embedding providers should use the existing strategy and template seams rather than parallel code paths — open an issue first.
