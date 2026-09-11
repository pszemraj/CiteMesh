# Contributing to CiteMesh

Thanks for your interest! CiteMesh is a pre-1.0 project moving quickly; small, focused pull requests are the easiest to review and land.

## Getting set up

Install PyTorch first, then CiteMesh — the canonical install instructions and the platform-split torch floors live in the [README](README.md#install). For development you want the editable install with every extra:

```bash
git clone https://github.com/pszemraj/CiteMesh.git && cd CiteMesh
pip install -e ".[all]"
```

Versions come from git tags via setuptools-scm; a fork or shallow clone without tags reports `0.0.post1.devN`, so run `git fetch --tags` before installing. The package uses a `src/` layout and the tests import the installed package, so rerun `pip install -e ".[all]"` after pulling a change that moves or renames modules.

Python >= 3.10. Everything except the `embeddings` extra runs without torch.

Semantic Scholar SDK support is bounded to `>=0.8.0,<0.13` because CiteMesh adapts its requester to preserve HTTP status codes. Check the service transport and pagination tests before widening that range.

## Before you open a PR

```bash
python -m pytest              # must be green (slow tests excluded by default)
ruff check .
ruff format --check .
```

The real CUDA embedding checks are opt-in because they may download the designated EmbeddingGemma checkpoint and require an available GPU:

```bash
python -m pytest -m "slow and cuda"
```

CI runs the same three commands on every push to `main` and every pull request: `ruff check .`, `ruff format --check .`, and `python -m pytest` across Python 3.10, 3.11, 3.12, and 3.13 on `ubuntu-latest` (CPU torch), plus a separate job that runs `python -m build`, `twine check dist/*`, and asserts the wheel ships `citemesh/py.typed` and `citemesh/visualization/dashboard/assets/dashboard.js`. The workflow is [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

If you run `python -m build` locally, delete any stale `build/` directory first. setuptools reuses `build/lib` between invocations, so a wheel built over an old tree can silently ship modules you deleted or moved:

```bash
rm -rf build dist && python -m build
```

- Add or update tests for behavior you change. The default suite is network-free and isolates the cache directory per test.
- The suite is white-box: patch the name where it is *used*, at the module that defines the binding under test. `patch("citemesh.strategies.recommendation.get_client")` replaces the binding that strategy actually calls; patching `citemesh.services.get_client` after the strategy module has imported the name does nothing.
- If you change CLI flags, defaults, cache layout, or environment variables, update the matching page under `docs/`.
- Release notes are the sole change history; no separate changelog is maintained.
- Follow the existing reST docstring style (`:param type name:`, `:return type:`).
- Never hard-wrap Markdown; editors soft-wrap.

Agent-assisted development notes live in [AGENTS.md](AGENTS.md).

## Reporting bugs / requesting features

Use the [issue templates](https://github.com/pszemraj/CiteMesh/issues/new/choose). For bugs, include the full `citemesh` command, the log output (ideally with `--log-level debug`), OS, Python version, and how you installed CiteMesh.

## Scope notes

- Live Semantic Scholar behavior (rate limits, payload quirks) changes over time; fixes there are very welcome.
- New graph strategies or embedding providers should implement the existing strategy/template seams rather than adding parallel code paths - open an issue first to discuss the design.
