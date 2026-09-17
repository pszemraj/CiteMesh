# Contributing to CiteMesh

CiteMesh is a pre-1.0 project moving quickly; small, focused pull requests are the easiest to review and land.

## Getting set up

Follow the [installation prerequisites](README.md#quick-start), then install an editable checkout:

```bash
git clone https://github.com/pszemraj/CiteMesh.git && cd CiteMesh
pip install -e ".[all]"
```

Versions come from git tags via setuptools-scm, so a shallow clone without tags reports `0.0.post1.devN` - run `git fetch --tags` first. Tests import the installed package from the `src/` layout, so rerun `pip install -e ".[all]"` after pulling a change that moves or renames modules.

Semantic Scholar requests use the HTTP transport in `services/semantic_scholar/client.py`; endpoint pagination and payload conversion live alongside it. Transport and pagination tests use mocked HTTP responses and clocks.

## Before you open a PR

```bash
python -m pytest              # must be green (slow tests excluded by default)
ruff check .
ruff format --check .
```

CI runs Ruff and the full network-free test suite in one Python 3.12 job (`ubuntu-latest`, CPU torch). Markdown-only changes skip CI, and new runs cancel older runs for the same branch or PR ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). Python-version matrices, distribution builds, and installation smoke tests are not part of CI. Real CUDA embedding checks stay opt-in; they may download the EmbeddingGemma checkpoint and need a GPU: `python -m pytest -m "slow and cuda"`.

Clear stale output before building locally (`rm -rf build dist && python -m build`): setuptools reuses `build/lib`, so a wheel built over an old tree can silently ship modules you deleted.

## Code conventions

- Prefer explicit code over clever shortcuts.
- Add or update tests for changed behavior. Tests are network-free and isolate the cache directory per test.
- Patch names where they are used: `patch("citemesh.strategies.recommendation.get_client")` replaces the strategy's binding; patching `citemesh.services.get_client` after import does not.
- Use reST field docstrings (`:param type name:`, `:return type:`) on every function, including test helpers. Comments explain constraints rather than narrating changes.
- Follow the [dependency rules](docs/internals/architecture.md#rules), including lazy imports for optional dependencies.
- Shared CLI/config vocabularies live in `core/choices.py`; import them instead of maintaining duplicate literals.
- Update the matching guide or reference when changing a CLI flag, default, cache layout, or environment variable. Do not hard-wrap Markdown.
- Use release notes for change history; do not maintain a separate changelog.

Agent environment and Git instructions: [AGENTS.md](AGENTS.md).

## Bugs, features, and scope

Use the [issue templates](https://github.com/pszemraj/CiteMesh/issues/new/choose); for bugs, include the full `citemesh` command, log output (ideally `--log-level debug`), OS, Python version, and how you installed CiteMesh. Fixes for live Semantic Scholar behavior (rate limits, payload quirks) are very welcome. New strategies or embedding providers should use the existing strategy and template seams rather than parallel code paths - open an issue first.
