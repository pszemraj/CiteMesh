# Contributing to CiteMesh

CiteMesh is a pre-1.0 project moving quickly; small, focused pull requests are the easiest to review and land.

## Getting set up

Follow the [installation prerequisites](README.md#quick-start), then install an editable checkout:

```bash
git clone https://github.com/pszemraj/CiteMesh.git && cd CiteMesh
pip install -e ".[all]"
```

Versions come from Git tags via setuptools-scm. Use a checkout with release tags (`git fetch --tags`); shallow history can report an incorrect development version. Tests import the installed package from the `src/` layout, so rerun `pip install -e ".[all]"` after pulling a change that moves or renames modules.

## Before you open a PR

```bash
python -m pytest              # must be green (slow tests excluded by default)
ruff check .
ruff format --check .
```

CI runs Ruff and the full network-free test suite in one Python 3.12 job (`ubuntu-latest`, CPU torch). Markdown-only changes skip CI, and new PR runs cancel older runs for that PR while every push to `main` completes ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). Python-version matrices, distribution builds, and installation smoke tests are not part of CI. Real CUDA embedding checks stay opt-in; they may download the EmbeddingGemma checkpoint and need a GPU: `python -m pytest -m "slow and cuda"`.

Clear stale output before building locally (`rm -rf build dist && python -m build`): setuptools reuses `build/lib`, so a wheel built over an old tree can silently ship modules you deleted.

## Code conventions

- Prefer explicit code over clever shortcuts.
- Add or update tests for changed behavior. Tests are network-free and isolate the cache directory per test; Semantic Scholar transport and pagination tests use mocked HTTP responses and clocks.
- Patch names where they are used: `patch("citemesh.strategies.recommendation.get_client")` replaces the strategy's binding; patching `citemesh.services.get_client` after import does not.
- Use reST field docstrings (`:param type name:`, `:return type:`) on every function, including test helpers. Comments explain constraints rather than narrating changes.
- Follow the [dependency rules](docs/internals/architecture.md#rules), including lazy imports for optional dependencies.
- Shared CLI/config vocabularies live in `core/choices.py`; import them instead of maintaining duplicate literals.
- Update the matching guide or reference when changing a CLI flag, default, cache layout, or environment variable. Do not hard-wrap Markdown.
- Use INFO only for brief phases and outcomes. Use WARNING when results are materially degraded or an explicitly requested runtime capability is unavailable. Put cache, provider, runtime, and scoring details at DEBUG; reproduce them with `--verbose` or `--log-level debug`.
- Use release notes for change history; do not maintain a separate changelog.

Agent environment and Git instructions: [AGENTS.md](AGENTS.md).

## Pages demo

The [Pages workflow](.github/workflows/pages.yml) publishes the [saved Megalodon dashboard](assets/examples/megalodon/dashboard.html) and its [collection package](assets/examples/megalodon/dashboard.citemesh.json) to the [live demo](https://pszemraj.github.io/CiteMesh/). It copies these files without rebuilding them, so update the saved HTML alongside dashboard template or style changes and review it in a browser. Pushes to public `main` redeploy automatically; manual workflow runs must also target `main`.

## Bugs, features, and scope

Use the [issue templates](https://github.com/pszemraj/CiteMesh/issues/new/choose); for bugs, include the full `citemesh` command, log output (ideally `--log-level debug`), OS, Python version, and how you installed CiteMesh. Fixes for live Semantic Scholar behavior (rate limits, payload quirks) are very welcome. New strategies or embedding providers should use the existing strategy and template seams rather than parallel code paths - open an issue first.
