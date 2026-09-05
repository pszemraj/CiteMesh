# Contributing to CiteMesh

Thanks for your interest! CiteMesh is a pre-1.0 project moving quickly; small, focused pull requests are the easiest to review and land.

## Getting set up

```bash
git clone https://github.com/pszemraj/CiteMesh.git && cd CiteMesh
pip install -e ".[all]"
```

Python >= 3.10. The `embeddings` extra needs torch (`>=2.9` Linux/Windows, `>=2.13` macOS); everything else runs without it.

## Before you open a PR

```bash
python -m pytest              # must be green (slow tests excluded by default)
ruff check .
ruff format --check .
```

The real CUDA embedding checks are opt-in because they may download the designated
EmbeddingGemma checkpoint and require an available GPU:

```bash
python -m pytest -m "slow and cuda"
```

- Add or update tests for behavior you change. The default suite is network-free and isolates the cache directory per test.
- If you change CLI flags, defaults, cache layout, or environment variables, update the matching page under `docs/`.
- Release notes are the sole change history; no separate changelog is maintained.
- Follow the existing reST docstring style (`:param type name:`, `:return type:`).

Agent-assisted development notes live in [AGENTS.md](AGENTS.md).

## Reporting bugs / requesting features

Use the [issue templates](https://github.com/pszemraj/CiteMesh/issues/new/choose). For bugs, include the full `citemesh` command, the log output (ideally with `--log-level debug`), OS, Python version, and how you installed CiteMesh.

## Scope notes

- Live Semantic Scholar behavior (rate limits, payload quirks) changes over time; fixes there are very welcome.
- New graph strategies or embedding providers should implement the existing strategy/template seams rather than adding parallel code paths - open an issue first to discuss the design.
- CI is intentionally small for the project's current solo-maintained, pre-user stage: lint/format, representative Linux/macOS tests, and a no-extras install smoke. Add jobs or matrix breadth only for a concrete compatibility bug or release requirement.
