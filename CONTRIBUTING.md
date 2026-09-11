# Contributing to CiteMesh

Thanks for your interest! CiteMesh is a pre-1.0 project moving quickly; small, focused pull requests are the easiest to review and land.

## Getting set up

Install PyTorch for your hardware using the [official installation selector](https://pytorch.org/get-started/locally/)
before installing CiteMesh, so pip does not choose an unintended build.

```bash
git clone https://github.com/pszemraj/CiteMesh.git && cd CiteMesh
pip install -e ".[all]"
```

Versions come from git tags via setuptools-scm; a fork or shallow clone without tags reports `0.0.post1.devN`, so run `git fetch --tags` before installing.

Python >= 3.10. The `embeddings` extra needs torch (`>=2.9` Linux/Windows, `>=2.13` macOS); everything else runs without it.

Semantic Scholar SDK support is bounded to `>=0.8.0,<0.13` because CiteMesh adapts
its requester to preserve HTTP status codes. Check the service transport and
pagination tests before widening that range.

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
