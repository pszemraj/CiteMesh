# CLI usage guide

`citemesh build` creates a graph; `search` finds seeds, `view` opens results, `cache` manages persisted data, and `config` saves defaults. See [installation](../../README.md#quick-start) and the [build pipeline](how-it-works.md).

## Common workflows

### Build a graph

```bash
# Minimal citation graph
citemesh build "arxiv:1706.03762" --strategy citation -p 20

# Dashboard collection in ./out (default); a second build adds to it
citemesh build "arxiv:1706.03762" -s hybrid -e dashboard -e csv --theme dark
citemesh build "arxiv:1810.04805" -s recommendation -e dashboard --theme dark

# Standalone single-file dashboard
citemesh build "arxiv:1706.03762" -s hybrid -e dashboard -o out/report.dashboard.html

# Embedding graph over a corpus slice, with a debug trace
citemesh build "arxiv:1810.04805" -s embedding --dataset-split "train[:2%]" \
  -e plotly --verbose --log-file out/run.log
```

Output directories, standalone files, and collection updates follow the [output-location rules](../reference/output-artifacts.md#output-location).

### Accepted identifiers

- DOI, bare or with a `doi:` prefix or `https://doi.org/` URL, canonicalized to lowercase (DOI names are case-insensitive)
- arXiv ID or URL (`arxiv:1706.03762`, `https://arxiv.org/abs/1706.03762`, `.../pdf/1706.03762.pdf`), `vN` suffixes normalized away; bare `1706.03762` works when S2 resolves it
- Semantic Scholar paper ID
- Free-form text under `--strategy embedding`, when S2 says the input is not a known paper (outages stay errors)

### Find a seed paper

```bash
citemesh search "attention mechanism transformers" --limit 5
citemesh build "<paper-id-from-search>" --strategy recommendation
# search what an arXiv-corpus build hydrated
citemesh search "long-context attention" --semantic-source arxiv-corpus
```

Search modes:

- `local` encodes the query and ranks it against cached retrieval-document vectors, returning cosine scores and full paper IDs. It fetches no paper metadata and fails if the selected cache is empty.
- `s2` performs Semantic Scholar keyword search and returns citation counts. Requests above 100 results are paginated, up to the service's 1,000-result relevance limit. Local search accepts any positive limit.
- `auto` tries local search first and falls back to S2 when the cache is empty or unavailable, logging the reason.

Local search derives its namespace from the [saved embedding settings](configuration.md#supported-keys). Override them with `--model`, `--model-profile`, `--model-revision`, `--device`, `--semantic-source`, `--dataset-source`, `--truncate-dim`, `--storage-precision`, or `--calibration-sample-size`. A selector implies local mode unless `--mode auto` is explicit; selectors conflict with `--mode s2`. Invalid explicit option combinations fail as usage errors instead of triggering fallback.

Match the [namespace used by the build](caching.md#embedding-namespaces). `--dataset-source` implies `arxiv-corpus` unless paired with a conflicting explicit source. Local results include the searched count and a copyable build command using the active model and recorded corpus scope; the command is omitted with a warning if the cache has no recorded split. [API-key configuration](configuration.md#api-key) and [retries](#appendix-b-troubleshooting) apply whenever search uses S2; `--s2-retry-budget` overrides the recovery-time cap for an S2 search or `auto` fallback.

### View a saved dashboard

`citemesh view [PATH]` opens a saved HTML export in a browser without rebuilding or starting a server. `PATH` defaults to `out/dashboard.html`; a directory resolves to its `dashboard.html`, and an explicit `.html`/`.htm` file works. A missing dashboard, non-HTML input, or failed launch exits `1`.

```bash
citemesh view
citemesh view out/my-collection
# pick a browser
citemesh view out/report.dashboard.html --browser google-chrome
```

### Save defaults, inspect the cache

```bash
citemesh config set defaults.semantic_source arxiv-corpus
citemesh cache scan
```

See [User Configuration](configuration.md) for saved defaults and [cache maintenance](caching.md#inspecting-and-clearing) for scan and reset behavior.

### Missing Semantic Scholar references

If Semantic Scholar returns no usable seed references or reference discovery is unavailable, CiteMesh tries the seed's arXiv HTML bibliography when it has an arXiv ID. This applies to citation, hybrid, and embedding candidate collection; corpus-only embedding search does not fetch HTML. Nonempty S2 reference results are unchanged.

Recovery uses explicit arXiv IDs and DOIs in bibliography entries, resolves available local metadata before S2, and uses arXiv's metadata API for remaining arXiv IDs. It respects reference and total-paper limits. A supplied arXiv version selects that HTML version, while graph identities remain version-independent. Missing HTML and unresolved references leave other available sources usable. Recovery finds outgoing references, not incoming citations, and does not imply complete bibliography coverage. Partial recovered lists are not used for shared-reference scoring or stored as complete S2 reference lists. No PDF/LaTeX extraction or extra dependency is required.

### Help and console output

Every command takes `-h`/`--help`, listing built-in defaults; [user configuration](configuration.md) shows how to inspect saved overrides. `--log-width` sizes result tables and logs but never help; [environment variables](../reference/environment.md#other-respected-variables) control color. Logs go to stderr.

`-v`/`--verbose` is shorthand for `--log-level debug`; it works before or after a subcommand. When several verbosity options are present, the last one wins. `--log-level debug --log-file out/run.log` adds option routing, cache and provider details, runtime configuration, retry attempts, and namespace decisions.

| Level | Output |
| --- | --- |
| `error` | The requested operation failed. |
| `warning` | A result was materially degraded or an explicitly requested runtime feature could not be used. |
| `info` | Brief build phases, useful outcomes, and saved output locations. |
| `debug` | Cache, provider, runtime, and scoring diagnostics for development. |

Progress bars appear only on an interactive terminal at `info` or `debug`.

## Option reference

Use `citemesh COMMAND --help` for the current flags, defaults, accepted values, and command examples. Build options are strategy-scoped: an explicit flag unsupported by the selected strategy is a CLI error, while [configured defaults](configuration.md#precedence) follow their documented applicability rules.

Behavior that needs more context than command help lives in [Strategies](strategies.md), [Caching and Data](caching.md), [Embedding Runtime](../reference/embedding-runtime.md), and [Output Artifacts](../reference/output-artifacts.md). The sweeps behind tuned embedding defaults live in [Defaults Tuning Study](../reference/defaults-tuning-study.md).

### Export formats

Formats, dashboard collection and standalone behavior, package schemas, sidecars, and determinism notes: [Output Artifacts](../reference/output-artifacts.md). Interactive exports need the [viz extra](../../README.md#quick-start).

![CiteMesh dashboard built with --theme light, with a paper selected](../../assets/ui-dashboard-light-theme.png)

_`--theme light` with a paper selected._

## Appendix B: Troubleshooting

Semantic Scholar calls retry with exponential full jitter: up to 30 attempts per HTTP request, the wait ceiling doubling from 2 seconds (4 for HTTP 429) to 60, and a numeric `Retry-After` honored as a floor up to 300 seconds per delay. Pagination retries only the failed page. Anonymous builds have a shared 90-second recovery budget for a candidate collection; failed requests, retry attempts, and recovery waits consume it, while healthy first requests, local embedding work, and rendering do not. Authenticated builds retain the attempt limit without an elapsed cap unless `--s2-retry-budget SECONDS` overrides it; pass `0` to remove the elapsed cap while keeping 30 attempts. An in-flight HTTP request cannot be cancelled at an exact recovery-budget boundary. HTTP 408 and server errors retry; bad parameters and rejected credentials fail immediately.

- **No results / paper not found**: check the identifier format and S2 availability.
- **Discovery check unavailable**: required acquisition failure exits nonzero without replacing prior outputs; retry later, [configure an API key](configuration.md#api-key), or use the local `arxiv-corpus` semantic source, which requires the embeddings extras and a downloaded, embedded corpus. The error identifies the operation, HTTP failure when available, attempts, recovery time, and stopping reason.
- **Partial outage**: citation and hybrid builds can continue while at least one requested source completes or returns a valid empty result. A failed source does not by itself prevent checking another source. The recovery budget is shared across all requests: when it is spent, further uncached S2 calls stop for that collection. Export metadata marks every source `complete`, `empty`, or `unavailable`, and a total outage exits nonzero rather than emitting a seed-only graph. Optional reference enrichment stops further network lookups after an outage while still reusing available cached reference lists.
- **Slow first embedding run or an older corpus cap in the logs**: see [hydration and resume](caching.md#corpus-hydration-and-resume).
