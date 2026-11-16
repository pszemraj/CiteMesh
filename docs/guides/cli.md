# CLI usage guide

`citemesh build` creates a graph; `search` finds seeds, `view` opens results, `cache` manages persisted data, and `config` saves defaults. See [installation](../../README.md#quick-start) and the [build pipeline](how-it-works.md).

## Common workflows

### Build a graph

```bash
# Minimal citation graph (static PNG, the default export)
citemesh build "arxiv:1706.03762" --strategy citation -p 20

# Dashboard collection in ./out, the default output directory; a second build adds to it
citemesh build "arxiv:1706.03762" -s hybrid -e dashboard -e csv --theme dark
citemesh build "arxiv:1810.04805" -s recommendation -e dashboard --theme dark

# Standalone single-file dashboard
citemesh build "arxiv:1706.03762" -s hybrid -e dashboard -o out/report.dashboard.html

# Embedding graph over a corpus slice, with a debug trace
citemesh build "arxiv:1810.04805" -s embedding --dataset-split "train[:2%]" \
  -e plotly --verbose --log-file out/run.log
```

Without `--export`, a build writes a static PNG. The `html`, `plotly`, and `dashboard` exports need the [viz extra](../../README.md#quick-start). Output directories, standalone files, and collection updates follow the [output-location rules](../reference/output-artifacts.md#output-location).

### Accepted identifiers

- DOI, bare or with a `doi:` prefix or `doi.org` URL; prefixed and URL forms normalize to lowercase, while a bare DOI keeps its input case
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
- `auto` tries local search first and falls back to S2 when the selected cache is empty or local dependencies or model loading are unavailable, logging the reason. An installed corpus that is unusable exits with status 1 instead of falling back. Causes: a pending import or refresh, uncertified membership, a requested view larger than the installed coverage, a source or runtime mismatch, or a model-fingerprint mismatch. Resume the pending operation or select matching settings before retrying.

Searching an installed portable index:

- `--cache-namespace NAMESPACE` selects the index directly. Its fixed model and source settings override unrelated saved defaults, and explicit conflicts are usage errors (exit 2).
- This selection never falls back to S2, including under `--mode auto`.
- Without it, search reads the namespace derived from the current model settings. That reaches an installed index only when those settings match its fixed ones; the default `candidates` source never does.
- When that namespace is empty, the `auto` fallback warning and the `--mode local` error name any installed corpora with a `--cache-namespace` command.
- For fixed settings, corpus views, and recovery, see [Portable corpus indexes](caching.md#portable-corpus-indexes).

Local search derives its namespace from the [saved embedding settings](configuration.md#supported-keys). Override them with `--model`, `--model-profile`, `--model-revision`, `--device`, `--semantic-source`, `--dataset-source`, `--truncate-dim`, `--storage-precision`, or `--calibration-sample-size`. A selector implies local mode unless `--mode auto` is explicit; selectors conflict with `--mode s2`. Invalid explicit option combinations fail as usage errors instead of triggering fallback.

Other search behavior:

- `--dataset-source` implies `arxiv-corpus` unless paired with a conflicting explicit source.
- Local results include the searched count and a copyable build command using the active model and recorded corpus scope; installed-index views are preserved in that command. The command is omitted with a warning if the cache has no recorded split.
- [API-key configuration](configuration.md#api-key) and [retries](#troubleshooting) apply whenever search uses S2; `--s2-retry-budget` overrides the recovery-time cap for an S2 search or `auto` fallback.

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

### Install, export, and refresh a corpus index

See [Portable corpus indexes](caching.md#portable-corpus-indexes) for the commands and their import, export, refresh, selection, and recovery behavior.

### Missing Semantic Scholar references

Hybrid builds request up to `--max-references` seed references from Semantic Scholar (1000 by default; `--max-references 0` turns reference discovery off) and rank every reference returned together with the other candidates. `--no-references` does not turn discovery off: it only skips the later fetch of the selected papers' reference lists used for bibliographic coupling. A retrieved reference that enters through another source still carries its known relationship to the seed.

If Semantic Scholar returns no usable seed references or reference discovery is unavailable, CiteMesh tries the seed's arXiv HTML bibliography when it has an arXiv ID. This applies to citation, hybrid, and embedding candidate collection. Corpus-only embedding search does not fetch HTML.

What recovery matches:

- Explicit arXiv IDs and DOIs in bibliography entries. Available local metadata is resolved before S2, and arXiv's metadata API resolves the remaining arXiv IDs.
- Entries without explicit identifiers, only when S2 returned an empty reference list and hybrid has a prepared corpus: a unique corpus paper with the same normalized title, full first-author name, and year. An arXiv preprint may be one year earlier than the bibliography's publication year; multiple matches across either year remain unresolved. This lookup scans the active corpus metadata once and makes no additional metadata API requests.

How recovered references are used:

- They compete under the existing reference and total-paper limits. Hybrid puts every recovered reference into the candidate pool up to the reference limit, where it is ranked like any other candidate; only a bibliography larger than the limit is cut first, by title and abstract similarity to the seed and then citation count.
- A supplied arXiv version selects that HTML version, while graph identities remain version-independent.
- Partial recovered lists are not used for shared-reference scoring or stored as complete S2 reference lists.

What recovery does not do:

- Missing HTML and unresolved references leave other available sources usable.
- It finds outgoing references, not incoming citations, and does not imply complete bibliography coverage.
- It needs no PDF/LaTeX extraction or extra dependency.

### Help and console output

Every command takes `-h`/`--help`, listing built-in defaults; [user configuration](configuration.md) shows how to inspect saved overrides. `--log-width` sizes result tables and logs but never help; [environment variables](../reference/environment.md#other-respected-variables) control color. Logs go to stderr.
Redirected logs use plain text without terminal padding or forced line breaks, unless you set `--log-width` or force color.

`-v`/`--verbose` is shorthand for `--log-level debug`; it works before or after a subcommand. When several verbosity options are present, the last one wins. `--log-level debug --log-file out/run.log` adds option routing, cache and provider details, runtime configuration, retry attempts, and namespace decisions.

| Level | Output |
| --- | --- |
| `error` | The requested operation failed. |
| `warning` | A result was materially degraded or an explicitly requested runtime feature could not be used. |
| `info` | Brief build phases, useful outcomes, and saved output locations. |
| `debug` | Cache, provider, runtime, and scoring diagnostics for development. |

Progress bars appear by default on an interactive terminal at `info` or `debug`, covering paper discovery, corpus queries, embedding batches, reference recovery, and reference enrichment. Provider listings with unknown totals show the completed count against `?`. Narrow terminals prioritize counts and ETA. Search results place year and score or citation count beneath each paper on narrow terminals, while keeping full paper IDs on their own lines.
Routine HTTP request logs stay hidden; long Semantic Scholar retry waits show a brief status at `info`, with request details at `debug`.

## Option reference

Use `citemesh COMMAND --help` for the current flags, defaults, accepted values, and command examples. Build options are strategy-scoped: an explicit flag unsupported by the selected strategy is a CLI error, while [configured defaults](configuration.md#precedence) follow their documented applicability rules.

Behavior that needs more context than command help lives in [Strategies](strategies.md), [Caching and Data](caching.md), [Embedding Runtime](../reference/embedding-runtime.md), and [Output Artifacts](../reference/output-artifacts.md). The sweeps behind tuned embedding defaults live in [Defaults Tuning Study](../reference/defaults-tuning-study.md).

### Export formats

Formats, dashboard collection and standalone behavior, package schemas, sidecars, and determinism notes: [Output Artifacts](../reference/output-artifacts.md). Interactive exports need the [viz extra](../../README.md#quick-start).

![CiteMesh dashboard built with --theme light, with a paper selected](../../assets/ui-dashboard-light-theme.png)

_`--theme light` with a paper selected._

## Troubleshooting

Semantic Scholar request policy:

- **Pacing**: authenticated requests run at 1 request/second, anonymous requests at 0.5.
- **Retries**: exponential full jitter, up to 30 attempts per HTTP request, the wait ceiling doubling from 2 seconds (4 for HTTP 429) to 60, and a numeric `Retry-After` honored as a floor up to 300 seconds per delay. Pagination retries only the failed page. HTTP 408 and server errors retry; bad parameters and rejected credentials fail immediately.
- **Recovery budget, anonymous**: builds share a 90-second budget for a candidate collection. Failed requests, retry attempts, and recovery waits consume it; healthy first requests, local embedding work, and rendering do not.
- **Recovery budget, authenticated**: builds keep the attempt limit with no elapsed cap unless `--s2-retry-budget SECONDS` overrides it; pass `0` to remove the elapsed cap while keeping 30 attempts.
- An in-flight HTTP request cannot be cancelled at an exact recovery-budget boundary.

Common symptoms:

- **No results / paper not found**: check the identifier format and S2 availability.
- **Discovery check unavailable**: required acquisition failure exits nonzero without replacing prior outputs; retry later, [configure an API key](configuration.md#api-key), or use the local `arxiv-corpus` semantic source, which requires the embeddings extras and a downloaded, embedded corpus. The error identifies the operation, HTTP failure when available, attempts, recovery time, and stopping reason.
- **Partial outage**: citation and hybrid builds can continue while at least one requested source completes or returns a valid empty result. A failed source does not by itself prevent checking another source. The recovery budget is shared across all requests: when it is spent, further uncached S2 calls stop for that collection. Export metadata marks every source `complete`, `empty`, or `unavailable`, and a total outage exits nonzero rather than emitting a seed-only graph. Optional reference enrichment stops further network lookups after an outage while still reusing available cached reference lists.
- **Slow first embedding run, or a warning that results come from a larger cached corpus**: see [hydration and resume](caching.md#corpus-hydration-and-resume).
