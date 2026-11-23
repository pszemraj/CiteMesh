# Semantic Scholar request policy

Why the Semantic Scholar (S2) client retries the way it does, and the measurements behind it. User-facing behavior and flags are in the [CLI request policy](../guides/cli.md#troubleshooting). Code lives in `src/citemesh/services/semantic_scholar/` (`client.py` for the request loop and pacing, `retry.py` for delays, `endpoints.py` for sections and batching).

## What S2 rate limiting looks like

S2's HTTP 429 responses come from AWS API Gateway (`x-amzn-errortype: TooManyRequestsException`). They carry no `Retry-After`, and the body suggests applying for a key even when the request sent one.

In a 120-request probe that interleaved keyed and anonymous requests (2026-11-22), 62% of keyed and 88% of anonymous requests got a 429. The share did not fall with waits from 1 second to over 60 seconds, including the first request after the machine sat idle. The rate also swings over time: an anonymous cold build an hour later got no 429 on any of its 7 requests.

A 429 from S2 therefore behaves like random load shedding, not a token bucket that refills while the client waits. Each retry is close to an independent draw, so the cost of recovery is the number of draws times the time spent between them. Waiting longer between draws buys nothing.

## Policy

- **Pacing.** One request at a time, at least 1.2 seconds apart with a key and 2 seconds apart without. This keeps the client inside S2's published per-key rate, so a 429 is S2's load shedding, not ours.
- **429 without `Retry-After`.** Retry at the next pacing slot. Never back off exponentially on a 429.
- **`Retry-After`.** Honored when present: the delay applies to the request and advances a client-wide cooldown, so every later request through that client waits too.
- **Server errors, HTTP 408, connection failures.** Exponential backoff from 4 seconds, doubling per attempt to a 300-second cap. These are real outages, where waiting helps.
- **Attempt limit.** Up to 30 attempts per request (`max_retries = 29` in `core/config.py`). Backoff state belongs to the request, not the section, so one bad page does not slow the requests after it.
- **Sections.** A section is one phase of the build (focal metadata, focal references, focal citations, recommendations, keyword search, bibliography-recovery metadata, selected-paper metadata, reference hydration). A request that exhausts its attempts marks its section exhausted, and later uncached requests in that section stop at once instead of each burning 30 attempts. Other sections are unaffected.
- **Recovery time limit.** `--s2-retry-budget SECONDS` caps recovery time per section. The default `0` leaves only the attempt limit.
- **Fewer requests.** Reference hydration fetches up to 500 papers per POST to `/paper/batch` and falls back to per-paper pagination only for papers whose batch row is incomplete (`len(references) != referenceCount`). Cached lists are reused unless a refresh is requested.

With a 62% rejection rate, a request needs about 2.6 attempts on average, about 3 seconds at keyed pacing. The chance that all 30 attempts are rejected is under one in a million.

## What it replaced

The earlier policy backed off on every failure, 429 included, from 4 seconds doubling to 300 seconds, and counted retries across the whole section. Against random 429s that meant each section's later requests started deep in the backoff schedule, waiting minutes for a draw that had the same odds as one taken a second later. Builds of a paper with no cache took up to an hour.

Cold hybrid builds of DPO (`arxiv:2305.18290`), with the paper and reference caches refreshed, on 2026-11-22/23:

| Policy | Access | Wall time | Requests | 429s |
| --- | --- | --- | --- | --- |
| Section-wide escalating backoff | Keyed | 25m51s | 26 | 19 |
| Section-wide escalating backoff | Anonymous | 1m12s | 11 | 4 |
| Retry 429 at pacing slot | Keyed | 58.8s | 18 | 11 |
| Retry 429 at pacing slot | Anonymous | 47.3s | 7 | 0 |

All four builds produced the same node and edge sets. Keyed and anonymous runs differ in request count because each run hit a different mix of 429s, cached recommendation fallbacks, and pagination. The anonymous rows ran at quieter times, so they do not show that anonymous access is faster. Per request, the probe found keyed access less likely to be rejected.

A retry library such as tenacity would not have changed this: the problem was the schedule, not the retry mechanics.

## Re-measuring

S2's behavior changes, so measure again before changing the policy. For a cold build without deleting the cache, add `--refresh-paper-cache --refresh-reference-cache`. For anonymous access, set `S2_API_KEY=""` for that command. `--log-level debug --log-file run.log` records every response's status and elapsed time, and whether the authentication header was sent.

Keep these rules unless new measurements contradict them:

- Do not back off exponentially on a 429 without `Retry-After`, unless measurements show the 429 share falling with longer waits.
- Do not share retry or backoff counts across requests. Only exhaustion is shared, within one section.
- Do not raise the request rate above S2's published per-key limit to make up for rejected requests.
