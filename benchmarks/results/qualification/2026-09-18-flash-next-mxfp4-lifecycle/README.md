# Flash Next installed lifecycle failures

Result: **failed; product qualification remains open**.

Both attempts used numerical/server source `bf06cbb2`, the installed candidate
wheel and the M5 Max 128 GiB MXFP4/NAS target. They are separate failed runs.

| Attempt | Observed result |
| --- | --- |
| Original comparator | Two complete responses have identical text, finish reason and token totals 46/32/78. Warm SSE adds cached_tokens=32; comparing the whole usage object incorrectly rejects it. Twelve remaining requests never ran. |
| Corrected cache comparator | Baseline, SSE, budgets 1/2 and stop pass. The disconnect probe receives its first token, then observes active_streams=0 while pending_jobs=1 and jobs_in_flight=1. The strict active-producer assertion fails. Drain/recovery and required MTP never run. |

Both owned workers/servers exit cleanly. Neither run reaches its original
payload postcheck. Separate later read-only followups each verify all 47 model
payloads and 52 installed runtime files; those checks do not replace the missing
original postchecks or change either lifecycle verdict.

The second failure exposed a server metric publication defect: active-stream
count was published only at the end of a worker tick, after its potentially long
first decode burst. The fix publishes stream ownership before acknowledging
startup, and republishes removal if the startup receiver has disconnected. A
local deterministic backend regression fails before the fix and passes after it.
Its first invocation used an incorrect exact filter and ran zero tests; that log
is retained and is not passing evidence. New-wheel M5 requalification is pending.

The corrected cache comparator preserves raw usage and validates cached tokens
as an integer prompt subset while comparing exact text, finish reason and all
three token totals. The active-producer disconnect requirement remains intact.
No token budget, timeout, model arithmetic or numerical tolerance was changed.

[lifecycle-failures.json.gz](lifecycle-failures.json.gz) retains complete failed
terminal records, source contracts, transfer receipts, logs and separate
integrity followups. Private locations are replaced with stable hash labels;
numeric/boolean/null values and failed verdicts are preserved. Gzip timestamp is
zero. Artifact identity and scope are in [summary.json](summary.json).
