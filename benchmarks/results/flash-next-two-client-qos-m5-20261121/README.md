# Flash Next request-level streaming baseline on M5 Max

This analysis replays the archived reservation schedule and its boundary
holdouts on MacBook Pro M5 Max 128 GB. The measured prompts contain
2,048, 2,049, 2,050 or 8,192 tokens. Each pair generated 96 tokens
per request and matched the corresponding solo output IDs. Each
arrival pattern has two trials. In staggered trials, both prompts
served once as the active decoder and once as the arriving request.

| Prompt tokens | Two-slot serial, output tok/s | Staggered pair, output tok/s | Existing decoder, longest event gap | Arriving request, longest event gap |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 21.17 | 21.42 | 2.215–2.225 s | 0.052–0.053 s |
| 2,049 | 22.50 | 22.44 | 2.037–2.041 s | 0.050–0.051 s |
| 2,050 | 24.01 | 24.11 | 1.855–1.858 s | 0.047–0.049 s |
| 8,192 | 6.99 | 7.40 | 2.905–3.054 s | 0.050–0.058 s |

The output-token rate includes wall time for both prompts and replies.
The two-slot serial baseline uses the same server profile as the
staggered pair. Each rate is the median of the two trials. The gaps
are per-client maxima between non-empty SSE events, shown as the range
over both prompt roles. The measured aggregate rate can stay close
to serial while the already active decoder pauses for seconds.

These are bounded reservation controls. The physical-prefill split
experiment and a sustained multi-client workload have separate
evidence. The [analysis](summary.json) includes all 32 trials, each
request's TTFT and longest gap, and the exact compressed input digests.
