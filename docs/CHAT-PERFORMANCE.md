# Chat speed measurements

AX Engine for macOS shows **Prefill** and **Decode** beneath each assistant reply.
Both are native evaluation averages in tokens per second, saved with the reply.

- **Prefill:** uncached input tokens divided by native prompt evaluation time.
  Physical prefix-cache reuse is excluded from the token count. A fully reused
  prompt displays `Cached`; unknown cache accounting does not produce a rate.
- **Decode:** native output evaluation tokens divided by native output evaluation
  time. The initial autoregressive output batch belongs to the prompt phase and
  is excluded from both decode counters. Reasoning and accepted speculative
  output are already covered by the runtime's committed-token accounting.
- **First text** and **Elapsed** remain client-observed latency. They include
  effects such as waiting and transport and are not throughput denominators.

While generation is running, rates are marked as measuring. Final native timing
arrives in the terminal usage event. A dash means unavailable, not zero speed.
Older runtimes, delegated backends without native timings, block-diffusion models,
failed/stopped replies and very short replies without a decode interval may not
provide both rates. Existing saved conversations continue to open without metrics.

The GUI uses `stream_options: {"include_usage": true}`. On native autoregressive
streams, the final usage-only SSE chunk can include this additive extension:

```json
{
  "choices": [],
  "usage": {
    "prompt_tokens": 1000,
    "completion_tokens": 83,
    "total_tokens": 1083,
    "ax_engine": {
      "metrics_version": 1,
      "prefill_tokens": 200,
      "prefill_time_us": 400000,
      "decode_tokens": 80,
      "decode_time_us": 2000000
    }
  }
}
```

This example reports 500 prefill tok/s and 40 decode tok/s. Its first three output
tokens are outside the decode measurement. Counters are per request, not global
server metrics. The timings come from the runtime's existing version-2 native
evaluation report and exclude consumer/network/display pacing. They include native
engine work; they are not pure GPU-kernel benchmarks or hardware certification.
The optional extension is absent when the backend cannot provide these timings.
A live rolling rate requires native progress measurements and is not estimated
from text length or SSE event arrival frequency.
