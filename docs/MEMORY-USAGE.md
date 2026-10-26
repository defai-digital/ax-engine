# Memory usage

AX Engine uses unified memory for model weights, request state, KV/prefix
caches, compiled graphs, and reusable MLX buffers. A high number after a
request is not by itself proof of a leak. Measure the GUI and server processes
separately.

## What unload releases

`POST /v1/model/unload`, non-default model idle eviction, and successful Ollama
`keep_alive: 0` unloads drain the selected model and join its worker. No hidden
parked-model cache retains unloaded weights. The worker releases its compiled
cache entries without evicting sibling workers' graphs. Unused MLX allocator
buffers are trimmed after retirement. Live sibling tensors and any in-flight
external owners remain valid.

Reloading pays session/weight initialization again. Keep a model loaded for
fast reuse; unload it to reclaim memory. The final resident model cannot be
unloaded through this API: stop the server to release it. A client disconnect
does not cancel an already-started unload.

## Observe before tuning

The authenticated `GET /v1/memory` endpoint reports:

- `phys_footprint_bytes`: the server's current physical footprint.
- `mlx.active_bytes`: live MLX allocations.
- `mlx.cache_bytes`: cached-but-free MLX buffers.
- `mlx.peak_bytes` and `lifetime_max_phys_footprint_bytes`: historical high-water
  marks; these need not fall when a request completes.

These are overlapping views, not additive categories: do not add MLX active
and cache bytes to the process footprint. Native counters can be unavailable
on delegated builds. A disabled `AX_API_POLISH` makes the endpoint return 404.

Warm the chosen workload, then repeat a fixed prompt/output size and record
post-request samples. Exercise cancellation, disconnected streams, repeated
model load/unload, and idle periods separately. Distinguish a plateau caused
by bounded caches from continued growth of live allocations. Use Instruments
Allocations/Leaks and Metal Resource Events when counters show unexplained
growth; a unit-test pass does not establish a real-model leak-free soak.

## Reduce retained and peak memory

1. Unload unused models; configure `--model-idle-timeout-secs` for non-default
   models when appropriate.
2. Set `--max-concurrent-requests` and constrain client context/output lengths.
   These reduce simultaneous request/KV residency.
3. Compare smaller `--prefill-chunk` values on the actual model. Smaller chunks
   can reduce temporary allocations at a throughput/latency cost.
4. Set `AX_MLX_PREFIX_CACHE_MAX_BYTES` when the automatic prefix capacity is too
   large. This policy applies separately to portable and native prefix tiers;
   it is not a process-wide budget. Reducing it can reject long reusable
   prefixes and increase prefill work.
5. Tune `AX_MLX_CACHE_LIMIT` for reusable allocator buffers. Avoid making zero
   the blanket default: allocation-heavy workloads benefit from reuse.
6. Use `load_policy: "memory_constrained"` for replacement when the temporary
   two-model peak of `availability_first` is undesirable. Observe memory and
   recovery behavior on the actual workload.

Do not infer an enforced process limit from `AX_MLX_MEMORY_LIMIT`: it controls
MLX allocation policy, not all Rust, Swift, HTTP, or operating-system memory.
Keep expert streaming on its existing Auto policy unless deliberately testing
an override. Changing cache or prefill settings needs an A/B comparison of
footprint, first-token latency, and decode throughput.

The macOS Playground separately bounds conversation text, message count, and
archive size. Export/clear a long conversation to release its visible history.
Dashboard history is already bounded by age and sample count.

## Swift streaming receive limits

The Swift SDK used by Playground reads one SSE event per consumer demand and
pauses its URLSession task between reads. It no longer maintains an unbounded
decoded-event queue. SSE lines, event blocks, and streaming HTTP error bodies
have separate byte limits, with explicit errors and transport cancellation on
overflow. These complement Playground's retained-history limits; they do not
cap Foundation's own buffering or successful non-streaming response bodies.
See [Swift SDK streaming](sdk/swift.md#streaming) for defaults, overrides, and
cancellation behavior.
