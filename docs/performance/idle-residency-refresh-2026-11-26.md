# First request after idle: keeping the wired weights resident

**Date:** 2026-11-26 · **Hosts:** MacBook Pro M5 Max 128 GB and Mac mini
M4 Pro 64 GB, macOS 27.2, pinned MLX 0.32.4 · **Packs:** Qwen 3.8 Flash
Next AXQ MXFP4, Qwen 3.8 27B AXQ MXFP4 · **Peer:** oMLX 0.8.2 · **Raw
runs:**
[`benchmarks/results/qwen38-m5-modes-20261125/first-text-after-idle.jsonl`](../../benchmarks/results/qwen38-m5-modes-20261125/README.md#first-text-after-the-server-sat-idle)

## Summary

A request that reached an AX Engine server after a few seconds without
work was slower than the same request sent back to back: 1.2 to 1.5 s
against 0.70 s for a 1,492-token Flash Next prompt on the M5 Max. oMLX
showed no such gap, so on the first request after a pause AX Engine was
0.4 to 0.8 s behind it. The server now asks MLX to re-request the GPU
residency of the wired weights once a second, and the same request takes
0.73 to 0.74 s after pauses of 3 to 60 seconds. Arithmetic is unaffected.

## What happened

AX Engine wires the model weights at startup (`mlx_set_wired_limit`). MLX
keeps the wired allocations in Metal residency sets with a standing
residency request. macOS drops that residency after a few seconds
without GPU work although the request stands, and the next evaluation
waits for the weights to become resident again. The MLX source says so
where it re-requests residency
(`mlx/backend/metal/resident.cpp`), behind
`MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS`, which is off by default and is
present in the pinned 0.32.4 wheel.

The cost does not depend on the prompt: it is the same 0.5 to 0.9 s for a
1,492-token and a 4,443-token Flash Next request, it starts between 1 and
3 seconds of idle, and it is complete by about 10. It is not MLX's buffer
cache: skipping every `clear_cache` call left it unchanged. The paired
benchmarks never saw it, because their probes follow the warmup requests
without a pause.

## What changed

`AX_MLX_RESIDENCY_REFRESH` (default on) sets
`MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS=1000` before MLX creates its
Metal device. `AX_MLX_RESIDENCY_REFRESH=0` leaves MLX's default. A value
of the MLX variable already in the environment wins over the AX switch in
both directions: `0` there keeps the refresh off, and a positive value
keeps it on even with `AX_MLX_RESIDENCY_REFRESH=0`.
Intervals of 250 and 1,000 ms remove the penalty equally. An idle server
with the refresh on used no measurable CPU time over 60 seconds.

## Measurements

First text in milliseconds, one request after each pause. One run per
configuration.

MacBook Pro M5 Max 128 GB, 1,492 input tokens:

| Pause before the request | 0 s | 3 s | 10 s | 30 s | 60 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash Next, refresh on (default) | 695 to 697 | 733 | 739 | 734 | 743 |
| Flash Next, `AX_MLX_RESIDENCY_REFRESH=0` | 694 to 698 | 1,161 | 1,447 | 1,507 | 1,501 |
| Flash Next, oMLX 0.8.2 | 699 to 701 | 741 | 744 | 749 | 749 |
| 27B, refresh on (default) | 1,496 to 1,500 | 1,541 | 1,542 | 1,551 | 1,547 |
| 27B, `AX_MLX_RESIDENCY_REFRESH=0` | 1,497 to 1,537 | 1,657 | 1,695 | 1,704 | 1,734 |
| 27B, oMLX 0.8.2 | 1,507 to 1,509 | 1,540 | 1,560 | 1,541 | 1,543 |

The 35 to 50 ms that remain after a pause are common to both engines.

Mac mini M4 Pro 64 GB, 27B, 413 input tokens:

| Pause before the request | 0 s | 3 s | 10 s | 30 s | 60 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Refresh on (default) | 3,145 to 3,148 | 3,189 | 3,191 | 3,187 | 3,188 |
| `AX_MLX_RESIDENCY_REFRESH=0` | 3,144 to 3,148 | 3,365 | 3,391 | 3,426 | 3,409 |

With `/v1/generate` and token input on the M5 Max, a 4,443-token Flash
Next request takes 2.13 s back to back, 2.99 s after 20 seconds without
the refresh and 2.17 s with it.

## Limits

- macOS drops residency for a reason when memory is short. With the
  refresh on, MLX asks for it again every second, for every residency set
  of the process and not only the weights. On a host under memory
  pressure that works against the system's reclaim and can cost other
  applications memory or add swapping. No such host was measured;
  `AX_MLX_RESIDENCY_REFRESH=0` restores the previous behaviour there.
- Measured on macOS 27.2 with two hosts and two packs.

## Reproduce

Start a server, send three warmup completions, then one completion with
`max_tokens` 1 after each pause, each with a first line no earlier request
used so that no stored prefix applies. Compare with
`AX_MLX_RESIDENCY_REFRESH=0`.
