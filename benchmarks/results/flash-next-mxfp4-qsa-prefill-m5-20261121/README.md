# Flash Next QSA prefill on M5 Max, 2026-11-21

MacBook Pro M5 Max 128 GB, macOS 27.2.1, internal SSD, Rust 1.99.1 and
MLX 0.32.4. The `release-server` executable is from `b36b7b287`, and the
[raw result](result.json) binds its hash and the inventory of the pinned
MXFP4 MTP pack. Both arms ran with resident experts, one active request,
a 32,768-token pool and the default MTP policy. Both recorded zero MTP
verifier steps and zero route errors. Each process warmed its kernels at
load. The gathered process ran first. Each fixed-length row had one
warm-up and two measured requests of 64 output tokens.

| Prompt tokens | Gathered TTFT | Indexed TTFT | Indexed TTFT change | Identical measured token sequences | First differences, zero-based |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2,048 | 1.647 s | 1.898 s | +15.2% | 2/2 | None |
| 8,192 | 9.476 s | 8.924 s | −5.8% | 0/2 | 7, 5 |
| 16,384 | 21.071 s | 19.009 s | −9.8% | 0/2 | 1, 1 |

TTFT is the median client-observed first-token latency. All six short
natural-language trajectories matched. The 2,048-token prompt fits the
dense prefill selection budget, and the arm order leaves process variation
in this timing. The long-prompt cells show a latency opportunity and
different greedy token sequences. The exact preset continues to use
gathered prefill. Indexed prefill remains available through the fast preset
or `AX_MLX_FLASH_NEXT_QSA_INDEXED_PREFILL=1`. The fixed workloads are a
bounded performance and output-identity check.
