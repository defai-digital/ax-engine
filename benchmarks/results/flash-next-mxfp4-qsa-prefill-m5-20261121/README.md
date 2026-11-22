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

## Matched fused-gather follow-up

The [real-pack state receipt](fused-index-pair/receipt.json) compares
separate and shared K/V indices inside the same fused gathered route.
The isolated `ae1aec6e9` probe used the same pinned model and software
on the hardware described above. It ran an 8,192-token prompt in
512-token chunks, starting each arm from fresh request state. All
final logits and serialized request state matched bitwise. Each arm
retained 864 fused takes. Sharing reduced explicit index tile creations
from 864 to 432. All 105 host pressure samples were normal.

| Order | Separate indices | Shared indices | Shared time change |
| --- | ---: | ---: | ---: |
| Separate, then shared | 9.496 s | 10.100 s | +6.36% |
| Shared, then separate | 9.937 s | 10.197 s | +2.62% |

Peak MLX allocation moved by about 4 MiB in opposite directions across
the two pairs. The change stays in the experimental branch. These
times cover prefill on one prompt inside an isolated Rust test. The
earlier numbers above measure client-observed TTFT on a different
attention route.

The [probe patch](fused-index-pair/source.patch.gz) applies to
`48065875512b8069fce51d9f1ab64b748a85f152`. Replay the receipt against
its test output, run metadata, and pressure samples from the repository
root.

```bash
python3 benchmarks/results/flash-next-mxfp4-qsa-prefill-m5-20261121/fused-index-pair/replay.py
```
