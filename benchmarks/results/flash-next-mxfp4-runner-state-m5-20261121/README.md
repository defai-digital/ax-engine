# Flash Next MTP runner-state controls on M5 Max

The pinned MXFP4 pack at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`
ran on MacBook Pro M5 Max 128 GB, macOS 27.2.1 and internal SSD.
The `b36b7b287` Rust 1.99.1 test build linked MLX 0.32.4. It used
the canonical singleton target schedule, diagnostic four-token KV
blocks, and an eight-token greedy output budget. The test bypassed
MTP cost calibration with `AX_FLASH_NEXT_MTP_COST=0` and disabled
the disk prefix cache.

| Prompt tokens | Matching output IDs | Verified MTP steps | Accepted MTP steps | Step errors | Byte-identical aligned positions |
| ---: | ---: | ---: | ---: | ---: | --- |
| 46 | 8/8 | 4 | 3 | 0 | 48, 50, 52 |
| 2,048 | 8/8 | 5 | 2 | 0 | 2,050, 2,051, 2,052, 2,054 |

The ordinary singleton continuation produced matching token IDs,
byte-identical trunk state and byte-identical F32 logits in both runs.
The 46-token prompt repeats the
[November 18 control](../qualification/2026-11-18-flash-next-mxfp4-runner-state/README.md).
The 2,048-token prompt extends this control to a larger prefill, and
all prompt and output IDs are in the receipts.

The [short](short.json.gz) and [2K](long.json.gz) receipts bind the
retained local state files by SHA-256. The
[short reader](short-reader.json) and [2K reader](long-reader.json)
validated those files on the M5. The [provenance](provenance.json)
binds the diagnostic binary, reader, pack and curated artifacts.

These measurements cover two bounded, greedy singleton continuations.
Private draft tensor safety, general quality, performance and MTP-S/P/D
remain open.
