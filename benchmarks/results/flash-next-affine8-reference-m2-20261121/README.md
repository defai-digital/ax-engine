# Flash Next same-host bounded PLE comparison

The [diagnostic](../../../scripts/run_flash_next_affine8_reference.py)
ran on Apple M2 Ultra 192 GB with macOS 27.2.1 and an external SSD. Each
arm used a fresh process, the same `de3463474` source, MLX and MLX Metal
0.32.2, and MLX-VLM `6102cb4ad`. The pack revision was
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. Both arms passed publisher
member and runtime checks before and after exercising the model.

The [paired adjudication](comparison.json) compares complete greedy
outputs on the same 512-, 2,048- and 8,192-token windows. All 96 output
IDs matched. The bounded arm materialized 384 fewer arrays and read
31,241,700 bytes from the PLE table for this workload. The comparison
uses the existing MLX quantized embedding arithmetic on Metal.

| Metric | Resident | Bounded PLE |
| --- | ---: | ---: |
| Peak MLX tracked memory | 120.014 GiB | 66.368 GiB |
| Peak process RSS | 99.263 GiB | 63.997 GiB |
| 512-token case total | 11.023 s | 20.749 s |
| 2,048-token case total | 11.505 s | 64.045 s |
| 8,192-token case total | 28.942 s | 64.515 s |

[Resident observations](resident.json) and [bounded observations](bounded.json)
retain all input hashes and output IDs. The [resident memory samples](resident-memory.jsonl.gz)
and [bounded memory samples](bounded-memory.jsonl.gz) record normal system
memory pressure throughout both arms. The [source receipt](source.json)
binds the diagnostic, collector and inventory. The row storage change
trades latency for memory within these diagnostic cases. The M5 smoke,
broader QA and the frozen release comparison have separate evidence.
