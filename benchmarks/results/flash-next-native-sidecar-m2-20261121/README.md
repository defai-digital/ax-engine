# Native Flash Next MXFP4 sidecar reader comparison

On a Mac Studio M2 Ultra 192 GB with macOS 27.2.1, MLX 0.32.4,
Rust 1.99.1 and an external APFS SSD, four native server processes
compared Flash Next MTP sidecar readers. The candidate keeps the head
attached under both tested policies. The same experimental MLX lazy
CPU trunk path loaded 15 shards and 1,942 selected tensors in each arm.
All 49 published members were verified before and after the experiment.
Each arm launched from a separate clone of the pinned pack on the same SSD.

| Policy | Sidecar reader | Spawn to ready | Trunk load | Sidecar load | Peak sampled startup RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| Disabled | Name-order mmap | 185.851 s | 28.482 s | 135.113 s | 73.349 GiB |
| Disabled | MLX lazy CPU | 81.001 s | 63.007 s | 2.164 s | 69.087 GiB |
| Required | MLX lazy CPU | 101.194 s | 81.604 s | 2.182 s | 65.923 GiB |
| Required | Name-order mmap | 135.557 s | 89.962 s | 27.816 s | 71.859 GiB |

Each policy matched all 96 greedy IDs between readers across 512-,
2,048- and 8,192-token inputs. The disabled requests saw an available
head and performed zero verifier steps. The required requests had
56 verified steps and 37 accepted draft tokens in each arm. All four
arms had zero route errors and normal pressure in the startup samples.
The observations sample startup at roughly 15-second intervals.

Reader order reversed between policies. The controller cache and
per-file page state were not controlled in these full native runs.
The identical trunk route varied from 28.482 to 89.962 seconds.
The readiness measurements are singles for each policy and reader.
The [counterbalanced sidecar-only experiment](../flash-next-mtp-sidecar-reader-m2-20261121/README.md)
quantifies the same reader change at known OS-page residency. The
results here cover this M2 setup and the tested one-request envelope.

## Reproduction

The [build and run receipt](result.json) binds the binary, source base,
[patch](server-sidecar-lab.patch.gz), prefill contract and MTP counters.
The [source receipt](source.json) also identifies the publisher
inventory and the startup samples. Decompress and apply the patch to the
recorded source base in a scratch checkout and build package `ax-engine-server` with
the `release-server` profile. Use the exact MXFP4 revision and
`AX_FLASH_NEXT_RESIDENT_LOAD_LAB=lazy-cpu` in every arm. Select the
sidecar reader via `AX_FLASH_NEXT_SIDECAR_LOAD_LAB`. Run resident,
one request slot, 2,048 context blocks, prefix reuse disabled and
generic n-gram disabled. Use the canonical verifier and the fixed
token-ID workloads in `scripts/run_flash_next_generate_matrix.py` with
32 generated tokens and ignore EOS. Clone the pack separately for each run.
