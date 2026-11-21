# Flash Next MXFP4 full native loader comparison on M2 Ultra

One fresh native process per loader ran on a Mac Studio M2 Ultra 192 GB
with macOS 27.2.1, an external APFS SSD, Rust 1.99.1 and MLX 0.32.4. Both
arms used the same experimental server binary and exact MXFP4 pack at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. The harness verified all 49
publisher files by size and SHA-256 before startup, then gave each
arm separate APFS clones. It ran resident with a single request slot,
32K context and MTP disabled.

| Measurement | Name-order mmap | MLX lazy CPU |
| --- | ---: | ---: |
| Process spawn to ready | 1,467.24 s | 193.98 s |
| Resident load and evaluation | 1,377.85 s | 80.34 s |
| Peak sampled startup RSS | 71.060 GiB | 73.641 GiB |
| Startup resource observations | 91 | 13 |
| Loaded shards / tensors | 15 / 1,942 | 15 / 1,942 |

The lazy process reached readiness 7.56 times as fast on this pair.
Both emitted the same 96 greedy token IDs across 512-, 2,048- and
8,192-token prompts, with zero MTP activation or route errors. System
memory pressure stayed at level 1 in both series. The measurements
sample RSS at roughly 15-second intervals during startup. They exclude
inference memory peaks and the time spent verifying and cloning the
publisher files. This pair observed a higher RSS peak with lazy CPU.
The [phase analysis](startup-phase-rss.json) locates both peaks after
resident weight evaluation and before the HTTP listener. Samples whose
one-second UTC interval overlaps a phase marker are unclassified. The
sampled RSS identifies the phase of the higher peak. Allocator and
kernel footprint accounting are separate measurements.

The first arm used mmap. The APFS files share the same underlying
storage, and their controller cache and full-model OS page state were
uncontrolled. This pair establishes native output identity and a
readiness improvement on the tested M2 setup. Primary-tier results on
the M5 and the separate startup failure control still inform any
default decision.

- [Raw inference and startup results](result.json)
- Startup samples [mmap](name-order-samples.jsonl) and [lazy CPU](lazy-cpu-samples.jsonl)
- [Exact pack member inventory](pack-inventory.json) and [source receipt](source.json)

Apply the [lab patch](server-loader-lab.patch) to the source commit in
the receipt with `git apply --unidiff-zero` in a scratch checkout. Build package `ax-engine-server` with the recorded
`release-server` profile. The experimental reader switch is
`AX_FLASH_NEXT_RESIDENT_LOAD_LAB=lazy-cpu`; use `name-order` for the
comparison. For the controls, disable prefix reuse and n-gram
acceleration, set a 2,048-block context budget and run the fixed token-ID
workloads from `scripts/run_flash_next_generate_matrix.py` with 32
generated tokens each. Obtain the model directory from the recorded
publisher revision and verify the manifest with the native toolchain.
