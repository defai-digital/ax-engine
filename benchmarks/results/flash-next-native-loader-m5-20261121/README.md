# Flash Next MXFP4 native loader comparison on M5 Max

The [native loader lab](../flash-next-native-loader-m2-20261121/README.md)
ran its same executable in two name-order mmap and two MLX lazy CPU arms
on a MacBook Pro M5 Max with 128 GB unified memory, an internal SSD,
macOS 27.2.1, Rust 1.99.1 and MLX 0.32.4. The server used the exact MXFP4
revision `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` in the 32K resident
one-slot profile with MTP disabled. Each arm used a separate APFS clone.
Three selected file ranges showed one resident page each immediately
before launch. Application-level samples cover selected ranges. Device
cache state and unsampled pages can vary across starts.

| Arm | Spawn to ready | Resident load and evaluation | Peak sampled RSS |
| --- | ---: | ---: | ---: |
| Name-order mmap 1 | 78.22 s | 56.82 s | 74.03 GiB |
| MLX lazy CPU 1 | 20.05 s | 5.57 s | 66.99 GiB |
| MLX lazy CPU 2 | 20.06 s | 5.31 s | 67.26 GiB |
| Name-order mmap 2 | 40.16 s | 27.15 s | 74.28 GiB |

The slower lazy start reached readiness in 49.94% of the faster mmap
start. Each arm matched all 128 output IDs over the 512-, 2,048-, 8,192-
and 32,736-token prompts. Readiness and post-request health passed.
The 32,769-token overflow case returned the expected
`context_length_exceeded` error. Sampled pressure stayed normal.

The experimental reader exceeded the timing regression band. In the
paired summary, first-token latency increased by 21.51% for 2,048
tokens and by 16.19% for 8,192 tokens. The 2,048-token case's decode
time increased by 13.95%. The data supports retaining the reader as a
lab option pending a repeat over more prompts.

The [raw result](result.json) binds both the executable and its source
to the earlier lab. The startup records for
[mmap 1](name-1-samples.jsonl.gz), [lazy CPU 1](lazy-1-samples.jsonl.gz),
[lazy CPU 2](lazy-2-samples.jsonl.gz) and [mmap 2](name-2-samples.jsonl.gz)
retain all resource observations.
