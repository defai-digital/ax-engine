# Selected Flash Next weight materialization on M2 Ultra

These are selected-tensor measurements from the pinned Flash Next MXFP4
and MXFP8 packs on a Mac Studio M2 Ultra 192 GB with macOS 27.2.1, an
external APFS SSD, Rust 1.99.1 and MLX 0.32.4. Each process opened one
of three real shards per format and materialized approximately 1.50 GiB
of MXFP4 or 1.08 GiB of MXFP8 in 68 to 139 selected tensors. All 72
observations produced the same per-shard digest of selected names,
dtypes, shapes and bytes across loaders and cache conditions.

| Format | Selected page state | Name-order mmap | Offset-order mmap | MLX lazy CPU |
| --- | --- | ---: | ---: | ---: |
| MXFP4 | Fully resident | 249.5 ms | 248.2 ms | 128.7 ms |
| MXFP4 | Fresh APFS clones | 14.474 s | 10.266 s | 2.207 s |
| MXFP8 | Fully resident | 189.3 ms | 190.6 ms | 99.5 ms |
| MXFP8 | Fresh APFS clones | 6.179 s | 6.291 s | 0.509 s |

The table reports medians for five fully resident observations and six
fresh-clone observations per loader and format. Every clone started
with at most one selected OS page resident. The APFS clones share the
same physical SSD, whose device cache was uncontrolled. Each raw
original-file series also includes one first-touch observation per
loader with zero or one page resident. Cache classification uses the
recorded `mincore` samples.

For fresh clones, median process RSS sampled immediately after eval
was 3.016/3.016/1.517 GiB on MXFP4 and 2.189/2.189/1.103 GiB on MXFP8
in table order. RSS includes mapped file pages. The timed window ends
after evaluation and before hashing. The sample does not report
private allocation, peak serving memory or full-model startup time.
MXFP8 remains a pack preview. The M5 full-model pair is a separate
test.

## Partial materialization failure control

The [Rust probe](lazy_eval_failure.rs) uses the same MLX C binding and
library as the selected loader trial. It generates three local 1 MiB
fixtures, evaluates the first shard and truncates the second after
loading its descriptors. The [recorded result](lazy-eval-failure.json)
shows that `try_eval` returned an error. Both intact fixtures
subsequently evaluated in the same process. This control covers the
binding's error propagation. Server readiness, partial-load teardown
and process footprint need server-level verification.

## Reproduction

The [source receipt](source.json) binds the measurements, the exact
[selection](selection.json) and the Rust probe by SHA-256. The
[shared probe patch and runner](../flash-next-weight-loader-lab-20261121/README.md)
describe the build and trial commands. Restore each manifest entry's
file path from the pinned local pack, preserving the recorded file
order and tensor names. Use the same script for original files and for
fresh clones on the pack's APFS volume.

Copy the failure probe into `crates/mlx-sys/examples/lazy_eval_failure.rs`
in the scratch checkout and build `-p mlx-sys --release --example
lazy_eval_failure` with Rust 1.99.1. Pass it a new subdirectory of a
temporary directory to generate its own fixtures.

- MXFP4 observations [original](mxfp4-original.jsonl) and [fresh clones](mxfp4-fresh-clones.jsonl)
- MXFP8 observations [original](mxfp8-original.jsonl) and [fresh clones](mxfp8-fresh-clones.jsonl)
