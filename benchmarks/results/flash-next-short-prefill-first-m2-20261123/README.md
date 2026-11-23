# Flash Next bounded short-prefill priority on M2 Ultra

Mac Studio, Apple M2 Ultra, 192 GiB unified memory, internal SSD, macOS
27.2.1, and MLX and MLX Metal 0.32.4. The signed server builds from
`927584c64` and the [recorded patch](source.patch.gz) with Rust 1.99.1.
Both arms use the same binary, the mapped reader, and the pinned MXFP4
pack `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. The driver verifies all
49 publisher files and the derived manifest before and after both
arms. The reserved two-slot profile starts with MTP, native prefix
reuse, and direct coalescing disabled.

TensorFold's whole-short-prefill turn and aged-long service rule
motivated the isolation of prefill ordering. With the switch on, the
native HTTP test gives the 512-token prompt one complete prefill turn
after its older 8,192-token peer completes 2,048 prompt tokens. On that
step the short request emits its first token and the long cursor
stays at 2,048. The FIFO control advances the long cursor to 4,096
while the short cursor stays at zero.

A second run cancels the short stream immediately after its first
token. Its replacement gets a fresh request ID. The next step
advances the older long request to 4,096 before the replacement can
prefill. The three completed pairs match 576 output IDs against
their own single-request outputs, and the cancelled prefix matches
its first solo token. All work counters drain, all 189 pressure
samples are normal, and peak physical footprint is 85.94 GiB.

This is a functional control for the bounded opt-in. The M5 comparison
still needs offered first-token latency, client decode-gap, and total
burst completion measurements.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
