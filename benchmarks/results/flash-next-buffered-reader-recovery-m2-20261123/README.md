# Flash Next aligned-buffered startup failure and restart on M2 Ultra

The signed candidate `d2220c26f` runs the pinned MXFP4 pack on a Mac Studio
M2 Ultra 192 GB with macOS 27.2.1, an internal SSD, Rust 1.99.1, and
MLX and MLX Metal 0.32.4. Both processes explicitly select the aligned
buffered reader, with one active request, a 32K context and MTP off.

The [existing test-only injector](../flash-next-native-loader-failure-m2-20261121/shard_read_fault.c)
returns `EIO` for a payload read from shard 12 after the server completed
shard 1. The signed native server reports the read failure and exits
with status 1 before opening HTTP. The second process uses the same
server, unchanged files, and the same port with the injector removed.

The restarted service matches 128 greedy output token IDs from the
[sealed M2 control workload](../flash-next-natural-document-prefix-http-m2-20261123/README.md).
It reports healthy, drains all five counters, and records zero MTP
verification and route errors. All nine sampled pressure readings are
normal. The harness checks all 49 publisher members before the fault
and after recovery.

This covers startup failure and restart with the experimental reader.
M5 release qualification retains its own checks.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
