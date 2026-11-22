# Flash Next aligned cache-policy routing on M2 Ultra

Mac Studio M2 Ultra 192 GB, external SSD, Rust 1.99.1 and MLX and MLX
Metal 0.32.4. The same `23c5c8cb8` release-server executable serves a
fresh local APFS view of the pinned MXFP4 pack. The launcher matches
all 49 publisher files and the frozen runtime manifest before and
after the four processes.

| Process | Activated reader | Exact HTTP controls | MTP verifier steps |
| --- | --- | ---: | ---: |
| Default, both switches off | Mapped | 7 | 0 |
| Default, buffered switch | Aligned buffered | 7 | 0 |
| Default, both switches | Aligned no-cache | 7 | 0 |
| Required MTP, buffered switch | Aligned buffered | 7 | 107 |

The process traces confirm all 15 resident shards and 1,942 selected
tensors in each arm. The seven checks cover repeated and extended
text, JSON, SSE, the request following SSE, and one-token completion.
Choices, stops, token accounting and activation counters match the
[prior mapped controls](../flash-next-aligned-native-http-m2-20261122/README.md).
Pressure reports normal around each process. The closed software
receipt records 4,790 Rust and 345 Python passes with no failures.

The compiler's debug-strip step initially could not load its LLVM
library. The staged server reapplies the pinned `--strip-debug` with
the correct library search path, verifies its code signature, and
matches the original instruction section. The exact staged digest
is used for all four processes. These are functional observations
with uncontrolled file-cache history. The M5 crossover measures
whether buffered and no-cache modes differ in serving performance.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
