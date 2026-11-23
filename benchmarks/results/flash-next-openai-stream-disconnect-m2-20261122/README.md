# Flash Next OpenAI streaming disconnect and capacity

A Mac Studio M2 Ultra 192 GB used the pinned resident MXFP4 pack on
external USB SSD with macOS 27.2.1 and MLX and MLX Metal 0.32.4.
Both builds use Rust 1.99.1. The retry adds the gauge correction
from runtime commit `5a21840d7` to the exact guarded source
`8cb06414c`. The packing and scheduling inputs match the
[guarded M5 core and HTTP control](../flash-next-native-mixed-admission-m5-20261122/README.md).

The original OpenAI test observes two live completion streams and
one scheduler wait. A third request through the chat endpoint is
rejected. After the first client closes, the admission count falls
to one and the worker stream count reads two at the deadline.
The deterministic regression reproduces two versus one during the
survivor's engine step. Publishing the stream count during removal
makes the focused test and all 34 service tests pass.

The retry again observes one scheduler wait and receives HTTP 429
with `concurrency_limit` through chat. The first client closes
after its first text event. The next observed admission and worker
counts are both one. The deferred completion returns the solo text
and usage for 8,192 prompt tokens and 96 generated tokens. The
replacement chat stream also matches its solo text, finish and
usage. Both finish their SSE streams and all five work counters
drain. All 116 memory-pressure readings are normal.

The live API test reads an aggregate scheduler gauge. The M5
stepwise control supplies the exact per-request waiting state and
cursor evidence. These tests cover functional capacity and
disconnect recovery in the experimental resident two-slot profile.

The archive retains the red observation, focused test result,
exact patch, signed build receipts, input hashes, and raw retry
receipt. Replay binds both driver versions to the same frozen inputs
and checks the reply, admission, wait and stream-drain observations.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
