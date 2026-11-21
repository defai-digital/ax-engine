# Flash Next fixed-width prefill interleave on M5 Max

MacBook Pro M5 Max 128 GB, macOS 27.2.1 and internal SSD. The test
executable at `c755bbebb` was built with Rust 1.99.1 and ran against
MLX 0.32.4. It loaded resident experts from the pinned MXFP4 pack at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. The harness checked the
publisher inventory before and after both prompt orders.

The [two archived prompts](../flash-next-mxfp4-prefill-partition-m5-20261121/prompts.jsonl.gz)
each contain 2,048 token IDs. Each prompt served as the primary in one
fresh process and as the sibling in the other. The primary's deferred
baseline used physical widths of 512, 512, 512 and 511 for cache-only
prefill, followed by the final prompt token through singleton decode.
The treatments materialized each of those same prefix forwards
independently, either alone or with a sibling decode between each
pair of primary forwards and before the final singleton.

Both treatment arms in both prompt orders matched all eight primary
greedy output IDs, the serialized state after prefill and the serialized
state after continuation. The sibling's tokens and final serialized
state matched its independent sequential baseline. The debug logs
record the physical widths in the deferred and materialized arms.

The [receipt](result.json) binds the test executable, inventory, model
manifest and prompt bytes. The [first order](prompt-0.log.gz) and
[reversed order](prompt-1.log.gz) retain the complete diagnostic traces.

## Served-path geometry

The `b36b7b287` release-server binary repeated the same two 2,048-token
prompts. It served four requests with one slot and twelve with two
slots on the 2,049-token reservation schedule. Every request reproduced
all 96 solo output IDs. The trace recorded physical prefix widths
512, 512, 512 and 511 and a final singleton for all 16 requests.

The [forward trace](serving-prefill-forwards.json) contains the physical
forwards after readiness and the hashes of the source logs. The
[serving result](serving-result.json.gz) retains the client output and
scheduler events. Measuring the latency of split scheduler turns
requires a separate run.

## One- and two-token tails

The `a6fc6eee8` Rust 1.99.1 diagnostic repeated the state comparison
with the archived 2,049- and 2,050-token prompt pairs in both orders.
For each input, the baseline used a legacy asynchronous 2,048-token
cache-only item and then the ordinary prompt-completing tail. The
treatment materialized the same four prefix forwards and the
cache-only singleton, with a sibling decode between turns in the
interleaved arm. The remaining one or two prompt tokens kept the
legacy tail route.

Both treatments matched all eight primary output IDs and serialized
prefill and continuation states in all four trials. Sibling tokens and
state matched the separate sequential control. The
[tail receipt](tail-result.json) binds the test executable, pinned
inventory, source prompts and all four debug logs.

## First split-serving activation attempt

The `91ca25c7f` release-server ran with an explicit two-request limit
and the split experiment enabled. The resident memory plan priced both
slots and reported `live-sequence-cap=uncapped`. The experiment had
treated that optional memory throttle as the configured request limit,
so all six candidate observations kept a single 2,048-token scheduler
turn. All six output arrays matched their solo controls. The prefill
disconnect probe reached stream completion without receiving an
intermediate prefill event. The [partial receipt](split-activation-attempt.json)
retains the inputs, outputs and startup decision for that failed
activation attempt. Latency and cancellation still require a run that
activates the split path.

## SSE split-boundary negative control

The [2,048-token follow-up](sse-cut-negative-control/README.md) activated
the physical split, matched all 28 output arrays, and completed six
disconnect and replacement trials. The original route audit rejected
the receipt because the logical boundary was absent from the SSE
progress frame. The linked receipt independently verifies the
physical work and preserves the audit failure.

The [corrected 2,048-token replay](sse-cut-2048/README.md) passed all
28 exact output controls and six disconnect/replacement controls.
Every physical piece of each split request carried its logical boundary.
The [2,049-token holdout](sse-cut-2049/README.md) passed the same
controls with a one-token tail after the logical boundary.
The [2,050-token holdout](sse-cut-2050/README.md) passed the same
controls with a two-token tail.
