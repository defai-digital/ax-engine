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
The served-path trace and client latency comparison remain separate.
