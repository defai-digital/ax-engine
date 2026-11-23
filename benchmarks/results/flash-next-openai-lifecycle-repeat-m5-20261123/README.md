# Flash Next guarded OpenAI lifecycle on M5 Max

One MacBook Pro M5 Max 128 GB with internal SSD ran the [same signed guarded
build and stream-gauge fix](../flash-next-openai-stream-disconnect-m2-20261122/README.md)
as the earlier M2 control. The runner uses MLX and MLX Metal 0.32.4, Rust 1.99.1,
macOS 27.2.1, and the pinned resident MXFP4 pack. The guarded profile admits two requests.
When both prompts are 8,192 tokens, one waits inside the scheduler. MTP, prefix
reuse, and the tail-deferral experiment are disabled.

After three solo responses, eight rounds alternate two distinct 8,192-token
completion prompts. Each round observes two admitted streams with one waiting,
rejects a third client with `429 concurrency_limit`, and disconnects the active
completion after its first content event. A short chat uses the freed admission
slot while the other completion is active. Both completed replies match their
respective solo text, finish reason and usage. All five work counters drain after
each round. All 16 completed replies match. The public API observations capture text and usage, and the earlier
native stepwise M5 control captured request-specific cursors.

Between the fourth and fifth rounds, a passive sampler records healthy responses,
zero work gauges, and an unchanged cumulative engine step count in 59 samples
spanning 29.55 seconds within the 30.01-second idle interval. Replay verifies
that the large solo replies can distinguish the alternating survivor roles and
that all 27 successful admission IDs, including the three solo controls, are
distinct. The publisher inventory and derived manifest match before and after.
All 384 memory-pressure samples are normal. Peak server physical footprint is
75.72 GiB, within the inherited 96 GiB run guard.

This functional control exercises bounded two-client admission and recovery.
The standard single-request release and concurrent serving throughput retain
their existing evidence requirements.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
