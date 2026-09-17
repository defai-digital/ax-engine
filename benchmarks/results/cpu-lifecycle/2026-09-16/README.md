# CPU and lifecycle regression evidence

These artifacts cover request bookkeeping, deterministic worker lifecycle,
media subprocess handling, and source packaging. No model weights were loaded.
They are not inference throughput, long-context model qualification, a serving
soak, or evidence for the Mac mini M5 / Mac Studio M5 Ultra product SKUs.

## Request count

Run from the repository root:

```sh
bash scripts/cargo-pinned.sh run -p ax-engine-core --example request_count_cost --release
```

`request-count.jsonl` compares the old `snapshots().len()` operation with
`records_len()` in the same optimized executable, over six trials with
alternating measurement order. The workload has 1 or 16 live requests, fixed
prompt tokens, and no generated history. Both methods return the same count.

At 131,072 prompt tokens, snapshot counting took a median 7.14 us for one
request and 161.01 us for sixteen. The direct scalar read is near the harness
timing floor; its ratio should not be interpreted as an application speedup.
The concrete eliminated copying is 512 KiB and 8 MiB respectively, per count.
Other scheduler snapshots remain unchanged.

## Worker and media lifecycle

`stepwise-recycle.json` records the same deterministic request before and after
the fix, with recycling disabled and with a one-tick threshold. Before the
fix, the threshold caused a second factory call, lost the live request, and
left its admission permit occupied. After the fix, the same request remains
accessible and its original session stays alive. The regression also checks
that cancellation releases capacity and allows idle recycling:

```sh
bash scripts/cargo-pinned.sh test -p ax-engine-server recycling_preserves_live_stepwise
bash scripts/cargo-pinned.sh test -p ax-engine-server tasks::tests
bash scripts/cargo-pinned.sh test -p ax-engine-server multimodal::process::tests
bash scripts/cargo-pinned.sh test -p ax-engine-server openai_chat_media_build_offloads
```

The subprocess tests cover independent output caps, deadline, cancellation,
reaping, and inherited output descriptors. The HTTP preparation regression
checks media saturation and the unaffected text-only path. `ffmpeg-smoke.txt`
records a separate successful test with ffmpeg 9.0.1, generating a three-frame
MP4 and checking sampled frame count, color, dimensions, and timestamps:

```sh
bash scripts/cargo-pinned.sh test -p ax-engine-server ffmpeg_native_video_smoke -- --ignored --nocapture
```

## New prompt artifact size

`prompt-artifact-size.json` measures the actual output of `write_prompt_tokens`
against the previous indented serialization of the same parsed object.
For 131,072 tokens, the JSON is 806,719 bytes instead of 1,462,119 bytes (44.8%
smaller). Token values, token SHA-256, schema, and filename identity are
unchanged. The other two workloads and their construction rules are included.
Existing benchmark artifact bytes and URLs were not rewritten. Installed
runtime size and the MLX metallib are unchanged.

```sh
.venv/bin/python -m pytest scripts/test_bench_mlx_inference_stack.py -k prompt_artifact
```

## Source distribution

`manifest.json` records an actual maturin source archive and successful offline
native Python-extension build from its extracted workspace. Cargo pruned 221
unused lockfile packages after maturin reduced the workspace; no dependency
version was added. Removing the extracted MLX pin then caused build exit 101,
even with `AX_MLX_VERSION_OVERRIDE=1`. The source archive includes both version
pins, the patched `block` dependency, and the license.

```sh
.venv/bin/python -m pytest scripts/test_source_distribution.py scripts/test_cargo_pinned.py
```

The manifest binds the measurements to source hashes and records host and
toolchain identity. Runtime speed conclusions require the separate target-SKU
inference-stack harness with `mlx_lm.benchmark` as primary baseline.
