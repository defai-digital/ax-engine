# Flash Next fixed-boundary prefix state on M2 Ultra

The native Flash Next MXFP4 pack at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` ran on Mac Studio M2 Ultra
192 GB with an external SSD, macOS 27.2.1 and MLX 0.32.4. The probe
built in an isolated `0f597f13c` checkout with the Rust 1.99.1
release-server profile. Its binary, source patch, model metadata,
controls and prompt IDs are bound by the [result](result.json).

Each of two fixed prompts used a canonical 2,048-token prefill
boundary, followed by 64 prompt tokens and 32 greedy outputs.
The probe matched a standalone seed's wire state against the cold
continuation's cache-only prefill at the same cut. Both stored cache
payloads were 172,269,937 bytes. Deserialization and model rebind
preserved the wire bytes. The restored and cold arms agreed on suffix
state, all 32 generated IDs, and state after decode for both prompts.
The continuation's first output differed from the standalone seed's
first output in both cases.

The alternate layout batched the final prompt row together with the
preceding rows. Its state digest differed in both cases. The scoped
comparison identifies a physical-boundary requirement for any future
extension path. The [HTTP controls](../flash-next-mxfp4-prefix-exact-m2-20261121/README.md)
still exercise full L1 reuse and the current cold fallback for strict
extensions. This fixed-cut state result supports an isolated extension
experiment on the current model. Product admission and sustained
performance evidence remain separate.

## Replay

Apply the [compressed test patch](state-probe.patch.gz) to the exact
source base in an isolated checkout. Set `AX_FLASH_NEXT_REAL_PACK`
to the admitted local pack and `AX_FLASH_NEXT_PREFIX_CASES` to the
[fixed prompts](prompts.json). Run the library test
`flash_next_aligned_extension_rehydrates_same_prefix_state` in
`ax-engine-mlx` with the `release-server` profile and
`--ignored --nocapture`. The test emits a JSON row per prompt.
