# Flash Next MXFP8 resident loader preview on M2 Ultra

The two native boots used the same experimental server build as the
[full MXFP4 comparison](../flash-next-native-loader-m2-20261121/README.md).
The [publisher inventory](../qualification/2026-11-18-flash-next-mxfp8-preview/pack-inventory.json)
binds all 61 files of the MXFP8 pack at
`48b9aa5f002f29a9f8ab7cfe5c7185e342595b1b`. The run used Mac Studio
M2 Ultra 192 GB with macOS 27.2.1, external APFS storage, MLX 0.32.4,
the preview opt-in and the resident lazy CPU loader.

| Measurement | MTP disabled | MTP required |
| --- | ---: | ---: |
| Spawn to ready | 254.59 s | 187.96 s |
| Resident load and evaluation | 125.13 s | 121.04 s |
| Peak sampled startup RSS | 130.13 GiB | 130.01 GiB |
| Resident shards / tensors | 27 / 1,842 | 27 / 1,842 |
| MTP checks | 0 | 34 |
| Drafts accepted / proposed | 0 / 0 | 28 / 32 |

Both routes emitted the same 64 greedy output IDs across a historical
16-token input and a fixed 512-token input, with 32 generated tokens per
request. The first 12 output IDs from the shorter input matched the
[earlier resident control](../qualification/2026-11-18-flash-next-mxfp8-preview/m2-resident-direct.json).
The required route used the canonical verifier and reported zero
fallbacks. Every startup sample reported normal system pressure.

This is resident functional evidence for the tested M2 Ultra 192 GB.
The two process starts used different MTP policies and occurred in
sequence. Timing variation across these arms includes cache and startup
effects. The M5 preview published separately uses expert streaming.
SKU qualification and catalog admission remain open.

- [Raw paired tokens and route data](result.json) and [source receipt](source.json)
- Startup samples [disabled](disabled-startup.jsonl) and [required](required-startup.jsonl)
