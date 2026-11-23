# Flash Next paired QSA gather for fixed-width prefill

The experimental copy reads two token-major BF16 K/V histories in one
Metal dispatch and writes the same tiles consumed by stock masked SDPA.
It handles sealed prefix and tail segments directly and keeps the
original token-zero padding and attention mask. The same pipeline
takes runtime tile and segment geometry. The first eligible call
evaluates a small fallible warmup and can retain the stock path if
the pipeline is unavailable.

The experiment is enabled by
`AX_MLX_FLASH_NEXT_QSA_PAIRED_GATHER=1` for a single-batch selected
prefill with two 256-channel BF16 KV heads and history up to 8,192
tokens. The existing selection, sub-batch cap and SDPA arithmetic
are used in both arms. The default uses the existing gather.

Both native tests load the same verified 49-member MXFP4 pack and use
the exact 8,192-token long prompt from the
[M5 mixed-step record](../flash-next-shared-step-clock-m5-20261123/README.md).
The same signed Rust 1.99.1 release-server test executable runs on
M5 Max 128 GB and M2 Ultra 192 GB, each on macOS 27.2.1 with internal
SSD and MLX 0.32.4. The 20 forwards retain four repetitions of
`512, 512, 512, 511, 1`. Both paths warm up before timing, then use
the paired–stock–stock–paired order. Every measured run matches
the full serialized state and final logits bitwise to stock.
Each paired arm dispatches 432 paired copies.

| Device | Stock forwards, seconds | Paired forwards, seconds | Paired / stock median |
| --- | ---: | ---: | ---: |
| M5 Max 128 GB | 9.260 / 9.663 | 8.904 / 9.426 | 0.969× |
| M2 Ultra 192 GB | 20.037 / 19.863 | 19.801 / 20.023 | 0.998× |

The two M5 orders measure 3.8% and 2.5% less model-forward time.
M5 CPU idle is 98.68% before and 83.70% after, below the 90% endpoint
contract, so this supports a directional diagnostic only. M5
peak MLX-tracked bytes rise by 111 MiB. M2 endpoints are 95.65% and
91.62% idle. All 292 pressure observations are normal. The same signed
binary and pinned MLX library also pass the paired gather component
test on M4 Pro 64 GB running macOS 27.0. That test covers padding,
duplicates and the segment boundary.

This evidence covers one fixed prompt and uses model forwards.
Same-load serving, primary-tier repeatability and default promotion
remain separate. The benchmark patch predates the accompanying
test-only fastpath classification and eval inventory updates. The
portable synthetic control runs the final Metal source from the repo
root and retains the original SDPA and tile layout.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
