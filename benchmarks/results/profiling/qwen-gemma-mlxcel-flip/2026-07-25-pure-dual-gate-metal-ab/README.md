# Pure-wall A/B: Gemma dual gate/up custom Metal (mbp-m5)

- Host: AKMBPM5MAX.local (mbp-m5)
- Model: models--ax-local--gemma-4-12b-it-4bit-assistant-mtp (MLP bits=8 gs64)
- Prompt: pure Gemma 13.8k (`max_tokens=1`)
- Flag: `AX_MLX_GEMMA_DUAL_GATE_UP_METAL`

## Result

| | cold mean |
|--|--|
| OFF | 8804 ms |
| ON | 74658 ms |

**ratio_on_over_off = 8.48×** → **default OFF** (want <0.925).

Full S0–S3 flip campaign not run (no pure headroom for S1 thr≥~21).
