# Flash Next long-context decode: RoPE table cache and segmented K/V history (2026-10-23)

Apple M5 Max, 128 GB, pack on the internal SSD (`AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP`
@ `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`), MTP off unless noted, `--prefill-chunk 2048
--max-batch-tokens 2048`, `--total-blocks 4096` for the 39k prompt. Campaign host, signal only:
not a certification run. Analysis:
[docs/performance/flash-next-long-context-decode-2026-10-23.md](../../../docs/performance/flash-next-long-context-decode-2026-10-23.md).

## Protocol

Same server build, kill switch on versus off, page cache purged before each arm, first-seen
prompts (`p1` 2,207 tokens, `long1` 8,976, `long32` 39,225), greedy, 128 generated tokens.
Decode is the rate after the first token. Text SHA-256 is compared across arms and rounds.

## Files

- `raw/kv-segments-ab-{off,def}-{1,2}.json`: sealed-prefix-plus-tail K/V history against the
  flat history (`AX_FLASH_NEXT_KV_SEGMENTS=0`), two rounds each (`harness/run_kv_segments_ab.sh`,
  `harness/fn_ab.py`).
- `raw/rope-cache-ab-{off,def}-{1,2}.jsonl`: QSA rotary table cache against
  `AX_FLASH_NEXT_ROPE_CACHE=0` (five prompts, `harness/fn_qa.py` texts scenario).
- `raw/long-context-qa-*`: canonical MTP (`mtp`), batched MTP verifier (`mtpb`), cancel
  mid-prefill then re-request (`cancel`) and a 16-request 9k soak, each with the K/V segments on
  (`def`) and off (`off`) (`harness/run_long_context_qa.sh`).
- `raw/stage-profile-{before,after}-*.json`: synchronized per-step stage attribution at 2.6k, 9k
  and 39k tokens of context (`flash_next_decode_context_stage_profile`, test-only marks inside
  QSA). Each stage is evaluated before the next starts, so the stage sum exceeds the real step.
- `harness/cache_append_probe.py`: the MLX microbenchmark of a per-step K/V concatenation.
