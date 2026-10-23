# Flash Next long-context decode and decode glue (2026-10-23)

Apple M5 Max, 128 GB, pack on the internal SSD (`AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP`
@ `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`), MTP off unless noted, `--prefill-chunk 2048
--max-batch-tokens 2048`, `--total-blocks 4096` for the 39k prompt. Campaign host, signal only:
not a certification run. Analysis:
[docs/performance/flash-next-long-context-decode-2026-10-23.md](../../../docs/performance/flash-next-long-context-decode-2026-10-23.md).

## Protocol

Same server build with a kill switch on versus off (or two consecutive builds), page cache purged before each
arm, first-seen prompts (`p1` 2,207 tokens, `long1` 8,976, `long32` 39,225), greedy, 128 generated tokens.
Decode is the rate after the first token. Text SHA-256 is compared across arms and rounds; it is identical in
every comparison recorded here.

## Files (all under `raw/` unless noted)

- `kv-segments-ab-{off,def}-{1,2}.json`: sealed-prefix-plus-tail K/V history against the flat history
  (`AX_FLASH_NEXT_KV_SEGMENTS=0`), two rounds each (`harness/run_kv_segments_ab.sh`, `harness/fn_ab.py`).
- `rope-cache-ab-{off,def}-{1,2}.jsonl`: QSA rotary table cache against `AX_FLASH_NEXT_ROPE_CACHE=0`.
- `gdn-native-ab-{off,on}-{1,2}.json`: the native singleton GDN kernel on (`AX_MLX_FLASH_NEXT_GDN_METAL=1`)
  against the portable recurrence, before it became the default (`harness/run_gdnmetal.sh`).
- `glue-ab-t9-{prev,new,newoff}-{1,2}.json`: previous build against the build with the pooled cache, rotary
  stretches, weight views and native GDN (and that build with those switches off) (`harness/run_t9.sh`).
- `glue-ab-wave2-{prev,new}-{1,2}.json`: previous build against grouped GDN heads, f32 gains, gate and
  route-tail stretches and identity slices (`harness/run_t11.sh`).
- `long-context-qa-*` and `final-build-qa-*`: canonical MTP (`mtp`), batched MTP verifier (`mtpb`), direct texts
  (`direct`, eight prompts from 28 to 39,225 tokens), cancel mid-prefill then re-request (`cancel`), 16-request 9k
  and 30-request short soaks, each with the switches of the change on (`def`) and off (`off`); `long-context-qa`
  is the K/V-segment build, `final-build-qa` the build with the native GDN gate kernels and without the native hyper-connection read (`harness/run_long_context_qa.sh`,
  `harness/run_final_build_qa.sh`, `harness/fn_qa.py`).
- `stage-profile-{before,after,final}-*.json`: synchronized per-step stage attribution at 2.6k, 9k and 39k tokens
  of context (`flash_next_decode_context_stage_profile`, test-only marks inside QSA). Each stage is evaluated
  before the next starts, so the stage sum exceeds the real step; compare growth with context, not totals.
- `op-census-*.json`: per-step MLX op census by wrapper and calling function (`mlx_sys::op_trace_begin`); the
  `m2-dev` files carry source lines (dev build), the `m5-release` file function names only.
- `final-build-streamed-and-image.txt`: streamed-experts (two requests) and image prefill plus decode (three
  requests) on the final build; the text SHA-256 values equal those of the earlier builds.
- `harness/cache_append_probe.py`: the MLX microbenchmark of a per-step K/V concatenation.
- `gdn-native-kernels-ab-{old,new}-{1,2,3}.json`: the native GDN gate kernels in the stretch mask (`new`) against
  the mask without them (`old`), same binary (`harness/run_t15.sh`).
- `hc-native-ab-{prev,new}-*.json`, `hc-native-ab-same-binary-{old,new}-*.json`, `hc-native-ballast-ab-{old,ballast,new}-*.json`:
  the first version of the native hyper-connection read (merged GEMV plus an elementwise kernel, measured slower and
  since replaced) against the previous build, against the same binary with the stretch
  masked out, and with an allocation-only control (`harness/run_t13.sh`, `run_t13b.sh`, `run_t14.sh`; the ballast
  arm was an M5-only build that kept the merged weights but never used them, so only the numbers are recorded).
- `op-census-m2-dev-shipping.json` and `op-census-m2-dev-all-stretches.json`: the census of the shipping mask (3,712
  ops per step) and with every stretch on.
- `exhaustive/`: the M2 Ultra exhaustive sweep and gate outputs, and the hyper-connection read timing probe.
- `hc-gemv-ab-{base,hc}-{1,2,3}.json` (M5 Max) and `hc-gemv-m2-ab-{nohc,def}-{1,2}.json` (M2 Ultra): the second
  version of the native hyper-connection read (two GEMV kernels with fused epilogues, default-on) against the same
  binary with the stretch masked out (`harness/run_t16.sh`, `harness/run_m2_hcgemv_ab.sh`; the M2 harness uses
  `harness/fn_ab.py` with the pack path of that host).
- `hc-gemv-qa-{direct,mtp,mtpb,cancel}-{def,off}.json`, `hc-gemv-qa-soak.json`, `hc-gemv-qa-soak2k.json` and
  `hc-gemv-qa-streamed-and-image.txt`: the QA matrix of the final build (`harness/run_t17_qa.sh`), the switches of
  the campaign on (`def`) and off (`off`); every text hash equals the off arm and the build before the read.
- `op-census-m2-dev-shipping-hcgemv.json`: the op census of the final shipping mask (3,040 ops per step).
