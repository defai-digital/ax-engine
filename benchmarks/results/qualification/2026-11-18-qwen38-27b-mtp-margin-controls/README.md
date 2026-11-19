# Qwen 3.8 27B MXFP4 paired greedy logit margins

On MacBook Pro M5 Max 128 GB, the native test replayed the 46 input tokens
for `instruction-001` from the AX-only performance pilot. Direct decode and
required MTP each reproduced their recorded 128 output tokens. The test then
replayed both routes with observation enabled and confirmed identical outputs
and stop reasons for each route. The pack, pilot artifact, executable and
runtime are bound in [provenance.json](provenance.json).
The pilot retained the benchmark's default repository label. Its explicit
model directory points to the pinned MXFP4 snapshot used for this replay.

The routes selected different tokens at 119 aligned positions. The first
difference was at index 7, counting from zero. With the same prior tokens,
direct selected `15217` with a 0.125 logit advantage over `2397`. The MTP
verifier selected `2397` with both candidates at 23.375. Every later
difference uses each route's own generated prefix. Across those positions
the largest paired gap was 29.3125.

[m5-paired-greedy.json](m5-paired-greedy.json) includes the input and outputs,
both scores at each differing position, each row's gap to its highest-scoring
other token, and the larger of the two route-specific paired gaps. Verifier
rows are taken from the committed output positions. Ordinary target rows are
used for singleton decisions. The replay materializes the retained logits
after completing each run.

This is one fixed-budget, greedy, untimed diagnostic. A separate test in the
same run also reproduced the [same-state audit](../2026-11-18-qwen38-27b-mxfp4-mtp-s-controls/),
including 69 accepted drafts over 38 verifier cycles. MTP-P qualification
and MTP-D promotion remain open.

## Rollback audit

The [M5 rollback audit](m5-rollback-audit.json) passed four synthetic
controls and two real-weight tests on the same pinned pack. The synthetic
controls verify that the captured correction comes from the singleton
replay that selected the emitted token. The real same-state control
retains committed KV boundaries and visible outputs from 38 verifier
cycles, including 69 accepted drafts. The independent reader checked all
119 differences from the paired 128-token pilot. The receipt binds the
tested binary and source file hashes.

## Native SDK session replay

On MacBook Pro M5 Max 128 GB, the native inference session reproduced the
recorded 512-token output for each route on the measured `coding-003`
prompt. Direct and required MTP ran in separate processes. Both reached
the fixed output budget and produced the same output with logit capture
enabled. The [measured token IDs](../2026-11-18-qwen38-27b-mxfp4-mtp-performance/m5-agent-coding-trials.jsonl)
and [build and runtime provenance](m5-native-session/provenance.json)
bind the replay to the original timing case.

There were 193 differences. The first was at zero-based index 319 with
the same prior tokens. The direct target chose `886` in a tie at 30.625.
The MTP verifier chose `4307` with a 0.125 logit advantage. The remaining
192 comparisons use the prefixes generated independently by each route.
All 193 pairs are in the [replay receipt](m5-native-session/paired.json).
This is an untimed, fixed-budget diagnostic for one request. MTP
qualification remains open.

## Reproduce the paired pilot

To replay the included case, convert the receipt to the test input format
and point the test at the [pinned snapshot](../2026-11-18-qwen38-27b-mxfp4-inventory/).
Run from the repository root and set `RECEIPT` to the path of
`m5-paired-greedy.json`.

```bash
jq '[{case_id, prompt_tokens, max_output_tokens, ignore_eos,
      direct_tokens: .direct.tokens, mtp_tokens: .mtp.tokens}]' \
  "$RECEIPT" > /tmp/qwen27b-mtp-margin-cases.json

AX_MLX_DFLASH2=0 \
AX_QWEN38_27B_MTP_MARGIN_CASE=/tmp/qwen27b-mtp-margin-cases.json \
AX_QWEN38_27B_MTP_S_MODEL_DIR=/path/to/pinned/snapshot \
rustup run 1.99.1 cargo test -p ax-engine-mlx --lib \
  qwen38_27b_report_paired_greedy_logit_margins \
  -- --ignored --nocapture --test-threads=1 > /tmp/qwen27b-mtp-margin-replay.log 2>&1

python3 scripts/verify_qwen38_27b_mtp_margin_replay.py \
  /tmp/qwen27b-mtp-margin-cases.json --rust-log /tmp/qwen27b-mtp-margin-replay.log
```

For offline validation, use the same case file and
`--report-json "$RECEIPT"`. For a new timing run, build the case file
from the measured prompt and output token IDs before running the test.

## Reproduce the SDK replay

Set `RECEIPT_DIR` to the path of `m5-native-session` and `MODEL_DIR` to
the path of the pinned snapshot. Run from the repository root with the
pinned MLX runtime. Each arm runs in a separate process so cached runtime
switches cannot cross between them.

```bash
export AX_MLX_NATIVE_CONFIRM=1 AX_MLX_DFLASH2=0
export AX_ENGINE_PREFIX_REUSE_DISABLED=1 AX_MLX_PREFIX_CACHE_DISK_DISABLED=1
export AX_MLX_PREFIX_CACHE_MAX_BYTES=0 AX_MLX_PREFIX_CACHE_MAX_ENTRIES=0
export AX_QWEN38_27B_MTP_S_MODEL_DIR="$MODEL_DIR"
export AX_QWEN38_27B_MTP_MARGIN_CASE="$RECEIPT_DIR/focused-cases.json"
unset AX_NO_SPEC

AX_QWEN38_REPLAY_ARM=direct AX_NO_SPEC=1 AX_MLX_SKIP_DECODE_ROUTE_TELEMETRY=0 \
AX_MLX_ROTATING_SLIDING_PREFILL=0 AX_GEMMA4_LOOP_DETECTION=off \
rustup run 1.99.1 cargo test --profile release-server -p ax-engine-sdk \
  --features qwen-replay-diagnostics --test qwen38_mtp_replay \
  qwen38_27b_replay_native_session_arm -- --exact --ignored --nocapture --test-threads=1

AX_QWEN38_REPLAY_ARM=mtp \
rustup run 1.99.1 cargo test --profile release-server -p ax-engine-sdk \
  --features qwen-replay-diagnostics --test qwen38_mtp_replay \
  qwen38_27b_replay_native_session_arm -- --exact --ignored --nocapture --test-threads=1

python3 scripts/verify_qwen38_27b_mtp_margin_replay.py \
  "$RECEIPT_DIR/focused-cases.json" --report-json "$RECEIPT_DIR/paired.json"
```
