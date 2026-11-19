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
