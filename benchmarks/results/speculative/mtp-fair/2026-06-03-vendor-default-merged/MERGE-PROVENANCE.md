# Vendor-default merged run

This directory contains:

- **MTPLX (6 cells)** and **AX Engine MTP / MTP+ngram (12 cells)** rows
  copied from `2026-06-02-qwen36-fair-v0632/`. These rows are reused
  because their underlying engines were not affected by the Lightning
  silent-thinking pathology discovered on 2026-06-03.
- **Lightning rows** (target: 12 cells) re-run on 2026-06-03 with the
  new diagnostic instrumentation (silent_thinking_suspected flag,
  content/reasoning stream separation, per-run server log, source
  identity capture).

## Why the Lightning rows were re-run

The 2026-06-02 Lightning rows reported decode_tok_s of 9-178 tok/s but
the visible output `text` was 8-19 chars for 1000 generated tokens
(ratio < 0.02). Closer inspection showed:

- `--no-thinking` only affects the chat template, not inference.
- Qwen3.6 with the closed `<think></think>` hint still generates think
  tokens silently which Lightning's reasoning parser strips from the
  stream but still counts in `completion_tokens`.
- The 178 tok/s case was a degenerate loop emitting "Function Function12"
  repeatedly.
- N-gram global auto-disable carried across cases, making "pure MTP"
  results dependent on case ordering.

The new run uses vendor-default labeling and captures enough diagnostic
data per row to detect any repeat of these failure modes.
