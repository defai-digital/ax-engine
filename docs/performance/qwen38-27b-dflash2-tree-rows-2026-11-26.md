# Qwen 3.8 27B DFlash2: 8 or 16 tree rows by output length

**Date:** 2026-11-26 · **Host:** MacBook Pro M5 Max 128 GB, macOS 27.2,
pinned MLX 0.32.4 · **Pack:** Qwen 3.8 27B AXQ MXFP4 with the DFlash2
drafter · **Raw runs:**
[`benchmarks/results/qwen38-27b-dflash2-tree-rows-20261126/`](../../benchmarks/results/qwen38-27b-dflash2-tree-rows-20261126/README.md)

DFlash2 verifies 8 draft-tree rows per round on M5 and later unless
`AX_MLX_DFLASH2_TREE_ROWS` says otherwise
([`dflash2_tree.rs`](../../crates/ax-engine-mlx/src/dflash2_tree.rs)).
The earlier measurement behind that default had 8 rows ahead at 64 output
tokens (111.4 against 108.4 tok/s, three prompts) and 16 rows ahead at 256
(102.8 against 100.8, seven prompts). This run asks whether a request
should pick 8 or 16 rows from its output budget. Nothing in the engine
changed.

## Result

Decode tok/s over seven prompts, arms in the order 8, 16, 8, 16 rows, 150
seconds of rest before each:

| Output cap | 8 rows, arms 1 and 3 | 16 rows, arms 2 and 4 | 16 against 8 |
| ---: | ---: | ---: | ---: |
| 64 | 102.5, 106.0 | 106.4, 106.4 | +2.1% |
| 128 | 97.8, 98.1 | 99.1, 98.7 | +1.0% |
| 192 | 96.3, 96.3 | 98.9, 98.7 | +2.6% |
| 256 | 97.9, 97.7 | 103.7, 103.2 | +5.8% |

- 16 rows are ahead at every cap, but the gain is not monotone in the cap:
  it is smallest at 128 tokens. The data give no threshold for a rule by
  output budget.
- Per prompt, 16 rows range from 14% slower to 22% faster than 8. Two
  prompts are 13 to 17% faster at every cap from 128 up and one is 9 to 14%
  slower at every cap; the two prose prompts change sign between caps. At 64
  tokens, the first three prompts of the corpus are all faster with 8 rows,
  so a three-prompt sample can favor either setting.
- **The two settings give different greedy text.** 14 of 32 replies differ
  between 8 and 16 rows; each setting repeats its own replies exactly in
  both of its arms. A wider verify forward rounds differently, and a
  changed token changes every acceptance after it. Some of the per-prompt
  spread is therefore a different reply, not a different speed.

## Decision

The default stays at 8 rows, and no request-local switch was built:

- Choosing the rows from `max_tokens` would make the first tokens of a
  reply depend on the cap the client sent. A request with a cap of 64 and
  one with a cap of 256 would no longer start the same way.
- A change of the default is a change of the greedy text, so it needs the
  same evidence as any numerics change, and a larger corpus than seven
  prompts. This run supports measuring 16 rows as a default on that basis;
  it does not support a threshold.

## Not done

- Splash was not measured on this corpus and these caps; the comparison
  was stopped before its first arm finished. The last comparison stands:
  about 9% behind Splash on 256-token outputs (100.8 against 111.3 tok/s,
  [tree-verify note](qwen38-m5-tree-verify-and-flash-next-2026-11-25.md)).
- Acceptance per round was not recorded (the round trace was off), so the
  split of the gain between acceptance and verify cost is not known.
