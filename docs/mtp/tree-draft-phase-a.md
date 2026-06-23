# Tree-draft speculative decoding — Phase A gate (do not relitigate)

**Question.** Can a *tree* of MTP drafts break the linear depth-D acceptance
ceiling on Qwen3.6 27B and yield a materially higher tokens-per-target-forward
than the current linear argmax chain?

**Verdict (2026-06-05, M5 Max).** No. In the realistic high-acceptance regime
the best-shaped tree buys ≤1.14x *projected* tokens/forward and saturates at 8
leaves. The projection is optimistic (assumes a free single-forward tree verify);
the real win after Phase B costs would be well under that. **Phase B (FFI +
tree-mask single-forward verify) is not justified.**

## Prototype

`crates/ax-engine-mlx/src/bin/tree_draft_probe.rs` (Phase A). Drives the model's
real MTP head and the real target verify forward
(`forward_all_positions_with_post_norm`), so every candidate path is verified as
an ordinary linear sequence with correct RoPE — no FFI changes. The linear arm is
the tree arm with branch schedule `[1,1,…]`, so both arms are the same code path
and differ only in candidate count. Greedy argmax acceptance ⇒ identical committed
trajectory across schedules (asserted as a cross-check). It exposes one new public
primitive, `mtp::mtp_head_step`, that returns the per-depth head logits so a caller
can branch (top-k), which `mtp_draft_tokens` (argmax chain only) cannot.

Run:
```bash
# tokenize a real prompt with the model chat template into comma-sep ids first
AX_TREE_PROMPT_FILE=/tmp/flappy_ids.txt \
AX_TREE_SCHEDULES="2,2,1,1,1;2,2,2,1,1;2,2,2,2,1;3,2,2" \
  ./target/release/tree_draft_probe <model_dir> 160
```

## Result — flappy suite (273-token real prompt, 160 committed, depth fixed by schedule)

| schedule        | leaves | accept/step | proj tok/fwd | vs same-depth linear |
|-----------------|-------:|------------:|-------------:|---------------------:|
| linear-d3       |      1 |       1.556 |        2.556 |               1.000x |
| linear-d5       |      1 |       1.776 |        2.776 |               1.000x |
| `[2,2,1,1,1]`   |      4 |       2.096 |        3.096 |               1.115x |
| `[1,2,2,1,1]`   |      4 |       1.927 |        2.927 |               1.055x |
| `[2,2,2,1,1]`   |      8 |       2.157 |        3.157 |               1.137x |
| `[2,2,2,2,1]`   |     16 |       2.157 |        3.157 |               1.137x |
| `[3,2,2]`       |     12 |       1.981 |        2.981 |               1.167x (vs d3) |
| `[2,2,2]`       |      8 |       1.825 |        2.825 |               1.105x (vs d3) |

Key observations:
- **Saturation:** widening from 8 → 16 leaves adds *zero* accept/step (2.157 =
  2.157). The branching headroom is exhausted by ~8 leaves.
- **Going deeper-linear captures most of it:** `[3,2,2]`'s projected 2.981 tok/fwd
  is barely above plain linear-d5 (2.776) — depth-3→5 on the linear chain is
  almost as good as a 12-leaf tree.
- **Regime dependence (the whole story):** on a synthetic high-entropy prompt the
  same `[2,2,1,1,1]` tree showed **1.71x** (linear 0.49 → tree 1.55 accept/step).
  The tree only wins where baseline acceptance is poor; in the real regime (where
  the model already accepts most drafts) the ceiling is already near-saturated and
  there is almost nothing left for a tree to capture. This is the empirical
  confirmation of the MTP-vs-mlx-lm parity root cause.

## Why the projection is optimistic (real win < table)

The table credits the tree a single free verify forward/step. A real Phase B
implementation would cost more:
- **Per-token RoPE positions** (tree node position = depth, not flatten index)
  require extending the scalar-offset `mlx-sys` rope FFI — the dense attention
  path threads one `token_offset` and assumes sequential positions.
- **Tree attention mask** (`ScaledDotProductAttentionMask::Array`) over the
  flattened tree, plus larger verify tensors (more nodes than a linear draft).
- **27B is linear-attention** (`linear_attention_enabled`, full attn every 4th
  layer). The recurrent conv/linear state cannot be `trim_to`'d back to a prefix,
  so each step still pays a commit/recompute forward over the accepted tokens
  (production: `recompute_committed_prefix`). A tree's deeper rejections make this
  more frequent, not less.

Under rejection-sampling acceptance (production default) rather than the probe's
strict argmax, the baseline accepts *more* per step, so the tree's marginal
headroom shrinks further — i.e. the probe **overstates** the tree benefit relative
to production. The ≤1.14x is an upper bound.

## Bottom line

Tree drafting does not break the 27B ceiling in the operating regime. The lever
that would (raising acceptance where it's already ~saturated) doesn't exist; the
only regime a tree helps (low baseline acceptance, e.g. hard fresh-generation) is
already handled by the draft confidence gate truncating low-confidence drafts. Do
not build Phase B for this model.
