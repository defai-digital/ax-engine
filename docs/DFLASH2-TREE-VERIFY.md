# DFlash2 tree verify on Apple silicon: step 0 measurements

The row-cost and block-size tables are historical measurements from branch
`track-l-lattice-tree`. The current engine verifies a draft tree in one
target forward on M5 and later hosts and has separate packed-cohort
controls.

## Current Qwen 3.8 27B MXFP4 controls

`AX_MLX_DFLASH2_COHORT_MAX_ROWS=4` caps each lane of a packed DFlash2
verifier forward at four rows, including the anchor. A global
`AX_MLX_DFLASH2_WINDOW` pin takes precedence. A solo request uses its
usual verifier policy. Leave `--disable-ngram-acceleration` absent when
testing DFlash2. That flag prevents the drafter from attaching. Check
`ax_mlx_dflash2_cohort_lanes` before comparing throughput.

`AX_MLX_DFLASH2_PROFILE_REPLAY=1` with `AX_MLX_DFLASH2_ROUND_TRACE=1`
materializes the convolution and recurrent state after partial-prefix
replay and reports graph build and evaluation time separately. Use a
separate run for throughput measurements because the diagnostic adds a
GPU synchronization point.
[M5 serving and replay controls](../benchmarks/results/qwen-cross-engine-mlx-20261125/README.md#qwen-38-27b-dflash2).

## One-forward draft tree

A chain round stops at the first draft the target rejects, even when the
drafter's second choice was right. A tree round verifies alternatives in
the same forward. The solo drafter already scores 16 candidates per
position and a 16 by 16 transition table between positions, so the tree
costs no extra drafter work.

- **Shape.** A Metal kernel grows the tree best first from that table.
  A node's children are the next position's candidates scored by their
  own logit plus the transition score from the node, at most four
  children per node, and a child's score is its parent's score times its
  share of that softmax. Rows are laid out in pre-order, so a node's
  ancestors precede it and a row's depth is its position offset.
- **Forward.** Full attention uses an ancestor mask and per-row RoPE
  offsets. The gated-delta layers run a tree kernel that keeps one
  recurrent state per depth and starts each row from its parent's state.
- **Commit.** The target's greedy tokens pick one path. The K/V rows are
  trimmed to that path and the path is replayed through the gated-delta
  state, as after a partially accepted chain.

`AX_MLX_DFLASH2_TREE_ROWS` sets the row count, anchor included (4 to 16;
`0` keeps the chain). Unset, it is 8 rows where a NAX kernel serves the
verify projections (M5 or later, including the pinned MLX 0.32.4 wheel
through AX's `qmv_nax` port) and the chain elsewhere. The tree covers solo
greedy rounds that verify the whole drafter block on the dense Qwen
gated-delta stack. A cut window, a routed-expert model and a cohort keep
the chain. `tree_rows` in each `dflash2_round` trace line is the row count
of that round, or 0 for a chain round.

The emitted tokens are the target's own greedy choices in both forms.
Off-chain rows put their keys in another order, so sums can differ in the
last bits and a near-tie can resolve differently than under the chain.

MacBook Pro M5 Max 128 GB, Qwen 3.8 27B MXFP4, pinned MLX 0.32.4, greedy,
interleaved arms of one binary:

| Verify | 3 prompts x 2, 64 tokens (tok/s) | 7 prompts, 256 tokens (tok/s) | Eval per round (ms) |
| --- | ---: | ---: | ---: |
| Chain, 8 rows | 97.6 | 95.4 | 36.2 |
| Tree, 8 rows | 111.6, 110.6, 111.3 | 101.2, 100.4 | 36.7 |
| Tree, 10 rows | | 98.4 | 39.8 |
| Tree, 16 rows | 108.4 | 102.8, 102.8 | 41.1 |

The tree rows include the draft-head subset (`AX_MLX_DFLASH2_HEAD_SUBSET`);
an 8-row tree with the whole head measured 108.1 tok/s on the 64-token
workload, and the chain with the subset 100.0. Accepted tokens per round
over the seven 256-token prompts are 4.1 for the chain, 4.4 at 8 rows and
5.0 at 16 rows. Rows 9 to 16 cost 3 to 4 ms more per round, because the LM
head and attention leave their 8-row kernels and the projections read a
second row tile, which takes back most of what the wider tree accepts.
That is why the default is 8 rows.

To price another tree shape before writing a kernel, record candidate
tables with `AX_MLX_DFLASH2_CANDIDATE_TRACE=<path>` and replay them:

```bash
python3 scripts/simulate_dflash2_tree.py <trace.jsonl> --rows 8 12 16 --children 2 4
```

## Historical experiments

The SGLang version of this work (automatosx `track_l`, Qwen3.8-27B on one NVIDIA B200) verifies a
16-node DFlash2 lattice tree instead of the 8-token chain, losslessly (the target checks every
token). There it gives 1.14-1.42x at one request over stock DFlash2 chain 8. Three lessons from that
work decide whether a Metal port is worth writing:

1. A 16-node tree beat 32 and 64 nodes: verify cost stays flat to about 16 rows, then grows.
2. The drafter is a small share of a round (7-9% on B200), so a tree pays off only through accepted
   tokens per verify, not through cheaper drafting.
3. Verify and prefill numerics must match the reference path, or greedy near-ties diverge and
   acceptance falls for reasons unrelated to the tree.

This note measures (1) and (2) on the M5 Max and checks block size against acceptance.

## Setup

- Apple M5 Max, 128 GB, macOS 27.2.1 (shared machine; every run below at load average under 2).
- Target `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4`; drafter `z-lab/Qwen3.8-27B-DFlash2` (block 8).
- Prompt context 512 tokens, 20 timed repetitions, medians.
- Two MLX builds: the admitted pin 0.32.4, and 0.32.5 from source (`scripts/build-mlx-source.sh`,
  `AX_MLX_VERSION_OVERRIDE=1`), which has the M5 neural-accelerator matmul kernels
  (`doctor`: `neural_accelerator: true`, `metallib_has_qmv_nax: true`).

## Verify cost by row count

`dflash2-cohort-probe` (packed rows on one cache: the engine's verify path, fused gated-delta kernels
up to 8 rows) and `verify_rows_cost_probe` (generic `forward_all_positions`, linear chain):

| rows | 0.32.4 packed ms | 0.32.5 packed ms | 0.32.4 generic ms | 0.32.5 generic ms |
|---|---|---|---|---|
| 1 | | | 31.3 | 31.5 |
| 4 | | | | 43.0 |
| 8 | 53.0 | **44.0** | 74.4 | 39.1 |
| 12 | | | 111.6 | 38.6 |
| 16 | 85.1 | **43.1** | 85.2 | 39.9 |
| 24 | 91.6 | 60.4 | 88.8 | 57.7 |
| 32 | 94.3 | 63.1 | 87.9 | 59.4 |

- On 0.32.5, 16 rows cost the same as 8 (43.1 vs 44.0 ms); 24-32 rows cost 40% more. The same
  shape as the B200, where the 16-node tree won.
- On 0.32.4, 16 rows cost 1.6x the 8-row block. A 16-node tree would have to raise accepted tokens
  per round by more than 60% to break even; on B200 it raised them by 14-42%. Not worth it on 0.32.4.
- The step from 8 to 16 rows on 0.32.4 is the quantized matmul, not the gated-delta recurrence:
  admitting 16 and 32 rows to the fused gated-delta verify kernels (experimental
  `AX_MLX_FUSED_GDN_VERIFY_MAX_SEQ=32`) left 16 rows at 88 ms.

## Drafter share of a round

`AX_COHORT_PROBE_DRAFT=1`, one lane, block 8: drafter 6.9 ms (0.32.4) / 5.8 ms (0.32.5) per round
against an 8-row verify of 53.0 / 44.0 ms, so **11.6% of a round** on both builds (B200: 7-9%).
Stage split on 0.32.4: MLP 3.5 ms, attention 2.6, LM head 2.5, the rest under 0.5 each.

## Block size and acceptance (0.32.4, end to end)

`AX_MLX_DFLASH2_BLOCK` (experimental) drafts and verifies a longer chain with the same drafter.
16 prompts per family, 512 tokens, greedy, single request, four repetitions in rotated order:

| block | code tok/s | math tok/s | chat tok/s |
|---|---|---|---|
| 8 (checkpoint) | 99-101 | 82-84 | 54-55 |
| 12 | 65-68 | 63-65 | 46-47 |
| 16 | 67-70 | 68-72 | 49-50 |

Round trace (`AX_MLX_DFLASH2_ROUND_TRACE=1`): accepted tokens per round 3.96 at block 8, 3.94 at 16,
3.74 at 24, while the round grows from 50.9 to 58.5 ms. The drafter was trained at block 8, so
drafting a longer chain adds no accepted tokens; the extra width has to come from branching (a
tree), as on B200.

MLX 0.32.5 end to end at block 8: code 144-150, math 118-122, chat 65-67 tok/s (0.32.4: 97-116,
81-88, 53-55); plain decoding 32-34 tok/s on both.

## Decision

The decision below predates the one-forward tree above and AX's own
`qmv_nax` port, which removed the dependence on MLX 0.32.5.

- A Metal 16-node tree verify is worth building, **on MLX 0.32.5 or later only**: there it costs the
  same as today's 8-row chain verify, so the B200 gain (1.14-1.42x in accepted tokens per verify)
  should carry over nearly whole. On the admitted pin 0.32.4 it would lose.
- Needed pieces: tree attention mask in the verify forward; a gated-delta tree recurrence (each node
  starts from its parent's state, the fused verify kernel with a parent index instead of `row - 1`);
  commit of the accepted path's states; and the lattice tree builder (DFlash2 16x16 transition
  scores, as in the SGLang `ax_lattice`).
- Prerequisite: move the MLX pin to 0.32.5 and recertify the bit-exactness gates (lesson 3).
- Keep 16 nodes; 24-32 nodes cost 40% more per verify on 0.32.5.

## Reproduce

```bash
cargo build --release -p ax-engine-microbench --bin dflash2-cohort-probe -p ax-engine-mlx --bin verify_rows_cost_probe
./target/release/dflash2-cohort-probe <target_dir> <drafter_dir> 512 20
AX_COHORT_PROBE_DRAFT=1 AX_COHORT_PROBE_DRAFT_STAGES=1 ./target/release/dflash2-cohort-probe <target_dir> <drafter_dir> 512 20
AX_ROWS=1,4,8,12,16,24,32 ./target/release/verify_rows_cost_probe <target_dir>
```

For 0.32.5, build with `MLX_LIB_DIR` / `MLX_INCLUDE_DIR` pointing at the source build and
`AX_MLX_VERSION_OVERRIDE=1` into a separate `CARGO_TARGET_DIR`.
