I now have comprehensive, quantified evidence. Let me compile the report.

## Performance Analysis: Qwen Linear-Attention MTP Path

### Summary
Overall performance health: **NEEDS ATTENTION** (the 6-bit regression is real but its cause is mislabeled in the existing write-up). I found **5 substantive findings** across the three questions, plus one important **correction** to the "default protocol = exact OFF" framing.

Key numbers extracted from the M5 benchmark artifacts (`2026-08-19-m5-qwen38-4bit-mtp-review/`):

| metric (per MTP step) | 4-bit default | 6-bit default |
|---|---|---|
| `ax_mtp_verify_forward_wall_us` | 29.7 ms | 32.9 ms |
| `ax_mtp_verify_eval_wall_us` | **9.2 ms** | **25.3 ms** |
| `ax_mtp_draft_wall_us` | 4.2 ms | 3.0 ms |
| `ax_mtp_rollback_wall_us` | 1.1 ms | 2.7 ms |
| `ax_mtp_cache_clone_wall_us` (total) | 222 µs | 227 µs |
| depth-0 acceptance | 76.9% | 72.3% |

---

### Q1 — Telemetry vs Behavior: `enabled: 0` is a terminal-step artifact, not a model rejection

**Location of the bug:** `crates/ax-engine-mlx/src/runner/mod.rs:3259-3265` (current, fixed) vs the stale binary that generated the JSONs.

**What the code actually does today:**

The current code calls a dedicated helper (`runner/mod.rs:3259`):

```rust
append_qwen_linear_mtp_exact_route_decisions(
    &mut route_metadata.crossover_decisions,
    self.qwen_linear_mtp_exact_eligible,        // → "eligible"
    self.qwen_linear_mtp_exact_enabled,         // → "enabled"  (MODEL-STABLE)
    exact_arithmetic_enabled,                   // → "active"   (PER-STEP)
    self.qwen_linear_mtp_exact_selection,       // → "selection"
);
```

`append_qwen_linear_mtp_exact_route_decisions` (`runner/mod.rs:12123-12150`) records **four** distinct keys: `eligible` (model-stable), `enabled` (`resolved_profile_enabled`, model-stable), `active` (`active_for_step`, per-step), and `selection`. The unit test at `runner/mod.rs:14020-14039` pins this split.

**The stale point that produced the artifacts:** The benchmark JSONs were generated at binary `0edc42b1` (pre-v7.1.4). That binary recorded `enabled` from the **per-step request-scoped** value `exact_arithmetic_enabled`, not the model-stable value. `exact_arithmetic_enabled` is defined at `runner/mod.rs:3209`:

```rust
let exact_arithmetic_enabled = qwen_linear_mtp_exact_scope_for_request(
    self.qwen_linear_mtp_exact_enabled,
    self.mtp_requested && !all_decode_items_mtp_bypassed,
);
```

`qwen_linear_mtp_exact_scope_for_request` (`runner/mod.rs:12112-12117`) is `resolved_profile_enabled && mtp_requested`. On the **terminal** step (a latched-bypass / short-budget / direct-fallback step), `mtp_requested` is false, so the recorded value drops to `0` and — because `upsert_route_decision` overwrites per step — that final `0` overwrote the stable `1`. The JSON confirms this exact pattern:

- `q4-exact.json:615-617` → `eligible: 1`, `enabled: 0`, **`selection: 2`** (ExplicitEnabled).
- `q4-default.json:615-617` → `eligible: 1`, `enabled: 0`, `selection: 1` (Auto).
- `q6-default.json:616-618` → `eligible: 1`, `enabled: 0`, `selection: 1`.

The `selection: 2` on the exact-env run is the smoking gun: it proves `resolve_qwen_linear_mtp_exact` (`fastpath.rs:528-535`) returned `ExplicitEnabled` at load time. **The 3.8 pack genuinely engages the exact profile** when env is set — `enabled: 0` is purely the terminal-step overwrite.

**What distinguishes 3.6 from 3.8:** *Nothing at the capability gate.* Both report `model_family == "qwen3_5"` — see `crates/ax-engine-server/src/model_load.rs:1513-1519`, where the Qwen3.6-27B and Qwen3.6-35B manifest signatures are literally keyed on `("qwen3_5", 64, 5120, …)` → `Qwen36_27b`, and the test at `model_load.rs:1797` writes `"model_family": "qwen3_5"` for `qwen3.6-27b-6bit`. `qwen_linear_mtp_exact_model_eligible` (`runner/mod.rs:12178-12199`) gates only on `model_family == "qwen3_5" && has_linear_attention && depth ∈ 1..=3 && all tensors supported` — no certification/tier field. Both 3.6 and 3.8 pass (`eligible: 1` in every JSON). The "certified 3.6 vs pending 3.8" difference is a **Tier-1/Tier-2 certification policy** (README verdict #3, `docs/mtp/README.md:17-18`), not a code gate. The `enabled: 1` vs `0` delta between the two benchmarks is only which step happened to be terminal.

**Note on the auto-default:** `resolve_qwen_linear_mtp_exact_with_override` (`fastpath.rs:508-520`) returns `(true, Auto)` when the env is **unset**. So the "default protocol" in these benchmarks is **Auto-exact = ON**, not OFF. This matters for Q2.

---

### Q2 — Replay vs Checkpoint Economics

#### (a) Can the checkpoint path be decoupled from the exact profile? **No — it is *defined by* the exact invariant.**

The checkpoint is not an independent cache feature; it is a snapshot of the batched verifier's recurrent linear-attention state, taken *during* the verify forward. See `runner/mod.rs:8736-8740`:

```rust
let mut verify_cache = state.cache.clone();
if !exact_linear_replay && !pending.is_empty() {
    verify_cache.begin_linear_prefix_capture(1);   // capture point = after verify_input[0]
}
```

`begin_linear_prefix_capture` (`kv_cache.rs:4604-4614`) and `restore_linear_prefix_checkpoint` (`kv_cache.rs:4650-4669`) capture/restore `conv_state` + `recurrent_state` per linear layer. The captured state is written by the **batched S=2..4 verify forward's first position**, and adopted on full accept (`runner/mod.rs:8895-8918`) or restored on a complete miss (`runner/mod.rs:8919-8941`).

Both operations are correct **only if the batched verify's per-position recurrent state is bit-identical to singleton decode**. That is exactly what the exact profile provides — it de-fuses the linear-attention kernels so S=2..4 per-row arithmetic matches S=1. The invariant is stated directly at `runner/mod.rs:8880-8885`:

> "Without the exact profile, the batched verifier's recurrent linear-attention state can differ from singleton production decode even when every draft token is accepted, so replay remains the fail-closed fallback."

The gate is `linear_mtp_requires_singleton_replay` (`runner/mod.rs:12201-12213`): it forces replay when `!exact_profile_enabled`. The de-fusion sites that create the invariant are e.g. `model/shared/linear_attention.rs:363-369` (skip fused Metal gate under exact) and `model/families/qwen3_linear.rs:89-103` (`fold_exact_attn_norm` for S=2..4). **So decoupling the checkpoint from the exact profile would re-introduce the batched-vs-singleton recurrent-state divergence and break MTP-on/off token equivalence.** The checkpoint *is* the exact-profile optimization; there is no separate mechanism to extract.

#### (b) Is there a cheaper replay? **Not without the invariant — the recurrent state is entangled with the full forward.**

The replay is `recompute_committed_prefix_with_argmax` (`ngram_accel.rs:1403-1424`):

```rust
let mut last_logits = forward_argmax(cfg, weights, &[last_token], cache, token_offset);  // 1 forward
cache.advance(1);
for (index, &token) in accepted_draft.iter().enumerate() {
    last_logits = forward_argmax(cfg, weights, &[token], cache, token_offset + index + 1);  // ac forwards
    cache.advance(1);
}
```

That is **(1 + ac) full S=1 forwards through all 64 layers**, plus one final eval. The recurrent linear-attention state is a per-layer function of that layer's input hidden state, which depends on all prior layers' outputs — so it **cannot be recomputed in isolation**. "Only recompute the recurrent state" would require checkpointing every layer's intermediate hidden, which is a strictly larger state than the recurrent snapshot the exact profile already captures. The dominant cost is the full forward (qmm/FFN/attention projections), not the recurrence update itself.

**Quantified economics (depth-1):**
- Exact/checkpoint path (full accept): 1 batched S=2 verify (~38.9 ms 4-bit) + adopt eval — **no replay**.
- Non-exact/replay path (full accept): 1 batched S=2 verify + **2 singleton S=1 forwards** (~2× 29 ms ≈ 58 ms extra).
- Observed: forcing `AX_MLX_MTP_LINEAR_EXACT_REPLAY=1` drops 4-bit from 39.95 → 18.75 tok/s (**2.13×**), matching the ~2× forward-work estimate.

---

### Q3 — Further Improvement Opportunities

#### [HIGH] The 6-bit 0.82× regression is dense-lm_head cost in the verify, *not* replay — and the fix already exists
**Location:** `fastpath.rs:640-659` (`AX_MLX_EXACT_DENSE_WEIGHT_T_GEMV`) + `fastpath.rs:661-675` (`AX_MLX_MTP_DENSE_HEAD_DRAFT_Q4`).

**Problem:** The 6-bit pack's target `lm_head` is **dense bf16 (2.54 GB)**, unlike the 4-bit AXQ pack's quantized head. The `verify_eval_wall_us` jumps from **9.2 ms (4-bit) → 25.3 ms (6-bit)**, a 2.75× delta that is the entire MTP net loss. The mechanism is documented in the code: the invariant dense kernel read `qw.weight` which `prepare_contiguous_decode_weight_t` replaced with a lazy transpose view, so `ensure_row_contiguous` **re-materialized the 2.54 GB head on every draft + verify call**. The code comment literally names it: *"the whole 0.80× MTP regression"* (draft 18.6 ms, verify-eval 22.9 ms on the 6-bit pack vs 4.1/6.4 ms on the quantized-head sibling).

**Status:** Both fixes are `env_flag_default_on!` in the current tree. The draft side is already visible in the artifact (draft_wall = 3.0 ms/step, down from the pre-fix 18.6 ms). The verify side (`verify_eval` still 25.3 ms) is the remaining gap — worth confirming the weight_t GEMV path actually engages on this pack, since `verify_forward_wall_us` (32.9 ms) being *larger* than `verify_eval_wall_us` (25.3 ms) is itself a red flag that eager host-side weight work still happens during graph build.

**Expected improvement:** if the weight_t GEMV path eliminates the per-call head re-materialization, 6-bit verify_eval should converge toward the 4-bit shape (~9 ms), recovering most of the 0.82× → ~1.1-1.2×.

#### [MEDIUM] `verify_forward_wall_us` (graph build) is suspiciously large — 29.7–32.9 ms/step
**Location:** `runner/mod.rs:8778` (timing around `forward_all_positions_with_post_norm` at `8770` / `forward_all_positions_with_post_norm_ids` at `8761`).

**Problem:** MLX lazy graph construction should be sub-millisecond; 30 ms of "build" time per step means eager work (the `ensure_row_contiguous` re-materialization above) is hiding inside the build. This is the same root cause as the HIGH finding, but worth calling out separately because it affects the 4-bit pack too (29.7 ms build vs 9.2 ms eval).

**Impact:** ~30 ms/step of host-side work that a compiled-closure verify (analogous to the draft head's `build_compiled_mtp_draft` at `mtp.rs:723-809`) would eliminate. This is the largest single target for the *4-bit* path.

#### [LOW] Cache clone is *not* a bottleneck — do not optimize it
**Location:** `runner/mod.rs:8736-8737`.

**Finding (correction to a common hypothesis):** `state.cache.clone()` is measured at `ax_mtp_cache_clone_wall_us` = **222-227 µs *total* (~2.2 µs/step)**. Despite cloning 64 layers of KV + recurrent state, it is negligible. Any "avoid the clone" refactor would be premature.

#### [LOW] Draft-head build is already well-optimized
**Location:** `mtp.rs:1295-1339` (`mtp_draft_tokens_greedy_async`) and the compiled closure at `mtp.rs:723-809`.

The draft path already (a) uses `async_eval` to overlap the draft GPU forward with verify graph build (`runner/mod.rs:8310-8345` chains the verify graph directly on the lazy draft arrays at `8743-8768`), and (b) builds the multi-depth graph with a single terminal `eval` (no per-depth barriers — `mtp.rs:1418-1426`). Draft cost is already only 3-4 ms/step. No action.

---

### Recommendations (Prioritized)

1. **Verify the `AX_MLX_EXACT_DENSE_WEIGHT_T_GEMV` path actually engages on the 6-bit pack** and re-measure — this is the single highest-impact item and is already coded but unconfirmed in the current artifacts (expected: recovers most of the 0.82× 6-bit loss).
2. **Investigate `verify_forward_wall_us` (~30 ms/step build)** for the 4-bit pack — confirms whether eager weight prep is still on the hot path; if so, a compiled verify closure (like the draft head already has) is the next big win.
3. **Correct the documentation/README** that attributes the 6-bit loss to "non-exact replay verify" — the default is Auto-exact, and the actual cause is dense-lm_head cost. (Also regenerate the telemetry from a v7.1.4+ binary so `enabled`/`active` are reported correctly, retiring the `enabled: 0` artifact.)
4. **Leave the cache clone and draft path alone** — both are already cheap/optimized; any work there is premature.

I did not modify any code (read-only analysis). No verification command is applicable since no files changed.