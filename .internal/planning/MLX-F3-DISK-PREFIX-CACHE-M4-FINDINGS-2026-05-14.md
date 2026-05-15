# F3 M4 — Cross-Restart Integration Validation (2026-05-14)

PRD: `MLX-DISK-PREFIX-CACHE-PRD-2026-05-14.md`
Parent PRD: `DS4-LEARNINGS-FOLLOWUP-PRD-2026-05-14.md` §6
Prior milestones:
- `MLX-F3-DISK-PREFIX-CACHE-M1-FINDINGS-2026-05-14.md`
- `MLX-F3-DISK-PREFIX-CACHE-M2-FINDINGS-2026-05-14.md`
- `MLX-F3-DISK-PREFIX-CACHE-M3-FINDINGS-2026-05-14.md`

## 1. Status: **M4 LANDED.** F3 disk prefix cache is now end-to-end validated against cold-prefill correctness across process restarts.

The M4 harness ran on Gemma 4 E2B 4-bit:

| Verdict | Correctness | Disk hits (phase B) |
|---|---|---|
| **PASS** | 2/2 token-exact | 2 (one per prompt) |

Artifact: `benchmarks/results/disk-prefix-cache-cross-restart/gemma4-e2b-2026-05-14.json`.

## 2. Harness shape

New script `scripts/verify_disk_prefix_cache_cross_restart.py`.

Modes:
- `--phase orchestrate` (default): spawns two subprocesses of itself in
  sequence with the same `AX_MLX_PREFIX_CACHE_DIR`, compares the
  captured token streams + telemetry.
- `--phase run-once`: worker mode. Loads the model, runs each corpus
  prompt once, writes a per-phase JSON to `--phase-artifact`.

Each phase loads the model freshly so the L1 in-memory cache cannot
satisfy a hit and the disk layer is exercised end-to-end. Exit codes
follow the bench convention: 0 PASS, 3 correctness, 4 no disk hit
observed on phase B (which would mean the L2 wire-up regressed).

The default 2-prompt corpus uses `--pad-to-block-size 16` because M2
only writes the largest block-aligned snapshot to disk; non-aligned
prompts produce no `.axkv` file.

## 3. Two correctness bugs surfaced and fixed by this milestone

M4 was supposed to be a validation run, but the first invocation
failed `FAIL_CORRECTNESS` (1/2 prompts diverged at index 0 in phase
B despite the disk hit firing). Two distinct bugs were uncovered.

### 3.1 The on-disk format dropped the greedy prefill output token

L1 hits restore three pieces of state:

1. `state.cache = snapshot.cache.clone()`
2. `state.prompt_prefix_tokens = reused_tokens.to_vec()`
3. `state.cached_prefill_output_token = snapshot.greedy_prefill_output_token`

The M2 L2 hit path replicated (1) and (2), but set
`cached_prefill_output_token = None` per the M1/M2 file format which
had no slot for it. The next bug (§3.2) hid this gap as long as
chunked_prefill was still running, but once we sliced the prefill
input to empty (the §3.2 fix), the decode path needed
`cached_prefill_output_token` to carry the L1-equivalent token.

**Fix.** Bumped `disk_prefix_cache.rs::FILE_VERSION` from 1 to 2,
reusing the previously-reserved 4-byte header slot for the prefill
token (sentinel `u32::MAX` = None). Introduced `DiskPrefixCacheEntry
{ payload, prefill_output_token }`; `insert` / `get` operate on the
struct. Runner store-path passes `snapshot_prefill_output_token`;
restore-path sets `state.cached_prefill_output_token =
entry.prefill_output_token`. Version-1 files are silently rejected
as a cache miss (fail-closed). Regression test:
`roundtrip_preserves_prefill_output_token`.

### 3.2 chunked_prefill double-wrote KV when probe over-claimed

The deeper bug. The runner-side probe (extended in M2 to consult
L2) returns the longest block-aligned prefix that has a snapshot.
For cross-restart, the scheduler's block table is empty, so:

- `item.reused_prefix_token_slice` = `[]` (scheduler under-reports)
- `item.input_token_slice` = the full 16-token prompt
- Probe finds the L2 entry → `state.cache.seq_len = 16` after restore
- Prefill mode then calls `chunked_prefill(prompt_tokens=full 16, ...)`
- chunked_prefill writes K/V at `offset = cache.seq_len = 16`, so it
  *appends* a duplicate 16 tokens at positions [16..31]
- The sampled `tok` is now the model's reading of a 32-token effective
  context where the first 16 tokens are repeated → divergence from
  the cold-prefill baseline at every position from token 0 onwards

This bug never surfaced in M2 because:
- The within-process multiturn flow has the scheduler's block table
  populated from turn 1, so `reused_prefix_token_slice` ≥ the
  probe-restored prefix and the duplicate KV write never happens.
- `verify_prefix_reuse_equivalence.py warm_repeat` happens to route
  the second-turn call through paths where the scheduler-tracked
  reuse already covers the duplicated portion.
- M2's smoke evidence (`disk_hits = 0` in within-process multiturn)
  explicitly never exercised the disk-restore path.

**Fix.** At the top of `Prefill` mode in `run()`, after
`restore_reused_prefix_state` returns, compute the "probe over-claim":

```rust
let probe_over_claim = if full_recompute_tokens.is_some() {
    0
} else {
    state
        .cache
        .seq_len
        .saturating_sub(item.reused_prefix_token_slice.len())
};
```

Then slice `prefill_tokens = &prefill_tokens_base[probe_over_claim..]`.
If the slice becomes empty (full-prompt restore), skip chunked_prefill
entirely and emit `state.cached_prefill_output_token` as `tok` —
the same value the producing cold prefill sampled and stored on
disk via §3.1. A defensive fallback runs `decode_one` on the last
reused token if the disk entry was written before M4 (no prefill
token captured); behaviour then matches L1's partial-prefix
recompute path.

This change is intentionally *only* active when probe over-claimed
relative to the scheduler. In the normal multiturn case
(`reused_prefix_token_slice` covers the cached prefix),
`probe_over_claim == 0` and `prefill_tokens == prefill_tokens_base`
— a zero-touch path for the workloads that the M2 evidence
validated.

### 3.3 Regression sweep

Both fixes were independently confirmed with the existing pre-merge
correctness gates:

- `cargo test -p ax-engine-mlx --lib`: **376 / 376 passed.**
- `cargo clippy -p ax-engine-mlx --all-targets --all-features -- -D
  warnings`: clean.
- `verify_prefix_reuse_equivalence.py --mode warm_repeat`
  (gemma-4-e2b-it-4bit, 5 prompts): **5 / 5 PASS.**
- `verify_prefix_reuse_equivalence.py --mode warm_extend`
  (gemma-4-e2b-it-4bit, 5 prompts): **5 / 5 PASS.**

## 4. M4 cross-restart evidence

### 4.1 Standard FA + sliding window — Gemma 4 E2B 4-bit

Single run, 2-prompt corpus, pad_to_block=16:

| Prompt | tokens | tokens_match | disk_hits_b | telemetry_b notes |
|---|---|---|---|---|
| p1_short_factoid | 16 | ✅ true | 1 | hit + re-insert + 0 evictions |
| p2_medium_explain | 32 | ✅ true | 1 | hit + re-insert + 0 evictions |

### 4.2 Hybrid MLA + linear attention — Qwen3.5-9B 4-bit

Same harness, same corpus, same pad_to_block, freshly built ax_engine
extension after the M3B advisory-lock landing:

| Prompt | tokens | tokens_match | disk_hits_b | telemetry_b notes |
|---|---|---|---|---|
| p1_short_factoid | 16 | ✅ true | 1 | hit + re-insert + 0 evictions |
| p2_medium_explain | 32 | ✅ true | 1 | hit + re-insert + 0 evictions |

Artifact: `benchmarks/results/disk-prefix-cache-cross-restart/qwen35-9b-2026-05-14.json`.

Qwen3.5 is the most aggressive disk-cache target in the supported tier
because (a) it uses GatedDeltaNet linear-attention layers whose
recurrent state cannot be rolled back with `trim_to`, (b) its
full-attention layers go through the MLA chunk-alignment safety gate
that M2 explicitly defers L2 hits on when `AX_DISABLE_MLA_PREFIX_RESTORE`
is unset (in this run the gate stays inert because the prompts are
exactly block-aligned). Token-exact equivalence under cross-restart
confirms the L1 alignment gate and the new prefill-token-on-disk
plumbing both translate cleanly to the hybrid path.

### 4.3 Pure MLA — GLM-4.7-Flash 4-bit

Same harness, same corpus, same pad_to_block:

| Prompt | tokens | tokens_match | disk_hits_b | telemetry_b notes |
|---|---|---|---|---|
| p1_short_factoid | 16 | ✅ true | 1 | hit + re-insert + 0 evictions |
| p2_medium_explain | 32 | ✅ true | 1 | hit + re-insert + 0 evictions |

Artifact: `benchmarks/results/disk-prefix-cache-cross-restart/glm47-flash-2026-05-14.json`.

GLM-4.7 is the pure-MLA target: every layer goes through MLA
attention with the kv_latent + k_pe split. The M2 store path explicitly
gates non-FA architectures (linear / sliding-window / MLA) on
block-aligned full-prompt snapshots only, so this run also confirms
that the gate's MLA branch is sound under cross-restart, not just
in-process.

All phase-B telemetry rows across the three runs show `disk_hits=1`
confirming the L2 restore actually fired (the previous M2 smoke ran
the same prompt within a single process and never exercised the
restore path).
Phase-B also re-inserts each snapshot after the model continues to
generate, which is expected behaviour: the runner refreshes the disk
entry from the post-restore cache state, so the entry stays warm
even when the same prompt is served by many process restarts.

Phase-A vs phase-B output token streams are byte-for-byte identical
across the covered architecture tiers, which is the M4 acceptance
criterion per PRD §8.2.

## 5. What this *doesn't* prove

- **Full AX-serving soak.** The M3C stress artifact covers the cache
  primitive with four processes and tight eviction pressure, but it
  does not load four full model-serving processes for a long soak.
- **Long-context / perf pressure.** M4 proves correctness and disk-hit
  wire-up, not the final long-context TTFT promotion numbers.

## 6. Files

| Path | Change |
|---|---|
| `crates/ax-engine-mlx/src/disk_prefix_cache.rs` | File version 1→2; `DiskPrefixCacheEntry { payload, prefill_output_token }`; `insert`/`get` take/return the struct; prefill-token slot in header (sentinel u32::MAX); regression test `roundtrip_preserves_prefill_output_token`. |
| `crates/ax-engine-mlx/src/runner.rs` | Slice `prefill_tokens` by `probe_over_claim` so chunked_prefill doesn't double-write KV; empty-slice path emits `cached_prefill_output_token`; restore-path now reads `entry.prefill_output_token`. |
| `scripts/verify_disk_prefix_cache_cross_restart.py` | New M4 harness: orchestrator + run-once worker. |
| `benchmarks/results/disk-prefix-cache-cross-restart/gemma4-e2b-2026-05-14.json` | M4 PASS evidence on Gemma 4 E2B (standard FA + sliding window). |
| `benchmarks/results/disk-prefix-cache-cross-restart/qwen35-9b-2026-05-14.json` | M4 PASS evidence on Qwen3.5-9B (hybrid MLA + linear attention). |
| `benchmarks/results/disk-prefix-cache-cross-restart/glm47-flash-2026-05-14.json` | M4 PASS evidence on GLM-4.7-Flash (pure MLA). |

## 7. Closure conditions for this M4 artifact

- ✅ Cross-restart correctness: phase A vs phase B token streams
  byte-equal across the corpus.
- ✅ `ax_mlx_prefix_cache_disk_hits` fires ≥ 1 in phase B.
- ✅ Two correctness bugs found by the harness, both fixed in this
  milestone.
- ✅ Existing pre-merge gates (`cargo test`, clippy,
  `verify_prefix_reuse_equivalence.py` both modes) all green.
- ✅ Harness checked in and reusable; takes ~3 minutes wall-clock
  for two prompts on Gemma 4 E2B.

PRD §6 row in the parent ledger should update from "M3 landed; M4
open" to "M4 architecture-tier validation landed; F3 docs/promotion
open".

---

**Status:** M4 closed. F3 disk prefix cache is end-to-end validated
across Gemma 4 E2B, Qwen3.5-9B, and GLM-4.7-Flash cross-restart.
After the M3C stress artifact, the remaining follow-up surface area
is full AX-serving soak / promotion and M5 docs.
