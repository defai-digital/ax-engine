# DS4-Learnings Follow-up PRD

Status: Open — Phase A/B/C closed in this cycle, five named follow-ups
defined for the next.
Date: 2026-05-14
Owner: AX Engine
Depends on: DS4-REFERENCE-LEARNINGS-PRD.md, KV-SCHEDULER-W1-EVIDENCE-REPORT.md, MLX-PREFILL-IMPROVEMENT-PRD.md
Related findings (this cycle):

- `MLX-FASTPATH-AUDIT-2026-05-14.md`
- `MLX-PHASE-B-CUSTOM-KERNEL-SPIKE-2026-05-14.md`
- `MLX-PHASE-B-BATCH-MICROBENCH-2026-05-14.md`
- `MLX-PHASE-B1-SIDECAR-KERNEL-FINDINGS-2026-05-14.md`
- `MLX-PHASE-B-GRAPH-SYNC-AUDIT-2026-05-14.md`
- `MLX-PHASE-C-MULTITURN-BASELINE-PLAN-2026-05-14.md`
- `MLX-PHASE-C-MULTITURN-BASELINE-FINDINGS-2026-05-14.md`

## 1. Summary

The 2026-05-14 DS4-learnings cycle landed three concrete improvements
and two NO-GO closures:

| Phase | Outcome | Headline |
|---|---|---|
| A | landed | `fastpath` env-flag framework + audit doc; `AX_DISABLE_TURBOQUANT_FUSED_DECODE` + `AX_NO_SPEC` uniform handling |
| B.0 | landed | kernel-chain batching probe — MLX `STRONG_FUSION` ratio 10.4× |
| B.1 | NO-GO | sidecar Metal kernel for post_attn / FFN hotspot loses by ~1% vs MLX-native (`rmsnorm-fused-probe`, `residual-rmsnorm-fused-probe`) |
| B graph-sync audit | NO-GO | 1 high-level `eval()` per decode step already at minimum; "150 syncs/step" is MLX-internal CB count |
| C | landed | runner-side probe + MLA prefill-chunk alignment; FA/SWA/Linear/MLA all reach steady-state iterative-chat TTFT (`growth_ratio ≈ 0.9–1.0`) |

The remaining DS4-learnings-adjacent investment falls into five
independent axes, each a multi-day deliverable. This PRD scopes each so
they can be picked up in any order based on user priority.

## 2. Decision Summary

| Item | Expected value | Risk | Effort | Section |
|---|---|---|---|---|
| F1. F16 weight path investigation | medium–high (one-shot prefill throughput on Mac) | medium (dtype migration, quant interop) | multi-day | §4 |
| F2. Larger MLA prefill chunk policy (per-request alignment) | medium (recover ~5× cold-prefill cost on MLA) | low–medium (touches scheduler boundary) | 2–3 days | §5 |
| F3. Durable disk prefix cache (Phase D) | high (cross-process / cross-restart hot-prefix reuse) | high (eviction, integrity, mmap policy) | week-scale | §6 |
| F4. Per-layer MLA Q/K/V bisect tooling | low–medium (closes the chunk-alignment workaround properly) | low (read-only instrumentation) | 2–3 days | §7 |
| F5. Server-side concurrent-prefill batching | very high (multi-request throughput) | high (scheduler + tokio rewiring) | week-scale | §8 |

§3 records what does *not* belong in this PRD's scope.

## 3. Out of scope

- **Sidecar Metal kernels for the post_attn / FFN hotspot.** Closed
  NO-GO this cycle (§Phase B.1 findings). Do not re-open without a
  fundamentally different angle (e.g. a fused quantized matmul kernel
  that Apple has not already published).
- **Removing high-level `eval()` calls from the decode loop.** Already
  at one per step (§Phase B graph-sync audit findings).
- **Replacing MLX `quantized_matmul` for the FFN projections.** Apple's
  kernel is already heavily optimised; this is the same NO-GO surface
  as B.1, scaled up.
- **Wiring the `metal/phase1-kernels.json` registry into the forward
  path.** CLAUDE.md gate: profiling evidence required first. The Phase
  B findings indicate that the evidence will not appear; the registry
  remains a reference, not a deployable.

## 4. F1 — F16 weight path investigation

### 4.1 Problem

Memory entry `project_perf_work` records two latent gaps after the
2026-05-01 serial-thread kernel fixes:

- F16 weights (vs the current bf16 dequant output target);
- 150 GPU syncs/step (now diagnosed as MLX-internal CB count, see §3).

The F16 hypothesis: Apple Silicon GPUs have native f16 matrix
hardware that may dispatch faster than bf16 for the post-dequant ops
on the FFN matmul path. The current path dequantises 4-bit weights to
bf16 by default. Switching the dequant target to f16 (where safe)
could deliver a measurable prefill throughput win without changing the
quant format on disk.

### 4.2 Pre-investigation probe

Smallest reasonable probe before committing to migration:

```text
crates/ax-engine-mlx/src/bin/dequant_dtype_probe.rs (proposed)
```

Mirror the `rmsnorm-fused-probe` pattern: one `quantized_matmul` call
with bf16 output, one with f16 output, same weights, same input, time
both. Decide based on:

- PASS (≥3% f16 win, bit-tolerance ≤ 1e-3 vs bf16 reference) → schedule
  weight-loader migration (§4.3);
- MARGINAL (<3%) → close NO-GO, document;
- REJECT (f16 slower or numerically unsafe) → close NO-GO immediately.

Bench shapes to cover: gemma4-e2b (`hidden=1536`, `intermediate=6144`)
and qwen3.5-9b (`hidden=4096`, `intermediate=8192`) gate / up / down
projections.

### 4.3 If probe is PASS

Weight loader change scope:

- `crates/ax-engine-mlx/src/weights.rs` — dequant output dtype is the
  load-time choice; thread an MLA / non-MLA / per-op switch.
- Validate quant correctness via the existing
  `verify_prefix_reuse_equivalence.py` and the embedding correctness
  harness; numerics tolerance bumps must be argued, not silently
  widened.
- Run `profile_kv_long_context_evidence.py` and the public README
  bench corpus to confirm no regression.

### 4.4 Decision gate

Open ticket only if the §4.2 probe PASSes. Skip otherwise; this is
explicitly a measurement-gated investment.

## 5. F2 — Larger MLA prefill chunk policy

### 5.1 Problem

The Phase C MLA fp-drift fix (commit `ade74c2f`) defaulted
`prefill_chunk` to 16 for MLA models. This eliminated the SDPA-kernel
shape mismatch but increases cold-prefill cost: GLM-4.7-Flash 2,048-
token cold prefill went from ~0.79 s to ~5.10 s (≈6.5×).

For chat workloads this is recovered by turn-2-onwards snapshot reuse
(net win by ~3 turns). For one-shot / batch users it is pure
regression. `AX_MLX_MLA_PREFILL_CHUNK=512` lets them opt out, but
losing warm-extend correctness is a steep price.

### 5.2 Hypothesis

The chunk-alignment requirement is: `chunk_size` divides `base_len`.
Stored snapshots are guaranteed `block_size_tokens`-aligned (16), so
`chunk_size = 16` is universally safe. But for the common case where
`base_len ≡ 0 mod 2^k` (e.g. 2,048-aligned), larger chunk sizes also
satisfy alignment and recover cold-prefill throughput.

Concretely: `base_len = 2048` is divisible by 16, 32, 64, …, 2048.
Cold and warm paths both produce the same SDPA shape sequence at any
divisor. If we know `base_len` at runtime, we can pick the largest
chunk size that divides it.

### 5.3 Design sketch

Move chunk-size selection from `from_artifacts` (per-process) to
per-request:

- `MlxRunner` keeps an `mla_max_prefill_chunk` (default 16, env-tunable)
  and a `mla_chunk_pow2_cap` (default 8 = 256-token max).
- `restore_reused_prefix_state` knows `reused_tokens.len()`; if the
  restore fires, it picks `chunk_size = largest 2^k ≤ mla_chunk_pow2_cap
  that divides reused_tokens.len() and the suffix length`. The
  selected chunk size is passed through to `chunked_prefill` for that
  request.
- For a cold prefill with no snapshot, fall back to the existing 16
  default — the safety constraint still holds in case a snapshot is
  stored at the end of this prefill and re-used later.

Per-request chunk size requires plumbing through the existing
function-level threading of `prefill_chunk`. Touches
`runner.rs:run_item`, `generate.rs:chunked_prefill`, and the prefill
metadata helpers.

### 5.4 Success criteria

- `verify_prefix_reuse_equivalence.py --mode warm_extend
  --pad-to-block-size {16, 512, 2048}` all 5/5 PASS with the new
  policy.
- `profile_kv_long_context_evidence.py` on GLM-4.7-Flash shows cold
  prefill within +/-10% of the pre-Phase-C baseline (~0.79 s).
- `profile_kv_multiturn_chat_evidence.py` on GLM-4.7-Flash retains
  `ttft_growth_ratio ≤ 1.0` over 10 turns.

### 5.5 Decision gate

Worth doing if cold-prefill speed matters to the planned product
surface (one-shot batch APIs, evals, README rows). Skip if the only
target is chat / agent workloads — Phase C already delivers there.

## 6. F3 — Durable disk prefix cache (Phase D)

### 6.1 Problem

The runner-side `MlxPrefixCache` is process-local. It does not
survive process restart or share across replicas. For server / agent
deployments where the same system prompt is re-used across many
sessions and process restarts, this leaves the largest possible TTFT
win on the floor.

DS4's `ds4_server.c` disk KV (key = SHA1 of rendered prompt bytes,
payload = exact token IDs + graph state) is the documented reference
shape.

### 6.2 Design sketch (initial only — full design is the deliverable)

- Key: `MlxPrefixCacheKey` already exists; persist its hash on disk
  alongside the payload.
- Payload: serialized `MlxKVCache` snapshot, including all layer KV
  tensors and metadata (capacity, dtype, rotating window state if
  any).
- Eviction: LRU over total bytes, file mtimes for tie-break.
- Integrity: payload checksum at load; reject on mismatch (do not
  silently degrade).
- Concurrency: file-lock on entry add/evict; readers can proceed
  without locks because entries are immutable after write.
- mmap: deliberately avoided per DS4 rationale (the engine already
  mmaps a large GGUF / safetensors). Re-validate this choice against
  AX's memory model before adoption.

### 6.3 Pre-conditions

Do not start this work until:

- `MlxPrefixCache` is exposed via a stable Rust trait (currently
  inline in `runner.rs`);
- Snapshot serialization for at least the FA + SWA architectures has
  a unit-tested round-trip; MLA's compressed-latent format will need
  its own format negotiation.

### 6.4 Success criteria

- A multi-process / restart-survival test: process A serves prompt P,
  exits; process B serves prompt P again on the same model artifacts,
  TTFT comparable to in-process warm_repeat.
- Cache file size bounded by the policy; eviction observable in the
  same telemetry namespace (`ax_mlx_prefix_cache_*`).
- `verify_prefix_reuse_equivalence.py` runs cleanly with the disk
  layer in front (no token drift).

## 7. F4 — Per-layer MLA Q/K/V bisect tooling

### 7.1 Problem

Phase C's MLA fix is a *workaround*: aligning the prefill chunk size
to 16 keeps cold and warm-extend on the same SDPA kernel, but it does
not explain *which* MLX SDPA shape-pair behaviour caused the original
fp-drift. The chunk-alignment workaround is safe today but is one
MLX-version away from being silently insufficient.

### 7.2 Goal

Build a debug bin that, for a given model + prompt + suffix:

- Runs cold prefill of `base + suffix` and saves `kv_latent[layer, position]`
  for each layer and each position to disk.
- Runs warm prefill (base, snapshot, restore, suffix) and saves the
  same.
- Diffs the two arrays element-wise per layer, reports the first
  layer / position with a non-zero diff at f16/bf16 ulp tolerance.

### 7.3 Design sketch

```text
crates/ax-engine-mlx/src/bin/mla_warm_extend_drift_probe.rs (proposed)
```

Borrow the loading machinery from `bench_main.rs`. Add per-layer
`tracing::trace`-style hooks gated by an env flag so the production
forward path is unaffected. Expected output:

```
layer 0:  q_nope diff max = 0.000000 (PASS)
          kv_latent diff max = 0.000000 (PASS)
          k_pe diff max = 0.000000 (PASS)
layer 1:  q_nope diff max = 0.000000 (PASS)
          kv_latent diff max = 4.2e-5 (FIRST DIVERGENCE)
          ...
```

### 7.4 Success criteria

Identifies the first divergent (layer, position, tensor) pair on
`p2_medium_explain` for the historical chunk=512 path. Once
identified, the next ticket can target the specific op (RoPE,
projection, etc.) that produces the drift, and either fix it inside
ax-engine or file a reproducer against MLX upstream.

### 7.5 Decision gate

Open only if the chunk-alignment workaround starts misbehaving on
a future MLX upgrade or a new MLA model family. Today the workaround
is sound; bisect-tool work is preventative.

## 8. F5 — Server-side concurrent-prefill batching

### 8.1 Problem

The current `ax-engine-server` (Axum) serialises prefill through one
runner instance. For server / multi-tenant deployments, prefill of
request N+1 blocks behind request N. This is the single largest
gap between AX and competitive serving stacks (vLLM, llama.cpp's
server batch mode).

This is the largest single-investment item in this PRD. It is named
here so the decision to defer is explicit, not accidental.

### 8.2 Pre-conditions

Do not start before:

- F2 (larger MLA prefill chunk) is decided one way or the other —
  batching policy interacts with chunk size.
- F3 (durable disk cache) is decided — disk-cache hits change the
  scheduler's prefill picture.

### 8.3 Design sketch (one paragraph; full design is the deliverable)

Two-level batching: (a) request-level (concurrent requests share a
queue; the scheduler picks N prefills to fuse into one larger forward
pass per step); (b) token-level (continuous batching where ready-to-
decode requests are interleaved with prefilling ones). DS4's
`ds4-server` deliberately does **not** do this; this is a divergence
point, not a port. The MLX runtime's lazy graph + chunked prefill
naturally supports interleaving — the missing piece is the scheduler
contract that says when to merge vs serialise. The right reference
is vLLM's PagedAttention block table + scheduler, not DS4.

### 8.4 Decision gate

Worth doing only if the planned product surface is a multi-tenant
server (not an on-device single-user agent). For on-device, F2 + F3
deliver most of the same user-visible improvement at a fraction of
the engineering cost.

## 9. Sequencing recommendation

Prefer ordering that resolves cheap measurements before expensive
implementations:

1. **F1 probe (one day)** — empirical decision on F16 weights. Cheap
   to run; large branch in the plan depending on outcome.
2. **F4 (2–3 days)** — bisect tooling exists to harden F2 against
   future MLX upgrades. Read-only instrumentation, low risk.
3. **F2 (2–3 days)** — chunk-policy refactor. Picks up cold-prefill
   throughput on MLA. Internal change, no contract impact.
4. **F3 (week-scale)** — durable disk cache. Real product feature.
5. **F5 (week-scale)** — server batching. Largest single bet;
   schedule only after F2 + F3 narrow the design space.

## 10. Closure conditions

This PRD closes when:

- F1 is decided one way or the other with a probe artifact;
- F2 is either landed (with the §5.4 evidence) or explicitly deferred
  with a recorded rationale;
- Each of F3, F4, F5 has its own follow-up PRD (split off from this
  one) or an explicit "not pursuing" decision recorded.

The five items are independent. Closing this PRD does not require
all five to ship; it requires each to have a recorded decision.

## 11. What this PRD is not

Not a roadmap for raw decode throughput. The Phase B NO-GOs in this
cycle establish that decode kernel and sync-count work has reached
its local minimum given the current MLX-native algorithm
decomposition. The items here all live above (workload, server) or
beside (weight format, MLA-specific algorithm) that floor.

---

**Author note (2026-05-14):** the Phase C MLA fix in `ade74c2f` is the
largest user-visible deliverable of this cycle. F2 above is the most
direct way to reclaim its cold-prefill cost without giving up the
warm-extend correctness it unlocks. Recommend F1 probe + F2
implementation as the smallest credible next slice.
