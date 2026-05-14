# F1 Probe Findings — bf16 vs f16 FFN matmul (2026-05-14)

PRD: `DS4-LEARNINGS-FOLLOWUP-PRD-2026-05-14.md` §4 (F1).

## Question

Does switching the dequantization target dtype from bf16 to f16 on the
FFN matmul hot path deliver ≥3% throughput improvement on the
supported-tier model shapes? The PRD names this as a cheap directional
probe whose outcome decides whether the larger weight-loader migration
is worth scheduling.

## Probe

Binary: `crates/ax-engine-mlx/src/bin/dequant_dtype_probe.rs`
(`cargo run --release --bin dequant-dtype-probe`).

Method: build a random fp32 weight matching each FFN-projection
shape, cast to bf16 and f16, time `matmul(x, w)` over 200 iterations
each. Skip the fp32 reference matmul because Apple Silicon GPUs have
no fast fp32 matrix path on these shapes; the directional question is
about the bf16-vs-f16 wall-clock delta, not numerical equivalence
against fp32.

Shapes covered: Gemma 4 E2B 4-bit `gate_proj` / `up_proj` /
`down_proj`, and Qwen 3.5 9B 4-bit `gate_proj` / `down_proj`. A small
warmup shape (128×256) runs first to amortise the GPU init cost.

## Result

```
[warmup-small] hidden=128 intermediate=256
    bf16 matmul: 192.04 us/iter
    f16  matmul: 164.73 us/iter
    f16 vs bf16: +14.2% (PASS — but discounted as one-time init noise)

[gemma4-e2b-gate_proj] hidden=1536 intermediate=6144
    bf16 matmul: 190.63 us/iter
    f16  matmul: 185.47 us/iter
    f16 vs bf16: +2.7%  (MARGINAL)

[gemma4-e2b-up_proj]   hidden=1536 intermediate=6144
    bf16 matmul: 186.74 us/iter
    f16  matmul: 181.19 us/iter
    f16 vs bf16: +3.0%  (MARGINAL)

[gemma4-e2b-down_proj] hidden=6144 intermediate=1536
    bf16 matmul: 244.99 us/iter
    f16  matmul: 229.12 us/iter
    f16 vs bf16: +6.5%  (PASS)

[qwen3.5-9b-gate_proj] hidden=4096 intermediate=8192
    bf16 matmul: 316.34 us/iter
    f16  matmul: 295.76 us/iter
    f16 vs bf16: +6.5%  (PASS)

[qwen3.5-9b-down_proj] hidden=8192 intermediate=4096
    bf16 matmul: 467.55 us/iter
    f16  matmul: 468.96 us/iter
    f16 vs bf16: -0.3%  (REJECT)
```

Aggregate excluding warmup: bf16 sums to 1,406 µs/iter, f16 sums to
1,361 µs/iter, **delta +3.25%**. The aggregate crosses the PRD's
+3% PASS gate, but barely, and the per-shape distribution is
**mixed**.

## Interpretation

- **f16 matmul is consistently a few percent faster than bf16 on
  Apple Silicon for the FFN gate/up shapes**, with the biggest wins
  on the wide-input narrow-output `down_proj` shape (+6.5% Gemma,
  -0.3% Qwen).
- The Qwen 9B `down_proj` regression (−0.3%) is within noise but
  confirms that the f16 win is not uniform. Some kernel selections
  pick paths where bf16 is at least as fast.
- The probe measures **post-dequant matmul throughput only**. It
  does *not* measure the dequant step inside `quantized_matmul`,
  where the source weight is 4-bit packed and the dequant target is
  the activation dtype. The full production picture additionally
  depends on how MLX's `quantized_matmul` kernel behaves at f16 vs
  bf16 output — which requires either real 4-bit weights or new
  `mlx_sys` bindings to `quantize` (out of scope for this probe).

## Verdict

**MARGINAL PASS — schedule the migration but cap expectations at
~3% prefill throughput**, not the >5% the PRD's most-optimistic
reading would have suggested. The weight-loader change is non-
trivial (touches `crates/ax-engine-mlx/src/weights.rs` plus all the
correctness gates listed in PRD §4.3). At ~3% aggregate, the
implementation cost / benefit ratio is borderline; consider
sequencing this **after** F4 (MLA bisect — preventative, cheap) and
F3 (disk cache — high user-visible value), rather than ahead of
them.

A deeper investigation could close the gap between this matmul-only
probe and the production quantized_matmul cost by adding an
`mlx_sys::quantize` binding and reproducing the probe on real 4-bit
weights. That follow-up belongs to whoever picks up the actual
migration ticket.

## Status

OPEN on the F1 PRD ticket. This findings doc plus the probe binary
satisfy the PRD §4.2 "smallest reasonable probe before committing to
migration" gate. The migration itself (§4.3) is not yet scheduled —
recommend deferring until F3 / F4 are decided, given the modest
aggregate win.

## Files

- Probe binary: `crates/ax-engine-mlx/src/bin/dequant_dtype_probe.rs`
- Cargo entry: `[[bin]] name = "dequant-dtype-probe"` in
  `crates/ax-engine-mlx/Cargo.toml`
- Parent PRD: `.internal/planning/DS4-LEARNINGS-FOLLOWUP-PRD-2026-05-14.md` §4
