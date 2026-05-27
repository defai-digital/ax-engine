# Low-Bit MLX Quantization Implementation Plan

**Date**: 2026-05-27
**PRD**: `.internal/prd/PRD-2026-05-27-experimental-low-bit-mlx-quantization.md`
**ADR**: `.internal/adr/ADR-005-experimental-low-bit-mlx-quantization.md`
**Session focus**: direct mode and n-gram mode. MTP is deferred except for
keeping benchmark labels unambiguous.

---

## Source Comparison Summary

### MLX

- `QuantizedLinear` stores `bits`, `group_size`, and `mode`, then passes them to
  `mx.quantized_matmul`.
- Metal quantized kernels support affine bits `{2, 3, 4, 5, 6, 8}`.
- 3-bit uses non-power-of-two packing and unpacking, so lower bandwidth does not
  guarantee proportional speedup.

### mlx-lm

- Plain affine quantization defaults to `(group_size=64, bits=4)`.
- Mixed recipes are the important low-bit comparison point:
  - `mixed_2_6`: 2-bit low layers, 6-bit selected sensitive layers;
  - `mixed_3_4`: 3-bit low layers, 4-bit selected sensitive layers;
  - `mixed_3_6`: 3-bit low layers, 6-bit selected sensitive layers;
  - `mixed_4_6`: 4-bit low layers, 6-bit selected sensitive layers.
- The predicate keeps higher bits for selected `v_proj`, `v_a_proj`, `v_b_proj`,
  `down_proj`, and `lm_head` layers.

### AX Current State

- Manifest validation accepts affine bits `[4, 5, 6, 8]`.
- Packed-column validation already computes expected packed columns from the
  tensor bit width.
- Direct mode is a target-model decode path and should show the cleanest low-bit
  effect.
- N-gram mode drafts from token history and verifies with the target model; low
  bits affect verifier/fallback cost, not draft quality.
- MTP must be split by component: main model, sidecar, draft-only LM head, and KV
  compression.

### Rapid-MLX Learnings For This Session

- Direct mode: Rapid-MLX points to cache reuse, state snapshots, prefill chunking,
  and MLX compute as the useful speed surfaces. It is not evidence that rewriting
  AX's host scheduler will materially improve single-stream direct decode.
- Direct mode: Rapid-MLX's hybrid state snapshots are relevant to repeated-prompt
  TTFT and cache hit behavior. AX already serializes full-attention, MLA, and
  linear-attention state, so the next step is measuring hit-rate and TTFT impact,
  not introducing a new cache format first.
- N-gram: Rapid-MLX SuffixDecoding is default-off, greedy-only, single-request,
  and workload-tiered. AX should copy that promotion posture: no global n-gram
  low-bit claim without accepted draft tokens on the relevant workload tier.
- N-gram: Rapid-MLX skips too-short drafts and cools down after repeated
  zero-accept verifies. AX already has no-draft reasons, cooldowns, adaptive
  draft length, and linear-attention guards; the missing piece is a benchmark
  matrix that turns those counters into promotion tiers.
- DFlash: Rapid-MLX's DFlash numbers are useful as a warning, not an immediate
  AX implementation target. It is alias-gated, workload-sensitive, and rejects
  MoE/low-bit shapes for v1.

## Best-Practice Implementation Rules

- Treat direct mode as the root baseline. No n-gram, MTP, cache, or TurboQuant row
  can prove a low-bit model-weight win by itself.
- Keep mechanisms separate in metadata: weight bits, prefix/state cache reuse,
  prefill step size, n-gram accepted tokens, and fallback route counters.
- Benchmark cold and warm cache paths separately. A warm TTFT win must not be
  summarized as decode throughput.
- Use mixed 3-bit recipes first. Do not start with all-3-bit or all-2-bit
  artifacts unless mixed recipes already pass direct and quality gates.
- Use workload tiers for n-gram promotion. Report free-form chat, structured
  JSON/tool-loop, code edit/input-output overlap, and long repeated context
  separately.
- Require token IDs or checksums for direct and n-gram rows so quality and
  divergence review can be repeated.
- Keep Qwen linear-attention n-gram conservative until local artifacts prove the
  draft source is useful: repeated evidence, bounded draft length, cooldown after
  misses, and direct fallback are required.

## Slice 1: Experimental Manifest Gate

Files:

- `crates/ax-engine-core/src/model.rs`
- existing manifest validation tests near the core model manifest tests

Tasks:

- Add an experimental gate for 3-bit affine quantization.
- Keep 2-bit gated separately and disabled by default.
- Accept 3-bit only for affine mode and supported group sizes.
- Keep production error messages fail-closed and explicit.
- Add tests for:
  - 3-bit rejected without gate;
  - 3-bit accepted with gate;
  - 2-bit rejected without its stronger gate;
  - mixed per-tensor quantization with 3-bit low layers and 4-bit sensitive layers;
  - packed-column calculation for 3-bit odd packing.

Validation:

```bash
cargo test -p ax-engine-core
```

## Slice 2: Loader Smoke And Shape Validation

Files:

- `crates/ax-engine-mlx/src/weights.rs`
- `crates/ax-engine-mlx/src/runner.rs`
- focused loader tests or model-artifact smoke harness

Tasks:

- Verify the loader does not assume power-of-two bit packing.
- Add route metadata for low-bit experimental mode if missing.
- Emit quantization summary fields in benchmark artifacts:
  - min bit width;
  - max bit width;
  - per-bit tensor counts;
  - low-bit experimental gate state.
- Ensure unsupported 2-bit/3-bit manifests fail before weight loading if the gate
  is absent.

Validation:

```bash
cargo test -p ax-engine-mlx
```

## Slice 3: mlx-lm Artifact Intake

Tasks:

- Convert or import Gemma E2B and Qwen A3B artifacts with mlx-lm `mixed_3_4`.
- Record exact commands in the benchmark artifact metadata.
- Generate AX manifests without committing model files.
- Inspect manifest summaries before benchmarking:
  - tensor count by bit width;
  - group size by bit width;
  - `lm_head` bit width;
  - sensitive projection bit widths.

Recommended first conversion shape:

```bash
python -m mlx_lm.convert \
  --hf-path <source-model> \
  --mlx-path <local-output-dir> \
  --quantize \
  --q-group-size 64 \
  --quant-predicate mixed_3_4
```

If the installed mlx-lm CLI uses different flag names, record the exact working
command in the artifact metadata and keep the recipe equivalent to `mixed_3_4`.

## Slice 4: Direct Decode Benchmark Gate

Tasks:

- Run direct greedy baselines with n-gram disabled.
- Compare 4-bit baseline vs 3-bit mixed artifact for:
  - Gemma E2B;
  - Qwen A3B or Qwen 27B linear-attention target;
  - at least prompt lengths 128, 512, and 2048;
  - generation length 128.
- Capture output token IDs for divergence review.
- Record prefix-cache and state-cache fields when available:
  - exact/shorter/longer prefix hit state;
  - restored token count;
  - TTFT with and without cache hit;
  - linear-attention state bytes.
- Compare prefill-step settings before changing defaults:
  - current default;
  - a smaller fairness-oriented step;
  - a larger throughput-oriented step.
- Emit or preserve route fields that distinguish:
  - native direct path;
  - direct fallback path;
  - unsupported CPU or delegated fallback;
  - linear-attention direct hotpath hit/miss when applicable.
- Capture token IDs or deterministic token checksums for every measured row.

Required artifact fields:

- model source and manifest path;
- quantization bit summary;
- prompt hash and seed;
- route metadata;
- cold TTFT;
- warm TTFT when prefix/state cache is used;
- direct decode tok/s;
- e2e tok/s;
- prefill step size;
- prefix/state cache hit type and restored token count;
- fallback counters;
- token IDs or token checksum;
- git state.

Promotion condition:

- 3-bit direct decode is faster than the matching 4-bit row outside normal noise;
- generated output does not show unacceptable divergence for the target prompt
  suite;
- no CPU fallback or unsupported route counter appears.
- repeated-prompt rows show whether any TTFT win came from low-bit weights,
  prefix/state cache reuse, or both.

Implementation notes:

- If benchmark artifacts do not already expose cold/warm TTFT, cache hit type,
  restored token count, or prefill step size, add those fields before running the
  model matrix.
- Keep route metadata stable enough that future README/performance scripts can
  reject rows with unsupported fallback automatically.

## Slice 5: N-gram Effective Throughput Gate

Tasks:

- Run n-gram rows only after direct mode passes.
- Compare against the matching direct same-policy baseline.
- Include both:
  - random-token mlx-lm-compatible contract;
  - input-output-overlap prompt suite.
- Record `ax_decode_claim_mode`, `ax_decode_claim_status`, accepted draft tokens,
  rejected draft tokens, cooldowns, and no-draft reasons.
- Record direct fallback token count and direct fallback wall time so rows that
  only benefit from faster direct fallback are not mislabeled as n-gram wins.
- Add a Rapid-MLX-style workload tier matrix:
  - free-form chat;
  - structured JSON/tool-loop;
  - code edit / input-output overlap;
  - long repeated context.
- Skip promotion for rows where acceleration came only from lower direct fallback
  cost and not from accepted n-gram drafts.
- For Qwen linear-attention models, keep the existing conservative gating:
  repeated n-gram evidence before drafting, no-draft request disablement, and
  direct fallback re-entry only when output builds a draft source.

Promotion condition:

- claim status is `ngram_acceleration_effective_throughput`;
- accepted tokens are non-zero and meaningful for the row;
- token-exact same-policy gate passes where required;
- fallback rows remain labeled as fallback and are not summarized as acceleration.
- workload tier is `structured_or_overlap_win` or equivalent, not a global
  all-prompts claim.

Implementation notes:

- Add a small benchmark summarizer that maps n-gram counters into one of:
  - `effective_throughput`;
  - `no_draft_fallback`;
  - `zero_accept_fallback`;
  - `direct_fallback_cost_reduction`;
  - `regression_or_neutral`.
- Keep the summarizer mechanical. It should read existing artifact fields and
  produce labels, not infer success from model names or expected behavior.
- For stochastic sampling, either keep n-gram out of the promoted matrix or add a
  verifier contract that proves the sampling distribution is preserved.

## Slice 6: Deferred MTP Component Gate

Deferred tasks:

- Test MTP with 3-bit main model and unchanged MTP sidecar.
- Separately test draft-only LM head lower-bit support if implemented.
- Separately test MTP sidecar low-bit manifests only after manifest and loader
  support exists.
- Keep TurboQuant KV tests separate from weight quantization.

Required labels:

- `main_model_bits`;
- `mtp_sidecar_bits`;
- `draft_lm_head_bits`;
- `kv_compression_mode`;
- MTP depth and draft sampler;
- accept/reject statistics.

Promotion condition:

- AX row is compared to a matching direct or MTPLX reference row with the same
  model component definitions;
- accepted-token and output-quality evidence supports the claim;
- no component-mixed row is presented as a generic 3-bit MTP win.

This slice is explicitly out of scope for the current direct/n-gram session.

## Slice 7: 2-bit Follow-Up Gate

Tasks:

- Do not start until 3-bit direct and n-gram gates have completed.
- Use `mixed_2_6` first.
- Require stricter token-output and quality review than 3-bit.
- Keep 2-bit behind an explicit gate and out of default server paths.

Promotion condition:

- separate ADR or ADR update accepts 2-bit support beyond experiment-only status.

## Suggested Initial Command Set

```bash
cargo test -p ax-engine-core
cargo test -p ax-engine-mlx
cargo test -p ax-engine-bench
bash scripts/check-bench-doctor.sh
```

Before any README or public performance update:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --quiet --no-fail-fast
```
