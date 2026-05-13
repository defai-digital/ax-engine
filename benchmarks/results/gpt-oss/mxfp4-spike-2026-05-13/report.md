# MXFP4 Dequant Correctness Spike — Phase B W1

Date: 2026-05-13
PRD: `.internal/planning/GPT-OSS-PHASE-B-NATIVE-PRD.md` (W1 — V1 spike)
Early-bird exception: PRD §1.5 allowed pulling this work into 4.8.x cycle.
Outcome: **PASS** (exact bit-equal match, exceeds PRD bar of `max_abs_diff < 1e-3`).

## 1. What was tested

Independent safetensors → sanitize → `mx.dequantize(..., mode='mxfp4')`
loader path versus mlx-lm's loaded weights on `openai/gpt-oss-20b`:

- **Samples**: 48 tensors = 4 random layers × 4 random experts × {gate_proj, up_proj, down_proj}.
  PRD asked for 16; ran 3× as the per-(layer,expert) cost is identical.
- **Random seed**: 1234. Layers `[0, 2, 3, 14]`, experts `[2, 5, 6, 22]`.
- **Pass bar**: `max_abs_diff < 1e-3` (BF16).
- **Result**: `max_abs_diff = 0.0` and `mean_abs_diff = 0.0` on **every** sample.

## 2. Headline finding (changes Phase B scope)

**MLX-core natively supports `mx.dequantize(..., mode='mxfp4', bits=4,
group_size=32)`** as of mlx 0.30+ (confirmed in mlx 0.31.2 ships with
mlx_lm 0.31.2 on the canonical host).

This collapses Phase B W3 (MXFP4 weight loader) from "implement FP4
E2M1 unpacking + E8M0 scale math from scratch" to "**safetensors →
mlx-array adapter + native dequant call**":

| Originally scoped in Phase B PRD §W3 | Actually required after spike |
|---|---|
| FP4 E2M1 unpacking (lookup table of 16 BF16 values) | Not needed — mlx-core handles it |
| E8M0 per-block scale formula (`2^(byte − 127)`) | Not needed — mlx-core handles it |
| Block-size 32 enforcement in kernel | Just pass `group_size=32` to `mx.dequantize` |
| `gate_up_proj` even/odd de-interleaving | Still required (Rust-side, see §3) |
| `_blocks` (uint8) → uint32 view + flatten | Still required (Rust-side) |
| `mlx-sys` binding work | Confirm `mlx_dequantize` is exposed with `mode` parameter (see §5) |

## 3. The ax-engine loader needs to do exactly this

```python
# 1. Read raw bytes from safetensors
blocks_u8 = safetensors.get_tensor("model.layers.N.mlp.experts.gate_up_proj_blocks")
scales_u8 = safetensors.get_tensor("model.layers.N.mlp.experts.gate_up_proj_scales")

# 2. Sanitize: view u8 blocks as u32, flatten last two dims
w_u32 = blocks_u8.view(uint32).flatten(-2)

# 3. De-interleave: even rows = gate, odd rows = up (gate_up_proj only)
gate_w = w_u32[..., ::2, :]
up_w   = w_u32[..., 1::2, :]
gate_s = scales_u8[..., ::2, :]
up_s   = scales_u8[..., 1::2, :]

# 4. Dequant (NATIVE — no FP4 math in our code)
gate_bf16 = mx.dequantize(gate_w[expert], scales=gate_s[expert],
                          group_size=32, bits=4, mode='mxfp4')

# down_proj is the same but without de-interleaving
down_w = blocks_u8.view(uint32).flatten(-2)
down_bf16 = mx.dequantize(down_w[expert], scales=scales_u8[expert],
                          group_size=32, bits=4, mode='mxfp4')
```

The Rust port is mechanical: `view + flatten` is a stride/shape change
(safe), de-interleaving is a slice operation, and dequantize is one
`mlx-sys` FFI call.

## 4. Why max_abs_diff is exactly 0.0

Both paths ultimately call the same MLX C API (`mlx_dequantize`) with
identical inputs:

- mlx-lm's loader: `safetensors → sanitize() → linear.weight/scales → mx.dequantize(...)`
- Our path: `safetensors → sanitize transform → mx.dequantize(...)`

So `0.0` confirms that the **transform pipeline before** the dequant
call is byte-equivalent. This is exactly the validation the spike
was designed to give. It does NOT independently verify the dequant
math; that is mlx-core's responsibility and is exercised any time
mlx_lm runs.

For a stronger "we built our own dequant" test, see §6 — not required
for spike pass, deferred unless we ever stop trusting mlx-core's
MXFP4 path.

## 5. mlx-sys API requirement (Phase B W3 prerequisite)

`crates/mlx-sys/` must expose `mlx_dequantize` (or whatever it's named
in the MLX C header) with at least these parameters:

- `w: mlx_array` (uint32-viewed packed blocks)
- `scales: mlx_array` (uint8 E8M0)
- `biases: Option<mlx_array>` (None for MXFP4)
- `group_size: i32` (32 for MXFP4)
- `bits: i32` (4 for MXFP4)
- `mode: &str` (`"mxfp4"`)
- `dtype: Option<mlx_dtype>` (target BF16 or unset)

**Phase B W3 task 0 (new)**: confirm this is bound in
`crates/mlx-sys/build.rs` / `crates/mlx-sys/src/...`. If not, add the
binding. This replaces the original W3 task list's items 1–5 (which
assumed AX implements dequant) — the loader now only needs:

1. Read `<name>_blocks` + `<name>_scales` from safetensors.
2. Apply the sanitize view/flatten/de-interleave transforms in Rust
   (no math, just stride/shape ops).
3. Call the bound `mlx_dequantize`.
4. Honor `modules_to_not_convert` (attention, router, embed, lm_head
   stay BF16 — no MXFP4 path).

Memory budget unchanged: 12 GB for 20B experts, ≈70 GB for 120B
experts when dequantized to BF16 (per the PRD).

## 6. Stronger test (deferred — not required for spike pass)

If we ever need to validate AX's MXFP4 path independently of mlx-core
(e.g., porting to a non-MLX backend), implement a Python reference
dequant from scratch using the OCP-MX spec FP4 E2M1 table + E8M0 scale
formula, and compare to mlx-core. That work was originally Phase B W3
items 1–5; it is now optional, only needed if AX adopts MXFP4 for
non-MLX targets.

## 7. Phase B implications

- **W3 effort estimate reduced**: from ~1 week to ~2 days.
- **W3 risk reduced**: the largest pre-spike unknown (FP4 dequant
  numerical correctness) is no longer in AX's surface area.
- **W4 (sinks), W5 (YaRN), W6 (clamped SwiGLU + bias + non-square)
  estimates unchanged**.
- **Overall Phase B**: trends to ~1.5 weeks instead of 2–3, dependent
  on §5 confirmation.

## 8. Files in this artifact

- `mxfp4_dequant_check.json` — full per-sample diff data (48 samples)
- `spike.log` — stdout from the spike script
- `report.md` — this file

## 9. Repro

```bash
python3 scripts/gpt_oss_mxfp4_spike.py \
  --output benchmarks/results/gpt-oss/mxfp4-spike-$(date +%Y-%m-%d)/mxfp4_dequant_check.json
```

Dependencies: `mlx_lm>=0.31`, `safetensors`, `openai/gpt-oss-20b`
weights cached under `~/.cache/huggingface/hub/`. Wall time on canonical
M5 Max 128 GB host: ~10 seconds after model load (~2.5 s).
