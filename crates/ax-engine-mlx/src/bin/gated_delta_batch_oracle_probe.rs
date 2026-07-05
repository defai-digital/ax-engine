//! ADR-037 P1 — batched gated-delta row-correctness oracle.
//!
//! ## The question this answers
//!
//! Batched decode for the Qwen 3.5/3.6 hybrids needs the linear-attention
//! step to run as `[B, ...]` with per-row conv windows and per-row recurrent
//! states. The production ops (`linear_attention_conv1d`, the decode
//! post-input Metal kernel, `gated_delta_kernel`, the gated RMSNorm) are all
//! *shape*-parameterized over a leading batch dimension already — but the
//! Phase-2a dense work proved that "accepts a batch dim" is not "batch
//! correct": `mlx_fast_rope` silently mis-rotated rows > 0 for `[B, H, 1, D]`
//! decode inputs, and hand-written Metal grids can be row-0-correct while
//! scrambling every other row. This probe checks, on real model weights,
//! that every op on the batched gated-delta path produces row `r` outputs
//! IDENTICAL to running row `r` alone — with **distinct** per-row inputs,
//! the configuration that hides nothing.
//!
//! ## What it checks (each vs per-row batch-1 calls of the same fn)
//!
//!  1. `linear_attention_conv1d` (portable conv + SiLU + window state)
//!  2. `linear_attention_decode_post_input_metal` (fused conv/split/norm)
//!  3. `gated_delta_kernel` seq==1 (decode Metal kernel, per-row f32 state)
//!  4. `gated_delta_kernel` seq==4 (verify-width loop kernel)
//!  5. `rms_norm_gated_with_full_gate_policy` (portable and Metal gate)
//!  6. `mlx_sys::rope` on `[B, H, 1, D]` — the known Phase-2a hazard, checked
//!     here because the hybrid's 1-in-4 full-attention layers need it; a
//!     FAIL is *recorded* (with the identical-rows control), not fatal to
//!     the gated-delta verdict.
//!
//! ## Usage
//!
//!   cargo run --release --bin gated_delta_batch_oracle_probe -- <model_dir>
//!
//! `<model_dir>` must be a gated-delta linear-attention model (e.g.
//! Qwen3.5-9B-MLX-4bit). Exit 0 iff oracles 1-5 are row-exact.

use std::env;
use std::path::Path;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::linear_attention_ops::{
    gated_delta_kernel, linear_attention_conv1d, linear_attention_decode_post_input_metal,
    rms_norm_gated_with_full_gate_policy,
};
use ax_engine_mlx::model::{LinearAttentionConfig, ModelConfig};
use ax_engine_mlx::weights::{LinearAttentionWeights, load_weights};
use mlx_sys::{MlxArray, MlxDtype, astype, concatenate, eval, rope, slice};

/// Deterministic LCG so every run and every row is reproducible and distinct.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // top 24 bits → [-1, 1)
        ((self.0 >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
}

fn randn(shape: &[i32], scale: f32, dtype: MlxDtype, lcg: &mut Lcg) -> MlxArray {
    let count: usize = shape.iter().map(|&d| d as usize).product();
    let data: Vec<f32> = (0..count).map(|_| lcg.next_f32() * scale).collect();
    let arr = MlxArray::from_raw_data(data.as_ptr().cast(), count * 4, shape, MlxDtype::Float32);
    if dtype == MlxDtype::Float32 {
        arr
    } else {
        astype(&arr, dtype, None)
    }
}

/// Max |a - b| in f32, computed host-side (arrays here are at most a few
/// hundred KiB; the probe is correctness-only, not timing).
fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    let a32 = astype(a, MlxDtype::Float32, None);
    let b32 = astype(b, MlxDtype::Float32, None);
    eval(&[&a32, &b32]);
    let (av, bv) = (a32.data_f32(), b32.data_f32());
    assert_eq!(av.len(), bv.len(), "oracle output shapes must match");
    av.iter()
        .zip(bv.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Slice row `r` (leading axis) of `a`, keeping the axis.
fn row(a: &MlxArray, r: i32) -> MlxArray {
    let shape = a.shape();
    let mut start = vec![0i32; shape.len()];
    let mut stop = shape.clone();
    start[0] = r;
    stop[0] = r + 1;
    let strides = vec![1i32; shape.len()];
    slice(a, &start, &stop, &strides, None)
}

/// Stack per-row arrays along the leading axis.
fn stack_rows(rows: &[MlxArray]) -> MlxArray {
    let refs: Vec<&MlxArray> = rows.iter().collect();
    concatenate(&refs, 0, None)
}

struct Verdict {
    name: &'static str,
    max_diff: f32,
    pass: bool,
    note: String,
}

fn check(name: &'static str, batched: &[MlxArray], singles: &[Vec<MlxArray>], tol: f32) -> Verdict {
    let b = singles.len();
    let mut worst = 0.0f32;
    for (out_idx, batched_out) in batched.iter().enumerate() {
        for (r, single_outs) in singles.iter().enumerate().take(b) {
            let d = max_abs_diff(&row(batched_out, r as i32), &single_outs[out_idx]);
            if d > worst {
                worst = d;
            }
        }
    }
    Verdict {
        name,
        max_diff: worst,
        pass: worst <= tol,
        note: String::new(),
    }
}

#[allow(clippy::too_many_lines)]
fn main() {
    let model_dir = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: gated_delta_batch_oracle_probe <linear-attention model_dir>");
        std::process::exit(2);
    });
    let b: usize = env::var("AX_ORACLE_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);

    let artifacts =
        NativeModelArtifacts::from_dir(Path::new(&model_dir)).expect("load model artifacts");
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let lin: &LinearAttentionConfig = cfg
        .linear_attention
        .as_ref()
        .expect("model must have gated-delta linear attention");
    let weights = load_weights(&artifacts).expect("load weights");
    let linear_layers: Vec<&LinearAttentionWeights> = weights
        .layers
        .iter()
        .filter_map(|l| l.linear_attn.as_ref())
        .collect();
    assert!(!linear_layers.is_empty(), "no linear-attention layers");
    // First, middle, last linear layers — kernel behavior is weight-dependent
    // only through shapes + a_log/dt_bias, but cover the spread anyway.
    let picks = [0usize, linear_layers.len() / 2, linear_layers.len() - 1];

    let (hk, hv) = (lin.num_key_heads as i32, lin.num_value_heads as i32);
    let (dk, dv) = (lin.key_head_dim as i32, lin.value_head_dim as i32);
    let conv_dim = (lin.num_key_heads * lin.key_head_dim * 2
        + lin.num_value_heads * lin.value_head_dim) as i32;
    let tail = lin.conv_kernel_dim as i32 - 1;

    println!("# gated-delta batched row-correctness oracle (ADR-037 P1)");
    println!("model_dir            {model_dir}");
    println!(
        "linear layers        {}   Hk {hk} Dk {dk}   Hv {hv} Dv {dv}   conv_dim {conv_dim}   k {}   B {b}",
        linear_layers.len(),
        lin.conv_kernel_dim
    );
    println!();

    let mut verdicts: Vec<Verdict> = Vec::new();
    let mut lcg = Lcg(0x5eed_c0de_2026_0705);

    for (pick_no, &li) in picks.iter().enumerate() {
        let lw = linear_layers[li];

        // ── 1. portable conv1d + SiLU + window state ──
        let qkv_rows: Vec<MlxArray> = (0..b)
            .map(|_| randn(&[1, 1, conv_dim], 0.7, MlxDtype::Bfloat16, &mut lcg))
            .collect();
        let state_rows: Vec<MlxArray> = (0..b)
            .map(|_| randn(&[1, tail, conv_dim], 0.7, MlxDtype::Bfloat16, &mut lcg))
            .collect();
        let qkv_b = stack_rows(&qkv_rows);
        let state_b = stack_rows(&state_rows);
        let (y_b, ns_b) = linear_attention_conv1d(lin, &qkv_b, &lw.conv1d_dense, Some(&state_b));
        let singles: Vec<Vec<MlxArray>> = (0..b)
            .map(|r| {
                let (y, ns) = linear_attention_conv1d(
                    lin,
                    &qkv_rows[r],
                    &lw.conv1d_dense,
                    Some(&state_rows[r]),
                );
                vec![y, ns]
            })
            .collect();
        if pick_no == 0 {
            verdicts.push(check(
                "conv1d+silu+window (portable)",
                &[y_b.clone(), ns_b.clone()],
                &singles,
                0.0,
            ));
        }

        // ── 2. fused decode post-input Metal kernel ──
        let fused_b = linear_attention_decode_post_input_metal(
            lin,
            &qkv_b,
            &lw.conv1d_dense,
            Some(&state_b),
            lin.q_scale,
            lin.k_scale,
            cfg.rms_norm_eps,
        );
        match fused_b {
            Some((qb, kb, vb, nsb)) if pick_no == 0 => {
                let singles: Vec<Vec<MlxArray>> = (0..b)
                    .map(|r| {
                        let (q, k, v, ns) = linear_attention_decode_post_input_metal(
                            lin,
                            &qkv_rows[r],
                            &lw.conv1d_dense,
                            Some(&state_rows[r]),
                            lin.q_scale,
                            lin.k_scale,
                            cfg.rms_norm_eps,
                        )
                        .expect("fused post-input accepted batch=1");
                        vec![q, k, v, ns]
                    })
                    .collect();
                verdicts.push(check(
                    "decode post-input (Metal fused)",
                    &[qb, kb, vb, nsb],
                    &singles,
                    0.0,
                ));
            }
            Some(_) => {}
            None if pick_no == 0 => {
                verdicts.push(Verdict {
                    name: "decode post-input (Metal fused)",
                    max_diff: 0.0,
                    pass: true,
                    note: "SKIP (kernel ineligible for this config)".into(),
                });
            }
            None => {}
        }

        // ── 3+4. gated-delta recurrence at seq 1 (decode Metal) and seq 4 ──
        for &seq in &[1i32, 4] {
            let q_rows: Vec<MlxArray> = (0..b)
                .map(|_| randn(&[1, seq, hk, dk], 0.6, MlxDtype::Bfloat16, &mut lcg))
                .collect();
            let k_rows: Vec<MlxArray> = (0..b)
                .map(|_| randn(&[1, seq, hk, dk], 0.6, MlxDtype::Bfloat16, &mut lcg))
                .collect();
            let v_rows: Vec<MlxArray> = (0..b)
                .map(|_| randn(&[1, seq, hv, dv], 0.6, MlxDtype::Bfloat16, &mut lcg))
                .collect();
            let a_rows: Vec<MlxArray> = (0..b)
                .map(|_| randn(&[1, seq, hv], 0.8, MlxDtype::Bfloat16, &mut lcg))
                .collect();
            let bg_rows: Vec<MlxArray> = (0..b)
                .map(|_| randn(&[1, seq, hv], 0.8, MlxDtype::Bfloat16, &mut lcg))
                .collect();
            let st_rows: Vec<MlxArray> = (0..b)
                .map(|_| randn(&[1, hv, dv, dk], 0.4, MlxDtype::Float32, &mut lcg))
                .collect();
            let (qb, kb, vb) = (
                stack_rows(&q_rows),
                stack_rows(&k_rows),
                stack_rows(&v_rows),
            );
            let (ab, bb, sb) = (
                stack_rows(&a_rows),
                stack_rows(&bg_rows),
                stack_rows(&st_rows),
            );
            let (y_b, s_b) =
                gated_delta_kernel(&qb, &kb, &vb, &lw.a_log, &ab, &lw.dt_bias, &bb, &sb);
            let singles: Vec<Vec<MlxArray>> = (0..b)
                .map(|r| {
                    let (y, s) = gated_delta_kernel(
                        &q_rows[r],
                        &k_rows[r],
                        &v_rows[r],
                        &lw.a_log,
                        &a_rows[r],
                        &lw.dt_bias,
                        &bg_rows[r],
                        &st_rows[r],
                    );
                    vec![y, s]
                })
                .collect();
            let name: &'static str = match (pick_no, seq) {
                (_, 1) => "gated-delta recurrence seq=1 (decode Metal)",
                _ => "gated-delta recurrence seq=4 (loop kernel)",
            };
            if pick_no == 0 {
                verdicts.push(check(name, &[y_b, s_b], &singles, 0.0));
            } else {
                // Other layers: fold into the same named check by verifying
                // silently and reporting only failures.
                let v = check(name, &[y_b, s_b], &singles, 0.0);
                if !v.pass {
                    verdicts.push(Verdict {
                        note: format!("layer index {li}"),
                        ..v
                    });
                }
            }
        }

        // ── 5. gated RMSNorm (portable + Metal full-gate) ──
        if pick_no == 0 {
            let h_rows: Vec<MlxArray> = (0..b)
                .map(|_| randn(&[1, 1, hv, dv], 0.6, MlxDtype::Bfloat16, &mut lcg))
                .collect();
            let z_rows: Vec<MlxArray> = (0..b)
                .map(|_| randn(&[1, 1, hv, dv], 0.6, MlxDtype::Bfloat16, &mut lcg))
                .collect();
            for (metal, name) in [
                (false, "gated rms_norm (portable)"),
                (true, "gated rms_norm (Metal full-gate)"),
            ] {
                let out_b = rms_norm_gated_with_full_gate_policy(
                    &stack_rows(&h_rows),
                    &stack_rows(&z_rows),
                    &lw.norm,
                    cfg.rms_norm_eps,
                    metal,
                );
                let singles: Vec<Vec<MlxArray>> = (0..b)
                    .map(|r| {
                        vec![rms_norm_gated_with_full_gate_policy(
                            &h_rows[r],
                            &z_rows[r],
                            &lw.norm,
                            cfg.rms_norm_eps,
                            metal,
                        )]
                    })
                    .collect();
                verdicts.push(check(name, &[out_b], &singles, 0.0));
            }
        }
    }

    // ── 6. mlx_fast_rope hazard on [B, H, 1, D] (full-attention layers) ──
    {
        let (h, d, offset) = (8i32, 128i32, 777i32);
        let mut lcg = Lcg(0x0f5e_ed42_2026_0705);
        let rows: Vec<MlxArray> = (0..b)
            .map(|_| randn(&[1, h, 1, d], 0.6, MlxDtype::Bfloat16, &mut lcg))
            .collect();
        let batched = rope(
            &stack_rows(&rows),
            d,
            false,
            Some(1.0e6),
            1.0,
            offset,
            None,
            None,
        );
        let mut worst = 0.0f32;
        for (r, row_in) in rows.iter().enumerate() {
            let single = rope(row_in, d, false, Some(1.0e6), 1.0, offset, None, None);
            let dmax = max_abs_diff(&row(&batched, r as i32), &single);
            if dmax > worst {
                worst = dmax;
            }
        }
        // Identical-rows control: the classic signature is identical-PASS /
        // distinct-FAIL with row 0 always correct.
        let same = stack_rows(&vec![rows[0].clone(); b]);
        let same_b = rope(&same, d, false, Some(1.0e6), 1.0, offset, None, None);
        let single0 = rope(&rows[0], d, false, Some(1.0e6), 1.0, offset, None, None);
        let mut same_worst = 0.0f32;
        for r in 0..b as i32 {
            let dmax = max_abs_diff(&row(&same_b, r), &single0);
            if dmax > same_worst {
                same_worst = dmax;
            }
        }
        verdicts.push(Verdict {
            name: "mlx rope [B,H,1,D] (full-attn layers; informational)",
            max_diff: worst,
            pass: worst == 0.0,
            note: format!("identical-rows control max_diff {same_worst}"),
        });
    }

    println!("{:<52} {:>12}  verdict", "oracle", "max_abs_diff");
    let mut hard_fail = false;
    for v in &verdicts {
        let is_rope = v.name.starts_with("mlx rope");
        let verdict = if !v.note.is_empty() && v.note.starts_with("SKIP") {
            "SKIP"
        } else if v.pass {
            "PASS"
        } else if is_rope {
            "FAIL (recorded; design must rope per row or pre-batch)"
        } else {
            hard_fail = true;
            "FAIL"
        };
        println!(
            "{:<52} {:>12.3e}  {}{}",
            v.name,
            v.max_diff,
            verdict,
            if v.note.is_empty() || v.note.starts_with("SKIP") {
                String::new()
            } else {
                format!("  [{}]", v.note)
            }
        );
    }
    println!();
    println!(
        "P1 gated-delta verdict: {}",
        if hard_fail { "FAIL" } else { "PASS" }
    );
    std::process::exit(if hard_fail { 1 } else { 0 });
}
