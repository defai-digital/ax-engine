//! ADR-028 Phase 1 F1 probe: DeepSeek V4 mHC `hc_pre` fused Metal kernel.
//!
//! V4's hyper-connection pre-mixing (sigmoid pre/post gates + softmax +
//! 20-iteration Sinkhorn comb) costs ~129 MLX dispatches per branch per
//! token (~119 in the Sinkhorn loop alone) on static 4x4 f32 shapes — at
//! decode that is pure dispatch overhead, ~65% of the whole-model dispatch
//! count (TECH-SPEC-DSV4-FUSED-KERNELS §2). This probe A/Bs the production
//! op chain against a single packed Metal dispatch, with numerical parity
//! as the hard gate (the fused-router promotion died on parity; ADR-003).
//!
//! Run:
//!   cargo run -p ax-engine-microbench --release --bin dsv4-hc-pre-probe
//!
//! Output: per-shape parity + wall-clock verdict for the ADR-028 Phase 1
//! record.

use std::time::Instant;

use mlx_sys::ops::cached_scalar;
use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, add, divide, eval,
    multiply, op_count_snapshot, op_count_take, reshape, sigmoid, slice, slice_last_dim, softmax,
    sum_axis,
};

/// V4 mHC geometry (HF manifest): hc_mult 4, 20 Sinkhorn iterations.
const HC: i32 = 4;
const MIXES_WIDTH: i32 = 2 * HC + HC * HC;
const SINKHORN_ITERS: i32 = 20;
const HC_EPS: f32 = 1e-5;
const DECODE_ITERS: usize = 2000;
const PREFILL_ITERS: usize = 300;

// Mirrors HC_PRE_KERNEL_SOURCE in ax-engine-mlx
// `model/shared/hyper_connection.rs`; keep the two sources in sync.
const KERNEL_SOURCE: &str = r#"
    uint tok = thread_position_in_grid.x;
    if (tok >= (uint)SeqTokens) {
        return;
    }

    const float eps = eps_arr[0];
    const device float* m = mixes + tok * (2 * HC + HC * HC);

    // Sigmoid gates use MLX's exact form: y = 1/(1+exp(|x|)), (x<0) ? y : 1-y.
    // Volatile products block fma contraction so multiply and add round
    // separately, matching the discrete MLX op kernels.
    for (int h = 0; h < HC; ++h) {
        volatile float prod = m[h] * scale[0];
        float logit = prod + base[h];
        float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(logit)));
        float sig = (logit < 0.0f) ? y : 1.0f - y;
        pre_out[tok * HC + h] = sig + eps;
    }
    for (int h = 0; h < HC; ++h) {
        volatile float prod = m[HC + h] * scale[1];
        float logit = prod + base[HC + h];
        float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(logit)));
        float sig = (logit < 0.0f) ? y : 1.0f - y;
        post_out[tok * HC + h] = sig * 2.0f;
    }

    // Comb logits + softmax over dst. MLX softmax: fast::exp, sequential
    // N_READS=4 sum, then multiply by the reciprocal (never a division).
    float comb[HC][HC]; // [src][dst]
    for (int s = 0; s < HC; ++s) {
        float row[HC];
        for (int d = 0; d < HC; ++d) {
            volatile float prod = m[2 * HC + s * HC + d] * scale[2];
            row[d] = prod + base[2 * HC + s * HC + d];
        }
        float mx = row[0];
        for (int d = 1; d < HC; ++d) {
            mx = (mx < row[d]) ? row[d] : mx;
        }
        volatile float sum = 0.0f;
        float exps[HC];
        for (int d = 0; d < HC; ++d) {
            exps[d] = metal::fast::exp(row[d] - mx);
            sum += exps[d];
        }
        float inv = 1.0f / sum;
        for (int d = 0; d < HC; ++d) {
            volatile float sm = exps[d] * inv;
            comb[s][d] = sm + eps;
        }
    }

    // Sinkhorn: one src-normalisation, then (iters-1) x (dst, src). Every
    // normalisation divides by (axis sum + eps); sums are 4-wide sequential
    // in index order like MLX's small-tensor reduce kernels.
    for (int it = 0; it < ITERS; ++it) {
        if (it > 0) {
            for (int s = 0; s < HC; ++s) {
                volatile float sum = 0.0f;
                for (int d = 0; d < HC; ++d) {
                    sum += comb[s][d];
                }
                float denom = sum + eps;
                for (int d = 0; d < HC; ++d) {
                    comb[s][d] = comb[s][d] / denom;
                }
            }
        }
        for (int d = 0; d < HC; ++d) {
            volatile float sum = 0.0f;
            for (int s = 0; s < HC; ++s) {
                sum += comb[s][d];
            }
            float denom = sum + eps;
            for (int s = 0; s < HC; ++s) {
                comb[s][d] = comb[s][d] / denom;
            }
        }
    }

    for (int s = 0; s < HC; ++s) {
        for (int d = 0; d < HC; ++d) {
            comb_out[tok * HC * HC + s * HC + d] = comb[s][d];
        }
    }
"#;

struct Shape {
    label: &'static str,
    seq: i32,
    iters: usize,
}

const SHAPES: &[Shape] = &[
    Shape {
        label: "decode s=1 hc=4 iters=20",
        seq: 1,
        iters: DECODE_ITERS,
    },
    Shape {
        label: "prefill s=128 hc=4 iters=20",
        seq: 128,
        iters: PREFILL_ITERS,
    },
];

fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data),
        shape,
        MlxDtype::Float32,
    )
}

/// Deterministic xorshift fill (same generator as the F0 probe) over
/// (min, max); mixes span logit ranges that engage sigmoid curvature.
fn fill_uniform(count: usize, min: f32, max: f32, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let raw = (state >> 32) as u32;
            (raw as f32 / u32::MAX as f32) * (max - min) + min
        })
        .collect()
}

fn build_input(shape: &Shape) -> (MlxArray, MlxArray, MlxArray) {
    let mixes = array_f32(
        &fill_uniform(
            (shape.seq * MIXES_WIDTH) as usize,
            -3.0,
            3.0,
            0x9e3779b97f4a7c15,
        ),
        &[1, shape.seq, MIXES_WIDTH],
    );
    let base = array_f32(
        &fill_uniform(MIXES_WIDTH as usize, -1.0, 1.0, 0x123456789abcdef0),
        &[MIXES_WIDTH],
    );
    let scale = array_f32(&fill_uniform(3, 0.5, 1.5, 0xfedcba0987654321), &[3]);
    eval(&[&mixes, &base, &scale]);
    (mixes, base, scale)
}

struct OpChainOut {
    pre: MlxArray,
    post: MlxArray,
    comb: MlxArray,
}

/// Production op chain, replicated call-for-call from
/// `hyper_connection.rs` (`hc_mixes` post-matmul + `hc_sinkhorn`).
fn op_chain(mixes: &MlxArray, base: &MlxArray, scale: &MlxArray) -> OpChainOut {
    let seq = mixes.shape()[1];
    let eps = cached_scalar(HC_EPS, MlxDtype::Float32);
    let scale_pre = slice(scale, &[0], &[1], &[1], None);
    let scale_post = slice(scale, &[1], &[2], &[1], None);
    let scale_comb = slice(scale, &[2], &[3], &[1], None);
    let base_pre = slice(base, &[0], &[HC], &[1], None);
    let base_post = slice(base, &[HC], &[2 * HC], &[1], None);
    let base_comb = slice(base, &[2 * HC], &[(2 + HC) * HC], &[1], None);

    let pre_logits = add(
        &multiply(&slice_last_dim(mixes, 0, HC, None), &scale_pre, None),
        &base_pre,
        None,
    );
    let pre = add(&sigmoid(&pre_logits, None), &eps, None);

    let post_logits = add(
        &multiply(&slice_last_dim(mixes, HC, 2 * HC, None), &scale_post, None),
        &base_post,
        None,
    );
    let post = multiply(
        &sigmoid(&post_logits, None),
        &cached_scalar(2.0, MlxDtype::Float32),
        None,
    );

    let comb_logits = reshape(
        &slice_last_dim(mixes, 2 * HC, (2 + HC) * HC, None),
        &[1, seq, HC, HC],
        None,
    );
    let comb_logits = add(
        &multiply(&comb_logits, &scale_comb, None),
        &reshape(&base_comb, &[HC, HC], None),
        None,
    );

    let mut comb = add(&softmax(&comb_logits, -1, None), &eps, None);
    let norm_src = |m: &MlxArray| divide(m, &add(&sum_axis(m, 2, true, None), &eps, None), None);
    let norm_dst = |m: &MlxArray| divide(m, &add(&sum_axis(m, 3, true, None), &eps, None), None);
    comb = norm_src(&comb);
    for _ in 1..SINKHORN_ITERS {
        comb = norm_dst(&comb);
        comb = norm_src(&comb);
    }
    OpChainOut { pre, post, comb }
}

#[allow(
    clippy::expect_used,
    reason = "the three-output kernel specification fixes the output vector length"
)]
fn fused_metal(
    kernel: &MlxMetalKernel,
    mixes: &MlxArray,
    base: &MlxArray,
    scale: &MlxArray,
    seq: i32,
) -> OpChainOut {
    let eps_arr = cached_scalar(HC_EPS, MlxDtype::Float32);
    let mut outputs = kernel.apply_with_template(
        &[mixes, base, scale, &eps_arr],
        &[
            KernelOutputSpec {
                shape: vec![1, seq, HC],
                dtype: MlxDtype::Float32,
            },
            KernelOutputSpec {
                shape: vec![1, seq, HC],
                dtype: MlxDtype::Float32,
            },
            KernelOutputSpec {
                shape: vec![1, seq, HC, HC],
                dtype: MlxDtype::Float32,
            },
        ],
        &[
            KernelTemplateArg::Int {
                name: "SeqTokens",
                value: seq,
            },
            KernelTemplateArg::Int {
                name: "HC",
                value: HC,
            },
            KernelTemplateArg::Int {
                name: "ITERS",
                value: SINKHORN_ITERS,
            },
        ],
        (seq, 1, 1),
        (seq.min(256), 1, 1),
        None,
    );
    OpChainOut {
        comb: outputs.pop().expect("comb output"),
        post: outputs.pop().expect("post output"),
        pre: outputs.pop().expect("pre output"),
    }
}

fn eval3(out: &OpChainOut) {
    eval(&[&out.pre, &out.post, &out.comb]);
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// f32 ULP at the reference max binade: ULP(x) = x * 2^-23. The chain is
/// f32 end to end, so bit-exact (diff == 0) is the bar; anything larger is
/// reported in ULP units against this scale.
fn ulp_at_max(reference: &[f32]) -> f32 {
    let max_abs = reference.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    (max_abs * 2f32.powi(-23)).max(1e-30)
}

fn time_loop<F: FnMut()>(label: &str, iters: usize, mut f: F) -> f64 {
    f();
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let per_iter_us = t0.elapsed().as_secs_f64() * 1.0e6 / iters as f64;
    println!("    {label}: {per_iter_us:.2} us/iter");
    per_iter_us
}

fn main() {
    println!(
        "V4 mHC hc_pre fused-kernel probe (hc={HC}, iters={SINKHORN_ITERS}, hc_eps={HC_EPS}, f32; gate: bit-exact, else <= ~1 f32 ULP)"
    );

    let kernel = MlxMetalKernel::new(
        "ax_dsv4_hc_pre_probe",
        &["mixes", "base", "scale", "eps_arr"],
        &["pre_out", "post_out", "comb_out"],
        KERNEL_SOURCE,
        "",
        true,
    );

    let mut parity_fail = false;
    let mut worst_ulp = 0.0_f32;
    for shape in SHAPES {
        println!("\n=== {} (iters={}) ===", shape.label, shape.iters);
        let (mixes, base, scale) = build_input(shape);

        // Parity: fused vs the production op chain, per output.
        let ops_out = op_chain(&mixes, &base, &scale);
        let fused_out = fused_metal(&kernel, &mixes, &base, &scale, shape.seq);
        eval3(&ops_out);
        eval3(&fused_out);
        let mut shape_ok = true;
        for (label, reference, fused) in [
            ("pre", &ops_out.pre, &fused_out.pre),
            ("post", &ops_out.post, &fused_out.post),
            ("comb", &ops_out.comb, &fused_out.comb),
        ] {
            let diff = max_abs_diff(reference.data_f32(), fused.data_f32());
            let ulp = ulp_at_max(reference.data_f32());
            let diff_ulp = diff / ulp;
            worst_ulp = worst_ulp.max(diff_ulp);
            if diff == 0.0 {
                println!("  parity: {label} bit-exact");
            } else {
                println!("  parity: {label} max abs diff {diff:.6e} = {diff_ulp:.2} f32 ULP");
            }
            // Spec gate: bit-exact preferred, <= ~1 ULP required for wiring.
            if diff_ulp > 1.5 {
                shape_ok = false;
            }
        }
        if !shape_ok {
            parity_fail = true;
            println!("  parity: FAIL");
            continue;
        }

        // Dispatch evidence: measured op-wrapper calls for one op-chain
        // build (includes free views), plus the structural non-view count.
        let prev = op_count_snapshot();
        let counted = op_chain(&mixes, &base, &scale);
        let wrappers = op_count_take(prev);
        eval3(&counted);
        println!(
            "  dispatches: op chain {wrappers} measured wrapper calls \
             (129 non-view: pre 4 + post 4 + comb_logits 2 + softmax+eps 2 + norm_src 3 + 19x(3+3)); \
             fused = 1 Metal dispatch"
        );

        let ops_us = time_loop("op chain", shape.iters, || {
            let y = op_chain(&mixes, &base, &scale);
            eval3(&y);
        });
        let fused_us = time_loop("fused metal", shape.iters, || {
            let y = fused_metal(&kernel, &mixes, &base, &scale, shape.seq);
            eval3(&y);
        });

        let delta_pct = (ops_us - fused_us) / ops_us * 100.0;
        println!(
            "  verdict: fused is {delta_pct:+.1}% vs op chain ({ops_us:.2} -> {fused_us:.2} us/iter)"
        );
    }

    if parity_fail {
        println!(
            "\nRESULT: FAIL — parity worse than ~1 f32 ULP on at least one shape; do not wire"
        );
        std::process::exit(1);
    }
    if worst_ulp == 0.0 {
        println!(
            "\nRESULT: parity bit-exact on all shapes; fused = 1 dispatch vs ~129 op-chain dispatches; see per-shape verdicts for the wiring decision"
        );
    } else {
        println!(
            "\nRESULT: parity within {worst_ulp:.2} f32 ULP on all shapes; see per-shape verdicts for the wiring decision"
        );
    }
}
