//! ADR-028 Phase 1.3 F2 probe: DeepSeek V4 two-source fused decode SDPA.
//!
//! V4 decode attention on compress layers physically concatenates the raw
//! sliding-window ring with the committed compressed-K rows (~8.5 MB copy per
//! CSA layer per token at 32k ctx) plus a matching bool-mask concat, then
//! runs one `scaled_dot_product_attention_with_mask_and_sinks` over
//! 128 + n_rows keys (K == V latent). This probe A/Bs that chain against a
//! single fused two-source online-softmax kernel (in-kernel mask + sink, no
//! concat), with numerical parity as the hard gate (fused-router precedent;
//! ADR-003).
//!
//! Key dispatch fact this probe is built around: MLX's fused SDPA kernels
//! support head dims {64..256}; V4's head_dim is 512, so the production call
//! runs MLX's *unfused fallback* — an op chain (bf16 pre-scale, matmul,
//! where, sink concat, precise softmax, matmul) whose bf16/f32 staging was
//! verified op-for-op to reproduce the native call bit-exactly (see the
//! probe notes in the Phase 1.3 record). Bit-exact reproduction of that
//! chain inside one streaming kernel is NOT expected: the steel GEMM/splitk
//! tile+MMA reduction order and the size-dependent block/looped softmax
//! thread mapping are unreachable in an online-softmax form. The probe
//! measures the exact divergence bound, attributes it per stage (scores-in
//! and probs-in ablations), and reports the perf upside.
//!
//! Run:
//!   cargo run -p ax-engine-microbench --release --bin dsv4-fused-sdpa-probe
//!
//! Output: per-shape parity + wall-clock verdict for the ADR-028 Phase 1.3
//! record.

use std::time::Instant;

use mlx_sys::ops::cached_scalar;
use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel,
    ScaledDotProductAttentionMask, astype, broadcast_to, concatenate, eval, expand_dims, matmul,
    multiply, op_count_snapshot, op_count_take, reshape,
    scaled_dot_product_attention_with_mask_and_sinks, slice, softmax_precise, transpose, unflatten,
    where_cond,
};

const H: i32 = 64;
const D: i32 = 512;
const WINDOW: i32 = 128;
/// Production scale: `1.0 / (head_dim as f32).sqrt()`.
fn scale() -> f32 {
    1.0 / (D as f32).sqrt()
}
/// `finfo(bfloat16).min` — the fallback's bool-mask fill value (exact bf16
/// bit pattern 0xFF7F widened to f32).
const BF16_MIN: f32 = f32::from_bits(0xFF7F_0000);
/// Ablation buffers feed per-head rows of up to 8192 keys.
const ABLATION_STRIDE: usize = 8192;

/// Fused two-source online-softmax decode SDPA (one 256-thread threadgroup
/// per query head, 8 simdgroups). bf16 stage roundings matched where a
/// streaming kernel can match them: the bf16 q pre-scale and the per-key
/// bf16 score rounding. The bf16 prob rounding of the fallback's two-pass
/// softmax is NOT reproduceable online (the final max/sum is unknown until
/// the row is consumed); exp uses `fast::exp` like MLX's softmax kernels.
const FUSED_SOURCE: &str = r#"
    uint head = threadgroup_position_in_grid.x;
    uint sg = thread_position_in_threadgroup.x / 32;
    uint lane = thread_position_in_threadgroup.x % 32;

    const int N = iparams[0];
    const int window = iparams[1];
    const float scale = fparams[0]; // f32(bf16(scale)) — pre-rounded

    thread float q[16];
    thread float o[16];
    threadgroup float outputs[8 * 32];
    threadgroup float max_scores[8];
    threadgroup float sum_exp_scores[8];

    const device bfloat16_t* qp = queries + head * 512 + lane * 16;
    for (int j = 0; j < 16; j++) {
        // Fallback stage: bf16(bf16(scale) * q) — exact f32 product, bf16 round.
        q[j] = static_cast<float>(static_cast<bfloat16_t>(scale * qp[j]));
        o[j] = 0;
    }

    float max_score = -0x1.fffffep+127f;
    float sum_exp_score = 0.0f;
    if (sg == 0) {
        max_score = static_cast<float>(sinks[head]);
        sum_exp_score = 1.0f;
    }

    for (int i = sg; i < N; i += 8) {
        bool use_key = (i < window) ? true : comp_mask[i - window];
        if (use_key) {
            const device bfloat16_t* kp = (i < window)
                ? (k_raw + i * 512 + lane * 16)
                : (comp_k + (i - window) * 512 + lane * 16);
            float score = 0;
            for (int j = 0; j < 16; j++) {
                score += q[j] * kp[j];
            }
            score = simd_sum(score);
            // Fallback stage: the score row materializes as bf16.
            score = static_cast<float>(static_cast<bfloat16_t>(score));

            float new_max = max(max_score, score);
            float factor = fast::exp(max_score - new_max);
            float exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            for (int j = 0; j < 16; j++) {
                o[j] = o[j] * factor + exp_score * kp[j];
            }
        }
    }

    if (lane == 0) {
        max_scores[sg] = max_score;
        sum_exp_scores[sg] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmax = max_scores[0];
    for (int g = 1; g < 8; g++) { gmax = max(gmax, max_scores[g]); }
    float denom = 0;
    for (int g = 0; g < 8; g++) {
        denom += sum_exp_scores[g] * fast::exp(max_scores[g] - gmax);
    }
    for (int i = 0; i < 16; i++) {
        outputs[lane * 8 + sg] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float acc = 0;
        for (int g = 0; g < 8; g++) {
            acc += outputs[lane * 8 + g] * fast::exp(max_scores[g] - gmax);
        }
        o[i] = acc / denom;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sg == 0) {
        device bfloat16_t* op = out + head * 512 + lane * 16;
        for (int i = 0; i < 16; i++) { op[i] = static_cast<bfloat16_t>(o[i]); }
    }
"#;

/// Ablation A: identical to FUSED_SOURCE but reads the fallback's exact
/// bf16 scores from `scores_in` — isolates softmax + V-dot divergence.
const SCORES_IN_SOURCE: &str = r#"
    uint head = threadgroup_position_in_grid.x;
    uint sg = thread_position_in_threadgroup.x / 32;
    uint lane = thread_position_in_threadgroup.x % 32;

    const int N = iparams[0];
    const int window = iparams[1];

    thread float o[16];
    threadgroup float outputs[8 * 32];
    threadgroup float max_scores[8];
    threadgroup float sum_exp_scores[8];
    for (int j = 0; j < 16; j++) { o[j] = 0; }

    float max_score = -0x1.fffffep+127f;
    float sum_exp_score = 0.0f;
    if (sg == 0) {
        max_score = static_cast<float>(sinks[head]);
        sum_exp_score = 1.0f;
    }

    for (int i = sg; i < N; i += 8) {
        bool use_key = (i < window) ? true : comp_mask[i - window];
        if (use_key) {
            const device bfloat16_t* kp = (i < window)
                ? (k_raw + i * 512 + lane * 16)
                : (comp_k + (i - window) * 512 + lane * 16);
            float score = static_cast<float>(scores_in[head * 8192 + i]);

            float new_max = max(max_score, score);
            float factor = fast::exp(max_score - new_max);
            float exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            for (int j = 0; j < 16; j++) {
                o[j] = o[j] * factor + exp_score * kp[j];
            }
        }
    }

    if (lane == 0) {
        max_scores[sg] = max_score;
        sum_exp_scores[sg] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmax = max_scores[0];
    for (int g = 1; g < 8; g++) { gmax = max(gmax, max_scores[g]); }
    float denom = 0;
    for (int g = 0; g < 8; g++) {
        denom += sum_exp_scores[g] * fast::exp(max_scores[g] - gmax);
    }
    for (int i = 0; i < 16; i++) {
        outputs[lane * 8 + sg] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float acc = 0;
        for (int g = 0; g < 8; g++) {
            acc += outputs[lane * 8 + g] * fast::exp(max_scores[g] - gmax);
        }
        o[i] = acc / denom;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sg == 0) {
        device bfloat16_t* op = out + head * 512 + lane * 16;
        for (int i = 0; i < 16; i++) { op[i] = static_cast<bfloat16_t>(o[i]); }
    }
"#;

/// Ablation B: V-dot only over the fallback's exact bf16 probs — isolates
/// the output-matmul reduction-order divergence.
const PROBS_IN_SOURCE: &str = r#"
    uint head = threadgroup_position_in_grid.x;
    uint sg = thread_position_in_threadgroup.x / 32;
    uint lane = thread_position_in_threadgroup.x % 32;
    const int N = iparams[0];
    const int window = iparams[1];
    thread float o[16];
    for (int j = 0; j < 16; j++) { o[j] = 0; }
    for (int i = sg; i < N; i += 8) {
        bool use_key = (i < window) ? true : comp_mask[i - window];
        if (use_key) {
            const device bfloat16_t* kp = (i < window)
                ? (k_raw + i * 512 + lane * 16)
                : (comp_k + (i - window) * 512 + lane * 16);
            float p = static_cast<float>(probs_in[head * 8192 + i]);
            for (int j = 0; j < 16; j++) { o[j] += p * kp[j]; }
        }
    }
    threadgroup float outputs[8 * 32];
    for (int i = 0; i < 16; i++) {
        outputs[lane * 8 + sg] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float acc = 0;
        for (int g = 0; g < 8; g++) { acc += outputs[lane * 8 + g]; }
        o[i] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0) {
        device bfloat16_t* op = out + head * 512 + lane * 16;
        for (int i = 0; i < 16; i++) { op[i] = static_cast<bfloat16_t>(o[i]); }
    }
"#;

struct Shape {
    label: &'static str,
    n_rows: i32,
    mask: MaskKind,
    iters: usize,
}

#[derive(Clone, Copy)]
enum MaskKind {
    /// All committed rows visible (short CSA ctx below top-k; HCA).
    AllVisible,
    /// Deterministic ~512-row selection (CSA top-k at long ctx).
    Top512,
}

const SHAPES: &[Shape] = &[
    Shape {
        label: "csa-2k all-visible (N=628)",
        n_rows: 500,
        mask: MaskKind::AllVisible,
        iters: 300,
    },
    Shape {
        label: "hca-32k visibility (N=378)",
        n_rows: 250,
        mask: MaskKind::AllVisible,
        iters: 300,
    },
    Shape {
        label: "csa-8k top-512 (N=2128)",
        n_rows: 2000,
        mask: MaskKind::Top512,
        iters: 200,
    },
    Shape {
        label: "csa-32k top-512 (N=8128)",
        n_rows: 8000,
        mask: MaskKind::Top512,
        iters: 100,
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

fn array_i32(data: &[i32], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data),
        shape,
        MlxDtype::Int32,
    )
}

fn array_bool(data: &[bool], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data),
        shape,
        MlxDtype::Bool,
    )
}

fn array_bf16_from_f32(data: &[f32], shape: &[i32]) -> MlxArray {
    let f = array_f32(data, shape);
    let b = astype(&f, MlxDtype::Bfloat16, None);
    eval(&[&b]);
    b
}

/// Deterministic xorshift fill over (min, max) (same generator as the F0/F1
/// probes).
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

struct Inputs {
    q: MlxArray,
    k_raw: MlxArray,
    comp_k: MlxArray,
    comp_mask: MlxArray,
    sinks_bf16: MlxArray,
}

fn build_inputs(shape: &Shape) -> Inputs {
    let n = shape.n_rows as usize;
    let q = array_bf16_from_f32(
        &fill_uniform(H as usize * D as usize, -1.0, 1.0, 0x9e3779b97f4a7c15),
        &[1, H, 1, D],
    );
    let k_raw = array_bf16_from_f32(
        &fill_uniform(WINDOW as usize * D as usize, -1.0, 1.0, 0x123456789abcdef0),
        &[1, 1, WINDOW, D],
    );
    let comp_k = array_bf16_from_f32(
        &fill_uniform(n * D as usize, -1.0, 1.0, 0xfedcba0987654321),
        &[1, 1, shape.n_rows, D],
    );
    let mask_data: Vec<bool> = match shape.mask {
        MaskKind::AllVisible => vec![true; n],
        MaskKind::Top512 => (0..n)
            .map(|i| ((i as u64).wrapping_mul(7919) % n as u64) < 512)
            .collect(),
    };
    let comp_mask = array_bool(&mask_data, &[1, shape.n_rows]);
    // Production stores sinks f32 and casts to the q dtype at the call site.
    let sinks_f32 = array_f32(
        &fill_uniform(H as usize, -2.0, 2.0, 0x0f0f0f0f0f0f0f0f),
        &[H],
    );
    let sinks_bf16 = astype(&sinks_f32, MlxDtype::Bfloat16, None);
    eval(&[&comp_mask, &sinks_bf16]);
    Inputs {
        q,
        k_raw,
        comp_k,
        comp_mask,
        sinks_bf16,
    }
}

/// Production chain, call-for-call: concat K + concat mask + one native
/// SDPA-with-sinks (which dispatches to MLX's unfused fallback at D=512).
fn production_chain(inp: &Inputs) -> MlxArray {
    let raw_mask = array_bool(&vec![true; WINDOW as usize], &[1, WINDOW]);
    let k_all = concatenate(&[&inp.k_raw, &inp.comp_k], 2, None);
    let mask = concatenate(&[&raw_mask, &inp.comp_mask], -1, None);
    scaled_dot_product_attention_with_mask_and_sinks(
        &inp.q,
        &k_all,
        &k_all,
        scale(),
        ScaledDotProductAttentionMask::Array(&mask),
        Some(&inp.sinks_bf16),
        None,
    )
}

/// Just the concat+mask prefix of the production chain (the work F2
/// eliminates); returned array is the K concat so the cost is real.
fn production_prefix(inp: &Inputs) -> (MlxArray, MlxArray) {
    let raw_mask = array_bool(&vec![true; WINDOW as usize], &[1, WINDOW]);
    let k_all = concatenate(&[&inp.k_raw, &inp.comp_k], 2, None);
    let mask = concatenate(&[&raw_mask, &inp.comp_mask], -1, None);
    eval(&[&k_all, &mask]);
    (k_all, mask)
}

/// f32(bf16(x)) with round-to-nearest-even, matching MLX's astype to bf16.
fn bf16_round(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

#[allow(
    clippy::expect_used,
    reason = "the one-output kernel specification fixes the output vector length"
)]
fn apply_one(kernel: &MlxMetalKernel, inputs: &[&MlxArray], out_dtype: MlxDtype) -> MlxArray {
    let mut outputs = kernel.apply_with_template(
        inputs,
        &[KernelOutputSpec {
            shape: vec![1, H, 1, D],
            dtype: out_dtype,
        }],
        &[] as &[KernelTemplateArg<'_>],
        (H * 256, 1, 1),
        (256, 1, 1),
        None,
    );
    outputs.pop().expect("kernel must produce one output")
}

fn fused_metal(kernel: &MlxMetalKernel, inp: &Inputs) -> MlxArray {
    let n = WINDOW + inp.comp_k.shape()[2];
    let iparams = array_i32(&[n, WINDOW], &[2]);
    let fparams = array_f32(&[bf16_round(scale())], &[1]);
    eval(&[&iparams, &fparams]);
    apply_one(
        kernel,
        &[
            &inp.q,
            &inp.k_raw,
            &inp.comp_k,
            &inp.comp_mask,
            &inp.sinks_bf16,
            &iparams,
            &fparams,
        ],
        MlxDtype::Bfloat16,
    )
}

/// Fallback-chain stage arrays (verified decomposition of the native D=512
/// call): masked bf16 scores `[1,1,64,1,N]` and bf16 probs `[1,1,64,1,N]`.
fn fallback_stage_arrays(inp: &Inputs) -> (MlxArray, MlxArray) {
    let n_rows = inp.comp_k.shape()[2];
    let n = WINDOW + n_rows;
    let raw_mask = array_bool(&vec![true; WINDOW as usize], &[1, WINDOW]);
    let k_all = concatenate(&[&inp.k_raw, &inp.comp_k], 2, None);
    let mask = concatenate(&[&raw_mask, &inp.comp_mask], -1, None);

    let q_scaled = multiply(&inp.q, &cached_scalar(scale(), MlxDtype::Bfloat16), None);
    let q_unf = unflatten(&q_scaled, 1, &[1, H], None);
    let kk = expand_dims(&k_all, 2, None);
    let scores = matmul(&q_unf, &transpose(&kk, &[0, 1, 2, 4, 3], None), None);
    let mask4 = broadcast_to(&mask, &[1, H, 1, n], None);
    let mask5 = unflatten(&mask4, -3, &[1, H], None);
    let scores = where_cond(
        &mask5,
        &scores,
        &cached_scalar(BF16_MIN, MlxDtype::Bfloat16),
        None,
    );

    let sinks5 = unflatten(
        &reshape(&inp.sinks_bf16, &[1, H, 1, 1], None),
        1,
        &[1, H],
        None,
    );
    let sink_col = broadcast_to(&sinks5, &[1, 1, H, 1, 1], None);
    let cat = concatenate(&[&sink_col, &scores], -1, None);
    let probs = softmax_precise(&cat, -1, None);
    let probs = slice(
        &probs,
        &[0, 0, 0, 0, 1],
        &[1, 1, H, 1, n + 1],
        &[1, 1, 1, 1, 1],
        None,
    );
    eval(&[&scores, &probs]);
    (scores, probs)
}

/// Pad a `[1,1,64,1,N]` bf16 stage array into a `[64, 8192]` bf16 buffer.
fn pad_stage(stage: &MlxArray, n: i32) -> MlxArray {
    let f = astype(stage, MlxDtype::Float32, None);
    eval(&[&f]);
    let data = f.data_f32();
    let mut padded = vec![0.0_f32; H as usize * ABLATION_STRIDE];
    for h in 0..H as usize {
        let src = &data[h * n as usize..(h + 1) * n as usize];
        padded[h * ABLATION_STRIDE..h * ABLATION_STRIDE + n as usize].copy_from_slice(src);
    }
    let out = array_f32(&padded, &[H, ABLATION_STRIDE as i32]);
    let out = astype(&out, MlxDtype::Bfloat16, None);
    eval(&[&out]);
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// bf16 ULP at the reference max binade: ULP(x) = x * 2^-7.
fn bf16_ulp_at_max(reference: &[f32]) -> f32 {
    let max_abs = reference.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    (max_abs * 2f32.powi(-7)).max(1e-30)
}

fn flips(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

fn to_f32(a: &MlxArray) -> MlxArray {
    let out = astype(a, MlxDtype::Float32, None);
    eval(&[&out]);
    out
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
        "V4 two-source fused decode SDPA probe (H={H}, D={D}, window={WINDOW}, bf16 KV, f32 accum; gate: bit-exact vs production chain)"
    );
    println!(
        "note: at D=512 the production native SDPA runs MLX's unfused fallback op chain (verified op-for-op)"
    );

    let fused_kernel = MlxMetalKernel::new(
        "ax_dsv4_sdpa_two_source_probe",
        &[
            "queries",
            "k_raw",
            "comp_k",
            "comp_mask",
            "sinks",
            "iparams",
            "fparams",
        ],
        &["out"],
        FUSED_SOURCE,
        "",
        true,
    );
    let scores_in_kernel = MlxMetalKernel::new(
        "ax_dsv4_sdpa_scores_in_probe",
        &[
            "queries",
            "k_raw",
            "comp_k",
            "comp_mask",
            "sinks",
            "iparams",
            "scores_in",
        ],
        &["out"],
        SCORES_IN_SOURCE,
        "",
        true,
    );
    let probs_in_kernel = MlxMetalKernel::new(
        "ax_dsv4_sdpa_probs_in_probe",
        &["k_raw", "comp_k", "comp_mask", "probs_in", "iparams"],
        &["out"],
        PROBS_IN_SOURCE,
        "",
        true,
    );

    let mut parity_fail = false;
    for shape in SHAPES {
        let n = WINDOW + shape.n_rows;
        println!("\n=== {} (iters={}) ===", shape.label, shape.iters);
        let inp = build_inputs(shape);

        // Reference: production chain.
        let reference = production_chain(&inp);
        eval(&[&reference]);

        // Fused two-source kernel.
        let fused = fused_metal(&fused_kernel, &inp);
        eval(&[&fused]);

        let ref_f = to_f32(&reference);
        let fus_f = to_f32(&fused);
        let diff = max_abs_diff(ref_f.data_f32(), fus_f.data_f32());
        let flip = flips(ref_f.data_f32(), fus_f.data_f32());
        let ulp = bf16_ulp_at_max(ref_f.data_f32());
        let total = ref_f.data_f32().len();
        println!(
            "  parity: flips={flip}/{total} ({:.1}%) max abs diff {diff:.6e} = {:.2} bf16 ULP@max",
            100.0 * flip as f64 / total as f64,
            diff / ulp
        );
        if diff != 0.0 {
            parity_fail = true;
        }

        // Ablation A: fallback-exact scores in — softmax+V-dot isolation.
        let (scores_fb, probs_fb) = fallback_stage_arrays(&inp);
        let scores_in = pad_stage(&scores_fb, n);
        let abl_a = {
            let iparams = array_i32(&[n, WINDOW], &[2]);
            eval(&[&iparams]);
            apply_one(
                &scores_in_kernel,
                &[
                    &inp.q,
                    &inp.k_raw,
                    &inp.comp_k,
                    &inp.comp_mask,
                    &inp.sinks_bf16,
                    &iparams,
                    &scores_in,
                ],
                MlxDtype::Bfloat16,
            )
        };
        let abl_a_f = to_f32(&abl_a);
        let diff_a = max_abs_diff(ref_f.data_f32(), abl_a_f.data_f32());
        let flip_a = flips(ref_f.data_f32(), abl_a_f.data_f32());
        println!(
            "  ablation scores-in (softmax+V only): flips={flip_a}/{total} ({:.2}%) max abs diff {diff_a:.6e}",
            100.0 * flip_a as f64 / total as f64
        );

        // Ablation B: fallback-exact probs in — V-dot only isolation.
        let probs_in = pad_stage(&probs_fb, n);
        let abl_b = {
            let iparams = array_i32(&[n, WINDOW], &[2]);
            eval(&[&iparams]);
            apply_one(
                &probs_in_kernel,
                &[&inp.k_raw, &inp.comp_k, &inp.comp_mask, &probs_in, &iparams],
                MlxDtype::Bfloat16,
            )
        };
        let abl_b_f = to_f32(&abl_b);
        let diff_b = max_abs_diff(ref_f.data_f32(), abl_b_f.data_f32());
        let flip_b = flips(ref_f.data_f32(), abl_b_f.data_f32());
        println!(
            "  ablation probs-in (V-dot only): flips={flip_b}/{total} ({:.3}%) max abs diff {diff_b:.6e}",
            100.0 * flip_b as f64 / total as f64
        );

        // Dispatch evidence.
        let prev = op_count_snapshot();
        let counted = production_chain(&inp);
        eval(&[&counted]);
        let wrappers = op_count_take(prev);
        println!(
            "  dispatches: production chain {wrappers} measured wrapper calls (concat K + concat mask + SDPA); fused = 1 Metal dispatch"
        );

        let prefix_us = time_loop("prefix (concat K + mask)", shape.iters, || {
            let _ = production_prefix(&inp);
        });
        let chain_us = time_loop("production chain", shape.iters, || {
            let y = production_chain(&inp);
            eval(&[&y]);
        });
        let fused_us = time_loop("fused two-source", shape.iters, || {
            let y = fused_metal(&fused_kernel, &inp);
            eval(&[&y]);
        });

        let delta_pct = (chain_us - fused_us) / chain_us * 100.0;
        println!(
            "  verdict: fused is {delta_pct:+.1}% vs production chain ({chain_us:.2} -> {fused_us:.2} us/iter; concat+mask prefix {prefix_us:.2} us/iter eliminated)"
        );
    }

    if parity_fail {
        println!(
            "\nRESULT: parity NOT bit-exact (see per-shape flip rates / ULP bounds) — do not wire; softmax-stage reduction order is the dominant divergence (ablations), followed by GEMM tile order"
        );
    } else {
        println!(
            "\nRESULT: parity bit-exact on all shapes; see per-shape verdicts for the wiring decision"
        );
    }
}
