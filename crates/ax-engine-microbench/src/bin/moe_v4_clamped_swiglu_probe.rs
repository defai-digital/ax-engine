//! ADR-028 Phase 1 probe: DeepSeek V4 clamped SwiGLU fused Metal kernel.
//!
//! V4's `silu(min(gate, limit)) * clip(up, ±limit)` deliberately blocks every
//! fused-activation fast path (`skip_fused_silu`), so V4 MoE decode currently
//! runs slice + slice + minimum + clip + silu_mul (or the compiled-closure
//! equivalent) per MoE layer per token. This probe A/Bs that current path
//! against a single packed Metal dispatch, with numerical parity as the hard
//! gate (the fused-router promotion died on parity; ADR-003).
//!
//! Run:
//!   cargo run -p ax-engine-microbench --release --bin moe-v4-clamped-swiglu-probe
//!
//! Output: per-shape parity + wall-clock verdict for the ADR-028 Phase 1
//! record.

use std::time::Instant;

use mlx_sys::ops::cached_scalar;
use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxClosure, MlxDtype, MlxMetalKernel,
    MlxVectorArray, astype, clip, eval, minimum, silu_mul, slice_last_dim,
};

/// `deepseek_v4.swiglu_limit` is pack-configured; 7.0 is the value llama.cpp
/// / vLLM ship for DeepSeek V4, used here as the probe point.
const LIMIT: f32 = 7.0;
/// Encoded as int*1e6 so the kernel can reconstruct it via a template arg
/// (the template API has no float variant; 7.0*1e9 would overflow i32).
const LIMIT_E6: i32 = (LIMIT * 1.0e6) as i32;
const DECODE_ITERS: usize = 2000;
const PREFILL_ITERS: usize = 300;
const THREADS_PER_GROUP: i32 = 256;

const KERNEL_SOURCE: &str = r#"
    uint idx = thread_position_in_grid.x;
    if (idx >= ElementCount) {
        return;
    }

    uint col = idx % HiddenDim;
    uint row = idx / HiddenDim;
    uint gate_idx = row * (HiddenDim * 2) + col;
    uint up_idx = gate_idx + HiddenDim;

    const float limit = float(LIMIT_E6) * 1.0e-6f;
    float gate_v = min(static_cast<float>(gate_up[gate_idx]), limit);
    float up_v = clamp(static_cast<float>(gate_up[up_idx]), -limit, limit);
    float activated = gate_v / (1.0f + exp(-gate_v));
    out[idx] = static_cast<T>(activated * up_v);
"#;

struct Shape {
    label: &'static str,
    seq: i32,
    top_k: i32,
    inter: i32,
    iters: usize,
}

const SHAPES: &[Shape] = &[
    Shape {
        label: "decode k=6 inter=512",
        seq: 1,
        top_k: 6,
        inter: 512,
        iters: DECODE_ITERS,
    },
    Shape {
        label: "decode k=6 inter=1024",
        seq: 1,
        top_k: 6,
        inter: 1024,
        iters: DECODE_ITERS,
    },
    Shape {
        label: "decode k=8 inter=2048",
        seq: 1,
        top_k: 8,
        inter: 2048,
        iters: DECODE_ITERS,
    },
    Shape {
        label: "prefill s=128 k=6 inter=1024",
        seq: 128,
        top_k: 6,
        inter: 1024,
        iters: PREFILL_ITERS,
    },
];

fn build_input(shape: &Shape) -> MlxArray {
    // Deterministic uniform(-10, 10) so a large share of values engages the
    // clamp at limit 7.0. Built in f32, then cast to bf16 like the real path.
    let dims = [1, shape.seq, shape.top_k, 1, 2 * shape.inter];
    let count: usize = dims.iter().map(|d| *d as usize).product();
    let mut data = Vec::with_capacity(count);
    let mut state: u64 = 0x9e3779b97f4a7c15;
    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let raw = (state >> 32) as u32;
        data.push((raw as f32 / u32::MAX as f32) * 20.0 - 10.0);
    }
    let f32_arr = MlxArray::from_raw_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data.as_slice()),
        &dims,
        MlxDtype::Float32,
    );
    let bf16 = astype(&f32_arr, MlxDtype::Bfloat16, None);
    eval(&[&bf16]);
    bf16
}

/// Current production path: split slices + clamped imperative ops.
fn current_ops(gate_up: &MlxArray, half: i32) -> MlxArray {
    let gate = slice_last_dim(gate_up, 0, half, None);
    let up = slice_last_dim(gate_up, half, half * 2, None);
    let pos = cached_scalar(LIMIT, gate_up.dtype());
    let neg = cached_scalar(-LIMIT, gate_up.dtype());
    let gate_c = minimum(&gate, &pos, None);
    let up_c = clip(&up, &neg, &pos, None);
    silu_mul(&gate_c, &up_c, None)
}

/// All-f32 reference chain: formula ground truth for the parity gate.
fn reference_f32(gate_up: &MlxArray, half: i32) -> MlxArray {
    let packed_f32 = astype(gate_up, MlxDtype::Float32, None);
    let gate = slice_last_dim(&packed_f32, 0, half, None);
    let up = slice_last_dim(&packed_f32, half, half * 2, None);
    let pos = cached_scalar(LIMIT, MlxDtype::Float32);
    let neg = cached_scalar(-LIMIT, MlxDtype::Float32);
    let gate_c = minimum(&gate, &pos, None);
    let up_c = clip(&up, &neg, &pos, None);
    let out = silu_mul(&gate_c, &up_c, None);
    eval(&[&out]);
    out
}

/// Current production fast path: the same chain as a compiled MLX closure
/// (cached across calls like `try_compiled_v4_clamped_swiglu`).
fn current_compiled(closure: &MlxClosure, gate_up: &MlxArray, half: i32) -> Option<MlxArray> {
    let gate = slice_last_dim(gate_up, 0, half, None);
    let up = slice_last_dim(gate_up, half, half * 2, None);
    let mut outputs = closure.try_apply(&[&gate, &up]).ok()?;
    (outputs.len() == 1).then(|| outputs.pop().unwrap_or_else(|| unreachable!()))
}

#[allow(
    clippy::expect_used,
    reason = "the one-output kernel specification fixes the output vector length"
)]
fn fused_metal(kernel: &MlxMetalKernel, gate_up: &MlxArray, half: i32) -> MlxArray {
    let shape = gate_up.shape();
    let mut out_shape = shape.clone();
    *out_shape.last_mut().expect("rank >= 1") = half;
    let element_count: i32 = out_shape.iter().product();
    let outputs = kernel.apply_with_template(
        &[gate_up],
        &[KernelOutputSpec {
            shape: out_shape,
            dtype: gate_up.dtype(),
        }],
        &[
            KernelTemplateArg::Dtype {
                name: "T",
                dtype: gate_up.dtype(),
            },
            KernelTemplateArg::Int {
                name: "HiddenDim",
                value: half,
            },
            KernelTemplateArg::Int {
                name: "ElementCount",
                value: element_count,
            },
            KernelTemplateArg::Int {
                name: "LIMIT_E6",
                value: LIMIT_E6,
            },
        ],
        (element_count, 1, 1),
        (THREADS_PER_GROUP, 1, 1),
        None,
    );
    outputs
        .into_iter()
        .next()
        .expect("kernel must produce one output")
}

fn to_f32(a: &MlxArray) -> MlxArray {
    let out = astype(a, MlxDtype::Float32, None);
    eval(&[&out]);
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// bf16 ULP at the reference magnitude: ULP(x) = x * 2^-7 for the max binade.
/// A bf16-rounded value sits within half a ULP of the f32 truth; two bf16
/// values computed through different rounding paths sit within ~1 ULP.
fn ulp_at_max(reference: &[f32]) -> f32 {
    let max_abs = reference.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    (max_abs * 2f32.powi(-7)).max(1e-4)
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

#[allow(
    clippy::expect_used,
    reason = "probe binary: a closure/apply failure is a hard probe failure, \
              and the one-output kernel specification fixes the output length"
)]
fn main() {
    println!(
        "V4 clamped SwiGLU fused-kernel probe (limit={LIMIT}, dtype=bf16, gates are ULP-derived)"
    );

    let kernel = MlxMetalKernel::new(
        "ax_dsv4_packed_clamped_swiglu_probe",
        &["gate_up"],
        &["out"],
        KERNEL_SOURCE,
        "",
        true,
    );
    let compiled = MlxClosure::new_dyn(|inputs: &MlxVectorArray| {
        let gate = inputs.get(0);
        let up = inputs.get(1);
        let pos = cached_scalar(LIMIT, gate.dtype());
        let neg = cached_scalar(-LIMIT, gate.dtype());
        let gate_c = minimum(&gate, &pos, None);
        let up_c = clip(&up, &neg, &pos, None);
        vec![silu_mul(&gate_c, &up_c, None)]
    })
    .compile(true)
    .expect("compiled clamped SwiGLU closure must build");

    let mut parity_fail = false;
    for shape in SHAPES {
        println!("\n=== {} (iters={}) ===", shape.label, shape.iters);
        let packed = build_input(shape);
        let half = shape.inter;

        // Parity: formula gate in f32, then rounding-level agreement vs the
        // two bf16 production variants.
        let ref_f32 = reference_f32(&packed, half);
        let ops_out = current_ops(&packed, half);
        let fused_out = fused_metal(&kernel, &packed, half);
        let compiled_out = current_compiled(&compiled, &packed, half).expect("compiled path");
        let fused_f = to_f32(&fused_out);
        let ops_f = to_f32(&ops_out);
        let compiled_f = to_f32(&compiled_out);
        let diff_formula = max_abs_diff(ref_f32.data_f32(), fused_f.data_f32());
        let diff_vs_compiled = max_abs_diff(compiled_f.data_f32(), fused_f.data_f32());
        let diff_vs_ops = max_abs_diff(ops_f.data_f32(), fused_f.data_f32());
        let ulp = ulp_at_max(ref_f32.data_f32());
        // Formula gate: fused (bf16-rounded) vs f32 truth within half a ULP
        // (×1.5 safety). Rounding gate: fused vs compiled bf16 within ~1 ULP.
        let formula_tol = ulp * 0.75;
        let rounding_tol = ulp * 1.5;
        println!(
            "  parity: vs f32 ref {diff_formula:.6} (gate {formula_tol:.6}); \
             vs compiled bf16 {diff_vs_compiled:.6} (gate {rounding_tol:.6}); vs ops bf16 {diff_vs_ops:.6}"
        );
        if diff_formula > formula_tol || diff_vs_compiled > rounding_tol {
            parity_fail = true;
            println!("  parity: FAIL");
            continue;
        }
        println!("  parity: ok (formula exact; bf16 delta is rounding-level)");

        let ops_us = time_loop("current ops", shape.iters, || {
            let y = current_ops(&packed, half);
            eval(&[&y]);
        });
        let compiled_us = time_loop("current compiled", shape.iters, || {
            let y = current_compiled(&compiled, &packed, half).expect("compiled path");
            eval(&[&y]);
        });
        let fused_us = time_loop("fused metal", shape.iters, || {
            let y = fused_metal(&kernel, &packed, half);
            eval(&[&y]);
        });

        let best_baseline = ops_us.min(compiled_us);
        let delta_pct = (best_baseline - fused_us) / best_baseline * 100.0;
        println!(
            "  verdict: fused is {delta_pct:+.1}% vs best current ({best_baseline:.2} -> {fused_us:.2} us/iter)"
        );
    }

    if parity_fail {
        println!("\nRESULT: FAIL — parity broken on at least one shape; do not wire");
        std::process::exit(1);
    }
    println!(
        "\nRESULT: parity holds on all shapes; see per-shape verdicts for the wiring decision"
    );
}
