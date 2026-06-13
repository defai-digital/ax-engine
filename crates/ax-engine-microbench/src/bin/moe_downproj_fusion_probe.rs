//! MoE down-projection fusion ceiling probe.
//!
//! Question: at single-stream decode shapes, is it worth fusing the MoE
//! down-projection (`gather_qmm`) with the routing-weighted sum into a single
//! custom Metal kernel (the "gather_qmm with weighted-sum epilogue" idea from
//! the Apple Core AI MoE article)?
//!
//! The article's win came from a *bandwidth bug*: Apple's `GatherMM` read all
//! experts. AX already reads only routed experts via MLX `gather_qmm`, so the
//! only thing a fused kernel can recover here is the intermediate
//! `[.., top_k, hidden]` down-projection write + the separate weighted-sum
//! dispatch. This probe bounds that recoverable time WITHOUT reimplementing
//! MLX's tuned int4 gather-matmul: it measures the real `gather_qmm` and the
//! real weighted-sum kernel at decode shapes, and reports
//!
//!     ceiling = t(gather_qmm + weighted_sum) - t(gather_qmm)
//!
//! i.e. everything the weighted-sum adds on top of the matmul. The true fused
//! win is strictly LESS than this (a fused kernel still pays the reduction
//! compute and the final write), so this is a hard upper bound. If even the
//! upper bound is a negligible fraction of a decode step, the fusion is not
//! worth building.
//!
//! Run:
//!   cargo run -p ax-engine-microbench --release --bin moe-downproj-fusion-probe
//!
//! Shapes default to a fine-grained MoE decode (35B-A3B / GLM-4.7-Flash class).

use std::time::Instant;

use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, astype, eval,
    expand_dims_axes,
    ops::{MlxQuantizationMode, add, gather_qmm, multiply, quantize, sum_axis},
    reshape,
};

// --- Decode shapes (fine-grained MoE, single-stream) -----------------------
const NUM_TOKENS: usize = 1; // batch*seq at decode
const TOP_K: usize = 8;
const HIDDEN: usize = 4096; // model hidden (down-proj OUT dim)
const MOE_INTER: usize = 768; // expert inner dim (down-proj IN dim)
const NUM_EXPERTS: usize = 128;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;
const NUM_LAYERS: usize = 48; // for per-token extrapolation

// Reference decode step for a 35B-A3B-class model: ~65 tok/s => ~15.4 ms/token.
// Used only to express the ceiling as a % of a decode step.
const DECODE_STEP_US: f64 = 1.0e6 / 65.0;

const ITERS: usize = 200;
const CHAIN: usize = 64; // dispatches per eval (amortizes the eval-sync floor)
const THREADS_PER_GROUP: i32 = 256;
const MAX_ABS_DIFF: f32 = 1.0e-3; // f32 correctness tolerance

// Existing production weighted-sum kernel body (verbatim from
// QWEN3_MOE_WEIGHTED_SUM_KERNEL_SOURCE in model/shared/mlp.rs).
const WEIGHTED_SUM_KERNEL_SOURCE: &str = r#"
    uint idx = thread_position_in_grid.x;
    if (idx >= ElementCount) {
        return;
    }

    uint hidden_idx = idx % HiddenDim;
    uint row = idx / HiddenDim;
    uint down_base = row * TopK * HiddenDim + hidden_idx;
    uint weight_base = row * TopK;
    float acc = 0.0f;

    for (uint k = 0; k < TopK; ++k) {
        float y = static_cast<float>(down_out[down_base + k * HiddenDim]);
        float w = static_cast<float>(top_k_weights[weight_base + k]);
        acc += y * w;
    }

    out[idx] = static_cast<OutT>(acc);
"#;

// Fused weighted-sum + residual add (matches the production
// QWEN3_MOE_WEIGHTED_SUM_RESIDUAL_KERNEL_SOURCE in model/shared/mlp.rs).
const WEIGHTED_SUM_RESIDUAL_KERNEL_SOURCE: &str = r#"
    uint idx = thread_position_in_grid.x;
    if (idx >= ElementCount) {
        return;
    }

    uint hidden_idx = idx % HiddenDim;
    uint row = idx / HiddenDim;
    uint down_base = row * TopK * HiddenDim + hidden_idx;
    uint weight_base = row * TopK;
    float acc = 0.0f;

    for (uint k = 0; k < TopK; ++k) {
        float y = static_cast<float>(down_out[down_base + k * HiddenDim]);
        float w = static_cast<float>(top_k_weights[weight_base + k]);
        acc += y * w;
    }

    acc += static_cast<float>(residual[idx]);

    out[idx] = static_cast<OutT>(acc);
"#;

fn xorshift(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 32) as u32
}

fn random_f32(len: usize, seed: u64, mean: f32, scale: f32) -> Vec<f32> {
    let mut state = seed;
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        let r = xorshift(&mut state) as f32 / u32::MAX as f32;
        v.push(r * scale + mean);
    }
    v
}

fn f32_array(data: &[f32], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data),
        shape,
        MlxDtype::Float32,
    )
}

fn bf16_array(data: &[f32], shape: &[i32]) -> MlxArray {
    astype(&f32_array(data, shape), MlxDtype::Bfloat16, None)
}

fn run_weighted_sum(
    kernel: &MlxMetalKernel,
    down_out: &MlxArray,
    weights: &MlxArray,
    dtype: MlxDtype,
) -> MlxArray {
    let element_count = (NUM_TOKENS * HIDDEN) as i32;
    let mut outputs = kernel.apply_with_template(
        &[down_out, weights],
        &[KernelOutputSpec {
            shape: vec![NUM_TOKENS as i32, HIDDEN as i32],
            dtype,
        }],
        &[
            KernelTemplateArg::Dtype {
                name: "OutT",
                dtype,
            },
            KernelTemplateArg::Int {
                name: "TopK",
                value: TOP_K as i32,
            },
            KernelTemplateArg::Int {
                name: "HiddenDim",
                value: HIDDEN as i32,
            },
            KernelTemplateArg::Int {
                name: "ElementCount",
                value: element_count,
            },
        ],
        (element_count, 1, 1),
        (THREADS_PER_GROUP, 1, 1),
        None,
    );
    outputs.pop().expect("weighted-sum kernel must produce output")
}

fn run_weighted_sum_residual(
    kernel: &MlxMetalKernel,
    down_out: &MlxArray,
    weights: &MlxArray,
    residual: &MlxArray,
    dtype: MlxDtype,
) -> MlxArray {
    let element_count = (NUM_TOKENS * HIDDEN) as i32;
    let mut outputs = kernel.apply_with_template(
        &[down_out, weights, residual],
        &[KernelOutputSpec {
            shape: vec![NUM_TOKENS as i32, HIDDEN as i32],
            dtype,
        }],
        &[
            KernelTemplateArg::Dtype {
                name: "OutT",
                dtype,
            },
            KernelTemplateArg::Int {
                name: "TopK",
                value: TOP_K as i32,
            },
            KernelTemplateArg::Int {
                name: "HiddenDim",
                value: HIDDEN as i32,
            },
            KernelTemplateArg::Int {
                name: "ElementCount",
                value: element_count,
            },
        ],
        (element_count, 1, 1),
        (THREADS_PER_GROUP, 1, 1),
        None,
    );
    outputs.pop().expect("residual weighted-sum kernel must produce output")
}

// One down-projection gather_qmm over top_k routed experts:
//   x [tokens, 1, 1, MOE_INTER] @ w[idx].T  ->  [tokens, top_k, HIDDEN]
fn run_downproj(
    x_exp: &MlxArray,
    packed: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
    indices: &MlxArray,
) -> MlxArray {
    let out = gather_qmm(
        x_exp,
        packed,
        scales,
        Some(biases),
        indices,
        true,
        Some(GROUP_SIZE),
        Some(BITS),
        false,
        None,
    );
    // Collapse any broadcast singletons to [tokens, top_k, HIDDEN].
    reshape(
        &out,
        &[NUM_TOKENS as i32, TOP_K as i32, HIDDEN as i32],
        None,
    )
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn main() {
    println!(
        "MoE down-proj fusion ceiling probe (tokens={NUM_TOKENS}, top_k={TOP_K}, \
         hidden={HIDDEN}, moe_inter={MOE_INTER}, experts={NUM_EXPERTS}, {BITS}-bit g{GROUP_SIZE})"
    );
    println!();

    // Synthetic quantized down-projection weights [experts, HIDDEN, MOE_INTER].
    let w_src = random_f32(NUM_EXPERTS * HIDDEN * MOE_INTER, 0xA5A5_1234, 0.0, 0.04);
    let w_bf16 = bf16_array(
        &w_src,
        &[NUM_EXPERTS as i32, HIDDEN as i32, MOE_INTER as i32],
    );
    let parts = quantize(
        &w_bf16,
        Some(GROUP_SIZE),
        Some(BITS),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    assert_eq!(parts.len(), 3, "quantize -> [packed, scales, biases]");
    let (packed, scales, biases) = (&parts[0], &parts[1], &parts[2]);

    // Decode activation into the down projection: [tokens, MOE_INTER] expanded
    // to the SwitchGLU gather layout [tokens, 1, 1, MOE_INTER].
    let x_src = random_f32(NUM_TOKENS * MOE_INTER, 0x1357_BD13, 0.0, 0.5);
    let x_base = bf16_array(&x_src, &[NUM_TOKENS as i32, MOE_INTER as i32]);
    let x_exp = expand_dims_axes(&x_base, &[-2, -3], None);

    // Routed expert ids: top_k distinct experts spread across the pool.
    let idx_data: Vec<u32> = (0..NUM_TOKENS * TOP_K)
        .map(|i| ((i * NUM_EXPERTS / TOP_K) % NUM_EXPERTS) as u32)
        .collect();
    let indices = MlxArray::from_raw_data(
        idx_data.as_ptr().cast(),
        std::mem::size_of_val(idx_data.as_slice()),
        &[NUM_TOKENS as i32, TOP_K as i32],
        MlxDtype::Uint32,
    );

    let weight_src = random_f32(NUM_TOKENS * TOP_K, 0x2468_ACE0, 0.0, 1.0);
    let weights = bf16_array(&weight_src, &[NUM_TOKENS as i32, TOP_K as i32]);

    let ws_kernel = MlxMetalKernel::new(
        "ax_moe_downproj_fusion_probe_ws",
        &["down_out", "top_k_weights"],
        &["out"],
        WEIGHTED_SUM_KERNEL_SOURCE,
        "",
        true,
    );

    // --- Correctness: kernel weighted-sum vs MLX reduce -------------------
    // Run the gate fully in f32 (same down_out feeds kernel and reference) so
    // any disagreement is layout/logic, not bf16 accumulation noise.
    println!("=== Weighted-sum kernel correctness (f32, vs MLX multiply+sum) ===");
    let down_out = run_downproj(&x_exp, packed, scales, biases, &indices);
    let down_f32 = astype(&down_out, MlxDtype::Float32, None);
    let weights_f32 = astype(&weights, MlxDtype::Float32, None);
    eval(&[&down_f32, &weights_f32]);

    let kernel_out = run_weighted_sum(&ws_kernel, &down_f32, &weights_f32, MlxDtype::Float32);
    let w_exp = expand_dims_axes(&weights_f32, &[-1], None); // [tokens, top_k, 1]
    let ref_out = sum_axis(&multiply(&down_f32, &w_exp, None), 1, false, None);
    eval(&[&kernel_out, &ref_out]);

    let diff = max_abs_diff(kernel_out.data_f32(), ref_out.data_f32());
    println!("  max abs diff: {diff:.5} (tolerance {MAX_ABS_DIFF:.5})");
    if diff > MAX_ABS_DIFF {
        println!("  status: FAIL");
        std::process::exit(1);
    }
    println!("  status: ok");
    println!();

    // --- Timing -----------------------------------------------------------
    // Per-`eval` sync (~150us) dwarfs these single-token kernels, so wrapping
    // each call in its own eval just measures the sync floor. Production builds
    // the whole per-token layer graph and evals ONCE — so we amortize: build
    // CHAIN dispatches into a single eval and divide. Distinct inputs per slot
    // defeat MLX common-subexpression elision.
    let xs: Vec<MlxArray> = (0..CHAIN)
        .map(|i| {
            let d = random_f32(NUM_TOKENS * MOE_INTER, 0x1357_BD13 ^ (i as u64).wrapping_mul(0x9E37), 0.0, 0.5);
            expand_dims_axes(&bf16_array(&d, &[NUM_TOKENS as i32, MOE_INTER as i32]), &[-2, -3], None)
        })
        .collect();
    let ws_v: Vec<MlxArray> = (0..CHAIN)
        .map(|i| {
            let d = random_f32(NUM_TOKENS * TOP_K, 0x2468_ACE0 ^ (i as u64).wrapping_mul(0x85EB), 0.0, 1.0);
            bf16_array(&d, &[NUM_TOKENS as i32, TOP_K as i32])
        })
        .collect();
    // Pre-materialize distinct down_out inputs for the ws-only measurement.
    let down_v: Vec<MlxArray> = xs
        .iter()
        .map(|x| run_downproj(x, packed, scales, biases, &indices))
        .collect();
    eval(&down_v.iter().collect::<Vec<_>>());

    let time_amortized = |label: &str, mut build: Box<dyn FnMut(usize) -> MlxArray>| -> f64 {
        let warm: Vec<MlxArray> = (0..CHAIN).map(&mut build).collect();
        eval(&warm.iter().collect::<Vec<_>>());
        let t0 = Instant::now();
        for _ in 0..ITERS {
            let outs: Vec<MlxArray> = (0..CHAIN).map(&mut build).collect();
            eval(&outs.iter().collect::<Vec<_>>());
        }
        let per = t0.elapsed().as_secs_f64() * 1.0e6 / (ITERS * CHAIN) as f64;
        println!("  {label}: {per:.4} us/dispatch (chain={CHAIN})");
        per
    };

    println!("=== In-graph cost, amortized over {CHAIN} dispatches/eval (iters={ITERS}) ===");
    let t_down = time_amortized(
        "gather_qmm (down-proj only)",
        Box::new(|i| run_downproj(&xs[i], packed, scales, biases, &indices)),
    );
    let t_ws = time_amortized(
        "weighted-sum kernel only",
        Box::new(|i| run_weighted_sum(&ws_kernel, &down_v[i], &ws_v[i], MlxDtype::Bfloat16)),
    );
    let t_comb = time_amortized(
        "gather_qmm + weighted-sum (current)",
        Box::new(|i| {
            let d = run_downproj(&xs[i], packed, scales, biases, &indices);
            run_weighted_sum(&ws_kernel, &d, &ws_v[i], MlxDtype::Bfloat16)
        }),
    );

    // --- Verdict ----------------------------------------------------------
    // Upper bound on the fused win per down-stage: everything the weighted-sum
    // adds on top of the matmul (intermediate write + dispatch + read + reduce).
    // The true fused kernel still pays reduce + final write, so real < this.
    let ceiling_per_stage = (t_comb - t_down).max(0.0);
    let ceiling_per_token = ceiling_per_stage * NUM_LAYERS as f64;
    let pct_of_step = ceiling_per_token / DECODE_STEP_US * 100.0;

    println!();
    println!("=== gather_qmm-epilogue ceiling (NEEDS a matmul rewrite to claim) ===");
    println!("  weighted-sum share of down-stage: {:.1}%", t_ws / t_comb * 100.0);
    println!(
        "  ceiling / token (x{NUM_LAYERS} layers): {ceiling_per_token:.1} us \
         ({pct_of_step:.2}% of a {DECODE_STEP_US:.0} us decode step)"
    );
    // IMPORTANT: this "ceiling" is mostly the weighted-sum kernel's OWN dispatch
    // (the reduction) — mandatory work. Recovering it means folding the reduce
    // into gather_qmm, i.e. reimplementing MLX's tuned int4 matmul (rejected:
    // likely slower than MLX). It is NOT the residual-add, which the A/B below
    // shows is already ~free. Do not read this number as a realizable win.
    println!(
        "  ^ mostly the reduction's own dispatch; realizable only by rewriting\n    \
         MLX's int4 gather-matmul. The residual-tail A/B below is the test that\n    \
         matters for a no-matmul-rewrite fusion."
    );

    // === The actual shipped change: fuse the weighted-sum + residual add =====
    // Capture the ceiling WITHOUT touching gather_qmm: replace the standalone
    // {weighted-sum kernel + MLX residual add} (two dispatches) with one fused
    // weighted-sum-residual kernel. This is what AX_MLX_MOE_FUSED_RESIDUAL does.
    println!();
    println!("=== Tail fusion A/B: {{ws + add}} vs fused {{ws+residual}} ===");
    let res_v: Vec<MlxArray> = (0..CHAIN)
        .map(|i| {
            let d = random_f32(NUM_TOKENS * HIDDEN, 0x0F0F_5A5A ^ (i as u64).wrapping_mul(0xC2B2), 0.0, 0.5);
            bf16_array(&d, &[NUM_TOKENS as i32, HIDDEN as i32])
        })
        .collect();
    let ws_res_kernel = MlxMetalKernel::new(
        "ax_moe_downproj_fusion_probe_ws_residual",
        &["down_out", "top_k_weights", "residual"],
        &["out"],
        WEIGHTED_SUM_RESIDUAL_KERNEL_SOURCE,
        "",
        true,
    );

    // Bit-exactness: fused kernel == ws-kernel-then-add (both in f32).
    let unfused_ref = {
        let ws = run_weighted_sum(&ws_kernel, &down_f32, &weights_f32, MlxDtype::Float32);
        add(&astype(&res_v[0], MlxDtype::Float32, None), &ws, None)
    };
    let fused_ref = run_weighted_sum_residual(
        &ws_res_kernel,
        &down_f32,
        &weights_f32,
        &astype(&res_v[0], MlxDtype::Float32, None),
        MlxDtype::Float32,
    );
    eval(&[&unfused_ref, &fused_ref]);
    let tail_diff = max_abs_diff(fused_ref.data_f32(), unfused_ref.data_f32());
    println!("  fused vs unfused max abs diff: {tail_diff:.6} (must be 0)");
    if tail_diff != 0.0 {
        println!("  status: FAIL");
        std::process::exit(1);
    }

    let t_unfused = time_amortized(
        "unfused: ws kernel + MLX add",
        Box::new(|i| {
            let ws = run_weighted_sum(&ws_kernel, &down_v[i], &ws_v[i], MlxDtype::Bfloat16);
            add(&res_v[i], &ws, None)
        }),
    );
    let t_fused = time_amortized(
        "fused:   ws+residual kernel",
        Box::new(|i| {
            run_weighted_sum_residual(&ws_res_kernel, &down_v[i], &ws_v[i], &res_v[i], MlxDtype::Bfloat16)
        }),
    );
    let saved_per_layer = (t_unfused - t_fused).max(0.0);
    let saved_per_token = saved_per_layer * NUM_LAYERS as f64;
    let saved_pct = saved_per_token / DECODE_STEP_US * 100.0;
    println!();
    println!(
        "  measured tail saving: {saved_per_layer:.3} us/layer -> {saved_per_token:.1} us/token \
         ({saved_pct:.2}% of a {DECODE_STEP_US:.0} us decode step)"
    );
    if saved_pct < 0.5 {
        println!(
            "  REJECT: fusing the residual add into the weighted-sum saves ~nothing.\n  \
             MLX's `add` is already ~free; the fused kernel's extra residual read\n  \
             offsets it. Not worth the kernel + wiring complexity."
        );
    } else {
        println!("  PROMISING: >=0.5% saving; worth an end-to-end A/B on a real MoE model.");
    }
    println!(
        "  NOTE: synthetic per-dispatch cost; an end-to-end run on a real MoE model\n  \
         is the final arbiter, but the synthetic test OVER-states per-dispatch overhead\n  \
         (64 dispatches/eval), so the real saving is <= this."
    );
}
