//! Phase 1 async-overlap probe (decode-dispatch-efficiency plan, item 1).
//!
//! `docs/performance/phase0-overlap-bandwidth-mtp.md` measured that the
//! direct-decode double buffer achieves no host/GPU overlap: injected host
//! delay is never absorbed and `mlx_async_eval` behaves as the de-facto
//! barrier. This probe isolates WHERE that serialization lives by removing
//! everything model-specific (KV cache, sampling, token readback) and timing
//! three loop shapes over a synthetic dependent-chain workload:
//!
//!   serial      — build step i, `eval(step_i)`, repeat (no async at all).
//!   pipelined   — the decode shape: build step i+1 from the *unevaluated*
//!                 step i output, `async_eval(step_{i+1})`, `eval(step_i)`.
//!   independent — same ordering as `pipelined` but each step's input is a
//!                 fresh constant array, so no cross-step data dependency.
//!
//! Each shape runs with an injected per-iteration host busy-spin S. If a
//! shape overlaps host work with GPU execution, its per-iteration wall stays
//! ~flat until S exceeds the GPU step time; if it serializes, wall grows by
//! ~S. Comparing `pipelined` vs `independent` further separates "MLX
//! async_eval semantics" from "dependency-triggered synchronization".
//!
//! Run:
//!   cargo run -p ax-engine-microbench --release --bin async-eval-overlap-probe
//!
//! Output: one table row per (shape, spin) plus async/eval bucket means.

use std::hint;
use std::time::Instant;

use std::sync::OnceLock;

use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, MlxQuantizationMode,
    add, argpartition_axis, astype, async_eval, eval, gather_qmm, matmul, quantize, reshape,
    slice_last_dim, sum_axis, tanh,
};

/// A trivial element-wise custom Metal kernel, the one op class the real AX
/// decode forward has (~17 per step) that none of the built-in-op repros do.
/// Tests whether custom-kernel dispatch is what defeats async_eval overlap.
static PROBE_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();

const PROBE_KERNEL_SOURCE: &str = r#"
    const uint i = thread_position_in_grid.x;
    if (i >= SIZE) { return; }
    out[i] = x[i];
"#;

fn probe_kernel() -> &'static MlxMetalKernel {
    PROBE_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_async_probe_identity",
            &["x"],
            &["out"],
            PROBE_KERNEL_SOURCE,
            "#include <metal_stdlib>\nusing namespace metal;",
            false,
        )
    })
}

fn apply_probe_kernel(x: &MlxArray) -> MlxArray {
    let tpg = 256usize;
    let grid = GQ_H.div_ceil(tpg) * tpg;
    probe_kernel()
        .apply_with_template(
            &[x],
            &[KernelOutputSpec {
                shape: vec![1, GQ_H as i32],
                dtype: MlxDtype::Bfloat16,
            }],
            &[KernelTemplateArg::Int {
                name: "SIZE",
                value: GQ_H as i32,
            }],
            (grid as i32, 1, 1),
            (tpg as i32, 1, 1),
            None,
        )
        .into_iter()
        .next()
        .expect("probe kernel must produce one output")
}

const N: usize = 3072;
const WARMUP: usize = 3;
const ITERS: usize = 24;
const SPINS_US: [u64; 4] = [0, 2_000, 4_000, 8_000];

// gather_qmm shapes: a 24-layer matvec chain mirroring batch=1 MoE decode.
const GQ_LAYERS: usize = 24;
const GQ_H: usize = 3072;
const GQ_EXPERTS: usize = 64;
const GQ_TOPK: usize = 8;

fn spin_us(dur_us: u64) {
    if dur_us == 0 {
        return;
    }
    let start = Instant::now();
    while (start.elapsed().as_micros() as u64) < dur_us {
        hint::spin_loop();
    }
}

fn constant_matrix(value: f32) -> MlxArray {
    let host = vec![value; N * N];
    let f32_arr = MlxArray::from_raw_data(
        host.as_ptr() as *const u8,
        host.len() * std::mem::size_of::<f32>(),
        &[N as i32, N as i32],
        MlxDtype::Float32,
    );
    let bf16 = astype(&f32_arr, MlxDtype::Bfloat16, None);
    eval(&[&bf16]);
    bf16
}

/// One synthetic "decode step": two big matmuls with tanh squashes so the
/// state magnitude stays bounded across arbitrarily many chained steps.
fn step(x: &MlxArray, w1: &MlxArray, w2: &MlxArray) -> MlxArray {
    let h = tanh(&matmul(x, w1, None), None);
    tanh(&matmul(&h, w2, None), None)
}

struct ShapeResult {
    per_iter_us: f64,
    async_us: f64,
    eval_us: f64,
}

fn run_serial(x0: &MlxArray, w1: &MlxArray, w2: &MlxArray, spin: u64) -> ShapeResult {
    let mut x = x0.clone();
    for _ in 0..WARMUP {
        x = step(&x, w1, w2);
        eval(&[&x]);
    }
    let mut eval_us = 0u128;
    let started = Instant::now();
    for _ in 0..ITERS {
        spin_us(spin);
        x = step(&x, w1, w2);
        let t = Instant::now();
        eval(&[&x]);
        eval_us += t.elapsed().as_micros();
    }
    ShapeResult {
        per_iter_us: started.elapsed().as_micros() as f64 / ITERS as f64,
        async_us: 0.0,
        eval_us: eval_us as f64 / ITERS as f64,
    }
}

fn run_pipelined(
    x0: &MlxArray,
    w1: &MlxArray,
    w2: &MlxArray,
    spin: u64,
    independent: bool,
) -> ShapeResult {
    // Bootstrap: pending = first step, submitted but not awaited — mirrors
    // `start_direct_pipeline`.
    let fresh = constant_matrix(0.01);
    let mut pending = step(x0, w1, w2);
    async_eval(&[&pending]);
    for _ in 0..WARMUP {
        let input = if independent {
            fresh.clone()
        } else {
            pending.clone()
        };
        let next = step(&input, w1, w2);
        async_eval(&[&next]);
        eval(&[&pending]);
        pending = next;
    }
    let mut async_us = 0u128;
    let mut eval_us = 0u128;
    let started = Instant::now();
    for _ in 0..ITERS {
        spin_us(spin);
        let input = if independent {
            fresh.clone()
        } else {
            pending.clone()
        };
        let next = step(&input, w1, w2);
        let t = Instant::now();
        async_eval(&[&next]);
        async_us += t.elapsed().as_micros();
        let t = Instant::now();
        eval(&[&pending]);
        eval_us += t.elapsed().as_micros();
        pending = next;
    }
    let total = started.elapsed().as_micros() as f64 / ITERS as f64;
    // Drain the trailing pending step so it does not leak into the next config.
    eval(&[&pending]);
    ShapeResult {
        per_iter_us: total,
        async_us: async_us as f64 / ITERS as f64,
        eval_us: eval_us as f64 / ITERS as f64,
    }
}

/// Weights for the gather_qmm chain: one shared 4-bit expert stack
/// `[E, H, H]`, a router projection `[H, E]`, and a dense ballast `[H, H]`.
struct GatherChain {
    wq: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
    router: MlxArray,
    dense: MlxArray,
    const_idx: MlxArray,
}

fn build_gather_chain() -> GatherChain {
    let host: Vec<f32> = (0..GQ_EXPERTS * GQ_H * GQ_H)
        .map(|i| ((i % 173) as f32 - 86.0) * 1.0e-4)
        .collect();
    let experts = MlxArray::from_raw_data(
        host.as_ptr() as *const u8,
        host.len() * std::mem::size_of::<f32>(),
        &[GQ_EXPERTS as i32, GQ_H as i32, GQ_H as i32],
        MlxDtype::Float32,
    );
    let experts = astype(&experts, MlxDtype::Bfloat16, None);
    let mut quantized = quantize(
        &experts,
        Some(64),
        Some(4),
        MlxQuantizationMode::Affine,
        None,
        None,
    )
    .into_iter();
    let wq = quantized.next().expect("quantize returns weight");
    let scales = quantized.next().expect("quantize returns scales");
    let biases = quantized.next().expect("quantize returns biases");

    let router_host = vec![0.01f32; GQ_H * GQ_EXPERTS];
    let router = MlxArray::from_raw_data(
        router_host.as_ptr() as *const u8,
        router_host.len() * std::mem::size_of::<f32>(),
        &[GQ_H as i32, GQ_EXPERTS as i32],
        MlxDtype::Float32,
    );
    let router = astype(&router, MlxDtype::Bfloat16, None);

    let dense_host = vec![0.005f32; GQ_H * GQ_H];
    let dense = MlxArray::from_raw_data(
        dense_host.as_ptr() as *const u8,
        dense_host.len() * std::mem::size_of::<f32>(),
        &[GQ_H as i32, GQ_H as i32],
        MlxDtype::Float32,
    );
    let dense = astype(&dense, MlxDtype::Bfloat16, None);

    let idx_host: Vec<u32> = (0..GQ_TOPK as u32)
        .map(|i| i * 7 % GQ_EXPERTS as u32)
        .collect();
    // 1-D indices: gather_qmm output is [idx..., M, N], so [K] indices with
    // x=[1,H] give [K,1,H] and a single axis-0 sum restores [1,H].
    let const_idx = MlxArray::from_raw_data(
        idx_host.as_ptr() as *const u8,
        idx_host.len() * std::mem::size_of::<u32>(),
        &[GQ_TOPK as i32],
        MlxDtype::Uint32,
    );
    eval(&[&wq, &scales, &biases, &router, &dense, &const_idx]);
    GatherChain {
        wq,
        scales,
        biases,
        router,
        dense,
        const_idx,
    }
}

/// One synthetic MoE decode step: 24 layers of
/// `x = tanh(sum_k(gather_qmm(x, experts, idx)) + x @ dense)`, with `idx`
/// either constant (pre-evaluated) or derived from `x` at runtime through
/// argpartition — the discriminator for whether runtime expert indices are
/// what breaks async overlap.
fn gather_step(
    chain: &GatherChain,
    x: &MlxArray,
    dynamic_indices: bool,
    with_kernel: bool,
) -> MlxArray {
    let e = GQ_EXPERTS as i32;
    let k = GQ_TOPK as i32;
    let mut x = x.clone();
    for _ in 0..GQ_LAYERS {
        let idx = if dynamic_indices {
            let logits = matmul(&x, &chain.router, None);
            let part = argpartition_axis(&logits, e - k, -1, None);
            let top = slice_last_dim(&part, e - k, e, None);
            reshape(&top, &[k], None)
        } else {
            chain.const_idx.clone()
        };
        let gathered = gather_qmm(
            &x,
            &chain.wq,
            &chain.scales,
            Some(&chain.biases),
            &idx,
            true,
            Some(64),
            Some(4),
            false,
            None,
        );
        // [K, 1, H] -> [1, H]
        let expert_sum = sum_axis(&gathered, 0, false, None);
        let ballast = matmul(&x, &chain.dense, None);
        x = tanh(&add(&expert_sum, &ballast, None), None);
        if with_kernel {
            x = apply_probe_kernel(&x);
        }
    }
    x
}

fn run_gather_pipelined(
    chain: &GatherChain,
    spin: u64,
    dynamic_indices: bool,
    with_kernel: bool,
) -> ShapeResult {
    let x_host = vec![0.01f32; GQ_H];
    let x0 = MlxArray::from_raw_data(
        x_host.as_ptr() as *const u8,
        x_host.len() * std::mem::size_of::<f32>(),
        &[1, GQ_H as i32],
        MlxDtype::Float32,
    );
    let x0 = astype(&x0, MlxDtype::Bfloat16, None);
    eval(&[&x0]);
    // Shape stability guard: one step must map [1, H] -> [1, H], otherwise
    // the chain silently grows a dimension per layer and the timings are
    // meaningless (bitten twice while writing this probe).
    let probe_step = gather_step(chain, &x0, dynamic_indices, with_kernel);
    eval(&[&probe_step]);
    assert_eq!(
        probe_step.shape(),
        vec![1, GQ_H as i32],
        "gather chain must preserve [1, H]"
    );

    let mut pending = gather_step(chain, &x0, dynamic_indices, with_kernel);
    async_eval(&[&pending]);
    for _ in 0..WARMUP {
        let next = gather_step(chain, &pending, dynamic_indices, with_kernel);
        async_eval(&[&next]);
        eval(&[&pending]);
        pending = next;
    }
    let mut async_us = 0u128;
    let mut eval_us = 0u128;
    let started = Instant::now();
    for _ in 0..ITERS {
        spin_us(spin);
        let next = gather_step(chain, &pending, dynamic_indices, with_kernel);
        let t = Instant::now();
        async_eval(&[&next]);
        async_us += t.elapsed().as_micros();
        let t = Instant::now();
        eval(&[&pending]);
        eval_us += t.elapsed().as_micros();
        pending = next;
    }
    let total = started.elapsed().as_micros() as f64 / ITERS as f64;
    eval(&[&pending]);
    ShapeResult {
        per_iter_us: total,
        async_us: async_us as f64 / ITERS as f64,
        eval_us: eval_us as f64 / ITERS as f64,
    }
}

fn main() {
    let w1 = constant_matrix(0.01);
    let w2 = constant_matrix(0.01);
    let x0 = constant_matrix(0.01);
    let chain = build_gather_chain();

    println!(
        "async-eval overlap probe: N={N}, {ITERS} iters/config, {WARMUP} warmup, dependent chain = tanh(tanh(X@W1)@W2)"
    );
    println!();
    println!("shape        spin_us  per_iter_us  async_call_us  eval_wait_us  delta_vs_spin0");

    for shape in [
        "serial",
        "pipelined",
        "independent",
        "gqmm_const",
        "gqmm_dyn",
        "gqmm_kernel",
    ] {
        let mut baseline = 0.0f64;
        for (idx, &spin) in SPINS_US.iter().enumerate() {
            let r = match shape {
                "serial" => run_serial(&x0, &w1, &w2, spin),
                "pipelined" => run_pipelined(&x0, &w1, &w2, spin, false),
                "independent" => run_pipelined(&x0, &w1, &w2, spin, true),
                "gqmm_const" => run_gather_pipelined(&chain, spin, false, false),
                "gqmm_dyn" => run_gather_pipelined(&chain, spin, true, false),
                _ => run_gather_pipelined(&chain, spin, false, true),
            };
            if idx == 0 {
                baseline = r.per_iter_us;
            }
            println!(
                "{shape:<12} {spin:>7} {:>12.1} {:>14.1} {:>13.1} {:>+15.1}",
                r.per_iter_us,
                r.async_us,
                r.eval_us,
                r.per_iter_us - baseline
            );
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        println!();
    }
    println!(
        "interpretation: a shape overlaps host with GPU iff delta_vs_spin0 stays ~0 while spin < per_iter baseline."
    );
    mlx_sys::clear_cache();
}
