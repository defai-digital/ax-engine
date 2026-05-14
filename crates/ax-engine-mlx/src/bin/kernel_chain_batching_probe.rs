//! Phase B.0 microbench — kernel-chain command-batching probe.
//!
//! Answers the follow-up from
//! `.internal/planning/MLX-PHASE-B-CUSTOM-KERNEL-SPIKE-2026-05-14.md`:
//! when N `MlxMetalKernel::apply` calls are chained without intervening
//! `eval`, does MLX coalesce them into one Metal command buffer, or
//! does each apply commit on its own?
//!
//! Method: register a trivial add-one kernel where per-element compute
//! is negligible, then time two cases over a CHAIN_LEN-deep dependency
//! chain:
//!
//!   Case A — chain CHAIN_LEN applies, single trailing `eval`.
//!   Case B — apply, `eval`, apply, `eval`, …  (CHAIN_LEN pairs).
//!
//! If MLX fuses chained applies, A is much faster than B because
//! submit / sync cost is paid once instead of CHAIN_LEN times. The
//! reported ratio `B / A` gives a coarse "fusion strength" signal that
//! the planning artifact's decision rules consume.
//!
//! Run:
//!   cargo run --release --bin kernel-chain-batching-probe
//!
//! Output: human-readable summary plus the verdict bucket. Paste into
//! `.internal/planning/MLX-PHASE-B-BATCH-MICROBENCH-<date>.md`.

use std::time::Instant;

use mlx_sys::{KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, eval};

const N: usize = 4096;
const CHAIN_LEN: usize = 16;
const ITERS: usize = 200;
const THREADS_PER_GROUP: i32 = 256;

const KERNEL_SOURCE: &str = r#"
    const uint i = thread_position_in_grid.x;
    if (i >= SIZE) {
        return;
    }
    out[i] = x[i] + 1.0f;
"#;

fn build_input() -> Vec<f32> {
    vec![0.0f32; N]
}

fn apply_once(kernel: &MlxMetalKernel, input: &MlxArray) -> MlxArray {
    // Grid = N threads, rounded up to the nearest THREADS_PER_GROUP. The
    // kernel's `i >= SIZE` guard makes the rounding safe.
    let tpg = THREADS_PER_GROUP as usize;
    let grid_x = N.div_ceil(tpg) * tpg;
    let outputs = kernel.apply_with_template(
        &[input],
        &[KernelOutputSpec {
            shape: vec![N as i32],
            dtype: MlxDtype::Float32,
        }],
        &[KernelTemplateArg::Int {
            name: "SIZE",
            value: N as i32,
        }],
        (grid_x as i32, 1, 1),
        (THREADS_PER_GROUP, 1, 1),
        None,
    );
    outputs
        .into_iter()
        .next()
        .expect("kernel must produce one output")
}

fn case_a_chain_then_single_eval(kernel: &MlxMetalKernel, input: &MlxArray) {
    let mut current = apply_once(kernel, input);
    for _ in 1..CHAIN_LEN {
        current = apply_once(kernel, &current);
    }
    eval(&[&current]);
}

fn case_b_eval_between(kernel: &MlxMetalKernel, input: &MlxArray) {
    let mut current = apply_once(kernel, input);
    eval(&[&current]);
    for _ in 1..CHAIN_LEN {
        current = apply_once(kernel, &current);
        eval(&[&current]);
    }
}

fn time_loop<F: FnMut()>(label: &str, iters: usize, mut f: F) -> f64 {
    // Warm up so the first compile / cache pass does not skew the loop.
    f();
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let per_iter_us = elapsed_ms * 1000.0 / iters as f64;
    println!("  {label}: {elapsed_ms:.1} ms total, {per_iter_us:.2} us/iter");
    per_iter_us
}

fn main() {
    println!("Phase B.0 kernel-chain batching probe (N={N}, chain_len={CHAIN_LEN}, iters={ITERS})");
    println!();

    let data = build_input();
    let input = MlxArray::from_raw_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data.as_slice()),
        &[N as i32],
        MlxDtype::Float32,
    );

    let kernel = MlxMetalKernel::new(
        "ax_chain_batching_probe",
        &["x"],
        &["out"],
        KERNEL_SOURCE,
        "",
        true,
    );

    // Correctness: chaining CHAIN_LEN add-ones from a zero input should
    // produce CHAIN_LEN in every element. If MLX reorders or drops a
    // dependency this will fail loudly before we report any timing.
    println!("=== Correctness ===");
    let mut current = apply_once(&kernel, &input);
    for _ in 1..CHAIN_LEN {
        current = apply_once(&kernel, &current);
    }
    eval(&[&current]);
    let values = current.data_f32();
    let expected = CHAIN_LEN as f32;
    let max_diff = values
        .iter()
        .fold(0.0_f32, |acc, &v| acc.max((v - expected).abs()));
    println!("  expected final value: {expected}");
    println!("  max abs diff:         {max_diff}");
    if max_diff > 1.0e-3 {
        println!("  status: FAIL");
        std::process::exit(1);
    }
    println!("  status: ok");
    println!();

    println!("=== Wall-clock ===");
    let case_a_us = time_loop("case A (chain → 1 eval)", ITERS, || {
        case_a_chain_then_single_eval(&kernel, &input);
    });
    let case_b_us = time_loop("case B (eval after each apply)", ITERS, || {
        case_b_eval_between(&kernel, &input);
    });
    println!();

    let ratio = case_b_us / case_a_us;
    println!("=== Verdict ===");
    println!("  ratio (case_B / case_A) = {ratio:.2}×");
    let verdict = if ratio > 4.0 {
        "STRONG_FUSION"
    } else if ratio > 2.0 {
        "MODERATE_FUSION"
    } else if ratio > 1.2 {
        "WEAK_FUSION"
    } else {
        "NO_FUSION"
    };
    println!("  bucket: {verdict}");
    println!();
    match verdict {
        "STRONG_FUSION" => {
            println!(
                "  implication: chained applies coalesce — Phase B can decompose hotspots into"
            );
            println!(
                "  multiple small kernels without paying per-kernel submit cost. Pick whichever"
            );
            println!("  kernel granularity is most readable.");
        }
        "MODERATE_FUSION" => {
            println!(
                "  implication: partial fusion — keep kernel count low per hotspot but fine-grained"
            );
            println!("  decomposition is not catastrophic.");
        }
        "WEAK_FUSION" | "NO_FUSION" => {
            println!(
                "  implication: each apply pays its own submit cost — Phase B must prefer one"
            );
            println!("  fused fat kernel per hotspot (DS4 `begin/end_commands` style). Per-kernel");
            println!("  granularity costs N× submit overhead.");
        }
        _ => unreachable!(),
    }
}
