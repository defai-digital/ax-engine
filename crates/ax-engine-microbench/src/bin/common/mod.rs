//! Helpers shared by the microbench probe binaries.

#![allow(dead_code)]

use std::time::Instant;

pub(crate) fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

pub(crate) fn time_loop<F: FnMut()>(label: &str, iters: usize, f: F) -> f64 {
    time_loop_indented(label, iters, "  ", f)
}

/// Like [`time_loop`], but with an explicit indent prefix for probes whose
/// section output is indented differently (the dequant-dtype probe prints its
/// section at four spaces).
pub(crate) fn time_loop_indented<F: FnMut()>(
    label: &str,
    iters: usize,
    indent: &str,
    mut f: F,
) -> f64 {
    // Warm up once so first-call compile / cache costs stay out of the loop.
    f();
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let per_iter_us = elapsed_ms * 1000.0 / iters as f64;
    println!("{indent}{label}: {elapsed_ms:.1} ms total, {per_iter_us:.2} us/iter");
    per_iter_us
}
