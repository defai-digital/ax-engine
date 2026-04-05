//! Standalone Criterion microbenchmarks.
//!
//! These stay independent from the `ax-engine-bench` library on purpose: they
//! target tiny synthetic CPU kernels and should remain runnable even when the
//! full benchmark harness evolves around model-loading, JSON reporting, or
//! Metal-specific profiling concerns.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn sum_bytes(data: &[u8]) -> u64 {
    let mut sum = 0u64;
    for &value in data {
        sum = sum.wrapping_add(value as u64);
    }
    black_box(sum)
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (&lhs, &rhs) in a.iter().zip(b.iter()) {
        acc += lhs * rhs;
    }
    black_box(acc)
}

fn bench_cpu_cache_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_cache_sweep");
    for kib in [32usize, 512, 8 * 1024] {
        let size = kib * 1024;
        let data: Vec<u8> = (0..size).map(|i| (i & 0xff) as u8).collect();
        group.bench_function(format!("sum_{kib}KiB"), |b| b.iter(|| sum_bytes(&data)));
    }
    group.finish();
}

fn bench_cpu_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_dot_product");
    for len in [1_024usize, 32_768, 262_144] {
        let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.25).collect();
        let b: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
        group.bench_function(format!("dot_{len}"), |bch| bch.iter(|| dot_f32(&a, &b)));
    }
    group.finish();
}

criterion_group!(benches, bench_cpu_cache_sweep, bench_cpu_dot_product);
criterion_main!(benches);
