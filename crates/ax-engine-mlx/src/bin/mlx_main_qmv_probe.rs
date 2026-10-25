//! Quantized-matmul probe for the pinned MLX wheel and an explicit MLX-main build.
//!
//! Times `quantized_matmul` at decode row counts 1, 2, 6, 8 and 16. Shapes are
//! the published Qwen 3.8 text configs, not a tuned microbench:
//!
//! - Flash Next (`qwen4_exp`): hidden 2560, 24 query heads, head dim 256
//!   (q_proj 2560 → 6144), expert intermediate 640.
//! - 27B dense: hidden 5120, q_proj 5120 → 6144, FFN 5120 → 17408.
//!
//! Each multi-row result is compared, after an exact bf16→f32 widening, with
//! the raw f32 bits of the same rows evaluated one at a time. This binary does
//! not change kernel dispatch; it calls MLX's quantized matmul and reports
//! whatever kernel that build selects.
//!
//! ```text
//! cargo run --release -p ax-engine-mlx --bin mlx_main_qmv_probe
//! ```
//!
//! `AX_QMV_PROBE_WARMUP` (default 2) and `AX_QMV_PROBE_ITERS` (default 5) select
//! the timed repetitions. The reported time is the upper median in milliseconds.

#[path = "common/mod.rs"]
mod common;

use std::process::ExitCode;
use std::time::Instant;

use mlx_sys::{
    MlxArray, MlxDtype, MlxQuantizationMode, astype, clear_cache, contiguous,
    gpu_device_architecture, quantize, quantized_matmul_with_mode, runtime_version, slice,
    try_eval,
};

use common::{env_usize, median_upper};

struct Projection {
    model: &'static str,
    name: &'static str,
    k: i32,
    n: i32,
}

struct Quant {
    label: &'static str,
    mode: MlxQuantizationMode,
    group: i32,
    bits: i32,
}

struct PackedWeight {
    weight: MlxArray,
    scales: MlxArray,
    biases: Option<MlxArray>,
}

const ROWS: [i32; 5] = [1, 2, 6, 8, 16];

const PROJECTIONS: [Projection; 5] = [
    Projection {
        model: "qwen3.8-flash-next",
        name: "q_proj",
        k: 2560,
        n: 6144,
    },
    Projection {
        model: "qwen3.8-flash-next",
        name: "expert",
        k: 2560,
        n: 640,
    },
    Projection {
        model: "qwen3.8-27b",
        name: "q_proj",
        k: 5120,
        n: 6144,
    },
    Projection {
        model: "qwen3.8-27b",
        name: "ffn_up",
        k: 5120,
        n: 17408,
    },
    Projection {
        model: "qwen3.8-27b",
        name: "ffn_down",
        k: 17408,
        n: 5120,
    },
];

const QUANTS: [Quant; 3] = [
    Quant {
        label: "mxfp4-g32",
        mode: MlxQuantizationMode::Mxfp4,
        group: 32,
        bits: 4,
    },
    Quant {
        label: "affine-6-g64",
        mode: MlxQuantizationMode::Affine,
        group: 64,
        bits: 6,
    },
    Quant {
        label: "affine-4-g64",
        mode: MlxQuantizationMode::Affine,
        group: 64,
        bits: 4,
    },
];

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("mlx_main_qmv_probe: {message}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<(), String> {
    let warmup = env_usize("AX_QMV_PROBE_WARMUP", 2)?;
    let iters = env_usize("AX_QMV_PROBE_ITERS", 5)?;
    if iters == 0 {
        return Err("AX_QMV_PROBE_ITERS must be at least 1".to_string());
    }
    let version = runtime_version()?;
    let arch = gpu_device_architecture().unwrap_or_else(|error| format!("unavailable ({error})"));
    println!("mlx_version={version}");
    println!("gpu_arch={arch}");
    println!("warmup={warmup} iters={iters} rows=1,2,6,8,16");
    println!(
        "{:<22} {:<8} {:<12} {:>2} {:>6} {:>6} {:>10} {:>8} {:<5}",
        "model", "proj", "quant", "M", "K", "N", "median_ms", "vs_m1", "bits"
    );

    let mut mismatched = false;
    for projection in &PROJECTIONS {
        let activation = activation(projection)?;
        for quant in &QUANTS {
            let packed = quantize_projection(projection, quant)?;
            let reference = stacked_row_bits(&activation, projection, quant, &packed)?;
            let mut baseline_ms = None;
            for rows in ROWS {
                let input = slice(&activation, &[0, 0], &[rows, projection.k], &[1, 1], None);
                let median_ms = time_matmul(&input, quant, &packed, warmup, iters)?;
                let vs_m1 = match baseline_ms {
                    None => {
                        baseline_ms = Some(median_ms);
                        1.0
                    }
                    Some(first) if first > 0.0 => median_ms / first,
                    Some(_) => 0.0,
                };
                let actual = f32_bits(&matmul(&input, quant, &packed)?, rows, projection.n)?;
                let prefix = rows as usize * projection.n as usize;
                let exact = reference.get(..prefix) == Some(actual.as_slice());
                if !exact {
                    mismatched = true;
                }
                println!(
                    "{:<22} {:<8} {:<12} {:>2} {:>6} {:>6} {:>10.3} {:>8.3} {}",
                    projection.model,
                    projection.name,
                    quant.label,
                    rows,
                    projection.k,
                    projection.n,
                    median_ms,
                    vs_m1,
                    if exact { "exact" } else { "DIFF" }
                );
            }
            clear_cache();
        }
    }
    if mismatched {
        return Err(
            "one or more multi-row results were not bit-exact against the stacked 1-row results"
                .to_string(),
        );
    }
    Ok(())
}

fn activation(projection: &Projection) -> Result<MlxArray, String> {
    let values = fill(16, projection.k, (projection.k as u32) ^ 0x00A5)?;
    let host = host_f32(&values, &[16, projection.k])?;
    let device = astype(&host, MlxDtype::Bfloat16, None);
    try_eval(&[&device])?;
    Ok(device)
}

fn quantize_projection(projection: &Projection, quant: &Quant) -> Result<PackedWeight, String> {
    if projection.k % quant.group != 0 {
        return Err(format!(
            "{} K {} is not divisible by group {}",
            projection.name, projection.k, quant.group
        ));
    }
    eprintln!(
        "quantizing {} {} {} (N={}, K={})",
        projection.model, projection.name, quant.label, projection.n, projection.k
    );
    let values = fill(projection.n, projection.k, (projection.n as u32) ^ 0x5A17)?;
    let host = host_f32(&values, &[projection.n, projection.k])?;
    let mut parts = quantize(
        &host,
        Some(quant.group),
        Some(quant.bits),
        quant.mode,
        None,
        None,
    );
    if parts.len() < 2 {
        return Err(format!(
            "{} produced {} tensors, expected weight and scales",
            quant.label,
            parts.len()
        ));
    }
    let biases = if parts.len() >= 3 {
        Some(parts.remove(2))
    } else {
        None
    };
    if quant.mode == MlxQuantizationMode::Affine && biases.is_none() {
        return Err(format!(
            "{} affine quantize did not return biases",
            quant.label
        ));
    }
    let scales = parts.remove(1);
    let weight = parts.remove(0);
    Ok(PackedWeight {
        weight,
        scales,
        biases,
    })
}

fn stacked_row_bits(
    activation: &MlxArray,
    projection: &Projection,
    quant: &Quant,
    packed: &PackedWeight,
) -> Result<Vec<u32>, String> {
    let mut bits = Vec::new();
    for row in 0..16 {
        let input = slice(
            activation,
            &[row, 0],
            &[row + 1, projection.k],
            &[1, 1],
            None,
        );
        let output = matmul(&input, quant, packed)?;
        bits.extend(f32_bits(&output, 1, projection.n)?);
    }
    Ok(bits)
}

fn matmul(input: &MlxArray, quant: &Quant, packed: &PackedWeight) -> Result<MlxArray, String> {
    Ok(quantized_matmul_with_mode(
        input,
        &packed.weight,
        &packed.scales,
        packed.biases.as_ref(),
        true,
        Some(quant.group),
        Some(quant.bits),
        quant.mode,
        None,
    ))
}

fn time_matmul(
    input: &MlxArray,
    quant: &Quant,
    packed: &PackedWeight,
    warmup: usize,
    iters: usize,
) -> Result<f64, String> {
    for _ in 0..warmup {
        let output = matmul(input, quant, packed)?;
        try_eval(&[&output])?;
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let started = Instant::now();
        let output = matmul(input, quant, packed)?;
        try_eval(&[&output])?;
        samples.push(started.elapsed().as_secs_f64() * 1.0e3);
    }
    Ok(median_upper(samples))
}

fn f32_bits(array: &MlxArray, rows: i32, cols: i32) -> Result<Vec<u32>, String> {
    let widened = contiguous(&astype(array, MlxDtype::Float32, None), None);
    try_eval(&[&widened])?;
    if widened.dtype() != MlxDtype::Float32 {
        return Err("quantized matmul result did not convert to f32".to_string());
    }
    let values = widened.data_f32();
    let expected = usize::try_from(rows)
        .ok()
        .and_then(|rows| {
            usize::try_from(cols)
                .ok()
                .and_then(|cols| rows.checked_mul(cols))
        })
        .ok_or_else(|| "result shape overflow".to_string())?;
    if values.len() != expected {
        return Err(format!(
            "result length {} does not match {rows} x {cols}",
            values.len()
        ));
    }
    Ok(values.iter().copied().map(f32::to_bits).collect())
}

fn host_f32(values: &[f32], shape: &[i32]) -> Result<MlxArray, String> {
    let bytes = values
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "host array byte length overflow".to_string())?;
    Ok(MlxArray::from_raw_data(
        values.as_ptr().cast(),
        bytes,
        shape,
        MlxDtype::Float32,
    ))
}

fn fill(rows: i32, cols: i32, seed: u32) -> Result<Vec<f32>, String> {
    let count = usize::try_from(rows)
        .ok()
        .and_then(|rows| {
            usize::try_from(cols)
                .ok()
                .and_then(|cols| rows.checked_mul(cols))
        })
        .ok_or_else(|| format!("cannot allocate a {rows} x {cols} matrix"))?;
    let mut values = Vec::with_capacity(count);
    let mut state = seed | 1;
    for _ in 0..count {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let unit = ((state >> 8) as f32) / (1u32 << 24) as f32;
        values.push(unit * 2.0 - 1.0);
    }
    Ok(values)
}
