//! Deterministic AX/MLX GDN prefill comparison. Emits raw JSON, never model claims.

use std::process::{Command, ExitCode};
use std::time::Instant;

use ax_engine_mlx::fastpath::qwen_gated_delta_prefill_mlx_enabled;
use ax_engine_mlx::linear_attention_ops::gated_delta_kernel;
use ax_engine_mlx::mlx_gated_delta::try_mlx_gated_delta_prefill;
use mlx_sys::{
    MlxArray, MlxDtype, astype, clear_cache, contiguous, device_active_bytes, get_peak_memory,
    gpu_device_architecture, reset_peak_memory, runtime_version, try_eval,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

struct Inputs {
    q: MlxArray,
    k: MlxArray,
    v: MlxArray,
    a_log: MlxArray,
    a: MlxArray,
    bias: MlxArray,
    b: MlxArray,
    state: MlxArray,
    hashes: Vec<String>,
    flash_next_gates: Option<(MlxArray, MlxArray)>,
}

impl Inputs {
    fn new(batch: i32, seq: i32, hk: i32, hv: i32, dtype: MlxDtype, flash_next: bool) -> Self {
        let mut hashes = Vec::new();
        let mut seed = 31418_u64;
        let mut array = |shape: &[i32], scale: f32, dtype| {
            let n = shape.iter().map(|&d| d as usize).product();
            let data: Vec<f32> = (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    ((seed >> 40) as f32 / (1_u64 << 24) as f32 - 0.5) * scale
                })
                .collect();
            let mut digest = Sha256::new();
            for value in &data {
                digest.update(value.to_le_bytes());
            }
            hashes.push(format!("{:x}", digest.finalize()));
            astype(
                &MlxArray::from_raw_data(data.as_ptr().cast(), n * 4, shape, MlxDtype::Float32),
                dtype,
                None,
            )
        };
        let mut inputs = Self {
            q: array(&[batch, seq, hk, 128], 0.16, dtype),
            k: array(&[batch, seq, hk, 128], 0.16, dtype),
            v: array(&[batch, seq, hv, 128], 0.30, dtype),
            a_log: array(&[hv], 0.4, MlxDtype::Float32),
            a: array(&[batch, seq, hv], 0.8, dtype),
            bias: array(&[hv], 0.5, MlxDtype::Float32),
            b: array(&[batch, seq, hv], 0.7, dtype),
            state: array(&[batch, hv, 128, 128], 0.04, MlxDtype::Float32),
            hashes,
            flash_next_gates: None,
        };
        if flash_next {
            // Match the dedicated trunk: normalize in FP32 after projection dtype
            // rounding, then apply query scaling. Gates are prepared before timing.
            let l2 = |x: &MlxArray| {
                let x = astype(x, MlxDtype::Float32, None);
                let squared = mlx_sys::sum_axis(&mlx_sys::multiply(&x, &x, None), -1, true, None);
                mlx_sys::multiply(
                    &x,
                    &mlx_sys::power(
                        &mlx_sys::add(&squared, &MlxArray::from_f32(1e-6), None),
                        &MlxArray::from_f32(-0.5),
                        None,
                    ),
                    None,
                )
            };
            inputs.q = mlx_sys::divide(&l2(&inputs.q), &MlxArray::from_f32(128.0_f32.sqrt()), None);
            inputs.k = l2(&inputs.k);
            inputs.v = astype(&inputs.v, MlxDtype::Float32, None);
            let a = mlx_sys::add(
                &astype(&inputs.a, MlxDtype::Float32, None),
                &inputs.bias,
                None,
            );
            let sp = mlx_sys::add(
                &mlx_sys::maximum(&a, &MlxArray::from_f32(0.0), None),
                &mlx_sys::log1p(
                    &mlx_sys::exp(
                        &mlx_sys::minimum(&a, &mlx_sys::negative(&a, None), None),
                        None,
                    ),
                    None,
                ),
                None,
            );
            let decay = mlx_sys::exp(
                &mlx_sys::negative(
                    &mlx_sys::multiply(&mlx_sys::exp(&inputs.a_log, None), &sp, None),
                    None,
                ),
                None,
            );
            let beta = astype(
                &astype(
                    &mlx_sys::sigmoid(&astype(&inputs.b, MlxDtype::Float32, None), None),
                    dtype,
                    None,
                ),
                MlxDtype::Float32,
                None,
            );
            inputs.flash_next_gates = Some((decay, beta));
        }
        inputs
    }

    fn eval(&self) -> Result<(), String> {
        if let Some((decay, beta)) = &self.flash_next_gates {
            try_eval(&[decay, beta])?;
        }
        try_eval(&[
            &self.q,
            &self.k,
            &self.v,
            &self.a_log,
            &self.a,
            &self.bias,
            &self.b,
            &self.state,
        ])
    }

    fn run(&self, upstream: bool) -> Result<(MlxArray, MlxArray), String> {
        if let Some((decay, beta)) = &self.flash_next_gates {
            return if upstream {
                ax_engine_mlx::flash_next_gdn::try_mlx_prefill(
                    &self.q,
                    &self.k,
                    &self.v,
                    decay,
                    beta,
                    &self.state,
                )
                .ok_or_else(|| "MLX Flash Next prefill rejected the probe shape".into())
            } else {
                ax_engine_mlx::flash_next_gdn::portable_recurrence(
                    &self.q,
                    &self.k,
                    &self.v,
                    decay,
                    beta,
                    &self.state,
                )
            };
        }
        if upstream {
            try_mlx_gated_delta_prefill(
                &self.q,
                &self.k,
                &self.v,
                &self.a_log,
                &self.a,
                &self.bias,
                &self.b,
                &self.state,
            )
            .ok_or_else(|| "MLX prefill path rejected the probe shape".into())
        } else {
            Ok(gated_delta_kernel(
                &self.q,
                &self.k,
                &self.v,
                &self.a_log,
                &self.a,
                &self.bias,
                &self.b,
                &self.state,
            ))
        }
    }
}

fn host(command: &str, args: &[&str]) -> Result<String, String> {
    let out = Command::new(command)
        .args(args)
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!("{command} failed"));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

fn values(a: &MlxArray) -> Result<Vec<f32>, String> {
    // A same-dtype cast can retain a strided view; raw reads require row order.
    let a = contiguous(&astype(a, MlxDtype::Float32, None), None);
    try_eval(&[&a])?;
    Ok(a.data_f32().to_vec())
}

fn error(a: &[f32], b: &[f32]) -> Result<f64, String> {
    if a.len() != b.len() || a.iter().chain(b).any(|v| !v.is_finite()) {
        return Err("non-finite or mismatched GDN outputs".into());
    }
    Ok(a.iter()
        .zip(b)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0_f64, f64::max))
}

struct Comparison {
    max_abs: Option<f64>,
    problem: Option<String>,
}

impl Comparison {
    fn passes(&self, tolerance: f64) -> bool {
        self.problem.is_none() && self.max_abs.is_some_and(|error| error <= tolerance)
    }
}

fn compare_arrays(actual: &MlxArray, expected: &MlxArray) -> Comparison {
    let result = if actual.shape() != expected.shape() {
        Err(format!(
            "mismatched GDN shapes: {:?} versus {:?}",
            actual.shape(),
            expected.shape()
        ))
    } else {
        values(actual).and_then(|a| values(expected).and_then(|b| error(&a, &b)))
    };
    match result {
        Ok(max_abs) => Comparison {
            max_abs: Some(max_abs),
            problem: None,
        },
        Err(problem) => Comparison {
            max_abs: None,
            problem: Some(problem),
        },
    }
}

fn write_report(output: &str, report: &Value) -> Result<(), String> {
    let encoded = serde_json::to_string_pretty(report).map_err(|e| e.to_string())?;
    std::fs::write(output, format!("{encoded}\n")).map_err(|e| e.to_string())?;
    println!("{encoded}");
    if report["correctness_pass"] != true {
        return Err("GDN numerical comparison failed; raw report retained".into());
    }
    Ok(())
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

fn timed(inputs: &Inputs, upstream: bool, reps: usize) -> Result<Value, String> {
    clear_cache();
    reset_peak_memory();
    for _ in 0..3 {
        let (y, s) = inputs.run(upstream)?;
        try_eval(&[&y, &s])?;
    }
    let mut latencies = Vec::new();
    for _ in 0..reps {
        let start = Instant::now();
        let (y, s) = inputs.run(upstream)?;
        try_eval(&[&y, &s])?;
        latencies.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(
        json!({"latencies_ms":latencies,"median_ms":median(&latencies),
              "peak_mlx_bytes":get_peak_memory()}),
    )
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let output = args.next().ok_or_else(|| {
        "usage: mlx-gated-delta-prefill-probe <output.json> [repetitions] [--flash-next] [--correctness-only]".to_owned()
    })?;
    let mut repetitions = None;
    let mut flash_next = false;
    let mut correctness_only = false;
    for argument in args {
        match argument.as_str() {
            "--flash-next" if !flash_next => flash_next = true,
            "--correctness-only" if !correctness_only => correctness_only = true,
            value if !value.starts_with('-') && repetitions.is_none() => {
                repetitions = Some(value.parse::<usize>().map_err(|e| e.to_string())?);
            }
            _ => return Err(format!("unexpected argument: {argument}")),
        }
    }
    let repetitions = repetitions.unwrap_or(5);
    if repetitions == 0 {
        return Err("repetitions must be positive".into());
    }
    if ax_engine_mlx::fastpath::flash_next_gdn_prefill_mlx_enabled() {
        return Err("set AX_MLX_FLASH_NEXT_GDN_PREFILL_MLX=0 for independent A/B dispatch".into());
    }
    if qwen_gated_delta_prefill_mlx_enabled() {
        return Err(
            "set AX_MLX_QWEN_GATED_DELTA_PREFILL_MLX=0 so the AX baseline is independent".into(),
        );
    }
    let start_load = host("sysctl", &["-n", "vm.loadavg"])?;
    let mut rows = Vec::new();
    let mut passed = true;
    let head_pairs: &[(i32, i32)] = if flash_next {
        &[(16, 48)]
    } else {
        &[(16, 32), (16, 48)]
    };
    let sequences: &[i32] = if flash_next {
        &[128, 131, 512, 1024, 2048]
    } else {
        &[32, 128, 512, 1024, 2048]
    };
    for &(hk, hv) in head_pairs {
        for batch in [1, 2] {
            for &seq in sequences {
                for dtype in [MlxDtype::Float32, MlxDtype::Float16, MlxDtype::Bfloat16] {
                    eprintln!("GDN B={batch} T={seq} Hk={hk} Hv={hv} {dtype:?}");
                    clear_cache();
                    let inputs = Inputs::new(batch, seq, hk, hv, dtype, flash_next);
                    inputs.eval()?;
                    let input_bytes = device_active_bytes();
                    let (ax_y, ax_s) = inputs.run(false)?;
                    let (mlx_y, mlx_s) = inputs.run(true)?;
                    let output_comparison = compare_arrays(&ax_y, &mlx_y);
                    let state_comparison = compare_arrays(&ax_s, &mlx_s);
                    let output_tolerance = if !flash_next && dtype == MlxDtype::Bfloat16 {
                        0.0005
                    } else {
                        0.00005
                    };
                    let ok = output_comparison.passes(output_tolerance)
                        && state_comparison.passes(0.00005);
                    passed &= ok;
                    let mut row = json!({"batch":batch,"seq":seq,"key_heads":hk,"value_heads":hv,
                        "head_dim":128,"dtype":format!("{dtype:?}"),"input_sha256":inputs.hashes,
                        "input_hash_order":["q","k","v","a_log","a","bias","b","initial_state"],
                        "active_input_mlx_bytes":input_bytes,"output_max_abs":output_comparison.max_abs,
                        "state_max_abs":state_comparison.max_abs,"output_tolerance":output_tolerance,"state_tolerance":0.00005,
                        "comparison_errors":{"output":output_comparison.problem,"state":state_comparison.problem},
                        "correctness_pass":ok,"trials":[]});
                    drop((ax_y, ax_s, mlx_y, mlx_s));
                    // Preserve failed comparisons without evaluating them again for timing.
                    if !ok || correctness_only {
                        rows.push(row);
                        continue;
                    }
                    let mut trials = Vec::new();
                    let mut ax_medians = Vec::new();
                    let mut mlx_medians = Vec::new();
                    for trial in 0..3 {
                        // Alternate which backend runs first to reduce ordering bias.
                        let first_mlx = trial % 2 == 1;
                        let first = timed(&inputs, first_mlx, repetitions)?;
                        let second = timed(&inputs, !first_mlx, repetitions)?;
                        let (ax, mlx) = if first_mlx {
                            (second, first)
                        } else {
                            (first, second)
                        };
                        ax_medians.push(ax["median_ms"].as_f64().ok_or("missing AX latency")?);
                        mlx_medians.push(mlx["median_ms"].as_f64().ok_or("missing MLX latency")?);
                        trials.push(
                            json!({"first":if first_mlx {"mlx"} else {"ax"},"ax":ax,"mlx":mlx}),
                        );
                    }
                    row["trials"] = json!(trials);
                    row["ax_median_ms"] = json!(median(&ax_medians));
                    row["mlx_median_ms"] = json!(median(&mlx_medians));
                    row["kernel_ratio_ax_over_mlx"] =
                        json!(median(&ax_medians) / median(&mlx_medians));
                    rows.push(row);
                }
            }
        }
    }
    let source_hash = |source: &str| format!("{:x}", Sha256::digest(source.as_bytes()));
    let report = json!({
        "schema":"ax.mlx.gdn_prefill_probe.v1",
        "recurrence":if flash_next {"dedicated_flash_next_fp32"} else {"shared_qwen"},
        "measurement":if correctness_only {"correctness_only"} else {"synthetic_kernel_timing"},
        "scope":"Synthetic kernel comparison; no model, MTP, or default promotion",
        "host":{"chip":host("sysctl",&["-n","machdep.cpu.brand_string"])? ,
            "memory_bytes":host("sysctl",&["-n","hw.memsize"])?.parse::<u64>().map_err(|e| e.to_string())?,
            "macos":host("sw_vers",&["-productVersion"])? ,"gpu_architecture":gpu_device_architecture()?},
        "mlx_version":runtime_version()?,"repository_head":host("git",&["rev-parse","HEAD"]).ok(),
        "host_load_average":{"start":start_load,"end":host("sysctl",&["-n","vm.loadavg"])?},
        "sources_sha256":{
            "probe":source_hash(include_str!("mlx_gated_delta_prefill_probe.rs")),
            "adapter":source_hash(include_str!("../../../ax-engine-mlx/src/mlx_gated_delta.rs")),
            "flash_next_recurrence":source_hash(include_str!("../../../ax-engine-mlx/src/flash_next_gdn.rs")),
            "flash_next_trunk":source_hash(include_str!("../../../ax-engine-mlx/src/model/shared/qwen4_exp_gdn.rs")),
            "ax_baseline":source_hash(include_str!("../../../ax-engine-mlx/src/linear_attention_ops.rs")),
            "flags":source_hash(include_str!("../../../ax-engine-mlx/src/fastpath.rs")),
            "binding":source_hash(include_str!("../../../mlx-sys/src/fast.rs")),
            "shim":source_hash(include_str!("../../../mlx-sys/native/ax_shim.cpp"))},
        "optimization_env":std::env::vars().filter(|(k,_)| k.starts_with("AX_MLX_") || k.starts_with("GATED_DELTA_")).collect::<std::collections::BTreeMap<_,_>>(),
        "seed":31418,"input_hash_scope":"Little-endian float32 source values before activation cast",
        "warmups_per_backend_trial":if correctness_only {0} else {3},"repetitions":if correctness_only {0} else {repetitions},"correctness_pass":passed,"rows":rows});
    write_report(&output, &report)
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn values_reads_strided_float32_in_logical_order() {
        let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(&data),
            &[2, 3],
            MlxDtype::Float32,
        );
        let view = mlx_sys::transpose(&array, &[1, 0], None);
        assert_eq!(
            values(&view).expect("read strided tensor"),
            [1., 4., 2., 5., 3., 6.]
        );
    }

    #[test]
    fn finite_error_does_not_overflow() {
        let observed = error(&[f32::MAX], &[-f32::MAX]).expect("finite operands");
        assert!(
            observed.is_finite(),
            "finite operands must retain a finite error"
        );
        assert_eq!(observed, 2.0 * f32::MAX as f64);
    }

    #[test]
    fn shape_mismatch_with_equal_size_fails_comparison() {
        let a = mlx_sys::zeros(&[2, 3], MlxDtype::Float32, None);
        let b = mlx_sys::zeros(&[3, 2], MlxDtype::Float32, None);
        let comparison = compare_arrays(&a, &b);
        assert!(!comparison.passes(0.00005));
        assert!(comparison.problem.expect("shape error").contains("shapes"));
    }

    #[test]
    fn nonfinite_comparisons_retain_failed_json() {
        for (index, invalid) in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
            .into_iter()
            .enumerate()
        {
            let a = MlxArray::from_raw_data(
                std::ptr::from_ref(&invalid).cast(),
                4,
                &[1],
                MlxDtype::Float32,
            );
            let b = mlx_sys::zeros(&[1], MlxDtype::Float32, None);
            let comparison = compare_arrays(&a, &b);
            assert!(!comparison.passes(0.00005));
            let report = json!({"correctness_pass":false,"rows":[{
                "output_max_abs":comparison.max_abs,"comparison_errors":{"output":comparison.problem},
                "correctness_pass":false,"trials":[]}]});
            let path = std::env::temp_dir().join(format!(
                "ax-gdn-nonfinite-{}-{index}.json",
                std::process::id()
            ));
            assert!(write_report(path.to_str().expect("temporary path"), &report).is_err());
            let retained: Value =
                serde_json::from_slice(&std::fs::read(&path).expect("retained JSON"))
                    .expect("valid JSON");
            std::fs::remove_file(path).expect("remove fixture");
            assert_eq!(retained, report);
            assert!(
                retained["rows"][0]["comparison_errors"]["output"]
                    .as_str()
                    .expect("numerical error")
                    .contains("non-finite")
            );
        }
    }
}
