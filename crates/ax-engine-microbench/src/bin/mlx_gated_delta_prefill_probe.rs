//! Deterministic AX/MLX GDN prefill comparison. Emits raw JSON, never model claims.

use std::process::{Command, ExitCode};
use std::time::Instant;

use ax_engine_mlx::fastpath::qwen_gated_delta_prefill_mlx_enabled;
use ax_engine_mlx::linear_attention_ops::gated_delta_kernel;
use ax_engine_mlx::mlx_gated_delta::try_mlx_gated_delta_prefill;
use mlx_sys::{
    MlxArray, MlxDtype, astype, clear_cache, device_active_bytes, get_peak_memory,
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
}

impl Inputs {
    fn new(batch: i32, seq: i32, hk: i32, hv: i32, dtype: MlxDtype) -> Self {
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
        Self {
            q: array(&[batch, seq, hk, 128], 0.16, dtype),
            k: array(&[batch, seq, hk, 128], 0.16, dtype),
            v: array(&[batch, seq, hv, 128], 0.30, dtype),
            a_log: array(&[hv], 0.4, MlxDtype::Float32),
            a: array(&[batch, seq, hv], 0.8, dtype),
            bias: array(&[hv], 0.5, MlxDtype::Float32),
            b: array(&[batch, seq, hv], 0.7, dtype),
            state: array(&[batch, hv, 128, 128], 0.04, MlxDtype::Float32),
            hashes,
        }
    }

    fn eval(&self) -> Result<(), String> {
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
    let a = astype(a, MlxDtype::Float32, None);
    try_eval(&[&a])?;
    Ok(a.data_f32().to_vec())
}

fn error(a: &[f32], b: &[f32]) -> Result<f32, String> {
    if a.len() != b.len() || a.iter().chain(b).any(|v| !v.is_finite()) {
        return Err("non-finite or mismatched GDN outputs".into());
    }
    Ok(a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max))
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
        "usage: mlx-gated-delta-prefill-probe <output.json> [repetitions]".to_owned()
    })?;
    let repetitions = args
        .next()
        .map(|v| v.parse::<usize>())
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or(5);
    if repetitions == 0 || args.next().is_some() {
        return Err("repetitions must be positive; unexpected arguments".into());
    }
    if qwen_gated_delta_prefill_mlx_enabled() {
        return Err(
            "set AX_MLX_QWEN_GATED_DELTA_PREFILL_MLX=0 so the AX baseline is independent".into(),
        );
    }
    let start_load = host("sysctl", &["-n", "vm.loadavg"])?;
    let mut rows = Vec::new();
    let mut passed = true;
    for (hk, hv) in [(16, 32), (16, 48)] {
        for batch in [1, 2] {
            for seq in [32, 128, 512, 1024, 2048] {
                for dtype in [MlxDtype::Float32, MlxDtype::Float16, MlxDtype::Bfloat16] {
                    eprintln!("GDN B={batch} T={seq} Hk={hk} Hv={hv} {dtype:?}");
                    clear_cache();
                    let inputs = Inputs::new(batch, seq, hk, hv, dtype);
                    inputs.eval()?;
                    let input_bytes = device_active_bytes();
                    let (ax_y, ax_s) = inputs.run(false)?;
                    let (mlx_y, mlx_s) = inputs.run(true)?;
                    let output_error = error(&values(&ax_y)?, &values(&mlx_y)?)?;
                    let state_error = error(&values(&ax_s)?, &values(&mlx_s)?)?;
                    let output_tolerance = if dtype == MlxDtype::Bfloat16 {
                        0.0005
                    } else {
                        0.00005
                    };
                    let ok = output_error <= output_tolerance && state_error <= 0.00005;
                    passed &= ok;
                    drop((ax_y, ax_s, mlx_y, mlx_s));
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
                    rows.push(json!({"batch":batch,"seq":seq,"key_heads":hk,"value_heads":hv,
                        "head_dim":128,"dtype":format!("{dtype:?}"),"input_sha256":inputs.hashes,
                        "input_hash_order":["q","k","v","a_log","a","bias","b","initial_state"],
                        "active_input_mlx_bytes":input_bytes,"output_max_abs":output_error,
                        "state_max_abs":state_error,"output_tolerance":output_tolerance,"state_tolerance":0.00005,
                        "correctness_pass":ok,"trials":trials,"ax_median_ms":median(&ax_medians),
                        "mlx_median_ms":median(&mlx_medians),"kernel_ratio_ax_over_mlx":median(&ax_medians)/median(&mlx_medians)}));
                }
            }
        }
    }
    let source_hash = |source: &str| format!("{:x}", Sha256::digest(source.as_bytes()));
    let report = json!({
        "schema":"ax.mlx.gdn_prefill_probe.v1", "scope":"Synthetic kernel comparison; no model, MTP, or default promotion",
        "host":{"chip":host("sysctl",&["-n","machdep.cpu.brand_string"])? ,
            "memory_bytes":host("sysctl",&["-n","hw.memsize"])?.parse::<u64>().map_err(|e| e.to_string())?,
            "macos":host("sw_vers",&["-productVersion"])? ,"gpu_architecture":gpu_device_architecture()?},
        "mlx_version":runtime_version()?,"repository_head":host("git",&["rev-parse","HEAD"]).ok(),
        "host_load_average":{"start":start_load,"end":host("sysctl",&["-n","vm.loadavg"])?},
        "sources_sha256":{
            "probe":source_hash(include_str!("mlx_gated_delta_prefill_probe.rs")),
            "adapter":source_hash(include_str!("../../../ax-engine-mlx/src/mlx_gated_delta.rs")),
            "ax_baseline":source_hash(include_str!("../../../ax-engine-mlx/src/linear_attention_ops.rs")),
            "flags":source_hash(include_str!("../../../ax-engine-mlx/src/fastpath.rs")),
            "binding":source_hash(include_str!("../../../mlx-sys/src/fast.rs")),
            "shim":source_hash(include_str!("../../../mlx-sys/native/ax_shim.cpp"))},
        "optimization_env":std::env::vars().filter(|(k,_)| k.starts_with("AX_MLX_") || k.starts_with("GATED_DELTA_")).collect::<std::collections::BTreeMap<_,_>>(),
        "seed":31418,"input_hash_scope":"Little-endian float32 source values before activation cast",
        "warmups_per_backend_trial":3,"repetitions":repetitions,"correctness_pass":passed,"rows":rows});
    let encoded = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    std::fs::write(output, format!("{encoded}\n")).map_err(|e| e.to_string())?;
    println!("{encoded}");
    if !passed {
        return Err("GDN numerical comparison failed; raw report retained".into());
    }
    Ok(())
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
