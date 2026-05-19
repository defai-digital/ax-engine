use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use mlx_sys::{
    MlxArray, MlxDtype, eval, gelu_approx, gelu_approx_mul, multiply, op_count_snapshot,
    op_count_take,
};
use serde_json::{Value, json};

const DEFAULT_ROWS: i32 = 2048;
const DEFAULT_COLS: i32 = 6144;
const DEFAULT_WARMUP: usize = 5;
const DEFAULT_ITERATIONS: usize = 30;
const CORRECTNESS_TOLERANCE: f32 = 1.0e-6;

#[derive(Clone, Debug)]
struct Config {
    rows: i32,
    cols: i32,
    warmup: usize,
    iterations: usize,
    json_out: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            rows: DEFAULT_ROWS,
            cols: DEFAULT_COLS,
            warmup: DEFAULT_WARMUP,
            iterations: DEFAULT_ITERATIONS,
            json_out: None,
        }
    }
}

struct Fixture {
    _gate_data: Vec<f32>,
    _up_data: Vec<f32>,
    gate: MlxArray,
    up: MlxArray,
}

#[derive(Clone, Debug)]
struct Measurement {
    samples_us: Vec<f64>,
    op_counts: Vec<u64>,
}

impl Measurement {
    fn stats_json(&self, name: &str) -> Value {
        let stats = Stats::from_samples(&self.samples_us);
        json!({
            "name": name,
            "unit": "microseconds",
            "samples": self.samples_us.len(),
            "mean": stats.mean,
            "median": stats.median,
            "min": stats.min,
            "max": stats.max,
            "op_count_median": median_u64(&self.op_counts),
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct Stats {
    mean: f64,
    median: f64,
    min: f64,
    max: f64,
}

impl Stats {
    fn from_samples(samples: &[f64]) -> Self {
        assert!(
            !samples.is_empty(),
            "measurement requires at least one sample"
        );
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let sum = sorted.iter().sum::<f64>();
        Self {
            mean: sum / sorted.len() as f64,
            median: median_f64_sorted(&sorted),
            min: sorted[0],
            max: sorted[sorted.len() - 1],
        }
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let config = parse_args(env::args().skip(1))?;
    if config.iterations == 0 {
        return Err("--iterations must be > 0".into());
    }
    if config.rows <= 0 || config.cols <= 0 {
        return Err("--rows and --cols must be > 0".into());
    }

    let fixture = Fixture::new(config.rows, config.cols)?;
    warmup(&fixture, config.warmup);
    let correctness = correctness(&fixture);

    let portable = measure("portable", config.iterations, || {
        portable_activation(&fixture)
    });
    let direct = measure("direct_cpp", config.iterations, || {
        direct_activation(&fixture)
    });

    let portable_stats = Stats::from_samples(&portable.samples_us);
    let direct_stats = Stats::from_samples(&direct.samples_us);
    let speedup = if direct_stats.median > 0.0 {
        portable_stats.median / direct_stats.median
    } else {
        0.0
    };

    let output = json!({
        "schema": "ax.microbench.v1",
        "surface": "direct-mlx-hotpath",
        "command": env::args().collect::<Vec<_>>().join(" "),
        "git": git_json(),
        "host": {
            "os": env::consts::OS,
            "arch": env::consts::ARCH,
        },
        "config": {
            "candidate": "gelu_approx_mul",
            "rows": config.rows,
            "cols": config.cols,
            "dtype": "float32",
            "warmup": config.warmup,
            "iterations": config.iterations,
        },
        "correctness": {
            "passed": correctness.max_abs_error <= CORRECTNESS_TOLERANCE,
            "max_abs_error": correctness.max_abs_error,
            "tolerance": CORRECTNESS_TOLERANCE,
            "shape": correctness.shape,
        },
        "measurements": [
            portable.stats_json("portable_gelu_approx_mul"),
            direct.stats_json("direct_cpp_gelu_approx_mul"),
            {
                "name": "direct_cpp_speedup_ratio",
                "unit": "ratio",
                "samples": 1,
                "mean": speedup,
                "median": speedup,
                "min": speedup,
                "max": speedup,
            }
        ],
    });

    let rendered = serde_json::to_string_pretty(&output).map_err(|err| err.to_string())?;
    if let Some(path) = config.json_out {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)
                .map_err(|err| format!("create {}: {err}", parent.display()))?;
        }
        fs::write(&path, rendered).map_err(|err| format!("write {}: {err}", path.display()))?;
    } else {
        println!("{rendered}");
    }

    Ok(())
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Config, String> {
    let mut config = Config::default();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--rows" => config.rows = parse_value(&arg, args.next())?,
            "--cols" => config.cols = parse_value(&arg, args.next())?,
            "--warmup" => config.warmup = parse_value(&arg, args.next())?,
            "--iterations" => config.iterations = parse_value(&arg, args.next())?,
            "--json-out" => config.json_out = Some(PathBuf::from(parse_string(&arg, args.next())?)),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(config)
}

fn parse_value<T>(flag: &str, value: Option<String>) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let raw = parse_string(flag, value)?;
    raw.parse::<T>()
        .map_err(|err| format!("{flag} value {raw:?} is invalid: {err}"))
}

fn parse_string(flag: &str, value: Option<String>) -> Result<String, String> {
    value.ok_or_else(|| format!("{flag} requires a value"))
}

fn print_help() {
    println!(
        "\
direct-mlx-hotpath-probe

Options:
  --rows <N>         Tensor rows, default {DEFAULT_ROWS}
  --cols <N>         Tensor cols, default {DEFAULT_COLS}
  --warmup <N>       Warmup iterations, default {DEFAULT_WARMUP}
  --iterations <N>   Measured iterations, default {DEFAULT_ITERATIONS}
  --json-out <PATH>  Write ax.microbench.v1 JSON to PATH instead of stdout
"
    );
}

impl Fixture {
    fn new(rows: i32, cols: i32) -> Result<Self, String> {
        let rows_usize = usize::try_from(rows).map_err(|_| "rows must fit usize")?;
        let cols_usize = usize::try_from(cols).map_err(|_| "cols must fit usize")?;
        let len = rows_usize
            .checked_mul(cols_usize)
            .ok_or("rows * cols overflowed usize")?;
        let gate_data = (0..len)
            .map(|idx| ((idx % 257) as f32 - 128.0) / 64.0)
            .collect::<Vec<_>>();
        let up_data = (0..len)
            .map(|idx| ((idx % 193) as f32 - 96.0) / 96.0)
            .collect::<Vec<_>>();
        let gate = MlxArray::from_raw_data(
            gate_data.as_ptr().cast(),
            std::mem::size_of_val(gate_data.as_slice()),
            &[rows, cols],
            MlxDtype::Float32,
        );
        let up = MlxArray::from_raw_data(
            up_data.as_ptr().cast(),
            std::mem::size_of_val(up_data.as_slice()),
            &[rows, cols],
            MlxDtype::Float32,
        );
        eval(&[&gate, &up]);
        Ok(Self {
            _gate_data: gate_data,
            _up_data: up_data,
            gate,
            up,
        })
    }
}

fn warmup(fixture: &Fixture, warmup: usize) {
    for _ in 0..warmup {
        eval(&[&portable_activation(fixture), &direct_activation(fixture)]);
    }
}

fn portable_activation(fixture: &Fixture) -> MlxArray {
    multiply(&gelu_approx(&fixture.gate, None), &fixture.up, None)
}

fn direct_activation(fixture: &Fixture) -> MlxArray {
    gelu_approx_mul(&fixture.gate, &fixture.up, None)
}

fn measure(_name: &str, iterations: usize, mut op: impl FnMut() -> MlxArray) -> Measurement {
    let mut samples_us = Vec::with_capacity(iterations);
    let mut op_counts = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let op_snapshot = op_count_snapshot();
        let started = Instant::now();
        let output = op();
        eval(&[&output]);
        samples_us.push(started.elapsed().as_secs_f64() * 1_000_000.0);
        op_counts.push(op_count_take(op_snapshot));
    }
    Measurement {
        samples_us,
        op_counts,
    }
}

struct Correctness {
    max_abs_error: f32,
    shape: Vec<i32>,
}

fn correctness(fixture: &Fixture) -> Correctness {
    let portable = portable_activation(fixture);
    let direct = direct_activation(fixture);
    eval(&[&portable, &direct]);
    let portable_data = portable.data_f32();
    let direct_data = direct.data_f32();
    let max_abs_error = portable_data
        .iter()
        .zip(direct_data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    Correctness {
        max_abs_error,
        shape: direct.shape(),
    }
}

fn median_f64_sorted(sorted: &[f64]) -> f64 {
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

fn median_u64(samples: &[u64]) -> u64 {
    assert!(
        !samples.is_empty(),
        "measurement requires at least one sample"
    );
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

fn git_json() -> Value {
    json!({
        "commit": command_stdout("git", &["rev-parse", "HEAD"]),
        "dirty": git_dirty(),
    })
}

fn command_stdout(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn git_dirty() -> Option<bool> {
    let unstaged = Command::new("git")
        .args(["diff", "--quiet", "--exit-code"])
        .status()
        .ok()?;
    let staged = Command::new("git")
        .args(["diff", "--cached", "--quiet", "--exit-code"])
        .status()
        .ok()?;
    Some(!unstaged.success() || !staged.success())
}
