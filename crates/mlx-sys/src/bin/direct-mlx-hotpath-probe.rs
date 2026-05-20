use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use mlx_sys::ops::{gelu_approx_mul_matmul, gelu_approx_quantized_ffn, gemma4_post_attn_ffn_block};
use mlx_sys::{
    MlxArray, MlxDtype, MlxQuantizationMode, add, as_strided, concatenate, eval, gelu_approx,
    gelu_approx_mul, matmul, multiply, op_count_snapshot, op_count_take,
    qk_norm_rope_bhsd_from_proj, quantize, quantized_matmul, qwen_linear_attention_inputs_packed,
    reshape, rms_norm, rope, slice_last_dim,
};
use serde_json::{Value, json};

const DEFAULT_ROWS: i32 = 2048;
const DEFAULT_COLS: i32 = 6144;
const DEFAULT_DOWN_COLS: i32 = 1536;
const DEFAULT_HEAD_DIM: i32 = 256;
const DEFAULT_GROUP_SIZE: i32 = 64;
const DEFAULT_BITS: i32 = 4;
const DEFAULT_LINEAR_NUM_KEY_HEADS: i32 = 16;
const DEFAULT_LINEAR_NUM_VALUE_HEADS: i32 = 64;
const DEFAULT_LINEAR_KEY_HEAD_DIM: i32 = 192;
const DEFAULT_LINEAR_VALUE_HEAD_DIM: i32 = 128;
const DEFAULT_WARMUP: usize = 5;
const DEFAULT_ITERATIONS: usize = 30;
const CORRECTNESS_TOLERANCE: f32 = 1.0e-6;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Candidate {
    Activation,
    ActivationDown,
    QuantizedFfn,
    QkNormRope,
    Gemma4PostAttnFfnBlock,
    QwenLinearAttentionInputsPacked,
}

impl Candidate {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "gelu_approx_mul" => Ok(Self::Activation),
            "gelu_approx_mul_matmul" => Ok(Self::ActivationDown),
            "gelu_approx_quantized_ffn" => Ok(Self::QuantizedFfn),
            "qk_norm_rope" => Ok(Self::QkNormRope),
            "gemma4_post_attn_ffn_block" => Ok(Self::Gemma4PostAttnFfnBlock),
            "qwen_linear_attention_inputs_packed" => Ok(Self::QwenLinearAttentionInputsPacked),
            other => Err(format!(
                "unknown --candidate {other:?}; expected gelu_approx_mul, gelu_approx_mul_matmul, gelu_approx_quantized_ffn, qk_norm_rope, gemma4_post_attn_ffn_block, or qwen_linear_attention_inputs_packed",
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Activation => "gelu_approx_mul",
            Self::ActivationDown => "gelu_approx_mul_matmul",
            Self::QuantizedFfn => "gelu_approx_quantized_ffn",
            Self::QkNormRope => "qk_norm_rope",
            Self::Gemma4PostAttnFfnBlock => "gemma4_post_attn_ffn_block",
            Self::QwenLinearAttentionInputsPacked => "qwen_linear_attention_inputs_packed",
        }
    }

    fn portable_measurement(self) -> &'static str {
        match self {
            Self::Activation => "portable_gelu_approx_mul",
            Self::ActivationDown => "portable_gelu_approx_mul_matmul",
            Self::QuantizedFfn => "portable_gelu_approx_quantized_ffn",
            Self::QkNormRope => "portable_qk_norm_rope",
            Self::Gemma4PostAttnFfnBlock => "portable_gemma4_post_attn_ffn_block",
            Self::QwenLinearAttentionInputsPacked => "portable_qwen_linear_attention_inputs_packed",
        }
    }

    fn direct_measurement(self) -> &'static str {
        match self {
            Self::Activation => "direct_cpp_gelu_approx_mul",
            Self::ActivationDown => "direct_cpp_gelu_approx_mul_matmul",
            Self::QuantizedFfn => "direct_cpp_gelu_approx_quantized_ffn",
            Self::QkNormRope => "direct_cpp_qk_norm_rope",
            Self::Gemma4PostAttnFfnBlock => "direct_cpp_gemma4_post_attn_ffn_block",
            Self::QwenLinearAttentionInputsPacked => {
                "direct_cpp_qwen_linear_attention_inputs_packed"
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Config {
    candidate: Candidate,
    rows: i32,
    cols: i32,
    down_cols: i32,
    head_dim: i32,
    group_size: i32,
    bits: i32,
    linear_num_key_heads: i32,
    linear_num_value_heads: i32,
    linear_key_head_dim: i32,
    linear_value_head_dim: i32,
    warmup: usize,
    iterations: usize,
    json_out: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            candidate: Candidate::Activation,
            rows: DEFAULT_ROWS,
            cols: DEFAULT_COLS,
            down_cols: DEFAULT_DOWN_COLS,
            head_dim: DEFAULT_HEAD_DIM,
            group_size: DEFAULT_GROUP_SIZE,
            bits: DEFAULT_BITS,
            linear_num_key_heads: DEFAULT_LINEAR_NUM_KEY_HEADS,
            linear_num_value_heads: DEFAULT_LINEAR_NUM_VALUE_HEADS,
            linear_key_head_dim: DEFAULT_LINEAR_KEY_HEAD_DIM,
            linear_value_head_dim: DEFAULT_LINEAR_VALUE_HEAD_DIM,
            warmup: DEFAULT_WARMUP,
            iterations: DEFAULT_ITERATIONS,
            json_out: None,
        }
    }
}

struct Fixture {
    _gate_data: Option<Vec<f32>>,
    _up_data: Option<Vec<f32>>,
    _input_data: Option<Vec<f32>>,
    _down_weight_data: Option<Vec<f32>>,
    _gate_up_weight_data: Option<Vec<f32>>,
    _proj_data: Option<Vec<f32>>,
    _norm_data: Option<Vec<f32>>,
    _post_norm_data: Option<Vec<f32>>,
    _layer_scalar_data: Option<Vec<f32>>,
    _linear_qkvz_weight_data: Option<Vec<f32>>,
    _linear_ba_weight_data: Option<Vec<f32>>,
    gate: Option<MlxArray>,
    up: Option<MlxArray>,
    down_weight: Option<MlxArray>,
    input: Option<MlxArray>,
    attn: Option<MlxArray>,
    proj: Option<MlxArray>,
    norm: Option<MlxArray>,
    post_norm: Option<MlxArray>,
    layer_scalar: Option<MlxArray>,
    gate_up_quantized: Option<QuantizedFixtureWeight>,
    down_quantized: Option<QuantizedFixtureWeight>,
    linear_attention: Option<LinearAttentionFixture>,
    group_size: i32,
    bits: i32,
    n_heads: i32,
    head_dim: i32,
}

struct LinearAttentionFixture {
    qkvz: QuantizedFixtureWeight,
    ba: QuantizedFixtureWeight,
    num_key_heads: i32,
    num_value_heads: i32,
    key_head_dim: i32,
    value_head_dim: i32,
}

struct QuantizedFixtureWeight {
    weight: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
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
    if config.rows <= 0 || config.cols <= 0 || config.down_cols <= 0 {
        return Err("--rows, --cols, and --down-cols must be > 0".into());
    }
    if config.group_size <= 0 || config.bits <= 0 {
        return Err("--group-size and --bits must be > 0".into());
    }
    if config.head_dim <= 0 {
        return Err("--head-dim must be > 0".into());
    }
    if matches!(
        config.candidate,
        Candidate::QuantizedFfn
            | Candidate::Gemma4PostAttnFfnBlock
            | Candidate::QwenLinearAttentionInputsPacked
    ) && !matches!(config.group_size, 32 | 64 | 128)
    {
        return Err("--group-size must be 32, 64, or 128 for quantized candidates".into());
    }
    if matches!(
        config.candidate,
        Candidate::QuantizedFfn | Candidate::Gemma4PostAttnFfnBlock
    ) && config.bits != 4
    {
        return Err("--bits must be 4 for quantized FFN candidates".into());
    }
    if config.candidate == Candidate::QkNormRope && config.cols % config.head_dim != 0 {
        return Err("--cols must be divisible by --head-dim for qk_norm_rope".into());
    }
    if config.candidate == Candidate::QwenLinearAttentionInputsPacked {
        if !(2..=8).contains(&config.bits) {
            return Err(
                "--bits must be between 2 and 8 for qwen_linear_attention_inputs_packed".into(),
            );
        }
        if config.linear_num_key_heads <= 0
            || config.linear_num_value_heads <= 0
            || config.linear_key_head_dim <= 0
            || config.linear_value_head_dim <= 0
        {
            return Err("--linear-* dimensions must be positive".into());
        }
        if config.linear_num_value_heads % config.linear_num_key_heads != 0 {
            return Err(
                "--linear-num-value-heads must be divisible by --linear-num-key-heads".into(),
            );
        }
    }

    let fixture = Fixture::new(
        config.candidate,
        config.rows,
        config.cols,
        config.down_cols,
        config.head_dim,
        config.group_size,
        config.bits,
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
    )?;
    warmup(&fixture, config.candidate, config.warmup);
    let correctness = correctness(&fixture, config.candidate);

    let portable = measure("portable", config.iterations, || {
        portable_candidate(&fixture, config.candidate)
    });
    let direct = measure("direct_cpp", config.iterations, || {
        direct_candidate(&fixture, config.candidate)
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
            "candidate": config.candidate.as_str(),
            "rows": config.rows,
            "cols": config.cols,
            "down_cols": config.down_cols,
            "head_dim": config.head_dim,
            "n_heads": if config.head_dim > 0 { config.cols / config.head_dim } else { 0 },
            "dtype": "float32",
            "group_size": config.group_size,
            "bits": config.bits,
            "linear_num_key_heads": config.linear_num_key_heads,
            "linear_num_value_heads": config.linear_num_value_heads,
            "linear_key_head_dim": config.linear_key_head_dim,
            "linear_value_head_dim": config.linear_value_head_dim,
            "warmup": config.warmup,
            "iterations": config.iterations,
        },
        "correctness": {
            "passed": correctness.max_abs_error <= CORRECTNESS_TOLERANCE,
            "max_abs_error": correctness.max_abs_error,
            "tolerance": CORRECTNESS_TOLERANCE,
            "shape": correctness.shape,
            "component_shapes": correctness.component_shapes,
        },
        "measurements": [
            portable.stats_json(config.candidate.portable_measurement()),
            direct.stats_json(config.candidate.direct_measurement()),
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
            "--candidate" => {
                config.candidate = Candidate::parse(&parse_string(&arg, args.next())?)?;
            }
            "--rows" => config.rows = parse_value(&arg, args.next())?,
            "--cols" => config.cols = parse_value(&arg, args.next())?,
            "--down-cols" => config.down_cols = parse_value(&arg, args.next())?,
            "--head-dim" => config.head_dim = parse_value(&arg, args.next())?,
            "--group-size" => config.group_size = parse_value(&arg, args.next())?,
            "--bits" => config.bits = parse_value(&arg, args.next())?,
            "--linear-num-key-heads" => {
                config.linear_num_key_heads = parse_value(&arg, args.next())?
            }
            "--linear-num-value-heads" => {
                config.linear_num_value_heads = parse_value(&arg, args.next())?
            }
            "--linear-key-head-dim" => config.linear_key_head_dim = parse_value(&arg, args.next())?,
            "--linear-value-head-dim" => {
                config.linear_value_head_dim = parse_value(&arg, args.next())?
            }
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
  --candidate <NAME> Select gelu_approx_mul, gelu_approx_mul_matmul, gelu_approx_quantized_ffn, qk_norm_rope, gemma4_post_attn_ffn_block, or qwen_linear_attention_inputs_packed
  --rows <N>         Tensor rows, default {DEFAULT_ROWS}
  --cols <N>         FFN/intermediate cols, default {DEFAULT_COLS}
  --down-cols <N>    Output/input cols for down-projection and quantized FFN, default {DEFAULT_DOWN_COLS}
  --head-dim <N>     Per-head dimension for qk_norm_rope, default {DEFAULT_HEAD_DIM}
  --group-size <N>   Affine quantization group size for quantized FFN, default {DEFAULT_GROUP_SIZE}
  --bits <N>         Affine quantization bits for quantized FFN, default {DEFAULT_BITS}
  --linear-num-key-heads <N>   Qwen linear-attention key heads, default {DEFAULT_LINEAR_NUM_KEY_HEADS}
  --linear-num-value-heads <N> Qwen linear-attention value heads, default {DEFAULT_LINEAR_NUM_VALUE_HEADS}
  --linear-key-head-dim <N>    Qwen linear-attention key head dim, default {DEFAULT_LINEAR_KEY_HEAD_DIM}
  --linear-value-head-dim <N>  Qwen linear-attention value head dim, default {DEFAULT_LINEAR_VALUE_HEAD_DIM}
  --warmup <N>       Warmup iterations, default {DEFAULT_WARMUP}
  --iterations <N>   Measured iterations, default {DEFAULT_ITERATIONS}
  --json-out <PATH>  Write ax.microbench.v1 JSON to PATH instead of stdout
"
    );
}

impl Fixture {
    #[allow(clippy::too_many_arguments)]
    fn new(
        candidate: Candidate,
        rows: i32,
        cols: i32,
        down_cols: i32,
        head_dim: i32,
        group_size: i32,
        bits: i32,
        linear_num_key_heads: i32,
        linear_num_value_heads: i32,
        linear_key_head_dim: i32,
        linear_value_head_dim: i32,
    ) -> Result<Self, String> {
        let rows_usize = usize::try_from(rows).map_err(|_| "rows must fit usize")?;
        let cols_usize = usize::try_from(cols).map_err(|_| "cols must fit usize")?;
        let down_cols_usize = usize::try_from(down_cols).map_err(|_| "down-cols must fit usize")?;
        if candidate == Candidate::QwenLinearAttentionInputsPacked {
            return Self::new_qwen_linear_attention_inputs_packed(
                rows,
                cols,
                rows_usize,
                cols_usize,
                group_size,
                bits,
                linear_num_key_heads,
                linear_num_value_heads,
                linear_key_head_dim,
                linear_value_head_dim,
            );
        }
        if candidate == Candidate::QuantizedFfn {
            return Self::new_quantized_ffn(
                rows,
                cols,
                down_cols,
                rows_usize,
                cols_usize,
                down_cols_usize,
                group_size,
                bits,
            );
        }
        if candidate == Candidate::Gemma4PostAttnFfnBlock {
            return Self::new_gemma4_post_attn_ffn_block(
                rows,
                cols,
                down_cols,
                rows_usize,
                cols_usize,
                down_cols_usize,
                group_size,
                bits,
            );
        }
        if candidate == Candidate::QkNormRope {
            return Self::new_qk_norm_rope(
                rows,
                cols,
                usize::try_from(rows).map_err(|_| "rows must fit usize")?,
                usize::try_from(cols).map_err(|_| "cols must fit usize")?,
                head_dim,
            );
        }

        let len = rows_usize
            .checked_mul(cols_usize)
            .ok_or("rows * cols overflowed usize")?;
        let gate_data = (0..len)
            .map(|idx| ((idx % 257) as f32 - 128.0) / 64.0)
            .collect::<Vec<_>>();
        let up_data = (0..len)
            .map(|idx| ((idx % 193) as f32 - 96.0) / 96.0)
            .collect::<Vec<_>>();
        let (down_weight_data, down_weight) = if candidate == Candidate::ActivationDown {
            let down_len = cols_usize
                .checked_mul(down_cols_usize)
                .ok_or("cols * down-cols overflowed usize")?;
            let data = (0..down_len)
                .map(|idx| ((idx % 251) as f32 - 125.0) / (251.0 * cols_usize as f32).sqrt())
                .collect::<Vec<_>>();
            let weight = MlxArray::from_raw_data(
                data.as_ptr().cast(),
                std::mem::size_of_val(data.as_slice()),
                &[cols, down_cols],
                MlxDtype::Float32,
            );
            (Some(data), Some(weight))
        } else {
            (None, None)
        };
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
        if let Some(weight) = down_weight.as_ref() {
            eval(&[&gate, &up, weight]);
        } else {
            eval(&[&gate, &up]);
        }
        Ok(Self {
            _gate_data: Some(gate_data),
            _up_data: Some(up_data),
            _input_data: None,
            _down_weight_data: down_weight_data,
            _gate_up_weight_data: None,
            _proj_data: None,
            _norm_data: None,
            _post_norm_data: None,
            _layer_scalar_data: None,
            _linear_qkvz_weight_data: None,
            _linear_ba_weight_data: None,
            gate: Some(gate),
            up: Some(up),
            down_weight,
            input: None,
            attn: None,
            proj: None,
            norm: None,
            post_norm: None,
            layer_scalar: None,
            gate_up_quantized: None,
            down_quantized: None,
            linear_attention: None,
            group_size,
            bits,
            n_heads: 0,
            head_dim: 0,
        })
    }

    fn new_qk_norm_rope(
        rows: i32,
        cols: i32,
        rows_usize: usize,
        cols_usize: usize,
        head_dim: i32,
    ) -> Result<Self, String> {
        if cols % head_dim != 0 {
            return Err("cols must divide by head-dim".into());
        }
        let len = rows_usize
            .checked_mul(cols_usize)
            .ok_or("rows * cols overflowed usize")?;
        let proj_data = (0..len)
            .map(|idx| ((idx % 257) as f32 - 128.0) / 128.0)
            .collect::<Vec<_>>();
        let norm_data = (0..head_dim)
            .map(|idx| 0.75 + (idx as f32 % 17.0) * 0.03125)
            .collect::<Vec<_>>();
        let proj = MlxArray::from_raw_data(
            proj_data.as_ptr().cast(),
            std::mem::size_of_val(proj_data.as_slice()),
            &[1, rows, cols],
            MlxDtype::Float32,
        );
        let norm = MlxArray::from_raw_data(
            norm_data.as_ptr().cast(),
            std::mem::size_of_val(norm_data.as_slice()),
            &[head_dim],
            MlxDtype::Float32,
        );
        eval(&[&proj, &norm]);
        Ok(Self {
            _gate_data: None,
            _up_data: None,
            _input_data: None,
            _down_weight_data: None,
            _gate_up_weight_data: None,
            _proj_data: Some(proj_data),
            _norm_data: Some(norm_data),
            _post_norm_data: None,
            _layer_scalar_data: None,
            _linear_qkvz_weight_data: None,
            _linear_ba_weight_data: None,
            gate: None,
            up: None,
            down_weight: None,
            input: None,
            attn: None,
            proj: Some(proj),
            norm: Some(norm),
            post_norm: None,
            layer_scalar: None,
            gate_up_quantized: None,
            down_quantized: None,
            linear_attention: None,
            group_size: DEFAULT_GROUP_SIZE,
            bits: DEFAULT_BITS,
            n_heads: cols / head_dim,
            head_dim,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn new_quantized_ffn(
        rows: i32,
        cols: i32,
        down_cols: i32,
        rows_usize: usize,
        cols_usize: usize,
        down_cols_usize: usize,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let input_len = rows_usize
            .checked_mul(down_cols_usize)
            .ok_or("rows * down-cols overflowed usize")?;
        let gate_up_weight_len = cols_usize
            .checked_mul(2)
            .and_then(|v| v.checked_mul(down_cols_usize))
            .ok_or("2 * cols * down-cols overflowed usize")?;
        let down_weight_len = down_cols_usize
            .checked_mul(cols_usize)
            .ok_or("down-cols * cols overflowed usize")?;

        let input_data = (0..input_len)
            .map(|idx| ((idx % 257) as f32 - 128.0) / 128.0)
            .collect::<Vec<_>>();
        let gate_up_weight_data = (0..gate_up_weight_len)
            .map(|idx| ((idx % 251) as f32 - 125.0) / (251.0 * down_cols_usize as f32).sqrt())
            .collect::<Vec<_>>();
        let down_weight_data = (0..down_weight_len)
            .map(|idx| ((idx % 241) as f32 - 120.0) / (241.0 * cols_usize as f32).sqrt())
            .collect::<Vec<_>>();

        let input = MlxArray::from_raw_data(
            input_data.as_ptr().cast(),
            std::mem::size_of_val(input_data.as_slice()),
            &[rows, down_cols],
            MlxDtype::Float32,
        );
        let gate_up_weight = MlxArray::from_raw_data(
            gate_up_weight_data.as_ptr().cast(),
            std::mem::size_of_val(gate_up_weight_data.as_slice()),
            &[cols * 2, down_cols],
            MlxDtype::Float32,
        );
        let down_weight = MlxArray::from_raw_data(
            down_weight_data.as_ptr().cast(),
            std::mem::size_of_val(down_weight_data.as_slice()),
            &[down_cols, cols],
            MlxDtype::Float32,
        );
        let gate_up_quantized = quantized_fixture_weight(&gate_up_weight, group_size, bits)?;
        let down_quantized = quantized_fixture_weight(&down_weight, group_size, bits)?;
        eval(&[
            &input,
            &gate_up_quantized.weight,
            &gate_up_quantized.scales,
            &gate_up_quantized.biases,
            &down_quantized.weight,
            &down_quantized.scales,
            &down_quantized.biases,
        ]);

        Ok(Self {
            _gate_data: None,
            _up_data: None,
            _input_data: Some(input_data),
            _down_weight_data: Some(down_weight_data),
            _gate_up_weight_data: Some(gate_up_weight_data),
            _proj_data: None,
            _norm_data: None,
            _post_norm_data: None,
            _layer_scalar_data: None,
            _linear_qkvz_weight_data: None,
            _linear_ba_weight_data: None,
            gate: None,
            up: None,
            down_weight: None,
            input: Some(input),
            attn: None,
            proj: None,
            norm: None,
            post_norm: None,
            layer_scalar: None,
            gate_up_quantized: Some(gate_up_quantized),
            down_quantized: Some(down_quantized),
            linear_attention: None,
            group_size,
            bits,
            n_heads: 0,
            head_dim: 0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn new_gemma4_post_attn_ffn_block(
        rows: i32,
        cols: i32,
        down_cols: i32,
        rows_usize: usize,
        cols_usize: usize,
        down_cols_usize: usize,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let hidden_len = rows_usize
            .checked_mul(down_cols_usize)
            .ok_or("rows * down-cols overflowed usize")?;
        let gate_up_weight_len = cols_usize
            .checked_mul(2)
            .and_then(|v| v.checked_mul(down_cols_usize))
            .ok_or("2 * cols * down-cols overflowed usize")?;
        let down_weight_len = down_cols_usize
            .checked_mul(cols_usize)
            .ok_or("down-cols * cols overflowed usize")?;

        let hidden_data = (0..hidden_len)
            .map(|idx| ((idx % 257) as f32 - 128.0) / 128.0)
            .collect::<Vec<_>>();
        let attn_data = (0..hidden_len)
            .map(|idx| ((idx % 193) as f32 - 96.0) / 192.0)
            .collect::<Vec<_>>();
        let norm_data = (0..down_cols_usize)
            .map(|idx| 0.75 + (idx as f32 % 31.0) * 0.00390625)
            .collect::<Vec<_>>();
        let post_norm_data = (0..down_cols_usize)
            .map(|idx| 0.875 + (idx as f32 % 29.0) * 0.001953125)
            .collect::<Vec<_>>();
        let layer_scalar_data = vec![0.9375_f32];
        let gate_up_weight_data = (0..gate_up_weight_len)
            .map(|idx| ((idx % 251) as f32 - 125.0) / (251.0 * down_cols_usize as f32).sqrt())
            .collect::<Vec<_>>();
        let down_weight_data = (0..down_weight_len)
            .map(|idx| ((idx % 241) as f32 - 120.0) / (241.0 * cols_usize as f32).sqrt())
            .collect::<Vec<_>>();

        let hidden = MlxArray::from_raw_data(
            hidden_data.as_ptr().cast(),
            std::mem::size_of_val(hidden_data.as_slice()),
            &[rows, down_cols],
            MlxDtype::Float32,
        );
        let attn = MlxArray::from_raw_data(
            attn_data.as_ptr().cast(),
            std::mem::size_of_val(attn_data.as_slice()),
            &[rows, down_cols],
            MlxDtype::Float32,
        );
        let norm = MlxArray::from_raw_data(
            norm_data.as_ptr().cast(),
            std::mem::size_of_val(norm_data.as_slice()),
            &[down_cols],
            MlxDtype::Float32,
        );
        let post_norm = MlxArray::from_raw_data(
            post_norm_data.as_ptr().cast(),
            std::mem::size_of_val(post_norm_data.as_slice()),
            &[down_cols],
            MlxDtype::Float32,
        );
        let layer_scalar = MlxArray::from_raw_data(
            layer_scalar_data.as_ptr().cast(),
            std::mem::size_of_val(layer_scalar_data.as_slice()),
            &[1],
            MlxDtype::Float32,
        );
        let gate_up_weight = MlxArray::from_raw_data(
            gate_up_weight_data.as_ptr().cast(),
            std::mem::size_of_val(gate_up_weight_data.as_slice()),
            &[cols * 2, down_cols],
            MlxDtype::Float32,
        );
        let down_weight = MlxArray::from_raw_data(
            down_weight_data.as_ptr().cast(),
            std::mem::size_of_val(down_weight_data.as_slice()),
            &[down_cols, cols],
            MlxDtype::Float32,
        );
        let gate_up_quantized = quantized_fixture_weight(&gate_up_weight, group_size, bits)?;
        let down_quantized = quantized_fixture_weight(&down_weight, group_size, bits)?;
        eval(&[
            &hidden,
            &attn,
            &norm,
            &post_norm,
            &layer_scalar,
            &gate_up_quantized.weight,
            &gate_up_quantized.scales,
            &gate_up_quantized.biases,
            &down_quantized.weight,
            &down_quantized.scales,
            &down_quantized.biases,
        ]);

        Ok(Self {
            _gate_data: None,
            _up_data: None,
            _input_data: Some(hidden_data),
            _down_weight_data: Some(down_weight_data),
            _gate_up_weight_data: Some(gate_up_weight_data),
            _proj_data: Some(attn_data),
            _norm_data: Some(norm_data),
            _post_norm_data: Some(post_norm_data),
            _layer_scalar_data: Some(layer_scalar_data),
            _linear_qkvz_weight_data: None,
            _linear_ba_weight_data: None,
            gate: None,
            up: None,
            down_weight: None,
            input: Some(hidden),
            attn: Some(attn),
            proj: None,
            norm: Some(norm),
            post_norm: Some(post_norm),
            layer_scalar: Some(layer_scalar),
            gate_up_quantized: Some(gate_up_quantized),
            down_quantized: Some(down_quantized),
            linear_attention: None,
            group_size,
            bits,
            n_heads: 0,
            head_dim: 0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn new_qwen_linear_attention_inputs_packed(
        rows: i32,
        cols: i32,
        rows_usize: usize,
        cols_usize: usize,
        group_size: i32,
        bits: i32,
        num_key_heads: i32,
        num_value_heads: i32,
        key_head_dim: i32,
        value_head_dim: i32,
    ) -> Result<Self, String> {
        let value_heads_per_key = num_value_heads / num_key_heads;
        let value_dim_per_key = value_heads_per_key
            .checked_mul(value_head_dim)
            .ok_or("value_heads_per_key * value_head_dim overflowed i32")?;
        let qkvz_per_key = key_head_dim
            .checked_mul(2)
            .and_then(|v| v.checked_add(value_dim_per_key.checked_mul(2)?))
            .ok_or("qkvz_per_key overflowed i32")?;
        let qkvz_out = num_key_heads
            .checked_mul(qkvz_per_key)
            .ok_or("num_key_heads * qkvz_per_key overflowed i32")?;
        let ba_out = num_key_heads
            .checked_mul(value_heads_per_key)
            .and_then(|v| v.checked_mul(2))
            .ok_or("packed ba output width overflowed i32")?;

        let qkvz_out_usize = usize::try_from(qkvz_out).map_err(|_| "qkvz_out must fit usize")?;
        let ba_out_usize = usize::try_from(ba_out).map_err(|_| "ba_out must fit usize")?;
        let input_len = rows_usize
            .checked_mul(cols_usize)
            .ok_or("rows * cols overflowed usize")?;
        let qkvz_weight_len = qkvz_out_usize
            .checked_mul(cols_usize)
            .ok_or("qkvz_out * cols overflowed usize")?;
        let ba_weight_len = ba_out_usize
            .checked_mul(cols_usize)
            .ok_or("ba_out * cols overflowed usize")?;

        let input_data = (0..input_len)
            .map(|idx| ((idx % 257) as f32 - 128.0) / 128.0)
            .collect::<Vec<_>>();
        let qkvz_weight_data = (0..qkvz_weight_len)
            .map(|idx| ((idx % 251) as f32 - 125.0) / (251.0 * cols_usize as f32).sqrt())
            .collect::<Vec<_>>();
        let ba_weight_data = (0..ba_weight_len)
            .map(|idx| ((idx % 241) as f32 - 120.0) / (241.0 * cols_usize as f32).sqrt())
            .collect::<Vec<_>>();

        let input = MlxArray::from_raw_data(
            input_data.as_ptr().cast(),
            std::mem::size_of_val(input_data.as_slice()),
            &[1, rows, cols],
            MlxDtype::Float32,
        );
        let qkvz_weight = MlxArray::from_raw_data(
            qkvz_weight_data.as_ptr().cast(),
            std::mem::size_of_val(qkvz_weight_data.as_slice()),
            &[qkvz_out, cols],
            MlxDtype::Float32,
        );
        let ba_weight = MlxArray::from_raw_data(
            ba_weight_data.as_ptr().cast(),
            std::mem::size_of_val(ba_weight_data.as_slice()),
            &[ba_out, cols],
            MlxDtype::Float32,
        );
        let qkvz = quantized_fixture_weight(&qkvz_weight, group_size, bits)?;
        let ba = quantized_fixture_weight(&ba_weight, group_size, bits)?;
        eval(&[
            &input,
            &qkvz.weight,
            &qkvz.scales,
            &qkvz.biases,
            &ba.weight,
            &ba.scales,
            &ba.biases,
        ]);

        Ok(Self {
            _gate_data: None,
            _up_data: None,
            _input_data: Some(input_data),
            _down_weight_data: None,
            _gate_up_weight_data: None,
            _proj_data: None,
            _norm_data: None,
            _post_norm_data: None,
            _layer_scalar_data: None,
            _linear_qkvz_weight_data: Some(qkvz_weight_data),
            _linear_ba_weight_data: Some(ba_weight_data),
            gate: None,
            up: None,
            down_weight: None,
            input: Some(input),
            attn: None,
            proj: None,
            norm: None,
            post_norm: None,
            layer_scalar: None,
            gate_up_quantized: None,
            down_quantized: None,
            linear_attention: Some(LinearAttentionFixture {
                qkvz,
                ba,
                num_key_heads,
                num_value_heads,
                key_head_dim,
                value_head_dim,
            }),
            group_size,
            bits,
            n_heads: 0,
            head_dim: 0,
        })
    }
}

fn quantized_fixture_weight(
    weight: &MlxArray,
    group_size: i32,
    bits: i32,
) -> Result<QuantizedFixtureWeight, String> {
    let mut q = quantize(
        weight,
        Some(group_size),
        Some(bits),
        MlxQuantizationMode::Affine,
        None,
        None,
    );
    if q.len() != 3 {
        return Err(format!(
            "affine quantize returned {} arrays, expected 3",
            q.len()
        ));
    }
    let biases = q.pop().expect("affine quantize must return biases");
    let scales = q.pop().expect("affine quantize must return scales");
    let weight = q.pop().expect("affine quantize must return packed weight");
    Ok(QuantizedFixtureWeight {
        weight,
        scales,
        biases,
    })
}

fn warmup(fixture: &Fixture, candidate: Candidate, warmup: usize) {
    for _ in 0..warmup {
        let portable = portable_candidate(fixture, candidate);
        let direct = direct_candidate(fixture, candidate);
        let mut arrays = portable.arrays();
        arrays.extend(direct.arrays());
        eval(&arrays);
    }
}

enum CandidateOutput {
    Single(MlxArray),
    LinearAttentionInputs {
        qkv: MlxArray,
        z: MlxArray,
        a: MlxArray,
        b: MlxArray,
    },
}

impl CandidateOutput {
    fn arrays(&self) -> Vec<&MlxArray> {
        match self {
            Self::Single(output) => vec![output],
            Self::LinearAttentionInputs { qkv, z, a, b } => vec![qkv, z, a, b],
        }
    }

    fn primary_shape(&self) -> Vec<i32> {
        match self {
            Self::Single(output) => output.shape(),
            Self::LinearAttentionInputs { qkv, .. } => qkv.shape(),
        }
    }

    fn component_shapes(&self) -> Option<Value> {
        match self {
            Self::Single(_) => None,
            Self::LinearAttentionInputs { qkv, z, a, b } => Some(json!({
                "qkv": qkv.shape(),
                "z": z.shape(),
                "a": a.shape(),
                "b": b.shape(),
            })),
        }
    }

    fn max_abs_error_against(&self, other: &Self) -> f32 {
        match (self, other) {
            (Self::Single(left), Self::Single(right)) => max_abs_error(left, right),
            (
                Self::LinearAttentionInputs {
                    qkv: l_qkv,
                    z: l_z,
                    a: l_a,
                    b: l_b,
                },
                Self::LinearAttentionInputs {
                    qkv: r_qkv,
                    z: r_z,
                    a: r_a,
                    b: r_b,
                },
            ) => [
                max_abs_error(l_qkv, r_qkv),
                max_abs_error(l_z, r_z),
                max_abs_error(l_a, r_a),
                max_abs_error(l_b, r_b),
            ]
            .into_iter()
            .fold(0.0_f32, f32::max),
            _ => panic!("candidate output variants must match"),
        }
    }
}

fn portable_candidate(fixture: &Fixture, candidate: Candidate) -> CandidateOutput {
    match candidate {
        Candidate::Activation => CandidateOutput::Single(multiply(
            &gelu_approx(fixture.gate(), None),
            fixture.up(),
            None,
        )),
        Candidate::ActivationDown => {
            let hidden = multiply(&gelu_approx(fixture.gate(), None), fixture.up(), None);
            CandidateOutput::Single(matmul(&hidden, fixture.down_weight(), None))
        }
        Candidate::QuantizedFfn => CandidateOutput::Single(portable_quantized_ffn(fixture)),
        Candidate::QkNormRope => CandidateOutput::Single(portable_qk_norm_rope(fixture)),
        Candidate::Gemma4PostAttnFfnBlock => {
            CandidateOutput::Single(portable_gemma4_post_attn_ffn_block(fixture))
        }
        Candidate::QwenLinearAttentionInputsPacked => {
            portable_qwen_linear_attention_inputs(fixture)
        }
    }
}

fn direct_candidate(fixture: &Fixture, candidate: Candidate) -> CandidateOutput {
    match candidate {
        Candidate::Activation => {
            CandidateOutput::Single(gelu_approx_mul(fixture.gate(), fixture.up(), None))
        }
        Candidate::ActivationDown => CandidateOutput::Single(gelu_approx_mul_matmul(
            fixture.gate(),
            fixture.up(),
            fixture.down_weight(),
            None,
        )),
        Candidate::QuantizedFfn => {
            let gate_up = fixture.gate_up_quantized();
            let down = fixture.down_quantized();
            CandidateOutput::Single(gelu_approx_quantized_ffn(
                fixture.input(),
                &gate_up.weight,
                &gate_up.scales,
                Some(&gate_up.biases),
                &down.weight,
                &down.scales,
                Some(&down.biases),
                fixture.group_size,
                fixture.bits,
                None,
            ))
        }
        Candidate::QkNormRope => CandidateOutput::Single(qk_norm_rope_bhsd_from_proj(
            fixture.proj(),
            Some(fixture.norm()),
            fixture.n_heads,
            fixture.head_dim,
            1.0e-6,
            fixture.head_dim,
            false,
            Some(10_000.0),
            0,
            None,
            None,
        )),
        Candidate::Gemma4PostAttnFfnBlock => {
            let gate_up = fixture.gate_up_quantized();
            let down = fixture.down_quantized();
            CandidateOutput::Single(gemma4_post_attn_ffn_block(
                fixture.input(),
                fixture.attn(),
                fixture.norm(),
                Some(fixture.post_norm()),
                Some(fixture.layer_scalar()),
                &gate_up.weight,
                &gate_up.scales,
                Some(&gate_up.biases),
                &down.weight,
                &down.scales,
                Some(&down.biases),
                fixture.group_size,
                fixture.bits,
                1.0e-6,
                None,
            ))
        }
        Candidate::QwenLinearAttentionInputsPacked => direct_qwen_linear_attention_inputs(fixture),
    }
}

impl Fixture {
    fn gate(&self) -> &MlxArray {
        self.gate
            .as_ref()
            .expect("gate is required for activation candidates")
    }

    fn up(&self) -> &MlxArray {
        self.up
            .as_ref()
            .expect("up is required for activation candidates")
    }

    fn down_weight(&self) -> &MlxArray {
        self.down_weight
            .as_ref()
            .expect("down weight is required for activation-down candidate")
    }

    fn input(&self) -> &MlxArray {
        self.input
            .as_ref()
            .expect("input is required for quantized FFN candidate")
    }

    fn attn(&self) -> &MlxArray {
        self.attn
            .as_ref()
            .expect("attention output is required for Gemma4 post-attention FFN block candidate")
    }

    fn gate_up_quantized(&self) -> &QuantizedFixtureWeight {
        self.gate_up_quantized
            .as_ref()
            .expect("gate/up weight is required for quantized FFN candidate")
    }

    fn down_quantized(&self) -> &QuantizedFixtureWeight {
        self.down_quantized
            .as_ref()
            .expect("down weight is required for quantized FFN candidate")
    }

    fn proj(&self) -> &MlxArray {
        self.proj
            .as_ref()
            .expect("projection is required for qk_norm_rope candidate")
    }

    fn norm(&self) -> &MlxArray {
        self.norm
            .as_ref()
            .expect("norm is required for qk_norm_rope candidate")
    }

    fn post_norm(&self) -> &MlxArray {
        self.post_norm
            .as_ref()
            .expect("post-FFN norm is required for Gemma4 post-attention FFN block candidate")
    }

    fn layer_scalar(&self) -> &MlxArray {
        self.layer_scalar
            .as_ref()
            .expect("layer scalar is required for Gemma4 post-attention FFN block candidate")
    }

    fn linear_attention(&self) -> &LinearAttentionFixture {
        self.linear_attention
            .as_ref()
            .expect("linear attention fixture is required for Qwen linear-attention candidate")
    }
}

fn portable_quantized_ffn(fixture: &Fixture) -> MlxArray {
    portable_quantized_ffn_for_input(fixture, fixture.input())
}

fn portable_quantized_ffn_for_input(fixture: &Fixture, input: &MlxArray) -> MlxArray {
    let gate_up_weight = fixture.gate_up_quantized();
    let down_weight = fixture.down_quantized();
    let gate_up = quantized_matmul(
        input,
        &gate_up_weight.weight,
        &gate_up_weight.scales,
        Some(&gate_up_weight.biases),
        true,
        Some(fixture.group_size),
        Some(fixture.bits),
        None,
    );
    let packed_dim = gate_up
        .shape()
        .last()
        .copied()
        .expect("gate/up output must have a last dimension");
    let half = packed_dim / 2;
    let gate = slice_last_dim(&gate_up, 0, half, None);
    let up = slice_last_dim(&gate_up, half, packed_dim, None);
    let hidden = gelu_approx_mul(&gate, &up, None);
    quantized_matmul(
        &hidden,
        &down_weight.weight,
        &down_weight.scales,
        Some(&down_weight.biases),
        true,
        Some(fixture.group_size),
        Some(fixture.bits),
        None,
    )
}

fn portable_gemma4_post_attn_ffn_block(fixture: &Fixture) -> MlxArray {
    let residual = add(fixture.input(), fixture.attn(), None);
    let normed = rms_norm(&residual, Some(fixture.norm()), 1.0e-6, None);
    let ffn = portable_quantized_ffn_for_input(fixture, &normed);
    let ffn = rms_norm(&ffn, Some(fixture.post_norm()), 1.0e-6, None);
    multiply(&add(&residual, &ffn, None), fixture.layer_scalar(), None)
}

fn portable_qk_norm_rope(fixture: &Fixture) -> MlxArray {
    let rows = fixture.proj().shape()[1];
    let cols = fixture.proj().shape()[2];
    let bhsd = as_strided(
        fixture.proj(),
        &[1, fixture.n_heads, rows, fixture.head_dim],
        &[
            i64::from(rows) * i64::from(cols),
            i64::from(fixture.head_dim),
            i64::from(cols),
            1,
        ],
        0,
        None,
    );
    let normed = rms_norm(&bhsd, Some(fixture.norm()), 1.0e-6, None);
    rope(
        &normed,
        fixture.head_dim,
        false,
        Some(10_000.0),
        1.0,
        0,
        None,
        None,
    )
}

fn portable_qwen_linear_attention_inputs(fixture: &Fixture) -> CandidateOutput {
    let linear = fixture.linear_attention();
    let value_heads_per_key = linear.num_value_heads / linear.num_key_heads;
    let value_dim_per_key = value_heads_per_key * linear.value_head_dim;
    let qkvz_per_key = linear.key_head_dim * 2 + value_dim_per_key * 2;

    let mixed_qkvz = quantized_matmul(
        fixture.input(),
        &linear.qkvz.weight,
        &linear.qkvz.scales,
        Some(&linear.qkvz.biases),
        true,
        Some(fixture.group_size),
        Some(fixture.bits),
        None,
    );
    let mixed_qkvz = reshape(
        &mixed_qkvz,
        &[
            1,
            fixture.input().shape()[1],
            linear.num_key_heads,
            qkvz_per_key,
        ],
        None,
    );
    let q = slice_last_dim(&mixed_qkvz, 0, linear.key_head_dim, None);
    let k = slice_last_dim(
        &mixed_qkvz,
        linear.key_head_dim,
        linear.key_head_dim * 2,
        None,
    );
    let v = slice_last_dim(
        &mixed_qkvz,
        linear.key_head_dim * 2,
        linear.key_head_dim * 2 + value_dim_per_key,
        None,
    );
    let z = slice_last_dim(
        &mixed_qkvz,
        linear.key_head_dim * 2 + value_dim_per_key,
        qkvz_per_key,
        None,
    );
    let qkv = concatenate(
        &[
            &reshape(
                &q,
                &[
                    1,
                    fixture.input().shape()[1],
                    linear.num_key_heads * linear.key_head_dim,
                ],
                None,
            ),
            &reshape(
                &k,
                &[
                    1,
                    fixture.input().shape()[1],
                    linear.num_key_heads * linear.key_head_dim,
                ],
                None,
            ),
            &reshape(
                &v,
                &[
                    1,
                    fixture.input().shape()[1],
                    linear.num_value_heads * linear.value_head_dim,
                ],
                None,
            ),
        ],
        2,
        None,
    );
    let z = reshape(
        &z,
        &[
            1,
            fixture.input().shape()[1],
            linear.num_value_heads,
            linear.value_head_dim,
        ],
        None,
    );

    let mixed_ba = quantized_matmul(
        fixture.input(),
        &linear.ba.weight,
        &linear.ba.scales,
        Some(&linear.ba.biases),
        true,
        Some(fixture.group_size),
        Some(fixture.bits),
        None,
    );
    let ba = reshape(
        &mixed_ba,
        &[
            1,
            fixture.input().shape()[1],
            linear.num_key_heads,
            value_heads_per_key * 2,
        ],
        None,
    );
    let b = reshape(
        &slice_last_dim(&ba, 0, value_heads_per_key, None),
        &[1, fixture.input().shape()[1], linear.num_value_heads],
        None,
    );
    let a = reshape(
        &slice_last_dim(&ba, value_heads_per_key, value_heads_per_key * 2, None),
        &[1, fixture.input().shape()[1], linear.num_value_heads],
        None,
    );

    CandidateOutput::LinearAttentionInputs { qkv, z, a, b }
}

fn direct_qwen_linear_attention_inputs(fixture: &Fixture) -> CandidateOutput {
    let linear = fixture.linear_attention();
    let (qkv, z, a, b) = qwen_linear_attention_inputs_packed(
        fixture.input(),
        &linear.qkvz.weight,
        Some(&linear.qkvz.scales),
        Some(&linear.qkvz.biases),
        &linear.ba.weight,
        Some(&linear.ba.scales),
        Some(&linear.ba.biases),
        linear.num_key_heads,
        linear.num_value_heads,
        linear.key_head_dim,
        linear.value_head_dim,
        fixture.group_size,
        fixture.bits,
        None,
    )
    .expect("direct Qwen linear-attention packed input shim should accept fixture shapes");
    CandidateOutput::LinearAttentionInputs { qkv, z, a, b }
}

fn measure(_name: &str, iterations: usize, mut op: impl FnMut() -> CandidateOutput) -> Measurement {
    let mut samples_us = Vec::with_capacity(iterations);
    let mut op_counts = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let op_snapshot = op_count_snapshot();
        let started = Instant::now();
        let output = op();
        let arrays = output.arrays();
        eval(&arrays);
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
    component_shapes: Option<Value>,
}

fn correctness(fixture: &Fixture, candidate: Candidate) -> Correctness {
    let portable = portable_candidate(fixture, candidate);
    let direct = direct_candidate(fixture, candidate);
    let mut arrays = portable.arrays();
    arrays.extend(direct.arrays());
    eval(&arrays);
    Correctness {
        max_abs_error: portable.max_abs_error_against(&direct),
        shape: direct.primary_shape(),
        component_shapes: direct.component_shapes(),
    }
}

fn max_abs_error(left: &MlxArray, right: &MlxArray) -> f32 {
    let left_data = left.data_f32();
    let right_data = right.data_f32();
    left_data
        .iter()
        .zip(right_data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
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
