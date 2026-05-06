use std::env;
use std::path::PathBuf;
use std::time::Instant;

use ax_engine_core::MlxTurboQuantPreset;
use ax_engine_mlx::turboquant::{
    TurboQuantBlockLayout, TurboQuantBlockLayoutConfig, TurboQuantCompressedBlockBuffer,
    TurboQuantCompressedDecodePlan, compare_decode_outputs, merge_attention_partition_stats,
    reference_decode_attention, reference_decode_attention_partition_stats,
};
use ax_engine_mlx::turboquant_metal::{
    turboquant_fused_cold_decode_metal, turboquant_fused_cold_decode_metal_head_serial,
    turboquant_fused_cold_decode_metal_two_stage,
};
use serde_json::json;

const SCHEMA_VERSION: &str = "ax.turboquant_fused_decode_microbench.v1";

#[derive(Clone, Debug)]
struct Config {
    cold_tokens: Vec<usize>,
    hot_tokens: usize,
    variants: Vec<String>,
    n_kv_heads: usize,
    head_dim: usize,
    block_tokens: usize,
    value_group_size: usize,
    repetitions: usize,
    warmup: usize,
    output: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            cold_tokens: vec![512, 2048],
            hot_tokens: 0,
            variants: known_variants()
                .iter()
                .map(|variant| (*variant).to_string())
                .collect(),
            n_kv_heads: 1,
            head_dim: 128,
            block_tokens: 256,
            value_group_size: 32,
            repetitions: 5,
            warmup: 1,
            output: None,
        }
    }
}

fn main() {
    let config = parse_args(env::args().skip(1).collect()).unwrap_or_else(|message| {
        eprintln!("{message}");
        std::process::exit(2);
    });
    let artifact = run(config).unwrap_or_else(|message| {
        eprintln!("{message}");
        std::process::exit(1);
    });
    let pretty = serde_json::to_string_pretty(&artifact).expect("serialize microbench artifact");
    if let Some(path) = artifact
        .get("output_path")
        .and_then(|value| value.as_str())
        .map(PathBuf::from)
    {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create output parent");
        }
        std::fs::write(&path, pretty.as_bytes()).expect("write output artifact");
        println!("Saved to {}", path.display());
    } else {
        println!("{pretty}");
    }
}

fn parse_args(args: Vec<String>) -> Result<Config, String> {
    let mut config = Config::default();
    let mut idx = 0usize;
    while idx < args.len() {
        let arg = &args[idx];
        let value = |idx: &mut usize| -> Result<String, String> {
            *idx += 1;
            args.get(*idx)
                .cloned()
                .ok_or_else(|| format!("{arg} requires a value"))
        };

        match arg.as_str() {
            "--cold-tokens" => {
                config.cold_tokens = parse_usize_list(&value(&mut idx)?, "--cold-tokens")?;
            }
            "--hot-tokens" => {
                config.hot_tokens = parse_usize(&value(&mut idx)?, "--hot-tokens")?;
            }
            "--variants" => {
                config.variants = parse_variants(&value(&mut idx)?)?;
            }
            "--n-kv-heads" => {
                config.n_kv_heads = parse_usize(&value(&mut idx)?, "--n-kv-heads")?;
            }
            "--head-dim" => {
                config.head_dim = parse_usize(&value(&mut idx)?, "--head-dim")?;
            }
            "--block-tokens" => {
                config.block_tokens = parse_usize(&value(&mut idx)?, "--block-tokens")?;
            }
            "--value-group-size" => {
                config.value_group_size = parse_usize(&value(&mut idx)?, "--value-group-size")?;
            }
            "--repetitions" => {
                config.repetitions = parse_usize(&value(&mut idx)?, "--repetitions")?;
            }
            "--warmup" => {
                config.warmup = parse_usize(&value(&mut idx)?, "--warmup")?;
            }
            "--output" => {
                config.output = Some(PathBuf::from(value(&mut idx)?));
            }
            "--help" | "-h" => {
                return Err(help());
            }
            other => {
                return Err(format!("unknown argument {other}\n\n{}", help()));
            }
        }
        idx += 1;
    }

    if config.cold_tokens.is_empty() {
        return Err("--cold-tokens must not be empty".to_string());
    }
    if config.repetitions == 0 {
        return Err("--repetitions must be > 0".to_string());
    }

    Ok(config)
}

fn run(config: Config) -> Result<serde_json::Value, String> {
    let mut rows = Vec::new();
    for &cold_tokens in &config.cold_tokens {
        let row = run_case(&config, cold_tokens)?;
        rows.push(row);
    }

    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "kernel": "turboquant_fused_cold_decode_k8v4",
        "decode_path": "fused_compressed_decode",
        "preset": "k8v4",
        "key_bits": 8,
        "value_bits": 4,
        "output_path": config.output.as_ref().map(|path| path.display().to_string()),
        "config": {
            "n_kv_heads": config.n_kv_heads,
            "head_dim": config.head_dim,
            "variants": config.variants,
            "hot_tokens": config.hot_tokens,
            "block_tokens": config.block_tokens,
            "value_group_size": config.value_group_size,
            "repetitions": config.repetitions,
            "warmup": config.warmup,
        },
        "rows": rows,
    }))
}

fn run_case(config: &Config, cold_tokens: usize) -> Result<serde_json::Value, String> {
    let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
        preset: MlxTurboQuantPreset::K8V4,
        block_tokens: config.block_tokens,
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        value_group_size: config.value_group_size,
    })
    .map_err(|error| error.to_string())?;
    let plan =
        TurboQuantCompressedDecodePlan::new(layout, cold_tokens, 0).map_err(|e| e.to_string())?;
    let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
    for token_index in 0..cold_tokens {
        buffer
            .write_token(token_index, &token_heads(token_index, config))
            .map_err(|error| error.to_string())?;
    }
    let queries = queries(config);
    let descriptor = plan
        .fused_decode_launch_descriptor(&buffer, &queries)
        .map_err(|error| error.to_string())?;

    let reference_started = Instant::now();
    let expected = buffer
        .debug_decode_attention_for_all_heads(&queries, cold_tokens)
        .map_err(|error| error.to_string())?;
    let reference_wall_us = elapsed_us(reference_started);
    let hot_tail_merge = if config.hot_tokens > 0 {
        Some(hot_tail_merge_quality(
            config,
            cold_tokens,
            &buffer,
            &queries,
        )?)
    } else {
        None
    };

    let workload = descriptor.workload();
    let mut variants = Vec::new();
    for (name, kernel) in [
        (
            "dim_parallel",
            turboquant_fused_cold_decode_metal as KernelFn,
        ),
        (
            "head_serial",
            turboquant_fused_cold_decode_metal_head_serial as KernelFn,
        ),
        (
            "two_stage_scores",
            turboquant_fused_cold_decode_metal_two_stage as KernelFn,
        ),
    ] {
        if config.variants.iter().any(|variant| variant == name) {
            variants.push(measure_variant(
                name, config, descriptor, &buffer, &queries, &expected, kernel,
            )?);
        }
    }

    Ok(json!({
        "cold_tokens": cold_tokens,
        "n_kv_heads": config.n_kv_heads,
        "head_dim": config.head_dim,
        "compressed_buffer_bytes": descriptor.compressed_buffer_bytes,
        "required_compressed_slots": descriptor.required_compressed_slots,
        "full_precision_cold_kv_bytes": workload.full_precision_cold_kv_bytes,
        "estimated_total_read_bytes": workload.estimated_total_read_bytes,
        "estimated_cold_saved_bytes": workload.estimated_cold_saved_bytes,
        "cold_compression_ratio_milli": workload.cold_compression_ratio_milli,
        "cpu_reference_wall_us": reference_wall_us,
        "hot_tail_merge": hot_tail_merge,
        "kernel_variants": variants,
    }))
}

fn hot_tail_merge_quality(
    config: &Config,
    cold_tokens: usize,
    buffer: &TurboQuantCompressedBlockBuffer,
    queries: &[Vec<f32>],
) -> Result<serde_json::Value, String> {
    let cold_stats = buffer
        .debug_decode_partition_stats_for_all_heads(queries, cold_tokens)
        .map_err(|error| error.to_string())?;

    let hot_tokens = (0..config.hot_tokens)
        .map(|offset| token_heads(cold_tokens + offset, config))
        .collect::<Vec<_>>();

    let mut expected_outputs = Vec::with_capacity(config.n_kv_heads);
    let mut actual_outputs = Vec::with_capacity(config.n_kv_heads);
    for (head_index, query) in queries.iter().enumerate() {
        let mut cold_head = buffer
            .debug_reconstruct_head_history(head_index, cold_tokens)
            .map_err(|error| error.to_string())?;
        let hot_head = hot_tokens
            .iter()
            .map(|heads| heads[head_index].clone())
            .collect::<Vec<_>>();
        cold_head.extend(hot_head.clone());
        expected_outputs.push(
            reference_decode_attention(query, &cold_head).map_err(|error| error.to_string())?,
        );

        let hot_stats = reference_decode_attention_partition_stats(query, &hot_head)
            .map_err(|error| error.to_string())?;
        actual_outputs.push(
            merge_attention_partition_stats(&[cold_stats[head_index].clone(), hot_stats])
                .map_err(|error| error.to_string())?,
        );
    }

    let comparison =
        compare_decode_outputs(&expected_outputs, &actual_outputs).map_err(|e| e.to_string())?;
    Ok(json!({
        "hot_tokens": config.hot_tokens,
        "contract": "shared_logsumexp_partition_merge",
        "quality": {
            "max_abs_diff": comparison.max_abs_diff,
            "mean_abs_diff": comparison.mean_abs_diff,
            "min_cosine_similarity": comparison.min_cosine_similarity,
        }
    }))
}

type KernelFn = fn(
    ax_engine_mlx::turboquant::TurboQuantFusedDecodeLaunchDescriptor,
    &TurboQuantCompressedBlockBuffer,
    &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, ax_engine_mlx::turboquant::TurboQuantCodecError>;

fn measure_variant(
    name: &str,
    config: &Config,
    descriptor: ax_engine_mlx::turboquant::TurboQuantFusedDecodeLaunchDescriptor,
    buffer: &TurboQuantCompressedBlockBuffer,
    queries: &[Vec<f32>],
    expected: &[Vec<f32>],
    kernel: KernelFn,
) -> Result<serde_json::Value, String> {
    for _ in 0..config.warmup {
        let _ = kernel(descriptor, buffer, queries).map_err(|error| error.to_string())?;
    }

    let mut metal_wall_us = Vec::with_capacity(config.repetitions);
    let mut actual = Vec::new();
    for _ in 0..config.repetitions {
        let started = Instant::now();
        actual = kernel(descriptor, buffer, queries).map_err(|error| error.to_string())?;
        metal_wall_us.push(elapsed_us(started));
    }

    let comparison = compare_decode_outputs(expected, &actual).map_err(|e| e.to_string())?;
    let workload = descriptor.workload();
    let metal_median = median_u128(&metal_wall_us);
    let token_heads_per_second =
        (descriptor.cold_tokens * config.n_kv_heads) as f64 / (metal_median / 1e6);
    let estimated_read_gib_s = if metal_median > 0.0 {
        workload.estimated_total_read_bytes as f64 / metal_median * 1e6 / 1024.0 / 1024.0 / 1024.0
    } else {
        0.0
    };

    Ok(json!({
        "name": name,
        "metal_wall_us": {
            "median": metal_median,
            "min": *metal_wall_us.iter().min().unwrap_or(&0),
            "max": *metal_wall_us.iter().max().unwrap_or(&0),
            "samples": metal_wall_us,
        },
        "token_heads_per_second": token_heads_per_second,
        "estimated_read_gib_s": estimated_read_gib_s,
        "quality": {
            "max_abs_diff": comparison.max_abs_diff,
            "mean_abs_diff": comparison.mean_abs_diff,
            "min_cosine_similarity": comparison.min_cosine_similarity,
        }
    }))
}

fn token_heads(token_index: usize, config: &Config) -> Vec<(Vec<f32>, Vec<f32>)> {
    (0..config.n_kv_heads)
        .map(|head_index| {
            let key = (0..config.head_dim)
                .map(|dim| deterministic_value(token_index, head_index, dim, 31, 17, 13))
                .collect();
            let value = (0..config.head_dim)
                .map(|dim| deterministic_value(token_index, head_index, dim, 19, 23, 29))
                .collect();
            (key, value)
        })
        .collect()
}

fn queries(config: &Config) -> Vec<Vec<f32>> {
    (0..config.n_kv_heads)
        .map(|head_index| {
            (0..config.head_dim)
                .map(|dim| deterministic_value(7, head_index, dim, 11, 37, 41))
                .collect()
        })
        .collect()
}

fn deterministic_value(
    token_index: usize,
    head_index: usize,
    dim: usize,
    token_mul: usize,
    head_mul: usize,
    dim_mul: usize,
) -> f32 {
    let raw = (token_index * token_mul + head_index * head_mul + dim * dim_mul) % 257;
    (raw as f32 - 128.0) / 128.0
}

fn median_u128(values: &[u128]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let len = sorted.len();
    if len == 0 {
        return 0.0;
    }
    if len.is_multiple_of(2) {
        (sorted[len / 2 - 1] as f64 + sorted[len / 2] as f64) / 2.0
    } else {
        sorted[len / 2] as f64
    }
}

fn elapsed_us(started: Instant) -> u128 {
    started.elapsed().as_micros()
}

fn parse_usize(value: &str, name: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("{name} must be a positive integer, got {value:?}"))
}

fn parse_usize_list(value: &str, name: &str) -> Result<Vec<usize>, String> {
    value
        .split(',')
        .map(|item| parse_usize(item.trim(), name))
        .collect()
}

fn parse_variants(value: &str) -> Result<Vec<String>, String> {
    let mut variants = Vec::new();
    for item in value.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        if !known_variants().contains(&item) {
            return Err(format!(
                "--variants contains unknown variant {item:?}; expected one of {}",
                known_variants().join(",")
            ));
        }
        if !variants.iter().any(|variant| variant == item) {
            variants.push(item.to_string());
        }
    }
    if variants.is_empty() {
        return Err("--variants must not be empty".to_string());
    }
    Ok(variants)
}

fn known_variants() -> [&'static str; 3] {
    ["dim_parallel", "head_serial", "two_stage_scores"]
}

fn help() -> String {
    "Usage: cargo run -p ax-engine-mlx --release --bin turboquant-microbench -- \\
       [--cold-tokens 512,2048] [--hot-tokens 0] [--n-kv-heads 1] [--head-dim 128] \\
       [--variants dim_parallel,head_serial,two_stage_scores] \\
       [--repetitions 5] [--warmup 1] [--output path.json]"
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_accepts_core_microbench_flags() {
        let config = parse_args(vec![
            "--cold-tokens".into(),
            "128,512".into(),
            "--variants".into(),
            "dim_parallel,two_stage_scores".into(),
            "--hot-tokens".into(),
            "16".into(),
            "--n-kv-heads".into(),
            "2".into(),
            "--repetitions".into(),
            "3".into(),
            "--output".into(),
            "out.json".into(),
        ])
        .expect("arguments should parse");

        assert_eq!(config.cold_tokens, vec![128, 512]);
        assert_eq!(config.hot_tokens, 16);
        assert_eq!(config.variants, vec!["dim_parallel", "two_stage_scores"]);
        assert_eq!(config.n_kv_heads, 2);
        assert_eq!(config.repetitions, 3);
        assert_eq!(config.output, Some(PathBuf::from("out.json")));
    }

    #[test]
    fn median_handles_even_and_odd_samples() {
        assert_eq!(median_u128(&[5, 1, 9]), 5.0);
        assert_eq!(median_u128(&[10, 2, 6, 4]), 5.0);
    }

    #[test]
    fn parse_args_rejects_unknown_variant() {
        let error = parse_args(vec!["--variants".into(), "bogus".into()])
            .expect_err("unknown variant should fail");
        assert!(error.contains("unknown variant"));
    }

    #[test]
    fn deterministic_values_stay_in_expected_range() {
        for token_index in 0..16 {
            let value = deterministic_value(token_index, 1, 127, 31, 17, 13);
            assert!((-1.0..=1.0).contains(&value));
        }
    }

    #[test]
    fn run_case_records_hot_tail_merge_quality_without_metal_variants() {
        let config = Config {
            cold_tokens: vec![8],
            hot_tokens: 2,
            variants: Vec::new(),
            n_kv_heads: 2,
            head_dim: 128,
            block_tokens: 4,
            value_group_size: 32,
            repetitions: 1,
            warmup: 0,
            output: None,
        };

        let row = run_case(&config, 8).expect("microbench row should build");
        let hot_tail_merge = row
            .get("hot_tail_merge")
            .and_then(|value| value.as_object())
            .expect("hot-tail merge evidence should be present");

        assert_eq!(
            hot_tail_merge
                .get("contract")
                .and_then(|value| value.as_str()),
            Some("shared_logsumexp_partition_merge")
        );
        let quality = hot_tail_merge
            .get("quality")
            .and_then(|value| value.as_object())
            .expect("quality should be recorded");
        assert!(
            quality
                .get("max_abs_diff")
                .and_then(|value| value.as_f64())
                .unwrap_or(1.0)
                < 1e-6
        );
    }
}
