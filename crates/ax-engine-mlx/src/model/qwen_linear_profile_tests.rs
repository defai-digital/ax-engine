//! Real-pack timing of the Qwen linear-attention family's prefill.
//!
//! Timing only: nothing here asserts a number. Each figure is the fastest of
//! a few repeats of one synchronous evaluation, the way a peer runtime's
//! modules are timed in isolation, so the two can be compared block by block.

use super::shared::linear_attention::{
    linear_attention_core, linear_attention_gate_out, linear_attention_inputs,
};
use super::shared::{ffn_swiglu, linear_attention_forward};
use super::*;
use ax_engine_core::NativeModelArtifacts;
use mlx_sys::{MlxDtype, astype, eval, rms_norm};
use std::path::PathBuf;
use std::time::Instant;

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse().ok())
}

/// A deterministic `[1, rows, width]` activation with values in ±0.5.
fn activation(rows: usize, width: usize, phase: usize, dtype: MlxDtype) -> MlxArray {
    let len = rows * width;
    let data: Vec<f32> = (0..len)
        .map(|i| (((i * 7 + phase) % 1021) as f32 / 1021.0) - 0.5)
        .collect();
    let array = MlxArray::from_raw_data(
        data.as_ptr().cast(),
        len * 4,
        &[1, rows as i32, width as i32],
        MlxDtype::Float32,
    );
    let array = astype(&array, dtype, None);
    eval(&[&array]);
    array
}

/// Fastest of `repeats` runs after one untimed run, in milliseconds.
fn best_ms(repeats: usize, mut run: impl FnMut()) -> f64 {
    run();
    (0..repeats)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::INFINITY, f64::min)
}

fn eval_with_cache(out: &MlxArray, cache: &MlxKVCache) {
    let mut refs = cache.collect_eval_refs();
    refs.push(out);
    eval(&refs);
}

/// `AX_QWEN_PROFILE_PACK` names the pack. `AX_QWEN_PROFILE_IDS_FILE` (a JSON
/// list of token ids) adds whole-forward timings in chunks of
/// `AX_QWEN_PROFILE_CHUNK` rows (default: the whole prompt but its last row),
/// from an emptied buffer cache with `AX_QWEN_PROFILE_CLEAR_CACHE=1`.
/// `AX_QWEN_PROFILE_ROWS` sets the width of the module timings (default
/// 1,496).
#[test]
#[ignore = "requires AX_QWEN_PROFILE_PACK; timing only"]
fn qwen_linear_prefill_module_latency() {
    let root = PathBuf::from(std::env::var_os("AX_QWEN_PROFILE_PACK").unwrap());
    let artifacts = NativeModelArtifacts::from_dir(root).unwrap();
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = crate::weights::load_weights(&artifacts).unwrap();
    let layers = cfg.layer_count;
    let repeats = env_usize("AX_QWEN_PROFILE_REPEATS").unwrap_or(6).max(1);

    if let Ok(path) = std::env::var("AX_QWEN_PROFILE_IDS_FILE") {
        let ids: Vec<u32> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let body = &ids[..ids.len() - 1];
        let chunk = env_usize("AX_QWEN_PROFILE_CHUNK")
            .unwrap_or(body.len())
            .max(1);
        // A served request starts from the allocator state the previous
        // request left: `AX_QWEN_PROFILE_CLEAR_CACHE=1` empties MLX's buffer
        // cache before each run, the way the serving prefill does after it.
        let clear = env_usize("AX_QWEN_PROFILE_CLEAR_CACHE") == Some(1);
        for run in 0..repeats {
            if clear {
                mlx_sys::clear_cache();
            }
            let mut cache = MlxKVCache::new(layers);
            let start = Instant::now();
            let mut offset = 0;
            for rows in body.chunks(chunk) {
                let hidden = forward_cache_only(&cfg, &weights, rows, &mut cache, offset);
                eval_with_cache(&hidden, &cache);
                offset += rows.len();
            }
            let prefill = start.elapsed().as_secs_f64() * 1e3;
            let start = Instant::now();
            let logits = forward(&cfg, &weights, &ids[ids.len() - 1..], &mut cache, offset);
            eval_with_cache(&logits, &cache);
            let last = start.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "qwen_prefill rows={} chunk={chunk} run={run} prefill_ms={prefill:.1} last_row_ms={last:.1}",
                body.len()
            );
        }
        families::qwen3_linear::clear_qwen_prefill_pending_ffn();
    }

    let rows = env_usize("AX_QWEN_PROFILE_ROWS").unwrap_or(1496);
    let seq = rows as i32;
    let dtype = weights.layers[0].attn_norm.dtype();
    let hidden = activation(rows, cfg.hidden_size, 3, dtype);
    let linear_index = (0..layers)
        .find(|&i| cfg.is_linear_attention_layer(i))
        .unwrap();
    let full_index = (0..layers)
        .find(|&i| !cfg.is_linear_attention_layer(i))
        .unwrap();
    let linear_layers = (0..layers)
        .filter(|&i| cfg.is_linear_attention_layer(i))
        .count();
    let w = &weights.layers[linear_index];
    let linear_cfg = cfg.linear_attention.as_ref().unwrap();
    let linear_w = w.linear_attn.as_ref().unwrap();
    let mut report = Vec::new();

    let block = best_ms(repeats, || {
        let mut cache = MlxKVCache::new(layers);
        let out =
            linear_attention_forward(&cfg, w, &hidden, &mut cache, linear_index, false, false);
        eval_with_cache(&out, &cache);
    });
    report.push(("linear-attention block", block, linear_layers));

    let inputs = || linear_attention_inputs(&cfg, linear_cfg, linear_w, &hidden, seq, false);
    let projections = best_ms(repeats, || {
        let (qkv, z, a, b) = inputs();
        eval(&[&qkv, &z, &a, &b]);
    });
    report.push(("  input projections", projections, linear_layers));
    let (qkv, z, a, b) = inputs();
    eval(&[&qkv, &z, &a, &b]);
    let core = |cache: &mut MlxKVCache| {
        linear_attention_core(
            &cfg,
            linear_cfg,
            linear_w,
            &qkv,
            &a,
            &b,
            cache,
            linear_index,
            false,
        )
    };
    let recurrence = best_ms(repeats, || {
        let mut cache = MlxKVCache::new(layers);
        let out = core(&mut cache);
        eval_with_cache(&out, &cache);
    });
    report.push(("  conv, norm and recurrence", recurrence, linear_layers));
    let mut cache = MlxKVCache::new(layers);
    let core_out = core(&mut cache);
    eval_with_cache(&core_out, &cache);
    let value_dim = linear_cfg.value_dim() as i32;
    let output = best_ms(repeats, || {
        let out = linear_attention_gate_out(
            &cfg,
            linear_w,
            &core_out,
            &z,
            seq,
            value_dim,
            linear_index,
            false,
        );
        eval(&[&out]);
    });
    report.push(("  output norm, gate and projection", output, linear_layers));

    let ffn = best_ms(repeats, || {
        let out = ffn_swiglu(&cfg, w, &hidden, None, linear_index);
        eval(&[&out]);
    });
    report.push(("feed-forward block", ffn, layers));

    let norm = best_ms(repeats, || {
        let out = rms_norm(&hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);
        eval(&[&out]);
    });
    report.push(("RMS norm", norm, layers * 2));

    let full_w = &weights.layers[full_index];
    let full_layer = best_ms(repeats, || {
        let mut cache = MlxKVCache::new(layers);
        let out =
            layer_forward_uncaptured(&cfg, full_w, &hidden, &mut cache, full_index, 0, None, None);
        eval_with_cache(&out, &cache);
    });
    report.push((
        "full-attention layer with its feed-forward",
        full_layer,
        layers - linear_layers,
    ));
    families::qwen3_linear::clear_qwen_prefill_pending_ffn();

    for (name, ms, count) in report {
        eprintln!(
            "qwen_module rows={rows} {name:44} {ms:8.3} ms x {count:3} = {:8.1} ms",
            ms * count as f64
        );
    }
}

/// Writes the next-token logits of real prompts, one `<index>.f32` file of
/// little-endian `f32` per prompt, so two processes that differ in one switch
/// can be compared logit by logit.
///
/// `AX_QWEN_LOGITS_IDS_FILE` is a JSON list of token-id lists and
/// `AX_QWEN_LOGITS_OUT` the output directory. Each prompt runs the schedule
/// of an uncached serving prefill: chunks of `AX_QWEN_PROFILE_CHUNK` rows
/// (default 1,024), the last one producing the logits. With
/// `AX_QWEN_LOGITS_SPLIT=1` the prompt instead stops at its last 16-token
/// boundary the way the boundary-snapshot schedule does: all rows before the
/// boundary but one, that row alone, then the rest. With
/// `AX_QWEN_LOGITS_LAST_ROW_STEP=1` the last prompt row runs as a forward of
/// its own after the chunks, the schedule of families that keep the
/// reference n-1 split. Tests keep the tensor-unit QSA kernel of Flash Next
/// off; `AX_QWEN_LOGITS_TENSOR_UNIT=1` runs it as serving does on a Neural
/// Accelerator host.
#[test]
#[ignore = "requires AX_QWEN_PROFILE_PACK, AX_QWEN_LOGITS_IDS_FILE and AX_QWEN_LOGITS_OUT"]
fn qwen_linear_prefill_logits_dump() {
    let root = PathBuf::from(std::env::var_os("AX_QWEN_PROFILE_PACK").unwrap());
    let artifacts = NativeModelArtifacts::from_dir(root).unwrap();
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = crate::weights::load_weights(&artifacts).unwrap();
    let prompts: Vec<Vec<u32>> = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("AX_QWEN_LOGITS_IDS_FILE").unwrap()).unwrap(),
    )
    .unwrap();
    let out = PathBuf::from(std::env::var_os("AX_QWEN_LOGITS_OUT").unwrap());
    std::fs::create_dir_all(&out).unwrap();
    let chunk = env_usize("AX_QWEN_PROFILE_CHUNK").unwrap_or(1024).max(1);
    let split = env_usize("AX_QWEN_LOGITS_SPLIT") == Some(1);
    let last_row_step = env_usize("AX_QWEN_LOGITS_LAST_ROW_STEP") == Some(1);
    let tensor_unit = env_usize("AX_QWEN_LOGITS_TENSOR_UNIT") == Some(1);
    assert!(
        !tensor_unit
            || crate::model::shared::qwen4_exp_attention_indexed::tensor_unit_kernel_verified(),
        "AX_QWEN_LOGITS_TENSOR_UNIT needs a host where the kernel passes its self-test"
    );

    let dump = || {
        for (index, ids) in prompts.iter().enumerate() {
            dump_prompt_logits(
                &cfg,
                &weights,
                ids,
                &out.join(format!("{index}.f32")),
                chunk,
                split,
                last_row_step,
            );
            eprintln!(
                "qwen_logits prompt={index} tokens={} split={split} last_row_step={last_row_step} \
                 tensor_unit={tensor_unit}",
                ids.len()
            );
        }
    };
    if tensor_unit {
        crate::model::shared::qwen4_exp_attention_indexed::with_tensor_unit_attention(true, dump);
    } else {
        dump();
    }
}

fn dump_prompt_logits(
    cfg: &ModelConfig,
    weights: &crate::weights::ModelWeights,
    ids: &[u32],
    path: &std::path::Path,
    chunk: usize,
    split: bool,
    last_row_step: bool,
) {
    mlx_sys::clear_cache();
    let mut cache = MlxKVCache::new(cfg.layer_count);
    let cache_only = |rows: &[u32], cache: &mut MlxKVCache| {
        for part in rows.chunks(chunk) {
            let offset = cache.seq_len();
            let hidden = forward_cache_only(cfg, weights, part, cache, offset);
            cache.advance(part.len());
            eval_with_cache(&hidden, cache);
        }
    };
    let boundary = ids.len() / 16 * 16;
    let tail_start = if split && boundary > 1 && boundary < ids.len() {
        cache_only(&ids[..boundary - 1], &mut cache);
        cache_only(&ids[boundary - 1..boundary], &mut cache);
        boundary
    } else if last_row_step && ids.len() > 1 {
        cache_only(&ids[..ids.len() - 1], &mut cache);
        ids.len() - 1
    } else {
        let last_chunk = (ids.len() - 1) / chunk * chunk;
        cache_only(&ids[..last_chunk], &mut cache);
        last_chunk
    };
    let offset = cache.seq_len();
    let logits = forward(cfg, weights, &ids[tail_start..], &mut cache, offset);
    let logits = astype(&logits, MlxDtype::Float32, None);
    eval_with_cache(&logits, &cache);
    let bytes: Vec<u8> = logits
        .data_f32()
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    std::fs::write(path, bytes).unwrap();
    families::qwen3_linear::clear_qwen_prefill_pending_ffn();
}
