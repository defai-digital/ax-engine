//! Whole-step compiled target verifier for dense Qwen3.5 hybrid models.
//!
//! The closure is pure: token ids, RoPE offset, full-attention K/V, and
//! gated-delta conv/recurrent states are explicit inputs. Updated state leaves
//! are explicit outputs and are installed into the speculative cache only
//! after a successful apply. Any unsupported shape, compile failure, or apply
//! error returns `None` so the caller runs the ordinary verifier unchanged.

use std::collections::HashMap;
use std::sync::{Mutex, Once, OnceLock};
use std::thread::ThreadId;

use mlx_sys::{MlxArray, MlxClosure, MlxDtype, MlxVectorArray, astype, reshape, rms_norm};

use super::{ModelConfig, families};
use crate::kv_cache::MlxKVCache;
use crate::per_layer_compile::try_apply_with_abort_safety;
use crate::weights::{LayerWeights, ModelWeights};

type WholeVerifyKey = (u64, i32, ThreadId);
type WholeVerifyCache = Mutex<HashMap<WholeVerifyKey, Option<MlxClosure>>>;
static WHOLE_VERIFY_CACHE: OnceLock<WholeVerifyCache> = OnceLock::new();
static WHOLE_VERIFY_DEBUG_ONCE: Once = Once::new();

type LinearLayerVerifyKey = (u64, usize, i32, ThreadId);
type LinearLayerVerifyCache = Mutex<HashMap<LinearLayerVerifyKey, Option<MlxClosure>>>;
static LINEAR_LAYER_VERIFY_CACHE: OnceLock<LinearLayerVerifyCache> = OnceLock::new();

/// Compile one complete gated-delta verifier layer while preserving the
/// existing paged full-attention route between linear-layer groups.
///
/// The compiled function is pure: hidden/conv/recurrent arrays enter and the
/// updated hidden/state plus the compact rollback tape leave explicitly. The
/// enclosing verifier still owns logical cache length and accept/reject
/// handling.
pub(crate) fn try_compiled_qwen_linear_verify_layer(
    cfg: &ModelConfig,
    weights: &LayerWeights,
    hidden: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
) -> Option<MlxArray> {
    let seq = hidden.shape().get(1).copied()?;
    if !crate::fastpath::mtp_linear_layer_compile_enabled()
        || !crate::fastpath::qwen_linear_mtp_target_verify_enabled()
        || crate::fastpath::qwen_linear_mtp_whole_verify_trace_enabled()
        || !cfg.model_family.eq_ignore_ascii_case("qwen3_5")
        || !(2..=4).contains(&seq)
        || weights.linear_attn.is_none()
        || weights.router_proj.is_some()
        || weights.ffn_post_norm.is_some()
        || cfg.uses_geglu
        || cache.linear_prefix_capture_after().is_none()
    {
        return None;
    }
    let (conv_state, recurrent_state) = cache.linear_state(layer_idx);
    let inputs = [
        hidden.clone(),
        conv_state?.clone(),
        recurrent_state?.clone(),
    ];
    let refs: Vec<&MlxArray> = inputs.iter().collect();
    let key = (
        cfg.compile_cache_identity,
        layer_idx,
        seq,
        std::thread::current().id(),
    );
    let cache_store = LINEAR_LAYER_VERIFY_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache_store.lock().ok()?;
    let outputs = if let Some(entry) = guard.get(&key) {
        try_apply_with_abort_safety(entry.as_ref()?, &refs)
    } else {
        let cfg_addr = cfg as *const ModelConfig as usize;
        let weights_addr = weights as *const LayerWeights as usize;
        let closure = MlxClosure::new_dyn(move |body_inputs: &MlxVectorArray| {
            let _trace = crate::fastpath::scoped_qwen_linear_mtp_whole_verify_trace(true);
            // SAFETY: the body is invoked synchronously by compile while both
            // borrows are live. Replays use the traced MLX graph and do not
            // execute this Rust closure again.
            let cfg = unsafe { &*(cfg_addr as *const ModelConfig) };
            let weights = unsafe { &*(weights_addr as *const LayerWeights) };
            let Some(result) = families::qwen3_linear::layer_forward_verify_functional(
                cfg,
                weights,
                &body_inputs.get(0),
                layer_idx,
                &body_inputs.get(1),
                &body_inputs.get(2),
            ) else {
                return Vec::new();
            };
            vec![
                result.hidden,
                result.state.conv_state,
                result.state.recurrent_state,
                result.state.qkv,
                result.state.a,
                result.state.tape,
            ]
        });
        match closure.compile(false) {
            Ok(compiled) => {
                let result = try_apply_with_abort_safety(&compiled, &refs);
                guard.insert(key, result.is_some().then_some(compiled));
                result
            }
            Err(error) => {
                tracing::warn!(
                    target: "ax_engine_mlx::whole_verify",
                    %error,
                    layer_idx,
                    seq,
                    "linear target-verifier layer compile failed; using imperative fallback",
                );
                guard.insert(key, None);
                None
            }
        }
    }?;
    drop(guard);
    let [hidden_out, conv_out, recurrent_out, qkv, a, tape]: [MlxArray; 6] =
        outputs.try_into().ok()?;
    cache.set_linear_state(layer_idx, conv_out, recurrent_out);
    cache.set_linear_mtp_tape_stash(layer_idx, qkv, a, tape);
    Some(hidden_out)
}

fn qwen_whole_verify_eligible(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &MlxKVCache,
    seq: usize,
) -> bool {
    if !crate::fastpath::mtp_whole_verify_compile_enabled()
        || !crate::fastpath::qwen_linear_mtp_target_verify_enabled()
        || !cfg.model_family.eq_ignore_ascii_case("qwen3_5")
        || !(2..=4).contains(&seq)
        || cfg.linear_attention.is_none()
        || cfg.moe_expert_count != 0
        || cfg.uses_geglu
        || cfg.hidden_size_per_layer_input != 0
        || cfg.global_sliding_window.is_some()
        || cfg.protected_prefix_sliding_window.is_some()
        || cfg.kv_cache_quant.iter().any(Option::is_some)
        || (weights.qwen3_vl_vision.is_some() && cache.mrope_position_delta() != 0)
        || weights.layers.len() != cfg.layer_count
    {
        return false;
    }
    if cfg
        .layer_configs
        .iter()
        .any(|layer| layer.sliding_window.is_some() || layer.kv_source_layer.is_some())
    {
        return false;
    }
    weights.layers.iter().enumerate().all(|(layer_idx, layer)| {
        if layer.router_proj.is_some()
            || layer.ffn_post_norm.is_some()
            || layer.per_layer_gate.is_some()
            || layer.layer_scalar.is_some()
        {
            return false;
        }
        if cfg.is_linear_attention_layer(layer_idx) {
            layer.linear_attn.is_some()
                && matches!(cache.linear_state(layer_idx), (Some(_), Some(_)))
        } else {
            layer.linear_attn.is_none() && cache.logical_layer_kv(layer_idx).is_some()
        }
    })
}

/// Trace the pure whole-model body. Output schema is logits, post-norm hidden,
/// then layer-ordered state: final conv/recurrent plus QKV/A/B per linear
/// layer and fixed-capacity K/V per full-attention layer.
fn qwen_whole_verify_body(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    inputs: &MlxVectorArray,
    seq: usize,
) -> Option<Vec<MlxArray>> {
    let ids = inputs.get(0);
    let rope_offset = inputs.get(1);
    let mut input_idx = 2;
    let mut hidden = super::embed_tokens_arr(&ids, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    hidden = super::shared::utils::maybe_weightless_embed_norm(cfg, hidden);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = super::shared::utils::scale_hidden(&hidden, scale);
    }

    // Build trace-time state from explicit inputs only. `mx::compile` retains
    // its source callable for possible retraces; a cache captured by that
    // callable would pin the first dispatch's leaves and permanently block
    // buffer donation.
    let mut shadow_cache = MlxKVCache::new(weights.layers.len());
    shadow_cache.begin_linear_prefix_capture(1);
    let mut state_outputs = Vec::with_capacity(weights.layers.len() * 7);
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        if cfg.is_linear_attention_layer(layer_idx) {
            let conv_state = inputs.get(input_idx);
            let recurrent_state = inputs.get(input_idx + 1);
            input_idx += 2;
            shadow_cache.set_linear_state(layer_idx, conv_state, recurrent_state);
            hidden = families::qwen3_linear::layer_forward(
                cfg,
                layer,
                &hidden,
                &mut shadow_cache,
                layer_idx,
                false,
                false,
            );
            let (conv_out, recurrent_out) = shadow_cache.linear_state(layer_idx);
            state_outputs.push(conv_out?.clone());
            state_outputs.push(recurrent_out?.clone());
            let (qkv, a, b) = shadow_cache.linear_mtp_projection_stash(layer_idx)?;
            state_outputs.push(qkv);
            state_outputs.push(a);
            state_outputs.push(b);
        } else {
            let cached_k = inputs.get(input_idx);
            let cached_v = inputs.get(input_idx + 1);
            let rope_cos = inputs.get(input_idx + 2);
            let rope_sin = inputs.get(input_idx + 3);
            input_idx += 4;
            let result = families::standard::layer_forward_verify_functional(
                cfg,
                layer,
                &hidden,
                layer_idx,
                &rope_offset,
                &cached_k,
                &cached_v,
                &rope_cos,
                &rope_sin,
            )?;
            hidden = result.hidden;
            state_outputs.push(result.k);
            state_outputs.push(result.v);
        }
    }

    let seq_i32 = i32::try_from(seq).ok()?;
    let normed = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);
    let logits = super::lm_head_verify_window_projection(
        &normed,
        &weights.lm_head,
        seq_i32,
        cfg.hidden_size as i32,
    );
    let logits = astype(&logits, MlxDtype::Float32, None);
    let logits = super::shared::utils::apply_final_logit_softcap(cfg, &logits);
    let logits = reshape(&logits, &[seq_i32, cfg.vocab_size as i32], None);
    let mut outputs = Vec::with_capacity(2 + state_outputs.len());
    outputs.push(logits);
    outputs.push(normed);
    outputs.extend(state_outputs);
    Some(outputs)
}

fn collect_inputs(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    ids: &MlxArray,
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> Option<Vec<MlxArray>> {
    let offset_i32 = i32::try_from(token_offset).ok()?;
    let seq = ids.shape().first().copied()?;
    let offset = MlxArray::from_raw_data(
        &offset_i32 as *const i32 as *const u8,
        std::mem::size_of::<i32>(),
        &[1],
        MlxDtype::Int32,
    );
    let mut inputs = Vec::with_capacity(2 + weights.layers.len() * 2);
    inputs.push(ids.clone());
    inputs.push(offset);
    // Reserve enough stable backing space for a normal generation so the
    // fixed-shape compiled callable does not retrace in the measured window.
    const RESERVE_TOKENS: usize = 262;
    let min_capacity = token_offset.checked_add(RESERVE_TOKENS)?;
    for layer_idx in 0..weights.layers.len() {
        if cfg.is_linear_attention_layer(layer_idx) {
            let (conv, recurrent) = cache.linear_state(layer_idx);
            inputs.push(conv?.clone());
            inputs.push(recurrent?.clone());
        } else {
            let (k, v) = cache.prepare_whole_verify_layer_kv(layer_idx, min_capacity)?;
            inputs.push(k);
            inputs.push(v);
            let (_, rope_theta, rope_dims, layer_rope_freqs, _, _, _) =
                super::config::layer_params(cfg, layer_idx);
            let rope_freqs = layer_rope_freqs.or(cfg.rope_freqs.as_ref());
            let (rope_base, rope_freqs_ref) = rope_freqs
                .map(|freqs| (None, Some(freqs)))
                .unwrap_or((Some(rope_theta), None));
            let (cos, sin) = super::shared::build_neox_rope_cos_sin(
                offset_i32,
                seq,
                rope_dims as i32,
                rope_base,
                rope_freqs_ref,
            );
            inputs.push(cos);
            inputs.push(sin);
        }
    }
    Some(inputs)
}

fn install_outputs(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    mut outputs: Vec<MlxArray>,
) -> Option<(MlxArray, MlxArray)> {
    let expected = 2
        + (0..weights.layers.len())
            .map(|layer_idx| {
                if cfg.is_linear_attention_layer(layer_idx) {
                    5
                } else {
                    2
                }
            })
            .sum::<usize>();
    if outputs.len() != expected || cache.linear_prefix_capture_after() != Some(1) {
        return None;
    }
    let logits = outputs.remove(0);
    let post_norm = outputs.remove(0);
    let mut output_idx = 0;
    for layer_idx in 0..weights.layers.len() {
        if cfg.is_linear_attention_layer(layer_idx) {
            cache.set_linear_state(
                layer_idx,
                outputs[output_idx].clone(),
                outputs[output_idx + 1].clone(),
            );
            cache.set_linear_mtp_projection_stash(
                layer_idx,
                outputs[output_idx + 2].clone(),
                outputs[output_idx + 3].clone(),
                outputs[output_idx + 4].clone(),
            );
            output_idx += 5;
        } else {
            cache.replace_layer_kv(
                layer_idx,
                outputs[output_idx].clone(),
                outputs[output_idx + 1].clone(),
            );
            output_idx += 2;
        }
    }
    Some((logits, post_norm))
}

/// Try a pure compiled whole target-verifier step and install its explicit
/// state outputs into `cache`. The caller retains ownership of logical cache
/// length and advances it exactly as on the imperative path.
pub(crate) fn try_whole_compiled_qwen_verify(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    ids: &MlxArray,
    seq: usize,
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> Option<(MlxArray, MlxArray)> {
    let eligible = qwen_whole_verify_eligible(cfg, weights, cache, seq);
    if seq >= 2 && std::env::var("AX_MLX_MTP_WHOLE_VERIFY_DEBUG").is_ok() {
        WHOLE_VERIFY_DEBUG_ONCE.call_once(|| {
            let bad_layer = weights.layers.iter().enumerate().find_map(|(idx, layer)| {
                let common_bad = layer.router_proj.is_some()
                    || layer.ffn_post_norm.is_some()
                    || layer.per_layer_gate.is_some()
                    || layer.layer_scalar.is_some();
                let state_bad = if cfg.is_linear_attention_layer(idx) {
                    layer.linear_attn.is_none()
                        || !matches!(cache.linear_state(idx), (Some(_), Some(_)))
                } else {
                    layer.linear_attn.is_some() || cache.logical_layer_kv(idx).is_none()
                };
                (common_bad || state_bad).then_some(idx)
            });
            eprintln!(
                "AX_WHOLE_VERIFY_DEBUG eligible={eligible} flag={} target={} family={} seq={seq} linear={} moe={} geglu={} pli={} global_swa={} protected_swa={} kv_quant={} vision={} layers={}/{} layer_cfg_bad={} bad_layer={bad_layer:?}",
                crate::fastpath::mtp_whole_verify_compile_enabled(),
                crate::fastpath::qwen_linear_mtp_target_verify_enabled(),
                cfg.model_family,
                cfg.linear_attention.is_some(),
                cfg.moe_expert_count,
                cfg.uses_geglu,
                cfg.hidden_size_per_layer_input,
                cfg.global_sliding_window.is_some(),
                cfg.protected_prefix_sliding_window.is_some(),
                cfg.kv_cache_quant.iter().any(Option::is_some),
                weights.qwen3_vl_vision.is_some() && cache.mrope_position_delta() != 0,
                weights.layers.len(),
                cfg.layer_count,
                cfg.layer_configs.iter().any(|layer| {
                    layer.sliding_window.is_some() || layer.kv_source_layer.is_some()
                }),
            );
        });
    }
    if !eligible {
        return None;
    }
    let inputs = collect_inputs(cfg, weights, ids, cache, token_offset)?;
    let refs: Vec<&MlxArray> = inputs.iter().collect();
    // Let the compiled verifier consume pending state producers directly.
    // A caller-side evaluation fence can serialize the accepted-cache graph
    // with the next verify call and defeats the scheduling benefit of a
    // whole-step trace. The ordinary acceptance fence still materializes all
    // value-bearing outputs before any host read.
    let key = (
        cfg.compile_cache_identity,
        i32::try_from(seq).ok()?,
        std::thread::current().id(),
    );
    let cache_store = WHOLE_VERIFY_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache_store.lock().ok()?;
    let outputs = if let Some(entry) = guard.get(&key) {
        let closure = entry.as_ref()?;
        try_apply_with_abort_safety(closure, &refs)
    } else {
        let cfg_addr = cfg as *const ModelConfig as usize;
        let weights_addr = weights as *const ModelWeights as usize;
        let closure = MlxClosure::new_dyn(move |body_inputs: &MlxVectorArray| {
            let _trace = crate::fastpath::scoped_qwen_linear_mtp_whole_verify_trace(true);
            // SAFETY: compilation happens synchronously while the caller holds
            // both borrows. The compiled graph owns MLX primitives, not these
            // Rust pointers; apply does not re-run the body.
            let cfg = unsafe { &*(cfg_addr as *const ModelConfig) };
            let weights = unsafe { &*(weights_addr as *const ModelWeights) };
            qwen_whole_verify_body(cfg, weights, body_inputs, seq).unwrap_or_default()
        });
        // Every state leaf is fixed-shape: K/V capacity stays constant and
        // logical position is an explicit tensor. Shape-specialized compile
        // avoids shapeless Slice limitations and gives MLX maximum fusion.
        match closure.compile(false) {
            Ok(compiled) => {
                let result = try_apply_with_abort_safety(&compiled, &refs);
                guard.insert(key, result.is_some().then_some(compiled));
                result
            }
            Err(error) => {
                tracing::warn!(
                    target: "ax_engine_mlx::whole_verify",
                    %error,
                    seq,
                    "whole target-verifier compile failed; using imperative fallback",
                );
                guard.insert(key, None);
                None
            }
        }
    }?;
    drop(guard);
    install_outputs(cfg, weights, cache, outputs)
}

pub(crate) fn clear_whole_verify_compile_cache() {
    if let Some(cache) = WHOLE_VERIFY_CACHE.get()
        && let Ok(mut guard) = cache.lock()
    {
        guard.clear();
    }
    if let Some(cache) = LINEAR_LAYER_VERIFY_CACHE.get()
        && let Ok(mut guard) = cache.lock()
    {
        guard.clear();
    }
}
