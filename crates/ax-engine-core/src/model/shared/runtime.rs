pub(super) fn q5k_prefill_enabled() -> bool {
    // Q5_K GPU prefill is a normal supported path now. The remaining env
    // surface only selects the routing variant for validation.
    true
}

pub(super) fn env_flag_enabled(var: &str) -> bool {
    std::env::var(var)
        .ok()
        .is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on"))
}

pub(super) fn env_flag_override(var: &str) -> Option<bool> {
    match std::env::var(var) {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" => Some(true),
            "0" | "false" | "off" => Some(false),
            _ => None,
        },
        Err(_) => None,
    }
}

pub(super) fn decode_fused_gelu_down_enabled() -> bool {
    match std::env::var("AX_METAL_DECODE_FUSED_GELU_DOWN") {
        Ok(v) => matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on"),
        Err(_) => true,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Q5KPrefillVariantOverride {
    Auto,
    Base,
    Small,
}

pub(super) fn q5k_prefill_variant_override() -> Q5KPrefillVariantOverride {
    let raw = std::env::var("AX_METAL_Q5K_PREFILL_VARIANT")
        .or_else(|_| std::env::var("AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT"));
    match raw {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "base" => Q5KPrefillVariantOverride::Base,
            "small" => Q5KPrefillVariantOverride::Small,
            _ => Q5KPrefillVariantOverride::Auto,
        },
        Err(_) => Q5KPrefillVariantOverride::Auto,
    }
}

fn gpu_decode_quant_dtype_supported(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::Q4_0
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
            | GgmlType::Q8_0
            | GgmlType::F32
    )
}

fn gpu_prefill_quant_dtype_supported(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::Q4_0 | GgmlType::Q4K | GgmlType::Q6K | GgmlType::Q8_0 | GgmlType::F32
    ) || (dtype == GgmlType::Q5K && q5k_prefill_enabled())
}

fn gpu_batch_logits_dtype_supported(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::Q4_0 | GgmlType::Q4K | GgmlType::Q6K | GgmlType::Q8_0
    )
}

/// Check if all layer-0 weight tensors use quant types supported by decode-only GPU path.
///
/// Decode can use additional fused matvec kernels (such as Q8_0) that are not yet available for
/// batch-prefill kernels.
pub(super) fn gpu_decode_quant_supported(weights: &WeightStore) -> bool {
    all_layers_match(weights, LAYER_SUFFIXES, gpu_decode_quant_dtype_supported)
}

pub(super) fn gpu_prefill_quant_blocker(weights: &WeightStore) -> Option<String> {
    first_layer_mismatch(weights, LAYER_SUFFIXES, gpu_prefill_quant_dtype_supported)
}

pub(super) fn gpu_prefill_uses_q5k(weights: &WeightStore) -> bool {
    q5k_prefill_enabled()
        && any_layers_match(weights, LAYER_SUFFIXES, |dtype| dtype == GgmlType::Q5K)
}

pub(super) fn gpu_prefill_q5k_small_n_auto_eligible(weights: &WeightStore) -> bool {
    q5k_prefill_small_n_auto_eligible_for_model(
        weights.predominant_quant(),
        any_layers_match(weights, LAYER_SUFFIXES, |dtype| dtype == GgmlType::Q5K),
    )
}

fn q5k_prefill_small_n_auto_eligible_for_model(
    predominant_quant: Option<GgmlType>,
    has_q5k_layer_weights: bool,
) -> bool {
    has_q5k_layer_weights && predominant_quant == Some(GgmlType::Q5K)
}

/// Return true when a quantized LM head can use the existing batched GPU matmul path.
pub(super) fn gpu_batch_logits_supported(dtype: GgmlType) -> bool {
    gpu_batch_logits_dtype_supported(dtype)
}

#[derive(Clone, Copy)]
pub(super) struct AttentionQkNormWeights<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
}

pub(super) fn lm_head_weight_name(weights: &WeightStore) -> &'static str {
    if weights.has("output.weight") {
        "output.weight"
    } else {
        "token_embd.weight"
    }
}

pub(super) fn lm_head_raw_with_dtype<'a>(
    weights: &'a WeightStore,
) -> anyhow::Result<(&'a [u8], GgmlType)> {
    weights.raw_with_dtype(lm_head_weight_name(weights))
}

pub(super) fn cache_output_head_keys<'a>(
    metal_ops: &MetalOps,
    weights: &'a WeightStore,
) -> anyhow::Result<(usize, &'a [u8], GgmlType, usize)> {
    let final_norm_w = weights.f32_slice("output_norm.weight")?;
    let output_norm_key = metal_ops.ensure_f32_cached(final_norm_w);
    let (lm_raw, lm_dtype) = lm_head_raw_with_dtype(weights)?;
    let lm_head_key = metal_ops.ensure_quant_cached(lm_raw);
    Ok((output_norm_key, lm_raw, lm_dtype, lm_head_key))
}

pub(super) fn cache_attention_qk_norm_keys(
    metal_ops: &MetalOps,
    weights: &WeightStore,
    prefix: &str,
) -> anyhow::Result<(Option<usize>, Option<usize>)> {
    let Some(norm_weights) = maybe_attention_qk_norm_weights(weights, prefix)? else {
        return Ok((None, None));
    };
    Ok((
        Some(metal_ops.ensure_f32_cached(norm_weights.q)),
        Some(metal_ops.ensure_f32_cached(norm_weights.k)),
    ))
}

pub(super) fn cache_optional_f32_key(
    metal_ops: &MetalOps,
    weights: &WeightStore,
    name: &str,
) -> anyhow::Result<Option<usize>> {
    if !weights.has(name) {
        return Ok(None);
    }
    let w = weights.f32_slice(name)?;
    Ok(Some(metal_ops.ensure_f32_cached(w)))
}

pub(super) fn cache_optional_prefixed_f32_key(
    metal_ops: &MetalOps,
    weights: &WeightStore,
    prefix: &str,
    suffix: &str,
) -> anyhow::Result<Option<usize>> {
    cache_optional_f32_key(metal_ops, weights, &format!("{prefix}.{suffix}"))
}

pub(super) fn ensure_precomputed_lm_head_f16(
    metal_ops: &MetalOps,
    raw: &[u8],
    dtype: GgmlType,
    vocab_size: u32,
    embedding_dim: u32,
) -> anyhow::Result<()> {
    match dtype {
        GgmlType::Q4K => {
            metal_ops.ensure_precomputed_q4k_f16_from_raw(raw, vocab_size, embedding_dim)?
        }
        GgmlType::Q6K => {
            metal_ops.ensure_precomputed_q6k_f16_from_raw(raw, vocab_size, embedding_dim)?
        }
        GgmlType::Q8_0 => {
            metal_ops.ensure_precomputed_q8_0_f16_from_raw(raw, vocab_size, embedding_dim)?
        }
        _ => {}
    }
    Ok(())
}

pub(super) fn ensure_precomputed_linear_f16(
    metal_ops: &MetalOps,
    raw: &[u8],
    dtype: GgmlType,
    m: u32,
    k: u32,
) -> anyhow::Result<()> {
    match dtype {
        GgmlType::Q4K => metal_ops.ensure_precomputed_q4k_f16_from_raw(raw, m, k)?,
        GgmlType::Q6K => metal_ops.ensure_precomputed_q6k_f16_from_raw(raw, m, k)?,
        GgmlType::Q8_0 => metal_ops.ensure_precomputed_q8_0_f16_from_raw(raw, m, k)?,
        _ => {}
    }
    Ok(())
}

pub(super) fn ensure_precomputed_linear_f16_many(
    metal_ops: &MetalOps,
    ops: &[(&[u8], GgmlType, u32, u32)],
) -> anyhow::Result<()> {
    for &(raw, dtype, m, k) in ops {
        ensure_precomputed_linear_f16(metal_ops, raw, dtype, m, k)?;
    }
    Ok(())
}

pub(super) fn write_normalized_single_logits(
    backend: &dyn Backend,
    hidden: &[f32],
    dim: usize,
    vocab_size: usize,
    weights: &WeightStore,
    logits: &mut [f32],
) -> anyhow::Result<()> {
    let (lm_raw, lm_dtype) = lm_head_raw_with_dtype(weights)?;
    anyhow::ensure!(logits.len() >= vocab_size, "logits buffer too small");
    backend.dequant_matmul(lm_raw, lm_dtype, hidden, logits, vocab_size, 1, dim);
    Ok(())
}

pub(super) fn write_normalized_single_logits_with_breakdown(
    backend: &dyn Backend,
    hidden: &[f32],
    dim: usize,
    vocab_size: usize,
    weights: &WeightStore,
    logits: &mut [f32],
    mut ops: Option<&mut OpBreakdown>,
) -> anyhow::Result<()> {
    if let Some(ref mut ops_ref) = ops {
        let t = OpTimer::start();
        write_normalized_single_logits(backend, hidden, dim, vocab_size, weights, logits)?;
        ops_ref.matmul += t.elapsed();
    } else {
        write_normalized_single_logits(backend, hidden, dim, vocab_size, weights, logits)?;
    }
    Ok(())
}

pub(super) fn apply_attention_norm_single(
    weights: &WeightStore,
    prefix: &str,
    hidden: &[f32],
    norm_buf: &mut [f32],
    rms_norm_eps: f32,
    mut ops: Option<&mut OpBreakdown>,
) -> anyhow::Result<()> {
    let attn_norm_w = if let Some(ref mut ops_ref) = ops {
        let t = OpTimer::start();
        let weights = weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?;
        ops_ref.dequant += t.elapsed();
        weights
    } else {
        weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?
    };

    if let Some(ref mut ops_ref) = ops {
        let t = OpTimer::start();
        rms_norm::rms_norm_out(hidden, attn_norm_w, norm_buf, rms_norm_eps);
        ops_ref.norm += t.elapsed();
    } else {
        rms_norm::rms_norm_out(hidden, attn_norm_w, norm_buf, rms_norm_eps);
    }
    Ok(())
}

pub(super) fn maybe_attention_qk_norm_weights<'a>(
    weights: &'a WeightStore,
    prefix: &str,
) -> anyhow::Result<Option<AttentionQkNormWeights<'a>>> {
    if !weights.has(&format!("{prefix}.attn_q_norm.weight")) {
        return Ok(None);
    }
    Ok(Some(AttentionQkNormWeights {
        q: weights.f32_slice(&format!("{prefix}.attn_q_norm.weight"))?,
        k: weights.f32_slice(&format!("{prefix}.attn_k_norm.weight"))?,
    }))
}

pub(super) fn apply_attention_qk_norm(
    q: &mut [f32],
    k: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    norm_weights: AttentionQkNormWeights<'_>,
    rms_norm_eps: f32,
) {
    per_head_rms_norm(q, n_heads, head_dim, norm_weights.q, rms_norm_eps);
    per_head_rms_norm(k, n_kv_heads, head_dim, norm_weights.k, rms_norm_eps);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn apply_optional_attention_qk_norm_single(
    weights: &WeightStore,
    prefix: &str,
    q: &mut [f32],
    k: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    mut ops: Option<&mut OpBreakdown>,
) -> anyhow::Result<()> {
    let norm_weights = if let Some(ref mut ops_ref) = ops {
        let t = OpTimer::start();
        let weights = maybe_attention_qk_norm_weights(weights, prefix)?;
        ops_ref.dequant += t.elapsed();
        weights
    } else {
        maybe_attention_qk_norm_weights(weights, prefix)?
    };
    let Some(norm_weights) = norm_weights else {
        return Ok(());
    };

    if let Some(ref mut ops_ref) = ops {
        let t = OpTimer::start();
        apply_attention_qk_norm(
            q,
            k,
            n_heads,
            n_kv_heads,
            head_dim,
            norm_weights,
            rms_norm_eps,
        );
        ops_ref.norm += t.elapsed();
    } else {
        apply_attention_qk_norm(
            q,
            k,
            n_heads,
            n_kv_heads,
            head_dim,
            norm_weights,
            rms_norm_eps,
        );
    }
    Ok(())
}

pub(super) fn apply_output_norm_single(
    weights: &WeightStore,
    hidden: &mut [f32],
    rms_norm_eps: f32,
    mut ops: Option<&mut OpBreakdown>,
) -> anyhow::Result<()> {
    let final_norm_w = if let Some(ref mut ops_ref) = ops {
        let t = OpTimer::start();
        let weights = weights.f32_slice("output_norm.weight")?;
        ops_ref.dequant += t.elapsed();
        weights
    } else {
        weights.f32_slice("output_norm.weight")?
    };

    if let Some(ref mut ops_ref) = ops {
        let t = OpTimer::start();
        rms_norm::rms_norm(hidden, final_norm_w, rms_norm_eps);
        ops_ref.norm += t.elapsed();
    } else {
        rms_norm::rms_norm(hidden, final_norm_w, rms_norm_eps);
    }
    Ok(())
}

fn all_layers_match(
    weights: &WeightStore,
    layer_suffixes: &[&str],
    is_supported: impl Fn(GgmlType) -> bool,
) -> bool {
    first_layer_mismatch(weights, layer_suffixes, is_supported).is_none()
}

fn any_layers_match(
    weights: &WeightStore,
    layer_suffixes: &[&str],
    predicate: impl Fn(GgmlType) -> bool,
) -> bool {
    for layer in 0usize.. {
        let probe = format!("blk.{layer}.{}", layer_suffixes[0]);
        if !weights.has(&probe) {
            break;
        }

        for suffix in layer_suffixes {
            let name = format!("blk.{layer}.{suffix}");
            if let Ok((_, dtype)) = weights.raw_with_dtype(&name)
                && predicate(dtype)
            {
                return true;
            }
        }
    }
    false
}

fn first_layer_mismatch(
    weights: &WeightStore,
    layer_suffixes: &[&str],
    is_supported: impl Fn(GgmlType) -> bool,
) -> Option<String> {
    for layer in 0usize.. {
        let probe = format!("blk.{layer}.{}", layer_suffixes[0]);
        if !weights.has(&probe) {
            break;
        }

        for suffix in layer_suffixes {
            let name = format!("blk.{layer}.{suffix}");
            match weights.raw_with_dtype(&name) {
                Ok((_, dtype)) if is_supported(dtype) => {}
                Ok((_, dtype)) => {
                    warn_gpu_path_issue_once(format!("unsupported:{name}:{dtype:?}"), || {
                        tracing::warn!(%name, ?dtype, "unsupported quant dtype for GPU path");
                    });
                    return Some(format!("{name}:{dtype:?}"));
                }
                Err(e) => {
                    warn_gpu_path_issue_once(format!("missing:{name}:{e}"), || {
                        tracing::warn!(
                            %name,
                            error = %e,
                            "missing or unreadable tensor for GPU path"
                        );
                    });
                    return Some(format!("{name}:missing"));
                }
            }
        }
    }
    None
}

fn warn_gpu_path_issue_once(key: String, warn: impl FnOnce()) {
    static WARNED_GPU_PATH_ISSUES: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

    let warned = WARNED_GPU_PATH_ISSUES.get_or_init(|| Mutex::new(HashSet::new()));
    let mut warned = warned
        .lock()
        .expect("WARNED_GPU_PATH_ISSUES mutex should not be poisoned");
    if warned.insert(key) {
        warn();
    }
}

fn gpu_batch_prefill_panic(dtype: GgmlType) -> ! {
    panic!(
        "GPU batch matmul only supports Q4_0, Q4_K, Q5_K, Q6_K, and Q8_0, got {:?}",
        dtype
    )
}

/// Apply per-head RMSNorm in-place.
///
/// `buf` contains `n_heads` concatenated vectors of size `head_dim`.
/// `weight` has length `head_dim` and is shared across all heads.
pub(super) fn per_head_rms_norm(
    buf: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    weight: &[f32],
    eps: f32,
) {
    debug_assert_eq!(buf.len(), n_heads * head_dim);
    debug_assert_eq!(weight.len(), head_dim);
    for head in buf.chunks_mut(head_dim) {
        rms_norm::rms_norm(head, weight, eps);
    }
}
