//! Shared helpers used by all three model implementations (LLaMA, Gemma3, Qwen3).

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use crate::backend::metal::MetalOps;
use crate::compute::rms_norm;
use crate::gguf::tensor::GgmlType;
use crate::model::weights::WeightStore;

const LAYER_SUFFIXES: &[&str] = &[
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

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

/// Encode a fused dequant+matvec dispatch for the appropriate quant type.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_dequant_matvec(
    metal_ops: &MetalOps,
    encoder: &ax_metal::MetalEncoder,
    weight: &ax_metal::MetalBuffer,
    input: &ax_metal::MetalBuffer,
    output: &ax_metal::MetalBuffer,
    m: u32,
    k: u32,
    dtype: GgmlType,
) {
    encode_dequant_matvec_with_config(
        metal_ops,
        encoder,
        weight,
        input,
        output,
        m,
        k,
        dtype,
        metal_ops.dequant_dispatch_config(),
    )
}

/// Encode a fused dequant+matvec dispatch using an explicit dispatch config.
///
/// Decode execution planning uses this to keep kernel routing aligned with the
/// typed plan rather than re-reading backend state at every call site.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_dequant_matvec_with_config(
    metal_ops: &MetalOps,
    encoder: &ax_metal::MetalEncoder,
    weight: &ax_metal::MetalBuffer,
    input: &ax_metal::MetalBuffer,
    output: &ax_metal::MetalBuffer,
    m: u32,
    k: u32,
    dtype: GgmlType,
    dispatch_config: ax_metal::DequantDispatchConfig,
) {
    match dtype {
        GgmlType::Q4_0 => metal_ops
            .dequant
            .encode_fused_matvec_q4_0(encoder, weight, input, output, m, k),
        GgmlType::Q8_0 => {
            if metal_ops
                .encode_precomputed_q4k_matvec_if_available(encoder, weight, input, output, m, k)
            {
                return;
            }
            metal_ops
                .dequant
                .encode_fused_matvec_q8_0(encoder, weight, input, output, m, k)
        }
        GgmlType::Q4K => {
            if metal_ops
                .encode_precomputed_q4k_matvec_if_available(encoder, weight, input, output, m, k)
            {
                return;
            }
            metal_ops.dequant.encode_fused_matvec_q4_k_with_config(
                encoder,
                weight,
                input,
                output,
                m,
                k,
                dispatch_config,
            )
        }
        GgmlType::Q5K => metal_ops.dequant.encode_fused_matvec_q5_k_with_config(
            encoder,
            weight,
            input,
            output,
            m,
            k,
            dispatch_config,
        ),
        GgmlType::Q6K => {
            if metal_ops
                .encode_precomputed_q4k_matvec_if_available(encoder, weight, input, output, m, k)
            {
                return;
            }
            metal_ops.dequant.encode_fused_matvec_q6_k_with_config(
                encoder,
                weight,
                input,
                output,
                m,
                k,
                dispatch_config,
            )
        }
        _ => panic!(
            "GPU phased dispatch only supports Q4_0, Q8_0, Q4_K, Q5_K, and Q6_K, got {:?}",
            dtype
        ),
    }
}

/// Encode a batched dequant+matmul: C[N×M] = B[N×K] × dequant(A[M×K])^T.
///
/// If `use_f16_io` is true, casts `input` to f16 into `input_f16` before dispatch
/// (avoids per-matmul output cast while keeping downstream buffers f32).
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_dequant_batch(
    dequant: &ax_metal::DequantKernels,
    elementwise: &ax_metal::ElementwiseKernels,
    encoder: &ax_metal::MetalEncoder,
    weight: &ax_metal::MetalBuffer,
    input: &ax_metal::MetalBuffer,
    output: &ax_metal::MetalBuffer,
    input_f16: &ax_metal::MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
    dtype: GgmlType,
    use_f16_io: bool,
    use_batch_simd: bool,
    use_q5k_small_n: bool,
) {
    if use_batch_simd {
        match dtype {
            GgmlType::Q4K => {
                dequant.encode_batch_simd_q4k(encoder, weight, input, output, m, n, k, m);
                return;
            }
            GgmlType::Q6K => {
                dequant.encode_batch_simd_q6k(encoder, weight, input, output, m, n, k, m);
                return;
            }
            _ => {}
        }
    }

    if use_f16_io || dtype == GgmlType::Q4_0 {
        elementwise.encode_cast_f32_to_f16(encoder, input, input_f16, n * k);
        match dtype {
            GgmlType::Q4_0 => {
                dequant.encode_fused_batch_q4_0_f16in(encoder, weight, input_f16, output, m, n, k)
            }
            GgmlType::Q4K => dequant.encode_fused_batch_q4_k_f16in_with_config(
                encoder,
                weight,
                input_f16,
                output,
                m,
                n,
                k,
                ax_metal::DequantDispatchConfig::default(),
            ),
            GgmlType::Q6K => dequant.encode_fused_batch_q6_k_f16in_with_config(
                encoder,
                weight,
                input_f16,
                output,
                m,
                n,
                k,
                ax_metal::DequantDispatchConfig::default(),
            ),
            _ => gpu_batch_prefill_panic(dtype),
        }
    } else {
        match dtype {
            GgmlType::Q4K => {
                dequant.encode_fused_batch_q4_k(encoder, weight, input, output, m, n, k)
            }
            GgmlType::Q5K => {
                if use_q5k_small_n {
                    dequant.encode_fused_batch_q5_k_small(encoder, weight, input, output, m, n, k)
                } else {
                    dequant.encode_fused_batch_q5_k(encoder, weight, input, output, m, n, k)
                }
            }
            GgmlType::Q6K => {
                dequant.encode_fused_batch_q6_k(encoder, weight, input, output, m, n, k)
            }
            _ => gpu_batch_prefill_panic(dtype),
        }
    }
}

/// Encode a batched dequant+matmul with pre-cast f16 input.
///
/// Caller is responsible for casting input to f16 in `input_f16` before calling.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_dequant_batch_f16in(
    metal_ops: &MetalOps,
    encoder: &ax_metal::MetalEncoder,
    weight: &ax_metal::MetalBuffer,
    input_f16: &ax_metal::MetalBuffer,
    output: &ax_metal::MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
    dtype: GgmlType,
) {
    match dtype {
        GgmlType::Q4_0 => {
            metal_ops
                .dequant
                .encode_fused_batch_q4_0_f16in(encoder, weight, input_f16, output, m, n, k);
        }
        GgmlType::Q8_0 => {
            if metal_ops.metal_q8_batch_native_shape_enabled(m, n, k) {
                metal_ops.dequant.encode_fused_batch_q8_0_f16in_with_config(
                    encoder,
                    weight,
                    input_f16,
                    output,
                    m,
                    n,
                    k,
                    metal_ops.dequant_dispatch_config(),
                );
                return;
            }
            if metal_ops
                .encode_precomputed_batch_if_available(encoder, weight, input_f16, output, m, n, k)
            {
                return;
            }
            panic!(
                "GPU batch matmul for Q8_0 requires precomputed dense f16 weight, got {:?}",
                dtype
            );
        }
        GgmlType::Q4K => {
            if metal_ops.encode_precomputed_q4k_batch_if_available(
                encoder, weight, input_f16, output, m, n, k,
            ) {
                return;
            }
            metal_ops.dequant.encode_fused_batch_q4_k_f16in_with_config(
                encoder,
                weight,
                input_f16,
                output,
                m,
                n,
                k,
                metal_ops.dequant_dispatch_config(),
            )
        }
        GgmlType::Q6K => {
            if metal_ops.encode_precomputed_q4k_batch_if_available(
                encoder, weight, input_f16, output, m, n, k,
            ) {
                return;
            }
            metal_ops.dequant.encode_fused_batch_q6_k_f16in_with_config(
                encoder,
                weight,
                input_f16,
                output,
                m,
                n,
                k,
                metal_ops.dequant_dispatch_config(),
            )
        }
        GgmlType::Q5K => metal_ops
            .dequant
            .encode_fused_batch_q5_k_f16in(encoder, weight, input_f16, output, m, n, k),
        _ => gpu_batch_prefill_panic(dtype),
    }
}

/// Encode a batched LM-head projection from `[n × k]` hidden states to
/// `[n × vocab]` logits.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_batch_logits(
    metal_ops: &MetalOps,
    encoder: &ax_metal::MetalEncoder,
    weight: &ax_metal::MetalBuffer,
    hidden: &ax_metal::MetalBuffer,
    hidden_f16: &ax_metal::MetalBuffer,
    logits: &ax_metal::MetalBuffer,
    vocab: u32,
    n_rows: u32,
    hidden_dim: u32,
    dtype: GgmlType,
    prefer_f16_io: bool,
    use_batch_simd: bool,
) {
    match dtype {
        GgmlType::Q4_0 | GgmlType::Q8_0 => {
            metal_ops.elementwise.encode_cast_f32_to_f16(
                encoder,
                hidden,
                hidden_f16,
                n_rows * hidden_dim,
            );
            encode_dequant_batch_f16in(
                metal_ops, encoder, weight, hidden_f16, logits, vocab, n_rows, hidden_dim, dtype,
            );
        }
        GgmlType::Q4K | GgmlType::Q6K => {
            if prefer_f16_io {
                metal_ops.elementwise.encode_cast_f32_to_f16(
                    encoder,
                    hidden,
                    hidden_f16,
                    n_rows * hidden_dim,
                );
                encode_dequant_batch_f16in(
                    metal_ops, encoder, weight, hidden_f16, logits, vocab, n_rows, hidden_dim,
                    dtype,
                );
            } else {
                encode_dequant_batch(
                    &metal_ops.dequant,
                    &metal_ops.elementwise,
                    encoder,
                    weight,
                    hidden,
                    logits,
                    hidden_f16,
                    vocab,
                    n_rows,
                    hidden_dim,
                    dtype,
                    false,
                    use_batch_simd,
                    false,
                );
            }
        }
        _ => panic!("GPU batch logits path does not support {:?}", dtype),
    }
}

/// Encode a fused pair of batched dequant+matmuls with pre-cast f16 input.
///
/// Dispatches gate and up projections in a single paired kernel.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_dequant_batch_pair_f16in(
    dequant: &ax_metal::DequantKernels,
    encoder: &ax_metal::MetalEncoder,
    w0: &ax_metal::MetalBuffer,
    w1: &ax_metal::MetalBuffer,
    input_f16: &ax_metal::MetalBuffer,
    out0: &ax_metal::MetalBuffer,
    out1: &ax_metal::MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
    dtype: GgmlType,
) {
    match dtype {
        GgmlType::Q4K => dequant
            .encode_fused_batch_pair_q4_k_f16in(encoder, w0, w1, input_f16, out0, out1, m, n, k),
        GgmlType::Q6K => dequant
            .encode_fused_batch_pair_q6_k_f16in(encoder, w0, w1, input_f16, out0, out1, m, n, k),
        GgmlType::Q8_0 => dequant
            .encode_fused_batch_pair_q8_0_f16in(encoder, w0, w1, input_f16, out0, out1, m, n, k),
        _ => panic!(
            "GPU batch pair matmul only supports Q4_K, Q6_K, and Q8_0, got {:?}",
            dtype
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::sync::MutexGuard;

    fn env_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("shared env test lock")
    }

    struct EnvVarRestore {
        key: String,
        previous: Option<OsString>,
    }

    impl Drop for EnvVarRestore {
        fn drop(&mut self) {
            match &self.previous {
                Some(prev) => unsafe {
                    std::env::set_var(&self.key, prev);
                },
                None => unsafe {
                    std::env::remove_var(&self.key);
                },
            }
        }
    }

    fn with_env_var<T>(key: &str, value: &str, f: impl FnOnce() -> T) -> T {
        let _guard = env_lock();
        let _restore = EnvVarRestore {
            key: key.to_string(),
            previous: std::env::var_os(key),
        };
        unsafe {
            std::env::set_var(key, value);
        }
        f()
    }

    #[test]
    fn test_q5k_is_supported_gpu_prefill_quant() {
        assert!(gpu_decode_quant_dtype_supported(GgmlType::Q5K));
        assert!(gpu_prefill_quant_dtype_supported(GgmlType::Q5K));
        assert!(!gpu_batch_logits_dtype_supported(GgmlType::Q5K));
    }

    #[test]
    fn test_q5k_prefill_variant_override_parses_known_values() {
        assert_eq!(
            q5k_prefill_variant_override(),
            Q5KPrefillVariantOverride::Auto
        );
        with_env_var("AX_METAL_Q5K_PREFILL_VARIANT", "base", || {
            assert_eq!(
                q5k_prefill_variant_override(),
                Q5KPrefillVariantOverride::Base
            );
        });
        with_env_var("AX_METAL_Q5K_PREFILL_VARIANT", "small", || {
            assert_eq!(
                q5k_prefill_variant_override(),
                Q5KPrefillVariantOverride::Small
            );
        });
        with_env_var("AX_METAL_Q5K_PREFILL_VARIANT", "auto", || {
            assert_eq!(
                q5k_prefill_variant_override(),
                Q5KPrefillVariantOverride::Auto
            );
        });
    }

    #[test]
    fn test_q5k_prefill_variant_override_accepts_legacy_env_alias() {
        with_env_var("AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT", "small", || {
            assert_eq!(
                q5k_prefill_variant_override(),
                Q5KPrefillVariantOverride::Small
            );
        });
    }

    #[test]
    fn test_q5k_prefill_small_n_auto_eligible_for_predominant_q5k_models_only() {
        assert!(q5k_prefill_small_n_auto_eligible_for_model(
            Some(GgmlType::Q5K),
            true
        ));
        assert!(!q5k_prefill_small_n_auto_eligible_for_model(
            Some(GgmlType::Q4K),
            true
        ));
        assert!(!q5k_prefill_small_n_auto_eligible_for_model(
            Some(GgmlType::Q5K),
            false
        ));
    }

    #[test]
    fn test_env_flag_enabled_parses_known_truthy_values() {
        let key = "AX_TEST_ENV_FLAG_ENABLED";
        assert!(!env_flag_enabled(key));
        with_env_var(key, "1", || {
            assert!(env_flag_enabled(key));
        });
        with_env_var(key, "true", || {
            assert!(env_flag_enabled(key));
        });
        with_env_var(key, "on", || {
            assert!(env_flag_enabled(key));
        });
        with_env_var(key, "false", || {
            assert!(!env_flag_enabled(key));
        });
        with_env_var(key, "", || {
            assert!(!env_flag_enabled(key));
        });
    }

    #[test]
    fn test_warn_gpu_path_issue_once_only_runs_first_warning() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static WARN_COUNT: AtomicUsize = AtomicUsize::new(0);
        let key = "test:unsupported:blk.0.attn_q.weight:Q5K".to_string();

        warn_gpu_path_issue_once(key.clone(), || {
            WARN_COUNT.fetch_add(1, Ordering::Relaxed);
        });
        warn_gpu_path_issue_once(key, || {
            WARN_COUNT.fetch_add(1, Ordering::Relaxed);
        });

        assert_eq!(WARN_COUNT.load(Ordering::Relaxed), 1);
    }
}
