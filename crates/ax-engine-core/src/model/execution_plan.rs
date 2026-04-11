use super::InferenceModel;
use super::config::ModelConfig;
use super::decode::{DecodeIntent, DecodeMode, DecodeSelection};
use super::forward::ForwardContext;
use super::gemma3::Gemma3Forward;
use super::gemma4::Gemma4Forward;
use super::shared::{
    Q5KPrefillVariantOverride, env_flag_enabled, env_flag_override, gpu_batch_logits_supported,
    gpu_prefill_quant_blocker, q5k_prefill_enabled, q5k_prefill_variant_override,
};
use crate::backend::metal::MetalOps;
use crate::gguf::tensor::GgmlType;
use crate::kv::{GpuKv, ModelKv};
use crate::model::weights::WeightStore;

// llama.cpp routes K-quant small-batch matvec through its dedicated ext kernels
// only for BS in [4, 8]. Keep AX aligned with that crossover until Q5_K earns a
// broader runtime policy from measured data.
const Q5K_PREFILL_SMALL_N_MIN: u32 = 4;
const Q5K_PREFILL_SMALL_N_MAX: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Q5KPrefillRoute {
    enabled: bool,
    small_n: bool,
    forced_variant: bool,
    use_f16_batch_io: bool,
}

fn q5k_prefill_route(
    has_q5k_weights: bool,
    q5k_small_n_auto_eligible: bool,
    n_tokens: u32,
) -> Q5KPrefillRoute {
    let enabled = has_q5k_weights && q5k_prefill_enabled();
    let override_variant = q5k_prefill_variant_override();
    let small_n = enabled
        && match override_variant {
            Q5KPrefillVariantOverride::Auto => {
                q5k_small_n_auto_eligible
                    && (Q5K_PREFILL_SMALL_N_MIN..=Q5K_PREFILL_SMALL_N_MAX).contains(&n_tokens)
            }
            Q5KPrefillVariantOverride::Base => false,
            Q5KPrefillVariantOverride::Small => true,
        };
    Q5KPrefillRoute {
        enabled,
        small_n,
        forced_variant: enabled && !matches!(override_variant, Q5KPrefillVariantOverride::Auto),
        // Keep Q5_K on the normal blocked/base projection route. The pair
        // experiment is FFN-local and should not flip QKV/attention over to
        // the f16-input path.
        use_f16_batch_io: false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeSyncPlan {
    Sequential,
    SingleCommandBuffer,
    Pipelined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeBarrierPlan {
    /// No explicit barriers — relies on serial encoder ordering.
    Implicit,
    /// Blanket `memoryBarrierWithScope(Buffers)` at every phase boundary.
    Explicit,
    /// Per-dispatch SmartBarrier tracking (llama.cpp pattern).
    ///
    /// Barriers are inserted only when a new dispatch reads a buffer that a
    /// pending dispatch wrote, or writes a buffer with any pending access.
    /// Used with the concurrent encoder — allows the GPU to overlap
    /// independent dispatches (e.g. separate Q/K/V matvecs, gate/up).
    Smart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeEncoderPlan {
    Serial,
    Concurrent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeQkvPlan {
    Split,
    Fused,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeScratchPlan {
    CpuScratch,
    SharedGpuScratch,
    HybridBackendOwned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillMode {
    Serial,
    CpuBatch,
    GpuBatch,
    GpuChunked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillAttentionPlan {
    BatchLocal,
    BatchLocalF16OutHd128,
    Cached,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum PrefillWoInputPlan {
    AttentionOutF32,
    MatmulScratchF16,
}

#[derive(Debug, Clone, Copy)]
pub struct GpuDecodeExecutionPlan {
    pub encoder: DecodeEncoderPlan,
    pub barriers: DecodeBarrierPlan,
    pub qkv: DecodeQkvPlan,
    pub kv_f16: bool,
    #[allow(dead_code)]
    pub kv_q8: bool,
    pub use_pair_matvec: bool,
    pub use_fused_silu_down: bool,
    pub attention_route: &'static str,
    pub attention_tier: &'static str,
    pub q4_k_candidate: &'static str,
    pub q4_k_tier: &'static str,
    pub q5_k_candidate: &'static str,
    pub q5_k_tier: &'static str,
    pub q6_k_candidate: &'static str,
    pub q6_k_tier: &'static str,
    pub dequant_dispatch: ax_engine_metal::DequantDispatchConfig,
    pub attention_dispatch: ax_engine_metal::AttentionDispatchConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub struct Qwen3DecodeLayerPlan {
    pub qkv: DecodeQkvPlan,
    pub qwen3_post: Qwen3PrefillQkvPost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuBatchPrefillExecutionPlan {
    pub kv_f16: bool,
    pub kv_q8: bool,
    pub use_f16_batch_io: bool,
    pub use_f16_pair: bool,
    pub use_fused_qkv: bool,
    pub use_batch_simd: bool,
    pub q5k_prefill: bool,
    pub q5k_prefill_small_n: bool,
    pub q5k_prefill_forced_variant: bool,
    pub split_rope_append: bool,
    pub attention: PrefillAttentionPlan,
    pub attention_sliding_window: u32,
    pub wo_input: PrefillWoInputPlan,
    pub attention_dispatch: ax_engine_metal::AttentionDispatchConfig,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gemma3PrefillLayerPlan {
    pub attention: PrefillAttentionPlan,
    pub sliding_window: u32,
    pub rope_base: f32,
    pub rope_start: f32,
    pub rope_step: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum PrefillFfnActivationPlan {
    SiluMulGateF32,
    SiluMulScratchF16,
    GeluMulGateF32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum PrefillResidualHandoffPlan {
    ResidualOnly,
    ResidualAddRmsNormF32,
    ResidualAddRmsNormF16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillLogitsPlan {
    BatchAllLogits,
    LastTokenMatvec,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillProjectionInputPlan {
    NormBufF32,
    MatmulScratchF16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillFfnLayerPlan {
    pub input: PrefillProjectionInputPlan,
    pub use_pair_kernel: bool,
    pub activation: PrefillFfnActivationPlan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Qwen3PrefillQkvPost {
    Separate,
    SeparateBias,
    SeparateQkNorm,
    SeparateBiasQkNorm,
    FusedBiasQkNorm,
    FusedQkNorm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillQkvLayerPlan {
    pub input: PrefillProjectionInputPlan,
    pub use_fused_projection: bool,
    pub qwen3_post: Qwen3PrefillQkvPost,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillExecutionPlan {
    pub mode: PrefillMode,
    pub chunk_len: Option<usize>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DecodeExecutionPlan {
    pub selection: DecodeSelection,
    pub sync: DecodeSyncPlan,
    pub scratch: DecodeScratchPlan,
    pub gpu: Option<GpuDecodeExecutionPlan>,
}

fn kv_precision_label(kv_f16: bool, kv_q8: bool) -> &'static str {
    if kv_q8 {
        "q8_0"
    } else if kv_f16 {
        "f16"
    } else {
        "f32"
    }
}

impl DecodeExecutionPlan {
    pub fn for_model(
        model: &InferenceModel,
        kv: &ModelKv,
        intent: DecodeIntent,
        allow_pipelined: bool,
    ) -> Self {
        let has_gpu_decode = kv.is_gpu() && model.metal_device().is_some();
        let qwen35_gpu_attention = kv.as_qwen35().and_then(|qwen_kv| qwen_kv.gpu_attention());
        let qwen35_hybrid_decode = model.arch_name() == "qwen35"
            && matches!(kv, ModelKv::Qwen35(_))
            && qwen35_gpu_attention.is_some()
            && model.use_gpu_decode()
            && model.metal_device().is_some();
        let supports_pipelined = has_gpu_decode && model.supports_pipelined_decode();

        let selection = if qwen35_hybrid_decode {
            qwen35_hybrid_decode_selection(
                intent,
                allow_pipelined,
                model.supports_pipelined_decode(),
            )
        } else {
            select_decode_execution_mode(
                intent,
                allow_pipelined,
                has_gpu_decode,
                supports_pipelined,
            )
        };
        let sync = match selection.mode {
            DecodeMode::Sequential => DecodeSyncPlan::Sequential,
            DecodeMode::SingleCb => DecodeSyncPlan::SingleCommandBuffer,
            DecodeMode::Pipelined => DecodeSyncPlan::Pipelined,
        };
        let scratch = if qwen35_hybrid_decode {
            DecodeScratchPlan::HybridBackendOwned
        } else {
            match sync {
                DecodeSyncPlan::Sequential => DecodeScratchPlan::CpuScratch,
                DecodeSyncPlan::SingleCommandBuffer | DecodeSyncPlan::Pipelined => {
                    DecodeScratchPlan::SharedGpuScratch
                }
            }
        };

        let gpu = if qwen35_hybrid_decode {
            model
                .metal_ops()
                .zip(qwen35_gpu_attention)
                .map(|(metal_ops, gpu_kv)| match sync {
                    DecodeSyncPlan::Sequential | DecodeSyncPlan::SingleCommandBuffer => {
                        Self::qwen35_single_cb(
                            metal_ops,
                            gpu_kv,
                            model.config.embedding_dim,
                            model.config.head_dim,
                            kv.seq_len().saturating_add(1),
                        )
                    }
                    DecodeSyncPlan::Pipelined => Self::qwen35_pipelined(
                        metal_ops,
                        gpu_kv,
                        model.config.embedding_dim,
                        model.config.head_dim,
                        kv.seq_len().saturating_add(1),
                    ),
                })
        } else {
            match sync {
                DecodeSyncPlan::Sequential => None,
                DecodeSyncPlan::SingleCommandBuffer => {
                    model
                        .metal_ops()
                        .zip(kv.as_gpu())
                        .map(|(metal_ops, gpu_kv)| {
                            single_cb_gpu_decode_plan(
                                model.arch_name(),
                                metal_ops,
                                gpu_kv,
                                model.config.embedding_dim,
                                model.config.head_dim,
                                kv.seq_len().saturating_add(1),
                            )
                        })
                }
                DecodeSyncPlan::Pipelined => {
                    model
                        .metal_ops()
                        .zip(kv.as_gpu())
                        .map(|(metal_ops, gpu_kv)| {
                            pipelined_gpu_decode_plan(
                                model.arch_name(),
                                metal_ops,
                                gpu_kv,
                                model.config.embedding_dim,
                                model.config.head_dim,
                                kv.seq_len().saturating_add(1),
                            )
                        })
                }
            }
        };

        Self {
            selection,
            sync,
            scratch,
            gpu,
        }
    }

    pub(crate) fn qwen35_single_cb(
        metal_ops: &MetalOps,
        gpu_kv: &GpuKv,
        embedding_dim: u32,
        head_dim: u32,
        attend_len: usize,
    ) -> GpuDecodeExecutionPlan {
        single_cb_gpu_decode_plan(
            "qwen35",
            metal_ops,
            gpu_kv,
            embedding_dim,
            head_dim,
            attend_len,
        )
    }

    pub(crate) fn qwen35_pipelined(
        metal_ops: &MetalOps,
        gpu_kv: &GpuKv,
        embedding_dim: u32,
        head_dim: u32,
        attend_len: usize,
    ) -> GpuDecodeExecutionPlan {
        pipelined_gpu_decode_plan(
            "qwen35",
            metal_ops,
            gpu_kv,
            embedding_dim,
            head_dim,
            attend_len,
        )
    }

    pub(crate) fn qwen3moe_single_cb(
        metal_ops: &MetalOps,
        gpu_kv: &GpuKv,
        embedding_dim: u32,
        head_dim: u32,
        attend_len: usize,
    ) -> GpuDecodeExecutionPlan {
        single_cb_gpu_decode_plan(
            "qwen3moe",
            metal_ops,
            gpu_kv,
            embedding_dim,
            head_dim,
            attend_len,
        )
    }

    pub(crate) fn qwen3moe_pipelined(
        metal_ops: &MetalOps,
        gpu_kv: &GpuKv,
        embedding_dim: u32,
        head_dim: u32,
        attend_len: usize,
    ) -> GpuDecodeExecutionPlan {
        pipelined_gpu_decode_plan(
            "qwen3moe",
            metal_ops,
            gpu_kv,
            embedding_dim,
            head_dim,
            attend_len,
        )
    }

    pub(crate) fn gemma3_single_cb(
        metal_ops: &MetalOps,
        gpu_kv: &GpuKv,
        embedding_dim: u32,
        head_dim: u32,
        attend_len: usize,
    ) -> GpuDecodeExecutionPlan {
        single_cb_gpu_decode_plan(
            "gemma3",
            metal_ops,
            gpu_kv,
            embedding_dim,
            head_dim,
            attend_len,
        )
    }

    pub(crate) fn gemma4_single_cb(
        metal_ops: &MetalOps,
        gpu_kv: &GpuKv,
        embedding_dim: u32,
        head_dim: u32,
        attend_len: usize,
    ) -> GpuDecodeExecutionPlan {
        single_cb_gpu_decode_plan(
            "gemma4",
            metal_ops,
            gpu_kv,
            embedding_dim,
            head_dim,
            attend_len,
        )
    }

    pub(crate) fn gemma3_pipelined(
        metal_ops: &MetalOps,
        gpu_kv: &GpuKv,
        embedding_dim: u32,
        head_dim: u32,
        attend_len: usize,
    ) -> GpuDecodeExecutionPlan {
        pipelined_gpu_decode_plan(
            "gemma3",
            metal_ops,
            gpu_kv,
            embedding_dim,
            head_dim,
            attend_len,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qwen3_prefill(
        metal_ops: &MetalOps,
        gpu_kv: &GpuKv,
        base_seq_len: usize,
        n_tokens: u32,
        head_dim: u32,
        sliding_window: u32,
        has_q5k_weights: bool,
        q5k_small_n_auto_eligible: bool,
    ) -> GpuBatchPrefillExecutionPlan {
        let q5k_route = q5k_prefill_route(has_q5k_weights, q5k_small_n_auto_eligible, n_tokens);
        let use_f16_batch_io = q5k_route.use_f16_batch_io
            || (!q5k_route.enabled && metal_ops.metal_batch_f16_io_enabled());
        GpuBatchPrefillExecutionPlan {
            kv_f16: gpu_kv.is_f16(),
            kv_q8: gpu_kv.is_q8(),
            use_f16_batch_io,
            use_f16_pair: !q5k_route.enabled && metal_ops.metal_batch_f16_pair_enabled(),
            use_fused_qkv: metal_ops.metal_fused_qkv_enabled(),
            use_batch_simd: !q5k_route.enabled && metal_ops.metal_batch_simd_enabled(),
            q5k_prefill: q5k_route.enabled,
            q5k_prefill_small_n: q5k_route.small_n,
            q5k_prefill_forced_variant: q5k_route.forced_variant,
            split_rope_append: false,
            attention: prefill_attention_plan(
                base_seq_len,
                sliding_window,
                use_f16_batch_io,
                head_dim,
                false,
                false,
            ),
            attention_sliding_window: sliding_window,
            wo_input: PrefillWoInputPlan::AttentionOutF32,
            attention_dispatch: metal_ops.attention_dispatch_config(),
        }
    }

    pub(crate) fn gemma3_prefill(
        metal_ops: &MetalOps,
        gpu_kv: &GpuKv,
        n_tokens: u32,
        has_q5k_weights: bool,
        q5k_small_n_auto_eligible: bool,
    ) -> GpuBatchPrefillExecutionPlan {
        let q5k_route = q5k_prefill_route(has_q5k_weights, q5k_small_n_auto_eligible, n_tokens);
        let use_f16_batch_io = q5k_route.use_f16_batch_io
            || (!q5k_route.enabled && metal_ops.metal_batch_f16_io_enabled());
        GpuBatchPrefillExecutionPlan {
            kv_f16: gpu_kv.is_f16(),
            kv_q8: gpu_kv.is_q8(),
            use_f16_batch_io,
            use_f16_pair: !q5k_route.enabled && metal_ops.metal_batch_f16_pair_enabled(),
            use_fused_qkv: metal_ops.metal_fused_qkv_enabled(),
            use_batch_simd: !q5k_route.enabled && metal_ops.metal_batch_simd_enabled(),
            q5k_prefill: q5k_route.enabled,
            q5k_prefill_small_n: q5k_route.small_n,
            q5k_prefill_forced_variant: q5k_route.forced_variant,
            split_rope_append: false,
            attention: PrefillAttentionPlan::Cached,
            attention_sliding_window: 0,
            wo_input: PrefillWoInputPlan::AttentionOutF32,
            attention_dispatch: metal_ops.attention_dispatch_config(),
        }
    }

    pub(crate) fn gemma3_prefill_layer(
        config: &ModelConfig,
        layer: usize,
        base_seq_len: usize,
        use_f16_batch_io: bool,
    ) -> Gemma3PrefillLayerPlan {
        gemma3_prefill_layer_plan(config, layer, base_seq_len, use_f16_batch_io)
    }

    #[cfg(test)]
    pub(crate) fn qwen3_prefill_ffn_layer(
        prefill_plan: &GpuBatchPrefillExecutionPlan,
        wg_dtype: GgmlType,
        wu_dtype: GgmlType,
        n_tokens: usize,
    ) -> PrefillFfnLayerPlan {
        PrefillFfnLayerPlan {
            input: if prefill_plan.use_f16_batch_io {
                PrefillProjectionInputPlan::MatmulScratchF16
            } else {
                PrefillProjectionInputPlan::NormBufF32
            },
            use_pair_kernel: prefill_plan.use_f16_batch_io
                && (prefill_plan.use_f16_pair || (wg_dtype == GgmlType::Q8_0 && n_tokens <= 256))
                && wg_dtype == wu_dtype,
            activation: PrefillFfnActivationPlan::SiluMulGateF32,
        }
    }

    pub(crate) fn gemma3_prefill_ffn_layer(
        prefill_plan: &GpuBatchPrefillExecutionPlan,
        wg_dtype: GgmlType,
        wu_dtype: GgmlType,
    ) -> PrefillFfnLayerPlan {
        PrefillFfnLayerPlan {
            input: if prefill_plan.use_f16_batch_io {
                PrefillProjectionInputPlan::MatmulScratchF16
            } else {
                PrefillProjectionInputPlan::NormBufF32
            },
            use_pair_kernel: prefill_plan.use_f16_batch_io
                && prefill_plan.use_f16_pair
                && wg_dtype == wu_dtype,
            activation: PrefillFfnActivationPlan::GeluMulGateF32,
        }
    }

    #[cfg(test)]
    pub(crate) fn qwen3_prefill_residual_handoff(
        is_last_layer: bool,
    ) -> PrefillResidualHandoffPlan {
        if is_last_layer {
            PrefillResidualHandoffPlan::ResidualOnly
        } else {
            PrefillResidualHandoffPlan::ResidualAddRmsNormF32
        }
    }

    pub(crate) fn gemma3_prefill_residual_handoff(
        is_last_layer: bool,
    ) -> PrefillResidualHandoffPlan {
        if is_last_layer {
            PrefillResidualHandoffPlan::ResidualOnly
        } else {
            PrefillResidualHandoffPlan::ResidualAddRmsNormF32
        }
    }

    pub(crate) fn prefill_logits_plan(emit_all_logits: bool) -> PrefillLogitsPlan {
        if emit_all_logits {
            PrefillLogitsPlan::BatchAllLogits
        } else {
            PrefillLogitsPlan::LastTokenMatvec
        }
    }

    #[cfg(test)]
    pub(crate) fn qwen3_prefill_qkv_layer(
        prefill_plan: &GpuBatchPrefillExecutionPlan,
        wq_dtype: GgmlType,
        wk_dtype: GgmlType,
        wv_dtype: GgmlType,
        has_bias: bool,
        has_qk_norm: bool,
    ) -> PrefillQkvLayerPlan {
        let use_fused_projection = prefill_plan.use_fused_qkv
            && wq_dtype == wk_dtype
            && wq_dtype == wv_dtype
            && matches!(wq_dtype, GgmlType::Q4K | GgmlType::Q6K);
        let qwen3_post = if use_fused_projection && has_bias && has_qk_norm {
            Qwen3PrefillQkvPost::FusedBiasQkNorm
        } else if use_fused_projection && !has_bias && has_qk_norm {
            Qwen3PrefillQkvPost::FusedQkNorm
        } else if has_bias && has_qk_norm {
            Qwen3PrefillQkvPost::SeparateBiasQkNorm
        } else if has_bias {
            Qwen3PrefillQkvPost::SeparateBias
        } else if has_qk_norm {
            Qwen3PrefillQkvPost::SeparateQkNorm
        } else {
            Qwen3PrefillQkvPost::Separate
        };
        PrefillQkvLayerPlan {
            input: if prefill_plan.use_f16_batch_io {
                PrefillProjectionInputPlan::MatmulScratchF16
            } else {
                PrefillProjectionInputPlan::NormBufF32
            },
            use_fused_projection,
            qwen3_post,
        }
    }

    pub(crate) fn gemma3_prefill_qkv_layer(
        prefill_plan: &GpuBatchPrefillExecutionPlan,
        wq_dtype: GgmlType,
        wk_dtype: GgmlType,
        wv_dtype: GgmlType,
    ) -> PrefillQkvLayerPlan {
        PrefillQkvLayerPlan {
            input: if prefill_plan.use_f16_batch_io {
                PrefillProjectionInputPlan::MatmulScratchF16
            } else {
                PrefillProjectionInputPlan::NormBufF32
            },
            use_fused_projection: prefill_plan.use_fused_qkv
                && wq_dtype == wk_dtype
                && wq_dtype == wv_dtype
                && matches!(wq_dtype, GgmlType::Q4K | GgmlType::Q6K),
            qwen3_post: Qwen3PrefillQkvPost::Separate,
        }
    }

    pub fn summary_label(&self) -> String {
        let mut parts = vec![
            format!("sync={}", self.sync.label()),
            format!("scratch={}", self.scratch.label()),
        ];
        if let Some(gpu) = &self.gpu {
            parts.push(format!("barriers={}", gpu.barriers.label()));
            parts.push(format!("qkv={}", gpu.qkv.label()));
            parts.push(format!("kv={}", kv_precision_label(gpu.kv_f16, gpu.kv_q8)));
            parts.push(format!(
                "attn={}/{}",
                gpu.attention_route, gpu.attention_tier
            ));
            parts.push(format!("q4k={}/{}", gpu.q4_k_candidate, gpu.q4_k_tier));
            parts.push(format!("q5k={}/{}", gpu.q5_k_candidate, gpu.q5_k_tier));
            parts.push(format!("q6k={}/{}", gpu.q6_k_candidate, gpu.q6_k_tier));
        }
        parts.join(" ")
    }
}

fn supports_backend_prefill_kv(config: &ModelConfig, kv: &ModelKv) -> bool {
    kv.as_gpu().is_some()
        || (matches!(config.architecture.as_str(), "qwen35" | "qwen35moe")
            && matches!(kv, ModelKv::Qwen35(_)))
}

fn qwen_gpu_prefill_plan() -> PrefillExecutionPlan {
    PrefillExecutionPlan {
        mode: PrefillMode::GpuBatch,
        chunk_len: None,
        reason: None,
    }
}

fn qwen35_hybrid_decode_selection(
    intent: DecodeIntent,
    allow_pipelined: bool,
    supports_pipelined: bool,
) -> DecodeSelection {
    if intent == DecodeIntent::Throughput && allow_pipelined {
        if supports_pipelined {
            return DecodeSelection {
                intent,
                mode: DecodeMode::Pipelined,
                fallback_reason: None,
            };
        }
        return DecodeSelection {
            intent,
            mode: DecodeMode::SingleCb,
            fallback_reason: Some(
                "qwen35 hybrid decode keeps the native single-CB path because pipelined decode is unavailable"
                    .to_string(),
            ),
        };
    }

    DecodeSelection {
        intent,
        mode: DecodeMode::SingleCb,
        fallback_reason: None,
    }
}

impl PrefillExecutionPlan {
    pub(crate) fn for_forward_batch(
        ctx: &ForwardContext<'_>,
        kv: &ModelKv,
        weights: &WeightStore,
        n_tokens: usize,
        emit_all_logits: bool,
    ) -> anyhow::Result<Self> {
        if env_flag_enabled("AX_SERIAL_PREFILL") {
            return Ok(Self {
                mode: PrefillMode::Serial,
                chunk_len: None,
                reason: Some("forced".to_string()),
            });
        }
        if n_tokens <= 1 {
            return Ok(Self {
                mode: PrefillMode::Serial,
                chunk_len: None,
                reason: Some("single_token".to_string()),
            });
        }
        if !ctx.backend.use_gpu_decode() {
            return Ok(Self {
                mode: PrefillMode::Serial,
                chunk_len: None,
                reason: Some("cpu_backend".to_string()),
            });
        }
        if !supports_backend_prefill_kv(ctx.config, kv) {
            return Ok(Self {
                mode: PrefillMode::Serial,
                chunk_len: None,
                reason: Some("cpu_kv".to_string()),
            });
        }
        if ctx.backend.metal_ops().is_none() {
            return Ok(Self {
                mode: PrefillMode::Serial,
                chunk_len: None,
                reason: Some("no_metal".to_string()),
            });
        }

        let lm_weight_name = if weights.has("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let lm_head_dtype_supported = matches!(
            weights.raw_with_dtype(lm_weight_name),
            Ok((_, dtype)) if gpu_batch_logits_supported(dtype)
        );

        let plan = match ctx.config.architecture.as_str() {
            "gemma3" => {
                if let Some(blocker) = gpu_prefill_quant_blocker(ctx.config, weights) {
                    Self {
                        mode: PrefillMode::Serial,
                        chunk_len: None,
                        reason: Some(format!("unsupported_quant:{blocker}")),
                    }
                } else {
                    let chunk_len = Gemma3Forward::gpu_prefill_chunk_len(ctx.config, n_tokens);
                    if emit_all_logits && !lm_head_dtype_supported {
                        Self {
                            mode: PrefillMode::Serial,
                            chunk_len: None,
                            reason: Some("unsupported_lm_head".to_string()),
                        }
                    } else if emit_all_logits && chunk_len.is_some() {
                        Self {
                            mode: PrefillMode::Serial,
                            chunk_len: None,
                            reason: Some("chunked_all_logits_unsupported".to_string()),
                        }
                    } else if let Some(chunk_len) = chunk_len {
                        Self {
                            mode: PrefillMode::GpuChunked,
                            chunk_len: Some(chunk_len),
                            reason: None,
                        }
                    } else {
                        Self {
                            mode: PrefillMode::GpuBatch,
                            chunk_len: None,
                            reason: None,
                        }
                    }
                }
            }
            "qwen35" | "qwen35moe" | "qwen3moe" => {
                if let Some(blocker) = gpu_prefill_quant_blocker(ctx.config, weights) {
                    Self {
                        mode: PrefillMode::Serial,
                        chunk_len: None,
                        reason: Some(format!("unsupported_quant:{blocker}")),
                    }
                } else {
                    qwen_gpu_prefill_plan()
                }
            }
            "gemma4" => {
                if let Some(blocker) = gpu_prefill_quant_blocker(ctx.config, weights) {
                    Self {
                        mode: PrefillMode::Serial,
                        chunk_len: None,
                        reason: Some(format!("unsupported_quant:{blocker}")),
                    }
                } else {
                    let chunk_len = Gemma4Forward::gpu_prefill_chunk_len(ctx.config, n_tokens);
                    if emit_all_logits && !lm_head_dtype_supported {
                        Self {
                            mode: PrefillMode::Serial,
                            chunk_len: None,
                            reason: Some("unsupported_lm_head".to_string()),
                        }
                    } else if emit_all_logits && chunk_len.is_some() {
                        Self {
                            mode: PrefillMode::Serial,
                            chunk_len: None,
                            reason: Some("chunked_all_logits_unsupported".to_string()),
                        }
                    } else if let Some(chunk_len) = chunk_len {
                        Self {
                            mode: PrefillMode::GpuChunked,
                            chunk_len: Some(chunk_len),
                            reason: None,
                        }
                    } else {
                        Self {
                            mode: PrefillMode::GpuBatch,
                            chunk_len: None,
                            reason: None,
                        }
                    }
                }
            }
            _ => Self {
                mode: PrefillMode::Serial,
                chunk_len: None,
                reason: Some("unsupported_arch".to_string()),
            },
        };

        Ok(plan)
    }

    pub fn summary_label(self) -> String {
        match self.mode {
            PrefillMode::Serial => {
                let reason = self.reason.unwrap_or_else(|| "fallback".to_string());
                format!("mode=serial reason={reason}")
            }
            PrefillMode::CpuBatch => "mode=cpu_batch".to_string(),
            PrefillMode::GpuBatch => "mode=gpu_batch".to_string(),
            PrefillMode::GpuChunked => {
                format!("mode=gpu_chunked chunk={}", self.chunk_len.unwrap_or(0))
            }
        }
    }
}

impl PrefillAttentionPlan {
    pub fn label(self) -> &'static str {
        match self {
            Self::BatchLocal => "local",
            Self::BatchLocalF16OutHd128 => "local_f16out_hd128",
            Self::Cached => "cached",
        }
    }
}

impl PrefillWoInputPlan {
    pub fn label(self) -> &'static str {
        match self {
            Self::AttentionOutF32 => "attn_f32",
            Self::MatmulScratchF16 => "scratch_f16",
        }
    }
}

impl GpuBatchPrefillExecutionPlan {
    pub fn summary_label(self, mode: &str, attention_route: &str) -> String {
        let mut summary = format!(
            "mode={mode} kv={} f16_io={} pair={} qkv={} batch_simd={} split_rope={} attn={} window={} wo_in={} attn_route={}",
            kv_precision_label(self.kv_f16, self.kv_q8),
            if self.use_f16_batch_io { "on" } else { "off" },
            if self.use_f16_pair { "on" } else { "off" },
            if self.use_fused_qkv { "fused" } else { "split" },
            if self.use_batch_simd { "on" } else { "off" },
            if self.split_rope_append { "on" } else { "off" },
            self.attention.label(),
            self.attention_sliding_window,
            self.wo_input.label(),
            attention_route,
        );
        if self.q5k_prefill_small_n {
            if self.q5k_prefill_forced_variant {
                summary.push_str(" q5k_prefill=small_n_forced");
            } else {
                summary.push_str(" q5k_prefill=small_n");
            }
        } else if self.q5k_prefill {
            if self.q5k_prefill_forced_variant {
                summary.push_str(" q5k_prefill=base_forced");
            } else {
                summary.push_str(" q5k_prefill=base");
            }
        }
        summary
    }
}

impl DecodeSyncPlan {
    pub fn label(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::SingleCommandBuffer => "single_cb",
            Self::Pipelined => "pipelined",
        }
    }
}

impl DecodeScratchPlan {
    pub fn label(self) -> &'static str {
        match self {
            Self::CpuScratch => "cpu",
            Self::SharedGpuScratch => "gpu_shared",
            Self::HybridBackendOwned => "hybrid_backend",
        }
    }
}

impl DecodeBarrierPlan {
    pub fn label(self) -> &'static str {
        match self {
            Self::Implicit => "implicit",
            Self::Explicit => "explicit",
            Self::Smart => "smart",
        }
    }
}

impl DecodeQkvPlan {
    pub fn label(self) -> &'static str {
        match self {
            Self::Split => "split",
            Self::Fused => "fused",
        }
    }
}

pub(crate) fn select_decode_execution_mode(
    intent: DecodeIntent,
    allow_pipelined: bool,
    has_gpu_decode: bool,
    supports_pipelined: bool,
) -> DecodeSelection {
    if intent == DecodeIntent::Throughput && allow_pipelined {
        if has_gpu_decode && supports_pipelined {
            return DecodeSelection {
                intent,
                mode: DecodeMode::Pipelined,
                fallback_reason: None,
            };
        }
        if has_gpu_decode {
            return DecodeSelection {
                intent,
                mode: DecodeMode::SingleCb,
                fallback_reason: Some(
                    "pipelined decode is unavailable for this model/backend combination"
                        .to_string(),
                ),
            };
        }
        return DecodeSelection {
            intent,
            mode: DecodeMode::Sequential,
            fallback_reason: Some("GPU decode unavailable; using sequential decode".to_string()),
        };
    }

    DecodeSelection {
        intent,
        mode: if has_gpu_decode {
            DecodeMode::SingleCb
        } else {
            DecodeMode::Sequential
        },
        fallback_reason: None,
    }
}

fn single_cb_gpu_decode_plan(
    arch_name: &str,
    metal_ops: &MetalOps,
    gpu_kv: &GpuKv,
    embedding_dim: u32,
    head_dim: u32,
    attend_len: usize,
) -> GpuDecodeExecutionPlan {
    let (dequant_dispatch, attention_dispatch) =
        metal_ops.decode_dispatch_configs_for_attend_len(attend_len as u32);
    let q4_k_selection = metal_ops
        .dequant
        .q4_k_matvec_candidate_with_config(embedding_dim, dequant_dispatch);
    let q5_k_selection = metal_ops
        .dequant
        .q5_k_matvec_candidate_with_config(embedding_dim, dequant_dispatch);
    let q6_k_selection = metal_ops
        .dequant
        .q6_k_matvec_candidate_with_config(embedding_dim, dequant_dispatch);
    let attention_selection =
        attention_dispatch.decode_candidate_selection(gpu_kv.is_f16(), head_dim, attend_len as u32);
    let encoder = if ax_engine_metal::concurrent_decode_enabled() {
        DecodeEncoderPlan::Concurrent
    } else {
        DecodeEncoderPlan::Serial
    };
    // Concurrent dispatch requires barriers — use Smart (per-dispatch conflict
    // detection) for minimal barrier count, matching llama.cpp's strategy.
    let barriers = if encoder == DecodeEncoderPlan::Concurrent {
        DecodeBarrierPlan::Smart
    } else {
        decode_barrier_plan_for_arch(arch_name)
    };
    GpuDecodeExecutionPlan {
        encoder,
        barriers,
        qkv: decode_qkv_plan_for_arch(
            arch_name,
            metal_ops.metal_fused_qkv_enabled(),
            metal_ops.metal_decode_fused_qkv_enabled(),
        ),
        kv_f16: gpu_kv.is_f16(),
        kv_q8: gpu_kv.is_q8(),
        use_pair_matvec: decode_pair_matvec_plan_for_arch(arch_name),
        use_fused_silu_down: decode_fused_silu_down_plan_for_arch(arch_name),
        attention_route: attention_selection.label(),
        attention_tier: attention_selection.stability.label(),
        q4_k_candidate: q4_k_selection.label(),
        q4_k_tier: q4_k_selection.stability.label(),
        q5_k_candidate: q5_k_selection.label(),
        q5_k_tier: q5_k_selection.stability.label(),
        q6_k_candidate: q6_k_selection.label(),
        q6_k_tier: q6_k_selection.stability.label(),
        dequant_dispatch,
        attention_dispatch,
    }
}

fn decode_pair_matvec_plan_for_arch(arch_name: &str) -> bool {
    env_flag_override("AX_METAL_DECODE_PAIR_MATVEC")
        .unwrap_or(matches!(arch_name, "qwen35" | "gemma3" | "gemma4"))
}

fn decode_fused_silu_down_plan_for_arch(arch_name: &str) -> bool {
    // Gemma4 uses GELU, not SiLU — do not include it here.
    env_flag_override("AX_METAL_DECODE_FUSED_SILU_DOWN").unwrap_or(matches!(arch_name, "gemma3"))
}

fn pipelined_gpu_decode_plan(
    arch_name: &str,
    metal_ops: &MetalOps,
    gpu_kv: &GpuKv,
    embedding_dim: u32,
    head_dim: u32,
    attend_len: usize,
) -> GpuDecodeExecutionPlan {
    single_cb_gpu_decode_plan(
        arch_name,
        metal_ops,
        gpu_kv,
        embedding_dim,
        head_dim,
        attend_len,
    )
}

fn prefill_attention_plan(
    base_seq_len: usize,
    sliding_window: u32,
    use_f16_batch_io: bool,
    head_dim: u32,
    prefill_attn_f16out_enabled: bool,
    prefill_use_cached0_enabled: bool,
) -> PrefillAttentionPlan {
    if base_seq_len == 0 && sliding_window == 0 && !prefill_use_cached0_enabled {
        if use_f16_batch_io && head_dim == 128 && prefill_attn_f16out_enabled {
            PrefillAttentionPlan::BatchLocalF16OutHd128
        } else {
            PrefillAttentionPlan::BatchLocal
        }
    } else {
        PrefillAttentionPlan::Cached
    }
}

fn gemma3_prefill_layer_plan(
    config: &ModelConfig,
    layer: usize,
    base_seq_len: usize,
    use_f16_batch_io: bool,
) -> Gemma3PrefillLayerPlan {
    let is_local = Gemma3Forward::use_sliding_window(layer, config);
    let rope_base = if is_local {
        config.rope_freq_base_local.unwrap_or(config.rope_freq_base)
    } else {
        config.rope_freq_base
    };
    let (rope_start, rope_step) = if is_local {
        (base_seq_len as f32, 1.0f32)
    } else {
        config.rope_scaling.scaled_start_step(base_seq_len)
    };
    let sliding_window = if is_local {
        config.sliding_window_size.unwrap_or(0)
    } else {
        0
    };
    Gemma3PrefillLayerPlan {
        attention: prefill_attention_plan(
            base_seq_len,
            sliding_window,
            use_f16_batch_io,
            config.head_dim,
            false,
            false,
        ),
        sliding_window,
        rope_base,
        rope_start,
        rope_step,
    }
}

fn decode_barrier_plan_for_arch(arch_name: &str) -> DecodeBarrierPlan {
    match arch_name {
        "qwen35" | "gemma3" | "gemma4" => {
            if super::shared::metal_decode_barriers_enabled() {
                DecodeBarrierPlan::Explicit
            } else {
                DecodeBarrierPlan::Implicit
            }
        }
        _ => DecodeBarrierPlan::Explicit,
    }
}

fn decode_qkv_plan_for_arch(
    arch_name: &str,
    _fused_qkv_enabled: bool,
    decode_fused_qkv_enabled: bool,
) -> DecodeQkvPlan {
    let use_fused_qkv = match arch_name {
        "gemma3" | "gemma4" => decode_fused_qkv_enabled,
        _ => false,
    };
    if use_fused_qkv {
        DecodeQkvPlan::Fused
    } else {
        DecodeQkvPlan::Split
    }
}

#[cfg(test)]
fn qkv_fusion_supported(wq_dtype: GgmlType, wk_dtype: GgmlType, wv_dtype: GgmlType) -> bool {
    wq_dtype == wk_dtype
        && wq_dtype == wv_dtype
        && matches!(wq_dtype, GgmlType::Q4K | GgmlType::Q6K | GgmlType::Q8_0)
}

#[allow(dead_code)]
#[cfg(test)]
pub(crate) fn qwen3_layer_plan_for_gpu(
    exec_plan: &GpuDecodeExecutionPlan,
    wq_dtype: GgmlType,
    wk_dtype: GgmlType,
    wv_dtype: GgmlType,
    has_bias: bool,
    has_qk_norm: bool,
) -> Qwen3DecodeLayerPlan {
    let use_fused_projection = exec_plan.qkv == DecodeQkvPlan::Fused
        && qkv_fusion_supported(wq_dtype, wk_dtype, wv_dtype)
        && has_qk_norm;
    let qkv = if use_fused_projection {
        DecodeQkvPlan::Fused
    } else {
        DecodeQkvPlan::Split
    };
    let qwen3_post = if use_fused_projection && has_bias {
        Qwen3PrefillQkvPost::FusedBiasQkNorm
    } else if use_fused_projection {
        Qwen3PrefillQkvPost::FusedQkNorm
    } else if has_bias && has_qk_norm {
        Qwen3PrefillQkvPost::SeparateBiasQkNorm
    } else if has_bias {
        Qwen3PrefillQkvPost::SeparateBias
    } else if has_qk_norm {
        Qwen3PrefillQkvPost::SeparateQkNorm
    } else {
        Qwen3PrefillQkvPost::Separate
    };
    Qwen3DecodeLayerPlan { qkv, qwen3_post }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::backend::metal::MetalBackend;
    use crate::model::InferenceModel;
    use crate::model::config::{GateActivation, RopeScaling};
    use std::sync::MutexGuard;

    fn tiny_config(arch: &str) -> crate::model::ModelConfig {
        let is_gemma3 = arch == "gemma3";
        let is_qwen35 = arch == "qwen35";
        crate::model::ModelConfig {
            architecture: arch.to_string(),
            n_layers: if is_qwen35 { 4 } else { 2 },
            n_heads: 2,
            n_kv_heads: 2,
            embedding_dim: 8,
            head_dim: 4,
            intermediate_dim: 16,
            context_length: 32,
            vocab_size: 8,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            has_qkv_bias: false,
            sliding_window_size: if is_gemma3 { Some(1024) } else { None },
            sliding_window_pattern: if is_gemma3 { Some(6) } else { None },
            gate_activation: if is_gemma3 {
                GateActivation::GELU
            } else {
                GateActivation::SiLU
            },
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: RopeScaling::None,
            embed_scale: is_gemma3,
            rope_freq_base_local: if is_gemma3 { Some(5000.0) } else { None },
            n_expert: None,
            n_expert_used: None,
            expert_intermediate_dim: None,
            qwen35_full_attention_interval: is_qwen35.then_some(4),
            qwen35_ssm_conv_kernel: is_qwen35.then_some(4),
            qwen35_ssm_inner_size: is_qwen35.then_some(8),
            qwen35_ssm_state_size: is_qwen35.then_some(2),
            qwen35_ssm_time_step_rank: is_qwen35.then_some(4),
            qwen35_ssm_group_count: is_qwen35.then_some(2),
            gemma4_head_dim_swa: None,
            gemma4_head_dim_global: None,
            gemma4_n_kv_heads_swa: None,
            gemma4_n_kv_heads_global: None,
            gemma4_rope_dim_swa: None,
            gemma4_rope_dim_global: None,
            final_logit_softcapping: None,
        }
    }

    fn env_lock() -> MutexGuard<'static, ()> {
        crate::test_env_lock()
    }

    struct EnvVarRestore {
        values: Vec<(String, Option<std::ffi::OsString>)>,
    }

    impl EnvVarRestore {
        fn many(vars: &[(&str, Option<&str>)]) -> Self {
            Self {
                values: vars
                    .iter()
                    .map(|(key, _)| ((*key).to_string(), std::env::var_os(key)))
                    .collect(),
            }
        }
    }

    impl Drop for EnvVarRestore {
        fn drop(&mut self) {
            for (key, previous) in self.values.iter().rev() {
                match previous {
                    Some(prev) => unsafe {
                        std::env::set_var(key, prev);
                    },
                    None => unsafe {
                        std::env::remove_var(key);
                    },
                }
            }
        }
    }

    fn with_env_vars<T>(vars: &[(&str, Option<&str>)], f: impl FnOnce() -> T) -> T {
        let _guard = env_lock();
        let _restore = EnvVarRestore::many(vars);
        for (key, value) in vars {
            match value {
                Some(value) => unsafe {
                    std::env::set_var(key, value);
                },
                None => unsafe {
                    std::env::remove_var(key);
                },
            }
        }
        f()
    }

    #[test]
    fn test_decode_execution_mode_prefers_pipelined_for_throughput_gpu() {
        let selection = select_decode_execution_mode(DecodeIntent::Throughput, true, true, true);
        assert_eq!(selection.mode, DecodeMode::Pipelined);
        assert!(selection.fallback_reason.is_none());
    }

    #[test]
    fn test_decode_execution_mode_falls_back_to_single_cb_when_pipeline_unavailable() {
        let selection = select_decode_execution_mode(DecodeIntent::Throughput, true, true, false);
        assert_eq!(selection.mode, DecodeMode::SingleCb);
        assert!(selection.fallback_reason.is_some());
    }

    #[test]
    fn test_decode_execution_mode_uses_sequential_without_gpu() {
        let selection = select_decode_execution_mode(DecodeIntent::Latency, true, false, false);
        assert_eq!(selection.mode, DecodeMode::Sequential);
        assert!(selection.fallback_reason.is_none());
    }

    #[test]
    fn test_supports_backend_prefill_kv_accepts_qwen35_hybrid_kv() {
        let cfg = tiny_config("qwen35");
        let kv = ModelKv::Qwen35(Box::new(crate::kv::Qwen3_5Kv::new(
            4, 2, 4, 32, 4, 4, 8, 2, 4, 2,
        )));
        assert!(supports_backend_prefill_kv(&cfg, &kv));
    }

    #[test]
    fn test_decode_scratch_plan_hybrid_backend_label() {
        assert_eq!(
            DecodeScratchPlan::HybridBackendOwned.label(),
            "hybrid_backend"
        );
    }

    #[test]
    fn test_qwen_gpu_prefill_plan_uses_gpu_batch() {
        let plan = qwen_gpu_prefill_plan();
        assert_eq!(plan.mode, PrefillMode::GpuBatch);
        assert_eq!(plan.chunk_len, None);
        assert_eq!(plan.reason, None);
    }

    #[test]
    fn test_prefill_execution_plan_summary_label_for_cpu_batch_mode() {
        let summary = PrefillExecutionPlan {
            mode: PrefillMode::CpuBatch,
            chunk_len: None,
            reason: None,
        }
        .summary_label();
        assert_eq!(summary, "mode=cpu_batch");
    }

    #[test]
    fn test_decode_execution_plan_summary_label_includes_attention_tier() {
        let plan = DecodeExecutionPlan {
            selection: DecodeSelection {
                intent: DecodeIntent::Latency,
                mode: DecodeMode::SingleCb,
                fallback_reason: None,
            },
            sync: DecodeSyncPlan::SingleCommandBuffer,
            scratch: DecodeScratchPlan::SharedGpuScratch,
            gpu: Some(GpuDecodeExecutionPlan {
                encoder: DecodeEncoderPlan::Serial,
                barriers: DecodeBarrierPlan::Implicit,
                qkv: DecodeQkvPlan::Fused,
                kv_f16: true,
                kv_q8: false,

                use_pair_matvec: false,
                use_fused_silu_down: false,
                attention_route: "splitk_hd256",
                attention_tier: "profile_preferred",
                q4_k_candidate: "q4_k.nr2",
                q4_k_tier: "profile_preferred",
                q5_k_candidate: "q5_k.base",
                q5_k_tier: "stable",
                q6_k_candidate: "q6_k.base",
                q6_k_tier: "stable",
                dequant_dispatch: ax_engine_metal::DequantDispatchConfig::default(),
                attention_dispatch: ax_engine_metal::AttentionDispatchConfig::default(),
            }),
        };

        let summary = plan.summary_label();
        assert!(summary.contains("attn=splitk_hd256/profile_preferred"));
        assert!(summary.contains("q4k=q4_k.nr2/profile_preferred"));
        assert!(summary.contains("q5k=q5_k.base/stable"));
        assert!(summary.contains("q6k=q6_k.base/stable"));
    }

    #[test]
    fn test_decode_execution_plan_summary_label_reports_q8_kv() {
        let plan = DecodeExecutionPlan {
            selection: DecodeSelection {
                intent: DecodeIntent::Latency,
                mode: DecodeMode::SingleCb,
                fallback_reason: None,
            },
            sync: DecodeSyncPlan::SingleCommandBuffer,
            scratch: DecodeScratchPlan::SharedGpuScratch,
            gpu: Some(GpuDecodeExecutionPlan {
                encoder: DecodeEncoderPlan::Serial,
                barriers: DecodeBarrierPlan::Implicit,
                qkv: DecodeQkvPlan::Fused,
                kv_f16: false,
                kv_q8: true,
                use_pair_matvec: false,
                use_fused_silu_down: false,
                attention_route: "splitk_hd256",
                attention_tier: "profile_preferred",
                q4_k_candidate: "q4_k.nr2",
                q4_k_tier: "profile_preferred",
                q5_k_candidate: "q5_k.base",
                q5_k_tier: "stable",
                q6_k_candidate: "q6_k.base",
                q6_k_tier: "stable",
                dequant_dispatch: ax_engine_metal::DequantDispatchConfig::default(),
                attention_dispatch: ax_engine_metal::AttentionDispatchConfig::default(),
            }),
        };

        let summary = plan.summary_label();
        assert!(summary.contains("kv=q8_0"));
    }

    #[test]
    fn test_decode_qkv_plan_for_gemma3_uses_decode_fused_toggle() {
        let plan = decode_qkv_plan_for_arch("gemma3", false, true);
        assert_eq!(plan, DecodeQkvPlan::Fused);
    }

    #[test]
    fn test_decode_pair_matvec_plan_defaults_on_for_qwen35() {
        let _guard = env_lock();
        unsafe { std::env::remove_var("AX_METAL_DECODE_PAIR_MATVEC") };
        assert!(decode_pair_matvec_plan_for_arch("qwen35"));
    }

    #[test]
    fn test_decode_fused_silu_down_plan_defaults_off_for_qwen35() {
        let _guard = env_lock();
        unsafe { std::env::remove_var("AX_METAL_DECODE_FUSED_SILU_DOWN") };
        assert!(!decode_fused_silu_down_plan_for_arch("qwen35"));
    }

    #[test]
    fn test_decode_pair_matvec_plan_respects_env_override_for_qwen35() {
        let _guard = env_lock();
        unsafe { std::env::set_var("AX_METAL_DECODE_PAIR_MATVEC", "off") };
        let plan = decode_pair_matvec_plan_for_arch("qwen35");
        unsafe { std::env::remove_var("AX_METAL_DECODE_PAIR_MATVEC") };
        assert!(!plan);
    }

    #[test]
    fn test_decode_fused_silu_down_plan_respects_env_override_for_qwen35() {
        let _guard = env_lock();
        unsafe { std::env::set_var("AX_METAL_DECODE_FUSED_SILU_DOWN", "off") };
        let plan = decode_fused_silu_down_plan_for_arch("qwen35");
        unsafe { std::env::remove_var("AX_METAL_DECODE_FUSED_SILU_DOWN") };
        assert!(!plan);
    }

    #[test]
    fn test_qwen3_layer_plan_uses_fused_qkv_when_supported() {
        let exec_plan = GpuDecodeExecutionPlan {
            encoder: DecodeEncoderPlan::Serial,
            barriers: DecodeBarrierPlan::Explicit,
            qkv: DecodeQkvPlan::Fused,
            kv_f16: true,
            kv_q8: false,
            use_pair_matvec: true,
            use_fused_silu_down: true,
            attention_route: "f16kv_hd128",
            attention_tier: "stable",
            q4_k_candidate: "q4_k.nr2",
            q4_k_tier: "profile_preferred",
            q5_k_candidate: "q5_k.ilp4",
            q5_k_tier: "stable",
            q6_k_candidate: "q6_k.nr2",
            q6_k_tier: "profile_preferred",
            dequant_dispatch: ax_engine_metal::DequantDispatchConfig::default(),
            attention_dispatch: ax_engine_metal::AttentionDispatchConfig::default(),
        };
        let layer = qwen3_layer_plan_for_gpu(
            &exec_plan,
            GgmlType::Q4K,
            GgmlType::Q4K,
            GgmlType::Q4K,
            true,
            true,
        );
        assert_eq!(layer.qkv, DecodeQkvPlan::Fused);
        assert_eq!(layer.qwen3_post, Qwen3PrefillQkvPost::FusedBiasQkNorm);
    }

    #[test]
    fn test_qwen3_layer_plan_falls_back_when_qk_norm_missing() {
        let exec_plan = GpuDecodeExecutionPlan {
            encoder: DecodeEncoderPlan::Serial,
            barriers: DecodeBarrierPlan::Explicit,
            qkv: DecodeQkvPlan::Fused,
            kv_f16: true,
            kv_q8: false,
            use_pair_matvec: true,
            use_fused_silu_down: true,
            attention_route: "f16kv_hd128",
            attention_tier: "stable",
            q4_k_candidate: "q4_k.nr2",
            q4_k_tier: "profile_preferred",
            q5_k_candidate: "q5_k.ilp4",
            q5_k_tier: "stable",
            q6_k_candidate: "q6_k.nr2",
            q6_k_tier: "profile_preferred",
            dequant_dispatch: ax_engine_metal::DequantDispatchConfig::default(),
            attention_dispatch: ax_engine_metal::AttentionDispatchConfig::default(),
        };
        let layer = qwen3_layer_plan_for_gpu(
            &exec_plan,
            GgmlType::Q4K,
            GgmlType::Q4K,
            GgmlType::Q4K,
            true,
            false,
        );
        assert_eq!(layer.qkv, DecodeQkvPlan::Split);
        assert_eq!(layer.qwen3_post, Qwen3PrefillQkvPost::SeparateBias);
    }

    #[test]
    fn test_decode_barrier_plan_for_glm_stays_explicit() {
        let plan = decode_barrier_plan_for_arch("glm");
        assert_eq!(plan, DecodeBarrierPlan::Explicit);
    }

    #[test]
    fn test_decode_barrier_plan_for_qwen35_defaults_implicit() {
        let _guard = env_lock();
        let _restore = EnvVarRestore::many(&[("AX_METAL_DECODE_BARRIERS", Some(""))]);
        unsafe {
            std::env::remove_var("AX_METAL_DECODE_BARRIERS");
        }
        let plan = decode_barrier_plan_for_arch("qwen35");
        assert_eq!(plan, DecodeBarrierPlan::Implicit);
    }

    #[test]
    fn test_decode_barrier_plan_for_qwen35_respects_env_override() {
        let plan = with_env_vars(&[("AX_METAL_DECODE_BARRIERS", Some("on"))], || {
            decode_barrier_plan_for_arch("qwen35")
        });
        assert_eq!(plan, DecodeBarrierPlan::Explicit);
    }

    #[test]
    fn test_qwen3_prefill_records_cached_attention_window() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model = InferenceModel::with_backend(tiny_config("gemma3"), Box::new(backend)).unwrap();
        let kv = model.create_model_kv();
        let Some(gpu_kv) = kv.as_gpu() else {
            return;
        };
        let Some(metal_ops) = model.metal_ops() else {
            return;
        };

        let plan =
            DecodeExecutionPlan::qwen3_prefill(metal_ops, gpu_kv, 0, 128, 128, 4096, false, false);
        assert_eq!(plan.attention, PrefillAttentionPlan::Cached);
        assert_eq!(plan.attention_sliding_window, 4096);
        assert_eq!(plan.wo_input, PrefillWoInputPlan::AttentionOutF32);
    }

    #[test]
    fn test_q5k_prefill_plan_forces_conservative_batch_knobs() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model = InferenceModel::with_backend(tiny_config("gemma3"), Box::new(backend)).unwrap();
        let kv = model.create_model_kv();
        let Some(gpu_kv) = kv.as_gpu() else {
            return;
        };
        let Some(metal_ops) = model.metal_ops() else {
            return;
        };

        let plan = with_env_vars(
            &[
                ("AX_METAL_Q5K_PREFILL_VARIANT", None),
                ("AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT", None),
            ],
            || DecodeExecutionPlan::qwen3_prefill(metal_ops, gpu_kv, 0, 8, 128, 4096, true, true),
        );
        assert!(!plan.use_f16_batch_io);
        assert!(!plan.use_f16_pair);
        assert!(!plan.use_batch_simd);
        assert!(plan.q5k_prefill);
        assert!(plan.q5k_prefill_small_n);
        assert!(
            plan.summary_label("gpu_batch", "cache/stable")
                .contains("q5k_prefill=small_n")
        );
    }

    #[test]
    fn test_q5k_prefill_plan_uses_base_route_for_larger_batches() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model = InferenceModel::with_backend(tiny_config("gemma3"), Box::new(backend)).unwrap();
        let kv = model.create_model_kv();
        let Some(gpu_kv) = kv.as_gpu() else {
            return;
        };
        let Some(metal_ops) = model.metal_ops() else {
            return;
        };

        let plan = with_env_vars(
            &[
                ("AX_METAL_Q5K_PREFILL_VARIANT", None),
                ("AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT", None),
            ],
            || DecodeExecutionPlan::qwen3_prefill(metal_ops, gpu_kv, 0, 16, 128, 4096, true, true),
        );
        assert!(plan.q5k_prefill);
        assert!(!plan.q5k_prefill_small_n);
        // PRD-PREFILL-DISPATCH-CONSOLIDATION: blocked kernel for all quant types.
        assert!(!plan.use_f16_batch_io);
        assert!(
            plan.summary_label("gpu_batch", "cache/stable")
                .contains("q5k_prefill=base")
        );
    }

    #[test]
    fn test_q5k_prefill_plan_uses_base_route_below_small_batch_window() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model = InferenceModel::with_backend(tiny_config("gemma3"), Box::new(backend)).unwrap();
        let kv = model.create_model_kv();
        let Some(gpu_kv) = kv.as_gpu() else {
            return;
        };
        let Some(metal_ops) = model.metal_ops() else {
            return;
        };

        let plan = with_env_vars(
            &[
                ("AX_METAL_Q5K_PREFILL_VARIANT", None),
                ("AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT", None),
            ],
            || DecodeExecutionPlan::qwen3_prefill(metal_ops, gpu_kv, 0, 2, 128, 4096, true, true),
        );
        assert!(plan.q5k_prefill);
        assert!(!plan.q5k_prefill_small_n);
        // PRD-PREFILL-DISPATCH-CONSOLIDATION: blocked kernel for all quant types.
        assert!(!plan.use_f16_batch_io);
        assert!(
            plan.summary_label("gpu_batch", "cache/stable")
                .contains("q5k_prefill=base")
        );
    }

    #[test]
    fn test_q5k_prefill_plan_uses_base_route_when_auto_is_not_eligible() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model = InferenceModel::with_backend(tiny_config("gemma3"), Box::new(backend)).unwrap();
        let kv = model.create_model_kv();
        let Some(gpu_kv) = kv.as_gpu() else {
            return;
        };
        let Some(metal_ops) = model.metal_ops() else {
            return;
        };

        let plan = with_env_vars(
            &[
                ("AX_METAL_Q5K_PREFILL_VARIANT", None),
                ("AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT", None),
            ],
            || DecodeExecutionPlan::qwen3_prefill(metal_ops, gpu_kv, 0, 16, 128, 4096, true, false),
        );
        assert!(plan.q5k_prefill);
        assert!(!plan.q5k_prefill_small_n);
        // PRD-PREFILL-DISPATCH-CONSOLIDATION: blocked kernel for all quant types.
        assert!(!plan.use_f16_batch_io);
        assert!(
            plan.summary_label("gpu_batch", "cache/stable")
                .contains("q5k_prefill=base")
        );
    }

    #[test]
    fn test_q5k_prefill_plan_can_force_base_variant() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model = InferenceModel::with_backend(tiny_config("gemma3"), Box::new(backend)).unwrap();
        let kv = model.create_model_kv();
        let Some(gpu_kv) = kv.as_gpu() else {
            return;
        };
        let Some(metal_ops) = model.metal_ops() else {
            return;
        };

        let plan = with_env_vars(
            &[
                ("AX_METAL_Q5K_PREFILL_VARIANT", Some("base")),
                ("AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT", None),
            ],
            || DecodeExecutionPlan::qwen3_prefill(metal_ops, gpu_kv, 0, 8, 128, 4096, true, true),
        );
        assert!(plan.q5k_prefill);
        assert!(!plan.q5k_prefill_small_n);
        assert!(plan.q5k_prefill_forced_variant);
        // PRD-PREFILL-DISPATCH-CONSOLIDATION: blocked kernel for all quant types.
        assert!(!plan.use_f16_batch_io);
        assert!(
            plan.summary_label("gpu_batch", "cache/stable")
                .contains("q5k_prefill=base_forced")
        );
    }

    #[test]
    fn test_q5k_prefill_plan_can_force_small_variant() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model = InferenceModel::with_backend(tiny_config("gemma3"), Box::new(backend)).unwrap();
        let kv = model.create_model_kv();
        let Some(gpu_kv) = kv.as_gpu() else {
            return;
        };
        let Some(metal_ops) = model.metal_ops() else {
            return;
        };

        let plan = with_env_vars(
            &[
                ("AX_METAL_Q5K_PREFILL_VARIANT", Some("small")),
                ("AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT", None),
            ],
            || {
                DecodeExecutionPlan::qwen3_prefill(
                    metal_ops, gpu_kv, 0, 128, 128, 4096, true, false,
                )
            },
        );
        assert!(plan.q5k_prefill);
        assert!(plan.q5k_prefill_small_n);
        assert!(plan.q5k_prefill_forced_variant);
        assert!(!plan.use_f16_batch_io);
        assert!(
            plan.summary_label("gpu_batch", "cache/stable")
                .contains("q5k_prefill=small_n_forced")
        );
    }

    #[test]
    fn test_prefill_attention_plan_uses_cached_when_sliding_window_is_nonzero() {
        let plan = prefill_attention_plan(0, 4096, false, 128, false, false);
        assert_eq!(plan, PrefillAttentionPlan::Cached);
    }

    #[test]
    fn test_prefill_attention_plan_uses_cached_when_cached0_is_forced() {
        let plan = prefill_attention_plan(0, 0, true, 128, true, true);
        assert_eq!(plan, PrefillAttentionPlan::Cached);
    }

    #[test]
    fn test_gemma3_prefill_layer_plan_uses_cached_local_attention() {
        let config = tiny_config("gemma3");
        let plan = gemma3_prefill_layer_plan(&config, 0, 0, false);
        assert_eq!(plan.attention, PrefillAttentionPlan::Cached);
        assert_eq!(plan.sliding_window, config.sliding_window_size.unwrap_or(0));
        assert_eq!(plan.rope_base, config.rope_freq_base_local.unwrap());
    }

    #[test]
    fn test_gemma3_prefill_layer_plan_uses_batch_local_for_global_layer_without_prefix() {
        let config = tiny_config("gemma3");
        let global_layer = (config.sliding_window_pattern.unwrap() - 1) as usize;
        let plan = gemma3_prefill_layer_plan(&config, global_layer, 0, false);
        assert_eq!(plan.attention, PrefillAttentionPlan::BatchLocal);
        assert_eq!(plan.sliding_window, 0);
        assert_eq!(plan.rope_base, config.rope_freq_base);
    }

    #[test]
    fn test_gpu_batch_prefill_summary_label_includes_route_and_attention() {
        let plan = GpuBatchPrefillExecutionPlan {
            kv_f16: true,
            kv_q8: false,
            use_f16_batch_io: true,
            use_f16_pair: false,
            use_fused_qkv: true,
            use_batch_simd: false,
            q5k_prefill: false,
            q5k_prefill_small_n: false,
            q5k_prefill_forced_variant: false,
            split_rope_append: true,
            attention: PrefillAttentionPlan::Cached,
            attention_sliding_window: 512,
            wo_input: PrefillWoInputPlan::AttentionOutF32,
            attention_dispatch: ax_engine_metal::AttentionDispatchConfig::default(),
        };
        let summary = plan.summary_label("gpu_batch", "cache/stable");
        assert!(summary.contains("mode=gpu_batch"));
        assert!(summary.contains("attn=cached"));
        assert!(summary.contains("window=512"));
        assert!(summary.contains("wo_in=attn_f32"));
        assert!(summary.contains("attn_route=cache/stable"));
    }

    #[test]
    fn test_gpu_batch_prefill_summary_label_reports_q8_kv() {
        let plan = GpuBatchPrefillExecutionPlan {
            kv_f16: false,
            kv_q8: true,
            use_f16_batch_io: false,
            use_f16_pair: false,
            use_fused_qkv: false,
            use_batch_simd: false,
            q5k_prefill: false,
            q5k_prefill_small_n: false,
            q5k_prefill_forced_variant: false,
            split_rope_append: false,
            attention: PrefillAttentionPlan::Cached,
            attention_sliding_window: 0,
            wo_input: PrefillWoInputPlan::AttentionOutF32,
            attention_dispatch: ax_engine_metal::AttentionDispatchConfig::default(),
        };
        let summary = plan.summary_label("gpu_batch", "cache/stable");
        assert!(summary.contains("kv=q8_0"));
    }

    #[test]
    fn test_prefill_execution_plan_summary_label_for_chunked_mode() {
        let summary = PrefillExecutionPlan {
            mode: PrefillMode::GpuChunked,
            chunk_len: Some(1024),
            reason: None,
        }
        .summary_label();
        assert_eq!(summary, "mode=gpu_chunked chunk=1024");
    }

    #[test]
    fn test_prefill_execution_plan_summary_label_includes_quant_blocker_detail() {
        let summary = PrefillExecutionPlan {
            mode: PrefillMode::Serial,
            chunk_len: None,
            reason: Some("unsupported_quant:blk.10.attn_v.weight:Q5K".to_string()),
        }
        .summary_label();
        assert_eq!(
            summary,
            "mode=serial reason=unsupported_quant:blk.10.attn_v.weight:Q5K"
        );
    }

    #[test]
    fn test_qwen3_prefill_ffn_layer_limits_q8_pair_to_small_batches() {
        let prefill_plan = GpuBatchPrefillExecutionPlan {
            kv_f16: true,
            kv_q8: false,
            use_f16_batch_io: true,
            use_f16_pair: false,
            use_fused_qkv: false,
            use_batch_simd: false,
            q5k_prefill: false,
            q5k_prefill_small_n: false,
            q5k_prefill_forced_variant: false,
            split_rope_append: false,
            attention: PrefillAttentionPlan::Cached,
            attention_sliding_window: 256,
            wo_input: PrefillWoInputPlan::AttentionOutF32,
            attention_dispatch: ax_engine_metal::AttentionDispatchConfig::default(),
        };
        let small = DecodeExecutionPlan::qwen3_prefill_ffn_layer(
            &prefill_plan,
            GgmlType::Q8_0,
            GgmlType::Q8_0,
            128,
        );
        let large = DecodeExecutionPlan::qwen3_prefill_ffn_layer(
            &prefill_plan,
            GgmlType::Q8_0,
            GgmlType::Q8_0,
            512,
        );
        assert!(small.use_pair_kernel);
        assert!(!large.use_pair_kernel);
        assert_eq!(small.input, PrefillProjectionInputPlan::MatmulScratchF16);
        assert_eq!(large.input, PrefillProjectionInputPlan::MatmulScratchF16);
        assert_eq!(small.activation, PrefillFfnActivationPlan::SiluMulGateF32);
        assert_eq!(large.activation, PrefillFfnActivationPlan::SiluMulGateF32);
    }

    #[test]
    fn test_gemma3_prefill_ffn_layer_requires_explicit_pair_toggle() {
        let prefill_plan = GpuBatchPrefillExecutionPlan {
            kv_f16: true,
            kv_q8: false,
            use_f16_batch_io: true,
            use_f16_pair: false,
            use_fused_qkv: false,
            use_batch_simd: false,
            q5k_prefill: false,
            q5k_prefill_small_n: false,
            q5k_prefill_forced_variant: false,
            split_rope_append: false,
            attention: PrefillAttentionPlan::Cached,
            attention_sliding_window: 0,
            wo_input: PrefillWoInputPlan::AttentionOutF32,
            attention_dispatch: ax_engine_metal::AttentionDispatchConfig::default(),
        };
        let layer = DecodeExecutionPlan::gemma3_prefill_ffn_layer(
            &prefill_plan,
            GgmlType::Q8_0,
            GgmlType::Q8_0,
        );
        assert_eq!(layer.input, PrefillProjectionInputPlan::MatmulScratchF16);
        assert!(!layer.use_pair_kernel);
        assert_eq!(layer.activation, PrefillFfnActivationPlan::GeluMulGateF32);
    }

    #[test]
    fn test_qwen3_prefill_residual_handoff_stays_f32() {
        assert_eq!(
            DecodeExecutionPlan::qwen3_prefill_residual_handoff(false),
            PrefillResidualHandoffPlan::ResidualAddRmsNormF32
        );
        assert_eq!(
            DecodeExecutionPlan::qwen3_prefill_residual_handoff(true),
            PrefillResidualHandoffPlan::ResidualOnly
        );
    }

    #[test]
    fn test_gemma3_prefill_residual_handoff_stays_f32() {
        assert_eq!(
            DecodeExecutionPlan::gemma3_prefill_residual_handoff(false),
            PrefillResidualHandoffPlan::ResidualAddRmsNormF32
        );
        assert_eq!(
            DecodeExecutionPlan::gemma3_prefill_residual_handoff(true),
            PrefillResidualHandoffPlan::ResidualOnly
        );
    }

    #[test]
    fn test_prefill_logits_plan_uses_batch_path_for_all_logits() {
        assert_eq!(
            DecodeExecutionPlan::prefill_logits_plan(true),
            PrefillLogitsPlan::BatchAllLogits
        );
    }

    #[test]
    fn test_prefill_logits_plan_uses_last_token_path_for_default_prefill() {
        assert_eq!(
            DecodeExecutionPlan::prefill_logits_plan(false),
            PrefillLogitsPlan::LastTokenMatvec
        );
    }

    #[test]
    fn test_qwen3_prefill_qkv_layer_reports_fused_bias_post_path() {
        let prefill_plan = GpuBatchPrefillExecutionPlan {
            kv_f16: true,
            kv_q8: false,
            use_f16_batch_io: false,
            use_f16_pair: false,
            use_fused_qkv: true,
            use_batch_simd: false,
            q5k_prefill: false,
            q5k_prefill_small_n: false,
            q5k_prefill_forced_variant: false,
            split_rope_append: false,
            attention: PrefillAttentionPlan::Cached,
            attention_sliding_window: 128,
            wo_input: PrefillWoInputPlan::AttentionOutF32,
            attention_dispatch: ax_engine_metal::AttentionDispatchConfig::default(),
        };
        let layer = DecodeExecutionPlan::qwen3_prefill_qkv_layer(
            &prefill_plan,
            GgmlType::Q4K,
            GgmlType::Q4K,
            GgmlType::Q4K,
            true,
            true,
        );
        assert_eq!(layer.input, PrefillProjectionInputPlan::NormBufF32);
        assert!(layer.use_fused_projection);
        assert_eq!(layer.qwen3_post, Qwen3PrefillQkvPost::FusedBiasQkNorm);
    }

    #[test]
    fn test_qwen3_prefill_qkv_layer_reports_fused_qknorm_post_path_without_bias() {
        let prefill_plan = GpuBatchPrefillExecutionPlan {
            kv_f16: true,
            kv_q8: false,
            use_f16_batch_io: false,
            use_f16_pair: false,
            use_fused_qkv: true,
            use_batch_simd: false,
            q5k_prefill: false,
            q5k_prefill_small_n: false,
            q5k_prefill_forced_variant: false,
            split_rope_append: false,
            attention: PrefillAttentionPlan::Cached,
            attention_sliding_window: 128,
            wo_input: PrefillWoInputPlan::AttentionOutF32,
            attention_dispatch: ax_engine_metal::AttentionDispatchConfig::default(),
        };
        let layer = DecodeExecutionPlan::qwen3_prefill_qkv_layer(
            &prefill_plan,
            GgmlType::Q6K,
            GgmlType::Q6K,
            GgmlType::Q6K,
            false,
            true,
        );
        assert_eq!(layer.input, PrefillProjectionInputPlan::NormBufF32);
        assert!(layer.use_fused_projection);
        assert_eq!(layer.qwen3_post, Qwen3PrefillQkvPost::FusedQkNorm);
    }

    #[test]
    fn test_qwen3_prefill_qkv_layer_reports_separate_bias_qknorm_variant() {
        let prefill_plan = GpuBatchPrefillExecutionPlan {
            kv_f16: true,
            kv_q8: false,
            use_f16_batch_io: false,
            use_f16_pair: false,
            use_fused_qkv: false,
            use_batch_simd: false,
            q5k_prefill: false,
            q5k_prefill_small_n: false,
            q5k_prefill_forced_variant: false,
            split_rope_append: false,
            attention: PrefillAttentionPlan::Cached,
            attention_sliding_window: 128,
            wo_input: PrefillWoInputPlan::AttentionOutF32,
            attention_dispatch: ax_engine_metal::AttentionDispatchConfig::default(),
        };
        let layer = DecodeExecutionPlan::qwen3_prefill_qkv_layer(
            &prefill_plan,
            GgmlType::Q4K,
            GgmlType::Q4K,
            GgmlType::Q4K,
            true,
            true,
        );
        assert_eq!(layer.input, PrefillProjectionInputPlan::NormBufF32);
        assert!(!layer.use_fused_projection);
        assert_eq!(layer.qwen3_post, Qwen3PrefillQkvPost::SeparateBiasQkNorm);
    }

    #[test]
    fn test_gemma3_prefill_layer_plan_uses_yarn_fallback_only_on_global_layers() {
        let mut config = tiny_config("gemma3");
        config.rope_scaling = RopeScaling::Yarn {
            factor: 4.0,
            ext_factor: 1.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            orig_ctx_len: 8192,
        };

        let local = gemma3_prefill_layer_plan(&config, 0, 128, false);
        assert_eq!(local.rope_start, 128.0);
        assert_eq!(local.rope_step, 1.0);

        let global = gemma3_prefill_layer_plan(&config, 5, 128, false);
        assert!((global.rope_start - 32.0).abs() < 1e-6);
        assert!((global.rope_step - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_for_model_gemma3_reports_plan_for_gpu_decode() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model = InferenceModel::with_backend(tiny_config("gemma3"), Box::new(backend)).unwrap();
        let kv = model.create_model_kv();
        if !kv.is_gpu() {
            return;
        }

        let plan = DecodeExecutionPlan::for_model(&model, &kv, DecodeIntent::Latency, false);
        assert!(plan.gpu.is_some());
        assert_eq!(plan.sync, DecodeSyncPlan::SingleCommandBuffer);
    }

    #[test]
    fn test_for_model_gemma3_reports_fused_qkv_plan_after_backend_configuration() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        backend
            .configure_for_model("gemma-3-12b-it", "Q4_K", "gemma3")
            .unwrap();
        let model = InferenceModel::with_backend(tiny_config("gemma3"), Box::new(backend)).unwrap();
        let kv = model.create_model_kv();
        if !kv.is_gpu() {
            return;
        }

        let plan = DecodeExecutionPlan::for_model(&model, &kv, DecodeIntent::Latency, false);
        assert_eq!(plan.gpu.unwrap().qkv, DecodeQkvPlan::Fused);
    }
}
