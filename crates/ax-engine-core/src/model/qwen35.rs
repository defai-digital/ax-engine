//! Qwen3.5 hybrid forward pass.
//!
//! The architecture alternates recurrent GDN layers with periodic full
//! attention layers. This implementation follows the upstream structure from
//! `llama.cpp` and `mistral.rs`:
//! - every layer uses `attn_norm`
//! - recurrent layers use fused `attn_qkv` + `attn_gate` + `ssm_*`
//! - full-attention layers use doubled Q projection (`q + gate`)
//! - every layer uses `post_attention_norm` before the FFN
//!
//! AX now routes the recurrent projections and recurrent primitives through the
//! active backend, including Metal kernels for the causal-conv and gated-delta
//! steps. The hybrid `ModelKv::Qwen35` state remains CPU-owned today, so Metal
//! recurrent execution still copies state through backend-owned kernels rather
//! than keeping it GPU-resident end to end.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use ax_engine_metal::MetalBuffer;

use crate::compute::attention::{self, AttentionParams};
use crate::compute::gdn;
use crate::compute::rms_norm;
use crate::compute::rope;
use crate::compute::silu;
use crate::kv::ModelKv;
use crate::metrics::OpBreakdown;
use crate::metrics::counters::OpTimer;
use crate::model::config::ModelConfig;
use crate::model::forward::{ForwardContext, ForwardPass};
use crate::model::shared::{
    encode_dequant_batch, encode_dequant_batch_f16in, env_flag_enabled, env_flag_override,
    per_head_rms_norm,
};
use crate::model::weights::WeightStore;

macro_rules! timed {
    ($ops:expr, $field:ident, $body:expr) => {{
        if let Some(ref mut ops) = $ops {
            let _t = OpTimer::start();
            let _r = $body;
            ops.$field += _t.elapsed();
            _r
        } else {
            $body
        }
    }};
}

macro_rules! timed_matmul_bucket {
    ($ops:expr, $field:ident, $body:expr) => {{
        if let Some(ref mut ops) = $ops {
            let _t = OpTimer::start();
            let _r = $body;
            let _elapsed = _t.elapsed();
            ops.matmul += _elapsed;
            ops.$field += _elapsed;
            _r
        } else {
            $body
        }
    }};
}

#[derive(Debug)]
pub struct Qwen35Forward;

/// Cached GPU weight buffer keys for Qwen3.5 recurrent (GDN) layers.
/// Used by the GPU unified decode path to avoid per-call weight lookups.
#[derive(Clone)]
struct Qwen35RecurrentLayerKeys {
    /// Fused QKV+gate projection (attn_qkv.weight).
    wqkv: usize,
    wqkv_dtype: crate::gguf::tensor::GgmlType,
    /// Gate projection (attn_gate.weight → Z).
    wgate: usize,
    wgate_dtype: crate::gguf::tensor::GgmlType,
    /// SSM beta projection.
    wbeta: usize,
    wbeta_dtype: crate::gguf::tensor::GgmlType,
    /// SSM alpha projection.
    walpha: usize,
    walpha_dtype: crate::gguf::tensor::GgmlType,
    /// SSM output projection (ssm_out.weight).
    wssm_out: usize,
    wssm_out_dtype: crate::gguf::tensor::GgmlType,
    /// Cached causal conv kernel for single-token GPU recurrent decode.
    conv_kernel: usize,
    /// Cached recurrent norm tensor for the GPU finalize step.
    ssm_norm: usize,
}

/// Per-layer cached keys for Qwen3.5 GPU unified decode.
#[derive(Clone)]
enum Qwen35GpuLayerKeys {
    /// Full-attention layer — uses standard CachedLayerKeys fields.
    FullAttention,
    /// Recurrent GDN layer — additional SSM weight keys.
    Recurrent(Qwen35RecurrentLayerKeys),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen35LayerType {
    FullAttention,
    RecurrentGdn,
}

#[derive(Debug, Clone, Copy)]
struct Qwen35RecurrentDims {
    conv_kernel: usize,
    inner_size: usize,
    state_size: usize,
    time_step_rank: usize,
    group_count: usize,
}

impl Qwen35RecurrentDims {
    fn key_dim(self) -> usize {
        self.group_count * self.state_size
    }

    fn value_dim(self) -> usize {
        self.inner_size
    }

    fn conv_dim(self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
}

type QuantOp<'a> = (&'a [u8], crate::gguf::tensor::GgmlType, usize);
type FusedQuantData = Arc<[u8]>;

enum Qwen35FullAttentionInputPlan<'a> {
    Split([QuantOp<'a>; 3]),
    Fused {
        raw: FusedQuantData,
        dtype: crate::gguf::tensor::GgmlType,
        rows: usize,
    },
}

#[derive(Clone, Copy)]
struct Qwen35AttentionNormWeights<'a> {
    q: &'a [f32],
    k: &'a [f32],
}

#[derive(Clone, Copy)]
struct Qwen35RecurrentRuntimeTensors<'a> {
    dt_bias: &'a [f32],
    a: &'a [f32],
    conv_kernel: &'a [f32],
    ssm_norm: &'a [f32],
}

impl Qwen35Forward {
    fn full_attention_fused_quant_cache()
    -> &'static Mutex<HashMap<(usize, usize, usize), FusedQuantData>> {
        static CACHE: OnceLock<Mutex<HashMap<(usize, usize, usize), FusedQuantData>>> =
            OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn fused_quant_rows_cached(wq: &[u8], wk: &[u8], wv: &[u8]) -> FusedQuantData {
        let key = (
            wq.as_ptr() as usize,
            wk.as_ptr() as usize,
            wv.as_ptr() as usize,
        );
        let cache = Self::full_attention_fused_quant_cache();
        let mut cache = cache
            .lock()
            .expect("qwen35 fused quant cache should not be poisoned");
        cache
            .entry(key)
            .or_insert_with(|| {
                let mut fused = Vec::with_capacity(wq.len() + wk.len() + wv.len());
                fused.extend_from_slice(wq);
                fused.extend_from_slice(wk);
                fused.extend_from_slice(wv);
                Arc::<[u8]>::from(fused)
            })
            .clone()
    }

    fn gpu_layer_keys_cache() -> &'static Mutex<HashMap<usize, Arc<[Qwen35GpuLayerKeys]>>> {
        static CACHE: OnceLock<Mutex<HashMap<usize, Arc<[Qwen35GpuLayerKeys]>>>> = OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn store_gpu_layer_keys(model_key: usize, layer_keys: Vec<Qwen35GpuLayerKeys>) {
        let cache = Self::gpu_layer_keys_cache();
        let mut cache = cache
            .lock()
            .expect("qwen35 gpu layer keys cache should not be poisoned");
        cache.insert(model_key, Arc::<[Qwen35GpuLayerKeys]>::from(layer_keys));
    }

    fn cached_gpu_layer_keys(model_key: usize) -> Option<Arc<[Qwen35GpuLayerKeys]>> {
        let cache = Self::gpu_layer_keys_cache();
        let cache = cache
            .lock()
            .expect("qwen35 gpu layer keys cache should not be poisoned");
        cache.get(&model_key).cloned()
    }

    fn assert_finite_if_enabled(
        label: &str,
        values: &[f32],
        layer: usize,
        position: usize,
    ) -> anyhow::Result<()> {
        if !env_flag_enabled("AX_QWEN35_ASSERT_FINITE") {
            return Ok(());
        }
        anyhow::ensure!(
            values.iter().all(|value| value.is_finite()),
            "qwen35 non-finite values at {label} (layer={layer}, position={position})"
        );
        Ok(())
    }

    fn gpu_decode_enabled() -> bool {
        env_flag_override("AX_QWEN35_GPU_DECODE").unwrap_or(true)
    }

    fn recurrent_dims(cfg: &ModelConfig) -> anyhow::Result<Qwen35RecurrentDims> {
        let dims = Qwen35RecurrentDims {
            conv_kernel: cfg
                .qwen35_ssm_conv_kernel
                .ok_or_else(|| anyhow::anyhow!("missing qwen35.ssm.conv_kernel"))?
                as usize,
            inner_size: cfg
                .qwen35_ssm_inner_size
                .ok_or_else(|| anyhow::anyhow!("missing qwen35.ssm.inner_size"))?
                as usize,
            state_size: cfg
                .qwen35_ssm_state_size
                .ok_or_else(|| anyhow::anyhow!("missing qwen35.ssm.state_size"))?
                as usize,
            time_step_rank: cfg
                .qwen35_ssm_time_step_rank
                .ok_or_else(|| anyhow::anyhow!("missing qwen35.ssm.time_step_rank"))?
                as usize,
            group_count: cfg
                .qwen35_ssm_group_count
                .ok_or_else(|| anyhow::anyhow!("missing qwen35.ssm.group_count"))?
                as usize,
        };
        anyhow::ensure!(dims.conv_kernel > 0, "qwen35 conv_kernel must be > 0");
        anyhow::ensure!(dims.state_size > 0, "qwen35 state_size must be > 0");
        anyhow::ensure!(dims.time_step_rank > 0, "qwen35 time_step_rank must be > 0");
        anyhow::ensure!(dims.group_count > 0, "qwen35 group_count must be > 0");
        anyhow::ensure!(dims.inner_size > 0, "qwen35 inner_size must be > 0");
        anyhow::ensure!(
            dims.inner_size == dims.state_size * dims.time_step_rank,
            "qwen35 inner_size ({}) must equal state_size ({}) * time_step_rank ({})",
            dims.inner_size,
            dims.state_size,
            dims.time_step_rank
        );
        anyhow::ensure!(
            dims.time_step_rank.is_multiple_of(dims.group_count),
            "qwen35 time_step_rank ({}) must be a multiple of group_count ({})",
            dims.time_step_rank,
            dims.group_count
        );
        Ok(dims)
    }

    fn qwen35_batch_projection_needs_f16_input(dtype: crate::gguf::tensor::GgmlType) -> bool {
        matches!(dtype, crate::gguf::tensor::GgmlType::Q8_0)
    }

    fn qwen35_batch_projection_supported(
        metal_ops: &crate::backend::metal::MetalOps,
        dtype: crate::gguf::tensor::GgmlType,
        m: u32,
        n: u32,
        k: u32,
    ) -> bool {
        matches!(
            dtype,
            crate::gguf::tensor::GgmlType::Q4K
                | crate::gguf::tensor::GgmlType::Q5K
                | crate::gguf::tensor::GgmlType::Q6K
        ) || (dtype == crate::gguf::tensor::GgmlType::Q8_0
            && (metal_ops.metal_q8_batch_native_shape_enabled(m, n, k)
                || metal_ops.metal_precompute_f16_enabled()))
    }

    fn prepare_qwen35_batch_projection_weight(
        metal_ops: &crate::backend::metal::MetalOps,
        raw: &[u8],
        dtype: crate::gguf::tensor::GgmlType,
        m: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        if dtype == crate::gguf::tensor::GgmlType::Q8_0 && metal_ops.metal_precompute_f16_enabled()
        {
            metal_ops.ensure_precomputed_q8_0_f16_from_raw(raw, m, k)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_qwen35_batch_projection(
        metal_ops: &crate::backend::metal::MetalOps,
        encoder: &ax_engine_metal::MetalEncoder,
        weight: &MetalBuffer,
        input_f32: &MetalBuffer,
        input_f16: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
        dtype: crate::gguf::tensor::GgmlType,
    ) {
        if Self::qwen35_batch_projection_needs_f16_input(dtype) {
            encode_dequant_batch_f16in(
                metal_ops, encoder, weight, input_f16, output, m, n, k, dtype,
            );
        } else {
            encode_dequant_batch(
                &metal_ops.dequant,
                &metal_ops.elementwise,
                encoder,
                weight,
                input_f32,
                output,
                input_f16,
                m,
                n,
                k,
                dtype,
                false,
                false,
                false,
            );
        }
    }

    /// Cache all Qwen3.5 model weights as GPU MetalBuffers.
    ///
    fn build_cached_model_keys_qwen35(
        metal_ops: &crate::backend::metal::MetalOps,
        weights: &WeightStore,
        cfg: &ModelConfig,
    ) -> anyhow::Result<()> {
        use crate::backend::metal::{CachedLayerKeys, CachedModelKeys};

        let n_layers = cfg.n_layers as usize;
        let mut layers = Vec::with_capacity(n_layers);
        let mut layer_types = Vec::with_capacity(n_layers);

        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");
            let is_recurrent = cfg.qwen35_is_recurrent_layer(layer);

            // Common weights: attn_norm, FFN (gate/up/down), post_attention_norm.
            let attn_norm_w = weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?;
            let attn_norm_key = metal_ops.ensure_f32_cached(attn_norm_w);
            let ffn_norm_w = weights.f32_slice(&format!("{prefix}.post_attention_norm.weight"))?;
            let ffn_norm_key = metal_ops.ensure_f32_cached(ffn_norm_w);
            let (wg_raw, wg_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.ffn_gate.weight"))?;
            let (wu_raw, wu_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_up.weight"))?;
            let (wd_raw, wd_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.ffn_down.weight"))?;
            let wg_key = metal_ops.ensure_quant_cached(wg_raw);
            let wu_key = metal_ops.ensure_quant_cached(wu_raw);
            let wd_key = metal_ops.ensure_quant_cached(wd_raw);

            if is_recurrent {
                // Recurrent layer: attn_qkv, attn_gate, ssm_* weights.
                let (wqkv_raw, wqkv_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.attn_qkv.weight"))?;
                let (wgate_raw, wgate_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.attn_gate.weight"))?;
                let (wbeta_raw, wbeta_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.ssm_beta.weight"))?;
                let (walpha_raw, walpha_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.ssm_alpha.weight"))?;
                let (wssm_out_raw, wssm_out_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.ssm_out.weight"))?;
                let wqkv_key = metal_ops.ensure_quant_cached(wqkv_raw);
                let wgate_key = metal_ops.ensure_quant_cached(wgate_raw);
                let wbeta_key = metal_ops.ensure_quant_cached(wbeta_raw);
                let walpha_key = metal_ops.ensure_quant_cached(walpha_raw);
                let wssm_out_key = metal_ops.ensure_quant_cached(wssm_out_raw);
                let conv_kernel = weights.f32_slice(&format!("{prefix}.ssm_conv1d.weight"))?;
                let conv_kernel_key = metal_ops.ensure_f32_cached(conv_kernel);
                let ssm_norm = weights.f32_slice(&format!("{prefix}.ssm_norm.weight"))?;
                let ssm_norm_key = metal_ops.ensure_f32_cached(ssm_norm);

                // Use dummy keys for wq/wk/wv/wo (recurrent layers don't have separate attention projections).
                layers.push(CachedLayerKeys {
                    attn_norm: attn_norm_key,
                    wq: wqkv_key,
                    wq_dtype: wqkv_dtype,
                    wk: wgate_key,
                    wk_dtype: wgate_dtype,
                    wv: wbeta_key,
                    wv_dtype: wbeta_dtype,
                    wo: wssm_out_key,
                    wo_dtype: wssm_out_dtype,
                    ffn_norm: ffn_norm_key,
                    wg: wg_key,
                    wg_dtype,
                    wu: wu_key,
                    wu_dtype,
                    wd: wd_key,
                    wd_dtype,
                    attn_q_norm: None,
                    attn_k_norm: None,
                    post_attn_norm: None,
                    post_ffn_norm: None,
                    q_bias: None,
                    k_bias: None,
                    v_bias: None,
                    wo_bias: None,
                    gate_bias: None,
                    up_bias: None,
                    down_bias: None,
                    moe_router: None,
                    moe_router_dtype: None,
                    moe_expert_gate: None,
                    moe_expert_up: None,
                    moe_expert_down: None,
                    moe_expert_dtype: None,
                    moe_shared_gate: None,
                    moe_shared_up: None,
                    moe_shared_down: None,
                    moe_shared_dtype: None,
                });
                layer_types.push(Qwen35GpuLayerKeys::Recurrent(Qwen35RecurrentLayerKeys {
                    wqkv: wqkv_key,
                    wqkv_dtype,
                    wgate: wgate_key,
                    wgate_dtype,
                    wbeta: wbeta_key,
                    wbeta_dtype,
                    walpha: walpha_key,
                    walpha_dtype,
                    wssm_out: wssm_out_key,
                    wssm_out_dtype,
                    conv_kernel: conv_kernel_key,
                    ssm_norm: ssm_norm_key,
                }));
            } else {
                // Full-attention layer: separate Q (doubled for gate), K, V, WO.
                let (wq_raw, wq_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.attn_q.weight"))?;
                let (wk_raw, wk_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.attn_k.weight"))?;
                let (wv_raw, wv_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.attn_v.weight"))?;
                let (wo_raw, wo_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.attn_output.weight"))?;
                let wq_key = metal_ops.ensure_quant_cached(wq_raw);
                let wk_key = metal_ops.ensure_quant_cached(wk_raw);
                let wv_key = metal_ops.ensure_quant_cached(wv_raw);
                let wo_key = metal_ops.ensure_quant_cached(wo_raw);
                let attn_q_norm_key = if weights.has(&format!("{prefix}.attn_q_norm.weight")) {
                    Some(metal_ops.ensure_f32_cached(
                        weights.f32_slice(&format!("{prefix}.attn_q_norm.weight"))?,
                    ))
                } else {
                    None
                };
                let attn_k_norm_key = if weights.has(&format!("{prefix}.attn_k_norm.weight")) {
                    Some(metal_ops.ensure_f32_cached(
                        weights.f32_slice(&format!("{prefix}.attn_k_norm.weight"))?,
                    ))
                } else {
                    None
                };
                layers.push(CachedLayerKeys {
                    attn_norm: attn_norm_key,
                    wq: wq_key,
                    wq_dtype,
                    wk: wk_key,
                    wk_dtype,
                    wv: wv_key,
                    wv_dtype,
                    wo: wo_key,
                    wo_dtype,
                    ffn_norm: ffn_norm_key,
                    wg: wg_key,
                    wg_dtype,
                    wu: wu_key,
                    wu_dtype,
                    wd: wd_key,
                    wd_dtype,
                    attn_q_norm: attn_q_norm_key,
                    attn_k_norm: attn_k_norm_key,
                    post_attn_norm: None,
                    post_ffn_norm: None,
                    q_bias: None,
                    k_bias: None,
                    v_bias: None,
                    wo_bias: None,
                    gate_bias: None,
                    up_bias: None,
                    down_bias: None,
                    moe_router: None,
                    moe_router_dtype: None,
                    moe_expert_gate: None,
                    moe_expert_up: None,
                    moe_expert_down: None,
                    moe_expert_dtype: None,
                    moe_shared_gate: None,
                    moe_shared_up: None,
                    moe_shared_down: None,
                    moe_shared_dtype: None,
                });
                layer_types.push(Qwen35GpuLayerKeys::FullAttention);
            }
        }

        let final_norm_w = weights.f32_slice("output_norm.weight")?;
        let output_norm_key = metal_ops.ensure_f32_cached(final_norm_w);
        let lm_weight_name = Self::lm_head_weight_name(weights);
        let (lm_raw, lm_dtype) = weights.raw_with_dtype(lm_weight_name)?;
        let lm_key = metal_ops.ensure_quant_cached(lm_raw);

        metal_ops.set_cached_model_keys(CachedModelKeys {
            layers,
            output_norm: output_norm_key,
            lm_head: lm_key,
            lm_head_dtype: lm_dtype,
        });
        Self::store_gpu_layer_keys(lm_key, layer_types);

        Ok(())
    }

    fn layer_type(cfg: &ModelConfig, layer: usize) -> Qwen35LayerType {
        if cfg.qwen35_is_recurrent_layer(layer) {
            Qwen35LayerType::RecurrentGdn
        } else {
            Qwen35LayerType::FullAttention
        }
    }

    fn rope_position(cfg: &ModelConfig, position: usize) -> f32 {
        cfg.rope_scaling.scaled_position(position)
    }

    #[allow(clippy::too_many_arguments)]
    fn batched_dequant_matmul_token_major(
        backend: &dyn crate::backend::Backend,
        a_quant: &[u8],
        dtype: crate::gguf::tensor::GgmlType,
        input_token_major: &[f32],
        output_token_major: &mut [f32],
        n_tokens: usize,
        out_dim: usize,
        in_dim: usize,
    ) {
        backend.dequant_matmul_token_major(
            a_quant,
            dtype,
            input_token_major,
            output_token_major,
            n_tokens,
            out_dim,
            in_dim,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_dequant_matmul_gpu_safe(
        backend: &dyn crate::backend::Backend,
        a_quant: &[u8],
        dtype: crate::gguf::tensor::GgmlType,
        input: &[f32],
        output: &mut [f32],
        m: usize,
        k: usize,
    ) {
        debug_assert_eq!(input.len(), k);
        debug_assert!(output.len() >= m);
        let ops = [(a_quant, dtype, m)];
        let mut outputs = [output];
        backend.safe_batch_dequant_matvec(&ops, input, k, &mut outputs);
    }

    fn decode_project_ops_gpu_safe(
        backend: &dyn crate::backend::Backend,
        input_ops: &[QuantOp<'_>],
        input: &[f32],
        k: usize,
        outputs: &mut [&mut [f32]],
    ) {
        backend.safe_batch_dequant_matvec(input_ops, input, k, outputs);
    }

    fn qwen35_recurrent_config(
        qwen_kv: &crate::kv::Qwen35Kv,
        dims: Qwen35RecurrentDims,
        rms_norm_eps: f32,
    ) -> gdn::Qwen35RecurrentConfig {
        gdn::Qwen35RecurrentConfig {
            conv_cache_len: qwen_kv.conv_cache_len(),
            conv_dim: qwen_kv.conv_dim(),
            group_count: dims.group_count,
            state_size: dims.state_size,
            time_step_rank: dims.time_step_rank,
            rms_norm_eps,
        }
    }

    fn validate_recurrent_layer_state(
        qwen_kv: &crate::kv::Qwen35Kv,
        recurrent_slot: usize,
        layer: usize,
        stage: &str,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            qwen_kv.is_recurrent_layer(layer),
            "qwen35 KV/state layer mapping mismatch at layer {layer}"
        );
        debug_assert_eq!(
            qwen_kv.recurrent_seqlen_offset(recurrent_slot),
            qwen_kv.seq_len(),
            "qwen35 recurrent slot {recurrent_slot} drifted from seq_len before {stage} layer {layer}"
        );
        Ok(())
    }

    fn maybe_fused_full_attention_input_plan<'a>(
        input_ops: [QuantOp<'a>; 3],
    ) -> Qwen35FullAttentionInputPlan<'a> {
        let [
            (wq_raw, wq_dtype, q_rows),
            (wk_raw, wk_dtype, k_rows),
            (wv_raw, wv_dtype, v_rows),
        ] = input_ops;
        if wq_dtype == wk_dtype && wq_dtype == wv_dtype {
            Qwen35FullAttentionInputPlan::Fused {
                raw: Self::fused_quant_rows_cached(wq_raw, wk_raw, wv_raw),
                dtype: wq_dtype,
                rows: q_rows + k_rows + v_rows,
            }
        } else {
            Qwen35FullAttentionInputPlan::Split([
                (wq_raw, wq_dtype, q_rows),
                (wk_raw, wk_dtype, k_rows),
                (wv_raw, wv_dtype, v_rows),
            ])
        }
    }

    fn full_attention_input_plan<'a>(
        weights: &'a WeightStore,
        prefix: &str,
        q_dim: usize,
        kv_dim: usize,
    ) -> anyhow::Result<Qwen35FullAttentionInputPlan<'a>> {
        let split_ops = Self::full_attention_input_ops(weights, prefix, q_dim, kv_dim)?;
        if !env_flag_enabled("AX_QWEN35_FUSED_FULL_ATTN_INPUT") {
            return Ok(Qwen35FullAttentionInputPlan::Split(split_ops));
        }
        Ok(Self::maybe_fused_full_attention_input_plan(split_ops))
    }

    fn full_attention_input_ops<'a>(
        weights: &'a WeightStore,
        prefix: &str,
        q_dim: usize,
        kv_dim: usize,
    ) -> anyhow::Result<[QuantOp<'a>; 3]> {
        let (wq_raw, wq_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_q.weight"))?;
        let (wk_raw, wk_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_k.weight"))?;
        let (wv_raw, wv_dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_v.weight"))?;
        Ok([
            (wq_raw, wq_dtype, q_dim * 2),
            (wk_raw, wk_dtype, kv_dim),
            (wv_raw, wv_dtype, kv_dim),
        ])
    }

    fn full_attention_output_op<'a>(
        weights: &'a WeightStore,
        prefix: &str,
        dim: usize,
    ) -> anyhow::Result<QuantOp<'a>> {
        let (raw, dtype) = weights.raw_with_dtype(&format!("{prefix}.attn_output.weight"))?;
        Ok((raw, dtype, dim))
    }

    fn project_full_attention_inputs<F>(
        input_ops: [QuantOp<'_>; 3],
        outputs: [&mut [f32]; 3],
        mut project: F,
    ) where
        F: FnMut(&[u8], crate::gguf::tensor::GgmlType, usize, &mut [f32]),
    {
        for ((raw, dtype, rows), out) in input_ops.into_iter().zip(outputs.into_iter()) {
            project(raw, dtype, rows, out);
        }
    }

    fn split_full_attention_fused_output(
        fused_output: &[f32],
        q_gate: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
    ) {
        debug_assert_eq!(k.len(), v.len());
        debug_assert_eq!(fused_output.len(), q_gate.len() + k.len() + v.len());
        let q_gate_end = q_gate.len();
        let k_end = q_gate_end + k.len();
        q_gate.copy_from_slice(&fused_output[..q_gate_end]);
        k.copy_from_slice(&fused_output[q_gate_end..k_end]);
        v.copy_from_slice(&fused_output[k_end..]);
    }

    fn split_full_attention_fused_output_batch(
        fused_output_batch: &[f32],
        q_gate_batch: &mut [f32],
        k_batch: &mut [f32],
        v_batch: &mut [f32],
        n_tokens: usize,
    ) {
        let q_gate_dim = q_gate_batch.len() / n_tokens;
        let kv_dim = k_batch.len() / n_tokens;
        debug_assert_eq!(k_batch.len(), v_batch.len());
        debug_assert_eq!(
            fused_output_batch.len(),
            n_tokens * (q_gate_dim + 2 * kv_dim)
        );
        for token_idx in 0..n_tokens {
            let fused_start = token_idx * (q_gate_dim + 2 * kv_dim);
            let fused_end = fused_start + q_gate_dim + 2 * kv_dim;
            let q_gate_start = token_idx * q_gate_dim;
            let k_start = token_idx * kv_dim;
            Self::split_full_attention_fused_output(
                &fused_output_batch[fused_start..fused_end],
                &mut q_gate_batch[q_gate_start..q_gate_start + q_gate_dim],
                &mut k_batch[k_start..k_start + kv_dim],
                &mut v_batch[k_start..k_start + kv_dim],
            );
        }
    }

    fn project_full_attention_inputs_batch<F>(
        input_plan: Qwen35FullAttentionInputPlan<'_>,
        q_gate_batch: &mut [f32],
        k_batch: &mut [f32],
        v_batch: &mut [f32],
        fused_output_batch: &mut [f32],
        n_tokens: usize,
        mut project: F,
    ) where
        F: FnMut(&[u8], crate::gguf::tensor::GgmlType, usize, &mut [f32]),
    {
        match input_plan {
            Qwen35FullAttentionInputPlan::Split(input_ops) => {
                Self::project_full_attention_inputs(
                    input_ops,
                    [q_gate_batch, k_batch, v_batch],
                    project,
                );
            }
            Qwen35FullAttentionInputPlan::Fused { raw, dtype, rows } => {
                debug_assert_eq!(fused_output_batch.len(), n_tokens * rows);
                project(raw.as_ref(), dtype, rows, fused_output_batch);
                Self::split_full_attention_fused_output_batch(
                    fused_output_batch,
                    q_gate_batch,
                    k_batch,
                    v_batch,
                    n_tokens,
                );
            }
        }
    }

    fn maybe_attention_qk_norm<'a>(
        weights: &'a WeightStore,
        prefix: &str,
    ) -> anyhow::Result<Option<Qwen35AttentionNormWeights<'a>>> {
        if !weights.has(&format!("{prefix}.attn_q_norm.weight")) {
            return Ok(None);
        }
        Ok(Some(Qwen35AttentionNormWeights {
            q: weights.f32_slice(&format!("{prefix}.attn_q_norm.weight"))?,
            k: weights.f32_slice(&format!("{prefix}.attn_k_norm.weight"))?,
        }))
    }

    fn recurrent_input_ops<'a>(
        weights: &'a WeightStore,
        prefix: &str,
        dims: Qwen35RecurrentDims,
    ) -> anyhow::Result<[QuantOp<'a>; 4]> {
        let (wqkv_raw, wqkv_dtype) =
            weights.raw_with_dtype(&format!("{prefix}.attn_qkv.weight"))?;
        let (wgate_raw, wgate_dtype) =
            weights.raw_with_dtype(&format!("{prefix}.attn_gate.weight"))?;
        let (wbeta_raw, wbeta_dtype) =
            weights.raw_with_dtype(&format!("{prefix}.ssm_beta.weight"))?;
        let (walpha_raw, walpha_dtype) =
            weights.raw_with_dtype(&format!("{prefix}.ssm_alpha.weight"))?;
        Ok([
            (wqkv_raw, wqkv_dtype, dims.conv_dim()),
            (wgate_raw, wgate_dtype, dims.inner_size),
            (wbeta_raw, wbeta_dtype, dims.time_step_rank),
            (walpha_raw, walpha_dtype, dims.time_step_rank),
        ])
    }

    fn recurrent_runtime_tensors<'a>(
        weights: &'a WeightStore,
        prefix: &str,
    ) -> anyhow::Result<Qwen35RecurrentRuntimeTensors<'a>> {
        Ok(Qwen35RecurrentRuntimeTensors {
            dt_bias: weights.f32_slice(&format!("{prefix}.ssm_dt.bias"))?,
            a: weights.f32_slice(&format!("{prefix}.ssm_a"))?,
            conv_kernel: weights.f32_slice(&format!("{prefix}.ssm_conv1d.weight"))?,
            ssm_norm: weights.f32_slice(&format!("{prefix}.ssm_norm.weight"))?,
        })
    }

    fn recurrent_output_op<'a>(
        weights: &'a WeightStore,
        prefix: &str,
        dim: usize,
    ) -> anyhow::Result<QuantOp<'a>> {
        let (raw, dtype) = weights.raw_with_dtype(&format!("{prefix}.ssm_out.weight"))?;
        Ok((raw, dtype, dim))
    }

    fn project_recurrent_inputs<F>(
        input_ops: [QuantOp<'_>; 4],
        outputs: [&mut [f32]; 4],
        mut project: F,
    ) where
        F: FnMut(&[u8], crate::gguf::tensor::GgmlType, usize, &mut [f32]),
    {
        for ((raw, dtype, rows), out) in input_ops.into_iter().zip(outputs.into_iter()) {
            project(raw, dtype, rows, out);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_recurrent_sequence<'a>(
        backend: &dyn crate::backend::Backend,
        weights: &'a WeightStore,
        prefix: &str,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        layer: usize,
        recurrent_slot_indices: &[usize],
        dims: Qwen35RecurrentDims,
        rms_norm_eps: f32,
        rec_qkv: &[f32],
        rec_beta: &mut [f32],
        rec_alpha: &mut [f32],
        rec_out: &mut [f32],
        n_tokens: usize,
    ) -> anyhow::Result<Qwen35RecurrentRuntimeTensors<'a>> {
        let runtime = Self::recurrent_runtime_tensors(weights, prefix)?;
        let qwen35_cfg = Self::qwen35_recurrent_config(qwen_kv, dims, rms_norm_eps);
        backend.qwen35_recurrent_sequence_for_kv(
            rec_qkv,
            rec_beta,
            rec_alpha,
            runtime.dt_bias,
            runtime.a,
            runtime.conv_kernel,
            qwen_kv,
            layer,
            recurrent_slot_indices,
            rec_out,
            n_tokens,
            qwen35_cfg,
        );
        Ok(runtime)
    }

    fn ffn_input_ops<'a>(
        weights: &'a WeightStore,
        prefix: &str,
        inter_dim: usize,
    ) -> anyhow::Result<[QuantOp<'a>; 2]> {
        let (wg_raw, wg_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_gate.weight"))?;
        let (wu_raw, wu_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_up.weight"))?;
        Ok([(wg_raw, wg_dtype, inter_dim), (wu_raw, wu_dtype, inter_dim)])
    }

    fn ffn_down_op<'a>(
        weights: &'a WeightStore,
        prefix: &str,
        dim: usize,
    ) -> anyhow::Result<QuantOp<'a>> {
        let (raw, dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_down.weight"))?;
        Ok((raw, dtype, dim))
    }

    fn finalize_recurrent_output(
        rec_out: &mut [f32],
        rec_z: &[f32],
        dims: Qwen35RecurrentDims,
        ssm_norm_w: &[f32],
        rms_norm_eps: f32,
    ) {
        for head in 0..dims.time_step_rank {
            let start = head * dims.state_size;
            let end = start + dims.state_size;
            rms_norm::rms_norm(&mut rec_out[start..end], ssm_norm_w, rms_norm_eps);
        }
        let mut z_gate = rec_z.to_vec();
        silu::silu(&mut z_gate);
        silu::elementwise_mul(rec_out, &z_gate);
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_recurrent_pair_matvec_with_config(
        metal_ops: &crate::backend::metal::MetalOps,
        encoder: &ax_engine_metal::MetalEncoder,
        wbeta: &MetalBuffer,
        walpha: &MetalBuffer,
        input: &MetalBuffer,
        beta_out: &MetalBuffer,
        alpha_out: &MetalBuffer,
        m: u32,
        k: u32,
        beta_dtype: crate::gguf::tensor::GgmlType,
        alpha_dtype: crate::gguf::tensor::GgmlType,
    ) -> bool {
        if beta_dtype != alpha_dtype {
            return false;
        }

        // Keep the precomputed-f16 route authoritative when either weight was
        // densified ahead of time; the paired quant kernels are only for the
        // raw quantized path.
        if metal_ops.has_precomputed_weight(wbeta) || metal_ops.has_precomputed_weight(walpha) {
            return false;
        }

        match beta_dtype {
            crate::gguf::tensor::GgmlType::Q4K => {
                metal_ops.dequant.encode_fused_matvec_pair_q4_k(
                    encoder, wbeta, walpha, input, beta_out, alpha_out, m, k,
                );
                true
            }
            crate::gguf::tensor::GgmlType::Q5K => {
                metal_ops.dequant.encode_fused_matvec_pair_q5_k(
                    encoder, wbeta, walpha, input, beta_out, alpha_out, m, k,
                );
                true
            }
            crate::gguf::tensor::GgmlType::Q6K => {
                metal_ops.dequant.encode_fused_matvec_pair_q6_k(
                    encoder, wbeta, walpha, input, beta_out, alpha_out, m, k,
                );
                true
            }
            crate::gguf::tensor::GgmlType::Q8_0 => {
                metal_ops.dequant.encode_fused_matvec_pair_q8_0(
                    encoder, wbeta, walpha, input, beta_out, alpha_out, m, k,
                );
                true
            }
            _ => false,
        }
    }

    fn rms_norm_token_major(
        input: &[f32],
        weight: &[f32],
        output: &mut [f32],
        n_tokens: usize,
        dim: usize,
        rms_norm_eps: f32,
    ) {
        for token_idx in 0..n_tokens {
            let start = token_idx * dim;
            let end = start + dim;
            rms_norm::rms_norm_out(
                &input[start..end],
                weight,
                &mut output[start..end],
                rms_norm_eps,
            );
        }
    }

    fn extract_q_from_q_gate(q_gate: &[f32], q: &mut [f32]) {
        debug_assert_eq!(q_gate.len(), q.len() * 2);
        q.copy_from_slice(&q_gate[..q.len()]);
    }

    #[cfg(test)]
    fn extract_q_from_q_gate_batch(
        q_gate_batch: &[f32],
        q_batch: &mut [f32],
        n_tokens: usize,
        q_dim: usize,
    ) {
        for token_idx in 0..n_tokens {
            let src_start = token_idx * q_dim * 2;
            let q_start = token_idx * q_dim;
            Self::extract_q_from_q_gate(
                &q_gate_batch[src_start..src_start + q_dim * 2],
                &mut q_batch[q_start..q_start + q_dim],
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_attention_qk_norm(
        q: &mut [f32],
        k: &mut [f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        norm_weights: Qwen35AttentionNormWeights<'_>,
        rms_norm_eps: f32,
    ) {
        per_head_rms_norm(q, n_heads, head_dim, norm_weights.q, rms_norm_eps);
        per_head_rms_norm(k, n_kv_heads, head_dim, norm_weights.k, rms_norm_eps);
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    fn apply_attention_qk_norm_batch(
        q_batch: &mut [f32],
        k_batch: &mut [f32],
        n_tokens: usize,
        q_dim: usize,
        kv_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        norm_weights: Qwen35AttentionNormWeights<'_>,
        rms_norm_eps: f32,
    ) {
        for token_idx in 0..n_tokens {
            let q_start = token_idx * q_dim;
            let k_start = token_idx * kv_dim;
            Self::apply_attention_qk_norm(
                &mut q_batch[q_start..q_start + q_dim],
                &mut k_batch[k_start..k_start + kv_dim],
                n_heads,
                n_kv_heads,
                head_dim,
                norm_weights,
                rms_norm_eps,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_rope(
        cfg: &ModelConfig,
        q: &mut [f32],
        k: &mut [f32],
        position: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        let rope_position = Self::rope_position(cfg, position);
        rope::apply_rope_multi_head_scaled(
            q,
            k,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_position,
            cfg.rope_freq_base,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    fn apply_rope_batch(
        cfg: &ModelConfig,
        q_batch: &mut [f32],
        k_batch: &mut [f32],
        n_tokens: usize,
        start_position: usize,
        q_dim: usize,
        kv_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        for token_idx in 0..n_tokens {
            let q_start = token_idx * q_dim;
            let k_start = token_idx * kv_dim;
            Self::apply_rope(
                cfg,
                &mut q_batch[q_start..q_start + q_dim],
                &mut k_batch[k_start..k_start + kv_dim],
                start_position + token_idx,
                n_heads,
                n_kv_heads,
                head_dim,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_full_attention_qk_batch(
        cfg: &ModelConfig,
        q_gate_batch: &[f32],
        q_batch: &mut [f32],
        k_batch: &mut [f32],
        n_tokens: usize,
        start_position: usize,
        q_dim: usize,
        kv_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        norm_weights: Option<Qwen35AttentionNormWeights<'_>>,
        rms_norm_eps: f32,
    ) {
        for token_idx in 0..n_tokens {
            let q_gate_start = token_idx * q_dim * 2;
            let q_start = token_idx * q_dim;
            let k_start = token_idx * kv_dim;
            let q = &mut q_batch[q_start..q_start + q_dim];
            let k = &mut k_batch[k_start..k_start + kv_dim];
            Self::extract_q_from_q_gate(&q_gate_batch[q_gate_start..q_gate_start + q_dim * 2], q);
            if let Some(norm_weights) = norm_weights {
                Self::apply_attention_qk_norm(
                    q,
                    k,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    norm_weights,
                    rms_norm_eps,
                );
            }
            Self::apply_rope(
                cfg,
                q,
                k,
                start_position + token_idx,
                n_heads,
                n_kv_heads,
                head_dim,
            );
        }
    }

    fn apply_attention_gate(gate: &mut [f32], attn_out: &mut [f32]) {
        debug_assert_eq!(gate.len(), attn_out.len());
        gdn::sigmoid_in_place(gate);
        silu::elementwise_mul(attn_out, gate);
    }

    fn apply_attention_gate_batch(
        q_gate_batch: &mut [f32],
        attn_out_batch: &mut [f32],
        n_tokens: usize,
        q_dim: usize,
    ) {
        for token_idx in 0..n_tokens {
            let gate_start = token_idx * q_dim * 2 + q_dim;
            let attn_start = token_idx * q_dim;
            Self::apply_attention_gate(
                &mut q_gate_batch[gate_start..gate_start + q_dim],
                &mut attn_out_batch[attn_start..attn_start + q_dim],
            );
        }
    }

    /// GPU-accelerated recurrent finalize: per-head RMS norm + SiLU gate.
    ///
    /// Replaces per-token CPU loops in `finalize_recurrent_output_batch` with
    /// GPU batch dispatches. Falls back to CPU when Metal is unavailable.
    #[allow(clippy::too_many_arguments)]
    fn try_finalize_recurrent_output_batch_gpu(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        rec_out_batch: &mut [f32],
        rec_z_batch: &[f32],
        n_tokens: usize,
        dims: Qwen35RecurrentDims,
        ssm_norm_w: &[f32],
        rms_norm_eps: f32,
    ) -> anyhow::Result<bool> {
        if n_tokens <= 1 {
            return Ok(false);
        }
        let Some(metal_ops) = backend.metal_ops() else {
            return Ok(false);
        };

        let total = n_tokens * dims.inner_size;
        let nw_key = metal_ops.ensure_f32_cached(ssm_norm_w);

        metal_ops.init_batch_scratches(cfg, n_tokens);
        let mut bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_mut() else {
            return Ok(false);
        };

        // Upload rec_out → gate_buf, rec_z → up_buf.
        // gate_buf is [N × inter_dim], up_buf is [N × inter_dim]; inner_size < inter_dim.
        unsafe {
            bs.gate_buf.as_mut_slice::<f32>()[..total].copy_from_slice(&rec_out_batch[..total]);
            bs.up_buf.as_mut_slice::<f32>()[..total].copy_from_slice(&rec_z_batch[..total]);
        }

        let weight_cache = metal_ops.lock_weight_cache();
        let nw_buf = weight_cache.get(&nw_key).unwrap();

        metal_ops.device.execute_sync(|encoder| {
            // 1. Per-head RMS norm on rec_out (in gate_buf).
            // inner_size = time_step_rank × state_size, treat as n_heads=time_step_rank, head_dim=state_size.
            metal_ops.elementwise.encode_per_head_rms_norm_batch(
                encoder,
                &bs.gate_buf,
                nw_buf,
                n_tokens as u32,
                dims.time_step_rank as u32,
                dims.state_size as u32,
                rms_norm_eps,
            );

            // 2. SiLU(rec_z) × rec_out: up_buf = silu(up_buf) * gate_buf.
            // After this, up_buf holds the final result.
            metal_ops.elementwise.encode_silu_elementwise_mul_batch(
                encoder,
                &bs.up_buf,
                &bs.gate_buf,
                dims.inner_size as u32,
                n_tokens as u32,
            );
            Ok(())
        })?;
        drop(weight_cache);

        // Read result from up_buf back to rec_out_batch.
        unsafe {
            rec_out_batch[..total].copy_from_slice(&bs.up_buf.as_slice::<f32>()[..total]);
        }
        drop(bs_guard);
        Ok(true)
    }

    fn finalize_recurrent_output_batch(
        rec_out_batch: &mut [f32],
        rec_z_batch: &[f32],
        n_tokens: usize,
        dims: Qwen35RecurrentDims,
        ssm_norm_w: &[f32],
        rms_norm_eps: f32,
    ) {
        for token_idx in 0..n_tokens {
            let rec_out_start = token_idx * dims.inner_size;
            let rec_out_end = rec_out_start + dims.inner_size;
            Self::finalize_recurrent_output(
                &mut rec_out_batch[rec_out_start..rec_out_end],
                &rec_z_batch[rec_out_start..rec_out_end],
                dims,
                ssm_norm_w,
                rms_norm_eps,
            );
        }
    }

    fn lm_head_weight_name(weights: &WeightStore) -> &'static str {
        if weights.has("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn write_single_logits(
        backend: &dyn crate::backend::Backend,
        hidden: &mut [f32],
        dim: usize,
        vocab_size: usize,
        rms_norm_eps: f32,
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        let final_norm_w = weights.f32_slice("output_norm.weight")?;
        rms_norm::rms_norm(hidden, final_norm_w, rms_norm_eps);
        Self::write_normalized_single_logits(backend, hidden, dim, vocab_size, weights, logits)
    }

    fn write_normalized_single_logits(
        backend: &dyn crate::backend::Backend,
        hidden: &[f32],
        dim: usize,
        vocab_size: usize,
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        let lm_weight_name = Self::lm_head_weight_name(weights);
        let (lm_raw, lm_dtype) = weights.raw_with_dtype(lm_weight_name)?;
        anyhow::ensure!(logits.len() >= vocab_size, "logits buffer too small");
        backend.dequant_matmul(lm_raw, lm_dtype, hidden, logits, vocab_size, 1, dim);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn write_normalized_batch_logits(
        backend: &dyn crate::backend::Backend,
        hidden: &[f32],
        n_tokens: usize,
        dim: usize,
        vocab_size: usize,
        weights: &WeightStore,
        logits_all: &mut [f32],
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            hidden.len() >= n_tokens * dim,
            "normalized hidden buffer too small for {n_tokens} tokens"
        );
        anyhow::ensure!(
            logits_all.len() >= n_tokens * vocab_size,
            "all-logits buffer too small for {n_tokens} tokens"
        );
        for token_idx in 0..n_tokens {
            let hidden_start = token_idx * dim;
            let logits_start = token_idx * vocab_size;
            Self::write_normalized_single_logits(
                backend,
                &hidden[hidden_start..hidden_start + dim],
                dim,
                vocab_size,
                weights,
                &mut logits_all[logits_start..logits_start + vocab_size],
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn full_attention_prefill_batch(
        backend: &dyn crate::backend::Backend,
        qwen_kv: &crate::kv::Qwen35Kv,
        layer: usize,
        q_batch: &[f32],
        k_batch: &[f32],
        v_batch: &[f32],
        attn_out_batch: &mut [f32],
        n_tokens: usize,
        params: &AttentionParams,
    ) {
        let prefix_len = qwen_kv.seq_len();
        if prefix_len == 0 {
            backend.attention_prefill(
                q_batch,
                k_batch,
                v_batch,
                attn_out_batch,
                n_tokens,
                params.n_heads,
                params.n_kv_heads,
                params.head_dim,
            );
        } else {
            attention::multi_head_attention_prefill_with_prefix(
                qwen_kv.attention_k_slice_including_current(layer, prefix_len),
                qwen_kv.attention_v_slice_including_current(layer, prefix_len),
                prefix_len,
                q_batch,
                k_batch,
                v_batch,
                attn_out_batch,
                n_tokens,
                params,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn try_full_attention_prefill_gpu_with_prefix(
        backend: &dyn crate::backend::Backend,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        layer: usize,
        q_batch: &[f32],
        k_batch: &[f32],
        v_batch: &[f32],
        attn_out_batch: &mut [f32],
        n_tokens: usize,
        full_attn_params: &AttentionParams,
    ) -> anyhow::Result<bool> {
        let prefix_len = qwen_kv.seq_len();
        if prefix_len == 0 {
            return Ok(false);
        }
        let Some(metal_ops) = backend.metal_ops() else {
            return Ok(false);
        };
        if qwen_kv.gpu_attention().is_none() {
            return Ok(false);
        }

        qwen_kv.attention_append_batch(layer, k_batch, v_batch, n_tokens);

        let q_len = n_tokens * full_attn_params.n_heads * full_attn_params.head_dim;
        debug_assert_eq!(q_batch.len(), q_len);
        debug_assert_eq!(attn_out_batch.len(), q_len);

        let Some(gpu_attention) = qwen_kv.gpu_attention() else {
            attention::multi_head_attention_prefill_with_prefix(
                qwen_kv.attention_k_slice_including_current(layer, prefix_len),
                qwen_kv.attention_v_slice_including_current(layer, prefix_len),
                prefix_len,
                q_batch,
                k_batch,
                v_batch,
                attn_out_batch,
                n_tokens,
                full_attn_params,
            );
            return Ok(true);
        };

        let buf_q = MetalBuffer::from_slice(metal_ops.device.device(), &q_batch[..q_len])?;
        let buf_o = MetalBuffer::new(
            metal_ops.device.device(),
            q_len * std::mem::size_of::<f32>(),
        )?;
        metal_ops.device.execute_sync(|encoder| {
            metal_ops
                .attention
                .encode_attention_prefill_cached_with_config(
                    encoder,
                    &buf_q,
                    gpu_attention.k_buffer(layer),
                    gpu_attention.v_buffer(layer),
                    &buf_o,
                    gpu_attention.is_f16(),
                    n_tokens as u32,
                    full_attn_params.n_heads as u32,
                    full_attn_params.n_kv_heads as u32,
                    full_attn_params.head_dim as u32,
                    prefix_len as u32,
                    0,
                    metal_ops.attention_dispatch_config(),
                );
            Ok(())
        })?;

        let result = unsafe { buf_o.as_slice::<f32>() };
        attn_out_batch[..q_len].copy_from_slice(&result[..q_len]);
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn try_full_attention_decode_gpu(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        qwen_kv: &crate::kv::Qwen35Kv,
        layer: usize,
        q: &[f32],
        attn_out: &mut [f32],
        full_attn_params: &AttentionParams,
    ) -> anyhow::Result<bool> {
        let Some(metal_ops) = backend.metal_ops() else {
            return Ok(false);
        };
        let Some(gpu_attention) = qwen_kv.gpu_attention() else {
            return Ok(false);
        };

        metal_ops.init_scratches(cfg);
        let scratch_guard = metal_ops.scratches();
        let Some(scratches) = scratch_guard.as_ref() else {
            return Ok(false);
        };

        let q_len = full_attn_params.n_heads * full_attn_params.head_dim;
        debug_assert_eq!(q.len(), q_len);
        debug_assert_eq!(attn_out.len(), q_len);
        unsafe {
            let q_dst = scratches.q_buf.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(q.as_ptr(), q_dst, q_len);
        }

        metal_ops.device.execute_sync(|encoder| {
            metal_ops
                .attention
                .encode_attention_decode_with_scratch_and_config(
                    encoder,
                    &scratches.q_buf,
                    gpu_attention.k_buffer(layer),
                    gpu_attention.v_buffer(layer),
                    &scratches.attn_out,
                    &scratches.splitk_partial_out,
                    &scratches.splitk_partial_lse,
                    gpu_attention.is_f16(),
                    full_attn_params.n_heads as u32,
                    full_attn_params.n_kv_heads as u32,
                    full_attn_params.head_dim as u32,
                    0,
                    (qwen_kv.seq_len() + 1) as u32,
                    metal_ops.attention_dispatch_config(),
                );
            Ok(())
        })?;

        unsafe {
            let attn_src = scratches.attn_out.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(attn_src, attn_out.as_mut_ptr(), q_len);
        }
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn write_last_batch_logits(
        backend: &dyn crate::backend::Backend,
        hidden: &mut [f32],
        n_tokens: usize,
        dim: usize,
        vocab_size: usize,
        rms_norm_eps: f32,
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        let last_hidden = &mut hidden[(n_tokens - 1) * dim..n_tokens * dim];
        Self::write_single_logits(
            backend,
            last_hidden,
            dim,
            vocab_size,
            rms_norm_eps,
            weights,
            logits,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_post_attention_ffn_batch(
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        hidden: &mut [f32],
        norm_buf: &mut [f32],
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        down_buf: &mut [f32],
        n_tokens: usize,
        dim: usize,
        inter_dim: usize,
        rms_norm_eps: f32,
    ) -> anyhow::Result<()> {
        let ffn_norm_w = weights.f32_slice(&format!("{prefix}.post_attention_norm.weight"))?;
        crate::model::layer_ops::apply_ffn_batch(
            backend,
            weights,
            prefix,
            hidden,
            norm_buf,
            gate_buf,
            up_buf,
            down_buf,
            n_tokens,
            dim,
            inter_dim,
            ffn_norm_w,
            rms_norm_eps,
            crate::model::layer_ops::FfnActivation::SiLU,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_post_attention_ffn_single(
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        hidden: &mut [f32],
        norm_buf: &mut [f32],
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        down_buf: &mut [f32],
        dim: usize,
        inter_dim: usize,
        rms_norm_eps: f32,
        ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        if ops.is_some() {
            // Profiled path: keep inline for per-op timing.
            let mut ops = ops;
            let ffn_norm_w = timed!(
                ops,
                dequant,
                weights.f32_slice(&format!("{prefix}.post_attention_norm.weight"))?
            );
            timed!(
                ops,
                norm,
                rms_norm::rms_norm_out(hidden, ffn_norm_w, norm_buf, rms_norm_eps)
            );
            let input_ops = Self::ffn_input_ops(weights, prefix, inter_dim)?;
            timed_matmul_bucket!(ops, matmul_input_proj, {
                let mut outputs = [&mut *gate_buf, &mut *up_buf];
                Self::decode_project_ops_gpu_safe(backend, &input_ops, norm_buf, dim, &mut outputs);
            });
            silu::silu_elementwise_mul(gate_buf, up_buf);
            let (wd_raw, wd_dtype, _) =
                timed!(ops, dequant, Self::ffn_down_op(weights, prefix, dim)?);
            timed_matmul_bucket!(
                ops,
                matmul_output_proj,
                Self::decode_dequant_matmul_gpu_safe(
                    backend, wd_raw, wd_dtype, gate_buf, down_buf, dim, inter_dim
                )
            );
            silu::elementwise_add(hidden, down_buf);
        } else {
            // Non-profiled path: delegate to shared implementation.
            let ffn_norm_w = weights.f32_slice(&format!("{prefix}.post_attention_norm.weight"))?;
            crate::model::layer_ops::apply_ffn_single(
                backend,
                weights,
                prefix,
                hidden,
                norm_buf,
                gate_buf,
                up_buf,
                down_buf,
                dim,
                inter_dim,
                ffn_norm_w,
                rms_norm_eps,
                crate::model::layer_ops::FfnActivation::SiLU,
            );
        }
        Ok(())
    }

    /// GPU-encoded attention norm + input projections in a single command buffer.
    ///
    /// Encodes: RMSNorm(hidden) → norm_buf, then for each (weight, out_dim):
    /// fused dequant matmul → output buffer. All in one execute_sync.
    ///
    /// Returns `true` if GPU path was used, `false` to fall back to CPU.
    #[allow(clippy::too_many_arguments)]
    fn try_norm_and_project_batch_gpu(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        hidden: &[f32],
        norm_buf: &mut [f32],
        projections: &[QuantOp<'_>],
        outputs: &mut [&mut [f32]],
        n_tokens: usize,
        dim: usize,
        rms_norm_eps: f32,
    ) -> anyhow::Result<bool> {
        debug_assert_eq!(projections.len(), outputs.len());
        if n_tokens <= 1 {
            return Ok(false);
        }
        let Some(metal_ops) = backend.metal_ops() else {
            return Ok(false);
        };

        if !projections.iter().all(|(_, dtype, out_dim)| {
            Self::qwen35_batch_projection_supported(
                metal_ops,
                *dtype,
                *out_dim as u32,
                n_tokens as u32,
                dim as u32,
            )
        }) {
            return Ok(false);
        }

        // Cache all weights + norm.
        let attn_norm_w = weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?;
        let nw_key = metal_ops.ensure_f32_cached(attn_norm_w);
        let mut proj_keys = Vec::with_capacity(projections.len());
        for (raw, dtype, out_dim) in projections {
            Self::prepare_qwen35_batch_projection_weight(
                metal_ops,
                raw,
                *dtype,
                *out_dim as u32,
                dim as u32,
            )?;
            let key = raw.as_ptr() as usize;
            metal_ops.ensure_quant_cached(raw);
            proj_keys.push(key);
        }

        metal_ops.init_batch_scratches(cfg, n_tokens);

        let mut bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_mut() else {
            return Ok(false);
        };

        // Copy hidden to GPU.
        unsafe {
            bs.hidden.as_mut_slice::<f32>()[..n_tokens * dim].copy_from_slice(hidden);
        }

        let weight_cache = metal_ops.lock_weight_cache();
        let nw_buf = weight_cache.get(&nw_key).unwrap();

        // Allocate temporary output buffers for projections that don't fit in
        // the standard scratch buffers.
        let mut temp_bufs: Vec<MetalBuffer> = Vec::new();
        for (_, _, out_dim) in projections {
            let size = n_tokens * out_dim * std::mem::size_of::<f32>();
            temp_bufs.push(
                MetalBuffer::new(metal_ops.device.device(), size)
                    .expect("Failed to alloc projection output buffer"),
            );
        }

        metal_ops.device.execute_sync(|encoder| {
            // 1. RMSNorm: hidden → norm_buf
            metal_ops.elementwise.encode_rms_norm_out_batch(
                encoder,
                &bs.hidden,
                nw_buf,
                &bs.norm_buf,
                dim as u32,
                n_tokens as u32,
                rms_norm_eps,
            );
            if projections
                .iter()
                .any(|(_, dtype, _)| Self::qwen35_batch_projection_needs_f16_input(*dtype))
            {
                metal_ops.elementwise.encode_cast_f32_to_f16(
                    encoder,
                    &bs.norm_buf,
                    &bs.matmul_in_f16,
                    (n_tokens * dim) as u32,
                );
            }
            // 2. All input projections from norm_buf.
            for (i, (_, dtype, out_dim)) in projections.iter().enumerate() {
                let w_buf = weight_cache.get(&proj_keys[i]).unwrap();
                Self::encode_qwen35_batch_projection(
                    metal_ops,
                    encoder,
                    w_buf,
                    &bs.norm_buf,
                    &bs.matmul_in_f16,
                    &temp_bufs[i],
                    *out_dim as u32,
                    n_tokens as u32,
                    dim as u32,
                    *dtype,
                );
            }
            Ok(())
        })?;
        drop(weight_cache);

        // Read back norm_buf and projection outputs.
        let norm_result = unsafe { &bs.norm_buf.as_slice::<f32>()[..n_tokens * dim] };
        norm_buf[..n_tokens * dim].copy_from_slice(norm_result);

        for (i, (_, _, out_dim)) in projections.iter().enumerate() {
            let result = unsafe { &temp_bufs[i].as_slice::<f32>()[..n_tokens * out_dim] };
            outputs[i][..n_tokens * out_dim].copy_from_slice(result);
        }
        drop(bs_guard);

        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_attention_norm_batch(
        weights: &WeightStore,
        prefix: &str,
        hidden: &[f32],
        norm_buf: &mut [f32],
        n_tokens: usize,
        dim: usize,
        rms_norm_eps: f32,
    ) -> anyhow::Result<()> {
        let attn_norm_w = weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?;
        Self::rms_norm_token_major(hidden, attn_norm_w, norm_buf, n_tokens, dim, rms_norm_eps);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_attention_norm_single(
        weights: &WeightStore,
        prefix: &str,
        hidden: &[f32],
        norm_buf: &mut [f32],
        rms_norm_eps: f32,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let attn_norm_w = timed!(
            ops,
            dequant,
            weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?
        );
        timed!(
            ops,
            norm,
            rms_norm::rms_norm_out(hidden, attn_norm_w, norm_buf, rms_norm_eps)
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    /// GPU-accelerated batch Q/K preparation: Q extraction + QK norm + RoPE.
    ///
    /// Replaces `prepare_full_attention_qk_batch` per-token CPU loops with
    /// GPU batch dispatches. Falls back to CPU when Metal is unavailable.
    #[allow(clippy::too_many_arguments)]
    fn try_prepare_full_attention_qk_batch_gpu(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        q_gate_batch: &[f32],
        q_batch: &mut [f32],
        k_batch: &mut [f32],
        n_tokens: usize,
        start_position: usize,
        q_dim: usize,
        kv_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> anyhow::Result<bool> {
        if n_tokens <= 1 {
            return Ok(false);
        }
        let Some(metal_ops) = backend.metal_ops() else {
            return Ok(false);
        };

        // CPU: extract Q from Q+gate (strided memcpy — fast, ~1ms for 512 tokens).
        for token_idx in 0..n_tokens {
            let src = token_idx * q_dim * 2;
            let dst = token_idx * q_dim;
            q_batch[dst..dst + q_dim].copy_from_slice(&q_gate_batch[src..src + q_dim]);
        }

        // Cache QK norm weights on GPU if they exist.
        let norm_weights = Self::maybe_attention_qk_norm(weights, prefix)?;
        let q_norm_key = norm_weights.map(|nw| metal_ops.ensure_f32_cached(nw.q));
        let k_norm_key = norm_weights.map(|nw| metal_ops.ensure_f32_cached(nw.k));

        // Upload Q and K to GPU batch scratch buffers.
        metal_ops.init_batch_scratches(cfg, n_tokens);
        let mut bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_mut() else {
            return Ok(false);
        };
        unsafe {
            bs.q_buf.as_mut_slice::<f32>()[..n_tokens * q_dim]
                .copy_from_slice(&q_batch[..n_tokens * q_dim]);
            bs.k_buf.as_mut_slice::<f32>()[..n_tokens * kv_dim]
                .copy_from_slice(&k_batch[..n_tokens * kv_dim]);
        }

        let (rope_start, rope_step) = cfg.rope_scaling.scaled_start_step(start_position);
        let weight_cache = metal_ops.lock_weight_cache();

        metal_ops.device.execute_sync(|encoder| {
            // 1. Per-head QK norm (if applicable).
            if let (Some(q_key), Some(k_key)) = (q_norm_key, k_norm_key) {
                let q_nw = weight_cache.get(&q_key).unwrap();
                let k_nw = weight_cache.get(&k_key).unwrap();
                metal_ops.elementwise.encode_per_head_rms_norm_batch(
                    encoder,
                    &bs.q_buf,
                    q_nw,
                    n_tokens as u32,
                    n_heads as u32,
                    head_dim as u32,
                    cfg.rms_norm_eps,
                );
                metal_ops.elementwise.encode_per_head_rms_norm_batch(
                    encoder,
                    &bs.k_buf,
                    k_nw,
                    n_tokens as u32,
                    n_kv_heads as u32,
                    head_dim as u32,
                    cfg.rms_norm_eps,
                );
            }

            // 2. Batch RoPE on Q and K.
            metal_ops.elementwise.encode_rope_batch(
                encoder,
                &bs.q_buf,
                &bs.k_buf,
                n_tokens as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                rope_start,
                rope_step,
                cfg.rope_freq_base,
            );
            Ok(())
        })?;
        drop(weight_cache);

        // Read back Q and K.
        unsafe {
            q_batch[..n_tokens * q_dim]
                .copy_from_slice(&bs.q_buf.as_slice::<f32>()[..n_tokens * q_dim]);
            k_batch[..n_tokens * kv_dim]
                .copy_from_slice(&bs.k_buf.as_slice::<f32>()[..n_tokens * kv_dim]);
        }
        drop(bs_guard);
        Ok(true)
    }

    /// GPU-accelerated attention gate: sigmoid(gate) × attn_out.
    ///
    /// Replaces `apply_attention_gate_batch` per-token CPU loops with a single
    /// GPU dispatch. Falls back to CPU when Metal is unavailable.
    #[allow(clippy::too_many_arguments)]
    fn try_apply_attention_gate_batch_gpu(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        q_gate_batch: &[f32],
        attn_out_batch: &mut [f32],
        n_tokens: usize,
        q_dim: usize,
    ) -> anyhow::Result<bool> {
        if n_tokens <= 1 {
            return Ok(false);
        }
        let Some(metal_ops) = backend.metal_ops() else {
            return Ok(false);
        };

        let total = n_tokens * q_dim;

        // CPU: extract gate values to contiguous buffer (strided copy).
        let mut gate_contiguous = vec![0.0f32; total];
        for token_idx in 0..n_tokens {
            let src = token_idx * q_dim * 2 + q_dim;
            let dst = token_idx * q_dim;
            gate_contiguous[dst..dst + q_dim].copy_from_slice(&q_gate_batch[src..src + q_dim]);
        }

        metal_ops.init_batch_scratches(cfg, n_tokens);
        let mut bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_mut() else {
            return Ok(false);
        };

        // Upload gate and attn_out to GPU scratch.
        // gate → gate_buf (inter_dim >= q_dim), attn_out → attn_out (q_dim).
        unsafe {
            bs.gate_buf.as_mut_slice::<f32>()[..total].copy_from_slice(&gate_contiguous);
            bs.attn_out.as_mut_slice::<f32>()[..total].copy_from_slice(attn_out_batch);
        }

        metal_ops.device.execute_sync(|encoder| {
            metal_ops.elementwise.encode_sigmoid_elementwise_mul(
                encoder,
                &bs.gate_buf,
                &bs.attn_out,
                total as u32,
            );
            Ok(())
        })?;

        // Read back attn_out.
        unsafe {
            attn_out_batch[..total].copy_from_slice(&bs.attn_out.as_slice::<f32>()[..total]);
        }
        drop(bs_guard);
        Ok(true)
    }

    fn run_full_attention_batch_layer(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        layer: usize,
        batch_position: usize,
        norm_buf: &[f32],
        q_gate_batch: &mut [f32],
        q_batch: &mut [f32],
        k_batch: &mut [f32],
        v_batch: &mut [f32],
        fused_input_batch: &mut [f32],
        attn_out_batch: &mut [f32],
        proj_buf: &mut [f32],
        n_tokens: usize,
        dim: usize,
        q_dim: usize,
        kv_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        full_attn_params: &AttentionParams,
        skip_input_projections: bool,
    ) -> anyhow::Result<()> {
        if !skip_input_projections {
            let input_plan = Self::full_attention_input_plan(weights, prefix, q_dim, kv_dim)?;
            Self::project_full_attention_inputs_batch(
                input_plan,
                q_gate_batch,
                k_batch,
                v_batch,
                fused_input_batch,
                n_tokens,
                |raw, dtype, rows, out| {
                    Self::batched_dequant_matmul_token_major(
                        backend, raw, dtype, norm_buf, out, n_tokens, rows, dim,
                    );
                },
            );
        }

        // GPU-accelerated Q extraction + QK norm + RoPE (replaces per-token CPU loops).
        let used_gpu_qk = Self::try_prepare_full_attention_qk_batch_gpu(
            cfg,
            backend,
            weights,
            prefix,
            q_gate_batch,
            q_batch,
            k_batch,
            n_tokens,
            qwen_kv.seq_len(),
            q_dim,
            kv_dim,
            n_heads,
            n_kv_heads,
            head_dim,
        )?;
        if !used_gpu_qk {
            Self::prepare_full_attention_qk_batch(
                cfg,
                q_gate_batch,
                q_batch,
                k_batch,
                n_tokens,
                qwen_kv.seq_len(),
                q_dim,
                kv_dim,
                n_heads,
                n_kv_heads,
                head_dim,
                Self::maybe_attention_qk_norm(weights, prefix)?,
                cfg.rms_norm_eps,
            );
        }

        let used_gpu_prefix_kv = Self::try_full_attention_prefill_gpu_with_prefix(
            backend,
            qwen_kv,
            layer,
            q_batch,
            k_batch,
            v_batch,
            attn_out_batch,
            n_tokens,
            full_attn_params,
        )?;
        if !used_gpu_prefix_kv {
            Self::full_attention_prefill_batch(
                backend,
                qwen_kv,
                layer,
                q_batch,
                k_batch,
                v_batch,
                attn_out_batch,
                n_tokens,
                full_attn_params,
            );
        }

        // GPU-accelerated attention gate (replaces per-token CPU sigmoid+mul).
        let used_gpu_gate = Self::try_apply_attention_gate_batch_gpu(
            cfg,
            backend,
            q_gate_batch,
            attn_out_batch,
            n_tokens,
            q_dim,
        )?;
        if !used_gpu_gate {
            Self::apply_attention_gate_batch(q_gate_batch, attn_out_batch, n_tokens, q_dim);
        }

        if !used_gpu_prefix_kv {
            qwen_kv.attention_append_batch(layer, k_batch, v_batch, n_tokens);
        }

        let (wo_raw, wo_dtype, _) = Self::full_attention_output_op(weights, prefix, dim)?;
        Self::batched_dequant_matmul_token_major(
            backend,
            wo_raw,
            wo_dtype,
            attn_out_batch,
            proj_buf,
            n_tokens,
            dim,
            q_dim,
        );
        Self::assert_finite_if_enabled(
            "full_attention_proj_batch",
            proj_buf,
            layer,
            batch_position,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_full_attention_single_layer(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        layer: usize,
        position: usize,
        norm_buf: &[f32],
        q_gate_buf: &mut [f32],
        q_buf: &mut [f32],
        k_buf: &mut [f32],
        v_buf: &mut [f32],
        attn_out: &mut [f32],
        proj_buf: &mut [f32],
        dim: usize,
        q_dim: usize,
        kv_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        full_attn_params: &AttentionParams,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        let input_ops = Self::full_attention_input_ops(weights, prefix, q_dim, kv_dim)?;

        timed_matmul_bucket!(ops, matmul_input_proj, {
            let mut outputs = [&mut *q_gate_buf, &mut *k_buf, &mut *v_buf];
            Self::decode_project_ops_gpu_safe(backend, &input_ops, norm_buf, dim, &mut outputs);
        });
        Self::extract_q_from_q_gate(q_gate_buf, q_buf);
        let gate_attn = &mut q_gate_buf[q_dim..];

        if let Some(norm_weights) = timed!(
            ops,
            dequant,
            Self::maybe_attention_qk_norm(weights, prefix)?
        ) {
            timed!(ops, norm, {
                Self::apply_attention_qk_norm(
                    q_buf,
                    k_buf,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    norm_weights,
                    cfg.rms_norm_eps,
                );
            });
        }

        timed!(
            ops,
            rope,
            Self::apply_rope(cfg, q_buf, k_buf, position, n_heads, n_kv_heads, head_dim,)
        );

        qwen_kv.attention_append(layer, k_buf, v_buf);
        let seq_len = qwen_kv.seq_len() + 1;
        let used_gpu_attention = timed!(
            ops,
            attention,
            Self::try_full_attention_decode_gpu(
                cfg,
                backend,
                qwen_kv,
                layer,
                q_buf,
                attn_out,
                full_attn_params,
            )?
        );
        if !used_gpu_attention {
            timed!(
                ops,
                attention,
                attention::multi_head_attention(
                    q_buf,
                    qwen_kv.attention_k_slice_including_current(layer, seq_len),
                    qwen_kv.attention_v_slice_including_current(layer, seq_len),
                    attn_out,
                    full_attn_params,
                    seq_len,
                )
            );
        }

        Self::apply_attention_gate(gate_attn, attn_out);

        let (wo_raw, wo_dtype, _) = timed!(
            ops,
            dequant,
            Self::full_attention_output_op(weights, prefix, dim)?
        );
        timed_matmul_bucket!(
            ops,
            matmul_output_proj,
            Self::decode_dequant_matmul_gpu_safe(
                backend, wo_raw, wo_dtype, attn_out, proj_buf, dim, q_dim
            )
        );
        Self::assert_finite_if_enabled("full_attention_proj", proj_buf, layer, position)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_recurrent_batch_layer(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        recurrent_slot: usize,
        layer: usize,
        batch_position: usize,
        dims: Qwen35RecurrentDims,
        recurrent_slot_indices: &[usize],
        norm_buf: &[f32],
        rec_qkv_batch: &mut [f32],
        rec_z_batch: &mut [f32],
        rec_beta_batch: &mut [f32],
        rec_alpha_batch: &mut [f32],
        rec_out_batch: &mut [f32],
        proj_buf: &mut [f32],
        n_tokens: usize,
        dim: usize,
        skip_input_projections: bool,
    ) -> anyhow::Result<()> {
        Self::validate_recurrent_layer_state(qwen_kv, recurrent_slot, layer, "prefill")?;
        if !skip_input_projections {
            let input_ops = Self::recurrent_input_ops(weights, prefix, dims)?;
            Self::project_recurrent_inputs(
                input_ops,
                [rec_qkv_batch, rec_z_batch, rec_beta_batch, rec_alpha_batch],
                |raw, dtype, rows, out| {
                    Self::batched_dequant_matmul_token_major(
                        backend, raw, dtype, norm_buf, out, n_tokens, rows, dim,
                    );
                },
            );
        }
        Self::assert_finite_if_enabled(
            "recurrent_qkv_input_batch",
            rec_qkv_batch,
            layer,
            batch_position,
        )?;

        rec_out_batch.fill(0.0);
        let runtime = Self::run_recurrent_sequence(
            backend,
            weights,
            prefix,
            qwen_kv,
            layer,
            recurrent_slot_indices,
            dims,
            cfg.rms_norm_eps,
            rec_qkv_batch,
            rec_beta_batch,
            rec_alpha_batch,
            rec_out_batch,
            n_tokens,
        )?;
        Self::assert_finite_if_enabled(
            "recurrent_kernel_output_batch",
            rec_out_batch,
            layer,
            batch_position,
        )?;

        let used_gpu_finalize = Self::try_finalize_recurrent_output_batch_gpu(
            cfg,
            backend,
            rec_out_batch,
            rec_z_batch,
            n_tokens,
            dims,
            runtime.ssm_norm,
            cfg.rms_norm_eps,
        )?;
        if !used_gpu_finalize {
            Self::finalize_recurrent_output_batch(
                rec_out_batch,
                rec_z_batch,
                n_tokens,
                dims,
                runtime.ssm_norm,
                cfg.rms_norm_eps,
            );
        }
        Self::assert_finite_if_enabled(
            "recurrent_output_batch",
            rec_out_batch,
            layer,
            batch_position,
        )?;

        let (ssm_out_raw, ssm_out_dtype, _) = Self::recurrent_output_op(weights, prefix, dim)?;
        Self::batched_dequant_matmul_token_major(
            backend,
            ssm_out_raw,
            ssm_out_dtype,
            rec_out_batch,
            proj_buf,
            n_tokens,
            dim,
            dims.inner_size,
        );
        Self::assert_finite_if_enabled("recurrent_proj_batch", proj_buf, layer, batch_position)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_recurrent_single_layer(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        recurrent_slot: usize,
        layer: usize,
        position: usize,
        dims: Qwen35RecurrentDims,
        recurrent_slot_indices: &[usize],
        norm_buf: &[f32],
        rec_qkv: &mut [f32],
        rec_z: &mut [f32],
        rec_beta: &mut [f32],
        rec_alpha: &mut [f32],
        rec_out: &mut [f32],
        proj_buf: &mut [f32],
        dim: usize,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        Self::validate_recurrent_layer_state(qwen_kv, recurrent_slot, layer, "decode")?;
        let input_ops = Self::recurrent_input_ops(weights, prefix, dims)?;
        timed_matmul_bucket!(ops, matmul_input_proj, {
            let mut outputs = [&mut *rec_qkv, &mut *rec_z, &mut *rec_beta, &mut *rec_alpha];
            Self::decode_project_ops_gpu_safe(backend, &input_ops, norm_buf, dim, &mut outputs);
        });
        Self::assert_finite_if_enabled("recurrent_qkv_input", rec_qkv, layer, position)?;
        Self::assert_finite_if_enabled(
            "recurrent_state_before_decode",
            qwen_kv.recurrent_state_for_slot(recurrent_slot, layer),
            layer,
            position,
        )?;

        // Recurrent state staging is backend-owned now so the model does not
        // need to know how state is mirrored.
        let runtime = timed!(
            ops,
            recurrent,
            Self::run_recurrent_sequence(
                backend,
                weights,
                prefix,
                qwen_kv,
                layer,
                recurrent_slot_indices,
                dims,
                cfg.rms_norm_eps,
                rec_qkv,
                rec_beta,
                rec_alpha,
                rec_out,
                1,
            )
        )?;
        Self::assert_finite_if_enabled("recurrent_kernel_output", rec_out, layer, position)?;

        Self::finalize_recurrent_output(rec_out, rec_z, dims, runtime.ssm_norm, cfg.rms_norm_eps);
        Self::assert_finite_if_enabled("recurrent_output", rec_out, layer, position)?;

        let (ssm_out_raw, ssm_out_dtype, _) = timed!(
            ops,
            dequant,
            Self::recurrent_output_op(weights, prefix, dim)?
        );
        timed_matmul_bucket!(
            ops,
            matmul_output_proj,
            Self::decode_dequant_matmul_gpu_safe(
                backend,
                ssm_out_raw,
                ssm_out_dtype,
                rec_out,
                proj_buf,
                dim,
                dims.inner_size,
            )
        );
        Self::assert_finite_if_enabled("recurrent_proj", proj_buf, layer, position)?;
        Ok(())
    }

    /// GPU-encoded layer tail: residual + FFN norm + gate/up matmul + SiLU +
    /// down matmul + final residual, all in one Metal command buffer.
    ///
    /// Returns `true` if the GPU path was used, `false` to fall back to CPU.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn try_apply_layer_tail_batch_gpu(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        hidden: &mut [f32],
        proj_buf: &[f32],
        n_tokens: usize,
        dim: usize,
        inter_dim: usize,
        rms_norm_eps: f32,
    ) -> anyhow::Result<bool> {
        let Some(metal_ops) = backend.metal_ops() else {
            return Ok(false);
        };
        let input_ops = Self::ffn_input_ops(weights, prefix, inter_dim)?;
        let (wd_raw, wd_dtype, _) = Self::ffn_down_op(weights, prefix, dim)?;
        let post_attn_norm_w =
            weights.f32_slice(&format!("{prefix}.post_attention_norm.weight"))?;

        // Check all dtypes are supported by fused batch kernels.
        let (wg_raw, wg_dtype, _) = input_ops[0];
        let (wu_raw, wu_dtype, _) = input_ops[1];
        let supported = |dt: crate::gguf::tensor::GgmlType| {
            matches!(
                dt,
                crate::gguf::tensor::GgmlType::Q4K
                    | crate::gguf::tensor::GgmlType::Q5K
                    | crate::gguf::tensor::GgmlType::Q6K
            )
        };
        if !supported(wg_dtype) || !supported(wu_dtype) || !supported(wd_dtype) {
            return Ok(false);
        }

        // Cache weight buffers.
        let wg_key = metal_ops.ensure_quant_cached(wg_raw);
        let wu_key = metal_ops.ensure_quant_cached(wu_raw);
        let wd_key = metal_ops.ensure_quant_cached(wd_raw);
        let nw_key = metal_ops.ensure_f32_cached(post_attn_norm_w);

        metal_ops.init_batch_scratches(cfg, n_tokens);

        let nt = n_tokens as u32;
        let eps = rms_norm_eps;

        let mut bs_guard = metal_ops.batch_scratches();
        let Some(bs) = bs_guard.as_mut() else {
            return Ok(false);
        };

        // Copy hidden + proj to GPU.
        unsafe {
            bs.hidden.as_mut_slice::<f32>()[..n_tokens * dim].copy_from_slice(hidden);
            bs.proj_buf.as_mut_slice::<f32>()[..n_tokens * dim].copy_from_slice(proj_buf);
        }

        let weight_cache = metal_ops.lock_weight_cache();
        let nw_buf = weight_cache.get(&nw_key).unwrap();
        let wg_buf = weight_cache.get(&wg_key).unwrap();
        let wu_buf = weight_cache.get(&wu_key).unwrap();
        let wd_buf = weight_cache.get(&wd_key).unwrap();

        metal_ops.device.execute_sync(|encoder| {
            // 1. Residual add: hidden += proj_buf
            metal_ops.elementwise.encode_elementwise_add_batch(
                encoder,
                &bs.hidden,
                &bs.proj_buf,
                dim as u32,
                nt,
            );
            // 2. RMSNorm: norm_buf = RMSNorm(hidden)
            metal_ops.elementwise.encode_rms_norm_out_batch(
                encoder,
                &bs.hidden,
                nw_buf,
                &bs.norm_buf,
                dim as u32,
                nt,
                eps,
            );
            // 3. Gate matmul: gate_buf = dequant(Wg) × norm_buf^T
            encode_dequant_batch(
                &metal_ops.dequant,
                &metal_ops.elementwise,
                encoder,
                wg_buf,
                &bs.norm_buf,
                &bs.gate_buf,
                &bs.matmul_in_f16,
                inter_dim as u32,
                nt,
                dim as u32,
                wg_dtype,
                false,
                false,
                false,
            );
            // 4. Up matmul: up_buf = dequant(Wu) × norm_buf^T
            encode_dequant_batch(
                &metal_ops.dequant,
                &metal_ops.elementwise,
                encoder,
                wu_buf,
                &bs.norm_buf,
                &bs.up_buf,
                &bs.matmul_in_f16,
                inter_dim as u32,
                nt,
                dim as u32,
                wu_dtype,
                false,
                false,
                false,
            );
            // 5. SiLU: gate_buf = SiLU(gate_buf) * up_buf
            metal_ops.elementwise.encode_silu_elementwise_mul_batch(
                encoder,
                &bs.gate_buf,
                &bs.up_buf,
                inter_dim as u32,
                nt,
            );
            // 6. Down matmul: proj_buf = dequant(Wd) × gate_buf^T
            encode_dequant_batch(
                &metal_ops.dequant,
                &metal_ops.elementwise,
                encoder,
                wd_buf,
                &bs.gate_buf,
                &bs.proj_buf,
                &bs.matmul_in_f16,
                dim as u32,
                nt,
                inter_dim as u32,
                wd_dtype,
                false,
                false,
                false,
            );
            // 7. Final residual: hidden += proj_buf (down output)
            metal_ops.elementwise.encode_elementwise_add_batch(
                encoder,
                &bs.hidden,
                &bs.proj_buf,
                dim as u32,
                nt,
            );
            Ok(())
        })?;
        drop(weight_cache);

        // Read back updated hidden.
        let result = unsafe { &bs.hidden.as_slice::<f32>()[..n_tokens * dim] };
        hidden.copy_from_slice(result);
        drop(bs_guard);

        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_layer_tail_batch(
        cfg: &ModelConfig,
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        hidden: &mut [f32],
        proj_buf: &[f32],
        norm_buf: &mut [f32],
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        down_buf: &mut [f32],
        n_tokens: usize,
        dim: usize,
        inter_dim: usize,
        rms_norm_eps: f32,
        layer: usize,
        batch_position: usize,
    ) -> anyhow::Result<()> {
        // Try GPU-encoded path first (single command buffer for entire FFN).
        if n_tokens > 1 {
            if let Ok(true) = Self::try_apply_layer_tail_batch_gpu(
                cfg,
                backend,
                weights,
                prefix,
                hidden,
                proj_buf,
                n_tokens,
                dim,
                inter_dim,
                rms_norm_eps,
            ) {
                Self::assert_finite_if_enabled(
                    "post_ffn_hidden_batch",
                    hidden,
                    layer,
                    batch_position,
                )?;
                return Ok(());
            }
        }
        // CPU fallback.
        silu::elementwise_add(hidden, proj_buf);
        Self::assert_finite_if_enabled("layer_hidden_batch", hidden, layer, batch_position)?;
        Self::apply_post_attention_ffn_batch(
            backend,
            weights,
            prefix,
            hidden,
            norm_buf,
            gate_buf,
            up_buf,
            down_buf,
            n_tokens,
            dim,
            inter_dim,
            rms_norm_eps,
        )?;
        Self::assert_finite_if_enabled("post_ffn_hidden_batch", hidden, layer, batch_position)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_layer_tail_single(
        backend: &dyn crate::backend::Backend,
        weights: &WeightStore,
        prefix: &str,
        hidden: &mut [f32],
        proj_buf: &[f32],
        norm_buf: &mut [f32],
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        down_buf: &mut [f32],
        dim: usize,
        inter_dim: usize,
        rms_norm_eps: f32,
        layer: usize,
        position: usize,
        ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        silu::elementwise_add(hidden, proj_buf);
        Self::assert_finite_if_enabled("layer_hidden", hidden, layer, position)?;
        Self::apply_post_attention_ffn_single(
            backend,
            weights,
            prefix,
            hidden,
            norm_buf,
            gate_buf,
            up_buf,
            down_buf,
            dim,
            inter_dim,
            rms_norm_eps,
            ops,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn write_all_batch_logits(
        backend: &dyn crate::backend::Backend,
        hidden: &[f32],
        n_tokens: usize,
        dim: usize,
        vocab_size: usize,
        rms_norm_eps: f32,
        weights: &WeightStore,
        logits_all: &mut Vec<f32>,
    ) -> anyhow::Result<()> {
        let final_norm_w = weights.f32_slice("output_norm.weight")?;
        let mut final_hidden = vec![0.0f32; hidden.len()];
        Self::rms_norm_token_major(
            hidden,
            final_norm_w,
            &mut final_hidden,
            n_tokens,
            dim,
            rms_norm_eps,
        );
        Self::assert_finite_if_enabled("final_norm_batch", &final_hidden, 0, 0)?;
        logits_all.resize(n_tokens * vocab_size, 0.0);
        // Qwen3.5 speculative verify currently stays finite through the
        // batched hybrid forward path and only destabilizes in the final
        // batched LM-head write. Reuse the proven per-token LM-head route for
        // all-logits emission until the batched logits projection is fixed.
        Self::write_normalized_batch_logits(
            backend,
            &final_hidden,
            n_tokens,
            dim,
            vocab_size,
            weights,
            logits_all.as_mut_slice(),
        )?;
        Self::assert_finite_if_enabled("logits_all_batch", logits_all.as_slice(), 0, 0)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_serial_fallback(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        start_position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: Option<&mut [f32]>,
        logits_all: Option<&mut Vec<f32>>,
    ) -> anyhow::Result<()> {
        match (logits, logits_all) {
            (Some(logits), None) => {
                for (i, &tid) in token_ids.iter().enumerate() {
                    logits.fill(0.0);
                    self.forward_single(ctx, tid, start_position + i, kv, weights, logits, None)?;
                }
                Ok(())
            }
            (None, Some(logits_all)) => {
                let vocab_size = ctx.config.vocab_size as usize;
                logits_all.resize(token_ids.len() * vocab_size, 0.0);
                for (i, &tid) in token_ids.iter().enumerate() {
                    let slot = &mut logits_all[i * vocab_size..(i + 1) * vocab_size];
                    slot.fill(0.0);
                    self.forward_single(ctx, tid, start_position + i, kv, weights, slot, None)?;
                }
                Ok(())
            }
            _ => anyhow::bail!(
                "qwen35 batch forward requires either last logits or all logits output"
            ),
        }
    }

    /// GPU-unified prefill: keeps hidden on GPU, encodes full-attention layers
    /// in single command buffers. Falls back to `forward_batch_impl` on failure.
    #[allow(clippy::too_many_arguments)]
    fn try_forward_batch_gpu_unified(
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: Option<&mut [f32]>,
        logits_all: Option<&mut Vec<f32>>,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<bool> {
        let total_t = OpTimer::start();
        let Some(metal_ops) = ctx.backend.metal_ops() else {
            return Ok(false);
        };
        let cfg = ctx.config;
        let backend = ctx.backend;
        let Some(qwen_kv) = kv.as_qwen35_mut() else {
            return Ok(false);
        };

        let dims = Self::recurrent_dims(cfg)?;
        let n_tokens = token_ids.len();
        let n_layers = cfg.n_layers as usize;
        let dim = cfg.embedding_dim as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let vocab_size = cfg.vocab_size as usize;
        let eps = cfg.rms_norm_eps;
        let batch_position = qwen_kv.seq_len();
        let recurrent_slot = qwen_kv.active_slot();
        let recurrent_slot_indices = [recurrent_slot];
        if qwen_kv.gpu_attention().is_some()
            && !qwen_kv.ensure_gpu_attention_capacity_for(batch_position + n_tokens)
        {
            return Ok(false);
        }

        // Check that all projection dtypes are GPU-supported for this batch shape.
        let gpu_batch_supported = |dt: crate::gguf::tensor::GgmlType, m: u32, k: u32| {
            Self::qwen35_batch_projection_supported(metal_ops, dt, m, n_tokens as u32, k)
        };

        // Quick check: verify at least one full-attention layer's weights are supported.
        for layer in 0..n_layers {
            if !cfg.qwen35_is_recurrent_layer(layer) {
                let prefix = format!("blk.{layer}");
                let (_, dt) = weights.raw_with_dtype(&format!("{prefix}.attn_q.weight"))?;
                if !gpu_batch_supported(dt, (q_dim * 2) as u32, dim as u32) {
                    return Ok(false);
                }
                break;
            }
        }

        // Init GPU batch scratch buffers.
        metal_ops.init_batch_scratches(cfg, n_tokens);

        let nt = n_tokens as u32;

        // Embed tokens directly into GPU hidden buffer (UMA write).
        {
            let mut bs_guard = metal_ops.batch_scratches();
            let Some(bs) = bs_guard.as_mut() else {
                return Ok(false);
            };
            let h = unsafe {
                std::slice::from_raw_parts_mut(
                    bs.hidden.contents().as_ptr() as *mut f32,
                    n_tokens * dim,
                )
            };
            for (i, &tid) in token_ids.iter().enumerate() {
                weights.dequantize_row(
                    "token_embd.weight",
                    tid as usize,
                    &mut h[i * dim..(i + 1) * dim],
                )?;
            }
        }

        // CPU buffers for recurrent layers.
        let mut rec_qkv_batch = vec![0.0f32; n_tokens * dims.conv_dim()];
        let mut rec_z_batch = vec![0.0f32; n_tokens * dims.inner_size];
        let mut rec_beta_batch = vec![0.0f32; n_tokens * dims.time_step_rank];
        let mut rec_alpha_batch = vec![0.0f32; n_tokens * dims.time_step_rank];
        let mut rec_out_batch = vec![0.0f32; n_tokens * dims.inner_size];

        let (rope_start, rope_step) = cfg.rope_scaling.scaled_start_step(batch_position);

        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");
            let is_recurrent = cfg.qwen35_is_recurrent_layer(layer);

            // ── Cache layer weights ──
            let attn_norm_w = weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?;
            let nw_key = metal_ops.ensure_f32_cached(attn_norm_w);
            let ffn_norm_w = weights.f32_slice(&format!("{prefix}.post_attention_norm.weight"))?;
            let ffn_nw_key = metal_ops.ensure_f32_cached(ffn_norm_w);

            let (wg_raw, wg_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.ffn_gate.weight"))?;
            let (wu_raw, wu_dtype) = weights.raw_with_dtype(&format!("{prefix}.ffn_up.weight"))?;
            let (wd_raw, wd_dtype) =
                weights.raw_with_dtype(&format!("{prefix}.ffn_down.weight"))?;
            if !gpu_batch_supported(wg_dtype, inter_dim as u32, dim as u32)
                || !gpu_batch_supported(wu_dtype, inter_dim as u32, dim as u32)
                || !gpu_batch_supported(wd_dtype, dim as u32, inter_dim as u32)
            {
                return Ok(false);
            }
            Self::prepare_qwen35_batch_projection_weight(
                metal_ops,
                wg_raw,
                wg_dtype,
                inter_dim as u32,
                dim as u32,
            )?;
            Self::prepare_qwen35_batch_projection_weight(
                metal_ops,
                wu_raw,
                wu_dtype,
                inter_dim as u32,
                dim as u32,
            )?;
            Self::prepare_qwen35_batch_projection_weight(
                metal_ops,
                wd_raw,
                wd_dtype,
                dim as u32,
                inter_dim as u32,
            )?;
            let wg_key = metal_ops.ensure_quant_cached(wg_raw);
            let wu_key = metal_ops.ensure_quant_cached(wu_raw);
            let wd_key = metal_ops.ensure_quant_cached(wd_raw);

            if !is_recurrent {
                // ══════════════════════════════════════════════════════════
                // Full-attention layer: single CB with GPU Q+gate split
                // ══════════════════════════════════════════════════════════
                let (wq_raw, wq_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.attn_q.weight"))?;
                let (wk_raw, wk_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.attn_k.weight"))?;
                let (wv_raw, wv_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.attn_v.weight"))?;
                let (wo_raw, wo_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.attn_output.weight"))?;
                if !gpu_batch_supported(wq_dtype, (q_dim * 2) as u32, dim as u32)
                    || !gpu_batch_supported(wk_dtype, kv_dim as u32, dim as u32)
                    || !gpu_batch_supported(wv_dtype, kv_dim as u32, dim as u32)
                    || !gpu_batch_supported(wo_dtype, dim as u32, q_dim as u32)
                {
                    return Ok(false);
                }
                Self::prepare_qwen35_batch_projection_weight(
                    metal_ops,
                    wq_raw,
                    wq_dtype,
                    (q_dim * 2) as u32,
                    dim as u32,
                )?;
                Self::prepare_qwen35_batch_projection_weight(
                    metal_ops,
                    wk_raw,
                    wk_dtype,
                    kv_dim as u32,
                    dim as u32,
                )?;
                Self::prepare_qwen35_batch_projection_weight(
                    metal_ops,
                    wv_raw,
                    wv_dtype,
                    kv_dim as u32,
                    dim as u32,
                )?;
                Self::prepare_qwen35_batch_projection_weight(
                    metal_ops,
                    wo_raw,
                    wo_dtype,
                    dim as u32,
                    q_dim as u32,
                )?;
                let wq_key = metal_ops.ensure_quant_cached(wq_raw);
                let wk_key = metal_ops.ensure_quant_cached(wk_raw);
                let wv_key = metal_ops.ensure_quant_cached(wv_raw);
                let wo_key = metal_ops.ensure_quant_cached(wo_raw);

                let norm_weights = Self::maybe_attention_qk_norm(weights, &prefix)?;
                let q_norm_key = norm_weights.map(|nw| metal_ops.ensure_f32_cached(nw.q));
                let k_norm_key = norm_weights.map(|nw| metal_ops.ensure_f32_cached(nw.k));

                {
                    let mut bs_guard = metal_ops.batch_scratches();
                    let Some(bs) = bs_guard.as_mut() else {
                        return Ok(false);
                    };
                    let weight_cache = metal_ops.lock_weight_cache();
                    let nw_buf = weight_cache.get(&nw_key).unwrap();
                    let wq_buf = weight_cache.get(&wq_key).unwrap();
                    let wk_buf = weight_cache.get(&wk_key).unwrap();
                    let wv_buf = weight_cache.get(&wv_key).unwrap();
                    let wo_buf = weight_cache.get(&wo_key).unwrap();
                    let ffn_nw_buf = weight_cache.get(&ffn_nw_key).unwrap();
                    let wg_buf = weight_cache.get(&wg_key).unwrap();
                    let wu_buf = weight_cache.get(&wu_key).unwrap();
                    let wd_buf = weight_cache.get(&wd_key).unwrap();

                    // Single CB: norm → QKV → split → QK norm → RoPE → KV append
                    //          → attention → gate → WO → residual → FFN
                    let cb_t = OpTimer::start();
                    metal_ops.device.execute_sync(|encoder| {
                        let qkv_uses_f16 = Self::qwen35_batch_projection_needs_f16_input(wq_dtype)
                            || Self::qwen35_batch_projection_needs_f16_input(wk_dtype)
                            || Self::qwen35_batch_projection_needs_f16_input(wv_dtype);
                        let ffn_input_uses_f16 =
                            Self::qwen35_batch_projection_needs_f16_input(wg_dtype)
                                || Self::qwen35_batch_projection_needs_f16_input(wu_dtype);
                        // 1. RMSNorm: hidden → norm_buf
                        metal_ops.elementwise.encode_rms_norm_out_batch(
                            encoder,
                            &bs.hidden,
                            nw_buf,
                            &bs.norm_buf,
                            dim as u32,
                            nt,
                            eps,
                        );
                        if qkv_uses_f16 {
                            metal_ops.elementwise.encode_cast_f32_to_f16(
                                encoder,
                                &bs.norm_buf,
                                &bs.matmul_in_f16,
                                nt * dim as u32,
                            );
                        }
                        // 2. Q+gate matmul: norm_buf → gate_buf [N × q_dim*2]
                        Self::encode_qwen35_batch_projection(
                            metal_ops,
                            encoder,
                            wq_buf,
                            &bs.norm_buf,
                            &bs.matmul_in_f16,
                            &bs.gate_buf,
                            (q_dim * 2) as u32,
                            nt,
                            dim as u32,
                            wq_dtype,
                        );
                        // 3. K matmul: norm_buf → k_buf
                        Self::encode_qwen35_batch_projection(
                            metal_ops,
                            encoder,
                            wk_buf,
                            &bs.norm_buf,
                            &bs.matmul_in_f16,
                            &bs.k_buf,
                            kv_dim as u32,
                            nt,
                            dim as u32,
                            wk_dtype,
                        );
                        // 4. V matmul: norm_buf → v_buf
                        Self::encode_qwen35_batch_projection(
                            metal_ops,
                            encoder,
                            wv_buf,
                            &bs.norm_buf,
                            &bs.matmul_in_f16,
                            &bs.v_buf,
                            kv_dim as u32,
                            nt,
                            dim as u32,
                            wv_dtype,
                        );
                        // 5. GPU Q+gate split: gate_buf → q_buf (Q), up_buf (gate)
                        metal_ops.elementwise.encode_split_qgate_batch(
                            encoder,
                            &bs.gate_buf,
                            &bs.q_buf,
                            &bs.up_buf,
                            nt,
                            q_dim as u32,
                        );
                        // 6. QK norm (if applicable)
                        if let (Some(q_key), Some(k_key)) = (q_norm_key, k_norm_key) {
                            let q_nw = weight_cache.get(&q_key).unwrap();
                            let k_nw = weight_cache.get(&k_key).unwrap();
                            metal_ops.elementwise.encode_per_head_rms_norm_batch(
                                encoder,
                                &bs.q_buf,
                                q_nw,
                                nt,
                                n_heads as u32,
                                head_dim as u32,
                                eps,
                            );
                            metal_ops.elementwise.encode_per_head_rms_norm_batch(
                                encoder,
                                &bs.k_buf,
                                k_nw,
                                nt,
                                n_kv_heads as u32,
                                head_dim as u32,
                                eps,
                            );
                        }
                        // 7. RoPE
                        metal_ops.elementwise.encode_rope_batch(
                            encoder,
                            &bs.q_buf,
                            &bs.k_buf,
                            nt,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            rope_start,
                            rope_step,
                            cfg.rope_freq_base,
                        );
                        // 8. KV append (GPU f32/f16 aware)
                        if let Some(gpu_attn) = qwen_kv.gpu_attention() {
                            let cache_offset = (batch_position * kv_dim) as u32;
                            metal_ops.elementwise.encode_kv_append_batch(
                                encoder,
                                &bs.k_buf,
                                gpu_attn.k_buffer(layer),
                                gpu_attn.is_f16(),
                                cache_offset,
                                kv_dim as u32,
                                kv_dim as u32,
                                nt,
                            );
                            metal_ops.elementwise.encode_kv_append_batch(
                                encoder,
                                &bs.v_buf,
                                gpu_attn.v_buffer(layer),
                                gpu_attn.is_f16(),
                                cache_offset,
                                kv_dim as u32,
                                kv_dim as u32,
                                nt,
                            );
                        }
                        // 9. Attention prefill
                        if batch_position == 0 {
                            metal_ops.attention.encode_attention_prefill_with_config(
                                encoder,
                                &bs.q_buf,
                                &bs.k_buf,
                                &bs.v_buf,
                                &bs.attn_out,
                                nt,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                metal_ops.attention_dispatch_config(),
                            );
                        } else if let Some(gpu_attn) = qwen_kv.gpu_attention() {
                            metal_ops
                                .attention
                                .encode_attention_prefill_cached_with_config(
                                    encoder,
                                    &bs.q_buf,
                                    gpu_attn.k_buffer(layer),
                                    gpu_attn.v_buffer(layer),
                                    &bs.attn_out,
                                    gpu_attn.is_f16(),
                                    nt,
                                    n_heads as u32,
                                    n_kv_heads as u32,
                                    head_dim as u32,
                                    batch_position as u32,
                                    0,
                                    metal_ops.attention_dispatch_config(),
                                );
                        }
                        // 10. Attention gate: sigmoid(up_buf) × attn_out
                        metal_ops.elementwise.encode_sigmoid_elementwise_mul(
                            encoder,
                            &bs.up_buf,
                            &bs.attn_out,
                            (n_tokens * q_dim) as u32,
                        );
                        if Self::qwen35_batch_projection_needs_f16_input(wo_dtype) {
                            metal_ops.elementwise.encode_cast_f32_to_f16(
                                encoder,
                                &bs.attn_out,
                                &bs.matmul_in_f16,
                                nt * q_dim as u32,
                            );
                        }
                        // 11. WO projection: attn_out → proj_buf
                        Self::encode_qwen35_batch_projection(
                            metal_ops,
                            encoder,
                            wo_buf,
                            &bs.attn_out,
                            &bs.matmul_in_f16,
                            &bs.proj_buf,
                            dim as u32,
                            nt,
                            q_dim as u32,
                            wo_dtype,
                        );
                        // 12. Residual + FFN norm
                        metal_ops
                            .elementwise
                            .encode_residual_add_rms_norm_out_batch(
                                encoder,
                                &bs.hidden,
                                &bs.proj_buf,
                                ffn_nw_buf,
                                &bs.norm_buf,
                                dim as u32,
                                nt,
                                eps,
                            );
                        if ffn_input_uses_f16 {
                            metal_ops.elementwise.encode_cast_f32_to_f16(
                                encoder,
                                &bs.norm_buf,
                                &bs.matmul_in_f16,
                                nt * dim as u32,
                            );
                        }
                        // 13-14. Gate + Up matmul
                        Self::encode_qwen35_batch_projection(
                            metal_ops,
                            encoder,
                            wg_buf,
                            &bs.norm_buf,
                            &bs.matmul_in_f16,
                            &bs.gate_buf,
                            inter_dim as u32,
                            nt,
                            dim as u32,
                            wg_dtype,
                        );
                        Self::encode_qwen35_batch_projection(
                            metal_ops,
                            encoder,
                            wu_buf,
                            &bs.norm_buf,
                            &bs.matmul_in_f16,
                            &bs.up_buf,
                            inter_dim as u32,
                            nt,
                            dim as u32,
                            wu_dtype,
                        );
                        // 15. SiLU
                        metal_ops.elementwise.encode_silu_elementwise_mul_batch(
                            encoder,
                            &bs.gate_buf,
                            &bs.up_buf,
                            inter_dim as u32,
                            nt,
                        );
                        if Self::qwen35_batch_projection_needs_f16_input(wd_dtype) {
                            metal_ops.elementwise.encode_cast_f32_to_f16(
                                encoder,
                                &bs.gate_buf,
                                &bs.matmul_in_f16,
                                nt * inter_dim as u32,
                            );
                        }
                        // 16. Down projection: gate_buf → proj_buf
                        Self::encode_qwen35_batch_projection(
                            metal_ops,
                            encoder,
                            wd_buf,
                            &bs.gate_buf,
                            &bs.matmul_in_f16,
                            &bs.proj_buf,
                            dim as u32,
                            nt,
                            inter_dim as u32,
                            wd_dtype,
                        );
                        // 17. Final residual: hidden += proj_buf
                        metal_ops.elementwise.encode_elementwise_add_batch(
                            encoder,
                            &bs.hidden,
                            &bs.proj_buf,
                            dim as u32,
                            nt,
                        );
                        Ok(())
                    })?;
                    if let Some(ref mut ops) = ops {
                        let elapsed = cb_t.elapsed();
                        ops.gpu_execute += elapsed;
                        ops.gpu_execute_layers += elapsed;
                    }

                    // CPU-only mirror: the GPU KV append already happened in the
                    // command buffer above.
                    {
                        let k_after_rope =
                            unsafe { &bs.k_buf.as_slice::<f32>()[..n_tokens * kv_dim] };
                        let v_slice = unsafe { &bs.v_buf.as_slice::<f32>()[..n_tokens * kv_dim] };
                        qwen_kv.attention_append_batch_cpu_mirror(
                            layer,
                            k_after_rope,
                            v_slice,
                            n_tokens,
                        );
                    }
                }
            } else {
                // ══════════════════════════════════════════════════════════
                // Recurrent layer: read hidden from GPU, process, write back
                // ══════════════════════════════════════════════════════════

                // Cache quantized weights BEFORE locking weight_cache (avoid deadlock).
                let input_ops = Self::recurrent_input_ops(weights, &prefix, dims)?;
                // Route only GPU-batch-supported dtypes to GPU; everything else to CPU.
                let is_gpu_encodable = &gpu_batch_supported;
                // Separate GPU-encodable (quantized) from CPU-only (F32) projections.
                let mut gpu_proj_indices = Vec::new();
                let mut cpu_proj_indices = Vec::new();
                let mut proj_keys = Vec::new();
                let mut proj_dtypes = Vec::new();
                let mut proj_dims: Vec<usize> = Vec::new();
                for (i, (raw, dtype, out_dim)) in input_ops.iter().enumerate() {
                    if is_gpu_encodable(*dtype, *out_dim as u32, dim as u32) {
                        Self::prepare_qwen35_batch_projection_weight(
                            metal_ops,
                            raw,
                            *dtype,
                            *out_dim as u32,
                            dim as u32,
                        )?;
                    }
                    proj_keys.push(metal_ops.ensure_quant_cached(raw));
                    proj_dtypes.push(*dtype);
                    proj_dims.push(*out_dim);
                    if is_gpu_encodable(*dtype, *out_dim as u32, dim as u32) {
                        gpu_proj_indices.push(i);
                    } else {
                        cpu_proj_indices.push(i);
                    }
                }
                let (ssm_out_raw, ssm_out_dtype, _) =
                    Self::recurrent_output_op(weights, &prefix, dim)?;
                let ssm_gpu =
                    gpu_batch_supported(ssm_out_dtype, dim as u32, dims.inner_size as u32);
                if ssm_gpu {
                    Self::prepare_qwen35_batch_projection_weight(
                        metal_ops,
                        ssm_out_raw,
                        ssm_out_dtype,
                        dim as u32,
                        dims.inner_size as u32,
                    )?;
                }
                let keep_rec_z_on_gpu = ssm_gpu && gpu_proj_indices.contains(&1);
                let ssm_key = metal_ops.ensure_quant_cached(ssm_out_raw);

                // Allocate temporary projection output buffers.
                let temp_bufs: Vec<ax_engine_metal::MetalBuffer> = proj_dims
                    .iter()
                    .map(|&d| {
                        ax_engine_metal::MetalBuffer::new(
                            metal_ops.device.device(),
                            n_tokens * d * std::mem::size_of::<f32>(),
                        )
                        .expect("alloc temp proj buf")
                    })
                    .collect();

                let mut norm_for_cpu = Vec::new();
                {
                    let mut bs_guard = metal_ops.batch_scratches();
                    let Some(bs) = bs_guard.as_mut() else {
                        return Ok(false);
                    };

                    // NOW lock weight cache for GPU encoding.
                    let weight_cache = metal_ops.lock_weight_cache();
                    let nw_buf = weight_cache.get(&nw_key).unwrap();

                    // CB1: norm + quantized input projections on GPU.
                    let cb_t = OpTimer::start();
                    metal_ops.device.execute_sync(|encoder| {
                        metal_ops.elementwise.encode_rms_norm_out_batch(
                            encoder,
                            &bs.hidden,
                            nw_buf,
                            &bs.norm_buf,
                            dim as u32,
                            nt,
                            eps,
                        );
                        if gpu_proj_indices
                            .iter()
                            .any(|&i| Self::qwen35_batch_projection_needs_f16_input(proj_dtypes[i]))
                        {
                            metal_ops.elementwise.encode_cast_f32_to_f16(
                                encoder,
                                &bs.norm_buf,
                                &bs.matmul_in_f16,
                                nt * dim as u32,
                            );
                        }
                        for &i in &gpu_proj_indices {
                            let w_buf = weight_cache.get(&proj_keys[i]).unwrap();
                            Self::encode_qwen35_batch_projection(
                                metal_ops,
                                encoder,
                                w_buf,
                                &bs.norm_buf,
                                &bs.matmul_in_f16,
                                &temp_bufs[i],
                                proj_dims[i] as u32,
                                nt,
                                dim as u32,
                                proj_dtypes[i],
                            );
                        }
                        Ok(())
                    })?;
                    if let Some(ref mut ops) = ops {
                        let elapsed = cb_t.elapsed();
                        ops.gpu_execute += elapsed;
                        ops.gpu_execute_layers += elapsed;
                    }

                    // Read GPU projection results and norm_buf before backend recurrent work.
                    let readback_t = OpTimer::start();
                    unsafe {
                        for &i in &gpu_proj_indices {
                            let out_slice =
                                &temp_bufs[i].as_slice::<f32>()[..n_tokens * proj_dims[i]];
                            match i {
                                0 => rec_qkv_batch[..out_slice.len()].copy_from_slice(out_slice),
                                1 => {
                                    if !keep_rec_z_on_gpu {
                                        rec_z_batch[..out_slice.len()].copy_from_slice(out_slice);
                                    }
                                }
                                2 => rec_beta_batch[..out_slice.len()].copy_from_slice(out_slice),
                                3 => rec_alpha_batch[..out_slice.len()].copy_from_slice(out_slice),
                                _ => {}
                            }
                        }
                        if !cpu_proj_indices.is_empty() {
                            norm_for_cpu = bs.norm_buf.as_slice::<f32>()[..n_tokens * dim].to_vec();
                        }
                    }
                    if let Some(ref mut ops) = ops {
                        ops.gpu_readback += readback_t.elapsed();
                    }
                }

                // CPU projections via Backend's optimized matmul (NEON SIMD).
                for &i in &cpu_proj_indices {
                    let (raw, dtype, out_dim) = input_ops[i];
                    let dst = match i {
                        0 => &mut rec_qkv_batch[..n_tokens * out_dim],
                        1 => &mut rec_z_batch[..n_tokens * out_dim],
                        2 => &mut rec_beta_batch[..n_tokens * out_dim],
                        3 => &mut rec_alpha_batch[..n_tokens * out_dim],
                        _ => unreachable!(),
                    };
                    timed_matmul_bucket!(
                        ops,
                        matmul_input_proj,
                        Self::batched_dequant_matmul_token_major(
                            backend,
                            raw,
                            dtype,
                            &norm_for_cpu,
                            dst,
                            n_tokens,
                            out_dim,
                            dim,
                        )
                    );
                }
                // Recurrent sequence via Backend (conv + gated_delta).
                // Batch scratch buffers are intentionally unlocked here because the
                // backend recurrent path may need them for other GPU work.
                rec_out_batch.fill(0.0);
                let runtime = timed!(
                    ops,
                    recurrent,
                    Self::run_recurrent_sequence(
                        backend,
                        weights,
                        &prefix,
                        qwen_kv,
                        layer,
                        &recurrent_slot_indices,
                        dims,
                        eps,
                        &rec_qkv_batch,
                        &mut rec_beta_batch,
                        &mut rec_alpha_batch,
                        &mut rec_out_batch,
                        n_tokens,
                    )
                )?;

                let ssm_norm_key = if ssm_gpu {
                    Some(metal_ops.ensure_f32_cached(runtime.ssm_norm))
                } else {
                    None
                };
                if !ssm_gpu {
                    Self::finalize_recurrent_output_batch(
                        &mut rec_out_batch,
                        &rec_z_batch,
                        n_tokens,
                        dims,
                        runtime.ssm_norm,
                        eps,
                    );
                }

                // SSM output projection: fall back to CPU if this batch shape cannot stay on GPU.
                let ssm_result = if !ssm_gpu {
                    let mut result = vec![0.0f32; n_tokens * dim];
                    timed_matmul_bucket!(
                        ops,
                        matmul_output_proj,
                        Self::batched_dequant_matmul_token_major(
                            backend,
                            ssm_out_raw,
                            ssm_out_dtype,
                            &rec_out_batch,
                            &mut result,
                            n_tokens,
                            dim,
                            dims.inner_size,
                        )
                    );
                    Some(result)
                } else {
                    None
                };

                {
                    let mut bs_guard = metal_ops.batch_scratches();
                    let Some(bs) = bs_guard.as_mut() else {
                        return Ok(false);
                    };

                    // CB2: residual + FFN (hidden on GPU).
                    if let Some(ref ssm_result) = ssm_result {
                        // CPU SSM result → attn_out on GPU.
                        unsafe {
                            bs.attn_out.as_mut_slice::<f32>()[..n_tokens * dim]
                                .copy_from_slice(ssm_result);
                        }
                    } else {
                        // GPU SSM: upload raw recurrent output and reuse GPU gate when available.
                        unsafe {
                            bs.gate_buf.as_mut_slice::<f32>()[..n_tokens * dims.inner_size]
                                .copy_from_slice(&rec_out_batch[..n_tokens * dims.inner_size]);
                            if !keep_rec_z_on_gpu {
                                bs.up_buf.as_mut_slice::<f32>()[..n_tokens * dims.inner_size]
                                    .copy_from_slice(&rec_z_batch[..n_tokens * dims.inner_size]);
                            }
                        }
                    }
                    drop(ssm_result);
                    let weight_cache = metal_ops.lock_weight_cache();
                    let ffn_nw_buf = weight_cache.get(&ffn_nw_key).unwrap();
                    let wg_buf = weight_cache.get(&wg_key).unwrap();
                    let wu_buf = weight_cache.get(&wu_key).unwrap();
                    let wd_buf = weight_cache.get(&wd_key).unwrap();
                    let cb_t = OpTimer::start();
                    metal_ops.device.execute_sync(|encoder| {
                        if ssm_gpu {
                            let rec_z_gpu_buf = if keep_rec_z_on_gpu {
                                Some(&temp_bufs[1])
                            } else {
                                None
                            };
                            let rec_z_buf = rec_z_gpu_buf.unwrap_or(&bs.up_buf);
                            let ssm_norm_buf = weight_cache.get(&ssm_norm_key.unwrap()).unwrap();
                            metal_ops.elementwise.encode_per_head_rms_norm_batch(
                                encoder,
                                &bs.gate_buf,
                                ssm_norm_buf,
                                nt,
                                dims.time_step_rank as u32,
                                dims.state_size as u32,
                                eps,
                            );
                            metal_ops.elementwise.encode_silu_elementwise_mul_batch(
                                encoder, rec_z_buf, &bs.gate_buf, dims.inner_size as u32, nt,
                            );
                            // GPU SSM: proj_buf (inner_size) → attn_out (dim)
                            if Self::qwen35_batch_projection_needs_f16_input(ssm_out_dtype) {
                                metal_ops.elementwise.encode_cast_f32_to_f16(
                                    encoder,
                                    rec_z_buf,
                                    &bs.matmul_in_f16,
                                    nt * dims.inner_size as u32,
                                );
                            }
                            let ssm_buf = weight_cache.get(&ssm_key).unwrap();
                            Self::encode_qwen35_batch_projection(
                                metal_ops,
                                encoder,
                                ssm_buf,
                                rec_z_buf,
                                &bs.matmul_in_f16,
                                &bs.attn_out,
                                dim as u32,
                                nt,
                                dims.inner_size as u32,
                                ssm_out_dtype,
                            );
                        }
                        // Residual: hidden += attn_out (SSM output) + FFN norm
                        metal_ops
                            .elementwise
                            .encode_residual_add_rms_norm_out_batch(
                                encoder,
                                &bs.hidden,
                                &bs.attn_out,
                                ffn_nw_buf,
                                &bs.norm_buf,
                                dim as u32,
                                nt,
                                eps,
                            );
                        if Self::qwen35_batch_projection_needs_f16_input(wg_dtype)
                            || Self::qwen35_batch_projection_needs_f16_input(wu_dtype)
                        {
                            metal_ops.elementwise.encode_cast_f32_to_f16(
                                encoder,
                                &bs.norm_buf,
                                &bs.matmul_in_f16,
                                nt * dim as u32,
                            );
                        }
                        // Gate + Up
                        Self::encode_qwen35_batch_projection(
                            metal_ops,
                            encoder,
                            wg_buf,
                            &bs.norm_buf,
                            &bs.matmul_in_f16,
                            &bs.gate_buf,
                            inter_dim as u32,
                            nt,
                            dim as u32,
                            wg_dtype,
                        );
                        Self::encode_qwen35_batch_projection(
                            metal_ops,
                            encoder,
                            wu_buf,
                            &bs.norm_buf,
                            &bs.matmul_in_f16,
                            &bs.up_buf,
                            inter_dim as u32,
                            nt,
                            dim as u32,
                            wu_dtype,
                        );
                        // SiLU
                        metal_ops.elementwise.encode_silu_elementwise_mul_batch(
                            encoder,
                            &bs.gate_buf,
                            &bs.up_buf,
                            inter_dim as u32,
                            nt,
                        );
                        if Self::qwen35_batch_projection_needs_f16_input(wd_dtype) {
                            metal_ops.elementwise.encode_cast_f32_to_f16(
                                encoder,
                                &bs.gate_buf,
                                &bs.matmul_in_f16,
                                nt * inter_dim as u32,
                            );
                        }
                        // Down
                        Self::encode_qwen35_batch_projection(
                            metal_ops,
                            encoder,
                            wd_buf,
                            &bs.gate_buf,
                            &bs.matmul_in_f16,
                            &bs.proj_buf,
                            dim as u32,
                            nt,
                            inter_dim as u32,
                            wd_dtype,
                        );
                        // Final residual: hidden += proj_buf
                        metal_ops.elementwise.encode_elementwise_add_batch(
                            encoder,
                            &bs.hidden,
                            &bs.proj_buf,
                            dim as u32,
                            nt,
                        );
                        Ok(())
                    })?;
                    if let Some(ref mut ops) = ops {
                        let elapsed = cb_t.elapsed();
                        ops.gpu_execute += elapsed;
                        ops.gpu_execute_layers += elapsed;
                    }
                } // end re-acquired bs_guard scope
            }
        }

        qwen_kv.finalize_batch(n_tokens);

        match (logits, logits_all) {
            (Some(logits), None) => {
                let mut bs_guard = metal_ops.batch_scratches();
                let Some(bs) = bs_guard.as_mut() else {
                    return Ok(false);
                };
                let mut h = unsafe {
                    bs.hidden.as_slice::<f32>()[((n_tokens - 1) * dim)..n_tokens * dim].to_vec()
                };
                drop(bs_guard);
                Self::write_single_logits(backend, &mut h, dim, vocab_size, eps, weights, logits)?;
            }
            (None, Some(logits_all)) => {
                let mut bs_guard = metal_ops.batch_scratches();
                let Some(bs) = bs_guard.as_mut() else {
                    return Ok(false);
                };
                let h = unsafe { bs.hidden.as_slice::<f32>()[..n_tokens * dim].to_vec() };
                drop(bs_guard);
                Self::write_all_batch_logits(
                    backend, &h, n_tokens, dim, vocab_size, eps, weights, logits_all,
                )?;
            }
            _ => unreachable!("validated by caller"),
        }
        if let Some(ref mut ops) = ops {
            ops.gpu += total_t.elapsed();
        }
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_impl(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        mut logits: Option<&mut [f32]>,
        mut logits_all: Option<&mut Vec<f32>>,
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            !token_ids.is_empty(),
            "qwen35 forward_batch requires at least one token"
        );
        anyhow::ensure!(
            logits.is_some() ^ logits_all.is_some(),
            "qwen35 batch forward requires either last logits or all logits output"
        );

        let force_serial = env_flag_enabled("AX_SERIAL_PREFILL");
        if force_serial || !ctx.backend.use_gpu_decode() || token_ids.len() <= 1 {
            return self.forward_batch_serial_fallback(
                ctx,
                token_ids,
                kv.seq_len(),
                kv,
                weights,
                logits,
                logits_all,
            );
        }

        if let Some(logits) = logits.as_mut() {
            if Self::try_forward_batch_gpu_unified(
                ctx,
                token_ids,
                kv,
                weights,
                Some(&mut **logits),
                None,
                ops.as_deref_mut(),
            )? {
                return Ok(());
            }
        } else if let Some(logits_all) = logits_all.as_mut()
            && Self::try_forward_batch_gpu_unified(
                ctx,
                token_ids,
                kv,
                weights,
                None,
                Some(&mut **logits_all),
                ops.as_deref_mut(),
            )?
        {
            return Ok(());
        }

        let Some(qwen_kv) = kv.as_qwen35_mut() else {
            anyhow::bail!("Qwen35Forward requires ModelKv::Qwen35");
        };
        let recurrent_slot = qwen_kv.active_slot();

        let cfg = ctx.config;
        let backend = ctx.backend;
        let dims = Self::recurrent_dims(cfg)?;
        let n_tokens = token_ids.len();
        let n_layers = cfg.n_layers as usize;
        let dim = cfg.embedding_dim as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let vocab_size = cfg.vocab_size as usize;
        let conv_dim = dims.conv_dim();
        let full_attn_params = AttentionParams::new(n_heads, n_kv_heads, head_dim);

        let mut hidden = vec![0.0f32; n_tokens * dim];
        for (i, &tid) in token_ids.iter().enumerate() {
            weights.dequantize_row(
                "token_embd.weight",
                tid as usize,
                &mut hidden[i * dim..(i + 1) * dim],
            )?;
        }

        let mut norm_buf = vec![0.0f32; n_tokens * dim];
        let mut proj_buf = vec![0.0f32; n_tokens * dim];
        let mut gate_buf = vec![0.0f32; n_tokens * inter_dim];
        let mut up_buf = vec![0.0f32; n_tokens * inter_dim];
        let mut down_buf = vec![0.0f32; n_tokens * dim];
        let mut q_gate_batch = vec![0.0f32; n_tokens * q_dim * 2];
        let mut q_batch = vec![0.0f32; n_tokens * q_dim];
        let mut k_batch = vec![0.0f32; n_tokens * kv_dim];
        let mut v_batch = vec![0.0f32; n_tokens * kv_dim];
        let mut fused_input_batch = vec![0.0f32; n_tokens * (q_dim * 2 + 2 * kv_dim)];
        let mut attn_out_batch = vec![0.0f32; n_tokens * q_dim];
        let mut rec_qkv_batch = vec![0.0f32; n_tokens * conv_dim];
        let mut rec_z_batch = vec![0.0f32; n_tokens * dims.inner_size];
        let mut rec_beta_batch = vec![0.0f32; n_tokens * dims.time_step_rank];
        let mut rec_alpha_batch = vec![0.0f32; n_tokens * dims.time_step_rank];
        let mut rec_out_batch = vec![0.0f32; n_tokens * dims.inner_size];
        let recurrent_slot_indices = [recurrent_slot];
        let batch_position = qwen_kv.seq_len();

        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");

            match Self::layer_type(cfg, layer) {
                Qwen35LayerType::FullAttention => {
                    // Try GPU-encoded norm + QKV input projections.
                    let input_plan =
                        Self::full_attention_input_plan(weights, &prefix, q_dim, kv_dim)?;
                    let gpu_projected = match &input_plan {
                        Qwen35FullAttentionInputPlan::Split(ops) => {
                            let mut outs: [&mut [f32]; 3] =
                                [&mut q_gate_batch, &mut k_batch, &mut v_batch];
                            Self::try_norm_and_project_batch_gpu(
                                cfg,
                                backend,
                                weights,
                                &prefix,
                                &hidden,
                                &mut norm_buf,
                                ops,
                                &mut outs,
                                n_tokens,
                                dim,
                                cfg.rms_norm_eps,
                            )?
                        }
                        _ => false,
                    };
                    if !gpu_projected {
                        Self::apply_attention_norm_batch(
                            weights,
                            &prefix,
                            &hidden,
                            &mut norm_buf,
                            n_tokens,
                            dim,
                            cfg.rms_norm_eps,
                        )?;
                    }
                    Self::assert_finite_if_enabled(
                        "attn_norm_output_batch",
                        &norm_buf,
                        layer,
                        batch_position,
                    )?;
                    Self::run_full_attention_batch_layer(
                        cfg,
                        backend,
                        weights,
                        &prefix,
                        qwen_kv,
                        layer,
                        batch_position,
                        &norm_buf,
                        &mut q_gate_batch,
                        &mut q_batch,
                        &mut k_batch,
                        &mut v_batch,
                        &mut fused_input_batch,
                        &mut attn_out_batch,
                        &mut proj_buf,
                        n_tokens,
                        dim,
                        q_dim,
                        kv_dim,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        &full_attn_params,
                        gpu_projected,
                    )?;
                }
                Qwen35LayerType::RecurrentGdn => {
                    // Try GPU-encoded norm + 4 recurrent input projections.
                    let input_ops = Self::recurrent_input_ops(weights, &prefix, dims)?;
                    let mut rec_outs: [&mut [f32]; 4] = [
                        &mut rec_qkv_batch,
                        &mut rec_z_batch,
                        &mut rec_beta_batch,
                        &mut rec_alpha_batch,
                    ];
                    let gpu_projected = Self::try_norm_and_project_batch_gpu(
                        cfg,
                        backend,
                        weights,
                        &prefix,
                        &hidden,
                        &mut norm_buf,
                        &input_ops,
                        &mut rec_outs,
                        n_tokens,
                        dim,
                        cfg.rms_norm_eps,
                    )?;
                    if !gpu_projected {
                        Self::apply_attention_norm_batch(
                            weights,
                            &prefix,
                            &hidden,
                            &mut norm_buf,
                            n_tokens,
                            dim,
                            cfg.rms_norm_eps,
                        )?;
                    }
                    Self::assert_finite_if_enabled(
                        "attn_norm_output_batch",
                        &norm_buf,
                        layer,
                        batch_position,
                    )?;
                    Self::run_recurrent_batch_layer(
                        cfg,
                        backend,
                        weights,
                        &prefix,
                        qwen_kv,
                        recurrent_slot,
                        layer,
                        batch_position,
                        dims,
                        &recurrent_slot_indices,
                        &norm_buf,
                        &mut rec_qkv_batch,
                        &mut rec_z_batch,
                        &mut rec_beta_batch,
                        &mut rec_alpha_batch,
                        &mut rec_out_batch,
                        &mut proj_buf,
                        n_tokens,
                        dim,
                        gpu_projected,
                    )?;
                }
            }

            Self::apply_layer_tail_batch(
                cfg,
                backend,
                weights,
                &prefix,
                &mut hidden,
                &proj_buf,
                &mut norm_buf,
                &mut gate_buf,
                &mut up_buf,
                &mut down_buf,
                n_tokens,
                dim,
                inter_dim,
                cfg.rms_norm_eps,
                layer,
                batch_position,
            )?;
        }

        qwen_kv.finalize_batch(n_tokens);

        match (logits, logits_all) {
            (Some(logits), None) => Self::write_last_batch_logits(
                backend,
                &mut hidden,
                n_tokens,
                dim,
                vocab_size,
                cfg.rms_norm_eps,
                weights,
                logits,
            ),
            (None, Some(logits_all)) => Self::write_all_batch_logits(
                backend,
                &hidden,
                n_tokens,
                dim,
                vocab_size,
                cfg.rms_norm_eps,
                weights,
                logits_all,
            ),
            _ => unreachable!("validated above"),
        }
    }

    /// GPU-accelerated single-token decode for Qwen3.5.
    ///
    /// Encodes full-attention layers using the shared GPU layer encoder (1 CB
    /// per layer) and recurrent layers with minimized CPU round-trips. This is
    /// the default single-token decode path when Metal recurrent support and
    /// GPU attention KV are available.
    ///
    /// Returns `Ok(true)` if GPU path was used, `Ok(false)` to fall back.
    #[allow(clippy::too_many_arguments)]
    fn try_forward_single_gpu(
        ctx: &ForwardContext,
        token_id: u32,
        position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<bool> {
        use crate::model::shared::encode_dequant_matvec_with_config;

        let Some(metal_ops) = ctx.backend.metal_ops() else {
            return Ok(false);
        };
        let cfg = ctx.config;
        let qwen_kv = kv
            .as_qwen35_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen35Forward requires ModelKv::Qwen35"))?;
        if qwen_kv.gpu_attention().is_none() {
            return Ok(false);
        }

        let dims = Self::recurrent_dims(cfg)?;
        let n_layers = cfg.n_layers as usize;
        let dim = cfg.embedding_dim as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let vocab_size = cfg.vocab_size as usize;
        let recurrent_slot = qwen_kv.active_slot();
        let eps = cfg.rms_norm_eps;
        let conv_cache_len = qwen_kv.conv_cache_len();

        // Cache weights on GPU.
        if !metal_ops.has_cached_model_keys() {
            Self::build_cached_model_keys_qwen35(metal_ops, weights, cfg)?;
        }
        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();
        let gpu_layer_keys = Self::cached_gpu_layer_keys(cached.lm_head)
            .ok_or_else(|| anyhow::anyhow!("missing cached qwen35 gpu layer keys"))?;
        metal_ops.init_scratches(cfg);
        let mut scratch_guard = metal_ops.scratches();
        let s = scratch_guard.as_mut().unwrap();
        let weight_cache = metal_ops.lock_weight_cache();

        let seq_len = qwen_kv.seq_len();
        let full_seq_len = seq_len + 1;
        if !qwen_kv.ensure_gpu_attention_capacity_for(full_seq_len) {
            return Ok(false);
        }

        // Build execution plan.
        let exec_plan = crate::model::execution_plan::DecodeExecutionPlan::qwen35_single_cb(
            metal_ops,
            qwen_kv.gpu_attention().unwrap(),
            cfg.embedding_dim,
            cfg.head_dim,
            full_seq_len,
        );
        let decode_barrier = |encoder: &ax_engine_metal::MetalEncoder| {
            if exec_plan.barriers == crate::model::execution_plan::DecodeBarrierPlan::Explicit {
                ax_engine_metal::barrier_buffers(encoder);
            }
        };

        let setup_t = OpTimer::start();

        // Embed token.
        {
            let h = unsafe {
                std::slice::from_raw_parts_mut(s.hidden.contents().as_ptr() as *mut f32, dim)
            };
            weights.dequantize_row("token_embd.weight", token_id as usize, h)?;
        }

        let rope_position = Self::rope_position(cfg, position);
        let kv_offset = (seq_len * kv_dim) as u32;

        for layer in 0..n_layers {
            if cfg.qwen35_is_recurrent_layer(layer) {
                metal_ops.sync_qwen35_slot_buffers_from_kv(qwen_kv, layer, recurrent_slot);
            }
        }

        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_encode += setup_t.elapsed();
        }

        let exec_t = OpTimer::start();
        metal_ops.device.execute_sync(|encoder| {
            // Layer 0 attention norm.
            let layer0_norm_t = OpTimer::start();
            let norm_w = weight_cache.get(&cached.layers[0].attn_norm).unwrap();
            metal_ops.elementwise.encode_rms_norm_out(
                encoder,
                &s.hidden,
                norm_w,
                &s.norm_buf,
                dim as u32,
                eps,
            );
            decode_barrier(encoder);
            if let Some(ref mut ops_ref) = ops {
                ops_ref.gpu_encode_layer_norm += layer0_norm_t.elapsed();
            }

            for layer in 0..n_layers {
                let lw = &cached.layers[layer];
                let next_norm = if layer + 1 < n_layers {
                    Some(
                        weight_cache
                            .get(&cached.layers[layer + 1].attn_norm)
                            .unwrap(),
                    )
                } else {
                    None
                };

                if !cfg.qwen35_is_recurrent_layer(layer) {
                    let q_norm_key = lw.attn_q_norm;
                    let k_norm_key = lw.attn_k_norm;
                    let gpu_attn = qwen_kv.gpu_attention().unwrap();

                    let qkv_t = OpTimer::start();
                    let wq = weight_cache.get(&lw.wq).unwrap();
                    let wk = weight_cache.get(&lw.wk).unwrap();
                    let wv = weight_cache.get(&lw.wv).unwrap();
                    encode_dequant_matvec_with_config(
                        metal_ops,
                        encoder,
                        wq,
                        &s.norm_buf,
                        &s.gate_buf,
                        (q_dim * 2) as u32,
                        dim as u32,
                        lw.wq_dtype,
                        exec_plan.dequant_dispatch,
                    );
                    encode_dequant_matvec_with_config(
                        metal_ops,
                        encoder,
                        wk,
                        &s.norm_buf,
                        &s.k_buf,
                        kv_dim as u32,
                        dim as u32,
                        lw.wk_dtype,
                        exec_plan.dequant_dispatch,
                    );
                    encode_dequant_matvec_with_config(
                        metal_ops,
                        encoder,
                        wv,
                        &s.norm_buf,
                        &s.v_buf,
                        kv_dim as u32,
                        dim as u32,
                        lw.wv_dtype,
                        exec_plan.dequant_dispatch,
                    );
                    metal_ops.elementwise.encode_split_qgate_batch(
                        encoder,
                        &s.gate_buf,
                        &s.q_buf,
                        &s.up_buf,
                        1,
                        q_dim as u32,
                    );
                    decode_barrier(encoder);
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_qkv += qkv_t.elapsed();
                    }

                    if let (Some(q_key), Some(k_key)) = (q_norm_key, k_norm_key) {
                        let qk_norm_t = OpTimer::start();
                        let q_nw = weight_cache.get(&q_key).unwrap();
                        let k_nw = weight_cache.get(&k_key).unwrap();
                        metal_ops.elementwise.encode_per_head_rms_norm_batch(
                            encoder,
                            &s.q_buf,
                            q_nw,
                            1,
                            n_heads as u32,
                            head_dim as u32,
                            eps,
                        );
                        metal_ops.elementwise.encode_per_head_rms_norm_batch(
                            encoder,
                            &s.k_buf,
                            k_nw,
                            1,
                            n_kv_heads as u32,
                            head_dim as u32,
                            eps,
                        );
                        decode_barrier(encoder);
                        if let Some(ref mut ops_ref) = ops {
                            ops_ref.gpu_encode_layer_norm += qk_norm_t.elapsed();
                        }
                    }

                    let rope_t = OpTimer::start();
                    metal_ops.elementwise.encode_rope_batch(
                        encoder,
                        &s.q_buf,
                        &s.k_buf,
                        1,
                        n_heads as u32,
                        n_kv_heads as u32,
                        head_dim as u32,
                        rope_position,
                        0.0,
                        cfg.rope_freq_base,
                    );
                    decode_barrier(encoder);
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_rope += rope_t.elapsed();
                    }

                    let kv_append_t = OpTimer::start();
                    metal_ops.elementwise.encode_kv_append(
                        encoder,
                        &s.k_buf,
                        gpu_attn.k_buffer(layer),
                        gpu_attn.is_f16(),
                        kv_offset,
                        kv_dim as u32,
                    );
                    metal_ops.elementwise.encode_kv_append(
                        encoder,
                        &s.v_buf,
                        gpu_attn.v_buffer(layer),
                        gpu_attn.is_f16(),
                        kv_offset,
                        kv_dim as u32,
                    );
                    decode_barrier(encoder);
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_kv_append += kv_append_t.elapsed();
                    }

                    let attn_t = OpTimer::start();
                    metal_ops
                        .attention
                        .encode_attention_decode_with_scratch_and_config(
                            encoder,
                            &s.q_buf,
                            gpu_attn.k_buffer(layer),
                            gpu_attn.v_buffer(layer),
                            &s.attn_out,
                            &s.splitk_partial_out,
                            &s.splitk_partial_lse,
                            gpu_attn.is_f16(),
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            0,
                            full_seq_len as u32,
                            exec_plan.attention_dispatch,
                        );
                    decode_barrier(encoder);

                    metal_ops.elementwise.encode_sigmoid_elementwise_mul(
                        encoder,
                        &s.up_buf,
                        &s.attn_out,
                        q_dim as u32,
                    );
                    decode_barrier(encoder);
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_attention += attn_t.elapsed();
                    }

                    let out_proj_t = OpTimer::start();
                    encode_dequant_matvec_with_config(
                        metal_ops,
                        encoder,
                        weight_cache.get(&lw.wo).unwrap(),
                        &s.attn_out,
                        &s.proj_buf,
                        dim as u32,
                        q_dim as u32,
                        lw.wo_dtype,
                        exec_plan.dequant_dispatch,
                    );
                    decode_barrier(encoder);
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_out_proj += out_proj_t.elapsed();
                    }

                    let residual_norm_t = OpTimer::start();
                    let ffn_nw = weight_cache.get(&lw.ffn_norm).unwrap();
                    metal_ops
                        .elementwise
                        .encode_residual_add_rms_norm_out_batch(
                            encoder,
                            &s.hidden,
                            &s.proj_buf,
                            ffn_nw,
                            &s.norm_buf,
                            dim as u32,
                            1,
                            eps,
                        );
                    decode_barrier(encoder);
                    if let Some(ref mut ops_ref) = ops {
                        let elapsed = residual_norm_t.elapsed();
                        ops_ref.gpu_encode_layer_residual += elapsed / 2;
                        ops_ref.gpu_encode_layer_norm += elapsed / 2;
                    }

                    let gate_up_t = OpTimer::start();
                    let wg = weight_cache.get(&lw.wg).unwrap();
                    let wu = weight_cache.get(&lw.wu).unwrap();
                    if !crate::model::shared::encode_dequant_matvec_pair_with_config(
                        metal_ops,
                        encoder,
                        wg,
                        wu,
                        &s.norm_buf,
                        &s.gate_buf,
                        &s.up_buf,
                        inter_dim as u32,
                        dim as u32,
                        lw.wg_dtype,
                        lw.wu_dtype,
                        exec_plan.dequant_dispatch,
                        exec_plan.use_pair_matvec,
                    ) {
                        encode_dequant_matvec_with_config(
                            metal_ops,
                            encoder,
                            wg,
                            &s.norm_buf,
                            &s.gate_buf,
                            inter_dim as u32,
                            dim as u32,
                            lw.wg_dtype,
                            exec_plan.dequant_dispatch,
                        );
                        encode_dequant_matvec_with_config(
                            metal_ops,
                            encoder,
                            wu,
                            &s.norm_buf,
                            &s.up_buf,
                            inter_dim as u32,
                            dim as u32,
                            lw.wu_dtype,
                            exec_plan.dequant_dispatch,
                        );
                    }
                    decode_barrier(encoder);
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_ffn += gate_up_t.elapsed();
                    }

                    let ffn_tail_t = OpTimer::start();
                    crate::model::shared::encode_gpu_ffn_decode_tail(
                        metal_ops,
                        encoder,
                        s,
                        &s.hidden,
                        weight_cache.get(&lw.wd).unwrap(),
                        lw.wd_dtype,
                        dim as u32,
                        inter_dim as u32,
                        eps,
                        exec_plan.dequant_dispatch,
                        exec_plan.use_fused_silu_down,
                        crate::model::layer_ops::FfnActivation::SiLU,
                        None,
                        next_norm,
                        &decode_barrier,
                    );
                    if let Some(ref mut ops_ref) = ops {
                        ops_ref.gpu_encode_layer_ffn += ffn_tail_t.elapsed();
                    }
                } else {
                    let recurrent_keys = match &gpu_layer_keys[layer] {
                        Qwen35GpuLayerKeys::Recurrent(keys) => keys,
                        Qwen35GpuLayerKeys::FullAttention => {
                            anyhow::bail!("expected recurrent qwen35 GPU keys for layer {layer}")
                        }
                    };
                    debug_assert!(kv_dim >= dims.time_step_rank);
                    let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
                    let recurrent_state_stride = qwen_kv.recurrent_state_len();
                    metal_ops.with_qwen35_recurrent_slot_buffer(
                        layer,
                        recurrent_slot,
                        conv_state_stride,
                        recurrent_state_stride,
                        |slot_buffers| {
                            let recurrent_norm_t = OpTimer::start();
                            let norm_w = weight_cache.get(&lw.attn_norm).unwrap();
                            metal_ops.elementwise.encode_rms_norm_out(
                                encoder,
                                &s.hidden,
                                norm_w,
                                &s.norm_buf,
                                dim as u32,
                                eps,
                            );
                            decode_barrier(encoder);
                            if let Some(ref mut ops_ref) = ops {
                                ops_ref.gpu_encode_layer_norm += recurrent_norm_t.elapsed();
                            }

                            let recurrent_input_t = OpTimer::start();
                            encode_dequant_matvec_with_config(
                                metal_ops,
                                encoder,
                                weight_cache.get(&recurrent_keys.wqkv).unwrap(),
                                &s.norm_buf,
                                &s.gate_buf,
                                dims.conv_dim() as u32,
                                dim as u32,
                                recurrent_keys.wqkv_dtype,
                                exec_plan.dequant_dispatch,
                            );
                            encode_dequant_matvec_with_config(
                                metal_ops,
                                encoder,
                                weight_cache.get(&recurrent_keys.wgate).unwrap(),
                                &s.norm_buf,
                                &s.attn_out,
                                dims.inner_size as u32,
                                dim as u32,
                                recurrent_keys.wgate_dtype,
                                exec_plan.dequant_dispatch,
                            );
                            let wbeta = weight_cache.get(&recurrent_keys.wbeta).unwrap();
                            let walpha = weight_cache.get(&recurrent_keys.walpha).unwrap();
                            if !Self::encode_recurrent_pair_matvec_with_config(
                                metal_ops,
                                encoder,
                                wbeta,
                                walpha,
                                &s.norm_buf,
                                &s.v_buf,
                                &s.proj_buf,
                                dims.time_step_rank as u32,
                                dim as u32,
                                recurrent_keys.wbeta_dtype,
                                recurrent_keys.walpha_dtype,
                            ) {
                                encode_dequant_matvec_with_config(
                                    metal_ops,
                                    encoder,
                                    wbeta,
                                    &s.norm_buf,
                                    &s.v_buf,
                                    dims.time_step_rank as u32,
                                    dim as u32,
                                    recurrent_keys.wbeta_dtype,
                                    exec_plan.dequant_dispatch,
                                );
                                encode_dequant_matvec_with_config(
                                    metal_ops,
                                    encoder,
                                    walpha,
                                    &s.norm_buf,
                                    &s.proj_buf,
                                    dims.time_step_rank as u32,
                                    dim as u32,
                                    recurrent_keys.walpha_dtype,
                                    exec_plan.dequant_dispatch,
                                );
                            }
                            decode_barrier(encoder);
                            if let Some(ref mut ops_ref) = ops {
                                ops_ref.gpu_encode_layer_qkv += recurrent_input_t.elapsed();
                            }

                            let recurrent_core_t = OpTimer::start();
                            metal_ops.gdn.encode_causal_conv_sequence(
                                encoder,
                                &s.gate_buf,
                                weight_cache.get(&recurrent_keys.conv_kernel).unwrap(),
                                &slot_buffers.conv_state,
                                &s.up_buf,
                                1,
                                conv_cache_len as u32,
                                dims.conv_dim() as u32,
                            );
                            decode_barrier(encoder);

                            if metal_ops.gdn.encode_single_token_gated_delta_fused(
                                encoder,
                                &s.up_buf,
                                &s.proj_buf,
                                &s.v_buf,
                                &slot_buffers.recurrent_state,
                                &s.q_buf,
                                dims.group_count as u32,
                                dims.time_step_rank as u32,
                                dims.state_size as u32,
                                eps,
                            ) {
                                decode_barrier(encoder);
                            } else {
                                metal_ops.gdn.encode_prepare_single_token_qkv(
                                    encoder,
                                    &s.up_buf,
                                    &s.q_buf,
                                    &s.norm_buf,
                                    &s.down_buf,
                                    dims.group_count as u32,
                                    dims.time_step_rank as u32,
                                    dims.state_size as u32,
                                    eps,
                                );
                                decode_barrier(encoder);

                                metal_ops.gdn.encode_gated_delta_sequence(
                                    encoder,
                                    &s.q_buf,
                                    &s.norm_buf,
                                    &s.down_buf,
                                    &s.proj_buf,
                                    &s.v_buf,
                                    &slot_buffers.recurrent_state,
                                    &s.q_buf,
                                    1,
                                    dims.time_step_rank as u32,
                                    dims.state_size as u32,
                                );
                                decode_barrier(encoder);
                            }

                            let ssm_norm = weight_cache.get(&recurrent_keys.ssm_norm).unwrap();
                            metal_ops.elementwise.encode_per_head_rms_norm_batch(
                                encoder,
                                &s.q_buf,
                                ssm_norm,
                                1,
                                dims.time_step_rank as u32,
                                dims.state_size as u32,
                                eps,
                            );
                            decode_barrier(encoder);

                            metal_ops.elementwise.encode_silu_elementwise_mul_batch(
                                encoder,
                                &s.attn_out,
                                &s.q_buf,
                                dims.inner_size as u32,
                                1,
                            );
                            decode_barrier(encoder);
                            if let Some(ref mut ops_ref) = ops {
                                ops_ref.gpu_encode_layer_attention += recurrent_core_t.elapsed();
                            }

                            let recurrent_out_proj_t = OpTimer::start();
                            encode_dequant_matvec_with_config(
                                metal_ops,
                                encoder,
                                weight_cache.get(&recurrent_keys.wssm_out).unwrap(),
                                &s.attn_out,
                                &s.proj_buf,
                                dim as u32,
                                dims.inner_size as u32,
                                recurrent_keys.wssm_out_dtype,
                                exec_plan.dequant_dispatch,
                            );
                            decode_barrier(encoder);
                            if let Some(ref mut ops_ref) = ops {
                                ops_ref.gpu_encode_layer_out_proj += recurrent_out_proj_t.elapsed();
                            }

                            let residual_norm_t = OpTimer::start();
                            let ffn_nw = weight_cache.get(&lw.ffn_norm).unwrap();
                            metal_ops
                                .elementwise
                                .encode_residual_add_rms_norm_out_batch(
                                    encoder,
                                    &s.hidden,
                                    &s.proj_buf,
                                    ffn_nw,
                                    &s.norm_buf,
                                    dim as u32,
                                    1,
                                    eps,
                                );
                            decode_barrier(encoder);
                            if let Some(ref mut ops_ref) = ops {
                                let elapsed = residual_norm_t.elapsed();
                                ops_ref.gpu_encode_layer_residual += elapsed / 2;
                                ops_ref.gpu_encode_layer_norm += elapsed / 2;
                            }

                            let gate_up_t = OpTimer::start();
                            let wg = weight_cache.get(&lw.wg).unwrap();
                            let wu = weight_cache.get(&lw.wu).unwrap();
                            if !crate::model::shared::encode_dequant_matvec_pair_with_config(
                                metal_ops,
                                encoder,
                                wg,
                                wu,
                                &s.norm_buf,
                                &s.gate_buf,
                                &s.up_buf,
                                inter_dim as u32,
                                dim as u32,
                                lw.wg_dtype,
                                lw.wu_dtype,
                                exec_plan.dequant_dispatch,
                                exec_plan.use_pair_matvec,
                            ) {
                                encode_dequant_matvec_with_config(
                                    metal_ops,
                                    encoder,
                                    wg,
                                    &s.norm_buf,
                                    &s.gate_buf,
                                    inter_dim as u32,
                                    dim as u32,
                                    lw.wg_dtype,
                                    exec_plan.dequant_dispatch,
                                );
                                encode_dequant_matvec_with_config(
                                    metal_ops,
                                    encoder,
                                    wu,
                                    &s.norm_buf,
                                    &s.up_buf,
                                    inter_dim as u32,
                                    dim as u32,
                                    lw.wu_dtype,
                                    exec_plan.dequant_dispatch,
                                );
                            }
                            decode_barrier(encoder);
                            if let Some(ref mut ops_ref) = ops {
                                ops_ref.gpu_encode_layer_ffn += gate_up_t.elapsed();
                            }

                            let ffn_tail_t = OpTimer::start();
                            crate::model::shared::encode_gpu_ffn_decode_tail(
                                metal_ops,
                                encoder,
                                s,
                                &s.hidden,
                                weight_cache.get(&lw.wd).unwrap(),
                                lw.wd_dtype,
                                dim as u32,
                                inter_dim as u32,
                                eps,
                                exec_plan.dequant_dispatch,
                                exec_plan.use_fused_silu_down,
                                crate::model::layer_ops::FfnActivation::SiLU,
                                None,
                                next_norm,
                                &decode_barrier,
                            );
                            if let Some(ref mut ops_ref) = ops {
                                ops_ref.gpu_encode_layer_ffn += ffn_tail_t.elapsed();
                            }
                        },
                    );
                }
            }

            crate::model::shared::encode_gpu_output_head(
                encoder,
                metal_ops,
                s,
                &s.hidden,
                &exec_plan,
                cached,
                &weight_cache,
                dim as u32,
                vocab_size as u32,
                eps,
            );
            Ok(())
        })?;
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_execute += exec_t.elapsed();
        }

        drop(weight_cache);
        drop(cached_guard);

        let rb_t = OpTimer::start();
        let mut mirror_k = vec![0.0f32; kv_dim];
        let mut mirror_v = vec![0.0f32; kv_dim];
        for layer in 0..n_layers {
            if cfg.qwen35_is_recurrent_layer(layer) {
                let conv_generation = qwen_kv.note_backend_conv_state_update(recurrent_slot, layer);
                let recurrent_generation =
                    qwen_kv.note_backend_recurrent_state_update(recurrent_slot, layer);
                let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
                let recurrent_state_stride = qwen_kv.recurrent_state_len();
                metal_ops.with_qwen35_recurrent_slot_buffer(
                    layer,
                    recurrent_slot,
                    conv_state_stride,
                    recurrent_state_stride,
                    |slot_buffers| {
                        slot_buffers.conv_synced_generation = Some(conv_generation);
                        slot_buffers.recurrent_synced_generation = Some(recurrent_generation);
                    },
                );
                continue;
            }

            {
                let gpu_attention = qwen_kv.gpu_attention().unwrap();
                gpu_attention.read_layer_token_into(layer, seq_len, &mut mirror_k, &mut mirror_v);
            }
            qwen_kv.attention_append_cpu_mirror(layer, &mirror_k, &mirror_v);
        }

        qwen_kv.finalize_token();

        let logits_gpu = unsafe { &s.logits_buf.as_slice::<f32>()[..vocab_size] };
        logits[..vocab_size].copy_from_slice(logits_gpu);
        if let Some(ref mut ops_ref) = ops {
            ops_ref.gpu_readback += rb_t.elapsed();
        }
        Ok(true)
    }
}

impl ForwardPass for Qwen35Forward {
    #[allow(clippy::too_many_arguments)]
    fn forward_single(
        &self,
        ctx: &ForwardContext,
        token_id: u32,
        position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
        mut ops: Option<&mut OpBreakdown>,
    ) -> anyhow::Result<()> {
        // Prefer the native Metal single-token decode path by default.
        // The host-orchestrated fallback remains available for debugging or
        // rollout control via `AX_QWEN35_GPU_DECODE=off`.
        if Self::gpu_decode_enabled() {
            if let Some(ops_ref) = ops.as_deref_mut() {
                let t = OpTimer::start();
                let gpu_result = Self::try_forward_single_gpu(
                    ctx,
                    token_id,
                    position,
                    kv,
                    weights,
                    logits,
                    Some(ops_ref),
                );
                ops_ref.gpu += t.elapsed();
                if let Ok(true) = gpu_result {
                    return Ok(());
                }
            } else if let Ok(true) =
                Self::try_forward_single_gpu(ctx, token_id, position, kv, weights, logits, None)
            {
                return Ok(());
            }
        }

        let cfg = ctx.config;
        let qwen_kv = kv
            .as_qwen35_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen35Forward requires ModelKv::Qwen35"))?;
        let recurrent_slot = qwen_kv.active_slot();

        let dims = Self::recurrent_dims(cfg)?;
        let backend = ctx.backend;
        let n_layers = cfg.n_layers as usize;
        let dim = cfg.embedding_dim as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let inter_dim = cfg.intermediate_dim as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let vocab_size = cfg.vocab_size as usize;

        let mut hidden = vec![0.0f32; dim];
        timed!(
            ops,
            dequant,
            weights.dequantize_row("token_embd.weight", token_id as usize, &mut hidden)?
        );

        let mut norm_buf = vec![0.0f32; dim];
        let mut q_gate_buf = vec![0.0f32; q_dim * 2];
        let mut q_buf = vec![0.0f32; q_dim];
        let mut k_buf = vec![0.0f32; kv_dim];
        let mut v_buf = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; q_dim];
        let mut proj_buf = vec![0.0f32; dim];
        let mut gate_buf = vec![0.0f32; inter_dim];
        let mut up_buf = vec![0.0f32; inter_dim];
        let mut down_buf = vec![0.0f32; dim];
        let mut rec_qkv = vec![0.0f32; dims.conv_dim()];
        let mut rec_z = vec![0.0f32; dims.inner_size];
        let mut rec_beta = vec![0.0f32; dims.time_step_rank];
        let mut rec_alpha = vec![0.0f32; dims.time_step_rank];
        let mut rec_out = vec![0.0f32; dims.inner_size];
        let recurrent_slot_indices = [recurrent_slot];
        let full_attn_params = AttentionParams::new(n_heads, n_kv_heads, head_dim);

        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");
            Self::apply_attention_norm_single(
                weights,
                &prefix,
                &hidden,
                &mut norm_buf,
                cfg.rms_norm_eps,
                ops.as_deref_mut(),
            )?;
            Self::assert_finite_if_enabled("attn_norm_output", &norm_buf, layer, position)?;

            match Self::layer_type(cfg, layer) {
                Qwen35LayerType::FullAttention => Self::run_full_attention_single_layer(
                    cfg,
                    backend,
                    weights,
                    &prefix,
                    qwen_kv,
                    layer,
                    position,
                    &norm_buf,
                    &mut q_gate_buf,
                    &mut q_buf,
                    &mut k_buf,
                    &mut v_buf,
                    &mut attn_out,
                    &mut proj_buf,
                    dim,
                    q_dim,
                    kv_dim,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    &full_attn_params,
                    ops.as_deref_mut(),
                )?,
                Qwen35LayerType::RecurrentGdn => Self::run_recurrent_single_layer(
                    cfg,
                    backend,
                    weights,
                    &prefix,
                    qwen_kv,
                    recurrent_slot,
                    layer,
                    position,
                    dims,
                    &recurrent_slot_indices,
                    &norm_buf,
                    &mut rec_qkv,
                    &mut rec_z,
                    &mut rec_beta,
                    &mut rec_alpha,
                    &mut rec_out,
                    &mut proj_buf,
                    dim,
                    ops.as_deref_mut(),
                )?,
            }

            Self::apply_layer_tail_single(
                backend,
                weights,
                &prefix,
                &mut hidden,
                &proj_buf,
                &mut norm_buf,
                &mut gate_buf,
                &mut up_buf,
                &mut down_buf,
                dim,
                inter_dim,
                cfg.rms_norm_eps,
                layer,
                position,
                ops.as_deref_mut(),
            )?;
        }

        qwen_kv.finalize_token();

        let final_norm_w = timed!(ops, dequant, weights.f32_slice("output_norm.weight")?);
        timed!(
            ops,
            norm,
            rms_norm::rms_norm(&mut hidden, final_norm_w, cfg.rms_norm_eps)
        );

        timed_matmul_bucket!(
            ops,
            matmul_lm_head,
            Self::write_normalized_single_logits(
                backend, &hidden, dim, vocab_size, weights, logits,
            )?
        );
        Ok(())
    }

    fn forward_batch(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        self.forward_batch_impl(ctx, token_ids, kv, weights, Some(logits), None, None)
    }

    fn forward_batch_profiled(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
        ops: &mut OpBreakdown,
    ) -> anyhow::Result<()> {
        let force_serial = env_flag_enabled("AX_SERIAL_PREFILL");
        if !force_serial && ctx.backend.use_gpu_decode() && token_ids.len() > 1 {
            return self.forward_batch_impl(
                ctx,
                token_ids,
                kv,
                weights,
                Some(logits),
                None,
                Some(ops),
            );
        }

        let start_pos = kv.seq_len();
        for (i, &tid) in token_ids.iter().enumerate() {
            logits.fill(0.0);
            self.forward_single(ctx, tid, start_pos + i, kv, weights, logits, Some(ops))?;
        }
        Ok(())
    }

    fn forward_batch_all_logits(
        &self,
        ctx: &ForwardContext,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits_all: &mut Vec<f32>,
    ) -> anyhow::Result<()> {
        self.forward_batch_impl(ctx, token_ids, kv, weights, None, Some(logits_all), None)
    }

    fn validate_config(&self, config: &ModelConfig) -> anyhow::Result<()> {
        anyhow::ensure!(
            config.architecture == "qwen35",
            "Qwen35Forward only supports qwen35, got {}",
            config.architecture
        );
        let _ = Self::recurrent_dims(config)?;
        anyhow::ensure!(
            config.qwen35_full_attention_interval.unwrap_or(0) > 0,
            "qwen35 full_attention_interval must be > 0"
        );
        Ok(())
    }

    fn arch_name(&self) -> &str {
        "qwen35"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::gguf::MetadataValue;
    use crate::gguf::header::GgufHeader;
    use crate::gguf::mmap::MappedModel;
    use crate::gguf::tensor::GgmlType;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_header(kv: Vec<(&str, MetadataValue)>) -> GgufHeader {
        let mut metadata = HashMap::new();
        for (k, v) in kv {
            metadata.insert(k.to_string(), v);
        }
        GgufHeader {
            version: 3,
            tensor_count: 0,
            metadata,
        }
    }

    fn align_to(offset: usize, alignment: usize) -> usize {
        offset.div_ceil(alignment) * alignment
    }

    fn push_string_metadata(buf: &mut Vec<u8>, key: &str, value: &str) {
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
        buf.extend_from_slice(value.as_bytes());
    }

    fn push_u32_metadata(buf: &mut Vec<u8>, key: &str, value: u32) {
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn push_tensor_info(
        buf: &mut Vec<u8>,
        name: &str,
        shape: &[u64],
        dtype: GgmlType,
        offset: u64,
    ) {
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &dim in shape {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        buf.extend_from_slice(&(dtype as u32).to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
        for &value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn build_qwen35_logits_test_gguf(
        output_norm: &[f32],
        output_weight: &[f32],
        dim: usize,
        vocab_size: usize,
    ) -> Vec<u8> {
        let alignment = 32usize;
        let output_norm_bytes = f32_bytes(output_norm);
        let output_weight_bytes = f32_bytes(output_weight);
        let output_weight_offset = align_to(output_norm_bytes.len(), alignment);

        let mut buf = Vec::new();
        buf.extend_from_slice(&crate::gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&crate::gguf::GGUF_VERSION.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        push_string_metadata(&mut buf, "general.architecture", "qwen35");
        push_u32_metadata(&mut buf, "general.alignment", alignment as u32);
        push_tensor_info(
            &mut buf,
            "output_norm.weight",
            &[dim as u64],
            GgmlType::F32,
            0,
        );
        push_tensor_info(
            &mut buf,
            "output.weight",
            &[dim as u64, vocab_size as u64],
            GgmlType::F32,
            output_weight_offset as u64,
        );
        let data_start = align_to(buf.len(), alignment);
        buf.resize(data_start, 0);
        buf.extend_from_slice(&output_norm_bytes);
        buf.resize(data_start + output_weight_offset, 0);
        buf.extend_from_slice(&output_weight_bytes);
        buf
    }

    fn write_test_gguf_to_temp(data: &[u8]) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ax-qwen35-logits-{}-{}.gguf",
            std::process::id(),
            unique
        ));
        std::fs::write(&path, data).unwrap();
        path
    }

    #[test]
    fn test_qwen35_layer_pattern() {
        let header = make_header(vec![
            (
                "general.architecture",
                MetadataValue::String("qwen35".into()),
            ),
            ("qwen35.block_count", MetadataValue::Uint32(8)),
            ("qwen35.attention.head_count", MetadataValue::Uint32(16)),
            ("qwen35.attention.head_count_kv", MetadataValue::Uint32(8)),
            ("qwen35.embedding_length", MetadataValue::Uint32(2048)),
            ("qwen35.attention.key_length", MetadataValue::Uint32(128)),
            ("qwen35.feed_forward_length", MetadataValue::Uint32(8192)),
            ("qwen35.context_length", MetadataValue::Uint32(4096)),
            ("qwen35.full_attention_interval", MetadataValue::Uint32(4)),
            ("qwen35.ssm.conv_kernel", MetadataValue::Uint32(4)),
            ("qwen35.ssm.inner_size", MetadataValue::Uint32(1024)),
            ("qwen35.ssm.state_size", MetadataValue::Uint32(128)),
            ("qwen35.ssm.time_step_rank", MetadataValue::Uint32(8)),
            ("qwen35.ssm.group_count", MetadataValue::Uint32(2)),
        ]);
        let cfg = ModelConfig::from_gguf(&header).unwrap();
        assert!(cfg.qwen35_is_recurrent_layer(0));
        assert!(cfg.qwen35_is_recurrent_layer(1));
        assert!(cfg.qwen35_is_recurrent_layer(2));
        assert!(!cfg.qwen35_is_recurrent_layer(3));
    }

    #[test]
    fn test_qwen35_validate_requires_recurrent_dims() {
        let fwd = Qwen35Forward;
        let cfg = ModelConfig {
            architecture: "qwen35".into(),
            n_layers: 4,
            n_heads: 16,
            n_kv_heads: 8,
            embedding_dim: 2048,
            head_dim: 128,
            intermediate_dim: 8192,
            context_length: 4096,
            vocab_size: 1000,
            rms_norm_eps: 1e-6,
            rope_freq_base: 10000.0,
            has_qkv_bias: false,
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: crate::model::config::GateActivation::SiLU,
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: crate::model::config::RopeScaling::None,
            embed_scale: false,
            rope_freq_base_local: None,
            n_expert: None,
            n_expert_used: None,
            expert_intermediate_dim: None,
            qwen35_full_attention_interval: Some(4),
            qwen35_ssm_conv_kernel: Some(4),
            qwen35_ssm_inner_size: Some(1024),
            qwen35_ssm_state_size: Some(128),
            qwen35_ssm_time_step_rank: Some(8),
            qwen35_ssm_group_count: Some(2),
        };
        fwd.validate_config(&cfg).unwrap();
    }

    #[test]
    fn test_qwen35_rope_position_honors_linear_scaling() {
        let cfg = ModelConfig {
            architecture: "qwen35".into(),
            n_layers: 4,
            n_heads: 16,
            n_kv_heads: 8,
            embedding_dim: 2048,
            head_dim: 128,
            intermediate_dim: 8192,
            context_length: 4096,
            vocab_size: 1000,
            rms_norm_eps: 1e-6,
            rope_freq_base: 10000.0,
            has_qkv_bias: false,
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: crate::model::config::GateActivation::SiLU,
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: crate::model::config::RopeScaling::Linear(8.0),
            embed_scale: false,
            rope_freq_base_local: None,
            n_expert: None,
            n_expert_used: None,
            expert_intermediate_dim: None,
            qwen35_full_attention_interval: Some(4),
            qwen35_ssm_conv_kernel: Some(4),
            qwen35_ssm_inner_size: Some(1024),
            qwen35_ssm_state_size: Some(128),
            qwen35_ssm_time_step_rank: Some(8),
            qwen35_ssm_group_count: Some(2),
        };

        assert!((Qwen35Forward::rope_position(&cfg, 16) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_qwen35_rope_position_uses_current_yarn_fallback_scaling() {
        let cfg = ModelConfig {
            architecture: "qwen35".into(),
            n_layers: 4,
            n_heads: 16,
            n_kv_heads: 8,
            embedding_dim: 2048,
            head_dim: 128,
            intermediate_dim: 8192,
            context_length: 4096,
            vocab_size: 1000,
            rms_norm_eps: 1e-6,
            rope_freq_base: 10000.0,
            has_qkv_bias: false,
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: crate::model::config::GateActivation::SiLU,
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: crate::model::config::RopeScaling::Yarn {
                factor: 8.0,
                ext_factor: 1.0,
                attn_factor: 1.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                orig_ctx_len: 8192,
            },
            embed_scale: false,
            rope_freq_base_local: None,
            n_expert: None,
            n_expert_used: None,
            expert_intermediate_dim: None,
            qwen35_full_attention_interval: Some(4),
            qwen35_ssm_conv_kernel: Some(4),
            qwen35_ssm_inner_size: Some(1024),
            qwen35_ssm_state_size: Some(128),
            qwen35_ssm_time_step_rank: Some(8),
            qwen35_ssm_group_count: Some(2),
        };

        assert!((Qwen35Forward::rope_position(&cfg, 16) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_qwen35_apply_rope_batch_uses_absolute_positions() {
        let cfg = ModelConfig {
            architecture: "qwen35".into(),
            n_layers: 4,
            n_heads: 1,
            n_kv_heads: 1,
            embedding_dim: 8,
            head_dim: 4,
            intermediate_dim: 16,
            context_length: 4096,
            vocab_size: 1000,
            rms_norm_eps: 1e-6,
            rope_freq_base: 10000.0,
            has_qkv_bias: false,
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: crate::model::config::GateActivation::SiLU,
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: crate::model::config::RopeScaling::None,
            embed_scale: false,
            rope_freq_base_local: None,
            n_expert: None,
            n_expert_used: None,
            expert_intermediate_dim: None,
            qwen35_full_attention_interval: Some(4),
            qwen35_ssm_conv_kernel: Some(4),
            qwen35_ssm_inner_size: Some(16),
            qwen35_ssm_state_size: Some(4),
            qwen35_ssm_time_step_rank: Some(4),
            qwen35_ssm_group_count: Some(1),
        };
        let n_tokens = 2usize;
        let start_position = 7usize;
        let q_dim = 4usize;
        let kv_dim = 4usize;
        let n_heads = 1usize;
        let n_kv_heads = 1usize;
        let head_dim = 4usize;
        let mut actual_q = vec![1.0f32, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5];
        let mut actual_k = vec![4.0f32, 3.0, 2.0, 1.0, 3.5, 2.5, 1.5, 0.5];
        let mut expected_q = actual_q.clone();
        let mut expected_k = actual_k.clone();

        for token_idx in 0..n_tokens {
            let q_start = token_idx * q_dim;
            let k_start = token_idx * kv_dim;
            rope::apply_rope_multi_head_scaled(
                &mut expected_q[q_start..q_start + q_dim],
                &mut expected_k[k_start..k_start + kv_dim],
                n_heads,
                n_kv_heads,
                head_dim,
                Qwen35Forward::rope_position(&cfg, start_position + token_idx),
                cfg.rope_freq_base,
            );
        }

        Qwen35Forward::apply_rope_batch(
            &cfg,
            &mut actual_q,
            &mut actual_k,
            n_tokens,
            start_position,
            q_dim,
            kv_dim,
            n_heads,
            n_kv_heads,
            head_dim,
        );

        for (actual, expected) in actual_q.iter().zip(expected_q.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        for (actual, expected) in actual_k.iter().zip(expected_k.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_qwen35_prepare_full_attention_qk_batch_matches_staged_path() {
        let cfg = ModelConfig {
            architecture: "qwen35".into(),
            n_layers: 4,
            n_heads: 1,
            n_kv_heads: 1,
            embedding_dim: 8,
            head_dim: 4,
            intermediate_dim: 16,
            context_length: 4096,
            vocab_size: 1000,
            rms_norm_eps: 1e-6,
            rope_freq_base: 10000.0,
            has_qkv_bias: false,
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: crate::model::config::GateActivation::SiLU,
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: crate::model::config::RopeScaling::None,
            embed_scale: false,
            rope_freq_base_local: None,
            n_expert: None,
            n_expert_used: None,
            expert_intermediate_dim: None,
            qwen35_full_attention_interval: Some(4),
            qwen35_ssm_conv_kernel: Some(4),
            qwen35_ssm_inner_size: Some(16),
            qwen35_ssm_state_size: Some(4),
            qwen35_ssm_time_step_rank: Some(4),
            qwen35_ssm_group_count: Some(1),
        };
        let q_gate_batch = vec![
            1.0f32, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.4, 1.5, 2.5, 3.5, 4.5, 0.5, 0.6, 0.7, 0.8,
        ];
        let mut actual_q = vec![0.0f32; 8];
        let mut actual_k = vec![4.0f32, 3.0, 2.0, 1.0, 3.5, 2.5, 1.5, 0.5];
        let mut expected_q = actual_q.clone();
        let mut expected_k = actual_k.clone();
        let norm_weights = Some(Qwen35AttentionNormWeights {
            q: &[1.0f32, 1.1, 1.2, 1.3],
            k: &[0.9f32, 1.0, 1.1, 1.2],
        });

        Qwen35Forward::extract_q_from_q_gate_batch(&q_gate_batch, &mut expected_q, 2, 4);
        Qwen35Forward::apply_attention_qk_norm_batch(
            &mut expected_q,
            &mut expected_k,
            2,
            4,
            4,
            1,
            1,
            4,
            norm_weights.unwrap(),
            cfg.rms_norm_eps,
        );
        Qwen35Forward::apply_rope_batch(
            &cfg,
            &mut expected_q,
            &mut expected_k,
            2,
            7,
            4,
            4,
            1,
            1,
            4,
        );

        Qwen35Forward::prepare_full_attention_qk_batch(
            &cfg,
            &q_gate_batch,
            &mut actual_q,
            &mut actual_k,
            2,
            7,
            4,
            4,
            1,
            1,
            4,
            norm_weights,
            cfg.rms_norm_eps,
        );

        for (actual, expected) in actual_q.iter().zip(expected_q.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        for (actual, expected) in actual_k.iter().zip(expected_k.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_qwen35_full_attention_input_plan_fuses_matching_dtypes() {
        let wq = [1u8, 2, 3, 4];
        let wk = [5u8, 6];
        let wv = [7u8, 8, 9];
        let plan = Qwen35Forward::maybe_fused_full_attention_input_plan([
            (&wq, GgmlType::Q4K, 8),
            (&wk, GgmlType::Q4K, 4),
            (&wv, GgmlType::Q4K, 4),
        ]);

        match plan {
            Qwen35FullAttentionInputPlan::Fused { raw, dtype, rows } => {
                assert_eq!(dtype, GgmlType::Q4K);
                assert_eq!(rows, 16);
                assert_eq!(raw.as_ref(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
            }
            Qwen35FullAttentionInputPlan::Split(_) => {
                panic!("expected fused full-attention input plan")
            }
        }
    }

    #[test]
    fn test_qwen35_split_full_attention_fused_output_batch_layout() {
        let fused = vec![
            1.0f32, 2.0, 3.0, 4.0, 10.0, 11.0, 20.0, 21.0, 5.0, 6.0, 7.0, 8.0, 12.0, 13.0, 22.0,
            23.0,
        ];
        let mut q_gate = vec![0.0f32; 8];
        let mut k = vec![0.0f32; 4];
        let mut v = vec![0.0f32; 4];

        Qwen35Forward::split_full_attention_fused_output_batch(
            &fused,
            &mut q_gate,
            &mut k,
            &mut v,
            2,
        );

        assert_eq!(q_gate, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(k, vec![10.0f32, 11.0, 12.0, 13.0]);
        assert_eq!(v, vec![20.0f32, 21.0, 22.0, 23.0]);
    }

    #[test]
    fn test_qwen35_validate_rejects_incompatible_head_expansion() {
        let fwd = Qwen35Forward;
        let cfg = ModelConfig {
            architecture: "qwen35".into(),
            n_layers: 4,
            n_heads: 16,
            n_kv_heads: 8,
            embedding_dim: 2048,
            head_dim: 128,
            intermediate_dim: 8192,
            context_length: 4096,
            vocab_size: 1000,
            rms_norm_eps: 1e-6,
            rope_freq_base: 10000.0,
            has_qkv_bias: false,
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: crate::model::config::GateActivation::SiLU,
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: crate::model::config::RopeScaling::None,
            embed_scale: false,
            rope_freq_base_local: None,
            n_expert: None,
            n_expert_used: None,
            expert_intermediate_dim: None,
            qwen35_full_attention_interval: Some(4),
            qwen35_ssm_conv_kernel: Some(4),
            qwen35_ssm_inner_size: Some(768),
            qwen35_ssm_state_size: Some(128),
            qwen35_ssm_time_step_rank: Some(6),
            qwen35_ssm_group_count: Some(4),
        };

        let err = fwd.validate_config(&cfg).unwrap_err();
        assert!(err.to_string().contains("multiple of group_count"));
    }

    #[test]
    fn test_qwen35_write_all_batch_logits_matches_per_token_reference() {
        let dim = 4usize;
        let vocab_size = 3usize;
        let n_tokens = 2usize;
        let rms_norm_eps = 1e-6f32;
        let output_norm = [1.0f32, 0.5, 1.5, 0.75];
        let output_weight = [
            0.25f32, -0.5, 1.0, 0.75, -1.0, 0.5, 0.25, -0.75, 0.1, 0.2, -0.3, 0.4,
        ];
        let hidden = vec![1.0f32, -2.0, 0.5, 3.0, -1.0, 0.25, 2.0, -0.5];
        let gguf = build_qwen35_logits_test_gguf(&output_norm, &output_weight, dim, vocab_size);
        let path = write_test_gguf_to_temp(&gguf);

        let result = (|| {
            let model = MappedModel::open(&path).unwrap();
            let weights = WeightStore::new(&model);
            let backend = CpuBackend;

            let mut actual = Vec::new();
            Qwen35Forward::write_all_batch_logits(
                &backend,
                &hidden,
                n_tokens,
                dim,
                vocab_size,
                rms_norm_eps,
                &weights,
                &mut actual,
            )
            .unwrap();

            let mut expected = vec![0.0f32; n_tokens * vocab_size];
            for token_idx in 0..n_tokens {
                let hidden_start = token_idx * dim;
                let logits_start = token_idx * vocab_size;
                let mut token_hidden = hidden[hidden_start..hidden_start + dim].to_vec();
                Qwen35Forward::write_single_logits(
                    &backend,
                    &mut token_hidden,
                    dim,
                    vocab_size,
                    rms_norm_eps,
                    &weights,
                    &mut expected[logits_start..logits_start + vocab_size],
                )
                .unwrap();
            }

            assert_eq!(actual.len(), expected.len());
            for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "logit {idx} mismatch: actual={actual}, expected={expected}"
                );
            }
        })();

        std::fs::remove_file(&path).ok();
        result
    }
}
