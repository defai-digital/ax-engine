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
//! AX routes the recurrent projections and recurrent primitives through the
//! active backend, including Metal kernels for the causal-conv and gated-delta
//! steps. GPU-resident recurrent buffers are enabled by default when allocation
//! succeeds, but CPU materialization and backend-owned state semantics still
//! exist as fallback paths.
//!
//! The batch prefill path (`try_forward_batch_gpu_unified`) supports inter-step
//! pipelining (`AX_QWEN35_PREFILL_PIPELINED`): recurrent tail command buffers
//! are submitted asynchronously, and K/V CPU mirror copies are deferred, so CPU
//! encoding of step N+1 overlaps GPU execution of step N.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use ax_engine_metal::MetalBuffer;
use rayon::prelude::*;

use crate::compute::attention::AttentionParams;
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
    apply_attention_norm_single, apply_output_norm_single, cache_attention_qk_norm_keys,
    cache_output_head_keys, encode_batch_logits, encode_dequant_batch, encode_dequant_batch_f16in,
    ensure_precomputed_linear_f16, env_flag_enabled, env_flag_override, gpu_batch_logits_supported,
    write_normalized_single_logits,
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
    /// Cached dt_bias tensor for prepare_alpha_beta on GPU.
    dt_bias: usize,
    /// Cached A tensor for prepare_alpha_beta on GPU.
    ssm_a: usize,
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

#[derive(Default)]
struct Qwen35CpuBatchFallbackScratch {
    hidden: Vec<f32>,
    norm_buf: Vec<f32>,
    proj_buf: Vec<f32>,
    gate_buf: Vec<f32>,
    up_buf: Vec<f32>,
    down_buf: Vec<f32>,
    q_gate_batch: Vec<f32>,
    q_batch: Vec<f32>,
    k_batch: Vec<f32>,
    v_batch: Vec<f32>,
    fused_input_batch: Vec<f32>,
    attn_out_batch: Vec<f32>,
    rec_qkv_batch: Vec<f32>,
    rec_z_batch: Vec<f32>,
    rec_beta_batch: Vec<f32>,
    rec_alpha_batch: Vec<f32>,
    rec_out_batch: Vec<f32>,
    final_hidden: Vec<f32>,
}

impl Qwen35CpuBatchFallbackScratch {
    #[allow(clippy::too_many_arguments)]
    fn ensure_lengths(
        &mut self,
        n_tokens: usize,
        dim: usize,
        inter_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        recurrent_slot_count: usize,
        conv_dim: usize,
        inner_size: usize,
        time_step_rank: usize,
    ) {
        self.hidden.resize(n_tokens * dim, 0.0);
        self.norm_buf.resize(n_tokens * dim, 0.0);
        self.proj_buf.resize(n_tokens * dim, 0.0);
        self.gate_buf.resize(n_tokens * inter_dim, 0.0);
        self.up_buf.resize(n_tokens * inter_dim, 0.0);
        self.down_buf.resize(n_tokens * dim, 0.0);
        self.q_gate_batch.resize(n_tokens * q_dim * 2, 0.0);
        self.q_batch.resize(n_tokens * q_dim, 0.0);
        self.k_batch.resize(n_tokens * kv_dim, 0.0);
        self.v_batch.resize(n_tokens * kv_dim, 0.0);
        self.fused_input_batch
            .resize(n_tokens * (q_dim * 2 + 2 * kv_dim), 0.0);
        self.attn_out_batch.resize(n_tokens * q_dim, 0.0);
        self.rec_qkv_batch
            .resize(n_tokens * recurrent_slot_count * conv_dim, 0.0);
        self.rec_z_batch
            .resize(n_tokens * recurrent_slot_count * inner_size, 0.0);
        self.rec_beta_batch
            .resize(n_tokens * recurrent_slot_count * time_step_rank, 0.0);
        self.rec_alpha_batch
            .resize(n_tokens * recurrent_slot_count * time_step_rank, 0.0);
        self.rec_out_batch
            .resize(n_tokens * recurrent_slot_count * inner_size, 0.0);
        self.final_hidden.resize(n_tokens * dim, 0.0);
    }
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

type Qwen35AttentionNormWeights<'a> = crate::model::shared::AttentionQkNormWeights<'a>;

#[derive(Clone, Copy)]
struct Qwen35RecurrentRuntimeTensors<'a> {
    dt_bias: &'a [f32],
    a: &'a [f32],
    conv_kernel: &'a [f32],
    ssm_norm: &'a [f32],
}

#[derive(Default, Clone, Copy)]
struct Qwen35RecurrentGpuPrefillStats {
    gpu_execute: Duration,
    gpu_readback: Duration,
}

#[derive(Default, Clone, Copy)]
struct Qwen35RecurrentQkvFastPathCheck {
    state_size_too_large: bool,
    group_divisibility_invalid: bool,
    missing_batch_scratches: bool,
    q_capacity_too_small: bool,
    k_capacity_too_small: bool,
    v_capacity_too_small: bool,
    gate_capacity_too_small: bool,
    up_capacity_too_small: bool,
}

impl Qwen35RecurrentQkvFastPathCheck {
    fn is_eligible(self) -> bool {
        !self.state_size_too_large
            && !self.group_divisibility_invalid
            && !self.missing_batch_scratches
            && !self.q_capacity_too_small
            && !self.k_capacity_too_small
            && !self.v_capacity_too_small
            && !self.gate_capacity_too_small
            && !self.up_capacity_too_small
    }
}

/// Parameters for encoding input projections into the fused recurrent CB.
struct FusedProjectionParams<'a> {
    nw_key: usize,
    gpu_proj_indices: &'a [usize],
    proj_keys: &'a [usize],
    proj_dtypes: &'a [crate::gguf::tensor::GgmlType],
    proj_dims: &'a [usize],
    dim: usize,
    eps: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum Qwen35PrefillRecurrentStateMode {
    Auto,
    CpuAlias,
    SlotBuffer,
    BackendOwned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum Qwen35PrefillAlphaBetaStorageMode {
    Auto,
    F32,
    F16,
}

impl Qwen35Forward {
    fn warn_gpu_decode_fallback_once(error: &anyhow::Error) {
        static WARNED_GPU_DECODE_FALLBACKS: OnceLock<Mutex<HashMap<String, ()>>> = OnceLock::new();

        let key = error.to_string();
        let warned = WARNED_GPU_DECODE_FALLBACKS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut warned = warned
            .lock()
            .expect("WARNED_GPU_DECODE_FALLBACKS mutex should not be poisoned");
        if warned.insert(key.clone(), ()).is_none() {
            tracing::warn!(
                error = %key,
                "Qwen3.5 native GPU decode fell back to host-orchestrated decode"
            );
        }
    }

    fn warn_prefill_unified_fallback_once(reason: &'static str) {
        static WARNED_GPU_PREFILL_FALLBACKS: OnceLock<Mutex<HashMap<&'static str, ()>>> =
            OnceLock::new();

        let warned = WARNED_GPU_PREFILL_FALLBACKS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut warned = warned
            .lock()
            .expect("WARNED_GPU_PREFILL_FALLBACKS mutex should not be poisoned");
        if warned.insert(reason, ()).is_none() {
            tracing::warn!(
                reason,
                "Qwen3.5 unified GPU prefill fell back to the older path"
            );
        }
    }

    const PARALLEL_EMBEDDING_DEQUANT_THRESHOLD: usize = 32;

    fn dequantize_token_embeddings_batch(
        weights: &WeightStore<'_>,
        token_ids: &[u32],
        hidden: &mut [f32],
        dim: usize,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            hidden.len() >= token_ids.len() * dim,
            "embedding output buffer too small for token batch"
        );
        let hidden = &mut hidden[..token_ids.len() * dim];
        if token_ids.len() >= Self::PARALLEL_EMBEDDING_DEQUANT_THRESHOLD {
            token_ids
                .par_iter()
                .zip(hidden.par_chunks_mut(dim))
                .try_for_each(|(&tid, row)| {
                    weights.dequantize_row("token_embd.weight", tid as usize, row)?;
                    Ok::<_, anyhow::Error>(())
                })?;
        } else {
            for (row, &tid) in hidden.chunks_mut(dim).zip(token_ids.iter()) {
                weights.dequantize_row("token_embd.weight", tid as usize, row)?;
            }
        }
        Ok(())
    }

    fn full_attention_fused_quant_cache()
    -> &'static Mutex<HashMap<(usize, usize, usize), FusedQuantData>> {
        #[allow(clippy::type_complexity)]
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

    fn gpu_batch_logits_enabled() -> bool {
        env_flag_override("AX_QWEN35_GPU_BATCH_LOGITS").unwrap_or(true)
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

    #[allow(dead_code)]
    fn qwen35_prefill_recurrent_state_mode() -> Qwen35PrefillRecurrentStateMode {
        match std::env::var("AX_QWEN35_PREFILL_RECURRENT_STATE_MODE") {
            Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                "cpu_alias" | "cpu_alias_only" | "cpu" => Qwen35PrefillRecurrentStateMode::CpuAlias,
                "slot_buffer" | "slot_buffer_only" | "backend_slot" | "gpu_slot" => {
                    Qwen35PrefillRecurrentStateMode::SlotBuffer
                }
                "backend_owned" | "persistent_slot_buffer" | "slot_buffer_carryover" => {
                    Qwen35PrefillRecurrentStateMode::BackendOwned
                }
                _ => Qwen35PrefillRecurrentStateMode::Auto,
            },
            Err(_) => Qwen35PrefillRecurrentStateMode::Auto,
        }
    }

    fn qwen35_prefill_recurrent_state_mode_for_tokens(
        n_tokens: usize,
    ) -> Qwen35PrefillRecurrentStateMode {
        let _ = n_tokens;
        match Self::qwen35_prefill_recurrent_state_mode() {
            Qwen35PrefillRecurrentStateMode::Auto => Qwen35PrefillRecurrentStateMode::BackendOwned,
            explicit => explicit,
        }
    }

    pub(crate) fn qwen35_prefill_force_backend_state_batch() -> bool {
        env_flag_enabled("AX_QWEN35_PREFILL_FORCE_BACKEND_STATE_BATCH")
    }

    fn cpu_batch_fallback_scratch() -> &'static Mutex<Qwen35CpuBatchFallbackScratch> {
        static SCRATCH: OnceLock<Mutex<Qwen35CpuBatchFallbackScratch>> = OnceLock::new();
        SCRATCH.get_or_init(|| Mutex::new(Qwen35CpuBatchFallbackScratch::default()))
    }

    pub(crate) fn qwen35_prefill_backend_state_batch_for_tokens(
        n_tokens: usize,
        layer_state_owner: crate::kv::Qwen35LayerStateOwner,
    ) -> bool {
        if Self::qwen35_prefill_force_backend_state_batch() {
            return true;
        }
        if layer_state_owner != crate::kv::Qwen35LayerStateOwner::CpuMaterialized {
            return true;
        }
        let _ = n_tokens;
        false
    }

    fn resolve_qwen35_prefill_recurrent_state_mode(
        n_tokens: usize,
        layer_state_owner: crate::kv::Qwen35LayerStateOwner,
    ) -> Qwen35PrefillRecurrentStateMode {
        match Self::qwen35_prefill_recurrent_state_mode() {
            Qwen35PrefillRecurrentStateMode::Auto => {
                if layer_state_owner != crate::kv::Qwen35LayerStateOwner::CpuMaterialized {
                    Qwen35PrefillRecurrentStateMode::BackendOwned
                } else {
                    Self::qwen35_prefill_recurrent_state_mode_for_tokens(n_tokens)
                }
            }
            explicit => explicit,
        }
    }

    #[allow(dead_code)]
    fn qwen35_prefill_alpha_beta_storage_mode() -> Qwen35PrefillAlphaBetaStorageMode {
        match std::env::var("AX_QWEN35_PREFILL_ALPHA_BETA_STORAGE_MODE") {
            Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                "f16" | "half" => Qwen35PrefillAlphaBetaStorageMode::F16,
                "f32" | "float" => Qwen35PrefillAlphaBetaStorageMode::F32,
                _ => Qwen35PrefillAlphaBetaStorageMode::Auto,
            },
            Err(_) => Qwen35PrefillAlphaBetaStorageMode::Auto,
        }
    }

    fn qwen35_prefill_alpha_beta_storage_mode_for_tokens(
        _n_tokens: usize,
    ) -> Qwen35PrefillAlphaBetaStorageMode {
        match Self::qwen35_prefill_alpha_beta_storage_mode() {
            Qwen35PrefillAlphaBetaStorageMode::Auto => Qwen35PrefillAlphaBetaStorageMode::F32,
            explicit => explicit,
        }
    }

    fn prepare_qwen35_handoff_alpha_beta(
        alpha_dst: &mut [f32],
        beta_dst: &mut [f32],
        alpha_src: &[f32],
        beta_src: &[f32],
        dt_bias: &[f32],
        a: &[f32],
    ) {
        assert_eq!(
            alpha_dst.len(),
            alpha_src.len(),
            "qwen35 recurrent alpha handoff length mismatch"
        );
        assert_eq!(
            beta_dst.len(),
            beta_src.len(),
            "qwen35 recurrent beta handoff length mismatch"
        );
        alpha_dst.copy_from_slice(alpha_src);
        beta_dst.copy_from_slice(beta_src);
        gdn::prepare_alpha_beta(alpha_dst, beta_dst, dt_bias, a);
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
                | crate::gguf::tensor::GgmlType::F32
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
        if metal_ops.metal_precompute_f16_enabled() {
            ensure_precomputed_linear_f16(metal_ops, raw, dtype, m, k)?;
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
        if dtype == crate::gguf::tensor::GgmlType::F32 {
            // Dense f32 weights (beta/alpha): simdgroup matmul, no dequant.
            metal_ops
                .matmul
                .encode_matmul(encoder, weight, input_f32, output, m, n, k);
        } else if Self::qwen35_batch_projection_needs_f16_input(dtype) {
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

            // Common weights: attn_norm, post_attention_norm.
            let attn_norm_w = weights.f32_slice(&format!("{prefix}.attn_norm.weight"))?;
            let attn_norm_key = metal_ops.ensure_f32_cached(attn_norm_w);
            let ffn_norm_w = weights.f32_slice(&format!("{prefix}.post_attention_norm.weight"))?;
            let ffn_norm_key = metal_ops.ensure_f32_cached(ffn_norm_w);

            // FFN weights: dense (ffn_gate/up/down) or MoE (ffn_gate_exps/etc).
            // MoE layers lack dense FFN tensors — use sentinel key 0 and handle
            // the MoE FFN on CPU during the forward pass.
            let layer_is_moe = Self::qwen35_layer_uses_moe(weights, &prefix);
            let (wg_key, wg_dtype, wu_key, wu_dtype, wd_key, wd_dtype) = if layer_is_moe {
                // MoE layer: no dense FFN weights to cache.
                // Use sentinel key 0 (never looked up — the per-layer encoding
                // detects MoE and falls back to CPU for the FFN step).
                (
                    0usize,
                    crate::gguf::tensor::GgmlType::F32,
                    0usize,
                    crate::gguf::tensor::GgmlType::F32,
                    0usize,
                    crate::gguf::tensor::GgmlType::F32,
                )
            } else {
                let (wg_raw, wg_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.ffn_gate.weight"))?;
                let (wu_raw, wu_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.ffn_up.weight"))?;
                let (wd_raw, wd_dtype) =
                    weights.raw_with_dtype(&format!("{prefix}.ffn_down.weight"))?;
                let wg_key = metal_ops.ensure_quant_cached(wg_raw);
                let wu_key = metal_ops.ensure_quant_cached(wu_raw);
                let wd_key = metal_ops.ensure_quant_cached(wd_raw);
                (wg_key, wg_dtype, wu_key, wu_dtype, wd_key, wd_dtype)
            };

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
                let dt_bias = weights.f32_slice(&format!("{prefix}.ssm_dt.bias"))?;
                let dt_bias_key = metal_ops.ensure_f32_cached(dt_bias);
                let ssm_a = weights.f32_slice(&format!("{prefix}.ssm_a"))?;
                let ssm_a_key = metal_ops.ensure_f32_cached(ssm_a);

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
                    dt_bias: dt_bias_key,
                    ssm_a: ssm_a_key,
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
                let (attn_q_norm_key, attn_k_norm_key) =
                    cache_attention_qk_norm_keys(metal_ops, weights, &prefix)?;
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

        let (output_norm_key, _lm_raw, lm_dtype, lm_key) =
            cache_output_head_keys(metal_ops, weights)?;

        metal_ops.set_cached_model_keys(CachedModelKeys {
            layers,
            output_norm: output_norm_key,
            lm_head: lm_key,
            lm_head_dtype: lm_dtype,
        });
        Self::store_gpu_layer_keys(lm_key, layer_types);

        Ok(())
    }
}

include!("qwen35/helpers.rs");
include!("qwen35/logits.rs");
include!("qwen35/attention.rs");
include!("qwen35/recurrent.rs");
include!("qwen35/batch.rs");
include!("qwen35/decode.rs");
#[cfg(test)]
mod tests;

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
                match gpu_result {
                    Ok(true) => return Ok(()),
                    Ok(false) => {}
                    Err(err) => Self::warn_gpu_decode_fallback_once(&err),
                }
            } else {
                match Self::try_forward_single_gpu(
                    ctx, token_id, position, kv, weights, logits, None,
                ) {
                    Ok(true) => return Ok(()),
                    Ok(false) => {}
                    Err(err) => Self::warn_gpu_decode_fallback_once(&err),
                }
            }
        }

        let cfg = ctx.config;
        let qwen_kv = kv
            .as_qwen35_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen35Forward requires ModelKv::Qwen35"))?;
        let recurrent_slot = qwen_kv.active_slot();

        let dims = Self::recurrent_dims(cfg)?;
        // MoE expert matmuls use CpuBackend internally (in attention.rs).
        // Projections and recurrent ops use the default backend (GPU on Metal)
        // for better throughput — GPU fused dequant matvec is faster than
        // CPU NEON for the large projection weights.
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
            apply_attention_norm_single(
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
                dim,
                inter_dim,
                cfg.rms_norm_eps,
                layer,
                position,
                ops.as_deref_mut(),
            )?;
        }

        qwen_kv.finalize_token();

        apply_output_norm_single(weights, &mut hidden, cfg.rms_norm_eps, ops.as_deref_mut())?;

        timed_matmul_bucket!(
            ops,
            matmul_lm_head,
            write_normalized_single_logits(backend, &hidden, dim, vocab_size, weights, logits)?
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
            matches!(config.architecture.as_str(), "qwen35" | "qwen35moe"),
            "Qwen35Forward only supports qwen35/qwen35moe, got {}",
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

    fn supports_pipelined_decode(&self, ctx: &ForwardContext) -> bool {
        Self::gpu_decode_enabled()
            && ctx.backend.metal_ops().is_some()
            && !Self::qwen35_is_moe(ctx.config)
    }

    fn embed_pipelined_token(
        &self,
        ctx: &ForwardContext,
        token_id: u32,
        hidden_buf: &ax_engine_metal::MetalBuffer,
        weights: &WeightStore,
    ) -> anyhow::Result<()> {
        let dim = ctx.config.embedding_dim as usize;
        let hidden = unsafe {
            std::slice::from_raw_parts_mut(hidden_buf.contents().as_ptr() as *mut f32, dim)
        };
        weights
            .dequantize_row("token_embd.weight", token_id as usize, hidden)
            .map(|_| ())
    }

    fn encode_pending_decode_step(
        &self,
        ctx: &ForwardContext,
        hidden_buf: &ax_engine_metal::MetalBuffer,
        position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
    ) -> anyhow::Result<Option<ax_engine_metal::PendingFrame>> {
        if Self::qwen35_is_moe(ctx.config) {
            return Ok(None);
        }
        if !Self::gpu_decode_enabled() {
            return Ok(None);
        }
        let Some(metal_ops) = ctx.backend.metal_ops() else {
            return Ok(None);
        };
        let Some(qwen_kv) = kv.as_qwen35_mut() else {
            return Ok(None);
        };
        if qwen_kv.gpu_attention().is_none() {
            return Ok(None);
        }
        let frame = Self::encode_qwen35_pending_step(
            metal_ops, ctx.config, hidden_buf, position, qwen_kv, weights,
        )?;
        Ok(Some(frame))
    }

    fn supports_fused_argmax(&self) -> bool {
        true
    }

    fn encode_pending_decode_step_with_argmax(
        &self,
        ctx: &ForwardContext,
        hidden_buf: &ax_engine_metal::MetalBuffer,
        position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
    ) -> anyhow::Result<Option<ax_engine_metal::PendingFrame>> {
        if Self::qwen35_is_moe(ctx.config) {
            return Ok(None);
        }
        if !Self::gpu_decode_enabled() {
            return Ok(None);
        }
        let Some(metal_ops) = ctx.backend.metal_ops() else {
            return Ok(None);
        };
        let Some(qwen_kv) = kv.as_qwen35_mut() else {
            return Ok(None);
        };
        if qwen_kv.gpu_attention().is_none() {
            return Ok(None);
        }
        let frame = Self::encode_qwen35_pending_step_with_argmax(
            metal_ops, ctx.config, hidden_buf, position, qwen_kv, weights,
        )?;
        Ok(Some(frame))
    }
}
