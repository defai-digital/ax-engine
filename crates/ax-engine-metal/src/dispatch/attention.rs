use super::*;

/// Threadgroup size for attention prefill kernel (must match shader constant).
const ATTN_TG: usize = 256;
/// Threadgroup size for alternative prefill kernel (must match ATTN2_TG in shader).
const ATTN2_TG: usize = 128;
/// Threadgroup size for default decode attention kernel (must match shader constant).
const ATTN_DEC_TG: usize = 256;
/// Threadgroup size for alternative decode attention kernel (must match ATTN_DEC2_TG in shader).
const ATTN_DEC2_TG: usize = 128;
/// Maximum head dimension supported by generic attention shaders
/// (must match MAX_HD in attention.metal).
const MAX_HEAD_DIM: u32 = 512;

pub struct AttentionKernels {
    prefill: ComputePipeline,
    prefill_hd256: ComputePipeline,
    prefill_v2: ComputePipeline,
    prefill_v2_hd128: ComputePipeline,
    prefill_ax_hd128: ComputePipeline,
    prefill_ax_f16out_hd128: ComputePipeline,
    prefill_ax_hd128_smem: ComputePipeline,
    prefill_ax_hd128_smem_f16: ComputePipeline,
    prefill_ax_hd128_bc64: ComputePipeline,
    prefill_fa2_hd128: ComputePipeline,
    /// FA2 with simdgroup_float8x8 matrix ops (HD=128).
    prefill_fa2_simd_hd128: ComputePipeline,
    /// FA2 full-half prefill (HD=128) — K/V staged as half, half×half→float MMA.
    prefill_fa2_half_hd128: ComputePipeline,
    /// FA2 v2 prefill attention (HD=128) — experimental A/B test kernel.
    prefill_fa2v2_hd128: ComputePipeline,
    /// FA2 v2 prefill attention (HD=256) — experimental A/B test kernel.
    prefill_fa2v2_hd256: ComputePipeline,
    prefill_fa2_simd_hd256: ComputePipeline,
    prefill_fa2_simd_hd64: ComputePipeline,
    prefill_cache: ComputePipeline,
    prefill_cache_f16kv: ComputePipeline,
    /// FA2-style 8-query-per-TG prefill (HD=256, f16 KV).
    prefill_cache_fa2_hd256: ComputePipeline,
    /// FA2 SIMD cached prefill (HD=128, f16 KV) with mask tile-skip.
    prefill_cache_fa2_simd_hd128: ComputePipeline,
    /// FA2 SIMD cached prefill (HD=64, f16 KV) with mask tile-skip.
    prefill_cache_fa2_simd_hd64: ComputePipeline,
    /// FA2 SIMD cached prefill (HD=256, f16 KV) with mask tile-skip.
    prefill_cache_fa2_simd_hd256: ComputePipeline,
    decode: ComputePipeline,
    decode_hd256: ComputePipeline,
    decode_f16kv: ComputePipeline,
    decode_f16kv_hd256: ComputePipeline,
    decode_v2: ComputePipeline,
    /// sdpa_vector decode (HD=256, f16 KV) — MLX-pattern lane-parallel.
    decode_sdpa_hd256: ComputePipeline,
    /// Optimized decode for head_dim=128 (Qwen3), f32 KV.
    decode_hd128: ComputePipeline,
    /// Optimized decode for head_dim=128 (Qwen3), f16 KV.
    decode_f16kv_hd128: ComputePipeline,
    /// Two-head-per-TG decode for head_dim=128, f16 KV.
    decode_f16kv_hd128_n2: ComputePipeline,
    /// Split-K partial decode for head_dim=128, f16 KV.
    decode_splitk_f16kv_hd128_partial: ComputePipeline,
    /// Split-K reduction decode for head_dim=128, f16 KV.
    decode_splitk_f16kv_hd128_reduce: ComputePipeline,
    /// Split-K partial decode for head_dim=256, f16 KV.
    decode_splitk_f16kv_hd256_partial: ComputePipeline,
    /// Split-K reduction decode for head_dim=256, f16 KV.
    decode_splitk_f16kv_hd256_reduce: ComputePipeline,
    /// sdpa_parallel decode for head_dim=128 — parallel reduction across simdgroups.
    decode_sdpa_parallel_hd128: ComputePipeline,
    /// sdpa_parallel decode for head_dim=256 — parallel reduction across simdgroups.
    decode_sdpa_parallel_hd256: ComputePipeline,
    /// sdpa_parallel decode for head_dim=128, f16 KV — parallel reduction across simdgroups.
    decode_sdpa_parallel_f16kv_hd128: ComputePipeline,
    /// sdpa_parallel decode for head_dim=256, f16 KV — parallel reduction across simdgroups.
    decode_sdpa_parallel_f16kv_hd256: ComputePipeline,
    /// Q8_0 KV decode attention (HD=128).
    decode_q8kv_hd128: ComputePipeline,
    /// Q8_0 KV decode attention (HD=256).
    decode_q8kv_hd256: ComputePipeline,
    /// Q8_0 KV cached prefill attention (generic HD).
    prefill_cache_q8kv: ComputePipeline,
    /// GQA-aware sdpa decode for head_dim=128, f16 KV.
    decode_sdpa_gqa_f16kv_hd128: ComputePipeline,
    /// GQA-aware sdpa decode for head_dim=256, f16 KV.
    decode_sdpa_gqa_f16kv_hd256: ComputePipeline,
}

impl AttentionKernels {
    /// Compile attention kernels from embedded Metal source.
    pub fn new(device: &MetalDevice) -> anyhow::Result<Self> {
        let prefill = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_f32",
        )
        .context("Failed to compile attention_prefill_f32 kernel")?;
        let prefill_hd256 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_f32_hd256",
        )
        .context("Failed to compile attention_prefill_f32_hd256 kernel")?;
        let prefill_v2 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_f32_v2",
        )
        .context("Failed to compile attention_prefill_f32_v2 kernel")?;
        let prefill_v2_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_f32_v2_hd128",
        )
        .context("Failed to compile attention_prefill_f32_v2_hd128 kernel")?;
        let prefill_ax_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_f32_ax_hd128",
        )
        .context("Failed to compile attention_prefill_f32_ax_hd128 kernel")?;
        let prefill_ax_f16out_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_f16_ax_hd128",
        )
        .context("Failed to compile attention_prefill_f16_ax_hd128 kernel")?;
        let prefill_ax_hd128_smem = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_f32_ax_hd128_smem",
        )
        .context("Failed to compile attention_prefill_f32_ax_hd128_smem kernel")?;
        let prefill_ax_hd128_smem_f16 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_f32_ax_hd128_smem_f16",
        )
        .context("Failed to compile attention_prefill_f32_ax_hd128_smem_f16 kernel")?;
        let prefill_ax_hd128_bc64 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_f32_ax_hd128_bc64",
        )
        .context("Failed to compile attention_prefill_f32_ax_hd128_bc64 kernel")?;
        let prefill_fa2_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_f32_fa2_hd128",
        )
        .context("Failed to compile attention_prefill_f32_fa2_hd128 kernel")?;
        let prefill_fa2_simd_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_fa2_simd_hd128",
        )
        .context("Failed to compile attention_prefill_fa2_simd_hd128 kernel")?;
        let prefill_fa2_half_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_fa2_half_hd128",
        )
        .context("Failed to compile attention_prefill_fa2_half_hd128 kernel")?;
        let prefill_fa2v2_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_fa2v2_hd128",
        )
        .context("Failed to compile attention_prefill_fa2v2_hd128 kernel")?;
        let prefill_fa2v2_hd256 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_fa2v2_hd256",
        )
        .context("Failed to compile attention_prefill_fa2v2_hd256 kernel")?;
        let prefill_fa2_simd_hd256 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_fa2_simd_hd256",
        )
        .context("Failed to compile attention_prefill_fa2_simd_hd256 kernel")?;
        let prefill_fa2_simd_hd64 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_fa2_simd_hd64",
        )
        .context("Failed to compile attention_prefill_fa2_simd_hd64 kernel")?;
        let prefill_cache = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_cache_f32",
        )
        .context("Failed to compile attention_prefill_cache_f32 kernel")?;
        let prefill_cache_f16kv = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_cache_f16kv",
        )
        .context("Failed to compile attention_prefill_cache_f16kv kernel")?;

        let decode = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_f32",
        )
        .context("Failed to compile attention_decode_f32 kernel")?;
        let decode_hd256 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_f32_hd256",
        )
        .context("Failed to compile attention_decode_f32_hd256 kernel")?;
        let decode_f16kv = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_f16kv",
        )
        .context("Failed to compile attention_decode_f16kv kernel")?;
        let decode_f16kv_hd256 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_f16kv_hd256",
        )
        .context("Failed to compile attention_decode_f16kv_hd256 kernel")?;
        let decode_v2 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_f32_v2",
        )
        .context("Failed to compile attention_decode_f32_v2 kernel")?;
        let decode_sdpa_hd256 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_sdpa_hd256",
        )
        .context("Failed to compile attention_decode_sdpa_hd256 kernel")?;
        let prefill_cache_fa2_hd256 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_cache_fa2_hd256",
        )
        .context("Failed to compile attention_prefill_cache_fa2_hd256 kernel")?;
        let prefill_cache_fa2_simd_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_fa2_simd_cached_f16kv_hd128",
        )
        .context("Failed to compile attention_prefill_fa2_simd_cached_f16kv_hd128 kernel")?;
        let prefill_cache_fa2_simd_hd64 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_fa2_simd_cached_f16kv_hd64",
        )
        .context("Failed to compile attention_prefill_fa2_simd_cached_f16kv_hd64 kernel")?;
        let prefill_cache_fa2_simd_hd256 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_fa2_simd_cached_f16kv_hd256",
        )
        .context("Failed to compile attention_prefill_fa2_simd_cached_f16kv_hd256 kernel")?;
        let decode_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_f32_hd128",
        )
        .context("Failed to compile attention_decode_f32_hd128 kernel")?;
        let decode_f16kv_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_f16kv_hd128",
        )
        .context("Failed to compile attention_decode_f16kv_hd128 kernel")?;
        let decode_f16kv_hd128_n2 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_f16kv_hd128_n2",
        )
        .context("Failed to compile attention_decode_f16kv_hd128_n2 kernel")?;
        let decode_splitk_f16kv_hd128_partial = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_splitk_f16kv_hd128_partial",
        )
        .context("Failed to compile attention_decode_splitk_f16kv_hd128_partial kernel")?;
        let decode_splitk_f16kv_hd128_reduce = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_splitk_f16kv_hd128_reduce",
        )
        .context("Failed to compile attention_decode_splitk_f16kv_hd128_reduce kernel")?;
        let decode_splitk_f16kv_hd256_partial = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_splitk_f16kv_hd256_partial",
        )
        .context("Failed to compile attention_decode_splitk_f16kv_hd256_partial kernel")?;
        let decode_splitk_f16kv_hd256_reduce = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_splitk_f16kv_hd256_reduce",
        )
        .context("Failed to compile attention_decode_splitk_f16kv_hd256_reduce kernel")?;
        let decode_sdpa_parallel_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_sdpa_parallel_hd128",
        )
        .context("Failed to compile attention_decode_sdpa_parallel_hd128 kernel")?;
        let decode_sdpa_parallel_hd256 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_sdpa_parallel_hd256",
        )
        .context("Failed to compile attention_decode_sdpa_parallel_hd256 kernel")?;
        let decode_sdpa_parallel_f16kv_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_sdpa_parallel_f16kv_hd128",
        )
        .context("Failed to compile attention_decode_sdpa_parallel_f16kv_hd128 kernel")?;
        let decode_sdpa_parallel_f16kv_hd256 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_sdpa_parallel_f16kv_hd256",
        )
        .context("Failed to compile attention_decode_sdpa_parallel_f16kv_hd256 kernel")?;
        let decode_q8kv_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_q8kv_hd128",
        )
        .context("Failed to compile attention_decode_q8kv_hd128 kernel")?;
        let prefill_cache_q8kv = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_cache_q8kv",
        )
        .context("Failed to compile attention_prefill_cache_q8kv kernel")?;
        let decode_q8kv_hd256 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_q8kv_hd256",
        )
        .context("Failed to compile attention_decode_q8kv_hd256 kernel")?;
        let decode_sdpa_gqa_f16kv_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_sdpa_gqa_f16kv_hd128",
        )
        .context("Failed to compile attention_decode_sdpa_gqa_f16kv_hd128 kernel")?;
        let decode_sdpa_gqa_f16kv_hd256 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_sdpa_gqa_f16kv_hd256",
        )
        .context("Failed to compile attention_decode_sdpa_gqa_f16kv_hd256 kernel")?;

        tracing::info!(
            prefill_max_threads = prefill.max_threads_per_threadgroup(),
            prefill_hd256_max_threads = prefill_hd256.max_threads_per_threadgroup(),
            prefill_v2_max_threads = prefill_v2.max_threads_per_threadgroup(),
            prefill_v2_hd128_max_threads = prefill_v2_hd128.max_threads_per_threadgroup(),
            prefill_ax_hd128_max_threads = prefill_ax_hd128.max_threads_per_threadgroup(),
            prefill_ax_f16out_hd128_max_threads =
                prefill_ax_f16out_hd128.max_threads_per_threadgroup(),
            prefill_ax_hd128_smem_max_threads = prefill_ax_hd128_smem.max_threads_per_threadgroup(),
            prefill_ax_hd128_smem_f16_max_threads =
                prefill_ax_hd128_smem_f16.max_threads_per_threadgroup(),
            prefill_ax_hd128_bc64_max_threads = prefill_ax_hd128_bc64.max_threads_per_threadgroup(),
            prefill_fa2_hd128_max_threads = prefill_fa2_hd128.max_threads_per_threadgroup(),
            prefill_fa2_simd_hd128_max_threads =
                prefill_fa2_simd_hd128.max_threads_per_threadgroup(),
            prefill_fa2_half_hd128_max_threads =
                prefill_fa2_half_hd128.max_threads_per_threadgroup(),
            prefill_fa2v2_hd128_max_threads = prefill_fa2v2_hd128.max_threads_per_threadgroup(),
            prefill_fa2v2_hd256_max_threads = prefill_fa2v2_hd256.max_threads_per_threadgroup(),
            prefill_fa2_simd_hd64_max_threads = prefill_fa2_simd_hd64.max_threads_per_threadgroup(),
            prefill_cache_max_threads = prefill_cache.max_threads_per_threadgroup(),
            prefill_cache_f16kv_max_threads = prefill_cache_f16kv.max_threads_per_threadgroup(),
            prefill_cache_fa2_hd256_max_threads =
                prefill_cache_fa2_hd256.max_threads_per_threadgroup(),
            decode_max_threads = decode.max_threads_per_threadgroup(),
            decode_hd256_max_threads = decode_hd256.max_threads_per_threadgroup(),
            decode_f16kv_max_threads = decode_f16kv.max_threads_per_threadgroup(),
            decode_f16kv_hd256_max_threads = decode_f16kv_hd256.max_threads_per_threadgroup(),
            decode_v2_max_threads = decode_v2.max_threads_per_threadgroup(),
            decode_sdpa_hd256_max_threads = decode_sdpa_hd256.max_threads_per_threadgroup(),
            decode_hd128_max_threads = decode_hd128.max_threads_per_threadgroup(),
            decode_f16kv_hd128_max_threads = decode_f16kv_hd128.max_threads_per_threadgroup(),
            decode_f16kv_hd128_n2_max_threads = decode_f16kv_hd128_n2.max_threads_per_threadgroup(),
            decode_splitk_f16kv_hd128_partial_max_threads =
                decode_splitk_f16kv_hd128_partial.max_threads_per_threadgroup(),
            decode_splitk_f16kv_hd128_reduce_max_threads =
                decode_splitk_f16kv_hd128_reduce.max_threads_per_threadgroup(),
            decode_splitk_f16kv_hd256_partial_max_threads =
                decode_splitk_f16kv_hd256_partial.max_threads_per_threadgroup(),
            decode_splitk_f16kv_hd256_reduce_max_threads =
                decode_splitk_f16kv_hd256_reduce.max_threads_per_threadgroup(),
            decode_sdpa_parallel_hd128_max_threads =
                decode_sdpa_parallel_hd128.max_threads_per_threadgroup(),
            decode_sdpa_parallel_hd256_max_threads =
                decode_sdpa_parallel_hd256.max_threads_per_threadgroup(),
            decode_sdpa_parallel_f16kv_hd128_max_threads =
                decode_sdpa_parallel_f16kv_hd128.max_threads_per_threadgroup(),
            decode_sdpa_parallel_f16kv_hd256_max_threads =
                decode_sdpa_parallel_f16kv_hd256.max_threads_per_threadgroup(),
            decode_sdpa_gqa_f16kv_hd128_max_threads =
                decode_sdpa_gqa_f16kv_hd128.max_threads_per_threadgroup(),
            decode_sdpa_gqa_f16kv_hd256_max_threads =
                decode_sdpa_gqa_f16kv_hd256.max_threads_per_threadgroup(),
            "Attention Metal kernels compiled (prefill + decode)",
        );

        Ok(Self {
            prefill,
            prefill_hd256,
            prefill_v2,
            prefill_v2_hd128,
            prefill_ax_hd128,
            prefill_ax_f16out_hd128,
            prefill_ax_hd128_smem,
            prefill_ax_hd128_smem_f16,
            prefill_ax_hd128_bc64,
            prefill_fa2_hd128,
            prefill_fa2_simd_hd128,
            prefill_fa2_half_hd128,
            prefill_fa2v2_hd128,
            prefill_fa2v2_hd256,
            prefill_fa2_simd_hd256,
            prefill_fa2_simd_hd64,
            prefill_cache,
            prefill_cache_f16kv,
            prefill_cache_fa2_hd256,
            prefill_cache_fa2_simd_hd128,
            prefill_cache_fa2_simd_hd64,
            prefill_cache_fa2_simd_hd256,
            decode,
            decode_hd256,
            decode_f16kv,
            decode_f16kv_hd256,
            decode_v2,
            decode_sdpa_hd256,
            decode_hd128,
            decode_f16kv_hd128,
            decode_f16kv_hd128_n2,
            decode_splitk_f16kv_hd128_partial,
            decode_splitk_f16kv_hd128_reduce,
            decode_splitk_f16kv_hd256_partial,
            decode_splitk_f16kv_hd256_reduce,
            decode_sdpa_parallel_hd128,
            decode_sdpa_parallel_hd256,
            decode_sdpa_parallel_f16kv_hd128,
            decode_sdpa_parallel_f16kv_hd256,
            decode_q8kv_hd128,
            decode_q8kv_hd256,
            prefill_cache_q8kv,
            decode_sdpa_gqa_f16kv_hd128,
            decode_sdpa_gqa_f16kv_hd256,
        })
    }

    /// Dispatch prefill attention with causal masking.
    ///
    /// Computes: O = softmax(Q × K^T / √head_dim) × V  (with causal mask)
    ///
    /// - `q`: [n_tokens × n_heads × head_dim] query vectors
    /// - `k`: [n_tokens × n_kv_heads × head_dim] key vectors
    /// - `v`: [n_tokens × n_kv_heads × head_dim] value vectors
    /// - `o`: [n_tokens × n_heads × head_dim] output buffer
    /// - `n_tokens`: number of tokens in the sequence
    /// - `n_heads`: number of query heads
    /// - `n_kv_heads`: number of KV heads (GQA: n_heads / n_kv_heads heads share one KV)
    /// - `head_dim`: dimension per head
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_with_config(
        &self,
        device: &MetalDevice,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        o: &MetalBuffer,
        n_tokens: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        config: AttentionDispatchConfig,
    ) -> anyhow::Result<()> {
        let selection = config.prefill_local_candidate_selection(n_tokens, head_dim);
        const FA2S_TG: usize = 128;
        let (pipeline, tg_width, groups_x) = match selection.candidate {
            AttentionPrefillCandidate::Fa2SimdHd128 => (
                &self.prefill_fa2_simd_hd128,
                FA2S_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::Fa2HalfHd128 => (
                &self.prefill_fa2_half_hd128,
                FA2S_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::Fa2v2Hd128 => (
                &self.prefill_fa2v2_hd128,
                FA2S_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::Fa2v2Hd256 => (
                &self.prefill_fa2v2_hd256,
                FA2S_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::Fa2SimdHd256 => (
                &self.prefill_fa2_simd_hd256,
                FA2S_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::Fa2SimdHd64 => (
                &self.prefill_fa2_simd_hd64,
                FA2S_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::Fa2Hd128 => (
                &self.prefill_fa2_hd128,
                ATTN_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::AxBc64 => (
                &self.prefill_ax_hd128_bc64,
                ATTN_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::AxSmemF16 => (
                &self.prefill_ax_hd128_smem_f16,
                ATTN_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::AxSmem => (
                &self.prefill_ax_hd128_smem,
                ATTN_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::AxHd128 => (
                &self.prefill_ax_hd128,
                ATTN_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::PrefillV2Hd128 => {
                (&self.prefill_v2_hd128, ATTN2_TG, n_tokens as usize)
            }
            AttentionPrefillCandidate::PrefillV2 => (&self.prefill_v2, ATTN2_TG, n_tokens as usize),
            AttentionPrefillCandidate::PrefillHd256 => {
                (&self.prefill_hd256, ATTN_TG, n_tokens as usize)
            }
            AttentionPrefillCandidate::Prefill => (&self.prefill, ATTN_TG, n_tokens as usize),
            AttentionPrefillCandidate::Cache
            | AttentionPrefillCandidate::CacheFa2Hd256
            | AttentionPrefillCandidate::CacheFa2SimdHd128
            | AttentionPrefillCandidate::CacheFa2SimdHd64
            | AttentionPrefillCandidate::CacheFa2SimdHd256 => {
                unreachable!("cached prefill candidates are not valid for local prefill")
            }
        };
        if attention_kernel_routing_log_enabled() {
            tracing::info!(
                n_tokens,
                n_heads,
                n_kv_heads,
                head_dim,
                profile = config.prefill_local_routing_profile_name(n_tokens),
                candidate = selection.label(),
                tier = selection.stability.label(),
                "attention_prefill kernel routing"
            );
        }
        device.execute_sync(|encoder| {
            crate::set_pipeline_cached(encoder, pipeline.state());
            // Bind buffers: Q=0, K=1, V=2, O=3
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(v.mtl_buffer()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(o.mtl_buffer()), 0, 3);
            }
            // Bind scalar parameters: indices 4-7
            bind_u32(encoder, 4, n_tokens);
            bind_u32(encoder, 5, n_heads);
            bind_u32(encoder, 6, n_kv_heads);
            bind_u32(encoder, 7, head_dim);
            // Grid: (n_tokens, n_heads) threadgroups, ATTN_TG threads each
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups_x,
                    height: n_heads as usize,
                    depth: 1,
                },
                MTLSize {
                    width: tg_width,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        })
    }

    /// Encode decode attention into an existing command encoder.
    ///
    /// Single-token attention: Q is one token, K/V are from the GPU KV cache.
    ///
    /// - `q`: [n_heads × head_dim] single query
    /// - `k_cache`: GPU KV cache K buffer for this layer
    /// - `v_cache`: GPU KV cache V buffer for this layer
    /// - `o`: [n_heads × head_dim] output buffer
    /// - `attend_start`: first token to attend to (sliding window offset)
    /// - `attend_len`: number of tokens to attend to
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_decode_with_config(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        o: &MetalBuffer,
        kv_f16: bool,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        attend_start: u32,
        attend_len: u32,
        config: AttentionDispatchConfig,
    ) {
        self.encode_attention_decode_with_stride_and_config(
            encoder,
            q,
            k_cache,
            v_cache,
            o,
            kv_f16,
            n_heads,
            n_kv_heads,
            head_dim,
            n_kv_heads * head_dim,
            attend_start,
            attend_len,
            config,
        );
    }

    /// Encode decode attention with an explicit KV row stride.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_decode_with_stride_and_config(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        o: &MetalBuffer,
        kv_f16: bool,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        kv_row_stride: u32,
        attend_start: u32,
        attend_len: u32,
        config: AttentionDispatchConfig,
    ) {
        assert!(
            head_dim <= MAX_HEAD_DIM,
            "head_dim {} exceeds MAX_HD ({}) in attention shader",
            head_dim,
            MAX_HEAD_DIM
        );
        let selection = attention_decode_candidate_selection(kv_f16, head_dim, attend_len, config);
        let (pipeline, tg_width, groups_x) = match selection.candidate {
            // sdpa_vector: MLX-pattern lane-parallel decode (HD=256, f16 KV).
            AttentionDecodeCandidate::SdpaHd256 => {
                (&self.decode_sdpa_hd256, ATTN_TG, n_heads as usize)
            }
            AttentionDecodeCandidate::F16KvHd256 => {
                (&self.decode_f16kv_hd256, ATTN_DEC_TG, n_heads as usize)
            }
            AttentionDecodeCandidate::F16KvHd128N2 => (
                &self.decode_f16kv_hd128_n2,
                ATTN_TG,
                (n_heads as usize).div_ceil(2),
            ),
            AttentionDecodeCandidate::F16KvHd128 => {
                (&self.decode_f16kv_hd128, ATTN_DEC2_TG, n_heads as usize)
            }
            AttentionDecodeCandidate::F16Kv => (&self.decode_f16kv, ATTN_DEC_TG, n_heads as usize),
            AttentionDecodeCandidate::Hd256 => (&self.decode_hd256, ATTN_DEC_TG, n_heads as usize),
            AttentionDecodeCandidate::Hd128 => (&self.decode_hd128, ATTN_DEC2_TG, n_heads as usize),
            AttentionDecodeCandidate::DecodeV2 => (&self.decode_v2, ATTN_DEC2_TG, n_heads as usize),
            AttentionDecodeCandidate::Decode => (&self.decode, ATTN_DEC_TG, n_heads as usize),
            AttentionDecodeCandidate::SplitKHd128 | AttentionDecodeCandidate::SplitKHd256 => {
                unreachable!("split-k attention candidate should use scratch path")
            }
            AttentionDecodeCandidate::SdpaParallelHd128
            | AttentionDecodeCandidate::SdpaParallelHd256 => {
                unreachable!("sdpa_parallel candidate should use dedicated path")
            }
            AttentionDecodeCandidate::SdpaGqaHd128 | AttentionDecodeCandidate::SdpaGqaHd256 => {
                unreachable!("sdpa_gqa candidate should use dedicated path")
            }
        };
        if attention_kernel_routing_log_enabled() {
            tracing::info!(
                n_heads,
                n_kv_heads,
                head_dim,
                attend_start,
                attend_len,
                profile = config.decode_routing_profile_name(attend_len),
                candidate = selection.label(),
                tier = selection.stability.label(),
                "attention_decode kernel routing"
            );
        }
        crate::set_pipeline_cached(encoder, pipeline.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k_cache.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v_cache.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(o.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n_heads);
        bind_u32(encoder, 5, n_kv_heads);
        bind_u32(encoder, 6, head_dim);
        bind_u32(encoder, 7, attend_start);
        bind_u32(encoder, 8, attend_len);
        bind_u32(encoder, 9, kv_row_stride);
        // Grid: n_heads threadgroups × ATTN_TG threads each
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_attention_decode_splitk_with_config(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        o: &MetalBuffer,
        partial_out: &MetalBuffer,
        partial_lse: &MetalBuffer,
        kv_f16: bool,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        attend_start: u32,
        attend_len: u32,
        config: AttentionDispatchConfig,
    ) {
        if !attention_decode_splitk_supported(kv_f16, head_dim) {
            self.encode_attention_decode_with_config(
                encoder,
                q,
                k_cache,
                v_cache,
                o,
                kv_f16,
                n_heads,
                n_kv_heads,
                head_dim,
                attend_start,
                attend_len,
                config,
            );
            return;
        }

        let chunk_size = attention_decode_splitk_chunk_size_with_config(config);
        let n_chunks = attend_len.div_ceil(chunk_size);
        let (partial_pipeline, reduce_pipeline, tg_width) = if head_dim == 256 {
            (
                &self.decode_splitk_f16kv_hd256_partial,
                &self.decode_splitk_f16kv_hd256_reduce,
                ATTN_TG,
            )
        } else {
            (
                &self.decode_splitk_f16kv_hd128_partial,
                &self.decode_splitk_f16kv_hd128_reduce,
                ATTN_DEC2_TG,
            )
        };

        crate::set_pipeline_cached(encoder, partial_pipeline.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k_cache.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v_cache.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(partial_out.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(partial_lse.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, n_heads);
        bind_u32(encoder, 6, n_kv_heads);
        bind_u32(encoder, 7, head_dim);
        bind_u32(encoder, 8, attend_start);
        bind_u32(encoder, 9, attend_len);
        bind_u32(encoder, 10, chunk_size);
        bind_u32(encoder, 11, n_chunks);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_chunks as usize,
                height: n_heads as usize,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );

        barrier_buffers(encoder);

        crate::set_pipeline_cached(encoder, reduce_pipeline.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(partial_out.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(partial_lse.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(o.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n_heads);
        bind_u32(encoder, 4, head_dim);
        bind_u32(encoder, 5, n_chunks);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_heads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode decode attention using caller-provided split-K scratch buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_decode_with_scratch_and_config(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        o: &MetalBuffer,
        partial_out: &MetalBuffer,
        partial_lse: &MetalBuffer,
        kv_f16: bool,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        attend_start: u32,
        attend_len: u32,
        config: AttentionDispatchConfig,
    ) {
        let selection = attention_decode_candidate_selection(kv_f16, head_dim, attend_len, config);
        let use_splitk = selection.candidate.is_splitk();
        if attention_kernel_routing_log_enabled() {
            tracing::info!(
                n_heads,
                n_kv_heads,
                head_dim,
                attend_start,
                attend_len,
                profile = config.decode_routing_profile_name(attend_len),
                candidate = selection.label(),
                tier = selection.stability.label(),
                "attention_decode_with_scratch kernel routing"
            );
        }
        // GQA-aware decode: share KV reads across query heads in same TG
        let use_sdpa_gqa = sdpa_gqa_decode_enabled() && kv_f16 && n_kv_heads < n_heads;
        if use_sdpa_gqa {
            self.encode_attention_decode_sdpa_gqa(
                encoder, q, k_cache, v_cache, o, n_heads, n_kv_heads, head_dim, attend_len,
            );
            return;
        }
        let use_sdpa_parallel = sdpa_parallel_decode_enabled() && matches!(head_dim, 128 | 256);
        if use_sdpa_parallel {
            self.encode_attention_decode_sdpa_parallel(
                encoder, q, k_cache, v_cache, o, kv_f16, n_heads, n_kv_heads, head_dim, attend_len,
            );
        } else if use_splitk {
            self.encode_attention_decode_splitk_with_config(
                encoder,
                q,
                k_cache,
                v_cache,
                o,
                partial_out,
                partial_lse,
                kv_f16,
                n_heads,
                n_kv_heads,
                head_dim,
                attend_start,
                attend_len,
                config,
            );
        } else {
            self.encode_attention_decode_with_config(
                encoder,
                q,
                k_cache,
                v_cache,
                o,
                kv_f16,
                n_heads,
                n_kv_heads,
                head_dim,
                attend_start,
                attend_len,
                config,
            );
        }
    }

    /// Encode sdpa_parallel decode attention (parallel reduction).
    ///
    /// Uses the `attention_decode_sdpa_parallel_[f16kv_]hd128` or `hd256` kernel
    /// depending on `head_dim` and `kv_f16`. Each threadgroup handles one head
    /// with 32 simdgroups (TG=1024) partitioning the KV sequence.
    ///
    /// - `q`: [n_heads × head_dim] query vector (single token)
    /// - `k`: [seq_len × n_kv_heads × head_dim] key cache
    /// - `v`: [seq_len × n_kv_heads × head_dim] value cache
    /// - `o`: [n_heads × head_dim] output buffer
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_decode_sdpa_parallel(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        o: &MetalBuffer,
        kv_f16: bool,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
    ) {
        let pipeline = match (kv_f16, head_dim) {
            (true, 256) => &self.decode_sdpa_parallel_f16kv_hd256,
            (true, _) => &self.decode_sdpa_parallel_f16kv_hd128,
            (false, 256) => &self.decode_sdpa_parallel_hd256,
            (false, _) => &self.decode_sdpa_parallel_hd128,
        };
        crate::set_pipeline_cached(encoder, pipeline.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(o.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n_heads);
        bind_u32(encoder, 5, n_kv_heads);
        bind_u32(encoder, 6, seq_len);
        // Grid: (1, n_heads, 1) — one threadgroup per head
        // TG: 1024 threads (32 simdgroups × 32 lanes)
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: n_heads as usize,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode GQA-aware sdpa decode attention.
    ///
    /// Uses `attention_decode_sdpa_gqa_f16kv_hd128` or `hd256` depending on
    /// `head_dim`. Each threadgroup handles one KV head and all its grouped
    /// query heads, sharing KV reads across the GQA ratio.
    ///
    /// - `q`: [n_heads × head_dim] query vector (single token)
    /// - `k`: [seq_len × n_kv_heads × head_dim] key cache (f16)
    /// - `v`: [seq_len × n_kv_heads × head_dim] value cache (f16)
    /// - `o`: [n_heads × head_dim] output buffer
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_decode_sdpa_gqa(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        o: &MetalBuffer,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
    ) {
        let pipeline = if head_dim == 256 {
            &self.decode_sdpa_gqa_f16kv_hd256
        } else {
            &self.decode_sdpa_gqa_f16kv_hd128
        };
        crate::set_pipeline_cached(encoder, pipeline.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(o.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n_heads);
        bind_u32(encoder, 5, n_kv_heads);
        bind_u32(encoder, 6, seq_len);
        // Grid: (1, n_kv_heads, 1) — one TG per KV head (not per query head)
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: n_kv_heads as usize,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode prefill attention into an existing command encoder.
    ///
    /// Does NOT create or commit a command buffer. Used for batching
    /// the prefill attention into a single command buffer with other ops.
    ///
    /// - `q`: [n_tokens × n_heads × head_dim] query vectors
    /// - `k`: [n_tokens × n_kv_heads × head_dim] key vectors
    /// - `v`: [n_tokens × n_kv_heads × head_dim] value vectors
    /// - `o`: [n_tokens × n_heads × head_dim] output buffer
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_prefill_with_config(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        o: &MetalBuffer,
        n_tokens: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        config: AttentionDispatchConfig,
    ) {
        let selection = config.prefill_local_candidate_selection(n_tokens, head_dim);
        if attention_kernel_routing_log_enabled() {
            tracing::info!(
                n_tokens,
                n_heads,
                n_kv_heads,
                head_dim,
                profile = config.prefill_local_routing_profile_name(n_tokens),
                candidate = selection.label(),
                tier = selection.stability.label(),
                "encode_attention_prefill kernel routing"
            );
        }
        // FA2 simd (half8x8 matrix ops) is the fastest path for HD=128.
        const FA2S_TG: usize = 128;
        let (pipeline, tg_width, groups_x) = match selection.candidate {
            AttentionPrefillCandidate::Fa2SimdHd128 => (
                &self.prefill_fa2_simd_hd128,
                FA2S_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::Fa2HalfHd128 => (
                &self.prefill_fa2_half_hd128,
                FA2S_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::Fa2v2Hd128 => (
                &self.prefill_fa2v2_hd128,
                FA2S_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::Fa2v2Hd256 => (
                &self.prefill_fa2v2_hd256,
                FA2S_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::Fa2SimdHd256 => (
                &self.prefill_fa2_simd_hd256,
                FA2S_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::Fa2SimdHd64 => (
                &self.prefill_fa2_simd_hd64,
                FA2S_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::Fa2Hd128 => (
                &self.prefill_fa2_hd128,
                ATTN_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::AxBc64 => (
                &self.prefill_ax_hd128_bc64,
                ATTN_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::AxSmemF16 => (
                &self.prefill_ax_hd128_smem_f16,
                ATTN_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::AxSmem => (
                &self.prefill_ax_hd128_smem,
                ATTN_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::AxHd128 => (
                &self.prefill_ax_hd128,
                ATTN_TG,
                (n_tokens as usize).div_ceil(8),
            ),
            AttentionPrefillCandidate::PrefillV2Hd128 => {
                (&self.prefill_v2_hd128, ATTN2_TG, n_tokens as usize)
            }
            AttentionPrefillCandidate::PrefillV2 => (&self.prefill_v2, ATTN2_TG, n_tokens as usize),
            AttentionPrefillCandidate::PrefillHd256 => {
                (&self.prefill_hd256, ATTN_TG, n_tokens as usize)
            }
            AttentionPrefillCandidate::Prefill => (&self.prefill, ATTN_TG, n_tokens as usize),
            AttentionPrefillCandidate::Cache
            | AttentionPrefillCandidate::CacheFa2Hd256
            | AttentionPrefillCandidate::CacheFa2SimdHd128
            | AttentionPrefillCandidate::CacheFa2SimdHd64
            | AttentionPrefillCandidate::CacheFa2SimdHd256 => {
                unreachable!("cached prefill candidates are not valid for local prefill")
            }
        };
        assert!(
            head_dim <= MAX_HEAD_DIM,
            "head_dim {} exceeds MAX_HD ({}) in attention shader",
            head_dim,
            MAX_HEAD_DIM
        );
        crate::set_pipeline_cached(encoder, pipeline.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(o.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n_tokens);
        bind_u32(encoder, 5, n_heads);
        bind_u32(encoder, 6, n_kv_heads);
        bind_u32(encoder, 7, head_dim);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: n_heads as usize,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode AX HD128 prefill attention with f16 output.
    ///
    /// Output buffer stores [n_tokens × n_heads × head_dim] in half precision.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_prefill_f16out_hd128(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        o_f16: &MetalBuffer,
        n_tokens: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
    ) {
        debug_assert_eq!(head_dim, 128);
        let pipeline = &self.prefill_ax_f16out_hd128;
        crate::set_pipeline_cached(encoder, pipeline.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(o_f16.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n_tokens);
        bind_u32(encoder, 5, n_heads);
        bind_u32(encoder, 6, n_kv_heads);
        bind_u32(encoder, 7, head_dim);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n_tokens as usize).div_ceil(8),
                height: n_heads as usize,
                depth: 1,
            },
            MTLSize {
                width: ATTN_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode batched prefill attention against existing KV cache.
    ///
    /// - `q`: [n_tokens × n_heads × head_dim] query vectors for suffix tokens
    /// - `k_cache`/`v_cache`: KV cache buffers containing restored prefix + appended suffix
    /// - `base_seq_len`: prefix length already present in KV cache before current suffix
    /// - `sliding_window`: 0 disables sliding window, otherwise per-query window size
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_prefill_cached_with_config(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        o: &MetalBuffer,
        kv_f16: bool,
        n_tokens: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        base_seq_len: u32,
        sliding_window: u32,
        config: AttentionDispatchConfig,
    ) {
        self.encode_attention_prefill_cached_with_stride_and_config(
            encoder,
            q,
            k_cache,
            v_cache,
            o,
            kv_f16,
            n_tokens,
            n_heads,
            n_kv_heads,
            head_dim,
            n_kv_heads * head_dim,
            base_seq_len,
            sliding_window,
            config,
        );
    }

    /// Encode cached prefill attention with an explicit KV row stride.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_prefill_cached_with_stride_and_config(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        o: &MetalBuffer,
        kv_f16: bool,
        n_tokens: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        kv_row_stride: u32,
        base_seq_len: u32,
        sliding_window: u32,
        config: AttentionDispatchConfig,
    ) {
        assert!(
            head_dim <= MAX_HEAD_DIM,
            "head_dim {} exceeds MAX_HD ({}) in attention shader",
            head_dim,
            MAX_HEAD_DIM
        );
        let selection = config.prefill_cached_candidate_selection(
            kv_f16,
            n_tokens,
            head_dim,
            base_seq_len,
            sliding_window,
        );
        if attention_kernel_routing_log_enabled() {
            tracing::info!(
                kv_f16,
                n_tokens,
                n_heads,
                n_kv_heads,
                head_dim,
                base_seq_len,
                sliding_window,
                profile = config.prefill_cached_routing_profile_name(
                    n_tokens,
                    base_seq_len,
                    sliding_window,
                ),
                candidate = selection.label(),
                tier = selection.stability.label(),
                mode = ?attention_prefill_fa2_mode_with_config(config),
                "encode_attention_prefill_cached kernel routing"
            );
        }
        // FA2 multi-query kernels: grid is (ceil(n_tokens/8), n_heads).
        let fa2_cached_pipeline = match selection.candidate {
            AttentionPrefillCandidate::CacheFa2SimdHd128 => {
                Some((&self.prefill_cache_fa2_simd_hd128, 128usize))
            }
            AttentionPrefillCandidate::CacheFa2SimdHd64 => {
                Some((&self.prefill_cache_fa2_simd_hd64, 128usize))
            }
            AttentionPrefillCandidate::CacheFa2SimdHd256 => {
                Some((&self.prefill_cache_fa2_simd_hd256, 128usize))
            }
            AttentionPrefillCandidate::CacheFa2Hd256 => {
                Some((&self.prefill_cache_fa2_hd256, 256usize))
            }
            _ => None,
        };
        if let Some((pipeline, tg_size)) = fa2_cached_pipeline {
            const FA2_Q: usize = 8;
            let n_tile_q = (n_tokens as usize).div_ceil(FA2_Q);
            crate::set_pipeline_cached(encoder, pipeline.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(k_cache.mtl_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(v_cache.mtl_buffer()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(o.mtl_buffer()), 0, 3);
            }
            bind_u32(encoder, 4, n_tokens);
            bind_u32(encoder, 5, n_heads);
            bind_u32(encoder, 6, n_kv_heads);
            bind_u32(encoder, 7, head_dim);
            bind_u32(encoder, 8, base_seq_len);
            bind_u32(encoder, 9, sliding_window);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: n_tile_q,
                    height: n_heads as usize,
                    depth: 1,
                },
                MTLSize {
                    width: tg_size,
                    height: 1,
                    depth: 1,
                },
            );
            return;
        }
        let pipeline = if kv_f16 {
            &self.prefill_cache_f16kv
        } else {
            &self.prefill_cache
        };
        crate::set_pipeline_cached(encoder, pipeline.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k_cache.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v_cache.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(o.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n_tokens);
        bind_u32(encoder, 5, n_heads);
        bind_u32(encoder, 6, n_kv_heads);
        bind_u32(encoder, 7, head_dim);
        bind_u32(encoder, 8, base_seq_len);
        bind_u32(encoder, 9, sliding_window);
        bind_u32(encoder, 10, kv_row_stride);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_tokens as usize,
                height: n_heads as usize,
                depth: 1,
            },
            MTLSize {
                width: ATTN_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Dispatch Q8_0 KV cached prefill attention.
    ///
    /// Same interface as `encode_attention_prefill_cached_with_config` but reads
    /// from Q8_0 quantized KV cache with inline dequant.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_prefill_cached_q8kv(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        o: &MetalBuffer,
        n_tokens: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        base_seq_len: u32,
        sliding_window: u32,
    ) {
        crate::set_pipeline_cached(encoder, self.prefill_cache_q8kv.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k_cache.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v_cache.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(o.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n_tokens);
        bind_u32(encoder, 5, n_heads);
        bind_u32(encoder, 6, n_kv_heads);
        bind_u32(encoder, 7, head_dim);
        bind_u32(encoder, 8, base_seq_len);
        bind_u32(encoder, 9, sliding_window);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_tokens as usize,
                height: n_heads as usize,
                depth: 1,
            },
            MTLSize {
                width: ATTN_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Dispatch Q8_0 KV decode attention (HD=128).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_decode_q8kv(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        o: &MetalBuffer,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        attend_start: u32,
        attend_len: u32,
    ) {
        crate::set_pipeline_cached(encoder, self.decode_q8kv_hd128.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k_cache.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v_cache.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(o.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n_heads);
        bind_u32(encoder, 5, n_kv_heads);
        bind_u32(encoder, 6, head_dim);
        bind_u32(encoder, 7, attend_start);
        bind_u32(encoder, 8, attend_len);
        const ATTN_DEC_Q8_TG: usize = 128;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_heads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ATTN_DEC_Q8_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Dispatch Q8_0 KV decode attention (HD=256).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_decode_q8kv_hd256(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        o: &MetalBuffer,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        attend_start: u32,
        attend_len: u32,
    ) {
        crate::set_pipeline_cached(encoder, self.decode_q8kv_hd256.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k_cache.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v_cache.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(o.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n_heads);
        bind_u32(encoder, 5, n_kv_heads);
        bind_u32(encoder, 6, head_dim);
        bind_u32(encoder, 7, attend_start);
        bind_u32(encoder, 8, attend_len);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_heads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ATTN_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_with_config(
        &self,
        device: &MetalDevice,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        o: &MetalBuffer,
        kv_f16: bool,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        attend_start: u32,
        attend_len: u32,
        config: AttentionDispatchConfig,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_attention_decode_with_config(
                encoder,
                q,
                k_cache,
                v_cache,
                o,
                kv_f16,
                n_heads,
                n_kv_heads,
                head_dim,
                attend_start,
                attend_len,
                config,
            );
            Ok(())
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_with_stride_and_config(
        &self,
        device: &MetalDevice,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        o: &MetalBuffer,
        kv_f16: bool,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        kv_row_stride: u32,
        attend_start: u32,
        attend_len: u32,
        config: AttentionDispatchConfig,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_attention_decode_with_stride_and_config(
                encoder,
                q,
                k_cache,
                v_cache,
                o,
                kv_f16,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_row_stride,
                attend_start,
                attend_len,
                config,
            );
            Ok(())
        })
    }

    /// Dispatch decode attention (standalone, creates own command buffer).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_splitk_with_config(
        &self,
        device: &MetalDevice,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        o: &MetalBuffer,
        kv_f16: bool,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        attend_start: u32,
        attend_len: u32,
        config: AttentionDispatchConfig,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            attention_decode_splitk_supported(kv_f16, head_dim),
            "split-K decode only supports f16 KV with head_dim 128 or 256"
        );
        let chunk_size = attention_decode_splitk_chunk_size_with_config(config);
        let n_chunks = attend_len.div_ceil(chunk_size) as usize;
        let partial_out = MetalBuffer::new(
            device.device(),
            n_heads as usize * n_chunks * head_dim as usize * std::mem::size_of::<f32>(),
        )?;
        let partial_lse = MetalBuffer::new(
            device.device(),
            n_heads as usize * n_chunks * std::mem::size_of::<f32>(),
        )?;
        device.execute_sync(|encoder| {
            self.encode_attention_decode_splitk_with_config(
                encoder,
                q,
                k_cache,
                v_cache,
                o,
                &partial_out,
                &partial_lse,
                kv_f16,
                n_heads,
                n_kv_heads,
                head_dim,
                attend_start,
                attend_len,
                config,
            );
            Ok(())
        })
    }
}
