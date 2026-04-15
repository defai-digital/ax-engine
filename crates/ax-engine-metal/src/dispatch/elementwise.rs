use super::*;

/// Threadgroup size for elementwise kernels (must match shader constant).
const ELEMENTWISE_TG_SIZE: usize = 256;

fn moe_weighted_reduce_slots8_vec4_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(
        || match std::env::var("AX_METAL_MOE_WEIGHTED_REDUCE_SLOTS8_VEC4") {
            Ok(v) => {
                let v = v.trim();
                !(v == "0"
                    || v.eq_ignore_ascii_case("false")
                    || v.eq_ignore_ascii_case("off")
                    || v.eq_ignore_ascii_case("no"))
            }
            Err(_) => true,
        },
    )
}

pub struct ElementwiseKernels {
    rms_norm: ComputePipeline,
    rms_norm_batch: ComputePipeline,
    rms_norm_out: ComputePipeline,
    rms_norm_out_batch: ComputePipeline,
    rms_norm_out_batch_vec4: ComputePipeline,
    rms_norm_out_batch_f16: ComputePipeline,
    residual_add_rms_norm_out_batch: ComputePipeline,
    residual_add_rms_norm_out_batch_f16: ComputePipeline,
    post_attn_norm_residual_add_rms_norm_out_batch: ComputePipeline,
    post_ffn_norm_residual_add_rms_norm_out_batch: ComputePipeline,
    rope: ComputePipeline,
    rope_batch: ComputePipeline,
    rope_neox_partial_batch: ComputePipeline,
    rope_neox_partial_freq_factors_batch: ComputePipeline,
    per_head_rms_norm: ComputePipeline,
    per_head_rms_norm_batch: ComputePipeline,
    per_head_rms_norm_silu_mul_batch: ComputePipeline,
    per_head_rms_norm_no_weight: ComputePipeline,
    per_head_rms_norm_no_weight_batch: ComputePipeline,
    qk_norm_rope_batch: ComputePipeline,
    qkv_split_qk_norm_rope_append_kv_batch_f32: ComputePipeline,
    qkv_split_qk_norm_rope_append_kv_batch_f16: ComputePipeline,
    /// Fused: QKV split + bias + QK norm + RoPE + KV append (Qwen3 with QKV bias).
    qkv_split_bias_qknorm_rope_append_kv_batch_f32: ComputePipeline,
    /// Same, with f16 KV cache (the DEFAULT path since max_seq_len ≥ 256 enables f16 KV).
    qkv_split_bias_qknorm_rope_append_kv_batch_f16: ComputePipeline,
    gelu_elementwise_mul: ComputePipeline,
    gelu_elementwise_mul_batch: ComputePipeline,
    gelu_elementwise_mul_batch_vec4: ComputePipeline,
    gelu_split_mul_batch: ComputePipeline,
    gelu_inplace: ComputePipeline,
    gelu_inplace_batch: ComputePipeline,
    silu_elementwise_mul: ComputePipeline,
    silu_elementwise_mul_batch: ComputePipeline,
    silu_elementwise_mul_batch_vec4: ComputePipeline,
    silu_elementwise_mul_batch_f16: ComputePipeline,
    sigmoid_elementwise_mul: ComputePipeline,
    sigmoid_scalar_mul_inplace: ComputePipeline,
    dense_row_dot_sigmoid_mul_inplace: ComputePipeline,
    sigmoid_inplace: ComputePipeline,
    sigmoid_inplace_vec4: ComputePipeline,
    softplus_bias_mul: ComputePipeline,
    softplus_bias_mul_sigmoid_pair: ComputePipeline,
    softplus_bias_mul_batch: ComputePipeline,
    softplus_bias_mul_sigmoid_pair_batch: ComputePipeline,
    l2_norm_per_head: ComputePipeline,
    elementwise_add: ComputePipeline,
    elementwise_add_batch: ComputePipeline,
    elementwise_add_batch_vec4: ComputePipeline,
    elementwise_weighted_add: ComputePipeline,
    cast_f32_to_f16: ComputePipeline,
    cast_f32_to_f16_vec4: ComputePipeline,
    cast_f16_to_f32: ComputePipeline,
    cast_f16_to_f32_vec4: ComputePipeline,
    qkv_split_batch: ComputePipeline,
    split_qgate_batch: ComputePipeline,
    qkv_split_rope_batch: ComputePipeline,
    qkv_split_rope_append_kv_batch_f32: ComputePipeline,
    qkv_split_rope_append_kv_batch_f16: ComputePipeline,
    kv_append_f32: ComputePipeline,
    kv_append_batch_f32: ComputePipeline,
    kv_append_f16: ComputePipeline,
    kv_append_batch_f16: ComputePipeline,
    kv_append_batch2_f32: ComputePipeline,
    kv_append_batch2_f16: ComputePipeline,
    kv_append_batch_q8_0: ComputePipeline,
    kv_append_batch2_q8_0: ComputePipeline,
    // General-purpose elementwise ops for GDN chunked graph.
    gen_exp: ComputePipeline,
    gen_mul: ComputePipeline,
    gen_add: ComputePipeline,
    gen_sub: ComputePipeline,
    gen_neg: ComputePipeline,
    gen_scale: ComputePipeline,
    // MoE expert dispatch
    pub moe_gather_rows: ComputePipeline,
    pub moe_weighted_scatter_add: ComputePipeline,
    pub moe_weighted_reduce_slots_add: ComputePipeline,
    pub moe_weighted_reduce_slots8_add_vec4: ComputePipeline,
    pub moe_softmax_topk: ComputePipeline,
    pub moe_apply_expert_scales: ComputePipeline,
    // GPU-side argmax for greedy decode
    argmax_f32: ComputePipeline,
}

impl ElementwiseKernels {
    /// Compile all elementwise kernels from embedded Metal source.
    pub fn new(device: &MetalDevice) -> anyhow::Result<Self> {
        let rms_norm =
            ComputePipeline::from_source(device.device(), ELEMENTWISE_SHADER_SRC, "rms_norm_f32")
                .context("Failed to compile rms_norm_f32 kernel")?;
        let rms_norm_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "rms_norm_batch_f32",
        )
        .context("Failed to compile rms_norm_batch_f32 kernel")?;

        let rms_norm_out = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "rms_norm_out_f32",
        )
        .context("Failed to compile rms_norm_out_f32 kernel")?;
        let rms_norm_out_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "rms_norm_out_batch_f32",
        )
        .context("Failed to compile rms_norm_out_batch_f32 kernel")?;
        let rms_norm_out_batch_vec4 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "rms_norm_out_batch_f32_vec4",
        )
        .context("Failed to compile rms_norm_out_batch_f32_vec4 kernel")?;
        let rms_norm_out_batch_f16 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "rms_norm_out_batch_f16",
        )
        .context("Failed to compile rms_norm_out_batch_f16 kernel")?;
        let residual_add_rms_norm_out_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "residual_add_rms_norm_out_batch_f32",
        )
        .context("Failed to compile residual_add_rms_norm_out_batch_f32 kernel")?;
        let residual_add_rms_norm_out_batch_f16 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "residual_add_rms_norm_out_batch_f16",
        )
        .context("Failed to compile residual_add_rms_norm_out_batch_f16 kernel")?;
        let post_attn_norm_residual_add_rms_norm_out_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "post_attn_norm_residual_add_rms_norm_out_batch_f32",
        )
        .context("Failed to compile post_attn_norm_residual_add_rms_norm_out_batch_f32 kernel")?;
        let post_ffn_norm_residual_add_rms_norm_out_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "post_ffn_norm_residual_add_rms_norm_out_batch_f32",
        )
        .context("Failed to compile post_ffn_norm_residual_add_rms_norm_out_batch_f32 kernel")?;

        let rope = ComputePipeline::from_source_with_constants(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "rope_f32_generic",
            &[FunctionConstant {
                index: 0,
                value: FunctionConstantValue::Bool(false),
            }],
        )
        .context("Failed to compile rope_f32 generic single-row kernel")?;
        let rope_batch = ComputePipeline::from_source_with_constants(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "rope_f32_generic",
            &[FunctionConstant {
                index: 0,
                value: FunctionConstantValue::Bool(true),
            }],
        )
        .context("Failed to compile rope_f32 generic batched kernel")?;
        let rope_neox_partial_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "rope_neox_partial_batch_f32",
        )
        .context("Failed to compile rope_neox_partial_batch_f32 kernel")?;
        let rope_neox_partial_freq_factors_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "rope_neox_partial_freq_factors_batch_f32",
        )
        .context("Failed to compile rope_neox_partial_freq_factors_batch_f32 kernel")?;

        let per_head_rms_norm = ComputePipeline::from_source_with_constants(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "per_head_rms_norm_f32_generic",
            &[FunctionConstant {
                index: 1,
                value: FunctionConstantValue::Bool(false),
            }],
        )
        .context("Failed to compile per_head_rms_norm_f32 generic single-row kernel")?;
        let per_head_rms_norm_batch = ComputePipeline::from_source_with_constants(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "per_head_rms_norm_f32_generic",
            &[FunctionConstant {
                index: 1,
                value: FunctionConstantValue::Bool(true),
            }],
        )
        .context("Failed to compile per_head_rms_norm_f32 generic batched kernel")?;
        let per_head_rms_norm_silu_mul_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "per_head_rms_norm_silu_mul_batch_f32",
        )
        .context("Failed to compile per_head_rms_norm_silu_mul_batch_f32 kernel")?;
        let per_head_rms_norm_no_weight = ComputePipeline::from_source_with_constants(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "per_head_rms_norm_no_weight_f32_generic",
            &[FunctionConstant {
                index: 1,
                value: FunctionConstantValue::Bool(false),
            }],
        )
        .context("Failed to compile per_head_rms_norm_no_weight_f32 generic single-row kernel")?;
        let per_head_rms_norm_no_weight_batch = ComputePipeline::from_source_with_constants(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "per_head_rms_norm_no_weight_f32_generic",
            &[FunctionConstant {
                index: 1,
                value: FunctionConstantValue::Bool(true),
            }],
        )
        .context("Failed to compile per_head_rms_norm_no_weight_f32 generic batched kernel")?;
        let qk_norm_rope_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "qk_norm_rope_batch_f32",
        )
        .context("Failed to compile qk_norm_rope_batch_f32 kernel")?;
        let qkv_split_qk_norm_rope_append_kv_batch_f32 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "qkv_split_qk_norm_rope_append_kv_batch_f32",
        )
        .context("Failed to compile qkv_split_qk_norm_rope_append_kv_batch_f32 kernel")?;
        let qkv_split_qk_norm_rope_append_kv_batch_f16 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "qkv_split_qk_norm_rope_append_kv_batch_f16",
        )
        .context("Failed to compile qkv_split_qk_norm_rope_append_kv_batch_f16 kernel")?;
        let qkv_split_bias_qknorm_rope_append_kv_batch_f32 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "qkv_split_bias_qknorm_rope_append_kv_batch_f32",
        )
        .context("Failed to compile qkv_split_bias_qknorm_rope_append_kv_batch_f32 kernel")?;
        let qkv_split_bias_qknorm_rope_append_kv_batch_f16 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "qkv_split_bias_qknorm_rope_append_kv_batch_f16",
        )
        .context("Failed to compile qkv_split_bias_qknorm_rope_append_kv_batch_f16 kernel")?;

        let gelu_elementwise_mul = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "gelu_elementwise_mul_f32",
        )
        .context("Failed to compile gelu_elementwise_mul_f32 kernel")?;
        let gelu_elementwise_mul_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "gelu_elementwise_mul_batch_f32",
        )
        .context("Failed to compile gelu_elementwise_mul_batch_f32 kernel")?;
        let gelu_elementwise_mul_batch_vec4 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "gelu_elementwise_mul_batch_f32_vec4",
        )
        .context("Failed to compile gelu_elementwise_mul_batch_f32_vec4 kernel")?;
        let gelu_split_mul_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "gelu_split_mul_batch_f32",
        )
        .context("Failed to compile gelu_split_mul_batch_f32 kernel")?;
        let gelu_inplace = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "gelu_inplace_f32",
        )
        .context("Failed to compile gelu_inplace_f32 kernel")?;
        let gelu_inplace_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "gelu_inplace_batch_f32",
        )
        .context("Failed to compile gelu_inplace_batch_f32 kernel")?;

        let silu_elementwise_mul = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "silu_elementwise_mul_f32",
        )
        .context("Failed to compile silu_elementwise_mul_f32 kernel")?;
        let silu_elementwise_mul_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "silu_elementwise_mul_batch_f32",
        )
        .context("Failed to compile silu_elementwise_mul_batch_f32 kernel")?;
        let silu_elementwise_mul_batch_vec4 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "silu_elementwise_mul_batch_f32_vec4",
        )
        .context("Failed to compile silu_elementwise_mul_batch_f32_vec4 kernel")?;
        let silu_elementwise_mul_batch_f16 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "silu_elementwise_mul_batch_f16",
        )
        .context("Failed to compile silu_elementwise_mul_batch_f16 kernel")?;
        let sigmoid_elementwise_mul = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "sigmoid_elementwise_mul_f32",
        )
        .context("Failed to compile sigmoid_elementwise_mul_f32 kernel")?;
        let sigmoid_scalar_mul_inplace = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "sigmoid_scalar_mul_inplace_f32",
        )
        .context("Failed to compile sigmoid_scalar_mul_inplace_f32 kernel")?;
        let dense_row_dot_sigmoid_mul_inplace = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "dense_row_dot_sigmoid_mul_inplace_f32",
        )
        .context("Failed to compile dense_row_dot_sigmoid_mul_inplace_f32 kernel")?;
        let sigmoid_inplace = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "sigmoid_inplace_f32",
        )
        .context("Failed to compile sigmoid_inplace_f32 kernel")?;
        let sigmoid_inplace_vec4 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "sigmoid_inplace_f32_vec4",
        )
        .context("Failed to compile sigmoid_inplace_f32_vec4 kernel")?;
        let softplus_bias_mul = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "softplus_bias_mul_f32",
        )
        .context("Failed to compile softplus_bias_mul_f32 kernel")?;
        let softplus_bias_mul_sigmoid_pair = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "softplus_bias_mul_sigmoid_pair_f32",
        )
        .context("Failed to compile softplus_bias_mul_sigmoid_pair_f32 kernel")?;
        let softplus_bias_mul_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "softplus_bias_mul_batch_f32",
        )
        .context("Failed to compile softplus_bias_mul_batch_f32 kernel")?;
        let softplus_bias_mul_sigmoid_pair_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "softplus_bias_mul_sigmoid_pair_batch_f32",
        )
        .context("Failed to compile softplus_bias_mul_sigmoid_pair_batch_f32 kernel")?;
        let l2_norm_per_head = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "l2_norm_per_head_f32",
        )
        .context("Failed to compile l2_norm_per_head_f32 kernel")?;

        let elementwise_add = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "elementwise_add_f32",
        )
        .context("Failed to compile elementwise_add_f32 kernel")?;
        let elementwise_add_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "elementwise_add_batch_f32",
        )
        .context("Failed to compile elementwise_add_batch_f32 kernel")?;
        let elementwise_add_batch_vec4 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "elementwise_add_batch_f32_vec4",
        )
        .context("Failed to compile elementwise_add_batch_f32_vec4 kernel")?;
        let elementwise_weighted_add = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "elementwise_weighted_add_f32",
        )
        .context("Failed to compile elementwise_weighted_add_f32 kernel")?;
        let cast_f32_to_f16 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "cast_f32_to_f16",
        )
        .context("Failed to compile cast_f32_to_f16 kernel")?;
        let cast_f16_to_f32 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "cast_f16_to_f32",
        )
        .context("Failed to compile cast_f16_to_f32 kernel")?;
        let cast_f32_to_f16_vec4 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "cast_f32_to_f16_vec4",
        )
        .context("Failed to compile cast_f32_to_f16_vec4 kernel")?;
        let cast_f16_to_f32_vec4 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "cast_f16_to_f32_vec4",
        )
        .context("Failed to compile cast_f16_to_f32_vec4 kernel")?;
        let qkv_split_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "qkv_split_batch_f32",
        )
        .context("Failed to compile qkv_split_batch_f32 kernel")?;
        let split_qgate_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "split_qgate_batch_f32",
        )
        .context("Failed to compile split_qgate_batch_f32 kernel")?;
        let qkv_split_rope_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "qkv_split_rope_batch_f32",
        )
        .context("Failed to compile qkv_split_rope_batch_f32 kernel")?;
        let qkv_split_rope_append_kv_batch_f32 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "qkv_split_rope_append_kv_batch_f32",
        )
        .context("Failed to compile qkv_split_rope_append_kv_batch_f32 kernel")?;
        let qkv_split_rope_append_kv_batch_f16 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "qkv_split_rope_append_kv_batch_f16",
        )
        .context("Failed to compile qkv_split_rope_append_kv_batch_f16 kernel")?;

        let kv_append_f32 =
            ComputePipeline::from_source(device.device(), ELEMENTWISE_SHADER_SRC, "kv_append_f32")
                .context("Failed to compile kv_append_f32 kernel")?;
        let kv_append_batch_f32 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "kv_append_batch_f32",
        )
        .context("Failed to compile kv_append_batch_f32 kernel")?;
        let kv_append_f16 =
            ComputePipeline::from_source(device.device(), ELEMENTWISE_SHADER_SRC, "kv_append_f16")
                .context("Failed to compile kv_append_f16 kernel")?;
        let kv_append_batch_f16 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "kv_append_batch_f16",
        )
        .context("Failed to compile kv_append_batch_f16 kernel")?;
        let kv_append_batch2_f32 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "kv_append_batch2_f32",
        )
        .context("Failed to compile kv_append_batch2_f32 kernel")?;
        let kv_append_batch2_f16 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "kv_append_batch2_f16",
        )
        .context("Failed to compile kv_append_batch2_f16 kernel")?;
        let kv_append_batch_q8_0 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "kv_append_batch_q8_0",
        )
        .context("Failed to compile kv_append_batch_q8_0 kernel")?;
        let kv_append_batch2_q8_0 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "kv_append_batch2_q8_0",
        )
        .context("Failed to compile kv_append_batch2_q8_0 kernel")?;

        let argmax_f32 =
            ComputePipeline::from_source(device.device(), ELEMENTWISE_SHADER_SRC, "argmax_f32")
                .context("Failed to compile argmax_f32 kernel")?;

        tracing::info!("Elementwise Metal kernels compiled (8 kernels)");

        Ok(Self {
            rms_norm,
            rms_norm_batch,
            rms_norm_out,
            rms_norm_out_batch,
            rms_norm_out_batch_vec4,
            rms_norm_out_batch_f16,
            residual_add_rms_norm_out_batch,
            residual_add_rms_norm_out_batch_f16,
            post_attn_norm_residual_add_rms_norm_out_batch,
            post_ffn_norm_residual_add_rms_norm_out_batch,
            rope,
            rope_batch,
            rope_neox_partial_batch,
            rope_neox_partial_freq_factors_batch,
            per_head_rms_norm,
            per_head_rms_norm_batch,
            per_head_rms_norm_silu_mul_batch,
            per_head_rms_norm_no_weight,
            per_head_rms_norm_no_weight_batch,
            qk_norm_rope_batch,
            qkv_split_qk_norm_rope_append_kv_batch_f32,
            qkv_split_qk_norm_rope_append_kv_batch_f16,
            qkv_split_bias_qknorm_rope_append_kv_batch_f32,
            qkv_split_bias_qknorm_rope_append_kv_batch_f16,
            gelu_elementwise_mul,
            gelu_elementwise_mul_batch,
            gelu_elementwise_mul_batch_vec4,
            gelu_split_mul_batch,
            gelu_inplace,
            gelu_inplace_batch,
            silu_elementwise_mul,
            silu_elementwise_mul_batch,
            silu_elementwise_mul_batch_vec4,
            silu_elementwise_mul_batch_f16,
            sigmoid_elementwise_mul,
            sigmoid_scalar_mul_inplace,
            dense_row_dot_sigmoid_mul_inplace,
            sigmoid_inplace,
            sigmoid_inplace_vec4,
            softplus_bias_mul,
            softplus_bias_mul_sigmoid_pair,
            softplus_bias_mul_batch,
            softplus_bias_mul_sigmoid_pair_batch,
            l2_norm_per_head,
            elementwise_add,
            elementwise_add_batch,
            elementwise_add_batch_vec4,
            elementwise_weighted_add,
            cast_f32_to_f16,
            cast_f32_to_f16_vec4,
            cast_f16_to_f32,
            cast_f16_to_f32_vec4,
            qkv_split_batch,
            split_qgate_batch,
            qkv_split_rope_batch,
            qkv_split_rope_append_kv_batch_f32,
            qkv_split_rope_append_kv_batch_f16,
            kv_append_f32,
            kv_append_batch_f32,
            kv_append_f16,
            kv_append_batch_f16,
            kv_append_batch2_f32,
            kv_append_batch2_f16,
            kv_append_batch_q8_0,
            kv_append_batch2_q8_0,
            gen_exp: ComputePipeline::from_source(
                device.device(),
                ELEMENTWISE_SHADER_SRC,
                "elementwise_exp_f32",
            )
            .context("Failed to compile elementwise_exp_f32")?,
            gen_mul: ComputePipeline::from_source(
                device.device(),
                ELEMENTWISE_SHADER_SRC,
                "elementwise_mul_f32",
            )
            .context("Failed to compile elementwise_mul_f32")?,
            gen_add: ComputePipeline::from_source(
                device.device(),
                ELEMENTWISE_SHADER_SRC,
                "elementwise_add_out_f32",
            )
            .context("Failed to compile elementwise_add_out_f32")?,
            gen_sub: ComputePipeline::from_source(
                device.device(),
                ELEMENTWISE_SHADER_SRC,
                "elementwise_sub_f32",
            )
            .context("Failed to compile elementwise_sub_f32")?,
            gen_neg: ComputePipeline::from_source(
                device.device(),
                ELEMENTWISE_SHADER_SRC,
                "elementwise_neg_f32",
            )
            .context("Failed to compile elementwise_neg_f32")?,
            gen_scale: ComputePipeline::from_source(
                device.device(),
                ELEMENTWISE_SHADER_SRC,
                "elementwise_scale_f32",
            )
            .context("Failed to compile elementwise_scale_f32")?,
            moe_gather_rows: ComputePipeline::from_source(
                device.device(),
                ELEMENTWISE_SHADER_SRC,
                "moe_gather_rows_f32",
            )
            .context("Failed to compile moe_gather_rows_f32")?,
            moe_weighted_scatter_add: ComputePipeline::from_source(
                device.device(),
                ELEMENTWISE_SHADER_SRC,
                "moe_weighted_scatter_add_f32",
            )
            .context("Failed to compile moe_weighted_scatter_add_f32")?,
            moe_weighted_reduce_slots_add: ComputePipeline::from_source(
                device.device(),
                ELEMENTWISE_SHADER_SRC,
                "moe_weighted_reduce_slots_add_f32",
            )
            .context("Failed to compile moe_weighted_reduce_slots_add_f32")?,
            moe_weighted_reduce_slots8_add_vec4: ComputePipeline::from_source(
                device.device(),
                ELEMENTWISE_SHADER_SRC,
                "moe_weighted_reduce_slots8_add_f32_vec4",
            )
            .context("Failed to compile moe_weighted_reduce_slots8_add_f32_vec4")?,
            moe_softmax_topk: ComputePipeline::from_source(
                device.device(),
                ELEMENTWISE_SHADER_SRC,
                "moe_softmax_topk_f32",
            )
            .context("Failed to compile moe_softmax_topk_f32")?,
            moe_apply_expert_scales: ComputePipeline::from_source(
                device.device(),
                ELEMENTWISE_SHADER_SRC,
                "moe_apply_expert_scales_f32",
            )
            .context("Failed to compile moe_apply_expert_scales_f32")?,
            argmax_f32,
        })
    }

    // ── Encode methods (for phased dispatch, no command buffer) ──────

    /// Encode in-place RMSNorm: x = x * weight / sqrt(mean(x^2) + eps)
    pub fn encode_rms_norm(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &MetalBuffer,
        weight: &MetalBuffer,
        n: u32,
        eps: f32,
    ) {
        crate::set_pipeline_cached(encoder, self.rms_norm.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n);
        bind_f32(encoder, 3, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode in-place batched RMSNorm across `n_rows` rows of size `n`.
    pub fn encode_rms_norm_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &MetalBuffer,
        weight: &MetalBuffer,
        n: u32,
        n_rows: u32,
        eps: f32,
    ) {
        crate::set_pipeline_cached(encoder, self.rms_norm_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n);
        bind_u32(encoder, 3, n_rows);
        bind_f32(encoder, 4, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_rows as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode out-of-place RMSNorm: out = x * weight / sqrt(mean(x^2) + eps)
    pub fn encode_rms_norm_out(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &MetalBuffer,
        weight: &MetalBuffer,
        out: &MetalBuffer,
        n: u32,
        eps: f32,
    ) {
        crate::set_pipeline_cached(encoder, self.rms_norm_out.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(out.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n);
        bind_f32(encoder, 4, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode batched out-of-place RMSNorm across `n_rows` rows of size `n`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rms_norm_out_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &MetalBuffer,
        weight: &MetalBuffer,
        out: &MetalBuffer,
        n: u32,
        n_rows: u32,
        eps: f32,
    ) {
        // Use float4-vectorized kernel when dim is divisible by 4 (always true for LLM dims).
        // Reference: llama.cpp kernel_rms_norm_fuse_impl<float4, F>.
        let use_vec4 = n.is_multiple_of(4);
        crate::set_pipeline_cached(
            encoder,
            if use_vec4 {
                self.rms_norm_out_batch_vec4.state()
            } else {
                self.rms_norm_out_batch.state()
            },
        );
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(out.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n);
        bind_u32(encoder, 4, n_rows);
        bind_f32(encoder, 5, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_rows as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode batched out-of-place RMSNorm across `n_rows` rows of size `n`,
    /// writing f16 output.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rms_norm_out_batch_f16(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &MetalBuffer,
        weight: &MetalBuffer,
        out_f16: &MetalBuffer,
        n: u32,
        n_rows: u32,
        eps: f32,
    ) {
        crate::set_pipeline_cached(encoder, self.rms_norm_out_batch_f16.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(out_f16.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n);
        bind_u32(encoder, 4, n_rows);
        bind_f32(encoder, 5, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_rows as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode batched residual add + out-of-place RMSNorm.
    ///
    /// For each row:
    /// - `hidden[row] += addend[row]`
    /// - `norm_out[row] = RMSNorm(hidden[row], weight)`
    #[allow(clippy::too_many_arguments)]
    pub fn encode_residual_add_rms_norm_out_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        hidden: &MetalBuffer,
        addend: &MetalBuffer,
        weight: &MetalBuffer,
        norm_out: &MetalBuffer,
        n: u32,
        n_rows: u32,
        eps: f32,
    ) {
        crate::set_pipeline_cached(encoder, self.residual_add_rms_norm_out_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(hidden.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(addend.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(weight.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(norm_out.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, n_rows);
        bind_f32(encoder, 6, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_rows as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode batched residual add + out-of-place RMSNorm, writing f16 norm output.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_residual_add_rms_norm_out_batch_f16(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        hidden: &MetalBuffer,
        addend: &MetalBuffer,
        weight: &MetalBuffer,
        norm_out_f16: &MetalBuffer,
        n: u32,
        n_rows: u32,
        eps: f32,
    ) {
        crate::set_pipeline_cached(encoder, self.residual_add_rms_norm_out_batch_f16.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(hidden.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(addend.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(weight.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(norm_out_f16.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, n_rows);
        bind_f32(encoder, 6, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_rows as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode RoPE on Q and K vectors.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rope(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k: &MetalBuffer,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        rope_dim: u32,
        position: f32,
        freq_base: f32,
    ) {
        self.encode_rope_yarn(
            encoder, q, k, n_q_heads, n_kv_heads, head_dim, rope_dim, position, freq_base, 1.0,
            0.0, 1.0, 32.0, 1.0, 0,
        );
    }

    /// Encode RoPE with YaRN parameters. When `ext_factor == 0.0`,
    /// this reduces to vanilla RoPE (backward compatible).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rope_yarn(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k: &MetalBuffer,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        rope_dim: u32,
        position: f32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        n_ctx_orig: u32,
    ) {
        let half_dim = head_dim / 2;
        let total_pairs = (n_q_heads + n_kv_heads) * half_dim;
        let dims = DispatchDims::d1(total_pairs as usize, ELEMENTWISE_TG_SIZE);

        crate::set_pipeline_cached(encoder, self.rope.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, 1);
        bind_u32(encoder, 3, n_q_heads);
        bind_u32(encoder, 4, n_kv_heads);
        bind_u32(encoder, 5, head_dim);
        bind_f32(encoder, 6, position);
        bind_f32(encoder, 7, 0.0);
        bind_f32(encoder, 8, freq_base);
        // G11: YaRN params (ext_factor=0 → vanilla RoPE)
        bind_f32(encoder, 9, freq_scale);
        bind_f32(encoder, 10, ext_factor);
        bind_f32(encoder, 11, attn_factor);
        bind_f32(encoder, 12, beta_fast);
        bind_f32(encoder, 13, beta_slow);
        bind_u32(encoder, 14, n_ctx_orig);
        bind_u32(encoder, 15, rope_dim);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode batched RoPE across `n_rows` tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rope_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k: &MetalBuffer,
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        rope_dim: u32,
        start_pos: f32,
        pos_step: f32,
        freq_base: f32,
    ) {
        let half_dim = head_dim / 2;
        let total_pairs = n_rows * (n_q_heads + n_kv_heads) * half_dim;
        let dims = DispatchDims::d1(total_pairs as usize, ELEMENTWISE_TG_SIZE);

        crate::set_pipeline_cached(encoder, self.rope_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n_rows);
        bind_u32(encoder, 3, n_q_heads);
        bind_u32(encoder, 4, n_kv_heads);
        bind_u32(encoder, 5, head_dim);
        bind_f32(encoder, 6, start_pos);
        bind_f32(encoder, 7, pos_step);
        bind_f32(encoder, 8, freq_base);
        // G11: YaRN params — vanilla defaults (ext_factor=0)
        bind_f32(encoder, 9, 1.0); // freq_scale
        bind_f32(encoder, 10, 0.0); // ext_factor
        bind_f32(encoder, 11, 1.0); // attn_factor
        bind_f32(encoder, 12, 32.0); // beta_fast
        bind_f32(encoder, 13, 1.0); // beta_slow
        bind_u32(encoder, 14, 0); // n_ctx_orig
        bind_u32(encoder, 15, rope_dim);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode NeoX split-half partial RoPE on batched Q/K buffers.
    ///
    /// Qwen3.5 rotates pairs `(x[i], x[i + rope_dim/2])` within the rotary
    /// prefix instead of adjacent pairs `(x[2i], x[2i+1])`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rope_batch_neox_partial(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k: &MetalBuffer,
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        rope_dim: u32,
        start_pos: f32,
        pos_step: f32,
        freq_base: f32,
    ) {
        let rope_dim_eff = if rope_dim == 0 {
            head_dim
        } else {
            rope_dim.min(head_dim)
        };
        let rope_pairs = rope_dim_eff / 2;
        let total_pairs = n_rows * (n_q_heads + n_kv_heads) * rope_pairs;
        let dims = DispatchDims::d1(total_pairs as usize, ELEMENTWISE_TG_SIZE);

        crate::set_pipeline_cached(encoder, self.rope_neox_partial_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n_rows);
        bind_u32(encoder, 3, n_q_heads);
        bind_u32(encoder, 4, n_kv_heads);
        bind_u32(encoder, 5, head_dim);
        bind_f32(encoder, 6, start_pos);
        bind_f32(encoder, 7, pos_step);
        bind_f32(encoder, 8, freq_base);
        bind_f32(encoder, 9, 1.0);
        bind_f32(encoder, 10, 0.0);
        bind_f32(encoder, 11, 1.0);
        bind_f32(encoder, 12, 32.0);
        bind_f32(encoder, 13, 1.0);
        bind_u32(encoder, 14, 0);
        bind_u32(encoder, 15, rope_dim);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode Gemma4 NeoX split-half RoPE with explicit frequency factors.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rope_batch_neox_partial_with_freq_factors(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k: &MetalBuffer,
        freq_factors: &MetalBuffer,
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        rope_dim: u32,
        start_pos: f32,
        pos_step: f32,
        freq_base: f32,
    ) {
        let rope_dim_eff = if rope_dim == 0 {
            head_dim
        } else {
            rope_dim.min(head_dim)
        };
        let rope_pairs = rope_dim_eff / 2;
        let total_pairs = n_rows * (n_q_heads + n_kv_heads) * rope_pairs;
        let dims = DispatchDims::d1(total_pairs as usize, ELEMENTWISE_TG_SIZE);

        crate::set_pipeline_cached(encoder, self.rope_neox_partial_freq_factors_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(freq_factors.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n_rows);
        bind_u32(encoder, 4, n_q_heads);
        bind_u32(encoder, 5, n_kv_heads);
        bind_u32(encoder, 6, head_dim);
        bind_f32(encoder, 7, start_pos);
        bind_f32(encoder, 8, pos_step);
        bind_f32(encoder, 9, freq_base);
        bind_u32(encoder, 10, rope_dim);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode per-head RMSNorm.
    pub fn encode_per_head_rms_norm(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        buf: &MetalBuffer,
        weight: &MetalBuffer,
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    ) {
        crate::set_pipeline_cached(encoder, self.per_head_rms_norm.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buf.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, 1);
        bind_u32(encoder, 3, n_heads);
        bind_u32(encoder, 4, head_dim);
        bind_f32(encoder, 5, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_heads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode batched per-head RMSNorm across `n_rows`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_per_head_rms_norm_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        buf: &MetalBuffer,
        weight: &MetalBuffer,
        n_rows: u32,
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    ) {
        crate::set_pipeline_cached(encoder, self.per_head_rms_norm_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buf.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n_rows);
        bind_u32(encoder, 3, n_heads);
        bind_u32(encoder, 4, head_dim);
        bind_f32(encoder, 5, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n_rows * n_heads) as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode batched `gate = SiLU(gate) * RMSNorm(src)` across `n_rows`,
    /// where RMSNorm is applied independently to each head.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_per_head_rms_norm_silu_mul_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        gate: &MetalBuffer,
        src: &MetalBuffer,
        weight: &MetalBuffer,
        n_rows: u32,
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    ) {
        crate::set_pipeline_cached(encoder, self.per_head_rms_norm_silu_mul_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(weight.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n_rows);
        bind_u32(encoder, 4, n_heads);
        bind_u32(encoder, 5, head_dim);
        bind_f32(encoder, 6, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n_rows * n_heads) as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode per-head RMSNorm without a learned weight.
    pub fn encode_per_head_rms_norm_no_weight(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        buf: &MetalBuffer,
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    ) {
        crate::set_pipeline_cached(encoder, self.per_head_rms_norm_no_weight.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buf.mtl_buffer()), 0, 0);
        }
        bind_u32(encoder, 1, 1);
        bind_u32(encoder, 2, n_heads);
        bind_u32(encoder, 3, head_dim);
        bind_f32(encoder, 4, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_heads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode batched per-head RMSNorm without a learned weight.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_per_head_rms_norm_no_weight_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        buf: &MetalBuffer,
        n_rows: u32,
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    ) {
        crate::set_pipeline_cached(encoder, self.per_head_rms_norm_no_weight_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buf.mtl_buffer()), 0, 0);
        }
        bind_u32(encoder, 1, n_rows);
        bind_u32(encoder, 2, n_heads);
        bind_u32(encoder, 3, head_dim);
        bind_f32(encoder, 4, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n_rows * n_heads) as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode Gemma-style post-attention RMSNorm + residual add + FFN RMSNorm.
    ///
    /// For each row:
    /// - `addend` is RMS-normalized with `post_weight`
    /// - `hidden += normalized(addend)`
    /// - `norm_out = RMSNorm(hidden, residual_weight)`
    #[allow(clippy::too_many_arguments)]
    pub fn encode_post_attn_norm_residual_add_rms_norm_out_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        hidden: &MetalBuffer,
        addend: &MetalBuffer,
        post_weight: &MetalBuffer,
        residual_weight: &MetalBuffer,
        norm_out: &MetalBuffer,
        n: u32,
        n_rows: u32,
        eps: f32,
    ) {
        encoder
            .setComputePipelineState(self.post_attn_norm_residual_add_rms_norm_out_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(hidden.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(addend.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(post_weight.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(residual_weight.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(norm_out.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, n);
        bind_u32(encoder, 6, n_rows);
        bind_f32(encoder, 7, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_rows as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode Gemma-style post-FFN RMSNorm + residual add + next-attention RMSNorm.
    ///
    /// For each row:
    /// - `addend` is RMS-normalized with `post_weight`
    /// - `hidden += normalized(addend)`
    /// - `norm_out = RMSNorm(hidden, residual_weight)`
    #[allow(clippy::too_many_arguments)]
    pub fn encode_post_ffn_norm_residual_add_rms_norm_out_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        hidden: &MetalBuffer,
        addend: &MetalBuffer,
        post_weight: &MetalBuffer,
        residual_weight: &MetalBuffer,
        norm_out: &MetalBuffer,
        n: u32,
        n_rows: u32,
        eps: f32,
    ) {
        crate::set_pipeline_cached(
            encoder,
            self.post_ffn_norm_residual_add_rms_norm_out_batch.state(),
        );
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(hidden.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(addend.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(post_weight.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(residual_weight.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(norm_out.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, n);
        bind_u32(encoder, 6, n_rows);
        bind_f32(encoder, 7, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_rows as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode fused Gemma-style Q/K per-head RMSNorm + RoPE across `n_rows`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_qk_norm_rope_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k: &MetalBuffer,
        q_weight: &MetalBuffer,
        k_weight: &MetalBuffer,
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        eps: f32,
        start_pos: f32,
        pos_step: f32,
        freq_base: f32,
    ) {
        crate::set_pipeline_cached(encoder, self.qk_norm_rope_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(q_weight.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(k_weight.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n_rows);
        bind_u32(encoder, 5, n_q_heads);
        bind_u32(encoder, 6, n_kv_heads);
        bind_u32(encoder, 7, head_dim);
        bind_f32(encoder, 8, eps);
        bind_f32(encoder, 9, start_pos);
        bind_f32(encoder, 10, pos_step);
        bind_f32(encoder, 11, freq_base);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n_rows * (n_q_heads + n_kv_heads)) as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode fused batched QKV split + Q/K norm + RoPE + KV append.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_qkv_split_qk_norm_rope_append_kv_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        q_weight: &MetalBuffer,
        k_weight: &MetalBuffer,
        cache_k: &MetalBuffer,
        cache_v: &MetalBuffer,
        cache_f16: bool,
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        eps: f32,
        start_pos: f32,
        pos_step: f32,
        freq_base: f32,
        cache_offset: u32,
        cache_stride: u32,
    ) {
        let half_dim = head_dim / 2;
        let q_pairs = n_q_heads * half_dim;
        let k_pairs = n_kv_heads * half_dim;
        let kv_dim = n_kv_heads * head_dim;
        let items_per_row = q_pairs + k_pairs + kv_dim;
        let total = (n_rows as usize) * (items_per_row as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(
            encoder,
            if cache_f16 {
                self.qkv_split_qk_norm_rope_append_kv_batch_f16.state()
            } else {
                self.qkv_split_qk_norm_rope_append_kv_batch_f32.state()
            },
        );
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(q_weight.mtl_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(k_weight.mtl_buffer()), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(cache_k.mtl_buffer()), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(cache_v.mtl_buffer()), 0, 7);
        }
        bind_u32(encoder, 8, n_rows);
        bind_u32(encoder, 9, n_q_heads);
        bind_u32(encoder, 10, n_kv_heads);
        bind_u32(encoder, 11, head_dim);
        bind_f32(encoder, 12, eps);
        bind_f32(encoder, 13, start_pos);
        bind_f32(encoder, 14, pos_step);
        bind_f32(encoder, 15, freq_base);
        bind_u32(encoder, 16, cache_offset);
        bind_u32(encoder, 17, cache_stride);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode fused QKV split + bias + QK norm + RoPE + KV cache append (Qwen3).
    ///
    /// Combines 7 dispatches (split, 3 bias, norm+RoPE, 2 KV append) into 1.
    /// `src` is the fused QKV output `[n_rows × (q_dim + 2*kv_dim)]`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_qkv_split_bias_qknorm_rope_append_kv_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        q_weight: &MetalBuffer,
        k_weight: &MetalBuffer,
        cache_k: &MetalBuffer,
        cache_v: &MetalBuffer,
        cache_f16: bool,
        q_bias: &MetalBuffer,
        k_bias: &MetalBuffer,
        v_bias: &MetalBuffer,
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        eps: f32,
        start_pos: f32,
        pos_step: f32,
        freq_base: f32,
        cache_offset: u32,
        cache_stride: u32,
    ) {
        let half_dim = head_dim / 2;
        let q_pairs = n_q_heads * half_dim;
        let k_pairs = n_kv_heads * half_dim;
        let kv_dim = n_kv_heads * head_dim;
        let items_per_row = q_pairs + k_pairs + kv_dim;
        let total = (n_rows as usize) * (items_per_row as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(
            encoder,
            if cache_f16 {
                self.qkv_split_bias_qknorm_rope_append_kv_batch_f16.state()
            } else {
                self.qkv_split_bias_qknorm_rope_append_kv_batch_f32.state()
            },
        );
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(q_weight.mtl_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(k_weight.mtl_buffer()), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(cache_k.mtl_buffer()), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(cache_v.mtl_buffer()), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(q_bias.mtl_buffer()), 0, 8);
            encoder.setBuffer_offset_atIndex(Some(k_bias.mtl_buffer()), 0, 9);
            encoder.setBuffer_offset_atIndex(Some(v_bias.mtl_buffer()), 0, 10);
        }
        bind_u32(encoder, 11, n_rows);
        bind_u32(encoder, 12, n_q_heads);
        bind_u32(encoder, 13, n_kv_heads);
        bind_u32(encoder, 14, head_dim);
        bind_f32(encoder, 15, eps);
        bind_f32(encoder, 16, start_pos);
        bind_f32(encoder, 17, pos_step);
        bind_f32(encoder, 18, freq_base);
        bind_u32(encoder, 19, cache_offset);
        bind_u32(encoder, 20, cache_stride);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode GELU(gate) * up.
    pub fn encode_gelu_elementwise_mul(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        n: u32,
    ) {
        let dims = DispatchDims::d1(n as usize, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.gelu_elementwise_mul.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode batched GELU(gate) * up across `n_rows` rows of length `n`.
    pub fn encode_gelu_elementwise_mul_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        n: u32,
        n_rows: u32,
    ) {
        let total = (n as usize) * (n_rows as usize);
        if n.is_multiple_of(4) {
            let dims = DispatchDims::d1(total / 4, ELEMENTWISE_TG_SIZE);
            crate::set_pipeline_cached(encoder, self.gelu_elementwise_mul_batch_vec4.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 1);
            }
            bind_u32(encoder, 2, n);
            bind_u32(encoder, 3, n_rows);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
        } else {
            let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
            crate::set_pipeline_cached(encoder, self.gelu_elementwise_mul_batch.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 1);
            }
            bind_u32(encoder, 2, n);
            bind_u32(encoder, 3, n_rows);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
        }
    }

    /// Encode batched `dst[row, i] = GELU(src[row, i]) * src[row, n + i]`.
    ///
    /// `src` is laid out as contiguous `[gate | up]` rows with width `2 * n`.
    pub fn encode_gelu_split_mul_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        n: u32,
        n_rows: u32,
    ) {
        let total = (n as usize) * (n_rows as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.gelu_split_mul_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n);
        bind_u32(encoder, 3, n_rows);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode in-place GELU: x[i] = GELU(x[i]).
    pub fn encode_gelu_inplace(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &MetalBuffer,
        n: u32,
    ) {
        let dims = DispatchDims::d1(n as usize, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.gelu_inplace.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 0);
        }
        bind_u32(encoder, 1, n);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode batched in-place GELU across `n_rows` rows of length `n`.
    pub fn encode_gelu_inplace_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &MetalBuffer,
        n: u32,
        n_rows: u32,
    ) {
        let total = (n as usize) * (n_rows as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.gelu_inplace_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 0);
        }
        bind_u32(encoder, 1, n);
        bind_u32(encoder, 2, n_rows);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode SiLU(gate) * up.
    pub fn encode_silu_elementwise_mul(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        n: u32,
    ) {
        let dims = DispatchDims::d1(n as usize, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.silu_elementwise_mul.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode sigmoid in-place: x[i] = sigmoid(x[i]).
    pub fn encode_sigmoid_inplace(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &MetalBuffer,
        n: u32,
    ) {
        if n.is_multiple_of(4) {
            let dims = DispatchDims::d1((n as usize) / 4, ELEMENTWISE_TG_SIZE);
            crate::set_pipeline_cached(encoder, self.sigmoid_inplace_vec4.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 0);
            }
            bind_u32(encoder, 1, n);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
        } else {
            let dims = DispatchDims::d1(n as usize, ELEMENTWISE_TG_SIZE);
            crate::set_pipeline_cached(encoder, self.sigmoid_inplace.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 0);
            }
            bind_u32(encoder, 1, n);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
        }
    }

    /// Encode softplus(alpha + bias) * a in-place.
    pub fn encode_softplus_bias_mul(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        alpha: &MetalBuffer,
        bias: &MetalBuffer,
        a: &MetalBuffer,
        n: u32,
    ) {
        let dims = DispatchDims::d1(n as usize, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.softplus_bias_mul.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(alpha.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(bias.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode fused softplus(alpha + bias) * a and sigmoid(beta) in-place.
    pub fn encode_softplus_bias_mul_sigmoid_pair(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        alpha: &MetalBuffer,
        beta: &MetalBuffer,
        bias: &MetalBuffer,
        a: &MetalBuffer,
        n: u32,
    ) {
        let dims = DispatchDims::d1(n as usize, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.softplus_bias_mul_sigmoid_pair.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(alpha.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(beta.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(bias.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode batched softplus(alpha + bias[head]) * a[head] in-place.
    pub fn encode_softplus_bias_mul_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        alpha: &MetalBuffer,
        bias: &MetalBuffer,
        a: &MetalBuffer,
        n: u32,
        head_dim: u32,
    ) {
        let dims = DispatchDims::d1(n as usize, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.softplus_bias_mul_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(alpha.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(bias.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n);
        bind_u32(encoder, 4, head_dim);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode batched fused softplus(alpha + bias[head]) * a[head] and
    /// sigmoid(beta) in-place.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_softplus_bias_mul_sigmoid_pair_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        alpha: &MetalBuffer,
        beta: &MetalBuffer,
        bias: &MetalBuffer,
        a: &MetalBuffer,
        n: u32,
        head_dim: u32,
    ) {
        let dims = DispatchDims::d1(n as usize, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.softplus_bias_mul_sigmoid_pair_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(alpha.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(beta.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(bias.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, head_dim);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode per-head L2 normalization.
    pub fn encode_l2_norm_per_head(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &MetalBuffer,
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    ) {
        crate::set_pipeline_cached(encoder, self.l2_norm_per_head.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 0);
        }
        bind_u32(encoder, 1, n_heads);
        bind_u32(encoder, 2, head_dim);
        bind_f32(encoder, 3, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_heads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256, // matches NORM_TG_SIZE in elementwise.metal
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode sigmoid(gate) * out in-place.
    ///
    /// Used by Qwen3.5 attention gate: applies sigmoid to the gate vector
    /// and multiplies with the attention output.
    pub fn encode_sigmoid_elementwise_mul(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        gate: &MetalBuffer,
        out: &MetalBuffer,
        n: u32,
    ) {
        let dims = DispatchDims::d1(n as usize, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.sigmoid_elementwise_mul.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(out.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode out[i] = sigmoid(gate[0]) * out[i].
    pub fn encode_sigmoid_scalar_mul_inplace(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        gate: &MetalBuffer,
        out: &MetalBuffer,
        n: u32,
    ) {
        let dims = DispatchDims::d1(n as usize, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.sigmoid_scalar_mul_inplace.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(out.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode gate = dot(row, x); out[i] = sigmoid(gate) * out[i].
    pub fn encode_dense_row_dot_sigmoid_mul_inplace(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        row: &MetalBuffer,
        x: &MetalBuffer,
        out: &MetalBuffer,
        k: u32,
        n: u32,
    ) {
        crate::set_pipeline_cached(encoder, self.dense_row_dot_sigmoid_mul_inplace.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(row.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(out.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, k);
        bind_u32(encoder, 4, n);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode batched SiLU(gate) * up across `n_rows` rows of length `n`.
    pub fn encode_silu_elementwise_mul_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        n: u32,
        n_rows: u32,
    ) {
        let total = (n as usize) * (n_rows as usize);
        if n.is_multiple_of(4) {
            let dims = DispatchDims::d1(total / 4, ELEMENTWISE_TG_SIZE);
            crate::set_pipeline_cached(encoder, self.silu_elementwise_mul_batch_vec4.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 1);
            }
            bind_u32(encoder, 2, n);
            bind_u32(encoder, 3, n_rows);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
        } else {
            let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
            crate::set_pipeline_cached(encoder, self.silu_elementwise_mul_batch.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 1);
            }
            bind_u32(encoder, 2, n);
            bind_u32(encoder, 3, n_rows);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
        }
    }

    /// Encode batched SiLU(gate) * up across `n_rows` rows of length `n`,
    /// writing result to f16 output buffer.
    pub fn encode_silu_elementwise_mul_batch_f16(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        out_f16: &MetalBuffer,
        n: u32,
        n_rows: u32,
    ) {
        let total = (n as usize) * (n_rows as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.silu_elementwise_mul_batch_f16.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(out_f16.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n);
        bind_u32(encoder, 4, n_rows);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode a += b.
    pub fn encode_elementwise_add(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        n: u32,
    ) {
        let dims = DispatchDims::d1(n as usize, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.elementwise_add.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode weighted add: dst[i] += scale * src[i].
    /// Used for MoE expert output accumulation.
    pub fn encode_elementwise_weighted_add(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        dst: &MetalBuffer,
        src: &MetalBuffer,
        scale: f32,
        n: u32,
    ) {
        let dims = DispatchDims::d1(n as usize, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.elementwise_weighted_add.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 1);
        }
        bind_f32(encoder, 2, scale);
        bind_u32(encoder, 3, n);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    // ── General-purpose elementwise ops (GDN graph building blocks) ───────

    /// Encode dst = exp(src) for `count` f32 elements.
    pub fn encode_gen_exp(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        count: u32,
    ) {
        let dims = DispatchDims::d1(count as usize, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.gen_exp.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, count);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode dst = a * b (elementwise) for `count` f32 elements.
    pub fn encode_gen_mul(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        dst: &MetalBuffer,
        count: u32,
    ) {
        let dims = DispatchDims::d1(count as usize, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.gen_mul.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, count);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode dst = a + b (elementwise, out-of-place) for `count` f32 elements.
    pub fn encode_gen_add(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        dst: &MetalBuffer,
        count: u32,
    ) {
        let dims = DispatchDims::d1(count as usize, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.gen_add.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, count);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode dst = a - b (elementwise) for `count` f32 elements.
    pub fn encode_gen_sub(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        dst: &MetalBuffer,
        count: u32,
    ) {
        let dims = DispatchDims::d1(count as usize, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.gen_sub.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, count);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode dst = -src (elementwise negate) for `count` f32 elements.
    pub fn encode_gen_neg(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        count: u32,
    ) {
        let dims = DispatchDims::d1(count as usize, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.gen_neg.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, count);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode dst = src * scale (broadcast scalar multiply) for `count` f32 elements.
    pub fn encode_gen_scale(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        scale: f32,
        count: u32,
    ) {
        let dims = DispatchDims::d1(count as usize, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.gen_scale.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
        }
        bind_f32(encoder, 2, scale);
        bind_u32(encoder, 3, count);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode MoE gather: dst[slot*dim..] = src[indices[slot]*dim..] for slot in 0..n_slots.
    pub fn encode_moe_gather(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        indices: &MetalBuffer,
        dst: &MetalBuffer,
        dim: u32,
        n_slots: u32,
    ) {
        let total = n_slots * dim;
        let dims = DispatchDims::d1(total as usize, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.moe_gather_rows.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(indices.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, dim);
        bind_u32(encoder, 4, n_slots);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode MoE weighted scatter-add:
    /// dst[indices[s]*dim + d] += weights[s] * src[s*dim + d].
    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_weighted_scatter(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        indices: &MetalBuffer,
        weights: &MetalBuffer,
        dst: &MetalBuffer,
        dim: u32,
        n_slots: u32,
    ) {
        let total = n_slots * dim;
        let dims = DispatchDims::d1(total as usize, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.moe_weighted_scatter_add.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(indices.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(weights.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, dim);
        bind_u32(encoder, 5, n_slots);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode per-token MoE slot reduction:
    /// dst[token, d] += sum_k weights[token, k] * src[token, k, d].
    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_weighted_reduce_slots(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        weights: &MetalBuffer,
        dst: &MetalBuffer,
        dim: u32,
        n_tokens: u32,
        n_expert_used: u32,
    ) {
        if n_expert_used == 8 && dim.is_multiple_of(4) && moe_weighted_reduce_slots8_vec4_enabled()
        {
            let total4 = n_tokens * (dim / 4);
            let dims = DispatchDims::d1(total4 as usize, ELEMENTWISE_TG_SIZE);
            crate::set_pipeline_cached(encoder, self.moe_weighted_reduce_slots8_add_vec4.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(weights.mtl_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 2);
            }
            bind_u32(encoder, 3, dim);
            bind_u32(encoder, 4, n_tokens);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
        } else {
            let total = n_tokens * dim;
            let dims = DispatchDims::d1(total as usize, ELEMENTWISE_TG_SIZE);
            crate::set_pipeline_cached(encoder, self.moe_weighted_reduce_slots_add.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(weights.mtl_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 2);
            }
            bind_u32(encoder, 3, dim);
            bind_u32(encoder, 4, n_tokens);
            bind_u32(encoder, 5, n_expert_used);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
        }
    }

    /// Encode batched a += b across `n_rows` rows of length `n`.
    /// Encode GPU softmax + top-k for MoE router.
    /// router_logits: [n_tokens, n_expert]. Output: expert_ids [n_tokens, n_expert_used] i32,
    /// expert_weights [n_tokens, n_expert_used] f32.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_softmax_topk(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        router_logits: &MetalBuffer,
        expert_ids: &MetalBuffer,
        expert_weights: &MetalBuffer,
        n_tokens: u32,
        n_expert: u32,
        n_expert_used: u32,
    ) {
        crate::set_pipeline_cached(encoder, self.moe_softmax_topk.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(router_logits.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(expert_ids.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(expert_weights.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n_expert);
        bind_u32(encoder, 4, n_expert_used);
        // Threadgroup memory: logits[ne] + probs[ne] + sel[neu] + wts[neu] + reduce[TG]
        let tg = (n_expert as usize).min(256);
        let smem = (n_expert as usize * 2 + n_expert_used as usize * 2 + tg)
            * std::mem::size_of::<f32>()
            + n_expert_used as usize * std::mem::size_of::<i32>();
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(smem, 0);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_tokens as _,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode per-selected-expert output scaling:
    /// `expert_weights[token, slot] *= expert_scales[expert_ids[token, slot]]`.
    pub fn encode_moe_apply_expert_scales(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        expert_ids: &MetalBuffer,
        expert_weights: &MetalBuffer,
        expert_scales: &MetalBuffer,
        n_tokens: u32,
        n_expert_used: u32,
    ) {
        let total = (n_tokens as usize) * (n_expert_used as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.moe_apply_expert_scales.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(expert_ids.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(expert_weights.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(expert_scales.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n_tokens);
        bind_u32(encoder, 4, n_expert_used);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    pub fn encode_elementwise_add_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        n: u32,
        n_rows: u32,
    ) {
        let total = (n as usize) * (n_rows as usize);
        if n.is_multiple_of(4) {
            let dims = DispatchDims::d1(total / 4, ELEMENTWISE_TG_SIZE);
            crate::set_pipeline_cached(encoder, self.elementwise_add_batch_vec4.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), 0, 1);
            }
            bind_u32(encoder, 2, n);
            bind_u32(encoder, 3, n_rows);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
        } else {
            let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
            crate::set_pipeline_cached(encoder, self.elementwise_add_batch.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), 0, 1);
            }
            bind_u32(encoder, 2, n);
            bind_u32(encoder, 3, n_rows);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
        }
    }

    /// Encode f32->f16 cast over `n` elements.
    pub fn encode_cast_f32_to_f16(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        n: u32,
    ) {
        if n.is_multiple_of(4) {
            let dims = DispatchDims::d1((n as usize) / 4, ELEMENTWISE_TG_SIZE);
            crate::set_pipeline_cached(encoder, self.cast_f32_to_f16_vec4.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
            }
            bind_u32(encoder, 2, n);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
        } else {
            let dims = DispatchDims::d1(n as usize, ELEMENTWISE_TG_SIZE);
            crate::set_pipeline_cached(encoder, self.cast_f32_to_f16.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
            }
            bind_u32(encoder, 2, n);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
        }
    }

    /// Encode f16->f32 cast over `n` elements.
    pub fn encode_cast_f16_to_f32(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        n: u32,
    ) {
        if n.is_multiple_of(4) {
            let dims = DispatchDims::d1((n as usize) / 4, ELEMENTWISE_TG_SIZE);
            crate::set_pipeline_cached(encoder, self.cast_f16_to_f32_vec4.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
            }
            bind_u32(encoder, 2, n);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
        } else {
            let dims = DispatchDims::d1(n as usize, ELEMENTWISE_TG_SIZE);
            crate::set_pipeline_cached(encoder, self.cast_f16_to_f32.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
            }
            bind_u32(encoder, 2, n);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
        }
    }

    /// Encode batched split from fused QKV rows into Q/K/V output buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_qkv_split_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        n_rows: u32,
        q_dim: u32,
        kv_dim: u32,
    ) {
        let fused_dim = q_dim + 2 * kv_dim;
        let total = (n_rows as usize) * (fused_dim as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.qkv_split_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n_rows);
        bind_u32(encoder, 5, q_dim);
        bind_u32(encoder, 6, kv_dim);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode batched split from fused Q+gate rows into separate Q and gate buffers.
    ///
    /// Used by Qwen3.5 full-attention layers where the Q projection output is
    /// interleaved per head: `[q_h0, g_h0, q_h1, g_h1, ...]`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_split_qgate_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        q: &MetalBuffer,
        gate: &MetalBuffer,
        n_rows: u32,
        q_dim: u32,
        head_dim: u32,
    ) {
        let total = (n_rows as usize) * (q_dim as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.split_qgate_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n_rows);
        bind_u32(encoder, 4, q_dim);
        bind_u32(encoder, 5, head_dim);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode fused batched QKV split + RoPE.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_qkv_split_rope_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        start_pos: f32,
        pos_step: f32,
        freq_base: f32,
    ) {
        let half_dim = head_dim / 2;
        let q_pairs = n_q_heads * half_dim;
        let k_pairs = n_kv_heads * half_dim;
        let kv_dim = n_kv_heads * head_dim;
        let items_per_row = q_pairs + k_pairs + kv_dim;
        let total = (n_rows as usize) * (items_per_row as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.qkv_split_rope_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n_rows);
        bind_u32(encoder, 5, n_q_heads);
        bind_u32(encoder, 6, n_kv_heads);
        bind_u32(encoder, 7, head_dim);
        bind_f32(encoder, 8, start_pos);
        bind_f32(encoder, 9, pos_step);
        bind_f32(encoder, 10, freq_base);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode fused batched QKV split + RoPE + KV append.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_qkv_split_rope_append_kv_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        cache_k: &MetalBuffer,
        cache_v: &MetalBuffer,
        cache_f16: bool,
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        start_pos: f32,
        pos_step: f32,
        freq_base: f32,
        cache_offset: u32,
        cache_stride: u32,
    ) {
        let half_dim = head_dim / 2;
        let q_pairs = n_q_heads * half_dim;
        let k_pairs = n_kv_heads * half_dim;
        let kv_dim = n_kv_heads * head_dim;
        let items_per_row = q_pairs + k_pairs + kv_dim;
        let total = (n_rows as usize) * (items_per_row as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(
            encoder,
            if cache_f16 {
                self.qkv_split_rope_append_kv_batch_f16.state()
            } else {
                self.qkv_split_rope_append_kv_batch_f32.state()
            },
        );
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(cache_k.mtl_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(cache_v.mtl_buffer()), 0, 5);
        }
        bind_u32(encoder, 6, n_rows);
        bind_u32(encoder, 7, n_q_heads);
        bind_u32(encoder, 8, n_kv_heads);
        bind_u32(encoder, 9, head_dim);
        bind_f32(encoder, 10, start_pos);
        bind_f32(encoder, 11, pos_step);
        bind_f32(encoder, 12, freq_base);
        bind_u32(encoder, 13, cache_offset);
        bind_u32(encoder, 14, cache_stride);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode KV cache append: copy `count` floats from src to dst at offset.
    pub fn encode_kv_append(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        dst_f16: bool,
        offset: u32,
        count: u32,
    ) {
        let dims = DispatchDims::d1(count as usize, ELEMENTWISE_TG_SIZE);
        let pipeline = if dst_f16 {
            self.kv_append_f16.state()
        } else {
            self.kv_append_f32.state()
        };
        crate::set_pipeline_cached(encoder, pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, offset);
        bind_u32(encoder, 3, count);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode batched KV append from src[N×count] into dst with row stride.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_kv_append_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        dst_f16: bool,
        dst_offset: u32,
        dst_row_stride: u32,
        count: u32,
        n_rows: u32,
    ) {
        let total = (count as usize) * (n_rows as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        let pipeline = if dst_f16 {
            self.kv_append_batch_f16.state()
        } else {
            self.kv_append_batch_f32.state()
        };
        crate::set_pipeline_cached(encoder, pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, dst_offset);
        bind_u32(encoder, 3, dst_row_stride);
        bind_u32(encoder, 4, count);
        bind_u32(encoder, 5, n_rows);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode batched append of both K and V in one dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_kv_append_batch_pair(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src_k: &MetalBuffer,
        src_v: &MetalBuffer,
        dst_k: &MetalBuffer,
        dst_v: &MetalBuffer,
        dst_f16: bool,
        dst_offset: u32,
        dst_row_stride: u32,
        count: u32,
        n_rows: u32,
    ) {
        let total = (count as usize) * (n_rows as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(
            encoder,
            if dst_f16 {
                self.kv_append_batch2_f16.state()
            } else {
                self.kv_append_batch2_f32.state()
            },
        );
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src_k.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(src_v.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(dst_k.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(dst_v.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, dst_offset);
        bind_u32(encoder, 5, dst_row_stride);
        bind_u32(encoder, 6, count);
        bind_u32(encoder, 7, n_rows);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode batched KV append with Q8_0 quantization (single buffer).
    ///
    /// src: f32 tensor [n_rows × kv_stride]
    /// dst: Q8_0 block buffer
    /// dst_row_offset: block offset to first row (seq_len × blocks_per_row)
    /// blocks_per_row: n_kv_heads × (head_dim / 32)
    /// kv_stride: n_kv_heads × head_dim (element count per row)
    #[allow(clippy::too_many_arguments)]
    pub fn encode_kv_append_batch_q8(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        dst_row_offset: u32,
        blocks_per_row: u32,
        kv_stride: u32,
        n_rows: u32,
    ) {
        let total_blocks = (blocks_per_row as usize) * (n_rows as usize);
        let dims = DispatchDims::d1(total_blocks, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.kv_append_batch_q8_0.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, dst_row_offset);
        bind_u32(encoder, 3, blocks_per_row);
        bind_u32(encoder, 4, kv_stride);
        bind_u32(encoder, 5, n_rows);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode batched K+V append with Q8_0 quantization (paired buffers).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_kv_append_batch_pair_q8(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src_k: &MetalBuffer,
        src_v: &MetalBuffer,
        dst_k: &MetalBuffer,
        dst_v: &MetalBuffer,
        dst_row_offset: u32,
        blocks_per_row: u32,
        kv_stride: u32,
        n_rows: u32,
    ) {
        let total_blocks = (blocks_per_row as usize) * (n_rows as usize);
        let dims = DispatchDims::d1(total_blocks, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.kv_append_batch2_q8_0.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src_k.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(src_v.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(dst_k.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(dst_v.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, dst_row_offset);
        bind_u32(encoder, 5, blocks_per_row);
        bind_u32(encoder, 6, kv_stride);
        bind_u32(encoder, 7, n_rows);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode a GPU buffer copy with custom source byte offset.
    ///
    /// Copies `count` floats: dst[dst_float_offset..] = src[src_byte_offset..].
    /// Uses the kv_append pipeline with custom buffer offset binding.
    /// This enables batch↔scratch transfers for batched prefill.
    pub fn encode_buffer_copy(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        src_byte_offset: usize,
        dst: &MetalBuffer,
        dst_float_offset: u32,
        count: u32,
    ) {
        let dims = DispatchDims::d1(count as usize, ELEMENTWISE_TG_SIZE);
        crate::set_pipeline_cached(encoder, self.kv_append_f32.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), src_byte_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, dst_float_offset);
        bind_u32(encoder, 3, count);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Dispatch GPU-side argmax: find index and value of maximum element.
    ///
    /// Runs a single threadgroup (TG=1024) parallel reduction. Results are
    /// written to `result_idx` (u32) and `result_val` (f32). Caller must
    /// sync the command buffer before reading results.
    pub fn encode_argmax_f32(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        data: &MetalBuffer,
        result_idx: &MetalBuffer,
        result_val: &MetalBuffer,
        n: u32,
    ) {
        crate::set_pipeline_cached(encoder, self.argmax_f32.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(data.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(result_idx.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(result_val.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n);
        // Single threadgroup of 1024 threads
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        );
    }

    // ── Standalone methods (create command buffer, for testing) ──────

    /// In-place RMSNorm on GPU (standalone, creates own command buffer).
    pub fn rms_norm(
        &self,
        device: &MetalDevice,
        x: &MetalBuffer,
        weight: &MetalBuffer,
        n: u32,
        eps: f32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_rms_norm(encoder, x, weight, n, eps);
            Ok(())
        })
    }

    /// Out-of-place RMSNorm on GPU (standalone).
    pub fn rms_norm_out(
        &self,
        device: &MetalDevice,
        x: &MetalBuffer,
        weight: &MetalBuffer,
        out: &MetalBuffer,
        n: u32,
        eps: f32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_rms_norm_out(encoder, x, weight, out, n, eps);
            Ok(())
        })
    }

    /// RoPE on GPU (standalone).
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        device: &MetalDevice,
        q: &MetalBuffer,
        k: &MetalBuffer,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        position: f32,
        freq_base: f32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_rope(
                encoder, q, k, n_q_heads, n_kv_heads, head_dim, head_dim, position, freq_base,
            );
            Ok(())
        })
    }

    /// Batched RoPE on GPU (standalone).
    #[allow(clippy::too_many_arguments)]
    pub fn rope_batch(
        &self,
        device: &MetalDevice,
        q: &MetalBuffer,
        k: &MetalBuffer,
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        start_pos: f32,
        pos_step: f32,
        freq_base: f32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_rope_batch(
                encoder, q, k, n_rows, n_q_heads, n_kv_heads, head_dim, head_dim, start_pos,
                pos_step, freq_base,
            );
            Ok(())
        })
    }

    /// Batched NeoX split-half RoPE with explicit frequency factors.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_batch_neox_partial_with_freq_factors(
        &self,
        device: &MetalDevice,
        q: &MetalBuffer,
        k: &MetalBuffer,
        freq_factors: &MetalBuffer,
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        start_pos: f32,
        pos_step: f32,
        freq_base: f32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_rope_batch_neox_partial_with_freq_factors(
                encoder,
                q,
                k,
                freq_factors,
                n_rows,
                n_q_heads,
                n_kv_heads,
                head_dim,
                head_dim,
                start_pos,
                pos_step,
                freq_base,
            );
            Ok(())
        })
    }

    /// Per-head RMSNorm on GPU (standalone).
    pub fn per_head_rms_norm(
        &self,
        device: &MetalDevice,
        buf: &MetalBuffer,
        weight: &MetalBuffer,
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_per_head_rms_norm(encoder, buf, weight, n_heads, head_dim, eps);
            Ok(())
        })
    }

    /// Batched per-head RMSNorm on GPU (standalone).
    #[allow(clippy::too_many_arguments)]
    pub fn per_head_rms_norm_batch(
        &self,
        device: &MetalDevice,
        buf: &MetalBuffer,
        weight: &MetalBuffer,
        n_rows: u32,
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_per_head_rms_norm_batch(
                encoder, buf, weight, n_rows, n_heads, head_dim, eps,
            );
            Ok(())
        })
    }

    /// Per-head RMSNorm without a learned weight on GPU (standalone).
    pub fn per_head_rms_norm_no_weight(
        &self,
        device: &MetalDevice,
        buf: &MetalBuffer,
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_per_head_rms_norm_no_weight(encoder, buf, n_heads, head_dim, eps);
            Ok(())
        })
    }

    /// Batched per-head RMSNorm without a learned weight on GPU (standalone).
    #[allow(clippy::too_many_arguments)]
    pub fn per_head_rms_norm_no_weight_batch(
        &self,
        device: &MetalDevice,
        buf: &MetalBuffer,
        n_rows: u32,
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_per_head_rms_norm_no_weight_batch(
                encoder, buf, n_rows, n_heads, head_dim, eps,
            );
            Ok(())
        })
    }

    /// GELU(gate) * up on GPU (standalone).
    pub fn gelu_elementwise_mul(
        &self,
        device: &MetalDevice,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        n: u32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_gelu_elementwise_mul(encoder, gate, up, n);
            Ok(())
        })
    }

    /// SiLU(gate) * up on GPU (standalone).
    pub fn silu_elementwise_mul(
        &self,
        device: &MetalDevice,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        n: u32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_silu_elementwise_mul(encoder, gate, up, n);
            Ok(())
        })
    }

    /// a += b on GPU (standalone).
    pub fn elementwise_add(
        &self,
        device: &MetalDevice,
        a: &MetalBuffer,
        b: &MetalBuffer,
        n: u32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_elementwise_add(encoder, a, b, n);
            Ok(())
        })
    }

    /// KV append on GPU (standalone, for testing).
    pub fn kv_append(
        &self,
        device: &MetalDevice,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        dst_f16: bool,
        offset: u32,
        count: u32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_kv_append(encoder, src, dst, dst_f16, offset, count);
            Ok(())
        })
    }

    /// Gemma-style post-attention RMSNorm + residual add + RMSNorm (standalone).
    #[allow(clippy::too_many_arguments)]
    pub fn post_attn_norm_residual_add_rms_norm_out_batch(
        &self,
        device: &MetalDevice,
        hidden: &MetalBuffer,
        addend: &MetalBuffer,
        post_weight: &MetalBuffer,
        residual_weight: &MetalBuffer,
        norm_out: &MetalBuffer,
        n: u32,
        n_rows: u32,
        eps: f32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_post_attn_norm_residual_add_rms_norm_out_batch(
                encoder,
                hidden,
                addend,
                post_weight,
                residual_weight,
                norm_out,
                n,
                n_rows,
                eps,
            );
            Ok(())
        })
    }

    /// Gemma-style post-FFN RMSNorm + residual add + RMSNorm (standalone).
    #[allow(clippy::too_many_arguments)]
    pub fn post_ffn_norm_residual_add_rms_norm_out_batch(
        &self,
        device: &MetalDevice,
        hidden: &MetalBuffer,
        addend: &MetalBuffer,
        post_weight: &MetalBuffer,
        residual_weight: &MetalBuffer,
        norm_out: &MetalBuffer,
        n: u32,
        n_rows: u32,
        eps: f32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_post_ffn_norm_residual_add_rms_norm_out_batch(
                encoder,
                hidden,
                addend,
                post_weight,
                residual_weight,
                norm_out,
                n,
                n_rows,
                eps,
            );
            Ok(())
        })
    }
}
