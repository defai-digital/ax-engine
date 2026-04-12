use crate::backend::metal::MetalOps;
use crate::gguf::tensor::GgmlType;
use crate::model::config::ModelConfig;
use crate::model::weights::WeightStore;

#[derive(Clone, Copy, Debug)]
pub(crate) struct RoutedMoeResidentLayerKeys {
    pub(crate) router: usize,
    pub(crate) router_dtype: GgmlType,
    pub(crate) gate: usize,
    pub(crate) gate_dtype: GgmlType,
    pub(crate) up: usize,
    pub(crate) up_dtype: GgmlType,
    pub(crate) down: usize,
    pub(crate) down_dtype: GgmlType,
    pub(crate) n_expert: usize,
    pub(crate) n_expert_used: usize,
    pub(crate) expert_inter_dim: usize,
    pub(crate) gate_stride: usize,
    pub(crate) up_stride: usize,
    pub(crate) down_stride: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct RoutedMoeLayerSupport<'a> {
    pub(crate) router_raw: &'a [u8],
    pub(crate) router_dtype: GgmlType,
    pub(crate) gate_raw: &'a [u8],
    pub(crate) gate_dtype: GgmlType,
    pub(crate) up_raw: &'a [u8],
    pub(crate) up_dtype: GgmlType,
    pub(crate) down_raw: &'a [u8],
    pub(crate) down_dtype: GgmlType,
    pub(crate) n_expert: usize,
    pub(crate) n_expert_used: usize,
    pub(crate) expert_inter_dim: usize,
    pub(crate) gate_stride: usize,
    pub(crate) up_stride: usize,
    pub(crate) down_stride: usize,
}

pub(crate) fn moe_router_dtype_supported(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::F32 | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8_0
    )
}

pub(crate) fn moe_routed_expert_dtype_supported(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8_0
    )
}

pub(crate) fn routed_moe_expert_dtypes(
    weights: &WeightStore,
    prefix: &str,
) -> anyhow::Result<(GgmlType, GgmlType, GgmlType)> {
    let gate_name = format!("{prefix}.ffn_gate_exps.weight");
    let up_name = format!("{prefix}.ffn_up_exps.weight");
    let down_name = format!("{prefix}.ffn_down_exps.weight");
    let (_, gate_dtype) = weights.raw_with_dtype(&gate_name)?;
    let (_, up_dtype) = weights.raw_with_dtype(&up_name)?;
    let (_, down_dtype) = weights.raw_with_dtype(&down_name)?;
    Ok((gate_dtype, up_dtype, down_dtype))
}

pub(crate) fn qwen3moe_concurrent_decode_enabled_for_layout(
    gate_dtype: GgmlType,
    up_dtype: GgmlType,
    _down_dtype: GgmlType,
) -> bool {
    if let Some(enabled) = crate::model::shared::env_flag_override("AX_QWEN3MOE_CONCURRENT_DECODE")
    {
        return enabled;
    }

    // Shipped Qwen3-Coder routed stacks use Q4/Q5 gate+up with a Q6 down
    // projection, so key the overlap policy on the gate/up pair.
    gate_dtype == up_dtype && matches!(gate_dtype, GgmlType::Q4K | GgmlType::Q5K)
}

pub(crate) fn tensor_output_rows(weights: &WeightStore, name: &str) -> anyhow::Result<usize> {
    let info = weights.info(name)?;
    match info.shape.as_slice() {
        [_input_dim] => Ok(1),
        [_input_dim, output_dim, ..] => Ok(*output_dim as usize),
        [] => anyhow::bail!("{name} has empty shape"),
    }
}

pub(crate) fn routed_moe_layer_support<'a>(
    cfg: &ModelConfig,
    weights: &'a WeightStore,
    prefix: &str,
    dim: usize,
) -> anyhow::Result<Option<RoutedMoeLayerSupport<'a>>> {
    let router_name = format!("{prefix}.ffn_gate_inp.weight");
    if !weights.has(&router_name) {
        return Ok(None);
    }

    let (router_raw, router_dtype) = weights.raw_with_dtype(&router_name)?;
    if !moe_router_dtype_supported(router_dtype) {
        return Ok(None);
    }

    let n_expert = cfg
        .n_expert
        .unwrap_or(tensor_output_rows(weights, &router_name)? as u32) as usize;
    let n_expert_used = cfg.n_expert_used.unwrap_or(0) as usize;
    anyhow::ensure!(n_expert > 0, "{prefix} requires n_expert > 0");
    anyhow::ensure!(n_expert_used > 0, "{prefix} requires n_expert_used > 0");
    anyhow::ensure!(
        n_expert_used <= n_expert,
        "{prefix} n_expert_used ({n_expert_used}) > n_expert ({n_expert})"
    );

    let gate_name = format!("{prefix}.ffn_gate_exps.weight");
    let up_name = format!("{prefix}.ffn_up_exps.weight");
    let down_name = format!("{prefix}.ffn_down_exps.weight");
    let expert_inter_dim = tensor_output_rows(weights, &gate_name)?;
    let (gate_raw, gate_dtype) = weights.raw_with_dtype(&gate_name)?;
    let (up_raw, up_dtype) = weights.raw_with_dtype(&up_name)?;
    let (down_raw, down_dtype) = weights.raw_with_dtype(&down_name)?;
    if !moe_routed_expert_dtype_supported(gate_dtype)
        || !moe_routed_expert_dtype_supported(up_dtype)
        || !moe_routed_expert_dtype_supported(down_dtype)
    {
        return Ok(None);
    }

    let gate_stride =
        crate::model::moe_utils::expert_byte_stride(gate_dtype, expert_inter_dim * dim);
    let up_stride = crate::model::moe_utils::expert_byte_stride(up_dtype, expert_inter_dim * dim);
    let down_stride =
        crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);

    Ok(Some(RoutedMoeLayerSupport {
        router_raw,
        router_dtype,
        gate_raw,
        gate_dtype,
        up_raw,
        up_dtype,
        down_raw,
        down_dtype,
        n_expert,
        n_expert_used,
        expert_inter_dim,
        gate_stride,
        up_stride,
        down_stride,
    }))
}

pub(crate) fn build_routed_moe_resident_layer_keys(
    metal_ops: &MetalOps,
    cfg: &ModelConfig,
    weights: &WeightStore,
    prefix: &str,
    dim: usize,
) -> anyhow::Result<Option<RoutedMoeResidentLayerKeys>> {
    let Some(layer) = routed_moe_layer_support(cfg, weights, prefix, dim)? else {
        return Ok(None);
    };
    let router = metal_ops.ensure_moe_quant_cached(layer.router_raw);
    if layer.router_dtype == GgmlType::F32 {
        let router_buf = {
            let cache = metal_ops.lock_moe_weight_cache();
            cache
                .get(&router)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing Metal buffer for routed MoE router"))?
        };
        metal_ops.ensure_precomputed_f32_f16(&router_buf, layer.n_expert as u32, dim as u32)?;
    }

    Ok(Some(RoutedMoeResidentLayerKeys {
        router,
        router_dtype: layer.router_dtype,
        gate: metal_ops.ensure_moe_quant_cached(layer.gate_raw),
        gate_dtype: layer.gate_dtype,
        up: metal_ops.ensure_moe_quant_cached(layer.up_raw),
        up_dtype: layer.up_dtype,
        down: metal_ops.ensure_moe_quant_cached(layer.down_raw),
        down_dtype: layer.down_dtype,
        n_expert: layer.n_expert,
        n_expert_used: layer.n_expert_used,
        expert_inter_dim: layer.expert_inter_dim,
        gate_stride: layer.gate_stride,
        up_stride: layer.up_stride,
        down_stride: layer.down_stride,
    }))
}
