//! Qwen4-Exp (`qwen4_exp`, Qwen3.8-Flash-Next) family: owns the packed
//! gated-residual stream.
//!
//! The residual stream is `[1, seq, hc_count * hidden]` (4 packed streams of
//! 2560 in the reference pack); this family expands the token embedding into
//! it, threads it through every layer end-to-end (each layer is
//! `mixed_input` → branch → `inject_write_back` for both the attention and
//! MoE-FFN hyper-connections), and collapses it with the root-level
//! `hyper_connection_mixer` before `lm_head`. There is no `model.norm` and no
//! per-layer input/post layernorm — the hyper-connections replace them, and
//! the root mixer's grouped RMSNorm doubles as the final normalization. The
//! generic `layer_forward` dispatch in `model/mod.rs` deliberately does NOT
//! route here — qwen4_exp forward passes go through the dedicated top-level
//! helpers (E-wide only at the boundaries), exactly like DeepSeek V4.
//!
//! Layer kinds (0-indexed): PLE lives at the top of the layer listed in
//! `ple_layer_ids` (1-based; layer 1 in the reference pack) and needs the raw
//! token ids every forward; gated-delta linear-attention layers run the
//! shared stack (with the family-scoped sigmoid output gate — see
//! `linear_attention_output_gate_sigmoid`); layers where
//! `(i + 1) % full_attention_interval == 0` run QSA sparse full attention.
//!
//! Authoritative spec: `.internal/planning/qwen38-flash-next-support.md`
//! sections "Layer map", "B. PLE floor", "C. Hyper-connections", "E. Hybrid
//! layer details", "F. Router".

use mlx_sys::{MlxArray, MlxDtype, add, astype, reshape};

use super::super::shared::{
    Qwen4ExpGatedResidualWeights, inject_write_back, linear_attention_forward, mixed_input,
    moe_experts_forward, moe_experts_forward_with_shared, moe_router_qwen3, qw,
    qwen4_exp_ple_forward, qwen4_exp_qsa_forward, rms_norm_one_plus_gamma, shared_expert_forward,
};
use super::super::{ModelConfig, embed_tokens_arr};
use crate::kv_cache::MlxKVCache;
use crate::weights::{
    LayerWeights, ModelWeights, Qwen4ExpHyperConnectionWeights, Qwen4ExpMtpWeights,
};

/// Map one loaded hyper-connection site (`Qwen4ExpHyperConnectionWeights`,
/// loader field names) to the gated-residual math struct
/// (`Qwen4ExpGatedResidualWeights`, checkpoint tensor names). The tensors
/// themselves are refcount-cloned, not copied.
pub(crate) fn gated_residual(w: &Qwen4ExpHyperConnectionWeights) -> Qwen4ExpGatedResidualWeights {
    Qwen4ExpGatedResidualWeights {
        hc_norm: w.hc_norm.clone(),
        input_mix_weight_down: w.input_mix_down.weight.clone(),
        input_mix_weight_up: w.input_mix_up.weight.clone(),
        block_inject_weight: w.block_inject.as_ref().map(|gate| gate.weight.clone()),
    }
}

/// Ids handed to the PLE n-gram hash on the batch=1 text path: when the pack
/// declares a tokenizer pad id distinct from EOS, pad positions hash as EOS
/// (planning doc section B.7's conv-mask substitution — the text trunk has no
/// conv mask). Host-side over the id slice, never in the graph; borrowed when
/// no pad id is known, when pad == EOS, or when no pad is present.
pub(crate) fn ple_input_ids(
    input_ids: &[u32],
    pad_token_id: Option<u32>,
    eos_token_id: u32,
) -> std::borrow::Cow<'_, [u32]> {
    let Some(pad) = pad_token_id else {
        return std::borrow::Cow::Borrowed(input_ids);
    };
    if pad == eos_token_id || !input_ids.contains(&pad) {
        return std::borrow::Cow::Borrowed(input_ids);
    }
    std::borrow::Cow::Owned(
        input_ids
            .iter()
            .map(|&id| if id == pad { eos_token_id } else { id })
            .collect(),
    )
}

/// One qwen4_exp layer over the packed residual stream:
/// `[1, seq, hc_count * hidden]` → `[1, seq, hc_count * hidden]`.
///
/// Order per layer (planning doc "Layer map"): PLE injection first (PLE layer
/// only — the raw `input_ids` feed the EOS-segment-aware n-gram hash and the
/// conv/hash state rings persist on `cache`), then the attention
/// hyper-connection (gated-delta linear attention or QSA sparse full
/// attention), then the MoE hyper-connection (qwen3 softmax router + routed
/// experts + sigmoid-gated shared expert).
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    packed_stream: &MlxArray,
    input_ids: &[u32],
    positions: &[[i32; 3]],
    cache: &mut MlxKVCache,
    layer_idx: usize,
) -> MlxArray {
    let Some(q4e) = cfg.qwen4_exp.as_ref() else {
        unreachable!("qwen4_exp family layer requires ModelConfig.qwen4_exp");
    };
    let Some(family) = w.qwen4_exp.as_ref() else {
        unreachable!("qwen4_exp family layer requires Qwen4ExpLayerWeights");
    };
    let (hc_count, hidden, eps) = (q4e.hc_count, cfg.hidden_size, cfg.rms_norm_eps);
    let seq = packed_stream.shape()[1] as usize;
    assert_eq!(
        input_ids.len(),
        seq,
        "qwen4_exp trunk input_ids must cover every new token"
    );

    // PLE floor: `H += PLE(H, input_ids)` at the top of the PLE layer, before
    // the attention hyper-connection, adding to all streams at once.
    let packed_owned;
    let packed_stream = if let Some(ple_weights) = family.ple.as_ref() {
        let ple_state = cache.qwen4_exp_ple_state();
        let ple = match qwen4_exp_ple_forward(
            packed_stream,
            input_ids,
            None, // conv_mask: trunk tokens are unpadded
            q4e.eos_token_id,
            q4e.ngram_size,
            q4e.heads_per_ngram,
            hc_count,
            eps,
            ple_weights,
            ple_state,
        ) {
            Ok(ple) => ple,
            // PLE can still fail after a successful load_weights: lazy shard
            // I/O on first gather, stale sidecar mappings, gather geometry,
            // or dequant errors. The trunk returns MlxArray, so panic with
            // the real error (a Result-typed trunk is a follow-up).
            Err(error) => {
                panic!("qwen4_exp PLE forward failed at layer {layer_idx}: {error}")
            }
        };
        cache.set_qwen4_exp_ple_state(ple.conv_ring, ple.token_ring);
        packed_owned = add(packed_stream, &ple.output, None);
        &packed_owned
    } else {
        packed_stream
    };

    // Attention hyper-connection branch.
    let attn_hc = gated_residual(&family.attn_hyper_connection);
    let branch_in = mixed_input(packed_stream, &attn_hc, hc_count, hidden, eps);
    let attn_out = if cfg.is_linear_attention_layer(layer_idx) {
        linear_attention_forward(cfg, w, &branch_in, cache, layer_idx, false, false)
    } else {
        let Some(indexer) = family.indexer.as_ref() else {
            unreachable!("qwen4_exp QSA layer {layer_idx} requires indexer weights");
        };
        qwen4_exp_qsa_forward(&branch_in, w, indexer, cache, layer_idx, positions, cfg)
    };
    let packed_stream =
        inject_write_back(packed_stream, &attn_out, &attn_hc, hc_count, hidden, eps);

    // MoE hyper-connection branch (every qwen4_exp layer is MoE): qwen3
    // softmax router (top-k, norm_topk_prob) + routed experts + shared expert.
    let mlp_hc = gated_residual(&family.mlp_hyper_connection);
    let branch_in = mixed_input(&packed_stream, &mlp_hc, hc_count, hidden, eps);
    let (top_k_indices, top_k_weights) = moe_router_qwen3(cfg, w, &branch_in);
    let ffn_out = if w.shared_gate_proj.is_some() {
        let shared_out = shared_expert_forward(cfg, w, &branch_in);
        moe_experts_forward_with_shared(
            cfg,
            w,
            &branch_in,
            &top_k_indices,
            &top_k_weights,
            &shared_out,
        )
    } else {
        moe_experts_forward(cfg, w, &branch_in, &top_k_indices, &top_k_weights)
    };
    inject_write_back(&packed_stream, &ffn_out, &mlp_hc, hc_count, hidden, eps)
}

/// MTP input fusion (planning doc section D.2): blend the previous token's
/// embedding with the main model's packed PRE-MIXER hidden into the draft
/// block's packed stream.
///
/// `packed_hidden` is `[1, seq, hc_count * hidden]` (the trunk output before
/// the root `hyper_connection_mixer`); `prev_token_arr` holds the `seq` token
/// ids that follow each packed row (uint32 `[seq]` or `[1]`). The hidden side
/// is normalized with ONE global RMS over the full packed width (not the
/// hyper-connection grouped norm), then reshaped to streams and projected
/// per stream with the shared `fc_hidden` matrix; the token side is embedded
/// from the SHARED table, RMS-normed over `hidden`, and projected with
/// `fc_embedding`. `fused_i = fc_embedding(norm(e)) + fc_hidden(norm(H)_i)`.
///
/// Both norms run through the family's `1 + γ` helper on raw HF-delta weights
/// (no loader-side shift). Returns `[1, seq, hc_count * hidden]` in the
/// packed stream dtype.
pub(crate) fn mtp_fuse_inputs(
    packed_hidden: &MlxArray,
    prev_token_arr: &MlxArray,
    mtp: &Qwen4ExpMtpWeights,
    weights: &ModelWeights,
    cfg: &ModelConfig,
) -> MlxArray {
    let Some(q4e) = cfg.qwen4_exp.as_ref() else {
        unreachable!("qwen4_exp MTP fusion requires ModelConfig.qwen4_exp");
    };
    let shape = packed_hidden.shape();
    assert_eq!(
        shape.len(),
        3,
        "qwen4_exp MTP fusion hidden must be [batch, seq, hc*hidden]"
    );
    let (batch, seq, width) = (shape[0], shape[1], shape[2]);
    let (hc, hidden) = (q4e.hc_count as i32, cfg.hidden_size as i32);
    assert_eq!(
        width,
        hc * hidden,
        "qwen4_exp MTP fusion width must equal hc_count * hidden_size"
    );

    // Token side: shared embedding table → RMSNorm (1+γ) over `hidden` →
    // fc_embedding.
    let embed = embed_tokens_arr(prev_token_arr, &weights.token_embedding, cfg.hidden_size);
    let embed = astype(&embed, MlxDtype::Bfloat16, None);
    let embed_shape = embed.shape();
    assert_eq!(
        embed_shape,
        vec![batch, seq, hidden],
        "qwen4_exp MTP fusion token ids must cover every packed row"
    );
    let e_normed = rms_norm_one_plus_gamma(&embed, &mtp.pre_fc_norm_embedding, cfg.rms_norm_eps);
    let e_normed = astype(&e_normed, MlxDtype::Bfloat16, None);
    let e_fc = qw(&e_normed, &mtp.fc_embedding);

    // Hidden side: ONE global RMSNorm (1+γ) across the full packed width,
    // then reshape to streams and apply fc_hidden per stream (one [hidden,
    // hidden] matrix shared by all streams — not a [hc*hidden] dense).
    let h_normed =
        rms_norm_one_plus_gamma(packed_hidden, &mtp.pre_fc_norm_hidden, cfg.rms_norm_eps);
    let h_normed = astype(&h_normed, MlxDtype::Bfloat16, None);
    let h_streams = reshape(&h_normed, &[batch, seq, hc, hidden], None);
    let h_flat = reshape(&h_streams, &[batch * seq * hc, hidden], None);
    let h_fc = qw(&h_flat, &mtp.fc_hidden);
    let h_fc = reshape(&h_fc, &[batch, seq, hc, hidden], None);

    let e_tiled = reshape(&e_fc, &[batch, seq, 1, hidden], None);
    let fused = add(&e_tiled, &h_fc, None);
    let fused = reshape(&fused, &[batch, seq, hc * hidden], None);
    astype(&fused, packed_hidden.dtype(), None)
}

/// The qwen4_exp MTP draft block over the fused packed stream:
/// `[1, seq, hc_count * hidden]` → `[1, seq, hc_count * hidden]`.
///
/// This is a dedicated composition of the landed family components, NOT a
/// reuse of [`layer_forward`]: the MTP block is always a full-attention QSA
/// layer (`layer_types = ["full_attention"]`) and has no PLE floor, so the
/// trunk's `is_linear_attention_layer` dispatch and PLE injection must not
/// run. Both hyper-connections and the qwen3 softmax router with routed
/// experts plus the sigmoid-gated shared expert compose exactly like a trunk
/// QSA layer. The QSA K/V lands in the caller's DEDICATED one-slot draft
/// cache at `layer_idx = 0`; the cache's PLE state is never touched.
pub(crate) fn mtp_block_forward(
    cfg: &ModelConfig,
    mtp: &Qwen4ExpMtpWeights,
    packed_stream: &MlxArray,
    positions: &[[i32; 3]],
    cache: &mut MlxKVCache,
) -> MlxArray {
    let Some(q4e) = cfg.qwen4_exp.as_ref() else {
        unreachable!("qwen4_exp MTP block requires ModelConfig.qwen4_exp");
    };
    let layer = mtp.layer.as_ref();
    let Some(family) = layer.qwen4_exp.as_ref() else {
        unreachable!("qwen4_exp MTP block requires Qwen4ExpLayerWeights");
    };
    let Some(indexer) = family.indexer.as_ref() else {
        unreachable!("qwen4_exp MTP block requires QSA indexer weights");
    };
    let (hc_count, hidden, eps) = (q4e.hc_count, cfg.hidden_size, cfg.rms_norm_eps);

    // Attention hyper-connection branch (always QSA).
    let attn_hc = gated_residual(&family.attn_hyper_connection);
    let branch_in = mixed_input(packed_stream, &attn_hc, hc_count, hidden, eps);
    let attn_out = qwen4_exp_qsa_forward(&branch_in, layer, indexer, cache, 0, positions, cfg);
    let packed = inject_write_back(packed_stream, &attn_out, &attn_hc, hc_count, hidden, eps);

    // MoE hyper-connection branch (the MTP block always carries the shared
    // expert): qwen3 softmax router + routed experts + shared expert.
    let mlp_hc = gated_residual(&family.mlp_hyper_connection);
    let branch_in = mixed_input(&packed, &mlp_hc, hc_count, hidden, eps);
    let (top_k_indices, top_k_weights) = moe_router_qwen3(cfg, layer, &branch_in);
    let shared_out = shared_expert_forward(cfg, layer, &branch_in);
    let ffn_out = moe_experts_forward_with_shared(
        cfg,
        layer,
        &branch_in,
        &top_k_indices,
        &top_k_weights,
        &shared_out,
    );
    inject_write_back(&packed, &ffn_out, &mlp_hc, hc_count, hidden, eps)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::model::ModelConfig;
    use crate::weights::load_weights;
    use ax_engine_core::{
        NativeModelArtifacts, NativeTensorDataType, NativeTensorQuantization, NativeTensorRole,
        NativeTensorSpec,
    };
    use mlx_sys::{MlxDtype, argmax, eval};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    // Tiny synthetic qwen4_exp geometry: 4 layers (0/1/2 gated-delta linear,
    // 3 QSA via full_attention_interval 4), PLE on 0-indexed layer 1
    // (`ple_layer_ids [2]` is 1-based), hidden 64, hc 4 → stream 256.
    const HIDDEN: i32 = 64;
    const HC: i32 = 4;
    const STREAM: i32 = HIDDEN * HC;
    const LOWRANK: i32 = 8;
    const VOCAB: i32 = 32;
    const N_HEADS: i32 = 4;
    const HEAD_DIM: i32 = 32;
    const EXPERTS: i32 = 4;
    const INTER: i32 = 32;
    const SHARD_ROWS: i32 = 16;
    const SHARD_DIM: i32 = 32;
    const LAYER_COUNT: usize = 4;
    const LINEAR_LAYERS: [u32; 3] = [0, 1, 2];
    const QSA_LAYER: u32 = 3;
    const PLE_LAYER: u32 = 1;
    const EOS: u32 = 30;
    /// Tokenizer pad id declared by the fixture manifest (distinct from EOS).
    const PAD: u32 = 31;
    const EPS: f64 = 1e-6;

    struct Q4eTensor {
        name: String,
        file: String,
        file_dtype: &'static str,
        file_shape: Vec<i32>,
        bytes: Vec<u8>,
        role: NativeTensorRole,
        layer_index: Option<u32>,
        manifest_dtype: NativeTensorDataType,
        source_quantized: bool,
        quantization: Option<NativeTensorQuantization>,
        logical_shape: Vec<u64>,
        in_manifest: bool,
    }

    fn fill_bytes(len: usize, bytes_per: usize, fill: impl Fn(usize) -> Vec<u8>) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(len * bytes_per);
        for index in 0..len {
            bytes.extend_from_slice(&fill(index));
        }
        bytes
    }

    fn bf16_bytes(len: usize, fill: f32) -> Vec<u8> {
        let bits = (fill.to_bits() >> 16) as u16;
        fill_bytes(len, 2, |_| bits.to_le_bytes().to_vec())
    }

    fn f32_bytes(len: usize, fill: f32) -> Vec<u8> {
        fill_bytes(len, 4, |_| fill.to_le_bytes().to_vec())
    }

    fn u32_bytes(len: usize, fill: u32) -> Vec<u8> {
        fill_bytes(len, 4, |_| fill.to_le_bytes().to_vec())
    }

    fn i64_bytes(values: &[i64]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn dense(
        name: &str,
        shape: &[i32],
        role: NativeTensorRole,
        layer: Option<u32>,
        fill: f32,
    ) -> Q4eTensor {
        let len: usize = shape.iter().map(|dim| *dim as usize).product();
        Q4eTensor {
            name: name.to_string(),
            file: "model.safetensors".to_string(),
            file_dtype: "BF16",
            file_shape: shape.to_vec(),
            bytes: bf16_bytes(len, fill),
            role,
            layer_index: layer,
            manifest_dtype: NativeTensorDataType::Bf16,
            source_quantized: false,
            quantization: None,
            logical_shape: shape.iter().map(|dim| *dim as u64).collect(),
            in_manifest: true,
        }
    }

    fn f32_tensor(
        name: &str,
        shape: &[i32],
        role: NativeTensorRole,
        layer: Option<u32>,
        fill: f32,
    ) -> Q4eTensor {
        let len: usize = shape.iter().map(|dim| *dim as usize).product();
        Q4eTensor {
            name: name.to_string(),
            file: "model.safetensors".to_string(),
            file_dtype: "F32",
            file_shape: shape.to_vec(),
            bytes: f32_bytes(len, fill),
            role,
            layer_index: layer,
            manifest_dtype: NativeTensorDataType::F32,
            source_quantized: false,
            quantization: None,
            logical_shape: shape.iter().map(|dim| *dim as u64).collect(),
            in_manifest: true,
        }
    }

    fn i64_tensor(
        name: &str,
        values: &[i64],
        role: NativeTensorRole,
        layer: Option<u32>,
    ) -> Q4eTensor {
        Q4eTensor {
            name: name.to_string(),
            file: "model.safetensors".to_string(),
            file_dtype: "I64",
            file_shape: vec![values.len() as i32],
            bytes: i64_bytes(values),
            role,
            layer_index: layer,
            manifest_dtype: NativeTensorDataType::I64,
            source_quantized: false,
            quantization: None,
            logical_shape: vec![values.len() as u64],
            in_manifest: true,
        }
    }

    /// One affine-quantized linear (`.weight` U32 packed + `.scales`/`.biases`
    /// BF16 sidecars). Only the `.weight` enters the manifest; the runtime
    /// resolves the sidecars by name convention.
    fn quant(
        name: &str,
        logical: &[i32],
        group_size: u32,
        bits: u32,
        role: NativeTensorRole,
        layer: Option<u32>,
        file: &str,
    ) -> (Q4eTensor, Q4eTensor, Q4eTensor) {
        let rank = logical.len();
        let in_dim = logical[rank - 1] as usize;
        let packed_cols = in_dim * bits as usize / 32;
        let groups = in_dim / group_size as usize;
        let mut weight_shape = logical.to_vec();
        weight_shape[rank - 1] = packed_cols as i32;
        let mut sidecar_shape = logical.to_vec();
        sidecar_shape[rank - 1] = groups as i32;
        let weight_len: usize = weight_shape.iter().map(|dim| *dim as usize).product();
        let sidecar_len: usize = sidecar_shape.iter().map(|dim| *dim as usize).product();
        let base = name.strip_suffix(".weight").unwrap_or(name);
        let sidecar = |suffix: &str, fill: f32| Q4eTensor {
            name: format!("{base}.{suffix}"),
            file: file.to_string(),
            file_dtype: "BF16",
            file_shape: sidecar_shape.clone(),
            bytes: bf16_bytes(sidecar_len, fill),
            role: NativeTensorRole::Other,
            layer_index: layer,
            manifest_dtype: NativeTensorDataType::Bf16,
            source_quantized: false,
            quantization: None,
            logical_shape: sidecar_shape.iter().map(|dim| *dim as u64).collect(),
            in_manifest: false,
        };
        (
            Q4eTensor {
                name: name.to_string(),
                file: file.to_string(),
                file_dtype: "U32",
                file_shape: weight_shape.clone(),
                bytes: u32_bytes(weight_len, 0x0101_0101),
                role,
                layer_index: layer,
                manifest_dtype: NativeTensorDataType::U32,
                source_quantized: true,
                quantization: Some(NativeTensorQuantization {
                    mode: "affine".to_string(),
                    group_size,
                    bits,
                }),
                logical_shape: weight_shape.iter().map(|dim| *dim as u64).collect(),
                in_manifest: true,
            },
            sidecar("scales", 1.0),
            sidecar("biases", 0.0),
        )
    }

    fn push_quant(
        tensors: &mut Vec<Q4eTensor>,
        name: &str,
        logical: &[i32],
        group_size: u32,
        bits: u32,
        role: NativeTensorRole,
        layer: Option<u32>,
    ) {
        let (weight, scales, biases) = quant(
            name,
            logical,
            group_size,
            bits,
            role,
            layer,
            "model.safetensors",
        );
        tensors.extend([weight, scales, biases]);
    }

    fn layer_prefix(layer: u32) -> String {
        format!("language_model.model.layers.{layer}")
    }

    fn push_hyper_connections(tensors: &mut Vec<Q4eTensor>, layer: u32) {
        let prefix = layer_prefix(layer);
        for site in ["attn_hyper_connection", "mlp_hyper_connection"] {
            tensors.push(dense(
                &format!("{prefix}.{site}.hc_norm.weight"),
                &[STREAM],
                NativeTensorRole::Other,
                Some(layer),
                1.0,
            ));
            tensors.push(dense(
                &format!("{prefix}.{site}.input_mix_weight_down.weight"),
                &[LOWRANK, STREAM],
                NativeTensorRole::Other,
                Some(layer),
                0.5,
            ));
            tensors.push(dense(
                &format!("{prefix}.{site}.input_mix_weight_up.weight"),
                &[STREAM, LOWRANK],
                NativeTensorRole::Other,
                Some(layer),
                0.5,
            ));
            tensors.push(dense(
                &format!("{prefix}.{site}.block_inject_weight.weight"),
                &[HC, STREAM],
                NativeTensorRole::Other,
                Some(layer),
                0.5,
            ));
        }
    }

    fn push_moe(tensors: &mut Vec<Q4eTensor>, layer: u32) {
        let prefix = layer_prefix(layer);
        push_quant(
            tensors,
            &format!("{prefix}.mlp.gate.weight"),
            &[EXPERTS, HIDDEN],
            64,
            8,
            NativeTensorRole::FfnGateInp,
            Some(layer),
        );
        for (proj, role) in [
            ("gate_proj", NativeTensorRole::FfnGateExps),
            ("up_proj", NativeTensorRole::FfnUpExps),
        ] {
            push_quant(
                tensors,
                &format!("{prefix}.mlp.switch_mlp.{proj}.weight"),
                &[EXPERTS, INTER, HIDDEN],
                64,
                6,
                role,
                Some(layer),
            );
        }
        push_quant(
            tensors,
            &format!("{prefix}.mlp.switch_mlp.down_proj.weight"),
            &[EXPERTS, HIDDEN, INTER],
            32,
            6,
            NativeTensorRole::FfnDownExps,
            Some(layer),
        );
        for (proj, role) in [
            ("gate_proj", NativeTensorRole::FfnSharedExpertGate),
            ("up_proj", NativeTensorRole::FfnSharedExpertUp),
        ] {
            push_quant(
                tensors,
                &format!("{prefix}.mlp.shared_expert.{proj}.weight"),
                &[INTER, HIDDEN],
                64,
                6,
                role,
                Some(layer),
            );
        }
        push_quant(
            tensors,
            &format!("{prefix}.mlp.shared_expert.down_proj.weight"),
            &[HIDDEN, INTER],
            32,
            6,
            NativeTensorRole::FfnSharedExpertDown,
            Some(layer),
        );
        push_quant(
            tensors,
            &format!("{prefix}.mlp.shared_expert_gate.weight"),
            &[1, HIDDEN],
            64,
            8,
            NativeTensorRole::FfnSharedExpertGateInp,
            Some(layer),
        );
    }

    fn push_linear_attention(tensors: &mut Vec<Q4eTensor>, layer: u32) {
        let prefix = layer_prefix(layer);
        push_quant(
            tensors,
            &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
            &[STREAM, HIDDEN],
            64,
            6,
            NativeTensorRole::LinearAttentionInProjQkv,
            Some(layer),
        );
        push_quant(
            tensors,
            &format!("{prefix}.linear_attn.in_proj_z.weight"),
            &[128, HIDDEN],
            64,
            6,
            NativeTensorRole::LinearAttentionInProjZ,
            Some(layer),
        );
        for proj in ["in_proj_a", "in_proj_b"] {
            let role = if proj == "in_proj_a" {
                NativeTensorRole::LinearAttentionInProjA
            } else {
                NativeTensorRole::LinearAttentionInProjB
            };
            push_quant(
                tensors,
                &format!("{prefix}.linear_attn.{proj}.weight"),
                &[4, HIDDEN],
                64,
                6,
                role,
                Some(layer),
            );
        }
        tensors.push(dense(
            &format!("{prefix}.linear_attn.conv1d.weight"),
            &[STREAM, 4, 1],
            NativeTensorRole::LinearAttentionConv1d,
            Some(layer),
            0.5,
        ));
        tensors.push(f32_tensor(
            &format!("{prefix}.linear_attn.dt_bias"),
            &[4],
            NativeTensorRole::LinearAttentionDtBias,
            Some(layer),
            0.1,
        ));
        tensors.push(f32_tensor(
            &format!("{prefix}.linear_attn.A_log"),
            &[4],
            NativeTensorRole::LinearAttentionALog,
            Some(layer),
            0.1,
        ));
        tensors.push(dense(
            &format!("{prefix}.linear_attn.norm.weight"),
            &[HEAD_DIM],
            NativeTensorRole::LinearAttentionNorm,
            Some(layer),
            1.0,
        ));
        push_quant(
            tensors,
            &format!("{prefix}.linear_attn.out_proj.weight"),
            &[HIDDEN, 128],
            64,
            6,
            NativeTensorRole::LinearAttentionOutProj,
            Some(layer),
        );
    }

    fn push_qsa_attention(tensors: &mut Vec<Q4eTensor>, layer: u32) {
        let prefix = layer_prefix(layer);
        push_quant(
            tensors,
            &format!("{prefix}.self_attn.q_proj.weight"),
            &[2 * N_HEADS * HEAD_DIM, HIDDEN],
            64,
            6,
            NativeTensorRole::AttentionQ,
            Some(layer),
        );
        for (proj, role) in [
            ("k_proj", NativeTensorRole::AttentionK),
            ("v_proj", NativeTensorRole::AttentionV),
        ] {
            push_quant(
                tensors,
                &format!("{prefix}.self_attn.{proj}.weight"),
                &[HEAD_DIM, HIDDEN],
                64,
                6,
                role,
                Some(layer),
            );
        }
        push_quant(
            tensors,
            &format!("{prefix}.self_attn.o_proj.weight"),
            &[HIDDEN, N_HEADS * HEAD_DIM],
            64,
            6,
            NativeTensorRole::AttentionO,
            Some(layer),
        );
        tensors.push(dense(
            &format!("{prefix}.self_attn.q_norm.weight"),
            &[HEAD_DIM],
            NativeTensorRole::AttentionQNorm,
            Some(layer),
            1.0,
        ));
        tensors.push(dense(
            &format!("{prefix}.self_attn.k_norm.weight"),
            &[HEAD_DIM],
            NativeTensorRole::AttentionKNorm,
            Some(layer),
            1.0,
        ));
        push_quant(
            tensors,
            &format!("{prefix}.self_attn.indexer.index_qk_proj.weight"),
            &[5 * HEAD_DIM, HIDDEN],
            64,
            6,
            NativeTensorRole::Other,
            Some(layer),
        );
        tensors.push(dense(
            &format!("{prefix}.self_attn.indexer.q_layernorm.weight"),
            &[HEAD_DIM],
            NativeTensorRole::Other,
            Some(layer),
            1.0,
        ));
        tensors.push(dense(
            &format!("{prefix}.self_attn.indexer.k_layernorm.weight"),
            &[HEAD_DIM],
            NativeTensorRole::Other,
            Some(layer),
            1.0,
        ));
    }

    /// PLE resident tensors (the n-gram table shards live in their own
    /// file — see `shard_tensors`). Two heads (heads_per_ngram 1, bigram +
    /// trigram) of dim 32 → hidden 64; tiny vocab sizes keep the 2-shard
    /// table gatherable.
    fn push_ple(tensors: &mut Vec<Q4eTensor>, layer: u32) {
        let prefix = format!("{}.ple", layer_prefix(layer));
        push_quant(
            tensors,
            &format!("{prefix}.key_proj.weight"),
            &[STREAM, HIDDEN],
            32,
            8,
            NativeTensorRole::Other,
            Some(layer),
        );
        push_quant(
            tensors,
            &format!("{prefix}.value_proj.weight"),
            &[HIDDEN, HIDDEN],
            32,
            8,
            NativeTensorRole::Other,
            Some(layer),
        );
        for norm in ["norm_key", "norm_query", "norm_conv"] {
            tensors.push(dense(
                &format!("{prefix}.{norm}.weight"),
                &[STREAM],
                NativeTensorRole::Other,
                Some(layer),
                1.0,
            ));
        }
        tensors.push(dense(
            &format!("{prefix}.conv1d.weight"),
            &[STREAM, 4, 1],
            NativeTensorRole::Other,
            Some(layer),
            0.5,
        ));
        let embedding = format!("{prefix}.ple_embedding");
        tensors.push(i64_tensor(
            &format!("{embedding}.layer_multipliers"),
            &[23703573157769, 20109073645365, 8052911324071],
            NativeTensorRole::Other,
            Some(layer),
        ));
        tensors.push(i64_tensor(
            &format!("{embedding}.ngram_heads_vocab_sizes"),
            &[SHARD_ROWS as i64; 2],
            NativeTensorRole::Other,
            Some(layer),
        ));
        tensors.push(i64_tensor(
            &format!("{embedding}.ngram_heads_offsets"),
            &[0, SHARD_ROWS as i64],
            NativeTensorRole::Other,
            Some(layer),
        ));
    }

    fn shard_tensors() -> Vec<Q4eTensor> {
        let embedding = format!("{}.ple.ple_embedding", layer_prefix(PLE_LAYER));
        let mut tensors = Vec::new();
        for shard in 0..2 {
            let (weight, scales, biases) = quant(
                &format!("{embedding}.ngram_embedding.shards.{shard}.weight"),
                &[SHARD_ROWS, SHARD_DIM],
                32,
                8,
                NativeTensorRole::Other,
                Some(PLE_LAYER),
                "ple_shards.safetensors",
            );
            tensors.extend([weight, scales, biases]);
        }
        tensors
    }

    fn write_file(dir: &Path, file: &str, tensors: &[&Q4eTensor]) -> HashMap<String, (u64, u64)> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        let mut spans = HashMap::new();
        for tensor in tensors {
            let start = data.len() as u64;
            data.extend_from_slice(&tensor.bytes);
            let end = data.len() as u64;
            header.insert(
                tensor.name.clone(),
                serde_json::json!({
                    "dtype": tensor.file_dtype,
                    "shape": tensor.file_shape,
                    "data_offsets": [start, end],
                }),
            );
            spans.insert(tensor.name.clone(), (start, end));
        }
        let header_bytes =
            serde_json::to_vec(&serde_json::Value::Object(header)).expect("header serializes");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&data);
        std::fs::write(dir.join(file), &bytes).expect("fixture safetensors should write");
        spans
    }

    fn unique_test_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ax-qwen4-exp-trunk-test-{label}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock")
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("fixture directory should create");
        dir
    }

    /// Write a forward-consistent 4-layer synthetic qwen4_exp pack and load
    /// the artifacts through the REAL manifest validation + weight loader.
    fn synthetic_artifacts(label: &str) -> (PathBuf, NativeModelArtifacts) {
        synthetic_artifacts_full(label, None)
    }

    /// The 31-tensor MTP sidecar at the tiny fixture geometry (bare `mtp.*`
    /// names, BF16 dense, packed `experts.gate_up_proj [E, 2I, H]`).
    fn mtp_sidecar_tensors() -> Vec<Q4eTensor> {
        let layer = "mtp.layers.0";
        let mut tensors = Vec::new();
        let mut push = |name: String, shape: &[i32], fill: f32| {
            tensors.push(dense(&name, shape, NativeTensorRole::Other, None, fill));
        };
        // Fusion norms stay raw HF-delta (small fills): the family norm
        // helpers own the `1 + γ` gain, so the loader must not shift these.
        push(
            "mtp.pre_fc_norm_embedding.weight".to_string(),
            &[HIDDEN],
            0.007_812_5,
        );
        push(
            "mtp.pre_fc_norm_hidden.weight".to_string(),
            &[STREAM],
            0.015_625,
        );
        push(
            "mtp.fc_embedding.weight".to_string(),
            &[HIDDEN, HIDDEN],
            0.031_25,
        );
        push(
            "mtp.fc_hidden.weight".to_string(),
            &[HIDDEN, HIDDEN],
            0.007_812_5,
        );
        for site in ["attn_hyper_connection", "mlp_hyper_connection"] {
            push(format!("{layer}.{site}.hc_norm.weight"), &[STREAM], 0.0625);
            push(
                format!("{layer}.{site}.input_mix_weight_down.weight"),
                &[LOWRANK, STREAM],
                0.125,
            );
            push(
                format!("{layer}.{site}.input_mix_weight_up.weight"),
                &[STREAM, LOWRANK],
                0.125,
            );
            push(
                format!("{layer}.{site}.block_inject_weight.weight"),
                &[HC, STREAM],
                0.25,
            );
        }
        let attn = format!("{layer}.self_attn");
        push(
            format!("{attn}.q_proj.weight"),
            &[2 * N_HEADS * HEAD_DIM, HIDDEN],
            0.031_25,
        );
        push(format!("{attn}.k_proj.weight"), &[HEAD_DIM, HIDDEN], 0.0625);
        push(format!("{attn}.v_proj.weight"), &[HEAD_DIM, HIDDEN], 0.0625);
        push(
            format!("{attn}.o_proj.weight"),
            &[HIDDEN, N_HEADS * HEAD_DIM],
            0.031_25,
        );
        push(format!("{attn}.q_norm.weight"), &[HEAD_DIM], 0.125);
        push(format!("{attn}.k_norm.weight"), &[HEAD_DIM], 0.125);
        push(
            format!("{attn}.indexer.index_qk_proj.weight"),
            &[5 * HEAD_DIM, HIDDEN],
            0.0625,
        );
        push(
            format!("{attn}.indexer.q_layernorm.weight"),
            &[HEAD_DIM],
            0.25,
        );
        push(
            format!("{attn}.indexer.k_layernorm.weight"),
            &[HEAD_DIM],
            0.25,
        );
        let mlp = format!("{layer}.mlp");
        push(format!("{mlp}.gate.weight"), &[EXPERTS, HIDDEN], 0.0625);
        push(
            format!("{mlp}.experts.gate_up_proj.weight"),
            &[EXPERTS, 2 * INTER, HIDDEN],
            0.031_25,
        );
        push(
            format!("{mlp}.experts.down_proj.weight"),
            &[EXPERTS, HIDDEN, INTER],
            0.031_25,
        );
        push(
            format!("{mlp}.shared_expert.gate_proj.weight"),
            &[INTER, HIDDEN],
            0.0625,
        );
        push(
            format!("{mlp}.shared_expert.up_proj.weight"),
            &[INTER, HIDDEN],
            0.0625,
        );
        push(
            format!("{mlp}.shared_expert.down_proj.weight"),
            &[HIDDEN, INTER],
            0.0625,
        );
        push(
            format!("{mlp}.shared_expert_gate.weight"),
            &[1, HIDDEN],
            0.125,
        );
        let mixer = "mtp.hyper_connection_mixer";
        push(format!("{mixer}.hc_norm.weight"), &[STREAM], 0.0625);
        push(
            format!("{mixer}.input_mix_weight_down.weight"),
            &[LOWRANK, STREAM],
            0.125,
        );
        push(
            format!("{mixer}.input_mix_weight_up.weight"),
            &[STREAM, LOWRANK],
            0.125,
        );
        tensors
    }

    /// [`synthetic_artifacts`] plus a full 31-tensor MTP sidecar and the
    /// predictor block declared in the manifest.
    fn synthetic_artifacts_with_mtp(label: &str) -> (PathBuf, NativeModelArtifacts) {
        synthetic_artifacts_full(label, Some(mtp_sidecar_tensors()))
    }

    /// [`synthetic_artifacts`] plus an INCOMPLETE sidecar (the four fusion
    /// tensors only): the loader must reject it and leave the trunk working.
    fn synthetic_artifacts_with_incomplete_mtp(label: &str) -> (PathBuf, NativeModelArtifacts) {
        let tensors: Vec<Q4eTensor> = mtp_sidecar_tensors().into_iter().take(4).collect();
        synthetic_artifacts_full(label, Some(tensors))
    }

    fn synthetic_artifacts_full(
        label: &str,
        mtp_sidecar: Option<Vec<Q4eTensor>>,
    ) -> (PathBuf, NativeModelArtifacts) {
        let dir = unique_test_dir(label);
        let mut tensors = Vec::new();
        push_quant(
            &mut tensors,
            "language_model.model.embed_tokens.weight",
            &[VOCAB, HIDDEN],
            32,
            8,
            NativeTensorRole::TokenEmbedding,
            None,
        );
        push_quant(
            &mut tensors,
            "language_model.lm_head.weight",
            &[VOCAB, HIDDEN],
            64,
            8,
            NativeTensorRole::LmHead,
            None,
        );
        let mixer = "language_model.model.hyper_connection_mixer";
        tensors.push(dense(
            &format!("{mixer}.hc_norm.weight"),
            &[STREAM],
            NativeTensorRole::Other,
            None,
            1.0,
        ));
        tensors.push(dense(
            &format!("{mixer}.input_mix_weight_down.weight"),
            &[LOWRANK, STREAM],
            NativeTensorRole::Other,
            None,
            0.5,
        ));
        tensors.push(dense(
            &format!("{mixer}.input_mix_weight_up.weight"),
            &[STREAM, LOWRANK],
            NativeTensorRole::Other,
            None,
            0.5,
        ));
        for layer in 0..LAYER_COUNT as u32 {
            push_hyper_connections(&mut tensors, layer);
            push_moe(&mut tensors, layer);
        }
        for layer in LINEAR_LAYERS {
            push_linear_attention(&mut tensors, layer);
        }
        push_qsa_attention(&mut tensors, QSA_LAYER);
        push_ple(&mut tensors, PLE_LAYER);
        let shards = shard_tensors();

        let resident_refs: Vec<&Q4eTensor> = tensors.iter().collect();
        let mut spans = write_file(&dir, "model.safetensors", &resident_refs);
        let shard_refs: Vec<&Q4eTensor> = shards.iter().collect();
        spans.extend(write_file(&dir, "ple_shards.safetensors", &shard_refs));
        let has_mtp_sidecar = mtp_sidecar.is_some();
        if let Some(sidecar) = mtp_sidecar.as_ref() {
            let sidecar_refs: Vec<&Q4eTensor> = sidecar.iter().collect();
            write_file(&dir, "mtp.safetensors", &sidecar_refs);
        }

        let specs: Vec<NativeTensorSpec> = tensors
            .iter()
            .chain(shards.iter())
            .filter(|tensor| tensor.in_manifest)
            .map(|tensor| {
                let span = spans
                    .get(&tensor.name)
                    .copied()
                    .expect("fixture tensor span");
                NativeTensorSpec {
                    name: tensor.name.clone(),
                    role: tensor.role,
                    layer_index: tensor.layer_index,
                    dtype: tensor.manifest_dtype,
                    source_tensor_type: None,
                    source_quantized: tensor.source_quantized,
                    quantization: tensor.quantization.clone(),
                    quantized_source: None,
                    shape: tensor.logical_shape.clone(),
                    file: tensor.file.clone().into(),
                    offset_bytes: span.0,
                    length_bytes: span.1 - span.0,
                }
            })
            .collect();
        let mut q4e_json = serde_json::json!({
            "hc_count": HC,
            "hc_lowrank": LOWRANK,
            "indexer_budget": 8,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": HEAD_DIM,
            "indexer_kv_heads": 1,
            "indexer_n_heads": 4,
            "ngram_size": 3,
            "heads_per_ngram": 1,
            "split_ngram_parts": 2,
            "ple_conv_kernel_size": 4,
            "ple_embed_dim": HIDDEN,
            "ple_layer_ids": [PLE_LAYER + 1],
            "output_gate_type": "sigmoid",
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": true,
            "shared_expert_intermediate_size": INTER,
            "eos_token_id": EOS,
            "pad_token_id": PAD
        });
        if has_mtp_sidecar {
            q4e_json["mtp"] = serde_json::json!({
                "num_hidden_layers": 1,
                "hybrid": true,
                "layer_types": ["full_attention"],
                "use_dedicated_embeddings": false
            });
        }
        let mut manifest: ax_engine_core::NativeModelManifest =
            serde_json::from_value(serde_json::json!({
                "schema_version": ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION,
                "model_family": "qwen4_exp",
                "tensor_format": "safetensors",
                "layer_count": LAYER_COUNT,
                "hidden_size": HIDDEN,
                "attention_head_count": N_HEADS,
                "attention_head_dim": HEAD_DIM,
                "kv_head_count": 1,
                "vocab_size": VOCAB,
                "attn_output_gate": true,
                "partial_rotary_factor": 0.25,
                "rope_theta": 10_000_000,
                "rms_norm_eps": EPS,
                "linear_attention": {
                    "full_attention_interval": 4,
                    "num_key_heads": 2,
                    "key_head_dim": HEAD_DIM,
                    "num_value_heads": 4,
                    "value_head_dim": HEAD_DIM,
                    "conv_kernel_dim": 4
                },
                "moe": {
                    "expert_count": EXPERTS,
                    "experts_per_token": 2,
                    "expert_intermediate_size": INTER
                },
                "qwen4_exp": q4e_json,
                "tensors": []
            }))
            .expect("qwen4_exp trunk fixture manifest should deserialize");
        manifest.tensors = specs;
        std::fs::write(
            dir.join("model-manifest.json"),
            serde_json::to_vec_pretty(&manifest).expect("manifest should serialize"),
        )
        .expect("manifest should write");
        let artifacts =
            NativeModelArtifacts::from_dir(&dir).expect("fixture manifest should validate");
        (dir, artifacts)
    }

    struct TrunkFixture {
        dir: PathBuf,
        cfg: ModelConfig,
        weights: crate::weights::ModelWeights,
    }

    fn trunk_fixture(label: &str) -> TrunkFixture {
        let (dir, artifacts) = synthetic_artifacts(label);
        let weights = load_weights(&artifacts).expect("synthetic qwen4_exp weights should load");
        let cfg = ModelConfig::from_manifest(artifacts.manifest());
        TrunkFixture { dir, cfg, weights }
    }

    impl Drop for TrunkFixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.dir).ok();
        }
    }

    struct MtpFixture {
        dir: PathBuf,
        cfg: ModelConfig,
        weights: crate::weights::ModelWeights,
    }

    fn mtp_fixture(label: &str) -> MtpFixture {
        let (dir, artifacts) = synthetic_artifacts_with_mtp(label);
        let weights =
            load_weights(&artifacts).expect("synthetic qwen4_exp+MTP weights should load");
        let cfg = ModelConfig::from_manifest(artifacts.manifest());
        MtpFixture { dir, cfg, weights }
    }

    impl Drop for MtpFixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.dir).ok();
        }
    }

    fn assert_finite_logits(logits: &MlxArray, what: &str) {
        eval(&[logits]);
        let f32_logits = mlx_sys::astype(logits, MlxDtype::Float32, None);
        eval(&[&f32_logits]);
        assert!(
            f32_logits.data_f32().iter().all(|v| v.is_finite()),
            "{what} must be finite"
        );
    }

    #[test]
    fn trunk_prefill_then_decode_carries_state() {
        let fixture = trunk_fixture("prefill-decode");
        let TrunkFixture { cfg, weights, .. } = &fixture;
        let mut cache = MlxKVCache::new(LAYER_COUNT);

        // Prefill a short sequence through the REAL trunk entry (dispatch path).
        let prompt = [1u32, 5, 9, 13];
        let logits = crate::model::forward(cfg, weights, &prompt, &mut cache, 0);
        cache.advance(prompt.len());
        assert_eq!(logits.shape(), vec![VOCAB]);
        assert_finite_logits(&logits, "prefill logits");

        // PLE state after prefill: the conv ring holds the 9-row tail plus the
        // 4 prompt rows (all inside the rewind window); the hash history is
        // the token ring's last two entries (context_len = ngram_size - 1 = 2).
        let (conv_ring, token_ring) = cache.qwen4_exp_ple_state();
        let conv_ring = conv_ring.expect("PLE conv ring must persist after prefill");
        assert_eq!(conv_ring.shape(), vec![1, 9 + 4, STREAM]);
        assert_eq!(conv_ring.dtype(), MlxDtype::Float32);
        let token_ring = token_ring.expect("PLE token ring must persist");
        assert_eq!(&token_ring[token_ring.len() - 2..], &[9, 13]);

        // QSA cache on layer 3 covers the prompt; one indexer block completed.
        let (k, v) = cache
            .qwen4_exp_qsa_kv(QSA_LAYER as usize)
            .expect("QSA layer must hold K/V after prefill");
        assert_eq!(k.shape(), vec![1, 1, 4, HEAD_DIM]);
        assert_eq!(v.shape(), vec![1, 1, 4, HEAD_DIM]);
        assert_eq!(cache.qwen4_exp_qsa_committed_blocks(QSA_LAYER as usize), 1);
        let block_k = cache
            .qwen4_exp_qsa_block_keys(QSA_LAYER as usize)
            .expect("one pooled indexer block must be committed");
        assert_eq!(block_k.shape(), vec![1, 1, 1, HEAD_DIM]);

        // Gated-delta state on the linear layers.
        for layer in LINEAR_LAYERS {
            let (conv, recurrent) = cache.linear_state(layer as usize);
            assert!(conv.is_some(), "linear layer {layer} must hold conv state");
            let recurrent = recurrent
                .unwrap_or_else(|| panic!("linear layer {layer} must hold recurrent state"));
            assert_eq!(recurrent.shape(), vec![1, 4, HEAD_DIM, HEAD_DIM]);
            assert_eq!(recurrent.dtype(), MlxDtype::Float32);
        }

        // Decode 3 known tokens through the same trunk (seq == 1 path).
        for (step, token) in [2u32, 4, 6].iter().enumerate() {
            let offset = cache.seq_len();
            let logits = crate::model::forward(cfg, weights, &[*token], &mut cache, offset);
            cache.advance(1);
            assert_eq!(logits.shape(), vec![VOCAB], "decode step {step}");
            assert_finite_logits(&logits, "decode logits");
        }

        // Cross-boundary carry-over: QSA K/V now spans 7 tokens; the indexer
        // block count is still 1 (7 / 4); the token ring's last two entries
        // are the last two fed tokens [4, 6].
        let (k, _) = cache
            .qwen4_exp_qsa_kv(QSA_LAYER as usize)
            .expect("QSA K/V must survive the prefill→decode boundary");
        assert_eq!(k.shape(), vec![1, 1, 7, HEAD_DIM]);
        assert_eq!(cache.qwen4_exp_qsa_committed_blocks(QSA_LAYER as usize), 1);
        let (_, token_ring) = cache.qwen4_exp_ple_state();
        let token_ring = token_ring.expect("PLE token ring");
        assert_eq!(&token_ring[token_ring.len() - 2..], &[4, 6]);

        // One lazy argmax-chained decode step (the direct-pipeline pattern):
        // the trunk reads the lazy token on the host for the PLE hash.
        let offset = cache.seq_len();
        let logits = crate::model::forward(cfg, weights, &[8u32], &mut cache, offset);
        cache.advance(1);
        let pending = argmax(&logits, None);
        let offset = cache.seq_len();
        let logits =
            crate::model::forward_lazy_single_argmax(cfg, weights, &pending, &mut cache, offset);
        cache.advance(1);
        assert_eq!(logits.shape(), vec![1, 1, VOCAB]);
        assert_finite_logits(&logits, "lazy decode logits");
    }

    #[test]
    fn split_prefill_decode_matches_single_forward() {
        let fixture = trunk_fixture("split-equivalence");
        let TrunkFixture { cfg, weights, .. } = &fixture;
        let ids = [3u32, 7, 11, 17, 23];

        // One-shot: the whole sequence in a single trunk forward.
        let mut cache_full = MlxKVCache::new(LAYER_COUNT);
        let logits_full = crate::model::forward(cfg, weights, &ids, &mut cache_full, 0);
        cache_full.advance(ids.len());

        // Split: prefill 3, then two single-token decode steps.
        let mut cache_split = MlxKVCache::new(LAYER_COUNT);
        let _ = crate::model::forward(cfg, weights, &ids[..3], &mut cache_split, 0);
        cache_split.advance(3);
        let _ = crate::model::forward(cfg, weights, &ids[3..4], &mut cache_split, 3);
        cache_split.advance(1);
        let logits_split = crate::model::forward(cfg, weights, &ids[4..5], &mut cache_split, 4);
        cache_split.advance(1);

        eval(&[&logits_full, &logits_split]);
        let full = logits_full.data_f32();
        let split = logits_split.data_f32();
        assert_eq!(full.len(), split.len());
        let argmax_of = |data: &[f32]| {
            data.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index)
        };
        assert_eq!(
            argmax_of(full),
            argmax_of(split),
            "split prefill+decode must preserve the greedy token"
        );
        let max_diff = full
            .iter()
            .zip(split.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.05,
            "split prefill+decode drifted from the single forward by {max_diff}"
        );
    }

    #[test]
    fn qwen4_exp_dispatches_to_the_dedicated_trunk() {
        // Registry route: qwen4_exp is a DedicatedTrunk family, so the generic
        // per-layer dispatch never sees its layers.
        assert_eq!(
            ax_engine_core::resolve_layer_forward_route("qwen4_exp"),
            Some(ax_engine_core::LayerForwardRoute::Qwen4Exp)
        );
        assert_eq!(
            ax_engine_core::LayerForwardRoute::Qwen4Exp.trunk_style(),
            ax_engine_core::TrunkStyle::DedicatedTrunk
        );
        // `forward` on the qwen4_exp config only returns when the dispatch
        // routed to the family trunk (the per-layer `layer_forward` arm fails
        // loudly otherwise) — covered end-to-end by the fixtures above.
        let fixture = trunk_fixture("dispatch");
        let TrunkFixture { cfg, weights, .. } = &fixture;
        assert!(cfg.qwen4_exp.is_some());
        let mut cache = MlxKVCache::new(LAYER_COUNT);
        let logits = crate::model::forward(cfg, weights, &[1u32], &mut cache, 0);
        cache.advance(1);
        assert_eq!(logits.shape(), vec![VOCAB]);
        assert_finite_logits(&logits, "dispatch logits");
    }

    #[test]
    fn trim_rewinds_ple_state_to_the_exact_prefix() {
        let fixture = trunk_fixture("ple-trim");
        let TrunkFixture { cfg, weights, .. } = &fixture;
        let prompt = [1u32, 5, 9, 13];

        // Path A: prefill 4 + decode [2, 4, 6], then a 2-token draft rollback
        // to 5 tokens (the rejected draft tail).
        let mut cache_a = MlxKVCache::new(LAYER_COUNT);
        let _ = crate::model::forward(cfg, weights, &prompt, &mut cache_a, 0);
        cache_a.advance(prompt.len());
        for token in [2u32, 4, 6] {
            let offset = cache_a.seq_len();
            let _ = crate::model::forward(cfg, weights, &[token], &mut cache_a, offset);
            cache_a.advance(1);
        }
        assert!(cache_a.trim_to(5));

        // Path B: prefill 4 + decode [2] only — the exact retained prefix.
        let mut cache_b = MlxKVCache::new(LAYER_COUNT);
        let _ = crate::model::forward(cfg, weights, &prompt, &mut cache_b, 0);
        cache_b.advance(prompt.len());
        let _ = crate::model::forward(cfg, weights, &[2u32], &mut cache_b, 4);
        cache_b.advance(1);

        let (ring_a, tokens_a) = cache_a.qwen4_exp_ple_state();
        let (ring_b, tokens_b) = cache_b.qwen4_exp_ple_state();
        let (ring_a, ring_b) = (
            ring_a.expect("trimmed conv ring"),
            ring_b.expect("prefix conv ring"),
        );
        eval(&[ring_a, ring_b]);
        assert_eq!(ring_a.shape(), ring_b.shape());
        let (data_a, data_b) = (ring_a.data_f32(), ring_b.data_f32());
        assert!(
            data_a.iter().zip(data_b.iter()).all(|(a, b)| a == b),
            "trimmed PLE conv ring must bitwise-equal the exact-prefix state"
        );
        assert_eq!(
            tokens_a.expect("trimmed token ring"),
            tokens_b.expect("prefix token ring"),
            "trimmed PLE token ring must equal the exact-prefix state"
        );
    }

    #[test]
    fn ple_hashes_pad_ids_as_eos() {
        let fixture = trunk_fixture("ple-pad-eos");
        let TrunkFixture { cfg, weights, .. } = &fixture;

        // The fixture manifest declares pad_token_id = PAD (distinct from
        // EOS): the PLE hash must see EOS at the pad position (planning doc
        // section B.7's substitution on the mask-free text path).
        let mut cache = MlxKVCache::new(LAYER_COUNT);
        let _ = crate::model::forward(cfg, weights, &[1u32, PAD, 9], &mut cache, 0);
        cache.advance(3);
        let (_, token_ring) = cache.qwen4_exp_ple_state();
        let token_ring = token_ring.expect("PLE token ring");
        assert_eq!(
            &token_ring[token_ring.len() - 3..],
            &[1, EOS as i64, 9],
            "pad positions must hash as EOS"
        );

        // The effective ids are exactly what feeding EOS directly produces.
        let mut reference = MlxKVCache::new(LAYER_COUNT);
        let _ = crate::model::forward(cfg, weights, &[1u32, EOS, 9], &mut reference, 0);
        reference.advance(3);
        let (_, reference_ring) = reference.qwen4_exp_ple_state();
        assert_eq!(token_ring, reference_ring.expect("reference token ring"));
    }

    #[test]
    fn ple_input_ids_substitutes_pad_with_eos() {
        let ids = [1u32, 31, 9];
        let substituted = ple_input_ids(&ids, Some(31), 30);
        assert_eq!(substituted.as_ref(), &[1, 30, 9]);
        // Borrowed (untouched) when no pad id is known, when pad == EOS, or
        // when no pad is present.
        assert!(matches!(
            ple_input_ids(&ids, None, 30),
            std::borrow::Cow::Borrowed(_)
        ));
        assert!(matches!(
            ple_input_ids(&ids, Some(30), 30),
            std::borrow::Cow::Borrowed(_)
        ));
        assert!(matches!(
            ple_input_ids(&[1u32, 2, 3], Some(31), 30),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    // ── Phase-2 MTP draft head ───────────────────────────────────────────────

    /// Packed hidden `[1, seq, hc*hidden]` with a distinct constant per
    /// stream (bf16-exact values) so global and grouped norms provably differ.
    fn packed_with_stream_constants(stream_values: [f32; 4], seq: i32) -> MlxArray {
        let mut data = Vec::with_capacity((seq * STREAM) as usize);
        for _ in 0..seq {
            for &value in &stream_values {
                for _ in 0..HIDDEN {
                    data.push(value);
                }
            }
        }
        let arr = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[1, seq, STREAM],
            MlxDtype::Float32,
        );
        astype(&arr, MlxDtype::Bfloat16, None)
    }

    fn read_f32(arr: &MlxArray) -> Vec<f32> {
        let flat = mlx_sys::contiguous(&astype(arr, MlxDtype::Float32, None), None);
        eval(&[&flat]);
        flat.data_f32().to_vec()
    }

    #[test]
    fn mtp_sidecar_round_trips_through_real_loader() {
        let fixture = mtp_fixture("loader-round-trip");
        let MtpFixture { cfg, weights, .. } = &fixture;
        assert!(
            cfg.qwen4_exp
                .as_ref()
                .and_then(|q4e| q4e.mtp.as_ref())
                .is_some_and(|mtp| mtp.num_hidden_layers == 1)
        );
        // Skip-guard regression: the Qwen dense-head loader stays closed.
        assert!(
            weights.mtp.is_none(),
            "no Qwen dense MTP head for qwen4_exp"
        );

        let mtp = weights
            .qwen4_exp_mtp
            .as_ref()
            .expect("complete 31-tensor sidecar must attach through the real loader");
        assert_eq!(mtp.max_depth, 1);
        assert_eq!(mtp.pre_fc_norm_embedding.shape(), vec![HIDDEN]);
        assert_eq!(mtp.pre_fc_norm_hidden.shape(), vec![STREAM]);
        assert_eq!(mtp.fc_embedding.weight.shape(), vec![HIDDEN, HIDDEN]);
        assert_eq!(mtp.fc_embedding.bits, 32, "BF16 dense sidecar projection");
        assert!(mtp.mixer.block_inject.is_none(), "MTP mixer has no inject");
        let family = mtp
            .layer
            .qwen4_exp
            .as_ref()
            .expect("MTP block family weights");
        assert!(family.indexer.is_some(), "MTP block is a QSA layer");
        assert!(family.ple.is_none(), "MTP block has no PLE floor");
        // Packed `experts.gate_up_proj [E, 2I, H]` split on axis 1 at load.
        assert_eq!(
            mtp.layer.gate_exps.as_ref().map(|w| w.weight.shape()),
            Some(vec![EXPERTS, INTER, HIDDEN])
        );
        assert_eq!(
            mtp.layer.up_exps.as_ref().map(|w| w.weight.shape()),
            Some(vec![EXPERTS, INTER, HIDDEN])
        );
        assert_eq!(
            mtp.layer.down_exps.as_ref().map(|w| w.weight.shape()),
            Some(vec![EXPERTS, HIDDEN, INTER])
        );
        // Raw HF-delta norms are NOT shifted (+1.0 lives in the norm helpers).
        let norm = read_f32(&mtp.pre_fc_norm_hidden);
        let mean = norm.iter().map(|v| v.abs()).sum::<f32>() / norm.len() as f32;
        assert!(
            (mean - 0.015_625).abs() < 0.01,
            "norm must stay raw (mean {mean}); a +1.0 shift would read ~1.0"
        );
    }

    #[test]
    fn mtp_incomplete_sidecar_attaches_nothing_and_trunk_still_forwards() {
        let (dir, artifacts) = synthetic_artifacts_with_incomplete_mtp("incomplete");
        let weights = load_weights(&artifacts).expect("trunk loads with a rejected sidecar");
        let cfg = ModelConfig::from_manifest(artifacts.manifest());
        assert!(
            weights.qwen4_exp_mtp.is_none(),
            "an incomplete sidecar must not yield a partial MTP head"
        );
        assert!(weights.mtp.is_none());
        let mut cache = MlxKVCache::new(LAYER_COUNT);
        let logits = crate::model::forward(&cfg, &weights, &[1u32, 5, 9], &mut cache, 0);
        cache.advance(3);
        assert_eq!(logits.shape(), vec![VOCAB]);
        assert_finite_logits(&logits, "trunk logits with a rejected sidecar");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn mtp_fuse_inputs_matches_cpu_reference_and_uses_global_norm() {
        let fixture = mtp_fixture("fusion");
        let MtpFixture { cfg, weights, .. } = &fixture;
        let mtp = weights.qwen4_exp_mtp.as_ref().expect("MTP attached");
        let q4e = cfg.qwen4_exp.as_ref().expect("qwen4_exp config");
        let eps = cfg.rms_norm_eps;
        let seq = 2i32;
        let stream_values = [0.25f32, 0.5, 0.75, 1.0];
        let packed = packed_with_stream_constants(stream_values, seq);
        let ids =
            MlxArray::from_raw_data([3u32, 5].as_ptr() as *const u8, 8, &[seq], MlxDtype::Uint32);

        // Sanity on the fixture's shared embedding table: 8-bit packed
        // 0x01010101 with scale 1.0 / bias 0.0 dequantizes to all 1.0 rows.
        let embed = embed_tokens_arr(&ids, &weights.token_embedding, cfg.hidden_size);
        let embed_data = read_f32(&embed);
        assert!(
            embed_data.iter().all(|v| (*v - 1.0).abs() < 1e-3),
            "fixture embedding rows must dequantize to 1.0"
        );

        // Global-vs-grouped distinction on the raw norm: grouped RMS (per
        // 64-dim stream) normalizes every constant stream to the same value;
        // ONE global RMS over the full 256 keeps the streams distinct.
        let global = rms_norm_one_plus_gamma(&packed, &mtp.pre_fc_norm_hidden, eps);
        let grouped = crate::model::shared::grouped_rms_norm(
            &packed,
            &mtp.pre_fc_norm_hidden,
            q4e.hc_count,
            cfg.hidden_size,
            eps,
        );
        let global_data = read_f32(&global);
        let grouped_data = read_f32(&grouped);
        let max_diff = global_data
            .iter()
            .zip(grouped_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 0.1,
            "global and grouped norms must differ on spread streams (diff {max_diff})"
        );

        // CPU reference: e = [1; 64] → rms(1+γ) → fc_e; H → ONE global rms
        // (1+γ) over 256 → reshape 4×64 → fc_h per stream; fused_i = e + h_i.
        let gamma_e = 1.0 + 0.007_812_5_f64;
        let gamma_h = 1.0 + 0.015_625_f64;
        let e_n = gamma_e; // rms([1; 64]) == 1
        let e_fc = (HIDDEN as f64) * e_n * 0.031_25_f64;
        let sum_sqr = stream_values
            .iter()
            .map(|v| (HIDDEN as f64) * (*v as f64) * (*v as f64))
            .sum::<f64>();
        let global_rms = (sum_sqr / (STREAM as f64) + f64::from(eps)).sqrt();
        let fused = mtp_fuse_inputs(&packed, &ids, mtp, weights, cfg);
        assert_eq!(fused.shape(), vec![1, seq, STREAM]);
        let fused_data = read_f32(&fused);
        // bf16 output rounding: one ulp at the ~2.0-2.8 output magnitude is
        // 0.015625; the norm division adds sub-ulp error. atol covers both.
        let atol = 0.02f32;
        for row in 0..(seq as usize) {
            for (i, &v) in stream_values.iter().enumerate() {
                let h_n = (v as f64) / global_rms * gamma_h;
                let h_fc = (HIDDEN as f64) * h_n * 0.007_812_5_f64;
                let expect = (e_fc + h_fc) as f32;
                for d in 0..(HIDDEN as usize) {
                    let got = fused_data[row * (STREAM as usize) + i * (HIDDEN as usize) + d];
                    assert!(
                        (got - expect).abs() < atol,
                        "fusion stream {i} row {row}: got {got}, expect {expect} (global RMS reference)"
                    );
                }
            }
            // The per-stream outputs must stay distinct (grouped fusion would
            // collapse them to one value).
            let s0 = fused_data[row * (STREAM as usize)];
            let s3 = fused_data[row * (STREAM as usize) + 3 * (HIDDEN as usize)];
            assert!(
                (s3 - s0).abs() > 0.1,
                "fusion must keep the global-norm per-stream distinction (s0 {s0}, s3 {s3})"
            );
        }
    }

    #[test]
    fn mtp_block_forward_matches_manual_composition_without_ple_side_effects() {
        let fixture = mtp_fixture("block-composition");
        let MtpFixture { cfg, weights, .. } = &fixture;
        let mtp = weights.qwen4_exp_mtp.as_ref().expect("MTP attached");
        let q4e = cfg.qwen4_exp.as_ref().expect("qwen4_exp config");
        let eps = cfg.rms_norm_eps;
        let packed = packed_with_stream_constants([0.25, 0.5, 0.75, 1.0], 2);
        let positions = [[0i32; 3], [1i32; 3]];

        let mut cache_block = MlxKVCache::new(1);
        let via_block = mtp_block_forward(cfg, mtp, &packed, &positions, &mut cache_block);

        // Manual composition of the same landed components.
        let mut cache_manual = MlxKVCache::new(1);
        let layer = mtp.layer.as_ref();
        let family = layer.qwen4_exp.as_ref().expect("family");
        let indexer = family.indexer.as_ref().expect("indexer");
        let via_manual = {
            let attn_hc = gated_residual(&family.attn_hyper_connection);
            let branch_in = mixed_input(&packed, &attn_hc, q4e.hc_count, cfg.hidden_size, eps);
            let attn_out = qwen4_exp_qsa_forward(
                &branch_in,
                layer,
                indexer,
                &mut cache_manual,
                0,
                &positions,
                cfg,
            );
            let packed_after_attn = inject_write_back(
                &packed,
                &attn_out,
                &attn_hc,
                q4e.hc_count,
                cfg.hidden_size,
                eps,
            );
            let mlp_hc = gated_residual(&family.mlp_hyper_connection);
            let branch_in = mixed_input(
                &packed_after_attn,
                &mlp_hc,
                q4e.hc_count,
                cfg.hidden_size,
                eps,
            );
            let (top_k_indices, top_k_weights) = moe_router_qwen3(cfg, layer, &branch_in);
            let shared_out = shared_expert_forward(cfg, layer, &branch_in);
            let ffn_out = moe_experts_forward_with_shared(
                cfg,
                layer,
                &branch_in,
                &top_k_indices,
                &top_k_weights,
                &shared_out,
            );
            inject_write_back(
                &packed_after_attn,
                &ffn_out,
                &mlp_hc,
                q4e.hc_count,
                cfg.hidden_size,
                eps,
            )
        };

        assert_eq!(via_block.shape(), vec![1, 2, STREAM]);
        assert_eq!(
            read_f32(&via_block),
            read_f32(&via_manual),
            "mtp_block_forward must equal the manual component composition"
        );

        // The block writes QSA layer 0 of the DEDICATED cache only, and has
        // no PLE side effects (the MTP block has no PLE floor). `seq_len`
        // advances at the head level, same as the trunk's per-layer contract.
        cache_block.advance(2);
        let (k, v) = cache_block
            .qwen4_exp_qsa_kv(0)
            .expect("block must append QSA K/V at layer 0");
        assert_eq!(k.shape(), vec![1, 1, 2, HEAD_DIM]);
        assert_eq!(v.shape(), vec![1, 1, 2, HEAD_DIM]);
        assert!(cache_block.qwen4_exp_qsa_kv(1).is_none());
        let (conv_ring, token_ring) = cache_block.qwen4_exp_ple_state();
        assert!(conv_ring.is_none(), "MTP cache must hold no PLE conv ring");
        assert!(
            token_ring.is_none(),
            "MTP cache must hold no PLE token ring"
        );
        // A main-trunk cache is never touched by the block.
        let main_cache = MlxKVCache::new(LAYER_COUNT);
        assert!(main_cache.qwen4_exp_qsa_kv(QSA_LAYER as usize).is_none());
    }

    #[test]
    fn mtp_logits_uses_shared_lm_head() {
        let fixture = mtp_fixture("logits");
        let MtpFixture { cfg, weights, .. } = &fixture;
        let mtp = weights.qwen4_exp_mtp.as_ref().expect("MTP attached");
        let q4e = cfg.qwen4_exp.as_ref().expect("qwen4_exp config");
        let packed = packed_with_stream_constants([0.25, 0.5, 0.75, 1.0], 1);

        let logits = crate::mtp::qwen4_exp_mtp_hidden_to_logits(&packed, mtp, weights, cfg);
        assert_eq!(logits.shape(), vec![VOCAB]);
        assert_finite_logits(&logits, "MTP draft logits");

        // Reference: MTP mixer collapse → the SHARED target lm_head (the
        // sidecar carries no dedicated head).
        let mixer = gated_residual(&mtp.mixer);
        let hidden = crate::model::shared::mixer_output(
            &packed,
            &mixer,
            q4e.hc_count,
            cfg.hidden_size,
            cfg.rms_norm_eps,
        );
        let reference = qw(&hidden, &weights.lm_head);
        let reference = astype(&reference, MlxDtype::Float32, None);
        let reference = reshape(&reference, &[VOCAB], None);
        assert_eq!(
            read_f32(&logits),
            read_f32(&reference),
            "draft logits must come from the shared lm_head over the MTP mixer output"
        );
    }

    #[test]
    fn mtp_packed_verify_matches_singleton_direct_argmax() {
        use crate::model::{
            forward_all_positions, forward_argmax, qwen4_exp_forward_all_positions_with_packed,
        };
        let fixture = mtp_fixture("packed-verify");
        let MtpFixture { cfg, weights, .. } = &fixture;
        let prompt = [1u32, 5, 9, 13];

        // (a) Singleton probe: packed-path argmax == production forward_argmax.
        let mut pack_cache = MlxKVCache::new(LAYER_COUNT);
        let _ = crate::model::forward(cfg, weights, &prompt, &mut pack_cache, 0);
        pack_cache.advance(prompt.len());
        let mut argmax_cache = MlxKVCache::new(LAYER_COUNT);
        let _ = crate::model::forward(cfg, weights, &prompt, &mut argmax_cache, 0);
        argmax_cache.advance(prompt.len());
        let offset = pack_cache.seq_len();
        // Greedy accept decisions use the ArgmaxOnly (native-dtype) logit
        // form — the same arithmetic as production `forward_argmax`.
        let (packed_logits, packed_h) = qwen4_exp_forward_all_positions_with_packed(
            cfg,
            weights,
            &[7u32],
            &mut pack_cache,
            offset,
            true,
        );
        assert_eq!(
            packed_logits.dtype(),
            MlxDtype::Bfloat16,
            "ArgmaxOnly mode must keep the native lm_head dtype (no f32 cast)"
        );
        let direct_logits = forward_argmax(cfg, weights, &[7u32], &mut argmax_cache, offset);
        let pack_pred = argmax(&packed_logits, None);
        let direct_pred = argmax(&direct_logits, None);
        eval(&[&pack_pred, &direct_pred, &packed_h]);
        assert_eq!(
            pack_pred.data_u32()[0],
            direct_pred.data_u32()[0],
            "singleton packed verify must match production forward_argmax"
        );
        assert_eq!(
            packed_h.shape(),
            vec![1, 1, STREAM],
            "packed residual must keep the pre-mixer width hc*hidden"
        );

        // (b) Multi-row packed verify matches the generic all-positions arm.
        let mut packed_cache = MlxKVCache::new(LAYER_COUNT);
        let (packed_all, _) = qwen4_exp_forward_all_positions_with_packed(
            cfg,
            weights,
            &prompt,
            &mut packed_cache,
            0,
            false,
        );
        assert_eq!(
            packed_all.dtype(),
            MlxDtype::Float32,
            "sampled-path mode keeps the f32 logit surface"
        );
        let mut generic_cache = MlxKVCache::new(LAYER_COUNT);
        let generic_all = forward_all_positions(cfg, weights, &prompt, &mut generic_cache, 0);
        let packed_pred = argmax(&packed_all, None);
        let generic_pred = argmax(&generic_all, None);
        eval(&[&packed_pred, &generic_pred]);
        assert_eq!(
            packed_pred.data_u32(),
            generic_pred.data_u32(),
            "packed verify rows must match the generic all-positions forward"
        );
    }

    #[test]
    fn mtp_end_to_end_draft_verify_matches_pure_greedy() {
        use crate::model::qwen4_exp_forward_all_positions_with_packed;
        use crate::sampling::Xorshift64;
        let fixture = mtp_fixture("e2e");
        let MtpFixture { cfg, weights, .. } = &fixture;
        let mtp = weights.qwen4_exp_mtp.as_ref().expect("MTP attached");
        let prompt = [1u32, 5, 9, 13];
        let prompt_len = prompt.len();

        // Prefill through the packed entry; sample the first token greedily.
        let mut main_cache = MlxKVCache::new(LAYER_COUNT);
        let (logits_all, packed) = qwen4_exp_forward_all_positions_with_packed(
            cfg,
            weights,
            &prompt,
            &mut main_cache,
            0,
            false,
        );
        main_cache.advance(prompt_len);
        let last = (prompt_len - 1) as i32;
        let last_logits =
            mlx_sys::slice(&logits_all, &[last, 0], &[last + 1, VOCAB], &[1, 1], None);
        let first_tok_arr = argmax(&last_logits, None);
        eval(&[&first_tok_arr, &packed]);
        let first_tok = first_tok_arr.data_u32()[0];

        // Warm the dedicated one-slot MTP QSA cache: each packed row pairs
        // with the token that FOLLOWS it (established Qwen/V4 contract).
        let mut mtp_cache = MlxKVCache::new(1);
        let mut history: Vec<u32> = prompt[1..].to_vec();
        history.push(first_tok);
        crate::mtp::qwen4_exp_mtp_warmup_cache(
            mtp,
            &packed,
            &history,
            weights,
            &mut mtp_cache,
            cfg,
            0,
        );
        assert_eq!(mtp_cache.seq_len(), prompt_len);

        // Draft at depth 1; a larger cap still yields exactly one token.
        let draft_hidden = mlx_sys::slice(
            &packed,
            &[0, last, 0],
            &[1, last + 1, STREAM],
            &[1, 1, 1],
            None,
        );
        let draft_hidden = reshape(&draft_hidden, &[1, 1, STREAM], None);
        let mut rng = Xorshift64::new(1);
        let (draft, log_probs, _dist, added, _margins) =
            crate::mtp::qwen4_exp_mtp_draft_tokens_gated(
                weights,
                cfg,
                &draft_hidden,
                first_tok,
                &mut mtp_cache,
                Some(8),
                &mut rng,
                0.0,
                1.0,
            );
        assert_eq!(
            draft.len(),
            1,
            "depth is hard-capped at the one shipped block"
        );
        assert_eq!(log_probs.len(), 1);
        assert!(log_probs[0].is_finite());
        assert_eq!(added, 1);
        assert_eq!(mtp_cache.seq_len(), prompt_len + 1);

        // Verify on production via the sequential greedy oracle.
        let token_offset = main_cache.seq_len();
        let seq = crate::mtp::sequential_greedy_qwen4_exp_mtp_verify(
            cfg,
            weights,
            &mut main_cache,
            first_tok,
            &draft,
            token_offset,
            (HC * HIDDEN) as usize,
        );

        // Pure greedy singleton stream for the same boundary.
        let mut pure_cache = MlxKVCache::new(LAYER_COUNT);
        let _ = crate::model::forward(cfg, weights, &prompt, &mut pure_cache, 0);
        pure_cache.advance(prompt_len);
        let mut pure_stream = Vec::new();
        let mut last_tok = first_tok;
        for _ in 0..2 {
            let off = pure_cache.seq_len();
            let logits =
                crate::model::forward_argmax(cfg, weights, &[last_tok], &mut pure_cache, off);
            pure_cache.advance(1);
            let pred = argmax(&logits, None);
            eval(&[&pred]);
            let tok = pred.data_u32()[0];
            pure_stream.push(tok);
            last_tok = tok;
        }

        let mut committed: Vec<u32> = draft[..seq.accept_count].to_vec();
        committed.push(seq.correction_token);
        assert_eq!(
            committed,
            pure_stream[..seq.accept_count + 1],
            "accepted draft + correction must equal pure singleton greedy"
        );
        assert_eq!(
            main_cache.seq_len(),
            token_offset + 1 + seq.accept_count,
            "production commits primary + accepted drafts only"
        );

        // Dedicated-cache isolation: the MTP cache covers warmup + draft at
        // QSA layer 0; the main cache's QSA layer 3 grew only by the verify
        // forwards (never by MTP warmup/draft).
        let (mtp_k, _) = mtp_cache
            .qwen4_exp_qsa_kv(0)
            .expect("MTP cache must hold QSA K/V at layer 0");
        assert_eq!(mtp_k.shape()[2] as usize, prompt_len + 1);
        let (main_k, _) = main_cache
            .qwen4_exp_qsa_kv(QSA_LAYER as usize)
            .expect("main cache QSA layer must hold K/V");
        assert_eq!(
            main_k.shape()[2] as usize,
            token_offset + 1 + seq.accept_count,
            "main QSA K/V must cover prefill + verify forwards only"
        );
        let (conv_ring, token_ring) = mtp_cache.qwen4_exp_ple_state();
        assert!(conv_ring.is_none() && token_ring.is_none());

        // Reject-path MTP cache trim (runner contract): drop the rejected
        // draft entry; the warmup prefix stays warm.
        let rejected = 1usize.saturating_sub(seq.accept_count.min(1));
        if rejected > 0 {
            let new_len = (prompt_len + 1) - rejected;
            assert!(
                mtp_cache.trim_to(new_len),
                "QSA rewind must accept the draft trim"
            );
            assert_eq!(mtp_cache.seq_len(), new_len);
        }
    }

    #[test]
    fn mtp_after_forced_prefix_advances_cache_and_caps_tail() {
        use crate::sampling::Xorshift64;
        let fixture = mtp_fixture("forced-prefix");
        let MtpFixture { cfg, weights, .. } = &fixture;
        let mut mtp_cache = MlxKVCache::new(1);
        let mut rng = Xorshift64::new(1);
        let first_hidden = packed_with_stream_constants([0.25, 0.5, 0.75, 1.0], 1);

        // Hybrid n-gram+MTP path: thread two forced tokens, then draft the
        // tail. The tail is depth-capped at 1; `added` counts both.
        let (tail, log_probs, _dist, added, _margins) =
            crate::mtp::qwen4_exp_mtp_draft_tokens_after_forced_prefix(
                weights,
                cfg,
                &first_hidden,
                3,
                &[5, 7],
                &mut mtp_cache,
                1,
                &mut rng,
                Some(0.0),
                1.0,
            );
        assert_eq!(tail.len(), 1, "tail depth caps at the one shipped block");
        assert_eq!(log_probs.len(), 1);
        assert_eq!(added, 3, "forced prefix + one tail token");
        assert_eq!(mtp_cache.seq_len(), 3);

        // Zero tail depth: only the forced prefix advances the cache.
        let (empty, _lp, _d, added2, _m2) =
            crate::mtp::qwen4_exp_mtp_draft_tokens_after_forced_prefix(
                weights,
                cfg,
                &first_hidden,
                3,
                &[9, 11],
                &mut mtp_cache,
                0,
                &mut rng,
                Some(0.0),
                1.0,
            );
        assert!(empty.is_empty());
        assert_eq!(added2, 2);
        assert_eq!(mtp_cache.seq_len(), 5);
    }
}
