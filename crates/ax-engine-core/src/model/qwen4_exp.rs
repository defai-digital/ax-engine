use super::*;

fn invalid(message: String) -> NativeModelError {
    NativeModelError::InvalidManifest { message }
}

fn require_u32(value: Option<u32>, field: &str) -> Result<u32, NativeModelError> {
    match value {
        Some(v) if v > 0 => Ok(v),
        Some(_) => Err(invalid(format!("qwen4_exp {field} must be > 0"))),
        None => Err(invalid(format!("qwen4_exp {field} is required"))),
    }
}

fn require_i32(value: u32, field: &str) -> Result<i32, NativeModelError> {
    i32::try_from(value).map_err(|_| invalid(format!("qwen4_exp {field} {value} exceeds i32::MAX")))
}

fn checked_mul(a: u64, b: u64, field: &str) -> Result<u64, NativeModelError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid(format!("qwen4_exp {field} overflow")))
}

pub(super) fn validate(manifest: &NativeModelManifest) -> Result<(), NativeModelError> {
    if manifest.model_family != "qwen4_exp" {
        return Err(invalid(format!(
            "qwen4_exp validator requires model_family qwen4_exp, got {:?}",
            manifest.model_family
        )));
    }
    if !matches!(
        manifest.qwen4_exp.output_gate_type.as_deref(),
        None | Some("sigmoid") | Some("silu")
    ) {
        return Err(invalid(format!(
            "qwen4_exp output_gate_type must be None, Some(\"sigmoid\"), or Some(\"silu\"), got {:?}",
            manifest.qwen4_exp.output_gate_type
        )));
    }
    if !manifest.qwen4_exp.never_eval_ngram_at_load {
        return Err(invalid(
            "qwen4_exp never_eval_ngram_at_load must be true".to_string(),
        ));
    }
    if !manifest.attn_output_gate {
        return Err(invalid(
            "qwen4_exp requires attn_output_gate=true (Q includes gate)".to_string(),
        ));
    }
    for (value, name) in [
        (manifest.layer_count, "layer_count"),
        (manifest.hidden_size, "hidden_size"),
        (manifest.attention_head_count, "attention_head_count"),
        (manifest.attention_head_dim, "attention_head_dim"),
        (manifest.kv_head_count, "kv_head_count"),
        (manifest.vocab_size, "vocab_size"),
    ] {
        if value == 0 {
            return Err(invalid(format!("qwen4_exp {name} must be > 0")));
        }
        require_i32(value, name)?;
    }
    if !manifest
        .attention_head_count
        .is_multiple_of(manifest.kv_head_count)
    {
        return Err(invalid(format!(
            "qwen4_exp attention_head_count {} must be divisible by kv_head_count {}",
            manifest.attention_head_count, manifest.kv_head_count
        )));
    }
    let hidden = u64::from(manifest.hidden_size);
    let vocab = u64::from(manifest.vocab_size);
    let n_heads = u64::from(manifest.attention_head_count);
    let head_dim = u64::from(manifest.attention_head_dim);
    let kv_heads = u64::from(manifest.kv_head_count);
    let q_rows = checked_mul(n_heads, head_dim, "attention q rows")?;
    let kv_rows = checked_mul(kv_heads, head_dim, "attention kv rows")?;
    let gated_q_rows = checked_mul(q_rows, 2, "gated attention q rows")?;
    for (value, name) in [
        (q_rows, "q_rows"),
        (kv_rows, "kv_rows"),
        (gated_q_rows, "gated_q_rows"),
    ] {
        if value == 0 || value > u64::from(i32::MAX as u32) {
            return Err(invalid(format!(
                "qwen4_exp {name} {value} invalid or exceeds i32"
            )));
        }
    }
    let cfg = &manifest.qwen4_exp;
    let hc_count = require_u32(cfg.hc_count, "hc_count")?;
    let hc_lowrank = require_u32(cfg.hc_lowrank, "hc_lowrank")?;
    if hc_count <= 1 {
        return Err(invalid(format!(
            "qwen4_exp hc_count must be > 1, got {hc_count}"
        )));
    }
    require_i32(hc_count, "hc_count")?;
    require_i32(hc_lowrank, "hc_lowrank")?;
    let packed = checked_mul(u64::from(hc_count), hidden, "hc packed width")?;
    if packed == 0 || packed > u64::from(i32::MAX as u32) {
        return Err(invalid(format!(
            "qwen4_exp hc packed width {packed} invalid"
        )));
    }
    let lowrank = u64::from(hc_lowrank);
    let streams = u64::from(hc_count);
    let indexer_budget = require_u32(cfg.indexer_budget, "indexer_budget")?;
    let indexer_ratio = require_u32(cfg.indexer_compress_ratio, "indexer_compress_ratio")?;
    let indexer_head_dim = require_u32(cfg.indexer_head_dim, "indexer_head_dim")?;
    let indexer_n_heads = require_u32(cfg.indexer_n_heads, "indexer_n_heads")?;
    let indexer_kv_heads = require_u32(cfg.indexer_kv_heads, "indexer_kv_heads")?;
    for (v, n) in [
        (indexer_budget, "indexer_budget"),
        (indexer_ratio, "indexer_compress_ratio"),
        (indexer_head_dim, "indexer_head_dim"),
        (indexer_n_heads, "indexer_n_heads"),
        (indexer_kv_heads, "indexer_kv_heads"),
    ] {
        require_i32(v, n)?;
    }
    if indexer_kv_heads != 1 {
        return Err(invalid(format!(
            "qwen4_exp indexer_kv_heads must be 1, got {indexer_kv_heads}"
        )));
    }
    if !indexer_n_heads.is_multiple_of(indexer_kv_heads) {
        return Err(invalid(format!(
            "qwen4_exp indexer_n_heads {indexer_n_heads} must be divisible by indexer_kv_heads {indexer_kv_heads}"
        )));
    }
    if !indexer_budget.is_multiple_of(indexer_ratio) {
        return Err(invalid(format!(
            "qwen4_exp indexer_budget {indexer_budget} must be divisible by indexer_compress_ratio {indexer_ratio}"
        )));
    }
    let factor = manifest
        .partial_rotary_factor
        .ok_or_else(|| invalid("qwen4_exp partial_rotary_factor is required".to_string()))?;
    if !(factor > 0.0 && factor <= 1.0 && factor.is_finite()) {
        return Err(invalid(format!(
            "qwen4_exp partial_rotary_factor must be in (0, 1], got {factor}"
        )));
    }
    let rotary_dim = (manifest.attention_head_dim as f32 * factor) as u32;
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
        return Err(invalid(format!(
            "qwen4_exp rotary_dim {rotary_dim} must be even and > 0"
        )));
    }
    if rotary_dim > indexer_head_dim {
        return Err(invalid(format!(
            "qwen4_exp rotary_dim {rotary_dim} must be <= indexer_head_dim {indexer_head_dim}"
        )));
    }
    let indexer_proj_out = checked_mul(
        u64::from(indexer_n_heads)
            .checked_add(1)
            .ok_or_else(|| invalid("qwen4_exp indexer proj rows overflow".to_string()))?,
        u64::from(indexer_head_dim),
        "indexer proj rows",
    )?;
    if indexer_proj_out > u64::from(i32::MAX as u32) {
        return Err(invalid(format!(
            "qwen4_exp indexer proj rows {indexer_proj_out} exceed i32"
        )));
    }
    let la = &manifest.linear_attention;
    let num_value_heads = require_u32(la.num_value_heads, "linear_attention.num_value_heads")?;
    let num_key_heads = require_u32(la.num_key_heads, "linear_attention.num_key_heads")?;
    let key_head_dim = require_u32(la.key_head_dim, "linear_attention.key_head_dim")?;
    let value_head_dim = require_u32(la.value_head_dim, "linear_attention.value_head_dim")?;
    let conv_kernel_dim = require_u32(la.conv_kernel_dim, "linear_attention.conv_kernel_dim")?;
    let full_interval = la
        .resolved_full_attention_interval(&manifest.model_family)
        .filter(|v| *v > 0)
        .ok_or_else(|| {
            invalid("qwen4_exp linear_attention.full_attention_interval is required".to_string())
        })?;
    for (v, n) in [
        (num_value_heads, "num_value_heads"),
        (num_key_heads, "num_key_heads"),
        (key_head_dim, "key_head_dim"),
        (value_head_dim, "value_head_dim"),
        (conv_kernel_dim, "conv_kernel_dim"),
        (full_interval, "full_attention_interval"),
    ] {
        require_i32(v, n)?;
    }
    if !num_value_heads.is_multiple_of(num_key_heads) {
        return Err(invalid(format!(
            "qwen4_exp linear num_value_heads {num_value_heads} must be divisible by num_key_heads {num_key_heads}"
        )));
    }
    let key_dim = checked_mul(
        u64::from(num_key_heads),
        u64::from(key_head_dim),
        "gdn key_dim",
    )?;
    let value_dim = checked_mul(
        u64::from(num_value_heads),
        u64::from(value_head_dim),
        "gdn value_dim",
    )?;
    let conv_dim = checked_mul(key_dim, 2, "gdn conv_dim")?
        .checked_add(value_dim)
        .ok_or_else(|| invalid("qwen4_exp gdn conv_dim overflow".to_string()))?;
    for (v, n) in [
        (key_dim, "key_dim"),
        (value_dim, "value_dim"),
        (conv_dim, "conv_dim"),
    ] {
        if v == 0 || v > u64::from(i32::MAX as u32) {
            return Err(invalid(format!("qwen4_exp {n} {v} invalid")));
        }
    }
    let moe = &manifest.moe;
    let expert_count = require_u32(moe.expert_count, "moe.expert_count")?;
    let experts_per_token = require_u32(moe.experts_per_token, "moe.experts_per_token")?;
    let expert_intermediate =
        require_u32(moe.expert_intermediate_size, "moe.expert_intermediate_size")?;
    for (v, n) in [
        (expert_count, "moe.expert_count"),
        (experts_per_token, "moe.experts_per_token"),
        (expert_intermediate, "moe.expert_intermediate_size"),
    ] {
        require_i32(v, n)?;
    }
    if experts_per_token > expert_count {
        return Err(invalid(format!(
            "qwen4_exp moe.experts_per_token {experts_per_token} must be <= expert_count {expert_count}"
        )));
    }
    if moe.shared_expert_count != Some(1) {
        return Err(invalid(format!(
            "qwen4_exp moe.shared_expert_count must be Some(1), got {:?}",
            moe.shared_expert_count
        )));
    }
    let ple_embed_dim = require_u32(cfg.ple_embed_dim, "ple_embed_dim")?;
    let ple_kernel = require_u32(cfg.ple_conv_kernel_size, "ple_conv_kernel_size")?;
    let ngram_size = require_u32(cfg.ngram_size, "ngram_size")?;
    let heads_per_ngram = require_u32(cfg.heads_per_ngram, "heads_per_ngram")?;
    let ngram_divisor = require_u32(cfg.ngram_vocab_divisor, "ngram_vocab_divisor")?;
    let ngram_base = require_u32(cfg.ngram_vocab_size_base, "ngram_vocab_size_base")?;
    let split_parts = require_u32(cfg.split_ngram_parts, "split_ngram_parts")?;
    for (v, n) in [
        (ple_embed_dim, "ple_embed_dim"),
        (ple_kernel, "ple_conv_kernel_size"),
        (ngram_size, "ngram_size"),
        (heads_per_ngram, "heads_per_ngram"),
        (ngram_divisor, "ngram_vocab_divisor"),
        (ngram_base, "ngram_vocab_size_base"),
        (split_parts, "split_ngram_parts"),
    ] {
        require_i32(v, n)?;
    }
    if cfg.ngram_seed.is_none() {
        return Err(invalid("qwen4_exp ngram_seed is required".to_string()));
    }
    if ngram_size < 2 {
        return Err(invalid(format!(
            "qwen4_exp ngram_size must be >= 2, got {ngram_size}"
        )));
    }
    let orders = u64::from(ngram_size - 1);
    let ple_heads = checked_mul(orders, u64::from(heads_per_ngram), "ple head count")?;
    if ple_heads == 0 || ple_heads > u64::from(i32::MAX as u32) {
        return Err(invalid(format!(
            "qwen4_exp ple head count {ple_heads} invalid"
        )));
    }
    if u64::from(ple_embed_dim) % ple_heads != 0 {
        return Err(invalid(format!(
            "qwen4_exp ple_embed_dim {} must be divisible by (ngram_size-1)*heads_per_ngram {}",
            ple_embed_dim, ple_heads
        )));
    }
    let ngram_row_width = u64::from(ple_embed_dim) / ple_heads;
    let conv_state_len = u64::from(ple_kernel - 1)
        .checked_mul(u64::from(ngram_size))
        .ok_or_else(|| invalid("qwen4_exp ple conv state overflow".to_string()))?;
    if conv_state_len > u64::from(i32::MAX as u32) {
        return Err(invalid(format!(
            "qwen4_exp ple conv state {conv_state_len} exceeds i32"
        )));
    }
    if manifest.layer_types.len() != manifest.layer_count as usize {
        return Err(invalid(format!(
            "qwen4_exp layer_types must contain one entry per layer, got {} for {}",
            manifest.layer_types.len(),
            manifest.layer_count
        )));
    }
    for (idx, kind) in manifest.layer_types.iter().enumerate() {
        if !matches!(
            kind.as_str(),
            "linear_attention" | "full_attention" | "qwen_sparse_attention"
        ) {
            return Err(invalid(format!(
                "qwen4_exp layer_types[{idx}] must be linear_attention, full_attention, or qwen_sparse_attention, got {kind:?}"
            )));
        }
    }
    if !manifest.layer_types.iter().any(|k| k == "linear_attention")
        || !manifest
            .layer_types
            .iter()
            .any(|k| matches!(k.as_str(), "full_attention" | "qwen_sparse_attention"))
    {
        return Err(invalid(
            "qwen4_exp requires at least one linear_attention and one full_attention layer"
                .to_string(),
        ));
    }
    if cfg.ple_layer_ids.is_empty() {
        return Err(invalid(
            "qwen4_exp ple_layer_ids must not be empty".to_string(),
        ));
    }
    {
        use std::collections::BTreeSet;
        let mut seen = BTreeSet::new();
        for id in &cfg.ple_layer_ids {
            if *id == 0 || *id > manifest.layer_count {
                return Err(invalid(format!(
                    "qwen4_exp ple_layer_ids entry {id} out of range 1..={}",
                    manifest.layer_count
                )));
            }
            if !seen.insert(*id) {
                return Err(invalid(format!("qwen4_exp ple_layer_ids duplicate {id}")));
            }
            let layer = (*id - 1) as usize;
            if manifest.layer_types[layer] != "linear_attention" {
                return Err(invalid(format!(
                    "qwen4_exp ple_layer_ids entry {id} must reference a linear_attention layer"
                )));
            }
        }
    }
    let is_ple_layer = |layer: u32| cfg.ple_layer_ids.contains(&(layer + 1));
    for tensor in &manifest.tensors {
        match tensor.role {
            NativeTensorRole::FinalNorm
            | NativeTensorRole::AttentionNorm
            | NativeTensorRole::FfnNorm => {
                return Err(invalid(format!(
                    "qwen4_exp must not provide generic role {:?} ({})",
                    tensor.role, tensor.name
                )));
            }
            _ => {}
        }
    }
    let token_embedding = required_global_tensor_spec(
        manifest,
        NativeTensorRole::TokenEmbedding,
        "token_embedding",
    )?;
    expect_matrix_shape(token_embedding, vocab, hidden, "token_embedding")?;
    if !manifest.tie_word_embeddings {
        let lm_head = required_global_tensor_spec(manifest, NativeTensorRole::LmHead, "lm_head")?;
        expect_matrix_shape(lm_head, vocab, hidden, "lm_head")?;
    } else if let Some(lm_head) = manifest_tensor(manifest, NativeTensorRole::LmHead, None) {
        expect_matrix_shape(lm_head, vocab, hidden, "lm_head")?;
    }
    let mixer_norm = required_global_tensor_spec(
        manifest,
        NativeTensorRole::Qwen4ExpHcMixerNorm,
        "hc_mixer_norm",
    )?;
    expect_vector_shape(mixer_norm, packed, "hc_mixer_norm")?;
    let mixer_down = required_global_tensor_spec(
        manifest,
        NativeTensorRole::Qwen4ExpHcMixerMixDown,
        "hc_mixer_mix_down",
    )?;
    expect_matrix_shape(mixer_down, lowrank, packed, "hc_mixer_mix_down")?;
    let mixer_up = required_global_tensor_spec(
        manifest,
        NativeTensorRole::Qwen4ExpHcMixerMixUp,
        "hc_mixer_mix_up",
    )?;
    expect_matrix_shape(mixer_up, packed, lowrank, "hc_mixer_mix_up")?;
    let hf_conv = matches!(manifest.weight_sanitize, WeightSanitize::HfToMlx);
    for layer in 0..manifest.layer_count {
        let kind = manifest.layer_types[layer as usize].as_str();
        let is_gdn = kind == "linear_attention";
        for (role, label, rows, cols) in [
            (
                NativeTensorRole::Qwen4ExpAttnHcMixDown,
                "attn_hc_mix_down",
                lowrank,
                packed,
            ),
            (
                NativeTensorRole::Qwen4ExpAttnHcMixUp,
                "attn_hc_mix_up",
                packed,
                lowrank,
            ),
            (
                NativeTensorRole::Qwen4ExpMlpHcMixDown,
                "mlp_hc_mix_down",
                lowrank,
                packed,
            ),
            (
                NativeTensorRole::Qwen4ExpMlpHcMixUp,
                "mlp_hc_mix_up",
                packed,
                lowrank,
            ),
        ] {
            let tensor = required_layer_tensor_spec(manifest, layer, role, label)?;
            expect_matrix_shape(tensor, rows, cols, label)?;
        }
        for (role, label) in [
            (NativeTensorRole::Qwen4ExpAttnHcNorm, "attn_hc_norm"),
            (NativeTensorRole::Qwen4ExpMlpHcNorm, "mlp_hc_norm"),
        ] {
            let tensor = required_layer_tensor_spec(manifest, layer, role, label)?;
            expect_vector_shape(tensor, packed, label)?;
        }
        for (role, label) in [
            (NativeTensorRole::Qwen4ExpAttnHcInject, "attn_hc_inject"),
            (NativeTensorRole::Qwen4ExpMlpHcInject, "mlp_hc_inject"),
        ] {
            let tensor = required_layer_tensor_spec(manifest, layer, role, label)?;
            expect_matrix_shape(tensor, streams, packed, label)?;
        }
        if is_gdn {
            let qkv = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::LinearAttentionInProjQkv,
                "linear_attention_in_proj_qkv",
            )?;
            expect_matrix_shape(qkv, conv_dim, hidden, "linear_attention_in_proj_qkv")?;
            let z = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::LinearAttentionInProjZ,
                "linear_attention_in_proj_z",
            )?;
            expect_matrix_shape(z, value_dim, hidden, "linear_attention_in_proj_z")?;
            let a = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::LinearAttentionInProjA,
                "linear_attention_in_proj_a",
            )?;
            expect_matrix_shape(
                a,
                u64::from(num_value_heads),
                hidden,
                "linear_attention_in_proj_a",
            )?;
            let b = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::LinearAttentionInProjB,
                "linear_attention_in_proj_b",
            )?;
            expect_matrix_shape(
                b,
                u64::from(num_value_heads),
                hidden,
                "linear_attention_in_proj_b",
            )?;
            let conv = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::LinearAttentionConv1d,
                "linear_attention_conv1d",
            )?;
            expect_gdn_conv(conv, conv_dim, u64::from(conv_kernel_dim), hf_conv)?;
            let dt = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::LinearAttentionDtBias,
                "linear_attention_dt_bias",
            )?;
            expect_vector_shape(dt, u64::from(num_value_heads), "linear_attention_dt_bias")?;
            let alog = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::LinearAttentionALog,
                "linear_attention_a_log",
            )?;
            expect_vector_shape(alog, u64::from(num_value_heads), "linear_attention_a_log")?;
            let norm = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::LinearAttentionNorm,
                "linear_attention_norm",
            )?;
            expect_vector_shape(norm, u64::from(value_head_dim), "linear_attention_norm")?;
            let out = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::LinearAttentionOutProj,
                "linear_attention_out_proj",
            )?;
            expect_matrix_shape(out, hidden, value_dim, "linear_attention_out_proj")?;
            for (role, label) in [
                (
                    NativeTensorRole::LinearAttentionInProjQkvz,
                    "linear_attention_in_proj_qkvz",
                ),
                (
                    NativeTensorRole::LinearAttentionInProjBa,
                    "linear_attention_in_proj_ba",
                ),
                (NativeTensorRole::AttentionQ, "attention_q"),
                (NativeTensorRole::AttentionK, "attention_k"),
                (NativeTensorRole::AttentionV, "attention_v"),
                (NativeTensorRole::AttentionO, "attention_o"),
                (NativeTensorRole::AttentionQNorm, "attention_q_norm"),
                (NativeTensorRole::AttentionKNorm, "attention_k_norm"),
                (NativeTensorRole::AttentionQkvPacked, "attention_qkv_packed"),
                (NativeTensorRole::Qwen4ExpIndexerQkProj, "indexer_qk_proj"),
                (NativeTensorRole::Qwen4ExpIndexerQNorm, "indexer_q_norm"),
                (NativeTensorRole::Qwen4ExpIndexerKNorm, "indexer_k_norm"),
            ] {
                if manifest_tensor(manifest, role, Some(layer)).is_some() {
                    return Err(invalid(format!(
                        "qwen4_exp GDN layer {layer} must not provide {label}"
                    )));
                }
            }
        } else {
            let q = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::AttentionQ,
                "attention_q",
            )?;
            expect_matrix_shape(q, gated_q_rows, hidden, "attention_q")?;
            let k = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::AttentionK,
                "attention_k",
            )?;
            expect_matrix_shape(k, kv_rows, hidden, "attention_k")?;
            if let Some(v) = manifest_tensor(manifest, NativeTensorRole::AttentionV, Some(layer)) {
                expect_matrix_shape(v, kv_rows, hidden, "attention_v")?;
            } else {
                return Err(invalid(format!(
                    "qwen4_exp QSA layer {layer} requires attention_v"
                )));
            }
            let o = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::AttentionO,
                "attention_o",
            )?;
            expect_matrix_shape(o, hidden, q_rows, "attention_o")?;
            let qn = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::AttentionQNorm,
                "attention_q_norm",
            )?;
            expect_vector_shape(qn, head_dim, "attention_q_norm")?;
            let kn = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::AttentionKNorm,
                "attention_k_norm",
            )?;
            expect_vector_shape(kn, head_dim, "attention_k_norm")?;
            let qk = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::Qwen4ExpIndexerQkProj,
                "indexer_qk_proj",
            )?;
            expect_matrix_shape(qk, indexer_proj_out, hidden, "indexer_qk_proj")?;
            let iqn = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::Qwen4ExpIndexerQNorm,
                "indexer_q_norm",
            )?;
            expect_vector_shape(iqn, u64::from(indexer_head_dim), "indexer_q_norm")?;
            let ikn = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::Qwen4ExpIndexerKNorm,
                "indexer_k_norm",
            )?;
            expect_vector_shape(ikn, u64::from(indexer_head_dim), "indexer_k_norm")?;
            for (role, label) in [
                (
                    NativeTensorRole::LinearAttentionInProjQkv,
                    "linear_attention_in_proj_qkv",
                ),
                (
                    NativeTensorRole::LinearAttentionInProjQkvz,
                    "linear_attention_in_proj_qkvz",
                ),
                (
                    NativeTensorRole::LinearAttentionInProjZ,
                    "linear_attention_in_proj_z",
                ),
                (
                    NativeTensorRole::LinearAttentionInProjA,
                    "linear_attention_in_proj_a",
                ),
                (
                    NativeTensorRole::LinearAttentionInProjB,
                    "linear_attention_in_proj_b",
                ),
                (
                    NativeTensorRole::LinearAttentionInProjBa,
                    "linear_attention_in_proj_ba",
                ),
                (
                    NativeTensorRole::LinearAttentionConv1d,
                    "linear_attention_conv1d",
                ),
                (
                    NativeTensorRole::LinearAttentionDtBias,
                    "linear_attention_dt_bias",
                ),
                (
                    NativeTensorRole::LinearAttentionALog,
                    "linear_attention_a_log",
                ),
                (
                    NativeTensorRole::LinearAttentionNorm,
                    "linear_attention_norm",
                ),
                (
                    NativeTensorRole::LinearAttentionOutProj,
                    "linear_attention_out_proj",
                ),
                (NativeTensorRole::AttentionQkvPacked, "attention_qkv_packed"),
            ] {
                if manifest_tensor(manifest, role, Some(layer)).is_some() {
                    return Err(invalid(format!(
                        "qwen4_exp QSA layer {layer} must not provide {label}"
                    )));
                }
            }
        }
        let gate_inp = required_layer_tensor_spec(
            manifest,
            layer,
            NativeTensorRole::FfnGateInp,
            "ffn_gate_inp",
        )?;
        expect_matrix_shape(gate_inp, u64::from(expert_count), hidden, "ffn_gate_inp")?;
        let has_packed =
            manifest_tensor(manifest, NativeTensorRole::FfnGateUpExpsPacked, Some(layer)).is_some();
        let has_gate =
            manifest_tensor(manifest, NativeTensorRole::FfnGateExps, Some(layer)).is_some();
        let has_up = manifest_tensor(manifest, NativeTensorRole::FfnUpExps, Some(layer)).is_some();
        if has_packed == (has_gate || has_up) || has_gate != has_up {
            return Err(invalid(format!(
                "qwen4_exp layer {layer} must provide exactly one expert layout: ffn_gate_up_exps_packed or ffn_gate_exps plus ffn_up_exps"
            )));
        }
        if has_packed {
            let packed_tensor = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::FfnGateUpExpsPacked,
                "ffn_gate_up_exps_packed",
            )?;
            expect_tensor_shape(
                packed_tensor,
                &[
                    u64::from(expert_count),
                    u64::from(expert_intermediate).saturating_mul(2),
                    hidden,
                ],
                "ffn_gate_up_exps_packed",
            )?;
        } else {
            let gate = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::FfnGateExps,
                "ffn_gate_exps",
            )?;
            expect_tensor_shape(
                gate,
                &[
                    u64::from(expert_count),
                    u64::from(expert_intermediate),
                    hidden,
                ],
                "ffn_gate_exps",
            )?;
            let up = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::FfnUpExps,
                "ffn_up_exps",
            )?;
            expect_tensor_shape(
                up,
                &[
                    u64::from(expert_count),
                    u64::from(expert_intermediate),
                    hidden,
                ],
                "ffn_up_exps",
            )?;
        }
        let down = required_layer_tensor_spec(
            manifest,
            layer,
            NativeTensorRole::FfnDownExps,
            "ffn_down_exps",
        )?;
        expect_tensor_shape(
            down,
            &[
                u64::from(expert_count),
                hidden,
                u64::from(expert_intermediate),
            ],
            "ffn_down_exps",
        )?;
        let shared_inp = required_layer_tensor_spec(
            manifest,
            layer,
            NativeTensorRole::FfnSharedExpertGateInp,
            "ffn_shared_expert_gate_inp",
        )?;
        expect_matrix_shape(shared_inp, 1, hidden, "ffn_shared_expert_gate_inp")?;
        let shared_gate = required_layer_tensor_spec(
            manifest,
            layer,
            NativeTensorRole::FfnSharedExpertGate,
            "ffn_shared_expert_gate",
        )?;
        expect_matrix_shape(
            shared_gate,
            u64::from(expert_intermediate),
            hidden,
            "ffn_shared_expert_gate",
        )?;
        let shared_up = required_layer_tensor_spec(
            manifest,
            layer,
            NativeTensorRole::FfnSharedExpertUp,
            "ffn_shared_expert_up",
        )?;
        expect_matrix_shape(
            shared_up,
            u64::from(expert_intermediate),
            hidden,
            "ffn_shared_expert_up",
        )?;
        let shared_down = required_layer_tensor_spec(
            manifest,
            layer,
            NativeTensorRole::FfnSharedExpertDown,
            "ffn_shared_expert_down",
        )?;
        expect_matrix_shape(
            shared_down,
            hidden,
            u64::from(expert_intermediate),
            "ffn_shared_expert_down",
        )?;
        for (role, label) in [
            (NativeTensorRole::FfnGate, "ffn_gate"),
            (NativeTensorRole::FfnUp, "ffn_up"),
            (NativeTensorRole::FfnDown, "ffn_down"),
            (NativeTensorRole::FfnGateUpPacked, "ffn_gate_up_packed"),
            (NativeTensorRole::FfnGateInpScale, "ffn_gate_inp_scale"),
            (
                NativeTensorRole::FfnGateInpCorrectionBias,
                "ffn_gate_inp_correction_bias",
            ),
            (
                NativeTensorRole::FfnGateInpExpertScale,
                "ffn_gate_inp_expert_scale",
            ),
            (NativeTensorRole::FfnDownExpsScale, "ffn_down_exps_scale"),
            (NativeTensorRole::FfnNorm2, "ffn_norm_2"),
            (NativeTensorRole::FfnPostNorm, "ffn_post_norm"),
            (NativeTensorRole::FfnPostNorm1, "ffn_post_norm_1"),
            (NativeTensorRole::FfnPostNorm2, "ffn_post_norm_2"),
            (NativeTensorRole::AttentionPostNorm, "attention_post_norm"),
            (
                NativeTensorRole::FfnGateUpExpsMxfp4Blocks,
                "ffn_gate_up_exps_mxfp4_blocks",
            ),
            (
                NativeTensorRole::FfnGateUpExpsMxfp4Scales,
                "ffn_gate_up_exps_mxfp4_scales",
            ),
            (
                NativeTensorRole::FfnDownExpsMxfp4Blocks,
                "ffn_down_exps_mxfp4_blocks",
            ),
            (
                NativeTensorRole::FfnDownExpsMxfp4Scales,
                "ffn_down_exps_mxfp4_scales",
            ),
        ] {
            if manifest_tensor(manifest, role, Some(layer)).is_some() {
                return Err(invalid(format!(
                    "qwen4_exp layer {layer} must not provide {label}"
                )));
            }
        }
        let ple_roles = [
            (NativeTensorRole::Qwen4ExpPleKeyProj, "ple_key_proj"),
            (NativeTensorRole::Qwen4ExpPleValueProj, "ple_value_proj"),
            (NativeTensorRole::Qwen4ExpPleConv1d, "ple_conv1d"),
            (NativeTensorRole::Qwen4ExpPleNormQuery, "ple_norm_query"),
            (NativeTensorRole::Qwen4ExpPleNormKey, "ple_norm_key"),
            (NativeTensorRole::Qwen4ExpPleNormConv, "ple_norm_conv"),
            (NativeTensorRole::Qwen4ExpPleHeadOffsets, "ple_head_offsets"),
            (
                NativeTensorRole::Qwen4ExpPleHeadVocabSizes,
                "ple_head_vocab_sizes",
            ),
            (NativeTensorRole::Qwen4ExpPleMultipliers, "ple_multipliers"),
        ];
        if is_ple_layer(layer) {
            let key = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::Qwen4ExpPleKeyProj,
                "ple_key_proj",
            )?;
            expect_matrix_shape(key, packed, u64::from(ple_embed_dim), "ple_key_proj")?;
            let value = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::Qwen4ExpPleValueProj,
                "ple_value_proj",
            )?;
            expect_matrix_shape(value, hidden, u64::from(ple_embed_dim), "ple_value_proj")?;
            for (role, label) in [
                (NativeTensorRole::Qwen4ExpPleNormQuery, "ple_norm_query"),
                (NativeTensorRole::Qwen4ExpPleNormKey, "ple_norm_key"),
                (NativeTensorRole::Qwen4ExpPleNormConv, "ple_norm_conv"),
            ] {
                let norm = required_layer_tensor_spec(manifest, layer, role, label)?;
                expect_vector_shape(norm, packed, label)?;
            }
            let conv = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::Qwen4ExpPleConv1d,
                "ple_conv1d",
            )?;
            expect_ple_conv(conv, packed, u64::from(ple_kernel), hf_conv)?;
            let mult = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::Qwen4ExpPleMultipliers,
                "ple_multipliers",
            )?;
            expect_hash_buffer(mult, u64::from(ngram_size), "ple_multipliers")?;
            let offsets = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::Qwen4ExpPleHeadOffsets,
                "ple_head_offsets",
            )?;
            expect_hash_buffer(offsets, ple_heads, "ple_head_offsets")?;
            let sizes = required_layer_tensor_spec(
                manifest,
                layer,
                NativeTensorRole::Qwen4ExpPleHeadVocabSizes,
                "ple_head_vocab_sizes",
            )?;
            expect_hash_buffer(sizes, ple_heads, "ple_head_vocab_sizes")?;
            validate_ngram_shards(manifest, layer, ngram_row_width)?;
        } else {
            for (role, label) in ple_roles {
                if manifest_tensor(manifest, role, Some(layer)).is_some() {
                    return Err(invalid(format!(
                        "qwen4_exp layer {layer} must not provide stray {label}"
                    )));
                }
            }
            if manifest
                .tensors
                .iter()
                .any(|t| t.role == NativeTensorRole::NgramEmbedding && t.layer_index == Some(layer))
            {
                return Err(invalid(format!(
                    "qwen4_exp layer {layer} must not provide stray ngram_embedding"
                )));
            }
        }
    }
    Ok(())
}

fn expect_gdn_conv(
    tensor: &NativeTensorSpec,
    channels: u64,
    kernel: u64,
    hf_layout: bool,
) -> Result<(), NativeModelError> {
    let expected = if hf_layout {
        vec![channels, 1, kernel]
    } else {
        vec![channels, kernel, 1]
    };
    if tensor.shape == expected {
        Ok(())
    } else {
        Err(invalid(format!(
            "tensor linear_attention_conv1d must have shape {:?}, got {:?}",
            expected, tensor.shape
        )))
    }
}

fn expect_ple_conv(
    tensor: &NativeTensorSpec,
    channels: u64,
    kernel: u64,
    hf_layout: bool,
) -> Result<(), NativeModelError> {
    let expected = if hf_layout {
        vec![channels, 1, kernel]
    } else {
        vec![channels, kernel, 1]
    };
    if tensor.shape == expected {
        Ok(())
    } else {
        Err(invalid(format!(
            "tensor ple_conv1d must have shape {:?}, got {:?}",
            expected, tensor.shape
        )))
    }
}

fn expect_hash_buffer(
    tensor: &NativeTensorSpec,
    expected_len: u64,
    label: &str,
) -> Result<(), NativeModelError> {
    if tensor.dtype != NativeTensorDataType::I64 {
        return Err(invalid(format!(
            "tensor {label} must use i64, got {:?}",
            tensor.dtype
        )));
    }
    if tensor.source_quantized || tensor.quantization.is_some() {
        return Err(invalid(format!("tensor {label} must not be quantized")));
    }
    expect_vector_shape(tensor, expected_len, label)
}

fn parse_ngram_shard(name: &str) -> Result<Option<usize>, NativeModelError> {
    if name.ends_with(".weight_scale") {
        return Err(invalid(format!(
            "qwen4_exp ngram table {name:?} uses unsupported weight_scale layout"
        )));
    }
    let base = name.strip_suffix(".weight").ok_or_else(|| {
        invalid(format!(
            "qwen4_exp ngram table {name:?} must end with .weight"
        ))
    })?;
    if let Some(pos) = base.rfind("shards.") {
        let suffix = &base[pos + "shards.".len()..];
        if !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) {
            let index: usize = suffix.parse().map_err(|_| {
                invalid(format!("qwen4_exp ngram shard index overflow in {name:?}"))
            })?;
            return Ok(Some(index));
        }
        return Err(invalid(format!(
            "qwen4_exp ngram shard name {name:?} is invalid"
        )));
    }
    if let Some(pos) = base.rfind("shard_") {
        let suffix = &base[pos + "shard_".len()..];
        if !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) {
            let index: usize = suffix.parse().map_err(|_| {
                invalid(format!("qwen4_exp ngram shard index overflow in {name:?}"))
            })?;
            return Ok(Some(index));
        }
        return Err(invalid(format!(
            "qwen4_exp ngram shard name {name:?} is invalid"
        )));
    }
    if base == "ngram_embedding" || base.ends_with(".ngram_embedding") {
        return Ok(None);
    }
    Err(invalid(format!(
        "qwen4_exp ngram table name {name:?} is invalid"
    )))
}

fn validate_ngram_shards(
    manifest: &NativeModelManifest,
    layer: u32,
    row_width: u64,
) -> Result<(), NativeModelError> {
    let specs: Vec<&NativeTensorSpec> = manifest
        .tensors
        .iter()
        .filter(|t| t.role == NativeTensorRole::NgramEmbedding && t.layer_index == Some(layer))
        .collect();
    if specs.is_empty() {
        return Err(invalid(format!(
            "qwen4_exp PLE layer {layer} is missing ngram_embedding"
        )));
    }
    let mut sharded: Vec<usize> = Vec::new();
    let mut unsharded = 0usize;
    let mut saw_underscore = false;
    let mut saw_dots = false;
    for spec in &specs {
        if spec.shape.len() != 2 || spec.shape[0] == 0 {
            return Err(invalid(format!(
                "qwen4_exp PLE layer {layer} ngram table {:?} must be [rows>0, {row_width}]",
                spec.name
            )));
        }
        if uses_packed_u32_storage(spec) {
            let expected = expected_packed_cols(row_width, spec)?;
            if spec.shape[1] != expected {
                return Err(invalid(format!(
                    "qwen4_exp PLE layer {layer} ngram table {:?} must have packed width {expected}, got {:?}",
                    spec.name, spec.shape
                )));
            }
        } else if spec.shape[1] != row_width {
            return Err(invalid(format!(
                "qwen4_exp PLE layer {layer} ngram table {:?} must have row width {row_width}, got {:?}",
                spec.name, spec.shape
            )));
        }
        if spec.shape[0] > u64::from(i32::MAX as u32) {
            return Err(invalid(format!(
                "qwen4_exp PLE layer {layer} ngram rows exceed i32"
            )));
        }
        match parse_ngram_shard(&spec.name)? {
            Some(index) => {
                sharded.push(index);
                if spec.name.contains("shards.") {
                    saw_dots = true;
                } else {
                    saw_underscore = true;
                }
            }
            None => unsharded += 1,
        }
    }
    if unsharded > 0 && !sharded.is_empty() {
        return Err(invalid(format!(
            "qwen4_exp PLE layer {layer} must not mix sharded and unsharded ngram tables"
        )));
    }
    if unsharded > 0 {
        if specs.len() != 1 {
            return Err(invalid(format!(
                "qwen4_exp PLE layer {layer} unsharded ngram table must be exactly one tensor"
            )));
        }
        return Ok(());
    }
    if saw_underscore && saw_dots {
        return Err(invalid(format!(
            "qwen4_exp PLE layer {layer} must not mix shard_N and shards.N styles"
        )));
    }
    sharded.sort_unstable();
    for (position, index) in sharded.iter().enumerate() {
        if *index != position {
            return Err(invalid(format!(
                "qwen4_exp PLE layer {layer} ngram shards must be contiguous from 0, got {sharded:?}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use std::path::PathBuf;

    fn spec(
        name: &str,
        role: NativeTensorRole,
        layer: Option<u32>,
        shape: Vec<u64>,
    ) -> NativeTensorSpec {
        NativeTensorSpec {
            name: name.to_string(),
            role,
            layer_index: layer,
            dtype: NativeTensorDataType::F16,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape,
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: 32,
        }
    }

    fn hash_spec(name: &str, layer: u32, len: u64) -> NativeTensorSpec {
        NativeTensorSpec {
            name: name.to_string(),
            role: if name.contains("multipliers") {
                NativeTensorRole::Qwen4ExpPleMultipliers
            } else if name.contains("offsets") {
                NativeTensorRole::Qwen4ExpPleHeadOffsets
            } else {
                NativeTensorRole::Qwen4ExpPleHeadVocabSizes
            },
            layer_index: Some(layer),
            dtype: NativeTensorDataType::I64,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape: vec![len],
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: 32,
        }
    }

    fn valid_manifest() -> NativeModelManifest {
        let tensors = vec![
            spec(
                "model.embed_tokens.weight",
                NativeTensorRole::TokenEmbedding,
                None,
                vec![32, 8],
            ),
            spec(
                "lm_head.weight",
                NativeTensorRole::LmHead,
                None,
                vec![32, 8],
            ),
            spec(
                "mixer.hc_norm.weight",
                NativeTensorRole::Qwen4ExpHcMixerNorm,
                None,
                vec![16],
            ),
            spec(
                "mixer.down.weight",
                NativeTensorRole::Qwen4ExpHcMixerMixDown,
                None,
                vec![4, 16],
            ),
            spec(
                "mixer.up.weight",
                NativeTensorRole::Qwen4ExpHcMixerMixUp,
                None,
                vec![16, 4],
            ),
            spec(
                "l0.attn_norm",
                NativeTensorRole::Qwen4ExpAttnHcNorm,
                Some(0),
                vec![16],
            ),
            spec(
                "l0.attn_down",
                NativeTensorRole::Qwen4ExpAttnHcMixDown,
                Some(0),
                vec![4, 16],
            ),
            spec(
                "l0.attn_up",
                NativeTensorRole::Qwen4ExpAttnHcMixUp,
                Some(0),
                vec![16, 4],
            ),
            spec(
                "l0.attn_inject",
                NativeTensorRole::Qwen4ExpAttnHcInject,
                Some(0),
                vec![2, 16],
            ),
            spec(
                "l0.mlp_norm",
                NativeTensorRole::Qwen4ExpMlpHcNorm,
                Some(0),
                vec![16],
            ),
            spec(
                "l0.mlp_down",
                NativeTensorRole::Qwen4ExpMlpHcMixDown,
                Some(0),
                vec![4, 16],
            ),
            spec(
                "l0.mlp_up",
                NativeTensorRole::Qwen4ExpMlpHcMixUp,
                Some(0),
                vec![16, 4],
            ),
            spec(
                "l0.mlp_inject",
                NativeTensorRole::Qwen4ExpMlpHcInject,
                Some(0),
                vec![2, 16],
            ),
            spec(
                "l0.qkv",
                NativeTensorRole::LinearAttentionInProjQkv,
                Some(0),
                vec![12, 8],
            ),
            spec(
                "l0.z",
                NativeTensorRole::LinearAttentionInProjZ,
                Some(0),
                vec![4, 8],
            ),
            spec(
                "l0.a",
                NativeTensorRole::LinearAttentionInProjA,
                Some(0),
                vec![2, 8],
            ),
            spec(
                "l0.b",
                NativeTensorRole::LinearAttentionInProjB,
                Some(0),
                vec![2, 8],
            ),
            spec(
                "l0.conv",
                NativeTensorRole::LinearAttentionConv1d,
                Some(0),
                vec![12, 4, 1],
            ),
            spec(
                "l0.dt",
                NativeTensorRole::LinearAttentionDtBias,
                Some(0),
                vec![2],
            ),
            spec(
                "l0.alog",
                NativeTensorRole::LinearAttentionALog,
                Some(0),
                vec![2],
            ),
            spec(
                "l0.norm",
                NativeTensorRole::LinearAttentionNorm,
                Some(0),
                vec![2],
            ),
            spec(
                "l0.out",
                NativeTensorRole::LinearAttentionOutProj,
                Some(0),
                vec![8, 4],
            ),
            spec(
                "l0.gate_inp",
                NativeTensorRole::FfnGateInp,
                Some(0),
                vec![2, 8],
            ),
            spec(
                "l0.gate_exps",
                NativeTensorRole::FfnGateExps,
                Some(0),
                vec![2, 8, 8],
            ),
            spec(
                "l0.up_exps",
                NativeTensorRole::FfnUpExps,
                Some(0),
                vec![2, 8, 8],
            ),
            spec(
                "l0.down_exps",
                NativeTensorRole::FfnDownExps,
                Some(0),
                vec![2, 8, 8],
            ),
            spec(
                "l0.shared_inp",
                NativeTensorRole::FfnSharedExpertGateInp,
                Some(0),
                vec![1, 8],
            ),
            spec(
                "l0.shared_gate",
                NativeTensorRole::FfnSharedExpertGate,
                Some(0),
                vec![8, 8],
            ),
            spec(
                "l0.shared_up",
                NativeTensorRole::FfnSharedExpertUp,
                Some(0),
                vec![8, 8],
            ),
            spec(
                "l0.shared_down",
                NativeTensorRole::FfnSharedExpertDown,
                Some(0),
                vec![8, 8],
            ),
            spec(
                "l0.ple.key_proj.weight",
                NativeTensorRole::Qwen4ExpPleKeyProj,
                Some(0),
                vec![16, 8],
            ),
            spec(
                "l0.ple.value_proj.weight",
                NativeTensorRole::Qwen4ExpPleValueProj,
                Some(0),
                vec![8, 8],
            ),
            spec(
                "l0.ple.conv1d.weight",
                NativeTensorRole::Qwen4ExpPleConv1d,
                Some(0),
                vec![16, 3, 1],
            ),
            spec(
                "l0.ple.norm_query.weight",
                NativeTensorRole::Qwen4ExpPleNormQuery,
                Some(0),
                vec![16],
            ),
            spec(
                "l0.ple.norm_key.weight",
                NativeTensorRole::Qwen4ExpPleNormKey,
                Some(0),
                vec![16],
            ),
            spec(
                "l0.ple.norm_conv.weight",
                NativeTensorRole::Qwen4ExpPleNormConv,
                Some(0),
                vec![16],
            ),
            hash_spec("l0.ple.layer_multipliers", 0, 3),
            hash_spec("l0.ple.ngram_heads_offsets", 0, 4),
            hash_spec("l0.ple.ngram_heads_vocab_sizes", 0, 4),
            spec(
                "model.layers.0.ple.ple_embedding.ngram_embedding.weight",
                NativeTensorRole::NgramEmbedding,
                Some(0),
                vec![8, 2],
            ),
            spec(
                "l1.attn_norm",
                NativeTensorRole::Qwen4ExpAttnHcNorm,
                Some(1),
                vec![16],
            ),
            spec(
                "l1.attn_down",
                NativeTensorRole::Qwen4ExpAttnHcMixDown,
                Some(1),
                vec![4, 16],
            ),
            spec(
                "l1.attn_up",
                NativeTensorRole::Qwen4ExpAttnHcMixUp,
                Some(1),
                vec![16, 4],
            ),
            spec(
                "l1.attn_inject",
                NativeTensorRole::Qwen4ExpAttnHcInject,
                Some(1),
                vec![2, 16],
            ),
            spec(
                "l1.mlp_norm",
                NativeTensorRole::Qwen4ExpMlpHcNorm,
                Some(1),
                vec![16],
            ),
            spec(
                "l1.mlp_down",
                NativeTensorRole::Qwen4ExpMlpHcMixDown,
                Some(1),
                vec![4, 16],
            ),
            spec(
                "l1.mlp_up",
                NativeTensorRole::Qwen4ExpMlpHcMixUp,
                Some(1),
                vec![16, 4],
            ),
            spec(
                "l1.mlp_inject",
                NativeTensorRole::Qwen4ExpMlpHcInject,
                Some(1),
                vec![2, 16],
            ),
            spec("l1.q", NativeTensorRole::AttentionQ, Some(1), vec![16, 8]),
            spec("l1.k", NativeTensorRole::AttentionK, Some(1), vec![4, 8]),
            spec("l1.v", NativeTensorRole::AttentionV, Some(1), vec![4, 8]),
            spec("l1.o", NativeTensorRole::AttentionO, Some(1), vec![8, 8]),
            spec("l1.qn", NativeTensorRole::AttentionQNorm, Some(1), vec![4]),
            spec("l1.kn", NativeTensorRole::AttentionKNorm, Some(1), vec![4]),
            spec(
                "l1.qk",
                NativeTensorRole::Qwen4ExpIndexerQkProj,
                Some(1),
                vec![12, 8],
            ),
            spec(
                "l1.iqn",
                NativeTensorRole::Qwen4ExpIndexerQNorm,
                Some(1),
                vec![4],
            ),
            spec(
                "l1.ikn",
                NativeTensorRole::Qwen4ExpIndexerKNorm,
                Some(1),
                vec![4],
            ),
            spec(
                "l1.gate_inp",
                NativeTensorRole::FfnGateInp,
                Some(1),
                vec![2, 8],
            ),
            spec(
                "l1.gate_up",
                NativeTensorRole::FfnGateUpExpsPacked,
                Some(1),
                vec![2, 16, 8],
            ),
            spec(
                "l1.down_exps",
                NativeTensorRole::FfnDownExps,
                Some(1),
                vec![2, 8, 8],
            ),
            spec(
                "l1.shared_inp",
                NativeTensorRole::FfnSharedExpertGateInp,
                Some(1),
                vec![1, 8],
            ),
            spec(
                "l1.shared_gate",
                NativeTensorRole::FfnSharedExpertGate,
                Some(1),
                vec![8, 8],
            ),
            spec(
                "l1.shared_up",
                NativeTensorRole::FfnSharedExpertUp,
                Some(1),
                vec![8, 8],
            ),
            spec(
                "l1.shared_down",
                NativeTensorRole::FfnSharedExpertDown,
                Some(1),
                vec![8, 8],
            ),
        ];
        NativeModelManifest {
            schema_version: AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: "qwen4_exp".to_string(),
            tensor_format: NativeTensorFormat::Safetensors,
            source_quantization: None,
            runtime_status: NativeRuntimeStatus::default(),
            layer_count: 2,
            hidden_size: 8,
            intermediate_size: 0,
            attention_head_count: 2,
            attention_head_dim: 4,
            kv_head_count: 1,
            vocab_size: 32,
            tie_word_embeddings: false,
            rope_theta: Some(10000),
            rope_theta_swa: None,
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_low_freq_factor: None,
            rope_high_freq_factor: None,
            rope_original_context_len: None,
            rope_beta_fast: None,
            rope_beta_slow: None,
            no_rope_layer_interval: 0,
            attn_temperature_floor: None,
            attn_temperature_scale: None,
            intermediate_size_mlp: 0,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            attn_output_gate: true,
            partial_rotary_factor: Some(0.5),
            rms_norm_eps: Some(1e-6),
            attention_value_from_key_layers: Vec::new(),
            attention_v_norm_no_scale_layers: Vec::new(),
            global_head_dim: None,
            global_kv_head_count: None,
            sliding_window_size: None,
            layer_types: vec!["linear_attention".to_string(), "full_attention".to_string()],
            kv_shared_source_layers: Default::default(),
            final_logit_softcapping: None,
            final_logits_scale: None,
            attention_scale_multiplier: None,
            post_norm_eps: None,
            hidden_states_scale: None,
            moe_norm_topk_prob: true,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            linear_attention: NativeLinearAttentionConfig {
                full_attention_interval: Some(2),
                num_value_heads: Some(2),
                num_key_heads: Some(1),
                key_head_dim: Some(4),
                value_head_dim: Some(2),
                conv_kernel_dim: Some(4),
            },
            mla_attention: Default::default(),
            moe: NativeMoeConfig {
                expert_count: Some(2),
                experts_per_token: Some(1),
                expert_intermediate_size: Some(8),
                layer_freq: None,
                first_dense_layers: None,
                shared_expert_count: Some(1),
                sigmoid_routing: false,
                routed_scaling_factor: None,
                n_group: None,
                topk_group: None,
            },
            glm_router: Default::default(),
            deepseek_v4: Default::default(),
            qwen4_exp: NativeQwen4ExpConfig {
                output_gate_type: Some("sigmoid".to_string()),
                ple_layer_ids: vec![1],
                ple_embed_dim: Some(8),
                ple_conv_kernel_size: Some(3),
                ngram_vocab_divisor: Some(4),
                ngram_seed: Some(1234),
                ngram_size: Some(3),
                ngram_vocab_size_base: Some(16),
                split_ngram_parts: Some(2),
                heads_per_ngram: Some(2),
                hc_count: Some(2),
                hc_lowrank: Some(4),
                indexer_budget: Some(4),
                indexer_head_dim: Some(4),
                indexer_n_heads: Some(2),
                indexer_kv_heads: Some(1),
                indexer_compress_ratio: Some(2),
                never_eval_ngram_at_load: true,
            },
            weight_sanitize: WeightSanitize::None,
            think_start_token_id: None,
            think_end_token_id: None,
            diffusion: Default::default(),
            dropped_tensors: Default::default(),
            kv_cache_quantization: None,
            tensors,
        }
    }

    fn expect_err_contains(manifest: &NativeModelManifest, needle: &str) {
        let err = validate(manifest).expect_err("mutation should fail");
        let message = err.to_string();
        assert!(
            message.contains(needle),
            "expected {needle:?} in {message:?}"
        );
    }

    #[test]
    fn valid_synthetic_manifest_passes() {
        validate(&valid_manifest()).expect("valid manifest should pass");
    }

    #[test]
    fn rejects_wrong_gate_type_and_family() {
        let mut manifest = valid_manifest();
        manifest.qwen4_exp.output_gate_type = Some("gelu".to_string());
        expect_err_contains(&manifest, "output_gate_type");
        manifest = valid_manifest();
        manifest.model_family = "qwen3_5".to_string();
        expect_err_contains(&manifest, "qwen4_exp");
    }

    #[test]
    fn accepts_silu_and_missing_gate_type() {
        let mut manifest = valid_manifest();
        manifest.qwen4_exp.output_gate_type = Some("silu".to_string());
        validate(&manifest).expect("silu output_gate_type should pass");
        manifest = valid_manifest();
        manifest.qwen4_exp.output_gate_type = None;
        validate(&manifest).expect("absent output_gate_type falls back to silu");
    }

    #[test]
    fn rejects_bad_qsa_and_rotary() {
        let mut manifest = valid_manifest();
        manifest.qwen4_exp.indexer_budget = Some(5);
        expect_err_contains(&manifest, "indexer_budget");
        manifest = valid_manifest();
        manifest.qwen4_exp.indexer_kv_heads = Some(2);
        expect_err_contains(&manifest, "indexer_kv_heads");
        manifest = valid_manifest();
        manifest.partial_rotary_factor = Some(0.75);
        expect_err_contains(&manifest, "rotary_dim");
        manifest = valid_manifest();
        manifest.partial_rotary_factor = Some(1.0);
        validate(&manifest).expect("full rotary width fits the index head");
        manifest.qwen4_exp.indexer_head_dim = Some(2);
        expect_err_contains(&manifest, "rotary_dim");
    }

    #[test]
    fn rejects_bad_ple_ids_and_generic_norms() {
        let mut manifest = valid_manifest();
        manifest.qwen4_exp.ple_layer_ids = vec![2];
        expect_err_contains(&manifest, "linear_attention");
        manifest = valid_manifest();
        manifest.tensors.push(spec(
            "model.norm.weight",
            NativeTensorRole::FinalNorm,
            None,
            vec![8],
        ));
        expect_err_contains(&manifest, "generic role");
    }

    #[test]
    fn rejects_bad_shapes_and_stray_roles() {
        let mut manifest = valid_manifest();
        let q = manifest
            .tensors
            .iter_mut()
            .find(|t| t.role == NativeTensorRole::AttentionQ)
            .unwrap();
        q.shape = vec![8, 8];
        expect_err_contains(&manifest, "attention_q");
        manifest = valid_manifest();
        manifest.tensors.push(spec(
            "l1.stray",
            NativeTensorRole::Qwen4ExpPleKeyProj,
            Some(1),
            vec![16, 8],
        ));
        expect_err_contains(&manifest, "stray");
        manifest = valid_manifest();
        manifest.tensors.push(spec(
            "l0.stray_qk",
            NativeTensorRole::Qwen4ExpIndexerQkProj,
            Some(0),
            vec![12, 8],
        ));
        expect_err_contains(&manifest, "must not provide");
    }

    #[test]
    fn rejects_bad_ngram_shards_and_hash_dtype() {
        let mut manifest = valid_manifest();
        let table = manifest
            .tensors
            .iter_mut()
            .find(|t| t.role == NativeTensorRole::NgramEmbedding)
            .unwrap();
        table.shape = vec![8, 3];
        expect_err_contains(&manifest, "row width");
        manifest = valid_manifest();
        let mult = manifest
            .tensors
            .iter_mut()
            .find(|t| t.role == NativeTensorRole::Qwen4ExpPleMultipliers)
            .unwrap();
        mult.dtype = NativeTensorDataType::U32;
        expect_err_contains(&manifest, "i64");
        manifest = valid_manifest();
        manifest.tensors.push(spec(
            "model.layers.0.ple.ple_embedding.ngram_embedding.shard_1.weight",
            NativeTensorRole::NgramEmbedding,
            Some(0),
            vec![4, 2],
        ));
        expect_err_contains(&manifest, "mix sharded and unsharded");
    }

    #[test]
    fn rejects_mixed_moe_layout_and_missing_shared() {
        let mut manifest = valid_manifest();
        manifest.tensors.push(spec(
            "l1.extra_gate",
            NativeTensorRole::FfnGateExps,
            Some(1),
            vec![2, 8, 8],
        ));
        expect_err_contains(&manifest, "exactly one expert layout");
        manifest = valid_manifest();
        manifest
            .tensors
            .retain(|t| t.role != NativeTensorRole::FfnSharedExpertDown);
        expect_err_contains(&manifest, "ffn_shared_expert_down");
    }
}
