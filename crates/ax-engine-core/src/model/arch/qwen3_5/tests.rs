use super::*;
use crate::backend::Backend;
use crate::backend::cpu::CpuBackend;
use crate::backend::metal::MetalBackend;
use crate::gguf::MetadataValue;
use crate::gguf::header::GgufHeader;
use crate::gguf::mmap::MappedModel;
use crate::gguf::tensor::GgmlType;
use std::collections::HashMap;
use std::ffi::OsString;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

struct EnvVarRestore {
    key: &'static str,
    previous: Option<OsString>,
}

impl Drop for EnvVarRestore {
    fn drop(&mut self) {
        match &self.previous {
            Some(prev) => unsafe { std::env::set_var(self.key, prev) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

fn with_env_var<T>(key: &'static str, value: Option<&str>, f: impl FnOnce() -> T) -> T {
    let _lock = crate::test_env_lock();
    let _restore = EnvVarRestore {
        key,
        previous: std::env::var_os(key),
    };
    match value {
        Some(v) => unsafe { std::env::set_var(key, v) },
        None => unsafe { std::env::remove_var(key) },
    }
    f()
}

fn with_env_vars<T>(vars: &[(&'static str, Option<&str>)], f: impl FnOnce() -> T) -> T {
    let _lock = crate::test_env_lock();
    let _restore: Vec<EnvVarRestore> = vars
        .iter()
        .map(|(key, _)| EnvVarRestore {
            key,
            previous: std::env::var_os(key),
        })
        .collect();
    for (key, value) in vars {
        match value {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
    }
    f()
}

fn workspace_model_path(file_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .join(file_name)
}

fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn argmax_index(values: &[f32]) -> usize {
    values
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap())
        .unwrap()
        .0
}

fn build_real_qwen35_35b_a3b_layer3_prompt_state(
    cfg: &ModelConfig,
    weights: &WeightStore,
    prompt_token_ids: &[u32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, crate::kv::Qwen3_5Kv) {
    let cpu = CpuBackend;
    let layer_idx = 3usize;
    assert!(!cfg.qwen35_is_recurrent_layer(layer_idx));

    let dim = cfg.embedding_dim as usize;
    let inter_dim = cfg.expert_intermediate_dim.unwrap() as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let dims = Qwen3_5Forward::recurrent_dims(cfg).unwrap();
    let recurrent_slot = 0usize;
    let recurrent_slot_indices = [recurrent_slot];
    let full_attn_params = AttentionParams::new(n_heads, n_kv_heads, head_dim);
    let n_tokens = prompt_token_ids.len();

    let mut expected_kv = crate::kv::Qwen3_5Kv::new(
        cfg.n_layers as usize,
        cfg.n_kv_heads as usize,
        cfg.head_dim as usize,
        32,
        cfg.qwen35_full_attention_interval.unwrap() as usize,
        cfg.qwen35_ssm_conv_kernel.unwrap() as usize,
        cfg.qwen35_ssm_inner_size.unwrap() as usize,
        cfg.qwen35_ssm_state_size.unwrap() as usize,
        cfg.qwen35_ssm_time_step_rank.unwrap() as usize,
        cfg.qwen35_ssm_group_count.unwrap() as usize,
    );
    let mut input_hidden_batch = vec![0.0f32; n_tokens * dim];
    let mut hidden_before_moe_batch = vec![0.0f32; n_tokens * dim];
    let mut expected_hidden_batch = vec![0.0f32; n_tokens * dim];
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

    for (position, &token_id) in prompt_token_ids.iter().enumerate() {
        let mut hidden = vec![0.0f32; dim];
        weights
            .dequantize_row("token_embd.weight", token_id as usize, &mut hidden)
            .unwrap();

        for layer in 0..=layer_idx {
            let prefix = format!("blk.{layer}");
            apply_attention_norm_single(
                weights,
                &prefix,
                &hidden,
                &mut norm_buf,
                cfg.rms_norm_eps,
                None,
            )
            .unwrap();

            if cfg.qwen35_is_recurrent_layer(layer) {
                Qwen3_5Forward::run_recurrent_single_layer(
                    cfg,
                    &cpu,
                    weights,
                    &prefix,
                    &mut expected_kv,
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
                    None,
                )
                .unwrap();
            } else {
                input_hidden_batch[position * dim..(position + 1) * dim].copy_from_slice(&hidden);
                Qwen3_5Forward::run_full_attention_single_layer(
                    cfg,
                    &cpu,
                    weights,
                    &prefix,
                    &mut expected_kv,
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
                    None,
                )
                .unwrap();
                let mut hidden_before_moe = hidden.clone();
                silu::elementwise_add(&mut hidden_before_moe, &proj_buf);
                hidden_before_moe_batch[position * dim..(position + 1) * dim]
                    .copy_from_slice(&hidden_before_moe);
            }

            Qwen3_5Forward::apply_layer_tail_single(
                cfg,
                &cpu,
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
                None,
            )
            .unwrap();

            if layer == layer_idx {
                expected_hidden_batch[position * dim..(position + 1) * dim]
                    .copy_from_slice(&hidden);
                break;
            }
        }

        if position + 1 != n_tokens {
            expected_kv.finalize_token();
        }
    }

    (
        input_hidden_batch,
        hidden_before_moe_batch,
        expected_hidden_batch,
        expected_kv,
    )
}

struct RealQwen35Layer1ResidentMoePromptState {
    dim: usize,
    expert_inter_dim: usize,
    n_expert: usize,
    n_expert_used: usize,
    gate_stride: usize,
    up_stride: usize,
    down_stride: usize,
    rms_norm_eps: f32,
    hidden_before_moe: Vec<f32>,
    expected_hidden_after_moe: Vec<f32>,
    ffn_norm_w_buf: ax_engine_metal::MetalBuffer,
    router_buf: ax_engine_metal::MetalBuffer,
    router_dtype: GgmlType,
    gate_expert_buf: ax_engine_metal::MetalBuffer,
    gate_dtype: GgmlType,
    up_expert_buf: ax_engine_metal::MetalBuffer,
    up_dtype: GgmlType,
    down_expert_buf: ax_engine_metal::MetalBuffer,
    down_dtype: GgmlType,
    shared_gate_buf: Option<ax_engine_metal::MetalBuffer>,
    shared_up_buf: Option<ax_engine_metal::MetalBuffer>,
    shared_down_buf: Option<ax_engine_metal::MetalBuffer>,
    shared_gate_inp_buf: Option<ax_engine_metal::MetalBuffer>,
    shared_dtype: Option<GgmlType>,
    shared_gate_inp_dtype: Option<GgmlType>,
    shared_inter_dim: usize,
    shared_gate_inp_rows: usize,
}

impl RealQwen35Layer1ResidentMoePromptState {
    fn shared_expert(&self) -> Option<crate::backend::metal::SharedExpertCachedBuffers<'_>> {
        if let (Some(gate), Some(up), Some(down), Some(dtype)) = (
            self.shared_gate_buf.as_ref(),
            self.shared_up_buf.as_ref(),
            self.shared_down_buf.as_ref(),
            self.shared_dtype,
        ) {
            Some(crate::backend::metal::SharedExpertCachedBuffers {
                gate,
                up,
                down,
                gate_inp: self.shared_gate_inp_buf.as_ref(),
                gate_inp_dtype: self.shared_gate_inp_dtype,
                dtype,
                inter_dim: self.shared_inter_dim,
                gate_inp_rows: self.shared_gate_inp_rows,
            })
        } else {
            None
        }
    }
}

fn build_real_qwen35_35b_a3b_layer1_resident_moe_prompt_state(
    cfg: &ModelConfig,
    weights: &WeightStore,
    tokenizer: &crate::tokenizer::Tokenizer,
    metal_ops: &crate::backend::metal::MetalOps,
) -> RealQwen35Layer1ResidentMoePromptState {
    let cpu = CpuBackend;
    let layer_idx = 1usize;
    assert!(cfg.qwen35_is_recurrent_layer(layer_idx));

    let dim = cfg.embedding_dim as usize;
    let inter_dim = cfg.expert_intermediate_dim.unwrap() as usize;
    let dims = Qwen3_5Forward::recurrent_dims(cfg).unwrap();
    let recurrent_slot = 0usize;
    let recurrent_slot_indices = [recurrent_slot];
    let prompt_token_ids = tokenizer.encode("The capital of France is", true);
    let token_id = prompt_token_ids[0];

    let make_kv = || {
        crate::kv::Qwen3_5Kv::new(
            cfg.n_layers as usize,
            cfg.n_kv_heads as usize,
            cfg.head_dim as usize,
            8,
            cfg.qwen35_full_attention_interval.unwrap() as usize,
            cfg.qwen35_ssm_conv_kernel.unwrap() as usize,
            cfg.qwen35_ssm_inner_size.unwrap() as usize,
            cfg.qwen35_ssm_state_size.unwrap() as usize,
            cfg.qwen35_ssm_time_step_rank.unwrap() as usize,
            cfg.qwen35_ssm_group_count.unwrap() as usize,
        )
    };

    let mut cpu_kv = make_kv();
    let mut hidden = vec![0.0f32; dim];
    weights
        .dequantize_row("token_embd.weight", token_id as usize, &mut hidden)
        .unwrap();
    let mut hidden_after_layer0 = hidden;

    let mut norm_buf = vec![0.0f32; dim];
    let mut proj_buf = vec![0.0f32; dim];
    let mut gate_buf = vec![0.0f32; inter_dim];
    let mut up_buf = vec![0.0f32; inter_dim];
    let mut down_buf = vec![0.0f32; dim];
    let mut layer0_rec_qkv = vec![0.0f32; dims.conv_dim()];
    let mut layer0_rec_z = vec![0.0f32; dims.inner_size];
    let mut layer0_rec_beta = vec![0.0f32; dims.time_step_rank];
    let mut layer0_rec_alpha = vec![0.0f32; dims.time_step_rank];
    let mut layer0_rec_out = vec![0.0f32; dims.inner_size];

    let prefix0 = "blk.0".to_string();
    apply_attention_norm_single(
        weights,
        &prefix0,
        &hidden_after_layer0,
        &mut norm_buf,
        cfg.rms_norm_eps,
        None,
    )
    .unwrap();
    Qwen3_5Forward::run_recurrent_single_layer(
        cfg,
        &cpu,
        weights,
        &prefix0,
        &mut cpu_kv,
        recurrent_slot,
        0,
        0,
        dims,
        &recurrent_slot_indices,
        &norm_buf,
        &mut layer0_rec_qkv,
        &mut layer0_rec_z,
        &mut layer0_rec_beta,
        &mut layer0_rec_alpha,
        &mut layer0_rec_out,
        &mut proj_buf,
        dim,
        None,
    )
    .unwrap();
    Qwen3_5Forward::apply_layer_tail_single(
        cfg,
        &cpu,
        weights,
        &prefix0,
        &mut hidden_after_layer0,
        &proj_buf,
        &mut norm_buf,
        &mut gate_buf,
        &mut up_buf,
        &mut down_buf,
        dim,
        inter_dim,
        cfg.rms_norm_eps,
        0,
        0,
        None,
    )
    .unwrap();

    let prefix = format!("blk.{layer_idx}");
    apply_attention_norm_single(
        weights,
        &prefix,
        &hidden_after_layer0,
        &mut norm_buf,
        cfg.rms_norm_eps,
        None,
    )
    .unwrap();

    let mut cpu_rec_qkv = vec![0.0f32; dims.conv_dim()];
    let mut cpu_rec_z = vec![0.0f32; dims.inner_size];
    let mut cpu_rec_beta = vec![0.0f32; dims.time_step_rank];
    let mut cpu_rec_alpha = vec![0.0f32; dims.time_step_rank];
    let mut cpu_rec_out = vec![0.0f32; dims.inner_size];
    let mut expected_proj = vec![0.0f32; dim];
    Qwen3_5Forward::run_recurrent_single_layer(
        cfg,
        &cpu,
        weights,
        &prefix,
        &mut cpu_kv,
        recurrent_slot,
        layer_idx,
        0,
        dims,
        &recurrent_slot_indices,
        &norm_buf,
        &mut cpu_rec_qkv,
        &mut cpu_rec_z,
        &mut cpu_rec_beta,
        &mut cpu_rec_alpha,
        &mut cpu_rec_out,
        &mut expected_proj,
        dim,
        None,
    )
    .unwrap();
    let mut hidden_before_moe = hidden_after_layer0.clone();
    silu::elementwise_add(&mut hidden_before_moe, &expected_proj);

    let mut expected_hidden_after_moe = hidden_before_moe.clone();
    let mut expected_norm_buf = vec![0.0f32; dim];
    Qwen3_5Forward::apply_post_attention_moe_single(
        cfg,
        &cpu,
        weights,
        &prefix,
        &mut expected_hidden_after_moe,
        &mut expected_norm_buf,
        dim,
        inter_dim,
        cfg.rms_norm_eps,
    )
    .unwrap();

    let router_name = format!("{prefix}.ffn_gate_inp.weight");
    let gate_name = format!("{prefix}.ffn_gate_exps.weight");
    let up_name = format!("{prefix}.ffn_up_exps.weight");
    let down_name = format!("{prefix}.ffn_down_exps.weight");
    let shared_gate_name = format!("{prefix}.ffn_gate_shexp.weight");
    let shared_up_name = format!("{prefix}.ffn_up_shexp.weight");
    let shared_down_name = format!("{prefix}.ffn_down_shexp.weight");
    let shared_gate_inp_name = format!("{prefix}.ffn_gate_inp_shexp.weight");

    let n_expert = cfg.n_expert.unwrap() as usize;
    let n_expert_used = cfg.n_expert_used.unwrap() as usize;
    let expert_inter_dim = Qwen3_5Forward::tensor_output_rows(weights, &gate_name).unwrap();
    let (router_raw, router_dtype) = weights.raw_with_dtype(&router_name).unwrap();
    let (gate_raw, gate_dtype) = weights.raw_with_dtype(&gate_name).unwrap();
    let (up_raw, up_dtype) = weights.raw_with_dtype(&up_name).unwrap();
    let (down_raw, down_dtype) = weights.raw_with_dtype(&down_name).unwrap();
    let gate_stride =
        crate::model::moe_utils::expert_byte_stride(gate_dtype, expert_inter_dim * dim);
    let up_stride = crate::model::moe_utils::expert_byte_stride(up_dtype, expert_inter_dim * dim);
    let down_stride =
        crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);

    let ffn_norm_w = weights
        .f32_slice(&format!("{prefix}.post_attention_norm.weight"))
        .unwrap();
    let ffn_norm_w_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), ffn_norm_w).unwrap();
    let router_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), router_raw).unwrap();
    let gate_expert_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), gate_raw).unwrap();
    let up_expert_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), up_raw).unwrap();
    let down_expert_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), down_raw).unwrap();

    let shared_gate_buf = weights.has(&shared_gate_name).then(|| {
        let (raw, _) = weights.raw_with_dtype(&shared_gate_name).unwrap();
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), raw).unwrap()
    });
    let shared_up_buf = weights.has(&shared_up_name).then(|| {
        let (raw, _) = weights.raw_with_dtype(&shared_up_name).unwrap();
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), raw).unwrap()
    });
    let shared_down_buf = weights.has(&shared_down_name).then(|| {
        let (raw, _) = weights.raw_with_dtype(&shared_down_name).unwrap();
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), raw).unwrap()
    });
    let shared_gate_inp_buf = weights.has(&shared_gate_inp_name).then(|| {
        let (raw, _) = weights.raw_with_dtype(&shared_gate_inp_name).unwrap();
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), raw).unwrap()
    });
    let shared_dtype = weights
        .has(&shared_gate_name)
        .then(|| weights.raw_with_dtype(&shared_gate_name).unwrap().1);
    let shared_gate_inp_dtype = weights
        .has(&shared_gate_inp_name)
        .then(|| weights.raw_with_dtype(&shared_gate_inp_name).unwrap().1);

    RealQwen35Layer1ResidentMoePromptState {
        dim,
        expert_inter_dim,
        n_expert,
        n_expert_used,
        gate_stride,
        up_stride,
        down_stride,
        rms_norm_eps: cfg.rms_norm_eps,
        hidden_before_moe,
        expected_hidden_after_moe,
        ffn_norm_w_buf,
        router_buf,
        router_dtype,
        gate_expert_buf,
        gate_dtype,
        up_expert_buf,
        up_dtype,
        down_expert_buf,
        down_dtype,
        shared_gate_buf,
        shared_up_buf,
        shared_down_buf,
        shared_gate_inp_buf,
        shared_dtype,
        shared_gate_inp_dtype,
        shared_inter_dim: if weights.has(&shared_gate_name) {
            Qwen3_5Forward::tensor_output_rows(weights, &shared_gate_name).unwrap()
        } else {
            0
        },
        shared_gate_inp_rows: if weights.has(&shared_gate_inp_name) {
            Qwen3_5Forward::tensor_output_rows(weights, &shared_gate_inp_name).unwrap()
        } else {
            0
        },
    }
}

fn run_real_qwen35_layer1_resident_moe_case(
    metal_ops: &crate::backend::metal::MetalOps,
    fixture: &RealQwen35Layer1ResidentMoePromptState,
    envs: &[(&'static str, Option<&str>)],
    iterations: usize,
) -> (std::time::Duration, Vec<f32>) {
    with_env_vars(envs, || {
        let mut hidden_gpu = ax_engine_metal::MetalBuffer::new(
            metal_ops.device.device(),
            fixture.dim * std::mem::size_of::<f32>(),
        )
        .unwrap();
        let shared_expert = fixture.shared_expert();
        let mut output = vec![0.0f32; fixture.dim];
        let start = Instant::now();
        for _ in 0..iterations {
            unsafe {
                hidden_gpu.as_mut_slice::<f32>()[..fixture.dim]
                    .copy_from_slice(&fixture.hidden_before_moe);
            }
            metal_ops
                .moe_ffn_gpu_resident_cached(
                    &hidden_gpu,
                    &fixture.ffn_norm_w_buf,
                    &fixture.router_buf,
                    fixture.router_dtype,
                    &fixture.gate_expert_buf,
                    fixture.gate_dtype,
                    &fixture.up_expert_buf,
                    fixture.up_dtype,
                    &fixture.down_expert_buf,
                    fixture.down_dtype,
                    1,
                    fixture.n_expert,
                    fixture.n_expert_used,
                    fixture.dim,
                    fixture.expert_inter_dim,
                    fixture.gate_stride,
                    fixture.up_stride,
                    fixture.down_stride,
                    fixture.rms_norm_eps,
                    shared_expert.as_ref(),
                )
                .unwrap();
        }
        let elapsed = start.elapsed();
        output.copy_from_slice(unsafe { &hidden_gpu.as_slice::<f32>()[..fixture.dim] });
        (elapsed, output)
    })
}

fn summarize_qwen35_state_diffs(
    cfg: &ModelConfig,
    cpu_kv: &crate::kv::ModelKv,
    actual_kv: &mut crate::kv::ModelKv,
) -> String {
    {
        let actual_qwen = actual_kv
            .as_qwen35_mut()
            .expect("expected qwen35 kv for state diff summary");
        actual_qwen.sync_attention_cpu_from_gpu_if_needed();
    }
    let cpu_qwen = cpu_kv
        .as_qwen35()
        .expect("expected qwen35 kv for cpu state diff summary");
    let actual_qwen = actual_kv
        .as_qwen35()
        .expect("expected qwen35 kv for gpu state diff summary");
    let seq_len = cpu_qwen.seq_len();
    let mut max_attn_k = (usize::MAX, 0.0f32);
    let mut max_attn_v = (usize::MAX, 0.0f32);
    let mut max_conv = (usize::MAX, 0.0f32);
    let mut max_recurrent = (usize::MAX, 0.0f32);

    for layer in 0..cfg.n_layers as usize {
        let attn_k_diff = max_abs_diff(
            cpu_qwen.attention_k_slice_including_current(layer, seq_len),
            actual_qwen.attention_k_slice_including_current(layer, seq_len),
        );
        if attn_k_diff > max_attn_k.1 {
            max_attn_k = (layer, attn_k_diff);
        }

        let attn_v_diff = max_abs_diff(
            cpu_qwen.attention_v_slice_including_current(layer, seq_len),
            actual_qwen.attention_v_slice_including_current(layer, seq_len),
        );
        if attn_v_diff > max_attn_v.1 {
            max_attn_v = (layer, attn_v_diff);
        }

        if cfg.qwen35_is_recurrent_layer(layer) {
            let conv_diff = max_abs_diff(
                cpu_qwen.conv_state_for_slot(0, layer),
                actual_qwen.conv_state_for_slot(0, layer),
            );
            if conv_diff > max_conv.1 {
                max_conv = (layer, conv_diff);
            }

            let recurrent_diff = max_abs_diff(
                cpu_qwen.recurrent_state_for_slot(0, layer),
                actual_qwen.recurrent_state_for_slot(0, layer),
            );
            if recurrent_diff > max_recurrent.1 {
                max_recurrent = (layer, recurrent_diff);
            }
        }
    }

    format!(
        "state_diffs seq_len={seq_len} max_attn_k=blk.{}:{:.6} max_attn_v=blk.{}:{:.6} max_conv=blk.{}:{:.6} max_recurrent=blk.{}:{:.6}",
        max_attn_k.0,
        max_attn_k.1,
        max_attn_v.0,
        max_attn_v.1,
        max_conv.0,
        max_conv.1,
        max_recurrent.0,
        max_recurrent.1,
    )
}

#[allow(clippy::too_many_arguments)]
fn assert_qwen35_serial_trace_step_matches(
    cfg: &ModelConfig,
    tokenizer: &crate::tokenizer::Tokenizer,
    metal_model: &crate::model::InferenceModel,
    stage: &str,
    position: usize,
    input_token: u32,
    cpu_logits: &[f32],
    metal_logits: &[f32],
    cpu_kv: &crate::kv::ModelKv,
    metal_kv: &mut crate::kv::ModelKv,
) {
    let expected_argmax = argmax_index(cpu_logits);
    let actual_argmax = argmax_index(metal_logits);
    let max_diff = max_abs_diff(cpu_logits, metal_logits);
    let scale = cpu_logits
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    let rel_diff = max_diff / scale;
    if expected_argmax != actual_argmax || rel_diff > 5e-2 {
        metal_model.sync_model_kv(metal_kv);
        let state_summary = summarize_qwen35_state_diffs(cfg, cpu_kv, metal_kv);
        panic!(
            "real Qwen3.5-35B-A3B serial native trace mismatch at stage={stage} position={position} input_token={input_token} input_text={:?} expected_argmax={} expected_text={:?} actual_argmax={} actual_text={:?} rel_diff={} max_diff={} cpu_seq_len={} gpu_seq_len={} {}",
            tokenizer.decode(&[input_token]),
            expected_argmax,
            tokenizer.decode(&[expected_argmax as u32]),
            actual_argmax,
            tokenizer.decode(&[actual_argmax as u32]),
            rel_diff,
            max_diff,
            cpu_kv.seq_len(),
            metal_kv.seq_len(),
            state_summary,
        );
    }
}

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

fn push_tensor_info(buf: &mut Vec<u8>, name: &str, shape: &[u64], dtype: GgmlType, offset: u64) {
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
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn quantize_q8_0_rows(values: &[f32], row_width: usize) -> Vec<u8> {
    assert!(
        row_width.is_multiple_of(32),
        "Q8_0 row width must be a multiple of 32"
    );
    assert_eq!(
        values.len() % row_width,
        0,
        "Q8_0 input length must be divisible by row width"
    );

    let mut bytes = Vec::with_capacity(values.len() / 32 * 34);
    for row in values.chunks_exact(row_width) {
        for block in row.chunks_exact(32) {
            let max_abs = block
                .iter()
                .fold(0.0f32, |acc, &value| acc.max(value.abs()));
            let scale = if max_abs == 0.0 {
                1.0
            } else {
                (max_abs / 127.0).max(1.0)
            };
            bytes.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
            for &value in block {
                let quant = (value / scale).round().clamp(-127.0, 127.0) as i8;
                bytes.push(quant as u8);
            }
        }
    }
    bytes
}

fn build_qwen35_logits_test_gguf_with_dtype(
    output_norm: &[f32],
    output_weight_bytes: &[u8],
    output_weight_dtype: GgmlType,
    dim: usize,
    vocab_size: usize,
) -> Vec<u8> {
    let alignment = 32usize;
    let output_norm_bytes = f32_bytes(output_norm);
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
        output_weight_dtype,
        output_weight_offset as u64,
    );
    let data_start = align_to(buf.len(), alignment);
    buf.resize(data_start, 0);
    buf.extend_from_slice(&output_norm_bytes);
    buf.resize(data_start + output_weight_offset, 0);
    buf.extend_from_slice(output_weight_bytes);
    buf
}

fn build_qwen35_logits_test_gguf(
    output_norm: &[f32],
    output_weight: &[f32],
    dim: usize,
    vocab_size: usize,
) -> Vec<u8> {
    let output_weight_bytes = f32_bytes(output_weight);
    build_qwen35_logits_test_gguf_with_dtype(
        output_norm,
        &output_weight_bytes,
        GgmlType::F32,
        dim,
        vocab_size,
    )
}

fn build_test_gguf_with_tensors(
    architecture: &str,
    tensors: &[(&str, &[u64], GgmlType, Vec<u8>)],
) -> Vec<u8> {
    let alignment = 32usize;
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut data_cursor = 0usize;
    for (_, _, _, bytes) in tensors {
        offsets.push(data_cursor);
        data_cursor = align_to(data_cursor + bytes.len(), alignment);
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(&crate::gguf::GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&crate::gguf::GGUF_VERSION.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    buf.extend_from_slice(&2u64.to_le_bytes());
    push_string_metadata(&mut buf, "general.architecture", architecture);
    push_u32_metadata(&mut buf, "general.alignment", alignment as u32);
    for ((name, shape, dtype, _), offset) in tensors.iter().zip(offsets.iter()) {
        push_tensor_info(&mut buf, name, shape, *dtype, *offset as u64);
    }

    let data_start = align_to(buf.len(), alignment);
    buf.resize(data_start, 0);
    for (((_, _, _, bytes), offset), next_offset) in tensors
        .iter()
        .zip(offsets.iter())
        .zip(offsets.iter().skip(1).chain(std::iter::once(&data_cursor)))
    {
        let target = data_start + *offset;
        if buf.len() < target {
            buf.resize(target, 0);
        }
        buf.extend_from_slice(bytes);
        let next_target = data_start + *next_offset;
        if buf.len() < next_target {
            buf.resize(next_target, 0);
        }
    }
    buf
}

#[test]
fn test_qwen35_prefill_recurrent_state_mode_defaults_to_auto() {
    with_env_var("AX_QWEN35_PREFILL_RECURRENT_STATE_MODE", None, || {
        assert_eq!(
            Qwen3_5Forward::qwen35_prefill_recurrent_state_mode(),
            Qwen35PrefillRecurrentStateMode::Auto
        );
    });
}

#[test]
fn test_qwen35_prefill_recurrent_state_mode_parses_slot_buffer() {
    with_env_var(
        "AX_QWEN35_PREFILL_RECURRENT_STATE_MODE",
        Some("slot_buffer"),
        || {
            assert_eq!(
                Qwen3_5Forward::qwen35_prefill_recurrent_state_mode(),
                Qwen35PrefillRecurrentStateMode::SlotBuffer
            );
        },
    );
}

#[test]
fn test_qwen35_prefill_recurrent_state_mode_parses_backend_owned() {
    with_env_var(
        "AX_QWEN35_PREFILL_RECURRENT_STATE_MODE",
        Some("backend_owned"),
        || {
            assert_eq!(
                Qwen3_5Forward::qwen35_prefill_recurrent_state_mode(),
                Qwen35PrefillRecurrentStateMode::BackendOwned
            );
        },
    );
}

#[test]
fn test_qwen35_prefill_recurrent_state_mode_auto_is_prompt_aware() {
    with_env_var("AX_QWEN35_PREFILL_RECURRENT_STATE_MODE", None, || {
        assert_eq!(
            Qwen3_5Forward::qwen35_prefill_recurrent_state_mode_for_tokens(32),
            Qwen35PrefillRecurrentStateMode::BackendOwned
        );
        assert_eq!(
            Qwen3_5Forward::qwen35_prefill_recurrent_state_mode_for_tokens(64),
            Qwen35PrefillRecurrentStateMode::BackendOwned
        );
        assert_eq!(
            Qwen3_5Forward::qwen35_prefill_recurrent_state_mode_for_tokens(96),
            Qwen35PrefillRecurrentStateMode::BackendOwned
        );
        assert_eq!(
            Qwen3_5Forward::qwen35_prefill_recurrent_state_mode_for_tokens(128),
            Qwen35PrefillRecurrentStateMode::BackendOwned
        );
    });
}

#[test]
fn test_qwen35_prefill_recurrent_state_mode_auto_preserves_backend_owned_owner() {
    with_env_var("AX_QWEN35_PREFILL_RECURRENT_STATE_MODE", None, || {
        assert_eq!(
            Qwen3_5Forward::resolve_qwen35_prefill_recurrent_state_mode(
                64,
                crate::kv::Qwen3_5LayerStateOwner::BackendOwned,
            ),
            Qwen35PrefillRecurrentStateMode::BackendOwned
        );
        assert_eq!(
            Qwen3_5Forward::resolve_qwen35_prefill_recurrent_state_mode(
                64,
                crate::kv::Qwen3_5LayerStateOwner::Split,
            ),
            Qwen35PrefillRecurrentStateMode::BackendOwned
        );
    });
}

#[test]
fn test_qwen35_prefill_force_backend_state_batch_defaults_off() {
    with_env_var("AX_QWEN35_PREFILL_FORCE_BACKEND_STATE_BATCH", None, || {
        assert!(!Qwen3_5Forward::qwen35_prefill_force_backend_state_batch());
    });
}

#[test]
fn test_qwen35_prefill_force_backend_state_batch_parses_on() {
    with_env_var(
        "AX_QWEN35_PREFILL_FORCE_BACKEND_STATE_BATCH",
        Some("1"),
        || {
            assert!(Qwen3_5Forward::qwen35_prefill_force_backend_state_batch());
        },
    );
}

#[test]
fn test_qwen35_prefill_backend_state_batch_auto_prefers_backend_owned_layers() {
    with_env_var("AX_QWEN35_PREFILL_FORCE_BACKEND_STATE_BATCH", None, || {
        assert!(
            Qwen3_5Forward::qwen35_prefill_backend_state_batch_for_tokens(
                32,
                crate::kv::Qwen3_5LayerStateOwner::BackendOwned,
            )
        );
        assert!(
            Qwen3_5Forward::qwen35_prefill_backend_state_batch_for_tokens(
                64,
                crate::kv::Qwen3_5LayerStateOwner::Split,
            )
        );
        assert!(
            Qwen3_5Forward::qwen35_prefill_backend_state_batch_for_tokens(
                128,
                crate::kv::Qwen3_5LayerStateOwner::BackendOwned,
            )
        );
        assert!(
            Qwen3_5Forward::qwen35_prefill_backend_state_batch_for_tokens(
                256,
                crate::kv::Qwen3_5LayerStateOwner::Split,
            )
        );
        assert!(
            !Qwen3_5Forward::qwen35_prefill_backend_state_batch_for_tokens(
                64,
                crate::kv::Qwen3_5LayerStateOwner::CpuMaterialized,
            )
        );
    });
}

#[test]
fn test_qwen35_fused_recurrent_gpu_candidate_enabled_for_backend_state_batch() {
    assert!(Qwen3_5Forward::qwen35_fused_recurrent_gpu_candidate(
        1,
        true,
        &[2, 3],
        true,
        true,
    ));
}

#[test]
fn test_qwen35_fused_recurrent_gpu_candidate_enabled_when_eligible() {
    assert!(Qwen3_5Forward::qwen35_fused_recurrent_gpu_candidate(
        1,
        true,
        &[2, 3],
        true,
        true,
    ));
}

#[test]
fn test_qwen35_unified_recurrent_gpu_plan_tracks_fused_dense_fast_path_inputs() {
    let plan = Qwen3_5Forward::qwen35_build_unified_recurrent_gpu_plan(
        1,
        &[0, 1, 2, 3],
        &[2, 3],
        true,
        true,
    );
    assert_eq!(
        plan,
        Qwen35UnifiedRecurrentGpuPlan {
            qkv_gpu_fast_path_enabled: true,
            keep_rec_z_on_gpu: true,
            fused_gpu_recurrent_layer_candidate: true,
            fused_gpu_recurrent_layer_uses_gpu_alpha_beta: true,
        }
    );
}

#[test]
fn test_qwen35_unified_recurrent_gpu_plan_disables_fused_path_for_multi_slot() {
    let plan = Qwen3_5Forward::qwen35_build_unified_recurrent_gpu_plan(
        2,
        &[0, 1, 2, 3],
        &[2, 3],
        true,
        true,
    );
    assert_eq!(
        plan,
        Qwen35UnifiedRecurrentGpuPlan {
            qkv_gpu_fast_path_enabled: true,
            keep_rec_z_on_gpu: false,
            fused_gpu_recurrent_layer_candidate: false,
            fused_gpu_recurrent_layer_uses_gpu_alpha_beta: true,
        }
    );
}

#[test]
fn test_qwen35_recurrent_projection_readback_plan_skips_fast_qkv_and_gpu_alpha_beta() {
    let plan = Qwen3_5Forward::qwen35_build_unified_recurrent_gpu_plan(
        1,
        &[0, 1, 2, 3],
        &[2, 3],
        true,
        true,
    );
    assert!(!Qwen3_5Forward::qwen35_should_readback_recurrent_projection(0, plan));
    assert!(!Qwen3_5Forward::qwen35_should_readback_recurrent_projection(1, plan));
    assert!(!Qwen3_5Forward::qwen35_should_readback_recurrent_projection(2, plan));
    assert!(!Qwen3_5Forward::qwen35_should_readback_recurrent_projection(3, plan));
}

#[test]
fn test_qwen35_recurrent_projection_readback_plan_reads_back_when_gpu_fusion_is_unavailable() {
    let plan = Qwen3_5Forward::qwen35_build_unified_recurrent_gpu_plan(
        1,
        &[0, 1, 2, 3],
        &[0, 1],
        false,
        true,
    );
    assert!(Qwen3_5Forward::qwen35_should_readback_recurrent_projection(
        0, plan
    ));
    assert!(Qwen3_5Forward::qwen35_should_readback_recurrent_projection(
        1, plan
    ));
    assert!(Qwen3_5Forward::qwen35_should_readback_recurrent_projection(
        2, plan
    ));
    assert!(Qwen3_5Forward::qwen35_should_readback_recurrent_projection(
        3, plan
    ));
}

#[test]
fn test_qwen35_unified_recurrent_tail_buffer_plan_aliases_only_dense_gpu_tail() {
    let plan = Qwen3_5Forward::qwen35_build_unified_recurrent_tail_buffer_plan(
        true, false, true, false, false,
    );
    assert_eq!(
        plan,
        Qwen35UnifiedRecurrentTailBufferPlan {
            use_unified_tail: true,
            alias_rec_out: true,
            alias_rec_z: true,
        }
    );
}

#[test]
fn test_qwen35_unified_recurrent_tail_buffer_plan_disables_aliases_for_moe_legacy_tail() {
    let plan = Qwen3_5Forward::qwen35_build_unified_recurrent_tail_buffer_plan(
        true, true, true, false, false,
    );
    assert_eq!(
        plan,
        Qwen35UnifiedRecurrentTailBufferPlan {
            use_unified_tail: false,
            alias_rec_out: false,
            alias_rec_z: false,
        }
    );
}

#[test]
fn test_qwen35_should_gpu_unpack_recurrent_output_disables_alias_for_backend_native_batch() {
    assert!(Qwen3_5Forward::qwen35_should_gpu_unpack_recurrent_output(
        false, 8, false,
    ));
    assert!(!Qwen3_5Forward::qwen35_should_gpu_unpack_recurrent_output(
        false, 8, true,
    ));
    assert!(!Qwen3_5Forward::qwen35_should_gpu_unpack_recurrent_output(
        true, 8, false,
    ));
    assert!(!Qwen3_5Forward::qwen35_should_gpu_unpack_recurrent_output(
        false, 1, false,
    ));
}

#[test]
fn test_qwen35_gpu_qkv_runtime_handoff_remains_enabled_for_moe_legacy_tail() {
    assert!(Qwen3_5Forward::qwen35_should_allow_gpu_qkv_runtime_handoff(
        false,
    ));
    assert!(Qwen3_5Forward::qwen35_should_allow_gpu_qkv_runtime_handoff(
        true,
    ));
    assert!(!Qwen3_5Forward::qwen35_should_use_unified_recurrent_tail(
        true, true,
    ));
    assert!(
        !Qwen3_5Forward::qwen35_should_keep_recurrent_output_on_gpu_for_unified_tail(
            true, true, true,
        )
    );
}

#[test]
fn test_qwen35_gpu_qkv_fast_path_policy_enables_eligible_check() {
    with_env_var("AX_QWEN35_GPU_QKV_FAST_PATH", None, || {
        assert!(Qwen3_5Forward::qwen35_should_enable_gpu_qkv_fast_path(
            Some(Qwen35RecurrentQkvFastPathCheck::default(),)
        ));
    });
    with_env_var("AX_QWEN35_GPU_QKV_FAST_PATH", Some("0"), || {
        assert!(!Qwen3_5Forward::qwen35_should_enable_gpu_qkv_fast_path(
            Some(Qwen35RecurrentQkvFastPathCheck::default(),)
        ));
    });
    with_env_var("AX_QWEN35_GPU_QKV_FAST_PATH", Some("1"), || {
        assert!(Qwen3_5Forward::qwen35_should_enable_gpu_qkv_fast_path(
            Some(Qwen35RecurrentQkvFastPathCheck {
                state_size_too_large: true,
                ..Default::default()
            },)
        ));
    });
}

#[test]
fn test_qwen35_gpu_qkv_fast_path_policy_rejects_ineligible_check() {
    assert!(!Qwen3_5Forward::qwen35_should_enable_gpu_qkv_fast_path(
        Some(Qwen35RecurrentQkvFastPathCheck {
            state_size_too_large: true,
            ..Default::default()
        },)
    ));
    assert!(!Qwen3_5Forward::qwen35_should_enable_gpu_qkv_fast_path(
        None
    ));
}

#[test]
fn test_qwen35_merged_projection_fused_recurrent_layer_is_dense_only() {
    with_env_var("AX_QWEN35_MERGED_FUSED_RECURRENT", None, || {
        assert!(!Qwen3_5Forward::qwen35_should_try_merged_projection_fused_recurrent_layer(false));
        assert!(!Qwen3_5Forward::qwen35_should_try_merged_projection_fused_recurrent_layer(true));
    });
    with_env_var("AX_QWEN35_MERGED_FUSED_RECURRENT", Some("0"), || {
        assert!(!Qwen3_5Forward::qwen35_should_try_merged_projection_fused_recurrent_layer(false));
        assert!(!Qwen3_5Forward::qwen35_should_try_merged_projection_fused_recurrent_layer(true));
    });
    with_env_var("AX_QWEN35_MERGED_FUSED_RECURRENT", Some("1"), || {
        assert!(Qwen3_5Forward::qwen35_should_try_merged_projection_fused_recurrent_layer(false));
        assert!(!Qwen3_5Forward::qwen35_should_try_merged_projection_fused_recurrent_layer(true));
    });
}

#[test]
fn test_qwen35_moe_merged_projection_fused_recurrent_layer_defaults_on_with_disable_override() {
    with_env_var("AX_QWEN35_MOE_MERGED_FUSED_RECURRENT", None, || {
        assert!(Qwen3_5Forward::qwen35_should_try_merged_projection_fused_recurrent_layer(true));
    });
    with_env_var("AX_QWEN35_MOE_MERGED_FUSED_RECURRENT", Some("0"), || {
        assert!(!Qwen3_5Forward::qwen35_should_try_merged_projection_fused_recurrent_layer(true));
    });
    with_env_var("AX_QWEN35_MOE_MERGED_FUSED_RECURRENT", Some("1"), || {
        assert!(Qwen3_5Forward::qwen35_should_try_merged_projection_fused_recurrent_layer(true));
    });
}

#[test]
fn test_qwen35_projected_moe_gpu_tail_defaults_on_for_batch() {
    with_env_var("AX_QWEN35_PROJECTED_MOE_GPU_TAIL", None, || {
        assert!(!Qwen3_5Forward::qwen35_should_try_projected_moe_gpu_tail(1));
        assert!(Qwen3_5Forward::qwen35_should_try_projected_moe_gpu_tail(
            128
        ));
    });
}

#[test]
fn test_qwen35_projected_moe_gpu_tail_env_override_can_disable_or_enable() {
    with_env_var("AX_QWEN35_PROJECTED_MOE_GPU_TAIL", Some("0"), || {
        assert!(!Qwen3_5Forward::qwen35_should_try_projected_moe_gpu_tail(1));
        assert!(!Qwen3_5Forward::qwen35_should_try_projected_moe_gpu_tail(
            128
        ));
    });
    with_env_var("AX_QWEN35_PROJECTED_MOE_GPU_TAIL", Some("1"), || {
        assert!(!Qwen3_5Forward::qwen35_should_try_projected_moe_gpu_tail(1));
        assert!(Qwen3_5Forward::qwen35_should_try_projected_moe_gpu_tail(
            128
        ));
    });
}

#[test]
fn test_qwen35_resident_moe_gpu_tail_defaults_on_for_moe_ssm_layers() {
    with_env_var("AX_QWEN35_RESIDENT_MOE_GPU_TAIL", None, || {
        assert!(Qwen3_5Forward::qwen35_should_try_resident_moe_gpu_tail(
            true, true
        ));
        assert!(!Qwen3_5Forward::qwen35_should_try_resident_moe_gpu_tail(
            false, true
        ));
        assert!(!Qwen3_5Forward::qwen35_should_try_resident_moe_gpu_tail(
            true, false
        ));
    });
}

#[test]
fn test_qwen35_resident_moe_gpu_tail_env_override_can_disable() {
    with_env_var("AX_QWEN35_RESIDENT_MOE_GPU_TAIL", Some("0"), || {
        assert!(!Qwen3_5Forward::qwen35_should_try_resident_moe_gpu_tail(
            true, true
        ));
    });
}

#[test]
fn test_qwen35_unified_recurrent_moe_gpu_tail_defaults_off() {
    with_env_var("AX_QWEN35_UNIFIED_RECURRENT_MOE_GPU_TAIL", None, || {
        assert!(!Qwen3_5Forward::qwen35_should_use_unified_recurrent_moe_gpu_tail());
    });
}

#[test]
fn test_qwen35_unified_recurrent_moe_gpu_tail_env_override_can_enable() {
    with_env_var(
        "AX_QWEN35_UNIFIED_RECURRENT_MOE_GPU_TAIL",
        Some("0"),
        || {
            assert!(!Qwen3_5Forward::qwen35_should_use_unified_recurrent_moe_gpu_tail());
        },
    );
    with_env_var(
        "AX_QWEN35_UNIFIED_RECURRENT_MOE_GPU_TAIL",
        Some("1"),
        || {
            assert!(Qwen3_5Forward::qwen35_should_use_unified_recurrent_moe_gpu_tail());
        },
    );
}

#[test]
fn test_qwen35_prefill_alpha_beta_storage_mode_defaults_to_auto() {
    with_env_var("AX_QWEN35_PREFILL_ALPHA_BETA_STORAGE_MODE", None, || {
        assert_eq!(
            Qwen3_5Forward::qwen35_prefill_alpha_beta_storage_mode(),
            Qwen35PrefillAlphaBetaStorageMode::Auto
        );
    });
}

#[test]
fn test_qwen35_prefill_alpha_beta_storage_mode_parses_f16() {
    with_env_var(
        "AX_QWEN35_PREFILL_ALPHA_BETA_STORAGE_MODE",
        Some("f16"),
        || {
            assert_eq!(
                Qwen3_5Forward::qwen35_prefill_alpha_beta_storage_mode(),
                Qwen35PrefillAlphaBetaStorageMode::F16
            );
        },
    );
}

#[test]
fn test_prepare_qwen35_handoff_alpha_beta_uses_projected_inputs() {
    let mut alpha = vec![0.0f32; 4];
    let mut beta = vec![0.0f32; 4];
    let alpha_src = vec![0.25f32, -0.5, 1.0, -1.5];
    let beta_src = vec![-2.0f32, -0.25, 0.5, 2.0];
    let dt_bias = vec![0.1f32, -0.2];
    let ssm_a = vec![0.5f32, 1.5];

    Qwen3_5Forward::prepare_qwen35_handoff_alpha_beta(
        &mut alpha, &mut beta, &alpha_src, &beta_src, &dt_bias, &ssm_a,
    );

    let mut expected_alpha = alpha_src.clone();
    let mut expected_beta = beta_src.clone();
    crate::compute::gdn::prepare_alpha_beta(
        &mut expected_alpha,
        &mut expected_beta,
        &dt_bias,
        &ssm_a,
    );

    assert_eq!(alpha, expected_alpha);
    assert_eq!(beta, expected_beta);
    assert_ne!(alpha, vec![0.0; 4], "alpha should reflect projected input");
    assert_ne!(beta, vec![0.0; 4], "beta should reflect projected input");
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
    let fwd = Qwen3_5Forward;
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
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };
    fwd.validate_config(&cfg).unwrap();
}

#[test]
fn test_qwen35_validate_accepts_qwen35moe_architecture() {
    let fwd = Qwen3_5Forward;
    let cfg = ModelConfig {
        architecture: "qwen35moe".into(),
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
        n_expert: Some(128),
        n_expert_used: Some(8),
        expert_intermediate_dim: Some(128),
        qwen35_full_attention_interval: Some(4),
        qwen35_ssm_conv_kernel: Some(4),
        qwen35_ssm_inner_size: Some(1024),
        qwen35_ssm_state_size: Some(128),
        qwen35_ssm_time_step_rank: Some(8),
        qwen35_ssm_group_count: Some(2),
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };
    fwd.validate_config(&cfg).unwrap();
}

#[test]
fn test_qwen35moe_shared_expert_gate_scales_output() {
    let _lock = crate::test_env_lock();
    let dim = 2usize;
    let shape_norm = [dim as u64];
    let shape_router = [dim as u64, 2];
    let shape_expert_in = [dim as u64, 1, 2];
    let shape_expert_out = [1, dim as u64, 2];
    let shape_shared_in = [dim as u64, 1];
    let shape_shared_out = [1, dim as u64];
    let shape_shared_gate = [dim as u64];
    let tensors: Vec<(&str, &[u64], GgmlType, Vec<u8>)> = vec![
        (
            "blk.0.post_attention_norm.weight",
            &shape_norm,
            GgmlType::F32,
            f32_bytes(&[1.0, 1.0]),
        ),
        (
            "blk.0.ffn_gate_inp.weight",
            &shape_router,
            GgmlType::F32,
            f32_bytes(&[0.0, 0.0, 0.0, 0.0]),
        ),
        (
            "blk.0.ffn_gate_exps.weight",
            &shape_expert_in,
            GgmlType::F32,
            f32_bytes(&[0.0, 0.0, 0.0, 0.0]),
        ),
        (
            "blk.0.ffn_up_exps.weight",
            &shape_expert_in,
            GgmlType::F32,
            f32_bytes(&[0.0, 0.0, 0.0, 0.0]),
        ),
        (
            "blk.0.ffn_down_exps.weight",
            &shape_expert_out,
            GgmlType::F32,
            f32_bytes(&[0.0, 0.0, 0.0, 0.0]),
        ),
        (
            "blk.0.ffn_gate_shexp.weight",
            &shape_shared_in,
            GgmlType::F32,
            f32_bytes(&[1.0, 0.0]),
        ),
        (
            "blk.0.ffn_up_shexp.weight",
            &shape_shared_in,
            GgmlType::F32,
            f32_bytes(&[2.0, 0.0]),
        ),
        (
            "blk.0.ffn_down_shexp.weight",
            &shape_shared_out,
            GgmlType::F32,
            f32_bytes(&[1.0, 1.0]),
        ),
        (
            "blk.0.ffn_gate_inp_shexp.weight",
            &shape_shared_gate,
            GgmlType::F32,
            f32_bytes(&[0.0, 0.0]),
        ),
    ];
    let gguf = build_test_gguf_with_tensors("qwen35moe", &tensors);
    let path = write_test_gguf_to_temp(&gguf);

    let cfg = ModelConfig {
        architecture: "qwen35moe".into(),
        n_layers: 1,
        n_heads: 1,
        n_kv_heads: 1,
        embedding_dim: dim as u32,
        head_dim: dim as u32,
        intermediate_dim: 1,
        context_length: 128,
        vocab_size: 16,
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
        n_expert: Some(2),
        n_expert_used: Some(1),
        expert_intermediate_dim: Some(1),
        qwen35_full_attention_interval: Some(4),
        qwen35_ssm_conv_kernel: Some(4),
        qwen35_ssm_inner_size: Some(16),
        qwen35_ssm_state_size: Some(4),
        qwen35_ssm_time_step_rank: Some(4),
        qwen35_ssm_group_count: Some(1),
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };

    let mut hidden = vec![1.0f32, 0.0];
    let mut norm_buf = vec![0.0f32; dim];
    let mut gate_buf = vec![0.0f32; 1];
    let mut up_buf = vec![0.0f32; 1];
    let mut down_buf = vec![0.0f32; dim];

    {
        let model = MappedModel::open(&path).unwrap();
        let weights = WeightStore::new(&model);

        Qwen3_5Forward::apply_post_attention_ffn_single(
            &cfg,
            &CpuBackend,
            &weights,
            "blk.0",
            &mut hidden,
            &mut norm_buf,
            &mut gate_buf,
            &mut up_buf,
            &mut down_buf,
            dim,
            1,
            cfg.rms_norm_eps,
            None,
        )
        .unwrap();
    }

    let mut norm = vec![0.0f32; dim];
    rms_norm::rms_norm_out(&[1.0f32, 0.0], &[1.0f32, 1.0], &mut norm, cfg.rms_norm_eps);
    let mut shared_gate = vec![norm[0]];
    let shared_up = vec![2.0 * norm[0]];
    silu::silu_elementwise_mul(&mut shared_gate, &shared_up);
    let expected_add = shared_gate[0] * 0.5;
    let expected = [1.0 + expected_add, expected_add];

    for (actual, expected) in hidden.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_qwen35moe_shared_expert_gate_batch_uses_post_attention_norm_input() {
    let _lock = crate::test_env_lock();
    let dim = 2usize;
    let n_tokens = 2usize;
    let shape_norm = [dim as u64];
    let shape_router = [dim as u64, 2];
    let shape_expert_in = [dim as u64, 1, 2];
    let shape_expert_out = [1, dim as u64, 2];
    let shape_shared_in = [dim as u64, 1];
    let shape_shared_out = [1, dim as u64];
    let shape_shared_gate = [dim as u64];
    let tensors: Vec<(&str, &[u64], GgmlType, Vec<u8>)> = vec![
        (
            "blk.0.post_attention_norm.weight",
            &shape_norm,
            GgmlType::F32,
            f32_bytes(&[1.0, 1.0]),
        ),
        (
            "blk.0.ffn_gate_inp.weight",
            &shape_router,
            GgmlType::F32,
            f32_bytes(&[0.0, 0.0, 0.0, 0.0]),
        ),
        (
            "blk.0.ffn_gate_exps.weight",
            &shape_expert_in,
            GgmlType::F32,
            f32_bytes(&[0.0, 0.0, 0.0, 0.0]),
        ),
        (
            "blk.0.ffn_up_exps.weight",
            &shape_expert_in,
            GgmlType::F32,
            f32_bytes(&[0.0, 0.0, 0.0, 0.0]),
        ),
        (
            "blk.0.ffn_down_exps.weight",
            &shape_expert_out,
            GgmlType::F32,
            f32_bytes(&[0.0, 0.0, 0.0, 0.0]),
        ),
        (
            "blk.0.ffn_gate_shexp.weight",
            &shape_shared_in,
            GgmlType::F32,
            f32_bytes(&[1.0, 0.0]),
        ),
        (
            "blk.0.ffn_up_shexp.weight",
            &shape_shared_in,
            GgmlType::F32,
            f32_bytes(&[2.0, 0.0]),
        ),
        (
            "blk.0.ffn_down_shexp.weight",
            &shape_shared_out,
            GgmlType::F32,
            f32_bytes(&[1.0, 1.0]),
        ),
        (
            "blk.0.ffn_gate_inp_shexp.weight",
            &shape_shared_gate,
            GgmlType::F32,
            f32_bytes(&[1.0, 0.0]),
        ),
    ];
    let gguf = build_test_gguf_with_tensors("qwen35moe", &tensors);
    let path = write_test_gguf_to_temp(&gguf);

    let cfg = ModelConfig {
        architecture: "qwen35moe".into(),
        n_layers: 1,
        n_heads: 1,
        n_kv_heads: 1,
        embedding_dim: dim as u32,
        head_dim: dim as u32,
        intermediate_dim: 1,
        context_length: 128,
        vocab_size: 16,
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
        n_expert: Some(2),
        n_expert_used: Some(1),
        expert_intermediate_dim: Some(1),
        qwen35_full_attention_interval: Some(4),
        qwen35_ssm_conv_kernel: Some(4),
        qwen35_ssm_inner_size: Some(16),
        qwen35_ssm_state_size: Some(4),
        qwen35_ssm_time_step_rank: Some(4),
        qwen35_ssm_group_count: Some(1),
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };

    let mut hidden = vec![2.0f32, 0.0, 0.5, 0.0];
    let hidden_before = hidden.clone();
    let mut norm_buf = vec![0.0f32; n_tokens * dim];
    let mut gate_buf = vec![0.0f32; n_tokens];
    let mut up_buf = vec![0.0f32; n_tokens];
    let mut down_buf = vec![0.0f32; n_tokens * dim];

    {
        let model = MappedModel::open(&path).unwrap();
        let weights = WeightStore::new(&model);

        Qwen3_5Forward::apply_post_attention_ffn_batch(
            &cfg,
            &CpuBackend,
            &weights,
            "blk.0",
            &mut hidden,
            &mut norm_buf,
            &mut gate_buf,
            &mut up_buf,
            &mut down_buf,
            n_tokens,
            dim,
            1,
            cfg.rms_norm_eps,
        )
        .unwrap();
    }

    let mut expected = hidden_before.clone();
    for token_idx in 0..n_tokens {
        let h = &hidden_before[token_idx * dim..(token_idx + 1) * dim];
        let mut norm = vec![0.0f32; dim];
        rms_norm::rms_norm_out(h, &[1.0f32, 1.0], &mut norm, cfg.rms_norm_eps);

        let mut shared_gate = vec![norm[0]];
        let shared_up = vec![2.0 * norm[0]];
        silu::silu_elementwise_mul(&mut shared_gate, &shared_up);
        let shared_down = shared_gate[0];
        let gate_scale = 1.0 / (1.0 + (-norm[0]).exp());

        expected[token_idx * dim] += shared_down * gate_scale;
        expected[token_idx * dim + 1] += shared_down * gate_scale;
    }

    for (actual, expected) in hidden.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_qwen35_shared_expert_gate_width_supported_accepts_scalar_and_vector() {
    assert!(Qwen3_5Forward::qwen35_shared_expert_gate_width_supported(
        1, 4096
    ));
    assert!(Qwen3_5Forward::qwen35_shared_expert_gate_width_supported(
        4096, 4096
    ));
    assert!(!Qwen3_5Forward::qwen35_shared_expert_gate_width_supported(
        2, 4096
    ));
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
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };

    assert!((Qwen3_5Forward::rope_position(&cfg, 16) - 2.0).abs() < 1e-6);
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
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };

    assert!((Qwen3_5Forward::rope_position(&cfg, 16) - 2.0).abs() < 1e-6);
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
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
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
        rope::apply_rope_multi_head_neox_partial_scaled(
            &mut expected_q[q_start..q_start + q_dim],
            &mut expected_k[k_start..k_start + kv_dim],
            n_heads,
            n_kv_heads,
            head_dim,
            head_dim.min(64),
            Qwen3_5Forward::rope_position(&cfg, start_position + token_idx),
            cfg.rope_freq_base,
        );
    }

    Qwen3_5Forward::apply_rope_batch(
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
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };
    let q_gate_batch = vec![
        1.0f32, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.4, 1.5, 2.5, 3.5, 4.5, 0.5, 0.6, 0.7, 0.8,
    ];
    let mut actual_q = vec![0.0f32; 8];
    let mut actual_k = vec![4.0f32, 3.0, 2.0, 1.0, 3.5, 2.5, 1.5, 0.5];
    let mut expected_q = actual_q.clone();
    let mut expected_k = actual_k.clone();
    let norm_weights = Qwen35AttentionNormWeights {
        q: &[1.0f32, 1.1, 1.2, 1.3],
        k: &[0.9f32, 1.0, 1.1, 1.2],
    };

    Qwen3_5Forward::extract_q_from_q_gate_batch(&q_gate_batch, &mut expected_q, 2, 4, 1, 4);
    Qwen3_5Forward::apply_attention_qk_norm_batch(
        &mut expected_q,
        &mut expected_k,
        2,
        4,
        4,
        1,
        1,
        4,
        norm_weights,
        cfg.rms_norm_eps,
    );
    Qwen3_5Forward::apply_rope_batch(&cfg, &mut expected_q, &mut expected_k, 2, 7, 4, 4, 1, 1, 4);

    Qwen3_5Forward::prepare_full_attention_qk_batch(
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
        Some(norm_weights),
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
fn test_qwen35_full_attention_q_gate_interleaved_layout() {
    let n_heads = 2usize;
    let head_dim = 2usize;
    let q_dim = n_heads * head_dim;

    // Per-head interleaved layout: [q_h0, g_h0, q_h1, g_h1].
    let q_gate = vec![1.0f32, 2.0, 0.1, 0.2, 3.0, 4.0, 0.3, 0.4];
    let mut q = vec![0.0f32; q_dim];
    Qwen3_5Forward::extract_q_from_q_gate(&q_gate, &mut q, n_heads, head_dim);
    assert_eq!(q, vec![1.0f32, 2.0, 3.0, 4.0]);

    let mut attn_out = vec![10.0f32, 20.0, 30.0, 40.0];
    Qwen3_5Forward::apply_attention_gate(&q_gate, &mut attn_out, n_heads, head_dim);
    let expected = [
        10.0 / (1.0 + (-0.1f32).exp()),
        20.0 / (1.0 + (-0.2f32).exp()),
        30.0 / (1.0 + (-0.3f32).exp()),
        40.0 / (1.0 + (-0.4f32).exp()),
    ];
    for (actual, expected) in attn_out.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn test_qwen35_full_attention_input_plan_fuses_matching_dtypes() {
    let wq = [1u8, 2, 3, 4];
    let wk = [5u8, 6];
    let wv = [7u8, 8, 9];
    let plan = Qwen3_5Forward::maybe_fused_full_attention_input_plan([
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
        1.0f32, 2.0, 3.0, 4.0, 10.0, 11.0, 20.0, 21.0, 5.0, 6.0, 7.0, 8.0, 12.0, 13.0, 22.0, 23.0,
    ];
    let mut q_gate = vec![0.0f32; 8];
    let mut k = vec![0.0f32; 4];
    let mut v = vec![0.0f32; 4];

    Qwen3_5Forward::split_full_attention_fused_output_batch(&fused, &mut q_gate, &mut k, &mut v, 2);

    assert_eq!(q_gate, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(k, vec![10.0f32, 11.0, 12.0, 13.0]);
    assert_eq!(v, vec![20.0f32, 21.0, 22.0, 23.0]);
}

#[test]
fn test_qwen35_validate_rejects_incompatible_head_expansion() {
    let fwd = Qwen3_5Forward;
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
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
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

    {
        let model = MappedModel::open(&path).unwrap();
        let weights = WeightStore::new(&model);
        let backend = CpuBackend;

        let mut actual = Vec::new();
        Qwen3_5Forward::write_all_batch_logits(
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
            Qwen3_5Forward::write_single_logits(
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
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_qwen35_cpu_batch_fallback_scratch_reuses_capacity() {
    let mut scratch = Qwen3_5CpuBatchFallbackScratch::default();
    scratch.ensure_lengths(8, 64, 128, 64, 32, 2, 192, 96, 6);
    let hidden_capacity = scratch.hidden.capacity();
    let rec_qkv_capacity = scratch.rec_qkv_batch.capacity();
    let final_hidden_capacity = scratch.final_hidden.capacity();

    scratch.ensure_lengths(4, 64, 128, 64, 32, 2, 192, 96, 6);

    assert_eq!(scratch.hidden.len(), 4 * 64);
    assert_eq!(scratch.rec_qkv_batch.len(), 4 * 2 * 192);
    assert_eq!(scratch.final_hidden.len(), 4 * 64);
    assert!(scratch.hidden.capacity() >= hidden_capacity);
    assert!(scratch.rec_qkv_batch.capacity() >= rec_qkv_capacity);
    assert!(scratch.final_hidden.capacity() >= final_hidden_capacity);
}

#[test]
fn test_qwen35_write_all_batch_logits_gpu_q8_0_matches_cpu_reference() {
    let Ok(backend) = MetalBackend::new() else {
        return;
    };

    let dim = 32usize;
    let vocab_size = 2usize;
    let n_tokens = 3usize;
    let rms_norm_eps = 1e-6f32;
    let output_norm = vec![1.0f32; dim];
    let output_weight_f32 = vec![
        1.0, -2.0, 0.0, 3.0, -1.0, 2.0, -3.0, 1.0, 0.0, 1.0, -1.0, 2.0, 3.0, -2.0, 1.0, 0.0, -1.0,
        2.0, 1.0, -3.0, 2.0, 0.0, 1.0, -2.0, 3.0, 1.0, -1.0, 2.0, 0.0, -2.0, 1.0, 3.0, -1.0, 0.0,
        2.0, -2.0, 1.0, 3.0, -1.0, 0.0, 2.0, -3.0, 1.0, 1.0, -2.0, 0.0, 3.0, -1.0, 2.0, -1.0, 0.0,
        1.0, -3.0, 2.0, 1.0, -2.0, 3.0, 0.0, 1.0, -1.0, 2.0, -2.0, 1.0, 0.0,
    ];
    let hidden = vec![
        1.0, -1.0, 2.0, 0.0, 3.0, -2.0, 1.0, 0.0, -1.0, 2.0, 1.0, -3.0, 2.0, 0.0, 1.0, -2.0, 3.0,
        1.0, -1.0, 2.0, 0.0, -2.0, 1.0, 3.0, -1.0, 0.0, 2.0, -3.0, 1.0, 1.0, -2.0, 0.0, -1.0, 0.5,
        2.0, -2.0, 1.0, 3.0, -1.0, 0.0, 2.0, -3.0, 1.0, 1.0, -2.0, 0.0, 3.0, -1.0, 2.0, -1.0, 0.0,
        1.0, -3.0, 2.0, 1.0, -2.0, 3.0, 0.0, 1.0, -1.0, 2.0, -2.0, 1.0, 0.0, 0.25, 1.5, -0.5, 2.0,
        -1.0, 0.0, 1.0, -2.0, 3.0, -1.0, 0.0, 2.0, -3.0, 1.0, 1.0, -2.0, 0.0, 3.0, -1.0, 2.0, -1.0,
        0.0, 1.0, -3.0, 2.0, 1.0, -2.0, 3.0, 0.0, 1.0, -1.0, 2.0, -2.0, 1.0, 0.0,
    ];
    let output_weight_q8_0 = quantize_q8_0_rows(&output_weight_f32, dim);
    let gguf = build_qwen35_logits_test_gguf_with_dtype(
        &output_norm,
        &output_weight_q8_0,
        GgmlType::Q8_0,
        dim,
        vocab_size,
    );
    let path = write_test_gguf_to_temp(&gguf);

    {
        let model = MappedModel::open(&path).unwrap();
        let weights = WeightStore::new(&model);
        let cpu_backend = CpuBackend;

        let mut actual = Vec::new();
        Qwen3_5Forward::write_all_batch_logits(
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

        let mut expected = Vec::new();
        Qwen3_5Forward::write_all_batch_logits(
            &cpu_backend,
            &hidden,
            n_tokens,
            dim,
            vocab_size,
            rms_norm_eps,
            &weights,
            &mut expected,
        )
        .unwrap();

        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 5e-2,
                "q8_0 gpu batch logit {idx} mismatch: actual={actual}, expected={expected}"
            );
        }
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_qwen35_gpu_batch_logits_enabled_env_override() {
    with_env_var("AX_QWEN35_GPU_BATCH_LOGITS", None, || {
        assert!(Qwen3_5Forward::gpu_batch_logits_enabled());
    });
    with_env_var("AX_QWEN35_GPU_BATCH_LOGITS", Some("0"), || {
        assert!(!Qwen3_5Forward::gpu_batch_logits_enabled());
    });
    with_env_var("AX_QWEN35_GPU_BATCH_LOGITS", Some("off"), || {
        assert!(!Qwen3_5Forward::gpu_batch_logits_enabled());
    });
    with_env_var("AX_QWEN35_GPU_BATCH_LOGITS", Some("1"), || {
        assert!(Qwen3_5Forward::gpu_batch_logits_enabled());
    });
}

#[test]
fn test_qwen35_gpu_decode_enabled_env_override() {
    let dense_cfg = ModelConfig {
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
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };
    let moe_cfg = ModelConfig {
        architecture: "qwen35moe".into(),
        n_expert: Some(128),
        n_expert_used: Some(8),
        expert_intermediate_dim: Some(128),
        ..dense_cfg.clone()
    };

    with_env_var("AX_QWEN35_GPU_DECODE", None, || {
        assert!(Qwen3_5Forward::gpu_decode_enabled_for_config(&dense_cfg));
        assert!(Qwen3_5Forward::gpu_decode_enabled_for_config(&moe_cfg));
    });
    with_env_var("AX_QWEN35_GPU_DECODE", Some("0"), || {
        assert!(!Qwen3_5Forward::gpu_decode_enabled_for_config(&dense_cfg));
        assert!(!Qwen3_5Forward::gpu_decode_enabled_for_config(&moe_cfg));
    });
    with_env_var("AX_QWEN35_GPU_DECODE", Some("off"), || {
        assert!(!Qwen3_5Forward::gpu_decode_enabled_for_config(&dense_cfg));
        assert!(!Qwen3_5Forward::gpu_decode_enabled_for_config(&moe_cfg));
    });
    with_env_var("AX_QWEN35_GPU_DECODE", Some("1"), || {
        assert!(Qwen3_5Forward::gpu_decode_enabled_for_config(&dense_cfg));
        assert!(Qwen3_5Forward::gpu_decode_enabled_for_config(&moe_cfg));
    });
}

#[test]
fn test_qwen35_gpu_batch_prefill_enabled_env_override() {
    with_env_var("AX_QWEN35_GPU_BATCH_PREFILL", None, || {
        assert!(Qwen3_5Forward::gpu_batch_prefill_enabled());
    });
    with_env_var("AX_QWEN35_GPU_BATCH_PREFILL", Some("0"), || {
        assert!(!Qwen3_5Forward::gpu_batch_prefill_enabled());
    });
    with_env_var("AX_QWEN35_GPU_BATCH_PREFILL", Some("off"), || {
        assert!(!Qwen3_5Forward::gpu_batch_prefill_enabled());
    });
    with_env_var("AX_QWEN35_GPU_BATCH_PREFILL", Some("1"), || {
        assert!(Qwen3_5Forward::gpu_batch_prefill_enabled());
    });
}

#[test]
fn test_qwen35_native_decode_layer_ranges_dense_use_single_range() {
    let cfg = ModelConfig {
        architecture: "qwen35".into(),
        n_layers: 8,
        n_heads: 8,
        n_kv_heads: 8,
        head_dim: 128,
        embedding_dim: 1024,
        intermediate_dim: 4096,
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
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };

    let ranges = Qwen3_5Forward::qwen35_native_decode_layer_ranges(&cfg, 8);
    assert_eq!(
        ranges,
        vec![Qwen3_5NativeDecodeLayerRange {
            start: 0,
            end_exclusive: 8
        }]
    );
}

#[test]
fn test_qwen35_native_decode_layer_ranges_moe_use_all_layers_by_default() {
    let cfg = ModelConfig {
        architecture: "qwen35moe".into(),
        n_layers: 8,
        n_heads: 8,
        n_kv_heads: 8,
        head_dim: 128,
        embedding_dim: 1024,
        intermediate_dim: 4096,
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
        n_expert: Some(128),
        n_expert_used: Some(8),
        expert_intermediate_dim: Some(128),
        qwen35_full_attention_interval: Some(4),
        qwen35_ssm_conv_kernel: Some(4),
        qwen35_ssm_inner_size: Some(1024),
        qwen35_ssm_state_size: Some(128),
        qwen35_ssm_time_step_rank: Some(8),
        qwen35_ssm_group_count: Some(2),
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };

    with_env_vars(
        &[
            ("AX_QWEN35_GPU_DECODE_LAYER_RANGES", None),
            ("AX_QWEN35_GPU_DECODE_COALESCE_RECURRENT", None),
        ],
        || {
            let ranges = Qwen3_5Forward::qwen35_native_decode_layer_ranges(&cfg, 8);
            assert_eq!(
                ranges,
                vec![Qwen3_5NativeDecodeLayerRange {
                    start: 0,
                    end_exclusive: 8
                }]
            );
        },
    );
    with_env_var("AX_QWEN35_GPU_DECODE_COALESCE_RECURRENT", Some("0"), || {
        let ranges = Qwen3_5Forward::qwen35_native_decode_layer_ranges(&cfg, 8);
        assert_eq!(
            ranges,
            vec![
                Qwen3_5NativeDecodeLayerRange {
                    start: 0,
                    end_exclusive: 1
                },
                Qwen3_5NativeDecodeLayerRange {
                    start: 1,
                    end_exclusive: 2
                },
                Qwen3_5NativeDecodeLayerRange {
                    start: 2,
                    end_exclusive: 3
                },
                Qwen3_5NativeDecodeLayerRange {
                    start: 3,
                    end_exclusive: 4
                },
                Qwen3_5NativeDecodeLayerRange {
                    start: 4,
                    end_exclusive: 5
                },
                Qwen3_5NativeDecodeLayerRange {
                    start: 5,
                    end_exclusive: 6
                },
                Qwen3_5NativeDecodeLayerRange {
                    start: 6,
                    end_exclusive: 7
                },
                Qwen3_5NativeDecodeLayerRange {
                    start: 7,
                    end_exclusive: 8
                },
            ]
        );
    });
    with_env_var("AX_QWEN35_GPU_DECODE_COALESCE_RECURRENT", Some("1"), || {
        let ranges = Qwen3_5Forward::qwen35_native_decode_layer_ranges(&cfg, 8);
        assert_eq!(
            ranges,
            vec![
                Qwen3_5NativeDecodeLayerRange {
                    start: 0,
                    end_exclusive: 3
                },
                Qwen3_5NativeDecodeLayerRange {
                    start: 3,
                    end_exclusive: 4
                },
                Qwen3_5NativeDecodeLayerRange {
                    start: 4,
                    end_exclusive: 7
                },
                Qwen3_5NativeDecodeLayerRange {
                    start: 7,
                    end_exclusive: 8
                },
            ]
        );
    });
    with_env_var(
        "AX_QWEN35_GPU_DECODE_LAYER_RANGES",
        Some("per_layer"),
        || {
            let ranges = Qwen3_5Forward::qwen35_native_decode_layer_ranges(&cfg, 8);
            assert_eq!(
                ranges,
                vec![
                    Qwen3_5NativeDecodeLayerRange {
                        start: 0,
                        end_exclusive: 1
                    },
                    Qwen3_5NativeDecodeLayerRange {
                        start: 1,
                        end_exclusive: 2
                    },
                    Qwen3_5NativeDecodeLayerRange {
                        start: 2,
                        end_exclusive: 3
                    },
                    Qwen3_5NativeDecodeLayerRange {
                        start: 3,
                        end_exclusive: 4
                    },
                    Qwen3_5NativeDecodeLayerRange {
                        start: 4,
                        end_exclusive: 5
                    },
                    Qwen3_5NativeDecodeLayerRange {
                        start: 5,
                        end_exclusive: 6
                    },
                    Qwen3_5NativeDecodeLayerRange {
                        start: 6,
                        end_exclusive: 7
                    },
                    Qwen3_5NativeDecodeLayerRange {
                        start: 7,
                        end_exclusive: 8
                    },
                ]
            );
        },
    );
    with_env_var(
        "AX_QWEN35_GPU_DECODE_LAYER_RANGES",
        Some("recurrent_runs"),
        || {
            let ranges = Qwen3_5Forward::qwen35_native_decode_layer_ranges(&cfg, 8);
            assert_eq!(
                ranges,
                vec![
                    Qwen3_5NativeDecodeLayerRange {
                        start: 0,
                        end_exclusive: 3
                    },
                    Qwen3_5NativeDecodeLayerRange {
                        start: 3,
                        end_exclusive: 4
                    },
                    Qwen3_5NativeDecodeLayerRange {
                        start: 4,
                        end_exclusive: 7
                    },
                    Qwen3_5NativeDecodeLayerRange {
                        start: 7,
                        end_exclusive: 8
                    },
                ]
            );
        },
    );
    with_env_var(
        "AX_QWEN35_GPU_DECODE_LAYER_RANGES",
        Some("all_layers"),
        || {
            let ranges = Qwen3_5Forward::qwen35_native_decode_layer_ranges(&cfg, 8);
            assert_eq!(
                ranges,
                vec![Qwen3_5NativeDecodeLayerRange {
                    start: 0,
                    end_exclusive: 8
                }]
            );
        },
    );
    with_env_vars(
        &[
            ("AX_QWEN35_GPU_DECODE_COALESCE_RECURRENT", Some("1")),
            (
                "AX_QWEN35_GPU_DECODE_COALESCE_RECURRENT_MAX_LAYERS",
                Some("2"),
            ),
        ],
        || {
            let ranges = Qwen3_5Forward::qwen35_native_decode_layer_ranges(&cfg, 8);
            assert_eq!(
                ranges,
                vec![
                    Qwen3_5NativeDecodeLayerRange {
                        start: 0,
                        end_exclusive: 2
                    },
                    Qwen3_5NativeDecodeLayerRange {
                        start: 2,
                        end_exclusive: 3
                    },
                    Qwen3_5NativeDecodeLayerRange {
                        start: 3,
                        end_exclusive: 4
                    },
                    Qwen3_5NativeDecodeLayerRange {
                        start: 4,
                        end_exclusive: 6
                    },
                    Qwen3_5NativeDecodeLayerRange {
                        start: 6,
                        end_exclusive: 7
                    },
                    Qwen3_5NativeDecodeLayerRange {
                        start: 7,
                        end_exclusive: 8
                    },
                ]
            );
        },
    );
}

#[test]
fn test_qwen35_native_decode_dispatch_plan_dense_keeps_base_encoder() {
    let cfg = ModelConfig {
        architecture: "qwen35".into(),
        n_layers: 8,
        n_heads: 8,
        n_kv_heads: 8,
        head_dim: 128,
        embedding_dim: 1024,
        intermediate_dim: 4096,
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
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };
    let base_plan = crate::model::execution_plan::GpuDecodeExecutionPlan {
        encoder: crate::model::execution_plan::DecodeEncoderPlan::Concurrent,
        barriers: crate::model::execution_plan::DecodeBarrierPlan::Smart,
        qkv: crate::model::execution_plan::DecodeQkvPlan::Split,
        kv_f16: false,
        kv_q8: false,
        use_pair_matvec: false,
        use_fused_silu_down: false,
        attention_route: "test",
        attention_tier: "test",
        q4_k_candidate: "test",
        q4_k_tier: "test",
        q5_k_candidate: "test",
        q5_k_tier: "test",
        q6_k_candidate: "test",
        q6_k_tier: "test",
        dequant_dispatch: ax_engine_metal::DequantDispatchConfig::default(),
        attention_dispatch: ax_engine_metal::AttentionDispatchConfig::default(),
    };

    let dispatch_plan = Qwen3_5Forward::qwen35_native_decode_dispatch_plan(&cfg, &base_plan, 8);
    assert_eq!(
        dispatch_plan.encoder,
        crate::model::execution_plan::DecodeEncoderPlan::Concurrent
    );
    assert_eq!(
        dispatch_plan.barriers,
        crate::model::execution_plan::DecodeBarrierPlan::Smart
    );
    assert_eq!(dispatch_plan.layer_ranges.len(), 1);
}

#[test]
fn test_qwen35_native_decode_dispatch_plan_forces_serial_for_coalesced_moe_runs() {
    let cfg = ModelConfig {
        architecture: "qwen35moe".into(),
        n_layers: 8,
        n_heads: 8,
        n_kv_heads: 8,
        head_dim: 128,
        embedding_dim: 1024,
        intermediate_dim: 4096,
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
        n_expert: Some(128),
        n_expert_used: Some(8),
        expert_intermediate_dim: Some(128),
        qwen35_full_attention_interval: Some(4),
        qwen35_ssm_conv_kernel: Some(4),
        qwen35_ssm_inner_size: Some(1024),
        qwen35_ssm_state_size: Some(128),
        qwen35_ssm_time_step_rank: Some(8),
        qwen35_ssm_group_count: Some(2),
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };
    let base_plan = crate::model::execution_plan::GpuDecodeExecutionPlan {
        encoder: crate::model::execution_plan::DecodeEncoderPlan::Concurrent,
        barriers: crate::model::execution_plan::DecodeBarrierPlan::Smart,
        qkv: crate::model::execution_plan::DecodeQkvPlan::Split,
        kv_f16: false,
        kv_q8: false,
        use_pair_matvec: false,
        use_fused_silu_down: false,
        attention_route: "test",
        attention_tier: "test",
        q4_k_candidate: "test",
        q4_k_tier: "test",
        q5_k_candidate: "test",
        q5_k_tier: "test",
        q6_k_candidate: "test",
        q6_k_tier: "test",
        dequant_dispatch: ax_engine_metal::DequantDispatchConfig::default(),
        attention_dispatch: ax_engine_metal::AttentionDispatchConfig::default(),
    };

    with_env_var("AX_QWEN35_GPU_DECODE_COALESCE_RECURRENT", Some("1"), || {
        let dispatch_plan = Qwen3_5Forward::qwen35_native_decode_dispatch_plan(&cfg, &base_plan, 8);
        assert_eq!(
            dispatch_plan.encoder,
            crate::model::execution_plan::DecodeEncoderPlan::Serial
        );
        assert_eq!(
            dispatch_plan.barriers,
            crate::model::execution_plan::DecodeBarrierPlan::Explicit
        );
        assert_eq!(
            dispatch_plan.layer_ranges,
            vec![
                Qwen3_5NativeDecodeLayerRange {
                    start: 0,
                    end_exclusive: 3
                },
                Qwen3_5NativeDecodeLayerRange {
                    start: 3,
                    end_exclusive: 4
                },
                Qwen3_5NativeDecodeLayerRange {
                    start: 4,
                    end_exclusive: 7
                },
                Qwen3_5NativeDecodeLayerRange {
                    start: 7,
                    end_exclusive: 8
                },
            ]
        );
    });
}

#[test]
fn test_qwen35_gpu_batch_prefill_enabled_for_config_defaults_on_for_dense_and_moe() {
    let dense_cfg = ModelConfig {
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
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };
    let moe_cfg = ModelConfig {
        architecture: "qwen35moe".into(),
        n_expert: Some(128),
        n_expert_used: Some(8),
        expert_intermediate_dim: Some(128),
        ..dense_cfg.clone()
    };

    with_env_var("AX_QWEN35_GPU_BATCH_PREFILL", None, || {
        assert!(Qwen3_5Forward::gpu_batch_prefill_enabled_for_config(
            &dense_cfg
        ));
        assert!(Qwen3_5Forward::gpu_batch_prefill_enabled_for_config(
            &moe_cfg
        ));
    });
    with_env_var("AX_QWEN35_GPU_BATCH_PREFILL", Some("1"), || {
        assert!(Qwen3_5Forward::gpu_batch_prefill_enabled_for_config(
            &dense_cfg
        ));
        assert!(Qwen3_5Forward::gpu_batch_prefill_enabled_for_config(
            &moe_cfg
        ));
    });
    with_env_var("AX_QWEN35_GPU_BATCH_PREFILL", Some("0"), || {
        assert!(!Qwen3_5Forward::gpu_batch_prefill_enabled_for_config(
            &dense_cfg
        ));
        assert!(!Qwen3_5Forward::gpu_batch_prefill_enabled_for_config(
            &moe_cfg
        ));
    });
}

#[test]
fn test_qwen35_recurrent_runtime_mode_helper_matches_unified_flag() {
    assert!(!Qwen3_5Forward::qwen35_should_use_unified_recurrent_runtime(false));
    assert!(Qwen3_5Forward::qwen35_should_use_unified_recurrent_runtime(
        true
    ));
}

#[test]
fn test_qwen35_recurrent_tail_mode_helper_uses_legacy_for_moe() {
    assert!(Qwen3_5Forward::qwen35_should_use_unified_recurrent_tail(
        true, false
    ));
    assert!(!Qwen3_5Forward::qwen35_should_use_unified_recurrent_tail(
        true, true
    ));
    assert!(!Qwen3_5Forward::qwen35_should_use_unified_recurrent_tail(
        false, false
    ));
}

#[test]
fn test_qwen35_gpu_pipelined_decode_enabled_env_override() {
    let dense_cfg = ModelConfig {
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
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
    };
    let moe_cfg = ModelConfig {
        architecture: "qwen35moe".into(),
        n_expert: Some(128),
        n_expert_used: Some(8),
        expert_intermediate_dim: Some(128),
        ..dense_cfg.clone()
    };

    with_env_vars(
        &[
            ("AX_QWEN35_GPU_DECODE", None),
            ("AX_QWEN35_GPU_PIPELINED_DECODE", None),
        ],
        || {
            assert!(!Qwen3_5Forward::gpu_pipelined_decode_enabled_for_config(
                &dense_cfg
            ));
            assert!(!Qwen3_5Forward::gpu_pipelined_decode_enabled_for_config(
                &moe_cfg
            ));
        },
    );
    with_env_vars(
        &[
            ("AX_QWEN35_GPU_DECODE", None),
            ("AX_QWEN35_GPU_PIPELINED_DECODE", Some("1")),
        ],
        || {
            assert!(Qwen3_5Forward::gpu_pipelined_decode_enabled_for_config(
                &dense_cfg
            ));
            assert!(!Qwen3_5Forward::gpu_pipelined_decode_enabled_for_config(
                &moe_cfg
            ));
        },
    );
    with_env_vars(
        &[
            ("AX_QWEN35_GPU_DECODE", Some("0")),
            ("AX_QWEN35_GPU_PIPELINED_DECODE", Some("1")),
        ],
        || {
            assert!(!Qwen3_5Forward::gpu_pipelined_decode_enabled_for_config(
                &dense_cfg
            ));
        },
    );
}

#[test]
fn test_qwen35_unified_prefill_enabled_env_override() {
    with_env_var("AX_QWEN35_UNIFIED_PREFILL", None, || {
        assert!(Qwen3_5Forward::unified_prefill_enabled());
    });
    with_env_var("AX_QWEN35_UNIFIED_PREFILL", Some("0"), || {
        assert!(!Qwen3_5Forward::unified_prefill_enabled());
    });
    with_env_var("AX_QWEN35_UNIFIED_PREFILL", Some("off"), || {
        assert!(!Qwen3_5Forward::unified_prefill_enabled());
    });
    with_env_var("AX_QWEN35_UNIFIED_PREFILL", Some("1"), || {
        assert!(Qwen3_5Forward::unified_prefill_enabled());
    });
}

#[test]
fn test_qwen35_unified_recurrent_enabled_env_override() {
    with_env_var("AX_QWEN35_ENABLE_UNIFIED_RECURRENT", None, || {
        assert!(Qwen3_5Forward::unified_recurrent_enabled());
    });
    with_env_var("AX_QWEN35_ENABLE_UNIFIED_RECURRENT", Some("0"), || {
        assert!(!Qwen3_5Forward::unified_recurrent_enabled());
    });
    with_env_var("AX_QWEN35_ENABLE_UNIFIED_RECURRENT", Some("off"), || {
        assert!(!Qwen3_5Forward::unified_recurrent_enabled());
    });
    with_env_var("AX_QWEN35_ENABLE_UNIFIED_RECURRENT", Some("1"), || {
        assert!(Qwen3_5Forward::unified_recurrent_enabled());
    });
}

#[test]
fn test_qwen35_prefill_pipelined_enabled_env_override() {
    with_env_var("AX_QWEN35_PREFILL_PIPELINED", None, || {
        assert!(!crate::model::prefill_schedule::prefill_inter_step_pipelined_enabled());
    });
    with_env_var("AX_QWEN35_PREFILL_PIPELINED", Some("0"), || {
        assert!(!crate::model::prefill_schedule::prefill_inter_step_pipelined_enabled());
    });
    with_env_var("AX_QWEN35_PREFILL_PIPELINED", Some("off"), || {
        assert!(!crate::model::prefill_schedule::prefill_inter_step_pipelined_enabled());
    });
    with_env_var("AX_QWEN35_PREFILL_PIPELINED", Some("1"), || {
        assert!(crate::model::prefill_schedule::prefill_inter_step_pipelined_enabled());
    });
}

#[test]
fn test_real_qwen35_35b_a3b_forward_batch_last_logits_match_cpu_when_gpu_batch_prefill_forced() {
    let _env_lock = crate::test_env_lock();
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&model.header).unwrap();
    let weights = WeightStore::new(&model);
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&model.header).unwrap();
    let prompt_token_ids = tokenizer.encode("The capital of France is", true);
    let vocab_size = cfg.vocab_size as usize;

    with_env_vars(
        &[
            ("AX_QWEN35_GPU_BATCH_PREFILL", Some("1")),
            ("AX_QWEN35_GPU_DECODE", Some("0")),
            ("AX_SERIAL_PREFILL", None),
            ("AX_QWEN35_PREFILL_PIPELINED", Some("0")),
        ],
        || {
            let cpu_model = crate::model::InferenceModel::new(cfg.clone()).unwrap();
            let metal_model = crate::model::InferenceModel::with_backend(
                cfg.clone(),
                Box::new(MetalBackend::new().unwrap()),
            )
            .unwrap();
            let mut cpu_kv = cpu_model.create_model_kv_for_weights(&weights);
            let mut metal_kv = metal_model.create_model_kv_for_weights(&weights);
            let mut cpu_logits = vec![0.0f32; vocab_size];
            let mut metal_logits = vec![0.0f32; vocab_size];

            cpu_model
                .forward_batch(&prompt_token_ids, &mut cpu_kv, &weights, &mut cpu_logits)
                .unwrap();
            metal_model
                .forward_batch(
                    &prompt_token_ids,
                    &mut metal_kv,
                    &weights,
                    &mut metal_logits,
                )
                .unwrap();

            let expected_argmax = argmax_index(&cpu_logits);
            let actual_argmax = argmax_index(&metal_logits);
            let max_diff = max_abs_diff(&cpu_logits, &metal_logits);
            let scale = cpu_logits
                .iter()
                .copied()
                .map(f32::abs)
                .fold(0.0f32, f32::max)
                .max(1.0);
            let rel_diff = max_diff / scale;
            if expected_argmax != actual_argmax || rel_diff > 5e-2 {
                metal_model.sync_model_kv(&mut metal_kv);
                let state_summary = summarize_qwen35_state_diffs(&cfg, &cpu_kv, &mut metal_kv);
                panic!(
                    "real Qwen3.5-35B-A3B forced GPU batch prefill logits mismatch: expected_argmax={} expected_text={:?} actual_argmax={} actual_text={:?} rel_diff={} max_diff={} cpu_seq_len={} gpu_seq_len={} {}",
                    expected_argmax,
                    tokenizer.decode(&[expected_argmax as u32]),
                    actual_argmax,
                    tokenizer.decode(&[actual_argmax as u32]),
                    rel_diff,
                    max_diff,
                    cpu_kv.seq_len(),
                    metal_kv.seq_len(),
                    state_summary,
                );
            }
        },
    );
}

fn assert_real_qwen35_layer0_unified_recurrent_batch_hidden_matches_cpu(
    model_file: &str,
    force_backend_state_batch: bool,
) {
    let _env_lock = crate::test_env_lock();
    let path = workspace_model_path(model_file);
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&model.header).unwrap();
    let weights = WeightStore::new(&model);
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&model.header).unwrap();
    let prompt_token_ids = tokenizer.encode("The capital of France is", true);

    let layer = 0usize;
    assert!(cfg.qwen35_is_recurrent_layer(layer));

    let cpu = CpuBackend;
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();
    let dims = Qwen3_5Forward::recurrent_dims(&cfg).unwrap();
    let dim = cfg.embedding_dim as usize;
    let inter_dim = cfg.intermediate_dim as usize;
    let n_tokens = prompt_token_ids.len();
    let recurrent_slot = 0usize;
    let recurrent_slot_indices = [recurrent_slot];

    let mut expected_hidden = vec![0.0f32; n_tokens * dim];
    Qwen3_5Forward::dequantize_token_embeddings_batch(
        &weights,
        &prompt_token_ids,
        &mut expected_hidden,
        dim,
    )
    .unwrap();
    let input_hidden = expected_hidden.clone();

    let mut cpu_kv = crate::kv::Qwen3_5Kv::new(
        cfg.n_layers as usize,
        cfg.n_kv_heads as usize,
        cfg.head_dim as usize,
        32,
        cfg.qwen35_full_attention_interval.unwrap() as usize,
        cfg.qwen35_ssm_conv_kernel.unwrap() as usize,
        cfg.qwen35_ssm_inner_size.unwrap() as usize,
        cfg.qwen35_ssm_state_size.unwrap() as usize,
        cfg.qwen35_ssm_time_step_rank.unwrap() as usize,
        cfg.qwen35_ssm_group_count.unwrap() as usize,
    );
    let mut norm_buf = vec![0.0f32; n_tokens * dim];
    let mut proj_buf = vec![0.0f32; n_tokens * dim];
    let mut gate_buf = vec![0.0f32; n_tokens * inter_dim];
    let mut up_buf = vec![0.0f32; n_tokens * inter_dim];
    let mut down_buf = vec![0.0f32; n_tokens * dim];
    let mut rec_qkv = vec![0.0f32; n_tokens * dims.conv_dim()];
    let mut rec_z = vec![0.0f32; n_tokens * dims.inner_size];
    let mut rec_beta = vec![0.0f32; n_tokens * dims.time_step_rank];
    let mut rec_alpha = vec![0.0f32; n_tokens * dims.time_step_rank];
    let mut rec_out = vec![0.0f32; n_tokens * dims.inner_size];
    let prefix = format!("blk.{layer}");

    Qwen3_5Forward::apply_attention_norm_batch(
        &weights,
        &prefix,
        &expected_hidden,
        &mut norm_buf,
        n_tokens,
        dim,
        cfg.rms_norm_eps,
    )
    .unwrap();
    Qwen3_5Forward::run_recurrent_batch_layer(
        &cfg,
        &cpu,
        &weights,
        &prefix,
        &mut cpu_kv,
        recurrent_slot,
        layer,
        0,
        dims,
        &recurrent_slot_indices,
        &norm_buf,
        &mut rec_qkv,
        &mut rec_z,
        &mut rec_beta,
        &mut rec_alpha,
        &mut rec_out,
        &mut proj_buf,
        n_tokens,
        dim,
        false,
    )
    .unwrap();
    Qwen3_5Forward::apply_layer_tail_batch(
        &cfg,
        &cpu,
        &weights,
        &prefix,
        &mut expected_hidden,
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
        0,
    )
    .unwrap();

    let mut actual_kv = crate::kv::Qwen3_5Kv::new(
        cfg.n_layers as usize,
        cfg.n_kv_heads as usize,
        cfg.head_dim as usize,
        32,
        cfg.qwen35_full_attention_interval.unwrap() as usize,
        cfg.qwen35_ssm_conv_kernel.unwrap() as usize,
        cfg.qwen35_ssm_inner_size.unwrap() as usize,
        cfg.qwen35_ssm_state_size.unwrap() as usize,
        cfg.qwen35_ssm_time_step_rank.unwrap() as usize,
        cfg.qwen35_ssm_group_count.unwrap() as usize,
    );
    actual_kv
        .enable_gpu_recurrent_state(&metal_ops.device)
        .unwrap();

    metal_ops.init_batch_scratches(&cfg, n_tokens);
    if !metal_ops.has_cached_model_keys() {
        Qwen3_5Forward::build_cached_model_keys_qwen35(metal_ops, &weights, &cfg).unwrap();
    }
    let cached_guard = metal_ops.cached_model_keys();
    let cached = cached_guard.as_ref().unwrap();
    let gpu_layer_keys = Qwen3_5Forward::cached_gpu_layer_keys(cached.lm_head).unwrap();
    let moe_layer_keys = Qwen3_5Forward::cached_moe_layer_keys(cached.lm_head).unwrap();
    let recurrent_keys = match &gpu_layer_keys[layer] {
        Qwen3_5GpuLayerKeys::Recurrent(keys) => keys,
        Qwen3_5GpuLayerKeys::FullAttention => panic!("expected recurrent layer"),
    };

    {
        let mut bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_mut().unwrap();
        unsafe {
            bs.hidden.as_mut_slice::<f32>()[..n_tokens * dim].copy_from_slice(&input_hidden);
        }
    }

    let (reason, tail_frame) = Qwen3_5Forward::run_unified_recurrent_batch_layer(
        metal_ops,
        &metal,
        &cfg,
        &mut actual_kv,
        &weights,
        true,
        true,
        false,
        &cached.layers[layer],
        recurrent_keys,
        moe_layer_keys[layer].as_ref(),
        &prefix,
        layer,
        0,
        recurrent_slot,
        &recurrent_slot_indices,
        1,
        dims,
        n_tokens,
        dim,
        inter_dim,
        n_tokens as u32,
        cfg.rms_norm_eps,
        force_backend_state_batch,
        None,
        false,
    )
    .unwrap();
    assert!(
        reason.is_none(),
        "unexpected unified recurrent fallback: {reason:?}"
    );
    assert!(
        tail_frame.is_none(),
        "layer0 test should not leave tail inflight"
    );

    let actual_hidden = {
        let bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_ref().unwrap();
        unsafe { bs.hidden.as_slice::<f32>()[..n_tokens * dim].to_vec() }
    };
    let (
        actual_proj_buf,
        actual_rec_out,
        actual_rec_qkv,
        actual_rec_z,
        actual_rec_beta,
        actual_rec_alpha,
    ) = {
        let scratch = Qwen3_5Forward::lock_cpu_batch_fallback_scratch();
        (
            scratch.proj_buf[..n_tokens * dim].to_vec(),
            scratch.rec_out_batch[..n_tokens * dims.inner_size].to_vec(),
            scratch.rec_qkv_batch[..n_tokens * dims.conv_dim()].to_vec(),
            scratch.rec_z_batch[..n_tokens * dims.inner_size].to_vec(),
            scratch.rec_beta_batch[..n_tokens * dims.time_step_rank].to_vec(),
            scratch.rec_alpha_batch[..n_tokens * dims.time_step_rank].to_vec(),
        )
    };
    let diff = max_abs_diff(&actual_hidden, &expected_hidden);
    let scale = expected_hidden
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    if diff / scale >= 5e-3 {
        metal.sync_qwen35_kv(&mut actual_kv);
        let conv_diff = max_abs_diff(
            cpu_kv.conv_state_for_slot(0, layer),
            actual_kv.conv_state_for_slot(0, layer),
        );
        let recurrent_diff = max_abs_diff(
            cpu_kv.recurrent_state_for_slot(0, layer),
            actual_kv.recurrent_state_for_slot(0, layer),
        );
        let proj_diff = max_abs_diff(&proj_buf, &actual_proj_buf);
        let rec_out_diff = max_abs_diff(&rec_out, &actual_rec_out);
        let rec_qkv_diff = max_abs_diff(&rec_qkv, &actual_rec_qkv);
        let rec_z_diff = max_abs_diff(&rec_z, &actual_rec_z);
        let rec_beta_diff = max_abs_diff(&rec_beta, &actual_rec_beta);
        let rec_alpha_diff = max_abs_diff(&rec_alpha, &actual_rec_alpha);
        panic!(
            "real {model_file} layer0 unified recurrent batch hidden mismatch: rel_diff={} max_diff={} conv_diff={} recurrent_diff={} proj_diff={} rec_out_diff={} rec_qkv_diff={} rec_z_diff={} rec_beta_diff={} rec_alpha_diff={} actual_hidden[0..8]={:?} expected_hidden[0..8]={:?} actual_proj[0..8]={:?} expected_proj[0..8]={:?}",
            diff / scale,
            diff,
            conv_diff,
            recurrent_diff,
            proj_diff,
            rec_out_diff,
            rec_qkv_diff,
            rec_z_diff,
            rec_beta_diff,
            rec_alpha_diff,
            &actual_hidden[..8],
            &expected_hidden[..8],
            &actual_proj_buf[..8],
            &proj_buf[..8],
        );
    }
}

#[test]
fn test_real_qwen35_4b_q8_0_forward_batch_last_logits_match_cpu_on_default_gpu_prefill() {
    let _env_lock = crate::test_env_lock();
    let path = workspace_model_path("Qwen3.5-4B-Q8_0.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&model.header).unwrap();
    let weights = WeightStore::new(&model);
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&model.header).unwrap();
    let prompt_token_ids = tokenizer.encode("The capital of France is", true);
    let vocab_size = cfg.vocab_size as usize;

    with_env_vars(
        &[
            ("AX_QWEN35_GPU_BATCH_PREFILL", None),
            ("AX_QWEN35_UNIFIED_PREFILL", None),
            ("AX_QWEN35_ENABLE_UNIFIED_RECURRENT", None),
            ("AX_QWEN35_MERGED_FUSED_RECURRENT", None),
            ("AX_QWEN35_GPU_QKV_FAST_PATH", None),
            ("AX_SERIAL_PREFILL", None),
            ("AX_QWEN35_PREFILL_PIPELINED", Some("0")),
        ],
        || {
            let cpu_model = crate::model::InferenceModel::new(cfg.clone()).unwrap();
            let metal_model = crate::model::InferenceModel::with_backend(
                cfg.clone(),
                Box::new(MetalBackend::new().unwrap()),
            )
            .unwrap();
            let mut cpu_kv = cpu_model.create_model_kv_for_weights(&weights);
            let mut metal_kv = metal_model.create_model_kv_for_weights(&weights);
            let mut cpu_logits = vec![0.0f32; vocab_size];
            let mut metal_logits = vec![0.0f32; vocab_size];

            cpu_model
                .forward_batch(&prompt_token_ids, &mut cpu_kv, &weights, &mut cpu_logits)
                .unwrap();
            metal_model
                .forward_batch(
                    &prompt_token_ids,
                    &mut metal_kv,
                    &weights,
                    &mut metal_logits,
                )
                .unwrap();

            let expected_argmax = argmax_index(&cpu_logits);
            let actual_argmax = argmax_index(&metal_logits);
            let max_diff = max_abs_diff(&cpu_logits, &metal_logits);
            let scale = cpu_logits
                .iter()
                .copied()
                .map(f32::abs)
                .fold(0.0f32, f32::max)
                .max(1.0);
            let rel_diff = max_diff / scale;
            if expected_argmax != actual_argmax || rel_diff > 5e-2 {
                metal_model.sync_model_kv(&mut metal_kv);
                let state_summary = summarize_qwen35_state_diffs(&cfg, &cpu_kv, &mut metal_kv);
                panic!(
                    "real Qwen3.5-4B-Q8_0 default GPU batch prefill logits mismatch: expected_argmax={} expected_text={:?} actual_argmax={} actual_text={:?} rel_diff={} max_diff={} cpu_seq_len={} gpu_seq_len={} {}",
                    expected_argmax,
                    tokenizer.decode(&[expected_argmax as u32]),
                    actual_argmax,
                    tokenizer.decode(&[actual_argmax as u32]),
                    rel_diff,
                    max_diff,
                    cpu_kv.seq_len(),
                    metal_kv.seq_len(),
                    state_summary,
                );
            }
        },
    );
}

#[test]
fn test_real_qwen35_35b_a3b_layer0_recurrent_batch_hidden_matches_cpu_with_backend_state_batch() {
    assert_real_qwen35_layer0_unified_recurrent_batch_hidden_matches_cpu(
        "Qwen3.5-35B-A3B-Q4_K_M.gguf",
        true,
    );
}

#[test]
#[ignore = "diagnostic: dense layer0 unified recurrent hidden parity still diverges from CPU"]
fn test_real_qwen35_4b_q8_0_layer0_recurrent_batch_hidden_matches_cpu_with_default_state_path() {
    assert_real_qwen35_layer0_unified_recurrent_batch_hidden_matches_cpu(
        "Qwen3.5-4B-Q8_0.gguf",
        false,
    );
}

#[test]
fn test_real_qwen35_35b_a3b_layer3_full_attention_hidden_before_moe_batch_matches_cpu_for_prompt_state()
 {
    let _env_lock = crate::test_env_lock();
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&model.header).unwrap();
    let weights = WeightStore::new(&model);
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();

    let layer_idx = 3usize;
    assert!(!cfg.qwen35_is_recurrent_layer(layer_idx));

    let dim = cfg.embedding_dim as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&model.header).unwrap();
    let prompt_token_ids = tokenizer.encode("The capital of France is", true);
    let n_tokens = prompt_token_ids.len();

    let (input_hidden_batch, expected_hidden_before_moe_batch, _, expected_kv) =
        build_real_qwen35_35b_a3b_layer3_prompt_state(&cfg, &weights, &prompt_token_ids);

    let mut actual_kv = crate::kv::Qwen3_5Kv::new(
        cfg.n_layers as usize,
        cfg.n_kv_heads as usize,
        cfg.head_dim as usize,
        32,
        cfg.qwen35_full_attention_interval.unwrap() as usize,
        cfg.qwen35_ssm_conv_kernel.unwrap() as usize,
        cfg.qwen35_ssm_inner_size.unwrap() as usize,
        cfg.qwen35_ssm_state_size.unwrap() as usize,
        cfg.qwen35_ssm_time_step_rank.unwrap() as usize,
        cfg.qwen35_ssm_group_count.unwrap() as usize,
    );

    metal_ops.init_batch_scratches(&cfg, n_tokens);
    if !metal_ops.has_cached_model_keys() {
        Qwen3_5Forward::build_cached_model_keys_qwen35(metal_ops, &weights, &cfg).unwrap();
    }
    let cached_guard = metal_ops.cached_model_keys();
    let cached = cached_guard.as_ref().unwrap();

    {
        let mut bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_mut().unwrap();
        unsafe {
            bs.hidden.as_mut_slice::<f32>()[..n_tokens * dim].copy_from_slice(&input_hidden_batch);
        }
    }

    let used = Qwen3_5Forward::build_qwen35_full_attention_hidden_before_moe_gpu(
        metal_ops,
        &cfg,
        &mut actual_kv,
        &cached.layers[layer_idx],
        layer_idx,
        0,
        n_tokens,
        dim,
        q_dim,
        kv_dim,
        n_heads,
        n_kv_heads,
        head_dim,
        n_tokens as u32,
        cfg.rms_norm_eps,
        false,
    )
    .unwrap();
    assert!(
        used,
        "expected layer3 full-attention hidden-before-MoE GPU path"
    );

    let actual_hidden_before_moe = {
        let bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_ref().unwrap();
        unsafe { bs.norm_buf.as_slice::<f32>()[..n_tokens * dim].to_vec() }
    };
    let hidden_diff = max_abs_diff(&actual_hidden_before_moe, &expected_hidden_before_moe_batch);
    let hidden_scale = expected_hidden_before_moe_batch
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        hidden_diff / hidden_scale < 1e-2,
        "real Qwen3.5-35B-A3B layer-3 full-attention hidden-before-MoE mismatch: rel_diff={}, max_diff={hidden_diff}, actual[0..8]={:?}, expected[0..8]={:?}",
        hidden_diff / hidden_scale,
        &actual_hidden_before_moe[..8],
        &expected_hidden_before_moe_batch[..8],
    );

    let _ = expected_kv;
    let _ = actual_kv;
}

#[test]
fn test_real_qwen35_35b_a3b_layer3_resident_moe_tail_from_staged_hidden_matches_cpu_for_prompt_state()
 {
    let _env_lock = crate::test_env_lock();
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&model.header).unwrap();
    let weights = WeightStore::new(&model);
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();

    let layer_idx = 3usize;
    let dim = cfg.embedding_dim as usize;
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&model.header).unwrap();
    let prompt_token_ids = tokenizer.encode("The capital of France is", true);
    let n_tokens = prompt_token_ids.len();

    let (_, hidden_before_moe_batch, expected_hidden_batch, _) =
        build_real_qwen35_35b_a3b_layer3_prompt_state(&cfg, &weights, &prompt_token_ids);

    metal_ops.init_batch_scratches(&cfg, n_tokens);
    if !metal_ops.has_cached_model_keys() {
        Qwen3_5Forward::build_cached_model_keys_qwen35(metal_ops, &weights, &cfg).unwrap();
    }
    let cached_guard = metal_ops.cached_model_keys();
    let cached = cached_guard.as_ref().unwrap();
    let moe_layer_keys = Qwen3_5Forward::cached_moe_layer_keys(cached.lm_head).unwrap();
    let moe_layer = moe_layer_keys[layer_idx]
        .as_ref()
        .expect("expected MoE layer keys for layer3");

    {
        let mut bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_mut().unwrap();
        unsafe {
            bs.norm_buf.as_mut_slice::<f32>()[..n_tokens * dim]
                .copy_from_slice(&hidden_before_moe_batch);
        }
    }

    assert!(
        Qwen3_5Forward::run_qwen35_projected_moe_resident_tail_from_staged_hidden(
            metal_ops,
            cached.layers[layer_idx].ffn_norm,
            moe_layer,
            n_tokens,
            dim,
            cfg.rms_norm_eps,
        )
        .unwrap()
    );

    let actual_hidden_batch = {
        let bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_ref().unwrap();
        unsafe { bs.hidden.as_slice::<f32>()[..n_tokens * dim].to_vec() }
    };
    let hidden_diff = max_abs_diff(&actual_hidden_batch, &expected_hidden_batch);
    let hidden_scale = expected_hidden_batch
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        hidden_diff / hidden_scale < 5e-3,
        "real Qwen3.5-35B-A3B layer-3 resident MoE tail from staged hidden mismatch: rel_diff={}, max_diff={hidden_diff}, actual[0..8]={:?}, expected[0..8]={:?}",
        hidden_diff / hidden_scale,
        &actual_hidden_batch[..8],
        &expected_hidden_batch[..8],
    );
}

#[test]
fn test_real_qwen35_35b_a3b_layer3_unified_full_attention_batch_matches_cpu_for_prompt_state() {
    let _env_lock = crate::test_env_lock();
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&model.header).unwrap();
    let weights = WeightStore::new(&model);
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();

    let layer_idx = 3usize;
    assert!(!cfg.qwen35_is_recurrent_layer(layer_idx));

    let dim = cfg.embedding_dim as usize;
    let inter_dim = cfg.expert_intermediate_dim.unwrap() as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&model.header).unwrap();
    let prompt_token_ids = tokenizer.encode("The capital of France is", true);
    let n_tokens = prompt_token_ids.len();

    let make_kv = || {
        crate::kv::Qwen3_5Kv::new(
            cfg.n_layers as usize,
            cfg.n_kv_heads as usize,
            cfg.head_dim as usize,
            32,
            cfg.qwen35_full_attention_interval.unwrap() as usize,
            cfg.qwen35_ssm_conv_kernel.unwrap() as usize,
            cfg.qwen35_ssm_inner_size.unwrap() as usize,
            cfg.qwen35_ssm_state_size.unwrap() as usize,
            cfg.qwen35_ssm_time_step_rank.unwrap() as usize,
            cfg.qwen35_ssm_group_count.unwrap() as usize,
        )
    };

    let (input_hidden_batch, _, expected_hidden_batch, expected_kv) =
        build_real_qwen35_35b_a3b_layer3_prompt_state(&cfg, &weights, &prompt_token_ids);

    let mut actual_kv = make_kv();
    metal_ops.init_batch_scratches(&cfg, n_tokens);
    if !metal_ops.has_cached_model_keys() {
        Qwen3_5Forward::build_cached_model_keys_qwen35(metal_ops, &weights, &cfg).unwrap();
    }
    let cached_guard = metal_ops.cached_model_keys();
    let cached = cached_guard.as_ref().unwrap();
    let moe_layer_keys = Qwen3_5Forward::cached_moe_layer_keys(cached.lm_head).unwrap();

    {
        let mut bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_mut().unwrap();
        unsafe {
            bs.hidden.as_mut_slice::<f32>()[..n_tokens * dim].copy_from_slice(&input_hidden_batch);
        }
    }

    let used = Qwen3_5Forward::run_unified_full_attention_batch_layer(
        metal_ops,
        &cfg,
        &mut actual_kv,
        &weights,
        false,
        &cached.layers[layer_idx],
        moe_layer_keys[layer_idx].as_ref(),
        &format!("blk.{layer_idx}"),
        layer_idx,
        0,
        n_tokens,
        dim,
        q_dim,
        kv_dim,
        inter_dim,
        n_heads,
        n_kv_heads,
        head_dim,
        n_tokens as u32,
        cfg.rms_norm_eps,
        None,
        false,
    )
    .unwrap();
    assert!(
        used,
        "expected unified full-attention MoE batch path to run"
    );

    let actual_hidden_batch = {
        let bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_ref().unwrap();
        unsafe { bs.hidden.as_slice::<f32>()[..n_tokens * dim].to_vec() }
    };
    let hidden_diff = max_abs_diff(&actual_hidden_batch, &expected_hidden_batch);
    let hidden_scale = expected_hidden_batch
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        hidden_diff / hidden_scale < 2e-2,
        "real Qwen3.5-35B-A3B layer-3 unified full-attention batch hidden mismatch: rel_diff={}, max_diff={hidden_diff}, actual[0..8]={:?}, expected[0..8]={:?}",
        hidden_diff / hidden_scale,
        &actual_hidden_batch[..8],
        &expected_hidden_batch[..8],
    );

    let _ = expected_kv;
    let _ = actual_kv;
}

#[test]
fn test_real_qwen35_35b_a3b_layer0_projected_hidden_before_moe_gpu_stage_matches_cpu() {
    let _env_lock = crate::test_env_lock();
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&model.header).unwrap();
    let weights = WeightStore::new(&model);
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&model.header).unwrap();
    let prompt_token_ids = tokenizer.encode("The capital of France is", true);

    let layer = 0usize;
    assert!(cfg.qwen35_is_recurrent_layer(layer));

    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();
    let dims = Qwen3_5Forward::recurrent_dims(&cfg).unwrap();
    let dim = cfg.embedding_dim as usize;
    let inter_dim = cfg.intermediate_dim as usize;
    let n_tokens = prompt_token_ids.len();
    let recurrent_slot = 0usize;
    let recurrent_slot_indices = [recurrent_slot];

    let mut input_hidden = vec![0.0f32; n_tokens * dim];
    Qwen3_5Forward::dequantize_token_embeddings_batch(
        &weights,
        &prompt_token_ids,
        &mut input_hidden,
        dim,
    )
    .unwrap();
    let prefix = format!("blk.{layer}");

    let mut actual_kv = crate::kv::Qwen3_5Kv::new(
        cfg.n_layers as usize,
        cfg.n_kv_heads as usize,
        cfg.head_dim as usize,
        32,
        cfg.qwen35_full_attention_interval.unwrap() as usize,
        cfg.qwen35_ssm_conv_kernel.unwrap() as usize,
        cfg.qwen35_ssm_inner_size.unwrap() as usize,
        cfg.qwen35_ssm_state_size.unwrap() as usize,
        cfg.qwen35_ssm_time_step_rank.unwrap() as usize,
        cfg.qwen35_ssm_group_count.unwrap() as usize,
    );
    actual_kv
        .enable_gpu_recurrent_state(&metal_ops.device)
        .unwrap();

    metal_ops.init_batch_scratches(&cfg, n_tokens);
    if !metal_ops.has_cached_model_keys() {
        Qwen3_5Forward::build_cached_model_keys_qwen35(metal_ops, &weights, &cfg).unwrap();
    }
    let cached_guard = metal_ops.cached_model_keys();
    let cached = cached_guard.as_ref().unwrap();
    let gpu_layer_keys = Qwen3_5Forward::cached_gpu_layer_keys(cached.lm_head).unwrap();
    let moe_layer_keys = Qwen3_5Forward::cached_moe_layer_keys(cached.lm_head).unwrap();
    let recurrent_keys = match &gpu_layer_keys[layer] {
        Qwen3_5GpuLayerKeys::Recurrent(keys) => keys,
        Qwen3_5GpuLayerKeys::FullAttention => panic!("expected recurrent layer"),
    };

    {
        let mut bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_mut().unwrap();
        unsafe {
            bs.hidden.as_mut_slice::<f32>()[..n_tokens * dim].copy_from_slice(&input_hidden);
        }
    }

    let (reason, tail_frame) = Qwen3_5Forward::run_unified_recurrent_batch_layer(
        metal_ops,
        &metal,
        &cfg,
        &mut actual_kv,
        &weights,
        true,
        true,
        false,
        &cached.layers[layer],
        recurrent_keys,
        moe_layer_keys[layer].as_ref(),
        &prefix,
        layer,
        0,
        recurrent_slot,
        &recurrent_slot_indices,
        1,
        dims,
        n_tokens,
        dim,
        inter_dim,
        n_tokens as u32,
        cfg.rms_norm_eps,
        true,
        None,
        false,
    )
    .unwrap();
    assert!(reason.is_none());
    assert!(tail_frame.is_none());

    let actual_proj_buf = {
        let scratch = Qwen3_5Forward::lock_cpu_batch_fallback_scratch();
        scratch.proj_buf[..n_tokens * dim].to_vec()
    };
    let mut expected_hidden_before_moe = input_hidden.clone();
    silu::elementwise_add(&mut expected_hidden_before_moe, &actual_proj_buf);

    {
        let mut bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_mut().unwrap();
        unsafe {
            bs.hidden.as_mut_slice::<f32>()[..n_tokens * dim].copy_from_slice(&input_hidden);
        }
    }
    assert!(
        Qwen3_5Forward::build_qwen35_projected_hidden_before_moe_gpu(
            metal_ops,
            n_tokens,
            dim,
            &actual_proj_buf,
        )
        .unwrap()
    );
    let actual_hidden_before_moe = {
        let bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_ref().unwrap();
        unsafe { bs.norm_buf.as_slice::<f32>()[..n_tokens * dim].to_vec() }
    };
    let diff = max_abs_diff(&actual_hidden_before_moe, &expected_hidden_before_moe);
    let scale = expected_hidden_before_moe
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 5e-3,
        "real Qwen3.5-35B-A3B layer0 projected hidden-before-moe GPU stage mismatch: rel_diff={}, max_diff={diff}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual_hidden_before_moe[..8],
        &expected_hidden_before_moe[..8],
    );
}

#[test]
#[ignore = "diagnostic helper: projected resident MoE tail remains experimental for multitoken recurrent prefill"]
fn test_real_qwen35_35b_a3b_layer0_projected_resident_moe_tail_from_staged_hidden_matches_cpu() {
    let _env_lock = crate::test_env_lock();
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&model.header).unwrap();
    let weights = WeightStore::new(&model);
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&model.header).unwrap();
    let prompt_token_ids = tokenizer.encode("The capital of France is", true);

    let layer = 0usize;
    let dim = cfg.embedding_dim as usize;
    let inter_dim = cfg.intermediate_dim as usize;
    let n_tokens = prompt_token_ids.len();
    let recurrent_slot = 0usize;
    let recurrent_slot_indices = [recurrent_slot];
    let dims = Qwen3_5Forward::recurrent_dims(&cfg).unwrap();

    let cpu = CpuBackend;
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();

    let mut input_hidden = vec![0.0f32; n_tokens * dim];
    Qwen3_5Forward::dequantize_token_embeddings_batch(
        &weights,
        &prompt_token_ids,
        &mut input_hidden,
        dim,
    )
    .unwrap();
    let mut expected_hidden = input_hidden.clone();

    let mut cpu_kv = crate::kv::Qwen3_5Kv::new(
        cfg.n_layers as usize,
        cfg.n_kv_heads as usize,
        cfg.head_dim as usize,
        32,
        cfg.qwen35_full_attention_interval.unwrap() as usize,
        cfg.qwen35_ssm_conv_kernel.unwrap() as usize,
        cfg.qwen35_ssm_inner_size.unwrap() as usize,
        cfg.qwen35_ssm_state_size.unwrap() as usize,
        cfg.qwen35_ssm_time_step_rank.unwrap() as usize,
        cfg.qwen35_ssm_group_count.unwrap() as usize,
    );
    let mut norm_buf = vec![0.0f32; n_tokens * dim];
    let mut proj_buf = vec![0.0f32; n_tokens * dim];
    let mut gate_buf = vec![0.0f32; n_tokens * inter_dim];
    let mut up_buf = vec![0.0f32; n_tokens * inter_dim];
    let mut down_buf = vec![0.0f32; n_tokens * dim];
    let mut rec_qkv = vec![0.0f32; n_tokens * dims.conv_dim()];
    let mut rec_z = vec![0.0f32; n_tokens * dims.inner_size];
    let mut rec_beta = vec![0.0f32; n_tokens * dims.time_step_rank];
    let mut rec_alpha = vec![0.0f32; n_tokens * dims.time_step_rank];
    let mut rec_out = vec![0.0f32; n_tokens * dims.inner_size];
    let prefix = format!("blk.{layer}");

    Qwen3_5Forward::apply_attention_norm_batch(
        &weights,
        &prefix,
        &expected_hidden,
        &mut norm_buf,
        n_tokens,
        dim,
        cfg.rms_norm_eps,
    )
    .unwrap();
    Qwen3_5Forward::run_recurrent_batch_layer(
        &cfg,
        &cpu,
        &weights,
        &prefix,
        &mut cpu_kv,
        recurrent_slot,
        layer,
        0,
        dims,
        &recurrent_slot_indices,
        &norm_buf,
        &mut rec_qkv,
        &mut rec_z,
        &mut rec_beta,
        &mut rec_alpha,
        &mut rec_out,
        &mut proj_buf,
        n_tokens,
        dim,
        false,
    )
    .unwrap();
    Qwen3_5Forward::apply_layer_tail_batch(
        &cfg,
        &cpu,
        &weights,
        &prefix,
        &mut expected_hidden,
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
        0,
    )
    .unwrap();

    metal_ops.init_batch_scratches(&cfg, n_tokens);
    if !metal_ops.has_cached_model_keys() {
        Qwen3_5Forward::build_cached_model_keys_qwen35(metal_ops, &weights, &cfg).unwrap();
    }
    let cached_guard = metal_ops.cached_model_keys();
    let cached = cached_guard.as_ref().unwrap();
    let moe_layer_keys = Qwen3_5Forward::cached_moe_layer_keys(cached.lm_head).unwrap();
    let moe_layer = moe_layer_keys[layer]
        .as_ref()
        .expect("expected MoE layer keys for layer0");

    {
        let mut bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_mut().unwrap();
        unsafe {
            bs.hidden.as_mut_slice::<f32>()[..n_tokens * dim].copy_from_slice(&input_hidden);
        }
    }
    assert!(
        Qwen3_5Forward::build_qwen35_projected_hidden_before_moe_gpu(
            metal_ops, n_tokens, dim, &proj_buf
        )
        .unwrap()
    );
    assert!(
        Qwen3_5Forward::run_qwen35_projected_moe_resident_tail_from_staged_hidden(
            metal_ops,
            cached.layers[layer].ffn_norm,
            moe_layer,
            n_tokens,
            dim,
            cfg.rms_norm_eps,
        )
        .unwrap()
    );

    let actual_hidden = {
        let bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_ref().unwrap();
        unsafe { bs.hidden.as_slice::<f32>()[..n_tokens * dim].to_vec() }
    };
    let diff = max_abs_diff(&actual_hidden, &expected_hidden);
    let scale = expected_hidden
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 5e-3,
        "real Qwen3.5-35B-A3B layer0 projected resident MoE tail mismatch: rel_diff={}, max_diff={diff}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual_hidden[..8],
        &expected_hidden[..8],
    );
}

#[test]
fn test_real_qwen35_35b_a3b_layer4_moe_resident_matches_cpu_for_multitoken_state() {
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&model.header).unwrap();
    let weights = WeightStore::new(&model);
    let cpu = CpuBackend;
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();

    let layer_idx = 4usize;
    assert!(cfg.qwen35_is_recurrent_layer(layer_idx));

    let dim = cfg.embedding_dim as usize;
    let inter_dim = cfg.expert_intermediate_dim.unwrap() as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let dims = Qwen3_5Forward::recurrent_dims(&cfg).unwrap();
    let recurrent_slot = 0usize;
    let recurrent_slot_indices = [recurrent_slot];
    let full_attn_params = AttentionParams::new(n_heads, n_kv_heads, head_dim);
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&model.header).unwrap();
    let prompt_token_ids = tokenizer.encode("The capital of France is", true);
    let final_position = prompt_token_ids.len() - 1;

    let mut qwen_kv = crate::kv::Qwen3_5Kv::new(
        cfg.n_layers as usize,
        cfg.n_kv_heads as usize,
        cfg.head_dim as usize,
        32,
        cfg.qwen35_full_attention_interval.unwrap() as usize,
        cfg.qwen35_ssm_conv_kernel.unwrap() as usize,
        cfg.qwen35_ssm_inner_size.unwrap() as usize,
        cfg.qwen35_ssm_state_size.unwrap() as usize,
        cfg.qwen35_ssm_time_step_rank.unwrap() as usize,
        cfg.qwen35_ssm_group_count.unwrap() as usize,
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
    let mut hidden_before_moe = None;

    for (position, &token_id) in prompt_token_ids.iter().enumerate() {
        let mut hidden = vec![0.0f32; dim];
        weights
            .dequantize_row("token_embd.weight", token_id as usize, &mut hidden)
            .unwrap();

        for layer in 0..=layer_idx {
            let prefix = format!("blk.{layer}");
            apply_attention_norm_single(
                &weights,
                &prefix,
                &hidden,
                &mut norm_buf,
                cfg.rms_norm_eps,
                None,
            )
            .unwrap();

            if cfg.qwen35_is_recurrent_layer(layer) {
                Qwen3_5Forward::run_recurrent_single_layer(
                    &cfg,
                    &cpu,
                    &weights,
                    &prefix,
                    &mut qwen_kv,
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
                    None,
                )
                .unwrap();
            } else {
                Qwen3_5Forward::run_full_attention_single_layer(
                    &cfg,
                    &cpu,
                    &weights,
                    &prefix,
                    &mut qwen_kv,
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
                    None,
                )
                .unwrap();
            }

            if position == final_position && layer == layer_idx {
                let mut captured = hidden.clone();
                silu::elementwise_add(&mut captured, &proj_buf);
                hidden_before_moe = Some(captured);
                break;
            }

            Qwen3_5Forward::apply_layer_tail_single(
                &cfg,
                &cpu,
                &weights,
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
                None,
            )
            .unwrap();
        }

        if position != final_position {
            qwen_kv.finalize_token();
        }
    }

    let hidden_before_moe = hidden_before_moe.expect("failed to capture layer-4 hidden before MoE");
    let mut expected_hidden = hidden_before_moe.clone();
    let mut expected_norm_buf = vec![0.0f32; dim];
    let prefix = format!("blk.{layer_idx}");
    Qwen3_5Forward::apply_post_attention_moe_single(
        &cfg,
        &cpu,
        &weights,
        &prefix,
        &mut expected_hidden,
        &mut expected_norm_buf,
        dim,
        inter_dim,
        cfg.rms_norm_eps,
    )
    .unwrap();

    let router_name = format!("{prefix}.ffn_gate_inp.weight");
    let gate_name = format!("{prefix}.ffn_gate_exps.weight");
    let up_name = format!("{prefix}.ffn_up_exps.weight");
    let down_name = format!("{prefix}.ffn_down_exps.weight");
    let shared_gate_name = format!("{prefix}.ffn_gate_shexp.weight");
    let shared_up_name = format!("{prefix}.ffn_up_shexp.weight");
    let shared_down_name = format!("{prefix}.ffn_down_shexp.weight");
    let shared_gate_inp_name = format!("{prefix}.ffn_gate_inp_shexp.weight");

    let expert_inter_dim = Qwen3_5Forward::tensor_output_rows(&weights, &gate_name).unwrap();
    let n_expert = cfg.n_expert.unwrap() as usize;
    let n_expert_used = cfg.n_expert_used.unwrap() as usize;
    let (router_raw, router_dtype) = weights.raw_with_dtype(&router_name).unwrap();
    let (gate_raw, gate_dtype) = weights.raw_with_dtype(&gate_name).unwrap();
    let (up_raw, up_dtype) = weights.raw_with_dtype(&up_name).unwrap();
    let (down_raw, down_dtype) = weights.raw_with_dtype(&down_name).unwrap();
    let gate_stride =
        crate::model::moe_utils::expert_byte_stride(gate_dtype, expert_inter_dim * dim);
    let up_stride = crate::model::moe_utils::expert_byte_stride(up_dtype, expert_inter_dim * dim);
    let down_stride =
        crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);

    let hidden_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), &hidden_before_moe)
            .unwrap();
    let ffn_norm_w = weights
        .f32_slice(&format!("{prefix}.post_attention_norm.weight"))
        .unwrap();
    let ffn_norm_w_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), ffn_norm_w).unwrap();
    let router_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), router_raw).unwrap();
    let gate_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), gate_raw).unwrap();
    let up_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), up_raw).unwrap();
    let down_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), down_raw).unwrap();

    let shared_gate_buf = weights.has(&shared_gate_name).then(|| {
        let (raw, _) = weights.raw_with_dtype(&shared_gate_name).unwrap();
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), raw).unwrap()
    });
    let shared_up_buf = weights.has(&shared_up_name).then(|| {
        let (raw, _) = weights.raw_with_dtype(&shared_up_name).unwrap();
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), raw).unwrap()
    });
    let shared_down_buf = weights.has(&shared_down_name).then(|| {
        let (raw, _) = weights.raw_with_dtype(&shared_down_name).unwrap();
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), raw).unwrap()
    });
    let shared_gate_inp_buf = weights.has(&shared_gate_inp_name).then(|| {
        let (raw, _) = weights.raw_with_dtype(&shared_gate_inp_name).unwrap();
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), raw).unwrap()
    });
    let shared_expert = if let (Some(gate), Some(up), Some(down)) = (
        shared_gate_buf.as_ref(),
        shared_up_buf.as_ref(),
        shared_down_buf.as_ref(),
    ) {
        let (_, shared_dtype) = weights.raw_with_dtype(&shared_gate_name).unwrap();
        let gate_inp_dtype = if weights.has(&shared_gate_inp_name) {
            Some(weights.raw_with_dtype(&shared_gate_inp_name).unwrap().1)
        } else {
            None
        };
        Some(crate::backend::metal::SharedExpertCachedBuffers {
            gate,
            up,
            down,
            gate_inp: shared_gate_inp_buf.as_ref(),
            gate_inp_dtype,
            dtype: shared_dtype,
            inter_dim: Qwen3_5Forward::tensor_output_rows(&weights, &shared_gate_name).unwrap(),
            gate_inp_rows: if weights.has(&shared_gate_inp_name) {
                Qwen3_5Forward::tensor_output_rows(&weights, &shared_gate_inp_name).unwrap()
            } else {
                0
            },
        })
    } else {
        None
    };

    metal_ops.init_batch_scratches(&cfg, 1);
    metal_ops
        .moe_ffn_gpu_resident_cached(
            &hidden_buf,
            &ffn_norm_w_buf,
            &router_buf,
            router_dtype,
            &gate_buf,
            gate_dtype,
            &up_buf,
            up_dtype,
            &down_buf,
            down_dtype,
            1,
            n_expert,
            n_expert_used,
            dim,
            expert_inter_dim,
            gate_stride,
            up_stride,
            down_stride,
            cfg.rms_norm_eps,
            shared_expert.as_ref(),
        )
        .unwrap();

    let actual_hidden = unsafe { hidden_buf.as_slice::<f32>()[..dim].to_vec() };
    let diff = max_abs_diff(&actual_hidden, &expected_hidden);
    let scale = expected_hidden
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 5e-3,
        "real Qwen3.5-35B-A3B layer-4 resident MoE mismatch on multitoken state: rel_diff={}, max_diff={diff}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual_hidden[..8],
        &expected_hidden[..8],
    );

    let output_norm_w = weights.f32_slice("output_norm.weight").unwrap();
    let (lm_head_raw, lm_head_dtype) = weights.raw_with_dtype("output.weight").unwrap();
    let vocab_size = cfg.vocab_size as usize;
    let mut expected_logits_hidden = expected_hidden.clone();
    apply_output_norm_single(
        &weights,
        &mut expected_logits_hidden,
        cfg.rms_norm_eps,
        None,
    )
    .unwrap();
    let mut expected_logits = vec![0.0f32; vocab_size];
    cpu.dequant_matmul(
        lm_head_raw,
        lm_head_dtype,
        &expected_logits_hidden,
        &mut expected_logits,
        vocab_size,
        1,
        dim,
    );
    let expected_argmax = expected_logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;

    let combined_hidden_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), &hidden_before_moe)
            .unwrap();
    let output_norm_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), output_norm_w).unwrap();
    let lm_head_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), lm_head_raw).unwrap();
    let logits_buf = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        vocab_size * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let moe_scratch = metal_ops.moe_batch_scratch_view().unwrap();
    let dequant_dispatch = metal_ops.dequant_dispatch_config();
    metal_ops
        .device
        .execute_sync(|encoder| {
            let barrier = crate::model::shared::DecodeBarrierCtx::new(
                encoder,
                crate::model::execution_plan::DecodeBarrierPlan::Smart,
            );
            barrier.pre_dispatch(&[&combined_hidden_buf], &[&combined_hidden_buf]);
            metal_ops.encode_moe_ffn_gpu_resident_cached_with_scratch(
                encoder,
                moe_scratch,
                &combined_hidden_buf,
                &ffn_norm_w_buf,
                &router_buf,
                router_dtype,
                &gate_buf,
                gate_dtype,
                &up_buf,
                up_dtype,
                &down_buf,
                down_dtype,
                1,
                n_expert,
                n_expert_used,
                dim,
                expert_inter_dim,
                gate_stride,
                up_stride,
                down_stride,
                cfg.rms_norm_eps,
                shared_expert.as_ref(),
                false,
            )?;
            barrier.post_dispatch(&[&combined_hidden_buf], &[&combined_hidden_buf]);
            barrier.step(encoder);

            barrier.pre_dispatch(&[&combined_hidden_buf], &[&combined_hidden_buf]);
            metal_ops.elementwise.encode_rms_norm(
                encoder,
                &combined_hidden_buf,
                &output_norm_buf,
                dim as u32,
                cfg.rms_norm_eps,
            );
            barrier.post_dispatch(&[&combined_hidden_buf], &[&combined_hidden_buf]);
            barrier.step(encoder);

            barrier.pre_dispatch(&[&combined_hidden_buf], &[&logits_buf]);
            match lm_head_dtype {
                GgmlType::Q8_0 => metal_ops.dequant.encode_fused_matvec_q8_0_with_config(
                    encoder,
                    &lm_head_buf,
                    &combined_hidden_buf,
                    &logits_buf,
                    vocab_size as u32,
                    dim as u32,
                    dequant_dispatch,
                ),
                GgmlType::Q4K => metal_ops.dequant.encode_fused_matvec_q4_k_with_config(
                    encoder,
                    &lm_head_buf,
                    &combined_hidden_buf,
                    &logits_buf,
                    vocab_size as u32,
                    dim as u32,
                    dequant_dispatch,
                ),
                GgmlType::Q5K => metal_ops.dequant.encode_fused_matvec_q5_k_with_config(
                    encoder,
                    &lm_head_buf,
                    &combined_hidden_buf,
                    &logits_buf,
                    vocab_size as u32,
                    dim as u32,
                    dequant_dispatch,
                ),
                GgmlType::Q6K => metal_ops.dequant.encode_fused_matvec_q6_k_with_config(
                    encoder,
                    &lm_head_buf,
                    &combined_hidden_buf,
                    &logits_buf,
                    vocab_size as u32,
                    dim as u32,
                    dequant_dispatch,
                ),
                GgmlType::F32 => metal_ops.matmul.encode_matvec(
                    encoder,
                    &lm_head_buf,
                    &combined_hidden_buf,
                    &logits_buf,
                    vocab_size as u32,
                    dim as u32,
                ),
                other => panic!("unsupported lm_head dtype for resident decode test: {other:?}"),
            }
            barrier.post_dispatch(&[&combined_hidden_buf], &[&logits_buf]);
            barrier.step(encoder);
            barrier.flush();
            Ok(())
        })
        .unwrap();

    let actual_logits = unsafe { logits_buf.as_slice::<f32>()[..vocab_size].to_vec() };
    let actual_argmax = actual_logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    assert_eq!(
        actual_argmax, expected_argmax,
        "real Qwen3.5-35B-A3B layer-4 resident MoE + output head argmax mismatch: actual={} expected={}",
        actual_argmax, expected_argmax
    );
}

#[test]
fn test_real_qwen35_35b_a3b_layer4_native_hidden_before_moe_matches_cpu_for_prompt_state() {
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&model.header).unwrap();
    let weights = WeightStore::new(&model);
    let cpu = CpuBackend;
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();

    let layer_idx = 4usize;
    let dim = cfg.embedding_dim as usize;
    let inter_dim = cfg.expert_intermediate_dim.unwrap() as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let dims = Qwen3_5Forward::recurrent_dims(&cfg).unwrap();
    let recurrent_slot = 0usize;
    let recurrent_slot_indices = [recurrent_slot];
    let full_attn_params = AttentionParams::new(n_heads, n_kv_heads, head_dim);
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&model.header).unwrap();
    let prompt_token_ids = tokenizer.encode("The capital of France is", true);
    let final_position = prompt_token_ids.len() - 1;

    let build_layer4_entry = || {
        let mut qwen_kv = crate::kv::Qwen3_5Kv::new(
            cfg.n_layers as usize,
            cfg.n_kv_heads as usize,
            cfg.head_dim as usize,
            32,
            cfg.qwen35_full_attention_interval.unwrap() as usize,
            cfg.qwen35_ssm_conv_kernel.unwrap() as usize,
            cfg.qwen35_ssm_inner_size.unwrap() as usize,
            cfg.qwen35_ssm_state_size.unwrap() as usize,
            cfg.qwen35_ssm_time_step_rank.unwrap() as usize,
            cfg.qwen35_ssm_group_count.unwrap() as usize,
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
        let mut hidden_before_layer4 = None;

        for (position, &token_id) in prompt_token_ids.iter().enumerate() {
            let mut hidden = vec![0.0f32; dim];
            weights
                .dequantize_row("token_embd.weight", token_id as usize, &mut hidden)
                .unwrap();

            for layer in 0..layer_idx {
                let prefix = format!("blk.{layer}");
                apply_attention_norm_single(
                    &weights,
                    &prefix,
                    &hidden,
                    &mut norm_buf,
                    cfg.rms_norm_eps,
                    None,
                )
                .unwrap();

                if cfg.qwen35_is_recurrent_layer(layer) {
                    Qwen3_5Forward::run_recurrent_single_layer(
                        &cfg,
                        &cpu,
                        &weights,
                        &prefix,
                        &mut qwen_kv,
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
                        None,
                    )
                    .unwrap();
                } else {
                    Qwen3_5Forward::run_full_attention_single_layer(
                        &cfg,
                        &cpu,
                        &weights,
                        &prefix,
                        &mut qwen_kv,
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
                        None,
                    )
                    .unwrap();
                }

                Qwen3_5Forward::apply_layer_tail_single(
                    &cfg,
                    &cpu,
                    &weights,
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
                    None,
                )
                .unwrap();
            }

            if position == final_position {
                hidden_before_layer4 = Some(hidden);
            } else {
                qwen_kv.finalize_token();
            }
        }

        (
            hidden_before_layer4.expect("failed to capture layer-4 entry hidden"),
            qwen_kv,
        )
    };

    let (cpu_hidden_before_layer4, mut cpu_kv) = build_layer4_entry();
    let (native_hidden_before_layer4, native_kv) = build_layer4_entry();
    assert_eq!(cpu_hidden_before_layer4, native_hidden_before_layer4);

    let prefix = format!("blk.{layer_idx}");
    let mut norm_buf = vec![0.0f32; dim];
    apply_attention_norm_single(
        &weights,
        &prefix,
        &cpu_hidden_before_layer4,
        &mut norm_buf,
        cfg.rms_norm_eps,
        None,
    )
    .unwrap();

    let mut cpu_rec_qkv = vec![0.0f32; dims.conv_dim()];
    let mut cpu_rec_z = vec![0.0f32; dims.inner_size];
    let mut cpu_rec_beta = vec![0.0f32; dims.time_step_rank];
    let mut cpu_rec_alpha = vec![0.0f32; dims.time_step_rank];
    let mut cpu_rec_out = vec![0.0f32; dims.inner_size];
    let mut expected_proj = vec![0.0f32; dim];
    Qwen3_5Forward::run_recurrent_single_layer(
        &cfg,
        &cpu,
        &weights,
        &prefix,
        &mut cpu_kv,
        recurrent_slot,
        layer_idx,
        final_position,
        dims,
        &recurrent_slot_indices,
        &norm_buf,
        &mut cpu_rec_qkv,
        &mut cpu_rec_z,
        &mut cpu_rec_beta,
        &mut cpu_rec_alpha,
        &mut cpu_rec_out,
        &mut expected_proj,
        dim,
        None,
    )
    .unwrap();
    let mut expected_hidden_before_moe = cpu_hidden_before_layer4.clone();
    silu::elementwise_add(&mut expected_hidden_before_moe, &expected_proj);

    let input_ops = Qwen3_5Forward::recurrent_input_ops(&weights, &prefix, dims).unwrap();
    let recurrent_dtypes = Qwen3_5NativeRecurrentDtypes {
        wqkv: input_ops[0].1,
        wgate: input_ops[1].1,
        wbeta: input_ops[2].1,
        walpha: input_ops[3].1,
        wssm_out: Qwen3_5Forward::recurrent_output_op(&weights, &prefix, dim)
            .unwrap()
            .1,
    };
    let runtime = Qwen3_5Forward::recurrent_runtime_tensors(&weights, &prefix).unwrap();
    let hidden_gpu = ax_engine_metal::MetalBuffer::from_slice(
        metal_ops.device.device(),
        &native_hidden_before_layer4,
    )
    .unwrap();
    let norm_gpu =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), &norm_buf).unwrap();
    let qkv_gpu = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        dims.conv_dim() * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let z_gpu = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        dims.inner_size * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let beta_gpu = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        dims.time_step_rank * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let alpha_gpu = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        dims.time_step_rank * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let recurrent_out_gpu = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        dims.inner_size * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let proj_gpu = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        dim * std::mem::size_of::<f32>(),
    )
    .unwrap();

    let (wqkv_raw, _, _) = input_ops[0];
    let (wgate_raw, _, _) = input_ops[1];
    let (wbeta_raw, _, _) = input_ops[2];
    let (walpha_raw, _, _) = input_ops[3];
    let (wssm_out_raw, _, _) = Qwen3_5Forward::recurrent_output_op(&weights, &prefix, dim).unwrap();
    let wqkv_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), wqkv_raw).unwrap();
    let wgate_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), wgate_raw).unwrap();
    let wbeta_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), wbeta_raw).unwrap();
    let walpha_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), walpha_raw).unwrap();
    let conv_kernel_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), runtime.conv_kernel)
            .unwrap();
    let ssm_norm_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), runtime.ssm_norm)
            .unwrap();
    let dt_bias_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), runtime.dt_bias)
            .unwrap();
    let ssm_a_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), runtime.a).unwrap();
    let wssm_out_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), wssm_out_raw).unwrap();

    let recurrent_weights = Qwen3_5NativeRecurrentCachedWeights {
        wqkv: &wqkv_buf,
        wgate: &wgate_buf,
        wbeta: &wbeta_buf,
        walpha: &walpha_buf,
        conv_kernel: &conv_kernel_buf,
        ssm_norm: &ssm_norm_buf,
        dt_bias: &dt_bias_buf,
        ssm_a: &ssm_a_buf,
        wssm_out: &wssm_out_buf,
    };
    let recurrent_scratch = Qwen3_5NativeRecurrentProjectionScratch {
        qkv: &qkv_gpu,
        z: &z_gpu,
        beta: &beta_gpu,
        alpha: &alpha_gpu,
    };
    let conv_state_stride = native_kv.conv_cache_len() * native_kv.conv_dim();
    let recurrent_state_stride = native_kv.recurrent_state_len();
    let _ = metal_ops.sync_qwen35_slot_buffers_from_kv(&native_kv, layer_idx, recurrent_slot);

    metal_ops
        .device
        .execute_sync(|encoder| {
            let barrier = crate::model::shared::DecodeBarrierCtx::new(
                encoder,
                crate::model::execution_plan::DecodeBarrierPlan::Explicit,
            );
            metal_ops.with_qwen35_recurrent_slot_buffer_for_kv(
                &native_kv,
                layer_idx,
                recurrent_slot,
                conv_state_stride,
                recurrent_state_stride,
                |slot_buffers| -> anyhow::Result<()> {
                    Qwen3_5Forward::encode_qwen35_native_recurrent_layer(
                        metal_ops,
                        encoder,
                        &barrier,
                        recurrent_weights,
                        recurrent_scratch,
                        slot_buffers,
                        &norm_gpu,
                        &recurrent_out_gpu,
                        &proj_gpu,
                        native_kv.conv_cache_len(),
                        dims,
                        dim,
                        cfg.rms_norm_eps,
                        recurrent_dtypes,
                        metal_ops.dequant_dispatch_config(),
                    )?;
                    metal_ops.elementwise.encode_elementwise_add_batch(
                        encoder,
                        &hidden_gpu,
                        &proj_gpu,
                        dim as u32,
                        1,
                    );
                    barrier.step(encoder);
                    barrier.flush();
                    Ok(())
                },
            )
        })
        .unwrap();

    let actual_hidden_before_moe = unsafe { hidden_gpu.as_slice::<f32>()[..dim].to_vec() };
    let diff = max_abs_diff(&actual_hidden_before_moe, &expected_hidden_before_moe);
    let scale = expected_hidden_before_moe
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 5e-3,
        "real Qwen3.5-35B-A3B layer-4 native recurrent tail mismatch on prompt state: rel_diff={}, max_diff={diff}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual_hidden_before_moe[..8],
        &expected_hidden_before_moe[..8],
    );
}

#[test]
#[ignore = "native qwen3.5 MoE GPU decode remains experimental; use targeted layer tests instead"]
fn test_real_qwen35_35b_a3b_layer0_native_hidden_matches_cpu_via_model_scratch() {
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let cpu = CpuBackend;
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&mapped.header).unwrap();
    let token_id = tokenizer.encode("The capital of France is", true)[0];

    let dim = cfg.embedding_dim as usize;
    let inter_dim = cfg.expert_intermediate_dim.unwrap() as usize;
    let dims = Qwen3_5Forward::recurrent_dims(&cfg).unwrap();
    let recurrent_slot = 0usize;
    let recurrent_slot_indices = [recurrent_slot];

    let mut cpu_kv = crate::kv::Qwen3_5Kv::new(
        cfg.n_layers as usize,
        cfg.n_kv_heads as usize,
        cfg.head_dim as usize,
        8,
        cfg.qwen35_full_attention_interval.unwrap() as usize,
        cfg.qwen35_ssm_conv_kernel.unwrap() as usize,
        cfg.qwen35_ssm_inner_size.unwrap() as usize,
        cfg.qwen35_ssm_state_size.unwrap() as usize,
        cfg.qwen35_ssm_time_step_rank.unwrap() as usize,
        cfg.qwen35_ssm_group_count.unwrap() as usize,
    );
    let mut expected_hidden = vec![0.0f32; dim];
    weights
        .dequantize_row("token_embd.weight", token_id as usize, &mut expected_hidden)
        .unwrap();
    let mut norm_buf = vec![0.0f32; dim];
    let mut proj_buf = vec![0.0f32; dim];
    let mut gate_buf = vec![0.0f32; inter_dim];
    let mut up_buf = vec![0.0f32; inter_dim];
    let mut down_buf = vec![0.0f32; dim];
    let mut rec_qkv = vec![0.0f32; dims.conv_dim()];
    let mut rec_z = vec![0.0f32; dims.inner_size];
    let mut rec_beta = vec![0.0f32; dims.time_step_rank];
    let mut rec_alpha = vec![0.0f32; dims.time_step_rank];
    let mut rec_out = vec![0.0f32; dims.inner_size];

    apply_attention_norm_single(
        &weights,
        "blk.0",
        &expected_hidden,
        &mut norm_buf,
        cfg.rms_norm_eps,
        None,
    )
    .unwrap();
    Qwen3_5Forward::run_recurrent_single_layer(
        &cfg,
        &cpu,
        &weights,
        "blk.0",
        &mut cpu_kv,
        recurrent_slot,
        0,
        0,
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
        None,
    )
    .unwrap();
    Qwen3_5Forward::apply_layer_tail_single(
        &cfg,
        &cpu,
        &weights,
        "blk.0",
        &mut expected_hidden,
        &proj_buf,
        &mut norm_buf,
        &mut gate_buf,
        &mut up_buf,
        &mut down_buf,
        dim,
        inter_dim,
        cfg.rms_norm_eps,
        0,
        0,
        None,
    )
    .unwrap();
    crate::model::shared::apply_output_norm_single(
        &weights,
        &mut expected_hidden,
        cfg.rms_norm_eps,
        None,
    )
    .unwrap();

    with_env_vars(
        &[
            ("AX_HYBRID_DECODE", Some("metal")),
            ("AX_QWEN35_GPU_DECODE", Some("1")),
            ("AX_QWEN35_GPU_BATCH_PREFILL", Some("0")),
            ("AX_QWEN35_DEBUG_MAX_LAYER", Some("0")),
        ],
        || {
            let backend = Box::new(MetalBackend::new().unwrap());
            let metal_ops_ptr: *const crate::backend::metal::MetalOps =
                backend.metal_ops().unwrap() as *const crate::backend::metal::MetalOps;
            let metal_model =
                crate::model::InferenceModel::with_backend(cfg.clone(), backend).unwrap();
            let mut metal_kv = metal_model.create_model_kv_for_weights(&weights);
            let mut logits = vec![0.0f32; cfg.vocab_size as usize];
            metal_model
                .forward_single(token_id, 0, &mut metal_kv, &weights, &mut logits)
                .unwrap();

            let metal_ops = unsafe { &*metal_ops_ptr };
            let scratch_guard = metal_ops.scratches();
            let scratch = scratch_guard.as_ref().unwrap();
            let actual_hidden = unsafe { scratch.hidden.as_slice::<f32>()[..dim].to_vec() };
            let diff = max_abs_diff(&actual_hidden, &expected_hidden);
            let scale = expected_hidden
                .iter()
                .copied()
                .map(f32::abs)
                .fold(0.0f32, f32::max)
                .max(1.0);
            assert!(
                diff / scale < 5e-3,
                "real Qwen3.5-35B-A3B layer-0 native output-norm hidden mismatch via model scratch: rel_diff={}, max_diff={diff}, actual[0..8]={:?}, expected[0..8]={:?}",
                diff / scale,
                &actual_hidden[..8],
                &expected_hidden[..8],
            );
        },
    );
}

#[test]
#[ignore = "native qwen3.5 MoE GPU decode remains experimental; use targeted layer tests instead"]
fn test_real_qwen35_35b_a3b_native_cached_layers01_hidden_matches_cpu() {
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&model.header).unwrap();
    let weights = WeightStore::new(&model);
    let cpu = CpuBackend;
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&model.header).unwrap();
    let token_id = tokenizer.encode("The capital of France is", true)[0];

    let dim = cfg.embedding_dim as usize;
    let inter_dim = cfg.expert_intermediate_dim.unwrap() as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let dims = Qwen3_5Forward::recurrent_dims(&cfg).unwrap();

    let make_kv = || {
        let mut kv = crate::kv::Qwen3_5Kv::new(
            cfg.n_layers as usize,
            cfg.n_kv_heads as usize,
            cfg.head_dim as usize,
            8,
            cfg.qwen35_full_attention_interval.unwrap() as usize,
            cfg.qwen35_ssm_conv_kernel.unwrap() as usize,
            cfg.qwen35_ssm_inner_size.unwrap() as usize,
            cfg.qwen35_ssm_state_size.unwrap() as usize,
            cfg.qwen35_ssm_time_step_rank.unwrap() as usize,
            cfg.qwen35_ssm_group_count.unwrap() as usize,
        );
        kv.enable_gpu_attention(&metal_ops.device, crate::kv::GpuKvDtype::F32)
            .unwrap();
        kv.enable_gpu_recurrent_state(&metal_ops.device).unwrap();
        kv
    };

    let mut expected_hidden = vec![0.0f32; dim];
    weights
        .dequantize_row("token_embd.weight", token_id as usize, &mut expected_hidden)
        .unwrap();

    let mut norm_buf = vec![0.0f32; dim];
    let mut proj_buf = vec![0.0f32; dim];
    let mut gate_buf = vec![0.0f32; inter_dim];
    let mut up_buf = vec![0.0f32; inter_dim];
    let mut down_buf = vec![0.0f32; dim];
    let mut rec_qkv = vec![0.0f32; dims.conv_dim()];
    let mut rec_z = vec![0.0f32; dims.inner_size];
    let mut rec_beta = vec![0.0f32; dims.time_step_rank];
    let mut rec_alpha = vec![0.0f32; dims.time_step_rank];
    let mut rec_out = vec![0.0f32; dims.inner_size];
    let recurrent_slot = 0usize;
    let recurrent_slot_indices = [recurrent_slot];
    let mut cpu_kv = crate::kv::Qwen3_5Kv::new(
        cfg.n_layers as usize,
        cfg.n_kv_heads as usize,
        cfg.head_dim as usize,
        8,
        cfg.qwen35_full_attention_interval.unwrap() as usize,
        cfg.qwen35_ssm_conv_kernel.unwrap() as usize,
        cfg.qwen35_ssm_inner_size.unwrap() as usize,
        cfg.qwen35_ssm_state_size.unwrap() as usize,
        cfg.qwen35_ssm_time_step_rank.unwrap() as usize,
        cfg.qwen35_ssm_group_count.unwrap() as usize,
    );

    let mut expected_after_layer0 = None;
    let mut expected_after_layer1 = None;
    for layer in 0..=1usize {
        let prefix = format!("blk.{layer}");
        apply_attention_norm_single(
            &weights,
            &prefix,
            &expected_hidden,
            &mut norm_buf,
            cfg.rms_norm_eps,
            None,
        )
        .unwrap();
        Qwen3_5Forward::run_recurrent_single_layer(
            &cfg,
            &cpu,
            &weights,
            &prefix,
            &mut cpu_kv,
            recurrent_slot,
            layer,
            0,
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
            None,
        )
        .unwrap();
        Qwen3_5Forward::apply_layer_tail_single(
            &cfg,
            &cpu,
            &weights,
            &prefix,
            &mut expected_hidden,
            &proj_buf,
            &mut norm_buf,
            &mut gate_buf,
            &mut up_buf,
            &mut down_buf,
            dim,
            inter_dim,
            cfg.rms_norm_eps,
            layer,
            0,
            None,
        )
        .unwrap();
        if layer == 0 {
            expected_after_layer0 = Some(expected_hidden.clone());
        } else {
            expected_after_layer1 = Some(expected_hidden.clone());
        }
    }
    let expected_after_layer0 = expected_after_layer0.unwrap();
    let expected_after_layer1 = expected_after_layer1.unwrap();

    let mut expected_skip_hidden = vec![0.0f32; dim];
    weights
        .dequantize_row(
            "token_embd.weight",
            token_id as usize,
            &mut expected_skip_hidden,
        )
        .unwrap();
    let mut skip_norm_buf = vec![0.0f32; dim];
    let mut skip_proj_buf = vec![0.0f32; dim];
    let mut skip_rec_qkv = vec![0.0f32; dims.conv_dim()];
    let mut skip_rec_z = vec![0.0f32; dims.inner_size];
    let mut skip_rec_beta = vec![0.0f32; dims.time_step_rank];
    let mut skip_rec_alpha = vec![0.0f32; dims.time_step_rank];
    let mut skip_rec_out = vec![0.0f32; dims.inner_size];
    let mut skip_cpu_kv = crate::kv::Qwen3_5Kv::new(
        cfg.n_layers as usize,
        cfg.n_kv_heads as usize,
        cfg.head_dim as usize,
        8,
        cfg.qwen35_full_attention_interval.unwrap() as usize,
        cfg.qwen35_ssm_conv_kernel.unwrap() as usize,
        cfg.qwen35_ssm_inner_size.unwrap() as usize,
        cfg.qwen35_ssm_state_size.unwrap() as usize,
        cfg.qwen35_ssm_time_step_rank.unwrap() as usize,
        cfg.qwen35_ssm_group_count.unwrap() as usize,
    );
    for layer in 0..=1usize {
        let prefix = format!("blk.{layer}");
        apply_attention_norm_single(
            &weights,
            &prefix,
            &expected_skip_hidden,
            &mut skip_norm_buf,
            cfg.rms_norm_eps,
            None,
        )
        .unwrap();
        Qwen3_5Forward::run_recurrent_single_layer(
            &cfg,
            &cpu,
            &weights,
            &prefix,
            &mut skip_cpu_kv,
            recurrent_slot,
            layer,
            0,
            dims,
            &recurrent_slot_indices,
            &skip_norm_buf,
            &mut skip_rec_qkv,
            &mut skip_rec_z,
            &mut skip_rec_beta,
            &mut skip_rec_alpha,
            &mut skip_rec_out,
            &mut skip_proj_buf,
            dim,
            None,
        )
        .unwrap();
        silu::elementwise_add(&mut expected_skip_hidden, &skip_proj_buf);
    }
    let expected_skip_ffn_after_layer1 = expected_skip_hidden;

    if !metal_ops.has_cached_model_keys() {
        Qwen3_5Forward::build_cached_model_keys_qwen35(metal_ops, &weights, &cfg).unwrap();
    }
    metal_ops.init_scratches(&cfg);
    metal_ops.init_batch_scratches(&cfg, 1);
    let cached_guard = metal_ops.cached_model_keys();
    let cached = cached_guard.as_ref().unwrap();
    let gpu_layer_keys = Qwen3_5Forward::cached_gpu_layer_keys(cached.lm_head).unwrap();
    let moe_layer_keys = Qwen3_5Forward::cached_moe_layer_keys(cached.lm_head).unwrap();
    let weight_cache = metal_ops.lock_weight_cache();
    let moe_weight_cache = metal_ops.lock_moe_weight_cache();
    let moe_scratch = metal_ops.moe_batch_scratch_view().unwrap();

    let run_until_layer = |max_layer: usize, skip_ffn: bool, expected_hidden: &[f32]| {
        let qwen_kv = make_kv();
        let gpu_attn = qwen_kv.gpu_attention().unwrap();
        let exec_plan = Qwen3_5Forward::qwen35_decode_plan(
            metal_ops,
            gpu_attn,
            cfg.embedding_dim,
            cfg.head_dim,
            1,
            false,
        );
        let rope_position = Qwen3_5Forward::rope_position(&cfg, 0);
        let mut scratch_guard = metal_ops.scratches();
        let scratch = scratch_guard.as_mut().unwrap();
        let mut input_hidden = vec![0.0f32; dim];
        weights
            .dequantize_row("token_embd.weight", token_id as usize, &mut input_hidden)
            .unwrap();
        unsafe {
            scratch.hidden.as_mut_slice::<f32>()[..dim].copy_from_slice(&input_hidden);
        }

        let max_layer_value = max_layer.to_string();
        with_env_vars(
            &[
                ("AX_QWEN35_DEBUG_MAX_LAYER", Some(max_layer_value.as_str())),
                (
                    "AX_QWEN35_DEBUG_SKIP_FFN",
                    if skip_ffn { Some("1") } else { None },
                ),
            ],
            || {
                for layer in 0..=max_layer {
                    metal_ops
                        .device
                        .execute_sync(|encoder| {
                            let barrier = crate::model::shared::DecodeBarrierCtx::new(
                                encoder,
                                exec_plan.barriers,
                            );
                            let mut no_ops = None;
                            Qwen3_5Forward::encode_qwen35_native_decode_layers(
                                metal_ops,
                                encoder,
                                &barrier,
                                &cfg,
                                &qwen_kv,
                                cached,
                                &gpu_layer_keys,
                                Some(&moe_layer_keys),
                                &weight_cache,
                                Some(&moe_weight_cache),
                                scratch,
                                Some(moe_scratch),
                                &exec_plan,
                                exec_plan.barriers
                                    == crate::model::execution_plan::DecodeBarrierPlan::Explicit,
                                layer,
                                layer + 1,
                                0,
                                1,
                                rope_position,
                                recurrent_slot,
                                dims,
                                dim,
                                inter_dim,
                                q_dim,
                                kv_dim,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                qwen_kv.conv_cache_len(),
                                cfg.rms_norm_eps,
                                &mut no_ops,
                            )?;
                            barrier.flush();
                            Ok(())
                        })
                        .unwrap();
                }
            },
        );

        let actual_hidden = unsafe { scratch.hidden.as_slice::<f32>()[..dim].to_vec() };
        let diff = max_abs_diff(&actual_hidden, expected_hidden);
        let scale = expected_hidden
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max)
            .max(1.0);
        assert!(
            diff / scale < 5e-3,
            "real Qwen3.5-35B-A3B cached native hidden mismatch at max_layer={max_layer} skip_ffn={skip_ffn}: rel_diff={}, max_diff={diff}, actual[0..8]={:?}, expected[0..8]={:?}",
            diff / scale,
            &actual_hidden[..8],
            &expected_hidden[..8],
        );
    };

    run_until_layer(0, false, &expected_after_layer0);
    run_until_layer(1, true, &expected_skip_ffn_after_layer1);
    run_until_layer(1, false, &expected_after_layer1);
}

#[test]
#[ignore = "native qwen3.5 MoE GPU decode remains experimental; use targeted layer tests instead"]
fn test_real_qwen35_35b_a3b_serial_native_trace_matches_cpu() {
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&mapped.header).unwrap();
    let prompt_token_ids = tokenizer.encode("The capital of France is", true);
    let vocab_size = cfg.vocab_size as usize;

    with_env_vars(
        &[
            ("AX_HYBRID_DECODE", Some("metal")),
            ("AX_QWEN35_GPU_DECODE", Some("1")),
            ("AX_QWEN35_GPU_BATCH_PREFILL", Some("0")),
            ("AX_SERIAL_PREFILL", None),
        ],
        || {
            let cpu_model = crate::model::InferenceModel::new(cfg.clone()).unwrap();
            let metal_model = crate::model::InferenceModel::with_backend(
                cfg.clone(),
                Box::new(MetalBackend::new().unwrap()),
            )
            .unwrap();
            let mut cpu_kv = cpu_model.create_model_kv_for_weights(&weights);
            let mut metal_kv = metal_model.create_model_kv_for_weights(&weights);
            let mut cpu_logits = vec![0.0f32; vocab_size];
            let mut metal_logits = vec![0.0f32; vocab_size];

            for (position, &token_id) in prompt_token_ids.iter().enumerate() {
                cpu_logits.fill(0.0);
                metal_logits.fill(0.0);
                cpu_model
                    .forward_single(token_id, position, &mut cpu_kv, &weights, &mut cpu_logits)
                    .unwrap();
                metal_model
                    .forward_single(
                        token_id,
                        position,
                        &mut metal_kv,
                        &weights,
                        &mut metal_logits,
                    )
                    .unwrap();
                assert_qwen35_serial_trace_step_matches(
                    &cfg,
                    &tokenizer,
                    &metal_model,
                    "prompt",
                    position,
                    token_id,
                    &cpu_logits,
                    &metal_logits,
                    &cpu_kv,
                    &mut metal_kv,
                );
            }

            let mut next_token = argmax_index(&cpu_logits) as u32;
            for generation_step in 0..2usize {
                let position = prompt_token_ids.len() + generation_step;
                cpu_logits.fill(0.0);
                metal_logits.fill(0.0);
                cpu_model
                    .forward_single(next_token, position, &mut cpu_kv, &weights, &mut cpu_logits)
                    .unwrap();
                metal_model
                    .forward_single(
                        next_token,
                        position,
                        &mut metal_kv,
                        &weights,
                        &mut metal_logits,
                    )
                    .unwrap();
                assert_qwen35_serial_trace_step_matches(
                    &cfg,
                    &tokenizer,
                    &metal_model,
                    "decode",
                    position,
                    next_token,
                    &cpu_logits,
                    &metal_logits,
                    &cpu_kv,
                    &mut metal_kv,
                );
                next_token = argmax_index(&cpu_logits) as u32;
            }
        },
    );
}

#[test]
#[ignore = "diagnostic helper for native qwen3.5 decode divergence; run manually while fixing decode parity"]
fn test_real_qwen35_35b_a3b_first_native_decode_layer_mismatch_after_prompt_token() {
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let cpu = CpuBackend;
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&mapped.header).unwrap();
    let token_id = tokenizer.encode("The capital of France is", true)[0];

    let dim = cfg.embedding_dim as usize;
    let inter_dim = cfg.expert_intermediate_dim.unwrap() as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let dims = Qwen3_5Forward::recurrent_dims(&cfg).unwrap();
    let recurrent_slot = 0usize;
    let recurrent_slot_indices = [recurrent_slot];
    let full_attn_params = AttentionParams::new(n_heads, n_kv_heads, head_dim);

    let mut cpu_kv = crate::kv::Qwen3_5Kv::new(
        cfg.n_layers as usize,
        cfg.n_kv_heads as usize,
        cfg.head_dim as usize,
        8,
        cfg.qwen35_full_attention_interval.unwrap() as usize,
        cfg.qwen35_ssm_conv_kernel.unwrap() as usize,
        cfg.qwen35_ssm_inner_size.unwrap() as usize,
        cfg.qwen35_ssm_state_size.unwrap() as usize,
        cfg.qwen35_ssm_time_step_rank.unwrap() as usize,
        cfg.qwen35_ssm_group_count.unwrap() as usize,
    );
    let mut native_kv = crate::kv::Qwen3_5Kv::new(
        cfg.n_layers as usize,
        cfg.n_kv_heads as usize,
        cfg.head_dim as usize,
        8,
        cfg.qwen35_full_attention_interval.unwrap() as usize,
        cfg.qwen35_ssm_conv_kernel.unwrap() as usize,
        cfg.qwen35_ssm_inner_size.unwrap() as usize,
        cfg.qwen35_ssm_state_size.unwrap() as usize,
        cfg.qwen35_ssm_time_step_rank.unwrap() as usize,
        cfg.qwen35_ssm_group_count.unwrap() as usize,
    );
    native_kv
        .enable_gpu_attention(&metal_ops.device, crate::kv::GpuKvDtype::F32)
        .unwrap();
    native_kv
        .enable_gpu_recurrent_state(&metal_ops.device)
        .unwrap();

    if !metal_ops.has_cached_model_keys() {
        Qwen3_5Forward::build_cached_model_keys_qwen35(metal_ops, &weights, &cfg).unwrap();
    }
    metal_ops.init_scratches(&cfg);
    metal_ops.init_batch_scratches(&cfg, 1);

    let cached_guard = metal_ops.cached_model_keys();
    let cached = cached_guard.as_ref().unwrap();
    let gpu_layer_keys = Qwen3_5Forward::cached_gpu_layer_keys(cached.lm_head).unwrap();
    let moe_layer_keys = Qwen3_5Forward::cached_moe_layer_keys(cached.lm_head).unwrap();
    let weight_cache = metal_ops.lock_weight_cache();
    let moe_weight_cache = metal_ops.lock_moe_weight_cache();
    let moe_scratch = metal_ops.moe_batch_scratch_view().unwrap();
    let exec_plan = Qwen3_5Forward::qwen35_decode_plan(
        metal_ops,
        native_kv.gpu_attention().unwrap(),
        cfg.embedding_dim,
        cfg.head_dim,
        1,
        false,
    );
    let rope_position = Qwen3_5Forward::rope_position(&cfg, 0);

    let mut expected_hidden = vec![0.0f32; dim];
    weights
        .dequantize_row("token_embd.weight", token_id as usize, &mut expected_hidden)
        .unwrap();
    let mut norm_buf = vec![0.0f32; dim];
    let mut proj_buf = vec![0.0f32; dim];
    let mut gate_buf = vec![0.0f32; inter_dim];
    let mut up_buf = vec![0.0f32; inter_dim];
    let mut down_buf = vec![0.0f32; dim];
    let mut q_gate_buf = vec![0.0f32; q_dim * 2];
    let mut q_buf = vec![0.0f32; q_dim];
    let mut k_buf = vec![0.0f32; kv_dim];
    let mut v_buf = vec![0.0f32; kv_dim];
    let mut attn_out = vec![0.0f32; q_dim];
    let mut rec_qkv = vec![0.0f32; dims.conv_dim()];
    let mut rec_z = vec![0.0f32; dims.inner_size];
    let mut rec_beta = vec![0.0f32; dims.time_step_rank];
    let mut rec_alpha = vec![0.0f32; dims.time_step_rank];
    let mut rec_out = vec![0.0f32; dims.inner_size];

    let mut scratch_guard = metal_ops.scratches();
    let scratch = scratch_guard.as_mut().unwrap();
    unsafe {
        scratch.hidden.as_mut_slice::<f32>()[..dim].copy_from_slice(&expected_hidden);
    }

    for layer in 0..cfg.n_layers as usize {
        let prefix = format!("blk.{layer}");
        apply_attention_norm_single(
            &weights,
            &prefix,
            &expected_hidden,
            &mut norm_buf,
            cfg.rms_norm_eps,
            None,
        )
        .unwrap();
        if cfg.qwen35_is_recurrent_layer(layer) {
            Qwen3_5Forward::run_recurrent_single_layer(
                &cfg,
                &cpu,
                &weights,
                &prefix,
                &mut cpu_kv,
                recurrent_slot,
                layer,
                0,
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
                None,
            )
            .unwrap();
        } else {
            Qwen3_5Forward::run_full_attention_single_layer(
                &cfg,
                &cpu,
                &weights,
                &prefix,
                &mut cpu_kv,
                layer,
                0,
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
                None,
            )
            .unwrap();
        }
        Qwen3_5Forward::apply_layer_tail_single(
            &cfg,
            &cpu,
            &weights,
            &prefix,
            &mut expected_hidden,
            &proj_buf,
            &mut norm_buf,
            &mut gate_buf,
            &mut up_buf,
            &mut down_buf,
            dim,
            inter_dim,
            cfg.rms_norm_eps,
            layer,
            0,
            None,
        )
        .unwrap();

        metal_ops
            .device
            .execute_sync(|encoder| {
                let barrier =
                    crate::model::shared::DecodeBarrierCtx::new(encoder, exec_plan.barriers);
                let mut no_ops = None;
                Qwen3_5Forward::encode_qwen35_native_decode_layers(
                    metal_ops,
                    encoder,
                    &barrier,
                    &cfg,
                    &native_kv,
                    cached,
                    &gpu_layer_keys,
                    Some(&moe_layer_keys),
                    &weight_cache,
                    Some(&moe_weight_cache),
                    scratch,
                    Some(moe_scratch),
                    &exec_plan,
                    exec_plan.barriers == crate::model::execution_plan::DecodeBarrierPlan::Explicit,
                    layer,
                    layer + 1,
                    0,
                    1,
                    rope_position,
                    recurrent_slot,
                    dims,
                    dim,
                    inter_dim,
                    q_dim,
                    kv_dim,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    native_kv.conv_cache_len(),
                    cfg.rms_norm_eps,
                    &mut no_ops,
                )?;
                barrier.flush();
                Ok(())
            })
            .unwrap();

        let actual_hidden = unsafe { scratch.hidden.as_slice::<f32>()[..dim].to_vec() };
        let diff = max_abs_diff(&actual_hidden, &expected_hidden);
        let scale = expected_hidden
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max)
            .max(1.0);
        assert!(
            diff / scale < 5e-3,
            "first native decode mismatch at layer={layer} recurrent={} rel_diff={} max_diff={} actual[0..8]={:?} expected[0..8]={:?}",
            cfg.qwen35_is_recurrent_layer(layer),
            diff / scale,
            diff,
            &actual_hidden[..8],
            &expected_hidden[..8],
        );
    }

    let cpu_model = crate::model::InferenceModel::new(cfg.clone()).unwrap();
    let mut cpu_model_kv = cpu_model.create_model_kv_for_weights(&weights);
    let mut cpu_logits = vec![0.0f32; cfg.vocab_size as usize];
    cpu_model
        .forward_single(token_id, 0, &mut cpu_model_kv, &weights, &mut cpu_logits)
        .unwrap();

    metal_ops
        .device
        .execute_sync(|encoder| {
            let barrier = crate::model::shared::DecodeBarrierCtx::new(encoder, exec_plan.barriers);
            crate::model::shared::encode_gpu_output_head(
                encoder,
                metal_ops,
                scratch,
                &scratch.hidden,
                &exec_plan,
                cached,
                &weight_cache,
                &barrier,
                dim as u32,
                cfg.vocab_size,
                cfg.rms_norm_eps,
            );
            barrier.flush();
            Ok(())
        })
        .unwrap();
    let actual_logits =
        unsafe { scratch.logits_buf.as_slice::<f32>()[..cfg.vocab_size as usize].to_vec() };
    let logits_diff = max_abs_diff(&actual_logits, &cpu_logits);
    let logits_scale = cpu_logits
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        logits_diff / logits_scale < 5e-2,
        "native decode output-head mismatch after hidden parity: rel_diff={} max_diff={} expected_argmax={} actual_argmax={}",
        logits_diff / logits_scale,
        logits_diff,
        argmax_index(&cpu_logits),
        argmax_index(&actual_logits),
    );
}

#[test]
#[ignore = "diagnostic helper for coalesced native decode range divergence; run manually while fixing decode parity"]
fn test_real_qwen35_35b_a3b_first_coalesced_native_decode_range_mismatch_after_prompt_token() {
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let cpu = CpuBackend;
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&mapped.header).unwrap();
    let token_id = tokenizer.encode("The capital of France is", true)[0];

    let dim = cfg.embedding_dim as usize;
    let inter_dim = cfg.expert_intermediate_dim.unwrap() as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let dims = Qwen3_5Forward::recurrent_dims(&cfg).unwrap();
    let recurrent_slot = 0usize;
    let recurrent_slot_indices = [recurrent_slot];
    let full_attn_params = AttentionParams::new(n_heads, n_kv_heads, head_dim);

    let mut cpu_kv = crate::kv::Qwen3_5Kv::new(
        cfg.n_layers as usize,
        cfg.n_kv_heads as usize,
        cfg.head_dim as usize,
        8,
        cfg.qwen35_full_attention_interval.unwrap() as usize,
        cfg.qwen35_ssm_conv_kernel.unwrap() as usize,
        cfg.qwen35_ssm_inner_size.unwrap() as usize,
        cfg.qwen35_ssm_state_size.unwrap() as usize,
        cfg.qwen35_ssm_time_step_rank.unwrap() as usize,
        cfg.qwen35_ssm_group_count.unwrap() as usize,
    );
    let mut native_kv = crate::kv::Qwen3_5Kv::new(
        cfg.n_layers as usize,
        cfg.n_kv_heads as usize,
        cfg.head_dim as usize,
        8,
        cfg.qwen35_full_attention_interval.unwrap() as usize,
        cfg.qwen35_ssm_conv_kernel.unwrap() as usize,
        cfg.qwen35_ssm_inner_size.unwrap() as usize,
        cfg.qwen35_ssm_state_size.unwrap() as usize,
        cfg.qwen35_ssm_time_step_rank.unwrap() as usize,
        cfg.qwen35_ssm_group_count.unwrap() as usize,
    );
    native_kv
        .enable_gpu_attention(&metal_ops.device, crate::kv::GpuKvDtype::F32)
        .unwrap();
    native_kv
        .enable_gpu_recurrent_state(&metal_ops.device)
        .unwrap();

    if !metal_ops.has_cached_model_keys() {
        Qwen3_5Forward::build_cached_model_keys_qwen35(metal_ops, &weights, &cfg).unwrap();
    }
    metal_ops.init_scratches(&cfg);
    metal_ops.init_batch_scratches(&cfg, 1);

    let cached_guard = metal_ops.cached_model_keys();
    let cached = cached_guard.as_ref().unwrap();
    let gpu_layer_keys = Qwen3_5Forward::cached_gpu_layer_keys(cached.lm_head).unwrap();
    let moe_layer_keys = Qwen3_5Forward::cached_moe_layer_keys(cached.lm_head).unwrap();
    let weight_cache = metal_ops.lock_weight_cache();
    let moe_weight_cache = metal_ops.lock_moe_weight_cache();
    let moe_scratch = metal_ops.moe_batch_scratch_view().unwrap();
    let exec_plan = Qwen3_5Forward::qwen35_decode_plan(
        metal_ops,
        native_kv.gpu_attention().unwrap(),
        cfg.embedding_dim,
        cfg.head_dim,
        1,
        false,
    );
    let rope_position = Qwen3_5Forward::rope_position(&cfg, 0);

    let mut expected_hidden = vec![0.0f32; dim];
    weights
        .dequantize_row("token_embd.weight", token_id as usize, &mut expected_hidden)
        .unwrap();
    let mut norm_buf = vec![0.0f32; dim];
    let mut proj_buf = vec![0.0f32; dim];
    let mut gate_buf = vec![0.0f32; inter_dim];
    let mut up_buf = vec![0.0f32; inter_dim];
    let mut down_buf = vec![0.0f32; dim];
    let mut q_gate_buf = vec![0.0f32; q_dim * 2];
    let mut q_buf = vec![0.0f32; q_dim];
    let mut k_buf = vec![0.0f32; kv_dim];
    let mut v_buf = vec![0.0f32; kv_dim];
    let mut attn_out = vec![0.0f32; q_dim];
    let mut rec_qkv = vec![0.0f32; dims.conv_dim()];
    let mut rec_z = vec![0.0f32; dims.inner_size];
    let mut rec_beta = vec![0.0f32; dims.time_step_rank];
    let mut rec_alpha = vec![0.0f32; dims.time_step_rank];
    let mut rec_out = vec![0.0f32; dims.inner_size];

    let mut scratch_guard = metal_ops.scratches();
    let scratch = scratch_guard.as_mut().unwrap();
    unsafe {
        scratch.hidden.as_mut_slice::<f32>()[..dim].copy_from_slice(&expected_hidden);
    }

    with_env_var("AX_QWEN35_GPU_DECODE_COALESCE_RECURRENT", Some("1"), || {
        let dispatch_plan = Qwen3_5Forward::qwen35_native_decode_dispatch_plan(
            &cfg,
            &exec_plan,
            cfg.n_layers as usize,
        );
        let dispatch_encoder = dispatch_plan.encoder;
        let dispatch_barriers = dispatch_plan.barriers;
        assert!(
            dispatch_plan
                .layer_ranges
                .iter()
                .any(|range| range.end_exclusive > range.start + 1),
            "expected at least one coalesced recurrent run"
        );

        for range in dispatch_plan.layer_ranges {
            for layer in range.start..range.end_exclusive {
                let prefix = format!("blk.{layer}");
                apply_attention_norm_single(
                    &weights,
                    &prefix,
                    &expected_hidden,
                    &mut norm_buf,
                    cfg.rms_norm_eps,
                    None,
                )
                .unwrap();
                if cfg.qwen35_is_recurrent_layer(layer) {
                    Qwen3_5Forward::run_recurrent_single_layer(
                        &cfg,
                        &cpu,
                        &weights,
                        &prefix,
                        &mut cpu_kv,
                        recurrent_slot,
                        layer,
                        0,
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
                        None,
                    )
                    .unwrap();
                } else {
                    Qwen3_5Forward::run_full_attention_single_layer(
                        &cfg,
                        &cpu,
                        &weights,
                        &prefix,
                        &mut cpu_kv,
                        layer,
                        0,
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
                        None,
                    )
                    .unwrap();
                }
                Qwen3_5Forward::apply_layer_tail_single(
                    &cfg,
                    &cpu,
                    &weights,
                    &prefix,
                    &mut expected_hidden,
                    &proj_buf,
                    &mut norm_buf,
                    &mut gate_buf,
                    &mut up_buf,
                    &mut down_buf,
                    dim,
                    inter_dim,
                    cfg.rms_norm_eps,
                    layer,
                    0,
                    None,
                )
                .unwrap();
            }

            if dispatch_encoder == crate::model::execution_plan::DecodeEncoderPlan::Concurrent {
                metal_ops
                    .device
                    .execute_sync_concurrent(|encoder| {
                        let barrier =
                            crate::model::shared::DecodeBarrierCtx::new(encoder, dispatch_barriers);
                        let mut no_ops = None;
                        Qwen3_5Forward::encode_qwen35_native_decode_layers(
                            metal_ops,
                            encoder,
                            &barrier,
                            &cfg,
                            &native_kv,
                            cached,
                            &gpu_layer_keys,
                            Some(&moe_layer_keys),
                            &weight_cache,
                            Some(&moe_weight_cache),
                            scratch,
                            Some(moe_scratch),
                            &exec_plan,
                            dispatch_barriers
                                == crate::model::execution_plan::DecodeBarrierPlan::Explicit,
                            range.start,
                            range.end_exclusive,
                            0,
                            1,
                            rope_position,
                            recurrent_slot,
                            dims,
                            dim,
                            inter_dim,
                            q_dim,
                            kv_dim,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            native_kv.conv_cache_len(),
                            cfg.rms_norm_eps,
                            &mut no_ops,
                        )?;
                        barrier.flush();
                        Ok(())
                    })
                    .unwrap();
            } else {
                metal_ops
                    .device
                    .execute_sync(|encoder| {
                        let barrier =
                            crate::model::shared::DecodeBarrierCtx::new(encoder, dispatch_barriers);
                        let mut no_ops = None;
                        Qwen3_5Forward::encode_qwen35_native_decode_layers(
                            metal_ops,
                            encoder,
                            &barrier,
                            &cfg,
                            &native_kv,
                            cached,
                            &gpu_layer_keys,
                            Some(&moe_layer_keys),
                            &weight_cache,
                            Some(&moe_weight_cache),
                            scratch,
                            Some(moe_scratch),
                            &exec_plan,
                            dispatch_barriers
                                == crate::model::execution_plan::DecodeBarrierPlan::Explicit,
                            range.start,
                            range.end_exclusive,
                            0,
                            1,
                            rope_position,
                            recurrent_slot,
                            dims,
                            dim,
                            inter_dim,
                            q_dim,
                            kv_dim,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            native_kv.conv_cache_len(),
                            cfg.rms_norm_eps,
                            &mut no_ops,
                        )?;
                        barrier.flush();
                        Ok(())
                    })
                    .unwrap();
            }

            let actual_hidden = unsafe { scratch.hidden.as_slice::<f32>()[..dim].to_vec() };
            let diff = max_abs_diff(&actual_hidden, &expected_hidden);
            let scale = expected_hidden
                .iter()
                .copied()
                .map(f32::abs)
                .fold(0.0f32, f32::max)
                .max(1.0);
            assert!(
                diff / scale < 5e-3,
                "first coalesced native decode range mismatch at range={:?} rel_diff={} max_diff={} actual[0..8]={:?} expected[0..8]={:?}",
                range,
                diff / scale,
                diff,
                &actual_hidden[..8],
                &expected_hidden[..8],
            );
        }
    });
}

#[test]
fn test_real_qwen35_35b_a3b_layer1_native_hidden_before_moe_matches_cpu_with_gpu_recurrent_state() {
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&model.header).unwrap();
    let weights = WeightStore::new(&model);
    let cpu = CpuBackend;
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();

    let layer_idx = 1usize;
    assert!(cfg.qwen35_is_recurrent_layer(layer_idx));

    let dim = cfg.embedding_dim as usize;
    let inter_dim = cfg.expert_intermediate_dim.unwrap() as usize;
    let dims = Qwen3_5Forward::recurrent_dims(&cfg).unwrap();
    let recurrent_slot = 0usize;
    let recurrent_slot_indices = [recurrent_slot];
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&model.header).unwrap();
    let prompt_token_ids = tokenizer.encode("The capital of France is", true);
    let token_id = prompt_token_ids[0];

    let make_kv = || {
        crate::kv::Qwen3_5Kv::new(
            cfg.n_layers as usize,
            cfg.n_kv_heads as usize,
            cfg.head_dim as usize,
            8,
            cfg.qwen35_full_attention_interval.unwrap() as usize,
            cfg.qwen35_ssm_conv_kernel.unwrap() as usize,
            cfg.qwen35_ssm_inner_size.unwrap() as usize,
            cfg.qwen35_ssm_state_size.unwrap() as usize,
            cfg.qwen35_ssm_time_step_rank.unwrap() as usize,
            cfg.qwen35_ssm_group_count.unwrap() as usize,
        )
    };

    let mut cpu_kv = make_kv();
    let mut native_kv = make_kv();
    native_kv
        .enable_gpu_recurrent_state(&metal_ops.device)
        .unwrap();
    let mut combined_kv = make_kv();
    combined_kv
        .enable_gpu_recurrent_state(&metal_ops.device)
        .unwrap();

    let mut hidden = vec![0.0f32; dim];
    weights
        .dequantize_row("token_embd.weight", token_id as usize, &mut hidden)
        .unwrap();
    let mut hidden_after_layer0 = hidden.clone();

    let mut norm_buf = vec![0.0f32; dim];
    let mut proj_buf = vec![0.0f32; dim];
    let mut gate_buf = vec![0.0f32; inter_dim];
    let mut up_buf = vec![0.0f32; inter_dim];
    let mut down_buf = vec![0.0f32; dim];

    let prefix0 = "blk.0".to_string();
    apply_attention_norm_single(
        &weights,
        &prefix0,
        &hidden_after_layer0,
        &mut norm_buf,
        cfg.rms_norm_eps,
        None,
    )
    .unwrap();
    let mut layer0_rec_qkv = vec![0.0f32; dims.conv_dim()];
    let mut layer0_rec_z = vec![0.0f32; dims.inner_size];
    let mut layer0_rec_beta = vec![0.0f32; dims.time_step_rank];
    let mut layer0_rec_alpha = vec![0.0f32; dims.time_step_rank];
    let mut layer0_rec_out = vec![0.0f32; dims.inner_size];
    Qwen3_5Forward::run_recurrent_single_layer(
        &cfg,
        &cpu,
        &weights,
        &prefix0,
        &mut cpu_kv,
        recurrent_slot,
        0,
        0,
        dims,
        &recurrent_slot_indices,
        &norm_buf,
        &mut layer0_rec_qkv,
        &mut layer0_rec_z,
        &mut layer0_rec_beta,
        &mut layer0_rec_alpha,
        &mut layer0_rec_out,
        &mut proj_buf,
        dim,
        None,
    )
    .unwrap();
    Qwen3_5Forward::apply_layer_tail_single(
        &cfg,
        &cpu,
        &weights,
        &prefix0,
        &mut hidden_after_layer0,
        &proj_buf,
        &mut norm_buf,
        &mut gate_buf,
        &mut up_buf,
        &mut down_buf,
        dim,
        inter_dim,
        cfg.rms_norm_eps,
        0,
        0,
        None,
    )
    .unwrap();

    let prefix = format!("blk.{layer_idx}");
    apply_attention_norm_single(
        &weights,
        &prefix,
        &hidden_after_layer0,
        &mut norm_buf,
        cfg.rms_norm_eps,
        None,
    )
    .unwrap();

    let mut cpu_rec_qkv = vec![0.0f32; dims.conv_dim()];
    let mut cpu_rec_z = vec![0.0f32; dims.inner_size];
    let mut cpu_rec_beta = vec![0.0f32; dims.time_step_rank];
    let mut cpu_rec_alpha = vec![0.0f32; dims.time_step_rank];
    let mut cpu_rec_out = vec![0.0f32; dims.inner_size];
    let mut expected_proj = vec![0.0f32; dim];
    Qwen3_5Forward::run_recurrent_single_layer(
        &cfg,
        &cpu,
        &weights,
        &prefix,
        &mut cpu_kv,
        recurrent_slot,
        layer_idx,
        0,
        dims,
        &recurrent_slot_indices,
        &norm_buf,
        &mut cpu_rec_qkv,
        &mut cpu_rec_z,
        &mut cpu_rec_beta,
        &mut cpu_rec_alpha,
        &mut cpu_rec_out,
        &mut expected_proj,
        dim,
        None,
    )
    .unwrap();
    let mut expected_hidden_before_moe = hidden_after_layer0.clone();
    silu::elementwise_add(&mut expected_hidden_before_moe, &expected_proj);

    let input_ops = Qwen3_5Forward::recurrent_input_ops(&weights, &prefix, dims).unwrap();
    let recurrent_dtypes = Qwen3_5NativeRecurrentDtypes {
        wqkv: input_ops[0].1,
        wgate: input_ops[1].1,
        wbeta: input_ops[2].1,
        walpha: input_ops[3].1,
        wssm_out: Qwen3_5Forward::recurrent_output_op(&weights, &prefix, dim)
            .unwrap()
            .1,
    };
    let runtime = Qwen3_5Forward::recurrent_runtime_tensors(&weights, &prefix).unwrap();
    let hidden_gpu =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), &hidden_after_layer0)
            .unwrap();
    let norm_gpu =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), &norm_buf).unwrap();
    let qkv_gpu = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        dims.conv_dim() * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let z_gpu = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        dims.inner_size * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let beta_gpu = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        dims.time_step_rank * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let alpha_gpu = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        dims.time_step_rank * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let recurrent_out_gpu = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        dims.inner_size * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let proj_gpu = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        dim * std::mem::size_of::<f32>(),
    )
    .unwrap();

    let (wqkv_raw, _, _) = input_ops[0];
    let (wgate_raw, _, _) = input_ops[1];
    let (wbeta_raw, _, _) = input_ops[2];
    let (walpha_raw, _, _) = input_ops[3];
    let (wssm_out_raw, _, _) = Qwen3_5Forward::recurrent_output_op(&weights, &prefix, dim).unwrap();
    let wqkv_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), wqkv_raw).unwrap();
    let wgate_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), wgate_raw).unwrap();
    let wbeta_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), wbeta_raw).unwrap();
    let walpha_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), walpha_raw).unwrap();
    let conv_kernel_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), runtime.conv_kernel)
            .unwrap();
    let ssm_norm_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), runtime.ssm_norm)
            .unwrap();
    let dt_bias_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), runtime.dt_bias)
            .unwrap();
    let ssm_a_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), runtime.a).unwrap();
    let wssm_out_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), wssm_out_raw).unwrap();

    let recurrent_weights = Qwen3_5NativeRecurrentCachedWeights {
        wqkv: &wqkv_buf,
        wgate: &wgate_buf,
        wbeta: &wbeta_buf,
        walpha: &walpha_buf,
        conv_kernel: &conv_kernel_buf,
        ssm_norm: &ssm_norm_buf,
        dt_bias: &dt_bias_buf,
        ssm_a: &ssm_a_buf,
        wssm_out: &wssm_out_buf,
    };
    let recurrent_scratch = Qwen3_5NativeRecurrentProjectionScratch {
        qkv: &qkv_gpu,
        z: &z_gpu,
        beta: &beta_gpu,
        alpha: &alpha_gpu,
    };
    let conv_state_stride = native_kv.conv_cache_len() * native_kv.conv_dim();
    let recurrent_state_stride = native_kv.recurrent_state_len();

    metal_ops
        .device
        .execute_sync(|encoder| {
            let barrier = crate::model::shared::DecodeBarrierCtx::new(
                encoder,
                crate::model::execution_plan::DecodeBarrierPlan::Explicit,
            );
            metal_ops.with_qwen35_recurrent_slot_buffer_for_kv(
                &native_kv,
                layer_idx,
                recurrent_slot,
                conv_state_stride,
                recurrent_state_stride,
                |slot_buffers| -> anyhow::Result<()> {
                    Qwen3_5Forward::encode_qwen35_native_recurrent_layer(
                        metal_ops,
                        encoder,
                        &barrier,
                        recurrent_weights,
                        recurrent_scratch,
                        slot_buffers,
                        &norm_gpu,
                        &recurrent_out_gpu,
                        &proj_gpu,
                        native_kv.conv_cache_len(),
                        dims,
                        dim,
                        cfg.rms_norm_eps,
                        recurrent_dtypes,
                        metal_ops.dequant_dispatch_config(),
                    )?;
                    metal_ops.elementwise.encode_elementwise_add_batch(
                        encoder,
                        &hidden_gpu,
                        &proj_gpu,
                        dim as u32,
                        1,
                    );
                    barrier.step(encoder);
                    barrier.flush();
                    Ok(())
                },
            )
        })
        .unwrap();

    let actual_hidden_before_moe = unsafe { hidden_gpu.as_slice::<f32>()[..dim].to_vec() };
    let diff = max_abs_diff(&actual_hidden_before_moe, &expected_hidden_before_moe);
    let scale = expected_hidden_before_moe
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 5e-3,
        "real Qwen3.5-35B-A3B layer-1 native recurrent hidden-before-moe mismatch with gpu_recurrent_state: rel_diff={}, max_diff={diff}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual_hidden_before_moe[..8],
        &expected_hidden_before_moe[..8],
    );

    let mut expected_hidden_after_moe = expected_hidden_before_moe.clone();
    let mut expected_norm_buf = vec![0.0f32; dim];
    Qwen3_5Forward::apply_post_attention_moe_single(
        &cfg,
        &cpu,
        &weights,
        &prefix,
        &mut expected_hidden_after_moe,
        &mut expected_norm_buf,
        dim,
        inter_dim,
        cfg.rms_norm_eps,
    )
    .unwrap();
    metal_ops.init_batch_scratches(&cfg, 1);
    let moe_scratch = metal_ops.moe_batch_scratch_view().unwrap();

    if !metal_ops.has_cached_model_keys() {
        Qwen3_5Forward::build_cached_model_keys_qwen35(metal_ops, &weights, &cfg).unwrap();
    }
    let cached_guard = metal_ops.cached_model_keys();
    let cached = cached_guard.as_ref().unwrap();
    let model_key = cached.lm_head;
    let gpu_layer_keys = Qwen3_5Forward::cached_gpu_layer_keys(model_key).unwrap();
    let moe_layer_keys = Qwen3_5Forward::cached_moe_layer_keys(model_key).unwrap();
    let cached_recurrent_keys = match &gpu_layer_keys[layer_idx] {
        Qwen3_5GpuLayerKeys::Recurrent(keys) => keys,
        Qwen3_5GpuLayerKeys::FullAttention => panic!("expected recurrent layer keys"),
    };
    let cached_moe_keys = moe_layer_keys[layer_idx]
        .as_ref()
        .expect("expected cached MoE keys for recurrent layer");
    let weight_cache = metal_ops.lock_weight_cache();
    let moe_weight_cache = metal_ops.lock_moe_weight_cache();
    let cached_hidden_gpu =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), &hidden_after_layer0)
            .unwrap();
    let mut cached_kv = make_kv();
    cached_kv
        .enable_gpu_recurrent_state(&metal_ops.device)
        .unwrap();
    let cached_conv_state_stride = cached_kv.conv_cache_len() * cached_kv.conv_dim();
    let cached_recurrent_state_stride = cached_kv.recurrent_state_len();
    let cached_ffn_norm_w = weight_cache
        .get(&cached.layers[layer_idx].ffn_norm)
        .expect("cached ffn norm buffer missing");
    let cached_recurrent_weights = Qwen3_5NativeRecurrentCachedWeights {
        wqkv: weight_cache
            .get(&cached_recurrent_keys.wqkv)
            .expect("cached wqkv buffer missing"),
        wgate: weight_cache
            .get(&cached_recurrent_keys.wgate)
            .expect("cached wgate buffer missing"),
        wbeta: weight_cache
            .get(&cached_recurrent_keys.wbeta)
            .expect("cached wbeta buffer missing"),
        walpha: weight_cache
            .get(&cached_recurrent_keys.walpha)
            .expect("cached walpha buffer missing"),
        conv_kernel: weight_cache
            .get(&cached_recurrent_keys.conv_kernel)
            .expect("cached conv kernel buffer missing"),
        ssm_norm: weight_cache
            .get(&cached_recurrent_keys.ssm_norm)
            .expect("cached ssm norm buffer missing"),
        dt_bias: weight_cache
            .get(&cached_recurrent_keys.dt_bias)
            .expect("cached dt bias buffer missing"),
        ssm_a: weight_cache
            .get(&cached_recurrent_keys.ssm_a)
            .expect("cached ssm_a buffer missing"),
        wssm_out: weight_cache
            .get(&cached_recurrent_keys.wssm_out)
            .expect("cached wssm_out buffer missing"),
    };
    let cached_recurrent_dtypes = Qwen3_5NativeRecurrentDtypes {
        wqkv: cached_recurrent_keys.wqkv_dtype,
        wgate: cached_recurrent_keys.wgate_dtype,
        wbeta: cached_recurrent_keys.wbeta_dtype,
        walpha: cached_recurrent_keys.walpha_dtype,
        wssm_out: cached_recurrent_keys.wssm_out_dtype,
    };
    let cached_shared_expert = cached_moe_keys.shared_expert.map(|shared| {
        crate::backend::metal::SharedExpertCachedBuffers {
            gate: moe_weight_cache
                .get(&shared.gate)
                .expect("cached shared gate buffer missing"),
            up: moe_weight_cache
                .get(&shared.up)
                .expect("cached shared up buffer missing"),
            down: moe_weight_cache
                .get(&shared.down)
                .expect("cached shared down buffer missing"),
            gate_inp: shared.gate_inp.map(|gate_inp| {
                moe_weight_cache
                    .get(&gate_inp)
                    .expect("cached shared gate_inp buffer missing")
            }),
            gate_inp_dtype: shared.gate_inp_dtype,
            dtype: shared.dtype,
            inter_dim: shared.inter_dim,
            gate_inp_rows: shared.gate_inp_rows,
        }
    });

    metal_ops
        .device
        .execute_sync(|encoder| {
            let barrier = crate::model::shared::DecodeBarrierCtx::new(
                encoder,
                crate::model::execution_plan::DecodeBarrierPlan::Explicit,
            );
            metal_ops.with_qwen35_recurrent_slot_buffer_for_kv(
                &cached_kv,
                layer_idx,
                recurrent_slot,
                cached_conv_state_stride,
                cached_recurrent_state_stride,
                |slot_buffers| -> anyhow::Result<()> {
                    Qwen3_5Forward::encode_qwen35_native_recurrent_layer(
                        metal_ops,
                        encoder,
                        &barrier,
                        cached_recurrent_weights,
                        recurrent_scratch,
                        slot_buffers,
                        &norm_gpu,
                        &recurrent_out_gpu,
                        &proj_gpu,
                        cached_kv.conv_cache_len(),
                        dims,
                        dim,
                        cfg.rms_norm_eps,
                        cached_recurrent_dtypes,
                        metal_ops.dequant_dispatch_config(),
                    )?;
                    metal_ops.elementwise.encode_elementwise_add_batch(
                        encoder,
                        &cached_hidden_gpu,
                        &proj_gpu,
                        dim as u32,
                        1,
                    );
                    barrier.step(encoder);
                    metal_ops.encode_moe_ffn_gpu_resident_cached_with_scratch(
                        encoder,
                        moe_scratch,
                        &cached_hidden_gpu,
                        cached_ffn_norm_w,
                        moe_weight_cache
                            .get(&cached_moe_keys.router)
                            .expect("cached router buffer missing"),
                        cached_moe_keys.router_dtype,
                        moe_weight_cache
                            .get(&cached_moe_keys.gate)
                            .expect("cached gate expert buffer missing"),
                        cached_moe_keys.gate_dtype,
                        moe_weight_cache
                            .get(&cached_moe_keys.up)
                            .expect("cached up expert buffer missing"),
                        cached_moe_keys.up_dtype,
                        moe_weight_cache
                            .get(&cached_moe_keys.down)
                            .expect("cached down expert buffer missing"),
                        cached_moe_keys.down_dtype,
                        1,
                        cached_moe_keys.n_expert,
                        cached_moe_keys.n_expert_used,
                        dim,
                        cached_moe_keys.expert_inter_dim,
                        cached_moe_keys.gate_stride,
                        cached_moe_keys.up_stride,
                        cached_moe_keys.down_stride,
                        cfg.rms_norm_eps,
                        cached_shared_expert.as_ref(),
                        true,
                    )?;
                    barrier.step(encoder);
                    barrier.flush();
                    Ok(())
                },
            )
        })
        .unwrap();

    let cached_actual_hidden = unsafe { cached_hidden_gpu.as_slice::<f32>()[..dim].to_vec() };
    let cached_diff = max_abs_diff(&cached_actual_hidden, &expected_hidden_after_moe);
    let cached_scale = expected_hidden_after_moe
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        cached_diff / cached_scale < 5e-3,
        "real Qwen3.5-35B-A3B layer-1 cached recurrent+resident-MoE mismatch: rel_diff={}, max_diff={cached_diff}, actual[0..8]={:?}, expected[0..8]={:?}",
        cached_diff / cached_scale,
        &cached_actual_hidden[..8],
        &expected_hidden_after_moe[..8],
    );

    let router_name = format!("{prefix}.ffn_gate_inp.weight");
    let (router_raw, router_dtype) = weights.raw_with_dtype(&router_name).unwrap();
    let n_expert = cfg.n_expert.unwrap() as usize;
    let n_expert_used = cfg.n_expert_used.unwrap() as usize;
    let mut expected_router_logits = vec![0.0f32; n_expert];
    let mut actual_router_logits = vec![0.0f32; n_expert];
    cpu.dequant_matmul(
        router_raw,
        router_dtype,
        &expected_hidden_before_moe,
        &mut expected_router_logits,
        n_expert,
        1,
        dim,
    );
    cpu.dequant_matmul(
        router_raw,
        router_dtype,
        &actual_hidden_before_moe,
        &mut actual_router_logits,
        n_expert,
        1,
        dim,
    );
    let (expected_ids, expected_weights) =
        crate::model::moe_utils::top_k_softmax(&expected_router_logits, n_expert_used);
    let (actual_ids, actual_weights) =
        crate::model::moe_utils::top_k_softmax(&actual_router_logits, n_expert_used);
    assert_eq!(
        actual_ids, expected_ids,
        "real Qwen3.5-35B-A3B layer-1 recurrent router top-k ids diverged: actual={actual_ids:?} expected={expected_ids:?}"
    );
    assert!(
        max_abs_diff(&actual_weights, &expected_weights) < 1e-4,
        "real Qwen3.5-35B-A3B layer-1 recurrent router top-k weights diverged: actual={actual_weights:?} expected={expected_weights:?}"
    );

    let gate_name = format!("{prefix}.ffn_gate_exps.weight");
    let up_name = format!("{prefix}.ffn_up_exps.weight");
    let down_name = format!("{prefix}.ffn_down_exps.weight");
    let shared_gate_name = format!("{prefix}.ffn_gate_shexp.weight");
    let shared_up_name = format!("{prefix}.ffn_up_shexp.weight");
    let shared_down_name = format!("{prefix}.ffn_down_shexp.weight");
    let shared_gate_inp_name = format!("{prefix}.ffn_gate_inp_shexp.weight");

    let expert_inter_dim = Qwen3_5Forward::tensor_output_rows(&weights, &gate_name).unwrap();
    let (gate_raw, gate_dtype) = weights.raw_with_dtype(&gate_name).unwrap();
    let (up_raw, up_dtype) = weights.raw_with_dtype(&up_name).unwrap();
    let (down_raw, down_dtype) = weights.raw_with_dtype(&down_name).unwrap();
    let gate_stride =
        crate::model::moe_utils::expert_byte_stride(gate_dtype, expert_inter_dim * dim);
    let up_stride = crate::model::moe_utils::expert_byte_stride(up_dtype, expert_inter_dim * dim);
    let down_stride =
        crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);

    let hidden_after_recurrent_buf = ax_engine_metal::MetalBuffer::from_slice(
        metal_ops.device.device(),
        &actual_hidden_before_moe,
    )
    .unwrap();
    let ffn_norm_w = weights
        .f32_slice(&format!("{prefix}.post_attention_norm.weight"))
        .unwrap();
    let ffn_norm_w_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), ffn_norm_w).unwrap();
    let router_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), router_raw).unwrap();
    let gate_expert_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), gate_raw).unwrap();
    let up_expert_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), up_raw).unwrap();
    let down_expert_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), down_raw).unwrap();

    let shared_gate_buf = weights.has(&shared_gate_name).then(|| {
        let (raw, _) = weights.raw_with_dtype(&shared_gate_name).unwrap();
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), raw).unwrap()
    });
    let shared_up_buf = weights.has(&shared_up_name).then(|| {
        let (raw, _) = weights.raw_with_dtype(&shared_up_name).unwrap();
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), raw).unwrap()
    });
    let shared_down_buf = weights.has(&shared_down_name).then(|| {
        let (raw, _) = weights.raw_with_dtype(&shared_down_name).unwrap();
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), raw).unwrap()
    });
    let shared_gate_inp_buf = weights.has(&shared_gate_inp_name).then(|| {
        let (raw, _) = weights.raw_with_dtype(&shared_gate_inp_name).unwrap();
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), raw).unwrap()
    });
    let shared_expert = if let (Some(gate), Some(up), Some(down)) = (
        shared_gate_buf.as_ref(),
        shared_up_buf.as_ref(),
        shared_down_buf.as_ref(),
    ) {
        let (_, shared_dtype) = weights.raw_with_dtype(&shared_gate_name).unwrap();
        let gate_inp_dtype = if weights.has(&shared_gate_inp_name) {
            Some(weights.raw_with_dtype(&shared_gate_inp_name).unwrap().1)
        } else {
            None
        };
        Some(crate::backend::metal::SharedExpertCachedBuffers {
            gate,
            up,
            down,
            gate_inp: shared_gate_inp_buf.as_ref(),
            gate_inp_dtype,
            dtype: shared_dtype,
            inter_dim: Qwen3_5Forward::tensor_output_rows(&weights, &shared_gate_name).unwrap(),
            gate_inp_rows: if weights.has(&shared_gate_inp_name) {
                Qwen3_5Forward::tensor_output_rows(&weights, &shared_gate_inp_name).unwrap()
            } else {
                0
            },
        })
    } else {
        None
    };

    metal_ops.init_batch_scratches(&cfg, 1);
    metal_ops
        .moe_ffn_gpu_resident_cached(
            &hidden_after_recurrent_buf,
            &ffn_norm_w_buf,
            &router_buf,
            router_dtype,
            &gate_expert_buf,
            gate_dtype,
            &up_expert_buf,
            up_dtype,
            &down_expert_buf,
            down_dtype,
            1,
            n_expert,
            n_expert_used,
            dim,
            expert_inter_dim,
            gate_stride,
            up_stride,
            down_stride,
            cfg.rms_norm_eps,
            shared_expert.as_ref(),
        )
        .unwrap();

    let actual_hidden_after_moe =
        unsafe { hidden_after_recurrent_buf.as_slice::<f32>()[..dim].to_vec() };
    let moe_diff = max_abs_diff(&actual_hidden_after_moe, &expected_hidden_after_moe);
    let moe_scale = expected_hidden_after_moe
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        moe_diff / moe_scale < 5e-3,
        "real Qwen3.5-35B-A3B layer-1 resident MoE mismatch after native recurrent: rel_diff={}, max_diff={moe_diff}, actual[0..8]={:?}, expected[0..8]={:?}",
        moe_diff / moe_scale,
        &actual_hidden_after_moe[..8],
        &expected_hidden_after_moe[..8],
    );

    let combined_hidden_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), &hidden_after_layer0)
            .unwrap();
    let moe_scratch = metal_ops.moe_batch_scratch_view().unwrap();
    metal_ops
        .device
        .execute_sync(|encoder| {
            let barrier = crate::model::shared::DecodeBarrierCtx::new(
                encoder,
                crate::model::execution_plan::DecodeBarrierPlan::Explicit,
            );
            metal_ops.with_qwen35_recurrent_slot_buffer_for_kv(
                &combined_kv,
                layer_idx,
                recurrent_slot,
                conv_state_stride,
                recurrent_state_stride,
                |slot_buffers| -> anyhow::Result<()> {
                    Qwen3_5Forward::encode_qwen35_native_recurrent_layer(
                        metal_ops,
                        encoder,
                        &barrier,
                        recurrent_weights,
                        recurrent_scratch,
                        slot_buffers,
                        &norm_gpu,
                        &recurrent_out_gpu,
                        &proj_gpu,
                        combined_kv.conv_cache_len(),
                        dims,
                        dim,
                        cfg.rms_norm_eps,
                        recurrent_dtypes,
                        metal_ops.dequant_dispatch_config(),
                    )?;
                    metal_ops.elementwise.encode_elementwise_add_batch(
                        encoder,
                        &combined_hidden_buf,
                        &proj_gpu,
                        dim as u32,
                        1,
                    );
                    barrier.step(encoder);
                    barrier.pre_dispatch(&[&combined_hidden_buf], &[&combined_hidden_buf]);
                    metal_ops.encode_moe_ffn_gpu_resident_cached_with_scratch(
                        encoder,
                        moe_scratch,
                        &combined_hidden_buf,
                        &ffn_norm_w_buf,
                        &router_buf,
                        router_dtype,
                        &gate_expert_buf,
                        gate_dtype,
                        &up_expert_buf,
                        up_dtype,
                        &down_expert_buf,
                        down_dtype,
                        1,
                        n_expert,
                        n_expert_used,
                        dim,
                        expert_inter_dim,
                        gate_stride,
                        up_stride,
                        down_stride,
                        cfg.rms_norm_eps,
                        shared_expert.as_ref(),
                        true,
                    )?;
                    barrier.post_dispatch(&[&combined_hidden_buf], &[&combined_hidden_buf]);
                    barrier.step(encoder);
                    barrier.flush();
                    Ok(())
                },
            )
        })
        .unwrap();

    let combined_actual_hidden = unsafe { combined_hidden_buf.as_slice::<f32>()[..dim].to_vec() };
    let combined_diff = max_abs_diff(&combined_actual_hidden, &expected_hidden_after_moe);
    let combined_scale = expected_hidden_after_moe
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        combined_diff / combined_scale < 5e-3,
        "real Qwen3.5-35B-A3B layer-1 combined recurrent->resident-MoE same-CB mismatch: rel_diff={}, max_diff={combined_diff}, actual[0..8]={:?}, expected[0..8]={:?}",
        combined_diff / combined_scale,
        &combined_actual_hidden[..8],
        &expected_hidden_after_moe[..8],
    );

    let mut scratch_kv = make_kv();
    scratch_kv
        .enable_gpu_recurrent_state(&metal_ops.device)
        .unwrap();
    metal_ops.init_scratches(&cfg);
    let mut decode_scratch_guard = metal_ops.scratches();
    let decode_scratch = decode_scratch_guard.as_mut().unwrap();
    unsafe {
        decode_scratch.hidden.as_mut_slice::<f32>()[..dim].copy_from_slice(&hidden_after_layer0);
    }
    let attn_norm_w = weights
        .f32_slice(&format!("{prefix}.attn_norm.weight"))
        .unwrap();
    let attn_norm_w_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), attn_norm_w).unwrap();
    metal_ops
        .device
        .execute_sync(|encoder| {
            let barrier = crate::model::shared::DecodeBarrierCtx::new(
                encoder,
                crate::model::execution_plan::DecodeBarrierPlan::Explicit,
            );
            metal_ops.elementwise.encode_rms_norm_out(
                encoder,
                &decode_scratch.hidden,
                &attn_norm_w_buf,
                &decode_scratch.norm_buf,
                dim as u32,
                cfg.rms_norm_eps,
            );
            barrier.step(encoder);
            metal_ops.with_qwen35_recurrent_slot_buffer_for_kv(
                &scratch_kv,
                layer_idx,
                recurrent_slot,
                conv_state_stride,
                recurrent_state_stride,
                |slot_buffers| -> anyhow::Result<()> {
                    Qwen3_5Forward::encode_qwen35_native_recurrent_layer(
                        metal_ops,
                        encoder,
                        &barrier,
                        recurrent_weights,
                        recurrent_scratch,
                        slot_buffers,
                        &decode_scratch.norm_buf,
                        &decode_scratch.up_buf,
                        &decode_scratch.proj_buf,
                        scratch_kv.conv_cache_len(),
                        dims,
                        dim,
                        cfg.rms_norm_eps,
                        recurrent_dtypes,
                        metal_ops.dequant_dispatch_config(),
                    )?;
                    metal_ops.elementwise.encode_elementwise_add_batch(
                        encoder,
                        &decode_scratch.hidden,
                        &decode_scratch.proj_buf,
                        dim as u32,
                        1,
                    );
                    barrier.step(encoder);
                    metal_ops.encode_moe_ffn_gpu_resident_cached_with_scratch(
                        encoder,
                        moe_scratch,
                        &decode_scratch.hidden,
                        &ffn_norm_w_buf,
                        &router_buf,
                        router_dtype,
                        &gate_expert_buf,
                        gate_dtype,
                        &up_expert_buf,
                        up_dtype,
                        &down_expert_buf,
                        down_dtype,
                        1,
                        n_expert,
                        n_expert_used,
                        dim,
                        expert_inter_dim,
                        gate_stride,
                        up_stride,
                        down_stride,
                        cfg.rms_norm_eps,
                        shared_expert.as_ref(),
                        true,
                    )?;
                    barrier.step(encoder);
                    barrier.flush();
                    Ok(())
                },
            )
        })
        .unwrap();

    let decode_scratch_actual = unsafe { decode_scratch.hidden.as_slice::<f32>()[..dim].to_vec() };
    let decode_scratch_diff = max_abs_diff(&decode_scratch_actual, &expected_hidden_after_moe);
    let decode_scratch_scale = expected_hidden_after_moe
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        decode_scratch_diff / decode_scratch_scale < 5e-3,
        "real Qwen3.5-35B-A3B layer-1 decode-scratch recurrent->resident-MoE mismatch: rel_diff={}, max_diff={decode_scratch_diff}, actual[0..8]={:?}, expected[0..8]={:?}",
        decode_scratch_diff / decode_scratch_scale,
        &decode_scratch_actual[..8],
        &expected_hidden_after_moe[..8],
    );
}

#[test]
fn test_real_qwen35_35b_a3b_layer1_resident_moe_fixture_matches_cpu() {
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&model.header).unwrap();
    let weights = WeightStore::new(&model);
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&model.header).unwrap();
    let fixture = build_real_qwen35_35b_a3b_layer1_resident_moe_prompt_state(
        &cfg, &weights, &tokenizer, metal_ops,
    );

    metal_ops.init_batch_scratches(&cfg, 1);
    let (_, actual_hidden) = run_real_qwen35_layer1_resident_moe_case(metal_ops, &fixture, &[], 1);
    let diff = max_abs_diff(&actual_hidden, &fixture.expected_hidden_after_moe);
    let scale = fixture
        .expected_hidden_after_moe
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 5e-3,
        "real Qwen3.5-35B-A3B layer-1 resident MoE fixture mismatch: rel_diff={}, max_diff={diff}, actual[0..8]={:?}, expected[0..8]={:?}",
        diff / scale,
        &actual_hidden[..8],
        &fixture.expected_hidden_after_moe[..8],
    );
}

#[test]
#[ignore = "diagnostic only; measures real layer-1 resident MoE routed/shared split"]
fn test_real_qwen35_35b_a3b_layer1_resident_moe_timing_split() {
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!(
            "skipping resident MoE timing split because {} is missing",
            path.display()
        );
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&model.header).unwrap();
    let weights = WeightStore::new(&model);
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&model.header).unwrap();
    let fixture = build_real_qwen35_35b_a3b_layer1_resident_moe_prompt_state(
        &cfg, &weights, &tokenizer, metal_ops,
    );
    let iterations = 64usize;

    metal_ops.init_batch_scratches(&cfg, 1);
    let (full_elapsed, full_hidden) =
        run_real_qwen35_layer1_resident_moe_case(metal_ops, &fixture, &[], iterations);
    let diff = max_abs_diff(&full_hidden, &fixture.expected_hidden_after_moe);
    let scale = fixture
        .expected_hidden_after_moe
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 5e-3,
        "real Qwen3.5-35B-A3B layer-1 resident MoE full timing case mismatch: rel_diff={}, max_diff={diff}",
        diff / scale,
    );

    let (routed_elapsed, _) = run_real_qwen35_layer1_resident_moe_case(
        metal_ops,
        &fixture,
        &[("AX_QWEN35_PROFILE_SKIP_SHARED_EXPERT", Some("1"))],
        iterations,
    );
    let (shared_elapsed, _) = run_real_qwen35_layer1_resident_moe_case(
        metal_ops,
        &fixture,
        &[("AX_QWEN35_PROFILE_SKIP_ROUTED_EXPERT", Some("1"))],
        iterations,
    );
    let (base_elapsed, _) = run_real_qwen35_layer1_resident_moe_case(
        metal_ops,
        &fixture,
        &[
            ("AX_QWEN35_PROFILE_SKIP_ROUTED_EXPERT", Some("1")),
            ("AX_QWEN35_PROFILE_SKIP_SHARED_EXPERT", Some("1")),
        ],
        iterations,
    );
    let (routed_pair_off_elapsed, _) = run_real_qwen35_layer1_resident_moe_case(
        metal_ops,
        &fixture,
        &[
            ("AX_QWEN35_PROFILE_SKIP_SHARED_EXPERT", Some("1")),
            ("AX_QWEN35_SELECTED_EXPERT_PAIR", Some("0")),
        ],
        iterations,
    );
    let (routed_weighted_down_elapsed, _) = run_real_qwen35_layer1_resident_moe_case(
        metal_ops,
        &fixture,
        &[
            ("AX_QWEN35_PROFILE_SKIP_SHARED_EXPERT", Some("1")),
            ("AX_QWEN35_SELECTED_WEIGHTED_DOWN", Some("1")),
        ],
        iterations,
    );
    let (routed_gate_up_elapsed, _) = run_real_qwen35_layer1_resident_moe_case(
        metal_ops,
        &fixture,
        &[
            ("AX_QWEN35_PROFILE_SKIP_SHARED_EXPERT", Some("1")),
            ("AX_QWEN35_PROFILE_SKIP_SELECTED_DOWN", Some("1")),
        ],
        iterations,
    );
    let (shared_ffn_elapsed, _) = run_real_qwen35_layer1_resident_moe_case(
        metal_ops,
        &fixture,
        &[
            ("AX_QWEN35_PROFILE_SKIP_ROUTED_EXPERT", Some("1")),
            ("AX_QWEN35_PROFILE_SKIP_SHARED_GATE_INP", Some("1")),
        ],
        iterations,
    );

    let full_ms = full_elapsed.as_secs_f64() * 1_000.0 / iterations as f64;
    let routed_ms = routed_elapsed.as_secs_f64() * 1_000.0 / iterations as f64;
    let shared_ms = shared_elapsed.as_secs_f64() * 1_000.0 / iterations as f64;
    let base_ms = base_elapsed.as_secs_f64() * 1_000.0 / iterations as f64;
    let routed_pair_off_ms = routed_pair_off_elapsed.as_secs_f64() * 1_000.0 / iterations as f64;
    let routed_weighted_down_ms =
        routed_weighted_down_elapsed.as_secs_f64() * 1_000.0 / iterations as f64;
    let routed_gate_up_ms = routed_gate_up_elapsed.as_secs_f64() * 1_000.0 / iterations as f64;
    let shared_ffn_ms = shared_ffn_elapsed.as_secs_f64() * 1_000.0 / iterations as f64;
    eprintln!(
        "Qwen3.5-35B-A3B layer1 resident MoE split: full={full_ms:.3} ms/iter base={base_ms:.3} ms/iter routed_only={routed_ms:.3} ms/iter shared_only={shared_ms:.3} ms/iter routed_delta={:.3} ms shared_delta={:.3} ms routed_pair_off={routed_pair_off_ms:.3} ms routed_weighted_down={routed_weighted_down_ms:.3} ms routed_gate_up={routed_gate_up_ms:.3} ms shared_ffn={shared_ffn_ms:.3} ms",
        (routed_ms - base_ms).max(0.0),
        (shared_ms - base_ms).max(0.0),
    );
    eprintln!(
        "Qwen3.5-35B-A3B layer1 resident MoE shares: routed={:.1}% shared={:.1}% pair_gain={:.3} ms weighted_down_gain={:.3} ms routed_down_delta={:.3} ms shared_gate_inp_delta={:.3} ms",
        (routed_ms / full_ms) * 100.0,
        (shared_ms / full_ms) * 100.0,
        routed_pair_off_ms - routed_ms,
        routed_ms - routed_weighted_down_ms,
        (routed_ms - routed_gate_up_ms).max(0.0),
        (shared_ms - shared_ffn_ms).max(0.0),
    );
}

#[test]
#[ignore = "diagnostic only; prints real layer-1 Qwen3.5-35B-A3B expert dtypes"]
fn test_real_qwen35_35b_a3b_layer1_expert_dtype_diagnostic() {
    let path = workspace_model_path("Qwen3.5-35B-A3B-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!(
            "skipping dtype diagnostic because {} is missing",
            path.display()
        );
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let weights = WeightStore::new(&model);
    let layer_idx = 1usize;
    let prefix = format!("blk.{layer_idx}");

    let gate_name = format!("{prefix}.ffn_gate_exps.weight");
    let up_name = format!("{prefix}.ffn_up_exps.weight");
    let down_name = format!("{prefix}.ffn_down_exps.weight");
    let shared_gate_name = format!("{prefix}.ffn_gate_shexp.weight");
    let shared_up_name = format!("{prefix}.ffn_up_shexp.weight");
    let shared_down_name = format!("{prefix}.ffn_down_shexp.weight");
    let shared_gate_inp_name = format!("{prefix}.ffn_gate_inp_shexp.weight");

    let (_, gate_dtype) = weights.raw_with_dtype(&gate_name).unwrap();
    let (_, up_dtype) = weights.raw_with_dtype(&up_name).unwrap();
    let (_, down_dtype) = weights.raw_with_dtype(&down_name).unwrap();
    let shared_gate_dtype = weights.raw_with_dtype(&shared_gate_name).unwrap().1;
    let shared_up_dtype = weights.raw_with_dtype(&shared_up_name).unwrap().1;
    let shared_down_dtype = weights.raw_with_dtype(&shared_down_name).unwrap().1;
    let shared_gate_inp_dtype = weights.raw_with_dtype(&shared_gate_inp_name).unwrap().1;

    eprintln!(
        "Qwen3.5-35B-A3B layer1 expert dtypes: gate={gate_dtype:?} up={up_dtype:?} down={down_dtype:?} shared_gate={shared_gate_dtype:?} shared_up={shared_up_dtype:?} shared_down={shared_down_dtype:?} shared_gate_inp={shared_gate_inp_dtype:?}"
    );
}
