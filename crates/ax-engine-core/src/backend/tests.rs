use super::*;
use crate::backend::cpu::CpuBackend;
use crate::model::ModelConfig;
use crate::model::config::{GateActivation, RopeScaling};

struct EnvVarRestore {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl Drop for EnvVarRestore {
    fn drop(&mut self) {
        match &self.previous {
            Some(prev) => unsafe { std::env::set_var(self.key, prev) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    crate::test_env_lock()
}

fn test_model_config() -> ModelConfig {
    ModelConfig {
        architecture: "gemma3".into(),
        n_layers: 32,
        n_heads: 32,
        n_kv_heads: 8,
        embedding_dim: 4096,
        head_dim: 128,
        intermediate_dim: 11008,
        context_length: 4096,
        vocab_size: 32000,
        rms_norm_eps: 1e-5,
        rope_freq_base: 10000.0,
        has_qkv_bias: false,
        sliding_window_size: None,
        sliding_window_pattern: None,
        gate_activation: GateActivation::SiLU,
        tie_word_embeddings: false,
        logit_scale: None,
        rope_scaling: RopeScaling::None,
        embed_scale: false,
        rope_freq_base_local: None,
        n_expert: None,
        n_expert_used: None,
        qwen35_full_attention_interval: None,
        qwen35_ssm_conv_kernel: None,
        qwen35_ssm_inner_size: None,
        qwen35_ssm_state_size: None,
        qwen35_ssm_time_step_rank: None,
        qwen35_ssm_group_count: None,
        gemma4_head_dim_swa: None,
        gemma4_head_dim_global: None,
        gemma4_n_kv_heads_swa: None,
        gemma4_n_kv_heads_global: None,
        gemma4_rope_dim_swa: None,
        gemma4_rope_dim_global: None,
        final_logit_softcapping: None,
        expert_intermediate_dim: None,
    }
}

#[test]
fn test_backend_config_default() {
    assert_eq!(BackendConfig::default(), BackendConfig::Hybrid);
}

#[test]
fn test_create_cpu_backend() {
    let backend = create_backend(BackendConfig::Cpu).unwrap();
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    let mut c = [0.0f32; 4];
    backend.matmul(&a, &b, &mut c, 2, 2, 2);
    assert!((c[0] - 19.0).abs() < 1e-4);
}

#[test]
fn test_hybrid_cpu_decode_does_not_use_gpu_decode() {
    let backend = create_backend(BackendConfig::HybridCpuDecode).unwrap();
    assert!(
        !backend.use_gpu_decode(),
        "HybridCpuDecode must return use_gpu_decode()=false"
    );
}

#[test]
fn test_hybrid_uses_gpu_decode() {
    let backend = create_backend(BackendConfig::Hybrid).unwrap();
    assert!(
        backend.use_gpu_decode(),
        "Hybrid must return use_gpu_decode()=true"
    );
}

#[test]
fn test_create_model_kv_uses_cpu_kv_for_hybrid_cpu_decode() {
    let backend = create_backend(BackendConfig::HybridCpuDecode).unwrap();
    let kv = create_model_kv(backend.as_ref(), &test_model_config());
    assert!(matches!(kv, crate::kv::ModelKv::Cpu(_)));
}

#[test]
fn test_resolve_backend_config_from_env_defaults_to_hybrid() {
    let _guard = env_lock();
    let _restore_cpu = EnvVarRestore {
        key: "AX_CPU_ONLY",
        previous: std::env::var_os("AX_CPU_ONLY"),
    };
    let _restore_decode = EnvVarRestore {
        key: "AX_HYBRID_DECODE",
        previous: std::env::var_os("AX_HYBRID_DECODE"),
    };
    unsafe {
        std::env::remove_var("AX_CPU_ONLY");
        std::env::remove_var("AX_HYBRID_DECODE");
    }

    assert_eq!(resolve_backend_config_from_env(), BackendConfig::Hybrid);
}

#[test]
fn test_resolve_backend_config_from_env_honors_cpu_only() {
    let _guard = env_lock();
    let _restore_cpu = EnvVarRestore {
        key: "AX_CPU_ONLY",
        previous: std::env::var_os("AX_CPU_ONLY"),
    };
    unsafe {
        std::env::set_var("AX_CPU_ONLY", "1");
    }

    assert_eq!(resolve_backend_config_from_env(), BackendConfig::Cpu);
}

#[test]
fn test_resolve_backend_config_from_env_honors_hybrid_cpu_decode() {
    let _guard = env_lock();
    let _restore_decode = EnvVarRestore {
        key: "AX_HYBRID_DECODE",
        previous: std::env::var_os("AX_HYBRID_DECODE"),
    };
    unsafe {
        std::env::set_var("AX_HYBRID_DECODE", "cpu");
    }

    assert_eq!(
        resolve_backend_config_from_env(),
        BackendConfig::HybridCpuDecode
    );
}

#[test]
fn test_resolve_backend_config_from_env_cpu_only_wins() {
    let _guard = env_lock();
    let _restore_cpu = EnvVarRestore {
        key: "AX_CPU_ONLY",
        previous: std::env::var_os("AX_CPU_ONLY"),
    };
    let _restore_decode = EnvVarRestore {
        key: "AX_HYBRID_DECODE",
        previous: std::env::var_os("AX_HYBRID_DECODE"),
    };
    unsafe {
        std::env::set_var("AX_CPU_ONLY", "1");
        std::env::set_var("AX_HYBRID_DECODE", "cpu");
    }

    assert_eq!(resolve_backend_config_from_env(), BackendConfig::Cpu);
}

#[test]
fn test_qwen35_recurrent_state_batch_requires_matching_lengths() {
    let slot_indices = [0usize, 1usize];
    let mut conv_state = vec![0.0f32; 8];
    let mut recurrent_state = vec![0.0f32; 8];
    let batch = Qwen3_5RecurrentStateBatch::new(
        0,
        &slot_indices,
        &mut conv_state,
        &mut recurrent_state,
        4,
        4,
    );

    assert_eq!(batch.slot_count(), 2);
    assert_eq!(batch.layer_idx(), 0);
    assert_eq!(batch.slot_indices(), &slot_indices);
    assert_eq!(batch.conv_state_stride(), 4);
    assert_eq!(batch.recurrent_state_stride(), 4);
}

#[test]
#[should_panic(expected = "qwen35 recurrent state batch must not contain duplicate slots")]
fn test_qwen35_recurrent_state_batch_rejects_duplicate_slots() {
    let slot_indices = [0usize, 0usize];
    let mut conv_state = vec![0.0f32; 8];
    let mut recurrent_state = vec![0.0f32; 8];
    let _ = Qwen3_5RecurrentStateBatch::new(
        0,
        &slot_indices,
        &mut conv_state,
        &mut recurrent_state,
        4,
        4,
    );
}

#[test]
#[should_panic(
    expected = "qwen35 recurrent slot batch slot 1 has seqlen_offset 1 != shared attention seq_len 0"
)]
fn test_qwen35_recurrent_sequence_for_kv_rejects_misaligned_slot_batch() {
    let backend = cpu::CpuBackend;
    let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
        conv_cache_len: 2,
        conv_dim: 6,
        group_count: 1,
        state_size: 2,
        time_step_rank: 1,
        rms_norm_eps: 1e-5,
    };
    let mut kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let slot1 = kv.allocate_recurrent_slot();
    kv.set_recurrent_seqlen_offset(slot1, 1);
    let slot_indices = [0usize, slot1];
    let tokens_per_slot = 1usize;
    let total_tokens = slot_indices.len() * tokens_per_slot;
    let qkv = vec![0.0f32; total_tokens * cfg.conv_dim];
    let mut alpha = vec![0.0f32; total_tokens * cfg.time_step_rank];
    let mut beta = vec![0.0f32; total_tokens * cfg.time_step_rank];
    let dt_bias = vec![0.0f32; cfg.time_step_rank];
    let a = vec![0.0f32; cfg.time_step_rank];
    let kernel = vec![0.0f32; (cfg.conv_cache_len + 1) * cfg.conv_dim];
    let mut out = vec![0.0f32; total_tokens * cfg.value_dim()];

    backend.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut beta,
        &mut alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut kv,
        0,
        &slot_indices,
        &mut out,
        tokens_per_slot,
        cfg,
    );
}

#[test]
fn test_qwen35_recurrent_sequence_via_backend_matches_per_slot_cpu_execution() {
    let backend = cpu::CpuBackend;
    let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
        conv_cache_len: 2,
        conv_dim: 6,
        group_count: 1,
        state_size: 2,
        time_step_rank: 1,
        rms_norm_eps: 1e-5,
    };
    let slot_indices = [3usize, 7usize];
    let tokens_per_slot = 2;
    let total_tokens = slot_indices.len() * tokens_per_slot;
    let conv_state_stride = cfg.conv_cache_len * cfg.conv_dim;
    let recurrent_state_stride = cfg.time_step_rank * cfg.state_size * cfg.state_size;
    let qkv = vec![
        0.5, 0.1, -0.2, 0.7, 0.3, 0.4, //
        -0.1, 0.2, 0.6, 0.8, -0.4, 0.9, //
        0.2, -0.3, 0.1, -0.5, 0.6, 0.7, //
        0.8, 0.4, -0.6, 0.2, 0.1, -0.2,
    ];
    let alpha_input = vec![0.3, 0.4, 0.2, 0.1];
    let beta_input = vec![0.1, 0.2, 0.3, 0.4];
    let dt_bias = vec![0.05];
    let a = vec![0.7];
    let kernel = vec![
        0.1, 0.2, 0.1, 0.2, 0.1, 0.2, //
        0.2, 0.1, 0.2, 0.1, 0.2, 0.1, //
        0.3, 0.4, 0.3, 0.4, 0.3, 0.4,
    ];
    let mut conv_state = vec![0.0f32; slot_indices.len() * conv_state_stride];
    let mut recurrent_state = vec![0.0f32; slot_indices.len() * recurrent_state_stride];
    let mut alpha = alpha_input.clone();
    let mut beta = beta_input.clone();
    let mut out = vec![0.0f32; total_tokens * cfg.value_dim()];

    {
        let mut state_batch = Qwen3_5RecurrentStateBatch::new(
            0,
            &slot_indices,
            &mut conv_state,
            &mut recurrent_state,
            conv_state_stride,
            recurrent_state_stride,
        );
        backend.qwen35_recurrent_sequence(
            &qkv,
            &mut beta,
            &mut alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut state_batch,
            &mut out,
            tokens_per_slot,
            cfg,
        );
    }

    for batch_idx in 0..slot_indices.len() {
        let token_start = batch_idx * tokens_per_slot;
        let token_end = token_start + tokens_per_slot;
        let qkv_start = token_start * cfg.conv_dim;
        let qkv_end = token_end * cfg.conv_dim;
        let gate_start = token_start * cfg.time_step_rank;
        let gate_end = token_end * cfg.time_step_rank;
        let out_start = token_start * cfg.value_dim();
        let out_end = token_end * cfg.value_dim();

        let mut expected_alpha = alpha_input[gate_start..gate_end].to_vec();
        let mut expected_beta = beta_input[gate_start..gate_end].to_vec();
        let mut expected_conv_state = vec![0.0f32; conv_state_stride];
        let mut expected_recurrent_state = vec![0.0f32; recurrent_state_stride];
        let mut expected_out = vec![0.0f32; tokens_per_slot * cfg.value_dim()];

        crate::compute::gdn::qwen35_recurrent_sequence(
            &qkv[qkv_start..qkv_end],
            &mut expected_beta,
            &mut expected_alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut expected_conv_state,
            &mut expected_recurrent_state,
            &mut expected_out,
            tokens_per_slot,
            cfg,
        );

        for (actual, expected) in alpha[gate_start..gate_end]
            .iter()
            .zip(expected_alpha.iter())
        {
            assert!((actual - expected).abs() < 1e-5);
        }
        for (actual, expected) in beta[gate_start..gate_end].iter().zip(expected_beta.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
        for (actual, expected) in out[out_start..out_end].iter().zip(expected_out.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
        let conv_start = batch_idx * conv_state_stride;
        let conv_end = conv_start + conv_state_stride;
        for (actual, expected) in conv_state[conv_start..conv_end]
            .iter()
            .zip(expected_conv_state.iter())
        {
            assert!((actual - expected).abs() < 1e-5);
        }
        let state_start = batch_idx * recurrent_state_stride;
        let state_end = state_start + recurrent_state_stride;
        for (actual, expected) in recurrent_state[state_start..state_end]
            .iter()
            .zip(expected_recurrent_state.iter())
        {
            assert!((actual - expected).abs() < 1e-5);
        }
    }
}

#[test]
fn test_qwen35_recurrent_sequence_for_kv_matches_gathered_state_batch_path() {
    let backend = cpu::CpuBackend;
    let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
        conv_cache_len: 3,
        conv_dim: 16,
        group_count: 2,
        state_size: 2,
        time_step_rank: 4,
        rms_norm_eps: 1e-5,
    };
    let layer_idx = 0usize;
    let tokens_per_slot = 2;
    let mut actual_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let mut expected_kv = crate::kv::Qwen3_5Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
    let slot1 = actual_kv.allocate_recurrent_slot();
    let expected_slot1 = expected_kv.allocate_recurrent_slot();
    assert_eq!(slot1, expected_slot1);
    let slot_indices = [0usize, slot1];
    let total_tokens = slot_indices.len() * tokens_per_slot;
    let conv_state_stride = actual_kv.conv_cache_len() * actual_kv.conv_dim();
    let recurrent_state_stride = actual_kv.recurrent_state_len();

    for (slot_pos, &slot_idx) in slot_indices.iter().enumerate() {
        actual_kv
            .conv_state_for_slot_mut(slot_idx, layer_idx)
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = (slot_pos as f32 + 1.0) * 0.1 + i as f32 * 0.001);
        actual_kv
            .recurrent_state_for_slot_mut(slot_idx, layer_idx)
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = (slot_pos as f32 + 1.0) * 0.2 + i as f32 * 0.002);
        expected_kv
            .conv_state_for_slot_mut(slot_idx, layer_idx)
            .copy_from_slice(actual_kv.conv_state_for_slot(slot_idx, layer_idx));
        expected_kv
            .recurrent_state_for_slot_mut(slot_idx, layer_idx)
            .copy_from_slice(actual_kv.recurrent_state_for_slot(slot_idx, layer_idx));
    }

    let qkv: Vec<f32> = (0..total_tokens * cfg.conv_dim)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.02)
        .collect();
    let alpha_input: Vec<f32> = (0..total_tokens * cfg.time_step_rank)
        .map(|i| ((i % 9) as f32 - 4.0) * 0.05)
        .collect();
    let beta_input: Vec<f32> = (0..total_tokens * cfg.time_step_rank)
        .map(|i| 0.1 + (i % 11) as f32 * 0.02)
        .collect();
    let dt_bias = vec![0.03, 0.04, 0.05, 0.06];
    let a = vec![0.2, 0.25, 0.3, 0.35];
    let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();

    let mut expected_conv_state = vec![0.0f32; slot_indices.len() * conv_state_stride];
    let mut expected_recurrent_state = vec![0.0f32; slot_indices.len() * recurrent_state_stride];
    expected_kv.gather_recurrent_state_batch(
        &slot_indices,
        layer_idx,
        &mut expected_conv_state,
        &mut expected_recurrent_state,
    );
    let mut expected_alpha = alpha_input.clone();
    let mut expected_beta = beta_input.clone();
    let mut expected_out = vec![0.0f32; total_tokens * cfg.value_dim()];
    {
        let mut state_batch = Qwen3_5RecurrentStateBatch::new(
            layer_idx,
            &slot_indices,
            &mut expected_conv_state,
            &mut expected_recurrent_state,
            conv_state_stride,
            recurrent_state_stride,
        );
        backend.qwen35_recurrent_sequence(
            &qkv,
            &mut expected_beta,
            &mut expected_alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut state_batch,
            &mut expected_out,
            tokens_per_slot,
            cfg,
        );
    }
    expected_kv.scatter_recurrent_state_batch(
        &slot_indices,
        layer_idx,
        &expected_conv_state,
        &expected_recurrent_state,
    );

    let mut actual_alpha = alpha_input.clone();
    let mut actual_beta = beta_input.clone();
    let mut actual_out = vec![0.0f32; total_tokens * cfg.value_dim()];
    backend.qwen35_recurrent_sequence_for_kv(
        &qkv,
        &mut actual_beta,
        &mut actual_alpha,
        &dt_bias,
        &a,
        &kernel,
        &mut actual_kv,
        layer_idx,
        &slot_indices,
        &mut actual_out,
        tokens_per_slot,
        cfg,
    );

    for (actual, expected) in actual_alpha.iter().zip(expected_alpha.iter()) {
        assert!((actual - expected).abs() < 1e-5);
    }
    for (actual, expected) in actual_beta.iter().zip(expected_beta.iter()) {
        assert!((actual - expected).abs() < 1e-5);
    }
    for (actual, expected) in actual_out.iter().zip(expected_out.iter()) {
        assert!((actual - expected).abs() < 1e-5);
    }
    for &slot_idx in &slot_indices {
        for (actual, expected) in actual_kv
            .conv_state_for_slot(slot_idx, layer_idx)
            .iter()
            .zip(expected_kv.conv_state_for_slot(slot_idx, layer_idx).iter())
        {
            assert!((actual - expected).abs() < 1e-5);
        }
        for (actual, expected) in actual_kv
            .recurrent_state_for_slot(slot_idx, layer_idx)
            .iter()
            .zip(
                expected_kv
                    .recurrent_state_for_slot(slot_idx, layer_idx)
                    .iter(),
            )
        {
            assert!((actual - expected).abs() < 1e-5);
        }
    }
}

#[test]
fn test_dequant_matmul_token_major_matches_f32_reference() {
    let backend = CpuBackend;
    let n_tokens = 2usize;
    let in_dim = 3usize;
    let out_dim = 2usize;
    let weights = [
        1.0f32, 2.0, 3.0, //
        -1.0, 0.5, 4.0,
    ];
    let input = [
        0.5f32, -1.0, 2.0, //
        3.0, 0.25, -2.0,
    ];
    let mut weight_bytes = Vec::with_capacity(weights.len() * std::mem::size_of::<f32>());
    for value in weights {
        weight_bytes.extend_from_slice(&value.to_le_bytes());
    }

    let mut actual = vec![0.0f32; n_tokens * out_dim];
    backend.dequant_matmul_token_major(
        &weight_bytes,
        GgmlType::F32,
        &input,
        &mut actual,
        n_tokens,
        out_dim,
        in_dim,
    );

    let expected = [
        1.0 * 0.5 - 2.0 * 1.0 + 3.0 * 2.0,
        -(1.0 * 0.5) - 0.5 * 1.0 + 4.0 * 2.0,
        1.0 * 3.0 + 2.0 * 0.25 - 3.0 * 2.0,
        -(1.0 * 3.0) + 0.5 * 0.25 - 4.0 * 2.0,
    ];

    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "token-major matmul mismatch at {idx}: actual={actual}, expected={expected}"
        );
    }
}
