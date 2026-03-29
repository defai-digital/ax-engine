use super::*;

use crate::backend::Backend;
use crate::backend::metal::CachedLayerKeys;
use crate::backend::metal::MetalBackend;
use crate::gguf::tensor::GgmlType;
use crate::kv::GpuKv;
use crate::model::config::GateActivation;
use crate::model::config::ModelConfig;

fn qwen35_test_config() -> ModelConfig {
    ModelConfig {
        architecture: "qwen35".into(),
        n_layers: 4,
        n_heads: 2,
        n_kv_heads: 1,
        embedding_dim: 8,
        head_dim: 4,
        intermediate_dim: 16,
        context_length: 32,
        vocab_size: 32,
        rms_norm_eps: 1e-6,
        rope_freq_base: 10000.0,
        has_qkv_bias: false,
        sliding_window_size: None,
        sliding_window_pattern: None,
        gate_activation: GateActivation::SiLU,
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
        qwen35_ssm_inner_size: Some(8),
        qwen35_ssm_state_size: Some(2),
        qwen35_ssm_time_step_rank: Some(4),
        qwen35_ssm_group_count: Some(1),
    }
}

#[test]
fn test_build_qwen35_prefill_route_plan_local_marks_full_attention_and_recurrent_layers() {
    let cfg = qwen35_test_config();
    let plan = build_qwen35_prefill_route_plan(&cfg, 0, 3, false, None);

    assert_eq!(plan.batch_position, 0);
    assert_eq!(plan.n_tokens, 3);
    assert_eq!(plan.layers.len(), cfg.n_layers as usize);
    assert!(matches!(
        plan.layers[0].kind,
        Qwen35PrefillLayerRouteKind::RecurrentGdn {
            force_backend_state_batch: false,
            has_projection_phase: true,
            has_runtime_phase: true,
            has_tail_graph_ir_schedule: _
        }
    ));
    assert!(matches!(
        plan.layers[3].kind,
        Qwen35PrefillLayerRouteKind::FullAttention {
            uses_cached_attention: false,
            has_graph_ir_schedule: _
        }
    ));
}

#[test]
fn test_build_qwen35_prefill_route_plan_cached_attention_marks_full_attention_cached() {
    let cfg = qwen35_test_config();
    let plan = build_qwen35_prefill_route_plan(&cfg, 5, 2, true, None);

    assert!(matches!(
        plan.layers[3].kind,
        Qwen35PrefillLayerRouteKind::FullAttention {
            uses_cached_attention: true,
            has_graph_ir_schedule: true,
        }
    ));
}

#[test]
fn test_build_qwen35_prefill_schedule_summary_counts_routes() {
    let cfg = qwen35_test_config();
    let schedule = build_qwen35_prefill_schedule(&cfg, 5, 2, true, None);
    let summary = summarize_qwen35_prefill_schedule(&schedule);

    assert_eq!(summary.full_attention_layers, 1);
    assert_eq!(summary.recurrent_layers, 3);
    assert_eq!(summary.backend_state_batch_layers, 0);
    assert_eq!(summary.full_attention_graph_ir_layers, 1);
    assert_eq!(summary.recurrent_tail_graph_ir_layers, 3);
    assert_eq!(summary.cached_attention_layers, 1);
}

#[test]
fn test_build_qwen35_prefill_schedule_emits_full_flow_phases() {
    let cfg = qwen35_test_config();
    let schedule = build_qwen35_prefill_schedule(&cfg, 5, 2, true, None);

    match &schedule.layers[0].kind {
        Qwen35PrefillLayerScheduleKind::RecurrentGdn { phases, .. } => {
            assert_eq!(phases.len(), 3);
            assert_eq!(phases[0].kind, Qwen35PrefillPhaseKind::RecurrentProjection);
            assert_eq!(phases[1].kind, Qwen35PrefillPhaseKind::RecurrentRuntime);
            assert_eq!(phases[2].kind, Qwen35PrefillPhaseKind::RecurrentTail);
            assert!(phases[2].uses_graph_ir);
        }
        other => panic!("expected recurrent schedule, got {other:?}"),
    }

    match &schedule.layers[3].kind {
        Qwen35PrefillLayerScheduleKind::FullAttention {
            uses_cached_attention,
            phases,
        } => {
            assert!(*uses_cached_attention);
            assert_eq!(phases.len(), 1);
            assert_eq!(phases[0].kind, Qwen35PrefillPhaseKind::FullAttention);
            assert!(phases[0].uses_graph_ir);
        }
        other => panic!("expected full-attention schedule, got {other:?}"),
    }
}

#[test]
fn test_build_qwen35_prefill_schedule_flattens_execution_steps() {
    let cfg = qwen35_test_config();
    let schedule = build_qwen35_prefill_schedule(&cfg, 5, 2, true, None);

    assert_eq!(schedule.steps.len(), 10);
    assert_eq!(schedule.steps[0].layer, 0);
    assert!(matches!(
        schedule.steps[0].kind,
        Qwen35PrefillExecutionStepKind::RecurrentProjection {
            force_backend_state_batch: false
        }
    ));
    assert!(matches!(
        schedule.steps[1].kind,
        Qwen35PrefillExecutionStepKind::RecurrentRuntime {
            force_backend_state_batch: false
        }
    ));
    assert!(matches!(
        schedule.steps[2].kind,
        Qwen35PrefillExecutionStepKind::RecurrentTail {
            force_backend_state_batch: false,
            uses_graph_ir: true
        }
    ));
    assert_eq!(schedule.steps[9].layer, 3);
    assert!(matches!(
        schedule.steps[9].kind,
        Qwen35PrefillExecutionStepKind::FullAttention {
            uses_cached_attention: true,
            uses_graph_ir: true,
        }
    ));
}

#[test]
fn test_build_qwen35_prefill_schedule_marks_backend_state_batch_for_backend_owned_layers() {
    let cfg = qwen35_test_config();
    let owners = vec![
        crate::kv::qwen35_kv::Qwen35LayerStateOwner::BackendOwned,
        crate::kv::qwen35_kv::Qwen35LayerStateOwner::CpuMaterialized,
        crate::kv::qwen35_kv::Qwen35LayerStateOwner::Split,
        crate::kv::qwen35_kv::Qwen35LayerStateOwner::CpuMaterialized,
    ];
    let schedule = build_qwen35_prefill_schedule(&cfg, 0, 8, false, Some(&owners));
    let summary = summarize_qwen35_prefill_schedule(&schedule);

    assert_eq!(summary.backend_state_batch_layers, 2);
    assert!(matches!(
        schedule.layers[0].kind,
        Qwen35PrefillLayerScheduleKind::RecurrentGdn {
            force_backend_state_batch: true,
            ..
        }
    ));
    assert!(matches!(
        schedule.layers[1].kind,
        Qwen35PrefillLayerScheduleKind::RecurrentGdn {
            force_backend_state_batch: false,
            ..
        }
    ));
    assert!(matches!(
        schedule.layers[2].kind,
        Qwen35PrefillLayerScheduleKind::RecurrentGdn {
            force_backend_state_batch: true,
            ..
        }
    ));
}

#[test]
fn test_build_qwen35_full_attention_prefill_schedule_local_contains_qwen_ops() {
    let backend = MetalBackend::new().unwrap();
    let metal_ops = backend.metal_ops().unwrap();
    let cfg = qwen35_test_config();
    let n_tokens = 3usize;
    let q_dim = cfg.n_heads as usize * cfg.head_dim as usize;
    let kv_dim = cfg.n_kv_heads as usize * cfg.head_dim as usize;
    let inter_dim = cfg.intermediate_dim as usize;

    metal_ops.init_batch_scratches(&cfg, n_tokens);

    let attn_norm = vec![1.0f32; cfg.embedding_dim as usize];
    let q_norm = vec![1.0f32; cfg.head_dim as usize];
    let k_norm = vec![1.0f32; cfg.head_dim as usize];
    let ffn_norm = vec![1.0f32; cfg.embedding_dim as usize];
    let q_weight = vec![0u8; 64];
    let k_weight = vec![1u8; 64];
    let v_weight = vec![2u8; 64];
    let o_weight = vec![3u8; 64];
    let wg_weight = vec![4u8; 64];
    let wu_weight = vec![5u8; 64];
    let wd_weight = vec![6u8; 64];

    let cached_layer = CachedLayerKeys {
        attn_norm: metal_ops.ensure_f32_cached(&attn_norm),
        wq: metal_ops.ensure_quant_cached(&q_weight),
        wq_dtype: GgmlType::Q8_0,
        wk: metal_ops.ensure_quant_cached(&k_weight),
        wk_dtype: GgmlType::Q4K,
        wv: metal_ops.ensure_quant_cached(&v_weight),
        wv_dtype: GgmlType::Q4K,
        wo: metal_ops.ensure_quant_cached(&o_weight),
        wo_dtype: GgmlType::Q8_0,
        ffn_norm: metal_ops.ensure_f32_cached(&ffn_norm),
        wg: metal_ops.ensure_quant_cached(&wg_weight),
        wg_dtype: GgmlType::Q4K,
        wu: metal_ops.ensure_quant_cached(&wu_weight),
        wu_dtype: GgmlType::Q8_0,
        wd: metal_ops.ensure_quant_cached(&wd_weight),
        wd_dtype: GgmlType::Q4K,
        attn_q_norm: Some(metal_ops.ensure_f32_cached(&q_norm)),
        attn_k_norm: Some(metal_ops.ensure_f32_cached(&k_norm)),
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
    };

    let bs_guard = metal_ops.batch_scratches();
    let bs = bs_guard.as_ref().unwrap();
    let weight_cache = metal_ops.lock_weight_cache();
    let schedule = build_qwen35_full_attention_prefill_schedule(
        &cfg,
        &cached_layer,
        &weight_cache,
        bs,
        None,
        0,
        0,
        n_tokens,
        q_dim,
        kv_dim,
        inter_dim,
        metal_ops.attention_dispatch_config(),
    )
    .unwrap();

    assert!(
        schedule
            .ops
            .iter()
            .any(|op| matches!(op, PrefillOp::SplitQGateBatch { .. }))
    );
    assert!(
        schedule
            .ops
            .iter()
            .any(|op| matches!(op, PrefillOp::PerHeadRmsNormBatch { .. }))
    );
    assert!(
        schedule
            .ops
            .iter()
            .any(|op| matches!(op, PrefillOp::SigmoidElementwiseMul { .. }))
    );
    assert!(
        schedule
            .ops
            .iter()
            .any(|op| matches!(op, PrefillOp::AttentionBatchLocal { .. }))
    );
}

#[test]
fn test_build_qwen35_full_attention_prefill_schedule_cached_uses_kv_append_and_cached_attention() {
    let backend = MetalBackend::new().unwrap();
    let metal_ops = backend.metal_ops().unwrap();
    let cfg = qwen35_test_config();
    let n_tokens = 2usize;
    let q_dim = cfg.n_heads as usize * cfg.head_dim as usize;
    let kv_dim = cfg.n_kv_heads as usize * cfg.head_dim as usize;
    let inter_dim = cfg.intermediate_dim as usize;

    metal_ops.init_batch_scratches(&cfg, n_tokens);
    let gpu_kv = GpuKv::new(
        &metal_ops.device,
        cfg.n_layers as usize,
        cfg.n_kv_heads as usize,
        cfg.head_dim as usize,
        cfg.context_length as usize,
        4,
    )
    .unwrap();

    let attn_norm = vec![1.0f32; cfg.embedding_dim as usize];
    let ffn_norm = vec![1.0f32; cfg.embedding_dim as usize];
    let q_weight = vec![0u8; 64];
    let k_weight = vec![1u8; 64];
    let v_weight = vec![2u8; 64];
    let o_weight = vec![3u8; 64];
    let wg_weight = vec![4u8; 64];
    let wu_weight = vec![5u8; 64];
    let wd_weight = vec![6u8; 64];

    let cached_layer = CachedLayerKeys {
        attn_norm: metal_ops.ensure_f32_cached(&attn_norm),
        wq: metal_ops.ensure_quant_cached(&q_weight),
        wq_dtype: GgmlType::Q4K,
        wk: metal_ops.ensure_quant_cached(&k_weight),
        wk_dtype: GgmlType::Q4K,
        wv: metal_ops.ensure_quant_cached(&v_weight),
        wv_dtype: GgmlType::Q4K,
        wo: metal_ops.ensure_quant_cached(&o_weight),
        wo_dtype: GgmlType::Q4K,
        ffn_norm: metal_ops.ensure_f32_cached(&ffn_norm),
        wg: metal_ops.ensure_quant_cached(&wg_weight),
        wg_dtype: GgmlType::Q4K,
        wu: metal_ops.ensure_quant_cached(&wu_weight),
        wu_dtype: GgmlType::Q4K,
        wd: metal_ops.ensure_quant_cached(&wd_weight),
        wd_dtype: GgmlType::Q4K,
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
    };

    let bs_guard = metal_ops.batch_scratches();
    let bs = bs_guard.as_ref().unwrap();
    let weight_cache = metal_ops.lock_weight_cache();
    let schedule = build_qwen35_full_attention_prefill_schedule(
        &cfg,
        &cached_layer,
        &weight_cache,
        bs,
        Some(&gpu_kv),
        1,
        5,
        n_tokens,
        q_dim,
        kv_dim,
        inter_dim,
        metal_ops.attention_dispatch_config(),
    )
    .unwrap();

    assert_eq!(
        schedule
            .ops
            .iter()
            .filter(|op| matches!(op, PrefillOp::KvAppendBatch { .. }))
            .count(),
        2
    );
    assert!(
        schedule
            .ops
            .iter()
            .any(|op| matches!(op, PrefillOp::AttentionCached { .. }))
    );
}

#[test]
fn test_build_qwen35_recurrent_tail_prefill_schedule_includes_ssm_and_ffn_tail_ops() {
    let backend = MetalBackend::new().unwrap();
    let metal_ops = backend.metal_ops().unwrap();
    let cfg = qwen35_test_config();
    let n_tokens = 2usize;
    let inter_dim = cfg.intermediate_dim as usize;

    metal_ops.init_batch_scratches(&cfg, n_tokens);
    let bs_guard = metal_ops.batch_scratches();
    let bs = bs_guard.as_ref().unwrap();

    let alloc_f32 = |count: usize| {
        ax_engine_metal::MetalBuffer::new(
            metal_ops.device.device(),
            count * std::mem::size_of::<f32>(),
        )
        .unwrap()
    };
    let rec_out = alloc_f32(n_tokens * 8);
    let rec_z = alloc_f32(n_tokens * 8);
    let ssm_norm = alloc_f32(8);
    let ssm_weight = alloc_f32(cfg.embedding_dim as usize * 8);
    let ffn_norm = alloc_f32(cfg.embedding_dim as usize);
    let wg = alloc_f32(inter_dim * cfg.embedding_dim as usize);
    let wu = alloc_f32(inter_dim * cfg.embedding_dim as usize);
    let wd = alloc_f32(cfg.embedding_dim as usize * inter_dim);

    let schedule = build_qwen35_recurrent_tail_prefill_schedule(
        bs,
        Some(Qwen35RecurrentTailProjection {
            rec_out: &rec_out,
            rec_z: &rec_z,
            ssm_norm: &ssm_norm,
            ssm_weight: &ssm_weight,
            ssm_dtype: GgmlType::Q8_0,
            time_step_rank: 4,
            state_size: 2,
            inner_dim: 8,
        }),
        &ffn_norm,
        &wg,
        GgmlType::Q4K,
        &wu,
        GgmlType::Q8_0,
        &wd,
        GgmlType::Q4K,
        n_tokens,
        cfg.embedding_dim as usize,
        inter_dim,
        cfg.rms_norm_eps,
    );

    assert!(
        schedule
            .ops
            .iter()
            .any(|op| matches!(op, PrefillOp::PerHeadRmsNormBatch { .. }))
    );
    assert!(
        schedule
            .ops
            .iter()
            .any(|op| matches!(op, PrefillOp::ResidualAddRmsNormOutBatch { .. }))
    );
    assert!(
        schedule
            .ops
            .iter()
            .any(|op| matches!(op, PrefillOp::SiluMulBatch { .. }))
    );
    assert!(
        schedule
            .ops
            .iter()
            .filter(|op| matches!(
                op,
                PrefillOp::DequantBatch { .. } | PrefillOp::DequantBatchF16In { .. }
            ))
            .count()
            >= 4
    );
}
