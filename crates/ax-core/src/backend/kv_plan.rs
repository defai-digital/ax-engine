use super::{Backend, RuntimePolicy};
use crate::kv::gpu_kv::GpuKvDtype;
use crate::kv::page::{KvCacheConfig, KvDtype, initial_token_capacity, recommended_page_size};
use crate::kv::{CpuKv, GpuKv, ModelKv, Qwen35Kv};
use crate::model::config::ModelConfig;
use anyhow::ensure;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPlanKind {
    Cpu,
    Gpu,
    Qwen35,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvRollbackPolicy {
    PreciseTruncate,
    ResetOnRollback,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvPlannerRequirements {
    pub require_precise_rollback: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuKvPlan {
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub page_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuKvPlan {
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub page_size: usize,
    pub dtype: GpuKvDtype,
    pub fallback: CpuKvPlan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35KvPlan {
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub attention_page_size: usize,
    pub full_attention_interval: usize,
    pub conv_kernel: usize,
    pub inner_size: usize,
    pub state_size: usize,
    pub time_step_rank: usize,
    pub group_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvPlan {
    Cpu(CpuKvPlan),
    Gpu(GpuKvPlan),
    Qwen35(Qwen35KvPlan),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvPlanMemoryEstimate {
    pub initial_bytes: usize,
    pub max_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCapacityPolicy {
    pub initial_tokens: usize,
    pub growth_tokens: usize,
    pub max_tokens: usize,
}

pub struct KvPlanner;

impl KvPlanner {
    pub fn plan(backend: &dyn Backend, config: &ModelConfig) -> KvPlan {
        Self::plan_with_requirements(backend, config, KvPlannerRequirements::default())
            .expect("default KV planning should not fail")
    }

    pub fn plan_with_requirements(
        backend: &dyn Backend,
        config: &ModelConfig,
        requirements: KvPlannerRequirements,
    ) -> anyhow::Result<KvPlan> {
        let page_size = recommended_page_size(config.n_kv_heads as usize, config.head_dim as usize);

        let plan = if config.architecture == "qwen35" {
            KvPlan::Qwen35(validate_qwen35_plan(config, page_size)?)
        } else {
            let cpu_fallback = CpuKvPlan {
                n_layers: config.n_layers as usize,
                n_kv_heads: config.n_kv_heads as usize,
                head_dim: config.head_dim as usize,
                max_seq_len: config.context_length as usize,
                page_size,
            };

            if backend.use_gpu_decode() {
                let runtime_policy = backend
                    .runtime_policy()
                    .unwrap_or_else(RuntimePolicy::resolved_defaults);
                KvPlan::Gpu(GpuKvPlan {
                    n_layers: config.n_layers as usize,
                    n_kv_heads: config.n_kv_heads as usize,
                    head_dim: config.head_dim as usize,
                    max_seq_len: config.context_length as usize,
                    page_size,
                    dtype: runtime_policy.gpu_kv_dtype(config.context_length as usize),
                    fallback: cpu_fallback,
                })
            } else {
                KvPlan::Cpu(cpu_fallback)
            }
        };

        if requirements.require_precise_rollback {
            ensure!(
                plan.rollback_policy() == KvRollbackPolicy::PreciseTruncate,
                "KV planning requires precise rollback, but architecture '{}' resolves to {:?}",
                config.architecture,
                plan.rollback_policy()
            );
        }

        Ok(plan)
    }
}

fn validate_qwen35_plan(
    config: &ModelConfig,
    attention_page_size: usize,
) -> anyhow::Result<Qwen35KvPlan> {
    let full_attention_interval = config.qwen35_full_attention_interval.unwrap_or(0) as usize;
    let conv_kernel = config.qwen35_ssm_conv_kernel.unwrap_or(0) as usize;
    let inner_size = config.qwen35_ssm_inner_size.unwrap_or(0) as usize;
    let state_size = config.qwen35_ssm_state_size.unwrap_or(0) as usize;
    let time_step_rank = config.qwen35_ssm_time_step_rank.unwrap_or(0) as usize;
    let group_count = config.qwen35_ssm_group_count.unwrap_or(0) as usize;

    ensure!(
        full_attention_interval > 0,
        "qwen35 full_attention_interval must be > 0"
    );
    ensure!(conv_kernel > 0, "qwen35 conv_kernel must be > 0");
    ensure!(inner_size > 0, "qwen35 inner_size must be > 0");
    ensure!(state_size > 0, "qwen35 state_size must be > 0");
    ensure!(time_step_rank > 0, "qwen35 time_step_rank must be > 0");
    ensure!(group_count > 0, "qwen35 group_count must be > 0");
    ensure!(
        inner_size == state_size * time_step_rank,
        "qwen35 inner_size ({inner_size}) must equal state_size ({state_size}) * time_step_rank ({time_step_rank})"
    );
    ensure!(
        time_step_rank.is_multiple_of(group_count),
        "qwen35 time_step_rank ({time_step_rank}) must be a multiple of group_count ({group_count})"
    );

    Ok(Qwen35KvPlan {
        n_layers: config.n_layers as usize,
        n_kv_heads: config.n_kv_heads as usize,
        head_dim: config.head_dim as usize,
        max_seq_len: config.context_length as usize,
        attention_page_size,
        full_attention_interval,
        conv_kernel,
        inner_size,
        state_size,
        time_step_rank,
        group_count,
    })
}

impl KvPlan {
    pub fn kind(&self) -> KvPlanKind {
        match self {
            Self::Cpu(_) => KvPlanKind::Cpu,
            Self::Gpu(_) => KvPlanKind::Gpu,
            Self::Qwen35(_) => KvPlanKind::Qwen35,
        }
    }

    pub fn build(&self, backend: &dyn Backend) -> ModelKv {
        match self {
            Self::Cpu(plan) => ModelKv::Cpu(build_cpu_kv(plan)),
            Self::Qwen35(plan) => ModelKv::Qwen35(build_qwen35_kv(plan)),
            Self::Gpu(plan) => build_gpu_kv(plan, backend),
        }
    }

    pub fn build_decode_compatible(
        &self,
        backend: &dyn Backend,
        gpu_decode_supported: bool,
    ) -> ModelKv {
        match self {
            Self::Gpu(plan) if !gpu_decode_supported => {
                tracing::info!(
                    "Decode resolved to CPU for active weights; allocating CPU KV fallback"
                );
                ModelKv::Cpu(build_cpu_kv(&plan.fallback))
            }
            _ => self.build(backend),
        }
    }

    pub fn rollback_policy(&self) -> KvRollbackPolicy {
        match self {
            Self::Cpu(_) | Self::Gpu(_) => KvRollbackPolicy::PreciseTruncate,
            Self::Qwen35(_) => KvRollbackPolicy::ResetOnRollback,
        }
    }

    pub fn memory_estimate(&self) -> KvPlanMemoryEstimate {
        match self {
            Self::Cpu(plan) => {
                let cfg = cpu_kv_cache_config(plan);
                KvPlanMemoryEstimate {
                    initial_bytes: cfg.initial_memory_bytes(),
                    max_bytes: cfg.max_memory_bytes(),
                }
            }
            Self::Gpu(plan) => {
                let elem_size = plan.dtype.elem_size();
                let stride = plan.n_kv_heads * plan.head_dim;
                let initial_cap = initial_token_capacity(plan.page_size, plan.max_seq_len);
                KvPlanMemoryEstimate {
                    initial_bytes: 2 * plan.n_layers * initial_cap * stride * elem_size,
                    max_bytes: 2 * plan.n_layers * plan.max_seq_len * stride * elem_size,
                }
            }
            Self::Qwen35(plan) => qwen35_memory_estimate(plan),
        }
    }

    pub fn summary_label(&self) -> String {
        match self {
            Self::Cpu(_) => "cpu/f32".to_string(),
            Self::Gpu(plan) => match plan.dtype {
                GpuKvDtype::F32 => "gpu/f32".to_string(),
                GpuKvDtype::F16 => "gpu/f16".to_string(),
            },
            Self::Qwen35(_) => "qwen35-hybrid".to_string(),
        }
    }

    pub fn capacity_policy(&self) -> KvCapacityPolicy {
        match self {
            Self::Cpu(plan) => KvCapacityPolicy {
                initial_tokens: initial_token_capacity(plan.page_size, plan.max_seq_len),
                growth_tokens: plan.page_size,
                max_tokens: plan.max_seq_len,
            },
            Self::Gpu(plan) => KvCapacityPolicy {
                initial_tokens: initial_token_capacity(plan.page_size, plan.max_seq_len),
                growth_tokens: plan.page_size,
                max_tokens: plan.max_seq_len,
            },
            Self::Qwen35(plan) => KvCapacityPolicy {
                initial_tokens: initial_token_capacity(plan.attention_page_size, plan.max_seq_len),
                growth_tokens: plan.attention_page_size,
                max_tokens: plan.max_seq_len,
            },
        }
    }
}

impl KvRollbackPolicy {
    pub fn label(self) -> &'static str {
        match self {
            Self::PreciseTruncate => "precise",
            Self::ResetOnRollback => "reset-only",
        }
    }
}

fn build_cpu_kv(plan: &CpuKvPlan) -> CpuKv {
    CpuKv::with_config(&cpu_kv_cache_config(plan))
}

fn build_qwen35_kv(plan: &Qwen35KvPlan) -> Qwen35Kv {
    Qwen35Kv::new_with_attention_page_size(
        plan.n_layers,
        plan.n_kv_heads,
        plan.head_dim,
        plan.max_seq_len,
        plan.attention_page_size,
        plan.full_attention_interval,
        plan.conv_kernel,
        plan.inner_size,
        plan.state_size,
        plan.time_step_rank,
        plan.group_count,
    )
}

fn build_gpu_kv(plan: &GpuKvPlan, backend: &dyn Backend) -> ModelKv {
    let Some(metal_ops) = backend.metal_ops() else {
        tracing::warn!(
            "Backend reported GPU decode support without Metal ops; falling back to CPU KV"
        );
        return ModelKv::Cpu(build_cpu_kv(&plan.fallback));
    };

    match GpuKv::new_with_dtype(
        &metal_ops.device,
        plan.n_layers,
        plan.n_kv_heads,
        plan.head_dim,
        plan.max_seq_len,
        plan.page_size,
        plan.dtype,
    ) {
        Ok(gpu_kv) => {
            tracing::info!(kv_dtype = ?plan.dtype, "Initialized GPU KV cache");
            ModelKv::Gpu(gpu_kv)
        }
        Err(e) => {
            tracing::warn!("GPU KV allocation failed, falling back to CPU: {e}");
            ModelKv::Cpu(build_cpu_kv(&plan.fallback))
        }
    }
}

fn cpu_kv_cache_config(plan: &CpuKvPlan) -> KvCacheConfig {
    KvCacheConfig {
        n_layers: plan.n_layers,
        n_kv_heads: plan.n_kv_heads,
        head_dim: plan.head_dim,
        max_seq_len: plan.max_seq_len,
        page_size: plan.page_size,
        dtype: KvDtype::F32,
    }
}

fn qwen35_memory_estimate(plan: &Qwen35KvPlan) -> KvPlanMemoryEstimate {
    let attention_cfg = KvCacheConfig {
        n_layers: plan.n_layers,
        n_kv_heads: plan.n_kv_heads,
        head_dim: plan.head_dim,
        max_seq_len: plan.max_seq_len,
        page_size: plan.attention_page_size,
        dtype: KvDtype::F32,
    };
    let recurrent_layers = (0..plan.n_layers)
        .filter(|layer| (layer + 1) % plan.full_attention_interval != 0)
        .count();
    let conv_cache_len = plan.conv_kernel.saturating_sub(1);
    let conv_dim = plan.inner_size + 2 * plan.group_count * plan.state_size;
    let recurrent_state_len = plan.time_step_rank * plan.state_size * plan.state_size;
    let fixed_state_bytes = recurrent_layers
        * (conv_cache_len * conv_dim + recurrent_state_len)
        * std::mem::size_of::<f32>();

    KvPlanMemoryEstimate {
        initial_bytes: attention_cfg.initial_memory_bytes() + fixed_state_bytes,
        max_bytes: attention_cfg.max_memory_bytes() + fixed_state_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::backend::KvPrecisionPolicy;
    use crate::model::config::{GateActivation, RopeScaling};

    struct TestBackend {
        use_gpu_decode: bool,
        runtime_policy: Option<RuntimePolicy>,
    }

    impl Backend for TestBackend {
        fn runtime_policy(&self) -> Option<RuntimePolicy> {
            self.runtime_policy.clone()
        }

        fn matmul(&self, _a: &[f32], _b: &[f32], _c: &mut [f32], _m: usize, _n: usize, _k: usize) {
            unreachable!("test backend matmul should not be called")
        }

        fn use_gpu_decode(&self) -> bool {
            self.use_gpu_decode
        }
    }

    fn base_config() -> ModelConfig {
        ModelConfig {
            architecture: "llama".into(),
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
        }
    }

    #[test]
    fn test_kv_planner_selects_cpu_for_cpu_decode() {
        let backend = TestBackend {
            use_gpu_decode: false,
            runtime_policy: None,
        };
        let plan = KvPlanner::plan(&backend, &base_config());

        match plan {
            KvPlan::Cpu(plan) => assert_eq!(plan.page_size, 64),
            other => panic!("expected CPU plan, got {other:?}"),
        }
    }

    #[test]
    fn test_kv_planner_selects_gpu_with_runtime_policy_dtype() {
        let backend = TestBackend {
            use_gpu_decode: true,
            runtime_policy: Some(
                RuntimePolicy::resolved_defaults()
                    .with_kv_precision_policy(KvPrecisionPolicy::ForceF16),
            ),
        };
        let plan = KvPlanner::plan(&backend, &base_config());

        match plan {
            KvPlan::Gpu(plan) => {
                assert_eq!(plan.dtype, GpuKvDtype::F16);
                assert_eq!(plan.page_size, 64);
            }
            other => panic!("expected GPU plan, got {other:?}"),
        }
    }

    #[test]
    fn test_kv_planner_selects_qwen35_hybrid_plan() {
        let backend = TestBackend {
            use_gpu_decode: true,
            runtime_policy: None,
        };
        let mut config = base_config();
        config.architecture = "qwen35".into();
        config.qwen35_full_attention_interval = Some(4);
        config.qwen35_ssm_conv_kernel = Some(4);
        config.qwen35_ssm_inner_size = Some(1024);
        config.qwen35_ssm_state_size = Some(128);
        config.qwen35_ssm_time_step_rank = Some(8);
        config.qwen35_ssm_group_count = Some(2);

        let plan = KvPlanner::plan(&backend, &config);

        match plan {
            KvPlan::Qwen35(plan) => {
                assert_eq!(plan.attention_page_size, 64);
                assert_eq!(plan.full_attention_interval, 4);
                assert_eq!(plan.conv_kernel, 4);
                assert_eq!(plan.inner_size, 1024);
            }
            other => panic!("expected Qwen35 plan, got {other:?}"),
        }
    }

    #[test]
    fn test_kv_plan_build_cpu_variant() {
        let backend = TestBackend {
            use_gpu_decode: false,
            runtime_policy: None,
        };

        let kv = KvPlanner::plan(&backend, &base_config()).build(&backend);

        assert!(matches!(kv, ModelKv::Cpu(_)));
    }

    #[test]
    fn test_gpu_plan_build_decode_compatible_falls_back_to_cpu_when_gpu_decode_is_unsupported() {
        let backend = TestBackend {
            use_gpu_decode: true,
            runtime_policy: None,
        };

        let kv = KvPlanner::plan(&backend, &base_config()).build_decode_compatible(&backend, false);

        assert!(matches!(kv, ModelKv::Cpu(_)));
    }

    #[test]
    fn test_qwen35_plan_marks_reset_on_rollback() {
        let backend = TestBackend {
            use_gpu_decode: true,
            runtime_policy: None,
        };
        let mut config = base_config();
        config.architecture = "qwen35".into();
        config.qwen35_full_attention_interval = Some(4);
        config.qwen35_ssm_conv_kernel = Some(4);
        config.qwen35_ssm_inner_size = Some(1024);
        config.qwen35_ssm_state_size = Some(128);
        config.qwen35_ssm_time_step_rank = Some(8);
        config.qwen35_ssm_group_count = Some(2);

        let plan = KvPlanner::plan(&backend, &config);

        assert_eq!(plan.rollback_policy(), KvRollbackPolicy::ResetOnRollback);
    }

    #[test]
    fn test_kv_planner_rejects_qwen35_when_precise_rollback_required() {
        let backend = TestBackend {
            use_gpu_decode: true,
            runtime_policy: None,
        };
        let mut config = base_config();
        config.architecture = "qwen35".into();
        config.qwen35_full_attention_interval = Some(4);
        config.qwen35_ssm_conv_kernel = Some(4);
        config.qwen35_ssm_inner_size = Some(1024);
        config.qwen35_ssm_state_size = Some(128);
        config.qwen35_ssm_time_step_rank = Some(8);
        config.qwen35_ssm_group_count = Some(2);

        let err = KvPlanner::plan_with_requirements(
            &backend,
            &config,
            KvPlannerRequirements {
                require_precise_rollback: true,
            },
        )
        .unwrap_err();

        assert!(err.to_string().contains("requires precise rollback"));
    }

    #[test]
    fn test_kv_planner_rejects_qwen35_with_incompatible_head_expansion() {
        let backend = TestBackend {
            use_gpu_decode: true,
            runtime_policy: None,
        };
        let mut config = base_config();
        config.architecture = "qwen35".into();
        config.qwen35_full_attention_interval = Some(4);
        config.qwen35_ssm_conv_kernel = Some(4);
        config.qwen35_ssm_inner_size = Some(768);
        config.qwen35_ssm_state_size = Some(128);
        config.qwen35_ssm_time_step_rank = Some(6);
        config.qwen35_ssm_group_count = Some(4);

        let err =
            KvPlanner::plan_with_requirements(&backend, &config, KvPlannerRequirements::default())
                .unwrap_err();

        assert!(err.to_string().contains("multiple of group_count"));
    }

    #[test]
    fn test_cpu_plan_memory_estimate_matches_page_config() {
        let backend = TestBackend {
            use_gpu_decode: false,
            runtime_policy: None,
        };
        let estimate = KvPlanner::plan(&backend, &base_config()).memory_estimate();

        assert_eq!(estimate.initial_bytes, 16_777_216);
        assert_eq!(estimate.max_bytes, 1_073_741_824);
    }

    #[test]
    fn test_gpu_plan_memory_estimate_respects_f16_dtype() {
        let backend = TestBackend {
            use_gpu_decode: true,
            runtime_policy: Some(
                RuntimePolicy::resolved_defaults()
                    .with_kv_precision_policy(KvPrecisionPolicy::ForceF16),
            ),
        };
        let estimate = KvPlanner::plan(&backend, &base_config()).memory_estimate();

        assert_eq!(estimate.initial_bytes, 8_388_608);
        assert_eq!(estimate.max_bytes, 536_870_912);
    }

    #[test]
    fn test_qwen35_plan_memory_estimate_includes_fixed_recurrent_state() {
        let backend = TestBackend {
            use_gpu_decode: true,
            runtime_policy: None,
        };
        let mut config = base_config();
        config.architecture = "qwen35".into();
        config.n_layers = 8;
        config.context_length = 1024;
        config.qwen35_full_attention_interval = Some(4);
        config.qwen35_ssm_conv_kernel = Some(4);
        config.qwen35_ssm_inner_size = Some(1024);
        config.qwen35_ssm_state_size = Some(128);
        config.qwen35_ssm_time_step_rank = Some(8);
        config.qwen35_ssm_group_count = Some(2);

        let estimate = KvPlanner::plan(&backend, &config).memory_estimate();

        assert_eq!(estimate.initial_bytes, 7_450_624);
        assert_eq!(estimate.max_bytes, 70_365_184);
    }

    #[test]
    fn test_capacity_policy_matches_planned_page_growth() {
        let backend = TestBackend {
            use_gpu_decode: false,
            runtime_policy: None,
        };
        let capacity = KvPlanner::plan(&backend, &base_config()).capacity_policy();

        assert_eq!(capacity.initial_tokens, 64);
        assert_eq!(capacity.growth_tokens, 64);
        assert_eq!(capacity.max_tokens, 4096);
    }
}
