use super::*;
use crate::backend::Backend;
use crate::backend::metal::MetalBackend;
use crate::gguf::mmap::MappedModel;
use crate::gguf::tensor::GgmlType;
use crate::model::InferenceModel;
use crate::model::config::ModelConfig;
use crate::model::weights::WeightStore;
use std::path::PathBuf;

struct EnvVarGuard {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.previous {
            Some(previous) => unsafe { std::env::set_var(self.key, previous) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

fn workspace_model_path(file_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .join(file_name)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[derive(Debug)]
struct MoeRouteDebug {
    router_logits: Vec<f32>,
    top_ids: Vec<usize>,
    top_weights: Vec<f32>,
}

fn cpu_moe_route_for_hidden(
    weights: &WeightStore,
    prefix: &str,
    hidden: &[f32],
    dim: usize,
    n_expert: usize,
    n_expert_used: usize,
    rms_norm_eps: f32,
) -> MoeRouteDebug {
    let mut norm_buf = vec![0.0f32; dim];
    let ffn_norm_w = weights
        .f32_slice(&format!("{prefix}.ffn_norm.weight"))
        .unwrap();
    crate::compute::rms_norm::rms_norm_out(hidden, ffn_norm_w, &mut norm_buf, rms_norm_eps);

    let (router_raw, router_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.ffn_gate_inp.weight"))
        .unwrap();
    let mut router_logits = vec![0.0f32; n_expert];
    let cpu = crate::backend::cpu::CpuBackend;
    cpu.dequant_matmul(
        router_raw,
        router_dtype,
        &norm_buf,
        &mut router_logits,
        n_expert,
        1,
        dim,
    );
    let (top_ids, top_weights) =
        crate::model::moe_utils::top_k_softmax(&router_logits, n_expert_used);
    MoeRouteDebug {
        router_logits,
        top_ids,
        top_weights,
    }
}

fn first_nonfinite(values: &[f32]) -> Option<(usize, f32)> {
    values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
}

fn staged_hidden(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let base = ((i * 13 % 97) as f32 / 48.5) - 1.0;
            let wobble = ((i * 7 % 31) as f32 - 15.0) * 0.0175;
            base + wobble
        })
        .collect()
}

#[test]
fn test_arch_name() {
    let fwd = Qwen3MoeForward;
    assert_eq!(fwd.arch_name(), "qwen3moe");
}

#[test]
fn test_moe_gpu_expert_dtype_supported() {
    assert!(Qwen3MoeForward::moe_gpu_expert_dtype_supported(
        GgmlType::Q4K
    ));
    assert!(Qwen3MoeForward::moe_gpu_expert_dtype_supported(
        GgmlType::Q5K
    ));
    assert!(Qwen3MoeForward::moe_gpu_expert_dtype_supported(
        GgmlType::Q6K
    ));
    assert!(Qwen3MoeForward::moe_gpu_expert_dtype_supported(
        GgmlType::Q8_0
    ));
    assert!(!Qwen3MoeForward::moe_gpu_expert_dtype_supported(
        GgmlType::F32
    ));
}

#[test]
fn test_qwen3moe_decode_layers_per_command_buffer_defaults_to_full_stack_for_supported_quants() {
    let _lock = crate::test_env_lock();
    let _guard = EnvVarGuard {
        key: "AX_QWEN3MOE_GPU_DECODE_LAYERS_PER_CB",
        previous: std::env::var_os("AX_QWEN3MOE_GPU_DECODE_LAYERS_PER_CB"),
    };
    unsafe { std::env::remove_var("AX_QWEN3MOE_GPU_DECODE_LAYERS_PER_CB") };

    assert_eq!(
        Qwen3MoeForward::qwen3moe_decode_layers_per_command_buffer(
            48,
            GgmlType::Q4K,
            GgmlType::Q4K,
            GgmlType::Q4K,
        ),
        48
    );
    assert_eq!(
        Qwen3MoeForward::qwen3moe_decode_layers_per_command_buffer(
            48,
            GgmlType::Q5K,
            GgmlType::Q5K,
            GgmlType::Q5K,
        ),
        48
    );
    assert_eq!(
        Qwen3MoeForward::qwen3moe_decode_layers_per_command_buffer(
            48,
            GgmlType::Q6K,
            GgmlType::Q6K,
            GgmlType::Q6K,
        ),
        48
    );
    assert_eq!(
        Qwen3MoeForward::qwen3moe_decode_layers_per_command_buffer(
            48,
            GgmlType::Q8_0,
            GgmlType::Q8_0,
            GgmlType::Q8_0,
        ),
        48
    );
    assert_eq!(
        Qwen3MoeForward::qwen3moe_decode_layers_per_command_buffer(
            48,
            GgmlType::Q4K,
            GgmlType::Q4K,
            GgmlType::Q6K,
        ),
        48
    );
    assert_eq!(
        Qwen3MoeForward::qwen3moe_decode_layers_per_command_buffer(
            48,
            GgmlType::Q5K,
            GgmlType::Q5K,
            GgmlType::Q8_0,
        ),
        48
    );
    assert_eq!(
        Qwen3MoeForward::qwen3moe_decode_layers_per_command_buffer(
            48,
            GgmlType::Q4K,
            GgmlType::F32,
            GgmlType::Q4K,
        ),
        1
    );
}

#[test]
fn test_qwen3moe_decode_layers_per_command_buffer_env_override_applies() {
    let _lock = crate::test_env_lock();

    let _four = EnvVarGuard::set("AX_QWEN3MOE_GPU_DECODE_LAYERS_PER_CB", "6");
    assert_eq!(
        Qwen3MoeForward::qwen3moe_decode_layers_per_command_buffer(
            48,
            GgmlType::Q8_0,
            GgmlType::Q8_0,
            GgmlType::Q8_0,
        ),
        6
    );
    drop(_four);

    let _invalid = EnvVarGuard::set("AX_QWEN3MOE_GPU_DECODE_LAYERS_PER_CB", "0");
    assert_eq!(
        Qwen3MoeForward::qwen3moe_decode_layers_per_command_buffer(
            48,
            GgmlType::Q8_0,
            GgmlType::Q8_0,
            GgmlType::Q8_0,
        ),
        48
    );

    drop(_invalid);

    let _large = EnvVarGuard::set("AX_QWEN3MOE_GPU_DECODE_LAYERS_PER_CB", "96");
    assert_eq!(
        Qwen3MoeForward::qwen3moe_decode_layers_per_command_buffer(
            48,
            GgmlType::Q8_0,
            GgmlType::Q8_0,
            GgmlType::Q8_0,
        ),
        48
    );
}

#[test]
fn test_qwen3moe_prefill_split_layer_keeps_q8_single_cb() {
    assert_eq!(
        Qwen3MoeForward::qwen3moe_prefill_split_layer(
            48,
            GgmlType::Q8_0,
            GgmlType::Q8_0,
            GgmlType::Q8_0,
            true,
        ),
        48
    );
    assert_eq!(
        Qwen3MoeForward::qwen3moe_prefill_split_layer(
            48,
            GgmlType::Q4K,
            GgmlType::Q4K,
            GgmlType::Q4K,
            true,
        ),
        24
    );
    assert_eq!(
        Qwen3MoeForward::qwen3moe_prefill_split_layer(
            48,
            GgmlType::Q6K,
            GgmlType::Q6K,
            GgmlType::Q6K,
            true,
        ),
        24
    );
    assert_eq!(
        Qwen3MoeForward::qwen3moe_prefill_split_layer(
            48,
            GgmlType::Q6K,
            GgmlType::Q6K,
            GgmlType::Q6K,
            false,
        ),
        48
    );
}

#[test]
fn test_qwen3moe_prefill_concurrent_enabled_defaults_by_quant() {
    let _lock = crate::test_env_lock();
    let _guard = EnvVarGuard {
        key: "AX_QWEN3MOE_PREFILL_CONCURRENT",
        previous: std::env::var_os("AX_QWEN3MOE_PREFILL_CONCURRENT"),
    };
    unsafe { std::env::remove_var("AX_QWEN3MOE_PREFILL_CONCURRENT") };

    assert!(!Qwen3MoeForward::qwen3moe_prefill_concurrent_enabled(
        GgmlType::Q4K,
        GgmlType::Q4K,
        GgmlType::Q4K,
    ));
    assert!(Qwen3MoeForward::qwen3moe_prefill_concurrent_enabled(
        GgmlType::Q5K,
        GgmlType::Q5K,
        GgmlType::Q5K,
    ));
    assert!(Qwen3MoeForward::qwen3moe_prefill_concurrent_enabled(
        GgmlType::Q6K,
        GgmlType::Q6K,
        GgmlType::Q6K,
    ));
    assert!(Qwen3MoeForward::qwen3moe_prefill_concurrent_enabled(
        GgmlType::Q8_0,
        GgmlType::Q8_0,
        GgmlType::Q8_0,
    ));
}

#[test]
fn test_qwen3moe_prefill_concurrent_enabled_env_override_applies() {
    let _lock = crate::test_env_lock();

    let _on = EnvVarGuard::set("AX_QWEN3MOE_PREFILL_CONCURRENT", "1");
    assert!(Qwen3MoeForward::qwen3moe_prefill_concurrent_enabled(
        GgmlType::Q4K,
        GgmlType::Q4K,
        GgmlType::Q4K,
    ));
    drop(_on);

    let _off = EnvVarGuard::set("AX_QWEN3MOE_PREFILL_CONCURRENT", "0");
    assert!(!Qwen3MoeForward::qwen3moe_prefill_concurrent_enabled(
        GgmlType::Q8_0,
        GgmlType::Q8_0,
        GgmlType::Q8_0,
    ));
}

#[test]
fn test_qwen3moe_blocked_q6q8_down_defaults_on() {
    let _lock = crate::test_env_lock();
    let _guard = EnvVarGuard {
        key: "AX_QWEN3MOE_BLOCKED_Q6Q8_DOWN",
        previous: std::env::var_os("AX_QWEN3MOE_BLOCKED_Q6Q8_DOWN"),
    };
    unsafe { std::env::remove_var("AX_QWEN3MOE_BLOCKED_Q6Q8_DOWN") };

    assert!(Qwen3MoeForward::qwen3moe_blocked_q6q8_down_enabled());
}

#[test]
fn test_qwen3moe_blocked_q6q8_down_env_override_applies() {
    let _lock = crate::test_env_lock();

    let _off = EnvVarGuard::set("AX_QWEN3MOE_BLOCKED_Q6Q8_DOWN", "0");
    assert!(!Qwen3MoeForward::qwen3moe_blocked_q6q8_down_enabled());
    drop(_off);

    let _on = EnvVarGuard::set("AX_QWEN3MOE_BLOCKED_Q6Q8_DOWN", "1");
    assert!(Qwen3MoeForward::qwen3moe_blocked_q6q8_down_enabled());
}

#[test]
fn test_qwen3moe_gpu_pipelined_decode_defaults_on() {
    let _lock = crate::test_env_lock();
    let _guard = EnvVarGuard {
        key: "AX_QWEN3MOE_GPU_PIPELINED_DECODE",
        previous: std::env::var_os("AX_QWEN3MOE_GPU_PIPELINED_DECODE"),
    };
    unsafe { std::env::remove_var("AX_QWEN3MOE_GPU_PIPELINED_DECODE") };

    assert!(Qwen3MoeForward::qwen3moe_gpu_pipelined_decode_enabled());
}

#[test]
fn test_qwen3moe_gpu_pipelined_decode_env_override_applies() {
    let _lock = crate::test_env_lock();

    let _on = EnvVarGuard::set("AX_QWEN3MOE_GPU_PIPELINED_DECODE", "1");
    assert!(Qwen3MoeForward::qwen3moe_gpu_pipelined_decode_enabled());
    drop(_on);

    let _off = EnvVarGuard::set("AX_QWEN3MOE_GPU_PIPELINED_DECODE", "0");
    assert!(!Qwen3MoeForward::qwen3moe_gpu_pipelined_decode_enabled());
}

#[test]
fn test_qwen3moe_concurrent_decode_defaults_by_quant() {
    let _lock = crate::test_env_lock();
    let _guard = EnvVarGuard {
        key: "AX_QWEN3MOE_CONCURRENT_DECODE",
        previous: std::env::var_os("AX_QWEN3MOE_CONCURRENT_DECODE"),
    };
    unsafe { std::env::remove_var("AX_QWEN3MOE_CONCURRENT_DECODE") };

    assert!(Qwen3MoeForward::qwen3moe_concurrent_decode_enabled(
        GgmlType::Q4K,
        GgmlType::Q4K,
        GgmlType::Q4K,
    ));
    assert!(Qwen3MoeForward::qwen3moe_concurrent_decode_enabled(
        GgmlType::Q5K,
        GgmlType::Q5K,
        GgmlType::Q5K,
    ));
    assert!(Qwen3MoeForward::qwen3moe_concurrent_decode_enabled(
        GgmlType::Q4K,
        GgmlType::Q4K,
        GgmlType::Q6K,
    ));
    assert!(Qwen3MoeForward::qwen3moe_concurrent_decode_enabled(
        GgmlType::Q5K,
        GgmlType::Q5K,
        GgmlType::Q8_0,
    ));
    assert!(!Qwen3MoeForward::qwen3moe_concurrent_decode_enabled(
        GgmlType::Q4K,
        GgmlType::Q5K,
        GgmlType::Q6K,
    ));
    assert!(!Qwen3MoeForward::qwen3moe_concurrent_decode_enabled(
        GgmlType::Q6K,
        GgmlType::Q6K,
        GgmlType::Q6K,
    ));
    assert!(!Qwen3MoeForward::qwen3moe_concurrent_decode_enabled(
        GgmlType::Q8_0,
        GgmlType::Q8_0,
        GgmlType::Q8_0,
    ));
}

#[test]
fn test_qwen3moe_concurrent_decode_env_override_applies() {
    let _lock = crate::test_env_lock();

    let _on = EnvVarGuard::set("AX_QWEN3MOE_CONCURRENT_DECODE", "1");
    assert!(Qwen3MoeForward::qwen3moe_concurrent_decode_enabled(
        GgmlType::Q8_0,
        GgmlType::Q8_0,
        GgmlType::Q8_0,
    ));
    drop(_on);

    let _off = EnvVarGuard::set("AX_QWEN3MOE_CONCURRENT_DECODE", "0");
    assert!(!Qwen3MoeForward::qwen3moe_concurrent_decode_enabled(
        GgmlType::Q4K,
        GgmlType::Q4K,
        GgmlType::Q4K,
    ));
}

#[test]
fn test_real_qwen3_coder_30b_a3b_gpu_dispatch_supported_for_all_shipped_quants() {
    for model_file in [
        "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        "Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf",
        "Qwen3-Coder-30B-A3B-Instruct-Q6_K.gguf",
        "Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf",
    ] {
        let path = workspace_model_path(model_file);
        if !path.exists() {
            continue;
        }

        let mapped = MappedModel::open(&path).unwrap();
        let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
        let weights = WeightStore::new(&mapped);
        assert_eq!(cfg.architecture, "qwen3moe");
        assert!(
            Qwen3MoeForward::moe_gpu_decode_supported(&cfg, &weights),
            "{model_file} attention weights should support GPU decode",
        );
        assert!(
            Qwen3MoeForward::moe_gpu_expert_dispatch_supported(&weights),
            "{model_file} routed experts should support GPU dispatch",
        );

        let model =
            InferenceModel::with_backend(cfg.clone(), Box::new(MetalBackend::new().unwrap()))
                .unwrap();
        let kv = model.create_model_kv_for_weights(&weights);
        assert!(
            kv.is_gpu(),
            "{model_file} should allocate GPU KV when GPU expert dispatch is available",
        );
    }
}

#[test]
fn test_real_qwen3_coder_30b_a3b_expert_dtypes_match_expected_layouts() {
    for (model_file, gate_expected, up_expected, down_expected) in [
        (
            "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
            GgmlType::Q4K,
            GgmlType::Q4K,
            GgmlType::Q6K,
        ),
        (
            "Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf",
            GgmlType::Q5K,
            GgmlType::Q5K,
            GgmlType::Q6K,
        ),
        (
            "Qwen3-Coder-30B-A3B-Instruct-Q6_K.gguf",
            GgmlType::Q6K,
            GgmlType::Q6K,
            GgmlType::Q6K,
        ),
        (
            "Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf",
            GgmlType::Q8_0,
            GgmlType::Q8_0,
            GgmlType::Q8_0,
        ),
    ] {
        let path = workspace_model_path(model_file);
        if !path.exists() {
            continue;
        }

        let mapped = MappedModel::open(&path).unwrap();
        let weights = WeightStore::new(&mapped);
        let (gate_dtype, up_dtype, down_dtype) =
            crate::model::shared::routed_moe_expert_dtypes(&weights, "blk.0").unwrap();

        assert_eq!(gate_dtype, gate_expected, "{model_file} gate dtype");
        assert_eq!(up_dtype, up_expected, "{model_file} up dtype");
        assert_eq!(down_dtype, down_expected, "{model_file} down dtype");
    }
}

#[test]
fn test_real_qwen3_coder_q6_q8_layer0_resident_moe_tail_matches_cpu_from_staged_hidden() {
    let _env_lock = crate::test_env_lock();
    let _smart_off = EnvVarGuard::set("AX_METAL_SMART_BARRIERS", "0");
    let _barriers_on = EnvVarGuard::set("AX_METAL_BARRIERS", "1");

    for model_file in [
        "Qwen3-Coder-30B-A3B-Instruct-Q6_K.gguf",
        "Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf",
    ] {
        let path = workspace_model_path(model_file);
        if !path.exists() {
            continue;
        }

        let mapped = MappedModel::open(&path).unwrap();
        let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
        let weights = WeightStore::new(&mapped);
        let metal = MetalBackend::new().unwrap();
        let metal_ops = metal.metal_ops().unwrap();

        let n_tokens = 8usize;
        let dim = cfg.embedding_dim as usize;
        let n_expert = cfg.n_expert.unwrap_or(0) as usize;
        let n_expert_used = cfg.n_expert_used.unwrap_or(0) as usize;
        let expert_inter_dim = cfg.expert_intermediate_dim.unwrap_or(0) as usize;
        let prefix = "blk.0";

        let (gate_dtype, up_dtype, down_dtype) =
            crate::model::shared::routed_moe_expert_dtypes(&weights, prefix).unwrap();
        let gate_stride =
            crate::model::moe_utils::expert_byte_stride(gate_dtype, expert_inter_dim * dim);
        let up_stride =
            crate::model::moe_utils::expert_byte_stride(up_dtype, expert_inter_dim * dim);
        let down_stride =
            crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);

        let input_hidden = staged_hidden(n_tokens * dim);
        let mut expected_hidden = input_hidden.clone();
        let mut moe_norm_buf = vec![0.0f32; n_tokens * dim];
        let mut moe_accum_buf = vec![0.0f32; n_tokens * dim];
        let mut moe_scratch = MoeSingleScratch {
            gate_buf: vec![0.0f32; expert_inter_dim],
            up_buf: vec![0.0f32; expert_inter_dim],
            down_buf: vec![0.0f32; dim],
            accum_buf: vec![0.0f32; dim],
            router_logits: vec![0.0f32; n_expert],
        };
        Qwen3MoeForward::apply_moe_ffn_batch(
            &crate::backend::cpu::CpuBackend,
            &weights,
            prefix,
            &mut expected_hidden,
            &mut moe_norm_buf,
            &mut moe_accum_buf,
            &mut moe_scratch,
            n_tokens,
            dim,
            n_expert,
            n_expert_used,
            expert_inter_dim,
            cfg.rms_norm_eps,
            None,
        )
        .unwrap();

        metal_ops.init_batch_scratches(&cfg, n_tokens);
        if !metal_ops.has_cached_model_keys() {
            Qwen3MoeForward::build_cached_model_keys(metal_ops, &weights, &cfg).unwrap();
        }

        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();
        let layer = &cached.layers[0];

        let ffn_norm = {
            let weight_cache = metal_ops.lock_weight_cache();
            weight_cache.get(&layer.ffn_norm).unwrap().clone()
        };
        let router = {
            let moe_weight_cache = metal_ops.lock_moe_weight_cache();
            moe_weight_cache
                .get(&layer.moe_router.unwrap())
                .unwrap()
                .clone()
        };
        let gate = {
            let moe_weight_cache = metal_ops.lock_moe_weight_cache();
            moe_weight_cache
                .get(&layer.moe_expert_gate.as_ref().unwrap()[0])
                .unwrap()
                .clone()
        };
        let up = {
            let moe_weight_cache = metal_ops.lock_moe_weight_cache();
            moe_weight_cache
                .get(&layer.moe_expert_up.as_ref().unwrap()[0])
                .unwrap()
                .clone()
        };
        let down = {
            let moe_weight_cache = metal_ops.lock_moe_weight_cache();
            moe_weight_cache
                .get(&layer.moe_expert_down.as_ref().unwrap()[0])
                .unwrap()
                .clone()
        };

        {
            let mut bs_guard = metal_ops.batch_scratches();
            let bs = bs_guard.as_mut().unwrap();
            unsafe {
                bs.hidden.as_mut_slice::<f32>()[..n_tokens * dim].copy_from_slice(&input_hidden);
            }
            let moe_scratch =
                crate::backend::metal::MoeBatchScratchView::from_batch_scratches(bs).unwrap();

            metal_ops
                .device
                .execute_sync(|encoder| {
                    metal_ops.encode_moe_ffn_gpu_resident_cached_with_scratch_with_policy(
                        encoder,
                        moe_scratch,
                        &bs.hidden,
                        &ffn_norm,
                        &router,
                        layer.moe_router_dtype.unwrap(),
                        &gate,
                        gate_dtype,
                        &up,
                        up_dtype,
                        &down,
                        down_dtype,
                        n_tokens,
                        n_expert,
                        n_expert_used,
                        dim,
                        expert_inter_dim,
                        gate_stride,
                        up_stride,
                        down_stride,
                        cfg.rms_norm_eps,
                        None,
                        true,
                        Qwen3MoeForward::qwen3moe_blocked_q6q8_down_enabled(),
                    )
                })
                .unwrap();
        }

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
            diff / scale < 1e-2,
            "{model_file} layer0 resident MoE batch tail mismatch: rel_diff={} max_diff={} actual[0..8]={:?} expected[0..8]={:?}",
            diff / scale,
            diff,
            &actual_hidden[..8],
            &expected_hidden[..8],
        );
    }
}

#[test]
fn test_real_qwen3_coder_q6_q8_layer0_projected_hidden_before_moe_gpu_stage_matches_cpu() {
    let _env_lock = crate::test_env_lock();

    for model_file in [
        "Qwen3-Coder-30B-A3B-Instruct-Q6_K.gguf",
        "Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf",
    ] {
        let path = workspace_model_path(model_file);
        if !path.exists() {
            continue;
        }

        let mapped = MappedModel::open(&path).unwrap();
        let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
        let weights = WeightStore::new(&mapped);
        let metal = MetalBackend::new().unwrap();
        let metal_ops = metal.metal_ops().unwrap();
        let cpu = crate::backend::cpu::CpuBackend;

        let n_tokens = 8usize;
        let dim = cfg.embedding_dim as usize;
        let n_heads = cfg.n_heads as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let nt = n_tokens as u32;
        let prefix = "blk.0";

        let input_hidden = staged_hidden(n_tokens * dim);
        let mut expected_hidden_before_moe = input_hidden.clone();
        let mut norm_buf = vec![0.0f32; n_tokens * dim];
        let mut q_buf = vec![0.0f32; n_tokens * q_dim];
        let mut k_buf = vec![0.0f32; n_tokens * kv_dim];
        let mut v_buf = vec![0.0f32; n_tokens * kv_dim];
        let mut attn_out = vec![0.0f32; n_tokens * q_dim];
        let mut proj_buf = vec![0.0f32; n_tokens * dim];

        let attn_norm_w = weights
            .f32_slice(&format!("{prefix}.attn_norm.weight"))
            .unwrap();
        for token in 0..n_tokens {
            let hidden = &expected_hidden_before_moe[token * dim..(token + 1) * dim];
            let norm = &mut norm_buf[token * dim..(token + 1) * dim];
            crate::compute::rms_norm::rms_norm_out(hidden, attn_norm_w, norm, cfg.rms_norm_eps);
        }

        let (wq_raw, wq_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_q.weight"))
            .unwrap();
        let (wk_raw, wk_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_k.weight"))
            .unwrap();
        let (wv_raw, wv_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_v.weight"))
            .unwrap();
        for token in 0..n_tokens {
            let norm = &norm_buf[token * dim..(token + 1) * dim];
            cpu.dequant_matmul(
                wq_raw,
                wq_dtype,
                norm,
                &mut q_buf[token * q_dim..(token + 1) * q_dim],
                q_dim,
                1,
                dim,
            );
            cpu.dequant_matmul(
                wk_raw,
                wk_dtype,
                norm,
                &mut k_buf[token * kv_dim..(token + 1) * kv_dim],
                kv_dim,
                1,
                dim,
            );
            cpu.dequant_matmul(
                wv_raw,
                wv_dtype,
                norm,
                &mut v_buf[token * kv_dim..(token + 1) * kv_dim],
                kv_dim,
                1,
                dim,
            );
        }

        if let Some(norm_weights) =
            crate::model::shared::maybe_attention_qk_norm_weights(&weights, prefix).unwrap()
        {
            for token in 0..n_tokens {
                crate::model::shared::apply_attention_qk_norm(
                    &mut q_buf[token * q_dim..(token + 1) * q_dim],
                    &mut k_buf[token * kv_dim..(token + 1) * kv_dim],
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    norm_weights,
                    cfg.rms_norm_eps,
                );
            }
        }

        for token in 0..n_tokens {
            crate::compute::rope::apply_rope_multi_head_neox_partial_scaled(
                &mut q_buf[token * q_dim..(token + 1) * q_dim],
                &mut k_buf[token * kv_dim..(token + 1) * kv_dim],
                n_heads,
                n_kv_heads,
                head_dim,
                head_dim,
                token as f32,
                cfg.rope_freq_base,
            );
        }

        let attn_params =
            crate::compute::attention::AttentionParams::new(n_heads, n_kv_heads, head_dim);
        crate::compute::attention::multi_head_attention_prefill(
            &q_buf,
            &k_buf,
            &v_buf,
            &mut attn_out,
            n_tokens,
            &attn_params,
        );

        let (wo_raw, wo_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_output.weight"))
            .unwrap();
        for token in 0..n_tokens {
            cpu.dequant_matmul(
                wo_raw,
                wo_dtype,
                &attn_out[token * q_dim..(token + 1) * q_dim],
                &mut proj_buf[token * dim..(token + 1) * dim],
                dim,
                1,
                q_dim,
            );
        }
        Qwen3MoeForward::parallel_elementwise_add(&mut expected_hidden_before_moe, &proj_buf);

        let metal_model =
            InferenceModel::with_backend(cfg.clone(), Box::new(MetalBackend::new().unwrap()))
                .unwrap();
        let mut metal_kv = metal_model.create_model_kv_for_weights(&weights);
        let gpu_kv = metal_kv.as_gpu_mut().unwrap();
        gpu_kv.ensure_capacity(&metal_ops.device, n_tokens).unwrap();

        metal_ops.init_batch_scratches(&cfg, n_tokens);
        if !metal_ops.has_cached_model_keys() {
            Qwen3MoeForward::build_cached_model_keys(metal_ops, &weights, &cfg).unwrap();
        }
        let cached_guard = metal_ops.cached_model_keys();
        let cached = cached_guard.as_ref().unwrap();
        let layer = &cached.layers[0];
        let weight_cache = metal_ops.lock_weight_cache();
        let fused_qkv_cache = metal_ops.lock_fused_qkv_weight_cache();

        let attn_norm = weight_cache.get(&layer.attn_norm).unwrap();
        let wq = weight_cache.get(&layer.wq).unwrap();
        let wk = weight_cache.get(&layer.wk).unwrap();
        let wv = weight_cache.get(&layer.wv).unwrap();
        let wo = weight_cache.get(&layer.wo).unwrap();
        let qn = layer.attn_q_norm.map(|key| weight_cache.get(&key).unwrap());
        let kn = layer.attn_k_norm.map(|key| weight_cache.get(&key).unwrap());
        let kv_k = gpu_kv.k_buffer(0);
        let kv_v = gpu_kv.v_buffer(0);
        let kv_stride = gpu_kv.kv_stride_for_layer(0) as u32;
        let has_q5k = crate::model::shared::gpu_prefill_uses_q5k(&weights);
        let q5k_small_n = crate::model::shared::gpu_prefill_q5k_small_n_auto_eligible(&weights);
        let prefill_plan = crate::model::execution_plan::DecodeExecutionPlan::gemma3_prefill(
            metal_ops,
            gpu_kv,
            nt,
            has_q5k,
            q5k_small_n,
        );

        expected_hidden_before_moe.copy_from_slice(&input_hidden);
        q_buf.fill(0.0);
        k_buf.fill(0.0);
        v_buf.fill(0.0);
        attn_out.fill(0.0);
        proj_buf.fill(0.0);

        let qkv_input = if prefill_plan.use_f16_batch_io {
            norm_buf
                .iter()
                .copied()
                .map(|v| half::f16::from_f32(v).to_f32())
                .collect::<Vec<_>>()
        } else {
            norm_buf.clone()
        };
        for token in 0..n_tokens {
            let norm = &qkv_input[token * dim..(token + 1) * dim];
            cpu.dequant_matmul(
                wq_raw,
                wq_dtype,
                norm,
                &mut q_buf[token * q_dim..(token + 1) * q_dim],
                q_dim,
                1,
                dim,
            );
            cpu.dequant_matmul(
                wk_raw,
                wk_dtype,
                norm,
                &mut k_buf[token * kv_dim..(token + 1) * kv_dim],
                kv_dim,
                1,
                dim,
            );
            cpu.dequant_matmul(
                wv_raw,
                wv_dtype,
                norm,
                &mut v_buf[token * kv_dim..(token + 1) * kv_dim],
                kv_dim,
                1,
                dim,
            );
        }

        if let Some(norm_weights) =
            crate::model::shared::maybe_attention_qk_norm_weights(&weights, prefix).unwrap()
        {
            for token in 0..n_tokens {
                crate::model::shared::apply_attention_qk_norm(
                    &mut q_buf[token * q_dim..(token + 1) * q_dim],
                    &mut k_buf[token * kv_dim..(token + 1) * kv_dim],
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    norm_weights,
                    cfg.rms_norm_eps,
                );
            }
        }

        for token in 0..n_tokens {
            crate::compute::rope::apply_rope_multi_head_neox_partial_scaled(
                &mut q_buf[token * q_dim..(token + 1) * q_dim],
                &mut k_buf[token * kv_dim..(token + 1) * kv_dim],
                n_heads,
                n_kv_heads,
                head_dim,
                head_dim,
                token as f32,
                cfg.rope_freq_base,
            );
        }

        let attn_params =
            crate::compute::attention::AttentionParams::new(n_heads, n_kv_heads, head_dim);
        crate::compute::attention::multi_head_attention_prefill(
            &q_buf,
            &k_buf,
            &v_buf,
            &mut attn_out,
            n_tokens,
            &attn_params,
        );

        let wo_input = if matches!(
            prefill_plan.wo_input,
            crate::model::execution_plan::PrefillWoInputPlan::MatmulScratchF16
        ) {
            attn_out
                .iter()
                .copied()
                .map(|v| half::f16::from_f32(v).to_f32())
                .collect::<Vec<_>>()
        } else {
            attn_out.clone()
        };
        for token in 0..n_tokens {
            cpu.dequant_matmul(
                wo_raw,
                wo_dtype,
                &wo_input[token * q_dim..(token + 1) * q_dim],
                &mut proj_buf[token * dim..(token + 1) * dim],
                dim,
                1,
                q_dim,
            );
        }
        Qwen3MoeForward::parallel_elementwise_add(&mut expected_hidden_before_moe, &proj_buf);

        let fused_qkv_m = q_dim + 2 * kv_dim;
        let fused_qkv_buf = if prefill_plan.use_fused_qkv
            && layer.wq_dtype == layer.wk_dtype
            && layer.wq_dtype == layer.wv_dtype
            && matches!(layer.wq_dtype, GgmlType::Q4K | GgmlType::Q6K)
        {
            fused_qkv_cache.get(&(layer.wq, layer.wk, layer.wv))
        } else {
            None
        };

        {
            let mut bs_guard = metal_ops.batch_scratches();
            let bs = bs_guard.as_mut().unwrap();
            unsafe {
                bs.hidden.as_mut_slice::<f32>()[..n_tokens * dim].copy_from_slice(&input_hidden);
            }

            metal_ops
                .device
                .execute_sync(|encoder| {
                    metal_ops.elementwise.encode_rms_norm_out_batch(
                        encoder,
                        &bs.hidden,
                        attn_norm,
                        &bs.norm_buf,
                        dim as u32,
                        nt,
                        cfg.rms_norm_eps,
                    );
                    ax_engine_metal::barrier_buffers(encoder);

                    let use_f16 = prefill_plan.use_f16_batch_io;
                    if let Some(fused_w) = fused_qkv_buf {
                        if use_f16 {
                            metal_ops.elementwise.encode_cast_f32_to_f16(
                                encoder,
                                &bs.norm_buf,
                                &bs.matmul_in_f16,
                                nt * dim as u32,
                            );
                            ax_engine_metal::barrier_buffers(encoder);
                            crate::model::shared::encode_dequant_batch_f16in(
                                metal_ops,
                                encoder,
                                fused_w,
                                &bs.matmul_in_f16,
                                &bs.qkv_buf,
                                fused_qkv_m as u32,
                                nt,
                                dim as u32,
                                layer.wq_dtype,
                            );
                        } else {
                            crate::model::shared::encode_dequant_batch(
                                &metal_ops.dequant,
                                &metal_ops.elementwise,
                                encoder,
                                fused_w,
                                &bs.norm_buf,
                                &bs.qkv_buf,
                                &bs.matmul_in_f16,
                                fused_qkv_m as u32,
                                nt,
                                dim as u32,
                                layer.wq_dtype,
                                false,
                                prefill_plan.use_batch_simd,
                                prefill_plan.q5k_prefill_small_n,
                            );
                        }
                        ax_engine_metal::barrier_buffers(encoder);
                    } else if use_f16 {
                        metal_ops.elementwise.encode_cast_f32_to_f16(
                            encoder,
                            &bs.norm_buf,
                            &bs.matmul_in_f16,
                            nt * dim as u32,
                        );
                        ax_engine_metal::barrier_buffers(encoder);
                        crate::model::shared::encode_dequant_batch_f16in(
                            metal_ops,
                            encoder,
                            wq,
                            &bs.matmul_in_f16,
                            &bs.q_buf,
                            q_dim as u32,
                            nt,
                            dim as u32,
                            layer.wq_dtype,
                        );
                        crate::model::shared::encode_dequant_batch_f16in(
                            metal_ops,
                            encoder,
                            wk,
                            &bs.matmul_in_f16,
                            &bs.k_buf,
                            kv_dim as u32,
                            nt,
                            dim as u32,
                            layer.wk_dtype,
                        );
                        crate::model::shared::encode_dequant_batch_f16in(
                            metal_ops,
                            encoder,
                            wv,
                            &bs.matmul_in_f16,
                            &bs.v_buf,
                            kv_dim as u32,
                            nt,
                            dim as u32,
                            layer.wv_dtype,
                        );
                        ax_engine_metal::barrier_buffers(encoder);
                    } else {
                        crate::model::shared::encode_dequant_batch(
                            &metal_ops.dequant,
                            &metal_ops.elementwise,
                            encoder,
                            wq,
                            &bs.norm_buf,
                            &bs.q_buf,
                            &bs.matmul_in_f16,
                            q_dim as u32,
                            nt,
                            dim as u32,
                            layer.wq_dtype,
                            false,
                            prefill_plan.use_batch_simd,
                            prefill_plan.q5k_prefill_small_n,
                        );
                        crate::model::shared::encode_dequant_batch(
                            &metal_ops.dequant,
                            &metal_ops.elementwise,
                            encoder,
                            wk,
                            &bs.norm_buf,
                            &bs.k_buf,
                            &bs.matmul_in_f16,
                            kv_dim as u32,
                            nt,
                            dim as u32,
                            layer.wk_dtype,
                            false,
                            prefill_plan.use_batch_simd,
                            prefill_plan.q5k_prefill_small_n,
                        );
                        crate::model::shared::encode_dequant_batch(
                            &metal_ops.dequant,
                            &metal_ops.elementwise,
                            encoder,
                            wv,
                            &bs.norm_buf,
                            &bs.v_buf,
                            &bs.matmul_in_f16,
                            kv_dim as u32,
                            nt,
                            dim as u32,
                            layer.wv_dtype,
                            false,
                            prefill_plan.use_batch_simd,
                            prefill_plan.q5k_prefill_small_n,
                        );
                        ax_engine_metal::barrier_buffers(encoder);
                    }

                    let fused_qkv_post = fused_qkv_buf.is_some() && qn.is_some() && kn.is_some();
                    if fused_qkv_post {
                        metal_ops
                            .elementwise
                            .encode_qkv_split_qk_norm_rope_append_kv_batch(
                                encoder,
                                &bs.qkv_buf,
                                &bs.q_buf,
                                &bs.k_buf,
                                &bs.v_buf,
                                qn.unwrap(),
                                kn.unwrap(),
                                kv_k,
                                kv_v,
                                prefill_plan.kv_f16,
                                nt,
                                n_heads as u32,
                                n_kv_heads as u32,
                                head_dim as u32,
                                cfg.rms_norm_eps,
                                0.0,
                                1.0,
                                cfg.rope_freq_base,
                                0,
                                kv_stride,
                            );
                        ax_engine_metal::barrier_buffers(encoder);
                    } else {
                        if fused_qkv_buf.is_some() {
                            metal_ops.elementwise.encode_qkv_split_batch(
                                encoder,
                                &bs.qkv_buf,
                                &bs.q_buf,
                                &bs.k_buf,
                                &bs.v_buf,
                                nt,
                                q_dim as u32,
                                kv_dim as u32,
                            );
                            ax_engine_metal::barrier_buffers(encoder);
                        }

                        if let (Some(qn_buf), Some(kn_buf)) = (qn, kn) {
                            metal_ops.elementwise.encode_per_head_rms_norm_batch(
                                encoder,
                                &bs.q_buf,
                                qn_buf,
                                nt,
                                n_heads as u32,
                                head_dim as u32,
                                cfg.rms_norm_eps,
                            );
                            metal_ops.elementwise.encode_per_head_rms_norm_batch(
                                encoder,
                                &bs.k_buf,
                                kn_buf,
                                nt,
                                n_kv_heads as u32,
                                head_dim as u32,
                                cfg.rms_norm_eps,
                            );
                            ax_engine_metal::barrier_buffers(encoder);
                        }

                        metal_ops.elementwise.encode_rope_batch_neox_partial(
                            encoder,
                            &bs.q_buf,
                            &bs.k_buf,
                            nt,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            head_dim as u32,
                            0.0,
                            1.0,
                            cfg.rope_freq_base,
                        );
                        ax_engine_metal::barrier_buffers(encoder);

                        metal_ops.elementwise.encode_kv_append_batch_pair(
                            encoder,
                            &bs.k_buf,
                            &bs.v_buf,
                            kv_k,
                            kv_v,
                            prefill_plan.kv_f16,
                            0,
                            kv_stride,
                            kv_dim as u32,
                            nt,
                        );
                        ax_engine_metal::barrier_buffers(encoder);
                    }

                    metal_ops
                        .attention
                        .encode_attention_prefill_cached_with_config(
                            encoder,
                            &bs.q_buf,
                            kv_k,
                            kv_v,
                            &bs.attn_out,
                            prefill_plan.kv_f16,
                            nt,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            0,
                            0,
                            prefill_plan.attention_dispatch,
                        );
                    ax_engine_metal::barrier_buffers(encoder);

                    crate::model::shared::encode_dequant_batch(
                        &metal_ops.dequant,
                        &metal_ops.elementwise,
                        encoder,
                        wo,
                        &bs.attn_out,
                        &bs.proj_buf,
                        &bs.matmul_in_f16,
                        dim as u32,
                        nt,
                        q_dim as u32,
                        layer.wo_dtype,
                        prefill_plan.use_f16_batch_io,
                        prefill_plan.use_batch_simd,
                        prefill_plan.q5k_prefill_small_n,
                    );
                    ax_engine_metal::barrier_buffers(encoder);

                    metal_ops.elementwise.encode_elementwise_add_batch(
                        encoder,
                        &bs.hidden,
                        &bs.proj_buf,
                        dim as u32,
                        nt,
                    );
                    Ok(())
                })
                .unwrap();
        }

        let (
            actual_hidden_before_moe,
            actual_q_buf,
            actual_k_buf,
            actual_v_buf,
            actual_attn_out,
            actual_proj_buf,
        ) = {
            let bs_guard = metal_ops.batch_scratches();
            let bs = bs_guard.as_ref().unwrap();
            unsafe {
                (
                    bs.hidden.as_slice::<f32>()[..n_tokens * dim].to_vec(),
                    bs.q_buf.as_slice::<f32>()[..n_tokens * q_dim].to_vec(),
                    bs.k_buf.as_slice::<f32>()[..n_tokens * kv_dim].to_vec(),
                    bs.v_buf.as_slice::<f32>()[..n_tokens * kv_dim].to_vec(),
                    bs.attn_out.as_slice::<f32>()[..n_tokens * q_dim].to_vec(),
                    bs.proj_buf.as_slice::<f32>()[..n_tokens * dim].to_vec(),
                )
            }
        };
        let diff = max_abs_diff(&actual_hidden_before_moe, &expected_hidden_before_moe);
        let scale = expected_hidden_before_moe
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max)
            .max(1.0);
        let q_diff = max_abs_diff(&actual_q_buf, &q_buf);
        let k_diff = max_abs_diff(&actual_k_buf, &k_buf);
        let v_diff = max_abs_diff(&actual_v_buf, &v_buf);
        let attn_diff = max_abs_diff(&actual_attn_out, &attn_out);
        let proj_diff = max_abs_diff(&actual_proj_buf, &proj_buf);
        assert!(
            diff / scale < 5e-2,
            "{model_file} layer0 projected hidden-before-moe GPU stage mismatch: rel_diff={} max_diff={} q_diff={} k_diff={} v_diff={} attn_diff={} proj_diff={} actual[0..8]={:?} expected[0..8]={:?}",
            diff / scale,
            diff,
            q_diff,
            k_diff,
            v_diff,
            attn_diff,
            proj_diff,
            &actual_hidden_before_moe[..8],
            &expected_hidden_before_moe[..8],
        );
    }
}

#[test]
fn test_real_qwen3_coder_q8_0_layer0_qk_norm_batch_matches_cpu() {
    let path = workspace_model_path("Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();
    let cpu = crate::backend::cpu::CpuBackend;

    let n_tokens = 8usize;
    let dim = cfg.embedding_dim as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let prefix = "blk.0";

    let input_hidden = staged_hidden(n_tokens * dim);
    let mut norm_buf = vec![0.0f32; n_tokens * dim];
    let mut q_pre = vec![0.0f32; n_tokens * q_dim];
    let mut k_pre = vec![0.0f32; n_tokens * kv_dim];

    let attn_norm_w = weights
        .f32_slice(&format!("{prefix}.attn_norm.weight"))
        .unwrap();
    for token in 0..n_tokens {
        let hidden = &input_hidden[token * dim..(token + 1) * dim];
        let norm = &mut norm_buf[token * dim..(token + 1) * dim];
        crate::compute::rms_norm::rms_norm_out(hidden, attn_norm_w, norm, cfg.rms_norm_eps);
    }

    let (wq_raw, wq_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_q.weight"))
        .unwrap();
    let (wk_raw, wk_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_k.weight"))
        .unwrap();
    for token in 0..n_tokens {
        let norm = &norm_buf[token * dim..(token + 1) * dim];
        cpu.dequant_matmul(
            wq_raw,
            wq_dtype,
            norm,
            &mut q_pre[token * q_dim..(token + 1) * q_dim],
            q_dim,
            1,
            dim,
        );
        cpu.dequant_matmul(
            wk_raw,
            wk_dtype,
            norm,
            &mut k_pre[token * kv_dim..(token + 1) * kv_dim],
            kv_dim,
            1,
            dim,
        );
    }

    let qk_norm_weights = crate::model::shared::maybe_attention_qk_norm_weights(&weights, prefix)
        .unwrap()
        .unwrap();
    let mut expected_q = q_pre.clone();
    let mut expected_k = k_pre.clone();
    for token in 0..n_tokens {
        crate::model::shared::apply_attention_qk_norm(
            &mut expected_q[token * q_dim..(token + 1) * q_dim],
            &mut expected_k[token * kv_dim..(token + 1) * kv_dim],
            n_heads,
            n_kv_heads,
            head_dim,
            qk_norm_weights,
            cfg.rms_norm_eps,
        );
    }

    let q_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), &q_pre).unwrap();
    let k_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), &k_pre).unwrap();
    let qn_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), qk_norm_weights.q)
            .unwrap();
    let kn_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), qk_norm_weights.k)
            .unwrap();

    metal_ops
        .device
        .execute_sync(|encoder| {
            metal_ops.elementwise.encode_per_head_rms_norm_batch(
                encoder,
                &q_buf,
                &qn_buf,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                cfg.rms_norm_eps,
            );
            ax_engine_metal::barrier_buffers(encoder);
            metal_ops.elementwise.encode_per_head_rms_norm_batch(
                encoder,
                &k_buf,
                &kn_buf,
                n_tokens as u32,
                n_kv_heads as u32,
                head_dim as u32,
                cfg.rms_norm_eps,
            );
            Ok(())
        })
        .unwrap();

    let actual_q = unsafe { q_buf.as_slice::<f32>()[..n_tokens * q_dim].to_vec() };
    let actual_k = unsafe { k_buf.as_slice::<f32>()[..n_tokens * kv_dim].to_vec() };
    let q_diff = max_abs_diff(&actual_q, &expected_q);
    let k_diff = max_abs_diff(&actual_k, &expected_k);
    let q_scale = expected_q
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    let k_scale = expected_k
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        q_diff / q_scale < 1e-3 && k_diff / k_scale < 1e-3,
        "real Qwen3-Coder Q8_0 qk norm batch mismatch: q_rel={} q_diff={} k_rel={} k_diff={}",
        q_diff / q_scale,
        q_diff,
        k_diff / k_scale,
        k_diff,
    );
}

#[test]
fn test_real_qwen3_coder_q8_0_layer0_attn_norm_batch_matches_cpu() {
    let path = workspace_model_path("Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();

    let n_tokens = 8usize;
    let dim = cfg.embedding_dim as usize;
    let prefix = "blk.0";

    let input_hidden = staged_hidden(n_tokens * dim);
    let attn_norm_w = weights
        .f32_slice(&format!("{prefix}.attn_norm.weight"))
        .unwrap();
    let mut expected = vec![0.0f32; n_tokens * dim];
    for token in 0..n_tokens {
        crate::compute::rms_norm::rms_norm_out(
            &input_hidden[token * dim..(token + 1) * dim],
            attn_norm_w,
            &mut expected[token * dim..(token + 1) * dim],
            cfg.rms_norm_eps,
        );
    }

    let hidden_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), &input_hidden).unwrap();
    let weight_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), attn_norm_w).unwrap();
    let out_buf = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        n_tokens * dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    metal_ops
        .device
        .execute_sync(|encoder| {
            metal_ops.elementwise.encode_rms_norm_out_batch(
                encoder,
                &hidden_buf,
                &weight_buf,
                &out_buf,
                dim as u32,
                n_tokens as u32,
                cfg.rms_norm_eps,
            );
            Ok(())
        })
        .unwrap();

    let actual = unsafe { out_buf.as_slice::<f32>()[..n_tokens * dim].to_vec() };
    let diff = max_abs_diff(&actual, &expected);
    let scale = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 1e-3,
        "real Qwen3-Coder Q8_0 attn norm batch mismatch: rel_diff={} max_diff={diff}",
        diff / scale,
    );
}

#[test]
fn test_real_qwen3_coder_q8_0_layer0_rope_batch_matches_cpu_after_qk_norm() {
    let path = workspace_model_path("Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();
    let cpu = crate::backend::cpu::CpuBackend;

    let n_tokens = 8usize;
    let dim = cfg.embedding_dim as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let prefix = "blk.0";

    let input_hidden = staged_hidden(n_tokens * dim);
    let mut norm_buf = vec![0.0f32; n_tokens * dim];
    let mut q_normed = vec![0.0f32; n_tokens * q_dim];
    let mut k_normed = vec![0.0f32; n_tokens * kv_dim];

    let attn_norm_w = weights
        .f32_slice(&format!("{prefix}.attn_norm.weight"))
        .unwrap();
    for token in 0..n_tokens {
        let hidden = &input_hidden[token * dim..(token + 1) * dim];
        let norm = &mut norm_buf[token * dim..(token + 1) * dim];
        crate::compute::rms_norm::rms_norm_out(hidden, attn_norm_w, norm, cfg.rms_norm_eps);
    }

    let (wq_raw, wq_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_q.weight"))
        .unwrap();
    let (wk_raw, wk_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_k.weight"))
        .unwrap();
    for token in 0..n_tokens {
        let norm = &norm_buf[token * dim..(token + 1) * dim];
        cpu.dequant_matmul(
            wq_raw,
            wq_dtype,
            norm,
            &mut q_normed[token * q_dim..(token + 1) * q_dim],
            q_dim,
            1,
            dim,
        );
        cpu.dequant_matmul(
            wk_raw,
            wk_dtype,
            norm,
            &mut k_normed[token * kv_dim..(token + 1) * kv_dim],
            kv_dim,
            1,
            dim,
        );
    }

    let qk_norm_weights = crate::model::shared::maybe_attention_qk_norm_weights(&weights, prefix)
        .unwrap()
        .unwrap();
    for token in 0..n_tokens {
        crate::model::shared::apply_attention_qk_norm(
            &mut q_normed[token * q_dim..(token + 1) * q_dim],
            &mut k_normed[token * kv_dim..(token + 1) * kv_dim],
            n_heads,
            n_kv_heads,
            head_dim,
            qk_norm_weights,
            cfg.rms_norm_eps,
        );
    }

    let mut expected_q = q_normed.clone();
    let mut expected_k = k_normed.clone();
    for token in 0..n_tokens {
        crate::compute::rope::apply_rope_multi_head_neox_partial_scaled(
            &mut expected_q[token * q_dim..(token + 1) * q_dim],
            &mut expected_k[token * kv_dim..(token + 1) * kv_dim],
            n_heads,
            n_kv_heads,
            head_dim,
            head_dim,
            token as f32,
            cfg.rope_freq_base,
        );
    }

    let q_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), &q_normed).unwrap();
    let k_buf =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), &k_normed).unwrap();
    metal_ops
        .device
        .execute_sync(|encoder| {
            metal_ops.elementwise.encode_rope_batch_neox_partial(
                encoder,
                &q_buf,
                &k_buf,
                n_tokens as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                head_dim as u32,
                0.0,
                1.0,
                cfg.rope_freq_base,
            );
            Ok(())
        })
        .unwrap();

    let actual_q = unsafe { q_buf.as_slice::<f32>()[..n_tokens * q_dim].to_vec() };
    let actual_k = unsafe { k_buf.as_slice::<f32>()[..n_tokens * kv_dim].to_vec() };
    let q_diff = max_abs_diff(&actual_q, &expected_q);
    let k_diff = max_abs_diff(&actual_k, &expected_k);
    let q_scale = expected_q
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    let k_scale = expected_k
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        q_diff / q_scale < 1e-3 && k_diff / k_scale < 1e-3,
        "real Qwen3-Coder Q8_0 rope batch mismatch after qk norm: q_rel={} q_diff={} k_rel={} k_diff={}",
        q_diff / q_scale,
        q_diff,
        k_diff / k_scale,
        k_diff,
    );
}

#[test]
fn test_real_qwen3_coder_q8_0_layer0_prefill_qkv_projection_matches_cpu_f16_input() {
    let path = workspace_model_path("Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();
    let cpu = crate::backend::cpu::CpuBackend;

    let n_tokens = 8usize;
    let dim = cfg.embedding_dim as usize;
    let q_dim = cfg.n_heads as usize * cfg.head_dim as usize;
    let kv_dim = cfg.n_kv_heads as usize * cfg.head_dim as usize;
    let prefix = "blk.0";

    let input_hidden = staged_hidden(n_tokens * dim);
    let mut norm_buf = vec![0.0f32; n_tokens * dim];
    let attn_norm_w = weights
        .f32_slice(&format!("{prefix}.attn_norm.weight"))
        .unwrap();
    for token in 0..n_tokens {
        let hidden = &input_hidden[token * dim..(token + 1) * dim];
        let norm = &mut norm_buf[token * dim..(token + 1) * dim];
        crate::compute::rms_norm::rms_norm_out(hidden, attn_norm_w, norm, cfg.rms_norm_eps);
    }

    let norm_f16_cpu: Vec<f32> = norm_buf
        .iter()
        .copied()
        .map(|v| half::f16::from_f32(v).to_f32())
        .collect();

    let (wq_raw, wq_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_q.weight"))
        .unwrap();
    let (wk_raw, wk_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_k.weight"))
        .unwrap();
    let (wv_raw, wv_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_v.weight"))
        .unwrap();
    assert_eq!(wq_dtype, GgmlType::Q8_0);
    assert_eq!(wk_dtype, GgmlType::Q8_0);
    assert_eq!(wv_dtype, GgmlType::Q8_0);

    let mut expected_q = vec![0.0f32; n_tokens * q_dim];
    let mut expected_k = vec![0.0f32; n_tokens * kv_dim];
    let mut expected_v = vec![0.0f32; n_tokens * kv_dim];
    for token in 0..n_tokens {
        let norm = &norm_f16_cpu[token * dim..(token + 1) * dim];
        cpu.dequant_matmul(
            wq_raw,
            wq_dtype,
            norm,
            &mut expected_q[token * q_dim..(token + 1) * q_dim],
            q_dim,
            1,
            dim,
        );
        cpu.dequant_matmul(
            wk_raw,
            wk_dtype,
            norm,
            &mut expected_k[token * kv_dim..(token + 1) * kv_dim],
            kv_dim,
            1,
            dim,
        );
        cpu.dequant_matmul(
            wv_raw,
            wv_dtype,
            norm,
            &mut expected_v[token * kv_dim..(token + 1) * kv_dim],
            kv_dim,
            1,
            dim,
        );
    }

    let norm_buf_gpu =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), &norm_buf).unwrap();
    let matmul_in_f16 = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        n_tokens * dim * std::mem::size_of::<half::f16>(),
    )
    .unwrap();
    let q_buf = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        n_tokens * q_dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let k_buf = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        n_tokens * kv_dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let v_buf = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        n_tokens * kv_dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let wq_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), wq_raw).unwrap();
    let wk_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), wk_raw).unwrap();
    let wv_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), wv_raw).unwrap();

    metal_ops
        .device
        .execute_sync(|encoder| {
            metal_ops.elementwise.encode_cast_f32_to_f16(
                encoder,
                &norm_buf_gpu,
                &matmul_in_f16,
                (n_tokens * dim) as u32,
            );
            ax_engine_metal::barrier_buffers(encoder);
            crate::model::shared::encode_dequant_batch_f16in(
                metal_ops,
                encoder,
                &wq_buf,
                &matmul_in_f16,
                &q_buf,
                q_dim as u32,
                n_tokens as u32,
                dim as u32,
                wq_dtype,
            );
            crate::model::shared::encode_dequant_batch_f16in(
                metal_ops,
                encoder,
                &wk_buf,
                &matmul_in_f16,
                &k_buf,
                kv_dim as u32,
                n_tokens as u32,
                dim as u32,
                wk_dtype,
            );
            crate::model::shared::encode_dequant_batch_f16in(
                metal_ops,
                encoder,
                &wv_buf,
                &matmul_in_f16,
                &v_buf,
                kv_dim as u32,
                n_tokens as u32,
                dim as u32,
                wv_dtype,
            );
            Ok(())
        })
        .unwrap();

    let actual_q = unsafe { q_buf.as_slice::<f32>()[..n_tokens * q_dim].to_vec() };
    let actual_k = unsafe { k_buf.as_slice::<f32>()[..n_tokens * kv_dim].to_vec() };
    let actual_v = unsafe { v_buf.as_slice::<f32>()[..n_tokens * kv_dim].to_vec() };

    for (label, actual, expected) in [
        ("q", &actual_q, &expected_q),
        ("k", &actual_k, &expected_k),
        ("v", &actual_v, &expected_v),
    ] {
        let diff = max_abs_diff(actual, expected);
        let scale = expected
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max)
            .max(1.0);
        assert!(
            diff / scale < 1e-3,
            "real Qwen3-Coder Q8_0 prefill {label} projection mismatch on f16 input: rel_diff={} max_diff={diff}",
            diff / scale,
        );
    }
}

#[test]
fn test_real_qwen3_coder_q8_0_layer0_prefill_qkv_projection_matches_cpu_f32_input() {
    let path = workspace_model_path("Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();
    let cpu = crate::backend::cpu::CpuBackend;

    let n_tokens = 8usize;
    let dim = cfg.embedding_dim as usize;
    let q_dim = cfg.n_heads as usize * cfg.head_dim as usize;
    let kv_dim = cfg.n_kv_heads as usize * cfg.head_dim as usize;
    let prefix = "blk.0";

    let input_hidden = staged_hidden(n_tokens * dim);
    let mut norm_buf = vec![0.0f32; n_tokens * dim];
    let attn_norm_w = weights
        .f32_slice(&format!("{prefix}.attn_norm.weight"))
        .unwrap();
    for token in 0..n_tokens {
        let hidden = &input_hidden[token * dim..(token + 1) * dim];
        let norm = &mut norm_buf[token * dim..(token + 1) * dim];
        crate::compute::rms_norm::rms_norm_out(hidden, attn_norm_w, norm, cfg.rms_norm_eps);
    }

    let (wq_raw, wq_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_q.weight"))
        .unwrap();
    let (wk_raw, wk_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_k.weight"))
        .unwrap();
    let (wv_raw, wv_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_v.weight"))
        .unwrap();
    assert_eq!(wq_dtype, GgmlType::Q8_0);
    assert_eq!(wk_dtype, GgmlType::Q8_0);
    assert_eq!(wv_dtype, GgmlType::Q8_0);

    let mut expected_q = vec![0.0f32; n_tokens * q_dim];
    let mut expected_k = vec![0.0f32; n_tokens * kv_dim];
    let mut expected_v = vec![0.0f32; n_tokens * kv_dim];
    for token in 0..n_tokens {
        let norm = &norm_buf[token * dim..(token + 1) * dim];
        cpu.dequant_matmul(
            wq_raw,
            wq_dtype,
            norm,
            &mut expected_q[token * q_dim..(token + 1) * q_dim],
            q_dim,
            1,
            dim,
        );
        cpu.dequant_matmul(
            wk_raw,
            wk_dtype,
            norm,
            &mut expected_k[token * kv_dim..(token + 1) * kv_dim],
            kv_dim,
            1,
            dim,
        );
        cpu.dequant_matmul(
            wv_raw,
            wv_dtype,
            norm,
            &mut expected_v[token * kv_dim..(token + 1) * kv_dim],
            kv_dim,
            1,
            dim,
        );
    }

    let norm_buf_gpu =
        ax_engine_metal::MetalBuffer::from_slice(metal_ops.device.device(), &norm_buf).unwrap();
    let matmul_in_f16 = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        n_tokens * dim * std::mem::size_of::<half::f16>(),
    )
    .unwrap();
    let q_buf = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        n_tokens * q_dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let k_buf = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        n_tokens * kv_dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let v_buf = ax_engine_metal::MetalBuffer::new(
        metal_ops.device.device(),
        n_tokens * kv_dim * std::mem::size_of::<f32>(),
    )
    .unwrap();
    let wq_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), wq_raw).unwrap();
    let wk_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), wk_raw).unwrap();
    let wv_buf =
        ax_engine_metal::MetalBuffer::from_bytes(metal_ops.device.device(), wv_raw).unwrap();

    metal_ops
        .device
        .execute_sync(|encoder| {
            crate::model::shared::encode_dequant_batch(
                &metal_ops.dequant,
                &metal_ops.elementwise,
                encoder,
                &wq_buf,
                &norm_buf_gpu,
                &q_buf,
                &matmul_in_f16,
                q_dim as u32,
                n_tokens as u32,
                dim as u32,
                wq_dtype,
                false,
                false,
                false,
            );
            crate::model::shared::encode_dequant_batch(
                &metal_ops.dequant,
                &metal_ops.elementwise,
                encoder,
                &wk_buf,
                &norm_buf_gpu,
                &k_buf,
                &matmul_in_f16,
                kv_dim as u32,
                n_tokens as u32,
                dim as u32,
                wk_dtype,
                false,
                false,
                false,
            );
            crate::model::shared::encode_dequant_batch(
                &metal_ops.dequant,
                &metal_ops.elementwise,
                encoder,
                &wv_buf,
                &norm_buf_gpu,
                &v_buf,
                &matmul_in_f16,
                kv_dim as u32,
                n_tokens as u32,
                dim as u32,
                wv_dtype,
                false,
                false,
                false,
            );
            Ok(())
        })
        .unwrap();

    for (label, actual, expected) in [
        (
            "q",
            unsafe { q_buf.as_slice::<f32>()[..n_tokens * q_dim].to_vec() },
            expected_q,
        ),
        (
            "k",
            unsafe { k_buf.as_slice::<f32>()[..n_tokens * kv_dim].to_vec() },
            expected_k,
        ),
        (
            "v",
            unsafe { v_buf.as_slice::<f32>()[..n_tokens * kv_dim].to_vec() },
            expected_v,
        ),
    ] {
        let diff = max_abs_diff(&actual, &expected);
        let scale = expected
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max)
            .max(1.0);
        assert!(
            diff / scale < 1e-3,
            "real Qwen3-Coder Q8_0 prefill {label} projection mismatch on f32 input: rel_diff={} max_diff={diff}",
            diff / scale,
        );
    }
}

#[test]
fn test_real_qwen3_coder_decode_plan_summary_reports_runtime_barrier_policy() {
    for (model_file, barrier_fragment) in [
        ("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", "barriers=smart"),
        ("Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf", "barriers=smart"),
        (
            "Qwen3-Coder-30B-A3B-Instruct-Q6_K.gguf",
            "barriers=explicit",
        ),
        (
            "Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf",
            "barriers=explicit",
        ),
    ] {
        let path = workspace_model_path(model_file);
        if !path.exists() {
            continue;
        }

        let mapped = MappedModel::open(&path).unwrap();
        let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
        let weights = WeightStore::new(&mapped);
        let model =
            InferenceModel::with_backend(cfg, Box::new(MetalBackend::new().unwrap())).unwrap();
        let kv = model.create_model_kv_for_weights(&weights);
        let summary = model
            .decode_plan_summary_for_weights(
                &weights,
                &kv,
                crate::model::DecodeIntent::Throughput,
                true,
            )
            .unwrap();

        assert!(
            summary.contains(barrier_fragment),
            "{model_file} summary `{summary}` missing `{barrier_fragment}`",
        );
    }
}

#[test]
fn test_real_qwen3_coder_q5_profiled_single_decode_records_gpu_work() {
    let path = workspace_model_path("Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&mapped.header).unwrap();
    let prompt_token_ids = tokenizer.encode("Hello", true);
    let Some(&token_id) = prompt_token_ids.first() else {
        return;
    };

    let metal_model =
        InferenceModel::with_backend(cfg.clone(), Box::new(MetalBackend::new().unwrap())).unwrap();
    let mut metal_kv = metal_model.create_model_kv_for_weights(&weights);
    let mut logits = vec![0.0f32; cfg.vocab_size as usize];

    logits.fill(0.0);
    let mut ops = crate::metrics::OpBreakdown::new();
    metal_model
        .forward_single_profiled(token_id, 0, &mut metal_kv, &weights, &mut logits, &mut ops)
        .unwrap();

    assert!(
        logits.iter().all(|value| value.is_finite()),
        "Qwen3-Coder Q5_K_M single decode produced non-finite logits",
    );
    assert!(
        ops.gpu > std::time::Duration::ZERO,
        "Qwen3-Coder Q5_K_M single decode did not record any GPU work: {}",
        ops.summary(),
    );
    assert!(
        ops.gpu_encode_layer_ffn > std::time::Duration::ZERO,
        "Qwen3-Coder Q5_K_M single decode did not record any FFN encode work: {}",
        ops.summary(),
    );
}

#[test]
fn test_real_qwen3_coder_q5_layer0_qkv_decode_matvec_matches_cpu() {
    let path = workspace_model_path("Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&mapped.header).unwrap();
    let prompt_token_ids = tokenizer.encode("Hello", true);
    let Some(&token_id) = prompt_token_ids.first() else {
        return;
    };

    let metal = MetalBackend::new().unwrap();
    let cpu = crate::backend::cpu::CpuBackend;
    let prefix = "blk.0";
    let dim = cfg.embedding_dim as usize;
    let q_dim = cfg.n_heads as usize * cfg.head_dim as usize;
    let kv_dim = cfg.n_kv_heads as usize * cfg.head_dim as usize;

    let mut hidden = vec![0.0f32; dim];
    weights
        .dequantize_row("token_embd.weight", token_id as usize, &mut hidden)
        .unwrap();
    let attn_norm_w = weights
        .f32_slice(&format!("{prefix}.attn_norm.weight"))
        .unwrap();
    let mut norm_buf = vec![0.0f32; dim];
    crate::compute::rms_norm::rms_norm_out(&hidden, attn_norm_w, &mut norm_buf, cfg.rms_norm_eps);

    let (wq_raw, wq_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_q.weight"))
        .unwrap();
    let (wk_raw, wk_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_k.weight"))
        .unwrap();
    let (wv_raw, wv_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_v.weight"))
        .unwrap();
    assert_eq!(wq_dtype, GgmlType::Q5K);
    assert_eq!(wk_dtype, GgmlType::Q5K);
    assert_eq!(wv_dtype, GgmlType::Q6K);

    let mut expected_q = vec![0.0f32; q_dim];
    let mut expected_k = vec![0.0f32; kv_dim];
    let mut expected_v = vec![0.0f32; kv_dim];
    cpu.dequant_matmul(wq_raw, wq_dtype, &norm_buf, &mut expected_q, q_dim, 1, dim);
    cpu.dequant_matmul(wk_raw, wk_dtype, &norm_buf, &mut expected_k, kv_dim, 1, dim);
    cpu.dequant_matmul(wv_raw, wv_dtype, &norm_buf, &mut expected_v, kv_dim, 1, dim);

    let mut actual_q_single = vec![0.0f32; q_dim];
    let mut actual_k_single = vec![0.0f32; kv_dim];
    let mut actual_v_single = vec![0.0f32; kv_dim];
    metal.dequant_matmul(
        wq_raw,
        wq_dtype,
        &norm_buf,
        &mut actual_q_single,
        q_dim,
        1,
        dim,
    );
    metal.dequant_matmul(
        wk_raw,
        wk_dtype,
        &norm_buf,
        &mut actual_k_single,
        kv_dim,
        1,
        dim,
    );
    metal.dequant_matmul(
        wv_raw,
        wv_dtype,
        &norm_buf,
        &mut actual_v_single,
        kv_dim,
        1,
        dim,
    );

    let mut actual_q_batch = vec![0.0f32; q_dim];
    let mut actual_k_batch = vec![0.0f32; kv_dim];
    let mut actual_v_batch = vec![0.0f32; kv_dim];
    metal.batch_dequant_matvec(
        &[
            (wq_raw, wq_dtype, q_dim),
            (wk_raw, wk_dtype, kv_dim),
            (wv_raw, wv_dtype, kv_dim),
        ],
        &norm_buf,
        dim,
        &mut [
            &mut actual_q_batch[..],
            &mut actual_k_batch[..],
            &mut actual_v_batch[..],
        ],
    );

    for (label, actual_single, actual_batch, expected) in [
        ("q", &actual_q_single, &actual_q_batch, &expected_q),
        ("k", &actual_k_single, &actual_k_batch, &expected_k),
        ("v", &actual_v_single, &actual_v_batch, &expected_v),
    ] {
        assert!(
            actual_single.iter().all(|value| value.is_finite()),
            "Qwen3-Coder Q5_K_M layer0 {label} single matvec produced non-finite values",
        );
        assert!(
            actual_batch.iter().all(|value| value.is_finite()),
            "Qwen3-Coder Q5_K_M layer0 {label} batch matvec produced non-finite values",
        );

        let single_diff = max_abs_diff(actual_single, expected);
        let batch_diff = max_abs_diff(actual_batch, expected);
        let pair_diff = max_abs_diff(actual_batch, actual_single);
        let scale = expected
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max)
            .max(1.0);
        assert!(
            single_diff / scale < 5e-2,
            "Qwen3-Coder Q5_K_M layer0 {label} single matvec mismatch: rel_diff={} max_diff={single_diff}",
            single_diff / scale,
        );
        assert!(
            batch_diff / scale < 5e-2,
            "Qwen3-Coder Q5_K_M layer0 {label} batch matvec mismatch: rel_diff={} max_diff={batch_diff}",
            batch_diff / scale,
        );
        assert!(
            pair_diff / scale < 1e-4,
            "Qwen3-Coder Q5_K_M layer0 {label} batch/single matvec disagree: rel_diff={} max_diff={pair_diff}",
            pair_diff / scale,
        );
    }
}

#[test]
fn test_real_qwen3_coder_q5_layer0_single_token_moe_tail_matches_cpu_with_generic_path() {
    let _env_lock = crate::test_env_lock();
    let _smart_off = EnvVarGuard::set("AX_METAL_SMART_BARRIERS", "0");
    let _barriers_on = EnvVarGuard::set("AX_METAL_BARRIERS", "1");
    let _selected_off = EnvVarGuard::set("AX_QWEN35_SELECTED_EXPERT_SINGLE_TOKEN", "0");

    let path = workspace_model_path("Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();

    let n_tokens = 1usize;
    let dim = cfg.embedding_dim as usize;
    let n_expert = cfg.n_expert.unwrap_or(0) as usize;
    let n_expert_used = cfg.n_expert_used.unwrap_or(0) as usize;
    let expert_inter_dim = cfg.expert_intermediate_dim.unwrap_or(0) as usize;
    let prefix = "blk.0";

    let (gate_dtype, up_dtype, down_dtype) =
        crate::model::shared::routed_moe_expert_dtypes(&weights, prefix).unwrap();
    let gate_stride =
        crate::model::moe_utils::expert_byte_stride(gate_dtype, expert_inter_dim * dim);
    let up_stride = crate::model::moe_utils::expert_byte_stride(up_dtype, expert_inter_dim * dim);
    let down_stride =
        crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);

    let input_hidden = staged_hidden(n_tokens * dim);
    let mut expected_hidden = input_hidden.clone();
    let mut moe_norm_buf = vec![0.0f32; n_tokens * dim];
    let mut moe_accum_buf = vec![0.0f32; n_tokens * dim];
    let mut moe_scratch = MoeSingleScratch {
        gate_buf: vec![0.0f32; expert_inter_dim],
        up_buf: vec![0.0f32; expert_inter_dim],
        down_buf: vec![0.0f32; dim],
        accum_buf: vec![0.0f32; dim],
        router_logits: vec![0.0f32; n_expert],
    };
    Qwen3MoeForward::apply_moe_ffn_batch(
        &crate::backend::cpu::CpuBackend,
        &weights,
        prefix,
        &mut expected_hidden,
        &mut moe_norm_buf,
        &mut moe_accum_buf,
        &mut moe_scratch,
        n_tokens,
        dim,
        n_expert,
        n_expert_used,
        expert_inter_dim,
        cfg.rms_norm_eps,
        None,
    )
    .unwrap();

    metal_ops.init_batch_scratches(&cfg, n_tokens);
    if !metal_ops.has_cached_model_keys() {
        Qwen3MoeForward::build_cached_model_keys(metal_ops, &weights, &cfg).unwrap();
    }

    let cached_guard = metal_ops.cached_model_keys();
    let cached = cached_guard.as_ref().unwrap();
    let layer = &cached.layers[0];

    let ffn_norm = {
        let weight_cache = metal_ops.lock_weight_cache();
        weight_cache.get(&layer.ffn_norm).unwrap().clone()
    };
    let router = {
        let moe_weight_cache = metal_ops.lock_moe_weight_cache();
        moe_weight_cache
            .get(&layer.moe_router.unwrap())
            .unwrap()
            .clone()
    };
    let gate = {
        let moe_weight_cache = metal_ops.lock_moe_weight_cache();
        moe_weight_cache
            .get(&layer.moe_expert_gate.as_ref().unwrap()[0])
            .unwrap()
            .clone()
    };
    let up = {
        let moe_weight_cache = metal_ops.lock_moe_weight_cache();
        moe_weight_cache
            .get(&layer.moe_expert_up.as_ref().unwrap()[0])
            .unwrap()
            .clone()
    };
    let down = {
        let moe_weight_cache = metal_ops.lock_moe_weight_cache();
        moe_weight_cache
            .get(&layer.moe_expert_down.as_ref().unwrap()[0])
            .unwrap()
            .clone()
    };

    {
        let mut bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_mut().unwrap();
        unsafe {
            bs.hidden.as_mut_slice::<f32>()[..n_tokens * dim].copy_from_slice(&input_hidden);
        }
        let moe_scratch =
            crate::backend::metal::MoeBatchScratchView::from_batch_scratches(bs).unwrap();

        metal_ops
            .device
            .execute_sync(|encoder| {
                metal_ops.encode_moe_ffn_gpu_resident_cached_with_scratch_with_policy(
                    encoder,
                    moe_scratch,
                    &bs.hidden,
                    &ffn_norm,
                    &router,
                    layer.moe_router_dtype.unwrap(),
                    &gate,
                    gate_dtype,
                    &up,
                    up_dtype,
                    &down,
                    down_dtype,
                    n_tokens,
                    n_expert,
                    n_expert_used,
                    dim,
                    expert_inter_dim,
                    gate_stride,
                    up_stride,
                    down_stride,
                    cfg.rms_norm_eps,
                    None,
                    true,
                    Qwen3MoeForward::qwen3moe_blocked_q6q8_down_enabled(),
                )
            })
            .unwrap();
    }

    let actual_hidden = {
        let bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_ref().unwrap();
        unsafe { bs.hidden.as_slice::<f32>()[..n_tokens * dim].to_vec() }
    };

    assert!(
        actual_hidden.iter().all(|value| value.is_finite()),
        "Qwen3-Coder Q5_K_M layer0 single-token MoE tail produced non-finite values with generic path",
    );
    let diff = max_abs_diff(&actual_hidden, &expected_hidden);
    let scale = expected_hidden
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 1e-2,
        "Qwen3-Coder Q5_K_M layer0 single-token MoE tail mismatch with generic path: rel_diff={} max_diff={} actual[0..8]={:?} expected[0..8]={:?}",
        diff / scale,
        diff,
        &actual_hidden[..8],
        &expected_hidden[..8],
    );
}

#[test]
fn test_real_qwen3_coder_q5_attention_only_decode_matches_cpu_per_layer() {
    let _env_lock = crate::test_env_lock();
    let _skip_routed = EnvVarGuard::set("AX_QWEN35_PROFILE_SKIP_ROUTED_EXPERT", "1");
    let _skip_shared = EnvVarGuard::set("AX_QWEN35_PROFILE_SKIP_SHARED_EXPERT", "1");
    let _smart_off = EnvVarGuard::set("AX_METAL_SMART_BARRIERS", "0");
    let _barriers_on = EnvVarGuard::set("AX_METAL_BARRIERS", "1");

    let path = workspace_model_path("Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&mapped.header).unwrap();
    let prompt_token_ids = tokenizer.encode("Hello", true);
    let Some(&token_id) = prompt_token_ids.first() else {
        return;
    };

    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();
    let mut metal_kv =
        InferenceModel::with_backend(cfg.clone(), Box::new(MetalBackend::new().unwrap()))
            .unwrap()
            .create_model_kv_for_weights(&weights);
    let gpu_kv = metal_kv.as_gpu_mut().unwrap();
    gpu_kv.ensure_capacity(&metal_ops.device, 1).unwrap();

    metal_ops.init_scratches(&cfg);
    metal_ops.init_batch_scratches(&cfg, 1);
    if !metal_ops.has_cached_model_keys() {
        Qwen3MoeForward::build_cached_model_keys(metal_ops, &weights, &cfg).unwrap();
    }

    let cached_guard = metal_ops.cached_model_keys();
    let cached = cached_guard.as_ref().unwrap();
    let weight_cache = metal_ops.lock_weight_cache();
    let moe_weight_cache = metal_ops.lock_moe_weight_cache();
    let mut scratch_guard = metal_ops.scratches();
    let scratch = scratch_guard.as_mut().unwrap();

    let dim = cfg.embedding_dim as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let (gate_dtype, up_dtype, down_dtype) =
        crate::model::shared::routed_moe_expert_dtypes(&weights, "blk.0").unwrap();
    let gate_stride = crate::model::moe_utils::expert_byte_stride(
        gate_dtype,
        cfg.expert_intermediate_dim.unwrap_or(0) as usize * dim,
    );
    let up_stride = crate::model::moe_utils::expert_byte_stride(
        up_dtype,
        cfg.expert_intermediate_dim.unwrap_or(0) as usize * dim,
    );
    let down_stride = crate::model::moe_utils::expert_byte_stride(
        down_dtype,
        dim * cfg.expert_intermediate_dim.unwrap_or(0) as usize,
    );
    let exec_plan = Qwen3MoeForward::qwen3moe_decode_plan(
        metal_ops,
        gpu_kv,
        cfg.embedding_dim,
        cfg.head_dim,
        1,
        false,
        gate_dtype,
        up_dtype,
        down_dtype,
    );
    let mut hidden_cpu = vec![0.0f32; dim];
    weights
        .dequantize_row("token_embd.weight", token_id as usize, &mut hidden_cpu)
        .unwrap();
    unsafe {
        scratch.hidden.as_mut_slice::<f32>()[..dim].copy_from_slice(&hidden_cpu);
    }

    let mut norm_buf = vec![0.0f32; dim];
    let mut q_buf = vec![0.0f32; q_dim];
    let mut k_buf = vec![0.0f32; kv_dim];
    let mut v_buf = vec![0.0f32; kv_dim];
    let mut attn_out = vec![0.0f32; q_dim];
    let mut proj_buf = vec![0.0f32; dim];
    let attn_params =
        crate::compute::attention::AttentionParams::new(n_heads, n_kv_heads, head_dim);
    let cpu = crate::backend::cpu::CpuBackend;

    for layer in 0..cfg.n_layers as usize {
        let prefix = format!("blk.{layer}");
        let attn_norm_w = weights
            .f32_slice(&format!("{prefix}.attn_norm.weight"))
            .unwrap();
        crate::compute::rms_norm::rms_norm_out(
            &hidden_cpu,
            attn_norm_w,
            &mut norm_buf,
            cfg.rms_norm_eps,
        );

        let (wq_raw, wq_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_q.weight"))
            .unwrap();
        let (wk_raw, wk_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_k.weight"))
            .unwrap();
        let (wv_raw, wv_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_v.weight"))
            .unwrap();
        let (wo_raw, wo_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_output.weight"))
            .unwrap();
        cpu.dequant_matmul(wq_raw, wq_dtype, &norm_buf, &mut q_buf, q_dim, 1, dim);
        cpu.dequant_matmul(wk_raw, wk_dtype, &norm_buf, &mut k_buf, kv_dim, 1, dim);
        cpu.dequant_matmul(wv_raw, wv_dtype, &norm_buf, &mut v_buf, kv_dim, 1, dim);

        if let Some(norm_weights) =
            crate::model::shared::maybe_attention_qk_norm_weights(&weights, &prefix).unwrap()
        {
            crate::model::shared::apply_attention_qk_norm(
                &mut q_buf,
                &mut k_buf,
                n_heads,
                n_kv_heads,
                head_dim,
                norm_weights,
                cfg.rms_norm_eps,
            );
        }

        crate::compute::rope::apply_rope_multi_head_neox_partial_scaled(
            &mut q_buf,
            &mut k_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            head_dim,
            0.0,
            cfg.rope_freq_base,
        );
        crate::compute::attention::multi_head_attention(
            &q_buf,
            &k_buf,
            &v_buf,
            &mut attn_out,
            &attn_params,
            1,
        );
        cpu.dequant_matmul(wo_raw, wo_dtype, &attn_out, &mut proj_buf, dim, 1, q_dim);
        Qwen3MoeForward::parallel_elementwise_add(&mut hidden_cpu, &proj_buf);

        metal_ops
            .device
            .execute_sync(|encoder| {
                let barrier =
                    crate::model::shared::DecodeBarrierCtx::new(encoder, exec_plan.barriers);
                let mut ops = None;
                Qwen3MoeForward::encode_qwen3moe_gpu_layer_range(
                    encoder,
                    &barrier,
                    metal_ops,
                    &cfg,
                    &scratch.hidden,
                    scratch,
                    gpu_kv,
                    cached,
                    &weight_cache,
                    &moe_weight_cache,
                    &exec_plan,
                    0,
                    layer,
                    layer + 1,
                    gate_dtype,
                    up_dtype,
                    down_dtype,
                    gate_stride,
                    up_stride,
                    down_stride,
                    &mut ops,
                )?;
                barrier.flush();
                Ok(())
            })
            .unwrap();

        let hidden_gpu = unsafe { scratch.hidden.as_slice::<f32>()[..dim].to_vec() };
        let diff = max_abs_diff(&hidden_gpu, &hidden_cpu);
        let scale = hidden_cpu
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max)
            .max(1.0);
        assert!(
            diff / scale < 5e-2,
            "Qwen3-Coder Q5_K_M attention-only decode mismatch after layer {layer}: rel_diff={} max_diff={} gpu[0..8]={:?} cpu[0..8]={:?}",
            diff / scale,
            diff,
            &hidden_gpu[..8],
            &hidden_cpu[..8],
        );
    }

    let mut final_hidden = hidden_cpu.clone();
    apply_output_norm_single(&weights, &mut final_hidden, cfg.rms_norm_eps, None).unwrap();
    let mut expected_logits = vec![0.0f32; cfg.vocab_size as usize];
    crate::model::shared::write_normalized_single_logits_with_breakdown(
        &cpu,
        &final_hidden,
        dim,
        cfg.vocab_size as usize,
        &weights,
        &mut expected_logits,
        None,
    )
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
    assert!(
        actual_logits.iter().all(|value| value.is_finite()),
        "Qwen3-Coder Q5_K_M output head produced non-finite logits on attention-only decode",
    );
    let logits_diff = max_abs_diff(&actual_logits, &expected_logits);
    let logits_scale = expected_logits
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        logits_diff / logits_scale < 5e-2,
        "Qwen3-Coder Q5_K_M output head mismatch on attention-only decode: rel_diff={} max_diff={} gpu_top={:?} cpu_top={:?}",
        logits_diff / logits_scale,
        logits_diff,
        actual_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)),
        expected_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)),
    );
}

#[test]
fn test_real_qwen3_coder_q5_layer6_single_token_moe_tail_matches_cpu_with_generic_path() {
    let _env_lock = crate::test_env_lock();
    let _smart_off = EnvVarGuard::set("AX_METAL_SMART_BARRIERS", "0");
    let _barriers_on = EnvVarGuard::set("AX_METAL_BARRIERS", "1");
    let _selected_off = EnvVarGuard::set("AX_QWEN35_SELECTED_EXPERT_SINGLE_TOKEN", "0");

    let path = workspace_model_path("Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&mapped.header).unwrap();
    let prompt_token_ids = tokenizer.encode("Hello", true);
    let Some(&token_id) = prompt_token_ids.first() else {
        return;
    };

    let layer = 6usize;
    let prefix = format!("blk.{layer}");
    let dim = cfg.embedding_dim as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let n_expert = cfg.n_expert.unwrap_or(0) as usize;
    let n_expert_used = cfg.n_expert_used.unwrap_or(0) as usize;
    let expert_inter_dim = cfg.expert_intermediate_dim.unwrap_or(0) as usize;

    let cpu = crate::backend::cpu::CpuBackend;
    let mut hidden = vec![0.0f32; dim];
    weights
        .dequantize_row("token_embd.weight", token_id as usize, &mut hidden)
        .unwrap();

    let mut norm_buf = vec![0.0f32; dim];
    let mut q_buf = vec![0.0f32; q_dim];
    let mut k_buf = vec![0.0f32; kv_dim];
    let mut v_buf = vec![0.0f32; kv_dim];
    let mut attn_out = vec![0.0f32; q_dim];
    let mut proj_buf = vec![0.0f32; dim];
    let mut moe_scratch = MoeSingleScratch {
        gate_buf: vec![0.0f32; expert_inter_dim],
        up_buf: vec![0.0f32; expert_inter_dim],
        down_buf: vec![0.0f32; dim],
        accum_buf: vec![0.0f32; dim],
        router_logits: vec![0.0f32; n_expert],
    };

    for current_layer in 0..=layer {
        let current_prefix = format!("blk.{current_layer}");
        let attn_norm_w = weights
            .f32_slice(&format!("{current_prefix}.attn_norm.weight"))
            .unwrap();
        crate::compute::rms_norm::rms_norm_out(
            &hidden,
            attn_norm_w,
            &mut norm_buf,
            cfg.rms_norm_eps,
        );

        let (wq_raw, wq_dtype) = weights
            .raw_with_dtype(&format!("{current_prefix}.attn_q.weight"))
            .unwrap();
        let (wk_raw, wk_dtype) = weights
            .raw_with_dtype(&format!("{current_prefix}.attn_k.weight"))
            .unwrap();
        let (wv_raw, wv_dtype) = weights
            .raw_with_dtype(&format!("{current_prefix}.attn_v.weight"))
            .unwrap();
        let (wo_raw, wo_dtype) = weights
            .raw_with_dtype(&format!("{current_prefix}.attn_output.weight"))
            .unwrap();

        cpu.dequant_matmul(wq_raw, wq_dtype, &norm_buf, &mut q_buf, q_dim, 1, dim);
        cpu.dequant_matmul(wk_raw, wk_dtype, &norm_buf, &mut k_buf, kv_dim, 1, dim);
        cpu.dequant_matmul(wv_raw, wv_dtype, &norm_buf, &mut v_buf, kv_dim, 1, dim);

        if let Some(norm_weights) =
            crate::model::shared::maybe_attention_qk_norm_weights(&weights, &current_prefix)
                .unwrap()
        {
            crate::model::shared::apply_attention_qk_norm(
                &mut q_buf,
                &mut k_buf,
                n_heads,
                n_kv_heads,
                head_dim,
                norm_weights,
                cfg.rms_norm_eps,
            );
        }

        crate::compute::rope::apply_rope_multi_head_neox_partial_scaled(
            &mut q_buf,
            &mut k_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            head_dim,
            0.0,
            cfg.rope_freq_base,
        );
        crate::compute::attention::multi_head_attention(
            &q_buf,
            &k_buf,
            &v_buf,
            &mut attn_out,
            &crate::compute::attention::AttentionParams::new(n_heads, n_kv_heads, head_dim),
            1,
        );
        cpu.dequant_matmul(wo_raw, wo_dtype, &attn_out, &mut proj_buf, dim, 1, q_dim);
        Qwen3MoeForward::parallel_elementwise_add(&mut hidden, &proj_buf);

        if current_layer < layer {
            Qwen3MoeForward::apply_moe_ffn_single(
                &cpu,
                &weights,
                &current_prefix,
                &mut hidden,
                &mut norm_buf,
                &mut moe_scratch,
                dim,
                n_expert,
                n_expert_used,
                expert_inter_dim,
                cfg.rms_norm_eps,
            )
            .unwrap();
        }
    }

    let input_hidden = hidden.clone();
    let mut expected_hidden = input_hidden.clone();
    Qwen3MoeForward::apply_moe_ffn_single(
        &cpu,
        &weights,
        &prefix,
        &mut expected_hidden,
        &mut norm_buf,
        &mut moe_scratch,
        dim,
        n_expert,
        n_expert_used,
        expert_inter_dim,
        cfg.rms_norm_eps,
    )
    .unwrap();

    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();
    metal_ops.init_batch_scratches(&cfg, 1);
    if !metal_ops.has_cached_model_keys() {
        Qwen3MoeForward::build_cached_model_keys(metal_ops, &weights, &cfg).unwrap();
    }

    let (gate_dtype, up_dtype, down_dtype) =
        crate::model::shared::routed_moe_expert_dtypes(&weights, &prefix).unwrap();
    let gate_stride =
        crate::model::moe_utils::expert_byte_stride(gate_dtype, expert_inter_dim * dim);
    let up_stride = crate::model::moe_utils::expert_byte_stride(up_dtype, expert_inter_dim * dim);
    let down_stride =
        crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);

    let cached_guard = metal_ops.cached_model_keys();
    let cached = cached_guard.as_ref().unwrap();
    let cached_layer = &cached.layers[layer];

    let ffn_norm = {
        let weight_cache = metal_ops.lock_weight_cache();
        weight_cache.get(&cached_layer.ffn_norm).unwrap().clone()
    };
    let router = {
        let moe_weight_cache = metal_ops.lock_moe_weight_cache();
        moe_weight_cache
            .get(&cached_layer.moe_router.unwrap())
            .unwrap()
            .clone()
    };
    let gate = {
        let moe_weight_cache = metal_ops.lock_moe_weight_cache();
        moe_weight_cache
            .get(&cached_layer.moe_expert_gate.as_ref().unwrap()[0])
            .unwrap()
            .clone()
    };
    let up = {
        let moe_weight_cache = metal_ops.lock_moe_weight_cache();
        moe_weight_cache
            .get(&cached_layer.moe_expert_up.as_ref().unwrap()[0])
            .unwrap()
            .clone()
    };
    let down = {
        let moe_weight_cache = metal_ops.lock_moe_weight_cache();
        moe_weight_cache
            .get(&cached_layer.moe_expert_down.as_ref().unwrap()[0])
            .unwrap()
            .clone()
    };

    {
        let mut bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_mut().unwrap();
        unsafe {
            bs.hidden.as_mut_slice::<f32>()[..dim].copy_from_slice(&input_hidden);
        }
        let moe_scratch =
            crate::backend::metal::MoeBatchScratchView::from_batch_scratches(bs).unwrap();

        metal_ops
            .device
            .execute_sync(|encoder| {
                metal_ops.encode_moe_ffn_gpu_resident_cached_with_scratch_with_policy(
                    encoder,
                    moe_scratch,
                    &bs.hidden,
                    &ffn_norm,
                    &router,
                    cached_layer.moe_router_dtype.unwrap(),
                    &gate,
                    gate_dtype,
                    &up,
                    up_dtype,
                    &down,
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
                    None,
                    true,
                    Qwen3MoeForward::qwen3moe_blocked_q6q8_down_enabled(),
                )
            })
            .unwrap();
    }

    let actual_hidden = {
        let bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_ref().unwrap();
        unsafe { bs.hidden.as_slice::<f32>()[..dim].to_vec() }
    };
    assert!(
        actual_hidden.iter().all(|value| value.is_finite()),
        "Qwen3-Coder Q5_K_M layer6 single-token MoE tail produced non-finite values with generic path",
    );
    let diff = max_abs_diff(&actual_hidden, &expected_hidden);
    let scale = expected_hidden
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 1e-2,
        "Qwen3-Coder Q5_K_M layer6 single-token MoE tail mismatch with generic path: rel_diff={} max_diff={} actual[0..8]={:?} expected[0..8]={:?}",
        diff / scale,
        diff,
        &actual_hidden[..8],
        &expected_hidden[..8],
    );

    {
        let mut bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_mut().unwrap();
        unsafe {
            bs.hidden.as_mut_slice::<f32>()[..dim].copy_from_slice(&input_hidden);
        }
        let moe_scratch =
            crate::backend::metal::MoeBatchScratchView::from_batch_scratches(bs).unwrap();

        metal_ops
            .device
            .execute_sync(|encoder| {
                metal_ops.encode_moe_ffn_gpu_resident_cached_with_scratch_with_policy(
                    encoder,
                    moe_scratch,
                    &bs.hidden,
                    &ffn_norm,
                    &router,
                    cached_layer.moe_router_dtype.unwrap(),
                    &gate,
                    gate_dtype,
                    &up,
                    up_dtype,
                    &down,
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
                    None,
                    true,
                    Qwen3MoeForward::qwen3moe_blocked_q6q8_down_enabled(),
                )
            })
            .unwrap();
    }

    let repeated_hidden = {
        let bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_ref().unwrap();
        unsafe { bs.hidden.as_slice::<f32>()[..dim].to_vec() }
    };
    assert!(
        repeated_hidden.iter().all(|value| value.is_finite()),
        "Qwen3-Coder Q5_K_M layer6 single-token MoE tail produced non-finite values on repeated scratch reuse",
    );
    let repeat_diff = max_abs_diff(&repeated_hidden, &expected_hidden);
    assert!(
        repeat_diff / scale < 1e-2,
        "Qwen3-Coder Q5_K_M layer6 repeated single-token MoE tail mismatch: rel_diff={} max_diff={} actual[0..8]={:?} expected[0..8]={:?}",
        repeat_diff / scale,
        repeat_diff,
        &repeated_hidden[..8],
        &expected_hidden[..8],
    );
}

#[test]
fn test_real_qwen3_coder_q5_generic_decode_matches_cpu_per_layer() {
    let _env_lock = crate::test_env_lock();
    let _selected_off = EnvVarGuard::set("AX_QWEN35_SELECTED_EXPERT_SINGLE_TOKEN", "0");
    let _smart_off = EnvVarGuard::set("AX_METAL_SMART_BARRIERS", "0");
    let _barriers_on = EnvVarGuard::set("AX_METAL_BARRIERS", "1");

    let path = workspace_model_path("Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&mapped.header).unwrap();
    let prompt_token_ids = tokenizer.encode("Hello", true);
    let Some(&token_id) = prompt_token_ids.first() else {
        return;
    };

    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();
    let mut metal_kv =
        InferenceModel::with_backend(cfg.clone(), Box::new(MetalBackend::new().unwrap()))
            .unwrap()
            .create_model_kv_for_weights(&weights);
    let gpu_kv = metal_kv.as_gpu_mut().unwrap();
    gpu_kv.ensure_capacity(&metal_ops.device, 1).unwrap();

    metal_ops.init_scratches(&cfg);
    metal_ops.init_batch_scratches(&cfg, 1);
    if !metal_ops.has_cached_model_keys() {
        Qwen3MoeForward::build_cached_model_keys(metal_ops, &weights, &cfg).unwrap();
    }

    let cached_guard = metal_ops.cached_model_keys();
    let cached = cached_guard.as_ref().unwrap();
    let weight_cache = metal_ops.lock_weight_cache();
    let moe_weight_cache = metal_ops.lock_moe_weight_cache();
    let mut scratch_guard = metal_ops.scratches();
    let scratch = scratch_guard.as_mut().unwrap();

    let dim = cfg.embedding_dim as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let n_expert = cfg.n_expert.unwrap_or(0) as usize;
    let n_expert_used = cfg.n_expert_used.unwrap_or(0) as usize;
    let expert_inter_dim = cfg.expert_intermediate_dim.unwrap_or(0) as usize;

    let (gate_dtype, up_dtype, down_dtype) =
        crate::model::shared::routed_moe_expert_dtypes(&weights, "blk.0").unwrap();
    let gate_stride =
        crate::model::moe_utils::expert_byte_stride(gate_dtype, expert_inter_dim * dim);
    let up_stride = crate::model::moe_utils::expert_byte_stride(up_dtype, expert_inter_dim * dim);
    let down_stride =
        crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);
    let exec_plan = Qwen3MoeForward::qwen3moe_decode_plan(
        metal_ops,
        gpu_kv,
        cfg.embedding_dim,
        cfg.head_dim,
        1,
        false,
        gate_dtype,
        up_dtype,
        down_dtype,
    );
    let split_moe_command_buffers = Qwen3MoeForward::qwen3moe_split_moe_decode_command_buffers(
        gate_dtype, up_dtype, down_dtype,
    );
    if std::env::var("AX_DEBUG_QWEN3_Q5_LAYER_DIFFS").is_ok() {
        eprintln!("[QWEN3-Q5-GENERIC] split_moe_command_buffers={split_moe_command_buffers}");
    }

    let cpu = crate::backend::cpu::CpuBackend;
    let mut hidden_cpu = vec![0.0f32; dim];
    weights
        .dequantize_row("token_embd.weight", token_id as usize, &mut hidden_cpu)
        .unwrap();
    unsafe {
        scratch.hidden.as_mut_slice::<f32>()[..dim].copy_from_slice(&hidden_cpu);
    }

    let mut norm_buf = vec![0.0f32; dim];
    let mut q_buf = vec![0.0f32; q_dim];
    let mut k_buf = vec![0.0f32; kv_dim];
    let mut v_buf = vec![0.0f32; kv_dim];
    let mut attn_out = vec![0.0f32; q_dim];
    let mut proj_buf = vec![0.0f32; dim];
    let mut moe_scratch = MoeSingleScratch {
        gate_buf: vec![0.0f32; expert_inter_dim],
        up_buf: vec![0.0f32; expert_inter_dim],
        down_buf: vec![0.0f32; dim],
        accum_buf: vec![0.0f32; dim],
        router_logits: vec![0.0f32; n_expert],
    };

    for layer in 0..cfg.n_layers as usize {
        let prefix = format!("blk.{layer}");
        let (layer_gate_dtype, layer_up_dtype, layer_down_dtype) =
            crate::model::shared::routed_moe_expert_dtypes(&weights, &prefix).unwrap();
        let layer_gate_stride =
            crate::model::moe_utils::expert_byte_stride(layer_gate_dtype, expert_inter_dim * dim);
        let layer_up_stride =
            crate::model::moe_utils::expert_byte_stride(layer_up_dtype, expert_inter_dim * dim);
        let layer_down_stride =
            crate::model::moe_utils::expert_byte_stride(layer_down_dtype, dim * expert_inter_dim);
        let attn_norm_w = weights
            .f32_slice(&format!("{prefix}.attn_norm.weight"))
            .unwrap();
        crate::compute::rms_norm::rms_norm_out(
            &hidden_cpu,
            attn_norm_w,
            &mut norm_buf,
            cfg.rms_norm_eps,
        );

        let (wq_raw, wq_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_q.weight"))
            .unwrap();
        let (wk_raw, wk_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_k.weight"))
            .unwrap();
        let (wv_raw, wv_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_v.weight"))
            .unwrap();
        let (wo_raw, wo_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_output.weight"))
            .unwrap();
        cpu.dequant_matmul(wq_raw, wq_dtype, &norm_buf, &mut q_buf, q_dim, 1, dim);
        cpu.dequant_matmul(wk_raw, wk_dtype, &norm_buf, &mut k_buf, kv_dim, 1, dim);
        cpu.dequant_matmul(wv_raw, wv_dtype, &norm_buf, &mut v_buf, kv_dim, 1, dim);

        if let Some(norm_weights) =
            crate::model::shared::maybe_attention_qk_norm_weights(&weights, &prefix).unwrap()
        {
            crate::model::shared::apply_attention_qk_norm(
                &mut q_buf,
                &mut k_buf,
                n_heads,
                n_kv_heads,
                head_dim,
                norm_weights,
                cfg.rms_norm_eps,
            );
        }

        crate::compute::rope::apply_rope_multi_head_neox_partial_scaled(
            &mut q_buf,
            &mut k_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            head_dim,
            0.0,
            cfg.rope_freq_base,
        );
        crate::compute::attention::multi_head_attention(
            &q_buf,
            &k_buf,
            &v_buf,
            &mut attn_out,
            &crate::compute::attention::AttentionParams::new(n_heads, n_kv_heads, head_dim),
            1,
        );
        cpu.dequant_matmul(wo_raw, wo_dtype, &attn_out, &mut proj_buf, dim, 1, q_dim);
        Qwen3MoeForward::parallel_elementwise_add(&mut hidden_cpu, &proj_buf);
        let pre_moe_hidden_cpu = hidden_cpu.clone();
        Qwen3MoeForward::apply_moe_ffn_single(
            &cpu,
            &weights,
            &prefix,
            &mut hidden_cpu,
            &mut norm_buf,
            &mut moe_scratch,
            dim,
            n_expert,
            n_expert_used,
            expert_inter_dim,
            cfg.rms_norm_eps,
        )
        .unwrap();

        if split_moe_command_buffers {
            metal_ops
                .device
                .execute_sync(|encoder| {
                    let barrier =
                        crate::model::shared::DecodeBarrierCtx::new(encoder, exec_plan.barriers);
                    let mut ops = None;
                    Qwen3MoeForward::encode_qwen3moe_gpu_layer_range_internal(
                        encoder,
                        &barrier,
                        metal_ops,
                        &cfg,
                        &scratch.hidden,
                        scratch,
                        gpu_kv,
                        cached,
                        &weight_cache,
                        &moe_weight_cache,
                        &exec_plan,
                        0,
                        layer,
                        layer + 1,
                        gate_dtype,
                        up_dtype,
                        down_dtype,
                        gate_stride,
                        up_stride,
                        down_stride,
                        &mut ops,
                        false,
                    )?;
                    barrier.flush();
                    Ok(())
                })
                .unwrap();

            let pre_moe_hidden_gpu = unsafe { scratch.hidden.as_slice::<f32>()[..dim].to_vec() };
            assert!(
                pre_moe_hidden_gpu.iter().all(|value| value.is_finite()),
                "Qwen3-Coder Q5_K_M split decode produced non-finite pre-MoE hidden after layer {layer}",
            );
            let pre_moe_diff = max_abs_diff(&pre_moe_hidden_gpu, &pre_moe_hidden_cpu);
            let pre_moe_scale = pre_moe_hidden_cpu
                .iter()
                .copied()
                .map(f32::abs)
                .fold(0.0f32, f32::max)
                .max(1.0);
            assert!(
                pre_moe_diff / pre_moe_scale < 5e-2,
                "Qwen3-Coder Q5_K_M split decode pre-MoE mismatch after layer {layer}: rel_diff={} max_diff={} gpu[0..8]={:?} cpu[0..8]={:?}",
                pre_moe_diff / pre_moe_scale,
                pre_moe_diff,
                &pre_moe_hidden_gpu[..8],
                &pre_moe_hidden_cpu[..8],
            );
            let gpu_input_route = cpu_moe_route_for_hidden(
                &weights,
                &prefix,
                &pre_moe_hidden_gpu,
                dim,
                n_expert,
                n_expert_used,
                cfg.rms_norm_eps,
            );
            let mut expected_hidden_from_gpu_input = pre_moe_hidden_gpu.clone();
            Qwen3MoeForward::apply_moe_ffn_single(
                &cpu,
                &weights,
                &prefix,
                &mut expected_hidden_from_gpu_input,
                &mut norm_buf,
                &mut moe_scratch,
                dim,
                n_expert,
                n_expert_used,
                expert_inter_dim,
                cfg.rms_norm_eps,
            )
            .unwrap();

            let cached_layer = &cached.layers[layer];
            let ffn_norm = weight_cache.get(&cached_layer.ffn_norm).unwrap();
            let router = moe_weight_cache
                .get(&cached_layer.moe_router.unwrap())
                .unwrap();
            let gate = moe_weight_cache
                .get(&cached_layer.moe_expert_gate.as_ref().unwrap()[0])
                .unwrap();
            let up = moe_weight_cache
                .get(&cached_layer.moe_expert_up.as_ref().unwrap()[0])
                .unwrap();
            let down = moe_weight_cache
                .get(&cached_layer.moe_expert_down.as_ref().unwrap()[0])
                .unwrap();
            let (bs_hidden, moe_scratch) = {
                let mut bs_guard = metal_ops.batch_scratches();
                let bs = bs_guard.as_mut().unwrap();
                unsafe {
                    bs.hidden.as_mut_slice::<f32>()[..dim]
                        .copy_from_slice(scratch.hidden.as_slice::<f32>()[..dim].as_ref());
                }
                (
                    bs.hidden.clone(),
                    crate::backend::metal::MoeBatchScratchView::from_batch_scratches(bs).unwrap(),
                )
            };
            metal_ops
                .device
                .execute_sync(|encoder| {
                    metal_ops.encode_moe_ffn_gpu_resident_cached_with_scratch_with_policy(
                        encoder,
                        moe_scratch,
                        &bs_hidden,
                        ffn_norm,
                        router,
                        cached_layer.moe_router_dtype.unwrap(),
                        gate,
                        layer_gate_dtype,
                        up,
                        layer_up_dtype,
                        down,
                        layer_down_dtype,
                        1,
                        n_expert,
                        n_expert_used,
                        dim,
                        expert_inter_dim,
                        layer_gate_stride,
                        layer_up_stride,
                        layer_down_stride,
                        cfg.rms_norm_eps,
                        None,
                        true,
                        Qwen3MoeForward::qwen3moe_blocked_q6q8_down_enabled(),
                    )
                })
                .unwrap();
            let (
                batch_hidden_after_moe,
                gpu_router_logits,
                gpu_ids,
                gpu_weights,
                gate_max_abs,
                up_max_abs,
                gate_nonfinite,
                up_nonfinite,
                down_nonfinite,
                accum_nonfinite,
            ) = {
                let bs_guard = metal_ops.batch_scratches();
                let bs = bs_guard.as_ref().unwrap();
                let hidden = unsafe { bs.hidden.as_slice::<f32>()[..dim].to_vec() };
                let router_logits = unsafe {
                    bs.moe_router_out.as_ref().unwrap().as_slice::<f32>()[..n_expert].to_vec()
                };
                let ids = unsafe {
                    bs.moe_expert_ids.as_ref().unwrap().as_slice::<i32>()[..n_expert_used].to_vec()
                };
                let weights = unsafe {
                    bs.moe_expert_weights.as_ref().unwrap().as_slice::<f32>()[..n_expert_used]
                        .to_vec()
                };
                let gate_nonfinite = unsafe {
                    let gate_out = &bs.moe_gate_out.as_ref().unwrap().as_slice::<f32>()
                        [..n_expert_used * expert_inter_dim];
                    first_nonfinite(gate_out)
                };
                let up_nonfinite = unsafe {
                    let up_out = &bs.moe_up_out.as_ref().unwrap().as_slice::<f32>()
                        [..n_expert_used * expert_inter_dim];
                    first_nonfinite(up_out)
                };
                let gate_max_abs = unsafe {
                    bs.moe_gate_out.as_ref().unwrap().as_slice::<f32>()
                        [..n_expert_used * expert_inter_dim]
                        .iter()
                        .copied()
                        .map(f32::abs)
                        .fold(0.0f32, f32::max)
                };
                let up_max_abs = unsafe {
                    bs.moe_up_out.as_ref().unwrap().as_slice::<f32>()
                        [..n_expert_used * expert_inter_dim]
                        .iter()
                        .copied()
                        .map(f32::abs)
                        .fold(0.0f32, f32::max)
                };
                let down_nonfinite = unsafe {
                    first_nonfinite(
                        &bs.moe_down_out.as_ref().unwrap().as_slice::<f32>()[..n_expert_used * dim],
                    )
                };
                let accum_nonfinite = unsafe {
                    first_nonfinite(&bs.moe_accum.as_ref().unwrap().as_slice::<f32>()[..dim])
                };
                (
                    hidden,
                    router_logits,
                    ids,
                    weights,
                    gate_max_abs,
                    up_max_abs,
                    gate_nonfinite,
                    up_nonfinite,
                    down_nonfinite,
                    accum_nonfinite,
                )
            };
            unsafe {
                scratch.hidden.as_mut_slice::<f32>()[..dim]
                    .copy_from_slice(&batch_hidden_after_moe);
            }
            if let Some((idx, value)) = first_nonfinite(&batch_hidden_after_moe) {
                let router_scale = gpu_input_route
                    .router_logits
                    .iter()
                    .copied()
                    .map(f32::abs)
                    .fold(0.0f32, f32::max)
                    .max(1.0);
                let router_rel_diff =
                    max_abs_diff(&gpu_router_logits, &gpu_input_route.router_logits) / router_scale;
                panic!(
                    "Qwen3-Coder Q5_K_M split MoE produced non-finite hidden after layer {layer} at index {idx} value={value}; cpu_route_ids={:?}; cpu_route_weights={:?}; gpu_ids={:?}; gpu_weights={:?}; router_rel_diff={router_rel_diff}; gate_max_abs={gate_max_abs}; up_max_abs={up_max_abs}; gate_nonfinite={gate_nonfinite:?}; up_nonfinite={up_nonfinite:?}; down_nonfinite={down_nonfinite:?}; accum_nonfinite={accum_nonfinite:?}",
                    gpu_input_route.top_ids, gpu_input_route.top_weights, gpu_ids, gpu_weights,
                );
            }
            let expected_gpu_ids: Vec<i32> = gpu_input_route
                .top_ids
                .iter()
                .map(|&expert| expert as i32)
                .collect();
            assert_eq!(
                gpu_ids, expected_gpu_ids,
                "Qwen3-Coder Q5_K_M split MoE route ids mismatch after layer {layer}: cpu_ids={:?} gpu_ids={:?}",
                gpu_input_route.top_ids, gpu_ids,
            );
            let weight_diff = max_abs_diff(&gpu_weights, &gpu_input_route.top_weights);
            assert!(
                weight_diff < 1e-4,
                "Qwen3-Coder Q5_K_M split MoE route weights mismatch after layer {layer}: max_diff={} cpu_weights={:?} gpu_weights={:?}",
                weight_diff,
                gpu_input_route.top_weights,
                gpu_weights,
            );
            let moe_diff = max_abs_diff(&batch_hidden_after_moe, &expected_hidden_from_gpu_input);
            let moe_scale = expected_hidden_from_gpu_input
                .iter()
                .copied()
                .map(f32::abs)
                .fold(0.0f32, f32::max)
                .max(1.0);
            assert!(
                moe_diff / moe_scale < 5e-2,
                "Qwen3-Coder Q5_K_M split MoE mismatch after layer {layer}: rel_diff={} max_diff={} actual[0..8]={:?} expected_from_gpu_input[0..8]={:?}",
                moe_diff / moe_scale,
                moe_diff,
                &batch_hidden_after_moe[..8],
                &expected_hidden_from_gpu_input[..8],
            );
        } else {
            metal_ops
                .device
                .execute_sync(|encoder| {
                    let barrier =
                        crate::model::shared::DecodeBarrierCtx::new(encoder, exec_plan.barriers);
                    let mut ops = None;
                    Qwen3MoeForward::encode_qwen3moe_gpu_layer_range(
                        encoder,
                        &barrier,
                        metal_ops,
                        &cfg,
                        &scratch.hidden,
                        scratch,
                        gpu_kv,
                        cached,
                        &weight_cache,
                        &moe_weight_cache,
                        &exec_plan,
                        0,
                        layer,
                        layer + 1,
                        gate_dtype,
                        up_dtype,
                        down_dtype,
                        gate_stride,
                        up_stride,
                        down_stride,
                        &mut ops,
                    )?;
                    barrier.flush();
                    Ok(())
                })
                .unwrap();
        }

        let hidden_gpu = unsafe { scratch.hidden.as_slice::<f32>()[..dim].to_vec() };
        assert!(
            hidden_gpu.iter().all(|value| value.is_finite()),
            "Qwen3-Coder Q5_K_M generic decode produced non-finite hidden after layer {layer}",
        );
        let diff = max_abs_diff(&hidden_gpu, &hidden_cpu);
        let scale = hidden_cpu
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max)
            .max(1.0);
        if std::env::var("AX_DEBUG_QWEN3_Q5_LAYER_DIFFS").is_ok() {
            eprintln!(
                "[QWEN3-Q5-GENERIC] layer={layer} rel_diff={} max_diff={} gpu0={:?} cpu0={:?}",
                diff / scale,
                diff,
                &hidden_gpu[..4],
                &hidden_cpu[..4],
            );
        }
        assert!(
            diff / scale < 5e-2,
            "Qwen3-Coder Q5_K_M generic decode mismatch after layer {layer}: rel_diff={} max_diff={} gpu[0..8]={:?} cpu[0..8]={:?}",
            diff / scale,
            diff,
            &hidden_gpu[..8],
            &hidden_cpu[..8],
        );
    }
}

#[test]
fn test_real_qwen3_coder_q5_moe_tail_sequence_matches_cpu_with_generic_path() {
    let _env_lock = crate::test_env_lock();
    let _selected_off = EnvVarGuard::set("AX_QWEN35_SELECTED_EXPERT_SINGLE_TOKEN", "0");
    let _smart_off = EnvVarGuard::set("AX_METAL_SMART_BARRIERS", "0");
    let _barriers_on = EnvVarGuard::set("AX_METAL_BARRIERS", "1");

    let path = workspace_model_path("Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&mapped.header).unwrap();
    let prompt_token_ids = tokenizer.encode("Hello", true);
    let Some(&token_id) = prompt_token_ids.first() else {
        return;
    };

    let dim = cfg.embedding_dim as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let n_expert = cfg.n_expert.unwrap_or(0) as usize;
    let n_expert_used = cfg.n_expert_used.unwrap_or(0) as usize;
    let expert_inter_dim = cfg.expert_intermediate_dim.unwrap_or(0) as usize;

    let cpu = crate::backend::cpu::CpuBackend;
    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();
    metal_ops.init_batch_scratches(&cfg, 1);
    if !metal_ops.has_cached_model_keys() {
        Qwen3MoeForward::build_cached_model_keys(metal_ops, &weights, &cfg).unwrap();
    }

    let cached_guard = metal_ops.cached_model_keys();
    let cached = cached_guard.as_ref().unwrap();

    let mut hidden = vec![0.0f32; dim];
    weights
        .dequantize_row("token_embd.weight", token_id as usize, &mut hidden)
        .unwrap();

    let mut norm_buf = vec![0.0f32; dim];
    let mut q_buf = vec![0.0f32; q_dim];
    let mut k_buf = vec![0.0f32; kv_dim];
    let mut v_buf = vec![0.0f32; kv_dim];
    let mut attn_out = vec![0.0f32; q_dim];
    let mut proj_buf = vec![0.0f32; dim];
    let mut moe_scratch = MoeSingleScratch {
        gate_buf: vec![0.0f32; expert_inter_dim],
        up_buf: vec![0.0f32; expert_inter_dim],
        down_buf: vec![0.0f32; dim],
        accum_buf: vec![0.0f32; dim],
        router_logits: vec![0.0f32; n_expert],
    };

    for layer in 0..=6usize {
        let prefix = format!("blk.{layer}");
        let attn_norm_w = weights
            .f32_slice(&format!("{prefix}.attn_norm.weight"))
            .unwrap();
        crate::compute::rms_norm::rms_norm_out(
            &hidden,
            attn_norm_w,
            &mut norm_buf,
            cfg.rms_norm_eps,
        );

        let (wq_raw, wq_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_q.weight"))
            .unwrap();
        let (wk_raw, wk_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_k.weight"))
            .unwrap();
        let (wv_raw, wv_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_v.weight"))
            .unwrap();
        let (wo_raw, wo_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_output.weight"))
            .unwrap();

        cpu.dequant_matmul(wq_raw, wq_dtype, &norm_buf, &mut q_buf, q_dim, 1, dim);
        cpu.dequant_matmul(wk_raw, wk_dtype, &norm_buf, &mut k_buf, kv_dim, 1, dim);
        cpu.dequant_matmul(wv_raw, wv_dtype, &norm_buf, &mut v_buf, kv_dim, 1, dim);

        if let Some(norm_weights) =
            crate::model::shared::maybe_attention_qk_norm_weights(&weights, &prefix).unwrap()
        {
            crate::model::shared::apply_attention_qk_norm(
                &mut q_buf,
                &mut k_buf,
                n_heads,
                n_kv_heads,
                head_dim,
                norm_weights,
                cfg.rms_norm_eps,
            );
        }

        crate::compute::rope::apply_rope_multi_head_neox_partial_scaled(
            &mut q_buf,
            &mut k_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            head_dim,
            0.0,
            cfg.rope_freq_base,
        );
        crate::compute::attention::multi_head_attention(
            &q_buf,
            &k_buf,
            &v_buf,
            &mut attn_out,
            &crate::compute::attention::AttentionParams::new(n_heads, n_kv_heads, head_dim),
            1,
        );
        cpu.dequant_matmul(wo_raw, wo_dtype, &attn_out, &mut proj_buf, dim, 1, q_dim);
        Qwen3MoeForward::parallel_elementwise_add(&mut hidden, &proj_buf);

        let input_hidden = hidden.clone();
        let mut expected_hidden = input_hidden.clone();
        Qwen3MoeForward::apply_moe_ffn_single(
            &cpu,
            &weights,
            &prefix,
            &mut expected_hidden,
            &mut norm_buf,
            &mut moe_scratch,
            dim,
            n_expert,
            n_expert_used,
            expert_inter_dim,
            cfg.rms_norm_eps,
        )
        .unwrap();

        let (gate_dtype, up_dtype, down_dtype) =
            crate::model::shared::routed_moe_expert_dtypes(&weights, &prefix).unwrap();
        let gate_stride =
            crate::model::moe_utils::expert_byte_stride(gate_dtype, expert_inter_dim * dim);
        let up_stride =
            crate::model::moe_utils::expert_byte_stride(up_dtype, expert_inter_dim * dim);
        let down_stride =
            crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);
        let cached_layer = &cached.layers[layer];

        let ffn_norm = {
            let weight_cache = metal_ops.lock_weight_cache();
            weight_cache.get(&cached_layer.ffn_norm).unwrap().clone()
        };
        let router = {
            let moe_weight_cache = metal_ops.lock_moe_weight_cache();
            moe_weight_cache
                .get(&cached_layer.moe_router.unwrap())
                .unwrap()
                .clone()
        };
        let gate = {
            let moe_weight_cache = metal_ops.lock_moe_weight_cache();
            moe_weight_cache
                .get(&cached_layer.moe_expert_gate.as_ref().unwrap()[0])
                .unwrap()
                .clone()
        };
        let up = {
            let moe_weight_cache = metal_ops.lock_moe_weight_cache();
            moe_weight_cache
                .get(&cached_layer.moe_expert_up.as_ref().unwrap()[0])
                .unwrap()
                .clone()
        };
        let down = {
            let moe_weight_cache = metal_ops.lock_moe_weight_cache();
            moe_weight_cache
                .get(&cached_layer.moe_expert_down.as_ref().unwrap()[0])
                .unwrap()
                .clone()
        };

        {
            let mut bs_guard = metal_ops.batch_scratches();
            let bs = bs_guard.as_mut().unwrap();
            unsafe {
                bs.hidden.as_mut_slice::<f32>()[..dim].copy_from_slice(&input_hidden);
            }
            let moe_scratch =
                crate::backend::metal::MoeBatchScratchView::from_batch_scratches(bs).unwrap();

            metal_ops
                .device
                .execute_sync(|encoder| {
                    metal_ops.encode_moe_ffn_gpu_resident_cached_with_scratch_with_policy(
                        encoder,
                        moe_scratch,
                        &bs.hidden,
                        &ffn_norm,
                        &router,
                        cached_layer.moe_router_dtype.unwrap(),
                        &gate,
                        gate_dtype,
                        &up,
                        up_dtype,
                        &down,
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
                        None,
                        true,
                        Qwen3MoeForward::qwen3moe_blocked_q6q8_down_enabled(),
                    )
                })
                .unwrap();
        }

        let actual_hidden = {
            let bs_guard = metal_ops.batch_scratches();
            let bs = bs_guard.as_ref().unwrap();
            unsafe { bs.hidden.as_slice::<f32>()[..dim].to_vec() }
        };
        assert!(
            actual_hidden.iter().all(|value| value.is_finite()),
            "Qwen3-Coder Q5_K_M MoE tail sequence produced non-finite values at layer {layer}",
        );
        let diff = max_abs_diff(&actual_hidden, &expected_hidden);
        let scale = expected_hidden
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max)
            .max(1.0);
        assert!(
            diff / scale < 1e-2,
            "Qwen3-Coder Q5_K_M MoE tail sequence mismatch at layer {layer}: rel_diff={} max_diff={} actual[0..8]={:?} expected[0..8]={:?}",
            diff / scale,
            diff,
            &actual_hidden[..8],
            &expected_hidden[..8],
        );

        metal_ops.init_scratches(&cfg);
        {
            let mut scratch_guard = metal_ops.scratches();
            let scratch = scratch_guard.as_mut().unwrap();
            unsafe {
                scratch.hidden.as_mut_slice::<f32>()[..dim].copy_from_slice(&input_hidden);
            }
        }
        {
            let scratch_guard = metal_ops.scratches();
            let scratch = scratch_guard.as_ref().unwrap();
            metal_ops
                .device
                .execute_sync(|encoder| {
                    metal_ops.encode_moe_ffn_gpu_resident_cached_with_policy(
                        encoder,
                        &scratch.hidden,
                        &ffn_norm,
                        &router,
                        cached_layer.moe_router_dtype.unwrap(),
                        &gate,
                        gate_dtype,
                        &up,
                        up_dtype,
                        &down,
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
                        None,
                        true,
                        Qwen3MoeForward::qwen3moe_blocked_q6q8_down_enabled(),
                    )
                })
                .unwrap();
        }

        let actual_hidden_decode_scratch = {
            let scratch_guard = metal_ops.scratches();
            let scratch = scratch_guard.as_ref().unwrap();
            unsafe { scratch.hidden.as_slice::<f32>()[..dim].to_vec() }
        };
        assert!(
            actual_hidden_decode_scratch
                .iter()
                .all(|value| value.is_finite()),
            "Qwen3-Coder Q5_K_M MoE tail sequence produced non-finite values on decode scratch at layer {layer}",
        );
        let decode_scratch_diff = max_abs_diff(&actual_hidden_decode_scratch, &expected_hidden);
        assert!(
            decode_scratch_diff / scale < 1e-2,
            "Qwen3-Coder Q5_K_M MoE tail sequence mismatch on decode scratch at layer {layer}: rel_diff={} max_diff={} actual[0..8]={:?} expected[0..8]={:?}",
            decode_scratch_diff / scale,
            decode_scratch_diff,
            &actual_hidden_decode_scratch[..8],
            &expected_hidden[..8],
        );

        hidden = expected_hidden;
    }
}

#[test]
fn test_real_qwen3_coder_q5_layer6_attention_residual_matches_cpu_after_full_gpu_prefix() {
    let _env_lock = crate::test_env_lock();
    let _selected_off = EnvVarGuard::set("AX_QWEN35_SELECTED_EXPERT_SINGLE_TOKEN", "0");
    let _smart_off = EnvVarGuard::set("AX_METAL_SMART_BARRIERS", "0");
    let _barriers_on = EnvVarGuard::set("AX_METAL_BARRIERS", "1");

    let path = workspace_model_path("Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&mapped.header).unwrap();
    let prompt_token_ids = tokenizer.encode("Hello", true);
    let Some(&token_id) = prompt_token_ids.first() else {
        return;
    };

    let dim = cfg.embedding_dim as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let n_expert = cfg.n_expert.unwrap_or(0) as usize;
    let n_expert_used = cfg.n_expert_used.unwrap_or(0) as usize;
    let expert_inter_dim = cfg.expert_intermediate_dim.unwrap_or(0) as usize;

    let metal = MetalBackend::new().unwrap();
    let metal_ops = metal.metal_ops().unwrap();
    let mut metal_kv =
        InferenceModel::with_backend(cfg.clone(), Box::new(MetalBackend::new().unwrap()))
            .unwrap()
            .create_model_kv_for_weights(&weights);
    let gpu_kv = metal_kv.as_gpu_mut().unwrap();
    gpu_kv.ensure_capacity(&metal_ops.device, 1).unwrap();

    metal_ops.init_scratches(&cfg);
    metal_ops.init_batch_scratches(&cfg, 1);
    if !metal_ops.has_cached_model_keys() {
        Qwen3MoeForward::build_cached_model_keys(metal_ops, &weights, &cfg).unwrap();
    }

    let cached_guard = metal_ops.cached_model_keys();
    let cached = cached_guard.as_ref().unwrap();
    let weight_cache = metal_ops.lock_weight_cache();
    let moe_weight_cache = metal_ops.lock_moe_weight_cache();

    let prefix0 = "blk.0";
    let (gate_dtype, up_dtype, down_dtype) =
        crate::model::shared::routed_moe_expert_dtypes(&weights, prefix0).unwrap();
    let gate_stride =
        crate::model::moe_utils::expert_byte_stride(gate_dtype, expert_inter_dim * dim);
    let up_stride = crate::model::moe_utils::expert_byte_stride(up_dtype, expert_inter_dim * dim);
    let down_stride =
        crate::model::moe_utils::expert_byte_stride(down_dtype, dim * expert_inter_dim);
    let mut exec_plan = Qwen3MoeForward::qwen3moe_decode_plan(
        metal_ops,
        gpu_kv,
        cfg.embedding_dim,
        cfg.head_dim,
        1,
        true,
        gate_dtype,
        up_dtype,
        down_dtype,
    );
    if std::env::var("AX_DEBUG_QWEN3_Q5_FORCE_EXPLICIT").is_ok() {
        exec_plan.barriers = crate::model::execution_plan::DecodeBarrierPlan::Explicit;
    }

    let cpu = crate::backend::cpu::CpuBackend;
    let mut hidden_cpu = vec![0.0f32; dim];
    weights
        .dequantize_row("token_embd.weight", token_id as usize, &mut hidden_cpu)
        .unwrap();
    let mut norm_buf = vec![0.0f32; dim];
    let mut q_buf = vec![0.0f32; q_dim];
    let mut k_buf = vec![0.0f32; kv_dim];
    let mut v_buf = vec![0.0f32; kv_dim];
    let mut attn_out = vec![0.0f32; q_dim];
    let mut proj_buf = vec![0.0f32; dim];
    let mut moe_scratch = MoeSingleScratch {
        gate_buf: vec![0.0f32; expert_inter_dim],
        up_buf: vec![0.0f32; expert_inter_dim],
        down_buf: vec![0.0f32; dim],
        accum_buf: vec![0.0f32; dim],
        router_logits: vec![0.0f32; n_expert],
    };

    {
        let mut scratch_guard = metal_ops.scratches();
        let scratch = scratch_guard.as_mut().unwrap();
        unsafe {
            scratch.hidden.as_mut_slice::<f32>()[..dim].copy_from_slice(&hidden_cpu);
        }
    }

    for layer in 0..6usize {
        let prefix = format!("blk.{layer}");
        let attn_norm_w = weights
            .f32_slice(&format!("{prefix}.attn_norm.weight"))
            .unwrap();
        crate::compute::rms_norm::rms_norm_out(
            &hidden_cpu,
            attn_norm_w,
            &mut norm_buf,
            cfg.rms_norm_eps,
        );

        let (wq_raw, wq_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_q.weight"))
            .unwrap();
        let (wk_raw, wk_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_k.weight"))
            .unwrap();
        let (wv_raw, wv_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_v.weight"))
            .unwrap();
        let (wo_raw, wo_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_output.weight"))
            .unwrap();

        cpu.dequant_matmul(wq_raw, wq_dtype, &norm_buf, &mut q_buf, q_dim, 1, dim);
        cpu.dequant_matmul(wk_raw, wk_dtype, &norm_buf, &mut k_buf, kv_dim, 1, dim);
        cpu.dequant_matmul(wv_raw, wv_dtype, &norm_buf, &mut v_buf, kv_dim, 1, dim);
        if let Some(norm_weights) =
            crate::model::shared::maybe_attention_qk_norm_weights(&weights, &prefix).unwrap()
        {
            crate::model::shared::apply_attention_qk_norm(
                &mut q_buf,
                &mut k_buf,
                n_heads,
                n_kv_heads,
                head_dim,
                norm_weights,
                cfg.rms_norm_eps,
            );
        }
        crate::compute::rope::apply_rope_multi_head_neox_partial_scaled(
            &mut q_buf,
            &mut k_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            head_dim,
            0.0,
            cfg.rope_freq_base,
        );
        crate::compute::attention::multi_head_attention(
            &q_buf,
            &k_buf,
            &v_buf,
            &mut attn_out,
            &crate::compute::attention::AttentionParams::new(n_heads, n_kv_heads, head_dim),
            1,
        );
        cpu.dequant_matmul(wo_raw, wo_dtype, &attn_out, &mut proj_buf, dim, 1, q_dim);
        Qwen3MoeForward::parallel_elementwise_add(&mut hidden_cpu, &proj_buf);
        Qwen3MoeForward::apply_moe_ffn_single(
            &cpu,
            &weights,
            &prefix,
            &mut hidden_cpu,
            &mut norm_buf,
            &mut moe_scratch,
            dim,
            n_expert,
            n_expert_used,
            expert_inter_dim,
            cfg.rms_norm_eps,
        )
        .unwrap();

        let scratch_guard = metal_ops.scratches();
        let scratch = scratch_guard.as_ref().unwrap();
        metal_ops
            .device
            .execute_sync(|encoder| {
                let barrier =
                    crate::model::shared::DecodeBarrierCtx::new(encoder, exec_plan.barriers);
                let mut ops = None;
                Qwen3MoeForward::encode_qwen3moe_gpu_layer_range(
                    encoder,
                    &barrier,
                    metal_ops,
                    &cfg,
                    &scratch.hidden,
                    scratch,
                    gpu_kv,
                    cached,
                    &weight_cache,
                    &moe_weight_cache,
                    &exec_plan,
                    0,
                    layer,
                    layer + 1,
                    gate_dtype,
                    up_dtype,
                    down_dtype,
                    gate_stride,
                    up_stride,
                    down_stride,
                    &mut ops,
                )?;
                barrier.flush();
                Ok(())
            })
            .unwrap();
    }

    let layer = 6usize;
    let prefix = format!("blk.{layer}");
    let attn_norm_w = weights
        .f32_slice(&format!("{prefix}.attn_norm.weight"))
        .unwrap();
    crate::compute::rms_norm::rms_norm_out(
        &hidden_cpu,
        attn_norm_w,
        &mut norm_buf,
        cfg.rms_norm_eps,
    );
    let (wq_raw, wq_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_q.weight"))
        .unwrap();
    let (wk_raw, wk_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_k.weight"))
        .unwrap();
    let (wv_raw, wv_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_v.weight"))
        .unwrap();
    let (wo_raw, wo_dtype) = weights
        .raw_with_dtype(&format!("{prefix}.attn_output.weight"))
        .unwrap();
    cpu.dequant_matmul(wq_raw, wq_dtype, &norm_buf, &mut q_buf, q_dim, 1, dim);
    cpu.dequant_matmul(wk_raw, wk_dtype, &norm_buf, &mut k_buf, kv_dim, 1, dim);
    cpu.dequant_matmul(wv_raw, wv_dtype, &norm_buf, &mut v_buf, kv_dim, 1, dim);
    if let Some(norm_weights) =
        crate::model::shared::maybe_attention_qk_norm_weights(&weights, &prefix).unwrap()
    {
        crate::model::shared::apply_attention_qk_norm(
            &mut q_buf,
            &mut k_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            norm_weights,
            cfg.rms_norm_eps,
        );
    }
    crate::compute::rope::apply_rope_multi_head_neox_partial_scaled(
        &mut q_buf,
        &mut k_buf,
        n_heads,
        n_kv_heads,
        head_dim,
        head_dim,
        0.0,
        cfg.rope_freq_base,
    );
    crate::compute::attention::multi_head_attention(
        &q_buf,
        &k_buf,
        &v_buf,
        &mut attn_out,
        &crate::compute::attention::AttentionParams::new(n_heads, n_kv_heads, head_dim),
        1,
    );
    cpu.dequant_matmul(wo_raw, wo_dtype, &attn_out, &mut proj_buf, dim, 1, q_dim);
    Qwen3MoeForward::parallel_elementwise_add(&mut hidden_cpu, &proj_buf);
    let expected_pre_moe_hidden = hidden_cpu.clone();

    let _skip_routed = EnvVarGuard::set("AX_QWEN35_PROFILE_SKIP_ROUTED_EXPERT", "1");
    let _skip_shared = EnvVarGuard::set("AX_QWEN35_PROFILE_SKIP_SHARED_EXPERT", "1");
    {
        let scratch_guard = metal_ops.scratches();
        let scratch = scratch_guard.as_ref().unwrap();
        metal_ops
            .device
            .execute_sync(|encoder| {
                let barrier =
                    crate::model::shared::DecodeBarrierCtx::new(encoder, exec_plan.barriers);
                let mut ops = None;
                Qwen3MoeForward::encode_qwen3moe_gpu_layer_range(
                    encoder,
                    &barrier,
                    metal_ops,
                    &cfg,
                    &scratch.hidden,
                    scratch,
                    gpu_kv,
                    cached,
                    &weight_cache,
                    &moe_weight_cache,
                    &exec_plan,
                    0,
                    layer,
                    layer + 1,
                    gate_dtype,
                    up_dtype,
                    down_dtype,
                    gate_stride,
                    up_stride,
                    down_stride,
                    &mut ops,
                )?;
                barrier.flush();
                Ok(())
            })
            .unwrap();
    }

    let actual_pre_moe_hidden = {
        let scratch_guard = metal_ops.scratches();
        let scratch = scratch_guard.as_ref().unwrap();
        unsafe { scratch.hidden.as_slice::<f32>()[..dim].to_vec() }
    };
    assert!(
        actual_pre_moe_hidden.iter().all(|value| value.is_finite()),
        "Qwen3-Coder Q5_K_M layer6 attention/residual produced non-finite values after full GPU prefix",
    );
    let diff = max_abs_diff(&actual_pre_moe_hidden, &expected_pre_moe_hidden);
    let scale = expected_pre_moe_hidden
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        diff / scale < 5e-2,
        "Qwen3-Coder Q5_K_M layer6 attention/residual mismatch after full GPU prefix: rel_diff={} max_diff={} actual[0..8]={:?} expected[0..8]={:?}",
        diff / scale,
        diff,
        &actual_pre_moe_hidden[..8],
        &expected_pre_moe_hidden[..8],
    );

    let mut expected_final_hidden = expected_pre_moe_hidden.clone();
    Qwen3MoeForward::apply_moe_ffn_single(
        &cpu,
        &weights,
        &prefix,
        &mut expected_final_hidden,
        &mut norm_buf,
        &mut moe_scratch,
        dim,
        n_expert,
        n_expert_used,
        expert_inter_dim,
        cfg.rms_norm_eps,
    )
    .unwrap();

    let cached_layer = &cached.layers[layer];
    let ffn_norm = weight_cache.get(&cached_layer.ffn_norm).unwrap().clone();
    let router = moe_weight_cache
        .get(&cached_layer.moe_router.unwrap())
        .unwrap()
        .clone();
    let gate = moe_weight_cache
        .get(&cached_layer.moe_expert_gate.as_ref().unwrap()[0])
        .unwrap()
        .clone();
    let up = moe_weight_cache
        .get(&cached_layer.moe_expert_up.as_ref().unwrap()[0])
        .unwrap()
        .clone();
    let down = moe_weight_cache
        .get(&cached_layer.moe_expert_down.as_ref().unwrap()[0])
        .unwrap()
        .clone();

    {
        let mut bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_mut().unwrap();
        unsafe {
            bs.hidden.as_mut_slice::<f32>()[..dim].copy_from_slice(&actual_pre_moe_hidden);
        }
        let moe_scratch =
            crate::backend::metal::MoeBatchScratchView::from_batch_scratches(bs).unwrap();
        metal_ops
            .device
            .execute_sync(|encoder| {
                metal_ops.encode_moe_ffn_gpu_resident_cached_with_scratch_with_policy(
                    encoder,
                    moe_scratch,
                    &bs.hidden,
                    &ffn_norm,
                    &router,
                    cached_layer.moe_router_dtype.unwrap(),
                    &gate,
                    gate_dtype,
                    &up,
                    up_dtype,
                    &down,
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
                    None,
                    true,
                    Qwen3MoeForward::qwen3moe_blocked_q6q8_down_enabled(),
                )
            })
            .unwrap();
    }

    let actual_final_hidden = {
        let bs_guard = metal_ops.batch_scratches();
        let bs = bs_guard.as_ref().unwrap();
        unsafe { bs.hidden.as_slice::<f32>()[..dim].to_vec() }
    };
    assert!(
        actual_final_hidden.iter().all(|value| value.is_finite()),
        "Qwen3-Coder Q5_K_M isolated layer6 MoE on GPU-prefix pre-MoE hidden produced non-finite values",
    );
    let final_diff = max_abs_diff(&actual_final_hidden, &expected_final_hidden);
    let final_scale = expected_final_hidden
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1.0);
    assert!(
        final_diff / final_scale < 5e-2,
        "Qwen3-Coder Q5_K_M isolated layer6 MoE mismatch on GPU-prefix pre-MoE hidden: rel_diff={} max_diff={} actual[0..8]={:?} expected[0..8]={:?}",
        final_diff / final_scale,
        final_diff,
        &actual_final_hidden[..8],
        &expected_final_hidden[..8],
    );

    metal_ops.init_scratches(&cfg);
    {
        let mut scratch_guard = metal_ops.scratches();
        let scratch = scratch_guard.as_mut().unwrap();
        unsafe {
            scratch.hidden.as_mut_slice::<f32>()[..dim].copy_from_slice(&actual_pre_moe_hidden);
        }
    }
    {
        let scratch_guard = metal_ops.scratches();
        let scratch = scratch_guard.as_ref().unwrap();
        metal_ops
            .device
            .execute_sync(|encoder| {
                metal_ops.encode_moe_ffn_gpu_resident_cached_with_policy(
                    encoder,
                    &scratch.hidden,
                    &ffn_norm,
                    &router,
                    cached_layer.moe_router_dtype.unwrap(),
                    &gate,
                    gate_dtype,
                    &up,
                    up_dtype,
                    &down,
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
                    None,
                    true,
                    Qwen3MoeForward::qwen3moe_blocked_q6q8_down_enabled(),
                )
            })
            .unwrap();
    }
    let actual_final_hidden_scratch = {
        let scratch_guard = metal_ops.scratches();
        let scratch = scratch_guard.as_ref().unwrap();
        unsafe { scratch.hidden.as_slice::<f32>()[..dim].to_vec() }
    };
    assert!(
        actual_final_hidden_scratch
            .iter()
            .all(|value| value.is_finite()),
        "Qwen3-Coder Q5_K_M isolated layer6 MoE on decode scratch hidden produced non-finite values",
    );
    let scratch_diff = max_abs_diff(&actual_final_hidden_scratch, &expected_final_hidden);
    assert!(
        scratch_diff / final_scale < 5e-2,
        "Qwen3-Coder Q5_K_M isolated layer6 MoE mismatch on decode scratch hidden: rel_diff={} max_diff={} actual[0..8]={:?} expected[0..8]={:?}",
        scratch_diff / final_scale,
        scratch_diff,
        &actual_final_hidden_scratch[..8],
        &expected_final_hidden[..8],
    );
}

#[test]
fn test_prepare_runtime_for_real_qwen3_coder_primes_cached_model_keys() {
    let _lock = crate::test_env_lock();

    let path = workspace_model_path("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let model = InferenceModel::with_backend(cfg, Box::new(MetalBackend::new().unwrap())).unwrap();

    let metal_ops = model.metal_ops_for_tests().unwrap();
    assert!(!metal_ops.has_cached_model_keys());

    model.prepare_runtime_for_weights(&weights).unwrap();
    assert!(metal_ops.has_cached_model_keys());

    // Second call should be a cheap no-op.
    model.prepare_runtime_for_weights(&weights).unwrap();
    assert!(metal_ops.has_cached_model_keys());
}

#[test]
fn test_prepare_runtime_for_real_qwen3_coder_primes_router_f16_cache() {
    let _lock = crate::test_env_lock();

    let path = workspace_model_path("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let model = InferenceModel::with_backend(cfg, Box::new(MetalBackend::new().unwrap())).unwrap();

    let metal_ops = model.metal_ops_for_tests().unwrap();
    let (router_raw, router_dtype) = weights.raw_with_dtype("blk.0.ffn_gate_inp.weight").unwrap();
    assert_eq!(router_dtype, GgmlType::F32);

    let router_key = metal_ops.ensure_moe_quant_cached(router_raw);
    let router_buf = {
        let cache = metal_ops.lock_moe_weight_cache();
        cache.get(&router_key).unwrap().clone()
    };
    assert!(!metal_ops.has_precomputed_weight(&router_buf));

    model.prepare_runtime_for_weights(&weights).unwrap();

    assert!(metal_ops.has_precomputed_weight(&router_buf));
}

#[test]
fn test_prepare_runtime_for_real_qwen3_coder_primes_fused_qkv_cache() {
    let _lock = crate::test_env_lock();

    let path = workspace_model_path("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }

    let mapped = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&mapped.header).unwrap();
    let weights = WeightStore::new(&mapped);
    let model = InferenceModel::with_backend(cfg, Box::new(MetalBackend::new().unwrap())).unwrap();

    let metal_ops = model.metal_ops_for_tests().unwrap();
    let mut fused_key = None;
    for layer in 0..model.config.n_layers as usize {
        let prefix = format!("blk.{layer}");
        let (wq_raw, wq_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_q.weight"))
            .unwrap();
        let (wk_raw, wk_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_k.weight"))
            .unwrap();
        let (wv_raw, wv_dtype) = weights
            .raw_with_dtype(&format!("{prefix}.attn_v.weight"))
            .unwrap();
        if wq_dtype == wk_dtype
            && wq_dtype == wv_dtype
            && matches!(wq_dtype, GgmlType::Q4K | GgmlType::Q6K)
        {
            fused_key = Some((
                wq_raw.as_ptr() as usize,
                wk_raw.as_ptr() as usize,
                wv_raw.as_ptr() as usize,
            ));
            break;
        }
    }
    let Some(fused_key) = fused_key else {
        return;
    };

    {
        let cache = metal_ops.lock_fused_qkv_weight_cache();
        assert!(!cache.contains_key(&fused_key));
    }

    model.prepare_runtime_for_weights(&weights).unwrap();

    let cache = metal_ops.lock_fused_qkv_weight_cache();
    assert!(cache.contains_key(&fused_key));
}
