use super::*;
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
