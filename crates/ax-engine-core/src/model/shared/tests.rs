use super::*;
use crate::gguf::MappedModel;
use crate::model::ModelConfig;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::MutexGuard;

fn env_lock() -> MutexGuard<'static, ()> {
    crate::test_env_lock()
}

struct EnvVarRestore {
    values: Vec<(String, Option<OsString>)>,
}

impl Drop for EnvVarRestore {
    fn drop(&mut self) {
        for (key, previous) in self.values.iter().rev() {
            match previous {
                Some(prev) => unsafe {
                    std::env::set_var(key, prev);
                },
                None => unsafe {
                    std::env::remove_var(key);
                },
            }
        }
    }
}

fn with_env_var<T>(key: &str, value: &str, f: impl FnOnce() -> T) -> T {
    let _guard = env_lock();
    let _restore = EnvVarRestore {
        values: vec![(key.to_string(), std::env::var_os(key))],
    };
    unsafe {
        std::env::set_var(key, value);
    }
    f()
}

fn with_env_vars<T>(vars: &[(&str, Option<&str>)], f: impl FnOnce() -> T) -> T {
    let _guard = env_lock();
    let _restore = EnvVarRestore {
        values: vars
            .iter()
            .map(|(key, _)| ((*key).to_string(), std::env::var_os(key)))
            .collect(),
    };
    for (key, value) in vars {
        match value {
            Some(value) => unsafe {
                std::env::set_var(key, value);
            },
            None => unsafe {
                std::env::remove_var(key);
            },
        }
    }
    f()
}

#[test]
fn test_q5k_is_supported_gpu_prefill_quant() {
    assert!(gpu_decode_quant_dtype_supported(GgmlType::Q5K));
    assert!(gpu_prefill_quant_dtype_supported(GgmlType::Q5K));
    assert!(gpu_batch_logits_dtype_supported(GgmlType::Q5K));
}

fn workspace_model_path(file_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .join(file_name)
}

#[test]
fn test_q5_1_is_supported_gpu_decode_and_prefill_quant() {
    assert!(gpu_decode_quant_dtype_supported(GgmlType::Q5_1));
    assert!(gpu_prefill_quant_dtype_supported(GgmlType::Q5_1));
    assert!(!gpu_batch_logits_dtype_supported(GgmlType::Q5_1));
}

#[test]
fn test_real_gemma4_q5km_is_gpu_decode_supported_with_mixed_q5_1_layer_weights() {
    let path = workspace_model_path("gemma-4-26B-A4B-it-Q5_K_M.gguf");
    if !path.exists() {
        return;
    }

    let model = MappedModel::open(&path).unwrap();
    let cfg = ModelConfig::from_gguf(&model.header).unwrap();
    let weights = WeightStore::new(&model);
    let has_q5_1_layer_weight = model.tensors.iter().any(|tensor| {
        tensor.dtype == GgmlType::Q5_1
            && tensor.name.starts_with("blk.")
            && LAYER_SUFFIXES
                .iter()
                .any(|suffix| tensor.name.ends_with(suffix))
    });

    assert!(
        has_q5_1_layer_weight,
        "expected Gemma4 Q5_K_M fixture to include at least one Q5_1 layer weight",
    );
    assert!(
        gpu_decode_quant_supported(&cfg, &weights),
        "Gemma4 Q5_K_M mixed-quant model should stay on GPU decode",
    );
}

#[test]
fn test_q5k_prefill_variant_override_parses_known_values() {
    with_env_vars(
        &[
            ("AX_METAL_Q5K_PREFILL_VARIANT", Some("auto")),
            ("AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT", None),
        ],
        || {
            assert_eq!(
                q5k_prefill_variant_override(),
                Q5KPrefillVariantOverride::Auto
            );
        },
    );
    with_env_vars(
        &[
            ("AX_METAL_Q5K_PREFILL_VARIANT", Some("base")),
            ("AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT", None),
        ],
        || {
            assert_eq!(
                q5k_prefill_variant_override(),
                Q5KPrefillVariantOverride::Base
            );
        },
    );
    with_env_vars(
        &[
            ("AX_METAL_Q5K_PREFILL_VARIANT", Some("small")),
            ("AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT", None),
        ],
        || {
            assert_eq!(
                q5k_prefill_variant_override(),
                Q5KPrefillVariantOverride::Small
            );
        },
    );
    with_env_vars(
        &[
            ("AX_METAL_Q5K_PREFILL_VARIANT", Some("auto")),
            ("AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT", Some("small")),
        ],
        || {
            // Primary variant override takes precedence when present.
            assert_eq!(
                q5k_prefill_variant_override(),
                Q5KPrefillVariantOverride::Auto
            );
        },
    );
    with_env_vars(
        &[
            ("AX_METAL_Q5K_PREFILL_VARIANT", None),
            ("AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT", None),
        ],
        || {
            assert_eq!(
                q5k_prefill_variant_override(),
                Q5KPrefillVariantOverride::Auto
            );
        },
    );
}

#[test]
fn test_q5k_prefill_variant_override_accepts_legacy_env_alias() {
    with_env_vars(
        &[
            ("AX_METAL_Q5K_PREFILL_VARIANT", None),
            ("AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT", Some("small")),
        ],
        || {
            assert_eq!(
                q5k_prefill_variant_override(),
                Q5KPrefillVariantOverride::Small
            );
        },
    );
}

#[test]
fn test_q5k_prefill_variant_override_ignores_legacy_alias_when_primary_is_set() {
    with_env_vars(
        &[
            ("AX_METAL_Q5K_PREFILL_VARIANT", Some("base")),
            ("AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT", Some("small")),
        ],
        || {
            assert_eq!(
                q5k_prefill_variant_override(),
                Q5KPrefillVariantOverride::Base
            );
        },
    );
}

#[test]
fn test_q5k_prefill_variant_override_rejects_unknown_values() {
    with_env_var("AX_METAL_Q5K_PREFILL_VARIANT", "invalid", || {
        assert_eq!(
            q5k_prefill_variant_override(),
            Q5KPrefillVariantOverride::Auto
        );
    });
}

#[test]
fn test_q5k_prefill_small_n_auto_eligible_for_predominant_q5k_models_only() {
    assert!(q5k_prefill_small_n_auto_eligible_for_model(
        Some(GgmlType::Q5K),
        true
    ));
    assert!(!q5k_prefill_small_n_auto_eligible_for_model(
        Some(GgmlType::Q4K),
        true
    ));
    assert!(!q5k_prefill_small_n_auto_eligible_for_model(
        Some(GgmlType::Q5K),
        false
    ));
}

#[test]
fn test_env_flag_enabled_parses_known_truthy_values() {
    let key = "AX_TEST_ENV_FLAG_ENABLED";
    assert!(!env_flag_enabled(key));
    with_env_var(key, "1", || {
        assert!(env_flag_enabled(key));
    });
    with_env_var(key, "true", || {
        assert!(env_flag_enabled(key));
    });
    with_env_var(key, "on", || {
        assert!(env_flag_enabled(key));
    });
    with_env_var(key, "false", || {
        assert!(!env_flag_enabled(key));
    });
    with_env_var(key, "", || {
        assert!(!env_flag_enabled(key));
    });
}

#[test]
fn test_warn_gpu_path_issue_once_only_runs_first_warning() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static WARN_COUNT: AtomicUsize = AtomicUsize::new(0);
    let key = "test:unsupported:blk.0.attn_q.weight:Q5K".to_string();

    warn_gpu_path_issue_once(key.clone(), || {
        WARN_COUNT.fetch_add(1, Ordering::Relaxed);
    });
    warn_gpu_path_issue_once(key, || {
        WARN_COUNT.fetch_add(1, Ordering::Relaxed);
    });

    assert_eq!(WARN_COUNT.load(Ordering::Relaxed), 1);
}
