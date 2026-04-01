use super::*;
use std::ffi::OsString;
use std::sync::MutexGuard;

fn env_lock() -> MutexGuard<'static, ()> {
    crate::test_env_lock()
}

struct EnvVarRestore {
    key: String,
    previous: Option<OsString>,
}

impl Drop for EnvVarRestore {
    fn drop(&mut self) {
        match &self.previous {
            Some(prev) => unsafe {
                std::env::set_var(&self.key, prev);
            },
            None => unsafe {
                std::env::remove_var(&self.key);
            },
        }
    }
}

fn with_env_var<T>(key: &str, value: &str, f: impl FnOnce() -> T) -> T {
    let _guard = env_lock();
    let _restore = EnvVarRestore {
        key: key.to_string(),
        previous: std::env::var_os(key),
    };
    unsafe {
        std::env::set_var(key, value);
    }
    f()
}

#[test]
fn test_q5k_is_supported_gpu_prefill_quant() {
    assert!(gpu_decode_quant_dtype_supported(GgmlType::Q5K));
    assert!(gpu_prefill_quant_dtype_supported(GgmlType::Q5K));
    assert!(gpu_batch_logits_dtype_supported(GgmlType::Q5K));
}

#[test]
fn test_q5k_prefill_variant_override_parses_known_values() {
    assert_eq!(
        q5k_prefill_variant_override(),
        Q5KPrefillVariantOverride::Auto
    );
    with_env_var("AX_METAL_Q5K_PREFILL_VARIANT", "base", || {
        assert_eq!(
            q5k_prefill_variant_override(),
            Q5KPrefillVariantOverride::Base
        );
    });
    with_env_var("AX_METAL_Q5K_PREFILL_VARIANT", "small", || {
        assert_eq!(
            q5k_prefill_variant_override(),
            Q5KPrefillVariantOverride::Small
        );
    });
    with_env_var("AX_METAL_Q5K_PREFILL_VARIANT", "auto", || {
        assert_eq!(
            q5k_prefill_variant_override(),
            Q5KPrefillVariantOverride::Auto
        );
    });
}

#[test]
fn test_q5k_prefill_variant_override_accepts_legacy_env_alias() {
    with_env_var("AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT", "small", || {
        assert_eq!(
            q5k_prefill_variant_override(),
            Q5KPrefillVariantOverride::Small
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
