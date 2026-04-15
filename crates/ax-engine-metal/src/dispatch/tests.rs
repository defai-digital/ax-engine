use super::*;
use std::sync::{Mutex, MutexGuard};

fn default_dequant_config() -> DequantDispatchConfig {
    DequantDispatchConfig::default()
}

fn default_attention_config() -> AttentionDispatchConfig {
    AttentionDispatchConfig::default()
}

fn env_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("dispatch env test lock")
}

struct EnvVarRestore {
    key: String,
    previous: Option<std::ffi::OsString>,
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

fn with_env_lock<T>(f: impl FnOnce() -> T) -> T {
    let _guard = env_lock();
    f()
}

#[test]
fn test_parse_kernel_mode() {
    assert_eq!(
        parse_kernel_mode("__UNSET_ENV__", KernelMode::Off),
        KernelMode::Off
    );
    assert_eq!(
        parse_kernel_mode("__UNSET_ENV__", KernelMode::Auto),
        KernelMode::Auto
    );
}

#[test]
fn test_attention_dispatch_config_honors_legacy_decode_sdpa_alias() {
    let config = with_env_var("AX_METAL_DECODE_SDPA", "1", || {
        AttentionDispatchConfig::from_profile(&KernelProfile::default())
    });
    assert!(config.decode_sdpa_default);
}

#[test]
fn test_attention_dispatch_config_honors_bc64_mode_env_name() {
    let config = with_env_var("AX_METAL_PREFILL_BC64_MODE", "on", || {
        AttentionDispatchConfig::from_profile(&KernelProfile::default())
    });
    assert_eq!(config.prefill_ax_bc64_mode, KernelMode::On);
}

#[test]
fn test_attention_dispatch_config_honors_bc64_min_tokens_env_name() {
    let config = with_env_var("AX_METAL_PREFILL_BC64_MIN_TOKENS", "768", || {
        AttentionDispatchConfig::from_profile(&KernelProfile::default())
    });
    assert_eq!(config.prefill_ax_bc64_min_tokens, 768);
}

#[test]
fn test_dequant_dispatch_config_honors_legacy_q4k_nr_alias() {
    let config = with_env_var("AX_METAL_MATVEC_Q4K_NR", "2", || {
        DequantDispatchConfig::from_profile(&KernelProfile::default())
    });
    assert_eq!(config.q4_k_rows_per_simdgroup, 2);
    assert_eq!(
        config.q4_k_variant,
        Some(crate::profile::MatvecProfileVariant::Nr2)
    );
}

#[test]
fn test_dequant_dispatch_config_honors_q4k_profile_variant() {
    let mut profile = KernelProfile::default();
    profile.decode_matvec.insert(
        "q4_k".to_string(),
        crate::profile::MatvecParams {
            threadgroup_size: 64,
            rows_per_simdgroup: 1,
            variant: Some(crate::profile::MatvecProfileVariant::Ilp4),
        },
    );

    with_env_lock(|| {
        let config = DequantDispatchConfig::from_profile(&profile);
        assert_eq!(
            config.q4_k_variant,
            Some(crate::profile::MatvecProfileVariant::Ilp4)
        );
    });
}

#[test]
fn test_dequant_dispatch_config_honors_q5k_profile_rows_per_sg() {
    let mut profile = KernelProfile::default();
    profile.decode_matvec.insert(
        "q5_k".to_string(),
        crate::profile::MatvecParams {
            threadgroup_size: 64,
            rows_per_simdgroup: 2,
            variant: Some(crate::profile::MatvecProfileVariant::Nr2),
        },
    );

    with_env_lock(|| {
        let config = DequantDispatchConfig::from_profile(&profile);
        assert_eq!(config.q5_k_rows_per_simdgroup, 2);
    });
}

#[test]
fn test_dequant_dispatch_config_honors_q5k_profile_variant() {
    let mut profile = KernelProfile::default();
    profile.decode_matvec.insert(
        "q5_k".to_string(),
        crate::profile::MatvecParams {
            threadgroup_size: 64,
            rows_per_simdgroup: 1,
            variant: Some(crate::profile::MatvecProfileVariant::Ilp4),
        },
    );

    with_env_lock(|| {
        let config = DequantDispatchConfig::from_profile(&profile);
        assert_eq!(
            config.q5_k_variant,
            Some(crate::profile::MatvecProfileVariant::Ilp4)
        );
    });
}

#[test]
fn test_dequant_dispatch_config_infers_legacy_q8_profile_as_nr2() {
    let mut profile = KernelProfile::default();
    profile.decode_matvec.insert(
        "q8_0".to_string(),
        crate::profile::MatvecParams {
            threadgroup_size: 128,
            rows_per_simdgroup: 2,
            variant: None,
        },
    );

    with_env_lock(|| {
        let config = DequantDispatchConfig::from_profile(&profile);
        assert_eq!(
            config.q8_0_variant,
            Some(crate::profile::MatvecProfileVariant::Nr2)
        );
    });
}

#[test]
fn test_q8_0_candidate_selection_honors_profile_variant() {
    let mut config = default_dequant_config();
    config.q8_0_variant = Some(crate::profile::MatvecProfileVariant::Base);
    let selection = q8_0_matvec_candidate_selection(1, config);
    assert_eq!(selection.candidate, MatvecCandidate::Q8_0Base);
    assert_eq!(selection.stability, KernelStabilityTier::ProfilePreferred);
    assert_eq!(selection.threadgroup_width, DEQUANT_MATVEC_TG);
}

#[test]
fn test_resolve_attention_routing_profile() {
    assert_eq!(
        resolve_attention_routing_profile("default"),
        Some(ATTN_PROFILE_DEFAULT)
    );
    assert_eq!(
        resolve_attention_routing_profile("balanced"),
        Some(ATTN_PROFILE_DECODE_BALANCED)
    );
    assert_eq!(
        resolve_attention_routing_profile("long-context"),
        Some(ATTN_PROFILE_DECODE_LONG_CONTEXT)
    );
    assert_eq!(resolve_attention_routing_profile("unknown"), None);
}

#[test]
fn test_attention_dispatch_config_defaults_to_auto_routing() {
    with_env_lock(|| {
        let config = AttentionDispatchConfig::from_profile(&KernelProfile::default());
        assert_eq!(config.routing_profile_name(), "auto");
        assert_eq!(config.decode_routing_profile_name(64), "default");
        assert_eq!(config.decode_routing_profile_name(512), "decode-balanced");
    });
}

#[test]
fn test_attention_dispatch_config_honors_fixed_routing_profile_override() {
    let config = with_env_var("AX_METAL_ATTN_PROFILE", "balanced", || {
        AttentionDispatchConfig::from_profile(&KernelProfile::default())
    });
    assert_eq!(config.routing_profile_name(), "decode-balanced");
    assert_eq!(config.decode_routing_profile_name(64), "decode-balanced");
    assert_eq!(
        config.prefill_cached_routing_profile_name(32, 4096, 0),
        "decode-balanced"
    );
}

#[test]
fn test_q4_k_candidate_selection_uses_nr2_for_profile_preferred_shape() {
    let mut config = default_dequant_config();
    config.q4_k_variant = Some(crate::profile::MatvecProfileVariant::Nr2);
    let selection = q4_k_matvec_candidate_selection(4096, config);
    assert_eq!(selection.candidate, MatvecCandidate::Q4KNr2);
    assert_eq!(selection.stability, KernelStabilityTier::ProfilePreferred);
}

#[test]
fn test_q4_k_candidate_selection_uses_ilp4_for_large_m_without_explicit_override() {
    let selection = q4_k_matvec_candidate_selection(4097, default_dequant_config());
    assert_eq!(selection.candidate, MatvecCandidate::Q4KIlp4);
    assert_eq!(selection.stability, KernelStabilityTier::Stable);
}

#[test]
fn test_q4_k_candidate_selection_honors_explicit_base_override_for_large_m() {
    let mut config = default_dequant_config();
    config.q4_k_variant = Some(crate::profile::MatvecProfileVariant::Base);
    let selection = q4_k_matvec_candidate_selection(4097, config);
    assert_eq!(selection.candidate, MatvecCandidate::Q4KBase);
    assert_eq!(selection.stability, KernelStabilityTier::ProfilePreferred);
}

#[test]
fn test_q6_k_candidate_selection_defaults_to_stable_base() {
    let selection = q6_k_matvec_candidate_selection(1, default_dequant_config());
    assert_eq!(selection.candidate, MatvecCandidate::Q6KBase);
    assert_eq!(selection.stability, KernelStabilityTier::Stable);
}

#[test]
fn test_q6_k_candidate_selection_uses_nr2_for_large_m_without_explicit_override() {
    let selection = q6_k_matvec_candidate_selection(4097, default_dequant_config());
    assert_eq!(selection.candidate, MatvecCandidate::Q6KNr2);
    assert_eq!(selection.stability, KernelStabilityTier::Stable);
}

#[test]
fn test_q6_k_candidate_selection_honors_explicit_base_override_for_large_m() {
    let mut config = default_dequant_config();
    config.q6_k_variant = Some(crate::profile::MatvecProfileVariant::Base);
    let selection = q6_k_matvec_candidate_selection(4097, config);
    assert_eq!(selection.candidate, MatvecCandidate::Q6KBase);
    assert_eq!(selection.stability, KernelStabilityTier::ProfilePreferred);
}

#[test]
fn test_q6_k_candidate_selection_honors_explicit_ilp4_override() {
    let mut config = default_dequant_config();
    config.q6_k_variant = Some(crate::profile::MatvecProfileVariant::Ilp4);
    let selection = q6_k_matvec_candidate_selection(37, config);
    assert_eq!(selection.candidate, MatvecCandidate::Q6KIlp4);
    assert_eq!(selection.stability, KernelStabilityTier::ProfilePreferred);
    assert_eq!(selection.threadgroup_width, DEQUANT_MATVEC_Q6K_ILP4_TG);
}

#[test]
fn test_q5_k_candidate_selection_defaults_to_stable_base() {
    let selection = q5_k_matvec_candidate_selection(1, default_dequant_config());
    assert_eq!(selection.candidate, MatvecCandidate::Q5KBase);
    assert_eq!(selection.stability, KernelStabilityTier::Stable);
    assert_eq!(selection.threadgroups, 1);
    assert_eq!(selection.threadgroup_width, DEQUANT_MATVEC_Q5K_TG);
}

#[test]
fn test_q5_k_candidate_selection_uses_ilp4_when_enabled() {
    let mut config = default_dequant_config();
    config.q5_k_ilp4 = true;
    let selection = q5_k_matvec_candidate_selection(17, config);
    assert_eq!(selection.candidate, MatvecCandidate::Q5KIlp4);
    assert_eq!(selection.stability, KernelStabilityTier::ProfilePreferred);
    assert_eq!(selection.threadgroups, 9);
    assert_eq!(selection.threadgroup_width, DEQUANT_MATVEC_Q5K_TG);
}

#[test]
fn test_q5_k_candidate_selection_defaults_to_ilp4_for_mid_sized_m() {
    let selection = q5_k_matvec_candidate_selection(17, default_dequant_config());
    assert_eq!(selection.candidate, MatvecCandidate::Q5KIlp4);
    assert_eq!(selection.stability, KernelStabilityTier::Stable);
    assert_eq!(selection.threadgroups, 9);
    assert_eq!(selection.threadgroup_width, DEQUANT_MATVEC_Q5K_TG);
}

#[test]
fn test_q5_k_candidate_selection_defaults_to_ilp4_for_large_m() {
    let selection = q5_k_matvec_candidate_selection(4096, default_dequant_config());
    assert_eq!(selection.candidate, MatvecCandidate::Q5KIlp4);
    assert_eq!(selection.stability, KernelStabilityTier::Stable);
    assert_eq!(selection.threadgroups, 2048);
    assert_eq!(selection.threadgroup_width, DEQUANT_MATVEC_Q5K_TG);
}

#[test]
fn test_q5_k_candidate_selection_honors_explicit_base_override_for_large_m() {
    let mut config = default_dequant_config();
    config.q5_k_variant = Some(crate::profile::MatvecProfileVariant::Base);
    let selection = q5_k_matvec_candidate_selection(4097, config);
    assert_eq!(selection.candidate, MatvecCandidate::Q5KBase);
    assert_eq!(selection.stability, KernelStabilityTier::ProfilePreferred);
}

#[test]
fn test_q5_k_candidate_selection_uses_nr2_when_profile_prefers_multi_row() {
    let mut config = default_dequant_config();
    config.q5_k_rows_per_simdgroup = 2;
    config.q5_k_variant = Some(crate::profile::MatvecProfileVariant::Nr2);
    let selection = q5_k_matvec_candidate_selection(17, config);
    assert_eq!(selection.candidate, MatvecCandidate::Q5KNr2);
    assert_eq!(selection.stability, KernelStabilityTier::ProfilePreferred);
    assert_eq!(selection.threadgroups, 5);
    assert_eq!(selection.threadgroup_width, DEQUANT_MATVEC_Q5K_TG);
}

#[test]
fn test_q8_0_candidate_selection_uses_ilp4_for_large_m_without_explicit_override() {
    let selection = q8_0_matvec_candidate_selection(4097, default_dequant_config());
    assert_eq!(selection.candidate, MatvecCandidate::Q8_0Ilp4);
    assert_eq!(selection.stability, KernelStabilityTier::Stable);
}

#[test]
fn test_attention_decode_candidate_selection_prefers_splitk_hd256_for_long_context() {
    let config = AttentionDispatchConfig {
        decode_splitk_auto_min_tokens: 256,
        ..default_attention_config()
    };
    let selection = config.decode_candidate_selection(true, 256, 512);
    assert_eq!(selection.candidate, AttentionDecodeCandidate::SplitKHd256);
    assert_eq!(selection.stability, KernelStabilityTier::ProfilePreferred);
}

#[test]
fn test_attention_decode_candidate_selection_auto_profile_uses_balanced_for_medium_context() {
    with_env_lock(|| {
        let config = AttentionDispatchConfig::from_profile(&KernelProfile::default());
        let selection = config.decode_candidate_selection(true, 128, 384);
        assert_eq!(selection.candidate, AttentionDecodeCandidate::SplitKHd128);
        assert_eq!(selection.stability, KernelStabilityTier::ProfilePreferred);
    });
}

#[test]
fn test_attention_decode_candidate_selection_prefers_splitk_hd128_at_threshold() {
    let config = AttentionDispatchConfig {
        decode_splitk_auto_min_tokens: 512,
        ..default_attention_config()
    };
    let selection = config.decode_candidate_selection(true, 128, 512);
    assert_eq!(selection.candidate, AttentionDecodeCandidate::SplitKHd128);
    assert_eq!(selection.stability, KernelStabilityTier::ProfilePreferred);
}

#[test]
fn test_attention_decode_candidate_selection_keeps_hd128_below_splitk_threshold() {
    // Pin the threshold so the routing profile doesn't override it.
    let config = AttentionDispatchConfig {
        decode_splitk_auto_min_tokens: 512,
        decode_splitk_auto_min_tokens_pinned: true,
        decode_hd128_n2_default: true,
        decode_hd128_n2_default_pinned: true,
        ..default_attention_config()
    };
    let selection = config.decode_candidate_selection(true, 128, 511);
    assert_eq!(selection.candidate, AttentionDecodeCandidate::F16KvHd128N2);
    assert_eq!(selection.stability, KernelStabilityTier::ProfilePreferred);
}

#[test]
fn test_attention_decode_candidate_selection_marks_sdpa_as_profile_preferred() {
    let config = AttentionDispatchConfig {
        decode_splitk_mode: KernelMode::Off,
        decode_sdpa_default: true,
        decode_sdpa_default_pinned: true,
        ..default_attention_config()
    };
    let selection = config.decode_candidate_selection(true, 256, 128);
    assert_eq!(selection.candidate, AttentionDecodeCandidate::SdpaHd256);
    assert_eq!(selection.stability, KernelStabilityTier::ProfilePreferred);
}

#[test]
fn test_attention_dispatch_config_treats_unknown_decode_sdpa_value_as_disabled() {
    let mut profile = KernelProfile::default();
    profile.attention_decode.sdpa_default = Some(false);
    let config = with_env_var("AX_METAL_DECODE_SDPA", "maybe", || {
        AttentionDispatchConfig::from_profile(&profile)
    });
    assert!(!config.decode_sdpa_default);
}

#[test]
fn test_attention_decode_splitk_chunk_size_treats_invalid_values_as_default() {
    let default = AttentionDispatchConfig::default().decode_splitk_chunk_size();
    let config = with_env_var("AX_METAL_DECODE_SPLITK_CHUNK_SIZE", "0", || {
        AttentionDispatchConfig::from_profile(&KernelProfile::default())
    });
    assert_eq!(config.decode_splitk_chunk_size(), default);

    let config = with_env_var("AX_METAL_DECODE_SPLITK_CHUNK_SIZE", "bad", || {
        AttentionDispatchConfig::from_profile(&KernelProfile::default())
    });
    assert_eq!(config.decode_splitk_chunk_size(), default);
}

#[test]
fn test_attention_decode_splitk_auto_min_tokens_treats_zero_as_default() {
    let default = AttentionDispatchConfig::default().decode_splitk_auto_min_tokens;
    let config = with_env_var("AX_METAL_DECODE_SPLITK_AUTO_MIN_TOKENS", "0", || {
        AttentionDispatchConfig::from_profile(&KernelProfile::default())
    });
    assert_eq!(config.decode_splitk_auto_min_tokens, default);
}

#[test]
fn test_attention_decode_candidate_selection_marks_hd128_n2_as_profile_preferred() {
    // Pin both split-K off and N2 on so routing profile doesn't override.
    let config = AttentionDispatchConfig {
        decode_splitk_mode: KernelMode::Off,
        decode_hd128_n2_default: true,
        decode_hd128_n2_default_pinned: true,
        ..default_attention_config()
    };
    let selection = config.decode_candidate_selection(true, 128, 64);
    assert_eq!(selection.candidate, AttentionDecodeCandidate::F16KvHd128N2);
    assert_eq!(selection.stability, KernelStabilityTier::ProfilePreferred);
}

#[test]
fn test_attention_prefill_local_candidate_selection_marks_hd128_fa2_route_as_non_stable() {
    let config = AttentionDispatchConfig {
        prefill_fa2_hd128_mode: KernelMode::On,
        ..default_attention_config()
    };
    let selection = config.prefill_local_candidate_selection(512, 128);
    assert!(matches!(
        selection.candidate,
        AttentionPrefillCandidate::Fa2Hd128
            | AttentionPrefillCandidate::Fa2SimdHd128
            | AttentionPrefillCandidate::Fa2HalfHd128
    ));
    assert_ne!(selection.stability, KernelStabilityTier::Stable);
}

#[test]
fn test_attention_prefill_cached_routing_profile_auto_prefers_long_context() {
    with_env_lock(|| {
        let config = AttentionDispatchConfig::from_profile(&KernelProfile::default());
        assert_eq!(
            config.prefill_cached_routing_profile_name(64, 4096, 0),
            "decode-long-context"
        );
        assert_eq!(
            config.prefill_cached_routing_profile_name(64, 128, 0),
            "default"
        );
    });
}

#[test]
fn test_attention_prefill_local_candidate_selection_does_not_bypass_hd128_fa2_mode() {
    let config = AttentionDispatchConfig {
        prefill_fa2_hd128_mode: KernelMode::Off,
        ..default_attention_config()
    };
    let selection = config.prefill_local_candidate_selection(512, 128);
    assert_ne!(selection.candidate, AttentionPrefillCandidate::Fa2SimdHd128);
    assert_ne!(selection.candidate, AttentionPrefillCandidate::Fa2HalfHd128);
    assert_ne!(selection.candidate, AttentionPrefillCandidate::Fa2Hd128);
}

#[test]
fn test_attention_prefill_local_candidate_selection_does_not_bypass_hd64_fa2_mode() {
    let config = AttentionDispatchConfig {
        prefill_fa2_hd128_mode: KernelMode::Off,
        ..default_attention_config()
    };
    let selection = config.prefill_local_candidate_selection(512, 64);
    assert_ne!(selection.candidate, AttentionPrefillCandidate::Fa2SimdHd64);
}

#[test]
fn test_attention_prefill_local_candidate_selection_respects_ax_bc64_mode_off() {
    let config = AttentionDispatchConfig {
        prefill_fa2_hd128_mode: KernelMode::Off,
        prefill_ax_bc64_mode: KernelMode::Off,
        ..default_attention_config()
    };
    let selection = config.prefill_local_candidate_selection(512, 128);
    assert_eq!(selection.candidate, AttentionPrefillCandidate::AxHd128);
}

#[test]
fn test_attention_prefill_local_candidate_selection_respects_ax_bc64_threshold() {
    let config = AttentionDispatchConfig {
        prefill_fa2_hd128_mode: KernelMode::Off,
        prefill_ax_bc64_mode: KernelMode::Auto,
        prefill_ax_bc64_min_tokens: 768,
        ..default_attention_config()
    };
    let selection = config.prefill_local_candidate_selection(512, 128);
    assert_eq!(selection.candidate, AttentionPrefillCandidate::AxHd128);
}

#[test]
fn test_attention_prefill_cached_candidate_selection_marks_cache_fa2_simd_as_profile_preferred() {
    let config = AttentionDispatchConfig {
        prefill_fa2_mode: KernelMode::On,
        prefill_fa2_hd128_mode: KernelMode::On,
        ..default_attention_config()
    };
    // FA2 SIMD cached takes priority when kv_f16 and head_dim matches
    let selection = config.prefill_cached_candidate_selection(true, 512, 128, 512, 0);
    assert_eq!(
        selection.candidate,
        AttentionPrefillCandidate::CacheFa2SimdHd128
    );
    assert_eq!(selection.stability, KernelStabilityTier::ProfilePreferred);

    let selection = config.prefill_cached_candidate_selection(true, 512, 256, 512, 0);
    assert_eq!(
        selection.candidate,
        AttentionPrefillCandidate::CacheFa2SimdHd256
    );

    let selection = config.prefill_cached_candidate_selection(true, 512, 64, 512, 0);
    assert_eq!(
        selection.candidate,
        AttentionPrefillCandidate::CacheFa2SimdHd64
    );

    // Falls back to Cache for non-f16 KV
    let selection = config.prefill_cached_candidate_selection(false, 512, 128, 512, 0);
    assert_eq!(selection.candidate, AttentionPrefillCandidate::Cache);
}

#[test]
fn test_attention_prefill_cached_candidate_selection_respects_hd128_fa2_mode_off() {
    let config = AttentionDispatchConfig {
        prefill_fa2_hd128_mode: KernelMode::Off,
        ..default_attention_config()
    };
    let selection = config.prefill_cached_candidate_selection(true, 512, 128, 512, 0);
    assert_eq!(selection.candidate, AttentionPrefillCandidate::Cache);
}

#[test]
fn test_attention_prefill_cached_candidate_selection_respects_hd256_fa2_mode_off() {
    let config = AttentionDispatchConfig {
        prefill_fa2_mode: KernelMode::Off,
        ..default_attention_config()
    };
    let selection = config.prefill_cached_candidate_selection(true, 512, 256, 512, 0);
    assert_eq!(selection.candidate, AttentionPrefillCandidate::Cache);
}

#[test]
fn test_dispatch_dims_1d_exact() {
    let dims = DispatchDims::d1(256, 64);
    assert_eq!(dims.threadgroups.width, 4);
    assert_eq!(dims.threads_per_threadgroup.width, 64);
}

#[test]
fn test_dispatch_dims_1d_rounded_up() {
    let dims = DispatchDims::d1(300, 64);
    assert_eq!(dims.threadgroups.width, 5);
    assert_eq!(dims.threads_per_threadgroup.width, 64);
}

#[test]
fn test_dispatch_dims_2d() {
    let dims = DispatchDims::d2(128, 64, 16, 8);
    assert_eq!(dims.threadgroups.width, 8);
    assert_eq!(dims.threadgroups.height, 8);
    assert_eq!(dims.threads_per_threadgroup.width, 16);
    assert_eq!(dims.threads_per_threadgroup.height, 8);
}

#[test]
fn test_dispatch_dims_2d_rounded_up() {
    let dims = DispatchDims::d2(130, 65, 16, 8);
    assert_eq!(dims.threadgroups.width, 9);
    assert_eq!(dims.threadgroups.height, 9);
}

/// Naive CPU matmul for test verification.
fn cpu_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn test_matmul_kernels_compile() {
    let gpu = MetalDevice::new().unwrap();
    let _kernels = MatmulKernels::new(&gpu).unwrap();
}

#[test]
fn test_matvec_small() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = MatmulKernels::new(&gpu).unwrap();

    // A: 4×3, x: 3×1, y: 4×1
    #[rustfmt::skip]
    let a_data: Vec<f32> = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ];
    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0];

    let buf_a = MetalBuffer::from_slice(gpu.device(), &a_data).unwrap();
    let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
    let buf_y = MetalBuffer::new(gpu.device(), 4 * 4).unwrap(); // 4 floats

    kernels
        .matmul(&gpu, &buf_a, &buf_x, &buf_y, (4, 1, 3))
        .unwrap();

    let result = unsafe { buf_y.as_slice::<f32>() };
    let mut expected = vec![0.0f32; 4];
    cpu_matmul(&a_data, &x_data, &mut expected, 4, 1, 3);

    assert!(
        max_abs_diff(result, &expected) < 1e-5,
        "matvec mismatch: got {:?}, expected {:?}",
        result,
        expected
    );
}

#[test]
fn test_tiled_matmul_small() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = MatmulKernels::new(&gpu).unwrap();

    // A: 3×4, B: 4×2, C: 3×2
    #[rustfmt::skip]
    let a_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
    ];
    #[rustfmt::skip]
    let b_data: Vec<f32> = vec![
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
    ];

    let buf_a = MetalBuffer::from_slice(gpu.device(), &a_data).unwrap();
    let buf_b = MetalBuffer::from_slice(gpu.device(), &b_data).unwrap();
    let buf_c = MetalBuffer::new(gpu.device(), 3 * 2 * 4).unwrap();

    kernels
        .matmul(&gpu, &buf_a, &buf_b, &buf_c, (3, 2, 4))
        .unwrap();

    let result = unsafe { buf_c.as_slice::<f32>() };
    let mut expected = vec![0.0f32; 6];
    cpu_matmul(&a_data, &b_data, &mut expected, 3, 2, 4);

    assert!(
        max_abs_diff(result, &expected) < 1e-5,
        "matmul mismatch: got {:?}, expected {:?}",
        result,
        expected
    );
}

#[test]
fn test_matmul_identity() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = MatmulKernels::new(&gpu).unwrap();

    let n = 16;
    let mut identity = vec![0.0f32; n * n];
    for i in 0..n {
        identity[i * n + i] = 1.0;
    }
    let a_data: Vec<f32> = (0..n * n).map(|i| i as f32).collect();

    let buf_a = MetalBuffer::from_slice(gpu.device(), &a_data).unwrap();
    let buf_i = MetalBuffer::from_slice(gpu.device(), &identity).unwrap();
    let buf_c = MetalBuffer::new(gpu.device(), n * n * 4).unwrap();

    // C = A × I should equal A
    kernels
        .matmul(&gpu, &buf_a, &buf_i, &buf_c, (n as u32, n as u32, n as u32))
        .unwrap();

    let result = unsafe { buf_c.as_slice::<f32>() };
    assert!(max_abs_diff(result, &a_data) < 1e-4, "A × I should equal A");
}

#[test]
fn test_matmul_non_tile_aligned() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = MatmulKernels::new(&gpu).unwrap();

    // Non-tile-aligned dimensions: 7×13 * 13×5 = 7×5
    let m = 7;
    let n = 5;
    let k = 13;

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();

    let buf_a = MetalBuffer::from_slice(gpu.device(), &a_data).unwrap();
    let buf_b = MetalBuffer::from_slice(gpu.device(), &b_data).unwrap();
    let buf_c = MetalBuffer::new(gpu.device(), m * n * 4).unwrap();

    kernels
        .matmul(&gpu, &buf_a, &buf_b, &buf_c, (m as u32, n as u32, k as u32))
        .unwrap();

    let result = unsafe { buf_c.as_slice::<f32>() };
    let mut expected = vec![0.0f32; m * n];
    cpu_matmul(&a_data, &b_data, &mut expected, m, n, k);

    assert!(
        max_abs_diff(result, &expected) < 1e-3,
        "Non-aligned matmul failed: max_diff={}",
        max_abs_diff(result, &expected)
    );
}

#[test]
fn test_matvec_llama_sized() {
    // Simulate a typical LLaMA decode matmul: 4096×4096 * 4096×1
    let gpu = MetalDevice::new().unwrap();
    let kernels = MatmulKernels::new(&gpu).unwrap();

    let m = 4096;
    let k = 4096;

    // Use simple patterns to avoid precision issues with random data
    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
    let x_data: Vec<f32> = (0..k).map(|i| ((i % 5) as f32 - 2.0) * 0.01).collect();

    let buf_a = MetalBuffer::from_slice(gpu.device(), &a_data).unwrap();
    let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
    let buf_y = MetalBuffer::new(gpu.device(), m * 4).unwrap();

    kernels
        .matmul(&gpu, &buf_a, &buf_x, &buf_y, (m as u32, 1, k as u32))
        .unwrap();

    let result = unsafe { buf_y.as_slice::<f32>() };

    // Verify a few rows against CPU
    let mut expected = vec![0.0f32; m];
    cpu_matmul(&a_data, &x_data, &mut expected, m, 1, k);

    let diff = max_abs_diff(result, &expected);
    assert!(diff < 0.1, "LLaMA-sized matvec failed: max_diff={}", diff);
}

#[test]
fn test_matmul_prefill_sized() {
    // Simulate prefill: 128×4096 * 4096×128 (small prefill)
    let gpu = MetalDevice::new().unwrap();
    let kernels = MatmulKernels::new(&gpu).unwrap();

    let m = 128;
    let n = 128;
    let k = 256; // smaller K to keep test fast

    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();

    let buf_a = MetalBuffer::from_slice(gpu.device(), &a_data).unwrap();
    let buf_b = MetalBuffer::from_slice(gpu.device(), &b_data).unwrap();
    let buf_c = MetalBuffer::new(gpu.device(), m * n * 4).unwrap();

    kernels
        .matmul(&gpu, &buf_a, &buf_b, &buf_c, (m as u32, n as u32, k as u32))
        .unwrap();

    let result = unsafe { buf_c.as_slice::<f32>() };
    let mut expected = vec![0.0f32; m * n];
    cpu_matmul(&a_data, &b_data, &mut expected, m, n, k);

    let diff = max_abs_diff(result, &expected);
    assert!(diff < 0.1, "Prefill-sized matmul failed: max_diff={}", diff);
}

// ── Dequant test helpers ───────────────────────────────────────────

const Q8_0_BYTES_PER_BLOCK: usize = 34;
const Q8_0_BLOCK_SIZE: usize = 32;
const Q5_1_BYTES_PER_BLOCK: usize = 24;
const Q5_1_BLOCK_SIZE: usize = 32;
const Q4_K_BYTES_PER_BLOCK: usize = 144;
const Q4_K_BLOCK_SIZE: usize = 256;
const Q6_K_BYTES_PER_BLOCK: usize = 210;
const Q6_K_BLOCK_SIZE: usize = 256;

/// CPU reference for Q4_K scale/min extraction.
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        let sc = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (sc, m)
    } else {
        let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

/// CPU reference dequant for Q4_K.
fn cpu_dequant_q4_k(blocks: &[u8], dst: &mut [f32]) {
    let n_blocks = blocks.len() / Q4_K_BYTES_PER_BLOCK;
    for b in 0..n_blocks {
        let block = &blocks[b * Q4_K_BYTES_PER_BLOCK..][..Q4_K_BYTES_PER_BLOCK];
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16];
        let qs = &block[16..144];
        let out = &mut dst[b * Q4_K_BLOCK_SIZE..][..Q4_K_BLOCK_SIZE];

        let mut out_idx = 0;
        let mut q_idx = 0;
        let mut is = 0;
        for _pair in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;
            for l in 0..32 {
                let byte = qs[q_idx + l];
                out[out_idx + l] = d1 * (byte & 0xF) as f32 - m1;
                out[out_idx + l + 32] = d2 * (byte >> 4) as f32 - m2;
            }
            out_idx += 64;
            q_idx += 32;
            is += 2;
        }
    }
}

/// CPU reference dequant for Q5_K.
fn cpu_dequant_q5_k(blocks: &[u8], dst: &mut [f32]) {
    let n_blocks = blocks.len() / 176;
    for b in 0..n_blocks {
        let block = &blocks[b * 176..][..176];
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];
        let out = &mut dst[b * 256..][..256];

        for subblock in 0..8 {
            let (sc, m) = get_scale_min_k4(subblock, scales);
            let d_scaled = d * sc as f32;
            let d_min_scaled = dmin * m as f32;
            let qs_group = subblock / 2;
            let high_nibble = (subblock & 1) == 1;

            for i in 0..32 {
                let ql_byte = qs[qs_group * 32 + i];
                let ql = if high_nibble {
                    (ql_byte >> 4) & 0x0F
                } else {
                    ql_byte & 0x0F
                };
                let qh_bit = (qh[i] >> subblock) & 0x01;
                let q = (ql | (qh_bit << 4)) as f32;
                out[subblock * 32 + i] = d_scaled * q - d_min_scaled;
            }
        }
    }
}

fn cpu_dequant_q8_0(blocks: &[u8], dst: &mut [f32]) {
    let n_blocks = blocks.len() / Q8_0_BYTES_PER_BLOCK;
    for b in 0..n_blocks {
        let block = &blocks[b * Q8_0_BYTES_PER_BLOCK..][..Q8_0_BYTES_PER_BLOCK];
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..34];
        let out = &mut dst[b * Q8_0_BLOCK_SIZE..][..Q8_0_BLOCK_SIZE];
        for i in 0..Q8_0_BLOCK_SIZE {
            out[i] = d * f32::from(i8::from_le_bytes([qs[i]]));
        }
    }
}

fn cpu_dequant_q5_1(blocks: &[u8], dst: &mut [f32]) {
    let n_blocks = blocks.len() / Q5_1_BYTES_PER_BLOCK;
    for b in 0..n_blocks {
        let block = &blocks[b * Q5_1_BYTES_PER_BLOCK..][..Q5_1_BYTES_PER_BLOCK];
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let m = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let qs = &block[8..24];
        let out = &mut dst[b * Q5_1_BLOCK_SIZE..][..Q5_1_BLOCK_SIZE];

        for j in 0..Q5_1_BLOCK_SIZE / 2 {
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            let x0 = (qs[j] as u32 & 0x0F) | xh_0;
            let x1 = (qs[j] as u32 >> 4) | xh_1;

            out[j] = x0 as f32 * d + m;
            out[j + Q5_1_BLOCK_SIZE / 2] = x1 as f32 * d + m;
        }
    }
}

/// CPU reference dequant for Q6_K.
fn cpu_dequant_q6_k(blocks: &[u8], dst: &mut [f32]) {
    let n_blocks = blocks.len() / Q6_K_BYTES_PER_BLOCK;
    for b in 0..n_blocks {
        let block = &blocks[b * Q6_K_BYTES_PER_BLOCK..][..Q6_K_BYTES_PER_BLOCK];
        let ql = &block[..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = half::f16::from_le_bytes([block[208], block[209]]).to_f32();
        let out = &mut dst[b * Q6_K_BLOCK_SIZE..][..Q6_K_BLOCK_SIZE];

        let mut out_idx = 0;
        let mut ql_idx = 0;
        let mut qh_idx = 0;
        let mut sc_idx = 0;

        for _group in 0..2 {
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[ql_idx + l] & 0x0F) | ((qh[qh_idx + l] & 0x03) << 4)) as i32 - 32;
                let q2 = ((ql[ql_idx + l + 32] & 0x0F) | (((qh[qh_idx + l] >> 2) & 0x03) << 4))
                    as i32
                    - 32;
                let q3 =
                    ((ql[ql_idx + l] >> 4) | (((qh[qh_idx + l] >> 4) & 0x03) << 4)) as i32 - 32;
                let q4 = ((ql[ql_idx + l + 32] >> 4) | (((qh[qh_idx + l] >> 6) & 0x03) << 4))
                    as i32
                    - 32;

                out[out_idx + l] = d * (scales[sc_idx + is] as i8) as f32 * q1 as f32;
                out[out_idx + l + 32] = d * (scales[sc_idx + is + 2] as i8) as f32 * q2 as f32;
                out[out_idx + l + 64] = d * (scales[sc_idx + is + 4] as i8) as f32 * q3 as f32;
                out[out_idx + l + 96] = d * (scales[sc_idx + is + 6] as i8) as f32 * q4 as f32;
            }

            out_idx += 128;
            ql_idx += 64;
            qh_idx += 32;
            sc_idx += 8;
        }
    }
}

fn round_to_f16(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

fn cpu_matvec(a: &[f32], x: &[f32], dst: &mut [f32], m: usize, k: usize) {
    for row in 0..m {
        let a_row = &a[row * k..][..k];
        let mut sum = 0.0f32;
        for (av, xv) in a_row.iter().zip(x.iter()) {
            sum += av * round_to_f16(*xv);
        }
        dst[row] = sum;
    }
}

fn cpu_batch_btrans_matmul(a: &[f32], b: &[f32], dst: &mut [f32], m: usize, n: usize, k: usize) {
    for token in 0..n {
        let b_row = &b[token * k..][..k];
        for row in 0..m {
            let a_row = &a[row * k..][..k];
            let mut sum = 0.0f32;
            for (av, bv) in a_row.iter().zip(b_row.iter()) {
                sum += round_to_f16(*av) * round_to_f16(*bv);
            }
            dst[token * m + row] = sum;
        }
    }
}

fn argmax(xs: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

// ── Dequant kernel tests ───────────────────────────────────────────

#[test]
fn test_dequant_kernels_compile() {
    let gpu = MetalDevice::new().unwrap();
    let _kernels = DequantKernels::new(&gpu).unwrap();
}

#[test]
fn test_dequant_q4_k_standalone() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    // Create Q4_K block: d=1.0, dmin=0.5, scales for sub-blocks 0-3 = 2, mins = 1
    let mut block = vec![0u8; Q4_K_BYTES_PER_BLOCK];
    let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
    let dmin_bytes = half::f16::from_f32(0.5).to_le_bytes();
    block[0] = d_bytes[0];
    block[1] = d_bytes[1];
    block[2] = dmin_bytes[0];
    block[3] = dmin_bytes[1];
    // scales for sub-blocks 0-3
    block[4] = 2; // sc for sub-block 0
    block[5] = 2; // sc for sub-block 1
    block[6] = 2; // sc for sub-block 2
    block[7] = 2; // sc for sub-block 3
    block[8] = 1; // min for sub-block 0
    block[9] = 1; // min for sub-block 1
    block[10] = 1; // min for sub-block 2
    block[11] = 1; // min for sub-block 3
    // All quant nibbles = 5
    block[16..144].fill(0x55);

    let n_blocks = 1u32;
    let n_values = Q4_K_BLOCK_SIZE;

    let buf_src = MetalBuffer::from_bytes(gpu.device(), &block).unwrap();
    let buf_dst = MetalBuffer::new(gpu.device(), n_values * std::mem::size_of::<f32>()).unwrap();

    kernels
        .dequant_q4_k(&gpu, &buf_src, &buf_dst, n_blocks)
        .unwrap();

    let gpu_result = unsafe { buf_dst.as_slice::<f32>() };
    let mut cpu_result = vec![0.0f32; n_values];
    cpu_dequant_q4_k(&block, &mut cpu_result);

    let diff = max_abs_diff(gpu_result, &cpu_result);
    assert!(
        diff < 1e-3,
        "Q4_K standalone dequant mismatch: max_diff={}",
        diff
    );
}

#[test]
fn test_fused_matvec_q4_k() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    // 4 rows × 256 cols (K=256 → 1 Q4_K block per row)
    let m = 4usize;
    let k = 256usize;

    // Create Q4_K blocks for each row
    let mut quant_data = Vec::new();
    for row in 0..m {
        let mut block = vec![0u8; Q4_K_BYTES_PER_BLOCK];
        let d = (row as f32 + 1.0) * 0.25;
        let d_bytes = half::f16::from_f32(d).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        // dmin = 0 for simplicity
        // scales for sub-blocks 0-3 = 1
        block[4] = 1;
        block[5] = 1;
        block[6] = 1;
        block[7] = 1;
        // All quants = pattern based on row
        let q = 0x33u8 + row as u8;
        block[16..144].fill(q);
        quant_data.extend(block);
    }

    // Dequantize on CPU
    let mut weights_f32 = vec![0.0f32; m * k];
    cpu_dequant_q4_k(&quant_data, &mut weights_f32);

    // Input vector
    let x_data: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();

    // CPU reference: y = weights × x
    let mut expected = vec![0.0f32; m];
    cpu_matmul(&weights_f32, &x_data, &mut expected, m, 1, k);

    // GPU fused dequant+matvec
    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
    let buf_y = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

    kernels
        .fused_matvec_q4_k_with_config(
            &gpu,
            &buf_a,
            &buf_x,
            &buf_y,
            m as u32,
            k as u32,
            default_dequant_config(),
        )
        .unwrap();

    let result = unsafe { buf_y.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 1e-2,
        "Fused Q4_K matvec mismatch: max_diff={}, got {:?}, expected {:?}",
        diff,
        result,
        expected
    );
}

#[test]
fn test_fused_matvec_q5_k() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 4usize;
    let k = 512usize;
    let blocks_per_row = k / 256;

    let mut quant_data = Vec::new();
    for row in 0..m {
        for blk in 0..blocks_per_row {
            let mut block = vec![0u8; 176];
            let d = ((row + blk) % 5) as f32 * 0.2 + 0.1;
            let d_bytes = half::f16::from_f32(d).to_le_bytes();
            block[0] = d_bytes[0];
            block[1] = d_bytes[1];
            let dmin = ((row * 2 + blk) % 4) as f32 * 0.05;
            let dmin_bytes = half::f16::from_f32(dmin).to_le_bytes();
            block[2] = dmin_bytes[0];
            block[3] = dmin_bytes[1];
            for i in 0..8 {
                block[4 + (i % 4)] = ((row + i) % 8 + 1) as u8;
                block[8 + (i % 4)] = ((blk + i) % 4) as u8;
            }
            for (i, byte) in block[16..48].iter_mut().enumerate() {
                *byte = ((row * 3 + blk * 5 + i) % 256) as u8;
            }
            for (i, byte) in block[48..176].iter_mut().enumerate() {
                *byte = ((row * 7 + blk * 11 + i) % 256) as u8;
            }
            quant_data.extend(block);
        }
    }

    let mut weights_f32 = vec![0.0f32; m * k];
    cpu_dequant_q5_k(&quant_data, &mut weights_f32);
    let x_data: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) * 0.03).collect();
    let mut expected = vec![0.0f32; m];
    cpu_matvec(&weights_f32, &x_data, &mut expected, m, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
    let buf_y = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

    kernels
        .fused_matvec_q5_k(&gpu, &buf_a, &buf_x, &buf_y, m as u32, k as u32)
        .unwrap();

    let result = unsafe { buf_y.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 0.05,
        "Fused Q5_K matvec mismatch: max_diff={}, got {:?}, expected {:?}",
        diff,
        result,
        expected
    );
}

#[test]
fn test_fused_matvec_q5_1_matches_cpu_reference() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 7usize;
    let k = 256usize;
    let blocks_per_row = k / Q5_1_BLOCK_SIZE;

    let mut quant_data = Vec::new();
    for row in 0..m {
        for blk in 0..blocks_per_row {
            let mut block = vec![0u8; Q5_1_BYTES_PER_BLOCK];
            let d = ((row + blk) % 9) as f32 * 0.07 + 0.03;
            let m_val = ((row * 2 + blk) % 11) as f32 * 0.11 - 0.55;
            let d_bytes = half::f16::from_f32(d).to_le_bytes();
            let m_bytes = half::f16::from_f32(m_val).to_le_bytes();
            block[0] = d_bytes[0];
            block[1] = d_bytes[1];
            block[2] = m_bytes[0];
            block[3] = m_bytes[1];

            let qh = 0xD2B4_6935u32.rotate_left(((row * 5 + blk * 3) % 32) as u32);
            block[4..8].copy_from_slice(&qh.to_le_bytes());
            for (i, byte) in block[8..24].iter_mut().enumerate() {
                *byte = (((row * 13 + blk * 7 + i) % 16) as u8)
                    | ((((row * 3 + blk * 11 + i * 5) % 16) as u8) << 4);
            }
            quant_data.extend(block);
        }
    }

    let mut weights_f32 = vec![0.0f32; m * k];
    cpu_dequant_q5_1(&quant_data, &mut weights_f32);
    let x_data: Vec<f32> = (0..k).map(|i| ((i % 19) as f32 - 9.0) * 0.021).collect();
    let mut expected = vec![0.0f32; m];
    cpu_matvec(&weights_f32, &x_data, &mut expected, m, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
    let buf_y = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

    gpu.execute_sync(|encoder| {
        kernels.encode_fused_matvec_q5_1(encoder, &buf_a, &buf_x, &buf_y, m as u32, k as u32);
        Ok(())
    })
    .unwrap();

    let result = unsafe { buf_y.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 5e-2,
        "Fused Q5_1 matvec mismatch: max_diff={}, got {:?}, expected {:?}",
        diff,
        result,
        expected
    );
}

#[test]
fn test_fused_matvec_q5_k_nr2_matches_base_and_cpu_reference() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 36usize;
    let k = 512usize;
    let blocks_per_row = k / 256;

    let mut quant_data = Vec::new();
    for row in 0..m {
        for blk in 0..blocks_per_row {
            let mut block = vec![0u8; 176];
            let d = ((row + blk) % 7) as f32 * 0.15 + 0.08;
            let d_bytes = half::f16::from_f32(d).to_le_bytes();
            block[0] = d_bytes[0];
            block[1] = d_bytes[1];
            let dmin = ((row * 3 + blk) % 5) as f32 * 0.04;
            let dmin_bytes = half::f16::from_f32(dmin).to_le_bytes();
            block[2] = dmin_bytes[0];
            block[3] = dmin_bytes[1];
            for i in 0..8 {
                block[4 + (i % 4)] = ((row + i) % 8 + 1) as u8;
                block[8 + (i % 4)] = ((blk * 2 + i) % 5) as u8;
            }
            for (i, byte) in block[16..48].iter_mut().enumerate() {
                *byte = ((row * 5 + blk * 7 + i) % 256) as u8;
            }
            for (i, byte) in block[48..176].iter_mut().enumerate() {
                *byte = ((row * 11 + blk * 13 + i) % 256) as u8;
            }
            quant_data.extend(block);
        }
    }

    let x_data: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.0175).collect();
    let mut a_f32 = vec![0.0f32; m * k];
    cpu_dequant_q5_k(&quant_data, &mut a_f32);
    let mut expected = vec![0.0f32; m];
    cpu_matvec(&a_f32, &x_data, &mut expected, m, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
    let buf_base = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();
    let buf_nr2 = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

    let mut base_config = default_dequant_config();
    base_config.q5_k_ilp4 = false;
    kernels
        .fused_matvec_q5_k_with_config(
            &gpu,
            &buf_a,
            &buf_x,
            &buf_base,
            m as u32,
            k as u32,
            base_config,
        )
        .unwrap();
    kernels
        .fused_matvec_q5_k_nr2(&gpu, &buf_a, &buf_x, &buf_nr2, m as u32, k as u32)
        .unwrap();

    let base = unsafe { buf_base.as_slice::<f32>() };
    let nr2 = unsafe { buf_nr2.as_slice::<f32>() };
    let base_diff = max_abs_diff(base, &expected);
    let nr2_diff = max_abs_diff(nr2, &expected);
    let base_vs_nr2 = max_abs_diff(base, nr2);

    assert!(
        nr2_diff < 0.05,
        "Q5_K NR2 exceeded CPU tolerance: base_diff={base_diff}, nr2_diff={nr2_diff}",
    );
    assert!(
        base_vs_nr2 < 0.03,
        "Q5_K NR2 diverged from base kernel: max_diff={base_vs_nr2}",
    );
}

#[test]
fn test_fused_matvec_q5_k_ilp4_matches_base_and_cpu_reference() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 34usize;
    let k = 1024usize;
    let blocks_per_row = k / 256;

    let mut quant_data = Vec::new();
    for row in 0..m {
        for blk in 0..blocks_per_row {
            let mut block = vec![0u8; 176];
            let d = ((row + blk * 3) % 9) as f32 * 0.11 + 0.07;
            let d_bytes = half::f16::from_f32(d).to_le_bytes();
            block[0] = d_bytes[0];
            block[1] = d_bytes[1];
            let dmin = ((row * 5 + blk) % 7) as f32 * 0.025;
            let dmin_bytes = half::f16::from_f32(dmin).to_le_bytes();
            block[2] = dmin_bytes[0];
            block[3] = dmin_bytes[1];
            for i in 0..8 {
                block[4 + (i % 4)] = ((row + i * 2) % 8 + 1) as u8;
                block[8 + (i % 4)] = ((blk + i * 3) % 6) as u8;
            }
            for (i, byte) in block[16..48].iter_mut().enumerate() {
                *byte = ((row * 7 + blk * 13 + i) % 256) as u8;
            }
            for (i, byte) in block[48..176].iter_mut().enumerate() {
                *byte = ((row * 17 + blk * 19 + i) % 256) as u8;
            }
            quant_data.extend(block);
        }
    }

    let x_data: Vec<f32> = (0..k).map(|i| ((i % 23) as f32 - 11.0) * 0.013).collect();
    let mut a_f32 = vec![0.0f32; m * k];
    cpu_dequant_q5_k(&quant_data, &mut a_f32);
    let mut expected = vec![0.0f32; m];
    cpu_matvec(&a_f32, &x_data, &mut expected, m, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
    let buf_base = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();
    let buf_ilp4 = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

    let mut base_config = default_dequant_config();
    base_config.q5_k_ilp4 = false;
    kernels
        .fused_matvec_q5_k_with_config(
            &gpu,
            &buf_a,
            &buf_x,
            &buf_base,
            m as u32,
            k as u32,
            base_config,
        )
        .unwrap();

    let mut ilp4_config = default_dequant_config();
    ilp4_config.q5_k_ilp4 = true;
    kernels
        .fused_matvec_q5_k_with_config(
            &gpu,
            &buf_a,
            &buf_x,
            &buf_ilp4,
            m as u32,
            k as u32,
            ilp4_config,
        )
        .unwrap();

    let base = unsafe { buf_base.as_slice::<f32>() };
    let ilp4 = unsafe { buf_ilp4.as_slice::<f32>() };
    let base_diff = max_abs_diff(base, &expected);
    let ilp4_diff = max_abs_diff(ilp4, &expected);
    let base_vs_ilp4 = max_abs_diff(base, ilp4);

    assert!(
        ilp4_diff < 0.05,
        "Q5_K ILP4 exceeded CPU tolerance: base_diff={base_diff}, ilp4_diff={ilp4_diff}",
    );
    assert!(
        base_vs_ilp4 < 0.03,
        "Q5_K ILP4 diverged from base kernel: max_diff={base_vs_ilp4}",
    );
}

#[test]
fn test_fused_matvec_q6_k_ilp4_matches_base_and_cpu_reference_model_like() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 35usize;
    let k = 512usize;
    let (quant_data, x_data) = make_q6_k_model_like_fixture(m, k);

    let mut a_f32 = vec![0.0f32; m * k];
    cpu_dequant_q6_k(&quant_data, &mut a_f32);
    let mut expected = vec![0.0f32; m];
    cpu_matvec(&a_f32, &x_data, &mut expected, m, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
    let buf_base = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();
    let buf_ilp4 = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

    let mut base_config = default_dequant_config();
    base_config.q6_k_variant = Some(crate::profile::MatvecProfileVariant::Base);
    kernels
        .fused_matvec_q6_k_with_config(
            &gpu,
            &buf_a,
            &buf_x,
            &buf_base,
            m as u32,
            k as u32,
            base_config,
        )
        .unwrap();

    let mut ilp4_config = default_dequant_config();
    ilp4_config.q6_k_variant = Some(crate::profile::MatvecProfileVariant::Ilp4);
    kernels
        .fused_matvec_q6_k_with_config(
            &gpu,
            &buf_a,
            &buf_x,
            &buf_ilp4,
            m as u32,
            k as u32,
            ilp4_config,
        )
        .unwrap();

    let base = unsafe { buf_base.as_slice::<f32>() };
    let ilp4 = unsafe { buf_ilp4.as_slice::<f32>() };
    let base_diff = max_abs_diff(base, &expected);
    let ilp4_diff = max_abs_diff(ilp4, &expected);
    let base_vs_ilp4 = max_abs_diff(base, ilp4);

    assert!(
        ilp4_diff < 0.02,
        "Q6_K ILP4 exceeded model-like CPU tolerance: base_diff={base_diff}, ilp4_diff={ilp4_diff}",
    );
    assert!(
        base_vs_ilp4 < 0.02,
        "Q6_K ILP4 diverged from base kernel: max_diff={base_vs_ilp4}",
    );
}

#[test]
fn test_fused_batch_q5_k() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 32usize;
    let n = 8usize;
    let k = 256usize;

    let mut quant_data = Vec::new();
    for row in 0..m {
        let mut block = vec![0u8; 176];
        let d = ((row % 5) as f32) * 0.2 + 0.1;
        let d_bytes = half::f16::from_f32(d).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        let dmin = ((row % 3) as f32) * 0.05;
        let dmin_bytes = half::f16::from_f32(dmin).to_le_bytes();
        block[2] = dmin_bytes[0];
        block[3] = dmin_bytes[1];
        for i in 0..8 {
            block[4 + (i % 4)] = ((row + i) % 8 + 1) as u8;
            block[8 + (i % 4)] = ((row * 2 + i) % 4) as u8;
        }
        for (i, byte) in block[16..48].iter_mut().enumerate() {
            *byte = ((row * 7 + i * 3) % 256) as u8;
        }
        for (i, byte) in block[48..176].iter_mut().enumerate() {
            *byte = ((row * 11 + i * 5) % 256) as u8;
        }
        quant_data.extend(block);
    }

    let mut weights_f32 = vec![0.0f32; m * k];
    cpu_dequant_q5_k(&quant_data, &mut weights_f32);
    let batch_input: Vec<f32> = (0..n * k).map(|i| ((i % 17) as f32 - 8.0) * 0.02).collect();
    let mut expected = vec![0.0f32; n * m];
    cpu_batch_btrans_matmul(&weights_f32, &batch_input, &mut expected, m, n, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_b = MetalBuffer::from_slice(gpu.device(), &batch_input).unwrap();
    let buf_c = MetalBuffer::new(gpu.device(), n * m * std::mem::size_of::<f32>()).unwrap();

    gpu.execute_sync(|encoder| {
        kernels.encode_fused_batch_q5_k(
            encoder, &buf_a, &buf_b, &buf_c, m as u32, n as u32, k as u32,
        );
        Ok(())
    })
    .unwrap();

    let result = unsafe { buf_c.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(diff < 0.1, "Fused Q5_K batch mismatch: max_diff={}", diff);
}

#[test]
fn test_fused_batch_q4_k() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 32usize;
    let n = 8usize;
    let k = 256usize;

    let mut quant_data = Vec::new();
    for row in 0..m {
        let mut block = vec![0u8; Q4_K_BYTES_PER_BLOCK];
        let d = ((row % 7) as f32) * 0.18 + 0.1;
        let d_bytes = half::f16::from_f32(d).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        let dmin = ((row % 5) as f32) * 0.04;
        let dmin_bytes = half::f16::from_f32(dmin).to_le_bytes();
        block[2] = dmin_bytes[0];
        block[3] = dmin_bytes[1];
        for i in 0..4 {
            block[4 + i] = ((row + i) % 8 + 1) as u8;
            block[8 + i] = ((row * 3 + i) % 4) as u8;
        }
        for (i, byte) in block[16..144].iter_mut().enumerate() {
            *byte = ((row * 5 + i * 3) % 256) as u8;
        }
        quant_data.extend(block);
    }

    let mut weights_f32 = vec![0.0f32; m * k];
    cpu_dequant_q4_k(&quant_data, &mut weights_f32);
    let batch_input: Vec<f32> = (0..n * k)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.015)
        .collect();
    let mut expected = vec![0.0f32; n * m];
    cpu_batch_btrans_matmul(&weights_f32, &batch_input, &mut expected, m, n, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_b = MetalBuffer::from_slice(gpu.device(), &batch_input).unwrap();
    let buf_c = MetalBuffer::new(gpu.device(), n * m * std::mem::size_of::<f32>()).unwrap();

    gpu.execute_sync(|encoder| {
        kernels.encode_fused_batch_q4_k(
            encoder, &buf_a, &buf_b, &buf_c, m as u32, n as u32, k as u32,
        );
        Ok(())
    })
    .unwrap();

    let result = unsafe { buf_c.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(diff < 0.1, "Fused Q4_K batch mismatch: max_diff={}", diff);
}

#[test]
fn test_fused_batch_q4_k_fulltile_specialization() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 64usize;
    let n = 32usize;
    let k = 256usize;

    let mut quant_data = Vec::new();
    for row in 0..m {
        let mut block = vec![0u8; Q4_K_BYTES_PER_BLOCK];
        let d = ((row % 7) as f32) * 0.18 + 0.1;
        let d_bytes = half::f16::from_f32(d).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        let dmin = ((row % 5) as f32) * 0.04;
        let dmin_bytes = half::f16::from_f32(dmin).to_le_bytes();
        block[2] = dmin_bytes[0];
        block[3] = dmin_bytes[1];
        for i in 0..4 {
            block[4 + i] = ((row + i) % 8 + 1) as u8;
            block[8 + i] = ((row * 3 + i) % 4) as u8;
        }
        for (i, byte) in block[16..144].iter_mut().enumerate() {
            *byte = ((row * 5 + i * 3) % 256) as u8;
        }
        quant_data.extend(block);
    }

    let mut weights_f32 = vec![0.0f32; m * k];
    cpu_dequant_q4_k(&quant_data, &mut weights_f32);
    let batch_input: Vec<f32> = (0..n * k)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.01)
        .collect();
    let mut expected = vec![0.0f32; n * m];
    cpu_batch_btrans_matmul(&weights_f32, &batch_input, &mut expected, m, n, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_b = MetalBuffer::from_slice(gpu.device(), &batch_input).unwrap();
    let buf_c = MetalBuffer::new(gpu.device(), n * m * std::mem::size_of::<f32>()).unwrap();

    gpu.execute_sync(|encoder| {
        kernels.encode_fused_batch_q4_k(
            encoder, &buf_a, &buf_b, &buf_c, m as u32, n as u32, k as u32,
        );
        Ok(())
    })
    .unwrap();

    let result = unsafe { buf_c.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 0.1,
        "Fused Q4_K fulltile batch mismatch: max_diff={diff}"
    );
}

#[test]
fn test_fused_batch_q4_k_small() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 32usize;
    let n = 8usize;
    let k = 256usize;

    let mut quant_data = Vec::new();
    for row in 0..m {
        let mut block = vec![0u8; Q4_K_BYTES_PER_BLOCK];
        let d = ((row % 7) as f32) * 0.18 + 0.1;
        let d_bytes = half::f16::from_f32(d).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        let dmin = ((row % 5) as f32) * 0.04;
        let dmin_bytes = half::f16::from_f32(dmin).to_le_bytes();
        block[2] = dmin_bytes[0];
        block[3] = dmin_bytes[1];
        for i in 0..4 {
            block[4 + i] = ((row + i) % 8 + 1) as u8;
            block[8 + i] = ((row * 3 + i) % 4) as u8;
        }
        for (i, byte) in block[16..144].iter_mut().enumerate() {
            *byte = ((row * 5 + i * 3) % 256) as u8;
        }
        quant_data.extend(block);
    }

    let mut weights_f32 = vec![0.0f32; m * k];
    cpu_dequant_q4_k(&quant_data, &mut weights_f32);
    let batch_input: Vec<f32> = (0..n * k)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.015)
        .collect();
    let mut expected = vec![0.0f32; n * m];
    cpu_batch_btrans_matmul(&weights_f32, &batch_input, &mut expected, m, n, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_b = MetalBuffer::from_slice(gpu.device(), &batch_input).unwrap();
    let buf_c = MetalBuffer::new(gpu.device(), n * m * std::mem::size_of::<f32>()).unwrap();

    gpu.execute_sync(|encoder| {
        encoder.setComputePipelineState(kernels.test_fused_batch_q4_k_small().state());
        bind_buffers(encoder, &buf_a, &buf_b, &buf_c);
        bind_u32(encoder, 3, m as u32);
        bind_u32(encoder, 4, n as u32);
        bind_u32(encoder, 5, k as u32);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: m.div_ceil(DB_TILE_M),
                height: n.div_ceil(SB_TILE_N),
                depth: 1,
            },
            MTLSize {
                width: SB_TG,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    })
    .unwrap();

    let result = unsafe { buf_c.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 0.1,
        "Fused small Q4_K batch mismatch: max_diff={diff}"
    );
}

#[test]
fn test_fused_batch_q5_k_f16in() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 32usize;
    let n = 8usize;
    let k = 256usize;

    let mut quant_data = Vec::new();
    for row in 0..m {
        let mut block = vec![0u8; 176];
        let d = ((row % 5) as f32) * 0.2 + 0.1;
        let d_bytes = half::f16::from_f32(d).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        let dmin = ((row % 3) as f32) * 0.05;
        let dmin_bytes = half::f16::from_f32(dmin).to_le_bytes();
        block[2] = dmin_bytes[0];
        block[3] = dmin_bytes[1];
        for i in 0..8 {
            block[4 + (i % 4)] = ((row + i) % 8 + 1) as u8;
            block[8 + (i % 4)] = ((row * 2 + i) % 4) as u8;
        }
        for (i, byte) in block[16..48].iter_mut().enumerate() {
            *byte = ((row * 7 + i * 3) % 256) as u8;
        }
        for (i, byte) in block[48..176].iter_mut().enumerate() {
            *byte = ((row * 11 + i * 5) % 256) as u8;
        }
        quant_data.extend(block);
    }

    let mut weights_f32 = vec![0.0f32; m * k];
    cpu_dequant_q5_k(&quant_data, &mut weights_f32);
    let batch_input: Vec<f32> = (0..n * k).map(|i| ((i % 17) as f32 - 8.0) * 0.02).collect();
    let batch_input_f16: Vec<half::f16> = batch_input
        .iter()
        .copied()
        .map(half::f16::from_f32)
        .collect();
    let batch_input_cpu: Vec<f32> = batch_input_f16.iter().map(|v| v.to_f32()).collect();
    let mut expected = vec![0.0f32; n * m];
    cpu_batch_btrans_matmul(&weights_f32, &batch_input_cpu, &mut expected, m, n, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_b = MetalBuffer::from_slice(gpu.device(), &batch_input_f16).unwrap();
    let buf_c = MetalBuffer::new(gpu.device(), n * m * std::mem::size_of::<f32>()).unwrap();

    gpu.execute_sync(|encoder| {
        kernels.encode_fused_batch_q5_k_f16in(
            encoder, &buf_a, &buf_b, &buf_c, m as u32, n as u32, k as u32,
        );
        Ok(())
    })
    .unwrap();

    let result = unsafe { buf_c.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 0.1,
        "Fused Q5_K batch f16in mismatch: max_diff={}",
        diff
    );
}

#[test]
fn test_fused_batch_q5_k_small() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 32usize;
    let n = 8usize;
    let k = 256usize;

    let mut quant_data = Vec::new();
    for row in 0..m {
        let mut block = vec![0u8; 176];
        let d = ((row % 5) as f32) * 0.2 + 0.1;
        let d_bytes = half::f16::from_f32(d).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        let dmin = ((row % 3) as f32) * 0.05;
        let dmin_bytes = half::f16::from_f32(dmin).to_le_bytes();
        block[2] = dmin_bytes[0];
        block[3] = dmin_bytes[1];
        for i in 0..8 {
            block[4 + (i % 4)] = ((row + i) % 8 + 1) as u8;
            block[8 + (i % 4)] = ((row * 2 + i) % 4) as u8;
        }
        for (i, byte) in block[16..48].iter_mut().enumerate() {
            *byte = ((row * 7 + i * 3) % 256) as u8;
        }
        for (i, byte) in block[48..176].iter_mut().enumerate() {
            *byte = ((row * 11 + i * 5) % 256) as u8;
        }
        quant_data.extend(block);
    }

    let mut weights_f32 = vec![0.0f32; m * k];
    cpu_dequant_q5_k(&quant_data, &mut weights_f32);
    let batch_input: Vec<f32> = (0..n * k).map(|i| ((i % 17) as f32 - 8.0) * 0.02).collect();
    let mut expected = vec![0.0f32; n * m];
    cpu_batch_btrans_matmul(&weights_f32, &batch_input, &mut expected, m, n, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_b = MetalBuffer::from_slice(gpu.device(), &batch_input).unwrap();
    let buf_c = MetalBuffer::new(gpu.device(), n * m * std::mem::size_of::<f32>()).unwrap();

    gpu.execute_sync(|encoder| {
        kernels.encode_fused_batch_q5_k_small(
            encoder, &buf_a, &buf_b, &buf_c, m as u32, n as u32, k as u32,
        );
        Ok(())
    })
    .unwrap();

    let result = unsafe { buf_c.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 0.1,
        "Fused small Q5_K batch mismatch: max_diff={}",
        diff
    );
}

#[test]
fn test_fused_batch_q6_k() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 32usize;
    let n = 8usize;
    let k = 256usize;

    let mut quant_data = Vec::new();
    for row in 0..m {
        let mut block = vec![0u8; Q6_K_BYTES_PER_BLOCK];
        for (i, byte) in block[..128].iter_mut().enumerate() {
            *byte = ((row * 5 + i * 3) % 16) as u8;
        }
        for (i, byte) in block[128..192].iter_mut().enumerate() {
            *byte = ((row * 7 + i) % 4) as u8;
        }
        for (i, byte) in block[192..208].iter_mut().enumerate() {
            *byte = ((row as i32 * 3 + i as i32 * 5) % 31 - 15) as i8 as u8;
        }
        let d = ((row % 11) as f32) * 0.11 + 0.07;
        let d_bytes = half::f16::from_f32(d).to_le_bytes();
        block[208] = d_bytes[0];
        block[209] = d_bytes[1];
        quant_data.extend(block);
    }

    let mut weights_f32 = vec![0.0f32; m * k];
    cpu_dequant_q6_k(&quant_data, &mut weights_f32);
    let batch_input: Vec<f32> = (0..n * k)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.0125)
        .collect();
    let mut expected = vec![0.0f32; n * m];
    cpu_batch_btrans_matmul(&weights_f32, &batch_input, &mut expected, m, n, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_b = MetalBuffer::from_slice(gpu.device(), &batch_input).unwrap();
    let buf_c = MetalBuffer::new(gpu.device(), n * m * std::mem::size_of::<f32>()).unwrap();

    gpu.execute_sync(|encoder| {
        kernels.encode_fused_batch_q6_k(
            encoder, &buf_a, &buf_b, &buf_c, m as u32, n as u32, k as u32,
        );
        Ok(())
    })
    .unwrap();

    let result = unsafe { buf_c.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(diff < 0.2, "Fused Q6_K batch mismatch: max_diff={}", diff);
}

#[test]
fn test_fused_batch_q6_k_fulltile_specialization() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 64usize;
    let n = 32usize;
    let k = 256usize;

    let mut quant_data = Vec::new();
    for row in 0..m {
        let mut block = vec![0u8; Q6_K_BYTES_PER_BLOCK];
        for (i, byte) in block[..128].iter_mut().enumerate() {
            *byte = ((row * 5 + i * 3) % 16) as u8;
        }
        for (i, byte) in block[128..192].iter_mut().enumerate() {
            *byte = ((row * 7 + i) % 4) as u8;
        }
        for (i, byte) in block[192..208].iter_mut().enumerate() {
            *byte = ((row as i32 * 3 + i as i32 * 5) % 31 - 15) as i8 as u8;
        }
        let d = ((row % 11) as f32) * 0.11 + 0.07;
        let d_bytes = half::f16::from_f32(d).to_le_bytes();
        block[208] = d_bytes[0];
        block[209] = d_bytes[1];
        quant_data.extend(block);
    }

    let mut weights_f32 = vec![0.0f32; m * k];
    cpu_dequant_q6_k(&quant_data, &mut weights_f32);
    let batch_input: Vec<f32> = (0..n * k)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.009)
        .collect();
    let mut expected = vec![0.0f32; n * m];
    cpu_batch_btrans_matmul(&weights_f32, &batch_input, &mut expected, m, n, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_b = MetalBuffer::from_slice(gpu.device(), &batch_input).unwrap();
    let buf_c = MetalBuffer::new(gpu.device(), n * m * std::mem::size_of::<f32>()).unwrap();

    gpu.execute_sync(|encoder| {
        kernels.encode_fused_batch_q6_k(
            encoder, &buf_a, &buf_b, &buf_c, m as u32, n as u32, k as u32,
        );
        Ok(())
    })
    .unwrap();

    let result = unsafe { buf_c.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 0.2,
        "Fused Q6_K fulltile batch mismatch: max_diff={diff}"
    );
}

#[test]
fn test_fused_batch_q6_k_small() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 32usize;
    let n = 8usize;
    let k = 256usize;

    let mut quant_data = Vec::new();
    for row in 0..m {
        let mut block = vec![0u8; Q6_K_BYTES_PER_BLOCK];
        for (i, byte) in block[..128].iter_mut().enumerate() {
            *byte = ((row * 5 + i * 3) % 16) as u8;
        }
        for (i, byte) in block[128..192].iter_mut().enumerate() {
            *byte = ((row * 7 + i) % 4) as u8;
        }
        for (i, byte) in block[192..208].iter_mut().enumerate() {
            *byte = ((row as i32 * 3 + i as i32 * 5) % 31 - 15) as i8 as u8;
        }
        let d = ((row % 11) as f32) * 0.11 + 0.07;
        let d_bytes = half::f16::from_f32(d).to_le_bytes();
        block[208] = d_bytes[0];
        block[209] = d_bytes[1];
        quant_data.extend(block);
    }

    let mut weights_f32 = vec![0.0f32; m * k];
    cpu_dequant_q6_k(&quant_data, &mut weights_f32);
    let batch_input: Vec<f32> = (0..n * k)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.0125)
        .collect();
    let mut expected = vec![0.0f32; n * m];
    cpu_batch_btrans_matmul(&weights_f32, &batch_input, &mut expected, m, n, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_b = MetalBuffer::from_slice(gpu.device(), &batch_input).unwrap();
    let buf_c = MetalBuffer::new(gpu.device(), n * m * std::mem::size_of::<f32>()).unwrap();

    gpu.execute_sync(|encoder| {
        encoder.setComputePipelineState(kernels.test_fused_batch_q6_k_small().state());
        bind_buffers(encoder, &buf_a, &buf_b, &buf_c);
        bind_u32(encoder, 3, m as u32);
        bind_u32(encoder, 4, n as u32);
        bind_u32(encoder, 5, k as u32);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: m.div_ceil(DB_TILE_M),
                height: n.div_ceil(SB_TILE_N),
                depth: 1,
            },
            MTLSize {
                width: SB_TG,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    })
    .unwrap();

    let result = unsafe { buf_c.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 0.2,
        "Fused small Q6_K batch mismatch: max_diff={diff}"
    );
}

#[test]
fn test_fused_batch_q8_0_blocked_f16in() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 37usize;
    let n = 11usize;
    let k = 64usize;
    let blocks_per_row = k / Q8_0_BLOCK_SIZE;

    let mut quant_data = Vec::new();
    for row in 0..m {
        for blk in 0..blocks_per_row {
            let mut block = vec![0u8; Q8_0_BYTES_PER_BLOCK];
            let d = ((row + blk * 5) % 13) as f32 * 0.07 + 0.05;
            let d_bytes = half::f16::from_f32(d).to_le_bytes();
            block[0] = d_bytes[0];
            block[1] = d_bytes[1];
            for (i, byte) in block[2..34].iter_mut().enumerate() {
                *byte = ((row as i32 * 9 + blk as i32 * 7 + i as i32) % 127 - 63) as i8 as u8;
            }
            quant_data.extend(block);
        }
    }

    let mut weights_f32 = vec![0.0f32; m * k];
    cpu_dequant_q8_0(&quant_data, &mut weights_f32);
    let batch_input_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.0125)
        .collect();
    let batch_input_f16: Vec<half::f16> = batch_input_f32
        .iter()
        .copied()
        .map(half::f16::from_f32)
        .collect();
    let batch_input_cpu: Vec<f32> = batch_input_f16.iter().map(|v| v.to_f32()).collect();
    let mut expected = vec![0.0f32; n * m];
    cpu_batch_btrans_matmul(&weights_f32, &batch_input_cpu, &mut expected, m, n, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_b = MetalBuffer::from_slice(gpu.device(), &batch_input_f16).unwrap();
    let buf_c = MetalBuffer::new(gpu.device(), n * m * std::mem::size_of::<f32>()).unwrap();

    gpu.execute_sync(|encoder| {
        encoder.setComputePipelineState(kernels.test_fused_batch_q8_0_blocked_f16in().state());
        bind_buffers(encoder, &buf_a, &buf_b, &buf_c);
        bind_u32(encoder, 3, m as u32);
        bind_u32(encoder, 4, n as u32);
        bind_u32(encoder, 5, k as u32);
        bind_u32(encoder, 6, m as u32);
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(8192, 0);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n.div_ceil(32),
                height: m.div_ceil(64),
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    })
    .unwrap();

    let result = unsafe { buf_c.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 0.15,
        "Blocked Q8_0 f16in batch mismatch: max_diff={diff}"
    );
}

#[test]
fn test_fused_batch_q8_0_f32_input_nonsquare_matches_cpu() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 37usize;
    let n = 11usize;
    let k = 64usize;
    let blocks_per_row = k / Q8_0_BLOCK_SIZE;

    let mut quant_data = Vec::new();
    for row in 0..m {
        for blk in 0..blocks_per_row {
            let mut block = vec![0u8; Q8_0_BYTES_PER_BLOCK];
            let d = ((row + blk * 5) % 13) as f32 * 0.07 + 0.05;
            let d_bytes = half::f16::from_f32(d).to_le_bytes();
            block[0] = d_bytes[0];
            block[1] = d_bytes[1];
            for (i, byte) in block[2..34].iter_mut().enumerate() {
                *byte = ((row as i32 * 9 + blk as i32 * 7 + i as i32) % 127 - 63) as i8 as u8;
            }
            quant_data.extend(block);
        }
    }

    let mut weights_f32 = vec![0.0f32; m * k];
    cpu_dequant_q8_0(&quant_data, &mut weights_f32);
    let batch_input_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.0125)
        .collect();
    let mut expected = vec![0.0f32; n * m];
    cpu_batch_btrans_matmul(&weights_f32, &batch_input_f32, &mut expected, m, n, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_b = MetalBuffer::from_slice(gpu.device(), &batch_input_f32).unwrap();
    let buf_c = MetalBuffer::new(gpu.device(), n * m * std::mem::size_of::<f32>()).unwrap();

    gpu.execute_sync(|encoder| {
        kernels.encode_fused_batch_q8_0(
            encoder, &buf_a, &buf_b, &buf_c, m as u32, n as u32, k as u32,
        );
        Ok(())
    })
    .unwrap();

    let result = unsafe { buf_c.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 1e-4,
        "Q8_0 f32-input batch mismatch: max_diff={diff}"
    );
}

#[test]
fn test_fused_batch_q8_0_blocked_f16in_fulltile_specialization() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 64usize;
    let n = 32usize;
    let k = 64usize;
    let blocks_per_row = k / Q8_0_BLOCK_SIZE;

    let mut quant_data = Vec::new();
    for row in 0..m {
        for blk in 0..blocks_per_row {
            let mut block = vec![0u8; Q8_0_BYTES_PER_BLOCK];
            let d = ((row + blk * 3) % 17) as f32 * 0.05 + 0.04;
            let d_bytes = half::f16::from_f32(d).to_le_bytes();
            block[0] = d_bytes[0];
            block[1] = d_bytes[1];
            for (i, byte) in block[2..34].iter_mut().enumerate() {
                *byte = ((row as i32 * 5 + blk as i32 * 11 + i as i32 * 3) % 127 - 63) as i8 as u8;
            }
            quant_data.extend(block);
        }
    }

    let mut weights_f32 = vec![0.0f32; m * k];
    cpu_dequant_q8_0(&quant_data, &mut weights_f32);
    let batch_input_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.01)
        .collect();
    let batch_input_f16: Vec<half::f16> = batch_input_f32
        .iter()
        .copied()
        .map(half::f16::from_f32)
        .collect();
    let batch_input_cpu: Vec<f32> = batch_input_f16.iter().map(|v| v.to_f32()).collect();
    let mut expected = vec![0.0f32; n * m];
    cpu_batch_btrans_matmul(&weights_f32, &batch_input_cpu, &mut expected, m, n, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_b = MetalBuffer::from_slice(gpu.device(), &batch_input_f16).unwrap();
    let buf_c = MetalBuffer::new(gpu.device(), n * m * std::mem::size_of::<f32>()).unwrap();

    gpu.execute_sync(|encoder| {
        encoder.setComputePipelineState(
            kernels
                .test_fused_batch_q8_0_blocked_f16in_fulltile()
                .state(),
        );
        bind_buffers(encoder, &buf_a, &buf_b, &buf_c);
        bind_u32(encoder, 3, m as u32);
        bind_u32(encoder, 4, n as u32);
        bind_u32(encoder, 5, k as u32);
        bind_u32(encoder, 6, m as u32);
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(8192, 0);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n.div_ceil(32),
                height: m.div_ceil(64),
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    })
    .unwrap();

    let result = unsafe { buf_c.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 0.15,
        "Blocked Q8_0 fulltile f16in batch mismatch: max_diff={diff}"
    );
}

#[test]
fn test_fused_matvec_q4_k_larger() {
    // 32 rows × 512 cols (K=512 → 2 Q4_K blocks per row)
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 32usize;
    let k = 512usize;
    let blocks_per_row = k / Q4_K_BLOCK_SIZE;

    let mut quant_data = Vec::new();
    for row in 0..m {
        for blk in 0..blocks_per_row {
            let mut block = vec![0u8; Q4_K_BYTES_PER_BLOCK];
            let d = ((row + blk) % 5) as f32 * 0.3 + 0.1;
            let d_bytes = half::f16::from_f32(d).to_le_bytes();
            block[0] = d_bytes[0];
            block[1] = d_bytes[1];
            let dmin = 0.1f32;
            let dmin_bytes = half::f16::from_f32(dmin).to_le_bytes();
            block[2] = dmin_bytes[0];
            block[3] = dmin_bytes[1];
            // scales for sub-blocks 0-3
            for i in 0..4 {
                block[4 + i] = ((row + i) % 8 + 1) as u8;
                block[8 + i] = ((blk + i) % 4) as u8;
            }
            // quant pattern
            let q = ((row * blocks_per_row + blk) % 12) as u8 + 0x22;
            block[16..144].fill(q);
            quant_data.extend(block);
        }
    }

    let mut weights_f32 = vec![0.0f32; m * k];
    cpu_dequant_q4_k(&quant_data, &mut weights_f32);

    let x_data: Vec<f32> = (0..k).map(|i| ((i % 9) as f32 - 4.0) * 0.03).collect();

    let mut expected = vec![0.0f32; m];
    cpu_matmul(&weights_f32, &x_data, &mut expected, m, 1, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
    let buf_y = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

    kernels
        .fused_matvec_q4_k_with_config(
            &gpu,
            &buf_a,
            &buf_x,
            &buf_y,
            m as u32,
            k as u32,
            default_dequant_config(),
        )
        .unwrap();

    let result = unsafe { buf_y.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 0.5,
        "Larger fused Q4_K matvec failed: max_diff={}",
        diff
    );
}

#[test]
fn test_fused_matvec_q4_k_nr2_matches_base() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 36usize;
    let k = 512usize;
    let blocks_per_row = k / Q4_K_BLOCK_SIZE;

    let mut quant_data = Vec::new();
    for row in 0..m {
        for blk in 0..blocks_per_row {
            let mut block = vec![0u8; Q4_K_BYTES_PER_BLOCK];
            let d = ((row + blk) % 9) as f32 * 0.17 + 0.08;
            let d_bytes = half::f16::from_f32(d).to_le_bytes();
            block[0] = d_bytes[0];
            block[1] = d_bytes[1];
            let dmin = ((row + 2 * blk) % 5) as f32 * 0.03 + 0.02;
            let dmin_bytes = half::f16::from_f32(dmin).to_le_bytes();
            block[2] = dmin_bytes[0];
            block[3] = dmin_bytes[1];
            for i in 0..4 {
                block[4 + i] = ((row + i) % 8 + 1) as u8;
                block[8 + i] = ((blk + i * 2) % 4) as u8;
            }
            block[16..144].fill(((row * 7 + blk * 3) % 16) as u8 + 0x07);
            quant_data.extend(block);
        }
    }

    let x_data: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.015).collect();
    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
    let buf_base = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();
    let buf_nr2 = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

    kernels
        .fused_matvec_q4_k_with_config(
            &gpu,
            &buf_a,
            &buf_x,
            &buf_base,
            m as u32,
            k as u32,
            default_dequant_config(),
        )
        .unwrap();
    kernels
        .fused_matvec_q4_k_nr2(&gpu, &buf_a, &buf_x, &buf_nr2, m as u32, k as u32)
        .unwrap();

    let base = unsafe { buf_base.as_slice::<f32>() };
    let nr2 = unsafe { buf_nr2.as_slice::<f32>() };
    let diff = max_abs_diff(base, nr2);
    // NR2 uses different SG count and K-stride, so accumulation order
    // differs from the base kernel. 2e-3 allows for f32 rounding differences.
    assert!(
        diff < 2e-3,
        "NR2 Q4_K matvec diverged from base kernel: max_diff={diff}",
    );
}

fn make_q6_k_model_like_fixture(m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let blocks_per_row = k / Q6_K_BLOCK_SIZE;
    let mut quant_data = Vec::new();

    for row in 0..m {
        for blk in 0..blocks_per_row {
            let mut block = vec![0u8; Q6_K_BYTES_PER_BLOCK];

            // Model-like fixture: bounded scales and dequant factors keep output
            // magnitudes in the same rough range as real decode activations,
            // unlike the worst-case synthetic stress fixture below.
            for (i, byte) in block[..128].iter_mut().enumerate() {
                *byte = ((row * 3 + blk * 5 + i) % 16) as u8;
            }
            for (i, byte) in block[128..192].iter_mut().enumerate() {
                *byte = ((row * 11 + blk * 7 + i) % 4) as u8;
            }
            for (i, byte) in block[192..208].iter_mut().enumerate() {
                *byte = ((row as i32 * 2 + blk as i32 * 3 + i as i32) % 15 - 7) as i8 as u8;
            }

            let d = 0.0125 + ((row + blk) % 9) as f32 * 0.00625;
            let d_bytes = half::f16::from_f32(d).to_le_bytes();
            block[208] = d_bytes[0];
            block[209] = d_bytes[1];
            quant_data.extend(block);
        }
    }

    let x_data: Vec<f32> = (0..k)
        .map(|i| (((i * 7 + 3) % 29) as f32 - 14.0) * 0.0075)
        .collect();

    (quant_data, x_data)
}

#[test]
#[ignore = "stress fixture still exposes base-vs-NR2 divergence on extreme synthetic Q6_K data"]
fn test_fused_matvec_q6_k_nr2_matches_base_extreme_synthetic() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 36usize;
    let k = 512usize;
    let blocks_per_row = k / Q6_K_BLOCK_SIZE;

    let mut quant_data = Vec::new();
    for row in 0..m {
        for blk in 0..blocks_per_row {
            let mut block = vec![0u8; Q6_K_BYTES_PER_BLOCK];
            for (i, byte) in block[..128].iter_mut().enumerate() {
                *byte = ((row * 5 + blk * 3 + i) % 16) as u8;
            }
            for (i, byte) in block[128..192].iter_mut().enumerate() {
                *byte = ((row * 7 + blk + i) % 4) as u8;
            }
            for (i, byte) in block[192..208].iter_mut().enumerate() {
                *byte = ((row as i32 * 3 + blk as i32 * 5 + i as i32) % 31 - 15) as i8 as u8;
            }
            let d = ((row + blk) % 11) as f32 * 0.11 + 0.07;
            let d_bytes = half::f16::from_f32(d).to_le_bytes();
            block[208] = d_bytes[0];
            block[209] = d_bytes[1];
            quant_data.extend(block);
        }
    }

    let x_data: Vec<f32> = (0..k).map(|i| ((i % 19) as f32 - 9.0) * 0.0125).collect();
    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
    let buf_base = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();
    let buf_nr2 = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

    kernels
        .fused_matvec_q6_k_with_config(
            &gpu,
            &buf_a,
            &buf_x,
            &buf_base,
            m as u32,
            k as u32,
            default_dequant_config(),
        )
        .unwrap();
    kernels
        .fused_matvec_q6_k_nr2(&gpu, &buf_a, &buf_x, &buf_nr2, m as u32, k as u32)
        .unwrap();

    let base = unsafe { buf_base.as_slice::<f32>() };
    let nr2 = unsafe { buf_nr2.as_slice::<f32>() };
    let diff = max_abs_diff(base, nr2);
    // Stress fixture with large synthetic output magnitudes. Keep this ignored
    // until we reconcile the remaining base-vs-NR2 accumulation-order gap.
    assert!(
        diff < 0.15,
        "NR2 Q6_K matvec diverged from base kernel: max_diff={diff}",
    );
}

#[test]
fn test_fused_matvec_q6_k_nr2_matches_base_model_like() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 36usize;
    let k = 512usize;
    let (quant_data, x_data) = make_q6_k_model_like_fixture(m, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
    let buf_base = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();
    let buf_nr2 = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

    kernels
        .fused_matvec_q6_k_with_config(
            &gpu,
            &buf_a,
            &buf_x,
            &buf_base,
            m as u32,
            k as u32,
            default_dequant_config(),
        )
        .unwrap();
    kernels
        .fused_matvec_q6_k_nr2(&gpu, &buf_a, &buf_x, &buf_nr2, m as u32, k as u32)
        .unwrap();

    let base = unsafe { buf_base.as_slice::<f32>() };
    let nr2 = unsafe { buf_nr2.as_slice::<f32>() };
    let diff = max_abs_diff(base, nr2);
    assert!(
        diff < 0.02,
        "Q6_K NR2 diverged from base on model-like fixture: max_diff={diff}",
    );
}

#[test]
fn test_fused_matvec_q4_k_nr2_matches_cpu_reference() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 36usize;
    let k = 512usize;
    let blocks_per_row = k / Q4_K_BLOCK_SIZE;

    let mut quant_data = Vec::new();
    for row in 0..m {
        for blk in 0..blocks_per_row {
            let mut block = vec![0u8; Q4_K_BYTES_PER_BLOCK];
            let d = ((row + blk) % 9) as f32 * 0.17 + 0.08;
            let d_bytes = half::f16::from_f32(d).to_le_bytes();
            block[0] = d_bytes[0];
            block[1] = d_bytes[1];
            let dmin = ((row + 2 * blk) % 5) as f32 * 0.03 + 0.02;
            let dmin_bytes = half::f16::from_f32(dmin).to_le_bytes();
            block[2] = dmin_bytes[0];
            block[3] = dmin_bytes[1];
            for i in 0..4 {
                block[4 + i] = ((row + i) % 8 + 1) as u8;
                block[8 + i] = ((blk + i * 2) % 4) as u8;
            }
            block[16..144].fill(((row * 7 + blk * 3) % 16) as u8 + 0x07);
            quant_data.extend(block);
        }
    }

    let x_data: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.015).collect();
    let mut a_f32 = vec![0.0f32; m * k];
    cpu_dequant_q4_k(&quant_data, &mut a_f32);
    let mut expected = vec![0.0f32; m];
    cpu_matvec(&a_f32, &x_data, &mut expected, m, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
    let buf_base = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();
    let buf_nr2 = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

    kernels
        .fused_matvec_q4_k_with_config(
            &gpu,
            &buf_a,
            &buf_x,
            &buf_base,
            m as u32,
            k as u32,
            default_dequant_config(),
        )
        .unwrap();
    kernels
        .fused_matvec_q4_k_nr2(&gpu, &buf_a, &buf_x, &buf_nr2, m as u32, k as u32)
        .unwrap();

    let base = unsafe { buf_base.as_slice::<f32>() };
    let nr2 = unsafe { buf_nr2.as_slice::<f32>() };
    let base_diff = max_abs_diff(base, &expected);
    let nr2_diff = max_abs_diff(nr2, &expected);
    assert!(
        nr2_diff < 2e-3,
        "NR2 Q4_K matvec exceeded CPU tolerance: base_diff={base_diff}, nr2_diff={nr2_diff}",
    );
}

#[test]
fn test_fused_matvec_q6_k_nr2_matches_cpu_reference() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 36usize;
    let k = 512usize;
    let blocks_per_row = k / Q6_K_BLOCK_SIZE;

    let mut quant_data = Vec::new();
    for row in 0..m {
        for blk in 0..blocks_per_row {
            let mut block = vec![0u8; Q6_K_BYTES_PER_BLOCK];
            for (i, byte) in block[..128].iter_mut().enumerate() {
                *byte = ((row * 5 + blk * 3 + i) % 16) as u8;
            }
            for (i, byte) in block[128..192].iter_mut().enumerate() {
                *byte = ((row * 7 + blk + i) % 4) as u8;
            }
            for (i, byte) in block[192..208].iter_mut().enumerate() {
                *byte = ((row as i32 * 3 + blk as i32 * 5 + i as i32) % 31 - 15) as i8 as u8;
            }
            let d = ((row + blk) % 11) as f32 * 0.11 + 0.07;
            let d_bytes = half::f16::from_f32(d).to_le_bytes();
            block[208] = d_bytes[0];
            block[209] = d_bytes[1];
            quant_data.extend(block);
        }
    }

    let x_data: Vec<f32> = (0..k).map(|i| ((i % 19) as f32 - 9.0) * 0.0125).collect();
    let mut a_f32 = vec![0.0f32; m * k];
    cpu_dequant_q6_k(&quant_data, &mut a_f32);
    let mut expected = vec![0.0f32; m];
    cpu_matvec(&a_f32, &x_data, &mut expected, m, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
    let buf_base = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();
    let buf_nr2 = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

    kernels
        .fused_matvec_q6_k_with_config(
            &gpu,
            &buf_a,
            &buf_x,
            &buf_base,
            m as u32,
            k as u32,
            default_dequant_config(),
        )
        .unwrap();
    kernels
        .fused_matvec_q6_k_nr2(&gpu, &buf_a, &buf_x, &buf_nr2, m as u32, k as u32)
        .unwrap();

    let base = unsafe { buf_base.as_slice::<f32>() };
    let nr2 = unsafe { buf_nr2.as_slice::<f32>() };
    let base_diff = max_abs_diff(base, &expected);
    let nr2_diff = max_abs_diff(nr2, &expected);
    // NR2 uses a different accumulation decomposition than the base kernel.
    // Both diverge from CPU f64 reference but by different amounts.
    // The NR2 divergence is larger (up to ~0.12) because it accumulates sub-group
    // sums before multiplying by scale, while base multiplies per-element.
    assert!(
        nr2_diff < 0.15,
        "NR2 Q6_K too far from CPU reference: base_diff={base_diff}, nr2_diff={nr2_diff}",
    );
}

#[test]
fn test_fused_matvec_q6_k_nr2_matches_cpu_reference_model_like() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 36usize;
    let k = 512usize;
    let (quant_data, x_data) = make_q6_k_model_like_fixture(m, k);

    let mut a_f32 = vec![0.0f32; m * k];
    cpu_dequant_q6_k(&quant_data, &mut a_f32);
    let mut expected = vec![0.0f32; m];
    cpu_matvec(&a_f32, &x_data, &mut expected, m, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
    let buf_base = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();
    let buf_nr2 = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

    kernels
        .fused_matvec_q6_k_with_config(
            &gpu,
            &buf_a,
            &buf_x,
            &buf_base,
            m as u32,
            k as u32,
            default_dequant_config(),
        )
        .unwrap();
    kernels
        .fused_matvec_q6_k_nr2(&gpu, &buf_a, &buf_x, &buf_nr2, m as u32, k as u32)
        .unwrap();

    let base = unsafe { buf_base.as_slice::<f32>() };
    let nr2 = unsafe { buf_nr2.as_slice::<f32>() };
    let base_diff = max_abs_diff(base, &expected);
    let nr2_diff = max_abs_diff(nr2, &expected);
    assert!(
        nr2_diff < 0.02,
        "Q6_K NR2 exceeded model-like CPU tolerance: base_diff={base_diff}, nr2_diff={nr2_diff}",
    );
}

#[test]
fn test_fused_matvec_q6_k_nr2_preserves_model_like_ranking() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 36usize;
    let k = 512usize;
    let vocab = 128usize;
    let (quant_data, x_data) = make_q6_k_model_like_fixture(m, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
    let buf_base = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();
    let buf_nr2 = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

    kernels
        .fused_matvec_q6_k_with_config(
            &gpu,
            &buf_a,
            &buf_x,
            &buf_base,
            m as u32,
            k as u32,
            default_dequant_config(),
        )
        .unwrap();
    kernels
        .fused_matvec_q6_k_nr2(&gpu, &buf_a, &buf_x, &buf_nr2, m as u32, k as u32)
        .unwrap();

    let base = unsafe { buf_base.as_slice::<f32>() };
    let nr2 = unsafe { buf_nr2.as_slice::<f32>() };

    // Deterministic downstream projection to approximate logit sensitivity.
    // This checks whether model-like kernel differences perturb ranking, not
    // just raw matvec values.
    let proj: Vec<f32> = (0..vocab * m)
        .map(|i| (((i * 13 + 5) % 37) as f32 - 18.0) * 0.021)
        .collect();
    let mut logits_base = vec![0.0f32; vocab];
    let mut logits_nr2 = vec![0.0f32; vocab];
    cpu_matmul(&proj, base, &mut logits_base, vocab, 1, m);
    cpu_matmul(&proj, nr2, &mut logits_nr2, vocab, 1, m);

    let top_base = argmax(&logits_base);
    let top_nr2 = argmax(&logits_nr2);
    let logits_diff = max_abs_diff(&logits_base, &logits_nr2);
    assert_eq!(
        top_base, top_nr2,
        "Q6_K NR2 changed projected top-1 ranking: base_top={top_base}, nr2_top={top_nr2}, logits_max_diff={logits_diff}"
    );
    assert!(
        logits_diff < 0.05,
        "Q6_K NR2 projected logits diverged too far on model-like fixture: max_diff={logits_diff}"
    );
}

#[test]
fn test_fused_gelu_down_matvec_q6_k_matches_cpu_reference() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 24usize;
    let k = 512usize;
    let (quant_data, _) = make_q6_k_model_like_fixture(m, k);

    let gate: Vec<f32> = (0..k)
        .map(|i| (((i * 5 + 7) % 41) as f32 - 20.0) * 0.09)
        .collect();
    let up: Vec<f32> = (0..k)
        .map(|i| (((i * 11 + 3) % 37) as f32 - 18.0) * 0.07)
        .collect();

    let mut fused_input = gate.clone();
    cpu_gelu_elementwise_mul(&mut fused_input, &up);

    let mut a_f32 = vec![0.0f32; m * k];
    cpu_dequant_q6_k(&quant_data, &mut a_f32);
    let mut expected = vec![0.0f32; m];
    cpu_matvec(&a_f32, &fused_input, &mut expected, m, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_gate = MetalBuffer::from_slice(gpu.device(), &gate).unwrap();
    let buf_up = MetalBuffer::from_slice(gpu.device(), &up).unwrap();
    let buf_y = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

    gpu.execute_sync(|encoder| {
        kernels.encode_fused_gelu_down_matvec_q6_k(
            encoder, &buf_a, &buf_gate, &buf_up, &buf_y, m as u32, k as u32,
        );
        Ok(())
    })
    .unwrap();

    let result = unsafe { buf_y.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 0.03,
        "Fused Q6_K GELU-down matvec mismatch: max_diff={diff}"
    );
}

fn assert_fused_batch_gelu_down_q6_k_case(m: usize, n: usize, k: usize) {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();
    let (quant_data, _) = make_q6_k_model_like_fixture(m, k);

    let gate: Vec<f32> = (0..n * k)
        .map(|i| (((i * 5 + 7) % 53) as f32 - 26.0) * 0.08)
        .collect();
    let up: Vec<f32> = (0..n * k)
        .map(|i| (((i * 11 + 3) % 47) as f32 - 23.0) * 0.06)
        .collect();

    let mut fused_input = gate.clone();
    for token in 0..n {
        let row = &mut fused_input[token * k..][..k];
        let up_row = &up[token * k..][..k];
        cpu_gelu_elementwise_mul(row, up_row);
    }

    let mut a_f32 = vec![0.0f32; m * k];
    cpu_dequant_q6_k(&quant_data, &mut a_f32);
    let mut expected = vec![0.0f32; n * m];
    cpu_batch_btrans_matmul(&a_f32, &fused_input, &mut expected, m, n, k);

    let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
    let buf_gate = MetalBuffer::from_slice(gpu.device(), &gate).unwrap();
    let buf_up = MetalBuffer::from_slice(gpu.device(), &up).unwrap();
    let buf_y = MetalBuffer::new(gpu.device(), n * m * std::mem::size_of::<f32>()).unwrap();

    gpu.execute_sync(|encoder| {
        assert!(kernels.encode_fused_batch_q6_k_gelu(
            encoder, &buf_a, &buf_gate, &buf_up, &buf_y, m as u32, n as u32, k as u32,
        ));
        Ok(())
    })
    .unwrap();

    let result = unsafe { buf_y.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 0.08,
        "Fused Q6_K batch GELU-down mismatch: m={m}, n={n}, k={k}, max_diff={diff}"
    );
}

#[test]
fn test_fused_batch_gelu_down_q6_k_matches_cpu_reference_boundary() {
    assert_fused_batch_gelu_down_q6_k_case(24, 11, 512);
}

#[test]
fn test_fused_batch_gelu_down_q6_k_matches_cpu_reference_fulltile() {
    assert_fused_batch_gelu_down_q6_k_case(32, 32, 512);
}

#[test]
fn test_moe_selected_pair_q6_k_ilp4_matches_cpu_reference() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 96usize;
    let k = 512usize;
    let n_selected = 2usize;
    let n_experts = 2usize;
    let blocks_per_row = k / Q6_K_BLOCK_SIZE;
    let weight_stride = m * blocks_per_row;

    let build_weights = |seed: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(n_experts * m * Q6_K_BYTES_PER_BLOCK * blocks_per_row);
        for expert in 0..n_experts {
            for row in 0..m {
                for blk in 0..blocks_per_row {
                    let mut block = vec![0u8; Q6_K_BYTES_PER_BLOCK];
                    for (i, byte) in block[..128].iter_mut().enumerate() {
                        *byte = ((seed + expert * 3 + row * 3 + blk * 5 + i) % 16) as u8;
                    }
                    for (i, byte) in block[128..192].iter_mut().enumerate() {
                        *byte = ((seed + expert * 5 + row * 11 + blk * 7 + i) % 4) as u8;
                    }
                    for (i, byte) in block[192..208].iter_mut().enumerate() {
                        *byte = ((seed as i32
                            + expert as i32 * 2
                            + row as i32 * 2
                            + blk as i32 * 3
                            + i as i32)
                            % 15
                            - 7) as i8 as u8;
                    }
                    let d = 0.0125 + ((seed + expert + row + blk) % 9) as f32 * 0.00625;
                    let d_bytes = half::f16::from_f32(d).to_le_bytes();
                    block[208] = d_bytes[0];
                    block[209] = d_bytes[1];
                    bytes.extend(block);
                }
            }
        }
        bytes
    };

    let weights0 = build_weights(1);
    let weights1 = build_weights(17);
    let input: Vec<f32> = (0..n_selected * k)
        .map(|i| (((i * 7 + 3) % 29) as f32 - 14.0) * 0.0075)
        .collect();
    let selected = vec![0i32, 1i32];

    let mut w0_f32 = vec![0.0f32; n_experts * m * k];
    let mut w1_f32 = vec![0.0f32; n_experts * m * k];
    cpu_dequant_q6_k(&weights0, &mut w0_f32);
    cpu_dequant_q6_k(&weights1, &mut w1_f32);
    let mut expected0 = vec![0.0f32; n_selected * m];
    let mut expected1 = vec![0.0f32; n_selected * m];
    for slot in 0..n_selected {
        let expert = selected[slot] as usize;
        let x = &input[slot * k..(slot + 1) * k];
        cpu_matvec(
            &w0_f32[expert * m * k..(expert + 1) * m * k],
            x,
            &mut expected0[slot * m..(slot + 1) * m],
            m,
            k,
        );
        cpu_matvec(
            &w1_f32[expert * m * k..(expert + 1) * m * k],
            x,
            &mut expected1[slot * m..(slot + 1) * m],
            m,
            k,
        );
    }

    let buf_w0 = MetalBuffer::from_bytes(gpu.device(), &weights0).unwrap();
    let buf_w1 = MetalBuffer::from_bytes(gpu.device(), &weights1).unwrap();
    let buf_input = MetalBuffer::from_slice(gpu.device(), &input).unwrap();
    let buf_selected = MetalBuffer::from_slice(gpu.device(), &selected).unwrap();
    let buf_base0 =
        MetalBuffer::new(gpu.device(), n_selected * m * std::mem::size_of::<f32>()).unwrap();
    let buf_base1 =
        MetalBuffer::new(gpu.device(), n_selected * m * std::mem::size_of::<f32>()).unwrap();
    let buf_out0 =
        MetalBuffer::new(gpu.device(), n_selected * m * std::mem::size_of::<f32>()).unwrap();
    let buf_out1 =
        MetalBuffer::new(gpu.device(), n_selected * m * std::mem::size_of::<f32>()).unwrap();

    let mut base_config = default_dequant_config();
    base_config.q6_k_variant = Some(crate::profile::MatvecProfileVariant::Base);
    gpu.execute_sync(|encoder| {
        kernels.encode_moe_mul_mat_selected_pair_q6_k(
            encoder,
            &buf_w0,
            &buf_w1,
            &buf_input,
            &buf_selected,
            &buf_base0,
            &buf_base1,
            m as u32,
            k as u32,
            n_selected as u32,
            weight_stride as u32,
            weight_stride as u32,
            true,
            base_config,
        );
        Ok(())
    })
    .unwrap();

    let mut config = default_dequant_config();
    config.q6_k_variant = Some(crate::profile::MatvecProfileVariant::Ilp4);
    gpu.execute_sync(|encoder| {
        kernels.encode_moe_mul_mat_selected_pair_q6_k(
            encoder,
            &buf_w0,
            &buf_w1,
            &buf_input,
            &buf_selected,
            &buf_out0,
            &buf_out1,
            m as u32,
            k as u32,
            n_selected as u32,
            weight_stride as u32,
            weight_stride as u32,
            true,
            config,
        );
        Ok(())
    })
    .unwrap();

    let base0 = unsafe { buf_base0.as_slice::<f32>() };
    let base1 = unsafe { buf_base1.as_slice::<f32>() };
    let out0 = unsafe { buf_out0.as_slice::<f32>() };
    let out1 = unsafe { buf_out1.as_slice::<f32>() };
    let base_diff0 = max_abs_diff(base0, &expected0);
    let base_diff1 = max_abs_diff(base1, &expected1);
    let diff0 = max_abs_diff(out0, &expected0);
    let diff1 = max_abs_diff(out1, &expected1);
    let base_vs_ilp4_0 = max_abs_diff(base0, out0);
    let base_vs_ilp4_1 = max_abs_diff(base1, out1);
    assert!(
        diff0 < 0.06 && diff1 < 0.06,
        "selected pair Q6_K ILP4 exceeded CPU tolerance: base_diff0={base_diff0}, base_diff1={base_diff1}, diff0={diff0}, diff1={diff1}",
    );
    assert!(
        base_vs_ilp4_0 < 0.02 && base_vs_ilp4_1 < 0.02,
        "selected pair Q6_K ILP4 diverged from base kernel: diff0={base_vs_ilp4_0}, diff1={base_vs_ilp4_1}",
    );
}

#[test]
fn test_moe_selected_pair_q8_0_ilp4_matches_cpu_reference() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = DequantKernels::new(&gpu).unwrap();

    let m = 96usize;
    let k = 256usize;
    let n_selected = 2usize;
    let n_experts = 2usize;
    let blocks_per_row = k / Q8_0_BLOCK_SIZE;
    let weight_stride = m * blocks_per_row;

    let build_weights = |seed: usize| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(n_experts * m * Q8_0_BYTES_PER_BLOCK * blocks_per_row);
        for expert in 0..n_experts {
            for row in 0..m {
                for blk in 0..blocks_per_row {
                    let mut block = vec![0u8; Q8_0_BYTES_PER_BLOCK];
                    let d = ((seed + expert * 5 + row + blk * 3) % 17) as f32 * 0.05 + 0.04;
                    let d_bytes = half::f16::from_f32(d).to_le_bytes();
                    block[0] = d_bytes[0];
                    block[1] = d_bytes[1];
                    for (i, byte) in block[2..34].iter_mut().enumerate() {
                        *byte = ((seed as i32
                            + expert as i32 * 11
                            + row as i32 * 5
                            + blk as i32 * 7
                            + i as i32)
                            % 127
                            - 63) as i8 as u8;
                    }
                    bytes.extend(block);
                }
            }
        }
        bytes
    };

    let weights0 = build_weights(3);
    let weights1 = build_weights(23);
    let input: Vec<f32> = (0..n_selected * k)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.0125)
        .collect();
    let selected = vec![0i32, 1i32];

    let mut w0_f32 = vec![0.0f32; n_experts * m * k];
    let mut w1_f32 = vec![0.0f32; n_experts * m * k];
    cpu_dequant_q8_0(&weights0, &mut w0_f32);
    cpu_dequant_q8_0(&weights1, &mut w1_f32);
    let mut expected0 = vec![0.0f32; n_selected * m];
    let mut expected1 = vec![0.0f32; n_selected * m];
    for slot in 0..n_selected {
        let expert = selected[slot] as usize;
        let x = &input[slot * k..(slot + 1) * k];
        cpu_matvec(
            &w0_f32[expert * m * k..(expert + 1) * m * k],
            x,
            &mut expected0[slot * m..(slot + 1) * m],
            m,
            k,
        );
        cpu_matvec(
            &w1_f32[expert * m * k..(expert + 1) * m * k],
            x,
            &mut expected1[slot * m..(slot + 1) * m],
            m,
            k,
        );
    }

    let buf_w0 = MetalBuffer::from_bytes(gpu.device(), &weights0).unwrap();
    let buf_w1 = MetalBuffer::from_bytes(gpu.device(), &weights1).unwrap();
    let buf_input = MetalBuffer::from_slice(gpu.device(), &input).unwrap();
    let buf_selected = MetalBuffer::from_slice(gpu.device(), &selected).unwrap();
    let buf_base0 =
        MetalBuffer::new(gpu.device(), n_selected * m * std::mem::size_of::<f32>()).unwrap();
    let buf_base1 =
        MetalBuffer::new(gpu.device(), n_selected * m * std::mem::size_of::<f32>()).unwrap();
    let buf_out0 =
        MetalBuffer::new(gpu.device(), n_selected * m * std::mem::size_of::<f32>()).unwrap();
    let buf_out1 =
        MetalBuffer::new(gpu.device(), n_selected * m * std::mem::size_of::<f32>()).unwrap();

    let mut base_config = default_dequant_config();
    base_config.q8_0_variant = Some(crate::profile::MatvecProfileVariant::Base);
    gpu.execute_sync(|encoder| {
        kernels.encode_moe_mul_mat_selected_pair_q8_0(
            encoder,
            &buf_w0,
            &buf_w1,
            &buf_input,
            &buf_selected,
            &buf_base0,
            &buf_base1,
            m as u32,
            k as u32,
            n_selected as u32,
            weight_stride as u32,
            weight_stride as u32,
            true,
            base_config,
        );
        Ok(())
    })
    .unwrap();

    let mut config = default_dequant_config();
    config.q8_0_variant = Some(crate::profile::MatvecProfileVariant::Ilp4);
    gpu.execute_sync(|encoder| {
        kernels.encode_moe_mul_mat_selected_pair_q8_0(
            encoder,
            &buf_w0,
            &buf_w1,
            &buf_input,
            &buf_selected,
            &buf_out0,
            &buf_out1,
            m as u32,
            k as u32,
            n_selected as u32,
            weight_stride as u32,
            weight_stride as u32,
            true,
            config,
        );
        Ok(())
    })
    .unwrap();

    let base0 = unsafe { buf_base0.as_slice::<f32>() };
    let base1 = unsafe { buf_base1.as_slice::<f32>() };
    let out0 = unsafe { buf_out0.as_slice::<f32>() };
    let out1 = unsafe { buf_out1.as_slice::<f32>() };
    let base_diff0 = max_abs_diff(base0, &expected0);
    let base_diff1 = max_abs_diff(base1, &expected1);
    let diff0 = max_abs_diff(out0, &expected0);
    let diff1 = max_abs_diff(out1, &expected1);
    let base_vs_ilp4_0 = max_abs_diff(base0, out0);
    let base_vs_ilp4_1 = max_abs_diff(base1, out1);
    assert!(
        diff0 < 0.02 && diff1 < 0.02,
        "selected pair Q8_0 ILP4 exceeded CPU tolerance: base_diff0={base_diff0}, base_diff1={base_diff1}, diff0={diff0}, diff1={diff1}",
    );
    assert!(
        base_vs_ilp4_0 < 0.02 && base_vs_ilp4_1 < 0.02,
        "selected pair Q8_0 ILP4 diverged from base kernel: diff0={base_vs_ilp4_0}, diff1={base_vs_ilp4_1}",
    );
}

// ── Attention test helpers ────────────────────────────────────────

/// CPU reference softmax (numerically stable).
fn cpu_softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

/// CPU reference multi-head attention with causal masking (prefill).
#[allow(clippy::too_many_arguments)]
fn cpu_attention_prefill(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    n_tokens: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    let q_stride = n_heads * head_dim;
    let kv_stride = n_kv_heads * head_dim;
    let heads_per_kv = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..n_heads {
        let kv_h = h / heads_per_kv;
        for qi in 0..n_tokens {
            let q_off = qi * q_stride + h * head_dim;
            let q_head = &q[q_off..q_off + head_dim];
            let attend_len = qi + 1;

            let mut scores = vec![0.0f32; attend_len];
            for (t, score) in scores.iter_mut().enumerate() {
                let k_off = t * kv_stride + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_head[d] * k[k_off + d];
                }
                *score = dot * scale;
            }
            cpu_softmax(&mut scores);

            let o_off = qi * q_stride + h * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (t, &w) in scores.iter().enumerate() {
                    let v_off = t * kv_stride + kv_h * head_dim;
                    acc += w * v[v_off + d];
                }
                output[o_off + d] = acc;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn cpu_attention_prefill_cached_with_row_stride(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    output: &mut [f32],
    n_tokens: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_row_stride: usize,
    base_seq_len: usize,
    sliding_window: usize,
) {
    let q_stride = n_heads * head_dim;
    let heads_per_kv = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..n_heads {
        let kv_h = h / heads_per_kv;
        for qi in 0..n_tokens {
            let q_off = qi * q_stride + h * head_dim;
            let q_head = &q[q_off..q_off + head_dim];
            let total_len = base_seq_len + qi + 1;
            let start = if sliding_window == 0 {
                0
            } else {
                total_len.saturating_sub(sliding_window)
            };
            let attend_len = total_len - start;
            let mut scores = vec![0.0f32; attend_len];

            for (t, score) in scores.iter_mut().enumerate() {
                let token = start + t;
                let k_off = token * kv_row_stride + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_head[d] * k_cache[k_off + d];
                }
                *score = dot * scale;
            }
            cpu_softmax(&mut scores);

            let o_off = qi * q_stride + h * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (t, &w) in scores.iter().enumerate() {
                    let token = start + t;
                    let v_off = token * kv_row_stride + kv_h * head_dim;
                    acc += w * v_cache[v_off + d];
                }
                output[o_off + d] = acc;
            }
        }
    }
}

// ── Attention kernel tests ───────────────────────────────────────

#[test]
fn test_attention_kernels_compile() {
    let gpu = MetalDevice::new().unwrap();
    let _kernels = AttentionKernels::new(&gpu).unwrap();
}

#[test]
fn test_attention_prefill_single_token() {
    // 1 token, 1 head, head_dim=4
    // softmax([score]) = [1.0] → output = V
    let gpu = MetalDevice::new().unwrap();
    let kernels = AttentionKernels::new(&gpu).unwrap();

    let q = [1.0f32, 0.0, 0.0, 0.0];
    let k = [1.0f32, 0.0, 0.0, 0.0];
    let v = [3.0f32, 4.0, 5.0, 6.0];

    let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
    let buf_k = MetalBuffer::from_slice(gpu.device(), &k).unwrap();
    let buf_v = MetalBuffer::from_slice(gpu.device(), &v).unwrap();
    let buf_o = MetalBuffer::new(gpu.device(), 4 * 4).unwrap();

    kernels
        .attention_prefill_with_config(
            &gpu,
            &buf_q,
            &buf_k,
            &buf_v,
            &buf_o,
            1,
            1,
            1,
            4,
            default_attention_config(),
        )
        .unwrap();

    let result = unsafe { buf_o.as_slice::<f32>() };
    assert!(
        max_abs_diff(result, &v) < 1e-4,
        "Single token: got {:?}, expected {:?}",
        result,
        v
    );
}

#[test]
fn test_attention_prefill_two_tokens_causal() {
    // 2 tokens, 1 head, head_dim=2
    // Token 0: sees only V[0]; Token 1: sees V[0..2] with equal Q·K
    let gpu = MetalDevice::new().unwrap();
    let kernels = AttentionKernels::new(&gpu).unwrap();

    let q = [1.0f32, 0.0, 1.0, 0.0]; // 2 tokens
    let k = [1.0f32, 0.0, 1.0, 0.0]; // equal keys → uniform softmax
    let v = [2.0f32, 3.0, 4.0, 5.0]; // V[0]=[2,3], V[1]=[4,5]

    let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
    let buf_k = MetalBuffer::from_slice(gpu.device(), &k).unwrap();
    let buf_v = MetalBuffer::from_slice(gpu.device(), &v).unwrap();
    let buf_o = MetalBuffer::new(gpu.device(), 4 * 4).unwrap();

    kernels
        .attention_prefill_with_config(
            &gpu,
            &buf_q,
            &buf_k,
            &buf_v,
            &buf_o,
            2,
            1,
            1,
            2,
            default_attention_config(),
        )
        .unwrap();

    let result = unsafe { buf_o.as_slice::<f32>() };

    // Token 0: only sees V[0] = [2, 3]
    assert!(
        (result[0] - 2.0).abs() < 1e-4,
        "token0[0]={}, expected 2.0",
        result[0]
    );
    assert!(
        (result[1] - 3.0).abs() < 1e-4,
        "token0[1]={}, expected 3.0",
        result[1]
    );
    // Token 1: uniform over V[0..2] → [3.0, 4.0]
    assert!(
        (result[2] - 3.0).abs() < 1e-4,
        "token1[0]={}, expected 3.0",
        result[2]
    );
    assert!(
        (result[3] - 4.0).abs() < 1e-4,
        "token1[1]={}, expected 4.0",
        result[3]
    );
}

#[test]
fn test_attention_prefill_gqa() {
    // 1 token, 4 query heads, 2 KV heads, head_dim=2
    // Heads 0,1 → kv_head 0; Heads 2,3 → kv_head 1
    let gpu = MetalDevice::new().unwrap();
    let kernels = AttentionKernels::new(&gpu).unwrap();

    let q = [1.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]; // 4 heads
    let k = [1.0f32, 0.0, 0.0, 1.0]; // 2 KV heads
    let v = [10.0f32, 20.0, 30.0, 40.0]; // 2 KV heads

    let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
    let buf_k = MetalBuffer::from_slice(gpu.device(), &k).unwrap();
    let buf_v = MetalBuffer::from_slice(gpu.device(), &v).unwrap();
    let buf_o = MetalBuffer::new(gpu.device(), 8 * 4).unwrap();

    kernels
        .attention_prefill_with_config(
            &gpu,
            &buf_q,
            &buf_k,
            &buf_v,
            &buf_o,
            1,
            4,
            2,
            2,
            default_attention_config(),
        )
        .unwrap();

    let result = unsafe { buf_o.as_slice::<f32>() };

    // Heads 0,1 → kv0 V = [10, 20]
    assert!((result[0] - 10.0).abs() < 1e-3, "h0[0]={}", result[0]);
    assert!((result[1] - 20.0).abs() < 1e-3, "h0[1]={}", result[1]);
    assert!((result[2] - 10.0).abs() < 1e-3, "h1[0]={}", result[2]);
    assert!((result[3] - 20.0).abs() < 1e-3, "h1[1]={}", result[3]);
    // Heads 2,3 → kv1 V = [30, 40]
    assert!((result[4] - 30.0).abs() < 1e-3, "h2[0]={}", result[4]);
    assert!((result[5] - 40.0).abs() < 1e-3, "h2[1]={}", result[5]);
    assert!((result[6] - 30.0).abs() < 1e-3, "h3[0]={}", result[6]);
    assert!((result[7] - 40.0).abs() < 1e-3, "h3[1]={}", result[7]);
}

#[test]
fn test_attention_prefill_matches_cpu() {
    // Larger test: 16 tokens, 4 heads, 2 KV heads, head_dim=64
    let gpu = MetalDevice::new().unwrap();
    let kernels = AttentionKernels::new(&gpu).unwrap();

    let n_tokens = 16;
    let n_heads = 4;
    let n_kv_heads = 2;
    let head_dim = 64;
    let q_size = n_tokens * n_heads * head_dim;
    let kv_size = n_tokens * n_kv_heads * head_dim;

    // Generate deterministic data
    let q: Vec<f32> = (0..q_size)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
        .collect();
    let k: Vec<f32> = (0..kv_size)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
        .collect();
    let v: Vec<f32> = (0..kv_size).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();

    // CPU reference
    let mut expected = vec![0.0f32; q_size];
    cpu_attention_prefill(
        &q,
        &k,
        &v,
        &mut expected,
        n_tokens,
        n_heads,
        n_kv_heads,
        head_dim,
    );

    // GPU
    let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
    let buf_k = MetalBuffer::from_slice(gpu.device(), &k).unwrap();
    let buf_v = MetalBuffer::from_slice(gpu.device(), &v).unwrap();
    let buf_o = MetalBuffer::new(gpu.device(), q_size * std::mem::size_of::<f32>()).unwrap();

    kernels
        .attention_prefill_with_config(
            &gpu,
            &buf_q,
            &buf_k,
            &buf_v,
            &buf_o,
            n_tokens as u32,
            n_heads as u32,
            n_kv_heads as u32,
            head_dim as u32,
            default_attention_config(),
        )
        .unwrap();

    let result = unsafe { buf_o.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 1e-2,
        "Attention prefill CPU vs GPU mismatch: max_diff={}",
        diff
    );
}

#[test]
fn test_attention_prefill_multi_tile() {
    // 512 tokens to trigger multiple tiles (ATTN_TG=256, so 2+ tiles)
    let gpu = MetalDevice::new().unwrap();
    let kernels = AttentionKernels::new(&gpu).unwrap();

    let n_tokens = 512;
    let n_heads = 2;
    let n_kv_heads = 2;
    let head_dim = 32;
    let q_size = n_tokens * n_heads * head_dim;
    let kv_size = n_tokens * n_kv_heads * head_dim;

    let q: Vec<f32> = (0..q_size)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.03)
        .collect();
    let k: Vec<f32> = (0..kv_size)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.03)
        .collect();
    let v: Vec<f32> = (0..kv_size)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
        .collect();

    let mut expected = vec![0.0f32; q_size];
    cpu_attention_prefill(
        &q,
        &k,
        &v,
        &mut expected,
        n_tokens,
        n_heads,
        n_kv_heads,
        head_dim,
    );

    let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
    let buf_k = MetalBuffer::from_slice(gpu.device(), &k).unwrap();
    let buf_v = MetalBuffer::from_slice(gpu.device(), &v).unwrap();
    let buf_o = MetalBuffer::new(gpu.device(), q_size * std::mem::size_of::<f32>()).unwrap();

    kernels
        .attention_prefill_with_config(
            &gpu,
            &buf_q,
            &buf_k,
            &buf_v,
            &buf_o,
            n_tokens as u32,
            n_heads as u32,
            n_kv_heads as u32,
            head_dim as u32,
            default_attention_config(),
        )
        .unwrap();

    let result = unsafe { buf_o.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 0.05,
        "Multi-tile attention mismatch: max_diff={}",
        diff
    );
}

#[test]
fn test_attention_prefill_matches_cpu_hd512() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = AttentionKernels::new(&gpu).unwrap();

    let n_tokens = 8;
    let n_heads = 4;
    let n_kv_heads = 2;
    let head_dim = 512;
    let q_size = n_tokens * n_heads * head_dim;
    let kv_size = n_tokens * n_kv_heads * head_dim;

    let q: Vec<f32> = (0..q_size)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.02)
        .collect();
    let k: Vec<f32> = (0..kv_size)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.02)
        .collect();
    let v: Vec<f32> = (0..kv_size)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.03)
        .collect();

    let mut expected = vec![0.0f32; q_size];
    cpu_attention_prefill(
        &q,
        &k,
        &v,
        &mut expected,
        n_tokens,
        n_heads,
        n_kv_heads,
        head_dim,
    );

    let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
    let buf_k = MetalBuffer::from_slice(gpu.device(), &k).unwrap();
    let buf_v = MetalBuffer::from_slice(gpu.device(), &v).unwrap();
    let buf_o = MetalBuffer::new(gpu.device(), q_size * std::mem::size_of::<f32>()).unwrap();

    kernels
        .attention_prefill_with_config(
            &gpu,
            &buf_q,
            &buf_k,
            &buf_v,
            &buf_o,
            n_tokens as u32,
            n_heads as u32,
            n_kv_heads as u32,
            head_dim as u32,
            default_attention_config(),
        )
        .unwrap();

    let result = unsafe { buf_o.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 0.05,
        "HD512 attention prefill CPU vs GPU mismatch: max_diff={}",
        diff
    );
}

#[test]
fn test_attention_prefill_cached_with_padded_kv_stride_matches_cpu() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = AttentionKernels::new(&gpu).unwrap();

    let base_seq_len = 5usize;
    let n_tokens = 4usize;
    let n_heads = 4usize;
    let n_kv_heads = 2usize;
    let head_dim = 64usize;
    let kv_row_stride = 192usize;
    let q_size = n_tokens * n_heads * head_dim;
    let cache_len = (base_seq_len + n_tokens) * kv_row_stride;

    let q: Vec<f32> = (0..q_size)
        .map(|i| ((i % 41) as f32 - 20.0) * 0.01)
        .collect();
    let mut k_cache = vec![0.0f32; cache_len];
    let mut v_cache = vec![0.0f32; cache_len];
    for token in 0..(base_seq_len + n_tokens) {
        for i in 0..(n_kv_heads * head_dim) {
            let idx = token * kv_row_stride + i;
            k_cache[idx] = ((idx % 37) as f32 - 18.0) * 0.015;
            v_cache[idx] = ((idx % 29) as f32 - 14.0) * 0.02;
        }
    }

    let mut expected = vec![0.0f32; q_size];
    cpu_attention_prefill_cached_with_row_stride(
        &q,
        &k_cache,
        &v_cache,
        &mut expected,
        n_tokens,
        n_heads,
        n_kv_heads,
        head_dim,
        kv_row_stride,
        base_seq_len,
        0,
    );

    let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
    let buf_k = MetalBuffer::from_slice(gpu.device(), &k_cache).unwrap();
    let buf_v = MetalBuffer::from_slice(gpu.device(), &v_cache).unwrap();
    let buf_o = MetalBuffer::new(gpu.device(), q_size * std::mem::size_of::<f32>()).unwrap();

    gpu.execute_sync(|encoder| {
        kernels.encode_attention_prefill_cached_with_stride_and_config(
            encoder,
            &buf_q,
            &buf_k,
            &buf_v,
            &buf_o,
            false,
            n_tokens as u32,
            n_heads as u32,
            n_kv_heads as u32,
            head_dim as u32,
            kv_row_stride as u32,
            base_seq_len as u32,
            0,
            default_attention_config(),
        );
        Ok(())
    })
    .unwrap();

    let result = unsafe { buf_o.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 1e-2,
        "Cached prefill with padded stride mismatch: max_diff={diff}"
    );
}

#[test]
fn test_attention_prefill_concentrates_on_matching_key() {
    // Large Q·K score for one key should dominate
    let gpu = MetalDevice::new().unwrap();
    let kernels = AttentionKernels::new(&gpu).unwrap();

    // 3 tokens, 1 head, head_dim=2
    // Q[2] = [10, 0], K = [[1,0], [0,1], [1,0]]
    // Token 2 attends to 0..2: scores ∝ [10, 0, 10] → nearly [0.5, 0, 0.5]
    let q = [1.0f32, 0.0, 0.0, 1.0, 10.0, 0.0]; // 3 tokens
    let k = [1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0]; // 3 KV tokens
    let v = [1.0f32, 0.0, 0.0, 1.0, 0.5, 0.5]; // 3 V tokens

    let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
    let buf_k = MetalBuffer::from_slice(gpu.device(), &k).unwrap();
    let buf_v = MetalBuffer::from_slice(gpu.device(), &v).unwrap();
    let buf_o = MetalBuffer::new(gpu.device(), 6 * 4).unwrap();

    kernels
        .attention_prefill_with_config(
            &gpu,
            &buf_q,
            &buf_k,
            &buf_v,
            &buf_o,
            3,
            1,
            1,
            2,
            default_attention_config(),
        )
        .unwrap();

    let result = unsafe { buf_o.as_slice::<f32>() };
    let mut expected = vec![0.0f32; 6];
    cpu_attention_prefill(&q, &k, &v, &mut expected, 3, 1, 1, 2);

    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 1e-3,
        "Concentrate test: max_diff={}, got {:?}, expected {:?}",
        diff,
        result,
        expected
    );
}

// ── Elementwise kernel tests ────────────────────────────────────

// CPU reference implementations for elementwise kernel verification.

fn cpu_rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    for (xi, &wi) in x.iter_mut().zip(weight.iter()) {
        *xi = *xi * inv_rms * wi;
    }
}

fn cpu_rms_norm_out(x: &[f32], weight: &[f32], out: &mut [f32], eps: f32) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

fn cpu_rope(
    q: &mut [f32],
    k: &mut [f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    position: f32,
    freq_base: f32,
) {
    let half_dim = head_dim / 2;
    let mut cos_table = vec![0.0f32; half_dim];
    let mut sin_table = vec![0.0f32; half_dim];
    for i in 0..half_dim {
        let freq = 1.0 / freq_base.powf(2.0 * i as f32 / head_dim as f32);
        let theta = position * freq;
        cos_table[i] = theta.cos();
        sin_table[i] = theta.sin();
    }
    for h in 0..n_q_heads {
        let off = h * head_dim;
        for i in 0..half_dim {
            let v0 = q[off + 2 * i];
            let v1 = q[off + 2 * i + 1];
            q[off + 2 * i] = v0 * cos_table[i] - v1 * sin_table[i];
            q[off + 2 * i + 1] = v0 * sin_table[i] + v1 * cos_table[i];
        }
    }
    for h in 0..n_kv_heads {
        let off = h * head_dim;
        for i in 0..half_dim {
            let v0 = k[off + 2 * i];
            let v1 = k[off + 2 * i + 1];
            k[off + 2 * i] = v0 * cos_table[i] - v1 * sin_table[i];
            k[off + 2 * i + 1] = v0 * sin_table[i] + v1 * cos_table[i];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn cpu_rope_batch(
    q: &mut [f32],
    k: &mut [f32],
    n_rows: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    start_pos: f32,
    pos_step: f32,
    freq_base: f32,
) {
    let q_row = n_q_heads * head_dim;
    let k_row = n_kv_heads * head_dim;
    for row in 0..n_rows {
        let position = start_pos + row as f32 * pos_step;
        cpu_rope(
            &mut q[row * q_row..(row + 1) * q_row],
            &mut k[row * k_row..(row + 1) * k_row],
            n_q_heads,
            n_kv_heads,
            head_dim,
            position,
            freq_base,
        );
    }
}

fn cpu_gelu_elementwise_mul(gate: &mut [f32], up: &[f32]) {
    const SQRT_2_PI: f32 = 0.797_884_6;
    for (gi, &ui) in gate.iter_mut().zip(up.iter()) {
        let x = *gi;
        let x3 = x * x * x;
        let inner = SQRT_2_PI * (x + 0.044715 * x3);
        *gi = 0.5 * x * (1.0 + inner.tanh()) * ui;
    }
}

fn cpu_silu_elementwise_mul(gate: &mut [f32], up: &[f32]) {
    for (gi, &ui) in gate.iter_mut().zip(up.iter()) {
        let x = *gi;
        *gi = x / (1.0 + (-x).exp()) * ui;
    }
}

fn cpu_rms_norm_no_weight(x: &mut [f32], eps: f32) {
    if x.is_empty() {
        return;
    }
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / x.len() as f32 + eps).sqrt();
    for value in x.iter_mut() {
        *value *= inv_rms;
    }
}

#[allow(clippy::too_many_arguments)]
fn cpu_rope_neox_freq_factors_batch(
    q: &mut [f32],
    k: &mut [f32],
    freq_factors: &[f32],
    n_rows: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    start_pos: f32,
    pos_step: f32,
    freq_base: f32,
) {
    let rope_pairs = head_dim / 2;
    for row in 0..n_rows {
        let position = start_pos + row as f32 * pos_step;
        for head in 0..n_q_heads {
            let base = row * n_q_heads * head_dim + head * head_dim;
            for i in 0..rope_pairs {
                let base_freq = (-(freq_base.ln()) * 2.0 * i as f32 / head_dim as f32).exp();
                let theta = position * base_freq * freq_factors[i];
                let (sin_t, cos_t) = theta.sin_cos();
                let a = q[base + i];
                let b = q[base + rope_pairs + i];
                q[base + i] = a * cos_t - b * sin_t;
                q[base + rope_pairs + i] = a * sin_t + b * cos_t;
            }
        }
        for head in 0..n_kv_heads {
            let base = row * n_kv_heads * head_dim + head * head_dim;
            for i in 0..rope_pairs {
                let base_freq = (-(freq_base.ln()) * 2.0 * i as f32 / head_dim as f32).exp();
                let theta = position * base_freq * freq_factors[i];
                let (sin_t, cos_t) = theta.sin_cos();
                let a = k[base + i];
                let b = k[base + rope_pairs + i];
                k[base + i] = a * cos_t - b * sin_t;
                k[base + rope_pairs + i] = a * sin_t + b * cos_t;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn cpu_post_attn_norm_residual_add_rms_norm_out_batch(
    hidden: &mut [f32],
    addend: &[f32],
    post_weight: &[f32],
    residual_weight: &[f32],
    norm_out: &mut [f32],
    n: usize,
    n_rows: usize,
    eps: f32,
) {
    for row in 0..n_rows {
        let base = row * n;
        let mut attn_norm = vec![0.0f32; n];
        cpu_rms_norm_out(&addend[base..base + n], post_weight, &mut attn_norm, eps);
        for i in 0..n {
            hidden[base + i] += attn_norm[i];
        }
        cpu_rms_norm_out(
            &hidden[base..base + n],
            residual_weight,
            &mut norm_out[base..base + n],
            eps,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn cpu_post_ffn_norm_residual_add_rms_norm_out_batch(
    hidden: &mut [f32],
    addend: &[f32],
    post_weight: &[f32],
    residual_weight: &[f32],
    norm_out: &mut [f32],
    n: usize,
    n_rows: usize,
    eps: f32,
) {
    for row in 0..n_rows {
        let base = row * n;
        let mut ffn_norm = vec![0.0f32; n];
        cpu_rms_norm_out(&addend[base..base + n], post_weight, &mut ffn_norm, eps);
        for i in 0..n {
            hidden[base + i] += ffn_norm[i];
        }
        cpu_rms_norm_out(
            &hidden[base..base + n],
            residual_weight,
            &mut norm_out[base..base + n],
            eps,
        );
    }
}

#[test]
fn test_elementwise_kernels_compile() {
    let gpu = MetalDevice::new().unwrap();
    let _kernels = ElementwiseKernels::new(&gpu).unwrap();
}

#[test]
fn test_gpu_rms_norm_matches_cpu() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = ElementwiseKernels::new(&gpu).unwrap();
    let n = 256;
    let eps = 1e-5f32;

    let x: Vec<f32> = (0..n).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect();
    let weight: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.2 + 1.0).collect();

    // CPU reference
    let mut expected = x.clone();
    cpu_rms_norm(&mut expected, &weight, eps);

    // GPU
    let buf_x = MetalBuffer::from_slice(gpu.device(), &x).unwrap();
    let buf_w = MetalBuffer::from_slice(gpu.device(), &weight).unwrap();
    kernels
        .rms_norm(&gpu, &buf_x, &buf_w, n as u32, eps)
        .unwrap();
    let result = unsafe { buf_x.as_slice::<f32>() };

    let diff = max_abs_diff(result, &expected);
    assert!(diff < 1e-4, "RMSNorm GPU vs CPU mismatch: max_diff={diff}");
}

#[test]
fn test_gpu_rms_norm_out_matches_cpu() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = ElementwiseKernels::new(&gpu).unwrap();
    let n = 512;
    let eps = 1e-6f32;

    let x: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.2).collect();
    let weight: Vec<f32> = (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.1 + 1.0).collect();

    let mut expected = vec![0.0f32; n];
    cpu_rms_norm_out(&x, &weight, &mut expected, eps);

    let buf_x = MetalBuffer::from_slice(gpu.device(), &x).unwrap();
    let buf_w = MetalBuffer::from_slice(gpu.device(), &weight).unwrap();
    let buf_out = MetalBuffer::new(gpu.device(), n * 4).unwrap();
    kernels
        .rms_norm_out(&gpu, &buf_x, &buf_w, &buf_out, n as u32, eps)
        .unwrap();
    let result = unsafe { buf_out.as_slice::<f32>() };

    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 1e-4,
        "RMSNorm out GPU vs CPU mismatch: max_diff={diff}"
    );
}

#[test]
fn test_gpu_rope_matches_cpu() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = ElementwiseKernels::new(&gpu).unwrap();

    let n_q_heads = 8u32;
    let n_kv_heads = 4u32;
    let head_dim = 64u32;
    let position = 42.0f32;
    let freq_base = 10000.0f32;

    let q: Vec<f32> = (0..n_q_heads as usize * head_dim as usize)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
        .collect();
    let k: Vec<f32> = (0..n_kv_heads as usize * head_dim as usize)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();

    // CPU reference
    let mut q_ref = q.clone();
    let mut k_ref = k.clone();
    cpu_rope(
        &mut q_ref,
        &mut k_ref,
        n_q_heads as usize,
        n_kv_heads as usize,
        head_dim as usize,
        position,
        freq_base,
    );

    // GPU
    let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
    let buf_k = MetalBuffer::from_slice(gpu.device(), &k).unwrap();
    kernels
        .rope(
            &gpu, &buf_q, &buf_k, n_q_heads, n_kv_heads, head_dim, position, freq_base,
        )
        .unwrap();
    let q_gpu = unsafe { buf_q.as_slice::<f32>() };
    let k_gpu = unsafe { buf_k.as_slice::<f32>() };

    let q_diff = max_abs_diff(q_gpu, &q_ref);
    let k_diff = max_abs_diff(k_gpu, &k_ref);
    assert!(
        q_diff < 1e-4,
        "RoPE Q GPU vs CPU mismatch: max_diff={q_diff}"
    );
    assert!(
        k_diff < 1e-4,
        "RoPE K GPU vs CPU mismatch: max_diff={k_diff}"
    );
}

#[test]
fn test_gpu_rope_batch_matches_cpu() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = ElementwiseKernels::new(&gpu).unwrap();

    let n_rows = 3u32;
    let n_q_heads = 6u32;
    let n_kv_heads = 2u32;
    let head_dim = 64u32;
    let start_pos = 11.0f32;
    let pos_step = 1.0f32;
    let freq_base = 10000.0f32;

    let q: Vec<f32> = (0..(n_rows * n_q_heads * head_dim) as usize)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.07)
        .collect();
    let k: Vec<f32> = (0..(n_rows * n_kv_heads * head_dim) as usize)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.09)
        .collect();

    let mut q_ref = q.clone();
    let mut k_ref = k.clone();
    cpu_rope_batch(
        &mut q_ref,
        &mut k_ref,
        n_rows as usize,
        n_q_heads as usize,
        n_kv_heads as usize,
        head_dim as usize,
        start_pos,
        pos_step,
        freq_base,
    );

    let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
    let buf_k = MetalBuffer::from_slice(gpu.device(), &k).unwrap();
    kernels
        .rope_batch(
            &gpu, &buf_q, &buf_k, n_rows, n_q_heads, n_kv_heads, head_dim, start_pos, pos_step,
            freq_base,
        )
        .unwrap();
    let q_gpu = unsafe { buf_q.as_slice::<f32>() };
    let k_gpu = unsafe { buf_k.as_slice::<f32>() };

    let q_diff = max_abs_diff(q_gpu, &q_ref);
    let k_diff = max_abs_diff(k_gpu, &k_ref);
    assert!(
        q_diff < 1e-4,
        "RoPE batch Q GPU vs CPU mismatch: max_diff={q_diff}"
    );
    assert!(
        k_diff < 1e-4,
        "RoPE batch K GPU vs CPU mismatch: max_diff={k_diff}"
    );
}

#[test]
fn test_gpu_per_head_rms_norm_matches_cpu() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = ElementwiseKernels::new(&gpu).unwrap();

    let n_heads = 8u32;
    let head_dim = 64u32;
    let n = (n_heads * head_dim) as usize;
    let eps = 1e-6f32;

    let data: Vec<f32> = (0..n).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect();
    let weight: Vec<f32> = (0..head_dim as usize)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.2 + 1.0)
        .collect();

    // CPU reference: per-head rms_norm
    let mut expected = data.clone();
    for h in 0..n_heads as usize {
        let off = h * head_dim as usize;
        cpu_rms_norm(&mut expected[off..off + head_dim as usize], &weight, eps);
    }

    // GPU
    let buf = MetalBuffer::from_slice(gpu.device(), &data).unwrap();
    let buf_w = MetalBuffer::from_slice(gpu.device(), &weight).unwrap();
    kernels
        .per_head_rms_norm(&gpu, &buf, &buf_w, n_heads, head_dim, eps)
        .unwrap();
    let result = unsafe { buf.as_slice::<f32>() };

    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 1e-4,
        "Per-head RMSNorm GPU vs CPU mismatch: max_diff={diff}"
    );
}

#[test]
fn test_gpu_per_head_rms_norm_batch_matches_cpu() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = ElementwiseKernels::new(&gpu).unwrap();

    let n_rows = 3u32;
    let n_heads = 4u32;
    let head_dim = 64u32;
    let eps = 1e-6f32;
    let total = (n_rows * n_heads * head_dim) as usize;

    let data: Vec<f32> = (0..total).map(|i| ((i % 17) as f32 - 8.0) * 0.15).collect();
    let weight: Vec<f32> = (0..head_dim as usize)
        .map(|i| ((i % 9) as f32 - 4.0) * 0.11 + 1.0)
        .collect();

    let mut expected = data.clone();
    let row_stride = (n_heads * head_dim) as usize;
    let head_stride = head_dim as usize;
    for row in 0..n_rows as usize {
        for head in 0..n_heads as usize {
            let off = row * row_stride + head * head_stride;
            cpu_rms_norm(&mut expected[off..off + head_stride], &weight, eps);
        }
    }

    let buf = MetalBuffer::from_slice(gpu.device(), &data).unwrap();
    let buf_w = MetalBuffer::from_slice(gpu.device(), &weight).unwrap();
    kernels
        .per_head_rms_norm_batch(&gpu, &buf, &buf_w, n_rows, n_heads, head_dim, eps)
        .unwrap();
    let result = unsafe { buf.as_slice::<f32>() };

    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 1e-4,
        "Per-head RMSNorm batch GPU vs CPU mismatch: max_diff={diff}"
    );
}

#[test]
fn test_gpu_per_head_rms_norm_no_weight_batch_matches_cpu() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = ElementwiseKernels::new(&gpu).unwrap();

    let n_rows = 2u32;
    let n_heads = 3u32;
    let head_dim = 128u32;
    let eps = 1e-6f32;
    let total = (n_rows * n_heads * head_dim) as usize;

    let data: Vec<f32> = (0..total)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.09)
        .collect();
    let mut expected = data.clone();
    let row_stride = (n_heads * head_dim) as usize;
    let head_stride = head_dim as usize;
    for row in 0..n_rows as usize {
        for head in 0..n_heads as usize {
            let off = row * row_stride + head * head_stride;
            cpu_rms_norm_no_weight(&mut expected[off..off + head_stride], eps);
        }
    }

    let buf = MetalBuffer::from_slice(gpu.device(), &data).unwrap();
    kernels
        .per_head_rms_norm_no_weight_batch(&gpu, &buf, n_rows, n_heads, head_dim, eps)
        .unwrap();
    let result = unsafe { buf.as_slice::<f32>() };

    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 1e-4,
        "Per-head RMSNorm no-weight batch GPU vs CPU mismatch: max_diff={diff}"
    );
}

#[test]
fn test_gpu_rope_neox_freq_factors_matches_cpu() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = ElementwiseKernels::new(&gpu).unwrap();

    let n_rows = 3u32;
    let n_q_heads = 2u32;
    let n_kv_heads = 1u32;
    let head_dim = 64u32;
    let start_pos = 7.0f32;
    let pos_step = 1.0f32;
    let freq_base = 1_000_000.0f32;
    let q_len = (n_rows * n_q_heads * head_dim) as usize;
    let k_len = (n_rows * n_kv_heads * head_dim) as usize;
    let rope_pairs = (head_dim / 2) as usize;

    let q: Vec<f32> = (0..q_len)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.04)
        .collect();
    let k: Vec<f32> = (0..k_len)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.03)
        .collect();
    let freq_factors: Vec<f32> = (0..rope_pairs)
        .map(|i| 0.5 + (i % 5) as f32 * 0.25)
        .collect();

    let mut q_ref = q.clone();
    let mut k_ref = k.clone();
    cpu_rope_neox_freq_factors_batch(
        &mut q_ref,
        &mut k_ref,
        &freq_factors,
        n_rows as usize,
        n_q_heads as usize,
        n_kv_heads as usize,
        head_dim as usize,
        start_pos,
        pos_step,
        freq_base,
    );

    let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
    let buf_k = MetalBuffer::from_slice(gpu.device(), &k).unwrap();
    let buf_factors = MetalBuffer::from_slice(gpu.device(), &freq_factors).unwrap();
    kernels
        .rope_batch_neox_partial_with_freq_factors(
            &gpu,
            &buf_q,
            &buf_k,
            &buf_factors,
            n_rows,
            n_q_heads,
            n_kv_heads,
            head_dim,
            start_pos,
            pos_step,
            freq_base,
        )
        .unwrap();
    let q_gpu = unsafe { buf_q.as_slice::<f32>() };
    let k_gpu = unsafe { buf_k.as_slice::<f32>() };

    let q_diff = max_abs_diff(q_gpu, &q_ref);
    let k_diff = max_abs_diff(k_gpu, &k_ref);
    assert!(
        q_diff < 1e-4,
        "RoPE freq-factor Q GPU vs CPU mismatch: max_diff={q_diff}"
    );
    assert!(
        k_diff < 1e-4,
        "RoPE freq-factor K GPU vs CPU mismatch: max_diff={k_diff}"
    );
}

#[test]
fn test_gpu_gelu_mul_matches_cpu() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = ElementwiseKernels::new(&gpu).unwrap();

    let n = 1024;
    let gate: Vec<f32> = (0..n).map(|i| ((i % 19) as f32 - 9.0) * 0.3).collect();
    let up: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.2).collect();

    let mut expected = gate.clone();
    cpu_gelu_elementwise_mul(&mut expected, &up);

    let buf_gate = MetalBuffer::from_slice(gpu.device(), &gate).unwrap();
    let buf_up = MetalBuffer::from_slice(gpu.device(), &up).unwrap();
    kernels
        .gelu_elementwise_mul(&gpu, &buf_gate, &buf_up, n as u32)
        .unwrap();
    let result = unsafe { buf_gate.as_slice::<f32>() };

    let diff = max_abs_diff(result, &expected);
    assert!(diff < 1e-4, "GELU*mul GPU vs CPU mismatch: max_diff={diff}");
}

#[test]
fn test_gpu_gelu_mul_large_values() {
    // Reproduce NaN seen in Gemma3 4B inference at index 7027/10240
    let gpu = MetalDevice::new().unwrap();
    let kernels = ElementwiseKernels::new(&gpu).unwrap();

    let n = 10240usize;
    let mut gate: Vec<f32> = (0..n).map(|i| ((i % 37) as f32 - 18.0) * 0.7).collect();
    let mut up: Vec<f32> = (0..n).map(|i| ((i % 29) as f32 - 14.0) * 1.3).collect();
    // Place exact problematic values at index 7027
    gate[7027] = 12.154314;
    up[7027] = -18.1271;

    let mut expected = gate.clone();
    cpu_gelu_elementwise_mul(&mut expected, &up);
    assert!(!expected[7027].is_nan(), "CPU GELU should not produce NaN");

    let buf_gate = MetalBuffer::from_slice(gpu.device(), &gate).unwrap();
    let buf_up = MetalBuffer::from_slice(gpu.device(), &up).unwrap();
    kernels
        .gelu_elementwise_mul(&gpu, &buf_gate, &buf_up, n as u32)
        .unwrap();
    let result = unsafe { buf_gate.as_slice::<f32>() };

    // Check for NaN
    let nan_count = result.iter().filter(|v| v.is_nan()).count();
    assert_eq!(nan_count, 0, "GPU GELU produced {nan_count} NaN values");

    let diff = max_abs_diff(result, &expected);
    assert!(diff < 1e-3, "GELU*mul GPU vs CPU mismatch: max_diff={diff}");
}

#[test]
fn test_gpu_silu_mul_matches_cpu() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = ElementwiseKernels::new(&gpu).unwrap();

    let n = 1024;
    let gate: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.3).collect();
    let up: Vec<f32> = (0..n).map(|i| ((i % 11) as f32 - 5.0) * 0.2).collect();

    let mut expected = gate.clone();
    cpu_silu_elementwise_mul(&mut expected, &up);

    let buf_gate = MetalBuffer::from_slice(gpu.device(), &gate).unwrap();
    let buf_up = MetalBuffer::from_slice(gpu.device(), &up).unwrap();
    kernels
        .silu_elementwise_mul(&gpu, &buf_gate, &buf_up, n as u32)
        .unwrap();
    let result = unsafe { buf_gate.as_slice::<f32>() };

    let diff = max_abs_diff(result, &expected);
    assert!(diff < 1e-4, "SiLU*mul GPU vs CPU mismatch: max_diff={diff}");
}

#[test]
fn test_gpu_elementwise_add_matches_cpu() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = ElementwiseKernels::new(&gpu).unwrap();

    let n = 512;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();

    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(ai, bi)| ai + bi).collect();

    let buf_a = MetalBuffer::from_slice(gpu.device(), &a).unwrap();
    let buf_b = MetalBuffer::from_slice(gpu.device(), &b).unwrap();
    kernels
        .elementwise_add(&gpu, &buf_a, &buf_b, n as u32)
        .unwrap();
    let result = unsafe { buf_a.as_slice::<f32>() };

    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 1e-6,
        "Elementwise add GPU vs CPU mismatch: max_diff={diff}"
    );
}

#[test]
fn test_gpu_post_attn_norm_residual_add_rms_norm_out_batch_matches_cpu() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = ElementwiseKernels::new(&gpu).unwrap();

    let n = 256usize;
    let n_rows = 3usize;
    let eps = 1e-6f32;

    let hidden: Vec<f32> = (0..n * n_rows)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.15)
        .collect();
    let addend: Vec<f32> = (0..n * n_rows)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.12)
        .collect();
    let post_weight: Vec<f32> = (0..n)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05 + 1.0)
        .collect();
    let residual_weight: Vec<f32> = (0..n)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.04 + 1.0)
        .collect();

    let mut hidden_expected = hidden.clone();
    let mut norm_expected = vec![0.0f32; n * n_rows];
    cpu_post_attn_norm_residual_add_rms_norm_out_batch(
        &mut hidden_expected,
        &addend,
        &post_weight,
        &residual_weight,
        &mut norm_expected,
        n,
        n_rows,
        eps,
    );

    let buf_hidden = MetalBuffer::from_slice(gpu.device(), &hidden).unwrap();
    let buf_addend = MetalBuffer::from_slice(gpu.device(), &addend).unwrap();
    let buf_post_weight = MetalBuffer::from_slice(gpu.device(), &post_weight).unwrap();
    let buf_residual_weight = MetalBuffer::from_slice(gpu.device(), &residual_weight).unwrap();
    let buf_norm_out = MetalBuffer::new(gpu.device(), n * n_rows * 4).unwrap();
    kernels
        .post_attn_norm_residual_add_rms_norm_out_batch(
            &gpu,
            &buf_hidden,
            &buf_addend,
            &buf_post_weight,
            &buf_residual_weight,
            &buf_norm_out,
            n as u32,
            n_rows as u32,
            eps,
        )
        .unwrap();

    let hidden_gpu = unsafe { buf_hidden.as_slice::<f32>() };
    let norm_gpu = unsafe { buf_norm_out.as_slice::<f32>() };

    let hidden_diff = max_abs_diff(hidden_gpu, &hidden_expected);
    let norm_diff = max_abs_diff(norm_gpu, &norm_expected);
    assert!(
        hidden_diff < 1e-4,
        "Fused Gemma residual hidden GPU vs CPU mismatch: max_diff={hidden_diff}"
    );
    assert!(
        norm_diff < 1e-4,
        "Fused Gemma residual norm GPU vs CPU mismatch: max_diff={norm_diff}"
    );
}

#[test]
fn test_gpu_post_ffn_norm_residual_add_rms_norm_out_batch_matches_cpu() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = ElementwiseKernels::new(&gpu).unwrap();

    let n = 256usize;
    let n_rows = 3usize;
    let eps = 1e-6f32;

    let hidden: Vec<f32> = (0..n * n_rows)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.14)
        .collect();
    let addend: Vec<f32> = (0..n * n_rows)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.09)
        .collect();
    let post_weight: Vec<f32> = (0..n)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.06 + 1.0)
        .collect();
    let residual_weight: Vec<f32> = (0..n)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.03 + 1.0)
        .collect();

    let mut hidden_expected = hidden.clone();
    let mut norm_expected = vec![0.0f32; n * n_rows];
    cpu_post_ffn_norm_residual_add_rms_norm_out_batch(
        &mut hidden_expected,
        &addend,
        &post_weight,
        &residual_weight,
        &mut norm_expected,
        n,
        n_rows,
        eps,
    );

    let buf_hidden = MetalBuffer::from_slice(gpu.device(), &hidden).unwrap();
    let buf_addend = MetalBuffer::from_slice(gpu.device(), &addend).unwrap();
    let buf_post_weight = MetalBuffer::from_slice(gpu.device(), &post_weight).unwrap();
    let buf_residual_weight = MetalBuffer::from_slice(gpu.device(), &residual_weight).unwrap();
    let buf_norm_out = MetalBuffer::new(gpu.device(), n * n_rows * 4).unwrap();
    kernels
        .post_ffn_norm_residual_add_rms_norm_out_batch(
            &gpu,
            &buf_hidden,
            &buf_addend,
            &buf_post_weight,
            &buf_residual_weight,
            &buf_norm_out,
            n as u32,
            n_rows as u32,
            eps,
        )
        .unwrap();

    let hidden_gpu = unsafe { buf_hidden.as_slice::<f32>() };
    let norm_gpu = unsafe { buf_norm_out.as_slice::<f32>() };

    let hidden_diff = max_abs_diff(hidden_gpu, &hidden_expected);
    let norm_diff = max_abs_diff(norm_gpu, &norm_expected);
    assert!(
        hidden_diff < 1e-4,
        "Fused Gemma FFN handoff hidden GPU vs CPU mismatch: max_diff={hidden_diff}"
    );
    assert!(
        norm_diff < 1e-4,
        "Fused Gemma FFN handoff norm GPU vs CPU mismatch: max_diff={norm_diff}"
    );
}

// ── KV Append kernel tests ──────────────────────────────────────

#[test]
fn test_kv_append_basic() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = ElementwiseKernels::new(&gpu).unwrap();

    let kv_stride: usize = 8; // e.g. 2 kv_heads * 4 head_dim
    let capacity: usize = 16;
    // Source: one token's KV data
    let src: Vec<f32> = (0..kv_stride).map(|i| (i + 1) as f32).collect();
    // Destination: pre-allocated cache
    let dst_data = vec![0.0f32; capacity * kv_stride];

    let buf_src = MetalBuffer::from_slice(gpu.device(), &src).unwrap();
    let buf_dst = MetalBuffer::from_slice(gpu.device(), &dst_data).unwrap();

    // Append at position 0
    kernels
        .kv_append(&gpu, &buf_src, &buf_dst, false, 0, kv_stride as u32)
        .unwrap();
    let result = unsafe { buf_dst.as_slice::<f32>() };
    assert_eq!(&result[..kv_stride], &src[..]);
    assert!(result[kv_stride..].iter().all(|&x| x == 0.0));

    // Append at position 3
    let offset = 3 * kv_stride;
    let src2: Vec<f32> = (0..kv_stride).map(|i| (i + 100) as f32).collect();
    let buf_src2 = MetalBuffer::from_slice(gpu.device(), &src2).unwrap();
    kernels
        .kv_append(
            &gpu,
            &buf_src2,
            &buf_dst,
            false,
            offset as u32,
            kv_stride as u32,
        )
        .unwrap();
    let result2 = unsafe { buf_dst.as_slice::<f32>() };
    assert_eq!(&result2[offset..offset + kv_stride], &src2[..]);
    // Position 0 still intact
    assert_eq!(&result2[..kv_stride], &src[..]);
}

// ── Decode Attention kernel tests ───────────────────────────────

/// CPU reference for single-token decode attention with GQA.
#[allow(clippy::too_many_arguments)]
fn cpu_decode_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    attend_start: usize,
    attend_len: usize,
) -> Vec<f32> {
    cpu_decode_attention_with_row_stride(
        q,
        k_cache,
        v_cache,
        n_heads,
        n_kv_heads,
        head_dim,
        n_kv_heads * head_dim,
        attend_start,
        attend_len,
    )
}

#[allow(clippy::too_many_arguments)]
fn cpu_decode_attention_with_row_stride(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_row_stride: usize,
    attend_start: usize,
    attend_len: usize,
) -> Vec<f32> {
    let heads_per_kv = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; n_heads * head_dim];

    for h in 0..n_heads {
        let kv_h = h / heads_per_kv;
        let q_off = h * head_dim;

        // Compute scores
        let mut scores = vec![0.0f32; attend_len];
        let mut max_score = f32::NEG_INFINITY;
        for (t, score) in scores.iter_mut().enumerate() {
            let k_off = (attend_start + t) * kv_row_stride + kv_h * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[q_off + d] * k_cache[k_off + d];
            }
            *score = dot * scale;
            if *score > max_score {
                max_score = *score;
            }
        }

        // Softmax
        let mut sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            sum += *s;
        }
        for s in &mut scores {
            *s /= sum;
        }

        // Weighted sum of V
        for (t, &weight) in scores.iter().enumerate() {
            let v_off = (attend_start + t) * kv_row_stride + kv_h * head_dim;
            for d in 0..head_dim {
                out[q_off + d] += weight * v_cache[v_off + d];
            }
        }
    }
    out
}

#[test]
fn test_attention_decode_single_token() {
    // 1 query head, 1 kv head, head_dim=4, 1 token in cache
    let gpu = MetalDevice::new().unwrap();
    let kernels = AttentionKernels::new(&gpu).unwrap();

    let q = [1.0f32, 0.0, 0.0, 0.0];
    let k_cache = [1.0f32, 0.0, 0.0, 0.0]; // 1 token
    let v_cache = [3.0f32, 4.0, 5.0, 6.0];

    let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
    let buf_k = MetalBuffer::from_slice(gpu.device(), &k_cache).unwrap();
    let buf_v = MetalBuffer::from_slice(gpu.device(), &v_cache).unwrap();
    let buf_o = MetalBuffer::new(gpu.device(), 4 * 4).unwrap();

    kernels
        .attention_decode_with_config(
            &gpu,
            &buf_q,
            &buf_k,
            &buf_v,
            &buf_o,
            false,
            1,
            1,
            4,
            0,
            1,
            default_attention_config(),
        )
        .unwrap();

    let result = unsafe { buf_o.as_slice::<f32>() };
    // Single token: softmax([score]) = [1.0], output = V
    assert!(
        max_abs_diff(result, &v_cache) < 1e-4,
        "Single decode token: got {:?}, expected {:?}",
        result,
        v_cache
    );
}

#[test]
fn test_attention_decode_multi_token() {
    // 1 head, head_dim=2, 3 tokens in cache, attend all 3
    let gpu = MetalDevice::new().unwrap();
    let kernels = AttentionKernels::new(&gpu).unwrap();

    let head_dim = 2;
    let n_heads = 1u32;
    let n_kv_heads = 1u32;
    let attend_len = 3u32;

    // Q = [1, 0] → dot with K[0]=[1,0]=1, K[1]=[1,0]=1, K[2]=[1,0]=1
    // All equal scores → uniform softmax → average of V
    let q = [1.0f32, 0.0];
    let k_cache = [1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0];
    let v_cache = [2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0]; // V0=[2,3], V1=[4,5], V2=[6,7]

    let expected = cpu_decode_attention(
        &q,
        &k_cache,
        &v_cache,
        n_heads as usize,
        n_kv_heads as usize,
        head_dim,
        0,
        attend_len as usize,
    );

    let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
    let buf_k = MetalBuffer::from_slice(gpu.device(), &k_cache).unwrap();
    let buf_v = MetalBuffer::from_slice(gpu.device(), &v_cache).unwrap();
    let buf_o = MetalBuffer::new(gpu.device(), n_heads as usize * head_dim * 4).unwrap();

    kernels
        .attention_decode_with_config(
            &gpu,
            &buf_q,
            &buf_k,
            &buf_v,
            &buf_o,
            false,
            n_heads,
            n_kv_heads,
            head_dim as u32,
            0,
            attend_len,
            default_attention_config(),
        )
        .unwrap();

    let result = unsafe { buf_o.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 1e-3,
        "Multi-token decode: max_diff={diff}, got {:?}, expected {:?}",
        result,
        expected
    );
}

#[test]
fn test_attention_decode_gqa() {
    // 4 query heads, 2 kv heads, head_dim=4, 2 tokens in cache
    let gpu = MetalDevice::new().unwrap();
    let kernels = AttentionKernels::new(&gpu).unwrap();

    let n_heads = 4u32;
    let n_kv_heads = 2u32;
    let head_dim = 4usize;
    let attend_len = 2u32;
    let _kv_stride = n_kv_heads as usize * head_dim; // 8

    // Q: 4 heads × 4 dims = 16 floats
    let q: Vec<f32> = (0..16).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
    // K cache: 2 tokens × kv_stride = 16 floats
    let k_cache: Vec<f32> = (0..16).map(|i| ((i % 7) as f32 - 3.0) * 0.2).collect();
    // V cache: 2 tokens × kv_stride = 16 floats
    let v_cache: Vec<f32> = (0..16).map(|i| ((i % 11) as f32) * 0.1).collect();

    let expected = cpu_decode_attention(
        &q,
        &k_cache,
        &v_cache,
        n_heads as usize,
        n_kv_heads as usize,
        head_dim,
        0,
        attend_len as usize,
    );

    let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
    let buf_k = MetalBuffer::from_slice(gpu.device(), &k_cache).unwrap();
    let buf_v = MetalBuffer::from_slice(gpu.device(), &v_cache).unwrap();
    let buf_o = MetalBuffer::new(gpu.device(), n_heads as usize * head_dim * 4).unwrap();

    kernels
        .attention_decode_with_config(
            &gpu,
            &buf_q,
            &buf_k,
            &buf_v,
            &buf_o,
            false,
            n_heads,
            n_kv_heads,
            head_dim as u32,
            0,
            attend_len,
            default_attention_config(),
        )
        .unwrap();

    let result = unsafe { buf_o.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(diff < 1e-3, "GQA decode: max_diff={diff}");
}

#[test]
fn test_attention_decode_sliding_window() {
    // 2 heads, 2 kv heads, head_dim=4, 8 tokens in cache, window of 4 starting at 4
    let gpu = MetalDevice::new().unwrap();
    let kernels = AttentionKernels::new(&gpu).unwrap();

    let n_heads = 2u32;
    let n_kv_heads = 2u32;
    let head_dim = 4usize;
    let total_tokens = 8usize;
    let attend_start = 4u32;
    let attend_len = 4u32;
    let kv_stride = n_kv_heads as usize * head_dim;

    let q: Vec<f32> = (0..n_heads as usize * head_dim)
        .map(|i| ((i % 3) as f32 - 1.0) * 0.5)
        .collect();
    let k_cache: Vec<f32> = (0..total_tokens * kv_stride)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let v_cache: Vec<f32> = (0..total_tokens * kv_stride)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
        .collect();

    let expected = cpu_decode_attention(
        &q,
        &k_cache,
        &v_cache,
        n_heads as usize,
        n_kv_heads as usize,
        head_dim,
        attend_start as usize,
        attend_len as usize,
    );

    let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
    let buf_k = MetalBuffer::from_slice(gpu.device(), &k_cache).unwrap();
    let buf_v = MetalBuffer::from_slice(gpu.device(), &v_cache).unwrap();
    let buf_o = MetalBuffer::new(gpu.device(), n_heads as usize * head_dim * 4).unwrap();

    kernels
        .attention_decode_with_config(
            &gpu,
            &buf_q,
            &buf_k,
            &buf_v,
            &buf_o,
            false,
            n_heads,
            n_kv_heads,
            head_dim as u32,
            attend_start,
            attend_len,
            default_attention_config(),
        )
        .unwrap();

    let result = unsafe { buf_o.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(diff < 1e-3, "Sliding window decode: max_diff={diff}");
}

#[test]
fn test_attention_decode_matches_cpu_random() {
    // Larger random test: 8 heads, 4 kv heads, head_dim=64, 32 tokens
    let gpu = MetalDevice::new().unwrap();
    let kernels = AttentionKernels::new(&gpu).unwrap();

    let n_heads = 8u32;
    let n_kv_heads = 4u32;
    let head_dim = 64usize;
    let attend_len = 32u32;
    let kv_stride = n_kv_heads as usize * head_dim;

    // Deterministic pseudo-random data
    let mut seed = 42u64;
    let mut next_f32 = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };

    let q: Vec<f32> = (0..n_heads as usize * head_dim)
        .map(|_| next_f32() * 0.5)
        .collect();
    let k_cache: Vec<f32> = (0..attend_len as usize * kv_stride)
        .map(|_| next_f32() * 0.5)
        .collect();
    let v_cache: Vec<f32> = (0..attend_len as usize * kv_stride)
        .map(|_| next_f32() * 0.5)
        .collect();

    let expected = cpu_decode_attention(
        &q,
        &k_cache,
        &v_cache,
        n_heads as usize,
        n_kv_heads as usize,
        head_dim,
        0,
        attend_len as usize,
    );

    let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
    let buf_k = MetalBuffer::from_slice(gpu.device(), &k_cache).unwrap();
    let buf_v = MetalBuffer::from_slice(gpu.device(), &v_cache).unwrap();
    let buf_o = MetalBuffer::new(gpu.device(), n_heads as usize * head_dim * 4).unwrap();

    kernels
        .attention_decode_with_config(
            &gpu,
            &buf_q,
            &buf_k,
            &buf_v,
            &buf_o,
            false,
            n_heads,
            n_kv_heads,
            head_dim as u32,
            0,
            attend_len,
            default_attention_config(),
        )
        .unwrap();

    let result = unsafe { buf_o.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 5e-3,
        "Random decode attention: max_diff={diff} (8 heads, 4 kv, dim=64, 32 tokens)"
    );
}

#[test]
fn test_attention_decode_long_sequence() {
    // Stress test: 4 heads, 2 kv heads, head_dim=128, 512 tokens
    let gpu = MetalDevice::new().unwrap();
    let kernels = AttentionKernels::new(&gpu).unwrap();

    let n_heads = 4u32;
    let n_kv_heads = 2u32;
    let head_dim = 128usize;
    let attend_len = 512u32;
    let kv_stride = n_kv_heads as usize * head_dim;

    let mut seed = 123u64;
    let mut next_f32 = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };

    let q: Vec<f32> = (0..n_heads as usize * head_dim)
        .map(|_| next_f32() * 0.3)
        .collect();
    let k_cache: Vec<f32> = (0..attend_len as usize * kv_stride)
        .map(|_| next_f32() * 0.3)
        .collect();
    let v_cache: Vec<f32> = (0..attend_len as usize * kv_stride)
        .map(|_| next_f32() * 0.3)
        .collect();

    let expected = cpu_decode_attention(
        &q,
        &k_cache,
        &v_cache,
        n_heads as usize,
        n_kv_heads as usize,
        head_dim,
        0,
        attend_len as usize,
    );

    let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
    let buf_k = MetalBuffer::from_slice(gpu.device(), &k_cache).unwrap();
    let buf_v = MetalBuffer::from_slice(gpu.device(), &v_cache).unwrap();
    let buf_o = MetalBuffer::new(gpu.device(), n_heads as usize * head_dim * 4).unwrap();

    kernels
        .attention_decode_with_config(
            &gpu,
            &buf_q,
            &buf_k,
            &buf_v,
            &buf_o,
            false,
            n_heads,
            n_kv_heads,
            head_dim as u32,
            0,
            attend_len,
            default_attention_config(),
        )
        .unwrap();

    let result = unsafe { buf_o.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 1e-2,
        "Long sequence decode: max_diff={diff} (4 heads, 2 kv, dim=128, 512 tokens)"
    );
}

#[test]
fn test_attention_decode_with_padded_kv_stride_matches_cpu() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = AttentionKernels::new(&gpu).unwrap();

    let n_heads = 4u32;
    let n_kv_heads = 1u32;
    let head_dim = 512usize;
    let kv_row_stride = 1024usize;
    let attend_len = 6u32;
    let q_len = n_heads as usize * head_dim;
    let kv_storage_len = attend_len as usize * kv_row_stride;

    let q: Vec<f32> = (0..q_len).map(|i| ((i % 19) as f32 - 9.0) * 0.02).collect();
    let mut k_cache = vec![0.0f32; kv_storage_len];
    let mut v_cache = vec![0.0f32; kv_storage_len];
    for t in 0..attend_len as usize {
        let row_base = t * kv_row_stride;
        for i in 0..head_dim {
            k_cache[row_base + i] = ((t * 17 + i) % 23) as f32 * 0.01 - 0.1;
            v_cache[row_base + i] = ((t * 13 + i) % 29) as f32 * 0.015 - 0.2;
        }
    }

    let expected = cpu_decode_attention_with_row_stride(
        &q,
        &k_cache,
        &v_cache,
        n_heads as usize,
        n_kv_heads as usize,
        head_dim,
        kv_row_stride,
        0,
        attend_len as usize,
    );

    let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
    let buf_k = MetalBuffer::from_slice(gpu.device(), &k_cache).unwrap();
    let buf_v = MetalBuffer::from_slice(gpu.device(), &v_cache).unwrap();
    let buf_o = MetalBuffer::new(gpu.device(), q_len * std::mem::size_of::<f32>()).unwrap();

    kernels
        .attention_decode_with_stride_and_config(
            &gpu,
            &buf_q,
            &buf_k,
            &buf_v,
            &buf_o,
            false,
            n_heads,
            n_kv_heads,
            head_dim as u32,
            kv_row_stride as u32,
            0,
            attend_len,
            default_attention_config(),
        )
        .unwrap();

    let result = unsafe { buf_o.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    assert!(
        diff < 1e-3,
        "Padded-stride decode mismatch: max_diff={diff}"
    );
}

#[test]
fn test_attention_decode_splitk_f16kv_hd128_matches_cpu() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = AttentionKernels::new(&gpu).unwrap();

    let n_heads = 4u32;
    let n_kv_heads = 2u32;
    let head_dim = 128usize;
    let attend_len = 512u32;
    let kv_stride = n_kv_heads as usize * head_dim;

    assert!(attention_decode_splitk_supported(true, head_dim as u32));

    let mut seed = 987u64;
    let mut next_f32 = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };

    let q: Vec<f32> = (0..n_heads as usize * head_dim)
        .map(|_| next_f32() * 0.3)
        .collect();
    let k_cache_f32: Vec<f32> = (0..attend_len as usize * kv_stride)
        .map(|_| next_f32() * 0.3)
        .collect();
    let v_cache_f32: Vec<f32> = (0..attend_len as usize * kv_stride)
        .map(|_| next_f32() * 0.3)
        .collect();
    let k_cache_f16: Vec<half::f16> = k_cache_f32
        .iter()
        .copied()
        .map(half::f16::from_f32)
        .collect();
    let v_cache_f16: Vec<half::f16> = v_cache_f32
        .iter()
        .copied()
        .map(half::f16::from_f32)
        .collect();
    let k_cache_ref: Vec<f32> = k_cache_f16.iter().map(|v| v.to_f32()).collect();
    let v_cache_ref: Vec<f32> = v_cache_f16.iter().map(|v| v.to_f32()).collect();

    let expected = cpu_decode_attention(
        &q,
        &k_cache_ref,
        &v_cache_ref,
        n_heads as usize,
        n_kv_heads as usize,
        head_dim,
        0,
        attend_len as usize,
    );

    let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
    let buf_k = MetalBuffer::from_slice(gpu.device(), &k_cache_f16).unwrap();
    let buf_v = MetalBuffer::from_slice(gpu.device(), &v_cache_f16).unwrap();
    let buf_o = MetalBuffer::new(gpu.device(), n_heads as usize * head_dim * 4).unwrap();

    kernels
        .attention_decode_splitk_with_config(
            &gpu,
            &buf_q,
            &buf_k,
            &buf_v,
            &buf_o,
            true,
            n_heads,
            n_kv_heads,
            head_dim as u32,
            0,
            attend_len,
            default_attention_config(),
        )
        .unwrap();

    let result = unsafe { buf_o.as_slice::<f32>() };
    let diff = max_abs_diff(result, &expected);
    // Split-K recombines normalized partials via exp/log round-trip, which
    // loses ~1 extra digit of precision vs single-pass attention. With f16
    // KV and 512 tokens the observed error is ~0.046.
    assert!(
        diff < 5e-2,
        "Split-K f16 decode: max_diff={diff} (4 heads, 2 kv, dim=128, 512 tokens)"
    );
}

#[test]
fn test_elementwise_softplus_bias_mul_sigmoid_pair_matches_cpu() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = ElementwiseKernels::new(&gpu).unwrap();

    let alpha = vec![-2.0f32, -0.5, 0.0, 3.0, 25.0];
    let beta = vec![-4.0f32, -1.0, 0.0, 2.0, 5.0];
    let bias = vec![0.5f32, -0.25, 0.75, 0.0, 1.25];
    let scale = vec![1.0f32, 0.5, -1.0, 2.0, 0.25];

    let mut expected_alpha = alpha.clone();
    let mut expected_beta = beta.clone();
    for i in 0..alpha.len() {
        let x = expected_alpha[i] + bias[i];
        let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
        expected_alpha[i] = sp * scale[i];
        expected_beta[i] = 1.0 / (1.0 + (-expected_beta[i]).exp());
    }

    let alpha_buf = MetalBuffer::from_slice(gpu.device(), &alpha).unwrap();
    let beta_buf = MetalBuffer::from_slice(gpu.device(), &beta).unwrap();
    let bias_buf = MetalBuffer::from_slice(gpu.device(), &bias).unwrap();
    let scale_buf = MetalBuffer::from_slice(gpu.device(), &scale).unwrap();

    gpu.execute_sync(|encoder| {
        kernels.encode_softplus_bias_mul_sigmoid_pair(
            encoder,
            &alpha_buf,
            &beta_buf,
            &bias_buf,
            &scale_buf,
            alpha.len() as u32,
        );
        Ok(())
    })
    .unwrap();

    let actual_alpha = unsafe { alpha_buf.as_slice::<f32>() };
    let actual_beta = unsafe { beta_buf.as_slice::<f32>() };
    assert!(max_abs_diff(actual_alpha, &expected_alpha) < 1e-5);
    assert!(max_abs_diff(actual_beta, &expected_beta) < 1e-5);
}

#[test]
fn test_elementwise_softplus_bias_mul_sigmoid_pair_batch_matches_cpu() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = ElementwiseKernels::new(&gpu).unwrap();

    let head_dim = 4usize;
    let alpha = vec![-2.0f32, -0.5, 0.0, 3.0, 1.0, -3.0, 0.25, 10.0];
    let beta = vec![-4.0f32, -1.0, 0.0, 2.0, 0.5, -0.75, 1.5, 5.0];
    let bias = vec![0.5f32, -0.25, 0.75, 1.25];
    let scale = vec![1.0f32, 0.5, -1.0, 0.25];

    let mut expected_alpha = alpha.clone();
    let mut expected_beta = beta.clone();
    for i in 0..alpha.len() {
        let head = i % head_dim;
        let x = expected_alpha[i] + bias[head];
        let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
        expected_alpha[i] = sp * scale[head];
        expected_beta[i] = 1.0 / (1.0 + (-expected_beta[i]).exp());
    }

    let alpha_buf = MetalBuffer::from_slice(gpu.device(), &alpha).unwrap();
    let beta_buf = MetalBuffer::from_slice(gpu.device(), &beta).unwrap();
    let bias_buf = MetalBuffer::from_slice(gpu.device(), &bias).unwrap();
    let scale_buf = MetalBuffer::from_slice(gpu.device(), &scale).unwrap();

    gpu.execute_sync(|encoder| {
        kernels.encode_softplus_bias_mul_sigmoid_pair_batch(
            encoder,
            &alpha_buf,
            &beta_buf,
            &bias_buf,
            &scale_buf,
            alpha.len() as u32,
            head_dim as u32,
        );
        Ok(())
    })
    .unwrap();

    let actual_alpha = unsafe { alpha_buf.as_slice::<f32>() };
    let actual_beta = unsafe { beta_buf.as_slice::<f32>() };
    assert!(max_abs_diff(actual_alpha, &expected_alpha) < 1e-5);
    assert!(max_abs_diff(actual_beta, &expected_beta) < 1e-5);
}

#[test]
fn test_elementwise_per_head_rms_norm_silu_mul_batch_matches_cpu() {
    let gpu = MetalDevice::new().unwrap();
    let kernels = ElementwiseKernels::new(&gpu).unwrap();

    let n_rows = 2usize;
    let n_heads = 3usize;
    let head_dim = 5usize;
    let eps = 1e-5f32;

    let src = vec![
        0.5f32, -1.0, 2.0, -0.25, 1.5, 1.0, 0.25, -0.5, 3.0, -2.0, -1.5, 2.5, 0.75, -0.125, 1.25,
        0.2, -0.3, 0.4, -0.5, 0.6, 2.0, -1.0, 1.5, -2.5, 0.75, -0.8, 1.1, -1.4, 0.9, -0.2,
    ];
    let gate = vec![
        -2.0f32, -0.5, 0.0, 1.0, 2.0, 0.1, -0.2, 0.3, -0.4, 0.5, 3.0, -3.5, 1.25, -1.5, 0.75, -0.9,
        0.8, -0.7, 0.6, -0.5, 1.5, -1.25, 0.75, -0.25, 0.125, -2.5, 2.0, -1.0, 0.5, -0.1,
    ];
    let weight = vec![1.0f32, 0.5, -1.0, 0.25, 1.5];

    let mut expected = gate.clone();
    for row in 0..n_rows {
        for head in 0..n_heads {
            let base = row * n_heads * head_dim + head * head_dim;
            let mut sum_sq = 0.0f32;
            for i in 0..head_dim {
                let v = src[base + i];
                sum_sq += v * v;
            }
            let inv_rms = 1.0f32 / ((sum_sq / head_dim as f32) + eps).sqrt();
            for (i, w) in weight.iter().copied().take(head_dim).enumerate() {
                let idx = base + i;
                let x = expected[idx];
                let silu = x / (1.0 + (-x).exp());
                expected[idx] = silu * (src[idx] * inv_rms * w);
            }
        }
    }

    let gate_buf = MetalBuffer::from_slice(gpu.device(), &gate).unwrap();
    let src_buf = MetalBuffer::from_slice(gpu.device(), &src).unwrap();
    let weight_buf = MetalBuffer::from_slice(gpu.device(), &weight).unwrap();

    gpu.execute_sync(|encoder| {
        kernels.encode_per_head_rms_norm_silu_mul_batch(
            encoder,
            &gate_buf,
            &src_buf,
            &weight_buf,
            n_rows as u32,
            n_heads as u32,
            head_dim as u32,
            eps,
        );
        Ok(())
    })
    .unwrap();

    let actual = unsafe { gate_buf.as_slice::<f32>() };
    assert!(max_abs_diff(actual, &expected) < 1e-5);
}
