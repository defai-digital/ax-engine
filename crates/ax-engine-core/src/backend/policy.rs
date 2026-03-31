use crate::kv::gpu_kv::GpuKvDtype;
use ax_engine_metal::{AttentionDispatchConfig, DequantDispatchConfig, KernelProfile};

fn normalized_arch_name(arch: &str) -> &str {
    match arch {
        "llama" => "llama",
        "qwen2" | "qwen3" => "qwen3",
        "qwen35" => "qwen35",
        "gemma" | "gemma2" | "gemma3" => "gemma3",
        _ => arch,
    }
}

fn parse_bool_toggle(v: &str) -> Option<bool> {
    let v = v.trim().to_ascii_lowercase();
    if v == "1" || v == "true" || v == "on" {
        Some(true)
    } else if v == "0" || v == "false" || v == "off" {
        Some(false)
    } else {
        None
    }
}

fn env_bool_with_arch_override(base: &str, arch: &str) -> Option<bool> {
    let arch_key = format!("{base}_{}", normalized_arch_name(arch).to_ascii_uppercase());
    if let Ok(v) = std::env::var(&arch_key) {
        return parse_bool_toggle(&v);
    }
    std::env::var(base).ok().and_then(|v| parse_bool_toggle(&v))
}

fn env_u32(base: &str, default: u32) -> u32 {
    std::env::var(base)
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(default)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AutoToggle {
    Off,
    On,
    Auto,
}

fn parse_auto_toggle(v: &str) -> Option<AutoToggle> {
    let v = v.trim().to_ascii_lowercase();
    match v.as_str() {
        "1" | "true" | "on" => Some(AutoToggle::On),
        "0" | "false" | "off" => Some(AutoToggle::Off),
        "auto" => Some(AutoToggle::Auto),
        _ => None,
    }
}

fn env_auto_toggle_with_arch_override(base: &str, arch: &str) -> Option<AutoToggle> {
    let arch_key = format!("{base}_{}", normalized_arch_name(arch).to_ascii_uppercase());
    if let Ok(v) = std::env::var(&arch_key) {
        return parse_auto_toggle(&v);
    }
    std::env::var(base).ok().and_then(|v| parse_auto_toggle(&v))
}

/// Policy for GPU KV storage precision.
///
/// This is resolved once into `RuntimePolicy` so model/KV construction does not
/// need to read env vars or duplicate the auto-selection rule.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum KvPrecisionPolicy {
    #[default]
    Auto,
    ForceF32,
    ForceF16,
    ForceQ8_0,
    ForceQ4_0,
}

impl KvPrecisionPolicy {
    fn from_env_override() -> Option<Self> {
        match std::env::var("AX_METAL_F16_KV_CACHE") {
            Ok(v) => {
                let v = v.trim().to_ascii_lowercase();
                Some(match v.as_str() {
                    "1" | "true" | "on" => Self::ForceF16,
                    "0" | "false" | "off" => Self::ForceF32,
                    "q8" | "q8_0" => Self::ForceQ8_0,
                    "q4" | "q4_0" => Self::ForceQ4_0,
                    _ => Self::Auto,
                })
            }
            Err(_) => None,
        }
    }

    pub fn gpu_kv_dtype(self, context_len: usize) -> GpuKvDtype {
        match self {
            Self::Auto => {
                if context_len >= 256 {
                    GpuKvDtype::F16
                } else {
                    GpuKvDtype::F32
                }
            }
            Self::ForceF32 => GpuKvDtype::F32,
            Self::ForceF16 => GpuKvDtype::F16,
            Self::ForceQ8_0 => GpuKvDtype::Q8_0,
            Self::ForceQ4_0 => GpuKvDtype::Q4_0,
        }
    }
}

/// Typed runtime policy resolved once per backend/model instance.
///
/// This is the normal policy source of truth for Metal-backed execution.
/// Profile loading and later runtime presets should hydrate this object,
/// while hot-path code consumes the derived dispatch snapshots.
#[derive(Debug, Clone)]
pub struct RuntimePolicy {
    kernel_profile: KernelProfile,
    dequant_dispatch: DequantDispatchConfig,
    attention_dispatch: AttentionDispatchConfig,
    kv_precision: KvPrecisionPolicy,
    batch_prefill_f16_io: bool,
    batch_prefill_pair_kernel: bool,
    fused_qkv_prefill: bool,
    decode_fused_qkv: bool,
    batch_simd: bool,
    precompute_f16: bool,
    autotune_f16in_batch_route: bool,
    q8_batch_native: bool,
    q8_native_m_min: u32,
    q8_native_m_max: u32,
    q8_native_k_min: u32,
    q8_native_k_max: u32,
}

impl Default for RuntimePolicy {
    fn default() -> Self {
        Self::from_kernel_profile(KernelProfile::default())
    }
}

impl RuntimePolicy {
    pub fn from_kernel_profile(kernel_profile: KernelProfile) -> Self {
        let dequant_dispatch = DequantDispatchConfig::from_profile(&kernel_profile);
        let attention_dispatch = AttentionDispatchConfig::from_profile(&kernel_profile);
        let batch_prefill_f16_io = kernel_profile.batch_prefill.prefer_f16_io;
        let batch_prefill_pair_kernel = kernel_profile.batch_prefill.prefer_pair_kernel;
        Self {
            kernel_profile,
            dequant_dispatch,
            attention_dispatch,
            kv_precision: KvPrecisionPolicy::default(),
            batch_prefill_f16_io,
            batch_prefill_pair_kernel,
            fused_qkv_prefill: true,
            decode_fused_qkv: false,
            batch_simd: ax_engine_metal::batch_simd_enabled(),
            precompute_f16: false,
            autotune_f16in_batch_route: false,
            q8_batch_native: true,
            q8_native_m_min: 0,
            q8_native_m_max: u32::MAX,
            q8_native_k_min: 0,
            q8_native_k_max: u32::MAX,
        }
    }

    pub fn resolved_defaults() -> Self {
        Self::default().with_env_overrides("default")
    }

    pub fn for_model(model_name: &str, quant: &str, architecture: &str) -> Self {
        Self::from_kernel_profile(KernelProfile::load(model_name, quant))
            .with_env_overrides(architecture)
    }

    pub fn kernel_profile(&self) -> &KernelProfile {
        &self.kernel_profile
    }

    pub fn dequant_dispatch_config(&self) -> DequantDispatchConfig {
        self.dequant_dispatch
    }

    pub fn attention_dispatch_config(&self) -> AttentionDispatchConfig {
        self.attention_dispatch
    }

    pub fn batch_prefill_prefers_f16_io(&self) -> bool {
        self.batch_prefill_f16_io
    }

    pub fn batch_prefill_prefers_pair_kernel(&self) -> bool {
        self.batch_prefill_pair_kernel
    }

    pub fn fused_qkv_prefill_enabled(&self) -> bool {
        self.fused_qkv_prefill
    }

    pub fn decode_fused_qkv_enabled(&self) -> bool {
        self.decode_fused_qkv
    }

    pub fn batch_simd_enabled(&self) -> bool {
        self.batch_simd
    }

    pub fn precompute_f16_enabled(&self) -> bool {
        self.precompute_f16
    }

    pub fn autotune_f16in_batch_route_enabled(&self) -> bool {
        self.autotune_f16in_batch_route
    }

    pub fn q8_batch_native_enabled(&self) -> bool {
        self.q8_batch_native
    }

    pub fn q8_batch_native_shape_enabled(&self, m: u32, _n: u32, k: u32) -> bool {
        self.q8_batch_native
            && m >= self.q8_native_m_min
            && m <= self.q8_native_m_max
            && k >= self.q8_native_k_min
            && k <= self.q8_native_k_max
    }

    pub fn kv_precision_policy(&self) -> KvPrecisionPolicy {
        self.kv_precision
    }

    pub fn gpu_kv_dtype(&self, context_len: usize) -> GpuKvDtype {
        self.kv_precision.gpu_kv_dtype(context_len)
    }

    pub fn uses_f16_gpu_kv(&self, context_len: usize) -> bool {
        self.gpu_kv_dtype(context_len) == GpuKvDtype::F16
    }

    pub fn with_dequant_dispatch(mut self, dequant_dispatch: DequantDispatchConfig) -> Self {
        self.dequant_dispatch = dequant_dispatch;
        self
    }

    pub fn with_kv_precision_policy(mut self, kv_precision: KvPrecisionPolicy) -> Self {
        self.kv_precision = kv_precision;
        self
    }

    fn with_env_overrides(mut self, architecture: &str) -> Self {
        if architecture == "gemma3" {
            self.decode_fused_qkv = true;
        }
        if let Some(kv_precision) = KvPrecisionPolicy::from_env_override() {
            self.kv_precision = kv_precision;
        }
        if let Some(batch_prefill_f16_io) =
            env_bool_with_arch_override("AX_METAL_BATCH_F16_IO", architecture)
        {
            self.batch_prefill_f16_io = batch_prefill_f16_io;
        }
        if let Some(batch_prefill_pair_kernel) =
            env_bool_with_arch_override("AX_METAL_BATCH_F16_PAIR", architecture)
        {
            self.batch_prefill_pair_kernel = batch_prefill_pair_kernel;
        }
        if let Some(fused_qkv_prefill) =
            env_bool_with_arch_override("AX_METAL_FUSED_QKV", architecture)
        {
            self.fused_qkv_prefill = fused_qkv_prefill;
        }
        if let Some(decode_fused_qkv) =
            env_bool_with_arch_override("AX_METAL_DECODE_FUSED_QKV", architecture)
        {
            self.decode_fused_qkv = decode_fused_qkv;
        }
        if let Some(batch_simd) = env_bool_with_arch_override("AX_METAL_BATCH_SIMD", architecture) {
            self.batch_simd = batch_simd;
        }
        self.autotune_f16in_batch_route =
            env_bool_with_arch_override("AX_METAL_AUTOTUNE", architecture).unwrap_or(false);
        self.q8_batch_native =
            env_bool_with_arch_override("AX_METAL_Q8_BATCH_NATIVE", architecture).unwrap_or(true);
        self.q8_native_m_min = env_u32("AX_METAL_Q8_NATIVE_M_MIN", 0);
        self.q8_native_m_max = env_u32("AX_METAL_Q8_NATIVE_M_MAX", u32::MAX);
        self.q8_native_k_min = env_u32("AX_METAL_Q8_NATIVE_K_MIN", 0);
        self.q8_native_k_max = env_u32("AX_METAL_Q8_NATIVE_K_MAX", u32::MAX);
        self.precompute_f16 =
            match env_auto_toggle_with_arch_override("AX_PRECOMPUTE_F16", architecture)
                .unwrap_or(AutoToggle::Auto)
            {
                AutoToggle::Off => false,
                AutoToggle::On => true,
                AutoToggle::Auto => false,
            };
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_policy_default_matches_profile_defaults() {
        let policy = RuntimePolicy::default();
        let profile = KernelProfile::default();

        assert_eq!(
            policy.dequant_dispatch_config(),
            DequantDispatchConfig::from_profile(&profile)
        );
        assert_eq!(
            policy.attention_dispatch_config(),
            AttentionDispatchConfig::from_profile(&profile)
        );
        assert_eq!(
            policy.batch_prefill_prefers_f16_io(),
            profile.batch_prefill.prefer_f16_io
        );
        assert_eq!(
            policy.batch_prefill_prefers_pair_kernel(),
            profile.batch_prefill.prefer_pair_kernel
        );
        assert!(policy.fused_qkv_prefill_enabled());
        assert!(!policy.decode_fused_qkv_enabled());
        assert_eq!(
            policy.batch_simd_enabled(),
            ax_engine_metal::batch_simd_enabled()
        );
        assert!(!policy.precompute_f16_enabled());
        assert!(!policy.autotune_f16in_batch_route_enabled());
        assert!(policy.q8_batch_native_shape_enabled(128, 32, 4096));
        assert_eq!(policy.kv_precision_policy(), KvPrecisionPolicy::Auto);
        assert_eq!(policy.gpu_kv_dtype(128), GpuKvDtype::F32);
        assert_eq!(policy.gpu_kv_dtype(256), GpuKvDtype::F16);
    }

    #[test]
    fn test_runtime_policy_keeps_decode_fused_qkv_disabled_for_qwen3_by_default() {
        let policy = RuntimePolicy::for_model("qwen3-8b", "q4_k_m", "qwen3");
        assert!(!policy.decode_fused_qkv_enabled());
    }

    #[test]
    fn test_runtime_policy_enables_decode_fused_qkv_for_gemma3_by_default() {
        let policy = RuntimePolicy::for_model("gemma3-12b", "q4_k_m", "gemma3");
        assert!(policy.decode_fused_qkv_enabled());
    }

    #[test]
    fn test_runtime_policy_with_dequant_dispatch_overrides_dispatch_only() {
        let policy = RuntimePolicy::default();
        let mut tuned = policy.dequant_dispatch_config();
        tuned.batch_f16in_small_n_threshold = 9;
        tuned.batch_f16in_small_m_max = 128;

        let tuned_policy = policy.clone().with_dequant_dispatch(tuned);

        assert_eq!(tuned_policy.dequant_dispatch_config(), tuned);
        assert_eq!(
            tuned_policy.attention_dispatch_config(),
            policy.attention_dispatch_config()
        );
        assert_eq!(
            tuned_policy.batch_prefill_prefers_f16_io(),
            policy.batch_prefill_prefers_f16_io()
        );
    }

    #[test]
    fn test_runtime_policy_with_kv_precision_policy_overrides_only_kv_selection() {
        let policy = RuntimePolicy::default();
        let forced = policy
            .clone()
            .with_kv_precision_policy(KvPrecisionPolicy::ForceF32);

        assert_eq!(forced.kv_precision_policy(), KvPrecisionPolicy::ForceF32);
        assert_eq!(forced.gpu_kv_dtype(4096), GpuKvDtype::F32);
        assert_eq!(
            forced.dequant_dispatch_config(),
            policy.dequant_dispatch_config()
        );
        assert_eq!(
            forced.attention_dispatch_config(),
            policy.attention_dispatch_config()
        );
    }
}
