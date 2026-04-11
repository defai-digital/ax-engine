//! Kernel dispatch profile system.
//!
//! Provides hardcoded per-architecture kernel dispatch parameters.
//! The primary constructor is `KernelProfile::for_head_dim(hd)`
//! which selects defaults based on the model's head dimension.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MatvecProfileVariant {
    Base,
    Nr2,
    Ilp4,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MatvecParams {
    #[serde(default = "default_matvec_tg_size")]
    pub threadgroup_size: u32,
    /// Number of output rows per simdgroup. llama.cpp uses 2 for Q4_K/Q6_K.
    /// When set to 2, selects the NR2 multi-row kernel variant (if available).
    /// Default: 1 (current AX kernel).
    #[serde(default = "default_matvec_rows_per_sg")]
    pub rows_per_simdgroup: u32,
    #[serde(default)]
    pub variant: Option<MatvecProfileVariant>,
}

fn default_matvec_tg_size() -> u32 {
    128
}

fn default_matvec_rows_per_sg() -> u32 {
    1
}

fn hardcoded_invariant_matvec_params(quant_type: &str) -> Option<MatvecParams> {
    match quant_type {
        "q4_k" => Some(MatvecParams {
            threadgroup_size: 64,
            rows_per_simdgroup: 2,
            variant: Some(MatvecProfileVariant::Nr2),
        }),
        "q5_k" => Some(MatvecParams {
            threadgroup_size: 64,
            rows_per_simdgroup: 1,
            variant: Some(MatvecProfileVariant::Base),
        }),
        "q6_k" => Some(MatvecParams {
            threadgroup_size: 64,
            rows_per_simdgroup: 2,
            variant: Some(MatvecProfileVariant::Nr2),
        }),
        "q8_0" => Some(MatvecParams {
            threadgroup_size: 128,
            rows_per_simdgroup: 2,
            variant: Some(MatvecProfileVariant::Nr2),
        }),
        _ => None,
    }
}

fn default_decode_matvec_profile() -> HashMap<String, MatvecParams> {
    HashMap::new()
}

impl Default for MatvecParams {
    fn default() -> Self {
        Self {
            threadgroup_size: default_matvec_tg_size(),
            rows_per_simdgroup: default_matvec_rows_per_sg(),
            variant: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AttentionDecodeParams {
    #[serde(default = "default_attn_decode_splitk_chunk_size")]
    pub splitk_chunk_size: u32,
    #[serde(default = "default_attn_decode_splitk_threshold")]
    pub splitk_threshold: u32,
    #[serde(default)]
    pub sdpa_default: Option<bool>,
    #[serde(default)]
    pub hd128_n2_default: Option<bool>,
}

fn default_attn_decode_splitk_chunk_size() -> u32 {
    256
}
fn default_attn_decode_splitk_threshold() -> u32 {
    512
}

impl Default for AttentionDecodeParams {
    fn default() -> Self {
        Self {
            splitk_chunk_size: default_attn_decode_splitk_chunk_size(),
            splitk_threshold: default_attn_decode_splitk_threshold(),
            sdpa_default: None,
            hd128_n2_default: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KvPrecisionMode {
    Auto,
    F32,
    F16,
    Q8_0,
}

fn default_kv_precision_mode() -> KvPrecisionMode {
    KvPrecisionMode::Auto
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct KvCacheParams {
    #[serde(default = "default_kv_precision_mode")]
    pub precision: KvPrecisionMode,
}

impl Default for KvCacheParams {
    fn default() -> Self {
        Self {
            precision: default_kv_precision_mode(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ProfileKernelMode {
    Off,
    On,
    Auto,
}

fn default_profile_kernel_mode_off() -> ProfileKernelMode {
    ProfileKernelMode::Off
}

fn default_profile_kernel_mode_auto() -> ProfileKernelMode {
    ProfileKernelMode::Auto
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BatchPrefillParams {
    #[serde(default = "default_batch_prefill_prefer_f16_io")]
    pub prefer_f16_io: bool,
    #[serde(default = "default_batch_prefill_prefer_pair")]
    pub prefer_pair_kernel: bool,
    #[serde(default = "default_batch_prefill_small_n_threshold")]
    pub small_n_threshold: u32,
    #[serde(default)]
    pub small_m_max: u32,
    #[serde(default = "default_batch_prefill_use_bn32")]
    pub use_bn32: bool,
    #[serde(default = "default_batch_prefill_use_bk32")]
    pub use_bk32: bool,
    #[serde(default = "default_batch_prefill_q8_f16in_full_min_n")]
    pub q8_f16in_full_min_n: u32,
}

fn default_batch_prefill_prefer_f16_io() -> bool {
    false // f16 B reads waste half the 128-byte cache line on Apple Silicon UMA.
    // f32 reads + inline cast is faster because it fills the full bus width.
}

fn default_batch_prefill_prefer_pair() -> bool {
    true
}

fn default_batch_prefill_small_n_threshold() -> u32 {
    1
}

fn default_batch_prefill_use_bn32() -> bool {
    false // BN=32 uses TG=128, which halves loading throughput. TG=256 always wins.
}

fn default_batch_prefill_use_bk32() -> bool {
    true
}

fn default_batch_prefill_q8_f16in_full_min_n() -> u32 {
    64
}

impl Default for BatchPrefillParams {
    fn default() -> Self {
        Self {
            prefer_f16_io: false,
            prefer_pair_kernel: true,
            small_n_threshold: default_batch_prefill_small_n_threshold(),
            small_m_max: 0,
            use_bn32: default_batch_prefill_use_bn32(),
            use_bk32: default_batch_prefill_use_bk32(),
            q8_f16in_full_min_n: default_batch_prefill_q8_f16in_full_min_n(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AttentionPrefillParams {
    #[serde(default = "default_profile_kernel_mode_off")]
    pub fa2_mode: ProfileKernelMode,
    #[serde(default = "default_profile_kernel_mode_auto")]
    pub fa2_hd128_mode: ProfileKernelMode,
    #[serde(default = "default_profile_kernel_mode_auto")]
    pub ax_bc64_mode: ProfileKernelMode,
    #[serde(default = "default_attention_prefill_fa2_auto_min_tokens")]
    pub fa2_auto_min_tokens: u32,
    #[serde(default = "default_attention_prefill_fa2_auto_min_base_seq")]
    pub fa2_auto_min_base_seq: u32,
    #[serde(default = "default_attention_prefill_fa2_hd128_auto_min_tokens")]
    pub fa2_hd128_auto_min_tokens: u32,
    #[serde(default = "default_attention_prefill_ax_bc64_min_tokens")]
    pub ax_bc64_min_tokens: u32,
}

fn default_attention_prefill_fa2_auto_min_tokens() -> u32 {
    512
}

fn default_attention_prefill_fa2_auto_min_base_seq() -> u32 {
    256
}

fn default_attention_prefill_fa2_hd128_auto_min_tokens() -> u32 {
    128
}

fn default_attention_prefill_ax_bc64_min_tokens() -> u32 {
    384
}

impl Default for AttentionPrefillParams {
    fn default() -> Self {
        Self {
            fa2_mode: default_profile_kernel_mode_off(),
            fa2_hd128_mode: default_profile_kernel_mode_auto(),
            ax_bc64_mode: default_profile_kernel_mode_auto(),
            fa2_auto_min_tokens: default_attention_prefill_fa2_auto_min_tokens(),
            fa2_auto_min_base_seq: default_attention_prefill_fa2_auto_min_base_seq(),
            fa2_hd128_auto_min_tokens: default_attention_prefill_fa2_hd128_auto_min_tokens(),
            ax_bc64_min_tokens: default_attention_prefill_ax_bc64_min_tokens(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct KernelProfile {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub generated: String,
    #[serde(default = "default_decode_matvec_profile")]
    pub decode_matvec: HashMap<String, MatvecParams>,
    #[serde(default)]
    pub batch_prefill: BatchPrefillParams,
    #[serde(default)]
    pub attention_decode: AttentionDecodeParams,
    #[serde(default)]
    pub attention_prefill: AttentionPrefillParams,
    #[serde(default)]
    pub kv_cache: KvCacheParams,
}

impl Default for KernelProfile {
    fn default() -> Self {
        Self {
            model: String::new(),
            source: "hardcoded".to_string(),
            generated: String::new(),
            decode_matvec: default_decode_matvec_profile(),
            batch_prefill: BatchPrefillParams::default(),
            attention_decode: AttentionDecodeParams::default(),
            attention_prefill: AttentionPrefillParams::default(),
            kv_cache: KvCacheParams::default(),
        }
    }
}

impl KernelProfile {
    /// Build a profile from shape properties (head_dim).
    /// This is the primary constructor -- no model name parsing needed.
    pub fn for_head_dim(head_dim: u32) -> Self {
        let mut profile = Self::default();
        // FA2 HD=256 (Gemma3) can't use the HD=128 FA2 kernels
        if head_dim == 256 {
            profile.attention_prefill.fa2_mode = ProfileKernelMode::Off;
        } else {
            profile.attention_prefill.fa2_mode = ProfileKernelMode::Auto;
        }
        profile
    }

    /// Build a profile from model name, inferring head_dim from architecture.
    pub fn for_model_arch(model_name: &str) -> Self {
        // Gemma3 models use head_dim=256, everything else uses head_dim=128.
        // Gemma4 has per-layer variable head dims (some 128, some 256), but the
        // default profile with fa2_mode=Auto handles this correctly since
        // dispatch checks head_dim per-layer at runtime.
        let head_dim = if model_name.to_lowercase().contains("gemma")
            && !model_name.to_lowercase().contains("gemma4")
            && !model_name.to_lowercase().contains("gemma-4")
        {
            256
        } else {
            128
        };
        Self::for_head_dim(head_dim)
    }

    pub fn load(model_name: &str, _quant: &str) -> Self {
        Self::for_model_arch(model_name)
    }

    pub fn matvec_params(&self, quant_type: &str) -> MatvecParams {
        self.decode_matvec
            .get(quant_type)
            .cloned()
            .unwrap_or_else(|| {
                self.decode_matvec
                    .get("default")
                    .cloned()
                    .or_else(|| hardcoded_invariant_matvec_params(quant_type))
                    .unwrap_or_default()
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_profile() {
        let profile = KernelProfile::default();
        assert_eq!(profile.source, "hardcoded");

        let q4k_matvec = profile.matvec_params("q4_k");
        assert_eq!(q4k_matvec.threadgroup_size, 64);
        assert_eq!(q4k_matvec.rows_per_simdgroup, 2);
        let q5k_matvec = profile.matvec_params("q5_k");
        assert_eq!(q5k_matvec.threadgroup_size, 64);
        assert_eq!(q5k_matvec.rows_per_simdgroup, 1);
        let q6k_matvec = profile.matvec_params("q6_k");
        assert_eq!(q6k_matvec.threadgroup_size, 64);
        assert_eq!(q6k_matvec.rows_per_simdgroup, 2);
        let q80_matvec = profile.matvec_params("q8_0");
        assert_eq!(q80_matvec.threadgroup_size, 128);
        assert_eq!(q80_matvec.rows_per_simdgroup, 2);
        assert!(!profile.batch_prefill.prefer_f16_io);
        assert!(profile.batch_prefill.prefer_pair_kernel);
        assert!(!profile.batch_prefill.use_bn32);
        // Default profile uses serde default (Off), not heuristic.
        assert_eq!(profile.attention_prefill.fa2_mode, ProfileKernelMode::Off);
    }

    #[test]
    fn test_for_head_dim_128_defaults() {
        let profile = KernelProfile::for_head_dim(128);
        // HD=128 models get FA2 Auto
        assert_eq!(profile.attention_prefill.fa2_mode, ProfileKernelMode::Auto);
        assert_eq!(
            profile.attention_prefill.fa2_hd128_mode,
            ProfileKernelMode::Auto
        );
        assert_eq!(
            profile.attention_prefill.ax_bc64_mode,
            ProfileKernelMode::Auto
        );
        assert!(!profile.batch_prefill.prefer_f16_io);
        assert_eq!(profile.attention_decode.hd128_n2_default, None);
    }

    #[test]
    fn test_for_head_dim_256_disables_fa2() {
        let profile = KernelProfile::for_head_dim(256);
        // HD=256 (Gemma3) disables FA2
        assert_eq!(profile.attention_prefill.fa2_mode, ProfileKernelMode::Off);
    }

    #[test]
    fn test_for_model_arch_gemma3_uses_hd256() {
        let profile = KernelProfile::for_model_arch("gemma-3-4b-it");
        assert_eq!(profile.attention_prefill.fa2_mode, ProfileKernelMode::Off);

        let profile = KernelProfile::for_model_arch("Gemma3-12B");
        assert_eq!(profile.attention_prefill.fa2_mode, ProfileKernelMode::Off);
    }

    #[test]
    fn test_for_model_arch_gemma4_uses_hd128() {
        // Gemma4 has per-layer variable head dims, default profile with Auto is correct
        let profile = KernelProfile::for_model_arch("Gemma4-27B");
        assert_eq!(profile.attention_prefill.fa2_mode, ProfileKernelMode::Auto);

        let profile = KernelProfile::for_model_arch("gemma-4-12b");
        assert_eq!(profile.attention_prefill.fa2_mode, ProfileKernelMode::Auto);
    }

    #[test]
    fn test_for_model_arch_non_gemma_uses_hd128() {
        let profile = KernelProfile::for_model_arch("Qwen3-8B-Q4_K_M");
        assert_eq!(profile.attention_prefill.fa2_mode, ProfileKernelMode::Auto);

        let profile = KernelProfile::for_model_arch("Llama-3-8B");
        assert_eq!(profile.attention_prefill.fa2_mode, ProfileKernelMode::Auto);

        let profile = KernelProfile::for_model_arch("Qwen3.5-9B");
        assert_eq!(profile.attention_prefill.fa2_mode, ProfileKernelMode::Auto);
    }

    #[test]
    fn test_hd128_prefill_dispatch_defaults() {
        // for_head_dim(128) should produce shape-based defaults
        let profile = KernelProfile::for_head_dim(128);
        let config = crate::dispatch::AttentionDispatchConfig::from_profile(&profile);

        // At 384 tokens, HD=128 should route to AxBc64 or FA2
        let selection = config.prefill_local_candidate_selection(384, 128);
        assert!(matches!(
            selection.candidate,
            crate::dispatch::AttentionPrefillCandidate::AxBc64
                | crate::dispatch::AttentionPrefillCandidate::Fa2SimdHd128
                | crate::dispatch::AttentionPrefillCandidate::Fa2Hd128
                | crate::dispatch::AttentionPrefillCandidate::Fa2HalfHd128
        ));
    }

    #[test]
    fn test_hd128_fa2_at_large_n_tokens() {
        let profile = KernelProfile::for_head_dim(128);
        let config = crate::dispatch::AttentionDispatchConfig::from_profile(&profile);

        // At 512 tokens with HD=128 and FA2 Auto, should route to FA2
        let selection = config.prefill_local_candidate_selection(512, 128);
        assert!(matches!(
            selection.candidate,
            crate::dispatch::AttentionPrefillCandidate::Fa2SimdHd128
                | crate::dispatch::AttentionPrefillCandidate::Fa2Hd128
                | crate::dispatch::AttentionPrefillCandidate::Fa2HalfHd128
                | crate::dispatch::AttentionPrefillCandidate::AxBc64
        ));
    }

    #[test]
    fn test_fallback_matvec_params() {
        let profile = KernelProfile::default();
        let unknown = profile.matvec_params("unknown_quant");
        assert_eq!(unknown.threadgroup_size, 128);
    }

    #[test]
    fn test_profile_json_missing_decode_matvec_uses_hardcoded_invariants() {
        let profile: KernelProfile = serde_json::from_str(
            r#"{
                "model": "test",
                "source": "unit",
                "generated": "2026-03-28"
            }"#,
        )
        .unwrap();

        let q4k = profile.matvec_params("q4_k");
        assert_eq!(q4k.threadgroup_size, 64);
        assert_eq!(q4k.rows_per_simdgroup, 2);

        let q80 = profile.matvec_params("q8_0");
        assert_eq!(q80.threadgroup_size, 128);
        assert_eq!(q80.rows_per_simdgroup, 2);
    }
}
