//! Kernel dispatch profile system.
//!
//! Provides JSON-driven configuration for kernel dispatch parameters.
//! This enables instant parameter iteration without recompilation.
//!
//! # Environment Variables
//!
//! - `AX_KERNEL_PROFILE_PATH`: Path to a JSON profile file (overrides auto-detection)
//!
//! # Profile Resolution Order
//!
//! 1. `AX_KERNEL_PROFILE_PATH` env var (if set)
//! 2. `perfs/<model>-<quant>.json` (exact match)
//! 3. `perfs/<architecture>.json` (e.g., `qwen3-8b.json`)
//! 4. `perfs/default.json`
//! 5. Hardcoded defaults

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

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
}

fn default_matvec_tg_size() -> u32 {
    128
}

fn default_matvec_rows_per_sg() -> u32 {
    1
}

impl Default for MatvecParams {
    fn default() -> Self {
        Self {
            threadgroup_size: default_matvec_tg_size(),
            rows_per_simdgroup: default_matvec_rows_per_sg(),
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct MatvecParamsOverride {
    #[serde(default)]
    pub threadgroup_size: Option<u32>,
    #[serde(default)]
    pub rows_per_simdgroup: Option<u32>,
}

impl MatvecParamsOverride {
    fn apply_to(&self, base: MatvecParams) -> MatvecParams {
        MatvecParams {
            threadgroup_size: self.threadgroup_size.unwrap_or(base.threadgroup_size),
            rows_per_simdgroup: self.rows_per_simdgroup.unwrap_or(base.rows_per_simdgroup),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct AttentionDecodeParamsOverride {
    #[serde(default)]
    pub splitk_chunk_size: Option<u32>,
    #[serde(default)]
    pub splitk_threshold: Option<u32>,
    #[serde(default)]
    pub sdpa_default: Option<bool>,
    #[serde(default)]
    pub hd128_n2_default: Option<bool>,
}

impl AttentionDecodeParamsOverride {
    fn apply_to(&self, base: AttentionDecodeParams) -> AttentionDecodeParams {
        AttentionDecodeParams {
            splitk_chunk_size: self.splitk_chunk_size.unwrap_or(base.splitk_chunk_size),
            splitk_threshold: self.splitk_threshold.unwrap_or(base.splitk_threshold),
            sdpa_default: self.sdpa_default.or(base.sdpa_default),
            hd128_n2_default: self.hd128_n2_default.or(base.hd128_n2_default),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct DecodeRegimeOverrides {
    #[serde(default)]
    pub decode_matvec: HashMap<String, MatvecParamsOverride>,
    #[serde(default)]
    pub attention_decode: AttentionDecodeParamsOverride,
}

fn default_decode_short_max_attend_len() -> u32 {
    512
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DecodeRegimeProfile {
    #[serde(default = "default_decode_short_max_attend_len")]
    pub short_max_attend_len: u32,
    #[serde(default)]
    pub short: DecodeRegimeOverrides,
    #[serde(default)]
    pub long: DecodeRegimeOverrides,
}

impl Default for DecodeRegimeProfile {
    fn default() -> Self {
        Self {
            short_max_attend_len: default_decode_short_max_attend_len(),
            short: DecodeRegimeOverrides::default(),
            long: DecodeRegimeOverrides::default(),
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
    #[serde(default = "default_profile_kernel_mode_off")]
    pub fa2_hd128_mode: ProfileKernelMode,
    #[serde(default = "default_profile_kernel_mode_auto")]
    pub mistral_bc64_mode: ProfileKernelMode,
    #[serde(default = "default_attention_prefill_fa2_auto_min_tokens")]
    pub fa2_auto_min_tokens: u32,
    #[serde(default = "default_attention_prefill_fa2_auto_min_base_seq")]
    pub fa2_auto_min_base_seq: u32,
    #[serde(default = "default_attention_prefill_fa2_hd128_auto_min_tokens")]
    pub fa2_hd128_auto_min_tokens: u32,
    #[serde(default = "default_attention_prefill_mistral_bc64_min_tokens")]
    pub mistral_bc64_min_tokens: u32,
}

fn default_attention_prefill_fa2_auto_min_tokens() -> u32 {
    512
}

fn default_attention_prefill_fa2_auto_min_base_seq() -> u32 {
    256
}

fn default_attention_prefill_fa2_hd128_auto_min_tokens() -> u32 {
    512
}

fn default_attention_prefill_mistral_bc64_min_tokens() -> u32 {
    384
}

impl Default for AttentionPrefillParams {
    fn default() -> Self {
        Self {
            fa2_mode: default_profile_kernel_mode_off(),
            fa2_hd128_mode: default_profile_kernel_mode_off(),
            mistral_bc64_mode: default_profile_kernel_mode_auto(),
            fa2_auto_min_tokens: default_attention_prefill_fa2_auto_min_tokens(),
            fa2_auto_min_base_seq: default_attention_prefill_fa2_auto_min_base_seq(),
            fa2_hd128_auto_min_tokens: default_attention_prefill_fa2_hd128_auto_min_tokens(),
            mistral_bc64_min_tokens: default_attention_prefill_mistral_bc64_min_tokens(),
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
    #[serde(default)]
    pub decode_matvec: HashMap<String, MatvecParams>,
    #[serde(default)]
    pub batch_prefill: BatchPrefillParams,
    #[serde(default)]
    pub attention_decode: AttentionDecodeParams,
    #[serde(default)]
    pub attention_prefill: AttentionPrefillParams,
    #[serde(default)]
    pub decode_regimes: Option<DecodeRegimeProfile>,
}

impl Default for KernelProfile {
    fn default() -> Self {
        let mut decode_matvec = HashMap::new();
        decode_matvec.insert("q4_k".to_string(), MatvecParams::default());
        decode_matvec.insert("q5_k".to_string(), MatvecParams::default());
        decode_matvec.insert("q6_k".to_string(), MatvecParams::default());
        decode_matvec.insert("q8_0".to_string(), MatvecParams::default());

        Self {
            model: String::new(),
            source: "hardcoded".to_string(),
            generated: String::new(),
            decode_matvec,
            batch_prefill: BatchPrefillParams::default(),
            attention_decode: AttentionDecodeParams::default(),
            attention_prefill: AttentionPrefillParams::default(),
            decode_regimes: None,
        }
    }
}

impl KernelProfile {
    pub fn load(model_name: &str, quant: &str) -> Self {
        Self::load_relative_to(Path::new("."), model_name, quant)
    }

    fn load_relative_to(base_dir: &Path, model_name: &str, quant: &str) -> Self {
        if let Some(profile) = Self::try_load_from_env() {
            return profile;
        }

        let sanitized_model = sanitize_name(model_name);
        let sanitized_quant = sanitize_name(quant);

        if let Some(profile) = Self::try_load_exact(base_dir, &sanitized_model, &sanitized_quant) {
            return profile;
        }

        let arch = extract_architecture(model_name);
        if let Some(profile) = Self::try_load_arch(base_dir, &arch) {
            return profile;
        }

        if let Some(profile) = Self::try_load_default(base_dir) {
            return profile;
        }

        Self::default()
    }

    fn try_load_from_env() -> Option<Self> {
        let path = std::env::var("AX_KERNEL_PROFILE_PATH")
            .ok()
            .filter(|s| !s.is_empty())?;
        Self::load_from_path(&path)
    }

    fn try_load_exact(base_dir: &Path, model: &str, quant: &str) -> Option<Self> {
        let filename = base_dir.join("perfs").join(format!("{model}-{quant}.json"));
        Self::load_from_path(&filename)
    }

    fn try_load_arch(base_dir: &Path, arch: &str) -> Option<Self> {
        let filename = base_dir.join("perfs").join(format!("{arch}.json"));
        Self::load_from_path(&filename)
    }

    fn try_load_default(base_dir: &Path) -> Option<Self> {
        Self::load_from_path(base_dir.join("perfs").join("default.json"))
    }

    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Option<Self> {
        let path = path.as_ref();
        match std::fs::read_to_string(path) {
            Ok(content) => match serde_json::from_str::<KernelProfile>(&content) {
                Ok(profile) => {
                    tracing::info!(
                        path = %path.display(),
                        model = %profile.model,
                        source = %profile.source,
                        "Loaded kernel profile"
                    );
                    Some(profile)
                }
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "Failed to parse kernel profile JSON"
                    );
                    None
                }
            },
            Err(e) => {
                tracing::debug!(path = %path.display(), error = %e, "Kernel profile not found");
                None
            }
        }
    }

    pub fn matvec_params(&self, quant_type: &str) -> MatvecParams {
        self.decode_matvec
            .get(quant_type)
            .cloned()
            .unwrap_or_else(|| {
                self.decode_matvec
                    .get("default")
                    .cloned()
                    .unwrap_or_default()
            })
    }

    pub fn effective_decode_profile(&self, attend_len: u32) -> Self {
        let Some(regimes) = &self.decode_regimes else {
            return self.clone();
        };
        let overrides = if attend_len <= regimes.short_max_attend_len {
            &regimes.short
        } else {
            &regimes.long
        };

        let mut effective = self.clone();
        for (quant, override_params) in &overrides.decode_matvec {
            let base = effective.matvec_params(quant);
            effective
                .decode_matvec
                .insert(quant.clone(), override_params.apply_to(base));
        }
        effective.attention_decode = overrides
            .attention_decode
            .apply_to(effective.attention_decode.clone());
        effective.decode_regimes = None;
        effective
    }
}

fn sanitize_name(name: &str) -> String {
    name.to_lowercase()
        .replace([' ', '.', '/'], "-")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
        .collect()
}

fn approx_size(size: f32, target: f32) -> bool {
    (size - target).abs() < 0.11
}

fn family_size_bucket(family: &str, size: f32) -> Option<&'static str> {
    match family {
        "qwen3" => {
            // Generative models: 6B+ only
            if approx_size(size, 6.0) || approx_size(size, 7.0) || approx_size(size, 8.0) {
                Some("8b")
            } else if approx_size(size, 14.0) {
                Some("14b")
            } else if approx_size(size, 32.0) {
                Some("32b")
            } else if approx_size(size, 72.0) {
                Some("72b")
            } else {
                None
            }
        }
        "qwen35" => {
            // Generative models: 6B+ only
            if approx_size(size, 7.0) || approx_size(size, 8.0) || approx_size(size, 9.0) {
                Some("9b")
            } else if approx_size(size, 27.0) {
                Some("27b")
            } else if approx_size(size, 30.0) {
                Some("30b")
            } else if approx_size(size, 35.0) {
                Some("35b")
            } else if approx_size(size, 397.0) {
                Some("397b")
            } else {
                None
            }
        }
        "gemma3" => {
            if approx_size(size, 4.0) {
                Some("4b")
            } else if approx_size(size, 12.0) {
                Some("12b")
            } else if approx_size(size, 27.0) {
                Some("27b")
            } else {
                None
            }
        }
        "llama3" => {
            // Generative models: 6B+ only
            if approx_size(size, 7.0) || approx_size(size, 8.0) {
                Some("8b")
            } else if approx_size(size, 70.0) {
                Some("70b")
            } else {
                None
            }
        }
        _ => None,
    }
}

fn extract_architecture(model_name: &str) -> String {
    let lower = model_name.to_lowercase();

    // Order matters: more specific patterns before general ones.

    // Qwen3.5 hybrid family has its own forward path and tuning namespace.
    if lower.contains("qwen3.5") || lower.contains("qwen35") {
        return match_size(&lower, "qwen35");
    }

    // Qwen family (qwen, qwen2, qwen2.5, qwen3) — uses qwen3 forward pass
    if lower.contains("qwen") {
        return match_size(&lower, "qwen3");
    }

    // Gemma family (gemma, gemma2, gemma3) — uses gemma3 forward pass
    if lower.contains("gemma") {
        return match_size(&lower, "gemma3");
    }

    // LLaMA family
    if lower.contains("llama") {
        return match_size(&lower, "llama3");
    }

    "default".to_string()
}

/// Extract size bucket from model name for profile filename resolution.
///
/// Uses numeric parsing to avoid substring false positives
/// (e.g. "27b" must not match "7b", "14b" must not match "4b").
fn match_size(lower: &str, family: &str) -> String {
    if let Some(size) = extract_param_billions(lower) {
        family_size_bucket(family, size)
            .map(|bucket| format!("{family}-{bucket}"))
            .unwrap_or_else(|| family.to_string())
    } else {
        family.to_string()
    }
}

/// Parse the first `<number>b` or `<number>B` pattern from a model name string.
/// Returns the size in billions as f32.
fn extract_param_billions(name: &str) -> Option<f32> {
    let bytes = name.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Find a digit
        if bytes[i].is_ascii_digit() {
            let start = i;
            // Consume digits and optional decimal point
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                i += 1;
            }
            // Check for trailing 'b' or 'B'
            if i < bytes.len() && (bytes[i] == b'b' || bytes[i] == b'B') {
                // Make sure 'b' is not part of a longer word (e.g. "block")
                let after_b = i + 1;
                let is_boundary = after_b >= bytes.len() || !bytes[after_b].is_ascii_alphabetic();
                if is_boundary && let Ok(val) = name[start..i].parse::<f32>() {
                    return Some(val);
                }
            }
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_profile() {
        let profile = KernelProfile::default();
        assert_eq!(profile.source, "hardcoded");

        let q4k_matvec = profile.matvec_params("q4_k");
        assert_eq!(q4k_matvec.threadgroup_size, 128);
        let q5k_matvec = profile.matvec_params("q5_k");
        assert_eq!(q5k_matvec.threadgroup_size, 128);
        assert_eq!(q5k_matvec.rows_per_simdgroup, 1);
        assert!(!profile.batch_prefill.prefer_f16_io);
        assert!(profile.batch_prefill.prefer_pair_kernel);
        assert!(!profile.batch_prefill.use_bn32);
        assert_eq!(profile.attention_prefill.fa2_mode, ProfileKernelMode::Off);
    }

    #[test]
    fn test_sanitize_name() {
        assert_eq!(sanitize_name("Qwen3-8B"), "qwen3-8b");
        assert_eq!(sanitize_name("LLaMA 3.1 8B"), "llama-3-1-8b");
    }

    #[test]
    fn test_extract_param_billions() {
        assert_eq!(extract_param_billions("Qwen3.5-27B-Q4_K_M"), Some(27.0));
        assert_eq!(extract_param_billions("Qwen3-14B-Instruct"), Some(14.0));
        assert_eq!(extract_param_billions("Meta-Llama-3.2-1B"), Some(1.0));
        assert_eq!(extract_param_billions("some-unknown-model"), None);
    }

    #[test]
    fn test_extract_architecture() {
        // Core families (6B+ generative only)
        assert_eq!(extract_architecture("Qwen3-8B-Q4_K_M"), "qwen3-8b");
        assert_eq!(extract_architecture("Qwen2.5-7B-Instruct"), "qwen3-8b");
        assert_eq!(extract_architecture("Qwen3-14B"), "qwen3-14b");
        assert_eq!(extract_architecture("gemma-3-4b-it"), "gemma3-4b");
        assert_eq!(extract_architecture("gemma-3-12b-it"), "gemma3-12b");
        assert_eq!(extract_architecture("Meta-Llama-3-8B"), "llama3-8b");

        // Qwen 3.5 — separate hybrid family, distinct from qwen3 (6B+ only)
        assert_eq!(extract_architecture("Qwen3.5-7B-Instruct"), "qwen35-9b");
        assert_eq!(extract_architecture("Qwen3.5-8B"), "qwen35-9b");
        assert_eq!(extract_architecture("Qwen3.5-27B-Q4_K_M"), "qwen35-27b");

        // Unknown falls to default
        assert_eq!(extract_architecture("some-unknown-model"), "default");
    }

    #[test]
    fn test_fallback_matvec_params() {
        let profile = KernelProfile::default();
        let unknown = profile.matvec_params("unknown_quant");
        assert_eq!(unknown.threadgroup_size, 128);
    }

    #[test]
    fn test_profile_json_roundtrip() {
        let profile = KernelProfile::default();
        let json = serde_json::to_string(&profile).unwrap();
        let parsed: KernelProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(profile, parsed);
    }

    #[test]
    fn test_profile_json_rejects_unknown_fields() {
        let json = r#"{
            "model": "default",
            "source": "test",
            "generated": "2026-03-22",
            "decode_matvec": {},
            "attention_decode": {},
            "unexpected": true
        }"#;
        assert!(serde_json::from_str::<KernelProfile>(json).is_err());
    }

    #[test]
    fn test_profile_json_accepts_generated_and_prefill_fields() {
        let json = r#"{
            "model": "default",
            "source": "test",
            "generated": "2026-03-23",
            "decode_matvec": {
                "q4_k": {
                    "threadgroup_size": 64,
                    "rows_per_simdgroup": 2
                }
            },
            "batch_prefill": {
                "prefer_f16_io": true,
                "prefer_pair_kernel": true,
                "small_n_threshold": 32,
                "small_m_max": 4096,
                "use_bn32": false,
                "use_bk32": false,
                "q8_f16in_full_min_n": 128
            },
            "attention_decode": {
                "splitk_chunk_size": 256,
                "splitk_threshold": 512,
                "sdpa_default": true,
                "hd128_n2_default": false
            },
            "attention_prefill": {
                "fa2_mode": "auto",
                "fa2_hd128_mode": "off",
                "mistral_bc64_mode": "off",
                "fa2_auto_min_tokens": 640,
                "fa2_auto_min_base_seq": 320,
                "fa2_hd128_auto_min_tokens": 768,
                "mistral_bc64_min_tokens": 1024
            }
        }"#;
        let profile: KernelProfile = serde_json::from_str(json).unwrap();
        assert_eq!(profile.generated, "2026-03-23");
        assert!(profile.batch_prefill.prefer_f16_io); // JSON has true
        assert_eq!(profile.batch_prefill.small_n_threshold, 32);
        assert_eq!(profile.attention_decode.sdpa_default, Some(true));
        assert_eq!(profile.attention_decode.hd128_n2_default, Some(false));
        assert_eq!(profile.attention_prefill.fa2_mode, ProfileKernelMode::Auto);
        assert_eq!(
            profile.attention_prefill.mistral_bc64_mode,
            ProfileKernelMode::Off
        );
        assert_eq!(profile.attention_prefill.fa2_auto_min_base_seq, 320);
        assert_eq!(profile.attention_prefill.mistral_bc64_min_tokens, 1024);
    }

    #[test]
    fn test_profile_json_accepts_decode_regimes() {
        let json = r#"{
            "model": "qwen3-8b",
            "source": "test",
            "generated": "2026-03-27",
            "decode_matvec": {
                "q4_k": {
                    "threadgroup_size": 64,
                    "rows_per_simdgroup": 1
                }
            },
            "decode_regimes": {
                "short_max_attend_len": 384,
                "short": {
                    "attention_decode": {
                        "splitk_threshold": 1024
                    }
                },
                "long": {
                    "decode_matvec": {
                        "q4_k": {
                            "rows_per_simdgroup": 2
                        }
                    },
                    "attention_decode": {
                        "splitk_threshold": 256,
                        "hd128_n2_default": true
                    }
                }
            }
        }"#;
        let profile: KernelProfile = serde_json::from_str(json).unwrap();
        let regimes = profile.decode_regimes.as_ref().unwrap();
        assert_eq!(regimes.short_max_attend_len, 384);
        assert_eq!(
            regimes.long.decode_matvec["q4_k"].rows_per_simdgroup,
            Some(2)
        );
        assert_eq!(regimes.long.attention_decode.hd128_n2_default, Some(true));
    }

    #[test]
    fn test_effective_decode_profile_applies_short_regime_overrides() {
        let profile: KernelProfile = serde_json::from_str(
            r#"{
                "model": "qwen3-8b",
                "source": "test",
                "generated": "2026-03-27",
                "decode_matvec": {
                    "q4_k": {
                        "threadgroup_size": 64,
                        "rows_per_simdgroup": 1
                    }
                },
                "attention_decode": {
                    "splitk_chunk_size": 256,
                    "splitk_threshold": 512
                },
                "decode_regimes": {
                    "short_max_attend_len": 384,
                    "short": {
                        "attention_decode": {
                            "splitk_threshold": 1024
                        }
                    },
                    "long": {
                        "decode_matvec": {
                            "q4_k": {
                                "rows_per_simdgroup": 2
                            }
                        },
                        "attention_decode": {
                            "splitk_threshold": 256,
                            "hd128_n2_default": true
                        }
                    }
                }
            }"#,
        )
        .unwrap();

        let short = profile.effective_decode_profile(128);
        assert_eq!(short.attention_decode.splitk_threshold, 1024);
        assert_eq!(short.attention_decode.hd128_n2_default, None);
        assert_eq!(short.matvec_params("q4_k").rows_per_simdgroup, 1);
        assert!(short.decode_regimes.is_none());
    }

    #[test]
    fn test_effective_decode_profile_applies_long_regime_overrides() {
        let profile: KernelProfile = serde_json::from_str(
            r#"{
                "model": "qwen35-9b",
                "source": "test",
                "generated": "2026-03-27",
                "decode_matvec": {
                    "q4_k": {
                        "threadgroup_size": 64,
                        "rows_per_simdgroup": 1
                    }
                },
                "attention_decode": {
                    "splitk_chunk_size": 256,
                    "splitk_threshold": 512
                },
                "decode_regimes": {
                    "short_max_attend_len": 384,
                    "long": {
                        "decode_matvec": {
                            "q4_k": {
                                "rows_per_simdgroup": 2
                            }
                        },
                        "attention_decode": {
                            "splitk_threshold": 256,
                            "sdpa_default": true
                        }
                    }
                }
            }"#,
        )
        .unwrap();

        let long = profile.effective_decode_profile(2048);
        assert_eq!(long.attention_decode.splitk_threshold, 256);
        assert_eq!(long.attention_decode.sdpa_default, Some(true));
        assert_eq!(long.matvec_params("q4_k").rows_per_simdgroup, 2);
        assert!(long.decode_regimes.is_none());
    }

    #[test]
    fn test_extended_family_profiles_resolve_and_load() {
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap();
        let cases = [
            ("Qwen3-8B-Instruct", "qwen3-8b", "qwen3-8b"),
            ("Qwen3-14B-Instruct", "qwen3-14b", "qwen3-14b"),
            ("Qwen3.5-7B-Instruct", "qwen35-9b", "qwen35-9b"),
            ("Qwen3.5-27B-Instruct", "qwen35-27b", "qwen35-27b"),
            ("gemma-3-4b-it", "gemma3-4b", "gemma3-4b"),
            ("Meta-Llama-3-8B-Instruct", "llama3-8b", "llama3-8b"),
        ];

        for (model_name, expected_arch, expected_profile_model) in cases {
            let arch = extract_architecture(model_name);
            assert_eq!(arch, expected_arch);

            let profile_path = workspace_root.join("perfs").join(format!("{arch}.json"));
            let profile = KernelProfile::load_from_path(&profile_path)
                .unwrap_or_else(|| panic!("expected profile file for {}", profile_path.display()));
            assert_eq!(profile.model, expected_profile_model);
            assert!(!profile.source.is_empty());

            let q4k = profile.matvec_params("q4_k");
            assert_eq!(q4k.threadgroup_size, 64);
            assert_eq!(q4k.rows_per_simdgroup, 2);

            let q6k = profile.matvec_params("q6_k");
            assert_eq!(q6k.threadgroup_size, 64);
            assert_eq!(q6k.rows_per_simdgroup, 2);

            let q5k = profile.matvec_params("q5_k");
            assert_eq!(q5k.threadgroup_size, 64);
            assert_eq!(q5k.rows_per_simdgroup, 1);
        }
    }

    #[test]
    fn test_missing_arch_profiles_fall_back_to_default() {
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap();
        let cases = [("Meta-Llama-3.2-1B-Instruct", "llama3")];

        for (model_name, arch) in cases {
            let arch_path = workspace_root.join("perfs").join(format!("{arch}.json"));
            assert!(
                !arch_path.exists(),
                "expected redundant profile to be removed: {}",
                arch_path.display()
            );

            let profile = KernelProfile::load_relative_to(&workspace_root, model_name, "q4_k_m");
            assert_eq!(profile.model, "default");
            assert_eq!(profile.source, "llama.cpp-params-2026-03-22");

            let q4k = profile.matvec_params("q4_k");
            assert_eq!(q4k.threadgroup_size, 64);
            assert_eq!(q4k.rows_per_simdgroup, 2);

            let q5k = profile.matvec_params("q5_k");
            assert_eq!(q5k.threadgroup_size, 64);
            assert_eq!(q5k.rows_per_simdgroup, 1);
        }
    }

    #[test]
    fn test_all_perf_profiles_load() {
        let perfs_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("perfs")
            .canonicalize()
            .unwrap();
        let mut profile_count = 0usize;
        for entry in std::fs::read_dir(&perfs_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            let profile = KernelProfile::load_from_path(&path)
                .unwrap_or_else(|| panic!("expected profile file for {}", path.display()));
            assert!(
                !profile.source.is_empty(),
                "missing source in {}",
                path.display()
            );
            let q5k = profile.matvec_params("q5_k");
            assert_eq!(
                q5k.threadgroup_size,
                64,
                "unexpected q5_k tg size in {}",
                path.display()
            );
            assert_eq!(
                q5k.rows_per_simdgroup,
                1,
                "unexpected q5_k rows_per_simdgroup in {}",
                path.display()
            );
            profile_count += 1;
        }
        assert!(profile_count > 0, "expected at least one profile in perfs/");
    }
}
