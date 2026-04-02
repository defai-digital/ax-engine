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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct MatvecParamsOverride {
    #[serde(default)]
    pub threadgroup_size: Option<u32>,
    #[serde(default)]
    pub rows_per_simdgroup: Option<u32>,
    #[serde(default)]
    pub variant: Option<MatvecProfileVariant>,
}

impl MatvecParamsOverride {
    fn apply_to(&self, base: MatvecParams) -> MatvecParams {
        MatvecParams {
            threadgroup_size: self.threadgroup_size.unwrap_or(base.threadgroup_size),
            rows_per_simdgroup: self.rows_per_simdgroup.unwrap_or(base.rows_per_simdgroup),
            variant: self.variant.or(base.variant),
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
    #[serde(default)]
    pub decode_regimes: Option<DecodeRegimeProfile>,
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
            decode_regimes: None,
        }
    }
}

impl KernelProfile {
    pub fn load(model_name: &str, quant: &str) -> Self {
        Self::load_relative_to(Path::new("."), model_name, quant)
    }

    fn load_relative_to(base_dir: &Path, model_name: &str, quant: &str) -> Self {
        if let Some(profile) = Self::try_load_from_env(model_name) {
            return profile;
        }

        let sanitized_model = sanitize_name(model_name);
        let sanitized_quant = sanitize_name(quant);

        if let Some(profile) =
            Self::try_load_exact(base_dir, &sanitized_model, &sanitized_quant, model_name)
        {
            return profile;
        }

        let arch = extract_architecture(model_name);
        if let Some(profile) = Self::try_load_arch(base_dir, &arch, model_name) {
            return profile;
        }

        if let Some(profile) = Self::try_load_default(base_dir, model_name) {
            return profile;
        }

        Self::default().apply_missing_model_heuristics(
            model_name, true, true, true, true, true, true, true, true, true, true,
        )
    }

    fn try_load_from_env(model_name: &str) -> Option<Self> {
        let path = std::env::var("AX_KERNEL_PROFILE_PATH")
            .ok()
            .filter(|s| !s.is_empty())?;
        Self::load_from_path_with_model_heuristics(&path, model_name)
    }

    fn try_load_exact(base_dir: &Path, model: &str, quant: &str, model_name: &str) -> Option<Self> {
        let filename = base_dir.join("perfs").join(format!("{model}-{quant}.json"));
        Self::load_from_path_with_model_heuristics(&filename, model_name)
    }

    fn try_load_arch(base_dir: &Path, arch: &str, model_name: &str) -> Option<Self> {
        let filename = base_dir.join("perfs").join(format!("{arch}.json"));
        Self::load_from_path_with_model_heuristics(&filename, model_name)
    }

    fn try_load_default(base_dir: &Path, model_name: &str) -> Option<Self> {
        Self::load_from_path_with_model_heuristics(
            base_dir.join("perfs").join("default.json"),
            model_name,
        )
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

    fn load_from_path_with_model_heuristics<P: AsRef<Path>>(
        path: P,
        fallback_model_name: &str,
    ) -> Option<Self> {
        let path = path.as_ref();
        let content = match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(e) => {
                tracing::debug!(path = %path.display(), error = %e, "Kernel profile not found");
                return None;
            }
        };
        let raw_json: serde_json::Value = match serde_json::from_str(&content) {
            Ok(value) => value,
            Err(e) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "Failed to parse kernel profile JSON"
                );
                return None;
            }
        };
        let profile: KernelProfile = match serde_json::from_value(raw_json.clone()) {
            Ok(profile) => profile,
            Err(e) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "Failed to decode kernel profile JSON"
                );
                return None;
            }
        };
        let effective_model_name = if profile.model.is_empty() || profile.model == "default" {
            fallback_model_name.to_string()
        } else {
            profile.model.clone()
        };
        let with_heuristics = profile.apply_missing_model_heuristics(
            &effective_model_name,
            !json_has_path(&raw_json, "attention_decode.splitk_threshold"),
            !json_has_path(&raw_json, "batch_prefill.prefer_f16_io"),
            !json_has_path(&raw_json, "decode_regimes.short_max_attend_len"),
            !json_has_path(
                &raw_json,
                "decode_regimes.long.attention_decode.splitk_threshold",
            ),
            !json_has_path(&raw_json, "attention_prefill.fa2_mode"),
            !json_has_path(&raw_json, "attention_prefill.fa2_hd128_mode"),
            !json_has_path(&raw_json, "attention_prefill.ax_bc64_mode"),
            !json_has_path(&raw_json, "attention_decode.hd128_n2_default"),
            !json_has_path(&raw_json, "attention_prefill.fa2_hd128_auto_min_tokens"),
            !json_has_path(&raw_json, "attention_prefill.ax_bc64_min_tokens"),
        );
        tracing::info!(
            path = %path.display(),
            model = %with_heuristics.model,
            source = %with_heuristics.source,
            "Loaded kernel profile"
        );
        Some(with_heuristics)
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

    #[allow(clippy::too_many_arguments)]
    fn apply_missing_model_heuristics(
        mut self,
        model_name: &str,
        missing_splitk_threshold: bool,
        missing_prefer_f16_io: bool,
        missing_decode_regime_short_max_attend_len: bool,
        missing_decode_regime_long_splitk_threshold: bool,
        missing_fa2_mode: bool,
        missing_fa2_hd128_mode: bool,
        missing_ax_bc64_mode: bool,
        missing_hd128_n2_default: bool,
        missing_fa2_hd128_auto_min_tokens: bool,
        missing_ax_bc64_min_tokens: bool,
    ) -> Self {
        let arch = extract_architecture(model_name);
        if missing_splitk_threshold {
            self.attention_decode.splitk_threshold =
                heuristic_decode_splitk_threshold_for_arch(&arch);
        }
        if missing_prefer_f16_io {
            self.batch_prefill.prefer_f16_io =
                heuristic_batch_prefill_prefer_f16_io_for_arch(&arch);
        }
        if missing_fa2_mode {
            self.attention_prefill.fa2_mode = heuristic_attention_prefill_fa2_mode_for_arch(&arch);
        }
        if missing_fa2_hd128_mode {
            self.attention_prefill.fa2_hd128_mode =
                heuristic_attention_prefill_fa2_hd128_mode_for_arch(&arch);
        }
        if missing_ax_bc64_mode {
            self.attention_prefill.ax_bc64_mode =
                heuristic_attention_prefill_ax_bc64_mode_for_arch(&arch);
        }
        if missing_hd128_n2_default {
            self.attention_decode.hd128_n2_default =
                heuristic_attention_decode_hd128_n2_default_for_arch(&arch);
        }
        if missing_fa2_hd128_auto_min_tokens {
            self.attention_prefill.fa2_hd128_auto_min_tokens =
                heuristic_attention_prefill_fa2_hd128_auto_min_tokens_for_arch(&arch);
        }
        if missing_ax_bc64_min_tokens {
            self.attention_prefill.ax_bc64_min_tokens =
                heuristic_attention_prefill_ax_bc64_min_tokens_for_arch(&arch);
        }
        let decode_regime_short_max_attend_len = if missing_decode_regime_short_max_attend_len {
            heuristic_decode_regime_short_max_attend_len_for_arch(&arch)
        } else {
            None
        };
        let decode_regime_long_splitk_threshold = if missing_decode_regime_long_splitk_threshold {
            heuristic_decode_regime_long_splitk_threshold_for_arch(&arch)
        } else {
            None
        };
        if decode_regime_short_max_attend_len.is_some()
            || decode_regime_long_splitk_threshold.is_some()
        {
            let decode_regimes = self
                .decode_regimes
                .get_or_insert_with(DecodeRegimeProfile::default);
            if let Some(short_max_attend_len) = decode_regime_short_max_attend_len {
                decode_regimes.short_max_attend_len = short_max_attend_len;
            }
            if let Some(splitk_threshold) = decode_regime_long_splitk_threshold {
                decode_regimes.long.attention_decode.splitk_threshold = Some(splitk_threshold);
            }
        }
        self
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
            // Generative models: 4B+ only
            if approx_size(size, 4.0) {
                Some("4b")
            } else if approx_size(size, 6.0) || approx_size(size, 7.0) || approx_size(size, 8.0) {
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
            if approx_size(size, 4.0) {
                Some("4b")
            } else if approx_size(size, 7.0) || approx_size(size, 8.0) || approx_size(size, 9.0) {
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

fn heuristic_decode_splitk_threshold_for_arch(arch: &str) -> u32 {
    match arch {
        "gemma3-4b" | "gemma3-12b" | "qwen35-4b" => 256,
        "default" | "qwen3-14b" | "qwen35-9b" => 512,
        "gemma3-27b" | "llama3-8b" | "llama3-70b" | "qwen3-8b" | "qwen3-32b" | "qwen35-27b" => 1024,
        _ => 512,
    }
}

fn heuristic_batch_prefill_prefer_f16_io_for_arch(_arch: &str) -> bool {
    // PRD-PREFILL-DISPATCH-CONSOLIDATION-2026-03-31:
    // All architectures now use the blocked kernel (identical to llama.cpp's
    // kernel_mul_mm) which does inline float→half cast in the B-loading phase.
    // The separate f32→f16 cast + f16in kernel path added 224 extra dispatches
    // per prefill and used a suboptimal tile geometry (BM=32/BN=64 vs
    // llama.cpp's BM=64/BN=32). Override per-arch with AX_METAL_BATCH_F16_IO=1.
    false
}

fn heuristic_decode_regime_short_max_attend_len_for_arch(arch: &str) -> Option<u32> {
    match arch {
        "qwen3-8b" | "qwen3-14b" | "qwen35-9b" => Some(384),
        "qwen3-32b" | "qwen35-27b" => Some(512),
        _ => None,
    }
}

fn heuristic_decode_regime_long_splitk_threshold_for_arch(arch: &str) -> Option<u32> {
    match arch {
        "qwen3-8b" | "qwen3-14b" | "qwen35-9b" => Some(256),
        "qwen3-32b" | "qwen35-27b" => Some(512),
        _ => None,
    }
}

fn heuristic_attention_prefill_fa2_mode_for_arch(arch: &str) -> ProfileKernelMode {
    // PRD-FA2-HALF-PRECISION-ATTENTION-2026-03-31: benchmarks show FA2 simd
    // is 7-24% faster than Mistral at P≥512 on Qwen3 8B (758 vs 700 tok/s
    // at P=512, 748 vs 603 at P=1024). Enable for all HD=128 models.
    // HD=256 models (Gemma3) are unaffected — they don't match FA2 HD=128.
    match arch {
        // Gemma3 uses HD=256, FA2 HD128 doesn't apply.
        "gemma3-4b" | "gemma3-12b" | "gemma3-27b" => ProfileKernelMode::Off,
        _ => ProfileKernelMode::Auto,
    }
}

fn heuristic_attention_prefill_fa2_hd128_mode_for_arch(arch: &str) -> ProfileKernelMode {
    match arch {
        // Llama3 8B now keeps FA2 available behind a long-prompt threshold.
        "llama3-8b" => ProfileKernelMode::Auto,
        _ => ProfileKernelMode::Auto,
    }
}

fn heuristic_attention_prefill_fa2_hd128_auto_min_tokens_for_arch(arch: &str) -> u32 {
    match arch {
        // Bench sweep 2026-04-01: Llama3 8B keeps BC64 at medium prompts but
        // FA2 wins again at 1024 prompt tokens.
        "llama3-8b" => 1024,
        // Bench sweep 2026-04-01: small 4B HD128 models prefer BC64 at 256,
        // then FA2 from 384+.
        "qwen3-4b" | "qwen35-4b" => 384,
        _ => default_attention_prefill_fa2_hd128_auto_min_tokens(),
    }
}

fn heuristic_attention_prefill_ax_bc64_mode_for_arch(_arch: &str) -> ProfileKernelMode {
    // With FA2 now preferred for HD=128 models, Mistral bc64 is Auto for all.
    ProfileKernelMode::Auto
}

fn heuristic_attention_decode_hd128_n2_default_for_arch(arch: &str) -> Option<bool> {
    match arch {
        "qwen3-8b" | "qwen35-4b" | "qwen35-9b" | "qwen35-27b" => Some(true),
        _ => None,
    }
}

fn heuristic_attention_prefill_ax_bc64_min_tokens_for_arch(arch: &str) -> u32 {
    match arch {
        "qwen3-4b" | "qwen35-4b" => 256,
        "llama3-8b" | "qwen3-8b" | "qwen3-14b" | "qwen35-9b" | "qwen35-27b" => 384,
        _ => default_attention_prefill_ax_bc64_min_tokens(),
    }
}

fn json_has_path(value: &serde_json::Value, flat_path: &str) -> bool {
    let mut current = value;
    for part in flat_path.split('.') {
        let Some(next) = current.get(part) else {
            return false;
        };
        current = next;
    }
    true
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
    use std::fs;
    use std::path::PathBuf;

    fn perf_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("perfs")
            .canonicalize()
            .unwrap()
    }

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
        // Core families
        assert_eq!(extract_architecture("Qwen3-4B-Q6_K"), "qwen3-4b");
        assert_eq!(extract_architecture("Qwen3-8B-Q4_K_M"), "qwen3-8b");
        assert_eq!(extract_architecture("Qwen2.5-7B-Instruct"), "qwen3-8b");
        assert_eq!(extract_architecture("Qwen3-14B"), "qwen3-14b");
        assert_eq!(extract_architecture("gemma-3-4b-it"), "gemma3-4b");
        assert_eq!(extract_architecture("gemma-3-12b-it"), "gemma3-12b");
        assert_eq!(extract_architecture("Meta-Llama-3-8B"), "llama3-8b");

        // Qwen 3.5 — separate hybrid family, distinct from qwen3
        assert_eq!(extract_architecture("Qwen3.5-4B-Instruct"), "qwen35-4b");
        assert_eq!(extract_architecture("Qwen3.5-7B-Instruct"), "qwen35-9b");
        assert_eq!(extract_architecture("Qwen3.5-8B"), "qwen35-9b");
        assert_eq!(extract_architecture("Qwen3.5-27B-Q4_K_M"), "qwen35-27b");

        // Unknown falls to default
        assert_eq!(extract_architecture("some-unknown-model"), "default");
    }

    #[test]
    fn test_heuristic_decode_splitk_threshold_matches_current_profile_buckets() {
        let cases = [
            ("default", 512),
            ("gemma3-4b", 256),
            ("gemma3-12b", 256),
            ("gemma3-27b", 1024),
            ("llama3-8b", 1024),
            ("llama3-70b", 1024),
            ("qwen3-8b", 1024),
            ("qwen3-14b", 512),
            ("qwen3-32b", 1024),
            ("qwen35-4b", 256),
            ("qwen35-9b", 512),
            ("qwen35-27b", 1024),
        ];
        for (arch, expected) in cases {
            assert_eq!(heuristic_decode_splitk_threshold_for_arch(arch), expected);
        }
    }

    #[test]
    fn test_heuristic_batch_prefill_prefer_f16_io_matches_current_profile_buckets() {
        // PRD-PREFILL-DISPATCH-CONSOLIDATION-2026-03-31: all archs use blocked
        // kernel (identical to llama.cpp kernel_mul_mm) with inline float→half.
        let cases = [
            ("default", false),
            ("gemma3-4b", false),
            ("gemma3-12b", false),
            ("gemma3-27b", false),
            ("llama3-8b", false),
            ("llama3-70b", false),
            ("qwen3-8b", false),
            ("qwen3-14b", false),
            ("qwen3-32b", false),
            ("qwen35-4b", false),
            ("qwen35-9b", false),
            ("qwen35-27b", false),
        ];
        for (arch, expected) in cases {
            assert_eq!(
                heuristic_batch_prefill_prefer_f16_io_for_arch(arch),
                expected
            );
        }
    }

    #[test]
    fn test_heuristic_attention_prefill_fa2_mode_matches_current_profile_buckets() {
        // FA2 simd is faster than Mistral at P≥512 for HD=128 models.
        // Off only for Gemma3 (HD=256, FA2 HD128 doesn't apply).
        let cases = [
            ("default", ProfileKernelMode::Auto),
            ("qwen3-4b", ProfileKernelMode::Auto),
            ("gemma3-4b", ProfileKernelMode::Off),
            ("gemma3-12b", ProfileKernelMode::Off),
            ("gemma3-27b", ProfileKernelMode::Off),
            ("llama3-8b", ProfileKernelMode::Auto),
            ("llama3-70b", ProfileKernelMode::Auto),
            ("qwen3-8b", ProfileKernelMode::Auto),
            ("qwen3-14b", ProfileKernelMode::Auto),
            ("qwen3-32b", ProfileKernelMode::Auto),
            ("qwen35-4b", ProfileKernelMode::Auto),
            ("qwen35-9b", ProfileKernelMode::Auto),
            ("qwen35-27b", ProfileKernelMode::Auto),
        ];
        for (arch, expected) in cases {
            assert_eq!(
                heuristic_attention_prefill_fa2_mode_for_arch(arch),
                expected
            );
        }
    }

    #[test]
    fn test_heuristic_attention_prefill_fa2_hd128_mode_matches_current_profile_buckets() {
        let cases = [
            ("default", ProfileKernelMode::Auto),
            ("qwen3-4b", ProfileKernelMode::Auto),
            ("gemma3-4b", ProfileKernelMode::Auto),
            ("gemma3-12b", ProfileKernelMode::Auto),
            ("gemma3-27b", ProfileKernelMode::Auto),
            ("llama3-8b", ProfileKernelMode::Auto),
            ("llama3-70b", ProfileKernelMode::Auto),
            ("qwen3-8b", ProfileKernelMode::Auto),
            ("qwen3-14b", ProfileKernelMode::Auto),
            ("qwen3-32b", ProfileKernelMode::Auto),
            ("qwen35-4b", ProfileKernelMode::Auto),
            ("qwen35-9b", ProfileKernelMode::Auto),
            ("qwen35-27b", ProfileKernelMode::Auto),
        ];
        for (arch, expected) in cases {
            assert_eq!(
                heuristic_attention_prefill_fa2_hd128_mode_for_arch(arch),
                expected
            );
        }
    }

    #[test]
    fn test_heuristic_attention_prefill_ax_bc64_mode_matches_current_profile_buckets() {
        // With FA2 now the preferred HD=128 path, Mistral bc64 is Auto for all.
        let cases = [
            ("default", ProfileKernelMode::Auto),
            ("qwen3-4b", ProfileKernelMode::Auto),
            ("gemma3-4b", ProfileKernelMode::Auto),
            ("gemma3-12b", ProfileKernelMode::Auto),
            ("gemma3-27b", ProfileKernelMode::Auto),
            ("llama3-8b", ProfileKernelMode::Auto),
            ("llama3-70b", ProfileKernelMode::Auto),
            ("qwen3-8b", ProfileKernelMode::Auto),
            ("qwen3-14b", ProfileKernelMode::Auto),
            ("qwen3-32b", ProfileKernelMode::Auto),
            ("qwen35-4b", ProfileKernelMode::Auto),
            ("qwen35-9b", ProfileKernelMode::Auto),
            ("qwen35-27b", ProfileKernelMode::Auto),
        ];
        for (arch, expected) in cases {
            assert_eq!(
                heuristic_attention_prefill_ax_bc64_mode_for_arch(arch),
                expected
            );
        }
    }

    #[test]
    fn test_heuristic_attention_decode_hd128_n2_default_matches_current_profile_buckets() {
        let cases = [
            ("default", None),
            ("gemma3-4b", None),
            ("gemma3-12b", None),
            ("gemma3-27b", None),
            ("llama3-8b", None),
            ("llama3-70b", None),
            ("qwen3-8b", Some(true)),
            ("qwen3-14b", None),
            ("qwen3-32b", None),
            ("qwen35-4b", Some(true)),
            ("qwen35-9b", Some(true)),
            ("qwen35-27b", Some(true)),
        ];
        for (arch, expected) in cases {
            assert_eq!(
                heuristic_attention_decode_hd128_n2_default_for_arch(arch),
                expected
            );
        }
    }

    #[test]
    fn test_heuristic_attention_prefill_ax_bc64_min_tokens_matches_current_profile_buckets() {
        let cases = [
            ("default", 384),
            ("qwen3-4b", 256),
            ("gemma3-4b", 384),
            ("gemma3-12b", 384),
            ("gemma3-27b", 384),
            ("llama3-8b", 384),
            ("llama3-70b", 384),
            ("qwen3-8b", 384),
            ("qwen3-14b", 384),
            ("qwen3-32b", 384),
            ("qwen35-4b", 256),
            ("qwen35-9b", 384),
            ("qwen35-27b", 384),
        ];
        for (arch, expected) in cases {
            assert_eq!(
                heuristic_attention_prefill_ax_bc64_min_tokens_for_arch(arch),
                expected
            );
        }
    }

    #[test]
    fn test_heuristic_attention_prefill_fa2_hd128_auto_min_tokens_matches_current_profile_buckets()
    {
        let cases = [
            ("default", 128),
            ("qwen3-4b", 384),
            ("gemma3-4b", 128),
            ("gemma3-12b", 128),
            ("gemma3-27b", 128),
            ("llama3-8b", 1024),
            ("llama3-70b", 128),
            ("qwen3-8b", 128),
            ("qwen3-14b", 128),
            ("qwen3-32b", 128),
            ("qwen35-4b", 384),
            ("qwen35-9b", 128),
            ("qwen35-27b", 128),
        ];
        for (arch, expected) in cases {
            assert_eq!(
                heuristic_attention_prefill_fa2_hd128_auto_min_tokens_for_arch(arch),
                expected
            );
        }
    }

    #[test]
    fn test_heuristic_decode_regime_short_max_attend_len_matches_current_profile_buckets() {
        let cases = [
            ("default", None),
            ("gemma3-4b", None),
            ("gemma3-12b", None),
            ("gemma3-27b", None),
            ("llama3-8b", None),
            ("llama3-70b", None),
            ("qwen3-8b", Some(384)),
            ("qwen3-14b", Some(384)),
            ("qwen3-32b", Some(512)),
            ("qwen35-4b", None),
            ("qwen35-9b", Some(384)),
            ("qwen35-27b", Some(512)),
        ];
        for (arch, expected) in cases {
            assert_eq!(
                heuristic_decode_regime_short_max_attend_len_for_arch(arch),
                expected
            );
        }
    }

    #[test]
    fn test_heuristic_decode_regime_long_splitk_threshold_matches_current_profile_buckets() {
        let cases = [
            ("default", None),
            ("gemma3-4b", None),
            ("gemma3-12b", None),
            ("gemma3-27b", None),
            ("llama3-8b", None),
            ("llama3-70b", None),
            ("qwen3-8b", Some(256)),
            ("qwen3-14b", Some(256)),
            ("qwen3-32b", Some(512)),
            ("qwen35-4b", None),
            ("qwen35-9b", Some(256)),
            ("qwen35-27b", Some(512)),
        ];
        for (arch, expected) in cases {
            assert_eq!(
                heuristic_decode_regime_long_splitk_threshold_for_arch(arch),
                expected
            );
        }
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

    #[test]
    fn test_current_profiles_match_hardcoded_matvec_invariants() {
        for entry in fs::read_dir(perf_dir()).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let profile = KernelProfile::load_from_path(&path).unwrap();
            for quant in ["q4_k", "q5_k", "q6_k", "q8_0"] {
                let expected = hardcoded_invariant_matvec_params(quant).unwrap();
                let actual = profile.matvec_params(quant);
                assert_eq!(
                    actual,
                    expected,
                    "profile {} drifted from hardcoded invariant for {quant}",
                    path.display()
                );
            }
        }
    }

    #[test]
    fn test_current_profiles_match_non_matvec_invariant_defaults() {
        let default_profile = KernelProfile::default();
        for entry in fs::read_dir(perf_dir()).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let profile = KernelProfile::load_from_path(&path).unwrap();
            assert_eq!(
                profile.batch_prefill.prefer_pair_kernel,
                default_profile.batch_prefill.prefer_pair_kernel,
                "profile {} drifted on batch_prefill.prefer_pair_kernel",
                path.display()
            );
            assert_eq!(
                profile.batch_prefill.use_bk32,
                default_profile.batch_prefill.use_bk32,
                "profile {} drifted on batch_prefill.use_bk32",
                path.display()
            );
            assert_eq!(
                profile.batch_prefill.use_bn32,
                default_profile.batch_prefill.use_bn32,
                "profile {} drifted on batch_prefill.use_bn32",
                path.display()
            );
            assert_eq!(
                profile.batch_prefill.q8_f16in_full_min_n,
                default_profile.batch_prefill.q8_f16in_full_min_n,
                "profile {} drifted on batch_prefill.q8_f16in_full_min_n",
                path.display()
            );
            assert_eq!(
                profile.batch_prefill.small_n_threshold,
                default_profile.batch_prefill.small_n_threshold,
                "profile {} drifted on batch_prefill.small_n_threshold",
                path.display()
            );
            assert_eq!(
                profile.batch_prefill.small_m_max,
                default_profile.batch_prefill.small_m_max,
                "profile {} drifted on batch_prefill.small_m_max",
                path.display()
            );
            assert_eq!(
                profile.attention_decode.splitk_chunk_size,
                default_profile.attention_decode.splitk_chunk_size,
                "profile {} drifted on attention_decode.splitk_chunk_size",
                path.display()
            );
            assert_eq!(
                profile.attention_prefill.fa2_auto_min_tokens,
                default_profile.attention_prefill.fa2_auto_min_tokens,
                "profile {} drifted on attention_prefill.fa2_auto_min_tokens",
                path.display()
            );
            assert_eq!(
                profile.attention_prefill.fa2_auto_min_base_seq,
                default_profile.attention_prefill.fa2_auto_min_base_seq,
                "profile {} drifted on attention_prefill.fa2_auto_min_base_seq",
                path.display()
            );
            assert_eq!(
                profile.attention_prefill.fa2_hd128_auto_min_tokens,
                default_profile.attention_prefill.fa2_hd128_auto_min_tokens,
                "profile {} drifted on attention_prefill.fa2_hd128_auto_min_tokens",
                path.display()
            );
            assert_eq!(
                profile.attention_prefill.fa2_hd128_mode,
                default_profile.attention_prefill.fa2_hd128_mode,
                "profile {} drifted on attention_prefill.fa2_hd128_mode",
                path.display()
            );
            assert_eq!(
                profile.kv_cache.precision,
                default_profile.kv_cache.precision,
                "profile {} drifted on kv_cache.precision",
                path.display()
            );
        }
    }

    #[test]
    fn test_current_profiles_match_splitk_threshold_heuristic() {
        for entry in fs::read_dir(perf_dir()).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let raw_profile = KernelProfile::load_from_path(&path).unwrap();
            let model_name = raw_profile.model.clone();
            let effective_profile =
                KernelProfile::load_from_path_with_model_heuristics(&path, &model_name).unwrap();
            let expected =
                heuristic_decode_splitk_threshold_for_arch(&extract_architecture(&model_name));
            assert_eq!(
                effective_profile.attention_decode.splitk_threshold,
                expected,
                "profile {} drifted from splitk threshold heuristic for model {}",
                path.display(),
                model_name
            );
        }
    }

    #[test]
    fn test_current_profiles_match_prefer_f16_io_heuristic() {
        for entry in fs::read_dir(perf_dir()).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let raw_profile = KernelProfile::load_from_path(&path).unwrap();
            let model_name = raw_profile.model.clone();
            let effective_profile =
                KernelProfile::load_from_path_with_model_heuristics(&path, &model_name).unwrap();
            let expected =
                heuristic_batch_prefill_prefer_f16_io_for_arch(&extract_architecture(&model_name));
            assert_eq!(
                effective_profile.batch_prefill.prefer_f16_io,
                expected,
                "profile {} drifted from prefer_f16_io heuristic for model {}",
                path.display(),
                model_name
            );
        }
    }

    #[test]
    fn test_current_profiles_match_decode_regime_short_max_attend_len_heuristic() {
        for entry in fs::read_dir(perf_dir()).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let raw_profile = KernelProfile::load_from_path(&path).unwrap();
            let model_name = raw_profile.model.clone();
            let effective_profile =
                KernelProfile::load_from_path_with_model_heuristics(&path, &model_name).unwrap();
            let expected = heuristic_decode_regime_short_max_attend_len_for_arch(
                &extract_architecture(&model_name),
            );
            assert_eq!(
                effective_profile
                    .decode_regimes
                    .as_ref()
                    .map(|regimes| regimes.short_max_attend_len),
                expected,
                "profile {} drifted from decode_regimes.short_max_attend_len heuristic for model {}",
                path.display(),
                model_name
            );
        }
    }

    #[test]
    fn test_current_profiles_match_decode_regime_long_splitk_threshold_heuristic() {
        for entry in fs::read_dir(perf_dir()).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let raw_profile = KernelProfile::load_from_path(&path).unwrap();
            let model_name = raw_profile.model.clone();
            let effective_profile =
                KernelProfile::load_from_path_with_model_heuristics(&path, &model_name).unwrap();
            let expected = heuristic_decode_regime_long_splitk_threshold_for_arch(
                &extract_architecture(&model_name),
            );
            assert_eq!(
                effective_profile
                    .decode_regimes
                    .as_ref()
                    .and_then(|regimes| { regimes.long.attention_decode.splitk_threshold }),
                expected,
                "profile {} drifted from decode_regimes.long.attention_decode.splitk_threshold heuristic for model {}",
                path.display(),
                model_name
            );
        }
    }

    #[test]
    fn test_current_profiles_match_attention_prefill_fa2_mode_heuristic() {
        for entry in fs::read_dir(perf_dir()).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let raw_profile = KernelProfile::load_from_path(&path).unwrap();
            let model_name = raw_profile.model.clone();
            let effective_profile =
                KernelProfile::load_from_path_with_model_heuristics(&path, &model_name).unwrap();
            let expected =
                heuristic_attention_prefill_fa2_mode_for_arch(&extract_architecture(&model_name));
            assert_eq!(
                effective_profile.attention_prefill.fa2_mode,
                expected,
                "profile {} drifted from attention_prefill.fa2_mode heuristic for model {}",
                path.display(),
                model_name
            );
        }
    }

    #[test]
    fn test_current_profiles_match_attention_prefill_fa2_hd128_mode_heuristic() {
        for entry in fs::read_dir(perf_dir()).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let raw_profile = KernelProfile::load_from_path(&path).unwrap();
            let model_name = raw_profile.model.clone();
            let effective_profile =
                KernelProfile::load_from_path_with_model_heuristics(&path, &model_name).unwrap();
            let expected = heuristic_attention_prefill_fa2_hd128_mode_for_arch(
                &extract_architecture(&model_name),
            );
            assert_eq!(
                effective_profile.attention_prefill.fa2_hd128_mode,
                expected,
                "profile {} drifted from attention_prefill.fa2_hd128_mode heuristic for model {}",
                path.display(),
                model_name
            );
        }
    }

    #[test]
    fn test_current_profiles_match_attention_prefill_fa2_hd128_auto_min_tokens_heuristic() {
        for entry in fs::read_dir(perf_dir()).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let raw_profile = KernelProfile::load_from_path(&path).unwrap();
            let model_name = raw_profile.model.clone();
            let effective_profile =
                KernelProfile::load_from_path_with_model_heuristics(&path, &model_name).unwrap();
            let expected = heuristic_attention_prefill_fa2_hd128_auto_min_tokens_for_arch(
                &extract_architecture(&model_name),
            );
            assert_eq!(
                effective_profile
                    .attention_prefill
                    .fa2_hd128_auto_min_tokens,
                expected,
                "profile {} drifted from attention_prefill.fa2_hd128_auto_min_tokens heuristic for model {}",
                path.display(),
                model_name
            );
        }
    }

    #[test]
    fn test_current_profiles_match_attention_prefill_ax_bc64_mode_heuristic() {
        for entry in fs::read_dir(perf_dir()).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let raw_profile = KernelProfile::load_from_path(&path).unwrap();
            let model_name = raw_profile.model.clone();
            let effective_profile =
                KernelProfile::load_from_path_with_model_heuristics(&path, &model_name).unwrap();
            let expected = heuristic_attention_prefill_ax_bc64_mode_for_arch(
                &extract_architecture(&model_name),
            );
            assert_eq!(
                effective_profile.attention_prefill.ax_bc64_mode,
                expected,
                "profile {} drifted from attention_prefill.ax_bc64_mode heuristic for model {}",
                path.display(),
                model_name
            );
        }
    }

    #[test]
    fn test_current_profiles_match_attention_decode_hd128_n2_default_heuristic() {
        for entry in fs::read_dir(perf_dir()).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let raw_profile = KernelProfile::load_from_path(&path).unwrap();
            let model_name = raw_profile.model.clone();
            let effective_profile =
                KernelProfile::load_from_path_with_model_heuristics(&path, &model_name).unwrap();
            let expected = heuristic_attention_decode_hd128_n2_default_for_arch(
                &extract_architecture(&model_name),
            );
            assert_eq!(
                effective_profile.attention_decode.hd128_n2_default,
                expected,
                "profile {} drifted from attention_decode.hd128_n2_default heuristic for model {}",
                path.display(),
                model_name
            );
        }
    }

    #[test]
    fn test_current_profiles_match_attention_prefill_ax_bc64_min_tokens_heuristic() {
        for entry in fs::read_dir(perf_dir()).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let raw_profile = KernelProfile::load_from_path(&path).unwrap();
            let model_name = raw_profile.model.clone();
            let effective_profile =
                KernelProfile::load_from_path_with_model_heuristics(&path, &model_name).unwrap();
            let expected = heuristic_attention_prefill_ax_bc64_min_tokens_for_arch(
                &extract_architecture(&model_name),
            );
            assert_eq!(
                effective_profile.attention_prefill.ax_bc64_min_tokens,
                expected,
                "profile {} drifted from attention_prefill.ax_bc64_min_tokens heuristic for model {}",
                path.display(),
                model_name
            );
        }
    }

    #[test]
    fn test_qwen35_9b_heuristics_keep_prefill_f16_io_disabled() {
        // PRD-PREFILL-DISPATCH-CONSOLIDATION: blocked kernel is now the
        // default for all archs, so prefer_f16_io is always false.
        let profile = KernelProfile::load("Qwen3.5-9B-Q4_K_M", "Q4_K");
        assert!(!profile.batch_prefill.prefer_f16_io);
    }

    #[test]
    fn test_qwen35_9b_heuristics_keep_hd128_n2_default_enabled() {
        let profile = KernelProfile::load("Qwen3.5-9B-Q4_K_M", "Q4_K");
        assert_eq!(profile.attention_decode.hd128_n2_default, Some(true));
    }

    #[test]
    fn test_pruned_profile_load_restores_heuristic_fields_without_overriding_explicit_values() {
        let pruned = r#"{
            "model": "qwen35-9b",
            "source": "unit",
            "generated": "2026-03-29",
            "attention_decode": {
                "hd128_n2_default": true
            }
        }"#;
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("ax-engine-pruned-qwen35-9b-test.json");
        fs::write(&path, pruned).unwrap();

        let profile =
            KernelProfile::load_from_path_with_model_heuristics(&path, "Qwen3.5-9B-Q4_K_M")
                .unwrap();

        assert_eq!(profile.attention_decode.splitk_threshold, 512);
        // PRD-PREFILL-DISPATCH-CONSOLIDATION: blocked kernel for all archs.
        assert!(!profile.batch_prefill.prefer_f16_io);
        assert_eq!(profile.attention_decode.hd128_n2_default, Some(true));
        // Qwen35 keeps FA2 HD128 on Auto after heuristic restoration.
        assert_eq!(profile.attention_prefill.fa2_mode, ProfileKernelMode::Auto);
        assert_eq!(
            profile.attention_prefill.fa2_hd128_mode,
            ProfileKernelMode::Auto
        );
        assert_eq!(profile.attention_prefill.fa2_hd128_auto_min_tokens, 128);
        assert_eq!(
            profile.attention_prefill.ax_bc64_mode,
            ProfileKernelMode::Auto
        );
        assert_eq!(profile.attention_prefill.ax_bc64_min_tokens, 384);
        let regimes = profile.decode_regimes.as_ref().unwrap();
        assert_eq!(regimes.short_max_attend_len, 384);
        assert_eq!(regimes.long.attention_decode.splitk_threshold, Some(256));

        let _ = fs::remove_file(path);
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
                "ax_bc64_mode": "off",
                "fa2_auto_min_tokens": 640,
                "fa2_auto_min_base_seq": 320,
                "fa2_hd128_auto_min_tokens": 768,
                "ax_bc64_min_tokens": 1024
            },
            "kv_cache": {
                "precision": "q8_0"
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
            profile.attention_prefill.fa2_hd128_mode,
            ProfileKernelMode::Off
        );
        assert_eq!(profile.kv_cache.precision, KvPrecisionMode::Q8_0);
        assert_eq!(
            profile.attention_prefill.ax_bc64_mode,
            ProfileKernelMode::Off
        );
        assert_eq!(profile.attention_prefill.fa2_auto_min_base_seq, 320);
        assert_eq!(profile.attention_prefill.ax_bc64_min_tokens, 1024);
    }

    #[test]
    fn test_llama3_8b_heuristics_keep_prefill_fa2_hd128_auto_with_long_prompt_threshold() {
        let profile = KernelProfile::load("Llama-3-8B-Instruct-GGUF-Q4_K_M", "Q4_K_M");
        assert_eq!(
            profile.attention_prefill.fa2_hd128_mode,
            ProfileKernelMode::Auto
        );
        assert_eq!(profile.attention_prefill.fa2_hd128_auto_min_tokens, 1024);
    }

    #[test]
    fn test_llama3_8b_profile_prefers_ax_bc64_prefill_route() {
        let profile = KernelProfile::load("Llama-3-8B-Instruct-GGUF-Q4_K_M", "Q4_K_M");
        let config = crate::dispatch::AttentionDispatchConfig::from_profile(&profile);
        let selection = config.prefill_local_candidate_selection(512, 128);
        assert_eq!(
            selection.candidate,
            crate::dispatch::AttentionPrefillCandidate::AxBc64
        );
    }

    #[test]
    fn test_llama3_8b_profile_prefers_fa2_prefill_route_at_1024() {
        let profile = KernelProfile::load("Llama-3-8B-Instruct-GGUF-Q4_K_M", "Q4_K_M");
        let config = crate::dispatch::AttentionDispatchConfig::from_profile(&profile);
        let selection = config.prefill_local_candidate_selection(1024, 128);
        assert!(matches!(
            selection.candidate,
            crate::dispatch::AttentionPrefillCandidate::Fa2SimdHd128
                | crate::dispatch::AttentionPrefillCandidate::Fa2Hd128
                | crate::dispatch::AttentionPrefillCandidate::Fa2HalfHd128
        ));
    }

    #[test]
    fn test_qwen3_4b_profile_prefers_ax_bc64_at_256_and_fa2_at_384() {
        let profile = KernelProfile::load("Qwen3-4B-Q6_K", "Q6_K");
        let config = crate::dispatch::AttentionDispatchConfig::from_profile(&profile);

        let short = config.prefill_local_candidate_selection(256, 128);
        assert_eq!(
            short.candidate,
            crate::dispatch::AttentionPrefillCandidate::AxBc64
        );

        let medium = config.prefill_local_candidate_selection(384, 128);
        assert!(matches!(
            medium.candidate,
            crate::dispatch::AttentionPrefillCandidate::Fa2SimdHd128
                | crate::dispatch::AttentionPrefillCandidate::Fa2Hd128
                | crate::dispatch::AttentionPrefillCandidate::Fa2HalfHd128
        ));
    }

    #[test]
    fn test_qwen35_4b_profile_prefers_ax_bc64_at_256_and_fa2_at_384() {
        let profile = KernelProfile::load("Qwen3.5-4B-Q8_0", "Q8_0");
        let config = crate::dispatch::AttentionDispatchConfig::from_profile(&profile);

        let short = config.prefill_local_candidate_selection(256, 128);
        assert_eq!(
            short.candidate,
            crate::dispatch::AttentionPrefillCandidate::AxBc64
        );

        let medium = config.prefill_local_candidate_selection(384, 128);
        assert!(matches!(
            medium.candidate,
            crate::dispatch::AttentionPrefillCandidate::Fa2SimdHd128
                | crate::dispatch::AttentionPrefillCandidate::Fa2Hd128
                | crate::dispatch::AttentionPrefillCandidate::Fa2HalfHd128
        ));
    }

    #[test]
    fn test_qwen35_4b_profile_enables_hd128_n2_decode_default() {
        let profile = KernelProfile::load("Qwen3.5-4B-Q8_0", "Q8_0");
        assert_eq!(profile.attention_decode.hd128_n2_default, Some(true));
    }

    #[test]
    fn test_qwen3_8b_profile_keeps_fa2_prefill_route() {
        let profile = KernelProfile::load("Qwen3-8B-Q4_K_M", "Q4_K_M");
        let config = crate::dispatch::AttentionDispatchConfig::from_profile(&profile);
        let selection = config.prefill_local_candidate_selection(512, 128);
        assert!(matches!(
            selection.candidate,
            crate::dispatch::AttentionPrefillCandidate::Fa2SimdHd128
                | crate::dispatch::AttentionPrefillCandidate::Fa2Hd128
                | crate::dispatch::AttentionPrefillCandidate::Fa2HalfHd128
        ));
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
            ("Qwen3.5-4B-Instruct", "qwen35-4b", "qwen35-4b"),
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
