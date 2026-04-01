//! Architecture registry: maps GGUF architecture names to forward pass implementations.

use std::path::Path;

use crate::gguf::MappedModel;
use crate::gguf::tensor::GgmlType;
use crate::model::config::ModelConfig;
use crate::model::forward::ForwardPass;
use crate::model::gemma3::Gemma3Forward;
use crate::model::llama::LlamaForward;
use crate::model::qwen3::Qwen3Forward;
use crate::model::qwen3_moe::Qwen3MoeForward;
use crate::model::qwen35::Qwen35Forward;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeSupportLevel {
    Full,
    PartialQuant { unsupported_types: Vec<GgmlType> },
    UnsupportedArch { arch: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeSupportResult {
    pub level: NativeSupportLevel,
    pub architecture: String,
    pub predominant_quant: Option<GgmlType>,
    pub n_tensors: usize,
    pub file_size_bytes: u64,
}

pub fn is_arch_supported(arch: &str) -> bool {
    matches!(
        arch,
        "llama"
            | "qwen2"
            | "qwen3"
            | "qwen2moe"
            | "qwen3moe"
            | "qwen35"
            | "qwen35moe"
            | "gemma"
            | "gemma2"
            | "gemma3"
    )
}

pub fn is_quant_supported(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::Q8_0
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
    )
}

/// Inspect a GGUF file and determine whether AX Engine can handle it natively.
///
/// This reads GGUF metadata and tensor info only; weights remain mmap-backed
/// and are not materialized or uploaded.
pub fn check_native_support(path: &Path) -> anyhow::Result<NativeSupportResult> {
    let mapped = MappedModel::open(path)?;
    let architecture = mapped.header.architecture().unwrap_or("llama").to_string();
    let predominant_quant = mapped.predominant_quant();
    let mut unsupported_types = mapped
        .tensors
        .iter()
        .map(|tensor| tensor.dtype)
        .filter(|dtype| !is_quant_supported(*dtype))
        .collect::<Vec<_>>();
    unsupported_types.sort_by_key(|dtype| *dtype as u32);
    unsupported_types.dedup();

    let level = if !is_arch_supported(&architecture) {
        NativeSupportLevel::UnsupportedArch {
            arch: architecture.clone(),
        }
    } else if unsupported_types.is_empty() {
        NativeSupportLevel::Full
    } else {
        NativeSupportLevel::PartialQuant { unsupported_types }
    };

    Ok(NativeSupportResult {
        level,
        architecture,
        predominant_quant,
        n_tensors: mapped.tensors.len(),
        file_size_bytes: mapped.file_size() as u64,
    })
}

/// Create the appropriate `ForwardPass` implementation for a given architecture name.
///
/// Architecture names come from the GGUF `general.architecture` metadata key.
pub fn forward_for_arch(arch: &str) -> anyhow::Result<Box<dyn ForwardPass>> {
    match arch {
        "llama" => Ok(Box::new(LlamaForward)),
        "phi3" => anyhow::bail!(
            "unsupported architecture: 'phi3'. Phi-3/Phi-4 support has been removed from AX."
        ),
        "qwen2" | "qwen3" => Ok(Box::new(Qwen3Forward)),
        "qwen2moe" | "qwen3moe" => Ok(Box::new(Qwen3MoeForward)),
        "qwen35" | "qwen35moe" => Ok(Box::new(Qwen35Forward)),
        "gemma" | "gemma2" | "gemma3" => Ok(Box::new(Gemma3Forward)),
        _ => anyhow::bail!(
            "unsupported architecture: '{arch}'. Supported: llama, qwen2, qwen3, qwen3moe, qwen35, qwen35moe, gemma, gemma2, gemma3"
        ),
    }
}

/// Create a `ForwardPass` with config awareness. Detects MoE variants
/// (e.g. Qwen3 with n_expert > 0) and routes to the appropriate implementation.
pub fn forward_for_arch_with_config(
    arch: &str,
    config: &ModelConfig,
) -> anyhow::Result<Box<dyn ForwardPass>> {
    // Detect MoE: if Qwen2/Qwen3 architecture with n_expert > 0, use MoE forward
    if matches!(arch, "qwen2" | "qwen3") && config.n_expert.is_some_and(|n| n > 0) {
        tracing::info!(
            n_expert = config.n_expert.unwrap(),
            n_expert_used = config.n_expert_used.unwrap_or(0),
            "Detected Qwen3 MoE model, using Qwen3MoeForward"
        );
        return Ok(Box::new(Qwen3MoeForward));
    }
    forward_for_arch(arch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_arch_supported_matches_native_matrix() {
        assert!(is_arch_supported("llama"));
        assert!(is_arch_supported("qwen3"));
        assert!(is_arch_supported("qwen35"));
        assert!(is_arch_supported("qwen35moe"));
        assert!(is_arch_supported("gemma3"));
        assert!(!is_arch_supported("mistral"));
    }

    #[test]
    fn test_is_quant_supported_matches_native_matrix() {
        assert!(is_quant_supported(GgmlType::F32));
        assert!(is_quant_supported(GgmlType::Q4K));
        assert!(is_quant_supported(GgmlType::Q6K));
        assert!(!is_quant_supported(GgmlType::Q4_0));
        assert!(!is_quant_supported(GgmlType::Q5_0));
        assert!(!is_quant_supported(GgmlType::Q2K));
        assert!(!is_quant_supported(GgmlType::Q3K));
        assert!(!is_quant_supported(GgmlType::Q8K));
    }

    #[test]
    fn test_llama_variants() {
        let fwd = forward_for_arch("llama").unwrap();
        assert_eq!(fwd.arch_name(), "llama");
    }

    #[test]
    fn test_phi3_removed() {
        let err = forward_for_arch("phi3").unwrap_err();
        assert!(
            err.to_string()
                .contains("Phi-3/Phi-4 support has been removed"),
            "got: {err}"
        );
    }

    #[test]
    fn test_qwen_variants() {
        for arch in &["qwen2", "qwen3"] {
            let fwd = forward_for_arch(arch).unwrap();
            assert_eq!(fwd.arch_name(), "qwen3");
        }
    }

    #[test]
    fn test_qwen35_variant() {
        let fwd = forward_for_arch("qwen35").unwrap();
        assert_eq!(fwd.arch_name(), "qwen35");
    }

    #[test]
    fn test_qwen35moe_variant() {
        let fwd = forward_for_arch("qwen35moe").unwrap();
        assert_eq!(fwd.arch_name(), "qwen35");
    }

    #[test]
    fn test_gemma_variants() {
        for arch in &["gemma", "gemma2", "gemma3"] {
            let fwd = forward_for_arch(arch).unwrap();
            assert_eq!(fwd.arch_name(), "gemma3");
        }
    }

    #[test]
    fn test_unsupported_arch() {
        let err = forward_for_arch("mamba").unwrap_err();
        assert!(
            err.to_string().contains("unsupported architecture"),
            "got: {err}"
        );
    }
}
