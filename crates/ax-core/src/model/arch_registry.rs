//! Architecture registry: maps GGUF architecture names to forward pass implementations.

use crate::model::forward::ForwardPass;
use crate::model::gemma3::Gemma3Forward;
use crate::model::llama::LlamaForward;
use crate::model::qwen3::Qwen3Forward;
use crate::model::qwen35::Qwen35Forward;

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
        "qwen35" => Ok(Box::new(Qwen35Forward)),
        "gemma" | "gemma2" | "gemma3" => Ok(Box::new(Gemma3Forward)),
        _ => anyhow::bail!(
            "unsupported architecture: '{arch}'. Supported: llama, qwen2, qwen3, qwen35, gemma, gemma2, gemma3"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
