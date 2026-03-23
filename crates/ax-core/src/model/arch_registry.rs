//! Architecture registry: maps GGUF architecture names to forward pass implementations.

use crate::model::forward::ForwardPass;
use crate::model::gemma3::Gemma3Forward;
use crate::model::llama::LlamaForward;
use crate::model::qwen3::Qwen3Forward;

/// Create the appropriate `ForwardPass` implementation for a given architecture name.
///
/// Architecture names come from the GGUF `general.architecture` metadata key.
pub fn forward_for_arch(arch: &str) -> anyhow::Result<Box<dyn ForwardPass>> {
    match arch {
        "llama" | "mistral" | "codellama" => Ok(Box::new(LlamaForward)),
        "qwen2" | "qwen3" => Ok(Box::new(Qwen3Forward)),
        "gemma" | "gemma2" | "gemma3" => Ok(Box::new(Gemma3Forward)),
        _ => anyhow::bail!(
            "unsupported architecture: '{arch}'. Supported: llama, mistral, codellama, qwen2, qwen3, gemma, gemma2, gemma3"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_variants() {
        for arch in &["llama", "mistral", "codellama"] {
            let fwd = forward_for_arch(arch).unwrap();
            assert_eq!(fwd.arch_name(), "llama");
        }
    }

    #[test]
    fn test_qwen_variants() {
        for arch in &["qwen2", "qwen3"] {
            let fwd = forward_for_arch(arch).unwrap();
            assert_eq!(fwd.arch_name(), "qwen3");
        }
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
