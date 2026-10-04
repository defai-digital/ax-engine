//! Product format policy for the two Qwen 3.8 AXQ targets.
//!
//! Source identity distinguishes 27B from older models sharing `qwen3_5`.
//! Companion affine8 tensors inside MX packs retain their exported precision.

use std::path::Path;

use crate::model::{NativeModelManifest, NativeTensorQuantization, NativeTensorRole};

pub(crate) fn validate(root: &Path, manifest: &NativeModelManifest) -> Result<(), String> {
    let flash_next = manifest.model_family == "qwen4_exp";
    if !flash_next && manifest.model_family != "qwen3_5" {
        return Ok(());
    }
    if !flash_next {
        let path = root.join("axquant_manifest.json");
        let data = match std::fs::read(&path) {
            Ok(data) => data,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(format!("cannot read AXQuant source identity: {error}")),
        };
        let metadata: serde_json::Value = serde_json::from_slice(&data)
            .map_err(|error| format!("invalid AXQuant source identity: {error}"))?;
        if metadata
            .pointer("/source_model/model_id")
            .and_then(serde_json::Value::as_str)
            != Some("Qwen/Qwen3.8-27B")
        {
            return Ok(());
        }
    }
    let mut has_mx_projection = false;
    for tensor in &manifest.tensors {
        if let Some(quant) = &tensor.quantization {
            validate_projection(quant, tensor.role, flash_next)
                .map_err(|message| format!("{}: {message}", tensor.name))?;
            has_mx_projection |= matches!(quant.mode.as_str(), "mxfp4" | "mxfp8");
        }
    }
    if !has_mx_projection {
        return Err("Qwen 3.8 AXQ requires MXFP4/MXFP8 decoder projections; standalone affine packs are retired".into());
    }
    Ok(())
}

fn validate_projection(
    quant: &NativeTensorQuantization,
    role: NativeTensorRole,
    flash_next: bool,
) -> Result<(), String> {
    let expert = matches!(
        role,
        NativeTensorRole::FfnGateExps
            | NativeTensorRole::FfnUpExps
            | NativeTensorRole::FfnDownExps
            | NativeTensorRole::FfnGateUpExpsPacked
    );
    if quant.mode == "mxfp4" && (quant.bits, quant.group_size) == (4, 32) {
        return Ok(());
    }
    if quant.mode == "mxfp8" && (quant.bits, quant.group_size) == (8, 32) {
        if flash_next {
            return Err("Flash Next MXFP8 admission requires an audited AXQ MTP pack; use the pinned MXFP4 MTP pack".into());
        }
        return Ok(());
    }
    if !expert && quant.mode == "affine" && quant.bits == 8 && matches!(quant.group_size, 32 | 64) {
        return Ok(());
    }
    Err("Qwen 3.8 standalone affine packs are retired; use MLX AXQ MXFP4/MXFP8 with MTP (protected affine8 companions remain valid)".into())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn quant(mode: &str, bits: u32, group_size: u32) -> NativeTensorQuantization {
        NativeTensorQuantization {
            mode: mode.into(),
            bits,
            group_size,
        }
    }

    #[test]
    fn source_identity_scopes_retirement_without_breaking_older_qwen() {
        let root = std::env::temp_dir().join(format!(
            "ax-qwen38-mx-policy-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let mut manifest: NativeModelManifest = serde_json::from_value(serde_json::json!({
            "schema_version": "ax.native_model.v1", "model_family": "qwen3_5",
            "tensor_format": "safetensors", "layer_count": 1, "hidden_size": 64,
            "attention_head_count": 1, "attention_head_dim": 64,
            "kv_head_count": 1, "vocab_size": 32, "tensors": [{
                "name": "projection", "role": "attention_q", "dtype": "u32",
                "shape": [64, 12], "file": "model.safetensors",
                "offset_bytes": 0, "length_bytes": 3072,
                "quantization": {"mode": "affine", "bits": 6, "group_size": 64}
            }]
        }))
        .unwrap();
        assert!(validate(&root, &manifest).is_ok());
        let metadata = root.join("axquant_manifest.json");
        std::fs::write(
            &metadata,
            r#"{"source_model":{"model_id":"Qwen/Qwen3.6-27B"}}"#,
        )
        .unwrap();
        assert!(validate(&root, &manifest).is_ok());
        std::fs::write(
            &metadata,
            r#"{"source_model":{"model_id":"Qwen/Qwen3.8-27B"}}"#,
        )
        .unwrap();
        assert!(validate(&root, &manifest).unwrap_err().contains("retired"));
        manifest.tensors[0].quantization = Some(quant("mxfp4", 4, 32));
        assert!(validate(&root, &manifest).is_ok());
        manifest.tensors[0].quantization = Some(quant("mxfp8", 8, 32));
        assert!(validate(&root, &manifest).is_ok());
        manifest.tensors[0].quantization = Some(quant("affine", 8, 64));
        assert!(validate(&root, &manifest).unwrap_err().contains("retired"));
        manifest.model_family = "qwen4_exp".into();
        std::fs::remove_file(metadata).unwrap();
        manifest.tensors[0].quantization = Some(quant("affine", 6, 64));
        assert!(validate(&root, &manifest).unwrap_err().contains("retired"));
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn retirement_cannot_be_bypassed_by_experimental_opt_ins() {
        for bits in [2, 4, 6, 8] {
            assert!(
                validate_projection(
                    &quant("affine", bits, 64),
                    NativeTensorRole::FfnGateExps,
                    true
                )
                .is_err()
            );
        }
        assert!(
            validate_projection(&quant("affine", 6, 64), NativeTensorRole::AttentionQ, false)
                .is_err()
        );
        assert!(
            validate_projection(&quant("affine", 4, 32), NativeTensorRole::AttentionQ, false)
                .is_err()
        );
    }

    #[test]
    fn mx_modes_keep_identity_and_protected_precision() {
        for flash in [false, true] {
            assert!(
                validate_projection(&quant("mxfp4", 4, 32), NativeTensorRole::FfnGateExps, flash)
                    .is_ok()
            );
            assert!(
                validate_projection(&quant("affine", 8, 64), NativeTensorRole::LmHead, flash)
                    .is_ok()
            );
            assert!(
                validate_projection(&quant("mxfp4", 4, 64), NativeTensorRole::AttentionQ, flash)
                    .is_err()
            );
            assert!(
                validate_projection(&quant("mxfp8", 4, 32), NativeTensorRole::AttentionQ, flash)
                    .is_err()
            );
        }
        assert!(
            validate_projection(&quant("mxfp8", 8, 32), NativeTensorRole::AttentionQ, false)
                .is_ok()
        );
        assert!(
            validate_projection(&quant("mxfp8", 8, 32), NativeTensorRole::FfnGateExps, true)
                .unwrap_err()
                .contains("audited")
        );
    }
}
