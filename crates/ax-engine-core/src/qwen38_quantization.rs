//! Product format policy for the two Qwen 3.8 AXQ targets.
//!
//! Supported packs: MLX AXQ MXFP4 / MXFP8 and affine 6-bit/group64 (the
//! quality-size balance point, ADR-050). Other standalone affine widths stay
//! out of product scope. Source identity distinguishes 27B from older models
//! sharing `qwen3_5`. Companion affine8 tensors keep their exported precision.

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
    let mut has_product_projection = false;
    for tensor in &manifest.tensors {
        if let Some(quant) = &tensor.quantization {
            validate_projection(quant, tensor.role, flash_next)
                .map_err(|message| format!("{}: {message}", tensor.name))?;
            has_product_projection |=
                matches!(quant.mode.as_str(), "mxfp4" | "mxfp8") || is_affine6(quant);
        }
    }
    if !has_product_projection {
        return Err(
            "Qwen 3.8 AXQ requires MXFP4/MXFP8 or affine 6-bit/group64 decoder projections".into(),
        );
    }
    Ok(())
}

/// The supported affine width: 6-bit, group 64.
fn is_affine6(quant: &NativeTensorQuantization) -> bool {
    quant.mode == "affine" && (quant.bits, quant.group_size) == (6, 64)
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
    if is_affine6(quant) {
        return Ok(());
    }
    if !expert && quant.mode == "affine" && quant.bits == 8 && matches!(quant.group_size, 32 | 64) {
        return Ok(());
    }
    Err("Qwen 3.8 supports MLX AXQ MXFP4/MXFP8 or affine 6-bit/group64 packs; other standalone affine widths are not supported (protected affine8 companions remain valid)".into())
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
    fn source_identity_scopes_the_policy_without_breaking_older_qwen() {
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
                "quantization": {"mode": "affine", "bits": 4, "group_size": 64}
            }]
        }))
        .unwrap();
        // Older Qwen models sharing `qwen3_5` keep every affine width.
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
        assert!(
            validate(&root, &manifest)
                .unwrap_err()
                .contains("not supported")
        );
        manifest.tensors[0].quantization = Some(quant("affine", 6, 64));
        assert!(
            validate(&root, &manifest).is_ok(),
            "27B affine 6-bit is supported"
        );
        manifest.tensors[0].quantization = Some(quant("mxfp4", 4, 32));
        assert!(validate(&root, &manifest).is_ok());
        manifest.tensors[0].quantization = Some(quant("mxfp8", 8, 32));
        assert!(validate(&root, &manifest).is_ok());
        manifest.tensors[0].quantization = Some(quant("affine", 8, 64));
        assert!(
            validate(&root, &manifest).unwrap_err().contains("requires"),
            "an affine8 companion alone is not a product pack"
        );
        manifest.model_family = "qwen4_exp".into();
        std::fs::remove_file(metadata).unwrap();
        manifest.tensors[0].quantization = Some(quant("affine", 6, 64));
        assert!(
            validate(&root, &manifest).is_ok(),
            "Flash Next affine 6-bit is supported"
        );
        manifest.tensors[0].quantization = Some(quant("affine", 4, 64));
        assert!(
            validate(&root, &manifest)
                .unwrap_err()
                .contains("not supported")
        );
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn only_affine_6bit_group64_is_admitted_among_standalone_affine_widths() {
        for flash in [false, true] {
            for role in [NativeTensorRole::FfnGateExps, NativeTensorRole::AttentionQ] {
                assert!(validate_projection(&quant("affine", 6, 64), role, flash).is_ok());
                assert!(validate_projection(&quant("affine", 6, 32), role, flash).is_err());
                for bits in [2, 3, 4] {
                    assert!(validate_projection(&quant("affine", bits, 64), role, flash).is_err());
                }
            }
            // Affine8 stays a non-expert companion only.
            assert!(
                validate_projection(
                    &quant("affine", 8, 64),
                    NativeTensorRole::FfnGateExps,
                    flash
                )
                .is_err()
            );
        }
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

    fn empty_pack_dir(label: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!(
            "ax-qwen38-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        root
    }

    fn manifest_from(family: &str, tensors: serde_json::Value) -> NativeModelManifest {
        serde_json::from_value(serde_json::json!({
            "schema_version": "ax.native_model.v1",
            "model_family": family,
            "tensor_format": "safetensors",
            "layer_count": 1,
            "hidden_size": 64,
            "attention_head_count": 1,
            "attention_head_dim": 64,
            "kv_head_count": 1,
            "vocab_size": 32,
            "tensors": tensors,
        }))
        .unwrap()
    }

    fn listed_tensor(name: &str, role: &str, quant: Option<(&str, u32, u32)>) -> serde_json::Value {
        let mut value = serde_json::json!({
            "name": name,
            "role": role,
            "dtype": "u32",
            "shape": [64, 12],
            "file": "optiq-mixed-2bit.safetensors",
            "offset_bytes": 0,
            "length_bytes": 8,
        });
        if let Some((mode, bits, group_size)) = quant {
            value["quantization"] = serde_json::json!({
                "mode": mode,
                "bits": bits,
                "group_size": group_size,
            });
        } else {
            value["dtype"] = serde_json::json!("bf16");
        }
        value
    }

    /// Admission reads the manifest only. The named weight file is absent on
    /// purpose: a loader that opened it would not be able to return this
    /// policy error.
    #[test]
    fn standalone_two_bit_rejected() {
        let root = empty_pack_dir("two-bit");
        for (bits, group_size) in [(2, 64), (2, 32)] {
            let manifest = manifest_from(
                "qwen4_exp",
                serde_json::json!([listed_tensor(
                    "layers.0.mlp.experts.gate_proj.weight",
                    "ffn_gate_exps",
                    Some(("affine", bits, group_size)),
                )]),
            );
            let error = validate(&root, &manifest).unwrap_err();
            assert!(
                error.contains("not supported"),
                "affine {bits}/{group_size}: {error}"
            );
        }
        std::fs::write(
            root.join("axquant_manifest.json"),
            r#"{"source_model":{"model_id":"Qwen/Qwen3.8-27B"}}"#,
        )
        .unwrap();
        let manifest = manifest_from(
            "qwen3_5",
            serde_json::json!([listed_tensor(
                "model.layers.0.self_attn.q_proj.weight",
                "attention_q",
                Some(("affine", 2, 64)),
            )]),
        );
        let error = validate(&root, &manifest).unwrap_err();
        assert!(error.contains("not supported"), "{error}");
        assert!(!root.join("optiq-mixed-2bit.safetensors").exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    /// OptiQ-style mix: 2-bit experts, a wider affine projection, and a plain
    /// norm tensor, with no AXQuant identity. Refused from the manifest before
    /// any weight file is opened. A folded-norm manifest that also lacks a
    /// supported product quantizer is refused the same way.
    #[test]
    fn optiq_mixed_pack_rejected() {
        let root = empty_pack_dir("optiq-mixed");
        let manifest = manifest_from(
            "qwen4_exp",
            serde_json::json!([
                listed_tensor(
                    "layers.0.mlp.experts.gate_proj.weight",
                    "ffn_gate_exps",
                    Some(("affine", 2, 32)),
                ),
                listed_tensor(
                    "layers.0.self_attn.q_proj.weight",
                    "attention_q",
                    Some(("affine", 4, 64)),
                ),
                listed_tensor(
                    "layers.0.self_attn.k_proj.weight",
                    "attention_k",
                    Some(("affine", 8, 64)),
                ),
                listed_tensor("layers.0.self_attn.q_norm.weight", "attention_q_norm", None,),
            ]),
        );
        let error = validate(&root, &manifest).unwrap_err();
        assert!(error.contains("not supported"), "{error}");
        let folded_only = manifest_from(
            "qwen4_exp",
            serde_json::json!([listed_tensor(
                "layers.0.self_attn.q_norm.weight",
                "attention_q_norm",
                None,
            )]),
        );
        let folded_error = validate(&root, &folded_only).unwrap_err();
        assert!(
            folded_error.contains("requires"),
            "folded norm pack without a supported quantizer identity: {folded_error}"
        );
        assert!(!root.join("optiq-mixed-2bit.safetensors").exists());
        assert!(!root.join("axquant_manifest.json").exists());
        std::fs::remove_dir_all(root).unwrap();
    }
}
