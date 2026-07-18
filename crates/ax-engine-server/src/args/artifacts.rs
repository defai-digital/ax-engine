use serde_json::Value;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use super::presets::PresetDefinition;

pub(super) const MODEL_MANIFEST_FILE: &str = "model-manifest.json";
const GEMMA4_ASSISTANT_MTP_CONTRACT_FILE: &str = "ax_gemma4_assistant_mtp.json";
const GLM_MTP_MANIFEST_FILE: &str = "ax_glm_mtp_manifest.json";

pub(super) fn hf_cache_roots(explicit_root: Option<PathBuf>) -> Vec<PathBuf> {
    if let Some(root) = explicit_root {
        return vec![root];
    }

    let mut roots = Vec::new();
    if let Some(root) = env::var_os("HF_HUB_CACHE").map(PathBuf::from) {
        roots.push(root);
    }
    if let Some(home) = env::var_os("HF_HOME").map(PathBuf::from) {
        roots.push(home.join("hub"));
    }
    if let Some(home) = env::var_os("HOME").map(PathBuf::from) {
        roots.push(home.join(".cache").join("huggingface").join("hub"));
    }
    dedupe_paths(roots)
}

pub(super) fn resolve_hf_cache_model_artifacts(
    preset: &PresetDefinition,
    roots: Vec<PathBuf>,
) -> Result<PathBuf, String> {
    if roots.is_empty() {
        return Err(
            "no Hugging Face cache root found; set HF_HUB_CACHE, HF_HOME, HOME, or pass --hf-cache-root"
                .to_string(),
        );
    }

    let mut candidates = Vec::new();
    let mut rejected = Vec::new();
    for root in roots {
        if !root.is_dir() {
            continue;
        }
        for path in hf_cache_candidate_dirs(&root, preset) {
            match validate_preset_model_artifacts(&path, preset) {
                Ok(()) => candidates.push(path),
                Err(message) => rejected.push(format!("{} ({message})", path.display())),
            }
        }
    }

    candidates.sort();
    candidates.dedup();

    match candidates.len() {
        1 => Ok(candidates.remove(0)),
        0 => {
            let mut message = format!(
                "no valid AX model artifacts found in Hugging Face cache for preset {}; \
                 expected config.json, {MODEL_MANIFEST_FILE}, safetensors, and model_type in {:?}",
                preset.label, preset.model_types
            );
            if !rejected.is_empty() {
                message.push_str("; rejected candidates: ");
                message.push_str(&rejected.join("; "));
            }
            message.push_str(
                "\nhint: if you downloaded a snapshot but haven't generated the manifest yet, run:\
                 \n  cargo run -p ax-engine-core --bin generate-manifest -- <snapshot-dir>\
                 \nor use the download script:\
                 \n  python scripts/download_model.py <org/repo-id>",
            );
            Err(message)
        }
        _ => Err(format!(
            "multiple Hugging Face cache candidates matched preset {}; pass --mlx-model-artifacts-dir explicitly: {}",
            preset.label,
            candidates
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

pub(crate) fn infer_model_id_from_artifacts(path: &Path) -> Result<Option<String>, String> {
    if path.join(GEMMA4_ASSISTANT_MTP_CONTRACT_FILE).is_file() {
        let contract = load_json(&path.join(GEMMA4_ASSISTANT_MTP_CONTRACT_FILE))?;
        if let Some(target_model_id) = contract.get("target_model_id").and_then(Value::as_str)
            && !target_model_id.trim().is_empty()
        {
            return Ok(Some(format!("{}-assistant-mtp", target_model_id.trim())));
        }
        return Ok(Some("gemma4-assistant-mtp".to_string()));
    }

    if path.join(GLM_MTP_MANIFEST_FILE).is_file() {
        return Ok(Some("glm4_moe_lite".to_string()));
    }

    let config_path = path.join("config.json");
    if !config_path.is_file() {
        return Ok(None);
    }
    let config = load_json(&config_path)?;
    let Some(model_type) = config_model_type(&config) else {
        return Ok(None);
    };
    let path_label = normalize_model_label(&path.display().to_string());

    Ok(match model_type {
        "glm4_moe_lite" => Some("glm4_moe_lite".to_string()),
        "gemma4" | "gemma4_unified" | "gemma4_unified_text" | "gemma4_assistant" => {
            Some(infer_gemma4_model_id(&path_label))
        }
        "diffusion_gemma" => Some("diffusiongemma".to_string()),
        value if value.starts_with("qwen3") => Some(infer_qwen_model_id(&path_label, value)),
        _ => None,
    })
}

fn infer_gemma4_model_id(path_label: &str) -> String {
    if path_label.contains("gemma-4-31b") {
        "gemma-4-31b-it".to_string()
    } else if path_label.contains("gemma-4-26b") {
        "gemma-4-26b-a4b-it".to_string()
    } else if path_label.contains("gemma-4-12b") {
        "gemma-4-12b-it".to_string()
    } else if path_label.contains("gemma-4-e4b") {
        "gemma-4-e4b-it".to_string()
    } else if path_label.contains("gemma-4-e2b") {
        "gemma-4-e2b-it".to_string()
    } else {
        "gemma4".to_string()
    }
}

fn infer_qwen_model_id(path_label: &str, model_type: &str) -> String {
    let mut model_id = if path_label.contains("qwen3-coder-next") {
        "qwen3-coder-next".to_string()
    } else if path_label.contains("qwen3-6-35b") || path_label.contains("qwen36-35b") {
        "qwen3.6-35b".to_string()
    } else if path_label.contains("qwen3-6-27b") || path_label.contains("qwen36-27b") {
        "qwen3.6-27b".to_string()
    } else if path_label.contains("qwen3-5-9b") || path_label.contains("qwen35-9b") {
        "qwen3.5-9b".to_string()
    } else if matches!(
        model_type,
        "qwen3_5_moe" | "qwen3_5_text" | "qwen3_next" | "qwen3_6" | "qwen3.6"
    ) {
        "qwen3.6".to_string()
    } else if matches!(model_type, "qwen3_5" | "qwen3.5") {
        "qwen3.5".to_string()
    } else {
        "qwen3".to_string()
    };

    if path_label.contains("mtp") && !model_id.contains("mtp") {
        model_id.push_str("-mtp");
    }
    model_id
}

fn hf_cache_candidate_dirs(root: &Path, preset: &PresetDefinition) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if looks_like_artifact_dir(root) && path_matches_preset(root, preset) {
        candidates.push(root.to_path_buf());
    }

    let Ok(entries) = fs::read_dir(root) else {
        return candidates;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() || !path_matches_preset(&path, preset) {
            continue;
        }
        let snapshots = path.join("snapshots");
        if snapshots.is_dir() {
            if let Ok(snapshot_entries) = fs::read_dir(&snapshots) {
                candidates.extend(
                    snapshot_entries
                        .flatten()
                        .map(|entry| entry.path())
                        .filter(|path| path.is_dir()),
                );
            }
        } else {
            candidates.push(path);
        }
    }

    candidates
}

fn validate_preset_model_artifacts(path: &Path, preset: &PresetDefinition) -> Result<(), String> {
    if !path.join("config.json").is_file() {
        return Err("missing config.json".to_string());
    }
    if !path.join(MODEL_MANIFEST_FILE).is_file() {
        return Err(format!(
            "missing {MODEL_MANIFEST_FILE}; generate it with:\
             \n  cargo run -p ax-engine-core --bin generate-manifest -- {}\
             \nor use the download script which handles this step:\
             \n  python scripts/download_model.py <org/repo-id>",
            path.display()
        ));
    }
    if !dir_contains_safetensors(path) {
        return Err("missing safetensors file".to_string());
    }

    let config = load_json(path.join("config.json").as_path())?;
    let model_type = config_model_type(&config).ok_or("missing model_type in config.json")?;
    if !preset.model_types.contains(&model_type) {
        return Err(format!(
            "model_type {model_type} does not match preset {} expected {:?}",
            preset.label, preset.model_types
        ));
    }

    Ok(())
}

fn looks_like_artifact_dir(path: &Path) -> bool {
    path.join("config.json").is_file()
}

fn path_matches_preset(path: &Path, preset: &PresetDefinition) -> bool {
    let normalized = normalize_model_label(&path.display().to_string());
    preset
        .aliases
        .iter()
        .any(|alias| normalized.contains(&normalize_model_label(alias)))
}

fn normalize_model_label(value: &str) -> String {
    value
        .to_ascii_lowercase()
        .replace("--", "-")
        .replace(['_', '/', '.'], "-")
}

fn dir_contains_safetensors(path: &Path) -> bool {
    fs::read_dir(path).is_ok_and(|entries| {
        entries.flatten().any(|entry| {
            entry
                .path()
                .extension()
                .and_then(|extension| extension.to_str())
                == Some("safetensors")
        })
    })
}

fn load_json(path: &Path) -> Result<Value, String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))
}

fn config_model_type(config: &Value) -> Option<&str> {
    config
        .get("model_type")
        .and_then(Value::as_str)
        .or_else(|| config.get("text_config")?.get("model_type")?.as_str())
}

fn dedupe_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut deduped = Vec::new();
    for path in paths {
        if !deduped.contains(&path) {
            deduped.push(path);
        }
    }
    deduped
}
