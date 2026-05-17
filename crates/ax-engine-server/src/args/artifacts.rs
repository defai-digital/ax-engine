use serde_json::Value;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use super::presets::PresetDefinition;

pub(super) const MODEL_MANIFEST_FILE: &str = "model-manifest.json";

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
