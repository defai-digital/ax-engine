//! DeepSeek-V4 Jinja equivalent to the canonical `encoding/encoding_dsv4.py`.
//!
//! Flash-0731 ships no `chat_template.jinja`. mlx-lm convert then emits a
//! tokenizer without one, so standard chat-template consumers see a raw user
//! string. The vendored template is pinned to a Hugging Face staff proposal
//! that implements the canonical encoder, including thinking, tools, tool
//! results, response formats, and multi-turn history. See the adjacent
//! `deepseek_v4_chat_template.LICENSE.txt` for its MIT license.
//! Source: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/014a5cfe6d1349d3d1096b2f8c15faaaa11819d5/chat_template.jinja>.

use std::fs;
use std::path::{Path, PathBuf};

use super::ConvertError;

/// Hugging Face revision containing the DeepSeek-V4 Jinja proposal.
pub const DEEPSEEK_V4_CHAT_TEMPLATE_REVISION: &str = "014a5cfe6d1349d3d1096b2f8c15faaaa11819d5";

/// SHA-256 of the vendored template at [`DEEPSEEK_V4_CHAT_TEMPLATE_REVISION`].
pub const DEEPSEEK_V4_CHAT_TEMPLATE_SHA256: &str =
    "c3f06ef01ca187c2a14151ab7464e4060a11380c4d082b4c8bcbf266ad932274";

pub const DEEPSEEK_V4_CHAT_TEMPLATE: &str = include_str!("deepseek_v4_chat_template.jinja");

fn dir_is_deepseek_v4(model_dir: &Path) -> bool {
    if let Ok(text) = fs::read_to_string(model_dir.join("model-manifest.json")) {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
            if value.get("model_family").and_then(|v| v.as_str()) == Some("deepseek_v4") {
                return true;
            }
        }
    }
    fs::read_to_string(model_dir.join("config.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|value| {
            value
                .get("model_type")
                .and_then(|v| v.as_str())
                .map(|model_type| model_type == "deepseek_v4")
        })
        .unwrap_or(false)
}

/// Write the canonical-equivalent V4 chat Jinja when a DeepSeek V4 pack has none.
pub fn ensure_deepseek_v4_chat_template(model_dir: &Path) -> Result<Option<PathBuf>, ConvertError> {
    if !dir_is_deepseek_v4(model_dir) {
        return Ok(None);
    }
    let path = model_dir.join("chat_template.jinja");
    if path.is_file() {
        let existing = fs::read_to_string(&path).map_err(|source| ConvertError::ReadFile {
            path: path.clone(),
            source,
        })?;
        if !existing.trim().is_empty() {
            return Ok(Some(path));
        }
    }
    fs::write(&path, DEEPSEEK_V4_CHAT_TEMPLATE).map_err(|source| ConvertError::ReadFile {
        path: path.clone(),
        source,
    })?;
    Ok(Some(path))
}
