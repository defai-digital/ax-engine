//! Official DeepSeek-V4 chat-mode Jinja (source `encoding/encoding_dsv4.py`).
//!
//! Flash-0731 ships no `chat_template.jinja`. mlx-lm convert then emits a
//! tokenizer without one, so generate sees a raw user string. This file is
//! the official single-turn *chat* (non-thinking) path:
//!
//!     BOS + User + text + Assistant + `</think>`
//!
//! AX-owned. Not copied from mlx-optiq.

use std::fs;
use std::path::{Path, PathBuf};

use super::ConvertError;

pub const DEEPSEEK_V4_CHAT_TEMPLATE: &str = "\
{%- set bos = '<｜begin▁of▁sentence｜>' -%}
{%- set user_sp = '<｜User｜>' -%}
{%- set asst_sp = '<｜Assistant｜>' -%}
{%- if enable_thinking is not defined -%}
    {%- set enable_thinking = false -%}
{%- endif -%}
{{- bos -}}
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{- message['content'] -}}
    {%- elif message['role'] == 'user' -%}
        {{- user_sp + message['content'] -}}
    {%- elif message['role'] == 'assistant' -%}
        {{- message['content'] + '<｜end▁of▁sentence｜>' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- asst_sp -}}
    {%- if enable_thinking -%}
        {{- '<think>' -}}
    {%- else -%}
        {{- '</think>' -}}
    {%- endif -%}
{%- endif -%}
";

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

/// Write official V4 chat Jinja when the pack is DeepSeek V4 and has none.
pub fn ensure_deepseek_v4_chat_template(model_dir: &Path) -> Result<Option<PathBuf>, ConvertError> {
    if !dir_is_deepseek_v4(model_dir) {
        return Ok(None);
    }
    let path = model_dir.join("chat_template.jinja");
    if path.is_file() {
        let len = path.metadata().map(|meta| meta.len()).unwrap_or(0);
        if len > 0 {
            return Ok(Some(path));
        }
    }
    fs::write(&path, DEEPSEEK_V4_CHAT_TEMPLATE).map_err(|source| ConvertError::ReadFile {
        path: path.clone(),
        source,
    })?;
    Ok(Some(path))
}
