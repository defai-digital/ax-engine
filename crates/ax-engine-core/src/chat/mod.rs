//! Chat prompt rendering helpers for instruction-tuned models.
//!
//! This module intentionally exposes a library-level rendering surface so
//! callers do not need to duplicate architecture-specific prompt wrappers.
//! If GGUF metadata carries a raw chat template string, it is surfaced via
//! [`gguf_chat_template`], but rendering currently uses the built-in family
//! renderers below.

use anyhow::{anyhow, ensure};

use crate::gguf::GgufHeader;
use crate::tokenizer::Tokenizer;

/// One structured chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatMessage<'a> {
    pub role: ChatRole,
    pub content: &'a str,
}

impl<'a> ChatMessage<'a> {
    pub fn new(role: ChatRole, content: &'a str) -> Self {
        Self { role, content }
    }

    pub fn system(content: &'a str) -> Self {
        Self::new(ChatRole::System, content)
    }

    pub fn user(content: &'a str) -> Self {
        Self::new(ChatRole::User, content)
    }

    pub fn assistant(content: &'a str) -> Self {
        Self::new(ChatRole::Assistant, content)
    }
}

/// Supported structured chat roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    fn header_name(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }

    fn gemma_name(self) -> &'static str {
        match self {
            Self::Assistant => "model",
            Self::System => "system",
            Self::User => "user",
        }
    }
}

/// Rendering options for chat prompt construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatRenderOptions {
    /// Append the model's assistant preamble for generation.
    pub add_generation_prompt: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfillPromptStyle {
    Pipe,
    Plain,
    Short,
}

impl InfillPromptStyle {
    fn control_tokens(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Self::Pipe => ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"),
            Self::Plain => ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"),
            Self::Short => ("<PRE>", "<SUF>", "<MID>"),
        }
    }
}

impl Default for ChatRenderOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
        }
    }
}

/// Return the raw GGUF chat template string if present.
///
/// Known keys vary across exporters, so this checks the common locations.
pub fn gguf_chat_template(header: &GgufHeader) -> Option<&str> {
    [
        "tokenizer.chat_template",
        "tokenizer.ggml.chat_template",
        "general.chat_template",
    ]
    .into_iter()
    .find_map(|key| header.get_str(key))
}

/// Render a structured chat transcript for a model family.
pub fn render_chat_messages(
    messages: &[ChatMessage<'_>],
    architecture: &str,
    options: ChatRenderOptions,
) -> String {
    match architecture {
        "gemma3" | "gemma4" | "gemma2" | "gemma" => render_gemma(messages, options),
        "qwen35" | "qwen35moe" | "qwen3moe" | "qwen3" | "qwen2" => render_qwen(messages, options),
        _ => render_fallback(messages, options),
    }
}

/// Render a single user message as a chat prompt for generation.
pub fn render_user_prompt(prompt: &str, architecture: &str) -> String {
    render_chat_messages(
        &[ChatMessage::user(prompt)],
        architecture,
        ChatRenderOptions::default(),
    )
}

/// Render one user turn, preserving the legacy first-turn vs follow-up behavior.
pub fn render_user_turn(prompt: &str, architecture: &str, first_turn: bool) -> String {
    if first_turn {
        return render_user_prompt(prompt, architecture);
    }

    match architecture {
        "gemma3" | "gemma4" | "gemma2" | "gemma" => render_chat_messages(
            &[ChatMessage::user(&format!("\n{prompt}"))],
            architecture,
            ChatRenderOptions::default(),
        ),
        _ => render_user_prompt(prompt, architecture),
    }
}

pub fn detect_infill_prompt_style(tokenizer: &Tokenizer) -> Option<InfillPromptStyle> {
    [
        InfillPromptStyle::Pipe,
        InfillPromptStyle::Plain,
        InfillPromptStyle::Short,
    ]
    .into_iter()
    .find(|style| {
        let (prefix, suffix, middle) = style.control_tokens();
        [prefix, suffix, middle]
            .into_iter()
            .all(|token| tokenizer.encode_with_options(token, false, true).len() == 1)
    })
}

pub fn render_infill_prompt(
    prefix: &str,
    suffix: &str,
    tokenizer: &Tokenizer,
) -> anyhow::Result<String> {
    ensure!(
        !(prefix.is_empty() && suffix.is_empty()),
        "infill requires a non-empty prefix or suffix"
    );
    let style = detect_infill_prompt_style(tokenizer)
        .ok_or_else(|| anyhow!("loaded model does not support infill"))?;
    let (prefix_token, suffix_token, middle_token) = style.control_tokens();
    Ok(format!(
        "{prefix_token}{prefix}{suffix_token}{suffix}{middle_token}"
    ))
}

fn render_gemma(messages: &[ChatMessage<'_>], options: ChatRenderOptions) -> String {
    let mut rendered = String::new();
    for message in messages {
        rendered.push_str("<start_of_turn>");
        rendered.push_str(message.role.gemma_name());
        rendered.push('\n');
        rendered.push_str(message.content);
        rendered.push_str("<end_of_turn>\n");
    }
    if options.add_generation_prompt {
        rendered.push_str("<start_of_turn>model\n");
    }
    rendered
}

fn render_qwen(messages: &[ChatMessage<'_>], options: ChatRenderOptions) -> String {
    let mut rendered = String::new();
    for message in messages {
        rendered.push_str("<|im_start|>");
        rendered.push_str(message.role.header_name());
        rendered.push('\n');
        rendered.push_str(message.content);
        rendered.push_str("<|im_end|>\n");
    }
    if options.add_generation_prompt {
        rendered.push_str("<|im_start|>assistant\n");
    }
    rendered
}

fn render_fallback(messages: &[ChatMessage<'_>], options: ChatRenderOptions) -> String {
    if let [message] = messages {
        return if options.add_generation_prompt && message.role == ChatRole::User {
            message.content.to_string()
        } else {
            format!("{}: {}", message.role.header_name(), message.content)
        };
    }

    let mut rendered = String::new();
    for message in messages {
        if !rendered.is_empty() {
            rendered.push('\n');
        }
        rendered.push_str(message.role.header_name());
        rendered.push_str(": ");
        rendered.push_str(message.content);
    }
    rendered
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::gguf::MetadataValue;
    use crate::tokenizer::{Tokenizer, Vocab, vocab::TokenType};

    use super::*;

    #[test]
    fn test_gguf_chat_template_reads_common_key() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.chat_template".to_string(),
            MetadataValue::String("{{- test -}}".to_string()),
        );
        let header = GgufHeader {
            version: 3,
            tensor_count: 0,
            metadata,
        };

        assert_eq!(gguf_chat_template(&header), Some("{{- test -}}"));
    }

    #[test]
    fn test_render_chat_messages_qwen() {
        let rendered = render_chat_messages(
            &[ChatMessage::system("Be terse"), ChatMessage::user("Hi")],
            "qwen3",
            ChatRenderOptions::default(),
        );
        assert_eq!(
            rendered,
            "<|im_start|>system\nBe terse<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_render_chat_messages_qwen35_matches_qwen_generation_prefix() {
        let rendered = render_chat_messages(
            &[ChatMessage::system("Be terse"), ChatMessage::user("Hi")],
            "qwen35moe",
            ChatRenderOptions::default(),
        );
        assert_eq!(
            rendered,
            "<|im_start|>system\nBe terse<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_render_chat_messages_gemma() {
        let rendered = render_chat_messages(
            &[ChatMessage::user("Hi"), ChatMessage::assistant("Hello")],
            "gemma3",
            ChatRenderOptions {
                add_generation_prompt: false,
            },
        );
        assert_eq!(
            rendered,
            "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\nHello<end_of_turn>\n"
        );
    }

    #[test]
    fn test_render_user_turn_preserves_follow_up_prefix_for_gemma() {
        let rendered = render_user_turn("Hi", "gemma3", false);
        assert_eq!(
            rendered,
            "<start_of_turn>user\n\nHi<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    fn make_infill_tokenizer(
        prefix_token: &str,
        suffix_token: &str,
        middle_token: &str,
    ) -> Tokenizer {
        let token_defs: Vec<(&str, f32, TokenType)> = vec![
            ("<unk>", 0.0, TokenType::Unknown),
            ("<s>", 0.0, TokenType::Control),
            ("</s>", 0.0, TokenType::Control),
            (prefix_token, 0.0, TokenType::Control),
            (suffix_token, 0.0, TokenType::Control),
            (middle_token, 0.0, TokenType::Control),
            ("body", -1.0, TokenType::Normal),
        ];
        let tokens: Vec<String> = token_defs.iter().map(|(t, _, _)| t.to_string()).collect();
        let scores: Vec<f32> = token_defs.iter().map(|(_, s, _)| *s).collect();
        let types: Vec<TokenType> = token_defs.iter().map(|(_, _, ty)| *ty).collect();
        let mut token_to_id = HashMap::new();
        for (i, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), i as u32);
        }
        Tokenizer::from_vocab(Vocab {
            tokens,
            scores,
            types,
            token_to_id,
            merge_ranks: None,
            bos_id: 1,
            eos_id: 2,
            unk_id: 0,
            add_bos: false,
            add_eos: false,
            add_space_prefix: false,
            model_type: "llama".to_string(),
            eot_id: None,
        })
    }

    #[test]
    fn test_detect_infill_prompt_style_pipe_tokens() {
        let tokenizer = make_infill_tokenizer("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>");
        assert_eq!(
            detect_infill_prompt_style(&tokenizer),
            Some(InfillPromptStyle::Pipe)
        );
    }

    #[test]
    fn test_detect_infill_prompt_style_plain_tokens() {
        let tokenizer = make_infill_tokenizer("<fim_prefix>", "<fim_suffix>", "<fim_middle>");
        assert_eq!(
            detect_infill_prompt_style(&tokenizer),
            Some(InfillPromptStyle::Plain)
        );
    }

    #[test]
    fn test_render_infill_prompt_short_tokens() {
        let tokenizer = make_infill_tokenizer("<PRE>", "<SUF>", "<MID>");
        let rendered = render_infill_prompt("fn foo() {\n", "\n}\n", &tokenizer).unwrap();
        assert_eq!(rendered, "<PRE>fn foo() {\n<SUF>\n}\n<MID>");
    }
}
