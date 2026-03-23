//! Chat prompt rendering helpers for instruction-tuned models.
//!
//! This module intentionally exposes a library-level rendering surface so
//! callers do not need to duplicate architecture-specific prompt wrappers.
//! If GGUF metadata carries a raw chat template string, it is surfaced via
//! [`gguf_chat_template`], but rendering currently uses the built-in family
//! renderers below.

use crate::gguf::GgufHeader;

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
        "gemma3" | "gemma2" | "gemma" => render_gemma(messages, options),
        "llama" => render_llama(messages, options),
        "qwen3" | "qwen2" => render_qwen(messages, options),
        "phi3" => render_phi3(messages, options),
        "falcon" => render_falcon(messages, options),
        "mistral" | "mixtral" => render_mistral(messages),
        "chatglm" | "glm4" | "glm" => render_glm(messages, options),
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
        "gemma3" | "gemma2" | "gemma" | "falcon" => render_chat_messages(
            &[ChatMessage::user(&format!("\n{prompt}"))],
            architecture,
            ChatRenderOptions::default(),
        ),
        _ => render_user_prompt(prompt, architecture),
    }
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

fn render_llama(messages: &[ChatMessage<'_>], options: ChatRenderOptions) -> String {
    let mut rendered = String::new();
    for message in messages {
        rendered.push_str("<|start_header_id|>");
        rendered.push_str(message.role.header_name());
        rendered.push_str("<|end_header_id|>\n\n");
        rendered.push_str(message.content);
        rendered.push_str("<|eot_id|>");
    }
    if options.add_generation_prompt {
        rendered.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
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

fn render_phi3(messages: &[ChatMessage<'_>], options: ChatRenderOptions) -> String {
    let mut rendered = String::new();
    for message in messages {
        rendered.push_str(match message.role {
            ChatRole::System => "<|system|>\n",
            ChatRole::User => "<|user|>\n",
            ChatRole::Assistant => "<|assistant|>\n",
        });
        rendered.push_str(message.content);
        rendered.push_str("<|end|>\n");
    }
    if options.add_generation_prompt {
        rendered.push_str("<|assistant|>\n");
    }
    rendered
}

fn render_glm(messages: &[ChatMessage<'_>], options: ChatRenderOptions) -> String {
    let mut rendered = String::new();
    for message in messages {
        rendered.push_str(match message.role {
            ChatRole::System => "<|system|>\n",
            ChatRole::User => "<|user|>\n",
            ChatRole::Assistant => "<|assistant|>\n",
        });
        rendered.push_str(message.content);
    }
    if options.add_generation_prompt {
        rendered.push_str("<|assistant|>");
    }
    rendered
}

fn render_falcon(messages: &[ChatMessage<'_>], options: ChatRenderOptions) -> String {
    let mut rendered = String::new();
    for message in messages {
        let prefix = match message.role {
            ChatRole::System => "System",
            ChatRole::User => "User",
            ChatRole::Assistant => "Assistant",
        };
        if !rendered.is_empty() {
            rendered.push('\n');
        }
        rendered.push_str(prefix);
        rendered.push_str(": ");
        rendered.push_str(message.content);
    }
    if options.add_generation_prompt {
        if !rendered.is_empty() {
            rendered.push('\n');
        }
        rendered.push_str("Assistant:");
    }
    rendered
}

fn render_mistral(messages: &[ChatMessage<'_>]) -> String {
    let mut rendered = String::new();
    let mut pending_system: Option<&str> = None;

    for message in messages {
        match message.role {
            ChatRole::System => {
                pending_system = Some(message.content);
            }
            ChatRole::User => {
                if !rendered.is_empty() {
                    rendered.push(' ');
                }
                rendered.push_str("[INST] ");
                if let Some(system) = pending_system.take() {
                    rendered.push_str("<<SYS>>\n");
                    rendered.push_str(system);
                    rendered.push_str("\n<</SYS>>\n\n");
                }
                rendered.push_str(message.content);
                rendered.push_str(" [/INST]");
            }
            ChatRole::Assistant => {
                if !rendered.is_empty() {
                    rendered.push(' ');
                }
                rendered.push_str(message.content);
            }
        }
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
    fn test_render_user_prompt_llama() {
        let rendered = render_user_prompt("Hi", "llama");
        assert_eq!(
            rendered,
            "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
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
}
