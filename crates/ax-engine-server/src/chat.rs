use serde_json::{Value, json};

// Pre-fills `<think>\n\n</think>\n\n` to signal the model to skip reasoning;
// applied to Qwen models whose chat templates default to this when
// `enable_thinking=false`. For reasoning-trained Qwen3.6 / Qwen3-Next /
// Qwen3-Coder-Next this is the wrong prefix (see #13), so those models use
// `QWEN_CHATML_ASSISTANT_GENERATION_PROMPT_THINKING` instead.
pub(crate) const QWEN_CHATML_ASSISTANT_GENERATION_PROMPT: &str =
    "<|im_start|>assistant\n<think>\n\n</think>\n\n";
pub(crate) const QWEN_CHATML_ASSISTANT_GENERATION_PROMPT_THINKING: &str =
    "<|im_start|>assistant\n<think>\n";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ChatPromptTemplate {
    QwenChatMl,
    Llama3,
    Gemma4,
    Glm47,
    Unsupported(ChatUnsupportedFamily),
    PlainRolePrefix,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ChatUnsupportedFamily {
    Gemma3,
    Llama4,
    Mistral3,
    Mixtral,
    DeepSeek,
}

impl ChatUnsupportedFamily {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Gemma3 => "gemma3",
            Self::Llama4 => "llama4",
            Self::Mistral3 => "mistral3",
            Self::Mixtral => "mixtral",
            Self::DeepSeek => "deepseek",
        }
    }
}

impl ChatPromptTemplate {
    pub(crate) fn for_model_id(model_id: &str) -> Self {
        let normalized = model_id.to_ascii_lowercase();
        if normalized.contains("qwen") {
            Self::QwenChatMl
        } else if normalized.contains("gemma-4") || normalized.contains("gemma4") {
            Self::Gemma4
        } else if normalized.contains("gemma-3") || normalized.contains("gemma3") {
            Self::Unsupported(ChatUnsupportedFamily::Gemma3)
        } else if normalized.contains("glm") {
            Self::Glm47
        } else if normalized.contains("llama-4")
            || normalized.contains("llama4")
            || normalized.contains("llama_4")
        {
            Self::Unsupported(ChatUnsupportedFamily::Llama4)
        } else if normalized.contains("llama-3")
            || normalized.contains("llama3")
            || normalized.contains("llama_3")
        {
            Self::Llama3
        } else if normalized.contains("mixtral") {
            Self::Unsupported(ChatUnsupportedFamily::Mixtral)
        } else if normalized.contains("mistral-3") || normalized.contains("mistral3") {
            Self::Unsupported(ChatUnsupportedFamily::Mistral3)
        } else if normalized.contains("deepseek") {
            Self::Unsupported(ChatUnsupportedFamily::DeepSeek)
        } else {
            Self::PlainRolePrefix
        }
    }
}

pub(crate) fn normalize_role(role: &str) -> Result<&'static str, String> {
    match role.trim() {
        "system" => Ok("system"),
        "user" => Ok("user"),
        "assistant" => Ok("assistant"),
        "tool" => Ok("tool"),
        "function" => Ok("function"),
        _ => Err(
            "unsupported chat role; expected one of system, user, assistant, tool, function"
                .to_string(),
        ),
    }
}

pub(crate) fn template_kwargs_for_model_id(model_id: &str) -> Option<Value> {
    // Qwen3.6 / Qwen3-Next / Qwen3-Coder-Next are reasoning models: their chat
    // templates inject `<think>\n\n</think>\n\n` when `enable_thinking=false`,
    // forcing the model to skip the reasoning step it was trained to produce.
    // On prompts that require reasoning (e.g. math), the model emits a short
    // preamble followed by `<|im_end|>` (in the stop sequence list) and the
    // response truncates after a handful of tokens. Leave the kwarg unset so
    // the template's default (thinking enabled) applies.
    if is_qwen_thinking_model(model_id) {
        return None;
    }
    matches!(
        ChatPromptTemplate::for_model_id(model_id),
        ChatPromptTemplate::QwenChatMl | ChatPromptTemplate::Glm47
    )
    .then(|| json!({"enable_thinking": false}))
}

pub(crate) fn is_qwen_thinking_model(model_id: &str) -> bool {
    let m = model_id.to_ascii_lowercase();
    m.contains("qwen") && (m.contains("3.6") || m.contains("3_6") || m.contains("next"))
}

pub(crate) fn stop_sequences(model_id: &str, mut user_stops: Vec<String>) -> Vec<String> {
    for native_stop in default_stop_sequences(ChatPromptTemplate::for_model_id(model_id)) {
        if !user_stops.iter().any(|existing| existing == &native_stop) {
            user_stops.push(native_stop);
        }
    }
    user_stops
}

pub(crate) fn default_stop_sequences(template: ChatPromptTemplate) -> Vec<String> {
    match template {
        ChatPromptTemplate::QwenChatMl => vec!["<|im_end|>".to_string()],
        ChatPromptTemplate::Llama3 => vec!["<|eot_id|>".to_string()],
        ChatPromptTemplate::Gemma4 => vec!["<turn|>".to_string()],
        ChatPromptTemplate::Glm47 => vec![
            "<|endoftext|>".to_string(),
            "<|user|>".to_string(),
            "<|observation|>".to_string(),
        ],
        ChatPromptTemplate::Unsupported(_) => Vec::new(),
        ChatPromptTemplate::PlainRolePrefix => Vec::new(),
    }
}

pub(crate) fn render_prompt(
    model_id: &str,
    messages: &[(String, String)],
) -> Result<String, String> {
    if messages.is_empty() {
        return Err("chat.completions requires at least one message".to_string());
    }

    let template = ChatPromptTemplate::for_model_id(model_id);
    if let ChatPromptTemplate::Unsupported(family) = template {
        return Err(format!(
            "OpenAI chat fallback for {family} is not supported yet; use a backend-native chat path or add a verified AX chat-template fixture before serving this family through raw completion fallback",
            family = family.label()
        ));
    }

    render_prompt_with_template(template, messages, is_qwen_thinking_model(model_id))
}

pub(crate) fn render_prompt_with_template(
    template: ChatPromptTemplate,
    messages: &[(String, String)],
    qwen_thinking_enabled: bool,
) -> Result<String, String> {
    let mut prompt = String::new();
    match template {
        ChatPromptTemplate::Llama3 => prompt.push_str("<|begin_of_text|>"),
        ChatPromptTemplate::Gemma4 => prompt.push_str("<bos>"),
        ChatPromptTemplate::Glm47 => prompt.push_str("[gMASK]<sop>"),
        ChatPromptTemplate::QwenChatMl | ChatPromptTemplate::PlainRolePrefix => {}
        ChatPromptTemplate::Unsupported(family) => {
            return Err(format!(
                "OpenAI chat fallback for {family} is not supported yet",
                family = family.label()
            ));
        }
    }
    for (role, content) in messages {
        let role = normalize_role(role)?;
        match template {
            ChatPromptTemplate::QwenChatMl => {
                prompt.push_str("<|im_start|>");
                prompt.push_str(role);
                prompt.push('\n');
                prompt.push_str(content);
                prompt.push_str("<|im_end|>\n");
            }
            ChatPromptTemplate::Llama3 => {
                prompt.push_str("<|start_header_id|>");
                prompt.push_str(role);
                prompt.push_str("<|end_header_id|>\n\n");
                prompt.push_str(content);
                prompt.push_str("<|eot_id|>");
            }
            ChatPromptTemplate::Gemma4 => {
                let turn = if role == "assistant" { "model" } else { role };
                prompt.push_str("<|turn>");
                prompt.push_str(turn);
                prompt.push('\n');
                prompt.push_str(content);
                prompt.push_str("<turn|>\n");
            }
            ChatPromptTemplate::Glm47 => {
                if matches!(role, "tool" | "function") {
                    prompt.push_str("<|observation|><tool_response>");
                    prompt.push_str(content);
                    prompt.push_str("</tool_response>");
                } else {
                    let tag = match role {
                        "assistant" => "<|assistant|>",
                        "system" => "<|system|>",
                        _ => "<|user|>",
                    };
                    prompt.push_str(tag);
                    if role == "assistant" {
                        prompt.push_str("</think>");
                        prompt.push_str(content.trim());
                    } else {
                        prompt.push_str(content);
                    }
                }
            }
            ChatPromptTemplate::PlainRolePrefix => {
                prompt.push_str(role);
                prompt.push_str(": ");
                prompt.push_str(content);
                prompt.push('\n');
            }
            ChatPromptTemplate::Unsupported(_) => {
                unreachable!("unsupported templates are rejected before rendering")
            }
        }
    }
    match template {
        ChatPromptTemplate::QwenChatMl => {
            if qwen_thinking_enabled {
                prompt.push_str(QWEN_CHATML_ASSISTANT_GENERATION_PROMPT_THINKING);
            } else {
                prompt.push_str(QWEN_CHATML_ASSISTANT_GENERATION_PROMPT);
            }
        }
        ChatPromptTemplate::Llama3 => {
            prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        ChatPromptTemplate::Gemma4 => prompt.push_str("<|turn>model\n"),
        ChatPromptTemplate::Glm47 => prompt.push_str("<|assistant|></think>"),
        ChatPromptTemplate::PlainRolePrefix => prompt.push_str("assistant:"),
        ChatPromptTemplate::Unsupported(_) => {
            unreachable!("unsupported templates are rejected before rendering")
        }
    }
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_sequences_merge_user_and_native_stops() {
        assert_eq!(
            stop_sequences("qwen3", vec!["custom".to_string()]),
            vec!["custom".to_string(), "<|im_end|>".to_string()]
        );
        assert_eq!(
            stop_sequences("Meta-Llama-3.1-8B-Instruct", vec!["<|eot_id|>".to_string()]),
            vec!["<|eot_id|>".to_string()]
        );
    }

    #[test]
    fn unsupported_known_families_fail_closed() {
        let messages = vec![("user".to_string(), "hello".to_string())];
        for model_id in [
            "google/gemma-3-4b-it",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "mistral3-small",
            "mixtral-8x7b-instruct",
            "deepseek-ai/DeepSeek-V3",
        ] {
            let error = render_prompt(model_id, &messages).expect_err("family should be rejected");
            assert!(
                error.contains("not supported yet"),
                "unexpected error for {model_id}: {error}"
            );
        }
    }

    #[test]
    fn qwen_thinking_suffix_is_explicit() {
        let messages = vec![("user".to_string(), "hi".to_string())];
        let thinking = render_prompt_with_template(ChatPromptTemplate::QwenChatMl, &messages, true)
            .expect("render");
        assert!(thinking.ends_with(QWEN_CHATML_ASSISTANT_GENERATION_PROMPT_THINKING));
        assert!(!thinking.contains("</think>"));

        let no_thinking =
            render_prompt_with_template(ChatPromptTemplate::QwenChatMl, &messages, false)
                .expect("render");
        assert!(no_thinking.ends_with(QWEN_CHATML_ASSISTANT_GENERATION_PROMPT));
    }
}
