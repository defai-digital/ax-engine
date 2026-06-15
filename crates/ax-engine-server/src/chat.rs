use std::path::Path;

use ax_engine_sdk::{EngineTokenizer, EngineTokenizerError};
use serde_json::{Value, json};

// Gemma 4 closes every conversational turn with this token. Instruction-tuned
// artifacts list it in `generation_config.json`'s `eos_token_id`; base
// (pretrained) artifacts do not, which is how we tell them apart below.
const GEMMA4_TURN_TERMINATOR: &str = "<turn|>";

// Gemma 4 frames assistant reasoning as a channel: `<|channel>thought\n` opens
// a thinking channel and `<channel|>` closes it; the user-facing answer follows
// the close marker. The generation prompt below pre-fills an empty thought
// channel, but the model can still re-open one mid-generation, so chat
// responses must strip the framing (the reference chat template does the same
// with its `strip_thinking` macro when re-rendering history).
const GEMMA4_CHANNEL_OPEN: &str = "<|channel>";
const GEMMA4_CHANNEL_CLOSE: &str = "<channel|>";

// Pre-fills `<think>\n\n</think>\n\n` to signal the model to skip reasoning.
// This matches Qwen chat templates rendered with `enable_thinking=false` and
// keeps OpenAI-compatible short responses from spending the output budget on
// visible reasoning text.
pub(crate) const QWEN_CHATML_ASSISTANT_GENERATION_PROMPT: &str =
    "<|im_start|>assistant\n<think>\n\n</think>\n\n";
pub(crate) const QWEN_CHATML_ASSISTANT_GENERATION_PROMPT_THINKING: &str =
    "<|im_start|>assistant\n<think>\n";
pub(crate) const QWEN_CHATML_ASSISTANT_GENERATION_PROMPT_NO_THINK: &str = "<|im_start|>assistant\n";

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
    matches!(
        ChatPromptTemplate::for_model_id(model_id),
        ChatPromptTemplate::QwenChatMl | ChatPromptTemplate::Glm47
    )
    .then(|| json!({"enable_thinking": false}))
}

pub(crate) fn requires_chat_template_artifact(model_id: &str) -> bool {
    matches!(
        ChatPromptTemplate::for_model_id(model_id),
        ChatPromptTemplate::Gemma4
    )
}

pub(crate) fn validate_native_chat_artifact(
    model_id: &str,
    artifacts_dir: Option<&Path>,
) -> Result<(), String> {
    if !requires_chat_template_artifact(model_id) {
        return Ok(());
    }

    let Some(artifacts_dir) = artifacts_dir else {
        return Ok(());
    };
    if !artifacts_dir.join("chat_template.jinja").is_file() {
        return Err(format!(
            "Gemma4 chat requires an instruction-tuned MLX artifact with chat_template.jinja; \
             {} does not provide one. Use a gemma-4-*-it artifact or route raw base-model prompts through /v1/completions.",
            artifacts_dir.display()
        ));
    }

    // A `chat_template.jinja` can be copied into a base (pretrained) artifact,
    // which would let a non-instruction-tuned model through the check above and
    // emit garbage that never stops (it never produces `<turn|>`, so requests
    // run to max_tokens). Confirm the artifact is genuinely instruction-tuned by
    // requiring its `generation_config.json` to stop on the turn terminator.
    validate_gemma4_instruct_eos(artifacts_dir)
}

// Returns Ok only when `generation_config.json` lists the Gemma 4 turn
// terminator (`<turn|>`) as an end-of-sequence token, which every instruction-
// tuned Gemma 4 artifact does and base artifacts do not.
fn validate_gemma4_instruct_eos(artifacts_dir: &Path) -> Result<(), String> {
    let not_instruct = |detail: &str| {
        format!(
            "Gemma4 chat requires an instruction-tuned MLX artifact; {} looks like a base \
             (pretrained) model ({detail}). Use a gemma-4-*-it artifact or route raw base-model \
             prompts through /v1/completions.",
            artifacts_dir.display()
        )
    };

    let turn_terminator_id = gemma4_turn_terminator_id(artifacts_dir)
        .ok_or_else(|| not_instruct("tokenizer does not define the <turn|> chat token"))?;

    let gen_config_path = artifacts_dir.join("generation_config.json");
    let gen_config_text = std::fs::read_to_string(&gen_config_path)
        .map_err(|err| not_instruct(&format!("cannot read generation_config.json: {err}")))?;
    let gen_config: Value = serde_json::from_str(&gen_config_text)
        .map_err(|err| not_instruct(&format!("generation_config.json is not valid JSON: {err}")))?;

    let eos = gen_config.get("eos_token_id");
    if eos.is_some_and(|value| json_contains_u64(value, turn_terminator_id)) {
        Ok(())
    } else {
        Err(not_instruct(
            "generation_config.json eos_token_id does not stop on <turn|>",
        ))
    }
}

// Resolves the token id of `<turn|>` from the artifact's `tokenizer.json`
// added-token table, so the eos check does not hardcode a vocabulary id.
fn gemma4_turn_terminator_id(artifacts_dir: &Path) -> Option<u64> {
    let tokenizer_text = std::fs::read_to_string(artifacts_dir.join("tokenizer.json")).ok()?;
    let tokenizer: Value = serde_json::from_str(&tokenizer_text).ok()?;
    tokenizer
        .get("added_tokens")?
        .as_array()?
        .iter()
        .find(|token| token.get("content").and_then(Value::as_str) == Some(GEMMA4_TURN_TERMINATOR))
        .and_then(|token| token.get("id"))
        .and_then(Value::as_u64)
}

// `eos_token_id` may be a single integer or an array of integers.
fn json_contains_u64(value: &Value, target: u64) -> bool {
    match value {
        Value::Number(_) => value.as_u64() == Some(target),
        Value::Array(items) => items.iter().any(|item| item.as_u64() == Some(target)),
        _ => false,
    }
}

#[cfg(test)]
pub(crate) fn is_qwen_thinking_model(model_id: &str) -> bool {
    let m = model_id.to_ascii_lowercase();
    if !m.contains("qwen") {
        return false;
    }
    if m.contains("3.6") || m.contains("3_6") || m.contains("qwen36") {
        return true;
    }
    if m.contains("qwen3-coder-next") || m.contains("qwen3-coder") {
        return false;
    }
    let normalized: String = m
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect();
    normalized.contains("qwen3-next")
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
        ChatPromptTemplate::Gemma4 => vec![GEMMA4_TURN_TERMINATOR.to_string()],
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

    render_prompt_with_template_for_model(model_id, template, messages, false)
}

fn render_prompt_with_template_for_model(
    model_id: &str,
    template: ChatPromptTemplate,
    messages: &[(String, String)],
    qwen_thinking_enabled: bool,
) -> Result<String, String> {
    render_prompt_internal(
        template,
        messages,
        qwen_assistant_generation_prompt(model_id, qwen_thinking_enabled),
    )
}

pub(crate) fn render_prompt_with_template(
    template: ChatPromptTemplate,
    messages: &[(String, String)],
    qwen_thinking_enabled: bool,
) -> Result<String, String> {
    render_prompt_internal(
        template,
        messages,
        if qwen_thinking_enabled {
            QWEN_CHATML_ASSISTANT_GENERATION_PROMPT_THINKING
        } else {
            QWEN_CHATML_ASSISTANT_GENERATION_PROMPT
        },
    )
}

fn render_prompt_internal(
    template: ChatPromptTemplate,
    messages: &[(String, String)],
    qwen_generation_prompt: &str,
) -> Result<String, String> {
    let mut prompt = String::new();
    let mut qwen_tool_response_open = false;
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
                if matches!(role, "tool" | "function") {
                    if !qwen_tool_response_open {
                        prompt.push_str("<|im_start|>user\n");
                        qwen_tool_response_open = true;
                    }
                    prompt.push_str("<tool_response>\n");
                    prompt.push_str(content);
                    prompt.push_str("\n</tool_response>\n");
                } else {
                    if qwen_tool_response_open {
                        prompt.push_str("<|im_end|>\n");
                        qwen_tool_response_open = false;
                    }
                    prompt.push_str("<|im_start|>");
                    prompt.push_str(role);
                    prompt.push('\n');
                    prompt.push_str(content);
                    prompt.push_str("<|im_end|>\n");
                }
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
    if qwen_tool_response_open {
        prompt.push_str("<|im_end|>\n");
    }
    match template {
        ChatPromptTemplate::QwenChatMl => {
            prompt.push_str(qwen_generation_prompt);
        }
        ChatPromptTemplate::Llama3 => {
            prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        ChatPromptTemplate::Gemma4 => {
            prompt.push_str("<|turn>model\n<|channel>thought\n<channel|>")
        }
        ChatPromptTemplate::Glm47 => prompt.push_str("<|assistant|></think>"),
        ChatPromptTemplate::PlainRolePrefix => prompt.push_str("assistant:"),
        ChatPromptTemplate::Unsupported(_) => {
            unreachable!("unsupported templates are rejected before rendering")
        }
    }
    Ok(prompt)
}

fn qwen_assistant_generation_prompt(model_id: &str, thinking_enabled: bool) -> &'static str {
    if thinking_enabled {
        return QWEN_CHATML_ASSISTANT_GENERATION_PROMPT_THINKING;
    }
    if is_qwen_non_thinking_only_model(model_id) {
        return QWEN_CHATML_ASSISTANT_GENERATION_PROMPT_NO_THINK;
    }
    QWEN_CHATML_ASSISTANT_GENERATION_PROMPT
}

pub(crate) fn is_qwen_non_thinking_only_model(model_id: &str) -> bool {
    let normalized = model_id.to_ascii_lowercase();
    normalized == "qwen3"
        || normalized.contains("qwen3-coder-next")
        || normalized.contains("qwen3-coder")
}

/// Token ids of the Gemma 4 channel markers, looked up from the model's
/// tokenizer. `None` for tokenizers that do not define them (non-Gemma4
/// models), which disables channel stripping.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Gemma4ChannelIds {
    pub(crate) open: u32,
    pub(crate) close: u32,
}

impl Gemma4ChannelIds {
    pub(crate) fn from_tokenizer(tokenizer: &EngineTokenizer) -> Option<Self> {
        Some(Self {
            open: tokenizer.token_to_id(GEMMA4_CHANNEL_OPEN)?,
            close: tokenizer.token_to_id(GEMMA4_CHANNEL_CLOSE)?,
        })
    }
}

/// Split generated tokens into text kept outside channels and the bodies of
/// channel spans.
///
/// Two span shapes occur in practice:
/// - explicit `<|channel>…<channel|>` spans (an unterminated one runs to the
///   end of the output), and
/// - a stray `<channel|>` close with no preceding open: the generation prompt
///   pre-fills `<|channel>thought\n<channel|>`, and the model often continues
///   that channel anyway (re-emitting `thought\n` as plain text plus optional
///   reasoning) before closing it. Everything before such a stray close is
///   channel content, not the answer.
fn split_gemma4_channels(tokens: &[u32], ids: Gemma4ChannelIds) -> (Vec<u32>, Vec<Vec<u32>>) {
    let mut kept = Vec::with_capacity(tokens.len());
    let mut channel_bodies = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i] == ids.open {
            let body_start = i + 1;
            match tokens[body_start..].iter().position(|&t| t == ids.close) {
                Some(offset) => {
                    channel_bodies.push(tokens[body_start..body_start + offset].to_vec());
                    i = body_start + offset + 1;
                }
                None => {
                    channel_bodies.push(tokens[body_start..].to_vec());
                    i = tokens.len();
                }
            }
        } else if tokens[i] == ids.close {
            if !kept.is_empty() {
                channel_bodies.push(std::mem::take(&mut kept));
            }
            i += 1;
        } else {
            kept.push(tokens[i]);
            i += 1;
        }
    }
    (kept, channel_bodies)
}

/// Drop the channel-name header line (e.g. `thought`) from a decoded channel
/// body. Keeps the body intact when the first line does not look like a bare
/// channel name.
pub(crate) fn strip_gemma4_channel_name_header(body: &str) -> &str {
    let Some((name, rest)) = body.split_once('\n') else {
        return body;
    };
    let name = name.trim();
    if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        rest
    } else {
        body
    }
}

/// Decode Gemma 4 chat output tokens with thinking-channel framing removed.
///
/// Mirrors the reference template's `strip_thinking` semantics: channel spans
/// are dropped and only text outside them is served. When the model puts its
/// entire answer inside a channel (so stripping would serve an empty message),
/// fall back to the last channel body minus its channel-name header — the
/// useful content is there.
pub(crate) fn decode_gemma4_chat_output(
    tokenizer: &EngineTokenizer,
    output_tokens: &[u32],
) -> Result<String, EngineTokenizerError> {
    decode_gemma4_chat_output_with_reasoning(tokenizer, output_tokens)
        .map(|(content, _reasoning)| content)
}

/// Like [`decode_gemma4_chat_output`], but also returns the decoded channel
/// bodies (channel-name headers removed) so callers serving an explicit
/// reasoning contract can expose them instead of discarding them. When the
/// fallback serves the last channel body as content, that body is excluded
/// from the reasoning text.
pub(crate) fn decode_gemma4_chat_output_with_reasoning(
    tokenizer: &EngineTokenizer,
    output_tokens: &[u32],
) -> Result<(String, Option<String>), EngineTokenizerError> {
    let Some(ids) = Gemma4ChannelIds::from_tokenizer(tokenizer) else {
        return Ok((tokenizer.decode(output_tokens, true)?, None));
    };
    let (kept, channel_bodies) = split_gemma4_channels(output_tokens, ids);
    if channel_bodies.is_empty() {
        return Ok((tokenizer.decode(output_tokens, true)?, None));
    }
    let mut body_texts = Vec::with_capacity(channel_bodies.len());
    for body in &channel_bodies {
        let body_text = tokenizer.decode(body, true)?;
        body_texts.push(strip_gemma4_channel_name_header(&body_text).to_string());
    }
    let kept_text = tokenizer.decode(&kept, true)?;
    let content = if kept_text.trim().is_empty() {
        body_texts.pop().unwrap_or(kept_text)
    } else {
        kept_text
    };
    let reasoning = body_texts.join("\n");
    let reasoning = (!reasoning.trim().is_empty()).then(|| reasoning.trim().to_string());
    Ok((content, reasoning))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

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

    const CHANNEL_IDS: Gemma4ChannelIds = Gemma4ChannelIds {
        open: 100,
        close: 101,
    };

    #[test]
    fn split_gemma4_channels_keeps_text_outside_closed_channel() {
        // <|channel> name+thinking <channel|> answer
        let (kept, bodies) = split_gemma4_channels(&[100, 7, 8, 9, 101, 20, 21], CHANNEL_IDS);
        assert_eq!(kept, vec![20, 21]);
        assert_eq!(bodies, vec![vec![7, 8, 9]]);
    }

    #[test]
    fn split_gemma4_channels_captures_unclosed_channel_body() {
        let (kept, bodies) = split_gemma4_channels(&[5, 100, 7, 8, 9], CHANNEL_IDS);
        assert_eq!(kept, vec![5]);
        assert_eq!(bodies, vec![vec![7, 8, 9]]);
    }

    #[test]
    fn split_gemma4_channels_passes_through_plain_output() {
        let (kept, bodies) = split_gemma4_channels(&[5, 6, 7], CHANNEL_IDS);
        assert_eq!(kept, vec![5, 6, 7]);
        assert!(bodies.is_empty());
    }

    #[test]
    fn split_gemma4_channels_treats_stray_close_as_channel_boundary() {
        // The model continued the prompt's pre-filled thought channel ("thought\n…")
        // and closed it with a bare <channel|> before the answer.
        let (kept, bodies) = split_gemma4_channels(&[40, 41, 101, 20, 21], CHANNEL_IDS);
        assert_eq!(kept, vec![20, 21]);
        assert_eq!(bodies, vec![vec![40, 41]]);
    }

    #[test]
    fn channel_name_header_is_stripped_from_body() {
        assert_eq!(
            strip_gemma4_channel_name_header("thought\nThe quick brown fox."),
            "The quick brown fox."
        );
        // A first line that is not a bare channel name stays intact.
        assert_eq!(
            strip_gemma4_channel_name_header("First line of prose.\nSecond line."),
            "First line of prose.\nSecond line."
        );
        // Single-line bodies stay intact.
        assert_eq!(strip_gemma4_channel_name_header("answer"), "answer");
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

    fn write_gemma4_tokenizer(artifact_dir: &Path) {
        // Minimal tokenizer.json that defines the `<turn|>` chat token at the
        // real Gemma 4 vocabulary id.
        fs::write(
            artifact_dir.join("tokenizer.json"),
            r#"{"added_tokens":[{"id":1,"content":"<eos>"},{"id":106,"content":"<turn|>"}]}"#,
        )
        .expect("tokenizer.json should write");
    }

    #[test]
    fn native_chat_artifact_validation_rejects_gemma4_without_chat_template() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let artifact_dir =
            std::env::temp_dir().join(format!("ax-engine-chat-gemma4-artifact-{unique}"));
        fs::create_dir_all(&artifact_dir).expect("artifact dir should create");

        let error = validate_native_chat_artifact("gemma4", Some(&artifact_dir))
            .expect_err("Gemma4 chat artifact must include chat_template.jinja");
        assert!(error.contains("chat_template.jinja"));

        fs::write(artifact_dir.join("chat_template.jinja"), "{# template #}")
            .expect("template marker should write");
        write_gemma4_tokenizer(&artifact_dir);
        fs::write(
            artifact_dir.join("generation_config.json"),
            r#"{"eos_token_id":[1,106,50]}"#,
        )
        .expect("generation_config.json should write");
        validate_native_chat_artifact("gemma4", Some(&artifact_dir))
            .expect("instruction-tuned Gemma4 artifact should pass");
        validate_native_chat_artifact("glm4_moe_lite", Some(&artifact_dir))
            .expect("non-Gemma families do not require Gemma4 template");

        fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
    }

    #[test]
    fn native_chat_artifact_validation_rejects_base_gemma4_even_with_copied_template() {
        // Regression: a base (pretrained) Gemma 4 artifact cannot follow the
        // chat turn format and emits non-stopping garbage. Copying a
        // chat_template.jinja into it must not be enough to pass the guard;
        // the base `generation_config.json` does not stop on `<turn|>`.
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let artifact_dir =
            std::env::temp_dir().join(format!("ax-engine-chat-gemma4-base-{unique}"));
        fs::create_dir_all(&artifact_dir).expect("artifact dir should create");
        fs::write(artifact_dir.join("chat_template.jinja"), "{# copied #}")
            .expect("template marker should write");
        write_gemma4_tokenizer(&artifact_dir);
        fs::write(
            artifact_dir.join("generation_config.json"),
            r#"{"eos_token_id":1}"#,
        )
        .expect("generation_config.json should write");

        let error = validate_native_chat_artifact("gemma4", Some(&artifact_dir))
            .expect_err("base Gemma4 artifact must be rejected for chat");
        assert!(
            error.contains("instruction-tuned") && error.contains("<turn|>"),
            "unexpected error: {error}"
        );

        fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
    }

    #[test]
    fn qwen_no_thinking_suffix_is_default() {
        let messages = vec![("user".to_string(), "hi".to_string())];
        let default_prompt = render_prompt("Qwen3.6-35B-A3B-4bit", &messages).expect("render");
        assert!(default_prompt.ends_with(QWEN_CHATML_ASSISTANT_GENERATION_PROMPT));

        let thinking = render_prompt_with_template(ChatPromptTemplate::QwenChatMl, &messages, true)
            .expect("render");
        assert!(thinking.ends_with(QWEN_CHATML_ASSISTANT_GENERATION_PROMPT_THINKING));
        assert!(!thinking.contains("</think>"));

        let no_thinking =
            render_prompt_with_template(ChatPromptTemplate::QwenChatMl, &messages, false)
                .expect("render");
        assert!(no_thinking.ends_with(QWEN_CHATML_ASSISTANT_GENERATION_PROMPT));
    }

    #[test]
    fn qwen_thinking_model_match_is_limited_to_known_reasoning_families() {
        for model_id in [
            "qwen3_6-35b-a3b-4bit",
            "Qwen3.6-35B-A3B-4bit",
            "mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit",
        ] {
            assert!(is_qwen_thinking_model(model_id), "{model_id}");
        }

        for model_id in [
            "Qwen3-Coder-Next-4bit",
            "qwen-nextgen-v2",
            "qwen2.5-next-demo",
            "qwen4-next",
        ] {
            assert!(!is_qwen_thinking_model(model_id), "{model_id}");
            assert_eq!(
                template_kwargs_for_model_id(model_id),
                Some(json!({"enable_thinking": false})),
                "{model_id}"
            );
        }
    }
}
