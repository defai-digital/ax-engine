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

// GPT-OSS Harmony control tokens (OpenAI Harmony / gpt-oss chat format).
// Distinct from Gemma 4's `<|channel>` / `<channel|>` markers.
const GPT_OSS_CHANNEL: &str = "<|channel|>";
const GPT_OSS_MESSAGE: &str = "<|message|>";
const GPT_OSS_END: &str = "<|end|>";
const GPT_OSS_RETURN: &str = "<|return|>";
const GPT_OSS_START: &str = "<|start|>";
const GPT_OSS_CALL: &str = "<|call|>";

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
    /// Llama 3.x Instruct: `<|start_header_id|>` / `<|eot_id|>`.
    Llama3,
    /// Llama 4 Instruct: `<|header_start|>` / `<|eot|>` (not Llama 3 markers).
    Llama4,
    /// Gemma 4 IT (empty thought prefill when thinking is off).
    Gemma4,
    Glm47,
    /// Devstral / Mistral Small-style: `[SYSTEM_PROMPT]…[/SYSTEM_PROMPT][INST]…[/INST]`.
    MistralInstruct,
    /// Ministral classic Instruct: system folded into the first `[INST]` block.
    MinistralInstruct,
    /// OpenAI Harmony format used by GPT-OSS (`<|start|>…<|message|>…<|end|>`).
    GptOssHarmony,
    Unsupported(ChatUnsupportedFamily),
    PlainRolePrefix,
}

/// Default system identity block for GPT-OSS Harmony (matches mlx-community templates).
const GPT_OSS_DEFAULT_SYSTEM: &str = "\
You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: low

# Valid channels: analysis, commentary, final. Channel must be included for every message.";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ChatUnsupportedFamily {
    Gemma3,
    Mixtral,
    DeepSeek,
}

impl ChatUnsupportedFamily {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Gemma3 => "gemma3",
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
        } else if normalized.contains("gemma-4")
            || normalized.contains("gemma4")
            || normalized.contains("diffusiongemma")
            || normalized.contains("diffusion-gemma")
            || normalized.contains("diffusion_gemma")
        {
            // DiffusionGemma shares turn markers with Gemma 4 IT; generation
            // prefill differs (no empty thought channel) and is specialized
            // via `is_diffusion_gemma` in the renderers.
            Self::Gemma4
        } else if normalized.contains("gemma-3") || normalized.contains("gemma3") {
            Self::Unsupported(ChatUnsupportedFamily::Gemma3)
        } else if normalized.contains("glm") {
            Self::Glm47
        } else if normalized.contains("llama-4")
            || normalized.contains("llama4")
            || normalized.contains("llama_4")
        {
            Self::Llama4
        } else if normalized.contains("llama-3")
            || normalized.contains("llama3")
            || normalized.contains("llama_3")
        {
            Self::Llama3
        } else if normalized.contains("mixtral") {
            Self::Unsupported(ChatUnsupportedFamily::Mixtral)
        } else if normalized.contains("ministral") {
            Self::MinistralInstruct
        } else if normalized.contains("mistral")
            || normalized.contains("devstral")
            || normalized.contains("codestral")
        {
            Self::MistralInstruct
        } else if normalized.contains("deepseek") {
            Self::Unsupported(ChatUnsupportedFamily::DeepSeek)
        } else if normalized.contains("gpt-oss")
            || normalized.contains("gpt_oss")
            || normalized.contains("gptoss")
        {
            Self::GptOssHarmony
        } else {
            // Families without a verified AX chat fixture use plain role prefixes.
            Self::PlainRolePrefix
        }
    }
}

/// DiffusionGemma IT uses Gemma 4 turn markers but does **not** pre-fill an
/// empty thought channel on the generation prompt (hub chat_template.jinja).
pub(crate) fn is_diffusion_gemma(model_id: &str) -> bool {
    let normalized = model_id.to_ascii_lowercase();
    normalized.contains("diffusiongemma")
        || normalized.contains("diffusion-gemma")
        || normalized.contains("diffusion_gemma")
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
    // Gemma 4 / Qwen / GLM all default thinking off for OpenAI-compatible short
    // answers. Callers that want thinking must opt in via the request
    // `reasoning` field (native) or chat_template_kwargs (delegated).
    matches!(
        ChatPromptTemplate::for_model_id(model_id),
        ChatPromptTemplate::QwenChatMl | ChatPromptTemplate::Glm47 | ChatPromptTemplate::Gemma4
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
        ChatPromptTemplate::Llama4 => vec!["<|eot|>".to_string()],
        ChatPromptTemplate::Gemma4 => vec![GEMMA4_TURN_TERMINATOR.to_string()],
        ChatPromptTemplate::Glm47 => vec![
            "<|endoftext|>".to_string(),
            "<|user|>".to_string(),
            "<|observation|>".to_string(),
        ],
        ChatPromptTemplate::MistralInstruct | ChatPromptTemplate::MinistralInstruct => {
            vec!["</s>".to_string(), "[INST]".to_string()]
        }
        // Generation ends on <|return|>; also stop if the model closes a turn
        // with <|end|> then starts another role (hallucinated multi-turn).
        ChatPromptTemplate::GptOssHarmony => {
            vec!["<|return|>".to_string(), "<|end|><|start|>".to_string()]
        }
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
        model_id,
        template,
        messages,
        qwen_assistant_generation_prompt(model_id, qwen_thinking_enabled),
    )
}

#[cfg(test)]
pub(crate) fn render_prompt_with_template(
    template: ChatPromptTemplate,
    messages: &[(String, String)],
    qwen_thinking_enabled: bool,
) -> Result<String, String> {
    render_prompt_internal(
        "qwen3",
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
    model_id: &str,
    template: ChatPromptTemplate,
    messages: &[(String, String)],
    qwen_generation_prompt: &str,
) -> Result<String, String> {
    let mut prompt = String::new();
    let mut qwen_tool_response_open = false;
    match template {
        ChatPromptTemplate::Llama3 | ChatPromptTemplate::Llama4 => {
            prompt.push_str("<|begin_of_text|>")
        }
        ChatPromptTemplate::Gemma4 => prompt.push_str("<bos>"),
        ChatPromptTemplate::Glm47 => prompt.push_str("[gMASK]<sop>"),
        ChatPromptTemplate::MistralInstruct | ChatPromptTemplate::MinistralInstruct => {
            prompt.push_str("<s>")
        }
        ChatPromptTemplate::GptOssHarmony => {
            // Always emit the Harmony system header first (identity + channels).
            // Intentional AX delta vs hub jinja: Reasoning: low + no live date,
            // and generation prefills final channel (below) for short answers.
            prompt.push_str("<|start|>system<|message|>");
            prompt.push_str(GPT_OSS_DEFAULT_SYSTEM);
            prompt.push_str("<|end|>");
        }
        ChatPromptTemplate::QwenChatMl | ChatPromptTemplate::PlainRolePrefix => {}
        ChatPromptTemplate::Unsupported(family) => {
            return Err(format!(
                "OpenAI chat fallback for {family} is not supported yet",
                family = family.label()
            ));
        }
    }
    let mut mistral_system: Option<&str> = None;
    let mut ministral_system: Option<&str> = None;
    let mut gpt_oss_system: Option<&str> = None;
    // Ministral hub template folds system into the *last* user turn, not the first.
    let ministral_last_user_idx = if matches!(template, ChatPromptTemplate::MinistralInstruct) {
        messages.iter().rposition(|(role, _)| matches!(normalize_role(role), Ok("user")))
    } else {
        None
    };
    for (msg_index, (role, content)) in messages.iter().enumerate() {
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
                // Intentional AX delta: omit Llama's default knowledge-date
                // system preamble when the caller did not supply a system msg.
                prompt.push_str("<|start_header_id|>");
                prompt.push_str(role);
                prompt.push_str("<|end_header_id|>\n\n");
                prompt.push_str(content);
                prompt.push_str("<|eot_id|>");
            }
            ChatPromptTemplate::Llama4 => {
                prompt.push_str("<|header_start|>");
                prompt.push_str(role);
                prompt.push_str("<|header_end|>\n\n");
                prompt.push_str(content);
                prompt.push_str("<|eot|>");
            }
            ChatPromptTemplate::Gemma4 => {
                let turn = if role == "assistant" { "model" } else { role };
                prompt.push_str("<|turn>");
                prompt.push_str(turn);
                prompt.push('\n');
                // Match the official template's strip_thinking macro: prior
                // model turns must not re-inject channel framing into prefill.
                if role == "assistant" {
                    prompt.push_str(&strip_gemma4_thinking_from_history(content));
                } else {
                    prompt.push_str(content);
                }
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
            ChatPromptTemplate::MistralInstruct => {
                // Devstral hub: [SYSTEM_PROMPT]…[/SYSTEM_PROMPT][INST]…[/INST]
                // assistant turns append eos_token (`</s>`).
                match role {
                    "system" => {
                        mistral_system = Some(content.as_str());
                    }
                    "user" => {
                        if let Some(system) = mistral_system.take() {
                            prompt.push_str("[SYSTEM_PROMPT]");
                            prompt.push_str(system);
                            prompt.push_str("[/SYSTEM_PROMPT]");
                        }
                        prompt.push_str("[INST]");
                        prompt.push_str(content);
                        prompt.push_str("[/INST]");
                    }
                    "assistant" => {
                        prompt.push_str(content);
                        prompt.push_str("</s>");
                    }
                    "tool" | "function" => {
                        prompt.push_str("[INST]");
                        prompt.push_str(content);
                        prompt.push_str("[/INST]");
                    }
                    _ => {}
                }
            }
            ChatPromptTemplate::MinistralInstruct => {
                // Ministral hub: system folds into the *last* user [INST];
                // assistant is ` ` + content + eos_token (`</s>`).
                match role {
                    "system" => {
                        ministral_system = Some(content.as_str());
                    }
                    "user" => {
                        prompt.push_str("[INST]");
                        if ministral_last_user_idx == Some(msg_index)
                            && let Some(system) = ministral_system.take()
                        {
                            prompt.push_str(system);
                            prompt.push_str("\n\n");
                        }
                        prompt.push_str(content);
                        prompt.push_str("[/INST]");
                    }
                    "assistant" => {
                        prompt.push(' ');
                        prompt.push_str(content.trim());
                        prompt.push_str("</s>");
                    }
                    "tool" | "function" => {
                        prompt.push_str("[INST]");
                        prompt.push_str(content);
                        prompt.push_str("[/INST]");
                    }
                    _ => {}
                }
            }
            ChatPromptTemplate::GptOssHarmony => {
                // HF template: first system message becomes a developer
                // instruction block; user/assistant use Harmony framing.
                match role {
                    "system" => {
                        if gpt_oss_system.is_none() {
                            gpt_oss_system = Some(content.as_str());
                        } else {
                            // Subsequent system lines are folded as developer append.
                            prompt.push_str("<|start|>developer<|message|># Instructions\n\n");
                            prompt.push_str(content);
                            prompt.push_str("<|end|>");
                        }
                    }
                    "user" => {
                        if let Some(system) = gpt_oss_system.take() {
                            prompt.push_str("<|start|>developer<|message|># Instructions\n\n");
                            prompt.push_str(system);
                            prompt.push_str("<|end|>");
                        }
                        prompt.push_str("<|start|>user<|message|>");
                        prompt.push_str(content);
                        prompt.push_str("<|end|>");
                    }
                    "assistant" => {
                        // Prior turns drop analysis CoT; only final channel is kept.
                        prompt.push_str("<|start|>assistant<|channel|>final<|message|>");
                        prompt.push_str(content);
                        prompt.push_str("<|end|>");
                    }
                    "tool" | "function" => {
                        prompt.push_str("<|start|>user<|message|>");
                        prompt.push_str(content);
                        prompt.push_str("<|end|>");
                    }
                    _ => {}
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
    // System-only chats still need the system block emitted before generation.
    if matches!(template, ChatPromptTemplate::MistralInstruct) {
        if let Some(system) = mistral_system {
            prompt.push_str("[SYSTEM_PROMPT]");
            prompt.push_str(system);
            prompt.push_str("[/SYSTEM_PROMPT]");
        }
    }
    if matches!(template, ChatPromptTemplate::MinistralInstruct) {
        if let Some(system) = ministral_system {
            prompt.push_str("[INST]");
            prompt.push_str(system);
            prompt.push_str("[/INST]");
        }
    }
    if matches!(template, ChatPromptTemplate::GptOssHarmony) {
        if let Some(system) = gpt_oss_system {
            prompt.push_str("<|start|>developer<|message|># Instructions\n\n");
            prompt.push_str(system);
            prompt.push_str("<|end|>");
        }
    }
    match template {
        ChatPromptTemplate::QwenChatMl => {
            prompt.push_str(qwen_generation_prompt);
        }
        ChatPromptTemplate::Llama3 => {
            prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        ChatPromptTemplate::Llama4 => {
            prompt.push_str("<|header_start|>assistant<|header_end|>\n\n");
        }
        ChatPromptTemplate::Gemma4 => {
            prompt.push_str("<|turn>model\n");
            // Gemma 4 IT pre-fills an empty thought channel when thinking is
            // off. DiffusionGemma's hub template opens the model turn only.
            if !is_diffusion_gemma(model_id) {
                prompt.push_str("<|channel>thought\n<channel|>");
            }
        }
        ChatPromptTemplate::Glm47 => prompt.push_str("<|assistant|></think>"),
        ChatPromptTemplate::MistralInstruct | ChatPromptTemplate::MinistralInstruct => {}
        // Prefill final channel so short chat answers do not spend the entire
        // budget on analysis CoT (mlx-community / Unsloth guidance for gpt-oss).
        // Intentional AX delta vs hub jinja, which leaves `<|start|>assistant`
        // without a channel prefill.
        ChatPromptTemplate::GptOssHarmony => {
            prompt.push_str("<|start|>assistant<|channel|>final<|message|>")
        }
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
    normalized == "qwen3" || is_qwen_coder_model(model_id)
}

pub(crate) fn is_qwen_coder_model(model_id: &str) -> bool {
    let normalized = normalize_model_id_token(model_id);
    normalized.contains("qwen3-coder-next") || normalized.contains("qwen3-coder")
}

pub(crate) fn normalize_model_id_token(model_id: &str) -> String {
    model_id
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect()
}

/// Coder-Next hub templates declare tools as XML `<function><name>…` blocks.
/// Qwen3.5 / Qwen3.6 AutomatosX hub templates declare tools as JSON lines and
/// call with function-XML — use [`qwen_tool_contract_style`] FunctionXml, not
/// this coder path.
pub(crate) fn uses_qwen_coder_xml_tool_contract(model_id: &str) -> bool {
    is_qwen_coder_model(model_id)
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

/// Token ids of GPT-OSS Harmony control markers. `None` when the tokenizer
/// does not define the required specials (non-GPT-OSS models), which disables
/// Harmony stream filtering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GptOssHarmonyIds {
    pub(crate) channel: u32,
    pub(crate) message: u32,
    pub(crate) end: u32,
    pub(crate) return_tok: u32,
    pub(crate) start: Option<u32>,
    pub(crate) call: Option<u32>,
}

impl GptOssHarmonyIds {
    pub(crate) fn from_tokenizer(tokenizer: &EngineTokenizer) -> Option<Self> {
        Some(Self {
            channel: tokenizer.token_to_id(GPT_OSS_CHANNEL)?,
            message: tokenizer.token_to_id(GPT_OSS_MESSAGE)?,
            end: tokenizer.token_to_id(GPT_OSS_END)?,
            return_tok: tokenizer.token_to_id(GPT_OSS_RETURN)?,
            start: tokenizer.token_to_id(GPT_OSS_START),
            call: tokenizer.token_to_id(GPT_OSS_CALL),
        })
    }

    pub(crate) fn is_control(self, token: u32) -> bool {
        token == self.channel
            || token == self.message
            || token == self.end
            || token == self.return_tok
            || self.start == Some(token)
            || self.call == Some(token)
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

/// Drop Gemma 4 thinking-channel spans from assistant history content.
///
/// Mirrors the official `chat_template.jinja` `strip_thinking` macro used when
/// re-rendering prior model turns: everything from `<|channel>` through
/// `<channel|>` is removed so channel framing never re-enters the prefill.
pub(crate) fn strip_gemma4_thinking_from_history(content: &str) -> String {
    let mut remaining = content;
    let mut rendered = String::new();
    while let Some(start) = remaining.find(GEMMA4_CHANNEL_OPEN) {
        rendered.push_str(&remaining[..start]);
        let body_start = start + GEMMA4_CHANNEL_OPEN.len();
        let Some(relative_end) = remaining[body_start..].find(GEMMA4_CHANNEL_CLOSE) else {
            remaining = "";
            break;
        };
        remaining = &remaining[body_start + relative_end + GEMMA4_CHANNEL_CLOSE.len()..];
    }
    rendered.push_str(remaining);
    rendered.trim().to_string()
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

/// Extract user-facing text from a GPT-OSS Harmony generation.
///
/// Models often emit:
/// `…<|channel|>analysis<|message|>…thinking…<|end|>`
/// `<|start|>assistant<|channel|>final<|message|>ANSWER<|return|>`
///
/// Chat completions should surface only the last `final` channel body.
/// Falls back to stripping control tokens when no final channel is present.
pub(crate) fn strip_gpt_oss_harmony_output(text: &str) -> String {
    const FINAL_OPEN: &str = "<|channel|>final<|message|>";
    const ANALYSIS_OPEN: &str = "<|channel|>analysis<|message|>";
    const COMMENTARY_OPEN: &str = "<|channel|>commentary<|message|>";

    // Prefer the last final-channel payload.
    if let Some(idx) = text.rfind(FINAL_OPEN) {
        let body = &text[idx + FINAL_OPEN.len()..];
        let body = body
            .split("<|return|>")
            .next()
            .unwrap_or(body)
            .split("<|end|>")
            .next()
            .unwrap_or(body)
            .split("<|start|>")
            .next()
            .unwrap_or(body);
        return body.trim().to_string();
    }

    // Drop analysis / commentary channel spans if present, keep remainder.
    let mut cleaned = text.to_string();
    for open in [ANALYSIS_OPEN, COMMENTARY_OPEN] {
        while let Some(start) = cleaned.find(open) {
            let after = start + open.len();
            let rest = &cleaned[after..];
            let end_rel = rest
                .find("<|end|>")
                .or_else(|| rest.find("<|return|>"))
                .or_else(|| rest.find("<|start|>"))
                .unwrap_or(rest.len());
            let end = after + end_rel;
            let end = if cleaned[end..].starts_with("<|end|>") {
                end + "<|end|>".len()
            } else if cleaned[end..].starts_with("<|return|>") {
                end + "<|return|>".len()
            } else {
                end
            };
            cleaned.replace_range(start..end, "");
        }
    }

    for token in [
        "<|start|>assistant",
        "<|start|>",
        "<|end|>",
        "<|return|>",
        "<|message|>",
        "<|channel|>",
        "<|call|>",
        "<|endoftext|>",
    ] {
        cleaned = cleaned.replace(token, "");
    }
    cleaned.trim().to_string()
}

/// Decode GPT-OSS chat tokens and return only the final-channel user text.
pub(crate) fn decode_gpt_oss_chat_output(
    tokenizer: &EngineTokenizer,
    output_tokens: &[u32],
) -> Result<String, EngineTokenizerError> {
    let raw = tokenizer.decode(output_tokens, false)?;
    Ok(strip_gpt_oss_harmony_output(&raw))
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

/// GLM 4.x structural tool-call markers. They are *special* tokens in GLM's
/// tokenizer, so a plain `skip_special_tokens` decode drops them and the
/// tool-call parser never sees the call (it surfaces as `nameKeyValue` text).
/// This list lets the decoder re-emit them as literal text while still skipping
/// unrelated control tokens (BOS/EOS/turn markers).
const GLM_TOOL_MARKERS: &[&str] = &[
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
];

/// Decode GLM chat output while preserving the structural tool-call markers as
/// literal text. Token runs between markers are decoded with
/// `skip_special_tokens = true` (so EOS/turn tokens stay stripped), and each
/// marker token is re-emitted as its text form. Falls back to a plain decode
/// when the output contains no tool-call markers.
pub(crate) fn decode_glm_chat_output(
    tokenizer: &EngineTokenizer,
    output_tokens: &[u32],
) -> Result<String, EngineTokenizerError> {
    let marker_ids: std::collections::HashMap<u32, &'static str> = GLM_TOOL_MARKERS
        .iter()
        .filter_map(|marker| tokenizer.token_to_id(marker).map(|id| (id, *marker)))
        .collect();
    if marker_ids.is_empty() || !output_tokens.iter().any(|tok| marker_ids.contains_key(tok)) {
        return tokenizer.decode(output_tokens, true);
    }
    let mut decoded = String::new();
    let mut run: Vec<u32> = Vec::new();
    for &token in output_tokens {
        if let Some(marker) = marker_ids.get(&token) {
            if !run.is_empty() {
                decoded.push_str(&tokenizer.decode(&run, true)?);
                run.clear();
            }
            decoded.push_str(marker);
        } else {
            run.push(token);
        }
    }
    if !run.is_empty() {
        decoded.push_str(&tokenizer.decode(&run, true)?);
    }
    Ok(decoded)
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
    fn llama_and_mistral_secondary_families_render_chat() {
        let messages = vec![("user".to_string(), "hello".to_string())];
        let llama = render_prompt("llama3.3-70b", &messages).expect("llama3 chat");
        assert!(llama.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(llama.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
        // Llama 4 hub templates use header_start / eot (not Llama 3 markers).
        let scout = render_prompt("llama4-scout", &messages).expect("llama4 chat");
        assert!(scout.contains("<|header_start|>user<|header_end|>"));
        assert!(scout.contains("<|eot|>"));
        assert!(!scout.contains("<|start_header_id|>"));
        assert!(scout.ends_with("<|header_start|>assistant<|header_end|>\n\n"));
        let mistral = render_prompt("mistral-small", &messages).expect("mistral chat");
        assert!(mistral.contains("[INST]hello[/INST]"));
        let ministral = render_prompt("ministral-8b", &messages).expect("ministral chat");
        assert!(ministral.contains("[INST]hello[/INST]"));
        assert_eq!(
            ChatPromptTemplate::for_model_id("ministral-8b"),
            ChatPromptTemplate::MinistralInstruct
        );
        let devstral = render_prompt("devstral-small", &messages).expect("devstral chat");
        assert!(devstral.contains("[INST]hello[/INST]"));
        assert_eq!(
            ChatPromptTemplate::for_model_id("devstral-small"),
            ChatPromptTemplate::MistralInstruct
        );
    }

    #[test]
    fn diffusiongemma_skips_empty_thought_prefill() {
        let messages = vec![("user".to_string(), "Hello".to_string())];
        let it = render_prompt("gemma4-e2b", &messages).expect("gemma4 it");
        assert!(
            it.ends_with("<|turn>model\n<|channel>thought\n<channel|>"),
            "Gemma 4 IT pre-fills empty thought: {it}"
        );
        let diffusion =
            render_prompt("diffusiongemma-26B-A4B-it-4bit", &messages).expect("diffusion");
        assert!(
            diffusion.ends_with("<|turn>model\n"),
            "DiffusionGemma opens model turn only: {diffusion}"
        );
        assert!(
            !diffusion.contains("<|channel>thought\n<channel|>"),
            "DiffusionGemma must not pre-fill empty thought"
        );
    }

    #[test]
    fn devstral_multi_turn_appends_eos_after_assistant() {
        // Hub Devstral: assistant content + eos_token (`</s>`).
        let messages = vec![
            ("system".to_string(), "Be concise.".to_string()),
            ("user".to_string(), "Hi".to_string()),
            ("assistant".to_string(), "Hello there".to_string()),
            ("user".to_string(), "Again".to_string()),
        ];
        let prompt = render_prompt("devstral-small", &messages).expect("devstral");
        assert_eq!(
            prompt,
            "<s>[SYSTEM_PROMPT]Be concise.[/SYSTEM_PROMPT][INST]Hi[/INST]Hello there</s>[INST]Again[/INST]"
        );
    }

    #[test]
    fn ministral_folds_system_into_last_user_inst() {
        // Hub Ministral: system is applied on the *last* user turn only.
        let simple = render_prompt(
            "ministral-8b",
            &[
                ("system".to_string(), "Be concise.".to_string()),
                ("user".to_string(), "Hello".to_string()),
            ],
        )
        .expect("ministral simple");
        assert_eq!(simple, "<s>[INST]Be concise.\n\nHello[/INST]");

        let multi = render_prompt(
            "ministral-8b",
            &[
                ("system".to_string(), "Be concise.".to_string()),
                ("user".to_string(), "Hi".to_string()),
                ("assistant".to_string(), "Hello there".to_string()),
                ("user".to_string(), "Again".to_string()),
            ],
        )
        .expect("ministral multi");
        assert_eq!(
            multi,
            "<s>[INST]Hi[/INST] Hello there</s>[INST]Be concise.\n\nAgain[/INST]"
        );
    }

    #[test]
    fn gpt_oss_uses_harmony_chat_template() {
        assert_eq!(
            ChatPromptTemplate::for_model_id("gpt-oss-20b"),
            ChatPromptTemplate::GptOssHarmony
        );
        assert_eq!(
            ChatPromptTemplate::for_model_id("gpt-oss-120b"),
            ChatPromptTemplate::GptOssHarmony
        );
        let messages = vec![("user".to_string(), "hello".to_string())];
        let prompt = render_prompt("gpt-oss-20b", &messages).expect("gpt-oss chat");
        assert!(
            prompt.contains("<|start|>system<|message|>"),
            "must emit Harmony system header: {prompt}"
        );
        assert!(
            prompt.contains("<|start|>user<|message|>hello<|end|>"),
            "must frame user turn: {prompt}"
        );
        assert!(
            prompt.ends_with("<|start|>assistant<|channel|>final<|message|>"),
            "must prefill final channel for generation: {prompt}"
        );
        let stops = stop_sequences("gpt-oss-20b", vec![]);
        assert!(
            stops.iter().any(|s| s == "<|return|>"),
            "must stop on <|return|>: {stops:?}"
        );
    }

    #[test]
    fn strip_gemma4_thinking_from_history_drops_channel_spans() {
        assert_eq!(
            strip_gemma4_thinking_from_history("<|channel>thought\nplan\n<channel|>Hello"),
            "Hello"
        );
        assert_eq!(
            strip_gemma4_thinking_from_history("plain answer"),
            "plain answer"
        );
        // Plain multi-turn Gemma path must strip prior assistant channels.
        let prompt = render_prompt(
            "gemma4-e2b",
            &[
                ("user".to_string(), "Hi".to_string()),
                (
                    "assistant".to_string(),
                    "<|channel>thought\nplan\n<channel|>Hello".to_string(),
                ),
                ("user".to_string(), "Again".to_string()),
            ],
        )
        .expect("render");
        assert!(prompt.contains("<|turn>model\nHello<turn|>\n"));
        assert!(!prompt.contains("plan"));
    }

    #[test]
    fn strip_gpt_oss_harmony_prefers_final_channel() {
        let raw = concat!(
            "<|channel|>analysis<|message|>thinking about greeting<|end|>",
            "<|start|>assistant<|channel|>final<|message|>Hi there!<|return|>"
        );
        assert_eq!(strip_gpt_oss_harmony_output(raw), "Hi there!");
    }

    #[test]
    fn strip_gpt_oss_harmony_handles_bare_final() {
        let raw = "<|channel|>final<|message|>42<|return|>";
        assert_eq!(strip_gpt_oss_harmony_output(raw), "42");
    }

    #[test]
    fn primary_productivity_families_select_chat_templates() {
        assert_eq!(
            ChatPromptTemplate::for_model_id("gemma4-31b"),
            ChatPromptTemplate::Gemma4
        );
        assert_eq!(
            ChatPromptTemplate::for_model_id("qwen3.6-35b"),
            ChatPromptTemplate::QwenChatMl
        );
        assert_eq!(
            ChatPromptTemplate::for_model_id("glm4.7-flash-4bit"),
            ChatPromptTemplate::Glm47
        );
    }

    #[test]
    fn diffusiongemma_model_id_selects_gemma4_chat_template() {
        // DiffusionGemma uses the Gemma4 turn-based chat format (<|turn>,
        // <turn|>, <|channel>, <channel|>). The model ID must map to
        // Gemma4 and NOT fall through to the gemma3 unsupported branch.
        for model_id in [
            "mlx-community/diffusiongemma-26B-A4B-it-4bit",
            "diffusion-gemma-26B-A4B-it",
            "diffusion_gemma",
            "DiffusionGemma-26B-A4B-it",
        ] {
            let template = ChatPromptTemplate::for_model_id(model_id);
            assert_eq!(
                template,
                ChatPromptTemplate::Gemma4,
                "{model_id} should use Gemma4 chat template"
            );
        }

        // Stop sequences must include the Gemma4 turn terminator.
        let stops = stop_sequences("diffusiongemma-26B-A4B-it-4bit", vec![]);
        assert!(
            stops.contains(&"<turn|>".to_string()),
            "DiffusionGemma stop sequences must include <turn|>: {stops:?}"
        );

        // Rendered prompt must use Gemma4 turn format.
        let messages = vec![("user".to_string(), "Hello".to_string())];
        let prompt = render_prompt("diffusiongemma-26B-A4B-it-4bit", &messages).expect("render");
        assert!(prompt.starts_with("<bos>"), "prompt must start with <bos>");
        assert!(
            prompt.contains("<|turn>user\n"),
            "prompt must contain Gemma4 user turn marker"
        );
        assert!(
            prompt.contains("<|turn>model\n"),
            "prompt must contain Gemma4 model generation prompt"
        );
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
        validate_native_chat_artifact("diffusiongemma-26B-A4B-it-4bit", Some(&artifact_dir))
            .expect("DiffusionGemma should share Gemma4 chat artifact validation");
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

    #[test]
    fn qwen36_uses_function_xml_tools_not_coder_declarations() {
        // Hub Qwen3.6 chat_template.jinja matches Qwen3.5 (JSON tool schemas +
        // function= calls). Only Coder-Next uses XML tool *declarations*.
        for model_id in [
            "qwen3.6-27b-8bit",
            "Qwen3.6-35B-A3B-4bit",
            "mlx-community/Qwen3.6-35B-A3B-4bit",
        ] {
            assert!(is_qwen_thinking_model(model_id), "{model_id}");
            assert!(!is_qwen_non_thinking_only_model(model_id), "{model_id}");
            assert!(
                !uses_qwen_coder_xml_tool_contract(model_id),
                "{model_id} must not use Coder-Next XML declarations"
            );
        }
        assert!(uses_qwen_coder_xml_tool_contract("Qwen3-Coder-Next-4bit"));
    }
}
