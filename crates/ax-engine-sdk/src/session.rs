use std::str::FromStr;
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{Context, anyhow, ensure};
use ax_engine_core::chat::{
    ChatMessage as CoreChatMessage, ChatRenderOptions, ChatRole as CoreChatRole,
    render_chat_messages,
};
use ax_engine_core::kv::ModelKv;
use ax_engine_core::model::WeightStore;
use ax_engine_core::sampling::{Sampler, SamplingConfig};

use crate::model::Model;

struct SessionState {
    kv: ModelKv,
    history: Vec<u32>,
}

pub struct Session {
    model: Model,
    state: Arc<Mutex<SessionState>>,
    max_context_tokens: usize,
    default_seed: Option<u64>,
}

pub struct TextStream {
    state: Option<LiveTextStreamState>,
    text: String,
    output: Option<GenerationOutput>,
}

struct LiveTextStreamState {
    model: Model,
    session_state: Arc<Mutex<SessionState>>,
    sampler: Sampler,
    next_token: Option<u32>,
    position: usize,
    remaining_tokens: usize,
    logits: Vec<f32>,
    pending: String,
    stop_strings: Vec<String>,
    finish_reason: FinishReason,
    done: bool,
    prompt_tokens: usize,
    completion_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, Default)]
pub struct SessionOptions {
    pub context_length: Option<usize>,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct GenerationOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub stop_strings: Vec<String>,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub text: String,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

impl std::fmt::Display for ChatRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::System => f.write_str("system"),
            Self::User => f.write_str("user"),
            Self::Assistant => f.write_str("assistant"),
        }
    }
}

impl ChatRole {
    fn into_core(self) -> CoreChatRole {
        match self {
            Self::System => CoreChatRole::System,
            Self::User => CoreChatRole::User,
            Self::Assistant => CoreChatRole::Assistant,
        }
    }
}

impl FromStr for ChatRole {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "system" | "developer" => Ok(Self::System),
            "user" => Ok(Self::User),
            "assistant" => Ok(Self::Assistant),
            other => Err(anyhow!("unsupported chat role '{other}'")),
        }
    }
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(ChatRole::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(ChatRole::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(ChatRole::Assistant, content)
    }
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_strings: Vec::new(),
            seed: None,
        }
    }
}

impl GenerationOptions {
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    pub fn top_k(mut self, k: i32) -> Self {
        self.top_k = k;
        self
    }

    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    pub fn min_p(mut self, p: f32) -> Self {
        self.min_p = p;
        self
    }

    pub fn repeat_penalty(mut self, penalty: f32) -> Self {
        self.repeat_penalty = penalty;
        self
    }

    pub fn repeat_last_n(mut self, n: i32) -> Self {
        self.repeat_last_n = n;
        self
    }

    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = penalty;
        self
    }

    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = penalty;
        self
    }

    /// Append a stop string. May be called multiple times to add multiple stop sequences.
    pub fn stop(mut self, s: impl Into<String>) -> Self {
        self.stop_strings.push(s.into());
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl Session {
    pub(crate) fn new(model: Model, options: SessionOptions) -> anyhow::Result<Self> {
        let native = model.native();
        let weights = WeightStore::new(&native.mapped);
        let kv = native.model.create_model_kv_for_weights(&weights);
        let state = SessionState {
            kv,
            history: Vec::new(),
        };

        let max_context_tokens = options
            .context_length
            .unwrap_or(usize::MAX)
            .min(model.context_length());
        ensure!(
            max_context_tokens > 0,
            "context_length must be greater than zero"
        );

        #[allow(clippy::arc_with_non_send_sync)]
        let state = Arc::new(Mutex::new(state));

        Ok(Self {
            model,
            state,
            max_context_tokens,
            default_seed: options.seed,
        })
    }

    pub fn context_length(&self) -> usize {
        self.max_context_tokens
    }

    pub fn position(&self) -> anyhow::Result<usize> {
        Ok(self.lock_state()?.history.len())
    }

    pub fn reset(&self) -> anyhow::Result<()> {
        let mut state = self.lock_state()?;
        state.kv.clear();
        state.history.clear();
        Ok(())
    }

    pub fn generate(
        &self,
        prompt: &str,
        options: GenerationOptions,
    ) -> anyhow::Result<GenerationOutput> {
        self.stream(prompt, options)?.into_output()
    }

    pub fn stream(&self, prompt: &str, options: GenerationOptions) -> anyhow::Result<TextStream> {
        let add_bos = self.position()? == 0;
        self.start_stream(prompt, add_bos, options)
    }

    pub fn chat(
        &self,
        messages: &[ChatMessage],
        options: GenerationOptions,
    ) -> anyhow::Result<GenerationOutput> {
        self.stream_chat(messages, options)?.into_output()
    }

    pub fn stream_chat(
        &self,
        messages: &[ChatMessage],
        options: GenerationOptions,
    ) -> anyhow::Result<TextStream> {
        ensure!(!messages.is_empty(), "chat messages must not be empty");
        let rendered_messages = messages
            .iter()
            .map(|message| CoreChatMessage::new(message.role.into_core(), message.content.as_str()))
            .collect::<Vec<_>>();
        let rendered = render_chat_messages(
            &rendered_messages,
            self.model.architecture(),
            ChatRenderOptions::default(),
        );
        self.reset()?;
        self.start_stream(&rendered, true, options)
    }

    fn start_stream(
        &self,
        prompt: &str,
        add_bos: bool,
        options: GenerationOptions,
    ) -> anyhow::Result<TextStream> {
        validate_generation_options(&options)?;
        let native = self.model.native();

        let weights = WeightStore::new(&native.mapped);
        let prompt_tokens = self.model.tokenize(prompt, add_bos);
        ensure!(
            !prompt_tokens.is_empty(),
            "prompt produced no tokens; provide a non-empty prompt"
        );

        let mut state_guard = self.lock_state()?;
        let position = state_guard.history.len();
        let remaining_decode_capacity =
            remaining_decode_capacity(prompt_tokens.len(), position, self.max_context_tokens)?;

        let mut logits = vec![0.0f32; self.model.config().vocab_size as usize];
        native
            .model
            .forward_batch(&prompt_tokens, &mut state_guard.kv, &weights, &mut logits)
            .context("prefill forward pass failed")?;
        state_guard.history.extend_from_slice(&prompt_tokens);

        let decode_position = position + prompt_tokens.len();
        let max_tokens = options.max_tokens.min(remaining_decode_capacity);
        let sampling = build_sampling_config(&options, self.default_seed);

        let live_state = if max_tokens == 0 {
            LiveTextStreamState {
                model: self.model.clone(),
                session_state: self.state.clone(),
                sampler: Sampler::new(SamplingConfig::default()),
                next_token: None,
                position: decode_position,
                remaining_tokens: 0,
                logits,
                pending: String::new(),
                stop_strings: filtered_stop_strings(options.stop_strings),
                finish_reason: FinishReason::Length,
                done: true,
                prompt_tokens: prompt_tokens.len(),
                completion_tokens: 0,
            }
        } else {
            let mut sampler = Sampler::new(sampling);
            let first_token = sampler.sample(&mut logits, &state_guard.history);
            LiveTextStreamState {
                model: self.model.clone(),
                session_state: self.state.clone(),
                sampler,
                next_token: Some(first_token),
                position: decode_position,
                remaining_tokens: max_tokens,
                logits,
                pending: String::new(),
                stop_strings: filtered_stop_strings(options.stop_strings),
                finish_reason: FinishReason::Stop,
                done: false,
                prompt_tokens: prompt_tokens.len(),
                completion_tokens: 0,
            }
        };
        drop(state_guard);

        Ok(TextStream {
            state: Some(live_state),
            text: String::new(),
            output: None,
        })
    }

    fn lock_state(&self) -> anyhow::Result<MutexGuard<'_, SessionState>> {
        self.state
            .lock()
            .map_err(|_| anyhow!("session state lock poisoned"))
    }
}

impl TextStream {
    pub fn next_chunk(&mut self) -> anyhow::Result<Option<String>> {
        let Some(state) = self.state.as_mut() else {
            return Ok(None);
        };

        let next = state.next_chunk()?;
        if let Some(chunk) = next.as_ref() {
            self.text.push_str(chunk);
        } else if let Some(state) = self.state.take() {
            self.output = Some(state.into_output(std::mem::take(&mut self.text)));
        }
        Ok(next)
    }

    pub fn is_done(&self) -> bool {
        self.output.is_some() || self.state.as_ref().is_none_or(|state| state.done)
    }

    pub fn output(&self) -> Option<&GenerationOutput> {
        self.output.as_ref()
    }

    pub fn into_output(mut self) -> anyhow::Result<GenerationOutput> {
        while self.next_chunk()?.is_some() {}
        // next_chunk sets self.output when it returns None (via the state.take() branch),
        // so this is always Some after the loop above.
        self.output.take().ok_or_else(|| {
            anyhow::anyhow!("TextStream::into_output: output not set after drain; this is a bug")
        })
    }
}

impl Iterator for TextStream {
    type Item = anyhow::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_chunk() {
            Ok(Some(chunk)) => Some(Ok(chunk)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

impl LiveTextStreamState {
    fn next_chunk(&mut self) -> anyhow::Result<Option<String>> {
        loop {
            if self.done {
                if self.pending.is_empty() {
                    return Ok(None);
                }
                return Ok(Some(std::mem::take(&mut self.pending)));
            }

            if self.remaining_tokens == 0 {
                self.finish_reason = FinishReason::Length;
                self.done = true;
                self.next_token = None;
                continue;
            }

            let token = match self.next_token {
                Some(token) => token,
                None => {
                    self.finish_reason = FinishReason::Stop;
                    self.done = true;
                    continue;
                }
            };

            if self.model.tokenizer().is_eos(token) {
                self.finish_reason = FinishReason::Stop;
                self.done = true;
                self.next_token = None;
                continue;
            }

            let piece = render_token_text(&self.model, token);
            if !piece.is_empty() {
                self.pending.push_str(&piece);
            }

            if let Some(stop_at) = first_stop_match(&self.pending, &self.stop_strings) {
                self.completion_tokens += 1;
                let chunk = self.pending[..stop_at].to_string();
                self.pending.clear();
                self.finish_reason = FinishReason::Stop;
                self.done = true;
                self.next_token = None;
                if chunk.is_empty() {
                    continue;
                }
                return Ok(Some(chunk));
            }

            let held_back = longest_partial_stop_suffix(&self.pending, &self.stop_strings);
            let safe_len = self.pending.len().saturating_sub(held_back);
            let chunk = if safe_len > 0 {
                let chunk = self.pending[..safe_len].to_string();
                self.pending.drain(..safe_len);
                Some(chunk)
            } else {
                None
            };

            {
                let mut state = self
                    .session_state
                    .lock()
                    .map_err(|_| anyhow!("session state lock poisoned"))?;
                state.history.push(token);
                self.completion_tokens += 1;
                self.remaining_tokens = self.remaining_tokens.saturating_sub(1);

                if self.remaining_tokens == 0 {
                    self.finish_reason = FinishReason::Length;
                    self.done = true;
                    self.next_token = None;
                } else {
                    let native = self.model.native();
                    let weights = WeightStore::new(&native.mapped);
                    self.logits.fill(0.0);
                    native
                        .model
                        .forward_single(
                            token,
                            self.position,
                            &mut state.kv,
                            &weights,
                            &mut self.logits,
                        )
                        .context("decode step failed")?;
                    self.position += 1;
                    self.next_token = Some(self.sampler.sample(&mut self.logits, &state.history));
                }
            }

            if let Some(chunk) = chunk
                && !chunk.is_empty()
            {
                return Ok(Some(chunk));
            }
        }
    }

    fn into_output(self, text: String) -> GenerationOutput {
        let usage = Usage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: self.prompt_tokens + self.completion_tokens,
        };
        GenerationOutput {
            text,
            finish_reason: self.finish_reason,
            usage,
        }
    }
}

fn build_sampling_config(options: &GenerationOptions, default_seed: Option<u64>) -> SamplingConfig {
    SamplingConfig {
        temperature: options.temperature,
        top_k: options.top_k,
        top_p: options.top_p,
        min_p: options.min_p,
        repeat_penalty: options.repeat_penalty,
        repeat_last_n: options.repeat_last_n,
        frequency_penalty: options.frequency_penalty,
        presence_penalty: options.presence_penalty,
        seed: options.seed.or(default_seed).unwrap_or(u64::MAX),
        ..SamplingConfig::default()
    }
}

fn remaining_decode_capacity(
    prompt_tokens_len: usize,
    used_context_tokens: usize,
    max_context_tokens: usize,
) -> anyhow::Result<usize> {
    let remaining_context = max_context_tokens.saturating_sub(used_context_tokens);
    ensure!(
        prompt_tokens_len <= remaining_context,
        "prompt does not fit in remaining context: {} tokens requested, {} available",
        prompt_tokens_len,
        remaining_context
    );
    Ok(remaining_context - prompt_tokens_len)
}

fn validate_generation_options(options: &GenerationOptions) -> anyhow::Result<()> {
    ensure!(
        options.max_tokens > 0,
        "max_tokens must be greater than zero"
    );
    ensure!(
        options.temperature.is_finite() && options.temperature >= 0.0,
        "temperature must be finite and non-negative"
    );
    ensure!(
        options.top_k >= -1,
        "top_k must be -1 (disabled) or greater"
    );
    ensure!(
        options.top_p.is_finite() && (0.0..=1.0).contains(&options.top_p),
        "top_p must be finite and between 0.0 and 1.0"
    );
    ensure!(
        options.min_p.is_finite() && (0.0..=1.0).contains(&options.min_p),
        "min_p must be finite and between 0.0 and 1.0"
    );
    ensure!(
        options.repeat_penalty.is_finite() && options.repeat_penalty > 0.0,
        "repeat_penalty must be finite and greater than zero"
    );
    ensure!(
        options.frequency_penalty.is_finite(),
        "frequency_penalty must be finite"
    );
    ensure!(
        options.presence_penalty.is_finite(),
        "presence_penalty must be finite"
    );
    ensure!(
        options.repeat_last_n >= -1,
        "repeat_last_n must be -1 or greater"
    );
    Ok(())
}

fn filtered_stop_strings(stop_strings: Vec<String>) -> Vec<String> {
    stop_strings
        .into_iter()
        .filter(|stop| !stop.is_empty())
        .collect()
}

fn render_token_text(model: &Model, token: u32) -> String {
    model
        .tokenizer()
        .render_token(token)
        .unwrap_or_else(|| model.tokenizer().decode(&[token]))
}

fn first_stop_match(output: &str, stop_strings: &[String]) -> Option<usize> {
    stop_strings
        .iter()
        .filter_map(|stop| output.find(stop))
        .min()
}

fn longest_partial_stop_suffix(output: &str, stop_strings: &[String]) -> usize {
    let mut longest = 0;

    for stop in stop_strings.iter().filter(|stop| !stop.is_empty()) {
        for (prefix_len, _) in stop.char_indices().skip(1) {
            if output.ends_with(&stop[..prefix_len]) {
                longest = longest.max(prefix_len);
            }
        }
    }

    longest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_role_from_str_accepts_common_roles() {
        assert_eq!("system".parse::<ChatRole>().unwrap(), ChatRole::System);
        assert_eq!("developer".parse::<ChatRole>().unwrap(), ChatRole::System);
        assert_eq!("user".parse::<ChatRole>().unwrap(), ChatRole::User);
    }

    #[test]
    fn test_chat_role_display() {
        assert_eq!(ChatRole::System.to_string(), "system");
        assert_eq!(ChatRole::User.to_string(), "user");
        assert_eq!(ChatRole::Assistant.to_string(), "assistant");
    }

    #[test]
    fn test_generation_options_builder_overrides_defaults() {
        let opts = GenerationOptions::default()
            .max_tokens(512)
            .temperature(0.2)
            .top_k(-1)
            .stop("</s>")
            .stop("<end>")
            .seed(42);

        assert_eq!(opts.max_tokens, 512);
        assert!((opts.temperature - 0.2).abs() < 1e-6);
        assert_eq!(opts.top_k, -1);
        assert_eq!(opts.stop_strings, vec!["</s>", "<end>"]);
        assert_eq!(opts.seed, Some(42));
        // Fields not set keep their defaults
        assert!((opts.top_p - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_longest_partial_stop_suffix_finds_prefix_overlap() {
        assert_eq!(
            longest_partial_stop_suffix("hello s", &[String::from("stop")]),
            1
        );
    }

    #[test]
    fn test_validate_generation_options_rejects_non_finite_sampling_values() {
        let options = GenerationOptions {
            repeat_penalty: f32::INFINITY,
            ..GenerationOptions::default()
        };
        let err = validate_generation_options(&options).unwrap_err();
        assert!(err.to_string().contains("repeat_penalty"));

        let options = GenerationOptions {
            frequency_penalty: f32::NAN,
            ..GenerationOptions::default()
        };
        let err = validate_generation_options(&options).unwrap_err();
        assert!(err.to_string().contains("frequency_penalty"));
    }

    #[test]
    fn test_validate_generation_options_rejects_invalid_top_k() {
        let options = GenerationOptions {
            top_k: -2,
            ..GenerationOptions::default()
        };
        let err = validate_generation_options(&options)
            .unwrap_err()
            .to_string();
        assert!(err.contains("top_k"));
    }

    #[test]
    fn test_validate_generation_options_rejects_zero_repeat_penalty() {
        let options = GenerationOptions {
            repeat_penalty: 0.0,
            ..GenerationOptions::default()
        };
        let err = validate_generation_options(&options).unwrap_err();
        assert!(err.to_string().contains("repeat_penalty"));
    }

    #[test]
    fn test_remaining_decode_capacity_rejects_prompt_overflow() {
        let err = remaining_decode_capacity(17, 0, 16).unwrap_err();
        assert!(err.to_string().contains("does not fit"));
    }

    #[test]
    fn test_remaining_decode_capacity_returns_zero_for_exact_fit() {
        assert_eq!(
            remaining_decode_capacity(16, 0, 16).expect("exact-fit prompt should succeed"),
            0
        );
    }
}
