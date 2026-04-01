use std::str::FromStr;
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;

use anyhow::{Context, anyhow, ensure};
use ax_engine_core::chat::{
    ChatMessage as CoreChatMessage, ChatRenderOptions, ChatRole as CoreChatRole,
    render_chat_messages,
};
use ax_engine_core::kv::ModelKv;
use ax_engine_core::model::WeightStore;
use ax_engine_core::sampling::{Sampler, SamplingConfig};

use crate::llama_cpp_process::RemoteGenerationOutput;
use crate::model::Model;

struct NativeSessionState {
    kv: ModelKv,
    history: Vec<u32>,
}

struct LlamaCppSessionState {
    prompt_prefix: String,
    history: Vec<u32>,
}

enum SessionState {
    Native(NativeSessionState),
    LlamaCpp(LlamaCppSessionState),
}

pub struct Session {
    model: Model,
    state: Arc<Mutex<SessionState>>,
    max_context_tokens: usize,
    default_seed: Option<u64>,
}

pub struct TextStream {
    state: Option<TextStreamState>,
    text: String,
    output: Option<GenerationOutput>,
}

enum TextStreamState {
    Native(Box<LiveTextStreamState>),
    LlamaCpp(RemoteTextStreamState),
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

struct RemoteTextStreamState {
    receiver: mpsc::Receiver<RemoteStreamEvent>,
    done: bool,
    finalize: RemoteFinalize,
}

enum RemoteStreamEvent {
    Chunk(String),
    Finished(RemoteGenerationOutput),
    Failed(String),
}

enum RemoteFinalize {
    Completion {
        model: Model,
        session_state: Arc<Mutex<SessionState>>,
        prompt_text: String,
        prompt_tokens: Vec<u32>,
    },
    Chat {
        model: Model,
        session_state: Arc<Mutex<SessionState>>,
        transcript: String,
        prompt_tokens_len: usize,
    },
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

impl Session {
    pub(crate) fn new(model: Model, options: SessionOptions) -> anyhow::Result<Self> {
        let state = if let Some(native) = model.native_loaded() {
            let weights = WeightStore::new(&native.mapped);
            let kv = native.model.create_model_kv_for_weights(&weights);
            SessionState::Native(NativeSessionState {
                kv,
                history: Vec::new(),
            })
        } else {
            SessionState::LlamaCpp(LlamaCppSessionState {
                prompt_prefix: String::new(),
                history: Vec::new(),
            })
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
        Ok(match &*self.lock_state()? {
            SessionState::Native(state) => state.history.len(),
            SessionState::LlamaCpp(state) => state.history.len(),
        })
    }

    pub fn reset(&self) -> anyhow::Result<()> {
        let mut state = self.lock_state()?;
        match &mut *state {
            SessionState::Native(state) => {
                state.kv.clear();
                state.history.clear();
            }
            SessionState::LlamaCpp(state) => {
                state.prompt_prefix.clear();
                state.history.clear();
            }
        }
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
        if self.model.native_loaded().is_some() {
            let add_bos = self.position()? == 0;
            self.start_native_stream_with_options(prompt, add_bos, options)
        } else {
            self.start_llama_cpp_completion_stream(prompt, options)
        }
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
        if self.model.native_loaded().is_some() {
            let rendered_messages = messages
                .iter()
                .map(|message| {
                    CoreChatMessage::new(message.role.into_core(), message.content.as_str())
                })
                .collect::<Vec<_>>();
            let rendered = render_chat_messages(
                &rendered_messages,
                self.model.architecture(),
                ChatRenderOptions::default(),
            );
            self.reset()?;
            self.start_native_stream_with_options(&rendered, true, options)
        } else {
            self.reset()?;
            self.start_llama_cpp_chat_stream(messages, options)
        }
    }

    fn start_native_stream_with_options(
        &self,
        prompt: &str,
        add_bos: bool,
        options: GenerationOptions,
    ) -> anyhow::Result<TextStream> {
        validate_generation_options(&options)?;
        let native = self
            .model
            .native_loaded()
            .ok_or_else(|| anyhow!("native model is not loaded"))?;

        let weights = WeightStore::new(&native.mapped);
        let prompt_tokens = self.model.tokenize(prompt, add_bos);
        ensure!(
            !prompt_tokens.is_empty(),
            "prompt produced no tokens; provide a non-empty prompt"
        );

        let mut state_guard = self.lock_state()?;
        let SessionState::Native(state) = &mut *state_guard else {
            return Err(anyhow!("expected native session state"));
        };
        let position = state.history.len();
        let remaining_context = self.max_context_tokens.saturating_sub(position);
        ensure!(
            prompt_tokens.len() <= remaining_context,
            "prompt does not fit in remaining context: {} tokens requested, {} available",
            prompt_tokens.len(),
            remaining_context
        );

        let mut logits = vec![0.0f32; self.model.config().vocab_size as usize];
        native
            .model
            .forward_batch(&prompt_tokens, &mut state.kv, &weights, &mut logits)
            .context("prefill forward pass failed")?;
        state.history.extend_from_slice(&prompt_tokens);

        let decode_position = position + prompt_tokens.len();
        let remaining_decode_capacity = self.max_context_tokens.saturating_sub(decode_position);
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
            let first_token = sampler.sample(&mut logits, &state.history);
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
            state: Some(TextStreamState::Native(Box::new(live_state))),
            text: String::new(),
            output: None,
        })
    }

    fn start_llama_cpp_completion_stream(
        &self,
        prompt: &str,
        mut options: GenerationOptions,
    ) -> anyhow::Result<TextStream> {
        validate_generation_options(&options)?;
        let remote = self
            .model
            .llama_cpp_loaded()
            .ok_or_else(|| anyhow!("llama.cpp model is not loaded"))?;

        let add_bos = self.position()? == 0;
        let prompt_tokens = self.model.tokenize(prompt, add_bos);
        ensure!(
            !prompt_tokens.is_empty(),
            "prompt produced no tokens; provide a non-empty prompt"
        );

        let mut state_guard = self.lock_state()?;
        let SessionState::LlamaCpp(state) = &mut *state_guard else {
            return Err(anyhow!("expected llama.cpp session state"));
        };
        let full_prompt = format!("{}{}", state.prompt_prefix, prompt);
        let remaining_context = self.max_context_tokens.saturating_sub(state.history.len());
        ensure!(
            prompt_tokens.len() <= remaining_context,
            "prompt does not fit in remaining context: {} tokens requested, {} available",
            prompt_tokens.len(),
            remaining_context
        );
        let remaining_decode_capacity = remaining_context.saturating_sub(prompt_tokens.len());
        let max_tokens = options.max_tokens.min(remaining_decode_capacity);
        drop(state_guard);

        if max_tokens == 0 {
            return Ok(completed_stream(GenerationOutput {
                text: String::new(),
                finish_reason: FinishReason::Length,
                usage: Usage {
                    prompt_tokens: prompt_tokens.len(),
                    completion_tokens: 0,
                    total_tokens: prompt_tokens.len(),
                },
            }));
        }
        options.max_tokens = max_tokens;
        options.stop_strings = filtered_stop_strings(options.stop_strings);

        let (tx, rx) = mpsc::channel();
        let prompt_text = prompt.to_string();
        let prompt_tokens_for_state = prompt_tokens.clone();
        let process = remote.process.clone();

        thread::spawn(move || {
            let stream_result = process.stream_completion(&full_prompt, &options, |chunk| {
                tx.send(RemoteStreamEvent::Chunk(chunk.to_string()))
                    .map_err(|_| anyhow!("stream receiver dropped"))?;
                Ok(())
            });

            match stream_result {
                Ok(output) => {
                    let _ = tx.send(RemoteStreamEvent::Finished(output));
                }
                Err(err) => {
                    let _ = tx.send(RemoteStreamEvent::Failed(err.to_string()));
                }
            }
        });

        Ok(TextStream {
            state: Some(TextStreamState::LlamaCpp(RemoteTextStreamState {
                receiver: rx,
                done: false,
                finalize: RemoteFinalize::Completion {
                    model: self.model.clone(),
                    session_state: self.state.clone(),
                    prompt_text,
                    prompt_tokens: prompt_tokens_for_state,
                },
            })),
            text: String::new(),
            output: None,
        })
    }

    fn start_llama_cpp_chat_stream(
        &self,
        messages: &[ChatMessage],
        mut options: GenerationOptions,
    ) -> anyhow::Result<TextStream> {
        validate_generation_options(&options)?;
        ensure!(!messages.is_empty(), "messages must not be empty");

        let remote = self
            .model
            .llama_cpp_loaded()
            .ok_or_else(|| anyhow!("llama.cpp model is not loaded"))?;
        let transcript = flatten_chat_messages(messages);
        let prompt_tokens = self.model.tokenize(&transcript, true);
        let remaining = self.max_context_tokens.saturating_sub(prompt_tokens.len());
        let max_tokens = options.max_tokens.min(remaining);
        if max_tokens == 0 {
            return Ok(completed_stream(GenerationOutput {
                text: String::new(),
                finish_reason: FinishReason::Length,
                usage: Usage {
                    prompt_tokens: prompt_tokens.len(),
                    completion_tokens: 0,
                    total_tokens: prompt_tokens.len(),
                },
            }));
        }
        options.max_tokens = max_tokens;
        options.stop_strings = filtered_stop_strings(options.stop_strings);

        let (tx, rx) = mpsc::channel();
        let messages_owned = messages.to_vec();
        let transcript_owned = transcript.clone();
        let prompt_tokens_len = prompt_tokens.len();
        let process = remote.process.clone();

        thread::spawn(move || {
            let stream_result = process.stream_chat(&messages_owned, &options, |chunk| {
                tx.send(RemoteStreamEvent::Chunk(chunk.to_string()))
                    .map_err(|_| anyhow!("stream receiver dropped"))?;
                Ok(())
            });

            match stream_result {
                Ok(output) => {
                    let _ = tx.send(RemoteStreamEvent::Finished(output));
                }
                Err(err) => {
                    let _ = tx.send(RemoteStreamEvent::Failed(err.to_string()));
                }
            }
        });

        Ok(TextStream {
            state: Some(TextStreamState::LlamaCpp(RemoteTextStreamState {
                receiver: rx,
                done: false,
                finalize: RemoteFinalize::Chat {
                    model: self.model.clone(),
                    session_state: self.state.clone(),
                    transcript: transcript_owned,
                    prompt_tokens_len,
                },
            })),
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

        match state {
            TextStreamState::Native(state) => {
                let next = state.next_chunk()?;
                if let Some(chunk) = next.as_ref() {
                    self.text.push_str(chunk);
                } else if let Some(TextStreamState::Native(state)) = self.state.take() {
                    self.output = Some(state.into_output(std::mem::take(&mut self.text)));
                }
                Ok(next)
            }
            TextStreamState::LlamaCpp(state) => match state.next_event()? {
                Some(RemoteStreamEvent::Chunk(chunk)) => {
                    self.text.push_str(&chunk);
                    Ok(Some(chunk))
                }
                Some(RemoteStreamEvent::Finished(output)) => {
                    let output = state.finish(output)?;
                    self.state.take();
                    self.output = Some(output);
                    Ok(None)
                }
                Some(RemoteStreamEvent::Failed(message)) => {
                    self.state.take();
                    Err(anyhow!(message))
                }
                None => {
                    self.state.take();
                    Ok(None)
                }
            },
        }
    }

    pub fn is_done(&self) -> bool {
        self.output.is_some()
            || self.state.as_ref().is_none_or(|state| match state {
                TextStreamState::Native(state) => state.done,
                TextStreamState::LlamaCpp(state) => state.done,
            })
    }

    pub fn output(&self) -> Option<&GenerationOutput> {
        self.output.as_ref()
    }

    pub fn into_output(mut self) -> anyhow::Result<GenerationOutput> {
        while self.next_chunk()?.is_some() {}
        if let Some(output) = self.output.take() {
            return Ok(output);
        }

        // Fallback: preserve any partial text accumulated from chunks before
        // the stream terminated unexpectedly (e.g., remote process crash).
        Ok(GenerationOutput {
            text: std::mem::take(&mut self.text),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        })
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
                let SessionState::Native(state) = &mut *state else {
                    return Err(anyhow!("expected native session state"));
                };
                state.history.push(token);
                self.completion_tokens += 1;
                self.remaining_tokens = self.remaining_tokens.saturating_sub(1);

                if self.remaining_tokens == 0 {
                    self.finish_reason = FinishReason::Length;
                    self.done = true;
                    self.next_token = None;
                } else {
                    let native = self
                        .model
                        .native_loaded()
                        .ok_or_else(|| anyhow!("native model is not loaded"))?;
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

impl RemoteTextStreamState {
    fn next_event(&mut self) -> anyhow::Result<Option<RemoteStreamEvent>> {
        if self.done {
            return Ok(None);
        }

        match self.receiver.recv() {
            Ok(event @ RemoteStreamEvent::Chunk(_)) => Ok(Some(event)),
            Ok(event @ RemoteStreamEvent::Finished(_)) => {
                self.done = true;
                Ok(Some(event))
            }
            Ok(event @ RemoteStreamEvent::Failed(_)) => {
                self.done = true;
                Ok(Some(event))
            }
            Err(_) => {
                self.done = true;
                Ok(None)
            }
        }
    }

    fn finish(&mut self, output: RemoteGenerationOutput) -> anyhow::Result<GenerationOutput> {
        match &self.finalize {
            RemoteFinalize::Completion {
                model,
                session_state,
                prompt_text,
                prompt_tokens,
            } => {
                let completion_tokens = model.tokenize(&output.text, false);
                let mut state = session_state
                    .lock()
                    .map_err(|_| anyhow!("session state lock poisoned"))?;
                let SessionState::LlamaCpp(state) = &mut *state else {
                    return Err(anyhow!("expected llama.cpp session state"));
                };
                state.prompt_prefix.push_str(prompt_text);
                state.prompt_prefix.push_str(&output.text);
                state.history.extend_from_slice(prompt_tokens);
                state.history.extend_from_slice(&completion_tokens);

                Ok(GenerationOutput {
                    text: output.text,
                    finish_reason: output.finish_reason,
                    usage: Usage {
                        prompt_tokens: prompt_tokens.len(),
                        completion_tokens: completion_tokens.len(),
                        total_tokens: prompt_tokens.len() + completion_tokens.len(),
                    },
                })
            }
            RemoteFinalize::Chat {
                model,
                session_state,
                transcript,
                prompt_tokens_len,
            } => {
                let history_text = format!("{transcript}{}", output.text);
                let history_tokens = model.tokenize(&history_text, true);
                let completion_tokens = model.tokenize(&output.text, false);
                let mut state = session_state
                    .lock()
                    .map_err(|_| anyhow!("session state lock poisoned"))?;
                let SessionState::LlamaCpp(state) = &mut *state else {
                    return Err(anyhow!("expected llama.cpp session state"));
                };
                state.prompt_prefix = history_text;
                state.history = history_tokens;

                Ok(GenerationOutput {
                    text: output.text,
                    finish_reason: output.finish_reason,
                    usage: Usage {
                        prompt_tokens: *prompt_tokens_len,
                        completion_tokens: completion_tokens.len(),
                        total_tokens: *prompt_tokens_len + completion_tokens.len(),
                    },
                })
            }
        }
    }
}

fn completed_stream(output: GenerationOutput) -> TextStream {
    TextStream {
        state: None,
        text: output.text.clone(),
        output: Some(output),
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
        options.top_p.is_finite() && (0.0..=1.0).contains(&options.top_p),
        "top_p must be finite and between 0.0 and 1.0"
    );
    ensure!(
        options.min_p.is_finite() && (0.0..=1.0).contains(&options.min_p),
        "min_p must be finite and between 0.0 and 1.0"
    );
    ensure!(
        options.repeat_penalty.is_finite() && options.repeat_penalty >= 0.0,
        "repeat_penalty must be finite and non-negative"
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

fn flatten_chat_messages(messages: &[ChatMessage]) -> String {
    let mut rendered = String::new();
    for message in messages {
        let role = match message.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        };
        rendered.push_str(role);
        rendered.push_str(": ");
        rendered.push_str(&message.content);
        rendered.push('\n');
    }
    rendered
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
}
