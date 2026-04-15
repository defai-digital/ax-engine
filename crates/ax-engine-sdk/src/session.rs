use std::str::FromStr;
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{Context, anyhow, ensure};
use ax_engine_core::chat::{
    ChatMessage as CoreChatMessage, ChatRenderOptions, ChatRole as CoreChatRole,
    render_chat_messages, render_infill_prompt,
};
use ax_engine_core::kv::{ModelKv, ModelKvSnapshot};
use ax_engine_core::model::WeightStore;
use ax_engine_core::sampling::{Sampler, SamplingConfig};

use crate::model::Model;

struct SessionState {
    kv: ModelKv,
    history: Vec<u32>,
    checkpoints: Vec<SessionCheckpoint>,
}

pub struct Session {
    model: Model,
    state: Arc<Mutex<SessionState>>,
    max_context_tokens: usize,
    default_seed: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PromptCacheStats {
    pub cached_tokens: usize,
    pub prompt_tokens: usize,
    pub prompt_tokens_evaluated: usize,
}

#[derive(Debug, Clone)]
pub struct SessionSnapshot {
    history: Vec<u32>,
    kv: Option<ModelKvSnapshot>,
    checkpoints: Vec<SessionCheckpoint>,
}

#[derive(Debug, Clone)]
struct SessionCheckpoint {
    position: usize,
    kv: ModelKvSnapshot,
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

struct PreparedPromptState {
    logits: Vec<f32>,
    decode_position: usize,
    prompt_tokens: usize,
    remaining_decode_capacity: usize,
    cache_stats: PromptCacheStats,
}

const QWEN35_CHECKPOINT_INTERVAL: usize = 64;
const QWEN35_TAIL_CHECKPOINT_TOKENS: usize = 32;

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
        native.model.prepare_runtime_for_weights(&weights)?;
        let kv = native.model.create_model_kv_for_weights(&weights);
        let state = SessionState {
            kv,
            history: Vec::new(),
            checkpoints: Vec::new(),
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
        state.checkpoints.clear();
        Ok(())
    }

    pub fn history_tokens(&self) -> anyhow::Result<Vec<u32>> {
        Ok(self.lock_state()?.history.clone())
    }

    pub fn snapshot(&self) -> anyhow::Result<SessionSnapshot> {
        let mut state = self.lock_state()?;
        Ok(SessionSnapshot {
            history: state.history.clone(),
            kv: state.kv.snapshot(),
            checkpoints: state.checkpoints.clone(),
        })
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
        let prompt_tokens = self.model.tokenize(prompt, add_bos);
        self.start_stream_tokens(prompt_tokens, options)
    }

    pub fn infill(
        &self,
        prefix: &str,
        suffix: &str,
        options: GenerationOptions,
    ) -> anyhow::Result<GenerationOutput> {
        self.stream_infill(prefix, suffix, options)?.into_output()
    }

    pub fn infill_with_stats(
        &self,
        prefix: &str,
        suffix: &str,
        options: GenerationOptions,
    ) -> anyhow::Result<(GenerationOutput, PromptCacheStats)> {
        let (stream, cache_stats) = self.stream_infill_with_stats(prefix, suffix, options)?;
        Ok((stream.into_output()?, cache_stats))
    }

    pub fn stream_infill(
        &self,
        prefix: &str,
        suffix: &str,
        options: GenerationOptions,
    ) -> anyhow::Result<TextStream> {
        let (stream, _) = self.stream_infill_with_stats(prefix, suffix, options)?;
        Ok(stream)
    }

    pub fn stream_infill_with_stats(
        &self,
        prefix: &str,
        suffix: &str,
        options: GenerationOptions,
    ) -> anyhow::Result<(TextStream, PromptCacheStats)> {
        let prompt_tokens = self.tokenize_infill_prompt(prefix, suffix)?;
        validate_generation_options(&options)?;
        let prepared = self.prepare_prompt_context(&prompt_tokens, true)?;
        let cache_stats = prepared.cache_stats;
        let stream = self.build_text_stream_from_prepared(prepared, options)?;
        Ok((stream, cache_stats))
    }

    pub fn generate_tokens(
        &self,
        prompt_tokens: &[u32],
        options: GenerationOptions,
    ) -> anyhow::Result<GenerationOutput> {
        self.stream_tokens(prompt_tokens, options)?.into_output()
    }

    pub fn generate_with_prefix_reuse(
        &self,
        prompt_tokens: &[u32],
        options: GenerationOptions,
    ) -> anyhow::Result<(GenerationOutput, PromptCacheStats)> {
        let (stream, cache_stats) = self.stream_with_prefix_reuse(prompt_tokens, options)?;
        Ok((stream.into_output()?, cache_stats))
    }

    pub fn stream_tokens(
        &self,
        prompt_tokens: &[u32],
        options: GenerationOptions,
    ) -> anyhow::Result<TextStream> {
        self.start_stream_tokens(prompt_tokens.to_vec(), options)
    }

    pub fn stream_with_prefix_reuse(
        &self,
        prompt_tokens: &[u32],
        options: GenerationOptions,
    ) -> anyhow::Result<(TextStream, PromptCacheStats)> {
        validate_generation_options(&options)?;
        let prepared = self.prepare_prompt_context(prompt_tokens, true)?;
        let cache_stats = prepared.cache_stats;
        let stream = self.build_text_stream_from_prepared(prepared, options)?;
        Ok((stream, cache_stats))
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
        let prompt_tokens = self.model.tokenize(&rendered, true);
        self.start_stream_tokens(prompt_tokens, options)
    }

    pub fn load_prompt_tokens(&self, prompt_tokens: &[u32]) -> anyhow::Result<PromptCacheStats> {
        if prompt_tokens.is_empty() {
            self.reset()?;
            return Ok(PromptCacheStats::default());
        }

        Ok(self
            .prepare_prompt_context(prompt_tokens, false)?
            .cache_stats)
    }

    pub fn restore_snapshot(&self, snapshot: &SessionSnapshot) -> anyhow::Result<PromptCacheStats> {
        if snapshot.history.is_empty() {
            self.reset()?;
            return Ok(PromptCacheStats::default());
        }

        if let Some(kv_snapshot) = snapshot.kv.as_ref() {
            let mut state = self.lock_state()?;
            state.kv.restore_snapshot(kv_snapshot)?;
            state.history = snapshot.history.clone();
            state.checkpoints = snapshot.checkpoints.clone();
            return Ok(PromptCacheStats {
                cached_tokens: snapshot.history.len(),
                prompt_tokens: snapshot.history.len(),
                prompt_tokens_evaluated: 0,
            });
        }

        self.load_prompt_tokens(&snapshot.history)
    }

    fn start_stream_tokens(
        &self,
        prompt_tokens: Vec<u32>,
        options: GenerationOptions,
    ) -> anyhow::Result<TextStream> {
        validate_generation_options(&options)?;
        let prepared = self.prepare_append_prompt(&prompt_tokens)?;
        self.build_text_stream_from_prepared(prepared, options)
    }

    fn lock_state(&self) -> anyhow::Result<MutexGuard<'_, SessionState>> {
        self.state
            .lock()
            .map_err(|_| anyhow!("session state lock poisoned"))
    }

    fn prepare_append_prompt(&self, prompt_tokens: &[u32]) -> anyhow::Result<PreparedPromptState> {
        self.prepare_prompt(prompt_tokens, None)
    }

    fn tokenize_infill_prompt(&self, prefix: &str, suffix: &str) -> anyhow::Result<Vec<u32>> {
        let rendered = render_infill_prompt(prefix, suffix, self.model.tokenizer())?;
        let prompt_tokens = self.model.tokenize_with_options(&rendered, true, true);
        ensure!(
            !prompt_tokens.is_empty(),
            "prompt produced no tokens; provide a non-empty prompt"
        );
        Ok(prompt_tokens)
    }

    fn prepare_prompt_context(
        &self,
        prompt_tokens: &[u32],
        recompute_last_token: bool,
    ) -> anyhow::Result<PreparedPromptState> {
        let cached_tokens = reusable_prefix_len_from_history(
            &self.lock_state()?.history,
            prompt_tokens,
            recompute_last_token,
        );
        self.prepare_prompt(prompt_tokens, Some(cached_tokens))
    }

    fn prepare_prompt(
        &self,
        prompt_tokens: &[u32],
        cached_tokens: Option<usize>,
    ) -> anyhow::Result<PreparedPromptState> {
        ensure!(
            !prompt_tokens.is_empty(),
            "prompt produced no tokens; provide a non-empty prompt"
        );

        let mut state_guard = self.lock_state()?;
        let mut reused_tokens = state_guard.history.len();
        let tokens_to_evaluate = if let Some(cached_tokens) = cached_tokens {
            ensure!(
                cached_tokens <= prompt_tokens.len(),
                "cached prompt prefix exceeds prompt length"
            );
            if state_guard.history.len() > cached_tokens {
                if supports_checkpointed_rewind(&state_guard.kv) {
                    let original_history = state_guard.history.clone();
                    reused_tokens =
                        restore_prefix_from_checkpoint(&mut state_guard, cached_tokens)?;
                    let mut replay = Vec::with_capacity(prompt_tokens.len() - reused_tokens);
                    replay.extend_from_slice(&original_history[reused_tokens..cached_tokens]);
                    replay.extend_from_slice(&prompt_tokens[cached_tokens..]);
                    replay
                } else {
                    state_guard.kv.truncate_to(cached_tokens);
                    state_guard.history.truncate(cached_tokens);
                    truncate_checkpoints_to(&mut state_guard, cached_tokens);
                    reused_tokens = cached_tokens;
                    prompt_tokens[cached_tokens..].to_vec()
                }
            } else {
                reused_tokens = cached_tokens;
                prompt_tokens[cached_tokens..].to_vec()
            }
        } else {
            prompt_tokens[reused_tokens..].to_vec()
        };

        let remaining_decode_capacity = remaining_decode_capacity(
            tokens_to_evaluate.len(),
            reused_tokens,
            self.max_context_tokens,
        )?;

        let logits =
            prefill_tokens_with_checkpoints(&self.model, &mut state_guard, &tokens_to_evaluate)
                .context("prefill forward pass failed")?;

        Ok(PreparedPromptState {
            logits,
            decode_position: reused_tokens + tokens_to_evaluate.len(),
            prompt_tokens: prompt_tokens.len(),
            remaining_decode_capacity,
            cache_stats: PromptCacheStats {
                cached_tokens: reused_tokens,
                prompt_tokens: prompt_tokens.len(),
                prompt_tokens_evaluated: tokens_to_evaluate.len(),
            },
        })
    }

    fn build_text_stream_from_prepared(
        &self,
        prepared: PreparedPromptState,
        options: GenerationOptions,
    ) -> anyhow::Result<TextStream> {
        let PreparedPromptState {
            mut logits,
            decode_position,
            prompt_tokens,
            remaining_decode_capacity,
            ..
        } = prepared;
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
                prompt_tokens,
                completion_tokens: 0,
            }
        } else {
            let mut sampler = Sampler::new(sampling);
            let first_token = sampler.sample(&mut logits, &self.lock_state()?.history);
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
                prompt_tokens,
                completion_tokens: 0,
            }
        };

        Ok(TextStream {
            state: Some(live_state),
            text: String::new(),
            output: None,
        })
    }
}

fn supports_checkpointed_rewind(kv: &ModelKv) -> bool {
    matches!(kv, ModelKv::Qwen35(_))
}

fn truncate_checkpoints_to(state: &mut SessionState, position: usize) {
    state
        .checkpoints
        .retain(|checkpoint| checkpoint.position <= position);
}

fn restore_prefix_from_checkpoint(
    state: &mut SessionState,
    target_prefix_len: usize,
) -> anyhow::Result<usize> {
    ensure!(
        target_prefix_len <= state.history.len(),
        "target prefix length exceeds history length"
    );

    let restore_position = state
        .checkpoints
        .iter()
        .rev()
        .find(|checkpoint| checkpoint.position <= target_prefix_len)
        .map(|checkpoint| checkpoint.position)
        .unwrap_or(0);

    if restore_position == 0 {
        state.kv.clear();
        state.history.clear();
        state.checkpoints.clear();
        return Ok(0);
    }

    let checkpoint = state
        .checkpoints
        .iter()
        .rev()
        .find(|checkpoint| checkpoint.position == restore_position)
        .cloned()
        .ok_or_else(|| anyhow!("missing checkpoint at position {restore_position}"))?;
    state.kv.restore_snapshot(&checkpoint.kv)?;
    state.history.truncate(restore_position);
    truncate_checkpoints_to(state, restore_position);
    Ok(restore_position)
}

fn prefill_tokens_with_checkpoints(
    model: &Model,
    state: &mut SessionState,
    tokens: &[u32],
) -> anyhow::Result<Vec<f32>> {
    let native = model.native();
    let weights = WeightStore::new(&native.mapped);
    let mut logits = vec![0.0f32; model.config().vocab_size as usize];

    if tokens.is_empty() {
        return Ok(logits);
    }

    if supports_checkpointed_rewind(&state.kv) {
        let tail_start = tokens.len().saturating_sub(QWEN35_TAIL_CHECKPOINT_TOKENS);
        for chunk in tokens[..tail_start].chunks(QWEN35_CHECKPOINT_INTERVAL) {
            if chunk.is_empty() {
                continue;
            }
            native
                .model
                .forward_batch(chunk, &mut state.kv, &weights, &mut logits)?;
            state.history.extend_from_slice(chunk);
            maybe_record_checkpoint(state, true)?;
        }

        for &token in &tokens[tail_start..] {
            native.model.forward_batch(
                std::slice::from_ref(&token),
                &mut state.kv,
                &weights,
                &mut logits,
            )?;
            state.history.push(token);
            maybe_record_checkpoint(state, true)?;
        }
        return Ok(logits);
    }

    native
        .model
        .forward_batch(tokens, &mut state.kv, &weights, &mut logits)?;
    state.history.extend_from_slice(tokens);
    Ok(logits)
}

fn maybe_record_checkpoint(state: &mut SessionState, force: bool) -> anyhow::Result<()> {
    if !state.kv.supports_snapshot() || state.history.is_empty() {
        return Ok(());
    }

    let position = state.history.len();
    if !force && !should_keep_checkpoint_position(position, position) {
        return Ok(());
    }

    let Some(snapshot) = state.kv.snapshot() else {
        return Ok(());
    };
    if let Some(last) = state.checkpoints.last_mut()
        && last.position == position
    {
        last.kv = snapshot;
    } else {
        state.checkpoints.push(SessionCheckpoint {
            position,
            kv: snapshot,
        });
    }
    prune_checkpoints(state);
    Ok(())
}

fn prune_checkpoints(state: &mut SessionState) {
    let current_len = state.history.len();
    state
        .checkpoints
        .retain(|checkpoint| should_keep_checkpoint_position(checkpoint.position, current_len));
}

fn should_keep_checkpoint_position(position: usize, current_len: usize) -> bool {
    position == current_len
        || position.is_multiple_of(QWEN35_CHECKPOINT_INTERVAL)
        || position >= current_len.saturating_sub(QWEN35_TAIL_CHECKPOINT_TOKENS)
}

impl SessionSnapshot {
    pub fn history_tokens(&self) -> &[u32] {
        &self.history
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
                let native = self.model.native();
                let weights = WeightStore::new(&native.mapped);
                let should_materialize_final_token =
                    self.remaining_tokens == 0 && supports_checkpointed_rewind(&state.kv);

                if self.remaining_tokens == 0 && !should_materialize_final_token {
                    self.finish_reason = FinishReason::Length;
                    self.done = true;
                    self.next_token = None;
                } else {
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
                    maybe_record_checkpoint(&mut state, true)?;

                    if self.remaining_tokens == 0 {
                        self.finish_reason = FinishReason::Length;
                        self.done = true;
                        self.next_token = None;
                    } else {
                        self.next_token =
                            Some(self.sampler.sample(&mut self.logits, &state.history));
                    }
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

fn shared_prefix_len(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right.iter())
        .take_while(|(left, right)| left == right)
        .count()
}

fn reusable_prefix_len_from_history(
    history: &[u32],
    prompt_tokens: &[u32],
    recompute_last_token: bool,
) -> usize {
    let mut prefix_len = shared_prefix_len(history, prompt_tokens);
    if recompute_last_token && prefix_len == prompt_tokens.len() && prefix_len > 0 {
        prefix_len -= 1;
    }
    prefix_len
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

    #[test]
    fn test_shared_prefix_len_counts_common_tokens() {
        assert_eq!(shared_prefix_len(&[1, 2, 3], &[1, 2, 9]), 2);
        assert_eq!(shared_prefix_len(&[1, 2, 3], &[4, 5, 6]), 0);
    }

    #[test]
    fn test_reusable_prefix_len_forces_last_token_recompute_for_identical_prompt() {
        assert_eq!(
            reusable_prefix_len_from_history(&[1, 2, 3], &[1, 2, 3], true),
            2
        );
        assert_eq!(
            reusable_prefix_len_from_history(&[1, 2, 3], &[1, 2, 3], false),
            3
        );
    }

    #[test]
    fn test_should_keep_checkpoint_position_keeps_interval_and_tail() {
        let current_len = 160;
        assert!(should_keep_checkpoint_position(64, current_len));
        assert!(should_keep_checkpoint_position(128, current_len));
        assert!(should_keep_checkpoint_position(159, current_len));
        assert!(!should_keep_checkpoint_position(95, current_len));
    }

    #[test]
    fn test_should_keep_checkpoint_position_always_keeps_current_position() {
        assert!(should_keep_checkpoint_position(17, 17));
    }

    #[test]
    fn test_session_exposes_infill_methods() {
        let infill_fn: fn(
            &Session,
            &str,
            &str,
            GenerationOptions,
        ) -> anyhow::Result<GenerationOutput> = Session::infill;
        #[allow(clippy::type_complexity)]
        let infill_with_stats_fn: fn(
            &Session,
            &str,
            &str,
            GenerationOptions,
        ) -> anyhow::Result<(
            GenerationOutput,
            PromptCacheStats,
        )> = Session::infill_with_stats;
        let stream_infill_fn: fn(
            &Session,
            &str,
            &str,
            GenerationOptions,
        ) -> anyhow::Result<TextStream> = Session::stream_infill;
        #[allow(clippy::type_complexity)]
        let stream_infill_with_stats_fn: fn(
            &Session,
            &str,
            &str,
            GenerationOptions,
        ) -> anyhow::Result<(
            TextStream,
            PromptCacheStats,
        )> = Session::stream_infill_with_stats;

        let _ = (
            infill_fn,
            infill_with_stats_fn,
            stream_infill_fn,
            stream_infill_with_stats_fn,
        );
    }

    #[test]
    #[ignore = "requires local GGUF FIM-capable model and Metal GPU"]
    fn test_infill_with_stats_reuses_prefix_cache() {
        let model_path = std::env::var("AX_ENGINE_TEST_FIM_MODEL")
            .expect("AX_ENGINE_TEST_FIM_MODEL must be set");
        let model =
            Model::load(model_path, crate::LoadOptions::default()).expect("test model should load");
        assert!(
            model.supports_infill(),
            "test model must expose native FIM tokens"
        );

        let session = model
            .session(SessionOptions::default())
            .expect("session should be created");
        let options = GenerationOptions::default()
            .max_tokens(1)
            .temperature(0.0)
            .top_k(1)
            .top_p(1.0)
            .seed(7);

        let (_, first_stats) = session
            .infill_with_stats("fn add(a: i32, b: i32) {\n", "\n}\n", options.clone())
            .expect("first infill should succeed");
        assert_eq!(first_stats.cached_tokens, 0);

        let (_, second_stats) = session
            .infill_with_stats("fn add(a: i32, b: i32) {\n", "\n}\n", options)
            .expect("second infill should succeed");
        assert!(
            second_stats.cached_tokens > 0,
            "expected cached prefix reuse on repeated infill, got {:?}",
            second_stats
        );
        assert!(
            second_stats.prompt_tokens_evaluated < second_stats.prompt_tokens,
            "expected reduced prompt evaluation on repeated infill, got {:?}",
            second_stats
        );
    }

    #[test]
    #[ignore = "requires local GGUF model and Metal GPU"]
    fn test_snapshot_restore_reuses_qwen35_prompt_cache() {
        let model_path =
            std::env::var("AX_ENGINE_TEST_MODEL").expect("AX_ENGINE_TEST_MODEL must be set");
        let model =
            Model::load(model_path, crate::LoadOptions::default()).expect("test model should load");
        assert_eq!(model.architecture(), "qwen35");

        let messages = [
            ChatMessage::system("Answer in six words max."),
            ChatMessage::user("Say hello to AX."),
        ];
        let rendered_messages = messages
            .iter()
            .map(|message| CoreChatMessage::new(message.role.into_core(), message.content.as_str()))
            .collect::<Vec<_>>();
        let rendered = render_chat_messages(
            &rendered_messages,
            model.architecture(),
            ChatRenderOptions::default(),
        );
        let prompt_tokens = model.tokenize_with_options(&rendered, true, true);
        let options = GenerationOptions::default()
            .max_tokens(8)
            .temperature(0.0)
            .top_k(1)
            .top_p(0.95)
            .min_p(0.05)
            .repeat_penalty(1.1)
            .seed(7);

        let session = model
            .session(SessionOptions::default())
            .expect("session should be created");
        let (first_output, first_stats) = session
            .generate_with_prefix_reuse(&prompt_tokens, options.clone())
            .expect("first generation should succeed");
        assert_eq!(first_stats.cached_tokens, 0);

        let snapshot = session.snapshot().expect("snapshot should succeed");
        assert_eq!(snapshot.history[..prompt_tokens.len()], prompt_tokens);
        assert!(
            snapshot
                .checkpoints
                .iter()
                .any(|checkpoint| checkpoint.position == prompt_tokens.len() - 1),
            "snapshot checkpoints should retain the reusable prompt tail"
        );

        let restored = model
            .session(SessionOptions::default())
            .expect("restored session should be created");
        restored
            .restore_snapshot(&snapshot)
            .expect("snapshot restore should succeed");

        let (second_output, second_stats) = restored
            .generate_with_prefix_reuse(&prompt_tokens, options)
            .expect("second generation should succeed");
        assert!(
            second_stats.cached_tokens > 0,
            "expected cached prefix reuse after snapshot restore, got {:?}",
            second_stats
        );
        assert_eq!(
            second_output.text, first_output.text,
            "snapshot restore must preserve generation semantics"
        );
    }
}
