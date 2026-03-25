use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{Context, ensure};
use ax_core::chat::{ChatMessage, ChatRenderOptions, ChatRole, render_chat_messages};
use ax_core::kv::ModelKv;
use ax_core::model::WeightStore;
use ax_core::sampling::{Sampler, SamplingConfig};
use pyo3::prelude::*;

use crate::errors::{py_runtime_error, py_value_error};
use crate::gil::allow_threads_unsend;
use crate::model::LoadedModel;

struct SessionState {
    kv: ModelKv,
    history: Vec<u32>,
}

struct GenerationOutput {
    text: String,
}

struct LiveTextStreamState {
    model: Arc<LoadedModel>,
    session_state: Arc<Mutex<Option<SessionState>>>,
    sampler: Sampler,
    next_token: Option<u32>,
    position: usize,
    remaining_tokens: usize,
    logits: Vec<f32>,
    pending: String,
    stop_strings: Vec<String>,
    done: bool,
}

#[derive(FromPyObject)]
pub struct PythonChatMessage {
    role: String,
    content: String,
}

#[pyclass(unsendable, module = "ax_engine")]
pub struct Session {
    model: Arc<LoadedModel>,
    state: Arc<Mutex<Option<SessionState>>>,
    max_context_tokens: usize,
    default_seed: Option<u64>,
}

#[pyclass(unsendable, module = "ax_engine")]
pub struct TextStream {
    state: Mutex<Option<LiveTextStreamState>>,
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl Session {
    #[getter]
    pub fn context_length(&self) -> PyResult<usize> {
        self.ensure_open()?;
        Ok(self.max_context_tokens)
    }

    #[getter]
    pub fn position(&self) -> PyResult<usize> {
        let state = self.lock_state().map_err(py_runtime_error)?;
        let state = state
            .as_ref()
            .ok_or_else(|| py_runtime_error(anyhow::anyhow!("session is closed")))?;
        Ok(state.history.len())
    }

    #[getter]
    pub fn closed(&self) -> bool {
        self.state
            .lock()
            .expect("session state lock poisoned")
            .is_none()
    }

    #[pyo3(signature = (
        prompt,
        max_tokens = 128,
        temperature = 0.8,
        top_k = 40,
        top_p = 0.9,
        min_p = 0.0,
        repeat_penalty = 1.0,
        stop = None,
        seed = None
    ))]
    pub fn generate(
        &self,
        py: Python<'_>,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        repeat_penalty: f32,
        stop: Option<Vec<String>>,
        seed: Option<u64>,
    ) -> PyResult<String> {
        let stop_strings = stop.unwrap_or_default();
        allow_threads_unsend(py, || {
            let stream = self.start_stream_inner(
                prompt,
                GenerationConfig {
                    max_tokens,
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    repeat_penalty,
                    stop_strings,
                    seed: seed.or(self.default_seed),
                },
            )?;
            self.collect_generation_output(stream)
        })
        .map(|result| result.text)
        .map_err(py_runtime_error)
    }

    #[pyo3(signature = (
        prompt,
        max_tokens = 128,
        temperature = 0.8,
        top_k = 40,
        top_p = 0.9,
        min_p = 0.0,
        repeat_penalty = 1.0,
        stop = None,
        seed = None
    ))]
    pub fn stream(
        &self,
        py: Python<'_>,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        repeat_penalty: f32,
        stop: Option<Vec<String>>,
        seed: Option<u64>,
    ) -> PyResult<TextStream> {
        let stop_strings = stop.unwrap_or_default();
        allow_threads_unsend(py, || {
            self.start_stream_inner(
                prompt,
                GenerationConfig {
                    max_tokens,
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    repeat_penalty,
                    stop_strings,
                    seed: seed.or(self.default_seed),
                },
            )
        })
        .map(|state| TextStream {
            state: Mutex::new(Some(state)),
        })
        .map_err(py_runtime_error)
    }

    #[pyo3(signature = (
        messages,
        max_tokens = 128,
        temperature = 0.8,
        top_k = 40,
        top_p = 0.9,
        min_p = 0.0,
        repeat_penalty = 1.0,
        stop = None,
        seed = None
    ))]
    pub fn chat(
        &self,
        py: Python<'_>,
        messages: Vec<PythonChatMessage>,
        max_tokens: usize,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        repeat_penalty: f32,
        stop: Option<Vec<String>>,
        seed: Option<u64>,
    ) -> PyResult<String> {
        let normalized = normalize_messages(&messages).map_err(py_value_error)?;
        let rendered_messages = normalized
            .iter()
            .map(|(role, content)| ChatMessage::new(*role, content.as_str()))
            .collect::<Vec<_>>();
        let rendered = render_chat_messages(
            &rendered_messages,
            self.model.model.arch_name(),
            ChatRenderOptions::default(),
        );
        self.reset();
        let stop_strings = stop.unwrap_or_default();
        allow_threads_unsend(py, || {
            let stream = self.start_stream_with_options_inner(
                &rendered,
                true,
                GenerationConfig {
                    max_tokens,
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    repeat_penalty,
                    stop_strings,
                    seed: seed.or(self.default_seed),
                },
            )?;
            self.collect_generation_output(stream)
        })
        .map(|result| result.text)
        .map_err(py_runtime_error)
    }

    pub fn reset(&self) {
        if let Ok(mut state) = self.lock_state()
            && let Some(state) = state.as_mut()
        {
            state.kv.clear();
            state.history.clear();
        }
    }

    pub fn close(&self) {
        let mut state = self.state.lock().expect("session state lock poisoned");
        state.take();
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) {
        self.close();
    }

    fn __repr__(&self) -> String {
        match self.lock_state() {
            Ok(state) => match state.as_ref() {
                Some(state) => format!(
                    "Session(position={}, context_length={}, closed=False)",
                    state.history.len(),
                    self.max_context_tokens
                ),
                None => "Session(closed=True)".to_string(),
            },
            Err(_) => "Session(closed=True)".to_string(),
        }
    }
}

#[pymethods]
impl TextStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<String>> {
        allow_threads_unsend(py, || self.next_chunk()).map_err(py_runtime_error)
    }

    fn __repr__(&self) -> String {
        match self.state.lock() {
            Ok(state) => match state.as_ref() {
                Some(state) => format!(
                    "TextStream(position={}, remaining_tokens={}, done={})",
                    state.position, state.remaining_tokens, state.done
                ),
                None => "TextStream(done=True)".to_string(),
            },
            Err(_) => "TextStream(done=True)".to_string(),
        }
    }
}

#[allow(clippy::arc_with_non_send_sync)]
impl Session {
    pub(crate) fn new(
        model: Arc<LoadedModel>,
        ctx_size: Option<usize>,
        seed: Option<u64>,
    ) -> anyhow::Result<Self> {
        let weights = WeightStore::new(&model.mapped);
        let kv = model.model.create_model_kv_for_weights(&weights);
        let max_context_tokens = ctx_size
            .unwrap_or(usize::MAX)
            .min(model.config.context_length as usize);
        Ok(Self {
            model,
            state: Arc::new(Mutex::new(Some(SessionState {
                kv,
                history: Vec::new(),
            }))),
            max_context_tokens,
            default_seed: seed,
        })
    }

    fn start_stream_inner(
        &self,
        prompt: &str,
        config: GenerationConfig,
    ) -> anyhow::Result<LiveTextStreamState> {
        let add_bos = self
            .lock_state()?
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("session is closed"))?
            .history
            .is_empty();
        self.start_stream_with_options_inner(prompt, add_bos, config)
    }

    fn start_stream_with_options_inner(
        &self,
        prompt: &str,
        add_bos: bool,
        config: GenerationConfig,
    ) -> anyhow::Result<LiveTextStreamState> {
        ensure!(
            config.max_tokens > 0,
            "max_tokens must be greater than zero"
        );
        ensure!(
            (0.0..=1.0).contains(&config.top_p),
            "top_p must be between 0.0 and 1.0"
        );
        ensure!(
            (0.0..=1.0).contains(&config.min_p),
            "min_p must be between 0.0 and 1.0"
        );
        ensure!(
            config.repeat_penalty >= 0.0,
            "repeat_penalty must be non-negative"
        );

        let weights = WeightStore::new(&self.model.mapped);
        let prompt_tokens = self.model.tokenizer.encode(prompt, add_bos);
        ensure!(
            !prompt_tokens.is_empty(),
            "prompt produced no tokens; provide a non-empty prompt"
        );

        let mut state_guard = self.lock_state()?;
        let state = state_guard
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("session is closed"))?;
        let position = state.history.len();
        let remaining_context = self.max_context_tokens.saturating_sub(position);
        ensure!(
            prompt_tokens.len() <= remaining_context,
            "prompt does not fit in remaining context: {} tokens requested, {} available",
            prompt_tokens.len(),
            remaining_context
        );

        let mut logits = vec![0.0f32; self.model.config.vocab_size as usize];
        self.model
            .model
            .forward_batch(&prompt_tokens, &mut state.kv, &weights, &mut logits)
            .context("prefill forward pass failed")?;
        state.history.extend_from_slice(&prompt_tokens);

        let decode_position = position + prompt_tokens.len();
        let remaining_decode_capacity = self.max_context_tokens.saturating_sub(decode_position);
        let max_tokens = config.max_tokens.min(remaining_decode_capacity);

        if max_tokens == 0 {
            return Ok(LiveTextStreamState {
                model: self.model.clone(),
                session_state: self.state.clone(),
                sampler: Sampler::new(SamplingConfig::default()),
                next_token: None,
                position: decode_position,
                remaining_tokens: 0,
                logits,
                pending: String::new(),
                stop_strings: config.stop_strings,
                done: true,
            });
        }

        let mut sampler = Sampler::new(SamplingConfig {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            min_p: config.min_p,
            repeat_penalty: config.repeat_penalty,
            seed: config.seed.unwrap_or(u64::MAX),
            ..SamplingConfig::default()
        });

        let first_token = sampler.sample(&mut logits, &state.history);

        Ok(LiveTextStreamState {
            model: self.model.clone(),
            session_state: self.state.clone(),
            sampler,
            next_token: Some(first_token),
            position: decode_position,
            remaining_tokens: max_tokens,
            logits,
            pending: String::new(),
            stop_strings: config.stop_strings,
            done: false,
        })
    }

    fn collect_generation_output(
        &self,
        mut stream: LiveTextStreamState,
    ) -> anyhow::Result<GenerationOutput> {
        let mut text = String::new();

        while let Some(chunk) = stream.next_chunk()? {
            text.push_str(&chunk);
        }

        Ok(GenerationOutput { text })
    }

    fn ensure_open(&self) -> PyResult<()> {
        let state = self.lock_state().map_err(py_runtime_error)?;
        if state.is_none() {
            return Err(py_runtime_error(anyhow::anyhow!("session is closed")));
        }
        Ok(())
    }

    fn lock_state(&self) -> anyhow::Result<MutexGuard<'_, Option<SessionState>>> {
        self.state
            .lock()
            .map_err(|_| anyhow::anyhow!("session state lock poisoned"))
    }
}

impl TextStream {
    fn next_chunk(&self) -> anyhow::Result<Option<String>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("stream state lock poisoned"))?;
        let stream = match state.as_mut() {
            Some(stream) => stream,
            None => return Ok(None),
        };
        let next = stream.next_chunk()?;
        if next.is_none() {
            state.take();
        }
        Ok(next)
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
                self.done = true;
                self.next_token = None;
                continue;
            }

            let token = match self.next_token {
                Some(token) => token,
                None => {
                    self.done = true;
                    continue;
                }
            };

            if self.model.tokenizer.is_eos(token) {
                self.done = true;
                self.next_token = None;
                continue;
            }

            let piece = render_token_text(&self.model.tokenizer, token);
            if !piece.is_empty() {
                self.pending.push_str(&piece);
            }

            if let Some(stop_at) = first_stop_match(&self.pending, &self.stop_strings) {
                let chunk = self.pending[..stop_at].to_string();
                self.pending.clear();
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
                let mut session_state = self
                    .session_state
                    .lock()
                    .map_err(|_| anyhow::anyhow!("session state lock poisoned"))?;
                let state = session_state
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("session is closed"))?;
                state.history.push(token);
                self.remaining_tokens = self.remaining_tokens.saturating_sub(1);

                if self.remaining_tokens == 0 {
                    self.done = true;
                    self.next_token = None;
                } else {
                    let weights = WeightStore::new(&self.model.mapped);
                    self.logits.fill(0.0);
                    self.model
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
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            state.take();
        }
    }
}

#[derive(Clone)]
struct GenerationConfig {
    max_tokens: usize,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    repeat_penalty: f32,
    stop_strings: Vec<String>,
    seed: Option<u64>,
}

fn normalize_messages(messages: &[PythonChatMessage]) -> Result<Vec<(ChatRole, String)>, String> {
    messages
        .iter()
        .map(|message| {
            let role = match message.role.trim().to_ascii_lowercase().as_str() {
                "system" | "developer" => ChatRole::System,
                "user" => ChatRole::User,
                "assistant" => ChatRole::Assistant,
                other => return Err(format!("unsupported chat role '{other}'")),
            };
            Ok((role, message.content.clone()))
        })
        .collect()
}

fn render_token_text(tokenizer: &ax_core::tokenizer::Tokenizer, token: u32) -> String {
    tokenizer
        .render_token(token)
        .unwrap_or_else(|| tokenizer.decode(&[token]))
}

fn first_stop_match(output: &str, stop_strings: &[String]) -> Option<usize> {
    stop_strings
        .iter()
        .filter(|stop| !stop.is_empty())
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
    fn test_longest_partial_stop_suffix_finds_prefix_overlap() {
        assert_eq!(
            longest_partial_stop_suffix("hello s", &[String::from("stop")]),
            1
        );
    }

    #[test]
    fn test_normalize_messages_accepts_common_roles() {
        let messages = vec![
            PythonChatMessage {
                role: "system".to_string(),
                content: "Be terse".to_string(),
            },
            PythonChatMessage {
                role: "user".to_string(),
                content: "Hi".to_string(),
            },
        ];
        let normalized = normalize_messages(&messages).unwrap();
        assert_eq!(normalized.len(), 2);
    }
}
