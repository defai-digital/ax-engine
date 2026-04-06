use std::sync::{Mutex, MutexGuard};

use ax_engine_sdk::{
    ChatMessage as SdkChatMessage, ChatRole as SdkChatRole, FinishReason as SdkFinishReason,
    GenerationOptions, GenerationOutput as SdkGenerationOutput, Session as SdkSession,
    TextStream as SdkTextStream,
};
use pyo3::prelude::*;

use crate::errors::{py_runtime_error, py_value_error};
use crate::gil::allow_threads_unsend;

fn option_state_is_closed<T>(state: &Mutex<Option<T>>) -> bool {
    state.lock().map(|guard| guard.is_none()).unwrap_or(true)
}

fn validate_generation_inputs(
    max_tokens: usize,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    repeat_penalty: f32,
    repeat_last_n: i32,
) -> Result<(), String> {
    if max_tokens == 0 {
        return Err("max_tokens must be greater than zero".to_string());
    }
    if !temperature.is_finite() || temperature < 0.0 {
        return Err("temperature must be finite and non-negative".to_string());
    }
    if top_k < -1 {
        return Err("top_k must be >= -1 (-1 = disabled)".to_string());
    }
    if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
        return Err("top_p must be finite and between 0.0 and 1.0".to_string());
    }
    if !min_p.is_finite() || !(0.0..=1.0).contains(&min_p) {
        return Err("min_p must be finite and between 0.0 and 1.0".to_string());
    }
    if !repeat_penalty.is_finite() || repeat_penalty <= 0.0 {
        return Err("repeat_penalty must be finite and greater than zero".to_string());
    }
    if repeat_last_n < -1 {
        return Err("repeat_last_n must be -1 or greater".to_string());
    }
    Ok(())
}

#[derive(FromPyObject)]
pub struct PythonChatMessage {
    role: String,
    content: String,
}

/// Final result of a completed generation, including the generated text,
/// the reason generation stopped, and token usage counts.
#[pyclass(module = "ax_engine")]
pub struct GenerationResult {
    /// The generated text.
    #[pyo3(get)]
    pub text: String,
    /// Why generation ended: ``"stop"`` (EOS or stop string hit) or
    /// ``"length"`` (``max_tokens`` exhausted).
    #[pyo3(get)]
    pub finish_reason: String,
    /// Number of tokens in the prompt.
    #[pyo3(get)]
    pub prompt_tokens: usize,
    /// Number of tokens generated.
    #[pyo3(get)]
    pub completion_tokens: usize,
    /// ``prompt_tokens + completion_tokens``.
    #[pyo3(get)]
    pub total_tokens: usize,
}

impl GenerationResult {
    fn from_sdk_output(output: &SdkGenerationOutput) -> Self {
        Self {
            text: output.text.clone(),
            finish_reason: match output.finish_reason {
                SdkFinishReason::Stop => "stop".to_string(),
                SdkFinishReason::Length => "length".to_string(),
            },
            prompt_tokens: output.usage.prompt_tokens,
            completion_tokens: output.usage.completion_tokens,
            total_tokens: output.usage.total_tokens,
        }
    }
}

#[pymethods]
impl GenerationResult {
    fn __repr__(&self) -> String {
        format!(
            "GenerationResult(finish_reason={:?}, prompt_tokens={}, completion_tokens={})",
            self.finish_reason, self.prompt_tokens, self.completion_tokens
        )
    }
}

#[pyclass(unsendable, module = "ax_engine")]
pub struct Session {
    pub(crate) inner: Mutex<Option<SdkSession>>,
}

#[pyclass(unsendable, module = "ax_engine")]
pub struct TextStream {
    pub(crate) inner: Mutex<Option<SdkTextStream>>,
    /// Captured after the stream is fully consumed; readable via `.result`.
    stored_output: Mutex<Option<SdkGenerationOutput>>,
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl Session {
    #[getter]
    pub fn context_length(&self) -> PyResult<usize> {
        self.with_session(|session| Ok(session.context_length()))
            .map_err(py_runtime_error)
    }

    #[getter]
    pub fn position(&self) -> PyResult<usize> {
        self.with_session(|session| session.position())
            .map_err(py_runtime_error)
    }

    #[getter]
    pub fn closed(&self) -> bool {
        option_state_is_closed(&self.inner)
    }

    /// Generate text and return only the output string.
    #[pyo3(signature = (
        prompt,
        max_tokens = 128,
        temperature = 0.8,
        top_k = 40,
        top_p = 0.9,
        min_p = 0.0,
        repeat_penalty = 1.0,
        repeat_last_n = 64,
        frequency_penalty = 0.0,
        presence_penalty = 0.0,
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
        repeat_last_n: i32,
        frequency_penalty: f32,
        presence_penalty: f32,
        stop: Option<Vec<String>>,
        seed: Option<u64>,
    ) -> PyResult<String> {
        let options = generation_options(
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repeat_penalty,
            repeat_last_n,
            frequency_penalty,
            presence_penalty,
            stop,
            seed,
        )
        .map_err(py_value_error)?;
        self.do_generate(py, prompt, options)
            .map(|output| output.text)
            .map_err(py_runtime_error)
    }

    /// Generate text and return a :class:`GenerationResult` with usage and finish reason.
    #[pyo3(signature = (
        prompt,
        max_tokens = 128,
        temperature = 0.8,
        top_k = 40,
        top_p = 0.9,
        min_p = 0.0,
        repeat_penalty = 1.0,
        repeat_last_n = 64,
        frequency_penalty = 0.0,
        presence_penalty = 0.0,
        stop = None,
        seed = None
    ))]
    pub fn generate_full(
        &self,
        py: Python<'_>,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        repeat_penalty: f32,
        repeat_last_n: i32,
        frequency_penalty: f32,
        presence_penalty: f32,
        stop: Option<Vec<String>>,
        seed: Option<u64>,
    ) -> PyResult<GenerationResult> {
        let options = generation_options(
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repeat_penalty,
            repeat_last_n,
            frequency_penalty,
            presence_penalty,
            stop,
            seed,
        )
        .map_err(py_value_error)?;
        self.do_generate(py, prompt, options)
            .map(|output| GenerationResult::from_sdk_output(&output))
            .map_err(py_runtime_error)
    }

    /// Return a streaming iterator of text chunks.
    #[pyo3(signature = (
        prompt,
        max_tokens = 128,
        temperature = 0.8,
        top_k = 40,
        top_p = 0.9,
        min_p = 0.0,
        repeat_penalty = 1.0,
        repeat_last_n = 64,
        frequency_penalty = 0.0,
        presence_penalty = 0.0,
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
        repeat_last_n: i32,
        frequency_penalty: f32,
        presence_penalty: f32,
        stop: Option<Vec<String>>,
        seed: Option<u64>,
    ) -> PyResult<TextStream> {
        let options = generation_options(
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repeat_penalty,
            repeat_last_n,
            frequency_penalty,
            presence_penalty,
            stop,
            seed,
        )
        .map_err(py_value_error)?;
        allow_threads_unsend(py, || {
            self.with_session(|session| session.stream(prompt, options))
        })
        .map(|stream| TextStream {
            inner: Mutex::new(Some(stream)),
            stored_output: Mutex::new(None),
        })
        .map_err(py_runtime_error)
    }

    /// Run a chat turn and return only the assistant reply string.
    #[pyo3(signature = (
        messages,
        max_tokens = 128,
        temperature = 0.8,
        top_k = 40,
        top_p = 0.9,
        min_p = 0.0,
        repeat_penalty = 1.0,
        repeat_last_n = 64,
        frequency_penalty = 0.0,
        presence_penalty = 0.0,
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
        repeat_last_n: i32,
        frequency_penalty: f32,
        presence_penalty: f32,
        stop: Option<Vec<String>>,
        seed: Option<u64>,
    ) -> PyResult<String> {
        let messages = normalize_messages(&messages).map_err(py_value_error)?;
        let options = generation_options(
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repeat_penalty,
            repeat_last_n,
            frequency_penalty,
            presence_penalty,
            stop,
            seed,
        )
        .map_err(py_value_error)?;
        self.do_chat(py, &messages, options)
            .map(|output| output.text)
            .map_err(py_runtime_error)
    }

    /// Run a chat turn and return a :class:`GenerationResult` with usage and finish reason.
    #[pyo3(signature = (
        messages,
        max_tokens = 128,
        temperature = 0.8,
        top_k = 40,
        top_p = 0.9,
        min_p = 0.0,
        repeat_penalty = 1.0,
        repeat_last_n = 64,
        frequency_penalty = 0.0,
        presence_penalty = 0.0,
        stop = None,
        seed = None
    ))]
    pub fn chat_full(
        &self,
        py: Python<'_>,
        messages: Vec<PythonChatMessage>,
        max_tokens: usize,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        repeat_penalty: f32,
        repeat_last_n: i32,
        frequency_penalty: f32,
        presence_penalty: f32,
        stop: Option<Vec<String>>,
        seed: Option<u64>,
    ) -> PyResult<GenerationResult> {
        let messages = normalize_messages(&messages).map_err(py_value_error)?;
        let options = generation_options(
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repeat_penalty,
            repeat_last_n,
            frequency_penalty,
            presence_penalty,
            stop,
            seed,
        )
        .map_err(py_value_error)?;
        self.do_chat(py, &messages, options)
            .map(|output| GenerationResult::from_sdk_output(&output))
            .map_err(py_runtime_error)
    }

    pub fn reset(&self) -> PyResult<()> {
        self.with_session(|session| session.reset())
            .map_err(py_runtime_error)
    }

    pub fn close(&self) {
        if let Ok(mut state) = self.inner.lock() {
            state.take();
        }
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
        match self.with_session(|session| session.position()) {
            Ok(position) => match self.with_session(|session| Ok(session.context_length())) {
                Ok(context_length) => format!(
                    "Session(position={}, context_length={}, closed=False)",
                    position, context_length
                ),
                Err(_) => "Session(closed=True)".to_string(),
            },
            Err(_) => "Session(closed=True)".to_string(),
        }
    }
}

impl Session {
    fn do_generate(
        &self,
        py: Python<'_>,
        prompt: &str,
        options: GenerationOptions,
    ) -> anyhow::Result<SdkGenerationOutput> {
        allow_threads_unsend(py, || {
            self.with_session(|session| session.generate(prompt, options))
        })
    }

    fn do_chat(
        &self,
        py: Python<'_>,
        messages: &[SdkChatMessage],
        options: GenerationOptions,
    ) -> anyhow::Result<SdkGenerationOutput> {
        allow_threads_unsend(py, || {
            self.with_session(|session| session.chat(messages, options))
        })
    }

    fn with_session<T>(
        &self,
        f: impl FnOnce(&SdkSession) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        let state = self.lock_state()?;
        let session = state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("session is closed"))?;
        f(session)
    }

    fn lock_state(&self) -> anyhow::Result<MutexGuard<'_, Option<SdkSession>>> {
        self.inner
            .lock()
            .map_err(|_| anyhow::anyhow!("session state lock poisoned"))
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

    /// Returns the :class:`GenerationResult` once the stream is fully consumed,
    /// or ``None`` if streaming is still in progress.
    #[getter]
    pub fn result(&self) -> PyResult<Option<GenerationResult>> {
        let stored = self
            .stored_output
            .lock()
            .map_err(|_| py_runtime_error(anyhow::anyhow!("stream output lock poisoned")))?;
        Ok(stored.as_ref().map(GenerationResult::from_sdk_output))
    }

    fn __repr__(&self) -> String {
        match self.inner.lock() {
            Ok(state) => {
                if state.is_some() {
                    "TextStream(done=False)".to_string()
                } else {
                    "TextStream(done=True)".to_string()
                }
            }
            Err(_) => "TextStream(done=True)".to_string(),
        }
    }
}

impl TextStream {
    fn next_chunk(&self) -> anyhow::Result<Option<String>> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("stream state lock poisoned"))?;
        let stream = match state.as_mut() {
            Some(stream) => stream,
            None => return Ok(None),
        };
        let next = stream.next_chunk()?;
        if next.is_none() {
            // Capture the output before dropping the stream so `.result` works after iteration.
            if let Some(output) = stream.output().cloned()
                && let Ok(mut stored) = self.stored_output.lock()
            {
                *stored = Some(output);
            }
            state.take();
        }
        Ok(next)
    }
}

#[allow(clippy::too_many_arguments)]
fn generation_options(
    max_tokens: usize,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    repeat_penalty: f32,
    repeat_last_n: i32,
    frequency_penalty: f32,
    presence_penalty: f32,
    stop: Option<Vec<String>>,
    seed: Option<u64>,
) -> Result<GenerationOptions, String> {
    validate_generation_inputs(
        max_tokens,
        temperature,
        top_k,
        top_p,
        min_p,
        repeat_penalty,
        repeat_last_n,
    )?;
    Ok(GenerationOptions {
        max_tokens,
        temperature,
        top_k,
        top_p,
        min_p,
        repeat_penalty,
        repeat_last_n,
        frequency_penalty,
        presence_penalty,
        stop_strings: stop.unwrap_or_default(),
        seed,
    })
}

fn normalize_messages(messages: &[PythonChatMessage]) -> Result<Vec<SdkChatMessage>, String> {
    if messages.is_empty() {
        return Err("messages must not be empty".to_string());
    }
    messages
        .iter()
        .map(|message| {
            let role = message
                .role
                .parse::<SdkChatRole>()
                .map_err(|err| err.to_string())?;
            Ok(SdkChatMessage::new(role, message.content.clone()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_generation_inputs_rejects_zero_max_tokens() {
        let err = validate_generation_inputs(0, 0.8, 40, 0.9, 0.0, 1.0, 64).unwrap_err();
        assert!(err.contains("max_tokens"));
    }

    #[test]
    fn test_validate_generation_inputs_rejects_invalid_top_k() {
        let err = validate_generation_inputs(16, 0.8, -2, 0.9, 0.0, 1.0, 64).unwrap_err();
        assert!(err.contains("top_k"));
    }

    #[test]
    fn test_validate_generation_inputs_accepts_top_k_disabled() {
        assert!(validate_generation_inputs(16, 0.8, -1, 0.9, 0.0, 1.0, 64).is_ok());
    }

    #[test]
    fn test_validate_generation_inputs_rejects_invalid_repeat_last_n() {
        let err = validate_generation_inputs(16, 0.8, 40, 0.9, 0.0, 1.0, -2).unwrap_err();
        assert!(err.contains("repeat_last_n"));
    }

    #[test]
    fn test_generation_options_rejects_zero_repeat_penalty() {
        let err =
            generation_options(16, 0.8, 40, 0.9, 0.0, 0.0, 64, 0.0, 0.0, None, None).unwrap_err();
        assert!(err.contains("repeat_penalty"));
    }

    #[test]
    fn test_normalize_messages_rejects_empty_message_list() {
        let err = normalize_messages(&[]).unwrap_err();
        assert!(err.contains("messages must not be empty"));
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

    #[test]
    fn test_generation_result_from_sdk_output_maps_finish_reason_stop() {
        use ax_engine_sdk::{FinishReason, GenerationOutput, Usage};
        let output = GenerationOutput {
            text: "hello".to_string(),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 3,
                total_tokens: 8,
            },
        };
        let result = GenerationResult::from_sdk_output(&output);
        assert_eq!(result.finish_reason, "stop");
        assert_eq!(result.prompt_tokens, 5);
        assert_eq!(result.total_tokens, 8);
    }

    #[test]
    fn test_generation_result_from_sdk_output_maps_finish_reason_length() {
        use ax_engine_sdk::{FinishReason, GenerationOutput, Usage};
        let output = GenerationOutput {
            text: "hi".to_string(),
            finish_reason: FinishReason::Length,
            usage: Usage {
                prompt_tokens: 2,
                completion_tokens: 1,
                total_tokens: 3,
            },
        };
        let result = GenerationResult::from_sdk_output(&output);
        assert_eq!(result.finish_reason, "length");
    }
}
