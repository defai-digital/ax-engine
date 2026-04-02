use std::sync::{Mutex, MutexGuard};

use ax_engine_sdk::{
    ChatMessage as SdkChatMessage, ChatRole as SdkChatRole, GenerationOptions,
    Session as SdkSession, TextStream as SdkTextStream,
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
    top_p: f32,
    min_p: f32,
    repeat_penalty: f32,
) -> Result<(), String> {
    if max_tokens == 0 {
        return Err("max_tokens must be greater than zero".to_string());
    }
    if !temperature.is_finite() || temperature < 0.0 {
        return Err("temperature must be finite and non-negative".to_string());
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
    Ok(())
}

#[derive(FromPyObject)]
pub struct PythonChatMessage {
    role: String,
    content: String,
}

#[pyclass(unsendable, module = "ax_engine")]
pub struct Session {
    pub(crate) inner: Mutex<Option<SdkSession>>,
}

#[pyclass(unsendable, module = "ax_engine")]
pub struct TextStream {
    pub(crate) inner: Mutex<Option<SdkTextStream>>,
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
        let options = generation_options(
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repeat_penalty,
            stop,
            seed,
        )
        .map_err(py_value_error)?;
        allow_threads_unsend(py, || {
            self.with_session(|session| session.generate(prompt, options))
                .map(|output| output.text)
        })
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
        let options = generation_options(
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repeat_penalty,
            stop,
            seed,
        )
        .map_err(py_value_error)?;
        allow_threads_unsend(py, || {
            self.with_session(|session| session.stream(prompt, options))
        })
        .map(|stream| TextStream {
            inner: Mutex::new(Some(stream)),
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
        let messages = normalize_messages(&messages).map_err(py_value_error)?;
        let options = generation_options(
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repeat_penalty,
            stop,
            seed,
        )
        .map_err(py_value_error)?;
        allow_threads_unsend(py, || {
            self.with_session(|session| session.chat(&messages, options))
                .map(|output| output.text)
        })
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

#[pymethods]
impl TextStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<String>> {
        allow_threads_unsend(py, || self.next_chunk()).map_err(py_runtime_error)
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

impl Session {
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
    stop: Option<Vec<String>>,
    seed: Option<u64>,
) -> Result<GenerationOptions, String> {
    validate_generation_inputs(max_tokens, temperature, top_p, min_p, repeat_penalty)?;
    Ok(GenerationOptions {
        max_tokens,
        temperature,
        top_k,
        top_p,
        min_p,
        repeat_penalty,
        stop_strings: stop.unwrap_or_default(),
        seed,
        ..GenerationOptions::default()
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
        let err = validate_generation_inputs(0, 0.8, 0.9, 0.0, 1.0).unwrap_err();
        assert!(err.contains("max_tokens"));
    }

    #[test]
    fn test_generation_options_rejects_zero_repeat_penalty() {
        let err = generation_options(16, 0.8, 40, 0.9, 0.0, 0.0, None, None).unwrap_err();
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
}
