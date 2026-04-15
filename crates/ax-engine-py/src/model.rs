use std::str::FromStr;
use std::sync::{Mutex, Once};

use ax_engine_sdk::{
    BackendKind, LoadOptions, Model as SdkModel, ModelInfo, SessionOptions as SdkSessionOptions,
};
use pyo3::prelude::*;

use crate::errors::{py_runtime_error, py_value_error};
use crate::gil::allow_threads_unsend;
use crate::session::Session;

fn option_state_is_closed<T>(state: &Mutex<Option<T>>) -> bool {
    state.lock().map(|guard| guard.is_none()).unwrap_or(true)
}

#[pyclass(unsendable, module = "ax_engine")]
pub struct Model {
    pub(crate) inner: Mutex<Option<SdkModel>>,
}

#[pymethods]
impl Model {
    #[staticmethod]
    #[pyo3(signature = (path, backend = None))]
    pub fn load(py: Python<'_>, path: &str, backend: Option<String>) -> PyResult<Self> {
        init_tracing();

        let backend = backend
            .as_deref()
            .map(BackendKind::from_str)
            .transpose()
            .map_err(|err| py_value_error(err.to_string()))?
            .unwrap_or_default();

        py.allow_threads(|| {
            SdkModel::load(
                path,
                LoadOptions {
                    backend,
                    ..LoadOptions::default()
                },
            )
        })
        .map(|model| Self {
            inner: Mutex::new(Some(model)),
        })
        .map_err(py_runtime_error)
    }

    #[getter]
    pub fn architecture(&self) -> PyResult<String> {
        Ok(self.loaded_model()?.architecture().to_string())
    }

    #[getter]
    pub fn context_length(&self) -> PyResult<usize> {
        Ok(self.loaded_model()?.context_length())
    }

    #[getter]
    pub fn vocab_size(&self) -> PyResult<usize> {
        Ok(self.loaded_model()?.vocab_size())
    }

    #[getter]
    pub fn bos_token_id(&self) -> PyResult<u32> {
        Ok(self.loaded_model()?.bos_token_id())
    }

    #[getter]
    pub fn eos_token_id(&self) -> PyResult<u32> {
        Ok(self.loaded_model()?.eos_token_id())
    }

    #[getter]
    pub fn model_name(&self) -> PyResult<Option<String>> {
        Ok(self.loaded_model()?.model_name().map(str::to_string))
    }

    #[getter]
    pub fn support_note(&self) -> PyResult<Option<String>> {
        Ok(self.loaded_model()?.support_note().map(str::to_string))
    }

    #[getter]
    pub fn backend(&self) -> PyResult<String> {
        Ok("native".to_string())
    }

    #[getter]
    pub fn supports_infill(&self) -> PyResult<bool> {
        Ok(self.loaded_model()?.supports_infill())
    }

    #[getter]
    pub fn closed(&self) -> bool {
        option_state_is_closed(&self.inner)
    }

    #[pyo3(signature = (text, add_special = false))]
    pub fn tokenize(&self, text: &str, add_special: bool) -> PyResult<Vec<u32>> {
        Ok(self.loaded_model()?.tokenize(text, add_special))
    }

    pub fn decode(&self, token_ids: Vec<u32>) -> PyResult<String> {
        Ok(self.loaded_model()?.decode(&token_ids))
    }

    pub fn session(
        &self,
        py: Python<'_>,
        ctx_size: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Session> {
        let model = self.loaded_model()?;
        allow_threads_unsend(py, || {
            model.session(SdkSessionOptions {
                context_length: ctx_size,
                seed,
            })
        })
        .map(|session| Session {
            inner: Mutex::new(Some(session)),
        })
        .map_err(py_runtime_error)
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
        match self.model_info() {
            Ok(info) => format!(
                "Model(architecture={:?}, vocab_size={}, context_length={}, closed=False)",
                info.architecture, info.vocab_size, info.context_length
            ),
            Err(_) => "Model(closed=True)".to_string(),
        }
    }

    pub fn close(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| {
            tracing::warn!("Model mutex poisoned during close, attempting recovery");
            e.into_inner()
        });
        inner.take();
    }
}

impl Model {
    pub(crate) fn loaded_model(&self) -> PyResult<SdkModel> {
        self.inner
            .lock()
            .map_err(|_| py_runtime_error(anyhow::anyhow!("model lock poisoned")))?
            .clone()
            .ok_or_else(|| py_runtime_error(anyhow::anyhow!("model is closed")))
    }

    fn model_info(&self) -> PyResult<ModelInfo> {
        Ok(self.loaded_model()?.info())
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| {
            tracing::warn!("Model mutex poisoned during drop, attempting recovery");
            e.into_inner()
        });
        inner.take();
    }
}

fn init_tracing() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_env_filter("ax_engine_core=warn,ax_engine_py=info,ax_engine_sdk=info")
            .try_init();
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_option_state_is_closed_treats_poisoned_lock_as_closed() {
        let state = Arc::new(Mutex::new(Some(1u8)));
        let state_for_thread = Arc::clone(&state);
        let _ = thread::spawn(move || {
            let _guard = state_for_thread.lock().expect("lock");
            panic!("poison");
        })
        .join();

        assert!(option_state_is_closed(&state));
    }

    #[test]
    fn test_model_exposes_supports_infill_getter() {
        let getter: fn(&Model) -> PyResult<bool> = Model::supports_infill;
        let _ = getter;
    }
}
