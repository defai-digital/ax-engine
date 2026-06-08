use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ax_engine_sdk::{
    EngineSession, EngineSessionConfig, GenerateSampling, PreviewBackendRequest,
    PreviewSessionConfigRequest, preview_support_tier_from_label,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};

use crate::dicts::{generate_response_dict, request_report_dict, runtime_dict, step_report_dict};
use crate::embedding::{floats_to_pybytes, parse_pooling};
use crate::errors::{py_engine_state_error, to_py_runtime_error};
use crate::request::{
    build_generate_request, delegated_http_timeouts_from_secs, multimodal_inputs_from_py,
};
use crate::stream::GenerateStreamIterator;

#[pyclass(module = "ax_engine._ax_engine", unsendable)]
pub(crate) struct Session {
    model_id: String,
    inner: Arc<Mutex<SessionSlot>>,
}

#[derive(Debug)]
pub(crate) enum SessionSlot {
    Ready(Box<EngineSession>),
    Streaming,
    Closed,
}

#[pymethods]
impl Session {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (model_id="qwen3".to_string(), *, deterministic=true, max_batch_tokens=2048, cache_group_id=0, block_size_tokens=16, total_blocks=1024, mlx=false, support_tier="llama_cpp", llama_cli_path="llama-cli".to_string(), llama_model_path=None, llama_server_url=None, mlx_lm_server_url=None, mlx_model_artifacts_dir=None, delegated_http_connect_timeout_secs=30, delegated_http_read_timeout_secs=300, delegated_http_write_timeout_secs=300))]
    fn new(
        model_id: String,
        deterministic: bool,
        max_batch_tokens: u32,
        cache_group_id: u16,
        block_size_tokens: u32,
        total_blocks: u32,
        mlx: bool,
        support_tier: &str,
        llama_cli_path: String,
        llama_model_path: Option<String>,
        llama_server_url: Option<String>,
        mlx_lm_server_url: Option<String>,
        mlx_model_artifacts_dir: Option<String>,
        delegated_http_connect_timeout_secs: u64,
        delegated_http_read_timeout_secs: u64,
        delegated_http_write_timeout_secs: u64,
    ) -> PyResult<Self> {
        let support_tier = preview_support_tier_from_label(support_tier)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let delegated_http_timeouts = delegated_http_timeouts_from_secs(
            delegated_http_connect_timeout_secs,
            delegated_http_read_timeout_secs,
            delegated_http_write_timeout_secs,
        )?;
        let effective_mlx_model_artifacts_dir = if mlx {
            mlx_model_artifacts_dir
                .clone()
                .or_else(|| llama_model_path.clone())
                .map(PathBuf::from)
        } else {
            None
        };
        let backend_request = if mlx {
            PreviewBackendRequest::shipping_mlx()
        } else if support_tier == ax_engine_sdk::SupportTier::LlamaCpp {
            PreviewBackendRequest::shipping_default_llama_cpp(
                PathBuf::from(&llama_cli_path),
                llama_model_path.as_ref().map(PathBuf::from),
                llama_server_url.clone(),
            )
            .with_delegated_http_timeouts(delegated_http_timeouts)
        } else {
            PreviewBackendRequest {
                support_tier,
                llama_cli_path: PathBuf::from(llama_cli_path),
                llama_model_path: llama_model_path.map(PathBuf::from),
                llama_server_url,
                mlx_lm_server_url,
                delegated_http_timeouts,
                ..PreviewBackendRequest::default()
            }
        };
        let config = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            cache_group_id: ax_engine_sdk::CacheGroupId(cache_group_id),
            block_size_tokens,
            total_blocks,
            deterministic,
            max_batch_tokens,
            backend_request,
            mlx_runtime_artifacts_dir: None,
            mlx_model_artifacts_dir: effective_mlx_model_artifacts_dir,
            mlx_disable_ngram_acceleration: false,
            mlx_mtp_disable_ngram_stacking: false,
            mlx_kv_compression: ax_engine_sdk::KvCompressionConfig::disabled(),
            mlx_prefill_chunk: None,
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let inner = EngineSession::new(config).map_err(to_py_runtime_error)?;

        Ok(Self {
            model_id,
            inner: Arc::new(Mutex::new(SessionSlot::Ready(Box::new(inner)))),
        })
    }

    #[getter]
    fn model_id(&self) -> String {
        self.model_id.clone()
    }

    #[getter]
    fn closed(&self) -> bool {
        self.inner
            .lock()
            .map(|slot| matches!(*slot, SessionSlot::Closed))
            .unwrap_or(true)
    }

    fn close(&mut self) -> PyResult<()> {
        let mut slot = self.inner.lock().map_err(|_| {
            py_engine_state_error("session mutex poisoned; session state is unrecoverable")
        })?;
        *slot = SessionSlot::Closed;
        Ok(())
    }

    fn runtime<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        self.with_session_ref(|session| Ok(runtime_dict(py, &session.runtime_report())))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (input_tokens=None, *, input_text=None, multimodal_inputs=None, max_output_tokens, temperature=0.0, top_p=1.0, top_k=0, min_p=None, repetition_penalty=1.0, repetition_context_size=None, seed=0, deterministic=None, ignore_eos=false, stop_sequences=None, metadata=None))]
    fn generate<'py>(
        &mut self,
        py: Python<'py>,
        input_tokens: Option<Vec<u32>>,
        input_text: Option<String>,
        multimodal_inputs: Option<&Bound<'_, PyAny>>,
        max_output_tokens: u32,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        min_p: Option<f32>,
        repetition_penalty: f32,
        repetition_context_size: Option<u32>,
        seed: u64,
        deterministic: Option<bool>,
        ignore_eos: bool,
        stop_sequences: Option<Vec<String>>,
        metadata: Option<String>,
    ) -> PyResult<Py<PyDict>> {
        let model_id = self.model_id.clone();
        let multimodal_inputs = multimodal_inputs_from_py(multimodal_inputs)?;
        let request = build_generate_request(
            model_id,
            input_tokens.unwrap_or_default(),
            input_text,
            multimodal_inputs,
            max_output_tokens,
            GenerateSampling {
                temperature,
                top_p,
                top_k,
                min_p,
                repetition_penalty,
                repetition_context_size,
                seed,
                deterministic,
                ignore_eos,
            },
            stop_sequences.unwrap_or_default(),
            metadata,
        );
        let inner = Arc::clone(&self.inner);
        let response = py.allow_threads(move || {
            let mut slot = inner
                .lock()
                .map_err(|_| py_engine_state_error("session mutex poisoned"))?;
            match &mut *slot {
                SessionSlot::Ready(session) => {
                    session.generate(request).map_err(to_py_runtime_error)
                }
                SessionSlot::Streaming => {
                    Err(py_engine_state_error("session has an active stream"))
                }
                SessionSlot::Closed => Err(py_engine_state_error("session is closed")),
            }
        })?;

        Ok(generate_response_dict(py, &response))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (input_tokens=None, *, input_text=None, multimodal_inputs=None, max_output_tokens, temperature=0.0, top_p=1.0, top_k=0, min_p=None, repetition_penalty=1.0, repetition_context_size=None, seed=0, deterministic=None, ignore_eos=false, stop_sequences=None, metadata=None))]
    fn submit(
        &mut self,
        input_tokens: Option<Vec<u32>>,
        input_text: Option<String>,
        multimodal_inputs: Option<&Bound<'_, PyAny>>,
        max_output_tokens: u32,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        min_p: Option<f32>,
        repetition_penalty: f32,
        repetition_context_size: Option<u32>,
        seed: u64,
        deterministic: Option<bool>,
        ignore_eos: bool,
        stop_sequences: Option<Vec<String>>,
        metadata: Option<String>,
    ) -> PyResult<u64> {
        let model_id = self.model_id.clone();
        let multimodal_inputs = multimodal_inputs_from_py(multimodal_inputs)?;
        let request = build_generate_request(
            model_id,
            input_tokens.unwrap_or_default(),
            input_text,
            multimodal_inputs,
            max_output_tokens,
            GenerateSampling {
                temperature,
                top_p,
                top_k,
                min_p,
                repetition_penalty,
                repetition_context_size,
                seed,
                deterministic,
                ignore_eos,
            },
            stop_sequences.unwrap_or_default(),
            metadata,
        );
        let inner = Arc::clone(&self.inner);
        Python::with_gil(|py| {
            py.allow_threads(move || {
                let mut slot = inner
                    .lock()
                    .map_err(|_| py_engine_state_error("session mutex poisoned"))?;
                match &mut *slot {
                    SessionSlot::Ready(session) => session
                        .submit_generate(request)
                        .map_err(to_py_runtime_error),
                    SessionSlot::Streaming => {
                        Err(py_engine_state_error("session has an active stream"))
                    }
                    SessionSlot::Closed => Err(py_engine_state_error("session is closed")),
                }
            })
        })
    }

    fn step<'py>(&mut self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        let inner = Arc::clone(&self.inner);
        let report = py.allow_threads(move || {
            let mut slot = inner
                .lock()
                .map_err(|_| py_engine_state_error("session mutex poisoned"))?;
            match &mut *slot {
                SessionSlot::Ready(session) => session.step_report().map_err(to_py_runtime_error),
                SessionSlot::Streaming => {
                    Err(py_engine_state_error("session has an active stream"))
                }
                SessionSlot::Closed => Err(py_engine_state_error("session is closed")),
            }
        })?;

        Ok(step_report_dict(py, &report))
    }

    fn snapshot<'py>(&self, py: Python<'py>, request_id: u64) -> PyResult<Option<Py<PyDict>>> {
        self.with_session_ref(|session| {
            Ok(session
                .request_report(request_id)
                .map(|report| request_report_dict(py, &report)))
        })
    }

    fn cancel(&mut self, request_id: u64) -> PyResult<()> {
        let inner = Arc::clone(&self.inner);
        Python::with_gil(|py| {
            py.allow_threads(move || {
                let mut slot = inner
                    .lock()
                    .map_err(|_| py_engine_state_error("session mutex poisoned"))?;
                match &mut *slot {
                    SessionSlot::Ready(session) => session
                        .cancel_request(request_id)
                        .map_err(to_py_runtime_error),
                    SessionSlot::Streaming => {
                        Err(py_engine_state_error("session has an active stream"))
                    }
                    SessionSlot::Closed => Err(py_engine_state_error("session is closed")),
                }
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (input_tokens=None, *, input_text=None, multimodal_inputs=None, max_output_tokens, temperature=0.0, top_p=1.0, top_k=0, min_p=None, repetition_penalty=1.0, repetition_context_size=None, seed=0, deterministic=None, ignore_eos=false, stop_sequences=None, metadata=None))]
    fn stream_generate<'py>(
        &mut self,
        py: Python<'py>,
        input_tokens: Option<Vec<u32>>,
        input_text: Option<String>,
        multimodal_inputs: Option<&Bound<'_, PyAny>>,
        max_output_tokens: u32,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        min_p: Option<f32>,
        repetition_penalty: f32,
        repetition_context_size: Option<u32>,
        seed: u64,
        deterministic: Option<bool>,
        ignore_eos: bool,
        stop_sequences: Option<Vec<String>>,
        metadata: Option<String>,
    ) -> PyResult<Py<GenerateStreamIterator>> {
        let model_id = self.model_id.clone();
        let multimodal_inputs = multimodal_inputs_from_py(multimodal_inputs)?;
        let mut slot = self
            .inner
            .lock()
            .map_err(|_| py_engine_state_error("session mutex poisoned"))?;
        let mut session = match std::mem::replace(&mut *slot, SessionSlot::Streaming) {
            SessionSlot::Ready(session) => *session,
            SessionSlot::Streaming => {
                return Err(py_engine_state_error(
                    "session already has an active stream",
                ));
            }
            SessionSlot::Closed => {
                return Err(py_engine_state_error("session is closed"));
            }
        };
        drop(slot);

        let request = build_generate_request(
            model_id,
            input_tokens.unwrap_or_default(),
            input_text,
            multimodal_inputs,
            max_output_tokens,
            GenerateSampling {
                temperature,
                top_p,
                top_k,
                min_p,
                repetition_penalty,
                repetition_context_size,
                seed,
                deterministic,
                ignore_eos,
            },
            stop_sequences.unwrap_or_default(),
            metadata,
        );
        let state = match py.allow_threads(|| session.stream_generate_state(request)) {
            Ok(state) => state,
            Err(error) => {
                if let Ok(mut slot) = self.inner.lock() {
                    *slot = SessionSlot::Ready(Box::new(session));
                }
                return Err(to_py_runtime_error(error));
            }
        };

        Py::new(
            py,
            GenerateStreamIterator {
                owner: Arc::clone(&self.inner),
                session: Some(session),
                state: Some(state),
            },
        )
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Compute a dense embedding for the given token IDs.
    ///
    /// Calls directly into the MLX runner without any HTTP overhead, making
    /// this equivalent in call depth to mlx-lm and mlx-swift-lm for benchmarking.
    ///
    /// Parameters
    /// ----------
    /// token_ids : list[int]
    ///     Pre-tokenized input (caller is responsible for appending EOS for
    ///     models like Qwen3-Embedding that require it).
    /// pooling : str
    ///     Pooling strategy: "last" (default), "mean", or "cls".
    /// normalize : bool
    ///     L2-normalize the output vector (default True).
    ///
    /// Returns
    /// -------
    /// list[float]
    ///     Embedding vector as a Python list.
    #[pyo3(signature = (token_ids, *, pooling="last", normalize=true))]
    fn embed(
        &self,
        py: Python<'_>,
        token_ids: Vec<u32>,
        pooling: &str,
        normalize: bool,
    ) -> PyResult<Vec<f32>> {
        let pooling_mode = parse_pooling(pooling)?;
        let inner = Arc::clone(&self.inner);
        py.allow_threads(move || {
            let slot = inner
                .lock()
                .map_err(|_| py_engine_state_error("session mutex poisoned"))?;
            match &*slot {
                SessionSlot::Ready(session) => session
                    .embed(&token_ids, pooling_mode, normalize)
                    .map_err(to_py_runtime_error),
                SessionSlot::Streaming => {
                    Err(py_engine_state_error("session has an active stream"))
                }
                SessionSlot::Closed => Err(py_engine_state_error("session is closed")),
            }
        })
    }

    /// Same as :meth:`embed` but returns the embedding as raw little-endian
    /// f32 bytes instead of a ``list[float]``. Avoids the per-element
    /// ``PyFloat`` allocation that dominates the post-eval cost for large
    /// hidden sizes; the caller can wrap the result with ``numpy.frombuffer``
    /// (zero-copy) or ``array.array('f', buf)``.
    #[pyo3(signature = (token_ids, *, pooling="last", normalize=true))]
    fn embed_bytes<'py>(
        &self,
        py: Python<'py>,
        token_ids: Vec<u32>,
        pooling: &str,
        normalize: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let pooling_mode = parse_pooling(pooling)?;
        let inner = Arc::clone(&self.inner);
        let floats: Vec<f32> = py.allow_threads(move || {
            let slot = inner
                .lock()
                .map_err(|_| py_engine_state_error("session mutex poisoned"))?;
            match &*slot {
                SessionSlot::Ready(session) => session
                    .embed(&token_ids, pooling_mode, normalize)
                    .map_err(to_py_runtime_error),
                SessionSlot::Streaming => {
                    Err(py_engine_state_error("session has an active stream"))
                }
                SessionSlot::Closed => Err(py_engine_state_error("session is closed")),
            }
        })?;
        floats_to_pybytes(py, &floats)
    }

    #[pyo3(signature = (batch_token_ids, *, pooling="last", normalize=true))]
    fn embed_batch(
        &self,
        py: Python<'_>,
        batch_token_ids: Vec<Vec<u32>>,
        pooling: &str,
        normalize: bool,
    ) -> PyResult<Vec<Vec<f32>>> {
        let pooling_mode = parse_pooling(pooling)?;
        let inner = Arc::clone(&self.inner);
        py.allow_threads(move || {
            let slot = inner
                .lock()
                .map_err(|_| py_engine_state_error("session mutex poisoned"))?;
            match &*slot {
                SessionSlot::Ready(session) => session
                    .embed_batch(&batch_token_ids, pooling_mode, normalize)
                    .map_err(to_py_runtime_error),
                SessionSlot::Streaming => {
                    Err(py_engine_state_error("session has an active stream"))
                }
                SessionSlot::Closed => Err(py_engine_state_error("session is closed")),
            }
        })
    }

    /// Same as :meth:`embed_batch` but returns one ``bytes`` blob per
    /// sequence (raw little-endian f32). See :meth:`embed_bytes` for
    /// the rationale.
    #[pyo3(signature = (batch_token_ids, *, pooling="last", normalize=true))]
    fn embed_batch_bytes<'py>(
        &self,
        py: Python<'py>,
        batch_token_ids: Vec<Vec<u32>>,
        pooling: &str,
        normalize: bool,
    ) -> PyResult<Vec<Bound<'py, PyBytes>>> {
        let pooling_mode = parse_pooling(pooling)?;
        let inner = Arc::clone(&self.inner);
        let vecs: Vec<Vec<f32>> = py.allow_threads(move || {
            let slot = inner
                .lock()
                .map_err(|_| py_engine_state_error("session mutex poisoned"))?;
            match &*slot {
                SessionSlot::Ready(session) => session
                    .embed_batch(&batch_token_ids, pooling_mode, normalize)
                    .map_err(to_py_runtime_error),
                SessionSlot::Streaming => {
                    Err(py_engine_state_error("session has an active stream"))
                }
                SessionSlot::Closed => Err(py_engine_state_error("session is closed")),
            }
        })?;
        vecs.iter().map(|v| floats_to_pybytes(py, v)).collect()
    }

    /// Return the batch embedding as one contiguous ``bytes`` blob plus
    /// ``(batch_size, hidden_size)``. Callers can wrap the buffer with
    /// ``numpy.frombuffer(buf, dtype='f4').reshape(B, H)`` for a
    /// zero-copy `[B, H]` ndarray. Saves the `B - 1` PyBytes allocations
    /// that :meth:`embed_batch_bytes` does plus the matching number of
    /// inner `Vec<f32>` allocations on the Rust side.
    #[pyo3(signature = (batch_token_ids, *, pooling="last", normalize=true))]
    fn embed_batch_flat_bytes<'py>(
        &self,
        py: Python<'py>,
        batch_token_ids: Vec<Vec<u32>>,
        pooling: &str,
        normalize: bool,
    ) -> PyResult<(Bound<'py, PyBytes>, usize, usize)> {
        let pooling_mode = parse_pooling(pooling)?;
        let inner = Arc::clone(&self.inner);
        let matrix: ax_engine_sdk::EmbeddingMatrix = py.allow_threads(move || {
            let slot = inner
                .lock()
                .map_err(|_| py_engine_state_error("session mutex poisoned"))?;
            match &*slot {
                SessionSlot::Ready(session) => session
                    .embed_batch_flat(&batch_token_ids, pooling_mode, normalize)
                    .map_err(to_py_runtime_error),
                SessionSlot::Streaming => {
                    Err(py_engine_state_error("session has an active stream"))
                }
                SessionSlot::Closed => Err(py_engine_state_error("session is closed")),
            }
        })?;
        let blob = floats_to_pybytes(py, &matrix.data)?;
        Ok((blob, matrix.batch_size, matrix.hidden_size))
    }

    #[pyo3(signature = (_exc_type=None, _exc=None, _traceback=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) {
        let _ = self.close();
    }
}

impl Session {
    fn with_session_ref<T>(&self, f: impl FnOnce(&EngineSession) -> PyResult<T>) -> PyResult<T> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| py_engine_state_error("session mutex poisoned"))?;
        match &*inner {
            SessionSlot::Ready(session) => f(session.as_ref()),
            SessionSlot::Streaming => Err(py_engine_state_error("session has an active stream")),
            SessionSlot::Closed => Err(py_engine_state_error("session is closed")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dicts::test_support::{
        dict_string, dict_tokens, normalize_measurement_fields, py_dict_to_json,
        sdk_request_report_json, sdk_stream_event_json,
    };
    use ax_engine_sdk::{DelegatedHttpTimeouts, GenerateRequest};
    use pyo3::types::PyDictMethods;
    use serde_json::{Value, json};
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Once;
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn init_python() {
        static PYTHON_INIT: Once = Once::new();
        PYTHON_INIT.call_once(pyo3::prepare_freethreaded_python);
    }

    fn llama_cpp_session() -> Session {
        let script_path = fake_llama_cpp_script();
        let model_path = std::env::temp_dir().join("ax-engine-python-llama-cpp-model.gguf");
        fs::write(&model_path, "fake gguf").expect("fake model should be written");

        Session::new(
            "qwen3".to_string(),
            true,
            2048,
            0,
            16,
            1024,
            false,
            "llama_cpp",
            script_path.display().to_string(),
            Some(model_path.display().to_string()),
            None,
            None,
            None,
            DelegatedHttpTimeouts::default_connect_secs(),
            DelegatedHttpTimeouts::default_io_secs(),
            DelegatedHttpTimeouts::default_io_secs(),
        )
        .expect("llama.cpp session should build")
    }

    fn llama_cpp_server_session(server_url: String) -> Session {
        Session::new(
            "qwen3".to_string(),
            true,
            2048,
            0,
            16,
            1024,
            false,
            "llama_cpp",
            "llama-cli".to_string(),
            None,
            Some(server_url),
            None,
            None,
            DelegatedHttpTimeouts::default_connect_secs(),
            DelegatedHttpTimeouts::default_io_secs(),
            DelegatedHttpTimeouts::default_io_secs(),
        )
        .expect("llama.cpp session should build")
    }

    #[test]
    fn python_session_routes_default_local_model_to_llama_cpp() {
        init_python();
        let session = Session::new(
            "qwen3".to_string(),
            true,
            2048,
            0,
            16,
            1024,
            false,
            "llama_cpp",
            "llama-cli".to_string(),
            Some("/tmp/qwen3.5-mlx".to_string()),
            None,
            None,
            None,
            DelegatedHttpTimeouts::default_connect_secs(),
            DelegatedHttpTimeouts::default_io_secs(),
            DelegatedHttpTimeouts::default_io_secs(),
        )
        .expect("default llama.cpp session should build");

        Python::with_gil(|py| {
            let runtime = session.runtime(py).expect("runtime should serialize");
            let runtime = runtime.bind(py);
            assert_eq!(dict_string(runtime, "selected_backend"), "llama_cpp");
        });
    }

    #[test]
    fn python_session_routes_default_gguf_model_to_llama_cpp() {
        init_python();
        let session = Session::new(
            "qwen3".to_string(),
            true,
            2048,
            0,
            16,
            1024,
            false,
            "llama_cpp",
            "llama-cli".to_string(),
            Some("/tmp/qwen3.5-9b-q4.gguf".to_string()),
            None,
            None,
            None,
            DelegatedHttpTimeouts::default_connect_secs(),
            DelegatedHttpTimeouts::default_io_secs(),
            DelegatedHttpTimeouts::default_io_secs(),
        )
        .expect("GGUF session should build");

        Python::with_gil(|py| {
            let runtime = session.runtime(py).expect("runtime should serialize");
            let runtime = runtime.bind(py);
            assert_eq!(dict_string(runtime, "selected_backend"), "llama_cpp");
        });
    }

    #[test]
    fn python_session_routes_mlx_lm_delegated_to_runtime_report() {
        init_python();
        let session = Session::new(
            "qwen3".to_string(),
            true,
            2048,
            0,
            16,
            1024,
            false,
            "mlx_lm_delegated",
            "llama-cli".to_string(),
            None,
            None,
            Some("http://127.0.0.1:8090".to_string()),
            None,
            DelegatedHttpTimeouts::default_connect_secs(),
            DelegatedHttpTimeouts::default_io_secs(),
            DelegatedHttpTimeouts::default_io_secs(),
        )
        .expect("mlx-lm delegated session should build");

        Python::with_gil(|py| {
            let runtime = session.runtime(py).expect("runtime should serialize");
            let runtime = runtime.bind(py);
            assert_eq!(dict_string(runtime, "selected_backend"), "mlx_lm_delegated");
            assert_eq!(dict_string(runtime, "support_tier"), "mlx_lm_delegated");
        });
    }

    fn sdk_llama_cpp_server_session(server_url: String) -> EngineSession {
        let config = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            backend_request: PreviewBackendRequest::shipping_default_llama_cpp(
                PathBuf::from("llama-cli"),
                None,
                Some(server_url),
            ),
            ..PreviewSessionConfigRequest::default()
        })
        .expect("sdk llama.cpp preview resolution should succeed");
        EngineSession::new(config).expect("sdk llama.cpp session should build")
    }

    fn spawn_llama_cpp_completion_server(
        response_body: String,
        assert_request: impl FnOnce(Value) + Send + 'static,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("request should arrive");
            let request = read_http_request(&mut stream);
            let header_end = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|index| index + 4)
                .expect("request should include header terminator");
            let body =
                String::from_utf8(request[header_end..].to_vec()).expect("body should be utf8");
            let payload: Value = serde_json::from_str(&body).expect("request body should be json");
            assert_request(payload);

            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream
                .write_all(response.as_bytes())
                .expect("response should write");
        });

        (format!("http://{address}"), handle)
    }

    fn spawn_llama_cpp_completion_stream_server(
        expected_requests: usize,
        chunks: Vec<Value>,
        assert_request: impl Fn(Value) + Send + Sync + 'static,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");
        let handle = thread::spawn(move || {
            for _ in 0..expected_requests {
                let (mut stream, _) = listener.accept().expect("request should arrive");
                let request = read_http_request(&mut stream);
                let header_end = request
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|index| index + 4)
                    .expect("request should include header terminator");
                let body =
                    String::from_utf8(request[header_end..].to_vec()).expect("body should be utf8");
                let payload: Value =
                    serde_json::from_str(&body).expect("request body should be json");
                assert_request(payload);

                let mut body = String::new();
                for chunk in &chunks {
                    body.push_str("data: ");
                    body.push_str(
                        &serde_json::to_string(chunk).expect("chunk payload should serialize"),
                    );
                    body.push_str("\n\n");
                }
                body.push_str("data: [DONE]\n\n");

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("response should write");
            }
        });

        (format!("http://{address}"), handle)
    }

    fn read_http_request(stream: &mut std::net::TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        let mut header_end = None;
        let mut content_length = None;

        loop {
            let bytes_read = stream.read(&mut buffer).expect("request should read");
            assert!(
                bytes_read > 0,
                "client closed connection before request completed"
            );
            request.extend_from_slice(&buffer[..bytes_read]);

            if header_end.is_none() {
                header_end = request
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|index| index + 4);
                if let Some(end) = header_end {
                    let headers =
                        String::from_utf8(request[..end].to_vec()).expect("headers should be utf8");
                    content_length = Some(parse_content_length(&headers));
                }
            }

            if let (Some(end), Some(length)) = (header_end, content_length) {
                if request.len() >= end + length {
                    request.truncate(end + length);
                    return request;
                }
            }
        }
    }

    fn parse_content_length(headers: &str) -> usize {
        headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                if name.eq_ignore_ascii_case("content-length") {
                    Some(
                        value
                            .trim()
                            .parse::<usize>()
                            .expect("content-length should parse"),
                    )
                } else {
                    None
                }
            })
            .expect("content-length header should exist")
    }

    fn sample_sdk_request(input_tokens: &[u32], max_output_tokens: u32) -> GenerateRequest {
        GenerateRequest {
            model_id: "qwen3".to_string(),
            input_tokens: input_tokens.to_vec(),
            input_text: None,
            multimodal_inputs: Default::default(),
            max_output_tokens,
            sampling: GenerateSampling::default(),
            stop_sequences: Vec::new(),
            metadata: None,
        }
    }

    fn fake_llama_cpp_script() -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ax-engine-python-llama-cpp-{unique}.py"));
        let script = r#"#!/usr/bin/env python3
from __future__ import annotations

import sys

args = sys.argv[1:]
prompt = args[args.index("--prompt") + 1]
sys.stdout.write(f"python::{prompt}")
"#;

        fs::write(&path, script).expect("fake script should be written");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let mut permissions = fs::metadata(&path)
                .expect("script metadata should exist")
                .permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&path, permissions).expect("script should be executable");
        }
        path
    }

    #[test]
    fn python_session_llama_cpp_generate_returns_text_fields() {
        init_python();
        Python::with_gil(|py| {
            let mut session = llama_cpp_session();
            let response = session
                .generate(
                    py,
                    None,
                    Some("hello from python".to_string()),
                    None,
                    2,
                    0.0,
                    1.0,
                    0,
                    None,
                    1.0,
                    None,
                    0,
                    None,
                    false,
                    None,
                    None,
                )
                .expect("llama.cpp generate should succeed");
            let response = response.bind(py);

            assert_eq!(dict_string(response, "model_id"), "qwen3");
            assert_eq!(
                response
                    .get_item("prompt_text")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "hello from python"
            );
            assert_eq!(
                response
                    .get_item("output_text")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "python::hello from python"
            );
            let runtime = response
                .get_item("runtime")
                .unwrap()
                .unwrap()
                .downcast_into::<PyDict>()
                .unwrap();
            assert_eq!(dict_string(&runtime, "selected_backend"), "llama_cpp");
            assert_eq!(dict_string(&runtime, "support_tier"), "llama_cpp");
        });
    }

    #[test]
    fn python_session_llama_cpp_server_generate_supports_token_prompts() {
        init_python();
        let (server_url, server_handle) = spawn_llama_cpp_completion_server(
            serde_json::json!({
                "content": "python server tokens",
                "tokens": [8, 9],
                "stop": true,
                "stop_type": "limit"
            })
            .to_string(),
            |payload| {
                assert_eq!(payload.get("prompt"), Some(&json!([1, 2, 3])));
                assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
            },
        );

        Python::with_gil(|py| {
            let mut session = llama_cpp_server_session(server_url);
            let response = session
                .generate(
                    py,
                    Some(vec![1, 2, 3]),
                    None,
                    None,
                    2,
                    0.0,
                    1.0,
                    0,
                    None,
                    1.0,
                    None,
                    0,
                    None,
                    false,
                    None,
                    None,
                )
                .expect("llama.cpp server generate should succeed");
            let response = response.bind(py);

            assert_eq!(dict_tokens(response, "prompt_tokens"), vec![1, 2, 3]);
            assert_eq!(dict_tokens(response, "output_tokens"), vec![8, 9]);
            assert_eq!(
                response
                    .get_item("output_text")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "python server tokens"
            );
        });

        server_handle
            .join()
            .expect("llama.cpp server thread should finish");
    }

    #[test]
    fn python_session_llama_cpp_server_stream_generate_matches_sdk_payloads() {
        init_python();
        let (server_url, server_handle) = spawn_llama_cpp_completion_stream_server(
            2,
            vec![
                serde_json::json!({
                    "content": "hello",
                    "tokens": [4],
                    "stop": false
                }),
                serde_json::json!({
                    "content": " world",
                    "tokens": [5],
                    "stop": true,
                    "stop_type": "limit"
                }),
            ],
            |payload| {
                assert_eq!(payload.get("prompt"), Some(&json!([1, 2, 3])));
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
            },
        );

        Python::with_gil(|py| {
            let mut session = llama_cpp_server_session(server_url.clone());
            let stream = session
                .stream_generate(
                    py,
                    Some(vec![1, 2, 3]),
                    None,
                    None,
                    2,
                    0.0,
                    1.0,
                    0,
                    None,
                    1.0,
                    None,
                    0,
                    None,
                    false,
                    None,
                    None,
                )
                .expect("llama.cpp stream should succeed");
            let mut actual = Vec::new();
            {
                let mut stream = stream.bind(py).borrow_mut();
                while let Some(event) = stream
                    .next_event_dict(py)
                    .expect("python stream event should succeed")
                {
                    actual.push(py_dict_to_json(event.bind(py)));
                }
            }

            let mut expected_session = sdk_llama_cpp_server_session(server_url);
            let mut expected = expected_session
                .stream_generate(sample_sdk_request(&[1, 2, 3], 2))
                .expect("sdk llama.cpp stream should start")
                .collect::<Result<Vec<_>, _>>()
                .expect("sdk llama.cpp stream should complete")
                .into_iter()
                .map(|event| sdk_stream_event_json(&event))
                .collect::<Vec<_>>();
            for payload in &mut actual {
                normalize_measurement_fields(payload);
            }
            for payload in &mut expected {
                normalize_measurement_fields(payload);
            }

            assert_eq!(actual, expected);
        });

        server_handle
            .join()
            .expect("llama.cpp server thread should finish");
    }

    #[test]
    fn python_session_llama_cpp_stepwise_request_control_matches_sdk() {
        init_python();
        let (server_url, server_handle) = spawn_llama_cpp_completion_stream_server(
            2,
            vec![
                serde_json::json!({
                    "content": "hello",
                    "tokens": [4],
                    "stop": false
                }),
                serde_json::json!({
                    "content": " world",
                    "tokens": [5],
                    "stop": true,
                    "stop_type": "limit"
                }),
            ],
            |payload| {
                assert_eq!(payload.get("prompt"), Some(&json!([1, 2, 3])));
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
            },
        );

        Python::with_gil(|py| {
            let mut session = llama_cpp_server_session(server_url.clone());
            let mut expected_session = sdk_llama_cpp_server_session(server_url);
            let expected_request_id = expected_session
                .submit_generate(sample_sdk_request(&[1, 2, 3], 2))
                .expect("sdk llama.cpp submit should succeed");
            let request_id = session
                .submit(
                    Some(vec![1, 2, 3]),
                    None,
                    None,
                    2,
                    0.0,
                    1.0,
                    0,
                    None,
                    1.0,
                    None,
                    0,
                    None,
                    false,
                    None,
                    None,
                )
                .expect("llama.cpp submit should succeed");
            assert_eq!(request_id, expected_request_id);

            for _ in 0..4 {
                let expected_snapshot = expected_session
                    .request_report(request_id)
                    .expect("sdk llama.cpp snapshot should exist");
                let snapshot = session
                    .snapshot(py, request_id)
                    .expect("snapshot should succeed")
                    .expect("llama.cpp request should still exist");
                let snapshot = snapshot.bind(py);
                let state = dict_string(snapshot, "state");
                assert_eq!(
                    py_dict_to_json(snapshot),
                    sdk_request_report_json(expected_snapshot.clone())
                );
                if state == "finished" {
                    break;
                }

                let expected_step = expected_session
                    .step_report()
                    .expect("sdk llama.cpp step should succeed");
                let step = session.step(py).expect("llama.cpp step should succeed");
                let step = step.bind(py);
                let mut actual_step_json = py_dict_to_json(step);
                let mut expected_step_json = serde_json::to_value(expected_step.clone())
                    .expect("sdk llama.cpp step should serialize");
                normalize_measurement_fields(&mut actual_step_json);
                normalize_measurement_fields(&mut expected_step_json);
                assert_eq!(actual_step_json, expected_step_json);
            }

            let terminal = session
                .snapshot(py, request_id)
                .expect("snapshot should succeed")
                .expect("terminal llama.cpp request should exist");
            let terminal = terminal.bind(py);
            let expected_terminal = expected_session
                .request_report(request_id)
                .expect("sdk llama.cpp terminal snapshot should exist");
            assert_eq!(
                py_dict_to_json(terminal),
                sdk_request_report_json(expected_terminal)
            );
        });

        server_handle
            .join()
            .expect("llama.cpp server thread should finish");
    }

    #[test]
    fn python_session_llama_cpp_stepwise_multiple_requests_match_sdk() {
        init_python();
        let expected_prompts = vec![json!([1, 2, 3]), json!([7, 8, 9])];
        let expected_prompts_for_request = expected_prompts.clone();
        let (server_url, server_handle) = spawn_llama_cpp_completion_stream_server(
            4,
            vec![
                serde_json::json!({
                    "content": "hello",
                    "tokens": [4],
                    "stop": false
                }),
                serde_json::json!({
                    "content": " world",
                    "tokens": [5],
                    "stop": true,
                    "stop_type": "limit"
                }),
            ],
            move |payload| {
                let prompt = payload.get("prompt").expect("prompt should be present");
                assert!(
                    expected_prompts_for_request
                        .iter()
                        .any(|candidate| prompt == candidate)
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
            },
        );

        Python::with_gil(|py| {
            let mut session = llama_cpp_server_session(server_url.clone());
            let mut expected_session = sdk_llama_cpp_server_session(server_url);
            let first_request_id = session
                .submit(
                    Some(vec![1, 2, 3]),
                    None,
                    None,
                    2,
                    0.0,
                    1.0,
                    0,
                    None,
                    1.0,
                    None,
                    0,
                    None,
                    false,
                    None,
                    None,
                )
                .expect("first llama.cpp submit should succeed");
            let second_request_id = session
                .submit(
                    Some(vec![7, 8, 9]),
                    None,
                    None,
                    2,
                    0.0,
                    1.0,
                    0,
                    None,
                    1.0,
                    None,
                    0,
                    None,
                    false,
                    None,
                    None,
                )
                .expect("second llama.cpp submit should succeed");
            expected_session
                .submit_generate_with_request_id(
                    first_request_id,
                    sample_sdk_request(&[1, 2, 3], 2),
                )
                .expect("first sdk llama.cpp submit should succeed");
            expected_session
                .submit_generate_with_request_id(
                    second_request_id,
                    sample_sdk_request(&[7, 8, 9], 2),
                )
                .expect("second sdk llama.cpp submit should succeed");

            for _ in 0..2 {
                let expected_step = expected_session
                    .step_report()
                    .expect("sdk llama.cpp aggregated step should succeed");
                let step = session
                    .step(py)
                    .expect("llama.cpp aggregated step should succeed");
                let step = step.bind(py);
                let mut actual_step_json = py_dict_to_json(step);
                let mut expected_step_json = serde_json::to_value(expected_step.clone())
                    .expect("sdk llama.cpp step should serialize");
                normalize_measurement_fields(&mut actual_step_json);
                normalize_measurement_fields(&mut expected_step_json);
                assert_eq!(actual_step_json, expected_step_json);

                for request_id in [first_request_id, second_request_id] {
                    let snapshot = session
                        .snapshot(py, request_id)
                        .expect("snapshot should succeed")
                        .expect("llama.cpp request should still exist");
                    let snapshot = snapshot.bind(py);
                    let expected_snapshot = expected_session
                        .request_report(request_id)
                        .expect("sdk llama.cpp snapshot should exist");
                    assert_eq!(
                        py_dict_to_json(snapshot),
                        sdk_request_report_json(expected_snapshot)
                    );
                }
            }
        });

        server_handle
            .join()
            .expect("llama.cpp server thread should finish");
    }

    #[test]
    fn python_session_llama_cpp_cancel_matches_sdk_cancelled_state() {
        init_python();
        let (server_url, server_handle) = spawn_llama_cpp_completion_stream_server(
            2,
            vec![serde_json::json!({
                "content": "hello",
                "tokens": [4],
                "stop": false
            })],
            |payload| {
                assert_eq!(payload.get("prompt"), Some(&json!([7, 8, 9])));
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            },
        );

        Python::with_gil(|py| {
            let mut session = llama_cpp_server_session(server_url.clone());
            let mut expected_session = sdk_llama_cpp_server_session(server_url);
            let request_id = session
                .submit(
                    Some(vec![7, 8, 9]),
                    None,
                    None,
                    2,
                    0.0,
                    1.0,
                    0,
                    None,
                    1.0,
                    None,
                    0,
                    None,
                    false,
                    None,
                    None,
                )
                .expect("llama.cpp submit should succeed");
            expected_session
                .submit_generate_with_request_id(request_id, sample_sdk_request(&[7, 8, 9], 2))
                .expect("sdk llama.cpp submit should succeed");

            session
                .cancel(request_id)
                .expect("llama.cpp cancel should succeed");
            expected_session
                .cancel_request(request_id)
                .expect("sdk llama.cpp cancel should succeed");

            let snapshot = session
                .snapshot(py, request_id)
                .expect("snapshot should succeed")
                .expect("cancelled llama.cpp request should exist");
            let snapshot = snapshot.bind(py);
            let expected_snapshot = expected_session
                .request_report(request_id)
                .expect("sdk llama.cpp cancelled snapshot should exist");

            assert_eq!(
                py_dict_to_json(snapshot),
                sdk_request_report_json(expected_snapshot)
            );
        });

        server_handle
            .join()
            .expect("llama.cpp server thread should finish");
    }
}
