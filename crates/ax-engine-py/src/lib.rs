use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ax_engine_sdk::{
    is_gguf_path, is_initial_native_mode_model_id, native_mode_model_requirement_message,
    preview_support_tier_from_label, CapabilityReport, CompatibilityBackendError,
    CompatibilityBackendKind, EngineSession, EngineSessionConfig, EngineSessionError,
    EngineStepReport, GenerateRequest, GenerateResponse, GenerateRouteReport, GenerateSampling,
    GenerateStreamEvent as SdkGenerateStreamEvent, GenerateStreamState, HostReport,
    MetalDispatchKernelStepReport, MetalDispatchNumericStepReport, MetalDispatchStepReport,
    MetalDispatchValidationStepReport, MetalToolchainReport, PreviewBackendRequest,
    PreviewSessionConfigRequest, RuntimeReport, SessionRequestReport, ToolStatusReport,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

#[pyclass(module = "ax_engine._ax_engine", unsendable)]
struct Session {
    model_id: String,
    inner: Arc<Mutex<SessionSlot>>,
}

#[derive(Debug)]
enum SessionSlot {
    Ready(Box<EngineSession>),
    Streaming,
    Closed,
}

#[pyclass(module = "ax_engine._ax_engine", unsendable)]
struct GenerateStreamIterator {
    owner: Arc<Mutex<SessionSlot>>,
    session: Option<EngineSession>,
    state: Option<GenerateStreamState>,
}

#[pymethods]
impl Session {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (model_id="qwen3_dense".to_string(), *, deterministic=true, max_batch_tokens=2048, cache_group_id=0, block_size_tokens=16, total_blocks=1024, native_mode=false, mlx=false, support_tier="compatibility", compat_backend="llama_cpp", compat_cli_path="llama-cli".to_string(), compat_model_path=None, compat_server_url=None, llama_fallback_cli_path="llama-cli".to_string(), llama_fallback_model_path=None, llama_fallback_server_url=None, native_runtime_artifacts_dir=None, native_model_artifacts_dir=None))]
    fn new(
        model_id: String,
        deterministic: bool,
        max_batch_tokens: u32,
        cache_group_id: u16,
        block_size_tokens: u32,
        total_blocks: u32,
        native_mode: bool,
        mlx: bool,
        support_tier: &str,
        compat_backend: &str,
        compat_cli_path: String,
        compat_model_path: Option<String>,
        compat_server_url: Option<String>,
        llama_fallback_cli_path: String,
        llama_fallback_model_path: Option<String>,
        llama_fallback_server_url: Option<String>,
        native_runtime_artifacts_dir: Option<String>,
        native_model_artifacts_dir: Option<String>,
    ) -> PyResult<Self> {
        if native_mode && mlx {
            return Err(PyValueError::new_err(
                "native_mode and mlx cannot both be enabled",
            ));
        }

        let support_tier = preview_support_tier_from_label(support_tier)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let mut compat_backend =
            compatibility_backend_kind_from_label(compat_backend).map_err(PyValueError::new_err)?;
        if !native_mode
            && support_tier == ax_engine_sdk::SupportTier::Compatibility
            && compat_model_path.is_some()
            && compat_server_url.is_none()
        {
            compat_backend = if compat_model_path
                .as_deref()
                .map(Path::new)
                .is_some_and(is_gguf_path)
            {
                CompatibilityBackendKind::LlamaCpp
            } else {
                CompatibilityBackendKind::Mlx
            };
        }
        let backend_request = if native_mode {
            if !is_initial_native_mode_model_id(&model_id) {
                return Err(PyValueError::new_err(
                    native_mode_model_requirement_message(&model_id),
                ));
            }
            let (llama_fallback_cli_path, llama_fallback_model_path, llama_fallback_server_url) =
                if llama_fallback_model_path.is_some() || llama_fallback_server_url.is_some() {
                    (
                        PathBuf::from(llama_fallback_cli_path),
                        llama_fallback_model_path.map(PathBuf::from),
                        llama_fallback_server_url,
                    )
                } else {
                    let compat_model_path = compat_model_path
                        .as_ref()
                        .map(Path::new)
                        .filter(|path| is_gguf_path(path))
                        .map(PathBuf::from);
                    (
                        PathBuf::from(&compat_cli_path),
                        compat_model_path,
                        compat_server_url.clone(),
                    )
                };
            PreviewBackendRequest::shipping_native_with_llama_fallback(
                llama_fallback_cli_path,
                llama_fallback_model_path,
                llama_fallback_server_url,
            )
        } else if mlx {
            PreviewBackendRequest::shipping_mlx_with_llama_fallback(
                PathBuf::from(&compat_cli_path),
                compat_model_path.as_ref().map(PathBuf::from),
                compat_server_url.clone(),
                PathBuf::from(llama_fallback_cli_path),
                llama_fallback_model_path.map(PathBuf::from),
                llama_fallback_server_url,
            )
        } else if support_tier == ax_engine_sdk::SupportTier::Compatibility
            && compat_backend == CompatibilityBackendKind::LlamaCpp
        {
            PreviewBackendRequest::shipping_default_llama_cpp(
                PathBuf::from(&compat_cli_path),
                compat_model_path.as_ref().map(PathBuf::from),
                compat_server_url.clone(),
            )
        } else {
            PreviewBackendRequest {
                support_tier,
                compat_backend,
                compat_cli_path: PathBuf::from(compat_cli_path),
                compat_model_path: compat_model_path.map(PathBuf::from),
                compat_server_url,
                ..PreviewBackendRequest::default()
            }
        };
        let config = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            cache_group_id: ax_engine_sdk::CacheGroupId(cache_group_id),
            block_size_tokens,
            total_blocks,
            deterministic,
            allow_deterministic_native_fallback: false,
            max_batch_tokens,
            backend_request,
            native_runtime_artifacts_dir: native_runtime_artifacts_dir.map(PathBuf::from),
            native_model_artifacts_dir: native_model_artifacts_dir.map(PathBuf::from),
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
            .unwrap_or(false)
    }

    fn close(&mut self) -> PyResult<()> {
        let mut slot = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("session mutex poisoned; session state is unrecoverable"))?;
        *slot = SessionSlot::Closed;
        Ok(())
    }

    fn runtime<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        self.with_session_ref(|session| Ok(runtime_dict(py, &session.runtime_report())))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (input_tokens=None, *, input_text=None, max_output_tokens, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0, seed=0, deterministic=None, metadata=None))]
    fn generate<'py>(
        &mut self,
        py: Python<'py>,
        input_tokens: Option<Vec<u32>>,
        input_text: Option<String>,
        max_output_tokens: u32,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        repetition_penalty: f32,
        seed: u64,
        deterministic: Option<bool>,
        metadata: Option<String>,
    ) -> PyResult<Py<PyDict>> {
        let model_id = self.model_id.clone();
        let request = build_generate_request(
            model_id,
            input_tokens.unwrap_or_default(),
            input_text,
            max_output_tokens,
            GenerateSampling {
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                seed,
                deterministic,
            },
            metadata,
        );
        let inner = Arc::clone(&self.inner);
        let response = py.allow_threads(move || {
            let mut slot = inner
                .lock()
                .map_err(|_| PyRuntimeError::new_err("session mutex poisoned"))?;
            match &mut *slot {
                SessionSlot::Ready(session) => {
                    session.generate(request).map_err(to_py_runtime_error)
                }
                SessionSlot::Streaming => {
                    Err(PyRuntimeError::new_err("session has an active stream"))
                }
                SessionSlot::Closed => Err(PyRuntimeError::new_err("session is closed")),
            }
        })?;

        Ok(generate_response_dict(py, &response))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (input_tokens=None, *, input_text=None, max_output_tokens, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0, seed=0, deterministic=None, metadata=None))]
    fn submit(
        &mut self,
        input_tokens: Option<Vec<u32>>,
        input_text: Option<String>,
        max_output_tokens: u32,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        repetition_penalty: f32,
        seed: u64,
        deterministic: Option<bool>,
        metadata: Option<String>,
    ) -> PyResult<u64> {
        let model_id = self.model_id.clone();
        let request = build_generate_request(
            model_id,
            input_tokens.unwrap_or_default(),
            input_text,
            max_output_tokens,
            GenerateSampling {
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                seed,
                deterministic,
            },
            metadata,
        );
        let inner = Arc::clone(&self.inner);
        Python::with_gil(|py| {
            py.allow_threads(move || {
                let mut slot = inner
                    .lock()
                    .map_err(|_| PyRuntimeError::new_err("session mutex poisoned"))?;
                match &mut *slot {
                    SessionSlot::Ready(session) => session
                        .submit_generate(request)
                        .map_err(to_py_runtime_error),
                    SessionSlot::Streaming => {
                        Err(PyRuntimeError::new_err("session has an active stream"))
                    }
                    SessionSlot::Closed => Err(PyRuntimeError::new_err("session is closed")),
                }
            })
        })
    }

    fn step<'py>(&mut self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        let inner = Arc::clone(&self.inner);
        let report = py.allow_threads(move || {
            let mut slot = inner
                .lock()
                .map_err(|_| PyRuntimeError::new_err("session mutex poisoned"))?;
            match &mut *slot {
                SessionSlot::Ready(session) => session.step_report().map_err(to_py_runtime_error),
                SessionSlot::Streaming => {
                    Err(PyRuntimeError::new_err("session has an active stream"))
                }
                SessionSlot::Closed => Err(PyRuntimeError::new_err("session is closed")),
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
                    .map_err(|_| PyRuntimeError::new_err("session mutex poisoned"))?;
                match &mut *slot {
                    SessionSlot::Ready(session) => session
                        .cancel_request(request_id)
                        .map_err(to_py_runtime_error),
                    SessionSlot::Streaming => {
                        Err(PyRuntimeError::new_err("session has an active stream"))
                    }
                    SessionSlot::Closed => Err(PyRuntimeError::new_err("session is closed")),
                }
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (input_tokens=None, *, input_text=None, max_output_tokens, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0, seed=0, deterministic=None, metadata=None))]
    fn stream_generate<'py>(
        &mut self,
        py: Python<'py>,
        input_tokens: Option<Vec<u32>>,
        input_text: Option<String>,
        max_output_tokens: u32,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        repetition_penalty: f32,
        seed: u64,
        deterministic: Option<bool>,
        metadata: Option<String>,
    ) -> PyResult<Py<GenerateStreamIterator>> {
        let model_id = self.model_id.clone();
        let mut slot = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("session mutex poisoned"))?;
        let mut session = match std::mem::replace(&mut *slot, SessionSlot::Streaming) {
            SessionSlot::Ready(session) => *session,
            SessionSlot::Streaming => {
                return Err(PyRuntimeError::new_err(
                    "session already has an active stream",
                ));
            }
            SessionSlot::Closed => {
                return Err(PyRuntimeError::new_err("session is closed"));
            }
        };
        drop(slot);

        let request = build_generate_request(
            model_id,
            input_tokens.unwrap_or_default(),
            input_text,
            max_output_tokens,
            GenerateSampling {
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                seed,
                deterministic,
            },
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

    #[pyo3(signature = (_exc_type=None, _exc=None, _traceback=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) {
        self.close();
    }
}

fn compatibility_backend_kind_from_label(value: &str) -> Result<CompatibilityBackendKind, String> {
    match value {
        "llama_cpp" => Ok(CompatibilityBackendKind::LlamaCpp),
        "vllm" => Ok(CompatibilityBackendKind::Vllm),
        "mistral_rs" => Ok(CompatibilityBackendKind::MistralRs),
        "mlx" => Ok(CompatibilityBackendKind::Mlx),
        other => Err(format!(
            "unsupported compat_backend {other}; expected llama_cpp, vllm, mistral_rs, or mlx"
        )),
    }
}

impl Session {
    fn with_session_ref<T>(&self, f: impl FnOnce(&EngineSession) -> PyResult<T>) -> PyResult<T> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("session mutex poisoned"))?;
        match &*inner {
            SessionSlot::Ready(session) => f(session.as_ref()),
            SessionSlot::Streaming => Err(PyRuntimeError::new_err("session has an active stream")),
            SessionSlot::Closed => Err(PyRuntimeError::new_err("session is closed")),
        }
    }
}

#[pymethods]
impl GenerateStreamIterator {
    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Py<PyDict>>> {
        self.next_event_dict(py)
    }
}

impl GenerateStreamIterator {
    fn next_event_dict<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Py<PyDict>>> {
        let Some(session) = self.session.as_mut() else {
            return Ok(None);
        };
        let Some(state) = self.state.as_mut() else {
            self.restore_session();
            return Ok(None);
        };

        match py.allow_threads(|| session.next_stream_event(state)) {
            Ok(Some(event)) => {
                let is_terminal = matches!(event, SdkGenerateStreamEvent::Response(_));
                let payload = stream_event_dict(py, &event);
                if is_terminal {
                    self.state = None;
                    self.restore_session();
                }
                Ok(Some(payload))
            }
            Ok(None) => {
                self.state = None;
                self.restore_session();
                Ok(None)
            }
            Err(error) => {
                self.state = None;
                self.restore_session();
                Err(to_py_runtime_error(error))
            }
        }
    }

    fn restore_session(&mut self) {
        let Some(session) = self.session.take() else {
            return;
        };

        let Ok(mut owner) = self.owner.lock() else {
            return;
        };
        match &*owner {
            SessionSlot::Closed => {}
            SessionSlot::Streaming => {
                *owner = SessionSlot::Ready(Box::new(session));
            }
            SessionSlot::Ready(_) => {}
        }
    }
}

impl Drop for GenerateStreamIterator {
    fn drop(&mut self) {
        self.restore_session();
    }
}

fn build_generate_request(
    model_id: String,
    input_tokens: Vec<u32>,
    input_text: Option<String>,
    max_output_tokens: u32,
    sampling: GenerateSampling,
    metadata: Option<String>,
) -> GenerateRequest {
    GenerateRequest {
        model_id,
        input_tokens,
        input_text,
        max_output_tokens,
        sampling,
        metadata,
    }
}

fn runtime_dict<'py>(py: Python<'py>, runtime: &RuntimeReport) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("selected_backend", enum_label(py, runtime.selected_backend))
        .expect("selected_backend should serialize");
    dict.set_item("support_tier", enum_label(py, runtime.support_tier))
        .expect("support_tier should serialize");
    dict.set_item(
        "resolution_policy",
        enum_label(py, runtime.resolution_policy),
    )
    .expect("resolution_policy should serialize");
    dict.set_item("capabilities", capability_dict(py, &runtime.capabilities))
        .expect("capabilities should serialize");
    if let Some(fallback_reason) = runtime.fallback_reason.as_deref() {
        dict.set_item("fallback_reason", fallback_reason)
            .expect("fallback_reason should serialize");
    }
    dict.set_item("host", host_dict(py, &runtime.host))
        .expect("host should serialize");
    dict.set_item(
        "metal_toolchain",
        metal_toolchain_dict(py, &runtime.metal_toolchain),
    )
    .expect("metal_toolchain should serialize");
    if let Some(native_runtime) = runtime.native_runtime.as_ref() {
        dict.set_item("native_runtime", native_runtime_dict(py, native_runtime))
            .expect("native_runtime should serialize");
    }
    if let Some(native_model) = runtime.native_model.as_ref() {
        dict.set_item("native_model", native_model_dict(py, native_model))
            .expect("native_model should serialize");
    }
    dict.unbind()
}

fn native_runtime_dict<'py>(
    py: Python<'py>,
    native_runtime: &ax_engine_sdk::NativeRuntimeReport,
) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("runner", enum_label(py, native_runtime.runner))
        .expect("native_runtime.runner should serialize");
    if let Some(source) = native_runtime.artifacts_source {
        dict.set_item("artifacts_source", enum_label(py, source))
            .expect("native_runtime.artifacts_source should serialize");
    }
    dict.unbind()
}

fn native_model_dict<'py>(
    py: Python<'py>,
    native_model: &ax_engine_sdk::NativeModelReport,
) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item(
        "artifacts_source",
        enum_label(py, native_model.artifacts_source),
    )
    .expect("native_model.artifacts_source should serialize");
    dict.set_item("model_family", &native_model.model_family)
        .expect("native_model.model_family should serialize");
    dict.set_item("tensor_format", enum_label(py, native_model.tensor_format))
        .expect("native_model.tensor_format should serialize");
    dict.set_item("layer_count", native_model.layer_count)
        .expect("native_model.layer_count should serialize");
    dict.set_item("tensor_count", native_model.tensor_count)
        .expect("native_model.tensor_count should serialize");
    dict.set_item("tie_word_embeddings", native_model.tie_word_embeddings)
        .expect("native_model.tie_word_embeddings should serialize");
    dict.set_item("bindings_prepared", native_model.bindings_prepared)
        .expect("native_model.bindings_prepared should serialize");
    dict.set_item("buffers_bound", native_model.buffers_bound)
        .expect("native_model.buffers_bound should serialize");
    dict.set_item("buffer_count", native_model.buffer_count)
        .expect("native_model.buffer_count should serialize");
    dict.set_item("buffer_bytes", native_model.buffer_bytes)
        .expect("native_model.buffer_bytes should serialize");
    dict.unbind()
}

fn capability_dict<'py>(py: Python<'py>, capabilities: &CapabilityReport) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("text_generation", capabilities.text_generation)
        .expect("capabilities should serialize");
    dict.set_item("token_streaming", capabilities.token_streaming)
        .expect("capabilities should serialize");
    dict.set_item("deterministic_mode", capabilities.deterministic_mode)
        .expect("capabilities should serialize");
    dict.set_item("prefix_reuse", capabilities.prefix_reuse)
        .expect("capabilities should serialize");
    dict.set_item(
        "long_context_validation",
        enum_label(py, capabilities.long_context_validation),
    )
    .expect("capabilities should serialize");
    dict.set_item(
        "benchmark_metrics",
        enum_label(py, capabilities.benchmark_metrics),
    )
    .expect("capabilities should serialize");
    dict.unbind()
}

fn host_dict<'py>(py: Python<'py>, host: &HostReport) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("os", &host.os)
        .expect("host.os should serialize");
    dict.set_item("arch", &host.arch)
        .expect("host.arch should serialize");
    if let Some(detected_soc) = host.detected_soc.as_deref() {
        dict.set_item("detected_soc", detected_soc)
            .expect("host.detected_soc should serialize");
    }
    dict.set_item("supported_native_runtime", host.supported_native_runtime)
        .expect("host.supported_native_runtime should serialize");
    dict.set_item(
        "unsupported_host_override_active",
        host.unsupported_host_override_active,
    )
    .expect("host.unsupported_host_override_active should serialize");
    dict.unbind()
}

fn metal_toolchain_dict<'py>(py: Python<'py>, toolchain: &MetalToolchainReport) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("fully_available", toolchain.fully_available)
        .expect("toolchain.fully_available should serialize");
    dict.set_item("metal", tool_status_dict(py, &toolchain.metal))
        .expect("toolchain.metal should serialize");
    dict.set_item("metallib", tool_status_dict(py, &toolchain.metallib))
        .expect("toolchain.metallib should serialize");
    dict.set_item("metal_ar", tool_status_dict(py, &toolchain.metal_ar))
        .expect("toolchain.metal_ar should serialize");
    dict.unbind()
}

fn tool_status_dict<'py>(py: Python<'py>, tool: &ToolStatusReport) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("available", tool.available)
        .expect("tool.available should serialize");
    if let Some(version) = tool.version.as_deref() {
        dict.set_item("version", version)
            .expect("tool.version should serialize");
    }
    dict.unbind()
}

fn route_dict<'py>(py: Python<'py>, route: &GenerateRouteReport) -> Py<PyDict> {
    let dict = PyDict::new(py);
    if let Some(value) = route.execution_plan.as_deref() {
        dict.set_item("execution_plan", value)
            .expect("execution_plan should serialize");
    }
    if let Some(value) = route.attention_route.as_deref() {
        dict.set_item("attention_route", value)
            .expect("attention_route should serialize");
    }
    if let Some(value) = route.kv_mode.as_deref() {
        dict.set_item("kv_mode", value)
            .expect("kv_mode should serialize");
    }
    if let Some(value) = route.prefix_cache_path.as_deref() {
        dict.set_item("prefix_cache_path", value)
            .expect("prefix_cache_path should serialize");
    }
    if let Some(value) = route.barrier_mode.as_deref() {
        dict.set_item("barrier_mode", value)
            .expect("barrier_mode should serialize");
    }
    if !route.crossover_decisions.is_empty() {
        let crossover = PyDict::new(py);
        for (key, value) in &route.crossover_decisions {
            crossover
                .set_item(key, value)
                .expect("crossover decision should serialize");
        }
        dict.set_item("crossover_decisions", crossover)
            .expect("crossover decisions should serialize");
    }
    dict.unbind()
}

fn request_report_dict<'py>(py: Python<'py>, report: &SessionRequestReport) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("request_id", report.request_id)
        .expect("request_id should serialize");
    dict.set_item("model_id", report.model_id.as_str())
        .expect("model_id should serialize");
    dict.set_item("state", enum_label(py, report.state))
        .expect("state should serialize");
    dict.set_item("prompt_tokens", report.prompt_tokens.clone())
        .expect("prompt_tokens should serialize");
    dict.set_item("processed_prompt_tokens", report.processed_prompt_tokens)
        .expect("processed_prompt_tokens should serialize");
    dict.set_item("output_tokens", report.output_tokens.clone())
        .expect("output_tokens should serialize");
    if !report.output_token_logprobs.is_empty() {
        dict.set_item(
            "output_token_logprobs",
            report.output_token_logprobs.clone(),
        )
        .expect("output_token_logprobs should serialize");
    }
    dict.set_item("prompt_len", report.prompt_len)
        .expect("prompt_len should serialize");
    dict.set_item("output_len", report.output_len)
        .expect("output_len should serialize");
    dict.set_item("max_output_tokens", report.max_output_tokens)
        .expect("max_output_tokens should serialize");
    dict.set_item("cancel_requested", report.cancel_requested)
        .expect("cancel_requested should serialize");
    if let Some(finish_reason) = report.finish_reason {
        dict.set_item("finish_reason", enum_label(py, finish_reason))
            .expect("finish_reason should serialize");
    }
    if let Some(terminal_stop_reason) = report.terminal_stop_reason {
        dict.set_item("terminal_stop_reason", enum_label(py, terminal_stop_reason))
            .expect("terminal_stop_reason should serialize");
    }
    if let Some(execution_plan_ref) = report.execution_plan_ref.as_deref() {
        dict.set_item("execution_plan_ref", execution_plan_ref)
            .expect("execution_plan_ref should serialize");
    }
    dict.set_item("route", route_dict(py, &report.route))
        .expect("route should serialize");
    dict.unbind()
}

fn step_report_dict<'py>(py: Python<'py>, report: &EngineStepReport) -> Py<PyDict> {
    let dict = PyDict::new(py);
    if let Some(step_id) = report.step_id {
        dict.set_item("step_id", step_id)
            .expect("step_id should serialize");
    }
    dict.set_item("scheduled_requests", report.scheduled_requests)
        .expect("scheduled_requests should serialize");
    dict.set_item("scheduled_tokens", report.scheduled_tokens)
        .expect("scheduled_tokens should serialize");
    dict.set_item("ttft_events", report.ttft_events)
        .expect("ttft_events should serialize");
    dict.set_item("prefix_hits", report.prefix_hits)
        .expect("prefix_hits should serialize");
    dict.set_item("kv_usage_blocks", report.kv_usage_blocks)
        .expect("kv_usage_blocks should serialize");
    dict.set_item("evictions", report.evictions)
        .expect("evictions should serialize");
    dict.set_item("cpu_time_us", report.cpu_time_us)
        .expect("cpu_time_us should serialize");
    dict.set_item("runner_time_us", report.runner_time_us)
        .expect("runner_time_us should serialize");
    if let Some(route) = report.route.as_ref() {
        dict.set_item("route", route_dict(py, route))
            .expect("route should serialize");
    }
    if let Some(metal_dispatch) = report.metal_dispatch.as_ref() {
        dict.set_item("metal_dispatch", metal_dispatch_dict(py, metal_dispatch))
            .expect("metal_dispatch should serialize");
    }
    dict.unbind()
}

fn metal_dispatch_dict<'py>(py: Python<'py>, report: &MetalDispatchStepReport) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("command_queue_label", report.command_queue_label.as_str())
        .expect("command_queue_label should serialize");
    dict.set_item("command_buffer_label", report.command_buffer_label.as_str())
        .expect("command_buffer_label should serialize");
    dict.set_item(
        "command_buffer_status",
        enum_label(py, report.command_buffer_status),
    )
    .expect("command_buffer_status should serialize");
    dict.set_item("runtime_device_name", report.runtime_device_name.as_str())
        .expect("runtime_device_name should serialize");
    dict.set_item(
        "runtime_required_pipeline_count",
        report.runtime_required_pipeline_count,
    )
    .expect("runtime_required_pipeline_count should serialize");
    dict.set_item(
        "runtime_max_thread_execution_width",
        report.runtime_max_thread_execution_width,
    )
    .expect("runtime_max_thread_execution_width should serialize");
    dict.set_item(
        "runtime_model_conditioned_inputs",
        report.runtime_model_conditioned_inputs,
    )
    .expect("runtime_model_conditioned_inputs should serialize");
    dict.set_item(
        "runtime_real_model_tensor_inputs",
        report.runtime_real_model_tensor_inputs,
    )
    .expect("runtime_real_model_tensor_inputs should serialize");
    dict.set_item(
        "runtime_complete_model_forward_supported",
        report.runtime_complete_model_forward_supported,
    )
    .expect("runtime_complete_model_forward_supported should serialize");
    dict.set_item(
        "runtime_model_bindings_prepared",
        report.runtime_model_bindings_prepared,
    )
    .expect("runtime_model_bindings_prepared should serialize");
    dict.set_item(
        "runtime_model_buffers_bound",
        report.runtime_model_buffers_bound,
    )
    .expect("runtime_model_buffers_bound should serialize");
    dict.set_item(
        "runtime_model_buffer_count",
        report.runtime_model_buffer_count,
    )
    .expect("runtime_model_buffer_count should serialize");
    dict.set_item(
        "runtime_model_buffer_bytes",
        report.runtime_model_buffer_bytes,
    )
    .expect("runtime_model_buffer_bytes should serialize");
    if let Some(model_family) = report.runtime_model_family.as_deref() {
        dict.set_item("runtime_model_family", model_family)
            .expect("runtime_model_family should serialize");
    }
    dict.set_item(
        "execution_direct_decode_token_count",
        report.execution_direct_decode_token_count,
    )
    .expect("execution_direct_decode_token_count should serialize");
    dict.set_item(
        "execution_direct_decode_checksum_lo",
        report.execution_direct_decode_checksum_lo,
    )
    .expect("execution_direct_decode_checksum_lo should serialize");
    dict.set_item(
        "execution_logits_output_count",
        report.execution_logits_output_count,
    )
    .expect("execution_logits_output_count should serialize");
    dict.set_item(
        "execution_remaining_logits_handle_count",
        report.execution_remaining_logits_handle_count,
    )
    .expect("execution_remaining_logits_handle_count should serialize");
    dict.set_item(
        "execution_model_bound_ffn_decode",
        report.execution_model_bound_ffn_decode,
    )
    .expect("execution_model_bound_ffn_decode should serialize");
    dict.set_item(
        "execution_real_model_forward_completed",
        report.execution_real_model_forward_completed,
    )
    .expect("execution_real_model_forward_completed should serialize");
    dict.set_item(
        "execution_prefix_native_dispatch_count",
        report.execution_prefix_native_dispatch_count,
    )
    .expect("execution_prefix_native_dispatch_count should serialize");
    dict.set_item(
        "execution_prefix_cpu_reference_dispatch_count",
        report.execution_prefix_cpu_reference_dispatch_count,
    )
    .expect("execution_prefix_cpu_reference_dispatch_count should serialize");
    dict.set_item(
        "execution_qkv_projection_token_count",
        report.execution_qkv_projection_token_count,
    )
    .expect("execution_qkv_projection_token_count should serialize");
    dict.set_item(
        "execution_layer_continuation_token_count",
        report.execution_layer_continuation_token_count,
    )
    .expect("execution_layer_continuation_token_count should serialize");
    dict.set_item(
        "execution_logits_projection_token_count",
        report.execution_logits_projection_token_count,
    )
    .expect("execution_logits_projection_token_count should serialize");
    dict.set_item(
        "execution_logits_vocab_scan_row_count",
        report.execution_logits_vocab_scan_row_count,
    )
    .expect("execution_logits_vocab_scan_row_count should serialize");
    dict.set_item(
        "binary_archive_state",
        enum_label(py, report.binary_archive_state),
    )
    .expect("binary_archive_state should serialize");
    dict.set_item(
        "binary_archive_attached_pipeline_count",
        report.binary_archive_attached_pipeline_count,
    )
    .expect("binary_archive_attached_pipeline_count should serialize");
    dict.set_item(
        "binary_archive_serialized",
        report.binary_archive_serialized,
    )
    .expect("binary_archive_serialized should serialize");
    dict.set_item("arena_token_capacity", report.arena_token_capacity)
        .expect("arena_token_capacity should serialize");
    dict.set_item("arena_slot_capacity", report.arena_slot_capacity)
        .expect("arena_slot_capacity should serialize");
    dict.set_item(
        "arena_attention_ref_capacity",
        report.arena_attention_ref_capacity,
    )
    .expect("arena_attention_ref_capacity should serialize");
    dict.set_item(
        "arena_gather_ref_capacity",
        report.arena_gather_ref_capacity,
    )
    .expect("arena_gather_ref_capacity should serialize");
    dict.set_item(
        "arena_gather_output_capacity",
        report.arena_gather_output_capacity,
    )
    .expect("arena_gather_output_capacity should serialize");
    dict.set_item("arena_copy_pair_capacity", report.arena_copy_pair_capacity)
        .expect("arena_copy_pair_capacity should serialize");
    dict.set_item("arena_sequence_capacity", report.arena_sequence_capacity)
        .expect("arena_sequence_capacity should serialize");
    dict.set_item("arena_reused_existing", report.arena_reused_existing)
        .expect("arena_reused_existing should serialize");
    dict.set_item("arena_grew_existing", report.arena_grew_existing)
        .expect("arena_grew_existing should serialize");

    let kernels = report
        .kernels
        .iter()
        .map(|kernel| metal_dispatch_kernel_dict(py, kernel))
        .collect::<Vec<_>>();
    dict.set_item("kernels", kernels)
        .expect("kernels should serialize");
    dict.set_item("numeric", metal_dispatch_numeric_dict(py, &report.numeric))
        .expect("numeric should serialize");
    dict.unbind()
}

fn metal_dispatch_kernel_dict<'py>(
    py: Python<'py>,
    kernel: &MetalDispatchKernelStepReport,
) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("function_name", kernel.function_name.as_str())
        .expect("function_name should serialize");
    dict.set_item("element_count", kernel.element_count)
        .expect("element_count should serialize");
    dict.set_item("threads_per_grid_width", kernel.threads_per_grid_width)
        .expect("threads_per_grid_width should serialize");
    dict.set_item(
        "threads_per_threadgroup_width",
        kernel.threads_per_threadgroup_width,
    )
    .expect("threads_per_threadgroup_width should serialize");
    dict.unbind()
}

fn metal_dispatch_numeric_dict<'py>(
    py: Python<'py>,
    numeric: &MetalDispatchNumericStepReport,
) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("key_cache_checksum", numeric.key_cache_checksum)
        .expect("key_cache_checksum should serialize");
    dict.set_item(
        "attention_output_checksum",
        numeric.attention_output_checksum,
    )
    .expect("attention_output_checksum should serialize");
    dict.set_item("gather_output_checksum", numeric.gather_output_checksum)
        .expect("gather_output_checksum should serialize");
    dict.set_item("copy_output_checksum", numeric.copy_output_checksum)
        .expect("copy_output_checksum should serialize");
    if let Some(validation) = numeric.validation.as_ref() {
        dict.set_item("validation", metal_dispatch_validation_dict(py, validation))
            .expect("validation should serialize");
    }
    dict.unbind()
}

fn metal_dispatch_validation_dict<'py>(
    py: Python<'py>,
    validation: &MetalDispatchValidationStepReport,
) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item(
        "expected_key_cache_checksum",
        validation.expected_key_cache_checksum,
    )
    .expect("expected_key_cache_checksum should serialize");
    dict.set_item(
        "expected_attention_output_checksum",
        validation.expected_attention_output_checksum,
    )
    .expect("expected_attention_output_checksum should serialize");
    dict.set_item(
        "expected_gather_output_checksum",
        validation.expected_gather_output_checksum,
    )
    .expect("expected_gather_output_checksum should serialize");
    dict.set_item(
        "expected_copy_output_checksum",
        validation.expected_copy_output_checksum,
    )
    .expect("expected_copy_output_checksum should serialize");
    dict.set_item(
        "attention_max_abs_diff_microunits",
        validation.attention_max_abs_diff_microunits,
    )
    .expect("attention_max_abs_diff_microunits should serialize");
    dict.unbind()
}

fn generate_response_dict<'py>(py: Python<'py>, response: &GenerateResponse) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("request_id", response.request_id)
        .expect("request_id should serialize");
    dict.set_item("model_id", response.model_id.as_str())
        .expect("model_id should serialize");
    dict.set_item("prompt_tokens", response.prompt_tokens.clone())
        .expect("prompt_tokens should serialize");
    if let Some(prompt_text) = response.prompt_text.as_deref() {
        dict.set_item("prompt_text", prompt_text)
            .expect("prompt_text should serialize");
    }
    dict.set_item("output_tokens", response.output_tokens.clone())
        .expect("output_tokens should serialize");
    if !response.output_token_logprobs.is_empty() {
        dict.set_item(
            "output_token_logprobs",
            response.output_token_logprobs.clone(),
        )
        .expect("output_token_logprobs should serialize");
    }
    if let Some(output_text) = response.output_text.as_deref() {
        dict.set_item("output_text", output_text)
            .expect("output_text should serialize");
    }
    dict.set_item("status", enum_label(py, response.status))
        .expect("status should serialize");
    if let Some(finish_reason) = response.finish_reason {
        dict.set_item("finish_reason", enum_label(py, finish_reason))
            .expect("finish_reason should serialize");
    }
    dict.set_item("step_count", response.step_count)
        .expect("step_count should serialize");
    if let Some(ttft_step) = response.ttft_step {
        dict.set_item("ttft_step", ttft_step)
            .expect("ttft_step should serialize");
    }
    dict.set_item("route", route_dict(py, &response.route))
        .expect("route should serialize");
    dict.set_item("runtime", runtime_dict(py, &response.runtime))
        .expect("runtime should serialize");
    dict.unbind()
}

fn stream_event_dict<'py>(py: Python<'py>, event: &SdkGenerateStreamEvent) -> Py<PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("event", event.event_name())
        .expect("event name should serialize");

    match event {
        SdkGenerateStreamEvent::Request(payload) => {
            dict.set_item("request", request_report_dict(py, &payload.request))
                .expect("request payload should serialize");
            dict.set_item("runtime", runtime_dict(py, &payload.runtime))
                .expect("runtime payload should serialize");
        }
        SdkGenerateStreamEvent::Step(payload) => {
            dict.set_item("request", request_report_dict(py, &payload.request))
                .expect("request payload should serialize");
            dict.set_item("step", step_report_dict(py, &payload.step))
                .expect("step payload should serialize");
            dict.set_item("delta_tokens", payload.delta_tokens.clone())
                .expect("delta tokens should serialize");
            if !payload.delta_token_logprobs.is_empty() {
                dict.set_item("delta_token_logprobs", payload.delta_token_logprobs.clone())
                    .expect("delta token logprobs should serialize");
            }
            if let Some(delta_text) = payload.delta_text.as_deref() {
                dict.set_item("delta_text", delta_text)
                    .expect("delta text should serialize");
            }
        }
        SdkGenerateStreamEvent::Response(payload) => {
            dict.set_item("response", generate_response_dict(py, &payload.response))
                .expect("response payload should serialize");
        }
    }

    dict.unbind()
}

fn enum_label<T>(py: Python<'_>, value: T) -> PyObject
where
    T: serde::Serialize,
{
    let string = serde_json::to_value(value)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .unwrap_or_else(|| "unknown".to_string());
    PyString::new(py, &string).into_any().unbind()
}

fn to_py_runtime_error(error: EngineSessionError) -> PyErr {
    match error {
        EngineSessionError::EmptyInputTokens
        | EngineSessionError::InvalidMaxOutputTokens
        | EngineSessionError::NativeBackendRequiresTokenizedInput
        | EngineSessionError::InvalidMaxBatchTokens
        | EngineSessionError::InvalidRequestId
        | EngineSessionError::UnsupportedSupportTier
        | EngineSessionError::CompatibilityBackendDoesNotSupportLifecycle { .. }
        | EngineSessionError::StatelessStreamRequiresCompatibilityBackend { .. }
        | EngineSessionError::Compatibility(CompatibilityBackendError::StreamingNotSupported {
            ..
        })
        | EngineSessionError::RequestDidNotTerminate { .. }
        | EngineSessionError::MissingRequestSnapshot { .. } => {
            PyValueError::new_err(error.to_string())
        }
        EngineSessionError::Compatibility(CompatibilityBackendError::MissingInputText {
            ..
        })
        | EngineSessionError::Compatibility(CompatibilityBackendError::MissingPromptInput {
            ..
        })
        | EngineSessionError::Compatibility(CompatibilityBackendError::UnsupportedTokenPrompt {
            ..
        })
        | EngineSessionError::Compatibility(
            CompatibilityBackendError::UnsupportedSamplingOption { .. },
        )
        | EngineSessionError::Compatibility(CompatibilityBackendError::AmbiguousPromptInput {
            ..
        })
        | EngineSessionError::Compatibility(CompatibilityBackendError::BackendConfigMismatch {
            ..
        }) => PyValueError::new_err(error.to_string()),
        EngineSessionError::BackendContract(_)
        | EngineSessionError::MissingCompatibilityBackendConfig { .. }
        | EngineSessionError::CompatibilityBackendConfigMismatch { .. }
        | EngineSessionError::CompatibilityFallbackMustUseLlamaCpp { .. }
        | EngineSessionError::CompatibilityFallbackRequiresNonStrictPolicy
        | EngineSessionError::CompatibilityStreamEndedBeforeStop { .. }
        | EngineSessionError::CompatibilityFallbackFailed { .. }
        | EngineSessionError::NativeStartupFallbackFailed { .. }
        | EngineSessionError::NativeRuntimeArtifactsRequired
        | EngineSessionError::Compatibility(CompatibilityBackendError::CommandLaunch { .. })
        | EngineSessionError::Compatibility(CompatibilityBackendError::CommandFailed { .. })
        | EngineSessionError::Compatibility(CompatibilityBackendError::CommandTimedOut { .. })
        | EngineSessionError::Compatibility(CompatibilityBackendError::NonUtf8Output { .. })
        | EngineSessionError::Compatibility(CompatibilityBackendError::SerializeRequestJson {
            ..
        })
        | EngineSessionError::Compatibility(CompatibilityBackendError::HttpRequest { .. })
        | EngineSessionError::Compatibility(CompatibilityBackendError::HttpStatus { .. })
        | EngineSessionError::Compatibility(CompatibilityBackendError::HttpResponseRead {
            ..
        })
        | EngineSessionError::Compatibility(CompatibilityBackendError::InvalidResponseJson {
            ..
        })
        | EngineSessionError::Compatibility(CompatibilityBackendError::EmptyChoicesInResponse {
            ..
        })
        | EngineSessionError::UnsupportedHostHardware { .. }
        | EngineSessionError::RequestReportInvariantViolation { .. }
        | EngineSessionError::StreamEndedWithoutResponse { .. }
        | EngineSessionError::NativeModelAutoConvert { .. }
        | EngineSessionError::Core(_)
        | EngineSessionError::MetalRuntime(_)
        | EngineSessionError::NativeModelGgufExportLaunch { .. }
        | EngineSessionError::NativeModelGgufExportFailed { .. } => {
            PyRuntimeError::new_err(error.to_string())
        }
    }
}

#[pymodule]
fn _ax_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Session>()?;
    m.add_class::<GenerateStreamIterator>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_engine_sdk::INITIAL_NATIVE_MODE_MODEL_ID;
    use pyo3::types::{PyDictMethods, PyList};
    use serde_json::{json, Map, Value};
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

    fn native_placeholder_sdk_config() -> EngineSessionConfig {
        EngineSessionConfig {
            allow_deterministic_native_fallback: true,
            ..EngineSessionConfig::default()
        }
    }

    fn preview_session() -> Session {
        let inner =
            EngineSession::new(native_placeholder_sdk_config()).expect("session should build");
        Session {
            model_id: "qwen3_dense".to_string(),
            inner: Arc::new(Mutex::new(SessionSlot::Ready(Box::new(inner)))),
        }
    }

    fn compatibility_session() -> Session {
        let script_path = fake_compat_script();
        let model_path = std::env::temp_dir().join("ax-engine-python-compat-model.gguf");
        fs::write(&model_path, "fake gguf").expect("fake model should be written");

        Session::new(
            "qwen3_dense".to_string(),
            true,
            2048,
            0,
            16,
            1024,
            false,
            false,
            "compatibility",
            "llama_cpp",
            script_path.display().to_string(),
            Some(model_path.display().to_string()),
            None,
            "llama-cli".to_string(),
            None,
            None,
            None,
            None,
        )
        .expect("compatibility session should build")
    }

    fn compatibility_server_session(server_url: String) -> Session {
        Session::new(
            "qwen3_dense".to_string(),
            true,
            2048,
            0,
            16,
            1024,
            false,
            false,
            "compatibility",
            "llama_cpp",
            "llama-cli".to_string(),
            None,
            Some(server_url),
            "llama-cli".to_string(),
            None,
            None,
            None,
            None,
        )
        .expect("compatibility session should build")
    }

    #[test]
    fn python_session_rejects_native_mode_for_non_initial_model() {
        init_python();
        let error = match Session::new(
            "qwen3_dense".to_string(),
            true,
            2048,
            0,
            16,
            1024,
            true,
            false,
            "compatibility",
            "llama_cpp",
            "llama-cli".to_string(),
            None,
            None,
            "llama-cli".to_string(),
            None,
            None,
            None,
            None,
        ) {
            Ok(_) => panic!("native mode should be allowlisted"),
            Err(error) => error,
        };

        assert!(error.to_string().contains(INITIAL_NATIVE_MODE_MODEL_ID));
    }

    #[test]
    fn python_session_routes_default_local_mlx_model_to_mlx_cli() {
        init_python();
        let session = Session::new(
            "qwen3_dense".to_string(),
            true,
            2048,
            0,
            16,
            1024,
            false,
            false,
            "compatibility",
            "llama_cpp",
            "llama-cli".to_string(),
            Some("/tmp/qwen3.5-mlx".to_string()),
            None,
            "llama-cli".to_string(),
            None,
            None,
            None,
            None,
        )
        .expect("default MLX session should build");

        Python::with_gil(|py| {
            let runtime = session.runtime(py).expect("runtime should serialize");
            let runtime = runtime.bind(py);
            assert_eq!(dict_string(runtime, "selected_backend"), "mlx");
        });
    }

    #[test]
    fn python_session_routes_default_gguf_model_to_llama_cpp() {
        init_python();
        let session = Session::new(
            "qwen3_dense".to_string(),
            true,
            2048,
            0,
            16,
            1024,
            false,
            false,
            "compatibility",
            "mlx",
            "llama-cli".to_string(),
            Some("/tmp/qwen3.5-9b-q4.gguf".to_string()),
            None,
            "llama-cli".to_string(),
            None,
            None,
            None,
            None,
        )
        .expect("GGUF session should build");

        Python::with_gil(|py| {
            let runtime = session.runtime(py).expect("runtime should serialize");
            let runtime = runtime.bind(py);
            assert_eq!(dict_string(runtime, "selected_backend"), "llama_cpp");
        });
    }

    fn sdk_compatibility_server_session(server_url: String) -> EngineSession {
        let config = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            backend_request: PreviewBackendRequest::shipping_default_llama_cpp(
                PathBuf::from("llama-cli"),
                None,
                Some(server_url),
            ),
            ..PreviewSessionConfigRequest::default()
        })
        .expect("sdk compatibility preview resolution should succeed");
        EngineSession::new(config).expect("sdk compatibility session should build")
    }

    fn spawn_compat_completion_server(
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

    fn spawn_compat_completion_stream_server(
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

    fn sdk_session() -> EngineSession {
        EngineSession::new(native_placeholder_sdk_config()).expect("sdk session should build")
    }

    fn sample_sdk_request(input_tokens: &[u32], max_output_tokens: u32) -> GenerateRequest {
        GenerateRequest {
            model_id: "qwen3_dense".to_string(),
            input_tokens: input_tokens.to_vec(),
            input_text: None,
            max_output_tokens,
            sampling: GenerateSampling::default(),
            metadata: None,
        }
    }

    fn fake_compat_script() -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ax-engine-python-compat-{unique}.py"));
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

    fn dict_string(dict: &Bound<'_, PyDict>, key: &str) -> String {
        dict.get_item(key)
            .unwrap()
            .unwrap()
            .extract::<String>()
            .unwrap()
    }

    fn dict_u64(dict: &Bound<'_, PyDict>, key: &str) -> u64 {
        dict.get_item(key)
            .unwrap()
            .unwrap()
            .extract::<u64>()
            .unwrap()
    }

    fn dict_u32(dict: &Bound<'_, PyDict>, key: &str) -> u32 {
        dict.get_item(key)
            .unwrap()
            .unwrap()
            .extract::<u32>()
            .unwrap()
    }

    fn dict_tokens(dict: &Bound<'_, PyDict>, key: &str) -> Vec<u32> {
        dict.get_item(key)
            .unwrap()
            .unwrap()
            .extract::<Vec<u32>>()
            .unwrap()
    }

    fn py_any_to_json(value: &Bound<'_, PyAny>) -> Value {
        if value.is_none() {
            return Value::Null;
        }
        if let Ok(dict) = value.downcast::<PyDict>() {
            return py_dict_to_json(dict);
        }
        if let Ok(list) = value.downcast::<PyList>() {
            return Value::Array(list.iter().map(|item| py_any_to_json(&item)).collect());
        }
        if let Ok(string) = value.extract::<String>() {
            return Value::String(string);
        }
        if let Ok(boolean) = value.extract::<bool>() {
            return Value::Bool(boolean);
        }
        if let Ok(number) = value.extract::<i64>() {
            return Value::Number(number.into());
        }
        if let Ok(number) = value.extract::<u64>() {
            return Value::Number(number.into());
        }
        if let Ok(number) = value.extract::<f64>() {
            return Value::Number(
                serde_json::Number::from_f64(number)
                    .expect("finite python float should convert to json number"),
            );
        }

        panic!("unsupported python value in test json conversion");
    }

    fn py_dict_to_json(dict: &Bound<'_, PyDict>) -> Value {
        let mut object = Map::new();
        for (key, value) in dict.iter() {
            object.insert(
                key.extract::<String>()
                    .expect("python dict keys should be strings"),
                py_any_to_json(&value),
            );
        }
        Value::Object(object)
    }

    fn normalize_measurement_fields(value: &mut Value) {
        match value {
            Value::Object(map) => {
                map.remove("cpu_time_us");
                map.remove("runner_time_us");
                for value in map.values_mut() {
                    normalize_measurement_fields(value);
                }
            }
            Value::Array(values) => {
                for value in values {
                    normalize_measurement_fields(value);
                }
            }
            _ => {}
        }
    }

    fn sdk_stream_event_json(event: &SdkGenerateStreamEvent) -> Value {
        match event {
            SdkGenerateStreamEvent::Request(payload) => json!({
                "event": "request",
                "runtime": payload.runtime,
                "request": payload.request,
            }),
            SdkGenerateStreamEvent::Step(payload) => {
                let mut json = json!({
                    "event": "step",
                    "request": payload.request,
                    "step": payload.step,
                    "delta_tokens": payload.delta_tokens,
                });
                if !payload.delta_token_logprobs.is_empty() {
                    json.as_object_mut()
                        .expect("step event json should be object")
                        .insert(
                            "delta_token_logprobs".to_string(),
                            json!(payload.delta_token_logprobs),
                        );
                }
                if let Some(delta_text) = payload.delta_text.as_deref() {
                    json.as_object_mut()
                        .expect("step event json should be object")
                        .insert("delta_text".to_string(), json!(delta_text));
                }
                json
            }
            SdkGenerateStreamEvent::Response(payload) => json!({
                "event": "response",
                "response": payload.response,
            }),
        }
    }

    #[test]
    fn python_session_runtime_reports_preview_metadata() {
        init_python();
        Python::with_gil(|py| {
            let session = preview_session();
            let runtime = session.runtime(py).expect("runtime should serialize");
            let runtime = runtime.bind(py);
            let expected = sdk_session().runtime_report();

            assert_eq!(dict_string(runtime, "selected_backend"), "ax_native");
            assert_eq!(dict_string(runtime, "support_tier"), "native_preview");
            assert_eq!(dict_string(runtime, "resolution_policy"), "strict_native");
            assert_eq!(
                py_dict_to_json(runtime),
                serde_json::to_value(&expected).expect("sdk runtime should serialize")
            );
            assert_eq!(
                dict_string(runtime, "selected_backend"),
                serde_json::to_value(expected.selected_backend)
                    .unwrap()
                    .as_str()
                    .unwrap()
            );
            assert_eq!(
                dict_string(runtime, "support_tier"),
                serde_json::to_value(expected.support_tier)
                    .unwrap()
                    .as_str()
                    .unwrap()
            );
            assert_eq!(
                dict_string(runtime, "resolution_policy"),
                serde_json::to_value(expected.resolution_policy)
                    .unwrap()
                    .as_str()
                    .unwrap()
            );
        });
    }

    #[test]
    fn python_session_generate_returns_terminal_response() {
        init_python();
        Python::with_gil(|py| {
            let mut session = preview_session();
            let response = session
                .generate(
                    py,
                    Some(vec![1, 2, 3]),
                    None,
                    2,
                    0.0,
                    1.0,
                    0,
                    1.0,
                    0,
                    None,
                    None,
                )
                .expect("generate should succeed");
            let response = response.bind(py);
            let expected = sdk_session()
                .generate(sample_sdk_request(&[1, 2, 3], 2))
                .expect("sdk generate should succeed");
            let actual_json = py_dict_to_json(response);
            let expected_json =
                serde_json::to_value(expected.clone()).expect("sdk response should serialize");

            assert_eq!(dict_u64(response, "request_id"), expected.request_id);
            assert_eq!(dict_string(response, "model_id"), expected.model_id);
            assert_eq!(
                dict_tokens(response, "prompt_tokens"),
                expected.prompt_tokens
            );
            assert_eq!(
                dict_tokens(response, "output_tokens"),
                expected.output_tokens
            );
            assert_eq!(
                dict_string(response, "status"),
                serde_json::to_value(expected.status)
                    .unwrap()
                    .as_str()
                    .unwrap()
            );
            assert_eq!(dict_u64(response, "step_count"), expected.step_count);
            assert_eq!(actual_json, expected_json);
        });
    }

    #[test]
    fn python_session_stepwise_request_control_reports_progress() {
        init_python();
        Python::with_gil(|py| {
            let mut session = preview_session();
            let mut expected_session = sdk_session();
            let expected_request_id = expected_session
                .submit_generate(sample_sdk_request(&[1, 2, 3], 2))
                .expect("sdk submit should succeed");
            let request_id = session
                .submit(
                    Some(vec![1, 2, 3]),
                    None,
                    2,
                    0.0,
                    1.0,
                    0,
                    1.0,
                    0,
                    None,
                    None,
                )
                .expect("submit should succeed");
            assert_eq!(request_id, expected_request_id);
            let initial = session
                .snapshot(py, request_id)
                .expect("snapshot should succeed")
                .expect("submitted request should exist");
            let initial = initial.bind(py);
            let expected_initial = expected_session
                .request_report(request_id)
                .expect("sdk snapshot should exist");

            assert_eq!(dict_string(initial, "state"), "waiting");
            assert_eq!(dict_u64(initial, "request_id"), expected_initial.request_id);
            assert_eq!(dict_string(initial, "model_id"), expected_initial.model_id);
            assert_eq!(
                dict_tokens(initial, "prompt_tokens"),
                expected_initial.prompt_tokens
            );
            assert_eq!(dict_u32(initial, "output_len"), expected_initial.output_len);
            assert_eq!(
                py_dict_to_json(initial),
                serde_json::to_value(expected_initial)
                    .expect("sdk initial snapshot should serialize")
            );

            for _ in 0..8 {
                let expected_snapshot = expected_session
                    .request_report(request_id)
                    .expect("sdk snapshot should succeed");
                let snapshot = session
                    .snapshot(py, request_id)
                    .expect("snapshot should succeed")
                    .expect("request should still exist");
                let snapshot = snapshot.bind(py);
                let state = dict_string(snapshot, "state");
                assert_eq!(
                    state,
                    serde_json::to_value(expected_snapshot.state)
                        .unwrap()
                        .as_str()
                        .unwrap()
                );
                assert_eq!(
                    dict_tokens(snapshot, "output_tokens"),
                    expected_snapshot.output_tokens
                );
                assert_eq!(
                    py_dict_to_json(snapshot),
                    serde_json::to_value(expected_snapshot.clone())
                        .expect("sdk snapshot should serialize")
                );
                if state == "finished" {
                    break;
                }

                let expected_step = expected_session
                    .step_report()
                    .expect("sdk step should succeed");
                let step = session.step(py).expect("step should succeed");
                let step = step.bind(py);
                let mut actual_step_json = py_dict_to_json(step);
                let mut expected_step_json =
                    serde_json::to_value(expected_step.clone()).expect("sdk step should serialize");
                assert_eq!(
                    dict_u32(step, "scheduled_requests"),
                    expected_step.scheduled_requests
                );
                assert_eq!(
                    dict_u32(step, "scheduled_tokens"),
                    expected_step.scheduled_tokens
                );
                normalize_measurement_fields(&mut actual_step_json);
                normalize_measurement_fields(&mut expected_step_json);
                assert_eq!(actual_step_json, expected_step_json);
            }

            let terminal = session
                .snapshot(py, request_id)
                .expect("snapshot should succeed")
                .expect("terminal request should still exist");
            let terminal = terminal.bind(py);
            let expected_terminal = expected_session
                .request_report(request_id)
                .expect("sdk terminal snapshot should exist");
            assert_eq!(
                dict_string(terminal, "state"),
                serde_json::to_value(expected_terminal.state)
                    .unwrap()
                    .as_str()
                    .unwrap()
            );
            assert_eq!(
                dict_u32(terminal, "output_len"),
                expected_terminal.output_len
            );
            assert_eq!(
                dict_tokens(terminal, "output_tokens"),
                expected_terminal.output_tokens
            );
            assert_eq!(
                py_dict_to_json(terminal),
                serde_json::to_value(expected_terminal)
                    .expect("sdk terminal snapshot should serialize")
            );
        });
    }

    #[test]
    fn python_session_cancel_matches_sdk_cancelled_state() {
        init_python();
        Python::with_gil(|py| {
            let mut session = preview_session();
            let mut expected_session = sdk_session();
            let request_id = session
                .submit(
                    Some(vec![7, 8, 9]),
                    None,
                    2,
                    0.0,
                    1.0,
                    0,
                    1.0,
                    0,
                    None,
                    None,
                )
                .expect("submit should succeed");
            expected_session
                .submit_generate_with_request_id(request_id, sample_sdk_request(&[7, 8, 9], 2))
                .expect("sdk submit should succeed");

            session.cancel(request_id).expect("cancel should succeed");
            expected_session
                .cancel_request(request_id)
                .expect("sdk cancel should succeed");

            let snapshot = session
                .snapshot(py, request_id)
                .expect("snapshot should succeed")
                .expect("cancelled request should exist");
            let snapshot = snapshot.bind(py);
            let expected_snapshot = expected_session
                .request_report(request_id)
                .expect("sdk cancelled snapshot should exist");

            assert_eq!(
                dict_string(snapshot, "state"),
                serde_json::to_value(expected_snapshot.state)
                    .unwrap()
                    .as_str()
                    .unwrap()
            );
            assert_eq!(
                dict_u64(snapshot, "request_id"),
                expected_snapshot.request_id
            );
            assert_eq!(
                snapshot
                    .get_item("cancel_requested")
                    .unwrap()
                    .unwrap()
                    .extract::<bool>()
                    .unwrap(),
                expected_snapshot.cancel_requested
            );
        });
    }

    #[test]
    fn python_session_stream_generate_matches_sdk_lifecycle_shape() {
        init_python();
        Python::with_gil(|py| {
            let mut session = preview_session();
            let stream = session
                .stream_generate(
                    py,
                    Some(vec![1, 2, 3]),
                    None,
                    2,
                    0.0,
                    1.0,
                    0,
                    1.0,
                    0,
                    None,
                    None,
                )
                .expect("stream should succeed");
            assert!(!session.closed());
            let active_stream_error = session
                .step(py)
                .expect_err("active stream should block reuse");
            assert!(active_stream_error.to_string().contains("active stream"));

            let mut stream = stream.bind(py).borrow_mut();
            let request_event = stream
                .next_event_dict(py)
                .expect("request event should succeed")
                .expect("request event should exist");
            let request_event = request_event.bind(py);
            assert_eq!(dict_string(request_event, "event"), "request");

            let request_payload = request_event
                .get_item("request")
                .unwrap()
                .unwrap()
                .downcast_into::<PyDict>()
                .unwrap();
            assert_eq!(dict_string(&request_payload, "state"), "waiting");

            let _step_event = stream
                .next_event_dict(py)
                .expect("step event should succeed")
                .expect("step event should exist");
            let _second_step_event = stream
                .next_event_dict(py)
                .expect("second step event should succeed")
                .expect("second step event should exist");
            let final_step_event = stream
                .next_event_dict(py)
                .expect("final step event should succeed")
                .expect("final step event should exist");
            let final_step_event = final_step_event.bind(py);
            assert_eq!(dict_string(final_step_event, "event"), "step");

            let response_event = stream
                .next_event_dict(py)
                .expect("response event should succeed")
                .expect("response event should exist");
            let response_event = response_event.bind(py);
            assert_eq!(dict_string(response_event, "event"), "response");
            let response_payload = response_event
                .get_item("response")
                .unwrap()
                .unwrap()
                .downcast_into::<PyDict>()
                .unwrap();
            assert_eq!(dict_tokens(&response_payload, "output_tokens"), vec![4, 5]);
            assert_eq!(dict_string(&response_payload, "status"), "finished");
            assert!(stream
                .next_event_dict(py)
                .expect("stream should finish")
                .is_none());
            drop(stream);

            let mut expected_session = sdk_session();
            let expected_events = expected_session
                .stream_generate(sample_sdk_request(&[1, 2, 3], 2))
                .expect("sdk stream should start")
                .collect::<Result<Vec<_>, _>>()
                .expect("sdk stream should complete");
            assert_eq!(expected_events.len(), 5);
            assert!(!session.closed());
            let next_response = session
                .generate(
                    py,
                    Some(vec![8, 9]),
                    None,
                    1,
                    0.0,
                    1.0,
                    0,
                    1.0,
                    0,
                    None,
                    None,
                )
                .expect("session should be reusable after stream completion");
            let next_response = next_response.bind(py);
            assert_eq!(dict_u64(next_response, "step_count"), 2);
            assert_eq!(dict_string(next_response, "status"), "finished");
        });
    }

    #[test]
    fn python_session_stream_generate_matches_sdk_payloads() {
        init_python();
        Python::with_gil(|py| {
            let mut session = preview_session();
            let stream = session
                .stream_generate(
                    py,
                    Some(vec![1, 2, 3]),
                    None,
                    2,
                    0.0,
                    1.0,
                    0,
                    1.0,
                    0,
                    None,
                    None,
                )
                .expect("stream should succeed");
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

            let mut expected_session = sdk_session();
            let mut expected = expected_session
                .stream_generate(sample_sdk_request(&[1, 2, 3], 2))
                .expect("sdk stream should start")
                .collect::<Result<Vec<_>, _>>()
                .expect("sdk stream should complete")
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
    }

    #[test]
    fn dropping_python_stream_iterator_restores_session_before_completion() {
        init_python();
        Python::with_gil(|py| {
            let mut session = preview_session();
            let stream = session
                .stream_generate(
                    py,
                    Some(vec![1, 2, 3]),
                    None,
                    2,
                    0.0,
                    1.0,
                    0,
                    1.0,
                    0,
                    None,
                    None,
                )
                .expect("stream should succeed");

            {
                let mut stream = stream.bind(py).borrow_mut();
                let request_event = stream
                    .next_event_dict(py)
                    .expect("request event should succeed")
                    .expect("request event should exist");
                let request_event = request_event.bind(py);
                assert_eq!(dict_string(request_event, "event"), "request");
            }

            let active_stream_error = session
                .snapshot(py, 1)
                .expect_err("active stream should block snapshot reuse");
            assert!(active_stream_error.to_string().contains("active stream"));

            drop(stream);

            let mut expected_session = sdk_session();
            {
                let mut expected_stream = expected_session
                    .stream_generate(sample_sdk_request(&[1, 2, 3], 2))
                    .expect("sdk stream should start");
                let _request_event = expected_stream
                    .next()
                    .expect("sdk request event should exist")
                    .expect("sdk request event should succeed");
                drop(expected_stream);
            }
            let expected = expected_session
                .generate(sample_sdk_request(&[8, 9], 1))
                .expect("sdk generate should succeed");
            let response = session
                .generate(
                    py,
                    Some(vec![8, 9]),
                    None,
                    1,
                    0.0,
                    1.0,
                    0,
                    1.0,
                    0,
                    None,
                    None,
                )
                .expect("session should be reusable after dropped stream");
            let response = response.bind(py);
            assert_eq!(dict_string(response, "status"), "finished");
            assert_eq!(
                dict_tokens(response, "output_tokens"),
                expected.output_tokens
            );
            assert_eq!(
                py_dict_to_json(response),
                serde_json::to_value(expected).expect("sdk response should serialize")
            );
        });
    }

    #[test]
    fn python_session_compatibility_generate_returns_text_fields() {
        init_python();
        Python::with_gil(|py| {
            let mut session = compatibility_session();
            let response = session
                .generate(
                    py,
                    None,
                    Some("hello from python".to_string()),
                    2,
                    0.0,
                    1.0,
                    0,
                    1.0,
                    0,
                    None,
                    None,
                )
                .expect("compatibility generate should succeed");
            let response = response.bind(py);

            assert_eq!(dict_string(response, "model_id"), "qwen3_dense");
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
            assert_eq!(dict_string(&runtime, "support_tier"), "compatibility");
        });
    }

    #[test]
    fn python_session_compatibility_server_generate_supports_token_prompts() {
        init_python();
        let (server_url, server_handle) = spawn_compat_completion_server(
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
            let mut session = compatibility_server_session(server_url);
            let response = session
                .generate(
                    py,
                    Some(vec![1, 2, 3]),
                    None,
                    2,
                    0.0,
                    1.0,
                    0,
                    1.0,
                    0,
                    None,
                    None,
                )
                .expect("compatibility server generate should succeed");
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
            .expect("compatibility server thread should finish");
    }

    #[test]
    fn python_session_compatibility_server_stream_generate_matches_sdk_payloads() {
        init_python();
        let (server_url, server_handle) = spawn_compat_completion_stream_server(
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
            let mut session = compatibility_server_session(server_url.clone());
            let stream = session
                .stream_generate(
                    py,
                    Some(vec![1, 2, 3]),
                    None,
                    2,
                    0.0,
                    1.0,
                    0,
                    1.0,
                    0,
                    None,
                    None,
                )
                .expect("compatibility stream should succeed");
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

            let mut expected_session = sdk_compatibility_server_session(server_url);
            let mut expected = expected_session
                .stream_generate(sample_sdk_request(&[1, 2, 3], 2))
                .expect("sdk compatibility stream should start")
                .collect::<Result<Vec<_>, _>>()
                .expect("sdk compatibility stream should complete")
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
            .expect("compatibility server thread should finish");
    }

    #[test]
    fn python_session_compatibility_stepwise_request_control_matches_sdk() {
        init_python();
        let (server_url, server_handle) = spawn_compat_completion_stream_server(
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
            let mut session = compatibility_server_session(server_url.clone());
            let mut expected_session = sdk_compatibility_server_session(server_url);
            let expected_request_id = expected_session
                .submit_generate(sample_sdk_request(&[1, 2, 3], 2))
                .expect("sdk compatibility submit should succeed");
            let request_id = session
                .submit(
                    Some(vec![1, 2, 3]),
                    None,
                    2,
                    0.0,
                    1.0,
                    0,
                    1.0,
                    0,
                    None,
                    None,
                )
                .expect("compatibility submit should succeed");
            assert_eq!(request_id, expected_request_id);

            for _ in 0..4 {
                let expected_snapshot = expected_session
                    .request_report(request_id)
                    .expect("sdk compatibility snapshot should exist");
                let snapshot = session
                    .snapshot(py, request_id)
                    .expect("snapshot should succeed")
                    .expect("compatibility request should still exist");
                let snapshot = snapshot.bind(py);
                let state = dict_string(snapshot, "state");
                assert_eq!(
                    py_dict_to_json(snapshot),
                    serde_json::to_value(expected_snapshot.clone())
                        .expect("sdk compatibility snapshot should serialize")
                );
                if state == "finished" {
                    break;
                }

                let expected_step = expected_session
                    .step_report()
                    .expect("sdk compatibility step should succeed");
                let step = session.step(py).expect("compatibility step should succeed");
                let step = step.bind(py);
                let mut actual_step_json = py_dict_to_json(step);
                let mut expected_step_json = serde_json::to_value(expected_step.clone())
                    .expect("sdk compatibility step should serialize");
                normalize_measurement_fields(&mut actual_step_json);
                normalize_measurement_fields(&mut expected_step_json);
                assert_eq!(actual_step_json, expected_step_json);
            }

            let terminal = session
                .snapshot(py, request_id)
                .expect("snapshot should succeed")
                .expect("terminal compatibility request should exist");
            let terminal = terminal.bind(py);
            let expected_terminal = expected_session
                .request_report(request_id)
                .expect("sdk compatibility terminal snapshot should exist");
            assert_eq!(
                py_dict_to_json(terminal),
                serde_json::to_value(expected_terminal)
                    .expect("sdk compatibility terminal snapshot should serialize")
            );
        });

        server_handle
            .join()
            .expect("compatibility server thread should finish");
    }

    #[test]
    fn python_session_compatibility_stepwise_multiple_requests_match_sdk() {
        init_python();
        let expected_prompts = vec![json!([1, 2, 3]), json!([7, 8, 9])];
        let expected_prompts_for_request = expected_prompts.clone();
        let (server_url, server_handle) = spawn_compat_completion_stream_server(
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
                assert!(expected_prompts_for_request
                    .iter()
                    .any(|candidate| prompt == candidate));
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
            },
        );

        Python::with_gil(|py| {
            let mut session = compatibility_server_session(server_url.clone());
            let mut expected_session = sdk_compatibility_server_session(server_url);
            let first_request_id = session
                .submit(
                    Some(vec![1, 2, 3]),
                    None,
                    2,
                    0.0,
                    1.0,
                    0,
                    1.0,
                    0,
                    None,
                    None,
                )
                .expect("first compatibility submit should succeed");
            let second_request_id = session
                .submit(
                    Some(vec![7, 8, 9]),
                    None,
                    2,
                    0.0,
                    1.0,
                    0,
                    1.0,
                    0,
                    None,
                    None,
                )
                .expect("second compatibility submit should succeed");
            expected_session
                .submit_generate_with_request_id(
                    first_request_id,
                    sample_sdk_request(&[1, 2, 3], 2),
                )
                .expect("first sdk compatibility submit should succeed");
            expected_session
                .submit_generate_with_request_id(
                    second_request_id,
                    sample_sdk_request(&[7, 8, 9], 2),
                )
                .expect("second sdk compatibility submit should succeed");

            for _ in 0..2 {
                let expected_step = expected_session
                    .step_report()
                    .expect("sdk compatibility aggregated step should succeed");
                let step = session
                    .step(py)
                    .expect("compatibility aggregated step should succeed");
                let step = step.bind(py);
                let mut actual_step_json = py_dict_to_json(step);
                let mut expected_step_json = serde_json::to_value(expected_step.clone())
                    .expect("sdk compatibility step should serialize");
                normalize_measurement_fields(&mut actual_step_json);
                normalize_measurement_fields(&mut expected_step_json);
                assert_eq!(actual_step_json, expected_step_json);

                for request_id in [first_request_id, second_request_id] {
                    let snapshot = session
                        .snapshot(py, request_id)
                        .expect("snapshot should succeed")
                        .expect("compatibility request should still exist");
                    let snapshot = snapshot.bind(py);
                    let expected_snapshot = expected_session
                        .request_report(request_id)
                        .expect("sdk compatibility snapshot should exist");
                    assert_eq!(
                        py_dict_to_json(snapshot),
                        serde_json::to_value(expected_snapshot)
                            .expect("sdk compatibility snapshot should serialize")
                    );
                }
            }
        });

        server_handle
            .join()
            .expect("compatibility server thread should finish");
    }

    #[test]
    fn python_session_compatibility_cancel_matches_sdk_cancelled_state() {
        init_python();
        let (server_url, server_handle) = spawn_compat_completion_stream_server(
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
            let mut session = compatibility_server_session(server_url.clone());
            let mut expected_session = sdk_compatibility_server_session(server_url);
            let request_id = session
                .submit(
                    Some(vec![7, 8, 9]),
                    None,
                    2,
                    0.0,
                    1.0,
                    0,
                    1.0,
                    0,
                    None,
                    None,
                )
                .expect("compatibility submit should succeed");
            expected_session
                .submit_generate_with_request_id(request_id, sample_sdk_request(&[7, 8, 9], 2))
                .expect("sdk compatibility submit should succeed");

            session
                .cancel(request_id)
                .expect("compatibility cancel should succeed");
            expected_session
                .cancel_request(request_id)
                .expect("sdk compatibility cancel should succeed");

            let snapshot = session
                .snapshot(py, request_id)
                .expect("snapshot should succeed")
                .expect("cancelled compatibility request should exist");
            let snapshot = snapshot.bind(py);
            let expected_snapshot = expected_session
                .request_report(request_id)
                .expect("sdk compatibility cancelled snapshot should exist");

            assert_eq!(
                py_dict_to_json(snapshot),
                serde_json::to_value(expected_snapshot)
                    .expect("sdk compatibility cancelled snapshot should serialize")
            );
        });

        server_handle
            .join()
            .expect("compatibility server thread should finish");
    }

    #[test]
    fn python_session_native_rejects_input_text() {
        init_python();
        Python::with_gil(|py| {
            let mut session = preview_session();
            let error = session
                .generate(
                    py,
                    None,
                    Some("native should reject text".to_string()),
                    2,
                    0.0,
                    1.0,
                    0,
                    1.0,
                    0,
                    None,
                    None,
                )
                .expect_err("native session should reject input_text");

            assert!(error
                .to_string()
                .contains("native preview session only accepts pre-tokenized input_tokens"));
        });
    }
}
