use ax_engine_sdk::{DelegatedHttpTimeouts, GenerateRequest, GenerateSampling};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn delegated_http_timeouts_from_secs(
    connect_secs: u64,
    read_secs: u64,
    write_secs: u64,
) -> PyResult<DelegatedHttpTimeouts> {
    if connect_secs == 0 || read_secs == 0 || write_secs == 0 {
        return Err(PyValueError::new_err(
            "delegated HTTP timeout values must be greater than zero",
        ));
    }
    Ok(DelegatedHttpTimeouts::from_secs(
        connect_secs,
        read_secs,
        write_secs,
    ))
}

pub(crate) fn build_generate_request(
    model_id: String,
    input_tokens: Vec<u32>,
    input_text: Option<String>,
    max_output_tokens: u32,
    sampling: GenerateSampling,
    stop_sequences: Vec<String>,
    metadata: Option<String>,
) -> GenerateRequest {
    GenerateRequest {
        model_id,
        input_tokens,
        input_text,
        max_output_tokens,
        sampling,
        stop_sequences,
        metadata,
    }
}
