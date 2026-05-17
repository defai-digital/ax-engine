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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delegated_http_timeouts_reject_zero_values() {
        let error = delegated_http_timeouts_from_secs(
            0,
            DelegatedHttpTimeouts::default_io_secs(),
            DelegatedHttpTimeouts::default_io_secs(),
        )
        .expect_err("zero connect timeout should fail closed");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn build_generate_request_preserves_python_arguments() {
        let request = build_generate_request(
            "qwen3".to_string(),
            vec![1, 2, 3],
            Some("hello".to_string()),
            8,
            GenerateSampling {
                temperature: 0.2,
                top_p: 0.9,
                top_k: 50,
                min_p: Some(0.05),
                seed: 42,
                deterministic: Some(true),
                repetition_penalty: 1.1,
                repetition_context_size: Some(64),
            },
            vec!["stop".to_string()],
            Some("metadata".to_string()),
        );

        assert_eq!(request.model_id, "qwen3");
        assert_eq!(request.input_tokens, vec![1, 2, 3]);
        assert_eq!(request.input_text.as_deref(), Some("hello"));
        assert_eq!(request.max_output_tokens, 8);
        assert_eq!(request.sampling.temperature, 0.2);
        assert_eq!(request.sampling.top_p, 0.9);
        assert_eq!(request.sampling.top_k, 50);
        assert_eq!(request.sampling.min_p, Some(0.05));
        assert_eq!(request.sampling.seed, 42);
        assert_eq!(request.sampling.deterministic, Some(true));
        assert_eq!(request.sampling.repetition_penalty, 1.1);
        assert_eq!(request.sampling.repetition_context_size, Some(64));
        assert_eq!(request.stop_sequences, vec!["stop"]);
        assert_eq!(request.metadata.as_deref(), Some("metadata"));
    }
}
