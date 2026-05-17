use ax_engine_sdk::EmbeddingPooling;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub(crate) fn parse_pooling(pooling: &str) -> PyResult<EmbeddingPooling> {
    match pooling {
        "last" => Ok(EmbeddingPooling::Last),
        "mean" => Ok(EmbeddingPooling::Mean),
        "cls" => Ok(EmbeddingPooling::Cls),
        other => Err(PyValueError::new_err(format!(
            "unknown pooling '{other}': expected 'last', 'mean', or 'cls'"
        ))),
    }
}

pub(crate) fn floats_to_pybytes<'py>(
    py: Python<'py>,
    floats: &[f32],
) -> PyResult<Bound<'py, PyBytes>> {
    // Write little-endian f32 bytes directly into the PyBytes buffer — avoids
    // a separate Vec<u8> staging allocation and the per-element PyFloat
    // allocation we would pay if we returned list[float].
    PyBytes::new_with(py, std::mem::size_of_val(floats), |out| {
        for (chunk, &value) in out.chunks_exact_mut(4).zip(floats.iter()) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        Ok(())
    })
}
