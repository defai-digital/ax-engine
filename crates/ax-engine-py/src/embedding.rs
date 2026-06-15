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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    fn init_python() {
        static PYTHON_INIT: Once = Once::new();
        PYTHON_INIT.call_once(pyo3::Python::initialize);
    }

    #[test]
    fn parse_pooling_accepts_supported_labels() {
        assert_eq!(parse_pooling("last").unwrap(), EmbeddingPooling::Last);
        assert_eq!(parse_pooling("mean").unwrap(), EmbeddingPooling::Mean);
        assert_eq!(parse_pooling("cls").unwrap(), EmbeddingPooling::Cls);
    }

    #[test]
    fn parse_pooling_rejects_unknown_label() {
        let error = parse_pooling("sum").expect_err("unsupported pooling should fail");

        assert!(
            error
                .to_string()
                .contains("expected 'last', 'mean', or 'cls'")
        );
    }

    #[test]
    fn floats_to_pybytes_uses_little_endian_f32_layout() {
        init_python();

        Python::attach(|py| {
            let bytes = floats_to_pybytes(py, &[1.0, -2.5]).expect("floats should encode");
            assert_eq!(
                bytes.as_bytes(),
                &[0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x20, 0xc0]
            );
        });
    }
}
