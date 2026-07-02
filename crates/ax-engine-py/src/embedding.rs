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
    let byte_len = floats.len() * 4;
    PyBytes::new_with(py, byte_len, |out| {
        // Bulk copy via u32 native-endian bytes. On Apple Silicon (always
        // little-endian), to_ne_bytes is identical to to_le_bytes and the
        // compiler lowers the loop to a single memcpy.
        for (chunk, &value) in out.chunks_exact_mut(4).zip(floats.iter()) {
            chunk.copy_from_slice(&value.to_ne_bytes());
        }
        Ok(())
    })
}

/// Build a `PyBytes` buffer from `raw` while optionally L2-normalizing rows
/// in the same pass.  When `normalize` is `true`, each contiguous row of
/// `hidden_size` floats is divided by its L2 norm (eps 1e-12 for stability)
/// *as it is written* into the output buffer.  This eliminates the
/// intermediate `Vec<f32>` that the two-step (normalize-then-copy) path would
/// otherwise require, saving one full `B*H` heap allocation per call.
///
/// `raw` must contain exactly `n_rows * hidden_size` floats.
pub(crate) fn embedding_matrix_to_pybytes<'py>(
    py: Python<'py>,
    raw: &[f32],
    hidden_size: usize,
    normalize: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let byte_len = raw.len() * 4;
    PyBytes::new_with(py, byte_len, |out| {
        // Bulk copy the raw f32 data into the PyBytes buffer.
        for (chunk, &value) in out.chunks_exact_mut(4).zip(raw.iter()) {
            chunk.copy_from_slice(&value.to_ne_bytes());
        }
        // Apply row-wise L2 normalization directly in the output buffer,
        // avoiding a separate Vec<f32> allocation.
        if normalize && hidden_size > 0 {
            for row_bytes in out.chunks_exact_mut(hidden_size * 4) {
                let mut sum_sq: f32 = 0.0;
                for chunk in row_bytes.chunks_exact(4) {
                    let val = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    sum_sq += val * val;
                }
                let inv_norm = 1.0_f32 / (sum_sq.sqrt() + 1e-12);
                for chunk in row_bytes.chunks_exact_mut(4) {
                    let val = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    let scaled = val * inv_norm;
                    chunk.copy_from_slice(&scaled.to_ne_bytes());
                }
            }
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

    #[test]
    fn embedding_matrix_to_pybytes_no_normalize() {
        init_python();
        Python::attach(|py| {
            let bytes = embedding_matrix_to_pybytes(py, &[1.0, 2.0, 3.0, 4.0], 2, false)
                .expect("should encode");
            let raw = bytes.as_bytes();
            let v0 = f32::from_ne_bytes(raw[0..4].try_into().unwrap());
            let v1 = f32::from_ne_bytes(raw[4..8].try_into().unwrap());
            let v2 = f32::from_ne_bytes(raw[8..12].try_into().unwrap());
            let v3 = f32::from_ne_bytes(raw[12..16].try_into().unwrap());
            assert_eq!([v0, v1, v2, v3], [1.0, 2.0, 3.0, 4.0]);
        });
    }

    #[test]
    fn embedding_matrix_to_pybytes_with_normalize() {
        init_python();
        Python::attach(|py| {
            // Row [3.0, 4.0] has L2 norm 5.0 -> normalized [0.6, 0.8]
            let bytes =
                embedding_matrix_to_pybytes(py, &[3.0, 4.0], 2, true).expect("should encode");
            let raw = bytes.as_bytes();
            let v0 = f32::from_ne_bytes(raw[0..4].try_into().unwrap());
            let v1 = f32::from_ne_bytes(raw[4..8].try_into().unwrap());
            assert!((v0 - 0.6).abs() < 1e-6);
            assert!((v1 - 0.8).abs() < 1e-6);
        });
    }
}
