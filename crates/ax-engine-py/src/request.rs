use ax_engine_sdk::{
    DelegatedHttpTimeouts, GenerateRequest, GenerateSampling, RequestMultimodalInputs,
    UnlimitedOcrImageRuntimeInput, UnlimitedOcrRuntimeInputs,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList, PyTuple};
use serde_json::{Map, Number, Value};

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

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_generate_request(
    model_id: String,
    input_tokens: Vec<u32>,
    input_text: Option<String>,
    multimodal_inputs: RequestMultimodalInputs,
    max_output_tokens: u32,
    sampling: GenerateSampling,
    stop_sequences: Vec<String>,
    metadata: Option<String>,
) -> GenerateRequest {
    GenerateRequest {
        model_id,
        input_tokens,
        input_text,
        multimodal_inputs,
        max_output_tokens,
        sampling,
        stop_sequences,
        metadata,
    }
}

pub(crate) fn multimodal_inputs_from_py(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<RequestMultimodalInputs> {
    let Some(value) = value else {
        return Ok(RequestMultimodalInputs::default());
    };
    if value.is_none() {
        return Ok(RequestMultimodalInputs::default());
    }
    if let Some(inputs) = unlimited_ocr_inputs_from_py(value)? {
        return Ok(inputs);
    }
    let json = py_any_to_json(value)?;
    serde_json::from_value(json).map_err(|error| {
        PyValueError::new_err(format!(
            "multimodal_inputs must match RequestMultimodalInputs schema: {error}"
        ))
    })
}

fn unlimited_ocr_inputs_from_py(
    value: &Bound<'_, PyAny>,
) -> PyResult<Option<RequestMultimodalInputs>> {
    let Ok(payload) = value.cast::<PyDict>() else {
        return Ok(None);
    };
    let Some(root_value) = payload.get_item("unlimited_ocr")? else {
        return Ok(None);
    };
    if payload.contains("gemma4_unified")? {
        return Err(PyValueError::new_err(
            "multimodal request may select only one provider schema",
        ));
    }
    let root = root_value.cast::<PyDict>().map_err(|_| {
        PyValueError::new_err("multimodal_inputs.unlimited_ocr must be a dictionary")
    })?;
    let image_token_id = root
        .get_item("image_token_id")?
        .ok_or_else(|| {
            PyValueError::new_err("multimodal_inputs.unlimited_ocr.image_token_id is required")
        })?
        .extract::<u32>()?;
    let soft_token_count = root
        .get_item("soft_token_count")?
        .ok_or_else(|| {
            PyValueError::new_err("multimodal_inputs.unlimited_ocr.soft_token_count is required")
        })?
        .extract::<u32>()?;
    let cropping = root
        .get_item("cropping")?
        .map(|value| value.extract::<bool>())
        .transpose()?
        .unwrap_or(true);
    let images_value = root.get_item("images")?.ok_or_else(|| {
        PyValueError::new_err("multimodal_inputs.unlimited_ocr.images is required")
    })?;
    let images_list = images_value.cast::<PyList>().map_err(|_| {
        PyValueError::new_err("multimodal_inputs.unlimited_ocr.images must be a list")
    })?;
    let mut images = Vec::with_capacity(images_list.len());
    for (index, item) in images_list.iter().enumerate() {
        let image = item.cast::<PyDict>().map_err(|_| {
            PyValueError::new_err(format!(
                "multimodal_inputs.unlimited_ocr.images[{index}] must be a dictionary"
            ))
        })?;
        let width = image
            .get_item("width")?
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "multimodal_inputs.unlimited_ocr.images[{index}].width is required"
                ))
            })?
            .extract::<u32>()?;
        let height = image
            .get_item("height")?
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "multimodal_inputs.unlimited_ocr.images[{index}].height is required"
                ))
            })?
            .extract::<u32>()?;
        let rgb_value = image.get_item("rgb_bytes")?.ok_or_else(|| {
            PyValueError::new_err(format!(
                "multimodal_inputs.unlimited_ocr.images[{index}].rgb_bytes is required"
            ))
        })?;
        let rgb_bytes = if let Ok(bytes) = rgb_value.cast::<PyBytes>() {
            bytes.as_bytes().to_vec()
        } else {
            rgb_value.extract::<Vec<u8>>().map_err(|_| {
                PyValueError::new_err(format!(
                    "multimodal_inputs.unlimited_ocr.images[{index}].rgb_bytes must be bytes or a list of byte values"
                ))
            })?
        };
        images.push(UnlimitedOcrImageRuntimeInput {
            width,
            height,
            rgb_bytes,
        });
    }

    Ok(Some(RequestMultimodalInputs {
        gemma4_unified: None,
        unlimited_ocr: Some(UnlimitedOcrRuntimeInputs {
            image_token_id,
            soft_token_count,
            cropping,
            images,
        }),
    }))
}

fn py_any_to_json(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(value) = value.extract::<bool>() {
        return Ok(Value::Bool(value));
    }
    if let Ok(value) = value.extract::<i64>() {
        return Ok(Value::Number(Number::from(value)));
    }
    if let Ok(value) = value.extract::<u64>() {
        return Ok(Value::Number(Number::from(value)));
    }
    if let Ok(value) = value.extract::<f64>() {
        return Number::from_f64(value).map(Value::Number).ok_or_else(|| {
            PyValueError::new_err("multimodal_inputs cannot contain NaN or infinite floats")
        });
    }
    if let Ok(value) = value.extract::<String>() {
        return Ok(Value::String(value));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        return py_dict_to_json(dict);
    }
    if let Ok(list) = value.cast::<PyList>() {
        return py_iterable_to_json_array(list.iter());
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        return py_iterable_to_json_array(tuple.iter());
    }
    Err(PyValueError::new_err(format!(
        "multimodal_inputs contains unsupported Python value of type {}",
        value
            .get_type()
            .name()
            .map(|name| name.to_string())
            .unwrap_or_else(|_| "<unknown>".to_string())
    )))
}

fn py_iterable_to_json_array<'py>(
    iter: impl Iterator<Item = Bound<'py, PyAny>>,
) -> PyResult<Value> {
    iter.map(|item| py_any_to_json(&item))
        .collect::<PyResult<Vec<_>>>()
        .map(Value::Array)
}

fn py_dict_to_json(dict: &Bound<'_, PyDict>) -> PyResult<Value> {
    let mut map = Map::new();
    for (key, value) in dict.iter() {
        let key = key.extract::<String>().map_err(|_| {
            PyValueError::new_err("multimodal_inputs dictionaries must use string keys")
        })?;
        map.insert(key, py_any_to_json(&value)?);
    }
    Ok(Value::Object(map))
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
            RequestMultimodalInputs::default(),
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
                no_repeat_ngram_size: 35,
                ngram_window: 128,
                ignore_eos: true,
            },
            vec!["stop".to_string()],
            Some("metadata".to_string()),
        );

        assert_eq!(request.model_id, "qwen3");
        assert_eq!(request.input_tokens, vec![1, 2, 3]);
        assert_eq!(request.input_text.as_deref(), Some("hello"));
        assert!(request.multimodal_inputs.is_empty());
        assert_eq!(request.max_output_tokens, 8);
        assert_eq!(request.sampling.temperature, 0.2);
        assert_eq!(request.sampling.top_p, 0.9);
        assert_eq!(request.sampling.top_k, 50);
        assert_eq!(request.sampling.min_p, Some(0.05));
        assert_eq!(request.sampling.seed, 42);
        assert_eq!(request.sampling.deterministic, Some(true));
        assert_eq!(request.sampling.repetition_penalty, 1.1);
        assert_eq!(request.sampling.repetition_context_size, Some(64));
        assert_eq!(request.sampling.no_repeat_ngram_size, 35);
        assert_eq!(request.sampling.ngram_window, 128);
        assert!(request.sampling.ignore_eos);
        assert_eq!(request.stop_sequences, vec!["stop"]);
        assert_eq!(request.metadata.as_deref(), Some("metadata"));
    }

    #[test]
    fn multimodal_inputs_from_py_deserializes_gemma4_image_payload() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let payload = PyDict::new(py);
            let root = PyDict::new(py);
            let image = PyDict::new(py);
            let span = PyDict::new(py);
            span.set_item("modality", "image").unwrap();
            span.set_item("placeholder_index", 1).unwrap();
            span.set_item("replacement_start", 3).unwrap();
            span.set_item("soft_token_count", 2).unwrap();
            span.set_item("replacement_token_count", 4).unwrap();
            image.set_item("span", span).unwrap();
            image
                .set_item("pixel_values", vec![0.0_f32, 1.0, 2.0, 3.0])
                .unwrap();
            image
                .set_item("pixel_position_ids", vec![[0_i32, 0_i32], [1, 0]])
                .unwrap();
            root.set_item("images", vec![image]).unwrap();
            root.set_item("audios", Vec::<Py<PyAny>>::new()).unwrap();
            root.set_item("videos", Vec::<Py<PyAny>>::new()).unwrap();
            payload.set_item("gemma4_unified", root).unwrap();

            let inputs = multimodal_inputs_from_py(Some(payload.as_any()))
                .expect("Python dict should deserialize");
            let gemma4 = inputs
                .gemma4_unified
                .expect("Gemma4 inputs should be present");
            assert_eq!(gemma4.images.len(), 1);
            assert_eq!(gemma4.images[0].span.soft_token_count, 2);
        });
    }

    #[test]
    fn multimodal_inputs_from_py_deserializes_unlimited_ocr_bytes_without_json_expansion() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let payload = PyDict::new(py);
            let root = PyDict::new(py);
            let image = PyDict::new(py);
            root.set_item("image_token_id", 128_815_u32).unwrap();
            root.set_item("soft_token_count", 273_u32).unwrap();
            image.set_item("width", 2_u32).unwrap();
            image.set_item("height", 1_u32).unwrap();
            image
                .set_item("rgb_bytes", PyBytes::new(py, &[0, 1, 2, 3, 4, 5]))
                .unwrap();
            root.set_item("images", vec![image]).unwrap();
            payload.set_item("unlimited_ocr", root).unwrap();

            let inputs = multimodal_inputs_from_py(Some(payload.as_any()))
                .expect("Python bytes payload should deserialize");
            let ocr = inputs
                .unlimited_ocr
                .expect("Unlimited-OCR inputs should be present");
            assert_eq!(ocr.image_token_id, 128_815);
            assert_eq!(ocr.soft_token_count, 273);
            assert!(ocr.cropping);
            assert_eq!(ocr.images.len(), 1);
            assert_eq!(ocr.images[0].rgb_bytes, vec![0, 1, 2, 3, 4, 5]);
        });
    }
}
