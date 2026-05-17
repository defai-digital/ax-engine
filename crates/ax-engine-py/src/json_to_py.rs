use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use serde_json::{Number, Value};

#[allow(dead_code)]
pub(crate) fn json_to_py(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(value) => value.into_py_any(py),
        Value::Number(value) => number_to_py(py, value),
        Value::String(value) => Ok(PyString::new(py, value).into_any().unbind()),
        Value::Array(values) => {
            let list = PyList::empty(py);
            for value in values {
                list.append(json_to_py(py, value)?)?;
            }
            Ok(list.into_any().unbind())
        }
        Value::Object(values) => {
            let dict = PyDict::new(py);
            for (key, value) in values {
                dict.set_item(key, json_to_py(py, value)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

fn number_to_py(py: Python<'_>, value: &Number) -> PyResult<PyObject> {
    if let Some(value) = value.as_i64() {
        return value.into_py_any(py);
    }
    if let Some(value) = value.as_u64() {
        return value.into_py_any(py);
    }
    if let Some(value) = value.as_f64() {
        return value.into_py_any(py);
    }
    Ok(py.None())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyDict, PyList};
    use serde_json::json;
    use std::sync::Once;

    fn init_python() {
        static PYTHON_INIT: Once = Once::new();
        PYTHON_INIT.call_once(pyo3::prepare_freethreaded_python);
    }

    #[test]
    fn converts_primitives_lists_objects_and_null() {
        init_python();
        Python::with_gil(|py| {
            let value = json!({
                "none": null,
                "bool": true,
                "int": -3,
                "uint": 7_u64,
                "float": 1.5,
                "string": "ok",
                "list": [1, null, "two"],
            });
            let object = json_to_py(py, &value).expect("json should convert");
            let dict = object
                .bind(py)
                .downcast::<PyDict>()
                .expect("object should be dict");
            assert!(dict.get_item("none").unwrap().unwrap().is_none());
            assert!(
                dict.get_item("bool")
                    .unwrap()
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            );
            assert_eq!(
                dict.get_item("int")
                    .unwrap()
                    .unwrap()
                    .extract::<i64>()
                    .unwrap(),
                -3
            );
            assert_eq!(
                dict.get_item("uint")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                7
            );
            assert_eq!(
                dict.get_item("float")
                    .unwrap()
                    .unwrap()
                    .extract::<f64>()
                    .unwrap(),
                1.5
            );
            assert_eq!(
                dict.get_item("string")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "ok"
            );
            let list = dict
                .get_item("list")
                .unwrap()
                .unwrap()
                .downcast_into::<PyList>()
                .unwrap();
            assert_eq!(list.len(), 3);
        });
    }
}
