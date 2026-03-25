use anyhow::Error;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pub fn py_runtime_error(error: Error) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

pub fn py_value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}
