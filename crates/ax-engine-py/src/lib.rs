#![allow(clippy::collapsible_if)]

mod dicts;
mod embedding;
mod errors;
mod json_to_py;
mod request;
mod session;
mod stream;

use pyo3::prelude::*;

use crate::errors::{EngineBackendError, EngineError, EngineInferenceError, EngineStateError};
use crate::session::Session;
use crate::stream::GenerateStreamIterator;

#[pymodule]
fn _ax_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("EngineError", m.py().get_type::<EngineError>())?;
    m.add(
        "EngineBackendError",
        m.py().get_type::<EngineBackendError>(),
    )?;
    m.add(
        "EngineInferenceError",
        m.py().get_type::<EngineInferenceError>(),
    )?;
    m.add("EngineStateError", m.py().get_type::<EngineStateError>())?;
    m.add_class::<Session>()?;
    m.add_class::<GenerateStreamIterator>()?;
    Ok(())
}
