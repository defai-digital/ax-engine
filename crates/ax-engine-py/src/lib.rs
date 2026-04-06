#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

mod errors;
mod gil;
mod model;
mod session;

use pyo3::prelude::*;

use crate::model::Model;
use crate::session::{GenerationResult, Session, TextStream};

#[pymodule]
fn _ax_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_class::<Session>()?;
    m.add_class::<TextStream>()?;
    m.add_class::<GenerationResult>()?;
    Ok(())
}
