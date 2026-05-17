use std::sync::{Arc, Mutex};

use ax_engine_sdk::{
    EngineSession, GenerateStreamEvent as SdkGenerateStreamEvent, GenerateStreamState,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::dicts::stream_event_dict;
use crate::errors::to_py_runtime_error;
use crate::session::SessionSlot;

#[pyclass(module = "ax_engine._ax_engine", unsendable)]
pub(crate) struct GenerateStreamIterator {
    pub(crate) owner: Arc<Mutex<SessionSlot>>,
    pub(crate) session: Option<EngineSession>,
    pub(crate) state: Option<GenerateStreamState>,
}

#[pymethods]
impl GenerateStreamIterator {
    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Py<PyDict>>> {
        self.next_event_dict(py)
    }
}

impl GenerateStreamIterator {
    pub(crate) fn next_event_dict<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Py<PyDict>>> {
        let Some(session) = self.session.as_mut() else {
            return Ok(None);
        };
        let Some(state) = self.state.as_mut() else {
            self.restore_session();
            return Ok(None);
        };

        match py.allow_threads(|| session.next_stream_event(state)) {
            Ok(Some(event)) => {
                let is_terminal = matches!(event, SdkGenerateStreamEvent::Response(_));
                let payload = stream_event_dict(py, &event);
                if is_terminal {
                    self.state = None;
                    self.restore_session();
                }
                Ok(Some(payload))
            }
            Ok(None) => {
                self.state = None;
                self.restore_session();
                Ok(None)
            }
            Err(error) => {
                self.state = None;
                self.restore_session();
                Err(to_py_runtime_error(error))
            }
        }
    }

    fn restore_session(&mut self) {
        let Some(session) = self.session.take() else {
            return;
        };

        let Ok(mut owner) = self.owner.lock() else {
            return;
        };
        match &*owner {
            SessionSlot::Closed => {}
            SessionSlot::Streaming => {
                *owner = SessionSlot::Ready(Box::new(session));
            }
            SessionSlot::Ready(_) => {}
        }
    }
}

impl Drop for GenerateStreamIterator {
    fn drop(&mut self) {
        self.restore_session();
    }
}
