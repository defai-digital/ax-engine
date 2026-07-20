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

        match py.detach(|| session.next_stream_event(state)) {
            Ok(Some(event)) => {
                let is_terminal = matches!(event, SdkGenerateStreamEvent::Response(_));
                let payload = stream_event_dict(py, &event);
                if is_terminal {
                    // Terminal Response: the native request is already done.
                    // Clear state before restore so we do not cancel a finished
                    // request (cancel is only for abandon/error paths).
                    self.state = None;
                }
                if is_terminal || payload.is_err() {
                    self.restore_session();
                }
                payload.map(Some)
            }
            Ok(None) => {
                // No more events. Leave `state` in place so restore_session can
                // cancel any still-live native request (Done phase cancel is
                // harmless; non-terminal abandon must cancel).
                self.restore_session();
                Ok(None)
            }
            Err(error) => {
                // Leave `state` set so restore_session cancels the in-flight
                // native request. Clearing state first was a silent cancel
                // skip: the request kept decoding/holding KV after the error.
                self.restore_session();
                Err(to_py_runtime_error(error))
            }
        }
    }

    fn restore_session(&mut self) {
        let Some(mut session) = self.session.take() else {
            return;
        };

        // A state still present here means the stream was abandoned before
        // reaching a terminal event (e.g. the Python iterator was dropped
        // mid-generation). Cancel the in-flight native request so it does not
        // keep co-decoding (and holding KV blocks) alongside every subsequent
        // generate/stream call on this session.
        if let Some(state) = self.state.take()
            && matches!(state, GenerateStreamState::Native(_))
        {
            let _ = session.cancel_request(state.request_id());
        }

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

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use ax_engine_sdk::{
        EngineSession, GenerateRequest, GenerateStreamEvent as SdkGenerateStreamEvent,
        GenerateStreamState,
    };

    use super::GenerateStreamIterator;
    use crate::session::SessionSlot;

    fn sample_request() -> GenerateRequest {
        GenerateRequest {
            model_id: "qwen3".to_string(),
            input_tokens: vec![1, 2],
            input_text: None,
            multimodal_inputs: Default::default(),
            max_output_tokens: 8,
            sampling: Default::default(),
            stop_sequences: Vec::new(),
            metadata: None,
        }
    }

    fn streaming_iterator(
        session: EngineSession,
        state: GenerateStreamState,
    ) -> GenerateStreamIterator {
        GenerateStreamIterator {
            owner: Arc::new(Mutex::new(SessionSlot::Streaming)),
            session: Some(session),
            state: Some(state),
        }
    }

    #[test]
    fn restore_session_cancels_native_request_when_state_still_present() {
        // Shared cancel path used by Drop and by Err/Ok(None) in next_event_dict.
        let mut session = EngineSession::new_deterministic_native_for_tests();
        let state = session
            .stream_generate_state(sample_request())
            .expect("native stream state should start");
        let request_id = state.request_id();
        assert!(
            session.has_active_stepwise_requests(),
            "submitted stream request must be active"
        );

        let owner = Arc::new(Mutex::new(SessionSlot::Streaming));
        let mut iter = GenerateStreamIterator {
            owner: Arc::clone(&owner),
            session: Some(session),
            state: Some(state),
        };
        iter.restore_session();

        let guard = owner.lock().expect("owner lock");
        let SessionSlot::Ready(session) = &*guard else {
            panic!("session should be restored to Ready after restore_session");
        };
        assert!(
            !session.has_active_stepwise_requests(),
            "restore_session with live native state must cancel the request"
        );
        let report = session
            .request_report(request_id)
            .expect("cancelled request should retain a report");
        assert!(
            report.cancel_requested
                || matches!(
                    report.state,
                    ax_engine_sdk::SessionRequestState::Cancelled
                        | ax_engine_sdk::SessionRequestState::Finished
                        | ax_engine_sdk::SessionRequestState::Failed
                ),
            "cancelled stream request should be terminal; got {report:?}"
        );
    }

    #[test]
    fn error_path_cancels_native_request_without_clearing_state_first() {
        // Regression: the Err arm used to do `state = None` then restore_session(),
        // so cancel_request never saw a live state and the native request kept
        // decoding. Drive a real next_stream_event error and the iterator path.
        use pyo3::Python;
        use std::sync::Once;

        static PYTHON_INIT: Once = Once::new();
        PYTHON_INIT.call_once(pyo3::Python::initialize);

        let mut session = EngineSession::new_deterministic_native_for_tests();
        let mut state = session
            .stream_generate_state(sample_request())
            .expect("native stream state should start");
        let request_id = state.request_id();
        // Advance past Request so the next call steps (and can hit the budget).
        let _ = session
            .next_stream_event(&mut state)
            .expect("request event")
            .expect("request event present");
        state.test_only_exhaust_native_step_budget();

        let mut iter = streaming_iterator(session, state);
        Python::attach(|py| {
            let err = iter
                .next_event_dict(py)
                .expect_err("exhausted step budget must surface as a stream error");
            let message = err.to_string();
            assert!(
                message.contains("did not terminate")
                    || message.contains("RequestDidNotTerminate")
                    || message.contains("terminate"),
                "unexpected error payload: {message}"
            );
        });

        assert!(
            iter.session.is_none(),
            "error path must restore the session to the owner"
        );
        assert!(
            iter.state.is_none(),
            "restore_session consumes state after cancel"
        );

        let guard = iter.owner.lock().expect("owner lock");
        let SessionSlot::Ready(session) = &*guard else {
            panic!("session should be Ready after error-path restore");
        };
        assert!(
            !session.has_active_stepwise_requests(),
            "error-path restore must cancel native request {request_id}"
        );
        let report = session
            .request_report(request_id)
            .expect("cancelled request should retain a report");
        assert!(
            report.cancel_requested
                || matches!(
                    report.state,
                    ax_engine_sdk::SessionRequestState::Cancelled
                        | ax_engine_sdk::SessionRequestState::Finished
                        | ax_engine_sdk::SessionRequestState::Failed
                ),
            "error-path cancel must terminalize the request; got {report:?}"
        );
    }

    #[test]
    fn terminal_response_clears_state_so_finished_request_is_not_cancelled() {
        let mut session = EngineSession::new_deterministic_native_for_tests();
        let mut state = session
            .stream_generate_state(sample_request())
            .expect("native stream state should start");
        let request_id = state.request_id();

        // Drain to a terminal Response through the real stream API.
        let mut saw_response = false;
        for _ in 0..64 {
            match session.next_stream_event(&mut state).expect("stream event") {
                Some(SdkGenerateStreamEvent::Response(_)) => {
                    saw_response = true;
                    break;
                }
                Some(_) => {}
                None => break,
            }
        }
        assert!(saw_response, "deterministic stream should emit Response");

        // Terminal path: clear state then restore — cancel must not re-fire on a
        // finished request (state is already None).
        let mut iter = streaming_iterator(session, state);
        iter.state = None;
        iter.restore_session();

        let guard = iter.owner.lock().expect("owner lock");
        let SessionSlot::Ready(session) = &*guard else {
            panic!("session should be Ready after terminal restore");
        };
        let report = session
            .request_report(request_id)
            .expect("finished request should retain a report");
        assert!(
            !report.cancel_requested,
            "terminal path must not mark a finished request as cancel_requested"
        );
    }
}
