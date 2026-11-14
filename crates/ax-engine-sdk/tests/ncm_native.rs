//! Opt-in real-weight functional controls. Never downloads weights in CI.
#![cfg(feature = "mlx-native")]
#![allow(clippy::unwrap_used)]
use ax_engine_core::ncm::NcmError;
use ax_engine_sdk::ncm::{NcmGenerateRequest, NcmOutputMode, NcmSamplingMode, NcmSession};
use std::{path::PathBuf, time::Duration};
fn request() -> NcmGenerateRequest {
    NcmGenerateRequest {
        sampling: NcmSamplingMode::Greedy,
        output_mode: NcmOutputMode::FullSequence128,
    }
}
#[test]
#[ignore = "requires pinned TinyStories weights and an external retained greedy trace"]
fn reference_free_identity_and_lifecycle() {
    let path = PathBuf::from(std::env::var_os("AX_NCM_TEST_MODEL").unwrap());
    let expected_path = PathBuf::from(std::env::var_os("AX_NCM_EXPECTED_TRACE").unwrap());
    let expected: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&expected_path).unwrap()).unwrap();
    let decoder = ax_engine_mlx::ncm::artifact::ByteDecoder::load(&path.join("bpe.json")).unwrap();
    for case in ["greedy-0", "seed17-0", "seed29"] {
        let trace: serde_json::Value = serde_json::from_slice(
            &std::fs::read(expected_path.parent().unwrap().join(format!("{case}.json"))).unwrap(),
        )
        .unwrap();
        let ids: Vec<u32> = serde_json::from_value(trace["tokens"][0].clone()).unwrap();
        assert_eq!(
            decoder.decode(ids).unwrap().text,
            trace["text"].as_str().unwrap()
        );
    }
    let session =
        NcmSession::load_experimental(&path, 2 * 1024 * 1024 * 1024, Duration::from_secs(60))
            .unwrap();
    let check = |response: ax_engine_sdk::ncm::NcmGenerateResponse| {
        assert_eq!(
            serde_json::json!(response.raw_bpe_ids),
            expected["tokens"][0]
        );
        assert_eq!(response.text, expected["text"].as_str().unwrap());
        assert_eq!(response.prompt_tokens, 0);
        assert_eq!(response.work_units, 14);
    };
    check(
        session
            .generate_unconditional(request())
            .unwrap()
            .finish()
            .unwrap(),
    );
    for boundary in 0..14 {
        let mut handle = session.generate_unconditional(request()).unwrap();
        assert!(matches!(
            session.generate_unconditional(request()),
            Err(NcmError::Busy)
        ));
        for _ in 0..boundary {
            assert!(handle.step().unwrap().1.is_none());
        }
        assert!(handle.cancellation().cancel());
        assert!(matches!(handle.step(), Err(NcmError::Cancelled)));
    }
    check(
        session
            .generate_unconditional(request())
            .unwrap()
            .finish()
            .unwrap(),
    );
    let mut dropped = session.generate_unconditional(request()).unwrap();
    for _ in 0..5 {
        dropped.step().unwrap();
    }
    drop(dropped);
    check(
        session
            .generate_unconditional(request())
            .unwrap()
            .finish()
            .unwrap(),
    );
    let mut handle = session.generate_unconditional(request()).unwrap();
    session.close();
    assert!(matches!(handle.step(), Err(NcmError::Cancelled)));
    assert!(session.generate_unconditional(request()).is_err());
    drop(handle);
    drop(session);
    let session =
        NcmSession::load_experimental(&path, 2 * 1024 * 1024 * 1024, Duration::from_nanos(1))
            .unwrap();
    let mut handle = session.generate_unconditional(request()).unwrap();
    assert!(matches!(handle.step(), Err(NcmError::Deadline)));
    assert!(session.generate_unconditional(request()).is_ok());
}
