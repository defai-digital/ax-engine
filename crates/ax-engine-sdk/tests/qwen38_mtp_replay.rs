//! Opt-in replay through the native session used by the server.
#![cfg(feature = "qwen-replay-diagnostics")]
#![allow(clippy::unwrap_used, clippy::panic)]

use ax_engine_mlx::qwen_mtp_logit_probe as probe;
use ax_engine_sdk::{
    EngineSession, EngineSessionConfig, GenerateRequest, GenerateResponse, GenerateSampling,
    GenerateStatus, MlxMtpPolicy, PreviewBackendRequest, PreviewSessionConfigRequest,
};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayCase {
    case_id: String,
    prompt_tokens: Vec<u32>,
    max_output_tokens: u32,
    ignore_eos: bool,
    direct_tokens: Vec<u32>,
    mtp_tokens: Vec<u32>,
}

fn generate(session: &mut EngineSession, case: &ReplayCase) -> GenerateResponse {
    let response = session
        .generate(GenerateRequest {
            model_id: "qwen3.8-27b".into(),
            input_tokens: case.prompt_tokens.clone(),
            input_text: None,
            multimodal_inputs: Default::default(),
            max_output_tokens: case.max_output_tokens,
            sampling: GenerateSampling {
                seed: 0,
                seed_explicit: true,
                ignore_eos: case.ignore_eos,
                ..GenerateSampling::default()
            },
            stop_sequences: Vec::new(),
            metadata: None,
            prefill_hints: Default::default(),
        })
        .unwrap();
    assert_eq!(response.status, GenerateStatus::Finished);
    response
}

fn require_measured_output(case: &ReplayCase, arm: &str, actual: &[u32], expected: &[u32]) {
    if actual != expected {
        let index = actual
            .iter()
            .zip(expected)
            .position(|(left, right)| left != right)
            .unwrap_or_else(|| actual.len().min(expected.len()));
        panic!(
            "{} {arm} differs from the recorded output at index {index}: actual={:?}, expected={:?}, lengths={}/{}",
            case.case_id,
            actual.get(index),
            expected.get(index),
            actual.len(),
            expected.len(),
        );
    }
}

#[test]
#[ignore = "requires the pinned Qwen 3.8 27B pack, measured cases and arm-specific environment"]
fn qwen38_27b_replay_native_session_arm() {
    let arm = env::var("AX_QWEN38_REPLAY_ARM").unwrap();
    let direct = match arm.as_str() {
        "direct" => true,
        "mtp" => false,
        _ => panic!("AX_QWEN38_REPLAY_ARM must be direct or mtp"),
    };
    assert_eq!(env::var("AX_MLX_DFLASH2").as_deref(), Ok("0"));
    assert_eq!(
        env::var("AX_ENGINE_PREFIX_REUSE_DISABLED").as_deref(),
        Ok("1")
    );
    assert_eq!(
        env::var("AX_NO_SPEC").ok().as_deref(),
        direct.then_some("1")
    );
    let model_dir = PathBuf::from(env::var_os("AX_QWEN38_27B_MTP_S_MODEL_DIR").unwrap());
    let manifest_sha256 = format!(
        "{:x}",
        Sha256::digest(fs::read(model_dir.join("model-manifest.json")).unwrap())
    );
    assert_eq!(
        manifest_sha256,
        "524163237ca2b879bff41f782a5eee6a1266b4f80f5d2c99e07117b7b899127a"
    );
    let input = PathBuf::from(env::var_os("AX_QWEN38_27B_MTP_MARGIN_CASE").unwrap());
    let cases: Vec<ReplayCase> = serde_json::from_slice(&fs::read(input).unwrap()).unwrap();
    assert!(!cases.is_empty());
    let config = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
        backend_request: PreviewBackendRequest::shipping_mlx(),
        mlx_model_artifacts_dir: Some(model_dir),
        mlx_disable_ngram_acceleration: direct,
        mlx_mtp_policy: if direct {
            MlxMtpPolicy::Disabled
        } else {
            MlxMtpPolicy::Required
        },
        mlx_mtp_disable_ngram_stacking: true,
        mlx_prefill_chunk: Some(2048),
        ..PreviewSessionConfigRequest::default()
    })
    .unwrap();
    let mut session = EngineSession::new(config).unwrap();
    for case in cases {
        let (measured, rival) = if direct {
            (&case.direct_tokens, &case.mtp_tokens)
        } else {
            (&case.mtp_tokens, &case.direct_tokens)
        };
        let baseline = generate(&mut session, &case);
        require_measured_output(&case, &arm, &baseline.output_tokens, measured);
        let indices: Vec<_> = measured
            .iter()
            .zip(rival)
            .enumerate()
            .filter_map(|(index, (left, right))| (left != right).then_some(index))
            .collect();
        let mut scores = Vec::new();
        if !indices.is_empty() {
            probe::start(case.prompt_tokens.len(), &indices);
            let observed = generate(&mut session, &case);
            let captured = probe::finish();
            require_measured_output(&case, &arm, &observed.output_tokens, measured);
            assert_eq!(observed.finish_reason, baseline.finish_reason);
            for index in indices {
                let row = captured
                    .scores(index, measured[index], rival[index])
                    .unwrap_or_else(|error| {
                        panic!("{} {arm} index {index}: {error}", case.case_id)
                    });
                scores.push(json!({"index": index, "row": row}));
            }
        }
        println!(
            "{}",
            json!({
                "case_id": case.case_id,
                "arm": arm,
                "manifest_sha256": manifest_sha256,
                "prompt_tokens": case.prompt_tokens,
                "max_output_tokens": case.max_output_tokens,
                "ignore_eos": case.ignore_eos,
                "tokens": baseline.output_tokens,
                "finish_reason": baseline.finish_reason,
                "route": baseline.route,
                "scores": scores,
            })
        );
    }
}
