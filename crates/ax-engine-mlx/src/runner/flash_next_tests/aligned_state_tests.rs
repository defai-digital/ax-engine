//! Real runner state evidence at identical consumed prefixes.

use super::*;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

fn write_blob(root: &Path, name: &str, bytes: &[u8]) -> serde_json::Value {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join(name))
        .unwrap();
    file.write_all(bytes).unwrap();
    serde_json::json!({
        "file": name, "bytes": bytes.len(), "sha256": format!("{:x}", Sha256::digest(bytes)),
    })
}

fn state_pair(
    root: &Path,
    label: &str,
    direct: &Qwen4ExpState,
    mtp: &Qwen4ExpState,
    layers: usize,
) -> serde_json::Value {
    let direct_bytes = flash_runner_state_bytes(direct, layers);
    let mtp_bytes = flash_runner_state_bytes(mtp, layers);
    serde_json::json!({
        "position": direct.position(),
        "mtp_position": mtp.position(),
        "direct": write_blob(root, &format!("{label}.direct.axkb"), &direct_bytes),
        "mtp": write_blob(root, &format!("{label}.mtp.axkb"), &mtp_bytes),
        "byte_exact": direct_bytes == mtp_bytes,
    })
}

fn decision(root: &Path, label: &str, logits: &MlxArray) -> serde_json::Value {
    let token = mlx_sys::argmax(logits, None);
    let values = mlx_sys::contiguous(&mlx_sys::astype(logits, MlxDtype::Float32, None), None);
    mlx_sys::eval(&[&token, &values]);
    let bytes: Vec<u8> = values
        .data_f32()
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    serde_json::json!({
        "token": token.data_u32()[0], "shape": values.shape(),
        "logits_f32_le": write_blob(root, &format!("{label}.logits.f32"), &bytes),
    })
}

fn collect(runner: &mut MlxRunner, prompt: &[u32], root: &Path) -> serde_json::Value {
    assert!(!prompt.is_empty());
    assert!(runner.has_mtp());
    assert!(matches!(
        runner.weights.qwen4_exp.as_ref().unwrap().target_schedule,
        Qwen4ExpTargetSchedule::CanonicalSingleton
    ));
    std::fs::create_dir(root).unwrap();
    let mut snapshots = [BTreeMap::new(), BTreeMap::new()];
    let mut runs = Vec::new();
    for (mode, states) in snapshots.iter_mut().enumerate() {
        *runner.prefix_cache.lock() = MlxPrefixCache::new(MlxPrefixCachePolicy {
            max_bytes: 64 * 1024 * 1024,
            max_entries: 128,
        });
        *runner.native_prefix_cache.lock() = MlxNativePrefixCache::new(MlxPrefixCachePolicy {
            max_bytes: 64 * 1024 * 1024,
            max_entries: 128,
        });
        runner.set_mtp_requested(mode == 1);
        runs.push(generate_observed(
            runner,
            prompt,
            usize::MAX,
            context(2100 + mode as u64, prompt.len(), 8),
            4,
            |state| {
                if state.position() >= prompt.len() {
                    assert!(states.insert(state.position(), state.clone()).is_none());
                }
            },
        ));
    }
    let trunk = runner.weights.qwen4_exp.as_ref().unwrap();
    let mut pairs = Vec::new();
    let mut continuation_position = None;
    for (&position, direct) in &snapshots[0] {
        let Some(mtp) = snapshots[1].get(&position) else {
            continue;
        };
        let consumed = position - prompt.len();
        if consumed == 0
            || consumed > runs[0].tokens.len().min(runs[1].tokens.len())
            || runs[0].tokens[..consumed] != runs[1].tokens[..consumed]
        {
            continue;
        }
        pairs.push(state_pair(
            root,
            &format!("prefix-{consumed}"),
            direct,
            mtp,
            trunk.layers.len(),
        ));
        if consumed + 1 < runs[0].tokens.len().min(runs[1].tokens.len()) {
            continuation_position = Some(position);
        }
    }
    let continuation = continuation_position.map(|position| {
        let consumed = position - prompt.len();
        let token = runs[0].tokens[consumed];
        let mut outputs = Vec::new();
        for states in &snapshots {
            outputs.push(
                crate::model::qwen4_exp::forward(
                    trunk,
                    &[token],
                    &states[&position],
                    runner.cfg.compile_cache_identity,
                    ProjectionBatchPolicy::Shared,
                )
                .unwrap(),
            );
        }
        let pair = state_pair(
            root,
            "continuation",
            &outputs[0].state,
            &outputs[1].state,
            trunk.layers.len(),
        );
        serde_json::json!({
            "from_position": position, "input_token": token,
            "direct_expected_token": runs[0].tokens[consumed + 1],
            "mtp_expected_token": runs[1].tokens[consumed + 1],
            "state": pair,
            "direct": decision(root, "continuation.direct", &outputs[0].logits),
            "mtp": decision(root, "continuation.mtp", &outputs[1].logits),
        })
    });
    let continuation_passed = continuation.as_ref().is_some_and(|c| {
        c["state"]["byte_exact"] == true
            && c["direct"]["token"] == c["direct_expected_token"]
            && c["mtp"]["token"] == c["mtp_expected_token"]
            && c["direct"]["token"] == c["mtp"]["token"]
    });
    let passed = runs.iter().all(|r| r.tokens.len() == 8)
        && runs[0].tokens == runs[1].tokens
        && runs[1].maximum("ax_mlx_flash_next_mtp_verified_steps") > 0
        && runs[1].maximum("ax_mlx_flash_next_mtp_step_errors") == 0
        && pairs.len() >= 2
        && pairs.iter().all(|p| p["byte_exact"] == true)
        && continuation_passed;
    let report = serde_json::json!({
        "schema": "ax.flash_next.runner_aligned_state.v1",
        "qualification": false, "release_ready": false,
        "mtp_certification": {"MTP-S":"not_assessed", "MTP-P":"not_assessed", "MTP-D":"not_assessed"},
        "diagnostic_passed": passed, "prompt_ids": prompt, "max_output_tokens": 8,
        "block_size_tokens": 4, "target_schedule": "canonical_singleton",
        "ple_initial_histories": trunk.layers.iter().map(|layer|
            layer.ple.as_ref().map(|ple| ple.layout.initial_history().recent().to_vec())
        ).collect::<Vec<_>>(),
        "direct_ids": runs[0].tokens, "mtp_ids": runs[1].tokens,
        "direct_routes": runs[0].routes, "mtp_routes": runs[1].routes,
        "direct_positions": snapshots[0].keys().collect::<Vec<_>>(),
        "mtp_positions": snapshots[1].keys().collect::<Vec<_>>(),
        "aligned_states": pairs, "continuation": continuation,
        "scope": "Trunk state and ordinary singleton continuation; private draft tensors and certification are not assessed",
    });
    write_blob(
        root,
        "result.json",
        &serde_json::to_vec_pretty(&report).unwrap(),
    );
    report
}

#[test]
#[ignore = "requires real Flash Next MXFP4 pack and exclusive AX_FLASH_NEXT_ALIGNED_RESULT_DIR"]
fn flash_next_real_runner_aligned_states_and_continuation() {
    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_REAL_PACK").unwrap());
    let artifacts = NativeModelArtifacts::from_dir(&root).unwrap();
    let prompt: Vec<u32> =
        serde_json::from_str(&std::env::var("AX_FLASH_NEXT_PROMPT_IDS").unwrap()).unwrap();
    let output = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_ALIGNED_RESULT_DIR").unwrap());
    let mut runner = MlxRunner::from_artifacts(&artifacts, 2048, true).unwrap();
    let report = collect(&mut runner, &prompt, &output);
    assert_eq!(
        report["diagnostic_passed"], true,
        "see retained result.json and state files"
    );
}

#[test]
#[ignore = "requires synthetic Flash Next MTP artifacts"]
fn flash_next_canonical_aligned_evidence_producer() {
    synthetic_evidence(false);
}

#[test]
#[ignore = "requires synthetic Flash Next MTP artifacts"]
fn flash_next_canonical_aligned_evidence_accepted() {
    synthetic_evidence(true);
}

fn synthetic_evidence(force_accept: bool) {
    let artifacts = artifacts();
    let mut weights = crate::weights::load_weights(&artifacts).unwrap();
    weights.qwen4_exp.as_mut().unwrap().target_schedule =
        Qwen4ExpTargetSchedule::CanonicalSingleton;
    if force_accept {
        let head = crate::weights::QuantizedWeight::new(
            mlx_sys::zeros(
                &[
                    artifacts.manifest().vocab_size as i32,
                    artifacts.manifest().hidden_size as i32,
                ],
                MlxDtype::Float32,
                None,
            ),
            None,
            None,
        );
        weights.lm_head = head.clone();
        weights.qwen4_exp.as_mut().unwrap().lm_head = head.clone();
        weights.qwen4_exp_mtp.as_mut().unwrap().graph.lm_head = head;
    }
    let shared = MlxSharedWeightsCell::new();
    shared.publish(Arc::new(weights));
    let mut runner = MlxRunner::from_artifacts_with_runtime_shares(
        &artifacts,
        2,
        true,
        true,
        None,
        Some(&shared),
    )
    .unwrap();
    let output = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_ALIGNED_RESULT_DIR").unwrap());
    let report = collect(&mut runner, &[1, 2, 3, 4, 5, 6, 7, 8, 9], &output);
    if force_accept {
        assert_eq!(report["mtp_ids"], serde_json::json!(vec![0; 8]));
        assert!(report["mtp_routes"].as_array().unwrap().iter().any(|row| {
            row.as_array().unwrap().iter().any(|gauge| {
                gauge[0] == "ax_mlx_flash_next_mtp_accepted_steps" && gauge[1].as_u64().unwrap() > 0
            })
        }));
    }
    assert_eq!(
        report["diagnostic_passed"], true,
        "see retained synthetic result.json"
    );
}
