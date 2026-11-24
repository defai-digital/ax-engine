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
    let layers = runner.weights.qwen4_exp.as_ref().unwrap().layers.len();
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
        let mut observations = Vec::new();
        runs.push(generate_observed(
            runner,
            prompt,
            usize::MAX,
            context(2100 + mode as u64, prompt.len(), 8),
            4,
            |state| {
                if state.position() >= prompt.len() {
                    observations.push(state.clone());
                }
            },
        ));
        for state in observations {
            let position = state.position();
            if let Some((previous_position, previous)) = states.last_key_value() {
                assert!(position >= *previous_position, "trunk position regressed");
                // Draining a pending direct token can emit without advancing the trunk.
                if position == *previous_position {
                    assert!(
                        flash_runner_state_bytes(previous, layers)
                            == flash_runner_state_bytes(&state, layers),
                        "trunk state changed at the same consumed position {position} in arm {mode}"
                    );
                    continue;
                }
            }
            states.insert(position, state);
        }
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
#[ignore = "requires the pinned Flash Next pack and output IDs from the signed HTTP control"]
fn flash_next_real_runner_prefill_geometry_witnesses() {
    use crate::model::qwen4_exp::profiling::gdn_geometry;
    use crate::model::qwen4_exp::profiling::{capture_geometry_forwards, take_geometry_forwards};
    use ax_engine_core::{
        CacheGroupId, DeterministicSampler, EngineCore, KvManagerConfig, ModelId,
        RequestSubmission, SamplingParams, SequenceNo,
    };

    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_REAL_PACK").unwrap());
    let prompt: Vec<u32> =
        serde_json::from_str(&std::env::var("AX_FLASH_NEXT_PROMPT_IDS").unwrap()).unwrap();
    let expected: Vec<u32> =
        serde_json::from_str(&std::env::var("AX_FLASH_NEXT_EXPECTED_IDS").unwrap()).unwrap();
    let width: usize = std::env::var("AX_FLASH_NEXT_GEOMETRY_WIDTH")
        .unwrap()
        .parse()
        .unwrap();
    let output = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_ALIGNED_RESULT_DIR").unwrap());
    assert!(matches!(width, 512 | 2048));
    assert_eq!(prompt.len(), 8192);
    assert_eq!(expected.len(), 64);
    assert!(!crate::fastpath::flash_next_qsa_grouped_eval_enabled());
    let artifacts = NativeModelArtifacts::from_dir(&root).unwrap();
    let mut runner = MlxRunner::from_artifacts_with_session_budget(
        &artifacts,
        width,
        false,
        true,
        None,
        None,
        32_768,
        Some(1),
    )
    .unwrap();
    runner.set_mtp_requested(false);
    runner.set_mtp_session_policy(MtpSessionPolicy::Auto);
    assert!(
        runner
            .weights
            .qwen4_exp
            .as_ref()
            .unwrap()
            .expert_stream
            .is_none()
    );
    let vocabulary = runner.cfg.vocab_size;
    let kv = KvManagerConfig::new(CacheGroupId(0), 16, 2048).unwrap();
    let mut core = EngineCore::with_runtime_components(kv, runner, DeterministicSampler);
    core.set_prefix_reuse_enabled(false);
    let request = core
        .submit(RequestSubmission {
            request_id: RequestId(8192),
            model_id: ModelId("qwen3.8-flash-next".into()),
            input_tokens: prompt.clone(),
            multimodal_inputs: Default::default(),
            sampling_params: SamplingParams {
                ignore_eos: true,
                ..Default::default()
            },
            max_output_tokens: expected.len() as u32,
            arrival_sequence: SequenceNo(0),
            metadata: None,
            prefill_hints: Default::default(),
        })
        .unwrap();

    let trace_gdn = crate::fastpath::env_flag("AX_FLASH_NEXT_GDN_GEOMETRY_REPLAY");
    let measure_tiling = crate::fastpath::env_flag("AX_FLASH_NEXT_RECORD_GDN_TILING");
    let tiled_wide = gdn_geometry::decay_tiling_enabled() && width == 2048;
    assert!(!tiled_wide || measure_tiling);
    if trace_gdn {
        assert!(crate::flash_next_gdn::enabled());
        gdn_geometry::begin();
    }
    capture_geometry_forwards();
    gdn_geometry::take_tiled_decay_count();
    if measure_tiling {
        mlx_sys::reset_peak_memory();
    }
    let started = std::time::Instant::now();
    let mut first_token_ns = None;
    let mut snapshot = core.request_manager().snapshot(request).unwrap();
    for _ in 0..128 {
        core.step(2048, true).unwrap();
        snapshot = core.request_manager().snapshot(request).unwrap();
        if first_token_ns.is_none() && !snapshot.generated_tokens.is_empty() {
            first_token_ns = Some(started.elapsed().as_nanos() as u64);
        }
        if snapshot.state.is_terminal() {
            break;
        }
    }
    let elapsed_ns = started.elapsed().as_nanos() as u64;
    let mlx_peak = measure_tiling.then(mlx_sys::device_peak_bytes).flatten();
    let forwards = take_geometry_forwards();
    assert_eq!(snapshot.state, ax_engine_core::RequestState::Finished);
    assert_eq!(snapshot.generated_tokens.len(), expected.len());
    if !tiled_wide {
        assert_eq!(
            snapshot.generated_tokens, expected,
            "scheduled native runner must reproduce HTTP IDs"
        );
    }
    std::fs::create_dir(&output).unwrap();
    let cuts: Vec<_> = forwards
        .iter()
        .take_while(|f| f.end <= prompt.len())
        .map(|f| [f.start, f.end])
        .collect();
    assert_eq!(cuts.first().unwrap()[0], 0);
    assert_eq!(*cuts.last().unwrap(), [8191, 8192]);
    assert!(cuts.windows(2).all(|pair| pair[0][1] == pair[1][0]));
    let widest = cuts.iter().map(|[start, end]| end - start).max().unwrap();
    assert!((width - 1..=width).contains(&widest));

    let mut witnesses = Vec::new();
    for (position, expected_id) in [
        (2047, None),
        (2048, None),
        (8192, Some(snapshot.generated_tokens[0])),
        (8193, Some(snapshot.generated_tokens[1])),
    ] {
        let Some(record) = forwards.iter().find(|f| f.end == position) else {
            assert_eq!(position, 2047);
            continue;
        };
        let state = record.state.as_ref().unwrap();
        assert_eq!(state.position(), position);
        let layer_hashes: Vec<_> = (0..state.layer_count())
            .map(|index| {
                let mut layer = Vec::new();
                state.write_layer(index, &mut layer, MlxKVCache::serialize_tensor);
                // Cache ownership is process-local. Compare the encoded tensor
                // and history payload, excluding the leading owner field.
                format!("{:x}", Sha256::digest(&layer[size_of::<u64>()..]))
            })
            .collect();
        let logits = expected_id.map(|id| {
            let array = record.logits.as_ref().unwrap();
            assert_eq!(array.shape(), [1, vocabulary as i32]);
            let row = decision(&output, &format!("position-{position}"), array);
            assert_eq!(row["token"], id);
            row
        });
        witnesses.push(serde_json::json!({
            "position": position, "layer_hashes": layer_hashes, "logits": logits,
        }));
    }
    let evidence = serde_json::json!({
        "qualification": false, "width": width, "cuts": cuts, "witnesses": witnesses,
        "output_ids": snapshot.generated_tokens,
        "stop_reason": format!("{:?}", snapshot.terminal_stop_reason),
    });
    write_blob(
        &output,
        "result.json",
        &serde_json::to_vec_pretty(&evidence).unwrap(),
    );
    if measure_tiling {
        let projections = gdn_geometry::take_tiled_decay_count();
        assert_eq!(projections > 0, tiled_wide);
        let measurement = serde_json::json!({
            "tiled_decay_projections": projections,
            "first_token_ns": first_token_ns.unwrap(),
            "total_ns": elapsed_ns,
            "mlx_peak_bytes": mlx_peak.unwrap(),
        });
        write_blob(
            &output,
            "tiling.json",
            &serde_json::to_vec_pretty(&measurement).unwrap(),
        );
    }
    if trace_gdn {
        write_blob(
            &output,
            "recurrence.json",
            &serde_json::to_vec_pretty(&gdn_geometry::finish()).unwrap(),
        );
    }
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
    let trunk = weights.qwen4_exp.as_mut().unwrap();
    trunk.target_schedule = Qwen4ExpTargetSchedule::CanonicalSingleton;
    if weights.qwen4_exp_mtp.is_none() {
        // This fixture supplies mixed floating-point and MX expert tensors.
        // Attach its head after selecting the test's canonical target route.
        let head =
            crate::weights::qwen4_exp_mtp::load(artifacts.root_dir(), artifacts.manifest(), trunk)
                .unwrap();
        weights.qwen4_exp_mtp = Some(Box::new(head));
    }
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
    let (report, audited) = crate::model::qwen4_exp_mtp::runner_audit::capture(|| {
        collect(&mut runner, &[1, 2, 3, 4, 5, 6, 7, 8, 9], &output)
    });
    assert!(!audited.is_empty());
    if force_accept {
        assert!(audited.iter().any(|step| step.accepted > 0));
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
