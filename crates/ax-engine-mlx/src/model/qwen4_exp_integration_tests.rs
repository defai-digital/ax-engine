//! Explicit artifact-backed tests for the production Flash Next forward boundary.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;
use ax_engine_core::{NativeModelArtifacts, NativeRuntimeStatus, WeightSanitize};
use std::path::{Path, PathBuf};

#[path = "qwen4_exp_mtp_diagnostics.rs"]
mod mtp_diagnostics;

fn assert_equal(a: &MlxArray, b: &MlxArray) {
    mlx_sys::eval(&[a, b]);
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.data_f32(), b.data_f32());
}

#[test]
#[ignore = "requires generated official Flash Next oracle artifacts and Metal"]
fn qwen4_exp_eval_failure_returns_error_without_committing_state() {
    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_ORACLE_DIR").unwrap());
    let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
    manifest.weight_sanitize = WeightSanitize::HfToMlx;
    manifest.runtime_status = NativeRuntimeStatus::default();
    let artifacts = NativeModelArtifacts::from_manifest_and_root(root, manifest).unwrap();
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let mut weights = crate::weights::load_weights(&artifacts).unwrap();
    let trunk = weights.qwen4_exp.as_mut().unwrap();
    let state = qwen4_exp::Qwen4ExpState::new(trunk, cfg.compile_cache_identity);
    let tokens = [1_u32, 2, 3];
    let expected = qwen4_exp::forward(
        trunk,
        &tokens,
        &state,
        cfg.compile_cache_identity,
        ProjectionBatchPolicy::Shared,
    )
    .unwrap();
    let original = trunk.lm_head.weight.clone();
    let broken = mlx_sys::MlxMetalKernel::new(
        "ax_flash_next_eval_failure_regression",
        &["input"],
        &["output"],
        "output[thread_position_in_grid.x] = ax_intentionally_undefined_symbol;",
        "",
        true,
    );
    let mut outputs = broken
        .try_apply_with_template(
            &[&original],
            &[mlx_sys::KernelOutputSpec {
                shape: original.shape(),
                dtype: original.dtype(),
            }],
            &[],
            (original.shape().iter().product(), 1, 1),
            (32, 1, 1),
            None,
        )
        .unwrap();
    trunk.lm_head.weight = outputs.remove(0);
    let failed = qwen4_exp::forward(
        trunk,
        &tokens,
        &state,
        cfg.compile_cache_identity,
        ProjectionBatchPolicy::Shared,
    );
    trunk.lm_head.weight = original;
    assert!(failed.err().unwrap().contains("evaluation failed"));
    assert_eq!(state.position(), 0);
    let recovered = qwen4_exp::forward(
        trunk,
        &tokens,
        &state,
        cfg.compile_cache_identity,
        ProjectionBatchPolicy::Shared,
    )
    .unwrap();
    assert_equal(&recovered.logits, &expected.logits);
    let encode = |state: qwen4_exp::Qwen4ExpState| {
        let mut cache = MlxKVCache::new_contiguous(cfg.layer_count);
        cache.advance(state.position());
        cache.qwen4_exp = Some(state);
        cache.serialize_to_bytes()
    };
    assert!(encode(recovered.state) == encode(expected.state));
}

fn compare_dedicated_and_production_generation(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    tokens: &[u32],
) -> Vec<u32> {
    let dedicated = weights.qwen4_exp.as_ref().unwrap();
    let initial = qwen4_exp::Qwen4ExpState::new(dedicated, cfg.compile_cache_identity);
    let whole = qwen4_exp::forward(
        dedicated,
        tokens,
        &initial,
        cfg.compile_cache_identity,
        ProjectionBatchPolicy::Shared,
    )
    .unwrap();
    // Production greedy prefill uses an n-1 cache-only prefix followed by a
    // singleton completion for this family. Compare identical MLX QMM shapes;
    // quantized whole-prompt and split-prompt arithmetic need not be bit exact.
    assert!(tokens.len() > 1);
    assert!(!crate::fastpath::skip_cache_only_split_for_family(
        &cfg.model_family,
        tokens.len()
    ));
    let prefix = qwen4_exp::forward(
        dedicated,
        &tokens[..tokens.len() - 1],
        &initial,
        cfg.compile_cache_identity,
        ProjectionBatchPolicy::Shared,
    )
    .unwrap();
    let split = qwen4_exp::forward(
        dedicated,
        &tokens[tokens.len() - 1..],
        &prefix.state,
        cfg.compile_cache_identity,
        ProjectionBatchPolicy::Shared,
    )
    .unwrap();
    let whole_logits = qwen4_exp_last_logits(cfg, &whole.logits, tokens.len());
    let split_logits = qwen4_exp_last_logits(cfg, &split.logits, 1);
    mlx_sys::eval(&[&whole_logits, &split_logits]);
    let max_error = whole_logits
        .data_f32()
        .iter()
        .zip(split_logits.data_f32())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("Flash Next whole-versus-split prefill maximum logit difference: {max_error}");
    let snapshot = |state: qwen4_exp::Qwen4ExpState| {
        let position = state.position();
        let mut cache = MlxKVCache::new_contiguous(cfg.layer_count);
        cache.qwen4_exp = Some(state);
        cache.advance(position);
        cache.serialize_to_bytes()
    };
    let expected_prefill_state = snapshot(split.state.clone());
    let mut expected = Vec::new();
    let mut expected_final_state = Vec::new();
    for (name, mut output) in [("whole", whole), ("split", split)] {
        let mut generated = Vec::new();
        for _ in 0..16 {
            let logits =
                qwen4_exp_last_logits(cfg, &output.logits, output.logits.shape()[0] as usize);
            let next = mlx_sys::argmax(&logits, None);
            mlx_sys::eval(&[&next]);
            let token = next.data_u32()[0];
            generated.push(token);
            output = qwen4_exp::forward(
                dedicated,
                &[token],
                &output.state,
                cfg.compile_cache_identity,
                ProjectionBatchPolicy::Shared,
            )
            .unwrap();
        }
        eprintln!("Flash Next dedicated {name} prefill tokens: {generated:?}");
        if name == "split" {
            expected = generated;
            expected_final_state = snapshot(output.state);
        }
    }
    let mut cache = MlxKVCache::new_contiguous(cfg.layer_count);
    let first = crate::generate::chunked_prefill(
        cfg,
        weights,
        tokens,
        &mut cache,
        tokens.len(),
        crate::sampling::MlxSamplingRequest::new(
            crate::sampling::MlxSamplingParams::greedy(),
            tokens,
        ),
        &mut crate::sampling::Xorshift64::new(0),
    );
    assert!(
        cache.serialize_to_bytes() == expected_prefill_state,
        "split prefill state must match"
    );
    let mut actual = vec![first];
    let mut pending = crate::generate::start_direct_pipeline(cfg, weights, first, &mut cache);
    for _ in 1..16 {
        let (token, next) =
            crate::generate::advance_direct_pipeline(cfg, weights, &pending, &mut cache);
        actual.push(token);
        pending = next;
    }
    mlx_sys::eval(&[&pending]);
    eprintln!("Flash Next dedicated tokens: {expected:?}; production tokens: {actual:?}");
    assert_eq!(
        actual, expected,
        "production pipeline must match the dedicated greedy trunk"
    );
    assert!(
        cache.serialize_to_bytes() == expected_final_state,
        "pipeline state must match after lookahead"
    );
    actual
}

#[test]
#[ignore = "requires a synthetic affine Flash Next expert fixture"]
fn qwen4_exp_selected_experts_preserve_hybrid_state_and_recover_io() {
    selected_experts_hybrid_state_and_recover_io(false);
}

#[test]
#[ignore = "requires a synthetic affine Flash Next expert fixture"]
fn qwen4_exp_selected_prefill_preserves_hybrid_state_and_recovers_io() {
    selected_experts_hybrid_state_and_recover_io(true);
}

fn selected_experts_hybrid_state_and_recover_io(prefill: bool) {
    use crate::expert_stream::StreamExpertsMode;
    use crate::weights::qwen4_exp::load_with_paging_policy;
    let source = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_SELECTED_FIXTURE").unwrap());
    let root = std::env::temp_dir().join(format!(
        "ax-selected-state-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(root.clone());
    for name in [
        "config.json",
        "model.safetensors",
        "experts.safetensors",
        "ax_expert_stream.json",
    ] {
        assert!(
            std::fs::metadata(source.join(name)).unwrap().len() < 16 * 1024 * 1024,
            "IO injection accepts only bounded synthetic fixtures"
        );
        std::fs::copy(source.join(name), root.join(name)).unwrap();
    }
    let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
    manifest.weight_sanitize = WeightSanitize::HfToMlx;
    manifest.runtime_status = NativeRuntimeStatus::default();
    let resident = load_with_paging_policy(&root, &manifest, StreamExpertsMode::Off, 1).unwrap();
    let mut selected = load_with_paging_policy(&root, &manifest, StreamExpertsMode::On, 1).unwrap();
    for layer in &mut selected.layers {
        layer.moe.enable_selected_decode_for_test();
        if prefill {
            layer.moe.enable_selected_prefill_for_test();
        }
    }
    let pager = selected.expert_stream.as_ref().unwrap();
    assert_eq!(pager.selected_payload_bytes_read().unwrap(), 0);
    assert_eq!(pager.cached_layer_count(), 0);
    let encode = |state: &qwen4_exp::Qwen4ExpState| {
        let mut cache = MlxKVCache::new_contiguous(manifest.layer_count as usize);
        cache.qwen4_exp = Some(state.clone());
        cache.advance(state.position());
        cache.serialize_to_bytes()
    };
    let mut control = qwen4_exp::Qwen4ExpState::new(&resident, 71);
    let mut state = qwen4_exp::Qwen4ExpState::new(&selected, 71);
    for tokens in [&[1, 2, 3, 4][..], &[5][..], &[6][..], &[7, 8][..]] {
        let bytes_before = pager.selected_payload_bytes_read().unwrap();
        let expected = qwen4_exp::forward(
            &resident,
            tokens,
            &control,
            71,
            ProjectionBatchPolicy::Shared,
        )
        .unwrap();
        let actual =
            qwen4_exp::forward(&selected, tokens, &state, 71, ProjectionBatchPolicy::Shared)
                .unwrap();
        assert_equal(&actual.logits, &expected.logits);
        assert_eq!(encode(&actual.state), encode(&expected.state));
        if prefill || tokens.len() == 1 {
            assert!(pager.selected_payload_bytes_read().unwrap() > bytes_before);
        } else {
            assert_eq!(pager.selected_payload_bytes_read().unwrap(), bytes_before);
        }
        state = actual.state;
        control = expected.state;
    }
    assert!(pager.selected_payload_bytes_read().unwrap() > 0);
    let before = encode(&state);
    let retry_tokens = if prefill { &[9, 10][..] } else { &[9][..] };
    let expected = qwen4_exp::forward(
        &resident,
        retry_tokens,
        &control,
        71,
        ProjectionBatchPolicy::Shared,
    )
    .unwrap();
    let file = root.join("experts.safetensors");
    let saved = std::fs::read(&file).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&file)
        .unwrap()
        .set_len(0)
        .unwrap();
    let error = qwen4_exp::forward(
        &selected,
        retry_tokens,
        &state,
        71,
        ProjectionBatchPolicy::Shared,
    )
    .err()
    .unwrap();
    assert!(
        error.contains("read row"),
        "unexpected selected IO failure: {error}"
    );
    assert_eq!(encode(&state), before);
    std::fs::write(file, saved).unwrap();
    let recovered = qwen4_exp::forward(
        &selected,
        retry_tokens,
        &state,
        71,
        ProjectionBatchPolicy::Shared,
    )
    .unwrap();
    assert_equal(&recovered.logits, &expected.logits);
    assert_eq!(encode(&recovered.state), encode(&expected.state));
}

#[test]
#[ignore = "requires an isolated real Flash Next candidate pack"]
fn qwen4_exp_real_pack_production_pipeline_matches_dedicated() {
    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_CANDIDATE_PACK_DIR").unwrap());
    let artifacts = NativeModelArtifacts::from_dir(&root).unwrap();
    assert_eq!(artifacts.manifest().model_family, "qwen4_exp");
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = crate::weights::load_weights(&artifacts).unwrap();
    let tokens: Vec<u32> =
        serde_json::from_str(&std::env::var("AX_FLASH_NEXT_PROMPT_IDS").unwrap()).unwrap();
    let generated = compare_dedicated_and_production_generation(&cfg, &weights, &tokens);
    if let Some(path) = std::env::var_os("AX_FLASH_NEXT_SMOKE_OUTPUT") {
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "qualification": false, "prompt_ids": tokens, "generated_ids": generated,
                "comparison": "dedicated greedy trunk versus production prefill and direct pipeline"
            }))
            .unwrap(),
        )
        .unwrap();
    }
}

#[test]
#[ignore = "requires an isolated real Flash Next pack; records same-pack residency controls"]
fn qwen4_exp_real_pack_residency_fingerprint() {
    use sha2::{Digest, Sha256};

    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_CANDIDATE_PACK_DIR").unwrap());
    let output_path = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_SMOKE_OUTPUT").unwrap());
    let tokens: Vec<u32> =
        serde_json::from_str(&std::env::var("AX_FLASH_NEXT_PROMPT_IDS").unwrap()).unwrap();
    let prefill_chunk_size = std::env::var("AX_FLASH_NEXT_PREFILL_CHUNK_SIZE")
        .ok()
        .map(|value| value.parse::<usize>().expect("prefill chunk size"));
    assert!((2..=4096).contains(&tokens.len()));
    if let Some(size) = prefill_chunk_size {
        assert!((1..=128).contains(&size));
    } else {
        assert!(
            tokens.len() <= 512,
            "long probes require bounded prefill chunks"
        );
    }
    let logits_dir = std::env::var_os("AX_FLASH_NEXT_LOGITS_DIR").map(PathBuf::from);
    if let Some(dir) = &logits_dir {
        std::fs::create_dir_all(dir).unwrap();
    }
    let dump_all_rows = std::env::var("AX_FLASH_NEXT_LOGITS_ALL_ROWS")
        .ok()
        .is_some_and(|value| value == "1");
    let save_logits = |name: &str, logits: &MlxArray| {
        if let Some(dir) = &logits_dir {
            let bytes: Vec<u8> = logits
                .data_f32()
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            std::fs::write(dir.join(format!("{name}.f32le")), bytes).unwrap();
        }
    };
    let expected_streaming = match std::env::var("AX_FLASH_NEXT_EXPECT_STREAMING")
        .unwrap()
        .as_str()
    {
        "0" => false,
        "1" => true,
        other => panic!("invalid expected streaming flag {other}"),
    };
    let artifacts = NativeModelArtifacts::from_dir_or_convert(&root).unwrap();
    assert!(
        artifacts.manifest().runtime_status.ready,
        "audited Flash Next packs must convert ready"
    );
    let teacher_force: Option<Vec<u32>> = std::env::var("AX_FLASH_NEXT_TEACHER_FORCE_IDS")
        .ok()
        .map(|value| serde_json::from_str(&value).unwrap());
    if let Some(ids) = &teacher_force {
        assert_eq!(ids.len(), 3, "expected three continuation inputs");
        assert!(ids.iter().all(|&id| id < artifacts.manifest().vocab_size));
    }
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let started = std::time::Instant::now();
    let weights = crate::weights::load_weights(&artifacts).unwrap();
    let load_seconds = started.elapsed().as_secs_f64();
    let trunk = weights.qwen4_exp.as_ref().unwrap();
    let streaming = weights.expert_stream.is_some();
    assert_eq!(streaming, expected_streaming);
    let table_bytes = || {
        trunk
            .layers
            .iter()
            .filter_map(|layer| layer.ple.as_ref())
            .map(|ple| ple.table.payload_bytes_read())
            .sum::<u64>()
    };
    assert_eq!(table_bytes(), 0);
    if let Some(pager) = &weights.expert_stream {
        assert_eq!(pager.cached_layer_count(), 0);
    }
    let active_after_load = mlx_sys::mempressure::device_active_bytes().unwrap();
    // A fixed request owner permits byte comparison across isolated processes.
    let owner = 3701;
    let initial = qwen4_exp::Qwen4ExpState::new(trunk, owner);
    let chunk_size = prefill_chunk_size.unwrap_or(tokens.len() - 1);
    let first_end = chunk_size.min(tokens.len() - 1);
    let mut prefix = qwen4_exp::forward(
        trunk,
        &tokens[..first_end],
        &initial,
        owner,
        ProjectionBatchPolicy::Shared,
    )
    .unwrap();
    let record_prefix = |index: usize, output: &qwen4_exp::Qwen4ExpOutput| {
        let shape = output.logits.shape();
        let last = mlx_sys::contiguous(
            &mlx_sys::slice(&output.logits, &[shape[0] - 1, 0], &shape, &[1, 1], None),
            None,
        );
        mlx_sys::try_eval(&[&last]).unwrap();
        save_logits(&format!("prefill-{index}-last"), &last);
        if dump_all_rows {
            mlx_sys::try_eval(&[&output.logits]).unwrap();
            save_logits(&format!("prefill-{index}-rows"), &output.logits);
            if let Some(dir) = &logits_dir {
                let chunk_len = usize::try_from(shape[0]).unwrap();
                let end = output.state.position();
                let start = end.saturating_sub(chunk_len);
                let positions: Vec<usize> = (start..end).collect();
                let sidecar = serde_json::json!({
                    "shape": shape,
                    "positions": positions,
                });
                std::fs::write(
                    dir.join(format!("prefill-{index}-rows.json")),
                    serde_json::to_vec_pretty(&sidecar).unwrap(),
                )
                .unwrap();
            }
        }
        let digest = |array: &MlxArray| {
            let mut hash = Sha256::new();
            for value in array.data_f32() {
                assert!(value.is_finite());
                hash.update(value.to_bits().to_le_bytes());
            }
            format!("{:x}", hash.finalize())
        };
        let mut cache = MlxKVCache::new_contiguous(cfg.layer_count);
        cache.advance(output.state.position());
        cache.qwen4_exp = Some(output.state.clone());
        let state = cache.serialize_to_bytes();
        serde_json::json!({
            "position": output.state.position(), "logits_shape": shape,
            "logits_f32_le_sha256": digest(&output.logits),
            "last_logits_f32_le_sha256": digest(&last),
            "state_bytes": state.len(), "state_sha256": format!("{:x}", Sha256::digest(&state)),
        })
    };
    let mut prefix_steps = Vec::new();
    if prefill_chunk_size.is_some() {
        prefix_steps.push(record_prefix(0, &prefix));
    }
    let mut consumed = first_end;
    while consumed < tokens.len() - 1 {
        let end = (consumed + chunk_size).min(tokens.len() - 1);
        prefix = qwen4_exp::forward(
            trunk,
            &tokens[consumed..end],
            &prefix.state,
            owner,
            ProjectionBatchPolicy::Shared,
        )
        .unwrap();
        prefix_steps.push(record_prefix(prefix_steps.len(), &prefix));
        consumed = end;
        eprintln!("prefill position={consumed}");
    }
    let selected_bytes_after_prefill = weights
        .expert_stream
        .as_ref()
        .map(|p| p.selected_payload_bytes_read().unwrap())
        .unwrap_or(0);
    let mut prefix_digest = Sha256::new();
    save_logits("prefix", &prefix.logits);
    for value in prefix.logits.data_f32() {
        assert!(value.is_finite());
        prefix_digest.update(value.to_bits().to_le_bytes());
    }
    let mut prefix_cache = MlxKVCache::new_contiguous(cfg.layer_count);
    prefix_cache.advance(prefix.state.position());
    prefix_cache.qwen4_exp = Some(prefix.state.clone());
    let prefix_state = prefix_cache.serialize_to_bytes();
    let prefix_record = serde_json::json!({
        "position": prefix.state.position(), "logits_shape": prefix.logits.shape(),
        "logits_f32_le_sha256": format!("{:x}", prefix_digest.finalize()),
        "state_bytes": prefix_state.len(),
        "state_sha256": format!("{:x}", Sha256::digest(&prefix_state)),
    });
    let mut output = qwen4_exp::forward(
        trunk,
        &tokens[tokens.len() - 1..],
        &prefix.state,
        owner,
        ProjectionBatchPolicy::Shared,
    )
    .unwrap();
    let mut records = Vec::new();
    let mut generated = Vec::new();
    for step in 0..4 {
        save_logits(&format!("decode-{step}"), &output.logits);
        let logits = output.logits.data_f32();
        assert!(logits.iter().all(|v| v.is_finite()));
        let mut digest = Sha256::new();
        for value in logits {
            digest.update(value.to_bits().to_le_bytes());
        }
        let mut cache = MlxKVCache::new_contiguous(cfg.layer_count);
        cache.advance(output.state.position());
        cache.qwen4_exp = Some(output.state.clone());
        let state_bytes = cache.serialize_to_bytes();
        let token = mlx_sys::argmax(&output.logits, None);
        mlx_sys::try_eval(&[&token]).unwrap();
        let token = token.data_u32()[0];
        generated.push(token);
        records.push(serde_json::json!({
            "step": step, "position": output.state.position(), "token": token,
            "logits_shape": output.logits.shape(),
            "logits_f32_le_sha256": format!("{:x}", digest.finalize()),
            "state_bytes": state_bytes.len(),
            "state_sha256": format!("{:x}", Sha256::digest(&state_bytes)),
        }));
        eprintln!("streaming={streaming} step={step} token={token}");
        if step < 3 {
            output = qwen4_exp::forward(
                trunk,
                &[teacher_force.as_ref().map_or(token, |ids| ids[step])],
                &output.state,
                owner,
                ProjectionBatchPolicy::Shared,
            )
            .unwrap();
        }
    }
    if let Some(pager) = &weights.expert_stream {
        assert!(pager.cached_layer_count() <= pager.budget_layers());
    }
    let mut result = serde_json::json!({
        "qualification": false, "family": "qwen4_exp", "prompt_ids": tokens,
        "prefill_schedule": "n-1 then singleton", "generated_ids": generated,
        "expert_streaming": streaming, "load_seconds": load_seconds,
        "cached_expert_layers_at_load": streaming.then_some(0), "table_payload_bytes_at_load": 0,
        "active_after_load": active_after_load, "peak_mlx_bytes": mlx_sys::get_peak_memory(),
        "table_payload_bytes_after_run": table_bytes(), "records": records,
        "prefix_record": prefix_record,
        "selected_expert_payload_bytes_after_prefill": selected_bytes_after_prefill,
        "selected_expert_payload_bytes": weights.expert_stream.as_ref()
            .map(|pager| pager.selected_payload_bytes_read().unwrap()).unwrap_or(0),
        "cached_expert_layers": weights.expert_stream.as_ref().map(|p| p.cached_layer_count()),
        "expert_layer_budget": weights.expert_stream.as_ref().map(|p| p.budget_layers()),
    });
    if let Some(size) = prefill_chunk_size {
        result["prefill_chunk_size"] = serde_json::json!(size);
        result["prefill_schedule"] = "chunked n-1 then singleton".into();
        result["prefix_steps"] = serde_json::json!(prefix_steps);
    }
    if let Some(ids) = teacher_force {
        result["teacher_force_ids"] = serde_json::json!(ids);
    }
    std::fs::write(output_path, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
}

#[test]
#[ignore = "requires generated official Flash Next oracle artifacts"]
fn qwen4_exp_production_cache_round_trip_and_verify_replay() {
    let root =
        PathBuf::from(std::env::var_os("AX_FLASH_NEXT_ORACLE_DIR").expect("oracle directory"));
    let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
    assert!(
        !manifest.runtime_status.ready,
        "oracle fixture has no axquant_manifest.json so layout stays unknown"
    );
    manifest.weight_sanitize = WeightSanitize::HfToMlx;
    // Only this in-memory synthetic fixture is admitted for integration tests.
    // The converter and all on-disk model manifests keep their public gate.
    manifest.runtime_status = NativeRuntimeStatus::default();
    let artifacts = NativeModelArtifacts::from_manifest_and_root(root.clone(), manifest).unwrap();
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = crate::weights::load_weights(&artifacts).unwrap();
    assert!(weights.final_norm.is_none());
    assert!(weights.layers.is_empty());
    assert!(weights.qwen4_exp.is_some());
    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("logits.json")).unwrap()).unwrap();
    let tokens: Vec<u32> = serde_json::from_value(fixture["tokens"].clone()).unwrap();
    compare_dedicated_and_production_generation(&cfg, &weights, &tokens[..4]);
    let split = 4;
    let mut prefix = MlxKVCache::new_contiguous(cfg.layer_count);
    let _ = forward(&cfg, &weights, &tokens[..split], &mut prefix, 0);
    prefix.advance(split);
    let before = prefix.serialize_to_bytes();
    let usage = prefix.usage_snapshot();
    assert!(usage.linear_state_bytes > 0);
    assert!(usage.logical_bytes > 0);
    assert_eq!(
        usage.logical_bytes + usage.linear_state_bytes,
        prefix.qwen4_exp.as_ref().unwrap().bytes() as u64
    );
    let mut restored = MlxKVCache::try_deserialize_from_bytes(&before).unwrap();
    restored
        .verify_restored_snapshot(cfg.layer_count, split, None)
        .unwrap();
    assert_eq!(before, restored.serialize_to_bytes());
    assert_eq!(usage, restored.usage_snapshot());
    assert!(!restored.trim_to(split - 1));
    assert_eq!(before, restored.serialize_to_bytes());

    let mut singleton = prefix.clone();
    let mut rows = Vec::new();
    for (index, token) in tokens[split..].iter().enumerate() {
        rows.push(forward(
            &cfg,
            &weights,
            &[*token],
            &mut singleton,
            split + index,
        ));
        singleton.advance(1);
    }
    let expected = mlx_sys::stack(&rows.iter().collect::<Vec<_>>(), 0, None);
    let verified = forward_all_positions(&cfg, &weights, &tokens[split..], &mut restored, split);
    restored.advance(tokens.len() - split);
    assert_equal(&verified, &expected);
    assert_eq!(
        singleton.serialize_to_bytes(),
        restored.serialize_to_bytes()
    );

    // Discard the verification branch, replay one accepted token, then continue.
    let mut replay = prefix.clone();
    let accepted = forward(
        &cfg,
        &weights,
        &tokens[split..split + 1],
        &mut replay,
        split,
    );
    replay.advance(1);
    assert_equal(&accepted, &rows[0]);
    let tail = forward_all_positions(&cfg, &weights, &tokens[split + 1..], &mut replay, split + 1);
    replay.advance(tokens.len() - split - 1);
    let expected_tail = mlx_sys::stack(&rows[1..].iter().collect::<Vec<_>>(), 0, None);
    assert_equal(&tail, &expected_tail);
    assert_eq!(replay.serialize_to_bytes(), singleton.serialize_to_bytes());

    // Exercise the actual n-gram acceptance/rejection entrypoint, including
    // every rejection index and a fully accepted draft.
    let seed = tokens[split];
    let mut reference = prefix.clone();
    let mut next_token = seed;
    let mut predicted = Vec::new();
    let mut checkpoints = Vec::new();
    for index in 0..4 {
        let logits = forward(&cfg, &weights, &[next_token], &mut reference, split + index);
        reference.advance(1);
        let argmax = mlx_sys::argmax(&logits, None);
        mlx_sys::eval(&[&argmax]);
        next_token = argmax.data_u32()[0];
        predicted.push(next_token);
        checkpoints.push(reference.serialize_to_bytes());
    }
    for accepted in 0..=3 {
        let mut draft = predicted[..3].to_vec();
        if accepted < 3 {
            draft[accepted] = (draft[accepted] + 1) % cfg.vocab_size as u32;
        }
        let mut branch = prefix.clone();
        let mut table = crate::ngram_accel::NgramTable::new();
        let output = crate::ngram_accel::ngram_accel_decode_step(
            &cfg,
            &weights,
            &mut branch,
            &mut table,
            seed,
            &draft,
            crate::ngram_accel::NgramDraftPolicy::majority(3, 1, 0.8),
            crate::sampling::MlxSamplingParams::greedy(),
            &[],
            &mut crate::sampling::Xorshift64::new(7),
        );
        assert_eq!(output, predicted[..accepted + 1]);
        assert_eq!(branch.serialize_to_bytes(), checkpoints[accepted]);
    }

    // A valid first row followed by an invalid ID must not publish the first row.
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        forward_all_positions(
            &cfg,
            &weights,
            &[tokens[split], u32::MAX],
            &mut prefix,
            split,
        )
    }));
    assert!(failed.is_err());
    assert_eq!(prefix.serialize_to_bytes(), before);

    // Normal use rejects foreign owner IDs; durable adoption explicitly validates
    // the model geometry before binding to this process's new identity.
    let mut foreign = prefix.clone();
    let state = foreign.qwen4_exp.as_mut().unwrap();
    state
        .rebind_for_model(
            weights.qwen4_exp.as_ref().unwrap(),
            cfg.compile_cache_identity.wrapping_add(1),
        )
        .unwrap();
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        forward(
            &cfg,
            &weights,
            &tokens[split..split + 1],
            &mut foreign,
            split,
        )
    }));
    assert!(failed.is_err());
    foreign
        .qwen4_exp
        .as_mut()
        .unwrap()
        .rebind_for_model(
            weights.qwen4_exp.as_ref().unwrap(),
            cfg.compile_cache_identity,
        )
        .unwrap();
    let rebound = forward(
        &cfg,
        &weights,
        &tokens[split..split + 1],
        &mut foreign,
        split,
    );
    assert_equal(&rebound, &rows[0]);

    // Reject truncation and mixed generic/family layer tags in the private wire.
    for length in [0, 39, before.len() - 1] {
        assert!(MlxKVCache::try_deserialize_from_bytes(&before[..length]).is_err());
    }
    let mut wrong_position = before.clone();
    wrong_position[8..16].copy_from_slice(&((split + 1) as u64).to_le_bytes());
    assert!(MlxKVCache::try_deserialize_from_bytes(&wrong_position).is_err());
    let mut mixed = before.clone();
    mixed[40] = 0;
    assert!(MlxKVCache::try_deserialize_from_bytes(&mixed).is_err());
    assert!(prefix.trim_to(0));
    assert!(prefix.qwen4_exp.is_none());
    assert_eq!(prefix.seq_len(), 0);
    assert_eq!(prefix.usage_snapshot().linear_state_bytes, 0);

    // Exercise real runner admission and its decode/prefill/pipeline warmups.
    // Keep ordinary n-gram acceleration enabled.
    let _runner = crate::runner::MlxRunner::from_artifacts(&artifacts, 4, false).unwrap();
}

#[test]
#[ignore = "requires generated official Flash Next oracle artifacts"]
fn qwen4_exp_disk_row_failure_preserves_committed_prefix() {
    let source =
        PathBuf::from(std::env::var_os("AX_FLASH_NEXT_ORACLE_DIR").expect("oracle directory"));
    let model = source.join("model.safetensors");
    assert!(
        std::fs::metadata(&model).unwrap().len() < 16 * 1024 * 1024,
        "failure injection requires the tiny oracle, never a real pack"
    );
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "ax-flash-next-row-failure-{}-{suffix}",
        std::process::id()
    ));
    std::fs::create_dir(&root).unwrap();
    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(root.clone());
    for name in ["config.json", "model.safetensors"] {
        std::fs::copy(source.join(name), root.join(name)).unwrap();
    }
    let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
    manifest.weight_sanitize = WeightSanitize::HfToMlx;
    manifest.runtime_status = NativeRuntimeStatus::default();
    let artifacts = NativeModelArtifacts::from_manifest_and_root(root.clone(), manifest).unwrap();
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = crate::weights::load_weights(&artifacts).unwrap();
    let mut cache = MlxKVCache::new_contiguous(cfg.layer_count);
    let _ = forward(&cfg, &weights, &[1, 2, 3, 4], &mut cache, 0);
    cache.advance(4);
    let checkpoint = cache.serialize_to_bytes();
    let mut control = cache.clone();
    let expected = forward_all_positions(&cfg, &weights, &[5, 6], &mut control, 4);
    control.advance(2);
    let payload = std::fs::read(root.join("model.safetensors")).unwrap();
    // Truncate only our private copy, preserving the inode held by the reader.
    // All resident weights have already evaluated; the next table gather fails.
    std::fs::OpenOptions::new()
        .write(true)
        .open(root.join("model.safetensors"))
        .unwrap()
        .set_len(0)
        .unwrap();
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        forward_all_positions(&cfg, &weights, &[5, 6], &mut cache, 4)
    }));
    assert!(failed.is_err());
    assert_eq!(cache.serialize_to_bytes(), checkpoint);
    std::fs::write(root.join("model.safetensors"), payload).unwrap();
    let resumed = forward_all_positions(&cfg, &weights, &[5, 6], &mut cache, 4);
    cache.advance(2);
    assert_equal(&resumed, &expected);
    assert_eq!(cache.serialize_to_bytes(), control.serialize_to_bytes());
}

#[test]
#[ignore = "requires generated official Flash Next oracle artifacts"]
fn qwen4_exp_paged_experts_match_resident_and_recover_io() {
    use crate::expert_stream::StreamExpertsMode;
    use crate::weights::qwen4_exp::load_with_paging_policy;
    use ax_engine_core::NativeTensorRole;
    let source = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_ORACLE_DIR").unwrap());
    assert!(
        std::fs::metadata(source.join("model.safetensors"))
            .unwrap()
            .len()
            < 16 * 1024 * 1024,
        "IO injection only accepts the tiny oracle"
    );
    let root = std::env::temp_dir().join(format!(
        "ax-flash-paged-oracle-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(root.clone());
    for name in ["config.json", "model.safetensors"] {
        std::fs::copy(source.join(name), root.join(name)).unwrap();
    }
    let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
    manifest.weight_sanitize = WeightSanitize::HfToMlx;
    manifest.runtime_status = NativeRuntimeStatus::default();
    let artifacts =
        NativeModelArtifacts::from_manifest_and_root(root.clone(), manifest.clone()).unwrap();
    let resident = load_with_paging_policy(&root, &manifest, StreamExpertsMode::Off, 1).unwrap();
    let tensors: Vec<_> = manifest
        .tensors
        .iter()
        .filter_map(|spec| {
            let proj = match spec.role {
                NativeTensorRole::FfnGateUpExpsPacked => "gate_up",
                NativeTensorRole::FfnGateExps => "gate",
                NativeTensorRole::FfnUpExps => "up",
                NativeTensorRole::FfnDownExps => "down",
                _ => return None,
            };
            Some(
                serde_json::json!({"name":spec.name,"file":spec.file,"layer":spec.layer_index,
            "proj":proj,"expert_axis":0,"num_experts":manifest.moe.expert_count,
            "bits":4,"group_size":64}),
            )
        })
        .collect();
    std::fs::write(root.join("ax_expert_stream.json"), serde_json::to_vec(&serde_json::json!({
        "schema_version":"axquant.expert-stream.v1","mode":"layer-stack","required":true,
        "num_experts":manifest.moe.expert_count,"experts_per_tok":manifest.moe.experts_per_token,
        "tensors":tensors
    })).unwrap()).unwrap();
    assert!(load_with_paging_policy(&root, &manifest, StreamExpertsMode::Off, 1).is_err());
    let streamed = load_with_paging_policy(&root, &manifest, StreamExpertsMode::Auto, 1).unwrap();
    let pager = streamed.expert_stream.as_ref().unwrap();
    assert_eq!(
        pager.cached_layer_count(),
        0,
        "load must not resolve expert payloads"
    );
    for layer in &streamed.layers {
        if let Some(ple) = &layer.ple {
            assert_eq!(ple.table.payload_bytes_read(), 0);
        }
    }
    let snapshot = |state: &qwen4_exp::Qwen4ExpState| {
        let mut cache = MlxKVCache::new_contiguous(manifest.layer_count as usize);
        cache.qwen4_exp = Some(state.clone());
        cache.advance(state.position());
        cache.serialize_to_bytes()
    };
    let mut reference = qwen4_exp::Qwen4ExpState::new(&resident, 91);
    let mut state = qwen4_exp::Qwen4ExpState::new(&streamed, 91);
    for tokens in [&[1, 2, 3, 4][..], &[5][..], &[6, 7][..]] {
        let expected = qwen4_exp::forward(
            &resident,
            tokens,
            &reference,
            91,
            ProjectionBatchPolicy::Shared,
        )
        .unwrap();
        let actual =
            qwen4_exp::forward(&streamed, tokens, &state, 91, ProjectionBatchPolicy::Shared)
                .unwrap();
        assert_equal(&actual.logits, &expected.logits);
        assert!(snapshot(&actual.state) == snapshot(&expected.state));
        assert_eq!(pager.cached_layer_count(), 1);
        assert_eq!(pager.cached_layer_indices(), [manifest.layer_count - 1]);
        state = actual.state;
        reference = expected.state;
    }
    let before = snapshot(&state);
    let expected = qwen4_exp::forward(
        &resident,
        &[8],
        &reference,
        91,
        ProjectionBatchPolicy::Shared,
    )
    .unwrap();
    // Rename only a private tiny fixture. Existing lexical-table file handles
    // remain usable, but the next uncached expert layer cannot reopen its shard.
    std::fs::rename(
        root.join("model.safetensors"),
        root.join("hidden.safetensors"),
    )
    .unwrap();
    let error = qwen4_exp::forward(&streamed, &[8], &state, 91, ProjectionBatchPolicy::Shared)
        .err()
        .expect("missing expert shard must fail");
    assert!(error.contains("paging"), "unexpected failure: {error}");
    assert!(snapshot(&state) == before);
    assert_eq!(pager.cached_layer_count(), 1);
    std::fs::rename(
        root.join("hidden.safetensors"),
        root.join("model.safetensors"),
    )
    .unwrap();
    let resumed =
        qwen4_exp::forward(&streamed, &[8], &state, 91, ProjectionBatchPolicy::Shared).unwrap();
    assert_equal(&resumed.logits, &expected.logits);
    assert!(snapshot(&resumed.state) == snapshot(&expected.state));
    let primary = crate::weights::load_weights(&artifacts).unwrap();
    assert_eq!(
        primary.expert_stream.as_ref().unwrap().cached_layer_count(),
        0
    );
    let runner = crate::runner::MlxRunner::from_artifacts(&artifacts, 4, false).unwrap();
    assert!(ax_engine_core::ExecutionRunner::native_expert_streaming_active(&runner));
}

#[test]
#[ignore = "requires an isolated real Flash Next pack and explicit expert paging"]
fn qwen4_exp_real_pack_paged_pipeline_matches_resident_record() {
    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_CANDIDATE_PACK_DIR").unwrap());
    let tokens: Vec<u32> =
        serde_json::from_str(&std::env::var("AX_FLASH_NEXT_PROMPT_IDS").unwrap()).unwrap();
    let expected: Vec<u32> =
        serde_json::from_str(&std::env::var("AX_FLASH_NEXT_EXPECTED_IDS").unwrap()).unwrap();
    assert!(
        (3..=4).contains(&expected.len()),
        "bounded smoke needs 3-4 resident control tokens"
    );
    let artifacts = NativeModelArtifacts::from_dir(&root).unwrap();
    assert_eq!(artifacts.manifest().model_family, "qwen4_exp");
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let start = std::time::Instant::now();
    let weights = crate::weights::load_weights(&artifacts).unwrap();
    let load_seconds = start.elapsed().as_secs_f64();
    let pager = weights
        .expert_stream
        .as_ref()
        .expect("explicit paging is required for this probe");
    assert_eq!(pager.cached_layer_count(), 0);
    let dedicated = weights.qwen4_exp.as_ref().unwrap();
    let table_bytes = || {
        dedicated
            .layers
            .iter()
            .filter_map(|layer| layer.ple.as_ref())
            .map(|ple| ple.table.payload_bytes_read())
            .sum::<u64>()
    };
    assert_eq!(table_bytes(), 0);
    let active_after_load = mlx_sys::mempressure::device_active_bytes().unwrap();
    let cache_after_load = mlx_sys::get_cache_memory();
    eprintln!(
        "Flash Next paged load: seconds={load_seconds}, active_bytes={active_after_load}, expert_layers=0, table_payload_bytes=0"
    );
    let start = std::time::Instant::now();
    let mut cache = MlxKVCache::new_contiguous(cfg.layer_count);
    let first = crate::generate::chunked_prefill(
        &cfg,
        &weights,
        &tokens,
        &mut cache,
        tokens.len(),
        crate::sampling::MlxSamplingRequest::new(
            crate::sampling::MlxSamplingParams::greedy(),
            &tokens,
        ),
        &mut crate::sampling::Xorshift64::new(0),
    );
    let mut generated = vec![first];
    eprintln!("Flash Next paged token: {first}");
    let mut pending = crate::generate::start_direct_pipeline(&cfg, &weights, first, &mut cache);
    for _ in 1..expected.len() {
        let (token, next) =
            crate::generate::advance_direct_pipeline(&cfg, &weights, &pending, &mut cache);
        generated.push(token);
        eprintln!("Flash Next paged token: {token}");
        pending = next;
    }
    mlx_sys::eval(&[&pending]);
    let result = serde_json::json!({
        "qualification":false,"route":"dedicated_qwen4_exp_with_expert_pager",
        "load_seconds":load_seconds,"generation_seconds":start.elapsed().as_secs_f64(),
        "prompt_ids":tokens,"expected_ids":expected,"generated_ids":generated,
        "active_after_load":active_after_load,"cache_after_load":cache_after_load,
        "active_after_run":mlx_sys::mempressure::device_active_bytes().unwrap(),"cache_after_run":mlx_sys::get_cache_memory(),
        "peak_mlx_bytes":mlx_sys::get_peak_memory(),"cached_expert_layers":pager.cached_layer_count(),
        "expert_layer_budget":pager.budget_layers(),"table_payload_bytes":table_bytes(),
    });
    if let Some(path) = std::env::var_os("AX_FLASH_NEXT_SMOKE_OUTPUT") {
        std::fs::write(path, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    }
    eprintln!("{result}");
    assert_eq!(
        generated, expected,
        "paged tokens must match the recorded resident control with identical prefill"
    );
    assert!(pager.cached_layer_count() <= pager.budget_layers());
}

fn flash_mtp_state_bytes(state: &qwen4_exp::Qwen4ExpState, layers: usize) -> Vec<u8> {
    let mut cache = MlxKVCache::new_contiguous(layers);
    cache.qwen4_exp = Some(state.clone());
    cache.advance(state.position());
    cache.serialize_to_bytes()
}

fn flash_next_mtp_greedy_token(output: &qwen4_exp::Qwen4ExpOutput) -> u32 {
    let token = mlx_sys::argmax(&output.logits, None);
    mlx_sys::eval(&[&token]);
    token.data_u32()[0]
}

fn observe_greedy_mismatch(
    identity: &mut crate::model::qwen4_exp_mtp::mtp_parity::GreedyIdentityReport,
    comparing: &mut bool,
    position: usize,
    expected: u32,
    actual: u32,
    margin: f32,
    tie_margin: f32,
) {
    use crate::model::qwen4_exp_mtp::mtp_parity;
    if !*comparing {
        return;
    }
    assert!(
        position <= identity.compared_positions,
        "comparison skipped a position"
    );
    identity.compared_positions = identity.compared_positions.max(position + 1);
    if expected == actual {
        return;
    }
    identity.greedy_identity = false;
    match mtp_parity::classify_greedy_mismatch(position, expected, actual, margin, tie_margin) {
        Ok(Some(tie)) => {
            identity.tie_divergences.push(tie);
            *comparing = false;
        }
        Ok(None) => {}
        Err(error) => panic!("{error}"),
    }
}

#[test]
fn flash_next_mtp_comparison_coverage_stops_at_first_difference() {
    use crate::model::qwen4_exp_mtp::mtp_parity::GreedyIdentityReport;
    for first_difference in [None, Some(0), Some(1), Some(15)] {
        let mut identity = GreedyIdentityReport::exact();
        let mut comparing = true;
        for position in 0..16 {
            let token = if first_difference == Some(position) {
                271
            } else {
                13
            };
            // A pending primary can be observed again when it is committed.
            for _ in 0..2 {
                observe_greedy_mismatch(
                    &mut identity,
                    &mut comparing,
                    position,
                    13,
                    token,
                    0.0,
                    0.5,
                );
            }
        }
        assert_eq!(
            identity.compared_positions,
            first_difference.map_or(16, |p| p + 1)
        );
        assert_eq!(identity.greedy_identity, first_difference.is_none());
        assert_eq!(
            identity.tie_divergences.len(),
            usize::from(first_difference.is_some())
        );
    }
}

/// Batched `verify_one` vs sequential singleton `forward` may flip argmax at a
/// documented near-tie. `sequential` is the singleton greedy token; `margin` is
/// top-two of that singleton logits row.
#[derive(Clone, Debug, PartialEq)]
struct VerifyOneGreedyDecision {
    position: &'static str,
    matched: bool,
    sequential: u32,
    batched: u32,
    margin: f32,
}

impl VerifyOneGreedyDecision {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "position": self.position,
            "matched": self.matched,
            "sequential": self.sequential,
            "batched": self.batched,
            "margin": self.margin,
        })
    }
}

fn classify_verify_one_greedy(
    batched: u32,
    sequential: u32,
    margin: f32,
    tie_margin: f32,
    position: &'static str,
) -> Result<VerifyOneGreedyDecision, String> {
    if batched != sequential && margin > tie_margin {
        return Err(format!(
            "MTP verify_one {position} greedy mismatch is not a documented near-tie: sequential={sequential} batched={batched} margin={margin} tie_margin={tie_margin}"
        ));
    }
    Ok(VerifyOneGreedyDecision {
        position,
        matched: batched == sequential,
        sequential,
        batched,
        margin,
    })
}

fn assert_verify_one_greedy_within_tie(
    batched: u32,
    sequential: u32,
    sequential_logits: &MlxArray,
    position: &'static str,
    tie_margin: f32,
) -> VerifyOneGreedyDecision {
    let margin = crate::model::qwen4_exp_mtp::top_two_margin(sequential_logits, 0).unwrap();
    classify_verify_one_greedy(batched, sequential, margin, tie_margin, position)
        .unwrap_or_else(|error| panic!("{error}"))
}

#[test]
fn qwen4_exp_mtp_verify_one_batched_greedy_tie_helper_matches_campaign_contract() {
    let tie_margin = 0.5;
    let matched = classify_verify_one_greedy(271, 271, 1.25, tie_margin, "bonus").unwrap();
    assert!(matched.matched);
    assert_eq!(matched.position, "bonus");
    let near = classify_verify_one_greedy(198, 271, 0.0, tie_margin, "bonus").unwrap();
    assert!(!near.matched);
    assert_eq!(near.sequential, 271);
    assert_eq!(near.batched, 198);
    assert_eq!(near.margin, 0.0);
    let wide = classify_verify_one_greedy(198, 271, 1.25, tie_margin, "bonus").unwrap_err();
    assert_eq!(
        wide,
        "MTP verify_one bonus greedy mismatch is not a documented near-tie: sequential=271 batched=198 margin=1.25 tie_margin=0.5"
    );
    let correction =
        classify_verify_one_greedy(561, 271, 1.25, tie_margin, "correction").unwrap_err();
    assert_eq!(
        correction,
        "MTP verify_one correction greedy mismatch is not a documented near-tie: sequential=271 batched=561 margin=1.25 tie_margin=0.5"
    );
}

fn flash_next_mtp_permute_draft_head(
    head: &mut crate::weights::qwen4_exp_mtp::Qwen4ExpMtpWeights,
) -> (bool, Option<u64>) {
    use crate::model::qwen4_exp_mtp::trained_head;
    if !trained_head::permute_head_requested() {
        return (false, None);
    }
    trained_head::permute_draft_output_projection(head, trained_head::PERMUTE_HEAD_SEED);
    (true, Some(trained_head::PERMUTE_HEAD_SEED))
}

#[test]
#[ignore = "requires synthetic Flash Next MTP artifacts or an explicit real selected-prefill pack"]
fn qwen4_exp_mtp_candidate_keeps_primary_tokens_and_state_exact() {
    use crate::model::qwen4_exp_mtp::{
        CandidateSession, head_forward, mtp_parity, top_two_margin, verify_one,
    };
    let (artifacts, tokens, mode, budget) =
        if let Some(root) = std::env::var_os("AX_FLASH_NEXT_REAL_PACK") {
            let artifacts = NativeModelArtifacts::from_dir(PathBuf::from(root)).unwrap();
            let tokens: Vec<u32> =
                serde_json::from_str(&std::env::var("AX_FLASH_NEXT_PROMPT_IDS").unwrap()).unwrap();
            assert!(
                !tokens.is_empty() && tokens.len() <= 128,
                "AX_FLASH_NEXT_PROMPT_IDS must have 1..=128 tokens, got {}",
                tokens.len()
            );
            assert_eq!(
                std::env::var("AX_MLX_FLASH_NEXT_SELECTED_EXPERTS").unwrap(),
                "1"
            );
            assert_eq!(
                std::env::var("AX_MLX_FLASH_NEXT_SELECTED_PREFILL").unwrap(),
                "1"
            );
            (
                artifacts,
                tokens,
                crate::expert_stream::StreamExpertsMode::On,
                4,
            )
        } else {
            let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_MTP_ORACLE_DIR").unwrap());
            let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
            manifest.weight_sanitize = WeightSanitize::HfToMlx;
            manifest.runtime_status = NativeRuntimeStatus::default();
            (
                NativeModelArtifacts::from_manifest_and_root(root, manifest).unwrap(),
                vec![1, 2, 3, 4, 5, 6, 7],
                crate::expert_stream::StreamExpertsMode::Off,
                12,
            )
        };
    let root = artifacts.root_dir();
    let trunk =
        crate::weights::qwen4_exp::load_with_paging_policy(root, artifacts.manifest(), mode, 1)
            .unwrap();
    let head = crate::weights::qwen4_exp_mtp::load(root, artifacts.manifest(), &trunk).unwrap();
    let owner = 901;
    let mut session = CandidateSession::prefill(&trunk, &head, &tokens, owner, owner + 1).unwrap();
    let selected_after_prefill = trunk
        .expert_stream
        .as_ref()
        .map(|pager| pager.selected_payload_bytes_read().unwrap());
    if std::env::var_os("AX_FLASH_NEXT_REAL_PACK").is_some() {
        assert!(selected_after_prefill.is_some());
    }
    if let Some(bytes) = selected_after_prefill {
        assert!(bytes > 0);
        let pager = trunk.expert_stream.as_ref().unwrap();
        {
            // Whole-layer paging keeps at most the current layer resident, so the
            // cache can hold fewer layers than the number that missed the cap.
            let cached = pager.cached_layer_count();
            let fallbacks = pager.selected_prefill_capacity_fallback_layers();
            assert!(
                cached <= fallbacks,
                "cached {cached} layers exceed {fallbacks} capacity fallbacks"
            );
            if fallbacks == 0 {
                assert_eq!(cached, 0);
            }
        }
    }
    let initial = qwen4_exp::Qwen4ExpState::new(&trunk, owner);
    let prefix = qwen4_exp::forward(
        &trunk,
        &tokens[..tokens.len() - 1],
        &initial,
        owner,
        ProjectionBatchPolicy::Shared,
    )
    .unwrap();
    let mut draft_reference = head_forward(
        &head,
        &prefix.stream_hidden,
        &tokens[1..],
        &qwen4_exp::Qwen4ExpState::new(&head.graph, owner + 1),
        owner + 1,
    )
    .unwrap()
    .state;
    let mut direct = qwen4_exp::forward(
        &trunk,
        &tokens[tokens.len() - 1..],
        &prefix.state,
        owner,
        ProjectionBatchPolicy::Shared,
    )
    .unwrap();
    let greedy = |output: &qwen4_exp::Qwen4ExpOutput| {
        let token = mlx_sys::argmax(&output.logits, None);
        mlx_sys::eval(&[&token]);
        token.data_u32()[0]
    };
    assert_eq!(session.primary, greedy(&direct));
    let before = flash_mtp_state_bytes(&session.trunk_state, trunk.layers.len());
    let draft_before = flash_mtp_state_bytes(&session.draft_state, 1);
    assert_eq!(
        before,
        flash_mtp_state_bytes(&direct.state, trunk.layers.len())
    );
    assert_eq!(draft_before, flash_mtp_state_bytes(&draft_reference, 1));
    let counters_before = (session.primary, session.proposed, session.accepted);
    assert_eq!(
        session.step(&trunk, &head, 0, &[]).unwrap_err(),
        "invalid Flash Next MTP budget or draft/trunk alignment"
    );
    assert_eq!(
        flash_mtp_state_bytes(&session.trunk_state, trunk.layers.len()),
        before
    );
    assert_eq!(flash_mtp_state_bytes(&session.draft_state, 1), draft_before);
    assert_eq!(
        (session.primary, session.proposed, session.accepted),
        counters_before
    );

    // Force both verifier outcomes independently of the synthetic head's quality.
    let primary = greedy(&direct);
    let target = qwen4_exp::forward(
        &trunk,
        &[primary],
        &direct.state,
        owner,
        ProjectionBatchPolicy::Shared,
    )
    .unwrap();
    let correct_draft = greedy(&target);
    let vocabulary = artifacts.manifest().vocab_size;
    assert!(vocabulary > 1);
    let wrong_draft = (correct_draft + 1) % vocabulary;
    let dtype = target.stream_hidden.dtype();
    let mut max_state_divergence = mtp_parity::MtpDivergence::zero();
    let mut max_logit_divergence = mtp_parity::MtpDivergence::zero();
    let mut accepted_step_arrays = Vec::new();
    let tie_margin = mtp_parity::mtp_tie_margin();
    let mut verify_one_correction = None;
    let mut verify_one_bonus = None;
    for (draft, remaining, accepted) in [
        (correct_draft, 2, true),
        (wrong_draft, 2, false),
        (correct_draft, 1, false),
    ] {
        let verified =
            verify_one(&trunk, &direct.state, owner, primary, draft, remaining, &[]).unwrap();
        assert_eq!(verified.accepted, accepted);
        assert_eq!(verified.committed.len(), if accepted { 2 } else { 1 });
        if accepted {
            let second = qwen4_exp::forward(
                &trunk,
                &[correct_draft],
                &target.state,
                owner,
                ProjectionBatchPolicy::Shared,
            )
            .unwrap();
            verify_one_correction = Some(assert_verify_one_greedy_within_tie(
                greedy(&verified.after_primary),
                greedy(&target),
                &target.logits,
                "correction",
                tie_margin,
            ));
            max_logit_divergence = max_logit_divergence.max_relative(
                mtp_parity::mtp_logits_divergence(&verified.after_primary.logits, &target.logits),
            );
            mtp_parity::assert_mtp_logits_close(
                &verified.after_primary.logits,
                &target.logits,
                dtype,
            );
            let after_draft = verified.after_draft.as_ref().unwrap();
            verify_one_bonus = Some(assert_verify_one_greedy_within_tie(
                verified.next_primary,
                greedy(&second),
                &second.logits,
                "bonus",
                tie_margin,
            ));
            max_logit_divergence = max_logit_divergence.max_relative(
                mtp_parity::mtp_logits_divergence(&after_draft.logits, &second.logits),
            );
            let records = mtp_parity::mtp_state_array_records(&after_draft.state, &second.state);
            mtp_parity::eprint_mtp_state_array_table("accepted verify_one", &records);
            max_state_divergence = max_state_divergence.max_relative(
                records
                    .iter()
                    .map(|record| record.divergence)
                    .fold(mtp_parity::MtpDivergence::zero(), |a, b| a.max_relative(b)),
            );
            if accepted_step_arrays.is_empty() {
                accepted_step_arrays = records;
            }
            mtp_parity::assert_mtp_logits_close(&after_draft.logits, &second.logits, dtype);
            mtp_parity::assert_mtp_state_close(&after_draft.state, &second.state, dtype);
        } else {
            assert!(verified.after_draft.is_none());
            assert_eq!(verified.next_primary, greedy(&target));
            assert_eq!(
                flash_mtp_state_bytes(&verified.after_primary.state, trunk.layers.len()),
                flash_mtp_state_bytes(&target.state, trunk.layers.len())
            );
        }
    }
    let terminal = verify_one(
        &trunk,
        &direct.state,
        owner,
        primary,
        correct_draft,
        2,
        &[correct_draft],
    )
    .unwrap();
    assert!(!terminal.accepted);
    assert!(terminal.after_draft.is_none());
    assert_eq!(terminal.committed, vec![primary]);
    assert_eq!(
        flash_mtp_state_bytes(&terminal.after_primary.state, trunk.layers.len()),
        flash_mtp_state_bytes(&target.state, trunk.layers.len())
    );
    assert_eq!(terminal.next_primary, correct_draft);
    assert_eq!(
        terminal.after_primary.state.position(),
        direct.state.position() + 1
    );
    let mut generated = Vec::new();
    let mut mtp_selected_decode_bytes = 0;
    let mut compared_state_steps = 0usize;
    let mut compared_generated_tokens = 0usize;
    let mut identity = mtp_parity::GreedyIdentityReport::exact();
    let mut comparing = true;
    while generated.len() < budget {
        let selected_before = trunk
            .expert_stream
            .as_ref()
            .map(|pager| pager.selected_payload_bytes_read().unwrap())
            .unwrap_or(0);
        let committed = session
            .step(&trunk, &head, budget - generated.len(), &[])
            .unwrap();
        let selected_after = trunk
            .expert_stream
            .as_ref()
            .map(|pager| pager.selected_payload_bytes_read().unwrap())
            .unwrap_or(0);
        mtp_selected_decode_bytes += selected_after - selected_before;
        assert!(!committed.is_empty() && committed.len() <= budget - generated.len());
        for token in &committed {
            if comparing {
                observe_greedy_mismatch(
                    &mut identity,
                    &mut comparing,
                    generated.len(),
                    greedy(&direct),
                    *token,
                    top_two_margin(&direct.logits, 0).unwrap(),
                    tie_margin,
                );
            }
            if comparing {
                draft_reference = head_forward(
                    &head,
                    &direct.stream_hidden,
                    &[*token],
                    &draft_reference,
                    owner + 1,
                )
                .unwrap()
                .state;
                direct = qwen4_exp::forward(
                    &trunk,
                    &[*token],
                    &direct.state,
                    owner,
                    ProjectionBatchPolicy::Shared,
                )
                .unwrap();
            }
            generated.push(*token);
        }
        if comparing {
            // Accepted steps read next_primary from the batched bonus row;
            // rejection is a singleton forward and stays bit-exact.
            if committed.len() > 1 {
                observe_greedy_mismatch(
                    &mut identity,
                    &mut comparing,
                    generated.len(),
                    greedy(&direct),
                    session.primary,
                    top_two_margin(&direct.logits, 0).unwrap(),
                    tie_margin,
                );
            } else {
                assert_eq!(session.primary, greedy(&direct));
            }
            max_state_divergence = max_state_divergence.max_relative(
                mtp_parity::mtp_state_divergence(&session.trunk_state, &direct.state),
            );
            mtp_parity::assert_mtp_state_close(&session.trunk_state, &direct.state, dtype);
            assert_eq!(
                session.draft_state.position() + 1,
                session.trunk_state.position()
            );
            max_state_divergence = max_state_divergence.max_relative(
                mtp_parity::mtp_state_divergence(&session.draft_state, &draft_reference),
            );
            mtp_parity::assert_mtp_state_close(&session.draft_state, &draft_reference, dtype);
            compared_state_steps += 1;
            compared_generated_tokens = generated.len();
        }
    }
    assert_eq!(generated.len(), budget);
    assert!(
        identity.identity_until_first_tie,
        "MTP greedy identity failed before a documented near-tie"
    );
    eprintln!(
        "Flash Next MTP control: proposed={}, accepted={}, tokens={generated:?}",
        session.proposed, session.accepted
    );
    let ulp_kinds: Vec<&str> = accepted_step_arrays
        .iter()
        .filter(|record| record.divergence.relative > 1e-5)
        .map(|record| record.kind)
        .collect();
    eprintln!("MTP accepted verify_one kinds with relative > 1e-5: {ulp_kinds:?}");
    let tolerance = mtp_parity::mtp_run_tolerance(dtype);
    let within_tolerance = max_state_divergence.relative <= tolerance.limit
        && max_logit_divergence.relative <= tolerance.limit;
    let state_arrays = mtp_parity::mtp_state_arrays_json(&accepted_step_arrays);
    if let Some(bytes) = selected_after_prefill {
        assert!(session.proposed > 0);
        assert!(mtp_selected_decode_bytes > 0);
        let pager = trunk.expert_stream.as_ref().unwrap();
        let fallback_layers = pager.selected_prefill_capacity_fallback_layers();
        let cached_layers = pager.cached_layer_count();
        // Whole-layer paging keeps at most the current layer resident.
        assert!(cached_layers <= fallback_layers);
        if fallback_layers == 0 {
            assert_eq!(cached_layers, 0);
        }
        let result = serde_json::json!({
            "qualification": false, "prompt_ids": tokens, "generated_ids": generated,
            "compared_state_steps": compared_state_steps,
            "compared_generated_tokens": compared_generated_tokens,
            "selected_payload_after_prefill": bytes,
            "selected_payload_after_run": pager.selected_payload_bytes_read().unwrap(),
            "proposed": session.proposed, "accepted": session.accepted,
            "mtp_only_selected_decode_payload_bytes": mtp_selected_decode_bytes,
            "prefill_primary_and_draft_state_exact": true,
            "draft_state_within_tolerance_each_compared_step": max_state_divergence.relative <= tolerance.limit,
            "selected_prefill_capacity_fallback_layers": fallback_layers,
            "cached_layer_count": cached_layers,
            "cached_whole_layers_after_run": cached_layers,
            "primary_state_within_tolerance_each_compared_step": max_state_divergence.relative <= tolerance.limit,
            "state_comparison_stopped_after_first_tie": !identity.greedy_identity,
            "forced_acceptance_rejection_budget_and_eos": true,
            "zero_budget_preserves_primary_and_draft_state": true,
            "logit_scale": max_logit_divergence.scale,
            "max_logit_abs_difference": max_logit_divergence.max_abs,
            "max_logit_relative_divergence": max_logit_divergence.relative,
            "max_state_abs_difference": max_state_divergence.max_abs,
            "max_state_relative_divergence": max_state_divergence.relative,
            "tie_margin": tie_margin,
            "verify_one_correction": verify_one_correction
                .as_ref()
                .map(VerifyOneGreedyDecision::to_json),
            "verify_one_bonus": verify_one_bonus
                .as_ref()
                .map(VerifyOneGreedyDecision::to_json),
            "tolerance": tolerance.limit,
            "tolerance_source": tolerance.source,
            "within_tolerance": within_tolerance,
            "greedy_identity": identity.greedy_identity,
            "identity_until_first_tie": identity.identity_until_first_tie,
            "tie_divergences": identity
                .tie_divergences
                .iter()
                .map(mtp_parity::TieDivergence::to_json)
                .collect::<Vec<_>>(),
            "state_arrays": state_arrays,
        });
        std::fs::write(
            std::env::var_os("AX_FLASH_NEXT_RESULT_PATH").unwrap(),
            serde_json::to_vec_pretty(&result).unwrap(),
        )
        .unwrap();
    } else if let Some(path) = std::env::var_os("AX_FLASH_NEXT_RESULT_PATH") {
        let result = serde_json::json!({
            "qualification": false,
            "generated_ids": generated,
            "proposed": session.proposed,
            "accepted": session.accepted,
            "logit_scale": max_logit_divergence.scale,
            "max_logit_abs_difference": max_logit_divergence.max_abs,
            "max_logit_relative_divergence": max_logit_divergence.relative,
            "max_state_abs_difference": max_state_divergence.max_abs,
            "max_state_relative_divergence": max_state_divergence.relative,
            "tie_margin": tie_margin,
            "verify_one_correction": verify_one_correction
                .as_ref()
                .map(VerifyOneGreedyDecision::to_json),
            "verify_one_bonus": verify_one_bonus
                .as_ref()
                .map(VerifyOneGreedyDecision::to_json),
            "tolerance": tolerance.limit,
            "tolerance_source": tolerance.source,
            "within_tolerance": within_tolerance,
            "greedy_identity": identity.greedy_identity,
            "identity_until_first_tie": identity.identity_until_first_tie,
            "tie_divergences": identity
                .tie_divergences
                .iter()
                .map(mtp_parity::TieDivergence::to_json)
                .collect::<Vec<_>>(),
            "state_arrays": state_arrays,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    }
}

#[test]
#[ignore = "requires synthetic Flash Next MTP candidate artifacts"]
fn qwen4_exp_mtp_candidate_io_failure_restores_both_states() {
    use crate::model::qwen4_exp_mtp::CandidateSession;
    let source = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_MTP_ORACLE_DIR").unwrap());
    let payload = std::fs::read(source.join("model.safetensors")).unwrap();
    assert!(
        payload.len() < 16 * 1024 * 1024,
        "failure injection only accepts tiny private fixtures"
    );
    let root = std::env::temp_dir().join(format!(
        "ax-flash-mtp-failure-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(root.clone());
    for name in [
        "config.json",
        "model.safetensors",
        "mtp.safetensors",
        "mtplx_runtime.json",
    ] {
        std::fs::copy(source.join(name), root.join(name)).unwrap();
    }
    let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
    manifest.weight_sanitize = WeightSanitize::HfToMlx;
    manifest.runtime_status = NativeRuntimeStatus::default();
    let artifacts = NativeModelArtifacts::from_manifest_and_root(root.clone(), manifest).unwrap();
    let trunk = crate::weights::qwen4_exp::load_with_paging_policy(
        &root,
        artifacts.manifest(),
        crate::expert_stream::StreamExpertsMode::Off,
        1,
    )
    .unwrap();
    let head = crate::weights::qwen4_exp_mtp::load(&root, artifacts.manifest(), &trunk).unwrap();
    let mut session = CandidateSession::prefill(&trunk, &head, &[1, 2, 3, 4], 1001, 1002).unwrap();
    let mut control = session.clone();
    let before = flash_mtp_state_bytes(&session.trunk_state, trunk.layers.len());
    let draft_before = flash_mtp_state_bytes(&session.draft_state, 1);
    std::fs::OpenOptions::new()
        .write(true)
        .open(root.join("model.safetensors"))
        .unwrap()
        .set_len(0)
        .unwrap();
    assert!(session.step(&trunk, &head, 2, &[]).is_err());
    assert_eq!(
        flash_mtp_state_bytes(&session.trunk_state, trunk.layers.len()),
        before
    );
    assert_eq!(flash_mtp_state_bytes(&session.draft_state, 1), draft_before);
    assert_eq!(session.primary, control.primary);
    assert_eq!((session.proposed, session.accepted), (0, 0));
    std::fs::write(root.join("model.safetensors"), payload).unwrap();
    assert_eq!(
        session.step(&trunk, &head, 2, &[]).unwrap(),
        control.step(&trunk, &head, 2, &[]).unwrap()
    );
    assert_eq!(
        flash_mtp_state_bytes(&session.trunk_state, trunk.layers.len()),
        flash_mtp_state_bytes(&control.trunk_state, trunk.layers.len())
    );
    assert_eq!(
        flash_mtp_state_bytes(&session.draft_state, 1),
        flash_mtp_state_bytes(&control.draft_state, 1)
    );
}

#[test]
#[ignore = "requires an isolated real Flash Next pack with the MTP sidecar"]
fn qwen4_exp_real_pack_mtp_candidate_matches_resident_record() {
    use crate::model::qwen4_exp_mtp::{CandidateSession, head_forward, mtp_parity, top_two_margin};
    let root = PathBuf::from(
        std::env::var_os("AX_FLASH_NEXT_CANDIDATE_PACK_DIR")
            .or_else(|| std::env::var_os("AX_FLASH_NEXT_REAL_PACK"))
            .unwrap(),
    );
    let tokens: Vec<u32> =
        serde_json::from_str(&std::env::var("AX_FLASH_NEXT_PROMPT_IDS").unwrap()).unwrap();
    let expected: Vec<u32> =
        serde_json::from_str(&std::env::var("AX_FLASH_NEXT_EXPECTED_IDS").unwrap()).unwrap();
    assert!((3..=8).contains(&expected.len()));
    let artifacts = NativeModelArtifacts::from_dir(&root).unwrap();
    let start = std::time::Instant::now();
    let trunk = crate::weights::qwen4_exp::load(&root, artifacts.manifest()).unwrap();
    let trunk_seconds = start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    let mut head =
        crate::weights::qwen4_exp_mtp::load(&root, artifacts.manifest(), &trunk).unwrap();
    let (head_permuted, permute_seed) = flash_next_mtp_permute_draft_head(&mut head);
    let head_seconds = start.elapsed().as_secs_f64();
    let terminal_ids = flash_next_mtp_load_terminal_ids(&root);
    eprintln!(
        "Flash Next MTP loaded: trunk_seconds={trunk_seconds}, head_seconds={head_seconds}, head_permuted={head_permuted}, terminal_ids={terminal_ids:?}, active_bytes={}",
        mlx_sys::mempressure::device_active_bytes().unwrap()
    );
    let owner = 1201;
    let draft_owner = 1202;
    let dtype = trunk.token_embedding.weight.dtype();
    let start = std::time::Instant::now();
    let mut session =
        CandidateSession::prefill(&trunk, &head, &tokens, owner, draft_owner).unwrap();
    let prefix = qwen4_exp::forward(
        &trunk,
        &tokens[..tokens.len() - 1],
        &qwen4_exp::Qwen4ExpState::new(&trunk, owner),
        owner,
        ProjectionBatchPolicy::Shared,
    )
    .unwrap();
    let mut draft_reference = head_forward(
        &head,
        &prefix.stream_hidden,
        &tokens[1..],
        &qwen4_exp::Qwen4ExpState::new(&head.graph, draft_owner),
        draft_owner,
    )
    .unwrap()
    .state;
    let mut direct = qwen4_exp::forward(
        &trunk,
        &tokens[tokens.len() - 1..],
        &prefix.state,
        owner,
        ProjectionBatchPolicy::Shared,
    )
    .unwrap();
    assert_eq!(session.primary, flash_next_mtp_greedy_token(&direct));
    assert_eq!(
        flash_mtp_state_bytes(&session.trunk_state, trunk.layers.len()),
        flash_mtp_state_bytes(&direct.state, trunk.layers.len())
    );
    assert_eq!(
        flash_mtp_state_bytes(&session.draft_state, 1),
        flash_mtp_state_bytes(&draft_reference, 1)
    );
    let mut generated = Vec::new();
    let mut agreement = Vec::new();
    let mut proposed = 0usize;
    let mut accepted = 0usize;
    let mut max_state_divergence = mtp_parity::MtpDivergence::zero();
    let mut last_trunk_arrays = Vec::new();
    let tie_margin = mtp_parity::mtp_tie_margin();
    let mut identity = mtp_parity::GreedyIdentityReport::exact();
    let mut comparing = true;
    let mut stopped_at_terminal = false;
    let mut terminal_position = None;
    while generated.len() < expected.len() {
        if terminal_ids.contains(&session.primary) {
            if comparing {
                observe_greedy_mismatch(
                    &mut identity,
                    &mut comparing,
                    generated.len(),
                    flash_next_mtp_greedy_token(&direct),
                    session.primary,
                    top_two_margin(&direct.logits, 0).unwrap(),
                    tie_margin,
                );
            }
            flash_next_mtp_note_terminal(
                &mut generated,
                session.primary,
                &mut stopped_at_terminal,
                &mut terminal_position,
            );
            break;
        }
        let remaining = expected.len() - generated.len();
        let proposed_before = session.proposed;
        let accepted_before = session.accepted;
        let draft_token = (remaining > 1).then(|| {
            flash_next_mtp_greedy_token(
                &head_forward(
                    &head,
                    &session.stream_hidden,
                    &[session.primary],
                    &session.draft_state,
                    draft_owner,
                )
                .unwrap(),
            )
        });
        let committed = session
            .step(&trunk, &head, remaining, &terminal_ids)
            .unwrap();
        eprintln!("Flash Next MTP committed: {committed:?}");
        assert!(!committed.is_empty());
        flash_next_mtp_note_proposal_agreement(
            &mut agreement,
            &mut proposed,
            &mut accepted,
            (proposed_before, accepted_before),
            (session.proposed, session.accepted),
            draft_token.map(|draft| {
                flash_next_mtp_draft_matches_committed(draft, &committed, session.primary)
            }),
        );
        for token in &committed {
            if comparing {
                observe_greedy_mismatch(
                    &mut identity,
                    &mut comparing,
                    generated.len(),
                    flash_next_mtp_greedy_token(&direct),
                    *token,
                    top_two_margin(&direct.logits, 0).unwrap(),
                    tie_margin,
                );
            }
            let is_terminal = terminal_ids.contains(token);
            if comparing && !is_terminal {
                draft_reference = head_forward(
                    &head,
                    &direct.stream_hidden,
                    &[*token],
                    &draft_reference,
                    draft_owner,
                )
                .unwrap()
                .state;
                direct = qwen4_exp::forward(
                    &trunk,
                    &[*token],
                    &direct.state,
                    owner,
                    ProjectionBatchPolicy::Shared,
                )
                .unwrap();
            }
            generated.push(*token);
            if is_terminal {
                stopped_at_terminal = true;
                terminal_position = Some(generated.len() - 1);
                break;
            }
        }
        if stopped_at_terminal {
            break;
        }
        if comparing {
            assert_eq!(session.primary, flash_next_mtp_greedy_token(&direct));
            last_trunk_arrays =
                mtp_parity::mtp_state_array_records(&session.trunk_state, &direct.state);
            max_state_divergence = max_state_divergence.max_relative(
                last_trunk_arrays
                    .iter()
                    .map(|record| record.divergence)
                    .fold(mtp_parity::MtpDivergence::zero(), |a, b| a.max_relative(b)),
            );
            mtp_parity::assert_mtp_state_close(&session.trunk_state, &direct.state, dtype);
            max_state_divergence = max_state_divergence.max_relative(
                mtp_parity::mtp_state_divergence(&session.draft_state, &draft_reference),
            );
            mtp_parity::assert_mtp_state_close(&session.draft_state, &draft_reference, dtype);
        }
    }
    let (expected_cut, _, _) = flash_next_mtp_truncate_at_terminal(&expected, &terminal_ids);
    let tolerance = mtp_parity::mtp_run_tolerance(dtype);
    if generated != expected_cut {
        assert!(
            !identity.tie_divergences.is_empty(),
            "generated_ids differ from expected_ids without a documented near-tie: generated={generated:?} expected={expected_cut:?}"
        );
    }
    let greedy_identity = identity.greedy_identity && generated == expected_cut;
    let within_tolerance =
        max_state_divergence.relative <= tolerance.limit && identity.identity_until_first_tie;
    let agreement_rate = crate::model::qwen4_exp_mtp::trained_head::agreement_rate(&agreement);
    assert_eq!(agreement.len(), proposed);
    assert_eq!(agreement.iter().filter(|agreed| **agreed).count(), accepted);
    let result = serde_json::json!({
        "qualification":false,"route":"flash_next_mtp_candidate_sequential_primary_verify",
        "trunk_load_seconds":trunk_seconds,"head_load_seconds":head_seconds,
        "generation_seconds":start.elapsed().as_secs_f64(),"prompt_ids":tokens,
        "expected_ids":expected,"generated_ids":generated.clone(),"greedy_tokens":generated.clone(),
        "proposed":proposed,
        "accepted":accepted,
        "head_permuted":head_permuted,
        "permute_seed":permute_seed,
        "draft_vs_primary_top1_agreement":agreement,
        "draft_vs_primary_top1_agreement_rate":agreement_rate,
        "peak_mlx_bytes":mlx_sys::get_peak_memory(),
        "active_bytes":mlx_sys::mempressure::device_active_bytes().unwrap(),
        "trunk_position":session.trunk_state.position(),"draft_position":session.draft_state.position(),
        "prefill_primary_and_draft_state_exact": true,
        "greedy_identity": greedy_identity,
        "identity_until_first_tie": identity.identity_until_first_tie,
        "tie_divergences": identity
            .tie_divergences
            .iter()
            .map(mtp_parity::TieDivergence::to_json)
            .collect::<Vec<_>>(),
        "logit_scale": serde_json::Value::Null,
        "max_logit_abs_difference": serde_json::Value::Null,
        "max_logit_relative_divergence": serde_json::Value::Null,
        "max_state_abs_difference": max_state_divergence.max_abs,
        "max_state_relative_divergence": max_state_divergence.relative,
        "tolerance": tolerance.limit,
        "tolerance_source": tolerance.source,
        "within_tolerance": within_tolerance,
        "stopped_at_terminal": stopped_at_terminal,
        "terminal_position": terminal_position,
        "compared_positions": identity.compared_positions,
        "terminal_ids": terminal_ids,
        "state_arrays": mtp_parity::mtp_state_arrays_json(&last_trunk_arrays),
        "note":"Reconstructed candidate head; primary verification is authoritative; no throughput claim"
    });
    if let Some(path) = std::env::var_os("AX_FLASH_NEXT_SMOKE_OUTPUT")
        .or_else(|| std::env::var_os("AX_FLASH_NEXT_RESULT_PATH"))
    {
        std::fs::write(path, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    }
    eprintln!("{result}");
    assert!(
        identity.identity_until_first_tie,
        "MTP greedy identity failed before a documented near-tie"
    );
    assert!(within_tolerance);
    assert!(proposed > 0);
    assert_eq!(
        session.draft_state.position() + 1,
        session.trunk_state.position()
    );
}

#[derive(Clone, Debug, serde::Deserialize)]
struct FlashNextMtpOraclePrompt {
    #[serde(default)]
    id: Option<String>,
    #[serde(alias = "input_ids")]
    prompt_ids: Vec<u32>,
    #[serde(default)]
    expected_ids: Option<Vec<u32>>,
}

#[derive(Debug, serde::Deserialize)]
struct FlashNextMtpOracleManifest {
    #[serde(default)]
    max_new_tokens: Option<usize>,
    requests: Vec<FlashNextMtpOraclePrompt>,
}

const FLASH_NEXT_TERMINAL_IDS_ENV: &str = "AX_FLASH_NEXT_TERMINAL_IDS";

struct FlashNextMtpOracleRun {
    generated: Vec<u32>,
    agreement: Vec<bool>,
    proposed: usize,
    accepted: usize,
    identity: crate::model::qwen4_exp_mtp::mtp_parity::GreedyIdentityReport,
    stopped_at_terminal: bool,
    terminal_position: Option<usize>,
    compared_positions: usize,
    too_short: bool,
}

fn flash_next_mtp_push_unique_id(ids: &mut Vec<u32>, value: u32) {
    if !ids.contains(&value) {
        ids.push(value);
    }
}

fn flash_next_mtp_collect_token_ids(value: &serde_json::Value, ids: &mut Vec<u32>) {
    match value {
        serde_json::Value::Number(number) => {
            if let Some(id) = number.as_u64().and_then(|raw| u32::try_from(raw).ok()) {
                flash_next_mtp_push_unique_id(ids, id);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                flash_next_mtp_collect_token_ids(item, ids);
            }
        }
        serde_json::Value::Object(object) => {
            if let Some(id) = object.get("id") {
                flash_next_mtp_collect_token_ids(id, ids);
            }
        }
        _ => {}
    }
}

fn flash_next_mtp_ids_from_eos_fields(value: &serde_json::Value) -> Vec<u32> {
    let mut ids = Vec::new();
    for key in ["eos_token_id", "eos_token_ids"] {
        if let Some(field) = value.get(key) {
            flash_next_mtp_collect_token_ids(field, &mut ids);
        }
    }
    if let Some(nested) = value.get("generation_config") {
        for id in flash_next_mtp_ids_from_eos_fields(nested) {
            flash_next_mtp_push_unique_id(&mut ids, id);
        }
    }
    ids
}

fn flash_next_mtp_parse_terminal_ids(raw: &str) -> Vec<u32> {
    let parsed: Vec<u32> = serde_json::from_str(raw).unwrap();
    assert!(
        !parsed.is_empty(),
        "{FLASH_NEXT_TERMINAL_IDS_ENV} must be a non-empty JSON array of token ids"
    );
    parsed
}

fn flash_next_mtp_terminal_ids_from_pack(root: &Path) -> Vec<u32> {
    let mut ids = Vec::new();
    for name in [
        "generation_config.json",
        "tokenizer_config.json",
        "config.json",
    ] {
        let path = root.join(name);
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        for id in flash_next_mtp_ids_from_eos_fields(&value) {
            flash_next_mtp_push_unique_id(&mut ids, id);
        }
    }
    ids
}

fn flash_next_mtp_load_terminal_ids(root: &Path) -> Vec<u32> {
    if let Ok(raw) = std::env::var(FLASH_NEXT_TERMINAL_IDS_ENV) {
        return flash_next_mtp_parse_terminal_ids(&raw);
    }
    let ids = flash_next_mtp_terminal_ids_from_pack(root);
    assert!(
        !ids.is_empty(),
        "Flash Next pack is missing eos_token_id in generation_config.json / tokenizer_config.json"
    );
    ids
}

fn flash_next_mtp_truncate_at_terminal(
    tokens: &[u32],
    terminal_ids: &[u32],
) -> (Vec<u32>, bool, Option<usize>) {
    match tokens.iter().position(|token| terminal_ids.contains(token)) {
        Some(index) => (tokens[..=index].to_vec(), true, Some(index)),
        None => (tokens.to_vec(), false, None),
    }
}

fn flash_next_mtp_too_short(generated: &[u32], terminal_ids: &[u32]) -> bool {
    match generated
        .iter()
        .position(|token| terminal_ids.contains(token))
    {
        Some(index) => index < 2,
        None => generated.len() < 2,
    }
}

fn flash_next_mtp_note_terminal(
    generated: &mut Vec<u32>,
    token: u32,
    stopped_at_terminal: &mut bool,
    terminal_position: &mut Option<usize>,
) {
    generated.push(token);
    *stopped_at_terminal = true;
    *terminal_position = Some(generated.len() - 1);
}

/// CandidateSession does not propose on every loop step: the final-budget
/// path (`remaining == 1`) and a terminal stop skip the proposal. Count an
/// agreement sample only when `proposed` increased during that step.
fn flash_next_mtp_note_proposal_agreement(
    agreement: &mut Vec<bool>,
    proposed: &mut usize,
    accepted: &mut usize,
    before: (usize, usize),
    after: (usize, usize),
    draft_matches: Option<bool>,
) {
    let (proposed_before, accepted_before) = before;
    let (proposed_after, accepted_after) = after;
    *proposed += proposed_after.saturating_sub(proposed_before);
    *accepted += accepted_after.saturating_sub(accepted_before);
    if proposed_after > proposed_before {
        agreement.push(
            draft_matches.expect("Flash Next MTP proposal requires a pre-step draft comparison"),
        );
    }
}

fn flash_next_mtp_draft_matches_committed(
    draft_token: u32,
    committed: &[u32],
    primary: u32,
) -> bool {
    let primary_next = if committed.len() >= 2 {
        committed[1]
    } else {
        primary
    };
    draft_token == primary_next
}

fn flash_next_mtp_oracle_prompts() -> (Vec<FlashNextMtpOraclePrompt>, usize) {
    if let Some(path) = std::env::var_os("AX_FLASH_NEXT_PROMPT_MANIFEST") {
        let parsed: FlashNextMtpOracleManifest =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert!(!parsed.requests.is_empty());
        let max_new = parsed.max_new_tokens.unwrap_or(16);
        assert!((3..=32).contains(&max_new));
        return (parsed.requests, max_new);
    }
    let tokens: Vec<u32> =
        serde_json::from_str(&std::env::var("AX_FLASH_NEXT_PROMPT_IDS").unwrap()).unwrap();
    let expected: Option<Vec<u32>> = std::env::var("AX_FLASH_NEXT_EXPECTED_IDS")
        .ok()
        .map(|raw| serde_json::from_str(&raw).unwrap());
    let max_new = std::env::var("AX_FLASH_NEXT_MAX_NEW_TOKENS")
        .ok()
        .map(|raw| raw.parse().unwrap())
        .or_else(|| expected.as_ref().map(Vec::len))
        .unwrap_or(16);
    assert!((3..=32).contains(&max_new));
    (
        vec![FlashNextMtpOraclePrompt {
            id: Some("env".into()),
            prompt_ids: tokens,
            expected_ids: expected,
        }],
        max_new,
    )
}

fn flash_next_mtp_oracle_generate(
    trunk: &crate::weights::qwen4_exp::Qwen4ExpWeights,
    head: &crate::weights::qwen4_exp_mtp::Qwen4ExpMtpWeights,
    tokens: &[u32],
    max_new: usize,
    owner: u64,
    draft_owner: u64,
    terminal_ids: &[u32],
) -> FlashNextMtpOracleRun {
    use crate::model::qwen4_exp_mtp::{CandidateSession, head_forward, mtp_parity, top_two_margin};
    assert!(!tokens.is_empty());
    let mut session = CandidateSession::prefill(trunk, head, tokens, owner, draft_owner).unwrap();
    let prefix_state = if tokens.len() > 1 {
        qwen4_exp::forward(
            trunk,
            &tokens[..tokens.len() - 1],
            &qwen4_exp::Qwen4ExpState::new(trunk, owner),
            owner,
            ProjectionBatchPolicy::Shared,
        )
        .unwrap()
        .state
    } else {
        qwen4_exp::Qwen4ExpState::new(trunk, owner)
    };
    let mut direct = qwen4_exp::forward(
        trunk,
        &tokens[tokens.len() - 1..],
        &prefix_state,
        owner,
        ProjectionBatchPolicy::Shared,
    )
    .unwrap();
    assert_eq!(session.primary, flash_next_mtp_greedy_token(&direct));
    let mut generated = Vec::new();
    let mut agreement = Vec::new();
    let mut proposed = 0usize;
    let mut accepted = 0usize;
    let tie_margin = mtp_parity::mtp_tie_margin();
    let mut identity = mtp_parity::GreedyIdentityReport::exact();
    let mut comparing = true;
    let mut stopped_at_terminal = false;
    let mut terminal_position = None;
    while generated.len() < max_new {
        if terminal_ids.contains(&session.primary) {
            if comparing {
                observe_greedy_mismatch(
                    &mut identity,
                    &mut comparing,
                    generated.len(),
                    flash_next_mtp_greedy_token(&direct),
                    session.primary,
                    top_two_margin(&direct.logits, 0).unwrap(),
                    tie_margin,
                );
            }
            flash_next_mtp_note_terminal(
                &mut generated,
                session.primary,
                &mut stopped_at_terminal,
                &mut terminal_position,
            );
            break;
        }
        let remaining = max_new - generated.len();
        let proposed_before = session.proposed;
        let accepted_before = session.accepted;
        let draft_token = (remaining > 1).then(|| {
            flash_next_mtp_greedy_token(
                &head_forward(
                    head,
                    &session.stream_hidden,
                    &[session.primary],
                    &session.draft_state,
                    draft_owner,
                )
                .unwrap(),
            )
        });
        let committed = session.step(trunk, head, remaining, terminal_ids).unwrap();
        assert!(!committed.is_empty() && committed.len() <= remaining);
        flash_next_mtp_note_proposal_agreement(
            &mut agreement,
            &mut proposed,
            &mut accepted,
            (proposed_before, accepted_before),
            (session.proposed, session.accepted),
            draft_token.map(|draft| {
                flash_next_mtp_draft_matches_committed(draft, &committed, session.primary)
            }),
        );
        for token in &committed {
            if comparing {
                observe_greedy_mismatch(
                    &mut identity,
                    &mut comparing,
                    generated.len(),
                    flash_next_mtp_greedy_token(&direct),
                    *token,
                    top_two_margin(&direct.logits, 0).unwrap(),
                    tie_margin,
                );
            }
            let is_terminal = terminal_ids.contains(token);
            if comparing && !is_terminal {
                direct = qwen4_exp::forward(
                    trunk,
                    &[*token],
                    &direct.state,
                    owner,
                    ProjectionBatchPolicy::Shared,
                )
                .unwrap();
            }
            generated.push(*token);
            if is_terminal {
                stopped_at_terminal = true;
                terminal_position = Some(generated.len() - 1);
                break;
            }
        }
        if stopped_at_terminal {
            break;
        }
        if comparing && generated.len() < max_new {
            observe_greedy_mismatch(
                &mut identity,
                &mut comparing,
                generated.len(),
                flash_next_mtp_greedy_token(&direct),
                session.primary,
                top_two_margin(&direct.logits, 0).unwrap(),
                tie_margin,
            );
        }
    }
    // The session counters are authoritative; the pre-step draft comparison
    // is recorded next to them instead of asserted equal (the M2 real-pack run
    // showed 3 samples against 2 session proposals on one request).
    let agreement_samples = agreement.len();
    let agreement_matches = agreement.iter().filter(|agreed| **agreed).count();
    eprintln!(
        "Flash Next MTP oracle request: session_proposed={proposed} session_accepted={accepted} agreement_samples={agreement_samples} agreement_matches={agreement_matches}"
    );
    FlashNextMtpOracleRun {
        too_short: flash_next_mtp_too_short(&generated, terminal_ids),
        compared_positions: identity.compared_positions,
        generated,
        agreement,
        proposed,
        accepted,
        identity,
        stopped_at_terminal,
        terminal_position,
    }
}

#[test]
fn flash_next_mtp_oracle_does_not_compare_past_im_end() {
    const IM_END: u32 = 248046;
    const END_OF_TEXT: u32 = 248044;
    const IM_START: u32 = 248045;
    const USER: u32 = 846;
    const NEWLINE: u32 = 198;
    let terminal = [IM_END, END_OF_TEXT];
    // M2 4-bit chat-template failure: after <|im_end|>, direct continued
    // with "\n", "<|im_start|>", "user" while MTP continued from a rejected
    // terminal draft. Those tokens are past the product stop.
    let direct = vec![1, 2, 3, 4, 5, IM_END, NEWLINE, IM_START, USER];
    let mtp = vec![1, 2, 3, 4, 5, IM_END, IM_START, USER];
    let (direct_cut, direct_stopped, direct_at) =
        flash_next_mtp_truncate_at_terminal(&direct, &terminal);
    let (mtp_cut, mtp_stopped, mtp_at) = flash_next_mtp_truncate_at_terminal(&mtp, &terminal);
    assert_eq!(direct_cut, vec![1, 2, 3, 4, 5, IM_END]);
    assert_eq!(mtp_cut, direct_cut);
    assert!(direct_stopped && mtp_stopped);
    assert_eq!(direct_at, Some(5));
    assert_eq!(mtp_at, Some(5));
    assert!(!flash_next_mtp_too_short(&direct_cut, &terminal));
    assert_eq!(direct_cut.len(), 6);
}

#[test]
fn flash_next_mtp_oracle_marks_too_short_before_eos_and_parses_pack_ids() {
    const IM_END: u32 = 248046;
    const END_OF_TEXT: u32 = 248044;
    let terminal = [IM_END, END_OF_TEXT];
    assert!(flash_next_mtp_too_short(&[IM_END], &terminal));
    assert!(flash_next_mtp_too_short(&[846, IM_END], &terminal));
    assert!(!flash_next_mtp_too_short(&[1, 2, IM_END], &terminal));
    assert!(!flash_next_mtp_too_short(&[1, 2, 3], &terminal));
    assert!(flash_next_mtp_too_short(&[1], &terminal));
    assert_eq!(
        flash_next_mtp_parse_terminal_ids("[248046, 248044]"),
        vec![IM_END, END_OF_TEXT]
    );
    let root = std::env::temp_dir().join(format!(
        "ax_flash_next_mtp_terminal_ids_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("generation_config.json"),
        r#"{"eos_token_id":[248046,248044],"pad_token_id":248044}"#,
    )
    .unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        r#"{"eos_token":"<|im_end|>","pad_token":"<|endoftext|>"}"#,
    )
    .unwrap();
    assert_eq!(
        flash_next_mtp_terminal_ids_from_pack(&root),
        vec![IM_END, END_OF_TEXT]
    );
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn flash_next_mtp_oracle_counts_proposals_only_when_session_proposes() {
    let mut agreement = Vec::new();
    let mut proposed = 0usize;
    let mut accepted = 0usize;
    // Final-budget / remaining == 1: the session does not propose.
    flash_next_mtp_note_proposal_agreement(
        &mut agreement,
        &mut proposed,
        &mut accepted,
        (0, 0),
        (0, 0),
        None,
    );
    // Two real proposals, then a terminal stop that does not propose.
    flash_next_mtp_note_proposal_agreement(
        &mut agreement,
        &mut proposed,
        &mut accepted,
        (0, 0),
        (1, 0),
        Some(false),
    );
    flash_next_mtp_note_proposal_agreement(
        &mut agreement,
        &mut proposed,
        &mut accepted,
        (1, 0),
        (2, 1),
        Some(true),
    );
    flash_next_mtp_note_proposal_agreement(
        &mut agreement,
        &mut proposed,
        &mut accepted,
        (2, 1),
        (2, 1),
        None,
    );
    assert_eq!(agreement, vec![false, true]);
    assert_eq!(agreement.len(), proposed);
    assert_eq!(agreement.iter().filter(|agreed| **agreed).count(), accepted);

    // Too-short: first token is already terminal, so there is no proposal.
    let mut too_short_agreement = Vec::new();
    let mut too_short_proposed = 0usize;
    let mut too_short_accepted = 0usize;
    flash_next_mtp_note_proposal_agreement(
        &mut too_short_agreement,
        &mut too_short_proposed,
        &mut too_short_accepted,
        (0, 0),
        (0, 0),
        None,
    );
    assert!(flash_next_mtp_too_short(&[248046], &[248046, 248044]));
    assert_eq!(too_short_agreement.len(), too_short_proposed);
    assert_eq!((too_short_proposed, too_short_accepted), (0, 0));

    // Stopped-at-terminal inside the budget still counts only real proposals.
    let stopped = vec![1, 2, 248046];
    assert!(!flash_next_mtp_too_short(&stopped, &[248046, 248044]));
    assert_eq!(agreement.len(), proposed);
}

#[test]
#[ignore = "requires an isolated real Flash Next pack with the MTP sidecar"]
fn qwen4_exp_real_pack_mtp_trained_head_oracle() {
    use crate::model::qwen4_exp_mtp::trained_head;
    let root = PathBuf::from(
        std::env::var_os("AX_FLASH_NEXT_CANDIDATE_PACK_DIR")
            .or_else(|| std::env::var_os("AX_FLASH_NEXT_REAL_PACK"))
            .unwrap(),
    );
    let (prompts, default_max_new) = flash_next_mtp_oracle_prompts();
    let artifacts = NativeModelArtifacts::from_dir(&root).unwrap();
    let terminal_ids = flash_next_mtp_load_terminal_ids(&root);
    let start = std::time::Instant::now();
    let trunk = crate::weights::qwen4_exp::load(&root, artifacts.manifest()).unwrap();
    let trunk_seconds = start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    let mut head =
        crate::weights::qwen4_exp_mtp::load(&root, artifacts.manifest(), &trunk).unwrap();
    let (head_permuted, permute_seed) = flash_next_mtp_permute_draft_head(&mut head);
    let head_seconds = start.elapsed().as_secs_f64();
    let mut requests = Vec::new();
    let mut proposed = 0usize;
    let mut accepted = 0usize;
    let mut agreement_observations = Vec::new();
    let mut too_short_requests = 0usize;
    let mut greedy_identity = true;
    let mut identity_until_first_tie = true;
    let mut tie_divergences = Vec::new();
    for (index, prompt) in prompts.iter().enumerate() {
        let max_new = prompt
            .expected_ids
            .as_ref()
            .map(Vec::len)
            .unwrap_or(default_max_new);
        assert!((3..=32).contains(&max_new));
        let owner = 2200 + (index as u64) * 2;
        let run = flash_next_mtp_oracle_generate(
            &trunk,
            &head,
            &prompt.prompt_ids,
            max_new,
            owner,
            owner + 1,
            &terminal_ids,
        );
        if let Some(expected) = &prompt.expected_ids {
            let (expected_cut, _, _) = flash_next_mtp_truncate_at_terminal(expected, &terminal_ids);
            if run.generated != expected_cut {
                assert!(
                    !run.identity.tie_divergences.is_empty(),
                    "generated_ids differ from expected_ids without a documented near-tie"
                );
            }
        }
        greedy_identity &= run.identity.greedy_identity;
        identity_until_first_tie &= run.identity.identity_until_first_tie;
        tie_divergences.extend(run.identity.tie_divergences.iter().cloned());
        if run.too_short {
            too_short_requests += 1;
        } else {
            proposed += run.proposed;
            accepted += run.accepted;
            agreement_observations.extend_from_slice(&run.agreement);
        }
        requests.push(serde_json::json!({
            "id": prompt.id.clone().unwrap_or_else(|| format!("request-{index}")),
            "prompt_ids": prompt.prompt_ids,
            "expected_ids": prompt.expected_ids,
            "greedy_tokens": run.generated.clone(),
            "generated_ids": run.generated,
            "proposed": run.proposed,
            "accepted": run.accepted,
            "stopped_at_terminal": run.stopped_at_terminal,
            "terminal_position": run.terminal_position,
            "compared_positions": run.compared_positions,
            "too_short": run.too_short,
            "draft_vs_primary_top1_agreement": run.agreement,
            "draft_vs_primary_top1_agreement_rate": trained_head::agreement_rate(&run.agreement),
            "greedy_identity": run.identity.greedy_identity,
            "identity_until_first_tie": run.identity.identity_until_first_tie,
            "tie_divergences": run.identity
                .tie_divergences
                .iter()
                .map(crate::model::qwen4_exp_mtp::mtp_parity::TieDivergence::to_json)
                .collect::<Vec<_>>(),
        }));
    }
    let acceptance_rate = if proposed == 0 {
        0.0
    } else {
        accepted as f64 / proposed as f64
    };
    let agreement_rate = trained_head::agreement_rate(&agreement_observations);
    let agreement_samples = agreement_observations.len();
    let agreed = agreement_observations.iter().filter(|step| **step).count();
    let result = serde_json::json!({
        "qualification": false,
        "route": "flash_next_mtp_trained_head_oracle",
        "head_permuted": head_permuted,
        "permute_seed": permute_seed,
        "trunk_load_seconds": trunk_seconds,
        "head_load_seconds": head_seconds,
        "proposed": proposed,
        "accepted": accepted,
        "acceptance_rate": acceptance_rate,
        "draft_vs_primary_top1_agreement_rate": agreement_rate,
        "draft_vs_primary_top1_agreement_samples": agreement_samples,
        "draft_vs_primary_top1_agreement_matches": agreed,
        "too_short_requests": too_short_requests,
        "terminal_ids": terminal_ids,
        "greedy_identity": greedy_identity,
        "identity_until_first_tie": identity_until_first_tie,
        "tie_divergences": tie_divergences
            .iter()
            .map(crate::model::qwen4_exp_mtp::mtp_parity::TieDivergence::to_json)
            .collect::<Vec<_>>(),
        "requests": requests,
        "note": "Trained-head oracle; primary verification is authoritative; no throughput claim"
    });
    if let Some(path) = std::env::var_os("AX_FLASH_NEXT_SMOKE_OUTPUT")
        .or_else(|| std::env::var_os("AX_FLASH_NEXT_RESULT_PATH"))
    {
        std::fs::write(path, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    }
    eprintln!("{result}");
    assert!(
        identity_until_first_tie,
        "MTP greedy identity failed before a documented near-tie"
    );
    assert!(proposed > 0);
}

#[test]
#[ignore = "requires synthetic Flash Next MTP candidate artifacts"]
fn qwen4_exp_mtp_candidate_accepts_with_exact_draft_history_and_budget() {
    use crate::model::qwen4_exp_mtp::{CandidateSession, head_forward, mtp_parity};
    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_MTP_ORACLE_DIR").unwrap());
    let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
    manifest.weight_sanitize = WeightSanitize::HfToMlx;
    manifest.runtime_status = NativeRuntimeStatus::default();
    let artifacts = NativeModelArtifacts::from_manifest_and_root(root.clone(), manifest).unwrap();
    let mut trunk = crate::weights::qwen4_exp::load_with_paging_policy(
        &root,
        artifacts.manifest(),
        crate::expert_stream::StreamExpertsMode::Off,
        1,
    )
    .unwrap();
    // Deterministic primary and draft output heads force real acceptance without
    // changing the stateful layers or bypassing the candidate's verifier.
    let shape = [
        artifacts.manifest().vocab_size as i32,
        artifacts.manifest().hidden_size as i32,
    ];
    let zeros = vec![0.0f32; (shape[0] * shape[1]) as usize];
    trunk.lm_head = QuantizedWeight::new(
        MlxArray::from_raw_data(
            zeros.as_ptr().cast(),
            std::mem::size_of_val(zeros.as_slice()),
            &shape,
            MlxDtype::Float32,
        ),
        None,
        None,
    );
    let head = crate::weights::qwen4_exp_mtp::load(&root, artifacts.manifest(), &trunk).unwrap();
    let (owner, draft_owner) = (1501, 1502);
    let mut session = CandidateSession::prefill(&trunk, &head, &[1], owner, draft_owner).unwrap();
    let mut direct = qwen4_exp::forward(
        &trunk,
        &[1],
        &qwen4_exp::Qwen4ExpState::new(&trunk, owner),
        owner,
        ProjectionBatchPolicy::Shared,
    )
    .unwrap();
    let mut draft_reference = qwen4_exp::Qwen4ExpState::new(&head.graph, draft_owner);
    let mut count = 0;
    while count < 5 {
        let remaining = 5 - count;
        let committed = session.step(&trunk, &head, remaining, &[]).unwrap();
        assert!(!committed.is_empty() && committed.len() <= remaining);
        for token in committed {
            assert_eq!(token, 0);
            // Independent singleton teacher-forced draft history: every committed
            // token contributes one pair, regardless of the candidate branch.
            draft_reference = head_forward(
                &head,
                &direct.stream_hidden,
                &[token],
                &draft_reference,
                draft_owner,
            )
            .unwrap()
            .state;
            direct = qwen4_exp::forward(
                &trunk,
                &[token],
                &direct.state,
                owner,
                ProjectionBatchPolicy::Shared,
            )
            .unwrap();
            count += 1;
        }
        mtp_parity::assert_mtp_state_close(
            &session.trunk_state,
            &direct.state,
            direct.stream_hidden.dtype(),
        );
        mtp_parity::assert_mtp_state_close(
            &session.draft_state,
            &draft_reference,
            direct.stream_hidden.dtype(),
        );
    }
    assert_eq!(count, 5);
    assert_eq!((session.proposed, session.accepted), (2, 2));
    let mut terminal = CandidateSession::prefill(&trunk, &head, &[1], owner, draft_owner).unwrap();
    assert_eq!(terminal.step(&trunk, &head, 5, &[0]).unwrap(), vec![0]);
    assert_eq!(terminal.accepted, 0);

    // Exercise the runner's already-emitted-primary mapping on the accepted,
    // terminal and final-budget branches, using the same nontrivial layers.
    let prefill = qwen4_exp::forward(
        &trunk,
        &[1],
        &qwen4_exp::Qwen4ExpState::new(&trunk, owner),
        owner,
        ProjectionBatchPolicy::Shared,
    )
    .unwrap();
    let mut cursor = crate::model::qwen4_exp_mtp::Qwen4ExpDraftCursor::new(&head, owner);
    cursor.absorb(&head, &[1], &prefill.stream_hidden).unwrap();
    for (remaining, stops, length, accepted) in [
        (2, vec![], 2, true),
        (1, vec![], 1, false),
        (5, vec![0], 1, false),
    ] {
        let mut trial = cursor.clone();
        let step = trial
            .step(&trunk, &head, &prefill.state, owner, 0, remaining, &stops)
            .unwrap();
        assert_eq!(step.emitted, vec![0; length]);
        assert_eq!(step.committed_len, length);
        assert_eq!(step.accepted, accepted);
        assert_eq!(
            step.trunk_state.position(),
            prefill.state.position() + length
        );
        assert!(trial.aligned(&step.trunk_state));
    }
}

#[test]
#[ignore = "requires synthetic Flash Next MTP candidate artifacts"]
fn flash_next_mtp_oracle_agreement_matches_proposals_when_stopped_inside_budget() {
    use crate::model::qwen4_exp_mtp::{CandidateSession, head_forward};
    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_MTP_ORACLE_DIR").unwrap());
    let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
    manifest.weight_sanitize = WeightSanitize::HfToMlx;
    manifest.runtime_status = NativeRuntimeStatus::default();
    let artifacts = NativeModelArtifacts::from_manifest_and_root(root.clone(), manifest).unwrap();
    let mut trunk = crate::weights::qwen4_exp::load_with_paging_policy(
        &root,
        artifacts.manifest(),
        crate::expert_stream::StreamExpertsMode::Off,
        1,
    )
    .unwrap();
    let shape = [
        artifacts.manifest().vocab_size as i32,
        artifacts.manifest().hidden_size as i32,
    ];
    let zeros = vec![0.0f32; (shape[0] * shape[1]) as usize];
    trunk.lm_head = QuantizedWeight::new(
        MlxArray::from_raw_data(
            zeros.as_ptr().cast(),
            std::mem::size_of_val(zeros.as_slice()),
            &shape,
            MlxDtype::Float32,
        ),
        None,
        None,
    );
    let head = crate::weights::qwen4_exp_mtp::load(&root, artifacts.manifest(), &trunk).unwrap();
    let (owner, draft_owner) = (1601, 1602);

    let too_short =
        flash_next_mtp_oracle_generate(&trunk, &head, &[1], 5, owner, draft_owner, &[0]);
    assert!(too_short.too_short);
    assert!(too_short.stopped_at_terminal);
    assert_eq!(too_short.compared_positions, 1);
    assert_eq!(too_short.agreement.len(), too_short.proposed);
    assert_eq!(too_short.proposed, 0);

    let full_budget =
        flash_next_mtp_oracle_generate(&trunk, &head, &[1], 5, owner + 2, draft_owner + 2, &[]);
    assert!(!full_budget.stopped_at_terminal);
    assert_eq!(full_budget.generated.len(), 5);
    assert_eq!(full_budget.compared_positions, 5);
    assert_eq!(full_budget.agreement.len(), full_budget.proposed);
    assert_eq!(full_budget.proposed, 2);

    let mut session =
        CandidateSession::prefill(&trunk, &head, &[1], owner + 4, draft_owner + 4).unwrap();
    let mut agreement = Vec::new();
    let mut proposed = 0usize;
    let mut accepted = 0usize;
    let mut generated = Vec::new();
    let max_new = 5;
    let mut stopped_at_terminal = false;
    while generated.len() < max_new {
        let remaining = max_new - generated.len();
        let terminal_ids: &[u32] = if session.proposed > 0 { &[0] } else { &[] };
        if terminal_ids.contains(&session.primary) {
            generated.push(session.primary);
            stopped_at_terminal = true;
            break;
        }
        let proposed_before = session.proposed;
        let accepted_before = session.accepted;
        let draft_token = (remaining > 1).then(|| {
            flash_next_mtp_greedy_token(
                &head_forward(
                    &head,
                    &session.stream_hidden,
                    &[session.primary],
                    &session.draft_state,
                    draft_owner + 4,
                )
                .unwrap(),
            )
        });
        let committed = session
            .step(&trunk, &head, remaining, terminal_ids)
            .unwrap();
        flash_next_mtp_note_proposal_agreement(
            &mut agreement,
            &mut proposed,
            &mut accepted,
            (proposed_before, accepted_before),
            (session.proposed, session.accepted),
            draft_token.map(|draft| {
                flash_next_mtp_draft_matches_committed(draft, &committed, session.primary)
            }),
        );
        for token in &committed {
            generated.push(*token);
            if terminal_ids.contains(token) {
                stopped_at_terminal = true;
                break;
            }
        }
        if stopped_at_terminal {
            break;
        }
    }
    assert!(stopped_at_terminal);
    assert!(generated.len() < max_new);
    assert!(session.proposed > 0);
    assert_eq!(agreement.len(), proposed);
    assert_eq!(agreement.len(), session.proposed);
    assert_eq!(agreement.iter().filter(|agreed| **agreed).count(), accepted);
}

#[test]
#[ignore = "requires synthetic Flash Next MTP candidate artifacts"]
fn qwen4_exp_mtp_prefill_cursor_matches_primary_across_quanta_and_budgets() {
    use crate::generate::{
        CacheOnlyBarrier, CacheOnlyPrefillLayout, chunked_prefill_cache_only,
        chunked_prefill_flash_next_mtp,
    };
    use crate::model::qwen4_exp_mtp::{Qwen4ExpDraftCursor, mtp_parity};
    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_MTP_ORACLE_DIR").unwrap());
    let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
    manifest.weight_sanitize = WeightSanitize::HfToMlx;
    manifest.runtime_status = NativeRuntimeStatus::default();
    let artifacts = NativeModelArtifacts::from_manifest_and_root(root.clone(), manifest).unwrap();
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let mut weights = crate::weights::load_weights(&artifacts).unwrap();
    weights.qwen4_exp_mtp = Some(Box::new(
        crate::weights::qwen4_exp_mtp::load(
            &root,
            artifacts.manifest(),
            weights.qwen4_exp.as_ref().unwrap(),
        )
        .unwrap(),
    ));
    let head = weights.qwen4_exp_mtp.as_ref().unwrap();
    let trunk = weights.qwen4_exp.as_ref().unwrap();
    let greedy = |output: &qwen4_exp::Qwen4ExpOutput| {
        let next = mlx_sys::argmax(&output.logits, None);
        mlx_sys::eval(&[&next]);
        next.data_u32()[0]
    };
    for prompt in [vec![1], vec![1, 2, 3, 4, 5, 6, 7]] {
        for quantum in [1, 3, 100] {
            let mut actual = MlxKVCache::new_contiguous(cfg.layer_count);
            let mut reference = MlxKVCache::new_contiguous(cfg.layer_count);
            let mut cursor = Some(Qwen4ExpDraftCursor::new(head, cfg.compile_cache_identity));
            let chunks: Vec<_> = prompt.chunks(quantum).collect();
            let mut first = None;
            for (index, chunk) in chunks.iter().enumerate() {
                let final_quantum = index + 1 == chunks.len();
                first = chunked_prefill_flash_next_mtp(
                    &cfg,
                    &weights,
                    chunk,
                    &mut actual,
                    2,
                    final_quantum,
                    &mut cursor,
                );
                if final_quantum {
                    let prefix = &chunk[..chunk.len() - 1];
                    for part in prefix.chunks(2) {
                        let offset = reference.seq_len();
                        forward_cache_only(&cfg, &weights, part, &mut reference, offset);
                        reference.advance(part.len());
                    }
                    let offset = reference.seq_len();
                    let logits = forward(
                        &cfg,
                        &weights,
                        &chunk[chunk.len() - 1..],
                        &mut reference,
                        offset,
                    );
                    reference.advance(1);
                    let token = mlx_sys::argmax(&logits, None);
                    mlx_sys::eval(&[&token]);
                    assert_eq!(first, Some(token.data_u32()[0]));
                } else {
                    assert!(first.is_none());
                    chunked_prefill_cache_only(
                        &cfg,
                        &weights,
                        chunk,
                        &mut reference,
                        2,
                        CacheOnlyPrefillLayout::PreserveFinalTokenStep,
                        CacheOnlyBarrier::Blocking,
                    );
                }
                assert_eq!(actual.serialize_to_bytes(), reference.serialize_to_bytes());
                assert!(
                    cursor
                        .as_ref()
                        .unwrap()
                        .aligned(actual.qwen4_exp.as_ref().unwrap())
                );
            }
            let start_cursor = cursor.unwrap();
            for budget in [1, 2, 3, 8] {
                let mut cursor = start_cursor.clone();
                let mut actual = actual.clone();
                let mut reference = reference.clone();
                let mut last = first.unwrap();
                let mut emitted = 0;
                while emitted < budget {
                    let step = cursor
                        .step(
                            trunk,
                            head,
                            actual.qwen4_exp.as_ref().unwrap(),
                            cfg.compile_cache_identity,
                            last,
                            budget - emitted,
                            &[],
                        )
                        .unwrap();
                    assert!(!step.emitted.is_empty());
                    assert!(step.emitted.len() <= budget - emitted);
                    assert_eq!(step.committed_len, step.emitted.len());
                    for &token in &step.emitted {
                        let output = qwen4_exp::forward(
                            trunk,
                            &[last],
                            reference.qwen4_exp.as_ref().unwrap(),
                            cfg.compile_cache_identity,
                            ProjectionBatchPolicy::Shared,
                        )
                        .unwrap();
                        assert_eq!(token, greedy(&output));
                        reference.qwen4_exp = Some(output.state);
                        reference.advance(1);
                        last = token;
                        emitted += 1;
                    }
                    actual.qwen4_exp = Some(step.trunk_state);
                    actual.advance(step.committed_len);
                    mtp_parity::assert_mtp_state_close(
                        actual.qwen4_exp.as_ref().unwrap(),
                        reference.qwen4_exp.as_ref().unwrap(),
                        trunk.token_embedding.weight.dtype(),
                    );
                    assert_eq!(actual.seq_len(), reference.seq_len());
                    assert!(cursor.aligned(actual.qwen4_exp.as_ref().unwrap()));
                }
                assert_eq!(emitted, budget);
            }
        }
    }
}
