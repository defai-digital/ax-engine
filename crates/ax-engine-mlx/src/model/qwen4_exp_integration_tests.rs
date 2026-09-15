//! Explicit artifact-backed tests for the production Flash Next forward boundary.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;
use ax_engine_core::{NativeModelArtifacts, NativeRuntimeStatus, WeightSanitize};
use std::path::PathBuf;

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
    assert!(!artifacts.manifest().runtime_status.ready);
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
        "production admission remains closed during integration"
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

#[test]
#[ignore = "requires synthetic Flash Next MTP artifacts or an explicit real selected-prefill pack"]
fn qwen4_exp_mtp_candidate_keeps_primary_tokens_and_state_exact() {
    use crate::model::qwen4_exp_mtp::{CandidateSession, head_forward, verify_one};
    let (artifacts, tokens, mode, budget) =
        if let Some(root) = std::env::var_os("AX_FLASH_NEXT_REAL_PACK") {
            let artifacts = NativeModelArtifacts::from_dir(PathBuf::from(root)).unwrap();
            let tokens: Vec<u32> =
                serde_json::from_str(&std::env::var("AX_FLASH_NEXT_PROMPT_IDS").unwrap()).unwrap();
            assert!((3..=16).contains(&tokens.len()));
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
        assert_eq!(
            trunk.expert_stream.as_ref().unwrap().cached_layer_count(),
            0
        );
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
    for (draft, remaining, accepted) in [
        (correct_draft, 2, true),
        (wrong_draft, 2, false),
        (correct_draft, 1, false),
    ] {
        let verified =
            verify_one(&trunk, &direct.state, owner, primary, draft, remaining, &[]).unwrap();
        assert_eq!(verified.accepted, accepted);
        assert_eq!(verified.committed.len(), if accepted { 2 } else { 1 });
        assert_eq!(
            flash_mtp_state_bytes(&verified.after_primary.state, trunk.layers.len()),
            flash_mtp_state_bytes(&target.state, trunk.layers.len())
        );
        if accepted {
            let second = qwen4_exp::forward(
                &trunk,
                &[correct_draft],
                &target.state,
                owner,
                ProjectionBatchPolicy::Shared,
            )
            .unwrap();
            assert_eq!(
                flash_mtp_state_bytes(
                    &verified.after_draft.as_ref().unwrap().state,
                    trunk.layers.len()
                ),
                flash_mtp_state_bytes(&second.state, trunk.layers.len())
            );
        } else {
            assert!(verified.after_draft.is_none());
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
            assert_eq!(*token, greedy(&direct));
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
        generated.extend(committed);
        assert_eq!(session.primary, greedy(&direct));
        assert_eq!(
            flash_mtp_state_bytes(&session.trunk_state, trunk.layers.len()),
            flash_mtp_state_bytes(&direct.state, trunk.layers.len())
        );
        assert_eq!(
            session.draft_state.position() + 1,
            session.trunk_state.position()
        );
        assert_eq!(
            flash_mtp_state_bytes(&session.draft_state, 1),
            flash_mtp_state_bytes(&draft_reference, 1)
        );
    }
    assert_eq!(generated.len(), budget);
    eprintln!(
        "Flash Next MTP control: proposed={}, accepted={}, tokens={generated:?}",
        session.proposed, session.accepted
    );
    if let Some(bytes) = selected_after_prefill {
        assert!(session.proposed > 0);
        assert!(mtp_selected_decode_bytes > 0);
        assert_eq!(
            trunk.expert_stream.as_ref().unwrap().cached_layer_count(),
            0
        );
        let result = serde_json::json!({
            "qualification": false, "generated_ids": generated,
            "selected_payload_after_prefill": bytes,
            "selected_payload_after_run": trunk.expert_stream.as_ref().unwrap()
                .selected_payload_bytes_read().unwrap(),
            "proposed": session.proposed, "accepted": session.accepted,
            "mtp_only_selected_decode_payload_bytes": mtp_selected_decode_bytes,
            "prefill_primary_and_draft_state_exact": true,
            "draft_state_exact_each_step": true,
            "cached_whole_layers_after_run": 0,
            "primary_state_exact_each_step": true,
            "forced_acceptance_rejection_budget_and_eos": true,
            "zero_budget_preserves_primary_and_draft_state": true,
        });
        std::fs::write(
            std::env::var_os("AX_FLASH_NEXT_RESULT_PATH").unwrap(),
            serde_json::to_vec_pretty(&result).unwrap(),
        )
        .unwrap();
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
    use crate::model::qwen4_exp_mtp::CandidateSession;
    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_CANDIDATE_PACK_DIR").unwrap());
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
    let head = crate::weights::qwen4_exp_mtp::load(&root, artifacts.manifest(), &trunk).unwrap();
    let head_seconds = start.elapsed().as_secs_f64();
    eprintln!(
        "Flash Next MTP loaded: trunk_seconds={trunk_seconds}, head_seconds={head_seconds}, active_bytes={}",
        mlx_sys::mempressure::device_active_bytes().unwrap()
    );
    let start = std::time::Instant::now();
    let mut session = CandidateSession::prefill(&trunk, &head, &tokens, 1201, 1202).unwrap();
    let mut generated = Vec::new();
    while generated.len() < expected.len() {
        let committed = session
            .step(&trunk, &head, expected.len() - generated.len(), &[])
            .unwrap();
        eprintln!("Flash Next MTP committed: {committed:?}");
        generated.extend(committed);
    }
    let result = serde_json::json!({
        "qualification":false,"route":"flash_next_mtp_candidate_sequential_primary_verify",
        "trunk_load_seconds":trunk_seconds,"head_load_seconds":head_seconds,
        "generation_seconds":start.elapsed().as_secs_f64(),"prompt_ids":tokens,
        "expected_ids":expected,"generated_ids":generated,"proposed":session.proposed,
        "accepted":session.accepted,"peak_mlx_bytes":mlx_sys::get_peak_memory(),
        "active_bytes":mlx_sys::mempressure::device_active_bytes().unwrap(),
        "trunk_position":session.trunk_state.position(),"draft_position":session.draft_state.position(),
        "note":"Reconstructed candidate head; primary verification is authoritative; no throughput claim"
    });
    if let Some(path) = std::env::var_os("AX_FLASH_NEXT_SMOKE_OUTPUT") {
        std::fs::write(path, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    }
    eprintln!("{result}");
    assert_eq!(generated, expected);
    assert!(session.proposed > 0);
    assert_eq!(
        session.draft_state.position() + 1,
        session.trunk_state.position()
    );
}

#[test]
#[ignore = "requires synthetic Flash Next MTP candidate artifacts"]
fn qwen4_exp_mtp_candidate_accepts_with_exact_draft_history_and_budget() {
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
        assert_eq!(
            flash_mtp_state_bytes(&session.trunk_state, trunk.layers.len()),
            flash_mtp_state_bytes(&direct.state, trunk.layers.len())
        );
        assert_eq!(
            flash_mtp_state_bytes(&session.draft_state, 1),
            flash_mtp_state_bytes(&draft_reference, 1)
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
fn qwen4_exp_mtp_prefill_cursor_matches_primary_across_quanta_and_budgets() {
    use crate::generate::{
        CacheOnlyBarrier, CacheOnlyPrefillLayout, chunked_prefill_cache_only,
        chunked_prefill_flash_next_mtp,
    };
    use crate::model::qwen4_exp_mtp::Qwen4ExpDraftCursor;
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
                    assert_eq!(actual.serialize_to_bytes(), reference.serialize_to_bytes());
                    assert!(cursor.aligned(actual.qwen4_exp.as_ref().unwrap()));
                }
                assert_eq!(emitted, budget);
            }
        }
    }
}
