//! Test-binary-only synchronized attribution. Barriers perturb scheduling;
//! these measurements are not end-to-end throughput benchmarks.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(serde::Serialize)]
struct Sample {
    layer: Option<usize>,
    stage: &'static str,
    seconds: f64,
}

struct Capture {
    layer: Option<usize>,
    previous: Instant,
    samples: Vec<Sample>,
}

thread_local! {
    static CAPTURE: RefCell<Option<Capture>> = const { RefCell::new(None) };
    static DUMP_LAYER: Cell<usize> = const { Cell::new(usize::MAX) };
    static DUMP_FORWARD: Cell<usize> = const { Cell::new(0) };
}

pub(crate) fn layer(index: usize) {
    DUMP_LAYER.set(index);
    CAPTURE.with_borrow_mut(|capture| {
        if let Some(capture) = capture {
            capture.layer = Some(index);
        }
    });
}

pub(crate) fn mark(stage: &'static str, arrays: &[&MlxArray]) {
    if let Some(root) = std::env::var_os("AX_FLASH_NEXT_FIRST_LAYER_DUMP") {
        if stage == "embedding" {
            DUMP_LAYER.set(usize::MAX);
            DUMP_FORWARD.set(DUMP_FORWARD.get() + 1);
        }
        if (stage == "embedding" || DUMP_LAYER.get() == 0)
            && let Some(array) = arrays.first()
        {
            let values = mlx_sys::astype(array, MlxDtype::Float32, None);
            mlx_sys::try_eval(&[&values]).expect("first-layer diagnostic evaluation");
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            let name = format!("forward-{}-{stage}", DUMP_FORWARD.get());
            let bytes: Vec<u8> = values
                .data_f32()
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            std::fs::write(root.join(format!("{name}.f32le")), bytes).unwrap();
            std::fs::write(
                root.join(format!("{name}.json")),
                serde_json::to_vec(&serde_json::json!({
                    "shape": array.shape(), "dtype": format!("{:?}", array.dtype()),
                }))
                .unwrap(),
            )
            .unwrap();
        }
    }
    CAPTURE.with_borrow_mut(|capture| {
        if let Some(capture) = capture {
            eval(arrays);
            let now = Instant::now();
            capture.samples.push(Sample {
                layer: capture.layer,
                stage,
                seconds: now.duration_since(capture.previous).as_secs_f64(),
            });
            capture.previous = now;
        }
    });
}

struct CaptureGuard;

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        CAPTURE.with_borrow_mut(|capture| *capture = None);
    }
}

fn snapshot(state: Qwen4ExpState) -> Vec<u8> {
    let mut cache = crate::kv_cache::MlxKVCache::new_contiguous(state.layers.len());
    cache.advance(state.position());
    cache.qwen4_exp = Some(state);
    cache.serialize_to_bytes()
}

#[test]
#[ignore = "requires a generated oracle or explicit real campaign pack; synchronized attribution only"]
fn flash_next_synchronized_operator_profile() {
    use crate::model::ModelConfig;
    use ax_engine_core::{NativeModelArtifacts, NativeRuntimeStatus, WeightSanitize};
    use std::path::PathBuf;

    let (artifacts, tokens) = if let Some(root) = std::env::var_os("AX_FLASH_NEXT_PROFILE_PACK") {
        let artifacts = NativeModelArtifacts::from_dir(PathBuf::from(root)).unwrap();
        let tokens: Vec<u32> =
            serde_json::from_str(&std::env::var("AX_FLASH_NEXT_PROMPT_IDS").unwrap()).unwrap();
        (artifacts, tokens)
    } else {
        let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_ORACLE_DIR").unwrap());
        let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
        manifest.weight_sanitize = WeightSanitize::HfToMlx;
        manifest.runtime_status = NativeRuntimeStatus::default();
        let fixture: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("logits.json")).unwrap()).unwrap();
        let tokens: Vec<u32> = serde_json::from_value(fixture["tokens"].clone()).unwrap();
        (
            NativeModelArtifacts::from_manifest_and_root(root, manifest).unwrap(),
            tokens,
        )
    };
    assert!((2..=128).contains(&tokens.len()));
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = crate::weights::load_weights(&artifacts).unwrap();
    let trunk = weights.qwen4_exp.as_ref().unwrap();
    let initial = Qwen4ExpState::new(trunk, cfg.compile_cache_identity);
    let prefix = forward(
        trunk,
        &tokens[..tokens.len() - 1],
        &initial,
        cfg.compile_cache_identity,
        ProjectionBatchPolicy::Shared,
    )
    .unwrap();
    let tail = &tokens[tokens.len() - 1..];
    let mut records = Vec::new();
    // Each pair uses identical inputs and immutable state. The control is
    // immediately followed by an instrumented replay with warm file caches.
    for (name, input, state) in [
        ("prefill", &tokens[..tokens.len() - 1], &initial),
        ("singleton", tail, &prefix.state),
    ] {
        let start = Instant::now();
        let control = forward(
            trunk,
            input,
            state,
            cfg.compile_cache_identity,
            ProjectionBatchPolicy::Shared,
        )
        .unwrap();
        let control_seconds = start.elapsed().as_secs_f64();
        CAPTURE.with_borrow_mut(|capture| {
            assert!(capture.is_none());
            *capture = Some(Capture {
                layer: None,
                previous: Instant::now(),
                samples: Vec::new(),
            });
        });
        let guard = CaptureGuard;
        let measured = forward(
            trunk,
            input,
            state,
            cfg.compile_cache_identity,
            ProjectionBatchPolicy::Shared,
        )
        .unwrap();
        let capture = CAPTURE.with_borrow_mut(Option::take).unwrap();
        drop(guard);
        assert_eq!(measured.logits.data_f32(), control.logits.data_f32());
        assert_eq!(snapshot(measured.state), snapshot(control.state));
        let mut totals = BTreeMap::new();
        for sample in &capture.samples {
            *totals.entry(sample.stage).or_insert(0.0) += sample.seconds;
        }
        records.push(serde_json::json!({
            "name": name, "tokens": input.len(), "control_seconds": control_seconds,
            "stage_seconds": totals, "samples": capture.samples,
            "logits_exact": true, "state_bytes_exact": true,
        }));
    }
    let evidence = serde_json::json!({
        "qualification": false, "method": "synchronized stage attribution with exact control replay",
        "cache_state": "warm after preceding same-input control; no OS cache purge",
        "expert_streaming": trunk.expert_stream.is_some(), "records": records,
        "peak_bytes": mlx_sys::get_peak_memory(),
    });
    if let Some(path) = std::env::var_os("AX_FLASH_NEXT_PROFILE_OUTPUT") {
        std::fs::write(path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
    }
    eprintln!("{evidence}");
}

#[test]
#[ignore = "requires the real campaign pack on the authorized M2 host; no qualification claim"]
fn flash_next_real_gdn_metal_control() {
    use crate::model::{ModelConfig, shared::qwen4_exp_gdn_metal::with_mode};
    use ax_engine_core::NativeModelArtifacts;
    use std::path::PathBuf;

    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_PROFILE_PACK").unwrap());
    let artifacts = NativeModelArtifacts::from_dir(root).unwrap();
    let tokens: Vec<u32> =
        serde_json::from_str(&std::env::var("AX_FLASH_NEXT_PROMPT_IDS").unwrap()).unwrap();
    let expected: Vec<u32> =
        serde_json::from_str(&std::env::var("AX_FLASH_NEXT_EXPECTED_IDS").unwrap()).unwrap();
    assert!((2..=32).contains(&tokens.len()));
    assert_eq!(expected.len(), 16);
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = crate::weights::load_weights(&artifacts).unwrap();
    let trunk = weights.qwen4_exp.as_ref().unwrap();
    assert!(
        trunk.expert_stream.is_none(),
        "resident operator experiment"
    );
    let initial = Qwen4ExpState::new(trunk, cfg.compile_cache_identity);
    let (prefix, _) = with_mode(false, || {
        forward(
            trunk,
            &tokens[..tokens.len() - 1],
            &initial,
            cfg.compile_cache_identity,
            ProjectionBatchPolicy::Shared,
        )
        .unwrap()
    });
    let mut records = Vec::new();
    let mut reference_logits: Vec<Vec<f32>> = Vec::new();
    let mut reference_state = Vec::new();
    // Alternate order after an untimed compilation warmup. Compare whole
    // greedy continuations; the prefix schedule and weight handles are shared.
    let _ = with_mode(true, || {
        forward(
            trunk,
            &tokens[tokens.len() - 1..],
            &prefix.state,
            cfg.compile_cache_identity,
            ProjectionBatchPolicy::Shared,
        )
        .unwrap()
    });
    for (round, native) in [false, true, true, false, false, true]
        .into_iter()
        .enumerate()
    {
        let start = Instant::now();
        let ((ids, logits, next_state), dispatches) = with_mode(native, || {
            let mut state = prefix.state.clone();
            let mut next = tokens[tokens.len() - 1];
            let mut ids = Vec::new();
            let mut logits = Vec::new();
            for _ in 0..expected.len() {
                let output = forward(
                    trunk,
                    &[next],
                    &state,
                    cfg.compile_cache_identity,
                    ProjectionBatchPolicy::Shared,
                )
                .unwrap();
                let selected = mlx_sys::argmax(&output.logits, None);
                eval(&[&selected]);
                next = selected.data_u32()[0];
                ids.push(next);
                logits.push(output.logits.data_f32().to_vec());
                state = output.state;
            }
            (ids, logits, state)
        });
        let seconds = start.elapsed().as_secs_f64();
        let state_position = next_state.position();
        let state_bytes = snapshot(next_state);
        if round == 0 {
            reference_logits.clone_from(&logits);
            reference_state.clone_from(&state_bytes);
        }
        let state_exact = state_bytes == reference_state;
        let mut max_logit_error = 0.0_f32;
        for (&actual, &control) in logits
            .iter()
            .flatten()
            .zip(reference_logits.iter().flatten())
        {
            assert!(actual.is_finite() && control.is_finite());
            max_logit_error = max_logit_error.max((actual - control).abs());
        }
        records.push(serde_json::json!({"round": round, "native": native,
            "seconds": seconds, "tokens": ids, "expected_ids": expected,
            "native_dispatches": dispatches, "max_logit_error": max_logit_error,
            "state_position": state_position, "state_bytes_exact": state_exact, "tokens_match": ids == expected}));
        let evidence = serde_json::json!({"qualification": false,
            "method": "same-prefix resident alternating singleton continuations; logits copied to host",
            "records": records, "peak_bytes": mlx_sys::get_peak_memory()});
        if let Some(path) = std::env::var_os("AX_FLASH_NEXT_PROFILE_OUTPUT") {
            std::fs::write(path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
        }
        eprintln!(
            "GDN control round={round} native={native} seconds={seconds} dispatches={dispatches} max_logit_error={max_logit_error} ids={ids:?}"
        );
        assert_eq!(
            ids, expected,
            "native candidate must preserve the resident token control"
        );
        assert_eq!(dispatches > 0, native);
        assert_eq!(
            max_logit_error, 0.0,
            "elementwise fusion must preserve exact logits"
        );
        assert!(state_exact, "elementwise fusion must preserve exact state");
    }
}
