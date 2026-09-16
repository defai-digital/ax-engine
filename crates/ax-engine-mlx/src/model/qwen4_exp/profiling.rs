//! Test-binary-only synchronized attribution. Barriers perturb scheduling;
//! these measurements are not end-to-end throughput benchmarks.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use sha2::{Digest, Sha256};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Stages dumped when `AX_FLASH_NEXT_DUMP_POSITIONS` is set and no stage
/// allow-list is provided. Full-sequence dumps of every stage at 2k tokens
/// are too large for the records-attribution campaign.
const DEFAULT_POSITION_STAGES: &[&str] = &[
    "embedding",
    "ple_rows",
    "ngram_lookup",
    "ple_delta",
    "ple_conv_activated",
    "ple_operator",
    "attention_hc_read",
    "gdn",
    "qsa_gather_indices",
    "qsa",
    "qsa_output",
    "attention_hc_write",
    "mlp_hc_read",
    "moe_compute",
    "mlp_hc_write",
    "lm_head",
];

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
    static DUMP_HC: Cell<usize> = const { Cell::new(0) };
    static DUMP_OFFSET: Cell<usize> = const { Cell::new(0) };
    static DUMP_CHUNK_LEN: Cell<usize> = const { Cell::new(0) };
}

pub(crate) fn layer(index: usize) {
    DUMP_LAYER.set(index);
    DUMP_HC.set(0);
    CAPTURE.with_borrow_mut(|capture| {
        if let Some(capture) = capture {
            capture.layer = Some(index);
        }
    });
}

pub(crate) fn begin_forward_dump() {
    DUMP_LAYER.set(usize::MAX);
    DUMP_FORWARD.set(DUMP_FORWARD.get() + 1);
}

pub(crate) fn begin_chunk(offset: usize, len: usize) {
    DUMP_OFFSET.set(offset);
    DUMP_CHUNK_LEN.set(len);
}

fn parse_usize_csv(value: &str) -> Vec<usize> {
    value
        .split(|ch: char| ch == ',' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| part.parse::<usize>().expect("diagnostic dump position"))
        .collect()
}

fn parse_stage_csv(value: &str) -> Vec<String> {
    value
        .split(|ch: char| ch == ',' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| part.to_string())
        .collect()
}

fn dump_positions() -> Option<BTreeSet<usize>> {
    let value = std::env::var("AX_FLASH_NEXT_DUMP_POSITIONS").ok()?;
    if value.trim().is_empty() {
        return None;
    }
    Some(parse_usize_csv(&value).into_iter().collect())
}

fn dump_layer_env() -> Option<usize> {
    std::env::var("AX_FLASH_NEXT_DUMP_LAYER")
        .ok()
        .map(|value| value.parse::<usize>().expect("diagnostic layer index"))
}

fn dump_stages() -> Option<Vec<String>> {
    match std::env::var("AX_FLASH_NEXT_DUMP_STAGES") {
        Ok(value) if !value.trim().is_empty() => Some(parse_stage_csv(&value)),
        _ if dump_positions().is_some() => Some(
            DEFAULT_POSITION_STAGES
                .iter()
                .map(|stage| (*stage).to_string())
                .collect(),
        ),
        _ => None,
    }
}

fn stage_allowed(stage: &str) -> bool {
    let Some(list) = dump_stages() else {
        return true;
    };
    if list.iter().any(|item| item == "*" || item == "all") {
        return true;
    }
    list.iter().any(|item| item == stage)
}

fn dump_root() -> Option<PathBuf> {
    if let Some(root) = std::env::var_os("AX_FLASH_NEXT_FIRST_LAYER_DUMP") {
        return Some(PathBuf::from(root));
    }
    if dump_positions().is_some() {
        let root = std::env::var_os("AX_FLASH_NEXT_LOGITS_DIR")?;
        return Some(PathBuf::from(root).join("stages"));
    }
    None
}

fn should_dump(stage: &str) -> bool {
    if !stage_allowed(stage) {
        return false;
    }
    if stage == "embedding" {
        return true;
    }
    let layer = DUMP_LAYER.get();
    if let Some(selected) = dump_layer_env() {
        return layer == selected;
    }
    if dump_positions().is_some() {
        return true;
    }
    layer == 0
}

/// Sequence axis and grouping for a chunk-length tensor.
/// Prefers `[1, chunk, ...]`, then an axis equal to `chunk_len`, then an
/// axis that is an integer multiple of `chunk_len` (token-major grouped
/// rows such as PLE n-gram gathers).
fn sequence_axis(shape: &[i32], chunk_len: i32) -> Option<(usize, i32)> {
    if chunk_len <= 0 || shape.is_empty() {
        return None;
    }
    if shape.len() >= 2 && shape[0] == 1 && shape[1] == chunk_len {
        return Some((1, 1));
    }
    for (axis, &dim) in shape.iter().enumerate() {
        if dim == chunk_len {
            return Some((axis, 1));
        }
    }
    // Token-major grouped rows (PLE n-gram gathers): [seq * heads, width].
    // Do not treat a trailing feature dim that happens to be a multiple of
    // the chunk length as a sequence axis.
    let dim0 = shape[0];
    if dim0 > chunk_len && dim0 % chunk_len == 0 {
        let group = dim0 / chunk_len;
        if group <= 64 {
            return Some((0, group));
        }
    }
    None
}

fn extract_position(
    data: &[f32],
    shape: &[i32],
    axis: usize,
    group: i32,
    local: usize,
) -> (Vec<f32>, Vec<i32>) {
    let dims: Vec<usize> = shape.iter().map(|&dim| dim as usize).collect();
    let before: usize = dims[..axis].iter().copied().product();
    let dim = dims[axis];
    let after: usize = dims[axis + 1..].iter().copied().product();
    let group = group as usize;
    let start = local.checked_mul(group).expect("dump position grouping");
    let end = start.checked_add(group).expect("dump position grouping");
    assert!(end <= dim, "dump position {local} exceeds axis {dim}");
    let mut out = Vec::with_capacity(before.saturating_mul(group).saturating_mul(after));
    for prefix in 0..before {
        for index in start..end {
            let offset = (prefix * dim + index) * after;
            out.extend_from_slice(&data[offset..offset + after]);
        }
    }
    let mut out_shape = shape.to_vec();
    out_shape[axis] = group as i32;
    (out, out_shape)
}

fn write_f32le(path: &Path, values: &[f32]) {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    std::fs::write(path, bytes).unwrap();
}

fn layer_label() -> serde_json::Value {
    let layer = DUMP_LAYER.get();
    if layer == usize::MAX {
        serde_json::Value::Null
    } else {
        serde_json::json!(layer)
    }
}

fn layer_name() -> String {
    let layer = DUMP_LAYER.get();
    if layer == usize::MAX {
        "none".to_string()
    } else {
        layer.to_string()
    }
}

fn stage_stem(stage: &str) -> String {
    if stage.starts_with("hc_") {
        format!("hc-{}-{stage}", DUMP_HC.get())
    } else {
        stage.to_string()
    }
}

fn dump_full(root: &Path, stage: &str, array: &MlxArray) {
    let values = mlx_sys::astype(array, MlxDtype::Float32, None);
    mlx_sys::try_eval(&[&values]).expect("first-layer diagnostic evaluation");
    std::fs::create_dir_all(root).unwrap();
    let name = if stage.starts_with("hc_") {
        format!(
            "forward-{}-hc-{}-{stage}",
            DUMP_FORWARD.get(),
            DUMP_HC.get()
        )
    } else {
        format!("forward-{}-{stage}", DUMP_FORWARD.get())
    };
    write_f32le(&root.join(format!("{name}.f32le")), values.data_f32());
    std::fs::write(
        root.join(format!("{name}.json")),
        serde_json::to_vec(&serde_json::json!({
            "shape": array.shape(), "dtype": format!("{:?}", array.dtype()),
            "layer": DUMP_LAYER.get(),
        }))
        .unwrap(),
    )
    .unwrap();
}

fn dump_selected(root: &Path, stage: &str, array: &MlxArray, wanted: &BTreeSet<usize>) {
    let chunk_len = DUMP_CHUNK_LEN.get();
    let offset = DUMP_OFFSET.get();
    if chunk_len == 0 {
        return;
    }
    let locals: Vec<usize> = wanted
        .iter()
        .copied()
        .filter(|&pos| pos >= offset && pos < offset + chunk_len)
        .map(|pos| pos - offset)
        .collect();
    if locals.is_empty() {
        return;
    }
    let values = mlx_sys::astype(array, MlxDtype::Float32, None);
    mlx_sys::try_eval(&[&values]).expect("position diagnostic evaluation");
    let shape = array.shape();
    let Some((axis, group)) = sequence_axis(&shape, chunk_len as i32) else {
        return;
    };
    std::fs::create_dir_all(root).unwrap();
    let data = values.data_f32();
    let stem = stage_stem(stage);
    let layer = layer_name();
    for local in locals {
        let position = offset + local;
        let (slice, out_shape) = extract_position(data, &shape, axis, group, local);
        let name = format!("layer-{layer}-{stem}-pos-{position}");
        write_f32le(&root.join(format!("{name}.f32le")), &slice);
        std::fs::write(
            root.join(format!("{name}.json")),
            serde_json::to_vec(&serde_json::json!({
                "shape": out_shape,
                "original_shape": shape,
                "dtype": format!("{:?}", array.dtype()),
                "layer": layer_label(),
                "stage": stage,
                "position": position,
                "sequence_axis": axis,
                "group": group,
                "chunk_start": offset,
                "chunk_len": chunk_len,
                "local_index": local,
                "forward": DUMP_FORWARD.get(),
            }))
            .unwrap(),
        )
        .unwrap();
    }
}

pub(crate) fn dump(stage: &'static str, arrays: &[&MlxArray]) {
    let Some(root) = dump_root() else {
        return;
    };
    if stage == "embedding" {
        begin_forward_dump();
    }
    if !should_dump(stage) {
        return;
    }
    let Some(array) = arrays.first() else {
        return;
    };
    if stage == "hc_scaled" {
        DUMP_HC.set(DUMP_HC.get() + 1);
    }
    if let Some(wanted) = dump_positions() {
        dump_selected(&root, stage, array, &wanted);
    } else {
        dump_full(&root, stage, array);
    }
}

/// Test-only n-gram row IDs and per-row hashes at dumped positions.
pub(crate) fn dump_ngram_lookup(token_count: usize, row_ids: &[u64], rows: &MlxArray) {
    if token_count == 0 || !row_ids.len().is_multiple_of(token_count) {
        return;
    }
    let Some(root) = dump_root() else {
        return;
    };
    let Some(wanted) = dump_positions() else {
        return;
    };
    if !should_dump("ngram_lookup") {
        return;
    }
    let heads = row_ids.len() / token_count;
    let chunk_len = DUMP_CHUNK_LEN.get();
    let offset = DUMP_OFFSET.get();
    if chunk_len == 0 || chunk_len != token_count {
        return;
    }
    let values = mlx_sys::astype(rows, MlxDtype::Float32, None);
    mlx_sys::try_eval(&[&values]).expect("n-gram lookup diagnostic evaluation");
    let data = values.data_f32();
    let width = if rows.shape().len() == 2 {
        rows.shape()[1] as usize
    } else {
        return;
    };
    if data.len() != token_count * heads * width {
        return;
    }
    std::fs::create_dir_all(&root).unwrap();
    let layer = layer_name();
    for pos in wanted {
        if pos < offset || pos >= offset + chunk_len {
            continue;
        }
        let local = pos - offset;
        let start = local * heads;
        let ids = &row_ids[start..start + heads];
        let mut hashes = Vec::with_capacity(heads);
        for head in 0..heads {
            let row_start = (start + head) * width;
            let row = &data[row_start..row_start + width];
            let mut hash = Sha256::new();
            for value in row {
                hash.update(value.to_le_bytes());
            }
            hashes.push(format!("{:x}", hash.finalize()));
        }
        let name = format!("layer-{layer}-ngram_lookup-pos-{pos}");
        std::fs::write(
            root.join(format!("{name}.json")),
            serde_json::to_vec_pretty(&serde_json::json!({
                "stage": "ngram_lookup",
                "layer": layer_label(),
                "position": pos,
                "heads_per_token": heads,
                "row_ids": ids,
                "row_sha256": hashes,
                "embedding_width": width,
                "chunk_start": offset,
                "chunk_len": chunk_len,
                "local_index": local,
                "forward": DUMP_FORWARD.get(),
            }))
            .unwrap(),
        )
        .unwrap();
    }
}

pub(crate) fn mark(stage: &'static str, arrays: &[&MlxArray]) {
    dump(stage, arrays);
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
fn dump_position_csv_parses_commas_and_spaces() {
    assert_eq!(parse_usize_csv("345,396, 479"), vec![345, 396, 479]);
    assert_eq!(
        parse_stage_csv("ple_rows,qsa_gather_indices"),
        vec!["ple_rows".to_string(), "qsa_gather_indices".to_string()]
    );
}

#[test]
fn default_position_stages_cover_attribution_pairs() {
    for stage in [
        "gdn",
        "qsa_output",
        "moe_compute",
        "attention_hc_read",
        "attention_hc_write",
        "mlp_hc_read",
        "mlp_hc_write",
        "ple_operator",
        "ngram_lookup",
        "ple_rows",
    ] {
        assert!(
            DEFAULT_POSITION_STAGES.contains(&stage),
            "{stage} must be in the records-attribution dump set"
        );
    }
}

#[test]
fn sequence_axis_prefers_batch_seq_layout() {
    assert_eq!(sequence_axis(&[1, 128, 2560], 128), Some((1, 1)));
    assert_eq!(sequence_axis(&[128, 151936], 128), Some((0, 1)));
    assert_eq!(sequence_axis(&[512, 64], 128), Some((0, 4)));
    assert_eq!(sequence_axis(&[4, 2560], 128), None);
}

#[test]
fn extract_position_takes_one_token_and_grouped_rows() {
    let hidden: Vec<f32> = (0..12).map(|value| value as f32).collect();
    let (slice, shape) = extract_position(&hidden, &[1, 4, 3], 1, 1, 2);
    assert_eq!(shape, vec![1, 1, 3]);
    assert_eq!(slice, vec![6.0, 7.0, 8.0]);
    let rows: Vec<f32> = (0..24).map(|value| value as f32).collect();
    let (grouped, grouped_shape) = extract_position(&rows, &[8, 3], 0, 2, 1);
    assert_eq!(grouped_shape, vec![2, 3]);
    assert_eq!(grouped, vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
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
