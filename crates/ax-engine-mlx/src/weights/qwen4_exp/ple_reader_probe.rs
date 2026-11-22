//! Test-only probe for the native PLE staging path on the pinned pack.

#![allow(clippy::unwrap_used)]

use super::*;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::time::Instant;

#[derive(Deserialize)]
struct Case {
    shape: String,
    chunk_tokens: usize,
    history_tokens: Vec<u32>,
    tokens: Vec<u32>,
}

fn load_metadata(
    root: &Path,
    specs: &[NativeTensorSpec],
    role: NativeTensorRole,
    layer: u32,
) -> Vec<u64> {
    let spec = specs
        .iter()
        .find(|spec| spec.layer_index == Some(layer) && spec.role == role)
        .unwrap();
    let selected = HashSet::from([spec.name.clone()]);
    let mut values = load_safetensors_filtered(
        &root.join(&spec.file),
        SafetensorsNameFilter::Keep(&selected),
    )
    .unwrap();
    let array = values.remove(&spec.name).unwrap();
    read_nonneg_i64_vec(&array, &spec.name).unwrap()
}

fn sample(table: &NgramTable, shape: &str, lookups: &[Vec<u64>], phase: &str) -> Value {
    let mut duration = Vec::with_capacity(lookups.len());
    let mut digest = Sha256::new();
    let initial = table.payload_bytes_read();
    for ids in lookups {
        let started = Instant::now();
        let planes = table.read_planes_for_probe(ids).unwrap();
        duration.push(started.elapsed().as_nanos() as u64);
        assert_eq!(planes.len(), 3);
        for plane in planes {
            digest.update((plane.len() as u64).to_le_bytes());
            digest.update(plane);
        }
    }
    json!({
        "shape": shape, "phase": phase, "elapsed_ns": duration, "payload_sha256": format!("{:x}", digest.finalize()),
        "bytes_read": table.payload_bytes_read() - initial,
        "rows_requested": lookups.iter().map(Vec::len).sum::<usize>(),
    })
}

#[test]
#[ignore = "requires the pinned Flash Next MXFP4 PLE table and frozen token fixture"]
fn flash_next_real_ple_reader_dispatch() {
    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_REAL_PACK").unwrap());
    let workload = std::fs::read(std::env::var_os("AX_FLASH_NEXT_PLE_WORKLOAD").unwrap()).unwrap();
    let cases: Vec<Case> = serde_json::from_slice(&workload).unwrap();
    let artifacts = NativeModelArtifacts::from_dir(&root).unwrap();
    let manifest = artifacts.manifest();
    assert_eq!(manifest.model_family, "qwen4_exp");
    assert_eq!(manifest.qwen4_exp.ple_layer_ids.len(), 1);
    // Pack placement uses one-based layer IDs and tensor specs are zero-based.
    let layer = manifest.qwen4_exp.ple_layer_ids[0].checked_sub(1).unwrap();
    let specs = &manifest.tensors;
    let heads = manifest.qwen4_exp.heads_per_ngram.unwrap() as usize;
    let order = manifest.qwen4_exp.ngram_size.unwrap() as usize;
    let width = manifest.qwen4_exp.ple_embed_dim.unwrap() as usize / ((order - 1) * heads);
    assert_eq!((heads * (order - 1), width), (16, 160));
    let table =
        NgramTable::open(&root, specs, layer, 128, width, DEFAULT_MAX_GATHER_BYTES).unwrap();
    let layout = NgramLayout::new(
        manifest.vocab_size,
        resolve_eos_token(&root, manifest.vocab_size).unwrap(),
        load_metadata(
            &root,
            specs,
            NativeTensorRole::Qwen4ExpPleMultipliers,
            layer,
        ),
        heads,
        load_metadata(
            &root,
            specs,
            NativeTensorRole::Qwen4ExpPleHeadVocabSizes,
            layer,
        ),
        load_metadata(
            &root,
            specs,
            NativeTensorRole::Qwen4ExpPleHeadOffsets,
            layer,
        ),
        table.rows(),
    )
    .unwrap();

    let reports: Vec<_> = ["decode", "prefill"]
        .into_iter()
        .flat_map(|shape| {
            let lookups: Vec<_> = cases
                .iter()
                .filter(|case| case.shape == shape)
                .flat_map(|case| {
                    assert_eq!(case.tokens.len() % case.chunk_tokens, 0);
                    let mut history = layout
                        .plan(&layout.initial_history(), &case.history_tokens)
                        .unwrap()
                        .next_history;
                    case.tokens
                        .chunks_exact(case.chunk_tokens)
                        .map(|tokens| {
                            let next = layout.plan(&history, tokens).unwrap();
                            history = next.next_history;
                            next.rows
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            assert!(!lookups.is_empty());
            let initial = sample(&table, shape, &lookups, "initial");
            let warm = sample(&table, shape, &lookups, "repeated");
            assert_eq!(initial["payload_sha256"], warm["payload_sha256"]);
            assert_eq!(initial["bytes_read"], warm["bytes_read"]);
            [initial, warm]
        })
        .collect();
    let record = json!({
        "reader_threads": crate::fastpath::flash_next_ngram_read_threads(),
        "reader_pool_active": table.reader_pool_active_for_probe(),
        "workload_sha256": format!("{:x}", Sha256::digest(&workload)),
        "unique_table_shards": 128, "embedding_width": width, "stages": reports,
    });
    println!("PLE_READ_PROBE {record}");
}
