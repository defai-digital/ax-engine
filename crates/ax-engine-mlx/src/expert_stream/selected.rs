//! Bounded reads of complete affine expert matrices along axis zero.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use mlx_sys::{MlxDtype, RowTensorMeta, SafetensorsRowReader};

use super::{ExpertProj, ExpertStreamManifest, LayerExpertStack, is_quantization_sidecar_name};
use crate::weights::QuantizedWeight;

#[derive(Clone, Copy, Default)]
pub(crate) struct SelectedExpertReadStats {
    pub(crate) gathers: u64,
    pub(crate) payload_bytes: u64,
}

thread_local! {
    static READ_STATS: std::cell::Cell<SelectedExpertReadStats> = const {
        std::cell::Cell::new(SelectedExpertReadStats { gathers: 0, payload_bytes: 0 })
    };
}

pub(crate) fn take_selected_expert_read_stats() -> SelectedExpertReadStats {
    READ_STATS.with(|stats| stats.replace(SelectedExpertReadStats::default()))
}

struct TensorRows {
    reader: usize,
    name: String,
    meta: RowTensorMeta,
}

struct ProjectionRows {
    projection: ExpertProj,
    weight: TensorRows,
    scales: TensorRows,
    biases: TensorRows,
    bits: i32,
    group_size: i32,
}

pub(super) struct SelectedExpertRows {
    readers: Vec<SafetensorsRowReader>,
    projections: Vec<ProjectionRows>,
    experts: u32,
    bytes_per_expert: usize,
    max_gather_bytes: usize,
}

impl SelectedExpertRows {
    pub(super) fn open(
        manifest: &ExpertStreamManifest,
        root: &Path,
        layer: u32,
        max_gather_bytes: usize,
    ) -> Result<Self, String> {
        let tensors: Vec<_> = manifest.tensors_for_layer(layer).collect();
        let weights: Vec<_> = tensors
            .iter()
            .copied()
            .filter(|tensor| !is_quantization_sidecar_name(&tensor.name))
            .collect();
        if weights.is_empty() {
            return Err(format!(
                "selected experts: no projections for layer {layer}"
            ));
        }
        let mut selections = BTreeMap::<PathBuf, Vec<String>>::new();
        let mut locations = HashMap::new();
        for tensor in &weights {
            let base = tensor.name.strip_suffix(".weight").ok_or_else(|| {
                format!("selected experts: unsupported weight name {}", tensor.name)
            })?;
            for name in [
                tensor.name.clone(),
                format!("{base}.scales"),
                format!("{base}.biases"),
            ] {
                let file = tensors
                    .iter()
                    .find(|entry| entry.name == name)
                    .map_or(&tensor.file, |entry| &entry.file);
                if locations.insert(name.clone(), file.clone()).is_some() {
                    return Err(format!("selected experts: duplicate tensor {name}"));
                }
                selections.entry(file.clone()).or_default().push(name);
            }
        }
        let mut readers = Vec::new();
        let mut file_indices = HashMap::new();
        for (file, names) in selections {
            let names: Vec<&str> = names.iter().map(String::as_str).collect();
            let reader = SafetensorsRowReader::open_selected_stacks(
                &root.join(&file),
                &names,
                max_gather_bytes,
            )?;
            file_indices.insert(file, readers.len());
            readers.push(reader);
        }
        let lookup = |name: String| -> Result<TensorRows, String> {
            let file = locations
                .get(&name)
                .ok_or_else(|| format!("selected experts: missing location {name}"))?;
            let index = *file_indices
                .get(file)
                .ok_or_else(|| "selected experts: missing reader".to_string())?;
            let meta = readers[index]
                .tensor_meta(&name)
                .ok_or_else(|| format!("selected experts: missing tensor {name}"))?;
            Ok(TensorRows {
                reader: index,
                name,
                meta,
            })
        };
        let mut projections = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut bytes_per_expert = 0usize;
        for tensor in weights {
            let projection = tensor
                .parsed_proj
                .ok_or_else(|| "selected experts: unvalidated projection".to_string())?;
            if !seen.insert(projection) {
                return Err("selected experts: duplicate projection".into());
            }
            if tensor.expert_axis != 0
                || tensor.num_experts != manifest.num_experts
                || !matches!((tensor.bits, tensor.group_size), (2, 32) | (4 | 6, 64))
            {
                return Err("selected experts: unsupported affine expert layout".into());
            }
            let base = tensor
                .name
                .strip_suffix(".weight")
                .ok_or_else(|| "selected experts: missing weight suffix".to_string())?;
            let weight = lookup(tensor.name.clone())?;
            let scales = lookup(format!("{base}.scales"))?;
            let biases = lookup(format!("{base}.biases"))?;
            if weight.meta.dtype != MlxDtype::Uint32
                || !matches!(
                    scales.meta.dtype,
                    MlxDtype::Float32 | MlxDtype::Float16 | MlxDtype::Bfloat16
                )
                || scales.meta.dtype != biases.meta.dtype
                || scales.meta.shape != biases.meta.shape
                || weight.meta.shape[..2] != scales.meta.shape[..2]
                || weight.meta.rows != manifest.num_experts as usize
                || u64::from(weight.meta.shape[2] as u32) * 32
                    != u64::from(scales.meta.shape[2] as u32)
                        * u64::from(tensor.group_size)
                        * u64::from(tensor.bits)
            {
                return Err(format!(
                    "selected experts: incompatible affine triplet {}",
                    tensor.name
                ));
            }
            for row in [&weight, &scales, &biases] {
                bytes_per_expert = bytes_per_expert
                    .checked_add(row.meta.row_bytes)
                    .ok_or_else(|| "selected experts: byte count overflow".to_string())?;
            }
            projections.push(ProjectionRows {
                projection,
                weight,
                scales,
                biases,
                bits: tensor.bits as i32,
                group_size: tensor.group_size as i32,
            });
        }
        Ok(Self {
            readers,
            projections,
            experts: manifest.num_experts,
            bytes_per_expert,
            max_gather_bytes,
        })
    }

    pub(super) fn payload_bytes_read(&self) -> u64 {
        self.readers
            .iter()
            .map(SafetensorsRowReader::payload_bytes_read)
            .sum()
    }

    fn requested_payload_bytes(&self, ids: &[u64]) -> Result<usize, String> {
        if ids.is_empty() || ids.iter().any(|&id| id >= u64::from(self.experts)) {
            return Err("selected experts: empty or out-of-range expert IDs".into());
        }
        ids.len()
            .checked_mul(self.bytes_per_expert)
            .ok_or_else(|| "selected experts: output byte count overflow".to_string())
    }

    /// A capacity miss is the only condition that permits a whole-layer fallback.
    pub(super) fn gather_if_fits(&self, ids: &[u64]) -> Result<Option<LayerExpertStack>, String> {
        let bytes = self.requested_payload_bytes(ids)?;
        if bytes > self.max_gather_bytes {
            return Ok(None);
        }
        self.gather_validated(ids, bytes).map(Some)
    }

    pub(super) fn gather(&self, ids: &[u64]) -> Result<LayerExpertStack, String> {
        let bytes = self.requested_payload_bytes(ids)?;
        if bytes > self.max_gather_bytes {
            return Err(format!(
                "selected experts: output {bytes} bytes exceeds budget {}",
                self.max_gather_bytes
            ));
        }
        self.gather_validated(ids, bytes)
    }

    fn gather_validated(&self, ids: &[u64], bytes: usize) -> Result<LayerExpertStack, String> {
        let read = |rows: &TensorRows| self.readers[rows.reader].gather_rows(&rows.name, ids);
        let mut stack = LayerExpertStack::default();
        for projection in &self.projections {
            let weight = QuantizedWeight {
                weight: read(&projection.weight)?,
                scales: Some(read(&projection.scales)?),
                biases: Some(read(&projection.biases)?),
                bits: projection.bits,
                group_size: projection.group_size,
                mode: "affine".into(),
                linear_bias: None,
                decode_weight_t: None,
                decode_q2_weight: None,
                decode_q2_scales: None,
                decode_q2_biases: None,
            };
            stack.insert(projection.projection, weight);
        }
        READ_STATS.with(|stats| {
            let previous = stats.get();
            stats.set(SelectedExpertReadStats {
                gathers: previous.gathers.saturating_add(1),
                payload_bytes: previous.payload_bytes.saturating_add(bytes as u64),
            });
        });
        Ok(stack)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::expert_stream::{ExpertLayerSource, ExpertStackPager};
    use crate::model::shared::ProjectionBatchPolicy;
    use crate::model::shared::qwen4_exp_moe::{
        Qwen4ExpExpertWeights, Qwen4ExpMoe, Qwen4ExpMoeWeights, Qwen4ExpResidentExperts,
    };
    use mlx_sys::{MlxArray, MlxQuantizationMode, astype, eval, quantize, reshape};
    use std::sync::Arc;

    fn dense(shape: &[i32], shift: usize, dtype: MlxDtype) -> MlxArray {
        let size = shape.iter().product::<i32>() as usize;
        let values: Vec<f32> = (0..size)
            .map(|i| (((i * 17 + shift) % 97) as f32 - 48.0) / 251.0)
            .collect();
        astype(
            &reshape(&MlxArray::from_f32_slice(&values), shape, None),
            dtype,
            None,
        )
    }

    fn fixture(
        root: &Path,
        bits: i32,
        group: i32,
        dtype: MlxDtype,
    ) -> (ExpertStreamManifest, Qwen4ExpResidentExperts) {
        let mut entries = Vec::new();
        let mut weights = Vec::new();
        for (offset, projection) in ["gate", "up", "down"].into_iter().enumerate() {
            let parts = quantize(
                &dense(&[8, 64, 64], offset * 7, dtype),
                Some(group),
                Some(bits),
                MlxQuantizationMode::Affine,
                None,
                None,
            );
            eval(&parts.iter().collect::<Vec<_>>());
            for (kind, array) in ["weight", "scales", "biases"].iter().zip(&parts) {
                let name = format!("expert.{projection}.{kind}");
                let file = format!("{projection}-{kind}.safetensors");
                let (storage, payload): (&str, Vec<u8>) = if *kind == "weight" {
                    (
                        "U32",
                        array
                            .data_u32()
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect(),
                    )
                } else {
                    let fp32 = astype(array, MlxDtype::Float32, None);
                    eval(&[&fp32]);
                    if dtype == MlxDtype::Bfloat16 {
                        (
                            "BF16",
                            fp32.data_f32()
                                .iter()
                                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                                .collect(),
                        )
                    } else {
                        (
                            "F32",
                            fp32.data_f32()
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect(),
                        )
                    }
                };
                let header = serde_json::to_vec(&serde_json::json!({name.clone(): {
                    "dtype": storage, "shape": array.shape(), "data_offsets": [0,payload.len()]
                }}))
                .unwrap();
                let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
                bytes.extend(header);
                bytes.extend(payload);
                std::fs::write(root.join(&file), bytes).unwrap();
                entries.push(
                    serde_json::json!({"name":name,"file":file,"layer":0,"proj":projection,
                    "expert_axis":0,"num_experts":8,"bits":bits,"group_size":group}),
                );
            }
            let mut weight = QuantizedWeight::new(
                parts[0].clone(),
                Some(parts[1].clone()),
                Some(parts[2].clone()),
            );
            weight.bits = bits;
            weight.group_size = group;
            weights.push(weight);
        }
        let manifest = ExpertStreamManifest::parse(
            &serde_json::to_vec(&serde_json::json!({
                "schema_version":"axquant.expert-stream.v1", "mode":"layer-stack", "required":true,
                "num_experts":8, "experts_per_tok":4, "tensors":entries
            }))
            .unwrap(),
        )
        .unwrap();
        (
            manifest,
            Qwen4ExpResidentExperts {
                gate: weights[0].clone(),
                up: weights[1].clone(),
                down: weights[2].clone(),
            },
        )
    }

    fn module(experts: Qwen4ExpExpertWeights, dtype: MlxDtype) -> Qwen4ExpMoe {
        let projection =
            |shape: &[i32], shift| QuantizedWeight::new(dense(shape, shift, dtype), None, None);
        Qwen4ExpMoe::new(
            64,
            64,
            64,
            8,
            4,
            true,
            Qwen4ExpMoeWeights {
                router: projection(&[8, 64], 2),
                experts,
                shared_gate: projection(&[64, 64], 3),
                shared_up: projection(&[64, 64], 5),
                shared_down: projection(&[64, 64], 7),
                shared_router: projection(&[1, 64], 11),
            },
        )
        .unwrap()
    }

    fn equal(a: &MlxArray, b: &MlxArray) {
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.dtype(), b.dtype());
        if a.dtype() == MlxDtype::Uint32 {
            eval(&[a, b]);
            assert_eq!(a.data_u32(), b.data_u32());
        } else {
            let a = astype(a, MlxDtype::Float32, None);
            let b = astype(b, MlxDtype::Float32, None);
            eval(&[&a, &b]);
            assert_eq!(a.data_f32(), b.data_f32());
        }
    }

    #[test]
    fn selected_prefill_union_and_capacity_fallback_preserve_shared_outputs() {
        for dtype in [MlxDtype::Float32, MlxDtype::Bfloat16] {
            for (bits, group) in [(2, 32), (4, 64), (6, 64)] {
                let root = std::env::temp_dir().join(format!(
                    "ax-selected-prefill-{}-{dtype:?}-{bits}",
                    std::process::id()
                ));
                std::fs::create_dir_all(&root).unwrap();
                let (manifest, resident) = fixture(&root, bits, group, dtype);
                let reader = SelectedExpertRows::open(&manifest, &root, 0, 1 << 20).unwrap();
                let row_bytes = reader.bytes_per_expert;
                let exact = SelectedExpertRows::open(&manifest, &root, 0, 6 * row_bytes).unwrap();
                assert!(exact.gather_if_fits(&[7, 0, 4, 1, 7, 2]).unwrap().is_some());
                let before = exact.payload_bytes_read();
                assert!(
                    exact
                        .gather_if_fits(&[7, 0, 4, 1, 7, 2, 3])
                        .unwrap()
                        .is_none()
                );
                assert_eq!(exact.payload_bytes_read(), before);
                assert!(exact.gather_if_fits(&[8]).is_err());
                assert!(exact.gather_if_fits(&[]).is_err());
                assert_eq!(exact.payload_bytes_read(), before);

                let pager = Arc::new(ExpertStackPager::new(Arc::new(manifest), root.clone(), 1));
                let source = Arc::new(ExpertLayerSource::new(Arc::clone(&pager), 0));
                let resident = module(Qwen4ExpExpertWeights::Resident(Box::new(resident)), dtype);
                let mut selected = module(Qwen4ExpExpertWeights::Streamed(source), dtype);
                selected.enable_selected_prefill_for_test();
                let input = dense(&[1, 2, 64], 31, dtype);
                equal(
                    &resident
                        .forward(&input, ProjectionBatchPolicy::Shared)
                        .unwrap(),
                    &selected
                        .forward(&input, ProjectionBatchPolicy::Shared)
                        .unwrap(),
                );
                assert_eq!(
                    pager.selected_payload_bytes_read().unwrap(),
                    0,
                    "the prefill flag alone must not enable selected reads"
                );
                selected.enable_selected_decode_for_test();
                let mut saw_compact_union = false;
                let mut saw_multi_row_union = false;
                for tokens in [2, 3, 7] {
                    for shift in [1, 7, 31, 51] {
                        let input = dense(&[1, tokens, 64], shift, dtype);
                        let before = pager.selected_payload_bytes_read().unwrap();
                        equal(
                            &resident
                                .forward(&input, ProjectionBatchPolicy::Shared)
                                .unwrap(),
                            &selected
                                .forward(&input, ProjectionBatchPolicy::Shared)
                                .unwrap(),
                        );
                        let rows = (pager.selected_payload_bytes_read().unwrap() - before)
                            / row_bytes as u64;
                        assert!((4..=8).contains(&rows));
                        saw_compact_union |= rows < 8;
                        saw_multi_row_union |= rows > 4;
                    }
                }
                assert!(saw_compact_union && saw_multi_row_union);
                let before = pager.selected_payload_bytes_read().unwrap();
                let input = dense(&[1, 3, 64], 31, dtype);
                equal(
                    &resident
                        .forward(&input, ProjectionBatchPolicy::RowExact)
                        .unwrap(),
                    &selected
                        .forward(&input, ProjectionBatchPolicy::RowExact)
                        .unwrap(),
                );
                let batch = dense(&[2, 3, 64], 17, dtype);
                equal(
                    &resident
                        .forward(&batch, ProjectionBatchPolicy::Shared)
                        .unwrap(),
                    &selected
                        .forward(&batch, ProjectionBatchPolicy::Shared)
                        .unwrap(),
                );
                assert_eq!(pager.selected_payload_bytes_read().unwrap(), before);
                assert_eq!(pager.selected_prefill_capacity_fallback_layers(), 0);

                let small =
                    Arc::new(SelectedExpertRows::open(pager.manifest(), &root, 0, 1).unwrap());
                pager
                    .selected_readers
                    .lock()
                    .unwrap()
                    .insert(0, Arc::clone(&small));
                equal(
                    &resident
                        .forward(&input, ProjectionBatchPolicy::Shared)
                        .unwrap(),
                    &selected
                        .forward(&input, ProjectionBatchPolicy::Shared)
                        .unwrap(),
                );
                assert_eq!(small.payload_bytes_read(), 0);
                assert_eq!(pager.selected_prefill_capacity_fallback_layers(), 1);
                assert_eq!(pager.cached_layer_count(), 1);
                let singleton = dense(&[1, 1, 64], 5, dtype);
                assert!(
                    selected
                        .forward(&singleton, ProjectionBatchPolicy::Shared)
                        .is_err(),
                    "singleton retains its strict cap error"
                );

                pager
                    .selected_readers
                    .lock()
                    .unwrap()
                    .insert(0, Arc::new(reader));
                let file = root.join("up-biases.safetensors");
                let saved = std::fs::read(&file).unwrap();
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(&file)
                    .unwrap()
                    .set_len(0)
                    .unwrap();
                let error = selected
                    .forward(&input, ProjectionBatchPolicy::Shared)
                    .err()
                    .unwrap();
                assert!(
                    error.contains("read row"),
                    "I/O error must not use the cached whole layer: {error}"
                );
                std::fs::write(file, saved).unwrap();
                equal(
                    &resident
                        .forward(&input, ProjectionBatchPolicy::Shared)
                        .unwrap(),
                    &selected
                        .forward(&input, ProjectionBatchPolicy::Shared)
                        .unwrap(),
                );
                std::fs::remove_dir_all(root).unwrap();
            }
        }
    }

    #[test]
    fn selected_affine_rows_and_moe_match_resident_for_all_audited_formats() {
        for dtype in [MlxDtype::Float32, MlxDtype::Bfloat16] {
            for (bits, group) in [(2, 32), (4, 64), (6, 64)] {
                let root = std::env::temp_dir().join(format!(
                    "ax-selected-{}-{dtype:?}-{bits}",
                    std::process::id()
                ));
                std::fs::create_dir_all(&root).unwrap();
                let (manifest, resident) = fixture(&root, bits, group, dtype);
                let reader = SelectedExpertRows::open(&manifest, &root, 0, 1 << 20).unwrap();
                assert_eq!(reader.payload_bytes_read(), 0);
                let _ = take_selected_expert_read_stats();
                let stack = reader.gather(&[7, 2, 7, 5]).unwrap();
                let stats = take_selected_expert_read_stats();
                assert_eq!(stats.gathers, 1);
                assert_eq!(stats.payload_bytes, (4 * reader.bytes_per_expert) as u64);
                assert_eq!(take_selected_expert_read_stats().gathers, 0);
                let ids = [7i32, 2, 7, 5];
                let ids = MlxArray::from_raw_data(ids.as_ptr().cast(), 16, &[4], MlxDtype::Int32);
                for (selected, original) in [
                    (stack.gate_exps.as_ref().unwrap(), &resident.gate),
                    (stack.up_exps.as_ref().unwrap(), &resident.up),
                    (stack.down_exps.as_ref().unwrap(), &resident.down),
                ] {
                    for (a, b) in [
                        (&selected.weight, &original.weight),
                        (
                            selected.scales.as_ref().unwrap(),
                            original.scales.as_ref().unwrap(),
                        ),
                        (
                            selected.biases.as_ref().unwrap(),
                            original.biases.as_ref().unwrap(),
                        ),
                    ] {
                        equal(a, &mlx_sys::take(b, &ids, 0, None));
                    }
                }
                assert_eq!(
                    reader.payload_bytes_read(),
                    (4 * reader.bytes_per_expert) as u64
                );
                let pager = Arc::new(ExpertStackPager::new(Arc::new(manifest), root.clone(), 1));
                let source = Arc::new(ExpertLayerSource::new(Arc::clone(&pager), 0));
                let resident = module(Qwen4ExpExpertWeights::Resident(Box::new(resident)), dtype);
                let mut selected = module(Qwen4ExpExpertWeights::Streamed(source), dtype);
                selected.enable_selected_decode_for_test();
                for shift in [4, 29] {
                    let input = dense(&[1, 1, 64], shift, dtype);
                    equal(
                        &resident
                            .forward(&input, ProjectionBatchPolicy::Shared)
                            .unwrap(),
                        &selected
                            .forward(&input, ProjectionBatchPolicy::Shared)
                            .unwrap(),
                    );
                }
                assert!(pager.selected_payload_bytes_read().unwrap() > 0);
                assert_eq!(pager.cached_layer_count(), 0);
                let bytes = pager.selected_payload_bytes_read().unwrap();
                let input = dense(&[1, 3, 64], 31, dtype);
                equal(
                    &resident
                        .forward(&input, ProjectionBatchPolicy::RowExact)
                        .unwrap(),
                    &selected
                        .forward(&input, ProjectionBatchPolicy::RowExact)
                        .unwrap(),
                );
                assert_eq!(pager.selected_payload_bytes_read().unwrap(), bytes);
                assert_eq!(pager.cached_layer_count(), 1);
                let input = dense(&[2, 1, 64], 43, dtype);
                equal(
                    &resident
                        .forward(&input, ProjectionBatchPolicy::Shared)
                        .unwrap(),
                    &selected
                        .forward(&input, ProjectionBatchPolicy::Shared)
                        .unwrap(),
                );
                assert_eq!(pager.selected_payload_bytes_read().unwrap(), bytes);
                let before = reader.payload_bytes_read();
                assert!(reader.gather(&[0, 8]).is_err());
                assert_eq!(reader.payload_bytes_read(), before);
                let small = SelectedExpertRows::open(pager.manifest(), &root, 0, 1).unwrap();
                assert!(small.gather(&[0]).is_err());
                assert_eq!(small.payload_bytes_read(), 0);
                let path = root.join("up-biases.safetensors");
                let saved = std::fs::read(&path).unwrap();
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .unwrap()
                    .set_len(0)
                    .unwrap();
                assert!(reader.gather(&[0]).is_err());
                std::fs::write(path, saved).unwrap();
                let recovered = reader.gather(&[7, 2, 7, 5]).unwrap();
                equal(
                    &recovered.up_exps.unwrap().weight,
                    &stack.up_exps.unwrap().weight,
                );
                std::fs::remove_dir_all(root).unwrap();
            }
        }
    }
}
