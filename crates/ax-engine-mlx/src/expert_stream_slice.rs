//! Per-expert row slicing for streamed packed expert stacks (pure Rust).
//!
//! [`crate::expert_stream::ExpertRowPager`] pages individual experts out of
//! packed `[E, ...]` safetensors tensors. This module parses each shard's JSON
//! header once (cached), computes axis-0 row byte ranges, and reads
//! single-expert rows with `pread` into fresh `MlxArray`s. No mapping is held
//! across calls, so streamed shards never pin pages the pager cannot evict.

use std::collections::HashMap;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use mlx_sys::{MlxArray, MlxDtype};

use crate::expert_stream::ExpertStreamTensor;

/// Map a safetensors dtype string to the array dtype (same convention as the
/// mlx-sys loader: FP8 payloads arrive as raw `Uint8` byte containers).
/// Kept local so per-expert paging needs no mlx-sys surface changes.
fn parse_safetensors_dtype(s: &str) -> Option<MlxDtype> {
    Some(match s {
        "F32" | "FLOAT32" => MlxDtype::Float32,
        "F16" | "FLOAT16" => MlxDtype::Float16,
        "BF16" | "BFLOAT16" => MlxDtype::Bfloat16,
        "F8_E4M3" | "F8_E8M0" => MlxDtype::Uint8,
        "I8" | "INT8" => MlxDtype::Int8,
        "I16" | "INT16" => MlxDtype::Int16,
        "I32" | "INT32" => MlxDtype::Int32,
        "I64" | "INT64" => MlxDtype::Int64,
        "U8" | "UINT8" => MlxDtype::Uint8,
        "U16" | "UINT16" => MlxDtype::Uint16,
        "U32" | "UINT32" => MlxDtype::Uint32,
        "U64" | "UINT64" => MlxDtype::Uint64,
        "BOOL" => MlxDtype::Bool,
        _ => return None,
    })
}

struct TensorMeta {
    dtype: MlxDtype,
    shape: Vec<i32>,
    /// Absolute file offsets of the tensor payload.
    start: usize,
    end: usize,
}

/// Parsed safetensors header of one shard.
pub struct ShardHeader {
    tensors: HashMap<String, TensorMeta>,
}

impl ShardHeader {
    fn tensor(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.get(name)
    }
}

/// Lazily parsed per-shard headers, shared by every pager read.
#[derive(Default)]
pub struct ShardHeaderCache {
    inner: Mutex<HashMap<PathBuf, Arc<ShardHeader>>>,
}

impl ShardHeaderCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn header(&self, path: &Path) -> Result<Arc<ShardHeader>, String> {
        let mut cache = self.inner.lock().expect("shard header cache lock");
        if let Some(header) = cache.get(path) {
            return Ok(header.clone());
        }
        let header = Arc::new(parse_shard_header(path)?);
        cache.insert(path.to_path_buf(), header.clone());
        Ok(header)
    }
}

fn parse_shard_header(path: &Path) -> Result<ShardHeader, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let file_len = file
        .metadata()
        .map_err(|e| format!("stat {}: {e}", path.display()))?
        .len() as usize;
    let mut prefix = [0u8; 8];
    file.read_exact_at(&mut prefix, 0)
        .map_err(|e| format!("read header length of {}: {e}", path.display()))?;
    let header_len = usize::try_from(u64::from_le_bytes(prefix)).map_err(|_| {
        format!(
            "safetensors header length does not fit usize in {}",
            path.display()
        )
    })?;
    let data_base = 8usize
        .checked_add(header_len)
        .ok_or_else(|| format!("safetensors header length overflows in {}", path.display()))?;
    if data_base > file_len {
        return Err(format!(
            "safetensors header length {header_len} exceeds file size {file_len} in {}",
            path.display()
        ));
    }
    let mut json_bytes = vec![0u8; header_len];
    file.read_exact_at(&mut json_bytes, 8)
        .map_err(|e| format!("read header of {}: {e}", path.display()))?;
    let header: serde_json::Value = serde_json::from_slice(&json_bytes)
        .map_err(|e| format!("parse safetensors header in {}: {e}", path.display()))?;
    let obj = header.as_object().ok_or_else(|| {
        format!(
            "safetensors header is not a JSON object in {}",
            path.display()
        )
    })?;

    let mut tensors = HashMap::new();
    for (name, entry) in obj {
        if name == "__metadata__" {
            continue;
        }
        let entry_obj = entry
            .as_object()
            .ok_or_else(|| format!("tensor entry {name} is not an object"))?;
        let dtype_str = entry_obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("tensor entry {name} missing dtype"))?;
        let dtype = parse_safetensors_dtype(dtype_str)
            .ok_or_else(|| format!("tensor entry {name}: unsupported dtype {dtype_str}"))?;
        let shape: Vec<i32> = entry_obj
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("tensor entry {name} missing shape"))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| format!("shape dim is not a u64 in {name}"))
                    .map(|x| x as i32)
            })
            .collect::<Result<_, _>>()?;
        let offsets = entry_obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("tensor entry {name} missing data_offsets"))?;
        if offsets.len() != 2 {
            return Err(format!(
                "tensor entry {name} data_offsets must have 2 elements"
            ));
        }
        let start = offsets[0]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .and_then(|v| data_base.checked_add(v))
            .ok_or_else(|| format!("tensor entry {name}: invalid data_offsets[0]"))?;
        let end = offsets[1]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .and_then(|v| data_base.checked_add(v))
            .ok_or_else(|| format!("tensor entry {name}: invalid data_offsets[1]"))?;
        if start > end || end > file_len {
            return Err(format!(
                "tensor entry {name}: data_offsets [{start},{end}] out of bounds (file {} bytes)",
                path.display()
            ));
        }
        tensors.insert(
            name.clone(),
            TensorMeta {
                dtype,
                shape,
                start,
                end,
            },
        );
    }
    Ok(ShardHeader { tensors })
}

/// One expert's rows sliced out of a packed `[E, ...]` tensor and its
/// quantization sidecars. Every array keeps a leading singleton axis
/// (`[1, ...]`) so per-expert rows concatenate back into a compacted stack.
pub struct ExpertProjRow {
    pub weight: MlxArray,
    pub scales: Option<MlxArray>,
    pub biases: Option<MlxArray>,
    pub linear_bias: Option<MlxArray>,
}

impl ExpertProjRow {
    /// Resident bytes accounted against the pager budget.
    pub fn bytes(&self) -> usize {
        self.weight.nbytes()
            + self.scales.as_ref().map_or(0, |a| a.nbytes())
            + self.biases.as_ref().map_or(0, |a| a.nbytes())
            + self.linear_bias.as_ref().map_or(0, |a| a.nbytes())
    }
}

fn read_row(
    file: &std::fs::File,
    meta: &TensorMeta,
    name: &str,
    expert: u32,
) -> Result<MlxArray, String> {
    let num_experts = meta.shape.first().copied().unwrap_or(0);
    if num_experts <= 0 {
        return Err(format!("tensor {name} has no expert axis"));
    }
    if expert as i32 >= num_experts {
        return Err(format!(
            "expert {expert} out of range for tensor {name} ({num_experts} experts)"
        ));
    }
    let total = meta.end - meta.start;
    let row_bytes = total / num_experts as usize;
    if row_bytes == 0 || !total.is_multiple_of(num_experts as usize) {
        return Err(format!(
            "tensor {name}: payload {total} bytes is not divisible into {num_experts} expert rows"
        ));
    }
    let offset = meta.start + expert as usize * row_bytes;
    let mut buf = vec![0u8; row_bytes];
    file.read_exact_at(&mut buf, offset as u64)
        .map_err(|e| format!("pread expert {expert} of {name}: {e}"))?;
    let mut row_shape = Vec::with_capacity(meta.shape.len());
    row_shape.push(1);
    row_shape.extend_from_slice(&meta.shape[1..]);
    Ok(MlxArray::from_raw_data(
        buf.as_ptr(),
        buf.len(),
        &row_shape,
        meta.dtype,
    ))
}

fn read_sidecar_row(
    file: &std::fs::File,
    header: &ShardHeader,
    name: &str,
    expert: u32,
    num_experts: u32,
) -> Result<Option<MlxArray>, String> {
    let Some(meta) = header.tensor(name) else {
        return Ok(None);
    };
    if meta.shape.first().copied().unwrap_or(0) != num_experts as i32 {
        return Err(format!(
            "sidecar {name} is not expert-major (shape {:?}, expected {num_experts} experts); \
             per-expert paging cannot slice it",
            meta.shape
        ));
    }
    read_row(file, meta, name, expert).map(Some)
}

/// Read the rows of `experts` (in request order) for one manifest tensor and
/// its `.scales` / `.biases` / `.bias` sidecars from the same shard.
pub fn read_expert_proj_rows(
    root: &Path,
    headers: &ShardHeaderCache,
    tensor: &ExpertStreamTensor,
    experts: &[u32],
) -> Result<Vec<ExpertProjRow>, String> {
    let path = root.join(&tensor.file);
    let header = headers.header(&path)?;
    let meta = header.tensor(&tensor.name).ok_or_else(|| {
        format!(
            "tensor {} missing from shard {}",
            tensor.name,
            path.display()
        )
    })?;
    if meta.shape.first().copied().unwrap_or(0) != tensor.num_experts as i32 {
        return Err(format!(
            "tensor {}: manifest expects {} experts, shard shape is {:?}",
            tensor.name, tensor.num_experts, meta.shape
        ));
    }
    let base = tensor
        .name
        .strip_suffix(".weight")
        .unwrap_or(tensor.name.as_str());
    let scales_name = format!("{base}.scales");
    let biases_name = format!("{base}.biases");
    let bias_name = format!("{base}.bias");

    let file = std::fs::File::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut rows = Vec::with_capacity(experts.len());
    for &expert in experts {
        let weight = read_row(&file, meta, &tensor.name, expert)?;
        let scales = read_sidecar_row(&file, &header, &scales_name, expert, tensor.num_experts)?;
        let biases = read_sidecar_row(&file, &header, &biases_name, expert, tensor.num_experts)?;
        let linear_bias = read_sidecar_row(&file, &header, &bias_name, expert, tensor.num_experts)?;
        rows.push(ExpertProjRow {
            weight,
            scales,
            biases,
            linear_bias,
        });
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_stream::ExpertProj;

    fn write_safetensors_f32(
        dir: &Path,
        file_name: &str,
        tensors: &[(&str, Vec<i32>, Vec<f32>)],
    ) -> PathBuf {
        let mut header = serde_json::Map::new();
        let mut data: Vec<u8> = Vec::new();
        for (name, shape, values) in tensors {
            let start = data.len();
            for value in values {
                data.extend_from_slice(&value.to_le_bytes());
            }
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [start, data.len()],
                }),
            );
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&data);
        let path = dir.join(file_name);
        std::fs::write(&path, &bytes).unwrap();
        path
    }

    fn fixture_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ax_expert_slice_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn manifest_tensor(name: &str, num_experts: u32) -> ExpertStreamTensor {
        ExpertStreamTensor {
            name: name.to_string(),
            file: PathBuf::from("experts.safetensors"),
            layer: 0,
            proj: "gate_up".into(),
            expert_axis: 0,
            num_experts,
            bits: 2,
            group_size: 64,
            parsed_proj: Some(ExpertProj::GateUp),
        }
    }

    fn filled_expert_values(experts: i32, out: i32, inn: i32) -> Vec<f32> {
        let mut values = Vec::with_capacity((experts * out * inn) as usize);
        for expert in 0..experts {
            let fill = 10.0 + expert as f32;
            values.resize(values.len() + (out * inn) as usize, fill);
        }
        values
    }

    #[test]
    fn slices_single_expert_rows_in_request_order() {
        let dir = fixture_dir("rows");
        write_safetensors_f32(
            &dir,
            "experts.safetensors",
            &[(
                "model.layers.0.mlp.switch_mlp.gate_up_proj.weight",
                vec![4, 2, 3],
                filled_expert_values(4, 2, 3),
            )],
        );
        let headers = ShardHeaderCache::new();
        let tensor = manifest_tensor("model.layers.0.mlp.switch_mlp.gate_up_proj.weight", 4);
        let rows = read_expert_proj_rows(&dir, &headers, &tensor, &[3, 0]).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].weight.shape(), vec![1, 2, 3]);
        mlx_sys::eval(&[&rows[0].weight, &rows[1].weight]);
        assert!(rows[0].weight.data_f32().iter().all(|v| *v == 13.0));
        assert!(rows[1].weight.data_f32().iter().all(|v| *v == 10.0));
        assert!(rows[0].scales.is_none() && rows[0].linear_bias.is_none());
        assert_eq!(rows[0].bytes(), 2 * 3 * 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn slices_quantization_sidecars_with_the_same_expert_axis() {
        let dir = fixture_dir("sidecars");
        write_safetensors_f32(
            &dir,
            "experts.safetensors",
            &[
                (
                    "model.layers.0.mlp.switch_mlp.gate_proj.weight",
                    vec![4, 2, 3],
                    filled_expert_values(4, 2, 3),
                ),
                (
                    "model.layers.0.mlp.switch_mlp.gate_proj.scales",
                    vec![4, 2],
                    filled_expert_values(4, 2, 1),
                ),
                (
                    "model.layers.0.mlp.switch_mlp.gate_proj.biases",
                    vec![4, 2],
                    filled_expert_values(4, 2, 1),
                ),
            ],
        );
        let headers = ShardHeaderCache::new();
        let tensor = manifest_tensor("model.layers.0.mlp.switch_mlp.gate_proj.weight", 4);
        let rows = read_expert_proj_rows(&dir, &headers, &tensor, &[2]).unwrap();
        let scales = rows[0].scales.as_ref().expect("scales row sliced");
        let biases = rows[0].biases.as_ref().expect("biases row sliced");
        assert_eq!(scales.shape(), vec![1, 2]);
        mlx_sys::eval(&[scales, biases]);
        assert!(scales.data_f32().iter().all(|v| *v == 12.0));
        assert!(biases.data_f32().iter().all(|v| *v == 12.0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_expert_count_mismatch_and_out_of_range() {
        let dir = fixture_dir("validate");
        write_safetensors_f32(
            &dir,
            "experts.safetensors",
            &[(
                "model.layers.0.mlp.switch_mlp.gate_up_proj.weight",
                vec![4, 2, 3],
                filled_expert_values(4, 2, 3),
            )],
        );
        let headers = ShardHeaderCache::new();
        let wrong_count = manifest_tensor("model.layers.0.mlp.switch_mlp.gate_up_proj.weight", 8);
        assert!(read_expert_proj_rows(&dir, &headers, &wrong_count, &[0]).is_err());
        let tensor = manifest_tensor("model.layers.0.mlp.switch_mlp.gate_up_proj.weight", 4);
        assert!(read_expert_proj_rows(&dir, &headers, &tensor, &[4]).is_err());
        let missing = manifest_tensor("model.layers.0.mlp.switch_mlp.missing.weight", 4);
        assert!(read_expert_proj_rows(&dir, &headers, &missing, &[0]).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_non_expert_major_sidecar() {
        let dir = fixture_dir("bad_sidecar");
        write_safetensors_f32(
            &dir,
            "experts.safetensors",
            &[
                (
                    "model.layers.0.mlp.switch_mlp.gate_proj.weight",
                    vec![4, 2, 3],
                    filled_expert_values(4, 2, 3),
                ),
                (
                    "model.layers.0.mlp.switch_mlp.gate_proj.scales",
                    vec![2, 4],
                    vec![0.5; 8],
                ),
            ],
        );
        let headers = ShardHeaderCache::new();
        let tensor = manifest_tensor("model.layers.0.mlp.switch_mlp.gate_proj.weight", 4);
        assert!(read_expert_proj_rows(&dir, &headers, &tensor, &[0]).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn header_cache_returns_the_same_parse() {
        let dir = fixture_dir("cache");
        let path = write_safetensors_f32(
            &dir,
            "experts.safetensors",
            &[("a", vec![2], vec![1.0, 2.0])],
        );
        let headers = ShardHeaderCache::new();
        let first = headers.header(&path).unwrap();
        let second = headers.header(&path).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
