use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;

use super::header::{GgufError, GgufHeader};
use super::tensor::{GgmlType, TensorInfo};

pub fn support_note_for_q5k_layer_presence(has_q5k_layer_weights: bool) -> Option<&'static str> {
    if has_q5k_layer_weights {
        Some(
            "Mixed-quant Q5_K layers use AX's GPU prefill route by default; AX selects the base or small-N path automatically.",
        )
    } else {
        None
    }
}

fn is_active_layer_weight_name(name: &str) -> bool {
    const LAYER_SUFFIXES: &[&str] = &[
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ];

    name.starts_with("blk.") && LAYER_SUFFIXES.iter().any(|suffix| name.ends_with(suffix))
}

/// A memory-mapped GGUF model file.
///
/// Provides zero-copy access to tensor data via mmap. The entire file is
/// mapped into the process address space; on macOS with UMA, this means
/// both CPU and Metal can access the same physical memory.
pub struct MappedModel {
    mmap: Mmap,
    /// Parsed header and metadata.
    pub header: GgufHeader,
    /// Tensor info entries, indexed by name for O(1) lookup.
    tensor_map: HashMap<String, TensorInfo>,
    /// All tensor info entries in file order.
    pub tensors: Vec<TensorInfo>,
    /// Byte offset where tensor data begins (after header + metadata + tensor info + alignment padding).
    pub data_offset: usize,
}

impl MappedModel {
    /// Open and memory-map a GGUF file, parsing header, metadata, and tensor info.
    ///
    /// This is a zero-copy operation for tensor data — weights remain on disk
    /// until accessed, and the OS pages them in on demand.
    pub fn open(path: &Path) -> Result<Self, GgufError> {
        let file = std::fs::File::open(path)?;
        let file_size = file.metadata()?.len() as usize;

        // SAFETY: We treat the mmap as read-only. The file must not be modified
        // while mapped (standard mmap contract).
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 24 {
            return Err(GgufError::FileTooSmall(mmap.len()));
        }

        // Parse header + metadata
        let (header, after_metadata) = GgufHeader::parse(&mmap)?;

        // Parse tensor info entries
        let (tensors, after_tensor_info) =
            TensorInfo::parse_all(&mmap, after_metadata, header.tensor_count)?;

        // Compute data offset: align to the file's alignment boundary
        let alignment = header.alignment() as usize;
        let data_offset = align_up(after_tensor_info, alignment).ok_or_else(|| {
            GgufError::InvalidMetadata(format!(
                "tensor data offset overflow: offset={after_tensor_info}, alignment={alignment}"
            ))
        })?;

        // Validate that data_offset doesn't exceed file size
        if data_offset > file_size {
            return Err(GgufError::InvalidMetadata(format!(
                "tensor data offset ({data_offset}) exceeds file size ({file_size})"
            )));
        }

        // Build name→info lookup map
        let tensor_map: HashMap<String, TensorInfo> = tensors
            .iter()
            .map(|t| (t.name.clone(), t.clone()))
            .collect();

        tracing::info!(
            path = %path.display(),
            version = header.version,
            tensors = tensors.len(),
            metadata_keys = header.metadata.len(),
            data_offset = data_offset,
            file_size = file_size,
            architecture = header.architecture().unwrap_or("unknown"),
            "GGUF file loaded"
        );

        Ok(Self {
            mmap,
            header,
            tensor_map,
            tensors,
            data_offset,
        })
    }

    /// Get tensor info by name.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_map.get(name)
    }

    /// Get a raw byte slice for a tensor's data (zero-copy from mmap).
    ///
    /// The returned slice points directly into the mmap'd file. For quantized
    /// tensors, this is the raw quantized block data that must be dequantized
    /// before use.
    pub fn tensor_data(&self, info: &TensorInfo) -> Result<&[u8], GgufError> {
        let start = self
            .data_offset
            .checked_add(info.offset as usize)
            .ok_or_else(|| GgufError::UnexpectedEof {
                offset: self.data_offset,
                needed: info.offset as usize,
                available: self.mmap.len().saturating_sub(self.data_offset),
            })?;
        let size = info.data_size() as usize;
        let end = start
            .checked_add(size)
            .ok_or_else(|| GgufError::UnexpectedEof {
                offset: start,
                needed: size,
                available: self.mmap.len().saturating_sub(start),
            })?;

        if end > self.mmap.len() {
            return Err(GgufError::UnexpectedEof {
                offset: start,
                needed: size,
                available: self.mmap.len().saturating_sub(start),
            });
        }

        Ok(&self.mmap[start..end])
    }

    /// Get raw tensor data by name.
    pub fn tensor_data_by_name(&self, name: &str) -> Result<&[u8], GgufError> {
        let info = self
            .tensor_info(name)
            .ok_or_else(|| GgufError::InvalidMetadata(format!("tensor not found: {name}")))?;
        self.tensor_data(info)
    }

    /// Total file size in bytes.
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Total size of all tensor data in bytes (sum of all tensor data sizes).
    pub fn total_tensor_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.data_size()).sum()
    }

    /// Print a summary of all tensors (for debugging / --verbose).
    pub fn print_tensor_summary(&self) {
        for t in &self.tensors {
            let shape_str: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
            tracing::debug!(
                name = %t.name,
                shape = %format!("[{}]", shape_str.join(", ")),
                dtype = %t.dtype,
                size_mb = t.data_size() as f64 / 1024.0 / 1024.0,
                "tensor"
            );
        }
    }

    /// Get the predominant quantization type across all tensors (by total bytes).
    pub fn predominant_quant(&self) -> Option<GgmlType> {
        let mut type_bytes: HashMap<GgmlType, u64> = HashMap::new();
        for t in &self.tensors {
            *type_bytes.entry(t.dtype).or_default() += t.data_size();
        }
        type_bytes
            .into_iter()
            .max_by_key(|(_, bytes)| *bytes)
            .map(|(dtype, _)| dtype)
    }

    pub fn uses_layer_quant(&self, dtype: GgmlType) -> bool {
        self.tensors
            .iter()
            .any(|tensor| tensor.dtype == dtype && is_active_layer_weight_name(&tensor.name))
    }

    pub fn support_note(&self) -> Option<&'static str> {
        support_note_for_q5k_layer_presence(self.uses_layer_quant(GgmlType::Q5K))
    }
}

/// Round `offset` up to the next multiple of `alignment`.
fn align_up(offset: usize, alignment: usize) -> Option<usize> {
    if alignment == 0 {
        return Some(offset);
    }
    let remainder = offset % alignment;
    if remainder == 0 {
        Some(offset)
    } else {
        offset.checked_add(alignment - remainder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 32), Some(0));
        assert_eq!(align_up(1, 32), Some(32));
        assert_eq!(align_up(31, 32), Some(32));
        assert_eq!(align_up(32, 32), Some(32));
        assert_eq!(align_up(33, 32), Some(64));
        assert_eq!(align_up(100, 64), Some(128));
        assert_eq!(align_up(128, 64), Some(128));
    }

    #[test]
    fn test_align_up_zero() {
        assert_eq!(align_up(42, 0), Some(42));
    }

    #[test]
    fn test_align_up_returns_none_on_overflow() {
        assert_eq!(align_up(usize::MAX - 7, 16), None);
    }

    #[test]
    fn test_support_note_for_q5k_layer_presence_marks_default_q5k_prefill() {
        assert!(
            support_note_for_q5k_layer_presence(true)
                .unwrap()
                .contains("GPU prefill route by default")
        );
        assert_eq!(support_note_for_q5k_layer_presence(false), None);
    }

    #[test]
    fn test_is_active_layer_weight_name_matches_transformer_projection_weights() {
        assert!(is_active_layer_weight_name("blk.0.attn_q.weight"));
        assert!(is_active_layer_weight_name("blk.10.attn_v.weight"));
        assert!(is_active_layer_weight_name("blk.31.ffn_down.weight"));
    }

    #[test]
    fn test_is_active_layer_weight_name_excludes_non_layer_or_non_projection_tensors() {
        assert!(!is_active_layer_weight_name("token_embd.weight"));
        assert!(!is_active_layer_weight_name("output.weight"));
        assert!(!is_active_layer_weight_name("blk.0.attn_norm.weight"));
        assert!(!is_active_layer_weight_name("blk.0.ffn_norm.weight"));
        assert!(!is_active_layer_weight_name("blk.0.attn_q.bias"));
    }

    /// Build a complete GGUF file in memory with header, metadata, tensor info, and tensor data.
    fn build_test_gguf_file() -> Vec<u8> {
        let mut buf = Vec::new();
        let alignment: u64 = 32;

        // --- Header ---
        buf.extend_from_slice(&super::super::GGUF_MAGIC.to_le_bytes()); // magic
        buf.extend_from_slice(&super::super::GGUF_VERSION.to_le_bytes()); // version
        buf.extend_from_slice(&2u64.to_le_bytes()); // tensor_count = 2
        buf.extend_from_slice(&2u64.to_le_bytes()); // metadata_kv_count = 2

        // --- Metadata KV 1: general.architecture = "llama" ---
        let key1 = "general.architecture";
        buf.extend_from_slice(&(key1.len() as u64).to_le_bytes());
        buf.extend_from_slice(key1.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes()); // type: string
        let val1 = "llama";
        buf.extend_from_slice(&(val1.len() as u64).to_le_bytes());
        buf.extend_from_slice(val1.as_bytes());

        // --- Metadata KV 2: general.alignment = 32 ---
        let key2 = "general.alignment";
        buf.extend_from_slice(&(key2.len() as u64).to_le_bytes());
        buf.extend_from_slice(key2.as_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes()); // type: uint32
        buf.extend_from_slice(&(alignment as u32).to_le_bytes());

        // --- Tensor info 1: "weight.q" [32] Q8_0 at offset 0 ---
        let t1_name = "weight.q";
        buf.extend_from_slice(&(t1_name.len() as u64).to_le_bytes());
        buf.extend_from_slice(t1_name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
        buf.extend_from_slice(&32u64.to_le_bytes()); // shape[0] = 32
        buf.extend_from_slice(&8u32.to_le_bytes()); // type = Q8_0
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset = 0

        // --- Tensor info 2: "bias" [32] F32 at offset 64 (after first tensor, aligned) ---
        let t2_name = "bias";
        buf.extend_from_slice(&(t2_name.len() as u64).to_le_bytes());
        buf.extend_from_slice(t2_name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
        buf.extend_from_slice(&32u64.to_le_bytes()); // shape[0] = 32
        buf.extend_from_slice(&0u32.to_le_bytes()); // type = F32
        // Q8_0 tensor: 32 elements = 1 block × 34 bytes; aligned to 64 = 64
        buf.extend_from_slice(&64u64.to_le_bytes()); // offset = 64

        // --- Padding to alignment ---
        let current = buf.len();
        let data_start = align_up(current, alignment as usize).expect("test GGUF alignment");
        buf.resize(data_start, 0);

        // --- Tensor data ---
        // Tensor 1: Q8_0, 1 block = 34 bytes
        buf.extend_from_slice(&[0u8; 34]);
        // Pad to 64
        buf.extend_from_slice(&[0u8; 30]);
        // Tensor 2: F32, 32 elements = 128 bytes
        buf.extend_from_slice(&[0u8; 128]);

        buf
    }

    #[test]
    fn test_mapped_model_from_bytes() {
        let data = build_test_gguf_file();

        // Write to a temp file
        let dir = std::env::temp_dir();
        let path = dir.join("ax_test_model.gguf");
        std::fs::write(&path, &data).unwrap();

        let model = MappedModel::open(&path).unwrap();

        // Verify header
        assert_eq!(model.header.version, 3);
        assert_eq!(model.header.tensor_count, 2);
        assert_eq!(model.header.architecture(), Some("llama"));
        assert_eq!(model.header.alignment(), 32);

        // Verify tensors
        assert_eq!(model.tensors.len(), 2);
        assert_eq!(model.tensors[0].name, "weight.q");
        assert_eq!(model.tensors[0].dtype, GgmlType::Q8_0);
        assert_eq!(model.tensors[1].name, "bias");
        assert_eq!(model.tensors[1].dtype, GgmlType::F32);

        // Verify tensor lookup
        let t = model.tensor_info("weight.q").unwrap();
        assert_eq!(t.shape, vec![32]);

        // Verify tensor data access
        let data_bytes = model.tensor_data(&model.tensors[0]).unwrap();
        assert_eq!(data_bytes.len(), 34); // 1 Q8_0 block

        let bias_data = model.tensor_data_by_name("bias").unwrap();
        assert_eq!(bias_data.len(), 128); // 32 × f32

        // Verify total size
        assert_eq!(model.total_tensor_bytes(), 34 + 128);

        // Verify predominant quant
        // F32 has 128 bytes, Q8_0 has 34 bytes → F32 is predominant
        assert_eq!(model.predominant_quant(), Some(GgmlType::F32));

        // Cleanup
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_tensor_not_found() {
        let data = build_test_gguf_file();
        let dir = std::env::temp_dir();
        let path = dir.join("ax_test_model_2.gguf");
        std::fs::write(&path, &data).unwrap();

        let model = MappedModel::open(&path).unwrap();
        assert!(model.tensor_info("nonexistent").is_none());
        let err = model.tensor_data_by_name("nonexistent").unwrap_err();
        assert!(matches!(err, GgufError::InvalidMetadata(_)));

        std::fs::remove_file(&path).ok();
    }
}
