use super::header::{Cursor, GgufError};

/// Tensor metadata parsed from GGUF tensor info section.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub shape: Vec<u64>,
    pub dtype: GgmlType,
    /// Offset relative to the start of the tensor data section (not the file).
    pub offset: u64,
}

/// GGML tensor data types (quantization formats).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
}

impl GgmlType {
    /// Block size for this quantization type.
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
        }
    }

    /// Bytes per block for this quantization type.
    pub fn bytes_per_block(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
        }
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
        }
    }

    /// Try to convert from a raw u32 type value.
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            _ => None,
        }
    }
}

impl std::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl TensorInfo {
    /// Total number of elements in this tensor.
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Total bytes occupied by this tensor's data.
    pub fn data_size(&self) -> u64 {
        let n = self.n_elements() as usize;
        let bs = self.dtype.block_size();
        let bpb = self.dtype.bytes_per_block();
        (n.div_ceil(bs) * bpb) as u64
    }

    /// Parse all tensor info entries from a byte slice starting at the given position.
    ///
    /// Returns the parsed tensors and the cursor position after all tensor info.
    pub(crate) fn parse_all(
        data: &[u8],
        start_offset: usize,
        count: u64,
    ) -> Result<(Vec<Self>, usize), GgufError> {
        let mut cursor = Cursor::new(data);
        // Advance cursor to start_offset by reading bytes
        // (Cursor starts at 0, we need to skip to start_offset)
        cursor.skip(start_offset)?;

        let mut tensors = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let name = cursor.read_string()?;
            let n_dims = cursor.read_u32()?;
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(cursor.read_u64()?);
            }
            let type_id = cursor.read_u32()?;
            let dtype = GgmlType::from_u32(type_id).ok_or(GgufError::UnknownTensorType(type_id))?;
            let offset = cursor.read_u64()?;

            tensors.push(TensorInfo {
                name,
                n_dims,
                shape,
                dtype,
                offset,
            });
        }

        Ok((tensors, cursor.position()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_roundtrip() {
        for type_id in [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
            let t = GgmlType::from_u32(type_id).unwrap();
            assert_eq!(t as u32, type_id);
        }
    }

    #[test]
    fn test_ggml_type_unknown() {
        assert!(GgmlType::from_u32(4).is_none()); // gap in enum
        assert!(GgmlType::from_u32(5).is_none());
        assert!(GgmlType::from_u32(99).is_none());
    }

    #[test]
    fn test_tensor_info_data_size() {
        // 4096 × 4096 F32 tensor = 4096*4096*4 bytes
        let t = TensorInfo {
            name: "test".to_string(),
            n_dims: 2,
            shape: vec![4096, 4096],
            dtype: GgmlType::F32,
            offset: 0,
        };
        assert_eq!(t.n_elements(), 4096 * 4096);
        assert_eq!(t.data_size(), 4096 * 4096 * 4);
    }

    #[test]
    fn test_tensor_info_data_size_q4_0() {
        // 4096 elements in Q4_0: 4096/32 blocks × 18 bytes/block = 2304
        let t = TensorInfo {
            name: "test".to_string(),
            n_dims: 1,
            shape: vec![4096],
            dtype: GgmlType::Q4_0,
            offset: 0,
        };
        assert_eq!(t.data_size(), 4096 / 32 * 18);
    }

    #[test]
    fn test_tensor_info_data_size_q4k() {
        // 4096 elements in Q4_K: 4096/256 blocks × 144 bytes/block = 2304
        let t = TensorInfo {
            name: "test".to_string(),
            n_dims: 1,
            shape: vec![4096],
            dtype: GgmlType::Q4K,
            offset: 0,
        };
        assert_eq!(t.data_size(), 4096 / 256 * 144);
    }

    #[test]
    fn test_parse_tensor_info() {
        // Build a buffer: skip first 24 bytes (simulating header), then one tensor info
        let start = 24;
        let mut data = vec![0u8; start];

        // Tensor name: "weight"
        data.extend_from_slice(&6u64.to_le_bytes());
        data.extend_from_slice(b"weight");
        // n_dims: 2
        data.extend_from_slice(&2u32.to_le_bytes());
        // shape: [4096, 4096]
        data.extend_from_slice(&4096u64.to_le_bytes());
        data.extend_from_slice(&4096u64.to_le_bytes());
        // type: Q4_0 (2)
        data.extend_from_slice(&2u32.to_le_bytes());
        // offset: 0
        data.extend_from_slice(&0u64.to_le_bytes());

        let (tensors, _) = TensorInfo::parse_all(&data, start, 1).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].name, "weight");
        assert_eq!(tensors[0].n_dims, 2);
        assert_eq!(tensors[0].shape, vec![4096, 4096]);
        assert_eq!(tensors[0].dtype, GgmlType::Q4_0);
        assert_eq!(tensors[0].offset, 0);
    }
}
