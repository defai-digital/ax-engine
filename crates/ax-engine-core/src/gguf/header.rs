use std::collections::HashMap;

use thiserror::Error;

const MAX_METADATA_ARRAY_LEN: usize = 1_000_000;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("invalid magic bytes: expected GGUF (0x46554747), got 0x{0:08x}")]
    InvalidMagic(u32),
    #[error("unsupported GGUF version: {0} (expected 3)")]
    UnsupportedVersion(u32),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid metadata: {0}")]
    InvalidMetadata(String),
    #[error("unknown metadata value type: {0}")]
    UnknownValueType(u32),
    #[error("unknown tensor type: {0}")]
    UnknownTensorType(u32),
    #[error(
        "unexpected end of data at offset {offset}, need {needed} bytes but only {available} remain"
    )]
    UnexpectedEof {
        offset: usize,
        needed: usize,
        available: usize,
    },
    #[error("file too small: {0} bytes (minimum 24 for GGUF header)")]
    FileTooSmall(usize),
    #[error("metadata array too large: {len} elements (max {max})")]
    MetadataArrayTooLarge { len: u64, max: usize },
}

/// Parsed GGUF file header and metadata.
#[derive(Debug)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: HashMap<String, MetadataValue>,
}

/// GGUF metadata value types.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
}

/// GGUF metadata value type IDs (from the spec).
#[repr(u32)]
enum ValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl ValueType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

/// A cursor over a byte slice for parsing GGUF binary data.
///
/// All reads are little-endian. The cursor advances automatically.
pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn ensure(&self, n: usize) -> Result<(), GgufError> {
        if self.remaining() < n {
            return Err(GgufError::UnexpectedEof {
                offset: self.pos,
                needed: n,
                available: self.remaining(),
            });
        }
        Ok(())
    }

    pub fn read_u8(&mut self) -> Result<u8, GgufError> {
        self.ensure(1)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    pub fn read_i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.read_u8()? as i8)
    }

    pub fn read_u16(&mut self) -> Result<u16, GgufError> {
        self.ensure(2)?;
        let v = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    pub fn read_i16(&mut self) -> Result<i16, GgufError> {
        Ok(self.read_u16()? as i16)
    }

    pub fn read_u32(&mut self) -> Result<u32, GgufError> {
        self.ensure(4)?;
        let v = u32::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    pub fn read_i32(&mut self) -> Result<i32, GgufError> {
        Ok(self.read_u32()? as i32)
    }

    pub fn read_u64(&mut self) -> Result<u64, GgufError> {
        self.ensure(8)?;
        let v = u64::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
            self.data[self.pos + 4],
            self.data[self.pos + 5],
            self.data[self.pos + 6],
            self.data[self.pos + 7],
        ]);
        self.pos += 8;
        Ok(v)
    }

    pub fn read_i64(&mut self) -> Result<i64, GgufError> {
        Ok(self.read_u64()? as i64)
    }

    pub fn read_f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    pub fn read_f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    /// Read a length-prefixed UTF-8 string (u64 length + bytes).
    pub fn read_string(&mut self) -> Result<String, GgufError> {
        let len = self.read_u64()? as usize;
        self.ensure(len)?;
        let bytes = &self.data[self.pos..self.pos + len];
        let s = std::str::from_utf8(bytes).map_err(|e| {
            GgufError::InvalidMetadata(format!("invalid UTF-8 string at offset {}: {e}", self.pos))
        })?;
        self.pos += len;
        Ok(s.to_string())
    }

    /// Read a bool (1 byte, 0 = false, nonzero = true).
    pub fn read_bool(&mut self) -> Result<bool, GgufError> {
        Ok(self.read_u8()? != 0)
    }

    /// Skip `n` bytes, advancing the cursor without reading.
    pub fn skip(&mut self, n: usize) -> Result<(), GgufError> {
        self.ensure(n)?;
        self.pos += n;
        Ok(())
    }
}

impl GgufHeader {
    /// Parse a GGUF header and all metadata from a byte slice.
    ///
    /// Returns the parsed header and the cursor position (byte offset after
    /// all metadata has been read — this is where tensor info starts).
    pub fn parse(data: &[u8]) -> Result<(Self, usize), GgufError> {
        if data.len() < 24 {
            return Err(GgufError::FileTooSmall(data.len()));
        }

        let mut cursor = Cursor::new(data);

        // Magic: 4 bytes
        let magic = cursor.read_u32()?;
        if magic != super::GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        // Version: 4 bytes
        let version = cursor.read_u32()?;
        if version != super::GGUF_VERSION {
            return Err(GgufError::UnsupportedVersion(version));
        }

        // Tensor count: 8 bytes
        let tensor_count = cursor.read_u64()?;

        // Metadata KV count: 8 bytes
        let kv_count = cursor.read_u64()?;

        // Parse metadata KV pairs
        // Cap both capacity and loop count to defend against malicious kv_count
        // in untrusted GGUF files. Legitimate models have <200 metadata keys.
        let max_kv = (kv_count as usize).min(4096);
        let mut metadata = HashMap::with_capacity(max_kv);
        for _ in 0..max_kv {
            let key = cursor.read_string()?;
            let value = read_metadata_value(&mut cursor)?;
            metadata.insert(key, value);
        }

        let pos = cursor.position();
        Ok((
            GgufHeader {
                version,
                tensor_count,
                metadata,
            },
            pos,
        ))
    }

    /// Get a string metadata value by key.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(MetadataValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Get a u32 metadata value by key (also converts from u16/u8).
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key) {
            Some(MetadataValue::Uint32(v)) => Some(*v),
            Some(MetadataValue::Uint16(v)) => Some(*v as u32),
            Some(MetadataValue::Uint8(v)) => Some(*v as u32),
            _ => None,
        }
    }

    /// Get a u64 metadata value by key (also converts from smaller unsigned types).
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.metadata.get(key) {
            Some(MetadataValue::Uint64(v)) => Some(*v),
            Some(MetadataValue::Uint32(v)) => Some(*v as u64),
            Some(MetadataValue::Uint16(v)) => Some(*v as u64),
            Some(MetadataValue::Uint8(v)) => Some(*v as u64),
            _ => None,
        }
    }

    /// Get a f32 metadata value by key.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key) {
            Some(MetadataValue::Float32(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get a bool metadata value by key.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.metadata.get(key) {
            Some(MetadataValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get a string array metadata value by key.
    pub fn get_str_array(&self, key: &str) -> Option<Vec<&str>> {
        match self.metadata.get(key) {
            Some(MetadataValue::Array(arr)) => {
                let mut result = Vec::with_capacity(arr.len());
                for v in arr {
                    match v {
                        MetadataValue::String(s) => result.push(s.as_str()),
                        _ => return None,
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }

    /// Get a f32 array metadata value by key.
    pub fn get_f32_array(&self, key: &str) -> Option<Vec<f32>> {
        match self.metadata.get(key) {
            Some(MetadataValue::Array(arr)) => {
                let mut result = Vec::with_capacity(arr.len());
                for v in arr {
                    match v {
                        MetadataValue::Float32(f) => result.push(*f),
                        _ => return None,
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }

    /// Get an i32 array metadata value by key.
    pub fn get_i32_array(&self, key: &str) -> Option<Vec<i32>> {
        match self.metadata.get(key) {
            Some(MetadataValue::Array(arr)) => {
                let mut result = Vec::with_capacity(arr.len());
                for v in arr {
                    match v {
                        MetadataValue::Int32(i) => result.push(*i),
                        _ => return None,
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }

    /// Get a bool array metadata value by key.
    pub fn get_bool_array(&self, key: &str) -> Option<Vec<bool>> {
        match self.metadata.get(key) {
            Some(MetadataValue::Array(arr)) => {
                let mut result = Vec::with_capacity(arr.len());
                for v in arr {
                    match v {
                        MetadataValue::Bool(b) => result.push(*b),
                        _ => return None,
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }

    /// Get the data alignment (defaults to 32 per spec).
    pub fn alignment(&self) -> u64 {
        self.get_u64("general.alignment").unwrap_or(32)
    }

    /// Get the model architecture string (e.g. "llama", "gemma").
    pub fn architecture(&self) -> Option<&str> {
        self.get_str("general.architecture")
    }
}

/// Read a single typed metadata value from the cursor.
fn read_metadata_value(cursor: &mut Cursor<'_>) -> Result<MetadataValue, GgufError> {
    let type_id = cursor.read_u32()?;
    read_typed_value(cursor, type_id)
}

/// Read a value of the given type from the cursor.
fn read_typed_value(cursor: &mut Cursor<'_>, type_id: u32) -> Result<MetadataValue, GgufError> {
    let vtype = ValueType::from_u32(type_id).ok_or(GgufError::UnknownValueType(type_id))?;

    match vtype {
        ValueType::Uint8 => Ok(MetadataValue::Uint8(cursor.read_u8()?)),
        ValueType::Int8 => Ok(MetadataValue::Int8(cursor.read_i8()?)),
        ValueType::Uint16 => Ok(MetadataValue::Uint16(cursor.read_u16()?)),
        ValueType::Int16 => Ok(MetadataValue::Int16(cursor.read_i16()?)),
        ValueType::Uint32 => Ok(MetadataValue::Uint32(cursor.read_u32()?)),
        ValueType::Int32 => Ok(MetadataValue::Int32(cursor.read_i32()?)),
        ValueType::Float32 => Ok(MetadataValue::Float32(cursor.read_f32()?)),
        ValueType::Bool => Ok(MetadataValue::Bool(cursor.read_bool()?)),
        ValueType::String => Ok(MetadataValue::String(cursor.read_string()?)),
        ValueType::Uint64 => Ok(MetadataValue::Uint64(cursor.read_u64()?)),
        ValueType::Int64 => Ok(MetadataValue::Int64(cursor.read_i64()?)),
        ValueType::Float64 => Ok(MetadataValue::Float64(cursor.read_f64()?)),
        ValueType::Array => {
            let elem_type_id = cursor.read_u32()?;
            let len_u64 = cursor.read_u64()?;
            if len_u64 > MAX_METADATA_ARRAY_LEN as u64 {
                return Err(GgufError::MetadataArrayTooLarge {
                    len: len_u64,
                    max: MAX_METADATA_ARRAY_LEN,
                });
            }
            let len = len_u64 as usize;
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                values.push(read_typed_value(cursor, elem_type_id)?);
            }
            Ok(MetadataValue::Array(values))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid GGUF v3 header in memory.
    fn build_test_gguf(metadata: &[(&str, u32, &[u8])], tensor_count: u64) -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic
        buf.extend_from_slice(&super::super::GGUF_MAGIC.to_le_bytes());
        // Version
        buf.extend_from_slice(&super::super::GGUF_VERSION.to_le_bytes());
        // Tensor count
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        // KV count
        buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

        // KV pairs
        for (key, type_id, value_bytes) in metadata {
            // Key: u64 len + bytes
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            // Value type
            buf.extend_from_slice(&type_id.to_le_bytes());
            // Value data
            buf.extend_from_slice(value_bytes);
        }

        buf
    }

    fn encode_gguf_string(s: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
        buf
    }

    #[test]
    fn test_parse_empty_header() {
        let data = build_test_gguf(&[], 0);
        let (header, pos) = GgufHeader::parse(&data).unwrap();
        assert_eq!(header.version, 3);
        assert_eq!(header.tensor_count, 0);
        assert!(header.metadata.is_empty());
        assert_eq!(pos, 24); // 4 + 4 + 8 + 8
    }

    #[test]
    fn test_metadata_array_length_is_capped() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(ValueType::Array as u32).to_le_bytes());
        buf.extend_from_slice(&(ValueType::Int32 as u32).to_le_bytes());
        buf.extend_from_slice(&((MAX_METADATA_ARRAY_LEN as u64) + 1).to_le_bytes());

        let mut cursor = Cursor::new(&buf);
        let err = read_metadata_value(&mut cursor).unwrap_err();
        assert!(matches!(
            err,
            GgufError::MetadataArrayTooLarge {
                len,
                max: MAX_METADATA_ARRAY_LEN
            } if len == (MAX_METADATA_ARRAY_LEN as u64) + 1
        ));
    }

    #[test]
    fn test_parse_string_metadata() {
        let value_bytes = encode_gguf_string("llama");
        let data = build_test_gguf(&[("general.architecture", 8, &value_bytes)], 0);
        let (header, _) = GgufHeader::parse(&data).unwrap();

        assert_eq!(header.architecture(), Some("llama"));
    }

    #[test]
    fn test_parse_u32_metadata() {
        let data = build_test_gguf(&[("llama.block_count", 4, &32u32.to_le_bytes())], 5);
        let (header, _) = GgufHeader::parse(&data).unwrap();

        assert_eq!(header.tensor_count, 5);
        assert_eq!(header.get_u32("llama.block_count"), Some(32));
    }

    #[test]
    fn test_parse_f32_metadata() {
        let data = build_test_gguf(
            &[(
                "llama.attention.layer_norm_rms_epsilon",
                6,
                &1e-5f32.to_le_bytes(),
            )],
            0,
        );
        let (header, _) = GgufHeader::parse(&data).unwrap();

        let eps = header
            .get_f32("llama.attention.layer_norm_rms_epsilon")
            .unwrap();
        assert!((eps - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_parse_bool_metadata() {
        let data = build_test_gguf(&[("general.file_type", 7, &[1u8])], 0);
        let (header, _) = GgufHeader::parse(&data).unwrap();

        assert_eq!(header.get_bool("general.file_type"), Some(true));
    }

    #[test]
    fn test_parse_multiple_kv() {
        let arch_val = encode_gguf_string("llama");
        let data = build_test_gguf(
            &[
                ("general.architecture", 8, &arch_val),
                ("llama.block_count", 4, &32u32.to_le_bytes()),
                ("llama.context_length", 4, &4096u32.to_le_bytes()),
            ],
            100,
        );
        let (header, _) = GgufHeader::parse(&data).unwrap();

        assert_eq!(header.tensor_count, 100);
        assert_eq!(header.architecture(), Some("llama"));
        assert_eq!(header.get_u32("llama.block_count"), Some(32));
        assert_eq!(header.get_u32("llama.context_length"), Some(4096));
    }

    #[test]
    fn test_invalid_magic() {
        let mut data = build_test_gguf(&[], 0);
        data[0] = 0xFF; // corrupt magic
        let err = GgufHeader::parse(&data).unwrap_err();
        assert!(matches!(err, GgufError::InvalidMagic(_)));
    }

    #[test]
    fn test_unsupported_version() {
        let mut data = build_test_gguf(&[], 0);
        // Overwrite version field (offset 4..8) with version 99
        data[4..8].copy_from_slice(&99u32.to_le_bytes());
        let err = GgufHeader::parse(&data).unwrap_err();
        assert!(matches!(err, GgufError::UnsupportedVersion(99)));
    }

    #[test]
    fn test_file_too_small() {
        let data = vec![0u8; 10];
        let err = GgufHeader::parse(&data).unwrap_err();
        assert!(matches!(err, GgufError::FileTooSmall(10)));
    }

    #[test]
    fn test_truncated_metadata() {
        // Header says 1 KV pair but data ends after header
        let data = build_test_gguf(&[], 0);
        let mut bad = data[..24].to_vec();
        // Overwrite kv_count to 1
        bad[16..24].copy_from_slice(&1u64.to_le_bytes());
        let err = GgufHeader::parse(&bad).unwrap_err();
        assert!(matches!(err, GgufError::UnexpectedEof { .. }));
    }

    #[test]
    fn test_alignment_default() {
        let data = build_test_gguf(&[], 0);
        let (header, _) = GgufHeader::parse(&data).unwrap();
        assert_eq!(header.alignment(), 32);
    }

    #[test]
    fn test_alignment_custom() {
        let data = build_test_gguf(&[("general.alignment", 4, &64u32.to_le_bytes())], 0);
        let (header, _) = GgufHeader::parse(&data).unwrap();
        assert_eq!(header.alignment(), 64);
    }

    #[test]
    fn test_u64_metadata() {
        let data = build_test_gguf(&[("big_number", 10, &u64::MAX.to_le_bytes())], 0);
        let (header, _) = GgufHeader::parse(&data).unwrap();
        assert_eq!(header.get_u64("big_number"), Some(u64::MAX));
    }

    #[test]
    fn test_array_metadata() {
        // Build array value: type=8 (string), len=2, "hello", "world"
        let mut arr_bytes = Vec::new();
        arr_bytes.extend_from_slice(&8u32.to_le_bytes()); // elem type: string
        arr_bytes.extend_from_slice(&2u64.to_le_bytes()); // length
        arr_bytes.extend_from_slice(&encode_gguf_string("hello"));
        arr_bytes.extend_from_slice(&encode_gguf_string("world"));

        let data = build_test_gguf(&[("tokens", 9, &arr_bytes)], 0);
        let (header, _) = GgufHeader::parse(&data).unwrap();

        let arr = header.get_str_array("tokens").unwrap();
        assert_eq!(arr, vec!["hello", "world"]);
    }
}
