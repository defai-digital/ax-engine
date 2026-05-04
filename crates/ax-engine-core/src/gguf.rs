//! GGUF file format reader producing `NativeModelArtifacts` directly from a
//! Q4_K_M GGUF without a Python export step.
//!
//! Binary format reference: llama.cpp ggml/include/gguf.h
//! Quant format reference:  llama.cpp ggml/src/ggml-quants.{h,c}

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Seek};
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::model::{
    AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION, NativeLinearAttentionConfig, NativeModelArtifacts,
    NativeModelManifest, NativeMoeConfig, NativeRuntimeStatus, NativeTensorDataType,
    NativeTensorFormat, NativeTensorRole, NativeTensorSpec,
};

// ---------------------------------------------------------------------------
// Constants from llama.cpp
// ---------------------------------------------------------------------------

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
const MAX_GGUF_STRING_LEN: usize = 1 << 20;
const QK_K: u64 = 256; // elements per Q4_K block
const Q4_K_BLOCK_BYTES: u64 = 144; // sizeof(block_q4_K)

// GGUF KV value types
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

// GGML tensor types
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q8_0: u32 = 8; // 32 elem/block, 34 bytes/block
const GGML_TYPE_Q4_K: u32 = 12; // 256 elem/block, 144 bytes/block
const GGML_TYPE_Q5_K: u32 = 13; // 256 elem/block, 176 bytes/block
const GGML_TYPE_Q6_K: u32 = 14; // 256 elem/block, 210 bytes/block
const GGML_TYPE_I32: u32 = 26; // 4 bytes per element, stored as-is
const GGML_TYPE_BF16: u32 = 30;

// Block sizes (bytes per block) — from llama.cpp ggml-quants.h
const Q8_0_BLOCK_BYTES: u64 = 34; // 2 (F16 scale) + 32 (int8)
const Q5_K_BLOCK_BYTES: u64 = 176; // 4 (d+dmin F16) + 12 (scales) + 32 (qh) + 128 (qs)
const Q6_K_BLOCK_BYTES: u64 = 210; // 128 (ql) + 64 (qh) + 16 (scales int8) + 2 (d F16)
const QK8_0: u64 = 32; // elements per Q8_0 block

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("I/O error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("not a GGUF file: bad magic bytes")]
    BadMagic,
    #[error("unsupported GGUF version {0} (expected 2 or 3)")]
    UnsupportedVersion(u32),
    #[error("unsupported GGUF KV type {0}")]
    UnknownKvType(u32),
    #[error("missing required metadata key: {0}")]
    MissingMetadata(&'static str),
    #[error("unsupported GGML tensor type {0} in tensor {1}")]
    UnsupportedTensorType(u32, String),
    #[error("tensor {name} has element count {elements} not divisible by QK_K=256")]
    UnalignedQ4K { name: String, elements: u64 },
    #[error("unrecognised architecture \"{0}\"")]
    UnknownArch(String),
    #[error("invalid manifest after GGUF parse: {0}")]
    InvalidManifest(String),
}

// ---------------------------------------------------------------------------
// Internal KV store
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum KvValue {
    UInt(u64),
    Int(i64),
    Float(f64),
    Str(String),
    // Other types are skipped
}

impl KvValue {
    fn as_uint(&self) -> Option<u64> {
        match self {
            KvValue::UInt(v) => Some(*v),
            KvValue::Int(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    fn as_float(&self) -> Option<f64> {
        match self {
            KvValue::Float(v) => Some(*v),
            KvValue::UInt(v) => Some(*v as f64),
            KvValue::Int(v) => Some(*v as f64),
            _ => None,
        }
    }
    fn as_str(&self) -> Option<&str> {
        match self {
            KvValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Low-level byte readers
// ---------------------------------------------------------------------------

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16(r: &mut impl Read) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(r: &mut impl Read) -> io::Result<i32> {
    Ok(read_u32(r)? as i32)
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64(r: &mut impl Read) -> io::Result<i64> {
    Ok(read_u64(r)? as i64)
}

fn read_f32(r: &mut impl Read) -> io::Result<f32> {
    let bits = read_u32(r)?;
    Ok(f32::from_bits(bits))
}

fn read_f64(r: &mut impl Read) -> io::Result<f64> {
    let bits = read_u64(r)?;
    Ok(f64::from_bits(bits))
}

fn read_gguf_string(r: &mut impl Read) -> io::Result<String> {
    let len = read_u64(r)? as usize;
    if len > MAX_GGUF_STRING_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("GGUF string length {len} exceeds maximum {MAX_GGUF_STRING_LEN}"),
        ));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn f64_to_u32(value: f64) -> Option<u32> {
    if value.is_finite() && value >= 0.0 && value <= f64::from(u32::MAX) {
        Some(value as u32)
    } else {
        None
    }
}

/// Read one KV value of the given type; return Some(KvValue) for types we
/// care about, None for skipped types (arrays of complex elements, etc.).
fn read_kv_value(r: &mut impl Read, typ: u32) -> io::Result<Option<KvValue>> {
    match typ {
        GGUF_TYPE_UINT8 => Ok(Some(KvValue::UInt(read_u8(r)? as u64))),
        GGUF_TYPE_INT8 => Ok(Some(KvValue::Int(read_u8(r)? as i8 as i64))),
        GGUF_TYPE_UINT16 => Ok(Some(KvValue::UInt(read_u16(r)? as u64))),
        GGUF_TYPE_INT16 => Ok(Some(KvValue::Int(read_u16(r)? as i16 as i64))),
        GGUF_TYPE_UINT32 => Ok(Some(KvValue::UInt(read_u32(r)? as u64))),
        GGUF_TYPE_INT32 => Ok(Some(KvValue::Int(read_i32(r)? as i64))),
        GGUF_TYPE_FLOAT32 => Ok(Some(KvValue::Float(read_f32(r)? as f64))),
        GGUF_TYPE_BOOL => Ok(Some(KvValue::UInt(read_u8(r)? as u64))),
        GGUF_TYPE_STRING => Ok(Some(KvValue::Str(read_gguf_string(r)?))),
        GGUF_TYPE_ARRAY => {
            let arr_type = read_u32(r)?;
            let n = read_u64(r)?;
            // Skip all array elements; we don't use any array KV for architecture metadata.
            for _ in 0..n {
                read_kv_value(r, arr_type)?;
            }
            Ok(None)
        }
        GGUF_TYPE_UINT64 => Ok(Some(KvValue::UInt(read_u64(r)?))),
        GGUF_TYPE_INT64 => Ok(Some(KvValue::Int(read_i64(r)?))),
        GGUF_TYPE_FLOAT64 => Ok(Some(KvValue::Float(read_f64(r)?))),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown GGUF KV type {other}"),
        )),
    }
}

// ---------------------------------------------------------------------------
// Tensor info
// ---------------------------------------------------------------------------

struct GgufTensorInfo {
    name: String,
    /// Dimensions in GGUF/ggml order (fastest-varying first = in_cols first for 2D).
    dims: Vec<u64>,
    ggml_type: u32,
    /// Byte offset from the start of the data section (not the file).
    data_offset: u64,
}

impl GgufTensorInfo {
    /// Total number of elements.
    fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Byte length of tensor data.
    fn byte_length(&self) -> Option<u64> {
        let n = self.n_elements();
        match self.ggml_type {
            GGML_TYPE_F32 | GGML_TYPE_I32 => Some(n * 4),
            GGML_TYPE_F16 => Some(n * 2),
            GGML_TYPE_BF16 => Some(n * 2),
            GGML_TYPE_Q8_0 => {
                if !n.is_multiple_of(QK8_0) {
                    return None;
                }
                Some((n / QK8_0) * Q8_0_BLOCK_BYTES)
            }
            GGML_TYPE_Q4_K => {
                if !n.is_multiple_of(QK_K) {
                    return None;
                }
                Some((n / QK_K) * Q4_K_BLOCK_BYTES)
            }
            GGML_TYPE_Q5_K => {
                if !n.is_multiple_of(QK_K) {
                    return None;
                }
                Some((n / QK_K) * Q5_K_BLOCK_BYTES)
            }
            GGML_TYPE_Q6_K => {
                if !n.is_multiple_of(QK_K) {
                    return None;
                }
                Some((n / QK_K) * Q6_K_BLOCK_BYTES)
            }
            _ => None,
        }
    }

    /// Logical shape (reversed from GGUF order = [out_rows, in_cols] for 2D).
    fn logical_shape(&self) -> Vec<u64> {
        if self.dims.len() <= 1 {
            self.dims.clone()
        } else {
            self.dims.iter().rev().cloned().collect()
        }
    }

    fn to_native_dtype(&self) -> Option<NativeTensorDataType> {
        match self.ggml_type {
            GGML_TYPE_F32 => Some(NativeTensorDataType::F32),
            GGML_TYPE_I32 => Some(NativeTensorDataType::U32),
            GGML_TYPE_F16 => Some(NativeTensorDataType::F16),
            GGML_TYPE_BF16 => Some(NativeTensorDataType::Bf16),
            GGML_TYPE_Q8_0 => Some(NativeTensorDataType::Q8Zero),
            GGML_TYPE_Q4_K => Some(NativeTensorDataType::Q4Km),
            GGML_TYPE_Q5_K => Some(NativeTensorDataType::Q5Km),
            GGML_TYPE_Q6_K => Some(NativeTensorDataType::Q6Km),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// GGUF file header parser
// ---------------------------------------------------------------------------

struct GgufHeader {
    kv: HashMap<String, KvValue>,
    tensors: Vec<GgufTensorInfo>,
    /// Absolute byte offset where the tensor data section begins.
    data_section_offset: u64,
}

fn parse_gguf_header(path: &Path) -> Result<GgufHeader, GgufError> {
    let mut f = fs::File::open(path).map_err(|e| GgufError::Io {
        path: path.to_owned(),
        source: e,
    })?;

    // Magic
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).map_err(|e| GgufError::Io {
        path: path.to_owned(),
        source: e,
    })?;
    if &magic != GGUF_MAGIC {
        return Err(GgufError::BadMagic);
    }

    // Version
    let version = read_u32(&mut f).map_err(|e| GgufError::Io {
        path: path.to_owned(),
        source: e,
    })?;
    if !(2..=3).contains(&version) {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let n_tensors = read_u64(&mut f).map_err(|e| GgufError::Io {
        path: path.to_owned(),
        source: e,
    })?;
    let n_kv = read_u64(&mut f).map_err(|e| GgufError::Io {
        path: path.to_owned(),
        source: e,
    })?;

    // KV pairs
    let mut kv: HashMap<String, KvValue> = HashMap::new();
    for _ in 0..n_kv {
        let key = read_gguf_string(&mut f).map_err(|e| GgufError::Io {
            path: path.to_owned(),
            source: e,
        })?;
        let typ = read_u32(&mut f).map_err(|e| GgufError::Io {
            path: path.to_owned(),
            source: e,
        })?;
        match read_kv_value(&mut f, typ) {
            Ok(Some(val)) => {
                kv.insert(key, val);
            }
            Ok(None) => {} // array or skipped
            Err(e) if e.kind() == io::ErrorKind::InvalidData => {
                return Err(GgufError::UnknownKvType(typ));
            }
            Err(e) => {
                return Err(GgufError::Io {
                    path: path.to_owned(),
                    source: e,
                });
            }
        }
    }

    // Tensor infos
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = read_gguf_string(&mut f).map_err(|e| GgufError::Io {
            path: path.to_owned(),
            source: e,
        })?;
        let n_dims = read_u32(&mut f).map_err(|e| GgufError::Io {
            path: path.to_owned(),
            source: e,
        })?;
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(read_u64(&mut f).map_err(|e| GgufError::Io {
                path: path.to_owned(),
                source: e,
            })?);
        }
        let ggml_type = read_u32(&mut f).map_err(|e| GgufError::Io {
            path: path.to_owned(),
            source: e,
        })?;
        let data_offset = read_u64(&mut f).map_err(|e| GgufError::Io {
            path: path.to_owned(),
            source: e,
        })?;
        tensors.push(GgufTensorInfo {
            name,
            dims,
            ggml_type,
            data_offset,
        });
    }

    // Data section start = current position aligned up to `alignment`
    let alignment = kv
        .get("general.alignment")
        .and_then(|v| v.as_uint())
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
    let header_end = f.stream_position().map_err(|e| GgufError::Io {
        path: path.to_owned(),
        source: e,
    })?;
    let data_section_offset = header_end.div_ceil(alignment) * alignment;

    Ok(GgufHeader {
        kv,
        tensors,
        data_section_offset,
    })
}

// ---------------------------------------------------------------------------
// Architecture → manifest mapping
// ---------------------------------------------------------------------------

/// GGUF tensor name suffix (after `blk.N.`) → AX NativeTensorRole
fn layer_tensor_role(suffix: &str) -> Option<NativeTensorRole> {
    match suffix {
        // Standard (dense) attention
        "attn_norm.weight" => Some(NativeTensorRole::AttentionNorm),
        "attn_q_norm.weight" => Some(NativeTensorRole::AttentionQNorm),
        "attn_k_norm.weight" => Some(NativeTensorRole::AttentionKNorm),
        "attn_q.weight" => Some(NativeTensorRole::AttentionQ),
        "attn_k.weight" => Some(NativeTensorRole::AttentionK),
        "attn_v.weight" => Some(NativeTensorRole::AttentionV),
        "attn_output.weight" => Some(NativeTensorRole::AttentionO),
        // FFN (shared between dense and linear attention layers)
        "ffn_norm.weight" | "post_attention_layernorm.weight" => Some(NativeTensorRole::FfnNorm),
        // post_attention_norm.weight is the post-attn norm in Gemma4 (AttentionPostNorm).
        // For Qwen35 it serves as the pre-FFN norm (no separate ffn_norm.weight); a
        // post-processing step below re-classifies it to FfnNorm in that case.
        "post_attention_norm.weight" => Some(NativeTensorRole::AttentionPostNorm),
        // post_ffw_norm.weight is the post-FFN norm in Gemma4.
        "post_ffw_norm.weight" => Some(NativeTensorRole::FfnPostNorm),
        "ffn_gate.weight" => Some(NativeTensorRole::FfnGate),
        "ffn_up.weight" => Some(NativeTensorRole::FfnUp),
        "ffn_down.weight" => Some(NativeTensorRole::FfnDown),
        "ffn_gate_up.weight" => Some(NativeTensorRole::FfnGateUpPacked),
        // Qwen3.5 linear attention tensors
        "attn_qkv.weight" => Some(NativeTensorRole::LinearAttentionInProjQkv),
        "attn_gate.weight" => Some(NativeTensorRole::LinearAttentionInProjZ),
        "ssm_alpha.weight" => Some(NativeTensorRole::LinearAttentionInProjA),
        "ssm_beta.weight" => Some(NativeTensorRole::LinearAttentionInProjB),
        "ssm_conv1d.weight" => Some(NativeTensorRole::LinearAttentionConv1d),
        "ssm_dt.bias" => Some(NativeTensorRole::LinearAttentionDtBias),
        "ssm_a" => Some(NativeTensorRole::LinearAttentionALog),
        "ssm_norm.weight" => Some(NativeTensorRole::LinearAttentionNorm),
        "ssm_out.weight" => Some(NativeTensorRole::LinearAttentionOutProj),
        _ => None,
    }
}

fn global_tensor_role(name: &str) -> Option<NativeTensorRole> {
    match name {
        "token_embd.weight" => Some(NativeTensorRole::TokenEmbedding),
        "output_norm.weight" => Some(NativeTensorRole::FinalNorm),
        "output.weight" => Some(NativeTensorRole::LmHead),
        _ => None,
    }
}

fn kv_uint(kv: &HashMap<String, KvValue>, key: &str) -> Option<u64> {
    kv.get(key)?.as_uint()
}

fn kv_float(kv: &HashMap<String, KvValue>, key: &str) -> Option<f64> {
    kv.get(key)?.as_float()
}

fn kv_str(kv: &HashMap<String, KvValue>, key: &str) -> Option<String> {
    kv.get(key)?.as_str().map(|s| s.to_owned())
}

/// Try several candidate keys, return the first that resolves to a u64.
fn kv_uint_multi(kv: &HashMap<String, KvValue>, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|k| kv_uint(kv, k))
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Parse a GGUF file and build a `NativeModelArtifacts` that points back into
/// the GGUF file for tensor data. Only Q4_K_M and float tensors are supported.
pub fn load_gguf(path: &Path) -> Result<NativeModelArtifacts, GgufError> {
    let header = parse_gguf_header(path)?;
    let kv = &header.kv;

    // --- Architecture ---
    let arch = kv_str(kv, "general.architecture")
        .ok_or(GgufError::MissingMetadata("general.architecture"))?
        .to_lowercase();

    // Normalise arch variants to the canonical AX family name
    let family = match arch.as_str() {
        "qwen3" => "qwen3_dense",
        "qwen3moe" | "qwen3_moe" => "qwen3_dense",
        "qwen35" | "qwen3.5" | "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" => "qwen35_dense",
        "qwen3next" | "qwen3.6" | "qwen3_6" => "qwen3next_dense",
        "gemma4" => "gemma4_dense",
        _ => return Err(GgufError::UnknownArch(arch.clone())),
    };

    // --- Metadata keys (arch-prefixed with fallbacks) ---
    let arch_keys: Vec<String> = vec![arch.clone(), arch.replace('.', "_"), arch.replace('.', "")];

    let get_arch_uint = |suffix: &str| -> Option<u64> {
        arch_keys
            .iter()
            .find_map(|a| kv_uint(kv, &format!("{a}.{suffix}")))
    };
    let get_arch_float = |suffix: &str| -> Option<f64> {
        arch_keys
            .iter()
            .find_map(|a| kv_float(kv, &format!("{a}.{suffix}")))
    };

    let layer_count =
        get_arch_uint("block_count").ok_or(GgufError::MissingMetadata("block_count"))? as u32;

    // vocab_size: prefer explicit metadata, fall back to the token_embd shape.
    // In GGUF, token_embd shape = [hidden_dim, vocab_size] (ggml order), so
    // shape[1] = vocab_size after reversing → use the last dim in GGUF order.
    let vocab_size = get_arch_uint("vocab_size")
        .or_else(|| {
            header
                .tensors
                .iter()
                .find(|t| t.name == "token_embd.weight")
                .and_then(|t| t.dims.last().copied())
        })
        .ok_or(GgufError::MissingMetadata("vocab_size"))? as u32;

    // hidden_size: prefer metadata, fall back to token_embd shape
    let hidden_size = get_arch_uint("embedding_length")
        .map(|v| v as u32)
        .or_else(|| {
            header
                .tensors
                .iter()
                .find(|t| t.name == "token_embd.weight")
                .and_then(|t| t.logical_shape().get(1).map(|&d| d as u32))
        })
        .ok_or(GgufError::MissingMetadata("embedding_length"))?;

    let attention_head_count = get_arch_uint("attention.head_count")
        .ok_or(GgufError::MissingMetadata("attention.head_count"))?
        as u32;
    let kv_head_count =
        get_arch_uint("attention.head_count_kv").unwrap_or(attention_head_count as u64) as u32;

    // head_dim: prefer explicit key, else hidden_size / head_count
    let attention_head_dim = get_arch_uint("attention.key_length")
        .map(|v| v as u32)
        .unwrap_or_else(|| hidden_size / attention_head_count.max(1));

    let rope_theta = get_arch_float("rope.freq_base").and_then(f64_to_u32);
    // SWA rope theta for ISWA models (e.g. Gemma4 uses 10000.0 for SWA layers).
    let rope_theta_swa = get_arch_float("rope.freq_base_swa").and_then(f64_to_u32);

    let partial_rotary_factor = get_arch_uint("rope.dimension_count")
        .filter(|&dim| attention_head_dim > 0 && dim < attention_head_dim as u64)
        .map(|dim| dim as f32 / attention_head_dim as f32);

    // --- Tensor mapping ---
    // Use a relative path: just the file name so resolve_tensor_path() works.
    let file_name = PathBuf::from(path.file_name().expect("GGUF path must be a file"));

    let mut tensors: Vec<NativeTensorSpec> = Vec::new();
    let mut has_lm_head = false;

    for info in &header.tensors {
        let abs_offset = header.data_section_offset + info.data_offset;
        let length_bytes = match info.byte_length() {
            Some(l) => l,
            None => {
                return Err(GgufError::UnsupportedTensorType(
                    info.ggml_type,
                    info.name.clone(),
                ));
            }
        };
        let dtype = match info.to_native_dtype() {
            Some(d) => d,
            None => {
                return Err(GgufError::UnsupportedTensorType(
                    info.ggml_type,
                    info.name.clone(),
                ));
            }
        };
        if dtype == NativeTensorDataType::Q4Km && info.n_elements() % QK_K != 0 {
            return Err(GgufError::UnalignedQ4K {
                name: info.name.clone(),
                elements: info.n_elements(),
            });
        }

        let logical_shape = info.logical_shape();

        // Match global tensors
        if let Some(role) = global_tensor_role(&info.name) {
            if role == NativeTensorRole::LmHead {
                has_lm_head = true;
            }
            tensors.push(NativeTensorSpec {
                name: info.name.clone(),
                role,
                layer_index: None,
                dtype,
                source_tensor_type: Some(ggml_type_name(info.ggml_type).to_owned()),
                source_quantized: matches!(
                    dtype,
                    NativeTensorDataType::Q4Km
                        | NativeTensorDataType::Q5Km
                        | NativeTensorDataType::Q6Km
                        | NativeTensorDataType::Q8Zero
                ),
                quantization: None,
                quantized_source: None,
                shape: logical_shape,
                file: file_name.clone(),
                offset_bytes: abs_offset,
                length_bytes,
            });
            continue;
        }

        // Match layer tensors: blk.{i}.{suffix}
        if let Some(rest) = info.name.strip_prefix("blk.") {
            if let Some(dot) = rest.find('.') {
                if let Ok(layer_index) = rest[..dot].parse::<u32>() {
                    let suffix = &rest[dot + 1..];
                    if let Some(role) = layer_tensor_role(suffix) {
                        tensors.push(NativeTensorSpec {
                            name: info.name.clone(),
                            role,
                            layer_index: Some(layer_index),
                            dtype,
                            source_tensor_type: Some(ggml_type_name(info.ggml_type).to_owned()),
                            source_quantized: matches!(
                                dtype,
                                NativeTensorDataType::Q4Km
                                    | NativeTensorDataType::Q5Km
                                    | NativeTensorDataType::Q6Km
                                    | NativeTensorDataType::Q8Zero
                            ),
                            quantization: None,
                            quantized_source: None,
                            shape: logical_shape,
                            file: file_name.clone(),
                            offset_bytes: abs_offset,
                            length_bytes,
                        });
                    }
                    // Unknown suffixes are silently skipped (linear attention, MoE router, etc.)
                    continue;
                }
            }
        }
        // Unknown global names are silently skipped
    }

    tensors.sort_by_key(|s| (s.layer_index, format!("{:?}", s.role)));

    // Post-process AttentionPostNorm → FfnNorm reclassification.
    // post_attention_norm.weight is initially parsed as AttentionPostNorm. For models like
    // Qwen35 that have no separate ffn_norm.weight, it acts as the pre-FFN norm instead.
    {
        let layers_with_ffn_norm: std::collections::HashSet<u32> = tensors
            .iter()
            .filter(|t| t.role == NativeTensorRole::FfnNorm)
            .filter_map(|t| t.layer_index)
            .collect();
        for t in tensors.iter_mut() {
            if t.role == NativeTensorRole::AttentionPostNorm {
                if let Some(li) = t.layer_index {
                    if !layers_with_ffn_norm.contains(&li) {
                        t.role = NativeTensorRole::FfnNorm;
                    }
                }
            }
        }
    }

    // Feed-forward intermediate size (optional in GGUF metadata; 0 means unknown)
    let intermediate_size = get_arch_uint("feed_forward_length")
        .map(|v| v as u32)
        .unwrap_or(0);

    // Linear attention (SSM / Mamba-style hybrid) configuration.
    // Key mappings from llama.cpp gguf-py (see qwen35_linear_attention_config):
    //   num_value_heads = ssm.time_step_rank  (or linear_num_value_heads)
    //   num_key_heads   = ssm.group_count     (or linear_num_key_heads)
    //   key_head_dim    = ssm.state_size      (or linear_key_head_dim)
    //   value_head_dim  derived from ssm.inner_size / num_value_heads
    //   conv_kernel_dim = ssm.conv_kernel     (or linear_conv_kernel_dim)
    let linear_attention = {
        let num_value_heads = kv_uint_multi(
            kv,
            &[
                &format!("{arch}.linear_num_value_heads"),
                "qwen35.linear_num_value_heads",
                &format!("{arch}.ssm.time_step_rank"),
            ],
        )
        .map(|v| v as u32);
        let num_key_heads = kv_uint_multi(
            kv,
            &[
                &format!("{arch}.linear_num_key_heads"),
                "qwen35.linear_num_key_heads",
                &format!("{arch}.ssm.group_count"),
            ],
        )
        .map(|v| v as u32);
        let key_head_dim = kv_uint_multi(
            kv,
            &[
                &format!("{arch}.linear_key_head_dim"),
                "qwen35.linear_key_head_dim",
                &format!("{arch}.ssm.state_size"),
            ],
        )
        .map(|v| v as u32);
        let ssm_inner_size = kv_uint_multi(kv, &[&format!("{arch}.ssm.inner_size")]);
        let value_head_dim = kv_uint_multi(
            kv,
            &[
                &format!("{arch}.linear_value_head_dim"),
                "qwen35.linear_value_head_dim",
            ],
        )
        .map(|v| v as u32)
        .or_else(|| {
            // Derive: value_head_dim = ssm_inner_size / num_value_heads
            let inner = ssm_inner_size?;
            let nv = num_value_heads? as u64;
            inner.checked_div(nv).map(|value| value as u32)
        });
        let conv_kernel_dim = kv_uint_multi(
            kv,
            &[
                &format!("{arch}.linear_conv_kernel_dim"),
                "qwen35.linear_conv_kernel_dim",
                &format!("{arch}.ssm.conv_kernel"),
            ],
        )
        .map(|v| v as u32);

        NativeLinearAttentionConfig {
            num_value_heads,
            num_key_heads,
            key_head_dim,
            value_head_dim,
            conv_kernel_dim,
        }
    };

    let manifest = NativeModelManifest {
        schema_version: AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
        model_family: family.to_string(),
        tensor_format: NativeTensorFormat::Gguf,
        source_quantization: None,
        runtime_status: NativeRuntimeStatus::default(),
        layer_count,
        hidden_size,
        intermediate_size,
        attention_head_count,
        attention_head_dim,
        kv_head_count,
        vocab_size,
        tie_word_embeddings: !has_lm_head,
        rope_theta,
        rope_theta_swa,
        query_pre_attn_scalar: None,
        attention_logit_softcap: None,
        attn_output_gate: false,
        partial_rotary_factor,
        attention_value_from_key_layers: Vec::new(),
        attention_v_norm_no_scale_layers: Vec::new(),
        global_head_dim: None,
        sliding_window_size: None,
        layer_types: Vec::new(),
        kv_shared_source_layers: Default::default(),
        final_logit_softcapping: None,
        linear_attention,
        moe: NativeMoeConfig::default(),
        tensors,
    };

    // Use the directory of the GGUF file as the root_dir so that
    // resolve_tensor_path() returns the full path correctly.
    let root_dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    NativeModelArtifacts::from_manifest_and_root(root_dir, manifest)
        .map_err(|e| GgufError::InvalidManifest(e.to_string()))
}

fn ggml_type_name(t: u32) -> &'static str {
    match t {
        GGML_TYPE_F32 => "f32",
        GGML_TYPE_F16 => "f16",
        GGML_TYPE_BF16 => "bf16",
        GGML_TYPE_Q8_0 => "q8_0",
        GGML_TYPE_Q4_K => "q4_k",
        GGML_TYPE_Q5_K => "q5_k",
        GGML_TYPE_Q6_K => "q6_k",
        _ => "unknown",
    }
}
