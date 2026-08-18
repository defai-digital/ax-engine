//! DeepSeek V4 quantized-checkpoint data conversion (FP8 dense + FP4 experts).
//!
//! The stock `convert_hf_model_dir` flow is metadata-only: it reads safetensors
//! headers and points the manifest at the original files. The real
//! DeepSeek-V4-Flash checkpoint ships quantized, so this module performs a
//! family-scoped data pass before mapping:
//!
//! - **FP8 dense weights** (`F8_E4M3` / `F8_E5M2`) with a sibling
//!   `<name>.scale` sidecar holding per-128-block e8m0 (or f32) scales are
//!   dequantized to BF16, matching llama.cpp `deepseek.py` `dequant_model`
//!   (reference lines 682-710: scale shape `[ceil(out/128), ceil(in/128)]`,
//!   `e8m0` byte `b` decodes to `2^(b-127)`). The BF16 result flows through
//!   the same path as any other dense BF16 weight — runtime load-time
//!   quantization is unchanged.
//! - **FP4 routed experts** `layers.{L}.ffn.experts.{E}.w{1,2,3}.weight`
//!   (packed nibbles, `U8` or `F4`) + `.scale` (e8m0) are stacked into the
//!   MXFP4 layout the engine already consumes for GPT-OSS experts
//!   (`gather_qmm` mode=`mxfp4`, group_size=32, bits=4): a `U32` weight tensor
//!   `[E, out, in/8]` plus a co-located `U8` `.scales` sidecar
//!   `[E, out, in/32]` in the same file.
//!
//! # Nibble order (no re-layout needed)
//!
//! HF safetensors store adjacent FP4 values as low/high nibbles per byte
//! (llama.cpp `deepseek.py` comment at lines 729-731). MLX's MXFP4 packing —
//! 8 nibbles per little-endian `u32`, value `2k` in the low nibble of byte
//! `k` — is byte-identical, which is why the GPT-OSS loader can view OpenAI
//! blocks directly as `u32` (`load_gpt_oss_openai_mxfp4_split_experts`). The nibble
//! re-layout in llama.cpp (lines 732-736: values 0..15 into low nibbles,
//! 16..31 into high) targets ggml's MXFP4 block format only and does **not**
//! apply here; repacking is pure per-expert stacking with validated shapes.
//!
//! Converted tensors are materialized into a generated safetensors file
//! ([`DEEPSEEK_V4_CONVERTED_SAFETENSORS_FILE`]) inside the model directory;
//! manifest specs point at it. Source `.scale` sidecars are consumed and
//! never appear in the manifest.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::model::{NativeTensorDataType, NativeTensorQuantization, NativeTensorSpec};

use super::{ConvertError, ModelFamily, SafetensorEntry, arch_u64, match_tensor};

/// Generated safetensors file holding dequantized FP8 dense weights and
/// stacked MXFP4 expert tensors. `find_safetensors_files` skips it so a
/// re-conversion never treats converter output as source input.
pub(crate) const DEEPSEEK_V4_CONVERTED_SAFETENSORS_FILE: &str = "model-deepseek-v4-ax.safetensors";

/// FP8 dense weights carry one e8m0/f32 scale per 128x128 block
/// (llama.cpp `deepseek.py` `dequant_fp8_weight`, lines 688-692).
const FP8_SCALE_BLOCK: usize = 128;
/// MXFP4 group: 32 four-bit values share one e8m0 scale byte.
const MXFP4_GROUP_SIZE: usize = 32;
/// Four-bit values per `u32` storage word.
const MXFP4_VALUES_PER_U32: usize = 8;

/// Result of the DeepSeek V4 quantized data pass.
pub(crate) struct DeepseekV4QuantizedConversion {
    /// Manifest specs for the converted tensors (BF16 dense / U32 experts).
    pub specs: Vec<NativeTensorSpec>,
    /// Source tensor names consumed by the pass (FP8 weights, FP4 experts,
    /// and every `.scale` sidecar); `map_tensors` skips them.
    pub consumed: BTreeSet<String>,
}

struct GeneratedTensor {
    name: String,
    dtype: &'static str,
    shape: Vec<u64>,
    data: Vec<u8>,
}

type ExpertTensorPair<'a> = (u32, &'a SafetensorEntry, &'a SafetensorEntry);

/// Decode one FP8 e4m3fn byte (sign / 4-bit exp bias 7 / 3-bit mantissa; the
/// `exp=15, mantissa=7` encoding is NaN, no infinity) to f32.
pub(crate) fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0_f32 } else { 1.0 };
    let exp = u32::from((byte >> 3) & 0x0F);
    let mantissa = u32::from(byte & 0x07);
    let magnitude = if exp == 0 {
        // Subnormal: (mantissa / 8) * 2^(1-7) = mantissa * 2^-9.
        mantissa as f32 * f32::exp2(-9.0)
    } else if exp == 0x0F && mantissa == 0x07 {
        return f32::NAN;
    } else {
        (1.0 + mantissa as f32 / 8.0) * f32::exp2(exp as f32 - 7.0)
    };
    sign * magnitude
}

/// Decode one FP8 e5m2 byte (IEEE-like: bias 15, infinities and NaN).
pub(crate) fn fp8_e5m2_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0_f32 } else { 1.0 };
    let exp = u32::from((byte >> 2) & 0x1F);
    let mantissa = u32::from(byte & 0x03);
    let magnitude = if exp == 0 {
        // Subnormal: (mantissa / 4) * 2^(1-15) = mantissa * 2^-16.
        mantissa as f32 * f32::exp2(-16.0)
    } else if exp == 0x1F {
        return if mantissa == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        };
    } else {
        (1.0 + mantissa as f32 / 4.0) * f32::exp2(exp as f32 - 15.0)
    };
    sign * magnitude
}

/// Decode one e8m0 scale byte: `2^(b - 127)`
/// (llama.cpp `deepseek.py` `_e8m0_to_float`, lines 634-640).
pub(crate) fn e8m0_to_f32(byte: u8) -> f32 {
    f32::exp2(f32::from(byte) - 127.0)
}

/// Round an f32 to BF16 (round-to-nearest-even on the dropped 16 bits).
pub(crate) fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        // Preserve NaN, forcing the quiet bit so it cannot round to infinity.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb);
    (rounded >> 16) as u16
}

/// Scale sidecar payload for [`dequantize_fp8_block_scales`].
pub(crate) enum BlockScaleBytes<'a> {
    /// One e8m0 byte per 128x128 block.
    E8m0(&'a [u8]),
    /// One little-endian f32 per 128x128 block.
    F32Le(&'a [u8]),
}

/// Dequantize an `[out, in]` FP8 matrix with per-128-block scales to BF16
/// bytes (little-endian), mirroring llama.cpp `dequant_fp8_weight`:
/// `scale` has shape `[ceil(out/128), ceil(in/128)]` and each element is
/// broadcast over its 128x128 block (edge blocks are clipped).
pub(crate) fn dequantize_fp8_block_scales(
    weights: &[u8],
    e5m2: bool,
    rows: usize,
    cols: usize,
    scales: BlockScaleBytes<'_>,
) -> Result<Vec<u8>, String> {
    if weights.len() != rows.saturating_mul(cols) {
        return Err(format!(
            "FP8 weight byte length {} does not match shape [{rows}, {cols}]",
            weights.len()
        ));
    }
    let scale_rows = rows.div_ceil(FP8_SCALE_BLOCK);
    let scale_cols = cols.div_ceil(FP8_SCALE_BLOCK);
    let scale_at = |sr: usize, sc: usize| -> Result<f32, String> {
        let index = sr * scale_cols + sc;
        match scales {
            BlockScaleBytes::E8m0(bytes) => bytes
                .get(index)
                .map(|&byte| e8m0_to_f32(byte))
                .ok_or_else(|| {
                    format!("e8m0 scale buffer too short: need index {index} (grid {scale_rows}x{scale_cols})")
                }),
            BlockScaleBytes::F32Le(bytes) => {
                let start = index
                    .checked_mul(4)
                    .ok_or_else(|| "f32 scale index overflowed".to_string())?;
                let chunk = bytes.get(start..start + 4).ok_or_else(|| {
                    format!("f32 scale buffer too short: need index {index} (grid {scale_rows}x{scale_cols})")
                })?;
                let array: [u8; 4] = chunk
                    .try_into()
                    .map_err(|_| "f32 scale chunk is not 4 bytes".to_string())?;
                Ok(f32::from_le_bytes(array))
            }
        }
    };

    let mut out = Vec::with_capacity(rows.saturating_mul(cols).saturating_mul(2));
    for r in 0..rows {
        for c in 0..cols {
            let byte = weights[r * cols + c];
            let weight = if e5m2 {
                fp8_e5m2_to_f32(byte)
            } else {
                fp8_e4m3_to_f32(byte)
            };
            let scale = scale_at(r / FP8_SCALE_BLOCK, c / FP8_SCALE_BLOCK)?;
            out.extend_from_slice(&f32_to_bf16_bits(weight * scale).to_le_bytes());
        }
    }
    Ok(out)
}

/// Stack per-expert MXFP4 payloads in expert-id order. `experts` holds
/// `(packed_weight_bytes, scale_bytes)` per expert with identical shapes
/// (`[rows, packed_cols]` weights, `[rows, n_blocks]` scales). Returns the
/// concatenated weight and scale buffers; byte order is preserved because HF
/// and MLX MXFP4 use the same adjacent low/high nibble packing (module docs).
pub(crate) fn stack_mxfp4_experts(
    experts: &[(&[u8], &[u8])],
    rows: usize,
    packed_cols: usize,
    n_blocks: usize,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let weight_len = rows.saturating_mul(packed_cols);
    let scale_len = rows.saturating_mul(n_blocks);
    let mut weights = Vec::with_capacity(experts.len().saturating_mul(weight_len));
    let mut scales = Vec::with_capacity(experts.len().saturating_mul(scale_len));
    for (index, (weight, scale)) in experts.iter().enumerate() {
        if weight.len() != weight_len {
            return Err(format!(
                "expert {index} packed weight byte length {} does not match [{rows}, {packed_cols}]",
                weight.len()
            ));
        }
        if scale.len() != scale_len {
            return Err(format!(
                "expert {index} scale byte length {} does not match [{rows}, {n_blocks}]",
                scale.len()
            ));
        }
        weights.extend_from_slice(weight);
        scales.extend_from_slice(scale);
    }
    Ok((weights, scales))
}

/// Run the DeepSeek V4 quantized data pass over parsed safetensors headers.
///
/// Returns `Ok(None)` when the checkpoint carries no FP8 weights or FP4
/// experts (fully dequantized / sanitized checkpoints keep the metadata-only
/// Phase-1 behavior).
pub(crate) fn convert_deepseek_v4_quantized_tensors(
    model_dir: &Path,
    model_type: &str,
    config: &serde_json::Value,
    family: &ModelFamily,
    all_tensors: &[SafetensorEntry],
) -> Result<Option<DeepseekV4QuantizedConversion>, ConvertError> {
    let by_name: BTreeMap<&str, &SafetensorEntry> = all_tensors
        .iter()
        .map(|entry| (entry.name.as_str(), entry))
        .collect();

    let mut generated: Vec<GeneratedTensor> = Vec::new();
    let mut consumed: BTreeSet<String> = BTreeSet::new();
    // (layer, projection) -> [(expert_id, weight entry, scale entry)].
    let mut expert_groups: BTreeMap<(u32, char), Vec<ExpertTensorPair<'_>>> = BTreeMap::new();

    // llama.cpp `dequant_model` iterates `.scale` sidecars and looks the
    // weight up by name (deepseek.py:694-703); do the same so sidecars are
    // always consumed together with their weight.
    for entry in all_tensors {
        let Some(base) = entry.name.strip_suffix(".scale") else {
            continue;
        };
        let weight_name = format!("{base}.weight");
        let Some(weight) = by_name.get(weight_name.as_str()).copied() else {
            continue;
        };
        if is_fp8_dtype(&weight.dtype) {
            if let Some((layer, expert_id, proj)) = parse_expert_weight_name(&weight.name) {
                return Err(invalid_quantized_contract(
                    model_type,
                    format!(
                        "per-expert FP8 tensor {} (layer {layer}, expert {expert_id}, w{proj}) cannot be stacked; \
only BF16/F32 per-expert weights (dropped) or FP4 packed experts are supported",
                        weight.name
                    ),
                ));
            }
            // FP8 tensors without a DeepSeek V4 role mapping (the AXQ
            // `mtp.safetensors` sidecar's attention and shared-expert
            // weights) have no manifest home: leave them and their scales
            // to the generic mapping (dropped ledger, fail-loud in strict
            // mode) instead of dequantizing into a spec that cannot map.
            if match_tensor(&weight.name, family).is_none() {
                continue;
            }
            let data = dequantize_fp8_tensor(model_dir, model_type, weight, entry)?;
            generated.push(GeneratedTensor {
                name: weight.name.clone(),
                dtype: "BF16",
                shape: weight.shape.clone(),
                data,
            });
            consumed.insert(weight.name.clone());
            consumed.insert(entry.name.clone());
        } else if is_fp4_packed_dtype(&weight.dtype) {
            let Some((layer, expert_id, proj)) = parse_expert_weight_name(&weight.name) else {
                // A packed 4-bit tensor outside the expert layout is not a
                // known V4 convention; leave both tensors to the generic
                // mapping (the scale lands in the dropped ledger, fail-loud).
                continue;
            };
            if !matches!(entry.dtype.as_str(), "U8" | "F8_E8M0") {
                return Err(invalid_quantized_contract(
                    model_type,
                    format!(
                        "FP4 expert scale {} must be U8/F8_E8M0 (e8m0), got {}",
                        entry.name, entry.dtype
                    ),
                ));
            }
            expert_groups
                .entry((layer, proj))
                .or_default()
                .push((expert_id, weight, entry));
        }
    }

    for ((layer, proj), mut experts) in expert_groups {
        experts.sort_by_key(|(expert_id, _, _)| *expert_id);
        let (weight_tensor, scale_tensor) =
            stack_expert_group(model_dir, model_type, config, layer, proj, &experts)?;
        for (_, weight, scale) in &experts {
            consumed.insert(weight.name.clone());
            consumed.insert(scale.name.clone());
        }
        generated.push(weight_tensor);
        generated.push(scale_tensor);
    }

    if generated.is_empty() {
        return Ok(None);
    }

    // Resolve manifest specs before touching disk so a naming/role problem
    // fails before any generated file is written.
    let mut pending_specs = Vec::new();
    for tensor in &generated {
        // The `.scales` payload rides along in the generated file; the
        // runtime resolves it by name (`take_weight_spec` looks up
        // `<base>.scales`), exactly like mlx-community GPT-OSS checkpoints.
        if tensor.name.ends_with(".scales") {
            continue;
        }
        let (role, layer_index) = match_tensor(&tensor.name, family).ok_or_else(|| {
            invalid_quantized_contract(
                model_type,
                format!(
                    "converted tensor {} has no DeepSeek V4 role mapping",
                    tensor.name
                ),
            )
        })?;
        let (dtype, source_quantized, quantization) = match tensor.dtype {
            "BF16" => (NativeTensorDataType::Bf16, false, None),
            "U32" => (
                NativeTensorDataType::U32,
                true,
                Some(NativeTensorQuantization {
                    mode: "mxfp4".to_string(),
                    group_size: MXFP4_GROUP_SIZE as u32,
                    bits: 4,
                }),
            ),
            other => {
                return Err(invalid_quantized_contract(
                    model_type,
                    format!("internal error: unexpected generated dtype {other}"),
                ));
            }
        };
        pending_specs.push((
            tensor.name.clone(),
            dtype,
            role,
            layer_index,
            source_quantized,
            quantization,
            tensor.shape.clone(),
        ));
    }

    let layout = write_generated_safetensors(model_dir, &generated)?;
    let mut specs = Vec::with_capacity(pending_specs.len());
    for (name, dtype, role, layer_index, source_quantized, quantization, shape) in pending_specs {
        let (offset_bytes, length_bytes) = layout.get(name.as_str()).copied().ok_or_else(|| {
            invalid_quantized_contract(
                model_type,
                format!("internal error: generated tensor {name} missing from layout"),
            )
        })?;
        specs.push(NativeTensorSpec {
            name,
            role,
            layer_index,
            dtype,
            source_tensor_type: None,
            source_quantized,
            quantization,
            quantized_source: None,
            shape,
            file: PathBuf::from(DEEPSEEK_V4_CONVERTED_SAFETENSORS_FILE),
            offset_bytes,
            length_bytes,
        });
    }

    Ok(Some(DeepseekV4QuantizedConversion { specs, consumed }))
}

fn is_fp8_dtype(dtype: &str) -> bool {
    matches!(dtype, "F8_E4M3" | "F8_E5M2")
}

/// Packed FP4 expert weights: two e2m1 nibbles per byte. `U8` is the HF
/// storage dtype (torch has no FP4 container); `F4` is the safetensors
/// logical FP4 dtype with the same adjacent low/high nibble byte order.
fn is_fp4_packed_dtype(dtype: &str) -> bool {
    matches!(dtype, "U8" | "F4")
}

/// Parse `layers.{L}.ffn.experts.{E}.w{1,2,3}.weight` → `(layer, expert, proj)`.
fn parse_expert_weight_name(name: &str) -> Option<(u32, u32, char)> {
    let rest = name.strip_prefix("layers.")?;
    let dot = rest.find('.')?;
    let layer: u32 = rest[..dot].parse().ok()?;
    let experts = rest[dot + 1..].strip_prefix("ffn.experts.")?;
    let dot = experts.find('.')?;
    let expert_id: u32 = experts[..dot].parse().ok()?;
    let proj = match &experts[dot + 1..] {
        "w1.weight" => '1',
        "w2.weight" => '2',
        "w3.weight" => '3',
        _ => return None,
    };
    Some((layer, expert_id, proj))
}

/// Stacked output projection name: w1 = gate, w2 = down, w3 = up (matches the
/// shared-expert mapping and llama.cpp's FFN_GATE_EXP/FFN_DOWN_EXP/FFN_UP_EXP).
fn stacked_projection_name(proj: char) -> &'static str {
    match proj {
        '1' => "gate",
        '2' => "down",
        _ => "up",
    }
}

fn read_tensor_bytes(model_dir: &Path, entry: &SafetensorEntry) -> Result<Vec<u8>, ConvertError> {
    let path = model_dir.join(&entry.file);
    let mut file = fs::File::open(&path).map_err(|source| ConvertError::ReadFile {
        path: path.clone(),
        source,
    })?;
    file.seek(SeekFrom::Start(entry.offset_bytes))
        .map_err(|source| ConvertError::ReadFile {
            path: path.clone(),
            source,
        })?;
    let length = usize::try_from(entry.length_bytes).map_err(|_| ConvertError::ReadFile {
        path: path.clone(),
        source: std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "tensor {} length {} does not fit in memory",
                entry.name, entry.length_bytes
            ),
        ),
    })?;
    let mut bytes = vec![0u8; length];
    file.read_exact(&mut bytes)
        .map_err(|source| ConvertError::ReadFile { path, source })?;
    Ok(bytes)
}

/// Read + dequantize one FP8 weight / scale pair to BF16 bytes.
fn dequantize_fp8_tensor(
    model_dir: &Path,
    model_type: &str,
    weight: &SafetensorEntry,
    scale: &SafetensorEntry,
) -> Result<Vec<u8>, ConvertError> {
    if weight.shape.len() != 2 {
        return Err(invalid_quantized_contract(
            model_type,
            format!(
                "FP8 weight {} must be 2-D with a 128-block scale grid, got shape {:?}",
                weight.name, weight.shape
            ),
        ));
    }
    let rows = usize::try_from(weight.shape[0])
        .map_err(|_| invalid_quantized_contract(model_type, "FP8 row count overflowed"))?;
    let cols = usize::try_from(weight.shape[1])
        .map_err(|_| invalid_quantized_contract(model_type, "FP8 column count overflowed"))?;
    let expected_scale_shape = [
        weight.shape[0].div_ceil(FP8_SCALE_BLOCK as u64),
        weight.shape[1].div_ceil(FP8_SCALE_BLOCK as u64),
    ];
    if scale.shape != expected_scale_shape {
        return Err(invalid_quantized_contract(
            model_type,
            format!(
                "FP8 scale {} must have shape {:?} (per-128-block grid for weight shape {:?}), got {:?}",
                scale.name, expected_scale_shape, weight.shape, scale.shape
            ),
        ));
    }
    let weight_bytes = read_tensor_bytes(model_dir, weight)?;
    let scale_bytes = read_tensor_bytes(model_dir, scale)?;
    let scales = match scale.dtype.as_str() {
        "F8_E8M0" | "U8" => BlockScaleBytes::E8m0(&scale_bytes),
        "F32" => BlockScaleBytes::F32Le(&scale_bytes),
        other => {
            return Err(invalid_quantized_contract(
                model_type,
                format!(
                    "FP8 scale {} must be F8_E8M0, U8, or F32, got {other}",
                    scale.name
                ),
            ));
        }
    };
    dequantize_fp8_block_scales(&weight_bytes, weight.dtype == "F8_E5M2", rows, cols, scales)
        .map_err(|message| {
            invalid_quantized_contract(model_type, format!("{}: {message}", weight.name))
        })
}

/// Validate and stack one `(layer, projection)` expert group into MXFP4
/// weight + scale payloads.
fn stack_expert_group(
    model_dir: &Path,
    model_type: &str,
    config: &serde_json::Value,
    layer: u32,
    proj: char,
    experts: &[ExpertTensorPair<'_>],
) -> Result<(GeneratedTensor, GeneratedTensor), ConvertError> {
    let label = format!("layers.{layer}.ffn.experts.*.w{proj}");
    let n_routed_experts = arch_u64(config, model_type, "n_routed_experts")
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            invalid_quantized_contract(
                model_type,
                format!("FP4 expert group {label} requires config n_routed_experts"),
            )
        })?;
    if experts.len() != n_routed_experts
        || !experts
            .iter()
            .enumerate()
            .all(|(index, (expert_id, _, _))| usize::try_from(*expert_id).ok() == Some(index))
    {
        let found: Vec<u32> = experts.iter().map(|(expert_id, _, _)| *expert_id).collect();
        return Err(invalid_quantized_contract(
            model_type,
            format!(
                "FP4 expert group {label} must provide experts 0..{n_routed_experts} contiguously, got {found:?}"
            ),
        ));
    }

    let first = experts
        .first()
        .map(|(_, weight, _)| *weight)
        .ok_or_else(|| {
            invalid_quantized_contract(model_type, format!("FP4 expert group {label} is empty"))
        })?;
    if first.shape.len() != 2 {
        return Err(invalid_quantized_contract(
            model_type,
            format!(
                "FP4 expert weight {} must be 2-D, got shape {:?}",
                first.name, first.shape
            ),
        ));
    }
    let rows_u64 = first.shape[0];
    // Logical column count: `U8` rows hold packed bytes (2 values/byte),
    // `F4` rows carry the logical count (same byte packing).
    let logical_cols_u64 = if first.dtype == "F4" {
        first.shape[1]
    } else {
        first.shape[1].saturating_mul(2)
    };
    if !logical_cols_u64.is_multiple_of(MXFP4_GROUP_SIZE as u64) || logical_cols_u64 == 0 {
        return Err(invalid_quantized_contract(
            model_type,
            format!(
                "FP4 expert weight {} must have a multiple of {MXFP4_GROUP_SIZE} logical columns, got {logical_cols_u64}",
                first.name
            ),
        ));
    }
    let packed_cols_u64 = logical_cols_u64 / 2;
    let n_blocks_u64 = logical_cols_u64 / MXFP4_GROUP_SIZE as u64;

    let mut payloads: Vec<(&[u8], &[u8])> = Vec::with_capacity(experts.len());
    let mut weight_buffers = Vec::with_capacity(experts.len());
    let mut scale_buffers = Vec::with_capacity(experts.len());
    for (_, weight, scale) in experts {
        if weight.shape != first.shape || weight.dtype != first.dtype {
            return Err(invalid_quantized_contract(
                model_type,
                format!(
                    "FP4 expert weight {} must share shape {:?} and dtype {} with the rest of the group, got {:?} {}",
                    weight.name, first.shape, first.dtype, weight.shape, weight.dtype
                ),
            ));
        }
        if scale.shape != [rows_u64, n_blocks_u64] {
            return Err(invalid_quantized_contract(
                model_type,
                format!(
                    "FP4 expert scale {} must have shape [{rows_u64}, {n_blocks_u64}], got {:?}",
                    scale.name, scale.shape
                ),
            ));
        }
        weight_buffers.push(read_tensor_bytes(model_dir, weight)?);
        scale_buffers.push(read_tensor_bytes(model_dir, scale)?);
    }
    for (weight, scale) in weight_buffers.iter().zip(scale_buffers.iter()) {
        payloads.push((weight.as_slice(), scale.as_slice()));
    }

    let rows = usize::try_from(rows_u64)
        .map_err(|_| invalid_quantized_contract(model_type, "FP4 row count overflowed"))?;
    let packed_cols = usize::try_from(packed_cols_u64)
        .map_err(|_| invalid_quantized_contract(model_type, "FP4 column count overflowed"))?;
    let n_blocks = usize::try_from(n_blocks_u64)
        .map_err(|_| invalid_quantized_contract(model_type, "FP4 block count overflowed"))?;
    let expert_count = experts.len() as u64;

    let (weights, scales) = stack_mxfp4_experts(&payloads, rows, packed_cols, n_blocks)
        .map_err(|message| invalid_quantized_contract(model_type, format!("{label}: {message}")))?;

    let projection = stacked_projection_name(proj);
    let weight_tensor = GeneratedTensor {
        name: format!("layers.{layer}.ffn.experts.{projection}.weight"),
        dtype: "U32",
        shape: vec![
            expert_count,
            rows_u64,
            logical_cols_u64 / MXFP4_VALUES_PER_U32 as u64,
        ],
        data: weights,
    };
    let scale_tensor = GeneratedTensor {
        name: format!("layers.{layer}.ffn.experts.{projection}.scales"),
        dtype: "U8",
        shape: vec![expert_count, rows_u64, n_blocks_u64],
        data: scales,
    };
    Ok((weight_tensor, scale_tensor))
}

/// Serialize the generated tensors as one safetensors file in the model
/// directory (atomic temp-file + rename) and return each tensor's absolute
/// `(offset, length)` within it.
fn write_generated_safetensors<'a>(
    model_dir: &Path,
    tensors: &'a [GeneratedTensor],
) -> Result<BTreeMap<&'a str, (u64, u64)>, ConvertError> {
    let final_path = model_dir.join(DEEPSEEK_V4_CONVERTED_SAFETENSORS_FILE);
    let mut header = BTreeMap::new();
    let mut layout: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
    let mut offset = 0u64;
    for tensor in tensors {
        let length = tensor.data.len() as u64;
        header.insert(
            tensor.name.clone(),
            serde_json::json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [offset, offset + length],
            }),
        );
        layout.insert(tensor.name.as_str(), (offset, length));
        offset = offset.saturating_add(length);
    }
    let header_json = serde_json::to_vec(&header).map_err(|source| ConvertError::ParseJson {
        path: final_path.clone(),
        source,
    })?;
    let data_base = 8u64.saturating_add(header_json.len() as u64);
    for value in layout.values_mut() {
        value.0 = value.0.saturating_add(data_base);
    }

    let temp_path = model_dir.join(format!(
        ".{DEEPSEEK_V4_CONVERTED_SAFETENSORS_FILE}.tmp-{}",
        std::process::id()
    ));
    let write_result = (|| -> std::io::Result<()> {
        let mut file = fs::File::create(&temp_path)?;
        file.write_all(&(header_json.len() as u64).to_le_bytes())?;
        file.write_all(&header_json)?;
        for tensor in tensors {
            file.write_all(&tensor.data)?;
        }
        file.flush()?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(source) = write_result {
        let _ = fs::remove_file(&temp_path);
        return Err(ConvertError::ReadFile {
            path: final_path,
            source,
        });
    }
    fs::rename(&temp_path, &final_path).map_err(|source| ConvertError::ReadFile {
        path: final_path,
        source,
    })?;
    Ok(layout)
}

fn invalid_quantized_contract(model_type: &str, message: impl Into<String>) -> ConvertError {
    ConvertError::InvalidModelContract {
        model_type: model_type.to_string(),
        message: message.into(),
    }
}
