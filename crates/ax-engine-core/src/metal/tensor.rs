use std::mem::size_of;

use crate::model::{NativeTensorDataType, NativeTensorSpec};

use super::MetalNativeTensorBufferBinding;

pub(super) fn tensor_matrix_row_prefix_f32(
    binding: &MetalNativeTensorBufferBinding,
    row: usize,
    width: usize,
) -> Option<Vec<f32>> {
    let (rows, cols) = tensor_matrix_dimensions(&binding.meta.spec)?;
    if row >= rows || width > cols {
        return None;
    }

    let base = row.checked_mul(cols)?;
    let mut values = Vec::with_capacity(width);
    for column in 0..width {
        values.push(tensor_scalar_f32(binding, base + column)?);
    }
    Some(values)
}

pub(super) fn tensor_3d_matrix_row_prefix_f32(
    binding: &MetalNativeTensorBufferBinding,
    outer_index: usize,
    row: usize,
    width: usize,
) -> Option<Vec<f32>> {
    let (outer_dim, row_count, col_count) = tensor_3d_dimensions(&binding.meta.spec)?;
    if outer_index >= outer_dim || row >= row_count || width > col_count {
        return None;
    }

    let base = outer_index
        .checked_mul(row_count)?
        .checked_add(row)?
        .checked_mul(col_count)?;
    let mut values = Vec::with_capacity(width);
    for column in 0..width {
        values.push(tensor_scalar_f32(binding, base + column)?);
    }
    Some(values)
}

pub(super) fn tensor_prefix_f32(
    binding: &MetalNativeTensorBufferBinding,
    width: usize,
) -> Option<Vec<f32>> {
    if width > tensor_element_count(&binding.meta.spec)? {
        return None;
    }

    let mut values = Vec::with_capacity(width);
    for index in 0..width {
        values.push(tensor_scalar_f32(binding, index)?);
    }
    Some(values)
}

pub(super) fn tensor_scalar_f32(
    binding: &MetalNativeTensorBufferBinding,
    element_index: usize,
) -> Option<f32> {
    let bytes = tensor_buffer_bytes(binding)?;
    let element_size = native_dtype_size_bytes(binding.meta.spec.dtype);
    let byte_offset = element_index.checked_mul(element_size)?;
    let end = byte_offset.checked_add(element_size)?;
    decode_native_tensor_scalar(binding.meta.spec.dtype, bytes.get(byte_offset..end)?)
}

pub(super) fn tensor_buffer_bytes(binding: &MetalNativeTensorBufferBinding) -> Option<&[u8]> {
    let length = usize::try_from(binding.meta.spec.length_bytes).ok()?;
    (binding.bytes.len() >= length).then_some(&binding.bytes[..length])
}

pub(super) fn tensor_matrix_dimensions(spec: &NativeTensorSpec) -> Option<(usize, usize)> {
    if spec.shape.len() != 2 {
        return None;
    }

    Some((
        usize::try_from(*spec.shape.first()?).ok()?,
        usize::try_from(*spec.shape.get(1)?).ok()?,
    ))
}

pub(super) fn tensor_3d_dimensions(spec: &NativeTensorSpec) -> Option<(usize, usize, usize)> {
    if spec.shape.len() != 3 {
        return None;
    }

    Some((
        usize::try_from(*spec.shape.first()?).ok()?,
        usize::try_from(*spec.shape.get(1)?).ok()?,
        usize::try_from(*spec.shape.get(2)?).ok()?,
    ))
}

pub(super) fn tensor_element_count(spec: &NativeTensorSpec) -> Option<usize> {
    spec.shape.iter().try_fold(1_usize, |count, dim| {
        count.checked_mul(usize::try_from(*dim).ok()?)
    })
}

pub(super) fn native_dtype_size_bytes(dtype: NativeTensorDataType) -> usize {
    match dtype {
        NativeTensorDataType::F16 | NativeTensorDataType::Bf16 => 2,
        NativeTensorDataType::F32 => 4,
        NativeTensorDataType::I8 | NativeTensorDataType::U8 => 1,
        // Quantized block types have no simple per-element byte size.
        // Callers must handle row offsets via q4km_row_byte_offset() or dequantize at load time.
        NativeTensorDataType::U32 => 4,
        NativeTensorDataType::Q4Km
        | NativeTensorDataType::Q5Km
        | NativeTensorDataType::Q6Km
        | NativeTensorDataType::Q8Zero => 0,
    }
}

pub(super) fn native_dense_effective_dtype(dtype: NativeTensorDataType) -> NativeTensorDataType {
    match dtype {
        NativeTensorDataType::I8 | NativeTensorDataType::U8 => NativeTensorDataType::F32,
        _ => dtype,
    }
}

/// Byte offset into a Q4Km weight buffer for a given row offset and row width.
/// row_width must be a multiple of 256 (QK_K).
pub(super) fn q4km_row_byte_offset(row_offset: usize, row_width: usize) -> Option<usize> {
    let n_blocks = row_width / 256;
    row_offset.checked_mul(n_blocks)?.checked_mul(144)
}

/// Exact Rust port of llama.cpp get_scale_min_k4.
fn q4km_get_scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, mn)
    }
}

/// Dequantize one 144-byte block_q4_K block -> 256 F16 values (512 bytes).
/// Exact port of llama.cpp dequantize_row_q4_K.
fn dequantize_q4km_block_to_f16(block: &[u8]) -> [u16; 256] {
    let d = decode_f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = decode_f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qs = &block[16..144];
    let mut out = [0u16; 256];
    let mut q_ptr = 0_usize;
    let mut elem = 0_usize;
    let mut is = 0_usize;
    for _ in 0..4 {
        let (sc0, m0) = q4km_get_scale_min(is, scales);
        let (sc1, m1) = q4km_get_scale_min(is + 1, scales);
        let d1 = d * sc0 as f32;
        let dm1 = dmin * m0 as f32;
        let d2 = d * sc1 as f32;
        let dm2 = dmin * m1 as f32;
        for l in 0..32 {
            out[elem] = encode_f32_to_f16_bits(d1 * (qs[q_ptr + l] & 0x0F) as f32 - dm1);
            elem += 1;
        }
        for l in 0..32 {
            out[elem] = encode_f32_to_f16_bits(d2 * (qs[q_ptr + l] >> 4) as f32 - dm2);
            elem += 1;
        }
        q_ptr += 32;
        is += 2;
    }
    out
}

/// Dequantize Q8_0 tensor to F16. Block = 2 bytes (F16 scale) + 32 int8 values = 34 bytes.
/// Matches llama.cpp dequantize_row_q8_0.
pub(super) fn dequantize_q8zero_tensor_to_f16(
    spec: &NativeTensorSpec,
    bytes: &[u8],
) -> Option<(NativeTensorDataType, Vec<u8>)> {
    let n = tensor_element_count(spec)?;
    if n % 32 != 0 {
        return None;
    }
    let n_blocks = n / 32;
    if bytes.len() < n_blocks * 34 {
        return None;
    }
    let mut f16: Vec<u8> = Vec::with_capacity(n * 2);
    for b in 0..n_blocks {
        let blk = &bytes[b * 34..];
        let d = decode_f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for i in 0..32 {
            let q = blk[2 + i] as i8;
            f16.extend_from_slice(&encode_f32_to_f16_bits(d * q as f32).to_le_bytes());
        }
    }
    Some((NativeTensorDataType::F16, f16))
}

/// Dequantize Q5_K tensor to F16.
/// Block = 4 (d+dmin F16) + 12 (scales) + 32 (qh 5th bits) + 128 (qs low 4 bits) = 176 bytes.
/// Matches llama.cpp dequantize_row_q5_K + get_scale_min_k4.
pub(super) fn dequantize_q5km_tensor_to_f16(
    spec: &NativeTensorSpec,
    bytes: &[u8],
) -> Option<(NativeTensorDataType, Vec<u8>)> {
    let n = tensor_element_count(spec)?;
    if n % 256 != 0 {
        return None;
    }
    let n_blocks = n / 256;
    if bytes.len() < n_blocks * 176 {
        return None;
    }
    let mut f16: Vec<u8> = Vec::with_capacity(n * 2);
    for b in 0..n_blocks {
        let blk = &bytes[b * 176..];
        let d = decode_f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = decode_f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let scales = &blk[4..16];
        let qh = &blk[16..48]; // 32 bytes, bit masks for 5th bit
        let qs = &blk[48..176]; // 128 bytes, 4 low bits
        let mut ql_ptr = 0_usize;
        let mut is = 0_usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _ in 0..4 {
            let (sc0, m0) = q4km_get_scale_min(is, scales);
            let (sc1, m1) = q4km_get_scale_min(is + 1, scales);
            let d1 = d * sc0 as f32;
            let dm1 = dmin * m0 as f32;
            let d2 = d * sc1 as f32;
            let dm2 = dmin * m1 as f32;
            for l in 0..32_usize {
                let lo = qs[ql_ptr + l] & 0xF;
                let hi: u8 = if (qh[l] & u1) != 0 { 16 } else { 0 };
                f16.extend_from_slice(
                    &encode_f32_to_f16_bits(d1 * (lo + hi) as f32 - dm1).to_le_bytes(),
                );
            }
            for l in 0..32_usize {
                let lo = qs[ql_ptr + l] >> 4;
                let hi: u8 = if (qh[l] & u2) != 0 { 16 } else { 0 };
                f16.extend_from_slice(
                    &encode_f32_to_f16_bits(d2 * (lo + hi) as f32 - dm2).to_le_bytes(),
                );
            }
            ql_ptr += 32;
            is += 2;
            u1 = u1.wrapping_shl(2);
            u2 = u2.wrapping_shl(2);
        }
    }
    Some((NativeTensorDataType::F16, f16))
}

/// Dequantize Q6_K tensor to F16.
/// Block = 128 (ql) + 64 (qh) + 16 (scales int8) + 2 (d F16) = 210 bytes per 256 elements.
/// Matches llama.cpp dequantize_row_q6_K.
pub(super) fn dequantize_q6km_tensor_to_f16(
    spec: &NativeTensorSpec,
    bytes: &[u8],
) -> Option<(NativeTensorDataType, Vec<u8>)> {
    let n = tensor_element_count(spec)?;
    if n % 256 != 0 {
        return None;
    }
    let n_blocks = n / 256;
    if bytes.len() < n_blocks * 210 {
        return None;
    }
    let mut f16: Vec<u8> = Vec::with_capacity(n * 2);
    for b in 0..n_blocks {
        let blk = &bytes[b * 210..];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let sc = &blk[192..208]; // int8 sub-block scales
        let d = decode_f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
        // Two outer passes of 128 elements each.
        let mut ql_off = 0_usize;
        let mut qh_off = 0_usize;
        let mut sc_off = 0_usize;
        for _ in 0..2 {
            let mut result = [0f32; 128];
            for l in 0..32_usize {
                let is = l / 16;
                let q1 =
                    (((ql[ql_off + l] & 0xF) | ((qh[qh_off + l] & 3) << 4)) as i8).wrapping_sub(32);
                let q2 = (((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8)
                    .wrapping_sub(32);
                let q3 = (((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8)
                    .wrapping_sub(32);
                let q4 = (((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8)
                    .wrapping_sub(32);
                result[l] = d * sc[sc_off + is] as i8 as f32 * q1 as f32;
                result[l + 32] = d * sc[sc_off + is + 2] as i8 as f32 * q2 as f32;
                result[l + 64] = d * sc[sc_off + is + 4] as i8 as f32 * q3 as f32;
                result[l + 96] = d * sc[sc_off + is + 6] as i8 as f32 * q4 as f32;
            }
            for &w in &result {
                f16.extend_from_slice(&encode_f32_to_f16_bits(w).to_le_bytes());
            }
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
    Some((NativeTensorDataType::F16, f16))
}

/// Dequantize an entire Q4Km tensor to F16 bytes.
/// Used at model-load time for token embedding tensors.
pub(super) fn dequantize_q4km_tensor_to_f16(
    spec: &NativeTensorSpec,
    bytes: &[u8],
) -> Option<(NativeTensorDataType, Vec<u8>)> {
    let n_elements = tensor_element_count(spec)?;
    if n_elements % 256 != 0 {
        return None;
    }
    let n_blocks = n_elements / 256;
    if bytes.len() < n_blocks * 144 {
        return None;
    }
    let mut f16_bytes: Vec<u8> = Vec::with_capacity(n_elements * 2);
    for b in 0..n_blocks {
        let block = &bytes[b * 144..(b + 1) * 144];
        for f16_bits in dequantize_q4km_block_to_f16(block) {
            f16_bytes.extend_from_slice(&f16_bits.to_le_bytes());
        }
    }
    Some((NativeTensorDataType::F16, f16_bytes))
}

pub(super) fn native_dense_shadow_bytes(
    spec: &NativeTensorSpec,
    source_bytes: &[u8],
) -> Option<(NativeTensorDataType, Vec<u8>)> {
    let native_dtype = native_dense_effective_dtype(spec.dtype);
    if native_dtype == spec.dtype {
        return Some((native_dtype, source_bytes.to_vec()));
    }

    let element_count = tensor_element_count(spec)?;
    let scalar_size = native_dtype_size_bytes(spec.dtype);
    let required_bytes = element_count.checked_mul(scalar_size)?;
    let source_prefix = source_bytes.get(..required_bytes)?;
    let mut promoted = Vec::with_capacity(element_count.checked_mul(size_of::<f32>())?);
    for scalar_bytes in source_prefix.chunks_exact(scalar_size) {
        promoted.extend_from_slice(
            &decode_native_tensor_scalar(spec.dtype, scalar_bytes)?.to_le_bytes(),
        );
    }
    Some((native_dtype, promoted))
}

pub(super) fn decode_native_tensor_scalar(
    dtype: NativeTensorDataType,
    bytes: &[u8],
) -> Option<f32> {
    match dtype {
        NativeTensorDataType::F16 => {
            let raw = u16::from_le_bytes(bytes.try_into().ok()?);
            Some(decode_f16_to_f32(raw))
        }
        NativeTensorDataType::Bf16 => {
            let raw = u16::from_le_bytes(bytes.try_into().ok()?);
            Some(f32::from_bits(u32::from(raw) << 16))
        }
        NativeTensorDataType::F32 => Some(f32::from_le_bytes(bytes.try_into().ok()?)),
        NativeTensorDataType::I8 => Some(i8::from_le_bytes([*bytes.first()?]) as f32),
        NativeTensorDataType::U8 => Some(*bytes.first()? as f32),
        NativeTensorDataType::U32 => {
            let raw = u32::from_le_bytes(bytes.try_into().ok()?);
            Some(raw as f32)
        }
        // Quantized types are not decoded element-by-element; dequantization is at load time.
        NativeTensorDataType::Q4Km
        | NativeTensorDataType::Q5Km
        | NativeTensorDataType::Q6Km
        | NativeTensorDataType::Q8Zero => None,
    }
}

pub(super) fn decode_f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x03ff);

    let f32_bits = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut mantissa_shifted = mantissa;
            let mut exponent_shift = -14_i32;
            while (mantissa_shifted & 0x0400) == 0 {
                mantissa_shifted <<= 1;
                exponent_shift -= 1;
            }
            mantissa_shifted &= 0x03ff;
            let exponent_bits = ((exponent_shift + 127) as u32) << 23;
            sign | exponent_bits | (mantissa_shifted << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => {
            let exponent_bits = (u32::from(exponent) + 112) << 23;
            sign | exponent_bits | (mantissa << 13)
        }
    };

    f32::from_bits(f32_bits)
}

pub(super) fn round_slice_to_native_dtype(values: &mut [f32], dtype: NativeTensorDataType) {
    for value in values.iter_mut() {
        *value = round_f32_to_native_dtype(*value, dtype);
    }
}

pub(super) fn round_f32_to_native_dtype(value: f32, dtype: NativeTensorDataType) -> f32 {
    match dtype {
        NativeTensorDataType::F32
        | NativeTensorDataType::I8
        | NativeTensorDataType::U8
        | NativeTensorDataType::U32
        | NativeTensorDataType::Q4Km
        | NativeTensorDataType::Q5Km
        | NativeTensorDataType::Q6Km
        | NativeTensorDataType::Q8Zero => value,
        NativeTensorDataType::Bf16 => round_f32_to_bf16(value),
        NativeTensorDataType::F16 => decode_f16_to_f32(encode_f32_to_f16_bits(value)),
    }
}

fn round_f32_to_bf16(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    let bits = value.to_bits();
    let rounding_bias = 0x7fff_u32 + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(rounding_bias) & 0xffff_0000)
}

pub(super) fn encode_f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ff_ff;

    if exponent == 0xff {
        if mantissa == 0 {
            return sign | 0x7c00;
        }
        return sign | 0x7e00;
    }

    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }

    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let mantissa_with_hidden_bit = mantissa | 0x80_00_00;
        let shift = 14 - half_exponent;
        let mut half_mantissa = (mantissa_with_hidden_bit >> shift) as u16;
        let round_bit = 1_u32 << (shift - 1);
        let remainder = mantissa_with_hidden_bit & ((1_u32 << shift) - 1);
        if remainder > round_bit || (remainder == round_bit && (half_mantissa & 1) != 0) {
            half_mantissa = half_mantissa.wrapping_add(1);
        }
        return sign | half_mantissa;
    }

    let mut half_mantissa = (mantissa >> 13) as u16;
    let round_bits = mantissa & 0x1fff;
    let mut half_exponent_bits = (half_exponent as u16) << 10;
    if round_bits > 0x1000 || (round_bits == 0x1000 && (half_mantissa & 1) != 0) {
        half_mantissa = half_mantissa.wrapping_add(1);
        if half_mantissa == 0x0400 {
            half_mantissa = 0;
            half_exponent_bits = half_exponent_bits.wrapping_add(0x0400);
            if half_exponent_bits >= 0x7c00 {
                return sign | 0x7c00;
            }
        }
    }

    sign | half_exponent_bits | half_mantissa
}
