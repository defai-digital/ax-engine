pub mod q4_k;
pub mod q5_0;
pub mod q5_1;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;

use crate::gguf::tensor::GgmlType;

/// Whether `dequantize` has a concrete implementation for this dtype.
pub fn dequant_supported(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q4K
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q5K
            | GgmlType::Q6K
            | GgmlType::Q8_0
    )
}

/// Dequantize a block of quantized data to f32.
///
/// Supports F32, F16, Q4_K, Q5_K, Q6_K, and Q8_0.
pub fn dequantize(dtype: GgmlType, src: &[u8], dst: &mut [f32]) {
    match dtype {
        GgmlType::F32 => {
            // Direct copy — reinterpret bytes as f32
            let floats = bytemuck_cast_f32(src);
            dst[..floats.len()].copy_from_slice(floats);
        }
        GgmlType::F16 => dequantize_f16(src, dst),
        GgmlType::BF16 => dequantize_bf16(src, dst),
        GgmlType::Q4K => q4_k::dequantize(src, dst),
        GgmlType::Q5K => q5_k::dequantize(src, dst),
        GgmlType::Q6K => q6_k::dequantize(src, dst),
        GgmlType::Q5_0 => q5_0::dequantize(src, dst),
        GgmlType::Q5_1 => q5_1::dequantize(src, dst),
        GgmlType::Q8_0 => q8_0::dequantize(src, dst),
        _ => {
            // Fail-safe fallback: avoid panicking in inference hot paths.
            // Callers should validate dtypes up front and treat this as unsupported.
            tracing::error!(
                ?dtype,
                "unsupported quant dtype in dequantize; filling output with zeros"
            );
            dst.fill(0.0);
        }
    }
}

/// Dequantize BF16 (bfloat16) to F32.
///
/// BF16 stores the sign + the 8-bit exponent + 7-bit mantissa of an F32.
/// Converting to F32 is a zero-extension: place the 16 bits in the upper half
/// of a 32-bit word and zero the lower 16 bits.
fn dequantize_bf16(src: &[u8], dst: &mut [f32]) {
    assert!(
        src.len().is_multiple_of(2),
        "BF16 src length {} is not even",
        src.len()
    );
    let n = src.len() / 2;
    assert!(
        dst.len() >= n,
        "BF16 dst too small: need {n}, have {}",
        dst.len()
    );
    for i in 0..n {
        let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
        dst[i] = f32::from_bits((bits as u32) << 16);
    }
}

/// Dequantize F16 (IEEE 754 half-precision) to F32.
fn dequantize_f16(src: &[u8], dst: &mut [f32]) {
    assert!(
        src.len().is_multiple_of(2),
        "F16 src length {} is not even",
        src.len()
    );
    let n = src.len() / 2;
    assert!(
        dst.len() >= n,
        "F16 dst too small: need {n}, have {}",
        dst.len()
    );
    for i in 0..n {
        let bytes = [src[i * 2], src[i * 2 + 1]];
        dst[i] = half::f16::from_le_bytes(bytes).to_f32();
    }
}

fn bytemuck_cast_f32(src: &[u8]) -> &[f32] {
    assert!(src.len().is_multiple_of(4));
    assert!(
        (src.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()),
        "bytemuck_cast_f32: source pointer {:#x} is not f32-aligned (mmap data should be page-aligned)",
        src.as_ptr() as usize,
    );
    // SAFETY: We've verified length is a multiple of 4 and pointer is f32-aligned.
    // mmap'd GGUF data is page-aligned, satisfying the alignment requirement.
    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_f16_basic() {
        // f16 for 1.0, 2.0, -0.5
        let vals = [1.0f32, 2.0, -0.5];
        let mut src = Vec::new();
        for &v in &vals {
            src.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        let mut dst = vec![0.0f32; 3];
        dequantize(GgmlType::F16, &src, &mut dst);
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                (dst[i] - v).abs() < 1e-3,
                "dst[{i}] = {}, expected {v}",
                dst[i]
            );
        }
    }

    #[test]
    fn test_dequantize_f16_zeros() {
        let src = [0u8; 8]; // 4 f16 zeros
        let mut dst = vec![1.0f32; 4];
        dequantize(GgmlType::F16, &src, &mut dst);
        for &v in &dst {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_dequantize_f32_passthrough() {
        let vals = [1.0f32, -2.5, 0.0, 42.0];
        let src: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut dst = vec![0.0f32; 4];
        dequantize(GgmlType::F32, &src, &mut dst);
        assert_eq!(dst, vals);
    }

    #[test]
    fn test_dequantize_bf16_basic() {
        // BF16 representation of 1.0: 0x3F80 (same top 16 bits as f32 1.0 = 0x3F800000)
        // BF16 representation of -2.0: 0xC000 (f32 -2.0 = 0xC0000000)
        let src: Vec<u8> = vec![0x80, 0x3F, 0x00, 0xC0]; // 1.0 then -2.0 in LE BF16
        let mut dst = vec![0.0f32; 2];
        dequantize(GgmlType::BF16, &src, &mut dst);
        assert!((dst[0] - 1.0).abs() < 1e-6, "expected 1.0, got {}", dst[0]);
        assert!(
            (dst[1] - (-2.0)).abs() < 1e-6,
            "expected -2.0, got {}",
            dst[1]
        );
    }

    #[test]
    fn test_dequantize_bf16_zero() {
        let src = [0u8; 4]; // two BF16 zeros
        let mut dst = vec![1.0f32; 2];
        dequantize(GgmlType::BF16, &src, &mut dst);
        assert_eq!(dst[0], 0.0);
        assert_eq!(dst[1], 0.0);
    }

    #[test]
    fn test_dequantize_unsupported_dtype_fills_zeros() {
        let src = vec![255u8; 32];
        let mut dst = vec![1.0f32; 8];
        dequantize(GgmlType::Q4_1, &src, &mut dst);
        assert!(dst.iter().all(|v| *v == 0.0));
    }
}
