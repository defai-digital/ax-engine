//! Dequantize Q4_K (Q4_K_M) blocks to f32.
//!
//! Q4_K super-block layout (144 bytes -> 256 values):
//!   - 2 bytes: f16 `d` (super-block scale for quantized scales)
//!   - 2 bytes: f16 `dmin` (super-block scale for quantized mins)
//!   - 12 bytes: `scales[12]` -- 8 scale values + 8 min values, 6-bit packed
//!   - 128 bytes: `qs[128]` -- 256 x 4-bit quants (2 per byte)
//!
//! 8 sub-blocks of 32 values each. Processed in 4 pairs:
//!   pair j (j=0..3): sub-blocks 2j and 2j+1
//!     - 32 bytes of qs: low nibble -> sub-block 2j, high nibble -> sub-block 2j+1
//!     - Each sub-block has its own 6-bit scale (sc) and 6-bit min (m)
//!     - value = d * sc * nibble - dmin * m

const Q4_K_BLOCK_SIZE: usize = 256;
const Q4_K_BYTES_PER_BLOCK: usize = 144;

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    assert!(
        src.len().is_multiple_of(Q4_K_BYTES_PER_BLOCK),
        "Q4_K src length {} is not a multiple of block size {}",
        src.len(),
        Q4_K_BYTES_PER_BLOCK
    );

    let n_blocks = src.len() / Q4_K_BYTES_PER_BLOCK;
    let n_values = n_blocks * Q4_K_BLOCK_SIZE;
    assert!(
        dst.len() >= n_values,
        "Q4_K dst length {} too small for {} values",
        dst.len(),
        n_values
    );

    for block_idx in 0..n_blocks {
        let block = &src[block_idx * Q4_K_BYTES_PER_BLOCK..][..Q4_K_BYTES_PER_BLOCK];
        let out = &mut dst[block_idx * Q4_K_BLOCK_SIZE..][..Q4_K_BLOCK_SIZE];

        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16];
        let qs = &block[16..144];

        let mut out_idx = 0;
        let mut q_idx = 0;
        let mut is = 0; // sub-block index

        // 4 pairs of sub-blocks, 64 values per pair
        for _j in 0..4 {
            // Get scale and min for the two sub-blocks in this pair
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);

            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;

            // 32 bytes → 64 values (32 per sub-block)
            for l in 0..32 {
                let byte = qs[q_idx + l];
                out[out_idx + l] = d1 * (byte & 0xF) as f32 - m1;
                out[out_idx + l + 32] = d2 * (byte >> 4) as f32 - m2;
            }

            out_idx += 64;
            q_idx += 32;
            is += 2;
        }
    }
}

/// Extract 6-bit scale and min values from the packed scales array.
///
/// Sub-blocks 0-3: simple 6-bit values from first 8 bytes.
/// Sub-blocks 4-7: 4 bits from bytes 8-11, 2 high bits from bytes 0-7.
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        let sc = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (sc, m)
    } else {
        let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_k_zeros() {
        // d=0, dmin=0 → all zeros regardless of quant values
        let block = [0u8; Q4_K_BYTES_PER_BLOCK];
        let mut dst = [f32::NAN; Q4_K_BLOCK_SIZE];
        dequantize(&block, &mut dst);

        for (i, &v) in dst.iter().enumerate() {
            assert!(v.abs() < 1e-6, "dst[{i}] = {v}, expected 0.0");
        }
    }

    #[test]
    fn test_q4_k_simple_scale() {
        let mut block = [0u8; Q4_K_BYTES_PER_BLOCK];

        // d = 1.0, dmin = 0.0 (no min offset)
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        // dmin = 0
        block[2] = 0;
        block[3] = 0;

        // All sub-block scales = 1 (6-bit), all mins = 0
        // For sub-blocks 0-3: scales[0..4] = 1, scales[4..8] = 0
        block[4] = 1; // scale for sub-block 0
        block[5] = 1; // scale for sub-block 1
        block[6] = 1; // scale for sub-block 2
        block[7] = 1; // scale for sub-block 3
        // scales[4..8] = 0 (mins for sub-blocks 0-3)
        // scales[8..12] = 0 (high bits for sub-blocks 4-7, but scale=0 there)

        // All quant nibbles = 0x55 → low nibble = 5, high nibble = 5
        block[16..144].fill(0x55);

        let mut dst = [0.0f32; Q4_K_BLOCK_SIZE];
        dequantize(&block, &mut dst);

        // First 128 values (sub-blocks 0-3):
        // d * sc * nibble - dmin * m = 1.0 * 1 * 5 - 0 = 5.0
        for (i, &v) in dst[..128].iter().enumerate() {
            assert!((v - 5.0).abs() < 0.01, "dst[{i}] = {v}, expected 5.0");
        }

        // Last 128 values (sub-blocks 4-7): scale=0, so all 0
        for (i, &v) in dst[128..256].iter().enumerate() {
            assert!(v.abs() < 0.01, "dst[{}] = {v}, expected 0.0", i + 128);
        }
    }

    #[test]
    fn test_q4_k_with_min() {
        let mut block = [0u8; Q4_K_BYTES_PER_BLOCK];

        // d = 1.0, dmin = 1.0
        let one = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = one[0];
        block[1] = one[1];
        block[2] = one[0];
        block[3] = one[1];

        // Sub-block 0: scale=2, min=3
        block[4] = 2; // scale for sub-block 0
        block[8] = 3; // min for sub-block 0

        // Quant nibble = 4 for first 32 values
        // value = d * sc * nibble - dmin * m = 1.0 * 2 * 4 - 1.0 * 3 = 8 - 3 = 5.0
        block[16..48].fill(0x44);

        let mut dst = [0.0f32; Q4_K_BLOCK_SIZE];
        dequantize(&block, &mut dst);

        // First 32 values (sub-block 0, low nibble):
        for (i, &v) in dst[..32].iter().enumerate() {
            assert!((v - 5.0).abs() < 0.01, "dst[{i}] = {v}, expected 5.0");
        }
    }

    #[test]
    fn test_q4_k_multiple_blocks() {
        let mut src = vec![0u8; Q4_K_BYTES_PER_BLOCK * 2];

        // Block 0: d=1.0, dmin=0, scale[0]=1, all quants=0x33 → 3*1=3.0
        let one = half::f16::from_f32(1.0).to_le_bytes();
        src[0] = one[0];
        src[1] = one[1];
        src[4] = 1;
        src[5] = 1;
        src[6] = 1;
        src[7] = 1;
        src[16..144].fill(0x33);

        // Block 1: same setup
        src[144] = one[0];
        src[145] = one[1];
        src[148] = 1;
        src[149] = 1;
        src[150] = 1;
        src[151] = 1;
        src[160..288].fill(0x33);

        let mut dst = vec![0.0f32; Q4_K_BLOCK_SIZE * 2];
        dequantize(&src, &mut dst);

        // Both blocks, sub-blocks 0-3: value = 1*1*3 - 0 = 3.0
        for (i, &v) in dst[..128].iter().enumerate() {
            assert!(
                (v - 3.0).abs() < 0.01,
                "block0 dst[{i}] = {v}, expected 3.0"
            );
        }
        for (i, &v) in dst[256..384].iter().enumerate() {
            assert!(
                (v - 3.0).abs() < 0.01,
                "block1 dst[{}] = {v}, expected 3.0",
                i + 256
            );
        }
    }

    #[test]
    fn test_get_scale_min_low_subblocks() {
        let scales = [10, 20, 30, 40, 50, 60, 1, 2, 0, 0, 0, 0];
        // j=0: sc = 10 & 63 = 10, m = 50 & 63 = 50
        assert_eq!(get_scale_min_k4(0, &scales), (10, 50));
        // j=1: sc = 20 & 63 = 20, m = 60 & 63 = 60
        assert_eq!(get_scale_min_k4(1, &scales), (20, 60));
    }
}
