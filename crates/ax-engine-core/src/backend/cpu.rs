use super::Backend;
use super::neon;
use crate::compute::matmul;
use crate::gguf::tensor::GgmlType;
use crate::kv::Qwen35Kv;

/// CPU backend using Apple Accelerate framework (cblas_sgemm) with
/// NEON-fused dequant+matvec for decode (n=1).
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn matmul(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        matmul::matmul_f32(a, b, c, m, n, k);
    }

    fn dequant_matmul(
        &self,
        a_quant: &[u8],
        dtype: GgmlType,
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        // For decode (n=1), use fused NEON kernels to avoid the intermediate
        // f32 allocation. For prefill (n>1), fall back to dequant + BLAS.
        if n == 1 {
            match dtype {
                GgmlType::Q4K => {
                    neon::fused_matvec_q4_k(a_quant, b, c, m, k);
                    return;
                }
                GgmlType::Q5K => {
                    neon::fused_matvec_q5_k(a_quant, b, c, m, k);
                    return;
                }
                GgmlType::Q8_0 => {
                    neon::fused_matvec_q8_0(a_quant, b, c, m, k);
                    return;
                }
                GgmlType::Q6K => {
                    neon::fused_matvec_q6_k(a_quant, b, c, m, k);
                    return;
                }
                _ => {}
            }
        }

        // Fallback: dequantize to f32, then BLAS matmul
        let mut a_f32 = vec![0.0f32; m * k];
        crate::quant::dequantize(dtype, a_quant, &mut a_f32);
        self.matmul(&a_f32, b, c, m, n, k);
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_recurrent_sequence_for_kv(
        &self,
        qkv_batch: &[f32],
        beta_batch: &mut [f32],
        alpha_batch: &mut [f32],
        dt_bias: &[f32],
        a: &[f32],
        conv_kernel: &[f32],
        qwen_kv: &mut Qwen35Kv,
        layer_idx: usize,
        slot_indices: &[usize],
        output_batch: &mut [f32],
        tokens_per_slot: usize,
        cfg: crate::compute::gdn::Qwen35RecurrentConfig,
    ) {
        let mut prepared_state_batch =
            qwen_kv.prepare_recurrent_state_batch(slot_indices, layer_idx);
        self.note_qwen35_prepared_state_batch_kind(prepared_state_batch.kind());
        let mut state_batch = prepared_state_batch.state_batch();
        self.qwen35_recurrent_sequence(
            qkv_batch,
            beta_batch,
            alpha_batch,
            dt_bias,
            a,
            conv_kernel,
            &mut state_batch,
            output_batch,
            tokens_per_slot,
            cfg,
        );
        prepared_state_batch.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_matmul() {
        let backend = CpuBackend;
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        backend.matmul(&a, &b, &mut c, 2, 2, 2);
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        assert!((c[0] - 19.0).abs() < 1e-4);
        assert!((c[1] - 22.0).abs() < 1e-4);
        assert!((c[2] - 43.0).abs() < 1e-4);
        assert!((c[3] - 50.0).abs() < 1e-4);
    }

    #[test]
    fn test_cpu_dequant_matmul_q8_0_fused() {
        let backend = CpuBackend;

        // Q8_0: d=1.0, qs=2, b=1.0 → dot = 32 * 2 = 64
        let mut block = [0u8; 34];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        block[2..34].fill(2i8 as u8);

        let b = [1.0f32; 32];
        let mut c = [0.0f32; 1];

        backend.dequant_matmul(&block, GgmlType::Q8_0, &b, &mut c, 1, 1, 32);
        assert!((c[0] - 64.0).abs() < 0.5, "expected 64.0, got {}", c[0]);
    }

    #[test]
    fn test_cpu_dequant_matmul_q6_k_fused() {
        let backend = CpuBackend;

        // Q6_K with d=0 → all zeros
        let block = vec![0u8; 210];
        let b = [1.0f32; 256];
        let mut c = [0.0f32; 1];

        backend.dequant_matmul(&block, GgmlType::Q6K, &b, &mut c, 1, 1, 256);
        assert!(c[0].abs() < 1e-4, "expected 0, got {}", c[0]);
    }

    #[test]
    fn test_cpu_dequant_matmul_q5_k_fused_matches_dequantized_reference() {
        let backend = CpuBackend;

        let m = 4;
        let k = 512;
        let blocks_per_row = k / 256;

        let mut quant_data = Vec::new();
        for row in 0..m {
            for blk in 0..blocks_per_row {
                let mut block = vec![0u8; 176];
                let d_val = (row as f32 + 1.0) * 0.05 + blk as f32 * 0.02;
                let d_bytes = half::f16::from_f32(d_val).to_le_bytes();
                block[0] = d_bytes[0];
                block[1] = d_bytes[1];
                let dmin_val = blk as f32 * 0.01;
                let dmin_bytes = half::f16::from_f32(dmin_val).to_le_bytes();
                block[2] = dmin_bytes[0];
                block[3] = dmin_bytes[1];
                for i in 0..8 {
                    block[4 + (i % 4)] = ((row + i) % 8 + 1) as u8;
                    block[8 + (i % 4)] = ((blk + i) % 4) as u8;
                }
                for (i, b) in block[16..48].iter_mut().enumerate() {
                    *b = ((row * 5 + blk * 3 + i) % 256) as u8;
                }
                for (i, b) in block[48..176].iter_mut().enumerate() {
                    *b = ((row * 11 + blk * 7 + i) % 256) as u8;
                }
                quant_data.extend(block);
            }
        }

        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01 - 2.56).collect();

        let mut weights = vec![0.0f32; m * k];
        crate::quant::q5_k::dequantize(&quant_data, &mut weights);
        let mut expected = vec![0.0f32; m];
        crate::compute::matmul::matmul_f32(&weights, &x, &mut expected, m, 1, k);

        let mut result = vec![0.0f32; m];
        backend.dequant_matmul(&quant_data, GgmlType::Q5K, &x, &mut result, m, 1, k);

        let max_diff = result
            .iter()
            .zip(expected.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 0.5,
            "Fused Q5_K matvec mismatch: max_diff={max_diff}, result={result:?}, expected={expected:?}"
        );
    }
}
