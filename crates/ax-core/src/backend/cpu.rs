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
                GgmlType::Q4_0 => {
                    neon::fused_matvec_q4_0(a_quant, b, c, m, k);
                    return;
                }
                GgmlType::Q4K => {
                    neon::fused_matvec_q4_k(a_quant, b, c, m, k);
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
        qwen_kv.assert_valid_recurrent_slot_batch(slot_indices, layer_idx);
        let value_dim = cfg.value_dim();
        for (batch_idx, &slot_idx) in slot_indices.iter().enumerate() {
            let token_start = batch_idx * tokens_per_slot;
            let token_end = token_start + tokens_per_slot;
            let qkv_start = token_start * cfg.conv_dim;
            let qkv_end = token_end * cfg.conv_dim;
            let gate_start = token_start * cfg.time_step_rank;
            let gate_end = token_end * cfg.time_step_rank;
            let out_start = token_start * value_dim;
            let out_end = token_end * value_dim;
            let (conv_state, recurrent_state) =
                qwen_kv.recurrent_buffers_for_slot_mut(slot_idx, layer_idx);
            crate::compute::gdn::qwen35_recurrent_sequence(
                &qkv_batch[qkv_start..qkv_end],
                &mut beta_batch[gate_start..gate_end],
                &mut alpha_batch[gate_start..gate_end],
                dt_bias,
                a,
                conv_kernel,
                conv_state,
                recurrent_state,
                &mut output_batch[out_start..out_end],
                tokens_per_slot,
                cfg,
            );
        }
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
    fn test_cpu_dequant_matmul_q4_0_fused() {
        // Verify the fused path is used for n=1 and produces correct results
        let backend = CpuBackend;

        // Create a Q4_0 block: d=1.0, all nibbles=9 → dequant to 1.0
        let mut block = [0u8; 18];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        block[2..18].fill(0x99); // nibble 9 → (9-8)*1 = 1.0

        let b = [1.0f32; 32]; // k=32
        let mut c = [0.0f32; 1]; // m=1, n=1

        backend.dequant_matmul(&block, GgmlType::Q4_0, &b, &mut c, 1, 1, 32);
        assert!((c[0] - 32.0).abs() < 0.5, "expected ~32.0, got {}", c[0]);
    }

    #[test]
    fn test_cpu_dequant_matmul_q4_0_fallback() {
        // Verify n>1 falls back to dequant+BLAS
        let backend = CpuBackend;

        // m=1, n=2, k=32 → not a matvec, should use fallback
        let mut block = [0u8; 18];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        block[2..18].fill(0x88); // nibble 8 → (8-8)*1 = 0

        let b = [1.0f32; 64]; // k=32, n=2 → b is 32×2
        let mut c = [0.0f32; 2]; // m=1, n=2

        backend.dequant_matmul(&block, GgmlType::Q4_0, &b, &mut c, 1, 2, 32);
        // All dequanted values are 0, so dot products should be 0
        assert!(c[0].abs() < 1e-4);
        assert!(c[1].abs() < 1e-4);
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
}
