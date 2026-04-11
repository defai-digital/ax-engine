use super::*;

use objc2_foundation::NSUInteger;

pub(super) const DEQUANT_MATVEC_TG: usize = 128;
/// Threadgroup size for the Q4_K NR2 decode matvec pilot.
pub(super) const DEQUANT_MATVEC_Q4K_NR2_TG: usize = 64;
/// Threadgroup size for the baseline Q5_K decode matvec kernel.
pub(super) const DEQUANT_MATVEC_Q5K_TG: usize = 64;
/// Threadgroup size for the Q6_K NR2 decode matvec pilot.
pub(super) const DEQUANT_MATVEC_Q6K_NR2_TG: usize = 64;
/// Number of output rows per threadgroup for the Q4_K NR2 kernel.
pub(super) const Q4K_NR2_ROWS: usize = 4;
/// Number of output rows per threadgroup for the Q5_K NR2 kernel.
pub(super) const Q5K_NR2_ROWS: usize = 4;
/// Number of output rows per threadgroup for the Q6_K NR2 kernel.
pub(super) const Q6K_NR2_ROWS: usize = 4;
/// Threadgroup size for the Q8_0 NR2 decode matvec kernel.
pub(super) const DEQUANT_MATVEC_Q8_0_NR2_TG: usize = 64;
/// Number of output rows per threadgroup for Q8_0 NR2 (2 SGs × 2 rows/SG).
pub(super) const Q8_0_NR2_ROWS: usize = 4;
/// Threadgroup size for the Q8_0 ILP4 decode matvec kernel.
pub(super) const DEQUANT_MATVEC_Q8_0_ILP4_TG: usize = 64;
/// Number of output rows per threadgroup for Q8_0 ILP4 (2 SGs × 1 row/SG).
pub(super) const Q8_0_ILP4_ROWS: usize = 2;
/// Threadgroup size for the Q4_K ILP4 decode matvec kernel.
pub(super) const DEQUANT_MATVEC_Q4K_ILP4_TG: usize = 64;
/// Number of output rows per threadgroup for Q4_K ILP4.
pub(super) const Q4K_ILP4_ROWS: usize = 2;
/// Threadgroup size for the Q6_K ILP4 decode matvec kernel.
pub(super) const DEQUANT_MATVEC_Q6K_ILP4_TG: usize = 64;
/// Number of output rows per threadgroup for Q6_K ILP4.
const Q6K_ILP4_ROWS: usize = 2;

/// Threadgroup size for standalone dequant kernels.
const DEQUANT_TG_SIZE: usize = 256;

/// Tile size for simdgroup dequant+matmul kernels (must match DQ_BM/DQ_BN in shader).
const DQ_TILE: usize = 32;

/// Threadgroup size for simdgroup dequant+matmul kernels (must match DQ_TG in shader).
const DQ_TG: usize = 128;

/// Tile size on M-axis for large batch dequant+matmul kernels (must match DB_BM).
pub(super) const DB_TILE_M: usize = 32;
/// Tile size on N-axis for large batch dequant+matmul kernels (must match DB_BN).
pub(super) const DB_TILE_N: usize = 64;
/// Threadgroup size for large batch dequant+matmul kernels (must match DB_TG).
const DB_TG: usize = 256;
/// 64x64 full-tile M size for f16in fast path (must match D64_BM).
const DB64_TILE_M: usize = 64;
/// 64x64 full-tile N size for f16in fast path (must match D64_BN).
const DB64_TILE_N: usize = 64;
/// Threadgroup size for 64x64 full-tile f16in kernels (must match D64_TG).
const DB64_TG: usize = 256;
/// 64x32 full-tile N size for f16in fast path (must match D32_BN).
const DB32_TILE_N: usize = 32;
/// Threadgroup size for 64x32 full-tile f16in kernels (must match D32_TG).
const DB32_TG: usize = 128;
/// Minimum active threads to enter BM=64 full-tile f16-in kernels.
const BATCH_F16IN_FULL_TILE_MIN_THREADS: usize = 32_768;
/// 32x32 full-tile M size for Q8 small-N fast path.
const DB32_TILE_M: usize = 32;
/// 32x32 full-tile threadgroup size for Q8 small-N fast path.
const DB32X32_TG: usize = 128;
/// Pair-kernel N tile size (must match PB_BN in shader).
const PB_TILE_N: usize = 32;
/// Pair-kernel threadgroup size (must match PB_TG in shader).
const PB_TG: usize = 128;
/// f16-input pair-kernel N tile size (must match P16_BN in shader).
const P16_TILE_N: usize = 64;
/// f16-input pair-kernel threadgroup size (must match P16_TG in shader).
const P16_TG: usize = 256;
/// Tile size on N-axis for small batch kernels (must match SB_BN).
pub(super) const SB_TILE_N: usize = 32;
/// Threadgroup size for small batch kernels (must match SB_TG).
pub(super) const SB_TG: usize = 128;
/// Default routing threshold for choosing small-vs-large batch kernels.
///
/// Disabled by default (`1`) because current small-N kernels regressed the
/// wider prompt-side workloads they were originally tested on. Keep the
/// threshold overridable for narrow `N<=8` route studies.
pub(super) const DEFAULT_BATCH_SMALL_N_THRESHOLD: u32 = 1;
pub(super) const BLOCKED_BC_OUT_FC_INDEX: NSUInteger = 1;
/// Q4_K and Q6_K blocks contain 256 quantized values. K must be a multiple of this.
const Q4_K_BLOCK_VALUES: usize = 256;
const Q5_K_BLOCK_VALUES: usize = 256;
const Q6_K_BLOCK_VALUES: usize = 256;
const Q8_0_BLOCK_VALUES: usize = 32;

fn batch_q4k_blocked_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_BATCH_Q4K_BLOCKED") {
        Ok(v) => parse_bool_env_flag(&v).unwrap_or(true),
        Err(_) => true, // Default ON
    })
}

fn batch_q6k_blocked_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_BATCH_Q6K_BLOCKED") {
        Ok(v) => parse_bool_env_flag(&v).unwrap_or(true),
        Err(_) => true, // Default ON
    })
}

fn batch_q5k_blocked_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_BATCH_Q5K_BLOCKED") {
        Ok(v) => parse_bool_env_flag(&v).unwrap_or(true),
        Err(_) => true, // Default ON
    })
}

fn batch_q8_blocked_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_BATCH_Q8_BLOCKED") {
        Ok(v) => parse_bool_env_flag(&v).unwrap_or(true),
        Err(_) => true,
    })
}

fn batch_q4k_small_n_threshold() -> u32 {
    static THRESHOLD: OnceLock<u32> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("AX_METAL_BATCH_Q4K_SMALL_N_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|v| *v > 1)
            .unwrap_or(DEFAULT_BATCH_SMALL_N_THRESHOLD)
    })
}

fn batch_q6k_small_n_threshold() -> u32 {
    static THRESHOLD: OnceLock<u32> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("AX_METAL_BATCH_Q6K_SMALL_N_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|v| *v > 1)
            .unwrap_or(DEFAULT_BATCH_SMALL_N_THRESHOLD)
    })
}

/// Pre-compiled dequantization compute pipelines.
///
/// Supports standalone dequant (Q4_K → f32), fused dequant+matvec (N=1),
/// and fused dequant+matmul with simdgroup_matrix (N>1).
/// Create once at init time, reuse for all dequant dispatches.
pub struct DequantKernels {
    dequant_q4_k: ComputePipeline,
    dequant_q6_k: ComputePipeline,
    fused_matvec_q5_k: ComputePipeline,
    /// Q5_K decode matvec with llama.cpp-style 4-way block interleaving.
    fused_matvec_q5_k_ilp4: ComputePipeline,
    /// Q5_K decode matvec with 2 rows per simdgroup and TG=64.
    fused_matvec_q5_k_nr2: ComputePipeline,
    /// Dual-output Q5_K decode matvec for gate+up experiments.
    fused_matvec_pair_q5_k: ComputePipeline,
    /// Fused SiLU(gate)*up + down projection for Q5_K decode.
    fused_silu_down_matvec_q5_k: ComputePipeline,
    /// Fused GELU(gate)*up + down projection for Q5_K decode.
    fused_gelu_down_matvec_q5_k: ComputePipeline,
    fused_matvec_q8_0: ComputePipeline,
    /// Q8_0 decode matvec with 2 rows per simdgroup and TG=64 (4 rows per TG).
    fused_matvec_q8_0_nr2: ComputePipeline,
    /// Q8_0 decode matvec with 4-way block interleaving and TG=64 (2 rows per TG).
    fused_matvec_q8_0_ilp4: ComputePipeline,
    /// Dual-output Q8_0 decode matvec for gate+up experiments.
    fused_matvec_pair_q8_0: ComputePipeline,
    /// Fused SiLU(gate)*up + down projection for Q8_0 decode.
    fused_silu_down_matvec_q8_0: ComputePipeline,
    /// Fused GELU(gate)*up + down projection for Q8_0 decode.
    fused_gelu_down_matvec_q8_0: ComputePipeline,
    fused_matvec_q5_0: ComputePipeline,
    fused_matvec_q5_1: ComputePipeline,
    fused_matvec_q4_k: ComputePipeline,
    /// Q4_K decode matvec with 2 rows per simdgroup and TG=64.
    fused_matvec_q4_k_nr2: ComputePipeline,
    /// Dual-output Q4_K decode matvec preserving nr2 geometry.
    fused_matvec_pair_q4_k: ComputePipeline,
    /// Fused SiLU(gate)*up + down projection for Q4_K decode.
    fused_silu_down_matvec_q4_k: ComputePipeline,
    /// Fused GELU(gate)*up + down projection for Q4_K decode.
    fused_gelu_down_matvec_q4_k: ComputePipeline,
    /// Q4_K decode matvec with 4-way block interleaving and TG=64 (2 rows per TG).
    fused_matvec_q4_k_ilp4: ComputePipeline,
    fused_matvec_dense_f16: ComputePipeline,
    fused_matvec_q6_k: ComputePipeline,
    /// Q6_K decode matvec with 2 rows per simdgroup and TG=64.
    fused_matvec_q6_k_nr2: ComputePipeline,
    /// Q6_K decode matvec with 4-way block interleaving and TG=64 (2 rows per TG).
    fused_matvec_q6_k_ilp4: ComputePipeline,
    /// Dual-output Q6_K decode matvec for gate+up experiments.
    fused_matvec_pair_q6_k: ComputePipeline,
    /// Fused SiLU(gate)*up + down projection for Q6_K decode.
    fused_silu_down_matvec_q6_k: ComputePipeline,
    /// Fused GELU(gate)*up + down projection for Q6_K decode.
    fused_gelu_down_matvec_q6_k: ComputePipeline,
    /// Simdgroup-accelerated fused dequant+matmul for Q4_K (N>1, prefill).
    fused_matmul_q4_k: ComputePipeline,
    /// Simdgroup-accelerated fused dequant+matmul for Q6_K (N>1, prefill).
    fused_matmul_q6_k: ComputePipeline,
    /// B-transposed batch dequant+matmul for Q4_K: C[N×M] = B[N×K] × dequant(A[M×K])^T.
    fused_batch_q4_k: ComputePipeline,
    /// Blocked-layout kernel (boundary-specialized): stride-8, 6KB TG, TG=128, 1.33 MACs/load.
    fused_batch_q4_k_blocked: ComputePipeline,
    /// Blocked-layout full-tile specialization with output boundary handling compiled out.
    fused_batch_q4_k_blocked_fulltile: ComputePipeline,
    /// BM=32 blocked variant for small M: 4KB TG, 2× more TGs.
    fused_batch_q4_k_blocked_bm32: ComputePipeline,
    /// BM=32 blocked full-tile specialization.
    fused_batch_q4_k_blocked_bm32_fulltile: ComputePipeline,
    /// Fused SiLU activation + blocked Q4_K down projection (boundary-specialized).
    fused_batch_q4_k_blocked_silu: ComputePipeline,
    /// Fused SiLU activation + blocked Q4_K down projection (full-tile).
    fused_batch_q4_k_blocked_silu_fulltile: ComputePipeline,
    /// Default B-transposed batch dequant+matmul for Q5_K.
    fused_batch_q5_k: ComputePipeline,
    /// Blocked-layout Q5_K kernel (boundary-specialized).
    fused_batch_q5_k_blocked: ComputePipeline,
    /// Blocked-layout Q5_K full-tile specialization.
    fused_batch_q5_k_blocked_fulltile: ComputePipeline,
    /// B-transposed batch dequant+matmul for Q5_K with f16 input and f32 output.
    fused_batch_q5_k_f16in: ComputePipeline,
    /// Blocked-layout Q5_K kernel with f16 input and f32 output (boundary-specialized).
    fused_batch_q5_k_blocked_f16in: ComputePipeline,
    /// Blocked-layout Q5_K f16-input full-tile specialization.
    fused_batch_q5_k_blocked_f16in_fulltile: ComputePipeline,
    /// Small-N B-transposed batch dequant+matmul for Q5_K.
    fused_batch_q5_k_small: ComputePipeline,
    /// 64x64 full-tile fast path for Q5_K with f16 input and f32 output.
    fused_batch_q5_k_f16in_full64: ComputePipeline,
    /// 64x32 full-tile fast path for Q5_K with f16 input and f32 output.
    fused_batch_q5_k_f16in_full32: ComputePipeline,
    /// Tail-N (N<32) fast path for Q5_K with f16 input and f32 output.
    fused_batch_q5_k_f16in_tail32: ComputePipeline,
    /// Small-N B-transposed batch dequant+matmul for Q5_K with f16 input/f32 output.
    fused_batch_q5_k_f16in_small: ComputePipeline,
    /// B-transposed batch dequant+matmul for Q6_K: C[N×M] = B[N×K] × dequant(A[M×K])^T.
    fused_batch_q6_k: ComputePipeline,
    /// Blocked-layout Q6_K kernel (boundary-specialized).
    fused_batch_q6_k_blocked: ComputePipeline,
    /// Blocked-layout Q6_K full-tile specialization.
    fused_batch_q6_k_blocked_fulltile: ComputePipeline,
    /// BM=32 blocked Q6_K variant for small M.
    fused_batch_q6_k_blocked_bm32: ComputePipeline,
    /// BM=32 blocked Q6_K full-tile specialization.
    fused_batch_q6_k_blocked_bm32_fulltile: ComputePipeline,
    /// Fused SiLU activation + blocked Q6_K down projection (boundary-specialized).
    fused_batch_q6_k_blocked_silu: ComputePipeline,
    /// Fused SiLU activation + blocked Q6_K down projection (full-tile).
    fused_batch_q6_k_blocked_silu_fulltile: ComputePipeline,
    /// B-transposed batch dequant+matmul for Q4_K with f16 input and f32 output.
    fused_batch_q4_k_f16in: ComputePipeline,
    /// Full-tile (BM=32, BN=64, TG=256) fast path for Q4_K with f16 input. No out_tile → 12 KB.
    fused_batch_q4_k_f16in_full: ComputePipeline,
    /// B-transposed batch dequant+matmul for Q6_K with f16 input and f32 output.
    fused_batch_q6_k_f16in: ComputePipeline,
    /// B-transposed batch dequant+matmul for Q8_0 with f32 input and f32 output.
    fused_batch_q8_0: ComputePipeline,
    /// B-transposed batch dequant+matmul for Q8_0 with f16 input and f32 output.
    fused_batch_q8_0_f16in: ComputePipeline,
    /// Blocked-layout Q8_0 kernel with f16 input and f32 output (boundary-specialized).
    fused_batch_q8_0_blocked_f16in: ComputePipeline,
    /// Blocked-layout Q8_0 f16-input full-tile specialization.
    fused_batch_q8_0_blocked_f16in_fulltile: ComputePipeline,
    /// Full-tile fast path for Q8_0 with f16 input and f32 output.
    fused_batch_q8_0_f16in_full: ComputePipeline,
    /// 64x64 full-tile fast path for Q8_0 with f16 input and f32 output.
    fused_batch_q8_0_f16in_full64: ComputePipeline,
    /// 64x32 full-tile fast path for Q8_0 with f16 input and f32 output.
    fused_batch_q8_0_f16in_full32: ComputePipeline,
    /// Tail-N (N<32) fast path for Q8_0 with f16 input and f32 output.
    fused_batch_q8_0_f16in_tail32: ComputePipeline,
    /// 32x32 full-tile fast path for Q8_0 with f16 input and f32 output.
    fused_batch_q8_0_f16in_full32x32: ComputePipeline,
    /// Small-N B-transposed batch dequant+matmul for Q8_0 with f16 input/f32 output.
    fused_batch_q8_0_f16in_small: ComputePipeline,
    /// Blocked Q5_0 batch with f16 input (BM=64, BN=32, BK=32).
    fused_batch_q5_0_blocked_f16in: ComputePipeline,
    /// Blocked Q5_1 batch with f16 input (BM=64, BN=32, BK=32).
    fused_batch_q5_1_blocked_f16in: ComputePipeline,
    /// 64x64 full-tile fast path for Q4_K with f16 input and f32 output.
    fused_batch_q4_k_f16in_full64: ComputePipeline,
    /// 64x64 full-tile BK=32 fast path for Q4_K with f16 input and f32 output.
    fused_batch_q4_k_f16in_full64_bk32: ComputePipeline,
    /// 64x64 full-tile fast path for Q6_K with f16 input and f32 output.
    fused_batch_q6_k_f16in_full64: ComputePipeline,
    /// 64x32 full-tile fast path for Q4_K with f16 input and f32 output.
    fused_batch_q4_k_f16in_full32: ComputePipeline,
    /// 64x32 full-tile fast path for Q6_K with f16 input and f32 output.
    fused_batch_q6_k_f16in_full32: ComputePipeline,
    /// 64x32 N-tail boundary kernel for Q4_K with f16 input and f32 output.
    fused_batch_q4_k_f16in_tail32: ComputePipeline,
    /// 64x32 N-tail boundary kernel for Q6_K with f16 input and f32 output.
    fused_batch_q6_k_f16in_tail32: ComputePipeline,
    /// Small-N B-transposed batch dequant+matmul for Q4_K with f16 input/f32 output.
    fused_batch_q4_k_f16in_small: ComputePipeline,
    /// Small-N B-transposed batch dequant+matmul for Q6_K with f16 input/f32 output.
    fused_batch_q6_k_f16in_small: ComputePipeline,
    /// Dual-output B-transposed batch dequant+matmul for Q4_K.
    fused_batch_pair_q4_k: ComputePipeline,
    /// Dual-output B-transposed batch dequant+matmul for Q6_K.
    fused_batch_pair_q6_k: ComputePipeline,
    /// Dual-output B-transposed batch dequant+matmul for Q4_K with f16 input.
    fused_batch_pair_q4_k_f16in: ComputePipeline,
    /// Dual-output B-transposed batch dequant+matmul for Q5_K with f16 input.
    fused_batch_pair_q5_k_f16in: ComputePipeline,
    /// Dual-output B-transposed batch dequant+matmul for Q6_K with f16 input.
    fused_batch_pair_q6_k_f16in: ComputePipeline,
    /// Dual-output B-transposed batch dequant+matmul for Q8_0 with f16 input.
    fused_batch_pair_q8_0_f16in: ComputePipeline,
    /// Full-tile fast path for dual-output Q8_0 with f16 input.
    fused_batch_pair_q8_0_f16in_full: ComputePipeline,
    /// Small-N B-transposed batch dequant+matmul for Q4_K.
    fused_batch_q4_k_small: ComputePipeline,
    /// Small-N B-transposed batch dequant+matmul for Q6_K.
    fused_batch_q6_k_small: ComputePipeline,
    /// Dense half×half batch matmul (B-transposed), float output.
    batch_matmul_btrans_f16_f32: ComputePipeline,
    /// 64x64 full-tile fast path for dense half×half batch matmul.
    batch_matmul_btrans_f16_f32_full64: ComputePipeline,
    /// BN=32/TG=128 Q4_K batch kernel: 12 KB TG memory → 2 TGs/SM (vs 1 for BN=64).
    /// Half A/B tiles + precomputed scales, same paired-nibble extraction as main kernel.
    /// Faster for N < DB_TILE_N (64): 50% fast-path tiles vs 100% boundary tiles.
    ///
    /// Note: rustc may report this as dead code when checking `ax-engine-metal` in isolation,
    /// because the only reads happen through cross-crate call paths (`ax-engine-core`).
    #[allow(dead_code)]
    fused_batch_q4_k_bn32: ComputePipeline,
    // MoE unified dispatch (mul_mat_id).
    pub moe_map0: ComputePipeline,
    pub moe_mul_mat_id_q4_k: ComputePipeline,
    pub moe_mul_mat_id_q4_k_blocked: ComputePipeline,
    pub moe_mul_mat_id_q5_k_blocked: ComputePipeline,
    pub moe_mul_mat_id_q6_k_blocked: ComputePipeline,
    pub moe_mul_mat_id_q6_k: ComputePipeline,
    pub moe_mul_mat_id_q8_0_blocked: ComputePipeline,
    pub moe_mul_mat_id_q8_0: ComputePipeline,
    pub moe_mul_mat_selected_q4_k_blocked: ComputePipeline,
    pub moe_mul_mat_selected_q4_k_matvec: ComputePipeline,
    pub moe_mul_mat_selected_pair_q4_k_blocked: ComputePipeline,
    pub moe_mul_mat_selected_pair_q4_k_matvec: ComputePipeline,
    pub moe_mul_mat_selected_weighted_q4_k_blocked: ComputePipeline,
    pub moe_mul_mat_selected_q5_k_blocked: ComputePipeline,
    pub moe_mul_mat_selected_q5_k_matvec: ComputePipeline,
    pub moe_mul_mat_selected_pair_q5_k_blocked: ComputePipeline,
    pub moe_mul_mat_selected_pair_q5_k_matvec: ComputePipeline,
    pub moe_mul_mat_selected_weighted_q5_k_blocked: ComputePipeline,
    pub moe_mul_mat_selected_pair_q6_k_matvec: ComputePipeline,
    pub moe_mul_mat_selected_pair_q6_k_matvec_nr2: ComputePipeline,
    pub moe_mul_mat_selected_q6_k_matvec: ComputePipeline,
    pub moe_mul_mat_selected_q6_k_matvec_nr2: ComputePipeline,
    pub moe_mul_mat_selected_weighted_q6_k_matvec: ComputePipeline,
    pub moe_mul_mat_selected_weighted_q6_k_matvec_nr2: ComputePipeline,
    pub moe_mul_mat_selected_pair_q8_0_matvec: ComputePipeline,
    pub moe_mul_mat_selected_pair_q8_0_matvec_nr2: ComputePipeline,
    pub moe_mul_mat_selected_q8_0_matvec: ComputePipeline,
    pub moe_mul_mat_selected_q8_0_matvec_nr2: ComputePipeline,
    pub moe_mul_mat_selected_weighted_q8_0_matvec: ComputePipeline,
    pub moe_mul_mat_selected_weighted_q8_0_matvec_nr2: ComputePipeline,
    pub moe_fused_silu_down_selected_weighted_q5_k_matvec: ComputePipeline,
    pub moe_fused_silu_down_selected_weighted_q5_k_matvec_slots8: ComputePipeline,
    pub moe_fused_silu_down_selected_weighted_q5_k_matvec_nr2: ComputePipeline,
    pub moe_fused_silu_down_selected_weighted_q5_k_matvec_slots8_nr2: ComputePipeline,
}

impl DequantKernels {
    pub fn q4_k_matvec_candidate_with_config(
        &self,
        m: u32,
        config: DequantDispatchConfig,
    ) -> MatvecCandidateSelection {
        q4_k_matvec_candidate_selection(m, config)
    }

    pub fn q5_k_matvec_candidate_with_config(
        &self,
        m: u32,
        config: DequantDispatchConfig,
    ) -> MatvecCandidateSelection {
        q5_k_matvec_candidate_selection(m, config)
    }

    pub fn q6_k_matvec_candidate_with_config(
        &self,
        m: u32,
        config: DequantDispatchConfig,
    ) -> MatvecCandidateSelection {
        q6_k_matvec_candidate_selection(m, config)
    }

    fn q4_k_matvec_dispatch_with_config(
        &self,
        m: u32,
        config: DequantDispatchConfig,
    ) -> (
        usize,
        usize,
        &ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    ) {
        let selection = q4_k_matvec_candidate_selection(m, config);
        let pipeline = match selection.candidate {
            MatvecCandidate::Q4KBase => self.fused_matvec_q4_k.state(),
            MatvecCandidate::Q4KNr2 => self.fused_matvec_q4_k_nr2.state(),
            MatvecCandidate::Q4KIlp4 => self.fused_matvec_q4_k_ilp4.state(),
            MatvecCandidate::Q5KBase
            | MatvecCandidate::Q5KIlp4
            | MatvecCandidate::Q5KNr2
            | MatvecCandidate::Q8_0Base
            | MatvecCandidate::Q8_0Nr2
            | MatvecCandidate::Q8_0Ilp4
            | MatvecCandidate::Q6KBase
            | MatvecCandidate::Q6KNr2
            | MatvecCandidate::Q6KIlp4 => {
                unreachable!()
            }
        };
        (
            selection.threadgroups,
            selection.threadgroup_width,
            pipeline,
        )
    }

    fn q6_k_matvec_dispatch_with_config(
        &self,
        m: u32,
        config: DequantDispatchConfig,
    ) -> (
        usize,
        usize,
        &ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    ) {
        let selection = q6_k_matvec_candidate_selection(m, config);
        let pipeline = match selection.candidate {
            MatvecCandidate::Q6KBase => self.fused_matvec_q6_k.state(),
            MatvecCandidate::Q6KNr2 => self.fused_matvec_q6_k_nr2.state(),
            MatvecCandidate::Q6KIlp4 => self.fused_matvec_q6_k_ilp4.state(),
            MatvecCandidate::Q4KBase
            | MatvecCandidate::Q4KNr2
            | MatvecCandidate::Q4KIlp4
            | MatvecCandidate::Q5KBase
            | MatvecCandidate::Q5KIlp4
            | MatvecCandidate::Q5KNr2
            | MatvecCandidate::Q8_0Base
            | MatvecCandidate::Q8_0Nr2
            | MatvecCandidate::Q8_0Ilp4 => unreachable!(),
        };
        (
            selection.threadgroups,
            selection.threadgroup_width,
            pipeline,
        )
    }

    fn q5_k_matvec_dispatch_with_config(
        &self,
        m: u32,
        config: DequantDispatchConfig,
    ) -> (
        usize,
        usize,
        &ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    ) {
        let selection = q5_k_matvec_candidate_selection(m, config);
        let pipeline = match selection.candidate {
            MatvecCandidate::Q5KBase => self.fused_matvec_q5_k.state(),
            MatvecCandidate::Q5KIlp4 => self.fused_matvec_q5_k_ilp4.state(),
            MatvecCandidate::Q5KNr2 => self.fused_matvec_q5_k_nr2.state(),
            MatvecCandidate::Q4KBase
            | MatvecCandidate::Q4KNr2
            | MatvecCandidate::Q4KIlp4
            | MatvecCandidate::Q8_0Base
            | MatvecCandidate::Q8_0Nr2
            | MatvecCandidate::Q8_0Ilp4
            | MatvecCandidate::Q6KBase
            | MatvecCandidate::Q6KNr2
            | MatvecCandidate::Q6KIlp4 => unreachable!(),
        };
        (
            selection.threadgroups,
            selection.threadgroup_width,
            pipeline,
        )
    }

    fn q6_k_selected_matvec_dispatch_with_config(
        &self,
        m: u32,
        config: DequantDispatchConfig,
    ) -> (
        usize,
        usize,
        &ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    ) {
        let selection = q6_k_matvec_candidate_selection(m, config);
        match selection.candidate {
            MatvecCandidate::Q6KBase => (
                selection.threadgroups,
                selection.threadgroup_width,
                self.moe_mul_mat_selected_q6_k_matvec.state(),
            ),
            MatvecCandidate::Q6KNr2 => (
                selection.threadgroups,
                selection.threadgroup_width,
                self.moe_mul_mat_selected_q6_k_matvec_nr2.state(),
            ),
            MatvecCandidate::Q6KIlp4 => {
                unreachable!("q6_k ilp4 is disabled in candidate selection")
            }
            MatvecCandidate::Q4KBase
            | MatvecCandidate::Q4KNr2
            | MatvecCandidate::Q4KIlp4
            | MatvecCandidate::Q5KBase
            | MatvecCandidate::Q5KIlp4
            | MatvecCandidate::Q5KNr2
            | MatvecCandidate::Q8_0Base
            | MatvecCandidate::Q8_0Nr2
            | MatvecCandidate::Q8_0Ilp4 => unreachable!(),
        }
    }

    fn q6_k_selected_pair_matvec_dispatch_with_config(
        &self,
        m: u32,
        config: DequantDispatchConfig,
    ) -> (
        usize,
        usize,
        &ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    ) {
        let selection = q6_k_matvec_candidate_selection(m, config);
        match selection.candidate {
            MatvecCandidate::Q6KBase => (
                selection.threadgroups,
                selection.threadgroup_width,
                self.moe_mul_mat_selected_pair_q6_k_matvec.state(),
            ),
            MatvecCandidate::Q6KNr2 => (
                selection.threadgroups,
                selection.threadgroup_width,
                self.moe_mul_mat_selected_pair_q6_k_matvec_nr2.state(),
            ),
            MatvecCandidate::Q6KIlp4 => {
                unreachable!("q6_k ilp4 is disabled in candidate selection")
            }
            MatvecCandidate::Q4KBase
            | MatvecCandidate::Q4KNr2
            | MatvecCandidate::Q4KIlp4
            | MatvecCandidate::Q5KBase
            | MatvecCandidate::Q5KIlp4
            | MatvecCandidate::Q5KNr2
            | MatvecCandidate::Q8_0Base
            | MatvecCandidate::Q8_0Nr2
            | MatvecCandidate::Q8_0Ilp4 => unreachable!(),
        }
    }

    fn q6_k_selected_weighted_matvec_dispatch_with_config(
        &self,
        m: u32,
        config: DequantDispatchConfig,
    ) -> (
        usize,
        usize,
        &ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    ) {
        let selection = q6_k_matvec_candidate_selection(m, config);
        match selection.candidate {
            MatvecCandidate::Q6KBase => (
                selection.threadgroups,
                selection.threadgroup_width,
                self.moe_mul_mat_selected_weighted_q6_k_matvec.state(),
            ),
            MatvecCandidate::Q6KNr2 => (
                selection.threadgroups,
                selection.threadgroup_width,
                self.moe_mul_mat_selected_weighted_q6_k_matvec_nr2.state(),
            ),
            MatvecCandidate::Q6KIlp4 => {
                unreachable!("q6_k ilp4 is disabled in candidate selection")
            }
            MatvecCandidate::Q4KBase
            | MatvecCandidate::Q4KNr2
            | MatvecCandidate::Q4KIlp4
            | MatvecCandidate::Q5KBase
            | MatvecCandidate::Q5KIlp4
            | MatvecCandidate::Q5KNr2
            | MatvecCandidate::Q8_0Base
            | MatvecCandidate::Q8_0Nr2
            | MatvecCandidate::Q8_0Ilp4 => unreachable!(),
        }
    }

    fn q8_0_selected_matvec_dispatch_with_config(
        &self,
        m: u32,
        config: DequantDispatchConfig,
    ) -> (
        usize,
        usize,
        &ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    ) {
        let selection = q8_0_matvec_candidate_selection(m, config);
        match selection.candidate {
            MatvecCandidate::Q8_0Base => (
                selection.threadgroups,
                selection.threadgroup_width,
                self.moe_mul_mat_selected_q8_0_matvec.state(),
            ),
            MatvecCandidate::Q8_0Nr2 => (
                selection.threadgroups,
                selection.threadgroup_width,
                self.moe_mul_mat_selected_q8_0_matvec_nr2.state(),
            ),
            // Keep the selected path on the stable implementations until a
            // selected-ilp4 routed kernel exists.
            MatvecCandidate::Q8_0Ilp4 => (
                m as usize,
                DEQUANT_MATVEC_TG,
                self.moe_mul_mat_selected_q8_0_matvec.state(),
            ),
            MatvecCandidate::Q4KBase
            | MatvecCandidate::Q4KNr2
            | MatvecCandidate::Q4KIlp4
            | MatvecCandidate::Q5KBase
            | MatvecCandidate::Q5KIlp4
            | MatvecCandidate::Q5KNr2
            | MatvecCandidate::Q6KBase
            | MatvecCandidate::Q6KNr2
            | MatvecCandidate::Q6KIlp4 => unreachable!(),
        }
    }

    fn q8_0_selected_pair_matvec_dispatch_with_config(
        &self,
        m: u32,
        config: DequantDispatchConfig,
    ) -> (
        usize,
        usize,
        &ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    ) {
        let selection = q8_0_matvec_candidate_selection(m, config);
        match selection.candidate {
            MatvecCandidate::Q8_0Base => (
                selection.threadgroups,
                selection.threadgroup_width,
                self.moe_mul_mat_selected_pair_q8_0_matvec.state(),
            ),
            MatvecCandidate::Q8_0Nr2 => (
                selection.threadgroups,
                selection.threadgroup_width,
                self.moe_mul_mat_selected_pair_q8_0_matvec_nr2.state(),
            ),
            MatvecCandidate::Q8_0Ilp4 => (
                m as usize,
                DEQUANT_MATVEC_TG,
                self.moe_mul_mat_selected_pair_q8_0_matvec.state(),
            ),
            MatvecCandidate::Q4KBase
            | MatvecCandidate::Q4KNr2
            | MatvecCandidate::Q4KIlp4
            | MatvecCandidate::Q5KBase
            | MatvecCandidate::Q5KIlp4
            | MatvecCandidate::Q5KNr2
            | MatvecCandidate::Q6KBase
            | MatvecCandidate::Q6KNr2
            | MatvecCandidate::Q6KIlp4 => unreachable!(),
        }
    }

    fn q8_0_selected_weighted_matvec_dispatch_with_config(
        &self,
        m: u32,
        config: DequantDispatchConfig,
    ) -> (
        usize,
        usize,
        &ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    ) {
        let selection = q8_0_matvec_candidate_selection(m, config);
        match selection.candidate {
            MatvecCandidate::Q8_0Base => (
                selection.threadgroups,
                selection.threadgroup_width,
                self.moe_mul_mat_selected_weighted_q8_0_matvec.state(),
            ),
            MatvecCandidate::Q8_0Nr2 => (
                selection.threadgroups,
                selection.threadgroup_width,
                self.moe_mul_mat_selected_weighted_q8_0_matvec_nr2.state(),
            ),
            MatvecCandidate::Q8_0Ilp4 => (
                m as usize,
                DEQUANT_MATVEC_TG,
                self.moe_mul_mat_selected_weighted_q8_0_matvec.state(),
            ),
            MatvecCandidate::Q4KBase
            | MatvecCandidate::Q4KNr2
            | MatvecCandidate::Q4KIlp4
            | MatvecCandidate::Q5KBase
            | MatvecCandidate::Q5KIlp4
            | MatvecCandidate::Q5KNr2
            | MatvecCandidate::Q6KBase
            | MatvecCandidate::Q6KNr2
            | MatvecCandidate::Q6KIlp4 => unreachable!(),
        }
    }

    /// Compile dequant kernels from embedded Metal source.
    pub fn new(device: &MetalDevice) -> anyhow::Result<Self> {
        let dequant_q4_k =
            ComputePipeline::from_source(device.device(), DEQUANT_SHADER_SRC, "dequant_q4_k")
                .context("Failed to compile dequant_q4_k kernel")?;
        let fused_matvec_q5_k = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q5_k",
        )
        .context("Failed to compile dequant_matvec_q5_k kernel")?;
        let fused_matvec_q5_k_ilp4 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q5_k_ilp4",
        )
        .or_else(|err| {
            tracing::warn!(
                error = %err,
                "q5_k ilp4 Metal kernel unavailable; falling back to baseline q5_k matvec"
            );
            ComputePipeline::from_source(device.device(), DEQUANT_SHADER_SRC, "dequant_matvec_q5_k")
        })
        .context("Failed to compile dequant_matvec_q5_k_ilp4 kernel")?;
        let fused_matvec_q5_k_nr2 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q5_k_nr2",
        )
        .or_else(|err| {
            tracing::warn!(
                error = %err,
                "q5_k nr2 Metal kernel unavailable; falling back to baseline q5_k matvec"
            );
            ComputePipeline::from_source(device.device(), DEQUANT_SHADER_SRC, "dequant_matvec_q5_k")
        })
        .context("Failed to compile dequant_matvec_q5_k_nr2 kernel")?;
        let fused_matvec_pair_q5_k = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_pair_q5_k",
        )
        .context("Failed to compile dequant_matvec_pair_q5_k kernel")?;
        let fused_silu_down_matvec_q5_k = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_silu_down_q5_k",
        )
        .context("Failed to compile dequant_matvec_silu_down_q5_k kernel")?;
        let fused_gelu_down_matvec_q5_k = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_gelu_down_q5_k",
        )
        .context("Failed to compile dequant_matvec_gelu_down_q5_k kernel")?;
        let fused_matvec_q8_0 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q8_0",
        )
        .context("Failed to compile dequant_matvec_q8_0 kernel")?;
        let fused_matvec_q8_0_nr2 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q8_0_nr2",
        )
        .context("Failed to compile dequant_matvec_q8_0_nr2 kernel")?;
        let fused_matvec_q8_0_ilp4 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q8_0_ilp4",
        )
        .context("Failed to compile dequant_matvec_q8_0_ilp4 kernel")?;
        let fused_matvec_pair_q8_0 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_pair_q8_0",
        )
        .context("Failed to compile dequant_matvec_pair_q8_0 kernel")?;
        let fused_silu_down_matvec_q8_0 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_silu_down_q8_0",
        )
        .context("Failed to compile dequant_matvec_silu_down_q8_0 kernel")?;
        let fused_gelu_down_matvec_q8_0 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_gelu_down_q8_0",
        )
        .context("Failed to compile dequant_matvec_gelu_down_q8_0 kernel")?;
        let fused_matvec_q5_0 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q5_0",
        )
        .context("Failed to compile dequant_matvec_q5_0 kernel")?;
        let fused_matvec_q5_1 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q5_1",
        )
        .context("Failed to compile dequant_matvec_q5_1 kernel")?;
        let fused_matvec_q4_k = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q4_k",
        )
        .context("Failed to compile dequant_matvec_q4_k kernel")?;
        let fused_matvec_q4_k_nr2 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q4_k_nr2",
        )
        .context("Failed to compile dequant_matvec_q4_k_nr2 kernel")?;
        let fused_matvec_pair_q4_k = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_pair_q4_k",
        )
        .context("Failed to compile dequant_matvec_pair_q4_k kernel")?;
        let fused_silu_down_matvec_q4_k = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_silu_down_q4_k",
        )
        .context("Failed to compile dequant_matvec_silu_down_q4_k kernel")?;
        let fused_gelu_down_matvec_q4_k = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_gelu_down_q4_k",
        )
        .context("Failed to compile dequant_matvec_gelu_down_q4_k kernel")?;
        let fused_matvec_q4_k_ilp4 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q4_k_ilp4",
        )
        .context("Failed to compile dequant_matvec_q4_k_ilp4 kernel")?;
        let fused_matvec_dense_f16 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_dense_f16",
        )
        .context("Failed to compile dequant_matvec_dense_f16 kernel")?;
        let dequant_q6_k =
            ComputePipeline::from_source(device.device(), DEQUANT_SHADER_SRC, "dequant_q6_k")
                .context("Failed to compile dequant_q6_k kernel")?;
        let fused_matvec_q6_k = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q6_k",
        )
        .context("Failed to compile dequant_matvec_q6_k kernel")?;
        let fused_matvec_q6_k_nr2 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q6_k_nr2",
        )
        .context("Failed to compile dequant_matvec_q6_k_nr2 kernel")?;
        let fused_matvec_q6_k_ilp4 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q6_k_ilp4",
        )
        .context("Failed to compile dequant_matvec_q6_k_ilp4 kernel")?;
        let fused_matvec_pair_q6_k = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_pair_q6_k",
        )
        .context("Failed to compile dequant_matvec_pair_q6_k kernel")?;
        let fused_silu_down_matvec_q6_k = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_silu_down_q6_k",
        )
        .context("Failed to compile dequant_matvec_silu_down_q6_k kernel")?;
        let fused_gelu_down_matvec_q6_k = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_gelu_down_q6_k",
        )
        .context("Failed to compile dequant_matvec_gelu_down_q6_k kernel")?;
        let fused_matmul_q4_k = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matmul_simdgroup_q4_k",
        )
        .context("Failed to compile dequant_matmul_simdgroup_q4_k kernel")?;
        let fused_matmul_q6_k = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matmul_simdgroup_q6_k",
        )
        .context("Failed to compile dequant_matmul_simdgroup_q6_k kernel")?;
        let fused_batch_q4_k =
            ComputePipeline::from_source(device.device(), DEQUANT_SHADER_SRC, "dequant_batch_q4_k")
                .context("Failed to compile dequant_batch_q4_k kernel")?;
        let fused_batch_q4_k_blocked = ComputePipeline::from_source_with_constants(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_blocked",
            &[FunctionConstant {
                index: BLOCKED_BC_OUT_FC_INDEX,
                value: FunctionConstantValue::Bool(true),
            }],
        )
        .context("Failed to compile dequant_batch_q4_k_blocked boundary kernel")?;
        let fused_batch_q4_k_blocked_fulltile = ComputePipeline::from_source_with_constants(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_blocked",
            &[FunctionConstant {
                index: BLOCKED_BC_OUT_FC_INDEX,
                value: FunctionConstantValue::Bool(false),
            }],
        )
        .context("Failed to compile dequant_batch_q4_k_blocked fulltile kernel")?;
        let fused_batch_q4_k_blocked_bm32 = ComputePipeline::from_source_with_constants(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_blocked_bm32",
            &[FunctionConstant {
                index: BLOCKED_BC_OUT_FC_INDEX,
                value: FunctionConstantValue::Bool(true),
            }],
        )
        .context("Failed to compile dequant_batch_q4_k_blocked_bm32 boundary kernel")?;
        let fused_batch_q4_k_blocked_bm32_fulltile = ComputePipeline::from_source_with_constants(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_blocked_bm32",
            &[FunctionConstant {
                index: BLOCKED_BC_OUT_FC_INDEX,
                value: FunctionConstantValue::Bool(false),
            }],
        )
        .context("Failed to compile dequant_batch_q4_k_blocked_bm32 fulltile kernel")?;
        let fused_batch_q4_k_blocked_silu = ComputePipeline::from_source_with_constants(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_blocked_silu",
            &[FunctionConstant {
                index: BLOCKED_BC_OUT_FC_INDEX,
                value: FunctionConstantValue::Bool(true),
            }],
        )
        .context("Failed to compile dequant_batch_q4_k_blocked_silu boundary kernel")?;
        let fused_batch_q4_k_blocked_silu_fulltile = ComputePipeline::from_source_with_constants(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_blocked_silu",
            &[FunctionConstant {
                index: BLOCKED_BC_OUT_FC_INDEX,
                value: FunctionConstantValue::Bool(false),
            }],
        )
        .context("Failed to compile dequant_batch_q4_k_blocked_silu fulltile kernel")?;
        let fused_batch_q5_k =
            ComputePipeline::from_source(device.device(), DEQUANT_SHADER_SRC, "dequant_batch_q5_k")
                .context("Failed to compile dequant_batch_q5_k kernel")?;
        let fused_batch_q5_k_blocked = ComputePipeline::from_source_with_constants(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q5_k_blocked",
            &[FunctionConstant {
                index: BLOCKED_BC_OUT_FC_INDEX,
                value: FunctionConstantValue::Bool(true),
            }],
        )
        .context("Failed to compile dequant_batch_q5_k_blocked boundary kernel")?;
        let fused_batch_q5_k_blocked_fulltile = ComputePipeline::from_source_with_constants(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q5_k_blocked",
            &[FunctionConstant {
                index: BLOCKED_BC_OUT_FC_INDEX,
                value: FunctionConstantValue::Bool(false),
            }],
        )
        .context("Failed to compile dequant_batch_q5_k_blocked fulltile kernel")?;
        let fused_batch_q5_k_f16in = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q5_k_f16in",
        )
        .context("Failed to compile dequant_batch_q5_k_f16in kernel")?;
        let fused_batch_q5_k_blocked_f16in = ComputePipeline::from_source_with_constants(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q5_k_blocked_f16in",
            &[FunctionConstant {
                index: BLOCKED_BC_OUT_FC_INDEX,
                value: FunctionConstantValue::Bool(true),
            }],
        )
        .context("Failed to compile dequant_batch_q5_k_blocked_f16in boundary kernel")?;
        let fused_batch_q5_k_blocked_f16in_fulltile = ComputePipeline::from_source_with_constants(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q5_k_blocked_f16in",
            &[FunctionConstant {
                index: BLOCKED_BC_OUT_FC_INDEX,
                value: FunctionConstantValue::Bool(false),
            }],
        )
        .context("Failed to compile dequant_batch_q5_k_blocked_f16in fulltile kernel")?;
        let fused_batch_q5_k_small = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q5_k_small",
        )
        .context("Failed to compile dequant_batch_q5_k_small kernel")?;
        let fused_batch_q5_k_f16in_full64 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q5_k_f16in_full64",
        )
        .context("Failed to compile dequant_batch_q5_k_f16in_full64 kernel")?;
        let fused_batch_q5_k_f16in_full32 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q5_k_f16in_full32",
        )
        .context("Failed to compile dequant_batch_q5_k_f16in_full32 kernel")?;
        let fused_batch_q5_k_f16in_tail32 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q5_k_f16in_tail32",
        )
        .context("Failed to compile dequant_batch_q5_k_f16in_tail32 kernel")?;
        let fused_batch_q5_k_f16in_small = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q5_k_f16in_small",
        )
        .context("Failed to compile dequant_batch_q5_k_f16in_small kernel")?;
        let fused_batch_q6_k =
            ComputePipeline::from_source(device.device(), DEQUANT_SHADER_SRC, "dequant_batch_q6_k")
                .context("Failed to compile dequant_batch_q6_k kernel")?;
        let fused_batch_q6_k_blocked = ComputePipeline::from_source_with_constants(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q6_k_blocked",
            &[FunctionConstant {
                index: BLOCKED_BC_OUT_FC_INDEX,
                value: FunctionConstantValue::Bool(true),
            }],
        )
        .context("Failed to compile dequant_batch_q6_k_blocked boundary kernel")?;
        let fused_batch_q6_k_blocked_fulltile = ComputePipeline::from_source_with_constants(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q6_k_blocked",
            &[FunctionConstant {
                index: BLOCKED_BC_OUT_FC_INDEX,
                value: FunctionConstantValue::Bool(false),
            }],
        )
        .context("Failed to compile dequant_batch_q6_k_blocked fulltile kernel")?;
        let fused_batch_q6_k_blocked_bm32 = ComputePipeline::from_source_with_constants(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q6_k_blocked_bm32",
            &[FunctionConstant {
                index: BLOCKED_BC_OUT_FC_INDEX,
                value: FunctionConstantValue::Bool(true),
            }],
        )
        .context("Failed to compile dequant_batch_q6_k_blocked_bm32 boundary kernel")?;
        let fused_batch_q6_k_blocked_bm32_fulltile = ComputePipeline::from_source_with_constants(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q6_k_blocked_bm32",
            &[FunctionConstant {
                index: BLOCKED_BC_OUT_FC_INDEX,
                value: FunctionConstantValue::Bool(false),
            }],
        )
        .context("Failed to compile dequant_batch_q6_k_blocked_bm32 fulltile kernel")?;
        let fused_batch_q6_k_blocked_silu = ComputePipeline::from_source_with_constants(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q6_k_blocked_silu",
            &[FunctionConstant {
                index: BLOCKED_BC_OUT_FC_INDEX,
                value: FunctionConstantValue::Bool(true),
            }],
        )
        .context("Failed to compile dequant_batch_q6_k_blocked_silu boundary kernel")?;
        let fused_batch_q6_k_blocked_silu_fulltile = ComputePipeline::from_source_with_constants(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q6_k_blocked_silu",
            &[FunctionConstant {
                index: BLOCKED_BC_OUT_FC_INDEX,
                value: FunctionConstantValue::Bool(false),
            }],
        )
        .context("Failed to compile dequant_batch_q6_k_blocked_silu fulltile kernel")?;
        let fused_batch_q4_k_f16in = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_f16in",
        )
        .context("Failed to compile dequant_batch_q4_k_f16in kernel")?;
        let fused_batch_q4_k_f16in_full = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_f16in_full",
        )
        .context("Failed to compile dequant_batch_q4_k_f16in_full kernel")?;
        let fused_batch_q6_k_f16in = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q6_k_f16in",
        )
        .context("Failed to compile dequant_batch_q6_k_f16in kernel")?;
        let fused_batch_q8_0 =
            ComputePipeline::from_source(device.device(), DEQUANT_SHADER_SRC, "dequant_batch_q8_0")
                .context("Failed to compile dequant_batch_q8_0 kernel")?;
        let fused_batch_q8_0_f16in = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q8_0_f16in",
        )
        .context("Failed to compile dequant_batch_q8_0_f16in kernel")?;
        let fused_batch_q8_0_blocked_f16in = ComputePipeline::from_source_with_constants(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q8_0_blocked_f16in",
            &[FunctionConstant {
                index: BLOCKED_BC_OUT_FC_INDEX,
                value: FunctionConstantValue::Bool(true),
            }],
        )
        .context("Failed to compile dequant_batch_q8_0_blocked_f16in boundary kernel")?;
        let fused_batch_q8_0_blocked_f16in_fulltile = ComputePipeline::from_source_with_constants(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q8_0_blocked_f16in",
            &[FunctionConstant {
                index: BLOCKED_BC_OUT_FC_INDEX,
                value: FunctionConstantValue::Bool(false),
            }],
        )
        .context("Failed to compile dequant_batch_q8_0_blocked_f16in fulltile kernel")?;
        let fused_batch_q8_0_f16in_full = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q8_0_f16in_full",
        )
        .context("Failed to compile dequant_batch_q8_0_f16in_full kernel")?;
        let fused_batch_q8_0_f16in_full64 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q8_0_f16in_full64",
        )
        .context("Failed to compile dequant_batch_q8_0_f16in_full64 kernel")?;
        let fused_batch_q8_0_f16in_full32 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q8_0_f16in_full32",
        )
        .context("Failed to compile dequant_batch_q8_0_f16in_full32 kernel")?;
        let fused_batch_q8_0_f16in_tail32 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q8_0_f16in_tail32",
        )
        .context("Failed to compile dequant_batch_q8_0_f16in_tail32 kernel")?;
        let fused_batch_q8_0_f16in_full32x32 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q8_0_f16in_full32x32",
        )
        .context("Failed to compile dequant_batch_q8_0_f16in_full32x32 kernel")?;
        let fused_batch_q8_0_f16in_small = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q8_0_f16in_small",
        )
        .context("Failed to compile dequant_batch_q8_0_f16in_small kernel")?;
        let fused_batch_q5_0_blocked_f16in = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q5_0_blocked_f16in",
        )
        .context("Failed to compile dequant_batch_q5_0_blocked_f16in kernel")?;
        let fused_batch_q5_1_blocked_f16in = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q5_1_blocked_f16in",
        )
        .context("Failed to compile dequant_batch_q5_1_blocked_f16in kernel")?;
        let fused_batch_q4_k_f16in_full64 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_f16in_full64",
        )
        .context("Failed to compile dequant_batch_q4_k_f16in_full64 kernel")?;
        let fused_batch_q6_k_f16in_full64 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q6_k_f16in_full64",
        )
        .context("Failed to compile dequant_batch_q6_k_f16in_full64 kernel")?;
        let fused_batch_q4_k_f16in_full64_bk32 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_f16in_full64_bk32",
        )
        .context("Failed to compile dequant_batch_q4_k_f16in_full64_bk32 kernel")?;
        let fused_batch_q4_k_f16in_small = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_f16in_small",
        )
        .context("Failed to compile dequant_batch_q4_k_f16in_small kernel")?;
        let fused_batch_q6_k_f16in_small = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q6_k_f16in_small",
        )
        .context("Failed to compile dequant_batch_q6_k_f16in_small kernel")?;
        let fused_batch_pair_q4_k = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_pair_q4_k",
        )
        .context("Failed to compile dequant_batch_pair_q4_k kernel")?;
        let fused_batch_pair_q6_k = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_pair_q6_k",
        )
        .context("Failed to compile dequant_batch_pair_q6_k kernel")?;
        let fused_batch_pair_q4_k_f16in = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_pair_q4_k_f16in",
        )
        .context("Failed to compile dequant_batch_pair_q4_k_f16in kernel")?;
        let fused_batch_pair_q5_k_f16in = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_pair_q5_k_f16in",
        )
        .context("Failed to compile dequant_batch_pair_q5_k_f16in kernel")?;
        let fused_batch_pair_q6_k_f16in = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_pair_q6_k_f16in",
        )
        .context("Failed to compile dequant_batch_pair_q6_k_f16in kernel")?;
        let fused_batch_pair_q8_0_f16in = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_pair_q8_0_f16in",
        )
        .context("Failed to compile dequant_batch_pair_q8_0_f16in kernel")?;
        let fused_batch_pair_q8_0_f16in_full = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_pair_q8_0_f16in_full",
        )
        .context("Failed to compile dequant_batch_pair_q8_0_f16in_full kernel")?;
        let fused_batch_q4_k_small = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_small",
        )
        .context("Failed to compile dequant_batch_q4_k_small kernel")?;
        let fused_batch_q6_k_small = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q6_k_small",
        )
        .context("Failed to compile dequant_batch_q6_k_small kernel")?;
        let batch_matmul_btrans_f16_f32 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "batch_matmul_btrans_f16_f32",
        )
        .context("Failed to compile batch_matmul_btrans_f16_f32 kernel")?;
        let batch_matmul_btrans_f16_f32_full64 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "batch_matmul_btrans_f16_f32_full64",
        )
        .context("Failed to compile batch_matmul_btrans_f16_f32_full64 kernel")?;
        let fused_batch_q4_k_bn32 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_bn32",
        )
        .context("Failed to compile dequant_batch_q4_k_bn32 kernel")?;
        let fused_batch_q4_k_f16in_full32 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_f16in_full32",
        )
        .context("Failed to compile dequant_batch_q4_k_f16in_full32 kernel")?;
        let fused_batch_q6_k_f16in_full32 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q6_k_f16in_full32",
        )
        .context("Failed to compile dequant_batch_q6_k_f16in_full32 kernel")?;
        let fused_batch_q4_k_f16in_tail32 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_f16in_tail32",
        )
        .context("Failed to compile dequant_batch_q4_k_f16in_tail32 kernel")?;
        let fused_batch_q6_k_f16in_tail32 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q6_k_f16in_tail32",
        )
        .context("Failed to compile dequant_batch_q6_k_f16in_tail32 kernel")?;

        tracing::info!(
            "Dequant Metal kernels compiled (Q8_0 + Q4_K + Q5_K + Q6_K, standalone + fused matvec + fused matmul + batch + full32 + tail32)"
        );

        Ok(Self {
            dequant_q4_k,
            dequant_q6_k,
            fused_matvec_q5_k,
            fused_matvec_q5_k_ilp4,
            fused_matvec_q5_k_nr2,
            fused_matvec_pair_q5_k,
            fused_silu_down_matvec_q5_k,
            fused_gelu_down_matvec_q5_k,
            fused_matvec_q8_0,
            fused_matvec_q8_0_nr2,
            fused_matvec_q8_0_ilp4,
            fused_matvec_pair_q8_0,
            fused_silu_down_matvec_q8_0,
            fused_gelu_down_matvec_q8_0,
            fused_matvec_q5_0,
            fused_matvec_q5_1,
            fused_matvec_q4_k,
            fused_matvec_q4_k_nr2,
            fused_matvec_pair_q4_k,
            fused_silu_down_matvec_q4_k,
            fused_gelu_down_matvec_q4_k,
            fused_matvec_q4_k_ilp4,
            fused_matvec_dense_f16,
            fused_matvec_q6_k,
            fused_matvec_q6_k_nr2,
            fused_matvec_q6_k_ilp4,
            fused_matvec_pair_q6_k,
            fused_silu_down_matvec_q6_k,
            fused_gelu_down_matvec_q6_k,
            fused_matmul_q4_k,
            fused_matmul_q6_k,
            fused_batch_q4_k,
            fused_batch_q4_k_blocked,
            fused_batch_q4_k_blocked_fulltile,
            fused_batch_q4_k_blocked_bm32,
            fused_batch_q4_k_blocked_bm32_fulltile,
            fused_batch_q4_k_blocked_silu,
            fused_batch_q4_k_blocked_silu_fulltile,
            fused_batch_q5_k,
            fused_batch_q5_k_blocked,
            fused_batch_q5_k_blocked_fulltile,
            fused_batch_q5_k_f16in,
            fused_batch_q5_k_blocked_f16in,
            fused_batch_q5_k_blocked_f16in_fulltile,
            fused_batch_q5_k_small,
            fused_batch_q5_k_f16in_full64,
            fused_batch_q5_k_f16in_full32,
            fused_batch_q5_k_f16in_tail32,
            fused_batch_q5_k_f16in_small,
            fused_batch_q6_k,
            fused_batch_q6_k_blocked,
            fused_batch_q6_k_blocked_fulltile,
            fused_batch_q6_k_blocked_bm32,
            fused_batch_q6_k_blocked_bm32_fulltile,
            fused_batch_q6_k_blocked_silu,
            fused_batch_q6_k_blocked_silu_fulltile,
            fused_batch_q4_k_f16in,
            fused_batch_q4_k_f16in_full,
            fused_batch_q6_k_f16in,
            fused_batch_q8_0,
            fused_batch_q8_0_f16in,
            fused_batch_q8_0_blocked_f16in,
            fused_batch_q8_0_blocked_f16in_fulltile,
            fused_batch_q8_0_f16in_full,
            fused_batch_q8_0_f16in_full64,
            fused_batch_q8_0_f16in_full32,
            fused_batch_q8_0_f16in_tail32,
            fused_batch_q8_0_f16in_full32x32,
            fused_batch_q8_0_f16in_small,
            fused_batch_q5_0_blocked_f16in,
            fused_batch_q5_1_blocked_f16in,
            fused_batch_q4_k_f16in_full64,
            fused_batch_q4_k_f16in_full64_bk32,
            fused_batch_q6_k_f16in_full64,
            fused_batch_q4_k_f16in_small,
            fused_batch_q6_k_f16in_small,
            fused_batch_pair_q4_k,
            fused_batch_pair_q6_k,
            fused_batch_pair_q4_k_f16in,
            fused_batch_pair_q5_k_f16in,
            fused_batch_pair_q6_k_f16in,
            fused_batch_pair_q8_0_f16in,
            fused_batch_pair_q8_0_f16in_full,
            fused_batch_q4_k_small,
            fused_batch_q6_k_small,
            batch_matmul_btrans_f16_f32,
            batch_matmul_btrans_f16_f32_full64,
            fused_batch_q4_k_bn32,
            fused_batch_q4_k_f16in_full32,
            fused_batch_q6_k_f16in_full32,
            fused_batch_q4_k_f16in_tail32,
            fused_batch_q6_k_f16in_tail32,
            moe_map0: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_id_map0",
            )
            .context("Failed to compile moe_mul_mat_id_map0 kernel")?,
            moe_mul_mat_id_q4_k: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_id_q4_k",
            )
            .context("Failed to compile moe_mul_mat_id_q4_k kernel")?,
            moe_mul_mat_id_q4_k_blocked: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_id_q4_k_blocked",
            )
            .context("Failed to compile moe_mul_mat_id_q4_k_blocked kernel")?,
            moe_mul_mat_id_q5_k_blocked: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_id_q5_k_blocked",
            )
            .context("Failed to compile moe_mul_mat_id_q5_k_blocked kernel")?,
            moe_mul_mat_id_q6_k_blocked: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_id_q6_k_blocked",
            )
            .context("Failed to compile moe_mul_mat_id_q6_k_blocked kernel")?,
            moe_mul_mat_id_q6_k: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_id_q6_k",
            )
            .context("Failed to compile moe_mul_mat_id_q6_k kernel")?,
            moe_mul_mat_id_q8_0_blocked: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_id_q8_0_blocked",
            )
            .context("Failed to compile moe_mul_mat_id_q8_0_blocked kernel")?,
            moe_mul_mat_id_q8_0: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_id_q8_0",
            )
            .context("Failed to compile moe_mul_mat_id_q8_0 kernel")?,
            moe_mul_mat_selected_q4_k_blocked: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_q4_k_blocked",
            )
            .context("Failed to compile moe_mul_mat_selected_q4_k_blocked kernel")?,
            moe_mul_mat_selected_q4_k_matvec: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_q4_k_matvec",
            )
            .context("Failed to compile moe_mul_mat_selected_q4_k_matvec kernel")?,
            moe_mul_mat_selected_pair_q4_k_blocked: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_pair_q4_k_blocked",
            )
            .context("Failed to compile moe_mul_mat_selected_pair_q4_k_blocked kernel")?,
            moe_mul_mat_selected_pair_q4_k_matvec: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_pair_q4_k_matvec",
            )
            .context("Failed to compile moe_mul_mat_selected_pair_q4_k_matvec kernel")?,
            moe_mul_mat_selected_weighted_q4_k_blocked: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_weighted_q4_k_blocked",
            )
            .context("Failed to compile moe_mul_mat_selected_weighted_q4_k_blocked kernel")?,
            moe_mul_mat_selected_q5_k_blocked: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_q5_k_blocked",
            )
            .context("Failed to compile moe_mul_mat_selected_q5_k_blocked kernel")?,
            moe_mul_mat_selected_q5_k_matvec: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_q5_k_matvec",
            )
            .context("Failed to compile moe_mul_mat_selected_q5_k_matvec kernel")?,
            moe_mul_mat_selected_pair_q5_k_blocked: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_pair_q5_k_blocked",
            )
            .context("Failed to compile moe_mul_mat_selected_pair_q5_k_blocked kernel")?,
            moe_mul_mat_selected_pair_q5_k_matvec: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_pair_q5_k_matvec",
            )
            .context("Failed to compile moe_mul_mat_selected_pair_q5_k_matvec kernel")?,
            moe_mul_mat_selected_weighted_q5_k_blocked: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_weighted_q5_k_blocked",
            )
            .context("Failed to compile moe_mul_mat_selected_weighted_q5_k_blocked kernel")?,
            moe_mul_mat_selected_pair_q6_k_matvec: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_pair_q6_k_matvec",
            )
            .context("Failed to compile moe_mul_mat_selected_pair_q6_k_matvec kernel")?,
            moe_mul_mat_selected_pair_q6_k_matvec_nr2: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_pair_q6_k_matvec_nr2",
            )
            .context("Failed to compile moe_mul_mat_selected_pair_q6_k_matvec_nr2 kernel")?,
            moe_mul_mat_selected_q6_k_matvec: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_q6_k_matvec",
            )
            .context("Failed to compile moe_mul_mat_selected_q6_k_matvec kernel")?,
            moe_mul_mat_selected_q6_k_matvec_nr2: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_q6_k_matvec_nr2",
            )
            .context("Failed to compile moe_mul_mat_selected_q6_k_matvec_nr2 kernel")?,
            moe_mul_mat_selected_weighted_q6_k_matvec: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_weighted_q6_k_matvec",
            )
            .context("Failed to compile moe_mul_mat_selected_weighted_q6_k_matvec kernel")?,
            moe_mul_mat_selected_weighted_q6_k_matvec_nr2: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_weighted_q6_k_matvec_nr2",
            )
            .context("Failed to compile moe_mul_mat_selected_weighted_q6_k_matvec_nr2 kernel")?,
            moe_mul_mat_selected_pair_q8_0_matvec: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_pair_q8_0_matvec",
            )
            .context("Failed to compile moe_mul_mat_selected_pair_q8_0_matvec kernel")?,
            moe_mul_mat_selected_pair_q8_0_matvec_nr2: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_pair_q8_0_matvec_nr2",
            )
            .context("Failed to compile moe_mul_mat_selected_pair_q8_0_matvec_nr2 kernel")?,
            moe_mul_mat_selected_q8_0_matvec: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_q8_0_matvec",
            )
            .context("Failed to compile moe_mul_mat_selected_q8_0_matvec kernel")?,
            moe_mul_mat_selected_q8_0_matvec_nr2: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_q8_0_matvec_nr2",
            )
            .context("Failed to compile moe_mul_mat_selected_q8_0_matvec_nr2 kernel")?,
            moe_mul_mat_selected_weighted_q8_0_matvec: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_weighted_q8_0_matvec",
            )
            .context("Failed to compile moe_mul_mat_selected_weighted_q8_0_matvec kernel")?,
            moe_mul_mat_selected_weighted_q8_0_matvec_nr2: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_mul_mat_selected_weighted_q8_0_matvec_nr2",
            )
            .context("Failed to compile moe_mul_mat_selected_weighted_q8_0_matvec_nr2 kernel")?,
            moe_fused_silu_down_selected_weighted_q5_k_matvec: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_fused_silu_down_selected_weighted_q5_k_matvec",
            )
            .context(
                "Failed to compile moe_fused_silu_down_selected_weighted_q5_k_matvec kernel",
            )?,
            moe_fused_silu_down_selected_weighted_q5_k_matvec_slots8: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_fused_silu_down_selected_weighted_q5_k_matvec_slots8",
            )
            .context(
                "Failed to compile moe_fused_silu_down_selected_weighted_q5_k_matvec_slots8 kernel",
            )?,
            moe_fused_silu_down_selected_weighted_q5_k_matvec_nr2: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_fused_silu_down_selected_weighted_q5_k_matvec_nr2",
            )
            .context(
                "Failed to compile moe_fused_silu_down_selected_weighted_q5_k_matvec_nr2 kernel",
            )?,
            moe_fused_silu_down_selected_weighted_q5_k_matvec_slots8_nr2: ComputePipeline::from_source(
                device.device(),
                DEQUANT_SHADER_SRC,
                "moe_fused_silu_down_selected_weighted_q5_k_matvec_slots8_nr2",
            )
            .context(
                "Failed to compile moe_fused_silu_down_selected_weighted_q5_k_matvec_slots8_nr2 kernel",
            )?,
        })
    }

    #[cfg(test)]
    pub(crate) fn test_fused_batch_q4_k_small(&self) -> &ComputePipeline {
        &self.fused_batch_q4_k_small
    }

    #[cfg(test)]
    pub(crate) fn test_fused_batch_q6_k_small(&self) -> &ComputePipeline {
        &self.fused_batch_q6_k_small
    }

    #[cfg(test)]
    pub(crate) fn test_fused_batch_q8_0_blocked_f16in(&self) -> &ComputePipeline {
        &self.fused_batch_q8_0_blocked_f16in
    }

    #[cfg(test)]
    pub(crate) fn test_fused_batch_q8_0_blocked_f16in_fulltile(&self) -> &ComputePipeline {
        &self.fused_batch_q8_0_blocked_f16in_fulltile
    }

    /// Encode dense half×half batch matmul in B-transposed layout.
    ///
    /// Computes C[N×M] = B[N×K] × A[M×K]^T with f32 accumulation/output.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_batch_matmul_btrans_f16_f32(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a_mk_f16: &MetalBuffer,
        b_nk_f16: &MetalBuffer,
        c_nm_f32: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        if m.is_multiple_of(DB64_TILE_M as u32) {
            let n_full = (n as usize / DB64_TILE_N) * DB64_TILE_N;
            let n_tail = (n as usize).saturating_sub(n_full);
            if n_full > 0 {
                let groups_x = (m as usize).div_ceil(DB64_TILE_M);
                crate::set_pipeline_cached(
                    encoder,
                    self.batch_matmul_btrans_f16_f32_full64.state(),
                );
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(a_mk_f16.mtl_buffer()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(b_nk_f16.mtl_buffer()), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(c_nm_f32.mtl_buffer()), 0, 2);
                }
                bind_u32(encoder, 3, m);
                bind_u32(encoder, 4, n_full as u32);
                bind_u32(encoder, 5, k);
                bind_u32(encoder, 6, m);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: groups_x,
                        height: n_full / DB64_TILE_N,
                        depth: 1,
                    },
                    MTLSize {
                        width: DB64_TG,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            if n_tail == 0 {
                return;
            }
            let b_off = n_full
                .checked_mul(k as usize)
                .and_then(|x| x.checked_mul(2))
                .expect("B offset overflow");
            let c_off = n_full
                .checked_mul(m as usize)
                .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                .expect("C offset overflow");
            let groups_x = (m as usize).div_ceil(DB_TILE_M);
            let groups_y = n_tail.div_ceil(DB_TILE_N);
            crate::set_pipeline_cached(encoder, self.batch_matmul_btrans_f16_f32.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(a_mk_f16.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(b_nk_f16.mtl_buffer()), b_off, 1);
                encoder.setBuffer_offset_atIndex(Some(c_nm_f32.mtl_buffer()), c_off, 2);
            }
            bind_u32(encoder, 3, m);
            bind_u32(encoder, 4, n_tail as u32);
            bind_u32(encoder, 5, k);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups_x,
                    height: groups_y,
                    depth: 1,
                },
                MTLSize {
                    width: DB_TG,
                    height: 1,
                    depth: 1,
                },
            );
            return;
        }

        let groups_x = (m as usize).div_ceil(DB_TILE_M);
        let groups_y = (n as usize).div_ceil(DB_TILE_N);
        crate::set_pipeline_cached(encoder, self.batch_matmul_btrans_f16_f32.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a_mk_f16.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b_nk_f16.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(c_nm_f32.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: groups_y,
                depth: 1,
            },
            MTLSize {
                width: DB_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Dequantize Q4_K data on GPU: quantized blocks → f32 output.
    ///
    /// - `src`: buffer containing `n_blocks` Q4_K blocks (144 bytes each)
    /// - `dst`: output buffer for `n_blocks * 256` f32 values
    /// - `n_blocks`: number of Q4_K blocks
    pub fn dequant_q4_k(
        &self,
        device: &MetalDevice,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        n_blocks: u32,
    ) -> anyhow::Result<()> {
        let dims = DispatchDims::d1(n_blocks as usize, DEQUANT_TG_SIZE);

        device.execute_sync(|encoder| {
            crate::set_pipeline_cached(encoder, self.dequant_q4_k.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
            }
            bind_u32(encoder, 2, n_blocks);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
            Ok(())
        })
    }

    /// Fused dequant Q5_K + matvec: y = dequant(A) × x.
    ///
    /// Baseline imported from llama.cpp's decode-only launch shape: TG=64 with
    /// 2 simdgroups, each computing one output row.
    pub fn fused_matvec_q5_k(
        &self,
        device: &MetalDevice,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        self.fused_matvec_q5_k_with_config(device, a, x, y, m, k, DequantDispatchConfig::default())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fused_matvec_q5_k_with_config(
        &self,
        device: &MetalDevice,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
        config: DequantDispatchConfig,
    ) -> anyhow::Result<()> {
        let (groups, tg_width, pipeline) = self.q5_k_matvec_dispatch_with_config(m, config);
        let dims = DispatchDims::d1(groups, 1);

        device.execute_sync(|encoder| {
            crate::set_pipeline_cached(encoder, pipeline);
            bind_buffers(encoder, a, x, y);
            bind_u32(encoder, 3, m);
            bind_u32(encoder, 4, k);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                MTLSize {
                    width: tg_width,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        })
    }

    /// Explicit NR2 Q5_K matvec route for A/B benchmarking and validation.
    pub fn fused_matvec_q5_k_nr2(
        &self,
        device: &MetalDevice,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        let dims = DispatchDims::d1((m as usize).div_ceil(Q5K_NR2_ROWS), 1);

        device.execute_sync(|encoder| {
            crate::set_pipeline_cached(encoder, self.fused_matvec_q5_k_nr2.state());
            bind_buffers(encoder, a, x, y);
            bind_u32(encoder, 3, m);
            bind_u32(encoder, 4, k);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                MTLSize {
                    width: DEQUANT_MATVEC_Q5K_TG,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        })
    }

    /// Fused dequant Q4_K + matvec: y = dequant(A) × x.
    ///
    /// - `a`: M × (K/256) Q4_K blocks (quantized weight matrix)
    /// - `x`: K f32 values (input vector)
    /// - `y`: M f32 values (output vector)
    /// - `m`: number of rows
    /// - `k`: number of columns (must be multiple of 256)
    #[allow(clippy::too_many_arguments)]
    pub fn fused_matvec_q4_k_with_config(
        &self,
        device: &MetalDevice,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
        config: DequantDispatchConfig,
    ) -> anyhow::Result<()> {
        let (groups, tg_width, pipeline) = self.q4_k_matvec_dispatch_with_config(m, config);
        let dims = DispatchDims::d1(groups, 1);

        device.execute_sync(|encoder| {
            crate::set_pipeline_cached(encoder, pipeline);
            bind_buffers(encoder, a, x, y);
            bind_u32(encoder, 3, m);
            bind_u32(encoder, 4, k);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                MTLSize {
                    width: tg_width,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        })
    }

    /// Explicit NR2 Q4_K matvec route for A/B benchmarking and validation.
    pub fn fused_matvec_q4_k_nr2(
        &self,
        device: &MetalDevice,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        let dims = DispatchDims::d1((m as usize).div_ceil(Q4K_NR2_ROWS), 1);

        device.execute_sync(|encoder| {
            crate::set_pipeline_cached(encoder, self.fused_matvec_q4_k_nr2.state());
            bind_buffers(encoder, a, x, y);
            bind_u32(encoder, 3, m);
            bind_u32(encoder, 4, k);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                MTLSize {
                    width: DEQUANT_MATVEC_Q4K_NR2_TG,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        })
    }

    /// Encode a fused Q5_K matvec dispatch into an existing encoder.
    pub fn encode_fused_matvec_q5_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        self.encode_fused_matvec_q5_k_with_config(
            encoder,
            a,
            x,
            y,
            m,
            k,
            DequantDispatchConfig::default(),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_matvec_q5_k_with_config(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
        config: DequantDispatchConfig,
    ) {
        let (groups, tg_width, pipeline) = self.q5_k_matvec_dispatch_with_config(m, config);
        let dims = DispatchDims::d1(groups, 1);
        crate::set_pipeline_cached(encoder, pipeline);
        bind_buffers(encoder, a, x, y);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a dual-output Q5_K matvec dispatch into an existing encoder.
    ///
    /// Computes `y0 = dequant(A0) × x` and `y1 = dequant(A1) × x` in one dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_matvec_pair_q5_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a0: &MetalBuffer,
        a1: &MetalBuffer,
        x: &MetalBuffer,
        y0: &MetalBuffer,
        y1: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        let dims = DispatchDims::d1((m as usize).div_ceil(2), 1);
        crate::set_pipeline_cached(encoder, self.fused_matvec_pair_q5_k.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a0.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(a1.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(y0.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(y1.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, m);
        bind_u32(encoder, 6, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            MTLSize {
                width: DEQUANT_MATVEC_Q5K_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a fused SiLU(gate) * up + Q5_K down matvec dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_silu_down_matvec_q5_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        // Q5_K fused shader: 2 rows/TG (tg_id * 2 + simd_id), TG=64 (2 SGs).
        let dims = DispatchDims::d1((m as usize).div_ceil(2), 1);
        crate::set_pipeline_cached(encoder, self.fused_silu_down_matvec_q5_k.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(y.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, m);
        bind_u32(encoder, 5, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            MTLSize {
                width: DEQUANT_MATVEC_Q5K_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a fused GELU(gate) * up + Q5_K down matvec dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_gelu_down_matvec_q5_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        // Q5_K fused shader: 2 rows/TG (tg_id * 2 + simd_id), TG=64 (2 SGs).
        let dims = DispatchDims::d1((m as usize).div_ceil(2), 1);
        crate::set_pipeline_cached(encoder, self.fused_gelu_down_matvec_q5_k.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(y.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, m);
        bind_u32(encoder, 5, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            MTLSize {
                width: DEQUANT_MATVEC_Q5K_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a fused Q8_0 matvec dispatch into an existing encoder.
    ///
    /// Does NOT create or commit a command buffer. Used for batching
    /// multiple matvec operations into a single command buffer.
    pub fn encode_fused_matvec_q8_0(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        let dims = DispatchDims::d1(m as usize, 1);
        crate::set_pipeline_cached(encoder, self.fused_matvec_q8_0.state());
        bind_buffers(encoder, a, x, y);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            MTLSize {
                width: DEQUANT_MATVEC_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a fused Q5_0 matvec dispatch into an existing encoder.
    pub fn encode_fused_matvec_q5_0(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        let dims = DispatchDims::d1(m as usize, 1);
        crate::set_pipeline_cached(encoder, self.fused_matvec_q5_0.state());
        bind_buffers(encoder, a, x, y);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            MTLSize {
                width: DEQUANT_MATVEC_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a fused Q5_1 matvec dispatch into an existing encoder.
    pub fn encode_fused_matvec_q5_1(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        let dims = DispatchDims::d1(m as usize, 1);
        crate::set_pipeline_cached(encoder, self.fused_matvec_q5_1.state());
        bind_buffers(encoder, a, x, y);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            MTLSize {
                width: DEQUANT_MATVEC_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a Q8_0 matvec with config-driven candidate selection.
    ///
    /// Selects between NR2 (preferred for x-reuse), ILP4, or base depending on config.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_matvec_q8_0_with_config(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
        config: DequantDispatchConfig,
    ) {
        let selection = q8_0_matvec_candidate_selection(m, config);
        let pipeline = match selection.candidate {
            MatvecCandidate::Q8_0Nr2 => self.fused_matvec_q8_0_nr2.state(),
            MatvecCandidate::Q8_0Ilp4 => self.fused_matvec_q8_0_ilp4.state(),
            _ => self.fused_matvec_q8_0.state(),
        };
        let groups = selection.threadgroups;
        let tg_width = selection.threadgroup_width;
        let dims = DispatchDims::d1(groups, 1);
        crate::set_pipeline_cached(encoder, pipeline);
        bind_buffers(encoder, a, x, y);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a NR2 Q8_0 matvec dispatch (TG=64, 4 rows per TG) into an existing encoder.
    ///
    /// Does NOT create or commit a command buffer. Standalone method for A/B benchmarking.
    pub fn encode_fused_matvec_q8_0_nr2(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        let dims = DispatchDims::d1((m as usize).div_ceil(Q8_0_NR2_ROWS), 1);
        crate::set_pipeline_cached(encoder, self.fused_matvec_q8_0_nr2.state());
        bind_buffers(encoder, a, x, y);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            MTLSize {
                width: DEQUANT_MATVEC_Q8_0_NR2_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode an ILP4 Q8_0 matvec dispatch (TG=64, 2 rows per TG) into an existing encoder.
    ///
    /// Does NOT create or commit a command buffer. Standalone method for A/B benchmarking.
    pub fn encode_fused_matvec_q8_0_ilp4(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        let dims = DispatchDims::d1((m as usize).div_ceil(Q8_0_ILP4_ROWS), 1);
        crate::set_pipeline_cached(encoder, self.fused_matvec_q8_0_ilp4.state());
        bind_buffers(encoder, a, x, y);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            MTLSize {
                width: DEQUANT_MATVEC_Q8_0_ILP4_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a dual-output Q8_0 matvec dispatch into an existing encoder.
    ///
    /// Computes `y0 = dequant(A0) × x` and `y1 = dequant(A1) × x` in one dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_matvec_pair_q8_0(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a0: &MetalBuffer,
        a1: &MetalBuffer,
        x: &MetalBuffer,
        y0: &MetalBuffer,
        y1: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        let dims = DispatchDims::d1(m as usize, 1);
        crate::set_pipeline_cached(encoder, self.fused_matvec_pair_q8_0.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a0.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(a1.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(y0.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(y1.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, m);
        bind_u32(encoder, 6, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            MTLSize {
                width: DEQUANT_MATVEC_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a fused SiLU(gate) * up + Q8_0 down matvec dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_silu_down_matvec_q8_0(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        // Q8_0 fused shader: 1 row per threadgroup, TG=128 (4 simdgroups).
        crate::set_pipeline_cached(encoder, self.fused_silu_down_matvec_q8_0.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(y.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, m);
        bind_u32(encoder, 5, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: m as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: DEQUANT_MATVEC_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a fused GELU(gate) * up + Q8_0 down matvec dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_gelu_down_matvec_q8_0(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        // Q8_0 fused shader: 1 row per threadgroup, TG=128 (4 simdgroups).
        crate::set_pipeline_cached(encoder, self.fused_gelu_down_matvec_q8_0.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(y.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, m);
        bind_u32(encoder, 5, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: m as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: DEQUANT_MATVEC_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a fused Q4_K matvec dispatch into an existing encoder.
    ///
    /// Does NOT create or commit a command buffer. Used for batching
    /// multiple matvec operations into a single command buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_matvec_q4_k_with_config(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
        config: DequantDispatchConfig,
    ) {
        let (groups, tg_width, pipeline) = self.q4_k_matvec_dispatch_with_config(m, config);
        let dims = DispatchDims::d1(groups, 1);
        crate::set_pipeline_cached(encoder, pipeline);
        bind_buffers(encoder, a, x, y);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode an ILP4 Q4_K matvec dispatch (TG=64, 2 rows per TG) into an existing encoder.
    ///
    /// Does NOT create or commit a command buffer. Standalone method for A/B benchmarking.
    pub fn encode_fused_matvec_q4_k_ilp4(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        let dims = DispatchDims::d1((m as usize).div_ceil(Q4K_ILP4_ROWS), 1);
        crate::set_pipeline_cached(encoder, self.fused_matvec_q4_k_ilp4.state());
        bind_buffers(encoder, a, x, y);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            MTLSize {
                width: DEQUANT_MATVEC_Q4K_ILP4_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a dual-output Q4_K matvec dispatch into an existing encoder.
    ///
    /// Computes `y0 = dequant(A0) × x` and `y1 = dequant(A1) × x` in one dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_matvec_pair_q4_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a0: &MetalBuffer,
        a1: &MetalBuffer,
        x: &MetalBuffer,
        y0: &MetalBuffer,
        y1: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        let dims = DispatchDims::d1((m as usize).div_ceil(Q4K_NR2_ROWS), 1);
        crate::set_pipeline_cached(encoder, self.fused_matvec_pair_q4_k.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a0.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(a1.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(y0.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(y1.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, m);
        bind_u32(encoder, 6, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            MTLSize {
                width: DEQUANT_MATVEC_Q4K_NR2_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a fused SiLU(gate) * up + Q4_K down matvec dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_silu_down_matvec_q4_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        // G14: nr2 geometry — TG=64, 4 rows/TG (2/SG).
        crate::set_pipeline_cached(encoder, self.fused_silu_down_matvec_q4_k.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(y.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, m);
        bind_u32(encoder, 5, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (m as usize).div_ceil(Q4K_NR2_ROWS),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: DEQUANT_MATVEC_Q4K_NR2_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a fused GELU(gate) * up + Q4_K down matvec dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_gelu_down_matvec_q4_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        crate::set_pipeline_cached(encoder, self.fused_gelu_down_matvec_q4_k.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(y.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, m);
        bind_u32(encoder, 5, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (m as usize).div_ceil(Q4K_NR2_ROWS),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: DEQUANT_MATVEC_Q4K_NR2_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a dense f16 matvec dispatch into an existing encoder.
    ///
    /// y = A[M×K] × x[K], where A and x are half, y is float.
    pub fn encode_fused_matvec_dense_f16(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        let dims = DispatchDims::d1(m as usize, 1);
        crate::set_pipeline_cached(encoder, self.fused_matvec_dense_f16.state());
        bind_buffers(encoder, a, x, y);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            MTLSize {
                width: DEQUANT_MATVEC_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Fused dequant Q8_0 + matvec: y = dequant(A) × x.
    ///
    /// - `a`: M × (K/32) Q8_0 blocks (quantized weight matrix)
    /// - `x`: K f32 values (input vector)
    /// - `y`: M f32 values (output vector)
    /// - `m`: number of rows
    /// - `k`: number of columns (must be multiple of 32)
    pub fn fused_matvec_q8_0(
        &self,
        device: &MetalDevice,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        let dims = DispatchDims::d1(m as usize, 1);

        device.execute_sync(|encoder| {
            crate::set_pipeline_cached(encoder, self.fused_matvec_q8_0.state());
            bind_buffers(encoder, a, x, y);
            bind_u32(encoder, 3, m);
            bind_u32(encoder, 4, k);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                MTLSize {
                    width: DEQUANT_MATVEC_TG,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        })
    }

    /// Dequantize Q6_K data on GPU: quantized blocks → f32 output.
    ///
    /// - `src`: buffer containing `n_blocks` Q6_K blocks (210 bytes each)
    /// - `dst`: output buffer for `n_blocks * 256` f32 values
    /// - `n_blocks`: number of Q6_K blocks
    pub fn dequant_q6_k(
        &self,
        device: &MetalDevice,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        n_blocks: u32,
    ) -> anyhow::Result<()> {
        let dims = DispatchDims::d1(n_blocks as usize, DEQUANT_TG_SIZE);

        device.execute_sync(|encoder| {
            crate::set_pipeline_cached(encoder, self.dequant_q6_k.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
            }
            bind_u32(encoder, 2, n_blocks);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
            Ok(())
        })
    }

    /// Fused dequant Q6_K + matvec: y = dequant(A) × x.
    ///
    /// - `a`: M × (K/256) Q6_K blocks (quantized weight matrix)
    /// - `x`: K f32 values (input vector)
    /// - `y`: M f32 values (output vector)
    /// - `m`: number of rows
    /// - `k`: number of columns (must be multiple of 256)
    #[allow(clippy::too_many_arguments)]
    pub fn fused_matvec_q6_k_with_config(
        &self,
        device: &MetalDevice,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
        config: DequantDispatchConfig,
    ) -> anyhow::Result<()> {
        let (groups, tg_width, pipeline) = self.q6_k_matvec_dispatch_with_config(m, config);
        let dims = DispatchDims::d1(groups, 1);

        device.execute_sync(|encoder| {
            crate::set_pipeline_cached(encoder, pipeline);
            bind_buffers(encoder, a, x, y);
            bind_u32(encoder, 3, m);
            bind_u32(encoder, 4, k);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                MTLSize {
                    width: tg_width,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        })
    }

    /// Explicit NR2 Q6_K matvec route for A/B benchmarking and validation.
    pub fn fused_matvec_q6_k_nr2(
        &self,
        device: &MetalDevice,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        let dims = DispatchDims::d1((m as usize).div_ceil(Q6K_NR2_ROWS), 1);

        device.execute_sync(|encoder| {
            crate::set_pipeline_cached(encoder, self.fused_matvec_q6_k_nr2.state());
            bind_buffers(encoder, a, x, y);
            bind_u32(encoder, 3, m);
            bind_u32(encoder, 4, k);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                MTLSize {
                    width: DEQUANT_MATVEC_Q6K_NR2_TG,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        })
    }

    /// Encode an ILP4 Q6_K matvec dispatch (TG=64, 2 rows per TG) into an existing encoder.
    ///
    /// Does NOT create or commit a command buffer. Standalone method for A/B benchmarking.
    pub fn encode_fused_matvec_q6_k_ilp4(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        let dims = DispatchDims::d1((m as usize).div_ceil(Q6K_ILP4_ROWS), 1);
        crate::set_pipeline_cached(encoder, self.fused_matvec_q6_k_ilp4.state());
        bind_buffers(encoder, a, x, y);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            MTLSize {
                width: DEQUANT_MATVEC_Q6K_ILP4_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a fused Q6_K matvec dispatch into an existing encoder.
    ///
    /// Does NOT create or commit a command buffer. Used for batching
    /// multiple matvec operations into a single command buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_matvec_q6_k_with_config(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
        config: DequantDispatchConfig,
    ) {
        let (groups, tg_width, pipeline) = self.q6_k_matvec_dispatch_with_config(m, config);
        let dims = DispatchDims::d1(groups, 1);
        crate::set_pipeline_cached(encoder, pipeline);
        bind_buffers(encoder, a, x, y);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a dual-output Q6_K matvec dispatch into an existing encoder.
    ///
    /// Computes `y0 = dequant(A0) × x` and `y1 = dequant(A1) × x` in one dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_matvec_pair_q6_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a0: &MetalBuffer,
        a1: &MetalBuffer,
        x: &MetalBuffer,
        y0: &MetalBuffer,
        y1: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        let dims = DispatchDims::d1(m as usize, 1);
        crate::set_pipeline_cached(encoder, self.fused_matvec_pair_q6_k.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a0.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(a1.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(y0.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(y1.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, m);
        bind_u32(encoder, 6, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            MTLSize {
                width: DEQUANT_MATVEC_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a fused SiLU(gate) * up + Q6_K down matvec dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_silu_down_matvec_q6_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        // Q6_K fused SiLU shader: 1 row per threadgroup, TG=128 (4 simdgroups).
        crate::set_pipeline_cached(encoder, self.fused_silu_down_matvec_q6_k.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(y.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, m);
        bind_u32(encoder, 5, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: m as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: DEQUANT_MATVEC_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a fused GELU(gate) * up + Q6_K down matvec dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_gelu_down_matvec_q6_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        crate::set_pipeline_cached(encoder, self.fused_gelu_down_matvec_q6_k.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(y.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, m);
        bind_u32(encoder, 5, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (m as usize).div_ceil(4),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: DEQUANT_MATVEC_Q6K_NR2_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a fused dequant Q4_K + batched matmul using simdgroup_matrix.
    ///
    /// C = dequant(A_q4k) × B, where:
    /// - `a`: M × (K/256) Q4_K blocks (quantized weight matrix)
    /// - `b`: K × N f32 values (input activation matrix)
    /// - `c`: M × N f32 values (output matrix)
    /// - `m`: number of output rows
    /// - `n`: number of output columns (batch size / n_tokens)
    /// - `k`: number of inner dimension (must be multiple of 256)
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_matmul_q4_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let groups_x = (n as usize).div_ceil(DQ_TILE);
        let groups_y = (m as usize).div_ceil(DQ_TILE);

        crate::set_pipeline_cached(encoder, self.fused_matmul_q4_k.state());
        bind_buffers(encoder, a, b, c);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: groups_y,
                depth: 1,
            },
            MTLSize {
                width: DQ_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a fused dequant Q6_K + batched matmul using simdgroup_matrix.
    ///
    /// C = dequant(A_q6k) × B, where:
    /// - `a`: M × (K/256) Q6_K blocks (quantized weight matrix)
    /// - `b`: K × N f32 values (input activation matrix)
    /// - `c`: M × N f32 values (output matrix)
    /// - `m`: number of output rows
    /// - `n`: number of output columns (batch size / n_tokens)
    /// - `k`: number of inner dimension (must be multiple of 256)
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_matmul_q6_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let groups_x = (n as usize).div_ceil(DQ_TILE);
        let groups_y = (m as usize).div_ceil(DQ_TILE);

        crate::set_pipeline_cached(encoder, self.fused_matmul_q6_k.state());
        bind_buffers(encoder, a, b, c);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: groups_y,
                depth: 1,
            },
            MTLSize {
                width: DQ_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a B-transposed batch dequant Q4_K + matmul.
    ///
    /// C[N × M] = B[N × K] × dequant(A[M × K])^T
    ///
    /// - `a`: M × (K/256) Q4_K blocks (quantized weight matrix)
    /// - `b`: N × K f32 values (batch input, rows = tokens)
    /// - `c`: N × M f32 values (batch output, rows = tokens)
    /// - `m`: output features (weight rows)
    /// - `n`: number of tokens (batch size)
    /// - `k`: input features (must be multiple of 256)
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_q4_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        // Blocked-layout kernel (llama.cpp pattern): stride-8 TG memory, 6KB,
        // TG=128, 1.33 MACs/load. Preferred for all K%256==0 workloads.
        // Uses dynamic threadgroup memory via [[threadgroup(0)]].
        if batch_q4k_blocked_enabled() && k.is_multiple_of(Q4_K_BLOCK_VALUES as u32) {
            const BLOCKED_TG: usize = 128;
            // Choose BM=32 for small M (doubles TG count for better GPU saturation).
            let use_bm32 = false; // BM=32 tested: no improvement on small M (GPU saturation not the bottleneck)
            let full_tile = if use_bm32 {
                (m as usize).is_multiple_of(32) && (n as usize).is_multiple_of(32)
            } else {
                (m as usize).is_multiple_of(64) && (n as usize).is_multiple_of(32)
            };
            // G24: full-tile path needs only sa+sb (6 KB for BM=64, 4 KB for
            // BM=32). Boundary path reuses sa as float output staging (8 KB).
            let (pipeline, bm, smem) = if use_bm32 {
                (
                    if full_tile {
                        &self.fused_batch_q4_k_blocked_bm32_fulltile
                    } else {
                        &self.fused_batch_q4_k_blocked_bm32
                    },
                    32usize,
                    if full_tile { 4096usize } else { 8192usize },
                )
            } else {
                (
                    if full_tile {
                        &self.fused_batch_q4_k_blocked_fulltile
                    } else {
                        &self.fused_batch_q4_k_blocked
                    },
                    64usize,
                    if full_tile { 6144usize } else { 8192usize },
                )
            };
            crate::set_pipeline_cached(encoder, pipeline.state());
            bind_buffers(encoder, a, b, c);
            bind_u32(encoder, 3, m);
            bind_u32(encoder, 4, n);
            bind_u32(encoder, 5, k);
            unsafe {
                encoder.setThreadgroupMemoryLength_atIndex(smem, 0);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (n as usize).div_ceil(32),
                    height: (m as usize).div_ceil(bm),
                    depth: 1,
                },
                MTLSize {
                    width: BLOCKED_TG,
                    height: 1,
                    depth: 1,
                },
            );
            return;
        }

        // For small N (< 64 rows), use the BN=32 kernel: half tiles, 12 KB TG memory → 2 TGs/SM.
        // At N=39: BN=64 has 100% boundary tiles; BN=32 has 50% fast-path + 50% boundary, 2 TGs/SM.
        if n < DB_TILE_N as u32 {
            let groups_x = (m as usize).div_ceil(DB_TILE_M);
            let groups_y = (n as usize).div_ceil(SB_TILE_N);
            crate::set_pipeline_cached(encoder, self.fused_batch_q4_k_bn32.state());
            bind_buffers(encoder, a, b, c);
            bind_u32(encoder, 3, m);
            bind_u32(encoder, 4, n);
            bind_u32(encoder, 5, k);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups_x,
                    height: groups_y,
                    depth: 1,
                },
                MTLSize {
                    width: SB_TG,
                    height: 1,
                    depth: 1,
                },
            );
            return;
        }
        let use_small = n < batch_q4k_small_n_threshold();

        let groups_x = (m as usize).div_ceil(DB_TILE_M);
        let groups_y = if use_small {
            (n as usize).div_ceil(SB_TILE_N)
        } else {
            (n as usize).div_ceil(DB_TILE_N)
        };

        crate::set_pipeline_cached(
            encoder,
            if use_small {
                self.fused_batch_q4_k_small.state()
            } else {
                self.fused_batch_q4_k.state()
            },
        );
        bind_buffers(encoder, a, b, c);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: groups_y,
                depth: 1,
            },
            MTLSize {
                width: if use_small { SB_TG } else { DB_TG },
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode the default B-transposed batch dequant Q5_K + matmul route.
    ///
    /// C[N × M] = B[N × K] × dequant(A[M × K])^T
    ///
    /// Uses the blocked `kernel_mul_mm`-style path by default when K is
    /// 256-aligned, matching the current Q4_K/Q6_K prefill architecture.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_q5_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        if batch_q5k_blocked_enabled() && k.is_multiple_of(Q5_K_BLOCK_VALUES as u32) {
            const BLOCKED_TG: usize = 128;
            let full_tile = (m as usize).is_multiple_of(64) && (n as usize).is_multiple_of(32);
            crate::set_pipeline_cached(
                encoder,
                if full_tile {
                    self.fused_batch_q5_k_blocked_fulltile.state()
                } else {
                    self.fused_batch_q5_k_blocked.state()
                },
            );
            bind_buffers(encoder, a, b, c);
            bind_u32(encoder, 3, m);
            bind_u32(encoder, 4, n);
            bind_u32(encoder, 5, k);
            unsafe {
                // G24: full-tile needs only sa+sb = 6 KB; boundary needs 8 KB.
                encoder.setThreadgroupMemoryLength_atIndex(if full_tile { 6144 } else { 8192 }, 0);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (n as usize).div_ceil(32),
                    height: (m as usize).div_ceil(64),
                    depth: 1,
                },
                MTLSize {
                    width: BLOCKED_TG,
                    height: 1,
                    depth: 1,
                },
            );
            return;
        }

        let groups_x = (m as usize).div_ceil(DB_TILE_M);
        let groups_y = (n as usize).div_ceil(DB_TILE_N);

        crate::set_pipeline_cached(encoder, self.fused_batch_q5_k.state());
        bind_buffers(encoder, a, b, c);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: groups_y,
                depth: 1,
            },
            MTLSize {
                width: DB_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode the default B-transposed batch dequant Q5_K + matmul with f16 input.
    ///
    /// C[N × M] = B[N × K] × dequant(A[M × K])^T
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_q5_k_f16in(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        if batch_q5k_blocked_enabled() && k.is_multiple_of(Q5_K_BLOCK_VALUES as u32) {
            const BLOCKED_TG: usize = 128;
            let full_tile = (m as usize).is_multiple_of(64) && (n as usize).is_multiple_of(32);
            crate::set_pipeline_cached(
                encoder,
                if full_tile {
                    self.fused_batch_q5_k_blocked_f16in_fulltile.state()
                } else {
                    self.fused_batch_q5_k_blocked_f16in.state()
                },
            );
            bind_buffers(encoder, a, b, c);
            bind_u32(encoder, 3, m);
            bind_u32(encoder, 4, n);
            bind_u32(encoder, 5, k);
            bind_u32(encoder, 6, m);
            unsafe {
                // G24: full-tile needs only sa+sb = 6 KB; boundary needs 8 KB.
                encoder.setThreadgroupMemoryLength_atIndex(if full_tile { 6144 } else { 8192 }, 0);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (n as usize).div_ceil(32),
                    height: (m as usize).div_ceil(64),
                    depth: 1,
                },
                MTLSize {
                    width: BLOCKED_TG,
                    height: 1,
                    depth: 1,
                },
            );
            return;
        }

        let groups_x = (m as usize).div_ceil(DB_TILE_M);
        let groups_y = (n as usize).div_ceil(DB_TILE_N);

        crate::set_pipeline_cached(encoder, self.fused_batch_q5_k_f16in.state());
        bind_buffers(encoder, a, b, c);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        bind_u32(encoder, 6, m);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: groups_y,
                depth: 1,
            },
            MTLSize {
                width: DB_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode the small-N B-transposed batch dequant Q5_K + matmul route.
    ///
    /// Runtime planning routes to this automatically for the current Q5_K
    /// small-batch window.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_q5_k_small(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let groups_x = (m as usize).div_ceil(DB_TILE_M);
        let groups_y = (n as usize).div_ceil(SB_TILE_N);

        crate::set_pipeline_cached(encoder, self.fused_batch_q5_k_small.state());
        bind_buffers(encoder, a, b, c);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: groups_y,
                depth: 1,
            },
            MTLSize {
                width: SB_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a B-transposed batch dequant Q5_K + matmul with f16 input and f32 output,
    /// selecting the best kernel variant based on M/N alignment and config.
    ///
    /// Selection priority: full64 > full32 > tail32 > small > base f16in.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_q5_k_f16in_with_config(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
        config: DequantDispatchConfig,
    ) {
        let small_n_threshold = config.batch_f16in_small_n_threshold;
        let small_m_max = config.batch_f16in_small_m_max;
        let use_small =
            small_n_threshold > 1 && n < small_n_threshold && small_m_max > 0 && m <= small_m_max;

        if !use_small {
            let m_full = (m as usize / DB64_TILE_M) * DB64_TILE_M;
            let m_tail = (m as usize).saturating_sub(m_full);
            // BN=32 occupancy gate (same logic as Q4_K/Q6_K).
            let bn32_preferred = config.batch_f16in_use_bn32;
            let tgs_bn32 = m_full.div_ceil(DB64_TILE_M) * (n as usize / DB32_TILE_N).max(1);
            let use_bn32 = bn32_preferred && (tgs_bn32 * DB32_TG) >= 32768;
            let n_tile = if use_bn32 { DB32_TILE_N } else { DB64_TILE_N };
            let n_full = (n as usize / n_tile) * n_tile;
            let n_tail = (n as usize).saturating_sub(n_full);
            let blocks_per_row = (k as usize) / 256;
            let a_row_bytes = blocks_per_row
                .checked_mul(176) // Q5_K bytes per block
                .expect("A row bytes overflow");

            // Full-tile occupancy gate (same logic as Q4_K/Q6_K).
            let tgs_full = m_full.div_ceil(DB64_TILE_M) * (n_full / n_tile).max(1);
            let tg_size_full = if use_bn32 { DB32_TG } else { DB64_TG };
            let use_full_tiles = (tgs_full * tg_size_full) >= 32768;

            if use_full_tiles && m_full > 0 && n_full > 0 {
                let groups_x = m_full.div_ceil(DB64_TILE_M);
                if use_bn32 {
                    crate::set_pipeline_cached(encoder, self.fused_batch_q5_k_f16in_full32.state());
                    bind_buffers(encoder, a, b, c);
                    bind_u32(encoder, 3, m_full as u32);
                    bind_u32(encoder, 4, n_full as u32);
                    bind_u32(encoder, 5, k);
                    bind_u32(encoder, 6, m); // destination row stride
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: groups_x,
                            height: n_full / DB32_TILE_N,
                            depth: 1,
                        },
                        MTLSize {
                            width: DB32_TG,
                            height: 1,
                            depth: 1,
                        },
                    );
                } else {
                    crate::set_pipeline_cached(encoder, self.fused_batch_q5_k_f16in_full64.state());
                    bind_buffers(encoder, a, b, c);
                    bind_u32(encoder, 3, m_full as u32);
                    bind_u32(encoder, 4, n_full as u32);
                    bind_u32(encoder, 5, k);
                    bind_u32(encoder, 6, m); // destination row stride
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: groups_x,
                            height: n_full / DB64_TILE_N,
                            depth: 1,
                        },
                        MTLSize {
                            width: DB64_TG,
                            height: 1,
                            depth: 1,
                        },
                    );
                }

                let encode_tail =
                    |a_off: usize, b_off: usize, c_off: usize, out_cols: usize, n_rows: usize| {
                        if out_cols == 0 || n_rows == 0 {
                            return;
                        }
                        let groups_x = out_cols.div_ceil(DB_TILE_M);
                        crate::set_pipeline_cached(encoder, self.fused_batch_q5_k_f16in.state());
                        unsafe {
                            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), a_off, 0);
                            encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), b_off, 1);
                            encoder.setBuffer_offset_atIndex(Some(c.mtl_buffer()), c_off, 2);
                        }
                        bind_u32(encoder, 3, out_cols as u32);
                        bind_u32(encoder, 4, n_rows as u32);
                        bind_u32(encoder, 5, k);
                        bind_u32(encoder, 6, m); // destination row stride
                        encoder.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: groups_x,
                                height: n_rows.div_ceil(DB_TILE_N),
                                depth: 1,
                            },
                            MTLSize {
                                width: DB_TG,
                                height: 1,
                                depth: 1,
                            },
                        );
                    };

                if m_full > 0 && n_tail > 0 {
                    let b_off = n_full
                        .checked_mul(k as usize)
                        .and_then(|x| x.checked_mul(2))
                        .expect("B offset overflow");
                    let c_off = n_full
                        .checked_mul(m as usize)
                        .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                        .expect("C offset overflow");
                    if use_bn32 {
                        crate::set_pipeline_cached(
                            encoder,
                            self.fused_batch_q5_k_f16in_tail32.state(),
                        );
                        unsafe {
                            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), b_off, 1);
                            encoder.setBuffer_offset_atIndex(Some(c.mtl_buffer()), c_off, 2);
                        }
                        bind_u32(encoder, 3, m_full as u32);
                        bind_u32(encoder, 4, n_tail as u32);
                        bind_u32(encoder, 5, k);
                        bind_u32(encoder, 6, m);
                        encoder.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: m_full / DB64_TILE_M,
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: DB32_TG,
                                height: 1,
                                depth: 1,
                            },
                        );
                    } else {
                        encode_tail(0, b_off, c_off, m_full, n_tail);
                    }
                }

                if m_tail > 0 && n_full > 0 {
                    let a_off = m_full.checked_mul(a_row_bytes).expect("A offset overflow");
                    let c_off = m_full
                        .checked_mul(std::mem::size_of::<f32>())
                        .expect("C col offset overflow");
                    encode_tail(a_off, 0, c_off, m_tail, n_full);
                }

                if m_tail > 0 && n_tail > 0 {
                    let a_off = m_full.checked_mul(a_row_bytes).expect("A offset overflow");
                    let b_off = n_full
                        .checked_mul(k as usize)
                        .and_then(|x| x.checked_mul(2))
                        .expect("B offset overflow");
                    let c_off = n_full
                        .checked_mul(m as usize)
                        .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                        .and_then(|x| x.checked_add(m_full * std::mem::size_of::<f32>()))
                        .expect("C offset overflow");
                    encode_tail(a_off, b_off, c_off, m_tail, n_tail);
                }
                return;
            }
        }

        let groups_x = (m as usize).div_ceil(DB_TILE_M);
        let groups_y = if use_small {
            (n as usize).div_ceil(SB_TILE_N)
        } else {
            (n as usize).div_ceil(DB_TILE_N)
        };
        crate::set_pipeline_cached(
            encoder,
            if use_small {
                self.fused_batch_q5_k_f16in_small.state()
            } else {
                self.fused_batch_q5_k_f16in.state()
            },
        );
        bind_buffers(encoder, a, b, c);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        bind_u32(encoder, 6, m);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: groups_y,
                depth: 1,
            },
            MTLSize {
                width: if use_small { SB_TG } else { DB_TG },
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a B-transposed batch dequant Q6_K + matmul.
    ///
    /// C[N × M] = B[N × K] × dequant(A[M × K])^T
    ///
    /// - `a`: M × (K/256) Q6_K blocks (quantized weight matrix)
    /// - `b`: N × K f32 values (batch input, rows = tokens)
    /// - `c`: N × M f32 values (batch output, rows = tokens)
    /// - `m`: output features (weight rows)
    /// - `n`: number of tokens (batch size)
    /// - `k`: input features (must be multiple of 256)
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_q6_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        // Blocked-layout Q6_K kernel (same architecture as Q4_K blocked).
        if batch_q6k_blocked_enabled() && k.is_multiple_of(Q6_K_BLOCK_VALUES as u32) {
            const BLOCKED_TG: usize = 128;
            let use_bm32 = false; // BM=32 tested: no improvement on small M (GPU saturation not the bottleneck)
            let full_tile = if use_bm32 {
                (m as usize).is_multiple_of(32) && (n as usize).is_multiple_of(32)
            } else {
                (m as usize).is_multiple_of(64) && (n as usize).is_multiple_of(32)
            };
            // G24: full-tile path needs only sa+sb (6 KB for BM=64, 4 KB for
            // BM=32). Boundary path reuses sa as float output staging (8 KB).
            let (pipeline, bm, smem) = if use_bm32 {
                (
                    if full_tile {
                        &self.fused_batch_q6_k_blocked_bm32_fulltile
                    } else {
                        &self.fused_batch_q6_k_blocked_bm32
                    },
                    32usize,
                    if full_tile { 4096usize } else { 8192usize },
                )
            } else {
                (
                    if full_tile {
                        &self.fused_batch_q6_k_blocked_fulltile
                    } else {
                        &self.fused_batch_q6_k_blocked
                    },
                    64usize,
                    if full_tile { 6144usize } else { 8192usize },
                )
            };
            crate::set_pipeline_cached(encoder, pipeline.state());
            bind_buffers(encoder, a, b, c);
            bind_u32(encoder, 3, m);
            bind_u32(encoder, 4, n);
            bind_u32(encoder, 5, k);
            unsafe {
                encoder.setThreadgroupMemoryLength_atIndex(smem, 0);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (n as usize).div_ceil(32),
                    height: (m as usize).div_ceil(bm),
                    depth: 1,
                },
                MTLSize {
                    width: BLOCKED_TG,
                    height: 1,
                    depth: 1,
                },
            );
            return;
        }

        let use_small = n < batch_q6k_small_n_threshold();

        let groups_x = (m as usize).div_ceil(DB_TILE_M);
        let groups_y = if use_small {
            (n as usize).div_ceil(SB_TILE_N)
        } else {
            (n as usize).div_ceil(DB_TILE_N)
        };

        crate::set_pipeline_cached(
            encoder,
            if use_small {
                self.fused_batch_q6_k_small.state()
            } else {
                self.fused_batch_q6_k.state()
            },
        );
        bind_buffers(encoder, a, b, c);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: groups_y,
                depth: 1,
            },
            MTLSize {
                width: if use_small { SB_TG } else { DB_TG },
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a fused SiLU activation + blocked Q4_K down projection.
    ///
    /// C[N × M] = (silu(gate) * up)[N × K] × dequant(A[M × K])^T
    ///
    /// Fuses the SiLU activation dispatch into the blocked Q4_K matmul,
    /// saving one Metal dispatch per prefill layer.
    /// - `a`: quantized weights [M × K/256] (Q4_K blocks)
    /// - `gate`: gate activations [N × K] (f32)
    /// - `up`: up activations [N × K] (f32)
    /// - `c`: output [N × M] (f32)
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_q4_k_silu(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        if !batch_q4k_blocked_enabled() || !k.is_multiple_of(Q4_K_BLOCK_VALUES as u32) {
            return; // fallback to separate silu+matmul
        }
        const BLOCKED_TG: usize = 128;
        let full_tile = (m as usize).is_multiple_of(64) && (n as usize).is_multiple_of(32);
        // Full-tile path needs only sa+sb (6 KB). Boundary path reuses sa
        // as float output staging (8 KB).
        let (pipeline, smem) = if full_tile {
            (&self.fused_batch_q4_k_blocked_silu_fulltile, 6144usize)
        } else {
            (&self.fused_batch_q4_k_blocked_silu, 8192usize)
        };
        crate::set_pipeline_cached(encoder, pipeline.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(c.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, m);
        bind_u32(encoder, 5, n);
        bind_u32(encoder, 6, k);
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(smem, 0);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n as usize).div_ceil(32),
                height: (m as usize).div_ceil(64),
                depth: 1,
            },
            MTLSize {
                width: BLOCKED_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a fused SiLU activation + blocked Q6_K down projection.
    ///
    /// C[N × M] = dequant(A[M × K]) × silu(gate[N × K]) * up[N × K]
    /// - `a`: Q6_K quantized down weights [M × K]
    /// - `gate`: gate activations [N × K] (f32)
    /// - `up`: up activations [N × K] (f32)
    /// - `c`: output [N × M] (f32)
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_q6_k_silu(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        if !batch_q6k_blocked_enabled() || !k.is_multiple_of(Q6_K_BLOCK_VALUES as u32) {
            return; // fallback to separate silu+matmul
        }
        const BLOCKED_TG: usize = 128;
        let full_tile = (m as usize).is_multiple_of(32) && (n as usize).is_multiple_of(32);
        // Both paths use 4096 bytes: sa[2KB] + sb[2KB] = 4KB, and boundary
        // output staging = NR0*NR1*4 = 32*32*4 = 4096 bytes (same size).
        let (pipeline, smem) = if full_tile {
            (&self.fused_batch_q6_k_blocked_silu_fulltile, 4096usize)
        } else {
            (&self.fused_batch_q6_k_blocked_silu, 4096usize)
        };
        crate::set_pipeline_cached(encoder, pipeline.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(c.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, m);
        bind_u32(encoder, 5, n);
        bind_u32(encoder, 6, k);
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(smem, 0);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n as usize).div_ceil(32),
                height: (m as usize).div_ceil(32),
                depth: 1,
            },
            MTLSize {
                width: BLOCKED_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a B-transposed batch dequant Q4_K + matmul with f16 input and f32 output.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_q4_k_f16in_with_config(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
        config: DequantDispatchConfig,
    ) {
        let small_n_threshold = config.batch_f16in_small_n_threshold;
        let small_m_max = config.batch_f16in_small_m_max;
        let use_small =
            small_n_threshold > 1 && n < small_n_threshold && small_m_max > 0 && m <= small_m_max;

        if !use_small {
            let m_full = (m as usize / DB64_TILE_M) * DB64_TILE_M;
            let m_tail = (m as usize).saturating_sub(m_full);

            // Full-tile kernels use BM=64 (no bounds checks, faster inner loop).
            // But BM=64 halves the M-group count vs BM=32. For small M this
            // starves the GPU. Gate: need ≥ 128 full-tile threadgroups to use
            // the full-tile path; otherwise fall through to the standard kernel
            // (BM=32, BN=64, TG=256) which has bounds checks but 2x more TGs.
            let bn32_preferred = config.batch_f16in_use_bn32;
            let tgs_bn32 = m_full.div_ceil(DB64_TILE_M) * (n as usize / DB32_TILE_N).max(1);
            let use_bn32 = bn32_preferred && (tgs_bn32 * DB32_TG) >= 32768;
            let n_tile = if use_bn32 { DB32_TILE_N } else { DB64_TILE_N };
            let n_full = (n as usize / n_tile) * n_tile;
            let n_tail = (n as usize).saturating_sub(n_full);
            let blocks_per_row = (k as usize) / 256;
            let a_row_bytes = blocks_per_row
                .checked_mul(144) // Q4_K bytes per block
                .expect("A row bytes overflow");

            // Skip full-tile path when it would produce too few threadgroups.
            let tgs_full = m_full.div_ceil(DB64_TILE_M) * (n_full / n_tile).max(1);
            let tg_size_full = if use_bn32 { DB32_TG } else { DB64_TG };
            let use_full_tiles = (tgs_full * tg_size_full) >= 32768;

            if use_full_tiles && m_full > 0 && n_full > 0 {
                // Full-tile path: dispatches full64/full32 for the aligned portion,
                // then tail kernels for boundary tiles, then returns.
                let groups_x = m_full.div_ceil(DB64_TILE_M);
                if use_bn32 {
                    crate::set_pipeline_cached(encoder, self.fused_batch_q4_k_f16in_full32.state());
                    bind_buffers(encoder, a, b, c);
                    bind_u32(encoder, 3, m_full as u32);
                    bind_u32(encoder, 4, n_full as u32);
                    bind_u32(encoder, 5, k);
                    bind_u32(encoder, 6, m); // destination row stride
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: groups_x,
                            height: n_full / DB32_TILE_N,
                            depth: 1,
                        },
                        MTLSize {
                            width: DB32_TG,
                            height: 1,
                            depth: 1,
                        },
                    );
                } else {
                    let use_bk32 = config.batch_f16in_use_bk32;
                    crate::set_pipeline_cached(
                        encoder,
                        if use_bk32 {
                            self.fused_batch_q4_k_f16in_full64_bk32.state()
                        } else {
                            self.fused_batch_q4_k_f16in_full64.state()
                        },
                    );
                    bind_buffers(encoder, a, b, c);
                    bind_u32(encoder, 3, m_full as u32);
                    bind_u32(encoder, 4, n_full as u32);
                    bind_u32(encoder, 5, k);
                    bind_u32(encoder, 6, m); // destination row stride
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: groups_x,
                            height: n_full / DB64_TILE_N,
                            depth: 1,
                        },
                        MTLSize {
                            width: DB64_TG,
                            height: 1,
                            depth: 1,
                        },
                    );
                }

                let encode_tail =
                    |a_off: usize, b_off: usize, c_off: usize, out_cols: usize, n_rows: usize| {
                        if out_cols == 0 || n_rows == 0 {
                            return;
                        }
                        let groups_x = out_cols.div_ceil(DB_TILE_M);
                        crate::set_pipeline_cached(encoder, self.fused_batch_q4_k_f16in.state());
                        unsafe {
                            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), a_off, 0);
                            encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), b_off, 1);
                            encoder.setBuffer_offset_atIndex(Some(c.mtl_buffer()), c_off, 2);
                        }
                        bind_u32(encoder, 3, out_cols as u32);
                        bind_u32(encoder, 4, n_rows as u32);
                        bind_u32(encoder, 5, k);
                        bind_u32(encoder, 6, m); // destination row stride
                        encoder.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: groups_x,
                                height: n_rows.div_ceil(DB_TILE_N),
                                depth: 1,
                            },
                            MTLSize {
                                width: DB_TG,
                                height: 1,
                                depth: 1,
                            },
                        );
                    };

                if m_full > 0 && n_tail > 0 {
                    let b_off = n_full
                        .checked_mul(k as usize)
                        .and_then(|x| x.checked_mul(2))
                        .expect("B offset overflow");
                    let c_off = n_full
                        .checked_mul(m as usize)
                        .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                        .expect("C offset overflow");
                    if use_bn32 {
                        crate::set_pipeline_cached(
                            encoder,
                            self.fused_batch_q4_k_f16in_tail32.state(),
                        );
                        unsafe {
                            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), b_off, 1);
                            encoder.setBuffer_offset_atIndex(Some(c.mtl_buffer()), c_off, 2);
                        }
                        bind_u32(encoder, 3, m_full as u32);
                        bind_u32(encoder, 4, n_tail as u32);
                        bind_u32(encoder, 5, k);
                        bind_u32(encoder, 6, m);
                        encoder.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: m_full / DB64_TILE_M,
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: DB32_TG,
                                height: 1,
                                depth: 1,
                            },
                        );
                    } else {
                        encode_tail(0, b_off, c_off, m_full, n_tail);
                    }
                }

                if m_tail > 0 && n_full > 0 {
                    let a_off = m_full.checked_mul(a_row_bytes).expect("A offset overflow");
                    let c_off = m_full
                        .checked_mul(std::mem::size_of::<f32>())
                        .expect("C col offset overflow");
                    encode_tail(a_off, 0, c_off, m_tail, n_full);
                }

                if m_tail > 0 && n_tail > 0 {
                    let a_off = m_full.checked_mul(a_row_bytes).expect("A offset overflow");
                    let b_off = n_full
                        .checked_mul(k as usize)
                        .and_then(|x| x.checked_mul(2))
                        .expect("B offset overflow");
                    let c_off = n_full
                        .checked_mul(m as usize)
                        .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                        .and_then(|x| x.checked_add(m_full * std::mem::size_of::<f32>()))
                        .expect("C offset overflow");
                    encode_tail(a_off, b_off, c_off, m_tail, n_tail);
                }
                return;
            }
        }

        // BM=32 full-tile path: no out_tile → 12 KB TG → 2 TGs/SM.
        // Use when both M and N are aligned and not small.
        if !use_small
            && (m as usize).is_multiple_of(DB_TILE_M)
            && (n as usize).is_multiple_of(DB_TILE_N)
        {
            let groups_x = (m as usize) / DB_TILE_M;
            let groups_y = (n as usize) / DB_TILE_N;
            crate::set_pipeline_cached(encoder, self.fused_batch_q4_k_f16in_full.state());
            bind_buffers(encoder, a, b, c);
            bind_u32(encoder, 3, m);
            bind_u32(encoder, 4, n);
            bind_u32(encoder, 5, k);
            // No C_STRIDE (buffer 6) — full kernel uses M as stride.
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups_x,
                    height: groups_y,
                    depth: 1,
                },
                MTLSize {
                    width: DB_TG,
                    height: 1,
                    depth: 1,
                },
            );
            return;
        }

        let groups_x = (m as usize).div_ceil(DB_TILE_M);
        let groups_y = if use_small {
            (n as usize).div_ceil(SB_TILE_N)
        } else {
            (n as usize).div_ceil(DB_TILE_N)
        };
        crate::set_pipeline_cached(
            encoder,
            if use_small {
                self.fused_batch_q4_k_f16in_small.state()
            } else {
                self.fused_batch_q4_k_f16in.state()
            },
        );
        bind_buffers(encoder, a, b, c);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        bind_u32(encoder, 6, m);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: groups_y,
                depth: 1,
            },
            MTLSize {
                width: if use_small { SB_TG } else { DB_TG },
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a B-transposed batch dequant Q6_K + matmul with f16 input and f32 output.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_q6_k_f16in_with_config(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
        config: DequantDispatchConfig,
    ) {
        let small_n_threshold = config.batch_f16in_small_n_threshold;
        let small_m_max = config.batch_f16in_small_m_max;
        let use_small =
            small_n_threshold > 1 && n < small_n_threshold && small_m_max > 0 && m <= small_m_max;

        if !use_small {
            let m_full = (m as usize / DB64_TILE_M) * DB64_TILE_M;
            let m_tail = (m as usize).saturating_sub(m_full);
            // BN=32 occupancy gate (same logic as Q4_K — see above).
            let bn32_preferred = config.batch_f16in_use_bn32;
            let tgs_bn32 = m_full.div_ceil(DB64_TILE_M) * (n as usize / DB32_TILE_N).max(1);
            let use_bn32 = bn32_preferred && (tgs_bn32 * DB32_TG) >= 32768;
            let n_tile = if use_bn32 { DB32_TILE_N } else { DB64_TILE_N };
            let n_full = (n as usize / n_tile) * n_tile;
            let n_tail = (n as usize).saturating_sub(n_full);
            let blocks_per_row = (k as usize) / 256;
            let a_row_bytes = blocks_per_row
                .checked_mul(210) // Q6_K bytes per block
                .expect("A row bytes overflow");

            // Full-tile occupancy gate (same logic as Q4_K).
            let tgs_full = m_full.div_ceil(DB64_TILE_M) * (n_full / n_tile).max(1);
            let tg_size_full = if use_bn32 { DB32_TG } else { DB64_TG };
            let use_full_tiles = (tgs_full * tg_size_full) >= 32768;

            if use_full_tiles && m_full > 0 && n_full > 0 {
                let groups_x = m_full.div_ceil(DB64_TILE_M);
                if use_bn32 {
                    crate::set_pipeline_cached(encoder, self.fused_batch_q6_k_f16in_full32.state());
                    bind_buffers(encoder, a, b, c);
                    bind_u32(encoder, 3, m_full as u32);
                    bind_u32(encoder, 4, n_full as u32);
                    bind_u32(encoder, 5, k);
                    bind_u32(encoder, 6, m); // destination row stride
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: groups_x,
                            height: n_full / DB32_TILE_N,
                            depth: 1,
                        },
                        MTLSize {
                            width: DB32_TG,
                            height: 1,
                            depth: 1,
                        },
                    );
                } else {
                    crate::set_pipeline_cached(encoder, self.fused_batch_q6_k_f16in_full64.state());
                    bind_buffers(encoder, a, b, c);
                    bind_u32(encoder, 3, m_full as u32);
                    bind_u32(encoder, 4, n_full as u32);
                    bind_u32(encoder, 5, k);
                    bind_u32(encoder, 6, m); // destination row stride
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: groups_x,
                            height: n_full / DB64_TILE_N,
                            depth: 1,
                        },
                        MTLSize {
                            width: DB64_TG,
                            height: 1,
                            depth: 1,
                        },
                    );
                }

                let encode_tail =
                    |a_off: usize, b_off: usize, c_off: usize, out_cols: usize, n_rows: usize| {
                        if out_cols == 0 || n_rows == 0 {
                            return;
                        }
                        let groups_x = out_cols.div_ceil(DB_TILE_M);
                        crate::set_pipeline_cached(encoder, self.fused_batch_q6_k_f16in.state());
                        unsafe {
                            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), a_off, 0);
                            encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), b_off, 1);
                            encoder.setBuffer_offset_atIndex(Some(c.mtl_buffer()), c_off, 2);
                        }
                        bind_u32(encoder, 3, out_cols as u32);
                        bind_u32(encoder, 4, n_rows as u32);
                        bind_u32(encoder, 5, k);
                        bind_u32(encoder, 6, m); // destination row stride
                        encoder.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: groups_x,
                                height: n_rows.div_ceil(DB_TILE_N),
                                depth: 1,
                            },
                            MTLSize {
                                width: DB_TG,
                                height: 1,
                                depth: 1,
                            },
                        );
                    };

                if m_full > 0 && n_tail > 0 {
                    let b_off = n_full
                        .checked_mul(k as usize)
                        .and_then(|x| x.checked_mul(2))
                        .expect("B offset overflow");
                    let c_off = n_full
                        .checked_mul(m as usize)
                        .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                        .expect("C offset overflow");
                    if use_bn32 {
                        crate::set_pipeline_cached(
                            encoder,
                            self.fused_batch_q6_k_f16in_tail32.state(),
                        );
                        unsafe {
                            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), b_off, 1);
                            encoder.setBuffer_offset_atIndex(Some(c.mtl_buffer()), c_off, 2);
                        }
                        bind_u32(encoder, 3, m_full as u32);
                        bind_u32(encoder, 4, n_tail as u32);
                        bind_u32(encoder, 5, k);
                        bind_u32(encoder, 6, m);
                        encoder.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: m_full / DB64_TILE_M,
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: DB32_TG,
                                height: 1,
                                depth: 1,
                            },
                        );
                    } else {
                        encode_tail(0, b_off, c_off, m_full, n_tail);
                    }
                }

                if m_tail > 0 && n_full > 0 {
                    let a_off = m_full.checked_mul(a_row_bytes).expect("A offset overflow");
                    let c_off = m_full
                        .checked_mul(std::mem::size_of::<f32>())
                        .expect("C col offset overflow");
                    encode_tail(a_off, 0, c_off, m_tail, n_full);
                }

                if m_tail > 0 && n_tail > 0 {
                    let a_off = m_full.checked_mul(a_row_bytes).expect("A offset overflow");
                    let b_off = n_full
                        .checked_mul(k as usize)
                        .and_then(|x| x.checked_mul(2))
                        .expect("B offset overflow");
                    let c_off = n_full
                        .checked_mul(m as usize)
                        .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                        .and_then(|x| x.checked_add(m_full * std::mem::size_of::<f32>()))
                        .expect("C offset overflow");
                    encode_tail(a_off, b_off, c_off, m_tail, n_tail);
                }
                return;
            }
        }

        let groups_x = (m as usize).div_ceil(DB_TILE_M);
        let groups_y = if use_small {
            (n as usize).div_ceil(SB_TILE_N)
        } else {
            (n as usize).div_ceil(DB_TILE_N)
        };
        crate::set_pipeline_cached(
            encoder,
            if use_small {
                self.fused_batch_q6_k_f16in_small.state()
            } else {
                self.fused_batch_q6_k_f16in.state()
            },
        );
        bind_buffers(encoder, a, b, c);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        bind_u32(encoder, 6, m);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: groups_y,
                depth: 1,
            },
            MTLSize {
                width: if use_small { SB_TG } else { DB_TG },
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a B-transposed batch dequant Q8_0 + matmul with f32 input and f32 output.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_q8_0(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        // DB_BM=32, DB_BN=64, DB_TG=256
        crate::set_pipeline_cached(encoder, self.fused_batch_q8_0.state());
        bind_buffers(encoder, a, b, c);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        bind_u32(encoder, 6, m); // C_STRIDE = M
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n as usize).div_ceil(64),  // DB_BN=64
                height: (m as usize).div_ceil(32), // DB_BM=32
                depth: 1,
            },
            MTLSize {
                width: 256, // DB_TG
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a B-transposed blocked Q5_0 batch dequant + matmul with f16 input.
    /// C[N×M] = B[N×K] × dequant(A[M×K])^T.  BM=64, BN=32, TG=128.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_q5_0_blocked_f16in(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        const BLOCKED_TG: usize = 128;
        crate::set_pipeline_cached(encoder, self.fused_batch_q5_0_blocked_f16in.state());
        bind_buffers(encoder, a, b, c);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        bind_u32(encoder, 6, m); // C_STRIDE = M
        unsafe { encoder.setThreadgroupMemoryLength_atIndex(8192, 0) }; // sa=4KB + sb=2KB + output staging
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n as usize).div_ceil(32),
                height: (m as usize).div_ceil(64),
                depth: 1,
            },
            MTLSize {
                width: BLOCKED_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a B-transposed blocked Q5_1 batch dequant + matmul with f16 input.
    /// C[N×M] = B[N×K] × dequant(A[M×K])^T.  BM=64, BN=32, TG=128.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_q5_1_blocked_f16in(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        const BLOCKED_TG: usize = 128;
        crate::set_pipeline_cached(encoder, self.fused_batch_q5_1_blocked_f16in.state());
        bind_buffers(encoder, a, b, c);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        bind_u32(encoder, 6, m); // C_STRIDE = M
        unsafe { encoder.setThreadgroupMemoryLength_atIndex(8192, 0) };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n as usize).div_ceil(32),
                height: (m as usize).div_ceil(64),
                depth: 1,
            },
            MTLSize {
                width: BLOCKED_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode a B-transposed batch dequant Q8_0 + matmul with f16 input and f32 output.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_q8_0_f16in_with_config(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
        config: DequantDispatchConfig,
    ) {
        // Small-N fast path: route to dedicated small kernel before other selection.
        let small_n_threshold = config.batch_f16in_small_n_threshold;
        let small_m_max = config.batch_f16in_small_m_max;
        if small_n_threshold > 1 && n < small_n_threshold && small_m_max > 0 && m <= small_m_max {
            let groups_x = (m as usize).div_ceil(DB_TILE_M);
            let groups_y = (n as usize).div_ceil(SB_TILE_N);
            crate::set_pipeline_cached(encoder, self.fused_batch_q8_0_f16in_small.state());
            bind_buffers(encoder, a, b, c);
            bind_u32(encoder, 3, m);
            bind_u32(encoder, 4, n);
            bind_u32(encoder, 5, k);
            bind_u32(encoder, 6, m);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups_x,
                    height: groups_y,
                    depth: 1,
                },
                MTLSize {
                    width: SB_TG,
                    height: 1,
                    depth: 1,
                },
            );
            return;
        }

        if batch_q8_blocked_enabled() && k.is_multiple_of(Q8_0_BLOCK_VALUES as u32) {
            const BLOCKED_TG: usize = 128;
            let full_tile = (m as usize).is_multiple_of(64) && (n as usize).is_multiple_of(32);
            crate::set_pipeline_cached(
                encoder,
                if full_tile {
                    self.fused_batch_q8_0_blocked_f16in_fulltile.state()
                } else {
                    self.fused_batch_q8_0_blocked_f16in.state()
                },
            );
            bind_buffers(encoder, a, b, c);
            bind_u32(encoder, 3, m);
            bind_u32(encoder, 4, n);
            bind_u32(encoder, 5, k);
            bind_u32(encoder, 6, m);
            unsafe {
                // G24: full-tile needs only sa+sb = 6 KB; boundary needs 8 KB.
                encoder.setThreadgroupMemoryLength_atIndex(if full_tile { 6144 } else { 8192 }, 0);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (n as usize).div_ceil(32),
                    height: (m as usize).div_ceil(64),
                    depth: 1,
                },
                MTLSize {
                    width: BLOCKED_TG,
                    height: 1,
                    depth: 1,
                },
            );
            return;
        }

        let can_use_full =
            (k as usize).is_multiple_of(64) && (n as usize) >= config.q8_f16in_full_min_n;
        let can_use_full64 = can_use_full && (m as usize).is_multiple_of(DB64_TILE_M) && {
            let m_full = (m as usize) / DB64_TILE_M;
            let n_full = (n as usize / DB64_TILE_N) * DB64_TILE_N;
            n_full > 0
                && (m_full).max(1) * (n_full / DB64_TILE_N).max(1) * DB64_TG
                    >= BATCH_F16IN_FULL_TILE_MIN_THREADS
        };
        if can_use_full64 {
            // BN=32 occupancy gate: need ≥ 32K total threads (same as Q4_K/Q6_K).
            // Also cap at N < 768 (original heuristic for untested large-N path).
            let bn32_preferred = config.batch_f16in_use_bn32 && (n as usize) < 768;
            let tgs_bn32 = (m as usize).div_ceil(DB64_TILE_M) * (n as usize / DB32_TILE_N).max(1);
            let use_bn32 = bn32_preferred && (tgs_bn32 * DB32_TG) >= 32768;
            let use_small32x32 =
                use_bn32 && (n as usize) <= 256 && (m as usize).is_multiple_of(DB32_TILE_M);
            let n_full = if use_bn32 {
                (n as usize / DB32_TILE_N) * DB32_TILE_N
            } else {
                (n as usize / DB64_TILE_N) * DB64_TILE_N
            };
            let n_tail = (n as usize).saturating_sub(n_full);
            if n_full > 0 {
                let groups_x = if use_small32x32 {
                    (m as usize) / DB32_TILE_M
                } else {
                    (m as usize) / DB64_TILE_M
                };
                let groups_y = if use_bn32 {
                    n_full / DB32_TILE_N
                } else {
                    n_full / DB64_TILE_N
                };
                if use_small32x32 {
                    crate::set_pipeline_cached(
                        encoder,
                        self.fused_batch_q8_0_f16in_full32x32.state(),
                    );
                } else if use_bn32 {
                    crate::set_pipeline_cached(encoder, self.fused_batch_q8_0_f16in_full32.state());
                } else {
                    crate::set_pipeline_cached(encoder, self.fused_batch_q8_0_f16in_full64.state());
                }
                bind_buffers(encoder, a, b, c);
                bind_u32(encoder, 3, m);
                bind_u32(encoder, 4, n_full as u32);
                bind_u32(encoder, 5, k);
                bind_u32(encoder, 6, m);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: groups_x,
                        height: groups_y,
                        depth: 1,
                    },
                    MTLSize {
                        width: if use_small32x32 {
                            DB32X32_TG
                        } else if use_bn32 {
                            DB32_TG
                        } else {
                            DB64_TG
                        },
                        height: 1,
                        depth: 1,
                    },
                );
            }
            if n_tail == 0 {
                return;
            }
            let b_off = n_full
                .checked_mul(k as usize)
                .and_then(|x| x.checked_mul(2))
                .expect("B offset overflow");
            let c_off = n_full
                .checked_mul(m as usize)
                .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                .expect("C offset overflow");
            if use_bn32 {
                crate::set_pipeline_cached(encoder, self.fused_batch_q8_0_f16in_tail32.state());
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), b_off, 1);
                    encoder.setBuffer_offset_atIndex(Some(c.mtl_buffer()), c_off, 2);
                }
                bind_u32(encoder, 3, m);
                bind_u32(encoder, 4, n_tail as u32);
                bind_u32(encoder, 5, k);
                bind_u32(encoder, 6, m);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: (m as usize) / DB64_TILE_M,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: DB32_TG,
                        height: 1,
                        depth: 1,
                    },
                );
            } else {
                let groups_x = (m as usize).div_ceil(DB_TILE_M);
                let groups_y = n_tail.div_ceil(DB_TILE_N);
                crate::set_pipeline_cached(encoder, self.fused_batch_q8_0_f16in.state());
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), b_off, 1);
                    encoder.setBuffer_offset_atIndex(Some(c.mtl_buffer()), c_off, 2);
                }
                bind_u32(encoder, 3, m);
                bind_u32(encoder, 4, n_tail as u32);
                bind_u32(encoder, 5, k);
                bind_u32(encoder, 6, m);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: groups_x,
                        height: groups_y,
                        depth: 1,
                    },
                    MTLSize {
                        width: DB_TG,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            return;
        }

        if can_use_full && (m as usize).is_multiple_of(DB_TILE_M) {
            let n_full = (n as usize / DB_TILE_N) * DB_TILE_N;
            let n_tail = (n as usize).saturating_sub(n_full);
            if n_full > 0 {
                let groups_x = (m as usize) / DB_TILE_M;
                let groups_y = n_full / DB_TILE_N;
                crate::set_pipeline_cached(encoder, self.fused_batch_q8_0_f16in_full.state());
                bind_buffers(encoder, a, b, c);
                bind_u32(encoder, 3, m);
                bind_u32(encoder, 4, n_full as u32);
                bind_u32(encoder, 5, k);
                bind_u32(encoder, 6, m);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: groups_x,
                        height: groups_y,
                        depth: 1,
                    },
                    MTLSize {
                        width: DB_TG,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            if n_tail == 0 {
                return;
            }
            let b_off = n_full
                .checked_mul(k as usize)
                .and_then(|x| x.checked_mul(2))
                .expect("B offset overflow");
            let c_off = n_full
                .checked_mul(m as usize)
                .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                .expect("C offset overflow");
            let groups_x = (m as usize).div_ceil(DB_TILE_M);
            let groups_y = n_tail.div_ceil(DB_TILE_N);
            crate::set_pipeline_cached(encoder, self.fused_batch_q8_0_f16in.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), b_off, 1);
                encoder.setBuffer_offset_atIndex(Some(c.mtl_buffer()), c_off, 2);
            }
            bind_u32(encoder, 3, m);
            bind_u32(encoder, 4, n_tail as u32);
            bind_u32(encoder, 5, k);
            bind_u32(encoder, 6, m);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups_x,
                    height: groups_y,
                    depth: 1,
                },
                MTLSize {
                    width: DB_TG,
                    height: 1,
                    depth: 1,
                },
            );
            return;
        }

        let groups_x = (m as usize).div_ceil(DB_TILE_M);
        let groups_y = (n as usize).div_ceil(DB_TILE_N);
        crate::set_pipeline_cached(encoder, self.fused_batch_q8_0_f16in.state());
        bind_buffers(encoder, a, b, c);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        bind_u32(encoder, 6, m);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: groups_y,
                depth: 1,
            },
            MTLSize {
                width: DB_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode dual-output B-transposed batch dequant Q4_K + matmul.
    ///
    /// C0/C1[N × M] = B[N × K] × dequant(A0/A1[M × K])^T
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_pair_q4_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a0: &MetalBuffer,
        a1: &MetalBuffer,
        b: &MetalBuffer,
        c0: &MetalBuffer,
        c1: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let groups_x = (m as usize).div_ceil(DB_TILE_M);
        let groups_y = (n as usize).div_ceil(PB_TILE_N);
        crate::set_pipeline_cached(encoder, self.fused_batch_pair_q4_k.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a0.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(a1.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(c0.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(c1.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, m);
        bind_u32(encoder, 6, n);
        bind_u32(encoder, 7, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: groups_y,
                depth: 1,
            },
            MTLSize {
                width: PB_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode dual-output B-transposed batch dequant Q6_K + matmul.
    ///
    /// C0/C1[N × M] = B[N × K] × dequant(A0/A1[M × K])^T
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_pair_q6_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a0: &MetalBuffer,
        a1: &MetalBuffer,
        b: &MetalBuffer,
        c0: &MetalBuffer,
        c1: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let groups_x = (m as usize).div_ceil(DB_TILE_M);
        let groups_y = (n as usize).div_ceil(PB_TILE_N);
        crate::set_pipeline_cached(encoder, self.fused_batch_pair_q6_k.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a0.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(a1.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(c0.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(c1.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, m);
        bind_u32(encoder, 6, n);
        bind_u32(encoder, 7, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: groups_y,
                depth: 1,
            },
            MTLSize {
                width: PB_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode dual-output B-transposed batch dequant Q4_K + matmul with f16 input.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_pair_q4_k_f16in(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a0: &MetalBuffer,
        a1: &MetalBuffer,
        b_f16: &MetalBuffer,
        c0: &MetalBuffer,
        c1: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let groups_x = (m as usize).div_ceil(DB_TILE_M);
        let groups_y = (n as usize).div_ceil(P16_TILE_N);
        crate::set_pipeline_cached(encoder, self.fused_batch_pair_q4_k_f16in.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a0.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(a1.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(b_f16.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(c0.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(c1.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, m);
        bind_u32(encoder, 6, n);
        bind_u32(encoder, 7, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: groups_y,
                depth: 1,
            },
            MTLSize {
                width: P16_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode dual-output B-transposed batch dequant Q6_K + matmul with f16 input.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_pair_q6_k_f16in(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a0: &MetalBuffer,
        a1: &MetalBuffer,
        b_f16: &MetalBuffer,
        c0: &MetalBuffer,
        c1: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let groups_x = (m as usize).div_ceil(DB_TILE_M);
        let groups_y = (n as usize).div_ceil(P16_TILE_N);
        crate::set_pipeline_cached(encoder, self.fused_batch_pair_q6_k_f16in.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a0.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(a1.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(b_f16.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(c0.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(c1.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, m);
        bind_u32(encoder, 6, n);
        bind_u32(encoder, 7, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: groups_y,
                depth: 1,
            },
            MTLSize {
                width: P16_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode dual-output B-transposed batch dequant Q5_K + matmul with f16 input.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_pair_q5_k_f16in(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a0: &MetalBuffer,
        a1: &MetalBuffer,
        b_f16: &MetalBuffer,
        c0: &MetalBuffer,
        c1: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let groups_x = (m as usize).div_ceil(DB_TILE_M);
        let groups_y = (n as usize).div_ceil(P16_TILE_N);
        crate::set_pipeline_cached(encoder, self.fused_batch_pair_q5_k_f16in.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a0.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(a1.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(b_f16.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(c0.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(c1.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, m);
        bind_u32(encoder, 6, n);
        bind_u32(encoder, 7, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: groups_y,
                depth: 1,
            },
            MTLSize {
                width: P16_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode dual-output B-transposed batch dequant Q8_0 + matmul with f16 input.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_pair_q8_0_f16in(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a0: &MetalBuffer,
        a1: &MetalBuffer,
        b_f16: &MetalBuffer,
        c0: &MetalBuffer,
        c1: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let can_use_full =
            (m as usize).is_multiple_of(DB_TILE_M) && (k as usize).is_multiple_of(64);
        if can_use_full {
            let n_full = (n as usize / P16_TILE_N) * P16_TILE_N;
            let n_tail = (n as usize).saturating_sub(n_full);
            if n_full > 0 {
                let groups_x = (m as usize) / DB_TILE_M;
                let groups_y = n_full / P16_TILE_N;
                crate::set_pipeline_cached(encoder, self.fused_batch_pair_q8_0_f16in_full.state());
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(a0.mtl_buffer()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(a1.mtl_buffer()), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(b_f16.mtl_buffer()), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(c0.mtl_buffer()), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(c1.mtl_buffer()), 0, 4);
                }
                bind_u32(encoder, 5, m);
                bind_u32(encoder, 6, n_full as u32);
                bind_u32(encoder, 7, k);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: groups_x,
                        height: groups_y,
                        depth: 1,
                    },
                    MTLSize {
                        width: P16_TG,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            if n_tail == 0 {
                return;
            }
            let b_off = n_full
                .checked_mul(k as usize)
                .and_then(|x| x.checked_mul(2))
                .expect("B offset overflow");
            let c_off = n_full
                .checked_mul(m as usize)
                .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                .expect("C offset overflow");
            let groups_x = (m as usize).div_ceil(DB_TILE_M);
            let groups_y = n_tail.div_ceil(P16_TILE_N);
            crate::set_pipeline_cached(encoder, self.fused_batch_pair_q8_0_f16in.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(a0.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(a1.mtl_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(b_f16.mtl_buffer()), b_off, 2);
                encoder.setBuffer_offset_atIndex(Some(c0.mtl_buffer()), c_off, 3);
                encoder.setBuffer_offset_atIndex(Some(c1.mtl_buffer()), c_off, 4);
            }
            bind_u32(encoder, 5, m);
            bind_u32(encoder, 6, n_tail as u32);
            bind_u32(encoder, 7, k);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups_x,
                    height: groups_y,
                    depth: 1,
                },
                MTLSize {
                    width: P16_TG,
                    height: 1,
                    depth: 1,
                },
            );
            return;
        }

        let groups_x = (m as usize).div_ceil(DB_TILE_M);
        let groups_y = (n as usize).div_ceil(P16_TILE_N);
        crate::set_pipeline_cached(encoder, self.fused_batch_pair_q8_0_f16in.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a0.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(a1.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(b_f16.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(c0.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(c1.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, m);
        bind_u32(encoder, 6, n);
        bind_u32(encoder, 7, k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: groups_y,
                depth: 1,
            },
            MTLSize {
                width: P16_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode MoE map0: build routing index on GPU.
    /// expert_ids: [n_tokens, n_expert_used] int32
    /// tpe: [n_expert] uint32 (output: token count per expert)
    /// hids: [n_expert, n_tokens] int32 (output: routing index)
    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_map0(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        expert_ids: &MetalBuffer,
        tpe: &MetalBuffer,
        hids: &MetalBuffer,
        active_experts: &MetalBuffer,
        n_tokens: u32,
        n_expert_used: u32,
        n_expert: u32,
    ) {
        crate::set_pipeline_cached(encoder, self.moe_map0.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(expert_ids.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(tpe.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(hids.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(active_experts.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n_tokens);
        bind_u32(encoder, 5, n_expert_used);
        bind_u32(encoder, 6, n_expert);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: n_expert as usize,
                height: 1,
                depth: 1,
            },
        );
    }
    /// Encode MoE mul_mat_id for Q4_K: unified expert matmul.
    /// weights: [n_expert, M, K/256] Q4_K blocks
    /// input: [n_tokens, K] f32
    /// output: [n_tokens * n_expert_used, M] f32
    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_mul_mat_id_q4_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        tpe: &MetalBuffer,
        hids: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n_tokens: u32,
        n_expert_used: u32,
        _n_expert: u32,
        weight_stride: u32,
        active_experts: &MetalBuffer,
        n_active_experts: u32,
        input_is_hid: bool,
    ) {
        // Use blocked 64×32 kernel when M >= 64, else fallback to 32×32.
        let use_blocked = m >= 64;
        if use_blocked {
            crate::set_pipeline_cached(encoder, self.moe_mul_mat_id_q4_k_blocked.state());
        } else {
            crate::set_pipeline_cached(encoder, self.moe_mul_mat_id_q4_k.state());
        }
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(input.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(tpe.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(hids.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, m);
        bind_u32(encoder, 6, k);
        bind_u32(encoder, 7, n_tokens);
        bind_u32(encoder, 8, n_expert_used);
        bind_u32(encoder, 9, weight_stride);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(active_experts.mtl_buffer()), 0, 10);
        }
        bind_u32(encoder, 11, u32::from(input_is_hid));
        // Blocked kernel needs threadgroup memory (8 KB for sa+sb, or 8 KB for output staging).
        if use_blocked {
            let smem = 8192usize; // 6 KB blocked + 2 KB margin (or 8 KB output staging)
            unsafe {
                encoder.setThreadgroupMemoryLength_atIndex(smem, 0);
            }
        }
        // Compact grid: only dispatch active experts.
        let m_tile = if use_blocked { 64usize } else { 32 };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n_tokens as usize).div_ceil(32),
                height: (m as usize).div_ceil(m_tile),
                depth: n_active_experts as usize,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode MoE mul_mat_id for Q5_K using the blocked 64x32 kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_mul_mat_id_q5_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        tpe: &MetalBuffer,
        hids: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n_tokens: u32,
        n_expert_used: u32,
        _n_expert: u32,
        weight_stride: u32,
        active_experts: &MetalBuffer,
        n_active_experts: u32,
        input_is_hid: bool,
    ) {
        crate::set_pipeline_cached(encoder, self.moe_mul_mat_id_q5_k_blocked.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(input.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(tpe.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(hids.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, m);
        bind_u32(encoder, 6, k);
        bind_u32(encoder, 7, n_tokens);
        bind_u32(encoder, 8, n_expert_used);
        bind_u32(encoder, 9, weight_stride);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(active_experts.mtl_buffer()), 0, 10);
            encoder.setThreadgroupMemoryLength_atIndex(8192, 0);
        }
        bind_u32(encoder, 11, u32::from(input_is_hid));
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n_tokens as usize).div_ceil(32),
                height: (m as usize).div_ceil(64),
                depth: n_active_experts as usize,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode MoE mul_mat_id for Q6_K using the blocked 64x32 kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_mul_mat_id_q6_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        tpe: &MetalBuffer,
        hids: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n_tokens: u32,
        n_expert_used: u32,
        _n_expert: u32,
        weight_stride: u32,
        active_experts: &MetalBuffer,
        n_active_experts: u32,
        allow_blocked_input_is_hid: bool,
        input_is_hid: bool,
    ) {
        // Gate/up always use the blocked half-tile kernel. For routed down,
        // stay on the f32 kernel unless the caller explicitly opts into the
        // faster blocked path for a model/quant family that has been validated.
        let use_blocked = !input_is_hid || allow_blocked_input_is_hid;
        if use_blocked {
            crate::set_pipeline_cached(encoder, self.moe_mul_mat_id_q6_k_blocked.state());
        } else {
            crate::set_pipeline_cached(encoder, self.moe_mul_mat_id_q6_k.state());
        }
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(input.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(tpe.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(hids.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, m);
        bind_u32(encoder, 6, k);
        bind_u32(encoder, 7, n_tokens);
        bind_u32(encoder, 8, n_expert_used);
        bind_u32(encoder, 9, weight_stride);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(active_experts.mtl_buffer()), 0, 10);
            if use_blocked {
                encoder.setThreadgroupMemoryLength_atIndex(8192, 0);
            }
        }
        bind_u32(encoder, 11, u32::from(input_is_hid));
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n_tokens as usize).div_ceil(32),
                height: (m as usize).div_ceil(if use_blocked { 64 } else { 32 }),
                depth: n_active_experts as usize,
            },
            MTLSize {
                width: if use_blocked { 128 } else { 256 },
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode MoE mul_mat_id for Q8_0.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_mul_mat_id_q8_0(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        tpe: &MetalBuffer,
        hids: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n_tokens: u32,
        n_expert_used: u32,
        _n_expert: u32,
        weight_stride: u32,
        active_experts: &MetalBuffer,
        n_active_experts: u32,
        allow_blocked_input_is_hid: bool,
        input_is_hid: bool,
    ) {
        // Gate/up always use the blocked half-tile kernel. For routed down,
        // stay on the f32 kernel unless the caller explicitly opts into the
        // faster blocked path for a model/quant family that has been validated.
        let use_blocked = !input_is_hid || allow_blocked_input_is_hid;
        if use_blocked {
            crate::set_pipeline_cached(encoder, self.moe_mul_mat_id_q8_0_blocked.state());
        } else {
            crate::set_pipeline_cached(encoder, self.moe_mul_mat_id_q8_0.state());
        }
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(input.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(tpe.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(hids.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, m);
        bind_u32(encoder, 6, k);
        bind_u32(encoder, 7, n_tokens);
        bind_u32(encoder, 8, n_expert_used);
        bind_u32(encoder, 9, weight_stride);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(active_experts.mtl_buffer()), 0, 10);
            if use_blocked {
                encoder.setThreadgroupMemoryLength_atIndex(8192, 0);
            }
        }
        bind_u32(encoder, 11, u32::from(input_is_hid));
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n_tokens as usize).div_ceil(32),
                height: (m as usize).div_ceil(if use_blocked { 64 } else { 32 }),
                depth: n_active_experts as usize,
            },
            MTLSize {
                width: if use_blocked { 128 } else { 256 },
                height: 1,
                depth: 1,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_mul_mat_selected_q4_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        selected_experts: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n_selected: u32,
        weight_stride: u32,
        input_is_slot_major: bool,
        use_matvec_kernel: bool,
    ) {
        if use_matvec_kernel {
            crate::set_pipeline_cached(encoder, self.moe_mul_mat_selected_q4_k_matvec.state());
        } else {
            crate::set_pipeline_cached(encoder, self.moe_mul_mat_selected_q4_k_blocked.state());
        }
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(input.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(selected_experts.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, m);
        bind_u32(encoder, 5, k);
        bind_u32(encoder, 6, n_selected);
        bind_u32(encoder, 7, weight_stride);
        bind_u32(encoder, 8, u32::from(input_is_slot_major));
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            if use_matvec_kernel {
                MTLSize {
                    width: (m as usize).div_ceil(2),
                    height: 1,
                    depth: n_selected as usize,
                }
            } else {
                MTLSize {
                    width: 1,
                    height: (m as usize).div_ceil(64),
                    depth: n_selected as usize,
                }
            },
            if use_matvec_kernel {
                MTLSize {
                    width: DEQUANT_MATVEC_Q4K_NR2_TG,
                    height: 1,
                    depth: 1,
                }
            } else {
                unsafe {
                    encoder.setThreadgroupMemoryLength_atIndex(8192usize, 0);
                }
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                }
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_mul_mat_selected_pair_q4_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights0: &MetalBuffer,
        weights1: &MetalBuffer,
        input: &MetalBuffer,
        selected_experts: &MetalBuffer,
        output0: &MetalBuffer,
        output1: &MetalBuffer,
        m: u32,
        k: u32,
        n_selected: u32,
        weight_stride0: u32,
        weight_stride1: u32,
        input_is_slot_major: bool,
        use_matvec_kernel: bool,
    ) {
        if use_matvec_kernel {
            crate::set_pipeline_cached(encoder, self.moe_mul_mat_selected_pair_q4_k_matvec.state());
        } else {
            crate::set_pipeline_cached(
                encoder,
                self.moe_mul_mat_selected_pair_q4_k_blocked.state(),
            );
        }
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights0.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weights1.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(input.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(selected_experts.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output0.mtl_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(output1.mtl_buffer()), 0, 5);
        }
        bind_u32(encoder, 6, m);
        bind_u32(encoder, 7, k);
        bind_u32(encoder, 8, n_selected);
        bind_u32(encoder, 9, weight_stride0);
        bind_u32(encoder, 10, weight_stride1);
        bind_u32(encoder, 11, u32::from(input_is_slot_major));
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            if use_matvec_kernel {
                MTLSize {
                    width: (m as usize).div_ceil(2),
                    height: 1,
                    depth: n_selected as usize,
                }
            } else {
                MTLSize {
                    width: 1,
                    height: (m as usize).div_ceil(64),
                    depth: n_selected as usize,
                }
            },
            if use_matvec_kernel {
                MTLSize {
                    width: DEQUANT_MATVEC_Q4K_NR2_TG,
                    height: 1,
                    depth: 1,
                }
            } else {
                unsafe {
                    encoder.setThreadgroupMemoryLength_atIndex(8192usize, 0);
                }
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                }
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_mul_mat_selected_weighted_q4_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        selected_experts: &MetalBuffer,
        expert_weights: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n_selected: u32,
        weight_stride: u32,
    ) {
        crate::set_pipeline_cached(
            encoder,
            self.moe_mul_mat_selected_weighted_q4_k_blocked.state(),
        );
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(input.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(selected_experts.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(expert_weights.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, m);
        bind_u32(encoder, 6, k);
        bind_u32(encoder, 7, n_selected);
        bind_u32(encoder, 8, weight_stride);
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(8192usize, 0);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: (m as usize).div_ceil(64),
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_mul_mat_selected_q5_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        selected_experts: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n_selected: u32,
        weight_stride: u32,
        input_is_slot_major: bool,
        use_matvec_kernel: bool,
    ) {
        crate::set_pipeline_cached(
            encoder,
            if use_matvec_kernel {
                self.moe_mul_mat_selected_q5_k_matvec.state()
            } else {
                self.moe_mul_mat_selected_q5_k_blocked.state()
            },
        );
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(input.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(selected_experts.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, m);
        bind_u32(encoder, 5, k);
        bind_u32(encoder, 6, n_selected);
        bind_u32(encoder, 7, weight_stride);
        bind_u32(encoder, 8, u32::from(input_is_slot_major));
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            if use_matvec_kernel {
                MTLSize {
                    width: (m as usize).div_ceil(Q5K_NR2_ROWS),
                    height: 1,
                    depth: n_selected as usize,
                }
            } else {
                MTLSize {
                    width: 1,
                    height: (m as usize).div_ceil(64),
                    depth: n_selected as usize,
                }
            },
            if use_matvec_kernel {
                MTLSize {
                    width: DEQUANT_MATVEC_Q5K_TG,
                    height: 1,
                    depth: 1,
                }
            } else {
                unsafe {
                    encoder.setThreadgroupMemoryLength_atIndex(8192usize, 0);
                }
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                }
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_mul_mat_selected_pair_q5_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights0: &MetalBuffer,
        weights1: &MetalBuffer,
        input: &MetalBuffer,
        selected_experts: &MetalBuffer,
        output0: &MetalBuffer,
        output1: &MetalBuffer,
        m: u32,
        k: u32,
        n_selected: u32,
        weight_stride0: u32,
        weight_stride1: u32,
        input_is_slot_major: bool,
        use_matvec_kernel: bool,
    ) {
        crate::set_pipeline_cached(
            encoder,
            if use_matvec_kernel {
                self.moe_mul_mat_selected_pair_q5_k_matvec.state()
            } else {
                self.moe_mul_mat_selected_pair_q5_k_blocked.state()
            },
        );
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights0.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weights1.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(input.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(selected_experts.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output0.mtl_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(output1.mtl_buffer()), 0, 5);
        }
        bind_u32(encoder, 6, m);
        bind_u32(encoder, 7, k);
        bind_u32(encoder, 8, n_selected);
        bind_u32(encoder, 9, weight_stride0);
        bind_u32(encoder, 10, weight_stride1);
        bind_u32(encoder, 11, u32::from(input_is_slot_major));
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            if use_matvec_kernel {
                MTLSize {
                    width: (m as usize).div_ceil(Q5K_NR2_ROWS),
                    height: 1,
                    depth: n_selected as usize,
                }
            } else {
                MTLSize {
                    width: 1,
                    height: (m as usize).div_ceil(64),
                    depth: n_selected as usize,
                }
            },
            if use_matvec_kernel {
                MTLSize {
                    width: DEQUANT_MATVEC_Q5K_TG,
                    height: 1,
                    depth: 1,
                }
            } else {
                unsafe {
                    encoder.setThreadgroupMemoryLength_atIndex(8192usize, 0);
                }
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                }
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_mul_mat_selected_pair_q6_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights0: &MetalBuffer,
        weights1: &MetalBuffer,
        input: &MetalBuffer,
        selected_experts: &MetalBuffer,
        output0: &MetalBuffer,
        output1: &MetalBuffer,
        m: u32,
        k: u32,
        n_selected: u32,
        weight_stride0: u32,
        weight_stride1: u32,
        input_is_slot_major: bool,
        config: DequantDispatchConfig,
    ) {
        let (threadgroups, threadgroup_width, pipeline) =
            self.q6_k_selected_pair_matvec_dispatch_with_config(m, config);
        crate::set_pipeline_cached(encoder, pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights0.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weights1.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(input.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(selected_experts.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output0.mtl_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(output1.mtl_buffer()), 0, 5);
        }
        bind_u32(encoder, 6, m);
        bind_u32(encoder, 7, k);
        bind_u32(encoder, 8, n_selected);
        bind_u32(encoder, 9, weight_stride0);
        bind_u32(encoder, 10, weight_stride1);
        bind_u32(encoder, 11, u32::from(input_is_slot_major));
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: threadgroups,
                height: 1,
                depth: n_selected as usize,
            },
            MTLSize {
                width: threadgroup_width,
                height: 1,
                depth: 1,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_mul_mat_selected_weighted_q5_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        selected_experts: &MetalBuffer,
        expert_weights: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n_selected: u32,
        weight_stride: u32,
    ) {
        crate::set_pipeline_cached(
            encoder,
            self.moe_mul_mat_selected_weighted_q5_k_blocked.state(),
        );
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(input.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(selected_experts.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(expert_weights.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, m);
        bind_u32(encoder, 6, k);
        bind_u32(encoder, 7, n_selected);
        bind_u32(encoder, 8, weight_stride);
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(8192usize, 0);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: (m as usize).div_ceil(64),
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_mul_mat_selected_q6_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        selected_experts: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n_selected: u32,
        weight_stride: u32,
        input_is_slot_major: bool,
        config: DequantDispatchConfig,
    ) {
        let (threadgroups, threadgroup_width, pipeline) =
            self.q6_k_selected_matvec_dispatch_with_config(m, config);
        crate::set_pipeline_cached(encoder, pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(input.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(selected_experts.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, m);
        bind_u32(encoder, 5, k);
        bind_u32(encoder, 6, n_selected);
        bind_u32(encoder, 7, weight_stride);
        bind_u32(encoder, 8, u32::from(input_is_slot_major));
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: threadgroups,
                height: 1,
                depth: n_selected as usize,
            },
            MTLSize {
                width: threadgroup_width,
                height: 1,
                depth: 1,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_mul_mat_selected_weighted_q6_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        selected_experts: &MetalBuffer,
        expert_weights: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n_selected: u32,
        weight_stride: u32,
        config: DequantDispatchConfig,
    ) {
        let (threadgroups, threadgroup_width, pipeline) =
            self.q6_k_selected_weighted_matvec_dispatch_with_config(m, config);
        crate::set_pipeline_cached(encoder, pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(input.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(selected_experts.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(expert_weights.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, m);
        bind_u32(encoder, 6, k);
        bind_u32(encoder, 7, n_selected);
        bind_u32(encoder, 8, weight_stride);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: threadgroups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threadgroup_width,
                height: 1,
                depth: 1,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_mul_mat_selected_pair_q8_0(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights0: &MetalBuffer,
        weights1: &MetalBuffer,
        input: &MetalBuffer,
        selected_experts: &MetalBuffer,
        output0: &MetalBuffer,
        output1: &MetalBuffer,
        m: u32,
        k: u32,
        n_selected: u32,
        weight_stride0: u32,
        weight_stride1: u32,
        input_is_slot_major: bool,
        config: DequantDispatchConfig,
    ) {
        let (threadgroups, threadgroup_width, pipeline) =
            self.q8_0_selected_pair_matvec_dispatch_with_config(m, config);
        crate::set_pipeline_cached(encoder, pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights0.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weights1.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(input.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(selected_experts.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output0.mtl_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(output1.mtl_buffer()), 0, 5);
        }
        bind_u32(encoder, 6, m);
        bind_u32(encoder, 7, k);
        bind_u32(encoder, 8, n_selected);
        bind_u32(encoder, 9, weight_stride0);
        bind_u32(encoder, 10, weight_stride1);
        bind_u32(encoder, 11, u32::from(input_is_slot_major));
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: threadgroups,
                height: 1,
                depth: n_selected as usize,
            },
            MTLSize {
                width: threadgroup_width,
                height: 1,
                depth: 1,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_mul_mat_selected_q8_0(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        selected_experts: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n_selected: u32,
        weight_stride: u32,
        input_is_slot_major: bool,
        config: DequantDispatchConfig,
    ) {
        let (threadgroups, threadgroup_width, pipeline) =
            self.q8_0_selected_matvec_dispatch_with_config(m, config);
        crate::set_pipeline_cached(encoder, pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(input.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(selected_experts.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, m);
        bind_u32(encoder, 5, k);
        bind_u32(encoder, 6, n_selected);
        bind_u32(encoder, 7, weight_stride);
        bind_u32(encoder, 8, u32::from(input_is_slot_major));
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: threadgroups,
                height: 1,
                depth: n_selected as usize,
            },
            MTLSize {
                width: threadgroup_width,
                height: 1,
                depth: 1,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_mul_mat_selected_weighted_q8_0(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        selected_experts: &MetalBuffer,
        expert_weights: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n_selected: u32,
        weight_stride: u32,
        config: DequantDispatchConfig,
    ) {
        let (threadgroups, threadgroup_width, pipeline) =
            self.q8_0_selected_weighted_matvec_dispatch_with_config(m, config);
        crate::set_pipeline_cached(encoder, pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(input.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(selected_experts.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(expert_weights.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, m);
        bind_u32(encoder, 6, k);
        bind_u32(encoder, 7, n_selected);
        bind_u32(encoder, 8, weight_stride);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: threadgroups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threadgroup_width,
                height: 1,
                depth: 1,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_fused_silu_down_selected_weighted_q5_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weights: &MetalBuffer,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        selected_experts: &MetalBuffer,
        expert_weights: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n_selected: u32,
        weight_stride: u32,
        use_slots8_kernel: bool,
        use_nr2_kernel: bool,
    ) {
        let dims = DispatchDims::d1(
            (m as usize).div_ceil(if use_nr2_kernel { Q5K_NR2_ROWS } else { 2 }),
            1,
        );
        let use_slots8 = use_slots8_kernel && n_selected == 8;
        crate::set_pipeline_cached(
            encoder,
            if use_nr2_kernel {
                if use_slots8 {
                    self.moe_fused_silu_down_selected_weighted_q5_k_matvec_slots8_nr2
                        .state()
                } else {
                    self.moe_fused_silu_down_selected_weighted_q5_k_matvec_nr2
                        .state()
                }
            } else {
                if use_slots8 {
                    self.moe_fused_silu_down_selected_weighted_q5_k_matvec_slots8
                        .state()
                } else {
                    self.moe_fused_silu_down_selected_weighted_q5_k_matvec
                        .state()
                }
            },
        );
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(selected_experts.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(expert_weights.mtl_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(output.mtl_buffer()), 0, 5);
        }
        bind_u32(encoder, 6, m);
        bind_u32(encoder, 7, k);
        if use_slots8 {
            bind_u32(encoder, 8, weight_stride);
        } else {
            bind_u32(encoder, 8, n_selected);
            bind_u32(encoder, 9, weight_stride);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            MTLSize {
                width: DEQUANT_MATVEC_Q5K_TG,
                height: 1,
                depth: 1,
            },
        );
    }
}
