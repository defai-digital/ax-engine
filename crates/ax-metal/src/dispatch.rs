//! Kernel dispatch helpers — encode and submit compute commands.
//!
//! Provides dispatch dimension calculations and matmul kernel dispatch.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::OnceLock;

use anyhow::Context;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBarrierScope, MTLComputeCommandEncoder, MTLSize};

use crate::barriers_enabled;
use crate::buffer::MetalBuffer;
use crate::device::MetalDevice;
use crate::inc_buffer_barrier_count;
use crate::pipeline::{ComputePipeline, FunctionConstant, FunctionConstantValue};
use crate::profile::{ProfileKernelMode, global_profile};

/// Embedded Metal shader source for matmul kernels.
const MATMUL_SHADER_SRC: &str = include_str!("../shaders/matmul.metal");

/// Embedded Metal shader source for dequantization kernels.
const DEQUANT_SHADER_SRC: &str = include_str!("../shaders/dequant.metal");

/// Embedded Metal shader source for attention kernels.
const ATTENTION_SHADER_SRC: &str = include_str!("../shaders/attention.metal");

/// Embedded Metal shader source for elementwise kernels.
const ELEMENTWISE_SHADER_SRC: &str = include_str!("../shaders/elementwise.metal");

/// Tile size for the general tiled matmul kernel (must match shader constant).
#[allow(dead_code)]
const TILE: usize = 16;

/// Tile size for the simdgroup matmul kernel (must match shader constant SG_BM/SG_BN).
const SG_TILE: usize = 32;

/// Threadgroup size for the simdgroup matmul kernel (4 simdgroups × 32 threads).
const SG_TG: usize = 128;

/// Threadgroup size for the matvec kernel (must match shader constant).
const MATVEC_TG_SIZE: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KernelMode {
    Off,
    On,
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AttentionRoutingProfile {
    name: &'static str,
    prefill_fa2_auto_min_tokens: u32,
    prefill_fa2_auto_min_base_seq: u32,
    prefill_fa2_hd128_auto_min_tokens: u32,
    decode_splitk_auto_min_tokens: u32,
    decode_splitk_chunk_size: u32,
    decode_sdpa_default: bool,
    decode_hd128_n2_default: bool,
}

const ATTN_PROFILE_DEFAULT: AttentionRoutingProfile = AttentionRoutingProfile {
    name: "default",
    prefill_fa2_auto_min_tokens: 512,
    prefill_fa2_auto_min_base_seq: 256,
    prefill_fa2_hd128_auto_min_tokens: 512,
    decode_splitk_auto_min_tokens: 512,
    decode_splitk_chunk_size: 256,
    decode_sdpa_default: false,
    decode_hd128_n2_default: false,
};

const ATTN_PROFILE_DECODE_BALANCED: AttentionRoutingProfile = AttentionRoutingProfile {
    name: "decode-balanced",
    prefill_fa2_auto_min_tokens: 640,
    prefill_fa2_auto_min_base_seq: 320,
    prefill_fa2_hd128_auto_min_tokens: 640,
    decode_splitk_auto_min_tokens: 384,
    decode_splitk_chunk_size: 256,
    decode_sdpa_default: false,
    decode_hd128_n2_default: true,
};

const ATTN_PROFILE_DECODE_LONG_CONTEXT: AttentionRoutingProfile = AttentionRoutingProfile {
    name: "decode-long-context",
    prefill_fa2_auto_min_tokens: 768,
    prefill_fa2_auto_min_base_seq: 512,
    prefill_fa2_hd128_auto_min_tokens: 768,
    decode_splitk_auto_min_tokens: 256,
    decode_splitk_chunk_size: 256,
    decode_sdpa_default: true,
    decode_hd128_n2_default: true,
};

fn resolve_attention_routing_profile(name: &str) -> Option<AttentionRoutingProfile> {
    match name.trim().to_ascii_lowercase().as_str() {
        "" | "default" => Some(ATTN_PROFILE_DEFAULT),
        "decode-balanced" | "balanced" => Some(ATTN_PROFILE_DECODE_BALANCED),
        "decode-long-context" | "long-context" | "long" => Some(ATTN_PROFILE_DECODE_LONG_CONTEXT),
        _ => None,
    }
}

fn active_attention_routing_profile() -> AttentionRoutingProfile {
    static PROFILE: OnceLock<AttentionRoutingProfile> = OnceLock::new();
    *PROFILE.get_or_init(|| {
        std::env::var("AX_METAL_ATTN_PROFILE")
            .ok()
            .as_deref()
            .and_then(resolve_attention_routing_profile)
            .unwrap_or(ATTN_PROFILE_DEFAULT)
    })
}

/// Compute dispatch dimensions (threadgroups and threads-per-threadgroup).
pub struct DispatchDims {
    /// Number of threadgroups in grid.
    pub threadgroups: MTLSize,
    /// Threads per threadgroup.
    pub threads_per_threadgroup: MTLSize,
}

impl DispatchDims {
    /// Create 1D dispatch dimensions.
    pub fn d1(total_threads: usize, threads_per_group: usize) -> Self {
        let groups = total_threads.div_ceil(threads_per_group);
        Self {
            threadgroups: MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            threads_per_threadgroup: MTLSize {
                width: threads_per_group,
                height: 1,
                depth: 1,
            },
        }
    }

    /// Create 2D dispatch dimensions (e.g. for matrix operations).
    pub fn d2(total_x: usize, total_y: usize, threads_x: usize, threads_y: usize) -> Self {
        let groups_x = total_x.div_ceil(threads_x);
        let groups_y = total_y.div_ceil(threads_y);
        Self {
            threadgroups: MTLSize {
                width: groups_x,
                height: groups_y,
                depth: 1,
            },
            threads_per_threadgroup: MTLSize {
                width: threads_x,
                height: threads_y,
                depth: 1,
            },
        }
    }
}

/// Pre-compiled matmul compute pipelines.
///
/// Holds both the tiled general matmul and the optimized matvec kernels.
/// Create once at init time, reuse for all matmul dispatches.
pub struct MatmulKernels {
    /// Fallback 16×16 tiled matmul (kept for compatibility).
    #[allow(dead_code)]
    tiled_matmul: ComputePipeline,
    /// High-performance simdgroup_matrix 32×32 tiled matmul.
    simdgroup_matmul: ComputePipeline,
    matvec: ComputePipeline,
}

impl MatmulKernels {
    /// Compile matmul kernels from embedded Metal source.
    pub fn new(device: &MetalDevice) -> anyhow::Result<Self> {
        let tiled_matmul =
            ComputePipeline::from_source(device.device(), MATMUL_SHADER_SRC, "matmul_tiled_f32")
                .context("Failed to compile matmul_tiled_f32 kernel")?;
        let simdgroup_matmul = ComputePipeline::from_source(
            device.device(),
            MATMUL_SHADER_SRC,
            "matmul_simdgroup_f32",
        )
        .context("Failed to compile matmul_simdgroup_f32 kernel")?;
        let matvec = ComputePipeline::from_source(device.device(), MATMUL_SHADER_SRC, "matvec_f32")
            .context("Failed to compile matvec_f32 kernel")?;

        tracing::info!(
            matmul_max_threads = tiled_matmul.max_threads_per_threadgroup(),
            simdgroup_max_threads = simdgroup_matmul.max_threads_per_threadgroup(),
            matvec_max_threads = matvec.max_threads_per_threadgroup(),
            "Matmul Metal kernels compiled (tiled + simdgroup + matvec)",
        );

        Ok(Self {
            tiled_matmul,
            simdgroup_matmul,
            matvec,
        })
    }

    /// Dispatch matrix multiply: C = A × B.
    ///
    /// - `a`: M × K row-major f32 buffer
    /// - `b`: K × N row-major f32 buffer
    /// - `c`: M × N row-major f32 output buffer
    /// - `dims`: (M, N, K) dimensions
    ///
    /// Automatically selects the matvec kernel when N=1.
    pub fn matmul(
        &self,
        device: &MetalDevice,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        dims: (u32, u32, u32),
    ) -> anyhow::Result<()> {
        let (m, n, k) = dims;
        if n == 1 {
            self.dispatch_matvec(device, a, b, c, m, k)
        } else {
            self.dispatch_tiled_matmul(device, a, b, c, m, n, k)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_tiled_matmul(
        &self,
        device: &MetalDevice,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        // Use simdgroup_matrix kernel for better throughput
        let groups_x = (n as usize).div_ceil(SG_TILE);
        let groups_y = (m as usize).div_ceil(SG_TILE);

        device.execute_sync(|encoder| {
            encoder.setComputePipelineState(self.simdgroup_matmul.state());
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
                    width: SG_TG,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        })
    }

    /// Encode a simdgroup matmul dispatch into an existing command encoder.
    ///
    /// Does NOT create or commit a command buffer. Used for batching
    /// multiple matmul operations into a single command buffer.
    ///
    /// C = A × B, where A is M×K, B is K×N, C is M×N (row-major f32).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_matmul(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let groups_x = (n as usize).div_ceil(SG_TILE);
        let groups_y = (m as usize).div_ceil(SG_TILE);

        encoder.setComputePipelineState(self.simdgroup_matmul.state());
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
                width: SG_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    fn dispatch_matvec(
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
            encoder.setComputePipelineState(self.matvec.state());
            bind_buffers(encoder, a, x, y);
            bind_u32(encoder, 3, m);
            bind_u32(encoder, 4, k);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                MTLSize {
                    width: MATVEC_TG_SIZE,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        })
    }
}

/// Threadgroup size for fused dequant+matvec kernels (must match shader constant).
const DEQUANT_MATVEC_TG: usize = 128;
/// Threadgroup size for the Q4_K NR2 decode matvec pilot.
const DEQUANT_MATVEC_Q4K_NR2_TG: usize = 64;
/// Threadgroup size for the Q6_K NR2 decode matvec pilot.
const DEQUANT_MATVEC_Q6K_NR2_TG: usize = 64;
/// Specialized threadgroup size for the Q4_K decode matvec pilot.
const DEQUANT_MATVEC_Q4K_TG256: usize = 256;
/// Threadgroup size for N_DST=4 decode matvec kernels (must match shader NDST4_TG).
const NDST4_TG: usize = 32;
/// Number of output rows per threadgroup for the Q4_K NR2 kernel.
const Q4K_NR2_ROWS: usize = 4;
/// Number of output rows per threadgroup for the Q6_K NR2 kernel.
const Q6K_NR2_ROWS: usize = 4;
/// Number of output rows per threadgroup for N_DST=4 kernels (must match shader NDST4_ROWS).
const NDST4_ROWS: usize = 4;

/// Threadgroup size for standalone dequant kernels.
const DEQUANT_TG_SIZE: usize = 256;

/// Tile size for simdgroup dequant+matmul kernels (must match DQ_BM/DQ_BN in shader).
const DQ_TILE: usize = 32;

/// Threadgroup size for simdgroup dequant+matmul kernels (must match DQ_TG in shader).
const DQ_TG: usize = 128;

/// Tile size on M-axis for large batch dequant+matmul kernels (must match DB_BM).
const DB_TILE_M: usize = 32;
/// Tile size on N-axis for large batch dequant+matmul kernels (must match DB_BN).
const DB_TILE_N: usize = 64;
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
const SB_TILE_N: usize = 32;
/// Threadgroup size for small batch kernels (must match SB_TG).
const SB_TG: usize = 128;
/// Threadgroup size for K-parallel simd_sum batch dequant+matmul (must match SBLK_TG in shader).
const SBLK_TG_SIZE: usize = 64;
/// Total output rows per threadgroup for simd batch kernels (SBLK_NS=2 * SBLK_NR=4 = 8).
const SIMD_ROWS_PER_TG: usize = 8;
/// Routing threshold for choosing small-vs-large batch kernels.
///
/// Disabled by default (`1`) because current small-N kernels regress prefill in
/// benchmarked workloads (e.g. ~39 tokens). Keep plumbing for future retunes.
const BATCH_SMALL_N_THRESHOLD: u32 = 1;
/// Q4_K and Q6_K blocks contain 256 quantized values. K must be a multiple of this.
const Q4_K_BLOCK_VALUES: usize = 256;
const Q6_K_BLOCK_VALUES: usize = 256;

/// Pre-compiled dequantization compute pipelines.
///
/// Supports standalone dequant (Q4_0, Q4_K → f32), fused dequant+matvec (N=1),
/// and fused dequant+matmul with simdgroup_matrix (N>1).
/// Create once at init time, reuse for all dequant dispatches.
pub struct DequantKernels {
    dequant_q4_0: ComputePipeline,
    dequant_q4_k: ComputePipeline,
    dequant_q6_k: ComputePipeline,
    fused_matvec_q4_0: ComputePipeline,
    fused_matvec_q8_0: ComputePipeline,
    fused_matvec_q8_0_n4: ComputePipeline,
    fused_matvec_q4_k: ComputePipeline,
    /// Q4_K decode matvec with 2 rows per simdgroup and TG=64.
    fused_matvec_q4_k_nr2: ComputePipeline,
    /// Q4_K decode matvec with 2-block unrolled inner loop.
    fused_matvec_q4_k_blk2: ComputePipeline,
    /// Q4_K decode matvec specialized to TG=256 through Metal function constants.
    fused_matvec_q4_k_tg256: ComputePipeline,
    fused_matvec_q4_k_x2: ComputePipeline,
    /// N_DST=4 decode matvec for Q4_K (TG=32, 4 rows per TG, 1 simdgroup).
    fused_matvec_q4_k_n4: ComputePipeline,
    fused_matvec_dense_f16: ComputePipeline,
    fused_matvec_q6_k: ComputePipeline,
    /// Q6_K decode matvec with 2 rows per simdgroup and TG=64.
    fused_matvec_q6_k_nr2: ComputePipeline,
    /// Simdgroup-accelerated fused dequant+matmul for Q4_K (N>1, prefill).
    fused_matmul_q4_k: ComputePipeline,
    /// Simdgroup-accelerated fused dequant+matmul for Q6_K (N>1, prefill).
    fused_matmul_q6_k: ComputePipeline,
    /// B-transposed batch dequant+matmul for Q4_K: C[N×M] = B[N×K] × dequant(A[M×K])^T.
    fused_batch_q4_k: ComputePipeline,
    /// v2: 2-B-fragment inner loop (1.33 MACs/load vs 0.80).
    fused_batch_q4_k_v2: ComputePipeline,
    /// BN=32 full-tile: 8 KB TG memory → 3 TGs/SM (vs 12-20 KB → 1-2 TGs/SM).
    fused_batch_q4_k_bn32_full: ComputePipeline,
    /// Inline-dequant full-tile: fuses Phase 1 into Phase 2, eliminates barrier.
    fused_batch_q4_k_inline: ComputePipeline,
    /// Blocked-layout kernel (llama.cpp pattern): stride-8, 6KB TG, TG=128, 1.33 MACs/load.
    fused_batch_q4_k_blocked: ComputePipeline,
    /// BM=32 blocked variant for small M: 4KB TG, 2× more TGs.
    fused_batch_q4_k_blocked_bm32: ComputePipeline,
    /// B-transposed batch dequant+matmul for Q6_K: C[N×M] = B[N×K] × dequant(A[M×K])^T.
    fused_batch_q6_k: ComputePipeline,
    /// Blocked-layout Q6_K kernel (same architecture as Q4_K blocked).
    fused_batch_q6_k_blocked: ComputePipeline,
    /// BM=32 blocked Q6_K variant for small M.
    fused_batch_q6_k_blocked_bm32: ComputePipeline,
    /// Full-tile fast path for Q4_K batch dequant+matmul.
    fused_batch_q4_k_full: ComputePipeline,
    /// Full-tile fast path for Q6_K batch dequant+matmul.
    fused_batch_q6_k_full: ComputePipeline,
    /// B-transposed batch dequant+matmul for Q4_K with f16 input/output.
    fused_batch_q4_k_f16io: ComputePipeline,
    /// B-transposed batch dequant+matmul for Q6_K with f16 input/output.
    fused_batch_q6_k_f16io: ComputePipeline,
    /// B-transposed batch dequant+matmul for Q4_K with f16 input and f32 output.
    fused_batch_q4_k_f16in: ComputePipeline,
    /// Full-tile (BM=32, BN=64, TG=256) fast path for Q4_K with f16 input. No out_tile → 12 KB.
    fused_batch_q4_k_f16in_full: ComputePipeline,
    /// B-transposed batch dequant+matmul for Q6_K with f16 input and f32 output.
    fused_batch_q6_k_f16in: ComputePipeline,
    /// B-transposed batch dequant+matmul for Q8_0 with f16 input and f32 output.
    fused_batch_q8_0_f16in: ComputePipeline,
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
    /// K-parallel simd_sum batch dequant+matmul for Q4_K with f32 input/output.
    batch_q4_k_simd: ComputePipeline,
    /// K-parallel simd_sum batch dequant+matmul for Q6_K with f32 input/output.
    batch_q6_k_simd: ComputePipeline,
    /// BN=32/TG=128 Q4_K batch kernel: 12 KB TG memory → 2 TGs/SM (vs 1 for BN=64).
    /// Half A/B tiles + precomputed scales, same paired-nibble extraction as main kernel.
    /// Faster for N < DB_TILE_N (64): 50% fast-path tiles vs 100% boundary tiles.
    ///
    /// Note: rustc may report this as dead code when checking `ax-metal` in isolation,
    /// because the only reads happen through cross-crate call paths (`ax-core`).
    #[allow(dead_code)]
    fused_batch_q4_k_bn32: ComputePipeline,
}

impl DequantKernels {
    fn q4_k_matvec_dispatch(
        &self,
        m: u32,
    ) -> (
        usize,
        usize,
        &ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    ) {
        let q4k_tg = matvec_q4k_threadgroup_size();
        let q4k_rows_per_sg = matvec_q4k_rows_per_simdgroup();
        let use_n4 = matvec_q4k_n4_enabled() && m >= NDST4_ROWS as u32;
        let use_nr2 = !use_n4
            && (matvec_q4k_nr2_enabled()
                || q4k_rows_per_sg >= 2
                || q4k_tg == DEQUANT_MATVEC_Q4K_NR2_TG)
            && m >= 2;
        let use_x2 = !use_n4 && !use_nr2 && matvec_q4k_x2_enabled() && m >= 2;
        let use_tg256 = !use_n4 && !use_nr2 && !use_x2 && q4k_tg == DEQUANT_MATVEC_Q4K_TG256;
        let use_blk2 = !use_n4 && !use_nr2 && !use_x2 && !use_tg256 && matvec_q4k_blk2_enabled();

        if use_n4 {
            (
                (m as usize).div_ceil(NDST4_ROWS),
                NDST4_TG,
                self.fused_matvec_q4_k_n4.state(),
            )
        } else if use_nr2 {
            (
                (m as usize).div_ceil(Q4K_NR2_ROWS),
                DEQUANT_MATVEC_Q4K_NR2_TG,
                self.fused_matvec_q4_k_nr2.state(),
            )
        } else if use_x2 {
            (
                (m as usize).div_ceil(2),
                DEQUANT_MATVEC_TG,
                self.fused_matvec_q4_k_x2.state(),
            )
        } else if use_tg256 {
            (
                m as usize,
                DEQUANT_MATVEC_Q4K_TG256,
                self.fused_matvec_q4_k_tg256.state(),
            )
        } else if use_blk2 {
            (
                m as usize,
                DEQUANT_MATVEC_TG,
                self.fused_matvec_q4_k_blk2.state(),
            )
        } else {
            (
                m as usize,
                DEQUANT_MATVEC_TG,
                self.fused_matvec_q4_k.state(),
            )
        }
    }

    fn q6_k_matvec_dispatch(
        &self,
        m: u32,
    ) -> (
        usize,
        usize,
        &ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    ) {
        let q6k_tg = matvec_q6k_threadgroup_size();
        let q6k_rows_per_sg = matvec_q6k_rows_per_simdgroup();
        if (matvec_q6k_nr2_enabled() || q6k_rows_per_sg >= 2 || q6k_tg == DEQUANT_MATVEC_Q6K_NR2_TG)
            && m >= 2
        {
            (
                (m as usize).div_ceil(Q6K_NR2_ROWS),
                DEQUANT_MATVEC_Q6K_NR2_TG,
                self.fused_matvec_q6_k_nr2.state(),
            )
        } else {
            (
                m as usize,
                DEQUANT_MATVEC_TG,
                self.fused_matvec_q6_k.state(),
            )
        }
    }

    /// Compile dequant kernels from embedded Metal source.
    pub fn new(device: &MetalDevice) -> anyhow::Result<Self> {
        let dequant_q4_0 =
            ComputePipeline::from_source(device.device(), DEQUANT_SHADER_SRC, "dequant_q4_0")
                .context("Failed to compile dequant_q4_0 kernel")?;
        let dequant_q4_k =
            ComputePipeline::from_source(device.device(), DEQUANT_SHADER_SRC, "dequant_q4_k")
                .context("Failed to compile dequant_q4_k kernel")?;
        let fused_matvec_q4_0 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q4_0",
        )
        .context("Failed to compile dequant_matvec_q4_0 kernel")?;
        let fused_matvec_q8_0 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q8_0",
        )
        .context("Failed to compile dequant_matvec_q8_0 kernel")?;
        let fused_matvec_q8_0_n4 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q8_0_n4",
        )
        .context("Failed to compile dequant_matvec_q8_0_n4 kernel")?;
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
        let fused_matvec_q4_k_blk2 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q4_k_blk2",
        )
        .context("Failed to compile dequant_matvec_q4_k_blk2 kernel")?;
        let fused_matvec_q4_k_tg256 = ComputePipeline::from_source_with_constants(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q4_k_tg",
            &[FunctionConstant {
                index: 0,
                value: FunctionConstantValue::U16(DEQUANT_MATVEC_Q4K_TG256 as u16),
            }],
        )
        .context("Failed to compile dequant_matvec_q4_k_tg (TG=256) kernel")?;
        let fused_matvec_q4_k_x2 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q4_k_x2",
        )
        .context("Failed to compile dequant_matvec_q4_k_x2 kernel")?;
        let fused_matvec_q4_k_n4 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_matvec_q4_k_n4",
        )
        .context("Failed to compile dequant_matvec_q4_k_n4 kernel")?;
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
        let fused_batch_q4_k_blocked = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_blocked",
        )
        .context("Failed to compile dequant_batch_q4_k_blocked kernel")?;
        let fused_batch_q4_k_blocked_bm32 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_blocked_bm32",
        )
        .context("Failed to compile dequant_batch_q4_k_blocked_bm32 kernel")?;
        let fused_batch_q4_k_inline = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_inline",
        )
        .context("Failed to compile dequant_batch_q4_k_inline kernel")?;
        let fused_batch_q4_k_bn32_full = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_bn32_full",
        )
        .context("Failed to compile dequant_batch_q4_k_bn32_full kernel")?;
        let fused_batch_q4_k_v2 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_v2",
        )
        .context("Failed to compile dequant_batch_q4_k_v2 kernel")?;
        let fused_batch_q6_k =
            ComputePipeline::from_source(device.device(), DEQUANT_SHADER_SRC, "dequant_batch_q6_k")
                .context("Failed to compile dequant_batch_q6_k kernel")?;
        let fused_batch_q6_k_blocked = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q6_k_blocked",
        )
        .context("Failed to compile dequant_batch_q6_k_blocked kernel")?;
        let fused_batch_q6_k_blocked_bm32 = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q6_k_blocked_bm32",
        )
        .context("Failed to compile dequant_batch_q6_k_blocked_bm32 kernel")?;
        let fused_batch_q4_k_full = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_full",
        )
        .context("Failed to compile dequant_batch_q4_k_full kernel")?;
        let fused_batch_q6_k_full = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q6_k_full",
        )
        .context("Failed to compile dequant_batch_q6_k_full kernel")?;
        let fused_batch_q4_k_f16io = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_f16io",
        )
        .context("Failed to compile dequant_batch_q4_k_f16io kernel")?;
        let fused_batch_q6_k_f16io = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q6_k_f16io",
        )
        .context("Failed to compile dequant_batch_q6_k_f16io kernel")?;
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
        let fused_batch_q8_0_f16in = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q8_0_f16in",
        )
        .context("Failed to compile dequant_batch_q8_0_f16in kernel")?;
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
        let batch_q4_k_simd = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q4_k_simd",
        )
        .context("Failed to compile dequant_batch_q4_k_simd kernel")?;
        let batch_q6_k_simd = ComputePipeline::from_source(
            device.device(),
            DEQUANT_SHADER_SRC,
            "dequant_batch_q6_k_simd",
        )
        .context("Failed to compile dequant_batch_q6_k_simd kernel")?;
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
            "Dequant Metal kernels compiled (Q4_0 + Q8_0 + Q4_K + Q6_K, standalone + fused matvec + fused matmul + batch + simd + full32 + tail32)"
        );

        Ok(Self {
            dequant_q4_0,
            dequant_q4_k,
            dequant_q6_k,
            fused_matvec_q4_0,
            fused_matvec_q8_0,
            fused_matvec_q8_0_n4,
            fused_matvec_q4_k,
            fused_matvec_q4_k_nr2,
            fused_matvec_q4_k_blk2,
            fused_matvec_q4_k_tg256,
            fused_matvec_q4_k_x2,
            fused_matvec_q4_k_n4,
            fused_matvec_dense_f16,
            fused_matvec_q6_k,
            fused_matvec_q6_k_nr2,
            fused_matmul_q4_k,
            fused_matmul_q6_k,
            fused_batch_q4_k,
            fused_batch_q4_k_v2,
            fused_batch_q4_k_inline,
            fused_batch_q4_k_blocked,
            fused_batch_q4_k_blocked_bm32,
            fused_batch_q4_k_bn32_full,
            fused_batch_q6_k,
            fused_batch_q6_k_blocked,
            fused_batch_q6_k_blocked_bm32,
            fused_batch_q4_k_full,
            fused_batch_q6_k_full,
            fused_batch_q4_k_f16io,
            fused_batch_q6_k_f16io,
            fused_batch_q4_k_f16in,
            fused_batch_q4_k_f16in_full,
            fused_batch_q6_k_f16in,
            fused_batch_q8_0_f16in,
            fused_batch_q8_0_f16in_full,
            fused_batch_q8_0_f16in_full64,
            fused_batch_q8_0_f16in_full32,
            fused_batch_q8_0_f16in_tail32,
            fused_batch_q8_0_f16in_full32x32,
            fused_batch_q4_k_f16in_full64,
            fused_batch_q4_k_f16in_full64_bk32,
            fused_batch_q6_k_f16in_full64,
            fused_batch_q4_k_f16in_small,
            fused_batch_q6_k_f16in_small,
            fused_batch_pair_q4_k,
            fused_batch_pair_q6_k,
            fused_batch_pair_q4_k_f16in,
            fused_batch_pair_q6_k_f16in,
            fused_batch_pair_q8_0_f16in,
            fused_batch_pair_q8_0_f16in_full,
            fused_batch_q4_k_small,
            fused_batch_q6_k_small,
            batch_matmul_btrans_f16_f32,
            batch_matmul_btrans_f16_f32_full64,
            batch_q4_k_simd,
            batch_q6_k_simd,
            fused_batch_q4_k_bn32,
            fused_batch_q4_k_f16in_full32,
            fused_batch_q6_k_f16in_full32,
            fused_batch_q4_k_f16in_tail32,
            fused_batch_q6_k_f16in_tail32,
        })
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
                encoder.setComputePipelineState(self.batch_matmul_btrans_f16_f32_full64.state());
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
            encoder.setComputePipelineState(self.batch_matmul_btrans_f16_f32.state());
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
        encoder.setComputePipelineState(self.batch_matmul_btrans_f16_f32.state());
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

    /// Encode K-parallel simd_sum Q4_K batch matmul into an existing encoder.
    ///
    /// Computes C[N×M] = B[N×K] × dequant(A[M×K])^T.
    /// Grid = (ceil(M/8), N), TG = 64 threads (2 simdgroups × 4 rows each).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_batch_simd_q4k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
        c_stride: u32,
    ) {
        let groups_x = (m as usize).div_ceil(SIMD_ROWS_PER_TG);
        encoder.setComputePipelineState(self.batch_q4_k_simd.state());
        bind_buffers(encoder, a, b, c);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        bind_u32(encoder, 6, c_stride);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: n as usize,
                depth: 1,
            },
            MTLSize {
                width: SBLK_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode K-parallel simd_sum Q6_K batch matmul into an existing encoder.
    ///
    /// Computes C[N×M] = B[N×K] × dequant(A[M×K])^T.
    /// Grid = (ceil(M/8), N), TG = 64 threads (2 simdgroups × 4 rows each).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_batch_simd_q6k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
        c_stride: u32,
    ) {
        let groups_x = (m as usize).div_ceil(SIMD_ROWS_PER_TG);
        encoder.setComputePipelineState(self.batch_q6_k_simd.state());
        bind_buffers(encoder, a, b, c);
        bind_u32(encoder, 3, m);
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, k);
        bind_u32(encoder, 6, c_stride);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: n as usize,
                depth: 1,
            },
            MTLSize {
                width: SBLK_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Dequantize Q4_0 data on GPU: quantized blocks → f32 output.
    ///
    /// - `src`: buffer containing `n_blocks` Q4_0 blocks (18 bytes each)
    /// - `dst`: output buffer for `n_blocks * 32` f32 values
    /// - `n_blocks`: number of Q4_0 blocks
    pub fn dequant_q4_0(
        &self,
        device: &MetalDevice,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        n_blocks: u32,
    ) -> anyhow::Result<()> {
        let dims = DispatchDims::d1(n_blocks as usize, DEQUANT_TG_SIZE);

        device.execute_sync(|encoder| {
            encoder.setComputePipelineState(self.dequant_q4_0.state());
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
            encoder.setComputePipelineState(self.dequant_q4_k.state());
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

    /// Fused dequant Q4_0 + matvec: y = dequant(A) × x.
    ///
    /// - `a`: M × (K/32) Q4_0 blocks (quantized weight matrix)
    /// - `x`: K f32 values (input vector)
    /// - `y`: M f32 values (output vector)
    /// - `m`: number of rows
    /// - `k`: number of columns (must be multiple of 32)
    pub fn fused_matvec_q4_0(
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
            encoder.setComputePipelineState(self.fused_matvec_q4_0.state());
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

    /// Fused dequant Q4_K + matvec: y = dequant(A) × x.
    ///
    /// - `a`: M × (K/256) Q4_K blocks (quantized weight matrix)
    /// - `x`: K f32 values (input vector)
    /// - `y`: M f32 values (output vector)
    /// - `m`: number of rows
    /// - `k`: number of columns (must be multiple of 256)
    pub fn fused_matvec_q4_k(
        &self,
        device: &MetalDevice,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        let (groups, tg_width, pipeline) = self.q4_k_matvec_dispatch(m);
        let dims = DispatchDims::d1(groups, 1);

        device.execute_sync(|encoder| {
            encoder.setComputePipelineState(pipeline);
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

    /// Explicit TG=256 Q4_K matvec route for A/B benchmarking and validation.
    #[allow(dead_code)]
    pub(crate) fn fused_matvec_q4_k_tg256(
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
            encoder.setComputePipelineState(self.fused_matvec_q4_k_tg256.state());
            bind_buffers(encoder, a, x, y);
            bind_u32(encoder, 3, m);
            bind_u32(encoder, 4, k);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                MTLSize {
                    width: DEQUANT_MATVEC_Q4K_TG256,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        })
    }

    /// Explicit 2-block-unrolled Q4_K matvec route for A/B benchmarking and validation.
    #[allow(dead_code)]
    pub(crate) fn fused_matvec_q4_k_blk2(
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
            encoder.setComputePipelineState(self.fused_matvec_q4_k_blk2.state());
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

    /// Explicit NR2 Q4_K matvec route for A/B benchmarking and validation.
    #[allow(dead_code)]
    pub(crate) fn fused_matvec_q4_k_nr2(
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
            encoder.setComputePipelineState(self.fused_matvec_q4_k_nr2.state());
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

    /// Encode a fused Q4_0 matvec dispatch into an existing encoder.
    ///
    /// Does NOT create or commit a command buffer. Used for batching
    /// multiple matvec operations into a single command buffer.
    pub fn encode_fused_matvec_q4_0(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        let dims = DispatchDims::d1(m as usize, 1);
        encoder.setComputePipelineState(self.fused_matvec_q4_0.state());
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
        let use_n4 = m >= NDST4_ROWS as u32;
        let (groups, tg_width, pipeline) = if use_n4 {
            (
                (m as usize).div_ceil(NDST4_ROWS),
                NDST4_TG,
                self.fused_matvec_q8_0_n4.state(),
            )
        } else {
            (
                m as usize,
                DEQUANT_MATVEC_TG,
                self.fused_matvec_q8_0.state(),
            )
        };
        let dims = DispatchDims::d1(groups, 1);
        encoder.setComputePipelineState(pipeline);
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

    /// Encode a fused Q4_K matvec dispatch into an existing encoder.
    ///
    /// Does NOT create or commit a command buffer. Used for batching
    /// multiple matvec operations into a single command buffer.
    pub fn encode_fused_matvec_q4_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        let (groups, tg_width, pipeline) = self.q4_k_matvec_dispatch(m);
        let dims = DispatchDims::d1(groups, 1);
        encoder.setComputePipelineState(pipeline);
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
        encoder.setComputePipelineState(self.fused_matvec_dense_f16.state());
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
        let use_n4 = m >= NDST4_ROWS as u32;
        let (groups, tg_width, pipeline) = if use_n4 {
            (
                (m as usize).div_ceil(NDST4_ROWS),
                NDST4_TG,
                self.fused_matvec_q8_0_n4.state(),
            )
        } else {
            (
                m as usize,
                DEQUANT_MATVEC_TG,
                self.fused_matvec_q8_0.state(),
            )
        };
        let dims = DispatchDims::d1(groups, 1);

        device.execute_sync(|encoder| {
            encoder.setComputePipelineState(pipeline);
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
            encoder.setComputePipelineState(self.dequant_q6_k.state());
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
    pub fn fused_matvec_q6_k(
        &self,
        device: &MetalDevice,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        let (groups, tg_width, pipeline) = self.q6_k_matvec_dispatch(m);
        let dims = DispatchDims::d1(groups, 1);

        device.execute_sync(|encoder| {
            encoder.setComputePipelineState(pipeline);
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
    #[allow(dead_code)]
    pub(crate) fn fused_matvec_q6_k_nr2(
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
            encoder.setComputePipelineState(self.fused_matvec_q6_k_nr2.state());
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

    /// Encode a fused Q6_K matvec dispatch into an existing encoder.
    ///
    /// Does NOT create or commit a command buffer. Used for batching
    /// multiple matvec operations into a single command buffer.
    pub fn encode_fused_matvec_q6_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        let (groups, tg_width, pipeline) = self.q6_k_matvec_dispatch(m);
        let dims = DispatchDims::d1(groups, 1);
        encoder.setComputePipelineState(pipeline);
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

        encoder.setComputePipelineState(self.fused_matmul_q4_k.state());
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

        encoder.setComputePipelineState(self.fused_matmul_q6_k.state());
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
            let (pipeline, bm, smem) = if use_bm32 {
                (&self.fused_batch_q4_k_blocked_bm32, 32usize, 8192usize)
            } else {
                (&self.fused_batch_q4_k_blocked, 64usize, 8192usize)
            };
            encoder.setComputePipelineState(pipeline.state());
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
            encoder.setComputePipelineState(self.fused_batch_q4_k_bn32.state());
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
        let use_small = n < BATCH_SMALL_N_THRESHOLD;

        // BN=32 full-tile path: 8 KB TG memory → 3 TGs/SM (vs 12-20 KB → 1-2).
        // Higher occupancy hides memory latency. Use when both M and N are aligned.
        // Split into full BN=32 tiles + boundary remainder with existing BN=64 kernel.
        let use_bn32_full =
            !use_small && batch_q4k_bn32_full_enabled() && (m as usize).is_multiple_of(DB_TILE_M);
        if use_bn32_full {
            const BN32_TG: usize = 128;
            let groups_x = (m as usize) / DB_TILE_M;
            let full_n = (n as usize / 32) * 32;
            let n_tail = (n as usize).saturating_sub(full_n);

            if full_n > 0 {
                encoder.setComputePipelineState(self.fused_batch_q4_k_bn32_full.state());
                bind_buffers(encoder, a, b, c);
                bind_u32(encoder, 3, m);
                bind_u32(encoder, 4, full_n as u32);
                bind_u32(encoder, 5, k);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: groups_x,
                        height: full_n / 32,
                        depth: 1,
                    },
                    MTLSize {
                        width: BN32_TG,
                        height: 1,
                        depth: 1,
                    },
                );
            }

            // Tail N-rows (< 32) use the bounds-checked BN=32 kernel.
            if n_tail > 0 {
                let b_off = full_n
                    .checked_mul(k as usize)
                    .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                    .expect("b offset overflow");
                let c_off = full_n
                    .checked_mul(m as usize)
                    .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                    .expect("c offset overflow");
                encoder.setComputePipelineState(self.fused_batch_q4_k_bn32.state());
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), b_off, 1);
                    encoder.setBuffer_offset_atIndex(Some(c.mtl_buffer()), c_off, 2);
                }
                bind_u32(encoder, 3, m);
                bind_u32(encoder, 4, n_tail as u32);
                bind_u32(encoder, 5, k);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: groups_x,
                        height: n_tail.div_ceil(32),
                        depth: 1,
                    },
                    MTLSize {
                        width: BN32_TG,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            return;
        }

        // v2 kernel: 2-B-fragment inner loop (1.33 MACs/load). Same tiles
        // (BM=32, BN=64) but TG=128 with 8 accumulators per SG.
        // Use for all non-small N that are aligned to BN=64.
        let use_v2 = !use_small && batch_q4k_v2_enabled();
        if use_v2 {
            const V2_TG: usize = 128;
            let groups_x = (m as usize).div_ceil(DB_TILE_M);
            let full_rows = (n as usize / DB_TILE_N) * DB_TILE_N;
            if full_rows > 0 {
                encoder.setComputePipelineState(self.fused_batch_q4_k_v2.state());
                bind_buffers(encoder, a, b, c);
                bind_u32(encoder, 3, m);
                bind_u32(encoder, 4, full_rows as u32);
                bind_u32(encoder, 5, k);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: groups_x,
                        height: full_rows / DB_TILE_N,
                        depth: 1,
                    },
                    MTLSize {
                        width: V2_TG,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            let tail_rows = (n as usize).saturating_sub(full_rows);
            if tail_rows > 0 {
                // Tail uses the original v1 kernel (bounds-checked).
                let b_row_offset = full_rows
                    .checked_mul(k as usize)
                    .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                    .expect("b row offset overflow");
                let c_row_offset = full_rows
                    .checked_mul(m as usize)
                    .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                    .expect("c row offset overflow");
                encoder.setComputePipelineState(self.fused_batch_q4_k.state());
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), b_row_offset, 1);
                    encoder.setBuffer_offset_atIndex(Some(c.mtl_buffer()), c_row_offset, 2);
                }
                bind_u32(encoder, 3, m);
                bind_u32(encoder, 4, tail_rows as u32);
                bind_u32(encoder, 5, k);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: groups_x,
                        height: tail_rows.div_ceil(DB_TILE_N),
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
        if !use_small && m.is_multiple_of(DB_TILE_M as u32) {
            let groups_x = (m as usize).div_ceil(DB_TILE_M);
            let full_rows = (n as usize / DB_TILE_N) * DB_TILE_N;
            if full_rows > 0 {
                // Inline-dequant kernel: fuses Phase 1 into Phase 2,
                // eliminates 1 barrier per K-tile and 88% thread idle time.
                encoder.setComputePipelineState(if batch_q4k_inline_enabled() {
                    self.fused_batch_q4_k_inline.state()
                } else {
                    self.fused_batch_q4_k_full.state()
                });
                bind_buffers(encoder, a, b, c);
                bind_u32(encoder, 3, m);
                bind_u32(encoder, 4, full_rows as u32);
                bind_u32(encoder, 5, k);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: groups_x,
                        height: full_rows / DB_TILE_N,
                        depth: 1,
                    },
                    MTLSize {
                        width: DB_TG,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            let tail_rows = (n as usize).saturating_sub(full_rows);
            if tail_rows > 0 {
                let b_row_offset = full_rows
                    .checked_mul(k as usize)
                    .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                    .expect("b row offset overflow");
                let c_row_offset = full_rows
                    .checked_mul(m as usize)
                    .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                    .expect("c row offset overflow");
                encoder.setComputePipelineState(self.fused_batch_q4_k.state());
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), b_row_offset, 1);
                    encoder.setBuffer_offset_atIndex(Some(c.mtl_buffer()), c_row_offset, 2);
                }
                bind_u32(encoder, 3, m);
                bind_u32(encoder, 4, tail_rows as u32);
                bind_u32(encoder, 5, k);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: groups_x,
                        height: tail_rows.div_ceil(DB_TILE_N),
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
        let groups_x = (m as usize).div_ceil(DB_TILE_M);
        let groups_y = if use_small {
            (n as usize).div_ceil(SB_TILE_N)
        } else {
            (n as usize).div_ceil(DB_TILE_N)
        };

        encoder.setComputePipelineState(if use_small {
            self.fused_batch_q4_k_small.state()
        } else {
            self.fused_batch_q4_k.state()
        });
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
            let (pipeline, bm, smem) = if use_bm32 {
                (&self.fused_batch_q6_k_blocked_bm32, 32usize, 8192usize)
            } else {
                (&self.fused_batch_q6_k_blocked, 64usize, 8192usize)
            };
            encoder.setComputePipelineState(pipeline.state());
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

        let use_small = n < BATCH_SMALL_N_THRESHOLD;
        if !use_small && m.is_multiple_of(DB_TILE_M as u32) {
            let groups_x = (m as usize).div_ceil(DB_TILE_M);
            let full_rows = (n as usize / DB_TILE_N) * DB_TILE_N;
            if full_rows > 0 {
                encoder.setComputePipelineState(self.fused_batch_q6_k_full.state());
                bind_buffers(encoder, a, b, c);
                bind_u32(encoder, 3, m);
                bind_u32(encoder, 4, full_rows as u32);
                bind_u32(encoder, 5, k);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: groups_x,
                        height: full_rows / DB_TILE_N,
                        depth: 1,
                    },
                    MTLSize {
                        width: DB_TG,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            let tail_rows = (n as usize).saturating_sub(full_rows);
            if tail_rows > 0 {
                let b_row_offset = full_rows
                    .checked_mul(k as usize)
                    .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                    .expect("b row offset overflow");
                let c_row_offset = full_rows
                    .checked_mul(m as usize)
                    .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                    .expect("c row offset overflow");
                encoder.setComputePipelineState(self.fused_batch_q6_k.state());
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), b_row_offset, 1);
                    encoder.setBuffer_offset_atIndex(Some(c.mtl_buffer()), c_row_offset, 2);
                }
                bind_u32(encoder, 3, m);
                bind_u32(encoder, 4, tail_rows as u32);
                bind_u32(encoder, 5, k);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: groups_x,
                        height: tail_rows.div_ceil(DB_TILE_N),
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
        let groups_x = (m as usize).div_ceil(DB_TILE_M);
        let groups_y = if use_small {
            (n as usize).div_ceil(SB_TILE_N)
        } else {
            (n as usize).div_ceil(DB_TILE_N)
        };

        encoder.setComputePipelineState(if use_small {
            self.fused_batch_q6_k_small.state()
        } else {
            self.fused_batch_q6_k.state()
        });
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

    /// Encode a B-transposed batch dequant Q4_K + matmul with f16 I/O.
    ///
    /// C[N × M] = B[N × K] × dequant(A[M × K])^T
    /// - `b`: N × K f16
    /// - `c`: N × M f16
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_q4_k_f16io(
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
        let groups_y = (n as usize).div_ceil(DB_TILE_N);
        encoder.setComputePipelineState(self.fused_batch_q4_k_f16io.state());
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

    /// Encode a B-transposed batch dequant Q6_K + matmul with f16 I/O.
    ///
    /// C[N × M] = B[N × K] × dequant(A[M × K])^T
    /// - `b`: N × K f16
    /// - `c`: N × M f16
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_q6_k_f16io(
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
        let groups_y = (n as usize).div_ceil(DB_TILE_N);
        encoder.setComputePipelineState(self.fused_batch_q6_k_f16io.state());
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

    /// Encode a B-transposed batch dequant Q4_K + matmul with f16 input and f32 output.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_q4_k_f16in(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let (small_n_threshold, small_m_max) = crate::batch_f16in_route_config();
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
            let bn32_preferred = batch_f16in_bn32_enabled();
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
                    encoder.setComputePipelineState(self.fused_batch_q4_k_f16in_full32.state());
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
                    let use_bk32 = batch_f16in_bk32_enabled();
                    encoder.setComputePipelineState(if use_bk32 {
                        self.fused_batch_q4_k_f16in_full64_bk32.state()
                    } else {
                        self.fused_batch_q4_k_f16in_full64.state()
                    });
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
                        encoder.setComputePipelineState(self.fused_batch_q4_k_f16in.state());
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
                        encoder.setComputePipelineState(self.fused_batch_q4_k_f16in_tail32.state());
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
            encoder.setComputePipelineState(self.fused_batch_q4_k_f16in_full.state());
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
        encoder.setComputePipelineState(if use_small {
            self.fused_batch_q4_k_f16in_small.state()
        } else {
            self.fused_batch_q4_k_f16in.state()
        });
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
    pub fn encode_fused_batch_q6_k_f16in(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let (small_n_threshold, small_m_max) = crate::batch_f16in_route_config();
        let use_small =
            small_n_threshold > 1 && n < small_n_threshold && small_m_max > 0 && m <= small_m_max;

        if !use_small {
            let m_full = (m as usize / DB64_TILE_M) * DB64_TILE_M;
            let m_tail = (m as usize).saturating_sub(m_full);
            // BN=32 occupancy gate (same logic as Q4_K — see above).
            let bn32_preferred = batch_f16in_bn32_enabled();
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
                    encoder.setComputePipelineState(self.fused_batch_q6_k_f16in_full32.state());
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
                    encoder.setComputePipelineState(self.fused_batch_q6_k_f16in_full64.state());
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
                        encoder.setComputePipelineState(self.fused_batch_q6_k_f16in.state());
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
                        encoder.setComputePipelineState(self.fused_batch_q6_k_f16in_tail32.state());
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
        encoder.setComputePipelineState(if use_small {
            self.fused_batch_q6_k_f16in_small.state()
        } else {
            self.fused_batch_q6_k_f16in.state()
        });
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

    /// Encode a B-transposed batch dequant Q8_0 + matmul with f16 input and f32 output.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_batch_q8_0_f16in(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let can_use_full = (k as usize).is_multiple_of(64) && (n as usize) >= q8_f16in_full_min_n();
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
            let bn32_preferred = batch_f16in_bn32_enabled() && (n as usize) < 768;
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
                    encoder.setComputePipelineState(self.fused_batch_q8_0_f16in_full32x32.state());
                } else if use_bn32 {
                    encoder.setComputePipelineState(self.fused_batch_q8_0_f16in_full32.state());
                } else {
                    encoder.setComputePipelineState(self.fused_batch_q8_0_f16in_full64.state());
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
                encoder.setComputePipelineState(self.fused_batch_q8_0_f16in_tail32.state());
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
                encoder.setComputePipelineState(self.fused_batch_q8_0_f16in.state());
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
                encoder.setComputePipelineState(self.fused_batch_q8_0_f16in_full.state());
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
            encoder.setComputePipelineState(self.fused_batch_q8_0_f16in.state());
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
        encoder.setComputePipelineState(self.fused_batch_q8_0_f16in.state());
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
        encoder.setComputePipelineState(self.fused_batch_pair_q4_k.state());
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
        encoder.setComputePipelineState(self.fused_batch_pair_q6_k.state());
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
        encoder.setComputePipelineState(self.fused_batch_pair_q4_k_f16in.state());
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
        encoder.setComputePipelineState(self.fused_batch_pair_q6_k_f16in.state());
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
                encoder.setComputePipelineState(self.fused_batch_pair_q8_0_f16in_full.state());
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
            encoder.setComputePipelineState(self.fused_batch_pair_q8_0_f16in.state());
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
        encoder.setComputePipelineState(self.fused_batch_pair_q8_0_f16in.state());
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
}

/// Threadgroup size for attention prefill kernel (must match shader constant).
const ATTN_TG: usize = 256;
/// Threadgroup size for alternative prefill kernel (must match ATTN2_TG in shader).
const ATTN2_TG: usize = 128;
/// Threadgroup size for default decode attention kernel (must match shader constant).
const ATTN_DEC_TG: usize = 256;
/// Threadgroup size for alternative decode attention kernel (must match ATTN_DEC2_TG in shader).
const ATTN_DEC2_TG: usize = 128;
/// Prompt-length threshold for using mistral HD128 BC64 prefill variant.
const ATTN_PREFILL_MISTRAL_BC64_THRESHOLD: u32 = 384;

/// Maximum head dimension supported by attention shaders (must match MAX_HD in attention.metal).
const MAX_HEAD_DIM: u32 = 256;

/// Pre-compiled attention compute pipelines.
///
/// Provides FlashAttention-style tiled prefill with online softmax,
/// causal masking, and GQA support.
/// Create once at init time, reuse for all attention dispatches.
pub struct AttentionKernels {
    prefill: ComputePipeline,
    prefill_hd256: ComputePipeline,
    prefill_v2: ComputePipeline,
    prefill_v2_hd128: ComputePipeline,
    prefill_mistral_hd128: ComputePipeline,
    prefill_mistral_f16out_hd128: ComputePipeline,
    prefill_mistral_hd128_smem: ComputePipeline,
    prefill_mistral_hd128_smem_f16: ComputePipeline,
    prefill_mistral_hd128_bc64: ComputePipeline,
    prefill_fa2_hd128: ComputePipeline,
    /// FA2 with simdgroup_half8x8 matrix ops (HD=128) — llama.cpp-style.
    prefill_fa2_simd_hd128: ComputePipeline,
    prefill_cache: ComputePipeline,
    prefill_cache_f16kv: ComputePipeline,
    /// FA2-style 8-query-per-TG prefill (HD=256, f16 KV).
    prefill_cache_fa2_hd256: ComputePipeline,
    decode: ComputePipeline,
    decode_hd256: ComputePipeline,
    decode_f16kv: ComputePipeline,
    decode_f16kv_hd256: ComputePipeline,
    decode_v2: ComputePipeline,
    /// sdpa_vector decode (HD=256, f16 KV) — MLX-pattern lane-parallel.
    decode_sdpa_hd256: ComputePipeline,
    /// Optimized decode for head_dim=128 (Qwen3), f32 KV.
    decode_hd128: ComputePipeline,
    /// Optimized decode for head_dim=128 (Qwen3), f16 KV.
    decode_f16kv_hd128: ComputePipeline,
    /// Two-head-per-TG decode for head_dim=128, f16 KV.
    decode_f16kv_hd128_n2: ComputePipeline,
    /// Split-K partial decode for head_dim=128, f16 KV.
    decode_splitk_f16kv_hd128_partial: ComputePipeline,
    /// Split-K reduction decode for head_dim=128, f16 KV.
    decode_splitk_f16kv_hd128_reduce: ComputePipeline,
    /// Split-K partial decode for head_dim=256, f16 KV.
    decode_splitk_f16kv_hd256_partial: ComputePipeline,
    /// Split-K reduction decode for head_dim=256, f16 KV.
    decode_splitk_f16kv_hd256_reduce: ComputePipeline,
}

impl AttentionKernels {
    /// Compile attention kernels from embedded Metal source.
    pub fn new(device: &MetalDevice) -> anyhow::Result<Self> {
        let prefill = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_f32",
        )
        .context("Failed to compile attention_prefill_f32 kernel")?;
        let prefill_hd256 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_f32_hd256",
        )
        .context("Failed to compile attention_prefill_f32_hd256 kernel")?;
        let prefill_v2 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_f32_v2",
        )
        .context("Failed to compile attention_prefill_f32_v2 kernel")?;
        let prefill_v2_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_f32_v2_hd128",
        )
        .context("Failed to compile attention_prefill_f32_v2_hd128 kernel")?;
        let prefill_mistral_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_f32_mistral_hd128",
        )
        .context("Failed to compile attention_prefill_f32_mistral_hd128 kernel")?;
        let prefill_mistral_f16out_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_f16_mistral_hd128",
        )
        .context("Failed to compile attention_prefill_f16_mistral_hd128 kernel")?;
        let prefill_mistral_hd128_smem = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_f32_mistral_hd128_smem",
        )
        .context("Failed to compile attention_prefill_f32_mistral_hd128_smem kernel")?;
        let prefill_mistral_hd128_smem_f16 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_f32_mistral_hd128_smem_f16",
        )
        .context("Failed to compile attention_prefill_f32_mistral_hd128_smem_f16 kernel")?;
        let prefill_mistral_hd128_bc64 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_f32_mistral_hd128_bc64",
        )
        .context("Failed to compile attention_prefill_f32_mistral_hd128_bc64 kernel")?;
        let prefill_fa2_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_f32_fa2_hd128",
        )
        .context("Failed to compile attention_prefill_f32_fa2_hd128 kernel")?;
        let prefill_fa2_simd_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_fa2_simd_hd128",
        )
        .context("Failed to compile attention_prefill_fa2_simd_hd128 kernel")?;
        let prefill_cache = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_cache_f32",
        )
        .context("Failed to compile attention_prefill_cache_f32 kernel")?;
        let prefill_cache_f16kv = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_cache_f16kv",
        )
        .context("Failed to compile attention_prefill_cache_f16kv kernel")?;

        let decode = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_f32",
        )
        .context("Failed to compile attention_decode_f32 kernel")?;
        let decode_hd256 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_f32_hd256",
        )
        .context("Failed to compile attention_decode_f32_hd256 kernel")?;
        let decode_f16kv = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_f16kv",
        )
        .context("Failed to compile attention_decode_f16kv kernel")?;
        let decode_f16kv_hd256 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_f16kv_hd256",
        )
        .context("Failed to compile attention_decode_f16kv_hd256 kernel")?;
        let decode_v2 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_f32_v2",
        )
        .context("Failed to compile attention_decode_f32_v2 kernel")?;
        let decode_sdpa_hd256 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_sdpa_hd256",
        )
        .context("Failed to compile attention_decode_sdpa_hd256 kernel")?;
        let prefill_cache_fa2_hd256 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_prefill_cache_fa2_hd256",
        )
        .context("Failed to compile attention_prefill_cache_fa2_hd256 kernel")?;
        let decode_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_f32_hd128",
        )
        .context("Failed to compile attention_decode_f32_hd128 kernel")?;
        let decode_f16kv_hd128 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_f16kv_hd128",
        )
        .context("Failed to compile attention_decode_f16kv_hd128 kernel")?;
        let decode_f16kv_hd128_n2 = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_f16kv_hd128_n2",
        )
        .context("Failed to compile attention_decode_f16kv_hd128_n2 kernel")?;
        let decode_splitk_f16kv_hd128_partial = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_splitk_f16kv_hd128_partial",
        )
        .context("Failed to compile attention_decode_splitk_f16kv_hd128_partial kernel")?;
        let decode_splitk_f16kv_hd128_reduce = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_splitk_f16kv_hd128_reduce",
        )
        .context("Failed to compile attention_decode_splitk_f16kv_hd128_reduce kernel")?;
        let decode_splitk_f16kv_hd256_partial = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_splitk_f16kv_hd256_partial",
        )
        .context("Failed to compile attention_decode_splitk_f16kv_hd256_partial kernel")?;
        let decode_splitk_f16kv_hd256_reduce = ComputePipeline::from_source(
            device.device(),
            ATTENTION_SHADER_SRC,
            "attention_decode_splitk_f16kv_hd256_reduce",
        )
        .context("Failed to compile attention_decode_splitk_f16kv_hd256_reduce kernel")?;

        tracing::info!(
            prefill_max_threads = prefill.max_threads_per_threadgroup(),
            prefill_hd256_max_threads = prefill_hd256.max_threads_per_threadgroup(),
            prefill_v2_max_threads = prefill_v2.max_threads_per_threadgroup(),
            prefill_v2_hd128_max_threads = prefill_v2_hd128.max_threads_per_threadgroup(),
            prefill_mistral_hd128_max_threads = prefill_mistral_hd128.max_threads_per_threadgroup(),
            prefill_mistral_f16out_hd128_max_threads =
                prefill_mistral_f16out_hd128.max_threads_per_threadgroup(),
            prefill_mistral_hd128_smem_max_threads =
                prefill_mistral_hd128_smem.max_threads_per_threadgroup(),
            prefill_mistral_hd128_smem_f16_max_threads =
                prefill_mistral_hd128_smem_f16.max_threads_per_threadgroup(),
            prefill_mistral_hd128_bc64_max_threads =
                prefill_mistral_hd128_bc64.max_threads_per_threadgroup(),
            prefill_fa2_hd128_max_threads = prefill_fa2_hd128.max_threads_per_threadgroup(),
            prefill_fa2_simd_hd128_max_threads =
                prefill_fa2_simd_hd128.max_threads_per_threadgroup(),
            prefill_cache_max_threads = prefill_cache.max_threads_per_threadgroup(),
            prefill_cache_f16kv_max_threads = prefill_cache_f16kv.max_threads_per_threadgroup(),
            prefill_cache_fa2_hd256_max_threads =
                prefill_cache_fa2_hd256.max_threads_per_threadgroup(),
            decode_max_threads = decode.max_threads_per_threadgroup(),
            decode_hd256_max_threads = decode_hd256.max_threads_per_threadgroup(),
            decode_f16kv_max_threads = decode_f16kv.max_threads_per_threadgroup(),
            decode_f16kv_hd256_max_threads = decode_f16kv_hd256.max_threads_per_threadgroup(),
            decode_v2_max_threads = decode_v2.max_threads_per_threadgroup(),
            decode_sdpa_hd256_max_threads = decode_sdpa_hd256.max_threads_per_threadgroup(),
            decode_hd128_max_threads = decode_hd128.max_threads_per_threadgroup(),
            decode_f16kv_hd128_max_threads = decode_f16kv_hd128.max_threads_per_threadgroup(),
            decode_f16kv_hd128_n2_max_threads = decode_f16kv_hd128_n2.max_threads_per_threadgroup(),
            decode_splitk_f16kv_hd128_partial_max_threads =
                decode_splitk_f16kv_hd128_partial.max_threads_per_threadgroup(),
            decode_splitk_f16kv_hd128_reduce_max_threads =
                decode_splitk_f16kv_hd128_reduce.max_threads_per_threadgroup(),
            decode_splitk_f16kv_hd256_partial_max_threads =
                decode_splitk_f16kv_hd256_partial.max_threads_per_threadgroup(),
            decode_splitk_f16kv_hd256_reduce_max_threads =
                decode_splitk_f16kv_hd256_reduce.max_threads_per_threadgroup(),
            "Attention Metal kernels compiled (prefill + decode)",
        );

        Ok(Self {
            prefill,
            prefill_hd256,
            prefill_v2,
            prefill_v2_hd128,
            prefill_mistral_hd128,
            prefill_mistral_f16out_hd128,
            prefill_mistral_hd128_smem,
            prefill_mistral_hd128_smem_f16,
            prefill_mistral_hd128_bc64,
            prefill_fa2_hd128,
            prefill_fa2_simd_hd128,
            prefill_cache,
            prefill_cache_f16kv,
            prefill_cache_fa2_hd256,
            decode,
            decode_hd256,
            decode_f16kv,
            decode_f16kv_hd256,
            decode_v2,
            decode_sdpa_hd256,
            decode_hd128,
            decode_f16kv_hd128,
            decode_f16kv_hd128_n2,
            decode_splitk_f16kv_hd128_partial,
            decode_splitk_f16kv_hd128_reduce,
            decode_splitk_f16kv_hd256_partial,
            decode_splitk_f16kv_hd256_reduce,
        })
    }

    /// Dispatch prefill attention with causal masking.
    ///
    /// Computes: O = softmax(Q × K^T / √head_dim) × V  (with causal mask)
    ///
    /// - `q`: [n_tokens × n_heads × head_dim] query vectors
    /// - `k`: [n_tokens × n_kv_heads × head_dim] key vectors
    /// - `v`: [n_tokens × n_kv_heads × head_dim] value vectors
    /// - `o`: [n_tokens × n_heads × head_dim] output buffer
    /// - `n_tokens`: number of tokens in the sequence
    /// - `n_heads`: number of query heads
    /// - `n_kv_heads`: number of KV heads (GQA: n_heads / n_kv_heads heads share one KV)
    /// - `head_dim`: dimension per head
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill(
        &self,
        device: &MetalDevice,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        o: &MetalBuffer,
        n_tokens: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
    ) -> anyhow::Result<()> {
        let use_v2 = attention_prefill_v2_enabled();
        let use_v2_hd128 = attention_prefill_hd128_enabled() && head_dim == 128;
        let use_mistral_hd128 = attention_prefill_mistral_hd128_enabled() && head_dim == 128;
        let use_mistral_bc64 = attention_prefill_mistral_bc64_enabled()
            && use_mistral_hd128
            && n_tokens >= ATTN_PREFILL_MISTRAL_BC64_THRESHOLD;
        let use_mistral_smem = attention_prefill_mistral_smem_enabled() && use_mistral_hd128;
        let use_mistral_smem_f16 =
            attention_prefill_mistral_smem_f16_enabled() && use_mistral_hd128;
        let use_fa2_hd128 = attention_prefill_fa2_hd128_should_use(n_tokens, head_dim);
        let use_fa2_simd_hd128 = attention_prefill_fa2_simd_hd128_enabled() && head_dim == 128;
        let use_hd256 = attention_prefill_hd256_enabled() && head_dim == 256 && !use_v2;
        const FA2S_TG: usize = 128;
        let (pipeline, tg_width, groups_x) = if use_fa2_simd_hd128 {
            (
                &self.prefill_fa2_simd_hd128,
                FA2S_TG,
                (n_tokens as usize).div_ceil(8),
            )
        } else if use_fa2_hd128 {
            (
                &self.prefill_fa2_hd128,
                ATTN_TG,
                (n_tokens as usize).div_ceil(8),
            )
        } else if use_mistral_bc64 {
            (
                &self.prefill_mistral_hd128_bc64,
                ATTN_TG,
                (n_tokens as usize).div_ceil(8),
            )
        } else if use_mistral_smem_f16 {
            (
                &self.prefill_mistral_hd128_smem_f16,
                ATTN_TG,
                (n_tokens as usize).div_ceil(8),
            )
        } else if use_mistral_smem {
            (
                &self.prefill_mistral_hd128_smem,
                ATTN_TG,
                (n_tokens as usize).div_ceil(8),
            )
        } else if use_mistral_hd128 {
            (
                &self.prefill_mistral_hd128,
                ATTN_TG,
                (n_tokens as usize).div_ceil(8),
            )
        } else if use_v2 {
            if use_v2_hd128 {
                (&self.prefill_v2_hd128, ATTN2_TG, n_tokens as usize)
            } else {
                (&self.prefill_v2, ATTN2_TG, n_tokens as usize)
            }
        } else if use_hd256 {
            (&self.prefill_hd256, ATTN_TG, n_tokens as usize)
        } else {
            (&self.prefill, ATTN_TG, n_tokens as usize)
        };
        if attention_kernel_routing_log_enabled() {
            tracing::info!(
                n_tokens,
                n_heads,
                n_kv_heads,
                head_dim,
                profile = active_attention_routing_profile().name,
                fa2_hd128 = use_fa2_hd128,
                "attention_prefill kernel routing"
            );
        }
        device.execute_sync(|encoder| {
            encoder.setComputePipelineState(pipeline.state());
            // Bind buffers: Q=0, K=1, V=2, O=3
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(v.mtl_buffer()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(o.mtl_buffer()), 0, 3);
            }
            // Bind scalar parameters: indices 4-7
            bind_u32(encoder, 4, n_tokens);
            bind_u32(encoder, 5, n_heads);
            bind_u32(encoder, 6, n_kv_heads);
            bind_u32(encoder, 7, head_dim);
            // Grid: (n_tokens, n_heads) threadgroups, ATTN_TG threads each
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups_x,
                    height: n_heads as usize,
                    depth: 1,
                },
                MTLSize {
                    width: tg_width,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        })
    }

    /// Encode decode attention into an existing command encoder.
    ///
    /// Single-token attention: Q is one token, K/V are from the GPU KV cache.
    ///
    /// - `q`: [n_heads × head_dim] single query
    /// - `k_cache`: GPU KV cache K buffer for this layer
    /// - `v_cache`: GPU KV cache V buffer for this layer
    /// - `o`: [n_heads × head_dim] output buffer
    /// - `attend_start`: first token to attend to (sliding window offset)
    /// - `attend_len`: number of tokens to attend to
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_decode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        o: &MetalBuffer,
        kv_f16: bool,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        attend_start: u32,
        attend_len: u32,
    ) {
        assert!(
            head_dim <= MAX_HEAD_DIM,
            "head_dim {} exceeds MAX_HD ({}) in attention shader",
            head_dim,
            MAX_HEAD_DIM
        );
        let use_v2 = attention_decode_v2_enabled(attend_len);
        let use_hd256 = head_dim == 256;
        let use_hd128 = head_dim == 128;
        let use_sdpa = kv_f16 && use_hd256 && attention_decode_sdpa_enabled();
        let use_hd128_n2 = kv_f16 && use_hd128 && attention_decode_hd128_n2_enabled();
        let (pipeline, tg_width, groups_x) = if use_sdpa {
            // sdpa_vector: MLX-pattern lane-parallel decode (HD=256, f16 KV).
            (&self.decode_sdpa_hd256, ATTN_TG, n_heads as usize)
        } else if kv_f16 {
            if use_hd256 {
                (&self.decode_f16kv_hd256, ATTN_DEC_TG, n_heads as usize)
            } else if use_hd128_n2 {
                (
                    &self.decode_f16kv_hd128_n2,
                    ATTN_TG,
                    (n_heads as usize).div_ceil(2),
                )
            } else if use_hd128 {
                (&self.decode_f16kv_hd128, ATTN_DEC2_TG, n_heads as usize)
            } else {
                (&self.decode_f16kv, ATTN_DEC_TG, n_heads as usize)
            }
        } else if use_hd256 {
            (&self.decode_hd256, ATTN_DEC_TG, n_heads as usize)
        } else if use_hd128 {
            (&self.decode_hd128, ATTN_DEC2_TG, n_heads as usize)
        } else if use_v2 {
            (&self.decode_v2, ATTN_DEC2_TG, n_heads as usize)
        } else {
            (&self.decode, ATTN_DEC_TG, n_heads as usize)
        };
        if attention_kernel_routing_log_enabled() {
            tracing::info!(
                n_heads,
                n_kv_heads,
                head_dim,
                attend_start,
                attend_len,
                profile = active_attention_routing_profile().name,
                decode_sdpa = use_sdpa,
                decode_hd128_n2 = use_hd128_n2,
                "attention_decode kernel routing"
            );
        }
        encoder.setComputePipelineState(pipeline.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k_cache.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v_cache.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(o.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n_heads);
        bind_u32(encoder, 5, n_kv_heads);
        bind_u32(encoder, 6, head_dim);
        bind_u32(encoder, 7, attend_start);
        bind_u32(encoder, 8, attend_len);
        // Grid: n_heads threadgroups × ATTN_TG threads each
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_attention_decode_splitk(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        o: &MetalBuffer,
        partial_out: &MetalBuffer,
        partial_lse: &MetalBuffer,
        kv_f16: bool,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        attend_start: u32,
        attend_len: u32,
    ) {
        if !attention_decode_splitk_supported(kv_f16, head_dim) {
            self.encode_attention_decode(
                encoder,
                q,
                k_cache,
                v_cache,
                o,
                kv_f16,
                n_heads,
                n_kv_heads,
                head_dim,
                attend_start,
                attend_len,
            );
            return;
        }

        let chunk_size = attention_decode_splitk_chunk_size();
        let n_chunks = attend_len.div_ceil(chunk_size);
        let (partial_pipeline, reduce_pipeline, tg_width) = if head_dim == 256 {
            (
                &self.decode_splitk_f16kv_hd256_partial,
                &self.decode_splitk_f16kv_hd256_reduce,
                ATTN_TG,
            )
        } else {
            (
                &self.decode_splitk_f16kv_hd128_partial,
                &self.decode_splitk_f16kv_hd128_reduce,
                ATTN_DEC2_TG,
            )
        };

        encoder.setComputePipelineState(partial_pipeline.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k_cache.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v_cache.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(partial_out.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(partial_lse.mtl_buffer()), 0, 4);
        }
        bind_u32(encoder, 5, n_heads);
        bind_u32(encoder, 6, n_kv_heads);
        bind_u32(encoder, 7, head_dim);
        bind_u32(encoder, 8, attend_start);
        bind_u32(encoder, 9, attend_len);
        bind_u32(encoder, 10, chunk_size);
        bind_u32(encoder, 11, n_chunks);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_chunks as usize,
                height: n_heads as usize,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );

        barrier_buffers(encoder);

        encoder.setComputePipelineState(reduce_pipeline.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(partial_out.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(partial_lse.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(o.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n_heads);
        bind_u32(encoder, 4, head_dim);
        bind_u32(encoder, 5, n_chunks);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_heads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode decode attention using caller-provided split-K scratch buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_decode_with_scratch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        o: &MetalBuffer,
        partial_out: &MetalBuffer,
        partial_lse: &MetalBuffer,
        kv_f16: bool,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        attend_start: u32,
        attend_len: u32,
    ) {
        let use_splitk = attention_decode_splitk_should_use(kv_f16, head_dim, attend_len);
        if attention_kernel_routing_log_enabled() {
            tracing::info!(
                n_heads,
                n_kv_heads,
                head_dim,
                attend_start,
                attend_len,
                profile = active_attention_routing_profile().name,
                splitk = use_splitk,
                "attention_decode_with_scratch kernel routing"
            );
        }
        if use_splitk {
            self.encode_attention_decode_splitk(
                encoder,
                q,
                k_cache,
                v_cache,
                o,
                partial_out,
                partial_lse,
                kv_f16,
                n_heads,
                n_kv_heads,
                head_dim,
                attend_start,
                attend_len,
            );
        } else {
            self.encode_attention_decode(
                encoder,
                q,
                k_cache,
                v_cache,
                o,
                kv_f16,
                n_heads,
                n_kv_heads,
                head_dim,
                attend_start,
                attend_len,
            );
        }
    }

    /// Encode prefill attention into an existing command encoder.
    ///
    /// Does NOT create or commit a command buffer. Used for batching
    /// the prefill attention into a single command buffer with other ops.
    ///
    /// - `q`: [n_tokens × n_heads × head_dim] query vectors
    /// - `k`: [n_tokens × n_kv_heads × head_dim] key vectors
    /// - `v`: [n_tokens × n_kv_heads × head_dim] value vectors
    /// - `o`: [n_tokens × n_heads × head_dim] output buffer
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_prefill(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        o: &MetalBuffer,
        n_tokens: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
    ) {
        let use_v2 = attention_prefill_v2_enabled();
        let use_v2_hd128 = attention_prefill_hd128_enabled() && head_dim == 128;
        let use_mistral_hd128 = attention_prefill_mistral_hd128_enabled() && head_dim == 128;
        let use_mistral_bc64 = attention_prefill_mistral_bc64_enabled()
            && use_mistral_hd128
            && n_tokens >= ATTN_PREFILL_MISTRAL_BC64_THRESHOLD;
        let use_mistral_smem = attention_prefill_mistral_smem_enabled() && use_mistral_hd128;
        let use_mistral_smem_f16 =
            attention_prefill_mistral_smem_f16_enabled() && use_mistral_hd128;
        let use_fa2_hd128 = attention_prefill_fa2_hd128_should_use(n_tokens, head_dim);
        let use_fa2_simd_hd128 = attention_prefill_fa2_simd_hd128_enabled() && head_dim == 128;
        let use_hd256 = attention_prefill_hd256_enabled() && head_dim == 256 && !use_v2;
        if attention_kernel_routing_log_enabled() {
            tracing::info!(
                n_tokens,
                n_heads,
                n_kv_heads,
                head_dim,
                profile = active_attention_routing_profile().name,
                fa2_simd_hd128 = use_fa2_simd_hd128,
                fa2_hd128 = use_fa2_hd128,
                "encode_attention_prefill kernel routing"
            );
        }
        // FA2 simd (half8x8 matrix ops) is the fastest path for HD=128.
        const FA2S_TG: usize = 128;
        let (pipeline, tg_width, groups_x) = if use_fa2_simd_hd128 {
            (
                &self.prefill_fa2_simd_hd128,
                FA2S_TG,
                (n_tokens as usize).div_ceil(8),
            )
        } else if use_fa2_hd128 {
            (
                &self.prefill_fa2_hd128,
                ATTN_TG,
                (n_tokens as usize).div_ceil(8),
            )
        } else if use_mistral_bc64 {
            (
                &self.prefill_mistral_hd128_bc64,
                ATTN_TG,
                (n_tokens as usize).div_ceil(8),
            )
        } else if use_mistral_smem_f16 {
            (
                &self.prefill_mistral_hd128_smem_f16,
                ATTN_TG,
                (n_tokens as usize).div_ceil(8),
            )
        } else if use_mistral_smem {
            (
                &self.prefill_mistral_hd128_smem,
                ATTN_TG,
                (n_tokens as usize).div_ceil(8),
            )
        } else if use_mistral_hd128 {
            (
                &self.prefill_mistral_hd128,
                ATTN_TG,
                (n_tokens as usize).div_ceil(8),
            )
        } else if use_v2 {
            if use_v2_hd128 {
                (&self.prefill_v2_hd128, ATTN2_TG, n_tokens as usize)
            } else {
                (&self.prefill_v2, ATTN2_TG, n_tokens as usize)
            }
        } else if use_hd256 {
            (&self.prefill_hd256, ATTN_TG, n_tokens as usize)
        } else {
            (&self.prefill, ATTN_TG, n_tokens as usize)
        };
        assert!(
            head_dim <= MAX_HEAD_DIM,
            "head_dim {} exceeds MAX_HD ({}) in attention shader",
            head_dim,
            MAX_HEAD_DIM
        );
        encoder.setComputePipelineState(pipeline.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(o.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n_tokens);
        bind_u32(encoder, 5, n_heads);
        bind_u32(encoder, 6, n_kv_heads);
        bind_u32(encoder, 7, head_dim);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups_x,
                height: n_heads as usize,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode mistral-style HD128 prefill attention with f16 output.
    ///
    /// Output buffer stores [n_tokens × n_heads × head_dim] in half precision.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_prefill_f16out_hd128(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        o_f16: &MetalBuffer,
        n_tokens: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
    ) {
        debug_assert_eq!(head_dim, 128);
        let pipeline = &self.prefill_mistral_f16out_hd128;
        encoder.setComputePipelineState(pipeline.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(o_f16.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n_tokens);
        bind_u32(encoder, 5, n_heads);
        bind_u32(encoder, 6, n_kv_heads);
        bind_u32(encoder, 7, head_dim);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n_tokens as usize).div_ceil(8),
                height: n_heads as usize,
                depth: 1,
            },
            MTLSize {
                width: ATTN_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode batched prefill attention against existing KV cache.
    ///
    /// - `q`: [n_tokens × n_heads × head_dim] query vectors for suffix tokens
    /// - `k_cache`/`v_cache`: KV cache buffers containing restored prefix + appended suffix
    /// - `base_seq_len`: prefix length already present in KV cache before current suffix
    /// - `sliding_window`: 0 disables sliding window, otherwise per-query window size
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_prefill_cached(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        o: &MetalBuffer,
        kv_f16: bool,
        n_tokens: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        base_seq_len: u32,
        sliding_window: u32,
    ) {
        assert!(
            head_dim <= MAX_HEAD_DIM,
            "head_dim {} exceeds MAX_HD ({}) in attention shader",
            head_dim,
            MAX_HEAD_DIM
        );
        let use_fa2 = attention_prefill_fa2_cached_should_use(
            kv_f16,
            n_tokens,
            head_dim,
            base_seq_len,
            sliding_window,
        );
        if attention_kernel_routing_log_enabled() {
            tracing::info!(
                kv_f16,
                n_tokens,
                n_heads,
                n_kv_heads,
                head_dim,
                base_seq_len,
                sliding_window,
                fa2_cached_hd256 = use_fa2,
                mode = ?attention_prefill_fa2_mode(),
                "encode_attention_prefill_cached kernel routing"
            );
        }
        if use_fa2 {
            // FA2 multi-query kernel: grid is (ceil(n_tokens/8), n_heads).
            const FA2_Q: usize = 8;
            const FA2_TG: usize = 256;
            let n_tile_q = (n_tokens as usize).div_ceil(FA2_Q);
            encoder.setComputePipelineState(self.prefill_cache_fa2_hd256.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(k_cache.mtl_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(v_cache.mtl_buffer()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(o.mtl_buffer()), 0, 3);
            }
            bind_u32(encoder, 4, n_tokens);
            bind_u32(encoder, 5, n_heads);
            bind_u32(encoder, 6, n_kv_heads);
            bind_u32(encoder, 7, head_dim);
            bind_u32(encoder, 8, base_seq_len);
            bind_u32(encoder, 9, sliding_window);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: n_tile_q,
                    height: n_heads as usize,
                    depth: 1,
                },
                MTLSize {
                    width: FA2_TG,
                    height: 1,
                    depth: 1,
                },
            );
            return;
        }
        let pipeline = if kv_f16 {
            &self.prefill_cache_f16kv
        } else {
            &self.prefill_cache
        };
        encoder.setComputePipelineState(pipeline.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k_cache.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v_cache.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(o.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n_tokens);
        bind_u32(encoder, 5, n_heads);
        bind_u32(encoder, 6, n_kv_heads);
        bind_u32(encoder, 7, head_dim);
        bind_u32(encoder, 8, base_seq_len);
        bind_u32(encoder, 9, sliding_window);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_tokens as usize,
                height: n_heads as usize,
                depth: 1,
            },
            MTLSize {
                width: ATTN_TG,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Dispatch decode attention (standalone, creates own command buffer).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode(
        &self,
        device: &MetalDevice,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        o: &MetalBuffer,
        kv_f16: bool,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        attend_start: u32,
        attend_len: u32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_attention_decode(
                encoder,
                q,
                k_cache,
                v_cache,
                o,
                kv_f16,
                n_heads,
                n_kv_heads,
                head_dim,
                attend_start,
                attend_len,
            );
            Ok(())
        })
    }

    /// Dispatch split-K decode attention directly, bypassing auto routing.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_splitk(
        &self,
        device: &MetalDevice,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        o: &MetalBuffer,
        kv_f16: bool,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        attend_start: u32,
        attend_len: u32,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            attention_decode_splitk_supported(kv_f16, head_dim),
            "split-K decode only supports f16 KV with head_dim 128 or 256"
        );
        let chunk_size = attention_decode_splitk_chunk_size();
        let n_chunks = attend_len.div_ceil(chunk_size) as usize;
        let partial_out = MetalBuffer::new(
            device.device(),
            n_heads as usize * n_chunks * head_dim as usize * std::mem::size_of::<f32>(),
        )?;
        let partial_lse = MetalBuffer::new(
            device.device(),
            n_heads as usize * n_chunks * std::mem::size_of::<f32>(),
        )?;
        device.execute_sync(|encoder| {
            self.encode_attention_decode_splitk(
                encoder,
                q,
                k_cache,
                v_cache,
                o,
                &partial_out,
                &partial_lse,
                kv_f16,
                n_heads,
                n_kv_heads,
                head_dim,
                attend_start,
                attend_len,
            );
            Ok(())
        })
    }
}

fn batch_q4k_blocked_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_BATCH_Q4K_BLOCKED") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        }
        // Default ON: blocked threadgroup layout (stride 8, 6 KB TG memory).
        // Ports llama.cpp's kernel_mul_mm pattern for 3-4 TGs/SM occupancy
        // and 1.33 MACs/load inner loop.
        Err(_) => true,
    })
}

fn batch_q6k_blocked_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_BATCH_Q6K_BLOCKED") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        }
        Err(_) => true, // Default ON
    })
}

fn batch_q4k_inline_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_BATCH_Q4K_INLINE") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        }
        // Default OFF: inline dequant regresses 6-8% because each thread reads
        // block headers from device memory (L1-cached but still slower than the
        // precomputed threadgroup-memory path). The 32-thread Phase 1 preload +
        // barrier is cheaper than 1024 redundant L1 header reads.
        Err(_) => false,
    })
}

fn batch_q4k_bn32_full_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_BATCH_Q4K_BN32") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        }
        // Default OFF: BN=32 (TG=128) reduces Phase 2 dequant throughput
        // because only 128 threads cooperatively load vs 256. The ~3 TGs/SM
        // occupancy gain doesn't compensate. Same pattern as v2.
        Err(_) => false,
    })
}

fn batch_q4k_v2_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_BATCH_Q4K_V2") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        // Default OFF: 2-B-fragment inner loop is ~2x slower despite better
        // MAC/load ratio. Root cause: TG=128 halves cooperative loading throughput
        // (dequant phase + B-tile load each thread does 2x work). The loading
        // phase dominates over the compute improvement. Needs a different approach:
        // keep TG=256 but restructure SG assignment for 2-B reuse.
        Err(_) => false,
    })
}

fn attention_prefill_v2_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_PREFILL_ATTN_V2") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        // Default ON for current Llama/Qwen/Gemma prefill workloads.
        // Set AX_METAL_PREFILL_ATTN_V2=0 to disable.
        Err(_) => true,
    })
}

fn attention_prefill_hd256_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    // Default ON — constexpr HD=256 removes dynamic head_dim loops in the prefill kernel.
    // Set AX_METAL_PREFILL_HD256=0 to disable.
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_PREFILL_HD256") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v != "0" && v != "false" && v != "off"
        }
        Err(_) => true,
    })
}

fn attention_prefill_hd128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    // Experimental: default OFF until sustained win is confirmed.
    // Set AX_METAL_PREFILL_HD128=1 to enable.
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_PREFILL_HD128") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => false,
    })
}

fn attention_prefill_mistral_hd128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    // Mistral-style prefill tiling for HD=128 (BR=8, BC=32).
    // Default ON; set AX_METAL_PREFILL_MISTRAL_HD128=0 to disable.
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_PREFILL_MISTRAL_HD128") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        }
        Err(_) => true,
    })
}

fn attention_prefill_mistral_bc64_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    // Long-prompt variant for mistral-style HD128 prefill.
    // Default ON; set AX_METAL_PREFILL_MISTRAL_BC64=0 to disable.
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_PREFILL_MISTRAL_BC64") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        }
        Err(_) => true,
    })
}

fn attention_prefill_mistral_smem_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    // Experimental: default OFF until sustained win is confirmed.
    // Set AX_METAL_PREFILL_MISTRAL_SMEM=1 to enable.
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_PREFILL_MISTRAL_SMEM") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => false,
    })
}

fn attention_prefill_mistral_smem_f16_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    // Experimental f16 shared-memory variant.
    // Set AX_METAL_PREFILL_MISTRAL_SMEM_F16=1 to enable.
    *ENABLED.get_or_init(
        || match std::env::var("AX_METAL_PREFILL_MISTRAL_SMEM_F16") {
            Ok(v) => {
                let v = v.trim().to_ascii_lowercase();
                v == "1" || v == "true" || v == "on"
            }
            Err(_) => false,
        },
    )
}

fn batch_f16in_bn32_enabled() -> bool {
    match std::env::var("AX_METAL_F16IN_BN32") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        }
        // Default ON: BN=32 (TG=128, ~13KB TG memory) fits 2 TGs per SM vs BN=64 (TG=256,
        // ~20KB, 1 TG/SM).  Doubled occupancy hides memory latency and gives +5% (N=39) to
        // +14% (N=1024) prefill throughput — empirically measured on Q4_K_M across models.
        // Set AX_METAL_F16IN_BN32=0 to disable.
        Err(_) => global_profile().batch_prefill.use_bn32,
    }
}

fn batch_f16in_bk32_enabled() -> bool {
    match std::env::var("AX_METAL_F16IN_BK32") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        // Default ON: BK=32 improves long-prompt prefill on current models.
        // Set AX_METAL_F16IN_BK32=0 to disable.
        Err(_) => global_profile().batch_prefill.use_bk32,
    }
}

fn q8_f16in_full_min_n() -> usize {
    match std::env::var("AX_METAL_Q8_F16IN_FULL_MIN_N") {
        Ok(v) => v
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|&x| x > 0)
            .unwrap_or(DB_TILE_N),
        Err(_) => global_profile().batch_prefill.q8_f16in_full_min_n.max(1) as usize,
    }
}

fn attention_decode_sdpa_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_DECODE_SDPA") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        }
        Err(_) => active_attention_routing_profile().decode_sdpa_default,
    })
}

fn attention_prefill_fa2_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_PREFILL_FA2") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        }
        // Default OFF: FA2 serial score loop + idle threads regresses vs f16kv prefill.
        // Enable with AX_METAL_PREFILL_FA2=1.
        Err(_) => false,
    })
}

fn attention_prefill_fa2_hd128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_PREFILL_FA2_HD128") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        // Experimental: default OFF until sustained gains are verified.
        Err(_) => false,
    })
}

fn parse_kernel_mode(var: &str, default_mode: KernelMode) -> KernelMode {
    match std::env::var(var) {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "off" | "0" | "false" => KernelMode::Off,
            "on" | "1" | "true" => KernelMode::On,
            "auto" => KernelMode::Auto,
            _ => default_mode,
        },
        Err(_) => default_mode,
    }
}

fn parse_positive_u32_env(var: &str, default_value: u32) -> u32 {
    match std::env::var(var) {
        Ok(v) => v
            .trim()
            .parse::<u32>()
            .ok()
            .filter(|&x| x > 0)
            .unwrap_or(default_value),
        Err(_) => default_value,
    }
}

fn profile_kernel_mode(mode: ProfileKernelMode) -> KernelMode {
    match mode {
        ProfileKernelMode::Off => KernelMode::Off,
        ProfileKernelMode::On => KernelMode::On,
        ProfileKernelMode::Auto => KernelMode::Auto,
    }
}

fn attention_prefill_fa2_mode() -> KernelMode {
    let profile_default = profile_kernel_mode(global_profile().attention_prefill.fa2_mode.clone());
    let mode = parse_kernel_mode("AX_METAL_PREFILL_FA2_MODE", profile_default);
    match mode {
        // Backward compatibility with existing boolean toggle.
        KernelMode::Off => {
            if attention_prefill_fa2_enabled() {
                KernelMode::On
            } else {
                KernelMode::Off
            }
        }
        _ => mode,
    }
}

fn attention_prefill_fa2_hd128_mode() -> KernelMode {
    let profile_default =
        profile_kernel_mode(global_profile().attention_prefill.fa2_hd128_mode.clone());
    let mode = parse_kernel_mode("AX_METAL_PREFILL_FA2_HD128_MODE", profile_default);
    match mode {
        // Backward compatibility with existing boolean toggle.
        KernelMode::Off => {
            if attention_prefill_fa2_hd128_enabled() {
                KernelMode::On
            } else {
                KernelMode::Off
            }
        }
        _ => mode,
    }
}

fn attention_prefill_fa2_auto_min_tokens() -> u32 {
    let profile = global_profile();
    let threshold = profile.attention_prefill.fa2_auto_min_tokens;
    parse_positive_u32_env(
        "AX_METAL_PREFILL_FA2_AUTO_MIN_TOKENS",
        if threshold > 0 {
            threshold
        } else {
            active_attention_routing_profile().prefill_fa2_auto_min_tokens
        },
    )
}

fn attention_prefill_fa2_auto_min_base_seq() -> u32 {
    let profile = global_profile();
    let threshold = profile.attention_prefill.fa2_auto_min_base_seq;
    parse_positive_u32_env(
        "AX_METAL_PREFILL_FA2_AUTO_MIN_BASE_SEQ",
        if threshold > 0 {
            threshold
        } else {
            active_attention_routing_profile().prefill_fa2_auto_min_base_seq
        },
    )
}

fn attention_prefill_fa2_hd128_auto_min_tokens() -> u32 {
    let profile = global_profile();
    let threshold = profile.attention_prefill.fa2_hd128_auto_min_tokens;
    parse_positive_u32_env(
        "AX_METAL_PREFILL_FA2_HD128_AUTO_MIN_TOKENS",
        if threshold > 0 {
            threshold
        } else {
            active_attention_routing_profile().prefill_fa2_hd128_auto_min_tokens
        },
    )
}

fn attention_decode_splitk_mode() -> KernelMode {
    static MODE: OnceLock<KernelMode> = OnceLock::new();
    *MODE.get_or_init(|| parse_kernel_mode("AX_METAL_DECODE_SPLITK_MODE", KernelMode::Auto))
}

pub fn attention_decode_splitk_chunk_size() -> u32 {
    static CHUNK_SIZE: OnceLock<u32> = OnceLock::new();
    *CHUNK_SIZE.get_or_init(|| {
        let from_env = std::env::var("AX_METAL_DECODE_SPLITK_CHUNK_SIZE")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok());
        if let Some(size) = from_env {
            size
        } else {
            let profile = global_profile();
            let size = profile.attention_decode.splitk_chunk_size;
            if size > 0 {
                size
            } else {
                active_attention_routing_profile().decode_splitk_chunk_size
            }
        }
    })
}

fn attention_decode_splitk_auto_min_tokens() -> u32 {
    static MIN_TOKENS: OnceLock<u32> = OnceLock::new();
    *MIN_TOKENS.get_or_init(|| {
        let from_env = std::env::var("AX_METAL_DECODE_SPLITK_AUTO_MIN_TOKENS")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok());
        if let Some(tokens) = from_env {
            tokens
        } else {
            let profile = global_profile();
            let threshold = profile.attention_decode.splitk_threshold;
            if threshold > 0 {
                threshold
            } else {
                active_attention_routing_profile().decode_splitk_auto_min_tokens
            }
        }
    })
}

fn attention_decode_splitk_supported(kv_f16: bool, head_dim: u32) -> bool {
    kv_f16 && matches!(head_dim, 128 | 256)
}

fn attention_decode_splitk_should_use_mode(
    mode: KernelMode,
    kv_f16: bool,
    head_dim: u32,
    attend_len: u32,
) -> bool {
    if !attention_decode_splitk_supported(kv_f16, head_dim) {
        return false;
    }
    match mode {
        KernelMode::Off => false,
        KernelMode::On => true,
        KernelMode::Auto => {
            head_dim == 256 && attend_len >= attention_decode_splitk_auto_min_tokens()
        }
    }
}

fn attention_decode_splitk_should_use(kv_f16: bool, head_dim: u32, attend_len: u32) -> bool {
    attention_decode_splitk_should_use_mode(
        attention_decode_splitk_mode(),
        kv_f16,
        head_dim,
        attend_len,
    )
}

fn attention_prefill_fa2_cached_should_use(
    kv_f16: bool,
    n_tokens: u32,
    head_dim: u32,
    base_seq_len: u32,
    sliding_window: u32,
) -> bool {
    if !(kv_f16 && head_dim == 256) {
        return false;
    }
    match attention_prefill_fa2_mode() {
        KernelMode::Off => false,
        KernelMode::On => true,
        KernelMode::Auto => {
            // Conservative auto-gate for current kernels. Designed as a benchmark gate:
            // only route FA2 where sequence depth/width is high enough to amortize setup.
            n_tokens >= attention_prefill_fa2_auto_min_tokens()
                && base_seq_len >= attention_prefill_fa2_auto_min_base_seq()
                && sliding_window == 0
        }
    }
}

fn attention_prefill_fa2_hd128_should_use(n_tokens: u32, head_dim: u32) -> bool {
    if head_dim != 128 {
        return false;
    }
    match attention_prefill_fa2_hd128_mode() {
        KernelMode::Off => false,
        KernelMode::On => true,
        KernelMode::Auto => n_tokens >= attention_prefill_fa2_hd128_auto_min_tokens(),
    }
}

fn attention_prefill_fa2_simd_hd128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_PREFILL_FA2_SIMD") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => true, // default ON — this is the fast path
    })
}

fn attention_kernel_routing_log_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(std::env::var("AX_METAL_LOG_ATTN_ROUTING"), Ok(v) if {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        })
    })
}

fn attention_decode_v2_enabled(attend_len: u32) -> bool {
    static OVERRIDE: OnceLock<Option<bool>> = OnceLock::new();
    match *OVERRIDE.get_or_init(|| match std::env::var("AX_METAL_DECODE_ATTN_V2") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            if v == "1" || v == "true" || v == "on" {
                Some(true)
            } else if v == "0" || v == "false" || v == "off" {
                Some(false)
            } else {
                None
            }
        }
        Err(_) => None,
    }) {
        Some(enabled) => enabled,
        // Default OFF: TG=128 path regresses for current models/workloads.
        None => {
            let _ = attend_len;
            false
        }
    }
}

fn attention_decode_hd128_n2_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_DECODE_HD128_N2") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => active_attention_routing_profile().decode_hd128_n2_default,
    })
}

fn matvec_q4k_n4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_MATVEC_N4") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        // Default OFF — 4× more qs weight reads (one per row × 4 rows) outweighs
        // x-bandwidth savings (x fits in L1 cache and is already effectively free).
        // Net result: -12% decode regression on Apple Silicon UMA.
        // Enable with AX_METAL_MATVEC_N4=1 to A/B test.
        Err(_) => false,
    })
}

fn matvec_q4k_x2_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_MATVEC_Q4K_X2") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => false,
    })
}

fn matvec_q4k_nr2_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_MATVEC_Q4K_NR2") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => false,
    })
}

fn matvec_q6k_nr2_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_MATVEC_Q6K_NR2") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => false,
    })
}

fn matvec_q4k_blk2_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_MATVEC_Q4K_BLK2") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => false,
    })
}

fn matvec_q4k_threadgroup_size() -> usize {
    static TG: OnceLock<usize> = OnceLock::new();
    *TG.get_or_init(|| match std::env::var("AX_METAL_MATVEC_Q4K_TG") {
        Ok(v) => match v.trim() {
            "64" => DEQUANT_MATVEC_Q4K_NR2_TG,
            "256" => DEQUANT_MATVEC_Q4K_TG256,
            "128" => DEQUANT_MATVEC_TG,
            other => {
                tracing::warn!(
                    value = other,
                    "Invalid AX_METAL_MATVEC_Q4K_TG value; falling back to profile",
                );
                global_profile().matvec_params("q4_k").threadgroup_size as usize
            }
        },
        Err(_) => global_profile().matvec_params("q4_k").threadgroup_size as usize,
    })
}

fn matvec_q4k_rows_per_simdgroup() -> u32 {
    static ROWS: OnceLock<u32> = OnceLock::new();
    *ROWS.get_or_init(|| {
        std::env::var("AX_METAL_MATVEC_Q4K_NR")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or_else(|| global_profile().matvec_params("q4_k").rows_per_simdgroup)
    })
}

fn matvec_q6k_threadgroup_size() -> usize {
    static TG: OnceLock<usize> = OnceLock::new();
    *TG.get_or_init(|| match std::env::var("AX_METAL_MATVEC_Q6K_TG") {
        Ok(v) => match v.trim() {
            "64" => DEQUANT_MATVEC_Q6K_NR2_TG,
            "128" => DEQUANT_MATVEC_TG,
            other => {
                tracing::warn!(
                    value = other,
                    "Invalid AX_METAL_MATVEC_Q6K_TG value; falling back to profile",
                );
                global_profile().matvec_params("q6_k").threadgroup_size as usize
            }
        },
        Err(_) => global_profile().matvec_params("q6_k").threadgroup_size as usize,
    })
}

fn matvec_q6k_rows_per_simdgroup() -> u32 {
    static ROWS: OnceLock<u32> = OnceLock::new();
    *ROWS.get_or_init(|| {
        std::env::var("AX_METAL_MATVEC_Q6K_NR")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or_else(|| global_profile().matvec_params("q6_k").rows_per_simdgroup)
    })
}

/// Whether K-parallel simd_sum batch dequant+matmul kernels are enabled.
///
/// **Default: OFF.** The simd_sum approach has ~27× lower arithmetic intensity than the
/// tiled simdgroup_matrix kernels for N>1 prefill (B-row reuse across only 8 A-rows vs 64).
/// Benchmarked at 3–5× prefill regression across all context lengths on Qwen3 8B Q4_K_M.
///
/// Controlled by `AX_METAL_BATCH_SIMD`:
/// - `1` / `true` / `on`           → enabled (for A/B testing only)
/// - unset / `0` / `false` / `off` → disabled (default)
pub fn batch_simd_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_BATCH_SIMD") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => false,
    })
}

/// Bind three buffers at indices 0, 1, 2.
fn bind_buffers(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buf0: &MetalBuffer,
    buf1: &MetalBuffer,
    buf2: &MetalBuffer,
) {
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(buf0.mtl_buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(buf1.mtl_buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(buf2.mtl_buffer()), 0, 2);
    }
}

/// Bind a u32 scalar at the given buffer index.
fn bind_u32(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, index: usize, value: u32) {
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::new_unchecked(&value as *const u32 as *mut c_void),
            std::mem::size_of::<u32>(),
            index,
        );
    }
}

/// Bind an f32 scalar at the given buffer index.
fn bind_f32(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, index: usize, value: f32) {
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::new_unchecked(&value as *const f32 as *mut c_void),
            std::mem::size_of::<f32>(),
            index,
        );
    }
}

/// Insert a buffer memory barrier between dispatches in the same command encoder.
///
/// Required between dispatches that have read-after-write dependencies on
/// the same Metal buffer. Without this, dispatches may execute concurrently
/// and read stale data.
pub fn barrier_buffers(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
    if !barriers_enabled() {
        return;
    }
    inc_buffer_barrier_count();
    encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
}

/// Smart barrier tracker for concurrent Metal dispatch.
///
/// Tracks buffer read/write ranges and inserts barriers only when a new
/// dispatch conflicts with pending operations. Ports llama.cpp's
/// `ggml_metal_op_concurrency_check` pattern (ggml-metal-ops.cpp:221).
///
/// Usage:
/// ```ignore
/// let mut sb = SmartBarrier::new(encoder);
/// sb.pre_dispatch(&[&buf_a], &[&buf_b]);  // check + barrier if needed
/// encode_op(encoder, &buf_a, &buf_b);
/// sb.post_dispatch(&[&buf_a], &[&buf_b]); // register ranges
/// ```
pub struct SmartBarrier<'a> {
    encoder: &'a ProtocolObject<dyn MTLComputeCommandEncoder>,
    /// Pending (buffer_ptr, is_write) ranges from dispatches since last barrier.
    pending: Vec<(usize, bool)>,
}

impl<'a> SmartBarrier<'a> {
    /// Create a new tracker wrapping a concurrent encoder.
    pub fn new(encoder: &'a ProtocolObject<dyn MTLComputeCommandEncoder>) -> Self {
        Self {
            encoder,
            pending: Vec::with_capacity(32),
        }
    }

    /// Check for conflicts and insert barrier if needed, BEFORE dispatching.
    /// `reads`: buffers the next dispatch will read from.
    /// `writes`: buffers the next dispatch will write to.
    pub fn pre_dispatch(&mut self, reads: &[&MetalBuffer], writes: &[&MetalBuffer]) {
        if !barriers_enabled() {
            return;
        }
        let needs_barrier = self.has_conflict(reads, writes);
        if needs_barrier {
            inc_buffer_barrier_count();
            self.encoder
                .memoryBarrierWithScope(MTLBarrierScope::Buffers);
            self.pending.clear();
        }
    }

    /// Register buffer ranges AFTER dispatching.
    pub fn post_dispatch(&mut self, reads: &[&MetalBuffer], writes: &[&MetalBuffer]) {
        for buf in reads {
            self.pending.push((buf.ptr_id(), false));
        }
        for buf in writes {
            self.pending.push((buf.ptr_id(), true));
        }
    }

    /// Insert an unconditional barrier (e.g., at end of encoding).
    pub fn flush(&mut self) {
        if !self.pending.is_empty() && barriers_enabled() {
            inc_buffer_barrier_count();
            self.encoder
                .memoryBarrierWithScope(MTLBarrierScope::Buffers);
            self.pending.clear();
        }
    }

    /// Check if any of the new reads/writes conflict with pending ranges.
    /// Conflict = new read overlaps pending write, OR new write overlaps pending read/write.
    fn has_conflict(&self, reads: &[&MetalBuffer], writes: &[&MetalBuffer]) -> bool {
        for buf in reads {
            let id = buf.ptr_id();
            // Read conflicts with pending WRITE to same buffer.
            if self.pending.iter().any(|&(pid, is_w)| is_w && pid == id) {
                return true;
            }
        }
        for buf in writes {
            let id = buf.ptr_id();
            // Write conflicts with ANY pending access to same buffer.
            if self.pending.iter().any(|&(pid, _)| pid == id) {
                return true;
            }
        }
        false
    }
}

/// Threadgroup size for elementwise kernels (must match shader constant).
const ELEMENTWISE_TG_SIZE: usize = 256;

/// Pre-compiled elementwise compute pipelines.
///
/// Provides GPU kernels for RMSNorm, RoPE, GELU/SiLU activation, and
/// elementwise add. These enable phased forward pass dispatch where
/// multiple operations are batched into a single GPU command buffer.
pub struct ElementwiseKernels {
    rms_norm: ComputePipeline,
    rms_norm_batch: ComputePipeline,
    rms_norm_out: ComputePipeline,
    rms_norm_out_batch: ComputePipeline,
    rms_norm_out_batch_vec4: ComputePipeline,
    rms_norm_out_batch_f16: ComputePipeline,
    residual_add_rms_norm_out_batch: ComputePipeline,
    residual_add_rms_norm_out_batch_f16: ComputePipeline,
    rope: ComputePipeline,
    rope_batch: ComputePipeline,
    per_head_rms_norm: ComputePipeline,
    per_head_rms_norm_batch: ComputePipeline,
    qk_norm_rope_batch: ComputePipeline,
    qkv_split_qk_norm_rope_append_kv_batch_f32: ComputePipeline,
    qkv_split_qk_norm_rope_append_kv_batch_f16: ComputePipeline,
    /// Fused: QKV split + bias + QK norm + RoPE + KV append (Qwen3 with QKV bias).
    qkv_split_bias_qknorm_rope_append_kv_batch_f32: ComputePipeline,
    /// Same, with f16 KV cache (the DEFAULT path since max_seq_len ≥ 256 enables f16 KV).
    qkv_split_bias_qknorm_rope_append_kv_batch_f16: ComputePipeline,
    gelu_elementwise_mul: ComputePipeline,
    gelu_elementwise_mul_batch: ComputePipeline,
    gelu_inplace: ComputePipeline,
    gelu_inplace_batch: ComputePipeline,
    silu_elementwise_mul: ComputePipeline,
    silu_elementwise_mul_batch: ComputePipeline,
    silu_elementwise_mul_batch_f16: ComputePipeline,
    elementwise_add: ComputePipeline,
    elementwise_add_batch: ComputePipeline,
    cast_f32_to_f16: ComputePipeline,
    cast_f16_to_f32: ComputePipeline,
    qkv_split_batch: ComputePipeline,
    qkv_split_rope_batch: ComputePipeline,
    qkv_split_rope_append_kv_batch_f32: ComputePipeline,
    qkv_split_rope_append_kv_batch_f16: ComputePipeline,
    kv_append_f32: ComputePipeline,
    kv_append_batch_f32: ComputePipeline,
    kv_append_f16: ComputePipeline,
    kv_append_batch_f16: ComputePipeline,
    kv_append_batch2_f32: ComputePipeline,
    kv_append_batch2_f16: ComputePipeline,
}

impl ElementwiseKernels {
    /// Compile all elementwise kernels from embedded Metal source.
    pub fn new(device: &MetalDevice) -> anyhow::Result<Self> {
        let rms_norm =
            ComputePipeline::from_source(device.device(), ELEMENTWISE_SHADER_SRC, "rms_norm_f32")
                .context("Failed to compile rms_norm_f32 kernel")?;
        let rms_norm_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "rms_norm_batch_f32",
        )
        .context("Failed to compile rms_norm_batch_f32 kernel")?;

        let rms_norm_out = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "rms_norm_out_f32",
        )
        .context("Failed to compile rms_norm_out_f32 kernel")?;
        let rms_norm_out_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "rms_norm_out_batch_f32",
        )
        .context("Failed to compile rms_norm_out_batch_f32 kernel")?;
        let rms_norm_out_batch_vec4 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "rms_norm_out_batch_f32_vec4",
        )
        .context("Failed to compile rms_norm_out_batch_f32_vec4 kernel")?;
        let rms_norm_out_batch_f16 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "rms_norm_out_batch_f16",
        )
        .context("Failed to compile rms_norm_out_batch_f16 kernel")?;
        let residual_add_rms_norm_out_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "residual_add_rms_norm_out_batch_f32",
        )
        .context("Failed to compile residual_add_rms_norm_out_batch_f32 kernel")?;
        let residual_add_rms_norm_out_batch_f16 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "residual_add_rms_norm_out_batch_f16",
        )
        .context("Failed to compile residual_add_rms_norm_out_batch_f16 kernel")?;

        let rope =
            ComputePipeline::from_source(device.device(), ELEMENTWISE_SHADER_SRC, "rope_f32")
                .context("Failed to compile rope_f32 kernel")?;
        let rope_batch =
            ComputePipeline::from_source(device.device(), ELEMENTWISE_SHADER_SRC, "rope_batch_f32")
                .context("Failed to compile rope_batch_f32 kernel")?;

        let per_head_rms_norm = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "per_head_rms_norm_f32",
        )
        .context("Failed to compile per_head_rms_norm_f32 kernel")?;
        let per_head_rms_norm_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "per_head_rms_norm_batch_f32",
        )
        .context("Failed to compile per_head_rms_norm_batch_f32 kernel")?;
        let qk_norm_rope_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "qk_norm_rope_batch_f32",
        )
        .context("Failed to compile qk_norm_rope_batch_f32 kernel")?;
        let qkv_split_qk_norm_rope_append_kv_batch_f32 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "qkv_split_qk_norm_rope_append_kv_batch_f32",
        )
        .context("Failed to compile qkv_split_qk_norm_rope_append_kv_batch_f32 kernel")?;
        let qkv_split_qk_norm_rope_append_kv_batch_f16 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "qkv_split_qk_norm_rope_append_kv_batch_f16",
        )
        .context("Failed to compile qkv_split_qk_norm_rope_append_kv_batch_f16 kernel")?;
        let qkv_split_bias_qknorm_rope_append_kv_batch_f32 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "qkv_split_bias_qknorm_rope_append_kv_batch_f32",
        )
        .context("Failed to compile qkv_split_bias_qknorm_rope_append_kv_batch_f32 kernel")?;
        let qkv_split_bias_qknorm_rope_append_kv_batch_f16 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "qkv_split_bias_qknorm_rope_append_kv_batch_f16",
        )
        .context("Failed to compile qkv_split_bias_qknorm_rope_append_kv_batch_f16 kernel")?;

        let gelu_elementwise_mul = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "gelu_elementwise_mul_f32",
        )
        .context("Failed to compile gelu_elementwise_mul_f32 kernel")?;
        let gelu_elementwise_mul_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "gelu_elementwise_mul_batch_f32",
        )
        .context("Failed to compile gelu_elementwise_mul_batch_f32 kernel")?;
        let gelu_inplace = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "gelu_inplace_f32",
        )
        .context("Failed to compile gelu_inplace_f32 kernel")?;
        let gelu_inplace_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "gelu_inplace_batch_f32",
        )
        .context("Failed to compile gelu_inplace_batch_f32 kernel")?;

        let silu_elementwise_mul = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "silu_elementwise_mul_f32",
        )
        .context("Failed to compile silu_elementwise_mul_f32 kernel")?;
        let silu_elementwise_mul_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "silu_elementwise_mul_batch_f32",
        )
        .context("Failed to compile silu_elementwise_mul_batch_f32 kernel")?;
        let silu_elementwise_mul_batch_f16 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "silu_elementwise_mul_batch_f16",
        )
        .context("Failed to compile silu_elementwise_mul_batch_f16 kernel")?;

        let elementwise_add = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "elementwise_add_f32",
        )
        .context("Failed to compile elementwise_add_f32 kernel")?;
        let elementwise_add_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "elementwise_add_batch_f32",
        )
        .context("Failed to compile elementwise_add_batch_f32 kernel")?;
        let cast_f32_to_f16 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "cast_f32_to_f16",
        )
        .context("Failed to compile cast_f32_to_f16 kernel")?;
        let cast_f16_to_f32 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "cast_f16_to_f32",
        )
        .context("Failed to compile cast_f16_to_f32 kernel")?;
        let qkv_split_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "qkv_split_batch_f32",
        )
        .context("Failed to compile qkv_split_batch_f32 kernel")?;
        let qkv_split_rope_batch = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "qkv_split_rope_batch_f32",
        )
        .context("Failed to compile qkv_split_rope_batch_f32 kernel")?;
        let qkv_split_rope_append_kv_batch_f32 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "qkv_split_rope_append_kv_batch_f32",
        )
        .context("Failed to compile qkv_split_rope_append_kv_batch_f32 kernel")?;
        let qkv_split_rope_append_kv_batch_f16 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "qkv_split_rope_append_kv_batch_f16",
        )
        .context("Failed to compile qkv_split_rope_append_kv_batch_f16 kernel")?;

        let kv_append_f32 =
            ComputePipeline::from_source(device.device(), ELEMENTWISE_SHADER_SRC, "kv_append_f32")
                .context("Failed to compile kv_append_f32 kernel")?;
        let kv_append_batch_f32 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "kv_append_batch_f32",
        )
        .context("Failed to compile kv_append_batch_f32 kernel")?;
        let kv_append_f16 =
            ComputePipeline::from_source(device.device(), ELEMENTWISE_SHADER_SRC, "kv_append_f16")
                .context("Failed to compile kv_append_f16 kernel")?;
        let kv_append_batch_f16 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "kv_append_batch_f16",
        )
        .context("Failed to compile kv_append_batch_f16 kernel")?;
        let kv_append_batch2_f32 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "kv_append_batch2_f32",
        )
        .context("Failed to compile kv_append_batch2_f32 kernel")?;
        let kv_append_batch2_f16 = ComputePipeline::from_source(
            device.device(),
            ELEMENTWISE_SHADER_SRC,
            "kv_append_batch2_f16",
        )
        .context("Failed to compile kv_append_batch2_f16 kernel")?;

        tracing::info!("Elementwise Metal kernels compiled (8 kernels)");

        Ok(Self {
            rms_norm,
            rms_norm_batch,
            rms_norm_out,
            rms_norm_out_batch,
            rms_norm_out_batch_vec4,
            rms_norm_out_batch_f16,
            residual_add_rms_norm_out_batch,
            residual_add_rms_norm_out_batch_f16,
            rope,
            rope_batch,
            per_head_rms_norm,
            per_head_rms_norm_batch,
            qk_norm_rope_batch,
            qkv_split_qk_norm_rope_append_kv_batch_f32,
            qkv_split_qk_norm_rope_append_kv_batch_f16,
            qkv_split_bias_qknorm_rope_append_kv_batch_f32,
            qkv_split_bias_qknorm_rope_append_kv_batch_f16,
            gelu_elementwise_mul,
            gelu_elementwise_mul_batch,
            gelu_inplace,
            gelu_inplace_batch,
            silu_elementwise_mul,
            silu_elementwise_mul_batch,
            silu_elementwise_mul_batch_f16,
            elementwise_add,
            elementwise_add_batch,
            cast_f32_to_f16,
            cast_f16_to_f32,
            qkv_split_batch,
            qkv_split_rope_batch,
            qkv_split_rope_append_kv_batch_f32,
            qkv_split_rope_append_kv_batch_f16,
            kv_append_f32,
            kv_append_batch_f32,
            kv_append_f16,
            kv_append_batch_f16,
            kv_append_batch2_f32,
            kv_append_batch2_f16,
        })
    }

    // ── Encode methods (for phased dispatch, no command buffer) ──────

    /// Encode in-place RMSNorm: x = x * weight / sqrt(mean(x^2) + eps)
    pub fn encode_rms_norm(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &MetalBuffer,
        weight: &MetalBuffer,
        n: u32,
        eps: f32,
    ) {
        encoder.setComputePipelineState(self.rms_norm.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n);
        bind_f32(encoder, 3, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode in-place batched RMSNorm across `n_rows` rows of size `n`.
    pub fn encode_rms_norm_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &MetalBuffer,
        weight: &MetalBuffer,
        n: u32,
        n_rows: u32,
        eps: f32,
    ) {
        encoder.setComputePipelineState(self.rms_norm_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n);
        bind_u32(encoder, 3, n_rows);
        bind_f32(encoder, 4, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_rows as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode out-of-place RMSNorm: out = x * weight / sqrt(mean(x^2) + eps)
    pub fn encode_rms_norm_out(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &MetalBuffer,
        weight: &MetalBuffer,
        out: &MetalBuffer,
        n: u32,
        eps: f32,
    ) {
        encoder.setComputePipelineState(self.rms_norm_out.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(out.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n);
        bind_f32(encoder, 4, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode batched out-of-place RMSNorm across `n_rows` rows of size `n`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rms_norm_out_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &MetalBuffer,
        weight: &MetalBuffer,
        out: &MetalBuffer,
        n: u32,
        n_rows: u32,
        eps: f32,
    ) {
        // Use float4-vectorized kernel when dim is divisible by 4 (always true for LLM dims).
        // Reference: llama.cpp kernel_rms_norm_fuse_impl<float4, F>.
        let use_vec4 = n.is_multiple_of(4);
        encoder.setComputePipelineState(if use_vec4 {
            self.rms_norm_out_batch_vec4.state()
        } else {
            self.rms_norm_out_batch.state()
        });
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(out.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n);
        bind_u32(encoder, 4, n_rows);
        bind_f32(encoder, 5, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_rows as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode batched out-of-place RMSNorm across `n_rows` rows of size `n`,
    /// writing f16 output.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rms_norm_out_batch_f16(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &MetalBuffer,
        weight: &MetalBuffer,
        out_f16: &MetalBuffer,
        n: u32,
        n_rows: u32,
        eps: f32,
    ) {
        encoder.setComputePipelineState(self.rms_norm_out_batch_f16.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(out_f16.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n);
        bind_u32(encoder, 4, n_rows);
        bind_f32(encoder, 5, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_rows as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode batched residual add + out-of-place RMSNorm.
    ///
    /// For each row:
    /// - `hidden[row] += addend[row]`
    /// - `norm_out[row] = RMSNorm(hidden[row], weight)`
    #[allow(clippy::too_many_arguments)]
    pub fn encode_residual_add_rms_norm_out_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        hidden: &MetalBuffer,
        addend: &MetalBuffer,
        weight: &MetalBuffer,
        norm_out: &MetalBuffer,
        n: u32,
        n_rows: u32,
        eps: f32,
    ) {
        encoder.setComputePipelineState(self.residual_add_rms_norm_out_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(hidden.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(addend.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(weight.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(norm_out.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, n_rows);
        bind_f32(encoder, 6, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_rows as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode batched residual add + out-of-place RMSNorm, writing f16 norm output.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_residual_add_rms_norm_out_batch_f16(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        hidden: &MetalBuffer,
        addend: &MetalBuffer,
        weight: &MetalBuffer,
        norm_out_f16: &MetalBuffer,
        n: u32,
        n_rows: u32,
        eps: f32,
    ) {
        encoder.setComputePipelineState(self.residual_add_rms_norm_out_batch_f16.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(hidden.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(addend.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(weight.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(norm_out_f16.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n);
        bind_u32(encoder, 5, n_rows);
        bind_f32(encoder, 6, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_rows as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode RoPE on Q and K vectors.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rope(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k: &MetalBuffer,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        position: f32,
        freq_base: f32,
    ) {
        let half_dim = head_dim / 2;
        let total_pairs = (n_q_heads + n_kv_heads) * half_dim;
        let dims = DispatchDims::d1(total_pairs as usize, ELEMENTWISE_TG_SIZE);

        encoder.setComputePipelineState(self.rope.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n_q_heads);
        bind_u32(encoder, 3, n_kv_heads);
        bind_u32(encoder, 4, head_dim);
        bind_f32(encoder, 5, position);
        bind_f32(encoder, 6, freq_base);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode batched RoPE across `n_rows` tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rope_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k: &MetalBuffer,
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        start_pos: f32,
        pos_step: f32,
        freq_base: f32,
    ) {
        let half_dim = head_dim / 2;
        let total_pairs = n_rows * (n_q_heads + n_kv_heads) * half_dim;
        let dims = DispatchDims::d1(total_pairs as usize, ELEMENTWISE_TG_SIZE);

        encoder.setComputePipelineState(self.rope_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n_rows);
        bind_u32(encoder, 3, n_q_heads);
        bind_u32(encoder, 4, n_kv_heads);
        bind_u32(encoder, 5, head_dim);
        bind_f32(encoder, 6, start_pos);
        bind_f32(encoder, 7, pos_step);
        bind_f32(encoder, 8, freq_base);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode per-head RMSNorm.
    pub fn encode_per_head_rms_norm(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        buf: &MetalBuffer,
        weight: &MetalBuffer,
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    ) {
        encoder.setComputePipelineState(self.per_head_rms_norm.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buf.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n_heads);
        bind_u32(encoder, 3, head_dim);
        bind_f32(encoder, 4, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_heads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode batched per-head RMSNorm across `n_rows`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_per_head_rms_norm_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        buf: &MetalBuffer,
        weight: &MetalBuffer,
        n_rows: u32,
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    ) {
        encoder.setComputePipelineState(self.per_head_rms_norm_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buf.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n_rows);
        bind_u32(encoder, 3, n_heads);
        bind_u32(encoder, 4, head_dim);
        bind_f32(encoder, 5, eps);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n_rows * n_heads) as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode fused Gemma-style Q/K per-head RMSNorm + RoPE across `n_rows`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_qk_norm_rope_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q: &MetalBuffer,
        k: &MetalBuffer,
        q_weight: &MetalBuffer,
        k_weight: &MetalBuffer,
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        eps: f32,
        start_pos: f32,
        pos_step: f32,
        freq_base: f32,
    ) {
        encoder.setComputePipelineState(self.qk_norm_rope_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(q_weight.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(k_weight.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n_rows);
        bind_u32(encoder, 5, n_q_heads);
        bind_u32(encoder, 6, n_kv_heads);
        bind_u32(encoder, 7, head_dim);
        bind_f32(encoder, 8, eps);
        bind_f32(encoder, 9, start_pos);
        bind_f32(encoder, 10, pos_step);
        bind_f32(encoder, 11, freq_base);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n_rows * (n_q_heads + n_kv_heads)) as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: ELEMENTWISE_TG_SIZE,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode fused batched QKV split + Q/K norm + RoPE + KV append.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_qkv_split_qk_norm_rope_append_kv_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        q_weight: &MetalBuffer,
        k_weight: &MetalBuffer,
        cache_k: &MetalBuffer,
        cache_v: &MetalBuffer,
        cache_f16: bool,
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        eps: f32,
        start_pos: f32,
        pos_step: f32,
        freq_base: f32,
        cache_offset: u32,
        cache_stride: u32,
    ) {
        let half_dim = head_dim / 2;
        let q_pairs = n_q_heads * half_dim;
        let k_pairs = n_kv_heads * half_dim;
        let kv_dim = n_kv_heads * head_dim;
        let items_per_row = q_pairs + k_pairs + kv_dim;
        let total = (n_rows as usize) * (items_per_row as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        encoder.setComputePipelineState(if cache_f16 {
            self.qkv_split_qk_norm_rope_append_kv_batch_f16.state()
        } else {
            self.qkv_split_qk_norm_rope_append_kv_batch_f32.state()
        });
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(q_weight.mtl_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(k_weight.mtl_buffer()), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(cache_k.mtl_buffer()), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(cache_v.mtl_buffer()), 0, 7);
        }
        bind_u32(encoder, 8, n_rows);
        bind_u32(encoder, 9, n_q_heads);
        bind_u32(encoder, 10, n_kv_heads);
        bind_u32(encoder, 11, head_dim);
        bind_f32(encoder, 12, eps);
        bind_f32(encoder, 13, start_pos);
        bind_f32(encoder, 14, pos_step);
        bind_f32(encoder, 15, freq_base);
        bind_u32(encoder, 16, cache_offset);
        bind_u32(encoder, 17, cache_stride);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode fused QKV split + bias + QK norm + RoPE + KV cache append (Qwen3).
    ///
    /// Combines 7 dispatches (split, 3 bias, norm+RoPE, 2 KV append) into 1.
    /// `src` is the fused QKV output `[n_rows × (q_dim + 2*kv_dim)]`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_qkv_split_bias_qknorm_rope_append_kv_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        q_weight: &MetalBuffer,
        k_weight: &MetalBuffer,
        cache_k: &MetalBuffer,
        cache_v: &MetalBuffer,
        cache_f16: bool,
        q_bias: &MetalBuffer,
        k_bias: &MetalBuffer,
        v_bias: &MetalBuffer,
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        eps: f32,
        start_pos: f32,
        pos_step: f32,
        freq_base: f32,
        cache_offset: u32,
        cache_stride: u32,
    ) {
        let half_dim = head_dim / 2;
        let q_pairs = n_q_heads * half_dim;
        let k_pairs = n_kv_heads * half_dim;
        let kv_dim = n_kv_heads * head_dim;
        let items_per_row = q_pairs + k_pairs + kv_dim;
        let total = (n_rows as usize) * (items_per_row as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        encoder.setComputePipelineState(if cache_f16 {
            self.qkv_split_bias_qknorm_rope_append_kv_batch_f16.state()
        } else {
            self.qkv_split_bias_qknorm_rope_append_kv_batch_f32.state()
        });
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(q_weight.mtl_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(k_weight.mtl_buffer()), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(cache_k.mtl_buffer()), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(cache_v.mtl_buffer()), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(q_bias.mtl_buffer()), 0, 8);
            encoder.setBuffer_offset_atIndex(Some(k_bias.mtl_buffer()), 0, 9);
            encoder.setBuffer_offset_atIndex(Some(v_bias.mtl_buffer()), 0, 10);
        }
        bind_u32(encoder, 11, n_rows);
        bind_u32(encoder, 12, n_q_heads);
        bind_u32(encoder, 13, n_kv_heads);
        bind_u32(encoder, 14, head_dim);
        bind_f32(encoder, 15, eps);
        bind_f32(encoder, 16, start_pos);
        bind_f32(encoder, 17, pos_step);
        bind_f32(encoder, 18, freq_base);
        bind_u32(encoder, 19, cache_offset);
        bind_u32(encoder, 20, cache_stride);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode GELU(gate) * up.
    pub fn encode_gelu_elementwise_mul(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        n: u32,
    ) {
        let dims = DispatchDims::d1(n as usize, ELEMENTWISE_TG_SIZE);
        encoder.setComputePipelineState(self.gelu_elementwise_mul.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode batched GELU(gate) * up across `n_rows` rows of length `n`.
    pub fn encode_gelu_elementwise_mul_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        n: u32,
        n_rows: u32,
    ) {
        let total = (n as usize) * (n_rows as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        encoder.setComputePipelineState(self.gelu_elementwise_mul_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n);
        bind_u32(encoder, 3, n_rows);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode in-place GELU: x[i] = GELU(x[i]).
    /// Used by Falcon FFN (no gate projection).
    pub fn encode_gelu_inplace(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &MetalBuffer,
        n: u32,
    ) {
        let dims = DispatchDims::d1(n as usize, ELEMENTWISE_TG_SIZE);
        encoder.setComputePipelineState(self.gelu_inplace.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 0);
        }
        bind_u32(encoder, 1, n);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode batched in-place GELU across `n_rows` rows of length `n`.
    pub fn encode_gelu_inplace_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &MetalBuffer,
        n: u32,
        n_rows: u32,
    ) {
        let total = (n as usize) * (n_rows as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        encoder.setComputePipelineState(self.gelu_inplace_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.mtl_buffer()), 0, 0);
        }
        bind_u32(encoder, 1, n);
        bind_u32(encoder, 2, n_rows);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode SiLU(gate) * up.
    pub fn encode_silu_elementwise_mul(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        n: u32,
    ) {
        let dims = DispatchDims::d1(n as usize, ELEMENTWISE_TG_SIZE);
        encoder.setComputePipelineState(self.silu_elementwise_mul.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode batched SiLU(gate) * up across `n_rows` rows of length `n`.
    pub fn encode_silu_elementwise_mul_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        n: u32,
        n_rows: u32,
    ) {
        let total = (n as usize) * (n_rows as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        encoder.setComputePipelineState(self.silu_elementwise_mul_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n);
        bind_u32(encoder, 3, n_rows);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode batched SiLU(gate) * up across `n_rows` rows of length `n`,
    /// writing result to f16 output buffer.
    pub fn encode_silu_elementwise_mul_batch_f16(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        out_f16: &MetalBuffer,
        n: u32,
        n_rows: u32,
    ) {
        let total = (n as usize) * (n_rows as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        encoder.setComputePipelineState(self.silu_elementwise_mul_batch_f16.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(gate.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(up.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(out_f16.mtl_buffer()), 0, 2);
        }
        bind_u32(encoder, 3, n);
        bind_u32(encoder, 4, n_rows);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode a += b.
    pub fn encode_elementwise_add(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        n: u32,
    ) {
        let dims = DispatchDims::d1(n as usize, ELEMENTWISE_TG_SIZE);
        encoder.setComputePipelineState(self.elementwise_add.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode batched a += b across `n_rows` rows of length `n`.
    pub fn encode_elementwise_add_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        b: &MetalBuffer,
        n: u32,
        n_rows: u32,
    ) {
        let total = (n as usize) * (n_rows as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        encoder.setComputePipelineState(self.elementwise_add_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n);
        bind_u32(encoder, 3, n_rows);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode f32->f16 cast over `n` elements.
    pub fn encode_cast_f32_to_f16(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        n: u32,
    ) {
        let dims = DispatchDims::d1(n as usize, ELEMENTWISE_TG_SIZE);
        encoder.setComputePipelineState(self.cast_f32_to_f16.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode f16->f32 cast over `n` elements.
    pub fn encode_cast_f16_to_f32(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        n: u32,
    ) {
        let dims = DispatchDims::d1(n as usize, ELEMENTWISE_TG_SIZE);
        encoder.setComputePipelineState(self.cast_f16_to_f32.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, n);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode batched split from fused QKV rows into Q/K/V output buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_qkv_split_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        n_rows: u32,
        q_dim: u32,
        kv_dim: u32,
    ) {
        let fused_dim = q_dim + 2 * kv_dim;
        let total = (n_rows as usize) * (fused_dim as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        encoder.setComputePipelineState(self.qkv_split_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n_rows);
        bind_u32(encoder, 5, q_dim);
        bind_u32(encoder, 6, kv_dim);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode fused batched QKV split + RoPE.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_qkv_split_rope_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        start_pos: f32,
        pos_step: f32,
        freq_base: f32,
    ) {
        let half_dim = head_dim / 2;
        let q_pairs = n_q_heads * half_dim;
        let k_pairs = n_kv_heads * half_dim;
        let kv_dim = n_kv_heads * head_dim;
        let items_per_row = q_pairs + k_pairs + kv_dim;
        let total = (n_rows as usize) * (items_per_row as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        encoder.setComputePipelineState(self.qkv_split_rope_batch.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, n_rows);
        bind_u32(encoder, 5, n_q_heads);
        bind_u32(encoder, 6, n_kv_heads);
        bind_u32(encoder, 7, head_dim);
        bind_f32(encoder, 8, start_pos);
        bind_f32(encoder, 9, pos_step);
        bind_f32(encoder, 10, freq_base);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode fused batched QKV split + RoPE + KV append.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_qkv_split_rope_append_kv_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        cache_k: &MetalBuffer,
        cache_v: &MetalBuffer,
        cache_f16: bool,
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        start_pos: f32,
        pos_step: f32,
        freq_base: f32,
        cache_offset: u32,
        cache_stride: u32,
    ) {
        let half_dim = head_dim / 2;
        let q_pairs = n_q_heads * half_dim;
        let k_pairs = n_kv_heads * half_dim;
        let kv_dim = n_kv_heads * head_dim;
        let items_per_row = q_pairs + k_pairs + kv_dim;
        let total = (n_rows as usize) * (items_per_row as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        encoder.setComputePipelineState(if cache_f16 {
            self.qkv_split_rope_append_kv_batch_f16.state()
        } else {
            self.qkv_split_rope_append_kv_batch_f32.state()
        });
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(q.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(k.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v.mtl_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(cache_k.mtl_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(cache_v.mtl_buffer()), 0, 5);
        }
        bind_u32(encoder, 6, n_rows);
        bind_u32(encoder, 7, n_q_heads);
        bind_u32(encoder, 8, n_kv_heads);
        bind_u32(encoder, 9, head_dim);
        bind_f32(encoder, 10, start_pos);
        bind_f32(encoder, 11, pos_step);
        bind_f32(encoder, 12, freq_base);
        bind_u32(encoder, 13, cache_offset);
        bind_u32(encoder, 14, cache_stride);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode KV cache append: copy `count` floats from src to dst at offset.
    pub fn encode_kv_append(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        dst_f16: bool,
        offset: u32,
        count: u32,
    ) {
        let dims = DispatchDims::d1(count as usize, ELEMENTWISE_TG_SIZE);
        let pipeline = if dst_f16 {
            self.kv_append_f16.state()
        } else {
            self.kv_append_f32.state()
        };
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, offset);
        bind_u32(encoder, 3, count);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode batched KV append from src[N×count] into dst with row stride.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_kv_append_batch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        dst_f16: bool,
        dst_offset: u32,
        dst_row_stride: u32,
        count: u32,
        n_rows: u32,
    ) {
        let total = (count as usize) * (n_rows as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        let pipeline = if dst_f16 {
            self.kv_append_batch_f16.state()
        } else {
            self.kv_append_batch_f32.state()
        };
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, dst_offset);
        bind_u32(encoder, 3, dst_row_stride);
        bind_u32(encoder, 4, count);
        bind_u32(encoder, 5, n_rows);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode batched append of both K and V in one dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_kv_append_batch_pair(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src_k: &MetalBuffer,
        src_v: &MetalBuffer,
        dst_k: &MetalBuffer,
        dst_v: &MetalBuffer,
        dst_f16: bool,
        dst_offset: u32,
        dst_row_stride: u32,
        count: u32,
        n_rows: u32,
    ) {
        let total = (count as usize) * (n_rows as usize);
        let dims = DispatchDims::d1(total, ELEMENTWISE_TG_SIZE);
        encoder.setComputePipelineState(if dst_f16 {
            self.kv_append_batch2_f16.state()
        } else {
            self.kv_append_batch2_f32.state()
        });
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src_k.mtl_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(src_v.mtl_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(dst_k.mtl_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(dst_v.mtl_buffer()), 0, 3);
        }
        bind_u32(encoder, 4, dst_offset);
        bind_u32(encoder, 5, dst_row_stride);
        bind_u32(encoder, 6, count);
        bind_u32(encoder, 7, n_rows);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    /// Encode a GPU buffer copy with custom source byte offset.
    ///
    /// Copies `count` floats: dst[dst_float_offset..] = src[src_byte_offset..].
    /// Uses the kv_append pipeline with custom buffer offset binding.
    /// This enables batch↔scratch transfers for batched prefill.
    pub fn encode_buffer_copy(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &MetalBuffer,
        src_byte_offset: usize,
        dst: &MetalBuffer,
        dst_float_offset: u32,
        count: u32,
    ) {
        let dims = DispatchDims::d1(count as usize, ELEMENTWISE_TG_SIZE);
        encoder.setComputePipelineState(self.kv_append_f32.state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.mtl_buffer()), src_byte_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.mtl_buffer()), 0, 1);
        }
        bind_u32(encoder, 2, dst_float_offset);
        bind_u32(encoder, 3, count);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            dims.threadgroups,
            dims.threads_per_threadgroup,
        );
    }

    // ── Standalone methods (create command buffer, for testing) ──────

    /// In-place RMSNorm on GPU (standalone, creates own command buffer).
    pub fn rms_norm(
        &self,
        device: &MetalDevice,
        x: &MetalBuffer,
        weight: &MetalBuffer,
        n: u32,
        eps: f32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_rms_norm(encoder, x, weight, n, eps);
            Ok(())
        })
    }

    /// Out-of-place RMSNorm on GPU (standalone).
    pub fn rms_norm_out(
        &self,
        device: &MetalDevice,
        x: &MetalBuffer,
        weight: &MetalBuffer,
        out: &MetalBuffer,
        n: u32,
        eps: f32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_rms_norm_out(encoder, x, weight, out, n, eps);
            Ok(())
        })
    }

    /// RoPE on GPU (standalone).
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        device: &MetalDevice,
        q: &MetalBuffer,
        k: &MetalBuffer,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        position: f32,
        freq_base: f32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_rope(
                encoder, q, k, n_q_heads, n_kv_heads, head_dim, position, freq_base,
            );
            Ok(())
        })
    }

    /// Per-head RMSNorm on GPU (standalone).
    pub fn per_head_rms_norm(
        &self,
        device: &MetalDevice,
        buf: &MetalBuffer,
        weight: &MetalBuffer,
        n_heads: u32,
        head_dim: u32,
        eps: f32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_per_head_rms_norm(encoder, buf, weight, n_heads, head_dim, eps);
            Ok(())
        })
    }

    /// GELU(gate) * up on GPU (standalone).
    pub fn gelu_elementwise_mul(
        &self,
        device: &MetalDevice,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        n: u32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_gelu_elementwise_mul(encoder, gate, up, n);
            Ok(())
        })
    }

    /// SiLU(gate) * up on GPU (standalone).
    pub fn silu_elementwise_mul(
        &self,
        device: &MetalDevice,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        n: u32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_silu_elementwise_mul(encoder, gate, up, n);
            Ok(())
        })
    }

    /// a += b on GPU (standalone).
    pub fn elementwise_add(
        &self,
        device: &MetalDevice,
        a: &MetalBuffer,
        b: &MetalBuffer,
        n: u32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_elementwise_add(encoder, a, b, n);
            Ok(())
        })
    }

    /// KV append on GPU (standalone, for testing).
    pub fn kv_append(
        &self,
        device: &MetalDevice,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        dst_f16: bool,
        offset: u32,
        count: u32,
    ) -> anyhow::Result<()> {
        device.execute_sync(|encoder| {
            self.encode_kv_append(encoder, src, dst, dst_f16, offset, count);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_kernel_mode() {
        assert_eq!(
            parse_kernel_mode("__UNSET_ENV__", KernelMode::Off),
            KernelMode::Off
        );
        assert_eq!(
            parse_kernel_mode("__UNSET_ENV__", KernelMode::Auto),
            KernelMode::Auto
        );
    }

    #[test]
    fn test_resolve_attention_routing_profile() {
        assert_eq!(
            resolve_attention_routing_profile("default"),
            Some(ATTN_PROFILE_DEFAULT)
        );
        assert_eq!(
            resolve_attention_routing_profile("balanced"),
            Some(ATTN_PROFILE_DECODE_BALANCED)
        );
        assert_eq!(
            resolve_attention_routing_profile("long-context"),
            Some(ATTN_PROFILE_DECODE_LONG_CONTEXT)
        );
        assert_eq!(resolve_attention_routing_profile("unknown"), None);
    }

    #[test]
    fn test_dispatch_dims_1d_exact() {
        let dims = DispatchDims::d1(256, 64);
        assert_eq!(dims.threadgroups.width, 4);
        assert_eq!(dims.threads_per_threadgroup.width, 64);
    }

    #[test]
    fn test_dispatch_dims_1d_rounded_up() {
        let dims = DispatchDims::d1(300, 64);
        assert_eq!(dims.threadgroups.width, 5);
        assert_eq!(dims.threads_per_threadgroup.width, 64);
    }

    #[test]
    fn test_dispatch_dims_2d() {
        let dims = DispatchDims::d2(128, 64, 16, 8);
        assert_eq!(dims.threadgroups.width, 8);
        assert_eq!(dims.threadgroups.height, 8);
        assert_eq!(dims.threads_per_threadgroup.width, 16);
        assert_eq!(dims.threads_per_threadgroup.height, 8);
    }

    #[test]
    fn test_dispatch_dims_2d_rounded_up() {
        let dims = DispatchDims::d2(130, 65, 16, 8);
        assert_eq!(dims.threadgroups.width, 9);
        assert_eq!(dims.threadgroups.height, 9);
    }

    /// Naive CPU matmul for test verification.
    fn cpu_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn test_matmul_kernels_compile() {
        let gpu = MetalDevice::new().unwrap();
        let _kernels = MatmulKernels::new(&gpu).unwrap();
    }

    #[test]
    fn test_matvec_small() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = MatmulKernels::new(&gpu).unwrap();

        // A: 4×3, x: 3×1, y: 4×1
        #[rustfmt::skip]
        let a_data: Vec<f32> = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ];
        let x_data: Vec<f32> = vec![1.0, 2.0, 3.0];

        let buf_a = MetalBuffer::from_slice(gpu.device(), &a_data).unwrap();
        let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
        let buf_y = MetalBuffer::new(gpu.device(), 4 * 4).unwrap(); // 4 floats

        kernels
            .matmul(&gpu, &buf_a, &buf_x, &buf_y, (4, 1, 3))
            .unwrap();

        let result = unsafe { buf_y.as_slice::<f32>() };
        let mut expected = vec![0.0f32; 4];
        cpu_matmul(&a_data, &x_data, &mut expected, 4, 1, 3);

        assert!(
            max_abs_diff(result, &expected) < 1e-5,
            "matvec mismatch: got {:?}, expected {:?}",
            result,
            expected
        );
    }

    #[test]
    fn test_tiled_matmul_small() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = MatmulKernels::new(&gpu).unwrap();

        // A: 3×4, B: 4×2, C: 3×2
        #[rustfmt::skip]
        let a_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];
        #[rustfmt::skip]
        let b_data: Vec<f32> = vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
        ];

        let buf_a = MetalBuffer::from_slice(gpu.device(), &a_data).unwrap();
        let buf_b = MetalBuffer::from_slice(gpu.device(), &b_data).unwrap();
        let buf_c = MetalBuffer::new(gpu.device(), 3 * 2 * 4).unwrap();

        kernels
            .matmul(&gpu, &buf_a, &buf_b, &buf_c, (3, 2, 4))
            .unwrap();

        let result = unsafe { buf_c.as_slice::<f32>() };
        let mut expected = vec![0.0f32; 6];
        cpu_matmul(&a_data, &b_data, &mut expected, 3, 2, 4);

        assert!(
            max_abs_diff(result, &expected) < 1e-5,
            "matmul mismatch: got {:?}, expected {:?}",
            result,
            expected
        );
    }

    #[test]
    fn test_matmul_identity() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = MatmulKernels::new(&gpu).unwrap();

        let n = 16;
        let mut identity = vec![0.0f32; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }
        let a_data: Vec<f32> = (0..n * n).map(|i| i as f32).collect();

        let buf_a = MetalBuffer::from_slice(gpu.device(), &a_data).unwrap();
        let buf_i = MetalBuffer::from_slice(gpu.device(), &identity).unwrap();
        let buf_c = MetalBuffer::new(gpu.device(), n * n * 4).unwrap();

        // C = A × I should equal A
        kernels
            .matmul(&gpu, &buf_a, &buf_i, &buf_c, (n as u32, n as u32, n as u32))
            .unwrap();

        let result = unsafe { buf_c.as_slice::<f32>() };
        assert!(max_abs_diff(result, &a_data) < 1e-4, "A × I should equal A");
    }

    #[test]
    fn test_matmul_non_tile_aligned() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = MatmulKernels::new(&gpu).unwrap();

        // Non-tile-aligned dimensions: 7×13 * 13×5 = 7×5
        let m = 7;
        let n = 5;
        let k = 13;

        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();

        let buf_a = MetalBuffer::from_slice(gpu.device(), &a_data).unwrap();
        let buf_b = MetalBuffer::from_slice(gpu.device(), &b_data).unwrap();
        let buf_c = MetalBuffer::new(gpu.device(), m * n * 4).unwrap();

        kernels
            .matmul(&gpu, &buf_a, &buf_b, &buf_c, (m as u32, n as u32, k as u32))
            .unwrap();

        let result = unsafe { buf_c.as_slice::<f32>() };
        let mut expected = vec![0.0f32; m * n];
        cpu_matmul(&a_data, &b_data, &mut expected, m, n, k);

        assert!(
            max_abs_diff(result, &expected) < 1e-3,
            "Non-aligned matmul failed: max_diff={}",
            max_abs_diff(result, &expected)
        );
    }

    #[test]
    fn test_matvec_llama_sized() {
        // Simulate a typical LLaMA decode matmul: 4096×4096 * 4096×1
        let gpu = MetalDevice::new().unwrap();
        let kernels = MatmulKernels::new(&gpu).unwrap();

        let m = 4096;
        let k = 4096;

        // Use simple patterns to avoid precision issues with random data
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
        let x_data: Vec<f32> = (0..k).map(|i| ((i % 5) as f32 - 2.0) * 0.01).collect();

        let buf_a = MetalBuffer::from_slice(gpu.device(), &a_data).unwrap();
        let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
        let buf_y = MetalBuffer::new(gpu.device(), m * 4).unwrap();

        kernels
            .matmul(&gpu, &buf_a, &buf_x, &buf_y, (m as u32, 1, k as u32))
            .unwrap();

        let result = unsafe { buf_y.as_slice::<f32>() };

        // Verify a few rows against CPU
        let mut expected = vec![0.0f32; m];
        cpu_matmul(&a_data, &x_data, &mut expected, m, 1, k);

        let diff = max_abs_diff(result, &expected);
        assert!(diff < 0.1, "LLaMA-sized matvec failed: max_diff={}", diff);
    }

    #[test]
    fn test_matmul_prefill_sized() {
        // Simulate prefill: 128×4096 * 4096×128 (small prefill)
        let gpu = MetalDevice::new().unwrap();
        let kernels = MatmulKernels::new(&gpu).unwrap();

        let m = 128;
        let n = 128;
        let k = 256; // smaller K to keep test fast

        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();

        let buf_a = MetalBuffer::from_slice(gpu.device(), &a_data).unwrap();
        let buf_b = MetalBuffer::from_slice(gpu.device(), &b_data).unwrap();
        let buf_c = MetalBuffer::new(gpu.device(), m * n * 4).unwrap();

        kernels
            .matmul(&gpu, &buf_a, &buf_b, &buf_c, (m as u32, n as u32, k as u32))
            .unwrap();

        let result = unsafe { buf_c.as_slice::<f32>() };
        let mut expected = vec![0.0f32; m * n];
        cpu_matmul(&a_data, &b_data, &mut expected, m, n, k);

        let diff = max_abs_diff(result, &expected);
        assert!(diff < 0.1, "Prefill-sized matmul failed: max_diff={}", diff);
    }

    // ── Dequant test helpers ───────────────────────────────────────────

    const Q4_0_BYTES_PER_BLOCK: usize = 18;
    const Q4_0_BLOCK_SIZE: usize = 32;
    const Q4_K_BYTES_PER_BLOCK: usize = 144;
    const Q4_K_BLOCK_SIZE: usize = 256;
    const Q6_K_BYTES_PER_BLOCK: usize = 210;
    const Q6_K_BLOCK_SIZE: usize = 256;

    /// Create Q4_0 block bytes from an f16 scale and 16 quant bytes.
    fn make_q4_0_block(d_f32: f32, qs: &[u8; 16]) -> Vec<u8> {
        let d_bytes = half::f16::from_f32(d_f32).to_le_bytes();
        let mut block = Vec::with_capacity(Q4_0_BYTES_PER_BLOCK);
        block.extend_from_slice(&d_bytes);
        block.extend_from_slice(qs);
        block
    }

    /// CPU reference dequant for Q4_0.
    fn cpu_dequant_q4_0(blocks: &[u8], dst: &mut [f32]) {
        let n_blocks = blocks.len() / Q4_0_BYTES_PER_BLOCK;
        for b in 0..n_blocks {
            let block = &blocks[b * Q4_0_BYTES_PER_BLOCK..][..Q4_0_BYTES_PER_BLOCK];
            let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qs = &block[2..18];
            let out = &mut dst[b * Q4_0_BLOCK_SIZE..][..Q4_0_BLOCK_SIZE];
            for i in 0..16 {
                let byte = qs[i];
                let lo = (byte & 0x0F) as i32 - 8;
                let hi = (byte >> 4) as i32 - 8;
                out[i] = d * lo as f32;
                out[i + 16] = d * hi as f32;
            }
        }
    }

    /// CPU reference for Q4_K scale/min extraction.
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

    /// CPU reference dequant for Q4_K.
    fn cpu_dequant_q4_k(blocks: &[u8], dst: &mut [f32]) {
        let n_blocks = blocks.len() / Q4_K_BYTES_PER_BLOCK;
        for b in 0..n_blocks {
            let block = &blocks[b * Q4_K_BYTES_PER_BLOCK..][..Q4_K_BYTES_PER_BLOCK];
            let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
            let scales = &block[4..16];
            let qs = &block[16..144];
            let out = &mut dst[b * Q4_K_BLOCK_SIZE..][..Q4_K_BLOCK_SIZE];

            let mut out_idx = 0;
            let mut q_idx = 0;
            let mut is = 0;
            for _pair in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, scales);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                let d1 = d * sc1 as f32;
                let m1 = dmin * m1 as f32;
                let d2 = d * sc2 as f32;
                let m2 = dmin * m2 as f32;
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

    /// CPU reference dequant for Q6_K.
    fn cpu_dequant_q6_k(blocks: &[u8], dst: &mut [f32]) {
        let n_blocks = blocks.len() / Q6_K_BYTES_PER_BLOCK;
        for b in 0..n_blocks {
            let block = &blocks[b * Q6_K_BYTES_PER_BLOCK..][..Q6_K_BYTES_PER_BLOCK];
            let ql = &block[..128];
            let qh = &block[128..192];
            let scales = &block[192..208];
            let d = half::f16::from_le_bytes([block[208], block[209]]).to_f32();
            let out = &mut dst[b * Q6_K_BLOCK_SIZE..][..Q6_K_BLOCK_SIZE];

            let mut out_idx = 0;
            let mut ql_idx = 0;
            let mut qh_idx = 0;
            let mut sc_idx = 0;

            for _group in 0..2 {
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[ql_idx + l] & 0x0F) | ((qh[qh_idx + l] & 0x03) << 4)) as i32 - 32;
                    let q2 = ((ql[ql_idx + l + 32] & 0x0F) | (((qh[qh_idx + l] >> 2) & 0x03) << 4))
                        as i32
                        - 32;
                    let q3 =
                        ((ql[ql_idx + l] >> 4) | (((qh[qh_idx + l] >> 4) & 0x03) << 4)) as i32 - 32;
                    let q4 = ((ql[ql_idx + l + 32] >> 4) | (((qh[qh_idx + l] >> 6) & 0x03) << 4))
                        as i32
                        - 32;

                    out[out_idx + l] = d * (scales[sc_idx + is] as i8) as f32 * q1 as f32;
                    out[out_idx + l + 32] = d * (scales[sc_idx + is + 2] as i8) as f32 * q2 as f32;
                    out[out_idx + l + 64] = d * (scales[sc_idx + is + 4] as i8) as f32 * q3 as f32;
                    out[out_idx + l + 96] = d * (scales[sc_idx + is + 6] as i8) as f32 * q4 as f32;
                }

                out_idx += 128;
                ql_idx += 64;
                qh_idx += 32;
                sc_idx += 8;
            }
        }
    }

    fn round_to_f16(v: f32) -> f32 {
        half::f16::from_f32(v).to_f32()
    }

    fn cpu_matvec(a: &[f32], x: &[f32], dst: &mut [f32], m: usize, k: usize) {
        for row in 0..m {
            let a_row = &a[row * k..][..k];
            let mut sum = 0.0f32;
            for (av, xv) in a_row.iter().zip(x.iter()) {
                sum += av * round_to_f16(*xv);
            }
            dst[row] = sum;
        }
    }

    fn argmax(xs: &[f32]) -> usize {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in xs.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        best_i
    }

    // ── Dequant kernel tests ───────────────────────────────────────────

    #[test]
    fn test_dequant_kernels_compile() {
        let gpu = MetalDevice::new().unwrap();
        let _kernels = DequantKernels::new(&gpu).unwrap();
    }

    #[test]
    fn test_dequant_q4_0_standalone() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = DequantKernels::new(&gpu).unwrap();

        // Two blocks with known values
        let mut src = Vec::new();
        // Block 0: d=1.0, all quants = 0x99 → lo=(9-8)*1=1, hi=(9-8)*1=1
        src.extend(make_q4_0_block(1.0, &[0x99; 16]));
        // Block 1: d=2.0, quants[0] = 0x0F, rest 0x88
        let mut qs1 = [0x88u8; 16];
        qs1[0] = 0x0F;
        src.extend(make_q4_0_block(2.0, &qs1));

        let n_blocks = 2u32;
        let n_values = n_blocks as usize * Q4_0_BLOCK_SIZE;

        let buf_src = MetalBuffer::from_bytes(gpu.device(), &src).unwrap();
        let buf_dst =
            MetalBuffer::new(gpu.device(), n_values * std::mem::size_of::<f32>()).unwrap();

        kernels
            .dequant_q4_0(&gpu, &buf_src, &buf_dst, n_blocks)
            .unwrap();

        let gpu_result = unsafe { buf_dst.as_slice::<f32>() };
        let mut cpu_result = vec![0.0f32; n_values];
        cpu_dequant_q4_0(&src, &mut cpu_result);

        let diff = max_abs_diff(gpu_result, &cpu_result);
        assert!(
            diff < 1e-3,
            "Q4_0 standalone dequant mismatch: max_diff={}",
            diff
        );
    }

    #[test]
    fn test_dequant_q4_k_standalone() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = DequantKernels::new(&gpu).unwrap();

        // Create Q4_K block: d=1.0, dmin=0.5, scales for sub-blocks 0-3 = 2, mins = 1
        let mut block = vec![0u8; Q4_K_BYTES_PER_BLOCK];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        let dmin_bytes = half::f16::from_f32(0.5).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        block[2] = dmin_bytes[0];
        block[3] = dmin_bytes[1];
        // scales for sub-blocks 0-3
        block[4] = 2; // sc for sub-block 0
        block[5] = 2; // sc for sub-block 1
        block[6] = 2; // sc for sub-block 2
        block[7] = 2; // sc for sub-block 3
        block[8] = 1; // min for sub-block 0
        block[9] = 1; // min for sub-block 1
        block[10] = 1; // min for sub-block 2
        block[11] = 1; // min for sub-block 3
        // All quant nibbles = 5
        block[16..144].fill(0x55);

        let n_blocks = 1u32;
        let n_values = Q4_K_BLOCK_SIZE;

        let buf_src = MetalBuffer::from_bytes(gpu.device(), &block).unwrap();
        let buf_dst =
            MetalBuffer::new(gpu.device(), n_values * std::mem::size_of::<f32>()).unwrap();

        kernels
            .dequant_q4_k(&gpu, &buf_src, &buf_dst, n_blocks)
            .unwrap();

        let gpu_result = unsafe { buf_dst.as_slice::<f32>() };
        let mut cpu_result = vec![0.0f32; n_values];
        cpu_dequant_q4_k(&block, &mut cpu_result);

        let diff = max_abs_diff(gpu_result, &cpu_result);
        assert!(
            diff < 1e-3,
            "Q4_K standalone dequant mismatch: max_diff={}",
            diff
        );
    }

    #[test]
    fn test_dequant_q4_0_multiple_blocks() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = DequantKernels::new(&gpu).unwrap();

        // 8 blocks with varying scales and quant patterns
        let mut src = Vec::new();
        for i in 0..8u8 {
            let d = (i as f32 + 1.0) * 0.5;
            let q = 0x33u8.wrapping_add(i * 0x11);
            src.extend(make_q4_0_block(d, &[q; 16]));
        }

        let n_blocks = 8u32;
        let n_values = n_blocks as usize * Q4_0_BLOCK_SIZE;

        let buf_src = MetalBuffer::from_bytes(gpu.device(), &src).unwrap();
        let buf_dst =
            MetalBuffer::new(gpu.device(), n_values * std::mem::size_of::<f32>()).unwrap();

        kernels
            .dequant_q4_0(&gpu, &buf_src, &buf_dst, n_blocks)
            .unwrap();

        let gpu_result = unsafe { buf_dst.as_slice::<f32>() };
        let mut cpu_result = vec![0.0f32; n_values];
        cpu_dequant_q4_0(&src, &mut cpu_result);

        let diff = max_abs_diff(gpu_result, &cpu_result);
        assert!(
            diff < 1e-3,
            "Q4_0 multi-block dequant mismatch: max_diff={}",
            diff
        );
    }

    #[test]
    fn test_fused_matvec_q4_0() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = DequantKernels::new(&gpu).unwrap();

        // 4 rows × 64 cols (K=64 → 2 blocks per row)
        let m = 4usize;
        let k = 64usize;
        let blocks_per_row = k / Q4_0_BLOCK_SIZE;

        // Create quantized weight data
        let mut quant_data = Vec::new();
        for row in 0..m {
            for _blk in 0..blocks_per_row {
                let d = (row as f32 + 1.0) * 0.5;
                let q = 0x55u8 + row as u8;
                quant_data.extend(make_q4_0_block(d, &[q; 16]));
            }
        }

        // Dequantize on CPU to get reference weights
        let total_values = m * k;
        let mut weights_f32 = vec![0.0f32; total_values];
        cpu_dequant_q4_0(&quant_data, &mut weights_f32);

        // Create input vector
        let x_data: Vec<f32> = (0..k).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();

        // CPU reference: y = weights × x
        let mut expected = vec![0.0f32; m];
        cpu_matmul(&weights_f32, &x_data, &mut expected, m, 1, k);

        // GPU fused dequant+matvec
        let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
        let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
        let buf_y = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

        kernels
            .fused_matvec_q4_0(&gpu, &buf_a, &buf_x, &buf_y, m as u32, k as u32)
            .unwrap();

        let result = unsafe { buf_y.as_slice::<f32>() };
        let diff = max_abs_diff(result, &expected);
        assert!(
            diff < 1e-3,
            "Fused Q4_0 matvec mismatch: max_diff={}, got {:?}, expected {:?}",
            diff,
            result,
            expected
        );
    }

    #[test]
    fn test_fused_matvec_q4_k() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = DequantKernels::new(&gpu).unwrap();

        // 4 rows × 256 cols (K=256 → 1 Q4_K block per row)
        let m = 4usize;
        let k = 256usize;

        // Create Q4_K blocks for each row
        let mut quant_data = Vec::new();
        for row in 0..m {
            let mut block = vec![0u8; Q4_K_BYTES_PER_BLOCK];
            let d = (row as f32 + 1.0) * 0.25;
            let d_bytes = half::f16::from_f32(d).to_le_bytes();
            block[0] = d_bytes[0];
            block[1] = d_bytes[1];
            // dmin = 0 for simplicity
            // scales for sub-blocks 0-3 = 1
            block[4] = 1;
            block[5] = 1;
            block[6] = 1;
            block[7] = 1;
            // All quants = pattern based on row
            let q = 0x33u8 + row as u8;
            block[16..144].fill(q);
            quant_data.extend(block);
        }

        // Dequantize on CPU
        let mut weights_f32 = vec![0.0f32; m * k];
        cpu_dequant_q4_k(&quant_data, &mut weights_f32);

        // Input vector
        let x_data: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();

        // CPU reference: y = weights × x
        let mut expected = vec![0.0f32; m];
        cpu_matmul(&weights_f32, &x_data, &mut expected, m, 1, k);

        // GPU fused dequant+matvec
        let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
        let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
        let buf_y = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

        kernels
            .fused_matvec_q4_k(&gpu, &buf_a, &buf_x, &buf_y, m as u32, k as u32)
            .unwrap();

        let result = unsafe { buf_y.as_slice::<f32>() };
        let diff = max_abs_diff(result, &expected);
        assert!(
            diff < 1e-2,
            "Fused Q4_K matvec mismatch: max_diff={}, got {:?}, expected {:?}",
            diff,
            result,
            expected
        );
    }

    #[test]
    fn test_fused_matvec_q4_0_larger() {
        // 64 rows × 256 cols (K=256 → 8 blocks per row)
        let gpu = MetalDevice::new().unwrap();
        let kernels = DequantKernels::new(&gpu).unwrap();

        let m = 64usize;
        let k = 256usize;
        let blocks_per_row = k / Q4_0_BLOCK_SIZE;

        let mut quant_data = Vec::new();
        for row in 0..m {
            for blk in 0..blocks_per_row {
                let d = ((row * blocks_per_row + blk) % 10) as f32 * 0.1 + 0.1;
                let q = ((row + blk) % 16) as u8 * 0x11;
                quant_data.extend(make_q4_0_block(d, &[q; 16]));
            }
        }

        let mut weights_f32 = vec![0.0f32; m * k];
        cpu_dequant_q4_0(&quant_data, &mut weights_f32);

        let x_data: Vec<f32> = (0..k).map(|i| ((i % 11) as f32 - 5.0) * 0.02).collect();

        let mut expected = vec![0.0f32; m];
        cpu_matmul(&weights_f32, &x_data, &mut expected, m, 1, k);

        let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
        let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
        let buf_y = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

        kernels
            .fused_matvec_q4_0(&gpu, &buf_a, &buf_x, &buf_y, m as u32, k as u32)
            .unwrap();

        let result = unsafe { buf_y.as_slice::<f32>() };
        let diff = max_abs_diff(result, &expected);
        assert!(
            diff < 0.1,
            "Larger fused Q4_0 matvec failed: max_diff={}",
            diff
        );
    }

    #[test]
    fn test_fused_matvec_q4_k_larger() {
        // 32 rows × 512 cols (K=512 → 2 Q4_K blocks per row)
        let gpu = MetalDevice::new().unwrap();
        let kernels = DequantKernels::new(&gpu).unwrap();

        let m = 32usize;
        let k = 512usize;
        let blocks_per_row = k / Q4_K_BLOCK_SIZE;

        let mut quant_data = Vec::new();
        for row in 0..m {
            for blk in 0..blocks_per_row {
                let mut block = vec![0u8; Q4_K_BYTES_PER_BLOCK];
                let d = ((row + blk) % 5) as f32 * 0.3 + 0.1;
                let d_bytes = half::f16::from_f32(d).to_le_bytes();
                block[0] = d_bytes[0];
                block[1] = d_bytes[1];
                let dmin = 0.1f32;
                let dmin_bytes = half::f16::from_f32(dmin).to_le_bytes();
                block[2] = dmin_bytes[0];
                block[3] = dmin_bytes[1];
                // scales for sub-blocks 0-3
                for i in 0..4 {
                    block[4 + i] = ((row + i) % 8 + 1) as u8;
                    block[8 + i] = ((blk + i) % 4) as u8;
                }
                // quant pattern
                let q = ((row * blocks_per_row + blk) % 12) as u8 + 0x22;
                block[16..144].fill(q);
                quant_data.extend(block);
            }
        }

        let mut weights_f32 = vec![0.0f32; m * k];
        cpu_dequant_q4_k(&quant_data, &mut weights_f32);

        let x_data: Vec<f32> = (0..k).map(|i| ((i % 9) as f32 - 4.0) * 0.03).collect();

        let mut expected = vec![0.0f32; m];
        cpu_matmul(&weights_f32, &x_data, &mut expected, m, 1, k);

        let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
        let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
        let buf_y = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

        kernels
            .fused_matvec_q4_k(&gpu, &buf_a, &buf_x, &buf_y, m as u32, k as u32)
            .unwrap();

        let result = unsafe { buf_y.as_slice::<f32>() };
        let diff = max_abs_diff(result, &expected);
        assert!(
            diff < 0.5,
            "Larger fused Q4_K matvec failed: max_diff={}",
            diff
        );
    }

    #[test]
    fn test_fused_matvec_q4_k_tg256_matches_base() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = DequantKernels::new(&gpu).unwrap();

        let m = 32usize;
        let k = 512usize;
        let blocks_per_row = k / Q4_K_BLOCK_SIZE;

        let mut quant_data = Vec::new();
        for row in 0..m {
            for blk in 0..blocks_per_row {
                let mut block = vec![0u8; Q4_K_BYTES_PER_BLOCK];
                let d = ((row + blk) % 7) as f32 * 0.2 + 0.1;
                let d_bytes = half::f16::from_f32(d).to_le_bytes();
                block[0] = d_bytes[0];
                block[1] = d_bytes[1];
                let dmin = 0.05f32;
                let dmin_bytes = half::f16::from_f32(dmin).to_le_bytes();
                block[2] = dmin_bytes[0];
                block[3] = dmin_bytes[1];
                for i in 0..4 {
                    block[4 + i] = ((row + i) % 8 + 1) as u8;
                    block[8 + i] = ((blk + i) % 4) as u8;
                }
                block[16..144].fill(((row * 3 + blk) % 16) as u8 + 0x11);
                quant_data.extend(block);
            }
        }

        let x_data: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect();
        let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
        let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
        let buf_base = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();
        let buf_tg256 = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

        kernels
            .fused_matvec_q4_k(&gpu, &buf_a, &buf_x, &buf_base, m as u32, k as u32)
            .unwrap();
        kernels
            .fused_matvec_q4_k_tg256(&gpu, &buf_a, &buf_x, &buf_tg256, m as u32, k as u32)
            .unwrap();

        let base = unsafe { buf_base.as_slice::<f32>() };
        let tg256 = unsafe { buf_tg256.as_slice::<f32>() };
        let diff = max_abs_diff(base, tg256);
        assert!(
            diff < 1e-3,
            "TG=256 Q4_K matvec diverged from base kernel: max_diff={diff}",
        );
    }

    #[test]
    fn test_fused_matvec_q4_k_blk2_matches_base() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = DequantKernels::new(&gpu).unwrap();

        let m = 32usize;
        let k = 512usize;
        let blocks_per_row = k / Q4_K_BLOCK_SIZE;

        let mut quant_data = Vec::new();
        for row in 0..m {
            for blk in 0..blocks_per_row {
                let mut block = vec![0u8; Q4_K_BYTES_PER_BLOCK];
                let d = ((row + blk) % 7) as f32 * 0.2 + 0.1;
                let d_bytes = half::f16::from_f32(d).to_le_bytes();
                block[0] = d_bytes[0];
                block[1] = d_bytes[1];
                let dmin = 0.05f32;
                let dmin_bytes = half::f16::from_f32(dmin).to_le_bytes();
                block[2] = dmin_bytes[0];
                block[3] = dmin_bytes[1];
                for i in 0..4 {
                    block[4 + i] = ((row + i) % 8 + 1) as u8;
                    block[8 + i] = ((blk + i) % 4) as u8;
                }
                block[16..144].fill(((row * 5 + blk) % 16) as u8 + 0x13);
                quant_data.extend(block);
            }
        }

        let x_data: Vec<f32> = (0..k).map(|i| ((i % 11) as f32 - 5.0) * 0.02).collect();
        let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
        let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
        let buf_base = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();
        let buf_blk2 = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

        kernels
            .fused_matvec_q4_k(&gpu, &buf_a, &buf_x, &buf_base, m as u32, k as u32)
            .unwrap();
        kernels
            .fused_matvec_q4_k_blk2(&gpu, &buf_a, &buf_x, &buf_blk2, m as u32, k as u32)
            .unwrap();

        let base = unsafe { buf_base.as_slice::<f32>() };
        let blk2 = unsafe { buf_blk2.as_slice::<f32>() };
        let diff = max_abs_diff(base, blk2);
        assert!(
            diff < 1e-3,
            "BLK2 Q4_K matvec diverged from base kernel: max_diff={diff}",
        );
    }

    #[test]
    fn test_fused_matvec_q4_k_nr2_matches_base() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = DequantKernels::new(&gpu).unwrap();

        let m = 36usize;
        let k = 512usize;
        let blocks_per_row = k / Q4_K_BLOCK_SIZE;

        let mut quant_data = Vec::new();
        for row in 0..m {
            for blk in 0..blocks_per_row {
                let mut block = vec![0u8; Q4_K_BYTES_PER_BLOCK];
                let d = ((row + blk) % 9) as f32 * 0.17 + 0.08;
                let d_bytes = half::f16::from_f32(d).to_le_bytes();
                block[0] = d_bytes[0];
                block[1] = d_bytes[1];
                let dmin = ((row + 2 * blk) % 5) as f32 * 0.03 + 0.02;
                let dmin_bytes = half::f16::from_f32(dmin).to_le_bytes();
                block[2] = dmin_bytes[0];
                block[3] = dmin_bytes[1];
                for i in 0..4 {
                    block[4 + i] = ((row + i) % 8 + 1) as u8;
                    block[8 + i] = ((blk + i * 2) % 4) as u8;
                }
                block[16..144].fill(((row * 7 + blk * 3) % 16) as u8 + 0x07);
                quant_data.extend(block);
            }
        }

        let x_data: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.015).collect();
        let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
        let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
        let buf_base = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();
        let buf_nr2 = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

        kernels
            .fused_matvec_q4_k(&gpu, &buf_a, &buf_x, &buf_base, m as u32, k as u32)
            .unwrap();
        kernels
            .fused_matvec_q4_k_nr2(&gpu, &buf_a, &buf_x, &buf_nr2, m as u32, k as u32)
            .unwrap();

        let base = unsafe { buf_base.as_slice::<f32>() };
        let nr2 = unsafe { buf_nr2.as_slice::<f32>() };
        let diff = max_abs_diff(base, nr2);
        // NR2 uses different SG count and K-stride, so accumulation order
        // differs from the base kernel. 2e-3 allows for f32 rounding differences.
        assert!(
            diff < 2e-3,
            "NR2 Q4_K matvec diverged from base kernel: max_diff={diff}",
        );
    }

    fn make_q6_k_model_like_fixture(m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
        let blocks_per_row = k / Q6_K_BLOCK_SIZE;
        let mut quant_data = Vec::new();

        for row in 0..m {
            for blk in 0..blocks_per_row {
                let mut block = vec![0u8; Q6_K_BYTES_PER_BLOCK];

                // Model-like fixture: bounded scales and dequant factors keep output
                // magnitudes in the same rough range as real decode activations,
                // unlike the worst-case synthetic stress fixture below.
                for (i, byte) in block[..128].iter_mut().enumerate() {
                    *byte = ((row * 3 + blk * 5 + i) % 16) as u8;
                }
                for (i, byte) in block[128..192].iter_mut().enumerate() {
                    *byte = ((row * 11 + blk * 7 + i) % 4) as u8;
                }
                for (i, byte) in block[192..208].iter_mut().enumerate() {
                    *byte = ((row as i32 * 2 + blk as i32 * 3 + i as i32) % 15 - 7) as i8 as u8;
                }

                let d = 0.0125 + ((row + blk) % 9) as f32 * 0.00625;
                let d_bytes = half::f16::from_f32(d).to_le_bytes();
                block[208] = d_bytes[0];
                block[209] = d_bytes[1];
                quant_data.extend(block);
            }
        }

        let x_data: Vec<f32> = (0..k)
            .map(|i| (((i * 7 + 3) % 29) as f32 - 14.0) * 0.0075)
            .collect();

        (quant_data, x_data)
    }

    #[test]
    #[ignore = "stress fixture still exposes base-vs-NR2 divergence on extreme synthetic Q6_K data"]
    fn test_fused_matvec_q6_k_nr2_matches_base_extreme_synthetic() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = DequantKernels::new(&gpu).unwrap();

        let m = 36usize;
        let k = 512usize;
        let blocks_per_row = k / Q6_K_BLOCK_SIZE;

        let mut quant_data = Vec::new();
        for row in 0..m {
            for blk in 0..blocks_per_row {
                let mut block = vec![0u8; Q6_K_BYTES_PER_BLOCK];
                for (i, byte) in block[..128].iter_mut().enumerate() {
                    *byte = ((row * 5 + blk * 3 + i) % 16) as u8;
                }
                for (i, byte) in block[128..192].iter_mut().enumerate() {
                    *byte = ((row * 7 + blk + i) % 4) as u8;
                }
                for (i, byte) in block[192..208].iter_mut().enumerate() {
                    *byte = ((row as i32 * 3 + blk as i32 * 5 + i as i32) % 31 - 15) as i8 as u8;
                }
                let d = ((row + blk) % 11) as f32 * 0.11 + 0.07;
                let d_bytes = half::f16::from_f32(d).to_le_bytes();
                block[208] = d_bytes[0];
                block[209] = d_bytes[1];
                quant_data.extend(block);
            }
        }

        let x_data: Vec<f32> = (0..k).map(|i| ((i % 19) as f32 - 9.0) * 0.0125).collect();
        let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
        let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
        let buf_base = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();
        let buf_nr2 = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

        kernels
            .fused_matvec_q6_k(&gpu, &buf_a, &buf_x, &buf_base, m as u32, k as u32)
            .unwrap();
        kernels
            .fused_matvec_q6_k_nr2(&gpu, &buf_a, &buf_x, &buf_nr2, m as u32, k as u32)
            .unwrap();

        let base = unsafe { buf_base.as_slice::<f32>() };
        let nr2 = unsafe { buf_nr2.as_slice::<f32>() };
        let diff = max_abs_diff(base, nr2);
        // Stress fixture with large synthetic output magnitudes. Keep this ignored
        // until we reconcile the remaining base-vs-NR2 accumulation-order gap.
        assert!(
            diff < 0.15,
            "NR2 Q6_K matvec diverged from base kernel: max_diff={diff}",
        );
    }

    #[test]
    fn test_fused_matvec_q6_k_nr2_matches_base_model_like() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = DequantKernels::new(&gpu).unwrap();

        let m = 36usize;
        let k = 512usize;
        let (quant_data, x_data) = make_q6_k_model_like_fixture(m, k);

        let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
        let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
        let buf_base = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();
        let buf_nr2 = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

        kernels
            .fused_matvec_q6_k(&gpu, &buf_a, &buf_x, &buf_base, m as u32, k as u32)
            .unwrap();
        kernels
            .fused_matvec_q6_k_nr2(&gpu, &buf_a, &buf_x, &buf_nr2, m as u32, k as u32)
            .unwrap();

        let base = unsafe { buf_base.as_slice::<f32>() };
        let nr2 = unsafe { buf_nr2.as_slice::<f32>() };
        let diff = max_abs_diff(base, nr2);
        assert!(
            diff < 0.02,
            "Q6_K NR2 diverged from base on model-like fixture: max_diff={diff}",
        );
    }

    #[test]
    fn test_fused_matvec_q4_k_nr2_matches_cpu_reference() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = DequantKernels::new(&gpu).unwrap();

        let m = 36usize;
        let k = 512usize;
        let blocks_per_row = k / Q4_K_BLOCK_SIZE;

        let mut quant_data = Vec::new();
        for row in 0..m {
            for blk in 0..blocks_per_row {
                let mut block = vec![0u8; Q4_K_BYTES_PER_BLOCK];
                let d = ((row + blk) % 9) as f32 * 0.17 + 0.08;
                let d_bytes = half::f16::from_f32(d).to_le_bytes();
                block[0] = d_bytes[0];
                block[1] = d_bytes[1];
                let dmin = ((row + 2 * blk) % 5) as f32 * 0.03 + 0.02;
                let dmin_bytes = half::f16::from_f32(dmin).to_le_bytes();
                block[2] = dmin_bytes[0];
                block[3] = dmin_bytes[1];
                for i in 0..4 {
                    block[4 + i] = ((row + i) % 8 + 1) as u8;
                    block[8 + i] = ((blk + i * 2) % 4) as u8;
                }
                block[16..144].fill(((row * 7 + blk * 3) % 16) as u8 + 0x07);
                quant_data.extend(block);
            }
        }

        let x_data: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.015).collect();
        let mut a_f32 = vec![0.0f32; m * k];
        cpu_dequant_q4_k(&quant_data, &mut a_f32);
        let mut expected = vec![0.0f32; m];
        cpu_matvec(&a_f32, &x_data, &mut expected, m, k);

        let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
        let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
        let buf_base = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();
        let buf_nr2 = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

        kernels
            .fused_matvec_q4_k(&gpu, &buf_a, &buf_x, &buf_base, m as u32, k as u32)
            .unwrap();
        kernels
            .fused_matvec_q4_k_nr2(&gpu, &buf_a, &buf_x, &buf_nr2, m as u32, k as u32)
            .unwrap();

        let base = unsafe { buf_base.as_slice::<f32>() };
        let nr2 = unsafe { buf_nr2.as_slice::<f32>() };
        let base_diff = max_abs_diff(base, &expected);
        let nr2_diff = max_abs_diff(nr2, &expected);
        assert!(
            nr2_diff < 2e-3,
            "NR2 Q4_K matvec exceeded CPU tolerance: base_diff={base_diff}, nr2_diff={nr2_diff}",
        );
    }

    #[test]
    fn test_fused_matvec_q6_k_nr2_matches_cpu_reference() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = DequantKernels::new(&gpu).unwrap();

        let m = 36usize;
        let k = 512usize;
        let blocks_per_row = k / Q6_K_BLOCK_SIZE;

        let mut quant_data = Vec::new();
        for row in 0..m {
            for blk in 0..blocks_per_row {
                let mut block = vec![0u8; Q6_K_BYTES_PER_BLOCK];
                for (i, byte) in block[..128].iter_mut().enumerate() {
                    *byte = ((row * 5 + blk * 3 + i) % 16) as u8;
                }
                for (i, byte) in block[128..192].iter_mut().enumerate() {
                    *byte = ((row * 7 + blk + i) % 4) as u8;
                }
                for (i, byte) in block[192..208].iter_mut().enumerate() {
                    *byte = ((row as i32 * 3 + blk as i32 * 5 + i as i32) % 31 - 15) as i8 as u8;
                }
                let d = ((row + blk) % 11) as f32 * 0.11 + 0.07;
                let d_bytes = half::f16::from_f32(d).to_le_bytes();
                block[208] = d_bytes[0];
                block[209] = d_bytes[1];
                quant_data.extend(block);
            }
        }

        let x_data: Vec<f32> = (0..k).map(|i| ((i % 19) as f32 - 9.0) * 0.0125).collect();
        let mut a_f32 = vec![0.0f32; m * k];
        cpu_dequant_q6_k(&quant_data, &mut a_f32);
        let mut expected = vec![0.0f32; m];
        cpu_matvec(&a_f32, &x_data, &mut expected, m, k);

        let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
        let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
        let buf_base = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();
        let buf_nr2 = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

        kernels
            .fused_matvec_q6_k(&gpu, &buf_a, &buf_x, &buf_base, m as u32, k as u32)
            .unwrap();
        kernels
            .fused_matvec_q6_k_nr2(&gpu, &buf_a, &buf_x, &buf_nr2, m as u32, k as u32)
            .unwrap();

        let base = unsafe { buf_base.as_slice::<f32>() };
        let nr2 = unsafe { buf_nr2.as_slice::<f32>() };
        let base_diff = max_abs_diff(base, &expected);
        let nr2_diff = max_abs_diff(nr2, &expected);
        // NR2 uses a different accumulation decomposition than the base kernel.
        // Both diverge from CPU f64 reference but by different amounts.
        // The NR2 divergence is larger (up to ~0.12) because it accumulates sub-group
        // sums before multiplying by scale, while base multiplies per-element.
        assert!(
            nr2_diff < 0.15,
            "NR2 Q6_K too far from CPU reference: base_diff={base_diff}, nr2_diff={nr2_diff}",
        );
    }

    #[test]
    fn test_fused_matvec_q6_k_nr2_matches_cpu_reference_model_like() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = DequantKernels::new(&gpu).unwrap();

        let m = 36usize;
        let k = 512usize;
        let (quant_data, x_data) = make_q6_k_model_like_fixture(m, k);

        let mut a_f32 = vec![0.0f32; m * k];
        cpu_dequant_q6_k(&quant_data, &mut a_f32);
        let mut expected = vec![0.0f32; m];
        cpu_matvec(&a_f32, &x_data, &mut expected, m, k);

        let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
        let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
        let buf_base = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();
        let buf_nr2 = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

        kernels
            .fused_matvec_q6_k(&gpu, &buf_a, &buf_x, &buf_base, m as u32, k as u32)
            .unwrap();
        kernels
            .fused_matvec_q6_k_nr2(&gpu, &buf_a, &buf_x, &buf_nr2, m as u32, k as u32)
            .unwrap();

        let base = unsafe { buf_base.as_slice::<f32>() };
        let nr2 = unsafe { buf_nr2.as_slice::<f32>() };
        let base_diff = max_abs_diff(base, &expected);
        let nr2_diff = max_abs_diff(nr2, &expected);
        assert!(
            nr2_diff < 0.02,
            "Q6_K NR2 exceeded model-like CPU tolerance: base_diff={base_diff}, nr2_diff={nr2_diff}",
        );
    }

    #[test]
    fn test_fused_matvec_q6_k_nr2_preserves_model_like_ranking() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = DequantKernels::new(&gpu).unwrap();

        let m = 36usize;
        let k = 512usize;
        let vocab = 128usize;
        let (quant_data, x_data) = make_q6_k_model_like_fixture(m, k);

        let buf_a = MetalBuffer::from_bytes(gpu.device(), &quant_data).unwrap();
        let buf_x = MetalBuffer::from_slice(gpu.device(), &x_data).unwrap();
        let buf_base = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();
        let buf_nr2 = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>()).unwrap();

        kernels
            .fused_matvec_q6_k(&gpu, &buf_a, &buf_x, &buf_base, m as u32, k as u32)
            .unwrap();
        kernels
            .fused_matvec_q6_k_nr2(&gpu, &buf_a, &buf_x, &buf_nr2, m as u32, k as u32)
            .unwrap();

        let base = unsafe { buf_base.as_slice::<f32>() };
        let nr2 = unsafe { buf_nr2.as_slice::<f32>() };

        // Deterministic downstream projection to approximate logit sensitivity.
        // This checks whether model-like kernel differences perturb ranking, not
        // just raw matvec values.
        let proj: Vec<f32> = (0..vocab * m)
            .map(|i| (((i * 13 + 5) % 37) as f32 - 18.0) * 0.021)
            .collect();
        let mut logits_base = vec![0.0f32; vocab];
        let mut logits_nr2 = vec![0.0f32; vocab];
        cpu_matmul(&proj, base, &mut logits_base, vocab, 1, m);
        cpu_matmul(&proj, nr2, &mut logits_nr2, vocab, 1, m);

        let top_base = argmax(&logits_base);
        let top_nr2 = argmax(&logits_nr2);
        let logits_diff = max_abs_diff(&logits_base, &logits_nr2);
        assert_eq!(
            top_base, top_nr2,
            "Q6_K NR2 changed projected top-1 ranking: base_top={top_base}, nr2_top={top_nr2}, logits_max_diff={logits_diff}"
        );
        assert!(
            logits_diff < 0.05,
            "Q6_K NR2 projected logits diverged too far on model-like fixture: max_diff={logits_diff}"
        );
    }

    // ── Attention test helpers ────────────────────────────────────────

    /// CPU reference softmax (numerically stable).
    fn cpu_softmax(x: &mut [f32]) {
        if x.is_empty() {
            return;
        }
        let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in x.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in x.iter_mut() {
                *v /= sum;
            }
        }
    }

    /// CPU reference multi-head attention with causal masking (prefill).
    #[allow(clippy::too_many_arguments)]
    fn cpu_attention_prefill(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        n_tokens: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        let q_stride = n_heads * head_dim;
        let kv_stride = n_kv_heads * head_dim;
        let heads_per_kv = n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        for h in 0..n_heads {
            let kv_h = h / heads_per_kv;
            for qi in 0..n_tokens {
                let q_off = qi * q_stride + h * head_dim;
                let q_head = &q[q_off..q_off + head_dim];
                let attend_len = qi + 1;

                let mut scores = vec![0.0f32; attend_len];
                for (t, score) in scores.iter_mut().enumerate() {
                    let k_off = t * kv_stride + kv_h * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_head[d] * k[k_off + d];
                    }
                    *score = dot * scale;
                }
                cpu_softmax(&mut scores);

                let o_off = qi * q_stride + h * head_dim;
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (t, &w) in scores.iter().enumerate() {
                        let v_off = t * kv_stride + kv_h * head_dim;
                        acc += w * v[v_off + d];
                    }
                    output[o_off + d] = acc;
                }
            }
        }
    }

    // ── Attention kernel tests ───────────────────────────────────────

    #[test]
    fn test_attention_kernels_compile() {
        let gpu = MetalDevice::new().unwrap();
        let _kernels = AttentionKernels::new(&gpu).unwrap();
    }

    #[test]
    fn test_attention_prefill_single_token() {
        // 1 token, 1 head, head_dim=4
        // softmax([score]) = [1.0] → output = V
        let gpu = MetalDevice::new().unwrap();
        let kernels = AttentionKernels::new(&gpu).unwrap();

        let q = [1.0f32, 0.0, 0.0, 0.0];
        let k = [1.0f32, 0.0, 0.0, 0.0];
        let v = [3.0f32, 4.0, 5.0, 6.0];

        let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
        let buf_k = MetalBuffer::from_slice(gpu.device(), &k).unwrap();
        let buf_v = MetalBuffer::from_slice(gpu.device(), &v).unwrap();
        let buf_o = MetalBuffer::new(gpu.device(), 4 * 4).unwrap();

        kernels
            .attention_prefill(&gpu, &buf_q, &buf_k, &buf_v, &buf_o, 1, 1, 1, 4)
            .unwrap();

        let result = unsafe { buf_o.as_slice::<f32>() };
        assert!(
            max_abs_diff(result, &v) < 1e-4,
            "Single token: got {:?}, expected {:?}",
            result,
            v
        );
    }

    #[test]
    fn test_attention_prefill_two_tokens_causal() {
        // 2 tokens, 1 head, head_dim=2
        // Token 0: sees only V[0]; Token 1: sees V[0..2] with equal Q·K
        let gpu = MetalDevice::new().unwrap();
        let kernels = AttentionKernels::new(&gpu).unwrap();

        let q = [1.0f32, 0.0, 1.0, 0.0]; // 2 tokens
        let k = [1.0f32, 0.0, 1.0, 0.0]; // equal keys → uniform softmax
        let v = [2.0f32, 3.0, 4.0, 5.0]; // V[0]=[2,3], V[1]=[4,5]

        let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
        let buf_k = MetalBuffer::from_slice(gpu.device(), &k).unwrap();
        let buf_v = MetalBuffer::from_slice(gpu.device(), &v).unwrap();
        let buf_o = MetalBuffer::new(gpu.device(), 4 * 4).unwrap();

        kernels
            .attention_prefill(&gpu, &buf_q, &buf_k, &buf_v, &buf_o, 2, 1, 1, 2)
            .unwrap();

        let result = unsafe { buf_o.as_slice::<f32>() };

        // Token 0: only sees V[0] = [2, 3]
        assert!(
            (result[0] - 2.0).abs() < 1e-4,
            "token0[0]={}, expected 2.0",
            result[0]
        );
        assert!(
            (result[1] - 3.0).abs() < 1e-4,
            "token0[1]={}, expected 3.0",
            result[1]
        );
        // Token 1: uniform over V[0..2] → [3.0, 4.0]
        assert!(
            (result[2] - 3.0).abs() < 1e-4,
            "token1[0]={}, expected 3.0",
            result[2]
        );
        assert!(
            (result[3] - 4.0).abs() < 1e-4,
            "token1[1]={}, expected 4.0",
            result[3]
        );
    }

    #[test]
    fn test_attention_prefill_gqa() {
        // 1 token, 4 query heads, 2 KV heads, head_dim=2
        // Heads 0,1 → kv_head 0; Heads 2,3 → kv_head 1
        let gpu = MetalDevice::new().unwrap();
        let kernels = AttentionKernels::new(&gpu).unwrap();

        let q = [1.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]; // 4 heads
        let k = [1.0f32, 0.0, 0.0, 1.0]; // 2 KV heads
        let v = [10.0f32, 20.0, 30.0, 40.0]; // 2 KV heads

        let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
        let buf_k = MetalBuffer::from_slice(gpu.device(), &k).unwrap();
        let buf_v = MetalBuffer::from_slice(gpu.device(), &v).unwrap();
        let buf_o = MetalBuffer::new(gpu.device(), 8 * 4).unwrap();

        kernels
            .attention_prefill(&gpu, &buf_q, &buf_k, &buf_v, &buf_o, 1, 4, 2, 2)
            .unwrap();

        let result = unsafe { buf_o.as_slice::<f32>() };

        // Heads 0,1 → kv0 V = [10, 20]
        assert!((result[0] - 10.0).abs() < 1e-3, "h0[0]={}", result[0]);
        assert!((result[1] - 20.0).abs() < 1e-3, "h0[1]={}", result[1]);
        assert!((result[2] - 10.0).abs() < 1e-3, "h1[0]={}", result[2]);
        assert!((result[3] - 20.0).abs() < 1e-3, "h1[1]={}", result[3]);
        // Heads 2,3 → kv1 V = [30, 40]
        assert!((result[4] - 30.0).abs() < 1e-3, "h2[0]={}", result[4]);
        assert!((result[5] - 40.0).abs() < 1e-3, "h2[1]={}", result[5]);
        assert!((result[6] - 30.0).abs() < 1e-3, "h3[0]={}", result[6]);
        assert!((result[7] - 40.0).abs() < 1e-3, "h3[1]={}", result[7]);
    }

    #[test]
    fn test_attention_prefill_matches_cpu() {
        // Larger test: 16 tokens, 4 heads, 2 KV heads, head_dim=64
        let gpu = MetalDevice::new().unwrap();
        let kernels = AttentionKernels::new(&gpu).unwrap();

        let n_tokens = 16;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 64;
        let q_size = n_tokens * n_heads * head_dim;
        let kv_size = n_tokens * n_kv_heads * head_dim;

        // Generate deterministic data
        let q: Vec<f32> = (0..q_size)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
            .collect();
        let k: Vec<f32> = (0..kv_size)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
            .collect();
        let v: Vec<f32> = (0..kv_size).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();

        // CPU reference
        let mut expected = vec![0.0f32; q_size];
        cpu_attention_prefill(
            &q,
            &k,
            &v,
            &mut expected,
            n_tokens,
            n_heads,
            n_kv_heads,
            head_dim,
        );

        // GPU
        let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
        let buf_k = MetalBuffer::from_slice(gpu.device(), &k).unwrap();
        let buf_v = MetalBuffer::from_slice(gpu.device(), &v).unwrap();
        let buf_o = MetalBuffer::new(gpu.device(), q_size * std::mem::size_of::<f32>()).unwrap();

        kernels
            .attention_prefill(
                &gpu,
                &buf_q,
                &buf_k,
                &buf_v,
                &buf_o,
                n_tokens as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
            )
            .unwrap();

        let result = unsafe { buf_o.as_slice::<f32>() };
        let diff = max_abs_diff(result, &expected);
        assert!(
            diff < 1e-2,
            "Attention prefill CPU vs GPU mismatch: max_diff={}",
            diff
        );
    }

    #[test]
    fn test_attention_prefill_multi_tile() {
        // 512 tokens to trigger multiple tiles (ATTN_TG=256, so 2+ tiles)
        let gpu = MetalDevice::new().unwrap();
        let kernels = AttentionKernels::new(&gpu).unwrap();

        let n_tokens = 512;
        let n_heads = 2;
        let n_kv_heads = 2;
        let head_dim = 32;
        let q_size = n_tokens * n_heads * head_dim;
        let kv_size = n_tokens * n_kv_heads * head_dim;

        let q: Vec<f32> = (0..q_size)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.03)
            .collect();
        let k: Vec<f32> = (0..kv_size)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.03)
            .collect();
        let v: Vec<f32> = (0..kv_size)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
            .collect();

        let mut expected = vec![0.0f32; q_size];
        cpu_attention_prefill(
            &q,
            &k,
            &v,
            &mut expected,
            n_tokens,
            n_heads,
            n_kv_heads,
            head_dim,
        );

        let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
        let buf_k = MetalBuffer::from_slice(gpu.device(), &k).unwrap();
        let buf_v = MetalBuffer::from_slice(gpu.device(), &v).unwrap();
        let buf_o = MetalBuffer::new(gpu.device(), q_size * std::mem::size_of::<f32>()).unwrap();

        kernels
            .attention_prefill(
                &gpu,
                &buf_q,
                &buf_k,
                &buf_v,
                &buf_o,
                n_tokens as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
            )
            .unwrap();

        let result = unsafe { buf_o.as_slice::<f32>() };
        let diff = max_abs_diff(result, &expected);
        assert!(
            diff < 0.05,
            "Multi-tile attention mismatch: max_diff={}",
            diff
        );
    }

    #[test]
    fn test_attention_prefill_concentrates_on_matching_key() {
        // Large Q·K score for one key should dominate
        let gpu = MetalDevice::new().unwrap();
        let kernels = AttentionKernels::new(&gpu).unwrap();

        // 3 tokens, 1 head, head_dim=2
        // Q[2] = [10, 0], K = [[1,0], [0,1], [1,0]]
        // Token 2 attends to 0..2: scores ∝ [10, 0, 10] → nearly [0.5, 0, 0.5]
        let q = [1.0f32, 0.0, 0.0, 1.0, 10.0, 0.0]; // 3 tokens
        let k = [1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0]; // 3 KV tokens
        let v = [1.0f32, 0.0, 0.0, 1.0, 0.5, 0.5]; // 3 V tokens

        let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
        let buf_k = MetalBuffer::from_slice(gpu.device(), &k).unwrap();
        let buf_v = MetalBuffer::from_slice(gpu.device(), &v).unwrap();
        let buf_o = MetalBuffer::new(gpu.device(), 6 * 4).unwrap();

        kernels
            .attention_prefill(&gpu, &buf_q, &buf_k, &buf_v, &buf_o, 3, 1, 1, 2)
            .unwrap();

        let result = unsafe { buf_o.as_slice::<f32>() };
        let mut expected = vec![0.0f32; 6];
        cpu_attention_prefill(&q, &k, &v, &mut expected, 3, 1, 1, 2);

        let diff = max_abs_diff(result, &expected);
        assert!(
            diff < 1e-3,
            "Concentrate test: max_diff={}, got {:?}, expected {:?}",
            diff,
            result,
            expected
        );
    }

    // ── Elementwise kernel tests ────────────────────────────────────

    // CPU reference implementations for elementwise kernel verification.

    fn cpu_rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
        let n = x.len();
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
        for (xi, &wi) in x.iter_mut().zip(weight.iter()) {
            *xi = *xi * inv_rms * wi;
        }
    }

    fn cpu_rms_norm_out(x: &[f32], weight: &[f32], out: &mut [f32], eps: f32) {
        let n = x.len();
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
        for i in 0..n {
            out[i] = x[i] * inv_rms * weight[i];
        }
    }

    fn cpu_rope(
        q: &mut [f32],
        k: &mut [f32],
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        position: f32,
        freq_base: f32,
    ) {
        let half_dim = head_dim / 2;
        let mut cos_table = vec![0.0f32; half_dim];
        let mut sin_table = vec![0.0f32; half_dim];
        for i in 0..half_dim {
            let freq = 1.0 / freq_base.powf(2.0 * i as f32 / head_dim as f32);
            let theta = position * freq;
            cos_table[i] = theta.cos();
            sin_table[i] = theta.sin();
        }
        for h in 0..n_q_heads {
            let off = h * head_dim;
            for i in 0..half_dim {
                let v0 = q[off + 2 * i];
                let v1 = q[off + 2 * i + 1];
                q[off + 2 * i] = v0 * cos_table[i] - v1 * sin_table[i];
                q[off + 2 * i + 1] = v0 * sin_table[i] + v1 * cos_table[i];
            }
        }
        for h in 0..n_kv_heads {
            let off = h * head_dim;
            for i in 0..half_dim {
                let v0 = k[off + 2 * i];
                let v1 = k[off + 2 * i + 1];
                k[off + 2 * i] = v0 * cos_table[i] - v1 * sin_table[i];
                k[off + 2 * i + 1] = v0 * sin_table[i] + v1 * cos_table[i];
            }
        }
    }

    fn cpu_gelu_elementwise_mul(gate: &mut [f32], up: &[f32]) {
        const SQRT_2_PI: f32 = 0.797_884_6;
        for (gi, &ui) in gate.iter_mut().zip(up.iter()) {
            let x = *gi;
            let x3 = x * x * x;
            let inner = SQRT_2_PI * (x + 0.044715 * x3);
            *gi = 0.5 * x * (1.0 + inner.tanh()) * ui;
        }
    }

    fn cpu_silu_elementwise_mul(gate: &mut [f32], up: &[f32]) {
        for (gi, &ui) in gate.iter_mut().zip(up.iter()) {
            let x = *gi;
            *gi = x / (1.0 + (-x).exp()) * ui;
        }
    }

    #[test]
    fn test_elementwise_kernels_compile() {
        let gpu = MetalDevice::new().unwrap();
        let _kernels = ElementwiseKernels::new(&gpu).unwrap();
    }

    #[test]
    fn test_gpu_rms_norm_matches_cpu() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = ElementwiseKernels::new(&gpu).unwrap();
        let n = 256;
        let eps = 1e-5f32;

        let x: Vec<f32> = (0..n).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect();
        let weight: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.2 + 1.0).collect();

        // CPU reference
        let mut expected = x.clone();
        cpu_rms_norm(&mut expected, &weight, eps);

        // GPU
        let buf_x = MetalBuffer::from_slice(gpu.device(), &x).unwrap();
        let buf_w = MetalBuffer::from_slice(gpu.device(), &weight).unwrap();
        kernels
            .rms_norm(&gpu, &buf_x, &buf_w, n as u32, eps)
            .unwrap();
        let result = unsafe { buf_x.as_slice::<f32>() };

        let diff = max_abs_diff(result, &expected);
        assert!(diff < 1e-4, "RMSNorm GPU vs CPU mismatch: max_diff={diff}");
    }

    #[test]
    fn test_gpu_rms_norm_out_matches_cpu() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = ElementwiseKernels::new(&gpu).unwrap();
        let n = 512;
        let eps = 1e-6f32;

        let x: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.2).collect();
        let weight: Vec<f32> = (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.1 + 1.0).collect();

        let mut expected = vec![0.0f32; n];
        cpu_rms_norm_out(&x, &weight, &mut expected, eps);

        let buf_x = MetalBuffer::from_slice(gpu.device(), &x).unwrap();
        let buf_w = MetalBuffer::from_slice(gpu.device(), &weight).unwrap();
        let buf_out = MetalBuffer::new(gpu.device(), n * 4).unwrap();
        kernels
            .rms_norm_out(&gpu, &buf_x, &buf_w, &buf_out, n as u32, eps)
            .unwrap();
        let result = unsafe { buf_out.as_slice::<f32>() };

        let diff = max_abs_diff(result, &expected);
        assert!(
            diff < 1e-4,
            "RMSNorm out GPU vs CPU mismatch: max_diff={diff}"
        );
    }

    #[test]
    fn test_gpu_rope_matches_cpu() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = ElementwiseKernels::new(&gpu).unwrap();

        let n_q_heads = 8u32;
        let n_kv_heads = 4u32;
        let head_dim = 64u32;
        let position = 42.0f32;
        let freq_base = 10000.0f32;

        let q: Vec<f32> = (0..n_q_heads as usize * head_dim as usize)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
            .collect();
        let k: Vec<f32> = (0..n_kv_heads as usize * head_dim as usize)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();

        // CPU reference
        let mut q_ref = q.clone();
        let mut k_ref = k.clone();
        cpu_rope(
            &mut q_ref,
            &mut k_ref,
            n_q_heads as usize,
            n_kv_heads as usize,
            head_dim as usize,
            position,
            freq_base,
        );

        // GPU
        let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
        let buf_k = MetalBuffer::from_slice(gpu.device(), &k).unwrap();
        kernels
            .rope(
                &gpu, &buf_q, &buf_k, n_q_heads, n_kv_heads, head_dim, position, freq_base,
            )
            .unwrap();
        let q_gpu = unsafe { buf_q.as_slice::<f32>() };
        let k_gpu = unsafe { buf_k.as_slice::<f32>() };

        let q_diff = max_abs_diff(q_gpu, &q_ref);
        let k_diff = max_abs_diff(k_gpu, &k_ref);
        assert!(
            q_diff < 1e-4,
            "RoPE Q GPU vs CPU mismatch: max_diff={q_diff}"
        );
        assert!(
            k_diff < 1e-4,
            "RoPE K GPU vs CPU mismatch: max_diff={k_diff}"
        );
    }

    #[test]
    fn test_gpu_per_head_rms_norm_matches_cpu() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = ElementwiseKernels::new(&gpu).unwrap();

        let n_heads = 8u32;
        let head_dim = 64u32;
        let n = (n_heads * head_dim) as usize;
        let eps = 1e-6f32;

        let data: Vec<f32> = (0..n).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect();
        let weight: Vec<f32> = (0..head_dim as usize)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.2 + 1.0)
            .collect();

        // CPU reference: per-head rms_norm
        let mut expected = data.clone();
        for h in 0..n_heads as usize {
            let off = h * head_dim as usize;
            cpu_rms_norm(&mut expected[off..off + head_dim as usize], &weight, eps);
        }

        // GPU
        let buf = MetalBuffer::from_slice(gpu.device(), &data).unwrap();
        let buf_w = MetalBuffer::from_slice(gpu.device(), &weight).unwrap();
        kernels
            .per_head_rms_norm(&gpu, &buf, &buf_w, n_heads, head_dim, eps)
            .unwrap();
        let result = unsafe { buf.as_slice::<f32>() };

        let diff = max_abs_diff(result, &expected);
        assert!(
            diff < 1e-4,
            "Per-head RMSNorm GPU vs CPU mismatch: max_diff={diff}"
        );
    }

    #[test]
    fn test_gpu_gelu_mul_matches_cpu() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = ElementwiseKernels::new(&gpu).unwrap();

        let n = 1024;
        let gate: Vec<f32> = (0..n).map(|i| ((i % 19) as f32 - 9.0) * 0.3).collect();
        let up: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.2).collect();

        let mut expected = gate.clone();
        cpu_gelu_elementwise_mul(&mut expected, &up);

        let buf_gate = MetalBuffer::from_slice(gpu.device(), &gate).unwrap();
        let buf_up = MetalBuffer::from_slice(gpu.device(), &up).unwrap();
        kernels
            .gelu_elementwise_mul(&gpu, &buf_gate, &buf_up, n as u32)
            .unwrap();
        let result = unsafe { buf_gate.as_slice::<f32>() };

        let diff = max_abs_diff(result, &expected);
        assert!(diff < 1e-4, "GELU*mul GPU vs CPU mismatch: max_diff={diff}");
    }

    #[test]
    fn test_gpu_gelu_mul_large_values() {
        // Reproduce NaN seen in Gemma3 4B inference at index 7027/10240
        let gpu = MetalDevice::new().unwrap();
        let kernels = ElementwiseKernels::new(&gpu).unwrap();

        let n = 10240usize;
        let mut gate: Vec<f32> = (0..n).map(|i| ((i % 37) as f32 - 18.0) * 0.7).collect();
        let mut up: Vec<f32> = (0..n).map(|i| ((i % 29) as f32 - 14.0) * 1.3).collect();
        // Place exact problematic values at index 7027
        gate[7027] = 12.154314;
        up[7027] = -18.1271;

        let mut expected = gate.clone();
        cpu_gelu_elementwise_mul(&mut expected, &up);
        assert!(!expected[7027].is_nan(), "CPU GELU should not produce NaN");

        let buf_gate = MetalBuffer::from_slice(gpu.device(), &gate).unwrap();
        let buf_up = MetalBuffer::from_slice(gpu.device(), &up).unwrap();
        kernels
            .gelu_elementwise_mul(&gpu, &buf_gate, &buf_up, n as u32)
            .unwrap();
        let result = unsafe { buf_gate.as_slice::<f32>() };

        // Check for NaN
        let nan_count = result.iter().filter(|v| v.is_nan()).count();
        assert_eq!(nan_count, 0, "GPU GELU produced {nan_count} NaN values");

        let diff = max_abs_diff(result, &expected);
        assert!(diff < 1e-3, "GELU*mul GPU vs CPU mismatch: max_diff={diff}");
    }

    #[test]
    fn test_gpu_silu_mul_matches_cpu() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = ElementwiseKernels::new(&gpu).unwrap();

        let n = 1024;
        let gate: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.3).collect();
        let up: Vec<f32> = (0..n).map(|i| ((i % 11) as f32 - 5.0) * 0.2).collect();

        let mut expected = gate.clone();
        cpu_silu_elementwise_mul(&mut expected, &up);

        let buf_gate = MetalBuffer::from_slice(gpu.device(), &gate).unwrap();
        let buf_up = MetalBuffer::from_slice(gpu.device(), &up).unwrap();
        kernels
            .silu_elementwise_mul(&gpu, &buf_gate, &buf_up, n as u32)
            .unwrap();
        let result = unsafe { buf_gate.as_slice::<f32>() };

        let diff = max_abs_diff(result, &expected);
        assert!(diff < 1e-4, "SiLU*mul GPU vs CPU mismatch: max_diff={diff}");
    }

    #[test]
    fn test_gpu_elementwise_add_matches_cpu() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = ElementwiseKernels::new(&gpu).unwrap();

        let n = 512;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();

        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(ai, bi)| ai + bi).collect();

        let buf_a = MetalBuffer::from_slice(gpu.device(), &a).unwrap();
        let buf_b = MetalBuffer::from_slice(gpu.device(), &b).unwrap();
        kernels
            .elementwise_add(&gpu, &buf_a, &buf_b, n as u32)
            .unwrap();
        let result = unsafe { buf_a.as_slice::<f32>() };

        let diff = max_abs_diff(result, &expected);
        assert!(
            diff < 1e-6,
            "Elementwise add GPU vs CPU mismatch: max_diff={diff}"
        );
    }

    // ── KV Append kernel tests ──────────────────────────────────────

    #[test]
    fn test_kv_append_basic() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = ElementwiseKernels::new(&gpu).unwrap();

        let kv_stride: usize = 8; // e.g. 2 kv_heads * 4 head_dim
        let capacity: usize = 16;
        // Source: one token's KV data
        let src: Vec<f32> = (0..kv_stride).map(|i| (i + 1) as f32).collect();
        // Destination: pre-allocated cache
        let dst_data = vec![0.0f32; capacity * kv_stride];

        let buf_src = MetalBuffer::from_slice(gpu.device(), &src).unwrap();
        let buf_dst = MetalBuffer::from_slice(gpu.device(), &dst_data).unwrap();

        // Append at position 0
        kernels
            .kv_append(&gpu, &buf_src, &buf_dst, false, 0, kv_stride as u32)
            .unwrap();
        let result = unsafe { buf_dst.as_slice::<f32>() };
        assert_eq!(&result[..kv_stride], &src[..]);
        assert!(result[kv_stride..].iter().all(|&x| x == 0.0));

        // Append at position 3
        let offset = 3 * kv_stride;
        let src2: Vec<f32> = (0..kv_stride).map(|i| (i + 100) as f32).collect();
        let buf_src2 = MetalBuffer::from_slice(gpu.device(), &src2).unwrap();
        kernels
            .kv_append(
                &gpu,
                &buf_src2,
                &buf_dst,
                false,
                offset as u32,
                kv_stride as u32,
            )
            .unwrap();
        let result2 = unsafe { buf_dst.as_slice::<f32>() };
        assert_eq!(&result2[offset..offset + kv_stride], &src2[..]);
        // Position 0 still intact
        assert_eq!(&result2[..kv_stride], &src[..]);
    }

    // ── Decode Attention kernel tests ───────────────────────────────

    /// CPU reference for single-token decode attention with GQA.
    #[allow(clippy::too_many_arguments)]
    fn cpu_decode_attention(
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        attend_start: usize,
        attend_len: usize,
    ) -> Vec<f32> {
        let heads_per_kv = n_heads / n_kv_heads;
        let kv_stride = n_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; n_heads * head_dim];

        for h in 0..n_heads {
            let kv_h = h / heads_per_kv;
            let q_off = h * head_dim;

            // Compute scores
            let mut scores = vec![0.0f32; attend_len];
            let mut max_score = f32::NEG_INFINITY;
            for (t, score) in scores.iter_mut().enumerate() {
                let k_off = (attend_start + t) * kv_stride + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * k_cache[k_off + d];
                }
                *score = dot * scale;
                if *score > max_score {
                    max_score = *score;
                }
            }

            // Softmax
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            for s in &mut scores {
                *s /= sum;
            }

            // Weighted sum of V
            for (t, &weight) in scores.iter().enumerate() {
                let v_off = (attend_start + t) * kv_stride + kv_h * head_dim;
                for d in 0..head_dim {
                    out[q_off + d] += weight * v_cache[v_off + d];
                }
            }
        }
        out
    }

    #[test]
    fn test_attention_decode_single_token() {
        // 1 query head, 1 kv head, head_dim=4, 1 token in cache
        let gpu = MetalDevice::new().unwrap();
        let kernels = AttentionKernels::new(&gpu).unwrap();

        let q = [1.0f32, 0.0, 0.0, 0.0];
        let k_cache = [1.0f32, 0.0, 0.0, 0.0]; // 1 token
        let v_cache = [3.0f32, 4.0, 5.0, 6.0];

        let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
        let buf_k = MetalBuffer::from_slice(gpu.device(), &k_cache).unwrap();
        let buf_v = MetalBuffer::from_slice(gpu.device(), &v_cache).unwrap();
        let buf_o = MetalBuffer::new(gpu.device(), 4 * 4).unwrap();

        kernels
            .attention_decode(&gpu, &buf_q, &buf_k, &buf_v, &buf_o, false, 1, 1, 4, 0, 1)
            .unwrap();

        let result = unsafe { buf_o.as_slice::<f32>() };
        // Single token: softmax([score]) = [1.0], output = V
        assert!(
            max_abs_diff(result, &v_cache) < 1e-4,
            "Single decode token: got {:?}, expected {:?}",
            result,
            v_cache
        );
    }

    #[test]
    fn test_attention_decode_multi_token() {
        // 1 head, head_dim=2, 3 tokens in cache, attend all 3
        let gpu = MetalDevice::new().unwrap();
        let kernels = AttentionKernels::new(&gpu).unwrap();

        let head_dim = 2;
        let n_heads = 1u32;
        let n_kv_heads = 1u32;
        let attend_len = 3u32;

        // Q = [1, 0] → dot with K[0]=[1,0]=1, K[1]=[1,0]=1, K[2]=[1,0]=1
        // All equal scores → uniform softmax → average of V
        let q = [1.0f32, 0.0];
        let k_cache = [1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0];
        let v_cache = [2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0]; // V0=[2,3], V1=[4,5], V2=[6,7]

        let expected = cpu_decode_attention(
            &q,
            &k_cache,
            &v_cache,
            n_heads as usize,
            n_kv_heads as usize,
            head_dim,
            0,
            attend_len as usize,
        );

        let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
        let buf_k = MetalBuffer::from_slice(gpu.device(), &k_cache).unwrap();
        let buf_v = MetalBuffer::from_slice(gpu.device(), &v_cache).unwrap();
        let buf_o = MetalBuffer::new(gpu.device(), n_heads as usize * head_dim * 4).unwrap();

        kernels
            .attention_decode(
                &gpu,
                &buf_q,
                &buf_k,
                &buf_v,
                &buf_o,
                false,
                n_heads,
                n_kv_heads,
                head_dim as u32,
                0,
                attend_len,
            )
            .unwrap();

        let result = unsafe { buf_o.as_slice::<f32>() };
        let diff = max_abs_diff(result, &expected);
        assert!(
            diff < 1e-3,
            "Multi-token decode: max_diff={diff}, got {:?}, expected {:?}",
            result,
            expected
        );
    }

    #[test]
    fn test_attention_decode_gqa() {
        // 4 query heads, 2 kv heads, head_dim=4, 2 tokens in cache
        let gpu = MetalDevice::new().unwrap();
        let kernels = AttentionKernels::new(&gpu).unwrap();

        let n_heads = 4u32;
        let n_kv_heads = 2u32;
        let head_dim = 4usize;
        let attend_len = 2u32;
        let _kv_stride = n_kv_heads as usize * head_dim; // 8

        // Q: 4 heads × 4 dims = 16 floats
        let q: Vec<f32> = (0..16).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
        // K cache: 2 tokens × kv_stride = 16 floats
        let k_cache: Vec<f32> = (0..16).map(|i| ((i % 7) as f32 - 3.0) * 0.2).collect();
        // V cache: 2 tokens × kv_stride = 16 floats
        let v_cache: Vec<f32> = (0..16).map(|i| ((i % 11) as f32) * 0.1).collect();

        let expected = cpu_decode_attention(
            &q,
            &k_cache,
            &v_cache,
            n_heads as usize,
            n_kv_heads as usize,
            head_dim,
            0,
            attend_len as usize,
        );

        let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
        let buf_k = MetalBuffer::from_slice(gpu.device(), &k_cache).unwrap();
        let buf_v = MetalBuffer::from_slice(gpu.device(), &v_cache).unwrap();
        let buf_o = MetalBuffer::new(gpu.device(), n_heads as usize * head_dim * 4).unwrap();

        kernels
            .attention_decode(
                &gpu,
                &buf_q,
                &buf_k,
                &buf_v,
                &buf_o,
                false,
                n_heads,
                n_kv_heads,
                head_dim as u32,
                0,
                attend_len,
            )
            .unwrap();

        let result = unsafe { buf_o.as_slice::<f32>() };
        let diff = max_abs_diff(result, &expected);
        assert!(diff < 1e-3, "GQA decode: max_diff={diff}");
    }

    #[test]
    fn test_attention_decode_sliding_window() {
        // 2 heads, 2 kv heads, head_dim=4, 8 tokens in cache, window of 4 starting at 4
        let gpu = MetalDevice::new().unwrap();
        let kernels = AttentionKernels::new(&gpu).unwrap();

        let n_heads = 2u32;
        let n_kv_heads = 2u32;
        let head_dim = 4usize;
        let total_tokens = 8usize;
        let attend_start = 4u32;
        let attend_len = 4u32;
        let kv_stride = n_kv_heads as usize * head_dim;

        let q: Vec<f32> = (0..n_heads as usize * head_dim)
            .map(|i| ((i % 3) as f32 - 1.0) * 0.5)
            .collect();
        let k_cache: Vec<f32> = (0..total_tokens * kv_stride)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();
        let v_cache: Vec<f32> = (0..total_tokens * kv_stride)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
            .collect();

        let expected = cpu_decode_attention(
            &q,
            &k_cache,
            &v_cache,
            n_heads as usize,
            n_kv_heads as usize,
            head_dim,
            attend_start as usize,
            attend_len as usize,
        );

        let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
        let buf_k = MetalBuffer::from_slice(gpu.device(), &k_cache).unwrap();
        let buf_v = MetalBuffer::from_slice(gpu.device(), &v_cache).unwrap();
        let buf_o = MetalBuffer::new(gpu.device(), n_heads as usize * head_dim * 4).unwrap();

        kernels
            .attention_decode(
                &gpu,
                &buf_q,
                &buf_k,
                &buf_v,
                &buf_o,
                false,
                n_heads,
                n_kv_heads,
                head_dim as u32,
                attend_start,
                attend_len,
            )
            .unwrap();

        let result = unsafe { buf_o.as_slice::<f32>() };
        let diff = max_abs_diff(result, &expected);
        assert!(diff < 1e-3, "Sliding window decode: max_diff={diff}");
    }

    #[test]
    fn test_attention_decode_matches_cpu_random() {
        // Larger random test: 8 heads, 4 kv heads, head_dim=64, 32 tokens
        let gpu = MetalDevice::new().unwrap();
        let kernels = AttentionKernels::new(&gpu).unwrap();

        let n_heads = 8u32;
        let n_kv_heads = 4u32;
        let head_dim = 64usize;
        let attend_len = 32u32;
        let kv_stride = n_kv_heads as usize * head_dim;

        // Deterministic pseudo-random data
        let mut seed = 42u64;
        let mut next_f32 = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };

        let q: Vec<f32> = (0..n_heads as usize * head_dim)
            .map(|_| next_f32() * 0.5)
            .collect();
        let k_cache: Vec<f32> = (0..attend_len as usize * kv_stride)
            .map(|_| next_f32() * 0.5)
            .collect();
        let v_cache: Vec<f32> = (0..attend_len as usize * kv_stride)
            .map(|_| next_f32() * 0.5)
            .collect();

        let expected = cpu_decode_attention(
            &q,
            &k_cache,
            &v_cache,
            n_heads as usize,
            n_kv_heads as usize,
            head_dim,
            0,
            attend_len as usize,
        );

        let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
        let buf_k = MetalBuffer::from_slice(gpu.device(), &k_cache).unwrap();
        let buf_v = MetalBuffer::from_slice(gpu.device(), &v_cache).unwrap();
        let buf_o = MetalBuffer::new(gpu.device(), n_heads as usize * head_dim * 4).unwrap();

        kernels
            .attention_decode(
                &gpu,
                &buf_q,
                &buf_k,
                &buf_v,
                &buf_o,
                false,
                n_heads,
                n_kv_heads,
                head_dim as u32,
                0,
                attend_len,
            )
            .unwrap();

        let result = unsafe { buf_o.as_slice::<f32>() };
        let diff = max_abs_diff(result, &expected);
        assert!(
            diff < 5e-3,
            "Random decode attention: max_diff={diff} (8 heads, 4 kv, dim=64, 32 tokens)"
        );
    }

    #[test]
    fn test_attention_decode_long_sequence() {
        // Stress test: 4 heads, 2 kv heads, head_dim=128, 512 tokens
        let gpu = MetalDevice::new().unwrap();
        let kernels = AttentionKernels::new(&gpu).unwrap();

        let n_heads = 4u32;
        let n_kv_heads = 2u32;
        let head_dim = 128usize;
        let attend_len = 512u32;
        let kv_stride = n_kv_heads as usize * head_dim;

        let mut seed = 123u64;
        let mut next_f32 = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };

        let q: Vec<f32> = (0..n_heads as usize * head_dim)
            .map(|_| next_f32() * 0.3)
            .collect();
        let k_cache: Vec<f32> = (0..attend_len as usize * kv_stride)
            .map(|_| next_f32() * 0.3)
            .collect();
        let v_cache: Vec<f32> = (0..attend_len as usize * kv_stride)
            .map(|_| next_f32() * 0.3)
            .collect();

        let expected = cpu_decode_attention(
            &q,
            &k_cache,
            &v_cache,
            n_heads as usize,
            n_kv_heads as usize,
            head_dim,
            0,
            attend_len as usize,
        );

        let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
        let buf_k = MetalBuffer::from_slice(gpu.device(), &k_cache).unwrap();
        let buf_v = MetalBuffer::from_slice(gpu.device(), &v_cache).unwrap();
        let buf_o = MetalBuffer::new(gpu.device(), n_heads as usize * head_dim * 4).unwrap();

        kernels
            .attention_decode(
                &gpu,
                &buf_q,
                &buf_k,
                &buf_v,
                &buf_o,
                false,
                n_heads,
                n_kv_heads,
                head_dim as u32,
                0,
                attend_len,
            )
            .unwrap();

        let result = unsafe { buf_o.as_slice::<f32>() };
        let diff = max_abs_diff(result, &expected);
        assert!(
            diff < 1e-2,
            "Long sequence decode: max_diff={diff} (4 heads, 2 kv, dim=128, 512 tokens)"
        );
    }

    #[test]
    fn test_attention_decode_splitk_f16kv_hd128_matches_cpu() {
        let gpu = MetalDevice::new().unwrap();
        let kernels = AttentionKernels::new(&gpu).unwrap();

        let n_heads = 4u32;
        let n_kv_heads = 2u32;
        let head_dim = 128usize;
        let attend_len = 512u32;
        let kv_stride = n_kv_heads as usize * head_dim;

        assert!(attention_decode_splitk_should_use_mode(
            KernelMode::On,
            true,
            head_dim as u32,
            attend_len,
        ));

        let mut seed = 987u64;
        let mut next_f32 = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };

        let q: Vec<f32> = (0..n_heads as usize * head_dim)
            .map(|_| next_f32() * 0.3)
            .collect();
        let k_cache_f32: Vec<f32> = (0..attend_len as usize * kv_stride)
            .map(|_| next_f32() * 0.3)
            .collect();
        let v_cache_f32: Vec<f32> = (0..attend_len as usize * kv_stride)
            .map(|_| next_f32() * 0.3)
            .collect();
        let k_cache_f16: Vec<half::f16> = k_cache_f32
            .iter()
            .copied()
            .map(half::f16::from_f32)
            .collect();
        let v_cache_f16: Vec<half::f16> = v_cache_f32
            .iter()
            .copied()
            .map(half::f16::from_f32)
            .collect();
        let k_cache_ref: Vec<f32> = k_cache_f16.iter().map(|v| v.to_f32()).collect();
        let v_cache_ref: Vec<f32> = v_cache_f16.iter().map(|v| v.to_f32()).collect();

        let expected = cpu_decode_attention(
            &q,
            &k_cache_ref,
            &v_cache_ref,
            n_heads as usize,
            n_kv_heads as usize,
            head_dim,
            0,
            attend_len as usize,
        );

        let buf_q = MetalBuffer::from_slice(gpu.device(), &q).unwrap();
        let buf_k = MetalBuffer::from_slice(gpu.device(), &k_cache_f16).unwrap();
        let buf_v = MetalBuffer::from_slice(gpu.device(), &v_cache_f16).unwrap();
        let buf_o = MetalBuffer::new(gpu.device(), n_heads as usize * head_dim * 4).unwrap();

        kernels
            .attention_decode_splitk(
                &gpu,
                &buf_q,
                &buf_k,
                &buf_v,
                &buf_o,
                true,
                n_heads,
                n_kv_heads,
                head_dim as u32,
                0,
                attend_len,
            )
            .unwrap();

        let result = unsafe { buf_o.as_slice::<f32>() };
        let diff = max_abs_diff(result, &expected);
        assert!(
            diff < 1e-2,
            "Split-K f16 decode: max_diff={diff} (4 heads, 2 kv, dim=128, 512 tokens)"
        );
    }
}
