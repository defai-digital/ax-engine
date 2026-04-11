//! Kernel dispatch helpers — encode and submit compute commands.
//!
//! Provides dispatch dimension calculations and matmul kernel dispatch.

use std::sync::OnceLock;

use anyhow::Context;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use crate::buffer::MetalBuffer;
use crate::device::MetalDevice;
use crate::pipeline::{ComputePipeline, FunctionConstant, FunctionConstantValue};
use crate::profile::{KernelProfile, MatvecProfileVariant, ProfileKernelMode};

mod attention;
mod common;
mod dequant;
mod elementwise;
mod gdn;
mod matmul;
#[cfg(test)]
mod tests;

pub use attention::AttentionKernels;
pub use common::{DispatchDims, SmartBarrier, barrier_buffers};
pub use dequant::DequantKernels;
pub use elementwise::ElementwiseKernels;
pub use gdn::GdnKernels;
pub use matmul::MatmulKernels;

use common::{bind_buffers, bind_buffers7, bind_f32, bind_u32};
#[cfg(test)]
use dequant::{DB_TILE_M, SB_TG, SB_TILE_N};
use dequant::{
    DEQUANT_MATVEC_Q4K_ILP4_TG, DEQUANT_MATVEC_Q4K_NR2_TG, DEQUANT_MATVEC_Q5K_TG,
    DEQUANT_MATVEC_Q6K_NR2_TG, DEQUANT_MATVEC_Q8_0_ILP4_TG, DEQUANT_MATVEC_Q8_0_NR2_TG,
    DEQUANT_MATVEC_TG, Q4K_ILP4_ROWS, Q4K_NR2_ROWS, Q5K_NR2_ROWS, Q6K_NR2_ROWS, Q8_0_ILP4_ROWS,
    Q8_0_NR2_ROWS,
};

/// Embedded Metal shader source for matmul kernels.
const MATMUL_SHADER_SRC: &str = include_str!("../../shaders/matmul.metal");

/// Embedded Metal shader source for dequantization kernels.
const DEQUANT_SHADER_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/dequant_runtime.metal"));

/// Embedded Metal shader source for attention kernels.
const ATTENTION_SHADER_SRC: &str = include_str!("../../shaders/attention.metal");

/// Embedded Metal shader source for elementwise kernels.
const ELEMENTWISE_SHADER_SRC: &str = include_str!("../../shaders/elementwise.metal");
/// Embedded Metal shader source for GDN recurrent kernels.
const GDN_SHADER_SRC: &str = include_str!("../../shaders/gdn.metal");

const GDN_CHUNK_THRESHOLD: u32 = 64;

/// Tile size for the general tiled matmul kernel (must match shader constant).
#[allow(dead_code)]
const TILE: usize = 16;

/// Tile size for the simdgroup matmul kernel (must match shader constant SG_BM/SG_BN).
const SG_TILE: usize = 32;

/// Threadgroup size for the simdgroup matmul kernel (4 simdgroups × 32 threads).
const SG_TG: usize = 128;

/// Threadgroup size for the matvec kernel (must match shader constant).
const MATVEC_TG_SIZE: usize = 256;

fn warn_q6k_ilp4_disabled_once() {
    static WARN_ONCE: OnceLock<()> = OnceLock::new();
    WARN_ONCE.get_or_init(|| {
        tracing::warn!(
            "Q6_K ILP4 matvec requested by profile but disabled due to correctness bug; falling back to NR2"
        );
    });
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelMode {
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
    prefill_fa2_auto_min_tokens: 16,
    prefill_fa2_auto_min_base_seq: 16,
    prefill_fa2_hd128_auto_min_tokens: 16,
    decode_splitk_auto_min_tokens: 32,
    decode_splitk_chunk_size: 128,
    decode_sdpa_default: false,
    decode_hd128_n2_default: false,
};

const ATTN_PROFILE_DECODE_BALANCED: AttentionRoutingProfile = AttentionRoutingProfile {
    name: "decode-balanced",
    prefill_fa2_auto_min_tokens: 16,
    prefill_fa2_auto_min_base_seq: 16,
    prefill_fa2_hd128_auto_min_tokens: 16,
    decode_splitk_auto_min_tokens: 32,
    decode_splitk_chunk_size: 128,
    decode_sdpa_default: false,
    decode_hd128_n2_default: true,
};

const ATTN_PROFILE_DECODE_LONG_CONTEXT: AttentionRoutingProfile = AttentionRoutingProfile {
    name: "decode-long-context",
    prefill_fa2_auto_min_tokens: 16,
    prefill_fa2_auto_min_base_seq: 16,
    prefill_fa2_hd128_auto_min_tokens: 16,
    decode_splitk_auto_min_tokens: 32,
    decode_splitk_chunk_size: 128,
    decode_sdpa_default: true,
    decode_hd128_n2_default: true,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttentionRoutingMode {
    Auto,
    Fixed(AttentionRoutingProfile),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionDispatchConfig {
    routing_profile_name: &'static str,
    routing_mode: AttentionRoutingMode,
    prefill_fa2_mode: KernelMode,
    prefill_fa2_hd128_mode: KernelMode,
    prefill_ax_bc64_mode: KernelMode,
    prefill_fa2_auto_min_tokens: u32,
    prefill_fa2_auto_min_base_seq: u32,
    prefill_fa2_hd128_auto_min_tokens: u32,
    prefill_fa2_auto_min_tokens_pinned: bool,
    prefill_fa2_auto_min_base_seq_pinned: bool,
    prefill_fa2_hd128_auto_min_tokens_pinned: bool,
    prefill_ax_bc64_min_tokens: u32,
    decode_splitk_mode: KernelMode,
    decode_splitk_chunk_size: u32,
    decode_splitk_auto_min_tokens: u32,
    decode_splitk_auto_min_tokens_pinned: bool,
    decode_sdpa_default: bool,
    decode_sdpa_default_pinned: bool,
    decode_hd128_n2_default: bool,
    decode_hd128_n2_default_pinned: bool,
}

impl Default for AttentionDispatchConfig {
    fn default() -> Self {
        Self::from_profile(&KernelProfile::default())
    }
}

impl AttentionDispatchConfig {
    pub fn from_profile(profile: &KernelProfile) -> Self {
        let default_profile = KernelProfile::default();
        let routing_override = active_attention_routing_profile_override();
        let (routing_mode, routing_profile_name, routing) = match routing_override {
            Some(profile) => (AttentionRoutingMode::Fixed(profile), profile.name, profile),
            None => (AttentionRoutingMode::Auto, "auto", ATTN_PROFILE_DEFAULT),
        };
        let prefill_fa2_mode = {
            let profile_default = profile_kernel_mode(profile.attention_prefill.fa2_mode.clone());
            let mode = parse_kernel_mode("AX_METAL_PREFILL_FA2_MODE", profile_default);
            match mode {
                KernelMode::Off => {
                    if attention_prefill_fa2_enabled() {
                        KernelMode::On
                    } else {
                        KernelMode::Off
                    }
                }
                _ => mode,
            }
        };
        let prefill_fa2_hd128_mode = {
            let profile_default =
                profile_kernel_mode(profile.attention_prefill.fa2_hd128_mode.clone());
            let mode = parse_kernel_mode("AX_METAL_PREFILL_FA2_HD128_MODE", profile_default);
            match mode {
                KernelMode::Off => {
                    if attention_prefill_fa2_hd128_enabled() {
                        KernelMode::On
                    } else {
                        KernelMode::Off
                    }
                }
                _ => mode,
            }
        };
        let prefill_ax_bc64_mode = {
            let profile_default =
                profile_kernel_mode(profile.attention_prefill.ax_bc64_mode.clone());
            let mode = parse_kernel_mode("AX_METAL_PREFILL_BC64_MODE", profile_default);
            match mode {
                KernelMode::Off => match std::env::var("AX_METAL_PREFILL_BC64")
                    .ok()
                    .and_then(|v| parse_bool_env_flag(&v))
                {
                    Some(true) => KernelMode::On,
                    Some(false) | None => KernelMode::Off,
                },
                _ => mode,
            }
        };
        let prefill_fa2_auto_min_tokens_pinned =
            std::env::var_os("AX_METAL_PREFILL_FA2_AUTO_MIN_TOKENS").is_some()
                || profile.attention_prefill.fa2_auto_min_tokens
                    != default_profile.attention_prefill.fa2_auto_min_tokens;
        let prefill_fa2_auto_min_tokens = parse_positive_u32_env(
            "AX_METAL_PREFILL_FA2_AUTO_MIN_TOKENS",
            if profile.attention_prefill.fa2_auto_min_tokens > 0 {
                profile.attention_prefill.fa2_auto_min_tokens
            } else {
                routing.prefill_fa2_auto_min_tokens
            },
        );
        let prefill_fa2_auto_min_base_seq_pinned =
            std::env::var_os("AX_METAL_PREFILL_FA2_AUTO_MIN_BASE_SEQ").is_some()
                || profile.attention_prefill.fa2_auto_min_base_seq
                    != default_profile.attention_prefill.fa2_auto_min_base_seq;
        let prefill_fa2_auto_min_base_seq = parse_positive_u32_env(
            "AX_METAL_PREFILL_FA2_AUTO_MIN_BASE_SEQ",
            if profile.attention_prefill.fa2_auto_min_base_seq > 0 {
                profile.attention_prefill.fa2_auto_min_base_seq
            } else {
                routing.prefill_fa2_auto_min_base_seq
            },
        );
        let prefill_fa2_hd128_auto_min_tokens_pinned =
            std::env::var_os("AX_METAL_PREFILL_FA2_HD128_AUTO_MIN_TOKENS").is_some()
                || profile.attention_prefill.fa2_hd128_auto_min_tokens
                    != default_profile.attention_prefill.fa2_hd128_auto_min_tokens;
        let prefill_fa2_hd128_auto_min_tokens = parse_positive_u32_env(
            "AX_METAL_PREFILL_FA2_HD128_AUTO_MIN_TOKENS",
            if profile.attention_prefill.fa2_hd128_auto_min_tokens > 0 {
                profile.attention_prefill.fa2_hd128_auto_min_tokens
            } else {
                routing.prefill_fa2_hd128_auto_min_tokens
            },
        );
        let prefill_ax_bc64_min_tokens = parse_positive_u32_env(
            "AX_METAL_PREFILL_BC64_MIN_TOKENS",
            profile.attention_prefill.ax_bc64_min_tokens,
        );
        let decode_splitk_mode = parse_kernel_mode("AX_METAL_DECODE_SPLITK_MODE", KernelMode::Auto);
        let decode_splitk_chunk_size = parse_positive_u32_env(
            "AX_METAL_DECODE_SPLITK_CHUNK_SIZE",
            if profile.attention_decode.splitk_chunk_size > 0 {
                profile.attention_decode.splitk_chunk_size
            } else {
                routing.decode_splitk_chunk_size
            },
        );
        let decode_splitk_auto_min_tokens_pinned =
            std::env::var_os("AX_METAL_DECODE_SPLITK_AUTO_MIN_TOKENS").is_some()
                || profile.attention_decode.splitk_threshold
                    != default_profile.attention_decode.splitk_threshold;
        let decode_splitk_auto_min_tokens = parse_positive_u32_env(
            "AX_METAL_DECODE_SPLITK_AUTO_MIN_TOKENS",
            if profile.attention_decode.splitk_threshold > 0 {
                profile.attention_decode.splitk_threshold
            } else {
                routing.decode_splitk_auto_min_tokens
            },
        );
        let decode_sdpa_default_pinned = std::env::var_os("AX_METAL_DECODE_SDPA").is_some()
            || profile.attention_decode.sdpa_default
                != default_profile.attention_decode.sdpa_default;
        let decode_sdpa_default = match std::env::var("AX_METAL_DECODE_SDPA").ok() {
            Some(v) => parse_bool_env_flag(&v).unwrap_or(false),
            None => profile
                .attention_decode
                .sdpa_default
                .unwrap_or(routing.decode_sdpa_default),
        };
        let decode_hd128_n2_default_pinned = std::env::var_os("AX_METAL_DECODE_HD128_N2").is_some()
            || profile.attention_decode.hd128_n2_default
                != default_profile.attention_decode.hd128_n2_default;
        let decode_hd128_n2_default = match std::env::var("AX_METAL_DECODE_HD128_N2").ok() {
            Some(v) => parse_bool_env_flag(&v).unwrap_or(false),
            None => profile
                .attention_decode
                .hd128_n2_default
                .unwrap_or(routing.decode_hd128_n2_default),
        };

        Self {
            routing_profile_name,
            routing_mode,
            prefill_fa2_mode,
            prefill_fa2_hd128_mode,
            prefill_ax_bc64_mode,
            prefill_fa2_auto_min_tokens,
            prefill_fa2_auto_min_base_seq,
            prefill_fa2_hd128_auto_min_tokens,
            prefill_fa2_auto_min_tokens_pinned,
            prefill_fa2_auto_min_base_seq_pinned,
            prefill_fa2_hd128_auto_min_tokens_pinned,
            prefill_ax_bc64_min_tokens,
            decode_splitk_mode,
            decode_splitk_chunk_size,
            decode_splitk_auto_min_tokens,
            decode_splitk_auto_min_tokens_pinned,
            decode_sdpa_default,
            decode_sdpa_default_pinned,
            decode_hd128_n2_default,
            decode_hd128_n2_default_pinned,
        }
    }

    pub fn routing_profile_name(&self) -> &'static str {
        self.routing_profile_name
    }

    fn effective_decode_routing_profile(&self, attend_len: u32) -> AttentionRoutingProfile {
        match self.routing_mode {
            AttentionRoutingMode::Fixed(profile) => profile,
            AttentionRoutingMode::Auto => {
                if attend_len >= 4096 {
                    ATTN_PROFILE_DECODE_LONG_CONTEXT
                } else if attend_len >= 256 {
                    ATTN_PROFILE_DECODE_BALANCED
                } else {
                    ATTN_PROFILE_DEFAULT
                }
            }
        }
    }

    fn effective_prefill_local_routing_profile(&self, n_tokens: u32) -> AttentionRoutingProfile {
        match self.routing_mode {
            AttentionRoutingMode::Fixed(profile) => profile,
            AttentionRoutingMode::Auto => {
                if n_tokens >= 768 {
                    ATTN_PROFILE_DECODE_BALANCED
                } else {
                    ATTN_PROFILE_DEFAULT
                }
            }
        }
    }

    fn effective_prefill_cached_routing_profile(
        &self,
        n_tokens: u32,
        base_seq_len: u32,
        sliding_window: u32,
    ) -> AttentionRoutingProfile {
        match self.routing_mode {
            AttentionRoutingMode::Fixed(profile) => profile,
            AttentionRoutingMode::Auto => {
                if sliding_window > 0 || base_seq_len >= 2048 {
                    ATTN_PROFILE_DECODE_LONG_CONTEXT
                } else if base_seq_len >= 256 || n_tokens >= 256 {
                    ATTN_PROFILE_DECODE_BALANCED
                } else {
                    ATTN_PROFILE_DEFAULT
                }
            }
        }
    }

    fn effective_decode_splitk_auto_min_tokens(&self, attend_len: u32) -> u32 {
        if self.decode_splitk_auto_min_tokens_pinned {
            self.decode_splitk_auto_min_tokens
        } else {
            self.effective_decode_routing_profile(attend_len)
                .decode_splitk_auto_min_tokens
        }
    }

    fn effective_decode_sdpa_default(&self, attend_len: u32) -> bool {
        if self.decode_sdpa_default_pinned {
            self.decode_sdpa_default
        } else {
            self.effective_decode_routing_profile(attend_len)
                .decode_sdpa_default
        }
    }

    fn effective_decode_hd128_n2_default(&self, attend_len: u32) -> bool {
        if self.decode_hd128_n2_default_pinned {
            self.decode_hd128_n2_default
        } else {
            self.effective_decode_routing_profile(attend_len)
                .decode_hd128_n2_default
        }
    }

    fn effective_prefill_fa2_auto_min_tokens_cached(
        &self,
        n_tokens: u32,
        base_seq_len: u32,
        sliding_window: u32,
    ) -> u32 {
        if self.prefill_fa2_auto_min_tokens_pinned {
            self.prefill_fa2_auto_min_tokens
        } else {
            self.effective_prefill_cached_routing_profile(n_tokens, base_seq_len, sliding_window)
                .prefill_fa2_auto_min_tokens
        }
    }

    fn effective_prefill_fa2_auto_min_base_seq_cached(
        &self,
        n_tokens: u32,
        base_seq_len: u32,
        sliding_window: u32,
    ) -> u32 {
        if self.prefill_fa2_auto_min_base_seq_pinned {
            self.prefill_fa2_auto_min_base_seq
        } else {
            self.effective_prefill_cached_routing_profile(n_tokens, base_seq_len, sliding_window)
                .prefill_fa2_auto_min_base_seq
        }
    }

    fn effective_prefill_fa2_hd128_auto_min_tokens_local(&self, n_tokens: u32) -> u32 {
        if self.prefill_fa2_hd128_auto_min_tokens_pinned {
            self.prefill_fa2_hd128_auto_min_tokens
        } else {
            self.effective_prefill_local_routing_profile(n_tokens)
                .prefill_fa2_hd128_auto_min_tokens
        }
    }

    pub fn decode_routing_profile_name(&self, attend_len: u32) -> &'static str {
        self.effective_decode_routing_profile(attend_len).name
    }

    pub fn prefill_local_routing_profile_name(&self, n_tokens: u32) -> &'static str {
        self.effective_prefill_local_routing_profile(n_tokens).name
    }

    pub fn prefill_cached_routing_profile_name(
        &self,
        n_tokens: u32,
        base_seq_len: u32,
        sliding_window: u32,
    ) -> &'static str {
        self.effective_prefill_cached_routing_profile(n_tokens, base_seq_len, sliding_window)
            .name
    }

    pub fn decode_splitk_chunk_size(&self) -> u32 {
        self.decode_splitk_chunk_size
    }

    pub fn with_decode_splitk_mode(mut self, mode: KernelMode) -> Self {
        self.decode_splitk_mode = mode;
        self
    }

    pub fn with_prefill_fa2_mode(mut self, mode: KernelMode) -> Self {
        self.prefill_fa2_mode = mode;
        self
    }

    pub fn with_prefill_fa2_hd128_mode(mut self, mode: KernelMode) -> Self {
        self.prefill_fa2_hd128_mode = mode;
        self
    }

    pub fn with_decode_sdpa_default(mut self, enabled: bool) -> Self {
        self.decode_sdpa_default = enabled;
        self
    }

    pub fn with_decode_hd128_n2_default(mut self, enabled: bool) -> Self {
        self.decode_hd128_n2_default = enabled;
        self
    }

    pub fn decode_route_label(self, kv_f16: bool, head_dim: u32, attend_len: u32) -> &'static str {
        self.decode_candidate_selection(kv_f16, head_dim, attend_len)
            .label()
    }

    pub fn decode_candidate_selection(
        self,
        kv_f16: bool,
        head_dim: u32,
        attend_len: u32,
    ) -> AttentionDecodeCandidateSelection {
        attention_decode_candidate_selection(kv_f16, head_dim, attend_len, self)
    }

    pub fn prefill_local_route_label(self, n_tokens: u32, head_dim: u32) -> &'static str {
        self.prefill_local_candidate_selection(n_tokens, head_dim)
            .label()
    }

    pub fn prefill_local_candidate_selection(
        self,
        n_tokens: u32,
        head_dim: u32,
    ) -> AttentionPrefillCandidateSelection {
        attention_prefill_local_candidate_selection(n_tokens, head_dim, self)
    }

    pub fn prefill_cached_route_label(
        self,
        kv_f16: bool,
        n_tokens: u32,
        head_dim: u32,
        base_seq_len: u32,
        sliding_window: u32,
    ) -> &'static str {
        self.prefill_cached_candidate_selection(
            kv_f16,
            n_tokens,
            head_dim,
            base_seq_len,
            sliding_window,
        )
        .label()
    }

    pub fn prefill_cached_candidate_selection(
        self,
        kv_f16: bool,
        n_tokens: u32,
        head_dim: u32,
        base_seq_len: u32,
        sliding_window: u32,
    ) -> AttentionPrefillCandidateSelection {
        attention_prefill_cached_candidate_selection(
            kv_f16,
            n_tokens,
            head_dim,
            base_seq_len,
            sliding_window,
            self,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DequantDispatchConfig {
    pub q4_k_threadgroup_size: usize,
    pub q4_k_rows_per_simdgroup: u32,
    pub q4_k_variant: Option<MatvecProfileVariant>,
    pub q5_k_rows_per_simdgroup: u32,
    pub q5_k_ilp4: bool,
    pub q5_k_variant: Option<MatvecProfileVariant>,
    pub q6_k_threadgroup_size: usize,
    pub q6_k_rows_per_simdgroup: u32,
    pub q6_k_variant: Option<MatvecProfileVariant>,
    pub q8_0_variant: Option<MatvecProfileVariant>,
    pub batch_f16in_small_n_threshold: u32,
    pub batch_f16in_small_m_max: u32,
    pub batch_f16in_use_bn32: bool,
    pub batch_f16in_use_bk32: bool,
    pub q8_f16in_full_min_n: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelStabilityTier {
    Stable,
    ProfilePreferred,
    Experimental,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatvecCandidate {
    Q4KBase,
    Q4KNr2,
    Q5KBase,
    Q5KIlp4,
    Q5KNr2,
    Q8_0Base,
    Q8_0Nr2,
    Q8_0Ilp4,
    Q4KIlp4,
    Q6KBase,
    Q6KNr2,
    Q6KIlp4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatvecCandidateSelection {
    pub candidate: MatvecCandidate,
    pub stability: KernelStabilityTier,
    pub threadgroups: usize,
    pub threadgroup_width: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionDecodeCandidate {
    Decode,
    DecodeV2,
    Hd128,
    Hd256,
    F16Kv,
    F16KvHd128,
    F16KvHd128N2,
    F16KvHd256,
    SdpaHd256,
    SplitKHd128,
    SplitKHd256,
    SdpaParallelHd128,
    SdpaParallelHd256,
    SdpaGqaHd128,
    SdpaGqaHd256,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionDecodeCandidateSelection {
    pub candidate: AttentionDecodeCandidate,
    pub stability: KernelStabilityTier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionPrefillCandidate {
    Prefill,
    PrefillHd256,
    PrefillV2,
    PrefillV2Hd128,
    AxHd128,
    AxBc64,
    AxSmem,
    AxSmemF16,
    Fa2Hd128,
    Fa2SimdHd128,
    Fa2HalfHd128,
    Fa2v2Hd128,
    Fa2v2Hd256,
    Fa2SimdHd256,
    Fa2SimdHd64,
    Cache,
    CacheFa2Hd256,
    CacheFa2SimdHd128,
    CacheFa2SimdHd64,
    CacheFa2SimdHd256,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionPrefillCandidateSelection {
    pub candidate: AttentionPrefillCandidate,
    pub stability: KernelStabilityTier,
}

impl Default for DequantDispatchConfig {
    fn default() -> Self {
        Self::from_profile(&KernelProfile::default())
    }
}

impl DequantDispatchConfig {
    pub fn from_profile(profile: &KernelProfile) -> Self {
        let q4_params = profile.matvec_params("q4_k");
        let q5_params = profile.matvec_params("q5_k");
        let q6_params = profile.matvec_params("q6_k");
        let q4_profile_override = profile.decode_matvec.get("q4_k");
        let q5_profile_override = profile.decode_matvec.get("q5_k");
        let q6_profile_override = profile.decode_matvec.get("q6_k");
        let q8_profile_override = profile.decode_matvec.get("q8_0");

        let q4_k_threadgroup_size = q4_params.threadgroup_size as usize;
        let q4_k_rows_per_simdgroup = q4_params.rows_per_simdgroup;
        let q4_k_variant = q4_profile_override.map(|params| {
            params.variant.unwrap_or(
                if params.rows_per_simdgroup >= 2
                    || params.threadgroup_size == DEQUANT_MATVEC_Q4K_NR2_TG as u32
                {
                    MatvecProfileVariant::Nr2
                } else {
                    MatvecProfileVariant::Base
                },
            )
        });

        let q5_k_rows_per_simdgroup = q5_params.rows_per_simdgroup;
        let q5_k_variant = q5_profile_override.map(|params| {
            params.variant.unwrap_or(if params.rows_per_simdgroup >= 2 {
                MatvecProfileVariant::Nr2
            } else {
                MatvecProfileVariant::Base
            })
        });
        let q5_k_ilp4 = matches!(q5_k_variant, Some(MatvecProfileVariant::Ilp4));

        let q6_k_threadgroup_size = q6_params.threadgroup_size as usize;
        let q6_k_rows_per_simdgroup = q6_params.rows_per_simdgroup;
        let q6_k_variant = q6_profile_override.map(|params| {
            params.variant.unwrap_or(
                if params.rows_per_simdgroup >= 2
                    || params.threadgroup_size == DEQUANT_MATVEC_Q6K_NR2_TG as u32
                {
                    MatvecProfileVariant::Nr2
                } else {
                    MatvecProfileVariant::Base
                },
            )
        });

        let q8_0_variant = q8_profile_override.map(|params| {
            params.variant.unwrap_or(if params.rows_per_simdgroup >= 2 {
                MatvecProfileVariant::Nr2
            } else if params.threadgroup_size == DEQUANT_MATVEC_Q8_0_ILP4_TG as u32 {
                MatvecProfileVariant::Ilp4
            } else {
                MatvecProfileVariant::Base
            })
        });

        Self {
            q4_k_threadgroup_size,
            q4_k_rows_per_simdgroup,
            q4_k_variant,
            q5_k_rows_per_simdgroup,
            q5_k_ilp4,
            q5_k_variant,
            q6_k_threadgroup_size,
            q6_k_rows_per_simdgroup,
            q6_k_variant,
            q8_0_variant,
            batch_f16in_small_n_threshold: profile.batch_prefill.small_n_threshold,
            batch_f16in_small_m_max: profile.batch_prefill.small_m_max,
            batch_f16in_use_bn32: profile.batch_prefill.use_bn32,
            batch_f16in_use_bk32: profile.batch_prefill.use_bk32,
            q8_f16in_full_min_n: profile.batch_prefill.q8_f16in_full_min_n.max(1) as usize,
        }
    }

    pub fn q4_k_preference_label(self) -> &'static str {
        if self.q4_k_rows_per_simdgroup >= 2
            || self.q4_k_threadgroup_size == DEQUANT_MATVEC_Q4K_NR2_TG
        {
            "nr2-preferred"
        } else {
            "default-preferred"
        }
    }

    pub fn q6_k_preference_label(self) -> &'static str {
        if self.q6_k_rows_per_simdgroup >= 2
            || self.q6_k_threadgroup_size == DEQUANT_MATVEC_Q6K_NR2_TG
        {
            "nr2-preferred"
        } else {
            "default-preferred"
        }
    }
}

impl KernelStabilityTier {
    pub fn label(self) -> &'static str {
        match self {
            Self::Stable => "stable",
            Self::ProfilePreferred => "profile_preferred",
            Self::Experimental => "experimental",
        }
    }
}

impl MatvecCandidate {
    pub fn label(self) -> &'static str {
        match self {
            Self::Q4KBase => "q4_k.base",
            Self::Q4KNr2 => "q4_k.nr2",
            Self::Q5KBase => "q5_k.base",
            Self::Q5KIlp4 => "q5_k.ilp4",
            Self::Q5KNr2 => "q5_k.nr2",
            Self::Q8_0Base => "q8_0.base",
            Self::Q8_0Nr2 => "q8_0.nr2",
            Self::Q8_0Ilp4 => "q8_0.ilp4",
            Self::Q4KIlp4 => "q4_k.ilp4",
            Self::Q6KBase => "q6_k.base",
            Self::Q6KNr2 => "q6_k.nr2",
            Self::Q6KIlp4 => "q6_k.ilp4",
        }
    }
}

impl MatvecCandidateSelection {
    pub fn label(self) -> &'static str {
        self.candidate.label()
    }
}

impl AttentionDecodeCandidate {
    pub fn label(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::DecodeV2 => "decode_v2",
            Self::Hd128 => "hd128",
            Self::Hd256 => "hd256",
            Self::F16Kv => "f16kv",
            Self::F16KvHd128 => "f16kv_hd128",
            Self::F16KvHd128N2 => "f16kv_hd128_n2",
            Self::F16KvHd256 => "f16kv_hd256",
            Self::SdpaHd256 => "sdpa_hd256",
            Self::SplitKHd128 => "splitk_hd128",
            Self::SplitKHd256 => "splitk_hd256",
            Self::SdpaParallelHd128 => "sdpa_parallel_hd128",
            Self::SdpaParallelHd256 => "sdpa_parallel_hd256",
            Self::SdpaGqaHd128 => "sdpa_gqa_hd128",
            Self::SdpaGqaHd256 => "sdpa_gqa_hd256",
        }
    }

    fn is_splitk(self) -> bool {
        matches!(self, Self::SplitKHd128 | Self::SplitKHd256)
    }
}

impl AttentionDecodeCandidateSelection {
    pub fn label(self) -> &'static str {
        self.candidate.label()
    }
}

impl AttentionPrefillCandidate {
    pub fn label(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::PrefillHd256 => "prefill_hd256",
            Self::PrefillV2 => "prefill_v2",
            Self::PrefillV2Hd128 => "prefill_v2_hd128",
            Self::AxHd128 => "ax_hd128",
            Self::AxBc64 => "ax_bc64",
            Self::AxSmem => "ax_smem",
            Self::AxSmemF16 => "ax_smem_f16",
            Self::Fa2Hd128 => "fa2_hd128",
            Self::Fa2SimdHd128 => "fa2_simd_hd128",
            Self::Fa2HalfHd128 => "fa2_half_hd128",
            Self::Fa2v2Hd128 => "fa2v2_hd128",
            Self::Fa2v2Hd256 => "fa2v2_hd256",
            Self::Fa2SimdHd256 => "fa2_simd_hd256",
            Self::Fa2SimdHd64 => "fa2_simd_hd64",
            Self::Cache => "cache",
            Self::CacheFa2Hd256 => "cache_fa2_hd256",
            Self::CacheFa2SimdHd128 => "cache_fa2_simd_hd128",
            Self::CacheFa2SimdHd64 => "cache_fa2_simd_hd64",
            Self::CacheFa2SimdHd256 => "cache_fa2_simd_hd256",
        }
    }
}

impl AttentionPrefillCandidateSelection {
    pub fn label(self) -> &'static str {
        self.candidate.label()
    }
}

fn q4_k_matvec_candidate_selection(
    m: u32,
    config: DequantDispatchConfig,
) -> MatvecCandidateSelection {
    if let Some(variant) = config.q4_k_variant {
        match variant {
            MatvecProfileVariant::Nr2 if m >= 2 => MatvecCandidateSelection {
                candidate: MatvecCandidate::Q4KNr2,
                stability: KernelStabilityTier::ProfilePreferred,
                threadgroups: (m as usize).div_ceil(Q4K_NR2_ROWS),
                threadgroup_width: DEQUANT_MATVEC_Q4K_NR2_TG,
            },
            MatvecProfileVariant::Ilp4 => MatvecCandidateSelection {
                candidate: MatvecCandidate::Q4KIlp4,
                stability: KernelStabilityTier::ProfilePreferred,
                threadgroups: (m as usize).div_ceil(Q4K_ILP4_ROWS),
                threadgroup_width: DEQUANT_MATVEC_Q4K_ILP4_TG,
            },
            MatvecProfileVariant::Base => MatvecCandidateSelection {
                candidate: MatvecCandidate::Q4KBase,
                stability: KernelStabilityTier::ProfilePreferred,
                threadgroups: m as usize,
                threadgroup_width: DEQUANT_MATVEC_TG,
            },
            _ => MatvecCandidateSelection {
                candidate: MatvecCandidate::Q4KBase,
                stability: KernelStabilityTier::ProfilePreferred,
                threadgroups: m as usize,
                threadgroup_width: DEQUANT_MATVEC_TG,
            },
        }
    } else if matvec_q4k_nr2_enabled() && m >= 2 {
        MatvecCandidateSelection {
            candidate: MatvecCandidate::Q4KNr2,
            stability: KernelStabilityTier::ProfilePreferred,
            threadgroups: (m as usize).div_ceil(Q4K_NR2_ROWS),
            threadgroup_width: DEQUANT_MATVEC_Q4K_NR2_TG,
        }
    } else if m < 2 {
        MatvecCandidateSelection {
            candidate: MatvecCandidate::Q4KBase,
            stability: KernelStabilityTier::Stable,
            threadgroups: m as usize,
            threadgroup_width: DEQUANT_MATVEC_TG,
        }
    } else if m <= 4096 {
        MatvecCandidateSelection {
            candidate: MatvecCandidate::Q4KNr2,
            stability: KernelStabilityTier::Stable,
            threadgroups: (m as usize).div_ceil(Q4K_NR2_ROWS),
            threadgroup_width: DEQUANT_MATVEC_Q4K_NR2_TG,
        }
    } else {
        MatvecCandidateSelection {
            candidate: MatvecCandidate::Q4KIlp4,
            stability: KernelStabilityTier::Stable,
            threadgroups: (m as usize).div_ceil(Q4K_ILP4_ROWS),
            threadgroup_width: DEQUANT_MATVEC_Q4K_ILP4_TG,
        }
    }
}

fn q6_k_matvec_candidate_selection(
    m: u32,
    config: DequantDispatchConfig,
) -> MatvecCandidateSelection {
    match config.q6_k_variant {
        Some(MatvecProfileVariant::Nr2) if m >= 2 => {
            return MatvecCandidateSelection {
                candidate: MatvecCandidate::Q6KNr2,
                stability: KernelStabilityTier::ProfilePreferred,
                threadgroups: (m as usize).div_ceil(Q6K_NR2_ROWS),
                threadgroup_width: DEQUANT_MATVEC_Q6K_NR2_TG,
            };
        }
        Some(MatvecProfileVariant::Ilp4) => {
            // Q6_K ILP4 has a correctness bug (only covers 16/32 positions per
            // sub-group). Fall through to NR2 instead of using ILP4.
            warn_q6k_ilp4_disabled_once();
        }
        Some(MatvecProfileVariant::Base) => {
            return MatvecCandidateSelection {
                candidate: MatvecCandidate::Q6KBase,
                stability: KernelStabilityTier::ProfilePreferred,
                threadgroups: m as usize,
                threadgroup_width: DEQUANT_MATVEC_TG,
            };
        }
        _ => {}
    }
    if matvec_q6k_nr2_enabled() && m >= 2 {
        return MatvecCandidateSelection {
            candidate: MatvecCandidate::Q6KNr2,
            stability: KernelStabilityTier::ProfilePreferred,
            threadgroups: (m as usize).div_ceil(Q6K_NR2_ROWS),
            threadgroup_width: DEQUANT_MATVEC_Q6K_NR2_TG,
        };
    }
    // Dimension-aware auto-selection when no profile preference.
    // NOTE: Q6_K ILP4 is disabled — the kernel's thread decomposition
    // (tid = lane/4, il = tid%4) only covers positions 0-15 per
    // 32-element sub-group, missing positions 16-31.  Use NR2 for all M≥2.
    if m < 2 {
        MatvecCandidateSelection {
            candidate: MatvecCandidate::Q6KBase,
            stability: KernelStabilityTier::Stable,
            threadgroups: m as usize,
            threadgroup_width: DEQUANT_MATVEC_TG,
        }
    } else {
        MatvecCandidateSelection {
            candidate: MatvecCandidate::Q6KNr2,
            stability: KernelStabilityTier::Stable,
            threadgroups: (m as usize).div_ceil(Q6K_NR2_ROWS),
            threadgroup_width: DEQUANT_MATVEC_Q6K_NR2_TG,
        }
    }
}

fn q5_k_matvec_candidate_selection(
    m: u32,
    config: DequantDispatchConfig,
) -> MatvecCandidateSelection {
    match config.q5_k_variant {
        Some(MatvecProfileVariant::Nr2) if m >= 2 => {
            return MatvecCandidateSelection {
                candidate: MatvecCandidate::Q5KNr2,
                stability: KernelStabilityTier::ProfilePreferred,
                threadgroups: (m as usize).div_ceil(Q5K_NR2_ROWS),
                threadgroup_width: DEQUANT_MATVEC_Q5K_TG,
            };
        }
        Some(MatvecProfileVariant::Ilp4) => {
            return MatvecCandidateSelection {
                candidate: MatvecCandidate::Q5KIlp4,
                stability: KernelStabilityTier::ProfilePreferred,
                threadgroups: (m as usize).div_ceil(2),
                threadgroup_width: DEQUANT_MATVEC_Q5K_TG,
            };
        }
        Some(MatvecProfileVariant::Base) => {
            return MatvecCandidateSelection {
                candidate: MatvecCandidate::Q5KBase,
                stability: KernelStabilityTier::ProfilePreferred,
                threadgroups: (m as usize).div_ceil(2),
                threadgroup_width: DEQUANT_MATVEC_Q5K_TG,
            };
        }
        _ => {}
    }
    if config.q5_k_rows_per_simdgroup >= 2 && m >= 2 {
        return MatvecCandidateSelection {
            candidate: MatvecCandidate::Q5KNr2,
            stability: KernelStabilityTier::ProfilePreferred,
            threadgroups: (m as usize).div_ceil(Q5K_NR2_ROWS),
            threadgroup_width: DEQUANT_MATVEC_Q5K_TG,
        };
    }
    if config.q5_k_ilp4 {
        return MatvecCandidateSelection {
            candidate: MatvecCandidate::Q5KIlp4,
            stability: KernelStabilityTier::ProfilePreferred,
            threadgroups: (m as usize).div_ceil(2),
            threadgroup_width: DEQUANT_MATVEC_Q5K_TG,
        };
    }
    // Dimension-aware auto-selection when no profile preference.
    if (2..=4096).contains(&m) {
        MatvecCandidateSelection {
            candidate: MatvecCandidate::Q5KNr2,
            stability: KernelStabilityTier::Stable,
            threadgroups: (m as usize).div_ceil(Q5K_NR2_ROWS),
            threadgroup_width: DEQUANT_MATVEC_Q5K_TG,
        }
    } else if m > 4096 {
        MatvecCandidateSelection {
            candidate: MatvecCandidate::Q5KIlp4,
            stability: KernelStabilityTier::Stable,
            threadgroups: (m as usize).div_ceil(2),
            threadgroup_width: DEQUANT_MATVEC_Q5K_TG,
        }
    } else {
        MatvecCandidateSelection {
            candidate: MatvecCandidate::Q5KBase,
            stability: KernelStabilityTier::Stable,
            threadgroups: (m as usize).div_ceil(2),
            threadgroup_width: DEQUANT_MATVEC_Q5K_TG,
        }
    }
}

fn q8_0_matvec_candidate_selection(
    m: u32,
    config: DequantDispatchConfig,
) -> MatvecCandidateSelection {
    // Explicit profile variant takes priority.
    match config.q8_0_variant {
        Some(MatvecProfileVariant::Nr2) if m >= 2 => {
            return MatvecCandidateSelection {
                candidate: MatvecCandidate::Q8_0Nr2,
                stability: KernelStabilityTier::ProfilePreferred,
                threadgroups: (m as usize).div_ceil(Q8_0_NR2_ROWS),
                threadgroup_width: DEQUANT_MATVEC_Q8_0_NR2_TG,
            };
        }
        Some(MatvecProfileVariant::Ilp4) => {
            return MatvecCandidateSelection {
                candidate: MatvecCandidate::Q8_0Ilp4,
                stability: KernelStabilityTier::ProfilePreferred,
                threadgroups: (m as usize).div_ceil(Q8_0_ILP4_ROWS),
                threadgroup_width: DEQUANT_MATVEC_Q8_0_ILP4_TG,
            };
        }
        Some(MatvecProfileVariant::Base) => {
            return MatvecCandidateSelection {
                candidate: MatvecCandidate::Q8_0Base,
                stability: KernelStabilityTier::ProfilePreferred,
                threadgroups: m as usize,
                threadgroup_width: DEQUANT_MATVEC_TG,
            };
        }
        _ => {}
    }
    // Dimension-aware auto-selection: nr2 for small/medium M (x-vector
    // reuse dominates), ilp4 for large M (block interleaving reduces
    // memory stalls). Threshold ~4096 rows aligns with typical FFN gate/up
    // projection sizes where ILP starts to win.
    if m < 2 {
        MatvecCandidateSelection {
            candidate: MatvecCandidate::Q8_0Base,
            stability: KernelStabilityTier::Stable,
            threadgroups: m as usize,
            threadgroup_width: DEQUANT_MATVEC_TG,
        }
    } else if m <= 4096 {
        MatvecCandidateSelection {
            candidate: MatvecCandidate::Q8_0Nr2,
            stability: KernelStabilityTier::Stable,
            threadgroups: (m as usize).div_ceil(Q8_0_NR2_ROWS),
            threadgroup_width: DEQUANT_MATVEC_Q8_0_NR2_TG,
        }
    } else {
        MatvecCandidateSelection {
            candidate: MatvecCandidate::Q8_0Ilp4,
            stability: KernelStabilityTier::Stable,
            threadgroups: (m as usize).div_ceil(Q8_0_ILP4_ROWS),
            threadgroup_width: DEQUANT_MATVEC_Q8_0_ILP4_TG,
        }
    }
}

fn sdpa_parallel_decode_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("AX_METAL_DECODE_SDPA_PARALLEL")
            .ok()
            .and_then(|v| parse_bool_env_flag(&v))
            .unwrap_or(false) // default OFF; kernel has no correctness tests and may produce wrong output
    })
}

fn sdpa_gqa_decode_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("AX_METAL_DECODE_SDPA_GQA")
            .ok()
            .and_then(|v| parse_bool_env_flag(&v))
            .unwrap_or(false)
    })
}

fn attention_decode_candidate_selection(
    kv_f16: bool,
    head_dim: u32,
    attend_len: u32,
    config: AttentionDispatchConfig,
) -> AttentionDecodeCandidateSelection {
    if attention_decode_splitk_should_use_with_config(kv_f16, head_dim, attend_len, config) {
        return AttentionDecodeCandidateSelection {
            candidate: match head_dim {
                256 => AttentionDecodeCandidate::SplitKHd256,
                _ => AttentionDecodeCandidate::SplitKHd128,
            },
            stability: KernelStabilityTier::ProfilePreferred,
        };
    }

    let use_v2 = attention_decode_v2_enabled(attend_len);
    let use_hd256 = head_dim == 256;
    let use_hd128 = head_dim == 128;
    let use_sdpa =
        kv_f16 && use_hd256 && attention_decode_sdpa_enabled_with_config(config, attend_len);
    let use_hd128_n2 =
        kv_f16 && use_hd128 && attention_decode_hd128_n2_enabled_with_config(config, attend_len);

    if use_sdpa {
        AttentionDecodeCandidateSelection {
            candidate: AttentionDecodeCandidate::SdpaHd256,
            stability: KernelStabilityTier::ProfilePreferred,
        }
    } else if kv_f16 {
        if use_hd256 {
            AttentionDecodeCandidateSelection {
                candidate: AttentionDecodeCandidate::F16KvHd256,
                stability: KernelStabilityTier::Stable,
            }
        } else if use_hd128_n2 {
            AttentionDecodeCandidateSelection {
                candidate: AttentionDecodeCandidate::F16KvHd128N2,
                stability: KernelStabilityTier::ProfilePreferred,
            }
        } else if use_hd128 {
            AttentionDecodeCandidateSelection {
                candidate: AttentionDecodeCandidate::F16KvHd128,
                stability: KernelStabilityTier::Stable,
            }
        } else {
            AttentionDecodeCandidateSelection {
                candidate: AttentionDecodeCandidate::F16Kv,
                stability: KernelStabilityTier::Stable,
            }
        }
    } else if use_hd256 {
        AttentionDecodeCandidateSelection {
            candidate: AttentionDecodeCandidate::Hd256,
            stability: KernelStabilityTier::Stable,
        }
    } else if use_hd128 {
        AttentionDecodeCandidateSelection {
            candidate: AttentionDecodeCandidate::Hd128,
            stability: KernelStabilityTier::Stable,
        }
    } else if use_v2 {
        AttentionDecodeCandidateSelection {
            candidate: AttentionDecodeCandidate::DecodeV2,
            stability: KernelStabilityTier::Experimental,
        }
    } else {
        AttentionDecodeCandidateSelection {
            candidate: AttentionDecodeCandidate::Decode,
            stability: KernelStabilityTier::Stable,
        }
    }
}

fn attention_prefill_local_candidate_selection(
    n_tokens: u32,
    head_dim: u32,
    config: AttentionDispatchConfig,
) -> AttentionPrefillCandidateSelection {
    let use_v2 = attention_prefill_v2_enabled();
    let use_v2_hd128 = attention_prefill_hd128_enabled() && head_dim == 128;
    let use_ax_hd128 = attention_prefill_ax_hd128_enabled() && head_dim == 128;
    let use_ax_bc64 =
        use_ax_hd128 && attention_prefill_ax_bc64_should_use_with_config(n_tokens, config);
    let use_ax_smem = attention_prefill_ax_smem_enabled() && use_ax_hd128;
    let use_ax_smem_f16 = attention_prefill_ax_smem_f16_enabled() && use_ax_hd128;
    let use_fa2_hd64_or_hd128 =
        attention_prefill_fa2_hd64_or_hd128_should_use_with_config(n_tokens, head_dim, config);
    let use_fa2_hd128 = use_fa2_hd64_or_hd128 && head_dim == 128;
    // The simd kernel is an implementation variant of the FA2 HD128 route and
    // must not bypass the profile's mode/threshold gating.
    let use_fa2_simd_hd128 = use_fa2_hd128 && attention_prefill_fa2_simd_hd128_enabled();
    let use_fa2_simd_hd64 =
        use_fa2_hd64_or_hd128 && head_dim == 64 && attention_prefill_fa2_simd_hd128_enabled();
    let use_hd256 = attention_prefill_hd256_enabled() && head_dim == 256 && !use_v2;

    if attention_prefill_fa2v2_enabled() && head_dim == 256 {
        return AttentionPrefillCandidateSelection {
            candidate: AttentionPrefillCandidate::Fa2v2Hd256,
            stability: KernelStabilityTier::Experimental,
        };
    }

    if attention_prefill_fa2v2_enabled() && head_dim == 128 {
        return AttentionPrefillCandidateSelection {
            candidate: AttentionPrefillCandidate::Fa2v2Hd128,
            stability: KernelStabilityTier::Experimental,
        };
    }

    // Full-half FA2 (K/V staged as half, half×half→float MMA) is the
    // fastest HD=128 path — matches llama.cpp's flash_attn_ext strategy.
    if attention_prefill_fa2_half_hd128_enabled() && use_fa2_simd_hd128 {
        return AttentionPrefillCandidateSelection {
            candidate: AttentionPrefillCandidate::Fa2HalfHd128,
            stability: KernelStabilityTier::Experimental,
        };
    }

    if use_fa2_simd_hd128 {
        AttentionPrefillCandidateSelection {
            candidate: AttentionPrefillCandidate::Fa2SimdHd128,
            stability: KernelStabilityTier::Experimental,
        }
    } else if use_fa2_simd_hd64 {
        AttentionPrefillCandidateSelection {
            candidate: AttentionPrefillCandidate::Fa2SimdHd64,
            stability: KernelStabilityTier::Experimental,
        }
    } else if use_fa2_hd128 {
        AttentionPrefillCandidateSelection {
            candidate: AttentionPrefillCandidate::Fa2Hd128,
            stability: KernelStabilityTier::ProfilePreferred,
        }
    } else if use_ax_bc64 {
        AttentionPrefillCandidateSelection {
            candidate: AttentionPrefillCandidate::AxBc64,
            stability: KernelStabilityTier::Experimental,
        }
    } else if use_ax_smem_f16 {
        AttentionPrefillCandidateSelection {
            candidate: AttentionPrefillCandidate::AxSmemF16,
            stability: KernelStabilityTier::Experimental,
        }
    } else if use_ax_smem {
        AttentionPrefillCandidateSelection {
            candidate: AttentionPrefillCandidate::AxSmem,
            stability: KernelStabilityTier::Experimental,
        }
    } else if use_ax_hd128 {
        AttentionPrefillCandidateSelection {
            candidate: AttentionPrefillCandidate::AxHd128,
            stability: KernelStabilityTier::ProfilePreferred,
        }
    } else if use_v2 {
        AttentionPrefillCandidateSelection {
            candidate: if use_v2_hd128 {
                AttentionPrefillCandidate::PrefillV2Hd128
            } else {
                AttentionPrefillCandidate::PrefillV2
            },
            stability: KernelStabilityTier::Stable,
        }
    } else if use_hd256 {
        AttentionPrefillCandidateSelection {
            candidate: AttentionPrefillCandidate::PrefillHd256,
            stability: KernelStabilityTier::Stable,
        }
    } else {
        AttentionPrefillCandidateSelection {
            candidate: AttentionPrefillCandidate::Prefill,
            stability: KernelStabilityTier::Stable,
        }
    }
}

fn attention_prefill_cached_candidate_selection(
    kv_f16: bool,
    n_tokens: u32,
    head_dim: u32,
    base_seq_len: u32,
    sliding_window: u32,
    config: AttentionDispatchConfig,
) -> AttentionPrefillCandidateSelection {
    // SIMD cached kernels are route variants, not unconditional overrides.
    if kv_f16 && attention_prefill_fa2_simd_hd128_enabled() {
        if attention_prefill_fa2_hd64_or_hd128_should_use_with_config(n_tokens, head_dim, config) {
            let candidate = match head_dim {
                128 => Some(AttentionPrefillCandidate::CacheFa2SimdHd128),
                64 => Some(AttentionPrefillCandidate::CacheFa2SimdHd64),
                _ => None,
            };
            if let Some(candidate) = candidate {
                return AttentionPrefillCandidateSelection {
                    candidate,
                    stability: KernelStabilityTier::ProfilePreferred,
                };
            }
        }
        if head_dim == 256
            && attention_prefill_fa2_cached_should_use_with_config(
                kv_f16,
                n_tokens,
                head_dim,
                base_seq_len,
                sliding_window,
                config,
            )
        {
            return AttentionPrefillCandidateSelection {
                candidate: AttentionPrefillCandidate::CacheFa2SimdHd256,
                stability: KernelStabilityTier::ProfilePreferred,
            };
        }
    }
    // Legacy FA2 cached (HD=256 only, different architecture)
    if attention_prefill_fa2_cached_should_use_with_config(
        kv_f16,
        n_tokens,
        head_dim,
        base_seq_len,
        sliding_window,
        config,
    ) {
        AttentionPrefillCandidateSelection {
            candidate: AttentionPrefillCandidate::CacheFa2Hd256,
            stability: KernelStabilityTier::ProfilePreferred,
        }
    } else {
        AttentionPrefillCandidateSelection {
            candidate: AttentionPrefillCandidate::Cache,
            stability: KernelStabilityTier::Stable,
        }
    }
}

fn resolve_attention_routing_profile(name: &str) -> Option<AttentionRoutingProfile> {
    match name.trim().to_ascii_lowercase().as_str() {
        "" | "default" => Some(ATTN_PROFILE_DEFAULT),
        "decode-balanced" | "balanced" => Some(ATTN_PROFILE_DECODE_BALANCED),
        "decode-long-context" | "long-context" | "long" => Some(ATTN_PROFILE_DECODE_LONG_CONTEXT),
        _ => None,
    }
}

fn active_attention_routing_profile_override() -> Option<AttentionRoutingProfile> {
    std::env::var("AX_METAL_ATTN_PROFILE")
        .ok()
        .as_deref()
        .and_then(resolve_attention_routing_profile)
}

fn attention_prefill_v2_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_bool_env_with_default("AX_METAL_PREFILL_ATTN_V2", true))
}

fn attention_prefill_hd256_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    // Default ON — constexpr HD=256 removes dynamic head_dim loops in the prefill kernel.
    // Set AX_METAL_PREFILL_HD256=0 to disable.
    *ENABLED.get_or_init(|| parse_bool_env_with_default("AX_METAL_PREFILL_HD256", true))
}

fn attention_prefill_hd128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    // Experimental: default OFF until sustained win is confirmed.
    // Set AX_METAL_PREFILL_HD128=1 to enable.
    *ENABLED.get_or_init(|| parse_bool_env_with_default("AX_METAL_PREFILL_HD128", false))
}

fn attention_prefill_ax_hd128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    // AX prefill tiling for HD=128 (BR=8, BC=32).
    // Default ON; set AX_METAL_PREFILL_AX_HD128=0 to disable.
    *ENABLED.get_or_init(|| parse_bool_env_with_default("AX_METAL_PREFILL_AX_HD128", true))
}

fn attention_prefill_ax_smem_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    // Experimental: default OFF until sustained win is confirmed.
    // Set AX_METAL_PREFILL_AX_SMEM=1 to enable.
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_PREFILL_AX_SMEM") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => false,
    })
}

fn attention_prefill_ax_smem_f16_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    // Experimental f16 shared-memory variant.
    // Set AX_METAL_PREFILL_AX_SMEM_F16=1 to enable.
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_PREFILL_AX_SMEM_F16") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => false,
    })
}

fn attention_prefill_fa2_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_bool_env_with_default("AX_METAL_PREFILL_FA2", false))
}

fn attention_prefill_fa2_hd128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_bool_env_with_default("AX_METAL_PREFILL_FA2_HD128", false))
}

fn parse_kernel_mode(var: &'static str, default_mode: KernelMode) -> KernelMode {
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

fn parse_positive_u32_env(var: &'static str, default_value: u32) -> u32 {
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

fn parse_bool_env_flag(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" => Some(true),
        "0" | "false" | "off" => Some(false),
        _ => None,
    }
}

fn parse_bool_env_with_default(var: &'static str, default_value: bool) -> bool {
    std::env::var(var)
        .ok()
        .and_then(|value| parse_bool_env_flag(&value))
        .unwrap_or(default_value)
}

fn profile_kernel_mode(mode: ProfileKernelMode) -> KernelMode {
    match mode {
        ProfileKernelMode::Off => KernelMode::Off,
        ProfileKernelMode::On => KernelMode::On,
        ProfileKernelMode::Auto => KernelMode::Auto,
    }
}

fn attention_decode_splitk_supported(kv_f16: bool, head_dim: u32) -> bool {
    kv_f16 && matches!(head_dim, 128 | 256)
}

fn attention_decode_sdpa_enabled_with_config(
    config: AttentionDispatchConfig,
    attend_len: u32,
) -> bool {
    config.effective_decode_sdpa_default(attend_len)
}

fn attention_prefill_fa2_mode_with_config(config: AttentionDispatchConfig) -> KernelMode {
    config.prefill_fa2_mode
}

fn attention_prefill_fa2_hd128_mode_with_config(config: AttentionDispatchConfig) -> KernelMode {
    config.prefill_fa2_hd128_mode
}

fn attention_prefill_ax_bc64_mode_with_config(config: AttentionDispatchConfig) -> KernelMode {
    config.prefill_ax_bc64_mode
}

fn attention_prefill_ax_bc64_min_tokens_with_config(config: AttentionDispatchConfig) -> u32 {
    config.prefill_ax_bc64_min_tokens
}

fn attention_prefill_ax_bc64_should_use_with_config(
    n_tokens: u32,
    config: AttentionDispatchConfig,
) -> bool {
    match attention_prefill_ax_bc64_mode_with_config(config) {
        KernelMode::Off => false,
        KernelMode::On => true,
        KernelMode::Auto => n_tokens >= attention_prefill_ax_bc64_min_tokens_with_config(config),
    }
}

fn attention_decode_splitk_mode_with_config(config: AttentionDispatchConfig) -> KernelMode {
    config.decode_splitk_mode
}

pub fn attention_decode_splitk_chunk_size_with_config(config: AttentionDispatchConfig) -> u32 {
    config.decode_splitk_chunk_size
}

fn attention_decode_splitk_auto_min_tokens_with_config(
    config: AttentionDispatchConfig,
    attend_len: u32,
) -> u32 {
    config.effective_decode_splitk_auto_min_tokens(attend_len)
}

fn attention_decode_splitk_should_use_with_config(
    kv_f16: bool,
    head_dim: u32,
    attend_len: u32,
    config: AttentionDispatchConfig,
) -> bool {
    if !attention_decode_splitk_supported(kv_f16, head_dim) {
        return false;
    }
    match attention_decode_splitk_mode_with_config(config) {
        KernelMode::Off => false,
        KernelMode::On => true,
        KernelMode::Auto => {
            matches!(head_dim, 128 | 256)
                && attend_len
                    >= attention_decode_splitk_auto_min_tokens_with_config(config, attend_len)
        }
    }
}

fn attention_prefill_fa2_cached_should_use_with_config(
    kv_f16: bool,
    n_tokens: u32,
    head_dim: u32,
    base_seq_len: u32,
    sliding_window: u32,
    config: AttentionDispatchConfig,
) -> bool {
    if !(kv_f16 && head_dim == 256) {
        return false;
    }
    match attention_prefill_fa2_mode_with_config(config) {
        KernelMode::Off => false,
        KernelMode::On => true,
        KernelMode::Auto => {
            // Conservative auto-gate for current kernels. Designed as a benchmark gate:
            // only route FA2 where sequence depth/width is high enough to amortize setup.
            n_tokens
                >= config.effective_prefill_fa2_auto_min_tokens_cached(
                    n_tokens,
                    base_seq_len,
                    sliding_window,
                )
                && base_seq_len
                    >= config.effective_prefill_fa2_auto_min_base_seq_cached(
                        n_tokens,
                        base_seq_len,
                        sliding_window,
                    )
                && sliding_window == 0
        }
    }
}

fn attention_prefill_fa2_hd64_or_hd128_should_use_with_config(
    n_tokens: u32,
    head_dim: u32,
    config: AttentionDispatchConfig,
) -> bool {
    if !matches!(head_dim, 64 | 128) {
        return false;
    }
    match attention_prefill_fa2_hd128_mode_with_config(config) {
        KernelMode::Off => false,
        KernelMode::On => true,
        KernelMode::Auto => {
            n_tokens >= config.effective_prefill_fa2_hd128_auto_min_tokens_local(n_tokens)
        }
    }
}

fn attention_prefill_fa2_simd_hd128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_PREFILL_FA2_SIMD") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => true, // default ON -- this is the fast path
    })
}

fn attention_prefill_fa2_half_hd128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("AX_METAL_PREFILL_FA2_HALF")
            .ok()
            .and_then(|v| parse_bool_env_flag(&v))
            // Default OFF: on Apple Silicon UMA, K/V staging through TG memory
            // adds overhead that negates the half×half MMA throughput advantage.
            // Direct device→register float loads (fa2_simd_hd128) are faster
            // because UMA has no VRAM→shared memory bandwidth bottleneck.
            .unwrap_or(false)
    })
}

fn attention_prefill_fa2v2_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("AX_METAL_PREFILL_FA2V2")
            .ok()
            .and_then(|v| parse_bool_env_flag(&v))
            // Default OFF: fa2v2 uses mixed half×float MMA which does not
            // benefit from Apple Silicon's half×half throughput advantage.
            // For HD=128, Fa2HalfHd128 is preferred. For HD=256, the
            // D-blocking approach is sound but mixed precision limits gain.
            .unwrap_or(false)
    })
}

fn attention_decode_hd128_n2_enabled_with_config(
    config: AttentionDispatchConfig,
    attend_len: u32,
) -> bool {
    config.effective_decode_hd128_n2_default(attend_len)
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
