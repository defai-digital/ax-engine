//! AX Engine — Metal compute backend
//!
//! Provides GPU-accelerated matmul and dequantization for prefill.
//! Phase 2 implementation.

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
compile_error!("ax-metal only supports aarch64-apple-darwin (Apple Silicon M3+)");

pub mod buffer;
pub mod device;
pub mod dispatch;
pub mod pipeline;
pub mod profile;

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicUsize};
use std::sync::atomic::{AtomicU64, Ordering};

pub use buffer::MetalBuffer;
pub use device::{DeviceInfo, InflightFrame, MetalDevice, PendingFrame};
pub use dispatch::{
    AttentionKernels, DequantKernels, DispatchDims, ElementwiseKernels, MatmulKernels,
    SmartBarrier, barrier_buffers, batch_simd_enabled,
};
pub use pipeline::{ComputePipeline, FunctionConstant, FunctionConstantValue};
pub use profile::{KernelProfile, global_profile, init_global_profile};

/// Type alias for the Metal compute command encoder protocol object.
/// Re-exported so downstream crates can reference the encoder type
/// without depending on objc2/objc2_metal directly.
pub type MetalEncoder = objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>;

/// Lightweight Metal hot-path counters used for performance analysis.
#[derive(Debug, Clone, Copy, Default)]
pub struct PerfCounters {
    pub command_buffers: u64,
    pub buffer_barriers: u64,
}

static COMMAND_BUFFER_COUNT: AtomicU64 = AtomicU64::new(0);
static BUFFER_BARRIER_COUNT: AtomicU64 = AtomicU64::new(0);
static BATCH_F16IN_SMALL_N_THRESHOLD: AtomicU32 = AtomicU32::new(1);
static BATCH_F16IN_SMALL_M_MAX: AtomicU32 = AtomicU32::new(0);
static BATCH_F16IN_ROUTE_INIT: AtomicUsize = AtomicUsize::new(0);

/// Reset global Metal performance counters.
pub fn reset_perf_counters() {
    COMMAND_BUFFER_COUNT.store(0, Ordering::Relaxed);
    BUFFER_BARRIER_COUNT.store(0, Ordering::Relaxed);
}

/// Read a snapshot of global Metal performance counters.
pub fn perf_counters() -> PerfCounters {
    PerfCounters {
        command_buffers: COMMAND_BUFFER_COUNT.load(Ordering::Relaxed),
        buffer_barriers: BUFFER_BARRIER_COUNT.load(Ordering::Relaxed),
    }
}

fn init_batch_f16in_route_defaults() {
    if BATCH_F16IN_ROUTE_INIT
        .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return;
    }

    let profile = crate::profile::global_profile();
    let n_threshold = match std::env::var("AX_METAL_F16IN_SMALL_N") {
        Ok(v) => v.trim().parse::<u32>().unwrap_or(1),
        Err(_) => profile.batch_prefill.small_n_threshold,
    };
    let m_max = match std::env::var("AX_METAL_F16IN_SMALL_M_MAX") {
        Ok(v) => v.trim().parse::<u32>().unwrap_or(0),
        Err(_) => profile.batch_prefill.small_m_max,
    };
    BATCH_F16IN_SMALL_N_THRESHOLD.store(n_threshold, Ordering::Relaxed);
    BATCH_F16IN_SMALL_M_MAX.store(m_max, Ordering::Relaxed);
}

/// Get current f16-input batch routing thresholds (small-kernel route).
///
/// Precedence:
/// 1) `AX_METAL_F16IN_SMALL_N` / `AX_METAL_F16IN_SMALL_M_MAX`
/// 2) kernel profile `batch_prefill.small_*`
/// 3) built-in defaults
pub fn batch_f16in_route_config() -> (u32, u32) {
    init_batch_f16in_route_defaults();
    (
        BATCH_F16IN_SMALL_N_THRESHOLD.load(Ordering::Relaxed),
        BATCH_F16IN_SMALL_M_MAX.load(Ordering::Relaxed),
    )
}

/// Override f16-input batch routing thresholds at runtime.
///
/// - `small_n_threshold`: route to small kernel when `n < threshold`
/// - `small_m_max`: and `m <= small_m_max` (0 disables small route)
pub fn set_batch_f16in_route_config(small_n_threshold: u32, small_m_max: u32) {
    init_batch_f16in_route_defaults();
    BATCH_F16IN_SMALL_N_THRESHOLD.store(small_n_threshold, Ordering::Relaxed);
    BATCH_F16IN_SMALL_M_MAX.store(small_m_max, Ordering::Relaxed);
}

pub(crate) fn sync_batch_f16in_route_from_profile(small_n_threshold: u32, small_m_max: u32) {
    if std::env::var("AX_METAL_F16IN_SMALL_N").is_ok()
        || std::env::var("AX_METAL_F16IN_SMALL_M_MAX").is_ok()
    {
        return;
    }
    set_batch_f16in_route_config(small_n_threshold, small_m_max);
}

pub(crate) fn inc_command_buffer_count() {
    COMMAND_BUFFER_COUNT.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn inc_buffer_barrier_count() {
    BUFFER_BARRIER_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Whether explicit Metal buffer barriers are enabled.
///
/// Controlled by `AX_METAL_BARRIERS`:
/// - unset / `1` / `true` / `on`  -> enabled (default)
/// - `0` / `false` / `off`        -> disabled
pub(crate) fn barriers_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_BARRIERS") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        }
        Err(_) => true,
    })
}
