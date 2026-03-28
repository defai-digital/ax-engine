//! AX Engine — Metal compute backend
//!
//! Provides GPU-accelerated matmul and dequantization for prefill.
//! Phase 2 implementation.

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
compile_error!("ax-engine-metal only supports aarch64-apple-darwin (Apple Silicon M3+)");

pub mod buffer;
pub mod device;
pub mod dispatch;
pub mod pipeline;
pub mod profile;

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::{cell::RefCell, thread_local};

pub use buffer::MetalBuffer;
pub use device::{DeviceInfo, InflightFrame, MetalDevice, PendingFrame};
pub use dispatch::{
    AttentionDecodeCandidate, AttentionDecodeCandidateSelection, AttentionDispatchConfig,
    AttentionKernels, AttentionPrefillCandidate, AttentionPrefillCandidateSelection,
    DequantDispatchConfig, DequantKernels, DispatchDims, ElementwiseKernels, GdnKernels,
    KernelMode, KernelStabilityTier, MatmulKernels, MatvecCandidate, MatvecCandidateSelection,
    SmartBarrier, barrier_buffers, batch_simd_enabled,
};
pub use pipeline::{ComputePipeline, FunctionConstant, FunctionConstantValue};
pub use profile::KernelProfile;

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

#[derive(Default)]
pub(crate) struct PerfCounterState {
    command_buffers: AtomicU64,
    buffer_barriers: AtomicU64,
}

static COMMAND_BUFFER_COUNT: AtomicU64 = AtomicU64::new(0);
static BUFFER_BARRIER_COUNT: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static ACTIVE_PERF_COUNTERS: RefCell<Vec<Arc<PerfCounterState>>> = const { RefCell::new(Vec::new()) };
}

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

pub(crate) fn inc_command_buffer_count() {
    COMMAND_BUFFER_COUNT.fetch_add(1, Ordering::Relaxed);
    ACTIVE_PERF_COUNTERS.with(|active| {
        if let Some(state) = active.borrow().last() {
            state.command_buffers.fetch_add(1, Ordering::Relaxed);
        }
    });
}

pub(crate) fn inc_buffer_barrier_count() {
    BUFFER_BARRIER_COUNT.fetch_add(1, Ordering::Relaxed);
    ACTIVE_PERF_COUNTERS.with(|active| {
        if let Some(state) = active.borrow().last() {
            state.buffer_barriers.fetch_add(1, Ordering::Relaxed);
        }
    });
}

pub(crate) fn new_perf_counter_state() -> Arc<PerfCounterState> {
    Arc::new(PerfCounterState::default())
}

pub(crate) fn reset_perf_counter_state(state: &Arc<PerfCounterState>) {
    state.command_buffers.store(0, Ordering::Relaxed);
    state.buffer_barriers.store(0, Ordering::Relaxed);
}

pub(crate) fn snapshot_perf_counter_state(state: &Arc<PerfCounterState>) -> PerfCounters {
    PerfCounters {
        command_buffers: state.command_buffers.load(Ordering::Relaxed),
        buffer_barriers: state.buffer_barriers.load(Ordering::Relaxed),
    }
}

pub(crate) fn with_active_perf_counters<R>(
    state: &Arc<PerfCounterState>,
    f: impl FnOnce() -> R,
) -> R {
    ACTIVE_PERF_COUNTERS.with(|active| active.borrow_mut().push(state.clone()));
    let result = f();
    ACTIVE_PERF_COUNTERS.with(|active| {
        active.borrow_mut().pop();
    });
    result
}

/// Whether explicit Metal buffer barriers are enabled.
///
/// Controlled by `AX_METAL_BARRIERS`:
/// - unset / `1` / `true` / `on`  -> enabled (default)
/// - `0` / `false` / `off`        -> disabled
pub(crate) fn barriers_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_bool_env_with_default("AX_METAL_BARRIERS", true))
}

/// Whether smart barrier tracking is enabled for concurrent prefill dispatch.
///
/// When enabled, `SmartBarrier` is used instead of `barrier_buffers()` to
/// track buffer read/write sets and only insert barriers when data hazards
/// exist. This allows the GPU to overlap independent dispatches.
///
/// Controlled by `AX_METAL_SMART_BARRIERS`:
/// - unset / `1` / `true` / `on`  -> enabled (default)
/// - `0` / `false` / `off`        -> disabled (falls back to barrier_buffers)
pub fn smart_barriers_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_bool_env_with_default("AX_METAL_SMART_BARRIERS", true))
}

fn parse_bool_env_flag(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" => Some(true),
        "0" | "false" | "off" => Some(false),
        _ => None,
    }
}

fn parse_bool_env_with_default(var: &'static str, default: bool) -> bool {
    std::env::var(var)
        .ok()
        .and_then(|value| parse_bool_env_flag(&value))
        .unwrap_or(default)
}
