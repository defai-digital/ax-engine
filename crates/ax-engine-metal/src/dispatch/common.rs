//! Shared dispatch helpers reused across Metal kernel families.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2_metal::MTLBarrierScope;

use super::*;

use crate::{barriers_enabled, inc_buffer_barrier_count};

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

pub(super) fn bind_buffers(
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

#[allow(clippy::too_many_arguments)]
pub(super) fn bind_buffers7(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buf0: &MetalBuffer,
    buf1: &MetalBuffer,
    buf2: &MetalBuffer,
    buf3: &MetalBuffer,
    buf4: &MetalBuffer,
    buf5: &MetalBuffer,
    buf6: &MetalBuffer,
) {
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(buf0.mtl_buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(buf1.mtl_buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(buf2.mtl_buffer()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(buf3.mtl_buffer()), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(buf4.mtl_buffer()), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(buf5.mtl_buffer()), 0, 5);
        encoder.setBuffer_offset_atIndex(Some(buf6.mtl_buffer()), 0, 6);
    }
}

/// Bind a u32 scalar at the given buffer index.
pub(super) fn bind_u32(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: usize,
    value: u32,
) {
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::new_unchecked(&value as *const u32 as *mut c_void),
            std::mem::size_of::<u32>(),
            index,
        );
    }
}

/// Bind an f32 scalar at the given buffer index.
pub(super) fn bind_f32(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: usize,
    value: f32,
) {
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
    /// When false, `pre_dispatch` always inserts a barrier (like `barrier_buffers`).
    smart: bool,
}

impl<'a> SmartBarrier<'a> {
    /// Create a new tracker wrapping a concurrent encoder.
    ///
    /// When `smart` is true, barriers are only inserted when data hazards are
    /// detected (llama.cpp pattern). When false, every `pre_dispatch` inserts
    /// a barrier (equivalent to `barrier_buffers()`).
    pub fn new(encoder: &'a ProtocolObject<dyn MTLComputeCommandEncoder>) -> Self {
        Self {
            encoder,
            pending: Vec::with_capacity(32),
            smart: crate::smart_barriers_enabled(),
        }
    }

    /// Check for conflicts and insert barrier if needed, BEFORE dispatching.
    /// `reads`: buffers the next dispatch will read from.
    /// `writes`: buffers the next dispatch will write to.
    pub fn pre_dispatch(&mut self, reads: &[&MetalBuffer], writes: &[&MetalBuffer]) {
        if !barriers_enabled() {
            return;
        }
        let needs_barrier = if self.smart {
            self.has_conflict(reads, writes)
        } else {
            !self.pending.is_empty()
        };
        if needs_barrier {
            inc_buffer_barrier_count();
            self.encoder
                .memoryBarrierWithScope(MTLBarrierScope::Buffers);
            self.pending.clear();
        }
    }

    /// Register buffer ranges AFTER dispatching.
    pub fn post_dispatch(&mut self, reads: &[&MetalBuffer], writes: &[&MetalBuffer]) {
        if !barriers_enabled() {
            return;
        }
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

    /// Unconditional barrier + clear tracking, then register new reads/writes.
    ///
    /// Use this as a drop-in replacement for `barrier_buffers(encoder)` when
    /// the caller knows a barrier is required but still wants SmartBarrier to
    /// track subsequent dispatches.
    pub fn barrier_and_track(&mut self, reads: &[&MetalBuffer], writes: &[&MetalBuffer]) {
        self.flush();
        self.post_dispatch(reads, writes);
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
