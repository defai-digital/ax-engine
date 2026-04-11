//! Metal residency set management for weight buffers.
//!
//! On macOS 15+, `MTLResidencySet` keeps GPU buffers wired in physical memory
//! so the OS does not page them under memory pressure.  This is critical for
//! large models (70B+) whose working set approaches the GPU limit.

use std::sync::Mutex;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLAllocation, MTLBuffer, MTLCommandQueue, MTLDevice, MTLResidencySet,
    MTLResidencySetDescriptor, MTLResource,
};

use crate::MetalBuffer;

/// Manages a single `MTLResidencySet` for model weight buffers.
///
/// Buffers are added incrementally via [`add_buffer`] during model loading,
/// then committed in bulk via [`commit`].  The set is attached to the
/// command queue so all command buffers automatically benefit.
pub struct ResidencyManager {
    inner: Mutex<ResidencyInner>,
}

struct ResidencyInner {
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    committed: bool,
}

impl ResidencyManager {
    /// Try to create a residency manager for the given device.
    ///
    /// Returns `None` if `MTLResidencySet` is not available (pre-macOS 15)
    /// or if creation fails.
    pub fn try_new(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Option<Self> {
        let desc = MTLResidencySetDescriptor::new();
        let set = device.newResidencySetWithDescriptor_error(&desc).ok()?;
        // Attach to queue so all command buffers use this residency set.
        queue.addResidencySet(&set);
        tracing::info!("Metal residency set created and attached to command queue");
        Some(Self {
            inner: Mutex::new(ResidencyInner {
                set,
                committed: false,
            }),
        })
    }

    /// Register a buffer in the residency set.
    ///
    /// The buffer is not made resident until [`commit`] is called.
    pub fn add_buffer(&self, buffer: &MetalBuffer) {
        let inner = self.inner.lock().unwrap();
        let mtl_buf: &ProtocolObject<dyn MTLBuffer> = buffer.mtl_buffer();
        // MTLBuffer: MTLResource: MTLAllocation — upcast through the chain.
        let resource: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(mtl_buf);
        let allocation: &ProtocolObject<dyn MTLAllocation> = ProtocolObject::from_ref(resource);
        inner.set.addAllocation(allocation);
    }

    /// Commit all pending additions and request residency.
    ///
    /// Call this after all weight buffers have been registered (typically
    /// after the first forward pass when all weights are cached).
    pub fn commit(&self) {
        let mut inner = self.inner.lock().unwrap();
        if !inner.committed {
            inner.set.commit();
            inner.set.requestResidency();
            let count = inner.set.allocationCount();
            let size_mb = inner.set.allocatedSize() / (1024 * 1024);
            tracing::info!(
                allocations = count,
                size_mb = size_mb,
                "Metal residency set committed"
            );
            inner.committed = true;
        }
    }

    /// Number of allocations in the set (including uncommitted).
    pub fn allocation_count(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.set.allocationCount()
    }

    /// Attach the residency set to an additional command queue.
    ///
    /// Each queue that submits command buffers benefiting from residency
    /// must have the set attached individually — sharing the `Arc` alone
    /// does not propagate the attachment.
    pub fn attach_to_queue(&self, queue: &ProtocolObject<dyn MTLCommandQueue>) {
        let inner = self.inner.lock().unwrap();
        queue.addResidencySet(&inner.set);
    }
}

impl Drop for ResidencyManager {
    fn drop(&mut self) {
        let inner = match self.inner.get_mut() {
            Ok(guard) => guard,
            Err(e) => e.into_inner(),
        };
        if inner.committed {
            inner.set.endResidency();
        }
    }
}
