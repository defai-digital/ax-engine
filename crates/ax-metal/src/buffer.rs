//! MTLBuffer management — GPU-accessible memory.
//!
//! On UMA (Apple Silicon), buffers share physical memory with CPU.
//! Shared-mode buffers are zero-copy: the CPU pointer returned by
//! `contents()` is the same physical pages the GPU reads.

use std::ffi::c_void;
use std::ptr::NonNull;

use anyhow::Context;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

/// GPU-accessible memory buffer backed by MTLBuffer.
pub struct MetalBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl MetalBuffer {
    /// Allocate a new buffer of `size` bytes with shared storage.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, size: usize) -> anyhow::Result<Self> {
        let buffer = device
            .newBufferWithLength_options(size as _, MTLResourceOptions::StorageModeShared)
            .context("Failed to allocate Metal buffer")?;
        Ok(Self { buffer })
    }

    /// Create a buffer initialized from a byte slice.
    pub fn from_bytes(device: &ProtocolObject<dyn MTLDevice>, data: &[u8]) -> anyhow::Result<Self> {
        let ptr =
            NonNull::new(data.as_ptr() as *mut c_void).context("Null data pointer for buffer")?;
        let buffer = unsafe {
            device
                .newBufferWithBytes_length_options(
                    ptr,
                    data.len() as _,
                    MTLResourceOptions::StorageModeShared,
                )
                .context("Failed to create Metal buffer from bytes")?
        };
        Ok(Self { buffer })
    }

    /// Create a buffer initialized from a typed slice.
    pub fn from_slice<T: Copy>(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[T],
    ) -> anyhow::Result<Self> {
        let byte_len = std::mem::size_of_val(data);
        let ptr =
            NonNull::new(data.as_ptr() as *mut c_void).context("Null data pointer for buffer")?;
        let buffer = unsafe {
            device
                .newBufferWithBytes_length_options(
                    ptr,
                    byte_len as _,
                    MTLResourceOptions::StorageModeShared,
                )
                .context("Failed to create Metal buffer from slice")?
        };
        Ok(Self { buffer })
    }

    /// Create a buffer that aliases an existing mutable slice without copying.
    ///
    /// # Safety
    ///
    /// The caller must ensure the backing allocation outlives the returned
    /// buffer and is not reallocated while the GPU may access it.
    /// Per Apple docs the pointer should be page-aligned, but Apple Silicon UMA
    /// with `StorageModeShared` accepts non-page-aligned pointers in practice.
    pub unsafe fn from_mut_slice_no_copy<T: Copy>(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &mut [T],
    ) -> anyhow::Result<Self> {
        let byte_len = std::mem::size_of_val(data);
        let ptr = NonNull::new(data.as_mut_ptr() as *mut c_void)
            .context("Null data pointer for buffer")?;
        let buffer = unsafe {
            device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr,
                    byte_len as _,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
                .context("Failed to create Metal no-copy buffer from mutable slice")?
        };
        Ok(Self { buffer })
    }

    /// CPU-accessible pointer to the buffer contents.
    pub fn contents(&self) -> NonNull<c_void> {
        self.buffer.contents()
    }

    /// Unique identifier for this buffer (the MTLBuffer pointer address).
    /// Used by [`SmartBarrier`] for overlap detection.
    pub fn ptr_id(&self) -> usize {
        // Use the MTLBuffer's GPU address as a stable identity.
        self.buffer.contents().as_ptr() as usize
    }

    /// Length of the buffer in bytes.
    pub fn len(&self) -> usize {
        self.buffer.length()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// View the buffer contents as a slice of `T`.
    ///
    /// # Safety
    ///
    /// The buffer must contain valid `T` data and be properly aligned.
    pub unsafe fn as_slice<T: Copy>(&self) -> &[T] {
        let ptr = self.contents().as_ptr() as *const T;
        let count = self.len() / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts(ptr, count) }
    }

    /// View the buffer contents as a mutable slice of `T`.
    ///
    /// # Safety
    ///
    /// The buffer must contain valid `T` data and be properly aligned.
    /// The caller must ensure exclusive access.
    pub unsafe fn as_mut_slice<T: Copy>(&mut self) -> &mut [T] {
        let ptr = self.contents().as_ptr() as *mut T;
        let count = self.len() / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts_mut(ptr, count) }
    }

    /// Access the underlying MTLBuffer protocol object.
    pub fn mtl_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalDevice;

    #[test]
    fn test_buffer_alloc() {
        let gpu = MetalDevice::new().unwrap();
        let buf = MetalBuffer::new(gpu.device(), 4096).unwrap();
        assert_eq!(buf.len(), 4096);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_buffer_from_f32_slice() {
        let gpu = MetalDevice::new().unwrap();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let buf = MetalBuffer::from_slice(gpu.device(), &data).unwrap();
        assert_eq!(buf.len(), 4 * std::mem::size_of::<f32>());

        let readback = unsafe { buf.as_slice::<f32>() };
        assert_eq!(readback, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_buffer_write_and_read() {
        let gpu = MetalDevice::new().unwrap();
        let mut buf = MetalBuffer::new(gpu.device(), 16).unwrap();
        unsafe {
            let slice = buf.as_mut_slice::<f32>();
            slice[0] = 42.0;
            slice[1] = -1.5;
            slice[2] = 0.0;
            slice[3] = 100.0;
        }
        let readback = unsafe { buf.as_slice::<f32>() };
        assert_eq!(readback, &[42.0, -1.5, 0.0, 100.0]);
    }

    #[test]
    fn test_buffer_from_bytes() {
        let gpu = MetalDevice::new().unwrap();
        let data: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let buf = MetalBuffer::from_bytes(gpu.device(), &data).unwrap();
        assert_eq!(buf.len(), 4);
        let readback = unsafe { buf.as_slice::<u8>() };
        assert_eq!(readback, &[0xDE, 0xAD, 0xBE, 0xEF]);
    }
}
