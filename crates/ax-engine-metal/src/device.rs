//! MTLDevice wrapper — GPU handle and command queue for Metal compute.
//!
//! On Apple Silicon, the GPU shares unified memory with the CPU,
//! so buffer allocations are zero-copy.

use anyhow::{Context, anyhow, bail};
use objc2::rc::Retained;
use objc2::rc::autoreleasepool;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLCreateSystemDefaultDevice, MTLDevice, MTLDispatchType,
    MTLGPUFamily,
};
use std::sync::Arc;

use crate::MetalBuffer;
use crate::residency::ResidencyManager;
use crate::{
    PerfCounterState, PerfCounters, inc_command_buffer_count, new_perf_counter_state,
    reset_perf_counter_state, snapshot_perf_counter_state, with_active_perf_counters,
};

/// A command buffer that has been fully encoded but not yet committed to the GPU.
///
/// Created by [`MetalDevice::encode_frame`]. Commit with [`MetalDevice::commit_frame`].
///
/// # Pipelined decode loop
///
/// The double-buffer decode pipeline works as follows:
///
/// ```text
/// embed(tok_0, buf_a) → encode_frame(buf_a) → commit_frame → InflightFrame
/// while running:  embed(tok_1, buf_b) → encode_frame(buf_b) → PendingFrame
/// wait_frame(inflight_0) → sample → embed(tok_1, buf_b already filled)
/// commit_frame(pending_1) → InflightFrame ...
/// ```
///
/// Because Metal command buffers capture the buffer *address* (not its contents)
/// at encode time, the embedding can be written to `buf_b` after `encode_frame`
/// but before `commit_frame`, and the GPU will see the correct data.
pub struct PendingFrame {
    cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

/// A command buffer that has been committed to the GPU (currently executing or done).
///
/// Created by [`MetalDevice::commit_frame`]. Block until complete with
/// [`MetalDevice::wait_frame`].
pub struct InflightFrame {
    cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

/// GPU device handle wrapping MTLDevice + command queue.
pub struct MetalDevice {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    perf_counters: Arc<PerfCounterState>,
    residency: Option<Arc<ResidencyManager>>,
}

/// Device capabilities.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// GPU name (e.g. "Apple M3 Max").
    pub name: String,
    /// Whether the device has unified memory (always true on Apple Silicon).
    pub unified_memory: bool,
    /// Recommended max working set size in bytes.
    pub max_working_set_bytes: u64,
    /// Maximum threads per threadgroup (width, height, depth).
    pub max_threads_per_threadgroup: (usize, usize, usize),
    /// GPU family support flags (runtime-detected).
    pub gpu_family: GpuFamilySupport,
}

/// Runtime GPU family feature detection.
#[derive(Debug, Clone, Copy)]
pub struct GpuFamilySupport {
    /// Apple7+ (A14/M1+): simdgroup matrix multiply, simdgroup reduction.
    pub apple7: bool,
    /// Apple9 (M3+): bfloat16 native support, mesh shaders.
    pub apple9: bool,
    /// Metal 3 (M3+): bfloat16, ray tracing, mesh shaders.
    pub metal3: bool,
    /// Metal 4 (M5+): tensor API (mpp::tensor_ops).
    pub metal4: bool,
}

impl MetalDevice {
    /// Create the default Metal device (system GPU).
    pub fn new() -> anyhow::Result<Self> {
        let (device, queue, info) = autoreleasepool(|_| -> anyhow::Result<_> {
            let device = MTLCreateSystemDefaultDevice()
                .ok_or_else(|| anyhow!(metal_device_unavailable_message()))?;

            let queue = device
                .newCommandQueue()
                .context("Failed to create Metal command queue")?;

            let info = Self::read_device_info(&device);
            Ok((device, queue, info))
        })?;
        tracing::info!(
            gpu = %info.name,
            unified_memory = info.unified_memory,
            working_set_mb = info.max_working_set_bytes / 1024 / 1024,
            apple7 = info.gpu_family.apple7,
            apple9 = info.gpu_family.apple9,
            metal3 = info.gpu_family.metal3,
            metal4 = info.gpu_family.metal4,
            "Metal device initialized",
        );

        let residency = if crate::residency_enabled() {
            ResidencyManager::try_new(&device, &queue).map(Arc::new)
        } else {
            None
        };

        Ok(Self {
            device,
            queue,
            perf_counters: new_perf_counter_state(),
            residency,
        })
    }

    fn read_device_info(device: &ProtocolObject<dyn MTLDevice>) -> DeviceInfo {
        let mtl_size = device.maxThreadsPerThreadgroup();
        let gpu_family = GpuFamilySupport {
            apple7: device.supportsFamily(MTLGPUFamily::Apple7),
            apple9: device.supportsFamily(MTLGPUFamily::Apple9),
            metal3: device.supportsFamily(MTLGPUFamily::Metal3),
            metal4: device.supportsFamily(MTLGPUFamily(5002)),
        };
        DeviceInfo {
            name: device.name().to_string(),
            unified_memory: device.hasUnifiedMemory(),
            max_working_set_bytes: device.recommendedMaxWorkingSetSize(),
            max_threads_per_threadgroup: (mtl_size.width, mtl_size.height, mtl_size.depth),
            gpu_family,
        }
    }

    /// Create a new MetalDevice sharing the same underlying GPU but with its own command queue.
    ///
    /// This avoids re-enumerating the system device and is useful when multiple
    /// subsystems need independent command submission (e.g., MetalBackend + MetalOps).
    pub fn clone_sharing_device(&self) -> anyhow::Result<Self> {
        let queue = self
            .device
            .newCommandQueue()
            .context("Failed to create Metal command queue for shared device")?;
        // Attach the shared residency set to the new queue — each queue
        // needs its own attachment for command buffers to benefit.
        if let Some(ref rm) = self.residency {
            rm.attach_to_queue(&queue);
        }
        Ok(Self {
            device: self.device.clone(),
            queue,
            perf_counters: self.perf_counters.clone(),
            residency: self.residency.clone(),
        })
    }

    /// Reset backend-local performance counters for this device family.
    pub fn reset_perf_counters(&self) {
        reset_perf_counter_state(&self.perf_counters);
    }

    /// Read backend-local performance counters for this device family.
    pub fn perf_counters(&self) -> PerfCounters {
        snapshot_perf_counter_state(&self.perf_counters)
    }

    /// Get device capabilities.
    pub fn info(&self) -> DeviceInfo {
        Self::read_device_info(&self.device)
    }

    /// Current GPU memory allocation in bytes.
    pub fn current_allocated_bytes(&self) -> u64 {
        self.device.currentAllocatedSize() as u64
    }

    /// Log memory pressure: current allocation vs recommended max.
    pub fn log_memory_pressure(&self) {
        let allocated = self.current_allocated_bytes();
        let recommended = self.device.recommendedMaxWorkingSetSize();
        let alloc_mb = allocated / 1024 / 1024;
        let rec_mb = recommended / 1024 / 1024;
        let pct = if recommended > 0 {
            (allocated as f64 / recommended as f64 * 100.0) as u32
        } else {
            0
        };
        if allocated > recommended {
            tracing::warn!(
                allocated_mb = alloc_mb,
                recommended_mb = rec_mb,
                utilization_pct = pct,
                "GPU memory pressure: allocated exceeds recommended working set",
            );
        } else {
            tracing::info!(
                allocated_mb = alloc_mb,
                recommended_mb = rec_mb,
                utilization_pct = pct,
                "GPU memory utilization",
            );
        }
    }

    /// Access the underlying MTLDevice protocol object.
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Access the command queue.
    pub fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }

    /// Create a new command buffer from the queue.
    ///
    /// Prefer `commandBufferWithUnretainedReferences` to avoid ObjC
    /// retain/release traffic on buffers that AX already owns explicitly.
    /// Fall back to the regular retained command buffer if Metal declines
    /// to create the unretained variant.
    pub fn command_buffer(&self) -> anyhow::Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        self.queue
            .commandBufferWithUnretainedReferences()
            .or_else(|| self.queue.commandBuffer())
            .context("Failed to create command buffer")
    }

    /// Execute a compute dispatch synchronously.
    ///
    /// Creates a command buffer, calls the closure to encode compute commands,
    /// commits, and waits for GPU completion.
    pub fn execute_sync<F>(&self, f: F) -> anyhow::Result<()>
    where
        F: FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>) -> anyhow::Result<()>,
    {
        with_active_perf_counters(&self.perf_counters, || {
            inc_command_buffer_count();
            let cmd_buf = self.command_buffer()?;
            let encoder = cmd_buf
                .computeCommandEncoder()
                .context("Failed to create compute command encoder")?;
            crate::reset_pipeline_cache();

            f(&encoder)?;

            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();

            if let Some(error) = cmd_buf.error() {
                bail!("Metal command buffer error: {:?}", error);
            }

            Ok(())
        })
    }

    /// Like [`execute_sync`] but creates the encoder with `MTLDispatchTypeConcurrent`.
    ///
    /// With a concurrent encoder, independent kernel dispatches (those whose
    /// memory ranges don't overlap) can execute in parallel on the GPU.
    /// Memory barriers (`memoryBarrierWithScope`) must be inserted explicitly
    /// between dependent dispatches — the GPU will NOT serialize them.
    ///
    /// This matches llama.cpp's graph_compute strategy: encode the entire
    /// forward pass into one command buffer with concurrent dispatch, using
    /// barriers only where data dependencies require serialization.
    pub fn execute_sync_concurrent<F>(&self, f: F) -> anyhow::Result<()>
    where
        F: FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>) -> anyhow::Result<()>,
    {
        with_active_perf_counters(&self.perf_counters, || {
            inc_command_buffer_count();
            let cmd_buf = self.command_buffer()?;
            let encoder = cmd_buf
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
                .context("Failed to create concurrent compute command encoder")?;
            crate::reset_pipeline_cache();

            f(&encoder)?;

            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();

            if let Some(error) = cmd_buf.error() {
                bail!("Metal command buffer error: {:?}", error);
            }

            Ok(())
        })
    }

    /// Encode and submit a serial command buffer without waiting for completion.
    ///
    /// Like [`execute_sync`] but returns an [`InflightFrame`] instead of blocking.
    /// The caller must later call [`wait_frame`] before reading any GPU-written
    /// buffers. Metal FIFO ordering guarantees this CB completes before any
    /// subsequently committed CB starts.
    pub fn execute_async<F>(&self, f: F) -> anyhow::Result<InflightFrame>
    where
        F: FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>) -> anyhow::Result<()>,
    {
        with_active_perf_counters(&self.perf_counters, || {
            inc_command_buffer_count();
            let cmd_buf = self.command_buffer()?;
            let encoder = cmd_buf
                .computeCommandEncoder()
                .context("Failed to create compute command encoder")?;
            crate::reset_pipeline_cache();

            f(&encoder)?;

            encoder.endEncoding();
            cmd_buf.commit();
            Ok(InflightFrame { cmd_buf })
        })
    }

    /// Encode and submit a concurrent command buffer without waiting for completion.
    ///
    /// Returns an [`InflightFrame`] that the caller can wait on via [`wait_frame`]
    /// when the results are actually needed. This matches llama.cpp's strategy
    /// of asynchronous GPU submission — the CPU can do useful work (e.g. KV
    /// finalization bookkeeping) while the GPU is still computing.
    pub fn execute_async_concurrent<F>(&self, f: F) -> anyhow::Result<InflightFrame>
    where
        F: FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>) -> anyhow::Result<()>,
    {
        with_active_perf_counters(&self.perf_counters, || {
            inc_command_buffer_count();
            let cmd_buf = self.command_buffer()?;
            let encoder = cmd_buf
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
                .context("Failed to create concurrent compute command encoder")?;
            crate::reset_pipeline_cache();

            f(&encoder)?;

            encoder.endEncoding();
            cmd_buf.commit();
            Ok(InflightFrame { cmd_buf })
        })
    }

    /// Encode compute commands into a new command buffer without committing.
    ///
    /// Returns a [`PendingFrame`] that can be committed later with [`commit_frame`].
    /// This separates the encoding phase from GPU submission, enabling the
    /// pipelined decode loop where CB N+1 is encoded while CB N executes.
    ///
    /// The closure receives an encoder and must return `Ok(())` to indicate
    /// successful encoding. On error the pending frame is discarded.
    pub fn encode_frame<F>(&self, f: F) -> anyhow::Result<PendingFrame>
    where
        F: FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>) -> anyhow::Result<()>,
    {
        with_active_perf_counters(&self.perf_counters, || {
            inc_command_buffer_count();
            let cmd_buf = self.command_buffer()?;
            let encoder = cmd_buf
                .computeCommandEncoder()
                .context("Failed to create compute command encoder")?;
            crate::reset_pipeline_cache();

            f(&encoder)?;

            encoder.endEncoding();
            Ok(PendingFrame { cmd_buf })
        })
    }

    /// Like [`encode_frame`] but creates the encoder with `MTLDispatchTypeConcurrent`.
    ///
    /// Independent dispatches within the command buffer can execute in parallel
    /// on the GPU.  The caller must insert explicit barriers (via
    /// `barrier_buffers`) between dependent dispatches.
    pub fn encode_frame_concurrent<F>(&self, f: F) -> anyhow::Result<PendingFrame>
    where
        F: FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>) -> anyhow::Result<()>,
    {
        with_active_perf_counters(&self.perf_counters, || {
            inc_command_buffer_count();
            let cmd_buf = self.command_buffer()?;
            let encoder = cmd_buf
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
                .context("Failed to create concurrent compute command encoder")?;
            crate::reset_pipeline_cache();

            f(&encoder)?;

            encoder.endEncoding();
            Ok(PendingFrame { cmd_buf })
        })
    }

    /// Commit a [`PendingFrame`] to the GPU (non-blocking).
    ///
    /// Returns an [`InflightFrame`] that can be waited on with [`wait_frame`].
    /// The caller must NOT modify any buffers that the CB reads after this call.
    pub fn commit_frame(&self, frame: PendingFrame) -> InflightFrame {
        frame.cmd_buf.commit();
        InflightFrame {
            cmd_buf: frame.cmd_buf,
        }
    }

    /// Block until an [`InflightFrame`] completes on the GPU.
    ///
    /// Returns `Ok(())` on success, or an error if the GPU reported a failure.
    pub fn wait_frame(&self, frame: InflightFrame) -> anyhow::Result<()> {
        // Only wait if the command buffer hasn't already completed.
        let status = frame.cmd_buf.status();
        if status != MTLCommandBufferStatus::Completed && status != MTLCommandBufferStatus::Error {
            frame.cmd_buf.waitUntilCompleted();
        }

        if let Some(error) = frame.cmd_buf.error() {
            bail!("Metal command buffer error: {:?}", error);
        }

        Ok(())
    }

    // --- Residency management ---------------------------------------------------

    /// Register a buffer in the residency set (no-op if residency is disabled).
    pub fn register_resident_buffer(&self, buffer: &MetalBuffer) {
        if let Some(ref rm) = self.residency {
            rm.add_buffer(buffer);
        }
    }

    /// Commit the residency set, making all registered buffers resident.
    ///
    /// Call once after all weight buffers have been cached (typically after the
    /// model's first forward pass or after `build_cached_model_keys`).
    pub fn commit_residency(&self) {
        if let Some(ref rm) = self.residency {
            rm.commit();
        }
    }

    /// Whether this device has an active residency manager.
    pub fn has_residency(&self) -> bool {
        self.residency.is_some()
    }
}

fn metal_device_unavailable_message() -> String {
    let mut msg = String::from(
        "MTLCreateSystemDefaultDevice returned nil. Metal is unavailable to this process. \
         On supported Apple Silicon hardware this usually indicates a process/runtime issue, \
         not missing M3+ hardware.",
    );

    if std::env::var_os("CODEX_SANDBOX").is_some() {
        msg.push_str(
            " This process is running inside a Codex sandbox; if Metal works outside the sandbox, \
             rerun the built binary without sandbox isolation.",
        );
    }

    msg
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EnvVarRestore {
        key: String,
        previous: Option<std::ffi::OsString>,
    }

    impl Drop for EnvVarRestore {
        fn drop(&mut self) {
            match &self.previous {
                Some(prev) => unsafe {
                    std::env::set_var(&self.key, prev);
                },
                None => unsafe {
                    std::env::remove_var(&self.key);
                },
            }
        }
    }

    #[test]
    fn test_device_init() {
        let device = MetalDevice::new().expect("Metal device should init on Apple Silicon");
        let info = device.info();
        assert!(!info.name.is_empty(), "GPU should have a name");
    }

    #[test]
    fn test_device_unified_memory() {
        let device = MetalDevice::new().unwrap();
        let info = device.info();
        assert!(
            info.unified_memory,
            "Apple Silicon should have unified memory"
        );
    }

    #[test]
    fn test_device_working_set() {
        let device = MetalDevice::new().unwrap();
        let info = device.info();
        assert!(
            info.max_working_set_bytes > 0,
            "Working set should be nonzero"
        );
    }

    #[test]
    fn test_device_max_threads() {
        let device = MetalDevice::new().unwrap();
        let info = device.info();
        assert!(
            info.max_threads_per_threadgroup.0 >= 256,
            "Apple Silicon should support at least 256 threads per threadgroup"
        );
    }

    #[test]
    fn test_command_buffer_creation() {
        let device = MetalDevice::new().unwrap();
        let cmd_buf = device
            .command_buffer()
            .expect("Should create command buffer");
        drop(cmd_buf);
    }

    #[test]
    fn test_unretained_command_buffer_creation() {
        let device = MetalDevice::new().unwrap();
        let cmd_buf = device
            .queue()
            .commandBufferWithUnretainedReferences()
            .expect("Should create unretained command buffer");
        drop(cmd_buf);
    }

    #[test]
    fn test_clone_sharing_device_shares_local_perf_counters() {
        let device = MetalDevice::new().unwrap();
        let shared = device.clone_sharing_device().unwrap();

        device.reset_perf_counters();

        device.execute_sync(|_| Ok(())).unwrap();
        shared.execute_sync(|_| Ok(())).unwrap();

        let base = device.perf_counters();
        let clone = shared.perf_counters();
        assert_eq!(base.command_buffers, 2);
        assert_eq!(clone.command_buffers, 2);
        assert_eq!(base.buffer_barriers, 0);
        assert_eq!(clone.buffer_barriers, 0);
    }

    #[test]
    fn test_execute_sync_double_values() {
        use crate::buffer::MetalBuffer;
        use crate::dispatch::DispatchDims;
        use crate::pipeline::ComputePipeline;
        use objc2_metal::MTLComputeCommandEncoder;

        let gpu = MetalDevice::new().unwrap();
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let buf = MetalBuffer::from_slice(gpu.device(), &input).unwrap();

        let shader_src = r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void double_values(
                device float* data [[buffer(0)]],
                uint idx [[thread_position_in_grid]]
            ) {
                data[idx] = data[idx] * 2.0;
            }
        "#;
        let pipeline =
            ComputePipeline::from_source(gpu.device(), shader_src, "double_values").unwrap();

        let n = input.len();
        let dims = DispatchDims::d1(n, pipeline.thread_execution_width());

        gpu.execute_sync(|encoder| {
            encoder.setComputePipelineState(pipeline.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(buf.mtl_buffer()), 0, 0);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
            Ok(())
        })
        .unwrap();

        let result = unsafe { buf.as_slice::<f32>() };
        assert_eq!(result, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    }

    #[test]
    fn test_execute_sync_vector_add() {
        use crate::buffer::MetalBuffer;
        use crate::dispatch::DispatchDims;
        use crate::pipeline::ComputePipeline;
        use objc2_metal::MTLComputeCommandEncoder;

        let gpu = MetalDevice::new().unwrap();
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let buf_a = MetalBuffer::from_slice(gpu.device(), &a).unwrap();
        let buf_b = MetalBuffer::from_slice(gpu.device(), &b).unwrap();
        let buf_c = MetalBuffer::new(gpu.device(), 4 * std::mem::size_of::<f32>()).unwrap();

        let shader_src = r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void vec_add(
                device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* c       [[buffer(2)]],
                uint idx [[thread_position_in_grid]]
            ) {
                c[idx] = a[idx] + b[idx];
            }
        "#;
        let pipeline = ComputePipeline::from_source(gpu.device(), shader_src, "vec_add").unwrap();

        let n = a.len();
        let dims = DispatchDims::d1(n, pipeline.thread_execution_width());

        gpu.execute_sync(|encoder| {
            encoder.setComputePipelineState(pipeline.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(buf_a.mtl_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(buf_b.mtl_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(buf_c.mtl_buffer()), 0, 2);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                dims.threadgroups,
                dims.threads_per_threadgroup,
            );
            Ok(())
        })
        .unwrap();

        let result = unsafe { buf_c.as_slice::<f32>() };
        assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_metal_device_unavailable_message_mentions_sandbox_when_present() {
        let key = "CODEX_SANDBOX";
        let _restore = EnvVarRestore {
            key: key.to_string(),
            previous: std::env::var_os(key),
        };
        // SAFETY: Test-only process-local environment mutation.
        unsafe { std::env::set_var(key, "seatbelt") };
        let msg = metal_device_unavailable_message();
        assert!(msg.contains("Codex sandbox"));
    }
}
