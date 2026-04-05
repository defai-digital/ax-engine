use super::*;

pub struct MatmulKernels {
    /// Fallback 16×16 tiled matmul (kept for compatibility).
    #[allow(dead_code)]
    tiled_matmul: ComputePipeline,
    /// High-performance simdgroup_matrix 32×32 tiled matmul (float tiles).
    simdgroup_matmul: ComputePipeline,
    /// Half-tile simdgroup matmul: half A/B tiles, float accumulator.
    simdgroup_matmul_half: ComputePipeline,
    /// BFloat16-tile simdgroup matmul (Metal 3+/M3+ only).
    simdgroup_matmul_bf16: Option<ComputePipeline>,
    /// Metal Tensor API matmul (Metal 4+/M5+ only).
    /// Uses mpp::tensor_ops::matmul2d for hardware-accelerated cooperative matmul.
    simdgroup_matmul_tensor: Option<ComputePipeline>,
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
        let simdgroup_matmul_half = ComputePipeline::from_source(
            device.device(),
            MATMUL_SHADER_SRC,
            "matmul_simdgroup_half_f32",
        )
        .context("Failed to compile matmul_simdgroup_half_f32 kernel")?;
        // BFloat16 kernel only compiles on Metal 3+ (M3+) devices.
        let has_metal3 = device.info().gpu_family.metal3;
        let simdgroup_matmul_bf16 = if has_metal3 {
            ComputePipeline::from_source(
                device.device(),
                MATMUL_SHADER_SRC,
                "matmul_simdgroup_bf16_f32",
            )
            .ok()
        } else {
            None
        };
        // Metal Tensor API kernel (Metal 4+/M5+ only). Requires both:
        // 1. SDK support (metal_tensor_api cfg set by build.rs)
        // 2. Runtime GPU support (Metal4 family)
        let has_metal4 = device.info().gpu_family.metal4;
        let simdgroup_matmul_tensor = if cfg!(metal_tensor_api) && has_metal4 {
            // The kernel is embedded in the precompiled metallib by build.rs.
            // Try to load it — will fail gracefully if not linked.
            ComputePipeline::from_source(
                device.device(),
                MATMUL_SHADER_SRC,
                "matmul_simdgroup_tensor_f32",
            )
            .ok()
        } else {
            None
        };
        let matvec = ComputePipeline::from_source(device.device(), MATMUL_SHADER_SRC, "matvec_f32")
            .context("Failed to compile matvec_f32 kernel")?;

        tracing::info!(
            matmul_max_threads = tiled_matmul.max_threads_per_threadgroup(),
            simdgroup_max_threads = simdgroup_matmul.max_threads_per_threadgroup(),
            simdgroup_half_max_threads = simdgroup_matmul_half.max_threads_per_threadgroup(),
            bf16_available = simdgroup_matmul_bf16.is_some(),
            tensor_available = simdgroup_matmul_tensor.is_some(),
            matvec_max_threads = matvec.max_threads_per_threadgroup(),
            "Matmul Metal kernels compiled (tiled + simdgroup + matvec)",
        );

        Ok(Self {
            tiled_matmul,
            simdgroup_matmul,
            simdgroup_matmul_half,
            simdgroup_matmul_bf16,
            simdgroup_matmul_tensor,
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

        // Prefer tensor API (Metal 4+/M5+) > bf16 (Metal 3+) > half.
        if let Some(tensor_pipeline) = &self.simdgroup_matmul_tensor {
            // Tensor API: NR0=64, NR1=32 tiles with dynamic TG memory.
            encoder.setComputePipelineState(tensor_pipeline.state());
            bind_buffers(encoder, a, b, c);
            bind_u32(encoder, 3, m);
            bind_u32(encoder, 4, n);
            bind_u32(encoder, 5, k);
            // 6 KB TG memory: sa(4KB) + sb(2KB). Boundary path reuses as float (8KB).
            let smem = if (m as usize).is_multiple_of(64) && (n as usize).is_multiple_of(32) {
                6144usize
            } else {
                8192usize
            };
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
                    width: SG_TG,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            // Prefer bf16 tiles on Metal 3+ (M3+) for better dynamic range,
            // fall back to half tiles otherwise. Both use float accumulators.
            let pipeline = self
                .simdgroup_matmul_bf16
                .as_ref()
                .unwrap_or(&self.simdgroup_matmul_half);
            encoder.setComputePipelineState(pipeline.state());
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
    }

    /// Encode an f32 matvec dispatch into an existing command encoder.
    ///
    /// y = A[M×K] × x[K], where A and x are f32, y is f32.
    pub fn encode_matvec(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
    ) {
        let dims = DispatchDims::d1(m as usize, 1);
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
