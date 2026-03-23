//! Compute pipeline management — compiled Metal shaders.
//!
//! Compiles Metal source code at runtime and creates pipeline state
//! objects ready for dispatch.

use std::ffi::c_void;
use std::ptr::NonNull;

use anyhow::Context;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_foundation::NSUInteger;
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLDataType, MTLDevice, MTLFunctionConstantValues,
    MTLLanguageVersion, MTLLibrary,
};

/// Typed value for a Metal function constant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FunctionConstantValue {
    Bool(bool),
    U16(u16),
    U32(u32),
    F32(f32),
}

impl FunctionConstantValue {
    fn data_type(self) -> MTLDataType {
        match self {
            Self::Bool(_) => MTLDataType::Bool,
            Self::U16(_) => MTLDataType::UShort,
            Self::U32(_) => MTLDataType::UInt,
            Self::F32(_) => MTLDataType::Float,
        }
    }
}

/// One function-constant binding for Metal pipeline specialization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FunctionConstant {
    pub index: NSUInteger,
    pub value: FunctionConstantValue,
}

/// A compiled compute pipeline ready for dispatch.
pub struct ComputePipeline {
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl ComputePipeline {
    /// Compile a compute pipeline from Metal source code.
    ///
    /// `source` is Metal Shading Language source code.
    /// `function_name` is the kernel function to use as the entry point.
    pub fn from_source(
        device: &ProtocolObject<dyn MTLDevice>,
        source: &str,
        function_name: &str,
    ) -> anyhow::Result<Self> {
        let library = compile_library(device, source)?;
        Self::from_library(device, &library, function_name)
    }

    /// Compile a compute pipeline from Metal source code, specializing the entry
    /// function with Metal function constants.
    pub fn from_source_with_constants(
        device: &ProtocolObject<dyn MTLDevice>,
        source: &str,
        function_name: &str,
        constants: &[FunctionConstant],
    ) -> anyhow::Result<Self> {
        let library = compile_library(device, source)?;
        Self::from_library_with_constants(device, &library, function_name, constants)
    }

    /// Create a pipeline from a precompiled Metal library.
    pub fn from_library(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
        function_name: &str,
    ) -> anyhow::Result<Self> {
        let ns_fn_name = NSString::from_str(function_name);
        let function = library
            .newFunctionWithName(&ns_fn_name)
            .with_context(|| format!("Function '{function_name}' not found in Metal library"))?;
        Self::from_function(device, &function, function_name)
    }

    /// Create a specialized pipeline from a precompiled Metal library using
    /// Metal function constants.
    pub fn from_library_with_constants(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
        function_name: &str,
        constants: &[FunctionConstant],
    ) -> anyhow::Result<Self> {
        let ns_fn_name = NSString::from_str(function_name);
        let constant_values = build_function_constant_values(constants);
        let function = library
            .newFunctionWithName_constantValues_error(&ns_fn_name, &constant_values)
            .with_context(|| {
                format!("Failed to specialize Metal function '{function_name}' with constants")
            })?;
        Self::from_function(device, &function, function_name)
    }

    /// Maximum total threads per threadgroup for this pipeline.
    pub fn max_threads_per_threadgroup(&self) -> usize {
        self.state.maxTotalThreadsPerThreadgroup()
    }

    /// SIMD execution width (warp size) for this pipeline.
    pub fn thread_execution_width(&self) -> usize {
        self.state.threadExecutionWidth()
    }

    /// Access the underlying pipeline state object.
    pub fn state(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.state
    }
}

fn compile_library(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
) -> anyhow::Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let ns_source = NSString::from_str(source);
    let options = metal_compile_options();
    device
        .newLibraryWithSource_options_error(&ns_source, Some(&options))
        .map_err(|e| anyhow::anyhow!("Metal shader compilation failed: {:?}", e))
}

/// Create Metal compile options optimized for Apple Silicon M3+.
///
/// - Metal Shading Language 3.1: enables Apple GPU family features (simdgroup
///   matrix ops, extended threadgroup memory, improved occupancy hints).
/// - Fast math: ON (default, but explicit for clarity). Allows FMA contraction,
///   reassociation, and reciprocal approximations.
fn metal_compile_options() -> Retained<MTLCompileOptions> {
    let options = MTLCompileOptions::new();
    // Metal 3.1 is supported on M1+ (macOS 14+). This is safe because
    // ax-engine requires aarch64-apple-darwin (Apple Silicon only).
    #[allow(deprecated)] // setFastMathEnabled is deprecated in favor of setMathMode
    {
        options.setLanguageVersion(MTLLanguageVersion::Version3_1);
        options.setFastMathEnabled(true);
    }
    options
}

fn build_function_constant_values(
    constants: &[FunctionConstant],
) -> Retained<MTLFunctionConstantValues> {
    let values = MTLFunctionConstantValues::new();
    for constant in constants {
        set_constant_value(&values, *constant);
    }
    values
}

fn set_constant_value(values: &MTLFunctionConstantValues, constant: FunctionConstant) {
    match constant.value {
        FunctionConstantValue::Bool(v) => {
            let mut value = v;
            let ptr = NonNull::from(&mut value).cast::<c_void>();
            unsafe {
                values.setConstantValue_type_atIndex(
                    ptr,
                    constant.value.data_type(),
                    constant.index,
                )
            };
        }
        FunctionConstantValue::U16(v) => {
            let mut value = v;
            let ptr = NonNull::from(&mut value).cast::<c_void>();
            unsafe {
                values.setConstantValue_type_atIndex(
                    ptr,
                    constant.value.data_type(),
                    constant.index,
                )
            };
        }
        FunctionConstantValue::U32(v) => {
            let mut value = v;
            let ptr = NonNull::from(&mut value).cast::<c_void>();
            unsafe {
                values.setConstantValue_type_atIndex(
                    ptr,
                    constant.value.data_type(),
                    constant.index,
                )
            };
        }
        FunctionConstantValue::F32(v) => {
            let mut value = v;
            let ptr = NonNull::from(&mut value).cast::<c_void>();
            unsafe {
                values.setConstantValue_type_atIndex(
                    ptr,
                    constant.value.data_type(),
                    constant.index,
                )
            };
        }
    }
}

impl ComputePipeline {
    fn from_function(
        device: &ProtocolObject<dyn MTLDevice>,
        function: &ProtocolObject<dyn objc2_metal::MTLFunction>,
        function_name: &str,
    ) -> anyhow::Result<Self> {
        let state = device
            .newComputePipelineStateWithFunction_error(function)
            .map_err(|e| anyhow::anyhow!("Failed to create compute pipeline: {:?}", e))?;

        let max_threads = state.maxTotalThreadsPerThreadgroup();
        let exec_width = state.threadExecutionWidth();
        tracing::debug!(
            function = function_name,
            max_threads,
            exec_width,
            "Compiled Metal compute pipeline",
        );

        Ok(Self { state })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MetalBuffer, MetalDevice};
    use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

    const TRIVIAL_SHADER: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void noop(uint idx [[thread_position_in_grid]]) {
            // intentionally empty
        }
    "#;

    const FUNCTION_CONSTANT_SHADER: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        constant bool USE_ADD [[function_constant(0)]];
        constant uint FACTOR [[function_constant(1)]];

        kernel void specialize_values(
            device float* data [[buffer(0)]],
            uint idx [[thread_position_in_grid]]
        ) {
            float factor = float(FACTOR);
            if (USE_ADD) {
                data[idx] = data[idx] + factor;
            } else {
                data[idx] = data[idx] * factor;
            }
        }
    "#;

    const DOUBLE_SHADER: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void double_values(
            device float* data [[buffer(0)]],
            uint idx [[thread_position_in_grid]]
        ) {
            data[idx] = data[idx] * 2.0;
        }
    "#;

    #[test]
    fn test_compile_trivial_shader() {
        let gpu = MetalDevice::new().unwrap();
        let pipeline = ComputePipeline::from_source(gpu.device(), TRIVIAL_SHADER, "noop").unwrap();
        assert!(pipeline.max_threads_per_threadgroup() >= 256);
        assert!(pipeline.thread_execution_width() > 0);
    }

    #[test]
    fn test_compile_double_shader() {
        let gpu = MetalDevice::new().unwrap();
        let pipeline =
            ComputePipeline::from_source(gpu.device(), DOUBLE_SHADER, "double_values").unwrap();
        assert!(pipeline.max_threads_per_threadgroup() >= 256);
    }

    #[test]
    fn test_compile_bad_source() {
        let gpu = MetalDevice::new().unwrap();
        let result = ComputePipeline::from_source(gpu.device(), "invalid metal code!!!", "noop");
        assert!(result.is_err(), "Invalid source should fail to compile");
    }

    #[test]
    fn test_compile_missing_function() {
        let gpu = MetalDevice::new().unwrap();
        let result =
            ComputePipeline::from_source(gpu.device(), TRIVIAL_SHADER, "nonexistent_function");
        assert!(result.is_err(), "Missing function should fail");
    }

    #[test]
    fn test_compile_and_dispatch_with_function_constants() {
        let gpu = MetalDevice::new().unwrap();

        let add_pipeline = ComputePipeline::from_source_with_constants(
            gpu.device(),
            FUNCTION_CONSTANT_SHADER,
            "specialize_values",
            &[
                FunctionConstant {
                    index: 0,
                    value: FunctionConstantValue::Bool(true),
                },
                FunctionConstant {
                    index: 1,
                    value: FunctionConstantValue::U32(3),
                },
            ],
        )
        .unwrap();

        let mul_pipeline = ComputePipeline::from_source_with_constants(
            gpu.device(),
            FUNCTION_CONSTANT_SHADER,
            "specialize_values",
            &[
                FunctionConstant {
                    index: 0,
                    value: FunctionConstantValue::Bool(false),
                },
                FunctionConstant {
                    index: 1,
                    value: FunctionConstantValue::U32(3),
                },
            ],
        )
        .unwrap();

        let add_buf = MetalBuffer::from_slice(gpu.device(), &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let mul_buf = MetalBuffer::from_slice(gpu.device(), &[1.0f32, 2.0, 3.0, 4.0]).unwrap();

        gpu.execute_sync(|encoder| {
            encoder.setComputePipelineState(add_pipeline.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(add_buf.mtl_buffer()), 0, 0);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 4,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        })
        .unwrap();

        gpu.execute_sync(|encoder| {
            encoder.setComputePipelineState(mul_pipeline.state());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(mul_buf.mtl_buffer()), 0, 0);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 4,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        })
        .unwrap();

        let add_out = unsafe { add_buf.as_slice::<f32>() };
        let mul_out = unsafe { mul_buf.as_slice::<f32>() };
        assert_eq!(add_out, &[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(mul_out, &[3.0, 6.0, 9.0, 12.0]);
    }
}
