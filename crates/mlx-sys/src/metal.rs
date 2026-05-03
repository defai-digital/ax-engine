use std::ffi::CString;
use std::ptr;

use crate::array::MlxArray;
use crate::ffi;
use crate::stream::MlxStream;

/// A compiled custom Metal kernel callable from within the MLX compute graph.
pub struct MlxMetalKernel {
    inner: ffi::mlx_fast_metal_kernel,
}

unsafe impl Send for MlxMetalKernel {}
unsafe impl Sync for MlxMetalKernel {}

impl MlxMetalKernel {
    /// Register a custom Metal kernel.
    ///
    /// `input_names` and `output_names` must match the buffer bindings in `source`.
    pub fn new(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
        header: &str,
        ensure_row_contiguous: bool,
    ) -> Self {
        unsafe {
            let c_name = CString::new(name).expect("Metal kernel name must not contain NUL bytes");
            let c_source =
                CString::new(source).expect("Metal kernel source must not contain NUL bytes");
            let c_header =
                CString::new(header).expect("Metal kernel header must not contain NUL bytes");

            let inputs = build_string_vec(input_names);
            let outputs = build_string_vec(output_names);

            let inner = ffi::mlx_fast_metal_kernel_new(
                c_name.as_ptr(),
                inputs,
                outputs,
                c_source.as_ptr(),
                c_header.as_ptr(),
                ensure_row_contiguous,
                false, // atomic_outputs
            );

            ffi::mlx_vector_string_free(inputs);
            ffi::mlx_vector_string_free(outputs);

            Self { inner }
        }
    }

    /// Call the kernel.
    ///
    /// Returns the output arrays in the order declared in `output_names`.
    pub fn apply(
        &self,
        inputs: &[&MlxArray],
        output_specs: &[KernelOutputSpec],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        s: Option<&MlxStream>,
    ) -> Vec<MlxArray> {
        unsafe {
            let stream = s
                .map(|s| s.inner)
                .unwrap_or_else(|| ffi::mlx_default_gpu_stream_new());

            // Build input vector.
            let in_vec = ffi::mlx_vector_array_new();
            for arr in inputs {
                ffi::mlx_vector_array_append_value(in_vec, arr.inner);
            }

            // Build config.
            let config = ffi::mlx_fast_metal_kernel_config_new();
            for spec in output_specs {
                ffi::mlx_fast_metal_kernel_config_add_output_arg(
                    config,
                    spec.shape.as_ptr(),
                    spec.shape.len(),
                    spec.dtype.to_ffi(),
                );
            }
            ffi::mlx_fast_metal_kernel_config_set_grid(config, grid.0, grid.1, grid.2);
            ffi::mlx_fast_metal_kernel_config_set_thread_group(
                config,
                thread_group.0,
                thread_group.1,
                thread_group.2,
            );

            let mut out_vec = ffi::mlx_vector_array_new();
            ffi::mlx_fast_metal_kernel_apply(&mut out_vec, self.inner, in_vec, config, stream);

            ffi::mlx_vector_array_free(in_vec);
            ffi::mlx_fast_metal_kernel_config_free(config);

            // Collect outputs.
            let n = ffi::mlx_vector_array_size(out_vec);
            let mut result = Vec::with_capacity(n);
            for i in 0..n {
                let mut arr = MlxArray::empty();
                ffi::mlx_vector_array_get(&mut arr.inner, out_vec, i);
                result.push(arr);
            }
            ffi::mlx_vector_array_free(out_vec);
            result
        }
    }
}

impl Drop for MlxMetalKernel {
    fn drop(&mut self) {
        if !self.inner.ctx.is_null() {
            unsafe { ffi::mlx_fast_metal_kernel_free(self.inner) };
            self.inner.ctx = ptr::null_mut();
        }
    }
}

/// Shape and dtype for one output buffer declared in the kernel.
pub struct KernelOutputSpec {
    pub shape: Vec<i32>,
    pub dtype: crate::array::MlxDtype,
}

fn build_string_vec(strs: &[&str]) -> ffi::mlx_vector_string {
    unsafe {
        let vec = ffi::mlx_vector_string_new();
        for s in strs {
            let cs =
                CString::new(*s).expect("Metal kernel argument names must not contain NUL bytes");
            ffi::mlx_vector_string_append_value(vec, cs.as_ptr());
        }
        vec
    }
}
