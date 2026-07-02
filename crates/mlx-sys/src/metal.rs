use std::ffi::CString;
use std::ptr;

use crate::array::{MlxArray, MlxDtype};
use crate::error::{last_error_message, panic_on_status, prepare_error_capture, status_to_result};
use crate::ffi;
use crate::stream::{MlxStream, default_gpu_raw};

/// A compiled custom Metal kernel callable from within the MLX compute graph.
pub struct MlxMetalKernel {
    inner: ffi::mlx_fast_metal_kernel,
    name: String,
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
        prepare_error_capture();
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

            if inner.ctx.is_null() {
                panic!("{}", last_error_message("mlx_fast_metal_kernel_new"));
            }

            Self {
                inner,
                name: name.to_string(),
            }
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
        self.apply_with_template(inputs, output_specs, &[], grid, thread_group, s)
    }

    /// Call the kernel with template arguments.
    ///
    /// MLX fast kernels use template arguments for compile-time constants such
    /// as dtypes and head dimensions.  The names must match the identifiers used
    /// in the kernel source.
    ///
    /// Panics on kernel-application failure. Callers with a fallback path
    /// should use [`Self::try_apply_with_template`] instead.
    pub fn apply_with_template(
        &self,
        inputs: &[&MlxArray],
        output_specs: &[KernelOutputSpec],
        template_args: &[KernelTemplateArg<'_>],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        s: Option<&MlxStream>,
    ) -> Vec<MlxArray> {
        self.try_apply_with_template(inputs, output_specs, template_args, grid, thread_group, s)
            .unwrap_or_else(|message| panic!("{message}"))
    }

    /// Call the kernel with template arguments, surfacing MLX errors as `Err`
    /// instead of killing the process (the recording handler installed by this
    /// crate keeps the process alive so the status code can be checked).
    ///
    /// Note that MLX evaluation is lazy: kernel-source compile errors only
    /// surface when the returned arrays are evaluated, so callers must pair
    /// this with [`crate::transforms::try_eval`] for full error coverage.
    pub fn try_apply_with_template(
        &self,
        inputs: &[&MlxArray],
        output_specs: &[KernelOutputSpec],
        template_args: &[KernelTemplateArg<'_>],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        s: Option<&MlxStream>,
    ) -> Result<Vec<MlxArray>, String> {
        prepare_error_capture();
        unsafe {
            let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);

            // Build input vector.
            let in_vec = ffi::mlx_vector_array_new();
            for arr in inputs {
                prepare_error_capture();
                let rc = ffi::mlx_vector_array_append_value(in_vec, arr.inner);
                panic_on_status("mlx_vector_array_append_value", rc);
            }

            // Build config.
            let config = ffi::mlx_fast_metal_kernel_config_new();
            if config.ctx.is_null() {
                ffi::mlx_vector_array_free(in_vec);
                return Err(last_error_message("mlx_fast_metal_kernel_config_new"));
            }
            for spec in output_specs {
                prepare_error_capture();
                let rc = ffi::mlx_fast_metal_kernel_config_add_output_arg(
                    config,
                    spec.shape.as_ptr(),
                    spec.shape.len(),
                    spec.dtype.to_ffi(),
                );
                if let Err(message) =
                    status_to_result("mlx_fast_metal_kernel_config_add_output_arg", rc)
                {
                    ffi::mlx_vector_array_free(in_vec);
                    ffi::mlx_fast_metal_kernel_config_free(config);
                    return Err(message);
                }
            }
            let mut template_names = Vec::with_capacity(template_args.len());
            for arg in template_args {
                let name = CString::new(arg.name())
                    .expect("Metal kernel template argument names must not contain NUL bytes");
                match *arg {
                    KernelTemplateArg::Dtype { dtype, .. } => {
                        prepare_error_capture();
                        let rc = ffi::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                            config,
                            name.as_ptr(),
                            dtype.to_ffi(),
                        );
                        if let Err(message) = status_to_result(
                            "mlx_fast_metal_kernel_config_add_template_arg_dtype",
                            rc,
                        ) {
                            ffi::mlx_vector_array_free(in_vec);
                            ffi::mlx_fast_metal_kernel_config_free(config);
                            return Err(message);
                        }
                    }
                    KernelTemplateArg::Int { value, .. } => {
                        prepare_error_capture();
                        let rc = ffi::mlx_fast_metal_kernel_config_add_template_arg_int(
                            config,
                            name.as_ptr(),
                            value,
                        );
                        if let Err(message) = status_to_result(
                            "mlx_fast_metal_kernel_config_add_template_arg_int",
                            rc,
                        ) {
                            ffi::mlx_vector_array_free(in_vec);
                            ffi::mlx_fast_metal_kernel_config_free(config);
                            return Err(message);
                        }
                    }
                    KernelTemplateArg::Bool { value, .. } => {
                        prepare_error_capture();
                        let rc = ffi::mlx_fast_metal_kernel_config_add_template_arg_bool(
                            config,
                            name.as_ptr(),
                            value,
                        );
                        if let Err(message) = status_to_result(
                            "mlx_fast_metal_kernel_config_add_template_arg_bool",
                            rc,
                        ) {
                            ffi::mlx_vector_array_free(in_vec);
                            ffi::mlx_fast_metal_kernel_config_free(config);
                            return Err(message);
                        }
                    }
                }
                template_names.push(name);
            }
            prepare_error_capture();
            let rc = ffi::mlx_fast_metal_kernel_config_set_grid(config, grid.0, grid.1, grid.2);
            if let Err(message) = status_to_result("mlx_fast_metal_kernel_config_set_grid", rc) {
                ffi::mlx_vector_array_free(in_vec);
                ffi::mlx_fast_metal_kernel_config_free(config);
                return Err(message);
            }
            prepare_error_capture();
            let rc = ffi::mlx_fast_metal_kernel_config_set_thread_group(
                config,
                thread_group.0,
                thread_group.1,
                thread_group.2,
            );
            if let Err(message) =
                status_to_result("mlx_fast_metal_kernel_config_set_thread_group", rc)
            {
                ffi::mlx_vector_array_free(in_vec);
                ffi::mlx_fast_metal_kernel_config_free(config);
                return Err(message);
            }

            let mut out_vec = ffi::mlx_vector_array_new();
            prepare_error_capture();
            let rc =
                ffi::mlx_fast_metal_kernel_apply(&mut out_vec, self.inner, in_vec, config, stream);

            ffi::mlx_vector_array_free(in_vec);
            ffi::mlx_fast_metal_kernel_config_free(config);

            if rc != 0 {
                ffi::mlx_vector_array_free(out_vec);
                return Err(last_error_message(&format!(
                    "MLX Metal kernel '{}' apply",
                    self.name
                )));
            }

            // Collect outputs.
            let n = ffi::mlx_vector_array_size(out_vec);
            if n == usize::MAX {
                ffi::mlx_vector_array_free(out_vec);
                return Err(last_error_message(&format!(
                    "MLX Metal kernel '{}' output size",
                    self.name
                )));
            }
            let mut result = Vec::with_capacity(n);
            for i in 0..n {
                let mut arr = MlxArray::empty();
                prepare_error_capture();
                let rc = ffi::mlx_vector_array_get(&mut arr.inner, out_vec, i);
                if let Err(message) = status_to_result("mlx_vector_array_get", rc) {
                    ffi::mlx_vector_array_free(out_vec);
                    return Err(message);
                }
                result.push(arr);
            }
            ffi::mlx_vector_array_free(out_vec);
            Ok(result)
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

pub enum KernelTemplateArg<'a> {
    Dtype { name: &'a str, dtype: MlxDtype },
    Int { name: &'a str, value: i32 },
    Bool { name: &'a str, value: bool },
}

impl KernelTemplateArg<'_> {
    fn name(&self) -> &str {
        match self {
            Self::Dtype { name, .. } | Self::Int { name, .. } | Self::Bool { name, .. } => name,
        }
    }
}

fn build_string_vec(strs: &[&str]) -> ffi::mlx_vector_string {
    unsafe {
        let vec = ffi::mlx_vector_string_new();
        for s in strs {
            let cs =
                CString::new(*s).expect("Metal kernel argument names must not contain NUL bytes");
            prepare_error_capture();
            let rc = ffi::mlx_vector_string_append_value(vec, cs.as_ptr());
            panic_on_status("mlx_vector_string_append_value", rc);
        }
        vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::try_eval;

    #[test]
    fn broken_kernel_source_surfaces_err_and_leaves_mlx_usable() {
        // A kernel with an MSL syntax error. Without the recording error
        // handler this would exit(-1) the whole process at eval time; the
        // recording handler turns it into an Err from try_eval.
        let kernel = MlxMetalKernel::new(
            "mlx_sys_test_broken_kernel",
            &["input"],
            &["output"],
            "this is not valid metal source;",
            "",
            true,
        );
        let input_data = [1.0f32, 2.0, 3.0, 4.0];
        let input = MlxArray::from_raw_data(
            input_data.as_ptr().cast(),
            std::mem::size_of_val(input_data.as_slice()),
            &[4],
            MlxDtype::Float32,
        );

        let outputs = kernel.try_apply_with_template(
            &[&input],
            &[KernelOutputSpec {
                shape: vec![4],
                dtype: MlxDtype::Float32,
            }],
            &[],
            (4, 1, 1),
            (4, 1, 1),
            None,
        );

        // The failure may surface at apply time or lazily at eval time; both
        // must be an Err rather than process death.
        let eval_result = match outputs {
            Err(message) => Err(message),
            Ok(outputs) => {
                let refs = outputs.iter().collect::<Vec<_>>();
                try_eval(&refs)
            }
        };
        let message = eval_result.expect_err("broken kernel must surface an error");
        assert!(
            message.contains("failed"),
            "error message should describe the failure: {message}"
        );

        // The process must remain usable: a valid kernel still runs.
        let valid = MlxMetalKernel::new(
            "mlx_sys_test_valid_kernel",
            &["input"],
            &["output"],
            "uint i = thread_position_in_grid.x; output[i] = input[i] * 2.0f;",
            "",
            true,
        );
        let outputs = valid
            .try_apply_with_template(
                &[&input],
                &[KernelOutputSpec {
                    shape: vec![4],
                    dtype: MlxDtype::Float32,
                }],
                &[],
                (4, 1, 1),
                (4, 1, 1),
                None,
            )
            .expect("valid kernel applies");
        let refs = outputs.iter().collect::<Vec<_>>();
        try_eval(&refs).expect("valid kernel evaluates after a prior failure");
        assert_eq!(outputs[0].data_f32(), &[2.0, 4.0, 6.0, 8.0]);
    }
}
