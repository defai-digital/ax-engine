//! Safe RAII wrappers for `mlx_closure` and `mlx_vector_array`, plus
//! `mlx_compile`. Used by ax-engine-mlx to wrap the transformer forward in
//! a compiled closure, fusing per-layer ops and amortizing the FFI dispatch
//! cost of each MLX C-API call.
//!
//! The closure body is provided as a boxed Rust trait object (`FnMut`).
//! A non-generic `extern "C"` trampoline forwards calls into that box.
//! Memory is owned by `MlxClosure::Drop`, which calls back into the C dtor
//! and frees the boxed payload exactly once.

use std::fmt;
use std::os::raw::{c_int, c_void};
use std::ptr;
use std::thread::{self, ThreadId};

use crate::array::{MlxArray, null_ffi_array};
use crate::ffi;

fn null_vector_array() -> ffi::mlx_vector_array {
    ffi::mlx_vector_array {
        ctx: ptr::null_mut(),
    }
}

fn null_closure() -> ffi::mlx_closure {
    ffi::mlx_closure {
        ctx: ptr::null_mut(),
    }
}

/// RAII wrapper for `mlx_vector_array`. Used as input/output container for
/// closure invocations.
pub struct MlxVectorArray {
    pub(crate) inner: ffi::mlx_vector_array,
    /// When false, Drop will not free — the underlying vec is owned by MLX
    /// (e.g., the inputs vec passed into a trampoline).
    owned: bool,
}

impl MlxVectorArray {
    pub fn new() -> Self {
        let inner = unsafe { ffi::mlx_vector_array_new() };
        Self { inner, owned: true }
    }

    /// Build from a slice of MlxArrays. Each array's handle is appended
    /// (incrementing the underlying refcount).
    pub fn from_arrays(arrays: &[&MlxArray]) -> Self {
        let v = Self::new();
        for arr in arrays {
            unsafe {
                ffi::mlx_vector_array_append_value(v.inner, arr.inner);
            }
        }
        v
    }

    /// Wrap an externally-owned vector array (e.g. inputs from MLX). Drop
    /// will not free it.
    fn from_borrowed(inner: ffi::mlx_vector_array) -> Self {
        Self {
            inner,
            owned: false,
        }
    }

    pub fn len(&self) -> usize {
        unsafe { ffi::mlx_vector_array_size(self.inner) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, idx: usize) -> MlxArray {
        let mut raw = null_ffi_array();
        unsafe {
            ffi::mlx_vector_array_get(&mut raw, self.inner, idx);
        }
        // SAFETY: mlx_vector_array_get returns an owned array reference
        // (refcounted internally); MlxArray::Drop will free it.
        unsafe { MlxArray::from_raw(raw) }
    }

    /// Transfer ownership of the inner handle to the caller. The returned
    /// raw handle must be freed by the caller (e.g. by writing into an MLX
    /// out-parameter). Self is consumed and will not free.
    fn into_raw(mut self) -> ffi::mlx_vector_array {
        self.owned = false;
        self.inner
    }
}

impl Default for MlxVectorArray {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MlxVectorArray {
    fn drop(&mut self) {
        if self.owned && !self.inner.ctx.is_null() {
            unsafe {
                ffi::mlx_vector_array_free(self.inner);
            }
            self.inner.ctx = ptr::null_mut();
        }
    }
}

/// Trait object stored in the closure's payload. The trampoline
/// dereferences this to invoke the Rust body.
type DynClosureBody = dyn FnMut(&MlxVectorArray) -> Vec<MlxArray> + Send + 'static;

unsafe extern "C" fn closure_trampoline(
    res: *mut ffi::mlx_vector_array,
    inputs: ffi::mlx_vector_array,
    payload: *mut c_void,
) -> c_int {
    // SAFETY: payload was created in `MlxClosure::new_dyn` from
    // `Box::into_raw(Box::new(Box<DynClosureBody>))`. While the closure is
    // alive on the C side, MLX will not free this pointer.
    let body: &mut Box<DynClosureBody> = unsafe { &mut *(payload as *mut Box<DynClosureBody>) };

    let inputs_wrapped = MlxVectorArray::from_borrowed(inputs);
    let outputs = body(&inputs_wrapped);

    let out_vec = MlxVectorArray::new();
    for arr in &outputs {
        unsafe {
            ffi::mlx_vector_array_append_value(out_vec.inner, arr.inner);
        }
    }

    // SAFETY: res is a valid out-pointer per the C API contract. We
    // transfer ownership of out_vec.inner to it.
    unsafe {
        *res = out_vec.into_raw();
    }

    0
}

unsafe extern "C" fn closure_dtor(payload: *mut c_void) {
    if payload.is_null() {
        return;
    }
    // SAFETY: mirrors the `Box::into_raw(Box<Box<DynClosureBody>>)` in
    // `MlxClosure::new_dyn`. Single ownership; MLX guarantees the dtor
    // runs exactly once.
    unsafe {
        let _ = Box::from_raw(payload as *mut Box<DynClosureBody>);
    }
}

/// RAII wrapper for `mlx_closure`. Holds either an original (user-built)
/// closure or its compiled form.
pub struct MlxClosure {
    pub(crate) inner: ffi::mlx_closure,
    compiled_on: Option<ThreadId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlxClosureApplyError {
    CompiledClosureThreadMismatch {
        compiled_on: ThreadId,
        current: ThreadId,
    },
}

impl fmt::Display for MlxClosureApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CompiledClosureThreadMismatch {
                compiled_on,
                current,
            } => write!(
                f,
                "compiled MLX closure cannot be applied on thread {current:?}; it was compiled on {compiled_on:?}"
            ),
        }
    }
}

impl MlxClosure {
    /// Wrap a Rust closure body as an MLX closure.
    ///
    /// The body is called either (a) once during `compile()` for tracing,
    /// (b) any number of times during `apply()`. The body MUST be pure with
    /// respect to MLX state — its only side effect should be building the
    /// returned graph nodes. Captured MlxArrays (e.g. weights) become
    /// constants in the compiled graph.
    pub fn new_dyn<F>(f: F) -> Self
    where
        F: FnMut(&MlxVectorArray) -> Vec<MlxArray> + Send + 'static,
    {
        let body: Box<DynClosureBody> = Box::new(f);
        let payload_box: Box<Box<DynClosureBody>> = Box::new(body);
        let payload_raw = Box::into_raw(payload_box) as *mut c_void;

        let cls = unsafe {
            ffi::mlx_closure_new_func_payload(
                Some(closure_trampoline),
                payload_raw,
                Some(closure_dtor),
            )
        };
        Self {
            inner: cls,
            compiled_on: None,
        }
    }

    /// Compile this closure with `mlx_compile`. The compiled closure can be
    /// reused across `apply()` calls; MLX will skip re-tracing and just
    /// dispatch the cached compiled graph (potentially with fused ops).
    ///
    /// `shapeless=true` lets the compiled closure accept different shapes
    /// without recompiling, at some cost in fusion aggressiveness. Use
    /// `false` when the input shape is stable across calls.
    pub fn compile(&self, shapeless: bool) -> Result<Self, &'static str> {
        let mut out = null_closure();
        let rc = unsafe { ffi::mlx_compile(&mut out, self.inner, shapeless) };
        if rc != 0 {
            return Err("mlx_compile failed");
        }
        Ok(Self {
            inner: out,
            compiled_on: Some(thread::current().id()),
        })
    }

    /// Invoke the closure on the given inputs. Returns the outputs as a
    /// `Vec<MlxArray>`. Each output is a refcounted handle owned by the
    /// caller.
    pub fn apply(&self, inputs: &[&MlxArray]) -> Vec<MlxArray> {
        self.try_apply(inputs)
            .expect("compiled MLX closure applied from the wrong thread")
    }

    /// Fallible variant of `apply`.
    ///
    /// MLX 0.31 keeps both default streams and Metal command encoders in
    /// thread-local registries. A closure compiled on one thread can abort
    /// inside `mlx_closure_apply` if it is applied on a different thread whose
    /// registry does not contain the compiled stream. Guard that contract here
    /// so production callers can fall back or compile a per-thread entry before
    /// crossing into the C API.
    pub fn try_apply(&self, inputs: &[&MlxArray]) -> Result<Vec<MlxArray>, MlxClosureApplyError> {
        if let Some(compiled_on) = self.compiled_on {
            let current = thread::current().id();
            if current != compiled_on {
                return Err(MlxClosureApplyError::CompiledClosureThreadMismatch {
                    compiled_on,
                    current,
                });
            }
        }
        // Count one MLX dispatch per closure invocation. For a compiled
        // closure this is the single underlying graph dispatch; for an
        // uncompiled closure it is the trampoline call. Either way, this
        // mirrors the per-op accounting used by every direct-FFI wrapper
        // so op-count A/Bs across compile vs imperative paths are fair.
        crate::op_count::bump();
        let in_vec = MlxVectorArray::from_arrays(inputs);
        let mut out_raw = null_vector_array();
        unsafe {
            ffi::mlx_closure_apply(&mut out_raw, self.inner, in_vec.inner);
        }
        let out_vec = MlxVectorArray {
            inner: out_raw,
            owned: true,
        };
        let len = out_vec.len();
        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            result.push(out_vec.get(i));
        }
        Ok(result)
    }
}

impl Drop for MlxClosure {
    fn drop(&mut self) {
        if !self.inner.ctx.is_null() {
            unsafe {
                ffi::mlx_closure_free(self.inner);
            }
            self.inner.ctx = ptr::null_mut();
        }
    }
}

// SAFETY: MlxClosure holds a refcounted C handle. The trampoline and
// payload boxes are managed via Send-bounded closures. MLX itself is
// thread-safe for closure invocation.
unsafe impl Send for MlxClosure {}
unsafe impl Sync for MlxClosure {}
unsafe impl Send for MlxVectorArray {}
unsafe impl Sync for MlxVectorArray {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::MlxDtype;
    use crate::ops::add;
    use crate::transforms::eval;

    fn const_f32_1d(values: &[f32]) -> MlxArray {
        MlxArray::from_raw_data(
            values.as_ptr() as *const u8,
            std::mem::size_of_val(values),
            &[values.len() as i32],
            MlxDtype::Float32,
        )
    }

    #[test]
    fn closure_apply_runs_body() {
        // Body: add input[0] to itself (i.e., 2*x).
        let closure = MlxClosure::new_dyn(|inputs: &MlxVectorArray| {
            let x = inputs.get(0);
            vec![add(&x, &x, None)]
        });

        let x = const_f32_1d(&[1.0, 2.0, 3.0, 4.0]);
        let out = closure.apply(&[&x]);
        assert_eq!(out.len(), 1);
        eval(&[&out[0]]);
        let buf = out[0].data_f32().to_vec();
        assert_eq!(buf, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn compiled_closure_matches_uncompiled() {
        // Body: x + x + x = 3x (3 ops to give compile something to fuse).
        let body_factory = || {
            MlxClosure::new_dyn(|inputs: &MlxVectorArray| {
                let x = inputs.get(0);
                let two_x = add(&x, &x, None);
                vec![add(&two_x, &x, None)]
            })
        };

        let raw = body_factory();
        let compiled = body_factory().compile(false).expect("compile must succeed");

        let x = const_f32_1d(&[1.0, 2.0, 3.0]);
        let raw_out = raw.apply(&[&x]);
        let comp_out = compiled.apply(&[&x]);
        eval(&[&raw_out[0], &comp_out[0]]);
        assert_eq!(raw_out[0].data_f32().to_vec(), vec![3.0, 6.0, 9.0]);
        assert_eq!(comp_out[0].data_f32().to_vec(), vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn compiled_closure_rejects_cross_thread_apply_before_mlx_abort() {
        let compiled = MlxClosure::new_dyn(|inputs: &MlxVectorArray| {
            let x = inputs.get(0);
            vec![add(&x, &x, None)]
        })
        .compile(false)
        .expect("compile must succeed");

        let handle = std::thread::spawn(move || {
            let x = const_f32_1d(&[1.0, 2.0, 3.0]);
            compiled.try_apply(&[&x])
        });
        let result = handle.join().expect("thread must not panic");
        let err = match result {
            Ok(_) => panic!("cross-thread apply must be rejected before calling mlx_closure_apply"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            MlxClosureApplyError::CompiledClosureThreadMismatch { .. }
        ));
    }
}
