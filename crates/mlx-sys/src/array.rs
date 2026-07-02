use std::fmt;
use std::ptr;

use crate::error::{
    ensure_error_handler, last_error_message, panic_on_status, prepare_error_capture,
    take_last_error,
};
use crate::ffi;

#[inline]
pub(crate) fn null_ffi_array() -> ffi::mlx_array {
    ffi::mlx_array {
        ctx: ptr::null_mut(),
    }
}

/// A reference-counted MLX N-dimensional array.
///
/// Wraps `mlx_array` and calls `mlx_array_free` on drop.
pub struct MlxArray {
    pub(crate) inner: ffi::mlx_array,
}

// MLX's C API wraps C++ shared ownership internally; this relies on atomic
// reference-count updates in that implementation rather than a stronger C API
// contract.
unsafe impl Send for MlxArray {}
unsafe impl Sync for MlxArray {}

impl MlxArray {
    /// Create from a raw `mlx_array`. Takes ownership — caller must not free.
    pub(crate) unsafe fn from_raw(inner: ffi::mlx_array) -> Self {
        Self { inner }
    }

    /// Build an empty (null) array. Useful as an out-parameter placeholder.
    pub(crate) fn empty() -> Self {
        Self {
            inner: null_ffi_array(),
        }
    }

    /// Whether this wrapper currently holds no MLX array.
    ///
    /// This is a defensive inspection helper for FFI wrappers; public callers
    /// cannot construct a null `MlxArray`.
    pub fn is_null(&self) -> bool {
        self.inner.ctx.is_null()
    }

    /// Create a 1-D f32 array from a slice.
    pub fn from_f32_slice(data: &[f32]) -> Self {
        unsafe {
            let shape = [data.len() as i32];
            prepare_error_capture();
            let arr = ffi::mlx_array_new_data(
                data.as_ptr() as *const _,
                shape.as_ptr(),
                1,
                ffi::mlx_dtype_::MLX_FLOAT32,
            );
            panic_on_null_array("mlx_array_new_data", arr);
            Self::from_raw(arr)
        }
    }

    /// Create a scalar f32 array (shape `[]`).
    pub fn from_f32(value: f32) -> Self {
        unsafe {
            prepare_error_capture();
            let arr = ffi::mlx_array_new_data(
                &value as *const f32 as *const _,
                ptr::null(),
                0,
                ffi::mlx_dtype_::MLX_FLOAT32,
            );
            panic_on_null_array("mlx_array_new_data", arr);
            Self::from_raw(arr)
        }
    }

    /// Create a 1-D f16 array from raw bytes (already packed as f16).
    pub fn from_f16_bytes(data: &[u8]) -> Self {
        assert!(
            data.len().is_multiple_of(2),
            "f16 data must have even byte length"
        );
        let len = data.len() / 2;
        unsafe {
            let shape = [len as i32];
            prepare_error_capture();
            let arr = ffi::mlx_array_new_data(
                data.as_ptr() as *const _,
                shape.as_ptr(),
                1,
                ffi::mlx_dtype_::MLX_FLOAT16,
            );
            panic_on_null_array("mlx_array_new_data", arr);
            Self::from_raw(arr)
        }
    }

    /// Create an array that borrows an externally-owned data buffer (e.g.
    /// a memory-mapped safetensors region). MLX does **not** copy the
    /// data; the `dtor` callback is invoked when MLX no longer references
    /// the array, at which point the caller's resource can be released.
    /// `payload` is passed through to `dtor` so the caller can carry
    /// arbitrary lifetime state (typically a boxed `Mmap` handle).
    ///
    /// # Safety
    ///
    /// `data` must remain valid until `dtor(payload)` is called. `byte_len`
    /// must cover the array's element count for `dtype`.
    pub unsafe fn from_managed_data(
        data: *const u8,
        byte_len: usize,
        shape: &[i32],
        dtype: MlxDtype,
        payload: *mut std::ffi::c_void,
        dtor: unsafe extern "C" fn(*mut std::ffi::c_void),
    ) -> Self {
        let element_count = shape
            .iter()
            .try_fold(1usize, |acc, dim| {
                usize::try_from(*dim)
                    .ok()
                    .and_then(|dim| acc.checked_mul(dim))
            })
            .expect("shape dimensions must be non-negative and fit in usize");
        let required_bytes = element_count
            .checked_mul(dtype.size_bytes())
            .expect("shape byte length must fit in usize");
        assert!(
            byte_len >= required_bytes,
            "managed data byte length {byte_len} is smaller than required {required_bytes}"
        );
        unsafe {
            prepare_error_capture();
            let arr = ffi::mlx_array_new_data_managed_payload(
                data as *mut _,
                shape.as_ptr(),
                shape.len() as i32,
                dtype.to_ffi(),
                payload,
                Some(dtor),
            );
            panic_on_null_array("mlx_array_new_data_managed_payload", arr);
            Self::from_raw(arr)
        }
    }

    /// Create an array from a raw data pointer with explicit shape and dtype.
    pub fn from_raw_data(data: *const u8, byte_len: usize, shape: &[i32], dtype: MlxDtype) -> Self {
        let element_count = shape
            .iter()
            .try_fold(1usize, |acc, dim| {
                usize::try_from(*dim)
                    .ok()
                    .and_then(|dim| acc.checked_mul(dim))
            })
            .expect("shape dimensions must be non-negative and fit in usize");
        let required_bytes = element_count
            .checked_mul(dtype.size_bytes())
            .expect("shape byte length must fit in usize");
        assert!(
            byte_len >= required_bytes,
            "raw data byte length {byte_len} is smaller than required {required_bytes}"
        );
        unsafe {
            prepare_error_capture();
            let arr = ffi::mlx_array_new_data(
                data as *const _,
                shape.as_ptr(),
                shape.len() as i32,
                dtype.to_ffi(),
            );
            panic_on_null_array("mlx_array_new_data", arr);
            Self::from_raw(arr)
        }
    }

    pub fn ndim(&self) -> usize {
        unsafe {
            ensure_error_handler();
            let n = ffi::mlx_array_ndim(self.inner);
            if n == usize::MAX {
                panic!("{}", last_error_message("mlx_array_ndim"));
            }
            n
        }
    }

    pub fn shape(&self) -> Vec<i32> {
        unsafe {
            ensure_error_handler();
            let ndim = ffi::mlx_array_ndim(self.inner);
            if ndim == usize::MAX {
                panic!("{}", last_error_message("mlx_array_ndim"));
            }
            let ptr = ffi::mlx_array_shape(self.inner);
            if ptr.is_null() || ndim == 0 {
                // A null pointer with ndim > 0 means mlx_array_shape errored
                // and the handler wrote the slot; drain it so the message is
                // not misattributed to a later call on this thread.
                if ptr.is_null() && ndim > 0 {
                    crate::error::clear_stale_error();
                }
                return vec![];
            }
            std::slice::from_raw_parts(ptr, ndim).to_vec()
        }
    }

    pub fn dtype(&self) -> MlxDtype {
        unsafe {
            ensure_error_handler();
            let d = ffi::mlx_array_dtype(self.inner);
            // The C++ shim returns MLX_BOOL as sentinel on error; check the
            // error slot to distinguish a genuine bool dtype from a failure.
            if d == ffi::mlx_dtype_::MLX_BOOL
                && let Some(msg) = take_last_error()
            {
                panic!("mlx_array_dtype failed: {msg}");
            }
            MlxDtype::from_ffi(d)
        }
    }

    pub fn nbytes(&self) -> usize {
        unsafe {
            ensure_error_handler();
            let n = ffi::mlx_array_nbytes(self.inner);
            if n == usize::MAX {
                panic!("{}", last_error_message("mlx_array_nbytes"));
            }
            n
        }
    }

    /// Read data as f32. Array must have been eval'd first.
    pub fn data_f32(&self) -> &[f32] {
        assert_eq!(
            self.dtype(),
            MlxDtype::Float32,
            "data_f32 requires Float32 dtype"
        );
        unsafe {
            ensure_error_handler();
            let ptr = ffi::mlx_array_data_float32(self.inner);
            if ptr.is_null() {
                if let Some(msg) = take_last_error() {
                    panic!("mlx_array_data_float32 failed: {msg}");
                }
                return &[];
            }
            let len = self.nbytes() / self.dtype().size_bytes();
            if len == 0 {
                return &[];
            }
            std::slice::from_raw_parts(ptr, len)
        }
    }

    /// Read data as u32. Array must have been eval'd first.
    pub fn data_u32(&self) -> &[u32] {
        assert_eq!(
            self.dtype(),
            MlxDtype::Uint32,
            "data_u32 requires Uint32 dtype"
        );
        unsafe {
            ensure_error_handler();
            let ptr = ffi::mlx_array_data_uint32(self.inner);
            if ptr.is_null() {
                if let Some(msg) = take_last_error() {
                    panic!("mlx_array_data_uint32 failed: {msg}");
                }
                return &[];
            }
            let len = self.nbytes() / self.dtype().size_bytes();
            if len == 0 {
                return &[];
            }
            std::slice::from_raw_parts(ptr, len)
        }
    }

    /// Read the first u32 element without dtype or length validation.
    ///
    /// This is for hot scalar-token paths where the caller already owns the
    /// contract: the array must be an eval'd MLX uint32 array, typically the
    /// result of `argmax`. It avoids extra FFI metadata calls (`dtype` and
    /// `nbytes`) on every generated token.
    pub fn first_u32_unchecked(&self) -> u32 {
        debug_assert_eq!(
            self.dtype(),
            MlxDtype::Uint32,
            "first_u32_unchecked requires Uint32 dtype"
        );
        unsafe {
            ensure_error_handler();
            let ptr = ffi::mlx_array_data_uint32(self.inner);
            if ptr.is_null() { 0 } else { *ptr }
        }
    }
}

impl Clone for MlxArray {
    fn clone(&self) -> Self {
        if self.inner.ctx.is_null() {
            return Self::empty();
        }
        unsafe {
            let mut dst = ffi::mlx_array_new();
            prepare_error_capture();
            let rc = ffi::mlx_array_set(&mut dst, self.inner);
            panic_on_status("mlx_array_set", rc);
            Self { inner: dst }
        }
    }
}

fn panic_on_null_array(operation: &str, arr: ffi::mlx_array) {
    if arr.ctx.is_null() {
        panic!("{}", last_error_message(operation));
    }
}

impl Drop for MlxArray {
    fn drop(&mut self) {
        if !self.inner.ctx.is_null() {
            unsafe { ffi::mlx_array_free(self.inner) };
            self.inner.ctx = ptr::null_mut();
        }
    }
}

impl MlxArray {
    /// Return raw byte pointer to evaluated CPU data.
    pub fn data_raw(&self) -> *const u8 {
        unsafe {
            ensure_error_handler();
            ffi::mlx_array_data_uint8(self.inner)
        }
    }
}

impl fmt::Debug for MlxArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MlxArray(shape={:?}, dtype={:?})",
            self.shape(),
            self.dtype()
        )
    }
}

/// MLX element dtype, mirroring `mlx_dtype_`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MlxDtype {
    Bool,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
    Bfloat16,
    Complex64,
}

impl MlxDtype {
    pub(crate) fn to_ffi(self) -> ffi::mlx_dtype_ {
        match self {
            Self::Bool => ffi::mlx_dtype_::MLX_BOOL,
            Self::Uint8 => ffi::mlx_dtype_::MLX_UINT8,
            Self::Uint16 => ffi::mlx_dtype_::MLX_UINT16,
            Self::Uint32 => ffi::mlx_dtype_::MLX_UINT32,
            Self::Uint64 => ffi::mlx_dtype_::MLX_UINT64,
            Self::Int8 => ffi::mlx_dtype_::MLX_INT8,
            Self::Int16 => ffi::mlx_dtype_::MLX_INT16,
            Self::Int32 => ffi::mlx_dtype_::MLX_INT32,
            Self::Int64 => ffi::mlx_dtype_::MLX_INT64,
            Self::Float16 => ffi::mlx_dtype_::MLX_FLOAT16,
            Self::Float32 => ffi::mlx_dtype_::MLX_FLOAT32,
            Self::Float64 => ffi::mlx_dtype_::MLX_FLOAT64,
            Self::Bfloat16 => ffi::mlx_dtype_::MLX_BFLOAT16,
            Self::Complex64 => ffi::mlx_dtype_::MLX_COMPLEX64,
        }
    }

    pub(crate) fn from_ffi(d: ffi::mlx_dtype_) -> Self {
        match d {
            ffi::mlx_dtype_::MLX_BOOL => Self::Bool,
            ffi::mlx_dtype_::MLX_UINT8 => Self::Uint8,
            ffi::mlx_dtype_::MLX_UINT16 => Self::Uint16,
            ffi::mlx_dtype_::MLX_UINT32 => Self::Uint32,
            ffi::mlx_dtype_::MLX_UINT64 => Self::Uint64,
            ffi::mlx_dtype_::MLX_INT8 => Self::Int8,
            ffi::mlx_dtype_::MLX_INT16 => Self::Int16,
            ffi::mlx_dtype_::MLX_INT32 => Self::Int32,
            ffi::mlx_dtype_::MLX_INT64 => Self::Int64,
            ffi::mlx_dtype_::MLX_FLOAT16 => Self::Float16,
            ffi::mlx_dtype_::MLX_FLOAT32 => Self::Float32,
            ffi::mlx_dtype_::MLX_FLOAT64 => Self::Float64,
            ffi::mlx_dtype_::MLX_BFLOAT16 => Self::Bfloat16,
            ffi::mlx_dtype_::MLX_COMPLEX64 => Self::Complex64,
        }
    }

    pub fn size_bytes(self) -> usize {
        match self {
            Self::Bool | Self::Uint8 | Self::Int8 => 1,
            Self::Uint16 | Self::Int16 | Self::Float16 | Self::Bfloat16 => 2,
            Self::Uint32 | Self::Int32 | Self::Float32 => 4,
            Self::Uint64 | Self::Int64 | Self::Float64 | Self::Complex64 => 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::astype;
    use crate::transforms::eval;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Round-trip `from_raw_data` -> `astype(f32)` -> `eval` -> `data_f32`
    /// for every dtype the shim supports.  This guards against the critical
    /// bug class where the C++ layer casts `data` to the wrong typed pointer,
    /// silently corrupting multi-byte elements.
    #[test]
    fn roundtrip_all_integer_dtypes_via_astype_f32() {
        // -- uint8 --
        let vals_u8: Vec<u8> = vec![0, 1, 127, 255];
        let a = MlxArray::from_raw_data(
            vals_u8.as_ptr(),
            std::mem::size_of_val(&vals_u8[..]),
            &[4],
            MlxDtype::Uint8,
        );
        assert_eq!(a.dtype(), MlxDtype::Uint8);
        let f = astype(&a, MlxDtype::Float32, None);
        eval(&[&f]);
        assert_eq!(f.data_f32(), &[0.0, 1.0, 127.0, 255.0]);

        // -- uint16 --
        let vals_u16: Vec<u16> = vec![0, 1, 1000, 65535];
        let a = MlxArray::from_raw_data(
            vals_u16.as_ptr() as *const u8,
            std::mem::size_of_val(&vals_u16[..]),
            &[4],
            MlxDtype::Uint16,
        );
        let f = astype(&a, MlxDtype::Float32, None);
        eval(&[&f]);
        assert_eq!(f.data_f32(), &[0.0, 1.0, 1000.0, 65535.0]);

        // -- uint32 --
        let vals_u32: Vec<u32> = vec![0, 1, 100_000, 4_000_000_000];
        let a = MlxArray::from_raw_data(
            vals_u32.as_ptr() as *const u8,
            std::mem::size_of_val(&vals_u32[..]),
            &[4],
            MlxDtype::Uint32,
        );
        let f = astype(&a, MlxDtype::Float32, None);
        eval(&[&f]);
        assert_eq!(f.data_f32(), &[0.0, 1.0, 100_000.0, 4_000_000_000.0]);

        // -- uint64 --
        let vals_u64: Vec<u64> = vec![0, 1, 100_000, 4_000_000_000];
        let a = MlxArray::from_raw_data(
            vals_u64.as_ptr() as *const u8,
            std::mem::size_of_val(&vals_u64[..]),
            &[4],
            MlxDtype::Uint64,
        );
        let f = astype(&a, MlxDtype::Float32, None);
        eval(&[&f]);
        assert_eq!(f.data_f32(), &[0.0, 1.0, 100_000.0, 4_000_000_000.0]);

        // -- int8 --
        let vals_i8: Vec<i8> = vec![-128, -1, 0, 127];
        let a = MlxArray::from_raw_data(
            vals_i8.as_ptr() as *const u8,
            std::mem::size_of_val(&vals_i8[..]),
            &[4],
            MlxDtype::Int8,
        );
        let f = astype(&a, MlxDtype::Float32, None);
        eval(&[&f]);
        assert_eq!(f.data_f32(), &[-128.0, -1.0, 0.0, 127.0]);

        // -- int16 --
        let vals_i16: Vec<i16> = vec![-32768, -1, 0, 32767];
        let a = MlxArray::from_raw_data(
            vals_i16.as_ptr() as *const u8,
            std::mem::size_of_val(&vals_i16[..]),
            &[4],
            MlxDtype::Int16,
        );
        let f = astype(&a, MlxDtype::Float32, None);
        eval(&[&f]);
        assert_eq!(f.data_f32(), &[-32768.0, -1.0, 0.0, 32767.0]);

        // -- int32 --
        let vals_i32: Vec<i32> = vec![-1_000_000, -1, 0, 1_000_000];
        let a = MlxArray::from_raw_data(
            vals_i32.as_ptr() as *const u8,
            std::mem::size_of_val(&vals_i32[..]),
            &[4],
            MlxDtype::Int32,
        );
        let f = astype(&a, MlxDtype::Float32, None);
        eval(&[&f]);
        assert_eq!(f.data_f32(), &[-1_000_000.0, -1.0, 0.0, 1_000_000.0]);

        // -- int64 --
        let vals_i64: Vec<i64> = vec![-1_000_000, -1, 0, 1_000_000];
        let a = MlxArray::from_raw_data(
            vals_i64.as_ptr() as *const u8,
            std::mem::size_of_val(&vals_i64[..]),
            &[4],
            MlxDtype::Int64,
        );
        let f = astype(&a, MlxDtype::Float32, None);
        eval(&[&f]);
        assert_eq!(f.data_f32(), &[-1_000_000.0, -1.0, 0.0, 1_000_000.0]);
    }

    #[test]
    fn roundtrip_float_dtypes_via_astype_f32() {
        // -- float32 --
        let vals_f32: Vec<f32> = vec![-2.5, 0.0, 1.0, 2.75];
        let a = MlxArray::from_raw_data(
            vals_f32.as_ptr() as *const u8,
            std::mem::size_of_val(&vals_f32[..]),
            &[4],
            MlxDtype::Float32,
        );
        eval(&[&a]);
        assert_eq!(a.data_f32(), &[-2.5, 0.0, 1.0, 2.75]);

        // -- float64 --
        // MLX Metal does not support float64 dispatch; just verify array
        // creation and metadata round-trip.
        let vals_f64: Vec<f64> = vec![-2.5, 0.0, 1.0, 2.75];
        let a = MlxArray::from_raw_data(
            vals_f64.as_ptr() as *const u8,
            std::mem::size_of_val(&vals_f64[..]),
            &[4],
            MlxDtype::Float64,
        );
        assert_eq!(a.dtype(), MlxDtype::Float64);
        assert_eq!(a.shape(), vec![4]);
        assert_eq!(a.nbytes(), 32);
    }

    #[test]
    fn roundtrip_bool_dtype() {
        let vals_bool: Vec<u8> = vec![0, 1, 1, 0];
        let a = MlxArray::from_raw_data(vals_bool.as_ptr(), vals_bool.len(), &[4], MlxDtype::Bool);
        assert_eq!(a.dtype(), MlxDtype::Bool);
        let f = astype(&a, MlxDtype::Float32, None);
        eval(&[&f]);
        assert_eq!(f.data_f32(), &[0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn managed_payload_deleter_fires_on_drop() {
        let counter: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));

        unsafe extern "C" fn test_deleter(payload: *mut std::ffi::c_void) {
            let arc = unsafe { Arc::from_raw(payload as *const AtomicUsize) };
            arc.fetch_add(1, Ordering::SeqCst);
        }

        {
            let counter_clone = counter.clone();
            let payload_ptr = Arc::into_raw(counter_clone) as *mut std::ffi::c_void;
            let vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let _arr = unsafe {
                MlxArray::from_managed_data(
                    vals.as_ptr() as *const u8,
                    std::mem::size_of_val(&vals[..]),
                    &[4],
                    MlxDtype::Float32,
                    payload_ptr,
                    test_deleter,
                )
            };
            // Evaluate so MLX actually materializes the array and can release the buffer.
            eval(&[&_arr]);
            // Array drops here.
        }

        // The deleter may fire lazily; force MLX to flush.
        crate::transforms::clear_cache();
        assert!(
            counter.load(Ordering::SeqCst) >= 1,
            "managed payload deleter must fire when array is released"
        );
    }

    #[test]
    fn array_metadata_matches_input() {
        let vals: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let a = MlxArray::from_raw_data(
            vals.as_ptr() as *const u8,
            std::mem::size_of_val(&vals[..]),
            &[2, 3, 4],
            MlxDtype::Float32,
        );
        assert_eq!(a.shape(), vec![2, 3, 4]);
        assert_eq!(a.ndim(), 3);
        assert_eq!(a.dtype(), MlxDtype::Float32);
        assert_eq!(a.nbytes(), 24 * 4);
    }

    #[test]
    fn scalar_array_has_empty_shape() {
        let s = MlxArray::from_f32(42.0);
        eval(&[&s]);
        assert_eq!(s.shape(), Vec::<i32>::new());
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.data_f32(), &[42.0]);
    }

    #[test]
    fn clone_produces_independent_copy() {
        let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);
        let b = a.clone();
        eval(&[&a, &b]);
        assert_eq!(a.data_f32(), b.data_f32());
    }
}
