use std::fmt;
use std::ptr;

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
            let arr = ffi::mlx_array_new_data(
                data.as_ptr() as *const _,
                shape.as_ptr(),
                1,
                ffi::mlx_dtype_::MLX_FLOAT32,
            );
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
            let arr = ffi::mlx_array_new_data(
                data.as_ptr() as *const _,
                shape.as_ptr(),
                1,
                ffi::mlx_dtype_::MLX_FLOAT16,
            );
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
            let arr = ffi::mlx_array_new_data(
                data as *const _,
                shape.as_ptr(),
                shape.len() as i32,
                dtype.to_ffi(),
            );
            Self::from_raw(arr)
        }
    }

    pub fn ndim(&self) -> usize {
        unsafe { ffi::mlx_array_ndim(self.inner) }
    }

    pub fn shape(&self) -> Vec<i32> {
        unsafe {
            let ndim = ffi::mlx_array_ndim(self.inner);
            let ptr = ffi::mlx_array_shape(self.inner);
            if ptr.is_null() || ndim == 0 {
                return vec![];
            }
            std::slice::from_raw_parts(ptr, ndim).to_vec()
        }
    }

    pub fn dtype(&self) -> MlxDtype {
        unsafe { MlxDtype::from_ffi(ffi::mlx_array_dtype(self.inner)) }
    }

    pub fn nbytes(&self) -> usize {
        unsafe { ffi::mlx_array_nbytes(self.inner) }
    }

    /// Read data as f32. Array must have been eval'd first.
    pub fn data_f32(&self) -> &[f32] {
        assert_eq!(
            self.dtype(),
            MlxDtype::Float32,
            "data_f32 requires Float32 dtype"
        );
        unsafe {
            let ptr = ffi::mlx_array_data_float32(self.inner);
            let len = self.nbytes() / self.dtype().size_bytes();
            if ptr.is_null() || len == 0 {
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
            let ptr = ffi::mlx_array_data_uint32(self.inner);
            let len = self.nbytes() / self.dtype().size_bytes();
            if ptr.is_null() || len == 0 {
                return &[];
            }
            std::slice::from_raw_parts(ptr, len)
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
            ffi::mlx_array_set(&mut dst, self.inner);
            Self { inner: dst }
        }
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
        unsafe { ffi::mlx_array_data_uint8(self.inner) }
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
