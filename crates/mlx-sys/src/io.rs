use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;
use std::ptr;

use crate::array::MlxArray;
use crate::ffi;
use crate::stream::MlxStream;

/// Load all tensors from a safetensors file. Returns a map of name → array.
///
/// Arrays are NOT yet evaluated; call `eval` on the ones you need.
/// Loading defaults to MLX's CPU stream because this path is file I/O and host
/// decode work; callers can pass an explicit stream when they need different
/// placement semantics.
pub fn load_safetensors(
    path: &Path,
    s: Option<&MlxStream>,
) -> Result<HashMap<String, MlxArray>, String> {
    let path_str = path.to_string_lossy();
    let c_path = CString::new(path_str.as_ref()).expect("path must not contain null bytes");

    let stream = s
        .map(|s| s.inner)
        .unwrap_or_else(|| unsafe { ffi::mlx_default_cpu_stream_new() });

    unsafe {
        let mut map = ffi::mlx_map_string_to_array_new();
        let mut meta = ffi::mlx_map_string_to_string {
            ctx: ptr::null_mut(),
        };

        let rc = ffi::mlx_load_safetensors(&mut map, &mut meta, c_path.as_ptr(), stream);
        if rc != 0 {
            ffi::mlx_map_string_to_array_free(map);
            if !meta.ctx.is_null() {
                ffi::mlx_map_string_to_string_free(meta);
            }
            return Err(format!(
                "failed to load safetensors {}: error code {}",
                path.display(),
                rc
            ));
        }
        if !meta.ctx.is_null() {
            ffi::mlx_map_string_to_string_free(meta);
        }

        // Iterate the map into a Rust HashMap.
        let mut result = HashMap::new();
        let it = ffi::mlx_map_string_to_array_iterator_new(map);
        loop {
            let mut key: *const std::os::raw::c_char = ptr::null();
            let mut val = MlxArray::empty();
            let rc = ffi::mlx_map_string_to_array_iterator_next(&mut key, &mut val.inner, it);
            if rc != 0 || key.is_null() {
                break;
            }
            let name = std::ffi::CStr::from_ptr(key).to_string_lossy().into_owned();
            result.insert(name, val);
        }
        ffi::mlx_map_string_to_array_iterator_free(it);
        ffi::mlx_map_string_to_array_free(map);
        Ok(result)
    }
}
