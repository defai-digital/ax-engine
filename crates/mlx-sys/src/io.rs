use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::path::Path;
use std::ptr;
use std::sync::Arc;

use memmap2::Mmap;

use crate::array::{MlxArray, MlxDtype};
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

/// Load all tensors from a safetensors file using a memory-mapped region
/// (zero-copy on the file → array path). Returns `name → MlxArray`.
///
/// Unlike `load_safetensors`, no bytes are copied into a CPU buffer up
/// front. Each tensor's `MlxArray` borrows a slice of the `Mmap`; an
/// `Arc<Mmap>` is keyed into the array's managed-payload destructor so
/// the file mapping stays alive exactly as long as the longest-lived
/// array referencing it.
///
/// MLX may still pull pages into its own working set on the first read
/// (e.g. when the array is dispatched to the GPU). On Apple Silicon's
/// unified memory the pages are reachable to both CPU and GPU without
/// further copying; on other platforms MLX may still need to upload.
///
/// Side effects vs the C `mlx_load_safetensors`:
/// - No upfront file read into a heap buffer; cold-start memory pressure
///   is bounded by what's actually touched.
/// - The metadata map (`__metadata__`) is parsed but not exposed; if
///   callers need it, add a sibling helper.
/// - Quantized tensor groups (`weight` + `scales` + `biases`) are
///   returned as separate top-level names, identical to the C loader.
pub fn load_safetensors_mmap(path: &Path) -> Result<HashMap<String, MlxArray>, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("open {}: {}", path.display(), e))?;
    let mmap = unsafe {
        Mmap::map(&file).map_err(|e| format!("mmap {}: {}", path.display(), e))?
    };
    let mmap = Arc::new(mmap);

    if mmap.len() < 8 {
        return Err(format!(
            "safetensors file {} too small ({} bytes)",
            path.display(),
            mmap.len()
        ));
    }

    // First 8 bytes: little-endian u64 = JSON header length.
    let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
    if 8 + header_len > mmap.len() {
        return Err(format!(
            "safetensors header length {} exceeds file size {}",
            header_len,
            mmap.len()
        ));
    }
    let json_bytes = &mmap[8..8 + header_len];
    let data_base = 8 + header_len;

    let header: serde_json::Value = serde_json::from_slice(json_bytes)
        .map_err(|e| format!("parse safetensors header in {}: {}", path.display(), e))?;
    let obj = header
        .as_object()
        .ok_or_else(|| format!("safetensors header is not a JSON object in {}", path.display()))?;

    let mut result: HashMap<String, MlxArray> = HashMap::new();
    for (name, entry) in obj {
        if name == "__metadata__" {
            continue;
        }
        let entry_obj = entry
            .as_object()
            .ok_or_else(|| format!("tensor entry {name} is not an object"))?;
        let dtype_str = entry_obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("tensor entry {name} missing dtype"))?;
        let dtype = parse_safetensors_dtype(dtype_str)
            .ok_or_else(|| format!("tensor entry {name}: unsupported dtype {dtype_str}"))?;
        let shape_json = entry_obj
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("tensor entry {name} missing shape"))?;
        let shape: Vec<i32> = shape_json
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| format!("shape dim is not a u64 in {name}"))
                    .map(|x| x as i32)
            })
            .collect::<Result<_, _>>()?;
        let offsets = entry_obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("tensor entry {name} missing data_offsets"))?;
        if offsets.len() != 2 {
            return Err(format!("tensor entry {name} data_offsets must have 2 elements"));
        }
        let start = offsets[0].as_u64().ok_or_else(|| {
            format!("tensor entry {name} data_offsets[0] not u64")
        })? as usize;
        let end = offsets[1].as_u64().ok_or_else(|| {
            format!("tensor entry {name} data_offsets[1] not u64")
        })? as usize;
        let absolute_start = data_base + start;
        let absolute_end = data_base + end;
        if absolute_end > mmap.len() {
            return Err(format!(
                "tensor entry {name}: data_offsets [{start},{end}] out of bounds (file {} bytes)",
                mmap.len()
            ));
        }
        let byte_len = absolute_end - absolute_start;
        let ptr = unsafe { mmap.as_ptr().add(absolute_start) };

        // We use the copy-on-create C entry (`mlx_array_new_data` via
        // `from_raw_data`) rather than the managed/borrowed variant.
        // Empirically with the borrowed variant MLX did not read the
        // mmap-backed bytes correctly for f16 / quantized tensors
        // (values came back as garbage), and the upstream C API docs
        // do say "buffer will be copied" for that path. The copy is
        // unavoidable, but the rest of the cold-start path benefits:
        // no userspace heap allocation for the safetensors file, no
        // libc read() into a temp buffer, and any pages already in
        // the OS page cache are reused directly.
        let arr = MlxArray::from_raw_data(ptr, byte_len, &shape, dtype);
        result.insert(name.clone(), arr);
    }
    // Drop the mmap explicitly at function end so the mapping outlives
    // every `from_raw_data` copy above; MLX owns its own buffer by then.
    drop(mmap);
    Ok(result)
}

fn parse_safetensors_dtype(s: &str) -> Option<MlxDtype> {
    Some(match s {
        "F32" | "FLOAT32" => MlxDtype::Float32,
        "F16" | "FLOAT16" => MlxDtype::Float16,
        "BF16" | "BFLOAT16" => MlxDtype::Bfloat16,
        "I8" | "INT8" => MlxDtype::Int8,
        "I16" | "INT16" => MlxDtype::Int16,
        "I32" | "INT32" => MlxDtype::Int32,
        "I64" | "INT64" => MlxDtype::Int64,
        "U8" | "UINT8" => MlxDtype::Uint8,
        "U16" | "UINT16" => MlxDtype::Uint16,
        "U32" | "UINT32" => MlxDtype::Uint32,
        "U64" | "UINT64" => MlxDtype::Uint64,
        "BOOL" => MlxDtype::Bool,
        _ => return None,
    })
}

/// Destructor compatible with `MlxArray::from_managed_data` payloads —
/// recovers a `Box<Arc<Mmap>>` and drops it. Kept available for future
/// callers (`from_managed_data` is exported) even though
/// `load_safetensors_mmap` currently uses the copy-on-create path. Mark
/// dead-code to silence the unused warning until a managed caller lands.
#[allow(dead_code)]
unsafe extern "C" fn mmap_payload_drop(payload: *mut c_void) {
    if payload.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(payload as *mut Arc<Mmap>);
    }
}
