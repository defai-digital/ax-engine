use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;
use std::ptr;
use std::sync::Arc;

use memmap2::{Advice, Mmap};

use crate::array::{MlxArray, MlxDtype};
use crate::error::{last_error_message, prepare_error_capture};
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

    // When no stream is provided, create a CPU stream RAII wrapper so the
    // wrapper object is freed on drop.  Previously `mlx_default_cpu_stream_new()`
    // was called and stored in a raw `ffi::mlx_stream` local — that raw handle was
    // never passed to `mlx_stream_free`, leaking the wrapper on every call.
    let _cpu_stream_guard;
    let stream = if let Some(s) = s {
        s.inner
    } else {
        _cpu_stream_guard = MlxStream::default_cpu();
        _cpu_stream_guard.inner
    };

    unsafe {
        let mut map = ffi::mlx_map_string_to_array_new();
        let mut meta = ffi::mlx_map_string_to_string {
            ctx: ptr::null_mut(),
        };

        prepare_error_capture();
        let rc = ffi::mlx_load_safetensors(&mut map, &mut meta, c_path.as_ptr(), stream);
        if rc != 0 {
            ffi::mlx_map_string_to_array_free(map);
            if !meta.ctx.is_null() {
                ffi::mlx_map_string_to_string_free(meta);
            }
            return Err(format!(
                "failed to load safetensors {}: {}",
                path.display(),
                last_error_message("mlx_load_safetensors")
            ));
        }
        if !meta.ctx.is_null() {
            ffi::mlx_map_string_to_string_free(meta);
        }

        // Iterate the map into a Rust HashMap. The C iterator returns 0 when it
        // writes an entry, 1 at end-of-iteration, and 2 when an MLX exception is
        // caught while materializing the value. Treat 2 as a genuine load
        // failure rather than reading it as a short map.
        let mut result = HashMap::new();
        let it = ffi::mlx_map_string_to_array_iterator_new(map);
        loop {
            let mut key: *const std::os::raw::c_char = ptr::null();
            let mut val = MlxArray::empty();
            let rc = ffi::mlx_map_string_to_array_iterator_next(&mut key, &mut val.inner, it);
            if rc == 0 && !key.is_null() {
                let name = std::ffi::CStr::from_ptr(key).to_string_lossy().into_owned();
                result.insert(name, val);
                continue;
            }
            if rc <= 1 {
                // 1 (or 0 with a null key) signals normal end-of-iteration.
                break;
            }
            // rc > 1: a genuine MLX error surfaced mid-iteration.
            let message = last_error_message("mlx_map_string_to_array_iterator_next");
            ffi::mlx_map_string_to_array_iterator_free(it);
            ffi::mlx_map_string_to_array_free(map);
            return Err(format!(
                "failed to load safetensors {}: {}",
                path.display(),
                message
            ));
        }
        ffi::mlx_map_string_to_array_iterator_free(it);
        ffi::mlx_map_string_to_array_free(map);
        Ok(result)
    }
}

/// Load all tensors from a memory-mapped safetensors file into MLX arrays.
/// Returns `name → MlxArray`.
///
/// Unlike `load_safetensors`, no bytes are copied into a CPU buffer up
/// front. Each tensor is created from a slice of the `Mmap`; MLX copies
/// that slice into its own array storage before this function drops the
/// mapping.
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
    load_safetensors_mmap_filtered(path, None)
}

/// Name-based selection for [`load_safetensors_filtered`].
pub enum SafetensorsNameFilter<'a> {
    /// Materialize only tensors whose names are in the set.
    Keep(&'a std::collections::HashSet<String>),
    /// Materialize every tensor except the names in the set.
    Exclude(&'a std::collections::HashSet<String>),
}

/// Load a subset of the tensors in a safetensors file without materializing
/// the rest. Uses the same memory-mapped path as [`load_safetensors_mmap`];
/// excluded tensors are never copied and never wired into MLX storage.
pub fn load_safetensors_filtered(
    path: &Path,
    filter: SafetensorsNameFilter<'_>,
) -> Result<HashMap<String, MlxArray>, String> {
    load_safetensors_mmap_filtered(path, Some(filter))
}

fn load_safetensors_mmap_filtered(
    path: &Path,
    filter: Option<SafetensorsNameFilter<'_>>,
) -> Result<HashMap<String, MlxArray>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
    load_safetensors_mmap_open_file(path, &file, filter, None)
}

/// Materialize selected tensors from the same open file used for admission.
/// The path identifies the file in errors and checks that its directory
/// entry still agrees with the descriptor after loading.
pub fn load_safetensors_filtered_from_file(
    path: &Path,
    file: &std::fs::File,
    filter: SafetensorsNameFilter<'_>,
) -> Result<HashMap<String, MlxArray>, String> {
    load_safetensors_mmap_open_file(path, file, Some(filter), None)
}

/// Counts kernel hints for the experimental guarded mmap reader.
#[derive(Debug, Default)]
pub struct SafetensorsReadAhead {
    pub attempts: usize,
    pub accepted: usize,
    pub accepted_bytes: usize,
}

/// Materialize selected tensors with bounded lookahead for each next tensor.
/// The hint affects file pages only. The guarded MLX-owned copy is unchanged.
pub fn load_safetensors_filtered_read_ahead(
    path: &Path,
    filter: SafetensorsNameFilter<'_>,
) -> Result<(HashMap<String, MlxArray>, SafetensorsReadAhead), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
    let mut stats = SafetensorsReadAhead::default();
    let arrays = load_safetensors_mmap_open_file(path, &file, Some(filter), Some(&mut stats))?;
    Ok((arrays, stats))
}

const MIN_READ_AHEAD_BYTES: usize = 256 * 1024;
const MAX_READ_AHEAD_BYTES: usize = 64 * 1024 * 1024;

fn read_ahead_range(start: usize, bytes: usize, page_size: usize) -> Option<(usize, usize)> {
    // Advise only complete pages inside the selected prefix. A page shared
    // with an adjacent tensor should never be part of the explicit hint.
    let page_mask = page_size - 1;
    let aligned_start = start.checked_add(page_mask)? & !page_mask;
    let aligned_end = start.checked_add(bytes.min(MAX_READ_AHEAD_BYTES))? & !page_mask;
    let len = aligned_end.checked_sub(aligned_start)?;
    (len >= MIN_READ_AHEAD_BYTES).then_some((aligned_start, len))
}

fn advise_tensor(
    mmap: &Mmap,
    tensor: &MappedTensor<'_>,
    page_size: usize,
    stats: &mut SafetensorsReadAhead,
) {
    let Some((offset, len)) = read_ahead_range(tensor.absolute_start, tensor.byte_len, page_size)
    else {
        return;
    };
    stats.attempts += 1;
    if mmap.advise_range(Advice::WillNeed, offset, len).is_ok() {
        stats.accepted += 1;
        stats.accepted_bytes += len;
    }
}

fn load_safetensors_mmap_open_file(
    path: &Path,
    file: &std::fs::File,
    filter: Option<SafetensorsNameFilter<'_>>,
    mut read_ahead: Option<&mut SafetensorsReadAhead>,
) -> Result<HashMap<String, MlxArray>, String> {
    // NAS/SMB-backed caches can replace or truncate a shard while it is
    // mapped. Stat before and after the copies and fail closed on any drift,
    // so a mutated file is refused instead of yielding torn tensors. This
    // cannot catch a SIGBUS raised by a truncation during the copy itself.
    let stat_before = file
        .metadata()
        .map_err(|e| format!("stat {}: {}", path.display(), e))?;
    let mmap = unsafe { Mmap::map(file).map_err(|e| format!("mmap {}: {}", path.display(), e))? };
    let mmap = Arc::new(mmap);

    if mmap.len() < 8 {
        return Err(format!(
            "safetensors file {} too small ({} bytes)",
            path.display(),
            mmap.len()
        ));
    }

    // First 8 bytes: little-endian u64 = JSON header length.
    let header_len_u64 = u64::from_le_bytes(mmap[..8].try_into().unwrap());
    let header_len = usize::try_from(header_len_u64)
        .map_err(|_| format!("safetensors header length {header_len_u64} does not fit usize"))?;
    let header_end = 8usize.checked_add(header_len).ok_or_else(|| {
        format!("safetensors header length {header_len} overflows when added to prefix")
    })?;
    if header_end > mmap.len() {
        return Err(format!(
            "safetensors header length {} exceeds file size {}",
            header_len,
            mmap.len()
        ));
    }
    let json_bytes = &mmap[8..header_end];
    let data_base = header_end;

    let header: serde_json::Value = serde_json::from_slice(json_bytes)
        .map_err(|e| format!("parse safetensors header in {}: {}", path.display(), e))?;
    let obj = header.as_object().ok_or_else(|| {
        format!(
            "safetensors header is not a JSON object in {}",
            path.display()
        )
    })?;

    let mut selected = obj
        .iter()
        .filter(|(name, _)| {
            name.as_str() != "__metadata__"
                && filter.as_ref().is_none_or(|filter| match filter {
                    SafetensorsNameFilter::Keep(names) => names.contains(name.as_str()),
                    SafetensorsNameFilter::Exclude(names) => !names.contains(name.as_str()),
                })
        })
        .map(|(name, entry)| parse_mapped_tensor(name, entry, data_base, mmap.len()))
        .peekable();
    let page_size = read_ahead.as_ref().and_then(|_| {
        // SAFETY: sysconf reads the machine page size without dereferencing a pointer.
        usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
            .ok()
            .filter(|value| value.is_power_of_two())
    });
    let mut result = HashMap::new();
    while let Some(tensor) = selected.next() {
        let tensor = tensor?;
        if let Some(stats) = read_ahead.as_deref_mut()
            && let Some(page_size) = page_size
            && let Some(Ok(next)) = selected.peek()
        {
            advise_tensor(&mmap, next, page_size, stats);
        }
        let ptr = unsafe { mmap.as_ptr().add(tensor.absolute_start) };

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
        let arr = MlxArray::from_raw_data(ptr, tensor.byte_len, &tensor.shape, tensor.dtype);
        result.insert(tensor.name.to_owned(), arr);
    }
    // Drop the mmap explicitly at function end so the mapping outlives
    // every `from_raw_data` copy above; MLX owns its own buffer by then.
    drop(mmap);
    ensure_unchanged_since(path, &stat_before)?;
    Ok(result)
}

struct MappedTensor<'a> {
    name: &'a str,
    shape: Vec<i32>,
    dtype: MlxDtype,
    absolute_start: usize,
    byte_len: usize,
}

fn parse_mapped_tensor<'a>(
    name: &'a str,
    entry: &serde_json::Value,
    data_base: usize,
    map_len: usize,
) -> Result<MappedTensor<'a>, String> {
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
    let mut shape: Vec<i32> = Vec::with_capacity(shape_json.len());
    let mut numel: usize = 1;
    for (i, v) in shape_json.iter().enumerate() {
        let d = v
            .as_u64()
            .ok_or_else(|| format!("shape dim is not a u64 in {name}"))?;
        let dim = i32::try_from(d)
            .map_err(|_| format!("tensor entry {name}: shape dim {i}={d} exceeds i32::MAX"))?;
        shape.push(dim);
        numel = numel
            .checked_mul(d as usize)
            .ok_or_else(|| format!("tensor entry {name}: shape {shape_json:?} overflows usize"))?;
    }
    let expected_byte_len = numel
        .checked_mul(dtype.size_bytes())
        .ok_or_else(|| format!("tensor entry {name}: byte length overflows usize"))?;
    let offsets = entry_obj
        .get("data_offsets")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("tensor entry {name} missing data_offsets"))?;
    if offsets.len() != 2 {
        return Err(format!(
            "tensor entry {name} data_offsets must have 2 elements"
        ));
    }
    let start_u64 = offsets[0]
        .as_u64()
        .ok_or_else(|| format!("tensor entry {name} data_offsets[0] not u64"))?;
    let end_u64 = offsets[1]
        .as_u64()
        .ok_or_else(|| format!("tensor entry {name} data_offsets[1] not u64"))?;
    if start_u64 > end_u64 {
        return Err(format!(
            "tensor entry {name}: data_offsets [{start_u64},{end_u64}] has start > end"
        ));
    }
    let start = usize::try_from(start_u64).map_err(|_| {
        format!("tensor entry {name}: data_offsets[0]={start_u64} does not fit usize")
    })?;
    let end = usize::try_from(end_u64).map_err(|_| {
        format!("tensor entry {name}: data_offsets[1]={end_u64} does not fit usize")
    })?;
    let absolute_start = data_base.checked_add(start).ok_or_else(|| {
        format!("tensor entry {name}: data_offsets start {start} overflows relative to data_base")
    })?;
    let absolute_end = data_base.checked_add(end).ok_or_else(|| {
        format!("tensor entry {name}: data_offsets end {end} overflows relative to data_base")
    })?;
    if absolute_end > map_len || absolute_start > map_len {
        return Err(format!(
            "tensor entry {name}: data_offsets [{start},{end}] out of bounds (file {} bytes)",
            map_len
        ));
    }
    let byte_len = absolute_end - absolute_start;
    // A payload shorter than the shape would trip the copy-time assert in
    // `from_raw_data` and abort the process; a longer one would silently
    // load the wrong tensor. Fail closed like the strict row reader.
    if byte_len != expected_byte_len {
        return Err(format!(
            "tensor entry {name}: data_offsets byte length {byte_len} does not match \
             shape {shape:?} dtype {dtype_str} expected {expected_byte_len}"
        ));
    }
    Ok(MappedTensor {
        name,
        shape,
        dtype,
        absolute_start,
        byte_len,
    })
}

/// Compare file metadata used to detect drift during a load.
/// Unix file identity detects replacement, and change time detects rewrites
/// that preserve modification time.
pub fn file_metadata_unchanged(before: &std::fs::Metadata, after: &std::fs::Metadata) -> bool {
    let modified = before.modified().ok();
    if before.len() != after.len() || modified.is_none() || modified != after.modified().ok() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if before.dev() != after.dev()
            || before.ino() != after.ino()
            || before.ctime() != after.ctime()
            || before.ctime_nsec() != after.ctime_nsec()
        {
            return false;
        }
    }
    true
}

/// Fail closed when `path` no longer matches the file opened before mapping.
fn ensure_unchanged_since(path: &Path, before: &std::fs::Metadata) -> Result<(), String> {
    let after = std::fs::metadata(path)
        .map_err(|e| format!("stat after load {}: {}", path.display(), e))?;
    if !file_metadata_unchanged(before, &after) {
        return Err(format!(
            "safetensors file {} changed while mapped (size {} -> {}, mtime {:?} -> {:?}); \
             size, timestamps, or file identity differ; refusing to load possibly-torn tensors",
            path.display(),
            before.len(),
            after.len(),
            before.modified().ok(),
            after.modified().ok()
        ));
    }
    Ok(())
}

/// Map a safetensors dtype string to the dtype used for the loaded array.
///
/// FP8 payloads (`F8_E4M3` blockwise weights, `F8_E8M0` block scales) have
/// no `MlxDtype` of their own; both are exactly 1 byte per element, so they
/// arrive as raw `Uint8` byte containers and the caller must dequantize
/// (e.g. `from_fp8` for E4M3, a `2^(b-127)` LUT for E8M0).
fn parse_safetensors_dtype(s: &str) -> Option<MlxDtype> {
    Some(match s {
        "F32" | "FLOAT32" => MlxDtype::Float32,
        "F16" | "FLOAT16" => MlxDtype::Float16,
        "BF16" | "BFLOAT16" => MlxDtype::Bfloat16,
        "F8_E4M3" | "F8_E8M0" => MlxDtype::Uint8,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::eval;
    use std::collections::HashSet;

    #[test]
    fn filtered_loader_rejects_inconsistent_tensor_geometry_without_panicking() {
        let dir = std::env::temp_dir().join(format!("ax-filtered-invalid-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.safetensors");
        for (dtype, shape) in [
            ("F16", serde_json::json!([16])),
            ("F32", serde_json::json!([2])),
            ("U8", serde_json::json!([4294967296u64])),
            (
                "U8",
                serde_json::json!([2147483647u64, 2147483647u64, 2147483647u64]),
            ),
        ] {
            let header = serde_json::to_vec(&serde_json::json!({"bad": {
                "dtype":dtype, "shape":shape, "data_offsets":[0,16]
            }}))
            .unwrap();
            let mut data = (header.len() as u64).to_le_bytes().to_vec();
            data.extend(header);
            data.extend([0u8; 16]);
            std::fs::write(&path, data).unwrap();
            let keep = HashSet::from(["bad".to_string()]);
            assert!(load_safetensors_filtered(&path, SafetensorsNameFilter::Keep(&keep)).is_err());
            assert!(
                load_safetensors_filtered(&path, SafetensorsNameFilter::Exclude(&keep))
                    .unwrap()
                    .is_empty()
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn drift_check_accepts_unchanged_file_and_rejects_size_or_mtime_change() {
        let dir = std::env::temp_dir().join(format!("ax-drift-check-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shard.bin");
        std::fs::write(&path, [0u8; 16]).unwrap();
        let before = std::fs::metadata(&path).unwrap();
        assert!(ensure_unchanged_since(&path, &before).is_ok());

        // Same size, later mtime.
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_modified(before.modified().unwrap() + std::time::Duration::from_secs(5))
            .unwrap();
        let err = ensure_unchanged_since(&path, &before).unwrap_err();
        assert!(err.contains("changed while mapped"), "{err}");

        // Truncated file.
        file.set_len(8).unwrap();
        let err = ensure_unchanged_since(&path, &before).unwrap_err();
        assert!(err.contains("size 16 -> 8"), "{err}");

        // Replaced by a path that no longer exists.
        drop(file);
        std::fs::remove_file(&path).unwrap();
        let err = ensure_unchanged_since(&path, &before).unwrap_err();
        assert!(err.contains("stat after load"), "{err}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn drift_check_rejects_file_replacement_with_same_size_and_mtime() {
        let dir = std::env::temp_dir().join(format!("ax-drift-replacement-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shard.bin");
        std::fs::write(&path, [0u8; 16]).unwrap();
        let original = std::fs::File::open(&path).unwrap();
        let before = original.metadata().unwrap();
        let replacement = dir.join("replacement.bin");
        std::fs::write(&replacement, [1u8; 16]).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&replacement)
            .unwrap()
            .set_modified(before.modified().unwrap())
            .unwrap();
        std::fs::rename(replacement, &path).unwrap();
        let after = std::fs::metadata(&path).unwrap();
        assert_eq!(before.len(), after.len());
        assert_eq!(before.modified().unwrap(), after.modified().unwrap());
        let result = ensure_unchanged_since(&path, &before);
        std::fs::remove_dir_all(dir).unwrap();
        assert!(result.is_err());
    }

    /// Build a minimal safetensors file in a temp dir, load it back, and
    /// verify the tensor data round-trips correctly.
    #[test]
    fn safetensors_write_and_load_round_trip() {
        let dir = std::env::temp_dir().join("ax_shim_io_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_tensors.safetensors");

        // Two float32 tensors: "a" shape [2,3] and "b" shape [4]
        let a_data: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
        let b_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let a_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(a_data.as_ptr() as *const u8, a_data.len() * 4) };
        let b_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(b_data.as_ptr() as *const u8, b_data.len() * 4) };

        // Safetensors format: 8-byte LE header length, JSON header, then raw data.
        let a_start = 0usize;
        let a_end = a_bytes.len();
        let b_start = a_end;
        let b_end = b_start + b_bytes.len();

        let header = serde_json::json!({
            "a": {
                "dtype": "F32",
                "shape": [2, 3],
                "data_offsets": [a_start, a_end]
            },
            "b": {
                "dtype": "F32",
                "shape": [4],
                "data_offsets": [b_start, b_end]
            }
        });
        let header_str = serde_json::to_string(&header).unwrap();
        let header_bytes = header_str.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut file = std::fs::File::create(&path).unwrap();
        std::io::Write::write_all(&mut file, &header_len.to_le_bytes()).unwrap();
        std::io::Write::write_all(&mut file, header_bytes).unwrap();
        std::io::Write::write_all(&mut file, a_bytes).unwrap();
        std::io::Write::write_all(&mut file, b_bytes).unwrap();
        drop(file);

        // Load via FFI path
        let tensors = load_safetensors(&path, None).expect("load_safetensors should succeed");
        assert!(
            tensors.contains_key("a"),
            "loaded tensors should contain 'a'"
        );
        assert!(
            tensors.contains_key("b"),
            "loaded tensors should contain 'b'"
        );

        let ta = tensors.get("a").unwrap();
        let tb = tensors.get("b").unwrap();
        assert_eq!(ta.shape(), vec![2, 3]);
        assert_eq!(ta.dtype(), MlxDtype::Float32);
        assert_eq!(tb.shape(), vec![4]);

        eval(&[ta, tb]);
        assert_eq!(ta.data_f32(), &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        assert_eq!(tb.data_f32(), &[10.0, 20.0, 30.0, 40.0]);

        // Load via mmap path
        let mmap_tensors = load_safetensors_mmap(&path).expect("mmap load should succeed");
        let ma = mmap_tensors.get("a").unwrap();
        let mb = mmap_tensors.get("b").unwrap();
        eval(&[ma, mb]);
        assert_eq!(ma.data_f32(), &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        assert_eq!(mb.data_f32(), &[10.0, 20.0, 30.0, 40.0]);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn write_safetensors_header_only(
        path: &std::path::Path,
        header: &serde_json::Value,
        payload: &[u8],
    ) {
        let header_bytes = serde_json::to_vec(header).unwrap();
        let mut file = std::fs::File::create(path).unwrap();
        std::io::Write::write_all(&mut file, &(header_bytes.len() as u64).to_le_bytes()).unwrap();
        std::io::Write::write_all(&mut file, &header_bytes).unwrap();
        std::io::Write::write_all(&mut file, payload).unwrap();
    }

    #[test]
    fn mmap_load_rejects_payloads_that_disagree_with_the_shape() {
        // A short payload would trip the copy-time assert and abort the
        // process; an oversized dim would truncate to a wrong i32 shape.
        let dir = std::env::temp_dir().join("ax_shim_io_shape_mismatch_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let short = dir.join("short.safetensors");
        write_safetensors_header_only(
            &short,
            &serde_json::json!({"t": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 4]}}),
            &[0u8; 4],
        );
        let error = load_safetensors_mmap(&short).expect_err("short payload must fail closed");
        assert!(error.contains("does not match"), "{error}");

        let long = dir.join("long.safetensors");
        write_safetensors_header_only(
            &long,
            &serde_json::json!({"t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 8]}}),
            &[0u8; 8],
        );
        assert!(load_safetensors_mmap(&long).is_err());

        let huge = dir.join("huge.safetensors");
        write_safetensors_header_only(
            &huge,
            &serde_json::json!({"t": {"dtype": "U8", "shape": [4294967297u64, 2], "data_offsets": [0, 2]}}),
            &[0u8; 2],
        );
        let error = load_safetensors_mmap(&huge).expect_err("oversized dim must fail closed");
        assert!(error.contains("exceeds i32::MAX"), "{error}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_ahead_preserves_filtered_tensor_data_and_validation() {
        let dir = std::env::temp_dir().join(format!("ax-mmap-read-ahead-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("guarded.safetensors");
        let selected_bytes = 2 * MIN_READ_AHEAD_BYTES;
        let count = selected_bytes / 4;
        let header = serde_json::json!({
            "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "excluded": {"dtype": "U32", "shape": [count], "data_offsets": [4, 4 + selected_bytes]},
            "z": {"dtype": "U32", "shape": [count], "data_offsets": [4 + selected_bytes, 4 + 2 * selected_bytes]},
        });
        let mut bytes = 1.5_f32.to_le_bytes().to_vec();
        bytes.resize(4 + selected_bytes, 0);
        bytes.extend((0..count as u32).flat_map(u32::to_le_bytes));
        write_safetensors_header_only(&path, &header, &bytes);
        let names = HashSet::from(["a".into(), "z".into()]);
        let load =
            || load_safetensors_filtered_read_ahead(&path, SafetensorsNameFilter::Keep(&names));
        let normal = load_safetensors_filtered(&path, SafetensorsNameFilter::Keep(&names)).unwrap();
        let (advised, stats) = load().unwrap();
        assert_eq!(stats.attempts, 1);
        assert!(stats.accepted <= 1);
        if stats.accepted == 1 {
            assert!((MIN_READ_AHEAD_BYTES..=selected_bytes).contains(&stats.accepted_bytes));
        }
        assert_eq!(advised.len(), 2);
        assert!(!advised.contains_key("excluded"));
        eval(&[&normal["a"], &normal["z"], &advised["a"], &advised["z"]]);
        assert_eq!(advised["a"].data_f32(), normal["a"].data_f32());
        assert_eq!(advised["z"].data_u32(), normal["z"].data_u32());
        assert_eq!(advised["z"].data_u32()[count - 1], count as u32 - 1);

        let mut bad = header;
        bad["z"]["shape"] = serde_json::json!([count + 1]);
        write_safetensors_header_only(&path, &bad, &bytes);
        let error = load().unwrap_err();
        assert!(error.contains("does not match"), "{error}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn read_ahead_range_stays_inside_selected_pages_and_cap() {
        let page = 16 * 1024;
        assert_eq!(
            read_ahead_range(page + 1, 2 * MAX_READ_AHEAD_BYTES, page),
            Some((2 * page, MAX_READ_AHEAD_BYTES - page))
        );
        assert_eq!(
            read_ahead_range(page, MAX_READ_AHEAD_BYTES, page),
            Some((page, MAX_READ_AHEAD_BYTES))
        );
        assert_eq!(read_ahead_range(1, MIN_READ_AHEAD_BYTES, page), None);
        assert_eq!(read_ahead_range(1, 0, page), None);
        assert_eq!(
            read_ahead_range(usize::MAX - 1, MAX_READ_AHEAD_BYTES, page),
            None
        );
    }

    #[test]
    #[ignore = "requires a pinned Flash Next MXFP4 embedding shard"]
    fn flash_next_read_ahead_matches_real_embedding_bytes() {
        let path = std::env::var_os("AX_FLASH_NEXT_READ_AHEAD_TEST_SHARD").unwrap();
        let path = Path::new(&path);
        let names = ["biases", "scales", "weight"]
            .map(|suffix| format!("language_model.model.embed_tokens.{suffix}"));
        let selected = HashSet::from(names.clone());
        let normal =
            load_safetensors_filtered(path, SafetensorsNameFilter::Keep(&selected)).unwrap();
        let (advised, hints) =
            load_safetensors_filtered_read_ahead(path, SafetensorsNameFilter::Keep(&selected))
                .unwrap();
        assert_eq!(normal.len(), names.len());
        assert_eq!(advised.len(), names.len());
        assert_eq!(hints.attempts, 2);
        let mut compared_bytes = 0;
        for name in names {
            let original = &normal[&name];
            let candidate = &advised[&name];
            assert_eq!(original.dtype(), candidate.dtype());
            assert_eq!(original.shape(), candidate.shape());
            eval(&[original, candidate]);
            let size = original.nbytes();
            // SAFETY: Evaluated contiguous arrays stay alive for both byte views.
            let original = unsafe { std::slice::from_raw_parts(original.data_raw(), size) };
            let candidate = unsafe { std::slice::from_raw_parts(candidate.data_raw(), size) };
            assert!(original == candidate, "changed bytes in {name}");
            compared_bytes += size;
        }
        println!("compared_bytes={compared_bytes} hints={hints:?}");
    }

    #[cfg(unix)]
    #[test]
    fn filtered_load_from_file_does_not_follow_a_replaced_directory() {
        let root = std::env::temp_dir().join(format!("ax-descriptor-swap-{}", std::process::id()));
        let current = root.join("current");
        let next = root.join("next");
        let previous = root.join("previous");
        std::fs::create_dir_all(&current).unwrap();
        std::fs::create_dir(&next).unwrap();
        let path = current.join("weights.safetensors");
        let next_path = next.join("weights.safetensors");
        let header =
            serde_json::json!({"t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}});
        write_safetensors_header_only(&path, &header, &1.0_f32.to_le_bytes());
        write_safetensors_header_only(&next_path, &header, &2.0_f32.to_le_bytes());
        let original = std::fs::File::open(&path).unwrap();
        let original_metadata = original.metadata().unwrap();
        let keep = HashSet::from(["t".to_string()]);
        let read_path = || load_safetensors_filtered(&path, SafetensorsNameFilter::Keep(&keep));
        let read_descriptor = || {
            load_safetensors_filtered_from_file(
                &path,
                &original,
                SafetensorsNameFilter::Keep(&keep),
            )
        };

        std::fs::rename(&current, &previous).unwrap();
        std::fs::rename(&next, &current).unwrap();
        let changed = read_path().unwrap();
        eval(&[&changed["t"]]);
        assert_eq!(changed["t"].data_f32(), &[2.0]);
        let error = read_descriptor().unwrap_err();
        assert!(error.contains("changed while mapped"), "{error}");

        std::fs::rename(&current, &next).unwrap();
        std::fs::rename(&previous, &current).unwrap();
        assert!(file_metadata_unchanged(
            &original_metadata,
            &std::fs::metadata(&path).unwrap()
        ));
        let restored = read_descriptor().unwrap();
        eval(&[&restored["t"]]);
        assert_eq!(restored["t"].data_f32(), &[1.0]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn filtered_load_selects_tensors_by_name() {
        let dir = std::env::temp_dir().join("ax_shim_io_filtered_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("filtered.safetensors");

        let a: Vec<f32> = vec![1.0, 2.0];
        let b: Vec<f32> = vec![3.0, 4.0, 5.0, 6.0];
        let mut data: Vec<u8> = Vec::new();
        let a_start = data.len();
        data.extend(a.iter().flat_map(|v| v.to_le_bytes()));
        let a_end = data.len();
        let b_start = data.len();
        data.extend(b.iter().flat_map(|v| v.to_le_bytes()));
        let b_end = data.len();

        let header = serde_json::json!({
            "a": {"dtype": "F32", "shape": [2], "data_offsets": [a_start, a_end]},
            "b": {"dtype": "F32", "shape": [4], "data_offsets": [b_start, b_end]}
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut file = std::fs::File::create(&path).unwrap();
        std::io::Write::write_all(&mut file, &(header_bytes.len() as u64).to_le_bytes()).unwrap();
        std::io::Write::write_all(&mut file, &header_bytes).unwrap();
        std::io::Write::write_all(&mut file, &data).unwrap();
        drop(file);

        let mut keep = std::collections::HashSet::new();
        keep.insert("a".to_string());
        let kept = load_safetensors_filtered(&path, SafetensorsNameFilter::Keep(&keep))
            .expect("keep-filtered load should succeed");
        assert!(kept.contains_key("a") && !kept.contains_key("b"));
        let ka = kept.get("a").unwrap();
        eval(&[ka]);
        assert_eq!(ka.data_f32(), &[1.0, 2.0]);

        let mut exclude = std::collections::HashSet::new();
        exclude.insert("a".to_string());
        let rest = load_safetensors_filtered(&path, SafetensorsNameFilter::Exclude(&exclude))
            .expect("exclude-filtered load should succeed");
        assert!(rest.contains_key("b") && !rest.contains_key("a"));
        let rb = rest.get("b").unwrap();
        eval(&[rb]);
        assert_eq!(rb.data_f32(), &[3.0, 4.0, 5.0, 6.0]);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
