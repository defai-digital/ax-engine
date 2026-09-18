//! Owned row-granular safetensors reader (no mmap, no whole-table reads).
//!
//! [`SafetensorsRowReader`] keeps an owned [`std::fs::File`] handle (opened-file
//! identity) and serves arbitrary row gathers via positional reads:
//! [`std::os::unix::fs::FileExt::read_exact_at`] on Unix and a
//! [`std::os::windows::fs::FileExt::seek_read`] loop on Windows. Other targets
//! fail explicitly instead of sharing a seek cursor.
//!
//! Open parses only the bounded JSON header (cap [`MAX_HEADER_BYTES`]) and
//! strictly validates tensor entries: supported dtype, positive rank-2
//! dimensions, exact `rows * cols * elem_size` byte length, checked arithmetic,
//! and file bounds. [`SafetensorsRowReader::open`] validates every entry, which
//! rejects real checkpoint files that mix rank-2 tables with unrelated rank-1
//! norms, integer multipliers, or rank-3 experts. Use
//! [`SafetensorsRowReader::open_selected`] to validate only the requested
//! tensors: the header is still read once and unrelated payload is never
//! touched. [`SafetensorsRowReader::open_selected_stacks`] explicitly selects
//! rank-3 expert stacks, preserving their trailing matrix dimensions.
//!
//! Gathers preserve caller order (including duplicates), copy only the
//! requested rows, enforce an immutable per-reader output byte budget checked
//! before any payload I/O, and account payload bytes for test evidence. Each
//! successful gather returns a new [`MlxArray`] with owned storage, so a later
//! gather, reader eviction, or dropping the reader never overwrites previously
//! returned results that feed lazy graphs.
//!
//! Supported dtypes: `F32`, `F16`, `BF16`, `U32` (plus `FLOAT32`, `FLOAT16`,
//! `BFLOAT16`, `UINT32` aliases). All other dtypes fail closed at open. `U32`
//! coverage here is raw row buffers only, not affine-quantization numeric
//! parity (owned by the quantization integration).

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::array::{MlxArray, MlxDtype};

/// Maximum accepted safetensors JSON header length (100,000,000 bytes).
///
/// Matches the MLX safetensors limit; the bound is exclusive, so a header of
/// exactly this length is rejected. [`SafetensorsRowReader::open`] and
/// [`SafetensorsRowReader::open_selected`] reject larger headers before
/// allocating.
pub const MAX_HEADER_BYTES: u64 = 100_000_000;

/// Default cap for a single [`SafetensorsRowReader::gather_rows`] output (256 MiB).
pub const DEFAULT_MAX_GATHER_BYTES: usize = 256 * 1024 * 1024;

/// Validated tensor entry exposed by [`SafetensorsRowReader`].
#[derive(Clone, Debug)]
pub struct RowTensorMeta {
    /// MLX dtype used for gathered rows.
    pub dtype: MlxDtype,
    /// Original safetensors dtype string (e.g. `"F32"`).
    pub dtype_str: String,
    /// Number of rows (shape[0]).
    pub rows: usize,
    /// Elements per axis-zero row (product of the trailing dimensions).
    pub cols: usize,
    /// Original dimensions as `i32` for direct [`MlxArray`] construction.
    pub shape: Vec<i32>,
    /// Bytes per row (`cols * elem_bytes`).
    pub row_bytes: usize,
    /// Bytes per element (4 for F32/U32, 2 for F16/BF16).
    pub elem_bytes: usize,
    /// Absolute file offset of the tensor payload start.
    pub data_start: u64,
    /// Absolute file offset of the tensor payload end (exclusive).
    pub data_end: u64,
}

impl RowTensorMeta {
    /// Total tensor payload bytes (`rows * row_bytes`).
    pub fn byte_len(&self) -> usize {
        self.rows.saturating_mul(self.row_bytes)
    }
}

/// Owned safetensors row reader.
///
/// Keeps the opened [`File`] so gathers observe opened-file identity rather
/// than re-resolving the path. Never mmaps and never reads whole tables: open
/// reads only the 8-byte length prefix plus the bounded JSON header, and each
/// gather reads only its requested rows.
///
/// Concise API: [`Self::open`], [`Self::open_with_budget`], and
/// [`Self::open_selected`] construct the reader with an immutable per-gather
/// byte budget; [`Self::tensor_meta`], [`Self::tensor_names`],
/// [`Self::max_gather_bytes`], [`Self::payload_bytes_read`], and
/// [`Self::gather_rows`] are the only operations.
pub struct SafetensorsRowReader {
    file: File,
    file_len: u64,
    tensors: HashMap<String, RowTensorMeta>,
    max_gather_bytes: usize,
    payload_bytes_read: AtomicU64,
}

impl std::fmt::Debug for SafetensorsRowReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafetensorsRowReader")
            .field("file_len", &self.file_len)
            .field("tensors", &self.tensors)
            .field("max_gather_bytes", &self.max_gather_bytes)
            .field(
                "payload_bytes_read",
                &self.payload_bytes_read.load(Ordering::Relaxed),
            )
            .finish_non_exhaustive()
    }
}

#[cfg(unix)]
fn pread_exact(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

/// Positional read without touching the shared file cursor.
///
/// [`std::os::windows::fs::FileExt::seek_read`] takes `&self` and an explicit
/// offset, so concurrent gathers cannot race, unlike a `try_clone` + `seek` +
/// `read` sequence on a shared cursor. Short reads are retried until the
/// buffer is full or EOF.
#[cfg(windows)]
fn pread_exact(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut read = 0usize;
    while read < buf.len() {
        match file.seek_read(&mut buf[read..], offset + read as u64) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            Ok(n) => read += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Explicit failure where no race-free positional read exists.
#[cfg(not(any(unix, windows)))]
fn pread_exact(_file: &File, _buf: &mut [u8], _offset: u64) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "SafetensorsRowReader requires Unix read_exact_at or Windows seek_read; \
         positional reads are unsupported on this target",
    ))
}

/// Map the supported row-reader dtype strings. All other inputs fail closed.
fn parse_row_dtype(s: &str) -> Option<MlxDtype> {
    Some(match s {
        "F32" | "FLOAT32" => MlxDtype::Float32,
        "F16" | "FLOAT16" => MlxDtype::Float16,
        "BF16" | "BFLOAT16" => MlxDtype::Bfloat16,
        "U32" | "UINT32" => MlxDtype::Uint32,
        _ => return None,
    })
}

/// Strictly validate one header entry with the constructor-selected rank.
///
/// Checks dtype support, positive dimensions within `i32::MAX`, exact
/// `rows * cols * elem_size` byte length with checked arithmetic, ordered
/// offsets, and absolute file bounds. [`SafetensorsRowReader::open`] applies
/// this to every entry; [`SafetensorsRowReader::open_selected`] applies it to
/// requested entries only.
fn validate_entry(
    name: &str,
    entry: &serde_json::Value,
    data_base: u64,
    file_len: u64,
    rank: usize,
) -> Result<RowTensorMeta, String> {
    let entry_obj = entry
        .as_object()
        .ok_or_else(|| format!("tensor entry {name} is not an object"))?;
    let dtype_str = entry_obj
        .get("dtype")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("tensor entry {name} missing dtype"))?;
    let dtype = parse_row_dtype(dtype_str)
        .ok_or_else(|| format!("tensor entry {name}: unsupported dtype {dtype_str}"))?;
    let shape_json = entry_obj
        .get("shape")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("tensor entry {name} missing shape"))?;
    if shape_json.len() != rank {
        return Err(format!(
            "tensor entry {name}: expected rank-{rank} shape, got rank {}",
            shape_json.len()
        ));
    }
    let mut dims_u64 = vec![0u64; rank];
    for (i, v) in shape_json.iter().enumerate() {
        let d = v
            .as_u64()
            .ok_or_else(|| format!("tensor entry {name}: shape dim {i} is not a u64"))?;
        if d == 0 {
            return Err(format!(
                "tensor entry {name}: shape dim {i} must be positive, got 0"
            ));
        }
        if d > i32::MAX as u64 {
            return Err(format!(
                "tensor entry {name}: shape dim {i}={d} exceeds i32::MAX"
            ));
        }
        dims_u64[i] = d;
    }
    let rows = dims_u64[0] as usize;
    let cols = dims_u64[1..].iter().try_fold(1usize, |product, &dim| {
        product
            .checked_mul(dim as usize)
            .ok_or_else(|| format!("tensor entry {name}: trailing dimensions overflow usize"))
    })?;
    let elem_bytes = dtype.size_bytes();
    let numel = rows
        .checked_mul(cols)
        .ok_or_else(|| format!("tensor entry {name}: rows*cols overflows usize"))?;
    let expected = numel
        .checked_mul(elem_bytes)
        .ok_or_else(|| format!("tensor entry {name}: byte length overflows usize"))?;
    let expected_u64 = u64::try_from(expected)
        .map_err(|_| format!("tensor entry {name}: byte length {expected} does not fit u64"))?;
    let offsets = entry_obj
        .get("data_offsets")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("tensor entry {name} missing data_offsets"))?;
    if offsets.len() != 2 {
        return Err(format!(
            "tensor entry {name} data_offsets must have 2 elements"
        ));
    }
    let start = offsets[0]
        .as_u64()
        .ok_or_else(|| format!("tensor entry {name} data_offsets[0] not u64"))?;
    let end = offsets[1]
        .as_u64()
        .ok_or_else(|| format!("tensor entry {name} data_offsets[1] not u64"))?;
    if start > end {
        return Err(format!(
            "tensor entry {name}: data_offsets [{start},{end}] has start > end"
        ));
    }
    let byte_len = end - start;
    if byte_len != expected_u64 {
        return Err(format!(
            "tensor entry {name}: data_offsets byte length {byte_len} does not match \
             shape {dims_u64:?} dtype {dtype_str} expected {expected_u64}"
        ));
    }
    let abs_start = data_base
        .checked_add(start)
        .ok_or_else(|| format!("tensor entry {name}: data_offsets start {start} overflows"))?;
    let abs_end = data_base
        .checked_add(end)
        .ok_or_else(|| format!("tensor entry {name}: data_offsets end {end} overflows"))?;
    if abs_start > file_len || abs_end > file_len {
        return Err(format!(
            "tensor entry {name}: data_offsets [{start},{end}] out of bounds \
             (file {file_len} bytes, data_base {data_base})"
        ));
    }
    let row_bytes = cols
        .checked_mul(elem_bytes)
        .ok_or_else(|| format!("tensor entry {name}: row byte length overflows"))?;
    let shape_i32 = dims_u64.iter().map(|&dim| dim as i32).collect();
    Ok(RowTensorMeta {
        dtype,
        dtype_str: dtype_str.to_string(),
        rows,
        cols,
        shape: shape_i32,
        row_bytes,
        elem_bytes,
        data_start: abs_start,
        data_end: abs_end,
    })
}

impl SafetensorsRowReader {
    /// Open a safetensors file for row gathers with the default byte budget.
    ///
    /// Reads only the 8-byte header length plus the bounded JSON header (cap
    /// [`MAX_HEADER_BYTES`]). Validates every tensor entry strictly; files
    /// that mix rank-2 tables with unrelated ranks or dtypes need
    /// [`Self::open_selected`] instead.
    pub fn open(path: &Path) -> Result<Self, String> {
        Self::open_inner(path, None, DEFAULT_MAX_GATHER_BYTES, 2)
    }

    /// Open with an explicit per-gather output byte budget.
    ///
    /// The budget is immutable for the life of the reader. Like [`Self::open`],
    /// validates every tensor entry strictly.
    pub fn open_with_budget(path: &Path, max_gather_bytes: usize) -> Result<Self, String> {
        Self::open_inner(path, None, max_gather_bytes, 2)
    }

    /// Open only the requested tensors with an explicit byte budget.
    ///
    /// Reads the header once and strictly validates each requested entry while
    /// ignoring all others, so real checkpoint files that mix rank-2 tables
    /// with unrelated rank-1 norms, integer multipliers, or rank-3 experts
    /// open successfully. Duplicate requests and names missing from the header
    /// are rejected. Unrelated payload is never touched, and unselected
    /// tensors are unknown to the returned reader.
    pub fn open_selected(
        path: &Path,
        tensor_names: &[&str],
        max_gather_bytes: usize,
    ) -> Result<Self, String> {
        Self::open_inner(path, Some(tensor_names), max_gather_bytes, 2)
    }

    /// Open selected rank-3 stacks for bounded axis-zero gathers.
    ///
    /// Each row is one complete matrix: `[experts, output, packed_input]`
    /// becomes `[selected_experts, output, packed_input]` after gather. The
    /// output/scales/biases encodings are unchanged; this reader performs no
    /// quantization interpretation. Existing rank-2 constructors stay strict.
    pub fn open_selected_stacks(
        path: &Path,
        tensor_names: &[&str],
        max_gather_bytes: usize,
    ) -> Result<Self, String> {
        Self::open_inner(path, Some(tensor_names), max_gather_bytes, 3)
    }

    fn open_inner(
        path: &Path,
        select: Option<&[&str]>,
        max_gather_bytes: usize,
        rank: usize,
    ) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
        let file_len = file
            .metadata()
            .map_err(|e| format!("metadata {}: {}", path.display(), e))?
            .len();
        if file_len < 8 {
            return Err(format!(
                "safetensors file {} too small ({} bytes)",
                path.display(),
                file_len
            ));
        }
        let mut len_buf = [0u8; 8];
        pread_exact(&file, &mut len_buf, 0)
            .map_err(|e| format!("read header length {}: {}", path.display(), e))?;
        let header_len = u64::from_le_bytes(len_buf);
        if header_len >= MAX_HEADER_BYTES {
            return Err(format!(
                "safetensors header length {header_len} reaches or exceeds cap {MAX_HEADER_BYTES} in {}",
                path.display()
            ));
        }
        let header_len_usize = usize::try_from(header_len)
            .map_err(|_| format!("safetensors header length {header_len} does not fit usize"))?;
        let data_base = 8u64.checked_add(header_len).ok_or_else(|| {
            format!("safetensors header length {header_len} overflows when added to prefix")
        })?;
        if data_base > file_len {
            return Err(format!(
                "safetensors header length {header_len} exceeds file size {file_len} in {}",
                path.display()
            ));
        }
        let mut header_buf = vec![0u8; header_len_usize];
        if header_len_usize > 0 {
            pread_exact(&file, &mut header_buf, 8)
                .map_err(|e| format!("read header {}: {}", path.display(), e))?;
        }
        let header: serde_json::Value = serde_json::from_slice(&header_buf)
            .map_err(|e| format!("parse safetensors header in {}: {}", path.display(), e))?;
        let obj = header.as_object().ok_or_else(|| {
            format!(
                "safetensors header is not a JSON object in {}",
                path.display()
            )
        })?;

        let mut tensors: HashMap<String, RowTensorMeta> = HashMap::new();
        match select {
            None => {
                for (name, entry) in obj {
                    if name == "__metadata__" {
                        continue;
                    }
                    let meta = validate_entry(name, entry, data_base, file_len, rank)?;
                    tensors.insert(name.clone(), meta);
                }
            }
            Some(names) => {
                let mut seen: HashSet<&str> = HashSet::with_capacity(names.len());
                for name in names {
                    if !seen.insert(*name) {
                        return Err(format!(
                            "open_selected {}: duplicate tensor request {name:?}",
                            path.display()
                        ));
                    }
                    let entry = obj.get(*name).ok_or_else(|| {
                        format!(
                            "open_selected {}: tensor {name:?} not found in header",
                            path.display()
                        )
                    })?;
                    let meta = validate_entry(name, entry, data_base, file_len, rank)?;
                    tensors.insert(name.to_string(), meta);
                }
            }
        }

        Ok(Self {
            file,
            file_len,
            tensors,
            max_gather_bytes,
            payload_bytes_read: AtomicU64::new(0),
        })
    }

    /// Immutable per-gather output byte budget chosen at open.
    pub fn max_gather_bytes(&self) -> usize {
        self.max_gather_bytes
    }

    /// Cumulative payload bytes read by successful gathers (excludes header).
    ///
    /// Each successful [`Self::gather_rows`] adds exactly
    /// `row_ids.len() * row_bytes`; failed gathers add nothing. Open never
    /// counts toward this total, so tests can prove only requested rows move.
    pub fn payload_bytes_read(&self) -> u64 {
        self.payload_bytes_read.load(Ordering::Relaxed)
    }

    /// Sorted tensor names known to this reader.
    ///
    /// For [`Self::open_selected`] this is exactly the requested selection.
    pub fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tensors.keys().cloned().collect();
        names.sort();
        names
    }

    /// Cloned metadata record for `name`, if known to this reader.
    pub fn tensor_meta(&self, name: &str) -> Option<RowTensorMeta> {
        self.tensors.get(name).cloned()
    }

    /// Gather axis-zero rows, preserving all trailing dimensions.
    ///
    /// Preserves caller order and duplicates; reads only the requested rows
    /// (one positional read per row, no whole-table scan). Fails when the
    /// tensor is unknown, any id is `>= rows`, the output would exceed
    /// [`Self::max_gather_bytes`], checked arithmetic overflows, or a payload
    /// read fails (e.g. the file was truncated after open).
    ///
    /// The budget is enforced before any payload I/O: over-budget and
    /// out-of-bounds gathers fail without reading or accounting a single row.
    /// The returned array uses the safe [`MlxArray::from_raw_data`] copy
    /// constructor with owned storage, so later gathers and dropping the
    /// reader cannot overwrite it.
    pub fn gather_rows(&self, tensor: &str, row_ids: &[u64]) -> Result<MlxArray, String> {
        let meta = self
            .tensors
            .get(tensor)
            .ok_or_else(|| format!("unknown tensor {tensor}"))?;
        let num_rows = row_ids.len();
        if num_rows > i32::MAX as usize {
            return Err(format!(
                "gather {tensor}: row count {num_rows} exceeds i32::MAX"
            ));
        }
        let row_bytes = meta.row_bytes;
        let output_bytes = num_rows
            .checked_mul(row_bytes)
            .ok_or_else(|| format!("gather {tensor}: output byte length overflows usize"))?;
        if output_bytes > self.max_gather_bytes {
            return Err(format!(
                "gather {tensor}: output {output_bytes} bytes exceeds budget {} bytes \
                 ({num_rows} rows x {row_bytes} bytes/row)",
                self.max_gather_bytes
            ));
        }
        if num_rows == 0 {
            let mut shape = meta.shape.clone();
            shape[0] = 0;
            let empty: Vec<u8> = Vec::new();
            return Ok(MlxArray::from_raw_data(
                empty.as_ptr(),
                0,
                &shape,
                meta.dtype,
            ));
        }
        // Validate every index before any payload I/O so failures add no
        // accounting and perform no partial reads.
        let rows_u64 = meta.rows as u64;
        for (pos, &idx) in row_ids.iter().enumerate() {
            if idx >= rows_u64 {
                return Err(format!(
                    "gather {tensor}: row id {idx} at position {pos} out of bounds (rows {})",
                    meta.rows
                ));
            }
        }
        let mut out = vec![0u8; output_bytes];
        let row_bytes_u64 = row_bytes as u64;
        for (i, &idx) in row_ids.iter().enumerate() {
            let row_off = idx
                .checked_mul(row_bytes_u64)
                .ok_or_else(|| format!("gather {tensor}: row offset overflows for row {idx}"))?;
            let abs = meta.data_start.checked_add(row_off).ok_or_else(|| {
                format!("gather {tensor}: absolute offset overflows for row {idx}")
            })?;
            let abs_end = abs
                .checked_add(row_bytes_u64)
                .ok_or_else(|| format!("gather {tensor}: absolute end overflows for row {idx}"))?;
            if abs > self.file_len || abs_end > self.file_len {
                return Err(format!(
                    "gather {tensor}: row {idx} range [{abs},{abs_end}] out of bounds \
                     (file {} bytes; file may be truncated)",
                    self.file_len
                ));
            }
            if abs < meta.data_start || abs_end > meta.data_end {
                return Err(format!(
                    "gather {tensor}: row {idx} range [{abs},{abs_end}] outside tensor bounds"
                ));
            }
            let dst_start = i.checked_mul(row_bytes).expect("output_bytes checked");
            let dst_end = dst_start + row_bytes;
            let dst = &mut out[dst_start..dst_end];
            pread_exact(&self.file, dst, abs).map_err(|e| {
                format!(
                    "gather {tensor}: read row {idx} at file offset {abs}: {e} \
                 (file may be truncated)"
                )
            })?;
        }
        self.payload_bytes_read
            .fetch_add(output_bytes as u64, Ordering::Relaxed);
        let mut shape = meta.shape.clone();
        shape[0] = num_rows as i32;
        Ok(MlxArray::from_raw_data(
            out.as_ptr(),
            out.len(),
            &shape,
            meta.dtype,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::add;
    use crate::transforms::eval;
    use std::path::{Path, PathBuf};

    #[test]
    fn rank3_selected_stacks_preserve_shape_dtype_order_and_owned_storage() {
        let dir = test_dir("rank3-dtypes");
        for (name, dtype, elem) in [
            ("F32", MlxDtype::Float32, 4usize),
            ("U32", MlxDtype::Uint32, 4),
            ("BF16", MlxDtype::Bfloat16, 2),
            ("F16", MlxDtype::Float16, 2),
        ] {
            let values: Vec<u32> = (0..24).map(|n| (n / 6 + n % 6) % 4).collect();
            let payload: Vec<u8> = values
                .iter()
                .flat_map(|&value| match name {
                    "F32" => (value as f32).to_le_bytes().to_vec(),
                    "U32" => value.to_le_bytes().to_vec(),
                    "BF16" => (((value as f32).to_bits() >> 16) as u16)
                        .to_le_bytes()
                        .to_vec(),
                    _ => [0u16, 0x3c00, 0x4000, 0x4200][value as usize]
                        .to_le_bytes()
                        .to_vec(),
                })
                .collect();
            let path = dir.join(format!("{name}.safetensors"));
            write_st_file(
                &path,
                serde_json::json!({
                    "experts": {"dtype": name, "shape": [4,2,3], "data_offsets": [0,payload.len()]},
                    "ignored": {"dtype": "I64", "shape": [1], "data_offsets": [0,8]}
                }),
                &payload,
            );
            assert!(SafetensorsRowReader::open_selected(&path, &["experts"], 1024).is_err());
            let reader =
                SafetensorsRowReader::open_selected_stacks(&path, &["experts"], 1024).unwrap();
            let meta = reader.tensor_meta("experts").unwrap();
            assert_eq!(meta.shape, [4, 2, 3]);
            assert_eq!(meta.cols, 6);
            assert_eq!(meta.row_bytes, 6 * elem);
            assert_eq!(reader.payload_bytes_read(), 0);
            let first = reader.gather_rows("experts", &[3, 1, 3]).unwrap();
            let next = reader.gather_rows("experts", &[0]).unwrap();
            let empty = reader.gather_rows("experts", &[]).unwrap();
            assert_eq!(first.dtype(), dtype);
            assert_eq!(first.shape(), [3, 2, 3]);
            assert_eq!(empty.shape(), [0, 2, 3]);
            assert_eq!(reader.payload_bytes_read(), (24 * elem) as u64);
            drop(reader);
            let first = crate::ops::astype(&first, MlxDtype::Float32, None);
            eval(&[&first, &next]);
            let expected: Vec<f32> = [3usize, 1, 3]
                .iter()
                .flat_map(|&expert| {
                    values[expert * 6..(expert + 1) * 6]
                        .iter()
                        .map(|&v| v as f32)
                })
                .collect();
            assert_eq!(first.data_f32(), expected);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rank3_stack_budget_and_all_indices_are_checked_before_payload_io() {
        let dir = test_dir("rank3-bounds");
        let path = dir.join("stack.safetensors");
        write_st_file(
            &path,
            serde_json::json!({
                "experts": {"dtype": "U32", "shape": [4,2,3], "data_offsets": [0,96]}
            }),
            &[0u8; 96],
        );
        let reader = SafetensorsRowReader::open_selected_stacks(&path, &["experts"], 24).unwrap();
        assert!(
            reader
                .gather_rows("experts", &[0, 1])
                .unwrap_err()
                .contains("budget")
        );
        assert_eq!(reader.payload_bytes_read(), 0);
        let reader = SafetensorsRowReader::open_selected_stacks(&path, &["experts"], 96).unwrap();
        assert!(
            reader
                .gather_rows("experts", &[0, 4])
                .unwrap_err()
                .contains("out of bounds")
        );
        assert_eq!(reader.payload_bytes_read(), 0);
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(reader.tensor_meta("experts").unwrap().data_start + 24)
            .unwrap();
        assert!(
            reader
                .gather_rows("experts", &[0, 1])
                .unwrap_err()
                .contains("read row 1")
        );
        assert_eq!(reader.payload_bytes_read(), 0);
        assert_eq!(
            reader.gather_rows("experts", &[0]).unwrap().shape(),
            [1, 2, 3]
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rank3_stack_metadata_rejects_wrong_rank_length_and_overflow() {
        let dir = test_dir("rank3-invalid");
        let path = dir.join("stack.safetensors");
        for shape in [
            vec![4, 6],
            vec![4, 2, 4],
            vec![4, 0, 3],
            vec![2147483647u64; 3],
        ] {
            write_st_file(
                &path,
                serde_json::json!({
                    "experts": {"dtype": "U32", "shape": shape, "data_offsets": [0,96]}
                }),
                &[0u8; 96],
            );
            assert!(SafetensorsRowReader::open_selected_stacks(&path, &["experts"], 1024).is_err());
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ax_io_rows_{}_{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_st_file(path: &Path, header: serde_json::Value, payload: &[u8]) {
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut f = std::fs::File::create(path).unwrap();
        std::io::Write::write_all(&mut f, &(header_bytes.len() as u64).to_le_bytes()).unwrap();
        std::io::Write::write_all(&mut f, &header_bytes).unwrap();
        std::io::Write::write_all(&mut f, payload).unwrap();
        f.sync_all().unwrap();
    }

    fn f32_rows_payload(rows: &[Vec<f32>]) -> Vec<u8> {
        let mut out = Vec::new();
        for r in rows {
            for &x in r {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    fn u32_rows_payload(rows: &[Vec<u32>]) -> Vec<u8> {
        let mut out = Vec::new();
        for r in rows {
            for &x in r {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    #[test]
    fn row_order_and_duplicates_preserved() {
        let dir = test_dir("order_dup");
        let path = dir.join("t.safetensors");
        // [4,3] F32 with distinct rows.
        let rows: Vec<Vec<f32>> = vec![
            vec![0.0, 1.0, 2.0],
            vec![10.0, 11.0, 12.0],
            vec![20.0, 21.0, 22.0],
            vec![30.0, 31.0, 32.0],
        ];
        let payload = f32_rows_payload(&rows);
        assert_eq!(payload.len(), 48);
        let header = serde_json::json!({
            "emb": {"dtype": "F32", "shape": [4, 3], "data_offsets": [0, 48]}
        });
        write_st_file(&path, header, &payload);

        let reader = SafetensorsRowReader::open(&path).expect("open must succeed");
        assert_eq!(reader.tensor_names(), vec!["emb".to_string()]);
        let meta = reader.tensor_meta("emb").unwrap();
        assert_eq!(meta.shape, vec![4, 3]);
        assert_eq!(meta.dtype, MlxDtype::Float32);
        assert_eq!((meta.rows, meta.cols), (4, 3));
        assert_eq!(meta.row_bytes, 12);
        assert_eq!(reader.payload_bytes_read(), 0);

        let out = reader
            .gather_rows("emb", &[2, 0, 2, 1])
            .expect("gather must succeed");
        assert_eq!(out.shape(), vec![4, 3]);
        assert_eq!(out.dtype(), MlxDtype::Float32);
        eval(&[&out]);
        assert_eq!(
            out.data_f32(),
            &[
                20.0, 21.0, 22.0, 0.0, 1.0, 2.0, 20.0, 21.0, 22.0, 10.0, 11.0, 12.0
            ]
        );
        // Exactly 4 rows x 12 bytes; duplicates count, untouched row 3 not read.
        assert_eq!(reader.payload_bytes_read(), 48);

        // Accounting is cumulative with no reset; the next gather adds 12.
        let one = reader.gather_rows("emb", &[1]).unwrap();
        eval(&[&one]);
        assert_eq!(one.data_f32(), &[10.0, 11.0, 12.0]);
        assert_eq!(reader.payload_bytes_read(), 60);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn supports_f16_bf16_and_raw_u32_buffers() {
        // U32 here exercises raw row buffers only, not affine-quantization
        // numeric parity (owned by the quantization integration).
        let dir = test_dir("dtypes");
        let path = dir.join("t.safetensors");
        // dense_f16 [2,2] F16 (8 bytes), dense_bf16 [2,3] BF16 (12 bytes),
        // raw_u32 [3,2] U32 (24 bytes of raw row data).
        let f16_payload: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let bf16_payload: Vec<u8> = (10u8..22u8).collect();
        let u32_rows: Vec<Vec<u32>> = vec![vec![100, 101], vec![200, 201], vec![300, 301]];
        let u32_payload = u32_rows_payload(&u32_rows);
        let mut payload = Vec::new();
        payload.extend_from_slice(&f16_payload);
        payload.extend_from_slice(&bf16_payload);
        payload.extend_from_slice(&u32_payload);
        let header = serde_json::json!({
            "dense_f16": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]},
            "dense_bf16": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [8, 20]},
            "raw_u32": {"dtype": "U32", "shape": [3, 2], "data_offsets": [20, 44]}
        });
        write_st_file(&path, header, &payload);

        let reader = SafetensorsRowReader::open(&path).unwrap();
        assert_eq!(
            reader.tensor_meta("dense_f16").unwrap().dtype,
            MlxDtype::Float16
        );
        assert_eq!(
            reader.tensor_meta("dense_bf16").unwrap().dtype,
            MlxDtype::Bfloat16
        );
        let u32_meta = reader.tensor_meta("raw_u32").unwrap();
        assert_eq!(u32_meta.dtype, MlxDtype::Uint32);
        assert_eq!(u32_meta.shape, vec![3, 2]);

        let a = reader.gather_rows("dense_f16", &[1]).unwrap();
        assert_eq!(a.shape(), vec![1, 2]);
        assert_eq!(a.dtype(), MlxDtype::Float16);
        assert_eq!(a.nbytes(), 4);

        let b = reader.gather_rows("dense_bf16", &[0, 1]).unwrap();
        assert_eq!(b.shape(), vec![2, 3]);
        assert_eq!(b.dtype(), MlxDtype::Bfloat16);
        assert_eq!(b.nbytes(), 12);

        let q = reader.gather_rows("raw_u32", &[2, 0]).unwrap();
        assert_eq!(q.shape(), vec![2, 2]);
        assert_eq!(q.dtype(), MlxDtype::Uint32);
        eval(&[&q]);
        assert_eq!(q.data_u32(), &[300, 301, 100, 101]);

        // 4 + 12 + 16 = 32 payload bytes total.
        assert_eq!(reader.payload_bytes_read(), 32);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_unsupported_dtype() {
        let dir = test_dir("bad_dtype");
        for (i, bad) in ["F64", "I8", "F8_E4M3", "BOOL", "XYZ"]
            .into_iter()
            .enumerate()
        {
            let path = dir.join(format!("t{i}.safetensors"));
            let header = serde_json::json!({
                "w": {"dtype": bad, "shape": [2, 2], "data_offsets": [0, 16]}
            });
            write_st_file(&path, header, &[0u8; 16]);
            let err = SafetensorsRowReader::open(&path).expect_err("must fail closed");
            assert!(
                err.contains("unsupported dtype") && err.contains(bad),
                "unexpected error for {bad}: {err}"
            );
            let err = SafetensorsRowReader::open_selected(&path, &["w"], 1024)
                .expect_err("selected open must fail closed too");
            assert!(
                err.contains("unsupported dtype") && err.contains(bad),
                "unexpected error for {bad}: {err}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_invalid_rank_and_zero_dims() {
        let dir = test_dir("bad_dims");
        // Rank 1, rank 3, zero rows, zero cols.
        let cases: Vec<(serde_json::Value, &str)> = vec![
            (serde_json::json!([4]), "rank"),
            (serde_json::json!([2, 2, 2]), "rank"),
            (serde_json::json!([0, 4]), "positive"),
            (serde_json::json!([4, 0]), "positive"),
        ];
        for (i, (shape, needle)) in cases.into_iter().enumerate() {
            let path = dir.join(format!("t{i}.safetensors"));
            let header = serde_json::json!({
                "w": {"dtype": "F32", "shape": shape, "data_offsets": [0, 0]}
            });
            write_st_file(&path, header, &[]);
            let err = SafetensorsRowReader::open(&path).expect_err("must reject shape");
            assert!(
                err.contains(needle),
                "case {i} error should contain {needle:?}: {err}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_mismatched_byte_length() {
        let dir = test_dir("bad_len");
        // [2,2] F32 expects exactly 16 bytes.
        for (i, offsets) in [vec![0, 12], vec![0, 20], vec![4, 16]]
            .into_iter()
            .enumerate()
        {
            let path = dir.join(format!("t{i}.safetensors"));
            let header = serde_json::json!({
                "w": {"dtype": "F32", "shape": [2, 2], "data_offsets": offsets}
            });
            write_st_file(&path, header, &[0u8; 20]);
            let err = SafetensorsRowReader::open(&path).expect_err("must reject length");
            assert!(
                err.contains("does not match"),
                "case {i} unexpected error: {err}"
            );
        }
        // start > end.
        let path = dir.join("rev.safetensors");
        let header = serde_json::json!({
            "w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [16, 0]}
        });
        write_st_file(&path, header, &[0u8; 16]);
        let err = SafetensorsRowReader::open(&path).expect_err("start>end must fail");
        assert!(err.contains("start > end"), "unexpected error: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_offsets_out_of_file_bounds() {
        let dir = test_dir("oob_open");
        let path = dir.join("t.safetensors");
        // Claims 16 bytes but file carries only 8 payload bytes.
        let header = serde_json::json!({
            "w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}
        });
        write_st_file(&path, header, &[7u8; 8]);
        let err = SafetensorsRowReader::open(&path).expect_err("must reject OOB");
        assert!(err.contains("out of bounds"), "unexpected error: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gather_rejects_out_of_bounds_and_unknown() {
        let dir = test_dir("oob_gather");
        let path = dir.join("t.safetensors");
        let rows: Vec<Vec<f32>> = vec![vec![0.0, 1.0], vec![10.0, 11.0], vec![20.0, 21.0]];
        let payload = f32_rows_payload(&rows);
        let header = serde_json::json!({
            "emb": {"dtype": "F32", "shape": [3, 2], "data_offsets": [0, 24]}
        });
        write_st_file(&path, header, &payload);
        let reader = SafetensorsRowReader::open(&path).unwrap();

        let err = reader.gather_rows("emb", &[0, 3]).expect_err("row 3 OOB");
        assert!(err.contains("out of bounds"), "unexpected: {err}");
        let err = reader.gather_rows("emb", &[100]).expect_err("row 100 OOB");
        assert!(err.contains("out of bounds"), "unexpected: {err}");
        // No negative-index case: row ids are u64, so negativity is
        // unrepresentable at the type level.
        let err = reader
            .gather_rows("missing", &[0])
            .expect_err("unknown tensor must fail");
        assert!(err.contains("unknown tensor"), "unexpected: {err}");
        // Failed gathers add no payload accounting.
        assert_eq!(reader.payload_bytes_read(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gather_enforces_byte_budget() {
        let dir = test_dir("budget");
        let path = dir.join("t.safetensors");
        // [4,2] F32: row_bytes = 8.
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let payload = f32_rows_payload(&rows);
        let header = serde_json::json!({
            "w": {"dtype": "F32", "shape": [4, 2], "data_offsets": [0, 32]}
        });
        write_st_file(&path, header, &payload);

        let reader = SafetensorsRowReader::open_with_budget(&path, 16).unwrap();
        assert_eq!(reader.max_gather_bytes(), 16);
        let ok = reader.gather_rows("w", &[0, 1]).unwrap();
        assert_eq!(ok.shape(), vec![2, 2]);
        assert_eq!(reader.payload_bytes_read(), 16);
        let err = reader
            .gather_rows("w", &[0, 1, 2])
            .expect_err("3 rows exceed 16-byte budget");
        assert!(err.contains("exceeds budget"), "unexpected: {err}");
        // Failed budget check adds nothing.
        assert_eq!(reader.payload_bytes_read(), 16);

        // The budget is immutable: a tighter budget needs a new reader.
        let tight = SafetensorsRowReader::open_with_budget(&path, 8).unwrap();
        assert_eq!(tight.max_gather_bytes(), 8);
        let err = tight
            .gather_rows("w", &[0, 1])
            .expect_err("2 rows exceed 8-byte budget");
        assert!(err.contains("exceeds budget"), "unexpected: {err}");
        let one = tight.gather_rows("w", &[3]).unwrap();
        eval(&[&one]);
        assert_eq!(one.data_f32(), &[7.0, 8.0]);

        // open_selected carries its own immutable budget too.
        let selected = SafetensorsRowReader::open_selected(&path, &["w"], 8).unwrap();
        assert_eq!(selected.max_gather_bytes(), 8);
        assert!(selected.gather_rows("w", &[0, 1]).is_err());
        assert!(selected.gather_rows("w", &[0]).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gather_fails_on_truncated_file() {
        let dir = test_dir("trunc");
        let path = dir.join("t.safetensors");
        let rows: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let payload = f32_rows_payload(&rows);
        let header = serde_json::json!({
            "w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}
        });
        write_st_file(&path, header, &payload);
        let reader = SafetensorsRowReader::open(&path).unwrap();
        // Sanity: one row reads before truncation.
        assert!(reader.gather_rows("w", &[0]).is_ok());
        assert_eq!(reader.payload_bytes_read(), 8);

        // Truncate the 16-byte payload away behind the kept handle.
        let full_len = std::fs::metadata(&path).unwrap().len();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(full_len - 16)
            .unwrap();
        let err = reader
            .gather_rows("w", &[1])
            .expect_err("truncated read must fail");
        assert!(
            err.contains("truncated") || err.contains("read row") || err.contains("out of bounds"),
            "unexpected: {err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn payload_bytes_count_only_requested_rows() {
        let dir = test_dir("payload");
        let path = dir.join("t.safetensors");
        // [4,4] F32: row_bytes = 16, whole table = 64.
        let rows: Vec<Vec<f32>> = (0..4)
            .map(|r| (0..4).map(|c| (r * 10 + c) as f32).collect())
            .collect();
        let payload = f32_rows_payload(&rows);
        let header = serde_json::json!({
            "w": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]}
        });
        write_st_file(&path, header, &payload);
        let reader = SafetensorsRowReader::open(&path).unwrap();
        assert_eq!(reader.payload_bytes_read(), 0);

        let two = reader.gather_rows("w", &[0, 3]).unwrap();
        assert_eq!(two.shape(), vec![2, 4]);
        assert_eq!(reader.payload_bytes_read(), 32);

        let dup = reader.gather_rows("w", &[1, 1, 1]).unwrap();
        assert_eq!(dup.shape(), vec![3, 4]);
        eval(&[&dup]);
        assert_eq!(
            dup.data_f32(),
            &[
                10.0, 11.0, 12.0, 13.0, 10.0, 11.0, 12.0, 13.0, 10.0, 11.0, 12.0, 13.0
            ]
        );
        // 32 + 3*16 = 80 cumulative; duplicates count each occurrence.
        assert_eq!(reader.payload_bytes_read(), 80);

        // Empty gathers read nothing and add nothing.
        let empty = reader.gather_rows("w", &[]).unwrap();
        assert_eq!(empty.shape(), vec![0, 4]);
        assert_eq!(empty.nbytes(), 0);
        assert_eq!(reader.payload_bytes_read(), 80);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn header_cap_and_truncation_rejected_without_large_alloc() {
        let dir = test_dir("header_cap");
        // The cap is exclusive: exactly MAX and above fail before allocating.
        for (name, claimed) in [("over", MAX_HEADER_BYTES + 1), ("exact", MAX_HEADER_BYTES)] {
            let p = dir.join(format!("{name}.safetensors"));
            {
                let mut f = std::fs::File::create(&p).unwrap();
                std::io::Write::write_all(&mut f, &claimed.to_le_bytes()).unwrap();
            }
            let err = SafetensorsRowReader::open(&p).expect_err("cap must reject");
            assert!(err.contains("reaches or exceeds cap"), "unexpected: {err}");
            let err = SafetensorsRowReader::open_selected(&p, &["w"], 8)
                .expect_err("selected open must reject cap too");
            assert!(err.contains("reaches or exceeds cap"), "unexpected: {err}");
        }

        // Header claims 100 bytes but file holds only 10 header bytes.
        let short = dir.join("short.safetensors");
        {
            let mut f = std::fs::File::create(&short).unwrap();
            std::io::Write::write_all(&mut f, &100u64.to_le_bytes()).unwrap();
            std::io::Write::write_all(&mut f, &[0u8; 10]).unwrap();
        }
        let err = SafetensorsRowReader::open(&short).expect_err("short file must fail");
        assert!(err.contains("exceeds file size"), "unexpected: {err}");

        // File smaller than the 8-byte prefix.
        let tiny = dir.join("tiny.safetensors");
        std::fs::write(&tiny, [1u8, 2, 3, 4]).unwrap();
        let err = SafetensorsRowReader::open(&tiny).expect_err("tiny must fail");
        assert!(err.contains("too small"), "unexpected: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_selected_ignores_unrelated_entries() {
        let dir = test_dir("mixed_selected");
        let path = dir.join("mixed.safetensors");
        // Real-checkpoint shape: one rank-2 F32 n-gram table plus unrelated
        // rank-1 norm, rank-1 I64 multipliers, and rank-3 experts.
        let table_rows: Vec<Vec<f32>> = vec![
            vec![0.0, 1.0],
            vec![10.0, 11.0],
            vec![20.0, 21.0],
            vec![30.0, 31.0],
        ];
        let mut payload = f32_rows_payload(&table_rows);
        payload.extend_from_slice(&[0u8; 16]); // norm: rank-1 F32 [4]
        payload.extend_from_slice(&[0u8; 32]); // multipliers: rank-1 I64 [4]
        payload.extend_from_slice(&[0u8; 32]); // experts: rank-3 F32 [2,2,2]
        assert_eq!(payload.len(), 112);
        let header = serde_json::json!({
            "ngram_table": {"dtype": "F32", "shape": [4, 2], "data_offsets": [0, 32]},
            "norm": {"dtype": "F32", "shape": [4], "data_offsets": [32, 48]},
            "layer_multipliers": {"dtype": "I64", "shape": [4], "data_offsets": [48, 80]},
            "experts": {"dtype": "F32", "shape": [2, 2, 2], "data_offsets": [80, 112]}
        });
        write_st_file(&path, header, &payload);

        // Whole-file open validates every entry, so it rejects the mixed file.
        let err = SafetensorsRowReader::open(&path).expect_err("open must reject mixed file");
        assert!(
            err.contains("rank") || err.contains("unsupported dtype"),
            "unexpected: {err}"
        );

        // Selected open validates only the requested entry.
        let reader = SafetensorsRowReader::open_selected(&path, &["ngram_table"], 1024).unwrap();
        assert_eq!(reader.tensor_names(), vec!["ngram_table".to_string()]);
        assert_eq!(reader.max_gather_bytes(), 1024);
        let meta = reader.tensor_meta("ngram_table").unwrap();
        assert_eq!((meta.rows, meta.cols), (4, 2));
        assert!(reader.tensor_meta("norm").is_none());

        let out = reader.gather_rows("ngram_table", &[3, 0]).unwrap();
        assert_eq!(out.shape(), vec![2, 2]);
        eval(&[&out]);
        assert_eq!(out.data_f32(), &[30.0, 31.0, 0.0, 1.0]);
        // Only the two selected rows moved; unrelated payload untouched.
        assert_eq!(reader.payload_bytes_read(), 16);

        // Unselected entries are not gatherable through this reader.
        let err = reader
            .gather_rows("norm", &[0])
            .expect_err("unselected entry must be unknown");
        assert!(err.contains("unknown tensor"), "unexpected: {err}");

        // Missing and duplicate requests fail without reading payload.
        let err = SafetensorsRowReader::open_selected(&path, &["ngram_table", "absent"], 1024)
            .expect_err("missing request must fail");
        assert!(err.contains("not found"), "unexpected: {err}");
        let err = SafetensorsRowReader::open_selected(&path, &["ngram_table", "ngram_table"], 1024)
            .expect_err("duplicate request must fail");
        assert!(err.contains("duplicate"), "unexpected: {err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn retained_result_survives_later_gather_and_reader_drop() {
        let dir = test_dir("retained");
        let path = dir.join("t.safetensors");
        let a_rows: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b_rows: Vec<Vec<f32>> = vec![vec![10.0, 20.0], vec![30.0, 40.0]];
        let mut payload = f32_rows_payload(&a_rows);
        payload.extend_from_slice(&f32_rows_payload(&b_rows));
        let header = serde_json::json!({
            "a": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
            "b": {"dtype": "F32", "shape": [2, 2], "data_offsets": [16, 32]}
        });
        write_st_file(&path, header, &payload);

        let reader = SafetensorsRowReader::open(&path).unwrap();
        // Gather A first, then build lazy arithmetic over it without
        // evaluating, so any storage reuse would corrupt the later eval.
        let a = reader.gather_rows("a", &[0, 1]).unwrap();
        let doubled = add(&a, &a, None);
        // A later gather must not overwrite A's storage.
        let b = reader.gather_rows("b", &[1, 0]).unwrap();
        assert_eq!(reader.payload_bytes_read(), 32);
        drop(reader);
        // Both results stay valid after the reader is gone.
        eval(&[&a, &doubled, &b]);
        assert_eq!(a.data_f32(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(doubled.data_f32(), &[2.0, 4.0, 6.0, 8.0]);
        assert_eq!(b.data_f32(), &[30.0, 40.0, 10.0, 20.0]);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
