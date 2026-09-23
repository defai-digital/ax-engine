//! Bounded Flash Next PLE n-gram table over safetensors row reads.
//!
//! This module serves embedding rows for one PLE layer without ever loading
//! a full table. Open reads only bounded JSON headers through
//! [`mlx_sys::SafetensorsRowReader::open_selected`]; gather reads only the
//! requested rows via positional reads and converts them to F32 with the
//! existing MLX dequantize and cast ops. Design notes that shaped this file:
//! full-tensor lazy Load would still move whole payloads on eval, so it is
//! avoided here; returned rows own their storage so later gathers cannot
//! overwrite them; payload bytes are counted so tests can prove that only
//! requested rows move. No reference code or comments are reused here.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use ax_engine_core::{NativeTensorDataType, NativeTensorRole, NativeTensorSpec};
use mlx_sys::{MlxArray, MlxDtype, RowTensorMeta, SafetensorsRowReader};

/// Errors for [`NgramTable`] open and gather.
///
/// The public API surfaces these as strings; the enum keeps each failure
/// typed inside this module so call sites return instead of halting.
#[derive(Clone, Debug, PartialEq, Eq)]
enum NgramTableError {
    EmptySpecsForLayer(u32),
    ExpectedMismatch {
        expected: usize,
        found: usize,
    },
    InvalidName(String),
    MixedLayouts(String),
    DuplicateShard(usize),
    MissingShards(Vec<usize>),
    BadExpectedShards,
    BadEmbeddingWidth,
    UnsupportedEncoding(String),
    MissingQuantization(String),
    InvalidBits {
        name: String,
        bits: u32,
    },
    InvalidGroup {
        name: String,
        group: u32,
    },
    InvalidMode {
        name: String,
        mode: String,
    },
    WidthMismatch {
        name: String,
        expected: usize,
        found: usize,
    },
    DtypeMismatch {
        name: String,
        expected: MlxDtype,
        found: MlxDtype,
    },
    PackedMismatch(String),
    Sidecar(String),
    Index(String),
    Path(String),
    Reader(String),
    RowBounds {
        row: u64,
        rows: u64,
        position: usize,
    },
    TooManyRows(usize),
    Budget {
        needed: usize,
        budget: usize,
    },
    Overflow(String),
    Gather(String),
    Internal(String),
}

impl std::fmt::Display for NgramTableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySpecsForLayer(layer) => {
                write!(f, "ngram table: no NgramEmbedding specs for layer {layer}")
            }
            Self::ExpectedMismatch { expected, found } => {
                write!(f, "ngram table: expected {expected} shards, found {found}")
            }
            Self::InvalidName(name) => {
                write!(f, "ngram table: invalid tensor name {name:?}")
            }
            Self::MixedLayouts(detail) => {
                write!(f, "ngram table: mixed layouts: {detail}")
            }
            Self::DuplicateShard(index) => {
                write!(f, "ngram table: duplicate shard index {index}")
            }
            Self::MissingShards(missing) => {
                write!(f, "ngram table: missing shard indexes {missing:?}")
            }
            Self::BadExpectedShards => {
                write!(f, "ngram table: expected_shards must be positive")
            }
            Self::BadEmbeddingWidth => {
                write!(
                    f,
                    "ngram table: embedding_width must be positive and fit i32"
                )
            }
            Self::UnsupportedEncoding(detail) => write!(f, "{detail}"),
            Self::MissingQuantization(name) => write!(
                f,
                "ngram table: tensor {name:?} needs explicit affine quantization"
            ),
            Self::InvalidBits { name, bits } => write!(
                f,
                "ngram table: tensor {name:?} has unsupported bits {bits} (need 2, 4, 6 or 8)"
            ),
            Self::InvalidGroup { name, group } => write!(
                f,
                "ngram table: tensor {name:?} has unsupported group {group} (need 32, 64 or 128)"
            ),
            Self::InvalidMode { name, mode } => write!(
                f,
                "ngram table: tensor {name:?} has unsupported mode {mode:?} (need affine)"
            ),
            Self::WidthMismatch {
                name,
                expected,
                found,
            } => write!(
                f,
                "ngram table: tensor {name:?} width {found} != expected {expected}"
            ),
            Self::DtypeMismatch {
                name,
                expected,
                found,
            } => write!(
                f,
                "ngram table: tensor {name:?} dtype {found:?} != expected {expected:?}"
            ),
            Self::PackedMismatch(detail) => {
                write!(f, "ngram table: packed width mismatch: {detail}")
            }
            Self::Sidecar(detail) => write!(f, "ngram table: sidecar invalid: {detail}"),
            Self::Index(detail) => write!(f, "ngram table: index invalid: {detail}"),
            Self::Path(detail) => write!(f, "ngram table: path invalid: {detail}"),
            Self::Reader(detail) => write!(f, "ngram table: row reader failed: {detail}"),
            Self::RowBounds {
                row,
                rows,
                position,
            } => write!(
                f,
                "ngram table: row {row} at position {position} out of bounds (rows {rows})"
            ),
            Self::TooManyRows(count) => {
                write!(f, "ngram table: row count {count} exceeds i32::MAX")
            }
            Self::Budget { needed, budget } => write!(
                f,
                "ngram table: needed {needed} bytes exceeds budget {budget} bytes"
            ),
            Self::Overflow(detail) => write!(f, "ngram table: arithmetic overflow: {detail}"),
            Self::Gather(detail) => write!(f, "ngram table: gather failed: {detail}"),
            Self::Internal(detail) => write!(f, "ngram table: internal error: {detail}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShardStyle {
    Underscore,
    Dots,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShardKind {
    Unsharded,
    Sharded { index: usize, style: ShardStyle },
}

fn parse_usize_digits(text: &str) -> Option<usize> {
    if text.is_empty() {
        return None;
    }
    for ch in text.chars() {
        if !ch.is_ascii_digit() {
            return None;
        }
    }
    let mut value: usize = 0;
    for ch in text.chars() {
        let digit = (ch as u8 - b'0') as usize;
        value = value.checked_mul(10)?.checked_add(digit)?;
    }
    Some(value)
}

fn parse_shard_kind(name: &str) -> Result<ShardKind, NgramTableError> {
    let base = name
        .strip_suffix(".weight")
        .ok_or_else(|| NgramTableError::InvalidName(name.to_string()))?;
    if let Some(pos) = base.rfind("shards.") {
        let suffix = base
            .get(pos + "shards.".len()..)
            .ok_or_else(|| NgramTableError::InvalidName(name.to_string()))?;
        if let Some(index) = parse_usize_digits(suffix) {
            return Ok(ShardKind::Sharded {
                index,
                style: ShardStyle::Dots,
            });
        }
    }
    if let Some(pos) = base.rfind("shard_") {
        let suffix = base
            .get(pos + "shard_".len()..)
            .ok_or_else(|| NgramTableError::InvalidName(name.to_string()))?;
        if let Some(index) = parse_usize_digits(suffix) {
            return Ok(ShardKind::Sharded {
                index,
                style: ShardStyle::Underscore,
            });
        }
    }
    if base == "ngram_embedding"
        || base.ends_with(".ngram_embedding")
        || base.ends_with("/ngram_embedding")
        || base.ends_with("ngram_embedding")
    {
        return Ok(ShardKind::Unsharded);
    }
    Err(NgramTableError::InvalidName(name.to_string()))
}

fn canonical_root_of(root: &Path) -> Result<PathBuf, NgramTableError> {
    root.canonicalize()
        .map_err(|e| NgramTableError::Path(format!("cannot resolve root {}: {e}", root.display())))
}

fn join_inside_root(
    root: &Path,
    canonical_root: &Path,
    relative: &Path,
) -> Result<PathBuf, NgramTableError> {
    crate::artifact_path::resolve_file(root, canonical_root, relative).map_err(|error| {
        NgramTableError::Path(format!(
            "cannot resolve {} inside {}: {error}",
            relative.display(),
            root.display()
        ))
    })
}

fn read_weight_map(root: &Path) -> Result<Option<HashMap<String, String>>, NgramTableError> {
    let index_path = root.join("model.safetensors.index.json");
    if !index_path.is_file() {
        return Ok(None);
    }
    let bytes = std::fs::read(&index_path)
        .map_err(|e| NgramTableError::Index(format!("cannot read index: {e}")))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| NgramTableError::Index(format!("cannot parse index: {e}")))?;
    let map = value
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| NgramTableError::Index("index lacks weight_map object".to_string()))?;
    let mut out = HashMap::new();
    for (key, val) in map {
        let file = val.as_str().ok_or_else(|| {
            NgramTableError::Index(format!("weight_map entry {key:?} is not a string"))
        })?;
        out.insert(key.clone(), file.to_string());
    }
    Ok(Some(out))
}

fn is_fp8_mxfp_hint(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("fp8")
        || lower.contains("mxfp")
        || lower.contains("nvfp")
        || lower.contains("e4m3")
        || lower.contains("e8m0")
}

fn unsupported_encoding(name: &str, detail: &str) -> NgramTableError {
    NgramTableError::UnsupportedEncoding(format!(
        "ngram table: tensor {name:?} uses unsupported encoding ({detail}); \
         table encodings outside dense BF16/F16/F32 and affine U32 2/4/6/8-bit are rejected, \
         and Flash Next model support remains incomplete"
    ))
}

fn dense_mlx_dtype(dtype: NativeTensorDataType) -> Option<MlxDtype> {
    match dtype {
        NativeTensorDataType::F32 => Some(MlxDtype::Float32),
        NativeTensorDataType::F16 => Some(MlxDtype::Float16),
        NativeTensorDataType::Bf16 => Some(MlxDtype::Bfloat16),
        _ => None,
    }
}

#[derive(Clone, Debug)]
enum ShardEncoding {
    Dense,
    Quantized {
        bits: u32,
        group_size: u32,
        scales_name: String,
        scales_file: PathBuf,
        scales_meta: Box<RowTensorMeta>,
        biases_name: String,
        biases_file: PathBuf,
        biases_meta: Box<RowTensorMeta>,
    },
}

#[derive(Clone, Debug)]
struct Shard {
    global_start: u64,
    rows: u64,
    weight_name: String,
    weight_file: PathBuf,
    weight_meta: RowTensorMeta,
    encoding: ShardEncoding,
}

impl Shard {
    fn staging_row_bytes(&self) -> usize {
        match &self.encoding {
            ShardEncoding::Dense => self.weight_meta.row_bytes,
            ShardEncoding::Quantized {
                scales_meta,
                biases_meta,
                ..
            } => self
                .weight_meta
                .row_bytes
                .saturating_add(scales_meta.row_bytes)
                .saturating_add(biases_meta.row_bytes),
        }
    }
}

/// Bounded row table for one Flash Next PLE layer.
pub struct NgramTable {
    shards: Vec<Shard>,
    readers: HashMap<PathBuf, SafetensorsRowReader>,
    total_rows: u64,
    embedding_width: usize,
    max_gather_bytes: usize,
}

struct SelectedShard {
    spec: NativeTensorSpec,
    shard_index: usize,
}

fn select_layer_shards(
    specs: &[NativeTensorSpec],
    layer: u32,
    expected_shards: usize,
) -> Result<Vec<SelectedShard>, NgramTableError> {
    if expected_shards == 0 {
        return Err(NgramTableError::BadExpectedShards);
    }
    let mut filtered: Vec<&NativeTensorSpec> = specs
        .iter()
        .filter(|s| s.role == NativeTensorRole::NgramEmbedding && s.layer_index == Some(layer))
        .collect();
    if filtered.is_empty() {
        return Err(NgramTableError::EmptySpecsForLayer(layer));
    }
    if filtered.len() != expected_shards {
        return Err(NgramTableError::ExpectedMismatch {
            expected: expected_shards,
            found: filtered.len(),
        });
    }
    let mut kinds: Vec<ShardKind> = Vec::new();
    for spec in &filtered {
        kinds.push(parse_shard_kind(&spec.name)?);
    }
    let has_sharded = kinds.iter().any(|k| matches!(k, ShardKind::Sharded { .. }));
    let has_unsharded = kinds.iter().any(|k| matches!(k, ShardKind::Unsharded));
    if has_sharded && has_unsharded {
        return Err(NgramTableError::MixedLayouts(
            "sharded and unsharded ngram_embedding names".to_string(),
        ));
    }
    if has_unsharded {
        if expected_shards != 1 {
            return Err(NgramTableError::ExpectedMismatch {
                expected: expected_shards,
                found: filtered.len(),
            });
        }
        let spec = filtered
            .pop()
            .ok_or_else(|| NgramTableError::Internal("empty after filter".to_string()))?;
        return Ok(vec![SelectedShard {
            spec: spec.clone(),
            shard_index: 0,
        }]);
    }
    let mut styles = HashSet::new();
    let mut indexes = Vec::new();
    for kind in &kinds {
        match kind {
            ShardKind::Sharded { index, style } => {
                styles.insert(*style as u8);
                indexes.push(*index);
            }
            ShardKind::Unsharded => {}
        }
    }
    if styles.len() > 1 {
        return Err(NgramTableError::MixedLayouts(
            "shard_N.weight and shards.N.weight styles".to_string(),
        ));
    }
    let mut seen = HashSet::new();
    for index in &indexes {
        if !seen.insert(*index) {
            return Err(NgramTableError::DuplicateShard(*index));
        }
    }
    let mut sorted = indexes.clone();
    sorted.sort_unstable();
    let mut missing = Vec::new();
    for want in 0..expected_shards {
        if !seen.contains(&want) {
            missing.push(want);
        }
    }
    if !missing.is_empty() {
        return Err(NgramTableError::MissingShards(missing));
    }
    let mut out: Vec<SelectedShard> = Vec::new();
    for spec in filtered {
        let kind = parse_shard_kind(&spec.name)?;
        match kind {
            ShardKind::Sharded { index, .. } => out.push(SelectedShard {
                spec: spec.clone(),
                shard_index: index,
            }),
            ShardKind::Unsharded => {
                return Err(NgramTableError::MixedLayouts(
                    "unexpected unsharded".to_string(),
                ));
            }
        }
    }
    out.sort_by_key(|s| s.shard_index);
    Ok(out)
}

struct QuantSpec {
    bits: u32,
    group_size: u32,
}

fn quant_spec_for(name: &str, spec: &NativeTensorSpec) -> Result<QuantSpec, NgramTableError> {
    if let Some(source) = spec.source_tensor_type.as_deref()
        && is_fp8_mxfp_hint(source)
    {
        return Err(unsupported_encoding(name, source));
    }
    let quant = spec
        .quantization
        .as_ref()
        .ok_or_else(|| NgramTableError::MissingQuantization(name.to_string()))?;
    if is_fp8_mxfp_hint(&quant.mode) {
        return Err(unsupported_encoding(name, &format!("mode {}", quant.mode)));
    }
    let mode = quant.mode.to_ascii_lowercase();
    if mode != "affine" && !mode.is_empty() {
        return Err(NgramTableError::InvalidMode {
            name: name.to_string(),
            mode: quant.mode.clone(),
        });
    }
    if !matches!(quant.bits, 2 | 4 | 6 | 8) {
        return Err(NgramTableError::InvalidBits {
            name: name.to_string(),
            bits: quant.bits,
        });
    }
    if !matches!(quant.group_size, 32 | 64 | 128) {
        return Err(NgramTableError::InvalidGroup {
            name: name.to_string(),
            group: quant.group_size,
        });
    }
    Ok(QuantSpec {
        bits: quant.bits,
        group_size: quant.group_size,
    })
}

fn check_dense_spec(name: &str, spec: &NativeTensorSpec) -> Result<MlxDtype, NgramTableError> {
    if let Some(source) = spec.source_tensor_type.as_deref()
        && is_fp8_mxfp_hint(source)
    {
        return Err(unsupported_encoding(name, source));
    }
    if let Some(quant) = spec.quantization.as_ref() {
        if is_fp8_mxfp_hint(&quant.mode) {
            return Err(unsupported_encoding(name, &format!("mode {}", quant.mode)));
        }
        return Err(NgramTableError::InvalidMode {
            name: name.to_string(),
            mode: quant.mode.clone(),
        });
    }
    dense_mlx_dtype(spec.dtype)
        .ok_or_else(|| unsupported_encoding(name, &format!("dtype {:?}", spec.dtype)))
}

impl NgramTable {
    pub fn open(
        root: &Path,
        specs: &[NativeTensorSpec],
        layer: u32,
        expected_shards: usize,
        embedding_width: usize,
        max_gather_bytes: usize,
    ) -> Result<Self, String> {
        Self::open_inner(
            root,
            specs,
            layer,
            expected_shards,
            embedding_width,
            max_gather_bytes,
        )
        .map_err(|e| e.to_string())
    }

    fn open_inner(
        root: &Path,
        specs: &[NativeTensorSpec],
        layer: u32,
        expected_shards: usize,
        embedding_width: usize,
        max_gather_bytes: usize,
    ) -> Result<Self, NgramTableError> {
        if embedding_width == 0 || embedding_width > i32::MAX as usize {
            return Err(NgramTableError::BadEmbeddingWidth);
        }
        let canonical_root = canonical_root_of(root)?;
        let selected = select_layer_shards(specs, layer, expected_shards)?;
        let weight_map = read_weight_map(root)?;
        let mut file_names: HashMap<PathBuf, HashSet<String>> = HashMap::new();
        let mut shard_files: Vec<PathBuf> = Vec::new();
        let mut shard_quant: Vec<Option<QuantSpec>> = Vec::new();
        let mut shard_dense: Vec<Option<MlxDtype>> = Vec::new();
        for item in &selected {
            let name = item.spec.name.clone();
            if matches!(
                item.spec.dtype,
                NativeTensorDataType::U8
                    | NativeTensorDataType::I8
                    | NativeTensorDataType::Q4Km
                    | NativeTensorDataType::Q5Km
                    | NativeTensorDataType::Q6Km
                    | NativeTensorDataType::Q8Zero
            ) {
                return Err(unsupported_encoding(
                    &name,
                    &format!("dtype {:?}", item.spec.dtype),
                ));
            }
            let weight_file = join_inside_root(root, &canonical_root, &item.spec.file)?;
            file_names
                .entry(weight_file.clone())
                .or_default()
                .insert(name.clone());
            shard_files.push(weight_file.clone());
            if item.spec.dtype == NativeTensorDataType::U32 {
                let quant = quant_spec_for(&name, &item.spec)?;
                let base = name
                    .strip_suffix(".weight")
                    .ok_or_else(|| NgramTableError::InvalidName(name.clone()))?;
                let scales_name = format!("{base}.scales");
                let biases_name = format!("{base}.biases");
                let (scales_file, biases_file) = match &weight_map {
                    Some(map) => {
                        let scales_rel = map.get(&scales_name).ok_or_else(|| {
                            NgramTableError::Sidecar(format!(
                                "tensor {scales_name:?} missing from index weight_map"
                            ))
                        })?;
                        let biases_rel = map.get(&biases_name).ok_or_else(|| {
                            NgramTableError::Sidecar(format!(
                                "tensor {biases_name:?} missing from index weight_map"
                            ))
                        })?;
                        (
                            join_inside_root(root, &canonical_root, Path::new(scales_rel))?,
                            join_inside_root(root, &canonical_root, Path::new(biases_rel))?,
                        )
                    }
                    None => (weight_file.clone(), weight_file.clone()),
                };
                file_names
                    .entry(scales_file)
                    .or_default()
                    .insert(scales_name);
                file_names
                    .entry(biases_file)
                    .or_default()
                    .insert(biases_name);
                shard_quant.push(Some(quant));
                shard_dense.push(None);
            } else {
                let dtype = check_dense_spec(&name, &item.spec)?;
                shard_quant.push(None);
                shard_dense.push(Some(dtype));
            }
        }
        let mut readers: HashMap<PathBuf, SafetensorsRowReader> = HashMap::new();
        for (file, names) in &file_names {
            let mut sorted: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            sorted.sort_unstable();
            let reader = SafetensorsRowReader::open_selected(file, &sorted, max_gather_bytes)
                .map_err(|e| NgramTableError::Reader(format!("{}: {e}", file.display())))?;
            readers.insert(file.clone(), reader);
        }
        let mut shards: Vec<Shard> = Vec::new();
        let mut total_rows: u64 = 0;
        for (pos, item) in selected.iter().enumerate() {
            let name = item.spec.name.clone();
            let weight_file = shard_files
                .get(pos)
                .ok_or_else(|| NgramTableError::Internal("shard file missing".to_string()))?
                .clone();
            let reader = readers
                .get(&weight_file)
                .ok_or_else(|| NgramTableError::Internal("weight reader missing".to_string()))?;
            let weight_meta = reader.tensor_meta(&name).ok_or_else(|| {
                NgramTableError::Reader(format!("tensor {name:?} missing after open"))
            })?;
            let rows_u64 = weight_meta.rows as u64;
            let encoding = match (
                shard_quant.get(pos).and_then(|q| q.as_ref()),
                shard_dense.get(pos).and_then(|d| *d),
            ) {
                (Some(quant), None) => {
                    if weight_meta.dtype != MlxDtype::Uint32 {
                        return Err(NgramTableError::DtypeMismatch {
                            name: name.clone(),
                            expected: MlxDtype::Uint32,
                            found: weight_meta.dtype,
                        });
                    }
                    let width_u64 = embedding_width as u64;
                    let prod = width_u64
                        .checked_mul(quant.bits as u64)
                        .ok_or_else(|| NgramTableError::Overflow("width*bits".to_string()))?;
                    if prod % 32 != 0 {
                        return Err(NgramTableError::PackedMismatch(format!(
                            "tensor {name:?} width {embedding_width} bits {} not divisible by 32",
                            quant.bits
                        )));
                    }
                    let packed = prod / 32;
                    let packed_usize = usize::try_from(packed)
                        .map_err(|_| NgramTableError::Overflow("packed cols".to_string()))?;
                    if weight_meta.cols != packed_usize {
                        return Err(NgramTableError::PackedMismatch(format!(
                            "tensor {name:?} packed cols {} != expected {packed_usize}",
                            weight_meta.cols
                        )));
                    }
                    if !embedding_width.is_multiple_of(quant.group_size as usize) {
                        return Err(NgramTableError::WidthMismatch {
                            name: name.clone(),
                            expected: embedding_width,
                            found: weight_meta.cols,
                        });
                    }
                    let num_groups = embedding_width / (quant.group_size as usize);
                    let base = name
                        .strip_suffix(".weight")
                        .ok_or_else(|| NgramTableError::InvalidName(name.clone()))?;
                    let scales_name = format!("{base}.scales");
                    let biases_name = format!("{base}.biases");
                    let (scales_file, biases_file) = match &weight_map {
                        Some(map) => {
                            let s = map.get(&scales_name).ok_or_else(|| {
                                NgramTableError::Sidecar(format!(
                                    "tensor {scales_name:?} missing from index"
                                ))
                            })?;
                            let b = map.get(&biases_name).ok_or_else(|| {
                                NgramTableError::Sidecar(format!(
                                    "tensor {biases_name:?} missing from index"
                                ))
                            })?;
                            (
                                join_inside_root(root, &canonical_root, Path::new(s))?,
                                join_inside_root(root, &canonical_root, Path::new(b))?,
                            )
                        }
                        None => (weight_file.clone(), weight_file.clone()),
                    };
                    let scales_meta = readers
                        .get(&scales_file)
                        .and_then(|r| r.tensor_meta(&scales_name))
                        .ok_or_else(|| {
                            NgramTableError::Sidecar(format!(
                                "tensor {scales_name:?} missing after open"
                            ))
                        })?;
                    let biases_meta = readers
                        .get(&biases_file)
                        .and_then(|r| r.tensor_meta(&biases_name))
                        .ok_or_else(|| {
                            NgramTableError::Sidecar(format!(
                                "tensor {biases_name:?} missing after open"
                            ))
                        })?;
                    if scales_meta.rows != weight_meta.rows || biases_meta.rows != weight_meta.rows
                    {
                        return Err(NgramTableError::Sidecar(format!(
                            "tensor {name:?} sidecar rows differ from weight rows"
                        )));
                    }
                    if scales_meta.cols != num_groups || biases_meta.cols != num_groups {
                        return Err(NgramTableError::Sidecar(format!(
                            "tensor {name:?} sidecar cols differ from groups {num_groups}"
                        )));
                    }
                    let ok_dtype = |d: MlxDtype| {
                        matches!(
                            d,
                            MlxDtype::Float32 | MlxDtype::Float16 | MlxDtype::Bfloat16
                        )
                    };
                    if !ok_dtype(scales_meta.dtype) || !ok_dtype(biases_meta.dtype) {
                        return Err(NgramTableError::Sidecar(format!(
                            "tensor {name:?} sidecar dtype must be dense float"
                        )));
                    }
                    if scales_meta.dtype != biases_meta.dtype {
                        return Err(NgramTableError::Sidecar(format!(
                            "tensor {name:?} scales/biases dtype differ"
                        )));
                    }
                    ShardEncoding::Quantized {
                        bits: quant.bits,
                        group_size: quant.group_size,
                        scales_name,
                        scales_file,
                        scales_meta: Box::new(scales_meta),
                        biases_name,
                        biases_file,
                        biases_meta: Box::new(biases_meta),
                    }
                }
                (None, Some(want)) => {
                    if weight_meta.dtype != want {
                        return Err(NgramTableError::DtypeMismatch {
                            name: name.clone(),
                            expected: want,
                            found: weight_meta.dtype,
                        });
                    }
                    if weight_meta.cols != embedding_width {
                        return Err(NgramTableError::WidthMismatch {
                            name: name.clone(),
                            expected: embedding_width,
                            found: weight_meta.cols,
                        });
                    }
                    ShardEncoding::Dense
                }
                _ => {
                    return Err(NgramTableError::Internal(
                        "shard encoding missing".to_string(),
                    ));
                }
            };
            let shard = Shard {
                global_start: total_rows,
                rows: rows_u64,
                weight_name: name,
                weight_file,
                weight_meta,
                encoding,
            };
            total_rows = total_rows
                .checked_add(rows_u64)
                .ok_or_else(|| NgramTableError::Overflow("total rows".to_string()))?;
            shards.push(shard);
        }
        Ok(Self {
            shards,
            readers,
            total_rows,
            embedding_width,
            max_gather_bytes,
        })
    }

    pub fn rows(&self) -> u64 {
        self.total_rows
    }

    pub fn payload_bytes_read(&self) -> u64 {
        let mut total: u64 = 0;
        for reader in self.readers.values() {
            total = total.saturating_add(reader.payload_bytes_read());
        }
        total
    }

    fn shard_for_row(&self, row: u64) -> Result<usize, NgramTableError> {
        let mut lo = 0usize;
        let mut hi = self.shards.len();
        let mut pick: Option<usize> = None;
        while lo < hi {
            let mid = lo.saturating_add(hi.saturating_sub(lo) / 2);
            let shard = self
                .shards
                .get(mid)
                .ok_or_else(|| NgramTableError::Internal("shard index missing".to_string()))?;
            if row < shard.global_start {
                hi = mid;
            } else {
                pick = Some(mid);
                lo = mid.saturating_add(1);
            }
        }
        let idx = pick.ok_or(NgramTableError::RowBounds {
            row,
            rows: self.total_rows,
            position: 0,
        })?;
        let shard = self
            .shards
            .get(idx)
            .ok_or_else(|| NgramTableError::Internal("shard pick missing".to_string()))?;
        let end = shard
            .global_start
            .checked_add(shard.rows)
            .ok_or_else(|| NgramTableError::Overflow("shard end".to_string()))?;
        if row >= end {
            return Err(NgramTableError::RowBounds {
                row,
                rows: self.total_rows,
                position: 0,
            });
        }
        Ok(idx)
    }

    pub fn gather(&self, row_ids: &[u64]) -> Result<MlxArray, String> {
        self.gather_inner(row_ids).map_err(|e| e.to_string())
    }

    fn gather_inner(&self, row_ids: &[u64]) -> Result<MlxArray, NgramTableError> {
        let count = row_ids.len();
        if count > i32::MAX as usize {
            return Err(NgramTableError::TooManyRows(count));
        }
        let width = self.embedding_width;
        let width_i32 = i32::try_from(width)
            .map_err(|_| NgramTableError::Overflow("width to i32".to_string()))?;
        let count_i32 = i32::try_from(count)
            .map_err(|_| NgramTableError::Overflow("count to i32".to_string()))?;
        if count == 0 {
            let empty: Vec<u8> = Vec::new();
            return Ok(MlxArray::from_raw_data(
                empty.as_ptr(),
                0,
                &[0, width_i32],
                MlxDtype::Float32,
            ));
        }
        for (pos, row) in row_ids.iter().enumerate() {
            if *row >= self.total_rows {
                return Err(NgramTableError::RowBounds {
                    row: *row,
                    rows: self.total_rows,
                    position: pos,
                });
            }
        }
        let elems = count
            .checked_mul(width)
            .ok_or_else(|| NgramTableError::Overflow("gather elements".to_string()))?;
        let output_bytes = elems
            .checked_mul(4)
            .ok_or_else(|| NgramTableError::Overflow("output bytes".to_string()))?;
        let mut staging: usize = 0;
        let mut assignment: Vec<usize> = Vec::new();
        assignment
            .try_reserve_exact(count)
            .map_err(|_| NgramTableError::Overflow("assignment reserve".to_string()))?;
        for row in row_ids {
            let idx = self.shard_for_row(*row)?;
            let shard = self
                .shards
                .get(idx)
                .ok_or_else(|| NgramTableError::Internal("assigned shard missing".to_string()))?;
            staging = staging
                .checked_add(shard.staging_row_bytes())
                .ok_or_else(|| NgramTableError::Overflow("staging bytes".to_string()))?;
            assignment.push(idx);
        }
        let total = output_bytes
            .checked_add(staging)
            .ok_or_else(|| NgramTableError::Overflow("total bytes".to_string()))?;
        if total > self.max_gather_bytes {
            return Err(NgramTableError::Budget {
                needed: total,
                budget: self.max_gather_bytes,
            });
        }
        let mut per_shard: Vec<Vec<(usize, u64)>> = Vec::new();
        per_shard
            .try_reserve_exact(self.shards.len())
            .map_err(|_| NgramTableError::Overflow("per-shard reserve".to_string()))?;
        for _ in 0..self.shards.len() {
            per_shard.push(Vec::new());
        }
        for (pos, row) in row_ids.iter().enumerate() {
            let idx = *assignment
                .get(pos)
                .ok_or_else(|| NgramTableError::Internal("assignment missing".to_string()))?;
            let shard = self
                .shards
                .get(idx)
                .ok_or_else(|| NgramTableError::Internal("shard missing".to_string()))?;
            let local = row
                .checked_sub(shard.global_start)
                .ok_or_else(|| NgramTableError::Overflow("local row".to_string()))?;
            let bucket = per_shard
                .get_mut(idx)
                .ok_or_else(|| NgramTableError::Internal("bucket missing".to_string()))?;
            bucket.push((pos, local));
        }
        let mut out: Vec<f32> = Vec::new();
        out.try_reserve_exact(elems)
            .map_err(|_| NgramTableError::Overflow("output reserve".to_string()))?;
        out.resize(elems, 0.0);
        for (shard_idx, bucket) in per_shard.iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let shard = self
                .shards
                .get(shard_idx)
                .ok_or_else(|| NgramTableError::Internal("gather shard missing".to_string()))?;
            let mut local_ids: Vec<u64> = Vec::new();
            local_ids
                .try_reserve_exact(bucket.len())
                .map_err(|_| NgramTableError::Overflow("local ids".to_string()))?;
            for (_, local) in bucket {
                local_ids.push(*local);
            }
            let block = self.gather_shard_block(shard, &local_ids)?;
            mlx_sys::try_eval(&[&block]).map_err(NgramTableError::Gather)?;
            if block.dtype() != MlxDtype::Float32 {
                return Err(NgramTableError::Gather("block is not F32".to_string()));
            }
            let shape = block.shape();
            let want_rows = i32::try_from(bucket.len())
                .map_err(|_| NgramTableError::Overflow("bucket rows".to_string()))?;
            if shape.len() != 2 || shape[0] != want_rows || shape[1] != width_i32 {
                return Err(NgramTableError::Gather(format!(
                    "block shape {shape:?} != [{}, {}]",
                    bucket.len(),
                    width
                )));
            }
            let data = block.data_f32();
            let need = bucket
                .len()
                .checked_mul(width)
                .ok_or_else(|| NgramTableError::Overflow("block elems".to_string()))?;
            if data.len() != need {
                return Err(NgramTableError::Gather("block length mismatch".to_string()));
            }
            for (slot, (caller_pos, _)) in bucket.iter().enumerate() {
                let src_start = slot
                    .checked_mul(width)
                    .ok_or_else(|| NgramTableError::Overflow("src start".to_string()))?;
                let src_end = src_start
                    .checked_add(width)
                    .ok_or_else(|| NgramTableError::Overflow("src end".to_string()))?;
                let dst_start = caller_pos
                    .checked_mul(width)
                    .ok_or_else(|| NgramTableError::Overflow("dst start".to_string()))?;
                let dst_end = dst_start
                    .checked_add(width)
                    .ok_or_else(|| NgramTableError::Overflow("dst end".to_string()))?;
                let src = data
                    .get(src_start..src_end)
                    .ok_or_else(|| NgramTableError::Internal("src slice".to_string()))?;
                let dst = out
                    .get_mut(dst_start..dst_end)
                    .ok_or_else(|| NgramTableError::Internal("dst slice".to_string()))?;
                dst.copy_from_slice(src);
            }
        }
        let byte_len = elems
            .checked_mul(4)
            .ok_or_else(|| NgramTableError::Overflow("final bytes".to_string()))?;
        let out_array = MlxArray::from_raw_data(
            out.as_ptr() as *const u8,
            byte_len,
            &[count_i32, width_i32],
            MlxDtype::Float32,
        );
        mlx_sys::try_eval(&[&out_array]).map_err(NgramTableError::Gather)?;
        Ok(out_array)
    }

    fn gather_shard_block(
        &self,
        shard: &Shard,
        local_ids: &[u64],
    ) -> Result<MlxArray, NgramTableError> {
        let weight_reader = self
            .readers
            .get(&shard.weight_file)
            .ok_or_else(|| NgramTableError::Internal("weight reader missing".to_string()))?;
        match &shard.encoding {
            ShardEncoding::Dense => {
                let rows = weight_reader
                    .gather_rows(&shard.weight_name, local_ids)
                    .map_err(NgramTableError::Gather)?;
                Ok(mlx_sys::astype(&rows, MlxDtype::Float32, None))
            }
            ShardEncoding::Quantized {
                bits,
                group_size,
                scales_name,
                scales_file,
                biases_name,
                biases_file,
                ..
            } => {
                let scales_reader = self.readers.get(scales_file).ok_or_else(|| {
                    NgramTableError::Internal("scales reader missing".to_string())
                })?;
                let biases_reader = self.readers.get(biases_file).ok_or_else(|| {
                    NgramTableError::Internal("biases reader missing".to_string())
                })?;
                let w = weight_reader
                    .gather_rows(&shard.weight_name, local_ids)
                    .map_err(NgramTableError::Gather)?;
                let s = scales_reader
                    .gather_rows(scales_name, local_ids)
                    .map_err(NgramTableError::Gather)?;
                let b = biases_reader
                    .gather_rows(biases_name, local_ids)
                    .map_err(NgramTableError::Gather)?;
                let group_i32 = i32::try_from(*group_size)
                    .map_err(|_| NgramTableError::Overflow("group i32".to_string()))?;
                let bits_i32 = i32::try_from(*bits)
                    .map_err(|_| NgramTableError::Overflow("bits i32".to_string()))?;
                let deq =
                    mlx_sys::dequantize(&w, &s, Some(&b), Some(group_i32), Some(bits_i32), None);
                Ok(mlx_sys::astype(&deq, MlxDtype::Float32, None))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_engine_core::NativeTensorQuantization;

    fn test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ax_ngram_table_{}_{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn u32_to_bytes(values: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn write_st(path: &Path, tensors: Vec<(String, String, Vec<u64>, Vec<u8>)>) {
        let mut header = serde_json::Map::new();
        let mut offset: u64 = 0;
        let mut payload: Vec<u8> = Vec::new();
        for (name, dtype, shape, bytes) in &tensors {
            let start = offset;
            let end = start + bytes.len() as u64;
            offset = end;
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            );
            payload.extend_from_slice(bytes);
        }
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut file = std::fs::File::create(path).unwrap();
        use std::io::Write;
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&payload).unwrap();
        file.sync_all().unwrap();
    }

    fn dense_spec(name: &str, layer: u32, file: &str, rows: u64, cols: u64) -> NativeTensorSpec {
        NativeTensorSpec {
            name: name.to_string(),
            role: NativeTensorRole::NgramEmbedding,
            layer_index: Some(layer),
            dtype: NativeTensorDataType::F32,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape: vec![rows, cols],
            file: PathBuf::from(file),
            offset_bytes: 0,
            length_bytes: 0,
        }
    }

    fn quant_spec(
        name: &str,
        layer: u32,
        file: &str,
        rows: u64,
        packed: u64,
        bits: u32,
        group: u32,
    ) -> NativeTensorSpec {
        NativeTensorSpec {
            name: name.to_string(),
            role: NativeTensorRole::NgramEmbedding,
            layer_index: Some(layer),
            dtype: NativeTensorDataType::U32,
            source_tensor_type: None,
            source_quantized: true,
            quantization: Some(NativeTensorQuantization {
                mode: "affine".to_string(),
                group_size: group,
                bits,
            }),
            quantized_source: None,
            shape: vec![rows, packed],
            file: PathBuf::from(file),
            offset_bytes: 0,
            length_bytes: 0,
        }
    }

    #[test]
    fn dense_two_shards_preserve_order_and_count_bytes() {
        let dir = test_dir("dense_order");
        let width = 4usize;
        let shard0: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];
        let shard1: Vec<f32> = vec![20.0, 21.0, 22.0, 23.0, 30.0, 31.0, 32.0, 33.0];
        let n0 = "model.layers.0.ple.ngram_embedding.shard_0.weight".to_string();
        let n1 = "model.layers.0.ple.ngram_embedding.shard_1.weight".to_string();
        write_st(
            &dir.join("s0.safetensors"),
            vec![(
                n0.clone(),
                "F32".to_string(),
                vec![2, 4],
                f32_to_bytes(&shard0),
            )],
        );
        write_st(
            &dir.join("s1.safetensors"),
            vec![(
                n1.clone(),
                "F32".to_string(),
                vec![2, 4],
                f32_to_bytes(&shard1),
            )],
        );
        let specs = vec![
            dense_spec(&n0, 0, "s0.safetensors", 2, 4),
            dense_spec(&n1, 0, "s1.safetensors", 2, 4),
        ];
        let table = NgramTable::open(&dir, &specs, 0, 2, width, 1 << 20).unwrap();
        assert_eq!(table.rows(), 4);
        assert_eq!(table.payload_bytes_read(), 0);
        let rows = vec![3u64, 0, 3, 1, 2];
        let out = table.gather(&rows).unwrap();
        mlx_sys::eval(&[&out]);
        assert_eq!(out.shape(), vec![5, 4]);
        assert_eq!(out.dtype(), MlxDtype::Float32);
        let got = out.data_f32().to_vec();
        let full = [shard0.clone(), shard1.clone()].concat();
        let mut want = Vec::new();
        for r in &rows {
            let start = (*r as usize) * width;
            want.extend_from_slice(&full[start..start + width]);
        }
        assert_eq!(got, want);
        assert_eq!(table.payload_bytes_read(), (5 * 4 * 4) as u64);
    }

    fn quant_fixture(
        bits: u32,
        group: u32,
        width: usize,
    ) -> (Vec<u32>, Vec<f32>, Vec<f32>, usize, usize) {
        let rows = 4usize;
        let values: Vec<f32> = (0..(rows * width))
            .map(|i| ((i % 13) as f32) * 0.25 - 1.5)
            .collect();
        let dense = MlxArray::from_raw_data(
            values.as_ptr() as *const u8,
            values.len() * 4,
            &[rows as i32, width as i32],
            MlxDtype::Float32,
        );
        let parts = mlx_sys::quantize(
            &dense,
            Some(group as i32),
            Some(bits as i32),
            mlx_sys::MlxQuantizationMode::Affine,
            None,
            None,
        );
        assert_eq!(parts.len(), 3);
        mlx_sys::eval(&[&parts[0], &parts[1], &parts[2]]);
        let packed_cols = parts[0].shape()[1] as usize;
        let groups = parts[1].shape()[1] as usize;
        let packed = parts[0].data_u32().to_vec();
        let scales = parts[1].data_f32().to_vec();
        let biases = parts[2].data_f32().to_vec();
        (packed, scales, biases, packed_cols, groups)
    }

    #[test]
    fn affine_parity_across_shards_for_2_4_6_8_bit() {
        for bits in [2u32, 4, 6, 8] {
            let group = 32u32;
            let width = 64usize;
            let (packed, scales, biases, packed_cols, groups) = quant_fixture(bits, group, width);
            assert_eq!(packed_cols, width * (bits as usize) / 32);
            assert_eq!(groups, width / (group as usize));
            let dir = test_dir(&format!("q{bits}"));
            let n0 = "model.layers.1.ple.ngram_embedding.shard_0.weight".to_string();
            let n1 = "model.layers.1.ple.ngram_embedding.shard_1.weight".to_string();
            let b0 = n0.strip_suffix(".weight").unwrap();
            let b1 = n1.strip_suffix(".weight").unwrap();
            let half = 2usize;
            let pack_row = packed_cols;
            let side_row = groups;
            let p0 = packed[0..half * pack_row].to_vec();
            let p1 = packed[half * pack_row..].to_vec();
            let s0 = scales[0..half * side_row].to_vec();
            let s1 = scales[half * side_row..].to_vec();
            let q0 = biases[0..half * side_row].to_vec();
            let q1 = biases[half * side_row..].to_vec();
            write_st(
                &dir.join("a.safetensors"),
                vec![
                    (
                        n0.clone(),
                        "U32".to_string(),
                        vec![2, packed_cols as u64],
                        u32_to_bytes(&p0),
                    ),
                    (
                        format!("{b0}.scales"),
                        "F32".to_string(),
                        vec![2, groups as u64],
                        f32_to_bytes(&s0),
                    ),
                    (
                        format!("{b0}.biases"),
                        "F32".to_string(),
                        vec![2, groups as u64],
                        f32_to_bytes(&q0),
                    ),
                ],
            );
            write_st(
                &dir.join("b.safetensors"),
                vec![
                    (
                        n1.clone(),
                        "U32".to_string(),
                        vec![2, packed_cols as u64],
                        u32_to_bytes(&p1),
                    ),
                    (
                        format!("{b1}.scales"),
                        "F32".to_string(),
                        vec![2, groups as u64],
                        f32_to_bytes(&s1),
                    ),
                    (
                        format!("{b1}.biases"),
                        "F32".to_string(),
                        vec![2, groups as u64],
                        f32_to_bytes(&q1),
                    ),
                ],
            );
            let specs = vec![
                quant_spec(&n0, 1, "a.safetensors", 2, packed_cols as u64, bits, group),
                quant_spec(&n1, 1, "b.safetensors", 2, packed_cols as u64, bits, group),
            ];
            let table = NgramTable::open(&dir, &specs, 1, 2, width, 1 << 24).unwrap();
            assert_eq!(table.rows(), 4);
            assert_eq!(table.payload_bytes_read(), 0);
            let full_w = MlxArray::from_raw_data(
                packed.as_ptr() as *const u8,
                packed.len() * 4,
                &[4, packed_cols as i32],
                MlxDtype::Uint32,
            );
            let full_s = MlxArray::from_raw_data(
                scales.as_ptr() as *const u8,
                scales.len() * 4,
                &[4, groups as i32],
                MlxDtype::Float32,
            );
            let full_b = MlxArray::from_raw_data(
                biases.as_ptr() as *const u8,
                biases.len() * 4,
                &[4, groups as i32],
                MlxDtype::Float32,
            );
            let full_d = mlx_sys::dequantize(
                &full_w,
                &full_s,
                Some(&full_b),
                Some(group as i32),
                Some(bits as i32),
                None,
            );
            let full_f = mlx_sys::astype(&full_d, MlxDtype::Float32, None);
            mlx_sys::eval(&[&full_f]);
            let full = full_f.data_f32().to_vec();
            let rows = vec![3u64, 0, 3, 1, 2];
            let out = table.gather(&rows).unwrap();
            mlx_sys::eval(&[&out]);
            assert_eq!(out.shape(), vec![5, width as i32]);
            let got = out.data_f32().to_vec();
            let mut want = Vec::new();
            for r in &rows {
                let start = (*r as usize) * width;
                want.extend_from_slice(&full[start..start + width]);
            }
            assert_eq!(got, want, "parity failed for {bits}-bit");
            let per_row = packed_cols * 4 + groups * 4 + groups * 4;
            assert_eq!(table.payload_bytes_read(), (rows.len() * per_row) as u64);
        }
    }

    fn tiny_dense_dir(name: &str) -> (PathBuf, Vec<NativeTensorSpec>) {
        let dir = test_dir(name);
        let n0 = "model.layers.2.ple.ngram_embedding.shard_0.weight".to_string();
        let n1 = "model.layers.2.ple.ngram_embedding.shard_1.weight".to_string();
        let r0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let r1: Vec<f32> = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        write_st(
            &dir.join("a.safetensors"),
            vec![(n0.clone(), "F32".to_string(), vec![2, 4], f32_to_bytes(&r0))],
        );
        write_st(
            &dir.join("b.safetensors"),
            vec![(n1.clone(), "F32".to_string(), vec![2, 4], f32_to_bytes(&r1))],
        );
        let specs = vec![
            dense_spec(&n0, 2, "a.safetensors", 2, 4),
            dense_spec(&n1, 2, "b.safetensors", 2, 4),
        ];
        (dir, specs)
    }

    #[test]
    fn bounds_budget_and_layout_errors_are_explicit() {
        let (dir, specs) = tiny_dense_dir("errs");
        let table = NgramTable::open(&dir, &specs, 2, 2, 4, 1 << 20).unwrap();
        assert_eq!(table.payload_bytes_read(), 0);
        let before = table.payload_bytes_read();
        assert!(table.gather(&[4]).is_err());
        assert_eq!(table.payload_bytes_read(), before);
        let tight = NgramTable::open(&dir, &specs, 2, 2, 4, 8).unwrap();
        assert!(tight.gather(&[0]).is_err());
        assert_eq!(tight.payload_bytes_read(), 0);
        let dup = vec![specs[0].clone(), specs[0].clone()];
        assert!(NgramTable::open(&dir, &dup, 2, 2, 4, 1 << 20).is_err());
        let gap_spec = dense_spec(
            "model.layers.2.ple.ngram_embedding.shard_5.weight",
            2,
            "a.safetensors",
            2,
            4,
        );
        let gap = vec![specs[0].clone(), gap_spec];
        assert!(NgramTable::open(&dir, &gap, 2, 2, 4, 1 << 20).is_err());
        let mixed_spec = dense_spec(
            "model.layers.2.ple.ngram_embedding.weight",
            2,
            "a.safetensors",
            2,
            4,
        );
        let mixed = vec![specs[0].clone(), mixed_spec];
        assert!(NgramTable::open(&dir, &mixed, 2, 2, 4, 1 << 20).is_err());
        let mut bad = specs[0].clone();
        bad.dtype = NativeTensorDataType::U8;
        assert!(NgramTable::open(&dir, &[bad], 2, 1, 4, 1 << 20).is_err());
        let mut mx = quant_spec(
            "model.layers.2.ple.ngram_embedding.shard_0.weight",
            2,
            "a.safetensors",
            2,
            2,
            4,
            32,
        );
        mx.quantization = Some(NativeTensorQuantization {
            mode: "mxfp4".to_string(),
            group_size: 32,
            bits: 4,
        });
        let err = NgramTable::open(&dir, std::slice::from_ref(&mx), 2, 1, 64, 1 << 20)
            .err()
            .unwrap();
        assert!(
            err.contains("unsupported"),
            "want explicit unsupported, got {err}"
        );
    }

    #[test]
    fn retained_output_survives_next_gather() {
        let (dir, specs) = tiny_dense_dir("retain");
        let table = NgramTable::open(&dir, &specs, 2, 2, 4, 1 << 20).unwrap();
        let first = table.gather(&[0, 3]).unwrap();
        mlx_sys::eval(&[&first]);
        let snapshot = first.data_f32().to_vec();
        let second = table.gather(&[1, 1, 2]).unwrap();
        mlx_sys::eval(&[&second]);
        assert_eq!(first.data_f32().to_vec(), snapshot);
        assert_eq!(second.shape(), vec![3, 4]);
    }
}
