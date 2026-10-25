//! Indexed sparse SDPA for Flash Next (`qwen4_exp`) QSA prefill.
//!
//! The gathered path ([`super::qwen4_exp_attention::attend_selected`]) copies
//! every query's selected K/V rows into a padded
//! `[queries, selected, kv_heads, head_dim]` tile and then attends it. For a
//! prefill chunk that tile is the dominant transient: a 2048-query chunk with
//! the shipped geometry selects about 2051 rows per query, so one layer's
//! gather is `2 * 2048 * 2051 * 2 * 256 * 2 B` ≈ 8.6 GiB of copies (priced per
//! sub-batch against [`super::qwen4_exp_attention::BATCHED_ATTENTION_BYTES`]).
//!
//! This module attends the same selected rows without that replication. The
//! kernel reads the ordered selection directly out of the sealed prefix and
//! the tail of the request's K/V history, stages one K/V row at a time into
//! threadgroup memory, and keeps a float32 online-softmax accumulator. Its
//! only transients are the packed index rows (`queries × selected × 4 B`) and
//! the output tile (`queries × q_heads × head_dim`), so no `queries × selected`
//! K/V tile is ever materialised.
//!
//! The device code lives in `metal/kernels/flash_next_qsa_indexed.metal`; this
//! module only binds it through `crates/mlx-sys` and stays fail-closed: any
//! geometry, dtype or segment shape the kernel does not cover returns
//! `Ok(None)` and the caller keeps the gathered path.
//!
//! Exactness: selection order, causal membership, the softmax scale, the GQA
//! mapping and the dtype transition are preserved, so softmax runs over the
//! same value set the gathered path attends to. The reduction order and the
//! online rescaling differ, so the result is not bit-identical to MLX's fused
//! SDPA. That is why this path is opt-in
//! (`AX_MLX_FLASH_NEXT_QSA_INDEXED_PREFILL`).

use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, LazyKernel, MlxArray, MlxDtype, contiguous, reshape,
    transpose, zeros,
};

use super::qwen4_exp_attention::{BATCHED_ATTENTION_MIN_QUERIES, Qwen4ExpAttentionError};
use super::qwen4_exp_kv_history::KvHistory;
use crate::qwen4_exp_qsa::QsaSelection;

type Result<T> = std::result::Result<T, Qwen4ExpAttentionError>;

/// Device code, kept beside the other native AX kernels. See the file header
/// there for the buffer contract.
const INDEXED_SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../metal/kernels/flash_next_qsa_indexed.metal"
));

/// `AX_MLX_FLASH_NEXT_QSA_INDEXED_PREFILL` — attend QSA prefill chunks by
/// reading the selected rows directly out of the segmented K/V history instead
/// of gathering a padded `[queries, selected, kv_heads, head_dim]` tile.
/// Default OFF; the gathered path is unchanged.
static INDEXED: LazyKernel = LazyKernel::new(
    "ax_flash_next_qsa_indexed_attention_v1",
    &[
        "q", "k_sealed", "k_tail", "v_sealed", "v_tail", "indices", "lengths", "scales", "params",
    ],
    &["out"],
    INDEXED_SOURCE,
    "",
    true,
);

/// Selected rows one query may keep. This bounds the host-side packed index
/// tile, not the kernel: the kernel walks each row once in selection order. A
/// selection wider than this is one the gathered path serves instead, so a
/// larger configured budget costs this optimisation, never correctness.
const MAX_SELECTED_WIDTH: i32 = 8192;

fn enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = INDEXED_OVERRIDE.with(std::cell::Cell::get) {
        return value;
    }
    // Unit tests keep the gathered path unless they opt in with
    // [`with_indexed_attention`]: the pinned numeric fixtures in this crate are
    // gathered-path comparisons.
    if cfg!(test) {
        return false;
    }
    crate::fastpath::flash_next_qsa_indexed_prefill_enabled()
}

#[cfg(test)]
thread_local! {
    static INDEXED_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static INDEXED_DISPATCHES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Run `action` with the indexed path forced on or off on this thread.
#[cfg(test)]
pub(crate) fn with_indexed_attention<T>(enabled: bool, action: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            INDEXED_OVERRIDE.with(|value| value.set(self.0));
        }
    }
    let _restore = Restore(INDEXED_OVERRIDE.with(|value| value.replace(Some(enabled))));
    action()
}

#[cfg(test)]
fn dispatches() -> u64 {
    INDEXED_DISPATCHES.with(std::cell::Cell::get)
}

/// Kernel dispatches issued on this thread so far, so a real-pack comparison
/// can prove the indexed arm actually ran instead of silently falling back.
#[cfg(test)]
pub(crate) fn dispatch_count() -> u64 {
    dispatches()
}

/// One chunk's selection packed into the row-major index tile.
struct PackedSelection {
    indices: MlxArray,
    lengths: MlxArray,
    width: i32,
}

/// Attend `queries` over the K/V rows `selection` chose, without gathering
/// them. Returns `Ok(None)` whenever this path does not apply, and the caller
/// keeps the gathered path.
///
/// `queries` is `[batch, q_heads, seq, head_dim]`. The kernel covers a single
/// batch row, multi-query prefill chunks, and the geometry admitted by
/// [`kernel_geometry_ok`].
#[allow(clippy::too_many_arguments)]
pub(super) fn attend_selected_indexed(
    queries: &MlxArray,
    keys: &KvHistory,
    values: &KvHistory,
    selection: &QsaSelection,
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
    compress_ratio: i32,
    scale: f32,
) -> Result<Option<MlxArray>> {
    if !enabled() {
        return Ok(None);
    }
    // Decode and verifier windows keep their existing arms: this path is
    // admitted for prefill chunks only. A singleton device selection carries
    // one exact-length row rather than the per-query tile the kernel walks.
    if selection.device_tokens().is_some() && !selection.device_multi_query() {
        return Ok(None);
    }
    let q_shape = queries.shape();
    if q_shape.len() != 4 || q_shape[0] != 1 {
        return Ok(None);
    }
    let (seq, dim) = (q_shape[2], q_shape[3]);
    if seq < BATCHED_ATTENTION_MIN_QUERIES || dim != head_dim || dim <= 0 {
        return Ok(None);
    }
    if !kernel_geometry_ok(q_heads, kv_heads, dim, queries.dtype()) {
        return Ok(None);
    }
    if keys.token_count() != values.token_count() {
        return Ok(None);
    }
    if keys.sealed_rows() != values.sealed_rows() {
        return Ok(None);
    }
    let total_rows = match i32::try_from(keys.token_count()) {
        Ok(total) => total,
        Err(_) => return Ok(None),
    };
    let Some(segments) = split_segments(keys, values, total_rows, kv_heads, head_dim) else {
        return Ok(None);
    };
    let Some(packed) = pack_selection(selection, seq, total_rows, compress_ratio)? else {
        return Ok(None);
    };
    let Some(kernel) = INDEXED.get() else {
        return Ok(None);
    };
    launch_indexed(
        kernel,
        queries,
        &segments,
        &packed.indices,
        &packed.lengths,
        q_heads,
        kv_heads,
        dim,
        seq,
        total_rows,
        packed.width,
        scale,
    )
}

/// Launch the indexed-attention pipeline for one packed selection.
///
/// The template holds only model geometry; the per-chunk row counts and the
/// selection width ride in a `params` buffer. This is the one launch the
/// request path and the load-time warm-up share, so a warm-up with the model's
/// own geometry compiles exactly the pipeline production dispatches.
#[allow(clippy::too_many_arguments)]
fn launch_indexed(
    kernel: &mlx_sys::MlxMetalKernel,
    queries: &MlxArray,
    segments: &Segments,
    indices: &MlxArray,
    lengths: &MlxArray,
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
    seq: i32,
    total_rows: i32,
    width: i32,
    scale: f32,
) -> Result<Option<MlxArray>> {
    let q_rows = contiguous(&transpose(queries, &[2, 1, 0, 3], None), None);
    let scale = MlxArray::from_raw_data(
        (&scale as *const f32).cast(),
        std::mem::size_of::<f32>(),
        &[1],
        MlxDtype::Float32,
    );
    // The row counts and the selection width change with the request position
    // and the chosen block count, so they travel in a buffer: the compiled
    // pipeline is specialised only by the model geometry, which lets the
    // load-time warm-up cover every later chunk.
    let params_data = [segments.sealed_rows, total_rows, width];
    let params = MlxArray::from_raw_data(
        params_data.as_ptr().cast(),
        std::mem::size_of_val(params_data.as_slice()),
        &[3],
        MlxDtype::Int32,
    );
    let template = [
        KernelTemplateArg::Dtype {
            name: "T",
            dtype: queries.dtype(),
        },
        KernelTemplateArg::Int {
            name: "HeadDim",
            value: head_dim,
        },
        KernelTemplateArg::Int {
            name: "Heads",
            value: q_heads,
        },
        KernelTemplateArg::Int {
            name: "KvHeads",
            value: kv_heads,
        },
        KernelTemplateArg::Int {
            name: "Group",
            value: q_heads / kv_heads,
        },
    ];
    let outputs = match kernel.try_apply_with_template(
        &[
            &q_rows,
            &segments.k_sealed,
            &segments.k_tail,
            &segments.v_sealed,
            &segments.v_tail,
            indices,
            lengths,
            &scale,
            &params,
        ],
        &[KernelOutputSpec {
            shape: vec![seq, q_heads, head_dim],
            dtype: queries.dtype(),
        }],
        &template,
        (q_heads / kv_heads * 32, seq, kv_heads),
        (q_heads / kv_heads * 32, 1, 1),
        None,
    ) {
        Ok(outputs) => outputs,
        // A kernel that fails to apply falls back instead of failing the
        // request: the gathered path computes the same value set.
        Err(_) => return Ok(None),
    };
    #[cfg(test)]
    INDEXED_DISPATCHES.with(|counter| counter.set(counter.get().saturating_add(1)));
    let Some(out) = outputs.into_iter().next() else {
        return Ok(None);
    };
    // [seq, q_heads, dim] -> [q_heads, seq, dim] -> [1, q_heads, seq, dim],
    // the layout the caller's gate and output stages expect.
    Ok(Some(reshape(
        &transpose(&out, &[1, 0, 2], None),
        &[1, q_heads, seq, head_dim],
        None,
    )))
}

/// Whether model load should compile the indexed QSA prefill kernel.
///
/// Only Flash Next packs ever dispatch it, `AX_MLX_LOAD_KERNEL_WARMUP=0` skips
/// every load-time compile, and `AX_MLX_FLASH_NEXT_QSA_INDEXED_PREFILL` is
/// default OFF, so a stock load compiles nothing.
pub(crate) fn indexed_prefill_warmup_applies(
    qwen4_exp: bool,
    kernel_warmup: bool,
    indexed_prefill: bool,
) -> bool {
    qwen4_exp && kernel_warmup && indexed_prefill
}

/// Compile the production indexed-attention pipeline once, at load.
///
/// The template is the model geometry (`T`, `HeadDim`, `Heads`, `KvHeads`,
/// `Group`), so one launch with the config's own values compiles the same
/// pipeline every later prefill chunk reuses; the per-chunk row counts and the
/// selection width travel in a buffer, not in the specialization. Best-effort:
/// the caller logs `Err` and the request path keeps the gathered fallback.
pub(crate) fn warm_indexed_prefill_kernel(
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
    dtype: MlxDtype,
) -> std::result::Result<(), String> {
    if !enabled() {
        return Err("indexed QSA prefill warm-up skipped: the switch is off".to_string());
    }
    if !kernel_geometry_ok(q_heads, kv_heads, head_dim, dtype) {
        return Err(format!(
            "indexed QSA prefill warm-up geometry is unsupported: q_heads={q_heads}, \
             kv_heads={kv_heads}, head_dim={head_dim}, dtype={dtype:?}"
        ));
    }
    let Some(kernel) = INDEXED.get() else {
        return Err("indexed QSA prefill warm-up could not build the kernel".to_string());
    };
    // The request path admits prefill chunks only, so the probe is the smallest
    // chunk-shaped window. Every query keeps one row: the kernel walks a single
    // index and writes one value row per query, which is enough to build and
    // evaluate the whole pipeline. The probe goes through [`launch_indexed`],
    // the same launch the request path uses.
    let seq = BATCHED_ATTENTION_MIN_QUERIES;
    let rows = seq;
    let keys = zeros(&[rows, kv_heads, head_dim], dtype, None);
    let values = zeros(&[rows, kv_heads, head_dim], dtype, None);
    let segments = Segments {
        sealed_rows: rows,
        // Every probe token is below `sealed_rows`, so the kernel never reads
        // the tail bindings; they only have to be valid arrays.
        k_tail: keys.clone(),
        v_tail: values.clone(),
        k_sealed: keys,
        v_sealed: values,
    };
    let indices_data: Vec<i32> = (0..seq).collect();
    let indices = MlxArray::from_raw_data(
        indices_data.as_ptr().cast(),
        std::mem::size_of_val(indices_data.as_slice()),
        &[seq, 1],
        MlxDtype::Int32,
    );
    let lengths_data = vec![1_i32; seq as usize];
    let lengths = MlxArray::from_raw_data(
        lengths_data.as_ptr().cast(),
        std::mem::size_of_val(lengths_data.as_slice()),
        &[seq],
        MlxDtype::Int32,
    );
    let queries = zeros(&[1, q_heads, seq, head_dim], dtype, None);
    let out = launch_indexed(
        kernel,
        &queries,
        &segments,
        &indices,
        &lengths,
        q_heads,
        kv_heads,
        head_dim,
        seq,
        rows,
        1,
        1.0 / (head_dim as f32).sqrt(),
    )
    .map_err(|error| format!("indexed QSA prefill warm-up failed: {error}"))?
    .ok_or_else(|| "indexed QSA prefill warm-up declined the probe launch".to_string())?;
    mlx_sys::try_eval(&[&out])
        .map_err(|error| format!("indexed QSA prefill warm-up failed: {error}"))
}

/// Geometry the kernel is compiled for: one warp per query head of a kv-head
/// group (so `q_heads / kv_heads` warps per threadgroup) and a head dimension
/// that splits evenly across that warp with a bounded register footprint.
fn kernel_geometry_ok(q_heads: i32, kv_heads: i32, head_dim: i32, dtype: MlxDtype) -> bool {
    if kv_heads < 1 || q_heads < 1 || q_heads % kv_heads != 0 {
        return false;
    }
    let group = q_heads / kv_heads;
    if !(1..=32).contains(&group) {
        return false;
    }
    if head_dim < 32 || head_dim % 32 != 0 || head_dim / 32 > 16 {
        return false;
    }
    matches!(
        dtype,
        MlxDtype::Float16 | MlxDtype::Bfloat16 | MlxDtype::Float32
    )
}

/// K/V rows the kernel reads, split exactly like the history's segments.
///
/// A history with one logical segment is passed in both bindings and sets
/// `sealed_rows` to the whole history, so the kernel's
/// `token < SealedRows` test always picks the real array and never reads the
/// duplicate binding. Shapes are `[rows, kv_heads, head_dim]`: the kernel only
/// runs for a single batch row, so the batch axis is dropped here instead of in
/// the kernel.
///
/// Every binding is checked against the kernel template (`KvHeads`, `HeadDim`,
/// the dtype) and against the logical row count, because the kernel indexes a
/// row as `row * (KvHeads * HeadDim) + kv_head * HeadDim` and reads the tail as
/// `token - SealedRows`. A history whose arrays disagree with the claimed
/// geometry returns `None` and the caller keeps the gathered path.
struct Segments {
    sealed_rows: i32,
    k_sealed: MlxArray,
    k_tail: MlxArray,
    v_sealed: MlxArray,
    v_tail: MlxArray,
}

fn split_segments(
    keys: &KvHistory,
    values: &KvHistory,
    total_rows: i32,
    kv_heads: i32,
    head_dim: i32,
) -> Option<Segments> {
    let (k_sealed_src, k_tail_src) = keys.segments();
    let (v_sealed_src, v_tail_src) = values.segments();
    let k_sealed = batch_row(k_sealed_src.or(k_tail_src)?, kv_heads, head_dim)?;
    let v_sealed = batch_row(v_sealed_src.or(v_tail_src)?, kv_heads, head_dim)?;
    let k_tail = match k_tail_src {
        Some(segment) => Some(batch_row(segment, kv_heads, head_dim)?),
        None => None,
    };
    let v_tail = match v_tail_src {
        Some(segment) => Some(batch_row(segment, kv_heads, head_dim)?),
        None => None,
    };
    // `split` is `Some` only for a history with both physical segments, which
    // is exactly when the sealed binding holds a strict prefix.
    let split = keys.sealed_rows();
    let sealed_rows = split.unwrap_or(total_rows);
    let dtype = k_sealed.dtype();
    if sealed_rows < 0
        || sealed_rows > total_rows
        || k_sealed.shape()[0] != sealed_rows
        || v_sealed.shape()[0] != sealed_rows
        || v_sealed.dtype() != dtype
    {
        return None;
    }
    let (k_tail, v_tail) = match split {
        Some(_) => {
            let (k_tail, v_tail) = (k_tail?, v_tail?);
            // The kernel reads `[SealedRows, TotalRows)` from the tail, so the
            // two segments must account for exactly the logical history.
            let tail_rows = total_rows.checked_sub(sealed_rows)?;
            if k_tail.shape()[0] != tail_rows
                || v_tail.shape()[0] != tail_rows
                || k_tail.dtype() != dtype
                || v_tail.dtype() != dtype
            {
                return None;
            }
            (k_tail, v_tail)
        }
        // One logical segment: both bindings hold the whole history and the
        // kernel never reads the tail slot.
        None => (k_sealed.clone(), v_sealed.clone()),
    };
    Some(Segments {
        sealed_rows,
        k_sealed,
        k_tail,
        v_sealed,
        v_tail,
    })
}

/// `[1, rows, kv_heads, head_dim]` -> `[rows, kv_heads, head_dim]`, or `None`
/// unless the batch is 1 and the row geometry is the one the kernel template
/// was compiled for.
fn batch_row(segment: &MlxArray, kv_heads: i32, head_dim: i32) -> Option<MlxArray> {
    let shape = segment.shape();
    if shape.len() != 4
        || shape[0] != 1
        || shape[1] < 0
        || shape[2] != kv_heads
        || shape[3] != head_dim
    {
        return None;
    }
    Some(reshape(segment, &[shape[1], shape[2], shape[3]], None))
}

/// Pack the ordered selection into the row-major index tile the kernel reads.
///
/// Host selections are copied from their token lists after the block-then-tail
/// shape and range checks. Device selections are already an int32 tile on the
/// GPU in the same valid-prefix-then-`-1` layout, so they are read in place:
/// the per-query lengths come from the closed-form segment sizes the selector
/// keeps on the host, and no index row is copied back to the host.
///
/// Returns `Ok(None)` when the selection does not have the shape the kernel
/// assumes, or when a row is out of range: the gathered path then handles it
/// unchanged.
fn pack_selection(
    selection: &QsaSelection,
    seq: i32,
    total_rows: i32,
    compress_ratio: i32,
) -> Result<Option<PackedSelection>> {
    // A device selection has no host token lists (they would cost a read-back
    // of exactly the tile this path exists to avoid), so it takes its own arm.
    if selection.device_multi_query() {
        return Ok(pack_device_selection(selection, seq));
    }
    let mut rows: Vec<&[i32]> = Vec::with_capacity(seq as usize);
    let mut longest = 0usize;
    for s in 0..seq as usize {
        let chosen = selection.tokens_for_query(0, s);
        if chosen.is_empty() || !selection_shape_ok(chosen, compress_ratio) {
            return Ok(None);
        }
        if chosen.iter().any(|&token| token < 0 || token >= total_rows) {
            return Ok(None);
        }
        longest = longest.max(chosen.len());
        rows.push(chosen);
    }
    // A selection wider than this path packs is one the kernel would still
    // attend correctly, so fall back rather than fail the request: the switch
    // must never turn a chunk the gathered path serves into an error.
    let width = match i32::try_from(longest) {
        Ok(width) if width > 0 && width <= MAX_SELECTED_WIDTH => width,
        Ok(_) | Err(_) => return Ok(None),
    };
    let mut indices = vec![-1_i32; seq as usize * width as usize];
    let mut lengths = Vec::with_capacity(seq as usize);
    for (row, chosen) in rows.iter().enumerate() {
        let start = row * width as usize;
        indices[start..start + chosen.len()].copy_from_slice(chosen);
        lengths.push(chosen.len() as i32);
    }
    Ok(Some(PackedSelection {
        indices: MlxArray::from_raw_data(
            indices.as_ptr().cast(),
            std::mem::size_of_val(indices.as_slice()),
            &[seq, width],
            MlxDtype::Int32,
        ),
        lengths: MlxArray::from_raw_data(
            lengths.as_ptr().cast(),
            std::mem::size_of_val(lengths.as_slice()),
            &[seq],
            MlxDtype::Int32,
        ),
        width,
    }))
}

/// Device selection: `[1, seq, width]` int32 rows with each query's visible
/// tokens packed at the front and `-1` after them, built on the GPU by
/// [`QsaSelection::compact_device_indices`] in the same order as the host
/// gather buffer.
///
/// The block-then-tail shape check is skipped here: it needs the token values,
/// and reading them back is the copy this path removes. The selector produced
/// these rows from the request's own cache, so they are in range by
/// construction, and `lengths` is still the exact count the kernel walks.
fn pack_device_selection(selection: &QsaSelection, seq: i32) -> Option<PackedSelection> {
    if seq < 1 {
        return None;
    }
    let compact = selection.compact_device_indices()?;
    let shape = compact.shape();
    if shape.len() != 3 || shape[0] != 1 || shape[1] != seq {
        return None;
    }
    let width = shape[2];
    if !(1..=MAX_SELECTED_WIDTH).contains(&width) {
        return None;
    }
    let mut lengths = Vec::with_capacity(seq as usize);
    for s in 0..seq as usize {
        let len = selection.selected_len(0, s);
        if len == 0 || len > width as usize {
            return None;
        }
        lengths.push(len as i32);
    }
    Some(PackedSelection {
        indices: reshape(&compact, &[seq, width], None),
        lengths: MlxArray::from_raw_data(
            lengths.as_ptr().cast(),
            std::mem::size_of_val(lengths.as_slice()),
            &[seq],
            MlxDtype::Int32,
        ),
        width,
    })
}

/// A QSA selection is a run of complete blocks, each `compress_ratio`
/// consecutive tokens starting at a block boundary, in selection order,
/// followed by the partial visible tail of fewer than `compress_ratio` rows.
///
/// The kernel relies on that layout to read a selected row by absolute
/// position (it does not need the block ids themselves), so a selection that
/// does not match it takes the gathered path.
fn selection_shape_ok(tokens: &[i32], compress_ratio: i32) -> bool {
    if compress_ratio < 1 || tokens.is_empty() {
        return false;
    }
    let ratio = compress_ratio as usize;
    let tail_len = tokens.len() % ratio;
    let blocks = (tokens.len() - tail_len) / ratio;
    for block in 0..blocks {
        let start = block * ratio;
        let first = tokens[start];
        if first < 0 || first % compress_ratio != 0 {
            return false;
        }
        if (0..ratio).any(|offset| tokens[start + offset] != first + offset as i32) {
            return false;
        }
    }
    let tail_start = blocks * ratio;
    (0..tail_len).all(|offset| tokens[tail_start + offset] == tokens[tail_start] + offset as i32)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use mlx_sys::{MlxDtype, eval, scaled_dot_product_attention, slice, take, transpose};

    const DIM: i32 = 32;
    const Q_HEADS: i32 = 4;
    const KV_HEADS: i32 = 2;
    const RATIO: i32 = 4;

    /// Deterministic, well-scaled data so fp16/bf16 rounding stays meaningful.
    fn data(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let mixed = (i as u32)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(seed.wrapping_mul(40_503));
                ((mixed >> 8) % 2001) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    fn array_f32(values: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            values.as_ptr().cast(),
            std::mem::size_of_val(values),
            shape,
            MlxDtype::Float32,
        )
    }

    /// Block-then-tail rows for a query at absolute position `visible - 1`:
    /// up to two complete `RATIO`-token blocks (chosen so different queries
    /// pick different blocks) followed by the partial visible tail.
    fn selected_rows(visible: i32, blocks: &[i32]) -> Vec<i32> {
        let complete = visible / RATIO;
        let mut chosen = Vec::new();
        for &block in blocks {
            let block = block % complete.max(1);
            for token in 0..RATIO {
                chosen.push(block * RATIO + token);
            }
        }
        let partial = complete * RATIO;
        for token in partial..visible {
            chosen.push(token);
        }
        chosen
    }

    /// Same as [`selected_rows`] but without folding the block index modulo the
    /// complete-block count, so an explicit order (including a reversal) is
    /// preserved.
    fn selected_rows_in_order(visible: i32, blocks: &[i32], complete: i32) -> Vec<i32> {
        let mut chosen = Vec::new();
        for &block in blocks {
            if complete == 0 || block >= complete {
                continue;
            }
            for token in 0..RATIO {
                chosen.push(block * RATIO + token);
            }
        }
        let partial = complete * RATIO;
        for token in partial..visible {
            chosen.push(token);
        }
        chosen
    }

    struct Fixture {
        queries: MlxArray,
        keys: KvHistory,
        values: KvHistory,
        selection: QsaSelection,
        seq: i32,
    }

    fn fixture(seq: i32, rows: i32, dtype: MlxDtype) -> Fixture {
        let q = data((seq * Q_HEADS * DIM) as usize, 1);
        let k = data((rows * KV_HEADS * DIM) as usize, 2);
        let v = data((rows * KV_HEADS * DIM) as usize, 3);
        let queries = mlx_sys::astype(&array_f32(&q, &[1, Q_HEADS, seq, DIM]), dtype, None);
        let rows_of = |values: &[f32]| {
            let widened = array_f32(values, &[1, rows, KV_HEADS, DIM]);
            mlx_sys::astype(&widened, dtype, None)
        };
        let mut chosen = Vec::new();
        for s in 0..seq {
            let visible = s + 1;
            let blocks = [s, s + 1];
            chosen.push(selected_rows(visible, &blocks));
        }
        Fixture {
            queries,
            keys: KvHistory::from_flat(rows_of(&k)),
            values: KvHistory::from_flat(rows_of(&v)),
            selection: QsaSelection::from_host_tokens(vec![chosen]),
            seq,
        }
    }

    /// Independent reference: gather each query's own rows and attend them
    /// unmasked. This is the same value set the gathered path attends to,
    /// written once here so the kernel is compared against the mathematical
    /// definition rather than against itself.
    fn gathered_reference(fixture: &Fixture, scale: f32) -> MlxArray {
        let keys = fixture.keys.flat().expect("flat keys");
        let values = fixture.values.flat().expect("flat values");
        let mut rows = Vec::new();
        for s in 0..fixture.seq as usize {
            let chosen = fixture.selection.tokens_for_query(0, s);
            let idx = MlxArray::from_raw_data(
                chosen.as_ptr().cast(),
                std::mem::size_of_val(chosen),
                &[chosen.len() as i32],
                MlxDtype::Int32,
            );
            let q = slice(
                &fixture.queries,
                &[0, 0, s as i32, 0],
                &[1, Q_HEADS, s as i32 + 1, DIM],
                &[1, 1, 1, 1],
                None,
            );
            let k = transpose(&take(&keys, &idx, 1, None), &[0, 2, 1, 3], None);
            let v = transpose(&take(&values, &idx, 1, None), &[0, 2, 1, 3], None);
            rows.push(scaled_dot_product_attention(
                &mlx_sys::contiguous(&q, None),
                &mlx_sys::contiguous(&k, None),
                &mlx_sys::contiguous(&v, None),
                scale,
                false,
                None,
            ));
        }
        let refs: Vec<&MlxArray> = rows.iter().collect();
        mlx_sys::concatenate(&refs, 2, None)
    }

    fn close(actual: &MlxArray, expected: &MlxArray, atol: f32, rtol: f32, label: &str) {
        let actual = mlx_sys::contiguous(&mlx_sys::astype(actual, MlxDtype::Float32, None), None);
        let expected =
            mlx_sys::contiguous(&mlx_sys::astype(expected, MlxDtype::Float32, None), None);
        eval(&[&actual, &expected]);
        assert_eq!(actual.shape(), expected.shape(), "{label} shape");
        for (i, (&a, &b)) in actual
            .data_f32()
            .iter()
            .zip(expected.data_f32())
            .enumerate()
        {
            let bound = atol + rtol * b.abs();
            assert!(
                (a - b).abs() <= bound,
                "{label}[{i}]: {a} vs {b} (|d| {} > {bound})",
                (a - b).abs()
            );
        }
    }

    fn attention(fixture: &Fixture, indexed: bool) -> (Option<MlxArray>, u64) {
        let before = dispatches();
        let result = with_indexed_attention(indexed, || {
            attend_selected_indexed(
                &fixture.queries,
                &fixture.keys,
                &fixture.values,
                &fixture.selection,
                Q_HEADS,
                KV_HEADS,
                DIM,
                RATIO,
                1.0 / (DIM as f32).sqrt(),
            )
            .unwrap()
        });
        (result, dispatches() - before)
    }

    /// A selection in the layout `select_tokens_device_prefill` produces: each
    /// row holds `blocks` complete blocks, then `-1` padding up to the block
    /// segment width, then the visible partial tail. Returns the device
    /// selection and the same tokens as dense block-then-tail host rows.
    fn device_selection(seq: i32, rows: i32, blocks: i32) -> (QsaSelection, Vec<Vec<i32>>) {
        let block_cols = blocks * RATIO;
        let tail_width = RATIO - 1;
        let width = block_cols + tail_width;
        let mut tile = vec![-1_i32; (seq * width) as usize];
        let mut segments = Vec::with_capacity(seq as usize);
        let mut host_rows = Vec::with_capacity(seq as usize);
        for s in 0..seq {
            let visible = s + 1;
            let complete = (visible / RATIO).max(1);
            let mut block_tokens = Vec::new();
            for rank in 0..blocks {
                let block = (s + rank) % complete;
                for token in 0..RATIO {
                    block_tokens.push(block * RATIO + token);
                }
            }
            let tail: Vec<i32> = (complete * RATIO..visible).collect();
            assert!(block_tokens.iter().all(|&token| token < rows));
            assert!(tail.iter().all(|&token| token < rows));
            let base = (s * width) as usize;
            tile[base..base + block_tokens.len()].copy_from_slice(&block_tokens);
            let tail_start = base + block_cols as usize;
            tile[tail_start..tail_start + tail.len()].copy_from_slice(&tail);
            segments.push((block_tokens.len() as i32, tail.len() as i32));
            let mut dense = block_tokens;
            dense.extend_from_slice(&tail);
            host_rows.push(dense);
        }
        let device = MlxArray::from_raw_data(
            tile.as_ptr().cast(),
            std::mem::size_of_val(tile.as_slice()),
            &[1, seq, width],
            MlxDtype::Int32,
        );
        (
            QsaSelection::from_device_rows(device, segments, tail_width),
            host_rows,
        )
    }

    #[test]
    fn indexed_attention_consumes_device_selected_rows_in_place() {
        // The device arm must attend the rows the selector packed on the GPU,
        // in the same order the host gather buffer uses, without a copy back.
        let seq = 20;
        let (selection, host_rows) = device_selection(seq, 48, 2);
        assert!(selection.device_multi_query());
        let device_fixture = Fixture {
            selection,
            ..fixture(seq, 48, MlxDtype::Float32)
        };
        let (out, dispatched) = attention(&device_fixture, true);
        assert_eq!(dispatched, 1, "device rows must dispatch the kernel");
        let out = out.expect("indexed output");
        let reference = gathered_reference(&device_fixture, 1.0 / (DIM as f32).sqrt());
        close(&out, &reference, 1e-5, 1e-4, "device rows");

        // The same tokens selected on the host must give bit-identical rows:
        // the selection's provenance may not change the arithmetic.
        let host = Fixture {
            selection: QsaSelection::from_host_tokens(vec![host_rows]),
            ..fixture(seq, 48, MlxDtype::Float32)
        };
        let (host_out, host_dispatched) = attention(&host, true);
        assert_eq!(host_dispatched, 1, "host rows dispatch the kernel");
        let (device_values, host_values) = (
            mlx_sys::contiguous(&mlx_sys::astype(&out, MlxDtype::Float32, None), None),
            mlx_sys::contiguous(
                &mlx_sys::astype(&host_out.expect("host output"), MlxDtype::Float32, None),
                None,
            ),
        );
        eval(&[&device_values, &host_values]);
        assert_eq!(device_values.shape(), host_values.shape(), "output shape");
        for (i, (&device, &host)) in device_values
            .data_f32()
            .iter()
            .zip(host_values.data_f32())
            .enumerate()
        {
            assert_eq!(device.to_bits(), host.to_bits(), "device vs host row [{i}]");
        }
    }

    #[test]
    fn indexed_attention_declines_a_singleton_device_selection() {
        // A decode-shaped device selection carries one exact-length row, not a
        // per-query tile: it must keep the path the decode arm already uses.
        let seq = 20;
        let (device, _) = device_selection(seq, 48, 2);
        let compact = device.compact_device_indices().expect("device tile");
        let width = compact.shape()[2];
        // `[batch, 1, kept]` is the decode-shaped device row.
        let row = slice(&compact, &[0, 0, 0], &[1, 1, width], &[1, 1, 1], None);
        let selection = QsaSelection::from_device_rows(row, Vec::new(), 0);
        assert!(selection.device_tokens().is_some());
        assert!(!selection.device_multi_query());
        let fixture = Fixture {
            selection,
            ..fixture(seq, 48, MlxDtype::Float32)
        };
        let (out, dispatched) = attention(&fixture, true);
        assert_eq!(dispatched, 0, "a singleton device row must not dispatch");
        assert!(out.is_none(), "a singleton device row falls back");
    }

    #[test]
    fn indexed_attention_returns_the_value_row_for_one_selected_token() {
        let seq = 20;
        let rows = 48;
        let q = data((seq * Q_HEADS * DIM) as usize, 1);
        let k = data((rows * KV_HEADS * DIM) as usize, 2);
        let v = data((rows * KV_HEADS * DIM) as usize, 3);
        let queries = array_f32(&q, &[1, Q_HEADS, seq, DIM]);
        let keys = KvHistory::from_flat(array_f32(&k, &[1, rows, KV_HEADS, DIM]));
        let values = KvHistory::from_flat(array_f32(&v, &[1, rows, KV_HEADS, DIM]));
        // Every query selects exactly one row: its own position.
        let mut chosen = Vec::new();
        for s in 0..seq {
            chosen.push(vec![s]);
        }
        let selection = QsaSelection::from_host_tokens(vec![chosen]);
        let out = with_indexed_attention(true, || {
            attend_selected_indexed(
                &queries, &keys, &values, &selection, Q_HEADS, KV_HEADS, DIM, RATIO, 0.5,
            )
            .unwrap()
        })
        .expect("indexed output");
        let out32 = mlx_sys::contiguous(&mlx_sys::astype(&out, MlxDtype::Float32, None), None);
        let expected = mlx_sys::contiguous(
            &mlx_sys::astype(&values.flat().unwrap(), MlxDtype::Float32, None),
            None,
        );
        eval(&[&out32, &expected]);
        let got = out32.data_f32();
        let want = expected.data_f32();
        // out[0, h, s, d] must equal v[0, s, h / group, d]
        for s in 0..seq as usize {
            for h in 0..Q_HEADS as usize {
                let kvh = h / (Q_HEADS / KV_HEADS) as usize;
                for d in 0..DIM as usize {
                    let g = got[((h * seq as usize) + s) * DIM as usize + d];
                    let w = want[(s * KV_HEADS as usize + kvh) * DIM as usize + d];
                    assert!((g - w).abs() < 1e-4, "s={s} h={h} d={d}: {g} vs {w}");
                }
            }
        }
    }

    #[test]
    fn indexed_attention_matches_the_gathered_values_in_fp32() {
        let fixture = fixture(20, 48, MlxDtype::Float32);
        let (out, dispatched) = attention(&fixture, true);
        assert_eq!(dispatched, 1, "the kernel must run, not fall back");
        let out = out.expect("indexed output");
        let reference = gathered_reference(&fixture, 1.0 / (DIM as f32).sqrt());
        close(&out, &reference, 1e-5, 1e-4, "fp32 indexed");
    }

    #[test]
    fn indexed_attention_keeps_half_precision_tolerance() {
        for dtype in [MlxDtype::Float16, MlxDtype::Bfloat16] {
            let fixture = fixture(20, 48, dtype);
            let (out, dispatched) = attention(&fixture, true);
            assert_eq!(dispatched, 1, "{dtype:?} must dispatch");
            let out = out.expect("indexed output");
            assert_eq!(out.dtype(), dtype, "output dtype follows the query");
            let reference = gathered_reference(&fixture, 1.0 / (DIM as f32).sqrt());
            close(&out, &reference, 2e-3, 2e-2, &format!("{dtype:?} indexed"));
        }
    }

    #[test]
    fn indexed_attention_matches_the_reference_in_selection_order() {
        // Per-query block order differs (and is reversed relative to the block
        // index), so the kernel has to read each row from its recorded
        // position instead of assuming cache order.
        let mut fixture = fixture(20, 48, MlxDtype::Float32);
        let mut rows = Vec::new();
        for s in 0..fixture.seq {
            let visible = s + 1;
            let complete = visible / RATIO;
            let blocks = if complete >= 2 { [1, 0] } else { [0, 0] };
            rows.push(selected_rows_in_order(visible, &blocks, complete));
        }
        for row in &rows {
            assert!(
                selection_shape_ok(row, RATIO),
                "generated rows keep the block-then-tail order"
            );
        }
        fixture.selection = QsaSelection::from_host_tokens(vec![rows]);
        let (out, dispatched) = attention(&fixture, true);
        assert_eq!(dispatched, 1, "reordered blocks still dispatch");
        let out = out.expect("indexed output");
        let reference = gathered_reference(&fixture, 1.0 / (DIM as f32).sqrt());
        close(&out, &reference, 1e-5, 1e-4, "reordered blocks");
    }

    /// Shipped-pack geometry: 24 query heads over 2 kv heads (group 12), head
    /// dimension 256, bf16, and the two-segment history the real 8k prefill
    /// has. The small fixture above covers ordering and tolerances; this one
    /// covers the geometry the kernel actually runs in production.
    fn shipped_geometry_matches(segments: Option<(usize, usize)>) {
        use crate::model::shared::qwen4_exp_kv_history::with_kv_segments;

        let (q_heads, kv_heads) = (24i32, 2i32);
        let dim = 256i32;
        let (seq, rows) = (20i32, 96i32);
        let q = data((seq * q_heads * dim) as usize, 7);
        let k = data((rows * kv_heads * dim) as usize, 11);
        let v = data((rows * kv_heads * dim) as usize, 13);
        let queries = mlx_sys::astype(
            &array_f32(&q, &[1, q_heads, seq, dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let history = |values: &[f32]| -> KvHistory {
            with_kv_segments(segments, || {
                let rows_array = mlx_sys::astype(
                    &array_f32(values, &[1, rows, kv_heads, dim]),
                    MlxDtype::Bfloat16,
                    None,
                );
                // Two appends so a segmented history really has a sealed prefix
                // and a tail instead of one flat array.
                let half = rows / 2;
                let first = slice(
                    &rows_array,
                    &[0, 0, 0, 0],
                    &[1, half, kv_heads, dim],
                    &[1, 1, 1, 1],
                    None,
                );
                let rest = slice(
                    &rows_array,
                    &[0, half, 0, 0],
                    &[1, rows, kv_heads, dim],
                    &[1, 1, 1, 1],
                    None,
                );
                KvHistory::empty().append(&first).append(&rest)
            })
        };
        let keys = history(&k);
        let values = history(&v);
        let mut chosen = Vec::new();
        for s in 0..seq {
            chosen.push(selected_rows(s + 1, &[s, s + 1]));
        }
        let selection = QsaSelection::from_host_tokens(vec![chosen]);
        let scale = 1.0 / (dim as f32).sqrt();

        let indexed = with_indexed_attention(true, || {
            attend_selected_indexed(
                &queries, &keys, &values, &selection, q_heads, kv_heads, dim, RATIO, scale,
            )
            .unwrap()
        })
        .expect("indexed output");

        // Same rows gathered the way the production fallback reads them.
        let flat_keys = keys.flat().expect("flat keys");
        let flat_values = values.flat().expect("flat values");
        let mut rows_out = Vec::new();
        for s in 0..seq as usize {
            let chosen = selection.tokens_for_query(0, s);
            let idx = MlxArray::from_raw_data(
                chosen.as_ptr().cast(),
                std::mem::size_of_val(chosen),
                &[chosen.len() as i32],
                MlxDtype::Int32,
            );
            let q_row = slice(
                &queries,
                &[0, 0, s as i32, 0],
                &[1, q_heads, s as i32 + 1, dim],
                &[1, 1, 1, 1],
                None,
            );
            let k_sel = transpose(&take(&flat_keys, &idx, 1, None), &[0, 2, 1, 3], None);
            let v_sel = transpose(&take(&flat_values, &idx, 1, None), &[0, 2, 1, 3], None);
            rows_out.push(scaled_dot_product_attention(
                &mlx_sys::contiguous(&q_row, None),
                &mlx_sys::contiguous(&k_sel, None),
                &mlx_sys::contiguous(&v_sel, None),
                scale,
                false,
                None,
            ));
        }
        let refs: Vec<&MlxArray> = rows_out.iter().collect();
        let reference = mlx_sys::concatenate(&refs, 2, None);
        close(&indexed, &reference, 2e-3, 2e-2, "shipped geometry");
    }

    #[test]
    fn indexed_attention_matches_at_shipped_geometry_single_segment() {
        shipped_geometry_matches(None);
    }

    #[test]
    fn indexed_attention_matches_at_shipped_geometry_across_segments() {
        shipped_geometry_matches(Some((48, 8)));
    }

    /// Production selection width: 512 complete blocks (2048 rows) plus the
    /// partial tail, over a history long enough that only a fraction of the
    /// visible blocks are kept. This is the shape the shipped pack uses past
    /// the dense short-context limit, and the only one the small fixtures do
    /// not reach.
    #[test]
    fn indexed_attention_matches_with_a_full_block_selection() {
        let (q_heads, kv_heads) = (24i32, 2i32);
        let dim = 256i32;
        let block_topk = 512i32;
        let (seq, first_visible, rows) = (64i32, 4096i32, 4224i32);
        let q = data((seq * q_heads * dim) as usize, 17);
        let k = data((rows * kv_heads * dim) as usize, 19);
        let v = data((rows * kv_heads * dim) as usize, 23);
        let queries = mlx_sys::astype(
            &array_f32(&q, &[1, q_heads, seq, dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let keys = KvHistory::from_flat(mlx_sys::astype(
            &array_f32(&k, &[1, rows, kv_heads, dim]),
            MlxDtype::Bfloat16,
            None,
        ));
        let values = KvHistory::from_flat(mlx_sys::astype(
            &array_f32(&v, &[1, rows, kv_heads, dim]),
            MlxDtype::Bfloat16,
            None,
        ));
        let mut chosen = Vec::new();
        for s in 0..seq {
            let visible = first_visible + s + 1;
            let complete = visible / RATIO;
            assert!(complete >= block_topk, "the fixture needs a full top-k");
            let mut order = Vec::new();
            for rank in 0..block_topk {
                // Distinct blocks in a non-monotonic order, like a score
                // ranking that is not cache order.
                order.push((rank * 37).rem_euclid(block_topk));
            }
            chosen.push(selected_rows_in_order(visible, &order, complete));
        }
        let selection = QsaSelection::from_host_tokens(vec![chosen]);
        assert!(
            selection_shape_ok(selection.tokens_for_query(0, 0), RATIO),
            "the full block selection keeps the block-then-tail shape"
        );
        let scale = 1.0 / (dim as f32).sqrt();

        let indexed = with_indexed_attention(true, || {
            attend_selected_indexed(
                &queries, &keys, &values, &selection, q_heads, kv_heads, dim, RATIO, scale,
            )
            .unwrap()
        })
        .expect("indexed output");

        let flat_keys = keys.flat().expect("flat keys");
        let flat_values = values.flat().expect("flat values");
        let mut rows_out = Vec::new();
        for s in 0..seq as usize {
            let chosen = selection.tokens_for_query(0, s);
            let idx = MlxArray::from_raw_data(
                chosen.as_ptr().cast(),
                std::mem::size_of_val(chosen),
                &[chosen.len() as i32],
                MlxDtype::Int32,
            );
            let q_row = slice(
                &queries,
                &[0, 0, s as i32, 0],
                &[1, q_heads, s as i32 + 1, dim],
                &[1, 1, 1, 1],
                None,
            );
            let k_sel = transpose(&take(&flat_keys, &idx, 1, None), &[0, 2, 1, 3], None);
            let v_sel = transpose(&take(&flat_values, &idx, 1, None), &[0, 2, 1, 3], None);
            rows_out.push(scaled_dot_product_attention(
                &mlx_sys::contiguous(&q_row, None),
                &mlx_sys::contiguous(&k_sel, None),
                &mlx_sys::contiguous(&v_sel, None),
                scale,
                false,
                None,
            ));
        }
        let refs: Vec<&MlxArray> = rows_out.iter().collect();
        let reference = mlx_sys::concatenate(&refs, 2, None);
        close(&indexed, &reference, 2e-3, 2e-2, "full block selection");
    }

    /// Grid-size independence: the same queries must produce the same rows
    /// whether they are dispatched in a 64-query chunk or inside a 1024-query
    /// chunk. Production prefill dispatches one grid per chunk, so a kernel
    /// that depends on `grid.y` would be wrong only at real chunk sizes.
    #[test]
    fn indexed_attention_is_independent_of_chunk_size() {
        let (q_heads, kv_heads) = (24i32, 2i32);
        let dim = 256i32;
        let big = 1024i32;
        let small = 64i32;
        let rows = 4224i32;
        let q = data((big * q_heads * dim) as usize, 29);
        let k = data((rows * kv_heads * dim) as usize, 31);
        let v = data((rows * kv_heads * dim) as usize, 37);
        let keys = KvHistory::from_flat(mlx_sys::astype(
            &array_f32(&k, &[1, rows, kv_heads, dim]),
            MlxDtype::Bfloat16,
            None,
        ));
        let values = KvHistory::from_flat(mlx_sys::astype(
            &array_f32(&v, &[1, rows, kv_heads, dim]),
            MlxDtype::Bfloat16,
            None,
        ));
        // Selection depends only on the query's absolute position, so the
        // first `small` queries have the same rows in both runs.
        let selection_for = |first_query: i32, count: i32| {
            let mut chosen = Vec::new();
            for s in 0..count {
                let visible = first_query + s + 1;
                let complete = visible / RATIO;
                let mut order = Vec::new();
                for rank in 0..complete.min(512) {
                    order.push(rank);
                }
                chosen.push(selected_rows_in_order(visible, &order, complete));
            }
            QsaSelection::from_host_tokens(vec![chosen])
        };
        let scale = 1.0 / (dim as f32).sqrt();
        // `[1, heads, seq, dim]` is head-major: a contiguous prefix of the
        // flat data is *not* the first `count` queries of every head, so slice
        // the sequence axis instead.
        let all_queries = mlx_sys::astype(
            &array_f32(&q, &[1, q_heads, big, dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let run = |first_query: i32, count: i32| -> MlxArray {
            let queries = slice(
                &all_queries,
                &[0, 0, first_query, 0],
                &[1, q_heads, first_query + count, dim],
                &[1, 1, 1, 1],
                None,
            );
            let selection = selection_for(first_query, count);
            with_indexed_attention(true, || {
                attend_selected_indexed(
                    &queries, &keys, &values, &selection, q_heads, kv_heads, dim, RATIO, scale,
                )
                .unwrap()
            })
            .expect("indexed output")
        };
        let wide = run(0, big);
        let wide_again = run(0, big);
        let narrow = run(0, small);
        let flat = |array: &MlxArray| -> Vec<f32> {
            let values =
                mlx_sys::contiguous(&mlx_sys::astype(array, MlxDtype::Float32, None), None);
            eval(&[&values]);
            values.data_f32().to_vec()
        };
        let (a, b, n) = (flat(&wide), flat(&wide_again), flat(&narrow));
        let repeat_delta = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        // The output is `[1, heads, seq, dim]`, so the two runs share a row at
        // `h * seq * dim + s * dim + d`, not at a token-major offset: `seq`
        // differs between them and a token-major index would compare different
        // heads.
        let (heads, dims) = (q_heads as usize, dim as usize);
        let mut mismatched = 0usize;
        let mut worst = 0.0f32;
        let mut first = None;
        for h in 0..heads {
            for s in 0..small as usize {
                for d in 0..dims {
                    let narrow_index = (h * small as usize + s) * dims + d;
                    let wide_index = (h * big as usize + s) * dims + d;
                    let delta = (n[narrow_index] - a[wide_index]).abs();
                    if delta > 0.0 {
                        mismatched += 1;
                        worst = worst.max(delta);
                        first.get_or_insert((h, s, d, delta));
                    }
                }
            }
        }
        eprintln!(
            "CHUNK_SIZE_DIAG repeat_delta={repeat_delta} mismatched_elems={mismatched}/{} worst={worst} first={first:?}",
            small as usize * heads * dims
        );
        assert_eq!(
            repeat_delta, 0.0,
            "the same call must be deterministic (repeat delta {repeat_delta})"
        );
        assert_eq!(
            mismatched, 0,
            "a {big}-query chunk disagrees with a {small}-query chunk on {mismatched} elements (worst {worst}, first {first:?})"
        );
    }

    #[test]
    fn indexed_attention_is_off_unless_the_switch_is_set() {
        // The opt-in guard is the first thing the entry point does, so with the
        // switch off a chunk that the kernel would otherwise cover still takes
        // the gathered path: no dispatch and no index tile.
        let fixture = fixture(64, 96, MlxDtype::Float32);
        let (out, dispatched) = attention(&fixture, false);
        assert_eq!(dispatched, 0, "the default arm must not dispatch");
        assert!(out.is_none(), "the default arm keeps the gathered path");
    }

    #[test]
    fn indexed_prefill_warmup_applies_only_for_flash_next_with_the_switch_on() {
        assert!(indexed_prefill_warmup_applies(true, true, true));
        assert!(
            !indexed_prefill_warmup_applies(false, true, true),
            "non-Flash-Next packs never compile the indexed kernel"
        );
        assert!(
            !indexed_prefill_warmup_applies(true, true, false),
            "the default-off switch must not compile it"
        );
        assert!(
            !indexed_prefill_warmup_applies(true, false, true),
            "AX_MLX_LOAD_KERNEL_WARMUP=0 skips load-time compiles"
        );
    }

    #[test]
    fn indexed_prefill_warmup_declines_when_the_switch_is_off() {
        // The entry point's opt-in guard runs before the kernel is touched, so
        // a warm-up call with the switch off compiles nothing.
        let skipped = with_indexed_attention(false, || {
            warm_indexed_prefill_kernel(24, 2, 256, MlxDtype::Bfloat16)
        });
        assert!(
            skipped.is_err(),
            "the switch off must not compile the indexed kernel"
        );
    }

    #[test]
    fn indexed_prefill_warmup_compiles_the_shipped_geometry() {
        // The probe uses the model geometry, the only thing the kernel
        // template specialises on, so this is the pipeline production
        // dispatches for every later chunk.
        with_indexed_attention(true, || {
            warm_indexed_prefill_kernel(24, 2, 256, MlxDtype::Bfloat16)
        })
        .expect("shipped geometry warm-up");
    }

    #[test]
    fn indexed_attention_declines_short_windows_and_decode_rows() {
        // Fewer than BATCHED_ATTENTION_MIN_QUERIES queries keep the gathered
        // path: the indexed kernel never runs and nothing changes.
        let fixture = fixture(4, 16, MlxDtype::Float32);
        let (out, dispatched) = attention(&fixture, true);
        assert_eq!(dispatched, 0, "short windows must not dispatch");
        assert!(out.is_none(), "short windows fall back");
    }

    #[test]
    fn indexed_attention_declines_out_of_range_and_foreign_selection_shapes() {
        let fixture = fixture(20, 48, MlxDtype::Float32);
        // A selection row that reaches past the cache is refused rather than
        // read out of bounds.
        let mut rows = Vec::new();
        for s in 0..fixture.seq {
            let mut chosen = selected_rows(s + 1, &[0]);
            chosen[0] = fixture.keys.token_count() as i32;
            rows.push(chosen);
        }
        let out_of_range = QsaSelection::from_host_tokens(vec![rows]);
        let before = dispatches();
        let result = with_indexed_attention(true, || {
            attend_selected_indexed(
                &fixture.queries,
                &fixture.keys,
                &fixture.values,
                &out_of_range,
                Q_HEADS,
                KV_HEADS,
                DIM,
                RATIO,
                1.0,
            )
            .unwrap()
        });
        assert!(result.is_none(), "out-of-range rows fall back");
        assert_eq!(dispatches(), before, "no dispatch for a refused selection");
    }

    #[test]
    fn indexed_attention_skips_out_of_range_device_tokens() {
        // The device index tile is never validated on the host, so a selector
        // regression can name a row past the end of the history. The kernel
        // must treat that token like `-1` and keep the rest of the row instead
        // of reading past the K/V arrays.
        use crate::model::shared::qwen4_exp_kv_history::with_kv_segments;

        let seq = 20i32;
        let (sealed, tail) = (64i32, 8i32);
        let rows = sealed + tail;
        let q = data((seq * Q_HEADS * DIM) as usize, 1);
        let k = data((rows * KV_HEADS * DIM) as usize, 2);
        let v = data((rows * KV_HEADS * DIM) as usize, 3);
        let queries = array_f32(&q, &[1, Q_HEADS, seq, DIM]);
        let history = |values: &[f32]| {
            with_kv_segments(Some((1, sealed as usize)), || {
                let array = array_f32(values, &[1, rows, KV_HEADS, DIM]);
                let first = slice(
                    &array,
                    &[0, 0, 0, 0],
                    &[1, sealed, KV_HEADS, DIM],
                    &[1, 1, 1, 1],
                    None,
                );
                let rest = slice(
                    &array,
                    &[0, sealed, 0, 0],
                    &[1, rows, KV_HEADS, DIM],
                    &[1, 1, 1, 1],
                    None,
                );
                KvHistory::empty().append(&first).append(&rest)
            })
        };
        let keys = history(&k);
        let values = history(&v);
        assert_eq!(
            keys.sealed_rows(),
            Some(sealed),
            "the fixture needs a real sealed prefix and tail"
        );

        let width = 2;
        let mut tile = vec![-1_i32; (seq * width) as usize];
        let mut segments = Vec::new();
        for s in 0..seq {
            let base = (s * width) as usize;
            // One in-range token, then one past the end of the history.
            tile[base] = s;
            tile[base + 1] = rows + s;
            segments.push((width, 0));
        }
        let device = MlxArray::from_raw_data(
            tile.as_ptr().cast(),
            std::mem::size_of_val(tile.as_slice()),
            &[1, seq, width],
            MlxDtype::Int32,
        );
        let selection = QsaSelection::from_device_rows(device, segments, 0);
        let before = dispatches();
        let out = with_indexed_attention(true, || {
            attend_selected_indexed(
                &queries, &keys, &values, &selection, Q_HEADS, KV_HEADS, DIM, RATIO, 0.5,
            )
            .unwrap()
        })
        .expect("indexed output");
        assert_eq!(
            dispatches() - before,
            1,
            "an out-of-range token still dispatches the kernel"
        );

        let out = mlx_sys::contiguous(&mlx_sys::astype(&out, MlxDtype::Float32, None), None);
        eval(&[&out]);
        let got = out.data_f32();
        // Only the in-range token contributes, so each row is that token's
        // value row.
        for s in 0..seq as usize {
            for h in 0..Q_HEADS as usize {
                let kvh = h / (Q_HEADS / KV_HEADS) as usize;
                for d in 0..DIM as usize {
                    let index = (h * seq as usize + s) * DIM as usize + d;
                    let want = v[(s * KV_HEADS as usize + kvh) * DIM as usize + d];
                    assert!(
                        (got[index] - want).abs() < 1e-4,
                        "s={s} h={h} d={d}: {} vs {want}",
                        got[index]
                    );
                }
            }
        }
    }

    #[test]
    fn indexed_attention_declines_history_geometry_that_disagrees_with_the_template() {
        // The kernel indexes K/V as `row * (KvHeads * HeadDim)` and reads the
        // tail as `token - SealedRows`, so a history whose arrays disagree with
        // the geometry the caller claims must fall back instead of reading the
        // wrong strides.
        let fixture = fixture(20, 48, MlxDtype::Float32);
        let attempt = |queries: &MlxArray, kv_heads: i32, head_dim: i32| {
            let before = dispatches();
            let result = with_indexed_attention(true, || {
                attend_selected_indexed(
                    queries,
                    &fixture.keys,
                    &fixture.values,
                    &fixture.selection,
                    Q_HEADS,
                    kv_heads,
                    head_dim,
                    RATIO,
                    1.0,
                )
                .unwrap()
            });
            (result, dispatches() - before)
        };
        // The fixture's histories hold `KV_HEADS` kv heads; a template claiming
        // a different count is not one the kernel can index.
        let (result, dispatched) = attempt(&fixture.queries, 1, DIM);
        assert!(result.is_none(), "kv_heads mismatch falls back");
        assert_eq!(dispatched, 0, "kv_heads mismatch must not dispatch");

        // Same from the other side: queries that claim twice the head dimension
        // the histories hold.
        let wide = array_f32(
            &data((20 * Q_HEADS * DIM * 2) as usize, 1),
            &[1, Q_HEADS, 20, DIM * 2],
        );
        let (result, dispatched) = attempt(&wide, KV_HEADS, DIM * 2);
        assert!(result.is_none(), "head_dim mismatch falls back");
        assert_eq!(dispatched, 0, "head_dim mismatch must not dispatch");
    }

    #[test]
    fn split_segments_requires_arrays_to_match_the_kernel_template() {
        let fixture = fixture(20, 48, MlxDtype::Float32);
        let total = fixture.keys.token_count() as i32;
        assert!(
            split_segments(&fixture.keys, &fixture.values, total, KV_HEADS, DIM).is_some(),
            "the fixture geometry is the template geometry"
        );
        assert!(
            split_segments(&fixture.keys, &fixture.values, total, KV_HEADS - 1, DIM).is_none(),
            "kv_heads must match the arrays"
        );
        assert!(
            split_segments(&fixture.keys, &fixture.values, total, KV_HEADS, DIM + 32).is_none(),
            "head_dim must match the arrays"
        );
        assert!(
            split_segments(&fixture.keys, &fixture.values, total + 4, KV_HEADS, DIM).is_none(),
            "the logical row count must be the sealed plus tail rows"
        );

        // The same accounting across two physical segments: the tail has to
        // hold exactly the rows after the seal.
        use crate::model::shared::qwen4_exp_kv_history::with_kv_segments;

        let sealed_rows = 32i32;
        let cut = |history: &KvHistory| -> KvHistory {
            let segments = history.segments();
            let flat = segments.0.or(segments.1).expect("single segment");
            let shape = flat.shape();
            with_kv_segments(Some((1, sealed_rows as usize)), || {
                let sealed = slice(
                    flat,
                    &[0, 0, 0, 0],
                    &[1, sealed_rows, shape[2], shape[3]],
                    &[1, 1, 1, 1],
                    None,
                );
                let tail = slice(
                    flat,
                    &[0, sealed_rows, 0, 0],
                    &[1, total, shape[2], shape[3]],
                    &[1, 1, 1, 1],
                    None,
                );
                KvHistory::empty().append(&sealed).append(&tail)
            })
        };
        let keys = cut(&fixture.keys);
        let values = cut(&fixture.values);
        assert_eq!(keys.sealed_rows(), Some(sealed_rows));
        assert!(
            split_segments(&keys, &values, total, KV_HEADS, DIM).is_some(),
            "a two-segment history whose rows add up is admitted"
        );
        assert!(
            split_segments(&keys, &values, total - 1, KV_HEADS, DIM).is_none(),
            "a tail that does not hold the rows after the seal is refused"
        );
    }

    #[test]
    fn pack_selection_declines_rows_wider_than_the_index_tile() {
        // A configured budget can select more rows than this path packs. That
        // chunk belongs to the gathered path: the switch must fall back, not
        // error the request.
        let (seq, blocks) = (16usize, 2050usize);
        let mut row = Vec::new();
        for block in 0..blocks {
            for token in 0..RATIO {
                row.push(block as i32 * RATIO + token);
            }
        }
        assert!(row.len() > MAX_SELECTED_WIDTH as usize);
        assert!(selection_shape_ok(&row, RATIO), "a full block run");
        let selection = QsaSelection::from_host_tokens(vec![vec![row; seq]]);
        let packed = pack_selection(&selection, seq as i32, 8200, RATIO).unwrap();
        assert!(packed.is_none(), "an over-wide selection falls back");
    }

    #[test]
    fn selection_shape_accepts_blocks_with_and_without_a_tail() {
        // Two complete blocks, no tail.
        assert!(selection_shape_ok(&[4, 5, 6, 7, 12, 13, 14, 15], 4));
        // Two complete blocks plus a partial tail.
        assert!(selection_shape_ok(&[4, 5, 6, 7, 0, 1, 2, 3, 24, 25], 4));
        // Tail only (a query before its first complete block plus a block).
        assert!(selection_shape_ok(&[2, 3], 4));
        // A block whose rows are not contiguous, or that does not start on a
        // block boundary, is not a selection the indexer can produce.
        assert!(!selection_shape_ok(&[4, 5, 6, 9], 4));
        assert!(!selection_shape_ok(&[5, 6, 7, 8], 4));
        assert!(!selection_shape_ok(&[], 4));
    }
}
