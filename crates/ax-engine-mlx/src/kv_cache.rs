use std::time::Instant;

use mlx_sys::{MlxArray, MlxDtype, concatenate, contiguous, eval, slice, slice_update, zeros};

use crate::kv_block_pool::{
    FaBlockPool, FaBlockPoolConfig, FaBlockPoolError, PhysicalBlockId,
    default_fa_block_pool_config, fa_kv_block_pool_enabled,
};

/// Pre-allocated chunk size (tokens).  The buffer grows by this amount each time
/// the logical sequence length exceeds capacity, so the number of grow operations
/// per session is at most ceil(total_tokens / CHUNK).
pub(crate) const KV_CHUNK_TOKENS: usize = 256;

fn chunk_ceiling(n: usize) -> usize {
    n.div_ceil(KV_CHUNK_TOKENS) * KV_CHUNK_TOKENS
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AppendShape {
    new_tokens: usize,
    n_kv_heads: i32,
    head_dim: i32,
    dtype: MlxDtype,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GlmMlaAppendShape {
    new_tokens: usize,
    latent_dim: i32,
    rope_dim: i32,
    dtype: MlxDtype,
}

fn validate_append_inputs(
    layer: usize,
    layer_count: usize,
    new_k: &MlxArray,
    new_v: &MlxArray,
) -> AppendShape {
    assert!(
        layer < layer_count,
        "KV cache layer {layer} out of bounds for {layer_count} layers"
    );

    let k_shape = new_k.shape();
    let v_shape = new_v.shape();
    assert_eq!(
        k_shape, v_shape,
        "KV cache append requires matching K/V shapes"
    );
    assert_eq!(
        k_shape.len(),
        4,
        "KV cache append expects [1, n_kv_heads, tokens, head_dim]"
    );
    assert_eq!(k_shape[0], 1, "KV cache append supports batch=1 only");
    assert!(
        k_shape[1] > 0 && k_shape[2] > 0 && k_shape[3] > 0,
        "KV cache append requires positive heads, tokens, and head_dim"
    );

    let k_dtype = new_k.dtype();
    assert_eq!(
        k_dtype,
        new_v.dtype(),
        "KV cache append requires matching K/V dtypes"
    );

    AppendShape {
        new_tokens: k_shape[2] as usize,
        n_kv_heads: k_shape[1],
        head_dim: k_shape[3],
        dtype: k_dtype,
    }
}

fn validate_glm_mla_append_inputs(
    layer: usize,
    layer_count: usize,
    new_kv_latent: &MlxArray,
    new_k_pe: &MlxArray,
) -> GlmMlaAppendShape {
    assert!(
        layer < layer_count,
        "GLM MLA cache layer {layer} out of bounds for {layer_count} layers"
    );

    let latent_shape = new_kv_latent.shape();
    let rope_shape = new_k_pe.shape();
    assert_eq!(
        latent_shape.len(),
        4,
        "GLM MLA latent cache append expects [1, 1, tokens, kv_lora_rank]"
    );
    assert_eq!(
        rope_shape.len(),
        4,
        "GLM MLA rope-key cache append expects [1, 1, tokens, qk_rope_head_dim]"
    );
    assert_eq!(latent_shape[0], 1, "GLM MLA cache supports batch=1 only");
    assert_eq!(rope_shape[0], 1, "GLM MLA cache supports batch=1 only");
    assert_eq!(
        latent_shape[1], 1,
        "GLM MLA latent cache stores one latent head"
    );
    assert_eq!(
        rope_shape[1], 1,
        "GLM MLA rope-key cache stores one RoPE head"
    );
    assert_eq!(
        latent_shape[2], rope_shape[2],
        "GLM MLA cache append requires matching latent/k_pe token counts"
    );
    assert!(
        latent_shape[2] > 0 && latent_shape[3] > 0 && rope_shape[3] > 0,
        "GLM MLA cache append requires positive tokens and dimensions"
    );

    let dtype = new_kv_latent.dtype();
    assert_eq!(
        dtype,
        new_k_pe.dtype(),
        "GLM MLA cache append requires matching latent/k_pe dtypes"
    );

    GlmMlaAppendShape {
        new_tokens: latent_shape[2] as usize,
        latent_dim: latent_shape[3],
        rope_dim: rope_shape[3],
        dtype,
    }
}

fn copy_token_range_to_rotating(
    source: &MlxArray,
    dest: &MlxArray,
    lkv: &LayerKV,
    token_start: usize,
    token_end: usize,
    window: usize,
) -> MlxArray {
    let mut out = dest.clone();
    let mut src_start = token_start;
    while src_start < token_end {
        let dst_start = src_start % window;
        let len = (token_end - src_start).min(window - dst_start);
        let src_stop = src_start + len;
        let dst_stop = dst_start + len;
        let segment = slice(
            source,
            &[0, 0, src_start as i32, 0],
            &[1, lkv.n_kv_heads, src_stop as i32, lkv.head_dim],
            &[1, 1, 1, 1],
            None,
        );
        out = slice_update(
            &out,
            &segment,
            &[0, 0, dst_start as i32, 0],
            &[1, lkv.n_kv_heads, dst_stop as i32, lkv.head_dim],
            &[1, 1, 1, 1],
            None,
        );
        src_start = src_stop;
    }
    out
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MlxKVCacheUsage {
    pub logical_tokens: usize,
    pub capacity_tokens: usize,
    pub logical_bytes: u64,
    pub capacity_bytes: u64,
    /// KV-backed attention layers, including sliding-window attention layers.
    pub full_attention_layers: usize,
    /// KV-backed layers with a configured sliding window.
    pub sliding_window_layers: usize,
    pub sliding_window_retained_tokens: usize,
    pub sliding_window_reclaimable_capacity_tokens: usize,
    pub sliding_window_reclaimable_capacity_bytes: u64,
    /// Sliding-window layers currently stored as rotating rings (backing
    /// store bounded to `window + slack` instead of O(context)).
    pub rotated_ring_layers: usize,
    /// Bounded-rollback slack configured for this cache's rings (0 = pure
    /// window-sized rings).
    pub rotating_ring_slack: usize,
    pub linear_state_layers: usize,
    pub linear_state_bytes: u64,
    pub growth_count: u64,
    /// Cumulative microseconds spent materializing dense FA views from private
    /// paged blocks (PR4). Zero when the contiguous path is used.
    pub paged_materialize_us: u64,
    /// Times a paged FA append fell back to contiguous growth because the
    /// private block pool was exhausted (fail-soft for production safety).
    pub paged_pool_exhaustion_fallbacks: u64,
}

#[derive(Clone)]
struct LayerKV {
    /// Full backing buffer: `[1, n_kv_heads, capacity, head_dim]`.
    k: MlxArray,
    v: MlxArray,
    /// Cached view `[1, n_kv_heads, 0..seq_len, head_dim]` returned by the last
    /// `append` call.  KV-shared layers (Gemma4 layers 24-41) read this directly
    /// via `peek_source_kv` instead of creating a second identical `slice` node.
    /// Having two separate `slice` nodes on the same backing buffer caused MLX to
    /// dispatch the slice kernel twice, adding ~12 µs × 40 dispatches ≈ 0.5 ms per
    /// decode step for E2B.
    last_k_view: Option<MlxArray>,
    last_v_view: Option<MlxArray>,
    n_kv_heads: i32,
    head_dim: i32,
    capacity: usize,
    rotating_window: Option<usize>,
    dtype: MlxDtype,
}

/// Private FA block list for one layer (PR4 paged path).
///
/// Each block is a full `[1, n_kv_heads, block_size, head_dim]` slab. Logical
/// tokens fill blocks left-to-right; SDPA consumers materialize a dense
/// `[1, n_kv_heads, T, head_dim]` view via [`PagedFaLayer::materialize`].
#[derive(Clone)]
struct PagedFaLayer {
    n_kv_heads: i32,
    head_dim: i32,
    dtype: MlxDtype,
    block_size: usize,
    block_ids: Vec<PhysicalBlockId>,
    k_blocks: Vec<MlxArray>,
    v_blocks: Vec<MlxArray>,
    last_k_view: Option<MlxArray>,
    last_v_view: Option<MlxArray>,
}

impl PagedFaLayer {
    fn capacity_tokens(&self) -> usize {
        self.block_ids.len().saturating_mul(self.block_size)
    }

    fn clear_views(&mut self) {
        self.last_k_view = None;
        self.last_v_view = None;
    }

    fn materialize(&self, token_start: usize, token_end: usize) -> (MlxArray, MlxArray) {
        assert!(
            token_end >= token_start,
            "paged materialize requires token_end >= token_start"
        );
        if token_end == token_start {
            let shape = [1i32, self.n_kv_heads, 0, self.head_dim];
            return (
                zeros(&shape, self.dtype, None),
                zeros(&shape, self.dtype, None),
            );
        }
        assert!(
            token_end <= self.capacity_tokens(),
            "paged materialize past capacity: end={token_end} cap={}",
            self.capacity_tokens()
        );

        let mut k_pieces: Vec<MlxArray> = Vec::new();
        let mut v_pieces: Vec<MlxArray> = Vec::new();
        let mut t = token_start;
        while t < token_end {
            let block_idx = t / self.block_size;
            let offset = t % self.block_size;
            let take = (token_end - t).min(self.block_size - offset);
            let start = [0i32, 0, offset as i32, 0];
            let stop = [1i32, self.n_kv_heads, (offset + take) as i32, self.head_dim];
            let strides = [1i32, 1, 1, 1];
            k_pieces.push(slice(
                &self.k_blocks[block_idx],
                &start,
                &stop,
                &strides,
                None,
            ));
            v_pieces.push(slice(
                &self.v_blocks[block_idx],
                &start,
                &stop,
                &strides,
                None,
            ));
            t += take;
        }
        if k_pieces.len() == 1 {
            return (k_pieces.remove(0), v_pieces.remove(0));
        }
        let k_refs: Vec<&MlxArray> = k_pieces.iter().collect();
        let v_refs: Vec<&MlxArray> = v_pieces.iter().collect();
        (concatenate(&k_refs, 2, None), concatenate(&v_refs, 2, None))
    }

    /// Clone block tensors into a fresh pool (private PR4: new physical IDs).
    fn clone_into_pool(&self, pool: &mut FaBlockPool) -> Result<Self, FaBlockPoolError> {
        let n = self.block_ids.len() as u32;
        let ids = pool.allocate(n)?;
        Ok(Self {
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            dtype: self.dtype,
            block_size: self.block_size,
            block_ids: ids,
            k_blocks: self.k_blocks.clone(),
            v_blocks: self.v_blocks.clone(),
            last_k_view: self.last_k_view.clone(),
            last_v_view: self.last_v_view.clone(),
        })
    }

    fn ensure_capacity(
        &mut self,
        pool: &mut FaBlockPool,
        tokens: usize,
        growth_count: &mut u64,
    ) -> Result<(), FaBlockPoolError> {
        let needed = if tokens == 0 {
            0
        } else {
            tokens.div_ceil(self.block_size)
        };
        while self.block_ids.len() < needed {
            let ids = pool.allocate(1)?;
            let shape = [1i32, self.n_kv_heads, self.block_size as i32, self.head_dim];
            self.block_ids.push(ids[0]);
            self.k_blocks.push(zeros(&shape, self.dtype, None));
            self.v_blocks.push(zeros(&shape, self.dtype, None));
            *growth_count = growth_count.saturating_add(1);
            self.clear_views();
        }
        Ok(())
    }

    fn free_blocks_beyond(&mut self, pool: &mut FaBlockPool, keep_tokens: usize) {
        let keep_blocks = if keep_tokens == 0 {
            0
        } else {
            keep_tokens.div_ceil(self.block_size)
        };
        if self.block_ids.len() <= keep_blocks {
            return;
        }
        let free_ids: Vec<PhysicalBlockId> = self.block_ids.drain(keep_blocks..).collect();
        self.k_blocks.truncate(keep_blocks);
        self.v_blocks.truncate(keep_blocks);
        // Best-effort free; a double-free would be a pool bug.
        let _ = pool.free(&free_ids);
        self.clear_views();
    }

    fn write_tokens(&mut self, write_start: usize, new_k: &MlxArray, new_v: &MlxArray) {
        let new_tokens = new_k.shape()[2] as usize;
        let write_end = write_start + new_tokens;
        assert!(
            write_end <= self.capacity_tokens(),
            "paged write past capacity"
        );
        let mut src = 0usize;
        let mut t = write_start;
        while t < write_end {
            let block_idx = t / self.block_size;
            let offset = t % self.block_size;
            let take = (write_end - t).min(self.block_size - offset);
            let src_start = [0i32, 0, src as i32, 0];
            let src_stop = [1i32, self.n_kv_heads, (src + take) as i32, self.head_dim];
            let strides = [1i32, 1, 1, 1];
            let k_seg = slice(new_k, &src_start, &src_stop, &strides, None);
            let v_seg = slice(new_v, &src_start, &src_stop, &strides, None);
            let dst_start = [0i32, 0, offset as i32, 0];
            let dst_stop = [1i32, self.n_kv_heads, (offset + take) as i32, self.head_dim];
            self.k_blocks[block_idx] = slice_update(
                &self.k_blocks[block_idx],
                &k_seg,
                &dst_start,
                &dst_stop,
                &strides,
                None,
            );
            self.v_blocks[block_idx] = slice_update(
                &self.v_blocks[block_idx],
                &v_seg,
                &dst_start,
                &dst_stop,
                &strides,
                None,
            );
            src += take;
            t += take;
        }
        self.clear_views();
    }
}

/// FA layer storage: contiguous production path or private paged blocks.
#[derive(Clone)]
enum FaLayerStorage {
    Contiguous(LayerKV),
    Paged(PagedFaLayer),
}

impl FaLayerStorage {
    fn rotating_window(&self) -> Option<usize> {
        match self {
            Self::Contiguous(lkv) => lkv.rotating_window,
            Self::Paged(_) => None,
        }
    }

    fn n_kv_heads(&self) -> i32 {
        match self {
            Self::Contiguous(lkv) => lkv.n_kv_heads,
            Self::Paged(p) => p.n_kv_heads,
        }
    }

    fn head_dim(&self) -> i32 {
        match self {
            Self::Contiguous(lkv) => lkv.head_dim,
            Self::Paged(p) => p.head_dim,
        }
    }

    fn dtype(&self) -> MlxDtype {
        match self {
            Self::Contiguous(lkv) => lkv.dtype,
            Self::Paged(p) => p.dtype,
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::Contiguous(lkv) => lkv.capacity,
            Self::Paged(p) => p.capacity_tokens(),
        }
    }

    fn clear_views(&mut self) {
        match self {
            Self::Contiguous(lkv) => {
                lkv.last_k_view = None;
                lkv.last_v_view = None;
            }
            Self::Paged(p) => p.clear_views(),
        }
    }

    fn as_contiguous_mut(&mut self) -> Option<&mut LayerKV> {
        match self {
            Self::Contiguous(lkv) => Some(lkv),
            Self::Paged(_) => None,
        }
    }

    fn as_contiguous(&self) -> Option<&LayerKV> {
        match self {
            Self::Contiguous(lkv) => Some(lkv),
            Self::Paged(_) => None,
        }
    }
}

#[derive(Clone)]
struct GlmMlaLayerCache {
    /// `[1, 1, capacity, kv_lora_rank]`, matching mlx-lm's latent KV cache.
    kv_latent: MlxArray,
    /// `[1, 1, capacity, qk_rope_head_dim]`, matching mlx-lm's RoPE key cache.
    k_pe: MlxArray,
    latent_dim: i32,
    rope_dim: i32,
    capacity: usize,
    dtype: MlxDtype,
}

#[derive(Clone, Default)]
struct LinearLayerState {
    /// Qwen3.5 gated-delta conv tail: `[1, conv_kernel - 1, conv_dim]`.
    conv_state: Option<MlxArray>,
    /// Qwen3.5 gated-delta recurrent state: `[1, value_heads, value_dim, key_dim]`.
    recurrent_state: Option<MlxArray>,
}

/// Destructor compatible with [`MlxArray::from_managed_data`]. Recovers
/// the `Box<Vec<u8>>` that owned the tensor's data buffer when
/// `try_deserialize_from_bytes` constructed the array, and drops it so
/// the heap allocation is freed.
///
/// # Safety
///
/// `payload` must have been produced by `Box::into_raw(Box::new(Vec<u8>))`
/// and not yet recovered or freed. MLX guarantees `payload` is non-null
/// and called exactly once per matching `from_managed_data` call.
unsafe extern "C" fn vec_payload_drop(payload: *mut std::ffi::c_void) {
    if payload.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(payload as *mut Vec<u8>);
    }
}

/// Error produced by [`MlxKVCache::try_deserialize_from_bytes`].
#[derive(Debug)]
pub enum MlxKVCacheSerializeError {
    /// Header magic did not match the F3-disk-cache wire format.
    BadMagic,
    /// Serialised payload was produced by an incompatible format version.
    UnsupportedVersion(u32),
    /// Payload ended before the structure required more bytes.
    UnexpectedEof,
    /// Encountered a dtype tag that does not map to a known MlxDtype.
    UnknownDtype(u8),
    /// Encountered an unknown layer-kind discriminator.
    UnknownLayerKind(u8),
    /// Tensor metadata declared an unsupported rank.
    BadShape(usize),
}

impl std::fmt::Display for MlxKVCacheSerializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic => write!(f, "kv-cache payload has wrong magic"),
            Self::UnsupportedVersion(v) => write!(f, "kv-cache payload version {v} unsupported"),
            Self::UnexpectedEof => write!(f, "kv-cache payload truncated"),
            Self::UnknownDtype(t) => write!(f, "kv-cache payload has unknown dtype tag {t}"),
            Self::UnknownLayerKind(t) => write!(f, "kv-cache payload has unknown layer kind {t}"),
            Self::BadShape(n) => write!(f, "kv-cache payload has bad tensor rank {n}"),
        }
    }
}

impl std::error::Error for MlxKVCacheSerializeError {}

fn read_exact_from(
    reader: &mut dyn std::io::Read,
    buf: &mut [u8],
) -> Result<(), MlxKVCacheSerializeError> {
    reader
        .read_exact(buf)
        .map_err(|_| MlxKVCacheSerializeError::UnexpectedEof)
}

fn read_u8_from(reader: &mut dyn std::io::Read) -> Result<u8, MlxKVCacheSerializeError> {
    let mut buf = [0u8; 1];
    read_exact_from(reader, &mut buf)?;
    Ok(buf[0])
}

fn read_u32_from(reader: &mut dyn std::io::Read) -> Result<u32, MlxKVCacheSerializeError> {
    let mut buf = [0u8; 4];
    read_exact_from(reader, &mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32_from(reader: &mut dyn std::io::Read) -> Result<i32, MlxKVCacheSerializeError> {
    let mut buf = [0u8; 4];
    read_exact_from(reader, &mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64_from(reader: &mut dyn std::io::Read) -> Result<u64, MlxKVCacheSerializeError> {
    let mut buf = [0u8; 8];
    read_exact_from(reader, &mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Borrowed view of a GLM-MLA layer's cached KV state. Returned by
/// [`MlxKVCache::glm_mla_layer_state`] for debug tooling that needs to
/// inspect per-layer cache contents without taking ownership.
pub struct GlmMlaLayerStateView<'a> {
    /// The full backing buffer for the latent KV cache; shape
    /// `[1, 1, capacity, latent_dim]`. Valid region is `[0..seq_len]`.
    pub kv_latent: &'a MlxArray,
    /// The full backing buffer for the RoPE key cache; shape
    /// `[1, 1, capacity, rope_dim]`. Valid region is `[0..seq_len]`.
    pub k_pe: &'a MlxArray,
    /// Inner dim of `kv_latent` — equal to the model's `kv_lora_rank`.
    pub latent_dim: i32,
    /// Inner dim of `k_pe` — equal to the model's `qk_rope_head_dim`.
    pub rope_dim: i32,
}

/// Per-request attention cache with chunked KV pre-allocation.
///
/// Full-attention KV shape convention:
/// `[1, n_kv_heads, seq_len, head_dim]` (batch=1, SDPA-native format).
///
/// ## Growth strategy
///
/// Unlike the naive approach that calls `concatenate` on every append (O(n) data
/// movement per step), this cache pre-allocates buffers in `KV_CHUNK_TOKENS`-sized
/// blocks and uses `slice_update` to write new tokens into the pre-allocated region.
/// Buffer growth (via concatenation with zeros) happens at most every `KV_CHUNK_TOKENS`
/// steps — typically 0–1 times per request for common prompt+decode lengths.
///
/// ## Draft rollback
///
/// `trim_to(prefix_len)` only updates `seq_len`.  The "trimmed" positions remain in
/// the backing buffer but are beyond the logical boundary, so SDPA never sees them.
/// The next `append` overwrites from `prefix_len`, restoring correctness.
pub struct MlxKVCache {
    layers: Vec<Option<FaLayerStorage>>,
    glm_mla_layers: Vec<Option<GlmMlaLayerCache>>,
    linear_layers: Vec<LinearLayerState>,
    /// Current logical sequence length (token count cached). Private so
    /// every mutation goes through [`Self::advance`] / [`Self::set_seq_len`]
    /// — the historical footgun was call sites bumping this field out of
    /// sync with what was actually appended.
    seq_len: usize,
    /// RoPE offset added to `seq_len` for positional encoding.  Used when
    /// the KV cache has fewer physical entries than the logical sequence
    /// position (e.g. after capped MTP warmup where only the last N tokens
    /// are warmed up but RoPE needs the full prompt offset).
    pub rope_offset: usize,
    growth_count: u64,
    use_rotating_sliding_decode: bool,
    /// Extra ring slots beyond the sliding window that keep speculative
    /// rollback (`trim_to`) sound after rotation. `0` = pure ring: exactly
    /// window-sized, mask-free single-token SDPA, no rollback permitted
    /// (the pre-bounded-rollback behavior). `> 0` = bounded ring: capacity
    /// `window + slack`, every SDPA over the ring needs a validity mask
    /// ([`SlidingRingLayout`]), and `trim_to` accepts rollbacks up to
    /// `slack` tokens deep.
    rotating_slack: usize,
    /// Private FA block pool when paged mode is active (flag or explicit
    /// constructor). `None` keeps the historical contiguous path.
    fa_pool: Option<FaBlockPool>,
    /// Cumulative µs spent in paged FA materialize (SDPA / serialize views).
    paged_materialize_us: u64,
    /// Count of paged→contiguous failovers when the private pool is exhausted.
    paged_pool_exhaustion_fallbacks: u64,
}

/// Ring geometry a forward presents to SDPA for a sliding-window layer when
/// the rotating decode path engages. `capacity == window + slack`; the ring
/// stores token `t` at slot `t % capacity`, so SDPA must mask slots whose
/// resident token falls outside a query's `(pos - window, pos]` range (or
/// was never written / rolled back). Produced by
/// [`MlxKVCache::sliding_ring_layout`]; the append site and every mask
/// builder must derive their decisions from this one predicate so view and
/// mask can never disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlidingRingLayout {
    pub window: usize,
    pub capacity: usize,
    /// Logical index of the first token this forward appends (`cache.seq_len`
    /// at forward entry).
    pub write_start: usize,
}

impl SlidingRingLayout {
    /// Whether SDPA needs an explicit slot-validity mask over the ring.
    /// Pure rings (`capacity == window`) are exactly full for single-token
    /// decode, so every slot is live and mask-free SDPA is correct.
    pub fn needs_mask(&self, seq: usize) -> bool {
        self.capacity > self.window || seq > 1
    }
}

impl Clone for MlxKVCache {
    fn clone(&self) -> Self {
        // Paged layers hold private physical IDs: re-allocate into a fresh
        // pool so free/trim on either side cannot double-free.
        let mut fa_pool = self.fa_pool.as_ref().map(|pool| {
            FaBlockPool::new(pool.config()).expect("clone reuses a validated pool config")
        });
        let layers = self
            .layers
            .iter()
            .map(|entry| match entry {
                None => None,
                Some(FaLayerStorage::Contiguous(lkv)) => {
                    Some(FaLayerStorage::Contiguous(lkv.clone()))
                }
                Some(FaLayerStorage::Paged(paged)) => {
                    let pool = fa_pool
                        .as_mut()
                        .expect("paged FA layer requires an FA block pool");
                    Some(FaLayerStorage::Paged(
                        paged
                            .clone_into_pool(pool)
                            .expect("clone must fit in a same-sized FA block pool"),
                    ))
                }
            })
            .collect();
        Self {
            layers,
            glm_mla_layers: self.glm_mla_layers.clone(),
            linear_layers: self.linear_layers.clone(),
            seq_len: self.seq_len,
            rope_offset: self.rope_offset,
            growth_count: self.growth_count,
            use_rotating_sliding_decode: self.use_rotating_sliding_decode,
            rotating_slack: self.rotating_slack,
            fa_pool,
            paged_materialize_us: self.paged_materialize_us,
            paged_pool_exhaustion_fallbacks: self.paged_pool_exhaustion_fallbacks,
        }
    }
}

impl MlxKVCache {
    // ── Wire-format constants for `serialize_to_bytes` / `try_deserialize_from_bytes` ──
    // The format is private to this module; see the F3 disk-cache PRD
    // (`MLX-DISK-PREFIX-CACHE-PRD-2026-05-14.md`) §3.3 / §4 for the rationale.
    /// File magic for the AX disk-format payload section. Distinct from
    /// `AXKV` (used by the future disk wrapper for outer file framing) so
    /// a partial / nested payload cannot be mistaken for a complete file.
    const SERIALIZE_MAGIC: &'static [u8; 4] = b"AXKB";
    const SERIALIZE_VERSION: u32 = 4;

    /// Private KV wire-format version, exposed for the durable prefix
    /// cache's canonical key (schema v3 commits to the payload version so a
    /// format bump cleanly invalidates older disk entries).
    pub const fn serialize_version() -> u32 {
        Self::SERIALIZE_VERSION
    }
    const LAYER_KIND_EMPTY: u8 = 0;
    const LAYER_KIND_FA: u8 = 1;
    const LAYER_KIND_MLA: u8 = 2;
    const LAYER_KIND_LINEAR: u8 = 3;
    const TENSOR_PRESENT_TAG: u8 = 1;
    const TENSOR_ABSENT_TAG: u8 = 0;

    pub fn new(num_layers: usize) -> Self {
        if fa_kv_block_pool_enabled() {
            Self::new_with_fa_block_pool(num_layers, default_fa_block_pool_config())
        } else {
            Self::new_contiguous(num_layers)
        }
    }

    /// Contiguous FA path (historical default). Used when the block-pool flag
    /// is off and by deserialize (wire format is always dense).
    pub fn new_contiguous(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            glm_mla_layers: (0..num_layers).map(|_| None).collect(),
            linear_layers: (0..num_layers)
                .map(|_| LinearLayerState::default())
                .collect(),
            seq_len: 0,
            rope_offset: 0,
            growth_count: 0,
            use_rotating_sliding_decode: false,
            rotating_slack: 0,
            fa_pool: None,
            paged_materialize_us: 0,
            paged_pool_exhaustion_fallbacks: 0,
        }
    }

    /// FA private block-pool path (PR4). Pure FA appends use paged storage and
    /// materialize dense K/V for SDPA; sliding/rotating layers stay contiguous.
    pub fn new_with_fa_block_pool(num_layers: usize, config: FaBlockPoolConfig) -> Self {
        let fa_pool = FaBlockPool::new(config).expect("FA block pool config must be non-zero");
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            glm_mla_layers: (0..num_layers).map(|_| None).collect(),
            linear_layers: (0..num_layers)
                .map(|_| LinearLayerState::default())
                .collect(),
            seq_len: 0,
            rope_offset: 0,
            growth_count: 0,
            use_rotating_sliding_decode: false,
            rotating_slack: 0,
            fa_pool: Some(fa_pool),
            paged_materialize_us: 0,
            paged_pool_exhaustion_fallbacks: 0,
        }
    }

    /// Align the private FA pool to the session `KvManager` geometry.
    ///
    /// Call after construction when the session knows `block_size_tokens` and
    /// `total_blocks`. No-op when the contiguous path is active. Replaces the
    /// pool only when empty (no live allocations).
    pub fn align_fa_block_pool_to_kv(
        &mut self,
        block_size_tokens: u32,
        max_blocks: u32,
    ) -> Result<(), FaBlockPoolError> {
        let Some(pool) = self.fa_pool.as_ref() else {
            return Ok(());
        };
        if pool.allocated_blocks() != 0 {
            return Ok(());
        }
        let config = FaBlockPoolConfig {
            block_size_tokens: block_size_tokens.max(1),
            max_blocks: max_blocks.max(1),
        };
        self.fa_pool = Some(FaBlockPool::new(config)?);
        Ok(())
    }

    /// Whether this cache owns a private FA block pool (paged pure-FA path).
    pub fn fa_block_pool_enabled(&self) -> bool {
        self.fa_pool.is_some()
    }

    /// Blocks available in the private FA pool, if paged mode is active.
    pub fn fa_block_pool_available(&self) -> Option<u32> {
        self.fa_pool.as_ref().map(FaBlockPool::available_blocks)
    }

    /// Current logical sequence length (token count cached).
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Advance the logical boundary after a forward pass appended `n`
    /// tokens to every KV-backed layer. Call once per forward (appends
    /// write at `seq_len` per layer, so the boundary must not move until
    /// all layers have appended).
    pub fn advance(&mut self, n: usize) {
        self.seq_len += n;
    }

    /// Set the logical boundary to an absolute position. Prefer
    /// [`Self::advance`] after forwards; this is for seeding a cache at a
    /// known position (prefill restore, warmup, tests). For rollback use
    /// [`Self::trim_to`], which also validates ring residency.
    pub fn set_seq_len(&mut self, n: usize) {
        self.seq_len = n;
    }

    pub fn set_rotating_sliding_decode(&mut self, enabled: bool) {
        self.use_rotating_sliding_decode = enabled;
    }

    /// Set the bounded-rollback slack for rotating sliding-window layers.
    /// Must be latched before the first rotating append of a request and
    /// never changed afterwards — converted rings keep their capacity.
    pub fn set_rotating_sliding_slack(&mut self, slack: usize) {
        self.rotating_slack = slack;
    }

    pub fn rotating_sliding_slack(&self) -> usize {
        self.rotating_slack
    }

    /// The ring geometry an `append_with_retained_window` call on a sliding
    /// layer will use for a forward of `seq` new tokens, or `None` when the
    /// forward stays on the ordered (non-rotating) path for that window.
    ///
    /// This is the single source of truth shared by the append site and the
    /// SDPA mask builders: a multi-token forward may enter the ring only in
    /// bounded mode (`rotating_slack > 0`) and only when it fits inside the
    /// slack, which also bounds the deepest `trim_to` rollback the ring can
    /// absorb (rolled-back tokens rewrite into their own `t % capacity`
    /// slots, so rollback itself is free).
    pub fn sliding_ring_layout(
        &self,
        window: Option<usize>,
        seq: usize,
    ) -> Option<SlidingRingLayout> {
        if !self.use_rotating_sliding_decode || seq == 0 {
            return None;
        }
        let window = window.filter(|window| *window > 0)?;
        if self.seq_len + seq <= window {
            return None;
        }
        let eligible = seq == 1 || (self.rotating_slack > 0 && seq <= self.rotating_slack);
        if !eligible {
            return None;
        }
        Some(SlidingRingLayout {
            window,
            capacity: window + self.rotating_slack,
            write_start: self.seq_len,
        })
    }

    // ── F3 M1: serialization for the disk-prefix-cache disk format ──
    //
    // These two methods are the foundation for the F3 disk-prefix-cache
    // PRD (MLX-DISK-PREFIX-CACHE-PRD-2026-05-14.md): a process-restart-
    // surviving second-tier cache. M1's goal is a round-trip-correct
    // wire format for the KV state. M2 (runner wiring), M3 (eviction +
    // concurrency), M4 (cross-restart validation), and M5 (docs) are
    // separate deliverables.
    //
    // Wire format (private to this module; the outer disk-file framing
    // is the F3 disk wrapper's responsibility):
    //
    //   header:
    //     magic[4]         = b"AXKB"
    //     version: u32     = 4 (v4 removed a legacy trailing payload section)
    //     seq_len: u64
    //     growth_count: u64
    //     rope_offset: u64  (v3+; extra RoPE position beyond seq_len,
    //                        e.g. after capped MTP warmup)
    //     layer_count: u32
    //     reserved: u32
    //
    //   for each layer 0..layer_count:
    //     kind: u8   (0=Empty, 1=FA, 2=MLA, 3=Linear)
    //     reserved[7]
    //     layer-kind-specific payload (see below)
    //
    //   FA payload:     [rotating_window: u64 (v3+; 0 = ordered storage,
    //                    otherwise the sliding window of a rotated ring whose
    //                    K/V tensors are slot-ordered, token t at slot
    //                    t % capacity)][K tensor][V tensor]
    //   MLA payload:    [kv_latent tensor][k_pe tensor]
    //   Linear payload: [tag:u8][optional conv_state][tag:u8][optional recurrent_state]
    //   Empty payload:  (nothing)
    //
    //   tensor encoding (32 bytes header + raw bytes):
    //     dtype: u8          (MlxDtype variant index 0..=13)
    //     ndim: u8           (1..=4)
    //     reserved[6]
    //     shape: [i32; 4]    (zero-padded for ndim<4)
    //     byte_count: u64
    //     bytes: [u8; byte_count]
    //
    fn dtype_to_tag(dtype: MlxDtype) -> u8 {
        match dtype {
            MlxDtype::Bool => 0,
            MlxDtype::Uint8 => 1,
            MlxDtype::Uint16 => 2,
            MlxDtype::Uint32 => 3,
            MlxDtype::Uint64 => 4,
            MlxDtype::Int8 => 5,
            MlxDtype::Int16 => 6,
            MlxDtype::Int32 => 7,
            MlxDtype::Int64 => 8,
            MlxDtype::Float16 => 9,
            MlxDtype::Float32 => 10,
            MlxDtype::Float64 => 11,
            MlxDtype::Bfloat16 => 12,
            MlxDtype::Complex64 => 13,
        }
    }

    fn dtype_from_tag(tag: u8) -> Result<MlxDtype, MlxKVCacheSerializeError> {
        Ok(match tag {
            0 => MlxDtype::Bool,
            1 => MlxDtype::Uint8,
            2 => MlxDtype::Uint16,
            3 => MlxDtype::Uint32,
            4 => MlxDtype::Uint64,
            5 => MlxDtype::Int8,
            6 => MlxDtype::Int16,
            7 => MlxDtype::Int32,
            8 => MlxDtype::Int64,
            9 => MlxDtype::Float16,
            10 => MlxDtype::Float32,
            11 => MlxDtype::Float64,
            12 => MlxDtype::Bfloat16,
            13 => MlxDtype::Complex64,
            other => return Err(MlxKVCacheSerializeError::UnknownDtype(other)),
        })
    }

    /// Serialize only the logical token prefix of a `[B, H, tokens, D]`
    /// tensor (token axis 2). The backing buffer is capacity-sized, so
    /// serializing it whole makes every snapshot cost O(capacity) bytes —
    /// and the prefix-snapshot store serializes one snapshot per
    /// block-aligned prefix, turning that into O(prefixes × capacity)
    /// memcpy per prefill. The wire format is unchanged: the deserializer
    /// derives capacity from the stored shape and regrows on append.
    fn serialize_tensor_logical(out: &mut Vec<u8>, arr: &MlxArray, logical_tokens: Option<usize>) {
        let shape = arr.shape();
        if let Some(tokens) = logical_tokens
            && shape.len() == 4
            && (shape[2] as usize) > tokens
        {
            let stop = [shape[0], shape[1], tokens as i32, shape[3]];
            let trimmed = slice(arr, &[0, 0, 0, 0], &stop, &[1, 1, 1, 1], None);
            // The slice is a strided view over the capacity buffer;
            // materialize it so `data_raw` reads row-contiguous bytes.
            let trimmed = contiguous(&trimmed, None);
            Self::serialize_tensor(out, &trimmed);
        } else {
            Self::serialize_tensor(out, arr);
        }
    }

    fn serialize_tensor(out: &mut Vec<u8>, arr: &MlxArray) {
        // Materialise before reading bytes; data_raw is only valid post-eval.
        eval(&[arr]);
        let dtype_tag = Self::dtype_to_tag(arr.dtype());
        let shape = arr.shape();
        let ndim = shape.len() as u8;
        debug_assert!(shape.len() <= 4, "tensor ndim must be ≤ 4");

        out.push(dtype_tag);
        out.push(ndim);
        out.extend_from_slice(&[0u8; 6]); // reserved
        let mut padded_shape = [0i32; 4];
        for (i, &s) in shape.iter().enumerate() {
            padded_shape[i] = s;
        }
        for s in padded_shape {
            out.extend_from_slice(&s.to_le_bytes());
        }

        let byte_count = arr.nbytes() as u64;
        out.extend_from_slice(&byte_count.to_le_bytes());

        // SAFETY: data_raw returns a host-visible pointer after eval. We
        // copy `byte_count` bytes; the slice is bounded by the array's
        // own reported size.
        unsafe {
            let ptr = arr.data_raw();
            let slice = std::slice::from_raw_parts(ptr, byte_count as usize);
            out.extend_from_slice(slice);
        }
    }

    /// Read one tensor from a streaming reader directly into its final
    /// owned buffer (spec §7.2 / DTPC-007). Peak transient cost beyond the
    /// completed tensors is one tensor buffer — not a second full-payload copy.
    fn read_tensor_from_reader(
        reader: &mut dyn std::io::Read,
    ) -> Result<MlxArray, MlxKVCacheSerializeError> {
        let dtype_tag = read_u8_from(reader)?;
        let ndim = read_u8_from(reader)? as usize;
        if ndim == 0 || ndim > 4 {
            return Err(MlxKVCacheSerializeError::BadShape(ndim));
        }
        let mut reserved = [0u8; 6];
        read_exact_from(reader, &mut reserved)?;
        let mut shape = [0i32; 4];
        for s in &mut shape {
            *s = read_i32_from(reader)?;
        }
        let dtype = Self::dtype_from_tag(dtype_tag)?;
        let byte_count = read_u64_from(reader)? as usize;
        Self::validate_tensor_byte_count(ndim, &shape, dtype, byte_count)?;
        let mut owned = vec![0u8; byte_count];
        read_exact_from(reader, &mut owned)?;
        Self::mlx_array_from_owned_bytes(owned, &shape[..ndim], dtype)
    }

    fn validate_tensor_byte_count(
        ndim: usize,
        shape: &[i32; 4],
        dtype: MlxDtype,
        byte_count: usize,
    ) -> Result<(), MlxKVCacheSerializeError> {
        // Pre-validate shape × dtype against `byte_count` so a tampered or
        // corrupted payload returns a structured error instead of tripping
        // the assert inside `MlxArray::from_managed_data`.
        let mut element_count: usize = 1;
        for &dim in shape[..ndim].iter() {
            if dim < 0 {
                return Err(MlxKVCacheSerializeError::BadShape(ndim));
            }
            element_count = element_count
                .checked_mul(dim as usize)
                .ok_or(MlxKVCacheSerializeError::BadShape(ndim))?;
        }
        let required_bytes = element_count
            .checked_mul(dtype.size_bytes())
            .ok_or(MlxKVCacheSerializeError::BadShape(ndim))?;
        if byte_count < required_bytes {
            return Err(MlxKVCacheSerializeError::BadShape(ndim));
        }
        Ok(())
    }

    fn mlx_array_from_owned_bytes(
        owned: Vec<u8>,
        shape: &[i32],
        dtype: MlxDtype,
    ) -> Result<MlxArray, MlxKVCacheSerializeError> {
        // `MlxArray::from_raw_data` borrows the data buffer (MLX does not
        // copy), so the deserializer must own the bytes for the array's
        // lifetime via `from_managed_data` + `vec_payload_drop`.
        let byte_count = owned.len();
        let owned: Box<Vec<u8>> = Box::new(owned);
        let data_ptr = owned.as_ptr();
        let payload = Box::into_raw(owned) as *mut std::ffi::c_void;
        // SAFETY: data_ptr points at heap memory owned by the boxed Vec.
        // The Vec's buffer outlives the MlxArray because `vec_payload_drop`
        // only fires when MLX releases the array's last reference.
        Ok(unsafe {
            MlxArray::from_managed_data(
                data_ptr,
                byte_count,
                shape,
                dtype,
                payload,
                vec_payload_drop,
            )
        })
    }

    /// Serialise the cache to a self-contained byte payload that
    /// `try_deserialize_from_bytes` can reconstruct. Format is private to
    /// this module and versioned via `SERIALIZE_VERSION`; cross-version
    /// reads return an error rather than silently degrading.
    ///
    pub fn serialize_to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(Self::SERIALIZE_MAGIC);
        out.extend_from_slice(&Self::SERIALIZE_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.seq_len as u64).to_le_bytes());
        out.extend_from_slice(&self.growth_count.to_le_bytes());
        out.extend_from_slice(&(self.rope_offset as u64).to_le_bytes());
        let layer_count = self.layers.len() as u32;
        out.extend_from_slice(&layer_count.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved

        for idx in 0..self.layers.len() {
            // Per-index disambiguation: at most one of the three layer
            // vectors is populated. The encoded `kind` byte tells the
            // reader which payload to expect.
            if let Some(fa) = &self.layers[idx] {
                out.push(Self::LAYER_KIND_FA);
                out.extend_from_slice(&[0u8; 7]);
                // Ring geometry must survive the round trip: a rotated
                // layer's buffer is slot-ordered, and restoring it as
                // ordered storage would make the first post-restore append
                // treat ring slots as a token-ordered prefix.
                // Paged FA always serializes as dense contiguous (no I-2 bump).
                let rotating_window = fa.rotating_window();
                out.extend_from_slice(&(rotating_window.unwrap_or(0) as u64).to_le_bytes());
                let logical_tokens = if rotating_window.is_none() {
                    Some(self.seq_len)
                } else {
                    None
                };
                match fa {
                    FaLayerStorage::Contiguous(lkv) => {
                        Self::serialize_tensor_logical(&mut out, &lkv.k, logical_tokens);
                        Self::serialize_tensor_logical(&mut out, &lkv.v, logical_tokens);
                    }
                    FaLayerStorage::Paged(paged) => {
                        // serialize is rare vs decode; time is not accumulated
                        // into `paged_materialize_us` (that field tracks live
                        // SDPA materialize on the append path).
                        let (k, v) = paged.materialize(0, self.seq_len);
                        Self::serialize_tensor_logical(&mut out, &k, logical_tokens);
                        Self::serialize_tensor_logical(&mut out, &v, logical_tokens);
                    }
                }
            } else if let Some(mla) = &self.glm_mla_layers[idx] {
                out.push(Self::LAYER_KIND_MLA);
                out.extend_from_slice(&[0u8; 7]);
                Self::serialize_tensor_logical(&mut out, &mla.kv_latent, Some(self.seq_len));
                Self::serialize_tensor_logical(&mut out, &mla.k_pe, Some(self.seq_len));
            } else if self.linear_layers[idx].conv_state.is_some()
                || self.linear_layers[idx].recurrent_state.is_some()
            {
                let linear = &self.linear_layers[idx];
                out.push(Self::LAYER_KIND_LINEAR);
                out.extend_from_slice(&[0u8; 7]);
                if let Some(arr) = &linear.conv_state {
                    out.push(Self::TENSOR_PRESENT_TAG);
                    Self::serialize_tensor(&mut out, arr);
                } else {
                    out.push(Self::TENSOR_ABSENT_TAG);
                }
                if let Some(arr) = &linear.recurrent_state {
                    out.push(Self::TENSOR_PRESENT_TAG);
                    Self::serialize_tensor(&mut out, arr);
                } else {
                    out.push(Self::TENSOR_ABSENT_TAG);
                }
            } else {
                out.push(Self::LAYER_KIND_EMPTY);
                out.extend_from_slice(&[0u8; 7]);
            }
        }

        out
    }

    /// Reconstruct a cache from a byte payload produced by
    /// `serialize_to_bytes`. Returns an error on magic / version
    /// mismatch, truncated data, unknown dtype tags, or shape errors —
    /// never silently degrades.
    pub fn try_deserialize_from_bytes(bytes: &[u8]) -> Result<Self, MlxKVCacheSerializeError> {
        let mut cursor = std::io::Cursor::new(bytes);
        Self::try_deserialize_from_reader(&mut cursor)
    }

    /// Streaming deserialize for durable L2 restore (spec §7.2 / §9).
    /// Each tensor is read directly into its final owned buffer while the
    /// caller updates integrity state (e.g. entry SHA-256). Prefer this
    /// over materializing a full payload `Vec` then copying again.
    pub fn try_deserialize_from_reader(
        reader: &mut dyn std::io::Read,
    ) -> Result<Self, MlxKVCacheSerializeError> {
        let mut magic = [0u8; 4];
        read_exact_from(reader, &mut magic)?;
        if magic != *Self::SERIALIZE_MAGIC {
            return Err(MlxKVCacheSerializeError::BadMagic);
        }
        let version = read_u32_from(reader)?;
        if version != Self::SERIALIZE_VERSION {
            return Err(MlxKVCacheSerializeError::UnsupportedVersion(version));
        }
        let seq_len = read_u64_from(reader)? as usize;
        let growth_count = read_u64_from(reader)?;
        let rope_offset = usize::try_from(read_u64_from(reader)?)
            .map_err(|_| MlxKVCacheSerializeError::BadShape(8))?;
        let layer_count = read_u32_from(reader)? as usize;
        let _reserved = read_u32_from(reader)?;

        // Wire format is always dense contiguous; do not inherit env-flag
        // paged mode into restored snapshots (I-2 payload is contiguous).
        let mut cache = Self::new_contiguous(layer_count);
        cache.seq_len = seq_len;
        cache.growth_count = growth_count;
        cache.rope_offset = rope_offset;

        for idx in 0..layer_count {
            let kind = read_u8_from(reader)?;
            let mut reserved = [0u8; 7];
            read_exact_from(reader, &mut reserved)?;
            match kind {
                k if k == Self::LAYER_KIND_EMPTY => continue,
                k if k == Self::LAYER_KIND_FA => {
                    let ring_window = usize::try_from(read_u64_from(reader)?)
                        .map_err(|_| MlxKVCacheSerializeError::BadShape(8))?;
                    let k_arr = Self::read_tensor_from_reader(reader)?;
                    let v_arr = Self::read_tensor_from_reader(reader)?;
                    let shape = k_arr.shape();
                    if shape.len() < 4 {
                        return Err(MlxKVCacheSerializeError::BadShape(shape.len()));
                    }
                    let capacity = shape[2] as usize;
                    let rotating_window = (ring_window != 0).then_some(ring_window);
                    if let Some(window) = rotating_window {
                        // A ring narrower than its window (or an ordered
                        // buffer shorter than seq_len) cannot have been
                        // produced by the serializer.
                        if capacity < window {
                            return Err(MlxKVCacheSerializeError::BadShape(shape.len()));
                        }
                        // Re-latch the ring configuration so post-restore
                        // appends reproduce the same geometry instead of
                        // reconverting (which would read slot-ordered data
                        // as token order).
                        cache.use_rotating_sliding_decode = true;
                        cache.rotating_slack = capacity - window;
                    } else if seq_len > capacity {
                        return Err(MlxKVCacheSerializeError::BadShape(shape.len()));
                    }
                    cache.layers[idx] = Some(FaLayerStorage::Contiguous(LayerKV {
                        last_k_view: None,
                        last_v_view: None,
                        n_kv_heads: shape[1],
                        head_dim: shape[3],
                        capacity,
                        rotating_window,
                        dtype: k_arr.dtype(),
                        k: k_arr,
                        v: v_arr,
                    }));
                }
                k if k == Self::LAYER_KIND_MLA => {
                    let kv_latent = Self::read_tensor_from_reader(reader)?;
                    let k_pe = Self::read_tensor_from_reader(reader)?;
                    let kv_shape = kv_latent.shape();
                    let pe_shape = k_pe.shape();
                    if kv_shape.len() < 4 || pe_shape.len() < 4 {
                        return Err(MlxKVCacheSerializeError::BadShape(kv_shape.len()));
                    }
                    cache.glm_mla_layers[idx] = Some(GlmMlaLayerCache {
                        latent_dim: kv_shape[3],
                        rope_dim: pe_shape[3],
                        capacity: kv_shape[2] as usize,
                        dtype: kv_latent.dtype(),
                        kv_latent,
                        k_pe,
                    });
                }
                k if k == Self::LAYER_KIND_LINEAR => {
                    let conv_present = read_u8_from(reader)?;
                    let conv_state = if conv_present == Self::TENSOR_PRESENT_TAG {
                        Some(Self::read_tensor_from_reader(reader)?)
                    } else {
                        None
                    };
                    let rec_present = read_u8_from(reader)?;
                    let recurrent_state = if rec_present == Self::TENSOR_PRESENT_TAG {
                        Some(Self::read_tensor_from_reader(reader)?)
                    } else {
                        None
                    };
                    cache.linear_layers[idx] = LinearLayerState {
                        conv_state,
                        recurrent_state,
                    };
                }
                other => return Err(MlxKVCacheSerializeError::UnknownLayerKind(other)),
            }
        }

        Ok(cache)
    }

    /// Append new K/V tokens for `layer` and return the full logical K/V for SDPA.
    ///
    /// `new_k` / `new_v` shape: `[1, n_kv_heads, new_tokens, head_dim]`
    ///
    /// Returns **owned** arrays sliced to `[1, n_kv_heads, seq_len + new_tokens, head_dim]`.
    pub fn append(
        &mut self,
        layer: usize,
        new_k: MlxArray,
        new_v: MlxArray,
    ) -> (MlxArray, MlxArray) {
        self.append_with_retained_window(layer, new_k, new_v, None)
    }

    /// Append new K/V tokens and return a logical view retained to `window` tokens.
    ///
    /// This is used for Gemma-family sliding-window decode.  Upstream `mlx_lm`
    /// and `mlx-swift-lm` use rotating caches for sliding layers, so SDPA only
    /// sees the retained window instead of the full context.  AX uses the same
    /// bounded backing store only when the request is on the non-rollback direct
    /// decode path; rollback-capable paths keep full backing storage and return a
    /// shorter view.
    ///
    /// `window = None` preserves the full-view behavior of `append`.
    pub fn append_with_retained_window(
        &mut self,
        layer: usize,
        new_k: MlxArray,
        new_v: MlxArray,
        window: Option<usize>,
    ) -> (MlxArray, MlxArray) {
        let append = validate_append_inputs(layer, self.layers.len(), &new_k, &new_v);
        let new_tokens = append.new_tokens;
        let write_start = self.seq_len;
        let write_end = write_start + new_tokens;
        let n_kv_heads = append.n_kv_heads;
        let head_dim = append.head_dim;
        let dtype = append.dtype;

        if let Some(ring) = self.sliding_ring_layout(window, new_tokens)
            && self.layers[layer].is_some()
        {
            return self.append_rotating_retained_window(layer, new_k, new_v, ring);
        }

        // Pure-FA paged path: only for empty or already-paged layers when a
        // private pool is present. Sliding retained windows stay contiguous.
        let use_paged = self.fa_pool.is_some()
            && window.is_none()
            && matches!(self.layers[layer], None | Some(FaLayerStorage::Paged(_)));
        if use_paged {
            return self.append_paged_fa(layer, new_k, new_v, write_start, write_end, append);
        }

        let entry = &mut self.layers[layer];
        match entry {
            None => {
                let capacity = chunk_ceiling(write_end);
                // Fresh-layer fast path: when the prompt is chunk-aligned
                // (capacity == new_tokens), skip zeros+slice_update by storing
                // new_k/new_v directly.  Capacity stays correct for usage_snapshot;
                // the first decode step grows to chunk_ceiling(new_tokens + 1) as
                // normal.  Saves ~6 MLX graph nodes per layer per chunk-aligned prefill
                // (e.g. 512-token prompt → 210 fewer nodes for a 35-layer model).
                if write_start == 0 && capacity == new_tokens {
                    let view_start = window
                        .filter(|&w| w > 0 && w < new_tokens)
                        .map(|w| new_tokens - w)
                        .unwrap_or(0);
                    let (k_view, v_view) = if view_start > 0 {
                        let s = view_start as i32;
                        let e = new_tokens as i32;
                        let kv = slice(
                            &new_k,
                            &[0, 0, s, 0],
                            &[1, n_kv_heads, e, head_dim],
                            &[1, 1, 1, 1],
                            None,
                        );
                        let vv = slice(
                            &new_v,
                            &[0, 0, s, 0],
                            &[1, n_kv_heads, e, head_dim],
                            &[1, 1, 1, 1],
                            None,
                        );
                        (kv, vv)
                    } else {
                        (new_k.clone(), new_v.clone())
                    };
                    self.growth_count = self.growth_count.saturating_add(1);
                    *entry = Some(FaLayerStorage::Contiguous(LayerKV {
                        k: new_k,
                        v: new_v,
                        last_k_view: Some(k_view.clone()),
                        last_v_view: Some(v_view.clone()),
                        n_kv_heads,
                        head_dim,
                        capacity,
                        rotating_window: None,
                        dtype,
                    }));
                    return (k_view, v_view);
                }
                let buf_shape = [1i32, n_kv_heads, capacity as i32, head_dim];
                let k_buf = zeros(&buf_shape, dtype, None);
                let v_buf = zeros(&buf_shape, dtype, None);
                let start = [0i32, 0, write_start as i32, 0];
                let stop = [1i32, n_kv_heads, write_end as i32, head_dim];
                let strides = [1i32, 1, 1, 1];
                let k_out = slice_update(&k_buf, &new_k, &start, &stop, &strides, None);
                let v_out = slice_update(&v_buf, &new_v, &start, &stop, &strides, None);
                self.growth_count = self.growth_count.saturating_add(1);
                *entry = Some(FaLayerStorage::Contiguous(LayerKV {
                    k: k_out,
                    v: v_out,
                    last_k_view: None,
                    last_v_view: None,
                    n_kv_heads,
                    head_dim,
                    capacity,
                    rotating_window: None,
                    dtype,
                }));
            }
            Some(FaLayerStorage::Paged(_)) => {
                unreachable!("paged FA layers must use append_paged_fa");
            }
            Some(FaLayerStorage::Contiguous(lkv)) => {
                // A rotated ring stores slots, not token order; writing at
                // logical positions (or growing, which copies slots as an
                // ordered prefix) would silently corrupt it. Ring-eligible
                // forwards must go through `append_rotating_retained_window`;
                // anything else on a rotated layer is a caller bug.
                assert!(
                    lkv.rotating_window.is_none(),
                    "ordered KV append on rotated ring layer {layer} (window {:?}, capacity {}): \
                     forward of {new_tokens} tokens is not ring-eligible \
                     (rotating_slack {}) and would corrupt slot-ordered state",
                    lkv.rotating_window,
                    lkv.capacity,
                    self.rotating_slack,
                );
                assert_eq!(
                    lkv.n_kv_heads, n_kv_heads,
                    "KV cache append cannot change n_kv_heads for an existing layer"
                );
                assert_eq!(
                    lkv.head_dim, head_dim,
                    "KV cache append cannot change head_dim for an existing layer"
                );
                assert_eq!(
                    lkv.dtype, dtype,
                    "KV cache append cannot change dtype for an existing layer"
                );
                if write_end > lkv.capacity {
                    // Grow: allocate a larger buffer and copy existing data.
                    let new_capacity = chunk_ceiling(write_end);
                    let buf_shape = [1i32, lkv.n_kv_heads, new_capacity as i32, lkv.head_dim];
                    let k_new = zeros(&buf_shape, lkv.dtype, None);
                    let v_new = zeros(&buf_shape, lkv.dtype, None);
                    let old_stop = [1i32, lkv.n_kv_heads, lkv.capacity as i32, lkv.head_dim];
                    let zero_start = [0i32, 0, 0, 0];
                    let ones = [1i32, 1, 1, 1];
                    lkv.k = slice_update(&k_new, &lkv.k, &zero_start, &old_stop, &ones, None);
                    lkv.v = slice_update(&v_new, &lkv.v, &zero_start, &old_stop, &ones, None);
                    lkv.capacity = new_capacity;
                    // Invalidate cached views — they point to the old (smaller) buffer.
                    lkv.last_k_view = None;
                    lkv.last_v_view = None;
                    self.growth_count = self.growth_count.saturating_add(1);
                }
                let start = [0i32, 0, write_start as i32, 0];
                let stop = [1i32, lkv.n_kv_heads, write_end as i32, lkv.head_dim];
                let strides = [1i32, 1, 1, 1];
                lkv.k = slice_update(&lkv.k, &new_k, &start, &stop, &strides, None);
                lkv.v = slice_update(&lkv.v, &new_v, &start, &stop, &strides, None);
            }
        }

        let lkv = self.layers[layer]
            .as_mut()
            .and_then(FaLayerStorage::as_contiguous_mut)
            .expect("contiguous FA append path");
        let view_start = window
            .filter(|window| *window > 0)
            .map(|window| write_end.saturating_sub(window))
            .unwrap_or(0);
        let start = view_start as i32;
        let end = write_end as i32;
        let k_view = slice(
            &lkv.k,
            &[0, 0, start, 0],
            &[1, lkv.n_kv_heads, end, lkv.head_dim],
            &[1, 1, 1, 1],
            None,
        );
        let v_view = slice(
            &lkv.v,
            &[0, 0, start, 0],
            &[1, lkv.n_kv_heads, end, lkv.head_dim],
            &[1, 1, 1, 1],
            None,
        );
        // Cache the views so KV-shared layers (Gemma4) can reuse them without
        // creating a second identical slice node on the same backing buffer.
        lkv.last_k_view = Some(k_view.clone());
        lkv.last_v_view = Some(v_view.clone());
        (k_view, v_view)
    }

    fn append_paged_fa(
        &mut self,
        layer: usize,
        new_k: MlxArray,
        new_v: MlxArray,
        write_start: usize,
        write_end: usize,
        append: AppendShape,
    ) -> (MlxArray, MlxArray) {
        let block_size = self
            .fa_pool
            .as_ref()
            .expect("paged append requires FA block pool")
            .config()
            .block_size_tokens as usize;

        if self.layers[layer].is_none() {
            self.layers[layer] = Some(FaLayerStorage::Paged(PagedFaLayer {
                n_kv_heads: append.n_kv_heads,
                head_dim: append.head_dim,
                dtype: append.dtype,
                block_size,
                block_ids: Vec::new(),
                k_blocks: Vec::new(),
                v_blocks: Vec::new(),
                last_k_view: None,
                last_v_view: None,
            }));
        }

        let need_demote = {
            let pool = self
                .fa_pool
                .as_mut()
                .expect("paged append requires FA block pool");
            let paged = match self.layers[layer]
                .as_mut()
                .expect("paged layer just created")
            {
                FaLayerStorage::Paged(p) => p,
                FaLayerStorage::Contiguous(_) => {
                    panic!("append_paged_fa on contiguous layer {layer}")
                }
            };
            assert_eq!(
                paged.n_kv_heads, append.n_kv_heads,
                "paged FA append cannot change n_kv_heads for an existing layer"
            );
            assert_eq!(
                paged.head_dim, append.head_dim,
                "paged FA append cannot change head_dim for an existing layer"
            );
            assert_eq!(
                paged.dtype, append.dtype,
                "paged FA append cannot change dtype for an existing layer"
            );
            match paged.ensure_capacity(pool, write_end, &mut self.growth_count) {
                Ok(()) => {
                    paged.write_tokens(write_start, &new_k, &new_v);
                    false
                }
                Err(FaBlockPoolError::Exhausted { .. }) => true,
                Err(err) => panic!("FA block pool error on layer {layer}: {err}"),
            }
        };

        if need_demote {
            // Fail-soft: materialize existing private blocks into contiguous
            // storage and finish this append on the historical growth path so
            // serving does not abort when the private pool is exhausted.
            self.paged_pool_exhaustion_fallbacks =
                self.paged_pool_exhaustion_fallbacks.saturating_add(1);
            self.demote_paged_layer_to_contiguous(layer, write_start);
            return self.append_with_retained_window(layer, new_k, new_v, None);
        }

        let paged = match self.layers[layer]
            .as_mut()
            .expect("paged layer after write")
        {
            FaLayerStorage::Paged(p) => p,
            FaLayerStorage::Contiguous(_) => unreachable!("paged path"),
        };
        let started = Instant::now();
        let (k_view, v_view) = paged.materialize(0, write_end);
        self.paged_materialize_us = self
            .paged_materialize_us
            .saturating_add(started.elapsed().as_micros() as u64);
        paged.last_k_view = Some(k_view.clone());
        paged.last_v_view = Some(v_view.clone());
        (k_view, v_view)
    }

    /// Convert a paged FA layer into contiguous storage up to `logical_len`
    /// tokens and free its private blocks. Used on pool exhaustion.
    fn demote_paged_layer_to_contiguous(&mut self, layer: usize, logical_len: usize) {
        let Some(FaLayerStorage::Paged(paged)) = self.layers.get_mut(layer).and_then(Option::take)
        else {
            return;
        };
        let started = Instant::now();
        let (k_view, v_view) = if logical_len == 0 {
            let shape = [1i32, paged.n_kv_heads, 0, paged.head_dim];
            (
                zeros(&shape, paged.dtype, None),
                zeros(&shape, paged.dtype, None),
            )
        } else {
            paged.materialize(0, logical_len)
        };
        self.paged_materialize_us = self
            .paged_materialize_us
            .saturating_add(started.elapsed().as_micros() as u64);
        if let Some(pool) = self.fa_pool.as_mut() {
            let _ = pool.free(&paged.block_ids);
        }
        let capacity = if logical_len == 0 {
            0
        } else {
            chunk_ceiling(logical_len)
        };
        let (k, v) = if logical_len == 0 || capacity == logical_len {
            (k_view, v_view)
        } else {
            let buf_shape = [1i32, paged.n_kv_heads, capacity as i32, paged.head_dim];
            let k_buf = zeros(&buf_shape, paged.dtype, None);
            let v_buf = zeros(&buf_shape, paged.dtype, None);
            let start = [0i32, 0, 0, 0];
            let stop = [1i32, paged.n_kv_heads, logical_len as i32, paged.head_dim];
            let strides = [1i32, 1, 1, 1];
            (
                slice_update(&k_buf, &k_view, &start, &stop, &strides, None),
                slice_update(&v_buf, &v_view, &start, &stop, &strides, None),
            )
        };
        if let Some(slot) = self.layers.get_mut(layer) {
            *slot = Some(FaLayerStorage::Contiguous(LayerKV {
                k,
                v,
                last_k_view: None,
                last_v_view: None,
                n_kv_heads: paged.n_kv_heads,
                head_dim: paged.head_dim,
                capacity,
                rotating_window: None,
                dtype: paged.dtype,
            }));
        }
    }

    fn append_rotating_retained_window(
        &mut self,
        layer: usize,
        new_k: MlxArray,
        new_v: MlxArray,
        ring: SlidingRingLayout,
    ) -> (MlxArray, MlxArray) {
        let SlidingRingLayout {
            window,
            capacity,
            write_start,
        } = ring;
        let new_tokens = new_k.shape()[2] as usize;
        let lkv = self.layers[layer]
            .as_mut()
            .and_then(FaLayerStorage::as_contiguous_mut)
            .expect(
                "rotating sliding decode requires an existing contiguous prefill cache \
                 (paged pure-FA layers are not ring-converted in PR4)",
            );
        if lkv.rotating_window != Some(window) || lkv.capacity != capacity {
            // Conversion reads the source at logical token indices, which is
            // only meaningful for ordered storage. An already-rotated layer
            // reaching here means the ring geometry changed mid-request
            // (window or slack drift) — reconverting would read slot-ordered
            // data as token order.
            assert!(
                lkv.rotating_window.is_none(),
                "sliding ring geometry changed mid-request for layer {layer}: \
                 existing window {:?} capacity {}, requested window {window} capacity {capacity}",
                lkv.rotating_window,
                lkv.capacity,
            );
            let k_old = lkv.k.clone();
            let v_old = lkv.v.clone();
            let buf_shape = [1i32, lkv.n_kv_heads, capacity as i32, lkv.head_dim];
            let k_new = zeros(&buf_shape, lkv.dtype, None);
            let v_new = zeros(&buf_shape, lkv.dtype, None);
            let old_start = write_start.saturating_add(1).saturating_sub(window);
            let old_end = write_start;
            let k_new =
                copy_token_range_to_rotating(&k_old, &k_new, lkv, old_start, old_end, capacity);
            let v_new =
                copy_token_range_to_rotating(&v_old, &v_new, lkv, old_start, old_end, capacity);
            lkv.k = k_new;
            lkv.v = v_new;
            lkv.capacity = capacity;
            lkv.rotating_window = Some(window);
            lkv.last_k_view = None;
            lkv.last_v_view = None;
        }

        if new_tokens == 1 {
            let write_pos = (write_start % capacity) as i32;
            let start = [0i32, 0, write_pos, 0];
            let stop = [1i32, lkv.n_kv_heads, write_pos + 1, lkv.head_dim];
            let strides = [1i32, 1, 1, 1];
            lkv.k = slice_update(&lkv.k, &new_k, &start, &stop, &strides, None);
            lkv.v = slice_update(&lkv.v, &new_v, &start, &stop, &strides, None);
            lkv.last_k_view = Some(lkv.k.clone());
            lkv.last_v_view = Some(lkv.v.clone());
            return (lkv.k.clone(), lkv.v.clone());
        }

        // Write the new tokens at their `t % capacity` slots. A multi-token
        // append (bounded rings only; `new_tokens <= slack < capacity`) wraps
        // at most once, so this loop issues one or two slice_updates.
        let mut src_start = 0usize;
        while src_start < new_tokens {
            let dst_start = (write_start + src_start) % capacity;
            let len = (new_tokens - src_start).min(capacity - dst_start);
            let (seg_k, seg_v) = if src_start == 0 && len == new_tokens {
                (new_k.clone(), new_v.clone())
            } else {
                let src_stop = (src_start + len) as i32;
                let seg_start = [0i32, 0, src_start as i32, 0];
                let seg_stop = [1i32, lkv.n_kv_heads, src_stop, lkv.head_dim];
                let ones = [1i32, 1, 1, 1];
                (
                    slice(&new_k, &seg_start, &seg_stop, &ones, None),
                    slice(&new_v, &seg_start, &seg_stop, &ones, None),
                )
            };
            let start = [0i32, 0, dst_start as i32, 0];
            let stop = [1i32, lkv.n_kv_heads, (dst_start + len) as i32, lkv.head_dim];
            let strides = [1i32, 1, 1, 1];
            lkv.k = slice_update(&lkv.k, &seg_k, &start, &stop, &strides, None);
            lkv.v = slice_update(&lkv.v, &seg_v, &start, &stop, &strides, None);
            src_start += len;
        }
        lkv.last_k_view = Some(lkv.k.clone());
        lkv.last_v_view = Some(lkv.v.clone());
        (lkv.k.clone(), lkv.v.clone())
    }

    /// Append GLM4MoELite MLA cache tokens and return full logical latent/KRoPE views.
    ///
    /// Shape convention follows mlx-lm's `cache.update_and_fetch(kv_latent, k_pe)`:
    /// `new_kv_latent`: `[1, 1, new_tokens, kv_lora_rank]`
    /// `new_k_pe`: `[1, 1, new_tokens, qk_rope_head_dim]`
    ///
    /// This cache stores the compressed MLA representation, not expanded K/V.
    pub fn append_glm_mla(
        &mut self,
        layer: usize,
        new_kv_latent: MlxArray,
        new_k_pe: MlxArray,
    ) -> (MlxArray, MlxArray) {
        let append = validate_glm_mla_append_inputs(
            layer,
            self.glm_mla_layers.len(),
            &new_kv_latent,
            &new_k_pe,
        );
        let new_tokens = append.new_tokens;
        let write_start = self.seq_len;
        let write_end = write_start + new_tokens;
        let dtype = append.dtype;
        let latent_dim = append.latent_dim;
        let rope_dim = append.rope_dim;

        let entry = &mut self.glm_mla_layers[layer];
        match entry {
            None => {
                let capacity = chunk_ceiling(write_end);
                if write_start == 0 && capacity == new_tokens {
                    self.growth_count = self.growth_count.saturating_add(1);
                    *entry = Some(GlmMlaLayerCache {
                        kv_latent: new_kv_latent.clone(),
                        k_pe: new_k_pe.clone(),
                        latent_dim,
                        rope_dim,
                        capacity,
                        dtype,
                    });
                    return (new_kv_latent, new_k_pe);
                }
                let latent_shape = [1i32, 1, capacity as i32, latent_dim];
                let rope_shape = [1i32, 1, capacity as i32, rope_dim];
                let latent_buf = zeros(&latent_shape, dtype, None);
                let rope_buf = zeros(&rope_shape, dtype, None);
                let start = [0i32, 0, write_start as i32, 0];
                let latent_stop = [1i32, 1, write_end as i32, latent_dim];
                let rope_stop = [1i32, 1, write_end as i32, rope_dim];
                let strides = [1i32, 1, 1, 1];
                let kv_latent = slice_update(
                    &latent_buf,
                    &new_kv_latent,
                    &start,
                    &latent_stop,
                    &strides,
                    None,
                );
                let k_pe = slice_update(&rope_buf, &new_k_pe, &start, &rope_stop, &strides, None);
                self.growth_count = self.growth_count.saturating_add(1);
                *entry = Some(GlmMlaLayerCache {
                    kv_latent,
                    k_pe,
                    latent_dim,
                    rope_dim,
                    capacity,
                    dtype,
                });
            }
            Some(cache) => {
                assert_eq!(
                    cache.latent_dim, latent_dim,
                    "GLM MLA cache append cannot change kv_lora_rank for an existing layer"
                );
                assert_eq!(
                    cache.rope_dim, rope_dim,
                    "GLM MLA cache append cannot change qk_rope_head_dim for an existing layer"
                );
                assert_eq!(
                    cache.dtype, dtype,
                    "GLM MLA cache append cannot change dtype for an existing layer"
                );
                if write_end > cache.capacity {
                    let new_capacity = chunk_ceiling(write_end);
                    let latent_shape = [1i32, 1, new_capacity as i32, cache.latent_dim];
                    let rope_shape = [1i32, 1, new_capacity as i32, cache.rope_dim];
                    let latent_new = zeros(&latent_shape, cache.dtype, None);
                    let rope_new = zeros(&rope_shape, cache.dtype, None);
                    let zero_start = [0i32, 0, 0, 0];
                    let latent_old_stop = [1i32, 1, cache.capacity as i32, cache.latent_dim];
                    let rope_old_stop = [1i32, 1, cache.capacity as i32, cache.rope_dim];
                    let ones = [1i32, 1, 1, 1];
                    cache.kv_latent = slice_update(
                        &latent_new,
                        &cache.kv_latent,
                        &zero_start,
                        &latent_old_stop,
                        &ones,
                        None,
                    );
                    cache.k_pe = slice_update(
                        &rope_new,
                        &cache.k_pe,
                        &zero_start,
                        &rope_old_stop,
                        &ones,
                        None,
                    );
                    cache.capacity = new_capacity;
                    self.growth_count = self.growth_count.saturating_add(1);
                }
                let start = [0i32, 0, write_start as i32, 0];
                let latent_stop = [1i32, 1, write_end as i32, cache.latent_dim];
                let rope_stop = [1i32, 1, write_end as i32, cache.rope_dim];
                let strides = [1i32, 1, 1, 1];
                cache.kv_latent = slice_update(
                    &cache.kv_latent,
                    &new_kv_latent,
                    &start,
                    &latent_stop,
                    &strides,
                    None,
                );
                cache.k_pe =
                    slice_update(&cache.k_pe, &new_k_pe, &start, &rope_stop, &strides, None);
            }
        }

        let cache = self.glm_mla_layers[layer].as_ref().unwrap();
        let end = write_end as i32;
        let kv_latent = slice(
            &cache.kv_latent,
            &[0, 0, 0, 0],
            &[1, 1, end, cache.latent_dim],
            &[1, 1, 1, 1],
            None,
        );
        let k_pe = slice(
            &cache.k_pe,
            &[0, 0, 0, 0],
            &[1, 1, end, cache.rope_dim],
            &[1, 1, 1, 1],
            None,
        );
        (kv_latent, k_pe)
    }

    /// Trim the logical boundary to `prefix_len` tokens (draft rollback).
    ///
    /// With chunked layout this is O(1) — no array data is modified.  The backing
    /// buffer retains its pre-allocated capacity.  The next `append` writes from
    /// `prefix_len`, overwriting any rejected draft positions.
    ///
    /// Returns `true` when the requested trim point was valid.  Invalid requests
    /// are clamped to the current logical length so a release build cannot extend
    /// the cache and make SDPA attend to unwritten positions.
    #[must_use]
    pub fn trim_to(&mut self, prefix_len: usize) -> bool {
        if prefix_len < self.seq_len {
            // Rotated layers can absorb a rollback only within their slack:
            // token `t` lives at slot `t % capacity`, so a token is still
            // resident iff nothing more than `capacity` positions newer has
            // overwritten it. After trimming to `L` with pre-trim end `E`,
            // the next forward reads keys back to `L - window + 1`; those
            // are intact iff `E - L <= capacity - window`. Pure rings
            // (`capacity == window`) therefore refuse every real trim, and
            // bounded rings refuse trims deeper than their slack.
            let rollback = self.seq_len - prefix_len;
            if self.layers.iter().flatten().any(|fa| {
                fa.as_contiguous().is_some_and(|lkv| {
                    lkv.rotating_window
                        .is_some_and(|window| rollback > lkv.capacity.saturating_sub(window))
                })
            }) {
                return false;
            }
        }
        let valid = prefix_len <= self.seq_len;
        let trimmed = prefix_len < self.seq_len;
        self.seq_len = prefix_len.min(self.seq_len);
        if trimmed {
            // The retained fast-path views still span the pre-trim write end,
            // including the rejected draft positions.  Drop them so any
            // consumer between this trim and the next append re-slices from
            // the logical boundary instead of attending over trimmed tokens.
            // Paged layers also free fully-empty trailing blocks back to the
            // private pool (fail-closed capacity bookkeeping).
            for fa in self.layers.iter_mut().flatten() {
                fa.clear_views();
            }
            if let Some(pool) = self.fa_pool.as_mut() {
                for fa in self.layers.iter_mut().flatten() {
                    if let FaLayerStorage::Paged(paged) = fa {
                        paged.free_blocks_beyond(pool, self.seq_len);
                    }
                }
            }
        }
        valid
    }

    /// Logical K/V view for `layer`, sliced to the current `seq_len`, or `None`
    /// if the layer has no entry yet.
    ///
    /// Used to seed a **pure** compiled MTP draft closure: the existing context
    /// is passed in as explicit closure inputs rather than read by capturing the
    /// mutable cache, which would make the compiled graph impure (the captured
    /// lazy KV would enter the trace as an un-passed constant and abort eval
    /// with "Attempting to eval an array without a primitive").
    pub fn logical_layer_kv(&self, layer: usize) -> Option<(MlxArray, MlxArray)> {
        let fa = self.layers.get(layer)?.as_ref()?;
        match fa {
            FaLayerStorage::Contiguous(lkv) => {
                debug_assert!(
                    lkv.rotating_window.is_none(),
                    "logical_layer_kv reads a [0, seq_len) prefix slice, which is meaningless \
                     on a rotated ring (MTP draft seeding never coexists with rotation)"
                );
                let end = self.seq_len as i32;
                let stop = [1, lkv.n_kv_heads, end, lkv.head_dim];
                let k = slice(&lkv.k, &[0, 0, 0, 0], &stop, &[1, 1, 1, 1], None);
                let v = slice(&lkv.v, &[0, 0, 0, 0], &stop, &[1, 1, 1, 1], None);
                Some((k, v))
            }
            FaLayerStorage::Paged(paged) => Some(paged.materialize(0, self.seq_len)),
        }
    }

    /// Commit a pure compiled MTP draft closure's threaded K/V back into the
    /// cache as a tight logical buffer (`capacity == length`) and set `seq_len`.
    ///
    /// `k`/`v` are the closure's final concatenated `[1, n_kv_heads, length,
    /// head_dim]` outputs (already evaluated).  Storing them tight is correct:
    /// the next imperative `append` sees `write_end > capacity` and grows via
    /// its normal chunk path, copying this buffer forward.  Replacing the layer
    /// entry also drops any stale views.
    pub fn set_layer_kv_logical(&mut self, layer: usize, k: MlxArray, v: MlxArray, seq_len: usize) {
        let shape = k.shape();
        debug_assert_eq!(shape.len(), 4, "set_layer_kv_logical expects a 4D K array");
        let n_kv_heads = shape[1];
        let length = shape[2] as usize;
        let head_dim = shape[3];
        let dtype = k.dtype();
        self.layers[layer] = Some(FaLayerStorage::Contiguous(LayerKV {
            last_k_view: Some(k.clone()),
            last_v_view: Some(v.clone()),
            k,
            v,
            n_kv_heads,
            head_dim,
            capacity: length,
            rotating_window: None,
            dtype,
        }));
        self.seq_len = seq_len;
    }

    /// Collect refs to all K and V backing buffers for bulk `eval`.
    ///
    /// Pass these alongside the output token to `mlx_sys::eval()` after each
    /// decode step.  Without this, every `slice_update` append leaves the
    /// backing buffer as a lazy graph node pointing at the previous step's
    /// buffer; after N steps each buffer is a chain of N `slice_update` nodes.
    /// Evaluating here materialises the chain into a flat buffer so the next
    /// step's `slice_update` has depth-1 ancestry instead of depth-N.
    ///
    /// Mirrors mlx_lm's `mx.eval(y, cache)` pattern.
    pub fn collect_eval_refs(&self) -> Vec<&MlxArray> {
        let mut refs = Vec::with_capacity(self.layers.len() * 4 + self.glm_mla_layers.len() * 2);
        for fa in self.layers.iter().flatten() {
            match fa {
                FaLayerStorage::Contiguous(lkv) => {
                    refs.push(&lkv.k);
                    refs.push(&lkv.v);
                }
                FaLayerStorage::Paged(paged) => {
                    for k in &paged.k_blocks {
                        refs.push(k);
                    }
                    for v in &paged.v_blocks {
                        refs.push(v);
                    }
                }
            }
        }
        for glm_mla in self.glm_mla_layers.iter().flatten() {
            refs.push(&glm_mla.kv_latent);
            refs.push(&glm_mla.k_pe);
        }
        for linear in &self.linear_layers {
            if let Some(conv_state) = &linear.conv_state {
                refs.push(conv_state);
            }
            if let Some(recurrent_state) = &linear.recurrent_state {
                refs.push(recurrent_state);
            }
        }
        refs
    }

    /// Read-only access to a single GLM-MLA layer's cached `kv_latent` and
    /// `k_pe` arrays plus their inner dims. Used by debug bins (notably the
    /// F4 warm-extend drift probe) to compare per-layer KV state between
    /// cold and warm prefill paths.
    ///
    /// Returns `None` when the layer index is out of range or when this
    /// model has no GLM-MLA layer at that index. The arrays are over-
    /// allocated to capacity; callers that want only the valid region must
    /// slice to `[0..self.seq_len]` themselves using `self.seq_len`.
    pub fn glm_mla_layer_state(&self, layer: usize) -> Option<GlmMlaLayerStateView<'_>> {
        let entry = self.glm_mla_layers.get(layer)?.as_ref()?;
        Some(GlmMlaLayerStateView {
            kv_latent: &entry.kv_latent,
            k_pe: &entry.k_pe,
            latent_dim: entry.latent_dim,
            rope_dim: entry.rope_dim,
        })
    }

    pub fn usage_snapshot(&self) -> MlxKVCacheUsage {
        self.usage_snapshot_with_layer_windows(&[])
    }

    pub fn usage_snapshot_with_layer_windows(
        &self,
        layer_windows: &[Option<usize>],
    ) -> MlxKVCacheUsage {
        let mut usage = MlxKVCacheUsage {
            logical_tokens: self.seq_len,
            growth_count: self.growth_count,
            rotating_ring_slack: self.rotating_slack,
            paged_materialize_us: self.paged_materialize_us,
            paged_pool_exhaustion_fallbacks: self.paged_pool_exhaustion_fallbacks,
            ..MlxKVCacheUsage::default()
        };

        for (layer_idx, fa) in self.layers.iter().enumerate() {
            let Some(fa) = fa else {
                continue;
            };
            if fa.rotating_window().is_some() {
                usage.rotated_ring_layers = usage.rotated_ring_layers.saturating_add(1);
            }
            let elements_per_token = (fa.n_kv_heads() as u64).saturating_mul(fa.head_dim() as u64);
            let bytes_per_element = fa.dtype().size_bytes() as u64;
            let bytes_per_token = elements_per_token
                .saturating_mul(bytes_per_element)
                .saturating_mul(2);
            let capacity = fa.capacity();

            usage.full_attention_layers = usage.full_attention_layers.saturating_add(1);
            usage.capacity_tokens = usage.capacity_tokens.saturating_add(capacity);
            usage.logical_bytes = usage
                .logical_bytes
                .saturating_add(bytes_per_token.saturating_mul(self.seq_len as u64));
            usage.capacity_bytes = usage
                .capacity_bytes
                .saturating_add(bytes_per_token.saturating_mul(capacity as u64));

            if let Some(window) = layer_windows.get(layer_idx).copied().flatten() {
                let retained_tokens = self.seq_len.min(window);
                let retained_capacity = chunk_ceiling(retained_tokens).min(capacity);
                let reclaimable_tokens = capacity.saturating_sub(retained_capacity);
                usage.sliding_window_layers = usage.sliding_window_layers.saturating_add(1);
                usage.sliding_window_retained_tokens = usage
                    .sliding_window_retained_tokens
                    .saturating_add(retained_tokens);
                usage.sliding_window_reclaimable_capacity_tokens = usage
                    .sliding_window_reclaimable_capacity_tokens
                    .saturating_add(reclaimable_tokens);
                usage.sliding_window_reclaimable_capacity_bytes = usage
                    .sliding_window_reclaimable_capacity_bytes
                    .saturating_add(bytes_per_token.saturating_mul(reclaimable_tokens as u64));
            }
        }

        for linear in &self.linear_layers {
            let layer_bytes = linear
                .conv_state
                .as_ref()
                .map(|array| array.nbytes() as u64)
                .unwrap_or(0)
                .saturating_add(
                    linear
                        .recurrent_state
                        .as_ref()
                        .map(|array| array.nbytes() as u64)
                        .unwrap_or(0),
                );
            if layer_bytes > 0 {
                usage.linear_state_layers = usage.linear_state_layers.saturating_add(1);
                usage.linear_state_bytes = usage.linear_state_bytes.saturating_add(layer_bytes);
            }
        }

        for glm_mla in self.glm_mla_layers.iter().flatten() {
            let elements_per_token =
                (glm_mla.latent_dim as u64).saturating_add(glm_mla.rope_dim as u64);
            let bytes_per_token =
                elements_per_token.saturating_mul(glm_mla.dtype.size_bytes() as u64);
            usage.full_attention_layers = usage.full_attention_layers.saturating_add(1);
            usage.capacity_tokens = usage.capacity_tokens.saturating_add(glm_mla.capacity);
            usage.logical_bytes = usage
                .logical_bytes
                .saturating_add(bytes_per_token.saturating_mul(self.seq_len as u64));
            usage.capacity_bytes = usage
                .capacity_bytes
                .saturating_add(bytes_per_token.saturating_mul(glm_mla.capacity as u64));
        }

        usage
    }

    /// Read the cached gated-delta states for a Qwen3.5 linear-attention layer.
    pub fn linear_state(&self, layer: usize) -> (Option<&MlxArray>, Option<&MlxArray>) {
        let state = &self.linear_layers[layer];
        (state.conv_state.as_ref(), state.recurrent_state.as_ref())
    }

    /// Store the gated-delta states for a Qwen3.5 linear-attention layer.
    pub fn set_linear_state(
        &mut self,
        layer: usize,
        conv_state: MlxArray,
        recurrent_state: MlxArray,
    ) {
        let state = &mut self.linear_layers[layer];
        state.conv_state = Some(conv_state);
        state.recurrent_state = Some(recurrent_state);
    }

    /// Read K/V already written by `source_layer` during the current forward pass.
    ///
    /// Used by KV-shared layers (e.g. Gemma4 layers 24-41) that attend against
    /// a prior layer's cache instead of computing their own K/V projections.
    ///
    /// Returns the views cached by the last `append` call, which are identical to
    /// what a fresh `slice(lkv.k, 0..seq_len+new_tokens)` would produce.  Reusing
    /// the same MLX graph node avoids a duplicate GPU kernel dispatch per KV-shared
    /// layer: for E2B (20 shared layers) this eliminates 40 extra slice kernels
    /// (~12 µs each), saving ~0.5 ms per decode step.
    ///
    /// `new_tokens` is retained for the panic check that validates the source layer
    /// was updated in the current forward pass.
    pub fn peek_source_kv(&self, source_layer: usize, new_tokens: usize) -> (MlxArray, MlxArray) {
        let fa = self.layers[source_layer]
            .as_ref()
            .expect("KV-shared source layer has no cached KV — source layer must appear earlier");
        match fa {
            FaLayerStorage::Contiguous(lkv) => {
                if lkv.rotating_window.is_some() {
                    // Rotated ring: the backing store IS the full ring view (the
                    // storing layer's append returned exactly this), and the ordered
                    // `[0, seq_len)` fallback below would slice past the ring's
                    // capacity. Consumers mask via the hoisted ring validity mask.
                    return (lkv.k.clone(), lkv.v.clone());
                }
                match (&lkv.last_k_view, &lkv.last_v_view) {
                    (Some(k), Some(v)) => (k.clone(), v.clone()),
                    _ => {
                        // Fallback: create fresh views (e.g., first append in a grow-then-slice sequence).
                        let end = (self.seq_len + new_tokens) as i32;
                        let k = slice(
                            &lkv.k,
                            &[0, 0, 0, 0],
                            &[1, lkv.n_kv_heads, end, lkv.head_dim],
                            &[1, 1, 1, 1],
                            None,
                        );
                        let v = slice(
                            &lkv.v,
                            &[0, 0, 0, 0],
                            &[1, lkv.n_kv_heads, end, lkv.head_dim],
                            &[1, 1, 1, 1],
                            None,
                        );
                        (k, v)
                    }
                }
            }
            FaLayerStorage::Paged(paged) => match (&paged.last_k_view, &paged.last_v_view) {
                (Some(k), Some(v)) => (k.clone(), v.clone()),
                _ => paged.materialize(0, self.seq_len + new_tokens),
            },
        }
    }

    /// Read K/V already stored for `layer` without mutating cache state.
    ///
    /// Gemma4 Assistant uses this to attend against target full/sliding K/V states
    /// from a separate assistant forward pass. Unlike `peek_source_kv`, this does
    /// not assert that the layer was written during the current forward call.
    pub fn peek_layer_kv(&self, layer: usize) -> Option<(MlxArray, MlxArray)> {
        let fa = self.layers.get(layer)?.as_ref()?;
        match fa {
            FaLayerStorage::Contiguous(lkv) => {
                if lkv.rotating_window.is_some() {
                    // Rotated ring: return the full ring — valid at any time,
                    // including right after a `trim_to` rollback cleared the cached
                    // views (when the ordered `[0, seq_len)` fallback below would
                    // slice past the ring's capacity). Consumers must apply the
                    // slot-validity mask derived from [`Self::layer_sliding_ring`].
                    return Some((lkv.k.clone(), lkv.v.clone()));
                }
                let (k_view, v_view) = match (&lkv.last_k_view, &lkv.last_v_view) {
                    (Some(k), Some(v)) => (k.clone(), v.clone()),
                    _ => {
                        let end = self.seq_len as i32;
                        let k = slice(
                            &lkv.k,
                            &[0, 0, 0, 0],
                            &[1, lkv.n_kv_heads, end, lkv.head_dim],
                            &[1, 1, 1, 1],
                            None,
                        );
                        let v = slice(
                            &lkv.v,
                            &[0, 0, 0, 0],
                            &[1, lkv.n_kv_heads, end, lkv.head_dim],
                            &[1, 1, 1, 1],
                            None,
                        );
                        (k, v)
                    }
                };
                Some((k_view, v_view))
            }
            FaLayerStorage::Paged(paged) => match (&paged.last_k_view, &paged.last_v_view) {
                (Some(k), Some(v)) => Some((k.clone(), v.clone())),
                _ => Some(paged.materialize(0, self.seq_len)),
            },
        }
    }

    /// The current ring geometry of `layer` if it has converted to a
    /// rotating ring, for consumers that read ring K/V outside a forward
    /// (e.g. the Gemma4 assistant drafter attending target KV between
    /// appends). `write_start` is set to the cache's current `seq_len`:
    /// a reader whose query logically sits at the *end* of the context
    /// builds its mask as `create_ring_sliding_mask(1, window, capacity,
    /// seq_len - 1)`, which keeps exactly the last `window` live tokens
    /// and automatically excludes slots holding rolled-back drafts (their
    /// resident-token index decodes below `seq_len - window` under the
    /// post-trim end).
    pub fn layer_sliding_ring(&self, layer: usize) -> Option<SlidingRingLayout> {
        let lkv = self.layers.get(layer)?.as_ref()?.as_contiguous()?;
        let window = lkv.rotating_window?;
        Some(SlidingRingLayout {
            window,
            capacity: lkv.capacity,
            write_start: self.seq_len,
        })
    }

    /// Read a fresh full-prefix K/V view for `layer`.
    ///
    /// Unlike `peek_layer_kv`, this intentionally ignores cached retained views
    /// from the most recent append. Diffusion denoising attends against the
    /// committed prompt prefix, so its bidirectional mask must match exactly
    /// `self.seq_len` cached keys.
    pub fn peek_layer_full_kv(&self, layer: usize) -> Option<(MlxArray, MlxArray)> {
        let fa = self.layers.get(layer)?.as_ref()?;
        match fa {
            FaLayerStorage::Contiguous(lkv) => {
                let end = self.seq_len as i32;
                let k = slice(
                    &lkv.k,
                    &[0, 0, 0, 0],
                    &[1, lkv.n_kv_heads, end, lkv.head_dim],
                    &[1, 1, 1, 1],
                    None,
                );
                let v = slice(
                    &lkv.v,
                    &[0, 0, 0, 0],
                    &[1, lkv.n_kv_heads, end, lkv.head_dim],
                    &[1, 1, 1, 1],
                    None,
                );
                Some((k, v))
            }
            FaLayerStorage::Paged(paged) => Some(paged.materialize(0, self.seq_len)),
        }
    }

    /// Replace the backing K/V arrays for a layer.
    ///
    /// Used by the whole-layer compiled decode path to update the cache
    /// with new arrays returned from the compiled closure. Unlike `append`,
    /// this replaces the full backing buffer rather than writing new tokens
    /// to the existing one. The `seq_len` is not incremented; callers must
    /// manage `seq_len` separately.
    ///
    /// On the paged FA path this forces the layer back to contiguous storage
    /// (compiled decode returns dense K/V).
    pub fn replace_layer_kv(&mut self, layer: usize, new_k: MlxArray, new_v: MlxArray) {
        let shape = new_k.shape();
        if shape.len() != 4 {
            return;
        }
        let n_kv_heads = shape[1];
        let capacity = shape[2] as usize;
        let head_dim = shape[3];
        let dtype = new_k.dtype();
        // If this layer was paged, free its blocks before replacing.
        let previous = self.layers.get_mut(layer).and_then(Option::take);
        if let Some(FaLayerStorage::Paged(paged)) = previous
            && let Some(pool) = self.fa_pool.as_mut()
        {
            let _ = pool.free(&paged.block_ids);
        }
        if let Some(slot) = self.layers.get_mut(layer) {
            *slot = Some(FaLayerStorage::Contiguous(LayerKV {
                k: new_k,
                v: new_v,
                last_k_view: None,
                last_v_view: None,
                n_kv_heads,
                head_dim,
                capacity,
                rotating_window: None,
                dtype,
            }));
        }
    }

    /// Reset cache entirely (e.g., between requests).
    pub fn reset(&mut self) {
        if let Some(pool) = self.fa_pool.as_mut() {
            for entry in self.layers.iter_mut().flatten() {
                if let FaLayerStorage::Paged(paged) = entry {
                    let _ = pool.free(&paged.block_ids);
                    paged.block_ids.clear();
                    paged.k_blocks.clear();
                    paged.v_blocks.clear();
                }
            }
        }
        for entry in &mut self.layers {
            *entry = None;
        }
        for entry in &mut self.glm_mla_layers {
            *entry = None;
        }
        for state in &mut self.linear_layers {
            *state = LinearLayerState::default();
        }
        self.seq_len = 0;
        self.rope_offset = 0;
        self.growth_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_sys::astype;

    fn contiguous_layer(cache: &MlxKVCache, layer: usize) -> &LayerKV {
        cache.layers[layer]
            .as_ref()
            .and_then(FaLayerStorage::as_contiguous)
            .expect("contiguous FA layer")
    }

    #[test]
    fn linear_state_is_eval_tracked_and_reset() {
        let mut cache = MlxKVCache::new(2);
        let conv = zeros(&[1, 3, 14], MlxDtype::Float32, None);
        let recurrent = zeros(&[1, 4, 8, 6], MlxDtype::Float32, None);

        cache.set_linear_state(1, conv, recurrent);

        let (conv_state, recurrent_state) = cache.linear_state(1);
        assert_eq!(conv_state.expect("conv state").shape(), vec![1, 3, 14]);
        assert_eq!(
            recurrent_state.expect("recurrent state").shape(),
            vec![1, 4, 8, 6]
        );
        assert_eq!(cache.collect_eval_refs().len(), 2);

        cache.reset();

        let (conv_state, recurrent_state) = cache.linear_state(1);
        assert!(conv_state.is_none());
        assert!(recurrent_state.is_none());
        assert!(cache.collect_eval_refs().is_empty());
    }

    #[test]
    fn linear_state_survives_trim_to() {
        let mut cache = MlxKVCache::new(1);
        let conv = zeros(&[1, 3, 14], MlxDtype::Float32, None);
        let recurrent = zeros(&[1, 4, 8, 6], MlxDtype::Float32, None);

        cache.seq_len = 8;
        cache.set_linear_state(0, conv, recurrent);
        assert!(cache.trim_to(4));

        assert_eq!(cache.seq_len, 4);
        let (conv_state, recurrent_state) = cache.linear_state(0);
        assert!(
            conv_state.is_some() && recurrent_state.is_some(),
            "linear recurrent state is not rolled back by seq_len trim"
        );
    }

    #[test]
    fn trim_to_does_not_extend_logical_sequence() {
        let mut cache = MlxKVCache::new(1);
        cache.seq_len = 8;

        assert!(!cache.trim_to(12));

        assert_eq!(
            cache.seq_len, 8,
            "invalid rollback points must not expose unwritten KV slots"
        );
    }

    #[test]
    fn clone_preserves_linear_state_for_draft_branch() {
        let mut cache = MlxKVCache::new(1);
        let conv = zeros(&[1, 3, 14], MlxDtype::Float32, None);
        let recurrent = zeros(&[1, 4, 8, 6], MlxDtype::Float32, None);

        cache.seq_len = 12;
        cache.set_linear_state(0, conv, recurrent);
        let branch = cache.clone();
        cache.reset();

        assert_eq!(branch.seq_len, 12);
        let (conv_state, recurrent_state) = branch.linear_state(0);
        assert!(conv_state.is_some());
        assert!(recurrent_state.is_some());
    }

    #[test]
    fn glm_mla_cache_appends_latent_and_rope_key_history() {
        let mut cache = MlxKVCache::new(1);
        let kv_latent = zeros(&[1, 1, 2, 512], MlxDtype::Bfloat16, None);
        let k_pe = zeros(&[1, 1, 2, 64], MlxDtype::Bfloat16, None);

        let (latent_history, rope_history) = cache.append_glm_mla(0, kv_latent, k_pe);

        assert_eq!(latent_history.shape(), vec![1, 1, 2, 512]);
        assert_eq!(rope_history.shape(), vec![1, 1, 2, 64]);
        assert_eq!(cache.collect_eval_refs().len(), 2);
        cache.seq_len = 2;

        let kv_latent = zeros(&[1, 1, 1, 512], MlxDtype::Bfloat16, None);
        let k_pe = zeros(&[1, 1, 1, 64], MlxDtype::Bfloat16, None);
        let (latent_history, rope_history) = cache.append_glm_mla(0, kv_latent, k_pe);

        assert_eq!(latent_history.shape(), vec![1, 1, 3, 512]);
        assert_eq!(rope_history.shape(), vec![1, 1, 3, 64]);
        assert_eq!(cache.collect_eval_refs().len(), 2);
    }

    #[test]
    fn usage_snapshot_tracks_glm_mla_compressed_cache_bytes() {
        let mut cache = MlxKVCache::new(1);
        let kv_latent = zeros(&[1, 1, 2, 512], MlxDtype::Bfloat16, None);
        let k_pe = zeros(&[1, 1, 2, 64], MlxDtype::Bfloat16, None);

        cache.append_glm_mla(0, kv_latent, k_pe);
        cache.seq_len = 2;

        let usage = cache.usage_snapshot();
        assert_eq!(usage.logical_tokens, 2);
        assert_eq!(usage.capacity_tokens, 256);
        assert_eq!(usage.full_attention_layers, 1);
        assert_eq!(usage.logical_bytes, 2304);
        assert_eq!(usage.capacity_bytes, 294_912);
        assert_eq!(usage.growth_count, 1);
    }

    #[test]
    fn reset_clears_glm_mla_cache_eval_refs() {
        let mut cache = MlxKVCache::new(1);
        let kv_latent = zeros(&[1, 1, 1, 512], MlxDtype::Bfloat16, None);
        let k_pe = zeros(&[1, 1, 1, 64], MlxDtype::Bfloat16, None);

        cache.append_glm_mla(0, kv_latent, k_pe);
        cache.seq_len = 1;
        assert_eq!(cache.collect_eval_refs().len(), 2);

        cache.reset();

        assert_eq!(cache.seq_len, 0);
        assert!(cache.collect_eval_refs().is_empty());
        assert_eq!(cache.usage_snapshot(), MlxKVCacheUsage::default());
    }

    #[test]
    fn usage_snapshot_tracks_full_attention_capacity_and_growth() {
        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 2, 3, 4], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 3, 4], MlxDtype::Bfloat16, None);

        cache.append(0, k, v);
        cache.seq_len = 3;

        let usage = cache.usage_snapshot();
        assert_eq!(usage.logical_tokens, 3);
        assert_eq!(usage.capacity_tokens, 256);
        assert_eq!(usage.full_attention_layers, 1);
        assert_eq!(usage.logical_bytes, 96);
        assert_eq!(usage.capacity_bytes, 8192);
        assert_eq!(usage.growth_count, 1);
    }

    #[test]
    fn peek_layer_full_kv_ignores_retained_last_view() {
        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 2, 5, 4], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 5, 4], MlxDtype::Bfloat16, None);

        cache.append_with_retained_window(0, k, v, Some(3));
        cache.seq_len = 5;

        let (retained_k, _) = cache.peek_layer_kv(0).expect("retained view");
        let (full_k, _) = cache.peek_layer_full_kv(0).expect("full view");
        assert_eq!(retained_k.shape(), vec![1, 2, 3, 4]);
        assert_eq!(full_k.shape(), vec![1, 2, 5, 4]);
    }

    #[test]
    fn multi_token_append_retains_window_plus_seq_view() {
        // Prefill 8 tokens (full view), then append a 4-token chunk with the
        // multi-token retained bound window + seq - 1 = 3 + 4 - 1 = 6: the
        // returned view must be the last 6 tokens of the 12-token history,
        // contents intact, while full storage stays available for rollback
        // and prefix-cache snapshots.
        let head_dim = 2usize;
        let fill = |start: usize, tokens: usize| -> MlxArray {
            let data: Vec<f32> = (start..start + tokens * head_dim)
                .map(|i| i as f32)
                .collect();
            let flat = MlxArray::from_f32_slice(&data);
            mlx_sys::reshape(&flat, &[1, 1, tokens as i32, head_dim as i32], None)
        };
        let read_f32 = |arr: &MlxArray| -> Vec<f32> {
            let arr = astype(arr, MlxDtype::Float32, None);
            eval(&[&arr]);
            let len = arr.nbytes() / std::mem::size_of::<f32>();
            let ptr = arr.data_raw() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
        };

        let mut cache = MlxKVCache::new(1);
        cache.append(0, fill(0, 8), fill(100, 8));
        cache.seq_len = 8;

        let (k_view, v_view) =
            cache.append_with_retained_window(0, fill(16, 4), fill(116, 4), Some(6));
        cache.seq_len = 12;

        assert_eq!(k_view.shape(), vec![1, 1, 6, head_dim as i32]);
        assert_eq!(v_view.shape(), vec![1, 1, 6, head_dim as i32]);
        // Last 6 tokens = prompt tokens 6..8 (values 12..16) + the 4 new
        // tokens (values 16..24).
        let expected_k: Vec<f32> = (12..24).map(|i| i as f32).collect();
        let expected_v: Vec<f32> = (112..124).map(|i| i as f32).collect();
        assert_eq!(read_f32(&k_view), expected_k);
        assert_eq!(read_f32(&v_view), expected_v);

        let (full_k, full_v) = cache.peek_layer_full_kv(0).expect("full view");
        assert_eq!(full_k.shape(), vec![1, 1, 12, head_dim as i32]);
        let full_k_data = read_f32(&full_k);
        assert_eq!(
            &full_k_data[..16],
            (0..16).map(|i| i as f32).collect::<Vec<_>>()
        );
        assert_eq!(
            &full_k_data[16..],
            (16..24).map(|i| i as f32).collect::<Vec<_>>()
        );
        assert_eq!(read_f32(&full_v).len(), 24);
    }

    #[test]
    fn usage_snapshot_tracks_sliding_window_trim_opportunity() {
        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 2, 300, 4], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 300, 4], MlxDtype::Bfloat16, None);

        cache.append(0, k, v);
        cache.seq_len = 300;

        let usage = cache.usage_snapshot_with_layer_windows(&[Some(128)]);
        assert_eq!(usage.capacity_tokens, 512);
        assert_eq!(usage.sliding_window_layers, 1);
        assert_eq!(usage.sliding_window_retained_tokens, 128);
        assert_eq!(usage.sliding_window_reclaimable_capacity_tokens, 256);
        assert_eq!(usage.sliding_window_reclaimable_capacity_bytes, 8192);
    }

    #[test]
    fn usage_snapshot_ignores_unwritten_sliding_window_layers() {
        let mut cache = MlxKVCache::new(2);
        let k = zeros(&[1, 2, 300, 4], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 300, 4], MlxDtype::Bfloat16, None);

        cache.append(0, k, v);
        cache.seq_len = 300;

        let usage = cache.usage_snapshot_with_layer_windows(&[Some(128), Some(128)]);
        assert_eq!(usage.full_attention_layers, 1);
        assert_eq!(usage.sliding_window_layers, 1);
        assert_eq!(usage.sliding_window_reclaimable_capacity_tokens, 256);
    }

    #[test]
    fn usage_snapshot_does_not_report_reclaimable_capacity_inside_window() {
        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 2, 120, 4], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 120, 4], MlxDtype::Bfloat16, None);

        cache.append(0, k, v);
        cache.seq_len = 120;

        let usage = cache.usage_snapshot_with_layer_windows(&[Some(512)]);
        assert_eq!(usage.capacity_tokens, 256);
        assert_eq!(usage.sliding_window_layers, 1);
        assert_eq!(usage.sliding_window_retained_tokens, 120);
        assert_eq!(usage.sliding_window_reclaimable_capacity_tokens, 0);
        assert_eq!(usage.sliding_window_reclaimable_capacity_bytes, 0);
    }

    #[test]
    #[should_panic(expected = "matching K/V shapes")]
    fn append_rejects_mismatched_kv_shapes() {
        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 2, 3, 4], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 4, 4], MlxDtype::Bfloat16, None);

        let _ = cache.append(0, k, v);
    }

    #[test]
    #[should_panic(expected = "cannot change head_dim")]
    fn append_rejects_existing_layer_shape_drift() {
        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 2, 3, 4], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 3, 4], MlxDtype::Bfloat16, None);
        let _ = cache.append(0, k, v);
        cache.seq_len = 3;

        let k = zeros(&[1, 2, 1, 5], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 1, 5], MlxDtype::Bfloat16, None);
        let _ = cache.append(0, k, v);
    }

    #[test]
    #[should_panic(expected = "requires matching K/V dtypes")]
    fn append_rejects_mismatched_kv_dtypes() {
        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 2, 3, 4], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 3, 4], MlxDtype::Float32, None);

        let _ = cache.append(0, k, v);
    }

    #[test]
    fn usage_snapshot_tracks_linear_state_bytes() {
        let mut cache = MlxKVCache::new(1);
        let conv = zeros(&[1, 3, 14], MlxDtype::Float32, None);
        let recurrent = zeros(&[1, 4, 8, 6], MlxDtype::Float32, None);

        cache.set_linear_state(0, conv, recurrent);

        let usage = cache.usage_snapshot();
        assert_eq!(usage.linear_state_layers, 1);
        assert_eq!(usage.linear_state_bytes, 936);
    }

    #[test]
    fn peek_source_kv_reuses_cached_views_from_append() {
        use mlx_sys::eval;
        // Two-layer cache: layer 0 is the source, layer 1 is a KV-shared consumer.
        let mut cache = MlxKVCache::new(2);

        let k = zeros(&[1, 1, 4, 8], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 1, 4, 8], MlxDtype::Bfloat16, None);
        let (k_from_append, _) = cache.append(0, k, v);
        cache.seq_len = 4;

        let (k_from_peek, _) = cache.peek_source_kv(0, 0);

        // Materialise both arrays. If peek returned the same lazy node as append,
        // the results must be numerically identical (same shape and dtype).
        eval(&[&k_from_append, &k_from_peek]);
        assert_eq!(k_from_append.shape(), k_from_peek.shape());
        assert_eq!(k_from_append.dtype(), k_from_peek.dtype());

        // After a buffer grow, last_k_view is cleared and peek falls back to a
        // fresh slice — verify the fallback also produces the correct shape.
        let k2 = zeros(&[1, 1, 300, 8], MlxDtype::Bfloat16, None);
        let v2 = zeros(&[1, 1, 300, 8], MlxDtype::Bfloat16, None);
        cache.append(0, k2, v2);
        cache.seq_len = 304;

        let (k_grow, _) = cache.peek_source_kv(0, 0);
        eval(&[&k_grow]);
        assert_eq!(k_grow.shape(), vec![1, 1, 304, 8]);
    }

    #[test]
    fn append_with_retained_window_returns_windowed_cached_views() {
        use mlx_sys::eval;

        let mut cache = MlxKVCache::new(2);
        let k = zeros(&[1, 1, 6, 8], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 1, 6, 8], MlxDtype::Bfloat16, None);

        let (k_from_append, v_from_append) = cache.append_with_retained_window(0, k, v, Some(4));
        cache.seq_len = 6;
        let (k_from_peek, v_from_peek) = cache.peek_source_kv(0, 0);

        eval(&[&k_from_append, &v_from_append, &k_from_peek, &v_from_peek]);
        assert_eq!(k_from_append.shape(), vec![1, 1, 4, 8]);
        assert_eq!(v_from_append.shape(), vec![1, 1, 4, 8]);
        assert_eq!(k_from_peek.shape(), vec![1, 1, 4, 8]);
        assert_eq!(v_from_peek.shape(), vec![1, 1, 4, 8]);
    }

    #[test]
    fn rotating_sliding_decode_uses_bounded_backing_store() {
        let mut cache = MlxKVCache::new(1);
        cache.set_rotating_sliding_decode(true);

        let k = zeros(&[1, 1, 6, 8], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 1, 6, 8], MlxDtype::Bfloat16, None);
        let (prefill_k, _) = cache.append(0, k, v);
        cache.seq_len = 6;
        assert_eq!(prefill_k.shape(), vec![1, 1, 6, 8]);

        let next_k = zeros(&[1, 1, 1, 8], MlxDtype::Bfloat16, None);
        let next_v = zeros(&[1, 1, 1, 8], MlxDtype::Bfloat16, None);
        let (decode_k, decode_v) = cache.append_with_retained_window(0, next_k, next_v, Some(4));

        let lkv = contiguous_layer(&cache, 0);
        assert_eq!(lkv.capacity, 4);
        assert_eq!(lkv.rotating_window, Some(4));
        assert_eq!(decode_k.shape(), vec![1, 1, 4, 8]);
        assert_eq!(decode_v.shape(), vec![1, 1, 4, 8]);
    }

    #[test]
    fn trim_to_rejects_rollback_after_rotating_sliding_decode() {
        let mut cache = MlxKVCache::new(1);
        cache.set_rotating_sliding_decode(true);
        let k = zeros(&[1, 1, 4, 8], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 1, 4, 8], MlxDtype::Bfloat16, None);
        cache.append(0, k, v);
        cache.seq_len = 4;

        let next_k = zeros(&[1, 1, 1, 8], MlxDtype::Bfloat16, None);
        let next_v = zeros(&[1, 1, 1, 8], MlxDtype::Bfloat16, None);
        cache.append_with_retained_window(0, next_k, next_v, Some(4));
        cache.seq_len = 5;

        assert!(!cache.trim_to(4));
        assert_eq!(cache.seq_len, 5);
    }

    // ── Bounded-rollback rotating ring tests ──

    /// `[1, 1, len, head_dim]` f32 array where token row `i` is filled with
    /// `values[i]`, so slot contents are identifiable after ring writes.
    fn tokens_f32(values: &[f32], head_dim: usize) -> MlxArray {
        let data: Vec<f32> = values
            .iter()
            .flat_map(|&value| std::iter::repeat_n(value, head_dim))
            .collect();
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data.as_slice()),
            &[1, 1, values.len() as i32, head_dim as i32],
            MlxDtype::Float32,
        )
    }

    /// First element of each token row of a `[1, 1, len, head_dim]` array.
    fn token_row_values(arr: &MlxArray, head_dim: usize) -> Vec<f32> {
        eval(&[arr]);
        arr.data_f32().chunks(head_dim).map(|row| row[0]).collect()
    }

    #[test]
    fn sliding_ring_layout_gates_by_mode_seq_and_crossing() {
        let mut cache = MlxKVCache::new(1);
        cache.seq_len = 10;
        // Rotation disabled: never a ring.
        assert_eq!(cache.sliding_ring_layout(Some(4), 1), None);
        cache.set_rotating_sliding_decode(true);
        // Pure mode: single-token only.
        let pure = cache.sliding_ring_layout(Some(4), 1).expect("pure ring");
        assert_eq!((pure.window, pure.capacity, pure.write_start), (4, 4, 10));
        assert!(!pure.needs_mask(1));
        assert_eq!(cache.sliding_ring_layout(Some(4), 2), None);
        // Bounded mode: multi-token up to the slack, always masked.
        cache.set_rotating_sliding_slack(3);
        let ring = cache.sliding_ring_layout(Some(4), 3).expect("bounded ring");
        assert_eq!((ring.window, ring.capacity, ring.write_start), (4, 7, 10));
        assert!(ring.needs_mask(1));
        assert_eq!(cache.sliding_ring_layout(Some(4), 4), None);
        // Not yet crossing the window, and no window at all: ordered path.
        cache.seq_len = 2;
        assert_eq!(cache.sliding_ring_layout(Some(4), 2), None);
        cache.seq_len = 10;
        assert_eq!(cache.sliding_ring_layout(None, 1), None);
    }

    #[test]
    fn bounded_ring_multi_token_append_places_tokens_by_slot_and_wraps() {
        const HD: usize = 4;
        let mut cache = MlxKVCache::new(1);
        cache.set_rotating_sliding_decode(true);
        cache.set_rotating_sliding_slack(3); // window 4 → capacity 7
        // Prefill 4 tokens (values 1..=4 for tokens 0..=3).
        let k = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
        let v = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
        cache.append(0, k, v);
        cache.seq_len = 4;

        // 3-token verify-style append crosses the window: tokens 4, 5, 6.
        let k = tokens_f32(&[5.0, 6.0, 7.0], HD);
        let v = tokens_f32(&[5.0, 6.0, 7.0], HD);
        let (ck, _) = cache.append_with_retained_window(0, k, v, Some(4));
        cache.seq_len = 7;

        let lkv = contiguous_layer(&cache, 0);
        assert_eq!(lkv.rotating_window, Some(4));
        assert_eq!(lkv.capacity, 7);
        assert_eq!(ck.shape(), vec![1, 1, 7, HD as i32]);
        // Conversion copies tokens 1..=3 (window - 1 back from write_start 4)
        // to slots 1..=3; new tokens 4..=6 land at slots 4..=6; slot 0 (token
        // 0's slot) was outside the copy range and stays zero.
        assert_eq!(
            token_row_values(&lkv.k, HD),
            vec![0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        );

        // Two single-token appends wrap: token 7 → slot 0, token 8 → slot 1.
        for (t, value) in [(7usize, 8.0f32), (8, 9.0)] {
            let k = tokens_f32(&[value], HD);
            let v = tokens_f32(&[value], HD);
            cache.append_with_retained_window(0, k, v, Some(4));
            cache.seq_len = t + 1;
        }
        let lkv = contiguous_layer(&cache, 0);
        assert_eq!(
            token_row_values(&lkv.k, HD),
            vec![8.0, 9.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        );
    }

    #[test]
    fn bounded_ring_trim_within_slack_rewrites_same_slots() {
        const HD: usize = 4;
        let mut cache = MlxKVCache::new(1);
        cache.set_rotating_sliding_decode(true);
        cache.set_rotating_sliding_slack(3);
        let k = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
        let v = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
        cache.append(0, k, v);
        cache.seq_len = 4;
        // Verify forward: draft tokens 4, 5, 6 (values 5, 6, 7).
        let k = tokens_f32(&[5.0, 6.0, 7.0], HD);
        let v = tokens_f32(&[5.0, 6.0, 7.0], HD);
        cache.append_with_retained_window(0, k, v, Some(4));
        cache.seq_len = 7;

        // Reject the last two draft tokens (rollback depth 2 <= slack 3).
        assert!(cache.trim_to(5));
        assert_eq!(cache.seq_len, 5);

        // The corrected continuation rewrites tokens 5 and 6 with new values;
        // they land in the same slots the rejected tokens occupied.
        let k = tokens_f32(&[60.0, 70.0], HD);
        let v = tokens_f32(&[60.0, 70.0], HD);
        cache.append_with_retained_window(0, k, v, Some(4));
        cache.seq_len = 7;
        let lkv = contiguous_layer(&cache, 0);
        assert_eq!(
            token_row_values(&lkv.k, HD),
            vec![0.0, 2.0, 3.0, 4.0, 5.0, 60.0, 70.0]
        );

        // Rollback deeper than the slack is refused (fail-closed).
        assert!(!cache.trim_to(3));
        assert_eq!(cache.seq_len, 7);
        // Pure rings (slack 0 on another cache) still refuse any real trim —
        // covered by trim_to_rejects_rollback_after_rotating_sliding_decode.
    }

    /// The Gemma4 assistant drafter reads target KV between appends via
    /// `peek_layer_kv` — including right after a verify rollback, when the
    /// cached views are cleared and the ordered `[0, seq_len)` fallback
    /// would slice past a ring's capacity. Rotated layers must return the
    /// full ring plus geometry for the slot-validity mask.
    #[test]
    fn peek_layer_kv_returns_full_ring_after_rollback() {
        const HD: usize = 4;
        let mut cache = MlxKVCache::new(1);
        cache.set_rotating_sliding_decode(true);
        cache.set_rotating_sliding_slack(3); // window 4 → capacity 7
        let k = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
        let v = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
        cache.append(0, k, v);
        cache.seq_len = 4;
        let k = tokens_f32(&[5.0, 6.0, 7.0], HD);
        let v = tokens_f32(&[5.0, 6.0, 7.0], HD);
        cache.append_with_retained_window(0, k, v, Some(4));
        cache.seq_len = 7;
        // Partial reject: views are cleared, seq_len (5) < ring capacity (7).
        assert!(cache.trim_to(5));

        let (k, _v) = cache.peek_layer_kv(0).expect("layer peek");
        assert_eq!(k.shape(), vec![1, 1, 7, HD as i32]);
        let ring = cache.layer_sliding_ring(0).expect("ring geometry");
        assert_eq!((ring.window, ring.capacity, ring.write_start), (4, 7, 5));

        // The end-anchored drafter mask keeps exactly the last `window` live
        // tokens (1..=4) and excludes the rolled-back slots (tokens 5, 6 at
        // slots 5, 6) and token 0's never-copied slot 0.
        let mask = crate::attention_mask::create_ring_sliding_mask(
            1,
            ring.window,
            ring.capacity,
            ring.write_start - 1,
        );
        eval(&[&mask]);
        let bits: Vec<u8> =
            unsafe { std::slice::from_raw_parts(mask.data_raw(), mask.nbytes()).to_vec() };
        assert_eq!(bits, vec![0, 1, 1, 1, 1, 0, 0]);

        // Un-rotated layers keep the ordered peek contract.
        let plain = MlxKVCache::new(1);
        assert!(plain.layer_sliding_ring(0).is_none());
    }

    /// End-to-end oracle: masked SDPA over the ring (unordered slots + slot-
    /// validity mask) must equal unmasked SDPA over the ordered sliding
    /// window from a plain cache, through a conversion → wrap → rollback →
    /// rewrite trajectory. This is the property that makes bounded rings a
    /// drop-in for ordered window views.
    #[test]
    fn bounded_ring_sdpa_matches_ordered_window_reference() {
        use crate::attention_mask::create_ring_sliding_mask;
        use mlx_sys::{ScaledDotProductAttentionMask, scaled_dot_product_attention_with_mask};
        const HD: usize = 4;
        const WINDOW: usize = 4;
        const SLACK: usize = 3;
        let scale = 1.0 / (HD as f32).sqrt();

        let mut ring_cache = MlxKVCache::new(1);
        ring_cache.set_rotating_sliding_decode(true);
        ring_cache.set_rotating_sliding_slack(SLACK);
        let mut plain_cache = MlxKVCache::new(1);

        // Distinct K/V rows per token so misplaced slots change the output.
        let tok = |t: usize| ((t + 1) as f32) * 0.25;
        let prefill: Vec<f32> = (0..WINDOW).map(tok).collect();
        for cache in [&mut ring_cache, &mut plain_cache] {
            let k = tokens_f32(&prefill, HD);
            let v = tokens_f32(&prefill, HD);
            cache.append(0, k, v);
            cache.seq_len = WINDOW;
        }

        // Trajectory: 3-token verify (tokens 4-6), reject 2 (trim to 5),
        // 2-token re-verify (tokens 5-6 with new values), then a single-token
        // step (token 7) that wraps the ring.
        struct Step {
            values: Vec<f32>,
            trim_to: Option<usize>,
        }
        let steps = [
            Step {
                values: vec![tok(4), tok(5), tok(6)],
                trim_to: Some(5),
            },
            Step {
                values: vec![9.5, 10.5],
                trim_to: None,
            },
            Step {
                values: vec![11.5],
                trim_to: None,
            },
        ];

        for step in steps {
            let seq = step.values.len();
            let write_start = ring_cache.seq_len;
            assert_eq!(write_start, plain_cache.seq_len, "caches stay in sync");
            let q = tokens_f32(&step.values, HD); // queries; values arbitrary

            // Ring side: append with the raw window, mask over the ring.
            let ring = ring_cache
                .sliding_ring_layout(Some(WINDOW), seq)
                .expect("trajectory stays on the ring past the window");
            let k = tokens_f32(&step.values, HD);
            let v = tokens_f32(&step.values, HD);
            let (ring_k, ring_v) = ring_cache.append_with_retained_window(0, k, v, Some(WINDOW));
            let ring_mask =
                create_ring_sliding_mask(seq, ring.window, ring.capacity, ring.write_start);
            let ring_out = scaled_dot_product_attention_with_mask(
                &q,
                &ring_k,
                &ring_v,
                scale,
                ScaledDotProductAttentionMask::Array(&ring_mask),
                None,
            );

            // Plain side: ordered append; reference is the ordered full view
            // with the standard causal sliding-window mask, so both sides run
            // the same masked-SDPA kernel and the comparison isolates ring
            // slot/mask placement (kernel-level masked-vs-unmasked numeric
            // drift is ~1e-3 in f32 and not what this test is about).
            let k = tokens_f32(&step.values, HD);
            let v = tokens_f32(&step.values, HD);
            plain_cache.append_with_retained_window(0, k, v, None);
            let plain = contiguous_layer(&plain_cache, 0);
            let write_end = (write_start + seq) as i32;
            let ones = [1i32, 1, 1, 1];
            let ordered_k = slice(
                &plain.k,
                &[0, 0, 0, 0],
                &[1, 1, write_end, HD as i32],
                &ones,
                None,
            );
            let ordered_v = slice(
                &plain.v,
                &[0, 0, 0, 0],
                &[1, 1, write_end, HD as i32],
                &ones,
                None,
            );
            let ordered_mask =
                crate::attention_mask::create_causal_mask(seq, write_start, Some(WINDOW));
            let want = scaled_dot_product_attention_with_mask(
                &q,
                &ordered_k,
                &ordered_v,
                scale,
                ScaledDotProductAttentionMask::Array(&ordered_mask),
                None,
            );
            eval(&[&ring_out, &want]);
            let got = ring_out.data_f32().to_vec();
            let want = want.data_f32().to_vec();
            for i in 0..seq {
                for d in 0..HD {
                    let g = got[i * HD + d];
                    let w = want[i * HD + d];
                    assert!(
                        (g - w).abs() < 1e-5,
                        "step query {i} dim {d}: ring {g} vs ordered {w}"
                    );
                }
            }

            let new_len = write_start + seq;
            ring_cache.seq_len = new_len;
            plain_cache.seq_len = new_len;
            if let Some(target) = step.trim_to {
                assert!(ring_cache.trim_to(target), "trim within slack");
                assert!(plain_cache.trim_to(target));
            }
        }
    }

    // ── F3 M1 serialize / deserialize round-trip tests ──

    fn build_fa_array_f32(seq_len: usize, n_kv_heads: i32, head_dim: i32) -> MlxArray {
        let total = (n_kv_heads as usize) * seq_len * (head_dim as usize);
        let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.001).collect();
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data.as_slice()),
            &[1, n_kv_heads, seq_len as i32, head_dim],
            MlxDtype::Float32,
        )
    }

    fn build_mla_latent_f32(seq_len: usize, inner: i32) -> MlxArray {
        let total = seq_len * (inner as usize);
        let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.0007).collect();
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data.as_slice()),
            &[1, 1, seq_len as i32, inner],
            MlxDtype::Float32,
        )
    }

    fn host_f32(arr: &MlxArray) -> Vec<f32> {
        // Materialize a tight C-contiguous buffer first: `data_f32` on a
        // lazy slice of an over-allocated KV buffer can otherwise surface
        // capacity padding and break token-exact comparisons.
        let tight = contiguous(arr, None);
        eval(&[&tight]);
        tight.data_f32().to_vec()
    }

    #[test]
    fn serialize_empty_cache_roundtrips() {
        let cache = MlxKVCache::new(3);
        let bytes = cache.serialize_to_bytes();
        let restored = MlxKVCache::try_deserialize_from_bytes(&bytes).expect("round-trip");
        assert_eq!(restored.seq_len, 0);
        assert_eq!(restored.layers.len(), 3);
        assert!(restored.layers.iter().all(Option::is_none));
        assert!(restored.glm_mla_layers.iter().all(Option::is_none));
        assert!(
            restored
                .linear_layers
                .iter()
                .all(|l| l.conv_state.is_none() && l.recurrent_state.is_none())
        );
    }

    #[test]
    fn serialize_fa_cache_roundtrips_values() {
        // Two FA layers, varying head counts to catch shape mistakes.
        let mut cache = MlxKVCache::new(2);
        let seq_len = 4;
        let k0 = build_fa_array_f32(seq_len, 2, 8);
        let v0 = build_fa_array_f32(seq_len, 2, 8);
        let k1 = build_fa_array_f32(seq_len, 4, 16);
        let v1 = build_fa_array_f32(seq_len, 4, 16);
        cache.layers[0] = Some(FaLayerStorage::Contiguous(LayerKV {
            last_k_view: None,
            last_v_view: None,
            n_kv_heads: 2,
            head_dim: 8,
            capacity: seq_len,
            rotating_window: None,
            dtype: MlxDtype::Float32,
            k: k0,
            v: v0,
        }));
        cache.layers[1] = Some(FaLayerStorage::Contiguous(LayerKV {
            last_k_view: None,
            last_v_view: None,
            n_kv_heads: 4,
            head_dim: 16,
            capacity: seq_len,
            rotating_window: None,
            dtype: MlxDtype::Float32,
            k: k1,
            v: v1,
        }));
        cache.seq_len = seq_len;
        cache.growth_count = 7;

        let bytes = cache.serialize_to_bytes();
        let restored = MlxKVCache::try_deserialize_from_bytes(&bytes).expect("round-trip");

        assert_eq!(restored.seq_len, seq_len);
        assert_eq!(restored.growth_count, 7);
        for layer in 0..2 {
            let orig = contiguous_layer(&cache, layer);
            let back = contiguous_layer(&restored, layer);
            assert_eq!(back.n_kv_heads, orig.n_kv_heads);
            assert_eq!(back.head_dim, orig.head_dim);
            assert_eq!(back.capacity, orig.capacity);
            assert_eq!(back.dtype, orig.dtype);
            assert_eq!(host_f32(&back.k), host_f32(&orig.k));
            assert_eq!(host_f32(&back.v), host_f32(&orig.v));
        }
    }

    #[test]
    fn serialize_mla_cache_roundtrips_values() {
        let mut cache = MlxKVCache::new(1);
        let seq_len = 6;
        let latent_dim = 4;
        let rope_dim = 2;
        let kv_latent = build_mla_latent_f32(seq_len, latent_dim);
        let k_pe = build_mla_latent_f32(seq_len, rope_dim);
        cache.glm_mla_layers[0] = Some(GlmMlaLayerCache {
            latent_dim,
            rope_dim,
            capacity: seq_len,
            dtype: MlxDtype::Float32,
            kv_latent,
            k_pe,
        });
        cache.seq_len = seq_len;

        let bytes = cache.serialize_to_bytes();
        let restored = MlxKVCache::try_deserialize_from_bytes(&bytes).expect("round-trip");

        let orig = cache.glm_mla_layers[0].as_ref().unwrap();
        let back = restored.glm_mla_layers[0]
            .as_ref()
            .expect("mla layer present");
        assert_eq!(back.latent_dim, orig.latent_dim);
        assert_eq!(back.rope_dim, orig.rope_dim);
        assert_eq!(back.capacity, orig.capacity);
        assert_eq!(back.dtype, orig.dtype);
        assert_eq!(host_f32(&back.kv_latent), host_f32(&orig.kv_latent));
        assert_eq!(host_f32(&back.k_pe), host_f32(&orig.k_pe));
    }

    #[test]
    fn deserialize_rejects_bad_magic() {
        let bytes = b"NOPE\x00\x00\x00\x00".to_vec();
        let result = MlxKVCache::try_deserialize_from_bytes(&bytes);
        assert!(matches!(
            result,
            Err(MlxKVCacheSerializeError::BadMagic) | Err(MlxKVCacheSerializeError::UnexpectedEof)
        ));
    }

    #[test]
    fn deserialize_rejects_unsupported_version() {
        let mut payload = MlxKVCache::SERIALIZE_MAGIC.to_vec();
        payload.extend_from_slice(&99u32.to_le_bytes()); // wrong version
        payload.extend_from_slice(&0u64.to_le_bytes()); // seq_len
        payload.extend_from_slice(&0u64.to_le_bytes()); // growth_count
        payload.extend_from_slice(&0u64.to_le_bytes()); // rope_offset
        payload.extend_from_slice(&0u32.to_le_bytes()); // layer_count
        payload.extend_from_slice(&0u32.to_le_bytes()); // reserved
        let result = MlxKVCache::try_deserialize_from_bytes(&payload);
        assert!(matches!(
            result,
            Err(MlxKVCacheSerializeError::UnsupportedVersion(99))
        ));
    }

    #[test]
    fn deserialize_rejects_truncated_payload() {
        let cache = MlxKVCache::new(1);
        let bytes = cache.serialize_to_bytes();
        // Cut off the last byte to simulate a torn write.
        let truncated = &bytes[..bytes.len() - 1];
        let result = MlxKVCache::try_deserialize_from_bytes(truncated);
        assert!(matches!(
            result,
            Err(MlxKVCacheSerializeError::UnexpectedEof)
        ));
    }

    #[test]
    fn deserialize_rejects_undersized_byte_count() {
        // Hand-craft a payload whose tensor header declares a shape
        // requiring more bytes than `byte_count` advertises. Without
        // the pre-validation guard, `MlxArray::from_managed_data` would
        // panic; with it, we surface a structured `BadShape` error.
        let mut payload = Vec::new();
        payload.extend_from_slice(MlxKVCache::SERIALIZE_MAGIC);
        payload.extend_from_slice(&MlxKVCache::SERIALIZE_VERSION.to_le_bytes());
        payload.extend_from_slice(&0u64.to_le_bytes()); // seq_len
        payload.extend_from_slice(&0u64.to_le_bytes()); // growth_count
        payload.extend_from_slice(&0u64.to_le_bytes()); // rope_offset
        payload.extend_from_slice(&1u32.to_le_bytes()); // layer_count
        payload.extend_from_slice(&0u32.to_le_bytes()); // reserved

        // Single FA layer
        payload.push(MlxKVCache::LAYER_KIND_FA);
        payload.extend_from_slice(&[0u8; 7]);
        payload.extend_from_slice(&0u64.to_le_bytes()); // rotating_window: none
        // K tensor header: f32, 4-dim shape [1, 2, 4, 8] = 64 elements × 4 bytes
        payload.push(MlxKVCache::dtype_to_tag(MlxDtype::Float32));
        payload.push(4);
        payload.extend_from_slice(&[0u8; 6]);
        payload.extend_from_slice(&1i32.to_le_bytes());
        payload.extend_from_slice(&2i32.to_le_bytes());
        payload.extend_from_slice(&4i32.to_le_bytes());
        payload.extend_from_slice(&8i32.to_le_bytes());
        // Declared byte_count = 1 (too small for the declared shape)
        payload.extend_from_slice(&1u64.to_le_bytes());
        payload.push(0u8);

        let err = MlxKVCache::try_deserialize_from_bytes(&payload)
            .err()
            .expect("undersized byte_count must be rejected");
        assert!(
            matches!(err, MlxKVCacheSerializeError::BadShape(4)),
            "expected BadShape(4), got {err:?}"
        );
    }

    #[test]
    fn deserialize_rejects_unknown_dtype_tag() {
        // Hand-craft a payload whose tensor header carries a dtype tag that
        // is not in `dtype_from_tag`'s match table. The deserializer must
        // fail-close (PRD §7.1 "per-layer cache type, tensor shape, dtype")
        // rather than silently accept an unknown dtype and trip MLX later.
        let mut payload = Vec::new();
        payload.extend_from_slice(MlxKVCache::SERIALIZE_MAGIC);
        payload.extend_from_slice(&MlxKVCache::SERIALIZE_VERSION.to_le_bytes());
        payload.extend_from_slice(&0u64.to_le_bytes()); // seq_len
        payload.extend_from_slice(&0u64.to_le_bytes()); // growth_count
        payload.extend_from_slice(&0u64.to_le_bytes()); // rope_offset
        payload.extend_from_slice(&1u32.to_le_bytes()); // layer_count
        payload.extend_from_slice(&0u32.to_le_bytes()); // reserved

        // Single FA layer with one tensor carrying an invalid dtype.
        payload.push(MlxKVCache::LAYER_KIND_FA);
        payload.extend_from_slice(&[0u8; 7]);
        payload.extend_from_slice(&0u64.to_le_bytes()); // rotating_window: none
        // 0xEE is not a valid dtype tag in dtype_from_tag's table.
        payload.push(0xEE);
        payload.push(4); // ndim
        payload.extend_from_slice(&[0u8; 6]); // reserved
        payload.extend_from_slice(&1i32.to_le_bytes());
        payload.extend_from_slice(&1i32.to_le_bytes());
        payload.extend_from_slice(&1i32.to_le_bytes());
        payload.extend_from_slice(&1i32.to_le_bytes());
        payload.extend_from_slice(&4u64.to_le_bytes()); // byte_count

        let err = MlxKVCache::try_deserialize_from_bytes(&payload)
            .err()
            .expect("unknown dtype tag must be rejected");
        assert!(
            matches!(err, MlxKVCacheSerializeError::UnknownDtype(0xEE)),
            "expected UnknownDtype(0xEE), got {err:?}"
        );
    }

    #[test]
    fn deserialize_rejects_unknown_layer_kind() {
        // PRD §7.1 requires per-layer cache type validation. An unknown
        // discriminator byte at the layer header position must fail-close.
        let mut payload = Vec::new();
        payload.extend_from_slice(MlxKVCache::SERIALIZE_MAGIC);
        payload.extend_from_slice(&MlxKVCache::SERIALIZE_VERSION.to_le_bytes());
        payload.extend_from_slice(&0u64.to_le_bytes()); // seq_len
        payload.extend_from_slice(&0u64.to_le_bytes()); // growth_count
        payload.extend_from_slice(&0u64.to_le_bytes()); // rope_offset
        payload.extend_from_slice(&1u32.to_le_bytes()); // layer_count
        payload.extend_from_slice(&0u32.to_le_bytes()); // reserved

        // 0x7F is intentionally outside the four documented layer kinds
        // (EMPTY/FA/MLA/LINEAR) but inside `u8`'s range.
        payload.push(0x7F);
        payload.extend_from_slice(&[0u8; 7]);

        let err = MlxKVCache::try_deserialize_from_bytes(&payload)
            .err()
            .expect("unknown layer kind must be rejected");
        assert!(
            matches!(err, MlxKVCacheSerializeError::UnknownLayerKind(0x7F)),
            "expected UnknownLayerKind(0x7F), got {err:?}"
        );
    }

    #[test]
    fn deserialize_rejects_zero_rank_tensor() {
        // Tensor rank 0 is never valid for an FA layer; the deserializer's
        // shape guard (`ndim == 0 || ndim > 4`) must reject before any
        // MlxArray construction is attempted. This is the dtype-aware
        // complement to `deserialize_rejects_undersized_byte_count`.
        let mut payload = Vec::new();
        payload.extend_from_slice(MlxKVCache::SERIALIZE_MAGIC);
        payload.extend_from_slice(&MlxKVCache::SERIALIZE_VERSION.to_le_bytes());
        payload.extend_from_slice(&0u64.to_le_bytes());
        payload.extend_from_slice(&0u64.to_le_bytes());
        payload.extend_from_slice(&0u64.to_le_bytes()); // rope_offset
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());

        payload.push(MlxKVCache::LAYER_KIND_FA);
        payload.extend_from_slice(&[0u8; 7]);
        payload.extend_from_slice(&0u64.to_le_bytes()); // rotating_window: none
        payload.push(MlxKVCache::dtype_to_tag(MlxDtype::Float32));
        payload.push(0); // ndim = 0 (invalid)
        payload.extend_from_slice(&[0u8; 6]);
        payload.extend_from_slice(&[0u8; 16]); // 4 i32 shape entries
        payload.extend_from_slice(&0u64.to_le_bytes());

        let err = MlxKVCache::try_deserialize_from_bytes(&payload)
            .err()
            .expect("rank 0 tensor must be rejected");
        assert!(
            matches!(err, MlxKVCacheSerializeError::BadShape(0)),
            "expected BadShape(0), got {err:?}"
        );
    }

    #[test]
    fn serialize_trims_fa_capacity_to_logical_seq_len() {
        // The backing buffer holds `capacity` tokens but only `seq_len`
        // are logical; the payload must carry the logical prefix only.
        let capacity = 8usize;
        let seq_len = 3usize;
        let head_dim = 4;
        let n_kv_heads = 2;
        let mut cache = MlxKVCache::new(1);
        let k = build_fa_array_f32(capacity, n_kv_heads, head_dim);
        let v = build_fa_array_f32(capacity, n_kv_heads, head_dim);
        // The slice is a strided view; materialize it before reading host
        // bytes (data_f32 on a strided view walks the backing buffer
        // linearly — the same hazard serialize_tensor_logical guards).
        let expected_k = host_f32(&contiguous(
            &slice(
                &k,
                &[0, 0, 0, 0],
                &[1, n_kv_heads, seq_len as i32, head_dim],
                &[1, 1, 1, 1],
                None,
            ),
            None,
        ));
        cache.layers[0] = Some(FaLayerStorage::Contiguous(LayerKV {
            last_k_view: None,
            last_v_view: None,
            n_kv_heads,
            head_dim,
            capacity,
            rotating_window: None,
            dtype: MlxDtype::Float32,
            k,
            v,
        }));
        cache.seq_len = seq_len;

        let trimmed_bytes = cache.serialize_to_bytes();
        let restored = MlxKVCache::try_deserialize_from_bytes(&trimmed_bytes).expect("round-trip");
        assert_eq!(restored.seq_len, seq_len);
        let restored_layer = contiguous_layer(&restored, 0);
        assert_eq!(
            restored_layer.capacity, seq_len,
            "restored capacity must equal the logical length, not the source capacity"
        );
        assert_eq!(
            host_f32(&restored_layer.k),
            expected_k,
            "restored K must match the logical prefix of the source buffer"
        );

        // Payload must scale with seq_len, not capacity: the same cache
        // reporting full capacity as logical length serializes strictly more.
        cache.seq_len = capacity;
        let full_bytes = cache.serialize_to_bytes();
        assert!(
            trimmed_bytes.len() < full_bytes.len(),
            "trimmed payload ({}) must be smaller than full-capacity payload ({})",
            trimmed_bytes.len(),
            full_bytes.len()
        );
    }

    #[test]
    fn serialize_roundtrips_rope_offset() {
        let mut cache = MlxKVCache::new(1);
        cache.seq_len = 4;
        cache.rope_offset = 9;

        let restored = MlxKVCache::try_deserialize_from_bytes(&cache.serialize_to_bytes())
            .expect("round-trip");

        assert_eq!(restored.seq_len, 4);
        assert_eq!(
            restored.rope_offset, 9,
            "rope_offset is positional state and must survive the round trip"
        );
    }

    #[test]
    fn serialize_roundtrips_rotated_ring_geometry_and_appends_continue() {
        const HD: usize = 4;
        // Build a bounded ring exactly like the rotating-decode tests:
        // window 4, slack 3 → capacity 7, tokens 0..=8 with 7 and 8 wrapped.
        let mut cache = MlxKVCache::new(1);
        cache.set_rotating_sliding_decode(true);
        cache.set_rotating_sliding_slack(3);
        let k = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
        let v = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
        cache.append(0, k, v);
        cache.seq_len = 4;
        for (t, value) in [(4usize, 5.0f32), (5, 6.0), (6, 7.0), (7, 8.0), (8, 9.0)] {
            let k = tokens_f32(&[value], HD);
            let v = tokens_f32(&[value], HD);
            cache.append_with_retained_window(0, k, v, Some(4));
            cache.seq_len = t + 1;
        }
        let expected_slots = token_row_values(&contiguous_layer(&cache, 0).k, HD);
        assert_eq!(expected_slots, vec![8.0, 9.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        let restored = MlxKVCache::try_deserialize_from_bytes(&cache.serialize_to_bytes())
            .expect("round-trip");
        let lkv = contiguous_layer(&restored, 0);
        assert_eq!(lkv.rotating_window, Some(4), "ring window must survive");
        assert_eq!(lkv.capacity, 7, "ring capacity must survive");
        assert_eq!(restored.rotating_sliding_slack(), 3, "slack re-latched");
        assert_eq!(
            token_row_values(&lkv.k, HD),
            expected_slots,
            "slot-ordered ring contents must survive byte-identical"
        );

        // Post-restore decode must keep rotating: token 9 lands at slot
        // 9 % 7 = 2, not at logical position 9 of an ordered buffer.
        let mut restored = restored;
        let k = tokens_f32(&[10.0], HD);
        let v = tokens_f32(&[10.0], HD);
        restored.append_with_retained_window(0, k, v, Some(4));
        restored.seq_len = 10;
        let lkv = contiguous_layer(&restored, 0);
        assert_eq!(lkv.capacity, 7, "restored ring must not regrow");
        assert_eq!(
            token_row_values(&lkv.k, HD),
            vec![8.0, 9.0, 10.0, 4.0, 5.0, 6.0, 7.0]
        );
    }

    #[test]
    #[should_panic(expected = "ordered KV append on rotated ring layer")]
    fn ordered_append_on_rotated_ring_fails_closed() {
        const HD: usize = 4;
        let mut cache = MlxKVCache::new(1);
        cache.set_rotating_sliding_decode(true);
        let k = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
        let v = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
        cache.append(0, k, v);
        cache.seq_len = 4;
        // Convert to a pure ring (window 4, slack 0).
        let k = tokens_f32(&[5.0], HD);
        let v = tokens_f32(&[5.0], HD);
        cache.append_with_retained_window(0, k, v, Some(4));
        cache.seq_len = 5;

        // A 2-token forward is not ring-eligible in pure mode; before the
        // fail-closed assert this fell through to the ordered path and
        // silently grew the ring, copying slots as a token-ordered prefix.
        let k = tokens_f32(&[6.0, 7.0], HD);
        let v = tokens_f32(&[6.0, 7.0], HD);
        let _ = cache.append_with_retained_window(0, k, v, Some(4));
    }

    #[test]
    fn deserialized_cache_outlives_input_buffer() {
        // Regression test for the lifetime bug fixed alongside this
        // commit: `from_raw_data` borrows its data pointer (per the
        // mlx-sys array.rs:80 doc, "MLX does **not** copy"), so handing
        // it a slice of the caller's input buffer would leave the
        // deserialised array dangling once that buffer is freed.
        // `try_deserialize_from_bytes` must construct arrays that own
        // their data via `from_managed_data` + a heap-Box deleter.
        //
        // This test arranges the scenario explicitly: build an input
        // buffer, deserialise from it, drop the buffer, then read the
        // cache's tensors. With the fix, every read returns the
        // original byte pattern; without the fix this exhibits
        // undefined behaviour (typically reads bogus values or
        // SIGBUS / SIGSEGV under MLX's evaluator).
        let seq_len = 4;
        let head_dim = 8;
        let n_kv_heads = 2;
        let original = {
            let mut cache = MlxKVCache::new(1);
            let k = build_fa_array_f32(seq_len, n_kv_heads, head_dim);
            let v = build_fa_array_f32(seq_len, n_kv_heads, head_dim);
            cache.layers[0] = Some(FaLayerStorage::Contiguous(LayerKV {
                last_k_view: None,
                last_v_view: None,
                n_kv_heads,
                head_dim,
                capacity: seq_len,
                rotating_window: None,
                dtype: MlxDtype::Float32,
                k,
                v,
            }));
            cache.seq_len = seq_len;
            cache
        };
        let expected_k = host_f32(&contiguous_layer(&original, 0).k);
        let expected_v = host_f32(&contiguous_layer(&original, 0).v);

        let restored = {
            let bytes = original.serialize_to_bytes();
            MlxKVCache::try_deserialize_from_bytes(&bytes).expect("round-trip")
            // `bytes` drops here. The restored cache must remain valid.
        };

        // Read the restored tensors AFTER the input buffer has been
        // dropped. If `read_tensor` had borrowed the slice, this would
        // be UB; the managed-data + heap-owned pattern keeps it sound.
        let restored_k = host_f32(&contiguous_layer(&restored, 0).k);
        let restored_v = host_f32(&contiguous_layer(&restored, 0).v);
        assert_eq!(restored_k, expected_k);
        assert_eq!(restored_v, expected_v);
    }

    // ── PR4 FA block-pool path: token-exact oracle vs contiguous ──

    fn fa_token_values(seq_len: usize, n_kv_heads: i32, head_dim: i32, base: f32) -> MlxArray {
        let total = (n_kv_heads as usize) * seq_len * (head_dim as usize);
        let data: Vec<f32> = (0..total).map(|i| base + (i as f32) * 0.01).collect();
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data.as_slice()),
            &[1, n_kv_heads, seq_len as i32, head_dim],
            MlxDtype::Float32,
        )
    }

    #[test]
    fn fa_paged_append_trim_oracle_matches_contiguous() {
        // block_size=4 so multi-block growth + partial last block exercise
        // materialize and free_blocks_beyond.
        let config = FaBlockPoolConfig {
            block_size_tokens: 4,
            max_blocks: 32,
        };
        let mut paged = MlxKVCache::new_with_fa_block_pool(1, config);
        let mut contig = MlxKVCache::new_contiguous(1);
        assert!(paged.fa_block_pool_enabled());
        assert!(!contig.fa_block_pool_enabled());

        let steps: &[(usize, f32)] = &[(3, 1.0), (5, 2.0), (1, 3.0), (6, 4.0)];
        let n_kv_heads = 2i32;
        let head_dim = 4i32;
        let mut seq = 0usize;
        for &(n, base) in steps {
            let k = fa_token_values(n, n_kv_heads, head_dim, base);
            let v = fa_token_values(n, n_kv_heads, head_dim, base + 0.5);
            let (pk, pv) = paged.append(0, k.clone(), v.clone());
            let (ck, cv) = contig.append(0, k, v);
            seq += n;
            paged.advance(n);
            contig.advance(n);
            eval(&[&pk, &pv, &ck, &cv]);
            assert_eq!(
                host_f32(&pk),
                host_f32(&ck),
                "K mismatch after append to {seq}"
            );
            assert_eq!(
                host_f32(&pv),
                host_f32(&cv),
                "V mismatch after append to {seq}"
            );
            assert_eq!(pk.shape(), ck.shape());
        }

        // Trim into the middle of a block; trailing full blocks free.
        assert!(paged.trim_to(7));
        assert!(contig.trim_to(7));
        let (pk, pv) = paged.logical_layer_kv(0).expect("paged layer");
        let (ck, cv) = contig.logical_layer_kv(0).expect("contig layer");
        eval(&[&pk, &pv, &ck, &cv]);
        assert_eq!(host_f32(&pk), host_f32(&ck), "K mismatch after trim_to(7)");
        assert_eq!(host_f32(&pv), host_f32(&cv), "V mismatch after trim_to(7)");

        // Re-append after trim overwrites the trimmed region.
        let k = fa_token_values(3, n_kv_heads, head_dim, 9.0);
        let v = fa_token_values(3, n_kv_heads, head_dim, 9.5);
        let (pk, pv) = paged.append(0, k.clone(), v.clone());
        let (ck, cv) = contig.append(0, k, v);
        paged.advance(3);
        contig.advance(3);
        eval(&[&pk, &pv, &ck, &cv]);
        assert_eq!(host_f32(&pk), host_f32(&ck), "K mismatch after re-append");
        assert_eq!(host_f32(&pv), host_f32(&cv), "V mismatch after re-append");

        // Serialize materializes dense; round-trip matches contiguous values.
        let paged_bytes = paged.serialize_to_bytes();
        let contig_bytes = contig.serialize_to_bytes();
        // Growth counts may differ (block vs chunk grow), so compare tensors
        // rather than full wire equality.
        let p_restored =
            MlxKVCache::try_deserialize_from_bytes(&paged_bytes).expect("paged serialize");
        let c_restored =
            MlxKVCache::try_deserialize_from_bytes(&contig_bytes).expect("contig serialize");
        assert_eq!(p_restored.seq_len(), c_restored.seq_len());
        assert_eq!(
            host_f32(&contiguous_layer(&p_restored, 0).k),
            host_f32(&contiguous_layer(&c_restored, 0).k)
        );
        assert_eq!(
            host_f32(&contiguous_layer(&p_restored, 0).v),
            host_f32(&contiguous_layer(&c_restored, 0).v)
        );
        assert!(!p_restored.fa_block_pool_enabled());
    }

    #[test]
    fn fa_paged_pool_exhaustion_demotes_to_contiguous() {
        let config = FaBlockPoolConfig {
            block_size_tokens: 4,
            max_blocks: 2, // only 8 tokens of private capacity
        };
        let mut cache = MlxKVCache::new_with_fa_block_pool(1, config);
        let k = fa_token_values(8, 1, 4, 1.0);
        let v = fa_token_values(8, 1, 4, 2.0);
        let _ = cache.append(0, k, v);
        cache.advance(8);
        assert_eq!(cache.fa_block_pool_available(), Some(0));

        let k2 = fa_token_values(1, 1, 4, 3.0);
        let v2 = fa_token_values(1, 1, 4, 4.0);
        let (out_k, _) = cache.append(0, k2, v2);
        cache.advance(1);
        eval(&[&out_k]);
        assert_eq!(out_k.shape()[2], 9);
        let usage = cache.usage_snapshot();
        assert_eq!(
            usage.paged_pool_exhaustion_fallbacks, 1,
            "pool exhaustion must demote rather than panic"
        );
        // Layer is now contiguous; further appends stay contiguous.
        assert!(matches!(
            cache.layers[0],
            Some(FaLayerStorage::Contiguous(_))
        ));
    }

    #[test]
    fn fa_paged_clone_diverges_without_double_free() {
        let config = FaBlockPoolConfig {
            block_size_tokens: 4,
            max_blocks: 16,
        };
        let mut a = MlxKVCache::new_with_fa_block_pool(1, config);
        let k = fa_token_values(5, 1, 4, 1.0);
        let v = fa_token_values(5, 1, 4, 2.0);
        let _ = a.append(0, k, v);
        a.advance(5);
        let b = a.clone();
        assert!(a.trim_to(2));
        // Clone must still hold its private blocks after source trim frees IDs.
        let (bk, _) = b.logical_layer_kv(0).expect("clone layer");
        eval(&[&bk]);
        assert_eq!(bk.shape()[2], 5);
        let k2 = fa_token_values(1, 1, 4, 9.0);
        let v2 = fa_token_values(1, 1, 4, 9.5);
        let (ak, _) = a.append(0, k2, v2);
        a.advance(1);
        eval(&[&ak]);
        assert_eq!(ak.shape()[2], 3);
        // Drop both; free lists must not double-free (pool drop is fine).
        drop(a);
        drop(b);
    }
}
