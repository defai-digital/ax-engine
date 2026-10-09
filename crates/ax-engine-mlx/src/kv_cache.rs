use std::time::Instant;

use mlx_sys::{
    MlxArray, MlxDtype, MlxQuantizationMode, concatenate, contiguous, dequantize_with_mode, eval,
    quantize, slice, slice_update, zeros,
};

use crate::kv_block_pool::{
    FaBlockPoolConfig, FaBlockPoolError, PhysicalBlockId, SharedFaBlockPool,
    default_fa_block_pool_config, fa_kv_block_pool_enabled,
};
use crate::paged_attention::PagedAttentionView;

use crate::model::KvQuantSpec;

/// Pre-allocated chunk size (tokens).  The buffer grows by this amount each time
/// the logical sequence length exceeds capacity, so the number of grow operations
/// per session is at most ceil(total_tokens / CHUNK).
pub(crate) const KV_CHUNK_TOKENS: usize = 256;

/// Env kill-switch for KV-cache quantization: set to `0` to disable KV-cache
/// quantization even when the model manifest declares a `kv_cache_quantization`
/// table. Read by the runtime quantization path (Phase 3b).
pub(crate) const AX_KV_QUANT_ENV: &str = "AX_KV_QUANT";

/// Env kill-switch check for KV-cache quantization (Phase 3b). Honored inside
/// [`MlxKVCache::set_kv_quant_table`] — the single place the gate is read — so
/// every injection site gets the same behavior.
fn kv_quant_env_disabled() -> bool {
    std::env::var(AX_KV_QUANT_ENV).is_ok_and(|value| value == "0")
}

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DeepseekV4AppendShape {
    new_tokens: usize,
    head_dim: i32,
    dtype: MlxDtype,
}

fn validate_deepseek_v4_append_inputs(
    layer: usize,
    layer_count: usize,
    new_k_latent: &MlxArray,
) -> DeepseekV4AppendShape {
    assert!(
        layer < layer_count,
        "DeepSeek V4 cache layer {layer} out of bounds for {layer_count} layers"
    );

    let shape = new_k_latent.shape();
    assert_eq!(
        shape.len(),
        4,
        "DeepSeek V4 latent K cache append expects [1, 1, tokens, head_dim]"
    );
    assert_eq!(shape[0], 1, "DeepSeek V4 cache supports batch=1 only");
    assert_eq!(
        shape[1], 1,
        "DeepSeek V4 cache stores one latent K head (K doubles as V)"
    );
    assert!(
        shape[2] > 0 && shape[3] > 0,
        "DeepSeek V4 cache append requires positive tokens and head_dim"
    );

    DeepseekV4AppendShape {
        new_tokens: shape[2] as usize,
        head_dim: shape[3],
        dtype: new_k_latent.dtype(),
    }
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
    /// Full-attention layers currently holding quantized packed storage.
    /// Zero when no manifest table applies, `AX_KV_QUANT=0`, or every table
    /// layer was demoted back to dense (ring engagement, repage).
    pub quantized_layers: usize,
    pub linear_state_layers: usize,
    pub linear_state_bytes: u64,
    pub growth_count: u64,
    /// Cumulative microseconds spent materializing dense FA views from private
    /// paged blocks (PR4). Zero when the contiguous path is used.
    pub paged_materialize_us: u64,
    /// Times a paged FA append fell back to contiguous growth because the
    /// owning block pool was exhausted. Production hard-cap callers fail the
    /// request after the dense compatibility graph is completed.
    pub paged_pool_exhaustion_fallbacks: u64,
    /// Blocks copied on first divergent write after a paged clone/adoption.
    pub paged_cow_copies: u64,
    /// Pool-wide physical blocks currently allocated. Gauge, not additive.
    pub paged_pool_blocks_used: u32,
    /// Pool-wide allocated blocks with more than one owning view. Gauge.
    pub paged_pool_shared_blocks: u32,
    /// Fixed-size K/V slabs currently allocated across all pool layers.
    pub paged_pool_slabs: u32,
    /// Real MLX bytes reserved by fixed K/V slabs, including slack rows.
    pub paged_pool_slab_bytes: u64,
    /// Pool-wide episodes that appended slabs after a layer's first slab.
    pub paged_pool_slab_grow_events: u64,
    /// Decode attention calls that consumed a single-slab block table.
    pub paged_attention_calls: u64,
    /// Eligible single-slab calls that fell back to gather + dense SDPA.
    pub paged_attention_fallbacks: u64,
}

#[derive(Clone)]
struct ProtectedPrefixRing {
    /// Number of prompt/image tokens retained permanently.
    prefix_len: usize,
    /// Number of generated-token slots rotated after the prefix.
    window: usize,
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
    protected_prefix_ring: Option<ProtectedPrefixRing>,
    dtype: MlxDtype,
}

/// Quantized backing store for one tensor (K or V) of a full-attention layer
/// (Phase 3b per-layer KV-cache quantization).
///
/// Grouping runs along `head_dim` — the last axis — so each token's group
/// boundaries are independent of its neighbors: an append quantizes only the
/// new token slice and `slice_update`s the three buffers, and a read
/// dequantizes exactly the requested token range back to the layer dtype.
/// The buffers grow with the same `KV_CHUNK_TOKENS` chunked mechanics as
/// [`LayerKV`].
#[derive(Clone)]
struct QuantizedTensorKV {
    /// Packed integers: `[1, n_kv_heads, capacity, head_dim * bits / 32]` u32.
    packed: MlxArray,
    /// Per-group scales: `[1, n_kv_heads, capacity, head_dim / group_size]`.
    scales: MlxArray,
    /// Per-group biases: same shape as `scales`.
    biases: MlxArray,
}

impl QuantizedTensorKV {
    fn packed_dim(head_dim: i32, bits: u32) -> i32 {
        head_dim * (bits as i32) / 32
    }

    fn group_dim(head_dim: i32, group_size: u32) -> i32 {
        head_dim / (group_size as i32)
    }

    fn zero_buffers(
        n_kv_heads: i32,
        capacity: usize,
        head_dim: i32,
        dtype: MlxDtype,
        spec: KvQuantSpec,
    ) -> Self {
        let packed_shape = [
            1i32,
            n_kv_heads,
            capacity as i32,
            Self::packed_dim(head_dim, spec.bits),
        ];
        let group_shape = [
            1i32,
            n_kv_heads,
            capacity as i32,
            Self::group_dim(head_dim, spec.group_size),
        ];
        Self {
            packed: zeros(&packed_shape, MlxDtype::Uint32, None),
            scales: zeros(&group_shape, dtype, None),
            biases: zeros(&group_shape, dtype, None),
        }
    }

    /// Quantize `new` (`[1, n_kv_heads, capacity, head_dim]`) and adopt the
    /// outputs directly as the backing buffers (chunk-aligned fast path).
    fn from_quantized(new: &MlxArray, spec: KvQuantSpec) -> Self {
        let mut parts = quantize(
            new,
            Some(spec.group_size as i32),
            Some(spec.bits as i32),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        assert!(
            parts.len() == 3,
            "affine quantize must return [packed, scales, biases]"
        );
        Self {
            biases: parts.remove(2),
            scales: parts.remove(1),
            packed: parts.remove(0),
        }
    }

    /// Quantize `new` (`[1, n_kv_heads, new_tokens, head_dim]`) and write it
    /// into the `[write_start..write_end)` token slots of the three buffers.
    fn write_tokens(
        &mut self,
        new: &MlxArray,
        write_start: usize,
        write_end: usize,
        n_kv_heads: i32,
        spec: KvQuantSpec,
    ) {
        let mut parts = quantize(
            new,
            Some(spec.group_size as i32),
            Some(spec.bits as i32),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        assert!(
            parts.len() == 3,
            "affine quantize must return [packed, scales, biases]"
        );
        let biases = parts.remove(2);
        let scales = parts.remove(1);
        let packed = parts.remove(0);
        let packed_dim = self.packed.shape()[3];
        let group_dim = self.scales.shape()[3];
        let start = [0i32, 0, write_start as i32, 0];
        let packed_stop = [1i32, n_kv_heads, write_end as i32, packed_dim];
        let group_stop = [1i32, n_kv_heads, write_end as i32, group_dim];
        let strides = [1i32, 1, 1, 1];
        self.packed = slice_update(&self.packed, &packed, &start, &packed_stop, &strides, None);
        self.scales = slice_update(&self.scales, &scales, &start, &group_stop, &strides, None);
        self.biases = slice_update(&self.biases, &biases, &start, &group_stop, &strides, None);
    }

    /// Dequantize the `[token_start..token_end)` token slice back to `dtype`.
    fn dequantize_tokens(
        &self,
        token_start: usize,
        token_end: usize,
        n_kv_heads: i32,
        head_dim: i32,
        dtype: MlxDtype,
        spec: KvQuantSpec,
    ) -> MlxArray {
        if token_end == token_start {
            return zeros(&[1i32, n_kv_heads, 0, head_dim], dtype, None);
        }
        let packed_dim = self.packed.shape()[3];
        let group_dim = self.scales.shape()[3];
        let start = [0i32, 0, token_start as i32, 0];
        let strides = [1i32, 1, 1, 1];
        // MLX 0.32.1's affine dequantize path does not preserve the leading
        // strides of token-range views into a capacity-sized buffer. Feeding
        // those views directly corrupts later heads (8-bit values can jump by
        // ~512). Materialize tight quantized views before dequantization; this
        // is also an explicit boundary between the cache's capacity layout and
        // the quantization kernel's dense last-axis contract.
        let packed = contiguous(
            &slice(
                &self.packed,
                &start,
                &[1, n_kv_heads, token_end as i32, packed_dim],
                &strides,
                None,
            ),
            None,
        );
        let scales = contiguous(
            &slice(
                &self.scales,
                &start,
                &[1, n_kv_heads, token_end as i32, group_dim],
                &strides,
                None,
            ),
            None,
        );
        let biases = contiguous(
            &slice(
                &self.biases,
                &start,
                &[1, n_kv_heads, token_end as i32, group_dim],
                &strides,
                None,
            ),
            None,
        );
        dequantize_with_mode(
            &packed,
            &scales,
            Some(&biases),
            Some(spec.group_size as i32),
            Some(spec.bits as i32),
            MlxQuantizationMode::Affine,
            None,
            Some(dtype),
            None,
        )
    }
}

/// Full-attention layer stored quantized (Phase 3b). Quantize-on-append,
/// dequantize-on-read; attention consumers keep receiving the same owned
/// dense `[1, n_kv_heads, tokens, head_dim]` views as the dense path.
///
/// Quantized layers never take the paged route and never become rotating /
/// protected-prefix rings (a ring engagement demotes the layer back to dense
/// storage first), so no ring geometry lives here.
#[derive(Clone)]
struct QuantizedLayerKV {
    k: QuantizedTensorKV,
    v: QuantizedTensorKV,
    /// Cached dense views returned by the last append; same reuse contract as
    /// [`LayerKV::last_k_view`] for KV-shared layers.
    last_k_view: Option<MlxArray>,
    last_v_view: Option<MlxArray>,
    n_kv_heads: i32,
    head_dim: i32,
    capacity: usize,
    dtype: MlxDtype,
    spec: KvQuantSpec,
}

impl QuantizedLayerKV {
    fn clear_views(&mut self) {
        self.last_k_view = None;
        self.last_v_view = None;
    }

    /// Grow all six buffers to `chunk_ceiling(write_end)` tokens, preserving
    /// packed contents. Mirrors [`LayerKV`]'s chunked growth.
    fn ensure_capacity(&mut self, write_end: usize, growth_count: &mut u64) {
        if write_end <= self.capacity {
            return;
        }
        let new_capacity = chunk_ceiling(write_end);
        let start = [0i32, 0, 0, 0];
        let old_token_stop = self.capacity as i32;
        let strides = [1i32, 1, 1, 1];
        let grow = |t: &mut QuantizedTensorKV| {
            let fresh = QuantizedTensorKV::zero_buffers(
                self.n_kv_heads,
                new_capacity,
                self.head_dim,
                self.dtype,
                self.spec,
            );
            let packed_dim = t.packed.shape()[3];
            let group_dim = t.scales.shape()[3];
            t.packed = slice_update(
                &fresh.packed,
                &t.packed,
                &start,
                &[1, self.n_kv_heads, old_token_stop, packed_dim],
                &strides,
                None,
            );
            t.scales = slice_update(
                &fresh.scales,
                &t.scales,
                &start,
                &[1, self.n_kv_heads, old_token_stop, group_dim],
                &strides,
                None,
            );
            t.biases = slice_update(
                &fresh.biases,
                &t.biases,
                &start,
                &[1, self.n_kv_heads, old_token_stop, group_dim],
                &strides,
                None,
            );
        };
        grow(&mut self.k);
        grow(&mut self.v);
        self.capacity = new_capacity;
        self.clear_views();
        *growth_count = growth_count.saturating_add(1);
    }

    fn write_tokens(&mut self, write_start: usize, new_k: &MlxArray, new_v: &MlxArray) {
        let write_end = write_start + new_k.shape()[2] as usize;
        assert!(
            write_end <= self.capacity,
            "quantized KV write past capacity"
        );
        self.k
            .write_tokens(new_k, write_start, write_end, self.n_kv_heads, self.spec);
        self.v
            .write_tokens(new_v, write_start, write_end, self.n_kv_heads, self.spec);
        self.clear_views();
    }

    /// Owned dense K/V views over `[token_start..token_end)`, dequantized to
    /// the layer dtype — the same contract dense layers return for SDPA.
    fn dense_view(&self, token_start: usize, token_end: usize) -> (MlxArray, MlxArray) {
        (
            self.k.dequantize_tokens(
                token_start,
                token_end,
                self.n_kv_heads,
                self.head_dim,
                self.dtype,
                self.spec,
            ),
            self.v.dequantize_tokens(
                token_start,
                token_end,
                self.n_kv_heads,
                self.head_dim,
                self.dtype,
                self.spec,
            ),
        )
    }

    /// Physical bytes held per token across K and V (packed + scales + biases).
    fn bytes_per_token(&self) -> u64 {
        let dtype_bytes = self.dtype.size_bytes() as u64;
        let per_tensor = (self.n_kv_heads as u64).saturating_mul(
            (QuantizedTensorKV::packed_dim(self.head_dim, self.spec.bits) as u64)
                .saturating_mul(4)
                .saturating_add(
                    (QuantizedTensorKV::group_dim(self.head_dim, self.spec.group_size) as u64)
                        .saturating_mul(dtype_bytes)
                        .saturating_mul(2),
                ),
        );
        per_tensor.saturating_mul(2)
    }
}

/// FA block list for one layer (private or runner-shared paged path).
///
/// Each block is a full `[1, n_kv_heads, block_size, head_dim]` slab. Logical
/// tokens fill blocks left-to-right; SDPA consumers materialize a dense
/// `[1, n_kv_heads, T, head_dim]` view via [`PagedFaLayer::materialize`].
#[derive(Clone)]
struct PagedFaLayer {
    layer_idx: usize,
    n_kv_heads: i32,
    head_dim: i32,
    dtype: MlxDtype,
    block_size: usize,
    block_ids: Vec<PhysicalBlockId>,
    /// Runner-sharing storage uses mlxcel-style fixed per-layer slabs. The
    /// private compatibility route keeps one array handle per block below.
    slab_storage: bool,
    k_blocks: Vec<MlxArray>,
    v_blocks: Vec<MlxArray>,
    last_k_view: Option<MlxArray>,
    last_v_view: Option<MlxArray>,
}

pub(crate) enum MlxAttentionKv {
    Dense { k: MlxArray, v: MlxArray },
    Paged(PagedAttentionView),
}

impl MlxAttentionKv {
    pub(crate) fn key_len(&self) -> usize {
        match self {
            Self::Dense { k, .. } => k.shape()[2] as usize,
            Self::Paged(view) => view.key_len,
        }
    }

    pub(crate) fn into_dense(self) -> (MlxArray, MlxArray) {
        match self {
            Self::Dense { k, v } => (k, v),
            Self::Paged(view) => view.materialize(),
        }
    }
}

/// Owns one all-or-nothing pool allocation until a repaged cache has adopted
/// every ID. This closes the unwind edge between allocation and construction:
/// MLX tensor slicing is expected to be infallible for validated shapes, but a
/// panic must still return the reserved IDs instead of leaking runner capacity.
struct FaBlockReservation {
    pool: SharedFaBlockPool,
    ids: Vec<PhysicalBlockId>,
    armed: bool,
}

impl FaBlockReservation {
    fn new(pool: SharedFaBlockPool, count: u32) -> Result<Self, FaBlockPoolError> {
        let ids = pool.allocate(count)?;
        Ok(Self {
            pool,
            ids,
            armed: true,
        })
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for FaBlockReservation {
    fn drop(&mut self) {
        if self.armed
            && let Err(error) = self.pool.free(&self.ids)
        {
            tracing::error!(
                target: "ax_engine_mlx::kv_pool",
                %error,
                "failed to release reserved FA blocks after repage construction",
            );
        }
    }
}

impl PagedFaLayer {
    fn capacity_tokens(&self) -> usize {
        self.block_ids.len().saturating_mul(self.block_size)
    }

    fn clear_views(&mut self) {
        self.last_k_view = None;
        self.last_v_view = None;
    }

    fn attention_view(
        &self,
        pool: &SharedFaBlockPool,
        key_len: usize,
    ) -> Option<PagedAttentionView> {
        if !self.slab_storage
            || !pool.native_attention_enabled()
            || key_len == 0
            || key_len > self.capacity_tokens()
        {
            return None;
        }
        let needed = key_len.div_ceil(self.block_size);
        let slab = pool.single_slab_snapshot(self.layer_idx, &self.block_ids[..needed])?;
        if slab.n_kv_heads != self.n_kv_heads
            || slab.head_dim != self.head_dim
            || slab.block_size != self.block_size
            || slab.dtype != self.dtype
        {
            return None;
        }
        let block_ids = slab.local_rows;
        let block_table = MlxArray::from_raw_data(
            block_ids.as_ptr().cast(),
            block_ids.len().saturating_mul(std::mem::size_of::<u32>()),
            &[block_ids.len() as i32],
            MlxDtype::Uint32,
        );
        Some(PagedAttentionView {
            k_slab: slab.k,
            v_slab: slab.v,
            block_table,
            block_ids,
            key_len,
            block_size: self.block_size,
            n_kv_heads: self.n_kv_heads as usize,
            head_dim: self.head_dim as usize,
        })
    }

    fn materialize(
        &self,
        pool: &SharedFaBlockPool,
        token_start: usize,
        token_end: usize,
    ) -> (MlxArray, MlxArray) {
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

        if self.slab_storage {
            return pool
                .gather_slab_tokens(self.layer_idx, &self.block_ids, token_start, token_end)
                .expect("shared paged FA layer requires initialized slab rows");
        }
        let mut k_pieces: Vec<MlxArray> = Vec::new();
        let mut v_pieces: Vec<MlxArray> = Vec::new();
        let mut t = token_start;
        while t < token_end {
            let block_idx = t / self.block_size;
            let offset = t % self.block_size;
            let take = (token_end - t).min(self.block_size - offset);
            let start = [0, 0, offset as i32, 0];
            let stop = [1, self.n_kv_heads, (offset + take) as i32, self.head_dim];
            let strides = [1i32, 1, 1, 1];
            let (k_source, v_source) = (&self.k_blocks[block_idx], &self.v_blocks[block_idx]);
            k_pieces.push(slice(k_source, &start, &stop, &strides, None));
            v_pieces.push(slice(v_source, &start, &stop, &strides, None));
            t += take;
        }
        if k_pieces.len() == 1 {
            return (k_pieces.remove(0), v_pieces.remove(0));
        }
        let k_refs: Vec<&MlxArray> = k_pieces.iter().collect();
        let v_refs: Vec<&MlxArray> = v_pieces.iter().collect();
        (concatenate(&k_refs, 2, None), concatenate(&v_refs, 2, None))
    }

    fn ensure_capacity(
        &mut self,
        pool: &SharedFaBlockPool,
        tokens: usize,
        growth_count: &mut u64,
    ) -> Result<(), FaBlockPoolError> {
        let needed = if tokens == 0 {
            0
        } else {
            tokens.div_ceil(self.block_size)
        };
        let missing = needed.saturating_sub(self.block_ids.len());
        if missing > 0 {
            let ids = pool.allocate(missing as u32)?;
            self.block_ids.extend(ids);
            if !self.slab_storage {
                let shape = [1i32, self.n_kv_heads, self.block_size as i32, self.head_dim];
                for _ in 0..missing {
                    self.k_blocks.push(zeros(&shape, self.dtype, None));
                    self.v_blocks.push(zeros(&shape, self.dtype, None));
                }
            }
            *growth_count = growth_count.saturating_add(missing as u64);
            self.clear_views();
        }
        if self.slab_storage && !self.block_ids.is_empty() {
            pool.ensure_layer_slab_storage(
                self.layer_idx,
                self.n_kv_heads,
                self.head_dim,
                self.dtype,
                &self.block_ids,
            )?;
        }
        Ok(())
    }

    fn free_blocks_beyond(&mut self, pool: &SharedFaBlockPool, keep_tokens: usize) {
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

    /// Move this view off every shared block that the pending write touches.
    /// Tensor handles are still clones of the old values until `write_tokens`
    /// applies functional updates. Pool ownership preparation is transactional
    /// across every intersecting block; on slab-backed pools the block content
    /// copy shares the same critical section, so a concurrent snapshot
    /// free/evict can never release a source row before it is copied.
    fn prepare_write(
        &mut self,
        pool: &SharedFaBlockPool,
        write_start: usize,
        write_end: usize,
    ) -> Result<u64, FaBlockPoolError> {
        if write_end <= write_start {
            return Ok(0);
        }
        let first = write_start / self.block_size;
        let last = (write_end - 1) / self.block_size;
        let last = last.min(self.block_ids.len().saturating_sub(1));
        let old_ids = self.block_ids[first..=last].to_vec();
        // The block-table swap below commits only when the ownership move —
        // and, for slab storage, the content copy — succeeded; a failed or
        // panicking copy leaves this view on its original block ids.
        let prepared = if self.slab_storage {
            pool.make_unique_many_with_slab_copy(self.layer_idx, &old_ids)?
        } else {
            pool.make_unique_many(&old_ids)?
        };
        let mut copies = 0u64;
        for (offset, (id, copied)) in prepared.into_iter().enumerate() {
            self.block_ids[first + offset] = id;
            copies = copies.saturating_add(u64::from(copied));
        }
        Ok(copies)
    }

    fn write_tokens(
        &mut self,
        pool: &SharedFaBlockPool,
        write_start: usize,
        new_k: &MlxArray,
        new_v: &MlxArray,
    ) -> Result<(), FaBlockPoolError> {
        let new_tokens = new_k.shape()[2] as usize;
        let write_end = write_start + new_tokens;
        assert!(
            write_end <= self.capacity_tokens(),
            "paged write past capacity"
        );
        if self.slab_storage {
            // Drop the previous lazy gather before moving the slab handle so
            // MLX can donate the destination buffer on the next update.
            self.clear_views();
            pool.write_slab_tokens(self.layer_idx, &self.block_ids, write_start, new_k, new_v)?;
            return Ok(());
        }
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
        Ok(())
    }
}

/// FA layer storage: contiguous production path, private paged blocks, or
/// quantized contiguous (Phase 3b).
#[derive(Clone)]
enum FaLayerStorage {
    Contiguous(LayerKV),
    Paged(PagedFaLayer),
    Quantized(QuantizedLayerKV),
}

impl FaLayerStorage {
    fn rotating_window(&self) -> Option<usize> {
        match self {
            Self::Contiguous(lkv) => lkv.rotating_window,
            Self::Paged(_) | Self::Quantized(_) => None,
        }
    }

    fn n_kv_heads(&self) -> i32 {
        match self {
            Self::Contiguous(lkv) => lkv.n_kv_heads,
            Self::Paged(p) => p.n_kv_heads,
            Self::Quantized(q) => q.n_kv_heads,
        }
    }

    fn head_dim(&self) -> i32 {
        match self {
            Self::Contiguous(lkv) => lkv.head_dim,
            Self::Paged(p) => p.head_dim,
            Self::Quantized(q) => q.head_dim,
        }
    }

    fn dtype(&self) -> MlxDtype {
        match self {
            Self::Contiguous(lkv) => lkv.dtype,
            Self::Paged(p) => p.dtype,
            Self::Quantized(q) => q.dtype,
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::Contiguous(lkv) => lkv.capacity,
            Self::Paged(p) => p.capacity_tokens(),
            Self::Quantized(q) => q.capacity,
        }
    }

    /// Physical bytes held per token (K + V). Dense storages report the
    /// element-count-based dense size; quantized storage reports packed +
    /// scales + biases bytes.
    fn bytes_per_token(&self) -> u64 {
        match self {
            Self::Quantized(q) => q.bytes_per_token(),
            Self::Contiguous(_) | Self::Paged(_) => {
                let elements_per_token =
                    (self.n_kv_heads() as u64).saturating_mul(self.head_dim() as u64);
                elements_per_token
                    .saturating_mul(self.dtype().size_bytes() as u64)
                    .saturating_mul(2)
            }
        }
    }

    fn clear_views(&mut self) {
        match self {
            Self::Contiguous(lkv) => {
                lkv.last_k_view = None;
                lkv.last_v_view = None;
            }
            Self::Paged(p) => p.clear_views(),
            Self::Quantized(q) => q.clear_views(),
        }
    }

    fn as_contiguous_mut(&mut self) -> Option<&mut LayerKV> {
        match self {
            Self::Contiguous(lkv) => Some(lkv),
            Self::Paged(_) | Self::Quantized(_) => None,
        }
    }

    fn as_contiguous(&self) -> Option<&LayerKV> {
        match self {
            Self::Contiguous(lkv) => Some(lkv),
            Self::Paged(_) | Self::Quantized(_) => None,
        }
    }

    fn as_quantized(&self) -> Option<&QuantizedLayerKV> {
        match self {
            Self::Quantized(q) => Some(q),
            Self::Contiguous(_) | Self::Paged(_) => None,
        }
    }

    fn as_quantized_mut(&mut self) -> Option<&mut QuantizedLayerKV> {
        match self {
            Self::Quantized(q) => Some(q),
            Self::Contiguous(_) | Self::Paged(_) => None,
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

/// DeepSeek V4 (Flash) per-layer attention cache.
///
/// The raw per-token latent K state: each token contributes ONE vector of
/// `head_dim` (=512) floats from the fused `wkv` projection (single latent
/// head; K doubles as V — there is NO V cache). Compress layers (ratio
/// 4/128) additionally carry Phase-3 compressor pipelines alongside the raw
/// buffer: the main CSA/HCA compressor and, for CSA layers, the LID indexer
/// compressor. Those stay separate fields so the raw append signature is
/// unchanged.
///
/// Note: the F3 disk wire format (`serialize_to_bytes`) does not yet encode
/// this layer kind; V4 prefix-cache serialization lands with the forward
/// wiring phase.
#[derive(Clone)]
struct DeepseekV4LayerCache {
    /// `[1, 1, capacity, head_dim]` — one latent K vector per token.
    k_latent: MlxArray,
    head_dim: i32,
    capacity: usize,
    dtype: MlxDtype,
    /// Phase-3 main compressor pipeline (CSA/HCA layers only).
    comp: Option<DeepseekV4CompressorCache>,
    /// Phase-3 LID indexer compressor pipeline (CSA ratio-4 layers only).
    indexer_comp: Option<DeepseekV4CompressorCache>,
}

/// DeepSeek V4 (Flash) Phase-3 compressor pipeline state (one per main
/// CSA/HCA compressor, plus one for the LID indexer compressor on CSA
/// layers).
///
/// Holds the F32 per-token state buffer (`kv_state`/`score_state` in
/// llama.cpp's `llama_dsv4_comp_state`, kept F32 there too) covering the
/// current partial block window — and, for the overlap (CSA) compressor,
/// the previous block's rows, so a completing block always finds its full
/// 2·ratio read window buffered — plus the committed compressed-K rows in
/// chunked storage (one row per completed block, RoPE'd at the block-start
/// position before append, stored in the raw-K dtype so the CSA/HCA concat
/// along the key axis needs no cast).
///
/// Bookkeeping invariant between forwards: the buffer spans positions
/// `[state_base, state_base + rows)`, `state_base + rows == token_offset`,
/// and `committed == token_offset / ratio` right after a forward whose end
/// crossed a block boundary.
#[derive(Clone)]
struct DeepseekV4CompressorCache {
    /// Buffered F32 per-token KV states `[rows, state_width]`
    /// (`coff*head_dim` main / `2*index_head_dim` indexer).
    state_kv: Option<MlxArray>,
    /// Buffered F32 per-token score states `[rows, state_width]`, ape added.
    state_score: Option<MlxArray>,
    /// Absolute token position of buffered row 0.
    state_base: usize,
    /// Compressed rows committed so far (== completed blocks).
    committed: usize,
    /// Committed compressed-K rows `[1, 1, capacity, row_dim]` (raw-K dtype).
    comp_k: Option<MlxArray>,
    capacity: usize,
    /// Compress ratio of this pipeline (4 = CSA overlap, 128 = HCA) — kept
    /// for block-boundary bookkeeping on `trim_to`.
    ratio: usize,
    /// Whether this is the overlap (coff=2) compressor: after a block
    /// completes, its rows stay buffered as the next block's previous half.
    overlap: bool,
}

#[derive(Clone, Default)]
struct LinearLayerState {
    /// Qwen3.5 gated-delta conv tail: `[1, conv_kernel - 1, conv_dim]`.
    conv_state: Option<MlxArray>,
    /// Qwen3.5 gated-delta recurrent state: `[1, value_heads, value_dim, key_dim]`.
    recurrent_state: Option<MlxArray>,
    /// Transient verifier checkpoint after a committed prefix.
    prefix_conv_state: Option<MlxArray>,
    /// Transient verifier checkpoint after a committed prefix.
    prefix_recurrent_state: Option<MlxArray>,
    /// Projected Q/K/V inputs retained only for an in-flight MTP verify.
    mtp_qkv: Option<MlxArray>,
    /// Projected gated-delta `a` input retained only for an in-flight verify.
    mtp_a: Option<MlxArray>,
    /// Projected gated-delta `b` input retained only for an in-flight verify.
    mtp_b: Option<MlxArray>,
    /// Normalized verifier Q rows retained for accepted-prefix replay.
    mtp_q: Option<MlxArray>,
    /// Normalized verifier K rows retained for accepted-prefix replay.
    mtp_k: Option<MlxArray>,
    /// Convolved verifier V rows retained for accepted-prefix replay.
    mtp_v: Option<MlxArray>,
    /// Compact float32 gated-delta replay tape for a whole-model verifier.
    mtp_tape: Option<MlxArray>,
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
    // SAFETY: per the function contract above — `payload` came from
    // `Box::into_raw(Box::new(Vec<u8>))` and is recovered exactly once.
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
    /// Restored payload carries a different layer count than the model
    /// it is being adopted for.
    LayerCountMismatch { expected: usize, actual: usize },
    /// Restored logical `seq_len` disagrees with the token count the
    /// caller is adopting the snapshot for.
    TokenCountMismatch { expected: usize, actual: usize },
    /// `seq_len > 0` but every layer deserialized as `EMPTY` — the
    /// payload claims a prefix it holds no state for.
    EmptySnapshot,
    /// A layer the model requires to carry linear-attention state is
    /// missing its conv or recurrent tensor.
    IncompleteLinearLayer(usize),
    /// Flash Next request state is inconsistent with its prefix or model.
    FlashNextState(String),
}

impl std::fmt::Display for MlxKVCacheSerializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FlashNextState(message) => write!(f, "Flash Next cache: {message}"),
            Self::BadMagic => write!(f, "kv-cache payload has wrong magic"),
            Self::UnsupportedVersion(v) => write!(f, "kv-cache payload version {v} unsupported"),
            Self::UnexpectedEof => write!(f, "kv-cache payload truncated"),
            Self::UnknownDtype(t) => write!(f, "kv-cache payload has unknown dtype tag {t}"),
            Self::UnknownLayerKind(t) => write!(f, "kv-cache payload has unknown layer kind {t}"),
            Self::BadShape(n) => write!(f, "kv-cache payload has bad tensor rank {n}"),
            Self::LayerCountMismatch { expected, actual } => write!(
                f,
                "kv-cache payload has {actual} layers but the model has {expected}"
            ),
            Self::TokenCountMismatch { expected, actual } => write!(
                f,
                "kv-cache payload holds {actual} tokens but {expected} were requested"
            ),
            Self::EmptySnapshot => {
                write!(f, "kv-cache payload claims tokens but every layer is empty")
            }
            Self::IncompleteLinearLayer(idx) => write!(
                f,
                "kv-cache payload is missing linear conv/recurrent state for layer {idx}"
            ),
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

/// Borrowed view of a DeepSeek V4 layer's cached latent K state. Returned by
/// [`MlxKVCache::deepseek_v4_layer_state`].
pub struct DeepseekV4LayerStateView<'a> {
    /// The full backing buffer for the latent K cache; shape
    /// `[1, 1, capacity, head_dim]`. Valid region is `[0..seq_len]`.
    pub k_latent: &'a MlxArray,
    /// Inner dim of `k_latent` — equal to the model's `head_dim` (512).
    pub head_dim: i32,
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
    pub(crate) qwen4_exp: Option<crate::model::qwen4_exp::Qwen4ExpState>,
    layers: Vec<Option<FaLayerStorage>>,
    glm_mla_layers: Vec<Option<GlmMlaLayerCache>>,
    deepseek_v4_layers: Vec<Option<DeepseekV4LayerCache>>,
    linear_layers: Vec<LinearLayerState>,
    /// Number of tokens after which a speculative verifier should capture each
    /// linear-attention layer's transient state. Checkpoints are intentionally
    /// excluded from durable snapshots and ordinary cache clones.
    linear_prefix_capture_after: Option<usize>,
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
    /// Signed multimodal position compression used by unified Qwen models.
    ///
    /// A visual run may occupy hundreds of KV slots while advancing MRoPE by
    /// only the largest temporal/spatial axis, so this is commonly negative.
    /// It is kept separate from `rope_offset`, whose existing callers require
    /// an unsigned physical-vs-logical cache offset.
    mrope_position_delta: i32,
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
    /// Synchronized FA block pool when paged mode is active (flag or explicit
    /// constructor). A runner may share this handle across request caches;
    /// private mode creates a handle visible to only this cache and its clones.
    fa_pool: Option<SharedFaBlockPool>,
    /// Cumulative µs spent in paged FA materialize (SDPA / serialize views).
    paged_materialize_us: u64,
    /// Count of paged→contiguous failovers when the owning pool is exhausted.
    paged_pool_exhaustion_fallbacks: u64,
    /// Cache-local count of block-level CoW replacements.
    paged_cow_copies: u64,
    paged_attention_calls: u64,
    paged_attention_fallbacks: u64,
    /// Per-layer KV quantization table from the model manifest (Phase 3b).
    /// `None` = full precision for every layer. Layers whose spec is `Some`
    /// store packed K/V and never take the paged route. Injected via
    /// [`Self::set_kv_quant_table`]; the `AX_KV_QUANT=0` kill-switch is
    /// honored there.
    kv_quant: Option<Vec<Option<KvQuantSpec>>>,
    /// Sticky: set when a production `hard_cap` pool exhausted. The layer still
    /// demotes to contiguous (correct data — proven token-exact by
    /// `fa_paged_pool_exhaustion_demotion_matches_contiguous_oracle`), but
    /// the caller must fail this request instead of returning a token
    /// computed past the operator's memory bound.
    hard_cap_exhausted: bool,
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
    pub(crate) fn needs_mask(&self, seq: usize) -> bool {
        self.capacity > self.window || seq > 1
    }
}

impl Clone for MlxKVCache {
    fn clone(&self) -> Self {
        let fa_pool = self.fa_pool.clone();
        if let Some(pool) = fa_pool.as_ref() {
            let ids: Vec<PhysicalBlockId> = self
                .layers
                .iter()
                .filter_map(Option::as_ref)
                .filter_map(|layer| match layer {
                    FaLayerStorage::Paged(paged) => Some(paged.block_ids.as_slice()),
                    FaLayerStorage::Contiguous(_) | FaLayerStorage::Quantized(_) => None,
                })
                .flatten()
                .copied()
                .collect();
            pool.retain(&ids)
                .expect("paged cache clone must retain valid physical block IDs");
        }
        let mut linear_layers = self.linear_layers.clone();
        for state in &mut linear_layers {
            state.prefix_conv_state = None;
            state.prefix_recurrent_state = None;
            state.mtp_qkv = None;
            state.mtp_a = None;
            state.mtp_b = None;
            state.mtp_q = None;
            state.mtp_k = None;
            state.mtp_v = None;
            state.mtp_tape = None;
        }
        Self {
            qwen4_exp: self.qwen4_exp.clone(),
            layers: self.layers.clone(),
            glm_mla_layers: self.glm_mla_layers.clone(),
            deepseek_v4_layers: self.deepseek_v4_layers.clone(),
            linear_layers,
            linear_prefix_capture_after: None,
            seq_len: self.seq_len,
            rope_offset: self.rope_offset,
            mrope_position_delta: self.mrope_position_delta,
            growth_count: self.growth_count,
            use_rotating_sliding_decode: self.use_rotating_sliding_decode,
            rotating_slack: self.rotating_slack,
            fa_pool,
            paged_materialize_us: self.paged_materialize_us,
            paged_pool_exhaustion_fallbacks: self.paged_pool_exhaustion_fallbacks,
            // CoW events are owned by the new view after cloning; do not
            // re-report the source view's historical counter.
            paged_cow_copies: 0,
            paged_attention_calls: 0,
            paged_attention_fallbacks: 0,
            hard_cap_exhausted: self.hard_cap_exhausted,
            kv_quant: self.kv_quant.clone(),
        }
    }
}

impl Drop for MlxKVCache {
    fn drop(&mut self) {
        let Some(pool) = self.fa_pool.as_ref() else {
            return;
        };
        for layer in self.layers.iter_mut().flatten() {
            let FaLayerStorage::Paged(paged) = layer else {
                continue;
            };
            let ids = std::mem::take(&mut paged.block_ids);
            if let Err(error) = pool.free(&ids) {
                tracing::error!(
                    target: "ax_engine_mlx::kv_pool",
                    %error,
                    "failed to release paged KV blocks during cache drop",
                );
            }
        }
    }
}

/// Largest per-layer count a snapshot header may declare (bounded before
/// any allocation; real models sit far below this).
const MAX_WIRE_LAYER_COUNT: usize = 4096;

impl MlxKVCache {
    // ── Wire-format constants for `serialize_to_bytes` / `try_deserialize_from_bytes` ──
    // The format is private to this module; see the F3 disk-cache PRD
    // (`MLX-DISK-PREFIX-CACHE-PRD-2026-05-14.md`) §3.3 / §4 for the rationale.
    /// File magic for the AX disk-format payload section. Distinct from
    /// `AXKV` (used by the future disk wrapper for outer file framing) so
    /// a partial / nested payload cannot be mistaken for a complete file.
    const SERIALIZE_MAGIC: &'static [u8; 4] = b"AXKB";
    const SERIALIZE_VERSION: u32 = 4;

    /// Magic for the standalone Flash Next MTP draft-cursor payload
    /// ("AX MTP Cursor"). Deliberately distinct from [`Self::SERIALIZE_MAGIC`]
    /// (`AXKB`) so a cursor payload can never be misread as a trunk KV-cache
    /// blob (or vice versa): a truncated or malformed cursor fails at the
    /// header check instead of being parsed as a trunk snapshot.
    ///
    /// Consumed by the draft-cursor codec and its tests; the codec is wired
    /// into the prefix-cache store path by a follow-up change, so it is dead
    /// in non-test builds for now.
    #[cfg_attr(not(test), allow(dead_code))]
    const MTP_CURSOR_MAGIC: &'static [u8; 4] = b"AXMC";
    #[cfg_attr(not(test), allow(dead_code))]
    const MTP_CURSOR_VERSION: u32 = 1;

    /// Private KV wire-format version, exposed for the durable prefix
    /// cache's canonical key (schema v3 commits to the payload version so a
    /// format bump cleanly invalidates older disk entries).
    pub(crate) const fn serialize_version() -> u32 {
        Self::SERIALIZE_VERSION
    }
    const LAYER_KIND_EMPTY: u8 = 0;
    const LAYER_KIND_FA: u8 = 1;
    const LAYER_KIND_MLA: u8 = 2;
    const LAYER_KIND_LINEAR: u8 = 3;
    const LAYER_KIND_FLASH_NEXT: u8 = 4;
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
    pub(crate) fn new_contiguous(num_layers: usize) -> Self {
        Self::new_with_fa_pool(num_layers, None)
    }

    /// FA private block-pool path (PR4). Pure FA appends use paged storage and
    /// materialize dense K/V for SDPA; sliding/rotating layers stay contiguous.
    pub(crate) fn new_with_fa_block_pool(num_layers: usize, config: FaBlockPoolConfig) -> Self {
        let fa_pool =
            SharedFaBlockPool::new(config).expect("FA block pool config must be non-zero");
        Self::new_with_shared_fa_block_pool(num_layers, fa_pool)
    }

    /// Build a paged cache backed by a runner-owned synchronized FA pool.
    pub(crate) fn new_with_shared_fa_block_pool(
        num_layers: usize,
        fa_pool: SharedFaBlockPool,
    ) -> Self {
        Self::new_with_fa_pool(num_layers, Some(fa_pool))
    }

    fn new_with_fa_pool(num_layers: usize, fa_pool: Option<SharedFaBlockPool>) -> Self {
        Self {
            qwen4_exp: None,
            layers: (0..num_layers).map(|_| None).collect(),
            glm_mla_layers: (0..num_layers).map(|_| None).collect(),
            deepseek_v4_layers: (0..num_layers).map(|_| None).collect(),
            linear_layers: (0..num_layers)
                .map(|_| LinearLayerState::default())
                .collect(),
            linear_prefix_capture_after: None,
            seq_len: 0,
            rope_offset: 0,
            mrope_position_delta: 0,
            growth_count: 0,
            use_rotating_sliding_decode: false,
            rotating_slack: 0,
            fa_pool,
            paged_materialize_us: 0,
            paged_pool_exhaustion_fallbacks: 0,
            paged_cow_copies: 0,
            paged_attention_calls: 0,
            paged_attention_fallbacks: 0,
            hard_cap_exhausted: false,
            kv_quant: None,
        }
    }

    #[cfg(test)]
    /// Whether this cache has an FA block pool (private or runner-shared).
    pub(crate) fn fa_block_pool_enabled(&self) -> bool {
        self.fa_pool.is_some()
    }

    /// Inject the per-layer KV quantization table from the model manifest
    /// (Phase 3b). Idempotent; safe to re-apply after a prefix-restore swaps
    /// the cache (restored snapshots are dense and re-quantize on first
    /// append).
    ///
    /// - `table.len() != num_layers` → warn and ignore.
    /// - `AX_KV_QUANT=0` → behave as if every spec were `None` (info log once
    ///   per process).
    /// - Specs outside the validated set (bits 4/6/8, group sizes 32/64/128)
    ///   or targeting a layer that has already become a rotating /
    ///   protected-prefix ring are rejected per layer (warn + full precision).
    pub(crate) fn set_kv_quant_table(&mut self, table: Vec<Option<KvQuantSpec>>) {
        if table.len() != self.layers.len() {
            tracing::warn!(
                target: "ax_engine_mlx::kv_cache",
                table_len = table.len(),
                num_layers = self.layers.len(),
                "KV quant table length does not match layer count; ignoring table",
            );
            return;
        }
        if kv_quant_env_disabled() {
            static LOGGED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if table.iter().any(Option::is_some)
                && !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                tracing::info!(
                    target: "ax_engine_mlx::kv_cache",
                    "AX_KV_QUANT=0: KV-cache quantization disabled; ignoring quant table",
                );
            }
            self.kv_quant = None;
            return;
        }
        let mut table = table;
        for (idx, slot) in table.iter_mut().enumerate() {
            let Some(spec) = *slot else {
                continue;
            };
            let valid = matches!(spec.bits, 4 | 6 | 8) && matches!(spec.group_size, 32 | 64 | 128);
            if !valid {
                tracing::warn!(
                    target: "ax_engine_mlx::kv_cache",
                    layer = idx,
                    bits = spec.bits,
                    group_size = spec.group_size,
                    "unsupported KV quant spec; storing layer at full precision",
                );
                *slot = None;
                continue;
            }
            // Quantized rings are out of scope for this phase: a layer that
            // has already converted to a rotating or protected-prefix ring
            // keeps its dense slot-ordered storage.
            let already_ring = self.layers[idx]
                .as_ref()
                .and_then(FaLayerStorage::as_contiguous)
                .is_some_and(|lkv| {
                    lkv.rotating_window.is_some() || lkv.protected_prefix_ring.is_some()
                });
            if already_ring {
                tracing::warn!(
                    target: "ax_engine_mlx::kv_cache",
                    layer = idx,
                    "KV quant spec rejected for ring layer; storing full precision",
                );
                *slot = None;
            }
        }
        self.kv_quant = if table.iter().any(Option::is_some) {
            Some(table)
        } else {
            None
        };
        // A table (re-)injected after prefix-restore adoption can find a
        // spec'd layer already holding paged storage (native L1 adoption).
        // Quantized layers never take the paged route, so demote those layers
        // back to contiguous now; they quantize on their next append.
        for idx in 0..self.layers.len() {
            if self.layer_kv_quant(idx).is_some()
                && matches!(self.layers[idx], Some(FaLayerStorage::Paged(_)))
            {
                tracing::debug!(
                    target: "ax_engine_mlx::kv_cache",
                    layer = idx,
                    "demoting adopted paged layer to contiguous for KV quantization",
                );
                self.demote_paged_layer_to_contiguous(idx, self.seq_len);
            }
        }
    }

    /// The effective quant spec for `layer`, if quantization is active.
    fn layer_kv_quant(&self, layer: usize) -> Option<KvQuantSpec> {
        self.kv_quant
            .as_ref()
            .and_then(|table| table.get(layer).copied())
            .flatten()
    }

    #[cfg(test)]
    /// Test/introspection hook: whether `layer` currently holds quantized
    /// storage.
    pub(crate) fn layer_is_quantized(&self, layer: usize) -> bool {
        self.layers
            .get(layer)
            .and_then(Option::as_ref)
            .is_some_and(|fa| fa.as_quantized().is_some())
    }

    #[cfg(test)]
    /// Whether any layer holds quantized storage (telemetry/tests).
    pub(crate) fn has_quantized_layers(&self) -> bool {
        self.layers
            .iter()
            .flatten()
            .any(|fa| fa.as_quantized().is_some())
    }

    /// Set once a production `hard_cap` pool has exhausted. The caller must
    /// fail the owning request instead of treating this forward as
    /// successful — see `MlxRunner::run_item`.
    pub(crate) fn hard_cap_exhausted(&self) -> bool {
        self.hard_cap_exhausted
    }

    #[cfg(test)]
    /// Blocks available in the owning FA pool, if paged mode is active.
    pub(crate) fn fa_block_pool_available(&self) -> Option<u32> {
        self.fa_pool
            .as_ref()
            .map(|pool| pool.snapshot().available_blocks)
    }

    #[cfg(test)]
    pub(crate) fn shares_fa_block_pool_with(&self, other: &Self) -> bool {
        match (&self.fa_pool, &other.fa_pool) {
            (Some(left), Some(right)) => left.same_pool(right),
            _ => false,
        }
    }

    pub(crate) fn uses_fa_block_pool(&self, pool: &SharedFaBlockPool) -> bool {
        self.fa_pool
            .as_ref()
            .is_some_and(|owned| owned.same_pool(pool))
    }

    /// Whether this cache is safe to retain in the runner-local native L1.
    ///
    /// Model-level eligibility is necessary but not sufficient: compiled or
    /// speculative paths can replace a paged layer with dense storage at
    /// runtime. Native L1 only promises physical-page adoption, so every
    /// layer must still be a structurally valid paged FA layer.
    pub(crate) fn is_native_fa_shareable(&self) -> bool {
        let slab_pool = self
            .fa_pool
            .as_ref()
            .is_some_and(SharedFaBlockPool::slab_storage_enabled);
        self.fa_pool.is_some()
            && self.seq_len > 0
            && !self.layers.is_empty()
            && self.layers.iter().all(|layer| {
                matches!(
                    layer,
                    Some(FaLayerStorage::Paged(paged))
                        if ((!paged.slab_storage
                            && paged.block_ids.len() == paged.k_blocks.len()
                            && paged.block_ids.len() == paged.v_blocks.len())
                            || (paged.slab_storage
                                && slab_pool
                                && paged.k_blocks.is_empty()
                                && paged.v_blocks.is_empty()))
                            && !paged.block_ids.is_empty()
                            && paged.capacity_tokens() >= self.seq_len
                )
            })
            && self.glm_mla_layers.iter().all(Option::is_none)
            && self.deepseek_v4_layers.iter().all(Option::is_none)
            && self
                .linear_layers
                .iter()
                .all(|state| state.conv_state.is_none() && state.recurrent_state.is_none())
    }

    /// Validate a dense standard-FA snapshot and return the layer-block slots
    /// required to rebuild it in `pool` without changing the wire format.
    pub(crate) fn fa_blocks_required_for_repage(
        &self,
        pool: &SharedFaBlockPool,
    ) -> Result<u32, FaBlockPoolError> {
        if self.seq_len == 0 || self.layers.is_empty() {
            return Err(FaBlockPoolError::InvalidConfig(
                "repage requires a non-empty standard-FA cache",
            ));
        }
        if self.glm_mla_layers.iter().any(Option::is_some)
            || self.deepseek_v4_layers.iter().any(Option::is_some)
            || self
                .linear_layers
                .iter()
                .any(|state| state.conv_state.is_some() || state.recurrent_state.is_some())
        {
            return Err(FaBlockPoolError::InvalidConfig(
                "repage supports standard FA only",
            ));
        }
        // Quantized layers (Phase 3b) are skipped: they stay contiguous —
        // dequantized to dense in the repaged clone — and never claim pool
        // blocks, so they are excluded from the block count.
        let mut dense_layers = 0u64;
        for layer in &self.layers {
            let Some(layer) = layer else {
                return Err(FaBlockPoolError::InvalidConfig(
                    "repage requires every layer to be dense FA",
                ));
            };
            let FaLayerStorage::Contiguous(layer) = layer else {
                if layer.as_quantized().is_some() {
                    continue;
                }
                return Err(FaBlockPoolError::InvalidConfig(
                    "repage requires every layer to be dense FA",
                ));
            };
            let shape = layer.k.shape();
            if layer.rotating_window.is_some()
                || layer.capacity < self.seq_len
                || shape.len() != 4
                || shape != layer.v.shape()
                || shape[0] != 1
                || shape[1] != layer.n_kv_heads
                || shape[2] as usize != layer.capacity
                || shape[3] != layer.head_dim
                || layer.k.dtype() != layer.v.dtype()
            {
                return Err(FaBlockPoolError::InvalidConfig(
                    "repage rejected incompatible FA layer geometry",
                ));
            }
            dense_layers = dense_layers.saturating_add(1);
        }
        if dense_layers == 0 {
            return Err(FaBlockPoolError::InvalidConfig(
                "repage requires at least one dense FA layer",
            ));
        }

        let block_size = pool.config().block_size_tokens as usize;
        let per_layer = self.seq_len.div_ceil(block_size) as u64;
        let total = per_layer.saturating_mul(dense_layers);
        u32::try_from(total)
            .map_err(|_| FaBlockPoolError::InvalidConfig("repage layer-block count exceeds u32"))
    }

    /// Clone a dense serialized standard-FA snapshot into the runner's shared
    /// paged representation. Allocation is one transaction; failure leaves
    /// both the source cache and pool ownership unchanged.
    pub(crate) fn clone_repage_into_shared_fa_pool(
        &self,
        pool: SharedFaBlockPool,
    ) -> Result<Self, FaBlockPoolError> {
        let required = self.fa_blocks_required_for_repage(&pool)?;
        let block_size = pool.config().block_size_tokens as usize;
        let blocks_per_layer = self.seq_len.div_ceil(block_size);
        let mut reservation = FaBlockReservation::new(pool.clone(), required)?;
        let mut paged_layers = Vec::with_capacity(self.layers.len());

        // Dense layers claim pool blocks left-to-right; quantized layers skip
        // the pool entirely and are dequantized to dense contiguous storage in
        // the clone (they re-quantize on the next append via the copied table).
        let mut dense_cursor = 0usize;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let Some(layer) = layer else {
                return Err(FaBlockPoolError::InvalidConfig(
                    "repage layout changed after validation",
                ));
            };
            if let Some(quantized) = layer.as_quantized() {
                tracing::debug!(
                    target: "ax_engine_mlx::kv_cache",
                    layer = layer_idx,
                    "repage skips quantized layer; keeping contiguous dense storage",
                );
                let (k, v) = quantized.dense_view(0, self.seq_len);
                paged_layers.push(Some(FaLayerStorage::Contiguous(LayerKV {
                    k,
                    v,
                    last_k_view: None,
                    last_v_view: None,
                    n_kv_heads: quantized.n_kv_heads,
                    head_dim: quantized.head_dim,
                    capacity: self.seq_len,
                    rotating_window: None,
                    protected_prefix_ring: None,
                    dtype: quantized.dtype,
                })));
                continue;
            }
            let FaLayerStorage::Contiguous(layer) = layer else {
                return Err(FaBlockPoolError::InvalidConfig(
                    "repage layout changed after validation",
                ));
            };
            let id_start = dense_cursor.saturating_mul(blocks_per_layer);
            dense_cursor = dense_cursor.saturating_add(1);
            let ids = reservation.ids[id_start..id_start + blocks_per_layer].to_vec();
            let slab_storage = pool.slab_storage_enabled();
            let mut k_blocks = Vec::with_capacity(if slab_storage { 0 } else { blocks_per_layer });
            let mut v_blocks = Vec::with_capacity(if slab_storage { 0 } else { blocks_per_layer });

            if slab_storage {
                pool.ensure_layer_slab_storage(
                    layer_idx,
                    layer.n_kv_heads,
                    layer.head_dim,
                    layer.dtype,
                    &ids,
                )?;
                // Serialized dense layers retain their growth capacity. Only
                // the logical prefix belongs in the page table; passing the
                // full backing buffer would overrun `blocks_per_layer`.
                let start = [0i32, 0, 0, 0];
                let stop = [1i32, layer.n_kv_heads, self.seq_len as i32, layer.head_dim];
                let strides = [1i32, 1, 1, 1];
                let logical_k = slice(&layer.k, &start, &stop, &strides, None);
                let logical_v = slice(&layer.v, &start, &stop, &strides, None);
                pool.write_slab_tokens(layer_idx, &ids, 0, &logical_k, &logical_v)?;
            }

            for block_idx in 0..if slab_storage { 0 } else { blocks_per_layer } {
                let token_start = block_idx.saturating_mul(block_size);
                let token_count = (self.seq_len - token_start).min(block_size);
                let token_end = token_start + token_count;
                let start = [0i32, 0, token_start as i32, 0];
                let stop = [1i32, layer.n_kv_heads, token_end as i32, layer.head_dim];
                let strides = [1i32, 1, 1, 1];
                let k_segment = slice(&layer.k, &start, &stop, &strides, None);
                let v_segment = slice(&layer.v, &start, &stop, &strides, None);
                if token_count == block_size {
                    k_blocks.push(k_segment);
                    v_blocks.push(v_segment);
                } else {
                    let shape = [1i32, layer.n_kv_heads, block_size as i32, layer.head_dim];
                    let block_start = [0i32, 0, 0, 0];
                    let block_stop = [1i32, layer.n_kv_heads, token_count as i32, layer.head_dim];
                    k_blocks.push(slice_update(
                        &zeros(&shape, layer.dtype, None),
                        &k_segment,
                        &block_start,
                        &block_stop,
                        &strides,
                        None,
                    ));
                    v_blocks.push(slice_update(
                        &zeros(&shape, layer.dtype, None),
                        &v_segment,
                        &block_start,
                        &block_stop,
                        &strides,
                        None,
                    ));
                }
            }

            paged_layers.push(Some(FaLayerStorage::Paged(PagedFaLayer {
                layer_idx,
                n_kv_heads: layer.n_kv_heads,
                head_dim: layer.head_dim,
                dtype: layer.dtype,
                block_size,
                block_ids: ids,
                slab_storage,
                k_blocks,
                v_blocks,
                last_k_view: None,
                last_v_view: None,
            })));
        }

        let mut repaged = Self::new_with_shared_fa_block_pool(self.layers.len(), pool);
        repaged.layers = paged_layers;
        repaged.seq_len = self.seq_len;
        repaged.rope_offset = self.rope_offset;
        repaged.growth_count = self.growth_count;
        repaged.paged_materialize_us = self.paged_materialize_us;
        repaged.paged_pool_exhaustion_fallbacks = self.paged_pool_exhaustion_fallbacks;
        repaged.paged_cow_copies = self.paged_cow_copies;
        repaged.paged_attention_calls = self.paged_attention_calls;
        repaged.paged_attention_fallbacks = self.paged_attention_fallbacks;
        repaged.hard_cap_exhausted = self.hard_cap_exhausted;
        repaged.kv_quant = self.kv_quant.clone();
        // Multimodal RoPE state travels with the tokens: a restored visual
        // prefix decodes at `mrope_decode_position(offset)`, not the physical
        // offset.
        repaged.mrope_position_delta = self.mrope_position_delta;
        reservation.disarm();
        Ok(repaged)
    }

    /// Conservative physical-block demand for appending `new_tokens` to a
    /// pure-FA cache. Callers gate this to layouts where every empty layer will
    /// become paged FA; hybrid layouts intentionally do not use the estimate.
    pub(crate) fn additional_fa_blocks_for_append(&self, new_tokens: usize) -> Option<u32> {
        let pool = self.fa_pool.as_ref()?;
        if new_tokens == 0 {
            return Some(0);
        }
        let block_size = pool.config().block_size_tokens as usize;
        let write_start = self.seq_len;
        let write_end = write_start.saturating_add(new_tokens);
        let needed_blocks = write_end.div_ceil(block_size);
        let mut additional = 0u64;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            match layer {
                // Quantized layers never take the paged route, so an empty
                // layer with a quant spec will not claim pool blocks.
                None if self.layer_kv_quant(layer_idx).is_some() => {}
                None => {
                    additional = additional.saturating_add(needed_blocks as u64);
                }
                Some(FaLayerStorage::Contiguous(_)) | Some(FaLayerStorage::Quantized(_)) => {}
                Some(FaLayerStorage::Paged(paged)) => {
                    additional = additional
                        .saturating_add(needed_blocks.saturating_sub(paged.block_ids.len()) as u64);
                    if !paged.block_ids.is_empty() {
                        let first = write_start / block_size;
                        let last = (write_end - 1) / block_size;
                        if first < paged.block_ids.len() {
                            for block_idx in
                                first..=last.min(paged.block_ids.len().saturating_sub(1))
                            {
                                if pool
                                    .ref_count(paged.block_ids[block_idx])
                                    .is_ok_and(|ref_count| ref_count > 1)
                                {
                                    additional = additional.saturating_add(1);
                                }
                            }
                        }
                    }
                }
            }
        }

        Some(additional.min(u64::from(u32::MAX)) as u32)
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
        let next = self
            .seq_len
            .checked_add(n)
            .expect("KV token count overflow");
        if let Some(state) = &self.qwen4_exp {
            assert_eq!(
                state.position(),
                next,
                "Flash Next cache advancement must match staged state"
            );
        }
        self.seq_len = next;
    }

    /// Set the logical boundary to an absolute position. Prefer
    /// [`Self::advance`] after forwards; this is for seeding a cache at a
    /// known position (prefill restore, warmup, tests). For rollback use
    /// [`Self::trim_to`], which also validates ring residency.
    pub(crate) fn set_seq_len(&mut self, n: usize) {
        if let Some(state) = &self.qwen4_exp {
            assert_eq!(
                state.position(),
                n,
                "Flash Next cache boundary must match recurrent state"
            );
        }
        self.seq_len = n;
    }

    /// Install the signed position compression computed by a visual prefill.
    pub(crate) fn set_mrope_position_delta(&mut self, delta: i32) {
        self.mrope_position_delta = delta;
    }

    pub(crate) fn mrope_position_delta(&self) -> i32 {
        self.mrope_position_delta
    }

    /// Convert a physical token offset to the shared T/H/W position used for
    /// post-prefill decode, clamping only malformed negative positions.
    pub(crate) fn mrope_decode_position(&self, token_offset: usize) -> usize {
        let position = (token_offset as i128) + i128::from(self.mrope_position_delta);
        usize::try_from(position.max(0)).unwrap_or(usize::MAX)
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

    pub(crate) fn rotating_sliding_slack(&self) -> usize {
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
    pub(crate) fn sliding_ring_layout(
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
        // Do not change an ordered cache into a capacity-wide ring halfway
        // through a multi-token verify. Keep one ordered SDPA geometry for the
        // crossing chunk, then rotate on the next forward. A cold append at
        // position zero may still initialize a ring directly.
        if seq > 1 && self.seq_len > 0 && self.seq_len < window {
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

    /// Whether any sliding-attention layer is physically slot-ordered.
    ///
    /// A rotated ring is valid for bounded decode appends but is not a
    /// portable prompt-prefix snapshot: a future request may warm-extend it
    /// with a multi-token prefill or select a rollback-capable route. Prefix
    /// stores/restores use this predicate to fail closed and recompute an
    /// ordered cache instead.
    pub(crate) fn has_rotated_sliding_layers(&self) -> bool {
        self.layers.iter().flatten().any(|layer| {
            layer.rotating_window().is_some()
                || layer
                    .as_contiguous()
                    .is_some_and(|lkv| lkv.protected_prefix_ring.is_some())
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

    pub(crate) fn serialize_tensor(out: &mut Vec<u8>, arr: &MlxArray) {
        // `eval` alone does not make a slice/transpose row-contiguous. Paged
        // materialization can return an exact-length view, bypassing the
        // trimming branch above, so normalize every tensor before `data_raw`.
        // MLX treats `contiguous` as a cheap identity for already-contiguous
        // arrays and as an owned materialization for strided views.
        let arr = contiguous(arr, None);
        eval(&[&arr]);
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
    pub(crate) fn read_tensor_from_reader(
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
        // Exact match only: a larger declared count would allocate attacker-
        // controlled sizes (e.g. 2^60) before the checksum can reject the
        // payload; a smaller count is truncated garbage.
        if byte_count != required_bytes {
            return Err(MlxKVCacheSerializeError::BadShape(ndim));
        }
        Ok(())
    }

    fn mlx_array_from_owned_bytes(
        owned: Vec<u8>,
        shape: &[i32],
        dtype: MlxDtype,
    ) -> Result<MlxArray, MlxKVCacheSerializeError> {
        // `MlxArray::from_raw_data` COPIES at construction (mlx-sys io.rs
        // documents the copy-on-create C entry). `from_managed_data` +
        // `vec_payload_drop` is used here to hand the buffer's ownership to
        // MLX instead — avoiding a second full-payload copy on restore —
        // and keeps correctness independent of MLX's copy behavior.
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
    /// Whether this cache holds state the wire format cannot carry (DeepSeek
    /// V4 compressor layers): storing it would produce a snapshot that
    /// claims tokens but restores as empty.
    pub(crate) fn has_unserializable_layers(&self) -> bool {
        self.deepseek_v4_layers.iter().any(Option::is_some)
    }

    pub(crate) fn serialize_to_bytes(&self) -> Vec<u8> {
        // Fail closed before writing anything: DeepSeek V4 compressor layers
        // cannot round-trip, and this infallible signature would otherwise
        // encode them as EMPTY layers that claim tokens but restore empty.
        // Store sites gate on `has_unserializable_layers`; this is the
        // backstop for any future caller that forgets the gate.
        assert!(
            !self.has_unserializable_layers(),
            "cannot serialize a cache with DeepSeek V4 compressor layers"
        );
        let mut out = Vec::new();
        out.extend_from_slice(Self::SERIALIZE_MAGIC);
        out.extend_from_slice(&Self::SERIALIZE_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.seq_len as u64).to_le_bytes());
        out.extend_from_slice(&self.growth_count.to_le_bytes());
        out.extend_from_slice(&(self.rope_offset as u64).to_le_bytes());
        let layer_count = self.layers.len() as u32;
        out.extend_from_slice(&layer_count.to_le_bytes());
        // Version 4 originally reserved this word as zero. Reusing it keeps
        // old payloads readable (zero delta) without a format bump.
        out.extend_from_slice(&self.mrope_position_delta.to_le_bytes());

        if let Some(state) = &self.qwen4_exp {
            state
                .validate_snapshot(self.layers.len(), self.seq_len)
                .unwrap_or_else(|error| panic!("cannot serialize Flash Next cache: {error}"));
            assert!(
                self.layers.iter().all(Option::is_none)
                    && self.glm_mla_layers.iter().all(Option::is_none)
                    && self.deepseek_v4_layers.iter().all(Option::is_none)
                    && self
                        .linear_layers
                        .iter()
                        .all(|layer| layer.conv_state.is_none() && layer.recurrent_state.is_none()),
                "Flash Next cannot serialize mixed-family cache state"
            );
            for idx in 0..state.layer_count() {
                out.push(Self::LAYER_KIND_FLASH_NEXT);
                out.extend_from_slice(&[0u8; 7]);
                state.write_layer(idx, &mut out, Self::serialize_tensor);
            }
            return out;
        }

        for idx in 0..self.layers.len() {
            // Per-index disambiguation: at most one of the three layer
            // vectors is populated. The encoded `kind` byte tells the
            // reader which payload to expect.
            if let Some(fa) = &self.layers[idx] {
                assert!(
                    fa.as_contiguous()
                        .is_none_or(|lkv| lkv.protected_prefix_ring.is_none()),
                    "protected-prefix decode caches are not portable prefix snapshots"
                );
                out.push(Self::LAYER_KIND_FA);
                out.extend_from_slice(&[0u8; 7]);
                // Ring geometry must survive the round trip: a rotated
                // layer's buffer is slot-ordered, and restoring it as
                // ordered storage would make the first post-restore append
                // treat ring slots as a token-ordered prefix.
                // Paged FA always serializes as dense contiguous (no I-2 bump).
                // Quantized FA also serializes dense: the wire format (v4) is
                // dense-only, so quantized layers are dequantized here and
                // re-quantized on the first append after restore.
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
                    FaLayerStorage::Quantized(quantized) => {
                        let (k, v) = quantized.dense_view(0, self.seq_len);
                        Self::serialize_tensor_logical(&mut out, &k, logical_tokens);
                        Self::serialize_tensor_logical(&mut out, &v, logical_tokens);
                    }
                    FaLayerStorage::Paged(paged) => {
                        // serialize is rare vs decode; time is not accumulated
                        // into `paged_materialize_us` (that field tracks live
                        // SDPA materialize on the append path).
                        let pool = self
                            .fa_pool
                            .as_ref()
                            .expect("paged FA serialization requires pool");
                        let (k, v) = paged.materialize(pool, 0, self.seq_len);
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
    pub(crate) fn try_deserialize_from_bytes(
        bytes: &[u8],
    ) -> Result<Self, MlxKVCacheSerializeError> {
        let mut cursor = std::io::Cursor::new(bytes);
        Self::try_deserialize_from_reader(&mut cursor)
    }

    /// Streaming deserialize for durable L2 restore (spec §7.2 / §9).
    /// Each tensor is read directly into its final owned buffer while the
    /// caller updates integrity state (e.g. entry SHA-256). Prefer this
    /// over materializing a full payload `Vec` then copying again.
    pub(crate) fn try_deserialize_from_reader(
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
        // The header is parsed before the payload hash is known; a corrupted
        // count would otherwise size the per-layer vectors eagerly and abort
        // the process on allocation instead of rejecting the entry.
        if layer_count > MAX_WIRE_LAYER_COUNT {
            return Err(MlxKVCacheSerializeError::BadShape(layer_count));
        }
        let mrope_position_delta = read_u32_from(reader)? as i32;

        // Wire format is always dense contiguous; do not inherit env-flag
        // paged mode into restored snapshots (I-2 payload is contiguous).
        let mut cache = Self::new_contiguous(layer_count);
        cache.seq_len = seq_len;
        cache.growth_count = growth_count;
        cache.rope_offset = rope_offset;
        cache.mrope_position_delta = mrope_position_delta;

        let mut flash_layers = Vec::new();
        for idx in 0..layer_count {
            let kind = read_u8_from(reader)?;
            let mut reserved = [0u8; 7];
            read_exact_from(reader, &mut reserved)?;
            match kind {
                k if k == Self::LAYER_KIND_FLASH_NEXT => {
                    flash_layers.push(crate::model::qwen4_exp::Qwen4ExpState::read_layer(
                        reader,
                        Self::read_tensor_from_reader,
                    )?);
                }
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
                    // K and V are written from one layer with one geometry;
                    // a V that disagrees can only come from a tampered or
                    // corrupt payload and would fail deep inside the next
                    // append instead of here.
                    if v_arr.shape() != shape || v_arr.dtype() != k_arr.dtype() {
                        return Err(MlxKVCacheSerializeError::BadShape(v_arr.shape().len()));
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
                        protected_prefix_ring: None,
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
                    // Latent and positional halves cover the same tokens, and
                    // the buffers must hold every restored token.
                    if pe_shape[2] != kv_shape[2]
                        || kv_shape[2] < 0
                        || seq_len > kv_shape[2] as usize
                    {
                        return Err(MlxKVCacheSerializeError::BadShape(pe_shape.len()));
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
                        prefix_conv_state: None,
                        prefix_recurrent_state: None,
                        mtp_qkv: None,
                        mtp_a: None,
                        mtp_b: None,
                        mtp_q: None,
                        mtp_k: None,
                        mtp_v: None,
                        mtp_tape: None,
                    };
                }
                other => return Err(MlxKVCacheSerializeError::UnknownLayerKind(other)),
            }
        }

        if !flash_layers.is_empty() {
            if flash_layers.len() != layer_count || rope_offset != 0 || mrope_position_delta != 0 {
                return Err(MlxKVCacheSerializeError::FlashNextState(
                    "mixed layer kinds or unsupported position offsets".into(),
                ));
            }
            cache.qwen4_exp = Some(
                crate::model::qwen4_exp::Qwen4ExpState::from_serialized_layers(
                    seq_len,
                    flash_layers,
                )?,
            );
        }

        Ok(cache)
    }

    /// Serialize a Flash Next MTP draft cursor as a standalone sidecar
    /// payload: a `Qwen4ExpState` (one QSA layer today, but the codec does
    /// not hardcode that beyond what `write_layer`/`read_layer` already
    /// require) plus the cursor's `stream_hidden` row tensor.
    ///
    /// This is deliberately NOT part of the trunk KV-cache wire format
    /// ([`Self::serialize_to_bytes`]). It rides alongside a portable prefix
    /// snapshot as an optional field so a request resumed from a prefix-cache
    /// hit can restore a working draft cursor instead of decoding directly
    /// (Spec A plumbing only; the restore policy is a separate change).
    ///
    /// The header magic is `AXMC`, distinct from the trunk's `AXKB`, so a
    /// trunk blob passed here (or a cursor blob passed to the trunk reader)
    /// fails at the header check instead of being misparsed.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn serialize_qwen4_exp_draft_cursor(
        state: &crate::model::qwen4_exp::Qwen4ExpState,
        stream_hidden: &MlxArray,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(Self::MTP_CURSOR_MAGIC);
        out.extend_from_slice(&Self::MTP_CURSOR_VERSION.to_le_bytes());
        out.extend_from_slice(&(state.position() as u64).to_le_bytes());
        let layer_count = state.layer_count() as u32;
        out.extend_from_slice(&layer_count.to_le_bytes());
        for index in 0..state.layer_count() {
            state.write_layer(index, &mut out, Self::serialize_tensor);
        }
        Self::serialize_tensor(&mut out, stream_hidden);
        out
    }

    /// Reconstruct a draft cursor written by
    /// [`Self::serialize_qwen4_exp_draft_cursor`]. Returns the rebuilt
    /// `Qwen4ExpState` and the `stream_hidden` row tensor. Rejects a payload
    /// whose magic is not `AXMC` immediately (so a trunk `AXKB` blob is never
    /// misread here) and never silently degrades on truncation: the layer
    /// assembly step validates structural completeness and returns a
    /// structured error instead.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn try_deserialize_qwen4_exp_draft_cursor(
        bytes: &[u8],
    ) -> Result<(crate::model::qwen4_exp::Qwen4ExpState, MlxArray), MlxKVCacheSerializeError> {
        let mut cursor = std::io::Cursor::new(bytes);
        let mut magic = [0u8; 4];
        read_exact_from(&mut cursor, &mut magic)?;
        if magic != *Self::MTP_CURSOR_MAGIC {
            return Err(MlxKVCacheSerializeError::BadMagic);
        }
        let version = read_u32_from(&mut cursor)?;
        if version != Self::MTP_CURSOR_VERSION {
            return Err(MlxKVCacheSerializeError::UnsupportedVersion(version));
        }
        let position = usize::try_from(read_u64_from(&mut cursor)?)
            .map_err(|_| MlxKVCacheSerializeError::BadShape(8))?;
        let layer_count = read_u32_from(&mut cursor)? as usize;
        // Bound the layer count before allocating, mirroring the trunk
        // reader: a corrupted header must not size a vector eagerly.
        if layer_count == 0 || layer_count > MAX_WIRE_LAYER_COUNT {
            return Err(MlxKVCacheSerializeError::BadShape(layer_count));
        }
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(layer_count)
            .map_err(|_| MlxKVCacheSerializeError::UnexpectedEof)?;
        for _ in 0..layer_count {
            layers.push(crate::model::qwen4_exp::Qwen4ExpState::read_layer(
                &mut cursor,
                Self::read_tensor_from_reader,
            )?);
        }
        // `from_serialized_layers` validates that every layer agrees on its
        // owner and that the QSA K/V and indexer histories advance together,
        // then that the decoded position/token count is self-consistent.
        let state =
            crate::model::qwen4_exp::Qwen4ExpState::from_serialized_layers(position, layers)?;
        let stream_hidden = Self::read_tensor_from_reader(&mut cursor)?;
        Ok((state, stream_hidden))
    }

    /// Structural completeness check for a snapshot restored from the
    /// portable L1 / durable L2 prefix cache. The wire format cannot
    /// express which layers the model requires, so a payload can
    /// deserialize cleanly while being structurally incomplete (e.g.
    /// every layer `EMPTY`, or a truncated layer count); adopting such a
    /// snapshot silently produces wrong generation. Fail closed instead.
    ///
    /// - `expected_layer_count` is the runner's model layer count; the
    ///   payload's own header value is not trusted.
    /// - `expected_tokens` is the prefix length the caller is adopting.
    /// - `required_linear_layers` marks positions that must carry both
    ///   conv and recurrent state. Pass `None` for families whose layer
    ///   kinds cannot be derived from config alone (weight-driven
    ///   classification), where MLP/MoE layers legitimately serialize
    ///   as `EMPTY`.
    pub(crate) fn verify_restored_snapshot(
        &self,
        expected_layer_count: usize,
        expected_tokens: usize,
        required_linear_layers: Option<&[bool]>,
    ) -> Result<(), MlxKVCacheSerializeError> {
        if self.layers.len() != expected_layer_count {
            return Err(MlxKVCacheSerializeError::LayerCountMismatch {
                expected: expected_layer_count,
                actual: self.layers.len(),
            });
        }
        if self.seq_len != expected_tokens {
            return Err(MlxKVCacheSerializeError::TokenCountMismatch {
                expected: expected_tokens,
                actual: self.seq_len,
            });
        }
        if let Some(state) = &self.qwen4_exp {
            if self.rope_offset != 0 || self.mrope_position_delta != 0 {
                return Err(MlxKVCacheSerializeError::FlashNextState(
                    "unsupported position offsets".into(),
                ));
            }
            return state
                .validate_snapshot(expected_layer_count, expected_tokens)
                .map_err(MlxKVCacheSerializeError::FlashNextState);
        }
        if expected_tokens == 0 {
            return Ok(());
        }
        let layer_has_state = |idx: usize| {
            self.layers[idx].is_some()
                || self.glm_mla_layers[idx].is_some()
                || self.deepseek_v4_layers[idx].is_some()
                || self.linear_layers[idx].conv_state.is_some()
                || self.linear_layers[idx].recurrent_state.is_some()
        };
        if !(0..self.layers.len()).any(layer_has_state) {
            return Err(MlxKVCacheSerializeError::EmptySnapshot);
        }
        if let Some(required) = required_linear_layers {
            for (idx, required_linear) in required.iter().enumerate().take(self.layers.len()) {
                if !*required_linear {
                    continue;
                }
                let state = &self.linear_layers[idx];
                if state.conv_state.is_none() || state.recurrent_state.is_none() {
                    return Err(MlxKVCacheSerializeError::IncompleteLinearLayer(idx));
                }
            }
        }
        Ok(())
    }

    /// Append new K/V tokens for `layer` and return the full logical K/V for SDPA.
    ///
    /// `new_k` / `new_v` shape: `[1, n_kv_heads, new_tokens, head_dim]`
    ///
    /// Returns **owned** arrays sliced to `[1, n_kv_heads, seq_len + new_tokens, head_dim]`.
    pub(crate) fn append(
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
    pub(crate) fn append_with_retained_window(
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

        // When ring layout engages, always use the rotating path so KV shape
        // matches capacity-wide ring masks hoisted before the layer loop.
        // Cold layers (None) initialize a zeroed ring and write new tokens at
        // t % capacity — previously they took the ordered path and returned a
        // windowed view while mask builders already emitted capacity masks.
        if let Some(ring) = self.sliding_ring_layout(window, new_tokens) {
            // Quantized rings are out of scope (Phase 3b): a quantized layer
            // that reaches ring eligibility demotes back to dense contiguous
            // storage and continues on the ordinary ring path.
            self.demote_quantized_layer_to_dense(layer);
            if self.layers[layer].is_none() {
                return self.append_rotating_cold(layer, new_k, new_v, ring, append);
            }
            return self.append_rotating_retained_window(layer, new_k, new_v, ring);
        }

        let quant_spec = self.layer_kv_quant(layer);
        // Pure-FA paged path: only for empty or already-paged layers when a
        // private pool is present. Sliding retained windows stay contiguous.
        // Quantized layers always take the dense route — they never produce
        // paged storage or a `MlxAttentionKv::Paged` view.
        let use_paged = self.fa_pool.is_some()
            && window.is_none()
            && quant_spec.is_none()
            && matches!(self.layers[layer], None | Some(FaLayerStorage::Paged(_)));
        if use_paged {
            return self
                .append_paged_fa(layer, new_k, new_v, write_start..write_end, append, false)
                .into_dense();
        }

        if let Some(spec) = quant_spec
            && let Some(views) =
                self.append_quantized_fa(layer, &new_k, &new_v, window, &append, spec)
        {
            return views;
        }

        // The table can be re-injected with this layer's spec cleared (env
        // kill-switch, ring rejection, unsupported geometry) after the layer
        // already holds quantized storage: demote it so the dense path below
        // never sees a `Quantized` slot.
        self.demote_quantized_layer_to_dense(layer);

        let entry = &mut self.layers[layer];
        match entry {
            None => {
                let capacity = chunk_ceiling(write_end);
                // Fresh-layer fast path: store new_k/new_v directly and skip
                // zeros+slice_update. Aligned prompts (capacity == new_tokens)
                // always take this path. Unaligned first writes (p128) take it
                // when `AX_MLX_EXACT_SIZE_FIRST_KV` is on (default): capacity
                // is the exact prompt length; the first decode step grows to
                // chunk_ceiling(new_tokens + 1).
                if crate::fastpath::should_exact_size_first_kv(write_start)
                    || (write_start == 0 && capacity == new_tokens)
                {
                    let capacity = new_tokens;
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
                        protected_prefix_ring: None,
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
                    protected_prefix_ring: None,
                    dtype,
                }));
            }
            Some(FaLayerStorage::Paged(_)) => {
                unreachable!("paged FA layers must use append_paged_fa");
            }
            Some(FaLayerStorage::Quantized(_)) => {
                unreachable!("quantized FA layers must use append_quantized_fa");
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
                     (rotating_slack {}, cache seq_len {}, rotating_decode {}) \
                     and would corrupt slot-ordered state",
                    lkv.rotating_window,
                    lkv.capacity,
                    self.rotating_slack,
                    self.seq_len,
                    self.use_rotating_sliding_decode,
                );
                assert!(
                    lkv.protected_prefix_ring.is_none(),
                    "ordered KV append on protected-prefix ring layer {layer}"
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
                    let new_capacity = chunk_ceiling(write_end);
                    if crate::fastpath::should_exact_size_kv_grow(
                        write_start,
                        lkv.capacity,
                        write_end,
                        new_capacity,
                    ) {
                        // Aligned tight grow (p2048 chunk 2: 1024→2048):
                        // concatenate old∥new instead of zeros + two slice_updates.
                        lkv.k = concatenate(&[&lkv.k, &new_k], 2, None);
                        lkv.v = concatenate(&[&lkv.v, &new_v], 2, None);
                        lkv.capacity = new_capacity;
                        lkv.last_k_view = None;
                        lkv.last_v_view = None;
                        self.growth_count = self.growth_count.saturating_add(1);
                    } else {
                        let buf_shape = [1i32, lkv.n_kv_heads, new_capacity as i32, lkv.head_dim];
                        let k_new = zeros(&buf_shape, lkv.dtype, None);
                        let v_new = zeros(&buf_shape, lkv.dtype, None);
                        let old_stop = [1i32, lkv.n_kv_heads, lkv.capacity as i32, lkv.head_dim];
                        let zero_start = [0i32, 0, 0, 0];
                        let ones = [1i32, 1, 1, 1];
                        lkv.k = slice_update(&k_new, &lkv.k, &zero_start, &old_stop, &ones, None);
                        lkv.v = slice_update(&v_new, &lkv.v, &zero_start, &old_stop, &ones, None);
                        lkv.capacity = new_capacity;
                        lkv.last_k_view = None;
                        lkv.last_v_view = None;
                        self.growth_count = self.growth_count.saturating_add(1);
                        let start = [0i32, 0, write_start as i32, 0];
                        let stop = [1i32, lkv.n_kv_heads, write_end as i32, lkv.head_dim];
                        let strides = [1i32, 1, 1, 1];
                        lkv.k = slice_update(&lkv.k, &new_k, &start, &stop, &strides, None);
                        lkv.v = slice_update(&lkv.v, &new_v, &start, &stop, &strides, None);
                    }
                } else {
                    let start = [0i32, 0, write_start as i32, 0];
                    let stop = [1i32, lkv.n_kv_heads, write_end as i32, lkv.head_dim];
                    let strides = [1i32, 1, 1, 1];
                    lkv.k = slice_update(&lkv.k, &new_k, &start, &stop, &strides, None);
                    lkv.v = slice_update(&lkv.v, &new_v, &start, &stop, &strides, None);
                }
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
        if crate::fastpath::should_skip_unused_full_kv_view_slice(
            view_start,
            write_end,
            lkv.capacity,
        ) {
            lkv.last_k_view = Some(lkv.k.clone());
            lkv.last_v_view = Some(lkv.v.clone());
            return (lkv.k.clone(), lkv.v.clone());
        }
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

    /// Quantize-on-append path for a layer whose spec is `Some` (Phase 3b).
    ///
    /// Handles all three entry states: `None` (fresh layer — quantize the new
    /// slice into freshly allocated buffers, with a chunk-aligned fast path
    /// mirroring the dense one), `Contiguous` (dense prefix — e.g. a restored
    /// serialized snapshot — quantized wholesale on this append, then the new
    /// slice appended quantized), and `Quantized` (steady state).
    ///
    /// Returns `None` when `head_dim` is not a multiple of the group size:
    /// the spec cannot apply to this geometry, so it is dropped from the
    /// table (warn once) and the caller falls through to dense storage.
    fn append_quantized_fa(
        &mut self,
        layer: usize,
        new_k: &MlxArray,
        new_v: &MlxArray,
        window: Option<usize>,
        append: &AppendShape,
        spec: KvQuantSpec,
    ) -> Option<(MlxArray, MlxArray)> {
        if append.head_dim % (spec.group_size as i32) != 0 {
            if let Some(table) = self.kv_quant.as_mut() {
                table[layer] = None;
            }
            tracing::warn!(
                target: "ax_engine_mlx::kv_cache",
                layer,
                head_dim = append.head_dim,
                group_size = spec.group_size,
                "KV quant group size does not divide head_dim; storing layer at full precision",
            );
            return None;
        }

        let write_start = self.seq_len;
        let write_end = write_start + append.new_tokens;
        let entry = &mut self.layers[layer];
        match entry {
            None => {
                let capacity = chunk_ceiling(write_end);
                // Fresh-layer fast path: quantize outputs become the backing
                // buffers directly (no zeros + slice_update). Unaligned first
                // writes (p128) take this path when exact-size first KV is on.
                if crate::fastpath::should_exact_size_first_kv(write_start)
                    || (write_start == 0 && capacity == append.new_tokens)
                {
                    let capacity = append.new_tokens;
                    let quantized = QuantizedLayerKV {
                        k: QuantizedTensorKV::from_quantized(new_k, spec),
                        v: QuantizedTensorKV::from_quantized(new_v, spec),
                        last_k_view: None,
                        last_v_view: None,
                        n_kv_heads: append.n_kv_heads,
                        head_dim: append.head_dim,
                        capacity,
                        dtype: append.dtype,
                        spec,
                    };
                    self.growth_count = self.growth_count.saturating_add(1);
                    *entry = Some(FaLayerStorage::Quantized(quantized));
                } else {
                    let mut quantized = QuantizedLayerKV {
                        k: QuantizedTensorKV::zero_buffers(
                            append.n_kv_heads,
                            capacity,
                            append.head_dim,
                            append.dtype,
                            spec,
                        ),
                        v: QuantizedTensorKV::zero_buffers(
                            append.n_kv_heads,
                            capacity,
                            append.head_dim,
                            append.dtype,
                            spec,
                        ),
                        last_k_view: None,
                        last_v_view: None,
                        n_kv_heads: append.n_kv_heads,
                        head_dim: append.head_dim,
                        capacity,
                        dtype: append.dtype,
                        spec,
                    };
                    quantized.write_tokens(write_start, new_k, new_v);
                    self.growth_count = self.growth_count.saturating_add(1);
                    *entry = Some(FaLayerStorage::Quantized(quantized));
                }
            }
            Some(FaLayerStorage::Contiguous(lkv)) => {
                // Dense prefix with a quant spec — a restored serialized
                // snapshot, a compiled-path replacement, or a demoted paged
                // layer. Quantize the retained dense prefix wholesale on this
                // append ("re-quantize on first append after restore").
                assert!(
                    lkv.rotating_window.is_none() && lkv.protected_prefix_ring.is_none(),
                    "ring layers never carry a KV quant spec (rejected at injection)"
                );
                assert_eq!(
                    lkv.n_kv_heads, append.n_kv_heads,
                    "KV cache append cannot change n_kv_heads for an existing layer"
                );
                assert_eq!(
                    lkv.head_dim, append.head_dim,
                    "KV cache append cannot change head_dim for an existing layer"
                );
                assert_eq!(
                    lkv.dtype, append.dtype,
                    "KV cache append cannot change dtype for an existing layer"
                );
                let capacity = chunk_ceiling(write_end);
                let mut quantized = QuantizedLayerKV {
                    k: QuantizedTensorKV::zero_buffers(
                        lkv.n_kv_heads,
                        capacity,
                        lkv.head_dim,
                        lkv.dtype,
                        spec,
                    ),
                    v: QuantizedTensorKV::zero_buffers(
                        lkv.n_kv_heads,
                        capacity,
                        lkv.head_dim,
                        lkv.dtype,
                        spec,
                    ),
                    last_k_view: None,
                    last_v_view: None,
                    n_kv_heads: lkv.n_kv_heads,
                    head_dim: lkv.head_dim,
                    capacity,
                    dtype: lkv.dtype,
                    spec,
                };
                if write_start > 0 {
                    let stop = [1i32, lkv.n_kv_heads, write_start as i32, lkv.head_dim];
                    let strides = [1i32, 1, 1, 1];
                    let old_k = slice(&lkv.k, &[0, 0, 0, 0], &stop, &strides, None);
                    let old_v = slice(&lkv.v, &[0, 0, 0, 0], &stop, &strides, None);
                    quantized.write_tokens(0, &old_k, &old_v);
                }
                quantized.write_tokens(write_start, new_k, new_v);
                self.growth_count = self.growth_count.saturating_add(1);
                *entry = Some(FaLayerStorage::Quantized(quantized));
            }
            Some(FaLayerStorage::Quantized(quantized)) => {
                assert_eq!(
                    quantized.n_kv_heads, append.n_kv_heads,
                    "KV cache append cannot change n_kv_heads for an existing layer"
                );
                assert_eq!(
                    quantized.head_dim, append.head_dim,
                    "KV cache append cannot change head_dim for an existing layer"
                );
                assert_eq!(
                    quantized.dtype, append.dtype,
                    "KV cache append cannot change dtype for an existing layer"
                );
                assert_eq!(
                    quantized.spec, spec,
                    "KV quant spec cannot change mid-request for an existing layer"
                );
                quantized.ensure_capacity(write_end, &mut self.growth_count);
                quantized.write_tokens(write_start, new_k, new_v);
            }
            Some(FaLayerStorage::Paged(_)) => {
                unreachable!("quantized FA layers never take the paged route");
            }
        }

        let quantized = self.layers[layer]
            .as_mut()
            .and_then(FaLayerStorage::as_quantized_mut)
            .expect("quantized FA append path");
        let view_start = window
            .filter(|window| *window > 0)
            .map(|window| write_end.saturating_sub(window))
            .unwrap_or(0);
        let (k_view, v_view) = quantized.dense_view(view_start, write_end);
        quantized.last_k_view = Some(k_view.clone());
        quantized.last_v_view = Some(v_view.clone());
        Some((k_view, v_view))
    }

    /// Dequantize a `Quantized` layer back to dense contiguous storage,
    /// preserving the logical prefix `[0, seq_len)`. No-op for other storage
    /// kinds. Used when a ring would engage on a quantized layer (quantized
    /// rings are out of scope for Phase 3b) or when the layer's spec was
    /// retracted after storage was created.
    fn demote_quantized_layer_to_dense(&mut self, layer: usize) {
        if !matches!(
            self.layers.get(layer),
            Some(Some(FaLayerStorage::Quantized(_)))
        ) {
            return;
        }
        let taken = self.layers.get_mut(layer).map(Option::take);
        let quantized = match taken {
            Some(Some(FaLayerStorage::Quantized(quantized))) => quantized,
            Some(other) => {
                // Guard above already matched Quantized; restore defensively.
                if let Some(slot) = self.layers.get_mut(layer) {
                    *slot = other;
                }
                return;
            }
            None => return,
        };
        tracing::warn!(
            target: "ax_engine_mlx::kv_cache",
            layer,
            "demoting quantized KV layer to dense storage (ring engagement or spec retraction)",
        );
        let logical_len = self.seq_len.min(quantized.capacity);
        let capacity = if logical_len == 0 {
            0
        } else {
            chunk_ceiling(logical_len)
        };
        let (k_view, v_view) = quantized.dense_view(0, logical_len);
        let (k, v) = if logical_len == 0 || capacity == logical_len {
            (k_view, v_view)
        } else {
            let buf_shape = [
                1i32,
                quantized.n_kv_heads,
                capacity as i32,
                quantized.head_dim,
            ];
            let k_buf = zeros(&buf_shape, quantized.dtype, None);
            let v_buf = zeros(&buf_shape, quantized.dtype, None);
            let start = [0i32, 0, 0, 0];
            let stop = [
                1i32,
                quantized.n_kv_heads,
                logical_len as i32,
                quantized.head_dim,
            ];
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
                n_kv_heads: quantized.n_kv_heads,
                head_dim: quantized.head_dim,
                capacity,
                rotating_window: None,
                protected_prefix_ring: None,
                dtype: quantized.dtype,
            }));
        }
    }

    /// Standard-FA append that can return a single-slab block-table view for
    /// single-token decode. All unsupported shapes retain the dense contract.
    pub(crate) fn append_with_retained_window_for_attention(
        &mut self,
        layer: usize,
        new_k: MlxArray,
        new_v: MlxArray,
        window: Option<usize>,
    ) -> MlxAttentionKv {
        let append = validate_append_inputs(layer, self.layers.len(), &new_k, &new_v);
        let write_start = self.seq_len;
        let write_end = write_start + append.new_tokens;
        let use_native_paged = append.new_tokens == 1
            && window.is_none()
            && self.layer_kv_quant(layer).is_none()
            && self
                .fa_pool
                .as_ref()
                .is_some_and(SharedFaBlockPool::native_attention_enabled)
            && matches!(self.layers[layer], None | Some(FaLayerStorage::Paged(_)));
        if use_native_paged {
            return self.append_paged_fa(layer, new_k, new_v, write_start..write_end, append, true);
        }
        let (k, v) = self.append_with_retained_window(layer, new_k, new_v, window);
        MlxAttentionKv::Dense { k, v }
    }

    /// Append one decode token while retaining the complete existing prefill
    /// plus a bounded ring of generated tokens.
    ///
    /// Unlimited-OCR's R-SWA is not ordinary sliding attention: image and text
    /// prompt K/V remain attendable forever, while only post-prefill decode K/V
    /// rotate through `window` physical slots. RoPE is applied by the caller at
    /// the monotonically increasing logical position, so the circular ordering
    /// of the decode suffix does not affect single-token attention.
    pub(crate) fn append_with_protected_prefix_window_for_attention(
        &mut self,
        layer: usize,
        new_k: MlxArray,
        new_v: MlxArray,
        window: usize,
    ) -> MlxAttentionKv {
        let append = validate_append_inputs(layer, self.layers.len(), &new_k, &new_v);
        assert_eq!(
            append.new_tokens, 1,
            "protected-prefix R-SWA only supports single-token decode"
        );
        assert!(window > 0, "protected-prefix R-SWA window must be positive");
        assert!(
            self.seq_len > 0,
            "protected-prefix R-SWA requires a completed prefill"
        );

        if matches!(self.layers[layer], Some(FaLayerStorage::Paged(_))) {
            self.demote_paged_layer_to_contiguous(layer, self.seq_len);
        }
        // Protected-prefix rings over quantized storage are out of scope
        // (Phase 3b): demote to dense before the ring engages.
        self.demote_quantized_layer_to_dense(layer);

        let lkv = self.layers[layer]
            .as_mut()
            .and_then(FaLayerStorage::as_contiguous_mut)
            .expect("protected-prefix R-SWA requires an existing contiguous prefill cache");
        assert!(
            lkv.rotating_window.is_none(),
            "protected-prefix and ordinary rotating rings cannot share a layer"
        );
        assert_eq!(
            lkv.n_kv_heads, append.n_kv_heads,
            "KV cache append cannot change n_kv_heads for an existing layer"
        );
        assert_eq!(
            lkv.head_dim, append.head_dim,
            "KV cache append cannot change head_dim for an existing layer"
        );
        assert_eq!(
            lkv.dtype, append.dtype,
            "KV cache append cannot change dtype for an existing layer"
        );

        let ring = lkv
            .protected_prefix_ring
            .get_or_insert(ProtectedPrefixRing {
                prefix_len: self.seq_len,
                window,
            })
            .clone();
        assert_eq!(
            ring.window, window,
            "protected-prefix R-SWA window changed mid-request"
        );
        assert!(
            self.seq_len >= ring.prefix_len,
            "protected-prefix R-SWA logical position moved before its prefill"
        );

        let required_capacity = ring.prefix_len.saturating_add(ring.window);
        if lkv.capacity < required_capacity {
            let new_capacity = chunk_ceiling(required_capacity);
            let shape = [1i32, lkv.n_kv_heads, new_capacity as i32, lkv.head_dim];
            let start = [0i32, 0, 0, 0];
            let old_stop = [1i32, lkv.n_kv_heads, lkv.capacity as i32, lkv.head_dim];
            let strides = [1i32, 1, 1, 1];
            lkv.k = slice_update(
                &zeros(&shape, lkv.dtype, None),
                &lkv.k,
                &start,
                &old_stop,
                &strides,
                None,
            );
            lkv.v = slice_update(
                &zeros(&shape, lkv.dtype, None),
                &lkv.v,
                &start,
                &old_stop,
                &strides,
                None,
            );
            lkv.capacity = new_capacity;
            self.growth_count = self.growth_count.saturating_add(1);
        }

        let decoded_before = self.seq_len - ring.prefix_len;
        let slot = ring.prefix_len + decoded_before % ring.window;
        let start = [0i32, 0, slot as i32, 0];
        let stop = [1i32, lkv.n_kv_heads, slot as i32 + 1, lkv.head_dim];
        let strides = [1i32, 1, 1, 1];
        lkv.k = slice_update(&lkv.k, &new_k, &start, &stop, &strides, None);
        lkv.v = slice_update(&lkv.v, &new_v, &start, &stop, &strides, None);

        let physical_len = ring.prefix_len + (decoded_before + 1).min(ring.window);
        let end = physical_len as i32;
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
        lkv.last_k_view = Some(k.clone());
        lkv.last_v_view = Some(v.clone());
        MlxAttentionKv::Dense { k, v }
    }

    pub(crate) fn record_paged_attention_result(&mut self, used_native: bool) {
        if used_native {
            self.paged_attention_calls = self.paged_attention_calls.saturating_add(1);
        } else {
            self.paged_attention_fallbacks = self.paged_attention_fallbacks.saturating_add(1);
        }
    }

    fn append_paged_fa(
        &mut self,
        layer: usize,
        new_k: MlxArray,
        new_v: MlxArray,
        write_range: std::ops::Range<usize>,
        append: AppendShape,
        prefer_native_attention: bool,
    ) -> MlxAttentionKv {
        let write_start = write_range.start;
        let write_end = write_range.end;
        let pool = self
            .fa_pool
            .as_ref()
            .expect("paged append requires FA block pool")
            .clone();
        let block_size = pool.config().block_size_tokens as usize;

        if self.layers[layer].is_none() {
            self.layers[layer] = Some(FaLayerStorage::Paged(PagedFaLayer {
                layer_idx: layer,
                n_kv_heads: append.n_kv_heads,
                head_dim: append.head_dim,
                dtype: append.dtype,
                block_size,
                block_ids: Vec::new(),
                slab_storage: pool.slab_storage_enabled(),
                k_blocks: Vec::new(),
                v_blocks: Vec::new(),
                last_k_view: None,
                last_v_view: None,
            }));
        }

        let preparation = {
            let paged = match self.layers[layer]
                .as_mut()
                .expect("paged layer just created")
            {
                FaLayerStorage::Paged(p) => p,
                FaLayerStorage::Contiguous(_) | FaLayerStorage::Quantized(_) => {
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
            match paged.ensure_capacity(&pool, write_end, &mut self.growth_count) {
                Ok(()) => match paged.prepare_write(&pool, write_start, write_end) {
                    Ok(cow_copies) => paged
                        .write_tokens(&pool, write_start, &new_k, &new_v)
                        .map(|()| cow_copies),
                    Err(err) => Err(err),
                },
                Err(err) => Err(err),
            }
        };

        let need_demote = match preparation {
            Ok(cow_copies) => {
                self.paged_cow_copies = self.paged_cow_copies.saturating_add(cow_copies);
                false
            }
            Err(FaBlockPoolError::Exhausted { .. }) => true,
            Err(err) => panic!("FA block pool error on layer {layer}: {err}"),
        };

        if need_demote {
            // Either way, materialize existing paged blocks into
            // contiguous storage first and finish this append on the
            // historical growth path, so the compute graph for this forward
            // stays well-formed and correct (proven token-exact by
            // fa_paged_pool_exhaustion_demotion_matches_contiguous_oracle)
            // regardless of policy. Under a production `hard_cap` this also
            // sticks hard_cap_exhausted so the caller fails the request
            // instead of returning a token computed past the operator's
            // memory bound; without an explicit cap this stays fail-soft.
            self.paged_pool_exhaustion_fallbacks =
                self.paged_pool_exhaustion_fallbacks.saturating_add(1);
            if self
                .fa_pool
                .as_ref()
                .is_some_and(|pool| pool.config().hard_cap)
            {
                self.hard_cap_exhausted = true;
            }
            self.demote_paged_layer_to_contiguous(layer, write_start);
            let (k, v) = self.append_with_retained_window(layer, new_k, new_v, None);
            return MlxAttentionKv::Dense { k, v };
        }

        let paged = match self.layers[layer]
            .as_mut()
            .expect("paged layer after write")
        {
            FaLayerStorage::Paged(p) => p,
            FaLayerStorage::Contiguous(_) | FaLayerStorage::Quantized(_) => {
                unreachable!("paged path")
            }
        };
        if prefer_native_attention && let Some(view) = paged.attention_view(&pool, write_end) {
            return MlxAttentionKv::Paged(view);
        }
        let started = Instant::now();
        let (k_view, v_view) = paged.materialize(&pool, 0, write_end);
        self.paged_materialize_us = self
            .paged_materialize_us
            .saturating_add(started.elapsed().as_micros() as u64);
        paged.last_k_view = Some(k_view.clone());
        paged.last_v_view = Some(v_view.clone());
        MlxAttentionKv::Dense {
            k: k_view,
            v: v_view,
        }
    }

    /// Convert a paged FA layer into contiguous storage up to `logical_len`
    /// tokens and release its paged blocks. Used on pool exhaustion.
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
            let pool = self
                .fa_pool
                .as_ref()
                .expect("paged FA demotion requires pool");
            paged.materialize(pool, 0, logical_len)
        };
        self.paged_materialize_us = self
            .paged_materialize_us
            .saturating_add(started.elapsed().as_micros() as u64);
        if let Some(pool) = self.fa_pool.as_ref() {
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
                protected_prefix_ring: None,
                dtype: paged.dtype,
            }));
        }
    }

    /// Cold start a sliding ring: allocate `capacity` slots and write the first
    /// multi-token (or single-token) append at `t % capacity`. Returns the full
    /// capacity K/V so SDPA can pair with a capacity-wide ring mask.
    fn append_rotating_cold(
        &mut self,
        layer: usize,
        new_k: MlxArray,
        new_v: MlxArray,
        ring: SlidingRingLayout,
        append: AppendShape,
    ) -> (MlxArray, MlxArray) {
        let SlidingRingLayout {
            window,
            capacity,
            write_start,
        } = ring;
        let new_tokens = append.new_tokens;
        assert!(
            new_tokens <= capacity,
            "rotating cold append cannot exceed ring capacity ({new_tokens} > {capacity})"
        );
        let buf_shape = [1i32, append.n_kv_heads, capacity as i32, append.head_dim];
        let mut k_buf = zeros(&buf_shape, append.dtype, None);
        let mut v_buf = zeros(&buf_shape, append.dtype, None);
        // Scatter new tokens into ring slots (wraps at most once).
        let mut src_start = 0usize;
        while src_start < new_tokens {
            let dst_start = (write_start + src_start) % capacity;
            let len = (new_tokens - src_start).min(capacity - dst_start);
            let (seg_k, seg_v) = if src_start == 0 && len == new_tokens {
                (new_k.clone(), new_v.clone())
            } else {
                let src_stop = (src_start + len) as i32;
                let seg_start = [0i32, 0, src_start as i32, 0];
                let seg_stop = [1i32, append.n_kv_heads, src_stop, append.head_dim];
                let ones = [1i32, 1, 1, 1];
                (
                    slice(&new_k, &seg_start, &seg_stop, &ones, None),
                    slice(&new_v, &seg_start, &seg_stop, &ones, None),
                )
            };
            let start = [0i32, 0, dst_start as i32, 0];
            let stop = [
                1i32,
                append.n_kv_heads,
                (dst_start + len) as i32,
                append.head_dim,
            ];
            let strides = [1i32, 1, 1, 1];
            k_buf = slice_update(&k_buf, &seg_k, &start, &stop, &strides, None);
            v_buf = slice_update(&v_buf, &seg_v, &start, &stop, &strides, None);
            src_start += len;
        }
        self.growth_count = self.growth_count.saturating_add(1);
        self.layers[layer] = Some(FaLayerStorage::Contiguous(LayerKV {
            k: k_buf.clone(),
            v: v_buf.clone(),
            n_kv_heads: append.n_kv_heads,
            head_dim: append.head_dim,
            capacity,
            rotating_window: Some(window),
            protected_prefix_ring: None,
            dtype: append.dtype,
            last_k_view: Some(k_buf.clone()),
            last_v_view: Some(v_buf.clone()),
        }));
        (k_buf, v_buf)
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
            assert!(
                lkv.protected_prefix_ring.is_none(),
                "ordinary sliding ring cannot replace a protected-prefix ring"
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
    pub(crate) fn append_glm_mla(
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
                if crate::fastpath::should_exact_size_first_kv(write_start)
                    || (write_start == 0 && capacity == new_tokens)
                {
                    let capacity = new_tokens;
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

    /// Append DeepSeek V4 (Flash) latent K tokens and return the full logical
    /// latent view.
    ///
    /// `new_k_latent`: `[1, 1, new_tokens, head_dim]` — ONE latent K vector
    /// per token (single latent head; K doubles as V, so there is no V cache).
    /// Supports both prefill batches and single-token decode appends.
    ///
    /// Returns `[1, 1, total_tokens, head_dim]`. Use
    /// [`Self::deepseek_v4_k_window`] for the sliding-window read-back.
    ///
    /// Phase-3 compressor state (compressed-K rows, indexer rows, buffered
    /// per-token states) lives alongside this buffer on
    /// `DeepseekV4LayerCache`; this method stays the raw-K append.
    pub(crate) fn append_deepseek_v4(&mut self, layer: usize, new_k_latent: MlxArray) -> MlxArray {
        let append =
            validate_deepseek_v4_append_inputs(layer, self.deepseek_v4_layers.len(), &new_k_latent);
        let new_tokens = append.new_tokens;
        let write_start = self.seq_len;
        let write_end = write_start + new_tokens;
        let dtype = append.dtype;
        let head_dim = append.head_dim;

        let entry = &mut self.deepseek_v4_layers[layer];
        match entry {
            None => {
                let capacity = chunk_ceiling(write_end);
                if write_start == 0 && capacity == new_tokens {
                    self.growth_count = self.growth_count.saturating_add(1);
                    *entry = Some(DeepseekV4LayerCache {
                        k_latent: new_k_latent.clone(),
                        head_dim,
                        capacity,
                        dtype,
                        comp: None,
                        indexer_comp: None,
                    });
                    return new_k_latent;
                }
                let buf_shape = [1i32, 1, capacity as i32, head_dim];
                let buf = zeros(&buf_shape, dtype, None);
                let k_latent = slice_update(
                    &buf,
                    &new_k_latent,
                    &[0, 0, write_start as i32, 0],
                    &[1, 1, write_end as i32, head_dim],
                    &[1, 1, 1, 1],
                    None,
                );
                self.growth_count = self.growth_count.saturating_add(1);
                *entry = Some(DeepseekV4LayerCache {
                    k_latent,
                    head_dim,
                    capacity,
                    dtype,
                    comp: None,
                    indexer_comp: None,
                });
            }
            Some(cache) => {
                assert_eq!(
                    cache.head_dim, head_dim,
                    "DeepSeek V4 cache append cannot change head_dim for an existing layer"
                );
                assert_eq!(
                    cache.dtype, dtype,
                    "DeepSeek V4 cache append cannot change dtype for an existing layer"
                );
                if write_end > cache.capacity {
                    let new_capacity = chunk_ceiling(write_end);
                    let buf_shape = [1i32, 1, new_capacity as i32, cache.head_dim];
                    let buf_new = zeros(&buf_shape, cache.dtype, None);
                    cache.k_latent = slice_update(
                        &buf_new,
                        &cache.k_latent,
                        &[0, 0, 0, 0],
                        &[1, 1, cache.capacity as i32, cache.head_dim],
                        &[1, 1, 1, 1],
                        None,
                    );
                    cache.capacity = new_capacity;
                    self.growth_count = self.growth_count.saturating_add(1);
                }
                cache.k_latent = slice_update(
                    &cache.k_latent,
                    &new_k_latent,
                    &[0, 0, write_start as i32, 0],
                    &[1, 1, write_end as i32, cache.head_dim],
                    &[1, 1, 1, 1],
                    None,
                );
            }
        }

        let cache = self.deepseek_v4_layers[layer]
            .as_ref()
            .expect("DeepSeek V4 cache entry must exist after append");
        slice(
            &cache.k_latent,
            &[0, 0, 0, 0],
            &[1, 1, write_end as i32, cache.head_dim],
            &[1, 1, 1, 1],
            None,
        )
    }

    #[cfg(test)]
    /// Read back the latent K window for a DeepSeek V4 layer: the last
    /// `sliding_window` entries of the logical cache, or the full logical
    /// cache when `sliding_window` is 0 (uncompressed layers attend the full
    /// window) or exceeds the cached length. `sliding_window` comes from
    /// `DeepseekV4Config` (via the model's sliding-window setting).
    ///
    /// Returns `None` when the layer has no V4 cache; otherwise
    /// `[1, 1, min(window, seq_len), head_dim]`.
    pub(crate) fn deepseek_v4_k_window(
        &self,
        layer: usize,
        sliding_window: usize,
    ) -> Option<MlxArray> {
        let cache = self.deepseek_v4_layers.get(layer)?.as_ref()?;
        let len = self.seq_len;
        let window = if sliding_window == 0 {
            len
        } else {
            sliding_window.min(len)
        };
        let start = (len - window) as i32;
        Some(slice(
            &cache.k_latent,
            &[0, 0, start, 0],
            &[1, 1, len as i32, cache.head_dim],
            &[1, 1, 1, 1],
            None,
        ))
    }

    #[cfg(test)]
    /// Read-only access to a single DeepSeek V4 layer's cached latent K array
    /// plus its inner dim. Mirrors [`Self::glm_mla_layer_state`] for debug
    /// tooling. The array is over-allocated to capacity; slice to
    /// `[0..self.seq_len]` for the valid region.
    pub(crate) fn deepseek_v4_layer_state(
        &self,
        layer: usize,
    ) -> Option<DeepseekV4LayerStateView<'_>> {
        let entry = self.deepseek_v4_layers.get(layer)?.as_ref()?;
        Some(DeepseekV4LayerStateView {
            k_latent: &entry.k_latent,
            head_dim: entry.head_dim,
        })
    }

    fn deepseek_v4_comp_ref(
        &self,
        layer: usize,
        indexer: bool,
    ) -> Option<&DeepseekV4CompressorCache> {
        let entry = self.deepseek_v4_layers.get(layer)?.as_ref()?;
        if indexer {
            entry.indexer_comp.as_ref()
        } else {
            entry.comp.as_ref()
        }
    }

    fn deepseek_v4_comp_mut(
        &mut self,
        layer: usize,
        indexer: bool,
    ) -> Option<&mut DeepseekV4CompressorCache> {
        let entry = self.deepseek_v4_layers.get_mut(layer)?.as_mut()?;
        if indexer {
            entry.indexer_comp.as_mut()
        } else {
            entry.comp.as_mut()
        }
    }

    /// Ensure a DeepSeek V4 compressor pipeline's cache exists for `layer`
    /// (Phase 3). `indexer` selects the LID indexer pipeline; `ratio` and
    /// `overlap` are recorded on creation for block-boundary bookkeeping on
    /// `trim_to`. No-op when the layer has no V4 cache yet — the compressor
    /// update runs after the raw append, which creates the entry.
    pub(crate) fn deepseek_v4_comp_ensure(
        &mut self,
        layer: usize,
        indexer: bool,
        ratio: usize,
        overlap: bool,
    ) {
        let Some(entry) = self
            .deepseek_v4_layers
            .get_mut(layer)
            .and_then(Option::as_mut)
        else {
            return;
        };
        let slot = if indexer {
            &mut entry.indexer_comp
        } else {
            &mut entry.comp
        };
        if slot.is_none() {
            *slot = Some(DeepseekV4CompressorCache {
                state_kv: None,
                state_score: None,
                state_base: 0,
                committed: 0,
                comp_k: None,
                capacity: 0,
                ratio,
                overlap,
            });
        }
    }

    /// Number of committed compressed-K rows (== completed blocks) for one
    /// compressor pipeline. `0` when the pipeline does not exist.
    pub(crate) fn deepseek_v4_comp_committed(&self, layer: usize, indexer: bool) -> usize {
        self.deepseek_v4_comp_ref(layer, indexer)
            .map_or(0, |comp| comp.committed)
    }

    /// Buffered F32 per-token compressor states: `(state_base, kv, score)`
    /// with `kv`/`score` `[rows, state_width]` covering positions
    /// `[state_base, state_base + rows)`. `None` when the buffer is empty.
    pub(crate) fn deepseek_v4_comp_states(
        &self,
        layer: usize,
        indexer: bool,
    ) -> Option<(usize, MlxArray, MlxArray)> {
        let comp = self.deepseek_v4_comp_ref(layer, indexer)?;
        match (&comp.state_kv, &comp.state_score) {
            (Some(kv), Some(score)) => Some((comp.state_base, kv.clone(), score.clone())),
            _ => None,
        }
    }

    /// Replace the buffered per-token compressor states after a forward
    /// (the compressor module slices the partial-window tail itself).
    /// `None`/`None` clears the buffer (positions restart at `state_base`).
    pub(crate) fn deepseek_v4_comp_replace_states(
        &mut self,
        layer: usize,
        indexer: bool,
        state_base: usize,
        kv: Option<MlxArray>,
        score: Option<MlxArray>,
    ) {
        let comp = self
            .deepseek_v4_comp_mut(layer, indexer)
            .expect("DeepSeek V4 compressor state must be ensured before state replace");
        comp.state_base = state_base;
        comp.state_kv = kv;
        comp.state_score = score;
    }

    /// Append committed compressed-K rows for one compressor pipeline and
    /// return the full committed view.
    ///
    /// `rows`: `[1, 1, n_blocks, row_dim]` — one compressed row per newly
    /// completed block, already RMSNormed, RoPE'd at the block-start
    /// positions, and cast to the raw-K dtype. Rows are written contiguously
    /// at the committed-row boundary, so a `trim_to` rewind followed by
    /// re-compression simply overwrites the rejected rows.
    ///
    /// Returns `[1, 1, committed, row_dim]`.
    pub(crate) fn append_deepseek_v4_comp_rows(
        &mut self,
        layer: usize,
        indexer: bool,
        rows: MlxArray,
    ) -> MlxArray {
        let shape = rows.shape();
        assert_eq!(
            shape.len(),
            4,
            "DeepSeek V4 compressed rows must be [1, 1, n, dim]"
        );
        assert_eq!(shape[0], 1, "DeepSeek V4 compressed rows batch must be 1");
        assert_eq!(
            shape[1], 1,
            "DeepSeek V4 compressed rows head count must be 1"
        );
        let new_rows = shape[2] as usize;
        let row_dim = shape[3];
        // Direct field access (not `deepseek_v4_comp_mut`) so the borrow
        // splitter sees `deepseek_v4_layers` and `growth_count` as disjoint.
        let comp = self
            .deepseek_v4_layers
            .get_mut(layer)
            .and_then(Option::as_mut)
            .and_then(|entry| {
                if indexer {
                    entry.indexer_comp.as_mut()
                } else {
                    entry.comp.as_mut()
                }
            })
            .expect("DeepSeek V4 compressor state must be ensured before row append");
        let write_start = comp.committed;
        let write_end = write_start + new_rows;

        match comp.comp_k.take() {
            None => {
                let capacity = chunk_ceiling(write_end.max(1));
                if write_start == 0 && capacity == new_rows {
                    self.growth_count = self.growth_count.saturating_add(1);
                    comp.capacity = capacity;
                    comp.comp_k = Some(rows.clone());
                    comp.committed = write_end;
                    return rows;
                }
                let buf = zeros(&[1, 1, capacity as i32, row_dim], rows.dtype(), None);
                let stored = slice_update(
                    &buf,
                    &rows,
                    &[0, 0, write_start as i32, 0],
                    &[1, 1, write_end as i32, row_dim],
                    &[1, 1, 1, 1],
                    None,
                );
                self.growth_count = self.growth_count.saturating_add(1);
                comp.capacity = capacity;
                comp.comp_k = Some(stored);
            }
            Some(mut buf) => {
                if write_end > comp.capacity {
                    let new_capacity = chunk_ceiling(write_end);
                    let grown = zeros(&[1, 1, new_capacity as i32, row_dim], buf.dtype(), None);
                    buf = slice_update(
                        &grown,
                        &buf,
                        &[0, 0, 0, 0],
                        &[1, 1, comp.capacity as i32, row_dim],
                        &[1, 1, 1, 1],
                        None,
                    );
                    comp.capacity = new_capacity;
                    self.growth_count = self.growth_count.saturating_add(1);
                }
                buf = slice_update(
                    &buf,
                    &rows,
                    &[0, 0, write_start as i32, 0],
                    &[1, 1, write_end as i32, row_dim],
                    &[1, 1, 1, 1],
                    None,
                );
                comp.comp_k = Some(buf);
            }
        }
        comp.committed = write_end;

        let comp = self
            .deepseek_v4_comp_ref(layer, indexer)
            .expect("DeepSeek V4 compressor state must exist after row append");
        let comp_k = comp
            .comp_k
            .as_ref()
            .expect("DeepSeek V4 compressed-K buffer must exist after row append");
        slice(
            comp_k,
            &[0, 0, 0, 0],
            &[1, 1, write_end as i32, row_dim],
            &[1, 1, 1, 1],
            None,
        )
    }

    /// Read back the committed compressed-K rows for one compressor pipeline:
    /// `[1, 1, committed, row_dim]`, or `None` when no rows are committed.
    pub(crate) fn deepseek_v4_comp_k(&self, layer: usize, indexer: bool) -> Option<MlxArray> {
        let comp = self.deepseek_v4_comp_ref(layer, indexer)?;
        let comp_k = comp.comp_k.as_ref()?;
        if comp.committed == 0 {
            return None;
        }
        Some(slice(
            comp_k,
            &[0, 0, 0, 0],
            &[1, 1, comp.committed as i32, comp_k.shape()[3]],
            &[1, 1, 1, 1],
            None,
        ))
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
        if self.qwen4_exp.is_some() && prefix_len < self.seq_len {
            // GDN and PLE state cannot be sliced. The verifier restores its
            // pre-call checkpoint and replays the accepted prefix instead.
            if prefix_len == 0 {
                self.reset();
                return true;
            }
            return false;
        }
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
                    lkv.protected_prefix_ring.is_some()
                        || lkv
                            .rotating_window
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
            // Paged layers also release fully-empty trailing blocks back to
            // their pool (fail-closed capacity bookkeeping).
            for fa in self.layers.iter_mut().flatten() {
                fa.clear_views();
            }
            if let Some(pool) = self.fa_pool.as_ref() {
                for fa in self.layers.iter_mut().flatten() {
                    if let FaLayerStorage::Paged(paged) = fa {
                        paged.free_blocks_beyond(pool, self.seq_len);
                    }
                }
            }
            self.rewind_deepseek_v4_comps(self.seq_len);
        }
        valid
    }

    /// Rewind DeepSeek V4 compressor bookkeeping after `trim_to`: committed
    /// compressed rows beyond `prefix_len / ratio` are logically dropped (the
    /// next forward re-compresses and overwrites them), and the per-token
    /// state buffer is re-sliced to the retained positions it still covers.
    ///
    /// The buffer only retains the last block-plus-partial window, so a
    /// rollback that reaches further back than the buffered range leaves the
    /// pipeline without the states it would need to recompress straddling
    /// blocks; llama.cpp solves this with rollback snapshot planes
    /// (`llama-kv-cache-dsv4.cpp` `state_snapshot_*`). That depth of draft
    /// rollback on compress layers is out of scope this phase — such trims
    /// clear the buffer (positions restart empty at `prefix_len`), which the
    /// compressor module treats as a best-effort partial window.
    fn rewind_deepseek_v4_comps(&mut self, prefix_len: usize) {
        for dsv4 in self.deepseek_v4_layers.iter_mut().flatten() {
            for comp in [&mut dsv4.comp, &mut dsv4.indexer_comp]
                .into_iter()
                .flatten()
            {
                let ratio = comp.ratio.max(1);
                let new_committed = comp.committed.min(prefix_len / ratio);
                comp.committed = new_committed;
                // Positions the next forward needs buffered: the previous
                // block's rows (overlap only) plus the partial current block.
                let keep_from = if comp.overlap && new_committed > 0 {
                    (new_committed - 1) * ratio
                } else {
                    new_committed * ratio
                };
                let rows = comp
                    .state_kv
                    .as_ref()
                    .map_or(0, |kv| kv.shape()[0] as usize);
                let buffered_from = comp.state_base;
                let buffered_to = buffered_from + rows;
                if keep_from >= buffered_from && prefix_len <= buffered_to && prefix_len > keep_from
                {
                    let start = (keep_from - buffered_from) as i32;
                    let stop = (prefix_len - buffered_from) as i32;
                    let width = comp.state_kv.as_ref().map_or(0, |kv| kv.shape()[1]);
                    comp.state_kv = comp
                        .state_kv
                        .as_ref()
                        .map(|kv| slice(kv, &[start, 0], &[stop, width], &[1, 1], None));
                    comp.state_score = comp
                        .state_score
                        .as_ref()
                        .map(|score| slice(score, &[start, 0], &[stop, width], &[1, 1], None));
                    comp.state_base = keep_from;
                } else {
                    // Not covered (or nothing to retain): clear the buffer.
                    comp.state_kv = None;
                    comp.state_score = None;
                    comp.state_base = prefix_len;
                }
            }
        }
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
                    lkv.rotating_window.is_none() && lkv.protected_prefix_ring.is_none(),
                    "logical_layer_kv reads a [0, seq_len) prefix slice, which is meaningless \
                     on a rotated ring (MTP draft seeding never coexists with rotation)"
                );
                let end = self.seq_len as i32;
                let stop = [1, lkv.n_kv_heads, end, lkv.head_dim];
                let k = slice(&lkv.k, &[0, 0, 0, 0], &stop, &[1, 1, 1, 1], None);
                let v = slice(&lkv.v, &[0, 0, 0, 0], &stop, &[1, 1, 1, 1], None);
                Some((k, v))
            }
            FaLayerStorage::Paged(paged) => Some(
                paged.materialize(
                    self.fa_pool
                        .as_ref()
                        .expect("paged logical view requires pool"),
                    0,
                    self.seq_len,
                ),
            ),
            FaLayerStorage::Quantized(quantized) => Some(quantized.dense_view(0, self.seq_len)),
        }
    }

    /// Prepare one dense, fixed-shape full-attention backing pair for the
    /// whole-model verifier and return cloned array handles.
    ///
    /// The compiled verifier threads the complete backing buffers through a
    /// dynamic `slice_update`; logical length stays in an explicit tensor
    /// input. Paged storage is materialized once on the speculative branch,
    /// and unsupported quantized/ring layouts fail closed.
    pub(crate) fn prepare_whole_verify_layer_kv(
        &mut self,
        layer: usize,
        min_capacity: usize,
    ) -> Option<(MlxArray, MlxArray)> {
        if matches!(self.layers.get(layer)?, Some(FaLayerStorage::Paged(_))) {
            self.demote_paged_layer_to_contiguous(layer, self.seq_len);
        }
        let (k, v, grew) = {
            let entry = self.layers.get_mut(layer)?.as_mut()?;
            let FaLayerStorage::Contiguous(lkv) = entry else {
                return None;
            };
            if lkv.rotating_window.is_some() || lkv.protected_prefix_ring.is_some() {
                return None;
            }
            let grew = lkv.capacity < min_capacity;
            if grew {
                let new_capacity = chunk_ceiling(min_capacity);
                let shape = [1, lkv.n_kv_heads, new_capacity as i32, lkv.head_dim];
                let k_new = zeros(&shape, lkv.dtype, None);
                let v_new = zeros(&shape, lkv.dtype, None);
                let start = [0, 0, 0, 0];
                let stop = [1, lkv.n_kv_heads, lkv.capacity as i32, lkv.head_dim];
                let strides = [1, 1, 1, 1];
                lkv.k = slice_update(&k_new, &lkv.k, &start, &stop, &strides, None);
                lkv.v = slice_update(&v_new, &lkv.v, &start, &stop, &strides, None);
                lkv.capacity = new_capacity;
                lkv.last_k_view = None;
                lkv.last_v_view = None;
            }
            (lkv.k.clone(), lkv.v.clone(), grew)
        };
        if grew {
            self.growth_count = self.growth_count.saturating_add(1);
        }
        Some((k, v))
    }

    /// Commit a pure compiled MTP draft closure's threaded K/V back into the
    /// cache as a tight logical buffer (`capacity == length`) and set `seq_len`.
    ///
    /// `k`/`v` are the closure's final concatenated `[1, n_kv_heads, length,
    /// head_dim]` outputs (already evaluated).  Storing them tight is correct:
    /// the next imperative `append` sees `write_end > capacity` and grows via
    /// its normal chunk path, copying this buffer forward.  Replacing the layer
    /// entry also drops any stale views.
    pub(crate) fn set_layer_kv_logical(
        &mut self,
        layer: usize,
        k: MlxArray,
        v: MlxArray,
        seq_len: usize,
    ) {
        let shape = k.shape();
        debug_assert_eq!(shape.len(), 4, "set_layer_kv_logical expects a 4D K array");
        let n_kv_heads = shape[1];
        let length = shape[2] as usize;
        let head_dim = shape[3];
        let dtype = k.dtype();
        // A mismatch would leave the next append writing past `capacity`
        // (zero-gap tokens SDPA would still attend to) or strand sibling
        // layers behind the global seq_len.
        debug_assert_eq!(
            length, seq_len,
            "set_layer_kv_logical commits tight buffers: K length must equal seq_len"
        );
        // Compiled/speculative paths return dense K/V. Release this view's
        // paged ownership before replacing it so native adopters keep their
        // own references and the pool never leaks the removed IDs.
        let previous = self.layers.get_mut(layer).and_then(Option::take);
        if let Some(FaLayerStorage::Paged(paged)) = previous
            && let Some(pool) = self.fa_pool.as_ref()
            && let Err(error) = pool.free(&paged.block_ids)
        {
            tracing::error!(
                target: "ax_engine_mlx::kv_pool",
                %error,
                "failed to release paged KV blocks during dense layer replacement",
            );
        }
        self.layers[layer] = Some(FaLayerStorage::Contiguous(LayerKV {
            last_k_view: Some(k.clone()),
            last_v_view: Some(v.clone()),
            k,
            v,
            n_kv_heads,
            head_dim,
            capacity: length,
            rotating_window: None,
            protected_prefix_ring: None,
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
        if let Some(state) = &self.qwen4_exp {
            refs.extend(state.arrays());
        }
        for fa in self.layers.iter().flatten() {
            match fa {
                FaLayerStorage::Contiguous(lkv) => {
                    refs.push(&lkv.k);
                    refs.push(&lkv.v);
                }
                FaLayerStorage::Quantized(quantized) => {
                    // Packed + scales + biases all mutate via slice_update on
                    // append, so they need the same lazy-chain flattening as
                    // dense buffers.
                    refs.push(&quantized.k.packed);
                    refs.push(&quantized.k.scales);
                    refs.push(&quantized.k.biases);
                    refs.push(&quantized.v.packed);
                    refs.push(&quantized.v.scales);
                    refs.push(&quantized.v.biases);
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
        for dsv4 in self.deepseek_v4_layers.iter().flatten() {
            refs.push(&dsv4.k_latent);
            for comp in [&dsv4.comp, &dsv4.indexer_comp].into_iter().flatten() {
                if let Some(comp_k) = &comp.comp_k {
                    refs.push(comp_k);
                }
                if let Some(state_kv) = &comp.state_kv {
                    refs.push(state_kv);
                }
                if let Some(state_score) = &comp.state_score {
                    refs.push(state_score);
                }
            }
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

    pub(crate) fn usage_snapshot(&self) -> MlxKVCacheUsage {
        self.usage_snapshot_with_layer_windows(&[])
    }

    pub(crate) fn usage_snapshot_with_layer_windows(
        &self,
        layer_windows: &[Option<usize>],
    ) -> MlxKVCacheUsage {
        let mut usage = MlxKVCacheUsage {
            logical_tokens: self.seq_len,
            growth_count: self.growth_count,
            rotating_ring_slack: self.rotating_slack,
            paged_materialize_us: self.paged_materialize_us,
            paged_pool_exhaustion_fallbacks: self.paged_pool_exhaustion_fallbacks,
            paged_cow_copies: self.paged_cow_copies,
            paged_attention_calls: self.paged_attention_calls,
            paged_attention_fallbacks: self.paged_attention_fallbacks,
            ..MlxKVCacheUsage::default()
        };
        if let Some(pool) = self.fa_pool.as_ref() {
            let snapshot = pool.snapshot();
            usage.paged_pool_blocks_used = snapshot.allocated_blocks;
            usage.paged_pool_shared_blocks = snapshot.shared_blocks;
            usage.paged_pool_slabs = snapshot.slab_count;
            usage.paged_pool_slab_bytes = snapshot.slab_bytes;
            usage.paged_pool_slab_grow_events = snapshot.slab_grow_events;
        }

        for (layer_idx, fa) in self.layers.iter().enumerate() {
            let Some(fa) = fa else {
                continue;
            };
            if fa.rotating_window().is_some() {
                usage.rotated_ring_layers = usage.rotated_ring_layers.saturating_add(1);
            }
            if fa.as_quantized().is_some() {
                usage.quantized_layers = usage.quantized_layers.saturating_add(1);
            }
            // Quantized layers report packed + scales + biases bytes per
            // token; dense storages report the dense element count.
            let bytes_per_token = fa.bytes_per_token();
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

        if let Some(state) = &self.qwen4_exp {
            state.account_usage(&mut usage);
        }

        usage
    }

    /// Read the cached gated-delta states for a Qwen3.5 linear-attention layer.
    pub fn linear_state(&self, layer: usize) -> (Option<&MlxArray>, Option<&MlxArray>) {
        let state = &self.linear_layers[layer];
        (state.conv_state.as_ref(), state.recurrent_state.as_ref())
    }

    /// Replace only contiguous full-attention storage from a speculative
    /// branch while preserving this cache's logical length and linear state.
    ///
    /// A relaxed hybrid-Qwen verifier can safely retain its newly written K/V
    /// backing on every acceptance outcome: rejected tail rows are outside the
    /// source cache's unchanged logical boundary and the next append overwrites
    /// them. Releasing the superseded source handles before `eval` also makes
    /// MLX buffer donation possible instead of copying every capacity-sized K/V
    /// buffer merely to preserve an already-obsolete rollback copy.
    pub(crate) fn rebind_contiguous_attention_storage_from(&mut self, source: &Self) -> bool {
        if self.layers.len() != source.layers.len()
            || self.fa_pool.is_some()
            || source.fa_pool.is_some()
            || self.glm_mla_layers.iter().any(Option::is_some)
            || source.glm_mla_layers.iter().any(Option::is_some)
            || self.deepseek_v4_layers.iter().any(Option::is_some)
            || source.deepseek_v4_layers.iter().any(Option::is_some)
        {
            return false;
        }
        let compatible =
            self.layers
                .iter()
                .zip(&source.layers)
                .all(|(current, replacement)| match (current, replacement) {
                    (None, None) => true,
                    (
                        Some(FaLayerStorage::Contiguous(current)),
                        Some(FaLayerStorage::Contiguous(replacement)),
                    ) => {
                        current.n_kv_heads == replacement.n_kv_heads
                            && current.head_dim == replacement.head_dim
                            && current.dtype == replacement.dtype
                            && current.rotating_window.is_none()
                            && current.protected_prefix_ring.is_none()
                            && replacement.rotating_window.is_none()
                            && replacement.protected_prefix_ring.is_none()
                    }
                    _ => false,
                });
        if !compatible {
            return false;
        }
        self.layers.clone_from(&source.layers);
        true
    }

    /// Store the gated-delta states for a Qwen3.5 linear-attention layer.
    pub(crate) fn set_linear_state(
        &mut self,
        layer: usize,
        conv_state: MlxArray,
        recurrent_state: MlxArray,
    ) {
        let state = &mut self.linear_layers[layer];
        state.conv_state = Some(conv_state);
        state.recurrent_state = Some(recurrent_state);
        if self.linear_prefix_capture_after.is_none() {
            state.prefix_conv_state = None;
            state.prefix_recurrent_state = None;
        }
    }

    /// Begin one transient speculative-verifier checkpoint across every
    /// linear-attention layer. The capture is not part of clone or wire state.
    pub fn begin_linear_prefix_capture(&mut self, after_tokens: usize) {
        assert!(
            after_tokens > 0,
            "linear prefix capture must follow a token"
        );
        for state in &mut self.linear_layers {
            state.prefix_conv_state = None;
            state.prefix_recurrent_state = None;
            state.mtp_qkv = None;
            state.mtp_a = None;
            state.mtp_b = None;
            state.mtp_q = None;
            state.mtp_k = None;
            state.mtp_v = None;
            state.mtp_tape = None;
        }
        self.linear_prefix_capture_after = Some(after_tokens);
    }

    pub(crate) fn linear_prefix_capture_after(&self) -> Option<usize> {
        self.linear_prefix_capture_after
    }

    /// Transient prefix checkpoint for one linear-attention layer, if captured.
    #[allow(dead_code)]
    pub(crate) fn linear_prefix_refs(&self, layer: usize) -> Option<(&MlxArray, &MlxArray)> {
        let state = self.linear_layers.get(layer)?;
        Some((
            state.prefix_conv_state.as_ref()?,
            state.prefix_recurrent_state.as_ref()?,
        ))
    }

    /// Store the transient state requested by [`Self::begin_linear_prefix_capture`].
    pub(crate) fn set_linear_prefix_checkpoint(
        &mut self,
        layer: usize,
        conv_state: MlxArray,
        recurrent_state: MlxArray,
    ) {
        assert!(
            self.linear_prefix_capture_after.is_some(),
            "linear prefix checkpoint requires an active capture"
        );
        let state = &mut self.linear_layers[layer];
        state.prefix_conv_state = Some(conv_state);
        state.prefix_recurrent_state = Some(recurrent_state);
    }

    /// Retain the already-computed gated-delta projections for a possible
    /// partial-accept replay. These are transient graph leaves: clones,
    /// snapshots, and the wire format intentionally omit them.
    pub(crate) fn set_linear_mtp_projection_stash(
        &mut self,
        layer: usize,
        qkv: MlxArray,
        a: MlxArray,
        b: MlxArray,
    ) {
        assert!(
            self.linear_prefix_capture_after.is_some(),
            "linear MTP projection stash requires an active capture"
        );
        let state = &mut self.linear_layers[layer];
        state.mtp_qkv = Some(qkv);
        state.mtp_a = Some(a);
        state.mtp_b = Some(b);
        state.mtp_q = None;
        state.mtp_k = None;
        state.mtp_v = None;
        state.mtp_tape = None;
    }

    /// Retain post-convolution Q/K/V rows that the verifier recurrence has
    /// already consumed. Rejection replay can slice these materialized arrays
    /// instead of rebuilding convolution and Q/K normalization.
    pub(crate) fn set_linear_mtp_processed_stash(
        &mut self,
        layer: usize,
        q: MlxArray,
        k: MlxArray,
        v: MlxArray,
    ) {
        assert!(
            self.linear_prefix_capture_after.is_some(),
            "linear MTP processed stash requires an active capture"
        );
        let state = &mut self.linear_layers[layer];
        state.mtp_q = Some(q);
        state.mtp_k = Some(k);
        state.mtp_v = Some(v);
    }

    pub(crate) fn linear_mtp_processed_stash(
        &self,
        layer: usize,
    ) -> Option<(MlxArray, MlxArray, MlxArray)> {
        let state = self.linear_layers.get(layer)?;
        Some((
            state.mtp_q.as_ref()?.clone(),
            state.mtp_k.as_ref()?.clone(),
            state.mtp_v.as_ref()?.clone(),
        ))
    }

    /// Clone the transient verify projections so a replay can mutate the
    /// destination cache without retaining a borrow into it.
    pub(crate) fn linear_mtp_projection_stash(
        &self,
        layer: usize,
    ) -> Option<(MlxArray, MlxArray, MlxArray)> {
        let state = self.linear_layers.get(layer)?;
        Some((
            state.mtp_qkv.as_ref()?.clone(),
            state.mtp_a.as_ref()?.clone(),
            state.mtp_b.as_ref()?.clone(),
        ))
    }

    /// Retain the whole-verifier replay leaves. `qkv` rebuilds the accepted
    /// convolution state and normalized K; `a` rebuilds decay; `tape` carries
    /// the already-computed per-token delta.
    pub(crate) fn set_linear_mtp_tape_stash(
        &mut self,
        layer: usize,
        qkv: MlxArray,
        a: MlxArray,
        tape: MlxArray,
    ) {
        assert!(
            self.linear_prefix_capture_after.is_some(),
            "linear MTP tape stash requires an active capture"
        );
        let state = &mut self.linear_layers[layer];
        state.mtp_qkv = Some(qkv);
        state.mtp_a = Some(a);
        state.mtp_b = None;
        state.mtp_q = None;
        state.mtp_k = None;
        state.mtp_v = None;
        state.mtp_tape = Some(tape);
    }

    pub(crate) fn linear_mtp_tape_stash(
        &self,
        layer: usize,
    ) -> Option<(MlxArray, MlxArray, MlxArray)> {
        let state = self.linear_layers.get(layer)?;
        Some((
            state.mtp_qkv.as_ref()?.clone(),
            state.mtp_a.as_ref()?.clone(),
            state.mtp_tape.as_ref()?.clone(),
        ))
    }

    /// Replace final verifier states with the captured committed-prefix states.
    ///
    /// Returns false without mutating any layer when the capture is incomplete.
    #[must_use]
    pub fn restore_linear_prefix_checkpoint(&mut self) -> bool {
        if self.linear_prefix_capture_after.is_none() {
            return false;
        }
        let complete = self.linear_layers.iter().all(|state| {
            let active = state.conv_state.is_some() || state.recurrent_state.is_some();
            !active || (state.prefix_conv_state.is_some() && state.prefix_recurrent_state.is_some())
        });
        if !complete {
            return false;
        }
        for state in &mut self.linear_layers {
            if state.conv_state.is_some() || state.recurrent_state.is_some() {
                state.conv_state = state.prefix_conv_state.take();
                state.recurrent_state = state.prefix_recurrent_state.take();
            }
            state.mtp_qkv = None;
            state.mtp_a = None;
            state.mtp_b = None;
            state.mtp_q = None;
            state.mtp_k = None;
            state.mtp_v = None;
            state.mtp_tape = None;
        }
        self.linear_prefix_capture_after = None;
        true
    }

    /// Drop a transient verifier checkpoint after the final state is committed.
    pub(crate) fn clear_linear_prefix_checkpoint(&mut self) {
        for state in &mut self.linear_layers {
            state.prefix_conv_state = None;
            state.prefix_recurrent_state = None;
            state.mtp_qkv = None;
            state.mtp_a = None;
            state.mtp_b = None;
            state.mtp_q = None;
            state.mtp_k = None;
            state.mtp_v = None;
            state.mtp_tape = None;
        }
        self.linear_prefix_capture_after = None;
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
    pub(crate) fn peek_source_kv(
        &self,
        source_layer: usize,
        new_tokens: usize,
    ) -> (MlxArray, MlxArray) {
        let fa = self.layers[source_layer]
            .as_ref()
            .expect("KV-shared source layer has no cached KV — source layer must appear earlier");
        match fa {
            FaLayerStorage::Contiguous(lkv) => {
                if let Some(ring) = &lkv.protected_prefix_ring {
                    if let (Some(k), Some(v)) = (&lkv.last_k_view, &lkv.last_v_view) {
                        return (k.clone(), v.clone());
                    }
                    let decoded = self.seq_len.saturating_sub(ring.prefix_len);
                    let end = (ring.prefix_len + decoded.min(ring.window)) as i32;
                    let stop = [1, lkv.n_kv_heads, end, lkv.head_dim];
                    return (
                        slice(&lkv.k, &[0, 0, 0, 0], &stop, &[1, 1, 1, 1], None),
                        slice(&lkv.v, &[0, 0, 0, 0], &stop, &[1, 1, 1, 1], None),
                    );
                }
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
                _ => paged.materialize(
                    self.fa_pool
                        .as_ref()
                        .expect("paged source view requires pool"),
                    0,
                    self.seq_len + new_tokens,
                ),
            },
            FaLayerStorage::Quantized(quantized) => {
                match (&quantized.last_k_view, &quantized.last_v_view) {
                    (Some(k), Some(v)) => (k.clone(), v.clone()),
                    // Fallback mirrors the dense one: dequantize the logical
                    // prefix through the current forward's write end.
                    _ => quantized.dense_view(0, self.seq_len + new_tokens),
                }
            }
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
                if let Some(ring) = &lkv.protected_prefix_ring {
                    if let (Some(k), Some(v)) = (&lkv.last_k_view, &lkv.last_v_view) {
                        return Some((k.clone(), v.clone()));
                    }
                    let decoded = self.seq_len.saturating_sub(ring.prefix_len);
                    let end = (ring.prefix_len + decoded.min(ring.window)) as i32;
                    let stop = [1, lkv.n_kv_heads, end, lkv.head_dim];
                    return Some((
                        slice(&lkv.k, &[0, 0, 0, 0], &stop, &[1, 1, 1, 1], None),
                        slice(&lkv.v, &[0, 0, 0, 0], &stop, &[1, 1, 1, 1], None),
                    ));
                }
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
                _ => Some(
                    paged.materialize(
                        self.fa_pool
                            .as_ref()
                            .expect("paged layer view requires pool"),
                        0,
                        self.seq_len,
                    ),
                ),
            },
            FaLayerStorage::Quantized(quantized) => {
                let views = match (&quantized.last_k_view, &quantized.last_v_view) {
                    (Some(k), Some(v)) => (k.clone(), v.clone()),
                    _ => quantized.dense_view(0, self.seq_len),
                };
                Some(views)
            }
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
    pub(crate) fn layer_sliding_ring(&self, layer: usize) -> Option<SlidingRingLayout> {
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
    pub(crate) fn peek_layer_full_kv(&self, layer: usize) -> Option<(MlxArray, MlxArray)> {
        let fa = self.layers.get(layer)?.as_ref()?;
        match fa {
            FaLayerStorage::Contiguous(lkv) => {
                assert!(
                    lkv.protected_prefix_ring.is_none(),
                    "a protected-prefix ring has no contiguous full logical KV view"
                );
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
            FaLayerStorage::Paged(paged) => Some(
                paged.materialize(
                    self.fa_pool
                        .as_ref()
                        .expect("paged full view requires pool"),
                    0,
                    self.seq_len,
                ),
            ),
            FaLayerStorage::Quantized(quantized) => Some(quantized.dense_view(0, self.seq_len)),
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
    pub(crate) fn replace_layer_kv(&mut self, layer: usize, new_k: MlxArray, new_v: MlxArray) {
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
            && let Some(pool) = self.fa_pool.as_ref()
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
                protected_prefix_ring: None,
                dtype,
            }));
        }
    }

    /// Reset cache entirely (e.g., between requests).
    pub(crate) fn reset(&mut self) {
        self.qwen4_exp = None;
        if let Some(pool) = self.fa_pool.as_ref() {
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
        for entry in &mut self.deepseek_v4_layers {
            *entry = None;
        }
        for state in &mut self.linear_layers {
            *state = LinearLayerState::default();
        }
        self.linear_prefix_capture_after = None;
        self.seq_len = 0;
        self.rope_offset = 0;
        self.mrope_position_delta = 0;
        self.growth_count = 0;
        // A hard-cap trip from the previous occupant must not leak into the
        // next one: the flag is per-occupancy state, and a reset cache gets
        // a fresh budget attempt (a real over-cap append re-sets it).
        self.hard_cap_exhausted = false;
    }
}

#[cfg(test)]
mod tests;
