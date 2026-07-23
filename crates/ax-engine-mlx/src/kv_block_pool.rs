//! FA-only physical KV block pool.
//!
//! Design: `docs/designs/kv-weak-surfaces-2026-07-14.md` Track C / PR4.
//!
//! This module is the **logical allocator** for fixed-size token blocks. Live
//! FA storage and SDPA materialization live in [`crate::kv_cache::MlxKVCache`]
//! when `AX_MLX_FA_KV_BLOCK_POOL=1` (default OFF). With the flag off, production
//! keeps contiguous per-request buffers.
//!
//! Scope (ADR-006, default OFF):
//! - private or runner-shared full-attention blocks
//! - transactional refcount retain/release and block-level CoW
//! - production configs are request-level fail-closed at exhaustion; direct
//!   tests may construct a fail-soft config
//! - private `max_blocks` follows `KvManagerConfig.total_blocks`; sharing mode
//!   uses layer-block slots (see ADR-006)
//! - FA append/trim/materialize wired behind the flag (see `MlxKVCache`)
//! - runner-shared fixed per-layer K/V slabs and gather-based SDPA
//!
//! Non-goals (later phases):
//! - MLA latent/k_pe paired blocks
//! - MLA/recurrent native page layouts

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;

use mlx_sys::{
    MlxArray, MlxDtype, concatenate, reshape, slice, slice_update, take, transpose, zeros,
};

fn env_flag_enabled(raw: Option<&str>) -> bool {
    raw.is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

/// Opt-in flag for FA private block-pool path in [`crate::kv_cache::MlxKVCache`].
/// Default: OFF.
pub fn fa_kv_block_pool_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let raw = std::env::var("AX_MLX_FA_KV_BLOCK_POOL").ok();
        env_flag_enabled(raw.as_deref())
    })
}

/// Second opt-in gate for one runner-wide FA pool plus native prefix sharing.
/// The caller must also require [`fa_kv_block_pool_enabled`]. Default: OFF.
pub fn fa_kv_block_sharing_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let raw = std::env::var("AX_MLX_FA_KV_BLOCK_SHARING").ok();
        env_flag_enabled(raw.as_deref())
    })
}

/// Third opt-in gate for the diagnostic native block-table kernel.
/// The runner also requires the base pool and sharing flags. Default: OFF.
pub fn fa_native_paged_attention_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let raw = std::env::var("AX_MLX_FA_NATIVE_PAGED_ATTENTION").ok();
        env_flag_enabled(raw.as_deref())
    })
}

/// Default private-pool geometry when the env flag is on.
///
/// `block_size_tokens` matches the usual `KvManager` block size (16).
/// `max_blocks` defaults to 8192 (131072 tokens) and can be overridden with
/// `AX_MLX_FA_KV_BLOCK_POOL_MAX_BLOCKS`; either way `max_blocks` is a real
/// memory budget (this fallback or, once `MlxRunner::align_fa_block_pool_to_kv`
/// runs, `KvManager.total_blocks`), so exhaustion always fails the request
/// instead of demoting to unbounded contiguous growth (see
/// `FaBlockPoolConfig::hard_cap`).
pub(crate) fn parse_fa_block_pool_max_blocks_override(raw: Option<&str>) -> Option<u32> {
    raw.and_then(|value| value.parse::<u32>().ok())
        .filter(|&value| value > 0)
}

pub(crate) fn fa_block_pool_max_blocks_override() -> Option<u32> {
    let raw = std::env::var("AX_MLX_FA_KV_BLOCK_POOL_MAX_BLOCKS").ok();
    parse_fa_block_pool_max_blocks_override(raw.as_deref())
}

pub fn default_fa_block_pool_config() -> FaBlockPoolConfig {
    let max_blocks = fa_block_pool_max_blocks_override().unwrap_or(8192);
    FaBlockPoolConfig {
        block_size_tokens: 16,
        max_blocks,
        hard_cap: true,
    }
}

/// Logical physical-block identifier (pool-local, not a GPU pointer).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PhysicalBlockId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FaBlockPoolConfig {
    pub block_size_tokens: u32,
    pub max_blocks: u32,
    /// When true, `max_blocks` is treated as a real memory budget:
    /// exhaustion fails the request instead of demoting to unbounded
    /// contiguous growth. Both production config paths
    /// (`default_fa_block_pool_config`, `MlxRunner::align_fa_block_pool_to_kv`)
    /// always set this — `max_blocks` is either the operator's explicit
    /// `AX_MLX_FA_KV_BLOCK_POOL_MAX_BLOCKS` or `KvManager.total_blocks`, and
    /// exceeding either is a real capacity problem, not just an
    /// under-sized scaffold default. `false` is reachable only via a
    /// directly-constructed `FaBlockPoolConfig` (tests exercise the
    /// fail-soft demotion path this way).
    pub hard_cap: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FaBlockPoolError {
    InvalidConfig(&'static str),
    Exhausted { requested: u32, available: u32 },
    UnknownBlock(PhysicalBlockId),
    UnallocatedBlock(PhysicalBlockId),
    DuplicateBlock(PhysicalBlockId),
    RefCountOverflow(PhysicalBlockId),
    DoubleFree(PhysicalBlockId),
}

impl std::fmt::Display for FaBlockPoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid FA block pool config: {msg}"),
            Self::Exhausted {
                requested,
                available,
            } => write!(
                f,
                "FA block pool exhausted: requested {requested}, available {available}"
            ),
            Self::UnknownBlock(id) => write!(f, "unknown physical block {}", id.0),
            Self::UnallocatedBlock(id) => write!(f, "physical block {} is not allocated", id.0),
            Self::DuplicateBlock(id) => write!(f, "duplicate physical block {}", id.0),
            Self::RefCountOverflow(id) => {
                write!(f, "reference count overflow for physical block {}", id.0)
            }
            Self::DoubleFree(id) => write!(f, "double free of physical block {}", id.0),
        }
    }
}

impl std::error::Error for FaBlockPoolError {}

/// FA block allocator with transactional reference counting.
#[derive(Debug)]
pub struct FaBlockPool {
    config: FaBlockPoolConfig,
    free: VecDeque<PhysicalBlockId>,
    /// Zero means free; positive values count paged views owning the block.
    ref_counts: Vec<u32>,
    allocated_count: u32,
    /// Allocated IDs whose refcount is greater than one. Maintained
    /// incrementally so per-step telemetry never scans the whole pool.
    shared_count: u32,
    cow_copies: u64,
}

impl FaBlockPool {
    pub fn new(config: FaBlockPoolConfig) -> Result<Self, FaBlockPoolError> {
        if config.block_size_tokens == 0 {
            return Err(FaBlockPoolError::InvalidConfig(
                "block_size_tokens must be > 0",
            ));
        }
        if config.max_blocks == 0 {
            return Err(FaBlockPoolError::InvalidConfig("max_blocks must be > 0"));
        }
        let free = (0..config.max_blocks)
            .rev()
            .map(PhysicalBlockId)
            .collect::<VecDeque<_>>();
        Ok(Self {
            config,
            free,
            ref_counts: vec![0; config.max_blocks as usize],
            allocated_count: 0,
            shared_count: 0,
            cow_copies: 0,
        })
    }

    pub fn config(&self) -> FaBlockPoolConfig {
        self.config
    }

    pub fn available_blocks(&self) -> u32 {
        self.free.len() as u32
    }

    pub fn allocated_blocks(&self) -> u32 {
        self.allocated_count
    }

    pub fn shared_blocks(&self) -> u32 {
        self.shared_count
    }

    pub fn cow_copies(&self) -> u64 {
        self.cow_copies
    }

    pub fn ref_count(&self, id: PhysicalBlockId) -> Result<u32, FaBlockPoolError> {
        self.ref_counts
            .get(id.0 as usize)
            .copied()
            .ok_or(FaBlockPoolError::UnknownBlock(id))
    }

    /// Allocate `n` blocks. Fail-closed: no partial allocation.
    pub fn allocate(&mut self, n: u32) -> Result<Vec<PhysicalBlockId>, FaBlockPoolError> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let available = self.available_blocks();
        if available < n {
            return Err(FaBlockPoolError::Exhausted {
                requested: n,
                available,
            });
        }
        let mut out = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let id = self.free.pop_back().expect("availability checked");
            let idx = id.0 as usize;
            debug_assert_eq!(self.ref_counts[idx], 0);
            self.ref_counts[idx] = 1;
            self.allocated_count = self.allocated_count.saturating_add(1);
            out.push(id);
        }
        Ok(out)
    }

    /// Retain one ownership reference for every distinct allocated ID.
    /// Validation is complete before any count changes.
    pub fn retain(&mut self, ids: &[PhysicalBlockId]) -> Result<(), FaBlockPoolError> {
        let mut seen = HashSet::with_capacity(ids.len());
        for id in ids {
            let idx = id.0 as usize;
            let Some(&ref_count) = self.ref_counts.get(idx) else {
                return Err(FaBlockPoolError::UnknownBlock(*id));
            };
            if !seen.insert(*id) {
                return Err(FaBlockPoolError::DuplicateBlock(*id));
            }
            if ref_count == 0 {
                return Err(FaBlockPoolError::UnallocatedBlock(*id));
            }
            if ref_count == u32::MAX {
                return Err(FaBlockPoolError::RefCountOverflow(*id));
            }
        }
        for id in ids {
            let ref_count = &mut self.ref_counts[id.0 as usize];
            if *ref_count == 1 {
                self.shared_count = self.shared_count.saturating_add(1);
            }
            *ref_count += 1;
        }
        Ok(())
    }

    /// Release ownership references. Unknown or double-free is fail-closed.
    ///
    /// Duplicate IDs in a single call are treated as double-free so the free
    /// list cannot gain the same block twice (and `allocated_count` cannot
    /// under-count).
    pub fn free(&mut self, ids: &[PhysicalBlockId]) -> Result<(), FaBlockPoolError> {
        // Validate all first so free is atomic.
        let mut seen = HashSet::with_capacity(ids.len());
        for id in ids {
            let idx = id.0 as usize;
            if idx >= self.ref_counts.len() {
                return Err(FaBlockPoolError::UnknownBlock(*id));
            }
            if self.ref_counts[idx] == 0 || !seen.insert(*id) {
                return Err(FaBlockPoolError::DoubleFree(*id));
            }
        }
        for id in ids {
            let idx = id.0 as usize;
            if self.ref_counts[idx] == 2 {
                self.shared_count = self.shared_count.saturating_sub(1);
            }
            self.ref_counts[idx] -= 1;
            if self.ref_counts[idx] == 0 {
                self.allocated_count = self.allocated_count.saturating_sub(1);
                self.free.push_back(*id);
            }
        }
        Ok(())
    }

    /// Return a unique ID for one owner of `id`.
    ///
    /// Refcount 1 is already unique. A shared block consumes one free block,
    /// moves the caller's reference to it, and leaves other owners on `id`.
    /// Exhaustion is atomic: no reference count changes.
    pub fn make_unique(
        &mut self,
        id: PhysicalBlockId,
    ) -> Result<(PhysicalBlockId, bool), FaBlockPoolError> {
        let idx = id.0 as usize;
        let Some(&ref_count) = self.ref_counts.get(idx) else {
            return Err(FaBlockPoolError::UnknownBlock(id));
        };
        if ref_count == 0 {
            return Err(FaBlockPoolError::UnallocatedBlock(id));
        }
        if ref_count == 1 {
            return Ok((id, false));
        }
        let available = self.available_blocks();
        let Some(replacement) = self.free.pop_back() else {
            return Err(FaBlockPoolError::Exhausted {
                requested: 1,
                available,
            });
        };
        self.ref_counts[replacement.0 as usize] = 1;
        if ref_count == 2 {
            self.shared_count = self.shared_count.saturating_sub(1);
        }
        self.ref_counts[idx] -= 1;
        self.allocated_count = self.allocated_count.saturating_add(1);
        self.cow_copies = self.cow_copies.saturating_add(1);
        Ok((replacement, true))
    }

    /// Batch form of [`Self::make_unique`]. Validation and capacity checks are
    /// completed before any owner moves, so multi-block append preparation is
    /// all-or-nothing at the ownership layer.
    pub fn make_unique_many(
        &mut self,
        ids: &[PhysicalBlockId],
    ) -> Result<Vec<(PhysicalBlockId, bool)>, FaBlockPoolError> {
        let mut seen = HashSet::with_capacity(ids.len());
        let mut replacements = 0u32;
        for id in ids {
            let Some(&ref_count) = self.ref_counts.get(id.0 as usize) else {
                return Err(FaBlockPoolError::UnknownBlock(*id));
            };
            if !seen.insert(*id) {
                return Err(FaBlockPoolError::DuplicateBlock(*id));
            }
            if ref_count == 0 {
                return Err(FaBlockPoolError::UnallocatedBlock(*id));
            }
            if ref_count > 1 {
                replacements = replacements.saturating_add(1);
            }
        }
        let available = self.available_blocks();
        if available < replacements {
            return Err(FaBlockPoolError::Exhausted {
                requested: replacements,
                available,
            });
        }

        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            let idx = id.0 as usize;
            let ref_count = self.ref_counts[idx];
            if ref_count == 1 {
                out.push((*id, false));
                continue;
            }
            let replacement = self
                .free
                .pop_back()
                .expect("batch replacement capacity was validated");
            self.ref_counts[replacement.0 as usize] = 1;
            if ref_count == 2 {
                self.shared_count = self.shared_count.saturating_sub(1);
            }
            self.ref_counts[idx] -= 1;
            self.allocated_count = self.allocated_count.saturating_add(1);
            self.cow_copies = self.cow_copies.saturating_add(1);
            out.push((replacement, true));
        }
        Ok(out)
    }

    /// Tokens represented by `n` full blocks.
    pub fn tokens_for_blocks(&self, n: u32) -> u32 {
        n.saturating_mul(self.config.block_size_tokens)
    }

    /// Blocks required to hold `tokens` (ceil).
    pub fn blocks_for_tokens(&self, tokens: u32) -> u32 {
        if tokens == 0 {
            return 0;
        }
        tokens.div_ceil(self.config.block_size_tokens)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FaBlockPoolSnapshot {
    pub config: FaBlockPoolConfig,
    pub available_blocks: u32,
    pub allocated_blocks: u32,
    pub shared_blocks: u32,
    pub cow_copies: u64,
    /// Fixed-size K/V slabs currently allocated across every layer.
    pub slab_count: u32,
    /// Real MLX bytes reserved by those slabs (K + V, including slack rows).
    pub slab_bytes: u64,
    /// Episodes that appended one or more slabs after a layer's first slab.
    pub slab_grow_events: u64,
}

/// mlxcel's proven allocation granularity: a layer grows by appending one
/// small slab, never by replacing a tensor sized to the whole logical pool.
pub(crate) const FA_POOL_SLAB_BLOCKS: usize = 32;

/// One native-attention-compatible slab snapshot. Native dispatch is allowed
/// only when all requested rows live in this same slab; the normal serving
/// path gathers across any number of slabs and runs MLX SDPA.
#[derive(Clone, Debug)]
pub(crate) struct FaSingleSlabSnapshot {
    pub(crate) k: MlxArray,
    pub(crate) v: MlxArray,
    pub(crate) local_rows: Vec<u32>,
    pub(crate) n_kv_heads: i32,
    pub(crate) head_dim: i32,
    pub(crate) block_size: usize,
    pub(crate) dtype: MlxDtype,
}

#[derive(Debug)]
struct FaSlab {
    /// `Option` permits a donation-safe move around `slice_update`: the old
    /// array handle is removed before the update, matching mlxcel #237.
    k: Option<MlxArray>,
    v: Option<MlxArray>,
}

#[derive(Debug)]
struct FaLayerSlabArena {
    n_kv_heads: i32,
    head_dim: i32,
    block_size: usize,
    dtype: MlxDtype,
    slabs: Vec<FaSlab>,
    block_rows: HashMap<PhysicalBlockId, usize>,
    free_rows: Vec<usize>,
    next_row: usize,
    grow_events: u64,
}

impl FaLayerSlabArena {
    fn new(
        block_size: usize,
        n_kv_heads: i32,
        head_dim: i32,
        dtype: MlxDtype,
        min_capacity: usize,
    ) -> Self {
        let mut out = Self {
            n_kv_heads,
            head_dim,
            block_size,
            dtype,
            slabs: Vec::new(),
            block_rows: HashMap::new(),
            free_rows: Vec::new(),
            next_row: 0,
            grow_events: 0,
        };
        out.append_slabs(min_capacity.max(1));
        out
    }

    fn validate_geometry(
        &self,
        block_size: usize,
        n_kv_heads: i32,
        head_dim: i32,
        dtype: MlxDtype,
    ) -> Result<(), FaBlockPoolError> {
        if self.block_size != block_size
            || self.n_kv_heads != n_kv_heads
            || self.head_dim != head_dim
            || self.dtype != dtype
        {
            return Err(FaBlockPoolError::InvalidConfig(
                "FA slab geometry changed within one layer",
            ));
        }
        Ok(())
    }

    fn capacity_rows(&self) -> usize {
        self.slabs.len().saturating_mul(FA_POOL_SLAB_BLOCKS)
    }

    fn append_slabs(&mut self, min_capacity: usize) {
        let target = min_capacity.div_ceil(FA_POOL_SLAB_BLOCKS).max(1);
        let shape = [
            FA_POOL_SLAB_BLOCKS as i32,
            self.block_size as i32,
            self.n_kv_heads,
            self.head_dim,
        ];
        while self.slabs.len() < target {
            self.slabs.push(FaSlab {
                k: Some(zeros(&shape, self.dtype, None)),
                v: Some(zeros(&shape, self.dtype, None)),
            });
        }
    }

    fn ensure_capacity(&mut self, min_capacity: usize) {
        if min_capacity <= self.capacity_rows() {
            return;
        }
        self.append_slabs(min_capacity);
        self.grow_events = self.grow_events.saturating_add(1);
    }

    fn presize_for_ids(&mut self, ids: &[PhysicalBlockId]) {
        let unassigned = ids
            .iter()
            .filter(|id| !self.block_rows.contains_key(id))
            .count();
        let minted = unassigned.saturating_sub(self.free_rows.len());
        self.ensure_capacity(self.next_row.saturating_add(minted));
    }

    fn assign_row(&mut self, id: PhysicalBlockId) -> usize {
        if let Some(&row) = self.block_rows.get(&id) {
            return row;
        }
        let row = self.free_rows.pop().unwrap_or_else(|| {
            let row = self.next_row;
            self.next_row = self.next_row.saturating_add(1);
            row
        });
        self.ensure_capacity(row.saturating_add(1));
        self.block_rows.insert(id, row);
        row
    }

    fn release_row(&mut self, id: PhysicalBlockId) {
        if let Some(row) = self.block_rows.remove(&id) {
            self.free_rows.push(row);
        }
    }

    fn slab_bytes(&self) -> u64 {
        self.slabs.iter().fold(0u64, |bytes, slab| {
            bytes
                .saturating_add(slab.k.as_ref().map_or(0, |array| array.nbytes()) as u64)
                .saturating_add(slab.v.as_ref().map_or(0, |array| array.nbytes()) as u64)
        })
    }
}

#[derive(Debug, Default)]
struct FaSlabStorage {
    layers: HashMap<usize, FaLayerSlabArena>,
    /// A global block id belongs to exactly one layer while it has a row.
    block_layers: HashMap<PhysicalBlockId, usize>,
}

impl FaSlabStorage {
    fn ensure_layer(
        &mut self,
        layer_idx: usize,
        block_size: usize,
        n_kv_heads: i32,
        head_dim: i32,
        dtype: MlxDtype,
        ids: &[PhysicalBlockId],
    ) -> Result<(), FaBlockPoolError> {
        if n_kv_heads <= 0 || head_dim <= 0 {
            return Err(FaBlockPoolError::InvalidConfig(
                "FA slab dimensions must be positive",
            ));
        }
        if !matches!(
            dtype,
            MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
        ) {
            return Err(FaBlockPoolError::InvalidConfig(
                "FA slabs support bf16/f16/f32 only",
            ));
        }
        for id in ids {
            if self
                .block_layers
                .get(id)
                .is_some_and(|owner| *owner != layer_idx)
            {
                return Err(FaBlockPoolError::InvalidConfig(
                    "physical FA block was assigned to multiple layers",
                ));
            }
        }
        if let Some(arena) = self.layers.get_mut(&layer_idx) {
            arena.validate_geometry(block_size, n_kv_heads, head_dim, dtype)?;
            arena.presize_for_ids(ids);
        } else {
            self.layers.insert(
                layer_idx,
                FaLayerSlabArena::new(block_size, n_kv_heads, head_dim, dtype, ids.len().max(1)),
            );
        }
        Ok(())
    }

    fn assign_row(
        &mut self,
        layer_idx: usize,
        id: PhysicalBlockId,
    ) -> Result<usize, FaBlockPoolError> {
        if self
            .block_layers
            .get(&id)
            .is_some_and(|owner| *owner != layer_idx)
        {
            return Err(FaBlockPoolError::InvalidConfig(
                "physical FA block was assigned to multiple layers",
            ));
        }
        let row = self
            .layers
            .get_mut(&layer_idx)
            .ok_or(FaBlockPoolError::InvalidConfig(
                "FA layer slab is not initialized",
            ))?
            .assign_row(id);
        self.block_layers.insert(id, layer_idx);
        Ok(row)
    }

    fn release_ids(&mut self, ids: &[PhysicalBlockId]) {
        for id in ids {
            if let Some(layer_idx) = self.block_layers.remove(id)
                && let Some(arena) = self.layers.get_mut(&layer_idx)
            {
                arena.release_row(*id);
            }
        }
    }
}

fn update_slab_tensor(
    slot: &mut Option<MlxArray>,
    update: &MlxArray,
    start: &[i32],
    stop: &[i32],
) -> Result<(), FaBlockPoolError> {
    let old = slot.take().ok_or(FaBlockPoolError::InvalidConfig(
        "FA slab tensor was moved without replacement",
    ))?;
    *slot = Some(slice_update(&old, update, start, stop, &[1, 1, 1, 1], None));
    Ok(())
}

/// Cloneable synchronized handle used for both private and runner-wide pools.
#[derive(Clone, Debug)]
pub struct SharedFaBlockPool {
    inner: Arc<Mutex<FaBlockPool>>,
    /// Present for runner-wide sharing. Storage is per-layer, fixed at 32 rows
    /// per slab, and grows only by appending slabs.
    slab_storage: Option<Arc<Mutex<FaSlabStorage>>>,
    /// Diagnostic-only custom kernel gate. Gather + MLX SDPA remains default.
    native_attention: bool,
}

impl SharedFaBlockPool {
    pub fn new(config: FaBlockPoolConfig) -> Result<Self, FaBlockPoolError> {
        Ok(Self {
            inner: Arc::new(Mutex::new(FaBlockPool::new(config)?)),
            slab_storage: None,
            native_attention: false,
        })
    }

    pub fn new_with_slab_storage(config: FaBlockPoolConfig) -> Result<Self, FaBlockPoolError> {
        Ok(Self {
            inner: Arc::new(Mutex::new(FaBlockPool::new(config)?)),
            slab_storage: Some(Arc::new(Mutex::new(FaSlabStorage::default()))),
            native_attention: false,
        })
    }

    pub fn new_with_native_slab_storage(
        config: FaBlockPoolConfig,
    ) -> Result<Self, FaBlockPoolError> {
        Ok(Self {
            inner: Arc::new(Mutex::new(FaBlockPool::new(config)?)),
            slab_storage: Some(Arc::new(Mutex::new(FaSlabStorage::default()))),
            native_attention: true,
        })
    }

    pub fn same_pool(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub fn slab_storage_enabled(&self) -> bool {
        self.slab_storage.is_some()
    }

    pub fn native_attention_enabled(&self) -> bool {
        self.native_attention
    }

    pub(crate) fn ensure_layer_slab_storage(
        &self,
        layer_idx: usize,
        n_kv_heads: i32,
        head_dim: i32,
        dtype: MlxDtype,
        ids: &[PhysicalBlockId],
    ) -> Result<(), FaBlockPoolError> {
        let Some(storage) = self.slab_storage.as_ref() else {
            return Err(FaBlockPoolError::InvalidConfig(
                "FA slab storage is not enabled",
            ));
        };
        let config = self.config();
        storage.lock().ensure_layer(
            layer_idx,
            config.block_size_tokens as usize,
            n_kv_heads,
            head_dim,
            dtype,
            ids,
        )
    }

    /// Copy complete K/V blocks after allocator COW moved an owner to new IDs.
    pub(crate) fn copy_slab_blocks(
        &self,
        layer_idx: usize,
        copies: &[(PhysicalBlockId, PhysicalBlockId)],
    ) -> Result<(), FaBlockPoolError> {
        if copies.is_empty() {
            return Ok(());
        }
        let Some(storage) = self.slab_storage.as_ref() else {
            return Err(FaBlockPoolError::InvalidConfig(
                "FA slab storage is not enabled",
            ));
        };
        let mut storage = storage.lock();
        let (block_size, n_kv_heads, head_dim, dtype) = {
            let arena = storage
                .layers
                .get(&layer_idx)
                .ok_or(FaBlockPoolError::InvalidConfig(
                    "FA layer slab is not initialized",
                ))?;
            (
                arena.block_size,
                arena.n_kv_heads,
                arena.head_dim,
                arena.dtype,
            )
        };
        let targets: Vec<PhysicalBlockId> = copies.iter().map(|(_, target)| *target).collect();
        storage.ensure_layer(layer_idx, block_size, n_kv_heads, head_dim, dtype, &targets)?;
        for &(source, target) in copies {
            let source_row = *storage
                .layers
                .get(&layer_idx)
                .and_then(|arena| arena.block_rows.get(&source))
                .ok_or(FaBlockPoolError::UnallocatedBlock(source))?;
            let target_row = storage.assign_row(layer_idx, target)?;
            let (source_slab, source_local) = (
                source_row / FA_POOL_SLAB_BLOCKS,
                (source_row % FA_POOL_SLAB_BLOCKS) as i32,
            );
            let (target_slab, target_local) = (
                target_row / FA_POOL_SLAB_BLOCKS,
                (target_row % FA_POOL_SLAB_BLOCKS) as i32,
            );
            let src_start = [source_local, 0, 0, 0];
            let src_stop = [source_local + 1, block_size as i32, n_kv_heads, head_dim];
            let dst_start = [target_local, 0, 0, 0];
            let dst_stop = [target_local + 1, block_size as i32, n_kv_heads, head_dim];
            let strides = [1, 1, 1, 1];
            let (k_block, v_block) = {
                let arena = storage.layers.get(&layer_idx).expect("validated layer");
                let source = &arena.slabs[source_slab];
                (
                    slice(
                        source.k.as_ref().expect("live K slab"),
                        &src_start,
                        &src_stop,
                        &strides,
                        None,
                    ),
                    slice(
                        source.v.as_ref().expect("live V slab"),
                        &src_start,
                        &src_stop,
                        &strides,
                        None,
                    ),
                )
            };
            let arena = storage.layers.get_mut(&layer_idx).expect("validated layer");
            update_slab_tensor(
                &mut arena.slabs[target_slab].k,
                &k_block,
                &dst_start,
                &dst_stop,
            )?;
            update_slab_tensor(
                &mut arena.slabs[target_slab].v,
                &v_block,
                &dst_start,
                &dst_stop,
            )?;
        }
        Ok(())
    }

    /// Write one logical token range into fixed per-layer slabs. The full span
    /// is presized once, then every `slice_update` moves the destination handle
    /// out first so MLX may donate the buffer.
    pub(crate) fn write_slab_tokens(
        &self,
        layer_idx: usize,
        block_ids: &[PhysicalBlockId],
        write_start: usize,
        new_k: &MlxArray,
        new_v: &MlxArray,
    ) -> Result<(), FaBlockPoolError> {
        let Some(storage) = self.slab_storage.as_ref() else {
            return Err(FaBlockPoolError::InvalidConfig(
                "FA slab storage is not enabled",
            ));
        };
        let shape = new_k.shape();
        if shape.len() != 4
            || shape != new_v.shape()
            || shape[0] != 1
            || new_k.dtype() != new_v.dtype()
        {
            return Err(FaBlockPoolError::InvalidConfig(
                "FA slab write requires matching [1,H,T,D] K/V",
            ));
        }
        let new_tokens = shape[2] as usize;
        if new_tokens == 0 {
            return Ok(());
        }
        let write_end = write_start.saturating_add(new_tokens);
        let block_size = self.config().block_size_tokens as usize;
        let first_block = write_start / block_size;
        let last_block = (write_end - 1) / block_size;
        let touched =
            block_ids
                .get(first_block..=last_block)
                .ok_or(FaBlockPoolError::InvalidConfig(
                    "FA slab write exceeds block table",
                ))?;
        let mut storage = storage.lock();
        storage.ensure_layer(
            layer_idx,
            block_size,
            shape[1],
            shape[3],
            new_k.dtype(),
            touched,
        )?;
        let rows = touched
            .iter()
            .map(|id| storage.assign_row(layer_idx, *id))
            .collect::<Result<Vec<_>, _>>()?;
        let k_slots = transpose(new_k, &[0, 2, 1, 3], None);
        let v_slots = transpose(new_v, &[0, 2, 1, 3], None);
        let mut source_token = 0usize;
        let mut logical_token = write_start;
        while logical_token < write_end {
            let logical_block = logical_token / block_size;
            let block_offset = logical_token % block_size;
            let count = (write_end - logical_token).min(block_size - block_offset);
            let row = rows[logical_block - first_block];
            let slab_idx = row / FA_POOL_SLAB_BLOCKS;
            let local_row = (row % FA_POOL_SLAB_BLOCKS) as i32;
            let src_start = [0, source_token as i32, 0, 0];
            let src_stop = [1, (source_token + count) as i32, shape[1], shape[3]];
            let dst_start = [local_row, block_offset as i32, 0, 0];
            let dst_stop = [
                local_row + 1,
                (block_offset + count) as i32,
                shape[1],
                shape[3],
            ];
            let strides = [1, 1, 1, 1];
            let k_segment = slice(&k_slots, &src_start, &src_stop, &strides, None);
            let v_segment = slice(&v_slots, &src_start, &src_stop, &strides, None);
            let arena = storage.layers.get_mut(&layer_idx).expect("validated layer");
            update_slab_tensor(
                &mut arena.slabs[slab_idx].k,
                &k_segment,
                &dst_start,
                &dst_stop,
            )?;
            update_slab_tensor(
                &mut arena.slabs[slab_idx].v,
                &v_segment,
                &dst_start,
                &dst_stop,
            )?;
            source_token += count;
            logical_token += count;
        }
        Ok(())
    }

    /// Gather block rows in logical order, using one `take(axis=0)` per
    /// maximal same-slab run. The lazy take/reshape/transpose chain can fuse
    /// into MLX SDPA, which is the default mlxcel serving path.
    pub(crate) fn gather_slab_tokens(
        &self,
        layer_idx: usize,
        block_ids: &[PhysicalBlockId],
        token_start: usize,
        token_end: usize,
    ) -> Result<(MlxArray, MlxArray), FaBlockPoolError> {
        let Some(storage) = self.slab_storage.as_ref() else {
            return Err(FaBlockPoolError::InvalidConfig(
                "FA slab storage is not enabled",
            ));
        };
        if token_end <= token_start {
            return Err(FaBlockPoolError::InvalidConfig(
                "FA slab gather requires a non-empty token range",
            ));
        }
        let storage = storage.lock();
        let arena = storage
            .layers
            .get(&layer_idx)
            .ok_or(FaBlockPoolError::InvalidConfig(
                "FA layer slab is not initialized",
            ))?;
        let needed = token_end.div_ceil(arena.block_size);
        let ids = block_ids
            .get(..needed)
            .ok_or(FaBlockPoolError::InvalidConfig(
                "FA slab gather exceeds block table",
            ))?;
        let rows = ids
            .iter()
            .map(|id| {
                arena
                    .block_rows
                    .get(id)
                    .copied()
                    .ok_or(FaBlockPoolError::UnallocatedBlock(*id))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let gather = |select_k: bool| -> Result<MlxArray, FaBlockPoolError> {
            let mut parts = Vec::new();
            let mut i = 0usize;
            while i < rows.len() {
                let slab_idx = rows[i] / FA_POOL_SLAB_BLOCKS;
                let mut local_rows = Vec::new();
                let mut j = i;
                while j < rows.len() && rows[j] / FA_POOL_SLAB_BLOCKS == slab_idx {
                    local_rows.push((rows[j] % FA_POOL_SLAB_BLOCKS) as i32);
                    j += 1;
                }
                let indices = MlxArray::from_raw_data(
                    local_rows.as_ptr().cast(),
                    local_rows.len().saturating_mul(std::mem::size_of::<i32>()),
                    &[local_rows.len() as i32],
                    MlxDtype::Int32,
                );
                let slab = if select_k {
                    arena.slabs[slab_idx].k.as_ref()
                } else {
                    arena.slabs[slab_idx].v.as_ref()
                }
                .ok_or(FaBlockPoolError::InvalidConfig(
                    "FA slab tensor was moved without replacement",
                ))?;
                parts.push(take(slab, &indices, 0, None));
                i = j;
            }
            let gathered = if parts.len() == 1 {
                parts.pop().expect("non-empty rows imply one gather")
            } else {
                let refs = parts.iter().collect::<Vec<_>>();
                concatenate(&refs, 0, None)
            };
            let flat = reshape(
                &gathered,
                &[
                    (rows.len().saturating_mul(arena.block_size)) as i32,
                    arena.n_kv_heads,
                    arena.head_dim,
                ],
                None,
            );
            let window = slice(
                &flat,
                &[token_start as i32, 0, 0],
                &[token_end as i32, arena.n_kv_heads, arena.head_dim],
                &[1, 1, 1],
                None,
            );
            let batched = reshape(
                &window,
                &[
                    1,
                    (token_end - token_start) as i32,
                    arena.n_kv_heads,
                    arena.head_dim,
                ],
                None,
            );
            Ok(transpose(&batched, &[0, 2, 1, 3], None))
        };
        Ok((gather(true)?, gather(false)?))
    }

    pub(crate) fn single_slab_snapshot(
        &self,
        layer_idx: usize,
        block_ids: &[PhysicalBlockId],
    ) -> Option<FaSingleSlabSnapshot> {
        if !self.native_attention || block_ids.is_empty() {
            return None;
        }
        let storage = self.slab_storage.as_ref()?.lock();
        let arena = storage.layers.get(&layer_idx)?;
        let rows = block_ids
            .iter()
            .map(|id| arena.block_rows.get(id).copied())
            .collect::<Option<Vec<_>>>()?;
        let slab_idx = rows[0] / FA_POOL_SLAB_BLOCKS;
        if rows.iter().any(|row| row / FA_POOL_SLAB_BLOCKS != slab_idx) {
            return None;
        }
        let slab = arena.slabs.get(slab_idx)?;
        Some(FaSingleSlabSnapshot {
            k: slab.k.as_ref()?.clone(),
            v: slab.v.as_ref()?.clone(),
            local_rows: rows
                .into_iter()
                .map(|row| (row % FA_POOL_SLAB_BLOCKS) as u32)
                .collect(),
            n_kv_heads: arena.n_kv_heads,
            head_dim: arena.head_dim,
            block_size: arena.block_size,
            dtype: arena.dtype,
        })
    }

    pub fn config(&self) -> FaBlockPoolConfig {
        self.inner.lock().config()
    }

    pub fn allocate(&self, n: u32) -> Result<Vec<PhysicalBlockId>, FaBlockPoolError> {
        self.inner.lock().allocate(n)
    }

    pub fn retain(&self, ids: &[PhysicalBlockId]) -> Result<(), FaBlockPoolError> {
        self.inner.lock().retain(ids)
    }

    pub fn free(&self, ids: &[PhysicalBlockId]) -> Result<(), FaBlockPoolError> {
        let mut pool = self.inner.lock();
        let released = ids
            .iter()
            .filter_map(|id| (pool.ref_count(*id).ok() == Some(1)).then_some(*id))
            .collect::<Vec<_>>();
        pool.free(ids)?;
        if !released.is_empty()
            && let Some(storage) = self.slab_storage.as_ref()
        {
            storage.lock().release_ids(&released);
        }
        Ok(())
    }

    pub fn make_unique(
        &self,
        id: PhysicalBlockId,
    ) -> Result<(PhysicalBlockId, bool), FaBlockPoolError> {
        self.inner.lock().make_unique(id)
    }

    pub fn make_unique_many(
        &self,
        ids: &[PhysicalBlockId],
    ) -> Result<Vec<(PhysicalBlockId, bool)>, FaBlockPoolError> {
        self.inner.lock().make_unique_many(ids)
    }

    pub fn ref_count(&self, id: PhysicalBlockId) -> Result<u32, FaBlockPoolError> {
        self.inner.lock().ref_count(id)
    }

    pub fn snapshot(&self) -> FaBlockPoolSnapshot {
        let pool = self.inner.lock();
        let (slab_count, slab_bytes, slab_grow_events) = self
            .slab_storage
            .as_ref()
            .map(|storage| {
                let storage = storage.lock();
                storage
                    .layers
                    .values()
                    .fold((0u32, 0u64, 0u64), |(count, bytes, grows), arena| {
                        (
                            count.saturating_add(arena.slabs.len() as u32),
                            bytes.saturating_add(arena.slab_bytes()),
                            grows.saturating_add(arena.grow_events),
                        )
                    })
            })
            .unwrap_or_default();
        FaBlockPoolSnapshot {
            config: pool.config(),
            available_blocks: pool.available_blocks(),
            allocated_blocks: pool.allocated_blocks(),
            shared_blocks: pool.shared_blocks(),
            cow_copies: pool.cow_copies(),
            slab_count,
            slab_bytes,
            slab_grow_events,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn array_f32(values: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            values.as_ptr().cast(),
            std::mem::size_of_val(values),
            shape,
            MlxDtype::Float32,
        )
    }

    #[test]
    fn env_flags_accept_only_documented_truthy_values() {
        assert!(!env_flag_enabled(None));
        assert!(!env_flag_enabled(Some("")));
        assert!(!env_flag_enabled(Some("0")));
        assert!(!env_flag_enabled(Some("yes")));
        assert!(env_flag_enabled(Some("1")));
        assert!(env_flag_enabled(Some("true")));
        assert!(env_flag_enabled(Some("TRUE")));
    }

    #[test]
    fn default_fa_block_pool_config_is_always_hard_cap() {
        // Regardless of whether AX_MLX_FA_KV_BLOCK_POOL_MAX_BLOCKS is set,
        // max_blocks here is a real memory budget (this fallback, later
        // corrected to KvManager.total_blocks by
        // MlxRunner::align_fa_block_pool_to_kv) — exhaustion must fail the
        // request rather than silently demote to unbounded contiguous
        // growth.
        assert!(default_fa_block_pool_config().hard_cap);
    }

    #[test]
    fn max_blocks_override_requires_a_positive_integer() {
        assert_eq!(parse_fa_block_pool_max_blocks_override(None), None);
        assert_eq!(parse_fa_block_pool_max_blocks_override(Some("")), None);
        assert_eq!(parse_fa_block_pool_max_blocks_override(Some("0")), None);
        assert_eq!(
            parse_fa_block_pool_max_blocks_override(Some("invalid")),
            None
        );
        assert_eq!(
            parse_fa_block_pool_max_blocks_override(Some("17")),
            Some(17)
        );
    }

    fn pool(max_blocks: u32) -> FaBlockPool {
        FaBlockPool::new(FaBlockPoolConfig {
            block_size_tokens: 16,
            max_blocks,
            hard_cap: false,
        })
        .expect("valid config")
    }

    #[test]
    fn allocate_and_free_roundtrip() {
        let mut p = pool(4);
        let a = p.allocate(2).expect("alloc");
        assert_eq!(a.len(), 2);
        assert_eq!(p.allocated_blocks(), 2);
        assert_eq!(p.available_blocks(), 2);
        p.free(&a).expect("free");
        assert_eq!(p.allocated_blocks(), 0);
        assert_eq!(p.available_blocks(), 4);
    }

    #[test]
    fn allocate_is_fail_closed_on_exhaustion() {
        let mut p = pool(2);
        let first = p.allocate(2).expect("full alloc");
        assert_eq!(first.len(), 2);
        let err = p.allocate(1).expect_err("must exhaust");
        assert!(matches!(
            err,
            FaBlockPoolError::Exhausted {
                requested: 1,
                available: 0
            }
        ));
        // First allocation still held.
        assert_eq!(p.allocated_blocks(), 2);
    }

    #[test]
    fn double_free_is_fail_closed() {
        let mut p = pool(2);
        let ids = p.allocate(1).unwrap();
        p.free(&ids).unwrap();
        let err = p.free(&ids).expect_err("double free");
        assert!(matches!(err, FaBlockPoolError::DoubleFree(_)));
    }

    #[test]
    fn retain_release_returns_block_only_at_zero() {
        let mut p = pool(2);
        let ids = p.allocate(1).expect("allocate");
        p.retain(&ids).expect("retain");
        assert_eq!(p.ref_count(ids[0]), Ok(2));
        assert_eq!(p.shared_blocks(), 1);
        p.free(&ids).expect("release first owner");
        assert_eq!(p.ref_count(ids[0]), Ok(1));
        assert_eq!(p.allocated_blocks(), 1);
        assert_eq!(p.available_blocks(), 1);
        p.free(&ids).expect("release final owner");
        assert_eq!(p.ref_count(ids[0]), Ok(0));
        assert_eq!(p.allocated_blocks(), 0);
        assert_eq!(p.available_blocks(), 2);
        let reused = p.allocate(1).expect("reuse after zero");
        assert_eq!(reused, ids, "an ID is reusable only after final release");
    }

    #[test]
    fn retain_validation_is_atomic() {
        let mut p = pool(3);
        let ids = p.allocate(1).expect("allocate");
        let free_id = PhysicalBlockId((ids[0].0 + 1) % 3);
        let err = p
            .retain(&[ids[0], free_id])
            .expect_err("unallocated ID must reject the whole retain");
        assert!(matches!(err, FaBlockPoolError::UnallocatedBlock(_)));
        assert_eq!(p.ref_count(ids[0]), Ok(1));

        let err = p
            .retain(&[ids[0], ids[0]])
            .expect_err("duplicate ID must reject the whole retain");
        assert!(matches!(err, FaBlockPoolError::DuplicateBlock(_)));
        assert_eq!(p.ref_count(ids[0]), Ok(1));
    }

    #[test]
    fn retain_overflow_is_atomic() {
        let mut p = pool(2);
        let ids = p.allocate(2).expect("allocate");
        p.ref_counts[ids[1].0 as usize] = u32::MAX;
        let err = p
            .retain(&ids)
            .expect_err("overflow must reject every retain");
        assert!(matches!(err, FaBlockPoolError::RefCountOverflow(_)));
        assert_eq!(p.ref_count(ids[0]), Ok(1));
        assert_eq!(p.ref_count(ids[1]), Ok(u32::MAX));
    }

    #[test]
    fn make_unique_cows_only_shared_blocks() {
        let mut p = pool(3);
        let ids = p.allocate(1).expect("allocate");
        assert_eq!(p.make_unique(ids[0]), Ok((ids[0], false)));
        p.retain(&ids).expect("share");
        let (replacement, copied) = p.make_unique(ids[0]).expect("CoW");
        assert!(copied);
        assert_ne!(replacement, ids[0]);
        assert_eq!(p.ref_count(ids[0]), Ok(1));
        assert_eq!(p.ref_count(replacement), Ok(1));
        assert_eq!(p.allocated_blocks(), 2);
        assert_eq!(p.shared_blocks(), 0);
        assert_eq!(p.cow_copies(), 1);
    }

    #[test]
    fn make_unique_exhaustion_preserves_refcounts() {
        let mut p = pool(1);
        let ids = p.allocate(1).expect("allocate");
        p.retain(&ids).expect("share");
        let err = p.make_unique(ids[0]).expect_err("no replacement block");
        assert!(matches!(
            err,
            FaBlockPoolError::Exhausted {
                requested: 1,
                available: 0
            }
        ));
        assert_eq!(p.ref_count(ids[0]), Ok(2));
        assert_eq!(p.allocated_blocks(), 1);
        assert_eq!(p.cow_copies(), 0);
    }

    #[test]
    fn make_unique_many_exhaustion_is_atomic() {
        let mut p = pool(3);
        let ids = p.allocate(2).expect("allocate");
        p.retain(&ids).expect("share both");
        let err = p
            .make_unique_many(&ids)
            .expect_err("one free block cannot replace two shared blocks");
        assert!(matches!(
            err,
            FaBlockPoolError::Exhausted {
                requested: 2,
                available: 1
            }
        ));
        assert_eq!(p.ref_count(ids[0]), Ok(2));
        assert_eq!(p.ref_count(ids[1]), Ok(2));
        assert_eq!(p.allocated_blocks(), 2);
        assert_eq!(p.shared_blocks(), 2);
        assert_eq!(p.cow_copies(), 0);
    }

    #[test]
    fn shared_handle_accounts_concurrent_retain_release() {
        let shared = SharedFaBlockPool::new(FaBlockPoolConfig {
            block_size_tokens: 16,
            max_blocks: 2,
            hard_cap: true,
        })
        .expect("pool");
        let ids = shared.allocate(1).expect("allocate");
        let mut workers = Vec::new();
        for _ in 0..8 {
            let pool = shared.clone();
            let ids = ids.clone();
            workers.push(std::thread::spawn(move || {
                pool.retain(&ids).expect("retain");
                pool.free(&ids).expect("release");
            }));
        }
        for worker in workers {
            worker.join().expect("worker");
        }
        assert_eq!(shared.ref_count(ids[0]), Ok(1));
        shared.free(&ids).expect("final release");
        assert_eq!(shared.snapshot().allocated_blocks, 0);
    }

    #[test]
    fn free_rejects_duplicate_ids_in_same_call() {
        let mut p = pool(2);
        let ids = p.allocate(1).unwrap();
        let err = p
            .free(&[ids[0], ids[0]])
            .expect_err("duplicate free must fail closed");
        assert!(matches!(err, FaBlockPoolError::DoubleFree(_)));
        // Pool unchanged: still one allocated block, free list not corrupted.
        assert_eq!(p.allocated_blocks(), 1);
        assert_eq!(p.available_blocks(), 1);
        p.free(&ids).expect("single free after rejected dup");
        assert_eq!(p.allocated_blocks(), 0);
        assert_eq!(p.available_blocks(), 2);
    }

    #[test]
    fn blocks_for_tokens_ceils() {
        let p = pool(8);
        assert_eq!(p.blocks_for_tokens(0), 0);
        assert_eq!(p.blocks_for_tokens(1), 1);
        assert_eq!(p.blocks_for_tokens(16), 1);
        assert_eq!(p.blocks_for_tokens(17), 2);
        assert_eq!(p.tokens_for_blocks(2), 32);
    }

    #[test]
    fn rejects_invalid_config() {
        assert!(matches!(
            FaBlockPool::new(FaBlockPoolConfig {
                block_size_tokens: 0,
                max_blocks: 1,
                hard_cap: false,
            }),
            Err(FaBlockPoolError::InvalidConfig(_))
        ));
        assert!(matches!(
            FaBlockPool::new(FaBlockPoolConfig {
                block_size_tokens: 16,
                max_blocks: 0,
                hard_cap: false,
            }),
            Err(FaBlockPoolError::InvalidConfig(_))
        ));
    }

    #[test]
    fn fixed_slabs_presize_gather_fragmented_rows_and_reuse_without_monolith() {
        let pool = SharedFaBlockPool::new_with_slab_storage(FaBlockPoolConfig {
            block_size_tokens: 2,
            // The old design allocated this entire ceiling on first write.
            max_blocks: 4096,
            hard_cap: true,
        })
        .expect("pool");
        let ids = pool.allocate(34).expect("allocate first generation");
        let k_values = (0..68).map(|value| value as f32).collect::<Vec<_>>();
        let v_values = k_values
            .iter()
            .map(|value| value + 1000.0)
            .collect::<Vec<_>>();
        pool.write_slab_tokens(
            0,
            &ids,
            0,
            &array_f32(&k_values, &[1, 1, 68, 1]),
            &array_f32(&v_values, &[1, 1, 68, 1]),
        )
        .expect("write across two slabs");
        let (k, v) = pool
            .gather_slab_tokens(0, &ids, 0, 68)
            .expect("gather across two slabs");
        mlx_sys::eval(&[&k, &v]);
        assert_eq!(k.data_f32(), k_values);
        assert_eq!(v.data_f32(), v_values);

        let snapshot = pool.snapshot();
        assert_eq!(snapshot.slab_count, 2);
        assert_eq!(snapshot.slab_bytes, 1024);
        assert_eq!(snapshot.slab_grow_events, 0, "whole prefill was presized");

        // Releasing in logical order and allocating again makes row order run
        // backwards across the slab boundary. Gather must still follow the
        // block table, and the existing two slabs must be reused unchanged.
        pool.free(&ids).expect("release first generation");
        let reused = pool.allocate(34).expect("allocate reused rows");
        let k_reused = (0..68)
            .map(|value| 10_000.0 + value as f32)
            .collect::<Vec<_>>();
        let v_reused = k_reused
            .iter()
            .map(|value| value + 1000.0)
            .collect::<Vec<_>>();
        pool.write_slab_tokens(
            0,
            &reused,
            0,
            &array_f32(&k_reused, &[1, 1, 68, 1]),
            &array_f32(&v_reused, &[1, 1, 68, 1]),
        )
        .expect("write reused fragmented rows");
        let (k, v) = pool
            .gather_slab_tokens(0, &reused, 0, 68)
            .expect("gather reused fragmented rows");
        mlx_sys::eval(&[&k, &v]);
        assert_eq!(k.data_f32(), k_reused);
        assert_eq!(v.data_f32(), v_reused);
        assert_eq!(pool.snapshot().slab_count, 2);
        pool.free(&reused).expect("release reused generation");
    }

    #[test]
    fn native_snapshot_is_offered_only_for_one_fixed_slab() {
        let pool = SharedFaBlockPool::new_with_native_slab_storage(FaBlockPoolConfig {
            block_size_tokens: 2,
            max_blocks: 64,
            hard_cap: true,
        })
        .expect("pool");
        let ids = pool.allocate(33).expect("ids");
        let values = vec![1.0f32; 66];
        let array = array_f32(&values, &[1, 1, 66, 1]);
        pool.write_slab_tokens(0, &ids, 0, &array, &array)
            .expect("write");
        assert!(pool.single_slab_snapshot(0, &ids[..32]).is_some());
        assert!(pool.single_slab_snapshot(0, &ids).is_none());
        pool.free(&ids).expect("free");
    }
}
