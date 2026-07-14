//! FA-only private physical KV block pool (scaffold).
//!
//! Design: `docs/designs/kv-weak-surfaces-2026-07-14.md` Track C / PR4.
//!
//! This module is a **pure logical allocator** for fixed-size token blocks.
//! It does not yet own Metal buffers or integrate into [`crate::kv_cache::MlxKVCache`].
//! When `AX_MLX_FA_KV_BLOCK_POOL=1`, callers may opt into the pool API; the default
//! production path remains contiguous per-request `MlxKVCache` buffers.
//!
//! Scope (PR4):
//! - private (non-shared) full-attention blocks only
//! - fail-closed exhaustion
//! - `max_blocks` is expected to match `KvManagerConfig.total_blocks`
//!
//! Non-goals (PR5+):
//! - cross-request block sharing / CoW
//! - MLA latent/k_pe paired blocks
//! - SDPA-facing materialize into live decode (measured later)

use std::collections::VecDeque;
use std::sync::OnceLock;

/// Opt-in flag for FA private block-pool scaffolding. Default: OFF.
pub fn fa_kv_block_pool_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_FA_KV_BLOCK_POOL")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Logical physical-block identifier (pool-local, not a GPU pointer).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PhysicalBlockId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FaBlockPoolConfig {
    pub block_size_tokens: u32,
    pub max_blocks: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FaBlockPoolError {
    InvalidConfig(&'static str),
    Exhausted { requested: u32, available: u32 },
    UnknownBlock(PhysicalBlockId),
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
            Self::DoubleFree(id) => write!(f, "double free of physical block {}", id.0),
        }
    }
}

impl std::error::Error for FaBlockPoolError {}

/// Private FA block allocator. Refcount is always 1 while allocated (no sharing).
#[derive(Debug)]
pub struct FaBlockPool {
    config: FaBlockPoolConfig,
    free: VecDeque<PhysicalBlockId>,
    /// `true` while the block is allocated to exactly one owner.
    allocated: Vec<bool>,
    allocated_count: u32,
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
            allocated: vec![false; config.max_blocks as usize],
            allocated_count: 0,
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

    /// Allocate `n` private blocks. Fail-closed: no partial allocation.
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
            debug_assert!(!self.allocated[idx]);
            self.allocated[idx] = true;
            self.allocated_count = self.allocated_count.saturating_add(1);
            out.push(id);
        }
        Ok(out)
    }

    /// Free private blocks. Unknown or double-free is fail-closed.
    ///
    /// Duplicate IDs in a single call are treated as double-free so the free
    /// list cannot gain the same block twice (and `allocated_count` cannot
    /// under-count).
    pub fn free(&mut self, ids: &[PhysicalBlockId]) -> Result<(), FaBlockPoolError> {
        // Validate all first so free is atomic.
        let mut seen = vec![false; self.allocated.len()];
        for id in ids {
            let idx = id.0 as usize;
            if idx >= self.allocated.len() {
                return Err(FaBlockPoolError::UnknownBlock(*id));
            }
            if !self.allocated[idx] || seen[idx] {
                return Err(FaBlockPoolError::DoubleFree(*id));
            }
            seen[idx] = true;
        }
        for id in ids {
            let idx = id.0 as usize;
            self.allocated[idx] = false;
            self.allocated_count = self.allocated_count.saturating_sub(1);
            self.free.push_back(*id);
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(max_blocks: u32) -> FaBlockPool {
        FaBlockPool::new(FaBlockPoolConfig {
            block_size_tokens: 16,
            max_blocks,
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
            }),
            Err(FaBlockPoolError::InvalidConfig(_))
        ));
        assert!(matches!(
            FaBlockPool::new(FaBlockPoolConfig {
                block_size_tokens: 16,
                max_blocks: 0,
            }),
            Err(FaBlockPoolError::InvalidConfig(_))
        ));
    }
}
