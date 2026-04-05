//! KV cache — single-owner design.
//!
//! # v2 Architecture Decision
//!
//! v1 had a split-brain design: `KvCache` (CPU) + `GpuKvCache` (GPU) with an
//! `advance_by()` call to keep seq_len in sync when the GPU path ran prefill
//! but the CPU path ran decode. This caused a P1 panic:
//!
//!   GPU batch prefill → `kv_cache.advance_by(209)` increments seq_len to 209
//!   CPU decode → `kv_cache.append()` tries to write at offset=209*stride
//!   KvCache buffer capacity was still 128 → out-of-range slice panic
//!
//! v2 solution: `ModelKv` is a single enum. The forward pass holds one `ModelKv`.
//! The backend creates the right variant at model initialization.
//!
//! - `MetalBackend` / `HybridBackend` → `ModelKv::Gpu`
//! - `CpuBackend` / `HybridCpuDecode` → `ModelKv::Cpu`
//!
//! `forward_batch_gpu_unified` takes `&mut GpuKv` directly — it is
//! architecturally impossible to have GPU batch prefill + CPU decode in the
//! same pass. If the caller holds `ModelKv::Cpu`, `forward_batch` falls
//! through to serial prefill automatically.
//!
//! There is no `advance_by()`.

use anyhow::bail;

pub mod cpu_kv;
pub mod gpu_kv;
pub mod page;
pub mod qwen3_5_kv;
pub mod qwen35_kv {
    pub use super::qwen3_5_kv::*;
}
mod qwen3_5;

pub use cpu_kv::{CpuKv, CpuKvSnapshot};
pub use gpu_kv::{GpuKv, GpuKvDtype};
pub(crate) use qwen3_5_kv::Qwen3_5LayerStateOwner;
pub use qwen3_5_kv::{
    Qwen3_5AttentionSnapshot, Qwen3_5Kv, Qwen3_5KvSnapshot, Qwen3_5PreparedRecurrentStateBatchKind,
    Qwen3_5RecurrentSlotSnapshot, Qwen35AttentionSnapshot, Qwen35Kv, Qwen35KvSnapshot,
    Qwen35PreparedRecurrentStateBatchKind, Qwen35RecurrentSlotSnapshot,
};

#[derive(Debug, Clone)]
pub enum ModelKvSnapshot {
    Cpu(CpuKvSnapshot),
    Qwen35(Qwen3_5KvSnapshot),
}

/// Single-owner KV cache. Eliminates the v1 CPU+GPU split-brain.
pub enum ModelKv {
    /// CPU RAM storage (f32). Used with CpuBackend or HybridCpuDecode.
    Cpu(CpuKv),
    /// GPU Metal buffer storage (f32 or f16). Used with MetalBackend or HybridBackend.
    Gpu(GpuKv),
    /// Hybrid recurrent state + attention KV for Qwen3.5.
    Qwen35(Box<Qwen3_5Kv>),
}

impl ModelKv {
    /// Current sequence length (number of tokens with KV stored).
    pub fn seq_len(&self) -> usize {
        match self {
            Self::Cpu(c) => c.seq_len(),
            Self::Gpu(g) => g.seq_len(),
            Self::Qwen35(q) => q.seq_len(),
        }
    }

    /// Attempt to get a mutable reference to the GPU KV cache.
    ///
    /// Returns `None` if this is a CPU KV cache. Forward passes use this to
    /// decide whether to take the GPU batch path — if None, they fall through
    /// to serial CPU prefill.
    pub fn as_gpu_mut(&mut self) -> Option<&mut GpuKv> {
        match self {
            Self::Gpu(g) => Some(g),
            Self::Cpu(_) => None,
            Self::Qwen35(_) => None,
        }
    }

    /// Attempt to get an immutable reference to the GPU KV cache.
    pub fn as_gpu(&self) -> Option<&GpuKv> {
        match self {
            Self::Gpu(g) => Some(g),
            Self::Cpu(_) => None,
            Self::Qwen35(_) => None,
        }
    }

    /// Get a reference to the CPU KV cache.
    ///
    /// Returns `None` if this is a GPU KV cache.
    pub fn as_cpu_mut(&mut self) -> Option<&mut CpuKv> {
        match self {
            Self::Cpu(c) => Some(c),
            Self::Gpu(_) => None,
            Self::Qwen35(_) => None,
        }
    }

    /// True if KV data is GPU-resident.
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Gpu(_))
    }

    /// Snapshot the current KV state when the backing implementation supports it.
    ///
    /// GPU KV snapshotting is not supported yet because AX does not maintain a
    /// CPU mirror of Metal buffers in v2.
    /// Check whether this KV cache supports snapshotting (without performing one).
    pub fn supports_snapshot(&self) -> bool {
        matches!(self, Self::Cpu(_) | Self::Qwen35(_))
    }

    pub fn snapshot(&mut self) -> Option<ModelKvSnapshot> {
        match self {
            Self::Cpu(c) => Some(ModelKvSnapshot::Cpu(c.snapshot())),
            Self::Gpu(_) => None,
            Self::Qwen35(q) => Some(ModelKvSnapshot::Qwen35(q.snapshot_active_slot())),
        }
    }

    /// Restore a previously captured KV snapshot.
    pub fn restore_snapshot(&mut self, snapshot: &ModelKvSnapshot) -> anyhow::Result<()> {
        match (self, snapshot) {
            (Self::Cpu(kv), ModelKvSnapshot::Cpu(snapshot)) => {
                kv.restore(snapshot);
                Ok(())
            }
            (Self::Qwen35(kv), ModelKvSnapshot::Qwen35(snapshot)) => {
                kv.restore_snapshot(snapshot);
                Ok(())
            }
            (Self::Gpu(_), _) => bail!("GPU KV snapshot restore is not supported"),
            (Self::Cpu(_), ModelKvSnapshot::Qwen35(_)) => {
                bail!("cannot restore qwen35 snapshot into cpu kv")
            }
            (Self::Qwen35(_), ModelKvSnapshot::Cpu(_)) => {
                bail!("cannot restore cpu snapshot into qwen35 kv")
            }
        }
    }

    /// Reset to zero tokens (keeps allocated memory/buffers).
    pub fn clear(&mut self) {
        match self {
            Self::Cpu(c) => c.clear(),
            Self::Gpu(g) => g.clear(),
            Self::Qwen35(q) => q.clear(),
        }
    }

    /// Rewind seq_len to `pos` without freeing memory or Metal buffers.
    ///
    /// Used by speculative decoding to roll back the KV state after a rejected
    /// draft token. Data beyond `pos` is overwritten on the next append.
    pub fn truncate_to(&mut self, pos: usize) {
        match self {
            Self::Cpu(c) => c.truncate_to(pos),
            Self::Gpu(g) => g.truncate_to(pos),
            Self::Qwen35(q) => q.truncate_to(pos),
        }
    }
}

#[cfg(test)]
mod tests;
