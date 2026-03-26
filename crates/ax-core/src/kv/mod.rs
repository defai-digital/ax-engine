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
pub mod qwen35_kv;

pub use cpu_kv::{CpuKv, CpuKvSnapshot};
pub use gpu_kv::{GpuKv, GpuKvDtype};
pub use qwen35_kv::{Qwen35Kv, Qwen35KvSnapshot};

#[derive(Debug, Clone)]
pub enum ModelKvSnapshot {
    Cpu(CpuKvSnapshot),
    Qwen35(Qwen35KvSnapshot),
}

/// Single-owner KV cache. Eliminates the v1 CPU+GPU split-brain.
pub enum ModelKv {
    /// CPU RAM storage (f32). Used with CpuBackend or HybridCpuDecode.
    Cpu(CpuKv),
    /// GPU Metal buffer storage (f32 or f16). Used with MetalBackend or HybridBackend.
    Gpu(GpuKv),
    /// Hybrid recurrent state + attention KV for Qwen3.5.
    Qwen35(Qwen35Kv),
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

    /// Get a reference to the Qwen3.5 hybrid cache/state.
    pub fn as_qwen35_mut(&mut self) -> Option<&mut Qwen35Kv> {
        match self {
            Self::Qwen35(q) => Some(q),
            Self::Cpu(_) | Self::Gpu(_) => None,
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
    pub fn snapshot(&self) -> Option<ModelKvSnapshot> {
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
mod tests {
    use super::*;

    #[test]
    fn test_model_kv_cpu_snapshot_round_trip() {
        let mut kv = ModelKv::Cpu(CpuKv::new(1, 1, 2, 16));
        if let ModelKv::Cpu(cpu_kv) = &mut kv {
            cpu_kv.append_and_advance(0, &[1.0, 2.0], &[3.0, 4.0]);
            cpu_kv.finalize_token();
        }

        let snapshot = kv.snapshot().expect("cpu kv should support snapshots");
        kv.clear();
        kv.restore_snapshot(&snapshot)
            .expect("cpu restore should succeed");

        let restored = kv.as_cpu_mut().expect("expected cpu kv");
        assert_eq!(restored.seq_len(), 1);
        assert_eq!(restored.k_slice(0, 1), &[1.0, 2.0]);
        assert_eq!(restored.v_slice(0, 1), &[3.0, 4.0]);
    }

    #[test]
    fn test_model_kv_qwen35_snapshot_round_trip() {
        let mut kv = ModelKv::Qwen35(Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2));
        if let ModelKv::Qwen35(qwen_kv) = &mut kv {
            let slot1 = qwen_kv.allocate_recurrent_slot();
            qwen_kv.set_active_slot(slot1);
            qwen_kv.conv_state_for_slot_mut(slot1, 0).fill(1.5);
            qwen_kv.recurrent_state_for_slot_mut(slot1, 0).fill(2.5);
            qwen_kv.attention_append(3, &[10.0, 11.0], &[12.0, 13.0]);
            qwen_kv.finalize_token();
        }

        let snapshot = kv.snapshot().expect("qwen35 kv should support snapshots");
        kv.clear();
        kv.restore_snapshot(&snapshot)
            .expect("qwen35 restore should succeed");

        let restored = kv.as_qwen35_mut().expect("expected qwen35 kv");
        assert_eq!(restored.active_slot(), 1);
        assert_eq!(restored.seq_len(), 1);
        assert_eq!(
            restored.attention_k_slice_including_current(3, 1),
            &[10.0, 11.0]
        );
        assert_eq!(
            restored.attention_v_slice_including_current(3, 1),
            &[12.0, 13.0]
        );
        assert!(restored.conv_state_for_slot(1, 0).iter().all(|&v| v == 1.5));
        assert!(
            restored
                .recurrent_state_for_slot(1, 0)
                .iter()
                .all(|&v| v == 2.5)
        );
    }

    #[test]
    fn test_model_kv_qwen35_snapshot_restores_into_fresh_kv() {
        let mut source = ModelKv::Qwen35(Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2));
        if let ModelKv::Qwen35(qwen_kv) = &mut source {
            let slot1 = qwen_kv.allocate_recurrent_slot();
            qwen_kv.set_active_slot(slot1);
            qwen_kv.conv_state_for_slot_mut(slot1, 0).fill(1.5);
            qwen_kv.recurrent_state_for_slot_mut(slot1, 0).fill(2.5);
            qwen_kv.attention_append(3, &[10.0, 11.0], &[12.0, 13.0]);
            qwen_kv.finalize_token();
        }

        let snapshot = source
            .snapshot()
            .expect("qwen35 kv should support snapshots");

        let mut restored = ModelKv::Qwen35(Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2));
        restored
            .restore_snapshot(&snapshot)
            .expect("qwen35 restore into fresh kv should succeed");

        let restored_qwen = restored.as_qwen35_mut().expect("expected qwen35 kv");
        assert_eq!(restored_qwen.active_slot(), 1);
        assert_eq!(restored_qwen.seq_len(), 1);
        assert_eq!(
            restored_qwen.attention_k_slice_including_current(3, 1),
            &[10.0, 11.0]
        );
        assert_eq!(
            restored_qwen.attention_v_slice_including_current(3, 1),
            &[12.0, 13.0]
        );
        assert!(
            restored_qwen
                .conv_state_for_slot(1, 0)
                .iter()
                .all(|&v| v == 1.5)
        );
        assert!(
            restored_qwen
                .recurrent_state_for_slot(1, 0)
                .iter()
                .all(|&v| v == 2.5)
        );
    }

    #[test]
    fn test_model_kv_gpu_snapshot_is_unsupported() {
        assert!(matches!(
            ModelKv::Cpu(CpuKv::new(1, 1, 2, 16)).snapshot(),
            Some(ModelKvSnapshot::Cpu(_))
        ));
    }

    #[test]
    fn test_model_kv_restore_rejects_mismatched_snapshot_variant() {
        let mut kv = ModelKv::Cpu(CpuKv::new(1, 1, 2, 16));
        let qwen_snapshot = ModelKv::Qwen35(Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2))
            .snapshot()
            .expect("qwen35 snapshot should exist");
        let err = kv
            .restore_snapshot(&qwen_snapshot)
            .expect_err("mismatched restore should fail");
        assert!(
            err.to_string()
                .contains("cannot restore qwen35 snapshot into cpu kv")
        );
    }
}
