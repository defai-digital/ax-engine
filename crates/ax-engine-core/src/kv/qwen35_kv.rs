//! Hybrid recurrent cache/state for Qwen3.5.
//!
//! Qwen3.5 mixes full-attention layers and recurrent GDN layers. Attention
//! layers still need a normal KV cache, while recurrent layers need:
//! - a short convolution history of length `conv_kernel - 1`
//! - the persistent delta-net state tensor

use std::borrow::Cow;

use super::page::{KvCacheConfig, KvDtype, recommended_page_size};
use super::{CpuKv, CpuKvSnapshot, GpuKv, GpuKvDtype};

#[derive(Debug, Clone)]
pub struct Qwen35RecurrentSlotSnapshot {
    conv_states: Vec<Vec<f32>>,
    recurrent_states: Vec<Vec<f32>>,
    seqlen_offset: usize,
}

#[derive(Debug, Clone)]
pub struct Qwen35KvSnapshot {
    attention: CpuKvSnapshot,
    recurrent_slot: Qwen35RecurrentSlotSnapshot,
    seq_len: usize,
    active_slot: usize,
}

#[derive(Debug, Clone)]
pub struct Qwen35AttentionSnapshot {
    attention: CpuKvSnapshot,
    seq_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qwen35StateOwner {
    CpuMaterialized,
    BackendOwned,
}

impl Qwen35StateOwner {
    fn from_cpu_stale(cpu_stale: bool) -> Self {
        if cpu_stale {
            Self::BackendOwned
        } else {
            Self::CpuMaterialized
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qwen35LayerStateOwner {
    CpuMaterialized,
    BackendOwned,
    Split,
}

impl Qwen35LayerStateOwner {}

#[derive(Debug, Clone)]
struct Qwen35RecurrentSlot {
    conv_states: Vec<Vec<f32>>,
    recurrent_states: Vec<Vec<f32>>,
    conv_state_generations: Vec<u64>,
    recurrent_state_generations: Vec<u64>,
    cpu_materialized_conv_generations: Vec<u64>,
    cpu_materialized_recurrent_generations: Vec<u64>,
    conv_state_pristine_zero: Vec<bool>,
    recurrent_state_pristine_zero: Vec<bool>,
    seqlen_offset: usize,
}

impl Qwen35RecurrentSlot {
    fn new(
        recurrent_layers: &[bool],
        conv_cache_len: usize,
        conv_dim: usize,
        recurrent_state_len: usize,
    ) -> Self {
        Self {
            conv_states: recurrent_layers
                .iter()
                .map(|&is_recurrent| {
                    if is_recurrent {
                        vec![0.0f32; conv_cache_len * conv_dim]
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            recurrent_states: recurrent_layers
                .iter()
                .map(|&is_recurrent| {
                    if is_recurrent {
                        vec![0.0f32; recurrent_state_len]
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            conv_state_generations: vec![1; recurrent_layers.len()],
            recurrent_state_generations: vec![1; recurrent_layers.len()],
            cpu_materialized_conv_generations: vec![1; recurrent_layers.len()],
            cpu_materialized_recurrent_generations: vec![1; recurrent_layers.len()],
            conv_state_pristine_zero: recurrent_layers.iter().map(|_| true).collect(),
            recurrent_state_pristine_zero: recurrent_layers.iter().map(|_| true).collect(),
            seqlen_offset: 0,
        }
    }

    fn clear(&mut self) {
        self.seqlen_offset = 0;
        for state in &mut self.conv_states {
            state.fill(0.0);
        }
        for state in &mut self.recurrent_states {
            state.fill(0.0);
        }
        self.touch_all_state();
        self.conv_state_pristine_zero.fill(true);
        self.recurrent_state_pristine_zero.fill(true);
    }

    fn snapshot(&self) -> Qwen35RecurrentSlotSnapshot {
        Qwen35RecurrentSlotSnapshot {
            conv_states: self.conv_states.clone(),
            recurrent_states: self.recurrent_states.clone(),
            seqlen_offset: self.seqlen_offset,
        }
    }

    fn restore(&mut self, snapshot: &Qwen35RecurrentSlotSnapshot) {
        assert_eq!(
            self.conv_states.len(),
            snapshot.conv_states.len(),
            "qwen35 recurrent snapshot layer count mismatch"
        );
        assert_eq!(
            self.recurrent_states.len(),
            snapshot.recurrent_states.len(),
            "qwen35 recurrent snapshot layer count mismatch"
        );
        for (dst, src) in self.conv_states.iter_mut().zip(snapshot.conv_states.iter()) {
            assert_eq!(
                dst.len(),
                src.len(),
                "qwen35 recurrent snapshot conv state size mismatch"
            );
            dst.copy_from_slice(src);
        }
        for (dst, src) in self
            .recurrent_states
            .iter_mut()
            .zip(snapshot.recurrent_states.iter())
        {
            assert_eq!(
                dst.len(),
                src.len(),
                "qwen35 recurrent snapshot recurrent state size mismatch"
            );
            dst.copy_from_slice(src);
        }
        self.seqlen_offset = snapshot.seqlen_offset;
        self.touch_all_state();
    }

    fn clone_preserving_ownership_from(&mut self, src_slot: &Qwen35RecurrentSlot) {
        assert_eq!(
            self.conv_states.len(),
            src_slot.conv_states.len(),
            "qwen35 recurrent slot layer count mismatch"
        );
        assert_eq!(
            self.recurrent_states.len(),
            src_slot.recurrent_states.len(),
            "qwen35 recurrent slot layer count mismatch"
        );
        for (layer_idx, (dst, src)) in self
            .conv_states
            .iter_mut()
            .zip(src_slot.conv_states.iter())
            .enumerate()
        {
            assert_eq!(
                dst.len(),
                src.len(),
                "qwen35 recurrent slot conv state size mismatch"
            );
            if src.is_empty()
                || src_slot.cpu_materialized_conv_generations[layer_idx]
                    == src_slot.conv_state_generations[layer_idx]
            {
                dst.copy_from_slice(src);
            }
        }
        for (layer_idx, (dst, src)) in self
            .recurrent_states
            .iter_mut()
            .zip(src_slot.recurrent_states.iter())
            .enumerate()
        {
            assert_eq!(
                dst.len(),
                src.len(),
                "qwen35 recurrent slot recurrent state size mismatch"
            );
            if src.is_empty()
                || src_slot.cpu_materialized_recurrent_generations[layer_idx]
                    == src_slot.recurrent_state_generations[layer_idx]
            {
                dst.copy_from_slice(src);
            }
        }
        self.conv_state_generations
            .copy_from_slice(&src_slot.conv_state_generations);
        self.recurrent_state_generations
            .copy_from_slice(&src_slot.recurrent_state_generations);
        self.cpu_materialized_conv_generations
            .copy_from_slice(&src_slot.cpu_materialized_conv_generations);
        self.cpu_materialized_recurrent_generations
            .copy_from_slice(&src_slot.cpu_materialized_recurrent_generations);
        self.conv_state_pristine_zero
            .copy_from_slice(&src_slot.conv_state_pristine_zero);
        self.recurrent_state_pristine_zero
            .copy_from_slice(&src_slot.recurrent_state_pristine_zero);
        self.seqlen_offset = src_slot.seqlen_offset;
    }

    fn bump_conv_generation(&mut self, layer: usize) -> u64 {
        let generation = &mut self.conv_state_generations[layer];
        *generation = generation.wrapping_add(1);
        if *generation == 0 {
            *generation = 1;
        }
        *generation
    }

    fn bump_recurrent_generation(&mut self, layer: usize) -> u64 {
        let generation = &mut self.recurrent_state_generations[layer];
        *generation = generation.wrapping_add(1);
        if *generation == 0 {
            *generation = 1;
        }
        *generation
    }

    fn touch_conv_state(&mut self, layer: usize) {
        let generation = self.bump_conv_generation(layer);
        self.cpu_materialized_conv_generations[layer] = generation;
        self.conv_state_pristine_zero[layer] = false;
    }

    fn touch_recurrent_state(&mut self, layer: usize) {
        let generation = self.bump_recurrent_generation(layer);
        self.cpu_materialized_recurrent_generations[layer] = generation;
        self.recurrent_state_pristine_zero[layer] = false;
    }

    fn touch_layer_state(&mut self, layer: usize) {
        let conv_generation = self.bump_conv_generation(layer);
        self.cpu_materialized_conv_generations[layer] = conv_generation;
        let recurrent_generation = self.bump_recurrent_generation(layer);
        self.cpu_materialized_recurrent_generations[layer] = recurrent_generation;
        self.conv_state_pristine_zero[layer] = false;
        self.recurrent_state_pristine_zero[layer] = false;
    }

    fn touch_all_state(&mut self) {
        for i in 0..self.conv_state_generations.len() {
            self.conv_state_generations[i] = self.conv_state_generations[i].wrapping_add(1);
            if self.conv_state_generations[i] == 0 {
                self.conv_state_generations[i] = 1;
            }
            self.cpu_materialized_conv_generations[i] = self.conv_state_generations[i];
            self.recurrent_state_generations[i] =
                self.recurrent_state_generations[i].wrapping_add(1);
            if self.recurrent_state_generations[i] == 0 {
                self.recurrent_state_generations[i] = 1;
            }
            self.cpu_materialized_recurrent_generations[i] = self.recurrent_state_generations[i];
        }
    }

    fn note_backend_conv_state(&mut self, layer: usize) -> u64 {
        self.conv_state_pristine_zero[layer] = false;
        self.bump_conv_generation(layer)
        // NOTE: do NOT update cpu_materialized_conv_generations — CPU is now stale
    }

    fn cpu_conv_state_stale(&self, layer: usize) -> bool {
        self.cpu_materialized_conv_generations[layer] != self.conv_state_generations[layer]
    }

    fn conv_state_owner(&self, layer: usize) -> Qwen35StateOwner {
        Qwen35StateOwner::from_cpu_stale(self.cpu_conv_state_stale(layer))
    }

    fn conv_state_pristine_zero(&self, layer: usize) -> bool {
        self.conv_state_pristine_zero[layer]
    }

    fn overwrite_conv_state_from_backend(
        &mut self,
        layer: usize,
        conv_state: &[f32],
        generation: u64,
    ) {
        assert_eq!(
            self.conv_states[layer].len(),
            conv_state.len(),
            "qwen35 conv backend sync size mismatch"
        );
        assert_eq!(
            self.conv_state_generations[layer], generation,
            "qwen35 conv backend sync generation mismatch"
        );
        self.conv_states[layer].copy_from_slice(conv_state);
        self.cpu_materialized_conv_generations[layer] = generation;
        self.conv_state_pristine_zero[layer] = false;
    }

    fn conv_state_generation(&self, layer: usize) -> u64 {
        self.conv_state_generations[layer]
    }

    fn mark_conv_state_backend_owned(&mut self, layer: usize) {
        self.cpu_materialized_conv_generations[layer] = 0;
        self.conv_state_pristine_zero[layer] = false;
    }

    fn recurrent_state_generation(&self, layer: usize) -> u64 {
        self.recurrent_state_generations[layer]
    }

    fn mark_recurrent_state_backend_owned(&mut self, layer: usize) {
        self.cpu_materialized_recurrent_generations[layer] = 0;
        self.recurrent_state_pristine_zero[layer] = false;
    }

    fn note_backend_recurrent_state(&mut self, layer: usize) -> u64 {
        self.recurrent_state_pristine_zero[layer] = false;
        self.bump_recurrent_generation(layer)
    }

    fn cpu_recurrent_state_stale(&self, layer: usize) -> bool {
        self.cpu_materialized_recurrent_generations[layer]
            != self.recurrent_state_generations[layer]
    }

    fn recurrent_state_owner(&self, layer: usize) -> Qwen35StateOwner {
        Qwen35StateOwner::from_cpu_stale(self.cpu_recurrent_state_stale(layer))
    }

    fn recurrent_state_pristine_zero(&self, layer: usize) -> bool {
        self.recurrent_state_pristine_zero[layer]
    }

    fn layer_state_owner(&self, layer: usize) -> Qwen35LayerStateOwner {
        match (
            self.conv_state_owner(layer),
            self.recurrent_state_owner(layer),
        ) {
            (Qwen35StateOwner::CpuMaterialized, Qwen35StateOwner::CpuMaterialized) => {
                Qwen35LayerStateOwner::CpuMaterialized
            }
            (Qwen35StateOwner::BackendOwned, Qwen35StateOwner::BackendOwned) => {
                Qwen35LayerStateOwner::BackendOwned
            }
            _ => Qwen35LayerStateOwner::Split,
        }
    }

    fn mark_layer_state_backend_owned(&mut self, layer: usize) {
        self.mark_conv_state_backend_owned(layer);
        self.mark_recurrent_state_backend_owned(layer);
    }

    fn overwrite_recurrent_state_from_backend(
        &mut self,
        layer: usize,
        recurrent_state: &[f32],
        generation: u64,
    ) {
        assert_eq!(
            self.recurrent_states[layer].len(),
            recurrent_state.len(),
            "qwen35 recurrent backend sync size mismatch"
        );
        assert_eq!(
            self.recurrent_state_generations[layer], generation,
            "qwen35 recurrent backend sync generation mismatch"
        );
        self.recurrent_states[layer].copy_from_slice(recurrent_state);
        self.cpu_materialized_recurrent_generations[layer] = generation;
        self.recurrent_state_pristine_zero[layer] = false;
    }
}

#[derive(Debug)]
struct Qwen35RecurrentPool {
    slots: Vec<Qwen35RecurrentSlot>,
    free_slots: Vec<usize>,
}

impl Qwen35RecurrentPool {
    fn new(
        recurrent_layers: &[bool],
        conv_cache_len: usize,
        conv_dim: usize,
        recurrent_state_len: usize,
    ) -> Self {
        Self {
            slots: vec![Qwen35RecurrentSlot::new(
                recurrent_layers,
                conv_cache_len,
                conv_dim,
                recurrent_state_len,
            )],
            free_slots: Vec::new(),
        }
    }

    fn is_allocated_slot(&self, slot_idx: usize) -> bool {
        slot_idx < self.slots.len() && (slot_idx == 0 || !self.free_slots.contains(&slot_idx))
    }

    fn allocate(
        &mut self,
        recurrent_layers: &[bool],
        conv_cache_len: usize,
        conv_dim: usize,
        recurrent_state_len: usize,
    ) -> usize {
        if let Some(slot_idx) = self.free_slots.pop() {
            self.slots[slot_idx].clear();
            return slot_idx;
        }
        let slot_idx = self.slots.len();
        self.slots.push(Qwen35RecurrentSlot::new(
            recurrent_layers,
            conv_cache_len,
            conv_dim,
            recurrent_state_len,
        ));
        slot_idx
    }

    fn ensure_allocated_slot(
        &mut self,
        slot_idx: usize,
        recurrent_layers: &[bool],
        conv_cache_len: usize,
        conv_dim: usize,
        recurrent_state_len: usize,
    ) {
        if slot_idx < self.slots.len() {
            self.free_slots.retain(|&free_slot| free_slot != slot_idx);
            return;
        }

        let original_len = self.slots.len();
        for _ in original_len..=slot_idx {
            self.slots.push(Qwen35RecurrentSlot::new(
                recurrent_layers,
                conv_cache_len,
                conv_dim,
                recurrent_state_len,
            ));
        }
        self.free_slots.extend((original_len..slot_idx).rev());
    }

    fn free(&mut self, slot_idx: usize) {
        assert!(
            slot_idx < self.slots.len(),
            "qwen35 recurrent slot out of range"
        );
        self.slots[slot_idx].clear();
        if slot_idx != 0 && !self.free_slots.contains(&slot_idx) {
            self.free_slots.push(slot_idx);
        }
    }

    fn clear_all(&mut self) {
        for slot in &mut self.slots {
            slot.clear();
        }
        self.free_slots.clear();
        self.free_slots.extend((1..self.slots.len()).rev());
    }

    fn slot(&self, slot_idx: usize) -> &Qwen35RecurrentSlot {
        assert!(
            self.is_allocated_slot(slot_idx),
            "qwen35 recurrent slot is not allocated"
        );
        &self.slots[slot_idx]
    }

    fn slot_mut(&mut self, slot_idx: usize) -> &mut Qwen35RecurrentSlot {
        assert!(
            self.is_allocated_slot(slot_idx),
            "qwen35 recurrent slot is not allocated"
        );
        &mut self.slots[slot_idx]
    }

    fn conv_state_mut(&mut self, slot_idx: usize, layer: usize) -> &mut [f32] {
        let slot = self.slot_mut(slot_idx);
        slot.touch_conv_state(layer);
        slot.conv_states[layer].as_mut_slice()
    }

    fn conv_state(&self, slot_idx: usize, layer: usize) -> &[f32] {
        self.slot(slot_idx).conv_states[layer].as_slice()
    }

    fn recurrent_state_mut(&mut self, slot_idx: usize, layer: usize) -> &mut [f32] {
        let slot = self.slot_mut(slot_idx);
        slot.touch_recurrent_state(layer);
        slot.recurrent_states[layer].as_mut_slice()
    }

    fn recurrent_state(&self, slot_idx: usize, layer: usize) -> &[f32] {
        self.slot(slot_idx).recurrent_states[layer].as_slice()
    }

    fn recurrent_buffers_mut(&mut self, slot_idx: usize, layer: usize) -> (&mut [f32], &mut [f32]) {
        let slot = self.slot_mut(slot_idx);
        slot.touch_layer_state(layer);
        let conv_state = slot.conv_states[layer].as_mut_slice();
        let recurrent_state = slot.recurrent_states[layer].as_mut_slice();
        (conv_state, recurrent_state)
    }

    fn touch_buffers(&mut self, slot_idx: usize, layer: usize) -> (u64, u64) {
        let slot = self.slot_mut(slot_idx);
        slot.touch_layer_state(layer);
        (
            slot.conv_state_generation(layer),
            slot.recurrent_state_generation(layer),
        )
    }

    fn snapshot(&self, slot_idx: usize) -> Qwen35RecurrentSlotSnapshot {
        self.slot(slot_idx).snapshot()
    }

    fn restore(&mut self, slot_idx: usize, snapshot: &Qwen35RecurrentSlotSnapshot) {
        self.slot_mut(slot_idx).restore(snapshot);
        self.free_slots.retain(|&s| s != slot_idx);
    }

    fn clone_slot_preserving_ownership(
        &mut self,
        src_slot_idx: usize,
        dst_slot_idx: usize,
        recurrent_layers: &[bool],
        conv_cache_len: usize,
        conv_dim: usize,
        recurrent_state_len: usize,
    ) {
        self.ensure_allocated_slot(
            dst_slot_idx,
            recurrent_layers,
            conv_cache_len,
            conv_dim,
            recurrent_state_len,
        );
        let src = self.slot(src_slot_idx).clone();
        self.slot_mut(dst_slot_idx)
            .clone_preserving_ownership_from(&src);
        self.free_slots.retain(|&s| s != dst_slot_idx);
    }

    fn seqlen_offset(&self, slot_idx: usize) -> usize {
        self.slot(slot_idx).seqlen_offset
    }

    fn set_seqlen_offset(&mut self, slot_idx: usize, seqlen_offset: usize) {
        self.slot_mut(slot_idx).seqlen_offset = seqlen_offset;
    }

    fn increment_seqlen_offset(&mut self, slot_idx: usize, delta: usize) {
        self.slot_mut(slot_idx).seqlen_offset += delta;
    }

    fn conv_state_generation(&self, slot_idx: usize, layer: usize) -> u64 {
        self.slot(slot_idx).conv_state_generation(layer)
    }

    fn recurrent_state_generation(&self, slot_idx: usize, layer: usize) -> u64 {
        self.slot(slot_idx).recurrent_state_generation(layer)
    }

    fn note_backend_recurrent_state(&mut self, slot_idx: usize, layer: usize) -> u64 {
        self.slot_mut(slot_idx).note_backend_recurrent_state(layer)
    }

    fn note_backend_conv_state(&mut self, slot_idx: usize, layer: usize) -> u64 {
        self.slot_mut(slot_idx).note_backend_conv_state(layer)
    }

    fn cpu_recurrent_state_stale(&self, slot_idx: usize, layer: usize) -> bool {
        self.slot(slot_idx).cpu_recurrent_state_stale(layer)
    }

    fn cpu_conv_state_stale(&self, slot_idx: usize, layer: usize) -> bool {
        self.slot(slot_idx).cpu_conv_state_stale(layer)
    }

    #[allow(dead_code)]
    fn conv_state_owner(&self, slot_idx: usize, layer: usize) -> Qwen35StateOwner {
        self.slot(slot_idx).conv_state_owner(layer)
    }

    #[allow(dead_code)]
    fn recurrent_state_owner(&self, slot_idx: usize, layer: usize) -> Qwen35StateOwner {
        self.slot(slot_idx).recurrent_state_owner(layer)
    }

    fn layer_state_owner(&self, slot_idx: usize, layer: usize) -> Qwen35LayerStateOwner {
        self.slot(slot_idx).layer_state_owner(layer)
    }

    fn mark_layer_state_backend_owned(&mut self, slot_idx: usize, layer: usize) {
        self.slot_mut(slot_idx)
            .mark_layer_state_backend_owned(layer);
    }

    fn conv_state_pristine_zero(&self, slot_idx: usize, layer: usize) -> bool {
        self.slot(slot_idx).conv_state_pristine_zero(layer)
    }

    fn recurrent_state_pristine_zero(&self, slot_idx: usize, layer: usize) -> bool {
        self.slot(slot_idx).recurrent_state_pristine_zero(layer)
    }

    fn overwrite_recurrent_state_from_backend(
        &mut self,
        slot_idx: usize,
        layer: usize,
        recurrent_state: &[f32],
        generation: u64,
    ) {
        self.slot_mut(slot_idx)
            .overwrite_recurrent_state_from_backend(layer, recurrent_state, generation);
    }

    fn overwrite_conv_state_from_backend(
        &mut self,
        slot_idx: usize,
        layer: usize,
        conv_state: &[f32],
        generation: u64,
    ) {
        self.slot_mut(slot_idx)
            .overwrite_conv_state_from_backend(layer, conv_state, generation);
    }
}

/// GPU-resident recurrent state buffers for one slot.
struct Qwen35GpuRecurrentSlotBuffers {
    /// conv_states[layer_idx] — `None` for non-recurrent layers.
    conv_states: Vec<Option<ax_engine_metal::MetalBuffer>>,
    /// recurrent_states[layer_idx] — `None` for non-recurrent layers.
    recurrent_states: Vec<Option<ax_engine_metal::MetalBuffer>>,
}

/// GPU-resident recurrent state for all slots.
struct Qwen35GpuRecurrentState {
    device: ax_engine_metal::MetalDevice,
    slots: Vec<Qwen35GpuRecurrentSlotBuffers>,
    conv_state_stride: usize,
    recurrent_state_stride: usize,
}

impl Qwen35GpuRecurrentState {
    fn allocate_slot(
        device: &ax_engine_metal::MetalDevice,
        recurrent_layers: &[bool],
        conv_state_stride: usize,
        recurrent_state_stride: usize,
    ) -> anyhow::Result<Qwen35GpuRecurrentSlotBuffers> {
        let mut conv_states = Vec::with_capacity(recurrent_layers.len());
        let mut recurrent_states = Vec::with_capacity(recurrent_layers.len());
        for &is_recurrent in recurrent_layers {
            if is_recurrent {
                let mut conv_buf = ax_engine_metal::MetalBuffer::new(
                    device.device(),
                    conv_state_stride * std::mem::size_of::<f32>(),
                )?;
                unsafe {
                    conv_buf.as_mut_slice::<f32>()[..conv_state_stride].fill(0.0);
                }
                let mut rec_buf = ax_engine_metal::MetalBuffer::new(
                    device.device(),
                    recurrent_state_stride * std::mem::size_of::<f32>(),
                )?;
                unsafe {
                    rec_buf.as_mut_slice::<f32>()[..recurrent_state_stride].fill(0.0);
                }
                conv_states.push(Some(conv_buf));
                recurrent_states.push(Some(rec_buf));
            } else {
                conv_states.push(None);
                recurrent_states.push(None);
            }
        }
        Ok(Qwen35GpuRecurrentSlotBuffers {
            conv_states,
            recurrent_states,
        })
    }

    fn clear_slot(&mut self, slot_idx: usize) {
        let slot = &mut self.slots[slot_idx];
        for buf in slot.conv_states.iter_mut().flatten() {
            unsafe {
                buf.as_mut_slice::<f32>()[..self.conv_state_stride].fill(0.0);
            }
        }
        for buf in slot.recurrent_states.iter_mut().flatten() {
            unsafe {
                buf.as_mut_slice::<f32>()[..self.recurrent_state_stride].fill(0.0);
            }
        }
    }
}

/// CPU-side hybrid cache for Qwen3.5.
pub struct Qwen35Kv {
    attention: CpuKv,
    attention_gpu: Option<GpuKv>,
    attention_gpu_device: Option<ax_engine_metal::MetalDevice>,
    attention_cpu_dirty: bool,
    attention_cpu_valid_prefix_len: usize,
    seq_len: usize,
    attention_n_kv_heads: usize,
    attention_head_dim: usize,
    attention_max_seq_len: usize,
    attention_page_size: usize,
    recurrent_layers: Vec<bool>,
    conv_cache_len: usize,
    conv_dim: usize,
    recurrent_state_len: usize,
    active_slot: usize,
    batch_slot_indices: Vec<usize>,
    recurrent_pool: Qwen35RecurrentPool,
    /// GPU-resident recurrent state (conv + SSM) for all slots.
    /// When present, these buffers are authoritative — CPU state is lazy backup.
    gpu_recurrent: Option<Qwen35GpuRecurrentState>,
}

impl std::fmt::Debug for Qwen35Kv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen35Kv")
            .field("seq_len", &self.seq_len)
            .field("active_slot", &self.active_slot)
            .field("batch_slot_indices", &self.batch_slot_indices)
            .field("attention_n_kv_heads", &self.attention_n_kv_heads)
            .field("attention_head_dim", &self.attention_head_dim)
            .field("attention_max_seq_len", &self.attention_max_seq_len)
            .field("attention_page_size", &self.attention_page_size)
            .field("has_gpu_attention", &self.attention_gpu.is_some())
            .field("has_gpu_recurrent", &self.gpu_recurrent.is_some())
            .field("recurrent_layers", &self.recurrent_layers.len())
            .field("conv_cache_len", &self.conv_cache_len)
            .field("conv_dim", &self.conv_dim)
            .field("recurrent_state_len", &self.recurrent_state_len)
            .finish()
    }
}

pub(crate) enum Qwen35PreparedRecurrentStateBatch<'a> {
    Direct {
        kind: Qwen35PreparedRecurrentStateBatchKind,
        layer_idx: usize,
        slot_indices: &'a [usize],
        conv_state: &'a mut [f32],
        recurrent_state: &'a mut [f32],
        conv_state_stride: usize,
        recurrent_state_stride: usize,
    },
    Gathered {
        kind: Qwen35PreparedRecurrentStateBatchKind,
        qwen_kv: &'a mut Qwen35Kv,
        layer_idx: usize,
        slot_indices: &'a [usize],
        conv_state_batch: Vec<f32>,
        recurrent_state_batch: Vec<f32>,
        conv_state_stride: usize,
        recurrent_state_stride: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35PreparedRecurrentStateBatchKind {
    CpuDirect,
    CpuDirectMaterializedFromBackend,
    CpuGathered,
    CpuGatheredMaterializedFromBackend,
}

impl<'a> Qwen35PreparedRecurrentStateBatch<'a> {
    pub(crate) fn kind(&self) -> Qwen35PreparedRecurrentStateBatchKind {
        match self {
            Self::Direct { kind, .. } | Self::Gathered { kind, .. } => *kind,
        }
    }

    pub(crate) fn state_batch(&mut self) -> crate::backend::Qwen35RecurrentStateBatch<'_> {
        match self {
            Self::Direct {
                layer_idx,
                slot_indices,
                conv_state,
                recurrent_state,
                conv_state_stride,
                recurrent_state_stride,
                ..
            } => crate::backend::Qwen35RecurrentStateBatch::new(
                *layer_idx,
                slot_indices,
                conv_state,
                recurrent_state,
                *conv_state_stride,
                *recurrent_state_stride,
            ),
            Self::Gathered {
                layer_idx,
                slot_indices,
                conv_state_batch,
                recurrent_state_batch,
                conv_state_stride,
                recurrent_state_stride,
                ..
            } => crate::backend::Qwen35RecurrentStateBatch::new(
                *layer_idx,
                slot_indices,
                conv_state_batch,
                recurrent_state_batch,
                *conv_state_stride,
                *recurrent_state_stride,
            ),
        }
    }

    pub(crate) fn finish(self) {
        match self {
            Self::Direct { .. } => {}
            Self::Gathered {
                qwen_kv,
                layer_idx,
                slot_indices,
                conv_state_batch,
                recurrent_state_batch,
                ..
            } => qwen_kv.scatter_recurrent_state_batch(
                slot_indices,
                layer_idx,
                &conv_state_batch,
                &recurrent_state_batch,
            ),
        }
    }
}

impl Qwen35Kv {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        full_attention_interval: usize,
        conv_kernel: usize,
        inner_size: usize,
        state_size: usize,
        time_step_rank: usize,
        group_count: usize,
    ) -> Self {
        let page_size = recommended_page_size(n_kv_heads, head_dim);
        Self::new_with_attention_page_size(
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            page_size,
            full_attention_interval,
            conv_kernel,
            inner_size,
            state_size,
            time_step_rank,
            group_count,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_attention_page_size(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        attention_page_size: usize,
        full_attention_interval: usize,
        conv_kernel: usize,
        inner_size: usize,
        state_size: usize,
        time_step_rank: usize,
        group_count: usize,
    ) -> Self {
        assert!(
            full_attention_interval > 0,
            "qwen35 full_attention_interval must be > 0"
        );
        assert!(conv_kernel > 0, "qwen35 conv_kernel must be > 0");
        assert!(inner_size > 0, "qwen35 inner_size must be > 0");
        assert!(state_size > 0, "qwen35 state_size must be > 0");
        assert!(time_step_rank > 0, "qwen35 time_step_rank must be > 0");
        assert!(group_count > 0, "qwen35 group_count must be > 0");
        assert!(
            inner_size == state_size * time_step_rank,
            "qwen35 inner_size ({inner_size}) must equal state_size ({state_size}) * time_step_rank ({time_step_rank})"
        );
        assert!(
            time_step_rank.is_multiple_of(group_count),
            "qwen35 time_step_rank ({time_step_rank}) must be a multiple of group_count ({group_count})"
        );
        let recurrent_layers = (0..n_layers)
            .map(|layer| (layer + 1) % full_attention_interval != 0)
            .collect::<Vec<_>>();
        let conv_cache_len = conv_kernel.saturating_sub(1);
        let conv_dim = inner_size + 2 * group_count * state_size;
        let recurrent_state_len = time_step_rank * state_size * state_size;
        let recurrent_pool = Qwen35RecurrentPool::new(
            &recurrent_layers,
            conv_cache_len,
            conv_dim,
            recurrent_state_len,
        );
        Self {
            attention: CpuKv::with_config(&KvCacheConfig {
                n_layers,
                n_kv_heads,
                head_dim,
                max_seq_len,
                page_size: attention_page_size,
                dtype: KvDtype::F32,
            }),
            attention_gpu: None,
            attention_gpu_device: None,
            attention_cpu_dirty: false,
            attention_cpu_valid_prefix_len: 0,
            seq_len: 0,
            attention_n_kv_heads: n_kv_heads,
            attention_head_dim: head_dim,
            attention_max_seq_len: max_seq_len,
            attention_page_size,
            recurrent_layers: recurrent_layers.clone(),
            conv_cache_len,
            conv_dim,
            recurrent_state_len,
            active_slot: 0,
            batch_slot_indices: Vec::new(),
            recurrent_pool,
            gpu_recurrent: None,
        }
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn gpu_attention(&self) -> Option<&GpuKv> {
        self.attention_gpu.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn gpu_attention_mut(&mut self) -> Option<&mut GpuKv> {
        self.attention_gpu.as_mut()
    }

    pub fn enable_gpu_attention(
        &mut self,
        device: &ax_engine_metal::MetalDevice,
        dtype: GpuKvDtype,
    ) -> anyhow::Result<()> {
        if self.attention_gpu.is_some() {
            return Ok(());
        }

        let gpu_attention = GpuKv::new_with_dtype(
            device,
            self.recurrent_layers.len(),
            self.attention_n_kv_heads,
            self.attention_head_dim,
            self.attention_max_seq_len,
            self.attention_page_size,
            dtype,
        )?;
        let gpu_device = device.clone_sharing_device()?;
        self.attention_gpu = Some(gpu_attention);
        self.attention_gpu_device = Some(gpu_device);
        self.sync_gpu_attention_from_cpu();
        Ok(())
    }

    /// Allocate GPU-resident MetalBuffers for all recurrent layer state.
    /// After this call, `gpu_recurrent_buffers()` returns direct buffer references
    /// and `sync_qwen35_slot_buffers_from_kv` is no longer needed for prefill.
    pub fn enable_gpu_recurrent_state(
        &mut self,
        device: &ax_engine_metal::MetalDevice,
    ) -> anyhow::Result<()> {
        if self.gpu_recurrent.is_some() {
            return Ok(());
        }
        let conv_state_stride = self.conv_cache_len * self.conv_dim;
        let recurrent_state_stride = self.recurrent_state_len;
        let gpu_device = device.clone_sharing_device()?;
        let mut slots = Vec::with_capacity(self.recurrent_pool.slots.len());
        for _slot in &self.recurrent_pool.slots {
            slots.push(Qwen35GpuRecurrentState::allocate_slot(
                device,
                &self.recurrent_layers,
                conv_state_stride,
                recurrent_state_stride,
            )?);
        }
        self.gpu_recurrent = Some(Qwen35GpuRecurrentState {
            device: gpu_device,
            slots,
            conv_state_stride,
            recurrent_state_stride,
        });
        Ok(())
    }

    /// Returns `true` if GPU-resident recurrent state buffers are available.
    pub fn has_gpu_recurrent_state(&self) -> bool {
        self.gpu_recurrent.is_some()
    }

    /// Returns GPU MetalBuffer references for recurrent state at (slot, layer).
    /// Returns `(conv_state, recurrent_state)`.
    pub fn gpu_recurrent_buffers(
        &self,
        slot_idx: usize,
        layer: usize,
    ) -> Option<(&ax_engine_metal::MetalBuffer, &ax_engine_metal::MetalBuffer)> {
        let gpu = self.gpu_recurrent.as_ref()?;
        let slot = &gpu.slots[slot_idx];
        let conv = slot.conv_states[layer].as_ref()?;
        let rec = slot.recurrent_states[layer].as_ref()?;
        Some((conv, rec))
    }

    /// Returns mutable GPU MetalBuffer references for recurrent state at (slot, layer).
    #[allow(dead_code)]
    pub(crate) fn gpu_recurrent_buffers_mut(
        &mut self,
        slot_idx: usize,
        layer: usize,
    ) -> Option<(
        &mut ax_engine_metal::MetalBuffer,
        &mut ax_engine_metal::MetalBuffer,
    )> {
        let gpu = self.gpu_recurrent.as_mut()?;
        let slot = &mut gpu.slots[slot_idx];
        let conv = slot.conv_states[layer].as_mut()?;
        let rec = slot.recurrent_states[layer].as_mut()?;
        Some((conv, rec))
    }

    pub fn is_recurrent_layer(&self, layer: usize) -> bool {
        self.recurrent_layers[layer]
    }

    pub(crate) fn layer_count(&self) -> usize {
        self.recurrent_layers.len()
    }

    pub fn attention_append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        self.attention.append_and_advance(layer, k, v);
        if self.ensure_gpu_attention_capacity_for(self.seq_len + 1)
            && let Some(gpu_attention) = self.attention_gpu.as_mut()
        {
            gpu_attention.append_layer(layer, k, v);
        }
    }

    pub fn attention_append_batch(
        &mut self,
        layer: usize,
        k_batch: &[f32],
        v_batch: &[f32],
        n_tokens: usize,
    ) {
        let gpu_ok = self.ensure_gpu_attention_capacity_for(self.seq_len + n_tokens)
            && self.attention_gpu.is_some();
        self.attention
            .append_batch(layer, k_batch, v_batch, n_tokens);
        if gpu_ok {
            if let Some(gpu_attention) = self.attention_gpu.as_mut() {
                gpu_attention.append_layer_batch(layer, k_batch, v_batch, n_tokens);
            }
        } else if self.attention_gpu.is_some() {
            self.attention_cpu_dirty = true;
        }
    }

    pub(crate) fn attention_append_batch_cpu_mirror(
        &mut self,
        layer: usize,
        k_batch: &[f32],
        v_batch: &[f32],
        n_tokens: usize,
    ) {
        self.attention
            .append_batch(layer, k_batch, v_batch, n_tokens);
    }

    pub(crate) fn mark_attention_cpu_dirty(&mut self) {
        if self.attention_gpu.is_some() {
            self.attention_cpu_dirty = true;
        }
    }

    pub(crate) fn sync_attention_cpu_from_gpu_if_needed(&mut self) {
        if !self.attention_cpu_dirty {
            return;
        }
        let Some(kv_stride) = self.attention_gpu.as_ref().map(GpuKv::kv_stride) else {
            self.attention_cpu_dirty = false;
            self.attention_cpu_valid_prefix_len = self.seq_len;
            return;
        };
        let valid_prefix = self.attention_cpu_valid_prefix_len.min(self.seq_len);
        if valid_prefix == self.seq_len {
            self.attention_cpu_dirty = false;
            self.attention_cpu_valid_prefix_len = self.seq_len;
            return;
        }

        let delta_tokens = self.seq_len - valid_prefix;
        let prefix_elems = delta_tokens * kv_stride;
        let mut k_prefix = vec![0.0f32; prefix_elems];
        let mut v_prefix = vec![0.0f32; prefix_elems];
        for layer in 0..self.recurrent_layers.len() {
            self.attention_gpu
                .as_ref()
                .expect("gpu attention should exist while syncing cpu mirror")
                .read_layer_range_into(
                    layer,
                    valid_prefix,
                    delta_tokens,
                    &mut k_prefix,
                    &mut v_prefix,
                );
            for token_offset in 0..delta_tokens {
                let src_start = token_offset * kv_stride;
                let src_end = src_start + kv_stride;
                self.attention
                    .k_token_mut(layer, valid_prefix + token_offset)
                    .copy_from_slice(&k_prefix[src_start..src_end]);
                self.attention
                    .v_token_mut(layer, valid_prefix + token_offset)
                    .copy_from_slice(&v_prefix[src_start..src_end]);
            }
        }
        self.attention_cpu_dirty = false;
        self.attention_cpu_valid_prefix_len = self.seq_len;
    }

    pub fn attention_k_slice_including_current(&self, layer: usize, n: usize) -> &[f32] {
        self.attention.k_slice_including_current(layer, n)
    }

    pub fn attention_v_slice_including_current(&self, layer: usize, n: usize) -> &[f32] {
        self.attention.v_slice_including_current(layer, n)
    }

    pub fn active_slot(&self) -> usize {
        self.active_slot
    }

    pub(crate) fn has_recurrent_slot(&self, slot_idx: usize) -> bool {
        self.recurrent_pool.is_allocated_slot(slot_idx)
    }

    pub(crate) fn has_layer(&self, layer_idx: usize) -> bool {
        layer_idx < self.recurrent_layers.len()
    }

    pub(crate) fn assert_valid_recurrent_slot_batch(&self, slot_indices: &[usize], layer: usize) {
        assert!(
            self.has_layer(layer),
            "qwen35 recurrent slot batch layer out of range"
        );
        assert!(
            self.is_recurrent_layer(layer),
            "qwen35 recurrent slot batch requires a recurrent layer"
        );
        for (i, &slot_idx) in slot_indices.iter().enumerate() {
            assert!(
                self.has_recurrent_slot(slot_idx),
                "qwen35 recurrent slot batch references an unallocated slot"
            );
            let slot_seq_len = self.recurrent_seqlen_offset(slot_idx);
            assert!(
                slot_seq_len == self.seq_len,
                "qwen35 recurrent slot batch slot {slot_idx} has seqlen_offset {slot_seq_len} != shared attention seq_len {}",
                self.seq_len
            );
            for &other_slot_idx in &slot_indices[..i] {
                assert!(
                    slot_idx != other_slot_idx,
                    "qwen35 recurrent slot batch must not contain duplicate slots"
                );
            }
        }
    }

    fn assert_valid_recurrent_slots_on_shared_timeline(&self, slot_indices: &[usize]) {
        assert!(
            !slot_indices.is_empty(),
            "qwen35 recurrent slot batch requires at least one slot"
        );
        for (i, &slot_idx) in slot_indices.iter().enumerate() {
            assert!(
                self.has_recurrent_slot(slot_idx),
                "qwen35 recurrent slot batch references an unallocated slot"
            );
            let slot_seq_len = self.recurrent_seqlen_offset(slot_idx);
            assert!(
                slot_seq_len == self.seq_len,
                "qwen35 recurrent slot batch slot {slot_idx} has seqlen_offset {slot_seq_len} != shared attention seq_len {}",
                self.seq_len
            );
            for &other_slot_idx in &slot_indices[..i] {
                assert!(
                    slot_idx != other_slot_idx,
                    "qwen35 recurrent slot batch must not contain duplicate slots"
                );
            }
        }
    }

    fn assert_slot_recurrent_state_materialized_on_cpu(&self, slot_idx: usize) {
        assert!(
            self.has_recurrent_slot(slot_idx),
            "qwen35 recurrent slot is not allocated"
        );
        for (layer_idx, &is_recurrent) in self.recurrent_layers.iter().enumerate() {
            if !is_recurrent {
                continue;
            }
            assert!(
                !self.recurrent_state_cpu_stale(slot_idx, layer_idx),
                "cannot snapshot qwen35 recurrent slot while backend-owned recurrent state is not materialized on CPU"
            );
            assert!(
                !self.conv_state_cpu_stale(slot_idx, layer_idx),
                "cannot snapshot qwen35 recurrent slot while backend-owned conv state is not materialized on CPU"
            );
        }
    }

    pub fn set_active_slot(&mut self, slot_idx: usize) {
        assert!(
            self.recurrent_pool.is_allocated_slot(slot_idx),
            "cannot activate free qwen35 recurrent slot"
        );
        let slot_seq_len = self.recurrent_pool.seqlen_offset(slot_idx);
        assert!(
            slot_seq_len == self.seq_len,
            "cannot activate qwen35 recurrent slot with seqlen_offset {slot_seq_len} != shared attention seq_len {}",
            self.seq_len
        );
        self.active_slot = slot_idx;
        self.clear_batch_slot_indices();
    }

    pub fn set_batch_slot_indices(&mut self, slot_indices: &[usize]) {
        if slot_indices.is_empty() {
            self.clear_batch_slot_indices();
            return;
        }
        self.assert_valid_recurrent_slots_on_shared_timeline(slot_indices);
        assert!(
            slot_indices.contains(&self.active_slot),
            "qwen35 recurrent slot batch requires the active slot to be included"
        );
        self.batch_slot_indices.clear();
        self.batch_slot_indices.extend_from_slice(slot_indices);
    }

    pub fn clear_batch_slot_indices(&mut self) {
        self.batch_slot_indices.clear();
    }

    pub(crate) fn recurrent_batch_slot_indices(&self) -> Cow<'_, [usize]> {
        if self.batch_slot_indices.is_empty() {
            Cow::Owned(vec![self.active_slot])
        } else {
            Cow::Borrowed(&self.batch_slot_indices)
        }
    }

    pub fn allocate_recurrent_slot(&mut self) -> usize {
        let slot_idx = self.recurrent_pool.allocate(
            &self.recurrent_layers,
            self.conv_cache_len,
            self.conv_dim,
            self.recurrent_state_len,
        );
        if let Some(gpu) = self.gpu_recurrent.as_mut() {
            if slot_idx < gpu.slots.len() {
                // Reused slot: zero GPU buffers to match cleared CPU state.
                gpu.clear_slot(slot_idx);
            }
            while gpu.slots.len() <= slot_idx {
                let new_slot = Qwen35GpuRecurrentState::allocate_slot(
                    &gpu.device,
                    &self.recurrent_layers,
                    gpu.conv_state_stride,
                    gpu.recurrent_state_stride,
                )
                .expect("Failed to allocate GPU recurrent slot buffers");
                gpu.slots.push(new_slot);
            }
        }
        slot_idx
    }

    pub fn free_recurrent_slot(&mut self, slot_idx: usize) {
        assert!(
            slot_idx != 0,
            "cannot free reserved qwen35 recurrent slot 0"
        );
        assert!(
            slot_idx != self.active_slot,
            "cannot free active qwen35 recurrent slot"
        );
        assert!(
            self.recurrent_pool.is_allocated_slot(slot_idx),
            "cannot free inactive qwen35 recurrent slot"
        );
        self.clear_batch_slot_indices();
        self.recurrent_pool.free(slot_idx);
    }

    pub fn recurrent_slot_snapshot(&mut self, slot_idx: usize) -> Qwen35RecurrentSlotSnapshot {
        // If GPU-resident state is authoritative, materialize it to CPU first.
        self.materialize_gpu_recurrent_to_cpu(slot_idx);
        self.assert_slot_recurrent_state_materialized_on_cpu(slot_idx);
        self.recurrent_pool.snapshot(slot_idx)
    }

    /// Copy GPU recurrent state back to CPU for the given slot,
    /// making the CPU copy current for snapshotting/cloning.
    fn materialize_gpu_recurrent_to_cpu(&mut self, slot_idx: usize) {
        let Some(gpu) = self.gpu_recurrent.as_ref() else {
            return;
        };
        if slot_idx >= gpu.slots.len() {
            return;
        }
        let slot_cpu = &mut self.recurrent_pool.slots[slot_idx];
        let gpu_slot = &gpu.slots[slot_idx];
        for (layer_idx, &is_recurrent) in self.recurrent_layers.iter().enumerate() {
            if !is_recurrent {
                continue;
            }
            if slot_cpu.cpu_conv_state_stale(layer_idx)
                && let Some(buf) = gpu_slot.conv_states[layer_idx].as_ref()
            {
                let stride = gpu
                    .conv_state_stride
                    .min(slot_cpu.conv_states[layer_idx].len());
                let generation = slot_cpu.conv_state_generation(layer_idx);
                unsafe {
                    slot_cpu.conv_states[layer_idx][..stride]
                        .copy_from_slice(&buf.as_slice::<f32>()[..stride]);
                }
                slot_cpu.cpu_materialized_conv_generations[layer_idx] = generation;
                slot_cpu.conv_state_pristine_zero[layer_idx] = false;
            }
            if slot_cpu.cpu_recurrent_state_stale(layer_idx)
                && let Some(buf) = gpu_slot.recurrent_states[layer_idx].as_ref()
            {
                let stride = gpu
                    .recurrent_state_stride
                    .min(slot_cpu.recurrent_states[layer_idx].len());
                let generation = slot_cpu.recurrent_state_generation(layer_idx);
                unsafe {
                    slot_cpu.recurrent_states[layer_idx][..stride]
                        .copy_from_slice(&buf.as_slice::<f32>()[..stride]);
                }
                slot_cpu.cpu_materialized_recurrent_generations[layer_idx] = generation;
                slot_cpu.recurrent_state_pristine_zero[layer_idx] = false;
            }
        }
    }

    pub fn clone_recurrent_slot(&mut self, src_slot_idx: usize, dst_slot_idx: usize) {
        let snapshot = self.recurrent_slot_snapshot(src_slot_idx);
        self.restore_recurrent_slot(dst_slot_idx, &snapshot);
    }

    pub(crate) fn clone_recurrent_slot_preserving_ownership(
        &mut self,
        src_slot_idx: usize,
        dst_slot_idx: usize,
    ) {
        assert!(
            self.has_recurrent_slot(src_slot_idx),
            "cannot clone from free qwen35 recurrent slot"
        );
        self.clear_batch_slot_indices();
        self.recurrent_pool.clone_slot_preserving_ownership(
            src_slot_idx,
            dst_slot_idx,
            &self.recurrent_layers,
            self.conv_cache_len,
            self.conv_dim,
            self.recurrent_state_len,
        );
        // Copy GPU buffers from source to destination slot.
        self.copy_gpu_recurrent_slot(src_slot_idx, dst_slot_idx);
    }

    /// Copy GPU recurrent buffers from one slot to another.
    fn copy_gpu_recurrent_slot(&mut self, src_slot_idx: usize, dst_slot_idx: usize) {
        let Some(gpu) = self.gpu_recurrent.as_mut() else {
            return;
        };
        // Ensure destination slot has GPU buffers allocated.
        while gpu.slots.len() <= dst_slot_idx {
            let new_slot = Qwen35GpuRecurrentState::allocate_slot(
                &gpu.device,
                &self.recurrent_layers,
                gpu.conv_state_stride,
                gpu.recurrent_state_stride,
            )
            .expect("Failed to allocate GPU recurrent slot buffers for clone");
            gpu.slots.push(new_slot);
        }
        if src_slot_idx >= gpu.slots.len() {
            return;
        }
        for (layer_idx, &is_recurrent) in self.recurrent_layers.iter().enumerate() {
            if !is_recurrent {
                continue;
            }
            let conv_stride = gpu.conv_state_stride;
            let rec_stride = gpu.recurrent_state_stride;
            // Copy conv state.
            let src_data: Vec<f32> =
                if let Some(buf) = gpu.slots[src_slot_idx].conv_states[layer_idx].as_ref() {
                    unsafe { buf.as_slice::<f32>()[..conv_stride].to_vec() }
                } else {
                    continue;
                };
            if let Some(buf) = gpu.slots[dst_slot_idx].conv_states[layer_idx].as_mut() {
                unsafe {
                    buf.as_mut_slice::<f32>()[..conv_stride].copy_from_slice(&src_data);
                }
            }
            // Copy recurrent state.
            let src_data: Vec<f32> =
                if let Some(buf) = gpu.slots[src_slot_idx].recurrent_states[layer_idx].as_ref() {
                    unsafe { buf.as_slice::<f32>()[..rec_stride].to_vec() }
                } else {
                    continue;
                };
            if let Some(buf) = gpu.slots[dst_slot_idx].recurrent_states[layer_idx].as_mut() {
                unsafe {
                    buf.as_mut_slice::<f32>()[..rec_stride].copy_from_slice(&src_data);
                }
            }
        }
    }

    pub fn restore_recurrent_slot(
        &mut self,
        slot_idx: usize,
        snapshot: &Qwen35RecurrentSlotSnapshot,
    ) {
        if slot_idx == self.active_slot {
            assert!(
                snapshot.seqlen_offset == self.seq_len,
                "cannot restore active qwen35 recurrent slot with seqlen_offset {} != shared attention seq_len {}",
                snapshot.seqlen_offset,
                self.seq_len
            );
        }
        self.recurrent_pool.ensure_allocated_slot(
            slot_idx,
            &self.recurrent_layers,
            self.conv_cache_len,
            self.conv_dim,
            self.recurrent_state_len,
        );
        self.clear_batch_slot_indices();
        self.recurrent_pool.restore(slot_idx, snapshot);
        // Sync restored CPU state to GPU buffers so they match.
        self.sync_gpu_recurrent_from_cpu(slot_idx);
    }

    /// Copy CPU recurrent state to GPU buffers for the given slot.
    /// Called after CPU state is modified (restore, clone) to keep GPU in sync.
    fn sync_gpu_recurrent_from_cpu(&mut self, slot_idx: usize) {
        let Some(gpu) = self.gpu_recurrent.as_mut() else {
            return;
        };
        if slot_idx >= gpu.slots.len() {
            return;
        }
        let slot_cpu = &self.recurrent_pool.slots[slot_idx];
        let gpu_slot = &mut gpu.slots[slot_idx];
        for (layer_idx, &is_recurrent) in self.recurrent_layers.iter().enumerate() {
            if !is_recurrent {
                continue;
            }
            if let Some(buf) = gpu_slot.conv_states[layer_idx].as_mut() {
                let cpu_data = &slot_cpu.conv_states[layer_idx];
                let stride = gpu.conv_state_stride.min(cpu_data.len());
                unsafe {
                    buf.as_mut_slice::<f32>()[..stride].copy_from_slice(&cpu_data[..stride]);
                }
            }
            if let Some(buf) = gpu_slot.recurrent_states[layer_idx].as_mut() {
                let cpu_data = &slot_cpu.recurrent_states[layer_idx];
                let stride = gpu.recurrent_state_stride.min(cpu_data.len());
                unsafe {
                    buf.as_mut_slice::<f32>()[..stride].copy_from_slice(&cpu_data[..stride]);
                }
            }
        }
    }

    pub fn snapshot_active_slot(&mut self) -> Qwen35KvSnapshot {
        self.snapshot_slot(self.active_slot)
    }

    pub fn attention_snapshot(&self) -> Qwen35AttentionSnapshot {
        Qwen35AttentionSnapshot {
            attention: self.attention.snapshot(),
            seq_len: self.seq_len,
        }
    }

    pub fn snapshot_slot(&mut self, slot_idx: usize) -> Qwen35KvSnapshot {
        self.materialize_gpu_recurrent_to_cpu(slot_idx);
        self.assert_slot_recurrent_state_materialized_on_cpu(slot_idx);
        assert!(
            slot_idx == self.active_slot,
            "cannot snapshot inactive qwen35 recurrent slot while attention KV is shared"
        );
        let slot_seq_len = self.recurrent_pool.seqlen_offset(slot_idx);
        assert!(
            slot_seq_len == self.seq_len,
            "cannot snapshot qwen35 recurrent slot with seqlen_offset {slot_seq_len} != shared attention seq_len {}",
            self.seq_len
        );
        Qwen35KvSnapshot {
            attention: self.attention.snapshot(),
            recurrent_slot: self.recurrent_pool.snapshot(slot_idx),
            seq_len: self.seq_len,
            active_slot: slot_idx,
        }
    }

    pub fn restore_snapshot(&mut self, snapshot: &Qwen35KvSnapshot) {
        assert!(
            snapshot.recurrent_slot.seqlen_offset == snapshot.seq_len,
            "cannot restore qwen35 snapshot with recurrent seqlen_offset {} != shared attention seq_len {}",
            snapshot.recurrent_slot.seqlen_offset,
            snapshot.seq_len
        );
        self.attention.restore(&snapshot.attention);
        self.recurrent_pool.clear_all();
        self.recurrent_pool.ensure_allocated_slot(
            snapshot.active_slot,
            &self.recurrent_layers,
            self.conv_cache_len,
            self.conv_dim,
            self.recurrent_state_len,
        );
        self.recurrent_pool
            .restore(snapshot.active_slot, &snapshot.recurrent_slot);
        self.seq_len = snapshot.seq_len;
        self.active_slot = snapshot.active_slot;
        self.attention_cpu_dirty = false;
        self.attention_cpu_valid_prefix_len = self.seq_len;
        self.clear_batch_slot_indices();
        self.sync_gpu_attention_from_cpu();
    }

    pub fn restore_attention_snapshot(&mut self, snapshot: &Qwen35AttentionSnapshot) {
        self.attention.restore(&snapshot.attention);
        self.seq_len = snapshot.seq_len;
        self.attention_cpu_dirty = false;
        self.attention_cpu_valid_prefix_len = self.seq_len;
        self.clear_batch_slot_indices();
        self.sync_gpu_attention_from_cpu();
    }

    pub fn truncate_attention_to(&mut self, pos: usize) {
        assert!(
            pos <= self.seq_len,
            "cannot truncate qwen35 attention to future pos {pos} > seq_len {}",
            self.seq_len
        );
        self.attention.truncate_to(pos);
        if let Some(gpu_attention) = self.attention_gpu.as_mut() {
            gpu_attention.truncate_to(pos);
        }
        self.seq_len = pos;
        self.clear_batch_slot_indices();
        let valid_prefix = self.attention_cpu_valid_prefix_len.min(pos);
        self.attention_cpu_valid_prefix_len = valid_prefix;
        self.attention_cpu_dirty = self.attention_gpu.is_some() && valid_prefix < pos;
    }

    pub fn recurrent_seqlen_offset(&self, slot_idx: usize) -> usize {
        self.recurrent_pool.seqlen_offset(slot_idx)
    }

    pub fn set_recurrent_seqlen_offset(&mut self, slot_idx: usize, seqlen_offset: usize) {
        if slot_idx == self.active_slot {
            assert!(
                seqlen_offset == self.seq_len,
                "cannot set active qwen35 recurrent slot seqlen_offset {seqlen_offset} != shared attention seq_len {}",
                self.seq_len
            );
        }
        self.clear_batch_slot_indices();
        self.recurrent_pool
            .set_seqlen_offset(slot_idx, seqlen_offset);
    }

    pub fn recurrent_state_generation(&self, slot_idx: usize, layer: usize) -> u64 {
        self.recurrent_pool
            .recurrent_state_generation(slot_idx, layer)
    }

    pub(crate) fn conv_state_generation(&self, slot_idx: usize, layer: usize) -> u64 {
        self.recurrent_pool.conv_state_generation(slot_idx, layer)
    }

    pub fn recurrent_state_cpu_stale(&self, slot_idx: usize, layer: usize) -> bool {
        self.recurrent_pool
            .cpu_recurrent_state_stale(slot_idx, layer)
    }

    #[allow(dead_code)]
    pub(crate) fn recurrent_state_owner(&self, slot_idx: usize, layer: usize) -> Qwen35StateOwner {
        self.recurrent_pool.recurrent_state_owner(slot_idx, layer)
    }

    pub(crate) fn note_backend_recurrent_state_update(
        &mut self,
        slot_idx: usize,
        layer: usize,
    ) -> u64 {
        self.recurrent_pool
            .note_backend_recurrent_state(slot_idx, layer)
    }

    pub(crate) fn sync_recurrent_state_from_backend(
        &mut self,
        slot_idx: usize,
        layer: usize,
        recurrent_state: &[f32],
        generation: u64,
    ) {
        self.recurrent_pool.overwrite_recurrent_state_from_backend(
            slot_idx,
            layer,
            recurrent_state,
            generation,
        );
    }

    pub fn conv_state_cpu_stale(&self, slot_idx: usize, layer: usize) -> bool {
        self.recurrent_pool.cpu_conv_state_stale(slot_idx, layer)
    }

    #[allow(dead_code)]
    pub(crate) fn conv_state_owner(&self, slot_idx: usize, layer: usize) -> Qwen35StateOwner {
        self.recurrent_pool.conv_state_owner(slot_idx, layer)
    }

    pub(crate) fn layer_state_owner(&self, slot_idx: usize, layer: usize) -> Qwen35LayerStateOwner {
        self.recurrent_pool.layer_state_owner(slot_idx, layer)
    }

    pub(crate) fn conv_state_pristine_zero(&self, slot_idx: usize, layer: usize) -> bool {
        self.recurrent_pool
            .conv_state_pristine_zero(slot_idx, layer)
    }

    pub(crate) fn recurrent_state_pristine_zero(&self, slot_idx: usize, layer: usize) -> bool {
        self.recurrent_pool
            .recurrent_state_pristine_zero(slot_idx, layer)
    }

    pub(crate) fn mark_layer_state_backend_owned(&mut self, slot_idx: usize, layer: usize) {
        self.recurrent_pool
            .mark_layer_state_backend_owned(slot_idx, layer);
    }

    pub(crate) fn note_backend_conv_state_update(&mut self, slot_idx: usize, layer: usize) -> u64 {
        self.recurrent_pool.note_backend_conv_state(slot_idx, layer)
    }

    pub(crate) fn sync_conv_state_from_backend(
        &mut self,
        slot_idx: usize,
        layer: usize,
        conv_state: &[f32],
        generation: u64,
    ) {
        self.recurrent_pool
            .overwrite_conv_state_from_backend(slot_idx, layer, conv_state, generation);
    }

    pub fn conv_state_for_slot_mut(&mut self, slot_idx: usize, layer: usize) -> &mut [f32] {
        self.recurrent_pool.conv_state_mut(slot_idx, layer)
    }

    pub fn conv_state_for_slot(&self, slot_idx: usize, layer: usize) -> &[f32] {
        self.recurrent_pool.conv_state(slot_idx, layer)
    }

    pub fn recurrent_state_for_slot_mut(&mut self, slot_idx: usize, layer: usize) -> &mut [f32] {
        self.recurrent_pool.recurrent_state_mut(slot_idx, layer)
    }

    pub fn recurrent_state_for_slot(&self, slot_idx: usize, layer: usize) -> &[f32] {
        self.recurrent_pool.recurrent_state(slot_idx, layer)
    }

    pub fn recurrent_buffers_for_slot_mut(
        &mut self,
        slot_idx: usize,
        layer: usize,
    ) -> (&mut [f32], &mut [f32]) {
        self.recurrent_pool.recurrent_buffers_mut(slot_idx, layer)
    }

    pub(crate) fn note_cpu_visible_layer_state_update(
        &mut self,
        slot_idx: usize,
        layer: usize,
    ) -> (u64, u64) {
        self.recurrent_pool.touch_buffers(slot_idx, layer)
    }

    pub(crate) fn prepare_recurrent_state_batch<'a>(
        &'a mut self,
        slot_indices: &'a [usize],
        layer: usize,
    ) -> Qwen35PreparedRecurrentStateBatch<'a> {
        self.assert_valid_recurrent_slot_batch(slot_indices, layer);
        let conv_state_stride = self.conv_cache_len() * self.conv_dim();
        let recurrent_state_stride = self.recurrent_state_len();
        if slot_indices.len() == 1 {
            let kind = if self.layer_state_owner(slot_indices[0], layer)
                == Qwen35LayerStateOwner::CpuMaterialized
            {
                Qwen35PreparedRecurrentStateBatchKind::CpuDirect
            } else {
                Qwen35PreparedRecurrentStateBatchKind::CpuDirectMaterializedFromBackend
            };
            let (conv_state, recurrent_state) =
                self.recurrent_buffers_for_slot_mut(slot_indices[0], layer);
            Qwen35PreparedRecurrentStateBatch::Direct {
                kind,
                layer_idx: layer,
                slot_indices,
                conv_state,
                recurrent_state,
                conv_state_stride,
                recurrent_state_stride,
            }
        } else {
            let mut conv_state_batch = vec![0.0f32; slot_indices.len() * conv_state_stride];
            let mut recurrent_state_batch =
                vec![0.0f32; slot_indices.len() * recurrent_state_stride];
            let kind = if slot_indices.iter().any(|&slot_idx| {
                self.layer_state_owner(slot_idx, layer) != Qwen35LayerStateOwner::CpuMaterialized
            }) {
                Qwen35PreparedRecurrentStateBatchKind::CpuGatheredMaterializedFromBackend
            } else {
                Qwen35PreparedRecurrentStateBatchKind::CpuGathered
            };
            self.gather_recurrent_state_batch(
                slot_indices,
                layer,
                &mut conv_state_batch,
                &mut recurrent_state_batch,
            );
            Qwen35PreparedRecurrentStateBatch::Gathered {
                kind,
                qwen_kv: self,
                layer_idx: layer,
                slot_indices,
                conv_state_batch,
                recurrent_state_batch,
                conv_state_stride,
                recurrent_state_stride,
            }
        }
    }

    pub fn gather_recurrent_state_batch(
        &self,
        slot_indices: &[usize],
        layer: usize,
        conv_state_batch: &mut [f32],
        recurrent_state_batch: &mut [f32],
    ) {
        self.assert_valid_recurrent_slot_batch(slot_indices, layer);
        let conv_state_len = self.conv_state_for_slot(0, layer).len();
        let recurrent_state_len = self.recurrent_state_for_slot(0, layer).len();
        assert_eq!(
            conv_state_batch.len(),
            slot_indices.len() * conv_state_len,
            "qwen35 gathered conv state batch has wrong length"
        );
        assert_eq!(
            recurrent_state_batch.len(),
            slot_indices.len() * recurrent_state_len,
            "qwen35 gathered recurrent state batch has wrong length"
        );

        for (batch_idx, &slot_idx) in slot_indices.iter().enumerate() {
            let conv_start = batch_idx * conv_state_len;
            let recurrent_start = batch_idx * recurrent_state_len;
            conv_state_batch[conv_start..conv_start + conv_state_len]
                .copy_from_slice(self.conv_state_for_slot(slot_idx, layer));
            recurrent_state_batch[recurrent_start..recurrent_start + recurrent_state_len]
                .copy_from_slice(self.recurrent_state_for_slot(slot_idx, layer));
        }
    }

    pub fn scatter_recurrent_state_batch(
        &mut self,
        slot_indices: &[usize],
        layer: usize,
        conv_state_batch: &[f32],
        recurrent_state_batch: &[f32],
    ) {
        self.assert_valid_recurrent_slot_batch(slot_indices, layer);
        let conv_state_len = self.conv_state_for_slot(0, layer).len();
        let recurrent_state_len = self.recurrent_state_for_slot(0, layer).len();
        assert_eq!(
            conv_state_batch.len(),
            slot_indices.len() * conv_state_len,
            "qwen35 scattered conv state batch has wrong length"
        );
        assert_eq!(
            recurrent_state_batch.len(),
            slot_indices.len() * recurrent_state_len,
            "qwen35 scattered recurrent state batch has wrong length"
        );

        for (batch_idx, &slot_idx) in slot_indices.iter().enumerate() {
            let conv_start = batch_idx * conv_state_len;
            let recurrent_start = batch_idx * recurrent_state_len;
            self.conv_state_for_slot_mut(slot_idx, layer)
                .copy_from_slice(&conv_state_batch[conv_start..conv_start + conv_state_len]);
            self.recurrent_state_for_slot_mut(slot_idx, layer)
                .copy_from_slice(
                    &recurrent_state_batch[recurrent_start..recurrent_start + recurrent_state_len],
                );
        }
    }

    pub fn conv_state_mut(&mut self, layer: usize) -> &mut [f32] {
        self.conv_state_for_slot_mut(self.active_slot, layer)
    }

    pub fn recurrent_state_mut(&mut self, layer: usize) -> &mut [f32] {
        self.recurrent_state_for_slot_mut(self.active_slot, layer)
    }

    pub fn recurrent_buffers_mut(&mut self, layer: usize) -> (&mut [f32], &mut [f32]) {
        self.recurrent_buffers_for_slot_mut(self.active_slot, layer)
    }

    pub fn clear(&mut self) {
        self.seq_len = 0;
        self.attention.clear();
        if let Some(gpu_attention) = self.attention_gpu.as_mut() {
            gpu_attention.clear();
        }
        self.attention_cpu_dirty = false;
        self.attention_cpu_valid_prefix_len = 0;
        self.active_slot = 0;
        self.clear_batch_slot_indices();
        self.recurrent_pool.clear_all();
        if let Some(gpu) = self.gpu_recurrent.as_mut() {
            for slot_idx in 0..gpu.slots.len() {
                gpu.clear_slot(slot_idx);
            }
        }
    }

    pub fn truncate_to(&mut self, pos: usize) {
        if pos >= self.seq_len {
            return;
        }

        if pos == 0 {
            self.clear();
            return;
        }

        // Qwen3.5 recurrent layers need historical state snapshots to support
        // arbitrary rollback. AX does not store those yet, so partial truncation
        // is unsupported and would silently destroy all KV state.
        panic!(
            "Qwen35Kv::truncate_to({pos}) is unsupported (seq_len={}): \
             recurrent history snapshots are not stored, so partial rollback \
             cannot preserve correctness. Only truncate_to(0) (full clear) \
             is safe.",
            self.seq_len
        );
    }

    pub fn finalize_token(&mut self) {
        let active_slot = [self.active_slot];
        self.finalize_batch_for_slots(&active_slot, 1);
        self.clear_batch_slot_indices();
    }

    pub fn finalize_batch(&mut self, n_tokens: usize) {
        let slot_indices = self.recurrent_batch_slot_indices().into_owned();
        self.finalize_batch_for_slots(&slot_indices, n_tokens);
        self.clear_batch_slot_indices();
    }

    pub fn finalize_batch_for_slots(&mut self, slot_indices: &[usize], n_tokens: usize) {
        if n_tokens == 0 {
            return;
        }

        self.assert_valid_recurrent_slots_on_shared_timeline(slot_indices);
        assert!(
            slot_indices.contains(&self.active_slot),
            "qwen35 shared-attention finalize requires the active recurrent slot to be included"
        );
        self.attention.finalize_batch(n_tokens);
        if let Some(gpu_attention) = self.attention_gpu.as_mut() {
            gpu_attention.finalize_batch(n_tokens);
        }
        self.seq_len += n_tokens;
        if !self.attention_cpu_dirty {
            self.attention_cpu_valid_prefix_len =
                self.attention_cpu_valid_prefix_len.saturating_add(n_tokens);
        }
        for &slot_idx in slot_indices {
            self.recurrent_pool
                .increment_seqlen_offset(slot_idx, n_tokens);
        }
    }

    pub fn conv_dim(&self) -> usize {
        self.conv_dim
    }

    pub fn conv_cache_len(&self) -> usize {
        self.conv_cache_len
    }

    pub fn recurrent_state_len(&self) -> usize {
        self.recurrent_state_len
    }

    pub(crate) fn ensure_gpu_attention_capacity_for(&mut self, needed: usize) -> bool {
        let Some(current_capacity) = self.attention_gpu.as_ref().map(|gpu| gpu.capacity()) else {
            return false;
        };
        if needed <= current_capacity {
            return true;
        }
        let Some(device) = self.attention_gpu_device.as_ref() else {
            tracing::warn!(
                needed,
                "Qwen3.5 GPU attention KV mirror has no Metal device for growth; disabling mirror"
            );
            self.attention_gpu = None;
            return false;
        };
        if let Err(err) = self
            .attention_gpu
            .as_mut()
            .expect("qwen35 gpu attention mirror must still exist before growth")
            .ensure_capacity(device, needed)
        {
            tracing::warn!(
                needed,
                "Qwen3.5 GPU attention KV mirror growth failed ({err}); disabling mirror"
            );
            self.attention_gpu = None;
            self.attention_gpu_device = None;
            return false;
        }
        true
    }

    fn sync_gpu_attention_from_cpu(&mut self) {
        if self.attention_gpu.is_none() {
            self.attention_cpu_dirty = false;
            self.attention_cpu_valid_prefix_len = self.seq_len;
            return;
        }
        if !self.ensure_gpu_attention_capacity_for(self.seq_len) {
            return;
        }
        let gpu_attention = self.attention_gpu.as_mut().unwrap();
        gpu_attention.clear();
        if self.seq_len == 0 {
            return;
        }

        for layer in 0..self.recurrent_layers.len() {
            gpu_attention.append_layer_batch(
                layer,
                self.attention.k_slice(layer, self.seq_len),
                self.attention.v_slice(layer, self.seq_len),
                self.seq_len,
            );
        }
        gpu_attention.finalize_batch(self.seq_len);
        self.attention_cpu_dirty = false;
        self.attention_cpu_valid_prefix_len = self.seq_len;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen35_kv_layer_pattern() {
        let kv = Qwen35Kv::new(8, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        assert!(kv.is_recurrent_layer(0));
        assert!(kv.is_recurrent_layer(1));
        assert!(kv.is_recurrent_layer(2));
        assert!(!kv.is_recurrent_layer(3));
        assert!(kv.is_recurrent_layer(4));
        assert!(kv.is_recurrent_layer(5));
        assert!(kv.is_recurrent_layer(6));
        assert!(!kv.is_recurrent_layer(7));
    }

    #[test]
    fn test_qwen35_kv_clear_resets_state() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        kv.conv_state_mut(0).fill(1.0);
        kv.recurrent_state_mut(0).fill(2.0);
        kv.finalize_token();
        kv.clear();
        assert_eq!(kv.seq_len(), 0);
        assert!(kv.conv_state_mut(0).iter().all(|&v| v == 0.0));
        assert!(kv.recurrent_state_mut(0).iter().all(|&v| v == 0.0));
        assert_eq!(kv.recurrent_seqlen_offset(kv.active_slot()), 0);
    }

    #[test]
    fn test_qwen35_kv_finalize_batch_for_slots_keeps_shared_timeline_aligned() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot1 = kv.allocate_recurrent_slot();

        kv.finalize_batch_for_slots(&[0, slot1], 2);

        assert_eq!(kv.seq_len(), 2);
        assert_eq!(kv.recurrent_seqlen_offset(0), 2);
        assert_eq!(kv.recurrent_seqlen_offset(slot1), 2);
    }

    #[test]
    fn test_qwen35_kv_finalize_batch_uses_explicit_batch_slot_indices() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot1 = kv.allocate_recurrent_slot();

        kv.set_batch_slot_indices(&[0, slot1]);
        kv.finalize_batch(2);

        assert_eq!(kv.seq_len(), 2);
        assert_eq!(kv.recurrent_seqlen_offset(0), 2);
        assert_eq!(kv.recurrent_seqlen_offset(slot1), 2);
        assert_eq!(kv.recurrent_batch_slot_indices().as_ref(), &[0]);
    }

    #[test]
    fn test_qwen35_kv_recurrent_batch_slot_indices_fallback_to_active_slot() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot1 = kv.allocate_recurrent_slot();

        assert_eq!(kv.recurrent_batch_slot_indices().as_ref(), &[0]);
        kv.set_active_slot(slot1);
        assert_eq!(kv.recurrent_batch_slot_indices().as_ref(), &[slot1]);

        kv.set_batch_slot_indices(&[slot1]);
        assert_eq!(kv.recurrent_batch_slot_indices().as_ref(), &[slot1]);

        kv.clear_batch_slot_indices();
        assert_eq!(kv.recurrent_batch_slot_indices().as_ref(), &[slot1]);
    }

    #[test]
    #[should_panic(
        expected = "qwen35 recurrent slot batch requires the active slot to be included"
    )]
    fn test_qwen35_kv_rejects_setting_batch_slot_indices_without_active_slot() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot1 = kv.allocate_recurrent_slot();

        kv.set_batch_slot_indices(&[slot1]);
    }

    #[test]
    #[should_panic(
        expected = "qwen35 shared-attention finalize requires the active recurrent slot to be included"
    )]
    fn test_qwen35_kv_rejects_finalizing_shared_timeline_without_active_slot() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot1 = kv.allocate_recurrent_slot();
        kv.finalize_batch_for_slots(&[slot1], 1);
    }

    #[test]
    #[should_panic(
        expected = "qwen35 recurrent slot batch slot 1 has seqlen_offset 1 != shared attention seq_len 0"
    )]
    fn test_qwen35_kv_rejects_finalizing_misaligned_multi_slot_shared_timeline() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot1 = kv.allocate_recurrent_slot();
        kv.set_recurrent_seqlen_offset(slot1, 1);
        kv.finalize_batch_for_slots(&[0, slot1], 1);
    }

    #[test]
    #[should_panic(expected = "multiple of group_count")]
    fn test_qwen35_kv_rejects_incompatible_head_expansion() {
        let _ = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 768, 128, 6, 4);
    }

    #[test]
    fn test_qwen35_kv_snapshot_restore_round_trips_slot_state() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot1 = kv.allocate_recurrent_slot();
        kv.conv_state_for_slot_mut(slot1, 0).fill(1.5);
        kv.recurrent_state_for_slot_mut(slot1, 0).fill(2.5);
        kv.set_recurrent_seqlen_offset(slot1, 11);

        let snapshot = kv.recurrent_slot_snapshot(slot1);

        kv.conv_state_for_slot_mut(slot1, 0).fill(0.0);
        kv.recurrent_state_for_slot_mut(slot1, 0).fill(0.0);
        kv.set_recurrent_seqlen_offset(slot1, 0);
        kv.restore_recurrent_slot(slot1, &snapshot);

        assert!(kv.conv_state_for_slot(slot1, 0).iter().all(|&v| v == 1.5));
        assert!(
            kv.recurrent_state_for_slot(slot1, 0)
                .iter()
                .all(|&v| v == 2.5)
        );
        assert_eq!(kv.recurrent_seqlen_offset(slot1), 11);
    }

    #[test]
    fn test_qwen35_kv_clone_recurrent_slot_copies_state_and_offset() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot1 = kv.allocate_recurrent_slot();
        let slot2 = kv.allocate_recurrent_slot();

        kv.conv_state_for_slot_mut(slot1, 0).fill(1.25);
        kv.recurrent_state_for_slot_mut(slot1, 0).fill(2.75);
        kv.set_recurrent_seqlen_offset(slot1, 9);

        kv.clone_recurrent_slot(slot1, slot2);

        assert!(kv.conv_state_for_slot(slot2, 0).iter().all(|&v| v == 1.25));
        assert!(
            kv.recurrent_state_for_slot(slot2, 0)
                .iter()
                .all(|&v| v == 2.75)
        );
        assert_eq!(kv.recurrent_seqlen_offset(slot2), 9);
    }

    #[test]
    fn test_qwen35_kv_restore_recurrent_slot_allocates_missing_slot() {
        let mut source = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot1 = source.allocate_recurrent_slot();
        source.conv_state_for_slot_mut(slot1, 0).fill(1.25);
        source.recurrent_state_for_slot_mut(slot1, 0).fill(2.75);
        source.set_recurrent_seqlen_offset(slot1, 13);
        let snapshot = source.recurrent_slot_snapshot(slot1);

        let mut restored = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        restored.restore_recurrent_slot(slot1, &snapshot);

        assert!(
            restored
                .conv_state_for_slot(slot1, 0)
                .iter()
                .all(|&v| v == 1.25)
        );
        assert!(
            restored
                .recurrent_state_for_slot(slot1, 0)
                .iter()
                .all(|&v| v == 2.75)
        );
        assert_eq!(restored.recurrent_seqlen_offset(slot1), 13);
    }

    #[test]
    fn test_qwen35_kv_allocates_and_reuses_recurrent_slots() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot = kv.allocate_recurrent_slot();
        assert_eq!(slot, 1);

        let mut snapshot = kv.recurrent_slot_snapshot(slot);
        snapshot.conv_states[0].fill(3.0);
        snapshot.conv_states[1].fill(3.0);
        snapshot.conv_states[2].fill(3.0);
        snapshot.recurrent_states[0].fill(4.0);
        snapshot.recurrent_states[1].fill(4.0);
        snapshot.recurrent_states[2].fill(4.0);
        snapshot.seqlen_offset = 9;
        kv.restore_recurrent_slot(slot, &snapshot);

        kv.free_recurrent_slot(slot);

        let reused = kv.allocate_recurrent_slot();
        assert_eq!(reused, slot);
        let snapshot = kv.recurrent_slot_snapshot(reused);
        assert!(
            snapshot
                .conv_states
                .iter()
                .all(|state| state.iter().all(|&v| v == 0.0))
        );
        assert!(
            snapshot
                .recurrent_states
                .iter()
                .all(|state| state.iter().all(|&v| v == 0.0))
        );
        assert_eq!(snapshot.seqlen_offset, 0);
    }

    #[test]
    #[should_panic(expected = "cannot activate free qwen35 recurrent slot")]
    fn test_qwen35_kv_rejects_activating_freed_slot() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot = kv.allocate_recurrent_slot();
        kv.free_recurrent_slot(slot);
        kv.set_active_slot(slot);
    }

    #[test]
    #[should_panic(expected = "cannot free inactive qwen35 recurrent slot")]
    fn test_qwen35_kv_rejects_freeing_same_slot_twice() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot = kv.allocate_recurrent_slot();
        kv.free_recurrent_slot(slot);
        kv.free_recurrent_slot(slot);
    }

    #[test]
    #[should_panic(expected = "cannot free reserved qwen35 recurrent slot 0")]
    fn test_qwen35_kv_rejects_freeing_reserved_slot_zero() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot1 = kv.allocate_recurrent_slot();
        kv.set_active_slot(slot1);
        kv.free_recurrent_slot(0);
    }

    #[test]
    #[should_panic(expected = "qwen35 recurrent slot is not allocated")]
    fn test_qwen35_kv_rejects_reading_freed_slot_state() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot = kv.allocate_recurrent_slot();
        kv.free_recurrent_slot(slot);
        let _ = kv.conv_state_for_slot(slot, 0);
    }

    #[test]
    #[should_panic(expected = "qwen35 recurrent slot is not allocated")]
    fn test_qwen35_kv_rejects_snapshotting_freed_slot() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot = kv.allocate_recurrent_slot();
        kv.free_recurrent_slot(slot);
        let _ = kv.recurrent_slot_snapshot(slot);
    }

    #[test]
    fn test_qwen35_kv_slot_specific_access_stays_isolated() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot1 = kv.allocate_recurrent_slot();

        kv.conv_state_for_slot_mut(0, 0).fill(1.0);
        kv.recurrent_state_for_slot_mut(0, 0).fill(2.0);
        kv.conv_state_for_slot_mut(slot1, 0).fill(3.0);
        kv.recurrent_state_for_slot_mut(slot1, 0).fill(4.0);

        kv.set_active_slot(slot1);
        assert!(kv.conv_state_mut(0).iter().all(|&v| v == 3.0));
        assert!(kv.recurrent_state_mut(0).iter().all(|&v| v == 4.0));
        assert_eq!(kv.recurrent_seqlen_offset(kv.active_slot()), 0);

        kv.set_active_slot(0);
        assert!(kv.conv_state_mut(0).iter().all(|&v| v == 1.0));
        assert!(kv.recurrent_state_mut(0).iter().all(|&v| v == 2.0));
        assert_eq!(kv.recurrent_seqlen_offset(kv.active_slot()), 0);
    }

    #[test]
    #[should_panic(
        expected = "cannot activate qwen35 recurrent slot with seqlen_offset 1 != shared attention seq_len 0"
    )]
    fn test_qwen35_kv_rejects_activating_slot_with_mismatched_shared_seq_len() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot1 = kv.allocate_recurrent_slot();
        kv.set_recurrent_seqlen_offset(slot1, 1);
        kv.set_active_slot(slot1);
    }

    #[test]
    fn test_qwen35_kv_gather_and_scatter_recurrent_state_batch() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot1 = kv.allocate_recurrent_slot();
        let slot_indices = [slot1, 0];

        kv.conv_state_for_slot_mut(0, 0).fill(1.0);
        kv.recurrent_state_for_slot_mut(0, 0).fill(2.0);
        kv.conv_state_for_slot_mut(slot1, 0).fill(3.0);
        kv.recurrent_state_for_slot_mut(slot1, 0).fill(4.0);

        let conv_state_len = kv.conv_state_for_slot(0, 0).len();
        let recurrent_state_len = kv.recurrent_state_for_slot(0, 0).len();
        let mut conv_batch = vec![0.0; slot_indices.len() * conv_state_len];
        let mut recurrent_batch = vec![0.0; slot_indices.len() * recurrent_state_len];
        kv.gather_recurrent_state_batch(&slot_indices, 0, &mut conv_batch, &mut recurrent_batch);

        assert!(conv_batch[..conv_state_len].iter().all(|&v| v == 3.0));
        assert!(
            recurrent_batch[..recurrent_state_len]
                .iter()
                .all(|&v| v == 4.0)
        );
        assert!(conv_batch[conv_state_len..].iter().all(|&v| v == 1.0));
        assert!(
            recurrent_batch[recurrent_state_len..]
                .iter()
                .all(|&v| v == 2.0)
        );

        conv_batch[..conv_state_len].fill(5.0);
        recurrent_batch[..recurrent_state_len].fill(6.0);
        conv_batch[conv_state_len..].fill(7.0);
        recurrent_batch[recurrent_state_len..].fill(8.0);
        kv.scatter_recurrent_state_batch(&slot_indices, 0, &conv_batch, &recurrent_batch);

        assert!(kv.conv_state_for_slot(0, 0).iter().all(|&v| v == 7.0));
        assert!(kv.recurrent_state_for_slot(0, 0).iter().all(|&v| v == 8.0));
        assert!(kv.conv_state_for_slot(slot1, 0).iter().all(|&v| v == 5.0));
        assert!(
            kv.recurrent_state_for_slot(slot1, 0)
                .iter()
                .all(|&v| v == 6.0)
        );
    }

    #[test]
    fn test_qwen35_kv_prepare_recurrent_state_batch_single_slot_writes_through() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot_indices = [0usize];

        {
            let mut prepared = kv.prepare_recurrent_state_batch(&slot_indices, 0);
            {
                let mut state_batch = prepared.state_batch();
                state_batch.conv_state_for_slot_mut(0).fill(1.25);
                state_batch.recurrent_state_for_slot_mut(0).fill(2.5);
            }
            prepared.finish();
        }

        assert!(kv.conv_state_for_slot(0, 0).iter().all(|&v| v == 1.25));
        assert!(kv.recurrent_state_for_slot(0, 0).iter().all(|&v| v == 2.5));
    }

    #[test]
    fn test_qwen35_kv_prepare_recurrent_state_batch_multi_slot_scatter_on_finish() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot1 = kv.allocate_recurrent_slot();
        let slot_indices = [0usize, slot1];

        {
            let mut prepared = kv.prepare_recurrent_state_batch(&slot_indices, 0);
            {
                let mut state_batch = prepared.state_batch();
                state_batch.conv_state_for_slot_mut(0).fill(3.0);
                state_batch.recurrent_state_for_slot_mut(0).fill(4.0);
                state_batch.conv_state_for_slot_mut(1).fill(5.0);
                state_batch.recurrent_state_for_slot_mut(1).fill(6.0);
            }
            prepared.finish();
        }

        assert!(kv.conv_state_for_slot(0, 0).iter().all(|&v| v == 3.0));
        assert!(kv.recurrent_state_for_slot(0, 0).iter().all(|&v| v == 4.0));
        assert!(kv.conv_state_for_slot(slot1, 0).iter().all(|&v| v == 5.0));
        assert!(
            kv.recurrent_state_for_slot(slot1, 0)
                .iter()
                .all(|&v| v == 6.0)
        );
    }

    #[test]
    fn test_qwen35_kv_state_generation_changes_on_cpu_mutation_and_reset() {
        let mut kv = Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let layer = 0usize;
        let base_generation = kv.recurrent_state_generation(0, layer);
        let base_conv_generation = kv.conv_state_generation(0, layer);

        kv.conv_state_for_slot_mut(0, layer)[0] = 1.0;
        let after_conv_mutation = kv.recurrent_state_generation(0, layer);
        let after_conv_cpu_generation = kv.conv_state_generation(0, layer);
        assert_eq!(after_conv_mutation, base_generation);
        assert!(after_conv_cpu_generation > base_conv_generation);

        kv.recurrent_state_for_slot_mut(0, layer)[0] = 2.0;
        let after_recurrent_mutation = kv.recurrent_state_generation(0, layer);
        assert!(after_recurrent_mutation > after_conv_mutation);

        let snapshot = kv.recurrent_slot_snapshot(0);
        kv.clear();
        let after_clear = kv.recurrent_state_generation(0, layer);
        assert!(after_clear > after_recurrent_mutation);

        kv.restore_recurrent_slot(0, &snapshot);
        let after_restore = kv.recurrent_state_generation(0, layer);
        assert!(after_restore > after_clear);
    }

    #[test]
    fn test_qwen35_kv_conv_only_cpu_mutation_keeps_backend_recurrent_state_stale() {
        let mut kv = Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let layer = 0usize;

        assert!(!kv.recurrent_state_cpu_stale(0, layer));
        let backend_generation = kv.note_backend_recurrent_state_update(0, layer);
        assert!(kv.recurrent_state_cpu_stale(0, layer));

        kv.conv_state_for_slot_mut(0, layer)[0] = 1.0;
        assert!(
            kv.recurrent_state_cpu_stale(0, layer),
            "conv-only CPU mutation must not mark backend-owned recurrent state as materialized"
        );
        assert!(
            kv.recurrent_state_generation(0, layer) == backend_generation,
            "conv-only CPU mutation must not advance recurrent generation"
        );
        assert!(
            kv.conv_state_generation(0, layer) > 1,
            "conv-only CPU mutation should still advance conv-state generation"
        );

        kv.recurrent_state_for_slot_mut(0, layer)[0] = 2.0;
        assert!(
            !kv.recurrent_state_cpu_stale(0, layer),
            "CPU recurrent mutation should make recurrent state materialized again"
        );
    }

    #[test]
    fn test_qwen35_kv_layer_state_owner_tracks_cpu_backend_and_split() {
        let mut kv = Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let layer = 0usize;

        assert_eq!(
            kv.layer_state_owner(0, layer),
            Qwen35LayerStateOwner::CpuMaterialized
        );
        assert_eq!(
            kv.conv_state_owner(0, layer),
            Qwen35StateOwner::CpuMaterialized
        );
        assert_eq!(
            kv.recurrent_state_owner(0, layer),
            Qwen35StateOwner::CpuMaterialized
        );

        kv.note_backend_recurrent_state_update(0, layer);
        assert_eq!(kv.layer_state_owner(0, layer), Qwen35LayerStateOwner::Split);
        assert_eq!(
            kv.conv_state_owner(0, layer),
            Qwen35StateOwner::CpuMaterialized
        );
        assert_eq!(
            kv.recurrent_state_owner(0, layer),
            Qwen35StateOwner::BackendOwned
        );

        kv.note_backend_conv_state_update(0, layer);
        assert_eq!(
            kv.layer_state_owner(0, layer),
            Qwen35LayerStateOwner::BackendOwned
        );

        kv.conv_state_for_slot_mut(0, layer)[0] = 1.0;
        assert_eq!(kv.layer_state_owner(0, layer), Qwen35LayerStateOwner::Split);

        kv.recurrent_buffers_for_slot_mut(0, layer).1[0] = 2.0;
        assert_eq!(
            kv.layer_state_owner(0, layer),
            Qwen35LayerStateOwner::CpuMaterialized
        );
    }

    #[test]
    fn test_qwen35_kv_pristine_zero_flags_track_clear_cpu_write_and_backend_write() {
        let mut kv = Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let layer = 0usize;

        assert!(kv.conv_state_pristine_zero(0, layer));
        assert!(kv.recurrent_state_pristine_zero(0, layer));

        kv.conv_state_for_slot_mut(0, layer)[0] = 1.0;
        assert!(!kv.conv_state_pristine_zero(0, layer));
        assert!(kv.recurrent_state_pristine_zero(0, layer));

        kv.clear();
        assert!(kv.conv_state_pristine_zero(0, layer));
        assert!(kv.recurrent_state_pristine_zero(0, layer));

        kv.note_backend_conv_state_update(0, layer);
        assert!(!kv.conv_state_pristine_zero(0, layer));
        assert!(kv.recurrent_state_pristine_zero(0, layer));

        kv.clear();
        kv.note_backend_recurrent_state_update(0, layer);
        assert!(kv.conv_state_pristine_zero(0, layer));
        assert!(!kv.recurrent_state_pristine_zero(0, layer));
    }

    #[test]
    fn test_qwen35_kv_mark_layer_state_backend_owned_makes_cpu_copy_stale_without_bump() {
        let mut kv = Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let layer = 0usize;
        let conv_generation = kv.conv_state_generation(0, layer);
        let recurrent_generation = kv.recurrent_state_generation(0, layer);

        kv.mark_layer_state_backend_owned(0, layer);

        assert!(kv.conv_state_cpu_stale(0, layer));
        assert!(kv.recurrent_state_cpu_stale(0, layer));
        assert_eq!(kv.conv_state_generation(0, layer), conv_generation);
        assert_eq!(
            kv.recurrent_state_generation(0, layer),
            recurrent_generation
        );
        assert_eq!(
            kv.layer_state_owner(0, layer),
            Qwen35LayerStateOwner::BackendOwned
        );
    }

    #[test]
    #[should_panic(
        expected = "cannot snapshot qwen35 recurrent slot while backend-owned recurrent state is not materialized on CPU"
    )]
    fn test_qwen35_kv_rejects_snapshotting_slot_with_backend_owned_recurrent_state() {
        let mut kv = Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        kv.note_backend_recurrent_state_update(0, 0);
        let _ = kv.recurrent_slot_snapshot(0);
    }

    #[test]
    #[should_panic(
        expected = "cannot snapshot qwen35 recurrent slot while backend-owned recurrent state is not materialized on CPU"
    )]
    fn test_qwen35_kv_rejects_full_snapshot_with_backend_owned_recurrent_state() {
        let mut kv = Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        kv.note_backend_recurrent_state_update(0, 0);
        let _ = kv.snapshot_active_slot();
    }

    #[test]
    fn test_qwen35_kv_full_snapshot_restores_attention_and_recurrent_state() {
        let mut kv = Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let slot1 = kv.allocate_recurrent_slot();
        kv.set_active_slot(slot1);

        kv.conv_state_for_slot_mut(slot1, 0).fill(1.5);
        kv.recurrent_state_for_slot_mut(slot1, 0).fill(2.5);
        kv.attention_append(3, &[10.0, 11.0], &[12.0, 13.0]);
        kv.finalize_token();

        let snapshot = kv.snapshot_active_slot();

        kv.clear();
        kv.restore_snapshot(&snapshot);

        assert_eq!(kv.active_slot(), slot1);
        assert_eq!(kv.seq_len(), 1);
        assert_eq!(kv.attention_k_slice_including_current(3, 1), &[10.0, 11.0]);
        assert_eq!(kv.attention_v_slice_including_current(3, 1), &[12.0, 13.0]);
        assert!(kv.conv_state_for_slot(slot1, 0).iter().all(|&v| v == 1.5));
        assert!(
            kv.recurrent_state_for_slot(slot1, 0)
                .iter()
                .all(|&v| v == 2.5)
        );
        assert_eq!(kv.recurrent_seqlen_offset(slot1), 1);
    }

    #[test]
    fn test_qwen35_kv_attention_snapshot_restores_shared_timeline_only() {
        let mut kv = Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let slot1 = kv.allocate_recurrent_slot();
        kv.attention_append(3, &[10.0, 11.0], &[12.0, 13.0]);
        kv.finalize_token();

        let snapshot = kv.attention_snapshot();

        kv.attention_append(3, &[20.0, 21.0], &[22.0, 23.0]);
        kv.finalize_token();
        kv.set_recurrent_seqlen_offset(slot1, 1);
        kv.conv_state_for_slot_mut(slot1, 0).fill(7.0);
        kv.recurrent_state_for_slot_mut(slot1, 0).fill(8.0);

        kv.restore_attention_snapshot(&snapshot);

        assert_eq!(kv.seq_len(), 1);
        assert_eq!(kv.attention_k_slice_including_current(3, 1), &[10.0, 11.0]);
        assert_eq!(kv.attention_v_slice_including_current(3, 1), &[12.0, 13.0]);
        assert_eq!(kv.recurrent_seqlen_offset(slot1), 1);
        assert!(kv.conv_state_for_slot(slot1, 0).iter().all(|&v| v == 7.0));
        assert!(
            kv.recurrent_state_for_slot(slot1, 0)
                .iter()
                .all(|&v| v == 8.0)
        );
    }

    #[test]
    fn test_qwen35_kv_sync_attention_cpu_from_gpu_if_needed_restores_cpu_mirror() {
        let device = ax_engine_metal::MetalDevice::new().expect("metal device");
        let mut kv = Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        kv.enable_gpu_attention(&device, GpuKvDtype::F32)
            .expect("enable gpu attention");

        let k = [10.0f32, 11.0];
        let v = [12.0f32, 13.0];
        let gpu_attention = kv
            .attention_gpu
            .as_mut()
            .expect("gpu attention should exist");
        gpu_attention.append_layer(3, &k, &v);
        gpu_attention.finalize_token();

        kv.mark_attention_cpu_dirty();
        kv.finalize_token();

        assert_eq!(kv.attention_k_slice_including_current(3, 1), &[0.0, 0.0]);
        assert_eq!(kv.attention_v_slice_including_current(3, 1), &[0.0, 0.0]);

        kv.sync_attention_cpu_from_gpu_if_needed();

        assert_eq!(kv.attention_k_slice_including_current(3, 1), &k);
        assert_eq!(kv.attention_v_slice_including_current(3, 1), &v);
    }

    #[test]
    fn test_qwen35_kv_incremental_attention_cpu_sync_preserves_valid_prefix() {
        let device = ax_engine_metal::MetalDevice::new().expect("metal device");
        let mut kv = Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        kv.enable_gpu_attention(&device, GpuKvDtype::F32)
            .expect("enable gpu attention");

        kv.attention_append(3, &[1.0, 2.0], &[3.0, 4.0]);
        kv.finalize_token();

        let gpu_attention = kv
            .attention_gpu
            .as_mut()
            .expect("gpu attention should exist");
        gpu_attention.append_layer(3, &[5.0, 6.0], &[7.0, 8.0]);
        gpu_attention.finalize_token();

        kv.mark_attention_cpu_dirty();
        kv.finalize_token();
        kv.sync_attention_cpu_from_gpu_if_needed();

        assert_eq!(
            kv.attention_k_slice_including_current(3, 2),
            &[1.0, 2.0, 5.0, 6.0]
        );
        assert_eq!(
            kv.attention_v_slice_including_current(3, 2),
            &[3.0, 4.0, 7.0, 8.0]
        );
    }

    #[test]
    fn test_qwen35_kv_truncate_attention_to_preserves_valid_cpu_prefix_when_gpu_dirty() {
        let device = ax_engine_metal::MetalDevice::new().expect("metal device");
        let mut kv = Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        kv.enable_gpu_attention(&device, GpuKvDtype::F32)
            .expect("enable gpu attention");

        kv.attention_append(3, &[1.0, 2.0], &[3.0, 4.0]);
        kv.finalize_token();

        {
            let gpu_attention = kv.gpu_attention_mut().expect("gpu attention should exist");
            gpu_attention.append_layer(3, &[5.0, 6.0], &[7.0, 8.0]);
            gpu_attention.finalize_token();
        }
        kv.mark_attention_cpu_dirty();
        kv.finalize_token();

        kv.truncate_attention_to(1);

        assert_eq!(kv.seq_len(), 1);
        assert_eq!(kv.attention_k_slice_including_current(3, 1), &[1.0, 2.0]);
        assert_eq!(kv.attention_v_slice_including_current(3, 1), &[3.0, 4.0]);
        kv.sync_attention_cpu_from_gpu_if_needed();
        assert_eq!(kv.attention_k_slice_including_current(3, 1), &[1.0, 2.0]);
        assert_eq!(kv.attention_v_slice_including_current(3, 1), &[3.0, 4.0]);
    }

    #[test]
    #[should_panic(
        expected = "cannot snapshot inactive qwen35 recurrent slot while attention KV is shared"
    )]
    fn test_qwen35_kv_rejects_full_snapshot_for_inactive_slot() {
        let mut kv = Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let slot1 = kv.allocate_recurrent_slot();
        let _ = kv.snapshot_slot(slot1);
    }

    #[test]
    #[should_panic(
        expected = "cannot set active qwen35 recurrent slot seqlen_offset 1 != shared attention seq_len 0"
    )]
    fn test_qwen35_kv_rejects_setting_active_slot_seq_len_offset_to_mismatched_shared_seq_len() {
        let mut kv = Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        kv.set_recurrent_seqlen_offset(0, 1);
    }

    #[test]
    #[should_panic(
        expected = "cannot restore active qwen35 recurrent slot with seqlen_offset 1 != shared attention seq_len 0"
    )]
    fn test_qwen35_kv_rejects_restoring_active_slot_with_mismatched_shared_seq_len() {
        let mut kv = Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let mut snapshot = kv.recurrent_slot_snapshot(0);
        snapshot.seqlen_offset = 1;
        kv.restore_recurrent_slot(0, &snapshot);
    }

    #[test]
    fn test_qwen35_kv_full_snapshot_restores_into_fresh_kv() {
        let mut source = Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let slot1 = source.allocate_recurrent_slot();
        source.set_active_slot(slot1);

        source.conv_state_for_slot_mut(slot1, 0).fill(1.5);
        source.recurrent_state_for_slot_mut(slot1, 0).fill(2.5);
        source.attention_append(3, &[10.0, 11.0], &[12.0, 13.0]);
        source.finalize_token();

        let snapshot = source.snapshot_active_slot();

        let mut restored = Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        restored.restore_snapshot(&snapshot);

        assert_eq!(restored.active_slot(), slot1);
        assert_eq!(restored.seq_len(), 1);
        assert_eq!(
            restored.attention_k_slice_including_current(3, 1),
            &[10.0, 11.0]
        );
        assert_eq!(
            restored.attention_v_slice_including_current(3, 1),
            &[12.0, 13.0]
        );
        assert!(
            restored
                .conv_state_for_slot(slot1, 0)
                .iter()
                .all(|&v| v == 1.5)
        );
        assert!(
            restored
                .recurrent_state_for_slot(slot1, 0)
                .iter()
                .all(|&v| v == 2.5)
        );
        assert_eq!(restored.recurrent_seqlen_offset(slot1), 1);
    }

    #[test]
    fn test_qwen35_kv_restore_snapshot_clears_unrelated_allocated_slots() {
        let mut source = Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let slot1 = source.allocate_recurrent_slot();
        source.set_active_slot(slot1);
        source.conv_state_for_slot_mut(slot1, 0).fill(1.5);
        source.recurrent_state_for_slot_mut(slot1, 0).fill(2.5);
        source.attention_append(3, &[10.0, 11.0], &[12.0, 13.0]);
        source.finalize_token();
        let snapshot = source.snapshot_active_slot();

        let mut restored = Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let slot1_restored = restored.allocate_recurrent_slot();
        let slot2 = restored.allocate_recurrent_slot();
        assert_eq!(slot1_restored, slot1);
        restored.conv_state_for_slot_mut(slot2, 0).fill(7.0);
        restored.recurrent_state_for_slot_mut(slot2, 0).fill(8.0);

        restored.restore_snapshot(&snapshot);

        assert_eq!(restored.active_slot(), slot1);
        assert!(!restored.has_recurrent_slot(slot2));
        let reused = restored.allocate_recurrent_slot();
        assert_eq!(reused, slot2);
        assert!(
            restored
                .conv_state_for_slot(reused, 0)
                .iter()
                .all(|&v| v == 0.0)
        );
        assert!(
            restored
                .recurrent_state_for_slot(reused, 0)
                .iter()
                .all(|&v| v == 0.0)
        );
    }

    #[test]
    fn test_qwen35_kv_27b_shaped_snapshot_restore_clears_unrelated_slots() {
        let mut source = Qwen35Kv::new(
            4,   // keep layer count small while exercising 27B-shaped recurrent strides
            4,   // Qwen3.5-27B KV heads
            256, // Qwen3.5-27B head dim
            16, 4, 4, 6144, // 48 value heads * 128 value head dim
            128,  // linear key head dim
            48,   // linear value heads
            16,   // linear key heads / group count
        );
        let slot1 = source.allocate_recurrent_slot();
        source.set_active_slot(slot1);
        source.conv_state_for_slot_mut(slot1, 0).fill(0.25);
        source.recurrent_state_for_slot_mut(slot1, 0).fill(0.5);
        source.attention_append(3, &vec![1.0; 4 * 256], &vec![2.0; 4 * 256]);
        source.finalize_token();
        let snapshot = source.snapshot_active_slot();

        let mut restored = Qwen35Kv::new(4, 4, 256, 16, 4, 4, 6144, 128, 48, 16);
        let slot1_restored = restored.allocate_recurrent_slot();
        let slot2 = restored.allocate_recurrent_slot();
        assert_eq!(slot1_restored, slot1);
        restored.conv_state_for_slot_mut(slot2, 0).fill(7.0);
        restored.recurrent_state_for_slot_mut(slot2, 0).fill(8.0);

        restored.restore_snapshot(&snapshot);

        assert_eq!(restored.active_slot(), slot1);
        assert!(!restored.has_recurrent_slot(slot2));
        assert_eq!(
            restored.attention_k_slice_including_current(3, 1)[..8],
            [1.0; 8]
        );
        assert_eq!(
            restored.attention_v_slice_including_current(3, 1)[..8],
            [2.0; 8]
        );
    }

    #[test]
    #[should_panic(expected = "qwen35 recurrent slot batch must not contain duplicate slots")]
    fn test_qwen35_kv_rejects_gathering_duplicate_slot_batch() {
        let kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let conv_state_len = kv.conv_state_for_slot(0, 0).len();
        let recurrent_state_len = kv.recurrent_state_for_slot(0, 0).len();
        let mut conv_batch = vec![0.0; 2 * conv_state_len];
        let mut recurrent_batch = vec![0.0; 2 * recurrent_state_len];
        kv.gather_recurrent_state_batch(&[0, 0], 0, &mut conv_batch, &mut recurrent_batch);
    }

    #[test]
    #[should_panic(
        expected = "qwen35 recurrent slot batch slot 1 has seqlen_offset 1 != shared attention seq_len 0"
    )]
    fn test_qwen35_kv_rejects_gathering_slot_batch_with_misaligned_shared_seq_len() {
        let mut kv = Qwen35Kv::new(4, 8, 128, 1024, 4, 4, 1024, 128, 8, 2);
        let slot1 = kv.allocate_recurrent_slot();
        kv.set_recurrent_seqlen_offset(slot1, 1);
        let conv_state_len = kv.conv_state_for_slot(0, 0).len();
        let recurrent_state_len = kv.recurrent_state_for_slot(0, 0).len();
        let mut conv_batch = vec![0.0; 2 * conv_state_len];
        let mut recurrent_batch = vec![0.0; 2 * recurrent_state_len];
        kv.gather_recurrent_state_batch(&[0, slot1], 0, &mut conv_batch, &mut recurrent_batch);
    }
}
