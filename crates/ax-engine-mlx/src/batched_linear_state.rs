//! Batched (multi-row) linear-attention recurrent-state store — Phase 3.7
//! foundation for batching Qwen3-Next (MoE + gated-delta) decode.
//!
//! ## Why this exists
//!
//! Full-attention batched decode stores per-row K/V in
//! [`crate::batched_kv_cache::BatchedKvCache`]. Gated-delta (linear-attention)
//! layers instead carry a fixed-size recurrent state per row —
//! `conv_state [1, conv_kernel-1, conv_dim]` and
//! `recurrent_state [1, value_heads, value_dim, key_dim]` in the single-request
//! path (`crate::kv_cache` `LinearLayerState`). To batch B concurrent
//! Qwen3-Next requests, those states must be held as `[B, ...]` so the
//! already-batch-capable `gated_delta_decode_kernel` (which takes a `batch`
//! parameter) can advance all rows in one dispatch.
//!
//! Unlike KV, linear state does **not** grow with sequence length — it is a
//! fixed per-row tensor updated in place each step. So this store is simpler
//! than `BatchedKvCache`: no lengths, no ragged padding, no append/max_len.
//!
//! Built and tested **in isolation**, deliberately NOT wired into the runner
//! yet (that is the numerical-integration increment). The correctness contract
//! is the oracle test: for any sequence of seeds/removes, **row `r` of the
//! batched state is byte-identical to the single-row state it was seeded from**.
//!
//! ## Layout
//!
//! Per layer, `conv` and `recurrent` are `Option<MlxArray>` of shape
//! `[batch, ...single_row_dims]` (batch on axis 0), or `None` before the first
//! row is added. `add_row`/`remove_row` grow/shrink axis 0; `remove_row` uses
//! swap-remove so `slot → row` stays a compact prefix, matching the runner's
//! slot bookkeeping.

use mlx_sys::{MlxArray, concatenate, slice};

/// One linear-attention layer's batched recurrent state.
#[derive(Default)]
struct BatchedLinearLayer {
    /// `[batch, conv_kernel-1, conv_dim]`, or `None` before any row is added.
    conv: Option<MlxArray>,
    /// `[batch, value_heads, value_dim, key_dim]`, or `None`.
    recurrent: Option<MlxArray>,
}

/// A batched linear-attention recurrent-state store across `num_layers`
/// gated-delta layers and up to `capacity` concurrent rows.
pub struct BatchedLinearState {
    layers: Vec<BatchedLinearLayer>,
    batch: usize,
    capacity: usize,
}

/// The batch (axis-0) length of a state tensor.
fn axis0_len(a: &MlxArray) -> i32 {
    a.shape().first().copied().unwrap_or(0)
}

/// Slice rows `[start, end)` off axis 0, preserving all trailing dims.
fn slice_rows(a: &MlxArray, start: i32, end: i32) -> MlxArray {
    let shape = a.shape();
    let mut lo = vec![0i32; shape.len()];
    let mut hi = shape.clone();
    let mut step = vec![1i32; shape.len()];
    lo[0] = start;
    hi[0] = end;
    step[0] = 1;
    slice(a, &lo, &hi, &step, None)
}

impl BatchedLinearState {
    /// A store for `num_layers` layers, up to `max_batch` rows, starting empty.
    pub fn with_capacity(num_layers: usize, max_batch: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| BatchedLinearLayer::default())
                .collect(),
            batch: 0,
            capacity: max_batch,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn is_empty(&self) -> bool {
        self.batch == 0
    }

    /// Append one row seeded from a per-row (`[1, ...]`) state for every layer.
    /// `conv[l]` / `recurrent[l]` are the single-row states for layer `l`, in
    /// layer order. Returns the new row's slot index. Panics if full or if the
    /// per-layer state count does not match `num_layers`.
    pub fn add_row(&mut self, conv: &[MlxArray], recurrent: &[MlxArray]) -> usize {
        assert!(self.batch < self.capacity, "batched linear state full");
        assert_eq!(conv.len(), self.layers.len(), "one conv state per layer");
        assert_eq!(
            recurrent.len(),
            self.layers.len(),
            "one recurrent state per layer"
        );
        let slot = self.batch;
        for (l, layer) in self.layers.iter_mut().enumerate() {
            debug_assert_eq!(
                axis0_len(&conv[l]),
                1,
                "seed conv state must be a single row"
            );
            debug_assert_eq!(
                axis0_len(&recurrent[l]),
                1,
                "seed recurrent state must be a single row"
            );
            layer.conv = Some(match layer.conv.take() {
                Some(existing) => concatenate(&[&existing, &conv[l]], 0, None),
                None => conv[l].clone(),
            });
            layer.recurrent = Some(match layer.recurrent.take() {
                Some(existing) => concatenate(&[&existing, &recurrent[l]], 0, None),
                None => recurrent[l].clone(),
            });
        }
        self.batch += 1;
        slot
    }

    /// Remove `slot` via swap-remove: the last row is moved into `slot`, then
    /// the batch shrinks by one — mirroring `Vec::swap_remove`, so the caller's
    /// `slot → request` mapping stays a compact prefix. No-op returning `false`
    /// if `slot` is out of range.
    pub fn remove_row(&mut self, slot: usize) -> bool {
        if slot >= self.batch {
            return false;
        }
        let last = self.batch - 1;
        for layer in &mut self.layers {
            layer.conv = Some(swap_remove_row(
                layer.conv.as_ref().expect("active layer has conv state"),
                slot,
                last,
            ));
            layer.recurrent = Some(swap_remove_row(
                layer
                    .recurrent
                    .as_ref()
                    .expect("active layer has recurrent state"),
                slot,
                last,
            ));
        }
        self.batch -= 1;
        true
    }

    /// The full `[batch, ...]` conv/recurrent state for `layer`, for feeding the
    /// batched gated-delta kernel. `None` if no rows are active.
    pub fn layer_state(&self, layer: usize) -> Option<(&MlxArray, &MlxArray)> {
        let l = self.layers.get(layer)?;
        Some((l.conv.as_ref()?, l.recurrent.as_ref()?))
    }

    /// Replace `layer`'s conv/recurrent with the kernel's updated `[batch, ...]`
    /// outputs. Panics if the leading dim does not match the active batch.
    pub fn update_layer(&mut self, layer: usize, conv: MlxArray, recurrent: MlxArray) {
        assert_eq!(
            axis0_len(&conv) as usize,
            self.batch,
            "updated conv state batch mismatch"
        );
        assert_eq!(
            axis0_len(&recurrent) as usize,
            self.batch,
            "updated recurrent state batch mismatch"
        );
        let l = &mut self.layers[layer];
        l.conv = Some(conv);
        l.recurrent = Some(recurrent);
    }

    /// Read row `row` of `layer` back as a single-row `[1, ...]` state — the
    /// inverse of `add_row`'s seed, used by the oracle test and any writeback.
    pub fn row_state(&self, layer: usize, row: usize) -> Option<(MlxArray, MlxArray)> {
        if row >= self.batch {
            return None;
        }
        let l = self.layers.get(layer)?;
        let conv = slice_rows(l.conv.as_ref()?, row as i32, row as i32 + 1);
        let recurrent = slice_rows(l.recurrent.as_ref()?, row as i32, row as i32 + 1);
        Some((conv, recurrent))
    }
}

/// Swap-remove row `slot`: move row `last` into `slot`, then drop the last row.
/// Byte-preserving on the surviving rows (concat of two slices, never a
/// re-materialize of the moved data).
fn swap_remove_row(a: &MlxArray, slot: usize, last: usize) -> MlxArray {
    let slot = slot as i32;
    let last = last as i32;
    if slot == last {
        // Removing the last row — just drop it.
        return slice_rows(a, 0, last);
    }
    let moved = slice_rows(a, last, last + 1);
    if slot == 0 {
        // [moved, 1..last)
        let tail = slice_rows(a, 1, last);
        return concatenate(&[&moved, &tail], 0, None);
    }
    // [0..slot), moved, (slot+1..last)
    let head = slice_rows(a, 0, slot);
    let tail = slice_rows(a, slot + 1, last);
    concatenate(&[&head, &moved, &tail], 0, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_sys::{MlxDtype, eval};

    /// A distinctive single-row conv/recurrent pair for row `seed`.
    fn seed_states(seed: f32, layers: usize) -> (Vec<MlxArray>, Vec<MlxArray>) {
        let conv: Vec<MlxArray> = (0..layers)
            .map(|l| {
                let data = [seed + l as f32 * 0.1, seed + l as f32 * 0.1 + 0.01];
                MlxArray::from_raw_data(
                    data.as_ptr() as *const u8,
                    std::mem::size_of_val(&data),
                    &[1, 2, 1],
                    MlxDtype::Float32,
                )
            })
            .collect();
        let recurrent: Vec<MlxArray> = (0..layers)
            .map(|l| {
                let data = [seed * 10.0 + l as f32, seed * 10.0 + l as f32 + 0.5];
                MlxArray::from_raw_data(
                    data.as_ptr() as *const u8,
                    std::mem::size_of_val(&data),
                    &[1, 1, 2, 1],
                    MlxDtype::Float32,
                )
            })
            .collect();
        (conv, recurrent)
    }

    fn assert_row_matches(state: &BatchedLinearState, row: usize, seed: f32, layers: usize) {
        let (exp_conv, exp_rec) = seed_states(seed, layers);
        for l in 0..layers {
            let (conv, rec) = state.row_state(l, row).expect("row present");
            eval(&[&conv, &rec, &exp_conv[l], &exp_rec[l]]);
            assert_eq!(
                conv.data_f32(),
                exp_conv[l].data_f32(),
                "conv row {row} layer {l}"
            );
            assert_eq!(
                rec.data_f32(),
                exp_rec[l].data_f32(),
                "recurrent row {row} layer {l}"
            );
        }
    }

    #[test]
    fn seed_and_readback_is_byte_identical() {
        const LAYERS: usize = 3;
        let mut state = BatchedLinearState::with_capacity(LAYERS, 8);
        for r in 0..4u32 {
            let (c, rec) = seed_states(r as f32 + 1.0, LAYERS);
            let slot = state.add_row(&c, &rec);
            assert_eq!(slot, r as usize);
        }
        assert_eq!(state.batch(), 4);
        for r in 0..4 {
            assert_row_matches(&state, r, r as f32 + 1.0, LAYERS);
        }
    }

    #[test]
    fn swap_remove_moves_last_row_into_the_hole() {
        const LAYERS: usize = 2;
        let mut state = BatchedLinearState::with_capacity(LAYERS, 8);
        // Rows seeded 1,2,3,4 in slots 0,1,2,3.
        for r in 1..=4u32 {
            let (c, rec) = seed_states(r as f32, LAYERS);
            state.add_row(&c, &rec);
        }
        // Remove slot 1 (seed 2): slot 3 (seed 4) moves into slot 1.
        assert!(state.remove_row(1));
        assert_eq!(state.batch(), 3);
        assert_row_matches(&state, 0, 1.0, LAYERS); // unchanged
        assert_row_matches(&state, 1, 4.0, LAYERS); // last row moved here
        assert_row_matches(&state, 2, 3.0, LAYERS); // unchanged
        assert!(state.row_state(0, 3).is_none()); // shrunk
    }

    #[test]
    fn remove_last_row_just_shrinks() {
        const LAYERS: usize = 1;
        let mut state = BatchedLinearState::with_capacity(LAYERS, 4);
        for r in 1..=3u32 {
            let (c, rec) = seed_states(r as f32, LAYERS);
            state.add_row(&c, &rec);
        }
        assert!(state.remove_row(2)); // remove last
        assert_eq!(state.batch(), 2);
        assert_row_matches(&state, 0, 1.0, LAYERS);
        assert_row_matches(&state, 1, 2.0, LAYERS);
    }

    #[test]
    fn update_layer_replaces_the_batched_state() {
        const LAYERS: usize = 1;
        let mut state = BatchedLinearState::with_capacity(LAYERS, 4);
        for r in 1..=2u32 {
            let (c, rec) = seed_states(r as f32, LAYERS);
            state.add_row(&c, &rec);
        }
        // Kernel-style update: a fresh [batch, ...] state for layer 0.
        let new_conv = mlx_sys::ops::zeros(&[2, 2, 1], MlxDtype::Float32, None);
        let new_rec = mlx_sys::ops::zeros(&[2, 1, 2, 1], MlxDtype::Float32, None);
        state.update_layer(0, new_conv, new_rec);
        let (c, rec) = state.row_state(0, 0).expect("row present");
        eval(&[&c, &rec]);
        assert!(c.data_f32().iter().all(|&x| x == 0.0));
        assert!(rec.data_f32().iter().all(|&x| x == 0.0));
    }
}
