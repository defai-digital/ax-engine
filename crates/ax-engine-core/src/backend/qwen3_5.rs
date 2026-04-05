//! Qwen3.5-specific backend helpers and recurrent state contracts.

use super::Backend;

/// Slot-indexed recurrent-state batch for Qwen3.5 hybrid layers.
///
/// The batch is expressed as one or more live recurrent slots, each with an
/// equal number of new tokens. The underlying state buffers stay flattened for
/// the current backend implementations, but slot ownership is explicit at the
/// API boundary so the runtime can evolve toward true `state_indices` execution
/// without another backend refactor.
///
/// This type intentionally remains in `backend` for now even though it is
/// Qwen3.5-specific. AX currently has only one hybrid recurrent architecture,
/// so keeping the batch contract next to the backend entrypoints avoids a
/// larger trait refactor until a second hybrid architecture arrives.
#[derive(Debug)]
pub struct Qwen3_5RecurrentStateBatch<'a> {
    layer_idx: usize,
    slot_indices: &'a [usize],
    conv_state_batch: &'a mut [f32],
    recurrent_state_batch: &'a mut [f32],
    conv_state_stride: usize,
    recurrent_state_stride: usize,
}

#[allow(non_camel_case_types)]
pub type Qwen35RecurrentStateBatch<'a> = Qwen3_5RecurrentStateBatch<'a>;

impl<'a> Qwen3_5RecurrentStateBatch<'a> {
    pub fn new(
        layer_idx: usize,
        slot_indices: &'a [usize],
        conv_state_batch: &'a mut [f32],
        recurrent_state_batch: &'a mut [f32],
        conv_state_stride: usize,
        recurrent_state_stride: usize,
    ) -> Self {
        assert!(
            !slot_indices.is_empty(),
            "qwen35 recurrent state batch requires at least one slot"
        );
        assert!(
            conv_state_stride > 0,
            "qwen35 recurrent conv state stride must be > 0"
        );
        assert!(
            recurrent_state_stride > 0,
            "qwen35 recurrent state stride must be > 0"
        );
        for (i, &slot_idx) in slot_indices.iter().enumerate() {
            for &other_slot_idx in &slot_indices[..i] {
                assert!(
                    slot_idx != other_slot_idx,
                    "qwen35 recurrent state batch must not contain duplicate slots"
                );
            }
        }
        assert_eq!(
            conv_state_batch.len(),
            slot_indices.len() * conv_state_stride,
            "qwen35 recurrent conv state batch has wrong length"
        );
        assert_eq!(
            recurrent_state_batch.len(),
            slot_indices.len() * recurrent_state_stride,
            "qwen35 recurrent state batch has wrong length"
        );
        Self {
            layer_idx,
            slot_indices,
            conv_state_batch,
            recurrent_state_batch,
            conv_state_stride,
            recurrent_state_stride,
        }
    }

    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    pub fn slot_count(&self) -> usize {
        self.slot_indices.len()
    }

    pub fn slot_indices(&self) -> &[usize] {
        self.slot_indices
    }

    pub fn slot_index(&self, batch_idx: usize) -> usize {
        self.slot_indices[batch_idx]
    }

    pub fn conv_state_stride(&self) -> usize {
        self.conv_state_stride
    }

    pub fn recurrent_state_stride(&self) -> usize {
        self.recurrent_state_stride
    }

    pub fn conv_state_for_slot_mut(&mut self, batch_idx: usize) -> &mut [f32] {
        let start = batch_idx * self.conv_state_stride;
        let end = start + self.conv_state_stride;
        &mut self.conv_state_batch[start..end]
    }

    pub fn recurrent_state_for_slot_mut(&mut self, batch_idx: usize) -> &mut [f32] {
        let start = batch_idx * self.recurrent_state_stride;
        let end = start + self.recurrent_state_stride;
        &mut self.recurrent_state_batch[start..end]
    }

    pub fn recurrent_buffers_for_slot_mut(&mut self, batch_idx: usize) -> (&mut [f32], &mut [f32]) {
        let conv_start = batch_idx * self.conv_state_stride;
        let conv_end = conv_start + self.conv_state_stride;
        let rec_start = batch_idx * self.recurrent_state_stride;
        let rec_end = rec_start + self.recurrent_state_stride;
        (
            &mut self.conv_state_batch[conv_start..conv_end],
            &mut self.recurrent_state_batch[rec_start..rec_end],
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_recurrent_sequence_via_backend(
    backend: &(impl Backend + ?Sized),
    qkv_batch: &[f32],
    beta_batch: &mut [f32],
    alpha_batch: &mut [f32],
    dt_bias: &[f32],
    a: &[f32],
    conv_kernel: &[f32],
    state_batch: &mut Qwen3_5RecurrentStateBatch<'_>,
    output_batch: &mut [f32],
    tokens_per_slot: usize,
    cfg: crate::compute::gdn::Qwen35RecurrentConfig,
) {
    assert!(
        tokens_per_slot > 0,
        "qwen35 recurrent sequence requires tokens_per_slot > 0"
    );
    let slot_count = state_batch.slot_count();
    let total_tokens = slot_count * tokens_per_slot;
    let key_dim = cfg.key_dim();
    let value_dim = cfg.value_dim();
    assert_eq!(
        qkv_batch.len(),
        total_tokens * cfg.conv_dim,
        "qwen35 recurrent qkv batch has wrong length"
    );
    assert_eq!(
        beta_batch.len(),
        total_tokens * cfg.time_step_rank,
        "qwen35 recurrent beta batch has wrong length"
    );
    assert_eq!(
        alpha_batch.len(),
        total_tokens * cfg.time_step_rank,
        "qwen35 recurrent alpha batch has wrong length"
    );
    assert_eq!(
        output_batch.len(),
        total_tokens * value_dim,
        "qwen35 recurrent output batch has wrong length"
    );
    assert_eq!(
        state_batch.conv_state_stride(),
        cfg.conv_cache_len * cfg.conv_dim,
        "qwen35 recurrent conv state stride mismatch"
    );
    assert_eq!(
        state_batch.recurrent_state_stride(),
        cfg.time_step_rank * cfg.state_size * cfg.state_size,
        "qwen35 recurrent state stride mismatch"
    );

    let mut conv_out_batch = vec![0.0f32; total_tokens * cfg.conv_dim];
    let mut q_batch = vec![0.0f32; total_tokens * value_dim];
    let mut k_batch = vec![0.0f32; total_tokens * value_dim];
    let mut v_batch = vec![0.0f32; total_tokens * value_dim];

    crate::compute::gdn::prepare_alpha_beta(alpha_batch, beta_batch, dt_bias, a);

    for batch_idx in 0..slot_count {
        let _slot_idx = state_batch.slot_index(batch_idx);
        let token_start = batch_idx * tokens_per_slot;
        let token_end = token_start + tokens_per_slot;
        let qkv_start = token_start * cfg.conv_dim;
        let qkv_end = token_end * cfg.conv_dim;
        backend.qwen35_causal_conv_sequence(
            &qkv_batch[qkv_start..qkv_end],
            conv_kernel,
            state_batch.conv_state_for_slot_mut(batch_idx),
            &mut conv_out_batch[qkv_start..qkv_end],
            tokens_per_slot,
            cfg.conv_cache_len,
            cfg.conv_dim,
        );
    }

    let mut q_lin = vec![0.0f32; key_dim];
    let mut k_lin = vec![0.0f32; key_dim];
    let mut q_rep = vec![0.0f32; value_dim];
    let mut k_rep = vec![0.0f32; value_dim];

    for token_idx in 0..total_tokens {
        let conv_start = token_idx * cfg.conv_dim;
        let conv_end = conv_start + cfg.conv_dim;
        let conv_out = &conv_out_batch[conv_start..conv_end];
        q_lin.copy_from_slice(&conv_out[..key_dim]);
        k_lin.copy_from_slice(&conv_out[key_dim..2 * key_dim]);
        let v_lin = &conv_out[2 * key_dim..2 * key_dim + value_dim];

        crate::compute::gdn::l2_norm_heads(
            &mut q_lin,
            cfg.group_count,
            cfg.state_size,
            cfg.rms_norm_eps,
        );
        crate::compute::gdn::l2_norm_heads(
            &mut k_lin,
            cfg.group_count,
            cfg.state_size,
            cfg.rms_norm_eps,
        );

        crate::compute::gdn::repeat_heads_into(
            &mut q_rep,
            &q_lin,
            cfg.group_count,
            cfg.time_step_rank,
            cfg.state_size,
        );
        crate::compute::gdn::repeat_heads_into(
            &mut k_rep,
            &k_lin,
            cfg.group_count,
            cfg.time_step_rank,
            cfg.state_size,
        );
        let out_start = token_idx * value_dim;
        let out_end = out_start + value_dim;
        q_batch[out_start..out_end].copy_from_slice(&q_rep);
        k_batch[out_start..out_end].copy_from_slice(&k_rep);
        v_batch[out_start..out_end].copy_from_slice(v_lin);
    }

    for batch_idx in 0..slot_count {
        let _slot_idx = state_batch.slot_index(batch_idx);
        let token_start = batch_idx * tokens_per_slot;
        let token_end = token_start + tokens_per_slot;
        let qkv_start = token_start * value_dim;
        let qkv_end = token_end * value_dim;
        let gate_start = token_start * cfg.time_step_rank;
        let gate_end = token_end * cfg.time_step_rank;
        backend.qwen35_gated_delta_sequence(
            &q_batch[qkv_start..qkv_end],
            &k_batch[qkv_start..qkv_end],
            &v_batch[qkv_start..qkv_end],
            &alpha_batch[gate_start..gate_end],
            &beta_batch[gate_start..gate_end],
            state_batch.recurrent_state_for_slot_mut(batch_idx),
            &mut output_batch[qkv_start..qkv_end],
            tokens_per_slot,
            cfg.time_step_rank,
            cfg.state_size,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn qwen3_5_recurrent_sequence_via_backend(
    backend: &(impl Backend + ?Sized),
    qkv_batch: &[f32],
    beta_batch: &mut [f32],
    alpha_batch: &mut [f32],
    dt_bias: &[f32],
    a: &[f32],
    conv_kernel: &[f32],
    state_batch: &mut Qwen3_5RecurrentStateBatch<'_>,
    output_batch: &mut [f32],
    tokens_per_slot: usize,
    cfg: crate::compute::gdn::Qwen35RecurrentConfig,
) {
    qwen35_recurrent_sequence_via_backend(
        backend,
        qkv_batch,
        beta_batch,
        alpha_batch,
        dt_bias,
        a,
        conv_kernel,
        state_batch,
        output_batch,
        tokens_per_slot,
        cfg,
    );
}
