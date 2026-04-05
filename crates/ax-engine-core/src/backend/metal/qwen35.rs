//! Qwen3.5-specific Metal backend support: recurrent scratch, slot sync, and perf counters.

use super::*;
use crate::backend::{Qwen3_5RecurrentStateBatch, qwen35_recurrent_sequence_via_backend};

pub(super) fn qwen35_gate_up_scratch_dims(
    config: &ModelConfig,
    q_dim: usize,
    inter_dim: usize,
) -> (usize, usize) {
    if !matches!(config.architecture.as_str(), "qwen35" | "qwen35moe") {
        return (inter_dim, inter_dim);
    }

    let recurrent_inner_dim = config.qwen35_ssm_inner_size.unwrap_or(0) as usize;
    let gate_dim = inter_dim.max(q_dim * 2).max(recurrent_inner_dim);
    let up_dim = inter_dim.max(q_dim).max(recurrent_inner_dim);
    (gate_dim, up_dim)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Qwen35RecurrentBatchPerfCounters {
    pub conv_ns: u64,
    pub pack_ns: u64,
    pub gated_delta_ns: u64,
    pub unpack_ns: u64,
    pub qkv_handoff_ns: u64,
    pub qkv_handoff_layers: u64,
    pub qkv_handoff_fused_tail_layers: u64,
    pub qkv_gpu_projection_layers: u64,
    pub qkv_fast_path_eligible_layers: u64,
    pub gpu_ssm_projection_layers: u64,
    pub qkv_fast_reject_state_size_layers: u64,
    pub qkv_fast_reject_group_divisibility_layers: u64,
    pub qkv_fast_reject_missing_batch_scratches_layers: u64,
    pub qkv_fast_reject_q_capacity_layers: u64,
    pub qkv_fast_reject_k_capacity_layers: u64,
    pub qkv_fast_reject_v_capacity_layers: u64,
    pub qkv_fast_reject_gate_capacity_layers: u64,
    pub qkv_fast_reject_up_capacity_layers: u64,
    pub qkv_handoff_cpu_alias_layers: u64,
    pub qkv_handoff_slot_buffer_layers: u64,
    pub qkv_handoff_backend_carryover_layers: u64,
    pub qkv_handoff_backend_zero_init_layers: u64,
    pub qkv_handoff_cpu_materialization_layers: u64,
    pub state_batch_backend_native_layers: u64,
    pub state_batch_cpu_direct_layers: u64,
    pub state_batch_cpu_direct_materialized_from_backend_layers: u64,
    pub state_batch_cpu_gathered_layers: u64,
    pub state_batch_cpu_gathered_materialized_from_backend_layers: u64,
}

#[derive(Default)]
pub(super) struct Qwen35RecurrentBatchPerfState {
    conv_ns: AtomicU64,
    pack_ns: AtomicU64,
    gated_delta_ns: AtomicU64,
    unpack_ns: AtomicU64,
    qkv_handoff_ns: AtomicU64,
    qkv_handoff_layers: AtomicU64,
    qkv_handoff_fused_tail_layers: AtomicU64,
    qkv_gpu_projection_layers: AtomicU64,
    qkv_fast_path_eligible_layers: AtomicU64,
    gpu_ssm_projection_layers: AtomicU64,
    qkv_fast_reject_state_size_layers: AtomicU64,
    qkv_fast_reject_group_divisibility_layers: AtomicU64,
    qkv_fast_reject_missing_batch_scratches_layers: AtomicU64,
    qkv_fast_reject_q_capacity_layers: AtomicU64,
    qkv_fast_reject_k_capacity_layers: AtomicU64,
    qkv_fast_reject_v_capacity_layers: AtomicU64,
    qkv_fast_reject_gate_capacity_layers: AtomicU64,
    qkv_fast_reject_up_capacity_layers: AtomicU64,
    qkv_handoff_cpu_alias_layers: AtomicU64,
    qkv_handoff_slot_buffer_layers: AtomicU64,
    qkv_handoff_backend_carryover_layers: AtomicU64,
    qkv_handoff_backend_zero_init_layers: AtomicU64,
    qkv_handoff_cpu_materialization_layers: AtomicU64,
    state_batch_backend_native_layers: AtomicU64,
    state_batch_cpu_direct_layers: AtomicU64,
    state_batch_cpu_direct_materialized_from_backend_layers: AtomicU64,
    state_batch_cpu_gathered_layers: AtomicU64,
    state_batch_cpu_gathered_materialized_from_backend_layers: AtomicU64,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Qwen35SlotBufferSyncOutcome {
    pub(crate) used_backend_carryover: bool,
    pub(crate) used_backend_zero_init: bool,
    pub(crate) used_cpu_materialization: bool,
}

impl Qwen35SlotBufferSyncOutcome {
    pub(crate) fn note_backend_carryover(&mut self) {
        self.used_backend_carryover = true;
    }

    fn note_backend_zero_init(&mut self) {
        self.used_backend_zero_init = true;
    }

    fn note_cpu_materialization(&mut self) {
        self.used_cpu_materialization = true;
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct Qwen35RecurrentSlotBufferKey {
    pub(super) layer_idx: usize,
    pub(super) slot_idx: usize,
    pub(super) conv_state_stride: usize,
    pub(super) recurrent_state_stride: usize,
}

pub(crate) struct Qwen35MetalSlotBuffers {
    pub(crate) conv_state: MetalBuffer,
    pub(crate) recurrent_state: MetalBuffer,
    pub(crate) conv_synced_generation: Option<u64>,
    pub(crate) recurrent_synced_generation: Option<u64>,
    pub(super) source_kv_identity: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct Qwen35RecurrentScratchBufferKey {
    pub(crate) tokens_per_slot: usize,
    pub(crate) conv_dim: usize,
    pub(crate) time_step_rank: usize,
    pub(crate) state_size: usize,
}

pub(super) struct Qwen35MetalRecurrentScratchBuffers {
    pub(crate) input: MetalBuffer,
    pub(crate) conv_out: MetalBuffer,
    pub(crate) q: MetalBuffer,
    pub(crate) k: MetalBuffer,
    pub(crate) v: MetalBuffer,
    pub(crate) gate: MetalBuffer,
    pub(crate) beta: MetalBuffer,
    pub(crate) output: MetalBuffer,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct Qwen35QkvHandoffScratchBufferKey {
    n_tokens: usize,
    conv_dim: usize,
    inner_size: usize,
    time_step_rank: usize,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct Qwen35BatchProjectionScratchBufferKey {
    n_tokens: usize,
    output_dims: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct Qwen35BatchLogitsScratchBufferKey {
    hidden_len: usize,
    logits_len: usize,
}

pub(crate) struct Qwen35MetalQkvHandoffScratchBuffers {
    pub(crate) conv_out: MetalBuffer,
    pub(crate) k: MetalBuffer,
    pub(crate) v: MetalBuffer,
    pub(crate) alpha: MetalBuffer,
    pub(crate) beta: MetalBuffer,
    pub(crate) alpha_f16: MetalBuffer,
    pub(crate) beta_f16: MetalBuffer,
}

pub(crate) struct Qwen35MetalRecurrentProjectionScratchBuffers {
    pub(crate) qkv: MetalBuffer,
    pub(crate) z: MetalBuffer,
    pub(crate) beta: MetalBuffer,
    pub(crate) alpha: MetalBuffer,
}

pub(crate) struct Qwen35MetalBatchProjectionScratchBuffers {
    pub(crate) outputs: Vec<MetalBuffer>,
}

pub(crate) struct Qwen35MetalBatchLogitsScratchBuffers {
    pub(crate) hidden: MetalBuffer,
    pub(crate) hidden_f16: MetalBuffer,
    pub(crate) logits: MetalBuffer,
}

impl Qwen35MetalSlotBuffers {
    fn new(
        device: &MetalDevice,
        conv_state_stride: usize,
        recurrent_state_stride: usize,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            conv_state: MetalBuffer::new(
                device.device(),
                conv_state_stride * std::mem::size_of::<f32>(),
            )?,
            recurrent_state: MetalBuffer::new(
                device.device(),
                recurrent_state_stride * std::mem::size_of::<f32>(),
            )?,
            conv_synced_generation: None,
            recurrent_synced_generation: None,
            source_kv_identity: None,
        })
    }
}

impl Qwen35MetalRecurrentScratchBuffers {
    fn new(device: &MetalDevice, key: Qwen35RecurrentScratchBufferKey) -> anyhow::Result<Self> {
        let value_dim = key.time_step_rank * key.state_size;
        Ok(Self {
            input: MetalBuffer::new(
                device.device(),
                key.tokens_per_slot * key.conv_dim * std::mem::size_of::<f32>(),
            )?,
            conv_out: MetalBuffer::new(
                device.device(),
                key.tokens_per_slot * key.conv_dim * std::mem::size_of::<f32>(),
            )?,
            q: MetalBuffer::new(
                device.device(),
                key.tokens_per_slot * value_dim * std::mem::size_of::<f32>(),
            )?,
            k: MetalBuffer::new(
                device.device(),
                key.tokens_per_slot * value_dim * std::mem::size_of::<f32>(),
            )?,
            v: MetalBuffer::new(
                device.device(),
                key.tokens_per_slot * value_dim * std::mem::size_of::<f32>(),
            )?,
            gate: MetalBuffer::new(
                device.device(),
                key.tokens_per_slot * key.time_step_rank * std::mem::size_of::<f32>(),
            )?,
            beta: MetalBuffer::new(
                device.device(),
                key.tokens_per_slot * key.time_step_rank * std::mem::size_of::<f32>(),
            )?,
            output: MetalBuffer::new(
                device.device(),
                key.tokens_per_slot * value_dim * std::mem::size_of::<f32>(),
            )?,
        })
    }
}

impl Qwen35MetalBatchLogitsScratchBuffers {
    fn new(device: &MetalDevice, key: Qwen35BatchLogitsScratchBufferKey) -> anyhow::Result<Self> {
        Ok(Self {
            hidden: MetalBuffer::new(device.device(), key.hidden_len * std::mem::size_of::<f32>())?,
            hidden_f16: MetalBuffer::new(
                device.device(),
                key.hidden_len * std::mem::size_of::<half::f16>(),
            )?,
            logits: MetalBuffer::new(device.device(), key.logits_len * std::mem::size_of::<f32>())?,
        })
    }
}

impl Qwen35MetalQkvHandoffScratchBuffers {
    fn new(device: &MetalDevice, key: Qwen35QkvHandoffScratchBufferKey) -> anyhow::Result<Self> {
        Ok(Self {
            conv_out: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.conv_dim * std::mem::size_of::<f32>(),
            )?,
            k: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.inner_size * std::mem::size_of::<f32>(),
            )?,
            v: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.inner_size * std::mem::size_of::<f32>(),
            )?,
            alpha: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.time_step_rank * std::mem::size_of::<f32>(),
            )?,
            beta: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.time_step_rank * std::mem::size_of::<f32>(),
            )?,
            alpha_f16: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.time_step_rank * std::mem::size_of::<half::f16>(),
            )?,
            beta_f16: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.time_step_rank * std::mem::size_of::<half::f16>(),
            )?,
        })
    }
}

impl Qwen35MetalRecurrentProjectionScratchBuffers {
    fn new(device: &MetalDevice, key: Qwen35QkvHandoffScratchBufferKey) -> anyhow::Result<Self> {
        Ok(Self {
            qkv: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.conv_dim * std::mem::size_of::<f32>(),
            )?,
            z: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.inner_size * std::mem::size_of::<f32>(),
            )?,
            beta: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.time_step_rank * std::mem::size_of::<f32>(),
            )?,
            alpha: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.time_step_rank * std::mem::size_of::<f32>(),
            )?,
        })
    }
}

impl Qwen35MetalBatchProjectionScratchBuffers {
    fn new(
        device: &MetalDevice,
        key: &Qwen35BatchProjectionScratchBufferKey,
    ) -> anyhow::Result<Self> {
        let mut outputs = Vec::with_capacity(key.output_dims.len());
        for &out_dim in &key.output_dims {
            outputs.push(MetalBuffer::new(
                device.device(),
                key.n_tokens * out_dim * std::mem::size_of::<f32>(),
            )?);
        }
        Ok(Self { outputs })
    }
}

impl MetalBackend {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen35_causal_conv_sequence_sync(
        &self,
        input: &MetalBuffer,
        kernel: &MetalBuffer,
        conv_state: &MetalBuffer,
        output: &MetalBuffer,
        n_tokens: u32,
        conv_cache_len: u32,
        conv_dim: u32,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            conv_cache_len <= 8,
            "qwen35 causal conv Metal kernel supports conv_cache_len <= 8, got {conv_cache_len}",
        );
        self.device.execute_sync(|encoder| {
            self.gdn_kernels.encode_causal_conv_sequence(
                encoder,
                input,
                kernel,
                conv_state,
                output,
                n_tokens,
                conv_cache_len,
                conv_dim,
            );
            Ok(())
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen35_gated_delta_sequence_sync(
        &self,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        gate: &MetalBuffer,
        beta: &MetalBuffer,
        state: &MetalBuffer,
        output: &MetalBuffer,
        n_tokens: u32,
        n_heads: u32,
        head_dim: u32,
    ) -> anyhow::Result<()> {
        self.device.execute_sync(|encoder| {
            self.gdn_kernels.encode_gated_delta_sequence(
                encoder, q, k, v, gate, beta, state, output, n_tokens, n_heads, head_dim,
            );
            Ok(())
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qwen35_prepare_multi_token_qkv_sync(
        &self,
        conv_out: &MetalBuffer,
        alpha: &[f32],
        beta: &[f32],
        q_out: &MetalBuffer,
        k_out: &MetalBuffer,
        v_out: &MetalBuffer,
        gate_out: &MetalBuffer,
        beta_out: &MetalBuffer,
        n_tokens: u32,
        group_count: u32,
        time_step_rank: u32,
        state_size: u32,
        eps: f32,
    ) -> anyhow::Result<bool> {
        let alpha_buf = unsafe { MetalBuffer::from_slice_no_copy(self.device.device(), alpha) }
            .unwrap_or_else(|_| {
                MetalBuffer::from_slice(self.device.device(), alpha)
                    .expect("Failed to create Metal buffer for qwen35 recurrent alpha batch")
            });
        let beta_buf = unsafe { MetalBuffer::from_slice_no_copy(self.device.device(), beta) }
            .unwrap_or_else(|_| {
                MetalBuffer::from_slice(self.device.device(), beta)
                    .expect("Failed to create Metal buffer for qwen35 recurrent beta batch")
            });
        let mut encoded = false;
        self.device.execute_sync(|encoder| {
            encoded = self.gdn_kernels.encode_prepare_multi_token_qkv(
                encoder,
                conv_out,
                &alpha_buf,
                &beta_buf,
                q_out,
                k_out,
                v_out,
                gate_out,
                beta_out,
                n_tokens,
                group_count,
                time_step_rank,
                state_size,
                eps,
            );
            Ok(())
        })?;
        Ok(encoded)
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_prepare_multi_token_qkv_with_fallback(
        &self,
        conv_out: &MetalBuffer,
        alpha: &[f32],
        beta: &[f32],
        q: &mut MetalBuffer,
        k: &mut MetalBuffer,
        v: &mut MetalBuffer,
        gate: &mut MetalBuffer,
        beta_out: &mut MetalBuffer,
        tokens_per_slot: usize,
        value_dim: usize,
        group_count: usize,
        time_step_rank: usize,
        state_size: usize,
        eps: f32,
    ) -> anyhow::Result<()> {
        let mut packed = false;
        if state_size <= 256 && time_step_rank.is_multiple_of(group_count) {
            packed = self.qwen35_prepare_multi_token_qkv_sync(
                conv_out,
                alpha,
                beta,
                q,
                k,
                v,
                gate,
                beta_out,
                tokens_per_slot as u32,
                group_count as u32,
                time_step_rank as u32,
                state_size as u32,
                eps,
            )?;
        }

        if !packed {
            let conv_out_slice = unsafe { conv_out.as_slice::<f32>() };
            unsafe {
                prepare_multi_token_gdn_bh_buffers(
                    conv_out_slice,
                    alpha,
                    beta,
                    &mut q.as_mut_slice::<f32>()[..tokens_per_slot * value_dim],
                    &mut k.as_mut_slice::<f32>()[..tokens_per_slot * value_dim],
                    &mut v.as_mut_slice::<f32>()[..tokens_per_slot * value_dim],
                    &mut gate.as_mut_slice::<f32>()[..tokens_per_slot * time_step_rank],
                    &mut beta_out.as_mut_slice::<f32>()[..tokens_per_slot * time_step_rank],
                    tokens_per_slot,
                    group_count,
                    time_step_rank,
                    state_size,
                    eps,
                );
            }
        }
        Ok(())
    }

    pub(super) fn try_clone_qwen35_recurrent_slot_from_backend_owned(
        &self,
        qwen_kv: &mut crate::kv::Qwen3_5Kv,
        src_slot_idx: usize,
        dst_slot_idx: usize,
    ) -> bool {
        if !qwen_kv.has_recurrent_slot(src_slot_idx) {
            return false;
        }
        let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
        let recurrent_state_stride = qwen_kv.recurrent_state_len();
        let kv_identity = qwen_kv as *const crate::kv::Qwen3_5Kv as usize;
        let mut stale_layers = Vec::new();
        {
            let slot_cache = self.ops.qwen35_recurrent_slot_buffers.lock().unwrap();
            for layer_idx in 0..qwen_kv.layer_count() {
                if !qwen_kv.is_recurrent_layer(layer_idx) {
                    continue;
                }
                let conv_stale = qwen_kv.conv_state_cpu_stale(src_slot_idx, layer_idx);
                let recurrent_stale = qwen_kv.recurrent_state_cpu_stale(src_slot_idx, layer_idx);
                if !conv_stale && !recurrent_stale {
                    continue;
                }
                let slot_key = Qwen35RecurrentSlotBufferKey {
                    layer_idx,
                    slot_idx: src_slot_idx,
                    conv_state_stride,
                    recurrent_state_stride,
                };
                let Some(slot_buffers) = slot_cache.get(&slot_key) else {
                    return false;
                };
                if slot_buffers.source_kv_identity != Some(kv_identity) {
                    return false;
                }
                let conv_generation = qwen_kv.conv_state_generation(src_slot_idx, layer_idx);
                let recurrent_generation =
                    qwen_kv.recurrent_state_generation(src_slot_idx, layer_idx);
                if conv_stale && slot_buffers.conv_synced_generation != Some(conv_generation) {
                    return false;
                }
                if recurrent_stale
                    && slot_buffers.recurrent_synced_generation != Some(recurrent_generation)
                {
                    return false;
                }
                stale_layers.push((layer_idx, conv_stale, recurrent_stale));
            }
        }

        qwen_kv.clone_recurrent_slot_preserving_ownership(src_slot_idx, dst_slot_idx);

        if stale_layers.is_empty() {
            return true;
        }

        let mut slot_cache = self.ops.qwen35_recurrent_slot_buffers.lock().unwrap();
        for (layer_idx, conv_stale, recurrent_stale) in stale_layers {
            let src_key = Qwen35RecurrentSlotBufferKey {
                layer_idx,
                slot_idx: src_slot_idx,
                conv_state_stride,
                recurrent_state_stride,
            };
            let (src_conv_ptr, src_recurrent_ptr) = {
                let src_buffers = slot_cache
                    .get(&src_key)
                    .expect("validated qwen35 source slot buffer should exist");
                let conv =
                    conv_stale.then(|| src_buffers.conv_state.contents().as_ptr() as *const f32);
                let recurrent = recurrent_stale
                    .then(|| src_buffers.recurrent_state.contents().as_ptr() as *const f32);
                (conv, recurrent)
            };
            let dst_key = Qwen35RecurrentSlotBufferKey {
                layer_idx,
                slot_idx: dst_slot_idx,
                conv_state_stride,
                recurrent_state_stride,
            };
            let dst_buffers = slot_cache.entry(dst_key).or_insert_with(|| {
                Qwen35MetalSlotBuffers::new(&self.device, conv_state_stride, recurrent_state_stride)
                    .expect("Failed to allocate qwen35 recurrent Metal slot buffers")
            });
            if let Some(src_conv_ptr) = src_conv_ptr {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_conv_ptr,
                        dst_buffers.conv_state.contents().as_ptr() as *mut f32,
                        conv_state_stride,
                    );
                }
                dst_buffers.conv_synced_generation =
                    Some(qwen_kv.conv_state_generation(dst_slot_idx, layer_idx));
            }
            if let Some(src_recurrent_ptr) = src_recurrent_ptr {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_recurrent_ptr,
                        dst_buffers.recurrent_state.contents().as_ptr() as *mut f32,
                        recurrent_state_stride,
                    );
                }
                dst_buffers.recurrent_synced_generation =
                    Some(qwen_kv.recurrent_state_generation(dst_slot_idx, layer_idx));
            }
            dst_buffers.source_kv_identity = Some(kv_identity);
        }

        true
    }

    /// Get or grow the reusable input buffer. Copies `data` into it.
    /// Returns a lock guard holding the buffer.
    pub(super) fn prepare_input(
        &self,
        data: &[f32],
    ) -> std::sync::MutexGuard<'_, (Option<MetalBuffer>, usize)> {
        let needed = std::mem::size_of_val(data);
        let mut guard = self.input_buf.lock().unwrap();
        if guard.1 < needed {
            guard.0 = Some(
                MetalBuffer::new(self.device.device(), needed)
                    .expect("Failed to allocate input Metal buffer"),
            );
            guard.1 = needed;
        }
        // Copy input data into the pre-allocated buffer via raw pointer
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                guard.0.as_ref().unwrap().contents().as_ptr() as *mut u8,
                needed,
            );
        }
        guard
    }

    /// Get or grow the reusable output buffer.
    pub(super) fn prepare_output(
        &self,
        size_bytes: usize,
    ) -> std::sync::MutexGuard<'_, (Option<MetalBuffer>, usize)> {
        let mut guard = self.output_buf.lock().unwrap();
        if guard.1 < size_bytes {
            guard.0 = Some(
                MetalBuffer::new(self.device.device(), size_bytes)
                    .expect("Failed to allocate output Metal buffer"),
            );
            guard.1 = size_bytes;
        }
        guard
    }

    fn ensure_dense_weight_buffer(
        &self,
        a_quant: &[u8],
        dtype: GgmlType,
        m: usize,
        k: usize,
    ) -> usize {
        let key = a_quant.as_ptr() as usize;
        let mut cache = self.weight_cache.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            if dtype == GgmlType::F32 {
                create_mmap_weight_buffer_from_bytes(&self.device, a_quant)
            } else {
                let mut a_f32 = vec![0.0f32; m * k];
                crate::quant::dequantize(dtype, a_quant, &mut a_f32);
                MetalBuffer::from_slice(self.device.device(), &a_f32)
                    .expect("Failed to create Metal buffer for dequantized weight tensor")
            }
        });
        key
    }

    pub(super) fn ensure_quant_weight_cached(&self, key: usize, data: &[u8]) {
        let mut cache = self.quant_weight_cache.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| create_mmap_weight_buffer_from_bytes(&self.device, data));
    }

    pub(super) fn safe_batch_dequant_matvec_dense(
        &self,
        ops: &[(&[u8], GgmlType, usize)],
        x: &[f32],
        k: usize,
        outputs: &mut [&mut [f32]],
    ) {
        debug_assert_eq!(ops.len(), outputs.len());
        if ops.is_empty() {
            return;
        }

        let input_guard = self.prepare_input(x);
        let buf_x = input_guard.0.as_ref().unwrap();

        let mut weight_keys = Vec::with_capacity(ops.len());
        for (a_quant, dtype, m) in ops {
            weight_keys.push(self.ensure_dense_weight_buffer(a_quant, *dtype, *m, k));
        }

        let mut output_bufs: Vec<MetalBuffer> = Vec::with_capacity(ops.len());
        for (_, _, m) in ops {
            output_bufs.push(
                MetalBuffer::new(self.device.device(), m * std::mem::size_of::<f32>())
                    .expect("Failed to allocate output Metal buffer for safe batch matvec"),
            );
        }

        let cache = self.weight_cache.lock().unwrap();
        self.device
            .execute_sync(|encoder| {
                for (i, (_, _, m)) in ops.iter().enumerate() {
                    let buf_a = cache.get(&weight_keys[i]).unwrap();
                    self.matmul_kernels.encode_matvec(
                        encoder,
                        buf_a,
                        buf_x,
                        &output_bufs[i],
                        *m as u32,
                        k as u32,
                    );
                }
                Ok(())
            })
            .expect("Metal safe batch matvec dispatch failed");
        drop(cache);

        for (i, (_, _, m)) in ops.iter().enumerate() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    output_bufs[i].contents().as_ptr() as *const f32,
                    outputs[i].as_mut_ptr(),
                    *m,
                );
            }
        }
    }

    fn qwen35_recurrent_slot_buffer_key(
        layer_idx: usize,
        slot_idx: usize,
        conv_state_stride: usize,
        recurrent_state_stride: usize,
    ) -> Qwen35RecurrentSlotBufferKey {
        Qwen35RecurrentSlotBufferKey {
            layer_idx,
            slot_idx,
            conv_state_stride,
            recurrent_state_stride,
        }
    }

    fn ensure_qwen35_recurrent_slot_buffers(
        &self,
        layer_idx: usize,
        slot_idx: usize,
        conv_state_stride: usize,
        recurrent_state_stride: usize,
    ) -> std::sync::MutexGuard<'_, FxHashMap<Qwen35RecurrentSlotBufferKey, Qwen35MetalSlotBuffers>>
    {
        let key = Self::qwen35_recurrent_slot_buffer_key(
            layer_idx,
            slot_idx,
            conv_state_stride,
            recurrent_state_stride,
        );
        let mut cache = self.ops.qwen35_recurrent_slot_buffers.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            Qwen35MetalSlotBuffers::new(&self.device, conv_state_stride, recurrent_state_stride)
                .expect("Failed to allocate qwen35 recurrent Metal slot buffers")
        });
        cache
    }

    fn qwen35_recurrent_scratch_buffer_key(
        tokens_per_slot: usize,
        cfg: crate::compute::gdn::Qwen35RecurrentConfig,
    ) -> Qwen35RecurrentScratchBufferKey {
        Qwen35RecurrentScratchBufferKey {
            tokens_per_slot,
            conv_dim: cfg.conv_dim,
            time_step_rank: cfg.time_step_rank,
            state_size: cfg.state_size,
        }
    }

    fn ensure_qwen35_recurrent_scratch_buffers(
        &self,
        tokens_per_slot: usize,
        cfg: crate::compute::gdn::Qwen35RecurrentConfig,
    ) -> std::sync::MutexGuard<
        '_,
        FxHashMap<Qwen35RecurrentScratchBufferKey, Qwen35MetalRecurrentScratchBuffers>,
    > {
        let key = Self::qwen35_recurrent_scratch_buffer_key(tokens_per_slot, cfg);
        let mut cache = self.qwen35_recurrent_scratch_buffers.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            Qwen35MetalRecurrentScratchBuffers::new(&self.device, key)
                .expect("Failed to allocate qwen35 recurrent Metal scratch buffers")
        });
        cache
    }

    fn get_qwen35_conv_kernel_buffer(
        &self,
        kernel: &[f32],
    ) -> std::sync::MutexGuard<'_, FxHashMap<usize, MetalBuffer>> {
        let key = kernel.as_ptr() as usize;
        let mut cache = self.qwen35_conv_kernel_cache.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            MetalBuffer::from_slice(self.device.device(), kernel)
                .expect("Failed to create Metal buffer for qwen35 conv kernel")
        });
        cache
    }

    fn qwen35_sync_slot_buffers_from_kv(
        qwen_kv: &crate::kv::Qwen3_5Kv,
        layer_idx: usize,
        slot_idx: usize,
        conv_state_stride: usize,
        recurrent_state_stride: usize,
        slot_buffers: &mut Qwen35MetalSlotBuffers,
    ) {
        let kv_identity = qwen_kv as *const crate::kv::Qwen3_5Kv as usize;
        if slot_buffers.source_kv_identity != Some(kv_identity) {
            slot_buffers.conv_synced_generation = None;
            slot_buffers.recurrent_synced_generation = None;
        }
        let conv_generation = qwen_kv.conv_state_generation(slot_idx, layer_idx);
        // Only copy conv_state CPU→GPU if GPU doesn't have the latest.
        if !qwen_kv.conv_state_cpu_stale(slot_idx, layer_idx) {
            // CPU has the latest version — copy to GPU.
            if slot_buffers.conv_synced_generation != Some(conv_generation) {
                unsafe {
                    slot_buffers.conv_state.as_mut_slice::<f32>()[..conv_state_stride]
                        .copy_from_slice(qwen_kv.conv_state_for_slot(slot_idx, layer_idx));
                }
                slot_buffers.conv_synced_generation = Some(conv_generation);
            }
        } else if slot_buffers.conv_synced_generation != Some(conv_generation) {
            // Backend-owned but GPU slot buffer has a different generation.
            // This can happen when GPU prefill sets backend-owned state and
            // then the slot/layer is reused. Accept the GPU version as
            // authoritative and update the tracked generation.
            tracing::debug!(
                slot_idx,
                layer_idx,
                gpu_gen = ?slot_buffers.conv_synced_generation,
                cpu_gen = conv_generation,
                "qwen35 conv state generation mismatch — accepting GPU version"
            );
            slot_buffers.conv_synced_generation = Some(conv_generation);
        }

        let recurrent_generation = qwen_kv.recurrent_state_generation(slot_idx, layer_idx);
        if !qwen_kv.recurrent_state_cpu_stale(slot_idx, layer_idx) {
            if slot_buffers.recurrent_synced_generation != Some(recurrent_generation) {
                unsafe {
                    slot_buffers.recurrent_state.as_mut_slice::<f32>()[..recurrent_state_stride]
                        .copy_from_slice(qwen_kv.recurrent_state_for_slot(slot_idx, layer_idx));
                }
                slot_buffers.recurrent_synced_generation = Some(recurrent_generation);
            }
        } else if slot_buffers.recurrent_synced_generation != Some(recurrent_generation) {
            tracing::debug!(
                slot_idx,
                layer_idx,
                gpu_gen = ?slot_buffers.recurrent_synced_generation,
                cpu_gen = recurrent_generation,
                "qwen35 recurrent state generation mismatch — accepting GPU version"
            );
            slot_buffers.recurrent_synced_generation = Some(recurrent_generation);
        }
        slot_buffers.source_kv_identity = Some(kv_identity);
    }

    fn sync_qwen35_slot_from_backend_if_needed(
        &self,
        qwen_kv: &mut crate::kv::Qwen3_5Kv,
        layer_idx: usize,
        slot_idx: usize,
    ) {
        let recurrent_stale = qwen_kv.recurrent_state_cpu_stale(slot_idx, layer_idx);
        let conv_stale = qwen_kv.conv_state_cpu_stale(slot_idx, layer_idx);
        if !recurrent_stale && !conv_stale {
            return;
        }

        let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
        let recurrent_state_stride = qwen_kv.recurrent_state_len();
        let slot_key = Self::qwen35_recurrent_slot_buffer_key(
            layer_idx,
            slot_idx,
            conv_state_stride,
            recurrent_state_stride,
        );
        let slot_cache = self.ops.qwen35_recurrent_slot_buffers.lock().unwrap();
        let slot_buffers = slot_cache.get(&slot_key).unwrap_or_else(|| {
            panic!(
                "qwen35 state for slot {slot_idx} layer {layer_idx} is backend-owned but Metal slot buffer is missing"
            )
        });
        let kv_identity = qwen_kv as *const crate::kv::Qwen3_5Kv as usize;
        assert_eq!(
            slot_buffers.source_kv_identity,
            Some(kv_identity),
            "qwen35 state for slot {slot_idx} layer {layer_idx} is backend-owned by a different KV instance"
        );
        if recurrent_stale {
            let generation = qwen_kv.recurrent_state_generation(slot_idx, layer_idx);
            assert_eq!(
                slot_buffers.recurrent_synced_generation,
                Some(generation),
                "qwen35 recurrent state for slot {slot_idx} layer {layer_idx} is backend-owned but Metal slot buffer generation does not match"
            );
            let recurrent_state = unsafe {
                &slot_buffers.recurrent_state.as_slice::<f32>()[..recurrent_state_stride]
            };
            qwen_kv.sync_recurrent_state_from_backend(
                slot_idx,
                layer_idx,
                recurrent_state,
                generation,
            );
        }
        if conv_stale {
            let generation = qwen_kv.conv_state_generation(slot_idx, layer_idx);
            assert_eq!(
                slot_buffers.conv_synced_generation,
                Some(generation),
                "qwen35 conv state for slot {slot_idx} layer {layer_idx} is backend-owned but Metal slot buffer generation does not match"
            );
            let conv_state =
                unsafe { &slot_buffers.conv_state.as_slice::<f32>()[..conv_state_stride] };
            qwen_kv.sync_conv_state_from_backend(slot_idx, layer_idx, conv_state, generation);
        }
    }

    pub(super) fn sync_qwen35_slots_from_backend_if_needed(
        &self,
        qwen_kv: &mut crate::kv::Qwen3_5Kv,
        layer_idx: usize,
        slot_indices: &[usize],
    ) {
        for &slot_idx in slot_indices {
            self.sync_qwen35_slot_from_backend_if_needed(qwen_kv, layer_idx, slot_idx);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen35_recurrent_sequence_with_slot_buffers(
        &self,
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
        if tokens_per_slot == 1 || cfg.conv_cache_len > 8 {
            qwen35_recurrent_sequence_with_backend(
                self,
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
            return;
        }

        assert!(
            tokens_per_slot > 0,
            "qwen35 recurrent sequence requires tokens_per_slot > 0"
        );
        let slot_count = state_batch.slot_count();
        let total_tokens = slot_count * tokens_per_slot;
        let value_dim = cfg.value_dim();
        let conv_state_stride = state_batch.conv_state_stride();
        let recurrent_state_stride = state_batch.recurrent_state_stride();
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
            conv_state_stride,
            cfg.conv_cache_len * cfg.conv_dim,
            "qwen35 recurrent conv state stride mismatch"
        );
        assert_eq!(
            recurrent_state_stride,
            cfg.time_step_rank * cfg.state_size * cfg.state_size,
            "qwen35 recurrent state stride mismatch"
        );

        crate::compute::gdn::prepare_alpha_beta(alpha_batch, beta_batch, dt_bias, a);
        let kernel_key = conv_kernel.as_ptr() as usize;
        let kernel_cache = self.get_qwen35_conv_kernel_buffer(conv_kernel);
        let buf_kernel = kernel_cache
            .get(&kernel_key)
            .expect("qwen35 conv kernel Metal buffer missing after allocation");
        let scratch_key = Self::qwen35_recurrent_scratch_buffer_key(tokens_per_slot, cfg);
        let mut scratch_cache = self.ensure_qwen35_recurrent_scratch_buffers(tokens_per_slot, cfg);
        let scratch_buffers = scratch_cache
            .get_mut(&scratch_key)
            .expect("qwen35 recurrent Metal scratch buffers missing after allocation");

        for batch_idx in 0..slot_count {
            let token_start = batch_idx * tokens_per_slot;
            let token_end = token_start + tokens_per_slot;
            let qkv_start = token_start * cfg.conv_dim;
            let qkv_end = token_end * cfg.conv_dim;
            let gate_start = token_start * cfg.time_step_rank;
            let gate_end = token_end * cfg.time_step_rank;
            let out_start = token_start * value_dim;
            let out_end = token_end * value_dim;
            let (conv_state_src, recurrent_state_src) =
                state_batch.recurrent_buffers_for_slot_mut(batch_idx);
            let conv_state_buf = unsafe {
                MetalBuffer::from_mut_slice_no_copy(self.device.device(), conv_state_src)
            }
            .expect("Failed to alias qwen35 conv state batch into Metal buffer");
            let recurrent_state_buf = unsafe {
                MetalBuffer::from_mut_slice_no_copy(self.device.device(), recurrent_state_src)
            }
            .expect("Failed to alias qwen35 recurrent state batch into Metal buffer");

            unsafe {
                scratch_buffers.input.as_mut_slice::<f32>()[..qkv_end - qkv_start]
                    .copy_from_slice(&qkv_batch[qkv_start..qkv_end]);
            }
            self.qwen35_causal_conv_sequence_sync(
                &scratch_buffers.input,
                buf_kernel,
                &conv_state_buf,
                &scratch_buffers.conv_out,
                tokens_per_slot as u32,
                cfg.conv_cache_len as u32,
                cfg.conv_dim as u32,
            )
            .expect("Metal qwen35 recurrent causal conv dispatch failed");

            if tokens_per_slot == 1 {
                let conv_out = unsafe { scratch_buffers.conv_out.as_slice::<f32>() };
                unsafe {
                    let q_dst = &mut scratch_buffers.q.as_mut_slice::<f32>()[..value_dim];
                    let k_dst = &mut scratch_buffers.k.as_mut_slice::<f32>()[..value_dim];
                    let v_dst = &mut scratch_buffers.v.as_mut_slice::<f32>()[..value_dim];
                    prepare_single_token_gdn_qkv(
                        &conv_out[..cfg.conv_dim],
                        q_dst,
                        k_dst,
                        v_dst,
                        cfg.group_count,
                        cfg.time_step_rank,
                        cfg.state_size,
                        cfg.rms_norm_eps,
                    );
                    scratch_buffers.gate.as_mut_slice::<f32>()[..cfg.time_step_rank]
                        .copy_from_slice(&alpha_batch[gate_start..gate_end]);
                    scratch_buffers.beta.as_mut_slice::<f32>()[..cfg.time_step_rank]
                        .copy_from_slice(&beta_batch[gate_start..gate_end]);
                }
            } else {
                self.qwen35_prepare_multi_token_qkv_with_fallback(
                    &scratch_buffers.conv_out,
                    &alpha_batch[gate_start..gate_end],
                    &beta_batch[gate_start..gate_end],
                    &mut scratch_buffers.q,
                    &mut scratch_buffers.k,
                    &mut scratch_buffers.v,
                    &mut scratch_buffers.gate,
                    &mut scratch_buffers.beta,
                    tokens_per_slot,
                    value_dim,
                    cfg.group_count,
                    cfg.time_step_rank,
                    cfg.state_size,
                    cfg.rms_norm_eps,
                )
                .expect("Metal qwen35 recurrent batch pack dispatch failed");
            }
            self.qwen35_gated_delta_sequence_sync(
                &scratch_buffers.q,
                &scratch_buffers.k,
                &scratch_buffers.v,
                &scratch_buffers.gate,
                &scratch_buffers.beta,
                &recurrent_state_buf,
                &scratch_buffers.output,
                tokens_per_slot as u32,
                cfg.time_step_rank as u32,
                cfg.state_size as u32,
            )
            .expect("Metal qwen35 recurrent gated-delta dispatch failed");

            let out_bhsv = unsafe { scratch_buffers.output.as_slice::<f32>() };
            if tokens_per_slot == 1 {
                output_batch[out_start..out_end].copy_from_slice(&out_bhsv[..value_dim]);
            } else {
                let output_slice = &mut output_batch[out_start..out_end];
                if let Ok(output_buf) = unsafe {
                    MetalBuffer::from_mut_slice_no_copy(self.device.device(), output_slice)
                } {
                    self.gdn_kernels
                        .unpack_bhsk_to_token_major(
                            &self.device,
                            &scratch_buffers.output,
                            &output_buf,
                            tokens_per_slot as u32,
                            cfg.time_step_rank as u32,
                            cfg.state_size as u32,
                        )
                        .expect("Metal qwen35 recurrent batch unpack dispatch failed");
                } else {
                    unpack_bhsk_to_token_major(
                        out_bhsv,
                        output_slice,
                        tokens_per_slot,
                        cfg.time_step_rank,
                        cfg.state_size,
                    );
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen35_recurrent_sequence_for_kv_with_slot_buffers(
        &self,
        qkv_batch: &[f32],
        beta_batch: &mut [f32],
        alpha_batch: &mut [f32],
        dt_bias: &[f32],
        a: &[f32],
        conv_kernel: &[f32],
        qwen_kv: &mut crate::kv::Qwen3_5Kv,
        layer_idx: usize,
        slot_indices: &[usize],
        output_batch: &mut [f32],
        tokens_per_slot: usize,
        cfg: crate::compute::gdn::Qwen35RecurrentConfig,
    ) {
        assert!(
            tokens_per_slot > 0,
            "qwen35 recurrent sequence requires tokens_per_slot > 0"
        );
        let slot_count = slot_indices.len();
        let total_tokens = slot_count * tokens_per_slot;
        let value_dim = cfg.value_dim();
        let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
        let recurrent_state_stride = qwen_kv.recurrent_state_len();
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

        crate::compute::gdn::prepare_alpha_beta(alpha_batch, beta_batch, dt_bias, a);
        let kernel_key = conv_kernel.as_ptr() as usize;
        let kernel_cache = self.get_qwen35_conv_kernel_buffer(conv_kernel);
        let buf_kernel = kernel_cache
            .get(&kernel_key)
            .expect("qwen35 conv kernel Metal buffer missing after allocation");
        let scratch_key = Self::qwen35_recurrent_scratch_buffer_key(tokens_per_slot, cfg);
        let mut scratch_cache = self.ensure_qwen35_recurrent_scratch_buffers(tokens_per_slot, cfg);
        let scratch_buffers = scratch_cache
            .get_mut(&scratch_key)
            .expect("qwen35 recurrent Metal scratch buffers missing after allocation");

        for (batch_idx, &slot_idx) in slot_indices.iter().enumerate() {
            let token_start = batch_idx * tokens_per_slot;
            let token_end = token_start + tokens_per_slot;
            let qkv_start = token_start * cfg.conv_dim;
            let qkv_end = token_end * cfg.conv_dim;
            let gate_start = token_start * cfg.time_step_rank;
            let gate_end = token_end * cfg.time_step_rank;
            let out_start = token_start * value_dim;
            let out_end = token_end * value_dim;

            if tokens_per_slot == 1 {
                // If GPU batch prefill left state in Qwen3_5Kv's GPU-resident
                // buffers, use those directly instead of the MetalOps slot
                // buffer cache (which is a separate, stale copy).
                if let Some((conv_buf, rec_buf)) =
                    qwen_kv.gpu_recurrent_buffers(slot_idx, layer_idx)
                {
                    // Debug: check if recurrent state is non-zero
                    if layer_idx == 0 && std::env::var("AX_DEBUG_MOE").is_ok() {
                        let rec_data = unsafe { rec_buf.as_slice::<f32>() };
                        let rec_sum: f32 = rec_data[..128.min(rec_data.len())]
                            .iter()
                            .map(|v| v.abs())
                            .sum();
                        let conv_data = unsafe { conv_buf.as_slice::<f32>() };
                        let conv_sum: f32 = conv_data[..128.min(conv_data.len())]
                            .iter()
                            .map(|v| v.abs())
                            .sum();
                        eprintln!(
                            "[DECODE STATE] layer={layer_idx} slot={slot_idx} rec_abs_sum={rec_sum:.4} conv_abs_sum={conv_sum:.4} rec[0..4]={:?} conv[0..4]={:?}",
                            &rec_data[..4.min(rec_data.len())],
                            &conv_data[..4.min(conv_data.len())]
                        );
                    }
                    // Copy input QKV to GPU scratch buffer.
                    unsafe {
                        scratch_buffers.input.as_mut_slice::<f32>()[..cfg.conv_dim]
                            .copy_from_slice(&qkv_batch[qkv_start..qkv_end]);
                    }
                    unsafe {
                        scratch_buffers.gate.as_mut_slice::<f32>()[..cfg.time_step_rank]
                            .copy_from_slice(&alpha_batch[gate_start..gate_end]);
                        scratch_buffers.beta.as_mut_slice::<f32>()[..cfg.time_step_rank]
                            .copy_from_slice(&beta_batch[gate_start..gate_end]);
                    }
                    self.device
                        .execute_sync(|encoder| {
                            self.gdn_kernels.encode_causal_conv_sequence(
                                encoder,
                                &scratch_buffers.input,
                                buf_kernel,
                                conv_buf,
                                &scratch_buffers.conv_out,
                                tokens_per_slot as u32,
                                cfg.conv_cache_len as u32,
                                cfg.conv_dim as u32,
                            );
                            self.gdn_kernels.encode_prepare_single_token_qkv(
                                encoder,
                                &scratch_buffers.conv_out,
                                &scratch_buffers.q,
                                &scratch_buffers.k,
                                &scratch_buffers.v,
                                cfg.group_count as u32,
                                cfg.time_step_rank as u32,
                                cfg.state_size as u32,
                                cfg.rms_norm_eps,
                            );
                            self.gdn_kernels.encode_gated_delta_sequence(
                                encoder,
                                &scratch_buffers.q,
                                &scratch_buffers.k,
                                &scratch_buffers.v,
                                &scratch_buffers.gate,
                                &scratch_buffers.beta,
                                rec_buf,
                                &scratch_buffers.output,
                                tokens_per_slot as u32,
                                cfg.time_step_rank as u32,
                                cfg.state_size as u32,
                            );
                            Ok(())
                        })
                        .expect("Metal qwen35 GPU-resident recurrent decode failed");
                    let out_bhsv = unsafe { scratch_buffers.output.as_slice::<f32>() };
                    output_batch[out_start..out_end].copy_from_slice(&out_bhsv[..value_dim]);
                    let conv_generation =
                        qwen_kv.note_backend_conv_state_update(slot_idx, layer_idx);
                    let backend_generation =
                        qwen_kv.note_backend_recurrent_state_update(slot_idx, layer_idx);
                    // Update MetalOps slot buffer cache to match, so subsequent
                    // decode tokens that may use the slot buffer path stay in sync.
                    if let Ok(mut slot_cache) = self.ops.qwen35_recurrent_slot_buffers.try_lock() {
                        let slot_key = Self::qwen35_recurrent_slot_buffer_key(
                            layer_idx,
                            slot_idx,
                            conv_state_stride,
                            recurrent_state_stride,
                        );
                        if let Some(slot_buffers) = slot_cache.get_mut(&slot_key) {
                            slot_buffers.conv_synced_generation = Some(conv_generation);
                            slot_buffers.recurrent_synced_generation = Some(backend_generation);
                        }
                    }
                    continue;
                }

                let mut slot_cache = self.ensure_qwen35_recurrent_slot_buffers(
                    layer_idx,
                    slot_idx,
                    conv_state_stride,
                    recurrent_state_stride,
                );
                let slot_key = Self::qwen35_recurrent_slot_buffer_key(
                    layer_idx,
                    slot_idx,
                    conv_state_stride,
                    recurrent_state_stride,
                );
                let slot_buffers = slot_cache
                    .get_mut(&slot_key)
                    .expect("qwen35 recurrent Metal slot buffers missing after allocation");

                // Sync state CPU→GPU only when GPU is stale (first token after
                // clear/restore/snapshot). On subsequent tokens both conv and
                // recurrent state are already resident on GPU from the previous
                // decode step.
                Self::qwen35_sync_slot_buffers_from_kv(
                    qwen_kv,
                    layer_idx,
                    slot_idx,
                    conv_state_stride,
                    recurrent_state_stride,
                    slot_buffers,
                );

                // GPU-resident single-token decode: conv1d + QKV prep + GDN
                // all run on GPU in one command buffer, eliminating CPU↔GPU
                // sync overhead for conv state (~4.2 MB per layer).

                // Copy input QKV to GPU scratch buffer.
                unsafe {
                    scratch_buffers.input.as_mut_slice::<f32>()[..cfg.conv_dim]
                        .copy_from_slice(&qkv_batch[qkv_start..qkv_end]);
                }

                // Copy gate/beta to GPU scratch buffers.
                unsafe {
                    scratch_buffers.gate.as_mut_slice::<f32>()[..cfg.time_step_rank]
                        .copy_from_slice(&alpha_batch[gate_start..gate_end]);
                    scratch_buffers.beta.as_mut_slice::<f32>()[..cfg.time_step_rank]
                        .copy_from_slice(&beta_batch[gate_start..gate_end]);
                }

                // Single CB: conv1d → QKV prep → GDN, all GPU-resident.
                self.device
                    .execute_sync(|encoder| {
                        // 1. GPU causal conv1d (updates conv_state in-place on GPU).
                        self.gdn_kernels.encode_causal_conv_sequence(
                            encoder,
                            &scratch_buffers.input,
                            buf_kernel,
                            &slot_buffers.conv_state,
                            &scratch_buffers.conv_out,
                            tokens_per_slot as u32,
                            cfg.conv_cache_len as u32,
                            cfg.conv_dim as u32,
                        );

                        // 2. GPU QKV preparation (norm + head split).
                        self.gdn_kernels.encode_prepare_single_token_qkv(
                            encoder,
                            &scratch_buffers.conv_out,
                            &scratch_buffers.q,
                            &scratch_buffers.k,
                            &scratch_buffers.v,
                            cfg.group_count as u32,
                            cfg.time_step_rank as u32,
                            cfg.state_size as u32,
                            cfg.rms_norm_eps,
                        );

                        // 3. GPU gated delta net (updates recurrent_state in-place).
                        self.gdn_kernels.encode_gated_delta_sequence(
                            encoder,
                            &scratch_buffers.q,
                            &scratch_buffers.k,
                            &scratch_buffers.v,
                            &scratch_buffers.gate,
                            &scratch_buffers.beta,
                            &slot_buffers.recurrent_state,
                            &scratch_buffers.output,
                            tokens_per_slot as u32,
                            cfg.time_step_rank as u32,
                            cfg.state_size as u32,
                        );
                        Ok(())
                    })
                    .expect("Metal qwen35 GPU-resident recurrent decode failed");

                // Read back only the output (16 KB), not conv/recurrent state.
                let out_bhsv = unsafe { scratch_buffers.output.as_slice::<f32>() };
                output_batch[out_start..out_end].copy_from_slice(&out_bhsv[..value_dim]);

                // Mark both conv and recurrent state as backend-owned
                // (GPU has the latest — no CPU copy needed until slot
                // clear/restore/snapshot).
                let conv_generation = qwen_kv.note_backend_conv_state_update(slot_idx, layer_idx);
                slot_buffers.conv_synced_generation = Some(conv_generation);
                let backend_generation =
                    qwen_kv.note_backend_recurrent_state_update(slot_idx, layer_idx);
                slot_buffers.recurrent_synced_generation = Some(backend_generation);
                slot_buffers.source_kv_identity =
                    Some(qwen_kv as *const crate::kv::Qwen3_5Kv as usize);
            } else {
                let conv_cpu_sync = (!qwen_kv.conv_state_cpu_stale(slot_idx, layer_idx))
                    .then(|| qwen_kv.conv_state_for_slot(slot_idx, layer_idx).to_vec());
                let recurrent_cpu_sync = (!qwen_kv.recurrent_state_cpu_stale(slot_idx, layer_idx))
                    .then(|| {
                        qwen_kv
                            .recurrent_state_for_slot(slot_idx, layer_idx)
                            .to_vec()
                    });
                if let Some((conv_buf, rec_buf)) =
                    qwen_kv.gpu_recurrent_buffers_mut(slot_idx, layer_idx)
                {
                    if let Some(conv_cpu_sync) = conv_cpu_sync.as_ref() {
                        unsafe {
                            conv_buf.as_mut_slice::<f32>()[..conv_state_stride]
                                .copy_from_slice(&conv_cpu_sync[..conv_state_stride]);
                        }
                    }
                    if let Some(recurrent_cpu_sync) = recurrent_cpu_sync.as_ref() {
                        unsafe {
                            rec_buf.as_mut_slice::<f32>()[..recurrent_state_stride]
                                .copy_from_slice(&recurrent_cpu_sync[..recurrent_state_stride]);
                        }
                    }
                    unsafe {
                        scratch_buffers.input.as_mut_slice::<f32>()[..qkv_end - qkv_start]
                            .copy_from_slice(&qkv_batch[qkv_start..qkv_end]);
                    }

                    let conv_t = Instant::now();
                    self.qwen35_causal_conv_sequence_sync(
                        &scratch_buffers.input,
                        buf_kernel,
                        conv_buf,
                        &scratch_buffers.conv_out,
                        tokens_per_slot as u32,
                        cfg.conv_cache_len as u32,
                        cfg.conv_dim as u32,
                    )
                    .expect("Metal qwen35 recurrent causal conv dispatch failed");
                    self.ops
                        .record_qwen35_recurrent_batch_conv(conv_t.elapsed());

                    let pack_t = Instant::now();
                    self.qwen35_prepare_multi_token_qkv_with_fallback(
                        &scratch_buffers.conv_out,
                        &alpha_batch[gate_start..gate_end],
                        &beta_batch[gate_start..gate_end],
                        &mut scratch_buffers.q,
                        &mut scratch_buffers.k,
                        &mut scratch_buffers.v,
                        &mut scratch_buffers.gate,
                        &mut scratch_buffers.beta,
                        tokens_per_slot,
                        value_dim,
                        cfg.group_count,
                        cfg.time_step_rank,
                        cfg.state_size,
                        cfg.rms_norm_eps,
                    )
                    .expect("Metal qwen35 recurrent batch pack dispatch failed");
                    self.ops
                        .record_qwen35_recurrent_batch_pack(pack_t.elapsed());

                    let gated_delta_t = Instant::now();
                    self.qwen35_gated_delta_sequence_sync(
                        &scratch_buffers.q,
                        &scratch_buffers.k,
                        &scratch_buffers.v,
                        &scratch_buffers.gate,
                        &scratch_buffers.beta,
                        rec_buf,
                        &scratch_buffers.output,
                        tokens_per_slot as u32,
                        cfg.time_step_rank as u32,
                        cfg.state_size as u32,
                    )
                    .expect("Metal qwen35 recurrent gated-delta dispatch failed");
                    self.ops
                        .record_qwen35_recurrent_batch_gated_delta(gated_delta_t.elapsed());

                    let out_bhsv = unsafe { scratch_buffers.output.as_slice::<f32>() };
                    let unpack_t = Instant::now();
                    let output_slice = &mut output_batch[out_start..out_end];
                    if let Ok(output_buf) = unsafe {
                        MetalBuffer::from_mut_slice_no_copy(self.device.device(), output_slice)
                    } {
                        self.gdn_kernels
                            .unpack_bhsk_to_token_major(
                                &self.device,
                                &scratch_buffers.output,
                                &output_buf,
                                tokens_per_slot as u32,
                                cfg.time_step_rank as u32,
                                cfg.state_size as u32,
                            )
                            .expect("Metal qwen35 recurrent batch unpack dispatch failed");
                    } else {
                        unpack_bhsk_to_token_major(
                            out_bhsv,
                            output_slice,
                            tokens_per_slot,
                            cfg.time_step_rank,
                            cfg.state_size,
                        );
                    }
                    self.ops
                        .record_qwen35_recurrent_batch_unpack(unpack_t.elapsed());

                    let conv_generation =
                        qwen_kv.note_backend_conv_state_update(slot_idx, layer_idx);
                    let recurrent_generation =
                        qwen_kv.note_backend_recurrent_state_update(slot_idx, layer_idx);
                    if let Ok(mut slot_cache) = self.ops.qwen35_recurrent_slot_buffers.try_lock() {
                        let slot_key = Self::qwen35_recurrent_slot_buffer_key(
                            layer_idx,
                            slot_idx,
                            conv_state_stride,
                            recurrent_state_stride,
                        );
                        if let Some(slot_buffers) = slot_cache.get_mut(&slot_key) {
                            slot_buffers.conv_synced_generation = Some(conv_generation);
                            slot_buffers.recurrent_synced_generation = Some(recurrent_generation);
                            slot_buffers.source_kv_identity =
                                Some(qwen_kv as *const crate::kv::Qwen3_5Kv as usize);
                        }
                    }
                    continue;
                }

                let mut slot_cache = self.ensure_qwen35_recurrent_slot_buffers(
                    layer_idx,
                    slot_idx,
                    conv_state_stride,
                    recurrent_state_stride,
                );
                let slot_key = Self::qwen35_recurrent_slot_buffer_key(
                    layer_idx,
                    slot_idx,
                    conv_state_stride,
                    recurrent_state_stride,
                );
                let slot_buffers = slot_cache
                    .get_mut(&slot_key)
                    .expect("qwen35 recurrent Metal slot buffers missing after allocation");

                // Keep the recurrent path device-primary for multi-token
                // prefill. Only reload CPU state when the host owns a fresher
                // version; otherwise keep using the cached backend buffers.
                Self::qwen35_sync_slot_buffers_from_kv(
                    qwen_kv,
                    layer_idx,
                    slot_idx,
                    conv_state_stride,
                    recurrent_state_stride,
                    slot_buffers,
                );
                unsafe {
                    scratch_buffers.input.as_mut_slice::<f32>()[..qkv_end - qkv_start]
                        .copy_from_slice(&qkv_batch[qkv_start..qkv_end]);
                }

                let conv_t = Instant::now();
                self.qwen35_causal_conv_sequence_sync(
                    &scratch_buffers.input,
                    buf_kernel,
                    &slot_buffers.conv_state,
                    &scratch_buffers.conv_out,
                    tokens_per_slot as u32,
                    cfg.conv_cache_len as u32,
                    cfg.conv_dim as u32,
                )
                .expect("Metal qwen35 recurrent causal conv dispatch failed");
                self.ops
                    .record_qwen35_recurrent_batch_conv(conv_t.elapsed());

                let pack_t = Instant::now();
                self.qwen35_prepare_multi_token_qkv_with_fallback(
                    &scratch_buffers.conv_out,
                    &alpha_batch[gate_start..gate_end],
                    &beta_batch[gate_start..gate_end],
                    &mut scratch_buffers.q,
                    &mut scratch_buffers.k,
                    &mut scratch_buffers.v,
                    &mut scratch_buffers.gate,
                    &mut scratch_buffers.beta,
                    tokens_per_slot,
                    value_dim,
                    cfg.group_count,
                    cfg.time_step_rank,
                    cfg.state_size,
                    cfg.rms_norm_eps,
                )
                .expect("Metal qwen35 recurrent batch pack dispatch failed");
                self.ops
                    .record_qwen35_recurrent_batch_pack(pack_t.elapsed());
                let gated_delta_t = Instant::now();
                self.qwen35_gated_delta_sequence_sync(
                    &scratch_buffers.q,
                    &scratch_buffers.k,
                    &scratch_buffers.v,
                    &scratch_buffers.gate,
                    &scratch_buffers.beta,
                    &slot_buffers.recurrent_state,
                    &scratch_buffers.output,
                    tokens_per_slot as u32,
                    cfg.time_step_rank as u32,
                    cfg.state_size as u32,
                )
                .expect("Metal qwen35 recurrent gated-delta dispatch failed");
                self.ops
                    .record_qwen35_recurrent_batch_gated_delta(gated_delta_t.elapsed());

                let out_bhsv = unsafe { scratch_buffers.output.as_slice::<f32>() };
                let unpack_t = Instant::now();
                let output_slice = &mut output_batch[out_start..out_end];
                if let Ok(output_buf) = unsafe {
                    MetalBuffer::from_mut_slice_no_copy(self.device.device(), output_slice)
                } {
                    self.gdn_kernels
                        .unpack_bhsk_to_token_major(
                            &self.device,
                            &scratch_buffers.output,
                            &output_buf,
                            tokens_per_slot as u32,
                            cfg.time_step_rank as u32,
                            cfg.state_size as u32,
                        )
                        .expect("Metal qwen35 recurrent batch unpack dispatch failed");
                } else {
                    unpack_bhsk_to_token_major(
                        out_bhsv,
                        output_slice,
                        tokens_per_slot,
                        cfg.time_step_rank,
                        cfg.state_size,
                    );
                }
                self.ops
                    .record_qwen35_recurrent_batch_unpack(unpack_t.elapsed());

                let conv_generation = qwen_kv.note_backend_conv_state_update(slot_idx, layer_idx);
                let recurrent_generation =
                    qwen_kv.note_backend_recurrent_state_update(slot_idx, layer_idx);
                slot_buffers.conv_synced_generation = Some(conv_generation);
                slot_buffers.recurrent_synced_generation = Some(recurrent_generation);
                slot_buffers.source_kv_identity =
                    Some(qwen_kv as *const crate::kv::Qwen3_5Kv as usize);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn qwen35_recurrent_sequence_with_backend(
    backend: &dyn Backend,
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

impl MetalOps {
    pub(crate) fn record_qwen35_recurrent_batch_conv(&self, elapsed: Duration) {
        self.qwen35_recurrent_batch_perf
            .conv_ns
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_pack(&self, elapsed: Duration) {
        self.qwen35_recurrent_batch_perf
            .pack_ns
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_gated_delta(&self, elapsed: Duration) {
        self.qwen35_recurrent_batch_perf
            .gated_delta_ns
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_unpack(&self, elapsed: Duration) {
        self.qwen35_recurrent_batch_perf
            .unpack_ns
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_handoff_gpu(&self, elapsed: Duration) {
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_ns
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_handoff(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_handoff_fused_tail(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_fused_tail_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_gpu_projection(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_gpu_projection_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_fast_path_eligible(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_fast_path_eligible_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_fast_reject_state_size(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_state_size_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_fast_reject_group_divisibility(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_group_divisibility_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_fast_reject_missing_batch_scratches(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_missing_batch_scratches_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_fast_reject_q_capacity(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_q_capacity_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_fast_reject_k_capacity(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_k_capacity_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_fast_reject_v_capacity(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_v_capacity_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_fast_reject_gate_capacity(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_gate_capacity_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_fast_reject_up_capacity(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_up_capacity_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_gpu_ssm_projection(&self) {
        self.qwen35_recurrent_batch_perf
            .gpu_ssm_projection_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    #[allow(dead_code)]
    pub(crate) fn record_qwen35_recurrent_batch_qkv_handoff_cpu_alias(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_cpu_alias_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    #[allow(dead_code)]
    pub(crate) fn record_qwen35_recurrent_batch_qkv_handoff_slot_buffer(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_slot_buffer_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    #[allow(dead_code)]
    pub(crate) fn record_qwen35_recurrent_batch_qkv_handoff_backend_carryover(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_backend_carryover_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    #[allow(dead_code)]
    pub(crate) fn record_qwen35_recurrent_batch_qkv_handoff_backend_zero_init(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_backend_zero_init_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    #[allow(dead_code)]
    pub(crate) fn record_qwen35_recurrent_batch_qkv_handoff_cpu_materialization(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_cpu_materialization_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_state_batch_backend_native(&self) {
        self.qwen35_recurrent_batch_perf
            .state_batch_backend_native_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_state_batch_cpu_direct(&self) {
        self.qwen35_recurrent_batch_perf
            .state_batch_cpu_direct_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_state_batch_cpu_direct_materialized_from_backend(
        &self,
    ) {
        self.qwen35_recurrent_batch_perf
            .state_batch_cpu_direct_materialized_from_backend_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_state_batch_cpu_gathered(&self) {
        self.qwen35_recurrent_batch_perf
            .state_batch_cpu_gathered_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_state_batch_cpu_gathered_materialized_from_backend(
        &self,
    ) {
        self.qwen35_recurrent_batch_perf
            .state_batch_cpu_gathered_materialized_from_backend_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn reset_qwen35_recurrent_batch_perf_counters(&self) {
        self.qwen35_recurrent_batch_perf
            .conv_ns
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .pack_ns
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .gated_delta_ns
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .unpack_ns
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_ns
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_fused_tail_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_gpu_projection_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_fast_path_eligible_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .gpu_ssm_projection_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_state_size_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_group_divisibility_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_missing_batch_scratches_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_q_capacity_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_k_capacity_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_v_capacity_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_gate_capacity_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_up_capacity_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_cpu_alias_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_slot_buffer_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_backend_carryover_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_backend_zero_init_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_cpu_materialization_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .state_batch_backend_native_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .state_batch_cpu_direct_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .state_batch_cpu_direct_materialized_from_backend_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .state_batch_cpu_gathered_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .state_batch_cpu_gathered_materialized_from_backend_layers
            .store(0, Ordering::Relaxed);
    }

    pub fn qwen35_recurrent_batch_perf_counters(&self) -> Qwen35RecurrentBatchPerfCounters {
        Qwen35RecurrentBatchPerfCounters {
            conv_ns: self
                .qwen35_recurrent_batch_perf
                .conv_ns
                .load(Ordering::Relaxed),
            pack_ns: self
                .qwen35_recurrent_batch_perf
                .pack_ns
                .load(Ordering::Relaxed),
            gated_delta_ns: self
                .qwen35_recurrent_batch_perf
                .gated_delta_ns
                .load(Ordering::Relaxed),
            unpack_ns: self
                .qwen35_recurrent_batch_perf
                .unpack_ns
                .load(Ordering::Relaxed),
            qkv_handoff_ns: self
                .qwen35_recurrent_batch_perf
                .qkv_handoff_ns
                .load(Ordering::Relaxed),
            qkv_handoff_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_handoff_layers
                .load(Ordering::Relaxed),
            qkv_handoff_fused_tail_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_handoff_fused_tail_layers
                .load(Ordering::Relaxed),
            qkv_gpu_projection_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_gpu_projection_layers
                .load(Ordering::Relaxed),
            qkv_fast_path_eligible_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_fast_path_eligible_layers
                .load(Ordering::Relaxed),
            gpu_ssm_projection_layers: self
                .qwen35_recurrent_batch_perf
                .gpu_ssm_projection_layers
                .load(Ordering::Relaxed),
            qkv_fast_reject_state_size_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_fast_reject_state_size_layers
                .load(Ordering::Relaxed),
            qkv_fast_reject_group_divisibility_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_fast_reject_group_divisibility_layers
                .load(Ordering::Relaxed),
            qkv_fast_reject_missing_batch_scratches_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_fast_reject_missing_batch_scratches_layers
                .load(Ordering::Relaxed),
            qkv_fast_reject_q_capacity_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_fast_reject_q_capacity_layers
                .load(Ordering::Relaxed),
            qkv_fast_reject_k_capacity_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_fast_reject_k_capacity_layers
                .load(Ordering::Relaxed),
            qkv_fast_reject_v_capacity_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_fast_reject_v_capacity_layers
                .load(Ordering::Relaxed),
            qkv_fast_reject_gate_capacity_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_fast_reject_gate_capacity_layers
                .load(Ordering::Relaxed),
            qkv_fast_reject_up_capacity_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_fast_reject_up_capacity_layers
                .load(Ordering::Relaxed),
            qkv_handoff_cpu_alias_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_handoff_cpu_alias_layers
                .load(Ordering::Relaxed),
            qkv_handoff_slot_buffer_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_handoff_slot_buffer_layers
                .load(Ordering::Relaxed),
            qkv_handoff_backend_carryover_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_handoff_backend_carryover_layers
                .load(Ordering::Relaxed),
            qkv_handoff_backend_zero_init_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_handoff_backend_zero_init_layers
                .load(Ordering::Relaxed),
            qkv_handoff_cpu_materialization_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_handoff_cpu_materialization_layers
                .load(Ordering::Relaxed),
            state_batch_backend_native_layers: self
                .qwen35_recurrent_batch_perf
                .state_batch_backend_native_layers
                .load(Ordering::Relaxed),
            state_batch_cpu_direct_layers: self
                .qwen35_recurrent_batch_perf
                .state_batch_cpu_direct_layers
                .load(Ordering::Relaxed),
            state_batch_cpu_direct_materialized_from_backend_layers: self
                .qwen35_recurrent_batch_perf
                .state_batch_cpu_direct_materialized_from_backend_layers
                .load(Ordering::Relaxed),
            state_batch_cpu_gathered_layers: self
                .qwen35_recurrent_batch_perf
                .state_batch_cpu_gathered_layers
                .load(Ordering::Relaxed),
            state_batch_cpu_gathered_materialized_from_backend_layers: self
                .qwen35_recurrent_batch_perf
                .state_batch_cpu_gathered_materialized_from_backend_layers
                .load(Ordering::Relaxed),
        }
    }

    pub(crate) fn with_qwen35_qkv_handoff_scratch<R>(
        &self,
        n_tokens: usize,
        conv_dim: usize,
        inner_size: usize,
        time_step_rank: usize,
        f: impl FnOnce(&mut Qwen35MetalQkvHandoffScratchBuffers) -> R,
    ) -> R {
        let key = Qwen35QkvHandoffScratchBufferKey {
            n_tokens,
            conv_dim,
            inner_size,
            time_step_rank,
        };
        let mut cache = self.qwen35_qkv_handoff_scratch_buffers.lock().unwrap();
        let scratch = cache.entry(key).or_insert_with(|| {
            Qwen35MetalQkvHandoffScratchBuffers::new(&self.device, key)
                .expect("Failed to allocate qwen35 recurrent handoff scratch buffers")
        });
        f(scratch)
    }

    pub(crate) fn with_qwen35_recurrent_projection_scratch<R>(
        &self,
        n_tokens: usize,
        conv_dim: usize,
        inner_size: usize,
        time_step_rank: usize,
        f: impl FnOnce(&mut Qwen35MetalRecurrentProjectionScratchBuffers) -> R,
    ) -> R {
        let key = Qwen35QkvHandoffScratchBufferKey {
            n_tokens,
            conv_dim,
            inner_size,
            time_step_rank,
        };
        let mut cache = self
            .qwen35_recurrent_projection_scratch_buffers
            .lock()
            .unwrap();
        let scratch = cache.entry(key).or_insert_with(|| {
            Qwen35MetalRecurrentProjectionScratchBuffers::new(&self.device, key)
                .expect("Failed to allocate qwen35 recurrent projection scratch buffers")
        });
        f(scratch)
    }

    pub(crate) fn with_qwen35_batch_projection_scratch<R>(
        &self,
        n_tokens: usize,
        output_dims: &[usize],
        f: impl FnOnce(&mut Qwen35MetalBatchProjectionScratchBuffers) -> R,
    ) -> R {
        let key = Qwen35BatchProjectionScratchBufferKey {
            n_tokens,
            output_dims: output_dims.to_vec(),
        };
        let mut cache = self.qwen35_batch_projection_scratch_buffers.lock().unwrap();
        let scratch = cache.entry(key.clone()).or_insert_with(|| {
            Qwen35MetalBatchProjectionScratchBuffers::new(&self.device, &key)
                .expect("Failed to allocate qwen35 batch projection scratch buffers")
        });
        f(scratch)
    }

    pub(crate) fn with_qwen35_batch_logits_scratch<R>(
        &self,
        hidden_len: usize,
        logits_len: usize,
        f: impl FnOnce(&mut Qwen35MetalBatchLogitsScratchBuffers) -> R,
    ) -> R {
        let key = Qwen35BatchLogitsScratchBufferKey {
            hidden_len,
            logits_len,
        };
        let mut cache = self.qwen35_batch_logits_scratch_buffers.lock().unwrap();
        let scratch = cache.entry(key).or_insert_with(|| {
            Qwen35MetalBatchLogitsScratchBuffers::new(&self.device, key)
                .expect("Failed to allocate qwen35 batch logits scratch buffers")
        });
        f(scratch)
    }

    pub(crate) fn sync_qwen35_slot_buffers_from_kv(
        &self,
        qwen_kv: &crate::kv::Qwen3_5Kv,
        layer_idx: usize,
        slot_idx: usize,
    ) -> Qwen35SlotBufferSyncOutcome {
        // Fast path: GPU-resident buffers live in Qwen3_5Kv directly — no sync needed.
        if qwen_kv.has_gpu_recurrent_state() {
            let mut outcome = Qwen35SlotBufferSyncOutcome::default();
            outcome.note_backend_carryover();
            return outcome;
        }
        let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
        let recurrent_state_stride = qwen_kv.recurrent_state_len();
        let slot_key = Qwen35RecurrentSlotBufferKey {
            layer_idx,
            slot_idx,
            conv_state_stride,
            recurrent_state_stride,
        };
        let recurrent_generation = qwen_kv.recurrent_state_generation(slot_idx, layer_idx);
        let mut outcome = Qwen35SlotBufferSyncOutcome::default();
        let mut slot_cache = self.qwen35_recurrent_slot_buffers.lock().unwrap();
        let slot_buffers = slot_cache.entry(slot_key).or_insert_with(|| {
            Qwen35MetalSlotBuffers::new(&self.device, conv_state_stride, recurrent_state_stride)
                .expect("Failed to allocate qwen35 recurrent Metal slot buffers")
        });
        let kv_identity = qwen_kv as *const crate::kv::Qwen3_5Kv as usize;
        if slot_buffers.source_kv_identity != Some(kv_identity) {
            slot_buffers.conv_synced_generation = None;
            slot_buffers.recurrent_synced_generation = None;
        }
        let conv_generation = qwen_kv.conv_state_generation(slot_idx, layer_idx);
        if !qwen_kv.conv_state_cpu_stale(slot_idx, layer_idx) {
            if slot_buffers.conv_synced_generation != Some(conv_generation) {
                if qwen_kv.conv_state_pristine_zero(slot_idx, layer_idx) {
                    unsafe {
                        slot_buffers.conv_state.as_mut_slice::<f32>()[..conv_state_stride]
                            .fill(0.0);
                    }
                    outcome.note_backend_zero_init();
                } else {
                    unsafe {
                        slot_buffers.conv_state.as_mut_slice::<f32>()[..conv_state_stride]
                            .copy_from_slice(qwen_kv.conv_state_for_slot(slot_idx, layer_idx));
                    }
                    outcome.note_cpu_materialization();
                }
                slot_buffers.conv_synced_generation = Some(conv_generation);
            }
        } else if slot_buffers.conv_synced_generation != Some(conv_generation) {
            // Backend-owned but GPU slot buffer has a different generation.
            // This can happen when GPU prefill sets backend-owned state and
            // then the slot/layer is reused. Accept the GPU version as
            // authoritative and update the tracked generation.
            tracing::debug!(
                slot_idx,
                layer_idx,
                gpu_gen = ?slot_buffers.conv_synced_generation,
                cpu_gen = conv_generation,
                "qwen35 conv state generation mismatch — accepting GPU version"
            );
            slot_buffers.conv_synced_generation = Some(conv_generation);
        } else {
            outcome.note_backend_carryover();
        }

        if !qwen_kv.recurrent_state_cpu_stale(slot_idx, layer_idx) {
            if slot_buffers.recurrent_synced_generation != Some(recurrent_generation) {
                if qwen_kv.recurrent_state_pristine_zero(slot_idx, layer_idx) {
                    unsafe {
                        slot_buffers.recurrent_state.as_mut_slice::<f32>()
                            [..recurrent_state_stride]
                            .fill(0.0);
                    }
                    outcome.note_backend_zero_init();
                } else {
                    unsafe {
                        slot_buffers.recurrent_state.as_mut_slice::<f32>()
                            [..recurrent_state_stride]
                            .copy_from_slice(qwen_kv.recurrent_state_for_slot(slot_idx, layer_idx));
                    }
                    outcome.note_cpu_materialization();
                }
                slot_buffers.recurrent_synced_generation = Some(recurrent_generation);
            }
        } else if slot_buffers.recurrent_synced_generation != Some(recurrent_generation) {
            tracing::debug!(
                slot_idx,
                layer_idx,
                gpu_gen = ?slot_buffers.recurrent_synced_generation,
                cpu_gen = recurrent_generation,
                "qwen35 recurrent state generation mismatch — accepting GPU version"
            );
            slot_buffers.recurrent_synced_generation = Some(recurrent_generation);
        } else {
            outcome.note_backend_carryover();
        }
        slot_buffers.source_kv_identity = Some(kv_identity);
        outcome
    }

    /// Provide mutable access to GPU slot buffers for a recurrent layer.
    ///
    /// When `qwen_kv` is provided and has GPU-resident state, uses those
    /// buffers directly (no mutex, no cache lookup). Otherwise falls back
    /// to the internal slot buffer cache.
    pub(crate) fn with_qwen35_recurrent_slot_buffer_for_kv<R>(
        &self,
        qwen_kv: &crate::kv::Qwen3_5Kv,
        layer_idx: usize,
        slot_idx: usize,
        conv_state_stride: usize,
        recurrent_state_stride: usize,
        f: impl FnOnce(&mut Qwen35MetalSlotBuffers) -> R,
    ) -> R {
        if let Some((conv_buf, rec_buf)) = qwen_kv.gpu_recurrent_buffers(slot_idx, layer_idx) {
            let mut wrapper = Qwen35MetalSlotBuffers {
                conv_state: conv_buf.clone(),
                recurrent_state: rec_buf.clone(),
                conv_synced_generation: None,
                recurrent_synced_generation: None,
                source_kv_identity: None,
            };
            return f(&mut wrapper);
        }
        self.with_qwen35_recurrent_slot_buffer(
            layer_idx,
            slot_idx,
            conv_state_stride,
            recurrent_state_stride,
            f,
        )
    }

    pub(crate) fn with_qwen35_recurrent_slot_buffer<R>(
        &self,
        layer_idx: usize,
        slot_idx: usize,
        conv_state_stride: usize,
        recurrent_state_stride: usize,
        f: impl FnOnce(&mut Qwen35MetalSlotBuffers) -> R,
    ) -> R {
        let key = Qwen35RecurrentSlotBufferKey {
            layer_idx,
            slot_idx,
            conv_state_stride,
            recurrent_state_stride,
        };
        let mut cache = self.qwen35_recurrent_slot_buffers.lock().unwrap();
        let slot_buffers = cache.entry(key).or_insert_with(|| {
            Qwen35MetalSlotBuffers::new(&self.device, conv_state_stride, recurrent_state_stride)
                .expect("Failed to allocate qwen35 recurrent Metal slot buffers")
        });
        f(slot_buffers)
    }

    fn qwen35_recurrent_scratch_buffer_key_for_ops(
        tokens_per_slot: usize,
        cfg: crate::compute::gdn::Qwen35RecurrentConfig,
    ) -> Qwen35RecurrentScratchBufferKey {
        Qwen35RecurrentScratchBufferKey {
            tokens_per_slot,
            conv_dim: cfg.conv_dim,
            time_step_rank: cfg.time_step_rank,
            state_size: cfg.state_size,
        }
    }

    fn ensure_qwen35_recurrent_scratch_buffers_for_ops(
        &self,
        tokens_per_slot: usize,
        cfg: crate::compute::gdn::Qwen35RecurrentConfig,
    ) -> std::sync::MutexGuard<
        '_,
        FxHashMap<Qwen35RecurrentScratchBufferKey, Qwen35MetalRecurrentScratchBuffers>,
    > {
        let key = Self::qwen35_recurrent_scratch_buffer_key_for_ops(tokens_per_slot, cfg);
        let mut cache = self.qwen35_recurrent_scratch_buffers.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            Qwen35MetalRecurrentScratchBuffers::new(&self.device, key)
                .expect("Failed to allocate qwen35 recurrent Metal scratch buffers")
        });
        cache
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_qwen35_single_token_recurrent_projected(
        &self,
        encoder: &ax_engine_metal::MetalEncoder,
        qkv_projected: &MetalBuffer,
        alpha_prepared: &MetalBuffer,
        beta_prepared: &MetalBuffer,
        conv_kernel: &MetalBuffer,
        slot_buffers: &Qwen35MetalSlotBuffers,
        output: &MetalBuffer,
        cfg: crate::compute::gdn::Qwen35RecurrentConfig,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            cfg.conv_cache_len <= 8,
            "qwen35 single-token recurrent decode requires conv_cache_len <= 8, got {}",
            cfg.conv_cache_len
        );
        let scratch_key = Self::qwen35_recurrent_scratch_buffer_key_for_ops(1, cfg);
        let mut scratch_cache = self.ensure_qwen35_recurrent_scratch_buffers_for_ops(1, cfg);
        let scratch_buffers = scratch_cache
            .get_mut(&scratch_key)
            .expect("qwen35 recurrent Metal scratch buffers missing after allocation");

        self.gdn.encode_causal_conv_sequence(
            encoder,
            qkv_projected,
            conv_kernel,
            &slot_buffers.conv_state,
            &scratch_buffers.conv_out,
            1,
            cfg.conv_cache_len as u32,
            cfg.conv_dim as u32,
        );
        self.gdn.encode_prepare_single_token_qkv(
            encoder,
            &scratch_buffers.conv_out,
            &scratch_buffers.q,
            &scratch_buffers.k,
            &scratch_buffers.v,
            cfg.group_count as u32,
            cfg.time_step_rank as u32,
            cfg.state_size as u32,
            cfg.rms_norm_eps,
        );
        self.gdn.encode_gated_delta_sequence(
            encoder,
            &scratch_buffers.q,
            &scratch_buffers.k,
            &scratch_buffers.v,
            alpha_prepared,
            beta_prepared,
            &slot_buffers.recurrent_state,
            output,
            1,
            cfg.time_step_rank as u32,
            cfg.state_size as u32,
        );
        Ok(())
    }

    /// Materialize GPU-resident recurrent state for a slot/layer back to CPU.
    ///
    /// If the GPU holds the latest version of conv or recurrent state (backend-owned),
    /// copies it back to the Qwen3_5Kv CPU buffers so CPU-side decode can proceed.
    /// No-op if state is already CPU-fresh.
    #[allow(dead_code)]
    pub(crate) fn materialize_qwen35_slot_state_to_cpu(
        &self,
        qwen_kv: &mut crate::kv::Qwen3_5Kv,
        layer_idx: usize,
        slot_idx: usize,
        conv_state_stride: usize,
        recurrent_state_stride: usize,
    ) {
        let key = Qwen35RecurrentSlotBufferKey {
            layer_idx,
            slot_idx,
            conv_state_stride,
            recurrent_state_stride,
        };
        let cache = self.qwen35_recurrent_slot_buffers.lock().unwrap();
        let Some(slot_buffers) = cache.get(&key) else {
            return; // No GPU buffers allocated for this slot/layer
        };

        if qwen_kv.conv_state_cpu_stale(slot_idx, layer_idx)
            && let Some(generation) = slot_buffers.conv_synced_generation
        {
            let conv_data =
                unsafe { &slot_buffers.conv_state.as_slice::<f32>()[..conv_state_stride] };
            qwen_kv.sync_conv_state_from_backend(slot_idx, layer_idx, conv_data, generation);
        }
        if qwen_kv.recurrent_state_cpu_stale(slot_idx, layer_idx)
            && let Some(generation) = slot_buffers.recurrent_synced_generation
        {
            let rec_data = unsafe {
                &slot_buffers.recurrent_state.as_slice::<f32>()[..recurrent_state_stride]
            };
            qwen_kv.sync_recurrent_state_from_backend(slot_idx, layer_idx, rec_data, generation);
        }
    }
}
