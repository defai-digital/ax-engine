/// LLaMA model state for inference.
///
/// Internally delegates to an architecture-specific `ForwardPass` implementation.
/// For "llama" architecture, this uses `LlamaForward`. Other architectures
/// (Qwen3, Gemma3) use their own implementations selected via the arch registry.
pub struct LlamaModel {
    pub config: ModelConfig,
    attn_params: AttentionParams,
    backend: Box<dyn Backend>,
    forward: Box<dyn ForwardPass>,
}

impl LlamaModel {
    fn forward_context(&self) -> ForwardContext<'_> {
        ForwardContext {
            config: &self.config,
            attn_params: &self.attn_params,
            backend: &*self.backend,
        }
    }

    /// Create a new LLaMA model with CPU backend (default).
    pub fn new(config: ModelConfig) -> anyhow::Result<Self> {
        Self::with_backend(config, Box::new(CpuBackend))
    }

    /// Create a new LLaMA model with a specific compute backend.
    ///
    /// The forward pass implementation is selected based on `config.architecture`
    /// via the architecture registry.
    pub fn with_backend(config: ModelConfig, backend: Box<dyn Backend>) -> anyhow::Result<Self> {
        let forward = crate::model::arch_registry::forward_for_arch_with_config(
            &config.architecture,
            &config,
        )?;

        if let Err(e) = forward.validate_config(&config) {
            tracing::warn!(
                arch = config.architecture,
                "Model config validation warning: {e}"
            );
        }

        let attn_params = AttentionParams::new(
            config.n_heads as usize,
            config.n_kv_heads as usize,
            config.head_dim as usize,
        );
        Ok(Self {
            config,
            attn_params,
            backend,
            forward,
        })
    }

    fn prefill_plan_has_q8_weights(&self, weights: &WeightStore) -> anyhow::Result<bool> {
        for layer in 0..self.config.n_layers as usize {
            for suffix in [
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ] {
                let name = format!("blk.{layer}.{suffix}");
                let (_, dtype) = weights.raw_with_dtype(&name)?;
                if dtype == GgmlType::Q8_0 {
                    return Ok(true);
                }
            }
        }

        let lm_weight_name = if weights.has("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let (_, lm_head_dtype) = weights.raw_with_dtype(lm_weight_name)?;
        Ok(lm_head_dtype == GgmlType::Q8_0)
    }

    fn prefill_attention_route(
        &self,
        plan: GpuBatchPrefillExecutionPlan,
        base_seq_len: usize,
        n_tokens: usize,
        head_dim: u32,
        sliding_window: u32,
    ) -> String {
        match plan.attention {
            PrefillAttentionPlan::BatchLocalF16OutHd128 => {
                "mistral_f16out_hd128/profile_preferred".to_string()
            }
            PrefillAttentionPlan::BatchLocal => {
                let selection = plan
                    .attention_dispatch
                    .prefill_local_candidate_selection(n_tokens as u32, head_dim);
                format!("{}/{}", selection.label(), selection.stability.label())
            }
            PrefillAttentionPlan::Cached => {
                let selection = plan.attention_dispatch.prefill_cached_candidate_selection(
                    plan.kv_f16,
                    n_tokens as u32,
                    head_dim,
                    base_seq_len as u32,
                    sliding_window,
                );
                format!("{}/{}", selection.label(), selection.stability.label())
            }
        }
    }

    /// Create a KV cache sized for this model.
    ///
    /// v2: returns `ModelKv` via the backend-side planner.
    /// Paged KV remains deferred to v2.1.
    pub fn create_model_kv(&self) -> ModelKv {
        self.kv_plan().build(self.backend.as_ref())
    }

    /// Create a KV cache matched to the active decode support of the loaded weights.
    ///
    /// Mixed-quant models can resolve to CPU decode even when the backend default is
    /// GPU decode. In that case we must allocate CPU KV up front to avoid a
    /// GPU-KV/CPU-decode mismatch at runtime.
    pub fn create_model_kv_for_weights(&self, weights: &WeightStore) -> ModelKv {
        self.kv_plan()
            .build_decode_compatible(self.backend.as_ref(), gpu_decode_quant_supported(weights))
    }

    /// Create a KV cache matched to the active decode support of the loaded weights,
    /// while honoring caller-specified planning requirements such as a smaller
    /// context length than the model maximum.
    pub fn create_model_kv_for_weights_with_requirements(
        &self,
        weights: &WeightStore,
        requirements: crate::backend::KvPlannerRequirements,
    ) -> anyhow::Result<ModelKv> {
        Ok(self
            .kv_plan_with_requirements(requirements)?
            .build_decode_compatible(self.backend.as_ref(), gpu_decode_quant_supported(weights)))
    }

    /// Resolve the backend-side KV allocation plan for this model.
    pub fn kv_plan(&self) -> crate::backend::KvPlan {
        crate::backend::KvPlanner::plan(self.backend.as_ref(), &self.config)
    }

    /// Flush backend-owned hybrid state into the KV before host-side control
    /// paths such as snapshotting inspect it.
    pub fn sync_model_kv(&self, kv: &mut ModelKv) {
        if let Some(qwen_kv) = kv.as_qwen35_mut() {
            self.backend.sync_qwen35_kv(qwen_kv);
        }
    }

    pub fn try_clone_qwen35_recurrent_slot_via_backend(
        &self,
        kv: &mut ModelKv,
        src_slot_idx: usize,
        dst_slot_idx: usize,
    ) -> anyhow::Result<bool> {
        let Some(qwen_kv) = kv.as_qwen35_mut() else {
            anyhow::bail!("qwen35 recurrent slot clone requires ModelKv::Qwen35");
        };
        Ok(self
            .backend
            .try_clone_qwen35_recurrent_slot(qwen_kv, src_slot_idx, dst_slot_idx))
    }

    /// Prime Qwen3.5 recurrent Metal slot buffers from the current KV state.
    ///
    /// This is intended for prefill experiments that want to measure a
    /// persistent backend-owned slot-buffer path instead of a cold fresh-KV
    /// upload on the first recurrent handoff. Non-Qwen3.5 or non-Metal paths
    /// are treated as no-ops.
    pub fn prime_qwen35_recurrent_slot_buffers(
        &self,
        kv: &mut ModelKv,
        slot_indices: &[usize],
    ) -> anyhow::Result<()> {
        if self.arch_name() != "qwen35" || slot_indices.is_empty() {
            return Ok(());
        }
        let Some(metal_ops) = self.backend.metal_ops() else {
            return Ok(());
        };
        let Some(qwen_kv) = kv.as_qwen35_mut() else {
            anyhow::bail!("qwen35 slot-buffer priming requires ModelKv::Qwen35");
        };

        for &slot_idx in slot_indices {
            anyhow::ensure!(
                qwen_kv.has_recurrent_slot(slot_idx),
                "qwen35 slot-buffer priming requires existing recurrent slot {slot_idx}"
            );
        }

        for layer_idx in 0..qwen_kv.layer_count() {
            if !qwen_kv.is_recurrent_layer(layer_idx) {
                continue;
            }
            for &slot_idx in slot_indices {
                metal_ops.sync_qwen35_slot_buffers_from_kv(qwen_kv, layer_idx, slot_idx);
                qwen_kv.mark_layer_state_backend_owned(slot_idx, layer_idx);
            }
        }
        Ok(())
    }

    pub fn kv_plan_with_requirements(
        &self,
        requirements: crate::backend::KvPlannerRequirements,
    ) -> anyhow::Result<crate::backend::KvPlan> {
        crate::backend::KvPlanner::plan_with_requirements(
            self.backend.as_ref(),
            &self.config,
            requirements,
        )
    }

    pub fn decode_plan_summary(
        &self,
        kv: &ModelKv,
        intent: DecodeIntent,
        allow_pipelined: bool,
    ) -> String {
        DecodeExecutionPlan::for_model(self, kv, intent, allow_pipelined).summary_label()
    }

    pub fn prefill_plan_summary(
        &self,
        weights: &WeightStore,
        kv: &ModelKv,
        n_tokens: usize,
    ) -> anyhow::Result<String> {
        let ctx = self.forward_context();
        let mode_plan =
            match PrefillExecutionPlan::for_forward_batch(&ctx, kv, weights, n_tokens, false) {
                Ok(plan) => plan,
                Err(_) if self.arch_name() == "qwen3" => {
                    return Ok("mode=serial reason=unsupported_qwen3_layout".to_string());
                }
                Err(e) => return Err(e),
            };
        if mode_plan.mode == PrefillMode::Serial {
            return Ok(mode_plan.summary_label());
        }

        if self.arch_name() == "qwen35" && matches!(kv, ModelKv::Qwen35(_)) {
            return Ok(format!(
                "{} kv=qwen35_hybrid recurrent=backend_owned",
                mode_plan.summary_label()
            ));
        }

        let gpu_kv = kv.as_gpu().unwrap();
        let metal_ops = self.metal_ops().unwrap();

        let base_seq_len = gpu_kv.seq_len();
        let summary = match self.arch_name() {
            "llama" => {
                let plan = DecodeExecutionPlan::llama_prefill(
                    metal_ops,
                    gpu_kv,
                    base_seq_len,
                    n_tokens as u32,
                    self.config.head_dim,
                    self.prefill_plan_has_q8_weights(weights)?,
                    gpu_prefill_uses_q5k(weights),
                    gpu_prefill_q5k_small_n_auto_eligible(weights),
                    metal_prefill_attn_f16out_enabled(),
                    metal_prefill_use_cached0_enabled(),
                    metal_prefill_split_rope_append_enabled(),
                );
                let route = self.prefill_attention_route(
                    plan,
                    base_seq_len,
                    n_tokens,
                    self.config.head_dim,
                    0,
                );
                plan.summary_label(
                    mode_plan.summary_label().trim_start_matches("mode="),
                    &route,
                )
            }
            "qwen3" => {
                let sliding_window = self.config.sliding_window_size.unwrap_or(0);
                let plan = DecodeExecutionPlan::qwen3_prefill(
                    metal_ops,
                    gpu_kv,
                    base_seq_len,
                    n_tokens as u32,
                    self.config.head_dim,
                    sliding_window,
                    gpu_prefill_uses_q5k(weights),
                    gpu_prefill_q5k_small_n_auto_eligible(weights),
                );
                let route = self.prefill_attention_route(
                    plan,
                    base_seq_len,
                    n_tokens,
                    self.config.head_dim,
                    plan.attention_sliding_window,
                );
                plan.summary_label(
                    mode_plan.summary_label().trim_start_matches("mode="),
                    &route,
                )
            }
            "gemma3" => {
                let plan = DecodeExecutionPlan::gemma3_prefill(
                    metal_ops,
                    gpu_kv,
                    n_tokens as u32,
                    gpu_prefill_uses_q5k(weights),
                    gpu_prefill_q5k_small_n_auto_eligible(weights),
                );
                let route = self.prefill_attention_route(
                    plan,
                    base_seq_len,
                    n_tokens,
                    self.config.head_dim,
                    0,
                );
                plan.summary_label(
                    mode_plan.summary_label().trim_start_matches("mode="),
                    &route,
                )
            }
            _ => mode_plan.summary_label(),
        };

        Ok(summary)
    }

    /// Run a single decode step: one token in, logits out.
    pub fn forward_single(
        &self,
        token_id: u32,
        position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        let ctx = self.forward_context();
        self.forward
            .forward_single(&ctx, token_id, position, kv, weights, logits, None)
    }

    /// Run batched prefill: process all tokens, return only last token's logits.
    ///
    /// Uses GPU batched attention when available, otherwise falls back to serial.
    pub fn forward_batch(
        &self,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        let ctx = self.forward_context();
        self.forward
            .forward_batch(&ctx, token_ids, kv, weights, logits)
    }

    /// Run batched prefill once while fanning the same token stream out to
    /// multiple Qwen3.5 recurrent slots on a shared attention timeline.
    ///
    /// This is a scoped helper around `forward_batch` for shared-prefix /
    /// branch warmup flows. The configured slot batch is cleared before the
    /// method returns.
    pub fn forward_batch_qwen35_shared_timeline(
        &self,
        token_ids: &[u32],
        kv: &mut ModelKv,
        qwen35_slot_indices: &[usize],
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        kv.with_qwen35_batch_slot_indices(qwen35_slot_indices, |kv| {
            self.forward_batch(token_ids, kv, weights, logits)
        })
    }

    /// Run batched prefill once while temporarily forking shared-timeline
    /// Qwen3.5 recurrent branches from the current active slot.
    ///
    /// When `qwen35_slot_count == 1`, this falls back to a normal
    /// `forward_batch`. For larger counts, branch slots are allocated, cloned
    /// from the current active slot, used for the prefill, then freed before
    /// returning.
    pub fn forward_batch_qwen35_shared_timeline_forked(
        &self,
        token_ids: &[u32],
        kv: &mut ModelKv,
        qwen35_slot_count: usize,
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        if qwen35_slot_count <= 1 {
            return self.forward_batch(token_ids, kv, weights, logits);
        }
        kv.with_qwen35_shared_timeline_branches(qwen35_slot_count, |kv, slot_indices| {
            self.forward_batch_qwen35_shared_timeline(token_ids, kv, slot_indices, weights, logits)
        })
    }

    /// Like `forward_batch_qwen35_shared_timeline_forked`, but branches from a
    /// specific Qwen3.5 recurrent slot instead of the current active slot.
    pub fn forward_batch_qwen35_shared_timeline_forked_from_slot(
        &self,
        token_ids: &[u32],
        kv: &mut ModelKv,
        source_slot: usize,
        qwen35_slot_count: usize,
        weights: &WeightStore,
        logits: &mut [f32],
    ) -> anyhow::Result<()> {
        kv.with_qwen35_shared_timeline_branches_from_slot(
            source_slot,
            qwen35_slot_count.max(1),
            |kv, slot_indices| {
                self.forward_batch_qwen35_shared_timeline(
                    token_ids,
                    kv,
                    slot_indices,
                    weights,
                    logits,
                )
            },
        )
    }

    /// Profiled batched prefill: same as `forward_batch` but records coarse
    /// operation timing into the provided `OpBreakdown`.
    pub fn forward_batch_profiled(
        &self,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
        ops: &mut OpBreakdown,
    ) -> anyhow::Result<()> {
        let ctx = self.forward_context();
        self.forward
            .forward_batch_profiled(&ctx, token_ids, kv, weights, logits, ops)
    }

    /// Profiled variant of `forward_batch_qwen35_shared_timeline`.
    pub fn forward_batch_profiled_qwen35_shared_timeline(
        &self,
        token_ids: &[u32],
        kv: &mut ModelKv,
        qwen35_slot_indices: &[usize],
        weights: &WeightStore,
        logits: &mut [f32],
        ops: &mut OpBreakdown,
    ) -> anyhow::Result<()> {
        kv.with_qwen35_batch_slot_indices(qwen35_slot_indices, |kv| {
            self.forward_batch_profiled(token_ids, kv, weights, logits, ops)
        })
    }

    /// Profiled variant of `forward_batch_qwen35_shared_timeline_forked`.
    pub fn forward_batch_profiled_qwen35_shared_timeline_forked(
        &self,
        token_ids: &[u32],
        kv: &mut ModelKv,
        qwen35_slot_count: usize,
        weights: &WeightStore,
        logits: &mut [f32],
        ops: &mut OpBreakdown,
    ) -> anyhow::Result<()> {
        if qwen35_slot_count <= 1 {
            return self.forward_batch_profiled(token_ids, kv, weights, logits, ops);
        }
        kv.with_qwen35_shared_timeline_branches(qwen35_slot_count, |kv, slot_indices| {
            self.forward_batch_profiled_qwen35_shared_timeline(
                token_ids,
                kv,
                slot_indices,
                weights,
                logits,
                ops,
            )
        })
    }

    /// Profiled variant of `forward_batch_qwen35_shared_timeline_forked_from_slot`.
    pub fn forward_batch_profiled_qwen35_shared_timeline_forked_from_slot(
        &self,
        token_ids: &[u32],
        kv: &mut ModelKv,
        source_slot: usize,
        qwen35_slot_count: usize,
        weights: &WeightStore,
        logits: &mut [f32],
        ops: &mut OpBreakdown,
    ) -> anyhow::Result<()> {
        kv.with_qwen35_shared_timeline_branches_from_slot(
            source_slot,
            qwen35_slot_count.max(1),
            |kv, slot_indices| {
                self.forward_batch_profiled_qwen35_shared_timeline(
                    token_ids,
                    kv,
                    slot_indices,
                    weights,
                    logits,
                    ops,
                )
            },
        )
    }

    /// Profiled decode step: same as `forward_single` but records per-operation
    /// timing into the provided `OpBreakdown`.
    pub fn forward_single_profiled(
        &self,
        token_id: u32,
        position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits: &mut [f32],
        ops: &mut OpBreakdown,
    ) -> anyhow::Result<()> {
        let ctx = self.forward_context();
        self.forward
            .forward_single(&ctx, token_id, position, kv, weights, logits, Some(ops))
    }

    /// Run a forward pass on N tokens and return logits for every position.
    ///
    /// `logits_all` is resized to `token_ids.len() * vocab_size` on return.
    /// `logits_all[i * vocab_size .. (i+1) * vocab_size]` contains the unnormalized
    /// logit vector for token position `kv.seq_len() + i` (before the call).
    ///
    /// Used by speculative decoding to obtain target model probabilities for
    /// all K+1 tokens (last accepted + K draft candidates) in one call.
    ///
    /// Implementation note (v2.0): runs N sequential `forward_single` calls.
    /// v2.1 will replace this with a single GPU batch dispatch that runs LM-head
    /// on every token position.
    pub fn forward_batch_all_logits(
        &self,
        token_ids: &[u32],
        kv: &mut ModelKv,
        weights: &WeightStore,
        logits_all: &mut Vec<f32>,
    ) -> anyhow::Result<()> {
        let ctx = self.forward_context();
        self.forward
            .forward_batch_all_logits(&ctx, token_ids, kv, weights, logits_all)
    }

    /// All-logits variant of `forward_batch_qwen35_shared_timeline`.
    pub fn forward_batch_all_logits_qwen35_shared_timeline(
        &self,
        token_ids: &[u32],
        kv: &mut ModelKv,
        qwen35_slot_indices: &[usize],
        weights: &WeightStore,
        logits_all: &mut Vec<f32>,
    ) -> anyhow::Result<()> {
        kv.with_qwen35_batch_slot_indices(qwen35_slot_indices, |kv| {
            self.forward_batch_all_logits(token_ids, kv, weights, logits_all)
        })
    }

    /// All-logits variant of `forward_batch_qwen35_shared_timeline_forked`.
    pub fn forward_batch_all_logits_qwen35_shared_timeline_forked(
        &self,
        token_ids: &[u32],
        kv: &mut ModelKv,
        qwen35_slot_count: usize,
        weights: &WeightStore,
        logits_all: &mut Vec<f32>,
    ) -> anyhow::Result<()> {
        if qwen35_slot_count <= 1 {
            return self.forward_batch_all_logits(token_ids, kv, weights, logits_all);
        }
        kv.with_qwen35_shared_timeline_branches(qwen35_slot_count, |kv, slot_indices| {
            self.forward_batch_all_logits_qwen35_shared_timeline(
                token_ids,
                kv,
                slot_indices,
                weights,
                logits_all,
            )
        })
    }

    /// All-logits variant of `forward_batch_qwen35_shared_timeline_forked_from_slot`.
    pub fn forward_batch_all_logits_qwen35_shared_timeline_forked_from_slot(
        &self,
        token_ids: &[u32],
        kv: &mut ModelKv,
        source_slot: usize,
        qwen35_slot_count: usize,
        weights: &WeightStore,
        logits_all: &mut Vec<f32>,
    ) -> anyhow::Result<()> {
        kv.with_qwen35_shared_timeline_branches_from_slot(
            source_slot,
            qwen35_slot_count.max(1),
            |kv, slot_indices| {
                self.forward_batch_all_logits_qwen35_shared_timeline(
                    token_ids,
                    kv,
                    slot_indices,
                    weights,
                    logits_all,
                )
            },
        )
    }

    /// Get the name of the architecture used by this model's forward pass.
    pub fn arch_name(&self) -> &str {
        self.forward.arch_name()
    }

    /// True when this model can safely use the current pipelined decode implementation.
    pub fn supports_pipelined_decode(&self) -> bool {
        self.forward
            .supports_pipelined_decode(&self.forward_context())
    }

    pub fn use_gpu_decode(&self) -> bool {
        self.backend.use_gpu_decode()
    }

    // ── Pipelining helpers (PERF-002) ────────────────────────────────────────

    /// Return the Metal device if this model uses a GPU backend, else `None`.
    pub fn metal_device(&self) -> Option<&ax_engine_metal::MetalDevice> {
        self.backend.metal_ops().map(|m| &m.device)
    }

    pub(crate) fn metal_ops(&self) -> Option<&MetalOps> {
        self.backend.metal_ops()
    }

    /// Reset backend-local Metal performance counters for this model.
    pub fn reset_metal_perf_counters(&self) {
        if let Some(device) = self.metal_device() {
            device.reset_perf_counters();
        }
        if let Some(ops) = self.backend.metal_ops() {
            ops.reset_qwen35_recurrent_batch_perf_counters();
        }
    }

    /// Read backend-local Metal performance counters for this model.
    pub fn read_metal_perf_counters(&self) -> ax_engine_metal::PerfCounters {
        self.metal_device()
            .map(ax_engine_metal::MetalDevice::perf_counters)
            .unwrap_or_default()
    }

    pub fn read_qwen35_recurrent_batch_perf_counters(
        &self,
    ) -> crate::backend::metal::Qwen35RecurrentBatchPerfCounters {
        self.backend
            .metal_ops()
            .map(crate::backend::metal::MetalOps::qwen35_recurrent_batch_perf_counters)
            .unwrap_or_default()
    }

    /// Allocate a Metal shared-memory buffer of `bytes` bytes.
    ///
    /// Returns `Err` if no Metal device is available.
    pub fn alloc_metal_buf(&self, bytes: usize) -> anyhow::Result<ax_engine_metal::MetalBuffer> {
        let m = self
            .backend
            .metal_ops()
            .ok_or_else(|| anyhow::anyhow!("No Metal backend available"))?;
        ax_engine_metal::MetalBuffer::new(m.device.device(), bytes)
    }

    /// Write the embedding for `token_id` into a Metal buffer (zero-copy UMA write).
    ///
    /// `buf` must be at least `embedding_dim * 4` bytes (f32 per element).
    pub fn embed_token_into(
        &self,
        token_id: u32,
        buf: &ax_engine_metal::MetalBuffer,
        weights: &WeightStore,
    ) -> anyhow::Result<()> {
        self.forward
            .embed_pipelined_token(&self.forward_context(), token_id, buf, weights)
    }

    /// Encode a single decode step into a [`ax_engine_metal::PendingFrame`] without committing.
    ///
    /// Returns `Some(frame)` if the model uses a GPU backend and `kv` is a GPU KV
    /// cache, otherwise `None` (caller should fall back to `forward_single`).
    ///
    /// **Pipelining contract** — the caller is responsible for:
    /// 1. Calling [`prewarm_kv_capacity`] before the decode loop (prevents
    ///    reallocation while a prior frame may be inflight on the GPU).
    /// 2. Writing the embedding into `hidden_buf` via [`embed_token_into`]
    ///    *after* encoding but *before* committing the frame.
    /// 3. Calling [`advance_gpu_kv_token`] after [`ax_engine_metal::MetalDevice::wait_frame`].
    pub fn encode_pending_decode_step(
        &self,
        hidden_buf: &ax_engine_metal::MetalBuffer,
        position: usize,
        kv: &mut ModelKv,
        weights: &WeightStore,
    ) -> anyhow::Result<Option<ax_engine_metal::PendingFrame>> {
        self.forward.encode_pending_decode_step(
            &self.forward_context(),
            hidden_buf,
            position,
            kv,
            weights,
        )
    }

    /// Pre-allocate GPU KV capacity for at least `needed` positions.
    ///
    /// Must be called once before the pipelined decode loop starts, so that
    /// `ensure_capacity` is guaranteed to be a no-op (and therefore safe to
    /// call while a command buffer is inflight).
    pub fn prewarm_kv_capacity(&self, kv: &mut ModelKv, needed: usize) -> anyhow::Result<()> {
        let Some(metal_ops) = self.backend.metal_ops() else {
            return Ok(());
        };
        if let Some(qwen_kv) = kv.as_qwen35_mut() {
            anyhow::ensure!(
                qwen_kv.ensure_gpu_attention_capacity_for(needed),
                "failed to prewarm qwen35 GPU attention KV capacity for pipelined decode"
            );
            return Ok(());
        }
        let Some(gpu_kv) = kv.as_gpu_mut() else {
            return Ok(());
        };
        gpu_kv.ensure_capacity(&metal_ops.device, needed)
    }

    /// Advance the GPU KV cache seq_len by 1 after a pipelined decode step completes.
    ///
    /// Must be called after [`ax_engine_metal::MetalDevice::wait_frame`], not before.
    pub fn advance_gpu_kv_token(&self, kv: &mut ModelKv) {
        if let Some(gpu_kv) = kv.as_gpu_mut() {
            gpu_kv.finalize_token();
            return;
        }
        let Some(qwen_kv) = kv.as_qwen35_mut() else {
            return;
        };
        let Some(metal_ops) = self.backend.metal_ops() else {
            return;
        };
        let recurrent_slot = qwen_kv.active_slot();
        for layer in 0..qwen_kv.layer_count() {
            if !qwen_kv.is_recurrent_layer(layer) {
                continue;
            }
            let conv_generation = qwen_kv.note_backend_conv_state_update(recurrent_slot, layer);
            let recurrent_generation =
                qwen_kv.note_backend_recurrent_state_update(recurrent_slot, layer);
            let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
            let recurrent_state_stride = qwen_kv.recurrent_state_len();
            metal_ops.with_qwen35_recurrent_slot_buffer(
                layer,
                recurrent_slot,
                conv_state_stride,
                recurrent_state_stride,
                |slot_buffers| {
                    slot_buffers.conv_synced_generation = Some(conv_generation);
                    slot_buffers.recurrent_synced_generation = Some(recurrent_generation);
                },
            );
        }
        qwen_kv.mark_attention_cpu_dirty();
        qwen_kv.finalize_token();
    }

    /// Copy logits from the GPU scratch buffer into the provided CPU slice.
    ///
    /// Must be called after the inflight frame has completed (after `wait_frame`).
    /// `logits` must have length >= `vocab_size`.
    pub fn read_gpu_logits(&self, logits: &mut [f32]) -> anyhow::Result<()> {
        let Some(metal_ops) = self.backend.metal_ops() else {
            anyhow::bail!("read_gpu_logits: no Metal backend");
        };
        metal_ops.init_scratches(&self.config);
        let guard = metal_ops.scratches();
        let s = guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GPU scratch buffers not initialized"))?;
        let vocab = self.config.vocab_size as usize;
        let logits_gpu = unsafe {
            std::slice::from_raw_parts(s.logits_buf.contents().as_ptr() as *const f32, vocab)
        };
        logits[..vocab].copy_from_slice(logits_gpu);
        self.forward
            .postprocess_pipelined_logits(&self.forward_context(), &mut logits[..vocab])?;
        Ok(())
    }
}
