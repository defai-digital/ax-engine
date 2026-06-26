use super::*;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct Gemma4UnifiedMultimodalTelemetry {
    pub(crate) prefill_requests: u32,
    pub(crate) image_inputs: u32,
    pub(crate) audio_inputs: u32,
    pub(crate) video_inputs: u32,
    pub(crate) visual_inputs: u32,
    pub(crate) prefix_cache_disabled: u32,
    pub(crate) mtp_prefill_warmup_skipped: u32,
}

impl Gemma4UnifiedMultimodalTelemetry {
    pub(crate) fn record_prefill(
        &mut self,
        inputs: &ax_engine_core::gemma4_unified::Gemma4UnifiedRuntimeInputs,
        mtp_available: bool,
    ) {
        self.prefill_requests = self.prefill_requests.saturating_add(1);
        let image_inputs = saturating_u32(inputs.images.len());
        let audio_inputs = saturating_u32(inputs.audios.len());
        let video_inputs = saturating_u32(inputs.videos.len());
        self.image_inputs = self.image_inputs.saturating_add(image_inputs);
        self.audio_inputs = self.audio_inputs.saturating_add(audio_inputs);
        self.video_inputs = self.video_inputs.saturating_add(video_inputs);
        self.visual_inputs = self
            .visual_inputs
            .saturating_add(image_inputs.saturating_add(video_inputs));
        self.prefix_cache_disabled = self.prefix_cache_disabled.saturating_add(1);
        if mtp_available {
            self.mtp_prefill_warmup_skipped = self.mtp_prefill_warmup_skipped.saturating_add(1);
        }
    }

    pub(crate) fn merge_from(&mut self, other: Self) {
        self.prefill_requests = self.prefill_requests.saturating_add(other.prefill_requests);
        self.image_inputs = self.image_inputs.saturating_add(other.image_inputs);
        self.audio_inputs = self.audio_inputs.saturating_add(other.audio_inputs);
        self.video_inputs = self.video_inputs.saturating_add(other.video_inputs);
        self.visual_inputs = self.visual_inputs.saturating_add(other.visual_inputs);
        self.prefix_cache_disabled = self
            .prefix_cache_disabled
            .saturating_add(other.prefix_cache_disabled);
        self.mtp_prefill_warmup_skipped = self
            .mtp_prefill_warmup_skipped
            .saturating_add(other.mtp_prefill_warmup_skipped);
    }

    pub(crate) fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
        if *self == Self::default() {
            return;
        }
        let entries = [
            (
                ROUTE_DECISION_AX_MLX_GEMMA4_UNIFIED_MULTIMODAL_PREFILL_REQUESTS,
                self.prefill_requests,
            ),
            (
                ROUTE_DECISION_AX_MLX_GEMMA4_UNIFIED_IMAGE_INPUTS,
                self.image_inputs,
            ),
            (
                ROUTE_DECISION_AX_MLX_GEMMA4_UNIFIED_AUDIO_INPUTS,
                self.audio_inputs,
            ),
            (
                ROUTE_DECISION_AX_MLX_GEMMA4_UNIFIED_VIDEO_INPUTS,
                self.video_inputs,
            ),
            (
                ROUTE_DECISION_AX_MLX_GEMMA4_UNIFIED_VISUAL_INPUTS,
                self.visual_inputs,
            ),
            (
                ROUTE_DECISION_AX_MLX_GEMMA4_UNIFIED_PREFIX_CACHE_DISABLED,
                self.prefix_cache_disabled,
            ),
            (
                ROUTE_DECISION_AX_MLX_GEMMA4_UNIFIED_MTP_PREFILL_WARMUP_SKIPPED,
                self.mtp_prefill_warmup_skipped,
            ),
        ];
        for (key, value) in entries {
            decisions.upsert_route_decision(key, value);
        }
    }
}

/// Maximum n-gram accept depth bucket exposed as route telemetry. Each
/// `record_draft` call bumps `accepts_by_depth[min(accept_count, MAX-1)]`.
/// Bucket 0 = draft attempt with zero accepted tokens; bucket k (k > 0) =
/// draft attempt with exactly k accepted tokens. The histogram is the
/// raw input PRD §8 Phase 6 requires for the n-gram acceptance-by-depth
/// claim. We cap at 8 because that comfortably exceeds the current
/// maximum draft length and keeps the route-decision surface small.
pub(crate) const NGRAM_ACCEPT_DEPTH_BUCKETS: usize = 8;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct NgramAccelerationTelemetry {
    pub(crate) draft_attempts: u32,
    pub(crate) draft_tokens: u32,
    pub(crate) accepted_tokens: u32,
    pub(crate) rejected_tokens: u32,
    pub(crate) full_accepts: u32,
    pub(crate) partial_rejects: u32,
    pub(crate) complete_misses: u32,
    pub(crate) no_draft_steps: u32,
    pub(crate) cooldown_steps: u32,
    pub(crate) cooldown_events: u32,
    pub(crate) cooldown_steps_scheduled: u32,
    pub(crate) request_disable_events: u32,
    pub(crate) request_disabled_steps: u32,
    pub(crate) fallback_no_candidate_steps: u32,
    pub(crate) fallback_confidence_filtered_steps: u32,
    pub(crate) fallback_short_output_steps: u32,
    pub(crate) fallback_linear_no_draft_steps: u32,
    pub(crate) policy_variant_code: u32,
    pub(crate) adaptive_draft_len_steps: u32,
    pub(crate) adaptive_draft_len_total: u32,
    pub(crate) prompt_class_code: u32,
    /// Per-attempt acceptance histogram. Index k counts draft attempts
    /// where exactly k tokens were accepted; attempts that accepted ≥ N
    /// tokens land in bucket N-1.
    pub(crate) accepts_by_depth: [u32; NGRAM_ACCEPT_DEPTH_BUCKETS],
    /// Steps where n-gram drafting was skipped because the model is outside a
    /// `<think>` block (think-gating enabled, not in think region).
    pub(crate) think_gated_steps: u32,
}

impl NgramAccelerationTelemetry {
    pub(crate) fn record_draft(&mut self, draft_len: usize, accept_count: usize) {
        self.draft_attempts = self.draft_attempts.saturating_add(1);
        self.draft_tokens = self.draft_tokens.saturating_add(saturating_u32(draft_len));
        self.accepted_tokens = self
            .accepted_tokens
            .saturating_add(saturating_u32(accept_count));
        self.rejected_tokens = self
            .rejected_tokens
            .saturating_add(saturating_u32(draft_len.saturating_sub(accept_count)));

        if accept_count == draft_len {
            self.full_accepts = self.full_accepts.saturating_add(1);
        } else if accept_count == 0 {
            self.complete_misses = self.complete_misses.saturating_add(1);
        } else {
            self.partial_rejects = self.partial_rejects.saturating_add(1);
        }

        // Per-attempt acceptance histogram. Saturate at the last bucket so
        // a draft length > NGRAM_ACCEPT_DEPTH_BUCKETS does not silently
        // skip the count.
        let bucket = accept_count.min(NGRAM_ACCEPT_DEPTH_BUCKETS - 1);
        self.accepts_by_depth[bucket] = self.accepts_by_depth[bucket].saturating_add(1);
    }

    pub(crate) fn record_no_draft(&mut self) {
        self.no_draft_steps = self.no_draft_steps.saturating_add(1);
    }

    pub(crate) fn record_no_draft_reason(&mut self, reason: Option<NgramDraftRejection>) {
        match reason {
            Some(NgramDraftRejection::ConfidenceFiltered) => {
                self.fallback_confidence_filtered_steps =
                    self.fallback_confidence_filtered_steps.saturating_add(1);
            }
            Some(NgramDraftRejection::NoCandidate) | None => {
                self.fallback_no_candidate_steps =
                    self.fallback_no_candidate_steps.saturating_add(1);
            }
        }
    }

    pub(crate) fn record_cooldown_step(&mut self) {
        self.cooldown_steps = self.cooldown_steps.saturating_add(1);
    }

    pub(crate) fn record_cooldown_event(&mut self, disabled_steps: u32) {
        self.cooldown_events = self.cooldown_events.saturating_add(1);
        self.cooldown_steps_scheduled =
            self.cooldown_steps_scheduled.saturating_add(disabled_steps);
    }

    pub(crate) fn record_request_disable_event(&mut self) {
        self.request_disable_events = self.request_disable_events.saturating_add(1);
    }

    pub(crate) fn record_request_disabled_step(&mut self) {
        self.request_disabled_steps = self.request_disabled_steps.saturating_add(1);
    }

    pub(crate) fn record_request_disabled_reason(&mut self, reason: NgramRequestDisableReason) {
        match reason {
            NgramRequestDisableReason::None => {}
            NgramRequestDisableReason::ShortOutputBudget => {
                self.fallback_short_output_steps =
                    self.fallback_short_output_steps.saturating_add(1);
            }
            NgramRequestDisableReason::LinearNoDraft => {
                self.fallback_linear_no_draft_steps =
                    self.fallback_linear_no_draft_steps.saturating_add(1);
            }
            NgramRequestDisableReason::LinearInitialNoDraft => {
                self.fallback_linear_no_draft_steps =
                    self.fallback_linear_no_draft_steps.saturating_add(1);
            }
        }
    }

    pub(crate) fn record_policy(
        &mut self,
        variant: NgramPolicyVariant,
        requested_draft_len: usize,
    ) {
        self.policy_variant_code = variant.route_code();
        self.adaptive_draft_len_steps = self.adaptive_draft_len_steps.saturating_add(1);
        self.adaptive_draft_len_total = self
            .adaptive_draft_len_total
            .saturating_add(saturating_u32(requested_draft_len));
    }

    pub(crate) fn record_prompt_class(&mut self, class_code: u32) {
        self.prompt_class_code = self.prompt_class_code.max(class_code);
    }

    pub(crate) fn merge_from(&mut self, other: Self) {
        self.draft_attempts = self.draft_attempts.saturating_add(other.draft_attempts);
        self.draft_tokens = self.draft_tokens.saturating_add(other.draft_tokens);
        self.accepted_tokens = self.accepted_tokens.saturating_add(other.accepted_tokens);
        self.rejected_tokens = self.rejected_tokens.saturating_add(other.rejected_tokens);
        self.full_accepts = self.full_accepts.saturating_add(other.full_accepts);
        self.partial_rejects = self.partial_rejects.saturating_add(other.partial_rejects);
        self.complete_misses = self.complete_misses.saturating_add(other.complete_misses);
        self.no_draft_steps = self.no_draft_steps.saturating_add(other.no_draft_steps);
        self.cooldown_steps = self.cooldown_steps.saturating_add(other.cooldown_steps);
        self.cooldown_events = self.cooldown_events.saturating_add(other.cooldown_events);
        self.cooldown_steps_scheduled = self
            .cooldown_steps_scheduled
            .saturating_add(other.cooldown_steps_scheduled);
        self.request_disable_events = self
            .request_disable_events
            .saturating_add(other.request_disable_events);
        self.request_disabled_steps = self
            .request_disabled_steps
            .saturating_add(other.request_disabled_steps);
        self.fallback_no_candidate_steps = self
            .fallback_no_candidate_steps
            .saturating_add(other.fallback_no_candidate_steps);
        self.fallback_confidence_filtered_steps = self
            .fallback_confidence_filtered_steps
            .saturating_add(other.fallback_confidence_filtered_steps);
        self.fallback_short_output_steps = self
            .fallback_short_output_steps
            .saturating_add(other.fallback_short_output_steps);
        self.fallback_linear_no_draft_steps = self
            .fallback_linear_no_draft_steps
            .saturating_add(other.fallback_linear_no_draft_steps);
        self.policy_variant_code = self.policy_variant_code.max(other.policy_variant_code);
        self.adaptive_draft_len_steps = self
            .adaptive_draft_len_steps
            .saturating_add(other.adaptive_draft_len_steps);
        self.adaptive_draft_len_total = self
            .adaptive_draft_len_total
            .saturating_add(other.adaptive_draft_len_total);
        self.prompt_class_code = self.prompt_class_code.max(other.prompt_class_code);
        for i in 0..NGRAM_ACCEPT_DEPTH_BUCKETS {
            self.accepts_by_depth[i] =
                self.accepts_by_depth[i].saturating_add(other.accepts_by_depth[i]);
        }
        self.think_gated_steps = self
            .think_gated_steps
            .saturating_add(other.think_gated_steps);
    }

    pub(crate) fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
        let entries = [
            ("ax_ngram_draft_attempts", self.draft_attempts),
            ("ax_ngram_draft_tokens", self.draft_tokens),
            ("ax_ngram_accepted_tokens", self.accepted_tokens),
            ("ax_ngram_rejected_tokens", self.rejected_tokens),
            ("ax_ngram_full_accepts", self.full_accepts),
            ("ax_ngram_partial_rejects", self.partial_rejects),
            ("ax_ngram_complete_misses", self.complete_misses),
            ("ax_ngram_no_draft_steps", self.no_draft_steps),
            ("ax_ngram_cooldown_steps", self.cooldown_steps),
            ("ax_ngram_cooldown_events", self.cooldown_events),
            (
                "ax_ngram_cooldown_steps_scheduled",
                self.cooldown_steps_scheduled,
            ),
            (
                "ax_ngram_request_disable_events",
                self.request_disable_events,
            ),
            (
                "ax_ngram_request_disabled_steps",
                self.request_disabled_steps,
            ),
            (
                "ax_ngram_fallback_no_candidate_steps",
                self.fallback_no_candidate_steps,
            ),
            (
                "ax_ngram_fallback_confidence_filtered_steps",
                self.fallback_confidence_filtered_steps,
            ),
            (
                "ax_ngram_fallback_short_output_steps",
                self.fallback_short_output_steps,
            ),
            (
                "ax_ngram_fallback_linear_no_draft_steps",
                self.fallback_linear_no_draft_steps,
            ),
            ("ax_ngram_policy_variant", self.policy_variant_code),
            (
                "ax_ngram_adaptive_draft_len_steps",
                self.adaptive_draft_len_steps,
            ),
            (
                "ax_ngram_adaptive_draft_len_total",
                self.adaptive_draft_len_total,
            ),
            ("ax_prompt_class_code", self.prompt_class_code),
            ("ax_ngram_think_gated_steps", self.think_gated_steps),
        ];

        for (key, value) in entries {
            decisions.upsert_route_decision(key, value);
        }

        // Acceptance-by-depth histogram (PRD §8 Phase 6 / I-6). Bucket k
        // counts draft attempts where exactly k tokens were accepted;
        // attempts with ≥ NGRAM_ACCEPT_DEPTH_BUCKETS accepted tokens
        // saturate into the last bucket.
        for (depth, count) in self.accepts_by_depth.iter().enumerate() {
            decisions.upsert_route_decision(ngram_accept_at_depth_key(depth), *count);
        }
    }
}

/// Returns the stable route-decision key for a given accept-depth bucket.
/// Hoisted out of the per-call hot path so the (static) key strings are
/// constructed once at compile time per bucket.
pub(crate) fn ngram_accept_at_depth_key(depth: usize) -> &'static str {
    // NGRAM_ACCEPT_DEPTH_BUCKETS = 8. If the constant changes, extend this
    // table — there is no runtime allocation by design.
    match depth {
        0 => "ax_ngram_accept_at_depth_0",
        1 => "ax_ngram_accept_at_depth_1",
        2 => "ax_ngram_accept_at_depth_2",
        3 => "ax_ngram_accept_at_depth_3",
        4 => "ax_ngram_accept_at_depth_4",
        5 => "ax_ngram_accept_at_depth_5",
        6 => "ax_ngram_accept_at_depth_6",
        7 => "ax_ngram_accept_at_depth_7",
        _ => "ax_ngram_accept_at_depth_overflow",
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct WeightLayoutTelemetry {
    pub(crate) dense_ffn_gate_up_packed_layers: u32,
    pub(crate) dense_ffn_split_gate_up_layers: u32,
    pub(crate) dense_attention_qkv_packed_layers: u32,
    pub(crate) dense_attention_split_qkv_layers: u32,
    pub(crate) linear_attention_qkvz_ba_packed_layers: u32,
    pub(crate) linear_attention_split_qkvba_layers: u32,
}

impl WeightLayoutTelemetry {
    pub(crate) fn from_weights(weights: &ModelWeights) -> Self {
        let mut telemetry = Self::default();
        for layer in &weights.layers {
            if layer.down_proj.is_some() {
                if layer.gate_up_packed.is_some() {
                    telemetry.dense_ffn_gate_up_packed_layers =
                        telemetry.dense_ffn_gate_up_packed_layers.saturating_add(1);
                } else if let Some(gate) = layer.gate_proj.as_ref()
                    && layer.up_proj.is_some()
                    && gate.bits != 5
                {
                    // 5-bit gate/up weights intentionally skip packing (not yet validated);
                    // only count bits≠5 split layers as unexpected hotpath fallbacks.
                    telemetry.dense_ffn_split_gate_up_layers =
                        telemetry.dense_ffn_split_gate_up_layers.saturating_add(1);
                }
            }
            // Track QKV packing for dense attention layers
            if layer.q_proj.is_some()
                || layer.k_proj.is_some()
                || layer.v_proj.is_some()
                || layer.qkv_packed.is_some()
            {
                if layer.qkv_packed.is_some() {
                    telemetry.dense_attention_qkv_packed_layers = telemetry
                        .dense_attention_qkv_packed_layers
                        .saturating_add(1);
                } else {
                    telemetry.dense_attention_split_qkv_layers =
                        telemetry.dense_attention_split_qkv_layers.saturating_add(1);
                }
            }
            if let Some(la) = layer.linear_attn.as_ref() {
                if la.in_proj_qkvz.is_some() && la.in_proj_ba.is_some() {
                    telemetry.linear_attention_qkvz_ba_packed_layers = telemetry
                        .linear_attention_qkvz_ba_packed_layers
                        .saturating_add(1);
                } else if la.in_proj_qkv.is_some()
                    && la.in_proj_z.is_some()
                    && la.in_proj_a.is_some()
                    && la.in_proj_b.is_some()
                {
                    telemetry.linear_attention_split_qkvba_layers = telemetry
                        .linear_attention_split_qkvba_layers
                        .saturating_add(1);
                }
            }
        }
        telemetry
    }

    pub(crate) fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
        decisions.upsert_route_decision(
            "ax_mlx_dense_ffn_gate_up_packed_layers",
            self.dense_ffn_gate_up_packed_layers,
        );
        decisions.upsert_route_decision(
            "ax_mlx_dense_ffn_split_gate_up_layers",
            self.dense_ffn_split_gate_up_layers,
        );
        decisions.upsert_route_decision(
            "ax_mlx_dense_attention_qkv_packed_layers",
            self.dense_attention_qkv_packed_layers,
        );
        decisions.upsert_route_decision(
            "ax_mlx_dense_attention_split_qkv_layers",
            self.dense_attention_split_qkv_layers,
        );
        decisions.upsert_route_decision(
            "ax_mlx_linear_attention_qkvz_ba_packed_layers",
            self.linear_attention_qkvz_ba_packed_layers,
        );
        decisions.upsert_route_decision(
            "ax_mlx_linear_attention_split_qkvba_layers",
            self.linear_attention_split_qkvba_layers,
        );
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Gemma4AssistantMtpTelemetry {
    pub(crate) draft_tokens: u32,
    pub(crate) accepted_tokens: u32,
    pub(crate) rejected_tokens: u32,
    pub(crate) corrections: u32,
    pub(crate) verify_forward_wall_us: u32,
    pub(crate) verify_eval_wall_us: u32,
    pub(crate) draft_forward_wall_us: u32,
}

impl Gemma4AssistantMtpTelemetry {
    pub(crate) fn record_submitted(&mut self, drafted: usize, draft_forward_wall_us: u32) {
        if drafted == 0 {
            return;
        }
        self.draft_tokens = self.draft_tokens.saturating_add(saturating_u32(drafted));
        self.draft_forward_wall_us = self
            .draft_forward_wall_us
            .saturating_add(draft_forward_wall_us);
    }

    pub(crate) fn record_verified(
        &mut self,
        drafted: usize,
        accepted: usize,
        verify_forward_wall_us: u32,
        verify_eval_wall_us: u32,
    ) {
        self.accepted_tokens = self
            .accepted_tokens
            .saturating_add(saturating_u32(accepted));
        let rejected = drafted.saturating_sub(accepted);
        self.rejected_tokens = self
            .rejected_tokens
            .saturating_add(saturating_u32(rejected));
        if rejected > 0 {
            self.corrections = self.corrections.saturating_add(1);
        }
        self.verify_forward_wall_us = self
            .verify_forward_wall_us
            .saturating_add(verify_forward_wall_us);
        self.verify_eval_wall_us = self.verify_eval_wall_us.saturating_add(verify_eval_wall_us);
    }

    pub(crate) fn merge_from(&mut self, other: Self) {
        self.draft_tokens = self.draft_tokens.saturating_add(other.draft_tokens);
        self.accepted_tokens = self.accepted_tokens.saturating_add(other.accepted_tokens);
        self.rejected_tokens = self.rejected_tokens.saturating_add(other.rejected_tokens);
        self.corrections = self.corrections.saturating_add(other.corrections);
        self.verify_forward_wall_us = self
            .verify_forward_wall_us
            .saturating_add(other.verify_forward_wall_us);
        self.verify_eval_wall_us = self
            .verify_eval_wall_us
            .saturating_add(other.verify_eval_wall_us);
        self.draft_forward_wall_us = self
            .draft_forward_wall_us
            .saturating_add(other.draft_forward_wall_us);
    }

    pub(crate) fn accept_rate_x1000(&self) -> u32 {
        self.accepted_tokens
            .saturating_mul(1000)
            .checked_div(self.draft_tokens)
            .unwrap_or(0)
    }
}

impl Gemma4AssistantMtpStatus {
    pub(crate) fn append_route_decisions(
        &self,
        telemetry: Gemma4AssistantMtpTelemetry,
        decisions: &mut impl RouteDecisionSink,
    ) {
        decisions.upsert_route_decision(
            "ax_mlx_gemma4_assistant_mtp_configured",
            u32::from(self.configured),
        );
        decisions.upsert_route_decision(
            "ax_mlx_gemma4_assistant_mtp_validated",
            u32::from(self.validated),
        );
        decisions.upsert_route_decision(
            "ax_mlx_gemma4_assistant_mtp_enabled",
            u32::from(self.enabled),
        );
        decisions.upsert_route_decision(
            "ax_mlx_gemma4_assistant_mtp_attach_failed",
            u32::from(self.attach_failed),
        );
        decisions.upsert_route_decision(
            "ax_mlx_gemma4_assistant_mtp_disable_reason",
            self.disable_reason.route_code(),
        );
        decisions.upsert_route_decision(
            "ax_mlx_gemma4_assistant_mtp_depth",
            saturating_u32(self.max_depth),
        );
        decisions.upsert_route_decision(
            "ax_mlx_gemma4_assistant_mtp_confidence_mode",
            gemma4_assistant_mtp_confidence_mode_from_env().route_code(),
        );
        // Resolved speculation profile (ADR-022) — route-visible so benchmarks
        // and logs prove which posture ran.
        decisions.upsert_route_decision(
            "ax_mlx_speculation_profile",
            speculation_profile_from_env().route_code(),
        );
        decisions.upsert_route_decision(
            "ax_mlx_gemma4_assistant_mtp_draft_tokens",
            telemetry.draft_tokens,
        );
        decisions.upsert_route_decision(
            "ax_mlx_gemma4_assistant_mtp_accepted_tokens",
            telemetry.accepted_tokens,
        );
        decisions.upsert_route_decision(
            "ax_mlx_gemma4_assistant_mtp_rejected_tokens",
            telemetry.rejected_tokens,
        );
        decisions.upsert_route_decision(
            "ax_mlx_gemma4_assistant_mtp_corrections",
            telemetry.corrections,
        );
        decisions.upsert_route_decision(
            "ax_mlx_gemma4_assistant_mtp_accept_rate_x1000",
            telemetry.accept_rate_x1000(),
        );
        decisions.upsert_route_decision(
            "ax_mlx_gemma4_assistant_mtp_verify_forward_wall_us",
            telemetry.verify_forward_wall_us,
        );
        decisions.upsert_route_decision(
            "ax_mlx_gemma4_assistant_mtp_verify_eval_wall_us",
            telemetry.verify_eval_wall_us,
        );
        decisions.upsert_route_decision(
            "ax_mlx_gemma4_assistant_mtp_draft_forward_wall_us",
            telemetry.draft_forward_wall_us,
        );
    }
}

/// Affine quantization bit-width summary computed once at `MlxRunner` construction.
///
/// Tracks per-bit tensor counts and the min/max bit width across all affine-quantized
/// tensors in the manifest. Emitted every step so benchmark artifacts can record
/// quantization recipe details without re-reading the manifest.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]

pub(crate) struct AffineQuantBitsTelemetry {
    /// Total count of affine-quantized tensors (U32 dtype with quantization metadata).
    pub(crate) affine_tensor_count: u32,
    /// Minimum bit width across all affine tensors; 0 when no affine tensors.
    pub(crate) min_affine_bits: u32,
    /// Maximum bit width across all affine tensors; 0 when no affine tensors.
    pub(crate) max_affine_bits: u32,
    /// Per-bit tensor counts for the bit widths MLX supports.
    pub(crate) affine_2bit_count: u32,
    pub(crate) affine_3bit_count: u32,
    pub(crate) affine_4bit_count: u32,
    pub(crate) affine_5bit_count: u32,
    pub(crate) affine_6bit_count: u32,
    pub(crate) affine_8bit_count: u32,
    /// 1 when `AX_ENGINE_3BIT_EXPERIMENTAL=1` was set at load time, else 0.
    pub(crate) experimental_3bit_gate: u32,
}

impl AffineQuantBitsTelemetry {
    pub(crate) fn from_specs(specs: &[ax_engine_core::NativeTensorSpec]) -> Self {
        let mut t = Self {
            experimental_3bit_gate: u32::from(
                std::env::var(ax_engine_core::AX_ENGINE_3BIT_EXPERIMENTAL_ENV).as_deref()
                    == Ok("1"),
            ),
            ..Default::default()
        };
        for spec in specs {
            let Some(q) = &spec.quantization else {
                continue;
            };
            if q.mode != "affine" {
                continue;
            }
            t.affine_tensor_count = t.affine_tensor_count.saturating_add(1);
            let bits = q.bits;
            if t.min_affine_bits == 0 || bits < t.min_affine_bits {
                t.min_affine_bits = bits;
            }
            if bits > t.max_affine_bits {
                t.max_affine_bits = bits;
            }
            match bits {
                2 => t.affine_2bit_count = t.affine_2bit_count.saturating_add(1),
                3 => t.affine_3bit_count = t.affine_3bit_count.saturating_add(1),
                4 => t.affine_4bit_count = t.affine_4bit_count.saturating_add(1),
                5 => t.affine_5bit_count = t.affine_5bit_count.saturating_add(1),
                6 => t.affine_6bit_count = t.affine_6bit_count.saturating_add(1),
                8 => t.affine_8bit_count = t.affine_8bit_count.saturating_add(1),
                _ => {}
            }
        }
        t
    }

    pub(crate) fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
        if self.affine_tensor_count > 0 {
            decisions.upsert_route_decision("ax_mlx_affine_tensor_count", self.affine_tensor_count);
            decisions.upsert_route_decision("ax_mlx_affine_min_bits", self.min_affine_bits);
            decisions.upsert_route_decision("ax_mlx_affine_max_bits", self.max_affine_bits);
        }
        if self.affine_2bit_count > 0 {
            decisions.upsert_route_decision("ax_mlx_affine_2bit_count", self.affine_2bit_count);
        }
        if self.affine_3bit_count > 0 {
            decisions.upsert_route_decision("ax_mlx_affine_3bit_count", self.affine_3bit_count);
        }
        if self.affine_4bit_count > 0 {
            decisions.upsert_route_decision("ax_mlx_affine_4bit_count", self.affine_4bit_count);
        }
        if self.affine_5bit_count > 0 {
            decisions.upsert_route_decision("ax_mlx_affine_5bit_count", self.affine_5bit_count);
        }
        if self.affine_6bit_count > 0 {
            decisions.upsert_route_decision("ax_mlx_affine_6bit_count", self.affine_6bit_count);
        }
        if self.affine_8bit_count > 0 {
            decisions.upsert_route_decision("ax_mlx_affine_8bit_count", self.affine_8bit_count);
        }
        // Always emit gate state so artifacts explicitly record experimental mode status.
        decisions
            .upsert_route_decision("ax_mlx_experimental_3bit_gate", self.experimental_3bit_gate);
    }
}
