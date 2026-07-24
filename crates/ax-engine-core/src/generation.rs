//! Generation paradigm contracts (ADR-038).
//!
//! Separates *how tokens become visible* from *what layers compute*.
//! Strategy metadata is static data — no dyn objects on hot paths.

use crate::model::NativeModelManifest;

/// High-level generation paradigm for a loaded model.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GenerationKind {
    /// Causal next-token (or multi-token speculative) decode.
    Autoregressive,
    /// Block-diffusion canvas denoise + causal commit (e.g. DiffusionGemma).
    BlockDiffusion,
    /// Bidirectional encoder + pooling (embedding models).
    EncoderEmbed,
}

impl GenerationKind {
    /// Derive generation kind from a native manifest without loading weights.
    ///
    /// Prefers structural signals (`diffusion` block, embedding family) over
    /// product labels, while still recognizing well-known family labels so
    /// older manifests remain classified correctly.
    pub fn from_manifest(manifest: &NativeModelManifest) -> Self {
        if is_encoder_embed_manifest(manifest) {
            return Self::EncoderEmbed;
        }
        if is_block_diffusion_manifest(manifest) {
            return Self::BlockDiffusion;
        }
        Self::Autoregressive
    }

    /// Stable wire / telemetry code for route decisions (`u32` crossover values).
    pub const fn telemetry_code(self) -> u32 {
        match self {
            Self::Autoregressive => 0,
            Self::BlockDiffusion => 1,
            Self::EncoderEmbed => 2,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Autoregressive => "autoregressive",
            Self::BlockDiffusion => "block_diffusion",
            Self::EncoderEmbed => "encoder_embed",
        }
    }
}

/// Scheduler / runner work-unit vocabulary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WorkUnitKind {
    PrefillChunk,
    TokenDecode,
    DenoiseStep,
    BlockCommit,
    EmbedForward,
}

/// Metrics boundary for the first user-visible output event.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FirstVisibleEventKind {
    /// AR TTFT: prefill + first next-token.
    FirstToken,
    /// Diffusion: first committed block.
    FirstBlock,
    /// Embedding path produces a vector, not a token stream.
    Embedding,
}

/// Lightweight request progress used for strategy planning (Phase 2).
///
/// Counts are logical; runners map them onto their own state machines.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GenerationProgress {
    /// Prompt tokens already consumed by prefill (0..prompt_len).
    pub processed_prompt_tokens: u32,
    pub prompt_len: u32,
    /// Visible output tokens (AR) or committed canvas tokens (diffusion).
    pub generated_visible_tokens: u32,
    /// Denoise steps completed in the current diffusion block (0 if N/A).
    pub denoise_steps_in_block: u32,
    /// Runner signals the canvas is ready for causal commit (diffusion only).
    pub commit_ready: bool,
    /// Whether the current diffusion block has been committed.
    pub block_committed: bool,
}

impl GenerationProgress {
    /// Build progress from scheduler-facing request counters.
    ///
    /// Diffusion-specific fields (`commit_ready`, `block_committed`, denoise
    /// steps) stay at defaults unless the runner overlays them.
    pub fn from_request_counters(
        processed_prompt_tokens: u32,
        prompt_len: u32,
        generated_visible_tokens: u32,
    ) -> Self {
        Self {
            processed_prompt_tokens,
            prompt_len,
            generated_visible_tokens,
            denoise_steps_in_block: 0,
            commit_ready: false,
            block_committed: false,
        }
    }
}

impl WorkUnitKind {
    /// Stable telemetry code for route decisions.
    pub const fn telemetry_code(self) -> u32 {
        match self {
            Self::PrefillChunk => 0,
            Self::TokenDecode => 1,
            Self::DenoiseStep => 2,
            Self::BlockCommit => 3,
            Self::EmbedForward => 4,
        }
    }
}

/// Static metadata for a generation paradigm (no runtime state).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GenerationStrategyDescriptor {
    pub kind: GenerationKind,
    pub first_visible: FirstVisibleEventKind,
}

impl GenerationStrategyDescriptor {
    pub const fn for_kind(kind: GenerationKind) -> Self {
        let first_visible = match kind {
            GenerationKind::Autoregressive => FirstVisibleEventKind::FirstToken,
            GenerationKind::BlockDiffusion => FirstVisibleEventKind::FirstBlock,
            GenerationKind::EncoderEmbed => FirstVisibleEventKind::Embedding,
        };
        Self {
            kind,
            first_visible,
        }
    }

    pub fn from_manifest(manifest: &NativeModelManifest) -> Self {
        Self::for_kind(GenerationKind::from_manifest(manifest))
    }

    /// Canonical work-unit sequence shape for documentation and planners.
    pub const fn default_work_units(self) -> &'static [WorkUnitKind] {
        match self.kind {
            GenerationKind::Autoregressive => {
                &[WorkUnitKind::PrefillChunk, WorkUnitKind::TokenDecode]
            }
            GenerationKind::BlockDiffusion => &[
                WorkUnitKind::PrefillChunk,
                WorkUnitKind::DenoiseStep,
                WorkUnitKind::BlockCommit,
            ],
            GenerationKind::EncoderEmbed => &[WorkUnitKind::EmbedForward],
        }
    }

    /// Plan the next work unit from strategy metadata + request progress.
    ///
    /// This is the Phase 2 strategy boundary: callers do not hard-code
    /// family-specific step shapes when they can use this planner.
    pub fn plan_next_work_unit(self, progress: GenerationProgress) -> WorkUnitKind {
        match self.kind {
            GenerationKind::EncoderEmbed => WorkUnitKind::EmbedForward,
            GenerationKind::Autoregressive => {
                if progress.processed_prompt_tokens < progress.prompt_len {
                    WorkUnitKind::PrefillChunk
                } else {
                    WorkUnitKind::TokenDecode
                }
            }
            GenerationKind::BlockDiffusion => {
                if progress.processed_prompt_tokens < progress.prompt_len {
                    WorkUnitKind::PrefillChunk
                } else if progress.block_committed {
                    // Next block starts with denoise on a fresh canvas.
                    WorkUnitKind::DenoiseStep
                } else if progress.commit_ready {
                    WorkUnitKind::BlockCommit
                } else {
                    WorkUnitKind::DenoiseStep
                }
            }
        }
    }

    /// Whether this strategy emits a token stream (vs a single embedding).
    pub const fn emits_token_stream(self) -> bool {
        !matches!(self.kind, GenerationKind::EncoderEmbed)
    }

    /// Metrics label for the first user-visible event (stable string).
    pub const fn first_visible_metric_label(self) -> &'static str {
        match self.first_visible {
            FirstVisibleEventKind::FirstToken => "ttft_first_token",
            FirstVisibleEventKind::FirstBlock => "time_to_first_block",
            FirstVisibleEventKind::Embedding => "embedding_latency",
        }
    }
}

pub(crate) fn is_encoder_embed_manifest(manifest: &NativeModelManifest) -> bool {
    matches!(
        manifest.model_family.as_str(),
        "embeddinggemma" | "gemma3_text"
    )
}

pub(crate) fn is_block_diffusion_manifest(manifest: &NativeModelManifest) -> bool {
    manifest.diffusion.is_enabled() || manifest.model_family == "diffusion_gemma"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION, NativeDiffusionConfig,
        NativeLinearAttentionConfig, NativeModelManifest, NativeMoeConfig, NativeRuntimeStatus,
        NativeTensorFormat, WeightSanitize,
    };

    fn base_manifest(family: &str) -> NativeModelManifest {
        NativeModelManifest {
            schema_version: AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: family.to_string(),
            tensor_format: NativeTensorFormat::Safetensors,
            source_quantization: None,
            runtime_status: NativeRuntimeStatus::default(),
            layer_count: 2,
            hidden_size: 128,
            intermediate_size: 256,
            attention_head_count: 4,
            attention_head_dim: 32,
            kv_head_count: 2,
            vocab_size: 1000,
            tie_word_embeddings: false,
            rope_theta: None,
            rope_theta_swa: None,
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_low_freq_factor: None,
            rope_high_freq_factor: None,
            rope_original_context_len: None,
            no_rope_layer_interval: 0,
            attn_temperature_floor: None,
            attn_temperature_scale: None,
            intermediate_size_mlp: 0,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            attn_output_gate: false,
            partial_rotary_factor: None,
            rms_norm_eps: None,
            attention_value_from_key_layers: Vec::new(),
            attention_v_norm_no_scale_layers: Vec::new(),
            global_head_dim: None,
            sliding_window_size: None,
            layer_types: Vec::new(),
            kv_shared_source_layers: Default::default(),
            final_logit_softcapping: None,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            linear_attention: NativeLinearAttentionConfig::default(),
            mla_attention: Default::default(),
            moe: NativeMoeConfig::default(),
            glm_router: Default::default(),
            weight_sanitize: WeightSanitize::default(),
            think_start_token_id: None,
            think_end_token_id: None,
            diffusion: NativeDiffusionConfig::default(),
            dropped_tensors: Default::default(),
            tensors: Vec::new(),
        }
    }

    #[test]
    fn generation_kind_qwen_is_autoregressive() {
        let m = base_manifest("qwen3");
        assert_eq!(
            GenerationKind::from_manifest(&m),
            GenerationKind::Autoregressive
        );
        assert_eq!(
            GenerationStrategyDescriptor::from_manifest(&m).first_visible,
            FirstVisibleEventKind::FirstToken
        );
    }

    #[test]
    fn generation_kind_diffusion_from_config_or_family() {
        let by_family = base_manifest("diffusion_gemma");
        assert_eq!(
            GenerationKind::from_manifest(&by_family),
            GenerationKind::BlockDiffusion
        );

        let mut by_config = base_manifest("gemma4");
        by_config.diffusion = NativeDiffusionConfig {
            canvas_size: Some(256),
            ..Default::default()
        };
        assert_eq!(
            GenerationKind::from_manifest(&by_config),
            GenerationKind::BlockDiffusion
        );
        assert_eq!(
            GenerationStrategyDescriptor::from_manifest(&by_config).first_visible,
            FirstVisibleEventKind::FirstBlock
        );
    }

    #[test]
    fn generation_kind_embedding() {
        let m = base_manifest("embeddinggemma");
        assert_eq!(
            GenerationKind::from_manifest(&m),
            GenerationKind::EncoderEmbed
        );
        assert_eq!(
            GenerationStrategyDescriptor::from_manifest(&m).first_visible,
            FirstVisibleEventKind::Embedding
        );
    }

    #[test]
    fn telemetry_codes_are_stable() {
        assert_eq!(GenerationKind::Autoregressive.telemetry_code(), 0);
        assert_eq!(GenerationKind::BlockDiffusion.telemetry_code(), 1);
        assert_eq!(GenerationKind::EncoderEmbed.telemetry_code(), 2);
    }

    #[test]
    fn strategy_plans_ar_prefill_then_decode() {
        let strat = GenerationStrategyDescriptor::for_kind(GenerationKind::Autoregressive);
        assert_eq!(
            strat.default_work_units(),
            &[WorkUnitKind::PrefillChunk, WorkUnitKind::TokenDecode]
        );
        let prefill = GenerationProgress {
            processed_prompt_tokens: 0,
            prompt_len: 128,
            ..Default::default()
        };
        assert_eq!(
            strat.plan_next_work_unit(prefill),
            WorkUnitKind::PrefillChunk
        );
        let decode = GenerationProgress {
            processed_prompt_tokens: 128,
            prompt_len: 128,
            generated_visible_tokens: 1,
            ..Default::default()
        };
        assert_eq!(strat.plan_next_work_unit(decode), WorkUnitKind::TokenDecode);
        assert_eq!(strat.first_visible_metric_label(), "ttft_first_token");
    }

    #[test]
    fn strategy_plans_diffusion_denoise_and_commit_shape() {
        let strat = GenerationStrategyDescriptor::for_kind(GenerationKind::BlockDiffusion);
        assert_eq!(
            strat.default_work_units(),
            &[
                WorkUnitKind::PrefillChunk,
                WorkUnitKind::DenoiseStep,
                WorkUnitKind::BlockCommit
            ]
        );
        let after_prefill = GenerationProgress {
            processed_prompt_tokens: 64,
            prompt_len: 64,
            denoise_steps_in_block: 0,
            block_committed: false,
            ..Default::default()
        };
        assert_eq!(
            strat.plan_next_work_unit(after_prefill),
            WorkUnitKind::DenoiseStep
        );
        // Until the runner marks commit_ready, planning stays on denoise.
        let mid_denoise = GenerationProgress {
            processed_prompt_tokens: 64,
            prompt_len: 64,
            denoise_steps_in_block: 5,
            commit_ready: false,
            block_committed: false,
            ..Default::default()
        };
        assert_eq!(
            strat.plan_next_work_unit(mid_denoise),
            WorkUnitKind::DenoiseStep
        );
        let ready = GenerationProgress {
            processed_prompt_tokens: 64,
            prompt_len: 64,
            denoise_steps_in_block: 12,
            commit_ready: true,
            block_committed: false,
            ..Default::default()
        };
        assert_eq!(strat.plan_next_work_unit(ready), WorkUnitKind::BlockCommit);
        assert_eq!(strat.first_visible_metric_label(), "time_to_first_block");
    }

    #[test]
    fn strategy_plans_embed_only() {
        let strat = GenerationStrategyDescriptor::for_kind(GenerationKind::EncoderEmbed);
        assert_eq!(strat.default_work_units(), &[WorkUnitKind::EmbedForward]);
        assert_eq!(
            strat.plan_next_work_unit(GenerationProgress::default()),
            WorkUnitKind::EmbedForward
        );
        assert!(!strat.emits_token_stream());
        assert_eq!(strat.first_visible_metric_label(), "embedding_latency");
    }

    #[test]
    fn runtime_classification_uses_generation_kind_not_family_alone() {
        // A gemma4 family label with diffusion config is BlockDiffusion.
        let mut m = base_manifest("gemma4");
        m.diffusion.canvas_size = Some(256);
        let kind = GenerationKind::from_manifest(&m);
        assert_eq!(kind, GenerationKind::BlockDiffusion);
        assert_ne!(m.model_family.as_str(), "diffusion_gemma");
        let strat = GenerationStrategyDescriptor::for_kind(kind);
        assert_eq!(strat.first_visible, FirstVisibleEventKind::FirstBlock);
    }

    #[test]
    fn progress_from_request_counters_drives_ar_planner() {
        let strat = GenerationStrategyDescriptor::for_kind(GenerationKind::Autoregressive);
        let prefill = GenerationProgress::from_request_counters(10, 100, 0);
        assert_eq!(
            strat.plan_next_work_unit(prefill),
            WorkUnitKind::PrefillChunk
        );
        let decode = GenerationProgress::from_request_counters(100, 100, 3);
        assert_eq!(strat.plan_next_work_unit(decode), WorkUnitKind::TokenDecode);
        assert_eq!(WorkUnitKind::PrefillChunk.telemetry_code(), 0);
        assert_eq!(WorkUnitKind::TokenDecode.telemetry_code(), 1);
        assert_eq!(WorkUnitKind::DenoiseStep.telemetry_code(), 2);
    }
}
