use mlx_sys::{MlxArray, eval};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Gemma4MoeProfileSnapshot {
    pub enabled: u32,
    pub decode_layers: u32,
    pub topk_selections: u32,
    pub sorted_gather_layers: u32,
    pub unsorted_gather_layers: u32,
    pub attention_wall_us: u32,
    pub dense_wall_us: u32,
    pub router_wall_us: u32,
    pub expert_wall_us: u32,
    pub post_wall_us: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LinearAttentionProfileSnapshot {
    pub enabled: u32,
    pub layers: u32,
    pub tokens: u32,
    pub direct_cpp_inputs_attempts: u32,
    pub direct_cpp_inputs_hits: u32,
    pub direct_cpp_inputs_fallbacks: u32,
    pub direct_cpp_inputs_profile_blocked: u32,
    pub direct_cpp_post_input_attempts: u32,
    pub direct_cpp_post_input_hits: u32,
    pub direct_cpp_post_input_fallbacks: u32,
    pub direct_cpp_post_input_profile_blocked: u32,
    pub decode_post_input_metal_attempts: u32,
    pub decode_post_input_metal_hits: u32,
    pub decode_post_input_metal_fallbacks: u32,
    pub decode_post_input_metal_profile_blocked: u32,
    pub projection_wall_us: u32,
    pub projection_qkvz_wall_us: u32,
    pub projection_ba_wall_us: u32,
    pub projection_qkv_wall_us: u32,
    pub projection_z_wall_us: u32,
    pub projection_a_wall_us: u32,
    pub projection_b_wall_us: u32,
    pub conv_wall_us: u32,
    pub qk_norm_wall_us: u32,
    pub recurrent_wall_us: u32,
    pub output_wall_us: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrefillProfileSnapshot {
    pub enabled: u32,
    pub prefill_steps: u32,
    pub layers: u32,
    pub tokens: u32,
    pub per_layer_input_wall_us: u32,
    pub pre_sdpa_wall_us: u32,
    pub pre_sdpa_qkv_proj_wall_us: u32,
    pub pre_sdpa_qk_norm_wall_us: u32,
    pub pre_sdpa_rope_kv_wall_us: u32,
    pub sdpa_wall_us: u32,
    pub post_attn_wall_us: u32,
    pub post_attn_ffn_wall_us: u32,
    pub post_attn_ffn_gate_up_wall_us: u32,
    pub post_attn_ffn_activation_wall_us: u32,
    pub post_attn_ffn_down_wall_us: u32,
    pub post_attn_output_proj_wall_us: u32,
    pub post_attn_residual_norm_wall_us: u32,
    pub post_attn_residual_gate_wall_us: u32,
    pub lm_head_wall_us: u32,
    pub moe_router_wall_us: u32,
    pub moe_expert_gate_up_wall_us: u32,
    pub moe_expert_activation_wall_us: u32,
    pub moe_expert_down_wall_us: u32,
    pub moe_expert_weighted_sum_wall_us: u32,
    pub moe_shared_expert_wall_us: u32,
}

/// Family-neutral MoE sub-stage profiling snapshot.
///
/// Enabled via `AX_MLX_MOE_PROFILE=1`. Unlike `DecodeProfileSnapshot` which
/// forces blocking `eval()` barriers at every stage, this snapshot records
/// lightweight wall-clock deltas between existing evaluation points inside
/// `moe_experts_forward_impl`. The ratios between sub-stages indicate where
/// dispatch overhead dominates.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MoeProfileSnapshot {
    pub enabled: u32,
    pub moe_layers: u32,
    pub router_us: u32,
    pub expert_gate_up_us: u32,
    pub expert_activation_us: u32,
    pub expert_down_us: u32,
    pub weighted_sum_us: u32,
    pub shared_expert_us: u32,
    pub total_us: u32,
}

/// Per-section wall time for the single-token lazy decode path.
///
/// Enabled via `AX_MLX_DECODE_PROFILE=1`.  Each stage timing forces a blocking
/// `eval()` to materialise the lazy graph at that point, so enabling the
/// profile **disables decode pipelining** and inflates step time relative to
/// production.  The ratios between stages are what matters for diagnosis.
/// Profiling must still preserve production route selection inside a stage
/// where possible; for example Gemma packed GeGLU Metal remains active and is
/// timed as the FFN activation substage instead of being replaced by the split
/// GEGLU fallback.
///
/// Stages mirror the decode path:
/// - `per_layer_input_wall_us` — `compute_per_layer_inputs_arr` (Gemma4 2B/4B
///   per-layer embed + project + RMSNorm + combine + slice).
/// - `pre_sdpa_wall_us` — qkv_project + reshape + QK/V norm + transpose +
///   RoPE + KV append. Dense full-attention layers only; linear-attention and
///   MLA layers contribute zero here.
/// - `pre_sdpa_qkv_proj_wall_us` — subset of pre_sdpa: just the
///   `qkv_project` call (Q/K/V quantized matmul(s), with the subsequent
///   reshape to BSHD).  Use `pre_sdpa - pre_sdpa_qkv_proj` to size the
///   "tail" (QK/V norm + transpose + RoPE + KV append) for ablation.
/// - `pre_sdpa_qk_norm_wall_us` — subset of pre_sdpa after QKV projection:
///   Q/K RMSNorm on BSHD tensors.
/// - `pre_sdpa_rope_kv_wall_us` — subset of pre_sdpa after QK norm: transpose,
///   V normalization, RoPE, and KV append.
/// - `sdpa_wall_us` — `scaled_dot_product_attention_with_mask` (or TurboQuant
///   fused fallback). Dense full-attention layers only.
/// - `post_attn_wall_us` — transpose-back, output projection, residual,
///   FFN/MoE, per-layer-input gating, layer_scalar.  Includes the full layer
///   tail for linear/MLA layers (since pre_sdpa/sdpa do not apply).
/// - `post_attn_ffn_wall_us` — subset of post_attn: just the FFN section
///   (`ffn_swiglu` for dense, MoE expert forward for sparse).  Excludes the
///   surrounding attention tail (transpose+o_proj+post_norm), residuals,
///   per-layer-input gating, and layer_scalar.  Use `post_attn - post_attn_ffn`
///   to size the non-FFN portion.
/// - `post_attn_output_proj_wall_us` — subset of post_attn: transpose-back,
///   reshape, attention output projection, optional output gate, and optional
///   post-attention norm.
/// - `post_attn_residual_norm_wall_us` — subset of post_attn: attention
///   residual add plus pre-FFN RMSNorm.
/// - `post_attn_residual_gate_wall_us` — subset of post_attn: FFN residual add,
///   optional Gemma per-layer-input gate/projection/norm, and optional layer
///   scalar.
/// - `lm_head_wall_us` — final RMSNorm + lm_head matmul + softcap + reshape.
/// - `moe_router_wall_us` — subset of post_attn_ffn: just the MoE router
///   (quantized_matmul + softmax + top-k + renorm).
/// - `moe_expert_gate_up_wall_us` — subset of post_attn_ffn: the expert
///   gate/up gather_qmm.
/// - `moe_expert_activation_wall_us` — subset of post_attn_ffn: the SwiGLU
///   activation on the expert gate_up output.
/// - `moe_expert_down_wall_us` — subset of post_attn_ffn: the expert down
///   gather_qmm.
/// - `moe_expert_weighted_sum_wall_us` — subset of post_attn_ffn: the
///   weighted-sum kernel (or MLX fallback) combining expert outputs.
/// - `moe_shared_expert_wall_us` — subset of post_attn_ffn: the shared
///   expert forward (gate/up/down projections + gate activation + multiply).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DecodeProfileSnapshot {
    pub enabled: u32,
    pub decode_steps: u32,
    pub layers: u32,
    pub per_layer_input_wall_us: u32,
    pub pre_sdpa_wall_us: u32,
    pub pre_sdpa_qkv_proj_wall_us: u32,
    pub pre_sdpa_qk_norm_wall_us: u32,
    pub pre_sdpa_rope_kv_wall_us: u32,
    pub sdpa_wall_us: u32,
    pub post_attn_wall_us: u32,
    pub post_attn_ffn_wall_us: u32,
    pub post_attn_ffn_gate_up_wall_us: u32,
    pub post_attn_ffn_activation_wall_us: u32,
    pub post_attn_ffn_down_wall_us: u32,
    pub post_attn_output_proj_wall_us: u32,
    pub post_attn_residual_norm_wall_us: u32,
    pub post_attn_residual_gate_wall_us: u32,
    pub lm_head_wall_us: u32,
    pub moe_router_wall_us: u32,
    pub moe_expert_gate_up_wall_us: u32,
    pub moe_expert_activation_wall_us: u32,
    pub moe_expert_down_wall_us: u32,
    pub moe_expert_weighted_sum_wall_us: u32,
    pub moe_shared_expert_wall_us: u32,
}

#[derive(Clone, Copy)]
pub(super) enum Gemma4MoeProfileStage {
    Attention,
    Dense,
    Router,
    Expert,
    Post,
}

#[derive(Clone, Copy)]
pub(super) enum LinearAttentionProfileStage {
    Projection,
    ProjectionQkvz,
    ProjectionBa,
    ProjectionQkv,
    ProjectionZ,
    ProjectionA,
    ProjectionB,
    Conv,
    QkNorm,
    Recurrent,
    Output,
}

#[derive(Clone, Copy)]
pub(crate) enum DecodeProfileStage {
    PerLayerInput,
    PreSdpa,
    PreSdpaQkvProj,
    PreSdpaQkNorm,
    PreSdpaRopeKv,
    Sdpa,
    PostAttn,
    PostAttnFfn,
    PostAttnFfnGateUp,
    PostAttnFfnActivation,
    PostAttnFfnDown,
    PostAttnOutputProj,
    PostAttnResidualNorm,
    PostAttnResidualGate,
    LmHead,
    MoeRouter,
    MoeExpertGateUp,
    MoeExpertActivation,
    MoeExpertDown,
    MoeExpertWeightedSum,
    MoeSharedExpert,
}

static GEMMA4_MOE_PROFILE: OnceLock<Mutex<Gemma4MoeProfileSnapshot>> = OnceLock::new();
static LINEAR_ATTENTION_PROFILE: OnceLock<Mutex<LinearAttentionProfileSnapshot>> = OnceLock::new();
static PREFILL_PROFILE: OnceLock<Mutex<PrefillProfileSnapshot>> = OnceLock::new();
static DECODE_PROFILE: OnceLock<Mutex<DecodeProfileSnapshot>> = OnceLock::new();
static MOE_PROFILE: OnceLock<Mutex<MoeProfileSnapshot>> = OnceLock::new();
static GEMMA4_MOE_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
static LINEAR_ATTENTION_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
static PREFILL_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
static DECODE_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
static MOE_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();

fn profile_env_enabled(cache: &'static OnceLock<bool>, name: &'static str) -> bool {
    *cache.get_or_init(|| {
        matches!(
            std::env::var(name).as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        )
    })
}

pub(super) fn gemma4_moe_profile_enabled() -> bool {
    profile_env_enabled(&GEMMA4_MOE_PROFILE_ENABLED, "AX_MLX_GEMMA4_MOE_PROFILE")
}

pub(super) fn linear_attention_profile_enabled() -> bool {
    profile_env_enabled(
        &LINEAR_ATTENTION_PROFILE_ENABLED,
        "AX_MLX_LINEAR_ATTENTION_PROFILE",
    )
}

pub(crate) fn prefill_profile_enabled() -> bool {
    profile_env_enabled(&PREFILL_PROFILE_ENABLED, "AX_MLX_PREFILL_PROFILE")
}

pub(crate) fn decode_profile_enabled() -> bool {
    profile_env_enabled(&DECODE_PROFILE_ENABLED, "AX_MLX_DECODE_PROFILE")
}

pub(crate) fn moe_profile_enabled() -> bool {
    profile_env_enabled(&MOE_PROFILE_ENABLED, "AX_MLX_MOE_PROFILE")
}

fn gemma4_moe_profile() -> &'static Mutex<Gemma4MoeProfileSnapshot> {
    GEMMA4_MOE_PROFILE.get_or_init(|| Mutex::new(Gemma4MoeProfileSnapshot::default()))
}

fn linear_attention_profile() -> &'static Mutex<LinearAttentionProfileSnapshot> {
    LINEAR_ATTENTION_PROFILE.get_or_init(|| Mutex::new(LinearAttentionProfileSnapshot::default()))
}

fn prefill_profile() -> &'static Mutex<PrefillProfileSnapshot> {
    PREFILL_PROFILE.get_or_init(|| Mutex::new(PrefillProfileSnapshot::default()))
}

fn decode_profile() -> &'static Mutex<DecodeProfileSnapshot> {
    DECODE_PROFILE.get_or_init(|| Mutex::new(DecodeProfileSnapshot::default()))
}

fn moe_profile() -> &'static Mutex<MoeProfileSnapshot> {
    MOE_PROFILE.get_or_init(|| Mutex::new(MoeProfileSnapshot::default()))
}

pub(super) fn saturating_profile_us(started: Instant) -> u32 {
    started.elapsed().as_micros().min(u32::MAX as u128) as u32
}

pub(super) fn record_gemma4_moe_decode_layer(topk_selections: usize, sorted_gather: bool) {
    let mut profile = gemma4_moe_profile().lock().unwrap();
    profile.enabled = 1;
    profile.decode_layers = profile.decode_layers.saturating_add(1);
    profile.topk_selections = profile
        .topk_selections
        .saturating_add(topk_selections.min(u32::MAX as usize) as u32);
    if sorted_gather {
        profile.sorted_gather_layers = profile.sorted_gather_layers.saturating_add(1);
    } else {
        profile.unsorted_gather_layers = profile.unsorted_gather_layers.saturating_add(1);
    }
}

pub(super) fn record_gemma4_moe_profile_stage(stage: Gemma4MoeProfileStage, wall_us: u32) {
    let mut profile = gemma4_moe_profile().lock().unwrap();
    profile.enabled = 1;
    let target = match stage {
        Gemma4MoeProfileStage::Attention => &mut profile.attention_wall_us,
        Gemma4MoeProfileStage::Dense => &mut profile.dense_wall_us,
        Gemma4MoeProfileStage::Router => &mut profile.router_wall_us,
        Gemma4MoeProfileStage::Expert => &mut profile.expert_wall_us,
        Gemma4MoeProfileStage::Post => &mut profile.post_wall_us,
    };
    *target = target.saturating_add(wall_us);
}

pub(super) fn record_linear_attention_profile_layer(tokens: i32) {
    let mut profile = linear_attention_profile().lock().unwrap();
    profile.enabled = 1;
    profile.layers = profile.layers.saturating_add(1);
    profile.tokens = profile.tokens.saturating_add(tokens.max(0) as u32);
}

pub(super) fn record_linear_attention_direct_cpp_inputs_attempt() {
    let mut profile = linear_attention_profile().lock().unwrap();
    profile.direct_cpp_inputs_attempts = profile.direct_cpp_inputs_attempts.saturating_add(1);
}

pub(super) fn record_linear_attention_direct_cpp_inputs_hit() {
    let mut profile = linear_attention_profile().lock().unwrap();
    profile.direct_cpp_inputs_hits = profile.direct_cpp_inputs_hits.saturating_add(1);
}

pub(super) fn record_linear_attention_direct_cpp_inputs_fallback() {
    let mut profile = linear_attention_profile().lock().unwrap();
    profile.direct_cpp_inputs_fallbacks = profile.direct_cpp_inputs_fallbacks.saturating_add(1);
}

pub(super) fn record_linear_attention_direct_cpp_inputs_profile_blocked() {
    let mut profile = linear_attention_profile().lock().unwrap();
    profile.direct_cpp_inputs_profile_blocked =
        profile.direct_cpp_inputs_profile_blocked.saturating_add(1);
}

pub(super) fn record_linear_attention_direct_cpp_post_input_attempt() {
    let mut profile = linear_attention_profile().lock().unwrap();
    profile.direct_cpp_post_input_attempts =
        profile.direct_cpp_post_input_attempts.saturating_add(1);
}

pub(super) fn record_linear_attention_direct_cpp_post_input_hit() {
    let mut profile = linear_attention_profile().lock().unwrap();
    profile.direct_cpp_post_input_hits = profile.direct_cpp_post_input_hits.saturating_add(1);
}

pub(super) fn record_linear_attention_direct_cpp_post_input_fallback() {
    let mut profile = linear_attention_profile().lock().unwrap();
    profile.direct_cpp_post_input_fallbacks =
        profile.direct_cpp_post_input_fallbacks.saturating_add(1);
}

pub(super) fn record_linear_attention_direct_cpp_post_input_profile_blocked() {
    let mut profile = linear_attention_profile().lock().unwrap();
    profile.direct_cpp_post_input_profile_blocked = profile
        .direct_cpp_post_input_profile_blocked
        .saturating_add(1);
}

pub(super) fn record_linear_attention_decode_post_input_metal_attempt() {
    let mut profile = linear_attention_profile().lock().unwrap();
    profile.decode_post_input_metal_attempts =
        profile.decode_post_input_metal_attempts.saturating_add(1);
}

pub(super) fn record_linear_attention_decode_post_input_metal_hit() {
    let mut profile = linear_attention_profile().lock().unwrap();
    profile.decode_post_input_metal_hits = profile.decode_post_input_metal_hits.saturating_add(1);
}

pub(super) fn record_linear_attention_decode_post_input_metal_fallback() {
    let mut profile = linear_attention_profile().lock().unwrap();
    profile.decode_post_input_metal_fallbacks =
        profile.decode_post_input_metal_fallbacks.saturating_add(1);
}

pub(super) fn record_linear_attention_decode_post_input_metal_profile_blocked() {
    let mut profile = linear_attention_profile().lock().unwrap();
    profile.decode_post_input_metal_profile_blocked = profile
        .decode_post_input_metal_profile_blocked
        .saturating_add(1);
}

pub(super) fn record_linear_attention_profile_stage(
    stage: LinearAttentionProfileStage,
    wall_us: u32,
) {
    let mut profile = linear_attention_profile().lock().unwrap();
    profile.enabled = 1;
    let target = match stage {
        LinearAttentionProfileStage::Projection => &mut profile.projection_wall_us,
        LinearAttentionProfileStage::ProjectionQkvz => &mut profile.projection_qkvz_wall_us,
        LinearAttentionProfileStage::ProjectionBa => &mut profile.projection_ba_wall_us,
        LinearAttentionProfileStage::ProjectionQkv => &mut profile.projection_qkv_wall_us,
        LinearAttentionProfileStage::ProjectionZ => &mut profile.projection_z_wall_us,
        LinearAttentionProfileStage::ProjectionA => &mut profile.projection_a_wall_us,
        LinearAttentionProfileStage::ProjectionB => &mut profile.projection_b_wall_us,
        LinearAttentionProfileStage::Conv => &mut profile.conv_wall_us,
        LinearAttentionProfileStage::QkNorm => &mut profile.qk_norm_wall_us,
        LinearAttentionProfileStage::Recurrent => &mut profile.recurrent_wall_us,
        LinearAttentionProfileStage::Output => &mut profile.output_wall_us,
    };
    *target = target.saturating_add(wall_us);
}

pub(super) fn record_prefill_profile_stage(stage: DecodeProfileStage, wall_us: u32) {
    let mut profile = prefill_profile().lock().unwrap();
    profile.enabled = 1;
    let target = match stage {
        DecodeProfileStage::PerLayerInput => &mut profile.per_layer_input_wall_us,
        DecodeProfileStage::PreSdpa => &mut profile.pre_sdpa_wall_us,
        DecodeProfileStage::PreSdpaQkvProj => &mut profile.pre_sdpa_qkv_proj_wall_us,
        DecodeProfileStage::PreSdpaQkNorm => &mut profile.pre_sdpa_qk_norm_wall_us,
        DecodeProfileStage::PreSdpaRopeKv => &mut profile.pre_sdpa_rope_kv_wall_us,
        DecodeProfileStage::Sdpa => &mut profile.sdpa_wall_us,
        DecodeProfileStage::PostAttn => &mut profile.post_attn_wall_us,
        DecodeProfileStage::PostAttnFfn => &mut profile.post_attn_ffn_wall_us,
        DecodeProfileStage::PostAttnFfnGateUp => &mut profile.post_attn_ffn_gate_up_wall_us,
        DecodeProfileStage::PostAttnFfnActivation => &mut profile.post_attn_ffn_activation_wall_us,
        DecodeProfileStage::PostAttnFfnDown => &mut profile.post_attn_ffn_down_wall_us,
        DecodeProfileStage::PostAttnOutputProj => &mut profile.post_attn_output_proj_wall_us,
        DecodeProfileStage::PostAttnResidualNorm => &mut profile.post_attn_residual_norm_wall_us,
        DecodeProfileStage::PostAttnResidualGate => &mut profile.post_attn_residual_gate_wall_us,
        DecodeProfileStage::LmHead => &mut profile.lm_head_wall_us,
        DecodeProfileStage::MoeRouter => &mut profile.moe_router_wall_us,
        DecodeProfileStage::MoeExpertGateUp => &mut profile.moe_expert_gate_up_wall_us,
        DecodeProfileStage::MoeExpertActivation => &mut profile.moe_expert_activation_wall_us,
        DecodeProfileStage::MoeExpertDown => &mut profile.moe_expert_down_wall_us,
        DecodeProfileStage::MoeExpertWeightedSum => &mut profile.moe_expert_weighted_sum_wall_us,
        DecodeProfileStage::MoeSharedExpert => &mut profile.moe_shared_expert_wall_us,
    };
    *target = target.saturating_add(wall_us);
}

pub(super) fn record_decode_profile_stage(stage: DecodeProfileStage, wall_us: u32) {
    let mut profile = decode_profile().lock().unwrap();
    profile.enabled = 1;
    let target = match stage {
        DecodeProfileStage::PerLayerInput => &mut profile.per_layer_input_wall_us,
        DecodeProfileStage::PreSdpa => &mut profile.pre_sdpa_wall_us,
        DecodeProfileStage::PreSdpaQkvProj => &mut profile.pre_sdpa_qkv_proj_wall_us,
        DecodeProfileStage::PreSdpaQkNorm => &mut profile.pre_sdpa_qk_norm_wall_us,
        DecodeProfileStage::PreSdpaRopeKv => &mut profile.pre_sdpa_rope_kv_wall_us,
        DecodeProfileStage::Sdpa => &mut profile.sdpa_wall_us,
        DecodeProfileStage::PostAttn => &mut profile.post_attn_wall_us,
        DecodeProfileStage::PostAttnFfn => &mut profile.post_attn_ffn_wall_us,
        DecodeProfileStage::PostAttnFfnGateUp => &mut profile.post_attn_ffn_gate_up_wall_us,
        DecodeProfileStage::PostAttnFfnActivation => &mut profile.post_attn_ffn_activation_wall_us,
        DecodeProfileStage::PostAttnFfnDown => &mut profile.post_attn_ffn_down_wall_us,
        DecodeProfileStage::PostAttnOutputProj => &mut profile.post_attn_output_proj_wall_us,
        DecodeProfileStage::PostAttnResidualNorm => &mut profile.post_attn_residual_norm_wall_us,
        DecodeProfileStage::PostAttnResidualGate => &mut profile.post_attn_residual_gate_wall_us,
        DecodeProfileStage::LmHead => &mut profile.lm_head_wall_us,
        DecodeProfileStage::MoeRouter => &mut profile.moe_router_wall_us,
        DecodeProfileStage::MoeExpertGateUp => &mut profile.moe_expert_gate_up_wall_us,
        DecodeProfileStage::MoeExpertActivation => &mut profile.moe_expert_activation_wall_us,
        DecodeProfileStage::MoeExpertDown => &mut profile.moe_expert_down_wall_us,
        DecodeProfileStage::MoeExpertWeightedSum => &mut profile.moe_expert_weighted_sum_wall_us,
        DecodeProfileStage::MoeSharedExpert => &mut profile.moe_shared_expert_wall_us,
    };
    *target = target.saturating_add(wall_us);
}

pub(super) fn record_prefill_profile_step(layers: u32, tokens: u32) {
    let mut profile = prefill_profile().lock().unwrap();
    profile.enabled = 1;
    profile.prefill_steps = profile.prefill_steps.saturating_add(1);
    profile.layers = profile.layers.saturating_add(layers);
    profile.tokens = profile.tokens.saturating_add(tokens);
}

pub(super) fn record_decode_profile_step(layers: u32) {
    let mut profile = decode_profile().lock().unwrap();
    profile.enabled = 1;
    profile.decode_steps = profile.decode_steps.saturating_add(1);
    profile.layers = profile.layers.saturating_add(layers);
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(crate) enum MoeProfileStage {
    Router,
    ExpertGateUp,
    ExpertActivation,
    ExpertDown,
    WeightedSum,
    SharedExpert,
}

pub(crate) fn record_moe_profile_layer() {
    if !moe_profile_enabled() {
        return;
    }
    let mut profile = moe_profile().lock().unwrap();
    profile.enabled = 1;
    profile.moe_layers = profile.moe_layers.saturating_add(1);
}

pub(crate) fn record_moe_profile_stage(stage: MoeProfileStage, wall_us: u32) {
    if !moe_profile_enabled() {
        return;
    }
    let mut profile = moe_profile().lock().unwrap();
    profile.enabled = 1;
    let target = match stage {
        MoeProfileStage::Router => &mut profile.router_us,
        MoeProfileStage::ExpertGateUp => &mut profile.expert_gate_up_us,
        MoeProfileStage::ExpertActivation => &mut profile.expert_activation_us,
        MoeProfileStage::ExpertDown => &mut profile.expert_down_us,
        MoeProfileStage::WeightedSum => &mut profile.weighted_sum_us,
        MoeProfileStage::SharedExpert => &mut profile.shared_expert_us,
    };
    *target = target.saturating_add(wall_us);
}

pub(crate) fn record_moe_profile_total(wall_us: u32) {
    if !moe_profile_enabled() {
        return;
    }
    let mut profile = moe_profile().lock().unwrap();
    profile.total_us = profile.total_us.saturating_add(wall_us);
}

pub(super) fn profile_eval_elapsed(
    enabled: bool,
    stage: Gemma4MoeProfileStage,
    started: Instant,
    targets: &[&MlxArray],
) {
    if enabled {
        eval(targets);
        record_gemma4_moe_profile_stage(stage, saturating_profile_us(started));
    }
}

pub(super) fn linear_attention_profile_eval_elapsed(
    enabled: bool,
    stage: LinearAttentionProfileStage,
    started: Instant,
    targets: &[&MlxArray],
) {
    if enabled {
        eval(targets);
        record_linear_attention_profile_stage(stage, saturating_profile_us(started));
    }
}

pub(super) fn decode_profile_eval_elapsed(
    enabled: bool,
    stage: DecodeProfileStage,
    started: Instant,
    targets: &[&MlxArray],
) {
    if enabled {
        eval(targets);
        record_decode_profile_stage(stage, saturating_profile_us(started));
    }
}

pub(crate) fn forward_profile_eval_elapsed(
    profile_decode: bool,
    profile_prefill: bool,
    stage: DecodeProfileStage,
    started: Instant,
    targets: &[&MlxArray],
) {
    if profile_decode {
        eval(targets);
        record_decode_profile_stage(stage, saturating_profile_us(started));
    } else if profile_prefill {
        eval(targets);
        record_prefill_profile_stage(stage, saturating_profile_us(started));
    }
}

pub fn take_gemma4_moe_profile_snapshot() -> Gemma4MoeProfileSnapshot {
    if !gemma4_moe_profile_enabled() {
        return Gemma4MoeProfileSnapshot::default();
    }
    let mut profile = gemma4_moe_profile().lock().unwrap();
    let snapshot = *profile;
    *profile = Gemma4MoeProfileSnapshot::default();
    snapshot
}

pub fn take_linear_attention_profile_snapshot() -> LinearAttentionProfileSnapshot {
    let mut profile = linear_attention_profile().lock().unwrap();
    let snapshot = *profile;
    *profile = LinearAttentionProfileSnapshot::default();
    snapshot
}

pub fn take_prefill_profile_snapshot() -> PrefillProfileSnapshot {
    if !prefill_profile_enabled() {
        return PrefillProfileSnapshot::default();
    }
    let mut profile = prefill_profile().lock().unwrap();
    let snapshot = *profile;
    *profile = PrefillProfileSnapshot::default();
    snapshot
}

pub fn take_decode_profile_snapshot() -> DecodeProfileSnapshot {
    if !decode_profile_enabled() {
        return DecodeProfileSnapshot::default();
    }
    let mut profile = decode_profile().lock().unwrap();
    let snapshot = *profile;
    *profile = DecodeProfileSnapshot::default();
    snapshot
}

pub fn take_moe_profile_snapshot() -> MoeProfileSnapshot {
    if !moe_profile_enabled() {
        return MoeProfileSnapshot::default();
    }
    let mut profile = moe_profile().lock().unwrap();
    let snapshot = *profile;
    *profile = MoeProfileSnapshot::default();
    snapshot
}

/// Per-stage wall time for the batched embedding forward
/// (`forward_for_embedding_batch` → `layer_forward_dense_embed`).
///
/// Enabled via `AX_MLX_EMBED_PROFILE=1`. Like `DecodeProfileSnapshot`, each
/// stage timer forces a blocking `eval()` to materialise the lazy graph at that
/// point, so enabling the profile **disables forward pipelining** and inflates
/// absolute wall time relative to production. The *ratios* between stages are
/// the diagnostic signal — they localise where the batched embedding path
/// spends its time (see `docs/EMBEDDINGS.md`).
///
/// Profiling forces the imperative path: the per-`(batch, max_len,
/// target_positions)` compiled closure is skipped so the stage barriers can be
/// inserted (a compiled closure is a single traced graph with no per-stage
/// boundary). A/B of compile on/off is decode-neutral for this path, so the
/// imperative breakdown is representative of the compiled one.
///
/// Stages mirror `layer_forward_dense_embed`:
/// - `embed_tokens_wall_us` — `build_embedding_batch_hidden` (token-id gather
///   from the quantized embedding table + bf16 cast + optional hidden scale).
///   Outside the layer loop; charged once per call.
/// - `attn_norm_wall_us` — pre-attention RMSNorm, or fused FFN-residual-add +
///   RMSNorm (`add_rms_norm_pair`) for layers after the first.
/// - `qkv_proj_wall_us` — `qkv_project` (Q/K/V quantized matmuls + reshape).
/// - `value_prep_wall_us` — `prepare_value_bhsd_from_proj` (V reshape +
///   optional V-norm to BHSD).
/// - `qk_norm_rope_wall_us` — Q/K RMSNorm + RoPE for both Q and K. NOTE the
///   embed path uses the generic `qk_norm_rope_bhsd_from_proj` (Qwen-family
///   direct route default-OFF), unlike the tuned prefill `layer_forward`.
/// - `sdpa_wall_us` — `full_precision_attention` (fused causal SDPA).
/// - `attn_out_proj_wall_us` — transpose-back + reshape + attention output
///   projection + fused pre-FFN RMSNorm (`add_rms_norm_pair`).
/// - `ffn_norm_wall_us` — pre-FFN RMSNorm (fused into `attn_out_proj_wall_us`
///   via `add_rms_norm_pair`; recorded as zero-cost for ABI stability).
/// - `ffn_wall_us` — `ffn_swiglu` (FFN residual add is deferred and fused into
///   the next layer's `attn_norm_wall_us`).
/// - `final_norm_pool_wall_us` — last-token/CLS extract (or full hidden for
///   Mean) + final RMSNorm + final FFN residual add. Outside the layer loop;
///   charged once per call.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EmbedProfileSnapshot {
    pub enabled: u32,
    pub calls: u32,
    pub layers: u32,
    pub batch: u32,
    pub tokens: u32,
    pub embed_tokens_wall_us: u32,
    pub attn_norm_wall_us: u32,
    pub qkv_proj_wall_us: u32,
    pub value_prep_wall_us: u32,
    pub qk_norm_rope_wall_us: u32,
    pub sdpa_wall_us: u32,
    pub attn_out_proj_wall_us: u32,
    pub ffn_norm_wall_us: u32,
    pub ffn_wall_us: u32,
    pub final_norm_pool_wall_us: u32,
}

#[derive(Clone, Copy)]
pub(crate) enum EmbedProfileStage {
    EmbedTokens,
    AttnNorm,
    QkvProj,
    ValuePrep,
    QkNormRope,
    Sdpa,
    AttnOutProj,
    FfnNorm,
    Ffn,
    FinalNormPool,
}

static EMBED_PROFILE: OnceLock<Mutex<EmbedProfileSnapshot>> = OnceLock::new();
static EMBED_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();

pub(crate) fn embed_profile_enabled() -> bool {
    profile_env_enabled(&EMBED_PROFILE_ENABLED, "AX_MLX_EMBED_PROFILE")
}

fn embed_profile() -> &'static Mutex<EmbedProfileSnapshot> {
    EMBED_PROFILE.get_or_init(|| Mutex::new(EmbedProfileSnapshot::default()))
}

pub(crate) fn record_embed_profile_stage(stage: EmbedProfileStage, wall_us: u32) {
    let mut profile = embed_profile().lock().unwrap();
    profile.enabled = 1;
    let target = match stage {
        EmbedProfileStage::EmbedTokens => &mut profile.embed_tokens_wall_us,
        EmbedProfileStage::AttnNorm => &mut profile.attn_norm_wall_us,
        EmbedProfileStage::QkvProj => &mut profile.qkv_proj_wall_us,
        EmbedProfileStage::ValuePrep => &mut profile.value_prep_wall_us,
        EmbedProfileStage::QkNormRope => &mut profile.qk_norm_rope_wall_us,
        EmbedProfileStage::Sdpa => &mut profile.sdpa_wall_us,
        EmbedProfileStage::AttnOutProj => &mut profile.attn_out_proj_wall_us,
        EmbedProfileStage::FfnNorm => &mut profile.ffn_norm_wall_us,
        EmbedProfileStage::Ffn => &mut profile.ffn_wall_us,
        EmbedProfileStage::FinalNormPool => &mut profile.final_norm_pool_wall_us,
    };
    *target = target.saturating_add(wall_us);
}

pub(crate) fn record_embed_profile_call(layers: u32, batch: u32, tokens: u32) {
    let mut profile = embed_profile().lock().unwrap();
    profile.enabled = 1;
    profile.calls = profile.calls.saturating_add(1);
    profile.layers = profile.layers.saturating_add(layers);
    profile.batch = profile.batch.saturating_add(batch);
    profile.tokens = profile.tokens.saturating_add(tokens);
}

pub(crate) fn embed_profile_eval_elapsed(
    enabled: bool,
    stage: EmbedProfileStage,
    started: Instant,
    targets: &[&MlxArray],
) {
    if enabled {
        eval(targets);
        record_embed_profile_stage(stage, saturating_profile_us(started));
    }
}

/// Snapshot the embedding profile counters and reset them. Unlike the prefill/
/// decode takers this does not gate on the enable flag, so a probe can read the
/// accumulated breakdown directly after driving the instrumented path.
pub fn take_embed_profile_snapshot() -> EmbedProfileSnapshot {
    let mut profile = embed_profile().lock().unwrap();
    let snapshot = *profile;
    *profile = EmbedProfileSnapshot::default();
    snapshot
}
