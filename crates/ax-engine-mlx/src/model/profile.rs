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
}

static GEMMA4_MOE_PROFILE: OnceLock<Mutex<Gemma4MoeProfileSnapshot>> = OnceLock::new();
static LINEAR_ATTENTION_PROFILE: OnceLock<Mutex<LinearAttentionProfileSnapshot>> = OnceLock::new();
static PREFILL_PROFILE: OnceLock<Mutex<PrefillProfileSnapshot>> = OnceLock::new();
static DECODE_PROFILE: OnceLock<Mutex<DecodeProfileSnapshot>> = OnceLock::new();
static GEMMA4_MOE_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
static LINEAR_ATTENTION_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
static PREFILL_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
static DECODE_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();

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
    let mut profile = prefill_profile().lock().unwrap();
    let snapshot = *profile;
    *profile = PrefillProfileSnapshot::default();
    snapshot
}

pub fn take_decode_profile_snapshot() -> DecodeProfileSnapshot {
    let mut profile = decode_profile().lock().unwrap();
    let snapshot = *profile;
    *profile = DecodeProfileSnapshot::default();
    snapshot
}
