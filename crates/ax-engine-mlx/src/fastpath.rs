//! Environment-backed optimization flags for ax-engine-mlx fast paths.
//!
//! Each accessor reads its environment variable once per process and caches
//! the result in a `OnceLock`. For opt-in flags, the value is parsed
//! case-insensitively after trimming ASCII whitespace; `1`, `true`, or `yes`
//! (any casing) engages the flag. Any other value (including unset) leaves the
//! flag disabled. Default-on flags use a separate parser and must document their
//! kill-switch semantics at the accessor.
//!
//! The pattern intentionally mirrors DS4's `ds4_metal_get_*` shape-gated
//! pipeline cache: every fast path declares an explicit predicate, a documented
//! opt-in or kill switch, and an explicit fallback. Co-locating the env-var
//! names here gives a single grep target for "which optimization flags does the
//! runtime expose?" and matches the W1.3 / W2.a audit conventions.
//!
//! Qwen linear-MTP exact arithmetic is the one model-scoped exception: its
//! environment value is still cached, but production runners install a
//! thread-local selection derived from the loaded artifact capability.

use std::cell::Cell;
use std::sync::OnceLock;

/// Maximum number of committed prompt transitions retained in the Qwen MTP
/// head cache. `0` means unlimited. Keeping this reader beside the other
/// process-cached fast-path knobs lets prefill capture and decode warmup share
/// one value instead of independently interpreting the environment.
pub fn mtp_warmup_cap() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_MTP_WARMUP_CAP")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(256)
    })
}

/// Optional fixed MTP proposal depth used for controlled admission trials.
/// Unset keeps the adaptive controller; positive values are clamped to the
/// model head's advertised maximum.
pub fn mtp_fixed_draft_depth() -> Option<usize> {
    static CACHED: OnceLock<Option<usize>> = OnceLock::new();
    *CACHED.get_or_init(|| parse_positive_usize_env("AX_MLX_MTP_FIXED_DRAFT_DEPTH"))
}

/// `AX_MLX_MTP_DEPTH3_HYSTERESIS` — keep a three-token proposal window after
/// accepting its first two drafts. This avoids alternating 3→2→3 on
/// high-acceptance Qwen MTP streams while still backing off after a zero- or
/// one-token accept.
///
/// **Default: OFF** pending matched M5 admission.
pub fn mtp_depth3_hysteresis_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| parse_bool_env("AX_MLX_MTP_DEPTH3_HYSTERESIS"))
}

/// `AX_MLX_MTP_DEPTH3_MISS_BACKOFF` — for a three-token Qwen head, start deep
/// and back off to two drafts only after a complete miss. Any accepted draft
/// restores depth three on the next cycle.
///
/// **Default: OFF** pending matched M5 admission.
pub fn mtp_depth3_miss_backoff_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| parse_bool_env("AX_MLX_MTP_DEPTH3_MISS_BACKOFF"))
}

fn parse_bool_value(raw: &str) -> bool {
    let trimmed = raw.trim();
    trimmed.eq_ignore_ascii_case("1")
        || trimmed.eq_ignore_ascii_case("true")
        || trimmed.eq_ignore_ascii_case("yes")
}

fn parse_bool_env(var: &str) -> bool {
    let Ok(raw) = std::env::var(var) else {
        return false;
    };
    parse_bool_value(&raw)
}

/// Parse an env var as a kill switch. Returns `true` when unset or set to a
/// truthy value (`1`/`true`/`yes`); returns `false` only when explicitly set
/// to a falsy value (`0`/`false`/`no`). Used by accessors that default ON in
/// production but expose an off-switch for safety.
fn parse_bool_env_default_on(var: &str) -> bool {
    let Ok(raw) = std::env::var(var) else {
        return true;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return true;
    }
    if trimmed.eq_ignore_ascii_case("0")
        || trimmed.eq_ignore_ascii_case("false")
        || trimmed.eq_ignore_ascii_case("no")
        || trimmed.eq_ignore_ascii_case("off")
    {
        return false;
    }
    // Any non-empty / non-falsy value is treated as truthy. Matches the
    // existing `parse_bool_env` semantics for the explicit-on case.
    true
}

fn parse_positive_usize_env(var: &str) -> Option<usize> {
    let raw = std::env::var(var).ok()?;
    let n: usize = raw.trim().parse().ok()?;
    (n > 0).then_some(n)
}

fn parse_nonnegative_f32_env(var: &str) -> Option<f32> {
    let raw = std::env::var(var).ok()?;
    parse_nonnegative_f32(&raw)
}

fn parse_nonnegative_f32(raw: &str) -> Option<f32> {
    let value: f32 = raw.trim().parse().ok()?;
    (value.is_finite() && value >= 0.0).then_some(value)
}

macro_rules! env_flag {
    ($(#[$meta:meta])* $fn_name:ident, $env_var:literal) => {
        $(#[$meta])*
        pub fn $fn_name() -> bool {
            static CACHED: OnceLock<bool> = OnceLock::new();
            *CACHED.get_or_init(|| parse_bool_env($env_var))
        }
    };
}

/// Default-on counterpart of `env_flag!`. Production code uses this for fast
/// paths that should run by default but need a documented kill switch
/// reachable via env var (e.g. `AX_MLX_PREFILL_FFN_COMPILE_SWIGLU=0`).
macro_rules! env_flag_default_on {
    ($(#[$meta:meta])* $fn_name:ident, $env_var:literal) => {
        $(#[$meta])*
        pub fn $fn_name() -> bool {
            static CACHED: OnceLock<bool> = OnceLock::new();
            *CACHED.get_or_init(|| parse_bool_env_default_on($env_var))
        }
    };
}

env_flag!(
    /// Engaged by `AX_NO_SPEC` (the CLAUDE.md-documented convention for
    /// forcing greedy direct decode). When set, `MlxRunner::from_artifacts`
    /// ORs this value into the `disable_ngram_acceleration` parameter, so
    /// the env switch is honored uniformly from CLI, server, and SDK entry
    /// points — not just from the bench CLI's argument parsing path. Truthy
    /// values follow the module-level parser contract.
    ngram_acceleration_disabled,
    "AX_NO_SPEC"
);

env_flag!(
    /// `AX_MLX_SKIP_DECODE_ROUTE_TELEMETRY` — on pure single-token decode
    /// steps, skip appending the large crossover-decision map (profile
    /// counters, layout telemetry, ngram/MTP counters). Prefill / multi-token
    /// steps and the first request still get full route metadata. Opt-in for
    /// competitive single-stream decode (M5 Max Qwen3.5-9B SSE).
    skip_decode_route_telemetry,
    "AX_MLX_SKIP_DECODE_ROUTE_TELEMETRY"
);

env_flag_default_on!(
    /// `AX_MLX_DECODE_SAMPLING_GPU_TOPK` — route exact top-k sampling through
    /// MLX `argpartition_axis` and gather only the top-k full-domain
    /// probabilities back to CPU.
    ///
    /// **Default: ON** (kill-switch via
    /// `AX_MLX_DECODE_SAMPLING_GPU_TOPK=0`).
    decode_sampling_gpu_topk_enabled,
    "AX_MLX_DECODE_SAMPLING_GPU_TOPK"
);

env_flag_default_on!(
    /// `AX_THINK_SOFT_CLOSE` — rank-based think soft-close: while inside an
    /// open think block and within the soft window ahead of the answer
    /// reserve / think cap, materialize logits and emit the think-close
    /// token early when it ranks in the model's own top-3 (ds4-style soft
    /// close, ds4_eval.c soft_limit_think_close_rank).
    ///
    /// **Default: ON** (kill-switch via `AX_THINK_SOFT_CLOSE=0`).
    think_soft_close_enabled,
    "AX_THINK_SOFT_CLOSE"
);

env_flag!(
    /// `AX_MLX_BATCHED_PREFILL` — route eligible cold text prefill items
    /// through one padded batched forward per planned cohort
    /// (`ax_engine_core::prefill_cohort` drives the grouping; the model must
    /// pass `model::supports_batched_prefill`). **Default: OFF** —
    /// experimental: parity with the sequential path is tolerance-verified,
    /// not byte-exact certified (padded batching changes reduction shapes).
    batched_prefill_enabled,
    "AX_MLX_BATCHED_PREFILL"
);

/// `AX_MLX_BATCHED_PREFILL_TOKENS` — override the padded-token admission
/// budget (`rows * max_len` cap) for one batched prefill cohort. Unset uses
/// `ax_engine_core::prefill_cohort::default_padded_token_budget` over the
/// session prefill chunk and the rows cap. `0` disables the cap.
pub fn batched_prefill_token_budget_override() -> Option<u32> {
    static CACHED: OnceLock<Option<u32>> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_BATCHED_PREFILL_TOKENS")
            .ok()
            .and_then(|raw| raw.trim().parse().ok())
    })
}

/// `AX_MLX_MTP_VERIFY_SUBMIT_LAYERS` — submit the speculative verify graph to
/// the GPU every N transformer layers instead of only at the end of the build.
///
/// **Default: 0 (off).** Set to a positive layer interval to enable.
///
/// The verify forward is the one decode graph with no double buffer: the host
/// builds every layer, then blocks in `eval`. On a dense model that build is a
/// small share of a step, because the step itself is long — it streams every
/// weight. On a sparse-expert model it is not: the step reads only the routed
/// experts and is several times shorter, so the same absolute build cost
/// becomes ~40% of a step, paid with the GPU idle. Measured on Qwen3.6-35B-A3B
/// AXQ (`df-macbookpro-m5`, AX Engine 6.14.0): 4.1 ms of build against a
/// 10.4 ms eval, on every speculative step.
///
/// Submitting each chunk with `async_eval` lets the GPU start the early layers
/// while the host still builds the later ones — the same overlap idea as
/// cache-only prefill chunks, but mid-loop submits only the residual
/// `hidden` (not the full cache-ref set). Side-output K/V and linear-attn
/// state are materialised by the caller's terminal eval; re-submitting them
/// every chunk causes MoE gather_qmm backpressure.
///
/// Exactness-preserving: `async_eval` schedules an already-built graph and
/// changes no operand, shape, or reduction order. Only the synchronisation
/// point moves.
///
/// **Buffer caps:** MLX charges a `gather_qmm`'s full expert stack against
/// `MLX_MAX_MB_PER_BUFFER`, so a too-low cap can split every MoE layer into
/// its own command buffer and blunt the overlap (see
/// `docs/performance/gather-qmm-async-serialization.md`). Do **not** raise
/// the cap solely to improve the speculative/direct A/B ratio on Qwen3.6
/// MoE: raising it lifts direct decode as well (and can regress prefill on
/// the `qwen3_5` family, which is why auto buffer caps exclude that family).
/// Measure absolute tok/s with the product buffer policy you already ship.
///
/// Measured sweet spot on 35B-A3B AXQ: interval **8** (default remains off).
pub fn mtp_verify_submit_layer_interval() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_MTP_VERIFY_SUBMIT_LAYERS")
            .ok()
            .and_then(|raw| raw.trim().parse().ok())
            .unwrap_or(0)
    })
}

/// Resolve the chunked-submit interval for one forward build.
///
/// Single-position builds return `0`: a `seq == 1` decode step is the direct
/// pipeline's, and that path already double-buffers. An interval at or beyond
/// the layer count also returns `0` — the sole submit it would schedule is the
/// caller's own terminating `eval`, so it is pure overhead.
pub(crate) fn verify_submit_interval_for_build(
    seq: usize,
    layer_count: usize,
    configured: usize,
) -> usize {
    // Depth-1 Qwen linear verify is S=2..4. Mid-loop async_eval of `hidden`
    // is a win on long prefill-shaped graphs (35B-A3B interval 8) and pure
    // overhead on the short teacher-forced step when stacked on
    // AX_MLX_PIPELINE_GRANULARITY=layer.
    if seq <= 4 || configured == 0 || configured >= layer_count {
        return 0;
    }
    configured
}

/// Default mid-loop submit stride for exact S=2..=4 verify when
/// `AX_MLX_MTP_VERIFY_SUBMIT_LAYERS` is unset.
///
/// Skipping every layer-boundary hint (`pipelinenohint`) left the GPU idle
/// during the 64-layer encode. Per-layer `PIPELINE=layer` on this short
/// graph is the other extreme. Four layers is the middle: MLX can fuse a
/// small chunk while host encode of the next chunk overlaps GPU.
///
/// Measured `bf56ee6b` used interval 8 instead of per-layer hints and
/// washed (verify_forward 2.907s vs 2.819s). Production stays on per-layer
/// PIPELINE; this helper remains for the unit contract.
#[cfg(test)]
pub(crate) const EXACT_SHORT_VERIFY_SUBMIT_DEFAULT: usize = 4;

/// Submit interval for exact Qwen linear MTP verify (`seq` 2..=4).
///
/// Official `AX_MLX_MTP_VERIFY_SUBMIT_LAYERS=8` is ignored by
/// [`verify_submit_interval_for_build`] on these lengths because it used to
/// *stack* on `PIPELINE=layer`. This helper uses the configured interval
/// *instead of* per-layer hints. Unset configured falls back to
/// [`EXACT_SHORT_VERIFY_SUBMIT_DEFAULT`].
#[cfg(test)]
pub(crate) fn exact_short_verify_submit_interval(
    seq: usize,
    layer_count: usize,
    configured: usize,
) -> usize {
    if !(2..=4).contains(&seq) || layer_count == 0 {
        return 0;
    }
    let n = if configured == 0 {
        EXACT_SHORT_VERIFY_SUBMIT_DEFAULT
    } else {
        configured
    };
    if n >= layer_count { 0 } else { n }
}

/// `AX_MLX_BATCHED_PREFILL_ROWS` — cap on rows per batched prefill cohort.
/// Default 8; `0` disables the cap.
pub fn batched_prefill_max_rows() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_BATCHED_PREFILL_ROWS")
            .ok()
            .and_then(|raw| raw.trim().parse().ok())
            .unwrap_or(8)
    })
}

env_flag_default_on!(
    /// `AX_MLX_LOAD_KERNEL_WARMUP` — at model load, compile the custom
    /// Metal kernel specializations the decode path will hit (currently
    /// the linear-attention gated-delta + conv1d family) so the lazy
    /// MSL→pipeline compile does not land inside the first request's
    /// latency. Best-effort: warm-up failure logs a warning and the
    /// model still loads.
    ///
    /// **Default: ON** (kill-switch via `AX_MLX_LOAD_KERNEL_WARMUP=0`).
    load_kernel_warmup_enabled,
    "AX_MLX_LOAD_KERNEL_WARMUP"
);

env_flag_default_on!(
    /// `AX_MLX_GEMMA4_ASSISTANT_MTP_COALESCED_VERIFY` — coalesce independent
    /// greedy Gemma 4 assistant-MTP verify graphs into one MLX completion
    /// barrier. The route stays row-exact: every request retains its own
    /// batch=1 graph, KV cache, acceptance decision, and assistant draft.
    ///
    /// **Default: ON** (kill-switch via
    /// `AX_MLX_GEMMA4_ASSISTANT_MTP_COALESCED_VERIFY=0`). The production route
    /// is fail-closed and only admits exact-greedy assistant-only drafts.
    gemma4_assistant_mtp_coalesced_verify_enabled,
    "AX_MLX_GEMMA4_ASSISTANT_MTP_COALESCED_VERIFY"
);

env_flag_default_on!(
    /// `AX_MLX_GEMMA4_ASSISTANT_MTP_SEQUENTIAL_ORACLE` — under temperature 0,
    /// re-verify Gemma assistant drafts with singleton production forwards so
    /// accepted tokens match MTP-off greedy (multi-token teacher-forced argmax
    /// can disagree on shared-KV / sliding-window / softcap paths).
    ///
    /// **Default: ON** for exactness. Kill-switch via
    /// `AX_MLX_GEMMA4_ASSISTANT_MTP_SEQUENTIAL_ORACLE=0` restores multi-token
    /// verify (faster; formal pilots must re-check exactness before claiming
    /// Tier 2). Coalesced greedy verify still uses the sequential oracle when
    /// this flag is on.
    gemma4_assistant_mtp_sequential_oracle_enabled,
    "AX_MLX_GEMMA4_ASSISTANT_MTP_SEQUENTIAL_ORACLE"
);

env_flag_default_on!(
    /// `AX_MLX_GEMMA4_ASSISTANT_MTP_CYCLE_GUARD` — under formal multi-token
    /// verify (`SEQUENTIAL_ORACLE=0`), force the pure-direct sequential oracle
    /// when the pending draft continues an established repetition cycle at the
    /// committed history tail. Cycle-continuation false accepts were the
    /// dominant formal Tier 2 divergence mode (teacher-forced multi-token
    /// matching a loop draft while sequential greedy would break the cycle).
    ///
    /// **Default: ON** (exactness-preserving: only routes *more* steps to the
    /// sequential oracle, never fewer). Kill-switch via
    /// `AX_MLX_GEMMA4_ASSISTANT_MTP_CYCLE_GUARD=0` for ablations.
    gemma4_assistant_mtp_cycle_guard_enabled,
    "AX_MLX_GEMMA4_ASSISTANT_MTP_CYCLE_GUARD"
);

env_flag!(
    /// `AX_MLX_GEMMA4_ASSISTANT_MTP_EARLY_GEN_PURE_DIRECT` — experimental:
    /// under formal multi-token verify (`SEQUENTIAL_ORACLE=0`), force the
    /// pure-direct sequential path for the first N generated tokens (see
    /// `GEMMA_MT_EARLY_GEN_PURE_DIRECT_TOKENS`).
    ///
    /// **Default: OFF.** On formal A/B this desynchronizes MTP-on from the
    /// MTP-off multi-token-compatible baseline (shared bounded ring /
    /// `forward_all_positions` singleton) and regressed agent-coding exactness
    /// in M5 retests. Kept as an opt-in probe only; residual general-long
    /// identity must be fixed on the multi-token path itself.
    gemma4_assistant_mtp_early_gen_pure_direct_enabled,
    "AX_MLX_GEMMA4_ASSISTANT_MTP_EARLY_GEN_PURE_DIRECT"
);

env_flag_default_on!(
    /// `AX_MLX_DECODE_MTP_TARGET_PROB_WORKSPACE` — reuse request-local CPU
    /// buffers while building/extracting MTP target probabilities.
    ///
    /// **Default: ON** (kill-switch via
    /// `AX_MLX_DECODE_MTP_TARGET_PROB_WORKSPACE=0`).
    decode_mtp_target_prob_workspace_enabled,
    "AX_MLX_DECODE_MTP_TARGET_PROB_WORKSPACE"
);

env_flag_default_on!(
    /// `AX_MLX_PREFILL_FFN_COMPILE_SWIGLU` — Qwen3 / GLM / shared-expert
    /// SwiGLU compile fusion (W1 spike K of fusion PRD).
    ///
    /// **Default: ON** (kill-switch via
    /// `AX_MLX_PREFILL_FFN_COMPILE_SWIGLU=0`).
    ///
    /// Routes `silu(gate) * up` chains in Qwen 3 dense FFN, Qwen MoE routed
    /// experts, the shared expert path, and any future SwiGLU consumer through
    /// a compiled closure with the same `MlxClosure::try_apply` fail-closed
    /// contract used by the embedding compile cache.
    prefill_ffn_compile_swiglu_enabled,
    "AX_MLX_PREFILL_FFN_COMPILE_SWIGLU"
);

env_flag_default_on!(
    /// `AX_MLX_PACK_QKV_PROJECTIONS` — materialize split dense-attention Q/K/V
    /// projections into one packed projection at load time.
    ///
    /// **Default: ON** (kill-switch via
    /// `AX_MLX_PACK_QKV_PROJECTIONS=0`).
    ///
    /// Mirrors the dense FFN gate/up packing contract: when Q/K/V quantization
    /// metadata is compatible, the loader materializes the concatenated weight
    /// before the forward path consumes it. Unsupported shapes fall back to the
    /// split-projection path in `weights.rs`.
    dense_attention_qkv_packing_enabled,
    "AX_MLX_PACK_QKV_PROJECTIONS"
);

env_flag_default_on!(
    /// `AX_MLX_PACK_DENSE_FFN_GATE_UP` — materialize dense FFN gate/up
    /// projections into one packed projection at load time.
    ///
    /// **Default: ON** (kill-switch via
    /// `AX_MLX_PACK_DENSE_FFN_GATE_UP=0`).
    ///
    /// Collapses the dense FFN gate and up matmuls into one quantized matmul
    /// plus a last-dim slice when the artifact ships split projections, the
    /// quantization metadata is compatible, and the family runtime consumes
    /// the packed route. Qwen runtimes intentionally keep split projections.
    dense_ffn_gate_up_packing_enabled,
    "AX_MLX_PACK_DENSE_FFN_GATE_UP"
);

env_flag_default_on!(
    /// `AX_MLX_GEGLU_MUL_METAL` — route split Gemma-family GEGLU
    /// `gelu_approx(gate) * up` through a custom MLX Metal elementwise node.
    ///
    /// **Default: ON** (kill-switch via `AX_MLX_GEGLU_MUL_METAL=0`).
    ///
    /// This covers MoE expert paths where gate/up projections are already
    /// materialized as separate tensors. Packed dense FFN layers use the
    /// narrower packed GEGLU kernel below.
    geglu_mul_metal_enabled,
    "AX_MLX_GEGLU_MUL_METAL"
);

env_flag_default_on!(
    /// `AX_MLX_DENSE_GEGLU_PACKED_METAL` — route packed dense Gemma-family
    /// GEGLU activation through a custom MLX Metal elementwise kernel.
    ///
    /// **Default: ON** (kill-switch via
    /// `AX_MLX_DENSE_GEGLU_PACKED_METAL=0`).
    ///
    /// This is narrower than the unstable `MlxClosure::compile` GeGLU path
    /// and narrower than the whole-FFN C++ shim: it only fuses the packed
    /// gate/up split plus `gelu_approx(gate) * up` activation into one lazy MLX
    /// graph node. Quantized gate/up and down matmuls remain the normal MLX
    /// operations, preserving profiling and avoiding the decode regression
    /// observed with the whole-FFN direct shim.
    dense_geglu_packed_metal_enabled,
    "AX_MLX_DENSE_GEGLU_PACKED_METAL"
);

env_flag_default_on!(
    /// `AX_MLX_DENSE_SWIGLU_PACKED_METAL` — route packed dense Qwen-family
    /// SwiGLU activation through a custom MLX Metal elementwise kernel.
    ///
    /// **Default: ON** (kill-switch via
    /// `AX_MLX_DENSE_SWIGLU_PACKED_METAL=0`).
    ///
    /// Mirrors the packed GEGLU fast path for dense FFN layers that already
    /// materialize gate/up as one projection. The Metal node fuses the last-dim
    /// split plus `silu(gate) * up`; unsupported shapes fall back to the existing
    /// compiled-closure / imperative SwiGLU path.
    dense_swiglu_packed_metal_enabled,
    "AX_MLX_DENSE_SWIGLU_PACKED_METAL"
);

env_flag_default_on!(
    /// `AX_MLX_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL` — decode-only Qwen dense
    /// FFN gate/up affine-quantized SwiGLU Metal kernel.
    ///
    /// **Default: ON** (kill-switch via
    /// `AX_MLX_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL=0`).
    ///
    /// On M5 Max Qwen3.5-9B-MLX-4bit, this path with the split-FFN compile
    /// short-circuit below measures ~110–111 tok/s pure decode vs ~107 when the
    /// host-side compiled split-FFN graph wins the race and skips the kernel.
    /// Model-specific shapes that regress, and otherwise unsupported shapes,
    /// fall back to two MLX quantized matmuls + SwiGLU.
    qwen_dense_ffn_gate_up_matvec_metal_enabled,
    "AX_MLX_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL"
);

const QWEN_LINEAR_MTP_EXACT_ENV: &str = "AX_MLX_QWEN_LINEAR_MTP_EXACT";

/// How the Qwen linear-MTP exact arithmetic profile was selected.
///
/// The route code is emitted by `MlxRunner` so benchmark artifacts distinguish
/// the production auto-capability path from an explicit operator override.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QwenLinearMtpExactSelection {
    /// The loaded model does not satisfy the runtime's exact-profile contract.
    Ineligible,
    /// The model satisfies the runtime contract and the environment is unset.
    Auto,
    /// The model satisfies the contract and the operator explicitly enabled it.
    ExplicitEnabled,
    /// The operator explicitly disabled the profile.
    ExplicitDisabled,
}

impl QwenLinearMtpExactSelection {
    pub(crate) fn route_code(self) -> u32 {
        match self {
            Self::Ineligible => 0,
            Self::Auto => 1,
            Self::ExplicitEnabled => 2,
            Self::ExplicitDisabled => 3,
        }
    }
}

fn qwen_linear_mtp_exact_env_override() -> Option<bool> {
    static CACHED: OnceLock<Option<bool>> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var(QWEN_LINEAR_MTP_EXACT_ENV)
            .ok()
            .map(|raw| parse_bool_value(&raw))
    })
}

fn resolve_qwen_linear_mtp_exact_with_override(
    model_eligible: bool,
    explicit: Option<bool>,
) -> (bool, QwenLinearMtpExactSelection) {
    if !model_eligible {
        return (false, QwenLinearMtpExactSelection::Ineligible);
    }
    match explicit {
        Some(true) => (true, QwenLinearMtpExactSelection::ExplicitEnabled),
        Some(false) => (false, QwenLinearMtpExactSelection::ExplicitDisabled),
        None => (true, QwenLinearMtpExactSelection::Auto),
    }
}

/// Resolve the per-model Qwen linear-MTP exact profile.
///
/// A truthy environment value records an explicit selection but cannot bypass
/// the model capability gate. A falsy or malformed value is a kill switch.
/// When the environment is unset, certified models select the profile
/// automatically.
pub(crate) fn resolve_qwen_linear_mtp_exact(
    model_eligible: bool,
) -> (bool, QwenLinearMtpExactSelection) {
    resolve_qwen_linear_mtp_exact_with_override(
        model_eligible,
        qwen_linear_mtp_exact_env_override(),
    )
}

thread_local! {
    /// Runner-scoped selection. `None` preserves the legacy explicit-env
    /// behavior for standalone probes that call model functions directly.
    static QWEN_LINEAR_MTP_EXACT_SCOPE: Cell<Option<bool>> = const { Cell::new(None) };
    /// Runner-scoped marker for an oMLX-style relaxed target verifier. This is
    /// deliberately separate from the singleton-exact arithmetic profile:
    /// verify-only fused preprocessing/native causal SDPA are safe here, while
    /// row-exact projection and residual routes must remain disabled.
    static QWEN_LINEAR_MTP_TARGET_VERIFY_SCOPE: Cell<bool> = const { Cell::new(false) };
    /// Runner-scoped marker for one relaxed Qwen MTP request call. Unlike the
    /// target-forward marker, this remains active during the matching cold
    /// prefill, drafting, and accepted-history refold, but is disabled for
    /// ordinary/direct sessions.
    static QWEN_LINEAR_MTP_RELAXED_SESSION_SCOPE: Cell<bool> = const { Cell::new(false) };
    /// Graph-construction marker for the whole Qwen target-verifier closure.
    /// Nested per-layer closures and eager/async submit hints must stay out of
    /// an enclosing `mlx_compile` trace, while arithmetic fast paths remain
    /// available to the traced body.
    static QWEN_LINEAR_MTP_WHOLE_VERIFY_TRACE_SCOPE: Cell<bool> = const { Cell::new(false) };
}

/// Restores the previous thread-local exact-profile selection on drop.
#[must_use]
pub(crate) struct QwenLinearMtpExactScope {
    previous: Option<bool>,
}

impl Drop for QwenLinearMtpExactScope {
    fn drop(&mut self) {
        QWEN_LINEAR_MTP_EXACT_SCOPE.with(|current| current.set(self.previous));
    }
}

/// Select the exact arithmetic profile for one model-runner call.
///
/// MLX graph construction is synchronous on the runner thread, so a scoped
/// thread-local keeps concurrent resident models isolated without mutating
/// process-wide environment state.
pub(crate) fn scoped_qwen_linear_mtp_exact(enabled: bool) -> QwenLinearMtpExactScope {
    let previous = QWEN_LINEAR_MTP_EXACT_SCOPE.with(|current| {
        let previous = current.get();
        current.set(Some(enabled));
        previous
    });
    QwenLinearMtpExactScope { previous }
}

/// Whether the current model call uses the Qwen linear-attention exact
/// speculative-verifier arithmetic contract.
///
/// Production runners install a per-model scope. Standalone diagnostic probes
/// that do not install a scope retain the historical
/// `AX_MLX_QWEN_LINEAR_MTP_EXACT=1` opt-in behavior.
pub fn qwen_linear_mtp_exact_enabled() -> bool {
    QWEN_LINEAR_MTP_EXACT_SCOPE.with(|current| {
        current
            .get()
            .unwrap_or_else(|| qwen_linear_mtp_exact_env_override().unwrap_or(false))
    })
}

/// Restores the previous relaxed target-verifier marker on drop.
#[must_use]
pub(crate) struct QwenLinearMtpTargetVerifyScope {
    previous: bool,
}

impl Drop for QwenLinearMtpTargetVerifyScope {
    fn drop(&mut self) {
        QWEN_LINEAR_MTP_TARGET_VERIFY_SCOPE.with(|current| current.set(self.previous));
    }
}

/// Mark one target-model forward as a short Qwen linear-MTP verifier.
pub(crate) fn scoped_qwen_linear_mtp_target_verify(
    enabled: bool,
) -> QwenLinearMtpTargetVerifyScope {
    let previous = QWEN_LINEAR_MTP_TARGET_VERIFY_SCOPE.with(|current| {
        let previous = current.get();
        current.set(enabled);
        previous
    });
    QwenLinearMtpTargetVerifyScope { previous }
}

/// Whether verify-only fast kernels may use their multi-token forms.
///
/// The exact profile historically implied this permission. Relaxed target
/// verification now carries it independently so turning row-exact arithmetic
/// off does not accidentally restore full-history f32 casts, array masks, and
/// portable GDN preprocessing.
pub fn qwen_linear_mtp_verify_fast_kernels_enabled() -> bool {
    qwen_linear_mtp_exact_enabled()
        || QWEN_LINEAR_MTP_TARGET_VERIFY_SCOPE.with(|current| current.get())
}

/// Whether the current model call is the relaxed Qwen linear-MTP target
/// verifier rather than ordinary prefill/decode or row-exact replay.
pub fn qwen_linear_mtp_target_verify_enabled() -> bool {
    QWEN_LINEAR_MTP_TARGET_VERIFY_SCOPE.with(|current| current.get())
}

env_flag!(
    /// `AX_MLX_MTP_TARGET_LAYER_COMPILE` — allow the existing fixed-shape
    /// Qwen S=2..=4 layer closures inside the relaxed target verifier. The
    /// row-exact profile retains its historical behavior independently.
    ///
    /// **Default: OFF** pending matched M5 verifier admission.
    mtp_target_layer_compile_enabled,
    "AX_MLX_MTP_TARGET_LAYER_COMPILE"
);

env_flag!(
    /// `AX_MLX_MTP_PACKED_VERIFY_FFN` — use the prepacked gate/up projection
    /// inside the fixed-shape Qwen target-verifier FFN closure. This keeps the
    /// verifier's relaxed arithmetic contract while removing one skinny QMM
    /// dispatch per eligible dense layer.
    ///
    /// **Default: OFF** pending matched M5 admission.
    mtp_packed_verify_ffn_enabled,
    "AX_MLX_MTP_PACKED_VERIFY_FFN"
);

/// Whether fixed-shape Qwen verifier layer closures may engage.
pub fn qwen_linear_mtp_layer_compile_enabled() -> bool {
    !qwen_linear_mtp_whole_verify_trace_enabled()
        && (qwen_linear_mtp_exact_enabled()
            || (qwen_linear_mtp_target_verify_enabled() && mtp_target_layer_compile_enabled()))
}

env_flag!(
    /// `AX_MLX_MTP_WHOLE_VERIFY_COMPILE` — compile the complete dense
    /// Qwen3.5 target-verifier step with explicit full-attention and
    /// gated-delta cache tensor inputs/outputs. Exact target semantics are
    /// preserved; failure falls back to the ordinary verifier.
    ///
    /// **Default: OFF** pending matched M5 admission.
    mtp_whole_verify_compile_enabled,
    "AX_MLX_MTP_WHOLE_VERIFY_COMPILE"
);

env_flag!(
    /// `AX_MLX_MTP_LINEAR_LAYER_COMPILE` — compile each complete gated-delta
    /// layer in a short Qwen3.5 target-verifier step while leaving the
    /// full-attention layers on their existing paged route. The closure
    /// threads conv/recurrent state and the compact replay tape explicitly.
    ///
    /// **Default: OFF** pending matched M5 admission.
    mtp_linear_layer_compile_enabled,
    "AX_MLX_MTP_LINEAR_LAYER_COMPILE"
);

/// Restores the previous whole-verifier trace marker on drop.
#[must_use]
pub(crate) struct QwenLinearMtpWholeVerifyTraceScope {
    previous: bool,
}

impl Drop for QwenLinearMtpWholeVerifyTraceScope {
    fn drop(&mut self) {
        QWEN_LINEAR_MTP_WHOLE_VERIFY_TRACE_SCOPE.with(|current| current.set(self.previous));
    }
}

/// Mark synchronous graph construction for one whole target-verifier closure.
pub(crate) fn scoped_qwen_linear_mtp_whole_verify_trace(
    enabled: bool,
) -> QwenLinearMtpWholeVerifyTraceScope {
    let previous = QWEN_LINEAR_MTP_WHOLE_VERIFY_TRACE_SCOPE.with(|current| {
        let previous = current.get();
        current.set(enabled);
        previous
    });
    QwenLinearMtpWholeVerifyTraceScope { previous }
}

/// Whether the current thread is tracing the enclosing whole verifier graph.
pub fn qwen_linear_mtp_whole_verify_trace_enabled() -> bool {
    QWEN_LINEAR_MTP_WHOLE_VERIFY_TRACE_SCOPE.with(|current| current.get())
}

/// Restores the previous relaxed Qwen MTP session marker on drop.
#[must_use]
pub(crate) struct QwenLinearMtpRelaxedSessionScope {
    previous: bool,
}

impl Drop for QwenLinearMtpRelaxedSessionScope {
    fn drop(&mut self) {
        QWEN_LINEAR_MTP_RELAXED_SESSION_SCOPE.with(|current| current.set(self.previous));
    }
}

/// Mark one runner call as relaxed Qwen MTP request work.
pub(crate) fn scoped_qwen_linear_mtp_relaxed_session(
    enabled: bool,
) -> QwenLinearMtpRelaxedSessionScope {
    let previous = QWEN_LINEAR_MTP_RELAXED_SESSION_SCOPE.with(|current| {
        let previous = current.get();
        current.set(enabled);
        previous
    });
    QwenLinearMtpRelaxedSessionScope { previous }
}

/// Whether the current runner call belongs to a relaxed Qwen MTP request.
pub fn qwen_linear_mtp_relaxed_session_enabled() -> bool {
    QWEN_LINEAR_MTP_RELAXED_SESSION_SCOPE.with(|current| current.get())
}

env_flag_default_on!(
    /// `AX_MLX_MTP_ASYNC_DUAL_GATE_UP` — co-submit the two dense FFN
    /// projections during relaxed Qwen MTP request work. Ordinary direct
    /// sessions and row-exact verification are unchanged.
    ///
    /// **Default: ON inside the relaxed MTP session scope only**
    /// (kill-switch via `AX_MLX_MTP_ASYNC_DUAL_GATE_UP=0`).
    mtp_async_dual_gate_up_enabled,
    "AX_MLX_MTP_ASYNC_DUAL_GATE_UP"
);

/// Whether one short Qwen MTP FFN should co-submit gate/up qmm.
pub fn should_mtp_async_dual_gate_up(model_family: &str, seq: i32) -> bool {
    should_mtp_async_dual_gate_up_for(
        mtp_async_dual_gate_up_enabled(),
        qwen_linear_mtp_relaxed_session_enabled(),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_mtp_async_dual_gate_up`].
pub fn should_mtp_async_dual_gate_up_for(
    enabled: bool,
    relaxed_session: bool,
    model_family: &str,
    seq: i32,
) -> bool {
    enabled
        && relaxed_session
        && !qwen_linear_mtp_whole_verify_trace_enabled()
        && (2..=4).contains(&seq)
        && model_family.eq_ignore_ascii_case("qwen3_5")
}

env_flag_default_on!(
    /// `AX_MLX_MTP_LA_OUT_PROJ_SILU_MUL_QMM` — fuse gated RMS output
    /// preparation into the quantized linear-attention output projection
    /// during relaxed Qwen MTP request work. Ordinary direct sessions and
    /// row-exact verification retain their existing arithmetic.
    ///
    /// **Default: ON inside the relaxed MTP session scope only**
    /// (kill-switch via `AX_MLX_MTP_LA_OUT_PROJ_SILU_MUL_QMM=0`).
    mtp_la_out_proj_silu_mul_qmm_enabled,
    "AX_MLX_MTP_LA_OUT_PROJ_SILU_MUL_QMM"
);

env_flag!(
    /// `AX_MLX_MTP_LINEAR_PROJECTED_REPLAY` — let a Qwen gated-delta MTP
    /// verifier adopt/restore its clone and, after a partial accept, reuse the
    /// projected QKV/A/B tensors to rebuild only recurrent state.
    ///
    /// This is the narrowly scoped oMLX 0.6.2 rollback design: full-attention
    /// KV is trimmed normally, while each linear layer replays its accepted
    /// prefix from the unchanged pre-verify state. When the exact profile is
    /// explicitly disabled this also opts into oMLX-style verifier arithmetic,
    /// which is target-verified but not guaranteed bit-identical to singleton
    /// direct decode. Default OFF until matched M5 admission completes.
    mtp_linear_projected_replay_enabled,
    "AX_MLX_MTP_LINEAR_PROJECTED_REPLAY"
);

env_flag!(
    /// `AX_MLX_MTP_RELAXED_TARGET_VERIFY` — keep the exact Qwen MTP draft
    /// head active while building the target verifier with stock MLX
    /// arithmetic. Requires projected replay so accepted recurrent state is
    /// derived from the same verifier graph. This is an explicit oMLX-style
    /// non-bit-exact performance experiment and is OFF by default.
    mtp_relaxed_target_verify_enabled,
    "AX_MLX_MTP_RELAXED_TARGET_VERIFY"
);

env_flag!(
    /// `AX_MLX_MTP_SPLIT_VERIFY_HIDDEN_EVAL` — materialize the target trunk's
    /// post-norm hidden rows before scheduling the verify-window LM head and
    /// argmax. This mirrors MTPLX's lazy-logits boundary and can reduce mixed
    /// graph co-residency on short Qwen verifier windows.
    ///
    /// **Default: OFF** pending matched M5 admission.
    mtp_split_verify_hidden_eval_enabled,
    "AX_MLX_MTP_SPLIT_VERIFY_HIDDEN_EVAL"
);

env_flag!(
    /// `AX_MLX_MTP_LAZY_ADOPT_STATE` — leave an accepted relaxed Qwen MTP
    /// verifier cache lazy until the next target forward consumes it. The
    /// acceptance barrier has already materialized the verifier logits; an
    /// immediate second eval of cache side outputs can serialize otherwise
    /// adjacent verifier graphs.
    ///
    /// **Default: OFF** pending matched M5 admission.
    mtp_lazy_adopt_state_enabled,
    "AX_MLX_MTP_LAZY_ADOPT_STATE"
);

env_flag!(
    /// `AX_MLX_MTP_REBIND_VERIFY_FA` — after a relaxed Qwen verifier has
    /// produced replacement full-attention K/V buffers, rebind the rollback
    /// source cache to those outputs before the evaluation fence.  The source
    /// retains its pre-verify gated-delta states and logical length, so partial
    /// replay and fallback recompute remain valid, while the superseded K/V
    /// handles no longer prevent MLX from donating `slice_update` inputs.
    ///
    /// **Default: OFF** pending matched M5 admission.
    mtp_rebind_verify_fa_enabled,
    "AX_MLX_MTP_REBIND_VERIFY_FA"
);

env_flag!(
    /// `AX_MLX_MTP_LINEAR_TAPE_CAPTURE` — record the compact gated-delta
    /// recurrence tape during a relaxed Qwen verifier instead of writing a
    /// second full recurrent-state checkpoint for every linear layer. Misses
    /// and partial accepts reconstruct only their committed prefix from the
    /// unchanged source state.
    ///
    /// **Default: OFF** pending matched M5 admission.
    mtp_linear_tape_capture_enabled,
    "AX_MLX_MTP_LINEAR_TAPE_CAPTURE"
);

env_flag!(
    /// `AX_MLX_MTP_SKIP_PREFIX_CHECKPOINT` — retain the verifier's projected
    /// QKV/A/B inputs but do not write a full recurrent-state checkpoint at
    /// the confirmed row. Accepted cycles pay no recovery cost; rejected
    /// cycles replay only their committed prefix from the unchanged source
    /// state, matching the oMLX/MTPLX rollback shape.
    ///
    /// **Default: OFF** pending matched M5 admission.
    mtp_skip_prefix_checkpoint_enabled,
    "AX_MLX_MTP_SKIP_PREFIX_CHECKPOINT"
);

env_flag!(
    /// `AX_MLX_MTP_REUSE_PROCESSED_GDN` — retain the verifier's already
    /// materialized normalized Q/K/V rows and reuse their accepted prefix
    /// during gated-delta rollback. This avoids repeating depthwise conv and
    /// Q/K normalization on misses and partial accepts; the recurrent update
    /// still runs from the unchanged pre-verify state.
    ///
    /// **Default: OFF** pending matched M5 admission.
    mtp_reuse_processed_gdn_enabled,
    "AX_MLX_MTP_REUSE_PROCESSED_GDN"
);

env_flag!(
    /// `AX_MLX_MTP_GDN_PREWORK_SIMD32` — use a SIMD-width, four-values-per-lane
    /// conv/SiLU/QK-normalization kernel for short Qwen target-verifier blocks.
    ///
    /// **Default: OFF** pending matched M5 admission.
    mtp_gdn_prework_simd32_enabled,
    "AX_MLX_MTP_GDN_PREWORK_SIMD32"
);

env_flag!(
    /// `AX_MLX_MTP_FUSED_GDN_VERIFY` — fuse the short Qwen verifier's
    /// depthwise conv, Q/K normalization, and gated-delta recurrence into one
    /// Metal dispatch. The kernel preserves activation-dtype rounding and the
    /// recurrent update order, and falls back on any unsupported shape.
    ///
    /// **Default: OFF** pending matched token-hash and M5 throughput admission.
    mtp_fused_gated_delta_verify_enabled,
    "AX_MLX_MTP_FUSED_GDN_VERIFY"
);

env_flag!(
    /// `AX_MLX_MTP_REFOLD_ACCEPTED_HISTORY` — rebuild committed Qwen MTP-head
    /// history from target-backbone hidden rows after each verify cycle. The
    /// draft cache is one position behind the proposed token list, so this also
    /// fixes the legacy rejection trim's shifted-entry accounting. OFF by
    /// default until matched acceptance and throughput admission completes.
    mtp_refold_accepted_history_enabled,
    "AX_MLX_MTP_REFOLD_ACCEPTED_HISTORY"
);

env_flag!(
    /// `AX_MLX_MTP_BATCHED_COMMITTED_FOLD` — fold accepted drafts plus the
    /// correction/bonus token through the Qwen MTP head in one batched pass,
    /// then seed the next greedy draft chain from the final folded row. This
    /// mirrors oMLX's committed-history cycle and avoids a second first-depth
    /// read of the MTP-head weights.
    ///
    /// **Default: OFF** pending matched M5 admission.
    mtp_batched_committed_fold_enabled,
    "AX_MLX_MTP_BATCHED_COMMITTED_FOLD"
);

env_flag!(
    /// `AX_MLX_MTP_LAST_COMMITTED_QUERY` — during a batched committed-history
    /// fold, project the attention query/gate only for the final row. Earlier
    /// rows contribute K/V history but their attention outputs are discarded.
    ///
    /// **Default: OFF** pending matched M5 admission.
    mtp_last_committed_query_enabled,
    "AX_MLX_MTP_LAST_COMMITTED_QUERY"
);

/// Widest sequence shape the exact verifier contract covers: S=1 singleton
/// replay plus S=2..=4 verify (`QWEN_LINEAR_EXACT_MAX_VERIFY_DRAFTS` = 3
/// drafts + bonus token).
pub const QWEN_LINEAR_MTP_EXACT_MAX_EXACT_SEQ: i32 = 4;

/// Exact-contract check scoped to the decode shapes the verify/replay
/// invariant actually constrains. The invariant is between the S=2..=4
/// verify forward and the S=1 in-session singleton replay — both consume the
/// same cache state, so it holds regardless of how earlier prefill chunks
/// (seq > 4) computed their projections. Callers guarding fusion decisions
/// that also affect prefill must use this seq-aware variant so fused prefill
/// kernels stay enabled under the exact profile; callers guarding
/// decode-only paths keep [`qwen_linear_mtp_exact_enabled`].
///
/// Caveat: fused prefill can flip greedy argmax on near-ties, so a run under
/// the narrowed scope is a new configuration whose token stream may differ
/// from the fully de-fused one — the verify/replay correctness mode is
/// unaffected.
pub fn qwen_linear_mtp_exact_for_seq(seq: i32) -> bool {
    qwen_linear_mtp_exact_enabled() && seq <= QWEN_LINEAR_MTP_EXACT_MAX_EXACT_SEQ
}

/// Exact S=2..=4 verify: `async_eval` kernel-boundary tensors (fused QKVZ+BA
/// projections, GatedDelta, FA SDPA) so their GPU work overlaps host encode
/// of the still-lazy portable RMS+SiLU gate + o_proj.
///
/// Factory `69522a58` kept trial-2 `39a36e3f` but `--full` washed/regressed
/// (verify work moved forward→eval; general-long 1.038→1.023). Unhooked.
/// Does not eval `hidden`, residual, or the portable gate output — those
/// grouping changes reproduced factory trial-2 `f4b5490d`.
#[cfg_attr(not(test), allow(dead_code))]
pub fn should_exact_verify_async_kernel_boundary(seq: i32) -> bool {
    qwen_linear_mtp_exact_enabled() && (2..=4).contains(&seq)
}

env_flag!(
    /// `AX_MLX_INVARIANT_MXFP4_QMV_FAST` — S=2 exact verify uses the
    /// microbatch MXFP4 `fp_qmv_fast` clone (weights loaded once, per-row
    /// arithmetic matches isolated MLX S=1). Default OFF: factory `--full`
    /// on Qwen3.8-27B MXFP4 diverged on agent-coding and ran general-long
    /// at ~0.84× vs MLX's own M=2 path.
    invariant_mxfp4_qmv_fast_enabled,
    "AX_MLX_INVARIANT_MXFP4_QMV_FAST"
);

env_flag!(
    /// `AX_MLX_QWEN_DENSE_FFN_MATVEC_EXT_BITS` — admit 6-bit and 8-bit
    /// affine weights into the Qwen decode gate/up SwiGLU and down matvec
    /// Metal kernels (6-bit via the MLX `qdot` 4-values-per-3-bytes
    /// unpacker; 8-bit via the existing power-of-2 shift unpack). Default
    /// OFF pending a formal A/B on the 6-bit flagship and 8-bit packs;
    /// 4-bit engagement is unchanged and stays under
    /// `AX_MLX_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL`.
    qwen_dense_ffn_matvec_ext_bits_enabled,
    "AX_MLX_QWEN_DENSE_FFN_MATVEC_EXT_BITS"
);

env_flag_default_on!(
    /// `AX_MLX_EXACT_DENSE_WEIGHT_T_GEMV` — under the exact MTP profile,
    /// dense projections whose head carries a contiguous `[in, out]`
    /// `decode_weight_t` read it directly through the multi-row GEMV
    /// instead of the invariant dense kernel. The invariant kernel reads
    /// `qw.weight`, which `prepare_contiguous_decode_weight_t` replaced
    /// with a lazy transpose view — `ensure_row_contiguous` then
    /// re-materializes the full head (2.54 GB on Qwen3.8-27B) on **every**
    /// draft and verify call. Measured on the 6bit-MTP pack (M5): draft
    /// 18.6 ms/step and verify-eval 22.9 ms/step against 4.1 / 6.4 on the
    /// quantized-head MXFP4 sibling — the whole 0.80× MTP regression.
    /// One arithmetic serves S=1..8 so MTP-off and verify stay mutually
    /// consistent; greedy streams may shift vs the old kernel, so packs
    /// certified on it need a re-cert pass.
    ///
    /// **Default: ON** (kill-switch via
    /// `AX_MLX_EXACT_DENSE_WEIGHT_T_GEMV=0`).
    exact_dense_weight_t_gemv_enabled,
    "AX_MLX_EXACT_DENSE_WEIGHT_T_GEMV"
);

env_flag_default_on!(
    /// `AX_MLX_MTP_DENSE_HEAD_DRAFT_Q4` — when the target lm_head is dense
    /// (unquantized) and no draft-head spec was configured, derive a 4-bit
    /// gs64 `draft_lm_head` at MTP load. Draft logits only propose tokens
    /// (verify decides on the target head's own arithmetic); the dense head
    /// otherwise costs a full-weight read per draft step — 2.54 GB on
    /// Qwen3.8-27B, ~6.4 ms of the 11.5 ms draft wall measured on the
    /// 6bit-MTP pack. 4-bit argmax tracks bf16 closely; the 2-bit decode
    /// overlay tried first collapsed acceptance and tripped the MTP bypass
    /// gate (2026-08-19 M5 A/B). Costs one ~320 MB buffer at load.
    ///
    /// **Default: ON** (kill-switch via `AX_MLX_MTP_DENSE_HEAD_DRAFT_Q4=0`).
    mtp_dense_head_draft_q4_enabled,
    "AX_MLX_MTP_DENSE_HEAD_DRAFT_Q4"
);

env_flag!(
    /// `AX_MLX_MTP_FORCE_REQUESTED` — treat every pack as certified for
    /// *default-on* model MTP, bypassing the `axquant_runtime.json` `"mtp"`
    /// certification gate (`enabled_by_default` + `optimized`/measured
    /// speedup >= 1.0x). For formal benches and certification runs against
    /// packs whose metadata has not been stamped yet. Route safety
    /// (`MtpModelPolicy::route_safe`), `--ax-direct`, and `AX_NO_SPEC`
    /// still apply — this only neutralizes the certification check, it
    /// cannot resurrect an unsafe or explicitly disabled route.
    /// Default OFF.
    mtp_force_requested,
    "AX_MLX_MTP_FORCE_REQUESTED"
);

env_flag!(
    /// `AX_MLX_DENSE_WIDE_GEMV` — dense (unquantized) projections with a
    /// contiguous `[in, out]` `decode_weight_t` and 1..=8 leading rows take
    /// a multi-row Metal GEMV that reads each weight element once and FMAs
    /// it against every row. MLX has no dense analogue of `qmv_wide`, so
    /// the S=1→2 step on its steel GEMM costs ~1.97× (measured M3 Max) —
    /// this is the MTP-verify / small-cohort shape on dense lm_heads.
    /// Default OFF pending the formal A/B.
    dense_wide_gemv_enabled,
    "AX_MLX_DENSE_WIDE_GEMV"
);

env_flag!(
    /// `AX_MLX_EXACT_RMS_GATE_METAL` — keep the fused RMS+SiLU gate Metal
    /// kernels under the exact MTP profile instead of the blanket portable
    /// fallback. Default OFF: factory MXFP4 measured any Metal gate on
    /// MTP-on flipping token 41 vs MTP-off (`f4b5490d`), and the exact
    /// profile has skipped the gate for every pack (affine included) since.
    /// A/B lever: if that flip was multi-token SDPA reduction drift (now
    /// handled by the singleton-fold / per-position verify paths) rather
    /// than the gate kernel itself, this returns rms_norm+silu+multiply+
    /// astype per LA layer per step to one Metal dispatch.
    exact_rms_gate_metal_enabled,
    "AX_MLX_EXACT_RMS_GATE_METAL"
);

env_flag!(
    /// `AX_MLX_MTP_ASYNC_DRAFT` — schedule the greedy zero-gate MTP draft
    /// with `async_eval` and defer host token extraction to the start of the
    /// next decode cycle, overlapping the draft head's GPU forward with
    /// per-token host work (detokenization, stream emission).
    ///
    /// **Default: OFF** (opt-in via `AX_MLX_MTP_ASYNC_DRAFT=1`).
    ///
    /// Exactness-preserving: the identical lazy draft graph is evaluated;
    /// only the synchronization point moves. Engages under the exact profile
    /// or the explicit projected-replay profile, with the confidence gate
    /// disabled, non-stochastic drafting, and skip-state off — the regime
    /// where the synchronous greedy path computes no log-probs or distributions.
    mtp_async_draft_enabled,
    "AX_MLX_MTP_ASYNC_DRAFT"
);

env_flag!(
    /// `AX_MLX_GEMMA_DUAL_GATE_UP_METAL` — multi-token Gemma dense FFN dual
    /// gate/up affine-quantized Metal kernel with fused GEGLU product.
    ///
    /// **Default: OFF** (opt-in via `AX_MLX_GEMMA_DUAL_GATE_UP_METAL=1`).
    ///
    /// Profile residual (mbp-m5 pure Gemma 13.8k): gate_up dual qmm ~3.3s
    /// (~38% pure wall); thr≥1.15 needs ~11% pure cut. v1/v2 Metal regressed
    /// pure wall (X re-read / idle-thread K stride). v3 is tiled GEMM
    /// (BM×BN×BK, full-TG coop loads + fused gelu_approx*up). Production stays
    /// on two MLX qmm + Metal GEGLU until pure wall proves a ≥~7.5% cut.
    gemma_dual_gate_up_metal_enabled,
    "AX_MLX_GEMMA_DUAL_GATE_UP_METAL"
);

env_flag_default_on!(
    /// `AX_MLX_LAYER_SCALAR_FUSED_ADD` — fuse Gemma-family residual add plus
    /// scalar layer-scale multiply into one custom MLX Metal elementwise node.
    ///
    /// **Default: ON** (kill-switch via
    /// `AX_MLX_LAYER_SCALAR_FUSED_ADD=0`).
    ///
    /// Gemma 4 E2B ships one scalar per layer (`shape=[1]`). The unfused path
    /// emits `add` then `multiply` in every decoder layer for every direct
    /// decode token. This path only engages when both inputs have identical
    /// shape/dtype and the scalar has exactly one element; all other shapes use
    /// the normal broadcast-safe MLX ops.
    layer_scalar_fused_add_enabled,
    "AX_MLX_LAYER_SCALAR_FUSED_ADD"
);

env_flag_default_on!(
    /// `AX_MLX_GEMMA4_PER_LAYER_INPUT_GATE_COMPILE` — compile the exact
    /// Gemma4 per-layer-input `gelu_approx(gate) * input` decode activation.
    ///
    /// **Default: ON** (kill-switch via
    /// `AX_MLX_GEMMA4_PER_LAYER_INPUT_GATE_COMPILE=0`).
    ///
    /// Uses a fixed-shape closure because E2B/E4B decode is always
    /// `[1, 1, D]`; this avoids the shapeless GEGLU stream failure observed on
    /// older MLX releases while matching mlx-lm's compiled GELU activation.
    gemma4_per_layer_input_gate_compile_enabled,
    "AX_MLX_GEMMA4_PER_LAYER_INPUT_GATE_COMPILE"
);

env_flag!(
    /// `AX_MLX_FUSED_PREFILL_ATTENTION` — collapse the offset-0 multi-token
    /// prefill attention chain (attn RMSNorm → packed-QKV qmm → per-head QK
    /// norm → RoPE → maskless "causal" SDPA → o-proj qmm) into one C++ shim
    /// call per layer (mlxcel `fused_causal_prefill_attention` residual,
    /// mlx_cxx_bridge.cpp ~4028).
    ///
    /// **Default: OFF** (opt-in A/B). Phase-1 eligibility is strict: first
    /// chunk only (`token_offset == 0`, empty cache), packed affine QKV,
    /// Gemma-family text layers without KV sharing, mrope, value norm, rope
    /// freq tables, rings, or protected prefixes; sliding-window layers only
    /// when `seq <= window` (causal ≡ windowed there). SDPA runs in the
    /// model dtype via MLX fast SDPA rather than the portable
    /// full-precision route, so outputs are close-but-not-bit-identical —
    /// keep opt-in until a greedy token-exactness pass is recorded.
    fused_prefill_attention_enabled,
    "AX_MLX_FUSED_PREFILL_ATTENTION"
);

env_flag!(
    /// `AX_MLX_QWEN_FUSED_PREFILL_ATTENTION` — Qwen-family entry to the
    /// mlxcel `fused_causal_prefill_attention` residual (attn RMSNorm →
    /// QKV → QK-norm → RoPE → maskless causal SDPA → o-proj).
    ///
    /// **Default: OFF**. Offset-0 remasured p2048 895.52 vs 891.02 (2026-08-13);
    /// offset chunks crashed SSE. Gemma stays on `AX_MLX_FUSED_PREFILL_ATTENTION`
    /// (also OFF). Not an FFN host-fusion/compile lever.
    qwen_fused_prefill_attention_enabled,
    "AX_MLX_QWEN_FUSED_PREFILL_ATTENTION"
);

env_flag!(
    /// `AX_MLX_QWEN_LINEAR_ADD_RMS_NORM` — fuse `hidden + attn` with the
    /// pre-FFN RMSNorm on Qwen linear-attention layers (`add_rms_norm_pair`).
    ///
    /// **Default: OFF**. AXQ remasured p2048 890.38 vs 891.02 (2026-08-13).
    /// Full-attn `standard::layer_forward` already uses this pair.
    qwen_linear_add_rms_norm_enabled,
    "AX_MLX_QWEN_LINEAR_ADD_RMS_NORM"
);

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_INTERLAYER_ADD_RMS` — on Qwen linear→linear
    /// prefill boundaries, defer the post-FFN residual add and fuse it with
    /// the next layer's attn RMSNorm (`add_rms_norm_pair`).
    ///
    /// **Default: OFF**. Community p2048 904.845/858=1.054599 (0.996× standing
    /// 908.5, 2026-08-13). AXQ p2048 886.952/862.825=1.027962 (0.995× q2only).
    /// Same class as linear-attn add_rms wash. Not FFN compile.
    qwen_prefill_interlayer_add_rms_enabled,
    "AX_MLX_QWEN_PREFILL_INTERLAYER_ADD_RMS"
);

/// Whether Qwen generate prefill should fuse post-FFN add into the next
/// linear layer's attn RMSNorm.
pub fn should_qwen_prefill_interlayer_add_rms(model_family: &str, seq: i32) -> bool {
    should_qwen_prefill_interlayer_add_rms_for(
        qwen_prefill_interlayer_add_rms_enabled(),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_qwen_prefill_interlayer_add_rms`].
pub fn should_qwen_prefill_interlayer_add_rms_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
) -> bool {
    enabled
        && seq > 1
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

/// Whether this linear layer should stash raw FFN for the next linear layer.
pub fn should_defer_qwen_prefill_ffn_residual(
    model_family: &str,
    seq: i32,
    layer_idx: usize,
    next_is_linear: bool,
    skip_post_attention_ffn: bool,
) -> bool {
    should_defer_qwen_prefill_ffn_residual_for(
        should_qwen_prefill_interlayer_add_rms(model_family, seq),
        next_is_linear,
        skip_post_attention_ffn,
        layer_idx,
    )
}

/// Pure helper for [`should_defer_qwen_prefill_ffn_residual`].
pub fn should_defer_qwen_prefill_ffn_residual_for(
    interlayer_enabled: bool,
    next_is_linear: bool,
    skip_post_attention_ffn: bool,
    layer_idx: usize,
) -> bool {
    interlayer_enabled && next_is_linear && !skip_post_attention_ffn && layer_idx < usize::MAX
}

/// Families whose full-attn prefill can use the fused causal chain.
pub fn fused_prefill_attention_family_supported(model_family: &str) -> bool {
    matches!(
        model_family,
        "gemma4" | "gemma4_vl" | "gemma3" | "qwen3_5" | "qwen3_next"
    )
}

env_flag_default_on!(
    /// `AX_MLX_GEMMA4_FUSED_PREFILL_ATTENTION_P128` — Gemma 4 contract p128
    /// only: attempt mlxcel `fused_causal_prefill_attention` (attn RMSNorm →
    /// packed QKV → QK-norm → RoPE → maskless causal SDPA → o-proj). Profile
    /// residual on `df-macbookpro-m5`: p128 prefill is layer-stack / first KV
    /// dominated; 80/96 layers are sliding with `seq <= window` so causal ≡
    /// windowed. Global `AX_MLX_FUSED_PREFILL_ATTENTION` stays OFF. Kill with
    /// `AX_MLX_GEMMA4_FUSED_PREFILL_ATTENTION_P128=0`.
    gemma4_fused_prefill_attention_p128_enabled,
    "AX_MLX_GEMMA4_FUSED_PREFILL_ATTENTION_P128"
);

/// Whether Gemma 4 contract p128 should attempt fused causal prefill attention.
pub fn should_gemma4_fused_prefill_p128(model_family: &str, seq: i32) -> bool {
    should_gemma4_fused_prefill_p128_for(
        gemma4_fused_prefill_attention_p128_enabled(),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_gemma4_fused_prefill_p128`].
pub fn should_gemma4_fused_prefill_p128_for(enabled: bool, model_family: &str, seq: i32) -> bool {
    enabled
        && seq == 128
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "gemma4" | "gemma4_unified"
        )
}

/// Fold Gemma sandwich `post_attention_layernorm` into the fused p128 C++
/// call so the first-KV layer does not pay a second RMS FFI after o-proj.
pub fn should_gemma4_fused_prefill_fold_post_norm(
    model_family: &str,
    seq: i32,
    has_post_norm: bool,
) -> bool {
    should_gemma4_fused_prefill_fold_post_norm_for(
        gemma4_fused_prefill_attention_p128_enabled(),
        model_family,
        seq,
        has_post_norm,
    )
}

/// Pure helper for [`should_gemma4_fused_prefill_fold_post_norm`].
pub fn should_gemma4_fused_prefill_fold_post_norm_for(
    fused_p128_enabled: bool,
    model_family: &str,
    seq: i32,
    has_post_norm: bool,
) -> bool {
    has_post_norm && should_gemma4_fused_prefill_p128_for(fused_p128_enabled, model_family, seq)
}

/// Whether this family should attempt fused causal prefill attention.
/// Qwen stays on its default-OFF flag. Gemma contract p128 uses
/// [`should_gemma4_fused_prefill_p128`]; other Gemma shapes keep the global
/// default-OFF probe.
pub fn fused_prefill_attention_should_try(model_family: &str) -> bool {
    fused_prefill_attention_should_try_for_seq(model_family, 0)
}

/// Sequence-aware entry used by the shipped layer forward.
pub fn fused_prefill_attention_should_try_for_seq(model_family: &str, seq: i32) -> bool {
    if !fused_prefill_attention_family_supported(model_family) {
        return false;
    }
    if model_family.starts_with("qwen") {
        return qwen_fused_prefill_attention_enabled();
    }
    if should_gemma4_fused_prefill_p128(model_family, seq) {
        return true;
    }
    fused_prefill_attention_enabled()
}

/// Qwen p2048's second 1024-token chunk crashed the offset fused
/// `qkv_rope_split` + `sdpa_oproj` pair. Offset-0 one-shot fuse stays on.
pub fn fused_prefill_qwen_skip_offset(model_family: &str, offset_chunk: bool) -> bool {
    model_family.starts_with("qwen") && offset_chunk
}

env_flag!(
    /// `AX_MLX_QWEN_SKIP_LINEAR_PREFILL_MASK` — do not materialize SDPA
    /// causal/offset masks for linear-attention layers.
    ///
    /// **Default: OFF**. Remasured binary `39ea84c7…` (2026-08-13): community
    /// p2048 904.629/858=1.054347; AXQ p2048 888.720/862.825=1.030012
    /// (0.997× q2only). Wash. Not FFN/add_rms/pipeline-block.
    qwen_skip_linear_prefill_mask_enabled,
    "AX_MLX_QWEN_SKIP_LINEAR_PREFILL_MASK"
);

/// Whether Qwen prefill should omit the SDPA mask on a linear-attn layer.
pub fn should_skip_linear_prefill_mask(model_family: &str, is_linear_layer: bool) -> bool {
    should_skip_linear_prefill_mask_for(
        qwen_skip_linear_prefill_mask_enabled(),
        model_family,
        is_linear_layer,
    )
}

/// Pure helper for [`should_skip_linear_prefill_mask`].
pub fn should_skip_linear_prefill_mask_for(
    enabled: bool,
    model_family: &str,
    is_linear_layer: bool,
) -> bool {
    enabled
        && is_linear_layer
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_EVAL_KV_ONLY` — on intermediate Qwen
    /// `forward_cache_only` chunks, `eval` KV + linear-state refs only.
    ///
    /// **Default: OFF**. Paired with skip-linear-mask; remasured wash
    /// (2026-08-13, `39ea84c7…`). Not the closed lazy-intermediate skip.
    qwen_prefill_eval_kv_only_enabled,
    "AX_MLX_QWEN_PREFILL_EVAL_KV_ONLY"
);

/// Whether an intermediate Qwen cache-only chunk should eval KV refs only.
pub fn should_qwen_prefill_eval_kv_only(
    model_family: &str,
    is_final_chunk: bool,
    total_tokens: usize,
) -> bool {
    should_qwen_prefill_eval_kv_only_for(
        qwen_prefill_eval_kv_only_enabled(),
        model_family,
        is_final_chunk,
        total_tokens,
    )
}

/// Pure helper for [`should_qwen_prefill_eval_kv_only`].
pub fn should_qwen_prefill_eval_kv_only_for(
    enabled: bool,
    model_family: &str,
    is_final_chunk: bool,
    total_tokens: usize,
) -> bool {
    enabled
        && !is_final_chunk
        && skip_cache_only_split_for_family(model_family, total_tokens)
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

env_flag_default_on!(
    /// `AX_MLX_EXACT_SIZE_FIRST_KV` — first prefill write stores K/V (and
    /// GLM-MLA latents) at the exact prompt length instead of
    /// `zeros` + `slice_update` into a `KV_CHUNK_TOKENS` (256) padded
    /// buffer.
    ///
    /// **Default: ON**. Contract p128 is the only Wave-1 prompt whose first
    /// write is not a 256-token multiple (`chunk_ceiling(128) == 256`).
    /// p512 / p2048 already take the aligned exact-store path, so the 2026-08-13
    /// p2048 remasure was a no-op. Kill with `AX_MLX_EXACT_SIZE_FIRST_KV=0`.
    exact_size_first_kv_enabled,
    "AX_MLX_EXACT_SIZE_FIRST_KV"
);

/// Whether a fresh-layer first write should store the prompt-sized buffer.
pub fn should_exact_size_first_kv(write_start: usize) -> bool {
    should_exact_size_first_kv_for(exact_size_first_kv_enabled(), write_start)
}

/// Pure helper for [`should_exact_size_first_kv`].
pub fn should_exact_size_first_kv_for(enabled: bool, write_start: usize) -> bool {
    enabled && write_start == 0
}

env_flag!(
    /// `AX_MLX_EXACT_SIZE_KV_GROW` — when a prefill chunk appends exactly
    /// at the current FA/MLA capacity and the new length is
    /// `KV_CHUNK_TOKENS`-aligned, `concatenate` old∥new instead of
    /// `zeros(new_cap)` + two `slice_update`s.
    ///
    /// **Default: OFF**. Remasured binary `1b4a9e67…` (2026-08-13): community
    /// p2048 903.997/858=1.053610; AXQ p2048 886.967/862.825=1.027980
    /// (0.995× q2only). Slight regression. Not FFN.
    exact_size_kv_grow_enabled,
    "AX_MLX_EXACT_SIZE_KV_GROW"
);

/// Whether a capacity-tight aligned grow should concatenate instead of zeros.
pub fn should_exact_size_kv_grow(
    write_start: usize,
    old_capacity: usize,
    write_end: usize,
    new_capacity: usize,
) -> bool {
    should_exact_size_kv_grow_for(
        exact_size_kv_grow_enabled(),
        write_start,
        old_capacity,
        write_end,
        new_capacity,
    )
}

/// Pure helper for [`should_exact_size_kv_grow`].
pub fn should_exact_size_kv_grow_for(
    enabled: bool,
    write_start: usize,
    old_capacity: usize,
    write_end: usize,
    new_capacity: usize,
) -> bool {
    enabled && write_start == old_capacity && write_end == new_capacity && write_end > old_capacity
}

env_flag!(
    /// `AX_MLX_SKIP_UNUSED_FULL_KV_VIEW_SLICE` — when the SDPA view is the
    /// entire FA backing buffer, return K/V directly instead of a no-op
    /// `slice`.
    ///
    /// **Default: OFF**. Remasured binary `6643006c…` (2026-08-13): community
    /// p2048 903.515/858=1.053048; AXQ p2048 887.394/862.825=1.028475
    /// (0.996× q2only). Wash. Not FFN.
    skip_unused_full_kv_view_slice_enabled,
    "AX_MLX_SKIP_UNUSED_FULL_KV_VIEW_SLICE"
);

/// Whether a full-buffer FA view can skip the identity slice.
pub fn should_skip_unused_full_kv_view_slice(
    view_start: usize,
    write_end: usize,
    capacity: usize,
) -> bool {
    should_skip_unused_full_kv_view_slice_for(
        skip_unused_full_kv_view_slice_enabled(),
        view_start,
        write_end,
        capacity,
    )
}

/// Pure helper for [`should_skip_unused_full_kv_view_slice`].
pub fn should_skip_unused_full_kv_view_slice_for(
    enabled: bool,
    view_start: usize,
    write_end: usize,
    capacity: usize,
) -> bool {
    enabled && view_start == 0 && write_end == capacity && capacity > 0
}

env_flag!(
    /// `AX_MLX_QWEN_SKIP_UNUSED_LA_OUT_RESHAPE` — after `rms_norm_gated`,
    /// skip `reshape([1,S,V])` when the tensor is already that shape
    /// before `out_proj` qmm.
    ///
    /// **Default: OFF**. Remasured binary `0928ddf1…` (2026-08-13): community
    /// p2048 904.239/858=1.053892; AXQ p2048 888.900/862.825=1.030221
    /// (0.998× q2only). Wash. Not the closed silu_mul fuse.
    qwen_skip_unused_la_out_reshape_enabled,
    "AX_MLX_QWEN_SKIP_UNUSED_LA_OUT_RESHAPE"
);

/// Whether LA `out_proj` can take `hidden` without a reshape.
pub fn should_skip_unused_la_out_reshape(shape: &[i32], seq: i32, value_dim: i32) -> bool {
    should_skip_unused_la_out_reshape_for(
        qwen_skip_unused_la_out_reshape_enabled(),
        shape,
        seq,
        value_dim,
    )
}

/// Pure helper for [`should_skip_unused_la_out_reshape`].
pub fn should_skip_unused_la_out_reshape_for(
    enabled: bool,
    shape: &[i32],
    seq: i32,
    value_dim: i32,
) -> bool {
    enabled && shape == [1, seq, value_dim]
}

env_flag!(
    /// `AX_MLX_QWEN_LA_REUSE_INITIAL_STATE_ZEROS` — reuse one Float32 zeros
    /// template for the initial gated-delta recurrent state instead of
    /// allocating `zeros` on every linear layer of p2048 chunk 1 (48
    /// identical shapes). Chunk 2 already has state. Not FFN, not
    /// GatedDelta tile/compile/contiguous.
    ///
    /// **Default: OFF**. Remasured binary `dc519b17…` (2026-08-13): community
    /// p2048 904.358/858=1.054032; AXQ p2048 888.016/862.825=1.029195
    /// (0.997× q2only). Wash. Not GatedDelta tile/compile/contiguous.
    qwen_la_reuse_initial_state_zeros_enabled,
    "AX_MLX_QWEN_LA_REUSE_INITIAL_STATE_ZEROS"
);

/// Whether the initial recurrent state should reuse a zeros template.
pub fn should_reuse_la_initial_state_zeros() -> bool {
    should_reuse_la_initial_state_zeros_for(qwen_la_reuse_initial_state_zeros_enabled())
}

/// Pure helper for [`should_reuse_la_initial_state_zeros`].
pub fn should_reuse_la_initial_state_zeros_for(enabled: bool) -> bool {
    enabled
}

env_flag!(
    /// `AX_MLX_PREFILL_CLEAR_CACHE_PER_CHUNK` — after each *intermediate*
    /// prefill chunk is evaluated, call MLX `clear_cache()` (return freelist
    /// to the OS / pool) before building the next chunk's graph.
    ///
    /// **Default: OFF**. Residual vs mlxcel `chunked_prefill_last_logits`
    /// (generate.rs / issue #672). On mbp-m5 pure Gemma 13.8k measured
    /// ~+1.9% cold wall vs final-only clear — thr path keeps final-only.
    /// Opt-in for peak-memory A/B.
    prefill_clear_cache_per_chunk_enabled,
    "AX_MLX_PREFILL_CLEAR_CACHE_PER_CHUNK"
);

/// `AX_MLX_PREFILL_TIME_DEBUG=1` — shared gate for prefill timing/engagement
/// diagnostics printed to stderr (see also the per-chunk build/eval split in
/// `generate.rs`). Diagnostic only.
pub fn prefill_time_debug_env() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("AX_MLX_PREFILL_TIME_DEBUG").as_deref() == Ok("1"))
}

/// Multi-model (sibling-resident) prefill-rotation hint.
///
/// Set by the server when more than one model is resident. Ring-rotated
/// multi-token prefill keeps SWA layers at O(window + chunk) storage, which
/// the S1 dual-model contract (Qwen3.5-9B stream + Gemma 4 12B long prefill,
/// M5 Max) measured as a decisive win over ordered prefill: Qwen stream
/// 19.65 vs 17.74 tok/s and Gemma prefill leg 8096 vs 9141 ms. Exclusive
/// single-model sessions keep ordered prefill (the
/// `AX_MLX_ROTATING_SLIDING_PREFILL` doc records the long-decode ring-carry
/// cost that motivated the default), so this hint scopes the rotation to
/// exactly the topology where it wins. `AX_MLX_ROTATING_SLIDING_PREFILL=1`
/// still forces rotation everywhere; `=0` only clears the env opt-in, not
/// this hint.
static SIBLING_PREFILL_ROTATION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Server hook: flag whether sibling models are resident (see
/// [`sibling_prefill_rotation`]).
pub fn set_sibling_prefill_rotation(enabled: bool) {
    SIBLING_PREFILL_ROTATION.store(enabled, std::sync::atomic::Ordering::Release);
}

/// Whether sibling-resident prefill rotation is currently requested.
///
/// `AX_MLX_SIBLING_PREFILL_ROTATION=0|off|false` hard-forces the hint off
/// (and `=1|on|true` forces it on) regardless of the server hook, so
/// A/B teardowns can isolate ring-rotated prefill storage. Unset keeps
/// the hint-driven behavior unchanged.
pub fn sibling_prefill_rotation() -> bool {
    static OVERRIDE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    match OVERRIDE.get_or_init(|| {
        std::env::var("AX_MLX_SIBLING_PREFILL_ROTATION")
            .ok()
            .and_then(|raw| match raw.trim().to_ascii_lowercase().as_str() {
                "0" | "off" | "false" => Some(false),
                "1" | "on" | "true" => Some(true),
                _ => None,
            })
    }) {
        Some(forced) => *forced,
        None => SIBLING_PREFILL_ROTATION.load(std::sync::atomic::Ordering::Acquire),
    }
}

env_flag_default_on!(
    /// `AX_MLX_ROTATING_SLIDING_DECODE` — use a rotating backing store for
    /// sliding-window KV layers on rollback-free direct greedy decode.
    ///
    /// **Default: ON** (kill-switch via
    /// `AX_MLX_ROTATING_SLIDING_DECODE=0`).
    ///
    /// This mirrors `mlx_lm`'s `RotatingKVCache` behavior for Gemma-family
    /// sliding-window layers once decode depth exceeds the window: SDPA sees a
    /// bounded window-sized backing store instead of a full-context buffer plus
    /// retained-window slice views. The runner only enables this for direct
    /// greedy decode, where no n-gram rollback or sampling replay is required.
    rotating_sliding_decode_enabled,
    "AX_MLX_ROTATING_SLIDING_DECODE"
);

env_flag!(
    /// `AX_MLX_ROTATING_SLIDING_PREFILL` — use rotating sliding-window KV
    /// during multi-token prefill (not only decode).
    ///
    /// **Default: OFF** (opt-in via `AX_MLX_ROTATING_SLIDING_PREFILL=1`).
    ///
    /// When enabled, prefill sizes rings as `window + prefill_chunk` so
    /// multi-token chunks fit the ring eligibility gate. That geometry cannot
    /// be safely shrunk for pure single-token decode without remapping, and
    /// keeping the oversized ring through long-context decode was measured to
    /// regress Gemma 4 decode@2048 by ~15–35% vs the 2026-07-14 pure-window
    /// path. Ordered prefill + pure window decode (default) restores that
    /// contract; enable this flag only for peak-memory A/B of long prefill.
    rotating_sliding_prefill_enabled,
    "AX_MLX_ROTATING_SLIDING_PREFILL"
);

env_flag_default_on!(
    /// `AX_MLX_ROTATING_BOUNDED_ROLLBACK` — extend rotating sliding-window
    /// KV to n-gram-active and sampled (non-greedy) requests via
    /// bounded-rollback rings.
    ///
    /// **Default: ON** (kill-switch via
    /// `AX_MLX_ROTATING_BOUNDED_ROLLBACK=0`); has no effect when
    /// `AX_MLX_ROTATING_SLIDING_DECODE=0` disables rotation entirely.
    ///
    /// Bounded rings allocate `window + slack` slots (slack covers the
    /// deepest n-gram verify forward, `MAX_DRAFT_LEN + 1`) so `trim_to` can
    /// roll back rejected draft tokens without reordering: a rolled-back
    /// token's successor rewrites the same `t % capacity` slot. SDPA over a
    /// bounded ring always carries a slot-validity mask
    /// (`create_ring_sliding_mask`). With this OFF, n-gram-active requests
    /// keep the pre-6.6.2 behavior: O(context) sliding-layer buffers with
    /// ordered window views; the rollback-free classes (direct sessions,
    /// sticky per-request n-gram disable) still rotate with pure
    /// window-sized rings.
    rotating_bounded_rollback_enabled,
    "AX_MLX_ROTATING_BOUNDED_ROLLBACK"
);

env_flag_default_on!(
    /// `AX_MLX_ROTATING_BOUNDED_MTP` — allow Gemma4 assistant-MTP requests
    /// onto bounded-rollback rotating rings.
    ///
    /// **Default: ON** (kill-switch via `AX_MLX_ROTATING_BOUNDED_MTP=0`);
    /// nested under `AX_MLX_ROTATING_BOUNDED_ROLLBACK` and
    /// `AX_MLX_ROTATING_SLIDING_DECODE`.
    ///
    /// The assistant's verify rollback is a `state.cache.trim_to` bounded by
    /// the pending draft (assistant depth + any stacked n-gram tokens), so
    /// the request latches a widened slack of
    /// `max(8, mtp_max_depth + MAX_DRAFT_LEN + 1)`. The drafter reads target
    /// sliding K/V through `peek_layer_kv`, which returns the full ring with
    /// a slot-validity mask once rotated. With this OFF, assistant-MTP
    /// requests keep O(context) sliding buffers (the pre-extension
    /// behavior); qwen/GLM MTP heads remain ring-excluded regardless (their
    /// models have no sliding windows).
    rotating_bounded_mtp_enabled,
    "AX_MLX_ROTATING_BOUNDED_MTP"
);

env_flag_default_on!(
    /// `AX_MLX_MULTI_TOKEN_WINDOW_VIEWS` — present sliding-window layers with a
    /// `window + seq - 1` retained K/V view on multi-token forwards (chunked
    /// prefill continuation chunks, n-gram verify, assistant-MTP verify)
    /// instead of the full-context view.
    ///
    /// **Default: ON** (kill-switch via `AX_MLX_MULTI_TOKEN_WINDOW_VIEWS=0`).
    ///
    /// Each query in a multi-token forward attends at most the `window` keys
    /// ending at its own position, so the chunk as a whole needs only the last
    /// `window + seq - 1` cached tokens. MLX masked SDPA does not skip
    /// masked-out K/V blocks, so the previous full-context view paid
    /// O(context) reads and scores per sliding layer per chunk — the dominant
    /// sliding-layer cost for long-context prefill and every speculative
    /// verify forward. `mlx_lm` gets the same bound from its
    /// `RotatingKVCache` prefill trim. Storage is unaffected (rollback and
    /// prefix-cache snapshots still see full backing buffers); only the view
    /// handed to SDPA and the matching mask width shrink. Multimodal
    /// media-overlay masks span the full context (media blocks may attend
    /// beyond the window), and the view width follows the hoisted mask, so
    /// those forwards keep full views.
    multi_token_window_views_enabled_env,
    "AX_MLX_MULTI_TOKEN_WINDOW_VIEWS"
);

/// Whether multi-token sliding layers use the retained window view.
///
/// Controlled by `AX_MLX_MULTI_TOKEN_WINDOW_VIEWS` (default ON). Pure-direct
/// rings and multi-token `window + seq - 1` views share geometry, so no
/// request-local override is required for Gemma assistant-MTP exactness.
pub fn multi_token_window_views_enabled() -> bool {
    multi_token_window_views_enabled_env()
}

thread_local! {
    static MOE_MT_BF16_IDENTITY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// MoE multi-token identity: keep SDPA/projections on bf16 singleton-compatible
/// path (no f32 multi-token upcast) so teacher-forced matches pure-direct.
#[must_use]
pub(crate) struct MoeMtBf16IdentityScope {
    prev: bool,
}

impl MoeMtBf16IdentityScope {
    pub(crate) fn new(enabled: bool) -> Self {
        let prev = MOE_MT_BF16_IDENTITY.with(|c| c.replace(enabled));
        Self { prev }
    }
}

impl Drop for MoeMtBf16IdentityScope {
    fn drop(&mut self) {
        MOE_MT_BF16_IDENTITY.with(|c| c.set(self.prev));
    }
}

pub(crate) fn scoped_moe_mt_bf16_identity(enabled: bool) -> MoeMtBf16IdentityScope {
    MoeMtBf16IdentityScope::new(enabled)
}

pub(crate) fn moe_mt_bf16_identity_enabled() -> bool {
    MOE_MT_BF16_IDENTITY.with(|c| c.get())
}

env_flag!(
    /// `AX_MLX_GEMMA_MT_PERPOS_FFN` — per-position dense FFN on short multi-token
    /// verify (seq 2..8). Improves 4-bit greedy exactness vs pure-direct but
    /// collapses multi-token speed (smokef12/17). Opt-in for 4-bit identity
    /// experiments; keep OFF for 6-bit / 31B formal Tier 2 speed.
    gemma_mt_perpos_ffn_enabled,
    "AX_MLX_GEMMA_MT_PERPOS_FFN"
);

env_flag_default_on!(
    /// `AX_MLX_MULTI_TOKEN_F32_ATTENTION` — upcast Q/K/V to f32 for SDPA on
    /// multi-token forwards (`seq > 1`) so teacher-forced verify stays closer
    /// to sequential singleton decode under bf16 accumulation drift.
    ///
    /// **Default: ON** for Gemma assistant-MTP Tier 2 greedy exactness
    /// (period-6 cycle-break near-ties). Kill-switch via
    /// `AX_MLX_MULTI_TOKEN_F32_ATTENTION=0`. Singleton decode (`seq == 1`) stays
    /// bf16 (enabling f32 on pure-direct regressed 12B6 general exactness).
    multi_token_f32_attention_enabled,
    "AX_MLX_MULTI_TOKEN_F32_ATTENTION"
);

env_flag_default_on!(
    /// `AX_MLX_GEMMA4_PREFILL_SKIP_UNUSED_F32_SDPA` — keep Gemma 4 contract
    /// prefill (`seq >= 128`) SDPA in the model dtype. The default-ON
    /// `AX_MLX_MULTI_TOKEN_F32_ATTENTION` upcast is for short teacher-forced
    /// MTP verify (`seq` 2..=8). On Gemma 4 p128 that upcast runs on every
    /// full-attention layer (96 on 12B) for a one-shot prefill mlxcel never
    /// pays. Decode `seq==1` and short MTP verify stay on f32. Not an MLP
    /// compile and not an attn-norm/QKV fuse. Kill with
    /// `AX_MLX_GEMMA4_PREFILL_SKIP_UNUSED_F32_SDPA=0`.
    gemma4_prefill_skip_unused_f32_sdpa_enabled,
    "AX_MLX_GEMMA4_PREFILL_SKIP_UNUSED_F32_SDPA"
);

/// Whether Gemma 4 contract prefill should skip the unused f32 SDPA upcast.
pub fn should_gemma4_prefill_skip_unused_f32_sdpa(model_family: &str, seq: i32) -> bool {
    should_gemma4_prefill_skip_unused_f32_sdpa_for(
        gemma4_prefill_skip_unused_f32_sdpa_enabled(),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_gemma4_prefill_skip_unused_f32_sdpa`].
pub fn should_gemma4_prefill_skip_unused_f32_sdpa_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
) -> bool {
    enabled
        && seq >= 128
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "gemma4" | "gemma4_unified"
        )
}

env_flag_default_on!(
    /// `AX_MLX_GEMMA4_PREFILL_SKIP_UNUSED_EMBED_CLIP` — skip the token-id
    /// `clip` before embedding gather on Gemma 4 contract prefill
    /// (`seq >= 128`). The clip exists so out-of-range client ids cannot
    /// read past the table; bench / generate prompts are in-range, and
    /// mlxcel never pays this gather-prep. Decode and short MTP verify
    /// keep the clip. Not an FFN compile. Kill with
    /// `AX_MLX_GEMMA4_PREFILL_SKIP_UNUSED_EMBED_CLIP=0`.
    gemma4_prefill_skip_unused_embed_clip_enabled,
    "AX_MLX_GEMMA4_PREFILL_SKIP_UNUSED_EMBED_CLIP"
);

env_flag_default_on!(
    /// `AX_MLX_GEMMA4_PREFILL_SKIP_UNUSED_LAST_RESIDUAL` — on the last
    /// transformer layer of Gemma 4 contract prefill (`seq >= 128`), slice
    /// residual inputs to the last token *before* add + pre-FFN RMSNorm.
    /// Fused causal still emits a full-seq last-layer `attn_proj`; the
    /// last-only FFN only needs the last residual row, so the prefix add
    /// is unused work mlxcel's lazy eval never pays. Does not
    /// `async_eval` and does not take last-query (that left fused and
    /// remasured wash). Decode and short MTP keep add-then-slice. Kill
    /// with `AX_MLX_GEMMA4_PREFILL_SKIP_UNUSED_LAST_RESIDUAL=0`.
    gemma4_prefill_skip_unused_last_residual_enabled,
    "AX_MLX_GEMMA4_PREFILL_SKIP_UNUSED_LAST_RESIDUAL"
);

env_flag!(
    /// `AX_MLX_GEMMA4_PREFILL_SKIP_UNUSED_LAST_FFN_PACKED` — on the last
    /// transformer layer of Gemma 4 contract prefill (`seq >= 128`), skip
    /// packed prefill qmm for the last-only 1-token FFN.
    ///
    /// **Default: OFF**. Remasured on `df-macbookpro-m5` (2026-08-15,
    /// `gemma4-axq-v7-lastffn` + repeat): 12B p128 647.28/657.70 vs fused
    /// 651.57/659.48 (0.982× / 0.997×). Leaving packed for a split last
    /// FFN does not move 1.10× and dipped the first fleet. Decode
    /// unharmed. Keep opt-in only.
    gemma4_prefill_skip_unused_last_ffn_packed_enabled,
    "AX_MLX_GEMMA4_PREFILL_SKIP_UNUSED_LAST_FFN_PACKED"
);

/// Whether Gemma 4 last-only prefill should skip unused packed prefill FFN.
pub fn should_gemma4_prefill_skip_unused_last_ffn_packed(
    model_family: &str,
    last_position_only: bool,
    seq: i32,
) -> bool {
    should_gemma4_prefill_skip_unused_last_ffn_packed_for(
        gemma4_prefill_skip_unused_last_ffn_packed_enabled(),
        model_family,
        last_position_only,
        seq,
    )
}

/// Pure helper for [`should_gemma4_prefill_skip_unused_last_ffn_packed`].
pub fn should_gemma4_prefill_skip_unused_last_ffn_packed_for(
    enabled: bool,
    model_family: &str,
    last_position_only: bool,
    seq: i32,
) -> bool {
    enabled
        && last_position_only
        && seq >= 128
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "gemma4" | "gemma4_unified"
        )
}

/// Whether Gemma 4 last-only prefill should skip unused prefix residual add.
pub fn should_gemma4_prefill_skip_unused_last_residual(
    model_family: &str,
    last_position_only: bool,
    seq: i32,
) -> bool {
    should_gemma4_prefill_skip_unused_last_residual_for(
        gemma4_prefill_skip_unused_last_residual_enabled(),
        model_family,
        last_position_only,
        seq,
    )
}

/// Pure helper for [`should_gemma4_prefill_skip_unused_last_residual`].
pub fn should_gemma4_prefill_skip_unused_last_residual_for(
    enabled: bool,
    model_family: &str,
    last_position_only: bool,
    seq: i32,
) -> bool {
    enabled
        && last_position_only
        && seq >= 128
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "gemma4" | "gemma4_unified"
        )
}

env_flag_default_on!(
    /// `AX_MLX_GEMMA4_PREFILL_BF16_EMBED` — dequantize Gemma 4 AXQ embed
    /// rows straight to BF16 on contract prefill (`seq >= 128`) and skip
    /// the follow-on `astype(..., bfloat16)` when the gather is already
    /// BF16. AXQ `embed_tokens` is 8-bit; the default path dequants to f32
    /// then casts. mlxcel never pays that unused f32 table. Does not
    /// `async_eval` and does not split the fused layer graph. Decode and
    /// short MTP keep f32 dequant + cast. Kill with
    /// `AX_MLX_GEMMA4_PREFILL_BF16_EMBED=0`.
    gemma4_prefill_bf16_embed_enabled,
    "AX_MLX_GEMMA4_PREFILL_BF16_EMBED"
);

/// Whether Gemma 4 contract prefill should dequant embeddings to BF16.
pub fn should_gemma4_prefill_bf16_embed(model_family: &str, seq: i32) -> bool {
    should_gemma4_prefill_bf16_embed_for(gemma4_prefill_bf16_embed_enabled(), model_family, seq)
}

/// Pure helper for [`should_gemma4_prefill_bf16_embed`].
pub fn should_gemma4_prefill_bf16_embed_for(enabled: bool, model_family: &str, seq: i32) -> bool {
    enabled
        && seq >= 128
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "gemma4" | "gemma4_unified"
        )
}

/// Whether Gemma 4 contract prefill should skip the unused embed-id clip.
pub fn should_gemma4_prefill_skip_unused_embed_clip(model_family: &str, seq: i32) -> bool {
    should_gemma4_prefill_skip_unused_embed_clip_for(
        gemma4_prefill_skip_unused_embed_clip_enabled(),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_gemma4_prefill_skip_unused_embed_clip`].
pub fn should_gemma4_prefill_skip_unused_embed_clip_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
) -> bool {
    enabled
        && seq >= 128
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "gemma4" | "gemma4_unified"
        )
}

env_flag_default_on!(
    /// `AX_MLX_GEMMA4_PREFILL_SKIP_UNUSED_LAYER_MASKS` — skip the per-forward
    /// `build_layer_masks_for_forward` hoist on Gemma 4 contract prefill
    /// (`seq >= 128`) when every layer would resolve to `None`. Fresh p128
    /// (and p512) sit inside the 1024-token sliding window, fused causal
    /// prefill is maskless, and `attention_mask_array` already returns
    /// `None`. mlxcel never allocates this per-layer `Vec`. Decode, short
    /// MTP verify, offset chunks, rotating rings, and `seq > window`
    /// (p2048) keep the hoist. Kill with
    /// `AX_MLX_GEMMA4_PREFILL_SKIP_UNUSED_LAYER_MASKS=0`.
    gemma4_prefill_skip_unused_layer_masks_enabled,
    "AX_MLX_GEMMA4_PREFILL_SKIP_UNUSED_LAYER_MASKS"
);

/// Whether Gemma 4 contract prefill should skip the unused layer-mask hoist.
pub fn should_gemma4_prefill_skip_unused_layer_masks(
    model_family: &str,
    seq: i32,
    key_len: usize,
    min_sliding_window: Option<usize>,
    rotating_slack: usize,
) -> bool {
    should_gemma4_prefill_skip_unused_layer_masks_for(
        gemma4_prefill_skip_unused_layer_masks_enabled(),
        model_family,
        seq,
        key_len,
        min_sliding_window,
        rotating_slack,
    )
}

/// Pure helper for [`should_gemma4_prefill_skip_unused_layer_masks`].
pub fn should_gemma4_prefill_skip_unused_layer_masks_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
    key_len: usize,
    min_sliding_window: Option<usize>,
    rotating_slack: usize,
) -> bool {
    if !enabled || rotating_slack > 0 || seq < 128 {
        return false;
    }
    if !matches!(
        model_family.to_ascii_lowercase().as_str(),
        "gemma4" | "gemma4_unified"
    ) {
        return false;
    }
    let seq_u = seq as usize;
    if key_len.saturating_sub(seq_u) > 0 {
        return false;
    }
    !matches!(min_sliding_window, Some(window) if seq_u > window)
}

env_flag!(
    /// `AX_MLX_GEMMA4_PREFILL_PIPELINE_HINT_P128` — after every non-final
    /// Gemma 4 contract-p128 layer, `async_eval(hidden)` so MLX can start
    /// layer N while the host builds N+1 (mlxcel `pipeline_hint` /
    /// `MLXCEL_PIPELINE_GRANULARITY=layer`).
    ///
    /// **Default: OFF**. Remasured on `df-macbookpro-m5` (2026-08-15,
    /// `gemma4-axq-v7-pipehint` + repeat): 12B p128 609.66/610.59 vs fused
    /// 651.57/659.48 (0.924× / 0.926×). 26B p128 0.978/0.980 vs baseline.
    /// Per-layer `async_eval` splits the fused lazy graph and undoes the
    /// skip-f32+fused win. Decode and p512/p2048 were unharmed. Keep
    /// opt-in only.
    gemma4_prefill_pipeline_hint_p128_enabled,
    "AX_MLX_GEMMA4_PREFILL_PIPELINE_HINT_P128"
);

env_flag!(
    /// `AX_MLX_GEMMA4_PREFILL_LAST_QUERY_P128` — on the last transformer
    /// layer of Gemma 4 contract p128, write full K/V then run Q / SDPA /
    /// o_proj on the last token only. Does **not** `async_eval` mid-graph.
    ///
    /// **Default: OFF**. Remasured on `df-macbookpro-m5` (2026-08-15,
    /// `gemma4-axq-v7-lastquery` + repeat): 12B p128 654.41/654.66 vs fused
    /// 651.57/659.48 (0.992× / 0.993×). One last-layer Q/SDPA/o_proj skip
    /// does not move 1.10× and leaves fused for a portable last layer.
    /// Decode unharmed. Keep opt-in only.
    gemma4_prefill_last_query_p128_enabled,
    "AX_MLX_GEMMA4_PREFILL_LAST_QUERY_P128"
);

/// Whether Gemma 4 last-only p128 should skip unused prefix Q / SDPA / o_proj.
pub fn should_gemma4_prefill_last_query_p128(
    model_family: &str,
    last_position_only: bool,
    seq: i32,
) -> bool {
    should_gemma4_prefill_last_query_p128_for(
        gemma4_prefill_last_query_p128_enabled(),
        model_family,
        last_position_only,
        seq,
    )
}

/// Pure helper for [`should_gemma4_prefill_last_query_p128`].
pub fn should_gemma4_prefill_last_query_p128_for(
    enabled: bool,
    model_family: &str,
    last_position_only: bool,
    seq: i32,
) -> bool {
    enabled
        && last_position_only
        && seq == 128
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "gemma4" | "gemma4_unified"
        )
}

/// Whether Gemma 4 contract p128 should submit a per-layer pipeline hint.
pub fn should_gemma4_prefill_pipeline_hint_p128(
    model_family: &str,
    seq: usize,
    layer_idx: usize,
    total_layers: usize,
) -> bool {
    should_gemma4_prefill_pipeline_hint_p128_for(
        gemma4_prefill_pipeline_hint_p128_enabled(),
        model_family,
        seq,
        layer_idx,
        total_layers,
    )
}

/// Pure helper for [`should_gemma4_prefill_pipeline_hint_p128`].
pub fn should_gemma4_prefill_pipeline_hint_p128_for(
    enabled: bool,
    model_family: &str,
    seq: usize,
    layer_idx: usize,
    total_layers: usize,
) -> bool {
    enabled
        && seq == 128
        && total_layers > 0
        && layer_idx + 1 < total_layers
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "gemma4" | "gemma4_unified"
        )
}

env_flag_default_on!(
    /// `AX_MLX_DENSE_LONG_MT_BF16_FOLD` — dense multi-token long history
    /// (`key_len >= 512`) uses the same bf16 singleton-query fold as MoE
    /// multi-token, avoiding full-history f32 K/V upcast.
    ///
    /// **Default: ON** for Gemma 12B/31B agent long Tier 2 speed (dense-sing-v4
    /// was exact under f32 fold at ~0.91× weighted). Kill-switch via
    /// `AX_MLX_DENSE_LONG_MT_BF16_FOLD=0` restores the prior f32 long fold.
    /// Short multi-token still uses f32 batched SDPA for general exactness.
    dense_long_mt_bf16_fold_enabled,
    "AX_MLX_DENSE_LONG_MT_BF16_FOLD"
);

env_flag!(
    /// `AX_MLX_DIRECT_CPP_GEMMA4_POST_ATTN_FFN` — opt-in direct C++ route for
    /// Gemma4 dense post-attention residual + FFN + layer-scalar orchestration.
    ///
    /// **Default: OFF — real-model A/B rejected promotion (2026-06-11).** The P0
    /// clean microbench artifact showed this large-block boundary beating the
    /// portable Rust/MLX FFI composition, but the full A/B on the two models that
    /// can engage the route (Gemma 4 31B and 12B 4-bit, `all_hits`) regressed
    /// decode to 0.89-0.97x and prefill to 0.91-0.98x;
    /// `check_direct_gemma4_ffn_route_promotion.py` decision: `not_promoted`.
    /// E2B/E4B (per-layer-embedding weights) and 26B-A4B (MoE router) cannot take
    /// the route at all. Artifacts:
    /// `benchmarks/results/inference/mlx-inference/2026-06-11-gemma4-ffn-route-ab/`.
    /// The production route is guarded to dense packed-quantized Gemma4 layers
    /// without per-layer input gating, profiling, last-position slicing, or
    /// active weight rotation.
    direct_cpp_gemma4_post_attn_ffn_enabled,
    "AX_MLX_DIRECT_CPP_GEMMA4_POST_ATTN_FFN"
);

env_flag!(
    /// `AX_MLX_DENSE_QMATMUL_RMS_NORM` — fuse the dense FFN down-projection
    /// and post-FFN RMSNorm into one C++ call (mlxcel post_feedforward_ln sits
    /// on FFN out before residual add; AX fuses down qmm + that rms).
    ///
    /// **Default: OFF**. First pure A/B on mbp-m5 looked ~6% faster but a
    /// 3-rep confirm (2026-07-25 pure-qmm-rms-confirm) was median ~1.00×
    /// (thermal noise). Prior Gemma 4 31B decode A/B also ~0.45% slower.
    /// Opt-in only until a cool multi-rep pure win holds.
    dense_qmatmul_rms_norm_enabled,
    "AX_MLX_DENSE_QMATMUL_RMS_NORM"
);

env_flag!(
    /// `AX_MLX_O_PROJ_QMATMUL_RMS_NORM` — fuse attention `o_proj` quantized
    /// matmul with Gemma sandwich `post_attention_layernorm` into one C++ call
    /// (`quantized_matmul_rms_norm`). Profile residual: pure Gemma 13.8k
    /// `post_attn_output_proj` ~0.78s (larger than rope_kv ~0.54s).
    ///
    /// mlxcel keeps `o_proj.forward` then `post_attention_layernorm.forward`
    /// as separate ops (`gemma4.rs` project_output + layer residual).
    ///
    /// **Default: OFF** (opt-in pure A/B). Only applies when `attn_post_norm`
    /// is present and there is no attention output gate.
    o_proj_qmatmul_rms_norm_enabled,
    "AX_MLX_O_PROJ_QMATMUL_RMS_NORM"
);

env_flag!(
    /// `AX_MLX_ATTN_NORM_QKV_FUSE` — fuse attention input RMSNorm with the
    /// packed dense QKV quantized matmul into one C++ call
    /// (`rms_norm_quantized_matmul`). Profile residual: pure Gemma 13.8k
    /// `pre_sdpa_qkv_proj` ~1.07s (next largest after gate_up / down / o_proj).
    ///
    /// mlxcel: `input_layernorm.forward` then separate Q/K/V (or opt-in fused
    /// QKV) — no norm+proj fuse (`gemma4.rs` layer residual).
    ///
    /// **Default: OFF** (opt-in pure A/B). Requires packed QKV + scales.
    attn_norm_qkv_fuse_enabled,
    "AX_MLX_ATTN_NORM_QKV_FUSE"
);

env_flag!(
    /// `AX_MLX_QWEN_ATTN_NORM_QKV_FUSE` — Qwen full-attn: fuse `attn_norm`
    /// with packed QKV `quantized_matmul` via `rms_norm_quantized_matmul`.
    ///
    /// **Default: OFF**. Four-lane remasure (binary `544a8b5d…`, 2026-08-13):
    /// community p2048 906.020/858.000=1.055968 (3d FAIL; 0.997× standing
    /// 908.5). p128 463.775 vs standing 472.770 (regression). AXQ --ax-direct
    /// panicked (`portable path materializes attn_norm`) when exact skipped
    /// the fuse after `normed` was cleared. Gemma stays OFF.
    qwen_attn_norm_qkv_fuse_enabled,
    "AX_MLX_QWEN_ATTN_NORM_QKV_FUSE"
);

env_flag_default_on!(
    /// `AX_MLX_GEMMA4_ATTN_NORM_QKV_FUSE_P128` — Gemma 4 contract p128 only:
    /// fuse `attn_norm` into the packed QKV `quantized_matmul` via
    /// `rms_norm_quantized_matmul`. Profile residual on `df-macbookpro-m5`:
    /// p128 prefill is forward-dominated; `pre_sdpa_qkv_proj` is the next
    /// named band after FFN (not another MLP compile). 80/96 Gemma 4 AXQ
    /// layers already hold packed QKV. p512/p2048 stay on the portable
    /// rms-then-qmm path. Kill with `AX_MLX_GEMMA4_ATTN_NORM_QKV_FUSE_P128=0`.
    gemma4_attn_norm_qkv_fuse_p128_enabled,
    "AX_MLX_GEMMA4_ATTN_NORM_QKV_FUSE_P128"
);

/// Whether Gemma 4 contract p128 should fuse attn RMSNorm into packed QKV.
pub fn should_gemma4_attn_norm_qkv_fuse_p128(model_family: &str, seq: i32) -> bool {
    should_gemma4_attn_norm_qkv_fuse_p128_for(
        gemma4_attn_norm_qkv_fuse_p128_enabled(),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_gemma4_attn_norm_qkv_fuse_p128`].
pub fn should_gemma4_attn_norm_qkv_fuse_p128_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
) -> bool {
    enabled
        && seq == 128
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "gemma4" | "gemma4_unified"
        )
}

/// Whether this family should fuse attn RMSNorm into packed QKV qmm.
pub fn should_attn_norm_qkv_fuse(model_family: &str, seq: i32) -> bool {
    should_attn_norm_qkv_fuse_for(
        qwen_attn_norm_qkv_fuse_enabled(),
        attn_norm_qkv_fuse_enabled(),
        gemma4_attn_norm_qkv_fuse_p128_enabled(),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_attn_norm_qkv_fuse`].
pub fn should_attn_norm_qkv_fuse_for(
    qwen_enabled: bool,
    global_enabled: bool,
    gemma4_p128_enabled: bool,
    model_family: &str,
    seq: i32,
) -> bool {
    if model_family.eq_ignore_ascii_case("qwen3_5") {
        return qwen_enabled;
    }
    if should_gemma4_attn_norm_qkv_fuse_p128_for(gemma4_p128_enabled, model_family, seq) {
        return true;
    }
    global_enabled
}

/// Whether the fused rms+QKV call will run. Exact / moe-mt identity skip
/// the fuse and still need a standalone `attn_norm`.
pub fn should_call_attn_norm_qkv_fuse(
    family_enabled: bool,
    packed_qkv: bool,
    has_kv_source: bool,
    skip_fuse: bool,
) -> bool {
    family_enabled && packed_qkv && !has_kv_source && !skip_fuse
}

env_flag!(
    /// `AX_MLX_DUAL_QMM_GEGLU` — multi-token split dense FFN: one C++ call for
    /// `gelu_approx(qmm(x,gate)) * qmm(x,up)` (no mx::compile). Profile residual:
    /// pure Gemma 13.8k `post_attn_ffn_gate_up` ~3.3s (only stage with thr≥21
    /// pure-cut headroom after measured rejects).
    ///
    /// mlxcel multi-token bits=8: two `UnifiedLinear::forward` +
    /// `compiled_geglu_approx_activation` (gemma4.rs ~917–920). Custom dual
    /// Metal (v1/v2) and dual-qmm compile already rejected on mbp-m5; this is
    /// the remaining host-FFI collapse of that same sequence.
    ///
    /// **Default: OFF** (opt-in pure A/B). Production stays portable dual qmm +
    /// Metal GEGLU.
    dual_qmm_geglu_enabled,
    "AX_MLX_DUAL_QMM_GEGLU"
);

env_flag!(
    /// `AX_MLX_COMPILED_GEGLU_ACTIVATION` — use mlxcel's process-static
    /// `mx::compile(shapeless=true)` GEGLU activation
    /// (`compiled_geglu_approx_activation` in mlx_cxx_bridge.cpp; gemma4.rs
    /// multi-token bits=8 FFN after dual qmm).
    ///
    /// **Default: OFF** (opt-in pure A/B). Production stays Metal GEGLU
    /// (`AX_MLX_GEGLU_MUL_METAL`) until pure wall under cache_eval proves a
    /// stable cut. When ON, takes precedence over Metal in `geglu()`.
    compiled_geglu_activation_enabled,
    "AX_MLX_COMPILED_GEGLU_ACTIVATION"
);

env_flag!(
    /// `AX_MLX_COMPILED_QGELU_AXQ_P128` — shape-compile the split affine
    /// GeGLU MLP (gate + up + gelu + down) for AXQ 4-bit (`group_size != 64`)
    /// contract p128. Community 4-bit gs=64 already uses mlxcel #680 shapeless
    /// compile; AXQ root 4/32 used to fall through to portable dual qmm.
    /// Shape-specific (not shapeless) so prefill qmm is not the decode kernel
    /// (#680 trap). p512 / p2048 stay portable.
    ///
    /// **Default: OFF**. Classified wash on `df-macbookpro-m5` during the
    /// Gemma 4 AXQ p128 1.10× unused-work track. The C++ shim
    /// `ax_mlx_compiled_gelu_approx_split_mlp` mirrors this predicate; keep
    /// both in lockstep. Set `=1` to force the experimental compile.
    compiled_qgelu_axq_p128_enabled,
    "AX_MLX_COMPILED_QGELU_AXQ_P128"
);

/// Whether AXQ 4-bit contract p128 should take the shape-compiled split MLP.
pub fn should_compiled_qgelu_axq_p128(group_size: i32, bits: i32, seq: i32) -> bool {
    should_compiled_qgelu_axq_p128_for(compiled_qgelu_axq_p128_enabled(), group_size, bits, seq)
}

/// Pure helper for [`should_compiled_qgelu_axq_p128`].
pub fn should_compiled_qgelu_axq_p128_for(
    enabled: bool,
    group_size: i32,
    bits: i32,
    seq: i32,
) -> bool {
    enabled && bits == 4 && group_size > 0 && group_size != 64 && seq == 128
}

env_flag!(
    /// `AX_MLX_ASYNC_DUAL_GATE_UP` — after multi-token dual gate/up qmm graphs
    /// are built (portable or shape-compiled), `async_eval([gate, up])` before
    /// GEGLU so both matmuls submit as one Metal command group.
    ///
    /// Profile residual: pure Gemma `post_attn_ffn_gate_up` ~3.26s. mlxcel
    /// multi-token bits=8 builds both `UnifiedLinear::forward` qmm then
    /// `compiled_geglu` (gemma4.rs ~917–920); MLX can schedule the pair when
    /// they share one eval frontier. AX otherwise may materialize gate then up
    /// serially through Metal GEGLU deps.
    ///
    /// **Default: OFF** (opt-in pure A/B under cache_eval).
    async_dual_gate_up_enabled,
    "AX_MLX_ASYNC_DUAL_GATE_UP"
);

env_flag!(
    /// `AX_MLX_GEMMA4_ASYNC_DUAL_GATE_UP_P128` — co-submit Gemma 4 split
    /// gate/up qmm with `async_eval` at contract p128 so MLX can schedule the
    /// pair as one Metal command group (mlxcel builds both `UnifiedLinear`
    /// then activation). Global `AX_MLX_ASYNC_DUAL_GATE_UP` stays OFF.
    /// Decode and short MTP verify (`seq < 128`) stay serial.
    ///
    /// **Default: OFF**. Remasured wash/dip on `df-macbookpro-m5` (2026-08-15)
    /// versus the skip-f32 + fused-p128 stack. Mid-graph `async_eval` is not
    /// part of the winning default. Keep opt-in only.
    gemma4_async_dual_gate_up_p128_enabled,
    "AX_MLX_GEMMA4_ASYNC_DUAL_GATE_UP_P128"
);

/// Whether Gemma 4 contract p128 should async-submit split gate/up.
pub fn should_gemma4_async_dual_gate_up_p128(model_family: &str, seq: i32) -> bool {
    should_gemma4_async_dual_gate_up_p128_for(
        gemma4_async_dual_gate_up_p128_enabled(),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_gemma4_async_dual_gate_up_p128`].
pub fn should_gemma4_async_dual_gate_up_p128_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
) -> bool {
    enabled
        && seq == 128
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "gemma4" | "gemma4_unified"
        )
}

env_flag!(
    /// `AX_MLX_GEMMA4_ASYNC_FIRST_KV_P128` — after the contract-p128 first KV
    /// write, `async_eval` the stored K/V so MLX can start that layer's first-KV
    /// / fused-attention graph while the host encodes the residual + FFN.
    ///
    /// **Default: OFF**. Remasured on `df-macbookpro-m5` (2026-08-15,
    /// `gemma4-axq-v7-asynckv` + repeat): 12B p128 616.45/624.15 vs fused-p128
    /// 651.57/659.48 (0.946× / 0.958×). The submit undoes the fused-p128 win.
    /// Decode unchanged. Keep opt-in only.
    gemma4_async_first_kv_p128_enabled,
    "AX_MLX_GEMMA4_ASYNC_FIRST_KV_P128"
);

/// Whether Gemma 4 contract p128 should async-submit the first KV write.
pub fn should_gemma4_async_first_kv_p128(model_family: &str, seq: i32) -> bool {
    should_gemma4_async_first_kv_p128_for(gemma4_async_first_kv_p128_enabled(), model_family, seq)
}

/// Pure helper for [`should_gemma4_async_first_kv_p128`].
pub fn should_gemma4_async_first_kv_p128_for(enabled: bool, model_family: &str, seq: i32) -> bool {
    enabled
        && seq == 128
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "gemma4" | "gemma4_unified"
        )
}

env_flag!(
    /// `AX_MLX_DUAL_AFFINE_QMM` — multi-token split gate/up as **one C++ call**
    /// returning `(gate, up)` without `mx::compile` and without GEGLU (Metal
    /// GEGLU stays on). Collapses two Rust→C++ qmm FFIs for pure gate_up
    /// residual (~3.26s). Unlike `AX_MLX_DUAL_QMM_GEGLU` (rejected 1.09×), this
    /// keeps production Metal GEGLU.
    ///
    /// mlxcel multi-token bits=8: two `UnifiedLinear::forward` (each one FFI) +
    /// activation (gemma4.rs ~917–920).
    ///
    /// **Default: OFF** (opt-in pure A/B under cache_eval). Pure 1.002× reject.
    dual_affine_qmm_enabled,
    "AX_MLX_DUAL_AFFINE_QMM"
);

env_flag!(
    /// `AX_MLX_DUAL_STREAM_GATE_UP` — issue multi-token gate/up affine qmm on
    /// two process-static GPU streams so independent matmuls can overlap on
    /// M5 Max. Uses the same C++ entry as dual_affine_qmm (Metal GEGLU kept).
    ///
    /// Profile residual: pure Gemma gate_up ~3.26s is two sequential large
    /// 8-bit qmms. Host-FFI dual alone was noise (1.002×); dual-stream targets
    /// GPU concurrency. mlxcel still uses sequential UnifiedLinear; this is an
    /// AX M5 Max experiment on the same residual.
    ///
    /// **Default: OFF** (opt-in pure A/B under cache_eval).
    dual_stream_gate_up_enabled,
    "AX_MLX_DUAL_STREAM_GATE_UP"
);

env_flag!(
    /// `AX_MLX_GEMMA4_DUAL_STREAM_GATE_UP_P128` — issue Gemma 4 split gate/up
    /// qmm on two process-static GPU streams at contract p128 so M5 Max can
    /// overlap the pair (mlxcel still runs two sequential UnifiedLinear).
    /// Compiled AXQ split-MLP stays off on this shape so the streams engage.
    ///
    /// **Default: OFF**. Remasured on `df-macbookpro-m5` (2026-08-15,
    /// `gemma4-axq-v7-dualstream`): 12B p128 567.19/614.92=0.922× and 31B
    /// 310.80/327.53=0.949×. Dual streams plus skipping compiled split-MLP
    /// regresses the fused-p128 stack. Keep opt-in only.
    gemma4_dual_stream_gate_up_p128_enabled,
    "AX_MLX_GEMMA4_DUAL_STREAM_GATE_UP_P128"
);

/// Whether Gemma 4 contract p128 should dual-stream split gate/up.
pub fn should_gemma4_dual_stream_gate_up_p128(model_family: &str, seq: i32) -> bool {
    should_gemma4_dual_stream_gate_up_p128_for(
        gemma4_dual_stream_gate_up_p128_enabled(),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_gemma4_dual_stream_gate_up_p128`].
pub fn should_gemma4_dual_stream_gate_up_p128_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
) -> bool {
    enabled
        && seq == 128
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "gemma4" | "gemma4_unified"
        )
}

env_flag!(
    /// `AX_MLX_CACHE_ONLY_CHUNK_EVAL` — materialise KV after **every** cache-only
    /// prefill chunk (not only the last). Pure greedy long prompts use the
    /// mlx-lm-style cache-only prefix (n−1 tokens) with deferred eval; for
    /// pure Gemma 13.8k that is ~27 lazy chunks in one graph.
    ///
    /// mlxcel residual (`generate.rs` `chunked_prefill_last_logits` ~293–323,
    /// issue #672): force `eval` between prefill chunks so the lazy graph and
    /// freelist never span the whole prompt. AX already does this on the
    /// non-cache-only loop (`eval_with_kv_refs` per chunk) but not on the
    /// cache-only pure path.
    ///
    /// **Default: OFF** (opt-in pure A/B). Short prompts (≤2–3 chunks) keep
    /// deferred final-barrier behaviour unless the flag is set.
    cache_only_chunk_eval_enabled,
    "AX_MLX_CACHE_ONLY_CHUNK_EVAL"
);

env_flag!(
    /// `AX_MLX_CACHE_ONLY_CHUNK_ASYNC_EVAL` — under cache-only chunk eval,
    /// submit intermediate chunk KV with `async_eval` instead of blocking
    /// `eval`, so the host can build chunk N+1 while GPU runs chunk N.
    /// The final cache-only chunk still uses a blocking barrier so decode
    /// sees fully materialised KV.
    ///
    /// Path A residual (mbp-m5 pure thr): multi-process keep_base already uses
    /// `CACHE_ONLY_CHUNK_EVAL=1` (~27 blocking barriers on Gemma 13.8k). Those
    /// barriers serialise host graph build behind GPU completion. Async
    /// intermediate barriers target host/GPU overlap without collapsing the
    /// #672 "no giant lazy graph" property (each chunk is still submitted).
    ///
    /// No-op unless `AX_MLX_CACHE_ONLY_CHUNK_EVAL=1`. **Default: OFF**.
    cache_only_chunk_async_eval_enabled,
    "AX_MLX_CACHE_ONLY_CHUNK_ASYNC_EVAL"
);

/// Whether a cache-only prefill chunk should use non-blocking KV submit.
///
/// Intermediate chunks under both `CACHE_ONLY_CHUNK_EVAL` and
/// `CACHE_ONLY_CHUNK_ASYNC_EVAL` async-submit; the final cache-only chunk
/// always blocks so the subsequent decode step sees settled KV.
pub fn cache_only_chunk_should_async_eval(is_final_cache_only_chunk: bool) -> bool {
    cache_only_chunk_should_async_eval_for(
        cache_only_chunk_eval_enabled(),
        cache_only_chunk_async_eval_enabled(),
        is_final_cache_only_chunk,
    )
}

/// Pure helper for [`cache_only_chunk_should_async_eval`] (unit-testable).
pub fn cache_only_chunk_should_async_eval_for(
    chunk_eval_enabled: bool,
    async_eval_enabled: bool,
    is_final_cache_only_chunk: bool,
) -> bool {
    chunk_eval_enabled && async_eval_enabled && !is_final_cache_only_chunk
}

/// mlxcel `MLXCEL_PIPELINE_GRANULARITY` parity — layer-boundary `async_eval`
/// hints during multi-layer prefill.
///
/// mlxcel residual (`mlxcel_core::utils::pipeline_hint`, models/gemma4.rs layer
/// loop): after each transformer layer, optionally `async_eval(hidden)` so MLX
/// can start executing layer N while host builds layer N+1 / weight traffic
/// overlaps. Documented for M5 (NA + GPU shader cores). Default `off` preserves
/// full-graph fusion (same as mlxcel).
///
/// Values of `AX_MLX_PIPELINE_GRANULARITY`:
/// - unset / `off` / empty → no intermediate eval
/// - `layer` → hint after every non-final layer
/// - `block:N` → hint every N layers (N≥1; invalid N falls back to 4)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineGranularity {
    Off,
    PerLayer,
    PerBlock(usize),
}

/// Parse `AX_MLX_PIPELINE_GRANULARITY` without caching (tests / diagnostics).
pub fn parse_pipeline_granularity(raw: &str) -> PipelineGranularity {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("off") {
        return PipelineGranularity::Off;
    }
    if trimmed.eq_ignore_ascii_case("layer") {
        return PipelineGranularity::PerLayer;
    }
    if let Some(rest) = trimmed
        .strip_prefix("block:")
        .or_else(|| trimmed.strip_prefix("BLOCK:"))
        .or_else(|| trimmed.strip_prefix("Block:"))
    {
        let n = rest.trim().parse::<usize>().unwrap_or(4).max(1);
        return PipelineGranularity::PerBlock(n);
    }
    // Unknown → fail-closed to off (preserve fusion).
    PipelineGranularity::Off
}

/// Process-cached pipeline granularity. Default OFF.
pub fn pipeline_granularity() -> PipelineGranularity {
    static CACHED: OnceLock<PipelineGranularity> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("AX_MLX_PIPELINE_GRANULARITY") {
        Ok(raw) => parse_pipeline_granularity(&raw),
        Err(_) => PipelineGranularity::Off,
    })
}

/// Qwen prefill fires an `async_eval` hint every this many layers when
/// [`should_qwen_prefill_pipeline_block`] is on. mlxcel `block:N` analog.
pub const QWEN_PREFILL_PIPELINE_BLOCK: usize = 8;

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_PIPELINE_BLOCK` — after every 8 non-final Qwen
    /// 3.5/3.6 prefill layers with `seq >= 1024`, `async_eval(hidden)` so
    /// GPU can start layer N while the host builds N+1.
    ///
    /// **Default: OFF**. Community p2048 904.335/858=1.054004 (0.995× standing
    /// 908.5, 2026-08-13). AXQ p2048 888.809/862.825=1.030115 (0.998× q2only).
    /// Same class as intermediate-chunk async_eval. Not FFN compile.
    qwen_prefill_pipeline_block_enabled,
    "AX_MLX_QWEN_PREFILL_PIPELINE_BLOCK"
);

/// Whether Qwen generate prefill should submit a layer-block pipeline hint.
pub fn should_qwen_prefill_pipeline_block(
    model_family: &str,
    seq: usize,
    layer_idx: usize,
    total_layers: usize,
) -> bool {
    should_qwen_prefill_pipeline_block_for(
        qwen_prefill_pipeline_block_enabled(),
        model_family,
        seq,
        layer_idx,
        total_layers,
        QWEN_PREFILL_PIPELINE_BLOCK,
    )
}

/// Pure helper for [`should_qwen_prefill_pipeline_block`].
pub fn should_qwen_prefill_pipeline_block_for(
    enabled: bool,
    model_family: &str,
    seq: usize,
    layer_idx: usize,
    total_layers: usize,
    block: usize,
) -> bool {
    enabled
        && seq >= 1024
        && block >= 1
        && total_layers > 0
        && layer_idx + 1 < total_layers
        && (layer_idx + 1).is_multiple_of(block)
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

/// Whether a layer-boundary pipeline hint should fire after `layer_idx`
/// (0-based) of `total_layers`. Never fires after the final layer.
pub fn pipeline_hint_should_fire(layer_idx: usize, total_layers: usize) -> bool {
    if total_layers == 0 || layer_idx + 1 >= total_layers {
        return false;
    }
    match pipeline_granularity() {
        PipelineGranularity::Off => false,
        PipelineGranularity::PerLayer => true,
        PipelineGranularity::PerBlock(n) => (layer_idx + 1).is_multiple_of(n),
    }
}

/// Blocking layer-boundary evaluation for prefill fairness diagnostics.
///
/// Unlike [`pipeline_granularity`], which only submits an `async_eval` hint,
/// `AX_MLX_PIPELINE_EVAL_GRANULARITY` inserts a completion barrier. This is an
/// opt-in physical probe for measuring whether shorter GPU command bursts
/// improve cross-process fairness; production defaults to `off`.
///
/// Accepted values are `off`, `layer`, `block:N` (`N >= 1`), `sublayer`, and
/// `yield:N` (`N >= 1` milliseconds). `sublayer` keeps the per-layer barriers
/// and adds a Gemma4 text-prefill barrier between attention output projection
/// and the post-attention FFN. `yield:N` is the dual-stream residual: fire a
/// barrier at multi-token layer boundaries only when ≥ `N` ms of wall time
/// have elapsed since the previous fire, capping GPU monopolization for a
/// sibling process without forcing a barrier on every layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineEvalGranularity {
    Off,
    PerLayer,
    PerBlock(usize),
    Sublayer,
    /// Wall-clock dual-stream yield: fire when ≥ N milliseconds elapsed.
    YieldMs(u64),
}

/// Parse `AX_MLX_PIPELINE_EVAL_GRANULARITY` without caching.
///
/// Malformed values fail closed to [`PipelineEvalGranularity::Off`] so a typo
/// cannot introduce blocking barriers into the normal prefill path.
pub fn parse_pipeline_eval_granularity(raw: &str) -> PipelineEvalGranularity {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("off") {
        return PipelineEvalGranularity::Off;
    }
    if trimmed.eq_ignore_ascii_case("layer") {
        return PipelineEvalGranularity::PerLayer;
    }
    if trimmed.eq_ignore_ascii_case("sublayer") {
        return PipelineEvalGranularity::Sublayer;
    }
    if let Some((prefix, value)) = trimmed.split_once(':') {
        let value = value.trim();
        if prefix.eq_ignore_ascii_case("block")
            && let Ok(n) = value.parse::<usize>()
            && n > 0
        {
            return PipelineEvalGranularity::PerBlock(n);
        }
        if prefix.eq_ignore_ascii_case("yield")
            && let Ok(n) = value.parse::<u64>()
            && n > 0
        {
            return PipelineEvalGranularity::YieldMs(n);
        }
    }
    PipelineEvalGranularity::Off
}

/// Process-cached blocking prefill-eval granularity. Default OFF.
pub fn pipeline_eval_granularity() -> PipelineEvalGranularity {
    static CACHED: OnceLock<PipelineEvalGranularity> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("AX_MLX_PIPELINE_EVAL_GRANULARITY") {
        Ok(raw) => parse_pipeline_eval_granularity(&raw),
        Err(_) => PipelineEvalGranularity::Off,
    })
}

/// Pure wall-clock yield predicate for dual-stream prefill barriers.
///
/// - Decode (`seq_len <= 1`) and the final layer never fire.
/// - First eligible boundary fires when `last_fire_ns` is `None`.
/// - Subsequent fires require `now_ns - last >= yield_ms * 1_000_000`.
///
/// Callers that fire must advance `last_fire_ns` to `now_ns` (see
/// [`pipeline_eval_should_fire`]).
pub fn pipeline_eval_yield_should_fire(
    last_fire_ns: Option<u64>,
    now_ns: u64,
    yield_ms: u64,
    seq_len: usize,
    layer_idx: usize,
    total_layers: usize,
) -> bool {
    if yield_ms == 0 || seq_len <= 1 || total_layers == 0 || layer_idx + 1 >= total_layers {
        return false;
    }
    let Some(last) = last_fire_ns else {
        return true;
    };
    let elapsed_ns = now_ns.saturating_sub(last);
    elapsed_ns >= yield_ms.saturating_mul(1_000_000)
}

fn pipeline_eval_should_fire_for(
    granularity: PipelineEvalGranularity,
    seq_len: usize,
    layer_idx: usize,
    total_layers: usize,
) -> bool {
    if seq_len <= 1 || total_layers == 0 || layer_idx + 1 >= total_layers {
        return false;
    }
    match granularity {
        PipelineEvalGranularity::Off => false,
        PipelineEvalGranularity::PerLayer | PipelineEvalGranularity::Sublayer => true,
        PipelineEvalGranularity::PerBlock(n) => (layer_idx + 1).is_multiple_of(n),
        // YieldMs needs wall-clock state; use the process path in
        // [`pipeline_eval_should_fire`]. Pure layer filters still apply.
        PipelineEvalGranularity::YieldMs(_) => false,
    }
}

/// Parse `AX_MLX_PIPELINE_EVAL_TAIL_LAYERS` without caching.
///
/// Dual-stream concurrent residual: force per-layer blocking eval on the last
/// `N` multi-token layers (final layer still exempt) so early prefill can use a
/// thr-oriented base granularity (`block:8`) while the tail yields to a sibling
/// decode process for stream-gap fairness. Default **0** (off). Malformed or
/// empty values fail closed to **0**.
pub fn parse_pipeline_eval_tail_layers(raw: &str) -> usize {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("off") {
        return 0;
    }
    trimmed.parse::<usize>().unwrap_or_default()
}

/// Process-cached tail-layer count. Default 0 (overlay off).
pub fn pipeline_eval_tail_layers() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("AX_MLX_PIPELINE_EVAL_TAIL_LAYERS") {
        Ok(raw) => parse_pipeline_eval_tail_layers(&raw),
        Err(_) => 0,
    })
}

/// Pure predicate: is `layer_idx` among the last `tail_n` multi-token layers
/// before the final layer?
///
/// Eligible layers are `0..total_layers-2` (final always exempt). Tail of size
/// `N` is `[total-1-N, total-2]` clamped to zero. Used by the dual-stream
/// concurrent residual so thr stacks can monopolize early layers then yield.
pub fn pipeline_eval_layer_in_tail(layer_idx: usize, total_layers: usize, tail_n: usize) -> bool {
    if tail_n == 0 || total_layers < 2 || layer_idx + 1 >= total_layers {
        return false;
    }
    let first_tail = total_layers.saturating_sub(1).saturating_sub(tail_n);
    layer_idx >= first_tail
}

/// Whether the diagnostic blocking barrier should fire after this layer.
///
/// Decode (`seq_len == 1`) and the final transformer layer are always exempt.
/// `yield:N` consults a process-wide last-fire timestamp (atomic) so multi-
/// process concurrent thr stacks can cap GPU monopolization in wall time.
/// When `AX_MLX_PIPELINE_EVAL_TAIL_LAYERS=N` is set, the last `N` multi-token
/// layers force a layer-eval barrier regardless of the base granularity.
pub fn pipeline_eval_should_fire(seq_len: usize, layer_idx: usize, total_layers: usize) -> bool {
    if seq_len <= 1 || total_layers == 0 || layer_idx + 1 >= total_layers {
        return false;
    }
    let tail_n = pipeline_eval_tail_layers();
    if pipeline_eval_layer_in_tail(layer_idx, total_layers, tail_n) {
        return true;
    }
    match pipeline_eval_granularity() {
        PipelineEvalGranularity::YieldMs(yield_ms) => {
            use std::sync::atomic::{AtomicU64, Ordering};
            use std::time::{SystemTime, UNIX_EPOCH};
            static LAST_FIRE_NS: AtomicU64 = AtomicU64::new(0);
            let now_ns = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            let last_raw = LAST_FIRE_NS.load(Ordering::Relaxed);
            let last = if last_raw == 0 { None } else { Some(last_raw) };
            if pipeline_eval_yield_should_fire(
                last,
                now_ns,
                yield_ms,
                seq_len,
                layer_idx,
                total_layers,
            ) {
                LAST_FIRE_NS.store(now_ns, Ordering::Relaxed);
                true
            } else {
                false
            }
        }
        other => pipeline_eval_should_fire_for(other, seq_len, layer_idx, total_layers),
    }
}

fn pipeline_sublayer_eval_should_fire_for(
    granularity: PipelineEvalGranularity,
    seq_len: usize,
    model_family: &str,
) -> bool {
    matches!(granularity, PipelineEvalGranularity::Sublayer)
        && seq_len > 1
        && model_family == "gemma4"
}

/// Whether to block after standard Gemma4 text attention output projection.
///
/// This exact family gate deliberately excludes decode, Gemma VL/unified,
/// assistant, diffusion, and every non-Gemma target from the diagnostic probe.
pub fn pipeline_sublayer_eval_should_fire(seq_len: usize, model_family: &str) -> bool {
    pipeline_sublayer_eval_should_fire_for(pipeline_eval_granularity(), seq_len, model_family)
}

env_flag!(
    /// `AX_MLX_NATIVE_OFFSET_CAUSAL` — for full-attention multi-token prefill
    /// with KV cache offset (`key_len > seq`), skip materializing an
    /// O(seq×key_len) bool causal array and use MLX `mask="causal"` instead.
    ///
    /// MLX steel SDPA sets `qL_off = key_len - query_len` so query i attends
    /// keys j ≤ offset+i — same rule as `create_causal_mask(seq, offset, None)`.
    /// mlxcel `causal_attention(window_size=0)` always uses
    /// `fast_scaled_dot_product_attention_causal` (native causal), never an
    /// array mask for full-window layers.
    ///
    /// **Default: OFF**. Gemma 13.8k A/B rejected (1.064×). Qwen 3.6 27B
    /// p2048 remasured 889.3 vs 891.0 (2026-08-13). Sliding-window layers
    /// still need explicit masks when the window constraint is active.
    native_offset_causal_enabled,
    "AX_MLX_NATIVE_OFFSET_CAUSAL"
);

env_flag!(
    /// `AX_MLX_DENSE_GEGLU_DOWN_FUSE` — multi-token split GEGLU product fused
    /// into the dense FFN down_proj quantized matmul (one C++ graph-build for
    /// gelu_approx(gate)*up → down qmm). Targets pure prefill residual after
    /// dual gate_up qmm (activation + down ~2.5s profiled).
    ///
    /// **Default: OFF** (opt-in A/B). mlxcel keeps activation then down separate
    /// for bits=8 multi-token (compiled_geglu then down_proj.forward).
    dense_geglu_down_fuse_enabled,
    "AX_MLX_DENSE_GEGLU_DOWN_FUSE"
);

env_flag!(
    /// `AX_MLX_QWEN_SWIGLU_DOWN_FUSE` — multi-token split SwiGLU product fused
    /// into the dense FFN down_proj quantized matmul (one C++ graph for
    /// `silu(gate)*up → down qmm`).
    ///
    /// **Default: OFF**. AXQ remasured p2048 876.21 vs 891.02 (2026-08-13).
    /// Gemma GEGLU fuse stays on `AX_MLX_DENSE_GEGLU_DOWN_FUSE` (also OFF).
    qwen_swiglu_down_fuse_enabled,
    "AX_MLX_QWEN_SWIGLU_DOWN_FUSE"
);

env_flag!(
    /// `AX_MLX_QWEN_DUAL_QMM_SWIGLU` — multi-token split gate/up as one C++
    /// call: `silu(qmm(x,gate)) * qmm(x,up)`. No `mx::compile`, no down fuse,
    /// no dual-stream. Targets p2048 `gate_up` 837ms + activation 54ms.
    ///
    /// **Default: OFF**. AXQ remasured p2048 875.21 vs 891.02 (2026-08-13).
    /// Gemma stays on `AX_MLX_DUAL_QMM_GEGLU` (also default OFF).
    qwen_dual_qmm_swiglu_enabled,
    "AX_MLX_QWEN_DUAL_QMM_SWIGLU"
);

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_DUAL_QMM_SWIGLU_METAL` — multi-token 4-bit
    /// gate/up qmm + SwiGLU in one Metal kernel using `simdgroup_matrix`
    /// 8×8 MMA.
    ///
    /// **Default: OFF**. Community remasured p2048 prefill ~179 vs 908
    /// (~0.20×, 2026-08-13). Same class as Gemma dual Metal 8.5× reject.
    /// Host-FFI `dual_qmm_swiglu` also stays OFF (875 vs 891).
    qwen_prefill_dual_qmm_swiglu_metal_enabled,
    "AX_MLX_QWEN_PREFILL_DUAL_QMM_SWIGLU_METAL"
);

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_FLAT_DOWN_QMM` — reshape Qwen split-prefill down
    /// activations `[B,S,I] → [B*S,I]` before the affine qmm, then restore
    /// `[B,S,H]`. Standalone down GEMM (no silu+down fuse).
    ///
    /// **Default: OFF**. AXQ remasured p2048 888.33 vs 891.02 (2026-08-13).
    /// Gemma stays on the 3-D qw path.
    qwen_prefill_flat_down_qmm_enabled,
    "AX_MLX_QWEN_PREFILL_FLAT_DOWN_QMM"
);

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_FLAT_FFN` — reshape Qwen split-prefill FFN
    /// activations `[B,S,H] → [B*S,H]` before gate/up/down qmm, then
    /// restore `[B,S,H]`. All three affine qmms see a 2-D leading dim.
    ///
    /// **Default: OFF**. Four-lane remasure (binary `77435b62…`, 2026-08-13):
    /// AXQ p2048 889.673/862.825=1.031116 (0.9985× q2only 891). Community
    /// p2048 909.796/858.000=1.060369 (3d FAIL). Same class as standalone
    /// flat-down 888. Standalone down flatten stays OFF.
    qwen_prefill_flat_ffn_enabled,
    "AX_MLX_QWEN_PREFILL_FLAT_FFN"
);

/// Whether Qwen prefill FFN should flatten `[B,S,H] → [B*S,H]`.
pub fn should_qwen_prefill_flat_ffn(model_family: &str, seq: i32, rank: usize) -> bool {
    should_qwen_prefill_flat_ffn_for(qwen_prefill_flat_ffn_enabled(), model_family, seq, rank)
}

/// Pure helper for [`should_qwen_prefill_flat_ffn`].
pub fn should_qwen_prefill_flat_ffn_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
    rank: usize,
) -> bool {
    enabled && seq > 1 && rank == 3 && model_family.eq_ignore_ascii_case("qwen3_5")
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_CONTIGUOUS_FFN` — materialize a contiguous
    /// `[B,S,H]` before Qwen split-prefill gate/up/down qmm.
    ///
    /// **Default: OFF**. Four-lane remasure (binary `f72c4606…`, 2026-08-13):
    /// AXQ p2048 891.078/862.825=1.032745 (1.0001× q2only 891). Community
    /// p2048 908.312/858.000=1.058639 (3d FAIL). Flat-FFN stays OFF.
    qwen_prefill_contiguous_ffn_enabled,
    "AX_MLX_QWEN_PREFILL_CONTIGUOUS_FFN"
);

/// Whether Qwen prefill FFN should `contiguous` the activation.
pub fn should_qwen_prefill_contiguous_ffn(model_family: &str, seq: i32, rank: usize) -> bool {
    should_qwen_prefill_contiguous_ffn_for(
        qwen_prefill_contiguous_ffn_enabled(),
        model_family,
        seq,
        rank,
    )
}

/// Pure helper for [`should_qwen_prefill_contiguous_ffn`].
pub fn should_qwen_prefill_contiguous_ffn_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
    rank: usize,
) -> bool {
    enabled && seq > 1 && rank == 3 && model_family.eq_ignore_ascii_case("qwen3_5")
}

env_flag!(
    /// `AX_MLX_QWEN_LA_OUT_PROJ_SILU_MUL_QMM` — Qwen linear-attention
    /// prefill output: `rms_norm` then `silu(z) * normed` fused into the
    /// `out_proj` quantized matmul.
    ///
    /// **Default: OFF**. Four-lane remasure (binary `2b846b04…`, 2026-08-13):
    /// AXQ p2048 890.888/862.825=1.032524 (0.9999× q2only 891). Community
    /// p2048 909.631/858.000=1.060177 (3d FAIL). SwiGLU→down stays OFF.
    qwen_la_out_proj_silu_mul_qmm_enabled,
    "AX_MLX_QWEN_LA_OUT_PROJ_SILU_MUL_QMM"
);

/// Whether Qwen linear-attn prefill should fuse gated RMS into out_proj qmm.
pub fn should_qwen_la_out_proj_silu_mul_qmm(model_family: &str, seq: i32) -> bool {
    should_qwen_la_out_proj_silu_mul_qmm_for(
        qwen_la_out_proj_silu_mul_qmm_enabled()
            || (mtp_la_out_proj_silu_mul_qmm_enabled()
                && qwen_linear_mtp_relaxed_session_enabled()),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_qwen_la_out_proj_silu_mul_qmm`].
pub fn should_qwen_la_out_proj_silu_mul_qmm_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
) -> bool {
    enabled && seq > 1 && model_family.eq_ignore_ascii_case("qwen3_5")
}

env_flag!(
    /// `AX_MLX_DIRECT_CPP_QK_NORM_ROPE` — opt-in direct C++ probe route for
    /// standard attention Q/K `as_strided -> rms_norm -> rope`.
    ///
    /// **Default: OFF**. This is intentionally not default-on: the microbench
    /// candidate reduced Rust op count, but production decode still needs a
    /// same-commit Gemma 4 E2B A/B before promotion. The route only engages
    /// when Q/K norm exists and the flat QK-norm diagnostic fallback is not
    /// active.
    direct_cpp_qk_norm_rope_enabled,
    "AX_MLX_DIRECT_CPP_QK_NORM_ROPE"
);

env_flag_default_on!(
    /// `AX_MLX_QWEN_DIRECT_CPP_QK_NORM_ROPE` — default Qwen-family direct C++
    /// route for standard-attention Q/K `as_strided -> rms_norm -> rope`.
    ///
    /// **Default: ON for Qwen3.5/Qwen3Next only** (kill-switch via
    /// `AX_MLX_QWEN_DIRECT_CPP_QK_NORM_ROPE=0`). This keeps the older global
    /// probe opt-in for non-Qwen families while reducing the full-attention
    /// decode op count on hybrid Qwen linear-attention models.
    qwen_direct_cpp_qk_norm_rope_enabled,
    "AX_MLX_QWEN_DIRECT_CPP_QK_NORM_ROPE"
);

env_flag!(
    /// `AX_MLX_QWEN_COMPILED_QK_NORM_ROPE` — wrap the Qwen base-RoPE
    /// `as_strided → rms_norm → rope(base)` C++ path in `mx::compile`.
    ///
    /// **Default: OFF**. Four-lane remasure (binary `41fd8313…`, 2026-08-13):
    /// AXQ p2048 890.684/862.825=1.032288 (0.9996× q2only 891). Community
    /// p2048 908.406/858.000=1.058749 (3d FAIL). Freqs compile stays OFF.
    qwen_compiled_qk_norm_rope_enabled,
    "AX_MLX_QWEN_COMPILED_QK_NORM_ROPE"
);

env_flag!(
    /// `AX_MLX_QWEN_GATED_DELTA_PREFILL_CONTIGUOUS` — `contiguous` Q/K/V/A/B
    /// before the GatedDelta prefill TG kernel.
    ///
    /// **Default: OFF**. Four-lane remasure (binary `6e56e7ed…`, 2026-08-13):
    /// AXQ p2048 889.558/862.825=1.030983 (0.9984× q2only 891). Community
    /// p2048 908.438/858.000=1.058787 (3d FAIL). Tile-512/streaming stay OFF.
    qwen_gated_delta_prefill_contiguous_enabled,
    "AX_MLX_QWEN_GATED_DELTA_PREFILL_CONTIGUOUS"
);

env_flag!(
    /// `AX_MLX_QWEN_LA_FUSED_QKVZ_BA_QMM` — one concatenated affine qmm for
    /// matching-bit packed QKVZ+BA on Qwen multi-token prefill.
    ///
    /// **Default: OFF**. Per-forward concat (binary `37125559…`) p2048
    /// 887.779/862.825=1.028921. Load-time concat (binary `1fa58239…`)
    /// community p2048 907.712/858=1.057940 (0.999× standing 908.5). Wash.
    qwen_la_fused_qkvz_ba_qmm_enabled,
    "AX_MLX_QWEN_LA_FUSED_QKVZ_BA_QMM"
);

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_DOWN_COMPILE` — shape-compile only the Qwen split
    /// prefill **down** qmm.
    ///
    /// **Default: OFF**. Four-lane remasure (binary `99f65ba3…`, 2026-08-13):
    /// community p2048 904.141/858=1.053779 (0.995× standing 908.5). 3a PASS.
    /// Same class as full split FFN compile 888.77. Keep imperative down qw.
    qwen_prefill_down_compile_enabled,
    "AX_MLX_QWEN_PREFILL_DOWN_COMPILE"
);

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_CHUNK_1536` — raise the linear-attention runner
    /// chunk cap to 1536 so p2048 is 1536+512.
    ///
    /// **Default: OFF**. Interim community p2048 ~898.7 vs standing 908.5
    /// (2026-08-13). Same class as single-2048 889–890. Keep two 1024s.
    qwen_prefill_chunk_1536_enabled,
    "AX_MLX_QWEN_PREFILL_CHUNK_1536"
);

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_CHUNK_1280` — raise the linear-attention runner
    /// chunk cap to 1280 so p2048 is 1280+768. FFN qmm runs at M=1280 then
    /// M=768. GatedDelta already tiles at 1024 when seq>1024, so this is not
    /// the closed 1536 (1536+512) or single-2048 path. Not FFN-host-fusion
    /// or dequant-dense.
    ///
    /// **Default: OFF**. Remasured binary `8c08e31b…` (2026-08-14): 3b
    /// 1.024008 / 3d 1.048763 wash (0.992× q2only). M=1280+768 does not
    /// beat two 1024s.
    qwen_prefill_chunk_1280_enabled,
    "AX_MLX_QWEN_PREFILL_CHUNK_1280"
);

env_flag!(
    /// `AX_MLX_QWEN_COMPILED_GATED_DELTA_PREFILL` — wrap the GatedDelta
    /// prefill TG oneshot (`qwen35_gated_delta_v3`) in `mx::compile`.
    ///
    /// **Default: OFF**. Community p2048 905.390/858=1.055235 (0.997× standing
    /// 908.5, 2026-08-13). Same class as compiled QK-RoPE / split FFN compile.
    qwen_compiled_gated_delta_prefill_enabled,
    "AX_MLX_QWEN_COMPILED_GATED_DELTA_PREFILL"
);

/// Whether GatedDelta prefill should use the compiled TG oneshot.
pub fn should_qwen_compiled_gated_delta_prefill(seq: i32) -> bool {
    should_qwen_compiled_gated_delta_prefill_for(qwen_compiled_gated_delta_prefill_enabled(), seq)
}

/// Pure helper for [`should_qwen_compiled_gated_delta_prefill`].
pub fn should_qwen_compiled_gated_delta_prefill_for(enabled: bool, seq: i32) -> bool {
    enabled && seq > 1
}

/// Minimum leading elements before Qwen packed FFN prefill compile engages.
/// 512-token packed compile was slower; p2048 is two 1024 chunks (unmeasured).
pub const QWEN_PACKED_FFN_PREFILL_COMPILE_MIN_LEADING: i64 = 1024;

env_flag!(
    /// `AX_MLX_QWEN_PACKED_FFN_PREFILL_COMPILE` — let the dense packed FFN
    /// prefill compile (`AX_MLX_DENSE_FFN_COMPILE_PREFILL`) engage on Qwen
    /// when `leading >= 1024`.
    ///
    /// **Default: OFF**. Community p2048 904.620/858=1.054337 (0.996× standing
    /// 908.5, 2026-08-13). Community 4-bit has no packed gate/up, so this
    /// cannot move 3d. 512-token packed compile stays closed. Same class as
    /// split FFN compile.
    qwen_packed_ffn_prefill_compile_enabled,
    "AX_MLX_QWEN_PACKED_FFN_PREFILL_COMPILE"
);

/// Whether Qwen packed FFN prefill should use the fixed-shape compile path.
pub fn should_qwen_packed_ffn_prefill_compile(model_family: &str, leading: i64) -> bool {
    should_qwen_packed_ffn_prefill_compile_for(
        qwen_packed_ffn_prefill_compile_enabled(),
        model_family,
        leading,
    )
}

/// Pure helper for [`should_qwen_packed_ffn_prefill_compile`].
pub fn should_qwen_packed_ffn_prefill_compile_for(
    enabled: bool,
    model_family: &str,
    leading: i64,
) -> bool {
    enabled
        && model_family.to_ascii_lowercase().starts_with("qwen")
        && leading >= QWEN_PACKED_FFN_PREFILL_COMPILE_MIN_LEADING
}

/// Whether Qwen split prefill should compile the standalone down qmm.
pub fn should_qwen_prefill_down_compile(seq: i32, leading: i64) -> bool {
    should_qwen_prefill_down_compile_for(qwen_prefill_down_compile_enabled(), seq, leading)
}

/// Pure helper for [`should_qwen_prefill_down_compile`].
pub fn should_qwen_prefill_down_compile_for(enabled: bool, seq: i32, leading: i64) -> bool {
    enabled && seq > 1 && leading >= QWEN_SPLIT_FFN_PREFILL_COMPILE_MIN_LEADING
}

/// Minimum sequence length before packed LA input compile engages.
/// 512-token packed FFN compile was slower; p2048 is two 1024 chunks.
pub const QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ: i32 = 1024;

env_flag!(
    /// `AX_MLX_QWEN_PACKED_LA_INPUTS_COMPILE` — compile packed QKVZ/BA
    /// projection (two affine qmm + reshape/slice/concat) when `seq >= 1024`.
    ///
    /// **Default: OFF**. Remasured binary `6b6b2e06…` (2026-08-13): community
    /// p2048 904.726/858=1.054460; AXQ p2048 888.959/862.825=1.030289
    /// (0.998× q2only). Wash. Not FFN/GatedDelta compile, not fused QKVZ+BA.
    qwen_packed_la_inputs_compile_enabled,
    "AX_MLX_QWEN_PACKED_LA_INPUTS_COMPILE"
);

/// Whether packed LA inputs should use the fixed-shape compile path.
pub fn should_qwen_packed_la_inputs_compile(seq: i32) -> bool {
    should_qwen_packed_la_inputs_compile_for(qwen_packed_la_inputs_compile_enabled(), seq)
}

/// Pure helper for [`should_qwen_packed_la_inputs_compile`].
pub fn should_qwen_packed_la_inputs_compile_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_LA_POST_INPUT_COMPILE` — compile the existing C++
    /// post-input block (conv1d + SiLU + split + qk RMSNorm + scale) when
    /// `seq >= 1024`.
    ///
    /// **Default: OFF**. Remasured binary `e535cf3e…` (2026-08-13): community
    /// p2048 911.056/858=1.061838; AXQ p2048 894.749/862.825=1.036999
    /// (1.004× q2only). Wash. Not packed-LA-inputs compile, not GatedDelta
    /// compile, not prefill post-input Metal.
    qwen_la_post_input_compile_enabled,
    "AX_MLX_QWEN_LA_POST_INPUT_COMPILE"
);

/// Whether LA post-input should use the fixed-shape compile path.
pub fn should_qwen_la_post_input_compile(seq: i32) -> bool {
    should_qwen_la_post_input_compile_for(qwen_la_post_input_compile_enabled(), seq)
}

/// Pure helper for [`should_qwen_la_post_input_compile`].
pub fn should_qwen_la_post_input_compile_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_LA_DUAL_STREAM_QKVZ_BA` — issue packed QKVZ and BA affine
    /// qmm on two GPU streams so M5 Max can overlap the two independent
    /// projections at `seq >= 1024`.
    ///
    /// **Default: OFF**. Remasured binary `f1d47194…` (2026-08-13): community
    /// p2048 894.153/858=1.042137 (0.984× standing); AXQ p2048
    /// 879.421/862.825=1.019234 (0.987× q2only). Regression. Same class as
    /// closed FFN dual-stream.
    qwen_la_dual_stream_qkvz_ba_enabled,
    "AX_MLX_QWEN_LA_DUAL_STREAM_QKVZ_BA"
);

/// Whether packed LA QKVZ/BA should issue on two GPU streams.
pub fn should_qwen_la_dual_stream_qkvz_ba(seq: i32) -> bool {
    should_qwen_la_dual_stream_qkvz_ba_for(qwen_la_dual_stream_qkvz_ba_enabled(), seq)
}

/// Pure helper for [`should_qwen_la_dual_stream_qkvz_ba`].
pub fn should_qwen_la_dual_stream_qkvz_ba_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_LA_FLAT_INPUTS` — reshape packed QKVZ/BA activations
    /// `[B,S,H]→[B*S,H]` before the two affine qmm at `seq >= 1024`.
    ///
    /// **Default: OFF**. Remasured binary `07de1419…` (2026-08-14): community
    /// p2048 904.487/858=1.054181; AXQ p2048 888.640/862.825=1.029919
    /// (0.997× q2only). Wash. Not whole-FFN flatten, not dual-stream.
    qwen_la_flat_inputs_enabled,
    "AX_MLX_QWEN_LA_FLAT_INPUTS"
);

/// Whether packed LA inputs should flatten to 2-D before qmm.
pub fn should_qwen_la_flat_inputs(seq: i32) -> bool {
    should_qwen_la_flat_inputs_for(qwen_la_flat_inputs_enabled(), seq)
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_CONTIGUOUS_LA_INPUT` — `contiguous([B,S,H])` the
    /// linear-attn activation before the two packed QKVZ/BA prefill qmms at
    /// `seq >= 1024`. Not FFN `contiguous([B,S,H])` (washed), not LA *weight*
    /// contiguous (washed), not QKV-before-conv1d (washed), not flatten
    /// (washed), not dequant-dense.
    ///
    /// **Default: OFF**. Remasured binary `2680dd89…` (2026-08-14): 3b
    /// 1.028100 / 3d 1.052324 wash (0.996× q2only). Contiguous LA input
    /// does not cut compute-bound qmm.
    qwen_prefill_contiguous_la_input_enabled,
    "AX_MLX_QWEN_PREFILL_CONTIGUOUS_LA_INPUT"
);

/// Whether LA prefill should `contiguous` the activation before QKVZ/BA qmm.
pub fn should_qwen_prefill_contiguous_la_input(model_family: &str, seq: i32) -> bool {
    should_qwen_prefill_contiguous_la_input_for(
        qwen_prefill_contiguous_la_input_enabled(),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_qwen_prefill_contiguous_la_input`].
pub fn should_qwen_prefill_contiguous_la_input_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
) -> bool {
    enabled
        && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

/// Pure helper for [`should_qwen_la_flat_inputs`].
pub fn should_qwen_la_flat_inputs_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_LA_CONTIGUOUS_QKV` — `contiguous` the packed QKV
    /// activation before the post-input depthwise conv1d at `seq >= 1024`.
    ///
    /// **Default: OFF**. Remasured binary `0f01c381…` (2026-08-13): community
    /// p2048 904.710/858=1.054442; AXQ p2048 887.915/862.825=1.029078
    /// (0.997× q2only). Wash. Not GatedDelta contiguous, not FFN contiguous.
    qwen_la_contiguous_qkv_enabled,
    "AX_MLX_QWEN_LA_CONTIGUOUS_QKV"
);

/// Whether packed LA QKV should be materialized before post-input conv1d.
pub fn should_qwen_la_contiguous_qkv(seq: i32) -> bool {
    should_qwen_la_contiguous_qkv_for(qwen_la_contiguous_qkv_enabled(), seq)
}

/// Pure helper for [`should_qwen_la_contiguous_qkv`].
pub fn should_qwen_la_contiguous_qkv_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_LA_PREFILL_Q2_PROJ` — use a load-time 2-bit gs32 overlay
    /// of packed QKVZ/BA for `seq >= 1024`. Decode keeps the checkpoint pack.
    ///
    /// **Default: OFF**. Remasured binary `82ffde4a…` (2026-08-14): community
    /// p2048 903.735/858=1.053305; AXQ p2048 889.075/862.825=1.030423
    /// (0.998× q2only). Wash. Not Hub requant, not 2-bit decode lm_head.
    qwen_la_prefill_q2_proj_enabled,
    "AX_MLX_QWEN_LA_PREFILL_Q2_PROJ"
);

/// Whether packed LA prefill should use the 2-bit projection overlay.
pub fn should_qwen_la_prefill_q2(seq: i32) -> bool {
    should_qwen_la_prefill_q2_for(qwen_la_prefill_q2_proj_enabled(), seq)
}

/// Pure helper for [`should_qwen_la_prefill_q2`].
pub fn should_qwen_la_prefill_q2_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_Q2_DOWN` — use a 2-bit gs32 overlay of dense
    /// `down_proj` for `seq >= 1024`. Decode and gate/up stay on the
    /// checkpoint pack.
    ///
    /// **Default: OFF**. Remasured wash/regression on M5 (3b 1.028080 /
    /// 3d 1.053854). Not Hub requant, not 2-bit LA QKVZ/BA (washed), not
    /// 2-bit decode lm_head.
    qwen_prefill_q2_down_enabled,
    "AX_MLX_QWEN_PREFILL_Q2_DOWN"
);

/// Whether Qwen split prefill should use a 2-bit down overlay.
pub fn should_qwen_prefill_q2_down(seq: i32) -> bool {
    should_qwen_prefill_q2_down_for(qwen_prefill_q2_down_enabled(), seq)
}

/// Pure helper for [`should_qwen_prefill_q2_down`].
pub fn should_qwen_prefill_q2_down_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_GD_PREFILL_CHUNKWISE` — split GatedDelta prefill at
    /// `seq >= 1024` into no-copy 256-token chunks (B=1 views, no
    /// `contiguous` materialize). Not tile-512 (copies + sequential 512 TG),
    /// not streaming, not compiled GD.
    ///
    /// **Default: OFF**. Remasured binary `282cf2fd…` (2026-08-14): 3b
    /// 1.031608 / 3d 1.054279 wash (0.999× q2only).
    qwen_gd_prefill_chunkwise_enabled,
    "AX_MLX_QWEN_GD_PREFILL_CHUNKWISE"
);

/// Whether GatedDelta prefill should use the no-copy 256-token chunkwise path.
pub fn should_qwen_gd_prefill_chunkwise(seq: i32) -> bool {
    should_qwen_gd_prefill_chunkwise_for(qwen_gd_prefill_chunkwise_enabled(), seq)
}

/// Pure helper for [`should_qwen_gd_prefill_chunkwise`].
pub fn should_qwen_gd_prefill_chunkwise_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_FFN_GS64` — runtime gs64 overlay of dense FFN
    /// gate/up/down (and packed gate+up) at `seq >= 1024`. Decode stays on
    /// the checkpoint group size. Same bits, not Hub requant, not 2-bit.
    ///
    /// **Default: OFF**. Remasured binary `4a2744c7…` (2026-08-14): 3b
    /// 1.021428 / 3d 1.051264 regression (0.989× q2only).
    qwen_prefill_ffn_gs64_enabled,
    "AX_MLX_QWEN_PREFILL_FFN_GS64"
);

/// Whether Qwen split/packed prefill should use a gs64 FFN overlay.
pub fn should_qwen_prefill_ffn_gs64(seq: i32) -> bool {
    should_qwen_prefill_ffn_gs64_for(qwen_prefill_ffn_gs64_enabled(), seq)
}

/// Pure helper for [`should_qwen_prefill_ffn_gs64`].
pub fn should_qwen_prefill_ffn_gs64_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_Q3_FFN` — 3-bit gs32 overlay of dense FFN
    /// gate/up/down (and packed gate+up) at `seq >= 1024`. Decode stays on
    /// the checkpoint pack. Not Hub requant, not 2-bit down (washed), not
    /// gs64 (regressed).
    ///
    /// **Default: OFF**. Remasured binary `dc7036c2…` (2026-08-14): 3b
    /// 0.998390 / 3d 1.014054 regression (0.967× q2only). MLX 3-bit qmm
    /// is slower than 4/6-bit at M=1024.
    qwen_prefill_q3_ffn_enabled,
    "AX_MLX_QWEN_PREFILL_Q3_FFN"
);

/// Whether Qwen split/packed prefill should use a 3-bit FFN overlay.
pub fn should_qwen_prefill_q3_ffn(seq: i32) -> bool {
    should_qwen_prefill_q3_ffn_for(qwen_prefill_q3_ffn_enabled(), seq)
}

/// Pure helper for [`should_qwen_prefill_q3_ffn`].
pub fn should_qwen_prefill_q3_ffn_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_CONTIGUOUS_FFN_WEIGHTS` — materialize contiguous
    /// FFN `weight`/`scales`/`biases` for `seq >= 1024` qmm. Decode keeps the
    /// checkpoint views. Not activation `contiguous([B,S,H])` (washed), not a
    /// fuse, not a bit-width overlay.
    ///
    /// **Default: OFF**. Remasured binary `99c3b4cc…` (2026-08-14): 3b
    /// 1.027826 / 3d 1.051032 wash (0.995× q2only).
    qwen_prefill_contiguous_ffn_weights_enabled,
    "AX_MLX_QWEN_PREFILL_CONTIGUOUS_FFN_WEIGHTS"
);

/// Whether Qwen prefill should use contiguous FFN quantized tensors.
pub fn should_qwen_prefill_contiguous_ffn_weights(seq: i32) -> bool {
    should_qwen_prefill_contiguous_ffn_weights_for(
        qwen_prefill_contiguous_ffn_weights_enabled(),
        seq,
    )
}

/// Pure helper for [`should_qwen_prefill_contiguous_ffn_weights`].
pub fn should_qwen_prefill_contiguous_ffn_weights_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_ASYNC_GATE_UP` — `async_eval([gate, up])` at
    /// `seq >= 1024` so the two split FFN qmms (community) or the packed
    /// gate+up qmm (AXQ) submit before SwiGLU/down is built. Not GPU
    /// dual-stream (closed), not pipeline-block `async_eval(hidden)` (washed),
    /// not a fuse/compile/bit-width overlay.
    ///
    /// **Default: OFF**. Remasured binary `aebcaa13…` (2026-08-14): 3b
    /// 1.024894 / 3d 1.049201 wash (0.992× q2only).
    qwen_prefill_async_gate_up_enabled,
    "AX_MLX_QWEN_PREFILL_ASYNC_GATE_UP"
);

/// Whether Qwen prefill should async-submit gate/up before down.
pub fn should_qwen_prefill_async_gate_up(seq: i32) -> bool {
    should_qwen_prefill_async_gate_up_for(qwen_prefill_async_gate_up_enabled(), seq)
}

/// Pure helper for [`should_qwen_prefill_async_gate_up`].
pub fn should_qwen_prefill_async_gate_up_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_FFN_F32_INPUT` — cast Qwen dense FFN activations
    /// to Float32 for `seq >= 1024` qmm, then restore the original dtype.
    /// Not a fuse, not compile, not a bit-width overlay, not async-gate-up.
    ///
    /// **Default: OFF**. Remasured binary `128d9a6c…` (2026-08-14): 3b
    /// 0.851037 / 3d 0.870636 regression (0.824× q2only). F32 activations
    /// make the steel qmm slower at M=1024.
    qwen_prefill_ffn_f32_input_enabled,
    "AX_MLX_QWEN_PREFILL_FFN_F32_INPUT"
);

/// Whether Qwen prefill FFN should run qmm in Float32.
pub fn should_qwen_prefill_ffn_f32_input(seq: i32) -> bool {
    should_qwen_prefill_ffn_f32_input_for(qwen_prefill_ffn_f32_input_enabled(), seq)
}

/// Pure helper for [`should_qwen_prefill_ffn_f32_input`].
pub fn should_qwen_prefill_ffn_f32_input_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_EVAL_FFN_INPUT` — `eval(&[x])` the Qwen dense
    /// FFN activation at `seq >= 1024` before gate/up/packed qmm so the
    /// residual+norm graph materializes once. Not a fuse, not compile, not
    /// a bit-width overlay, not async-gate-up, not F32-input.
    ///
    /// **Default: OFF**. Remasured binary `56b9ea50…` (2026-08-14): 3b
    /// 1.021296 / 3d 1.043878 wash (0.989× q2only). Extra sync eval does
    /// not cut compute-bound qmm.
    qwen_prefill_eval_ffn_input_enabled,
    "AX_MLX_QWEN_PREFILL_EVAL_FFN_INPUT"
);

/// Whether Qwen prefill FFN should eval its input before qmm.
pub fn should_qwen_prefill_eval_ffn_input(seq: i32) -> bool {
    should_qwen_prefill_eval_ffn_input_for(qwen_prefill_eval_ffn_input_enabled(), seq)
}

/// Pure helper for [`should_qwen_prefill_eval_ffn_input`].
pub fn should_qwen_prefill_eval_ffn_input_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_EVAL_LA_INPUT` — `eval(&[x])` the Qwen linear-
    /// attention activation at `seq >= 1024` before QKVZ/BA qmm so the
    /// residual+norm graph materializes once. Not a fuse, not compile, not
    /// a bit-width overlay, not async-gate-up, not F32-input, not FFN-input
    /// eval (closed wash).
    ///
    /// **Default: OFF**. Remasured binary `e2e8dc60…` (2026-08-14): 3b
    /// 1.018881 / 3d 1.044070 wash (0.987× q2only). Sync eval of LA input
    /// does not cut compute-bound qmm.
    qwen_prefill_eval_la_input_enabled,
    "AX_MLX_QWEN_PREFILL_EVAL_LA_INPUT"
);

/// Whether Qwen prefill LA should eval its input before qmm.
pub fn should_qwen_prefill_eval_la_input(seq: i32) -> bool {
    should_qwen_prefill_eval_la_input_for(qwen_prefill_eval_la_input_enabled(), seq)
}

/// Pure helper for [`should_qwen_prefill_eval_la_input`].
pub fn should_qwen_prefill_eval_la_input_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_ASYNC_LA_OUTPUTS` — `async_eval([qkv,z,a,b])`
    /// at `seq >= 1024` after packed LA projections so the QKVZ/BA qmm
    /// submits before conv/GatedDelta is built. Not GPU dual-stream
    /// (closed), not FFN async-gate-up (closed), not sync eval of LA/FFN
    /// input (closed washes), not a fuse/compile/bit-width overlay.
    ///
    /// **Default: OFF**. Remasured binary `d27396ad…` (2026-08-14): 3b
    /// 1.029902 / 3d 1.054108 wash (0.997× q2only). Async submit of LA
    /// outputs does not cut compute-bound qmm.
    qwen_prefill_async_la_outputs_enabled,
    "AX_MLX_QWEN_PREFILL_ASYNC_LA_OUTPUTS"
);

/// Whether Qwen prefill should async-submit packed LA outputs.
pub fn should_qwen_prefill_async_la_outputs(seq: i32) -> bool {
    should_qwen_prefill_async_la_outputs_for(qwen_prefill_async_la_outputs_enabled(), seq)
}

/// Pure helper for [`should_qwen_prefill_async_la_outputs`].
pub fn should_qwen_prefill_async_la_outputs_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_ASYNC_PACKED_GATE_UP` — `async_eval([packed])`
    /// at `seq >= 1024` after the packed gate+up qmm so it submits before
    /// SwiGLU/down is built. Not split `async_eval([gate,up])` (closed),
    /// not GPU dual-stream, not a fuse/compile/bit-width overlay.
    ///
    /// **Default: OFF**. Remasured binary `7e6557af…` (2026-08-14): 3b
    /// 1.026856 / 3d 1.050946 wash (0.994× q2only). Async submit of packed
    /// gate+up does not cut compute-bound qmm.
    qwen_prefill_async_packed_gate_up_enabled,
    "AX_MLX_QWEN_PREFILL_ASYNC_PACKED_GATE_UP"
);

/// Whether Qwen prefill should async-submit packed gate+up before SwiGLU.
pub fn should_qwen_prefill_async_packed_gate_up(seq: i32) -> bool {
    should_qwen_prefill_async_packed_gate_up_for(qwen_prefill_async_packed_gate_up_enabled(), seq)
}

/// Pure helper for [`should_qwen_prefill_async_packed_gate_up`].
pub fn should_qwen_prefill_async_packed_gate_up_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_CONTIGUOUS_LA_WEIGHTS` — contiguous QKVZ/BA
    /// `weight`/`scales`/`biases` for `seq >= 1024` qmm. Decode keeps the
    /// checkpoint views. Not FFN contiguous-weights (closed wash), not
    /// activation contiguous, not a fuse/compile/bit-width overlay.
    ///
    /// **Default: OFF**. Remasured binary `8578ce78…` (2026-08-14): 3b
    /// 1.025765 / 3d 1.050233 wash (0.993× q2only). Contiguous LA weights
    /// do not cut compute-bound qmm.
    qwen_prefill_contiguous_la_weights_enabled,
    "AX_MLX_QWEN_PREFILL_CONTIGUOUS_LA_WEIGHTS"
);

/// Whether Qwen prefill should use contiguous LA quantized tensors.
pub fn should_qwen_prefill_contiguous_la_weights(seq: i32) -> bool {
    should_qwen_prefill_contiguous_la_weights_for(qwen_prefill_contiguous_la_weights_enabled(), seq)
}

/// Pure helper for [`should_qwen_prefill_contiguous_la_weights`].
pub fn should_qwen_prefill_contiguous_la_weights_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_EVAL_ATTN_INPUT` — `eval(&[x])` the Qwen full-
    /// attention activation at `seq >= 1024` before QKVO qmm so the
    /// residual+norm graph materializes once. Not a fuse, not compile, not
    /// a bit-width overlay, not FFN/LA input eval (closed washes).
    ///
    /// **Default: OFF**. Remasured binary `03568893…` (2026-08-14): 3b
    /// 1.023343 / 3d 1.047094 wash (0.991× q2only). Sync eval of attn
    /// input does not cut compute-bound qmm.
    qwen_prefill_eval_attn_input_enabled,
    "AX_MLX_QWEN_PREFILL_EVAL_ATTN_INPUT"
);

/// Whether Qwen prefill attention should eval its input before qmm.
pub fn should_qwen_prefill_eval_attn_input(model_family: &str, seq: i32) -> bool {
    should_qwen_prefill_eval_attn_input_for(
        qwen_prefill_eval_attn_input_enabled(),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_qwen_prefill_eval_attn_input`].
pub fn should_qwen_prefill_eval_attn_input_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
) -> bool {
    enabled
        && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_EVAL_FFN_HIDDEN` — `eval(&[h])` the Qwen SwiGLU
    /// activation at `seq >= 1024` before down qmm so silu_mul materializes
    /// once. Not eval of FFN *input* (closed wash), not a fuse, not compile,
    /// not a bit-width overlay.
    ///
    /// **Default: OFF**. Remasured binary `5aa63cc4…` (2026-08-14): 3b
    /// 1.015480 / 3d 1.038232 regression (0.983× q2only). Sync eval of
    /// SwiGLU hidden before down does not cut compute-bound qmm.
    qwen_prefill_eval_ffn_hidden_enabled,
    "AX_MLX_QWEN_PREFILL_EVAL_FFN_HIDDEN"
);

/// Whether Qwen prefill FFN should eval SwiGLU hidden before down qmm.
pub fn should_qwen_prefill_eval_ffn_hidden(seq: i32) -> bool {
    should_qwen_prefill_eval_ffn_hidden_for(qwen_prefill_eval_ffn_hidden_enabled(), seq)
}

/// Pure helper for [`should_qwen_prefill_eval_ffn_hidden`].
pub fn should_qwen_prefill_eval_ffn_hidden_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_CONTIGUOUS_ATTN_WEIGHTS` — contiguous QKVO /
    /// packed-QKV `weight`/`scales`/`biases` for `seq >= 1024` qmm. Decode
    /// keeps the checkpoint views. Not FFN/LA contiguous-weights (closed
    /// washes), not a fuse/compile/bit-width overlay.
    ///
    /// **Default: OFF**. Remasured binary `a91776cc…` (2026-08-14): 3b
    /// 1.026977 / 3d 1.050236 wash (0.994× q2only). Contiguous attn
    /// weights do not cut compute-bound qmm.
    qwen_prefill_contiguous_attn_weights_enabled,
    "AX_MLX_QWEN_PREFILL_CONTIGUOUS_ATTN_WEIGHTS"
);

/// Whether Qwen prefill should use contiguous attention quantized tensors.
pub fn should_qwen_prefill_contiguous_attn_weights(model_family: &str, seq: i32) -> bool {
    should_qwen_prefill_contiguous_attn_weights_for(
        qwen_prefill_contiguous_attn_weights_enabled(),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_qwen_prefill_contiguous_attn_weights`].
pub fn should_qwen_prefill_contiguous_attn_weights_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
) -> bool {
    enabled
        && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_SKIP_UNUSED_LA_OUT` — on cache-only last-layer
    /// prefill (`skip_post_attention_ffn`), skip LA `out_proj` and the
    /// discarded residual add after conv/recurrent state is written. Not
    /// skip unused LA out *reshape* (closed wash), not a fuse/compile.
    ///
    /// **Default: OFF**. Remasured binary `ce7dd8ae…` (2026-08-14): 3b
    /// 1.026321 / 3d 1.050153 wash (0.994× q2only). Skipping unused last-
    /// layer LA out_proj does not cut compute-bound qmm.
    qwen_prefill_skip_unused_la_out_enabled,
    "AX_MLX_QWEN_PREFILL_SKIP_UNUSED_LA_OUT"
);

/// Whether Qwen cache-only last-layer prefill should skip unused LA out_proj.
pub fn should_qwen_prefill_skip_unused_la_out(
    model_family: &str,
    skip_post_attention_ffn: bool,
    seq: i32,
) -> bool {
    should_qwen_prefill_skip_unused_la_out_for(
        qwen_prefill_skip_unused_la_out_enabled(),
        model_family,
        skip_post_attention_ffn,
        seq,
    )
}

/// Pure helper for [`should_qwen_prefill_skip_unused_la_out`].
pub fn should_qwen_prefill_skip_unused_la_out_for(
    enabled: bool,
    model_family: &str,
    skip_post_attention_ffn: bool,
    seq: i32,
) -> bool {
    enabled
        && skip_post_attention_ffn
        && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_ASYNC_DOWN` — `async_eval([down])` at
    /// `seq >= 1024` after the dense FFN down qmm so it submits before
    /// residual/next-layer rms is built. Not split async-gate-up (closed),
    /// not async-packed-gate-up (closed), not pipeline-block (closed).
    ///
    /// **Default: OFF**. Remeasured on `df-macbookpro-m5` after-async-down
    /// (`09d9a68d…`) 3b 1.025088 / 3d 1.048606 wash (0.993× q2only).
    qwen_prefill_async_down_enabled,
    "AX_MLX_QWEN_PREFILL_ASYNC_DOWN"
);

/// Whether Qwen prefill should async-submit FFN down before residual.
pub fn should_qwen_prefill_async_down(seq: i32) -> bool {
    should_qwen_prefill_async_down_for(qwen_prefill_async_down_enabled(), seq)
}

/// Pure helper for [`should_qwen_prefill_async_down`].
pub fn should_qwen_prefill_async_down_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_LAST_TOKEN_O_PROJ` — on last-position-only
    /// generate prefill (`seq >= 1024`), slice SDPA / LA recurrent output
    /// to the last token *before* flatten + o_proj / LA out_proj. KV and
    /// linear state are already written. Not skip-unused-LA-out (cache-only
    /// skip of the whole last-layer out_proj), not a fuse/compile/bit-width
    /// overlay, not eval/async/contiguous.
    ///
    /// **Default: OFF**. Remasured binary `ad3de508…` (2026-08-14): 3b
    /// 1.026098 / 3d 1.051811 wash (0.994× q2only). One last-layer o_proj
    /// slice cannot move the 1.15/1.20 bars.
    qwen_prefill_last_token_o_proj_enabled,
    "AX_MLX_QWEN_PREFILL_LAST_TOKEN_O_PROJ"
);

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_REUSE_ROPE` — precompute NeoX cos/sin once per
    /// prefill forward (`seq >= 1024`) and apply to every full-attn Q/K.
    /// Not compiled QK-RoPE (closed), not fused prefill attention (closed),
    /// not last-token o_proj (closed 3d miss), not eval/async/contiguous.
    ///
    /// **Default: OFF**. Remasured binary `fc8b7496…` (2026-08-14): community
    /// and AXQ p2048 crashed (`SSE stream ended without a terminal response`)
    /// on the first 1024-token chunk. Skipping fused C++ qk_norm_rope for the
    /// portable apply is a regression. Helpers stay for the unit tests.
    qwen_prefill_reuse_rope_enabled,
    "AX_MLX_QWEN_PREFILL_REUSE_ROPE"
);

/// Whether Qwen prefill should reuse one cos/sin table across full-attn layers.
pub fn should_qwen_prefill_reuse_rope(model_family: &str, seq: i32) -> bool {
    should_qwen_prefill_reuse_rope_for(qwen_prefill_reuse_rope_enabled(), model_family, seq)
}

/// Pure helper for [`should_qwen_prefill_reuse_rope`].
pub fn should_qwen_prefill_reuse_rope_for(enabled: bool, model_family: &str, seq: i32) -> bool {
    enabled
        && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

/// Whether Qwen last-only prefill should run o_proj on the last token only.
pub fn should_qwen_prefill_last_token_o_proj(
    model_family: &str,
    last_position_only: bool,
    seq: i32,
) -> bool {
    should_qwen_prefill_last_token_o_proj_for(
        qwen_prefill_last_token_o_proj_enabled(),
        model_family,
        last_position_only,
        seq,
    )
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_LAST_QUERY_SDPA` — on last-position-only generate
    /// prefill (`seq >= 1024`), write full K/V then run Q+SDPA on the last
    /// query only. Implied when `AX_MLX_QWEN_PREFILL_LAST_QUERY_Q_PROJ` is
    /// on (Q is already S=1). Not last-token o_proj (closed: slices *after*
    /// full SDPA), not skip-unused-LA-out (last 27B layer is full-attn).
    ///
    /// **Default: OFF**. Remasured binary `6d4e0d38…` (2026-08-14): 3b
    /// 1.029076 / 3d 1.054203 wash (0.997× q2only). Last-query SDPA is one
    /// full-attn layer and does not cut compute-bound qmm.
    qwen_prefill_last_query_sdpa_enabled,
    "AX_MLX_QWEN_PREFILL_LAST_QUERY_SDPA"
);

/// Whether Qwen last-only prefill should SDPA the last query only.
/// Last-token Q proj and skip-unused-QK-norm already yield S=1 Q, so those
/// flags imply this one.
pub fn should_qwen_prefill_last_query_sdpa(
    model_family: &str,
    last_position_only: bool,
    seq: i32,
) -> bool {
    should_qwen_prefill_last_query_sdpa_for(
        qwen_prefill_last_query_sdpa_enabled()
            || qwen_prefill_last_query_q_proj_enabled()
            || qwen_prefill_skip_unused_qk_norm_enabled(),
        model_family,
        last_position_only,
        seq,
    )
}

/// Pure helper for [`should_qwen_prefill_last_query_sdpa`].
pub fn should_qwen_prefill_last_query_sdpa_for(
    enabled: bool,
    model_family: &str,
    last_position_only: bool,
    seq: i32,
) -> bool {
    enabled
        && last_position_only
        && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_LAST_QUERY_Q_PROJ` — on last-position-only
    /// generate prefill (`seq >= 1024`), write full K/V then run `q_proj`
    /// on the last token only. Not last-query SDPA (closed: still paid
    /// full-seq Q qmm then sliced), not last-token o_proj (closed).
    ///
    /// **Default: OFF**. First remasure `f763ca23…` crashed p2048 (Q S=1,
    /// SDPA used full-seq). Cleanup remasure binary `13a85878…` (2026-08-14):
    /// no crash; 3a PASS; 3d 903.383/857.999=**1.052896 FAIL** (need 986.699);
    /// vs q2only 0.994 wash. AXQ killed after 3d FAIL. Last-layer Q skip
    /// does not cut compute-bound qmm.
    qwen_prefill_last_query_q_proj_enabled,
    "AX_MLX_QWEN_PREFILL_LAST_QUERY_Q_PROJ"
);

/// Whether Qwen last-only prefill should project Q on the last token only.
pub fn should_qwen_prefill_last_query_q_proj(
    model_family: &str,
    last_position_only: bool,
    seq: i32,
) -> bool {
    should_qwen_prefill_last_query_q_proj_for(
        qwen_prefill_last_query_q_proj_enabled(),
        model_family,
        last_position_only,
        seq,
    )
}

/// Pure helper for [`should_qwen_prefill_last_query_q_proj`].
pub fn should_qwen_prefill_last_query_q_proj_for(
    enabled: bool,
    model_family: &str,
    last_position_only: bool,
    seq: i32,
) -> bool {
    enabled
        && last_position_only
        && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_SKIP_UNUSED_QK_NORM` — on last-position-only
    /// generate prefill (`seq >= 1024`), slice full-seq Q to the last token
    /// *before* QK-norm + RoPE so the discarded prefix does not pay RMSNorm.
    /// Last-token Q proj already yields S=1 (no-op). Not last-query SDPA
    /// (closed: still paid full-seq Q-norm then sliced), not FFN /
    /// contiguous / compile / skip-astype.
    ///
    /// **Default: OFF**. Remasured binary `ece19de3…` (2026-08-14): 3a PASS;
    /// 3d 903.798/857.999=**1.053380 FAIL** (need 986.699); vs q2only 0.995
    /// wash. AXQ killed after 3d FAIL. Last-layer QK-norm skip does not cut
    /// compute-bound qmm.
    qwen_prefill_skip_unused_qk_norm_enabled,
    "AX_MLX_QWEN_PREFILL_SKIP_UNUSED_QK_NORM"
);

/// Whether Qwen last-only prefill should skip unused prefix QK-norm.
pub fn should_qwen_prefill_skip_unused_qk_norm(
    model_family: &str,
    last_position_only: bool,
    seq: i32,
) -> bool {
    should_qwen_prefill_skip_unused_qk_norm_for(
        qwen_prefill_skip_unused_qk_norm_enabled(),
        model_family,
        last_position_only,
        seq,
    )
}

/// Pure helper for [`should_qwen_prefill_skip_unused_qk_norm`].
pub fn should_qwen_prefill_skip_unused_qk_norm_for(
    enabled: bool,
    model_family: &str,
    last_position_only: bool,
    seq: i32,
) -> bool {
    enabled
        && last_position_only
        && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

/// Pure helper for [`should_qwen_prefill_last_token_o_proj`].
pub fn should_qwen_prefill_last_token_o_proj_for(
    enabled: bool,
    model_family: &str,
    last_position_only: bool,
    seq: i32,
) -> bool {
    enabled
        && last_position_only
        && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_ASYNC_SDPA` — `async_eval([sdpa])` at
    /// `seq >= 1024` after full-attn SDPA so it submits before flatten +
    /// o_proj is built. Not async-down (closed), not async-LA-outputs
    /// (closed), not fused-attn (closed), not reuse-RoPE (crashed).
    ///
    /// **Default: OFF**. Remasured binary `db50b0f2…` (2026-08-14): 3b
    /// 1.028962 / 3d 1.050581 wash (0.996× q2only). Async submit of SDPA
    /// does not cut compute-bound qmm.
    qwen_prefill_async_sdpa_enabled,
    "AX_MLX_QWEN_PREFILL_ASYNC_SDPA"
);

/// Whether Qwen prefill should async-submit SDPA before o_proj.
pub fn should_qwen_prefill_async_sdpa(model_family: &str, seq: i32) -> bool {
    should_qwen_prefill_async_sdpa_for(qwen_prefill_async_sdpa_enabled(), model_family, seq)
}

/// Pure helper for [`should_qwen_prefill_async_sdpa`].
pub fn should_qwen_prefill_async_sdpa_for(enabled: bool, model_family: &str, seq: i32) -> bool {
    enabled
        && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_ASYNC_GD` — `async_eval([gd])` at `seq >= 1024`
    /// after GatedDelta so the recurrent kernel submits before
    /// rms_norm_gated + out_proj is built. Not async-LA-outputs (packed
    /// QKVZ/BA, closed), not async-SDPA (closed), not GD-chunkwise (closed).
    ///
    /// **Default: OFF**. Remasured binary `af8582ff…` (2026-08-14): 3b
    /// 1.031631 / 3d 1.054870 wash (0.999× q2only). Async submit of
    /// GatedDelta does not cut compute-bound qmm.
    qwen_prefill_async_gd_enabled,
    "AX_MLX_QWEN_PREFILL_ASYNC_GD"
);

/// Whether Qwen prefill should async-submit GatedDelta before LA out_proj.
pub fn should_qwen_prefill_async_gd(seq: i32) -> bool {
    should_qwen_prefill_async_gd_for(qwen_prefill_async_gd_enabled(), seq)
}

/// Pure helper for [`should_qwen_prefill_async_gd`].
pub fn should_qwen_prefill_async_gd_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_EVAL_GD` — `eval([gd])` at `seq >= 1024` after
    /// GatedDelta so the recurrent kernel materializes once before
    /// rms_norm_gated + out_proj. Not async-GD (closed wash), not
    /// eval-LA-input (before QKVZ/BA, closed), not eval-FFN-hidden (closed).
    ///
    /// **Default: OFF**. Remasured binary `09808d15…` (2026-08-14): 3b
    /// 1.024523 / 3d 1.047978 wash/regression (0.992× q2only). Sync eval of
    /// GatedDelta does not cut compute-bound qmm.
    qwen_prefill_eval_gd_enabled,
    "AX_MLX_QWEN_PREFILL_EVAL_GD"
);

/// Whether Qwen prefill should eval GatedDelta before LA out_proj.
pub fn should_qwen_prefill_eval_gd(seq: i32) -> bool {
    should_qwen_prefill_eval_gd_for(qwen_prefill_eval_gd_enabled(), seq)
}

/// Pure helper for [`should_qwen_prefill_eval_gd`].
pub fn should_qwen_prefill_eval_gd_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_CONTIGUOUS_GD` — `contiguous(gd)` at
    /// `seq >= 1024` after GatedDelta so rms_norm_gated + out_proj see a
    /// packed activation. Not GD-contiguous *inputs* (closed), not FFN/LA/attn
    /// weight contiguous (closed), not eval/async-GD (closed).
    ///
    /// **Default: OFF**. Remasured binary `2863e243…` (2026-08-14): 3b
    /// 1.029082 / 3d 1.053013 wash (0.997× q2only). Contiguous GatedDelta
    /// output does not cut compute-bound qmm.
    qwen_prefill_contiguous_gd_enabled,
    "AX_MLX_QWEN_PREFILL_CONTIGUOUS_GD"
);

/// Whether Qwen prefill should contiguous GatedDelta before LA out_proj.
pub fn should_qwen_prefill_contiguous_gd(seq: i32) -> bool {
    should_qwen_prefill_contiguous_gd_for(qwen_prefill_contiguous_gd_enabled(), seq)
}

/// Pure helper for [`should_qwen_prefill_contiguous_gd`].
pub fn should_qwen_prefill_contiguous_gd_for(enabled: bool, seq: i32) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_SPLIT_PACKED` — at `seq >= 1024`, run Qwen dense
    /// FFN gate/up as two steel `quantized_matmul`s instead of one packed
    /// qmm with 2× output rows. Gemma prefill already prefers split because
    /// two qmatmuls beat packed at 128/512/2048. Not a fuse, compile, async,
    /// contiguous, eval, tile, or bit-width overlay. Not the closed 4-bit
    /// *pack* (that merged split → packed and regressed).
    ///
    /// **Default: OFF**. Remasured binary `efe6e151…` (2026-08-14): 3b
    /// 1.025586 / 3d 1.050229 wash (0.993× q2only). Two steel qmatmuls do
    /// not beat packed 2×-wide qmm at M=1024 on this 27B path.
    qwen_prefill_split_packed_enabled,
    "AX_MLX_QWEN_PREFILL_SPLIT_PACKED"
);

/// Whether Qwen prefill should split packed gate/up into two qmms.
pub fn should_qwen_prefill_split_packed(model_family: &str, seq: i32) -> bool {
    should_qwen_prefill_split_packed_for(qwen_prefill_split_packed_enabled(), model_family, seq)
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_DEQUANT_DENSE` — at `seq >= 1024`, dequantize
    /// affine weights once and run steel dense `matmul` instead of
    /// `quantized_matmul`. This changes the compute kernel (dense GEMM vs
    /// fused dequant+MMA), not overlay/eval/async/contiguous/split/bit-width.
    /// Not a Hub requant. Decode GEMV (seq=1) stays on steel qmm.
    ///
    /// **Default: OFF**. Remasured binary `81a35c6f…` (2026-08-14) with
    /// `AX_MLX_QWEN_PREFILL_DEQUANT_DENSE=1`: 3b **0.998513** / 3d
    /// **1.020320** regression (0.967× q2only). Dense GEMM is slower than
    /// steel qmm at M=1024 on this 27B path.
    qwen_prefill_dequant_dense_enabled,
    "AX_MLX_QWEN_PREFILL_DEQUANT_DENSE"
);

/// Whether Qwen prefill should replace qmm with dequant + dense GEMM.
pub fn should_qwen_prefill_dequant_dense(model_family: &str, seq: i32) -> bool {
    should_qwen_prefill_dequant_dense_for(qwen_prefill_dequant_dense_enabled(), model_family, seq)
}

env_flag!(
    /// `AX_MLX_QWEN_LA_NORM_QKVZ_FUSE` — at `seq >= 1024`, fuse linear-attn
    /// `attn_norm` into packed QKVZ/BA `quantized_matmul` via
    /// `rms_norm_quantized_matmul`. Not FFN fuse, not dequant-dense, not the
    /// closed full-attn `AX_MLX_QWEN_ATTN_NORM_QKV_FUSE`.
    ///
    /// **Default: OFF**. Remasured binary `1fe62f3b…` (2026-08-14): 3b
    /// 1.027604 / 3d 1.052036 wash (0.995× q2only). Fusing RMSNorm into LA
    /// qmm does not cut compute-bound qmm.
    qwen_la_norm_qkvz_fuse_enabled,
    "AX_MLX_QWEN_LA_NORM_QKVZ_FUSE"
);

/// Whether Qwen prefill should fuse attn RMSNorm into LA QKVZ/BA qmm.
pub fn should_qwen_la_norm_qkvz_fuse(model_family: &str, seq: i32) -> bool {
    should_qwen_la_norm_qkvz_fuse_for(qwen_la_norm_qkvz_fuse_enabled(), model_family, seq)
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_SKIP_BF16_ASTYPE` — when the embedding gather is
    /// already BF16, skip the redundant `astype(..., bfloat16)` on Qwen
    /// prefill (`seq > 1`). Not FFN, not contiguous, not compile.
    ///
    /// **Default: OFF**. Remasured binary `f2afbf68…` (2026-08-14): 3b
    /// 1.027348 / 3d 1.052426 wash (0.995× q2only). Skipping a no-op
    /// embed astype does not cut compute-bound qmm.
    qwen_prefill_skip_bf16_astype_enabled,
    "AX_MLX_QWEN_PREFILL_SKIP_BF16_ASTYPE"
);

/// Whether Qwen prefill should skip a no-op BF16 astype.
pub fn should_qwen_prefill_skip_bf16_astype(model_family: &str, seq: i32) -> bool {
    should_qwen_prefill_skip_bf16_astype_for(
        qwen_prefill_skip_bf16_astype_enabled(),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_qwen_prefill_skip_bf16_astype`].
pub fn should_qwen_prefill_skip_bf16_astype_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
) -> bool {
    enabled
        && seq > 1
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_FLAT_QMM` — reshape every Qwen prefill `qw`
    /// activation `[B,S,H] → [B*S,H]` before steel `quantized_matmul`, then
    /// restore `[B,S,out]`. Not FFN-only flat-ffn (closed wash), not last-
    /// layer Q/SDPA (closed crash/wash), not dequant-dense (regression).
    /// Changes the qmm leading shape so steel can pick a 2-D GEMM kernel.
    ///
    /// **Default: OFF**. Remasured binary `8926e7f1…` (2026-08-14): community
    /// p2048 903.763/857.999=1.053338 wash (0.995× q2only). Flattening every
    /// prefill qmm to 2-D does not beat 3-D steel qmm. AXQ killed after 3d
    /// FAIL. Helpers stay for the unit tests.
    qwen_prefill_flat_qmm_enabled,
    "AX_MLX_QWEN_PREFILL_FLAT_QMM"
);

/// Whether every Qwen prefill qmm should flatten to a 2-D leading dim.
pub fn should_qwen_prefill_flat_qmm(seq: i32, rank: usize) -> bool {
    should_qwen_prefill_flat_qmm_for(qwen_prefill_flat_qmm_enabled(), seq, rank)
}

/// Pure helper for [`should_qwen_prefill_flat_qmm`].
pub fn should_qwen_prefill_flat_qmm_for(enabled: bool, seq: i32, rank: usize) -> bool {
    enabled && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ && rank == 3
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_TILE_QMM` — run steel `quantized_matmul` on
    /// 512-token slices of `[B,S,H]` at `seq >= 1024`, then concatenate.
    /// Changes the qmm leading dim (M=512 vs M=1024). Not flatten-qmm
    /// (closed: rank change), not chunk-1280/single-2048 (closed), not
    /// last-layer Q/SDPA/QK-norm, not bit-width overlay.
    ///
    /// **Default: OFF**. Remasured binary `6bf46184…` (2026-08-14): 3a PASS
    /// (p2048 pre 869.531/931.742=0.933231); 3d 869.531/857.999=**1.013440
    /// FAIL** (need 986.699); vs q2only **0.957 regression**. Two M=512
    /// steel launches lose to one M=1024. AXQ killed after 3d FAIL.
    qwen_prefill_tile_qmm_enabled,
    "AX_MLX_QWEN_PREFILL_TILE_QMM"
);

/// Whether Qwen prefill qmm should tile the sequence dim.
pub fn should_qwen_prefill_tile_qmm(model_family: &str, seq: i32) -> bool {
    should_qwen_prefill_tile_qmm_for(qwen_prefill_tile_qmm_enabled(), model_family, seq)
}

/// Pure helper for [`should_qwen_prefill_tile_qmm`].
pub fn should_qwen_prefill_tile_qmm_for(enabled: bool, model_family: &str, seq: i32) -> bool {
    enabled
        && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

pub const QWEN_PREFILL_QMM_TILE: i32 = 512;

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_DUAL_AFFINE_QMM` — one C++ call for Qwen split
    /// prefill gate+up steel `quantized_matmul` at `seq >= 1024`. Not
    /// tile/flatten (closed), not compile-block (compiled dual / split FFN
    /// compile closed), not dual-stream (regression), not dual_qmm_swiglu
    /// (regression), not last-layer skip. Qwen never reached the existing
    /// Gemma `uses_geglu` dual-affine branch.
    ///
    /// **Default: OFF**. Remasured binary `94067aea…` (2026-08-15): 3a PASS
    /// (p2048 pre 901.800/930.594=0.969058); 3d 901.800/857.999=**1.051050
    /// FAIL** (need 986.699); vs q2only 0.993 wash. One C++ FFI for two
    /// steel qmms does not beat two Rust qw. AXQ killed after 3d FAIL.
    qwen_prefill_dual_affine_qmm_enabled,
    "AX_MLX_QWEN_PREFILL_DUAL_AFFINE_QMM"
);

/// Whether Qwen split prefill should issue gate+up as one dual-affine qmm.
pub fn should_qwen_prefill_dual_affine_qmm(model_family: &str, seq: i32) -> bool {
    should_qwen_prefill_dual_affine_qmm_for(
        qwen_prefill_dual_affine_qmm_enabled(),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_qwen_prefill_dual_affine_qmm`].
pub fn should_qwen_prefill_dual_affine_qmm_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
) -> bool {
    enabled
        && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_SKIP_UNUSED_EMBED_CLIP` — skip the Metal `clip`
    /// of token ids before embed gather on Qwen prefill (`seq >= 1024`).
    /// Contract prompts are in-range; the clip is a safety gather for
    /// malformed client ids. Not FFN/qmm/compile/last-layer, not
    /// skip-astype / async-embed (closed washes).
    ///
    /// **Default: OFF**. Remasured binary `3ee188a3…` (2026-08-15): 3a PASS
    /// (p2048 pre 903.379/931.044=0.970286); 3d 903.379/857.999=**1.052891
    /// FAIL** (need 986.699); vs q2only 0.994 wash. AXQ killed after 3d
    /// FAIL. Skipping one embed-id clip does not cut compute-bound qmm.
    qwen_prefill_skip_unused_embed_clip_enabled,
    "AX_MLX_QWEN_PREFILL_SKIP_UNUSED_EMBED_CLIP"
);

env_flag_default_on!(
    /// `AX_MLX_QWEN_PREFILL_SKIP_UNUSED_F32_SDPA` — keep Qwen prefill SDPA
    /// in the model dtype at `seq >= 128` (was `>= 1024`; lowered to the
    /// Gemma 4 contract gate for the fleet short-prefill miss — p128/p512
    /// paid the upcast on every full-attn layer). `AX_MLX_MULTI_TOKEN_F32_ATTENTION`
    /// is default-ON for Gemma short teacher-forced verify; on 27B prefill
    /// it upcasts every full-attn Q/K/V to f32 (16 layers × two 1024
    /// chunks). Decode `seq==1` and short MTP verify stay on the Gemma
    /// exactness path. Not FFN/qmm/compile/last-layer, not skip-embed-clip.
    ///
    /// **Default: ON** (kill switch `=0`). Remasured binary `fbf0b12d…`
    /// (2026-08-15): 3a PASS (p2048 pre 914.746/931.574=0.981936); 3d
    /// 914.746/857.999=**1.066139 FAIL** (need 986.699); vs q2only
    /// **1.0068** small gain, not 1.15. AXQ killed after 3d FAIL. bf16 SDPA
    /// does not cut compute-bound qmm.
    /// M3 Max re-measure (2026-08-16, qwen3.8-27b-axq-4bit, fair_prefill_
    /// bench_probe, 5 reps): p2048 192.1 vs 191.9 baseline (wash); p10240
    /// **172.0 vs 166.7 (+3.2%)** — the f32 K/V astype cost grows with the
    /// cached prefix, so the win appears at agentic prompt lengths the p2048
    /// gate never exercised. Do NOT stack with NATIVE_OFFSET_CAUSAL: the
    /// pair regresses (176.6 @ p2048, 165.6 @ p10240).
    qwen_prefill_skip_unused_f32_sdpa_enabled,
    "AX_MLX_QWEN_PREFILL_SKIP_UNUSED_F32_SDPA"
);

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_BF16_EMBED_DEQUANT` — dequantize gathered
    /// embedding rows to BF16 instead of the MLX default f32, then
    /// `astype` to BF16. Not skip-astype (that only skips a no-op when
    /// the gather is already BF16), not skip-embed-clip, not FFN/qmm/
    /// compile/last-layer/f32-SDPA.
    ///
    /// **Default: OFF**. Remasured binary `5afde179…` (2026-08-15): 3a PASS
    /// (p2048 pre 901.813/930.039=0.969651); 3d 901.813/857.999=**1.051065
    /// FAIL** (need 986.699); vs q2only 0.993 wash. AXQ killed after 3d
    /// FAIL. BF16 embed dequant does not cut compute-bound qmm.
    qwen_prefill_bf16_embed_dequant_enabled,
    "AX_MLX_QWEN_PREFILL_BF16_EMBED_DEQUANT"
);

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_NATIVE_OFFSET_CAUSAL` — on Qwen prefill
    /// `seq >= 1024`, use MLX native `mask="causal"` for the offset-1024
    /// second chunk instead of an O(seq×key) bool array. Prior
    /// `AX_MLX_NATIVE_OFFSET_CAUSAL` remasure (889 vs 891) ran under the
    /// default-ON f32 SDPA upcast; this pairs with
    /// `AX_MLX_QWEN_PREFILL_SKIP_UNUSED_F32_SDPA`. Not FFN/qmm/compile/
    /// last-layer, not embed-clip / bf16-dequant.
    ///
    /// **Default: OFF**. Remasured binary `890e5d30…` (2026-08-15) stacked
    /// with `SKIP_UNUSED_F32_SDPA=1`: 3a PASS (p2048 pre
    /// 915.689/930.961=0.983596); 3d 915.689/857.999=**1.067238 FAIL**
    /// (need 986.699); vs q2only 1.008; vs f32-skip-only 1.001 wash.
    /// AXQ killed after 3d FAIL. Native causal does not add on top of bf16
    /// SDPA.
    qwen_prefill_native_offset_causal_enabled,
    "AX_MLX_QWEN_PREFILL_NATIVE_OFFSET_CAUSAL"
);

env_flag_default_on!(
    /// `AX_MLX_NAX_ATTENTION` — allow Neural Accelerator attention policy
    /// (Qwen full-attn native `mask="causal"` at `seq >= 1024`) on M5+
    /// running macOS 26.2+.
    ///
    /// **Default: ON** when [`crate::hardware::neural_accelerator_active`]
    /// is true. Kill-switch via `AX_MLX_NAX_ATTENTION=0`. Off-switch only:
    /// the flag cannot enable the route on M1–M4 or macOS < 26.2.
    nax_attention_allowed,
    "AX_MLX_NAX_ATTENTION"
);

/// Hardware + kill-switch predicate for NAX attention policy.
pub fn nax_attention_enabled() -> bool {
    nax_attention_enabled_for(
        nax_attention_allowed(),
        crate::hardware::neural_accelerator_active(),
    )
}

/// Pure helper for [`nax_attention_enabled`].
pub fn nax_attention_enabled_for(allowed: bool, neural_accelerator_active: bool) -> bool {
    allowed && neural_accelerator_active
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_SKIP_UNUSED_SWIGLU_COMPILE` — at `seq >= 1024`,
    /// run Qwen `silu(gate)*up` as imperative `silu_mul` instead of the
    /// default-ON `AX_MLX_PREFILL_FFN_COMPILE_SWIGLU` closure. Packed Qwen
    /// prefill compile is already known slower at the 512-token boundary;
    /// this is the shipped split-activation compile tax (mutex + try_apply
    /// on every dense FFN layer × two 1024 chunks). Not an opt-in FFN-block
    /// / down / dual-gate compile remasure (those closed as wash).
    ///
    /// **Default: OFF**.
    qwen_prefill_skip_unused_swiglu_compile_enabled,
    "AX_MLX_QWEN_PREFILL_SKIP_UNUSED_SWIGLU_COMPILE"
);

/// Whether Qwen prefill should skip the unused SwiGLU compile.
pub fn should_qwen_prefill_skip_unused_swiglu_compile(model_family: &str, seq: i32) -> bool {
    should_qwen_prefill_skip_unused_swiglu_compile_for(
        qwen_prefill_skip_unused_swiglu_compile_enabled(),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_qwen_prefill_skip_unused_swiglu_compile`].
pub fn should_qwen_prefill_skip_unused_swiglu_compile_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
) -> bool {
    enabled
        && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

/// Whether Qwen prefill should use native offset causal SDPA.
///
/// The explicit env opt-in stays available on every host. On M5+ with
/// macOS 26.2+ the same predicate is also armed by [`nax_attention_enabled`]
/// so offset chunks (`seq >= 1024`) skip the O(seq×key) array mask and let
/// MLX pick NAX fused causal SDPA. Gemma and sliding-window layers are
/// unchanged.
pub fn should_qwen_prefill_native_offset_causal(model_family: &str, seq: i32) -> bool {
    should_qwen_prefill_native_offset_causal_for(
        qwen_prefill_native_offset_causal_enabled() || nax_attention_enabled(),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_qwen_prefill_native_offset_causal`].
pub fn should_qwen_prefill_native_offset_causal_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
) -> bool {
    enabled
        && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

/// Whether Qwen prefill should dequant embeddings directly to BF16.
pub fn should_qwen_prefill_bf16_embed_dequant(model_family: &str, seq: i32) -> bool {
    should_qwen_prefill_bf16_embed_dequant_for(
        qwen_prefill_bf16_embed_dequant_enabled(),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_qwen_prefill_bf16_embed_dequant`].
pub fn should_qwen_prefill_bf16_embed_dequant_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
) -> bool {
    enabled
        && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

/// Whether Qwen prefill should skip the unused f32 SDPA upcast.
pub fn should_qwen_prefill_skip_unused_f32_sdpa(model_family: &str, seq: i32) -> bool {
    should_qwen_prefill_skip_unused_f32_sdpa_for(
        qwen_prefill_skip_unused_f32_sdpa_enabled(),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_qwen_prefill_skip_unused_f32_sdpa`].
///
/// Contract-prefill gate is `seq >= 128`, mirroring
/// [`should_gemma4_prefill_skip_unused_f32_sdpa_for`]: the f32 upcast exists
/// for short teacher-forced MTP verify (`seq` 2..=8), and decode `seq == 1`
/// never arms the guard. The former `seq >= 1024` gate left every p128/p512
/// full-attention layer paying a Q/K/V f32 round-trip that mlxcel never pays
/// (PRD-M5-FLEET-AX-VS-MLXCEL short-prefill miss).
pub fn should_qwen_prefill_skip_unused_f32_sdpa_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
) -> bool {
    enabled
        && seq >= 128
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next" | "qwen3_vl_moe" | "qwen3_vl"
        )
}

/// Whether Qwen prefill should skip the unused embed-id clip.
pub fn should_qwen_prefill_skip_unused_embed_clip(model_family: &str, seq: i32) -> bool {
    should_qwen_prefill_skip_unused_embed_clip_for(
        qwen_prefill_skip_unused_embed_clip_enabled(),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_qwen_prefill_skip_unused_embed_clip`].
pub fn should_qwen_prefill_skip_unused_embed_clip_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
) -> bool {
    enabled
        && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_ASYNC_EMBED` — after token embed (+ optional
    /// `hidden_states_scale`), `async_eval([hidden])` at `seq >= 1024` so
    /// the GPU starts the gather while the host builds the first-layer
    /// graph. Not skip-astype (closed wash), not pipeline-block
    /// `async_eval(hidden)` after layers (closed), not post-qmm async
    /// (gate-up / packed / LA / down / SDPA / GD, all closed).
    ///
    /// **Default: OFF**. Remasured binary `a3d8e261…` (2026-08-14): 3b
    /// 1.029260 / 3d 1.054017 wash (0.997× q2only). Async submit of the
    /// embed gather does not cut compute-bound qmm.
    qwen_prefill_async_embed_enabled,
    "AX_MLX_QWEN_PREFILL_ASYNC_EMBED"
);

/// Whether Qwen prefill should async-submit the embedding gather.
pub fn should_qwen_prefill_async_embed(model_family: &str, seq: i32) -> bool {
    should_qwen_prefill_async_embed_for(qwen_prefill_async_embed_enabled(), model_family, seq)
}

/// Pure helper for [`should_qwen_prefill_async_embed`].
pub fn should_qwen_prefill_async_embed_for(enabled: bool, model_family: &str, seq: i32) -> bool {
    enabled
        && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

/// Pure helper for [`should_qwen_la_norm_qkvz_fuse`].
pub fn should_qwen_la_norm_qkvz_fuse_for(enabled: bool, model_family: &str, seq: i32) -> bool {
    enabled
        && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

/// Pure helper for [`should_qwen_prefill_dequant_dense`].
pub fn should_qwen_prefill_dequant_dense_for(enabled: bool, model_family: &str, seq: i32) -> bool {
    enabled
        && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

/// Pure helper for [`should_qwen_prefill_split_packed`].
pub fn should_qwen_prefill_split_packed_for(enabled: bool, model_family: &str, seq: i32) -> bool {
    enabled
        && seq >= QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "qwen3_5" | "qwen3_next"
        )
}

/// Whether Qwen prefill should merge matching-bit QKVZ/BA into one qmm.
pub fn should_qwen_la_fused_qkvz_ba_qmm(seq: i32, same_quant: bool) -> bool {
    should_qwen_la_fused_qkvz_ba_qmm_for(qwen_la_fused_qkvz_ba_qmm_enabled(), seq, same_quant)
}

/// Pure helper for [`should_qwen_la_fused_qkvz_ba_qmm`].
pub fn should_qwen_la_fused_qkvz_ba_qmm_for(enabled: bool, seq: i32, same_quant: bool) -> bool {
    enabled && seq > 1 && same_quant
}

/// Whether GatedDelta prefill should materialize contiguous QKV/AB.
pub fn should_qwen_gated_delta_prefill_contiguous(seq: i32) -> bool {
    should_qwen_gated_delta_prefill_contiguous_for(
        qwen_gated_delta_prefill_contiguous_enabled(),
        seq,
    )
}

/// Pure helper for [`should_qwen_gated_delta_prefill_contiguous`].
pub fn should_qwen_gated_delta_prefill_contiguous_for(enabled: bool, seq: i32) -> bool {
    enabled && seq > 1
}

env_flag!(
    /// `AX_MLX_GEMMA_DIRECT_CPP_QK_NORM_ROPE` — opt-in Gemma-family direct C++
    /// route for standard-attention Q/K `as_strided -> rms_norm -> rope`
    /// (including proportional-RoPE full-attention layers that pass freqs).
    ///
    /// **Default: OFF**. Residual vs mlxcel `compiled_q_path_proportional`:
    /// AX already had the C++ composite. Enabling it for gemma* pure 13.8k
    /// prefill on mbp-m5 (2026-07-24) measured ~+1.6% cold wall vs portable
    /// multi-FFI path (pure-gemma-qkrope-ab). Keep opt-in for decode-focused
    /// A/B; pure prefill thr path remains portable.
    ///
    /// Pair with `AX_MLX_COMPILED_QK_NORM_ROPE=1` (mlx-sys C++) to also wrap
    /// the freqs path in `mx::compile` (mlxcel `compiled_q_path_proportional`
    /// parity). Both flags stay opt-in until pure A/B shows a stable cut.
    gemma_direct_cpp_qk_norm_rope_enabled,
    "AX_MLX_GEMMA_DIRECT_CPP_QK_NORM_ROPE"
);

env_flag!(
    /// `AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_INPUTS` — opt-in direct C++ route
    /// for Qwen linear-attention packed QKVZ/BA projection staging. This
    /// global flag force-enables the route for any compatible caller shape.
    ///
    /// **Default: OFF**. The Qwen3.5/Qwen3Next production default is controlled
    /// by `AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_INPUTS`; keep this separate
    /// opt-in surface for A/B and non-Qwen compatibility probes.
    direct_cpp_linear_attention_inputs_enabled,
    "AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_INPUTS"
);

env_flag_default_on!(
    /// `AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_INPUTS` — default Qwen
    /// linear-attention packed QKVZ/BA projection staging direct C++ route.
    ///
    /// **Default: ON for Qwen3.5/Qwen3Next only** (kill-switch via
    /// `AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_INPUTS=0`). The route skips
    /// per-op MLX FFI dispatches for packed projection, reshape, slice, and
    /// concat staging before the Qwen gated-delta block. It is family-scoped
    /// because the verified win is on Qwen linear-attention decode when paired
    /// with the post-input route.
    qwen_direct_cpp_linear_attention_inputs_enabled,
    "AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_INPUTS"
);

env_flag!(
    /// `AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT` — opt-in direct C++
    /// route for the Qwen linear-attention post-input block. This global flag
    /// force-enables the route for any compatible caller shape.
    ///
    /// **Default: OFF**. The Qwen3.5/Qwen3Next production default is controlled
    /// by `AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT`; keep this
    /// separate opt-in surface for A/B and non-Qwen compatibility probes.
    direct_cpp_linear_attention_post_input_enabled,
    "AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT"
);

env_flag_default_on!(
    /// `AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT` — default Qwen
    /// linear-attention post-input direct C++ route.
    ///
    /// **Default: ON for Qwen3.5/Qwen3Next only** (kill-switch via
    /// `AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT=0`). The route
    /// fuses conv1d (with cached-state carry), SiLU, last-dim split into q/k/v,
    /// head-major reshape, per-head RMSNorm on q and k, and scale constants
    /// into one Rust→C++ round-trip while leaving the gated-delta Metal kernel
    /// and all non-Qwen families on their existing paths.
    qwen_direct_cpp_linear_attention_post_input_enabled,
    "AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT"
);

env_flag_default_on!(
    /// `AX_MLX_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL` — route Qwen
    /// linear-attention single-token decode post-input work through one Metal
    /// kernel.
    ///
    /// **Default: ON** (kill-switch via
    /// `AX_MLX_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL=0`).
    ///
    /// This is narrower than the direct C++ post-input route: it only engages
    /// for `seq=1`, an existing cached conv state, equal Q/K/V head dims, and a
    /// power-of-two head dim. Unsupported shapes fall back to the existing
    /// C++/portable post-input paths.
    qwen_linear_attention_decode_post_input_metal_enabled,
    "AX_MLX_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL"
);

env_flag!(
    /// `AX_MLX_QWEN_LINEAR_ATTENTION_PREFILL_POST_INPUT_METAL` — route Qwen
    /// linear-attention multi-token post-input (conv + SiLU + split + QK-norm)
    /// through the existing Metal kernel (`Seq` is already a template).
    ///
    /// **Default: OFF**. AXQ remasured p2048 874.96 vs 891.02 (2026-08-13).
    /// Decode stays on the seq<=4 default-ON flag.
    qwen_linear_attention_prefill_post_input_metal_enabled,
    "AX_MLX_QWEN_LINEAR_ATTENTION_PREFILL_POST_INPUT_METAL"
);

env_flag_default_on!(
    /// `AX_MLX_QWEN_GATED_DELTA_DECODE_METAL` — route Qwen single-token
    /// GatedDelta recurrent updates through the decode-specialized Metal
    /// kernel.
    ///
    /// **Default: ON** for A/B until the Qwen decode benchmark decides whether
    /// this specialization is net-positive with the post-input Metal route.
    qwen_gated_delta_decode_metal_enabled,
    "AX_MLX_QWEN_GATED_DELTA_DECODE_METAL"
);

env_flag!(
    /// `AX_MLX_QWEN_GATED_DELTA_PREFILL_STREAMING` — route long multi-token
    /// GatedDelta prefill (seq > 512) through a streaming Metal kernel that
    /// fuses g/beta each step without a CacheCapacity-sized TG array.
    ///
    /// **Default: OFF** (opt-in via
    /// `AX_MLX_QWEN_GATED_DELTA_PREFILL_STREAMING=1`).
    ///
    /// p128/p512 stay on the 512 TG kernel either way. Default-on streaming
    /// (one 2048 forward) was remasured on df-macbookpro-m5 AXQ p2048 at
    /// 871 vs 891 tok/s for two 1024 TG chunks (2026-08-13). Keep 1024.
    qwen_gated_delta_prefill_streaming_enabled,
    "AX_MLX_QWEN_GATED_DELTA_PREFILL_STREAMING"
);

env_flag!(
    /// `AX_MLX_QWEN_GATED_DELTA_PREFILL_TILE_512` — tile multi-token GatedDelta
    /// prefill at the 512 TG specialization when seq > 512.
    ///
    /// **Default: OFF**. Alone on two 1024 chunks: 892.80 vs 891.02.
    /// Combined with one 2048 FFN: 889.96 vs 891.02 (2026-08-13).
    qwen_gated_delta_prefill_tile_512_enabled,
    "AX_MLX_QWEN_GATED_DELTA_PREFILL_TILE_512"
);

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_SINGLE_2048` — raise the linear-attention runner
    /// chunk cap to 2048 so p2048 is one FFN pass (M=2048).
    ///
    /// **Default: OFF**. With tile-512 remasured p2048 889.96 vs 891.02
    /// (2026-08-13). Same class as 2048+tile-1024 887. Keep two 1024 FFN
    /// stacks. Streaming stays default-OFF.
    qwen_prefill_single_2048_enabled,
    "AX_MLX_QWEN_PREFILL_SINGLE_2048"
);

env_flag_default_on!(
    /// `AX_MLX_PACK_LINEAR_ATTENTION_PROJECTIONS` — load-time packing for Qwen
    /// linear-attention projections.
    ///
    /// **Default: ON** (kill-switch via
    /// `AX_MLX_PACK_LINEAR_ATTENTION_PROJECTIONS=0`).
    ///
    /// Materializes split QKV/Z/A/B projections into packed QKVZ/BA projections
    /// when the artifact layout and quantization metadata are compatible. This
    /// reduces per-layer projection dispatch count on Qwen 3.6 dense and MoE
    /// linear-attention layers while preserving a fail-closed split fallback for
    /// incompatible shapes.
    linear_attention_projection_packing_enabled,
    "AX_MLX_PACK_LINEAR_ATTENTION_PROJECTIONS"
);

env_flag_default_on!(
    /// `AX_MLX_LINEAR_ATTENTION_RMS_NORM_GATE_METAL` — route Qwen
    /// linear-attention post-RMSNorm gating through a custom MLX Metal
    /// elementwise kernel.
    ///
    /// **Default: ON** (kill-switch via
    /// `AX_MLX_LINEAR_ATTENTION_RMS_NORM_GATE_METAL=0`).
    ///
    /// Keeps RMSNorm itself on the normal MLX path, then fuses the following
    /// `silu(gate.float32) * normed.float32 -> hidden dtype` chain into one
    /// lazy MLX graph node. Unsupported shapes/dtypes fall back to the existing
    /// MLX operation chain.
    linear_attention_rms_norm_gate_metal_enabled,
    "AX_MLX_LINEAR_ATTENTION_RMS_NORM_GATE_METAL"
);

env_flag_default_on!(
    /// `AX_MLX_MOE_FUSE_SHARED_EXPERT_ADD` — fuse the shared-expert add
    /// into the Qwen3 MoE weighted-sum Metal kernel.
    ///
    /// **Default: ON** (kill-switch via
    /// `AX_MLX_MOE_FUSE_SHARED_EXPERT_ADD=0`).
    ///
    /// When the shared expert is present and the weighted-sum Metal kernel
    /// is eligible, the shared-expert output is added inside the same kernel
    /// that combines the top-k expert outputs — eliminating one `add`
    /// dispatch per MoE layer. **Decode-only (seq==1):** at prefill the
    /// weighted-sum is bandwidth-bound, where the fused kernel's extra input
    /// read costs more than the dispatch it saves, so prefill falls back to
    /// the separate `add`. Also falls back when the kernel is ineligible
    /// (dtype or shape mismatch) or the flag is off.
    moe_fuse_shared_expert_add_enabled,
    "AX_MLX_MOE_FUSE_SHARED_EXPERT_ADD"
);

env_flag_default_on!(
    /// `AX_MLX_QWEN3_MOE_NARROW_SOFTMAX` — narrow softmax for the Qwen3
    /// MoE router. The router does argpartition on raw logits to find top-k
    /// indices, then applies `softmax_precise` only to the selected top-k
    /// subset (matching the Gemma4 router pattern). This eliminates the
    /// full-width softmax over all experts (128–512), reducing per-layer
    /// router overhead on decode.
    ///
    /// **Default: ON** (kill-switch via `AX_MLX_QWEN3_MOE_NARROW_SOFTMAX=0`).
    /// Promoted from opt-in after validation confirmed token-for-token
    /// equivalence with the `precise=True` reference path.
    qwen3_moe_narrow_softmax_enabled,
    "AX_MLX_QWEN3_MOE_NARROW_SOFTMAX"
);

env_flag!(
    /// `AX_MLX_MOE_PROFILE` — family-neutral MoE sub-stage profiling.
    ///
    /// **Default: OFF** (opt-in diagnostic). When enabled, the MoE expert
    /// forward path records per-sub-stage wall times (router, gate_up,
    /// activation, down, weighted_sum, shared_expert) into a dedicated
    /// `MoeProfileSnapshot`. Unlike `AX_MLX_DECODE_PROFILE` which forces
    /// blocking `eval()` at every stage and disables decode pipelining,
    /// this flag records lightweight wall-clock deltas without forcing
    /// evaluation barriers. Use for MoE-specific hotspot diagnosis.
    moe_profile_enabled,
    "AX_MLX_MOE_PROFILE"
);

env_flag!(
    /// `AX_MLX_MOE_LAYER_COMPILE` — enable per-layer compiled MoE decode
    /// closure.
    ///
    /// **Default: OFF** (opt-in via `AX_MLX_MOE_LAYER_COMPILE=1`).
    /// Each MoE layer's decode forward path is wrapped in an `MlxClosure`
    /// compiled via `mlx_compile` with `shapeless=true`. Only engages for
    /// `seq == 1` (decode). Falls back to the uncompiled path on
    /// compilation or apply failure, permanently per layer.
    ///
    /// History. Default-on originally; reverted 2026-06-19 (`19120c10`)
    /// after long-running-process crashes. The real abort vector — a Rust
    /// panic from an in-body op-status failure unwinding across the C++
    /// trampoline, fatal under the release `panic = "abort"` profile — is
    /// now closed by construction (poison propagation,
    /// `mlx_sys::error::ClosureBodyGuard`), so opting in is safe: failures
    /// degrade to per-layer imperative fallbacks. Briefly re-promoted
    /// default-on on 2026-07-17, then reverted the same day when the
    /// review found the promotion evidence invalid: on gather-routed MoE
    /// (Qwen3-Next class) MLX cannot shapeless-compile the closure at all
    /// ("[Primitive::output_shapes] GatherQMM cannot infer output shapes"
    /// — every layer falls back permanently, one warn each), so the
    /// measured +1.6% was pair noise on a path that never engaged; on
    /// Gemma-4-26B-A4B the dual-path closure does engage and measured a
    /// neutral 1.003 (3 interleaved pairs, parity clean). No family shows
    /// a ≥1.01 win, so per ADR-003 D5 the flag stays opt-in. Upstream
    /// follow-up candidate: GatherQMM `output_shapes` support in MLX
    /// compile would make this promotable on MoE.
    moe_layer_compile_enabled,
    "AX_MLX_MOE_LAYER_COMPILE"
);

/// `AX_MLX_DENSE_FFN_COMPILE` — enable per-layer compiled dense FFN
/// decode closure.
///
/// **Default: ON** (kill-switch via `AX_MLX_DENSE_FFN_COMPILE=0`).
/// Default-on for Qwen 3.6 decode throughput. Set
/// `AX_MLX_DENSE_FFN_COMPILE=0` to disable if stream-registry issues
/// are observed. Each dense FFN layer's decode forward is wrapped in an
/// `MlxClosure` compiled via `mlx_compile` with `shapeless=true`,
/// collapsing the gate_up projection + split + SwiGLU activation + down
/// projection + optional post-norm into a single compiled graph. Only
/// engages for `seq == 1` (decode) and SwiGLU activation families
/// (GEGLU's `gelu_approx` tree is known to abort under MLX compilation).
/// Falls back to the uncompiled path on compilation failure.
pub fn dense_ffn_compile_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let value = *CACHED.get_or_init(|| parse_bool_env_default_on("AX_MLX_DENSE_FFN_COMPILE"));
    if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::info!(
            target = "ax_engine_mlx",
            enabled = value,
            "AX_MLX_DENSE_FFN_COMPILE resolved (set =0 to disable)"
        );
    }
    value
}

env_flag_default_on!(
    /// `AX_MLX_DENSE_FFN_COMPILE_PREFILL` — compile dense packed FFN for
    /// multi-token prefill with a **per-shape** cache.
    ///
    /// **Default: ON** (kill-switch via `AX_MLX_DENSE_FFN_COMPILE_PREFILL=0`).
    ///
    /// Unlike decode (`shapeless=true`, seq always 1), prefill compiles with
    /// `shapeless=false` and keys the cache by leading element count so each
    /// prompt length gets a correct fixed-shape graph. Required because MLX
    /// shapeless compile of quantized matmul is not shape-polymorphic
    /// (see mlp unit test). Requires packed gate/up. SwiGLU uses `silu_mul`;
    /// GEGLU (Gemma) uses the Metal-backed `geglu` helper inside the closure.
    ///
    /// Short prompts skip compile when `leading_elements` is below
    /// [`DENSE_FFN_PREFILL_COMPILE_MIN_LEADING`] so compile cost is not paid
    /// on 128-token microbenches (2026-07-12 short-prompt regression under
    /// unconditional default-on). Long Gemma prompts (512+) amortize compile.
    /// Qwen packed prefill stays imperative because a paired 2w/5m check found
    /// the fixed-shape compile slower at the 512-token boundary.
    dense_ffn_compile_prefill_enabled,
    "AX_MLX_DENSE_FFN_COMPILE_PREFILL"
);

env_flag!(
    /// `AX_MLX_QWEN_COMPILED_DUAL_GATE_UP` — compile the two split affine
    /// gate/up qmms on Qwen multi-token prefill (`mx::compile`, shape-specific).
    ///
    /// **Default: OFF**. AXQ p2048 remasured 890.96 vs 891.02 for two
    /// imperative qw (2026-08-13). Gemma stays on `AX_MLX_COMPILED_DUAL_GATE_UP`
    /// (also default OFF). Dual-stream GPU overlap remains default-OFF.
    qwen_compiled_dual_gate_up_enabled,
    "AX_MLX_QWEN_COMPILED_DUAL_GATE_UP"
);

env_flag!(
    /// `AX_MLX_QWEN_SPLIT_FFN_PREFILL_COMPILE` — shape-compile the Qwen
    /// **split** FFN (gate + up + SwiGLU + down) for multi-token prefill.
    ///
    /// **Default: OFF**. AXQ remasured p2048 888.77 vs 891.02 for imperative
    /// split qw (2026-08-13). Packed Qwen prefill compile stays forbidden.
    qwen_split_ffn_prefill_compile_enabled,
    "AX_MLX_QWEN_SPLIT_FFN_PREFILL_COMPILE"
);

/// Minimum `batch * seq` before Qwen split FFN prefill compile engages.
/// 128 covers every formal 27B contract shape; shorter prompts stay
/// imperative so compile tax is not paid on decode-adjacent microbenches.
pub const QWEN_SPLIT_FFN_PREFILL_COMPILE_MIN_LEADING: i64 = 128;

/// Minimum leading element count (product of non-last dims) before dense FFN
/// prefill compile engages. `batch * seq` for standard `[B,S,H]` layouts;
/// 256 covers mid-length prompts; README 128-token rows stay uncompiled
/// so short-prompt microbenches avoid compile tax.
pub const DENSE_FFN_PREFILL_COMPILE_MIN_LEADING: i64 = 256;

/// Gemma 4 contract p128 leading count. Packed prefill compile normally
/// waits for [`DENSE_FFN_PREFILL_COMPILE_MIN_LEADING`]; this shape is the
/// measured miss (`df-macbookpro-m5` skip-f32+fused residual).
pub const GEMMA4_PACKED_FFN_COMPILE_P128_LEADING: i64 = 128;

env_flag!(
    /// `AX_MLX_GEMMA4_PACKED_FFN_COMPILE_P128` — at contract p128, take the
    /// packed gate/up path and shape-compile the dense FFN (Metal GEGLU
    /// inside the closure). Split gate/up stays on for p512/p2048.
    ///
    /// **Default: OFF**. Remasured on `df-macbookpro-m5` (2026-08-15,
    /// `gemma4-axq-v7-packffn` + repeat): 12B p128 656.50/651.95 vs fused
    /// 651.57/659.48 (1.008× / 1.001×). 26B p128 592.95/592.23 vs fused
    /// 604.22/599.45 (0.981× / 0.980×). No 1.10×; 26B dips vs fused.
    /// Keep opt-in only.
    gemma4_packed_ffn_compile_p128_enabled,
    "AX_MLX_GEMMA4_PACKED_FFN_COMPILE_P128"
);

/// Whether Gemma 4 contract p128 should compile packed dense FFN.
pub fn should_gemma4_packed_ffn_compile_p128(model_family: &str, seq: i32) -> bool {
    should_gemma4_packed_ffn_compile_p128_for(
        gemma4_packed_ffn_compile_p128_enabled(),
        model_family,
        seq,
    )
}

/// Pure helper for [`should_gemma4_packed_ffn_compile_p128`].
pub fn should_gemma4_packed_ffn_compile_p128_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
) -> bool {
    enabled
        && seq == 128
        && matches!(
            model_family.to_ascii_lowercase().as_str(),
            "gemma4" | "gemma4_unified"
        )
}

env_flag_default_on!(
    /// `AX_MLX_AUTO_BUFFER_CAPS` — auto-raise MLX Metal command-buffer caps
    /// for many-large-tensor (MoE-class) checkpoints.
    ///
    /// **Default: ON** (kill-switch via `AX_MLX_AUTO_BUFFER_CAPS=0`).
    ///
    /// MLX splits a Metal command buffer once accumulated input bytes exceed
    /// `MLX_MAX_MB_PER_BUFFER` (default 40–50 MB), counting each expert
    /// stack at its full size; on MoE checkpoints every layer splits and the
    /// scheduler backpressure turns `async_eval` into a barrier (zero
    /// host/GPU overlap). When the loaded checkpoint has at least
    /// [`crate::weights::BUFFER_CAP_MIN_BIG_TENSORS`] tensors above the MLX
    /// default cap and the user has not set the MLX variables themselves,
    /// the loader raises them to 1024 MB / 1000 ops before the first GPU op.
    /// Explicit `MLX_MAX_*_PER_BUFFER` values remain authoritative for every
    /// family. The raise applies only to families with positive server-path
    /// evidence (`qwen3_next`/Coder-Next: +22–25% decode, bit-identical);
    /// Gemma, unlimited-OCR, and `qwen3_5` (Qwen3.5/3.6 hybrids) retain MLX
    /// defaults — on the server path the giant command buffers cost those
    /// families prefill throughput and (for `qwen3_5` MoE) a one-way
    /// per-request prefill degradation, with no decode win materializing
    /// outside decode-trace. On M5+ with macOS 26.2+ the raise is further
    /// restricted to `qwen3_next` (the only family with a measured M5
    /// decode win); other eligible families keep MLX defaults so giant
    /// command buffers do not stall NAX. Pre-M5 hosts still raise
    /// **optimistically on first process decision** for eligible families
    /// (including dense-first loads) so multi-model servers that load
    /// Llama then MoE still get the MoE win.
    /// Evidence and exclusion A/Bs:
    /// `docs/performance/gather-qmm-async-serialization.md`.
    auto_buffer_caps_enabled,
    "AX_MLX_AUTO_BUFFER_CAPS"
);

env_flag!(
    /// `AX_MLX_MOE_ROUTER_FUSED_METAL` — enable fused MoE router Metal
    /// kernel for decode.
    ///
    /// **Default: OFF** (opt-in). When the model uses the Qwen3 narrow-softmax
    /// router path and the logits are castable to f32, the post-matmul router
    /// chain (argpartition + take_along_axis + softmax + renormalize) is
    /// collapsed into a single Metal kernel dispatch. Decode-only (seq==1).
    /// Falls back to the MLX op path when ineligible.
    ///
    /// **Not promoted (2026-07-16 A/B, Qwen3-Coder-Next-4bit,
    /// `scripts/ab_moe_router_fused.py`, 5×256-step interleaved reps):**
    /// route reach was 100% (attempts=hits=12768/run, zero fallbacks) but
    /// decode was 0.9949x baseline (median 69.86 vs 70.22 tok/s) and greedy
    /// parity is broken: the kernel returns f32 softmax weights while the
    /// fallback's subset-softmax stays bf16, perturbing every MoE layer
    /// output; top-k boundary selections flip from the first decode
    /// forward's layer 1 and the token stream diverges deterministically.
    /// Per the fused-downproj precedent, "more accurate but different" is
    /// not shippable. Raw artifacts:
    /// `benchmarks/results/inference/mlx-inference/2026-07-16-qwen3-coder-next-router-fused-ab/`.
    moe_router_fused_metal_enabled,
    "AX_MLX_MOE_ROUTER_FUSED_METAL"
);

env_flag!(
    /// `AX_MLX_LINEAR_ATTENTION_WHOLE_LAYER_METAL` — enable whole-layer
    /// Metal kernel for linear-attention decode.
    ///
    /// **Default: OFF** (opt-in). When eligible (seq==1 decode), runs the
    /// compositional whole-layer linear-attention Metal path (gated-delta +
    /// Metal gate helpers + projections). Decode-only. Falls back when
    /// ineligible. Single-dispatch mega-kernel remains residual.
    linear_attention_whole_layer_metal_enabled,
    "AX_MLX_LINEAR_ATTENTION_WHOLE_LAYER_METAL"
);

env_flag!(
    /// `AX_MLX_MOE_DEEP_EXPERT_BLOCK_METAL` — enable deep expert-block
    /// fusion Metal path for MoE decode.
    ///
    /// **Default: OFF** (opt-in). Compositional path: gather_qmm gate_up →
    /// Metal fused activation/unsort → gather_qmm down → Metal weighted-sum.
    /// Decode-only batch=1. Falls back when ineligible. Single-dispatch
    /// 4-bit mega-kernel remains residual.
    moe_deep_expert_block_metal_enabled,
    "AX_MLX_MOE_DEEP_EXPERT_BLOCK_METAL"
);

env_flag!(
    /// `AX_MLX_MOE_FUSED_EXPERT_BLOCK` — enable fused MoE expert block
    /// Metal kernel for decode.
    ///
    /// **Default: OFF** (opt-in). When the model is eligible (seq==1,
    /// compatible dtype, unsorted gather), the activation + squeeze +
    /// unsort chain is routed through a fused Metal kernel, reducing
    /// dispatch count per MoE layer. Falls back to the standard dispatch
    /// sequence when ineligible.
    moe_fused_expert_block_enabled,
    "AX_MLX_MOE_FUSED_EXPERT_BLOCK"
);

env_flag!(
    /// `AX_MLX_MOE_EXPERT_PARALLEL` — enable expert-parallel Metal dispatch
    /// for MoE prefill.
    ///
    /// **Default: OFF** (opt-in). When enabled and the prefill sequence
    /// length is > 1, expert tokens are binned per-expert and the load-
    /// balance is checked. Falls back to sequential `gather_qmm` when
    /// the token distribution is highly skewed (max_bin > 2x mean_bin)
    /// or the parallel kernel is not yet available.
    moe_expert_parallel_enabled,
    "AX_MLX_MOE_EXPERT_PARALLEL"
);

env_flag_default_on!(
    /// `AX_MLX_MOE_SWIGLU_PACKED_METAL` — route the MoE expert SwiGLU
    /// activation through the same packed Metal kernel used by the dense
    /// FFN path.
    ///
    /// **Default: ON** (kill-switch via
    /// `AX_MLX_MOE_SWIGLU_PACKED_METAL=0`).
    ///
    /// When the MoE expert gate_up projection is packed (the common Qwen3
    /// path), the gather_qmm output is passed directly to the packed
    /// `ax_qwen_packed_swiglu_v1` kernel, which fuses the last-dim split,
    /// SiLU, and multiply into one dispatch instead of slice + slice +
    /// silu_mul. **Decode-only (seq==1):** at prefill the tensor is large
    /// and bandwidth-bound, where the separate slice+silu_mul ops are
    /// faster than the single packed dispatch, so prefill uses the split
    /// path. Also falls back when the kernel is ineligible or the flag is
    /// off.
    moe_swiglu_packed_metal_enabled,
    "AX_MLX_MOE_SWIGLU_PACKED_METAL"
);

env_flag_default_on!(
    /// `AX_MLX_MOE_GEGLU_PACKED_METAL` — route the MoE expert GEGLU
    /// activation (Gemma4 MoE) through the packed Metal kernel used by the
    /// dense FFN path.
    ///
    /// **Default: ON** (kill-switch via `AX_MLX_MOE_GEGLU_PACKED_METAL=0`).
    ///
    /// When the MoE expert gate_up projection is packed (Gemma4 MoE), the
    /// gather_qmm output is passed directly to `packed_geglu_metal_impl`,
    /// which fuses the last-dim split, GELU-approx, and multiply into one
    /// dispatch instead of slice + slice + gelu_approx_mul. Saves 2 MLX
    /// graph nodes per MoE layer per decode step (~48 nodes on Gemma4 27B
    /// with 24 MoE layers). Engages for decode and for **moderate prefill**
    /// (`seq <= MOE_PACKED_GEGLU_PREFILL_MAX_SEQ`); very long prefill keeps
    /// the split path where separate ops are bandwidth-friendlier.
    moe_geglu_packed_metal_enabled,
    "AX_MLX_MOE_GEGLU_PACKED_METAL"
);

/// Prefill seq ceiling for MoE packed GeGLU Metal. Above this, fall back to
/// split activation (large gather tensors become bandwidth-bound).
pub const MOE_PACKED_GEGLU_PREFILL_MAX_SEQ: usize = 512;

env_flag_default_on!(
    /// `AX_MLX_F32_PACK_BF16_NORMALIZE` — cast stray F32 floating tensors
    /// (norms, quantization scales/biases, embeddings) to BF16 at load for
    /// quantized qwen3_5-class packs. Holo3 35B 6bit ships them F32,
    /// promoting the whole activation stream to f32: M5 Max measurements
    /// were ~25% slower at p2048 prefill and ~5% slower at decode vs the
    /// BF16 sibling (Ornith 6bit, same 35B-A3B graph). Not a requant —
    /// quantized integer payloads are untouched, and dense F32 checkpoints
    /// remain F32.
    ///
    /// **Default: ON** (kill switch `=0`).
    f32_pack_bf16_normalize_enabled,
    "AX_MLX_F32_PACK_BF16_NORMALIZE"
);

/// Prefill seq ceiling for MoE packed SwiGLU Metal (Qwen3 MoE experts).
///
/// The packed kernel is decode-only ("prefill is bandwidth-bound; split
/// slice+silu_mul is faster"). M5 A/B on Ornith 35B 6bit (2026-08-18,
/// two rounds, reps 5): band=512 measured +1.2%/-1.7% at p128 and
/// +0.6%/+0.3% at p512 — a wash within session noise, so the shipped
/// decode-only behavior stays. `AX_MLX_MOE_SWIGLU_PREFILL_MAX_SEQ=N`
/// remains for future-host A/Bs; default 0.
pub fn moe_packed_swiglu_prefill_max_seq() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED
        .get_or_init(|| parse_positive_usize_env("AX_MLX_MOE_SWIGLU_PREFILL_MAX_SEQ").unwrap_or(0))
}

/// Seq ceiling for the fused MoE shared-expert weighted-sum kernel.
///
/// Shipped threshold is 64 ("beyond this the weighted-sum is
/// bandwidth-bound and the fused kernel's extra input read costs more").
/// `AX_MLX_MOE_SHARED_FUSION_SEQ_THRESHOLD=N` overrides for A/B on the
/// 35B-A3B class contract shapes.
pub fn moe_shared_fusion_seq_threshold(default_threshold: usize) -> usize {
    static CACHED: OnceLock<Option<usize>> = OnceLock::new();
    CACHED
        .get_or_init(|| parse_positive_usize_env("AX_MLX_MOE_SHARED_FUSION_SEQ_THRESHOLD"))
        .unwrap_or(default_threshold)
}

/// Tuning override for the MLA prefill chunk size. Smaller chunks let
/// cold and warm-extend prefill paths produce the same SDPA Q/K shape
/// sequence over the same absolute positions, avoiding the reproduced
/// GLM-4.7-Flash warm_extend fp-drift diagnosed by
/// `verify_prefix_reuse_equivalence.py --mode warm_extend`. The canonical
/// default-path harness passes 5/5 with a real prefix-cache hit after this
/// change. `MlxRunner::from_artifacts` defaults to 16 for MLA models when
/// this env is unset. Set
/// `AX_MLX_MLA_PREFILL_CHUNK=N` to override (larger N trades correctness
/// margin for prefill throughput). Returns `None` when unset/invalid;
/// callers supply their own MLA default.
pub fn mla_prefill_chunk_override() -> Option<usize> {
    static CACHED: OnceLock<Option<usize>> = OnceLock::new();
    *CACHED.get_or_init(|| parse_positive_usize_env("AX_MLX_MLA_PREFILL_CHUNK"))
}

/// Default `prefill_chunk` value applied when a model has MLA layers
/// and `AX_MLX_MLA_PREFILL_CHUNK` is unset. Sized to the prefix-cache
/// block_size so the chunked_prefill loop produces the same SDPA shape
/// sequence whether the prompt was processed cold or restored from a
/// snapshot and extended.
pub const MLA_DEFAULT_PREFILL_CHUNK: usize = 16;

/// Resolve the effective prefill chunk before any caller performs prefill
/// work. MLA models use the MLA-specific default/override; other models keep
/// the caller-selected value. The result is always at least one token so the
/// chunked-prefill loop cannot receive a zero-sized chunk.
pub fn resolve_prefill_chunk(
    has_mla_attention: bool,
    requested_prefill_chunk: usize,
    mla_override: Option<usize>,
) -> usize {
    let resolved = if has_mla_attention {
        mla_override.unwrap_or(MLA_DEFAULT_PREFILL_CHUNK)
    } else {
        requested_prefill_chunk
    };
    resolved.max(1)
}

/// Optional large cold-prefill chunk for MLA (`AX_MLX_MLA_COLD_PREFILL_CHUNK`).
///
/// Default is unset: MLA cold prefill matches the warm-extend chunk (R2) so
/// snapshot producers and cold full-prefix runs share one SDPA shape trail.
/// Setting this opt-in restores the historical dual-path cold throughput
/// experiment and can re-open warm_extend token drift — use only with the
/// equivalence harness.
pub fn mla_cold_prefill_chunk_override() -> Option<usize> {
    static CACHED: OnceLock<Option<usize>> = OnceLock::new();
    *CACHED.get_or_init(|| parse_positive_usize_env("AX_MLX_MLA_COLD_PREFILL_CHUNK"))
}

/// Resolve MLA cold-prefill chunk size.
///
/// - Default (R2): same as warm-extend `warm_prefill_chunk` so store + cold
///   baselines stay token-exact under warm_extend.
/// - Opt-in large cold: `AX_MLX_MLA_COLD_PREFILL_CHUNK=N` (throughput only).
pub fn resolve_mla_cold_prefill_chunk(
    warm_prefill_chunk: usize,
    cold_override: Option<usize>,
) -> usize {
    cold_override.unwrap_or(warm_prefill_chunk).max(1)
}

/// Whether a prefill is cold (empty cache) or warm-extend (non-empty cache).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum PrefillChunkMode {
    /// `seq_len == 0` — produce into an empty cache (store producer / cold baseline).
    Cold = 0,
    /// `seq_len > 0` — extend after restore or partial prefill.
    WarmExtend = 1,
}

/// Select the prefill chunk for a request from cache occupancy.
///
/// Entry-point contract (design Track A / PR2 matrix): every path that runs
/// chunked prefill must use this rule so cold and warm trails cannot silently
/// swap fields. Returns `(chunk_tokens, mode)`.
pub fn select_prefill_chunk_for_request(
    seq_len: usize,
    cold_prefill_chunk: usize,
    warm_prefill_chunk: usize,
) -> (usize, PrefillChunkMode) {
    if seq_len == 0 {
        (cold_prefill_chunk.max(1), PrefillChunkMode::Cold)
    } else {
        (warm_prefill_chunk.max(1), PrefillChunkMode::WarmExtend)
    }
}

/// Long-prompt Metal-friendly prefill chunk (M5 Max Gemma-12B 4-bit pure
/// sweep 2026-07-24: 512 tok/s-best; 1536/2048 slower).
pub const LONG_PROMPT_PREFILL_CHUNK: usize = 512;
/// Remaining-prompt threshold that engages [`long_prompt_prefill_chunk`].
pub const LONG_PROMPT_PREFILL_THRESHOLD: usize = 2048;

/// Cap applied to long remaining prompts. Default
/// [`LONG_PROMPT_PREFILL_CHUNK`]; override with
/// `AX_MLX_LONG_PROMPT_PREFILL_CHUNK=N` for pure / S1 thr A/B of intermediate
/// sizes (e.g. 768) that the original 512-vs-1536/2048 sweep did not cover.
pub fn long_prompt_prefill_chunk() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_positive_usize_env("AX_MLX_LONG_PROMPT_PREFILL_CHUNK")
            .unwrap_or(LONG_PROMPT_PREFILL_CHUNK)
            .max(1)
    })
}

/// Whether the Gemma-12B 512-token long-prompt clamp applies to this family.
///
/// The clamp is Gemma-12B M5 evidence (2026-07-24). Applying it to
/// `qwen3_5` (Qwen 3.5/3.6 linear 27B) splits a 2048-token prefill into
/// four evals and is the p2048 cell that misses mlx_lm. Decode on this
/// family is not SWA-ring-bound, so the Gemma decode penalty does not
/// apply. `muse_glimmer` is exempt for the same shape reason: its sliding
/// window is 2048, so a 512-token clamp buys no SWA trimming below the
/// window and only splits a 2048-token prefill into four evals. Other
/// families keep the historical clamp.
pub fn long_prompt_prefill_clamp_applies(model_family: &str) -> bool {
    !(model_family.eq_ignore_ascii_case("qwen3_5")
        || model_family.eq_ignore_ascii_case("muse_glimmer")
        || model_family.eq_ignore_ascii_case("qwen3_vl_moe")
        || model_family.eq_ignore_ascii_case("qwen3_vl"))
}

/// Scale a base prefill chunk for the remaining prompt length.
///
/// Long remaining prompts clamp to [`long_prompt_prefill_chunk`] so formal S1
/// thr keeps the pure envelope when the session base is larger (e.g. 1536).
/// Short prompts keep `base_chunk` (S0 34-token prompts are a single chunk
/// either way, so TTFT is dominated by warmup/host, not chunk size).
pub fn scale_prefill_chunk_for_remaining(base_chunk: usize, remaining_tokens: usize) -> usize {
    scale_prefill_chunk_for_remaining_in_family(base_chunk, remaining_tokens, "")
}

env_flag_default_on!(
    /// `AX_MLX_QWEN_MOE_PREFILL_SINGLE_2048` — MoE hybrid linear-attention
    /// models (Qwen 3.6 35B-A3B class) prefill in one 2048-token chunk
    /// instead of two 1024 chunks. Per-chunk MoE argsort/gather/dispatch
    /// overhead scales with chunk count; `df-macbookpro-m5` A/B
    /// (2026-08-17, 35B AXQ 6bit, p2048, reps 5): **+11.9%** prefill
    /// (2813.59 → 3149.26 tok/s), decode flat. Dense hybrids (27B) keep the
    /// 1024 TG tile (single-2048 measured wash there, 2026-08-13).
    ///
    /// **Default: ON** (kill switch `=0`).
    qwen_moe_prefill_single_2048_enabled,
    "AX_MLX_QWEN_MOE_PREFILL_SINGLE_2048"
);

env_flag!(
    /// `AX_MLX_SKIP_COLD_PREFILL_CACHE_CLEAR` — keep MLX's buffer freelist
    /// across requests instead of dropping it before every cold prefill.
    /// The historical clear protects decode `eval_kv_refs` wall from a
    /// previous request's residency, but it also forces every buffer in the
    /// prefill graph to be re-allocated inside the measured window — a cost
    /// mlxcel never pays (its process-wide pool stays warm; it clears
    /// *after* the prefill timer). p128 short-prefill lever for
    /// PRD-M5-FLEET-AX-VS-MLXCEL.
    ///
    /// **Default: OFF** (clear preserved). M5 A/B 2026-08-17
    /// (`df-macbookpro-m5`, 27B AXQ 6bit, p128/p512, reps 5): **wash** —
    /// p128 378.55 (off) vs 378.10 (on), p512 703.77 vs 703.34. The clear
    /// is not the short-prefill gap; do not reopen without new evidence.
    skip_cold_prefill_cache_clear_enabled,
    "AX_MLX_SKIP_COLD_PREFILL_CACHE_CLEAR"
);

/// Drop MLX's graph/buffer cache before a cold prefill so the previous
/// request's decode residency does not inflate `eval_kv_refs` wall.
pub fn should_clear_mlx_cache_before_cold_prefill(seq_len: usize) -> bool {
    should_clear_mlx_cache_before_cold_prefill_for(skip_cold_prefill_cache_clear_enabled(), seq_len)
}

/// Pure helper for [`should_clear_mlx_cache_before_cold_prefill`].
pub fn should_clear_mlx_cache_before_cold_prefill_for(skip_enabled: bool, seq_len: usize) -> bool {
    seq_len == 0 && !skip_enabled
}

/// Contract shapes (p128/p512/p2048) should use one prefill forward.
///
/// The n−1 cache-only split is the mlx_lm-shaped high-water path, but
/// mlxcel-bench-decode does a single 128/512/2048 forward. On
/// `df-macbookpro-m5` Wave-1 that split left Gemma 4 E2B p128 at 0.35×
/// mlxcel. Skip it for Certified non-DeepSeek families up to 2048.
/// `muse_glimmer` (dense standard route, SWA window 2048 ≥ every contract
/// shape) joins the list for the Wave-2 AXQ lane: its split cost is the
/// same two-forward shape, and its sliding window never trims below 2048.
pub fn skip_cache_only_split_for_family(model_family: &str, total_tokens: usize) -> bool {
    if !(1..=2048).contains(&total_tokens) {
        return false;
    }
    matches!(
        model_family.to_ascii_lowercase().as_str(),
        "qwen3_5"
            | "qwen3_next"
            | "qwen3"
            | "gemma4"
            | "glm4_moe_lite"
            | "gpt_oss"
            | "muse_glimmer"
            | "qwen3_vl"
            | "qwen3_vl_moe"
    )
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_INTERMEDIATE_ASYNC_EVAL` — after each non-final
    /// Qwen 3.5/3.6 prefill chunk, `async_eval` KV so GPU runs chunk N while
    /// the host builds chunk N+1.
    ///
    /// Qwen 3.6 27B p2048 remasured 890.5 vs 891.0 for the lazy two-chunk
    /// graph (2026-08-13). **Default: OFF**.
    qwen_prefill_intermediate_async_eval_enabled,
    "AX_MLX_QWEN_PREFILL_INTERMEDIATE_ASYNC_EVAL"
);

/// Whether a non-final Qwen prefill chunk should async-submit KV.
pub fn should_async_eval_intermediate_qwen_prefill(
    model_family: &str,
    is_final_chunk: bool,
) -> bool {
    should_async_eval_intermediate_qwen_prefill_for(
        qwen_prefill_intermediate_async_eval_enabled(),
        model_family,
        is_final_chunk,
    )
}

/// Pure helper for [`should_async_eval_intermediate_qwen_prefill`].
pub fn should_async_eval_intermediate_qwen_prefill_for(
    enabled: bool,
    model_family: &str,
    is_final_chunk: bool,
) -> bool {
    enabled && !is_final_chunk && model_family.eq_ignore_ascii_case("qwen3_5")
}

env_flag!(
    /// `AX_MLX_QWEN_PREFILL_LAZY_INTERMEDIATE` — skip the blocking
    /// `eval_with_kv_refs` after a non-final Qwen 3.5/3.6 `--ax-direct`
    /// chunk on contract totals `1..=2048`.
    ///
    /// **Default: OFF**. Four-lane remasure (binary `b50c209f…`, 2026-08-13):
    /// AXQ p2048 889.887/862.825=1.031365 (0.9987× q2only 891). Community
    /// p2048 910.410/858.000=1.061085 (3d still FAIL). Same class as
    /// intermediate async_eval 890.5. Cache-only prefix still last-chunk-evals.
    qwen_prefill_lazy_intermediate_enabled,
    "AX_MLX_QWEN_PREFILL_LAZY_INTERMEDIATE"
);

/// Whether a non-final Qwen `--ax-direct` chunk should stay lazy.
pub fn should_keep_lazy_intermediate_qwen_prefill(
    model_family: &str,
    is_final_chunk: bool,
    total_tokens: usize,
) -> bool {
    should_keep_lazy_intermediate_qwen_prefill_for(
        qwen_prefill_lazy_intermediate_enabled(),
        model_family,
        is_final_chunk,
        total_tokens,
    )
}

/// Pure helper for [`should_keep_lazy_intermediate_qwen_prefill`].
pub fn should_keep_lazy_intermediate_qwen_prefill_for(
    enabled: bool,
    model_family: &str,
    is_final_chunk: bool,
    total_tokens: usize,
) -> bool {
    enabled
        && !is_final_chunk
        && model_family.eq_ignore_ascii_case("qwen3_5")
        && skip_cache_only_split_for_family(model_family, total_tokens)
}

/// Family-aware variant of [`scale_prefill_chunk_for_remaining`].
pub fn scale_prefill_chunk_for_remaining_in_family(
    base_chunk: usize,
    remaining_tokens: usize,
    model_family: &str,
) -> usize {
    let base = base_chunk.max(1);
    if remaining_tokens >= LONG_PROMPT_PREFILL_THRESHOLD
        && long_prompt_prefill_clamp_applies(model_family)
    {
        base.clamp(1, long_prompt_prefill_chunk())
    } else {
        base
    }
}

/// Token count for constructor JIT warm-up. Non-MLA models keep the historical
/// small warm-up prompt. MLA models warm at least one full effective chunk so
/// the compiled prefill graph matches the default chunk-aligned runtime path.
pub fn prefill_warmup_token_count(
    has_mla_attention: bool,
    effective_prefill_chunk: usize,
) -> usize {
    if has_mla_attention {
        effective_prefill_chunk.max(1)
    } else {
        8
    }
}

/// Prefill lengths to JIT at runner construction.
///
/// Always includes the historical lightweight warm-up plus short interactive
/// prompt shapes (32/34/64). The flip S0 contract uses 34 prompt tokens; those
/// graphs are shape-sensitive on hybrid Qwen3.5 (linear + full attention).
pub fn prefill_warmup_token_lengths(
    has_mla_attention: bool,
    effective_prefill_chunk: usize,
) -> Vec<usize> {
    let base = prefill_warmup_token_count(has_mla_attention, effective_prefill_chunk).max(1);
    let mut lengths = vec![base, 32, 34, 64];
    lengths.sort_unstable();
    lengths.dedup();
    lengths
}

/// Disk prefix-cache directory. When `AX_MLX_PREFIX_CACHE_DIR=<path>`
/// is set (and `AX_MLX_PREFIX_CACHE_DISK_DISABLED` is not engaged),
/// `MlxRunner` opens an L2 file-backed prefix cache rooted at that
/// directory and writes snapshots there alongside the in-memory L1
/// store. Unset by default — the disk cache is **opt-in**.
/// Cached at first read per the module-level OnceLock contract.
pub fn prefix_cache_dir() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    static CACHED: OnceLock<Option<PathBuf>> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            let raw = std::env::var("AX_MLX_PREFIX_CACHE_DIR").ok()?;
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(PathBuf::from(trimmed))
            }
        })
        .clone()
}

env_flag!(
    /// **Defensive kill switch.** Engaged by
    /// `AX_MLX_PREFIX_CACHE_DISK_DISABLED`, this forces the L2 disk
    /// prefix cache off even when `AX_MLX_PREFIX_CACHE_DIR` is set.
    /// Used by operators who want to disable the disk path without
    /// editing the cache-directory environment variable (e.g. to
    /// isolate a regression to the L1-only path during diagnosis).
    prefix_cache_disk_disabled,
    "AX_MLX_PREFIX_CACHE_DISK_DISABLED"
);

env_flag!(
    /// **Defensive kill switch.** Engaged by `AX_DISABLE_MLA_PREFIX_RESTORE`,
    /// this re-engages the historical `mla_extend_unsafe` safety gate in
    /// `restore_reused_prefix_state` that refused to restore an MLA snapshot
    /// for Prefill-mode requests. The gate was originally added because
    /// post-restore `chunked_prefill` over a suffix drifted fp-wise from a
    /// cold full-prefill on GLM-4.7-Flash. Evidence points to
    /// shape-dependent SDPA kernel selection in MLX, where cold and warm
    /// paths dispatched different chunk shapes. Aligning the MLA prefill
    /// chunk size to the prefix-cache block size (default 16; see
    /// `MLA_DEFAULT_PREFILL_CHUNK`) **and** matching cold production to that
    /// same chunk (R2; see `resolve_mla_cold_prefill_chunk`) is required for
    /// warm_extend token-exact parity when a real snapshot hit occurs.
    /// This flag exists as a fail-closed escape hatch if a future workload
    /// exposes a residual drift vector those chunk-alignment rules miss.
    mla_prefix_restore_disabled,
    "AX_DISABLE_MLA_PREFIX_RESTORE"
);

// ── DiffusionGemma denoise-loop overrides ──────────────────────────────────
//
// These accessors let benchmark campaigns sweep convergence thresholds and
// toggle the compiled denoise closure without recompiling. All are read once
// per process and cached via OnceLock.

env_flag!(
    /// `AX_MLX_GEMMA4_ASSISTANT_COMPILE` — reserved for pure-graph assistant
    /// MTP compile (Phase B).
    ///
    /// **Default: OFF** (opt-in via `AX_MLX_GEMMA4_ASSISTANT_COMPILE=1`).
    ///
    /// History: a Phase-4 scaffold wrapped the imperative assistant forward
    /// in an uncompiled `MlxClosure` that re-synced scalars every depth and
    /// could only add overhead. That wrapper is removed; when this flag is
    /// set the runner still runs the real forward path until a pure
    /// `mlx_compile` design (target KV + dynamic RoPE as array inputs) lands
    /// with same-artifact A/B evidence for default-on promotion.
    gemma4_assistant_compile_enabled,
    "AX_MLX_GEMMA4_ASSISTANT_COMPILE"
);

env_flag!(
    /// `AX_MLX_GEMMA4_ASSISTANT_LAZY_MULTI_DEPTH` — fuse multi-depth Gemma
    /// assistant drafting into a single materialize.
    ///
    /// **Default: OFF** (opt-in via `AX_MLX_GEMMA4_ASSISTANT_LAZY_MULTI_DEPTH=1`).
    ///
    /// Builds the full depth chain lazily (argmax token of depth `d` feeds
    /// embedding of depth `d+1` without a host sync) and materialises all
    /// draft tokens + GPU-exact confidences in one `eval`. Host-side
    /// confidence gates still apply after materialisation and stop the
    /// accepted prefix at the first miss — same correctness contract as the
    /// per-depth sync loop. Depth-1 drafts are unchanged.
    ///
    /// Same-artifact A/B on gemma-4-12b-it-4bit-ffn4-assistant-mtp (depth 2,
    /// n-gram stacking off, flappy + long_code, gen=256) was accept-neutral
    /// but not a clear decode win: deep drafts rarely clear the 0.999 deep
    /// gate on 12B, so the always-fused chain pays for depth-1 forwards that
    /// the gated early-stop path already ran when the first gate passes.
    /// Keep opt-in for workloads where deep drafts fire often (e.g. looser
    /// deep gate probes).
    gemma4_assistant_lazy_multi_depth_enabled,
    "AX_MLX_GEMMA4_ASSISTANT_LAZY_MULTI_DEPTH"
);

env_flag_default_on!(
    /// `AX_MLX_GEMMA4_ASSISTANT_DEEP_NEEDS_FIRST_CONF` — only spend a second
    /// (deep) assistant forward when the first draft token's confidence
    /// already clears the deep gate.
    ///
    /// **Default: ON** (kill-switch via
    /// `AX_MLX_GEMMA4_ASSISTANT_DEEP_NEEDS_FIRST_CONF=0`).
    ///
    /// Mirrors vLLM Gemma 4 MTP practice of starting with a small
    /// `num_speculative_tokens` and dynamic speculation depth: if the
    /// assistant is not already extremely confident on position 0 (the same
    /// bar required to keep a deep draft), a frozen-KV recurrent step is
    /// unlikely to clear the deep gate, so the extra forward is pure waste.
    /// Accept rate is unchanged when deep drafts were never kept; when they
    /// fire, conf0 is typically already above the deep bar.
    gemma4_assistant_deep_needs_first_conf_enabled,
    "AX_MLX_GEMMA4_ASSISTANT_DEEP_NEEDS_FIRST_CONF"
);

env_flag!(
    /// `AX_DIFFUSION_NO_EMBEDDING_CACHE` — opt-out of per-layer embedding
    /// input caching on the imperative denoise fallback. Default: cache is
    /// **ON** for non-full-pipeline paths (fingerprint skip of ~46 embed
    /// dispatches when tokens are unchanged). Full-pipeline compile does
    /// not use this cache (`mlx_compile` purity).
    diffusion_no_embedding_cache,
    "AX_DIFFUSION_NO_EMBEDDING_CACHE"
);

env_flag!(
    /// `AX_DIFFUSION_NO_KV_CONCAT_BUFFER` — opt-out of pre-allocated KV
    /// concatenation buffers on the imperative denoise fallback. Default:
    /// buffer path is **ON** for non-full-pipeline paths (`slice_update` +
    /// `contiguous`/`eval`, bit-matched to re-concatenate). Full-pipeline
    /// compile does not use these buffers (`mlx_compile` purity).
    diffusion_no_kv_concat_buffer,
    "AX_DIFFUSION_NO_KV_CONCAT_BUFFER"
);

// Legacy opt-in names kept so older bench scripts still force-enable (no-ops
// when the new defaults already enable the path). Prefer the `NO_*` kill
// switches for new work.
env_flag!(
    /// Legacy: `AX_DIFFUSION_EMBEDDING_CACHE=1` force-enable (redundant with default ON).
    diffusion_embedding_cache_enabled,
    "AX_DIFFUSION_EMBEDDING_CACHE"
);

env_flag!(
    /// Legacy: `AX_DIFFUSION_KV_CONCAT_BUFFER=1` force-enable (redundant with default ON).
    diffusion_kv_concat_buffer_enabled,
    "AX_DIFFUSION_KV_CONCAT_BUFFER"
);

env_flag!(
    /// `AX_DIFFUSION_NO_FULL_PIPELINE` — opt-out of the full-pipeline compiled
    /// closure that fuses forward + softmax + entropy + sampling + acceptance
    /// into a single MLX graph (~280 dispatches → 1). Supersedes the
    /// forward-only compiled closure. **Default ON** for best performance.
    diffusion_no_full_pipeline,
    "AX_DIFFUSION_NO_FULL_PIPELINE"
);

env_flag!(
    /// `AX_DIFFUSION_NO_COMPILED_FORWARD` — opt-out of the compiled
    /// forward closure that is enabled by default when self-conditioning
    /// is off. When set to `1`, the imperative forward path is used.
    diffusion_no_compiled_forward,
    "AX_DIFFUSION_NO_COMPILED_FORWARD"
);

env_flag_default_on!(
    /// `AX_MTP_COMPILED_HEAD` — compile the multi-depth MTP draft chain
    /// into a single `mlx_compile`-fused closure dispatch.
    ///
    /// **Default: ON** (kill switch via `AX_MTP_COMPILED_HEAD=0`).
    ///
    /// Wraps the full multi-depth Qwen MTP head recurrence (forward + post-norm
    /// + logits across all D draft depths) in one `MlxClosure::compile` call to
    /// fuse ops across the chain.  The closure is **pure**: it captures only
    /// model constants (cfg/weights/head), receives the existing context as the
    /// explicit inputs `init_k`/`init_v`, threads the new per-depth K/V
    /// functionally (concat, no cache mutation), and emits the final K/V as
    /// outputs for the caller to commit.  This satisfies `mlx_compile`'s
    /// pure-function contract (see `MlxClosure::new_dyn`).
    ///
    /// The RoPE offset is passed as an `MlxArray` runtime input (via
    /// `mlx_fast_rope_dynamic`) rather than baked as a constant, so the
    /// compiled closure is reused across decode steps without recompilation.
    ///
    /// Applies to the Qwen MTP head only.  GLM (MLA latent cache)
    /// deliberately stays on the imperative path.  Gemma assistant-MTP is a
    /// separate path and also ignores this flag.
    mtp_compiled_head_enabled,
    "AX_MTP_COMPILED_HEAD"
);

env_flag!(
    /// `AX_MTP_COMPILED_HEAD_FIXED_KV` — give the compiled multi-depth Qwen
    /// draft head a fixed-capacity K/V buffer and an explicit tensor write
    /// offset. This avoids concatenating the complete MTP history at every
    /// draft depth and makes one compiled closure reusable across steps.
    ///
    /// **Default: OFF** pending matched M5 admission.
    mtp_compiled_head_fixed_kv_enabled,
    "AX_MTP_COMPILED_HEAD_FIXED_KV"
);

env_flag!(
    /// `AX_DIFFUSION_NO_SKIP_COMMIT` — opt-out of the causal commit
    /// skip that is enabled by default on convergence with high
    /// acceptance. When set to `1`, the causal commit pass always runs.
    diffusion_no_skip_commit,
    "AX_DIFFUSION_NO_SKIP_COMMIT"
);

env_flag!(
    /// `AX_DIFFUSION_PROFILE` — enable per-layer timing output for the
    /// bidirectional denoiser forward pass. When set to `1`, each layer
    /// call in `forward_bidirectional` is timed and reported via
    /// `eprintln!`, giving per-step observability into the denoise
    /// pipeline. Default OFF; opt-in for profiling.
    diffusion_profile_enabled,
    "AX_DIFFUSION_PROFILE"
);

env_flag_default_on!(
    /// `AX_MLX_BATCHED_SHARED_PROJ` — route batched-decode projections
    /// (QKV, attention output, FFN, lm_head) through a single batched
    /// `quantized_matmul` (`ProjectionBatchPolicy::Shared`) instead of the
    /// per-row `RowExact` loop.
    ///
    /// **Default: ON** (kill-switch via `AX_MLX_BATCHED_SHARED_PROJ=0`, which
    /// restores the per-row bit-exact path). `RowExact` runs one
    /// `quantized_matmul` per batch row so each row is bit-identical to
    /// single-request decode, but it re-reads the weight B times and so does
    /// not amortize the weight read — the batched FFN + lm_head then dominate
    /// and cap aggregate scaling at ~1.24× (Phase 3.4,
    /// docs/performance/batched-decode-ceiling.md). `Shared` reads each weight
    /// once for all rows and amortizes: **+56% aggregate throughput at batch=8
    /// on Llama-8B-4bit** (65→97 tok/s, 1.23×→1.92× scaling).
    ///
    /// The batched vs per-row `quantized_matmul` bf16 accumulation drift
    /// (~2.3e-2) does **not** flip greedy argmax: the decoded token stream is
    /// byte-identical to RowExact across three dense checkpoints (Llama-3.1-8B,
    /// Qwen3-4B, Ministral-8B) and the 25 batched correctness tests pass with
    /// this policy. Batched decode is itself opt-in (`AX_MLX_BATCHED_DECODE`),
    /// so the appropriate certification is greedy-token equivalence, which this
    /// meets; the kill-switch restores bit-exactness if a checkpoint ever needs
    /// it.
    batched_shared_projections_enabled,
    "AX_MLX_BATCHED_SHARED_PROJ"
);

/// Diffusion convergence: mean entropy threshold below which strict
/// convergence triggers. Defaults to 0.005 when unset.
pub fn diffusion_entropy_threshold() -> Option<f32> {
    static CACHED: OnceLock<Option<f32>> = OnceLock::new();
    *CACHED.get_or_init(|| parse_nonnegative_f32_env("AX_DIFFUSION_ENTROPY_THRESHOLD"))
}

/// Diffusion convergence: update-rate threshold below which adaptive
/// convergence triggers. Defaults to 0.075 (7.5%) when unset.
pub fn diffusion_acceptance_rate_threshold() -> Option<f32> {
    static CACHED: OnceLock<Option<f32>> = OnceLock::new();
    *CACHED.get_or_init(|| parse_nonnegative_f32_env("AX_DIFFUSION_ACCEPTANCE_RATE_THRESHOLD"))
}

/// Diffusion convergence: entropy plateau delta below which plateau
/// convergence triggers (after step 16 warmup). Defaults to 0.001 when unset.
pub fn diffusion_entropy_plateau_delta() -> Option<f32> {
    static CACHED: OnceLock<Option<f32>> = OnceLock::new();
    *CACHED.get_or_init(|| parse_nonnegative_f32_env("AX_DIFFUSION_ENTROPY_PLATEAU_DELTA"))
}

/// Diffusion: maximum denoise steps per block. Defaults to 48 when unset.
pub fn diffusion_max_steps() -> Option<usize> {
    static CACHED: OnceLock<Option<usize>> = OnceLock::new();
    *CACHED.get_or_init(|| parse_positive_usize_env("AX_DIFFUSION_MAX_STEPS"))
}

/// Diffusion: max denoise steps to run per engine decode call when multi-step
/// scheduling is enabled. `None` / unset means monoblock (run until
/// convergence or `max_denoise_steps` inside one call).
pub fn diffusion_steps_per_engine_step() -> Option<usize> {
    static CACHED: OnceLock<Option<usize>> = OnceLock::new();
    *CACHED.get_or_init(|| parse_positive_usize_env("AX_DIFFUSION_STEPS_PER_ENGINE_STEP"))
}

/// Diffusion: steps between convergence checks. Defaults to 1 (check every
/// step). Larger values reduce per-step scalar evals (negligible — see A/B) but
/// detect convergence on a coarser grid, overshooting the true convergence step
/// and wasting denoise passes. Kept as an override for benchmarking only.
pub fn diffusion_check_interval() -> Option<usize> {
    static CACHED: OnceLock<Option<usize>> = OnceLock::new();
    *CACHED.get_or_init(|| parse_positive_usize_env("AX_DIFFUSION_CHECK_INTERVAL"))
}

/// Diffusion sampler strategy override. Returns the raw env-var string when
/// `AX_DIFFUSION_SAMPLER` is set (e.g. `"confidence_threshold"` or
/// `"entropy_bound"`). The caller maps the string to `DiffusionSampler`.
pub fn diffusion_sampler() -> Option<String> {
    static CACHED: OnceLock<Option<String>> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            std::env::var("AX_DIFFUSION_SAMPLER")
                .ok()
                .map(|s| s.trim().to_lowercase())
        })
        .clone()
}

/// Diffusion confidence-threshold sampler: accept positions whose peak
/// softmax probability exceeds this value. Defaults to 0.9 when unset.
pub fn diffusion_confidence_threshold() -> Option<f32> {
    static CACHED: OnceLock<Option<f32>> = OnceLock::new();
    *CACHED.get_or_init(|| parse_nonnegative_f32_env("AX_DIFFUSION_CONFIDENCE_THRESHOLD"))
}

/// Diffusion temperature schedule override. Returns the raw env-var string
/// when `AX_DIFFUSION_TEMPERATURE_SCHEDULE` is set (e.g. `"exponential"` or
/// `"linear"`). `None` keeps the manifest default (Linear).
pub fn diffusion_temperature_schedule() -> Option<String> {
    static CACHED: OnceLock<Option<String>> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            std::env::var("AX_DIFFUSION_TEMPERATURE_SCHEDULE")
                .ok()
                .map(|s| s.trim().to_lowercase())
        })
        .clone()
}

/// Diffusion self-conditioning skip threshold. When the canvas acceptance
/// rate exceeds this value, the expensive `prob × embed_table` matmul is
/// skipped because the self-conditioning signal barely changes. Defaults to
/// 0.95 when unset.
pub fn diffusion_sc_skip_acceptance_rate() -> Option<f32> {
    static CACHED: OnceLock<Option<f32>> = OnceLock::new();
    *CACHED.get_or_init(|| parse_nonnegative_f32_env("AX_DIFFUSION_SC_SKIP_ACCEPTANCE_RATE"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(name: &str, value: &str) -> bool {
        // SAFETY: each test owns a disjoint set of env-var names. Remove
        // before asserting so a failing assert does not leak the var.
        unsafe {
            std::env::set_var(name, value);
        }
        let observed = parse_bool_env(name);
        unsafe {
            std::env::remove_var(name);
        }
        observed
    }

    #[test]
    fn parse_bool_env_treats_truthy_values_as_engaged() {
        // Exercises canonical casing, all-upper, mixed case, and surrounding
        // whitespace to lock in the parser contract documented at the module
        // level.
        for value in [
            "1", "true", "TRUE", "True", "tRuE", "yes", "YES", "Yes", " 1 ", "\ttrue\n",
        ] {
            let name = format!("AX_FASTPATH_TEST_TRUTHY_{}", value.trim());
            assert!(probe(&name, value), "expected truthy for {value:?}");
        }
    }

    #[test]
    fn parse_bool_env_rejects_other_values() {
        for value in ["0", "false", "no", "off", "on", "", "anything", "  "] {
            let name = format!("AX_FASTPATH_TEST_FALSY_{}", value.trim());
            assert!(!probe(&name, value), "expected falsy for {value:?}");
        }
    }

    #[test]
    fn parse_bool_env_unset_is_false() {
        assert!(!parse_bool_env("AX_FASTPATH_TEST_DEFINITELY_UNSET"));
    }

    #[test]
    fn qwen_linear_mtp_exact_resolution_is_capability_bounded() {
        assert_eq!(
            resolve_qwen_linear_mtp_exact_with_override(false, None),
            (false, QwenLinearMtpExactSelection::Ineligible)
        );
        assert_eq!(
            resolve_qwen_linear_mtp_exact_with_override(false, Some(true)),
            (false, QwenLinearMtpExactSelection::Ineligible),
            "an env opt-in must not bypass the hard model gate"
        );
        assert_eq!(
            resolve_qwen_linear_mtp_exact_with_override(true, None),
            (true, QwenLinearMtpExactSelection::Auto)
        );
        assert_eq!(
            resolve_qwen_linear_mtp_exact_with_override(true, Some(true)),
            (true, QwenLinearMtpExactSelection::ExplicitEnabled)
        );
        assert_eq!(
            resolve_qwen_linear_mtp_exact_with_override(true, Some(false)),
            (false, QwenLinearMtpExactSelection::ExplicitDisabled)
        );
    }

    #[test]
    fn qwen_linear_mtp_exact_scope_is_nested_and_restored() {
        let baseline = qwen_linear_mtp_exact_enabled();
        {
            let _outer = scoped_qwen_linear_mtp_exact(true);
            assert!(qwen_linear_mtp_exact_enabled());
            {
                let _inner = scoped_qwen_linear_mtp_exact(false);
                assert!(!qwen_linear_mtp_exact_enabled());
            }
            assert!(qwen_linear_mtp_exact_enabled());
        }
        assert_eq!(qwen_linear_mtp_exact_enabled(), baseline);
    }

    #[test]
    fn relaxed_target_verify_enables_only_verify_fast_kernel_marker() {
        let exact_baseline = qwen_linear_mtp_exact_enabled();
        let fast_baseline = qwen_linear_mtp_verify_fast_kernels_enabled();
        let target_baseline = qwen_linear_mtp_target_verify_enabled();
        {
            let _exact = scoped_qwen_linear_mtp_exact(false);
            assert!(!qwen_linear_mtp_exact_enabled());
            assert!(!qwen_linear_mtp_verify_fast_kernels_enabled());
            assert!(!qwen_linear_mtp_target_verify_enabled());
            {
                let _verify = scoped_qwen_linear_mtp_target_verify(true);
                assert!(!qwen_linear_mtp_exact_enabled());
                assert!(qwen_linear_mtp_verify_fast_kernels_enabled());
                assert!(qwen_linear_mtp_target_verify_enabled());
            }
            assert!(!qwen_linear_mtp_verify_fast_kernels_enabled());
            assert!(!qwen_linear_mtp_target_verify_enabled());
        }
        assert_eq!(qwen_linear_mtp_exact_enabled(), exact_baseline);
        assert_eq!(qwen_linear_mtp_verify_fast_kernels_enabled(), fast_baseline);
        assert_eq!(qwen_linear_mtp_target_verify_enabled(), target_baseline);
    }

    #[test]
    fn relaxed_mtp_async_dual_gate_up_is_scope_family_and_window_gated() {
        assert!(should_mtp_async_dual_gate_up_for(true, true, "qwen3_5", 3));
        assert!(should_mtp_async_dual_gate_up_for(true, true, "QWEN3_5", 4));
        assert!(!should_mtp_async_dual_gate_up_for(
            false, true, "qwen3_5", 3
        ));
        assert!(!should_mtp_async_dual_gate_up_for(
            true, false, "qwen3_5", 3
        ));
        assert!(!should_mtp_async_dual_gate_up_for(true, true, "qwen3_5", 1));
        assert!(!should_mtp_async_dual_gate_up_for(true, true, "gemma4", 3));
    }

    #[test]
    fn relaxed_mtp_session_scope_restores_nested_state() {
        let baseline = qwen_linear_mtp_relaxed_session_enabled();
        {
            let _outer = scoped_qwen_linear_mtp_relaxed_session(true);
            assert!(qwen_linear_mtp_relaxed_session_enabled());
            {
                let _inner = scoped_qwen_linear_mtp_relaxed_session(false);
                assert!(!qwen_linear_mtp_relaxed_session_enabled());
            }
            assert!(qwen_linear_mtp_relaxed_session_enabled());
        }
        assert_eq!(qwen_linear_mtp_relaxed_session_enabled(), baseline);
    }

    #[test]
    fn exact_verify_async_kernel_boundary_is_s2_to_s4_only() {
        let baseline = qwen_linear_mtp_exact_enabled();
        {
            let _exact = scoped_qwen_linear_mtp_exact(true);
            assert!(!should_exact_verify_async_kernel_boundary(1));
            assert!(should_exact_verify_async_kernel_boundary(2));
            assert!(should_exact_verify_async_kernel_boundary(3));
            assert!(should_exact_verify_async_kernel_boundary(4));
            assert!(!should_exact_verify_async_kernel_boundary(5));
        }
        {
            let _off = scoped_qwen_linear_mtp_exact(false);
            assert!(!should_exact_verify_async_kernel_boundary(2));
        }
        assert_eq!(qwen_linear_mtp_exact_enabled(), baseline);
    }

    fn probe_default_on(name: &str, value: &str) -> bool {
        // SAFETY: each test owns a disjoint set of env-var names. Remove
        // before asserting so a failing assert does not leak the var.
        unsafe {
            std::env::set_var(name, value);
        }
        let observed = parse_bool_env_default_on(name);
        unsafe {
            std::env::remove_var(name);
        }
        observed
    }

    #[test]
    fn parse_bool_env_default_on_only_rejects_explicit_falsy_values() {
        assert!(parse_bool_env_default_on(
            "AX_FASTPATH_TEST_DEFAULT_ON_UNSET"
        ));
        for value in [
            "0", "false", "FALSE", "False", "no", "NO", "No", "off", "OFF",
        ] {
            let name = format!("AX_FASTPATH_TEST_DEFAULT_ON_FALSY_{}", value.trim());
            assert!(
                !probe_default_on(&name, value),
                "expected explicit falsy for {value:?}"
            );
        }
        for value in ["", " ", "1", "true", "yes", "anything"] {
            let name = format!(
                "AX_FASTPATH_TEST_DEFAULT_ON_TRUTHY_{}",
                value.trim().replace(' ', "space")
            );
            assert!(
                probe_default_on(&name, value),
                "expected default-on truthy for {value:?}"
            );
        }
    }

    #[test]
    fn linear_attention_projection_packing_uses_default_on_kill_switch_contract() {
        assert!(parse_bool_env_default_on(
            "AX_FASTPATH_TEST_LINEAR_ATTENTION_PACK_UNSET"
        ));
        assert!(!probe_default_on(
            "AX_FASTPATH_TEST_LINEAR_ATTENTION_PACK_DISABLED",
            "0"
        ));
        assert!(probe_default_on(
            "AX_FASTPATH_TEST_LINEAR_ATTENTION_PACK_ENABLED",
            "1"
        ));
    }

    #[test]
    fn direct_cpp_linear_attention_inputs_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_DIRECT_LINEAR_ATTENTION_INPUTS_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_DIRECT_LINEAR_ATTENTION_INPUTS_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_DIRECT_LINEAR_ATTENTION_INPUTS_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_direct_cpp_linear_attention_inputs_uses_default_on_contract() {
        assert!(parse_bool_env_default_on(
            "AX_FASTPATH_TEST_QWEN_DIRECT_LINEAR_ATTENTION_INPUTS_UNSET"
        ));
        assert!(!probe_default_on(
            "AX_FASTPATH_TEST_QWEN_DIRECT_LINEAR_ATTENTION_INPUTS_DISABLED",
            "0"
        ));
        assert!(probe_default_on(
            "AX_FASTPATH_TEST_QWEN_DIRECT_LINEAR_ATTENTION_INPUTS_ENABLED",
            "1"
        ));
    }

    #[test]
    fn direct_cpp_linear_attention_post_input_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_DIRECT_LINEAR_ATTENTION_POST_INPUT_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_DIRECT_LINEAR_ATTENTION_POST_INPUT_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_DIRECT_LINEAR_ATTENTION_POST_INPUT_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_direct_cpp_linear_attention_post_input_uses_default_on_contract() {
        assert!(parse_bool_env_default_on(
            "AX_FASTPATH_TEST_QWEN_DIRECT_LINEAR_ATTENTION_POST_INPUT_UNSET"
        ));
        assert!(!probe_default_on(
            "AX_FASTPATH_TEST_QWEN_DIRECT_LINEAR_ATTENTION_POST_INPUT_DISABLED",
            "0"
        ));
        assert!(probe_default_on(
            "AX_FASTPATH_TEST_QWEN_DIRECT_LINEAR_ATTENTION_POST_INPUT_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_linear_attention_prefill_post_input_metal_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_LINEAR_ATTENTION_PREFILL_POST_INPUT_METAL_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_LINEAR_ATTENTION_PREFILL_POST_INPUT_METAL_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_LINEAR_ATTENTION_PREFILL_POST_INPUT_METAL_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_linear_attention_decode_post_input_metal_uses_default_on_contract() {
        assert!(parse_bool_env_default_on(
            "AX_FASTPATH_TEST_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL_UNSET"
        ));
        assert!(!probe_default_on(
            "AX_FASTPATH_TEST_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL_DISABLED",
            "0"
        ));
        assert!(probe_default_on(
            "AX_FASTPATH_TEST_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL_ENABLED",
            "1"
        ));
    }

    #[test]
    fn fused_prefill_attention_qwen_is_family_scoped_and_default_on() {
        assert!(super::fused_prefill_attention_family_supported("qwen3_5"));
        assert!(super::fused_prefill_attention_family_supported(
            "qwen3_next"
        ));
        assert!(super::fused_prefill_attention_family_supported("gemma4"));
        assert!(!super::fused_prefill_attention_family_supported(
            "glm4_moe_lite"
        ));
        assert!(
            !super::fused_prefill_attention_should_try("qwen3_5"),
            "Qwen fused prefill stays default-OFF after 895 vs 891"
        );
        assert!(!super::fused_prefill_qwen_skip_offset("qwen3_5", false));
        assert!(super::fused_prefill_qwen_skip_offset("qwen3_5", true));
        assert!(!super::fused_prefill_qwen_skip_offset("gemma4", true));
        assert!(
            !super::fused_prefill_attention_should_try("gemma4"),
            "Gemma fused prefill without a seq stays default-OFF"
        );
        assert!(
            !super::fused_prefill_attention_should_try_for_seq("gemma4", 512),
            "Gemma p512 fused prefill stays default-OFF"
        );
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_FUSED_PREFILL_ATTENTION_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_FUSED_PREFILL_ATTENTION_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_FUSED_PREFILL_ATTENTION_ENABLED",
            "1"
        ));
    }

    #[test]
    fn gemma4_fused_prefill_p128_is_seq_and_family_gated() {
        assert!(should_gemma4_fused_prefill_p128_for(true, "gemma4", 128));
        assert!(should_gemma4_fused_prefill_p128_for(
            true,
            "gemma4_unified",
            128
        ));
        assert!(super::fused_prefill_attention_should_try_for_seq(
            "gemma4", 128
        ));
        assert!(
            !should_gemma4_fused_prefill_p128_for(true, "gemma4", 512),
            "p512 must stay on the portable attention path"
        );
        assert!(!should_gemma4_fused_prefill_p128_for(true, "gemma4", 2048));
        assert!(!should_gemma4_fused_prefill_p128_for(true, "gemma4", 1));
        assert!(!should_gemma4_fused_prefill_p128_for(true, "qwen3_5", 128));
        assert!(!should_gemma4_fused_prefill_p128_for(false, "gemma4", 128));
    }

    #[test]
    fn gemma4_fused_prefill_fold_post_norm_requires_p128_and_weight() {
        assert!(should_gemma4_fused_prefill_fold_post_norm_for(
            true, "gemma4", 128, true
        ));
        assert!(should_gemma4_fused_prefill_fold_post_norm_for(
            true,
            "gemma4_unified",
            128,
            true
        ));
        assert!(
            !should_gemma4_fused_prefill_fold_post_norm_for(true, "gemma4", 128, false),
            "no sandwich post-norm means the fused call stays o-proj only"
        );
        assert!(!should_gemma4_fused_prefill_fold_post_norm_for(
            true, "gemma4", 512, true
        ));
        assert!(!should_gemma4_fused_prefill_fold_post_norm_for(
            true, "qwen3_5", 128, true
        ));
        assert!(!should_gemma4_fused_prefill_fold_post_norm_for(
            false, "gemma4", 128, true
        ));
    }

    #[test]
    fn gemma4_async_dual_gate_up_p128_is_seq_and_family_gated() {
        assert!(should_gemma4_async_dual_gate_up_p128_for(
            true, "gemma4", 128
        ));
        assert!(should_gemma4_async_dual_gate_up_p128_for(
            true,
            "gemma4_unified",
            128
        ));
        assert!(
            !should_gemma4_async_dual_gate_up_p128_for(true, "gemma4", 512),
            "p512 stays on the serial gate/up submit"
        );
        assert!(!should_gemma4_async_dual_gate_up_p128_for(
            true, "gemma4", 2048
        ));
        assert!(!should_gemma4_async_dual_gate_up_p128_for(
            true, "gemma4", 1
        ));
        assert!(!should_gemma4_async_dual_gate_up_p128_for(
            true, "qwen3_5", 128
        ));
        assert!(!should_gemma4_async_dual_gate_up_p128_for(
            false, "gemma4", 128
        ));
    }

    #[test]
    fn gemma4_async_first_kv_p128_is_seq_and_family_gated() {
        assert!(should_gemma4_async_first_kv_p128_for(true, "gemma4", 128));
        assert!(should_gemma4_async_first_kv_p128_for(
            true,
            "gemma4_unified",
            128
        ));
        assert!(
            !should_gemma4_async_first_kv_p128_for(true, "gemma4", 512),
            "p512 stays on the lazy first-KV submit"
        );
        assert!(!should_gemma4_async_first_kv_p128_for(true, "gemma4", 2048));
        assert!(!should_gemma4_async_first_kv_p128_for(true, "gemma4", 1));
        assert!(!should_gemma4_async_first_kv_p128_for(true, "qwen3_5", 128));
        assert!(!should_gemma4_async_first_kv_p128_for(false, "gemma4", 128));
    }

    #[test]
    fn gemma4_dual_stream_gate_up_p128_is_seq_and_family_gated() {
        assert!(should_gemma4_dual_stream_gate_up_p128_for(
            true, "gemma4", 128
        ));
        assert!(should_gemma4_dual_stream_gate_up_p128_for(
            true,
            "gemma4_unified",
            128
        ));
        assert!(
            !should_gemma4_dual_stream_gate_up_p128_for(true, "gemma4", 512),
            "p512 stays on the serial / compiled split-MLP path"
        );
        assert!(!should_gemma4_dual_stream_gate_up_p128_for(
            true, "gemma4", 2048
        ));
        assert!(!should_gemma4_dual_stream_gate_up_p128_for(
            true, "gemma4", 1
        ));
        assert!(!should_gemma4_dual_stream_gate_up_p128_for(
            true, "qwen3_5", 128
        ));
        assert!(!should_gemma4_dual_stream_gate_up_p128_for(
            false, "gemma4", 128
        ));
    }

    #[test]
    fn gemma4_packed_ffn_compile_p128_is_seq_and_family_gated() {
        assert!(should_gemma4_packed_ffn_compile_p128_for(
            true, "gemma4", 128
        ));
        assert!(should_gemma4_packed_ffn_compile_p128_for(
            true,
            "gemma4_unified",
            128
        ));
        assert!(
            !should_gemma4_packed_ffn_compile_p128_for(true, "gemma4", 512),
            "p512 keeps split gate/up and the 256-leading compile floor"
        );
        assert!(!should_gemma4_packed_ffn_compile_p128_for(
            true, "gemma4", 2048
        ));
        assert!(!should_gemma4_packed_ffn_compile_p128_for(
            true, "gemma4", 1
        ));
        assert!(!should_gemma4_packed_ffn_compile_p128_for(
            true, "qwen3_5", 128
        ));
        assert!(!should_gemma4_packed_ffn_compile_p128_for(
            false, "gemma4", 128
        ));
    }

    #[test]
    fn qwen_gated_delta_decode_metal_uses_default_on_contract() {
        assert!(parse_bool_env_default_on(
            "AX_FASTPATH_TEST_QWEN_GATED_DELTA_DECODE_METAL_UNSET"
        ));
        assert!(!probe_default_on(
            "AX_FASTPATH_TEST_QWEN_GATED_DELTA_DECODE_METAL_DISABLED",
            "0"
        ));
        assert!(probe_default_on(
            "AX_FASTPATH_TEST_QWEN_GATED_DELTA_DECODE_METAL_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_gated_delta_prefill_contiguous_is_seq_gated() {
        assert!(should_qwen_gated_delta_prefill_contiguous_for(true, 1024));
        assert!(should_qwen_gated_delta_prefill_contiguous_for(true, 2));
        assert!(
            !should_qwen_gated_delta_prefill_contiguous_for(true, 1),
            "decode already uses a contiguous row-0 path"
        );
        assert!(!should_qwen_gated_delta_prefill_contiguous_for(false, 1024));
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_GATED_DELTA_PREFILL_CONTIGUOUS_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_GATED_DELTA_PREFILL_CONTIGUOUS_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_GATED_DELTA_PREFILL_CONTIGUOUS_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_la_fused_qkvz_ba_qmm_is_seq_and_quant_gated() {
        assert!(should_qwen_la_fused_qkvz_ba_qmm_for(true, 1024, true));
        assert!(should_qwen_la_fused_qkvz_ba_qmm_for(true, 2, true));
        assert!(
            !should_qwen_la_fused_qkvz_ba_qmm_for(true, 1, true),
            "decode keeps matching-bits two-qmm packing"
        );
        assert!(!should_qwen_la_fused_qkvz_ba_qmm_for(true, 1024, false));
        assert!(!should_qwen_la_fused_qkvz_ba_qmm_for(false, 1024, true));
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_LA_FUSED_QKVZ_BA_QMM_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_LA_FUSED_QKVZ_BA_QMM_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_LA_FUSED_QKVZ_BA_QMM_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_prefill_down_compile_is_seq_and_leading_gated() {
        assert!(should_qwen_prefill_down_compile_for(true, 1024, 128));
        assert!(should_qwen_prefill_down_compile_for(true, 2, 2048));
        assert!(!should_qwen_prefill_down_compile_for(true, 1, 128));
        assert!(!should_qwen_prefill_down_compile_for(true, 1024, 64));
        assert!(!should_qwen_prefill_down_compile_for(false, 1024, 128));
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_PREFILL_DOWN_COMPILE_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_PREFILL_DOWN_COMPILE_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_PREFILL_DOWN_COMPILE_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_prefill_chunk_1536_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_PREFILL_CHUNK_1536_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_PREFILL_CHUNK_1536_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_PREFILL_CHUNK_1536_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_prefill_chunk_1280_uses_opt_in_contract() {
        assert!(
            !qwen_prefill_chunk_1280_enabled(),
            "closed 1280 chunk stays default-off"
        );
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_PREFILL_CHUNK_1280_ENABLED",
            "1"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_PREFILL_CHUNK_1280_DISABLED",
            "0"
        ));
    }

    #[test]
    fn qwen_compiled_gated_delta_prefill_is_seq_gated() {
        assert!(should_qwen_compiled_gated_delta_prefill_for(true, 1024));
        assert!(should_qwen_compiled_gated_delta_prefill_for(true, 2));
        assert!(!should_qwen_compiled_gated_delta_prefill_for(true, 1));
        assert!(!should_qwen_compiled_gated_delta_prefill_for(false, 1024));
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_COMPILED_GATED_DELTA_PREFILL_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_COMPILED_GATED_DELTA_PREFILL_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_COMPILED_GATED_DELTA_PREFILL_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_packed_la_inputs_compile_is_seq_gated() {
        assert!(should_qwen_packed_la_inputs_compile_for(true, 1024));
        assert!(should_qwen_packed_la_inputs_compile_for(true, 2048));
        assert!(
            !should_qwen_packed_la_inputs_compile_for(true, 512),
            "512-token packed LA compile stays closed"
        );
        assert!(!should_qwen_packed_la_inputs_compile_for(true, 1));
        assert!(!should_qwen_packed_la_inputs_compile_for(false, 1024));
    }

    #[test]
    fn qwen_la_post_input_compile_is_seq_gated() {
        assert!(should_qwen_la_post_input_compile_for(true, 1024));
        assert!(should_qwen_la_post_input_compile_for(true, 2048));
        assert!(
            !should_qwen_la_post_input_compile_for(true, 512),
            "512-token post-input compile stays closed"
        );
        assert!(!should_qwen_la_post_input_compile_for(true, 1));
        assert!(!should_qwen_la_post_input_compile_for(false, 1024));
    }

    #[test]
    fn qwen_la_dual_stream_qkvz_ba_is_seq_gated() {
        assert!(should_qwen_la_dual_stream_qkvz_ba_for(true, 1024));
        assert!(should_qwen_la_dual_stream_qkvz_ba_for(true, 2048));
        assert!(
            !should_qwen_la_dual_stream_qkvz_ba_for(true, 512),
            "512-token LA dual-stream stays closed"
        );
        assert!(!should_qwen_la_dual_stream_qkvz_ba_for(true, 1));
        assert!(!should_qwen_la_dual_stream_qkvz_ba_for(false, 1024));
    }

    #[test]
    fn qwen_la_flat_inputs_is_seq_gated() {
        assert!(should_qwen_la_flat_inputs_for(true, 1024));
        assert!(should_qwen_la_flat_inputs_for(true, 2048));
        assert!(
            !should_qwen_la_flat_inputs_for(true, 512),
            "512-token LA flatten stays closed"
        );
        assert!(!should_qwen_la_flat_inputs_for(true, 1));
        assert!(!should_qwen_la_flat_inputs_for(false, 1024));
    }

    #[test]
    fn qwen_prefill_contiguous_la_input_is_seq_and_family_gated() {
        assert!(should_qwen_prefill_contiguous_la_input_for(
            true, "qwen3_5", 1024
        ));
        assert!(should_qwen_prefill_contiguous_la_input_for(
            true,
            "qwen3_next",
            2048
        ));
        assert!(
            !should_qwen_prefill_contiguous_la_input_for(true, "qwen3_5", 512),
            "512-token LA input contiguous stays closed"
        );
        assert!(!should_qwen_prefill_contiguous_la_input_for(
            true, "qwen3_5", 1
        ));
        assert!(!should_qwen_prefill_contiguous_la_input_for(
            true, "gemma4", 1024
        ));
        assert!(!should_qwen_prefill_contiguous_la_input_for(
            false, "qwen3_5", 1024
        ));
    }

    #[test]
    fn qwen_la_contiguous_qkv_is_seq_gated() {
        assert!(should_qwen_la_contiguous_qkv_for(true, 1024));
        assert!(should_qwen_la_contiguous_qkv_for(true, 2048));
        assert!(
            !should_qwen_la_contiguous_qkv_for(true, 512),
            "512-token LA qkv contiguous stays closed"
        );
        assert!(!should_qwen_la_contiguous_qkv_for(true, 1));
        assert!(!should_qwen_la_contiguous_qkv_for(false, 1024));
    }

    #[test]
    fn qwen_la_prefill_q2_is_seq_gated() {
        assert!(should_qwen_la_prefill_q2_for(true, 1024));
        assert!(should_qwen_la_prefill_q2_for(true, 2048));
        assert!(
            !should_qwen_la_prefill_q2_for(true, 512),
            "512-token LA q2 overlay stays closed"
        );
        assert!(!should_qwen_la_prefill_q2_for(true, 1));
        assert!(!should_qwen_la_prefill_q2_for(false, 1024));
    }

    #[test]
    fn qwen_prefill_q2_down_is_seq_gated() {
        assert!(should_qwen_prefill_q2_down_for(true, 1024));
        assert!(should_qwen_prefill_q2_down_for(true, 2048));
        assert!(
            !should_qwen_prefill_q2_down_for(true, 512),
            "512-token FFN down q2 overlay stays closed"
        );
        assert!(!should_qwen_prefill_q2_down_for(true, 1));
        assert!(!should_qwen_prefill_q2_down_for(false, 1024));
    }

    #[test]
    fn qwen_gd_prefill_chunkwise_is_seq_gated() {
        assert!(should_qwen_gd_prefill_chunkwise_for(true, 1024));
        assert!(should_qwen_gd_prefill_chunkwise_for(true, 2048));
        assert!(
            !should_qwen_gd_prefill_chunkwise_for(true, 512),
            "512-token GD chunkwise stays on the oneshot TG path"
        );
        assert!(!should_qwen_gd_prefill_chunkwise_for(true, 1));
        assert!(!should_qwen_gd_prefill_chunkwise_for(false, 1024));
    }

    #[test]
    fn qwen_prefill_ffn_gs64_is_seq_gated() {
        assert!(should_qwen_prefill_ffn_gs64_for(true, 1024));
        assert!(should_qwen_prefill_ffn_gs64_for(true, 2048));
        assert!(
            !should_qwen_prefill_ffn_gs64_for(true, 512),
            "512-token FFN gs64 overlay stays closed"
        );
        assert!(!should_qwen_prefill_ffn_gs64_for(true, 1));
        assert!(!should_qwen_prefill_ffn_gs64_for(false, 1024));
    }

    #[test]
    fn qwen_prefill_q3_ffn_is_seq_gated() {
        assert!(should_qwen_prefill_q3_ffn_for(true, 1024));
        assert!(should_qwen_prefill_q3_ffn_for(true, 2048));
        assert!(
            !should_qwen_prefill_q3_ffn_for(true, 512),
            "512-token FFN q3 overlay stays closed"
        );
        assert!(!should_qwen_prefill_q3_ffn_for(true, 1));
        assert!(!should_qwen_prefill_q3_ffn_for(false, 1024));
    }

    #[test]
    fn qwen_prefill_contiguous_ffn_weights_is_seq_gated() {
        assert!(should_qwen_prefill_contiguous_ffn_weights_for(true, 1024));
        assert!(should_qwen_prefill_contiguous_ffn_weights_for(true, 2048));
        assert!(
            !should_qwen_prefill_contiguous_ffn_weights_for(true, 512),
            "512-token FFN weight contiguous stays closed"
        );
        assert!(!should_qwen_prefill_contiguous_ffn_weights_for(true, 1));
        assert!(!should_qwen_prefill_contiguous_ffn_weights_for(false, 1024));
    }

    #[test]
    fn qwen_prefill_async_gate_up_is_seq_gated() {
        assert!(should_qwen_prefill_async_gate_up_for(true, 1024));
        assert!(should_qwen_prefill_async_gate_up_for(true, 2048));
        assert!(
            !should_qwen_prefill_async_gate_up_for(true, 512),
            "512-token async gate/up stays closed"
        );
        assert!(!should_qwen_prefill_async_gate_up_for(true, 1));
        assert!(!should_qwen_prefill_async_gate_up_for(false, 1024));
    }

    #[test]
    fn qwen_prefill_ffn_f32_input_is_seq_gated() {
        assert!(should_qwen_prefill_ffn_f32_input_for(true, 1024));
        assert!(should_qwen_prefill_ffn_f32_input_for(true, 2048));
        assert!(
            !should_qwen_prefill_ffn_f32_input_for(true, 512),
            "512-token FFN f32 input stays closed"
        );
        assert!(!should_qwen_prefill_ffn_f32_input_for(true, 1));
        assert!(!should_qwen_prefill_ffn_f32_input_for(false, 1024));
    }

    #[test]
    fn qwen_prefill_eval_ffn_input_is_seq_gated() {
        assert!(should_qwen_prefill_eval_ffn_input_for(true, 1024));
        assert!(should_qwen_prefill_eval_ffn_input_for(true, 2048));
        assert!(
            !should_qwen_prefill_eval_ffn_input_for(true, 512),
            "512-token FFN input eval stays closed"
        );
        assert!(!should_qwen_prefill_eval_ffn_input_for(true, 1));
        assert!(!should_qwen_prefill_eval_ffn_input_for(false, 1024));
    }

    #[test]
    fn qwen_prefill_eval_la_input_is_seq_gated() {
        assert!(should_qwen_prefill_eval_la_input_for(true, 1024));
        assert!(should_qwen_prefill_eval_la_input_for(true, 2048));
        assert!(
            !should_qwen_prefill_eval_la_input_for(true, 512),
            "512-token LA input eval stays closed"
        );
        assert!(!should_qwen_prefill_eval_la_input_for(true, 1));
        assert!(!should_qwen_prefill_eval_la_input_for(false, 1024));
    }

    #[test]
    fn qwen_prefill_async_la_outputs_is_seq_gated() {
        assert!(should_qwen_prefill_async_la_outputs_for(true, 1024));
        assert!(should_qwen_prefill_async_la_outputs_for(true, 2048));
        assert!(
            !should_qwen_prefill_async_la_outputs_for(true, 512),
            "512-token async LA outputs stays closed"
        );
        assert!(!should_qwen_prefill_async_la_outputs_for(true, 1));
        assert!(!should_qwen_prefill_async_la_outputs_for(false, 1024));
    }

    #[test]
    fn qwen_prefill_async_packed_gate_up_is_seq_gated() {
        assert!(should_qwen_prefill_async_packed_gate_up_for(true, 1024));
        assert!(should_qwen_prefill_async_packed_gate_up_for(true, 2048));
        assert!(
            !should_qwen_prefill_async_packed_gate_up_for(true, 512),
            "512-token async packed gate/up stays closed"
        );
        assert!(!should_qwen_prefill_async_packed_gate_up_for(true, 1));
        assert!(!should_qwen_prefill_async_packed_gate_up_for(false, 1024));
    }

    #[test]
    fn qwen_prefill_contiguous_la_weights_is_seq_gated() {
        assert!(should_qwen_prefill_contiguous_la_weights_for(true, 1024));
        assert!(should_qwen_prefill_contiguous_la_weights_for(true, 2048));
        assert!(
            !should_qwen_prefill_contiguous_la_weights_for(true, 512),
            "512-token LA weight contiguous stays closed"
        );
        assert!(!should_qwen_prefill_contiguous_la_weights_for(true, 1));
        assert!(!should_qwen_prefill_contiguous_la_weights_for(false, 1024));
    }

    #[test]
    fn qwen_prefill_eval_attn_input_is_seq_and_family_gated() {
        assert!(should_qwen_prefill_eval_attn_input_for(
            true, "qwen3_5", 1024
        ));
        assert!(should_qwen_prefill_eval_attn_input_for(
            true,
            "qwen3_next",
            2048
        ));
        assert!(
            !should_qwen_prefill_eval_attn_input_for(true, "qwen3_5", 512),
            "512-token attn input eval stays closed"
        );
        assert!(!should_qwen_prefill_eval_attn_input_for(true, "qwen3_5", 1));
        assert!(!should_qwen_prefill_eval_attn_input_for(
            true, "gemma4", 1024
        ));
        assert!(!should_qwen_prefill_eval_attn_input_for(
            false, "qwen3_5", 1024
        ));
    }

    #[test]
    fn qwen_prefill_eval_ffn_hidden_is_seq_gated() {
        assert!(should_qwen_prefill_eval_ffn_hidden_for(true, 1024));
        assert!(should_qwen_prefill_eval_ffn_hidden_for(true, 2048));
        assert!(
            !should_qwen_prefill_eval_ffn_hidden_for(true, 512),
            "512-token FFN hidden eval stays closed"
        );
        assert!(!should_qwen_prefill_eval_ffn_hidden_for(true, 1));
        assert!(!should_qwen_prefill_eval_ffn_hidden_for(false, 1024));
    }

    #[test]
    fn qwen_prefill_contiguous_attn_weights_is_seq_and_family_gated() {
        assert!(should_qwen_prefill_contiguous_attn_weights_for(
            true, "qwen3_5", 1024
        ));
        assert!(should_qwen_prefill_contiguous_attn_weights_for(
            true,
            "qwen3_next",
            2048
        ));
        assert!(
            !should_qwen_prefill_contiguous_attn_weights_for(true, "qwen3_5", 512),
            "512-token attn weight contiguous stays closed"
        );
        assert!(!should_qwen_prefill_contiguous_attn_weights_for(
            true, "qwen3_5", 1
        ));
        assert!(!should_qwen_prefill_contiguous_attn_weights_for(
            true, "gemma4", 1024
        ));
        assert!(!should_qwen_prefill_contiguous_attn_weights_for(
            false, "qwen3_5", 1024
        ));
    }

    #[test]
    fn qwen_prefill_skip_unused_la_out_is_seq_family_and_skip_gated() {
        assert!(should_qwen_prefill_skip_unused_la_out_for(
            true, "qwen3_5", true, 1024
        ));
        assert!(should_qwen_prefill_skip_unused_la_out_for(
            true,
            "qwen3_next",
            true,
            2048
        ));
        assert!(
            !should_qwen_prefill_skip_unused_la_out_for(true, "qwen3_5", true, 512),
            "512-token unused LA out skip stays closed"
        );
        assert!(!should_qwen_prefill_skip_unused_la_out_for(
            true, "qwen3_5", false, 1024
        ));
        assert!(!should_qwen_prefill_skip_unused_la_out_for(
            true, "gemma4", true, 1024
        ));
        assert!(!should_qwen_prefill_skip_unused_la_out_for(
            false, "qwen3_5", true, 1024
        ));
    }

    #[test]
    fn qwen_prefill_async_down_is_seq_gated() {
        assert!(should_qwen_prefill_async_down_for(true, 1024));
        assert!(should_qwen_prefill_async_down_for(true, 2048));
        assert!(
            !should_qwen_prefill_async_down_for(true, 512),
            "512-token async down stays closed"
        );
        assert!(!should_qwen_prefill_async_down_for(true, 1));
        assert!(!should_qwen_prefill_async_down_for(false, 1024));
    }

    #[test]
    fn qwen_prefill_last_query_q_proj_is_seq_family_and_last_only_gated() {
        assert!(should_qwen_prefill_last_query_q_proj_for(
            true, "qwen3_5", true, 1024
        ));
        assert!(should_qwen_prefill_last_query_q_proj_for(
            true,
            "qwen3_next",
            true,
            2048
        ));
        assert!(!should_qwen_prefill_last_query_q_proj_for(
            true, "qwen3_5", true, 512
        ));
        assert!(!should_qwen_prefill_last_query_q_proj_for(
            true, "qwen3_5", false, 1024
        ));
        assert!(!should_qwen_prefill_last_query_q_proj_for(
            true, "gemma4", true, 1024
        ));
        assert!(!should_qwen_prefill_last_query_q_proj_for(
            false, "qwen3_5", true, 1024
        ));
        assert!(
            should_qwen_prefill_last_query_sdpa_for(
                should_qwen_prefill_last_query_q_proj_for(true, "qwen3_5", true, 1024),
                "qwen3_5",
                true,
                1024,
            ),
            "last-token Q is already S=1; SDPA length must follow Q"
        );
    }

    #[test]
    fn qwen_prefill_skip_unused_qk_norm_is_seq_family_and_last_only_gated() {
        assert!(should_qwen_prefill_skip_unused_qk_norm_for(
            true, "qwen3_5", true, 1024
        ));
        assert!(should_qwen_prefill_skip_unused_qk_norm_for(
            true,
            "qwen3_next",
            true,
            2048
        ));
        assert!(!should_qwen_prefill_skip_unused_qk_norm_for(
            true, "qwen3_5", true, 512
        ));
        assert!(!should_qwen_prefill_skip_unused_qk_norm_for(
            true, "qwen3_5", false, 1024
        ));
        assert!(!should_qwen_prefill_skip_unused_qk_norm_for(
            true, "gemma4", true, 1024
        ));
        assert!(!should_qwen_prefill_skip_unused_qk_norm_for(
            false, "qwen3_5", true, 1024
        ));
        assert!(
            should_qwen_prefill_last_query_sdpa_for(
                should_qwen_prefill_skip_unused_qk_norm_for(true, "qwen3_5", true, 1024),
                "qwen3_5",
                true,
                1024,
            ),
            "prefix QK-norm skip leaves S=1 Q; SDPA length must follow Q"
        );
    }

    #[test]
    fn qwen_prefill_last_query_sdpa_is_seq_family_and_last_only_gated() {
        assert!(should_qwen_prefill_last_query_sdpa_for(
            true, "qwen3_5", true, 1024
        ));
        assert!(should_qwen_prefill_last_query_sdpa_for(
            true,
            "qwen3_next",
            true,
            2048
        ));
        assert!(
            !should_qwen_prefill_last_query_sdpa_for(true, "qwen3_5", true, 512),
            "512-token last-query SDPA stays closed"
        );
        assert!(!should_qwen_prefill_last_query_sdpa_for(
            true, "qwen3_5", false, 1024
        ));
        assert!(!should_qwen_prefill_last_query_sdpa_for(
            true, "gemma4", true, 1024
        ));
        assert!(!should_qwen_prefill_last_query_sdpa_for(
            false, "qwen3_5", true, 1024
        ));
    }

    #[test]
    fn qwen_prefill_last_token_o_proj_is_seq_family_and_last_only_gated() {
        assert!(should_qwen_prefill_last_token_o_proj_for(
            true, "qwen3_5", true, 1024
        ));
        assert!(should_qwen_prefill_last_token_o_proj_for(
            true,
            "qwen3_next",
            true,
            2048
        ));
        assert!(
            !should_qwen_prefill_last_token_o_proj_for(true, "qwen3_5", true, 512),
            "512-token last-token o_proj stays closed"
        );
        assert!(!should_qwen_prefill_last_token_o_proj_for(
            true, "qwen3_5", false, 1024
        ));
        assert!(!should_qwen_prefill_last_token_o_proj_for(
            true, "gemma4", true, 1024
        ));
        assert!(!should_qwen_prefill_last_token_o_proj_for(
            false, "qwen3_5", true, 1024
        ));
    }

    #[test]
    fn qwen_prefill_reuse_rope_is_seq_and_family_gated() {
        assert!(should_qwen_prefill_reuse_rope_for(true, "qwen3_5", 1024));
        assert!(should_qwen_prefill_reuse_rope_for(true, "qwen3_next", 2048));
        assert!(
            !should_qwen_prefill_reuse_rope_for(true, "qwen3_5", 512),
            "512-token rope reuse stays closed"
        );
        assert!(!should_qwen_prefill_reuse_rope_for(true, "qwen3_5", 1));
        assert!(!should_qwen_prefill_reuse_rope_for(true, "gemma4", 1024));
        assert!(!should_qwen_prefill_reuse_rope_for(false, "qwen3_5", 1024));
    }

    #[test]
    fn qwen_prefill_async_sdpa_is_seq_and_family_gated() {
        assert!(should_qwen_prefill_async_sdpa_for(true, "qwen3_5", 1024));
        assert!(should_qwen_prefill_async_sdpa_for(true, "qwen3_next", 2048));
        assert!(
            !should_qwen_prefill_async_sdpa_for(true, "qwen3_5", 512),
            "512-token async SDPA stays closed"
        );
        assert!(!should_qwen_prefill_async_sdpa_for(true, "qwen3_5", 1));
        assert!(!should_qwen_prefill_async_sdpa_for(true, "gemma4", 1024));
        assert!(!should_qwen_prefill_async_sdpa_for(false, "qwen3_5", 1024));
    }

    #[test]
    fn qwen_prefill_async_gd_is_seq_gated() {
        assert!(should_qwen_prefill_async_gd_for(true, 1024));
        assert!(should_qwen_prefill_async_gd_for(true, 2048));
        assert!(
            !should_qwen_prefill_async_gd_for(true, 512),
            "512-token async GD stays closed"
        );
        assert!(!should_qwen_prefill_async_gd_for(true, 1));
        assert!(!should_qwen_prefill_async_gd_for(false, 1024));
    }

    #[test]
    fn qwen_prefill_eval_gd_is_seq_gated() {
        assert!(should_qwen_prefill_eval_gd_for(true, 1024));
        assert!(should_qwen_prefill_eval_gd_for(true, 2048));
        assert!(
            !should_qwen_prefill_eval_gd_for(true, 512),
            "512-token eval GD stays closed"
        );
        assert!(!should_qwen_prefill_eval_gd_for(true, 1));
        assert!(!should_qwen_prefill_eval_gd_for(false, 1024));
    }

    #[test]
    fn qwen_prefill_contiguous_gd_is_seq_gated() {
        assert!(should_qwen_prefill_contiguous_gd_for(true, 1024));
        assert!(should_qwen_prefill_contiguous_gd_for(true, 2048));
        assert!(
            !should_qwen_prefill_contiguous_gd_for(true, 512),
            "512-token contiguous GD stays closed"
        );
        assert!(!should_qwen_prefill_contiguous_gd_for(true, 1));
        assert!(!should_qwen_prefill_contiguous_gd_for(false, 1024));
    }

    #[test]
    fn qwen_prefill_split_packed_is_seq_and_family_gated() {
        assert!(should_qwen_prefill_split_packed_for(true, "qwen3_5", 1024));
        assert!(should_qwen_prefill_split_packed_for(
            true,
            "qwen3_next",
            2048
        ));
        assert!(
            !should_qwen_prefill_split_packed_for(true, "qwen3_5", 512),
            "512-token split-packed stays closed"
        );
        assert!(!should_qwen_prefill_split_packed_for(true, "qwen3_5", 1));
        assert!(!should_qwen_prefill_split_packed_for(true, "gemma4", 1024));
        assert!(!should_qwen_prefill_split_packed_for(
            false, "qwen3_5", 1024
        ));
    }

    #[test]
    fn qwen_prefill_dequant_dense_is_seq_and_family_gated() {
        assert!(should_qwen_prefill_dequant_dense_for(true, "qwen3_5", 1024));
        assert!(should_qwen_prefill_dequant_dense_for(
            true,
            "qwen3_next",
            2048
        ));
        assert!(
            !should_qwen_prefill_dequant_dense_for(true, "qwen3_5", 512),
            "512-token dequant-dense stays closed"
        );
        assert!(!should_qwen_prefill_dequant_dense_for(true, "qwen3_5", 1));
        assert!(!should_qwen_prefill_dequant_dense_for(true, "gemma4", 1024));
        assert!(!should_qwen_prefill_dequant_dense_for(
            false, "qwen3_5", 1024
        ));
    }

    #[test]
    fn qwen_la_norm_qkvz_fuse_is_seq_and_family_gated() {
        assert!(should_qwen_la_norm_qkvz_fuse_for(true, "qwen3_5", 1024));
        assert!(should_qwen_la_norm_qkvz_fuse_for(true, "qwen3_next", 2048));
        assert!(
            !should_qwen_la_norm_qkvz_fuse_for(true, "qwen3_5", 512),
            "512-token LA norm fuse stays closed"
        );
        assert!(!should_qwen_la_norm_qkvz_fuse_for(true, "qwen3_5", 1));
        assert!(!should_qwen_la_norm_qkvz_fuse_for(true, "gemma4", 1024));
        assert!(!should_qwen_la_norm_qkvz_fuse_for(false, "qwen3_5", 1024));
    }

    #[test]
    fn qwen_prefill_skip_bf16_astype_is_seq_and_family_gated() {
        assert!(should_qwen_prefill_skip_bf16_astype_for(
            true, "qwen3_5", 1024
        ));
        assert!(should_qwen_prefill_skip_bf16_astype_for(
            true,
            "qwen3_next",
            2
        ));
        assert!(!should_qwen_prefill_skip_bf16_astype_for(
            true, "qwen3_5", 1
        ));
        assert!(!should_qwen_prefill_skip_bf16_astype_for(
            true, "gemma4", 1024
        ));
        assert!(!should_qwen_prefill_skip_bf16_astype_for(
            false, "qwen3_5", 1024
        ));
    }

    #[test]
    fn qwen_prefill_flat_qmm_is_seq_and_rank_gated() {
        assert!(should_qwen_prefill_flat_qmm_for(true, 1024, 3));
        assert!(should_qwen_prefill_flat_qmm_for(true, 2048, 3));
        assert!(!should_qwen_prefill_flat_qmm_for(true, 512, 3));
        assert!(!should_qwen_prefill_flat_qmm_for(true, 1024, 2));
        assert!(!should_qwen_prefill_flat_qmm_for(true, 1, 3));
        assert!(!should_qwen_prefill_flat_qmm_for(false, 1024, 3));
    }

    #[test]
    fn qwen_prefill_tile_qmm_is_seq_and_family_gated() {
        assert!(should_qwen_prefill_tile_qmm_for(true, "qwen3_5", 1024));
        assert!(should_qwen_prefill_tile_qmm_for(true, "qwen3_next", 2048));
        assert!(!should_qwen_prefill_tile_qmm_for(true, "qwen3_5", 512));
        assert!(!should_qwen_prefill_tile_qmm_for(true, "qwen3_5", 1));
        assert!(!should_qwen_prefill_tile_qmm_for(true, "gemma4", 1024));
        assert!(!should_qwen_prefill_tile_qmm_for(false, "qwen3_5", 1024));
    }

    #[test]
    fn qwen_prefill_dual_affine_qmm_is_seq_and_family_gated() {
        assert!(should_qwen_prefill_dual_affine_qmm_for(
            true, "qwen3_5", 1024
        ));
        assert!(should_qwen_prefill_dual_affine_qmm_for(
            true,
            "qwen3_next",
            2048
        ));
        assert!(!should_qwen_prefill_dual_affine_qmm_for(
            true, "qwen3_5", 512
        ));
        assert!(!should_qwen_prefill_dual_affine_qmm_for(
            true, "gemma4", 1024
        ));
        assert!(!should_qwen_prefill_dual_affine_qmm_for(
            false, "qwen3_5", 1024
        ));
    }

    #[test]
    fn qwen_prefill_skip_unused_embed_clip_is_seq_and_family_gated() {
        assert!(should_qwen_prefill_skip_unused_embed_clip_for(
            true, "qwen3_5", 1024
        ));
        assert!(should_qwen_prefill_skip_unused_embed_clip_for(
            true,
            "qwen3_next",
            2048
        ));
        assert!(!should_qwen_prefill_skip_unused_embed_clip_for(
            true, "qwen3_5", 512
        ));
        assert!(!should_qwen_prefill_skip_unused_embed_clip_for(
            true, "gemma4", 1024
        ));
        assert!(!should_qwen_prefill_skip_unused_embed_clip_for(
            false, "qwen3_5", 1024
        ));
    }

    #[test]
    fn qwen_prefill_skip_unused_f32_sdpa_is_seq_and_family_gated() {
        assert!(should_qwen_prefill_skip_unused_f32_sdpa_for(
            true, "qwen3_5", 1024
        ));
        assert!(should_qwen_prefill_skip_unused_f32_sdpa_for(
            true,
            "qwen3_next",
            2048
        ));
        assert!(
            should_qwen_prefill_skip_unused_f32_sdpa_for(true, "qwen3_5", 128),
            "contract p128 prefill must skip the f32 upcast"
        );
        assert!(should_qwen_prefill_skip_unused_f32_sdpa_for(
            true, "qwen3_5", 512
        ));
        assert!(
            !should_qwen_prefill_skip_unused_f32_sdpa_for(true, "qwen3_5", 8),
            "short MTP verify must keep f32 SDPA"
        );
        assert!(!should_qwen_prefill_skip_unused_f32_sdpa_for(
            true, "qwen3_5", 127
        ));
        assert!(
            should_qwen_prefill_skip_unused_f32_sdpa_for(true, "qwen3_vl_moe", 128),
            "VL-MoE text prefill shares the qwen full-attention graphs"
        );
        assert!(should_qwen_prefill_skip_unused_f32_sdpa_for(
            true, "qwen3_vl", 128
        ));
        assert!(!should_qwen_prefill_skip_unused_f32_sdpa_for(
            true, "gemma4", 1024
        ));
        assert!(!should_qwen_prefill_skip_unused_f32_sdpa_for(
            false, "qwen3_5", 1024
        ));
    }

    #[test]
    fn gemma4_prefill_skip_unused_f32_sdpa_is_seq_and_family_gated() {
        assert!(should_gemma4_prefill_skip_unused_f32_sdpa_for(
            true, "gemma4", 128
        ));
        assert!(should_gemma4_prefill_skip_unused_f32_sdpa_for(
            true,
            "gemma4_unified",
            128
        ));
        assert!(should_gemma4_prefill_skip_unused_f32_sdpa_for(
            true, "gemma4", 512
        ));
        assert!(
            !should_gemma4_prefill_skip_unused_f32_sdpa_for(true, "gemma4", 8),
            "short MTP verify must keep f32 SDPA"
        );
        assert!(!should_gemma4_prefill_skip_unused_f32_sdpa_for(
            true, "gemma4", 1
        ));
        assert!(!should_gemma4_prefill_skip_unused_f32_sdpa_for(
            true, "qwen3_5", 128
        ));
        assert!(!should_gemma4_prefill_skip_unused_f32_sdpa_for(
            false, "gemma4", 128
        ));
    }

    #[test]
    fn gemma4_prefill_skip_unused_embed_clip_is_seq_and_family_gated() {
        assert!(should_gemma4_prefill_skip_unused_embed_clip_for(
            true, "gemma4", 128
        ));
        assert!(should_gemma4_prefill_skip_unused_embed_clip_for(
            true,
            "gemma4_unified",
            128
        ));
        assert!(should_gemma4_prefill_skip_unused_embed_clip_for(
            true, "gemma4", 2048
        ));
        assert!(
            !should_gemma4_prefill_skip_unused_embed_clip_for(true, "gemma4", 8),
            "short MTP verify must keep the embed clip"
        );
        assert!(!should_gemma4_prefill_skip_unused_embed_clip_for(
            true, "gemma4", 1
        ));
        assert!(!should_gemma4_prefill_skip_unused_embed_clip_for(
            true, "qwen3_5", 128
        ));
        assert!(!should_gemma4_prefill_skip_unused_embed_clip_for(
            false, "gemma4", 128
        ));
    }

    #[test]
    fn gemma4_prefill_bf16_embed_is_seq_and_family_gated() {
        assert!(should_gemma4_prefill_bf16_embed_for(true, "gemma4", 128));
        assert!(should_gemma4_prefill_bf16_embed_for(
            true,
            "gemma4_unified",
            128
        ));
        assert!(should_gemma4_prefill_bf16_embed_for(true, "gemma4", 2048));
        assert!(
            !should_gemma4_prefill_bf16_embed_for(true, "gemma4", 8),
            "short MTP verify must keep f32 embed dequant"
        );
        assert!(!should_gemma4_prefill_bf16_embed_for(true, "gemma4", 1));
        assert!(!should_gemma4_prefill_bf16_embed_for(true, "qwen3_5", 128));
        assert!(!should_gemma4_prefill_bf16_embed_for(false, "gemma4", 128));
    }

    #[test]
    fn gemma4_prefill_skip_unused_last_residual_is_seq_last_layer_and_family_gated() {
        assert!(
            should_gemma4_prefill_skip_unused_last_residual_for(true, "gemma4", true, 128),
            "shipped skip-unused-last-residual must accept contract p128 last layer"
        );
        assert!(should_gemma4_prefill_skip_unused_last_residual_for(
            true,
            "gemma4_unified",
            true,
            128
        ));
        assert!(should_gemma4_prefill_skip_unused_last_residual_for(
            true, "gemma4", true, 2048
        ));
        assert!(
            !should_gemma4_prefill_skip_unused_last_residual_for(true, "gemma4", false, 128),
            "non-final layers keep full-seq add_rms"
        );
        assert!(
            !should_gemma4_prefill_skip_unused_last_residual_for(true, "gemma4", true, 8),
            "short MTP verify keeps add-then-slice"
        );
        assert!(!should_gemma4_prefill_skip_unused_last_residual_for(
            true, "gemma4", true, 1
        ));
        assert!(!should_gemma4_prefill_skip_unused_last_residual_for(
            true, "qwen3_5", true, 128
        ));
        assert!(!should_gemma4_prefill_skip_unused_last_residual_for(
            false, "gemma4", true, 128
        ));
    }

    #[test]
    fn gemma4_prefill_skip_unused_last_ffn_packed_is_seq_last_layer_and_family_gated() {
        assert!(
            should_gemma4_prefill_skip_unused_last_ffn_packed_for(true, "gemma4", true, 128),
            "shipped skip-unused-last-ffn-packed must accept contract p128 last layer"
        );
        assert!(should_gemma4_prefill_skip_unused_last_ffn_packed_for(
            true,
            "gemma4_unified",
            true,
            128
        ));
        assert!(should_gemma4_prefill_skip_unused_last_ffn_packed_for(
            true, "gemma4", true, 2048
        ));
        assert!(
            !should_gemma4_prefill_skip_unused_last_ffn_packed_for(true, "gemma4", false, 128),
            "non-final layers keep packed/split prefill policy"
        );
        assert!(
            !should_gemma4_prefill_skip_unused_last_ffn_packed_for(true, "gemma4", true, 8),
            "short MTP verify keeps packed last-layer FFN"
        );
        assert!(!should_gemma4_prefill_skip_unused_last_ffn_packed_for(
            true, "gemma4", true, 1
        ));
        assert!(!should_gemma4_prefill_skip_unused_last_ffn_packed_for(
            true, "qwen3_5", true, 128
        ));
        assert!(!should_gemma4_prefill_skip_unused_last_ffn_packed_for(
            false, "gemma4", true, 128
        ));
    }

    #[test]
    fn gemma4_prefill_skip_unused_layer_masks_is_seq_window_and_family_gated() {
        assert!(should_gemma4_prefill_skip_unused_layer_masks_for(
            true,
            "gemma4",
            128,
            128,
            Some(1024),
            0
        ));
        assert!(should_gemma4_prefill_skip_unused_layer_masks_for(
            true,
            "gemma4_unified",
            512,
            512,
            Some(1024),
            0
        ));
        assert!(
            should_gemma4_prefill_skip_unused_layer_masks_for(true, "gemma4", 128, 128, None, 0),
            "full-attn offset-0 prefill is already maskless"
        );
        assert!(
            !should_gemma4_prefill_skip_unused_layer_masks_for(
                true,
                "gemma4",
                2048,
                2048,
                Some(1024),
                0
            ),
            "p2048 exceeds the 1024-token window and must keep the hoist"
        );
        assert!(
            !should_gemma4_prefill_skip_unused_layer_masks_for(true, "gemma4", 8, 8, Some(1024), 0),
            "short MTP verify must keep the hoist"
        );
        assert!(!should_gemma4_prefill_skip_unused_layer_masks_for(
            true,
            "gemma4",
            1,
            1,
            Some(1024),
            0
        ));
        assert!(!should_gemma4_prefill_skip_unused_layer_masks_for(
            true,
            "gemma4",
            128,
            256,
            Some(1024),
            0
        ));
        assert!(!should_gemma4_prefill_skip_unused_layer_masks_for(
            true,
            "gemma4",
            128,
            128,
            Some(1024),
            4
        ));
        assert!(!should_gemma4_prefill_skip_unused_layer_masks_for(
            true,
            "qwen3_5",
            128,
            128,
            Some(1024),
            0
        ));
        assert!(!should_gemma4_prefill_skip_unused_layer_masks_for(
            false,
            "gemma4",
            128,
            128,
            Some(1024),
            0
        ));
    }

    #[test]
    fn gemma4_prefill_pipeline_hint_p128_is_seq_layer_and_family_gated() {
        assert!(
            should_gemma4_prefill_pipeline_hint_p128_for(true, "gemma4", 128, 0, 48),
            "shipped p128 pipeline hint must fire after non-final layers"
        );
        assert!(should_gemma4_prefill_pipeline_hint_p128_for(
            true,
            "gemma4_unified",
            128,
            46,
            48
        ));
        assert!(
            !should_gemma4_prefill_pipeline_hint_p128_for(true, "gemma4", 128, 47, 48),
            "final layer stays lazy so logits eval owns the last barrier"
        );
        assert!(
            !should_gemma4_prefill_pipeline_hint_p128_for(true, "gemma4", 512, 0, 48),
            "p512 keeps full-graph fusion"
        );
        assert!(!should_gemma4_prefill_pipeline_hint_p128_for(
            true, "gemma4", 2048, 0, 48
        ));
        assert!(
            !should_gemma4_prefill_pipeline_hint_p128_for(true, "gemma4", 8, 0, 48),
            "short MTP verify stays lazy"
        );
        assert!(!should_gemma4_prefill_pipeline_hint_p128_for(
            true, "gemma4", 1, 0, 48
        ));
        assert!(!should_gemma4_prefill_pipeline_hint_p128_for(
            true, "qwen3_5", 128, 0, 48
        ));
        assert!(!should_gemma4_prefill_pipeline_hint_p128_for(
            false, "gemma4", 128, 0, 48
        ));
        assert!(
            !pipeline_hint_should_fire(0, 48),
            "global AX_MLX_PIPELINE_GRANULARITY stays off; only the Gemma p128 predicate fires"
        );
    }

    #[test]
    fn gemma4_prefill_last_query_p128_is_seq_last_layer_and_family_gated() {
        assert!(
            should_gemma4_prefill_last_query_p128_for(true, "gemma4", true, 128),
            "shipped last-query must accept contract p128 last layer"
        );
        assert!(should_gemma4_prefill_last_query_p128_for(
            true,
            "gemma4_unified",
            true,
            128
        ));
        assert!(
            !should_gemma4_prefill_last_query_p128_for(true, "gemma4", false, 128),
            "non-final layers must keep full-seq fused attention"
        );
        assert!(
            !should_gemma4_prefill_last_query_p128_for(true, "gemma4", true, 512),
            "p512 last layer stays on fused full-seq"
        );
        assert!(!should_gemma4_prefill_last_query_p128_for(
            true, "gemma4", true, 2048
        ));
        assert!(
            !should_gemma4_prefill_last_query_p128_for(true, "gemma4", true, 8),
            "short MTP verify keeps full-seq last-layer attention"
        );
        assert!(!should_gemma4_prefill_last_query_p128_for(
            true, "gemma4", true, 1
        ));
        assert!(!should_gemma4_prefill_last_query_p128_for(
            true, "qwen3_5", true, 128
        ));
        assert!(!should_gemma4_prefill_last_query_p128_for(
            false, "gemma4", true, 128
        ));
    }

    #[test]
    fn qwen_prefill_skip_unused_swiglu_compile_is_seq_and_family_gated() {
        assert!(should_qwen_prefill_skip_unused_swiglu_compile_for(
            true, "qwen3_5", 1024
        ));
        assert!(should_qwen_prefill_skip_unused_swiglu_compile_for(
            true,
            "qwen3_next",
            2048
        ));
        assert!(!should_qwen_prefill_skip_unused_swiglu_compile_for(
            true, "qwen3_5", 512
        ));
        assert!(!should_qwen_prefill_skip_unused_swiglu_compile_for(
            true, "gemma4", 1024
        ));
        assert!(!should_qwen_prefill_skip_unused_swiglu_compile_for(
            false, "qwen3_5", 1024
        ));
    }

    #[test]
    fn qwen_prefill_native_offset_causal_is_seq_and_family_gated() {
        assert!(should_qwen_prefill_native_offset_causal_for(
            true, "qwen3_5", 1024
        ));
        assert!(should_qwen_prefill_native_offset_causal_for(
            true,
            "qwen3_next",
            2048
        ));
        assert!(!should_qwen_prefill_native_offset_causal_for(
            true, "qwen3_5", 512
        ));
        assert!(!should_qwen_prefill_native_offset_causal_for(
            true, "gemma4", 1024
        ));
        assert!(!should_qwen_prefill_native_offset_causal_for(
            false, "qwen3_5", 1024
        ));
    }

    #[test]
    fn nax_attention_is_off_switch_only() {
        assert!(nax_attention_enabled_for(true, true));
        assert!(!nax_attention_enabled_for(false, true));
        assert!(!nax_attention_enabled_for(true, false));
        assert!(!nax_attention_enabled_for(false, false));
    }

    #[test]
    fn nax_attention_arms_qwen_native_offset_causal_on_m5() {
        let _hw =
            crate::hardware::override_hardware(crate::hardware::HardwareCapabilities::m5_na());
        assert!(nax_attention_enabled_for(
            true,
            crate::hardware::neural_accelerator_active()
        ));
        assert!(should_qwen_prefill_native_offset_causal_for(
            nax_attention_enabled(),
            "qwen3_5",
            1024
        ));
        assert!(should_qwen_prefill_native_offset_causal("qwen3_5", 1024));
        assert!(should_qwen_prefill_native_offset_causal("qwen3_next", 2048));
        assert!(!should_qwen_prefill_native_offset_causal("qwen3_5", 512));
        assert!(!should_qwen_prefill_native_offset_causal("gemma4", 1024));
    }

    #[test]
    fn nax_attention_does_not_arm_on_m4() {
        let _hw = crate::hardware::override_hardware(crate::hardware::HardwareCapabilities::m4());
        assert!(!nax_attention_enabled());
        if !qwen_prefill_native_offset_causal_enabled() {
            assert!(!should_qwen_prefill_native_offset_causal("qwen3_5", 1024));
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn live_nax_attention_follows_detected_hardware() {
        let active = crate::hardware::neural_accelerator_active();
        eprintln!(
            "live nax_attention: enabled={} allowed={} neural_accelerator_active={}",
            nax_attention_enabled(),
            nax_attention_allowed(),
            active
        );
        assert_eq!(
            nax_attention_enabled(),
            nax_attention_allowed() && active,
            "kill-switch AND hardware must both be true to arm NAX attention"
        );
        if nax_attention_enabled() {
            assert!(should_qwen_prefill_native_offset_causal("qwen3_5", 1024));
            assert!(should_qwen_prefill_native_offset_causal("qwen3_next", 2048));
            assert!(!should_qwen_prefill_native_offset_causal("qwen3_5", 512));
            assert!(!should_qwen_prefill_native_offset_causal("gemma4", 1024));
        }
    }

    #[test]
    fn qwen_prefill_bf16_embed_dequant_is_seq_and_family_gated() {
        assert!(should_qwen_prefill_bf16_embed_dequant_for(
            true, "qwen3_5", 1024
        ));
        assert!(should_qwen_prefill_bf16_embed_dequant_for(
            true,
            "qwen3_next",
            2048
        ));
        assert!(!should_qwen_prefill_bf16_embed_dequant_for(
            true, "qwen3_5", 512
        ));
        assert!(!should_qwen_prefill_bf16_embed_dequant_for(
            true, "gemma4", 1024
        ));
        assert!(!should_qwen_prefill_bf16_embed_dequant_for(
            false, "qwen3_5", 1024
        ));
    }

    #[test]
    fn qwen_prefill_async_embed_is_seq_and_family_gated() {
        assert!(should_qwen_prefill_async_embed_for(true, "qwen3_5", 1024));
        assert!(should_qwen_prefill_async_embed_for(
            true,
            "qwen3_next",
            1024
        ));
        assert!(!should_qwen_prefill_async_embed_for(true, "qwen3_5", 512));
        assert!(!should_qwen_prefill_async_embed_for(true, "qwen3_5", 1));
        assert!(!should_qwen_prefill_async_embed_for(true, "gemma4", 1024));
        assert!(!should_qwen_prefill_async_embed_for(false, "qwen3_5", 1024));
    }

    #[test]
    fn qwen_packed_ffn_prefill_compile_is_leading_gated() {
        assert!(should_qwen_packed_ffn_prefill_compile_for(
            true, "qwen3_5", 1024
        ));
        assert!(should_qwen_packed_ffn_prefill_compile_for(
            true, "QWEN3_5", 2048
        ));
        assert!(
            !should_qwen_packed_ffn_prefill_compile_for(true, "qwen3_5", 512),
            "512-token packed compile stays closed"
        );
        assert!(!should_qwen_packed_ffn_prefill_compile_for(
            true, "gemma4", 1024
        ));
        assert!(!should_qwen_packed_ffn_prefill_compile_for(
            false, "qwen3_5", 1024
        ));
    }

    #[test]
    fn qwen_compiled_qk_norm_rope_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_COMPILED_QK_NORM_ROPE_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_COMPILED_QK_NORM_ROPE_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_COMPILED_QK_NORM_ROPE_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_direct_cpp_qk_norm_rope_uses_default_on_contract() {
        assert!(parse_bool_env_default_on(
            "AX_FASTPATH_TEST_QWEN_DIRECT_CPP_QK_NORM_ROPE_UNSET"
        ));
        assert!(!probe_default_on(
            "AX_FASTPATH_TEST_QWEN_DIRECT_CPP_QK_NORM_ROPE_DISABLED",
            "0"
        ));
        assert!(probe_default_on(
            "AX_FASTPATH_TEST_QWEN_DIRECT_CPP_QK_NORM_ROPE_ENABLED",
            "1"
        ));
    }

    #[test]
    fn gemma_direct_cpp_qk_norm_rope_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_GEMMA_DIRECT_CPP_QK_NORM_ROPE_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_GEMMA_DIRECT_CPP_QK_NORM_ROPE_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_GEMMA_DIRECT_CPP_QK_NORM_ROPE_ENABLED",
            "1"
        ));
    }

    #[test]
    fn gemma_dual_gate_up_metal_uses_opt_in_contract() {
        // Pure-wall A/B on mbp-m5 measured ~8.5× regression when default-on;
        // production remains opt-in only.
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_GEMMA_DUAL_GATE_UP_METAL_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_GEMMA_DUAL_GATE_UP_METAL_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_GEMMA_DUAL_GATE_UP_METAL_ENABLED",
            "1"
        ));
    }

    #[test]
    fn o_proj_qmatmul_rms_norm_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_O_PROJ_QMATMUL_RMS_NORM_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_O_PROJ_QMATMUL_RMS_NORM_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_O_PROJ_QMATMUL_RMS_NORM_ENABLED",
            "1"
        ));
    }

    #[test]
    fn attn_norm_qkv_fuse_uses_opt_in_contract() {
        assert!(!parse_bool_env("AX_FASTPATH_TEST_ATTN_NORM_QKV_FUSE_UNSET"));
        assert!(!probe("AX_FASTPATH_TEST_ATTN_NORM_QKV_FUSE_DISABLED", "0"));
        assert!(probe("AX_FASTPATH_TEST_ATTN_NORM_QKV_FUSE_ENABLED", "1"));
    }

    #[test]
    fn qwen_attn_norm_qkv_fuse_is_family_scoped_and_opt_in() {
        assert!(should_attn_norm_qkv_fuse_for(
            true, false, false, "qwen3_5", 128
        ));
        assert!(should_attn_norm_qkv_fuse_for(
            true, false, false, "QWEN3_5", 2048
        ));
        assert!(
            !should_attn_norm_qkv_fuse_for(false, false, false, "qwen3_5", 128),
            "Qwen kill-switch must disable the fuse"
        );
        assert!(
            !should_attn_norm_qkv_fuse_for(true, false, false, "gemma4", 512),
            "Gemma p512 stays on the global default-OFF flag"
        );
        assert!(should_attn_norm_qkv_fuse_for(
            false, true, false, "gemma4", 512
        ));
        assert!(should_call_attn_norm_qkv_fuse(true, true, false, false));
        assert!(
            !should_call_attn_norm_qkv_fuse(true, true, false, true),
            "exact / moe-mt skip must keep standalone attn_norm"
        );
        assert!(!should_call_attn_norm_qkv_fuse(true, false, false, false));
        assert!(!should_call_attn_norm_qkv_fuse(true, true, true, false));
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_ATTN_NORM_QKV_FUSE_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_ATTN_NORM_QKV_FUSE_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_ATTN_NORM_QKV_FUSE_ENABLED",
            "1"
        ));
    }

    #[test]
    fn gemma4_attn_norm_qkv_fuse_p128_is_seq_and_family_gated() {
        assert!(should_gemma4_attn_norm_qkv_fuse_p128_for(
            true, "gemma4", 128
        ));
        assert!(should_gemma4_attn_norm_qkv_fuse_p128_for(
            true,
            "gemma4_unified",
            128
        ));
        assert!(should_attn_norm_qkv_fuse_for(
            false, false, true, "gemma4", 128
        ));
        assert!(
            !should_gemma4_attn_norm_qkv_fuse_p128_for(true, "gemma4", 512),
            "p512 must stay portable so the p128 fuse cannot regress longer cells"
        );
        assert!(!should_gemma4_attn_norm_qkv_fuse_p128_for(
            true, "gemma4", 2048
        ));
        assert!(!should_gemma4_attn_norm_qkv_fuse_p128_for(
            true, "gemma4", 1
        ));
        assert!(!should_gemma4_attn_norm_qkv_fuse_p128_for(
            true, "qwen3_5", 128
        ));
        assert!(!should_gemma4_attn_norm_qkv_fuse_p128_for(
            false, "gemma4", 128
        ));
        assert!(!should_attn_norm_qkv_fuse_for(
            false, false, true, "gemma4", 512
        ));
    }

    #[test]
    fn native_offset_causal_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_NATIVE_OFFSET_CAUSAL_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_NATIVE_OFFSET_CAUSAL_DISABLED",
            "0"
        ));
        assert!(probe("AX_FASTPATH_TEST_NATIVE_OFFSET_CAUSAL_ENABLED", "1"));
    }

    #[test]
    fn dual_qmm_geglu_uses_opt_in_contract() {
        assert!(!parse_bool_env("AX_FASTPATH_TEST_DUAL_QMM_GEGLU_UNSET"));
        assert!(!probe("AX_FASTPATH_TEST_DUAL_QMM_GEGLU_DISABLED", "0"));
        assert!(probe("AX_FASTPATH_TEST_DUAL_QMM_GEGLU_ENABLED", "1"));
    }

    #[test]
    fn cache_only_chunk_eval_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_CACHE_ONLY_CHUNK_EVAL_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_CACHE_ONLY_CHUNK_EVAL_DISABLED",
            "0"
        ));
        assert!(probe("AX_FASTPATH_TEST_CACHE_ONLY_CHUNK_EVAL_ENABLED", "1"));
    }

    #[test]
    fn cache_only_chunk_async_eval_only_for_non_final_under_both_flags() {
        // Both off / either off → never async.
        assert!(!cache_only_chunk_should_async_eval_for(false, false, false));
        assert!(!cache_only_chunk_should_async_eval_for(true, false, false));
        assert!(!cache_only_chunk_should_async_eval_for(false, true, false));
        // Final chunk always blocks even when both flags are on.
        assert!(!cache_only_chunk_should_async_eval_for(true, true, true));
        // Intermediate chunk under both flags → async.
        assert!(cache_only_chunk_should_async_eval_for(true, true, false));
    }

    #[test]
    fn prefill_clear_cache_per_chunk_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_PREFILL_CLEAR_CACHE_PER_CHUNK_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_PREFILL_CLEAR_CACHE_PER_CHUNK_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_PREFILL_CLEAR_CACHE_PER_CHUNK_ENABLED",
            "1"
        ));
    }

    #[test]
    fn parse_pipeline_granularity_matches_mlxcel_contract() {
        assert_eq!(parse_pipeline_granularity(""), PipelineGranularity::Off);
        assert_eq!(parse_pipeline_granularity("off"), PipelineGranularity::Off);
        assert_eq!(parse_pipeline_granularity("OFF"), PipelineGranularity::Off);
        assert_eq!(
            parse_pipeline_granularity("layer"),
            PipelineGranularity::PerLayer
        );
        assert_eq!(
            parse_pipeline_granularity("LAYER"),
            PipelineGranularity::PerLayer
        );
        assert_eq!(
            parse_pipeline_granularity("block:4"),
            PipelineGranularity::PerBlock(4)
        );
        assert_eq!(
            parse_pipeline_granularity("block:1"),
            PipelineGranularity::PerBlock(1)
        );
        assert_eq!(
            parse_pipeline_granularity("block:0"),
            PipelineGranularity::PerBlock(1),
            "N=0 clamps to 1"
        );
        assert_eq!(
            parse_pipeline_granularity("block:xyz"),
            PipelineGranularity::PerBlock(4),
            "invalid N falls back to 4"
        );
        assert_eq!(
            parse_pipeline_granularity("garbage"),
            PipelineGranularity::Off
        );
    }

    #[test]
    fn parse_pipeline_eval_granularity_is_strict_and_case_insensitive() {
        assert_eq!(
            parse_pipeline_eval_granularity(""),
            PipelineEvalGranularity::Off
        );
        assert_eq!(
            parse_pipeline_eval_granularity(" OFF "),
            PipelineEvalGranularity::Off
        );
        assert_eq!(
            parse_pipeline_eval_granularity("layer"),
            PipelineEvalGranularity::PerLayer
        );
        assert_eq!(
            parse_pipeline_eval_granularity("LAYER"),
            PipelineEvalGranularity::PerLayer
        );
        assert_eq!(
            parse_pipeline_eval_granularity(" sublayer "),
            PipelineEvalGranularity::Sublayer
        );
        assert_eq!(
            parse_pipeline_eval_granularity("SUBLAYER"),
            PipelineEvalGranularity::Sublayer
        );
        assert_eq!(
            parse_pipeline_eval_granularity("block:4"),
            PipelineEvalGranularity::PerBlock(4)
        );
        assert_eq!(
            parse_pipeline_eval_granularity(" BLOCK:1 "),
            PipelineEvalGranularity::PerBlock(1)
        );
        assert_eq!(
            parse_pipeline_eval_granularity("yield:16"),
            PipelineEvalGranularity::YieldMs(16)
        );
        assert_eq!(
            parse_pipeline_eval_granularity(" YIELD:8 "),
            PipelineEvalGranularity::YieldMs(8)
        );
        assert_eq!(
            parse_pipeline_eval_granularity("block:0"),
            PipelineEvalGranularity::Off
        );
        assert_eq!(
            parse_pipeline_eval_granularity("yield:0"),
            PipelineEvalGranularity::Off
        );
        assert_eq!(
            parse_pipeline_eval_granularity("block:xyz"),
            PipelineEvalGranularity::Off
        );
        assert_eq!(
            parse_pipeline_eval_granularity("yield:xyz"),
            PipelineEvalGranularity::Off
        );
        assert_eq!(
            parse_pipeline_eval_granularity("garbage"),
            PipelineEvalGranularity::Off
        );
    }

    #[test]
    fn pipeline_eval_granularity_only_blocks_multi_token_non_final_layers() {
        use PipelineEvalGranularity::{Off, PerBlock, PerLayer, Sublayer, YieldMs};

        assert!(!pipeline_eval_should_fire_for(Off, 8, 0, 4));
        assert!(!pipeline_eval_should_fire_for(PerLayer, 1, 0, 4));
        assert!(pipeline_eval_should_fire_for(PerLayer, 8, 0, 4));
        assert!(pipeline_eval_should_fire_for(PerLayer, 8, 2, 4));
        assert!(!pipeline_eval_should_fire_for(PerLayer, 8, 3, 4));
        assert!(!pipeline_eval_should_fire_for(PerLayer, 8, 0, 0));
        assert!(pipeline_eval_should_fire_for(Sublayer, 8, 0, 4));
        assert!(!pipeline_eval_should_fire_for(Sublayer, 1, 0, 4));
        assert!(!pipeline_eval_should_fire_for(Sublayer, 8, 3, 4));

        assert!(!pipeline_eval_should_fire_for(PerBlock(2), 8, 0, 6));
        assert!(pipeline_eval_should_fire_for(PerBlock(2), 8, 1, 6));
        assert!(pipeline_eval_should_fire_for(PerBlock(2), 8, 3, 6));
        assert!(
            !pipeline_eval_should_fire_for(PerBlock(2), 8, 5, 6),
            "final layer remains exempt even when it closes a block"
        );
        // YieldMs pure path is wall-clock only; layer filters still apply via
        // pipeline_eval_yield_should_fire, not the layer-index matcher.
        assert!(!pipeline_eval_should_fire_for(YieldMs(16), 8, 0, 4));
    }

    #[test]
    fn pipeline_eval_yield_predicate_is_wall_clock_and_fail_closed() {
        // First eligible boundary always fires.
        assert!(pipeline_eval_yield_should_fire(
            None,
            1_000_000_000,
            16,
            8,
            0,
            4
        ));
        // Within window: no fire.
        assert!(!pipeline_eval_yield_should_fire(
            Some(1_000_000_000),
            1_000_000_000 + 15_000_000,
            16,
            8,
            1,
            4
        ));
        // At/after window: fire.
        assert!(pipeline_eval_yield_should_fire(
            Some(1_000_000_000),
            1_000_000_000 + 16_000_000,
            16,
            8,
            1,
            4
        ));
        // Decode / final layer / zero ms never fire.
        assert!(!pipeline_eval_yield_should_fire(None, 1, 16, 1, 0, 4));
        assert!(!pipeline_eval_yield_should_fire(None, 1, 16, 8, 3, 4));
        assert!(!pipeline_eval_yield_should_fire(None, 1, 0, 8, 0, 4));
    }

    #[test]
    fn parse_pipeline_eval_tail_layers_is_fail_closed() {
        assert_eq!(parse_pipeline_eval_tail_layers(""), 0);
        assert_eq!(parse_pipeline_eval_tail_layers("off"), 0);
        assert_eq!(parse_pipeline_eval_tail_layers("OFF"), 0);
        assert_eq!(parse_pipeline_eval_tail_layers("12"), 12);
        assert_eq!(parse_pipeline_eval_tail_layers(" 8 "), 8);
        assert_eq!(parse_pipeline_eval_tail_layers("0"), 0);
        assert_eq!(parse_pipeline_eval_tail_layers("xyz"), 0);
        assert_eq!(parse_pipeline_eval_tail_layers("-1"), 0);
    }

    #[test]
    fn pipeline_eval_layer_in_tail_covers_last_n_before_final() {
        // total=40 layers (0..39); final=39 exempt; tail=8 → layers 31..38.
        assert!(!pipeline_eval_layer_in_tail(30, 40, 8));
        assert!(pipeline_eval_layer_in_tail(31, 40, 8));
        assert!(pipeline_eval_layer_in_tail(38, 40, 8));
        assert!(!pipeline_eval_layer_in_tail(39, 40, 8));
        // Off / tiny models.
        assert!(!pipeline_eval_layer_in_tail(0, 40, 0));
        assert!(!pipeline_eval_layer_in_tail(0, 1, 8));
        // Tail larger than eligible set: all non-final layers.
        assert!(pipeline_eval_layer_in_tail(0, 4, 100));
        assert!(pipeline_eval_layer_in_tail(2, 4, 100));
        assert!(!pipeline_eval_layer_in_tail(3, 4, 100));
    }

    #[test]
    fn pipeline_sublayer_eval_is_limited_to_gemma4_multi_token_prefill() {
        use PipelineEvalGranularity::{Off, PerLayer, Sublayer};

        assert!(!pipeline_sublayer_eval_should_fire_for(Off, 8, "gemma4"));
        assert!(!pipeline_sublayer_eval_should_fire_for(
            PerLayer, 8, "gemma4"
        ));
        assert!(!pipeline_sublayer_eval_should_fire_for(
            Sublayer, 1, "gemma4"
        ));
        assert!(pipeline_sublayer_eval_should_fire_for(
            Sublayer, 8, "gemma4"
        ));
        assert!(!pipeline_sublayer_eval_should_fire_for(
            Sublayer, 8, "qwen3_5"
        ));
        assert!(!pipeline_sublayer_eval_should_fire_for(
            Sublayer,
            8,
            "gemma4_vl"
        ));
        assert!(!pipeline_sublayer_eval_should_fire_for(
            Sublayer,
            8,
            "gemma4_unified"
        ));
    }

    #[test]
    fn direct_cpp_gemma4_post_attn_ffn_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_DIRECT_GEMMA4_POST_ATTN_FFN_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_DIRECT_GEMMA4_POST_ATTN_FFN_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_DIRECT_GEMMA4_POST_ATTN_FFN_ENABLED",
            "1"
        ));
    }

    #[test]
    fn dense_swiglu_packed_metal_uses_default_on_kill_switch_contract() {
        assert!(parse_bool_env_default_on(
            "AX_FASTPATH_TEST_DENSE_SWIGLU_PACKED_METAL_UNSET"
        ));
        assert!(!probe_default_on(
            "AX_FASTPATH_TEST_DENSE_SWIGLU_PACKED_METAL_DISABLED",
            "0"
        ));
        assert!(probe_default_on(
            "AX_FASTPATH_TEST_DENSE_SWIGLU_PACKED_METAL_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_gated_delta_prefill_streaming_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_GATED_DELTA_PREFILL_STREAMING_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_GATED_DELTA_PREFILL_STREAMING_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_GATED_DELTA_PREFILL_STREAMING_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_gated_delta_prefill_tile_512_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_GATED_DELTA_PREFILL_TILE_512_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_GATED_DELTA_PREFILL_TILE_512_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_GATED_DELTA_PREFILL_TILE_512_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_prefill_single_2048_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_PREFILL_SINGLE_2048_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_PREFILL_SINGLE_2048_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_PREFILL_SINGLE_2048_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_prefill_flat_ffn_is_family_seq_and_rank_gated() {
        assert!(should_qwen_prefill_flat_ffn_for(true, "qwen3_5", 1024, 3));
        assert!(should_qwen_prefill_flat_ffn_for(true, "QWEN3_5", 128, 3));
        assert!(
            !should_qwen_prefill_flat_ffn_for(true, "qwen3_5", 1, 3),
            "decode must stay on the 3-D qw path"
        );
        assert!(!should_qwen_prefill_flat_ffn_for(true, "qwen3_5", 1024, 2));
        assert!(!should_qwen_prefill_flat_ffn_for(false, "qwen3_5", 1024, 3));
        assert!(!should_qwen_prefill_flat_ffn_for(true, "gemma4", 1024, 3));
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_PREFILL_FLAT_FFN_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_PREFILL_FLAT_FFN_DISABLED",
            "0"
        ));
        assert!(probe("AX_FASTPATH_TEST_QWEN_PREFILL_FLAT_FFN_ENABLED", "1"));
    }

    #[test]
    fn qwen_prefill_contiguous_ffn_is_family_seq_and_rank_gated() {
        assert!(should_qwen_prefill_contiguous_ffn_for(
            true, "qwen3_5", 1024, 3
        ));
        assert!(should_qwen_prefill_contiguous_ffn_for(
            true, "QWEN3_5", 128, 3
        ));
        assert!(
            !should_qwen_prefill_contiguous_ffn_for(true, "qwen3_5", 1, 3),
            "decode must not pay a contiguous copy"
        );
        assert!(!should_qwen_prefill_contiguous_ffn_for(
            true, "qwen3_5", 1024, 2
        ));
        assert!(!should_qwen_prefill_contiguous_ffn_for(
            false, "qwen3_5", 1024, 3
        ));
        assert!(!should_qwen_prefill_contiguous_ffn_for(
            true, "gemma4", 1024, 3
        ));
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_PREFILL_CONTIGUOUS_FFN_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_PREFILL_CONTIGUOUS_FFN_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_PREFILL_CONTIGUOUS_FFN_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_la_out_proj_silu_mul_qmm_is_family_and_seq_gated() {
        assert!(should_qwen_la_out_proj_silu_mul_qmm_for(
            true, "qwen3_5", 1024
        ));
        assert!(should_qwen_la_out_proj_silu_mul_qmm_for(
            true, "QWEN3_5", 128
        ));
        assert!(
            !should_qwen_la_out_proj_silu_mul_qmm_for(true, "qwen3_5", 1),
            "decode keeps rms_norm_gated + qw"
        );
        assert!(!should_qwen_la_out_proj_silu_mul_qmm_for(
            false, "qwen3_5", 1024
        ));
        assert!(!should_qwen_la_out_proj_silu_mul_qmm_for(
            true, "gemma4", 1024
        ));
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_LA_OUT_PROJ_SILU_MUL_QMM_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_LA_OUT_PROJ_SILU_MUL_QMM_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_LA_OUT_PROJ_SILU_MUL_QMM_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_dense_ffn_gate_up_matvec_metal_uses_default_on_kill_switch_contract() {
        assert!(parse_bool_env_default_on(
            "AX_FASTPATH_TEST_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL_UNSET"
        ));
        assert!(!probe_default_on(
            "AX_FASTPATH_TEST_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL_DISABLED",
            "0"
        ));
        assert!(probe_default_on(
            "AX_FASTPATH_TEST_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_linear_mtp_exact_env_override_uses_truthy_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_LINEAR_MTP_EXACT_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_LINEAR_MTP_EXACT_DISABLED",
            "0"
        ));
        assert!(probe("AX_FASTPATH_TEST_QWEN_LINEAR_MTP_EXACT_ENABLED", "1"));
    }

    #[test]
    fn invariant_mxfp4_qmv_fast_is_opt_in() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_INVARIANT_MXFP4_QMV_FAST_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_INVARIANT_MXFP4_QMV_FAST_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_INVARIANT_MXFP4_QMV_FAST_ENABLED",
            "1"
        ));
    }

    #[test]
    fn dense_ffn_compile_uses_default_on_kill_switch_contract() {
        assert!(parse_bool_env_default_on(
            "AX_FASTPATH_TEST_DENSE_FFN_COMPILE_UNSET"
        ));
        assert!(!probe_default_on(
            "AX_FASTPATH_TEST_DENSE_FFN_COMPILE_DISABLED",
            "0"
        ));
        assert!(probe_default_on(
            "AX_FASTPATH_TEST_DENSE_FFN_COMPILE_ENABLED",
            "1"
        ));
    }

    #[test]
    fn dense_ffn_compile_prefill_uses_default_on_with_min_leading() {
        assert!(parse_bool_env_default_on(
            "AX_FASTPATH_TEST_DENSE_FFN_COMPILE_PREFILL_UNSET"
        ));
        assert!(!probe_default_on(
            "AX_FASTPATH_TEST_DENSE_FFN_COMPILE_PREFILL_DISABLED",
            "0"
        ));
        assert!(probe_default_on(
            "AX_FASTPATH_TEST_DENSE_FFN_COMPILE_PREFILL_ENABLED",
            "1"
        ));
        assert_eq!(super::DENSE_FFN_PREFILL_COMPILE_MIN_LEADING, 256);
        assert_eq!(super::MOE_PACKED_GEGLU_PREFILL_MAX_SEQ, 512);
    }

    #[test]
    fn qwen_compiled_dual_gate_up_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_COMPILED_DUAL_GATE_UP_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_COMPILED_DUAL_GATE_UP_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_COMPILED_DUAL_GATE_UP_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_split_ffn_prefill_compile_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_SPLIT_FFN_PREFILL_COMPILE_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_SPLIT_FFN_PREFILL_COMPILE_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_SPLIT_FFN_PREFILL_COMPILE_ENABLED",
            "1"
        ));
        assert_eq!(super::QWEN_SPLIT_FFN_PREFILL_COMPILE_MIN_LEADING, 128);
    }

    #[test]
    fn qwen_linear_add_rms_norm_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_LINEAR_ADD_RMS_NORM_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_LINEAR_ADD_RMS_NORM_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_LINEAR_ADD_RMS_NORM_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_prefill_pipeline_block_is_family_seq_and_stride_gated() {
        assert!(should_qwen_prefill_pipeline_block_for(
            true, "qwen3_5", 1024, 7, 64, 8
        ));
        assert!(should_qwen_prefill_pipeline_block_for(
            true,
            "QWEN3_NEXT",
            2048,
            15,
            64,
            8
        ));
        assert!(
            !should_qwen_prefill_pipeline_block_for(true, "qwen3_5", 512, 7, 64, 8),
            "short prefill stays one lazy graph"
        );
        assert!(!should_qwen_prefill_pipeline_block_for(
            true, "qwen3_5", 1024, 6, 64, 8
        ));
        assert!(
            !should_qwen_prefill_pipeline_block_for(true, "qwen3_5", 1024, 63, 64, 8),
            "never fire after the final layer"
        );
        assert!(!should_qwen_prefill_pipeline_block_for(
            true, "gemma4", 1024, 7, 64, 8
        ));
        assert!(!should_qwen_prefill_pipeline_block_for(
            false, "qwen3_5", 1024, 7, 64, 8
        ));
        assert_eq!(super::QWEN_PREFILL_PIPELINE_BLOCK, 8);
    }

    #[test]
    fn qwen_prefill_interlayer_add_rms_is_family_and_seq_gated() {
        assert!(should_qwen_prefill_interlayer_add_rms_for(
            true, "qwen3_5", 1024
        ));
        assert!(should_qwen_prefill_interlayer_add_rms_for(
            true,
            "QWEN3_NEXT",
            128
        ));
        assert!(!should_qwen_prefill_interlayer_add_rms_for(
            true, "qwen3_5", 1
        ));
        assert!(!should_qwen_prefill_interlayer_add_rms_for(
            true, "gemma4", 1024
        ));
        assert!(!should_qwen_prefill_interlayer_add_rms_for(
            false, "qwen3_5", 1024
        ));
        assert!(should_defer_qwen_prefill_ffn_residual_for(
            true, true, false, 3
        ));
        assert!(
            !should_defer_qwen_prefill_ffn_residual_for(true, false, false, 3),
            "do not defer into a full-attn layer"
        );
        assert!(!should_defer_qwen_prefill_ffn_residual_for(
            true, true, true, 3
        ));
    }

    #[test]
    fn qwen_swiglu_down_fuse_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_SWIGLU_DOWN_FUSE_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_SWIGLU_DOWN_FUSE_DISABLED",
            "0"
        ));
        assert!(probe("AX_FASTPATH_TEST_QWEN_SWIGLU_DOWN_FUSE_ENABLED", "1"));
    }

    #[test]
    fn qwen_prefill_dual_qmm_swiglu_metal_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_PREFILL_DUAL_QMM_SWIGLU_METAL_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_PREFILL_DUAL_QMM_SWIGLU_METAL_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_PREFILL_DUAL_QMM_SWIGLU_METAL_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_prefill_flat_down_qmm_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_PREFILL_FLAT_DOWN_QMM_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_PREFILL_FLAT_DOWN_QMM_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_PREFILL_FLAT_DOWN_QMM_ENABLED",
            "1"
        ));
    }

    #[test]
    fn qwen_dual_qmm_swiglu_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_DUAL_QMM_SWIGLU_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_DUAL_QMM_SWIGLU_DISABLED",
            "0"
        ));
        assert!(probe("AX_FASTPATH_TEST_QWEN_DUAL_QMM_SWIGLU_ENABLED", "1"));
    }

    #[test]
    fn gemma4_assistant_compile_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_COMPILE_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_COMPILE_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_COMPILE_ENABLED",
            "1"
        ));
    }

    #[test]
    fn gemma4_assistant_mtp_cycle_guard_uses_default_on_kill_switch_contract() {
        assert!(parse_bool_env_default_on(
            "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_MTP_CYCLE_GUARD_UNSET"
        ));
        assert!(!probe_default_on(
            "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_MTP_CYCLE_GUARD_DISABLED",
            "0"
        ));
        assert!(probe_default_on(
            "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_MTP_CYCLE_GUARD_ENABLED",
            "1"
        ));
    }

    #[test]
    fn gemma4_assistant_lazy_multi_depth_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_LAZY_MULTI_DEPTH_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_LAZY_MULTI_DEPTH_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_LAZY_MULTI_DEPTH_ENABLED",
            "1"
        ));
    }

    #[test]
    fn gemma4_assistant_deep_needs_first_conf_uses_default_on_kill_switch_contract() {
        assert!(parse_bool_env_default_on(
            "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_DEEP_NEEDS_FIRST_CONF_UNSET"
        ));
        assert!(!probe_default_on(
            "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_DEEP_NEEDS_FIRST_CONF_DISABLED",
            "0"
        ));
        assert!(probe_default_on(
            "AX_FASTPATH_TEST_GEMMA4_ASSISTANT_DEEP_NEEDS_FIRST_CONF_ENABLED",
            "1"
        ));
    }

    #[test]
    fn verify_chunked_submit_is_opt_in_and_multi_position_only() {
        // Default (unset env resolves to 0) never splits a build.
        assert_eq!(verify_submit_interval_for_build(2, 40, 0), 0);
        // A single-position build belongs to the direct pipeline, which
        // already double-buffers; splitting it would only add submits.
        assert_eq!(verify_submit_interval_for_build(1, 40, 8), 0);
        // A speculative verify build splits at the configured interval.
        assert_eq!(verify_submit_interval_for_build(2, 40, 8), 0);
        assert_eq!(verify_submit_interval_for_build(4, 40, 8), 0);
        assert_eq!(verify_submit_interval_for_build(5, 40, 4), 4);
        // An interval that cannot produce a submit before the caller's own
        // terminating eval is pure overhead, so it is refused.
        assert_eq!(verify_submit_interval_for_build(2, 40, 40), 0);
        assert_eq!(verify_submit_interval_for_build(2, 40, 64), 0);
    }

    #[test]
    fn exact_short_verify_uses_configured_interval_instead_of_zero() {
        // Official harness sets VERIFY_SUBMIT_LAYERS=8; honor it as the
        // sole mid-loop stride (not stacked on PIPELINE=layer).
        assert_eq!(exact_short_verify_submit_interval(2, 64, 8), 8);
        assert_eq!(exact_short_verify_submit_interval(4, 64, 8), 8);
        assert_eq!(
            exact_short_verify_submit_interval(2, 64, 0),
            EXACT_SHORT_VERIFY_SUBMIT_DEFAULT
        );
        assert_eq!(exact_short_verify_submit_interval(1, 64, 8), 0);
        assert_eq!(exact_short_verify_submit_interval(5, 64, 8), 0);
        assert_eq!(exact_short_verify_submit_interval(2, 8, 8), 0);
        assert_eq!(exact_short_verify_submit_interval(2, 0, 8), 0);
    }

    #[test]
    fn moe_router_fused_metal_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_MOE_ROUTER_FUSED_METAL_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_MOE_ROUTER_FUSED_METAL_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_MOE_ROUTER_FUSED_METAL_ENABLED",
            "1"
        ));
    }

    #[test]
    fn linear_attention_whole_layer_metal_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_LINEAR_ATTENTION_WHOLE_LAYER_METAL_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_LINEAR_ATTENTION_WHOLE_LAYER_METAL_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_LINEAR_ATTENTION_WHOLE_LAYER_METAL_ENABLED",
            "1"
        ));
    }

    #[test]
    fn moe_deep_expert_block_metal_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_MOE_DEEP_EXPERT_BLOCK_METAL_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_MOE_DEEP_EXPERT_BLOCK_METAL_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_MOE_DEEP_EXPERT_BLOCK_METAL_ENABLED",
            "1"
        ));
    }

    #[test]
    fn geglu_mul_metal_uses_default_on_kill_switch_contract() {
        assert!(parse_bool_env_default_on(
            "AX_FASTPATH_TEST_GEGLU_MUL_METAL_UNSET"
        ));
        assert!(!probe_default_on(
            "AX_FASTPATH_TEST_GEGLU_MUL_METAL_DISABLED",
            "0"
        ));
        assert!(probe_default_on(
            "AX_FASTPATH_TEST_GEGLU_MUL_METAL_ENABLED",
            "1"
        ));
    }

    #[test]
    fn gemma4_per_layer_input_gate_compile_uses_default_on_kill_switch_contract() {
        assert!(parse_bool_env_default_on(
            "AX_FASTPATH_TEST_GEMMA4_PER_LAYER_INPUT_GATE_COMPILE_UNSET"
        ));
        assert!(!probe_default_on(
            "AX_FASTPATH_TEST_GEMMA4_PER_LAYER_INPUT_GATE_COMPILE_DISABLED",
            "0"
        ));
        assert!(probe_default_on(
            "AX_FASTPATH_TEST_GEMMA4_PER_LAYER_INPUT_GATE_COMPILE_ENABLED",
            "1"
        ));
    }

    #[test]
    fn linear_attention_rms_norm_gate_metal_uses_default_on_kill_switch_contract() {
        assert!(parse_bool_env_default_on(
            "AX_FASTPATH_TEST_LINEAR_ATTENTION_RMS_NORM_GATE_METAL_UNSET"
        ));
        assert!(!probe_default_on(
            "AX_FASTPATH_TEST_LINEAR_ATTENTION_RMS_NORM_GATE_METAL_DISABLED",
            "0"
        ));
        assert!(probe_default_on(
            "AX_FASTPATH_TEST_LINEAR_ATTENTION_RMS_NORM_GATE_METAL_ENABLED",
            "1"
        ));
    }

    fn probe_usize(name: &str, value: &str) -> Option<usize> {
        // SAFETY: each test owns a disjoint set of env-var names. Remove
        // before asserting so a failing assert does not leak the var.
        unsafe {
            std::env::set_var(name, value);
        }
        let observed = parse_positive_usize_env(name);
        unsafe {
            std::env::remove_var(name);
        }
        observed
    }

    #[test]
    fn parse_positive_usize_env_accepts_positive_values() {
        assert_eq!(probe_usize("AX_FASTPATH_TEST_USIZE_16", "16"), Some(16));
        assert_eq!(
            probe_usize("AX_FASTPATH_TEST_USIZE_TRIMMED", " 32 "),
            Some(32)
        );
    }

    #[test]
    fn parse_positive_usize_env_rejects_unset_zero_and_invalid_values() {
        assert_eq!(
            parse_positive_usize_env("AX_FASTPATH_TEST_USIZE_UNSET"),
            None
        );
        for value in ["0", "", "no", "-1", "1.5"] {
            let name = format!("AX_FASTPATH_TEST_BAD_USIZE_{}", value.replace('-', "neg"));
            assert_eq!(
                probe_usize(&name, value),
                None,
                "expected None for {value:?}"
            );
        }
    }

    #[test]
    fn shared_fusion_threshold_preserves_each_callers_default_when_unset() {
        if std::env::var_os("AX_MLX_MOE_SHARED_FUSION_SEQ_THRESHOLD").is_some() {
            return;
        }

        assert_eq!(moe_shared_fusion_seq_threshold(64), 64);
        assert_eq!(moe_shared_fusion_seq_threshold(128), 128);
    }

    #[test]
    fn parse_nonnegative_f32_accepts_finite_zero_and_positive_values() {
        assert_eq!(parse_nonnegative_f32("0"), Some(0.0));
        assert_eq!(parse_nonnegative_f32("1e-5"), Some(1.0e-5));
        assert_eq!(parse_nonnegative_f32(" 0.25 "), Some(0.25));
    }

    #[test]
    fn parse_nonnegative_f32_rejects_negative_invalid_and_nonfinite_values() {
        for value in ["-0.1", "NaN", "inf", "-inf", "", "no"] {
            assert_eq!(
                parse_nonnegative_f32(value),
                None,
                "expected invalid sparse threshold for {value:?}"
            );
        }
    }

    #[test]
    fn scale_prefill_chunk_for_remaining_clamps_long_prompts_only() {
        assert_eq!(scale_prefill_chunk_for_remaining(1536, 34), 1536);
        assert_eq!(scale_prefill_chunk_for_remaining(512, 34), 512);
        assert_eq!(scale_prefill_chunk_for_remaining(1024, 512), 1024);
        assert_eq!(
            scale_prefill_chunk_for_remaining(1536, LONG_PROMPT_PREFILL_THRESHOLD),
            long_prompt_prefill_chunk()
        );
        assert_eq!(
            scale_prefill_chunk_for_remaining(1536, 13_826),
            long_prompt_prefill_chunk().min(1536)
        );
        assert_eq!(scale_prefill_chunk_for_remaining(256, 13_826), 256);
        assert_eq!(scale_prefill_chunk_for_remaining(0, 100), 1);
    }

    #[test]
    fn long_prompt_prefill_clamp_skips_qwen3_5() {
        assert!(!long_prompt_prefill_clamp_applies("qwen3_5"));
        assert!(!long_prompt_prefill_clamp_applies("QWEN3_5"));
        assert!(!long_prompt_prefill_clamp_applies("muse_glimmer"));
        assert!(!long_prompt_prefill_clamp_applies("qwen3_vl_moe"));
        assert!(!long_prompt_prefill_clamp_applies("qwen3_vl"));
        assert!(long_prompt_prefill_clamp_applies("gemma4"));
        assert!(long_prompt_prefill_clamp_applies("qwen3_next"));
        assert_eq!(
            scale_prefill_chunk_for_remaining_in_family(2048, 2048, "qwen3_5"),
            2048
        );
        assert_eq!(
            scale_prefill_chunk_for_remaining_in_family(2048, 2048, "muse_glimmer"),
            2048
        );
        assert_eq!(
            scale_prefill_chunk_for_remaining_in_family(2048, 2048, "gemma4"),
            long_prompt_prefill_chunk()
        );
    }

    #[test]
    fn cold_prefill_clears_mlx_cache_only_on_empty_kv() {
        assert!(should_clear_mlx_cache_before_cold_prefill(0));
        assert!(!should_clear_mlx_cache_before_cold_prefill(1));
        assert!(!should_clear_mlx_cache_before_cold_prefill(2048));
        assert!(should_clear_mlx_cache_before_cold_prefill_for(false, 0));
        assert!(
            !should_clear_mlx_cache_before_cold_prefill_for(true, 0),
            "skip flag must keep the buffer pool warm on cold prefill"
        );
        assert!(!should_clear_mlx_cache_before_cold_prefill_for(true, 1));
    }

    #[test]
    fn qwen_prefill_intermediate_async_eval_is_family_and_chunk_gated() {
        assert!(should_async_eval_intermediate_qwen_prefill_for(
            true, "qwen3_5", false
        ));
        assert!(should_async_eval_intermediate_qwen_prefill_for(
            true, "QWEN3_5", false
        ));
        assert!(
            !should_async_eval_intermediate_qwen_prefill_for(true, "qwen3_5", true),
            "final chunk must still block so decode sees settled KV"
        );
        assert!(!should_async_eval_intermediate_qwen_prefill_for(
            false, "qwen3_5", false
        ));
        assert!(!should_async_eval_intermediate_qwen_prefill_for(
            true, "gemma4", false
        ));
        assert!(!should_async_eval_intermediate_qwen_prefill_for(
            true,
            "qwen3_next",
            false
        ));
    }

    #[test]
    fn qwen_prefill_lazy_intermediate_is_family_total_and_chunk_gated() {
        assert!(should_keep_lazy_intermediate_qwen_prefill_for(
            true, "qwen3_5", false, 2048
        ));
        assert!(should_keep_lazy_intermediate_qwen_prefill_for(
            true, "QWEN3_5", false, 2048
        ));
        assert!(
            should_keep_lazy_intermediate_qwen_prefill_for(true, "qwen3_5", false, 128),
            "single-chunk contract totals still match the skip_cache_only gate"
        );
        assert!(
            !should_keep_lazy_intermediate_qwen_prefill_for(true, "qwen3_5", true, 2048),
            "final chunk must still eval so decode sees settled KV"
        );
        assert!(!should_keep_lazy_intermediate_qwen_prefill_for(
            false, "qwen3_5", false, 2048
        ));
        assert!(!should_keep_lazy_intermediate_qwen_prefill_for(
            true, "qwen3_5", false, 2049
        ));
        assert!(!should_keep_lazy_intermediate_qwen_prefill_for(
            true, "gemma4", false, 2048
        ));
        assert!(!should_keep_lazy_intermediate_qwen_prefill_for(
            true,
            "qwen3_next",
            false,
            2048
        ));
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_PREFILL_LAZY_INTERMEDIATE_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_PREFILL_LAZY_INTERMEDIATE_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_PREFILL_LAZY_INTERMEDIATE_ENABLED",
            "1"
        ));
    }

    #[test]
    fn certified_non_deepseek_skips_cache_only_split_on_contract_shapes() {
        for family in [
            "qwen3_5",
            "qwen3_next",
            "qwen3",
            "gemma4",
            "glm4_moe_lite",
            "gpt_oss",
            "muse_glimmer",
            "qwen3_vl",
            "qwen3_vl_moe",
        ] {
            assert!(
                skip_cache_only_split_for_family(family, 128),
                "{family} p128"
            );
            assert!(skip_cache_only_split_for_family(family, 2048));
            assert!(!skip_cache_only_split_for_family(family, 2049));
        }
        assert!(!skip_cache_only_split_for_family("qwen3_5", 0));
        assert!(!skip_cache_only_split_for_family("deepseek_v32", 128));
        assert!(!skip_cache_only_split_for_family("gemma4_vl", 128));
    }

    #[test]
    fn qwen_skip_linear_prefill_mask_is_family_and_layer_gated() {
        assert!(should_skip_linear_prefill_mask_for(true, "qwen3_5", true));
        assert!(should_skip_linear_prefill_mask_for(
            true,
            "QWEN3_NEXT",
            true
        ));
        assert!(
            !should_skip_linear_prefill_mask_for(true, "qwen3_5", false),
            "full-attn layers still need the offset mask"
        );
        assert!(!should_skip_linear_prefill_mask_for(true, "gemma4", true));
        assert!(!should_skip_linear_prefill_mask_for(false, "qwen3_5", true));
    }

    #[test]
    fn qwen_prefill_eval_kv_only_is_intermediate_and_family_gated() {
        assert!(should_qwen_prefill_eval_kv_only_for(
            true, "qwen3_5", false, 2048
        ));
        assert!(
            !should_qwen_prefill_eval_kv_only_for(true, "qwen3_5", true, 2048),
            "final chunk still evals logits + KV"
        );
        assert!(!should_qwen_prefill_eval_kv_only_for(
            true, "qwen3_5", false, 2049
        ));
        assert!(!should_qwen_prefill_eval_kv_only_for(
            true, "gemma4", false, 2048
        ));
        assert!(!should_qwen_prefill_eval_kv_only_for(
            false, "qwen3_5", false, 2048
        ));
    }

    #[test]
    fn exact_size_first_kv_is_write_start_gated() {
        assert!(should_exact_size_first_kv_for(true, 0));
        assert!(
            !should_exact_size_first_kv_for(true, 128),
            "append after the first write still grows in KV_CHUNK_TOKENS"
        );
        assert!(!should_exact_size_first_kv_for(false, 0));
    }

    #[test]
    fn exact_size_first_kv_targets_unaligned_contract_p128() {
        // The product flag only changes the first write. Of the fleet
        // contract prompts, only 128 is not a KV_CHUNK_TOKENS multiple, so
        // p512/p2048 already skip zeros+slice_update without the flag.
        const CHUNK: usize = crate::kv_cache::KV_CHUNK_TOKENS;
        assert_eq!(CHUNK, 256);
        assert_ne!(128 % CHUNK, 0, "p128 must take the exact-size first write");
        assert_eq!(512 % CHUNK, 0);
        assert_eq!(2048 % CHUNK, 0);
        assert!(
            should_exact_size_first_kv_for(true, 0),
            "fresh-layer first write is the only exact-size site"
        );
        assert!(!should_exact_size_first_kv_for(true, 128));
    }

    #[test]
    fn compiled_qgelu_axq_p128_is_layout_and_seq_gated() {
        assert!(should_compiled_qgelu_axq_p128_for(true, 32, 4, 128));
        assert!(
            !should_compiled_qgelu_axq_p128_for(true, 64, 4, 128),
            "community gs64/bits=4 stays on the shapeless #680 path"
        );
        assert!(
            !should_compiled_qgelu_axq_p128_for(true, 32, 4, 512),
            "p512 must stay portable so the p128 compile cannot regress longer cells"
        );
        assert!(!should_compiled_qgelu_axq_p128_for(true, 32, 4, 2048));
        assert!(!should_compiled_qgelu_axq_p128_for(true, 32, 4, 1));
        assert!(!should_compiled_qgelu_axq_p128_for(true, 32, 8, 128));
        assert!(!should_compiled_qgelu_axq_p128_for(true, 0, 4, 128));
        assert!(!should_compiled_qgelu_axq_p128_for(false, 32, 4, 128));
    }

    #[test]
    fn exact_size_kv_grow_is_aligned_tight_append() {
        assert!(should_exact_size_kv_grow_for(true, 1024, 1024, 2048, 2048));
        assert!(
            !should_exact_size_kv_grow_for(true, 128, 128, 129, 256),
            "decode +1 must keep the padded zeros grow"
        );
        assert!(!should_exact_size_kv_grow_for(true, 512, 1024, 1536, 1536));
        assert!(!should_exact_size_kv_grow_for(
            false, 1024, 1024, 2048, 2048
        ));
    }

    #[test]
    fn skip_unused_full_kv_view_slice_is_full_buffer_gated() {
        assert!(should_skip_unused_full_kv_view_slice_for(
            true, 0, 2048, 2048
        ));
        assert!(
            !should_skip_unused_full_kv_view_slice_for(true, 0, 128, 256),
            "padded first write still needs the live-token slice"
        );
        assert!(!should_skip_unused_full_kv_view_slice_for(
            true, 1024, 2048, 2048
        ));
        assert!(!should_skip_unused_full_kv_view_slice_for(
            false, 0, 2048, 2048
        ));
    }

    #[test]
    fn skip_unused_la_out_reshape_is_shape_gated() {
        assert!(should_skip_unused_la_out_reshape_for(
            true,
            &[1, 1024, 2048],
            1024,
            2048
        ));
        assert!(
            !should_skip_unused_la_out_reshape_for(true, &[1, 1024, 32, 64], 1024, 2048),
            "BHSD still needs the flatten into [1,S,V]"
        );
        assert!(!should_skip_unused_la_out_reshape_for(
            false,
            &[1, 1024, 2048],
            1024,
            2048
        ));
    }

    #[test]
    fn reuse_la_initial_state_zeros_is_flag_gated() {
        assert!(should_reuse_la_initial_state_zeros_for(true));
        assert!(!should_reuse_la_initial_state_zeros_for(false));
    }

    #[test]
    fn long_prompt_prefill_chunk_defaults_to_512() {
        // Process-cached via OnceLock; assert the default constant and that
        // the live helper never returns zero.
        assert_eq!(LONG_PROMPT_PREFILL_CHUNK, 512);
        assert!(long_prompt_prefill_chunk() >= 1);
    }

    #[test]
    fn resolve_prefill_chunk_defaults_mla_to_chunk_aligned_size() {
        assert_eq!(
            resolve_prefill_chunk(true, 256, None),
            MLA_DEFAULT_PREFILL_CHUNK
        );
    }

    #[test]
    fn resolve_prefill_chunk_allows_mla_override() {
        assert_eq!(resolve_prefill_chunk(true, 256, Some(32)), 32);
    }

    #[test]
    fn resolve_prefill_chunk_preserves_non_mla_request() {
        assert_eq!(resolve_prefill_chunk(false, 256, Some(32)), 256);
    }

    #[test]
    fn resolve_prefill_chunk_clamps_zero_for_all_models() {
        assert_eq!(resolve_prefill_chunk(false, 0, None), 1);
        assert_eq!(resolve_prefill_chunk(true, 0, Some(0)), 1);
    }

    #[test]
    fn resolve_mla_cold_prefill_chunk_defaults_to_warm_trail() {
        assert_eq!(
            resolve_mla_cold_prefill_chunk(MLA_DEFAULT_PREFILL_CHUNK, None),
            MLA_DEFAULT_PREFILL_CHUNK
        );
    }

    #[test]
    fn resolve_mla_cold_prefill_chunk_allows_throughput_override() {
        assert_eq!(resolve_mla_cold_prefill_chunk(16, Some(2048)), 2048);
    }

    #[test]
    fn select_prefill_chunk_for_request_matrix_cold_vs_warm() {
        // Empty cache always uses cold field (even if larger than warm).
        assert_eq!(
            select_prefill_chunk_for_request(0, 2048, 16),
            (2048, PrefillChunkMode::Cold)
        );
        // Restored / partial cache always uses warm field.
        assert_eq!(
            select_prefill_chunk_for_request(1, 2048, 16),
            (16, PrefillChunkMode::WarmExtend)
        );
        assert_eq!(
            select_prefill_chunk_for_request(128, 16, 16),
            (16, PrefillChunkMode::WarmExtend)
        );
        // R2 default: both fields equal → mode still distinguishes occupancy.
        assert_eq!(
            select_prefill_chunk_for_request(0, 16, 16),
            (16, PrefillChunkMode::Cold)
        );
        // Zero chunks clamp to 1 so the loop cannot stall.
        assert_eq!(
            select_prefill_chunk_for_request(0, 0, 0),
            (1, PrefillChunkMode::Cold)
        );
    }

    #[test]
    fn select_prefill_chunk_recompute_after_reset_is_cold() {
        // After cache.reset() / failed restore, seq_len is 0 → cold trail.
        let after_reset_seq = 0usize;
        assert_eq!(
            select_prefill_chunk_for_request(after_reset_seq, 16, 16).1,
            PrefillChunkMode::Cold
        );
    }

    #[test]
    fn prefill_warmup_token_count_preserves_non_mla_lightweight_warmup() {
        assert_eq!(prefill_warmup_token_count(false, 256), 8);
    }

    #[test]
    fn prefill_warmup_token_count_uses_effective_mla_chunk() {
        assert_eq!(
            prefill_warmup_token_count(true, MLA_DEFAULT_PREFILL_CHUNK),
            MLA_DEFAULT_PREFILL_CHUNK
        );
        assert_eq!(prefill_warmup_token_count(true, 32), 32);
    }

    #[test]
    fn prefill_warmup_token_count_clamps_mla_zero() {
        assert_eq!(prefill_warmup_token_count(true, 0), 1);
    }

    #[test]
    fn prefill_warmup_token_lengths_cover_short_prompt_serving_shapes() {
        let lengths = prefill_warmup_token_lengths(false, 256);
        assert!(lengths.contains(&8), "historical lightweight warm-up");
        assert!(lengths.contains(&32));
        assert!(lengths.contains(&34), "flip S0 prompt length must be warm");
        assert!(lengths.contains(&64));
        // Sorted unique
        let mut sorted = lengths.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(lengths, sorted);
    }
}
