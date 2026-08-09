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
/// while the host still builds the later ones — the same overlap
/// `async_eval_kv_refs` already gives cache-only prefill chunks.
///
/// Exactness-preserving: `async_eval` schedules an already-built graph and
/// changes no operand, shape, or reduction order. Only the synchronisation
/// point moves.
///
/// Pair it with a raised `MLX_MAX_MB_PER_BUFFER` on MoE checkpoints. MLX
/// charges a `gather_qmm`'s full expert stack against its per-command-buffer
/// byte cap, so at the default cap every MoE layer already forces a
/// command-buffer split and the submit degenerates into a barrier — which
/// would make chunked submission slower, not faster. See
/// `docs/performance/gather-qmm-async-serialization.md`.
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
    if seq <= 1 || configured == 0 || configured >= layer_count {
        return 0;
    }
    configured
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

env_flag!(
    /// `AX_MLX_MTP_ASYNC_DRAFT` — schedule the greedy zero-gate MTP draft
    /// with `async_eval` and defer host token extraction to the start of the
    /// next decode cycle, overlapping the draft head's GPU forward with
    /// per-token host work (detokenization, stream emission).
    ///
    /// **Default: OFF** (opt-in via `AX_MLX_MTP_ASYNC_DRAFT=1`).
    ///
    /// Exactness-preserving: the identical lazy draft graph is evaluated;
    /// only the synchronization point moves. Engages only under the exact
    /// profile with the confidence gate disabled, non-stochastic drafting,
    /// and skip-state off — the regime where the synchronous greedy path
    /// computes no log-probs or distributions.
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
    multi_token_window_views_enabled,
    "AX_MLX_MULTI_TOKEN_WINDOW_VIEWS"
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
    /// Profile residual: pure Gemma 13.8k `sdpa` ~1.22s; 8/48 full-attention
    /// layers grow full-context SDPA and currently pay array masks after the
    /// first prefill chunk.
    ///
    /// **Default: OFF** (opt-in pure A/B). Sliding-window layers still need
    /// explicit masks when the window constraint is active.
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
    /// The legacy tiered TG-cache kernel remains the production default: it
    /// matches the README high-water cells on p=128/512, and the medium 1024
    /// specialization (with the runner's linear-attention chunk clamp) is
    /// still the best measured long-prompt path on Qwen 3.6 27B. Streaming is
    /// retained for A/B on very long prompts where TG occupancy dominates.
    qwen_gated_delta_prefill_streaming_enabled,
    "AX_MLX_QWEN_GATED_DELTA_PREFILL_STREAMING"
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

/// Minimum leading element count (product of non-last dims) before dense FFN
/// prefill compile engages. `batch * seq` for standard `[B,S,H]` layouts;
/// 256 covers mid-length prompts; README 128-token rows stay uncompiled
/// so short-prompt microbenches avoid compile tax.
pub const DENSE_FFN_PREFILL_COMPILE_MIN_LEADING: i64 = 256;

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
    /// outside decode-trace. For eligible families, caps raise
    /// **optimistically on first process decision** (including dense-first
    /// loads) so multi-model servers that load Llama then MoE still get the
    /// MoE win.
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

/// Scale a base prefill chunk for the remaining prompt length.
///
/// Long remaining prompts clamp to [`long_prompt_prefill_chunk`] so formal S1
/// thr keeps the pure envelope when the session base is larger (e.g. 1536).
/// Short prompts keep `base_chunk` (S0 34-token prompts are a single chunk
/// either way, so TTFT is dominated by warmup/host, not chunk size).
pub fn scale_prefill_chunk_for_remaining(base_chunk: usize, remaining_tokens: usize) -> usize {
    let base = base_chunk.max(1);
    if remaining_tokens >= LONG_PROMPT_PREFILL_THRESHOLD {
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
        for value in ["0", "false", "FALSE", "False", "no", "NO", "No"] {
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
        assert_eq!(verify_submit_interval_for_build(2, 40, 8), 8);
        assert_eq!(verify_submit_interval_for_build(5, 40, 4), 4);
        // An interval that cannot produce a submit before the caller's own
        // terminating eval is pure overhead, so it is refused.
        assert_eq!(verify_submit_interval_for_build(2, 40, 40), 0);
        assert_eq!(verify_submit_interval_for_build(2, 40, 64), 0);
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
