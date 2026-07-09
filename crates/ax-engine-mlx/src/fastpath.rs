//! Process-wide environment flags for ax-engine-mlx optimization fast paths.
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

use std::sync::OnceLock;

fn parse_bool_env(var: &str) -> bool {
    let Ok(raw) = std::env::var(var) else {
        return false;
    };
    let trimmed = raw.trim();
    trimmed.eq_ignore_ascii_case("1")
        || trimmed.eq_ignore_ascii_case("true")
        || trimmed.eq_ignore_ascii_case("yes")
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
    /// Engaged by `AX_DISABLE_TURBOQUANT_FUSED_DECODE` (truthy values per
    /// the module-level parser contract). Forces every layer's TurboQuant
    /// fused-decode candidate to `Disabled`, routing decode through the
    /// full-precision SDPA fallback. The `Disabled` status reuses the
    /// existing `record_turboquant_decode_candidate` telemetry bucket, so
    /// the env path is observable without a counter-schema change.
    turboquant_fused_decode_disabled,
    "AX_DISABLE_TURBOQUANT_FUSED_DECODE"
);

env_flag_default_on!(
    /// `AX_TURBOQUANT_INCREMENTAL_DECODE` — maintain the TurboQuant shadow
    /// compressed runtime buffer on short decode advances (1-4 new cold tokens)
    /// instead of waiting for the next 256-token block boundary.
    ///
    /// **Default: ON** (kill-switch via `AX_TURBOQUANT_INCREMENTAL_DECODE=0`).
    turboquant_incremental_decode_enabled,
    "AX_TURBOQUANT_INCREMENTAL_DECODE"
);

env_flag_default_on!(
    /// `AX_TURBOQUANT_INCREMENTAL_UPLOAD` — refresh the TurboQuant shadow
    /// Metal buffer by uploading only the byte range written since the last
    /// sync (`slice_update` into the existing device array) instead of
    /// re-uploading the whole compressed buffer. Falls back to a full upload
    /// automatically when the buffer grows (256-token block boundary).
    ///
    /// **Default: ON** (kill-switch via `AX_TURBOQUANT_INCREMENTAL_UPLOAD=0`).
    turboquant_incremental_upload_enabled,
    "AX_TURBOQUANT_INCREMENTAL_UPLOAD"
);

/// Minimum total context tokens before the TurboQuant fused compressed-decode
/// path engages. Below this threshold every layer routes through full-precision
/// SDPA because the per-step CPU sync overhead (query readback + hot-tail
/// merge) dominates at short contexts. Defaults to 4096; configurable via
/// `AX_TURBOQUANT_FUSED_DECODE_MIN_CONTEXT`.
pub fn turboquant_fused_decode_min_context_tokens() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_positive_usize_env("AX_TURBOQUANT_FUSED_DECODE_MIN_CONTEXT").unwrap_or(4096)
    })
}

/// Sparse-V minimum normalized attention weight for the TurboQuant two-stage
/// value-sum kernel. Defaults to the PRD value (`1e-5`) and treats invalid or
/// negative input as the default.
pub fn turboquant_sparse_v_threshold() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_nonnegative_f32_env("AX_TURBOQUANT_SPARSE_V_THRESHOLD").unwrap_or(1.0e-5)
    })
}

/// Minimum total context length before sparse-V thresholding engages. Shorter
/// contexts use threshold 0 so the two-stage path remains dense.
pub fn turboquant_sparse_v_min_context_tokens() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_positive_usize_env("AX_TURBOQUANT_SPARSE_V_MIN_CONTEXT").unwrap_or(4096)
    })
}

pub fn turboquant_sparse_v_threshold_for_context(total_context_tokens: usize) -> f32 {
    if total_context_tokens >= turboquant_sparse_v_min_context_tokens() {
        turboquant_sparse_v_threshold()
    } else {
        0.0
    }
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
    /// plus a last-dim slice when the artifact ships split projections and the
    /// quantization metadata is compatible.
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

env_flag!(
    /// `AX_MLX_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL` — opt in to AX's
    /// decode-only Qwen dense FFN gate/up affine-quantized SwiGLU kernel.
    ///
    /// **Default: OFF**. This is deliberately separate from loader-time
    /// `AX_MLX_PACK_DENSE_FFN_GATE_UP`: Qwen3.6 dense FFNs stay on split
    /// gate/up weights, while decode may fuse both split projections and the
    /// SwiGLU activation behind a custom Metal kernel. Unsupported shapes fall
    /// back to the existing two MLX quantized matmuls plus activation.
    qwen_dense_ffn_gate_up_matvec_metal_enabled,
    "AX_MLX_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL"
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
    /// `benchmarks/results/mlx-inference/2026-06-11-gemma4-ffn-route-ab/`.
    /// The production route is guarded to dense packed-quantized Gemma4 layers
    /// without per-layer input gating, profiling, last-position slicing, or
    /// active weight rotation.
    direct_cpp_gemma4_post_attn_ffn_enabled,
    "AX_MLX_DIRECT_CPP_GEMMA4_POST_ATTN_FFN"
);

env_flag!(
    /// `AX_MLX_DENSE_QMATMUL_RMS_NORM` — fuse the dense FFN down-projection
    /// and post-FFN RMSNorm into one C++ call.
    ///
    /// **Default: OFF**. A/B on Gemma 4 31B showed ~0.45% regression. The
    /// C++ wrapper overhead (10 parameters, optional biases conversion) exceeds
    /// the savings from one fewer Rust→C FFI crossing. MLX graph node count is
    /// unchanged either way. Left as opt-in for future re-evaluation.
    dense_qmatmul_rms_norm_enabled,
    "AX_MLX_DENSE_QMATMUL_RMS_NORM"
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
    /// compiled via `mlx_compile` with `shapeless=true`. The compiled
    /// closure is cached per `(layer_index, thread_id)` and reused across
    /// decode steps, collapsing ~10 per-layer MLX dispatches into a single
    /// compiled graph. Only engages for `seq == 1` (decode). Falls back to
    /// the uncompiled path on compilation failure.
    ///
    /// Previously default-on; reverted to opt-in because the compiled MoE
    /// closure can crash in long-running processes when MLX's thread-local
    /// stream registry becomes invalid. `catch_unwind` does not protect
    /// against Rust panics under `panic = "abort"` (release profile).
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

env_flag!(
    /// `AX_MLX_MOE_ROUTER_FUSED_METAL` — enable fused MoE router Metal
    /// kernel for decode.
    ///
    /// **Default: OFF** (opt-in). When the model uses the Qwen3 narrow-softmax
    /// router path and the logits are castable to f32, the post-matmul router
    /// chain (argpartition + take_along_axis + softmax + renormalize) is
    /// collapsed into a single Metal kernel dispatch. Decode-only (seq==1).
    /// Falls back to the MLX op path when ineligible.
    moe_router_fused_metal_enabled,
    "AX_MLX_MOE_ROUTER_FUSED_METAL"
);

env_flag!(
    /// `AX_MLX_LINEAR_ATTENTION_WHOLE_LAYER_METAL` — enable whole-layer
    /// Metal kernel for linear-attention decode.
    ///
    /// **Default: OFF** (opt-in). When eligible (seq==1, qwen3_5/qwen3_next
    /// family), the entire linear-attention decode path (RMSNorm + QKVZ/BA
    /// projection + conv1d + SiLU + per-head RMSNorm + gated-delta + output
    /// projection) is fused into a single Metal kernel dispatch. Decode-only.
    /// Falls back to the multi-dispatch path when ineligible.
    ///
    /// Status: scaffold — kernel body requires multi-week Metal engineering.
    linear_attention_whole_layer_metal_enabled,
    "AX_MLX_LINEAR_ATTENTION_WHOLE_LAYER_METAL"
);

env_flag!(
    /// `AX_MLX_MOE_DEEP_EXPERT_BLOCK_METAL` — enable deep expert-block
    /// fusion Metal kernel for MoE decode.
    ///
    /// **Default: OFF** (opt-in). Fuses gather_qmm(gate_up) + SwiGLU +
    /// gather_qmm(down) + weighted-sum into a single Metal kernel dispatch,
    /// achieving dense-class bandwidth utilization for MoE layers. Decode-only.
    /// Falls back to the standard multi-dispatch path when ineligible.
    ///
    /// Status: scaffold — kernel body requires multi-week Metal engineering.
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
    /// with 24 MoE layers). **Decode-only (seq==1):** prefill uses the
    /// split path where separate ops are faster on large tensors.
    moe_geglu_packed_metal_enabled,
    "AX_MLX_MOE_GEGLU_PACKED_METAL"
);

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
    /// `MLA_DEFAULT_PREFILL_CHUNK`) makes the canonical default-path
    /// warm_extend harness pass 5/5 with a real prefix-cache hit. This flag
    /// exists as a fail-closed escape hatch if a future workload exposes a
    /// drift vector the chunk-alignment fix does not cover.
    mla_prefix_restore_disabled,
    "AX_DISABLE_MLA_PREFIX_RESTORE"
);

// ── DiffusionGemma denoise-loop overrides ──────────────────────────────────
//
// These accessors let benchmark campaigns sweep convergence thresholds and
// toggle the compiled denoise closure without recompiling. All are read once
// per process and cached via OnceLock.

env_flag!(
    /// `AX_MLX_GEMMA4_ASSISTANT_COMPILE` — wrap the Gemma4 assistant MTP
    /// forward in an `MlxClosure` compiled via `mlx_compile`.
    ///
    /// **Default: OFF** (opt-in via `AX_MLX_GEMMA4_ASSISTANT_COMPILE=1`).
    /// The assistant attends the target's frozen KV cache for each draft
    /// depth; the compiled graph fuses the ~100+ per-step MLX dispatches
    /// (embed, pre-projection, N assistant layers with target SDPA, final
    /// norm, lm_head, post-projection) into a single compiled graph.
    /// Falls back to the imperative `gemma4_assistant_forward_one` path
    /// when disabled or compilation fails.
    gemma4_assistant_compile_enabled,
    "AX_MLX_GEMMA4_ASSISTANT_COMPILE"
);

env_flag!(
    /// `AX_DIFFUSION_EMBEDDING_CACHE` — cache per-layer embedding inputs
    /// across denoise steps when token IDs are unchanged. Uses a
    /// GPU-side sum fingerprint to detect changes (2 dispatches + 1
    /// eval) and skips 46 embedding dispatches on cache hit. Default
    /// OFF; opt-in for benchmarking.
    diffusion_embedding_cache_enabled,
    "AX_DIFFUSION_EMBEDDING_CACHE"
);

env_flag!(
    /// `AX_DIFFUSION_KV_CONCAT_BUFFER` — pre-allocate KV concatenation
    /// buffers on the first denoise step and update the canvas slice via
    /// `slice_update` on subsequent steps, avoiding re-`concatenate`-ing the
    /// prompt prefix. **Known issue:** this `slice_update` path is not
    /// bit-equivalent to the canonical `concatenate` path (≈237/256 token
    /// divergence on a 512-token block, perturbing convergence) and shows no
    /// throughput gain in a bit-exact configuration. Default OFF; opt-in only
    /// for benchmarking a future bit-exact reimplementation.
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
    fn qwen_dense_ffn_gate_up_matvec_metal_uses_opt_in_contract() {
        assert!(!parse_bool_env(
            "AX_FASTPATH_TEST_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL_UNSET"
        ));
        assert!(!probe(
            "AX_FASTPATH_TEST_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL_DISABLED",
            "0"
        ));
        assert!(probe(
            "AX_FASTPATH_TEST_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL_ENABLED",
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
    fn gemma4_per_layer_gelu_mul_metal_uses_default_on_kill_switch_contract() {
        assert!(parse_bool_env_default_on(
            "AX_FASTPATH_TEST_GEMMA4_PER_LAYER_GELU_MUL_METAL_UNSET"
        ));
        assert!(!probe_default_on(
            "AX_FASTPATH_TEST_GEMMA4_PER_LAYER_GELU_MUL_METAL_DISABLED",
            "0"
        ));
        assert!(probe_default_on(
            "AX_FASTPATH_TEST_GEMMA4_PER_LAYER_GELU_MUL_METAL_ENABLED",
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
    fn turboquant_incremental_decode_uses_default_on_kill_switch_contract() {
        assert!(parse_bool_env_default_on(
            "AX_FASTPATH_TEST_TURBOQUANT_INCREMENTAL_UNSET"
        ));
        assert!(!probe_default_on(
            "AX_FASTPATH_TEST_TURBOQUANT_INCREMENTAL_DISABLED",
            "0"
        ));
        assert!(probe_default_on(
            "AX_FASTPATH_TEST_TURBOQUANT_INCREMENTAL_ENABLED",
            "1"
        ));
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
}
