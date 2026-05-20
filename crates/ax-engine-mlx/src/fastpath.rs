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
//!
//! See `.internal/planning/MLX-FASTPATH-AUDIT-2026-05-14.md` for the full
//! audit and gap analysis.

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

env_flag!(
    /// `AX_MLX_DIRECT_CPP_GEMMA4_POST_ATTN_FFN` — opt-in direct C++ route for
    /// Gemma4 dense post-attention residual + FFN + layer-scalar orchestration.
    ///
    /// **Default: OFF**. The P0 clean microbench artifact proves this large-block
    /// boundary can beat the portable Rust/`mlx-c` composition on the candidate
    /// shape, but promotion still requires real-model A/B artifacts. The
    /// production route is guarded to dense packed-quantized Gemma4 layers without
    /// per-layer input gating, profiling, last-position slicing, or active weight
    /// rotation.
    direct_cpp_gemma4_post_attn_ffn_enabled,
    "AX_MLX_DIRECT_CPP_GEMMA4_POST_ATTN_FFN"
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

env_flag!(
    /// `AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_INPUTS` — opt-in direct C++ route
    /// for Qwen linear-attention packed QKVZ/BA projection staging.
    ///
    /// **Default: OFF**. The shim skips per-op `mlx-c` dispatches for the
    /// packed projection, reshape, slice, and concat boundary. Decode A/B
    /// on Qwen 3.6 27B 4-bit (M5 Max, 3 reps with 20s inter-row cooldown
    /// and a 30s pre-run thermal settle, 100% fastpath hit rate over 144
    /// invocations) measured both prefill and decode neutral within the
    /// run-to-run noise floor (±0.3% prefill, ±0.1% decode) across
    /// 128/512/2048 prompts. The fused projection saves per-op FFI
    /// dispatch but does not move the wall-clock needle in production
    /// decode shape — the per-step gap to `mlx_lm.benchmark` lives in a
    /// different stage. The flag stays opt-in because neutral evidence is
    /// not enough to justify flipping the default surface area. See
    /// `benchmarks/results/mlx-inference/ab-direct-cpp-linear-inputs/`
    /// (raw artifacts).
    direct_cpp_linear_attention_inputs_enabled,
    "AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_INPUTS"
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
