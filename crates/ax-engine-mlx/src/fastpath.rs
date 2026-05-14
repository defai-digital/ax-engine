//! Process-wide environment flags for ax-engine-mlx optimization fast paths.
//!
//! Each accessor reads its environment variable once per process and caches
//! the result in a `OnceLock`. The value is parsed case-insensitively after
//! trimming ASCII whitespace; `1`, `true`, or `yes` (any casing) engages the
//! flag. Any other value (including unset) leaves the flag disabled. Most flags
//! are conservative kill switches that force a fallback path, but investigation
//! flags may explicitly opt into unsafe diagnostics and must document that
//! behavior at the accessor.
//!
//! The pattern intentionally mirrors DS4's `ds4_metal_get_*` shape-gated
//! pipeline cache: every fast path declares an explicit predicate, an
//! explicit kill switch, and an explicit fallback. Co-locating the env-var
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

/// **Investigation tunable.** When set, overrides
/// `DEFAULT_PREFILL_CHUNK` for MLA models. Smaller chunks let cold and
/// warm-extend prefill paths produce the same SDPA Q/K shape sequence
/// over the same absolute positions; the hypothesis is that
/// shape-dependent SDPA kernel selection in MLX is the root cause of
/// the warm_extend fp-drift on GLM-4.7-Flash. Returns `None` when the
/// env var is unset or invalid. Value must be a positive integer.
pub fn mla_prefill_chunk_override() -> Option<usize> {
    static CACHED: OnceLock<Option<usize>> = OnceLock::new();
    *CACHED.get_or_init(|| parse_positive_usize_env("AX_MLX_MLA_PREFILL_CHUNK"))
}

env_flag!(
    /// **Investigation override.** Engaged by `AX_ALLOW_MLA_PREFIX_RESTORE`,
    /// this bypasses the `mla_extend_unsafe` safety gate in
    /// `restore_reused_prefix_state`. The gate normally refuses to
    /// restore an MLA snapshot when the request mode is Prefill, because
    /// the post-restore `chunked_prefill` over a suffix has been observed
    /// to drift fp-wise from a cold full-prefill on GLM-4.7-Flash (see
    /// `verify_prefix_reuse_equivalence.py --mode warm_extend` and the
    /// audit comment in `runner.rs`). Setting this flag is intentionally
    /// unsafe for production: it exists so the equivalence harness can
    /// reproduce and isolate the drift, and so a future fix can be
    /// regression-tested against the harness before relaxing the gate
    /// by default.
    mla_prefix_restore_forced,
    "AX_ALLOW_MLA_PREFIX_RESTORE"
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
}
