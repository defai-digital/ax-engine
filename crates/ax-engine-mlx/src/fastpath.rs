//! Process-wide kill switches for ax-engine-mlx optimization fast paths.
//!
//! Each accessor reads its environment variable once per process and caches
//! the result in a `OnceLock`. The value is parsed case-insensitively after
//! trimming ASCII whitespace; `1`, `true`, or `yes` (any casing) engages the
//! kill switch and forces the slow fallback path. Any other value (including
//! unset) leaves the fast path enabled.
//!
//! The pattern intentionally mirrors DS4's `ds4_metal_get_*` shape-gated
//! pipeline cache: every fast path declares an explicit predicate, an
//! explicit kill switch, and an explicit fallback. Co-locating the env-var
//! names here gives a single grep target for "which optimizations does the
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

macro_rules! kill_switch {
    ($(#[$meta:meta])* $fn_name:ident, $env_var:literal) => {
        $(#[$meta])*
        pub fn $fn_name() -> bool {
            static CACHED: OnceLock<bool> = OnceLock::new();
            *CACHED.get_or_init(|| parse_bool_env($env_var))
        }
    };
}

kill_switch!(
    /// Engaged by `AX_DISABLE_TURBOQUANT_FUSED_DECODE` (truthy values per
    /// the module-level parser contract). Forces every layer's TurboQuant
    /// fused-decode candidate to `Disabled`, routing decode through the
    /// full-precision SDPA fallback. The `Disabled` status reuses the
    /// existing `record_turboquant_decode_candidate` telemetry bucket, so
    /// the env path is observable without a counter-schema change.
    turboquant_fused_decode_disabled,
    "AX_DISABLE_TURBOQUANT_FUSED_DECODE"
);

kill_switch!(
    /// Engaged by `AX_NO_SPEC` (the CLAUDE.md-documented convention for
    /// forcing greedy direct decode). When set, `MlxRunner::from_artifacts`
    /// ORs this value into the `disable_ngram_acceleration` parameter, so
    /// the env switch is honored uniformly from CLI, server, and SDK entry
    /// points — not just from the bench CLI's argument parsing path. Truthy
    /// values follow the module-level parser contract.
    ngram_acceleration_disabled,
    "AX_NO_SPEC"
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
}
