//! Process-wide kill switches for ax-engine-mlx optimization fast paths.
//!
//! Each accessor reads its environment variable once per process and caches
//! the result in a `OnceLock`. Setting the variable to `1`, `true`, `TRUE`,
//! `yes`, or `YES` engages the kill switch and forces the slow fallback path;
//! any other value (including unset) leaves the fast path enabled.
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
    matches!(
        std::env::var(var).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
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
    /// `AX_DISABLE_TURBOQUANT_FUSED_DECODE=1` forces every layer's TurboQuant
    /// fused-decode candidate to `Disabled`, routing decode through the
    /// full-precision SDPA fallback. Used for same-session A/B against the
    /// fused-decode path without rebuilding the manifest or reloading
    /// weights. The `Disabled` status is recorded via the existing
    /// `record_turboquant_decode_candidate` telemetry, so the fallback
    /// reason is observable without additional logging.
    turboquant_fused_decode_disabled,
    "AX_DISABLE_TURBOQUANT_FUSED_DECODE"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bool_env_treats_truthy_values_as_engaged() {
        for value in ["1", "true", "TRUE", "yes", "YES"] {
            // Each iteration uses a unique env var name so concurrent tests
            // cannot observe each other's writes.
            let var = format!("AX_FASTPATH_TEST_TRUTHY_{value}");
            // SAFETY: each test owns a disjoint set of env-var names. Remove
            // before asserting so a failing assert does not leak the var.
            unsafe {
                std::env::set_var(&var, value);
            }
            let observed = parse_bool_env(&var);
            unsafe {
                std::env::remove_var(&var);
            }
            assert!(observed, "expected truthy for {value:?}");
        }
    }

    #[test]
    fn parse_bool_env_rejects_other_values() {
        for value in ["0", "false", "no", "off", "", "anything"] {
            let var = format!("AX_FASTPATH_TEST_FALSY_{value}");
            unsafe {
                std::env::set_var(&var, value);
            }
            let observed = parse_bool_env(&var);
            unsafe {
                std::env::remove_var(&var);
            }
            assert!(!observed, "expected falsy for {value:?}");
        }
    }

    #[test]
    fn parse_bool_env_unset_is_false() {
        assert!(!parse_bool_env("AX_FASTPATH_TEST_DEFINITELY_UNSET"));
    }
}
