//! Process-wide kill switches for the opt-in MLX prefix-cache features.
//!
//! Both switches default to off. They live in `ax-engine-core` so the
//! request edge (server) and the runner (MLX) resolve the same value: the
//! edge must not pay to derive semantic cut hints when the runner would
//! ignore them, and the runner must not build a cut plan or an execution
//! signature for a request that carries none.
//!
//! Values are read once per process; the default path is unchanged when a
//! variable is unset, empty, `0`, `false`, `off`, `no`, or junk.

use std::sync::OnceLock;

/// Opt-in value-priced prefix restore (`AX_MLX_PREFIX_RESTORE_VALUE`).
pub const PREFIX_RESTORE_VALUE_ENV: &str = "AX_MLX_PREFIX_RESTORE_VALUE";

/// Opt-in semantic snapshot cut points (`AX_MLX_PREFIX_SEMANTIC_CUTS`).
pub const PREFIX_SEMANTIC_CUTS_ENV: &str = "AX_MLX_PREFIX_SEMANTIC_CUTS";

/// Truthy set shared by the two switches: `1`, `true`, `yes`, `on`
/// (case-insensitive, trimmed).
fn parse_bool_value(raw: &str) -> bool {
    let trimmed = raw.trim_ascii();
    trimmed.eq_ignore_ascii_case("1")
        || trimmed.eq_ignore_ascii_case("true")
        || trimmed.eq_ignore_ascii_case("yes")
        || trimmed.eq_ignore_ascii_case("on")
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|raw| parse_bool_value(&raw))
}

/// `AX_MLX_PREFIX_RESTORE_VALUE` — rank restore candidates by the prefill
/// time they save minus their restore cost.
pub fn restore_value_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag(PREFIX_RESTORE_VALUE_ENV))
}

/// `AX_MLX_PREFIX_SEMANTIC_CUTS` — capture in-place snapshots at
/// assistant-turn starts.
///
/// Opt-in and partition-changing by design: capturing a cut splits the cold
/// prefill at that token position, and the switch is part of the execution
/// signature the durable key carries, so snapshots never cross a setting
/// change. Outputs therefore equal a cold run under the *same* setting, not
/// necessarily the switch-off output (that run uses a different prefill
/// chunk partition).
pub fn semantic_cuts_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag(PREFIX_SEMANTIC_CUTS_ENV))
}

/// True when either switch is on. The request edge uses this to skip
/// byte-offset tokenization and hint construction on the default path.
pub fn any_enabled() -> bool {
    restore_value_enabled() || semantic_cuts_enabled()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bool_values_match_the_flag_contract() {
        for raw in ["1", "true", "TRUE", " yes ", "on"] {
            assert!(parse_bool_value(raw), "{raw:?} must be truthy");
        }
        for raw in ["", "0", "false", "off", "no", "junk", "2"] {
            assert!(!parse_bool_value(raw), "{raw:?} must be falsy");
        }
    }
}
