//! Batched-decode certification policy (WS-T1 / R-T1 Decision A).
//!
//! Product decision: **bit-exact B>1** vs per-row greedy is required for
//! certification. When MoE amortization conflicts with per-row numerics,
//! use deterministic reduction and **per-row / RowExact residual fallback** —
//! amortization yields. See ADR-010 and
//! `docs/performance/batched-hybrid-moe-linear-decode.md`.

/// Certification mode for hybrid MoE / linear families.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchedMoECertMode {
    /// Bit-exact vs per-row; residual shapes fall back to RowExact.
    BitExactWithRowExactFallback,
}

impl BatchedMoECertMode {
    pub const DECISION_A: Self = Self::BitExactWithRowExactFallback;

    pub fn label(self) -> &'static str {
        match self {
            Self::BitExactWithRowExactFallback => "bit_exact_row_exact_fallback",
        }
    }

    /// Required batch sizes for release certification (PRD M6).
    pub fn required_batches(self) -> &'static [u32] {
        match self {
            Self::BitExactWithRowExactFallback => &[2, 4, 8],
        }
    }

    /// Whether an uncertified opt-in env may bypass the gate after Decision A.
    /// Always false for public claims; residual only via explicit bench tools.
    pub fn allows_uncertified_public_default(self) -> bool {
        false
    }
}

/// True when the model is in the Decision A certification scope: hybrid
/// Qwen 3.5 / 3.6 families always, and plain Qwen 3 (`qwen3`) when the model
/// is MoE (`has_moe`). Dense Qwen 3 is out of scope — HF `qwen3_moe` and
/// dense `qwen3` share the manifest family label, so the MoE config
/// disambiguates.
pub fn family_requires_decision_a_cert(model_family: &str, has_moe: bool) -> bool {
    if matches!(
        model_family,
        "qwen3_5" | "qwen3_next" | "qwen3.5" | "qwen3.6" | "qwen3_6"
    ) {
        return true;
    }
    has_moe && model_family == "qwen3"
}

/// Expert accumulation order for deterministic batched MoE reduction.
/// Fixed ascending expert-id order matches a common per-row reference when
/// each row's active set is sorted identically before scatter.
pub fn deterministic_expert_order(active_expert_ids: &mut [u32]) {
    active_expert_ids.sort_unstable();
}

/// Env: `AX_MLX_BATCHED_MOE_ROW_EXACT` — when on (default for Decision A
/// models), `ffn_batched` runs MoE per-row instead of shared `gather_qmm`.
/// Opt out with `=0` for uncertified amortized throughput experiments.
pub const ENV_BATCHED_MOE_ROW_EXACT: &str = "AX_MLX_BATCHED_MOE_ROW_EXACT";

/// True when batched MoE must use per-row (RowExact) expert execution for
/// bit-exact greedy parity (Decision A). `has_moe` comes from the model's
/// MoE config (e.g. `ModelConfig::moe_expert_count > 0`).
pub fn row_exact_moe_enabled(model_family: &str, has_moe: bool) -> bool {
    row_exact_moe_enabled_with_env(
        model_family,
        has_moe,
        std::env::var(ENV_BATCHED_MOE_ROW_EXACT).ok().as_deref(),
    )
}

fn row_exact_moe_enabled_with_env(
    model_family: &str,
    has_moe: bool,
    env_value: Option<&str>,
) -> bool {
    if let Some(v) = env_value {
        let v = v.trim().to_ascii_lowercase();
        if matches!(v.as_str(), "0" | "false" | "off" | "no") {
            return false;
        }
        if matches!(v.as_str(), "1" | "true" | "on" | "yes") {
            return true;
        }
    }
    // Default: Decision A models use RowExact so B>1 cert is achievable.
    family_requires_decision_a_cert(model_family, has_moe)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_a_batches_and_label() {
        let mode = BatchedMoECertMode::DECISION_A;
        assert_eq!(mode.required_batches(), &[2, 4, 8]);
        assert_eq!(mode.label(), "bit_exact_row_exact_fallback");
        assert!(!mode.allows_uncertified_public_default());
    }

    #[test]
    fn decision_a_scope_is_moe_aware_for_plain_qwen3() {
        // Hybrid families are always in scope.
        assert!(family_requires_decision_a_cert("qwen3_next", true));
        assert!(family_requires_decision_a_cert("qwen3_5", true));
        // Plain Qwen3-MoE joins the scope; dense Qwen3 stays out.
        assert!(family_requires_decision_a_cert("qwen3", true));
        assert!(!family_requires_decision_a_cert("qwen3", false));
        // Other MoE families remain out of Decision A scope.
        assert!(!family_requires_decision_a_cert("gemma4", true));
    }

    #[test]
    fn row_exact_defaults_follow_decision_a_scope() {
        assert!(row_exact_moe_enabled_with_env("qwen3", true, None));
        assert!(!row_exact_moe_enabled_with_env("qwen3", false, None));
        assert!(row_exact_moe_enabled_with_env("qwen3_next", true, None));
        assert!(!row_exact_moe_enabled_with_env("gemma4", true, None));
    }

    #[test]
    fn row_exact_env_override_wins_over_default() {
        assert!(!row_exact_moe_enabled_with_env("qwen3", true, Some("0")));
        assert!(!row_exact_moe_enabled_with_env(
            "qwen3_next",
            true,
            Some("off")
        ));
        assert!(row_exact_moe_enabled_with_env("qwen3", false, Some("1")));
        assert!(row_exact_moe_enabled_with_env("gemma4", true, Some("yes")));
        // Unrecognized values fall through to the default.
        assert!(row_exact_moe_enabled_with_env("qwen3", true, Some("maybe")));
        assert!(!row_exact_moe_enabled_with_env(
            "qwen3",
            false,
            Some("maybe")
        ));
    }

    #[test]
    fn expert_order_is_stable() {
        let mut ids = [7, 2, 5, 2];
        deterministic_expert_order(&mut ids);
        assert_eq!(ids, [2, 2, 5, 7]);
    }
}
