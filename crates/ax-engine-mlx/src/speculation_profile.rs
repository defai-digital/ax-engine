//! Speculation profile presets (ADR-022).
//!
//! A single selector — `AX_MLX_SPECULATION_PROFILE` (server CLI:
//! `--speculation-profile` / `-s`, hidden alias `--spec`) — bundles the
//! per-knob MTP and n-gram speculative-decode configuration into named
//! postures: `auto`, `coding`, `agentic`, `chatbot`.
//!
//! The profile is a convenience layer over the existing env knobs, not a new
//! decode path. It only selects gate values and n-gram policy among the paths
//! the runtime already has; committed tokens are always the verified target
//! tokens regardless of profile.
//!
//! Resolution precedence (highest first): an explicit per-knob env var > the
//! selected profile preset > the built-in default. So a profile never silently
//! overwrites a value the user set explicitly.
//!
//! `auto` is the default and is **temperature-driven**: at low/zero request
//! temperature it defers entirely to the built-in defaults (no regression, and
//! — per ADR-021 — no speculative lowering of the Gemma assistant gate); at
//! higher temperature it raises gates to protect sampling diversity from the
//! greedy argmax-match bias.
//!
//! Calibration (PRD-2026-06-09-gemma4-12b-mtp-speedup R5): the 2026-06-09 12B
//! ablation showed lowering the Gemma assistant gate does not add code throughput
//! (the greedy drafter is so peaked that the shipped default and a lower gate
//! propose the same drafts at the same accept). So `coding`/`agentic` DEFER to
//! the shipped Gemma default rather than override it. For Qwen, the gate-throughput
//! sweep (`docs/MTP-DRAFT-GATE-THROUGHPUT.md`) shows the optimum varies by
//! workload: `coding` lowers to 0.85, `agentic` to 0.80 (vs the 0.90 default).
//! Only the diversity regime (`chatbot` / high-temperature `auto`) raises the
//! gate, to preserve sampled-chat diversity.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

/// Speculative-decode posture selected by `AX_MLX_SPECULATION_PROFILE`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeculationProfile {
    /// Temperature-driven default: built-in defaults at low temperature,
    /// diversity-preserving at high temperature.
    Auto,
    /// Low-temperature, sharply-peaked content. Defers to the shipped gate
    /// defaults (the ablation showed lowering the Gemma gate does not help code
    /// throughput, and the shipped default already is the throughput setting).
    Coding,
    /// Low-temperature structured output (tools/JSON/reasoning). Same gate as
    /// `coding`; reserved for tightened n-gram structured-output safety.
    Agentic,
    /// Higher-temperature conversational output. Conservative gate / utility
    /// n-gram to protect reply diversity.
    Chatbot,
}

/// How the resolved value was chosen, for route telemetry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResolutionSource {
    /// An explicit per-knob env var was set and won.
    Explicit,
    /// The selected profile preset supplied the value.
    Profile,
    /// Neither applied; the built-in default was used.
    Default,
}

impl ResolutionSource {
    pub fn route_code(self) -> u32 {
        match self {
            Self::Default => 0,
            Self::Profile => 1,
            Self::Explicit => 2,
        }
    }
}

/// Request temperature at/above which `auto` switches from the throughput regime
/// to the diversity-preserving regime.
pub const AUTO_DIVERSITY_TEMPERATURE: f32 = 0.5;

// Informed by the 2026-06-09 12B-4bit-FFN ablation
// (`benchmarks/results/gemma4-assistant-mtp/2026-06-09-gemma4-12b-ffn4-mtp-phase4-focused`):
// assistant-MTP beats direct decode ~2.8x, and lowering the Gemma assistant gate
// does not add code throughput (the greedy drafter is so peaked that the shipped
// default and a lower gate propose the same drafts at the same ~95% accept). So
// `coding`/`agentic` DEFER to the shipped Gemma default (they return `None`)
// instead of overriding it — robust to the default being retuned. Only the
// diversity regime raises the gate, to cut the greedy argmax-match bias on flat
// sampled chat text (Gemma assistant drafts are greedy, so a higher gate makes
// MTP fire less and preserves more sampled diversity).
const CHATBOT_GEMMA_FIRST_GATE: f32 = 0.999;
const CHATBOT_GEMMA_DEEP_GATE: f32 = 0.999;
/// Qwen's shipped default (0.90) is the throughput sweet spot for repetitive
/// workloads (`docs/MTP-DRAFT-GATE-THROUGHPUT.md`). `coding` lowers the gate to
/// 0.85 (the `python_modules_long` optimum, +13.6% tok/fwd); `agentic` lowers
/// to 0.80 (the `long_code` optimum, +17.6% tok/fwd). Only the diversity regime
/// raises the gate above 0.90.
const CODING_QWEN_GATE: f32 = 0.85;
const AGENTIC_QWEN_GATE: f32 = 0.80;
const CHATBOT_QWEN_GATE: f32 = 0.99;

impl SpeculationProfile {
    /// Parse the selector value. Accepts the canonical names plus short aliases.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "coding" | "code" => Some(Self::Coding),
            "agentic" | "agent" => Some(Self::Agentic),
            "chatbot" | "chat" => Some(Self::Chatbot),
            _ => None,
        }
    }

    /// Stable route-telemetry code.
    pub fn route_code(self) -> u32 {
        match self {
            Self::Auto => 0,
            Self::Coding => 1,
            Self::Agentic => 2,
            Self::Chatbot => 3,
        }
    }

    /// True when `auto` should treat this temperature as diversity-sensitive.
    /// `None` (temperature unknown at the call site) is treated as low.
    fn is_diversity_temperature(temperature: Option<f32>) -> bool {
        matches!(temperature, Some(t) if t.is_finite() && t >= AUTO_DIVERSITY_TEMPERATURE)
    }

    /// Concrete regime an explicit profile applies, or what `auto` resolves to at
    /// the given temperature. `auto` at low temperature returns `None` (defer to
    /// built-in defaults — no regression, no speculative Gemma lowering).
    fn effective(self, temperature: Option<f32>) -> Option<Self> {
        match self {
            Self::Auto => {
                if Self::is_diversity_temperature(temperature) {
                    Some(Self::Chatbot)
                } else {
                    None
                }
            }
            other => Some(other),
        }
    }

    /// Gemma assistant first-position gate this profile prescribes, or `None` to
    /// defer to the shipped default. `coding`/`agentic` defer (the ablation found
    /// lowering the Gemma gate does not help code throughput, so the shipped
    /// default already is the throughput setting); only the diversity regime
    /// raises it to cut greedy argmax-match bias on flat sampled text.
    /// `temperature` drives `auto`.
    pub fn gemma_first_gate(self, temperature: Option<f32>) -> Option<f32> {
        match self.effective(temperature)? {
            Self::Coding | Self::Agentic => None,
            Self::Chatbot => Some(CHATBOT_GEMMA_FIRST_GATE),
            Self::Auto => None,
        }
    }

    /// Gemma assistant deep-position gate this profile prescribes, or `None` to
    /// defer to the shipped default (which is already conservative).
    pub fn gemma_deep_gate(self, temperature: Option<f32>) -> Option<f32> {
        match self.effective(temperature)? {
            Self::Coding | Self::Agentic => None,
            Self::Chatbot => Some(CHATBOT_GEMMA_DEEP_GATE),
            Self::Auto => None,
        }
    }

    /// Qwen fused MTP gate this profile prescribes, or `None` to keep the shipped
    /// 0.90 default. `coding` lowers to 0.85 (`python_modules_long` optimum),
    /// `agentic` lowers to 0.80 (`long_code` optimum); only the diversity regime
    /// raises the gate above 0.90.
    pub fn qwen_gate(self, temperature: Option<f32>) -> Option<f32> {
        match self.effective(temperature)? {
            Self::Coding => Some(CODING_QWEN_GATE),
            Self::Agentic => Some(AGENTIC_QWEN_GATE),
            Self::Chatbot => Some(CHATBOT_QWEN_GATE),
            Self::Auto => None,
        }
    }

    /// Whether this profile prefers the n-gram utility gate when the n-gram gate
    /// policy is not explicitly set. Diversity/chatbot prefers utility (prose
    /// rarely benefits from n-gram and stale matches can hurt).
    pub fn prefers_ngram_utility(self, temperature: Option<f32>) -> bool {
        matches!(self.effective(temperature), Some(Self::Chatbot))
    }

    /// Whether this profile tightens n-gram structured-output safety when the
    /// safety mode is not explicitly set. Agentic protects JSON/tool-call syntax.
    pub fn tightens_ngram_safety(self, temperature: Option<f32>) -> bool {
        matches!(self.effective(temperature), Some(Self::Agentic))
    }
}

/// Process-level programmatic override, encoded as `route_code() + 1` so that
/// `0` means "no override". Set once by the SDK/server from the
/// `--speculation-profile` CLI flag (a safe alternative to mutating the process
/// environment, which the server/SDK crates cannot do under `unsafe_code =
/// "forbid"`). It is checked before the env var, so an explicit flag wins over a
/// stale `AX_MLX_SPECULATION_PROFILE` in the environment.
static PROFILE_OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// Install a programmatic speculation-profile override (server/SDK CLI path).
/// Idempotent and last-write-wins; intended to be called once at startup before
/// decoding begins.
pub fn set_speculation_profile_override(profile: SpeculationProfile) {
    PROFILE_OVERRIDE.store(profile.route_code() as u8 + 1, Ordering::Relaxed);
}

fn speculation_profile_override() -> Option<SpeculationProfile> {
    match PROFILE_OVERRIDE.load(Ordering::Relaxed) {
        1 => Some(SpeculationProfile::Auto),
        2 => Some(SpeculationProfile::Coding),
        3 => Some(SpeculationProfile::Agentic),
        4 => Some(SpeculationProfile::Chatbot),
        _ => None,
    }
}

/// Resolved speculation profile: programmatic override (CLI) first, else
/// `AX_MLX_SPECULATION_PROFILE` (cached), else [`SpeculationProfile::Auto`].
pub fn speculation_profile_from_env() -> SpeculationProfile {
    if let Some(profile) = speculation_profile_override() {
        return profile;
    }
    static CACHED: OnceLock<SpeculationProfile> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_SPECULATION_PROFILE")
            .ok()
            .and_then(|raw| SpeculationProfile::parse(&raw))
            .unwrap_or(SpeculationProfile::Auto)
    })
}

/// Generic precedence resolver: explicit value wins, else the profile preset,
/// else the built-in default. Returns the chosen value and its source for
/// telemetry.
pub fn resolve_gate(
    explicit: Option<f32>,
    preset: Option<f32>,
    default: f32,
) -> (f32, ResolutionSource) {
    if let Some(v) = explicit {
        (v, ResolutionSource::Explicit)
    } else if let Some(v) = preset {
        (v, ResolutionSource::Profile)
    } else {
        (default, ResolutionSource::Default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_canonical_and_aliases() {
        assert_eq!(
            SpeculationProfile::parse("auto"),
            Some(SpeculationProfile::Auto)
        );
        assert_eq!(
            SpeculationProfile::parse("CODING"),
            Some(SpeculationProfile::Coding)
        );
        assert_eq!(
            SpeculationProfile::parse("code"),
            Some(SpeculationProfile::Coding)
        );
        assert_eq!(
            SpeculationProfile::parse(" agent "),
            Some(SpeculationProfile::Agentic)
        );
        assert_eq!(
            SpeculationProfile::parse("chat"),
            Some(SpeculationProfile::Chatbot)
        );
        assert_eq!(SpeculationProfile::parse("nonsense"), None);
    }

    #[test]
    fn route_codes_are_stable() {
        assert_eq!(SpeculationProfile::Auto.route_code(), 0);
        assert_eq!(SpeculationProfile::Coding.route_code(), 1);
        assert_eq!(SpeculationProfile::Agentic.route_code(), 2);
        assert_eq!(SpeculationProfile::Chatbot.route_code(), 3);
    }

    #[test]
    fn coding_and_agentic_defer_to_shipped_gemma_default() {
        // The ablation found lowering the Gemma gate does not help code
        // throughput, so coding/agentic defer to the shipped default (None),
        // never overriding it — at any temperature.
        assert_eq!(SpeculationProfile::Coding.gemma_first_gate(Some(0.0)), None);
        assert_eq!(SpeculationProfile::Coding.gemma_first_gate(Some(0.9)), None);
        assert_eq!(
            SpeculationProfile::Agentic.gemma_first_gate(Some(0.0)),
            None
        );
        assert_eq!(SpeculationProfile::Agentic.gemma_deep_gate(Some(0.0)), None);
        // The diversity regime still raises the Gemma gate.
        assert_eq!(
            SpeculationProfile::Chatbot.gemma_first_gate(Some(0.9)),
            Some(0.999)
        );
    }

    #[test]
    fn auto_defers_to_default_at_low_temperature() {
        // No speculative lowering of Gemma's shipped 0.999 default (ADR-021).
        assert_eq!(SpeculationProfile::Auto.gemma_first_gate(Some(0.0)), None);
        assert_eq!(SpeculationProfile::Auto.gemma_first_gate(None), None);
        assert_eq!(SpeculationProfile::Auto.qwen_gate(Some(0.2)), None);
    }

    #[test]
    fn auto_protects_diversity_at_high_temperature() {
        // Gemma stays at its conservative default value; Qwen is raised from 0.90.
        assert_eq!(
            SpeculationProfile::Auto.gemma_first_gate(Some(0.7)),
            Some(0.999)
        );
        assert_eq!(SpeculationProfile::Auto.qwen_gate(Some(0.7)), Some(0.99));
    }

    #[test]
    fn qwen_workload_optimal_gates_for_coding_and_agentic() {
        // `docs/MTP-DRAFT-GATE-THROUGHPUT.md`: coding → 0.85 (python_modules_long
        // optimum), agentic → 0.80 (long_code optimum).
        assert_eq!(SpeculationProfile::Coding.qwen_gate(Some(0.0)), Some(0.85));
        assert_eq!(SpeculationProfile::Agentic.qwen_gate(Some(0.0)), Some(0.80));
        assert_eq!(SpeculationProfile::Chatbot.qwen_gate(Some(0.0)), Some(0.99));
    }

    #[test]
    fn ngram_policy_hints_match_profiles() {
        assert!(SpeculationProfile::Agentic.tightens_ngram_safety(Some(0.0)));
        assert!(!SpeculationProfile::Coding.tightens_ngram_safety(Some(0.0)));
        assert!(SpeculationProfile::Chatbot.prefers_ngram_utility(Some(0.0)));
        assert!(SpeculationProfile::Auto.prefers_ngram_utility(Some(0.7)));
        assert!(!SpeculationProfile::Auto.prefers_ngram_utility(Some(0.0)));
    }

    #[test]
    fn override_maps_round_trip() {
        // Exercises the encoding used by the programmatic override without
        // mutating the shared static (which other tests in this binary rely on
        // being unset). The decode arm mirrors `speculation_profile_override`.
        for (p, code) in [
            (SpeculationProfile::Auto, 1u8),
            (SpeculationProfile::Coding, 2),
            (SpeculationProfile::Agentic, 3),
            (SpeculationProfile::Chatbot, 4),
        ] {
            assert_eq!(p.route_code() as u8 + 1, code);
        }
    }

    #[test]
    fn resolve_gate_precedence() {
        assert_eq!(
            resolve_gate(Some(0.95), Some(0.90), 0.999),
            (0.95, ResolutionSource::Explicit)
        );
        assert_eq!(
            resolve_gate(None, Some(0.90), 0.999),
            (0.90, ResolutionSource::Profile)
        );
        assert_eq!(
            resolve_gate(None, None, 0.999),
            (0.999, ResolutionSource::Default)
        );
    }
}
