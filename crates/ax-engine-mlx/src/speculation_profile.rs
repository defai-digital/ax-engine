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
//! The per-profile Gemma gate values are CALIBRATED from the 2026-06-09
//! 12B-4bit-FFN gate ablation (PRD-2026-06-09-gemma4-12b-mtp-speedup R5): the
//! sweep found that lowering the Gemma assistant gate does not improve code
//! throughput (gate 0.90 is −2% to −4.5% vs the 0.999 default at identical
//! accept), so every Gemma posture keeps 0.999 and the presets differentiate via
//! n-gram policy. Qwen keeps its separately-validated 0.90 default. See the
//! constants below for the measured numbers.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

/// Speculative-decode posture selected by `AX_MLX_SPECULATION_PROFILE`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeculationProfile {
    /// Temperature-driven default: built-in defaults at low temperature,
    /// diversity-preserving at high temperature.
    Auto,
    /// Low-temperature, sharply-peaked content. Keeps the calibrated Gemma gate
    /// (0.999 — the ablation showed lowering it does not help code throughput);
    /// Qwen uses its validated 0.90 default.
    Coding,
    /// Low-temperature structured output (tools/JSON/reasoning). Same calibrated
    /// gate as `coding`; reserved for tightened n-gram structured-output safety.
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

// CALIBRATED from the 2026-06-09 12B-4bit-FFN gate ablation
// (`benchmarks/results/gemma4-assistant-mtp/2026-06-09-gemma4-12b-ffn4-mtp-phase4-focused`).
//
// Key finding: lowering the Gemma assistant gate does NOT help throughput on
// code. The greedy assistant drafter is so peaked that gate 0.90 proposes the
// SAME drafts as 0.999 (identical 95.5% accept) but is consistently SLOWER
// (flappy 100.2 vs 102.3, long_code 97.7 vs 99.4, python_modules_long 100.3 vs
// 105.0 tok/s — −2% to −4.5%). This is the opposite of the Qwen MTP head, where
// 0.90 wins. So every Gemma posture keeps the accept-maximizing 0.999 default;
// the speed comes from assistant-MTP itself (~2.8x over direct), not from gate
// loosening. The presets differentiate Gemma only via n-gram policy, and Qwen
// via its own validated gate.
const CODING_GEMMA_FIRST_GATE: f32 = 0.999;
const CODING_GEMMA_DEEP_GATE: f32 = 0.999;
const AGENTIC_GEMMA_FIRST_GATE: f32 = 0.999;
const AGENTIC_GEMMA_DEEP_GATE: f32 = 0.999;
/// Gemma's diversity-preserving gate equals its measured-optimal accept-
/// maximizing default (0.999): `chatbot`/high-temp `auto` keep Gemma as it ships.
const CHATBOT_GEMMA_FIRST_GATE: f32 = 0.999;
const CHATBOT_GEMMA_DEEP_GATE: f32 = 0.999;
/// Qwen's shipped default (0.90) is already the throughput sweet spot
/// (`docs/MTP-DRAFT-GATE-THROUGHPUT.md`), so `coding`/`agentic` defer to it
/// (return `None`); only the diversity regime raises it.
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
    /// defer to the built-in default. `temperature` drives `auto`.
    pub fn gemma_first_gate(self, temperature: Option<f32>) -> Option<f32> {
        match self.effective(temperature)? {
            Self::Coding => Some(CODING_GEMMA_FIRST_GATE),
            Self::Agentic => Some(AGENTIC_GEMMA_FIRST_GATE),
            Self::Chatbot => Some(CHATBOT_GEMMA_FIRST_GATE),
            Self::Auto => None,
        }
    }

    /// Gemma assistant deep-position gate this profile prescribes, or `None`.
    pub fn gemma_deep_gate(self, temperature: Option<f32>) -> Option<f32> {
        match self.effective(temperature)? {
            Self::Coding => Some(CODING_GEMMA_DEEP_GATE),
            Self::Agentic => Some(AGENTIC_GEMMA_DEEP_GATE),
            Self::Chatbot => Some(CHATBOT_GEMMA_DEEP_GATE),
            Self::Auto => None,
        }
    }

    /// Qwen fused MTP gate this profile prescribes, or `None` to keep the shipped
    /// 0.90 default. `coding`/`agentic` defer (0.90 is already the sweet spot);
    /// only the diversity regime raises it.
    pub fn qwen_gate(self, temperature: Option<f32>) -> Option<f32> {
        match self.effective(temperature)? {
            Self::Coding | Self::Agentic => None,
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
    fn coding_keeps_calibrated_gemma_gate_at_any_temperature() {
        // The 2026-06-09 ablation found lowering the Gemma gate hurts code
        // throughput, so `coding` keeps the measured-optimal 0.999 — temperature
        // independent (explicit profile applies regardless of temperature).
        assert_eq!(
            SpeculationProfile::Coding.gemma_first_gate(Some(0.0)),
            Some(0.999)
        );
        assert_eq!(
            SpeculationProfile::Coding.gemma_first_gate(Some(0.9)),
            Some(0.999)
        );
        assert_eq!(
            SpeculationProfile::Agentic.gemma_deep_gate(Some(0.0)),
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
    fn qwen_defers_for_coding_and_agentic() {
        assert_eq!(SpeculationProfile::Coding.qwen_gate(Some(0.0)), None);
        assert_eq!(SpeculationProfile::Agentic.qwen_gate(Some(0.0)), None);
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
