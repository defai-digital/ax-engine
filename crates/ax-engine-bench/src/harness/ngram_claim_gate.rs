//! Same-policy n-gram claim gate (invariant I-6, PRD §8 Phase 6).
//!
//! Given a `direct` (baseline) decode row and an `ngram` candidate row, this
//! module returns whether the n-gram candidate can be promoted as a greedy
//! exact same-policy claim. The pairing requires *identical* model identity,
//! prompt hash, random seed, max token budget, and sampler settings. Token
//! IDs produced under each row must match exactly; the first mismatch index
//! is returned in the failure artifact so downstream tooling can include it
//! verbatim per PRD §7.2.
//!
//! This is the engine-side complement to
//! `scripts/bench_mlx_inference_stack.py::ax_decode_claim_mode`: the bench
//! script labels rows, this module decides whether two labeled rows can be
//! paired into a single same-policy claim. The two surfaces are deliberately
//! kept in different runtimes (Python aggregation lives next to the bench
//! script, Rust gate lives next to the workload fixtures that produce row
//! material) so neither side can silently relax the other's checks.
//!
//! The gate is invoked by `scripts/run-serving-stress.sh` aggregation hooks
//! and by the bench Python pipeline once both sides emit paired baseline /
//! candidate rows. It is intentionally callable from outside this crate so
//! the same code path enforces I-6 for any caller.

#![allow(dead_code)]

use serde::Serialize;
use serde_json::{Value, json};

/// Identifying fields a `direct` and `ngram` row must share for the gate to
/// even consider promotion. Mismatch on any one of these aborts pairing
/// before token comparison.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct RowIdentity {
    pub model_id: String,
    /// Hash of the prompt token stream. The bench harness emits this as a
    /// hex SHA256 prefix; the gate is content-agnostic about the exact
    /// encoding, but the two rows must produce the same string.
    pub prompt_hash: String,
    pub seed: u64,
    pub max_output_tokens: u32,
    /// Encoded sampler configuration. Canonicalized by the producer so
    /// `{temperature: 0}` and `{}` compare equal across rows.
    pub sampler_signature: String,
}

/// Outcome of a single `(direct, ngram)` pairing.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum PromotionDecision {
    /// The candidate can be promoted; downstream consumers may label this
    /// row as a same-policy greedy-exact accelerated row.
    Promoted,
    /// Identity fields differ between baseline and candidate; promotion
    /// requires identical inputs. `reason` carries a human-readable summary
    /// for artifact embedding.
    IdentityMismatch { reason: String },
    /// Identity matched but generated token IDs differ. `first_mismatch_index`
    /// is the zero-based index of the first differing token; `baseline_len`
    /// and `candidate_len` are the full token-stream lengths.
    TokensDiverged {
        first_mismatch_index: usize,
        baseline_len: usize,
        candidate_len: usize,
    },
    /// Caller asked us to promote a sampling-mode candidate. PRD §7.1
    /// forbids this without a probability-ratio acceptance + residual
    /// correction implementation. Refuse fail-closed.
    SamplingModeRefused,
}

impl PromotionDecision {
    /// Stable status label for artifact embedding.
    pub fn status(&self) -> &'static str {
        match self {
            PromotionDecision::Promoted => "promoted",
            PromotionDecision::IdentityMismatch { .. } => "identity_mismatch",
            PromotionDecision::TokensDiverged { .. } => "tokens_diverged",
            PromotionDecision::SamplingModeRefused => "sampling_mode_refused",
        }
    }

    pub fn is_promoted(&self) -> bool {
        matches!(self, PromotionDecision::Promoted)
    }

    pub fn to_artifact_json(&self) -> Value {
        match self {
            PromotionDecision::Promoted => json!({
                "outcome": "promoted",
                "status": self.status(),
            }),
            PromotionDecision::IdentityMismatch { reason } => json!({
                "outcome": "identity_mismatch",
                "status": self.status(),
                "reason": reason,
            }),
            PromotionDecision::TokensDiverged {
                first_mismatch_index,
                baseline_len,
                candidate_len,
            } => json!({
                "outcome": "tokens_diverged",
                "status": self.status(),
                "first_mismatch_index": first_mismatch_index,
                "baseline_len": baseline_len,
                "candidate_len": candidate_len,
            }),
            PromotionDecision::SamplingModeRefused => json!({
                "outcome": "sampling_mode_refused",
                "status": self.status(),
                "reason": "sampling-mode candidate cannot be promoted as distribution-exact \
                           without probability-ratio acceptance + residual correction (PRD §7.1)",
            }),
        }
    }
}

/// Returns the first index at which two token vectors differ. If neither is
/// a prefix of the other, returns the index where they first disagree; if
/// one is a prefix of the other, returns the shorter length.
fn first_mismatch(baseline: &[u32], candidate: &[u32]) -> usize {
    for (i, (a, b)) in baseline.iter().zip(candidate.iter()).enumerate() {
        if a != b {
            return i;
        }
    }
    baseline.len().min(candidate.len())
}

/// Evaluate a same-policy greedy-mode promotion. The `is_sampling_mode` flag
/// short-circuits the gate when either row was produced under
/// `temperature > 0`, `top_p < 1.0`, `top_k > 0`, or `repetition_penalty != 1.0`
/// — see PRD §7.2.
pub fn evaluate_greedy_promotion(
    baseline_identity: &RowIdentity,
    baseline_tokens: &[u32],
    candidate_identity: &RowIdentity,
    candidate_tokens: &[u32],
    is_sampling_mode: bool,
) -> PromotionDecision {
    if is_sampling_mode {
        return PromotionDecision::SamplingModeRefused;
    }
    if baseline_identity != candidate_identity {
        return PromotionDecision::IdentityMismatch {
            reason: identity_mismatch_reason(baseline_identity, candidate_identity),
        };
    }
    if baseline_tokens.len() != candidate_tokens.len() || baseline_tokens != candidate_tokens {
        return PromotionDecision::TokensDiverged {
            first_mismatch_index: first_mismatch(baseline_tokens, candidate_tokens),
            baseline_len: baseline_tokens.len(),
            candidate_len: candidate_tokens.len(),
        };
    }
    PromotionDecision::Promoted
}

fn identity_mismatch_reason(a: &RowIdentity, b: &RowIdentity) -> String {
    let mut parts = Vec::new();
    if a.model_id != b.model_id {
        parts.push(format!("model_id ({} vs {})", a.model_id, b.model_id));
    }
    if a.prompt_hash != b.prompt_hash {
        parts.push("prompt_hash".to_string());
    }
    if a.seed != b.seed {
        parts.push(format!("seed ({} vs {})", a.seed, b.seed));
    }
    if a.max_output_tokens != b.max_output_tokens {
        parts.push(format!(
            "max_output_tokens ({} vs {})",
            a.max_output_tokens, b.max_output_tokens
        ));
    }
    if a.sampler_signature != b.sampler_signature {
        parts.push("sampler_signature".to_string());
    }
    if parts.is_empty() {
        "unspecified identity field".to_string()
    } else {
        parts.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(model: &str, seed: u64, max_tokens: u32) -> RowIdentity {
        RowIdentity {
            model_id: model.to_string(),
            prompt_hash: "deadbeef".to_string(),
            seed,
            max_output_tokens: max_tokens,
            sampler_signature: "greedy".to_string(),
        }
    }

    #[test]
    fn matching_identity_and_tokens_promote() {
        let baseline = id("qwen3", 0, 64);
        let candidate = id("qwen3", 0, 64);
        let tokens = vec![1, 2, 3, 4];
        let outcome =
            evaluate_greedy_promotion(&baseline, &tokens, &candidate, &tokens.clone(), false);
        assert!(outcome.is_promoted());
        assert_eq!(outcome.status(), "promoted");
    }

    #[test]
    fn token_mismatch_records_first_index() {
        let baseline = id("qwen3", 0, 64);
        let candidate = id("qwen3", 0, 64);
        let a = vec![1, 2, 3, 4];
        let b = vec![1, 2, 9, 4];
        let outcome = evaluate_greedy_promotion(&baseline, &a, &candidate, &b, false);
        match outcome {
            PromotionDecision::TokensDiverged {
                first_mismatch_index,
                baseline_len,
                candidate_len,
            } => {
                assert_eq!(first_mismatch_index, 2);
                assert_eq!(baseline_len, 4);
                assert_eq!(candidate_len, 4);
            }
            other => panic!("expected TokensDiverged, got {other:?}"),
        }
    }

    #[test]
    fn token_length_mismatch_returns_diverged() {
        let baseline = id("qwen3", 0, 64);
        let candidate = id("qwen3", 0, 64);
        let a = vec![1, 2, 3];
        let b = vec![1, 2, 3, 4];
        let outcome = evaluate_greedy_promotion(&baseline, &a, &candidate, &b, false);
        match outcome {
            PromotionDecision::TokensDiverged {
                first_mismatch_index,
                baseline_len: 3,
                candidate_len: 4,
            } => {
                // first_mismatch returns min length when one is prefix of other.
                assert_eq!(first_mismatch_index, 3);
            }
            other => panic!("expected TokensDiverged length variant, got {other:?}"),
        }
    }

    #[test]
    fn identity_mismatch_short_circuits_before_tokens() {
        let baseline = id("qwen3", 0, 64);
        let candidate = id("qwen3", 1, 64); // different seed
        let tokens = vec![1, 2, 3, 4];
        let outcome =
            evaluate_greedy_promotion(&baseline, &tokens, &candidate, &tokens.clone(), false);
        match outcome {
            PromotionDecision::IdentityMismatch { reason } => {
                assert!(reason.contains("seed"));
            }
            other => panic!("expected IdentityMismatch, got {other:?}"),
        }
    }

    #[test]
    fn sampling_mode_refused_even_with_matching_inputs() {
        // Critical invariant: a sampling-mode candidate is rejected
        // regardless of token equality. This guards against a regression
        // where a tester might "manually align" outputs and try to
        // promote a sampling row.
        let baseline = id("qwen3", 0, 64);
        let candidate = id("qwen3", 0, 64);
        let tokens = vec![1, 2, 3, 4];
        let outcome =
            evaluate_greedy_promotion(&baseline, &tokens, &candidate, &tokens.clone(), true);
        assert!(matches!(outcome, PromotionDecision::SamplingModeRefused));
        assert_eq!(outcome.status(), "sampling_mode_refused");
    }

    #[test]
    fn artifact_json_carries_required_fields_per_outcome() {
        let promoted = PromotionDecision::Promoted.to_artifact_json();
        assert_eq!(promoted["status"], "promoted");

        let diverged = PromotionDecision::TokensDiverged {
            first_mismatch_index: 5,
            baseline_len: 7,
            candidate_len: 8,
        }
        .to_artifact_json();
        assert_eq!(diverged["first_mismatch_index"], 5);
        assert_eq!(diverged["baseline_len"], 7);
        assert_eq!(diverged["candidate_len"], 8);

        let mismatch = PromotionDecision::IdentityMismatch {
            reason: "seed (0 vs 1)".to_string(),
        }
        .to_artifact_json();
        assert_eq!(mismatch["reason"], "seed (0 vs 1)");

        let sampling = PromotionDecision::SamplingModeRefused.to_artifact_json();
        assert!(
            sampling["reason"]
                .as_str()
                .map(|s| s.contains("sampling-mode"))
                .unwrap_or(false)
        );
    }

    #[test]
    fn identical_identity_reports_no_mismatch_parts() {
        let a = id("qwen3", 0, 64);
        let b = id("qwen3", 0, 64);
        // identity_mismatch_reason is a private helper used only when we
        // already know inputs differ; assertion here protects against a
        // future caller that invokes it on equal inputs.
        let r = identity_mismatch_reason(&a, &b);
        assert_eq!(r, "unspecified identity field");
    }
}
