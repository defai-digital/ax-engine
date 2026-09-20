//! Bounded first-divergence collection, separate from MTP qualification.

use super::*;
use crate::model::qwen4_exp_mtp::{
    CandidateSession, CandidateStepObservation, mtp_parity, top_two_margin,
};
use crate::weights::{qwen4_exp::Qwen4ExpWeights, qwen4_exp_mtp::Qwen4ExpMtpWeights};
use serde_json::{Value, json};

fn state_rows(a: &qwen4_exp::Qwen4ExpState, b: &qwen4_exp::Qwen4ExpState) -> Value {
    mtp_parity::mtp_state_arrays_json(&mtp_parity::mtp_state_array_records(a, b))
}

fn logits_summary(logits: &MlxArray) -> Value {
    let token = mlx_sys::argmax(logits, None);
    mlx_sys::eval(&[&token]);
    json!({"token": token.data_u32()[0],
        "top_two_margin": top_two_margin(logits, 0).ok(),
        "margin_error": top_two_margin(logits, 0).err(),
        "logits_dtype": format!("{:?}", logits.dtype())})
}

fn logits_row(output: &qwen4_exp::Qwen4ExpOutput) -> Value {
    logits_summary(&output.logits)
}

fn logits_comparison(a: &MlxArray, b: &MlxArray) -> Value {
    let difference = mtp_parity::mtp_logits_divergence(a, b);
    json!({"actual_verifier": logits_summary(a), "checkpoint_singleton": logits_summary(b),
           "max_abs": difference.max_abs, "relative": difference.relative})
}

fn floating_array_exact(a: &MlxArray, b: &MlxArray) -> bool {
    if a.dtype() != b.dtype() || a.shape() != b.shape() {
        return false;
    }
    let a = mlx_sys::contiguous(&astype(a, MlxDtype::Float32, None), None);
    let b = mlx_sys::contiguous(&astype(b, MlxDtype::Float32, None), None);
    mlx_sys::try_eval(&[&a, &b]).unwrap();
    a.data_f32()
        .iter()
        .zip(b.data_f32())
        .all(|(a, b)| a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits())
}

fn observation_exact(
    a: &CandidateStepObservation,
    b: &CandidateStepObservation,
    layers: usize,
) -> bool {
    a.target_schedule == b.target_schedule
        && a.draft_token == b.draft_token
        && if a.target_schedule == "canonical_singleton" {
            floating_array_exact(&a.next_logits, &b.next_logits)
        } else {
            mtp_parity::mtp_logits_divergence(&a.next_logits, &b.next_logits).max_abs == 0.0
        }
        && match (&a.verification_logits, &b.verification_logits) {
            (Some(a), Some(b)) => mtp_parity::mtp_logits_divergence(a, b).max_abs == 0.0,
            (None, None) => true,
            _ => false,
        }
        && match (
            &a.canonical_correction_logits,
            &b.canonical_correction_logits,
        ) {
            (Some(a), Some(b)) => floating_array_exact(a, b),
            (None, None) => true,
            _ => false,
        }
        && match (&a.canonical_primary, &b.canonical_primary) {
            (Some(a), Some(b)) => {
                flash_mtp_state_bytes(&a.state, layers) == flash_mtp_state_bytes(&b.state, layers)
                    && floating_array_exact(&a.stream_hidden, &b.stream_hidden)
            }
            (None, None) => true,
            _ => false,
        }
}

fn difference_value(position: usize, source: &str, logits: &MlxArray, mtp_token: u32) -> Value {
    let values = mlx_sys::contiguous(&astype(logits, MlxDtype::Float32, None), None);
    mlx_sys::try_eval(&[&values]).unwrap();
    assert_eq!(values.shape().len(), 2);
    assert_eq!(values.shape()[0], 1);
    let values = values.data_f32();
    let nonfinite = values.iter().filter(|v| !v.is_finite()).count();
    let token = mlx_sys::argmax(logits, None);
    mlx_sys::eval(&[&token]);
    let token = token.data_u32()[0];
    assert_ne!(token, mtp_token);
    json!({"position": position, "source": source, "mtp_token": mtp_token,
        "direct": {"token": token, "top_two_margin": top_two_margin(logits, 0).ok(),
            "margin_error": top_two_margin(logits, 0).err(), "nonfinite_logits": nonfinite,
            "logits_dtype": format!("{:?}", logits.dtype()), "top1_logit": values[token as usize],
            "mtp_token_logit": values[mtp_token as usize]}})
}

fn same_session(a: &CandidateSession, b: &CandidateSession, layers: usize) -> bool {
    a.primary == b.primary
        && floating_array_exact(&a.primary_logits, &b.primary_logits)
        && a.proposed == b.proposed
        && a.accepted == b.accepted
        && flash_mtp_state_bytes(&a.trunk_state, layers)
            == flash_mtp_state_bytes(&b.trunk_state, layers)
        && flash_mtp_state_bytes(&a.draft_state, 1) == flash_mtp_state_bytes(&b.draft_state, 1)
        && mtp_parity::mtp_logits_divergence(&a.stream_hidden, &b.stream_hidden).max_abs == 0.0
}

struct DiagnosticModel<'a> {
    trunk: &'a Qwen4ExpWeights,
    head: &'a Qwen4ExpMtpWeights,
    owner: u64,
    terminal_ids: &'a [u32],
}

impl DiagnosticModel<'_> {
    fn singleton(&self, token: u32, state: &qwen4_exp::Qwen4ExpState) -> qwen4_exp::Qwen4ExpOutput {
        qwen4_exp::forward(
            self.trunk,
            &[token],
            state,
            self.owner,
            ProjectionBatchPolicy::Shared,
        )
        .unwrap()
    }

    /// Compare observed verifier rows with identical checkpoint singletons.
    /// A fresh proposal is never substituted for the proposal actually used.
    #[allow(clippy::too_many_arguments)]
    fn replay(
        &self,
        before: &CandidateSession,
        actual: &CandidateSession,
        consumed: &[u32],
        observation: &CandidateStepObservation,
        direct_before: &qwen4_exp::Qwen4ExpState,
        direct_after: &qwen4_exp::Qwen4ExpOutput,
        remaining: usize,
    ) -> Value {
        let mut repeat = before.clone();
        let (repeated, repeat_observation) = repeat
            .step_observed(self.trunk, self.head, remaining, self.terminal_ids)
            .unwrap();
        let stable = repeated == consumed
            && same_session(actual, &repeat, self.trunk.layers.len())
            && observation_exact(observation, &repeat_observation, self.trunk.layers.len());
        let first = self.singleton(before.primary, &before.trunk_state);
        let first_repeat = self.singleton(before.primary, &before.trunk_state);
        let singleton_stable =
            mtp_parity::mtp_logits_divergence(&first.logits, &first_repeat.logits).max_abs == 0.0
                && flash_mtp_state_bytes(&first.state, self.trunk.layers.len())
                    == flash_mtp_state_bytes(&first_repeat.state, self.trunk.layers.len());
        let second = observation
            .draft_token
            .map(|draft| self.singleton(draft, &first.state));
        let verifier_rows = observation.verification_logits.as_ref().map(|logits| {
            let width = logits.shape()[1];
            let row0 = slice(logits, &[0, 0], &[1, width], &[1, 1], None);
            let row1 = slice(logits, &[1, 0], &[2, width], &[1, 1], None);
            json!({"correction": logits_comparison(&row0, &first.logits),
                "bonus": logits_comparison(&row1, &second.as_ref().unwrap().logits)})
        });
        let canonical_rows = observation
            .canonical_correction_logits
            .as_ref()
            .map(|logits| {
                json!({"correction": logits_comparison(logits, &first.logits),
                "bonus": (consumed.len() == 2).then(|| logits_comparison(
                    &observation.next_logits, &second.as_ref().unwrap().logits))})
            });
        let count = direct_after.state.position() - direct_before.position();
        assert!(count > 0 && count <= consumed.len());
        let mut direct_repeat = self.singleton(consumed[0], direct_before);
        for &token in &consumed[1..count] {
            direct_repeat = self.singleton(token, &direct_repeat.state);
        }
        let direct_stable =
            mtp_parity::mtp_logits_divergence(&direct_repeat.logits, &direct_after.logits).max_abs
                == 0.0
                && flash_mtp_state_bytes(&direct_repeat.state, self.trunk.layers.len())
                    == flash_mtp_state_bytes(&direct_after.state, self.trunk.layers.len());
        let committed_singleton = if consumed.len() == 2 {
            second.as_ref().unwrap()
        } else {
            &first
        };
        json!({
            "same_checkpoint_session_repeat_exact": stable,
            "same_checkpoint_mtp_state_singleton_repeat_exact": singleton_stable,
            "same_checkpoint_direct_trajectory_repeat_exact": direct_stable,
            "direct_repeated_tokens": &consumed[..count],
            "remaining": remaining, "primary": before.primary, "actual_draft": observation.draft_token,
            "accepted": consumed.len() == 2, "consumed": consumed,
            "actual_next": logits_summary(&observation.next_logits),
            "target_schedule": observation.target_schedule,
            "actual_verifier_rows": verifier_rows,
            "actual_canonical_singleton_rows": canonical_rows,
            "before_common_prefix_state_arrays": state_rows(&before.trunk_state, direct_before),
            "same_checkpoint_post_state_arrays": state_rows(&actual.trunk_state, &committed_singleton.state),
            "note": "Verifier logits and proposal are captured from the actual test session, including rejected windows. Repeated singleton rows use identical tokens/checkpoints. Full-window row state is never treated as a one-token state.",
        })
    }

    fn collect(&self, tokens: &[u32], max_new: usize, mut save: impl FnMut(&Value)) -> Value {
        assert!(!tokens.is_empty() && tokens.len() <= 2048);
        assert!((3..=256).contains(&max_new));
        let mut session =
            CandidateSession::prefill(self.trunk, self.head, tokens, self.owner, self.owner + 1)
                .unwrap();
        let repeat =
            CandidateSession::prefill(self.trunk, self.head, tokens, self.owner, self.owner + 1)
                .unwrap();
        let prefill_stable = same_session(&session, &repeat, self.trunk.layers.len());
        let prefix = if tokens.len() > 1 {
            qwen4_exp::forward(
                self.trunk,
                &tokens[..tokens.len() - 1],
                &qwen4_exp::Qwen4ExpState::new(self.trunk, self.owner),
                self.owner,
                ProjectionBatchPolicy::Shared,
            )
            .unwrap()
            .state
        } else {
            qwen4_exp::Qwen4ExpState::new(self.trunk, self.owner)
        };
        let mut direct = self.singleton(*tokens.last().unwrap(), &prefix);
        let mut full_state_identity =
            flash_mtp_state_bytes(&session.trunk_state, self.trunk.layers.len())
                == flash_mtp_state_bytes(&direct.state, self.trunk.layers.len());
        let mut stream_hidden_identity =
            floating_array_exact(&session.stream_hidden, &direct.stream_hidden);
        let mut canonical_logits_identity =
            floating_array_exact(&session.primary_logits, &direct.logits);
        let prefill = json!({"repeat_session_exact": prefill_stable,
            "direct": logits_row(&direct), "mtp_token": session.primary,
            "serialized_state_exact": full_state_identity,
            "stream_hidden_exact": stream_hidden_identity,
            "logits_exact": canonical_logits_identity,
            "state_arrays": state_rows(&session.trunk_state, &direct.state)});
        let mut matched = Vec::new();
        let mut steps = Vec::new();
        let mut difference = None;
        let mut stopped_at_terminal = false;
        let mut first_state_difference = None;
        if session.primary != flash_next_mtp_greedy_token(&direct) {
            difference = Some(difference_value(
                0,
                "prefill",
                &direct.logits,
                session.primary,
            ));
        }
        while difference.is_none() && matched.len() < max_new {
            if self.terminal_ids.contains(&session.primary) {
                matched.push(session.primary);
                stopped_at_terminal = true;
                break;
            }
            let before = session.clone();
            let direct_before = direct.state.clone();
            let remaining = max_new - matched.len() - 1;
            if remaining == 0 {
                matched.push(session.primary);
                break;
            }
            let start = matched.len();
            let (consumed, observation) = session
                .step_observed(self.trunk, self.head, remaining, self.terminal_ids)
                .unwrap();
            assert!(!consumed.is_empty() && consumed.len() <= remaining);
            let mut rows = Vec::new();
            let mut primary_exact = None;
            for (index, &token) in consumed.iter().enumerate() {
                rows.push(logits_row(&direct));
                if token != flash_next_mtp_greedy_token(&direct) {
                    difference = Some(difference_value(
                        matched.len(),
                        if index == 0 { "primary" } else { "correction" },
                        &direct.logits,
                        token,
                    ));
                    break;
                }
                matched.push(token);
                if self.terminal_ids.contains(&token) {
                    stopped_at_terminal = true;
                    break;
                }
                direct = self.singleton(token, &direct.state);
                if index == 0
                    && let (Some(primary), Some(logits)) = (
                        &observation.canonical_primary,
                        &observation.canonical_correction_logits,
                    )
                {
                    let state_exact =
                        flash_mtp_state_bytes(&primary.state, self.trunk.layers.len())
                            == flash_mtp_state_bytes(&direct.state, self.trunk.layers.len());
                    let hidden_exact =
                        floating_array_exact(&primary.stream_hidden, &direct.stream_hidden);
                    let logits_exact = floating_array_exact(logits, &direct.logits);
                    if !state_exact
                        && first_state_difference.is_none()
                        && mtp_parity::mtp_state_divergence(&primary.state, &direct.state).max_abs
                            > 0.0
                    {
                        first_state_difference =
                            Some(json!({"generated_prefix_length": matched.len(),
                            "state_arrays": state_rows(&primary.state, &direct.state)}));
                    }
                    full_state_identity &= state_exact;
                    stream_hidden_identity &= hidden_exact;
                    canonical_logits_identity &= logits_exact;
                    primary_exact = Some(json!({"generated_prefix_length": matched.len(),
                        "state": state_exact, "stream_hidden": hidden_exact,
                        "logits": logits_exact}));
                }
            }
            if difference.is_none()
                && !stopped_at_terminal
                && matched.len() < max_new
                && session.primary != flash_next_mtp_greedy_token(&direct)
            {
                let value = difference_value(
                    matched.len(),
                    if consumed.len() == 2 {
                        "bonus"
                    } else if observation.target_schedule == "canonical_singleton" {
                        "correction"
                    } else {
                        "singleton_replay"
                    },
                    &direct.logits,
                    session.primary,
                );
                difference = Some(value);
            }
            let aligned_prefix = matched.len() == start + consumed.len()
                && direct.state.position() == session.trunk_state.position();
            let compare_state = (difference.is_none() && !stopped_at_terminal)
                || (observation.target_schedule == "canonical_singleton" && aligned_prefix);
            let state_difference = if compare_state {
                let d = mtp_parity::mtp_state_divergence(&session.trunk_state, &direct.state);
                if d.max_abs > 0.0 && first_state_difference.is_none() {
                    first_state_difference = Some(json!({"generated_prefix_length": matched.len(),
                        "state_arrays": state_rows(&session.trunk_state, &direct.state)}));
                }
                Some(json!({"max_abs": d.max_abs, "relative": d.relative}))
            } else {
                None
            };
            let full_state_exact = compare_state.then(|| {
                flash_mtp_state_bytes(&session.trunk_state, self.trunk.layers.len())
                    == flash_mtp_state_bytes(&direct.state, self.trunk.layers.len())
            });
            full_state_identity &= full_state_exact.unwrap_or(true);
            let stream_hidden_exact = compare_state
                .then(|| floating_array_exact(&session.stream_hidden, &direct.stream_hidden));
            stream_hidden_identity &= stream_hidden_exact.unwrap_or(true);
            let actual_next_logits_exact = aligned_prefix
                .then(|| floating_array_exact(&observation.next_logits, &direct.logits));
            canonical_logits_identity &= actual_next_logits_exact.unwrap_or(true);
            steps.push(json!({"position": start, "remaining": remaining,
                "target_schedule": observation.target_schedule,
                "consumed": consumed, "next_primary": session.primary, "direct_rows": rows,
                "actual_draft": observation.draft_token, "actual_next": logits_summary(&observation.next_logits),
                "common_prefix_state_difference": state_difference,
                "common_prefix_serialized_state_exact": full_state_exact,
                "common_prefix_stream_hidden_exact": stream_hidden_exact,
                "canonical_primary_exact": primary_exact,
                "actual_next_logits_exact": actual_next_logits_exact,
                "proposed": session.proposed, "accepted": session.accepted}));
            if difference.is_some() {
                save(&json!({"prompt_ids": tokens, "max_new_tokens": max_new,
                    "collection_complete": false, "phase": "checkpoint_replay_pending",
                    "qualification": false, "release_ready": false, "greedy_identity": false,
                    "matching_generated_prefix_ids": matched, "compared_equal_positions": matched.len(),
                    "prefill": prefill, "steps": steps, "first_difference": difference,
                    "first_state_difference": first_state_difference}));
            }
            if let Some(value) = difference.as_mut() {
                value["checkpoint_replay"] = std::panic::catch_unwind(std::panic::AssertUnwindSafe(||
                    self.replay(&before, &session, &consumed, &observation, &direct_before, &direct, remaining)
                )).unwrap_or_else(|_| json!({"error": "checkpoint replay failed; see native stderr", "completed": false}));
            }
            if stopped_at_terminal {
                break;
            }
        }
        json!({"prompt_ids": tokens, "max_new_tokens": max_new,
            "collection_complete": true, "phase": "complete",
            "matching_generated_prefix_ids": matched, "compared_equal_positions": matched.len(),
            "greedy_identity": difference.is_none(), "first_difference": difference,
            "prefill": prefill, "steps": steps, "stopped_at_terminal": stopped_at_terminal,
            "first_state_difference": first_state_difference,
            "compared_full_state_identity": full_state_identity,
            "compared_stream_hidden_identity": stream_hidden_identity,
            "compared_canonical_logits_identity": canonical_logits_identity,
            "proposed": session.proposed, "accepted": session.accepted,
            "qualification": false, "release_ready": false,
            "scope": "Stops at first actual token difference; no tolerance or near-tie acceptance. No HTTP trajectory identity is inferred."})
    }
}

#[test]
#[ignore = "requires an explicit Flash Next pack, prompt manifest and result path"]
fn flash_next_first_token_divergence() {
    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_REAL_PACK").unwrap());
    let manifest: FlashNextMtpOracleManifest = serde_json::from_str(
        &std::fs::read_to_string(std::env::var_os("AX_FLASH_NEXT_PROMPT_MANIFEST").unwrap())
            .unwrap(),
    )
    .unwrap();
    assert!(!manifest.requests.is_empty() && manifest.requests.len() <= 8);
    assert!(manifest.requests.iter().all(|p| p.expected_ids.is_none()));
    let max_new = manifest.max_new_tokens.unwrap_or(256);
    assert!((3..=256).contains(&max_new));
    let path = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_RESULT_PATH").unwrap());
    assert!(!path.exists(), "preserve previous diagnostic output");
    let mut result = json!({"schema": "ax.flash_next.first_token_diagnostic.v1",
        "qualification": false, "release_ready": false, "collection_complete": false,
        "phase": "loading", "requests": []});
    save_record(&path, &result);
    let artifacts = if let Some(path) = std::env::var_os("AX_FLASH_NEXT_NATIVE_MANIFEST") {
        let manifest = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        NativeModelArtifacts::from_manifest_and_root(root.clone(), manifest).unwrap()
    } else {
        NativeModelArtifacts::from_dir(&root).unwrap()
    };
    let trunk = crate::weights::qwen4_exp::load(&root, artifacts.manifest()).unwrap();
    result["target_schedule"] = json!(crate::model::qwen4_exp_mtp::target_schedule_name(&trunk));
    let head = crate::weights::qwen4_exp_mtp::load(&root, artifacts.manifest(), &trunk).unwrap();
    let terminal_ids = flash_next_mtp_load_terminal_ids(&root);
    result["terminal_ids"] = json!(terminal_ids);
    result["terminal_source"] = json!(if std::env::var_os(FLASH_NEXT_TERMINAL_IDS_ENV).is_some() {
        "explicit environment override"
    } else {
        "pack generation_config/config"
    });
    result["prefill_schedule"] =
        json!("single whole n-1 prefix, then singleton; no server chunk dispatcher");
    result["budget_schedule"] = json!(
        "pending primary is already emitted for remaining-output accounting; no final-budget forward"
    );
    result["native_tensor_metadata"] = json!(artifacts.tensor_specs());
    result["loaded_lm_head"] = json!({"mode": trunk.lm_head.mode, "bits": trunk.lm_head.bits,
        "group_size": trunk.lm_head.group_size, "weight_dtype": format!("{:?}", trunk.lm_head.weight.dtype()),
        "scales_dtype": trunk.lm_head.scales.as_ref().map(|a| format!("{:?}", a.dtype()))});
    result["fastpath_scope"] = json!({
        "qwen_linear_mtp_exact": crate::fastpath::qwen_linear_mtp_exact_enabled(),
        "qwen_linear_mtp_target_verify": crate::fastpath::qwen_linear_mtp_target_verify_enabled(),
        "qwen_linear_mtp_verify_fast_kernels": crate::fastpath::qwen_linear_mtp_verify_fast_kernels_enabled(),
        "invariant_mxfp4_qmv_fast": crate::fastpath::invariant_mxfp4_qmv_fast_enabled(),
        "scope": "native test thread; does not enter runner guards"});
    let keys = [
        "AX_ENGINE_FLASH_NEXT_EXPERIMENTAL",
        "AX_MLX_FLASH_NEXT_SELECTED_EXPERTS",
        "AX_MLX_FLASH_NEXT_SELECTED_PREFILL",
        "AX_STREAM_EXPERTS",
        "AX_STREAM_EXPERT_LAYERS",
        "AX_MLX_INVARIANT_MXFP4_QMV_FAST",
        "AX_MLX_DENSE_WIDE_GEMV",
        "AX_MLX_EXACT_DENSE_WEIGHT_T_GEMV",
    ];
    result["environment_overrides"] = json!(
        keys.iter()
            .filter_map(|key| std::env::var(key).ok().map(|value| (*key, value)))
            .collect::<std::collections::BTreeMap<_, _>>()
    );
    result["phase"] = json!("collecting");
    save_record(&path, &result);
    for (index, prompt) in manifest.requests.iter().enumerate() {
        let model = DiagnosticModel {
            trunk: &trunk,
            head: &head,
            owner: 7300 + 2 * index as u64,
            terminal_ids: &terminal_ids,
        };
        result["requests"].as_array_mut().unwrap().push(json!({
            "id": prompt.id, "prompt_ids": prompt.prompt_ids, "collection_complete": false,
            "phase": "prefill", "qualification": false, "release_ready": false}));
        save_record(&path, &result);
        let mut request = model.collect(&prompt.prompt_ids, max_new, |partial| {
            result["requests"][index] = partial.clone();
            result["requests"][index]["id"] = json!(prompt.id);
            save_record(&path, &result);
        });
        request["id"] = json!(prompt.id);
        result["requests"][index] = request;
        save_record(&path, &result);
    }
    result["collection_complete"] = json!(true);
    result["phase"] = json!("complete");
    save_record(&path, &result);
}

#[test]
fn first_difference_retains_ties_and_wide_margins() {
    for data in [[2.0f32, 2.0, 0.0], [3.0, 1.0, 0.0]] {
        let logits = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(&data),
            &[1, 3],
            MlxDtype::Float32,
        );
        let event = difference_value(87, "bonus", &logits, 2);
        assert_eq!(event["position"], 87);
        assert_eq!(event["mtp_token"], 2);
        assert_eq!(event["direct"]["mtp_token_logit"], 0.0);
        assert_eq!(
            event["direct"]["top_two_margin"],
            (data[0] - data[1]) as f64
        );
        assert!(event.get("accepted").is_none());
    }
}

#[test]
#[ignore = "requires generated tiny Flash Next MTP fixtures"]
fn first_difference_synthetic_budget_and_terminal() {
    synthetic_budget_and_terminal(false);
}

#[test]
#[ignore = "requires generated tiny Flash Next MTP fixtures"]
fn canonical_first_difference_synthetic_budget_and_terminal() {
    synthetic_budget_and_terminal(true);
}

fn synthetic_budget_and_terminal(canonical: bool) {
    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_MTP_ORACLE_DIR").unwrap());
    let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
    manifest.weight_sanitize = WeightSanitize::HfToMlx;
    manifest.runtime_status = NativeRuntimeStatus::default();
    let mut trunk = crate::weights::qwen4_exp::load(&root, &manifest).unwrap();
    if canonical {
        trunk.target_schedule =
            crate::weights::qwen4_exp::Qwen4ExpTargetSchedule::CanonicalSingleton;
    }
    let head = crate::weights::qwen4_exp_mtp::load(&root, &manifest, &trunk).unwrap();
    let tokens = [1, 2, 3, 4, 5, 6, 7];
    let model = DiagnosticModel {
        trunk: &trunk,
        head: &head,
        owner: 7300,
        terminal_ids: &[],
    };
    let result = model.collect(&tokens, 33, |_| {});
    assert_eq!(result["prefill"]["repeat_session_exact"], true);
    assert_eq!(result["greedy_identity"], true);
    assert_eq!(result["compared_equal_positions"], 33);
    assert_eq!(
        result["matching_generated_prefix_ids"]
            .as_array()
            .unwrap()
            .len(),
        33
    );
    assert_eq!(result["stopped_at_terminal"], false);
    assert_eq!(result["qualification"], false);
    if canonical {
        assert_eq!(result["prefill"]["logits_exact"], true);
        assert_eq!(result["compared_full_state_identity"], true);
        assert_eq!(result["compared_stream_hidden_identity"], true);
        assert_eq!(result["compared_canonical_logits_identity"], true);
        for step in result["steps"].as_array().unwrap() {
            let primary = &step["canonical_primary_exact"];
            assert_eq!(
                primary["generated_prefix_length"],
                step["position"].as_u64().unwrap() + 1
            );
            assert_eq!(primary["state"], true);
            assert_eq!(primary["stream_hidden"], true);
            assert_eq!(primary["logits"], true);
            assert_eq!(step["actual_next_logits_exact"], true);
        }
    }
    if let Some(path) = std::env::var_os("AX_FLASH_NEXT_SYNTHETIC_DIAGNOSTIC_OUTPUT") {
        let path = PathBuf::from(path);
        assert!(!path.exists());
        let mut request = result.clone();
        request["id"] = json!("synthetic");
        save_record(
            &path,
            &json!({"collection_complete": true, "phase": "complete",
            "qualification": false, "release_ready": false, "requests": [request]}),
        );
    }
    let before = CandidateSession::prefill(&trunk, &head, &tokens, 7300, 7301).unwrap();
    let mut actual = before.clone();
    let (consumed, observation) = actual.step_observed(&trunk, &head, 3, &[]).unwrap();
    let mut direct = model.singleton(consumed[0], &before.trunk_state);
    for &token in &consumed[1..] {
        direct = model.singleton(token, &direct.state);
    }
    let replay = model.replay(
        &before,
        &actual,
        &consumed,
        &observation,
        &before.trunk_state,
        &direct,
        3,
    );
    assert_eq!(replay["same_checkpoint_session_repeat_exact"], true);
    assert_eq!(
        replay["same_checkpoint_mtp_state_singleton_repeat_exact"],
        true
    );
    assert_eq!(
        replay["same_checkpoint_direct_trajectory_repeat_exact"],
        true
    );
    let mut altered = observation.clone();
    altered.draft_token = altered.draft_token.map(|token| (token + 1) % 32);
    assert!(!observation_exact(
        &observation,
        &altered,
        trunk.layers.len()
    ));
    actual.primary = (actual.primary + 1) % 32;
    let replay = model.replay(
        &before,
        &actual,
        &consumed,
        &observation,
        &before.trunk_state,
        &direct,
        3,
    );
    assert_eq!(replay["same_checkpoint_session_repeat_exact"], false);
    let first = result["matching_generated_prefix_ids"][0].as_u64().unwrap() as u32;
    let terminal = [first];
    let stopped = DiagnosticModel {
        terminal_ids: &terminal,
        ..model
    }
    .collect(&tokens, 256, |_| {});
    assert_eq!(stopped["compared_equal_positions"], 1);
    assert_eq!(stopped["stopped_at_terminal"], true);
    assert_eq!(stopped["proposed"], 0);
}

fn save_record(path: &Path, result: &Value) {
    let temporary = path.with_extension("tmp");
    std::fs::write(&temporary, serde_json::to_vec_pretty(result).unwrap()).unwrap();
    std::fs::rename(temporary, path).unwrap();
}

#[test]
fn first_difference_retains_nonfinite_logit_failure() {
    let data = [f32::INFINITY, 1.0, 0.0];
    let logits = MlxArray::from_raw_data(
        data.as_ptr() as *const u8,
        std::mem::size_of_val(&data),
        &[1, 3],
        MlxDtype::Float32,
    );
    let event = difference_value(4, "correction", &logits, 2);
    assert_eq!(event["direct"]["nonfinite_logits"], 1);
    assert!(event["direct"]["top_two_margin"].is_null());
    assert!(event["direct"]["margin_error"].is_string());
}
