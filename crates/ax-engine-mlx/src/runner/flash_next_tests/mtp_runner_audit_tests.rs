use super::*;
use crate::model::qwen4_exp_mtp::runner_audit;
use std::collections::BTreeMap;

fn pinned_canonical_runner() -> MlxRunner {
    let root = std::env::var_os("AX_FLASH_NEXT_REAL_PACK").unwrap();
    let artifacts = NativeModelArtifacts::from_dir(root).unwrap();
    let runner = MlxRunner::from_artifacts(&artifacts, 2048, true).unwrap();
    assert!(runner.has_mtp());
    assert!(matches!(
        runner.weights.qwen4_exp.as_ref().unwrap().target_schedule,
        Qwen4ExpTargetSchedule::CanonicalSingleton
    ));
    runner
}

#[test]
#[ignore = "requires the pinned resident Flash Next MXFP4 pack"]
fn flash_next_runner_mtp_uses_same_state_greedy_decisions() {
    let mut runner = pinned_canonical_runner();
    let mut runs = Vec::new();
    let mut accepted = 0;
    let mut rejected = 0;
    for (index, (length, budget)) in [(16, 64), (512, 64), (16, 2)].into_iter().enumerate() {
        let prompt: Vec<u32> = (1..=length).collect();
        let base = 6100 + index as u64 * 2;
        let audit = audit_case(
            &mut runner,
            &prompt,
            context(base, prompt.len(), budget),
            AuditPolicy::Diagnostic,
        );
        let audited = audit.decisions.len();
        let accepted_in_run = audit.accepted;
        rejected += audit.rejected;
        if budget == 2 {
            assert_eq!(accepted_in_run, 0);
        } else {
            assert!(audited > 0);
        }
        accepted += accepted_in_run;
        runs.push(serde_json::json!({
            "prompt": prompt, "budget": budget,
            "direct_tokens": audit.direct, "mtp_tokens": audit.mtp,
            "decisions": audit.decisions,
        }));
    }
    assert!(accepted > 0 && rejected > 0);
    println!(
        "{}",
        serde_json::json!({
            "schema": "ax-engine.flash-next.runner-mtp-greedy-audit.v1",
            "qualification": false, "accepted": accepted, "rejected": rejected, "runs": runs,
        })
    );
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AuditPolicy {
    Diagnostic,
    Required,
}

struct CaseAudit {
    direct: Vec<u32>,
    mtp: Vec<u32>,
    stop_reason: Option<StopReason>,
    decisions: Vec<runner_audit::Decision>,
    accepted: usize,
    rejected: usize,
    output_decisions: usize,
    direct_stop_reason: Option<StopReason>,
    activation: BTreeMap<String, u32>,
}

fn audit_case(
    runner: &mut MlxRunner,
    prompt: &[u32],
    direct_ctx: RunnerRequestContext,
    policy: AuditPolicy,
) -> CaseAudit {
    let budget = direct_ctx.max_output_tokens as usize;
    let mtp_ctx = RunnerRequestContext {
        request_id: RequestId(direct_ctx.request_id.0 + 1),
        ..direct_ctx
    };
    if policy == AuditPolicy::Required {
        runner.set_mtp_session_policy(MtpSessionPolicy::Disabled);
    }
    runner.set_mtp_requested(false);
    let (direct, silent) = runner_audit::capture(|| generate(runner, prompt, 2048, direct_ctx));
    assert!(silent.is_empty());
    assert_eq!(direct.maximum("ax_mlx_flash_next_mtp_verified_steps"), 0);
    if !direct_ctx.ignore_eos {
        assert!(!runner.states.lock().contains_key(&direct_ctx.request_id));
    }
    if policy == AuditPolicy::Required {
        runner.set_mtp_session_policy(MtpSessionPolicy::Required);
    }
    runner.set_mtp_requested(true);
    let (mtp, decisions) = runner_audit::capture(|| generate(runner, prompt, 2048, mtp_ctx));
    if direct_ctx.ignore_eos {
        assert_eq!(mtp.tokens.len(), budget);
        assert_eq!(direct.tokens.len(), budget);
    } else {
        for run in [&direct, &mtp] {
            let (last, preceding) = run.tokens.split_last().unwrap();
            assert!(
                preceding
                    .iter()
                    .all(|id| !runner.terminal_token_ids.contains(id))
            );
            match (run.stop_reason, policy) {
                (Some(StopReason::EosToken), _) => {
                    assert!(runner.terminal_token_ids.contains(last));
                    assert!(run.tokens.len() <= budget);
                    if policy == AuditPolicy::Diagnostic {
                        assert!(run.tokens.len() < budget);
                    }
                }
                (Some(StopReason::MaxOutputTokens), AuditPolicy::Required) => {
                    assert_eq!(run.tokens.len(), budget);
                }
                (reason, _) => panic!("unexpected stop reason: {reason:?}"),
            }
        }
        assert!(!runner.states.lock().contains_key(&mtp_ctx.request_id));
    }
    assert_eq!(mtp.maximum("ax_mlx_flash_next_mtp_step_errors"), 0);
    assert_eq!(
        decisions.len(),
        mtp.maximum("ax_mlx_flash_next_mtp_verified_steps") as usize
    );
    let mut covered_until = 1;
    let mut output_decisions = 0;
    let mut accepted_in_run = 0;
    let mut rejected_in_run = 0;
    for decided in &decisions {
        let primary_at = decided.position.checked_sub(prompt.len()).unwrap();
        let start = primary_at + 1;
        let end = start + decided.emitted.len();
        assert!(covered_until <= start && end <= mtp.tokens.len());
        assert_eq!(decided.remaining, budget - start);
        assert_eq!(mtp.tokens[primary_at], decided.primary);
        assert_eq!(&mtp.tokens[start..end], &decided.emitted);
        covered_until = end;
        output_decisions += decided.emitted.len();
        accepted_in_run += decided.accepted;
        rejected_in_run += usize::from(decided.draft.is_some() && decided.accepted == 0);
    }
    assert_eq!(
        accepted_in_run,
        mtp.maximum("ax_mlx_flash_next_mtp_accepted_steps") as usize
    );
    let mut activation = BTreeMap::<String, u32>::new();
    if policy == AuditPolicy::Required {
        for (key, value) in mtp.routes.iter().flatten() {
            if key.starts_with("ax_mlx_flash_next_mtp_") {
                let recorded = activation.entry(key.clone()).or_default();
                *recorded = (*recorded).max(*value);
            }
        }
    }
    CaseAudit {
        direct: direct.tokens,
        mtp: mtp.tokens,
        stop_reason: mtp.stop_reason,
        decisions,
        accepted: accepted_in_run,
        rejected: rejected_in_run,
        output_decisions,
        direct_stop_reason: direct.stop_reason,
        activation,
    }
}

#[derive(serde::Deserialize)]
struct AuditWorkload {
    name: String,
    prompt: Vec<u32>,
    budget: u32,
}

#[test]
#[ignore = "requires pinned Flash Next MXFP4 and an explicit workload inventory"]
fn flash_next_runner_mtp_audits_long_workloads() {
    audit_workloads(
        "AX_FLASH_NEXT_MTP_AUDIT_INPUTS",
        "AX_FLASH_NEXT_MTP_AUDIT_OUTPUT",
        true,
    );
}

#[test]
#[ignore = "requires pinned Flash Next MXFP4 and an explicit natural-stop workload inventory"]
fn flash_next_runner_mtp_audits_natural_stops() {
    audit_workloads(
        "AX_FLASH_NEXT_MTP_NATURAL_STOP_INPUTS",
        "AX_FLASH_NEXT_MTP_NATURAL_STOP_OUTPUT",
        false,
    );
}

fn audit_workloads(inputs_env: &str, output_env: &str, ignore_eos: bool) {
    let inputs = std::env::var_os(inputs_env).unwrap();
    let cases: Vec<AuditWorkload> =
        serde_json::from_slice(&std::fs::read(inputs).unwrap()).unwrap();
    assert!(!cases.is_empty());
    let mut runner = pinned_canonical_runner();
    let vocab = runner.cfg.vocab_size;
    let mut names = std::collections::BTreeSet::new();
    let mut observations = Vec::new();
    for (index, case) in cases.iter().enumerate() {
        assert!(!case.name.is_empty() && names.insert(&case.name));
        assert!(case.budget > 1 && !case.prompt.is_empty());
        assert!(case.prompt.len() + case.budget as usize <= 32768);
        assert!(case.prompt.iter().all(|&token| (token as usize) < vocab));
        let request = RunnerRequestContext {
            ignore_eos,
            ..context(8200 + index as u64 * 2, case.prompt.len(), case.budget)
        };
        let audit = audit_case(&mut runner, &case.prompt, request, AuditPolicy::Diagnostic);
        assert!(
            audit.output_decisions > 0,
            "{} did not exercise MTP",
            case.name
        );
        let changed: Vec<_> = (0..audit.direct.len().max(audit.mtp.len()))
            .filter(|&index| audit.direct.get(index) != audit.mtp.get(index))
            .collect();
        let mut observation = serde_json::json!({
            "name": case.name, "input_tokens": case.prompt.len(), "budget": case.budget,
            "direct_tokens": audit.direct, "mtp_tokens": audit.mtp,
            "verified_steps": audit.decisions.len(), "accepted": audit.accepted,
            "rejected": audit.rejected, "output_positions_audited": audit.output_decisions,
            "changed_from_direct": changed, "decisions": audit.decisions,
        });
        if !ignore_eos {
            assert_eq!(audit.output_decisions, audit.mtp.len() - 1, "{}", case.name);
            assert!(audit.accepted > 0, "{} did not accept a draft", case.name);
            observation["stop_reason"] = serde_json::json!(audit.stop_reason);
        }
        observations.push(observation);
    }
    let schema = if ignore_eos {
        "ax-engine.flash-next.runner-mtp-greedy-long-audit.v1"
    } else {
        "ax-engine.flash-next.runner-mtp-greedy-natural-stop-audit.v1"
    };
    let report = serde_json::json!({
        "schema": schema,
        "qualification": false, "prefill_quantum": 2048, "ignore_eos": ignore_eos,
        "cases": observations,
    });
    let output = PathBuf::from(std::env::var_os(output_env).unwrap());
    use std::io::Write;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap()
        .write_all(&serde_json::to_vec_pretty(&report).unwrap())
        .unwrap();
}

#[test]
#[ignore = "requires the pinned resident Flash Next MXFP4 pack and the frozen QA cohort"]
fn flash_next_runner_mtp_audits_required_qa_cohort() {
    use std::io::{BufWriter, Write};

    let inputs = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_MTP_QA_INPUTS").unwrap());
    let cases: Vec<AuditWorkload> =
        serde_json::from_slice(&std::fs::read(inputs).unwrap()).unwrap();
    assert_eq!(cases.len(), 105);
    let mut runner = pinned_canonical_runner();
    let output = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_MTP_QA_OUTPUT").unwrap());
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    let mut output = BufWriter::new(file);
    let mut accepted = 0;
    let mut checked = 0;
    let mut names = std::collections::BTreeSet::new();
    for (index, case) in cases.iter().enumerate() {
        assert!(!case.name.is_empty() && names.insert(&case.name));
        assert!(case.budget > 0 && !case.prompt.is_empty());
        assert!(case.prompt.len() + case.budget as usize <= 32768);
        assert!(
            case.prompt
                .iter()
                .all(|&token| (token as usize) < runner.cfg.vocab_size)
        );
        let request = RunnerRequestContext {
            ignore_eos: false,
            ..context(9200 + index as u64 * 2, case.prompt.len(), case.budget)
        };
        let audit = audit_case(&mut runner, &case.prompt, request, AuditPolicy::Required);
        accepted += audit.accepted;
        checked += audit.decisions.len();
        let changed: Vec<_> = (0..audit.direct.len().max(audit.mtp.len()))
            .filter(|&index| audit.direct.get(index) != audit.mtp.get(index))
            .collect();
        let row = serde_json::json!({
            "kind": "case", "name": case.name, "input_tokens": case.prompt.len(), "budget": case.budget,
            "direct_tokens": audit.direct, "mtp_tokens": audit.mtp,
            "direct_stop_reason": audit.direct_stop_reason, "mtp_stop_reason": audit.stop_reason,
            "verified_steps": audit.decisions.len(), "accepted": audit.accepted, "rejected": audit.rejected,
            "output_positions_audited": audit.output_decisions, "changed_from_direct": changed,
            "activation": audit.activation, "decisions": audit.decisions,
        });
        serde_json::to_writer(&mut output, &row).unwrap();
        writeln!(output).unwrap();
        output.flush().unwrap();
    }
    assert!(accepted > 0 && checked >= accepted);
    writeln!(
        output,
        "{}",
        serde_json::json!({
            "kind": "complete", "schema": "ax-engine.flash-next.runner-required-qa-audit.v1",
            "qualification": false, "cases": cases.len(), "accepted_drafts": accepted,
            "verified_steps": checked, "prefill_quantum": 2048,
        })
    )
    .unwrap();
    output.flush().unwrap();
}
