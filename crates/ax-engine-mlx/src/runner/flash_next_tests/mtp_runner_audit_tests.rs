use super::*;
use crate::model::qwen4_exp_mtp::runner_audit;

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
        let audit = audit_case(&mut runner, &prompt, budget, base);
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

struct CaseAudit {
    direct: Vec<u32>,
    mtp: Vec<u32>,
    decisions: Vec<runner_audit::Decision>,
    accepted: usize,
    rejected: usize,
    output_decisions: usize,
}

fn audit_case(runner: &mut MlxRunner, prompt: &[u32], budget: u32, base: u64) -> CaseAudit {
    runner.set_mtp_requested(false);
    let (direct, silent) = runner_audit::capture(|| {
        generate(runner, prompt, 2048, context(base, prompt.len(), budget))
    });
    assert!(silent.is_empty());
    assert_eq!(direct.maximum("ax_mlx_flash_next_mtp_verified_steps"), 0);
    runner.set_mtp_requested(true);
    let (mtp, decisions) = runner_audit::capture(|| {
        generate(
            runner,
            prompt,
            2048,
            context(base + 1, prompt.len(), budget),
        )
    });
    assert_eq!(mtp.tokens.len(), budget as usize);
    assert_eq!(direct.tokens.len(), budget as usize);
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
        assert_eq!(decided.remaining, budget as usize - start);
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
    CaseAudit {
        direct: direct.tokens,
        mtp: mtp.tokens,
        decisions,
        accepted: accepted_in_run,
        rejected: rejected_in_run,
        output_decisions,
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
    let inputs = std::env::var_os("AX_FLASH_NEXT_MTP_AUDIT_INPUTS").unwrap();
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
        let audit = audit_case(
            &mut runner,
            &case.prompt,
            case.budget,
            8200 + index as u64 * 2,
        );
        assert!(
            audit.output_decisions > 0,
            "{} did not exercise MTP",
            case.name
        );
        let changed: Vec<_> = audit
            .direct
            .iter()
            .zip(&audit.mtp)
            .enumerate()
            .filter_map(|(index, (direct, mtp))| (direct != mtp).then_some(index))
            .collect();
        observations.push(serde_json::json!({
            "name": case.name, "input_tokens": case.prompt.len(), "budget": case.budget,
            "direct_tokens": audit.direct, "mtp_tokens": audit.mtp,
            "verified_steps": audit.decisions.len(), "accepted": audit.accepted,
            "rejected": audit.rejected, "output_positions_audited": audit.output_decisions,
            "changed_from_direct": changed, "decisions": audit.decisions,
        }));
    }
    let report = serde_json::json!({
        "schema": "ax-engine.flash-next.runner-mtp-greedy-long-audit.v1",
        "qualification": false, "prefill_quantum": 2048, "ignore_eos": true,
        "cases": observations,
    });
    let output = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_MTP_AUDIT_OUTPUT").unwrap());
    use std::io::Write;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap()
        .write_all(&serde_json::to_vec_pretty(&report).unwrap())
        .unwrap();
}
