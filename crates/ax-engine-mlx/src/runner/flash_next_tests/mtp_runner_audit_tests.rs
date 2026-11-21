use super::*;
use crate::model::qwen4_exp_mtp::runner_audit;

#[test]
#[ignore = "requires the pinned resident Flash Next MXFP4 pack"]
fn flash_next_runner_mtp_uses_same_state_greedy_decisions() {
    let root = std::env::var_os("AX_FLASH_NEXT_REAL_PACK").unwrap();
    let artifacts = NativeModelArtifacts::from_dir(root).unwrap();
    let mut runner = MlxRunner::from_artifacts(&artifacts, 2048, true).unwrap();
    assert!(runner.has_mtp());
    assert!(matches!(
        runner.weights.qwen4_exp.as_ref().unwrap().target_schedule,
        Qwen4ExpTargetSchedule::CanonicalSingleton
    ));
    let mut runs = Vec::new();
    let mut accepted = 0;
    let mut rejected = 0;
    for (index, (length, budget)) in [(16, 64), (512, 64), (16, 2)].into_iter().enumerate() {
        let prompt: Vec<u32> = (1..=length).collect();
        let base = 6100 + index as u64 * 2;
        runner.set_mtp_requested(false);
        let (direct, silent) = runner_audit::capture(|| {
            generate(&runner, &prompt, 2048, context(base, prompt.len(), budget))
        });
        assert!(silent.is_empty());
        assert_eq!(direct.maximum("ax_mlx_flash_next_mtp_verified_steps"), 0);
        runner.set_mtp_requested(true);
        let (mtp, decisions) = runner_audit::capture(|| {
            generate(
                &runner,
                &prompt,
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
        let mut accepted_in_run = 0;
        for decided in &decisions {
            let primary_at = decided.position.checked_sub(prompt.len()).unwrap();
            let start = primary_at + 1;
            let end = start + decided.emitted.len();
            assert!(covered_until <= start && end <= mtp.tokens.len());
            assert_eq!(decided.remaining, budget as usize - start);
            assert_eq!(mtp.tokens[primary_at], decided.primary);
            assert_eq!(&mtp.tokens[start..end], &decided.emitted);
            covered_until = end;
            accepted_in_run += decided.accepted;
            rejected += usize::from(decided.draft.is_some() && decided.accepted == 0);
        }
        assert_eq!(
            accepted_in_run,
            mtp.maximum("ax_mlx_flash_next_mtp_accepted_steps") as usize
        );
        if budget == 2 {
            assert_eq!(accepted_in_run, 0);
        } else {
            assert!(!decisions.is_empty());
        }
        accepted += accepted_in_run;
        runs.push(serde_json::json!({
            "prompt": prompt, "budget": budget,
            "direct_tokens": direct.tokens, "mtp_tokens": mtp.tokens,
            "decisions": decisions,
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
