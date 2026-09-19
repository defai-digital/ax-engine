//! Raw private-cursor transaction evidence; never a qualification grant.

use super::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

const STEPS: usize = 8;
const OWNER: u64 = 3701;

type Continuation = (Qwen4ExpState, Qwen4ExpDraftCursor, u32, Vec<u32>);

struct Writer {
    root: PathBuf,
    written: HashSet<String>,
}

impl Writer {
    fn new(root: &Path) -> Self {
        std::fs::create_dir(root).unwrap();
        Self {
            root: root.into(),
            written: HashSet::new(),
        }
    }

    fn blob(&mut self, bytes: &[u8]) -> Value {
        let digest = format!("{:x}", Sha256::digest(bytes));
        let name = format!("{digest}.bin");
        if self.written.insert(digest.clone()) {
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(self.root.join(&name))
                .unwrap()
                .write_all(bytes)
                .unwrap();
        }
        json!({"file": name, "bytes": bytes.len(), "sha256": digest})
    }

    fn record(&self, name: &str, value: &Value) {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.root.join(name))
            .unwrap()
            .write_all(&serde_json::to_vec_pretty(value).unwrap())
            .unwrap();
    }

    fn array(&mut self, array: &MlxArray) -> Value {
        let bytes: Vec<u8> = bits(array).iter().flat_map(|v| v.to_le_bytes()).collect();
        json!({"shape": array.shape(), "dtype": format!("{:?}", array.dtype()),
            "f32_le": self.blob(&bytes)})
    }

    fn state(&mut self, state: &Qwen4ExpState, layers: usize) -> Value {
        json!({"position": state.position(), "axkb": self.blob(&state_bytes(state, layers))})
    }

    fn cursor(&mut self, cursor: &Qwen4ExpDraftCursor) -> Value {
        json!({"state": self.state(&cursor.draft_state, 1),
            "hidden": self.array(cursor.stream_hidden.as_ref().unwrap()),
            "owner": cursor.owner, "proposed": cursor.proposed, "accepted": cursor.accepted})
    }

    fn checkpoint(
        &mut self,
        state: &Qwen4ExpState,
        cursor: &Qwen4ExpDraftCursor,
        layers: usize,
    ) -> Value {
        json!({"trunk": self.state(state, layers), "cursor": self.cursor(cursor)})
    }

    fn output(&mut self, output: &Qwen4ExpOutput, layers: usize) -> Value {
        json!({"state": self.state(&output.state, layers),
            "logits": self.array(&output.logits), "hidden": self.array(&output.hidden),
            "stream_hidden": self.array(&output.stream_hidden)})
    }
}

fn histories(graph: &Qwen4ExpWeights) -> Value {
    json!(
        graph
            .layers
            .iter()
            .map(|l| l
                .ple
                .as_ref()
                .map(|ple| ple.layout.initial_history().recent().to_vec()))
            .collect::<Vec<_>>()
    )
}

struct Oracle {
    first: Qwen4ExpOutput,
    last: Qwen4ExpOutput,
    cursor: Qwen4ExpDraftCursor,
    proposal_logits: Option<MlxArray>,
    draft: Option<u32>,
    consumed: Vec<u32>,
    next: u32,
    accepted: bool,
}

// Ordinary target and head calls never invoke advance/step/verify_one.
#[allow(clippy::too_many_arguments)]
fn oracle(
    trunk: &Qwen4ExpWeights,
    head: &Qwen4ExpMtpWeights,
    state: &Qwen4ExpState,
    before: &Qwen4ExpDraftCursor,
    primary: u32,
    remaining: usize,
    terminal: &[u32],
) -> Oracle {
    let first = ordinary(trunk, &[primary], state, OWNER);
    let correction = next_token(&first).unwrap();
    let proposed = head_forward(
        head,
        before.stream_hidden.as_ref().unwrap(),
        &[primary],
        &before.draft_state,
        before.owner,
    )
    .unwrap();
    let draft = (remaining > 1).then(|| next_token(&proposed).unwrap());
    let accepted = draft == Some(correction)
        && !terminal.contains(&primary)
        && !terminal.contains(&correction);
    let mut cursor = before.clone();
    cursor.draft_state = proposed.state;
    let proposal_logits = draft.map(|_| proposed.logits);
    let last = if accepted {
        cursor.draft_state = head_forward(
            head,
            &first.stream_hidden,
            &[correction],
            &cursor.draft_state,
            cursor.owner,
        )
        .unwrap()
        .state;
        ordinary(trunk, &[correction], &first.state, OWNER)
    } else {
        Qwen4ExpOutput {
            state: first.state.clone(),
            logits: first.logits.clone(),
            hidden: first.hidden.clone(),
            stream_hidden: first.stream_hidden.clone(),
        }
    };
    cursor.stream_hidden = Some(last.stream_hidden.clone());
    cursor.proposed += usize::from(remaining > 1);
    cursor.accepted += usize::from(accepted);
    let next = next_token(&last).unwrap();
    Oracle {
        first,
        last,
        cursor,
        proposal_logits,
        draft,
        consumed: if accepted {
            vec![primary, correction]
        } else {
            vec![primary]
        },
        next,
        accepted,
    }
}

fn calls() -> Vec<Vec<u32>> {
    TRUNK_FORWARD_TOKENS.with(|calls| calls.borrow().clone())
}

fn verifier_cases(
    writer: &mut Writer,
    trunk: &Qwen4ExpWeights,
    state: &Qwen4ExpState,
    primary: u32,
    prompt: &[u32],
) -> Vec<Value> {
    let correction = next_token(&ordinary(trunk, &[primary], state, OWNER)).unwrap();
    let vocab = u32::try_from(trunk.lm_head.weight.shape()[0]).unwrap();
    assert!(vocab > 1);
    let wrong = (correction + 1) % vocab;
    let mut cases = Vec::new();
    for (label, draft, remaining, terminal) in [
        ("matching-draft", correction, 3, vec![]),
        ("mismatching-draft", wrong, 3, vec![]),
        ("one-slot", correction, 1, vec![]),
        ("zero-budget", correction, 0, vec![]),
        ("primary-terminal", correction, 3, vec![primary]),
        ("correction-terminal", correction, 3, vec![correction]),
    ] {
        let _hooks = TargetHooks::new();
        let layers = trunk.layers.len();
        let before = writer.state(state, layers);
        let reference = (remaining > 0).then(|| {
            let first = ordinary(trunk, &[primary], state, OWNER);
            let accepted = remaining > 1
                && draft == correction
                && !terminal.contains(&primary)
                && !terminal.contains(&correction);
            let second = accepted.then(|| ordinary(trunk, &[draft], &first.state, OWNER));
            json!({"after_primary": writer.output(&first, layers),
                "after_draft": second.as_ref().map(|o| writer.output(o, layers))})
        });
        let after_oracle = writer.state(state, layers);
        TargetHooks::reset();
        let result = verify_one(trunk, state, OWNER, primary, draft, remaining, &terminal);
        let (actual, error) = match result {
            Ok(step) => (
                Some(
                    json!({"accepted": step.accepted, "committed": step.committed,
                "next": step.next_primary, "after_primary": writer.output(&step.after_primary, layers),
                "after_draft": step.after_draft.as_ref().map(|o| writer.output(o, layers))}),
                ),
                None,
            ),
            Err(error) => (None, Some(error)),
        };
        let record = json!({"label": label, "tokens": prompt, "primary": primary,
            "draft": draft, "remaining": remaining, "terminal_ids": terminal,
            "before": before, "after_oracle": after_oracle, "input_after": writer.state(state, layers),
            "reference": reference, "actual": actual, "error": error, "target_calls": calls()});
        writer.record(&format!("verifier-{label}.json"), &record);
        cases.push(record);
    }
    cases
}

fn advanced_record(
    writer: &mut Writer,
    step: &AdvancedStep,
    cursor: &Qwen4ExpDraftCursor,
    layers: usize,
) -> Value {
    json!({"checkpoint": writer.checkpoint(&step.trunk_state, cursor, layers),
        "consumed": step.consumed, "next": step.next_primary,
        "accepted": step.accepted, "draft": step.observation.draft_token,
        "correction_logits": writer.array(step.observation.canonical_correction_logits.as_ref().unwrap()),
        "next_logits": writer.array(&step.observation.next_logits),
        "primary_state": writer.state(&step.observation.canonical_primary.as_ref().unwrap().state, layers),
        "primary_hidden": writer.array(&step.observation.canonical_primary.as_ref().unwrap().stream_hidden),
        "target_calls": calls()})
}

#[derive(Clone, Copy, PartialEq)]
enum Failure {
    None,
    First,
    Second,
    Catchup,
}

impl Failure {
    fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::First => "target_1",
            Self::Second => "target_2",
            Self::Catchup => "accepted_catchup",
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    writer: &mut Writer,
    trunk: &Qwen4ExpWeights,
    head: &Qwen4ExpMtpWeights,
    state: &Qwen4ExpState,
    initial: &Qwen4ExpDraftCursor,
    tokens: &[u32],
    primary: u32,
    remaining: usize,
    terminal: &[u32],
    failure: Failure,
    label: &str,
) -> (Value, Option<Continuation>) {
    let _hooks = TargetHooks::new();
    let layers = trunk.layers.len();
    let before = writer.checkpoint(state, initial, layers);
    let expected =
        (remaining > 0).then(|| oracle(trunk, head, state, initial, primary, remaining, terminal));
    let reference = expected.as_ref().map(|o| json!({
        "checkpoint": writer.checkpoint(&o.last.state, &o.cursor, layers),
        "consumed": o.consumed, "next": o.next, "accepted": o.accepted, "draft": o.draft,
        "proposal_logits": o.proposal_logits.as_ref().map(|l| writer.array(l)),
        "correction_logits": writer.array(&o.first.logits), "next_logits": writer.array(&o.last.logits),
        "primary_state": writer.state(&o.first.state, layers),
        "primary_hidden": writer.array(&o.first.stream_hidden),
    }));
    let after_oracle = writer.checkpoint(state, initial, layers);
    assert!(
        matches!(failure, Failure::None | Failure::First)
            || expected.as_ref().is_some_and(|o| o.accepted)
    );
    TargetHooks::reset();
    FAIL_TARGET_CALL.with(|f| {
        f.set(match failure {
            Failure::First => Some(1),
            Failure::Second => Some(2),
            _ => None,
        })
    });
    FAIL_ACCEPTED_CATCHUP.with(|f| f.set(failure == Failure::Catchup));
    let mut cursor = initial.clone();
    let result = cursor.advance(trunk, head, state, OWNER, primary, remaining, terminal);
    let attempted_calls = calls();
    let input_after = writer.state(state, layers);
    let (actual, error, continuation) = match result {
        Ok(step) => {
            let record = advanced_record(writer, &step, &cursor, layers);
            let continuation = Some((
                step.trunk_state,
                cursor.clone(),
                step.next_primary,
                step.consumed,
            ));
            (Some(record), None, continuation)
        }
        Err(e) => (None, Some(e), None),
    };
    let after = writer.cursor(&cursor);
    TargetHooks::reset();
    let (retry, retry_error) = if failure != Failure::None && error.is_some() {
        match cursor.advance(trunk, head, state, OWNER, primary, remaining, terminal) {
            Ok(step) => (Some(advanced_record(writer, &step, &cursor, layers)), None),
            Err(error) => (None, Some(error)),
        }
    } else {
        (None, None)
    };
    let record = json!({"label": label, "tokens": tokens, "primary": primary,
        "remaining": remaining, "terminal_ids": terminal, "failure": failure.name(),
        "before": before, "after_oracle": after_oracle, "reference": reference,
        "actual": actual, "error": error, "attempted_calls": attempted_calls,
        "input_after": input_after, "cursor_after": after, "retry": retry, "retry_error": retry_error,
        "initial_after": writer.checkpoint(state, initial, layers)});
    writer.record(&format!("cursor-{label}.json"), &record);
    (record, continuation)
}

fn collect(
    trunk: &Qwen4ExpWeights,
    head: &Qwen4ExpMtpWeights,
    prompt: &[u32],
    root: &Path,
    arm: &str,
) {
    assert!((3..=512).contains(&prompt.len()));
    assert!(matches!(
        trunk.target_schedule,
        Qwen4ExpTargetSchedule::CanonicalSingleton
    ));
    let _scope = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let mut writer = Writer::new(root);
    let mut cursor = Qwen4ExpDraftCursor::new(head, OWNER);
    let prefix = ordinary(
        trunk,
        &prompt[..prompt.len() - 1],
        &Qwen4ExpState::new(trunk, OWNER),
        OWNER,
    );
    cursor
        .absorb(head, &prompt[..prompt.len() - 1], &prefix.stream_hidden)
        .unwrap();
    let last = ordinary(trunk, &prompt[prompt.len() - 1..], &prefix.state, OWNER);
    cursor
        .absorb(head, &prompt[prompt.len() - 1..], &last.stream_hidden)
        .unwrap();
    let initial_primary_logits = writer.array(&last.logits);
    let initial_hidden = writer.array(&last.stream_hidden);
    let mut primary = next_token(&last).unwrap();
    let mut state = last.state;
    let mut tokens = prompt.to_vec();
    let initial = (state.clone(), cursor.clone(), primary);
    let mut accepted_checkpoint = None;
    let mut cases = Vec::new();
    for index in 0..STEPS {
        let (record, continuation) = run_case(
            &mut writer,
            trunk,
            head,
            &state,
            &cursor,
            &tokens,
            primary,
            3,
            &[],
            Failure::None,
            &format!("trajectory-{index}"),
        );
        if record["actual"]["accepted"] == true && accepted_checkpoint.is_none() {
            accepted_checkpoint = Some((state.clone(), cursor.clone(), primary, tokens.clone()));
        }
        cases.push(record);
        let Some((next_state, next_cursor, next, consumed)) = continuation else {
            break;
        };
        state = next_state;
        cursor = next_cursor;
        primary = next;
        tokens.extend(consumed);
    }
    let (state, cursor, primary) = initial;
    let correction = next_token(&ordinary(trunk, &[primary], &state, OWNER)).unwrap();
    for (label, budget, terminal) in [
        ("zero-budget", 0, vec![]),
        ("one-slot", 1, vec![]),
        ("primary-terminal", 3, vec![primary]),
        ("correction-terminal", 3, vec![correction]),
    ] {
        cases.push(
            run_case(
                &mut writer,
                trunk,
                head,
                &state,
                &cursor,
                prompt,
                primary,
                budget,
                &terminal,
                Failure::None,
                label,
            )
            .0,
        );
    }
    cases.push(
        run_case(
            &mut writer,
            trunk,
            head,
            &state,
            &cursor,
            prompt,
            primary,
            3,
            &[],
            Failure::First,
            "failure-target-1",
        )
        .0,
    );
    if let Some((state, cursor, primary, tokens)) = &accepted_checkpoint {
        for (failure, label) in [
            (Failure::Second, "failure-target-2"),
            (Failure::Catchup, "failure-catchup"),
        ] {
            cases.push(
                run_case(
                    &mut writer,
                    trunk,
                    head,
                    state,
                    cursor,
                    tokens,
                    *primary,
                    3,
                    &[],
                    failure,
                    label,
                )
                .0,
            );
        }
    }
    let verifier = verifier_cases(&mut writer, trunk, &state, primary, prompt);
    let report = json!({"schema": "ax.flash_next.mtp_transactions.v2",
        "qualification": false, "release_ready": false,
        "mtp_certification": {"MTP-S":"not_assessed", "MTP-P":"not_assessed", "MTP-D":"not_assessed"},
        "target_schedule":"canonical_singleton", "arm": arm, "prompt_ids": prompt,
        "head_permutation_seed": (arm == "permuted").then_some(trained_head::PERMUTE_HEAD_SEED),
        "trunk_owner": OWNER, "head_owner": cursor.owner, "trajectory_steps": STEPS,
        "initial_primary_logits": initial_primary_logits, "initial_hidden": initial_hidden,
        "trunk_initial_histories": histories(trunk), "head_initial_histories": histories(&head.graph),
        "accepted_failure_coverage": accepted_checkpoint.is_some(), "cases": cases,
        "verifier_cases": verifier,
        "scope":"Private cursor transactions, forced verifier controls and ordinary singleton replay; not full MTP-S qualification"});
    writer.record("result.json", &report);
}

#[test]
#[ignore = "requires bounded synthetic F32/BF16 artifacts and exclusive transaction output"]
fn flash_next_synthetic_private_cursor_evidence() {
    let (mut trunk, mut head) = fixture();
    let arm = std::env::var("AX_FLASH_NEXT_TRANSACTION_ARM").unwrap();
    assert!(matches!(
        arm.as_str(),
        "synthetic_accept" | "synthetic_reject"
    ));
    set_constant_head(&mut trunk.lm_head, &[3]);
    set_constant_head(
        &mut head.graph.lm_head,
        &[if arm == "synthetic_accept" { 3 } else { 17 }],
    );
    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_TRANSACTION_RESULT_DIR").unwrap());
    collect(&trunk, &head, &[1, 2, 3, 4, 5, 6, 7, 8, 9], &root, &arm);
}

#[test]
#[ignore = "requires real Flash Next MXFP4 pack and exclusive transaction output"]
fn flash_next_real_private_cursor_evidence() {
    let arm = std::env::var("AX_FLASH_NEXT_TRANSACTION_ARM").unwrap();
    assert!(
        matches!(arm.as_str(), "trained" | "permuted"),
        "unsupported transaction arm"
    );
    let prompt: Vec<u32> =
        serde_json::from_str(&std::env::var("AX_FLASH_NEXT_PROMPT_IDS").unwrap()).unwrap();
    assert!((3..=512).contains(&prompt.len()));
    let output = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_TRANSACTION_RESULT_DIR").unwrap());
    assert!(
        matches!(std::fs::symlink_metadata(&output), Err(error)
        if error.kind() == std::io::ErrorKind::NotFound),
        "transaction output already exists or is inaccessible"
    );
    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_REAL_PACK").unwrap());
    let artifacts = ax_engine_core::NativeModelArtifacts::from_dir(&root).unwrap();
    let trunk = crate::weights::qwen4_exp::load_with_paging_policy(
        &root,
        artifacts.manifest(),
        crate::expert_stream::StreamExpertsMode::Auto,
        1,
    )
    .unwrap();
    let mut head =
        crate::weights::qwen4_exp_mtp::load(&root, artifacts.manifest(), &trunk).unwrap();
    if arm == "permuted" {
        trained_head::permute_draft_output_projection(&mut head, trained_head::PERMUTE_HEAD_SEED);
    }
    collect(&trunk, &head, &prompt, &output, &arm);
}

#[test]
fn transaction_entry_rejects_invalid_arm_before_model_access() {
    let missing = std::env::temp_dir().join(format!(
        "ax-missing-transaction-pack-{}",
        std::process::id()
    ));
    assert!(!missing.exists());
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["model::qwen4_exp_mtp::canonical_tests::transaction_evidence::flash_next_real_private_cursor_evidence",
               "--exact", "--ignored", "--nocapture", "--test-threads=1"])
        .env("AX_FLASH_NEXT_TRANSACTION_ARM", "unsupported")
        .env("AX_FLASH_NEXT_REAL_PACK", &missing)
        .env("AX_FLASH_NEXT_PROMPT_IDS", "[1,2,3]")
        .env("AX_FLASH_NEXT_TRANSACTION_RESULT_DIR", missing.join("output"))
        .output().unwrap();
    assert!(!child.status.success());
    assert!(String::from_utf8_lossy(&child.stderr).contains("unsupported transaction arm"));
}
