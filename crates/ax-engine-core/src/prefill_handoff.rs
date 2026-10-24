//! Adaptive prefill chunks and one-shot prefix handoff.
//!
//! All three behaviours are off unless the matching environment variable is
//! exactly `1` at engine construction (or a test setter enables them):
//! `AX_ENGINE_ADAPTIVE_PREFILL`, `AX_MLX_PRESSURE_REQUEUE`,
//! `AX_MLX_CANCEL_KEEP_PREFIX`.
//!
//! Pressure control reads the MLX memory limit only. It does not claim to
//! cover mmap or the rest of host memory. Device KV capture from the prefix
//! store work is not in this tree, so a token-prefix blob is not a restorable
//! KV layout: callers treat that as an unsupported layout and recompute
//! instead of reporting an SSD hit. A chunk-history mismatch is a miss.

use crate::ids::RequestId;

/// Route decision carrying the text-prefill chunk the scheduler actually
/// applied. The MLX runner uses it so its internal chunk size does not
/// shrink or enlarge that choice. Absent when adaptive prefill is off.
pub const ADAPTIVE_PREFILL_CHUNK_ROUTE_KEY: &str = "ax_scheduler_adaptive_prefill_chunk_tokens";

/// Largest text prefill chunk the resident-budget contract allows.
pub const MODEL_PREFILL_SAFETY_CAP: u32 = 2048;

/// At most two pressure spills per request. The 64-step memory-block
/// failure bound stays in the request manager.
pub const MAX_PRESSURE_SPILLS: u8 = 2;

const EXECUTION_SIGNATURE_OFFSET: u64 = 0xcbf29ce484222325;
const EXECUTION_SIGNATURE_PRIME: u64 = 0x100000001b3;

/// Chunk sizes chosen from the committed cursor. `explicit` overrides the
/// idle and decoding defaults when set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillChunkPolicy {
    pub idle: u32,
    pub decoding: u32,
    pub explicit: Option<u32>,
}

impl Default for PrefillChunkPolicy {
    fn default() -> Self {
        Self {
            idle: 2048,
            decoding: 512,
            explicit: None,
        }
    }
}

impl PrefillChunkPolicy {
    /// Policy quantum before fair share, pressure, and budget are applied.
    /// A prefill request is not a decode competitor; callers pass
    /// `decode_executable` only when some other candidate is actually a decode.
    pub fn quantum(self, decode_executable: bool) -> u32 {
        if let Some(explicit) = self.explicit {
            return explicit;
        }
        if decode_executable {
            self.decoding
        } else {
            self.idle
        }
    }
}

/// Inputs to the text-prefill minimum. Multimodal items do not use this.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptivePrefillCapInputs {
    pub policy: PrefillChunkPolicy,
    pub decode_executable: bool,
    /// `max_prefill_tokens_per_request_per_step`. Zero means no extra cap.
    pub explicit_request_cap: u32,
    pub fair_share: Option<u32>,
    pub pressure_cap: Option<u32>,
    pub global_budget: u32,
    pub model_safety_cap: u32,
}

/// Minimum of the policy quantum, the explicit cap, fair share, pressure,
/// the global budget, and the model safety cap.
pub fn adaptive_text_prefill_cap(inputs: AdaptivePrefillCapInputs) -> u32 {
    let mut cap = inputs.policy.quantum(inputs.decode_executable);
    if inputs.explicit_request_cap > 0 {
        cap = cap.min(inputs.explicit_request_cap);
    }
    if let Some(fair_share) = inputs.fair_share {
        cap = cap.min(fair_share);
    }
    if let Some(pressure_cap) = inputs.pressure_cap {
        cap = cap.min(pressure_cap);
    }
    cap = cap.min(inputs.global_budget);
    if inputs.model_safety_cap > 0 {
        cap = cap.min(inputs.model_safety_cap);
    }
    cap
}

/// Stable identity of a committed token prefix and the chunk sizes that
/// produced it. A different chunk history does not match.
pub fn execution_signature(tokens: &[u32], chunk_history: &[u32]) -> u64 {
    let mut hash = EXECUTION_SIGNATURE_OFFSET;
    for token in tokens.iter().chain(chunk_history.iter()) {
        hash ^= u64::from(*token);
        hash = hash.wrapping_mul(EXECUTION_SIGNATURE_PRIME);
    }
    hash ^= u64::from(tokens.len() as u32);
    hash = hash.wrapping_mul(EXECUTION_SIGNATURE_PRIME);
    hash ^ u64::from(chunk_history.len() as u32)
}

/// Prompt prefix published only after the engine accepts an execution update.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommittedPrefix {
    pub tokens: Vec<u32>,
    pub cut: u32,
    pub execution_signature: u64,
    pub chunk_history: Vec<u32>,
    pub snapshot: Vec<u8>,
    /// True only when the runner stored a device KV capture. A token-prefix
    /// blob is not enough to adopt KV on resume.
    pub device_capture: bool,
}

impl CommittedPrefix {
    pub fn from_committed_tokens(
        tokens: Vec<u32>,
        chunk_history: Vec<u32>,
        device_capture: bool,
    ) -> Self {
        let cut = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        let execution_signature = execution_signature(&tokens, &chunk_history);
        let mut prefix = Self {
            tokens,
            cut,
            execution_signature,
            chunk_history,
            snapshot: Vec::new(),
            device_capture,
        };
        prefix.snapshot = encode_prefix_snapshot(&prefix);
        prefix
    }
}

/// `true` when `chunk_history` cannot adopt `stored`. Used when a later
/// capture's chunk history disagrees with the prefix that was committed.
pub fn chunk_history_is_miss(stored: &CommittedPrefix, chunk_history: &[u32]) -> bool {
    stored.chunk_history.as_slice() != chunk_history
        || execution_signature(&stored.tokens, &stored.chunk_history) != stored.execution_signature
}

fn encode_prefix_snapshot(prefix: &CommittedPrefix) -> Vec<u8> {
    let mut bytes =
        Vec::with_capacity(24 + prefix.tokens.len() * 4 + prefix.chunk_history.len() * 4);
    bytes.extend_from_slice(b"AXPFX1");
    bytes.extend_from_slice(&prefix.cut.to_le_bytes());
    bytes.extend_from_slice(&prefix.execution_signature.to_le_bytes());
    bytes.push(u8::from(prefix.device_capture));
    push_u32s(&mut bytes, &prefix.chunk_history);
    push_u32s(&mut bytes, &prefix.tokens);
    bytes
}

fn push_u32s(bytes: &mut Vec<u8>, values: &[u32]) {
    let len = u32::try_from(values.len()).unwrap_or(u32::MAX);
    bytes.extend_from_slice(&len.to_le_bytes());
    for value in values.iter().take(len as usize) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

/// Why the engine asked the runner to checkpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckpointReason {
    Preempt,
    Cancel,
    PressureSpill,
}

/// Result of one checkpoint attempt. `ssd_hit` is set only after a durable
/// write of a supported layout. Unsupported layouts stay a miss.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckpointReceipt {
    pub request_id: RequestId,
    pub cut: u32,
    pub committed: bool,
    pub ssd_hit: bool,
    pub unsupported_layout: bool,
    pub device_capture: bool,
}

impl CheckpointReceipt {
    pub fn miss(request_id: RequestId, cut: u32) -> Self {
        Self {
            request_id,
            cut,
            committed: false,
            ssd_hit: false,
            unsupported_layout: true,
            device_capture: false,
        }
    }

    pub fn adopts_cursor(self, committed_limit: u32) -> bool {
        self.committed
            && self.ssd_hit
            && self.device_capture
            && !self.unsupported_layout
            && self.cut <= committed_limit
    }
}

/// MLX active-memory sample. `limit_bytes` is the effective MLX limit `L`.
/// Zero means `L` is unknown and this control stays off.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryPressureSample {
    pub active_bytes: u64,
    pub limit_bytes: u64,
    pub idle_cache_bytes: u64,
    pub next_chunk_workspace_bytes: u64,
}

impl MemoryPressureSample {
    pub fn hard_bytes(self) -> Option<u64> {
        (self.limit_bytes > 0).then_some(self.limit_bytes.saturating_mul(90) / 100)
    }

    pub fn resume_bytes(self) -> Option<u64> {
        (self.limit_bytes > 0).then_some(self.limit_bytes.saturating_mul(80) / 100)
    }

    /// Active bytes after the idle cache is reclaimed, plus the next chunk.
    pub fn projected_after_reclaim(self) -> u64 {
        self.active_bytes
            .saturating_sub(self.idle_cache_bytes)
            .saturating_add(self.next_chunk_workspace_bytes)
    }

    pub fn over_hard(self) -> bool {
        self.hard_bytes()
            .is_some_and(|hard| self.projected_after_reclaim() >= hard)
    }

    /// Resume only while the next chunk still fits under the resume line.
    pub fn ready_to_resume(self) -> bool {
        self.resume_bytes()
            .is_some_and(|resume| self.projected_after_reclaim() < resume)
    }
}

/// Text, MTP-off, prefill that has a committed cursor and no output token.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpillVictimProfile {
    pub text: bool,
    pub mtp_active: bool,
    pub generated_tokens: u32,
    pub processed_prompt_tokens: u32,
    pub prompt_len: u32,
    pub runnable: bool,
    pub spill_attempts: u8,
}

pub fn spill_victim_eligible(profile: SpillVictimProfile) -> bool {
    profile.text
        && !profile.mtp_active
        && profile.generated_tokens == 0
        && profile.runnable
        && profile.spill_attempts < MAX_PRESSURE_SPILLS
        && profile.processed_prompt_tokens > 0
        && profile.processed_prompt_tokens < profile.prompt_len
}

/// `1` enables the switch. Unset, `0`, and every other value leave it off.
pub(crate) fn env_switch_enabled(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|value| value == "1")
}

#[derive(Debug)]
pub(crate) struct BackgroundPrefixWriter {
    tx: Option<std::sync::mpsc::SyncSender<Vec<u8>>>,
}

impl BackgroundPrefixWriter {
    pub(crate) fn disabled() -> Self {
        Self { tx: None }
    }

    /// Bounded queue. The sender never waits for the write or for `fsync`.
    pub(crate) fn start() -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(4);
        let spawned = std::thread::Builder::new()
            .name("ax-prefix-ckpt".to_string())
            .spawn(move || {
                while let Ok(bytes) = rx.recv() {
                    let _ = write_checkpoint_bytes(&bytes);
                }
            });
        match spawned {
            Ok(_join) => Self { tx: Some(tx) },
            Err(_error) => Self { tx: None },
        }
    }

    pub(crate) fn try_submit(&self, bytes: Vec<u8>) -> bool {
        match &self.tx {
            Some(tx) => tx.try_send(bytes).is_ok(),
            None => false,
        }
    }
}

fn write_checkpoint_bytes(bytes: &[u8]) -> std::io::Result<()> {
    let dir = std::env::temp_dir().join("ax-engine-prefix-ckpt");
    std::fs::create_dir_all(&dir)?;
    static NEXT_CHECKPOINT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let seq = NEXT_CHECKPOINT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = dir.join(format!("pfx-{}-{seq}.bin", std::process::id()));
    // No fsync: cancel and spill must not wait for durability beyond the
    // in-memory receipt the caller already observed.
    std::fs::write(path, bytes)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::engine::EngineCore;
    use crate::generation::GenerationKind;
    use crate::ids::{CacheGroupId, ModelId, RequestId, SequenceNo, StepId};
    use crate::kv::KvManagerConfig;
    use crate::request::{RequestState, RequestSubmission};
    use crate::runner::{DeterministicRunner, ExecutionRunner, RunnerInput, RunnerOutput};
    use crate::sampling::SamplingParams;
    use crate::scheduler::{Scheduler, SchedulerInput};
    use std::sync::{Arc, Mutex};

    #[test]
    fn incompatible_chunk_history_is_a_miss() {
        let stored = CommittedPrefix::from_committed_tokens(vec![1, 2, 3, 4], vec![4], false);
        assert!(!chunk_history_is_miss(&stored, &[4]));
        assert!(chunk_history_is_miss(&stored, &[2, 2]));
        assert!(chunk_history_is_miss(&stored, &[8]));
    }

    #[test]
    fn adaptive_chunk_resumes_at_live_cursor() {
        let mut engine = engine_with_blocks(512);
        engine.set_adaptive_prefill(true);
        let prompt: Vec<u32> = (0..5000).collect();
        engine
            .submit(submission(1, 1, prompt.clone(), 4))
            .expect("submit");
        let first = engine.step(4096, true).expect("first step");
        let first_item = &first
            .schedule_plan
            .execution_batch
            .as_ref()
            .expect("batch")
            .items[0];
        assert_eq!(first_item.position_range.start, 0);
        assert_eq!(first_item.scheduled_token_count, 2048);
        assert_eq!(first_item.input_token_slice, prompt[..2048]);
        assert_eq!(
            decision_value(&first.schedule_plan, ADAPTIVE_PREFILL_CHUNK_ROUTE_KEY),
            Some(2048)
        );
        let after = engine
            .request_manager()
            .snapshot(RequestId(1))
            .expect("snapshot");
        assert_eq!(after.processed_prompt_tokens, 2048);

        let second = engine.step(4096, true).expect("second step");
        let second_item = &second.schedule_plan.execution_batch.expect("batch").items[0];
        assert_eq!(second_item.position_range.start, 2048);
        assert_eq!(second_item.scheduled_token_count, 2048);
        assert_eq!(second_item.input_token_slice, prompt[2048..4096]);

        // Another prefill is not a decode competitor, so the idle quantum stays
        // 2048. A real decode candidate selects the decoding quantum.
        let scheduler = Scheduler::new();
        let mut idle = SchedulerInput::new(
            StepId(1),
            vec![
                text_snapshot(1, 1, &prompt, 2048, &[]),
                text_snapshot(2, 2, &prompt, 0, &[]),
            ],
            None,
            10_000,
        );
        idle.adaptive_prefill = true;
        idle.model_prefill_safety_cap = MODEL_PREFILL_SAFETY_CAP;
        let idle_plan = scheduler.plan(&idle);
        let idle_item = idle_plan
            .execution_batch
            .expect("idle batch")
            .items
            .into_iter()
            .find(|item| item.request_id == RequestId(1))
            .expect("resumed prefill");
        assert_eq!(idle_item.position_range.start, 2048);
        assert_eq!(idle_item.scheduled_token_count, 2048);

        let mut decoding = SchedulerInput::new(
            StepId(2),
            vec![
                text_snapshot(1, 1, &prompt, 2048, &[]),
                text_snapshot(3, 3, &[9, 9, 9, 9], 4, &[7]),
            ],
            None,
            10_000,
        );
        decoding.adaptive_prefill = true;
        decoding.model_prefill_safety_cap = MODEL_PREFILL_SAFETY_CAP;
        let decoding_plan = scheduler.plan(&decoding);
        let decoding_item = decoding_plan
            .execution_batch
            .expect("decoding batch")
            .items
            .into_iter()
            .find(|item| item.request_id == RequestId(1))
            .expect("capped prefill");
        assert_eq!(decoding_item.position_range.start, 2048);
        assert_eq!(decoding_item.scheduled_token_count, 512);
        assert_eq!(decoding_item.input_token_slice, prompt[2048..2560]);
    }

    #[test]
    fn pressure_preserves_decode_and_atomic_media() {
        let prompt = vec![7u32; 2000];
        let scheduler = Scheduler::new();
        let mut with_decode = SchedulerInput::new(
            StepId(4),
            vec![
                text_snapshot(1, 1, &prompt, 0, &[]),
                text_snapshot(2, 2, &[1, 2, 3, 4], 4, &[8]),
            ],
            Some("kv_low_free_blocks:1/8".into()),
            5000,
        );
        with_decode.adaptive_prefill = true;
        with_decode.model_prefill_safety_cap = MODEL_PREFILL_SAFETY_CAP;
        let plan = scheduler.plan(&with_decode);
        let batch = plan.execution_batch.expect("decode batch");
        let decode = batch
            .items
            .iter()
            .find(|item| item.request_id == RequestId(2))
            .expect("decode");
        assert_eq!(decode.scheduled_token_count, 1);
        let text = batch
            .items
            .iter()
            .find(|item| item.request_id == RequestId(1))
            .expect("text prefill");
        assert!(text.scheduled_token_count <= 256);
        assert!(text.scheduled_token_count <= 512);

        let mut media_input = SchedulerInput::new(
            StepId(5),
            vec![{
                let mut media = text_snapshot(3, 3, &vec![3u32; 300], 0, &[]);
                media.has_multimodal_inputs = true;
                media
            }],
            Some("kv_low_free_blocks:1/8".into()),
            5000,
        );
        media_input.adaptive_prefill = true;
        media_input.model_prefill_safety_cap = MODEL_PREFILL_SAFETY_CAP;
        let media_plan = scheduler.plan(&media_input);
        let media = &media_plan.execution_batch.expect("media batch").items[0];
        assert_eq!(media.scheduled_token_count, 300);

        assert!(spill_victim_eligible(SpillVictimProfile {
            text: true,
            mtp_active: false,
            generated_tokens: 0,
            processed_prompt_tokens: 128,
            prompt_len: 2000,
            runnable: true,
            spill_attempts: 0,
        }));
        assert!(!spill_victim_eligible(SpillVictimProfile {
            text: true,
            mtp_active: false,
            generated_tokens: 1,
            processed_prompt_tokens: 4,
            prompt_len: 4,
            runnable: true,
            spill_attempts: 0,
        }));
        assert!(!spill_victim_eligible(SpillVictimProfile {
            text: false,
            mtp_active: false,
            generated_tokens: 0,
            processed_prompt_tokens: 10,
            prompt_len: 300,
            runnable: true,
            spill_attempts: 0,
        }));
        assert!(!spill_victim_eligible(SpillVictimProfile {
            text: true,
            mtp_active: true,
            generated_tokens: 0,
            processed_prompt_tokens: 10,
            prompt_len: 300,
            runnable: true,
            spill_attempts: 0,
        }));
    }

    #[test]
    fn spill_ack_precedes_release() {
        let script = Script::success();
        let mut engine = scripted_engine(script.clone());
        engine.set_pressure_requeue(true);
        engine
            .submit(submission(1, 1, vec![1, 2, 3, 4, 5, 6, 7, 8], 2))
            .expect("submit");
        engine.step(4, true).expect("commit a prefix");
        engine.step(4, true).expect("spill");
        let events = script.events.lock().expect("events").clone();
        let checkpoint = events.iter().position(|event| event == "checkpoint");
        let release = events
            .iter()
            .position(|event| event == "release_restorable");
        assert!(
            checkpoint
                .is_some_and(|checkpoint| release.is_some_and(|release| checkpoint < release))
        );
        assert!(!events.iter().any(|event| event == "release_request_state"));
        assert_eq!(engine.ssd_spill_hits(), 1);
        let snapshot = engine
            .request_manager()
            .snapshot(RequestId(1))
            .expect("snapshot");
        assert_eq!(snapshot.state, RequestState::BlockedOnMemory);
        assert_eq!(snapshot.processed_prompt_tokens, 4);
    }

    #[test]
    fn spill_failure_recomputes() {
        let script = Script::fail();
        let mut engine = scripted_engine(script.clone());
        engine.set_pressure_requeue(true);
        engine
            .submit(submission(1, 1, vec![1, 2, 3, 4, 5, 6, 7, 8], 2))
            .expect("submit");
        engine.step(4, true).expect("commit");
        engine.step(4, true).expect("failed spill");
        let events = script.events.lock().expect("events").clone();
        assert!(events.contains(&"checkpoint".to_string()));
        assert!(events.contains(&"release_request_state".to_string()));
        assert!(!events.iter().any(|event| event == "release_restorable"));
        assert_eq!(engine.ssd_spill_hits(), 0);
        let snapshot = engine
            .request_manager()
            .snapshot(RequestId(1))
            .expect("snapshot");
        assert_eq!(snapshot.processed_prompt_tokens, 0);
        assert_eq!(snapshot.state, RequestState::Runnable);

        let starved = Script::success();
        let mut no_space = scripted_engine(starved.clone());
        no_space.set_pressure_requeue(true);
        no_space.set_spill_budget_bytes(0);
        no_space
            .submit(submission(1, 1, vec![1, 2, 3, 4, 5, 6, 7, 8], 2))
            .expect("submit");
        no_space.step(4, true).expect("commit");
        no_space.step(4, true).expect("reservation miss");
        let events = starved.events.lock().expect("events").clone();
        assert!(
            !events.iter().any(|event| event == "checkpoint"),
            "a failed reservation must not write a checkpoint"
        );
        assert_eq!(no_space.ssd_spill_hits(), 0);
        assert_eq!(
            no_space
                .request_manager()
                .snapshot(RequestId(1))
                .expect("snapshot")
                .processed_prompt_tokens,
            0
        );
    }

    #[test]
    fn cancel_keeps_only_committed_prefix() {
        let script = Script::success();
        script
            .unconfirmed_tokens
            .lock()
            .expect("unconfirmed")
            .extend_from_slice(&[90, 91, 92]);
        let mut engine = scripted_engine(script.clone());
        engine.set_cancel_keep_prefix(true);
        let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8];
        engine
            .submit(submission(7, 1, prompt.clone(), 2))
            .expect("submit");
        engine.step(3, true).expect("partial prefill");
        let committed = engine
            .request_manager()
            .snapshot(RequestId(7))
            .expect("snapshot")
            .processed_prompt_tokens;
        assert_eq!(committed, 3);
        engine.cancel(RequestId(7)).expect("cancel");
        let saved = engine.cancel_checkpoints();
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].cut, 3);
        assert_eq!(saved[0].tokens, prompt[..3]);
        assert!(!saved[0].tokens.iter().any(|token| *token >= 90));
        let events = script.events.lock().expect("events").clone();
        assert!(
            !events.iter().any(|event| event == "checkpoint"),
            "cancel persists the accepted prefix and does not capture runner state"
        );
        let snapshot = engine
            .request_manager()
            .snapshot(RequestId(7))
            .expect("terminal snapshot");
        assert_eq!(snapshot.state, RequestState::Cancelled);
    }

    #[test]
    fn cancel_checkpoint_is_idempotent() {
        let mut engine = scripted_engine(Script::success());
        engine.set_cancel_keep_prefix(true);
        engine
            .submit(submission(3, 1, vec![4, 5, 6, 7], 2))
            .expect("submit");
        engine.step(2, true).expect("commit");
        engine.cancel(RequestId(3)).expect("first cancel");
        assert!(engine.cancel(RequestId(3)).is_err());
        assert_eq!(engine.cancel_checkpoints().len(), 1);
        assert_eq!(engine.cancel_checkpoint_attempts(), 1);
    }

    #[test]
    fn pressure_retry_is_bounded() {
        let failing = Script::fail();
        let mut engine = scripted_engine(failing.clone());
        engine.set_pressure_requeue(true);
        engine
            .submit(submission(1, 1, vec![1, 2, 3, 4, 5, 6, 7, 8], 2))
            .expect("submit");
        for _ in 0..6 {
            engine.step(4, true).expect("step");
        }
        let checkpoints = failing
            .events
            .lock()
            .expect("events")
            .iter()
            .filter(|event| *event == "checkpoint")
            .count();
        assert_eq!(checkpoints, 2);
        assert_eq!(engine.spill_attempts_for(RequestId(1)), 2);
        assert_eq!(engine.ssd_spill_hits(), 0);

        let holding = Script::success();
        let mut held = scripted_engine(holding);
        held.set_pressure_requeue(true);
        held.submit(submission(1, 1, vec![1, 2, 3, 4, 5, 6, 7, 8], 2))
            .expect("submit");
        let mut failed_on = None;
        for step in 1..=80 {
            held.step(4, true).expect("step");
            let snapshot = held
                .request_manager()
                .snapshot(RequestId(1))
                .expect("snapshot");
            if snapshot.state == RequestState::Failed {
                failed_on = Some(step);
                break;
            }
        }
        let failed_on = failed_on.expect("64-step bound fails the held request");
        assert!(failed_on <= 70, "failed_on={failed_on}");
        assert!(held.spill_attempts_for(RequestId(1)) <= MAX_PRESSURE_SPILLS);
        assert_eq!(held.ssd_spill_hits(), 1);
    }

    fn decision_value(plan: &crate::scheduler::SchedulePlan, key: &str) -> Option<u32> {
        plan.execution_batch.as_ref().and_then(|batch| {
            batch
                .route_metadata
                .crossover_decisions
                .iter()
                .find(|(existing, _)| existing == key)
                .map(|(_, value)| *value)
        })
    }

    fn engine_with_blocks(total_blocks: u32) -> EngineCore {
        EngineCore::with_kv_config(KvManagerConfig::validated(
            CacheGroupId(2),
            16,
            total_blocks,
        ))
    }

    fn scripted_engine(script: Script) -> EngineCore {
        EngineCore::with_runtime_components(
            KvManagerConfig::validated(CacheGroupId(2), 16, 64),
            ScriptedRunner { script },
            crate::sampling::DeterministicSampler,
        )
    }

    fn submission(
        request_id: u64,
        arrival: u64,
        input_tokens: Vec<u32>,
        max_output_tokens: u32,
    ) -> RequestSubmission {
        RequestSubmission {
            request_id: RequestId(request_id),
            model_id: ModelId("qwen3".into()),
            input_tokens,
            multimodal_inputs: Default::default(),
            sampling_params: SamplingParams::default(),
            max_output_tokens,
            arrival_sequence: SequenceNo(arrival),
            metadata: None,
            prefill_hints: Default::default(),
        }
    }

    fn text_snapshot(
        request_id: u64,
        arrival: u64,
        prompt: &[u32],
        processed: u32,
        generated: &[u32],
    ) -> crate::request::RequestSnapshot {
        crate::request::RequestSnapshot {
            request_id: RequestId(request_id),
            model_id: ModelId("qwen3".into()),
            arrival_sequence: SequenceNo(arrival),
            state: RequestState::Runnable,
            prompt_tokens: prompt.to_vec(),
            processed_prompt_tokens: processed,
            generated_tokens: generated.to_vec(),
            generated_token_logprobs: vec![Some(0.0); generated.len()],
            prompt_len: prompt.len() as u32,
            generated_len: generated.len() as u32,
            max_output_tokens: 16,
            cancel_requested: false,
            has_multimodal_inputs: false,
            execution_plan_ref: None,
            route_metadata_hint: crate::scheduler::RouteMetadata::empty(),
            terminal_stop_reason: None,
            last_error: None,
            generation_kind: GenerationKind::Autoregressive,
            diffusion_denoise_steps_in_block: 0,
            diffusion_commit_ready: false,
            prefill_hints: Default::default(),
            diffusion_block_committed: false,
            route_deferred_steps: 0,
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum ReceiptMode {
        Adopt,
        Fail,
    }

    #[derive(Clone)]
    struct Script {
        events: Arc<Mutex<Vec<String>>>,
        mode: ReceiptMode,
        unconfirmed_tokens: Arc<Mutex<Vec<u32>>>,
    }

    impl Script {
        fn success() -> Self {
            Self {
                events: Arc::new(Mutex::new(Vec::new())),
                mode: ReceiptMode::Adopt,
                unconfirmed_tokens: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn fail() -> Self {
            let mut script = Self::success();
            script.mode = ReceiptMode::Fail;
            script
        }
    }

    struct ScriptedRunner {
        script: Script,
    }

    impl std::fmt::Debug for ScriptedRunner {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("ScriptedRunner")
        }
    }

    impl ExecutionRunner for ScriptedRunner {
        fn run(&self, input: RunnerInput) -> RunnerOutput {
            DeterministicRunner.run(input)
        }

        fn checkpoint_request(
            &self,
            request_id: RequestId,
            _reason: CheckpointReason,
            committed_limit: u32,
        ) -> CheckpointReceipt {
            if let Ok(mut events) = self.script.events.lock() {
                events.push("checkpoint".to_string());
            }
            match self.script.mode {
                ReceiptMode::Adopt => CheckpointReceipt {
                    request_id,
                    cut: committed_limit,
                    committed: true,
                    ssd_hit: true,
                    unsupported_layout: false,
                    device_capture: true,
                },
                ReceiptMode::Fail => CheckpointReceipt {
                    request_id,
                    cut: committed_limit,
                    committed: false,
                    ssd_hit: false,
                    unsupported_layout: false,
                    device_capture: false,
                },
            }
        }

        fn release_request_state(&self, _request_id: RequestId) {
            if let Ok(mut events) = self.script.events.lock() {
                events.push("release_request_state".to_string());
            }
        }

        fn release_restorable_state(&self, request_id: RequestId) {
            if let Ok(mut events) = self.script.events.lock() {
                events.push("release_restorable".to_string());
            }
            let _ = request_id;
        }

        fn memory_pressure_sample(&self) -> Option<MemoryPressureSample> {
            Some(MemoryPressureSample {
                active_bytes: 1_000,
                limit_bytes: 1_000,
                idle_cache_bytes: 0,
                next_chunk_workspace_bytes: 0,
            })
        }

        fn mtp_active(&self) -> bool {
            false
        }
    }
}
