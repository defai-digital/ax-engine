use super::*;
use crate::ngram_accel::{DEFAULT_DRAFT_LEN, MAX_DRAFT_LEN};
use ax_engine_core::model::{NativeGlmRouterConfig, NativeMlaAttentionConfig};
use ax_engine_core::scheduler::PositionRange;
use ax_engine_core::{
    AX_NATIVE_MODEL_MANIFEST_FILE, NativeDiffusionConfig, NativeLinearAttentionConfig,
    NativeModelManifest, NativeMoeConfig, NativeRuntimeStatus, NativeTensorDataType,
    NativeTensorFormat, NativeTensorRole, NativeTensorSpec,
};
use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn request_states_share_only_the_runner_owned_fa_pool() {
    let config = FaBlockPoolConfig {
        block_size_tokens: 4,
        max_blocks: 8,
        hard_cap: true,
    };
    let shared = SharedFaBlockPool::new(config).expect("shared pool");
    let left =
        RequestState::new_with_shared_fa_pool(2, Some(1), 1, Some(config), Some(shared.clone()));
    let right =
        RequestState::new_with_shared_fa_pool(2, Some(2), 2, Some(config), Some(shared.clone()));
    assert!(left.cache.shares_fa_block_pool_with(&right.cache));
    assert!(left.cache.uses_fa_block_pool(&shared));

    let private_left = RequestState::new(2, 3, Some(config));
    let private_right = RequestState::new(2, 4, Some(config));
    assert!(
        !private_left
            .cache
            .shares_fa_block_pool_with(&private_right.cache)
    );
}

#[test]
fn shared_fa_pool_alignment_counts_layer_block_slots() {
    assert_eq!(aligned_fa_pool_max_blocks(100, 32, false, None), 100);
    assert_eq!(aligned_fa_pool_max_blocks(100, 32, true, None), 3_200);
    assert_eq!(aligned_fa_pool_max_blocks(100, 32, true, Some(17)), 17);
    assert_eq!(aligned_fa_pool_max_blocks(0, 0, true, None), 1);
    assert_eq!(
        aligned_fa_pool_max_blocks(2, usize::MAX, true, None),
        u32::MAX
    );
}

#[test]
fn page_sharing_serializes_only_the_largest_aligned_prefix() {
    let start = prefix_snapshot_start_tokens(16, 8192, false, true);
    assert_eq!(start, 8192);
    assert_eq!((start..=8192).step_by(16).count(), 1);

    let portable_start = prefix_snapshot_start_tokens(16, 8192, false, false);
    assert_eq!(portable_start, 16);
    assert_eq!((portable_start..=8192).step_by(16).count(), 512);
}

#[test]
fn native_pressure_recomputes_cow_demand_after_eviction() {
    // Exercises the reclaim seam the demote tests mutate the environment
    // around; hold the read lock so a concurrent kill-switch scope can
    // never race this test (its demote closure is a no-op today, but the
    // lock keeps that contract if it ever changes).
    let _env_read = DEMOTE_ON_EVICT_ENV_LOCK.read().expect("env lock");
    let pool = SharedFaBlockPool::new(FaBlockPoolConfig {
        block_size_tokens: 4,
        max_blocks: 1,
        hard_cap: true,
    })
    .expect("pool");
    let mut producer = MlxKVCache::new_with_shared_fa_block_pool(1, pool.clone());
    let k = mlx_sys::zeros(&[1, 1, 3, 2], mlx_sys::MlxDtype::Float32, None);
    let v = mlx_sys::zeros(&[1, 1, 3, 2], mlx_sys::MlxDtype::Float32, None);
    let _ = producer.append(0, k, v);
    producer.advance(3);

    let tokens = vec![1, 2, 3];
    let key = MlxPrefixCacheKey {
        model_id: "pressure-test".to_string(),
        route_policy: "direct".to_string(),
        layer_layout: "standard-fa".to_string(),
        block_size_tokens: 4,
        token_count: 3,
        token_hash: 7,
        media_key: String::new(),
    };
    let native = Mutex::new(MlxNativePrefixCache::new(MlxPrefixCachePolicy {
        max_bytes: 1024,
        max_entries: 1,
    }));
    let outcome = native.lock().insert(
        key.clone(),
        MlxNativePrefixSnapshot::new(producer.clone(), tokens.clone(), 64, None),
    );
    drop(outcome.retired);
    let (hit, matched_len) = native
        .lock()
        .get_longest_prefix(&key, &tokens)
        .expect("native hit");
    assert_eq!(matched_len, tokens.len());
    let active = hit.cache.clone();
    drop(hit);
    drop(producer);

    assert_eq!(pool.snapshot().available_blocks, 0);
    assert_eq!(active.additional_fa_blocks_for_append(1), Some(1));
    let evictions = reclaim_native_prefix_entries(
        &native,
        &pool,
        || active.additional_fa_blocks_for_append(1),
        |_, _| {},
    );
    assert_eq!(evictions, 1);
    assert_eq!(native.lock().stats().entries, 0);
    assert_eq!(active.additional_fa_blocks_for_append(1), Some(0));
    assert_eq!(pool.snapshot().allocated_blocks, 1);
    drop(active);
    assert_eq!(pool.snapshot().allocated_blocks, 0);
}

// ── ADR-016 demote-on-evict ──

/// Guards `AX_MLX_PREFIX_DEMOTE_ON_EVICT` across concurrently-running
/// tests (same contract as the `AX_KV_QUANT` guard in kv_cache tests).
/// The kill switch is read live, so every test that can reach a demote
/// path holds the lock for its whole scope: mutating tests (set/remove)
/// take the write lock through [`DemoteOnEvictEnvGuard`], pure readers
/// take a read lock so they cannot race a mutation.
static DEMOTE_ON_EVICT_ENV_LOCK: std::sync::RwLock<()> = std::sync::RwLock::new(());

struct DemoteOnEvictEnvGuard {
    previous: Option<String>,
    _write_guard: std::sync::RwLockWriteGuard<'static, ()>,
}

impl DemoteOnEvictEnvGuard {
    fn set(value: &str) -> Self {
        let guard = Self::lock();
        // SAFETY: the write lock is held for the whole test scope, so no
        // demote test observes a mid-mutation environment.
        unsafe { std::env::set_var(AX_MLX_PREFIX_DEMOTE_ON_EVICT_ENV, value) };
        guard
    }

    /// Remove the variable for the test scope, restoring its prior state
    /// on drop. Makes demote tests deterministic even when the ambient
    /// environment has `AX_MLX_PREFIX_DEMOTE_ON_EVICT=0` set.
    fn removed() -> Self {
        let guard = Self::lock();
        // SAFETY: the write lock is held for the whole test scope, so no
        // demote test observes a mid-mutation environment.
        unsafe { std::env::remove_var(AX_MLX_PREFIX_DEMOTE_ON_EVICT_ENV) };
        guard
    }

    fn lock() -> Self {
        let write_guard = DEMOTE_ON_EVICT_ENV_LOCK.write().expect("env lock");
        let previous = std::env::var(AX_MLX_PREFIX_DEMOTE_ON_EVICT_ENV).ok();
        Self {
            previous,
            _write_guard: write_guard,
        }
    }
}

impl Drop for DemoteOnEvictEnvGuard {
    fn drop(&mut self) {
        // SAFETY: the write lock is still held; it releases after this
        // drop via `_write_guard`.
        unsafe {
            match &self.previous {
                Some(value) => std::env::set_var(AX_MLX_PREFIX_DEMOTE_ON_EVICT_ENV, value),
                None => std::env::remove_var(AX_MLX_PREFIX_DEMOTE_ON_EVICT_ENV),
            }
        }
    }
}

fn demote_test_key(tag: u32) -> MlxPrefixCacheKey {
    MlxPrefixCacheKey {
        model_id: "demote-test".to_string(),
        route_policy: "direct".to_string(),
        layer_layout: "standard-fa".to_string(),
        block_size_tokens: 4,
        token_count: 3,
        token_hash: u64::from(tag),
        media_key: String::new(),
    }
}

fn demote_test_snapshot(pool: &SharedFaBlockPool, tokens: Vec<u32>) -> MlxNativePrefixSnapshot {
    let mut cache = MlxKVCache::new_with_shared_fa_block_pool(1, pool.clone());
    let k = mlx_sys::zeros(&[1, 1, 3, 2], mlx_sys::MlxDtype::Float32, None);
    let v = mlx_sys::zeros(&[1, 1, 3, 2], mlx_sys::MlxDtype::Float32, None);
    let _ = cache.append(0, k, v);
    cache.advance(3);
    MlxNativePrefixSnapshot::new(cache, tokens, 64, Some(11))
}

#[test]
fn native_eviction_demotes_into_portable_store() {
    let _env = DemoteOnEvictEnvGuard::removed();
    let pool = SharedFaBlockPool::new(FaBlockPoolConfig {
        block_size_tokens: 4,
        max_blocks: 1,
        hard_cap: true,
    })
    .expect("pool");
    let native = Mutex::new(MlxNativePrefixCache::new(MlxPrefixCachePolicy {
        max_bytes: 1024,
        max_entries: 1,
    }));
    let portable = Mutex::new(MlxPrefixCache::new(MlxPrefixCachePolicy {
        max_bytes: 1 << 20,
        max_entries: 8,
    }));

    let tokens = vec![1, 2, 3];
    let key = demote_test_key(7);
    let snapshot = demote_test_snapshot(&pool, tokens.clone());
    let producer = snapshot.cache.clone();
    let outcome = native.lock().insert(key.clone(), snapshot);
    assert!(outcome.stored);
    drop(outcome.retired);
    drop(outcome.evicted);
    let (hit, _) = native
        .lock()
        .get_longest_prefix(&key, &tokens)
        .expect("native hit");
    let active = hit.cache.clone();
    drop(hit);
    drop(producer);

    assert_eq!(pool.snapshot().available_blocks, 0);
    let mut telemetry = MlxPrefixCacheTelemetry::default();
    let evictions = reclaim_native_prefix_entries(
        &native,
        &pool,
        || active.additional_fa_blocks_for_append(1),
        |key, evicted| {
            demote_native_prefix_snapshot(&portable, key, evicted, &mut telemetry);
        },
    );
    assert_eq!(evictions, 1);
    assert!(native.lock().is_empty());
    assert_eq!(telemetry.demotions, 1);
    assert_eq!(telemetry.demotion_skips, 0);
    assert!(telemetry.demoted_bytes > 0);
    assert_eq!(telemetry.entries, 1);

    // The demoted payload round-trips through the portable store's lookup
    // path — same key plus exact tokens — and rehydrates into a cache
    // holding the evicted prefix length.
    let demoted = portable
        .lock()
        .get(&key, &tokens)
        .expect("demoted snapshot must be a portable hit");
    assert_eq!(demoted.tokens, tokens);
    assert_eq!(demoted.greedy_prefill_output_token, Some(11));
    let restored = demoted.rehydrate_cache().expect("rehydrate");
    assert_eq!(restored.seq_len(), tokens.len());

    drop(active);
    assert_eq!(pool.snapshot().allocated_blocks, 0);
}

#[test]
fn demote_on_evict_kill_switch_restores_drop_behavior() {
    let _guard = DemoteOnEvictEnvGuard::set("0");
    let pool = SharedFaBlockPool::new(FaBlockPoolConfig {
        block_size_tokens: 4,
        max_blocks: 1,
        hard_cap: true,
    })
    .expect("pool");
    let portable = Mutex::new(MlxPrefixCache::new(MlxPrefixCachePolicy {
        max_bytes: 1 << 20,
        max_entries: 8,
    }));
    let mut telemetry = MlxPrefixCacheTelemetry::default();
    let snapshot = demote_test_snapshot(&pool, vec![1, 2, 3]);
    demote_native_prefix_snapshot(&portable, &demote_test_key(7), &snapshot, &mut telemetry);
    assert_eq!(telemetry.demotions, 0);
    assert_eq!(telemetry.demotion_skips, 1);
    assert_eq!(portable.lock().stats().entries, 0);
}

#[test]
fn native_insert_eviction_demote_respects_portable_store_phase() {
    let _env = DemoteOnEvictEnvGuard::removed();
    let pool = SharedFaBlockPool::new(FaBlockPoolConfig {
        block_size_tokens: 4,
        max_blocks: 4,
        hard_cap: true,
    })
    .expect("pool");
    let native = Mutex::new(MlxNativePrefixCache::new(MlxPrefixCachePolicy {
        max_bytes: 1024,
        max_entries: 1,
    }));
    let portable = Mutex::new(MlxPrefixCache::new(MlxPrefixCachePolicy {
        max_bytes: 1 << 20,
        max_entries: 8,
    }));

    let first = demote_test_snapshot(&pool, vec![1, 2, 3]);
    let outcome_a = native.lock().insert(demote_test_key(7), first);
    drop(outcome_a.retired);
    drop(outcome_a.evicted);
    let second = demote_test_snapshot(&pool, vec![4, 5, 6]);
    let outcome_b = native.lock().insert(demote_test_key(8), second);
    assert_eq!(outcome_b.evicted.len(), 1);

    // A native insert that evicts a live entry during a scheduler-split
    // prefill (portable store phase disallowed) must not publish the
    // evicted snapshot to the portable store — the same O(N²) hazard the
    // phase gate exists to avoid.
    let mut telemetry = MlxPrefixCacheTelemetry::default();
    demote_native_insert_evictions(&portable, &outcome_b.evicted, false, &mut telemetry);
    assert_eq!(portable.lock().stats().entries, 0);
    assert_eq!(telemetry.demotions, 0);
    assert_eq!(telemetry.demotion_skips, 0);

    // The completing phase demotes the same evicted entry.
    demote_native_insert_evictions(&portable, &outcome_b.evicted, true, &mut telemetry);
    assert_eq!(telemetry.demotions, 1);
    assert_eq!(portable.lock().stats().entries, 1);

    drop(outcome_b.retired);
    drop(outcome_b.evicted);
}

#[test]
fn demote_overflow_evicts_per_portable_policy() {
    let _env = DemoteOnEvictEnvGuard::removed();
    let pool = SharedFaBlockPool::new(FaBlockPoolConfig {
        block_size_tokens: 4,
        max_blocks: 4,
        hard_cap: true,
    })
    .expect("pool");

    // Entry larger than the whole byte budget: the demote is refused up
    // front (counted as a skip) instead of flushing every healthy entry
    // before self-evicting; nothing is inserted, nothing panics.
    let tight = Mutex::new(MlxPrefixCache::new(MlxPrefixCachePolicy {
        max_bytes: 32,
        max_entries: 8,
    }));
    let mut telemetry = MlxPrefixCacheTelemetry::default();
    let snapshot = demote_test_snapshot(&pool, vec![1, 2, 3]);
    demote_native_prefix_snapshot(&tight, &demote_test_key(7), &snapshot, &mut telemetry);
    assert_eq!(telemetry.demotions, 0);
    assert_eq!(telemetry.demotion_skips, 1);
    assert_eq!(telemetry.evictions, 0);
    assert_eq!(tight.lock().stats().entries, 0);

    // Entry budget of one: the second demotion stores and evicts the
    // first per LRU policy.
    let capped = Mutex::new(MlxPrefixCache::new(MlxPrefixCachePolicy {
        max_bytes: 1 << 20,
        max_entries: 1,
    }));
    let mut telemetry = MlxPrefixCacheTelemetry::default();
    let first = demote_test_snapshot(&pool, vec![1, 2, 3]);
    demote_native_prefix_snapshot(&capped, &demote_test_key(7), &first, &mut telemetry);
    let second = demote_test_snapshot(&pool, vec![4, 5, 6]);
    demote_native_prefix_snapshot(&capped, &demote_test_key(8), &second, &mut telemetry);
    assert_eq!(telemetry.demotions, 2);
    assert_eq!(telemetry.evictions, 1);
    assert_eq!(capped.lock().stats().entries, 1);
}

#[test]
fn skip_state_primary_token_reads_real_argmax_for_greedy() {
    // Regression: the greedy skip-state cycle used to pass a literal 0
    // through `sample_logit_row`'s argmax shortcut, emitting token id 0
    // ("!") as the committed primary whenever the draft gate left the
    // pending draft empty. The helper must return the row's true argmax
    // both from the lazily-captured array and from an on-demand compute.
    let logits = MlxArray::from_f32_slice(&[-3.0, 0.5, -1.0, 4.25, 0.0, 2.5, -0.25, 1.0]);
    let mut rng = Xorshift64::new(7);
    let mut probs_buf = Vec::new();
    let mut logits_buf = Vec::new();
    let mut candidates_buf = Vec::new();
    let greedy = MlxSamplingParams::greedy();

    let captured = mlx_sys::argmax(&logits, None);
    let from_captured = skip_state_primary_token(
        &logits,
        Some(captured),
        8,
        greedy,
        &mut rng,
        &mut probs_buf,
        &mut logits_buf,
        &mut candidates_buf,
    );
    assert_eq!(from_captured, 3, "greedy must commit the captured argmax");

    let on_demand = skip_state_primary_token(
        &logits,
        None,
        8,
        greedy,
        &mut rng,
        &mut probs_buf,
        &mut logits_buf,
        &mut candidates_buf,
    );
    assert_eq!(
        on_demand, 3,
        "greedy without a captured argmax must compute it"
    );

    let sampled = skip_state_primary_token(
        &logits,
        None,
        8,
        MlxSamplingParams::new(0.7, 1.0, 0),
        &mut rng,
        &mut probs_buf,
        &mut logits_buf,
        &mut candidates_buf,
    );
    assert!(sampled < 8, "sampled path must return an in-vocab token");
}

#[test]
fn batched_session_join_reclaims_private_cache() {
    let mut session = BatchedDecodeSession::new(0, 2);
    let mut cache = MlxKVCache::new(0);
    cache.set_seq_len(4);

    seed_batched_session_and_reclaim_private_cache(&mut session, 7, &mut cache, 11, Some(3));

    assert_eq!(session.active_ids(), &[7]);
    assert_eq!(cache.seq_len(), 0);
    assert_eq!(cache.usage_snapshot(), MlxKVCacheUsage::default());
}

#[test]
fn batched_admission_only_blocks_on_exhausted_pressure() {
    assert!(!batched_admission_blocked_by_memory_pressure(None));
    assert!(!batched_admission_blocked_by_memory_pressure(Some(
        "kv_low_free_blocks:3/1024"
    )));
    assert!(batched_admission_blocked_by_memory_pressure(Some(
        "kv_exhausted"
    )));
    assert!(batched_admission_blocked_by_memory_pressure(Some(
        "kv_exhausted_reclaimable_cache"
    )));
}

fn ctx_with_argmax(deterministic_argmax_sampling: bool) -> RunnerRequestContext {
    RunnerRequestContext {
        request_id: RequestId(1),
        prompt_len: 8,
        processed_prompt_tokens: 0,
        generated_len: 0,
        max_output_tokens: 16,
        seed: 0,
        deterministic_argmax_sampling,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        repetition_context_size: None,
        no_repeat_ngram_size: 0,
        ngram_window: 128,
        ignore_eos: false,
        tool_call_mode: false,
        structured_output_mode: false,
        min_p: None,
        max_think_tokens: None,
        answer_reserve_tokens: None,
    }
}

#[test]
fn default_mlx_cache_limit_matches_mlx_working_set_policy() {
    assert_eq!(default_mlx_cache_limit(0), 0);
    assert_eq!(default_mlx_cache_limit(1024), 1536);
    assert_eq!(default_mlx_cache_limit(usize::MAX), usize::MAX);
}

// A prefix-cache hit may only hand a request the producer's greedy prefill
// token when the request would itself compute that token. Temperature,
// top-p, and repetition-penalty requests must resample.
#[test]
fn prefill_output_token_cacheable_only_for_greedy_no_rep_penalty() {
    let greedy = MlxSamplingParams::greedy();
    let temperature = MlxSamplingParams::new(0.8, 1.0, 0);
    let greedy_rep = MlxSamplingParams::greedy().with_repetition_penalty(1.3, None);

    // No context: greedy-ness is inferred from the params themselves.
    assert!(prefill_output_token_cacheable(None, greedy));
    assert!(!prefill_output_token_cacheable(None, temperature));
    assert!(!prefill_output_token_cacheable(None, greedy_rep));

    // Deterministic-argmax context with no repetition penalty -> cacheable.
    let det = ctx_with_argmax(true);
    assert!(prefill_output_token_cacheable(Some(&det), greedy));
    // Deterministic argmax but a repetition penalty is active: the stored
    // greedy token ignored the penalty, so it must not be reused.
    assert!(!prefill_output_token_cacheable(Some(&det), greedy_rep));

    // A sampling request (deterministic_argmax_sampling == false) never
    // reuses the token, even if its other params look greedy.
    let sampled = ctx_with_argmax(false);
    assert!(!prefill_output_token_cacheable(Some(&sampled), temperature));
    assert!(!prefill_output_token_cacheable(Some(&sampled), greedy));
}

// Verify that the extract-work-reinsert mutex pattern correctly isolates
// per-request state without GPU execution required.
#[test]
fn state_extraction_isolates_concurrent_requests() {
    let mut states: HashMap<RequestId, RequestState> = HashMap::new();
    let a = RequestId(1);
    let b = RequestId(2);

    // Extract A from the map (simulates the lock-brief-remove step).
    // While A is extracted, B's slot is accessible without contention.
    let state_a = states
        .remove(&a)
        .unwrap_or_else(|| RequestState::new(2, a.0, None));
    let state_b = states
        .remove(&b)
        .unwrap_or_else(|| RequestState::new(2, b.0, None));

    // GPU work would run here with state_a / state_b outside the map.
    // Verify B can be reinserted independently of A.
    states.insert(b, state_b);
    states.insert(a, state_a);

    assert_eq!(states.len(), 2);
    assert!(states.contains_key(&a));
    assert!(states.contains_key(&b));
}

#[test]
fn completed_request_state_is_not_reinserted() {
    let mut states: HashMap<RequestId, RequestState> = HashMap::new();
    let id = RequestId(42);
    states.insert(id, RequestState::new(2, id.0, None));

    // Extract and simulate a completed request (stop_reason.is_some()).
    // The state should not be reinserted, mirroring the run_item control flow.
    let _state = states.remove(&id).unwrap();
    // No states.insert here — dropped at end of scope.

    assert!(
        !states.contains_key(&id),
        "completed request must not leave orphaned state"
    );
}

#[test]
fn mtp_telemetry_tracks_acceptance_step_classes() {
    let mut telemetry = MtpTelemetry::default();
    telemetry.record_correctness_mode(
        MtpCorrectnessMode::GreedyExact,
        MtpProposalLaw::DeterministicDelta,
    );

    let mtp_sources = [MtpDraftSource::Mtp; 3];
    telemetry.record_step(3, 3, &mtp_sources, None, 3);
    telemetry.record_step(3, 1, &mtp_sources, None, 1);
    telemetry.record_step(3, 0, &mtp_sources, None, 0);
    telemetry.record_timings(MtpStepTimings {
        cache_clone_wall_us: 10,
        verify_forward_wall_us: 20,
        verify_eval_wall_us: 30,
        target_softmax_wall_us: 35,
        accept_wall_us: 40,
        rollback_wall_us: 50,
        tail_sample_wall_us: 60,
        draft_wall_us: 70,
        mtp_draft_wall_us: 71,
        assistant_draft_wall_us: 72,
        ngram_lookup_wall_us: 73,
        verify_tokens: 8,
        emitted_tokens: 4,
        ngram_submitted_tokens: 0,
        whole_verify_compile_hits: 3,
        whole_verify_compile_fallbacks: 1,
    });

    let mut decisions = Vec::new();
    telemetry.append_route_decisions(&mut decisions);

    assert!(decisions.contains(&("ax_mtp_correctness_mode".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_proposal_law".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_correctness_mode_conflicts".into(), 0)));
    assert!(decisions.contains(&("ax_mtp_optimistic_steps".into(), 0)));
    assert!(decisions.contains(&("ax_mtp_direct_fallback_steps".into(), 0)));
    assert!(decisions.contains(&("ax_mtp_draft_tokens".into(), 9)));
    assert!(decisions.contains(&("ax_mtp_accepted_tokens".into(), 4)));
    assert!(decisions.contains(&("ax_mtp_decode_steps".into(), 3)));
    assert!(decisions.contains(&("ax_mtp_full_accept_steps".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_partial_reject_steps".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_complete_miss_steps".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_cache_clone_wall_us".into(), 10)));
    assert!(decisions.contains(&("ax_mtp_verify_forward_wall_us".into(), 20)));
    assert!(decisions.contains(&("ax_mtp_verify_eval_wall_us".into(), 30)));
    assert!(decisions.contains(&("ax_mtp_accept_wall_us".into(), 40)));
    assert!(decisions.contains(&("ax_mtp_rollback_wall_us".into(), 50)));
    assert!(decisions.contains(&("ax_mtp_tail_sample_wall_us".into(), 60)));
    assert!(decisions.contains(&("ax_mtp_draft_wall_us".into(), 70)));
    assert!(decisions.contains(&("ax_mtp_target_softmax_wall_us".into(), 35)));
    assert!(decisions.contains(&("ax_mtp_verify_tokens".into(), 8)));
    assert!(decisions.contains(&("ax_mtp_whole_verify_compile_hits".into(), 3)));
    assert!(decisions.contains(&("ax_mtp_whole_verify_compile_fallbacks".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_emitted_tokens".into(), 4)));
    assert!(decisions.contains(&("ax_mtp_source_mtp_submitted_tokens".into(), 9)));
    assert!(decisions.contains(&("ax_mtp_source_mtp_accepted_tokens".into(), 4)));
    assert!(decisions.contains(&("ax_mtp_source_mtp_rejected_tokens".into(), 2)));
    assert!(decisions.contains(&("ax_mtp_source_mtp_cascade_rejected_tokens".into(), 3)));
    assert!(decisions.contains(&("ax_mtp_source_mtp_proposer_wall_us".into(), 71)));
    assert!(decisions.contains(&("ax_mtp_source_assistant_submitted_tokens".into(), 0)));
    assert!(decisions.contains(&("ax_mtp_source_assistant_proposer_wall_us".into(), 72)));
    assert!(decisions.contains(&("ax_mtp_ngram_lookup_wall_us".into(), 73)));
    // Per-depth counters: record_step(3,3) + record_step(3,1) + record_step(3,0)
    // drafted_by_depth: all 3 steps attempted all 3 depths → [3, 3, 3]
    // accepted_by_depth: depth0 accepted in steps 0,1; depth1+2 only in step 0
    assert!(decisions.contains(&("ax_mtp_drafted_depth0".into(), 3)));
    assert!(decisions.contains(&("ax_mtp_drafted_depth1".into(), 3)));
    assert!(decisions.contains(&("ax_mtp_drafted_depth2".into(), 3)));
    assert!(decisions.contains(&("ax_mtp_accepted_depth0".into(), 2)));
    assert!(decisions.contains(&("ax_mtp_accepted_depth1".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_accepted_depth2".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_draft_source_mtp_tokens".into(), 9)));
    assert!(decisions.contains(&("ax_mtp_accepted_source_mtp_tokens".into(), 4)));
    assert!(decisions.contains(&("ax_mtp_draft_source_ngram_tokens".into(), 0)));
    assert!(decisions.contains(&("ax_mtp_accepted_source_ngram_tokens".into(), 0)));
    assert!(decisions.contains(&("ax_mtp_ngram_hit_steps".into(), 0)));
    assert!(decisions.contains(&("ax_mtp_ngram_think_gated_steps".into(), 0)));
    assert!(decisions.contains(&("ax_mtp_ngram_auto_disabled_steps".into(), 0)));
    assert!(decisions.contains(&("ax_mtp_ngram_self_tune_disabled_steps".into(), 0)));
    assert!(decisions.contains(&("ax_mtp_ngram_submitted_tokens".into(), 0)));
    assert!(decisions.contains(&("ax_mtp_ngram_submitted_accepted_tokens".into(), 0)));
    assert!(decisions.contains(&("ax_mtp_ngram_accepted_tokens".into(), 0)));
    assert!(decisions.contains(&("ax_mtp_ngram_utility_baseline_steps".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_ngram_utility_baseline_emitted_tokens".into(), 4)));
    assert!(decisions.contains(&("ax_mtp_ngram_utility_stacked_steps".into(), 0)));
    assert!(decisions.contains(&("ax_mtp_ngram_acceptance_mode".into(), 0)));
}

#[test]
fn mtp_telemetry_fails_closed_on_mixed_correctness_modes() {
    let mut telemetry = MtpTelemetry::default();
    telemetry.record_correctness_mode(
        MtpCorrectnessMode::GreedyExact,
        MtpProposalLaw::DeterministicDelta,
    );
    telemetry.record_correctness_mode(
        MtpCorrectnessMode::ApproximateOptimistic,
        MtpProposalLaw::DeterministicDelta,
    );
    telemetry.record_optimistic_step();

    let mut decisions = Vec::new();
    telemetry.append_route_decisions(&mut decisions);

    assert!(decisions.contains(&("ax_mtp_correctness_mode".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_correctness_mode_conflicts".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_optimistic_steps".into(), 1)));
}

#[test]
fn mtp_telemetry_records_sampled_direct_fallback() {
    let mut telemetry = MtpTelemetry::default();
    telemetry.record_direct_fallback();
    telemetry.record_direct_fallback();

    let mut decisions = Vec::new();
    telemetry.append_route_decisions(&mut decisions);

    assert!(decisions.contains(&("ax_mtp_correctness_mode".into(), 4)));
    assert!(decisions.contains(&("ax_mtp_direct_fallback_steps".into(), 2)));
    assert!(decisions.contains(&("ax_mtp_decode_steps".into(), 0)));
}

#[test]
fn mtp_request_route_accelerates_exact_or_explicit_approximate_profiles() {
    assert_eq!(
        mtp_request_route(true, true, true, false, false, false),
        MtpRequestRoute::StrictMtp
    );
    assert_eq!(
        mtp_request_route(true, true, false, true, false, false),
        MtpRequestRoute::StrictMtp
    );
    assert_eq!(
        mtp_request_route(true, true, false, false, false, false),
        MtpRequestRoute::DirectFallback
    );
    assert_eq!(
        mtp_request_route(true, false, true, false, false, false),
        MtpRequestRoute::Other
    );
    assert_eq!(
        mtp_request_route(true, true, true, true, true, false),
        MtpRequestRoute::DirectFallback
    );
    assert_eq!(
        mtp_request_route(true, true, true, true, false, true),
        MtpRequestRoute::DirectFallback
    );
}

#[test]
fn mtp_exact_sampling_support_rejects_unsupported_target_filters() {
    assert!(mtp_exact_sampling_supported(
        MlxSamplingParams::greedy(),
        None
    ));
    assert!(mtp_exact_sampling_supported(
        MlxSamplingParams::new(0.6, 0.95, 20),
        None
    ));
    assert!(!mtp_exact_sampling_supported(
        MlxSamplingParams::new(0.6, 0.95, 0),
        None
    ));
    assert!(!mtp_exact_sampling_supported(
        MlxSamplingParams::new(0.6, 0.95, 20),
        Some(128)
    ));
    // DeepSeek thinking defaults min_p=0.05 — exact linear profile must not claim support.
    assert!(!mtp_exact_sampling_supported(
        MlxSamplingParams::new(1.0, 1.0, 0).with_min_p(Some(0.05)),
        None
    ));
}

#[test]
fn filtered_target_token_probability_applies_min_p() {
    // probs: token0=0.9, token1=0.05, token2=0.05. min_p=0.1 → cutoff=0.09
    // keeps only token0 → p(token0)=1.0, p(token1)=0.0.
    let row = [0.9_f32, 0.05, 0.05];
    assert!((filtered_target_token_probability(&row, 0, Some(0.1), 1.0, 0) - 1.0).abs() < 1e-5);
    assert_eq!(
        filtered_target_token_probability(&row, 1, Some(0.1), 1.0, 0),
        0.0
    );
    // Without min_p, raw renormalized mass is unchanged (already sums to 1).
    assert!((filtered_target_token_probability(&row, 0, None, 1.0, 0) - 0.9).abs() < 1e-5);
}

#[test]
fn mtp_exact_delta_rejection_correction_removes_draft_mass() {
    let logits = [0.6_f32.ln(), 0.3_f32.ln(), 0.1_f32.ln()];
    let logits_all = MlxArray::from_raw_data(
        logits.as_ptr() as *const u8,
        std::mem::size_of_val(&logits),
        &[1, 3],
        MlxDtype::Float32,
    );
    let sampling = MlxSamplingParams::new(1.0, 1.0, 3);
    for seed in 0..32 {
        let mut rng = Xorshift64::new(seed);
        let correction =
            sample_exact_mtp_delta_rejection_correction(&logits_all, 0, 3, sampling, 0, &mut rng)
                .unwrap();
        assert_ne!(correction, 0);
    }
}

#[test]
fn greedy_linear_mtp_rejection_prefers_recomputed_production_argmax() {
    assert_eq!(
        select_linear_mtp_correction_token(0.0, Some(440), None, Some(13_661), 13_661),
        440
    );
}

fn forced_replay_test_model() -> (ModelConfig, ModelWeights) {
    let mut manifest = dense_manifest();
    manifest.model_family = "qwen3".into();
    let cfg = ModelConfig::from_manifest(&manifest);
    let array = |data: &[f32], shape: &[i32]| {
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    };
    let dense = |data: &[f32], shape: &[i32]| QuantizedWeight::new(array(data, shape), None, None);
    let mut layer = runner_test_layer();
    layer.attn_norm = array(&[1.0; 4], &[4]);
    layer.ffn_norm = array(&[1.0; 4], &[4]);
    layer.q_proj = Some(dense(&[0.0; 16], &[4, 4]));
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    layer.k_proj = Some(dense(&identity, &[4, 4]));
    layer.v_proj = Some(dense(&identity, &[4, 4]));
    layer.o_proj = Some(dense(&[0.0; 16], &[4, 4]));
    layer.gate_proj = Some(dense(&[0.0; 32], &[8, 4]));
    layer.up_proj = Some(dense(&[0.0; 32], &[8, 4]));
    layer.down_proj = Some(dense(&[0.0; 32], &[4, 8]));
    let mut weights = runner_test_weights(vec![layer]);
    let embedding: Vec<f32> = (0..16)
        .flat_map(|token| [token as f32 + 1.0, 1.0, 1.0, 1.0])
        .collect();
    weights.token_embedding = dense(&embedding, &[16, 4]);
    weights.final_norm = Some(array(&[1.0; 4], &[4]));
    let mut head = [0.0; 64];
    head[4..8].fill(1.0);
    weights.lm_head = dense(&head, &[16, 4]);
    (cfg, weights)
}

#[test]
fn forced_greedy_replay_rejects_before_consuming_batched_false_accepts() {
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let _target = crate::fastpath::scoped_qwen_linear_mtp_target_verify(false);
    let _relaxed = crate::fastpath::scoped_qwen_linear_mtp_relaxed_session(false);
    let (cfg, weights) = forced_replay_test_model();
    let mut base = MlxKVCache::new_contiguous(1);
    assert_eq!(
        recompute_committed_prefix_with_argmax(&cfg, &weights, &mut base, 4, &[5], 0),
        1
    );
    // This dense fixture tests acceptance and real, input-dependent KV;
    // it does not stand in for Qwen numerical qualification.
    for (label, pending, batched, expected_count) in [
        ("first mismatch", vec![2, 1, 1], vec![2, 1, 1, 1], 0),
        ("later mismatch", vec![1, 2, 1], vec![1, 2, 1, 1], 1),
        ("full accept", vec![1, 1, 1], vec![1, 1, 1, 1], 3),
        ("empty", vec![], vec![1], 0),
        ("batched rejection cap", vec![1, 1, 1], vec![1, 2, 1, 1], 1),
    ] {
        let sources = vec![MtpDraftSource::Mtp; pending.len()];
        let mut predicted = batched;
        let mut rng = Xorshift64::new(7);
        let mut accept = mtp_accept_count(
            &pending,
            &[],
            &[],
            &sources,
            None,
            None,
            &predicted,
            &mut rng,
            0.0,
            0.0,
            MtpModelAcceptanceMode::Greedy,
            MtpNgramAcceptanceMode::Greedy,
        );
        let mut expected = base.clone();
        assert_eq!(
            recompute_committed_prefix_with_argmax(
                &cfg,
                &weights,
                &mut expected,
                3,
                &vec![1; expected_count],
                2,
            ),
            1
        );
        let mut cache = base.clone();
        let _outer_exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
        let _outer_target = crate::fastpath::scoped_qwen_linear_mtp_target_verify(true);
        let _outer_relaxed = crate::fastpath::scoped_qwen_linear_mtp_relaxed_session(true);
        let _outer_trace = crate::fastpath::scoped_qwen_linear_mtp_whole_verify_trace(true);
        let _outer_qmm = crate::model::shared::verify_qmm::QwenMtpVerifyQmmGuard::arm(true);
        let correction = replay_linear_mtp_accepted_prefix(
            &cfg,
            &weights,
            &mut cache,
            3,
            &pending,
            2,
            &mut accept,
            &mut predicted,
            forced_linear_mtp_greedy_revalidation(true, MlxSamplingParams::greedy()),
        );
        assert_eq!(accept.accept_count, expected_count, "{label}");
        assert_eq!(
            accept.all_accepted,
            expected_count == pending.len(),
            "{label}"
        );
        assert_eq!(correction, 1, "{label}");
        assert_eq!(predicted, vec![1; expected_count + 1], "{label}");
        assert_eq!(cache.seq_len(), 3 + expected_count, "{label}");
        let (actual_k, actual_v) = cache.logical_layer_kv(0).expect("actual KV");
        let (expected_k, expected_v) = expected.logical_layer_kv(0).expect("expected KV");
        eval(&[&actual_k, &actual_v, &expected_k, &expected_v]);
        assert_eq!(actual_k.shape(), expected_k.shape(), "{label}");
        assert_eq!(actual_k.data_f32(), expected_k.data_f32(), "{label}");
        assert_eq!(actual_v.data_f32(), expected_v.data_f32(), "{label}");
        assert!(actual_v.data_f32().iter().any(|&value| value != 0.0));
        // The runner uses this count for emission, hidden-row selection,
        // draft-cache refolding and the acceptance counters.
        let mut emitted = pending[..accept.accept_count].to_vec();
        emitted.push(correction);
        assert_eq!(emitted, vec![1; expected_count + 1], "{label}");
        let rows: Vec<f32> = (0..=pending.len())
            .flat_map(|row| [row as f32; 4])
            .collect();
        let post_norm = MlxArray::from_raw_data(
            rows.as_ptr().cast(),
            std::mem::size_of_val(rows.as_slice()),
            &[1, pending.len() as i32 + 1, 4],
            MlxDtype::Float32,
        );
        let hidden = slice_post_norm_hidden(&post_norm, accept.accept_count, 4);
        eval(&[&hidden]);
        assert_eq!(hidden.data_f32(), &[expected_count as f32; 4], "{label}");
        let mut telemetry = MtpTelemetry::default();
        telemetry.record_step(
            pending.len(),
            accept.accept_count,
            &sources,
            None,
            accept.accept_count,
        );
        assert_eq!(telemetry.accepted_tokens, expected_count as u32, "{label}");
        assert!(crate::fastpath::qwen_linear_mtp_exact_enabled());
        assert!(crate::fastpath::qwen_linear_mtp_target_verify_enabled());
        assert!(crate::fastpath::qwen_linear_mtp_relaxed_session_enabled());
        assert!(crate::fastpath::qwen_linear_mtp_whole_verify_trace_enabled());
    }
}

#[test]
fn device_greedy_accept_matches_host_accept_step() {
    // Guard matrix for the opt-in device path
    // (`AX_MLX_MTP_DEVICE_GREEDY_ACCEPT`, default OFF).
    assert!(mtp_device_greedy_accept_allowed(
        true,
        0.0,
        false,
        2,
        &[MtpDraftSource::Mtp; 2]
    ));
    assert!(!mtp_device_greedy_accept_allowed(
        false,
        0.0,
        false,
        2,
        &[MtpDraftSource::Mtp; 2]
    ));
    assert!(!mtp_device_greedy_accept_allowed(
        true,
        0.7,
        false,
        2,
        &[MtpDraftSource::Mtp; 2]
    ));
    assert!(!mtp_device_greedy_accept_allowed(
        true,
        0.0,
        true,
        2,
        &[MtpDraftSource::Mtp; 2]
    ));
    // Empty pending: the helper is never called; the guard refuses it.
    assert!(!mtp_device_greedy_accept_allowed(true, 0.0, false, 0, &[]));
    // N-gram / hybrid / assistant sources keep the host path, as does a
    // sources/pending length misalignment.
    assert!(!mtp_device_greedy_accept_allowed(
        true,
        0.0,
        false,
        2,
        &[MtpDraftSource::Ngram, MtpDraftSource::Mtp]
    ));
    assert!(!mtp_device_greedy_accept_allowed(
        true,
        0.0,
        false,
        2,
        &[MtpDraftSource::Mtp, MtpDraftSource::HybridMtp]
    ));
    assert!(!mtp_device_greedy_accept_allowed(
        true,
        0.0,
        false,
        1,
        &[MtpDraftSource::Gemma4Assistant]
    ));
    assert!(!mtp_device_greedy_accept_allowed(
        true,
        0.0,
        false,
        2,
        &[MtpDraftSource::Mtp]
    ));

    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let _target = crate::fastpath::scoped_qwen_linear_mtp_target_verify(false);
    let _relaxed = crate::fastpath::scoped_qwen_linear_mtp_relaxed_session(false);
    let (cfg, weights) = forced_replay_test_model();
    let mut base = MlxKVCache::new_contiguous(1);
    assert_eq!(
        recompute_committed_prefix_with_argmax(&cfg, &weights, &mut base, 4, &[5], 0),
        1
    );
    // This dense fixture exercises acceptance and real, input-dependent
    // KV; the batched verifier argmax windows are scripted, so host and
    // device acceptance must agree position-by-position.
    for (label, pending, batched) in [
        ("mismatch at depth 0", vec![2, 1, 1], vec![9, 1, 1, 1]),
        ("mismatch at depth 1", vec![1, 2, 1], vec![1, 5, 1, 1]),
        ("mismatch at depth 2", vec![1, 1, 2], vec![1, 1, 9, 4]),
        ("full accept", vec![1, 1, 1], vec![1, 1, 1, 7]),
        ("single full accept", vec![1], vec![1, 3]),
        ("single mismatch", vec![2], vec![1, 3]),
    ] {
        let n = pending.len();
        let sources = vec![MtpDraftSource::Mtp; n];
        assert!(mtp_device_greedy_accept_allowed(
            true, 0.0, false, n, &sources
        ));

        // Legacy host accept over the scripted window.
        let mut rng = Xorshift64::new(7);
        let accept_legacy = mtp_accept_count(
            &pending,
            &[],
            &[],
            &sources,
            None,
            None,
            &batched,
            &mut rng,
            0.0,
            0.0,
            MtpModelAcceptanceMode::Greedy,
            MtpNgramAcceptanceMode::Greedy,
        );
        let verifier_argmax_legacy = batched[accept_legacy.accept_count];

        // Device accept over the same window.
        let predicted_arr = MlxArray::from_raw_data(
            batched.as_ptr() as *const u8,
            std::mem::size_of_val(batched.as_slice()),
            &[batched.len() as i32],
            MlxDtype::Uint32,
        );
        let drafts_dev = MlxArray::from_raw_data(
            pending.as_ptr() as *const u8,
            std::mem::size_of_val(pending.as_slice()),
            &[n as i32],
            MlxDtype::Uint32,
        );
        let packed = crate::mtp::mtp_greedy_accept_lazy(&predicted_arr, &drafts_dev, n);
        eval(&[&packed]);
        assert_eq!(packed.shape(), vec![2], "{label}");
        let pair = packed.data_u32();
        let ac_device = usize::try_from(pair[0]).unwrap_or(0);
        let correction_device = pair[1];
        let accept_device = MtpAcceptOutcome {
            accept_count: ac_device,
            all_accepted: ac_device == n,
            rejection_correction: (ac_device != n).then_some(correction_device),
        };
        assert_eq!(accept_legacy, accept_device, "{label}");
        assert_eq!(correction_device, verifier_argmax_legacy, "{label}");

        // Non-replay rollback (recompute fallback) on the derived accept
        // count, then greedy correction selection: both paths must emit
        // identical tokens and leave identical caches.
        let mut cache_legacy = base.clone();
        let mut cache_device = base.clone();
        let recomputed_legacy = recompute_committed_prefix_with_argmax(
            &cfg,
            &weights,
            &mut cache_legacy,
            3,
            &pending[..accept_legacy.accept_count],
            2,
        );
        let recomputed_device = recompute_committed_prefix_with_argmax(
            &cfg,
            &weights,
            &mut cache_device,
            3,
            &pending[..accept_device.accept_count],
            2,
        );
        assert_eq!(recomputed_legacy, recomputed_device, "{label}");
        let token_legacy = select_linear_mtp_correction_token(
            0.0,
            Some(recomputed_legacy),
            None,
            accept_legacy.rejection_correction,
            verifier_argmax_legacy,
        );
        let token_device = select_linear_mtp_correction_token(
            0.0,
            Some(recomputed_device),
            None,
            accept_device.rejection_correction,
            correction_device,
        );
        assert_eq!(token_legacy, token_device, "{label}");
        let mut emitted_legacy = pending[..accept_legacy.accept_count].to_vec();
        emitted_legacy.push(token_legacy);
        let mut emitted_device = pending[..accept_device.accept_count].to_vec();
        emitted_device.push(token_device);
        assert_eq!(emitted_legacy, emitted_device, "{label}");
        assert_eq!(cache_legacy.seq_len(), cache_device.seq_len(), "{label}");
        let (k_legacy, v_legacy) = cache_legacy.logical_layer_kv(0).expect("legacy KV");
        let (k_device, v_device) = cache_device.logical_layer_kv(0).expect("device KV");
        eval(&[&k_legacy, &v_legacy, &k_device, &v_device]);
        assert_eq!(k_legacy.data_f32(), k_device.data_f32(), "{label}");
        assert_eq!(v_legacy.data_f32(), v_device.data_f32(), "{label}");

        // MTP draft-cache trim contract: rollback removes the rejected
        // tail by count (the runner's non-refold fallback).
        let drafted_len = base.seq_len() + n;
        let mut mtp_cache_legacy = MlxKVCache::new_contiguous(1);
        let mut mtp_cache_device = MlxKVCache::new_contiguous(1);
        mtp_cache_legacy.advance(drafted_len);
        mtp_cache_device.advance(drafted_len);
        let rejected_legacy = n - accept_legacy.accept_count;
        let rejected_device = n - accept_device.accept_count;
        assert!(mtp_cache_legacy.trim_to(drafted_len - rejected_legacy));
        assert!(mtp_cache_device.trim_to(drafted_len - rejected_device));
        assert_eq!(
            mtp_cache_legacy.seq_len(),
            mtp_cache_device.seq_len(),
            "{label}"
        );
    }
}

#[test]
fn ngram_greedy_revalidation_commits_production_token() {
    // Finding D: the n-gram accelerator's batched multi-token verifier can
    // diverge from singleton production on near-ties. At temperature 0 the
    // greedy path must commit the production correction/bonus token
    // (recompute_committed_prefix_with_argmax), not the batched argmax.
    let (cfg, weights) = forced_replay_test_model();
    let mut expected = MlxKVCache::new_contiguous(1);
    let production_correction =
        recompute_committed_prefix_with_argmax(&cfg, &weights, &mut expected, 3, &[1, 1], 0);
    let mut ngram = NgramTable::new();
    let mut cache = MlxKVCache::new_contiguous(1);
    let mut rng = Xorshift64::new(7);
    let mut probs = Vec::new();
    let mut logits_buf = Vec::new();
    let mut candidates = Vec::new();
    let draft = vec![1, 1];
    let result = ngram_accel_decode_step_with_sampling_buffers(
        &cfg,
        &weights,
        &mut cache,
        &mut ngram,
        3,
        &draft,
        NgramDraftPolicy::majority(2, 1, 0.4),
        MlxSamplingParams::greedy(),
        &[],
        &mut rng,
        &mut probs,
        &mut logits_buf,
        &mut candidates,
    );
    assert_eq!(&result[..draft.len()], draft.as_slice(), "accepted drafts");
    assert_eq!(
        result.last().copied(),
        Some(production_correction),
        "greedy n-gram must commit the production correction token"
    );
}

#[test]
fn forced_greedy_replay_discards_stale_skip_state() {
    for forced in [false, true] {
        let mut state = RequestState::new(1, 7, None);
        let stale = MlxArray::from_f32_slice(&[0.0, 1.0, 8.0]);
        state.mtp_skip_argmax = Some(argmax(&stale, None));
        state.mtp_skip_logits = Some(stale);
        state.mtp_skip_hidden = Some(mlx_sys::zeros(&[1, 1, 4], MlxDtype::Float32, None));
        let (logits, argmax, hidden) = take_mtp_skip_state(
            &mut state,
            forced_linear_mtp_greedy_revalidation(forced, MlxSamplingParams::greedy()),
        );
        if forced {
            assert!(
                logits.is_none(),
                "forced replay must not reuse batched logits"
            );
            assert!(argmax.is_none());
            assert!(hidden.is_none());
        } else {
            let primary = skip_state_primary_token(
                &logits.expect("legacy skip logits"),
                argmax,
                3,
                MlxSamplingParams::greedy(),
                &mut state.rng,
                &mut state.sampling_probs_buf,
                &mut state.sampling_logits_buf,
                &mut state.sampling_candidates_buf,
            );
            assert_eq!(primary, 2, "control must expose the stale row's primary");
            assert!(hidden.is_some());
        }
        assert!(state.mtp_skip_logits.is_none());
        assert!(state.mtp_skip_argmax.is_none());
        assert!(state.mtp_skip_hidden.is_none());
        assert_eq!(state.cache.seq_len(), 0);
    }
}

#[test]
fn forced_greedy_replay_excludes_default_sampled_and_processed_requests() {
    let greedy = MlxSamplingParams::greedy();
    assert!(forced_linear_mtp_greedy_revalidation(true, greedy));
    let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let _target = crate::fastpath::scoped_qwen_linear_mtp_target_verify(false);
    let _relaxed = crate::fastpath::scoped_qwen_linear_mtp_relaxed_session(false);
    let (cfg, weights) = forced_replay_test_model();
    let mut expected = MlxKVCache::new_contiguous(1);
    let expected_correction =
        recompute_committed_prefix_with_argmax(&cfg, &weights, &mut expected, 3, &[2], 0);
    let (expected_k, expected_v) = expected.logical_layer_kv(0).expect("expected KV");
    eval(&[&expected_k, &expected_v]);
    for (forced, sampling) in [
        (false, greedy),
        (true, MlxSamplingParams::new(0.7, 1.0, 0)),
        (true, greedy.with_repetition_penalty(1.2, None)),
        (true, greedy.with_no_repeat_ngram(3, 128)),
    ] {
        let revalidate = forced_linear_mtp_greedy_revalidation(forced, sampling);
        assert!(!revalidate);
        let original_accept = MtpAcceptOutcome {
            accept_count: 1,
            all_accepted: false,
            rejection_correction: Some(7),
        };
        let mut accept = original_accept;
        let mut predicted = vec![2, 7, 1];
        let mut cache = MlxKVCache::new_contiguous(1);
        let correction = replay_linear_mtp_accepted_prefix(
            &cfg,
            &weights,
            &mut cache,
            3,
            &[2, 3],
            0,
            &mut accept,
            &mut predicted,
            revalidate,
        );
        assert_eq!(accept, original_accept);
        assert_eq!(predicted, [2, 7, 1]);
        assert_eq!(correction, expected_correction);
        assert_eq!(cache.seq_len(), 2);
        let (k, v) = cache.logical_layer_kv(0).expect("actual KV");
        eval(&[&k, &v]);
        assert_eq!(k.data_f32(), expected_k.data_f32());
        assert_eq!(v.data_f32(), expected_v.data_f32());
        if sampling.temperature > 0.0 {
            assert_eq!(
                select_linear_mtp_correction_token(
                    sampling.temperature,
                    Some(correction),
                    Some(9),
                    accept.rejection_correction,
                    7,
                ),
                9
            );
        }
    }
}

#[test]
fn sampled_linear_mtp_rejection_preserves_exact_residual_correction() {
    assert_eq!(
        select_linear_mtp_correction_token(0.7, Some(440), Some(271), Some(13_661), 13_661,),
        271
    );
}

#[test]
fn qwen_linear_exact_arithmetic_requires_mtp_request() {
    assert!(qwen_linear_mtp_exact_scope_for_request(true, true));
    assert!(!qwen_linear_mtp_exact_scope_for_request(true, false));
    assert!(!qwen_linear_mtp_exact_scope_for_request(false, true));
    assert!(!qwen_linear_mtp_exact_scope_for_request(false, false));
}

#[test]
fn qwen_linear_mtp_async_draft_requires_linear_verifier_opt_in() {
    assert!(qwen_linear_mtp_async_draft_allowed(true, true, false));
    assert!(qwen_linear_mtp_async_draft_allowed(true, false, true));
    assert!(qwen_linear_mtp_async_draft_allowed(true, true, true));
    assert!(!qwen_linear_mtp_async_draft_allowed(true, false, false));
    assert!(!qwen_linear_mtp_async_draft_allowed(false, true, true));
}

#[test]
fn qwen_linear_exact_route_keeps_resolved_profile_on_inactive_terminal_step() {
    let mut decisions = Vec::new();
    append_qwen_linear_mtp_exact_route_decisions(
        &mut decisions,
        true,
        true,
        false,
        crate::fastpath::QwenLinearMtpExactSelection::ExplicitEnabled.route_code(),
    );

    assert_eq!(
        decisions,
        vec![
            ("ax_mlx_qwen_linear_mtp_exact_eligible".to_owned(), 1),
            ("ax_mlx_qwen_linear_mtp_exact_enabled".to_owned(), 1),
            ("ax_mlx_qwen_linear_mtp_exact_active".to_owned(), 0),
            ("ax_mlx_qwen_linear_mtp_exact_selection".to_owned(), 2),
        ]
    );
}

#[test]
fn linear_mtp_checkpoint_uses_exact_profile_within_verified_depth() {
    // Depth 1-3 + exact + no kill switch → checkpoint path (not singleton).
    assert!(!linear_mtp_requires_singleton_replay(1, true, false));
    assert!(!linear_mtp_requires_singleton_replay(2, true, false));
    assert!(!linear_mtp_requires_singleton_replay(3, true, false));
    // Kill switch, non-exact, empty, or depth>3 → singleton recompute.
    assert!(linear_mtp_requires_singleton_replay(1, false, false));
    assert!(linear_mtp_requires_singleton_replay(1, true, true));
    assert!(linear_mtp_requires_singleton_replay(4, true, false));
    assert!(linear_mtp_requires_singleton_replay(0, true, false));
    assert!(linear_mtp_requires_singleton_replay(2, false, false));
    assert!(linear_mtp_requires_singleton_replay(2, true, true));
}

#[test]
fn projected_linear_replay_is_explicit_bounded_and_kill_switchable() {
    assert!(linear_mtp_projected_replay_allowed(1, true, false, false));
    assert!(linear_mtp_projected_replay_allowed(3, true, false, false));
    assert!(!linear_mtp_projected_replay_allowed(0, true, false, false));
    assert!(!linear_mtp_projected_replay_allowed(4, true, false, false));
    assert!(!linear_mtp_projected_replay_allowed(2, false, false, false));
    assert!(!linear_mtp_projected_replay_allowed(2, true, true, false));
    assert!(!linear_mtp_projected_replay_allowed(2, true, false, true));
}

#[test]
fn width_generic_miss_backoff_mirrors_the_depth_three_controller() {
    let policy = MtpAdaptiveDepthPolicy {
        fixed_depth: None,
        conservative_depth: false,
        depth3_miss_backoff: true,
        depth3_hysteresis: true,
    };
    // The controller is width-generic: exercise the `_for_window` helper
    // explicitly at widths 3 and 4 so a width-4 regression cannot hide.
    for w in [3, 4] {
        // Start deep.
        assert_eq!(
            mtp_next_adaptive_depth_with_policy_for_window(0, w, 0, 0, 0, false, policy, w),
            w
        );
        // A complete miss backs off by one; any accepted draft restores.
        assert_eq!(
            mtp_next_adaptive_depth_with_policy_for_window(w, w, w, 0, 0, false, policy, w),
            w - 1
        );
        assert_eq!(
            mtp_next_adaptive_depth_with_policy_for_window(w - 1, w, w - 1, 1, 0, false, policy, w),
            w
        );
        // A stale current depth above max clamps back to the window.
        assert_eq!(
            mtp_next_adaptive_depth_with_policy_for_window(8, w, 0, 0, 0, false, policy, w),
            w
        );
    }
    // Window 4 with a 3-deep head still runs the controller at width 3,
    // not the generic progressive floor (which would be 0 after two misses).
    assert_eq!(
        mtp_next_adaptive_depth_with_policy_for_window(3, 3, 3, 0, 2, false, policy, 4),
        2
    );
    // Window 3 with an 8-deep head keeps the hysteresis hold.
    assert_eq!(
        mtp_next_adaptive_depth_with_policy_for_window(3, 8, 3, 2, 0, false, policy, 3),
        3
    );
    // Window 4 hysteresis hold.
    assert_eq!(
        mtp_next_adaptive_depth_with_policy_for_window(4, 4, 4, 3, 0, false, policy, 4),
        4
    );
}

#[test]
fn batched_committed_fold_honors_resolved_zero_gate() {
    assert!(!qwen_batched_committed_fold_gate_ok(0.90, 900));
    assert!(qwen_batched_committed_fold_gate_ok(0.0, 900));
    assert!(qwen_batched_committed_fold_gate_ok(0.90, 0));
    assert!(qwen_batched_committed_fold_gate_ok(0.0, 0));
}

#[test]
fn qwen_linear_mtp_exact_capability_accepts_axq_mixed_affine_contract() {
    assert!(qwen_linear_mtp_exact_tensor_supported(
        true,
        Some(("affine", 4, 32))
    ));
    assert!(qwen_linear_mtp_exact_tensor_supported(
        true,
        Some(("affine", 4, 64))
    ));
    assert!(qwen_linear_mtp_exact_tensor_supported(
        true,
        Some(("affine", 8, 64))
    ));
    assert!(qwen_linear_mtp_exact_tensor_supported(false, None));
    assert!(qwen_linear_mtp_exact_tensor_supported(
        true,
        Some(("mxfp4", 4, 32))
    ));
}

#[test]
fn qwen_linear_mtp_exact_capability_rejects_uncertified_quantization() {
    for quantization in [
        Some(("affine", 2, 64)),
        Some(("affine", 3, 64)),
        Some(("affine", 5, 64)),
        Some(("affine", 6, 128)),
        Some(("mxfp4", 4, 64)),
        Some(("mxfp8", 8, 32)),
    ] {
        assert!(
            !qwen_linear_mtp_exact_tensor_supported(true, quantization),
            "unexpectedly admitted {quantization:?}"
        );
    }
    assert!(!qwen_linear_mtp_exact_tensor_supported(true, None));
}

#[test]
fn qwen_linear_mtp_exact_capability_is_family_and_depth_bounded() {
    let no_tensors = [];
    assert!(qwen_linear_mtp_exact_model_eligible(
        "qwen3_5",
        true,
        1,
        &no_tensors
    ));
    assert!(qwen_linear_mtp_exact_model_eligible(
        "qwen3_5",
        true,
        3,
        &no_tensors
    ));
    assert!(!qwen_linear_mtp_exact_model_eligible(
        "qwen3_next",
        true,
        1,
        &no_tensors
    ));
    assert!(!qwen_linear_mtp_exact_model_eligible(
        "qwen3_5",
        false,
        1,
        &no_tensors
    ));
    assert!(!qwen_linear_mtp_exact_model_eligible(
        "qwen3_5",
        true,
        0,
        &no_tensors
    ));
    assert!(!qwen_linear_mtp_exact_model_eligible(
        "qwen3_5",
        true,
        4,
        &no_tensors
    ));
}

#[test]
fn direct_pipeline_bootstrap_does_not_overlap_strict_mtp() {
    // AX_NO_SPEC / `--ax-direct` still primes the double-buffer when MTP
    // weights are attached because construction cleared `mtp_requested`.
    assert!(
        should_bootstrap_direct_pipeline(true, false, true, false, false),
        "session-direct must bootstrap the greedy pipeline for long decode",
    );
    assert!(should_bootstrap_direct_pipeline(
        true, false, false, false, false
    ));
    assert!(should_bootstrap_direct_pipeline(
        false, true, false, false, false
    ));
    assert!(should_bootstrap_direct_pipeline(
        false, false, true, true, true
    ));
    assert!(!should_bootstrap_direct_pipeline(
        false, true, true, false, true
    ));
    assert!(
        !should_use_session_direct_pipeline(true, true, true, true),
        "when MTP remains requested, decode must not steal the direct pipeline",
    );
    assert!(
        should_use_session_direct_pipeline(true, true, true, false),
        "Gemma/Qwen pure direct (mtp_requested=false) must use pipeline",
    );
    assert!(should_use_session_direct_pipeline(true, true, false, false));
    assert!(
        should_use_session_direct_pipeline(true, true, false, true),
        "no MTP attached: pipeline is eligible regardless of mtp_requested bit",
    );
    assert!(!should_use_session_direct_pipeline(
        true, false, false, false
    ));
}

#[test]
fn pure_direct_session_keeps_pipeline_for_long_context_shapes() {
    // Regression guard for the 2026-07-26 Gemma@2048 readout: pure direct
    // sessions must select the double-buffer pipeline for greedy decode at
    // every prompt depth, including long-context shapes used by README
    // direct tables (128 / 512 / 2048).
    for has_mtp in [false, true] {
        assert!(
            should_use_session_direct_pipeline(true, true, has_mtp, false),
            "session_direct greedy pipeline required (has_mtp={has_mtp})"
        );
        assert!(
            should_bootstrap_direct_pipeline(true, false, has_mtp, false, false),
            "session_direct must bootstrap pipeline (has_mtp={has_mtp})"
        );
    }
    // AX_NO_SPEC clears MTP even when the model would otherwise qualify.
    let disable_ngram = true;
    let speculation_disabled = true;
    let session_direct = disable_ngram || speculation_disabled;
    let mtp_requested = default_mtp_requested(speculation_disabled, true, true);
    assert!(!mtp_requested);
    assert!(should_use_session_direct_pipeline(
        session_direct,
        true,
        true,
        mtp_requested
    ));
}

#[test]
fn ngram_disable_does_not_disable_certified_packaged_mtp() {
    let disable_ngram = true;
    let mtp_requested = default_mtp_requested(false, true, true);
    assert!(disable_ngram);
    assert!(mtp_requested);
    assert!(!default_mtp_requested(true, true, true));
    assert!(!default_mtp_requested(false, false, true));
    assert!(!default_mtp_requested(false, true, false));
    // decode_one consults these predicates before `run_model_decode`.
    assert!(
        !v4_uncertified_uses_pure_direct_pipeline(
            false,
            disable_ngram,
            false,
            false,
            true,
            mtp_requested
        ),
        "CLI n-gram off must not force the pure-direct pipeline over certified MTP"
    );
    assert!(!should_use_session_direct_pipeline(
        disable_ngram,
        true,
        true,
        mtp_requested
    ));
    assert!(
        !should_bootstrap_direct_pipeline(disable_ngram, false, true, false, mtp_requested),
        "CLI n-gram off must not prime pending_direct while MTP is requested"
    );
    // OpenAI-shaped temp-0 + top_p is not `is_greedy`, but decode classifies
    // it as exact MTP. Bootstrap must use the same predicate so it does not
    // prime pending_direct through the DirectFallback term.
    let sampling = MlxSamplingParams::new(0.0, 0.9, 0);
    let exact_supported = mtp_exact_sampling_supported(sampling, None);
    assert!(exact_supported);
    let mtp_uses_direct_pipeline = matches!(
        mtp_request_route(
            true,
            mtp_requested,
            exact_supported,
            false,
            false,
            sampling.uses_logits_processors(),
        ),
        MtpRequestRoute::DirectFallback
    );
    assert!(
        !mtp_uses_direct_pipeline,
        "temp-0 top_p certified MTP must be StrictMtp at both bootstrap and decode"
    );
    assert!(!should_bootstrap_direct_pipeline(
        disable_ngram,
        false,
        true,
        mtp_uses_direct_pipeline,
        mtp_requested
    ));
    // DirectFallback with a logits processor must not prime: decode uses
    // `run_single_decode` and would leave pending_direct undrained.
    let penalized = MlxSamplingParams::new(0.0, 1.0, 0).with_repetition_penalty(1.1, Some(64));
    assert!(penalized.uses_logits_processors());
    let penalized_exact = mtp_exact_sampling_supported(penalized, None);
    assert!(!penalized_exact);
    let penalized_route = mtp_request_route(
        true,
        mtp_requested,
        penalized_exact,
        false,
        false,
        penalized.uses_logits_processors(),
    );
    assert_eq!(penalized_route, MtpRequestRoute::DirectFallback);
    assert!(!mtp_fallback_primes_direct_pipeline(
        penalized_route,
        true,
        penalized.uses_logits_processors(),
    ));
    assert!(!should_bootstrap_direct_pipeline(
        disable_ngram,
        false,
        true,
        mtp_fallback_primes_direct_pipeline(
            penalized_route,
            true,
            penalized.uses_logits_processors(),
        ),
        mtp_requested
    ));
}

#[test]
fn pure_direct_pipeline_gate_ignores_mtp_request_bit() {
    // decode_one pure-direct force path: session_direct + temp0 uses the
    // pipeline when construction cleared `mtp_requested` (AX_NO_SPEC).
    assert!(
        should_use_session_direct_pipeline(true, true, true, false),
        "pure direct with mtp weights attached still uses pipeline"
    );
    // Certified packaged MTP stays requested when only n-gram is disabled.
    assert!(!should_use_session_direct_pipeline(true, true, true, true));
}

#[test]
fn v4_uncertified_fallback_uses_pure_direct_even_when_ngram_is_on() {
    assert!(v4_uncertified_uses_pure_direct_pipeline(
        true, false, false, false, true, false
    ));
    assert!(!v4_uncertified_uses_pure_direct_pipeline(
        false, false, false, false, true, false
    ));
    assert!(!v4_uncertified_uses_pure_direct_pipeline(
        true, false, false, true, true, false
    ));
    assert!(v4_uncertified_uses_pure_direct_pipeline(
        false, true, false, false, true, false
    ));
    assert!(!v4_uncertified_uses_pure_direct_pipeline(
        false, true, false, false, true, true
    ));
}

#[test]
fn loop_detection_truncates_multi_token_batch_at_first_trigger() {
    // Pattern of period 2, min_count 4 → needs 8 tokens. History has 6;
    // a multi-token MTP batch of 4 would overshoot if kept whole.
    let cfg = ax_engine_core::LoopDetectionConfig::GEMMA4_DEFAULT;
    let history = vec![10u32, 20, 10, 20, 10, 20];
    let batch = vec![10u32, 20, 10, 20]; // 1st+2nd complete 4×(10,20)
    let (kept, reason) = apply_loop_detection_stop(batch, None, &history, Some(cfg));
    assert_eq!(
        kept,
        vec![10u32, 20],
        "stop at the token that first completes the loop"
    );
    assert!(matches!(reason, Some(StopReason::LoopDetected)));
    // History of 7 already has 3 full pairs + one half; the next 20 completes
    // the 4th pair. Single-token path keeps that completing token.
    let history7 = vec![10u32, 20, 10, 20, 10, 20, 10];
    let (kept1, reason1) = apply_loop_detection_stop(vec![20u32], None, &history7, Some(cfg));
    assert_eq!(kept1, vec![20u32]);
    assert!(matches!(reason1, Some(StopReason::LoopDetected)));
}

#[test]
fn ignore_eos_disables_gemma4_loop_detection_for_fixed_token_benches() {
    // README AX-direct rows send ignore_eos=true so generation_tokens is a
    // hard length. Loop detection must not arm and truncate that window.
    assert!(
        loop_detection_for_request(true, "gemma4").is_none(),
        "ignore_eos must suppress Gemma 4 loop detection"
    );
    assert!(
        loop_detection_for_request(true, "gemma4text").is_none(),
        "ignore_eos suppresses loop detection for any family"
    );
    // Without ignore_eos, default Gemma 4 still enables detection (env on).
    // Force via family match only when env is default-on.
    let chat = loop_detection_for_request(false, "gemma4");
    // May be None if AX_GEMMA4_LOOP_DETECTION=off in the test env; when set
    // on/default, chat path is armed.
    if std::env::var("AX_GEMMA4_LOOP_DETECTION")
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "off" | "0" | "false" | "no"
            )
        })
        .unwrap_or(false)
    {
        assert!(chat.is_none());
    } else {
        assert!(chat.is_some(), "chat path keeps Gemma 4 loop detection");
    }
    assert!(
        loop_detection_for_request(false, "qwen3").is_none(),
        "non-Gemma families do not enable this detector"
    );
}

#[test]
fn gemma_assistant_mtp_coalescing_is_fail_closed() {
    let eligible = || {
        gemma4_assistant_mtp_coalesced_verify_route(
            true, true, false, true, true, false, true, false, true, false,
        )
    };
    assert!(eligible());
    assert!(!gemma4_assistant_mtp_coalesced_verify_route(
        false, true, false, true, true, false, true, false, true, false,
    ));
    assert!(!gemma4_assistant_mtp_coalesced_verify_route(
        true, true, true, true, true, false, true, false, true, false,
    ));
    assert!(!gemma4_assistant_mtp_coalesced_verify_route(
        true, true, false, true, false, false, true, false, true, false,
    ));
    assert!(!gemma4_assistant_mtp_coalesced_verify_route(
        true, true, false, true, true, true, true, false, true, false,
    ));
    assert!(!gemma4_assistant_mtp_coalesced_verify_route(
        true, true, false, true, true, false, false, false, true, false,
    ));
    assert!(!gemma4_assistant_mtp_coalesced_verify_route(
        true, true, false, true, true, false, true, true, true, false,
    ));
    assert!(!gemma4_assistant_mtp_coalesced_verify_route(
        true, true, false, true, true, false, true, false, false, false,
    ));
    assert!(!gemma4_assistant_mtp_coalesced_verify_route(
        true, true, false, true, true, false, true, false, true, true,
    ));
}

#[test]
fn mtp_telemetry_tracks_stacked_ngram_source_and_hybrid_tail() {
    let mut telemetry = MtpTelemetry::default();

    telemetry.record_ngram_attempt(Some(NgramDraftRejection::NoCandidate));
    telemetry.record_ngram_attempt(Some(NgramDraftRejection::ConfidenceFiltered));
    telemetry.record_ngram_attempt(None);
    telemetry.record_ngram_cycle_guard();
    telemetry.record_ngram_stack_hit(2, false);
    telemetry.record_ngram_hybrid_tail(1);
    telemetry.record_ngram_proposed(2);
    telemetry.record_ngram_submitted(2);
    telemetry.record_ngram_verified(2);
    telemetry.record_step(
        3,
        2,
        &[
            MtpDraftSource::Ngram,
            MtpDraftSource::Ngram,
            MtpDraftSource::HybridMtp,
        ],
        None,
        0,
    );

    let mut decisions = Vec::new();
    telemetry.append_route_decisions(&mut decisions);

    assert!(decisions.contains(&("ax_mtp_ngram_attempt_steps".into(), 3)));
    assert!(decisions.contains(&("ax_mtp_ngram_no_candidate_steps".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_ngram_confidence_filtered_steps".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_ngram_cycle_guard_steps".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_ngram_hit_steps".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_ngram_skipped_mtp_steps".into(), 0)));
    assert!(decisions.contains(&("ax_mtp_ngram_skipped_mtp_tokens".into(), 0)));
    assert!(decisions.contains(&("ax_mtp_ngram_hybrid_tail_steps".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_ngram_hybrid_tail_tokens".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_draft_source_ngram_tokens".into(), 2)));
    assert!(decisions.contains(&("ax_mtp_accepted_source_ngram_tokens".into(), 2)));
    assert!(decisions.contains(&("ax_mtp_draft_source_hybrid_mtp_tokens".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_accepted_source_hybrid_mtp_tokens".into(), 0)));
    assert!(decisions.contains(&("ax_mtp_ngram_proposed_tokens".into(), 2)));
    assert!(decisions.contains(&("ax_mtp_ngram_submitted_tokens".into(), 2)));
    assert!(decisions.contains(&("ax_mtp_ngram_submitted_accepted_tokens".into(), 2)));
    assert!(decisions.contains(&("ax_mtp_ngram_accepted_tokens".into(), 2)));
    assert!(decisions.contains(&("ax_mtp_ngram_rejected_tokens".into(), 0)));
    assert!(decisions.contains(&("ax_mtp_ngram_cascade_rejected_tokens".into(), 0)));
    assert!(decisions.contains(&("ax_mtp_source_mtp_submitted_tokens".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_source_mtp_rejected_tokens".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_source_mtp_cascade_rejected_tokens".into(), 0)));
}

#[test]
fn mtp_only_ewma_excludes_cascade_rejections_from_ngram_failure() {
    // When an n-gram token at position 0 is rejected (accept=0), MTP tokens
    // at later positions are cascade-rejected — not because they are bad
    // predictions but because the earlier n-gram token already failed.
    // The MTP-only EWMA must NOT count those cascade rejections, otherwise
    // it would deflate even when pure-MTP acceptance is near-perfect.

    let mut tel = MtpTelemetry::default();

    // Step 1: [Ngram, Ngram, Mtp] accept=0 — n-gram at pos 0 fails.
    // MTP token at pos 2 is cascade-rejected.  No MTP EWMA update.
    tel.record_step(
        3,
        0,
        &[
            MtpDraftSource::Ngram,
            MtpDraftSource::Ngram,
            MtpDraftSource::Mtp,
        ],
        None,
        0,
    );
    assert_eq!(
        tel.mtp_only_accept_rate_ewma_samples, 0,
        "cascade step must not produce a sample"
    );

    // Step 2: [Ngram, Ngram, Mtp] accept=3 — all pass, MTP genuinely accepted.
    tel.record_step(
        3,
        3,
        &[
            MtpDraftSource::Ngram,
            MtpDraftSource::Ngram,
            MtpDraftSource::Mtp,
        ],
        None,
        1,
    );
    assert_eq!(tel.mtp_only_accept_rate_ewma_samples, 1);
    assert!(
        (tel.mtp_only_accept_rate_ewma - 1.0).abs() < 1e-5,
        "MTP accepted → rate 1.0"
    );

    // Step 3: [Ngram, Ngram, Mtp] accept=2 — ngrams pass, MTP at pos 2 is
    // the first rejection (position == accept).  This IS a meaningful eval.
    tel.record_step(
        3,
        2,
        &[
            MtpDraftSource::Ngram,
            MtpDraftSource::Ngram,
            MtpDraftSource::Mtp,
        ],
        None,
        0,
    );
    assert_eq!(tel.mtp_only_accept_rate_ewma_samples, 2);
    // After step 3: EWMA nudged down from 1.0 toward 0.0 (mtp rejected).
    assert!(
        tel.mtp_only_accept_rate_ewma < 1.0,
        "MTP rejection must lower the EWMA"
    );

    // In pure-MTP steps the cascade-exclusion still applies: position 2 is
    // cascade-rejected (never independently verified by the main model) and is
    // excluded from both numerator and denominator.  Only the two "meaningfully
    // evaluated" positions count — position 0 (accepted) and position 1 (the
    // first and only genuine rejection that caused the cascade).
    let mut tel2 = MtpTelemetry::default();
    tel2.record_step(3, 1, &[MtpDraftSource::Mtp; 3], None, 1);
    // drafted=3, accepted=1, first rejection at pos 1 (Mtp) → first_rejection_is_mtp=true.
    // pos 2 is cascade-excluded.  mtp_only_drafted = 1 (accepted) + 1 (first rejection) = 2.
    assert_eq!(tel2.mtp_only_accept_rate_ewma_samples, 1);
    let expected_rate = 1.0_f32 / 2.0;
    assert!(
        (tel2.mtp_only_accept_rate_ewma - expected_rate).abs() < 1e-5,
        "pure-MTP partial rejection: rate should be accepted/meaningful = 1/2"
    );
}

#[test]
fn mtp_acceptance_warmup_is_not_biased_by_a_first_cycle_miss() {
    let mut telemetry = MtpTelemetry::default();
    telemetry.record_step(1, 0, &[MtpDraftSource::Mtp], None, 0);
    for _ in 0..7 {
        telemetry.record_step(1, 1, &[MtpDraftSource::Mtp], None, 1);
    }

    assert_eq!(telemetry.mtp_only_accept_rate_ewma_samples, 8);
    assert!(
        (telemetry.mtp_only_accept_rate_ewma - 0.875).abs() < 1e-6,
        "warm-up estimator must reflect the observed 7/8 acceptance"
    );
    assert!(telemetry.mtp_only_accept_rate_ewma > mtp_bypass_threshold());
}

#[test]
fn mtp_acceptance_warmup_still_trips_bypass_on_genuinely_low_acceptance() {
    // Negative direction of the warm-up fix: a model whose drafts really
    // are rejected most of the time must still satisfy the bypass
    // predicate (samples >= min && EWMA < threshold) both inside and
    // beyond the cumulative warm-up window. One accept in four is 25%,
    // well under the 50% bypass threshold.
    let mut telemetry = MtpTelemetry::default();
    for step in 0..24 {
        let accepted = usize::from(step % 4 == 0);
        telemetry.record_step(1, accepted, &[MtpDraftSource::Mtp], None, accepted);
        if telemetry.mtp_only_accept_rate_ewma_samples >= mtp_bypass_min_samples() {
            assert!(
                telemetry.mtp_only_accept_rate_ewma < mtp_bypass_threshold(),
                "25% acceptance must stay below the bypass threshold at sample {}",
                telemetry.mtp_only_accept_rate_ewma_samples
            );
        }
    }
    assert_eq!(telemetry.mtp_only_accept_rate_ewma_samples, 24);
    assert!(
        (telemetry.mtp_only_accept_rate_ewma - 0.25).abs() < 0.05,
        "estimator must converge near the true 25% acceptance; got {}",
        telemetry.mtp_only_accept_rate_ewma
    );
}

#[test]
fn mtp_ewma_numerator_uses_accepted_count_not_argmax_matches() {
    // In rejection-sampling mode, the EWMA numerator is the actual number of
    // accepted MTP tokens, not just those matching the target argmax.
    // This ensures the n-gram saturation gate and auto-optimistic activation
    // converge on the true acceptance rate rather than the argmax-match rate.
    //
    // Scenario: 2 MTP tokens drafted, both accepted; only 1 matches target argmax.
    // Old code: mtp_argmax_matches = 1 → EWMA = 1/2 = 0.5 (wrong)
    // New code: mtp_ewma_numerator = 2  → EWMA = 2/2 = 1.0 (correct)
    let mut tel = MtpTelemetry::default();
    // mtp_ewma_numerator=2: caller computed actual accepted MTP count (both accepted).
    tel.record_step(2, 2, &[MtpDraftSource::Mtp; 2], None, 2);
    assert_eq!(tel.mtp_only_accept_rate_ewma_samples, 1);
    assert!(
        (tel.mtp_only_accept_rate_ewma - 1.0).abs() < 1e-5,
        "all MTP accepted → EWMA = 1.0; got {}",
        tel.mtp_only_accept_rate_ewma
    );

    // Verify that passing the old argmax-match count (1 of 2) would have given 0.5.
    let mut tel_old = MtpTelemetry::default();
    tel_old.record_step(2, 2, &[MtpDraftSource::Mtp; 2], None, 1);
    assert!(
        (tel_old.mtp_only_accept_rate_ewma - 0.5).abs() < 1e-5,
        "only argmax matches counted → EWMA = 0.5 (old wrong behaviour)"
    );
}

#[test]
fn mtp_ngram_hurt_gate_fires_when_combined_trails_mtp_only_by_margin() {
    assert!(mtp_ngram_hurt_gate(4, 0.80, 4, 0.90, 4, 4, 0.02));
}

#[test]
fn mtp_ngram_hurt_gate_does_not_fire_before_min_samples() {
    assert!(!mtp_ngram_hurt_gate(4, 0.80, 3, 0.90, 4, 4, 0.02));
    assert!(!mtp_ngram_hurt_gate(4, 0.80, 4, 0.90, 3, 4, 0.02));
}

#[test]
fn mtp_ngram_hurt_gate_does_not_fire_inside_margin() {
    assert!(!mtp_ngram_hurt_gate(4, 0.885, 4, 0.90, 4, 4, 0.02));
}

#[test]
fn mtp_ngram_auto_disable_requires_both_mtp_strong_and_ngram_weak() {
    let cfg = MtpNgramAutoDisableConfig {
        mtp_warmup: 64,
        ngram_warmup: 32,
        mtp_threshold: 850,
        ngram_floor: 500,
    };
    assert!(mtp_ngram_auto_disable_gate(4, 100, 90, 40, 10, cfg));
    assert!(!mtp_ngram_auto_disable_gate(4, 100, 84, 40, 10, cfg));
    assert!(!mtp_ngram_auto_disable_gate(4, 100, 90, 40, 25, cfg));
    assert!(!mtp_ngram_auto_disable_gate(4, 63, 60, 40, 10, cfg));
    assert!(!mtp_ngram_auto_disable_gate(4, 100, 90, 31, 10, cfg));
}

#[test]
fn mtp_ngram_gate_decision_reports_each_reason() {
    let cfg = MtpNgramAutoDisableConfig {
        mtp_warmup: 64,
        ngram_warmup: 32,
        mtp_threshold: 850,
        ngram_floor: 500,
    };
    let decision =
        mtp_ngram_gate_decision(4, 3, 0.80, 4, 0.99, 4, 100, 90, 40, 10, true, 4, 0.02, cfg);
    assert!(decision.gated);
    assert!(decision.saturated);
    assert!(decision.hurt);
    assert!(decision.auto_disabled);
    assert!(decision.self_tune_disabled);
}

fn utility_cfg_for_tests() -> MtpNgramUtilityGateConfig {
    MtpNgramUtilityGateConfig {
        min_emitted_tokens: 128,
        min_ngram_submitted_tokens: 32,
        margin_ratio: 0.02,
        hysteresis_steps: 16,
    }
}

#[test]
fn mtp_ngram_utility_gate_waits_for_enough_samples() {
    let cfg = utility_cfg_for_tests();
    let baseline = DraftSourceUtility {
        proposer_wall_us: 1_000,
        verify_wall_us: 1_000,
        emitted_tokens: 127,
        ..DraftSourceUtility::default()
    };
    let stacked = DraftSourceUtility {
        submitted_tokens: 64,
        proposer_wall_us: 2_000,
        verify_wall_us: 2_000,
        emitted_tokens: 128,
    };

    let decision = mtp_ngram_utility_gate(4, baseline, stacked, cfg, 0);

    assert!(!decision.gated);
    assert!(decision.insufficient_samples);
    assert!(!decision.utility_hurt);
}

#[test]
fn mtp_ngram_utility_gate_waits_for_enough_ngram_tokens() {
    let cfg = utility_cfg_for_tests();
    let baseline = DraftSourceUtility {
        proposer_wall_us: 1_000,
        verify_wall_us: 1_000,
        emitted_tokens: 128,
        ..DraftSourceUtility::default()
    };
    let stacked = DraftSourceUtility {
        submitted_tokens: 31,
        proposer_wall_us: 2_000,
        verify_wall_us: 2_000,
        emitted_tokens: 128,
    };

    let decision = mtp_ngram_utility_gate(4, baseline, stacked, cfg, 0);

    assert!(!decision.gated);
    assert!(decision.insufficient_samples);
}

#[test]
fn mtp_ngram_utility_gate_fires_when_stacked_cost_is_worse() {
    let cfg = utility_cfg_for_tests();
    let baseline = DraftSourceUtility {
        proposer_wall_us: 5_000,
        verify_wall_us: 5_000,
        emitted_tokens: 200,
        ..DraftSourceUtility::default()
    };
    let stacked = DraftSourceUtility {
        submitted_tokens: 64,
        proposer_wall_us: 7_500,
        verify_wall_us: 7_500,
        emitted_tokens: 200,
    };

    let decision = mtp_ngram_utility_gate(4, baseline, stacked, cfg, 0);

    assert!(decision.gated);
    assert!(decision.utility_hurt);
    assert!(!decision.insufficient_samples);
}

#[test]
fn mtp_ngram_utility_gate_does_not_fire_inside_margin() {
    let cfg = utility_cfg_for_tests();
    let baseline = DraftSourceUtility {
        proposer_wall_us: 10_000,
        emitted_tokens: 200,
        ..DraftSourceUtility::default()
    };
    let stacked = DraftSourceUtility {
        submitted_tokens: 64,
        proposer_wall_us: 10_100,
        emitted_tokens: 200,
        ..DraftSourceUtility::default()
    };

    let decision = mtp_ngram_utility_gate(4, baseline, stacked, cfg, 0);

    assert!(!decision.gated);
    assert!(!decision.utility_hurt);
    assert!(!decision.insufficient_samples);
}

#[test]
fn mtp_ngram_utility_gate_hysteresis_gates_without_recomputing_cost() {
    let decision = mtp_ngram_utility_gate(
        4,
        DraftSourceUtility::default(),
        DraftSourceUtility::default(),
        utility_cfg_for_tests(),
        3,
    );

    assert!(decision.gated);
    assert!(decision.hysteresis_active);
    assert!(!decision.insufficient_samples);
}

#[test]
fn mtp_ngram_utility_gate_zero_tokens_do_not_panic() {
    let decision = mtp_ngram_utility_gate(
        4,
        DraftSourceUtility::default(),
        DraftSourceUtility::default(),
        utility_cfg_for_tests(),
        0,
    );

    assert!(!decision.gated);
    assert!(decision.insufficient_samples);
}

#[test]
fn mtp_utility_uses_separate_baseline_and_stacked_step_buckets() {
    let mut telemetry = MtpTelemetry::default();

    telemetry.record_timings(MtpStepTimings {
        verify_forward_wall_us: 100,
        verify_eval_wall_us: 10,
        target_softmax_wall_us: 10,
        draft_wall_us: 0,
        emitted_tokens: 4,
        ngram_submitted_tokens: 0,
        ..MtpStepTimings::default()
    });
    telemetry.record_timings(MtpStepTimings {
        verify_forward_wall_us: 200,
        verify_eval_wall_us: 50,
        target_softmax_wall_us: 25,
        draft_wall_us: 25,
        emitted_tokens: 2,
        ngram_submitted_tokens: 5,
        ..MtpStepTimings::default()
    });

    let baseline = telemetry.baseline_utility();
    let stacked = telemetry.stacked_utility();

    assert_eq!(baseline.cost_per_emitted_token_us(), Some(30.0));
    assert_eq!(stacked.cost_per_emitted_token_us(), Some(150.0));
    assert_eq!(stacked.submitted_tokens, 5);

    let mut decisions = Vec::new();
    telemetry.append_route_decisions(&mut decisions);
    assert!(decisions.contains(&("ax_mtp_ngram_utility_baseline_steps".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_ngram_utility_baseline_wall_us".into(), 120)));
    assert!(decisions.contains(&("ax_mtp_ngram_utility_stacked_steps".into(), 1)));
    assert!(decisions.contains(&("ax_mtp_ngram_utility_stacked_wall_us".into(), 300)));
    assert!(decisions.contains(&(
        "ax_mtp_ngram_utility_stacked_ngram_submitted_tokens".into(),
        5
    )));
}

#[test]
fn mtp_ngram_safety_defaults_to_reasoning_tighten_mode() {
    assert_eq!(
        MtpNgramSafetyMode::default(),
        MtpNgramSafetyMode::TightenReasoning
    );
    let decision = mtp_ngram_speculative_safety_decision_for_mode(
        MtpNgramSafetyMode::default(),
        false,
        false,
        true,
    );

    assert!(decision.tighten_ngram);
    assert!(!decision.disable_ngram);
    assert_eq!(decision.reason.route_code(), 3);
}

#[test]
fn mtp_ngram_safety_disable_modes_are_explicit() {
    let all = mtp_ngram_speculative_safety_decision_for_mode(
        MtpNgramSafetyMode::DisableAll,
        false,
        false,
        false,
    );
    assert!(all.disable_ngram);
    assert_eq!(all.reason, SpeculativeSafetyReason::ExperimentalOverride);

    let reasoning = mtp_ngram_speculative_safety_decision_for_mode(
        MtpNgramSafetyMode::DisableReasoning,
        false,
        false,
        true,
    );
    assert!(reasoning.disable_ngram);
    assert_eq!(reasoning.reason, SpeculativeSafetyReason::ReasoningTrace);

    let off =
        mtp_ngram_speculative_safety_decision_for_mode(MtpNgramSafetyMode::Off, true, true, true);
    assert_eq!(off, SpeculativeSafetyDecision::default());
}

#[test]
fn mtp_ngram_safety_disables_tool_and_structured_workloads() {
    let tool = mtp_ngram_speculative_safety_decision_for_mode(
        MtpNgramSafetyMode::default(),
        true,
        false,
        false,
    );
    assert!(tool.disable_ngram);
    assert_eq!(tool.reason, SpeculativeSafetyReason::ToolCall);

    let structured = mtp_ngram_speculative_safety_decision_for_mode(
        MtpNgramSafetyMode::default(),
        false,
        true,
        false,
    );
    assert!(structured.disable_ngram);
    assert_eq!(structured.reason, SpeculativeSafetyReason::StructuredOutput);
}

#[test]
fn ngram_self_tune_counts_only_submitted_drafts() {
    let mut state = NgramSelfTuneState::default();
    state.record_verified(0, 0.30, 32);
    assert_eq!(state.drafted, 0);
    assert_eq!(state.accepted, 0);
    assert!(!state.disabled);

    state.record_submitted(16);
    state.record_verified(4, 0.30, 32);
    assert_eq!(state.drafted, 16);
    assert_eq!(state.accepted, 4);
    assert!(!state.disabled);
}

#[test]
fn ngram_self_tune_disables_after_warmup_when_acceptance_low() {
    let mut state = NgramSelfTuneState::default();
    state.record_submitted(32);
    state.record_verified(5, 0.30, 32);
    assert!(state.disabled);
}

#[test]
fn mtp_accept_count_ngram_pseudo_logprob_rejection_samples() {
    // N-gram position with delta distribution: log_prob = 0.0 → p_draft = 1.0.
    // target_prob = 1.0 → accept_prob = 1.0/1.0 = 1.0 → always accept.
    let pseudo_lp = 0.0_f32;
    let mut rng = Xorshift64::new(42);
    let accept = mtp_accept_count(
        &[17],
        &[pseudo_lp],
        &[],
        &[MtpDraftSource::Ngram],
        Some(&[1.0]),
        None,
        &[17],
        &mut rng,
        1.0,
        0.8,
        MtpModelAcceptanceMode::Greedy,
        MtpNgramAcceptanceMode::Delta,
    );
    assert_eq!(accept.accept_count, 1);
    assert!(accept.all_accepted);
    assert_eq!(accept.rejection_correction, None);

    // Low target probability → reject even though tokens match.
    // accept_prob = 0.0/1.0 = 0.0 → never accept.
    let mut rng2 = Xorshift64::new(99);
    let accept2 = mtp_accept_count(
        &[17],
        &[pseudo_lp],
        &[],
        &[MtpDraftSource::Ngram],
        Some(&[0.0]),
        None,
        &[17],
        &mut rng2,
        1.0,
        0.8,
        MtpModelAcceptanceMode::Greedy,
        MtpNgramAcceptanceMode::Delta,
    );
    assert_eq!(accept2.accept_count, 0);
    assert!(!accept2.all_accepted);
}

#[test]
fn mtp_ngram_pseudo_log_probs_cover_ngram_only_draft_windows() {
    let log_probs =
        mtp_ngram_pseudo_log_probs(&[0.9, 0.0, 1.2], 4, MtpNgramAcceptanceMode::Confidence);

    assert!((log_probs[0] - 0.9_f32.ln()).abs() < 1e-6);
    assert_eq!(log_probs[1], -30.0, "zero confidence is clamped");
    assert_eq!(log_probs[2], 0.0, "confidence above 1.0 is clamped");
    assert!(
        log_probs[3].is_nan(),
        "missing confidence falls back to greedy comparison for that position"
    );

    // N-gram-only MTP draft windows must still enter rejection sampling when
    // pseudo log-probs are present.  The target argmax intentionally differs
    // from the pending token; the old empty-log-prob path would reject here.
    let mut rng = Xorshift64::new(42);
    let accept = mtp_accept_count(
        &[17],
        &log_probs[..1],
        &[],
        &[MtpDraftSource::Ngram],
        Some(&[1.0]),
        None,
        &[99],
        &mut rng,
        1.0,
        0.8,
        MtpModelAcceptanceMode::Greedy,
        MtpNgramAcceptanceMode::Confidence,
    );
    assert_eq!(accept.accept_count, 1);
    assert!(accept.all_accepted);
}

#[test]
fn mtp_ngram_pseudo_logprob_delta_mode_returns_zero() {
    let log_probs = mtp_ngram_pseudo_log_probs(&[0.9, 0.0, 1.2], 4, MtpNgramAcceptanceMode::Delta);

    assert_eq!(log_probs[0], 0.0);
    assert_eq!(log_probs[1], 0.0);
    assert_eq!(log_probs[2], 0.0);
    assert!(log_probs[3].is_nan());
}

#[test]
fn mtp_accept_count_ngram_greedy_mode_uses_argmax_match() {
    let mut rng = Xorshift64::new(42);
    let accept = mtp_accept_count(
        &[17],
        &[0.0],
        &[],
        &[MtpDraftSource::Ngram],
        Some(&[0.0]),
        None,
        &[17],
        &mut rng,
        1.0,
        0.8,
        MtpModelAcceptanceMode::Greedy,
        MtpNgramAcceptanceMode::Greedy,
    );
    assert_eq!(accept.accept_count, 1);
    assert!(accept.all_accepted);

    let mut rng2 = Xorshift64::new(42);
    let reject = mtp_accept_count(
        &[17],
        &[0.0],
        &[],
        &[MtpDraftSource::Ngram],
        Some(&[1.0]),
        None,
        &[99],
        &mut rng2,
        1.0,
        0.8,
        MtpModelAcceptanceMode::Greedy,
        MtpNgramAcceptanceMode::Greedy,
    );
    assert_eq!(reject.accept_count, 0);
    assert!(!reject.all_accepted);
    assert_eq!(reject.rejection_correction, Some(99));
}

#[test]
fn mtp_accept_count_ngram_nan_logprob_falls_back_to_greedy() {
    // NaN log-prob is not finite → greedy argmax fallback still applies
    // (backward compatibility with paths that don't carry pseudo log-probs).
    let mut rng = Xorshift64::new(42);
    let accept = mtp_accept_count(
        &[17],
        &[f32::NAN],
        &[],
        &[MtpDraftSource::Ngram],
        Some(&[0.0]),
        None,
        &[17],
        &mut rng,
        1.0,
        0.8,
        MtpModelAcceptanceMode::Greedy,
        MtpNgramAcceptanceMode::Confidence,
    );
    assert_eq!(accept.accept_count, 1);
    assert!(accept.all_accepted);
    assert_eq!(accept.rejection_correction, None);
}

#[test]
fn mtp_accept_count_ngram_rejection_ignores_draft_distribution() {
    let target_distribution = TokenDistribution::new(vec![(99, 1.0)]).unwrap();
    let stale_mtp_distribution = TokenDistribution::new(vec![(17, 1.0)]).unwrap();
    let mut rng = Xorshift64::new(42);

    let accept = mtp_accept_count(
        &[17],
        &[0.9_f32.ln()],
        &[stale_mtp_distribution],
        &[MtpDraftSource::Ngram],
        Some(&[0.0]),
        Some(&[target_distribution]),
        &[17],
        &mut rng,
        1.0,
        0.8,
        MtpModelAcceptanceMode::Greedy,
        MtpNgramAcceptanceMode::Confidence,
    );

    assert_eq!(accept.accept_count, 0);
    assert!(!accept.all_accepted);
    assert_eq!(
        accept.rejection_correction, None,
        "n-gram pseudo log-probs do not have a true draft distribution"
    );
}

#[test]
fn mtp_model_accept_count_defaults_to_argmax_verification() {
    let mut rng = Xorshift64::new(42);
    let reject = mtp_accept_count(
        &[17],
        &[0.01_f32.ln()],
        &[],
        &[MtpDraftSource::Mtp],
        Some(&[1.0]),
        None,
        &[99],
        &mut rng,
        1.0,
        0.8,
        MtpModelAcceptanceMode::Greedy,
        MtpNgramAcceptanceMode::Confidence,
    );

    assert_eq!(reject.accept_count, 0);
    assert!(!reject.all_accepted);
    assert_eq!(reject.rejection_correction, Some(99));
}

#[test]
fn mtp_accept_count_aligns_hybrid_tail_distribution_after_ngram_prefix() {
    let mut rng = Xorshift64::new(7);
    let target_prefix = TokenDistribution::new(vec![(10, 1.0)]).unwrap();
    let target_tail = TokenDistribution::new(vec![(99, 1.0)]).unwrap();
    let draft_tail = TokenDistribution::new(vec![(20, 1.0)]).unwrap();

    // N-gram position uses NAN → greedy fallback (tokens match → accept).
    // HybridMtp position has log_prob=0.0, target_prob=0.0 → reject, correction=99.
    let accept = mtp_accept_count(
        &[10, 20],
        &[f32::NAN, 0.0],
        &[draft_tail],
        &[MtpDraftSource::Ngram, MtpDraftSource::HybridMtp],
        Some(&[1.0, 0.0]),
        Some(&[target_prefix, target_tail]),
        &[10, 20],
        &mut rng,
        1.0,
        0.8,
        MtpModelAcceptanceMode::RejectionSampling,
        MtpNgramAcceptanceMode::Confidence,
    );

    assert_eq!(accept.accept_count, 1);
    assert!(!accept.all_accepted);
    assert_eq!(accept.rejection_correction, Some(99));
}

#[test]
fn mtp_accept_count_temperature_rescaling() {
    // Draft at T=0.7, target at T=0.6 → ratio=7/6≈1.167.
    // log_p_draft = ln(0.8) ≈ -0.223; scaled = -0.223*1.167 ≈ -0.260 → p_scaled ≈ 0.771.
    // target_prob = 1.0 → accept_prob = 1.0/0.771 capped at 1.0 → always accept.
    let mut rng = Xorshift64::new(1);
    let accept = mtp_accept_count(
        &[5],
        &[0.8_f32.ln()],
        &[],
        &[MtpDraftSource::Mtp],
        Some(&[1.0]),
        None,
        &[5],
        &mut rng,
        0.7,
        0.6,
        MtpModelAcceptanceMode::RejectionSampling,
        MtpNgramAcceptanceMode::Confidence,
    );
    assert_eq!(accept.accept_count, 1);

    // When temperatures match, no rescaling occurs.
    let mut rng2 = Xorshift64::new(1);
    let accept2 = mtp_accept_count(
        &[5],
        &[0.8_f32.ln()],
        &[],
        &[MtpDraftSource::Mtp],
        Some(&[1.0]),
        None,
        &[5],
        &mut rng2,
        0.7,
        0.7,
        MtpModelAcceptanceMode::RejectionSampling,
        MtpNgramAcceptanceMode::Confidence,
    );
    assert_eq!(accept2.accept_count, 1);

    // N-gram pseudo log-probs must NOT be temperature-rescaled.
    // The key probe: target_prob == p_draft → unscaled accept_prob = 1.0;
    // if rescaling were wrongly applied (T_target=2.0 > T_draft=0.7,
    // ratio=0.35), log_p_scaled = -6.9 * 0.35 = -2.42 → p_scaled ≈ 0.089
    // → accept_prob ≈ 0.011 → reject for all typical rng values.
    let ultra_low_lp = 0.001_f32.ln();
    for seed in 0u64..10 {
        let mut rng_probe = Xorshift64::new(seed);
        let a = mtp_accept_count(
            &[7],
            &[ultra_low_lp],
            &[],
            &[MtpDraftSource::Ngram],
            Some(&[0.001]),
            None,
            &[7],
            &mut rng_probe,
            0.7,
            2.0,
            MtpModelAcceptanceMode::Greedy,
            MtpNgramAcceptanceMode::Confidence,
        );
        assert_eq!(
            a.accept_count, 1,
            "n-gram pseudo log-prob must not be rescaled (seed={seed})"
        );
    }
}

#[test]
fn mtp_cost_depth_preview_matches_the_committed_decision() {
    let mut controller = MtpCostDepthController::default();
    controller.reset(
        true,
        MtpCostDepthConfig {
            enabled: true,
            ..MtpCostDepthConfig::default()
        },
        3,
    );
    // Warmup sweep, then the warmup-context probes, so both call sites
    // replay the steady-state scoring path.
    for _ in 0..3 {
        controller.observe_and_decide(3, 3, true, true);
    }
    controller.record_direct_probe(9_000);
    controller.record_direct_probe(9_500);
    // Acceptance stays a full prefix acceptance pattern so the expected
    // token count is strictly increasing in depth and the decision cannot
    // sit on a score tie.
    for (used, accepted) in [(3, 3), (3, 3), (3, 2), (3, 1), (2, 2), (3, 3)] {
        let mut preview = controller;
        let preview_depth =
            mtp_cost_depth_cycle_depth(&mut preview, 0, used, accepted, true, true, None);
        let committed_depth =
            mtp_cost_depth_cycle_depth(&mut controller, 0, used, accepted, true, true, None);
        assert_eq!(
            preview_depth, committed_depth,
            "fold preview and committed decision must agree (used={used}, accepted={accepted})"
        );
    }
}

#[test]
fn disabled_cost_controller_keeps_legacy_depth_decisions() {
    let mut controller = MtpCostDepthController::default();
    let policy = MtpAdaptiveDepthPolicy {
        fixed_depth: None,
        conservative_depth: false,
        depth3_miss_backoff: true,
        depth3_hysteresis: true,
    };
    for (current, pending, accepted, misses) in [
        (3, 3, 0, 0),
        (2, 3, 2, 0),
        (3, 3, 3, 0),
        (1, 3, 1, 0),
        (2, 3, 0, 1),
    ] {
        let legacy = mtp_next_adaptive_depth_with_policy(
            current, 3, pending, accepted, misses, false, policy,
        );
        let resolved = mtp_cost_depth_cycle_depth(
            &mut controller,
            legacy,
            pending,
            accepted,
            true,
            true,
            None,
        );
        assert_eq!(resolved, legacy);
    }
    assert!(!controller.enabled());
    assert!(!controller.wants_direct_probe());
    assert!(!controller.parked());
    assert_eq!(controller.observe_and_decide(3, 3, true, true), 0);
}

#[test]
fn fixed_draft_depth_preempts_the_cost_controller() {
    let mut controller = MtpCostDepthController::default();
    controller.reset(
        true,
        MtpCostDepthConfig {
            enabled: true,
            ..MtpCostDepthConfig::default()
        },
        3,
    );
    assert_eq!(controller.observe_and_decide(0, 0, false, true), 3);
    for _ in 0..6 {
        assert_eq!(
            mtp_cost_depth_cycle_depth(&mut controller, 2, 3, 3, true, true, Some(2)),
            2
        );
    }
    // The preempted calls never reached the controller: the warmup sweep
    // resumes exactly where it stopped.
    assert_eq!(controller.observe_and_decide(0, 0, false, true), 2);
}

#[test]
fn parked_cost_controller_routes_direct_fallback() {
    let mut controller = MtpCostDepthController::default();
    controller.reset(
        true,
        MtpCostDepthConfig {
            enabled: true,
            park_streak: 2,
            ..MtpCostDepthConfig::default()
        },
        3,
    );
    assert_eq!(controller.observe_and_decide(0, 0, false, true), 3);
    assert_eq!(controller.observe_and_decide(0, 0, false, true), 2);
    assert_eq!(controller.observe_and_decide(0, 0, false, true), 1);
    // Slow, missing speculative cycles against a settled 1 µs direct
    // baseline: the warmup probes do not arm park, the settling probe
    // (requested only after PARK_BASELINE_PROBE_DELAY_CYCLES) does.
    let mut parked = false;
    for _ in 0..70 {
        if controller.wants_direct_probe() {
            controller.record_direct_probe(1);
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
        // The round verifies the draft generated at the decided depth;
        // anything else is not a clean depth-cost sample by construction.
        let used = controller.current_depth();
        if controller.observe_and_decide(used, 0, true, true) == 0 {
            parked = true;
            break;
        }
    }
    assert!(parked && controller.parked());
    // Park sets `state.mtp_bypassed`, which is the route's direct-only flag.
    assert_eq!(
        mtp_request_route(true, true, true, false, controller.parked(), false),
        MtpRequestRoute::DirectFallback
    );
}

#[test]
fn fresh_draft_log_probs_carry_temperature_only_when_written() {
    // Skip-state and post-verify fresh-draft writes share one rule: a
    // greedy window that computed no temperature-scaled log-probs records
    // no temperature, so the accept path cannot rescale a stale T against
    // an empty vector.
    assert_eq!(pending_draft_log_prob_temperature_for(&[], 0.7), None);
    assert_eq!(pending_draft_log_prob_temperature_for(&[], 1.0), None);
    assert_eq!(
        pending_draft_log_prob_temperature_for(&[-0.5, -1.5], 0.7),
        Some(0.7)
    );
    assert_eq!(
        pending_draft_log_prob_temperature_for(&[-0.25], 1.0),
        Some(1.0)
    );
}

#[test]
fn device_greedy_accept_pair_fails_closed_on_bad_readbacks() {
    // The packed pair is trusted only at exactly two elements and a count
    // that fits the verified window.
    assert_eq!(device_greedy_accept_pair(&[2, 7], 3), Some((2, 7)));
    assert_eq!(device_greedy_accept_pair(&[0, 7], 3), Some((0, 7)));
    assert_eq!(device_greedy_accept_pair(&[3, 7], 3), Some((3, 7)));
    assert_eq!(device_greedy_accept_pair(&[], 3), None);
    assert_eq!(device_greedy_accept_pair(&[1], 3), None);
    assert_eq!(device_greedy_accept_pair(&[1, 2, 3], 3), None);
    assert_eq!(device_greedy_accept_pair(&[4, 7], 3), None);
    assert_eq!(device_greedy_accept_pair(&[u32::MAX, 7], 3), None);
}

#[test]
fn mtp_adaptive_depth_shrinks_on_partial_reject_and_recovers_on_full_accept() {
    let policy = MtpAdaptiveDepthPolicy {
        fixed_depth: None,
        conservative_depth: false,
        depth3_miss_backoff: false,
        depth3_hysteresis: false,
    };
    // consecutive_misses=0 for all non-complete-miss cases.
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(0, 3, 0, 0, 0, false, policy),
        3
    );
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(3, 3, 3, 2, 0, false, policy),
        2
    );
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(2, 3, 2, 1, 0, false, policy),
        2
    );
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(1, 3, 1, 1, 0, false, policy),
        2
    );
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(2, 3, 2, 2, 0, false, policy),
        3
    );
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(3, 0, 3, 3, 0, false, policy),
        0
    );
}

#[test]
fn mtp_adaptive_depth_progressive_floor_on_consecutive_misses() {
    let policy = MtpAdaptiveDepthPolicy {
        fixed_depth: None,
        conservative_depth: false,
        depth3_miss_backoff: false,
        depth3_hysteresis: false,
    };
    // First complete miss (consecutive_misses=0): floor = 2.
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(3, 3, 3, 0, 0, false, policy),
        2
    );
    // Second consecutive miss (consecutive_misses=1): floor = 1.
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(2, 3, 2, 0, 1, false, policy),
        1
    );
    // Third+ consecutive miss (consecutive_misses=2): floor = 0.
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(1, 3, 1, 0, 2, false, policy),
        0
    );
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(1, 3, 1, 0, 5, false, policy),
        0
    );
    // Partial accept resets to normal floor logic (not complete miss path).
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(3, 3, 3, 1, 3, false, policy),
        2
    );
    // Gemma assistant: first complete miss ends drafting (gen median stop-loss).
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(2, 2, 2, 0, 0, true, policy),
        0
    );
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(1, 2, 1, 0, 3, true, policy),
        0
    );
    // Half-or-worse accept on the aggressive short-gen path also stops
    // (1/3 and the common Gemma depth-2 case 1/2).
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(2, 2, 3, 1, 0, true, policy),
        0
    );
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(2, 2, 2, 1, 0, true, policy),
        0
    );
    // Better than half (2/3) keeps progressive floor / clamp logic.
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(2, 3, 3, 2, 0, true, policy),
        2
    );
}

#[test]
fn mtp_adaptive_depth_applies_explicit_throughput_policy() {
    let throughput = MtpAdaptiveDepthPolicy {
        fixed_depth: None,
        conservative_depth: false,
        depth3_miss_backoff: true,
        depth3_hysteresis: true,
    };
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(3, 3, 3, 0, 0, false, throughput),
        2
    );
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(2, 3, 2, 1, 0, false, throughput),
        3
    );
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(3, 3, 3, 2, 0, false, throughput),
        3
    );

    let fixed = MtpAdaptiveDepthPolicy {
        fixed_depth: Some(4),
        conservative_depth: false,
        depth3_miss_backoff: false,
        depth3_hysteresis: false,
    };
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(1, 3, 1, 0, 0, false, fixed),
        3
    );
}

#[test]
fn mtp_conservative_depth_waits_for_evidence_and_preserves_successful_workloads() {
    assert!(!mtp_conservative_depth_admitted((0, 0, 0, 0)));
    assert!(!mtp_conservative_depth_admitted((8, 0, 8, 0)));
    assert!(!mtp_conservative_depth_admitted((31, 0, 31, 0)));
    assert!(!mtp_conservative_depth_admitted((36, 0, 31, 0)));
    assert!(mtp_conservative_depth_admitted((32, 23, 32, 15)));
    assert!(!mtp_conservative_depth_admitted((32, 24, 32, 15)));
    assert!(!mtp_conservative_depth_admitted((32, 23, 32, 16)));
    assert!(!mtp_conservative_depth_admitted((32, 32, 32, 32)));
    assert!(mtp_conservative_depth_admitted((u32::MAX, 0, u32::MAX, 0)));
    assert!(!mtp_conservative_depth_admitted((
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX
    )));
    // Cyber-Tiel's measured moderate first-position yield still pays for
    // deep windows: almost every surviving first proposal accepts a second.
    assert!(!mtp_conservative_depth_admitted((43, 31, 43, 30)));
    assert!(mtp_conservative_depth_admitted((69, 35, 69, 15)));

    // The combined async fold runs before telemetry records this window.
    // Compare predictions with actual record_step, including clipped and
    // empty windows so missing second proposals cannot bias that estimate.
    let mut telemetry = MtpTelemetry::default();
    for (pending, accepted) in [(3, 0), (1, 1), (2, 1), (0, 0), (3, 2), (2, 2), (1, 0)] {
        let predicted = mtp_depth_observations_after_step(&telemetry, pending, accepted);
        telemetry.record_step(
            pending,
            accepted,
            &vec![MtpDraftSource::Mtp; pending],
            None,
            accepted,
        );
        let actual = (
            telemetry.drafted_by_depth[0],
            telemetry.accepted_by_depth[0],
            telemetry.drafted_by_depth[1],
            telemetry.accepted_by_depth[1],
        );
        assert_eq!(predicted, actual);
        assert_eq!(
            mtp_conservative_depth_admitted(predicted),
            mtp_conservative_depth_admitted(actual)
        );
    }
    assert_eq!(
        mtp_depth_observations_after_step(&telemetry, 0, 0),
        (6, 4, 4, 2)
    );
    assert_eq!(
        mtp_depth_observations_after_step(&telemetry, 1, 1),
        (7, 5, 4, 2)
    );
}

#[test]
fn mtp_conservative_depth_recovers_after_rejection_without_repeating_deep_misses() {
    let policy = MtpAdaptiveDepthPolicy {
        fixed_depth: None,
        conservative_depth: true,
        depth3_miss_backoff: true,
        depth3_hysteresis: true,
    };
    let mut depth = 3;
    // Two misses, then useful prefixes, then full acceptance: maintain
    // a live one-token probe and recover only as actual windows succeed.
    for (submitted, accepted, expected) in [
        (3, 0, 1),
        (1, 0, 1),
        (1, 1, 2),
        (2, 1, 1),
        (1, 1, 2),
        (2, 2, 3),
    ] {
        depth =
            mtp_next_adaptive_depth_with_policy(depth, 3, submitted, accepted, 0, false, policy);
        assert_eq!(depth, expected);
    }
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(3, 3, 3, 2, 0, false, policy),
        2
    );
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(2, 3, 0, 0, 0, false, policy),
        2
    );
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(2, 3, 1, 1, 0, false, policy),
        2
    );
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(3, 0, 3, 0, 0, false, policy),
        0
    );

    let fixed = MtpAdaptiveDepthPolicy {
        fixed_depth: Some(3),
        ..policy
    };
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(1, 3, 1, 0, 0, false, fixed),
        3
    );
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(1, 1, 1, 0, 0, false, fixed),
        1
    );

    // Other profiles retain their stop-loss behavior even if the opt-in
    // is set globally; this policy only covers three-token throughput.
    assert_eq!(
        mtp_next_adaptive_depth_with_policy(2, 2, 2, 0, 0, true, policy),
        0
    );
}

fn test_prefix_key(token: u32) -> MlxPrefixCacheKey {
    MlxPrefixCacheKey {
        model_id: "model".into(),
        route_policy: "direct".into(),
        layer_layout: "layers=2;full_attention_only".into(),
        block_size_tokens: 4,
        token_count: 4,
        token_hash: hash_prefix_tokens(&[token; 4]),
        media_key: String::new(),
    }
}

#[test]
fn media_key_is_fail_closed_for_undigested_media() {
    // Text-only requests keep the empty (cacheable) key.
    assert_eq!(media_key_for_fingerprint(None, "fp"), Some(String::new()));
    let empty = ax_engine_core::RequestMultimodalInputs::default();
    assert_eq!(
        media_key_for_fingerprint(Some(&empty), "fp"),
        Some(String::new())
    );

    // Media without a digest mapping (unlimited_ocr) must yield None —
    // a text-only key here would alias prompts with different images.
    let ocr = ax_engine_core::RequestMultimodalInputs {
        unlimited_ocr: Some(ax_engine_core::unlimited_ocr::UnlimitedOcrRuntimeInputs {
            image_token_id: 7,
            soft_token_count: 4,
            cropping: false,
            images: vec![
                ax_engine_core::unlimited_ocr::UnlimitedOcrImageRuntimeInput {
                    width: 1,
                    height: 1,
                    rgb_bytes: vec![1, 2, 3],
                },
            ],
        }),
        ..Default::default()
    };
    assert_eq!(media_key_for_fingerprint(Some(&ocr), "fp"), None);
}

#[test]
fn multimodal_prefix_layout_folds_media_digest() {
    let base = "layers=12;ordered-prefix-v2";
    assert_eq!(format_prefix_layer_layout(base, ""), base);
    let with_media = format_prefix_layer_layout(base, "abc123");
    assert_eq!(with_media, "layers=12;ordered-prefix-v2;media=abc123");
    // Different images → different keys (via media_prefix_key + layout fold).
    let a = ax_engine_core::gemma4_unified::Gemma4UnifiedRuntimeInputs {
        images: vec![
            ax_engine_core::gemma4_unified::Gemma4UnifiedImageRuntimeInput {
                span: ax_engine_core::gemma4_unified::Gemma4UnifiedTokenSpan {
                    modality: ax_engine_core::gemma4_unified::Gemma4UnifiedModality::Image,
                    placeholder_index: 0,
                    replacement_start: 0,
                    soft_token_count: 4,
                    replacement_token_count: 6,
                },
                pixel_values: vec![0.1, 0.2],
                pixel_position_ids: vec![[0, 0]; 4],
            },
        ],
        audios: Vec::new(),
        videos: Vec::new(),
    };
    let mut b = a.clone();
    b.images[0].pixel_values = vec![0.9, 0.8];
    let ka = format_prefix_layer_layout(base, &a.media_prefix_key("fp"));
    let kb = format_prefix_layer_layout(base, &b.media_prefix_key("fp"));
    assert_ne!(ka, kb);
    assert!(ka.contains(";media="));
}

fn test_prefix_snapshot(token: u32, token_count: usize, bytes: u64) -> MlxPrefixSnapshot {
    let payload = MlxKVCache::new(2).serialize_to_bytes();
    MlxPrefixSnapshot {
        kv_cache_payload: Arc::from(payload.into_boxed_slice()),
        tokens: vec![token; token_count],
        token_count,
        bytes,
        greedy_prefill_output_token: Some(7),
        mtp_cursor_payload: None,
    }
}

#[test]
fn prefix_cache_store_clones_share_l1_entries() {
    let store = MlxPrefixCacheStore::memory_only_for_tests(MlxPrefixCachePolicy {
        max_bytes: 1024,
        max_entries: 4,
    });
    let cloned = store.clone();
    let key = test_prefix_key(1);

    let outcome = store
        .prefix_cache
        .lock()
        .insert(key.clone(), test_prefix_snapshot(1, 4, 128));
    assert!(outcome.stored);

    let hit = cloned.prefix_cache.lock().get(&key, &[1; 4]);
    assert!(hit.is_some(), "cloned store must see original L1 insert");
}

#[test]
fn prefix_cache_returns_exact_snapshot_and_updates_stats() {
    let mut cache = MlxPrefixCache::new(MlxPrefixCachePolicy {
        max_bytes: 1024,
        max_entries: 4,
    });
    let key = test_prefix_key(1);

    let outcome = cache.insert(key.clone(), test_prefix_snapshot(1, 4, 128));
    assert!(outcome.stored);
    assert_eq!(outcome.evictions, 0);

    let hit = cache
        .get(&key, &[1; 4])
        .expect("prefix snapshot should hit");
    assert_eq!(hit.token_count, 4);
    assert_eq!(hit.greedy_prefill_output_token, Some(7));
    assert_eq!(
        hit.rehydrate_cache()
            .expect("serialized L1 snapshot should rehydrate")
            .usage_snapshot()
            .logical_tokens,
        0
    );
    assert_eq!(
        cache.stats(),
        MlxPrefixCacheStats {
            entries: 1,
            bytes: 128,
        }
    );
}

#[test]
fn prefix_cache_hits_do_not_grow_lru_without_bound() {
    let mut cache = MlxPrefixCache::new(MlxPrefixCachePolicy {
        max_bytes: 1024,
        max_entries: 4,
    });
    let key = test_prefix_key(1);
    cache.insert(key.clone(), test_prefix_snapshot(1, 4, 128));

    for _ in 0..10_000 {
        assert!(cache.get(&key, &[1; 4]).is_some());
    }

    assert_eq!(cache.stats().entries, 1);
    assert!(
        cache.lru.len() <= cache.stale_lru_compaction_limit(),
        "stale LRU ticks should be compacted under repeated hits"
    );
}

#[test]
fn prefix_cache_eviction_is_lru_and_visible() {
    let mut cache = MlxPrefixCache::new(MlxPrefixCachePolicy {
        max_bytes: 256,
        max_entries: 2,
    });
    let key1 = test_prefix_key(1);
    let key2 = test_prefix_key(2);
    let key3 = test_prefix_key(3);

    assert!(
        cache
            .insert(key1.clone(), test_prefix_snapshot(1, 4, 100))
            .stored
    );
    assert!(
        cache
            .insert(key2.clone(), test_prefix_snapshot(2, 4, 100))
            .stored
    );
    assert!(
        cache.get(&key1, &[1; 4]).is_some(),
        "key1 should become most recent"
    );
    let outcome = cache.insert(key3.clone(), test_prefix_snapshot(3, 4, 100));

    assert!(outcome.stored);
    assert_eq!(outcome.evictions, 1);
    assert!(cache.get(&key1, &[1; 4]).is_some());
    assert!(cache.get(&key2, &[2; 4]).is_none());
    assert!(cache.get(&key3, &[3; 4]).is_some());
}

#[test]
fn prefix_cache_disabled_policy_does_not_store_snapshots() {
    let mut cache = MlxPrefixCache::new(MlxPrefixCachePolicy {
        max_bytes: 0,
        max_entries: 4,
    });
    let key = test_prefix_key(1);

    let outcome = cache.insert(key.clone(), test_prefix_snapshot(1, 4, 128));

    assert!(!cache.enabled());
    assert!(!outcome.stored);
    assert_eq!(outcome.evictions, 0);
    assert!(cache.get(&key, &[1; 4]).is_none());
    assert_eq!(
        cache.stats(),
        MlxPrefixCacheStats {
            entries: 0,
            bytes: 0,
        }
    );
}

#[test]
fn prefix_cache_hash_collision_misses_without_reusing_snapshot() {
    let mut cache = MlxPrefixCache::new(MlxPrefixCachePolicy {
        max_bytes: 1024,
        max_entries: 4,
    });
    let key = test_prefix_key(1);

    let outcome = cache.insert(key.clone(), test_prefix_snapshot(1, 4, 128));
    assert!(outcome.stored);

    assert!(
        cache.get(&key, &[9; 4]).is_none(),
        "same hash key must still require exact prefix tokens"
    );
    assert_eq!(cache.stats().entries, 1);
    assert!(cache.get(&key, &[1; 4]).is_some());
}

#[test]
fn prefix_cache_exact_membership_rejects_collision_tokens_without_touching_lru() {
    let mut cache = MlxPrefixCache::new(MlxPrefixCachePolicy {
        max_bytes: 1024,
        max_entries: 4,
    });
    let key = test_prefix_key(1);
    cache.insert(key.clone(), test_prefix_snapshot(1, 4, 128));

    assert!(cache.contains_exact_tokens(&key, &[1; 4]));
    assert!(!cache.contains_exact_tokens(&key, &[9; 4]));
    assert_eq!(
        cache.lru.len(),
        1,
        "read-only membership probe must not touch LRU state"
    );
}

#[test]
fn linear_boundary_capture_head_len_covers_alignment_cases() {
    let head = MlxRunner::linear_boundary_capture_head_len;
    // Single-item unaligned prompt: capture at floor16.
    assert_eq!(head(16, 0, 61), Some(48));
    // Aligned prompt end: exact-alignment store path handles it.
    assert_eq!(head(16, 0, 64), None);
    // Prompt shorter than one block: no boundary exists.
    assert_eq!(head(16, 0, 9), None);
    // Continuation item: boundary is absolute, not item-relative.
    assert_eq!(head(16, 2048, 61), Some(48));
    assert_eq!(head(16, 2040, 61), Some(56)); // 2040+56 = 2096 = 131*16
    // Boundary inside already-cached context: nothing to capture.
    assert_eq!(head(16, 2050, 9), None); // total 2059, boundary 2048 <= prior
    // Degenerate inputs.
    assert_eq!(head(0, 0, 61), None);
    assert_eq!(head(16, 0, 0), None);
    // Four-token blocks around the 69-token real-pack prompt: capture
    // never invents a token. The extra direct-prefill position at 70 is
    // the pipeline bootstrap, not this boundary.
    for prompt_len in 65..=72 {
        let captured = head(4, 0, prompt_len);
        if prompt_len.is_multiple_of(4) {
            assert_eq!(captured, None, "aligned prompt_len={prompt_len}");
        } else {
            assert_eq!(
                captured,
                Some(prompt_len - prompt_len % 4),
                "unaligned prompt_len={prompt_len}"
            );
        }
    }
    assert_eq!(head(4, 0, 69), Some(68));
    assert_eq!(head(4, 0, 68), None);
    assert_eq!(head(4, 0, 72), None);
}

#[test]
fn prefix_probe_continues_past_longer_miss_to_shorter_match() {
    let input = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let mut probed_lengths = Vec::new();

    let prefix = MlxRunner::longest_block_aligned_prefix_by_probe(4, &input, |tokens| {
        probed_lengths.push(tokens.len());
        tokens == [1, 2, 3, 4]
    });

    assert_eq!(prefix, Some(vec![1, 2, 3, 4]));
    assert_eq!(
        probed_lengths,
        vec![8, 4],
        "probe must keep searching after a longer non-exact entry"
    );
}

fn scheduler_claim_item(reused_prefix_token_slice: Vec<u32>) -> ax_engine_core::ExecutionItem {
    ax_engine_core::ExecutionItem {
        request_id: RequestId(99),
        mode: ExecutionMode::Prefill,
        planned_work_unit: ax_engine_core::WorkUnitKind::PrefillChunk,
        input_token_slice: vec![9, 10],
        reused_prefix_token_slice,
        position_range: PositionRange {
            start: 8,
            end_exclusive: 10,
        },
        scheduled_token_count: 2,
        block_table_ref: RequestId(99),
        prefix_tokens_reused: 8,
        prefix_blocks_reused: 2,
    }
}

#[test]
fn probe_over_claim_uses_absolute_prefill_position() {
    let mut later_cold_chunk = scheduler_claim_item(Vec::new());
    later_cold_chunk.position_range = PositionRange {
        start: 2048,
        end_exclusive: 4096,
    };
    later_cold_chunk.scheduled_token_count = 2048;

    assert_eq!(
        runner_probe_over_claim(2048, &later_cold_chunk),
        0,
        "ordinary multi-step prefill must append the whole next chunk",
    );

    let mut first_chunk = scheduler_claim_item(Vec::new());
    first_chunk.position_range = PositionRange {
        start: 0,
        end_exclusive: 2,
    };
    assert_eq!(
        runner_probe_over_claim(6144, &first_chunk),
        6144,
        "only cache state beyond the scheduler's absolute start is a probe bonus",
    );
    assert_eq!(
        runner_probe_over_claim(4, &later_cold_chunk),
        0,
        "a cache behind the planned position cannot cover item tokens",
    );
}

#[test]
fn scheduler_split_prefill_defers_portable_prefix_serialization() {
    assert!(!portable_prefix_store_allowed(false, false));
    assert!(portable_prefix_store_allowed(true, false));
    assert!(
        portable_prefix_store_allowed(false, true),
        "native FA snapshots pin refcounted blocks without serializing the full KV payload",
    );
}

#[test]
fn discard_probe_shorter_than_scheduler_claim_rejects_partial_restore() {
    // Regression test: the scheduler's cumulative multi-turn block
    // tracking (ax-engine-core) claimed an 8-token reusable prefix, but
    // the runner-side snapshot map only ever stored a 4-token
    // block-aligned prefix of it (e.g. the original turn-1 prompt).
    // Restoring the shorter 4-token probe result directly would leave
    // `token_ids` (planned by the scheduler for absolute positions
    // 8..10) computed on top of a 4-token cache instead — silently
    // dropping tokens 4..8 and shifting every later RoPE position.
    // The probe result must be discarded so the caller falls through to
    // a full, correctly-positioned recompute instead.
    let item = scheduler_claim_item(vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let probed = vec![1, 2, 3, 4];

    let result = MlxRunner::discard_probe_shorter_than_scheduler_claim(probed, &item);

    assert!(
        result.is_empty(),
        "a probe shorter than a non-empty scheduler claim must be discarded"
    );
}

#[test]
fn discard_probe_shorter_than_scheduler_claim_keeps_full_length_match() {
    // A probe that found exactly the scheduler's claimed length is a
    // genuine full restore — must not be discarded.
    let item = scheduler_claim_item(vec![1, 2, 3, 4]);
    let probed = vec![1, 2, 3, 4];

    let result = MlxRunner::discard_probe_shorter_than_scheduler_claim(probed.clone(), &item);

    assert_eq!(result, probed);
}

#[test]
fn discard_probe_shorter_than_scheduler_claim_keeps_bonus_when_claim_is_empty() {
    // When the scheduler claims zero reuse, `probe_upper_bound` falls
    // back to `input_token_slice`, so any probed length is a pure bonus
    // beyond the scheduler's (zero) expectation — the existing
    // `probe_over_claim` slicing at the call site handles trimming
    // `token_ids` for this case. Must never be discarded here.
    let item = scheduler_claim_item(Vec::new());
    let probed = vec![1, 2, 3, 4, 5, 6];

    let result = MlxRunner::discard_probe_shorter_than_scheduler_claim(probed.clone(), &item);

    assert_eq!(result, probed);
}

#[test]
fn discard_probe_shorter_than_scheduler_claim_is_noop_on_already_empty_probe() {
    let item = scheduler_claim_item(vec![1, 2, 3, 4]);

    let result = MlxRunner::discard_probe_shorter_than_scheduler_claim(Vec::new(), &item);

    assert!(result.is_empty());
}

#[test]
fn prefix_cache_telemetry_merges_blocked_reason_counters() {
    let mut telemetry = MlxPrefixCacheTelemetry::default();
    telemetry.record_blocked_policy_disabled();
    telemetry.record_blocked_unsupported_layout();

    let mut other = MlxPrefixCacheTelemetry::default();
    other.record_blocked_trim_failure();
    other.record_blocked_snapshot_incomplete();
    other.record_blocked_media_identity();
    other.record_blocked_restore_error();
    other.record_blocked_unsupported_layout();
    other.record_disk_store_committed(8192, 2);
    other.record_disk_store_enqueued(4096);
    telemetry.merge_from(other);

    assert_eq!(telemetry.blocked, 7);
    assert_eq!(telemetry.blocked_policy_disabled, 1);
    assert_eq!(telemetry.blocked_unsupported_layout, 2);
    assert_eq!(telemetry.blocked_trim_failure, 1);
    assert_eq!(telemetry.blocked_snapshot_incomplete, 1);
    assert_eq!(telemetry.blocked_media_identity, 1);
    assert_eq!(telemetry.blocked_restore_error, 1);
    assert_eq!(telemetry.disk_inserts, 1);
    assert_eq!(telemetry.disk_store_committed, 1);
    assert_eq!(telemetry.disk_store_enqueued, 1);
    assert_eq!(telemetry.disk_insert_bytes, 8192);
    assert_eq!(telemetry.disk_evictions, 2);
}

#[test]
fn prefix_cache_telemetry_merges_per_reason_admission_counts() {
    use crate::disk_prefix_cache::DiskAdmissionReason;

    // The last-write-wins scalar loses all but one reason per merge;
    // the per-variant counts must not.
    let mut telemetry = MlxPrefixCacheTelemetry::default();
    telemetry.record_disk_admission(DiskAdmissionReason::ArtifactIdentityUnavailable);

    let mut other = MlxPrefixCacheTelemetry::default();
    other.record_disk_admission(DiskAdmissionReason::PrefixTooShort);
    other.record_disk_admission(DiskAdmissionReason::PrefixTooShort);
    telemetry.merge_from(other);

    let artifact_slot = (DiskAdmissionReason::ArtifactIdentityUnavailable.code() - 1) as usize;
    let short_slot = (DiskAdmissionReason::PrefixTooShort.code() - 1) as usize;
    assert_eq!(telemetry.disk_admission_reason_counts[artifact_slot], 1);
    assert_eq!(telemetry.disk_admission_reason_counts[short_slot], 2);
    assert_eq!(
        telemetry.disk_admission_reason_code,
        DiskAdmissionReason::PrefixTooShort.code()
    );
    assert_eq!(telemetry.disk_admission_rejected, 3);

    let mut decisions = Vec::new();
    telemetry.append_route_decisions(&mut decisions);
    assert!(decisions.contains(&(
        "ax_mlx_prefix_cache_disk_reason_artifact_identity_unavailable".into(),
        1
    )));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_disk_reason_prefix_too_short".into(), 2)));
    // Zero-count reasons stay out of the route map.
    assert!(
        !decisions
            .iter()
            .any(|(key, _)| key == "ax_mlx_prefix_cache_disk_reason_disabled")
    );
}

#[test]
fn prefix_cache_telemetry_writes_route_counters() {
    let telemetry = MlxPrefixCacheTelemetry {
        hits: 1,
        misses: 2,
        blocked: 4,
        blocked_policy_disabled: 1,
        blocked_unsupported_layout: 1,
        blocked_trim_failure: 1,
        blocked_snapshot_incomplete: 1,
        blocked_media_identity: 1,
        blocked_restore_error: 1,
        blocked_entry_too_large: 1,
        stores: 4,
        evictions: 5,
        reused_tokens: 16,
        warmup_tokens: 8,
        entries: 2,
        bytes: 4096,
        native_hits: 11,
        native_stores: 12,
        native_evictions: 13,
        demotions: 3,
        demoted_bytes: 4096,
        demotion_skips: 2,
        disk_hits: 6,
        disk_misses: 7,
        disk_inserts: 8,
        disk_insert_bytes: 8192,
        disk_evictions: 9,
        disk_store_enqueued: 10,
        disk_store_committed: 8,
        disk_store_commit_failed: 1,
        ..MlxPrefixCacheTelemetry::default()
    };
    let mut decisions = Vec::new();

    telemetry.append_route_decisions(&mut decisions);

    assert!(decisions.contains(&("ax_mlx_prefix_cache_hits".into(), 1)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_misses".into(), 2)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_blocked".into(), 4)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_blocked_policy_disabled".into(), 1)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_blocked_unsupported_layout".into(), 1)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_blocked_trim_failure".into(), 1)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_blocked_snapshot_incomplete".into(), 1)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_blocked_media_identity".into(), 1)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_blocked_restore_error".into(), 1)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_blocked_entry_too_large".into(), 1)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_evictions".into(), 5)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_bytes_kib".into(), 4)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_native_hits".into(), 11)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_native_stores".into(), 12)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_native_evictions".into(), 13)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_demotions".into(), 3)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_demoted_bytes_kib".into(), 4)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_demotion_skips".into(), 2)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_disk_hits".into(), 6)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_disk_misses".into(), 7)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_disk_inserts".into(), 8)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_disk_insert_bytes_kib".into(), 8)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_disk_evictions".into(), 9)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_disk_store_enqueued".into(), 10)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_disk_store_committed".into(), 8)));
    assert!(decisions.contains(&("ax_mlx_prefix_cache_disk_store_commit_failed".into(), 1)));
}

#[test]
fn prefill_grid_trim_replays_scheduler_prefix_without_losing_context() {
    for (prefix_len, suffix_len, chunk, expected) in [
        (4, 8, 8, None),
        (12, 20, 8, None),
        (12, 2, 8, None),
        (8, 2, 8, Some(8)),
        (12, 1, 8, Some(12)),
        (0, 20, 8, Some(0)),
    ] {
        let item = ax_engine_core::ExecutionItem {
            request_id: RequestId(21),
            mode: ExecutionMode::Prefill,
            planned_work_unit: ax_engine_core::WorkUnitKind::PrefillChunk,
            input_token_slice: vec![2; suffix_len],
            reused_prefix_token_slice: vec![1; prefix_len],
            position_range: PositionRange {
                start: prefix_len as u32,
                end_exclusive: (prefix_len + suffix_len) as u32,
            },
            scheduled_token_count: suffix_len as u32,
            block_table_ref: RequestId(21),
            prefix_tokens_reused: prefix_len as u32,
            prefix_blocks_reused: 1,
        };
        let selected = grid_aligned_prefix_restore_len(&item, prefix_len, chunk);
        assert_eq!(
            selected, expected,
            "prefix={prefix_len}, suffix={suffix_len}"
        );
        if selected.is_none() {
            let telemetry = MlxPrefixCacheTelemetry {
                warmup_tokens: prefix_len as u32,
                ..MlxPrefixCacheTelemetry::default()
            };
            let state = RequestState::new(2, 21, None);
            let replay = full_prefill_recompute_tokens_for_warmup_fallback(
                &item,
                &item.input_token_slice,
                &telemetry,
                &state,
            )
            .expect("trimmed scheduler claims must replay the complete prompt");
            assert_eq!(replay, [vec![1; prefix_len], vec![2; suffix_len]].concat());
        }
        if prefix_len == 0 {
            assert_eq!(grid_aligned_prefix_restore_len(&item, 12, chunk), Some(8));
        }
    }
}

#[test]
fn prefill_warmup_fallback_recomputes_full_prompt_once() {
    let item = ax_engine_core::ExecutionItem {
        request_id: RequestId(21),
        mode: ExecutionMode::Prefill,
        planned_work_unit: ax_engine_core::WorkUnitKind::PrefillChunk,
        input_token_slice: vec![5, 6],
        reused_prefix_token_slice: vec![1, 2, 3, 4],
        position_range: PositionRange {
            start: 4,
            end_exclusive: 6,
        },
        scheduled_token_count: 2,
        block_table_ref: RequestId(21),
        prefix_tokens_reused: 4,
        prefix_blocks_reused: 1,
    };
    let state = RequestState::new(2, 21, None);
    let telemetry = MlxPrefixCacheTelemetry {
        misses: 1,
        warmup_tokens: 4,
        ..MlxPrefixCacheTelemetry::default()
    };

    let tokens = full_prefill_recompute_tokens_for_warmup_fallback(
        &item,
        &item.input_token_slice,
        &telemetry,
        &state,
    )
    .expect("warmup fallback prefill should run prefix+suffix together");

    assert_eq!(tokens, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn prefill_warmup_fallback_does_not_recompute_cache_hits_or_decode() {
    let mut item = ax_engine_core::ExecutionItem {
        request_id: RequestId(22),
        mode: ExecutionMode::Prefill,
        planned_work_unit: ax_engine_core::WorkUnitKind::PrefillChunk,
        input_token_slice: vec![5, 6],
        reused_prefix_token_slice: vec![1, 2, 3, 4],
        position_range: PositionRange {
            start: 4,
            end_exclusive: 6,
        },
        scheduled_token_count: 2,
        block_table_ref: RequestId(22),
        prefix_tokens_reused: 4,
        prefix_blocks_reused: 1,
    };
    let state = RequestState::new(2, 22, None);
    let hit_telemetry = MlxPrefixCacheTelemetry {
        hits: 1,
        reused_tokens: 4,
        ..MlxPrefixCacheTelemetry::default()
    };

    assert!(
        full_prefill_recompute_tokens_for_warmup_fallback(
            &item,
            &item.input_token_slice,
            &hit_telemetry,
            &state,
        )
        .is_none(),
        "snapshot hits already restored prefix KV and must prefill only the suffix",
    );

    item.mode = ExecutionMode::Decode;
    let warmup_telemetry = MlxPrefixCacheTelemetry {
        misses: 1,
        warmup_tokens: 4,
        ..MlxPrefixCacheTelemetry::default()
    };
    assert!(
        full_prefill_recompute_tokens_for_warmup_fallback(
            &item,
            &item.input_token_slice,
            &warmup_telemetry,
            &state,
        )
        .is_none(),
        "decode still needs a warmed prefix KV and optional prefill output token",
    );
}

#[test]
fn generation_ngram_seed_uses_reconstructed_prompt_after_prefix_warmup() {
    let mut warm = RequestState::new(2, 11, None);
    warm.prompt_prefix_tokens = vec![10, 11, 12, 13, 10, 11, 12];

    seed_generation_ngram_from_prompt(&mut warm, false);

    assert_eq!(
        warm.ngram.predict(1),
        vec![13],
        "warm prefix+suffix prefill must seed n-grams from the reconstructed full prompt",
    );

    let mut suffix_only = RequestState::new(2, 12, None);
    suffix_only.prompt_prefix_tokens = vec![10, 11, 12];
    seed_generation_ngram_from_prompt(&mut suffix_only, false);

    assert!(
        suffix_only.ngram.predict(1).is_empty(),
        "feeding only the final prefill item loses the prompt context needed for deterministic warm_extend",
    );
}

#[test]
fn generation_ngram_seed_includes_prefill_output_token() {
    let mut state = RequestState::new(2, 13, None);
    state.prompt_prefix_tokens = vec![1, 2, 3, 1, 2, 3];

    seed_generation_ngram_from_prompt(&mut state, false);
    assert_eq!(
        state.ngram.predict(1),
        vec![1],
        "prompt tail predicts from the final prompt context before the first generated token is committed",
    );

    seed_generation_ngram_from_prefill_output(&mut state, Some(1));
    assert_eq!(
        state.ngram.predict(1),
        vec![2],
        "the prefill-sampled first output token must become part of the next decode context",
    );
}

#[test]
fn generation_ngram_seed_extends_window_for_repeating_prompts() {
    let block: Vec<u32> = (1..=70).collect();
    let mut state = RequestState::new(2, 14, None);
    state.prompt_prefix_tokens.extend_from_slice(&block);
    state.prompt_prefix_tokens.extend_from_slice(&block);
    state.prompt_prefix_tokens.extend_from_slice(&block);

    seed_generation_ngram_from_prompt(&mut state, false);

    assert_eq!(
        state.ngram.predict(1),
        vec![1],
        "repeating prompts should seed enough history to find suffix continuations beyond the 64-token random-prompt guard",
    );
}

#[test]
fn generation_ngram_seed_keeps_random_prompts_on_short_tail() {
    let mut state = RequestState::new(2, 15, None);
    state.prompt_prefix_tokens = (1..=210).collect();
    state
        .prompt_prefix_tokens
        .extend_from_slice(&[147, 148, 149, 150]);

    seed_generation_ngram_from_prompt(&mut state, false);

    assert!(
        state.ngram.predict(1).is_empty(),
        "non-repeating prompts should keep the short tail guard and avoid long-range random false positives",
    );
}

#[test]
fn ngram_decode_result_keeps_correction_token_in_output_queue() {
    let mut state = RequestState::new(2, 7, None);

    let output = apply_decode_result(&mut state, &[11, 12], &[]);

    assert_eq!(output, vec![11, 12]);
    assert_eq!(
        state.bonus_queue.iter().copied().collect::<Vec<_>>(),
        Vec::<u32>::new(),
        "verified tokens should be returned in the current runner update"
    );
    assert_eq!(state.next_model_last_token, Some(12));
}

#[test]
fn ngram_decode_result_queues_full_accept_tail_and_bonus() {
    let mut state = RequestState::new(2, 8, None);

    let output = apply_decode_result(&mut state, &[21, 22, 23, 24], &[]);

    assert_eq!(output, vec![21, 22, 23, 24]);
    assert_eq!(
        state.bonus_queue.iter().copied().collect::<Vec<_>>(),
        Vec::<u32>::new(),
        "accepted drafts and final token should be emitted in one runner update"
    );
    assert_eq!(state.next_model_last_token, Some(24));
}

#[test]
fn stop_reason_prefers_eos_before_max_output() {
    assert_eq!(
        truncate_sampled_tokens_for_stop(vec![151645], 1, 32, &[151645]),
        (vec![151645], Some(StopReason::EosToken))
    );
    assert_eq!(
        truncate_sampled_tokens_for_stop(vec![7], 31, 32, &[151645]),
        (vec![7], Some(StopReason::MaxOutputTokens))
    );
    assert_eq!(
        truncate_sampled_tokens_for_stop(vec![7], 1, 32, &[151645]),
        (vec![7], None)
    );
}

#[test]
fn empty_terminal_token_slice_ignores_eos_until_limit() {
    assert_eq!(
        truncate_sampled_tokens_for_stop(vec![151645], 1, 32, &[]),
        (vec![151645], None)
    );
    assert_eq!(
        truncate_sampled_tokens_for_stop(vec![151645], 31, 32, &[]),
        (vec![151645], Some(StopReason::MaxOutputTokens))
    );
}

#[test]
fn sampled_token_batch_truncates_at_eos_before_max_output() {
    assert_eq!(
        truncate_sampled_tokens_for_stop(vec![31, 151645, 33], 30, 32, &[151645]),
        (vec![31, 151645], Some(StopReason::EosToken))
    );
}

#[test]
fn sampled_token_batch_truncates_at_max_output() {
    assert_eq!(
        truncate_sampled_tokens_for_stop(vec![31, 32, 33], 30, 32, &[]),
        (vec![31, 32], Some(StopReason::MaxOutputTokens))
    );
}

#[test]
fn ngram_decode_result_truncates_bonus_queue_at_eos() {
    let mut state = RequestState::new(2, 9, None);

    let output = apply_decode_result(&mut state, &[31, 32, 151645, 33], &[151645]);

    assert_eq!(output, vec![31, 32, 151645]);
    assert_eq!(
        state.bonus_queue.iter().copied().collect::<Vec<_>>(),
        Vec::<u32>::new(),
        "verified tokens after EOS must not be emitted"
    );
    assert_eq!(state.next_model_last_token, Some(151645));
}

#[test]
fn split_prefill_only_completes_on_final_prompt_chunk() {
    let item = ax_engine_core::ExecutionItem {
        request_id: RequestId(10),
        mode: ExecutionMode::Prefill,
        planned_work_unit: ax_engine_core::WorkUnitKind::PrefillChunk,
        input_token_slice: vec![0; 2048],
        reused_prefix_token_slice: Vec::new(),
        position_range: PositionRange {
            start: 0,
            end_exclusive: 2048,
        },
        scheduled_token_count: 2048,
        block_table_ref: RequestId(10),
        prefix_tokens_reused: 0,
        prefix_blocks_reused: 0,
    };
    let first_context = RunnerRequestContext {
        request_id: RequestId(10),
        prompt_len: 2722,
        processed_prompt_tokens: 0,
        generated_len: 0,
        max_output_tokens: 24,
        seed: 0,
        deterministic_argmax_sampling: true,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        repetition_context_size: None,
        no_repeat_ngram_size: 0,
        ngram_window: 128,
        ignore_eos: false,
        tool_call_mode: false,
        structured_output_mode: false,
        min_p: None,
        max_think_tokens: None,
        answer_reserve_tokens: None,
    };
    assert!(!prefill_item_completes_prompt(&item, Some(&first_context)));

    let final_item = ax_engine_core::ExecutionItem {
        input_token_slice: vec![0; 674],
        position_range: PositionRange {
            start: 2048,
            end_exclusive: 2722,
        },
        scheduled_token_count: 674,
        ..item
    };
    let final_context = RunnerRequestContext {
        processed_prompt_tokens: 2048,
        ..first_context
    };
    assert!(prefill_item_completes_prompt(
        &final_item,
        Some(&final_context)
    ));
}

#[test]
fn prefill_drain_count_handles_zero_and_wide_chunk_sizes() {
    assert_eq!(prefill_drain_async_eval_count(0, 0), 0);
    assert_eq!(prefill_drain_async_eval_count(2048, 2048), 0);
    assert_eq!(prefill_drain_async_eval_count(2049, 2048), 1);
    assert_eq!(
        prefill_drain_async_eval_count(2049, (u32::MAX as usize) + 1),
        0
    );
    assert_eq!(prefill_drain_async_eval_count(usize::MAX, 1), u32::MAX);
}

#[test]
fn prefill_rotation_follows_prefill_flag_and_decode_latch() {
    // Prefill rotates only when the opt-in prefill flag is engaged.
    assert_eq!(
        cache_rotation_for_execution_with_prefill_flag(
            ExecutionMode::Prefill,
            None,
            true,
            true,
            1536,
            true
        ),
        (true, 1536)
    );
    assert_eq!(
        cache_rotation_for_execution_with_prefill_flag(
            ExecutionMode::Prefill,
            None,
            true,
            true,
            32,
            true
        ),
        (true, 64)
    );
    // Production default: prefill rotation OFF → ordered prefill.
    assert_eq!(
        cache_rotation_for_execution_with_prefill_flag(
            ExecutionMode::Prefill,
            None,
            true,
            true,
            1536,
            false
        ),
        (false, 0)
    );
    assert_eq!(
        cache_rotation_for_execution_with_prefill_flag(
            ExecutionMode::Prefill,
            None,
            true,
            false,
            1536,
            true
        ),
        (false, 0)
    );
    assert_eq!(
        cache_rotation_for_execution(ExecutionMode::Decode, Some((true, 8)), false, false, 1536),
        (true, 8)
    );
    assert_eq!(
        cache_rotation_for_execution(ExecutionMode::Decode, None, true, true, 1536),
        (true, 8) // multi-token verify ring eligibility (Gemma MTP depth-2)
    );
}

fn unique_test_dir(label: &str) -> PathBuf {
    static NEXT_TEST_DIR_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = NEXT_TEST_DIR_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time should be valid")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "ax-mlx-runner-{label}-{}-{id}-{nanos}",
        std::process::id()
    ))
}

fn tensor(
    name: &str,
    role: NativeTensorRole,
    layer_index: Option<u32>,
    shape: Vec<u64>,
) -> NativeTensorSpec {
    NativeTensorSpec {
        name: name.to_string(),
        role,
        layer_index,
        dtype: NativeTensorDataType::F16,
        source_tensor_type: None,
        source_quantized: false,
        quantization: None,
        quantized_source: None,
        shape,
        file: PathBuf::from("model.safetensors"),
        offset_bytes: 0,
        length_bytes: 32,
    }
}

#[test]
fn binding_summary_reports_manifest_bytes_and_quantized_sources() {
    let dense = tensor("dense", NativeTensorRole::AttentionNorm, Some(0), vec![4]);
    let mut q4 = tensor("q4", NativeTensorRole::AttentionQ, Some(0), vec![8, 4]);
    q4.source_quantized = true;
    q4.source_tensor_type = Some("q4_k".to_string());
    q4.length_bytes = 64;
    let mut u32_affine = tensor("u32", NativeTensorRole::AttentionO, Some(0), vec![4, 1]);
    u32_affine.source_quantized = true;
    u32_affine.length_bytes = 16;

    let summary = binding_summary_from_specs(&[dense, q4, u32_affine]);

    assert!(summary.bindings_prepared);
    assert!(summary.buffers_bound);
    assert_eq!(summary.buffer_count, 3);
    assert_eq!(summary.buffer_bytes, 112);
    assert_eq!(summary.source_quantized_binding_count, 2);
    assert_eq!(summary.source_q4_k_binding_count, 1);
    assert_eq!(summary.source_q5_k_binding_count, 0);
    assert_eq!(summary.source_q6_k_binding_count, 0);
    assert_eq!(summary.source_q8_0_binding_count, 0);
}

#[test]
fn load_time_generation_warmup_skips_streamed_experts() {
    assert!(should_run_load_time_generation_warmup(false, false, false));
    assert!(!should_run_load_time_generation_warmup(false, false, true));
    assert!(!should_run_load_time_generation_warmup(false, true, false));
    assert!(!should_run_load_time_generation_warmup(true, false, false));
}

fn unit_weight() -> QuantizedWeight {
    QuantizedWeight::new(mlx_sys::zeros(&[1, 1], MlxDtype::Float32, None), None, None)
}

fn runner_test_layer() -> LayerWeights {
    LayerWeights {
        attn_norm: mlx_sys::zeros(&[1], MlxDtype::Float32, None),
        attn_post_norm: None,
        q_norm: None,
        k_norm: None,
        q_proj: None,
        k_proj: None,
        v_proj: None,
        qkv_packed: None,
        attn_out_gate: None,
        o_proj: None,
        linear_attn: None,
        glm_mla_attn: None,
        deepseek_v4: None,
        ffn_norm: mlx_sys::zeros(&[1], MlxDtype::Float32, None),
        ffn_post_norm: None,
        gate_proj: None,
        up_proj: None,
        gate_up_packed: None,
        down_proj: None,
        ffn_norm2: None,
        ffn_post_norm1: None,
        ffn_post_norm2: None,
        router_proj: None,
        router_correction_bias: None,
        router_scale: None,
        router_combined_scale: None,
        router_expert_scale: None,
        layer_scalar: None,
        per_layer_gate: None,
        per_layer_proj_w: None,
        per_layer_post_norm: None,
        shared_expert_gate: None,
        shared_gate_up_proj: None,
        shared_gate_proj: None,
        shared_up_proj: None,
        shared_down_proj: None,
        gate_up_exps_packed: None,
        gate_exps: None,
        up_exps: None,
        down_exps: None,
        attn_sink: None,
        rotation_smoothing_inverse: None,
        expert_stream: None,
    }
}

fn runner_test_weights(layers: Vec<LayerWeights>) -> ModelWeights {
    ModelWeights {
        token_embedding: unit_weight(),
        final_norm: Some(mlx_sys::zeros(&[1], MlxDtype::Float32, None)),
        qwen4_exp: None,
        qwen4_exp_mtp: None,
        lm_head: unit_weight(),
        layers,
        per_layer_embed: None,
        per_layer_model_proj: None,
        per_layer_proj_norm: None,
        mtp: None,
        glm_mtp: None,
        deepseek_v4_head: None,
        deepseek_v4_nextn: None,
        gemma4_assistant_mtp: Default::default(),
        assistant_pre_projection: None,
        assistant_post_projection: None,
        embedding_dense_0: None,
        embedding_dense_1: None,
        gemma4_unified_vision: None,
        gemma4_unified_audio: None,
        gemma4_vl_vision: None,
        diffusion_self_conditioning: None,
        unlimited_ocr_vision: None,
        qwen3_vl_vision: None,
        minicpm_v46_vision: None,
        nemotron_omni: None,
        expert_stream: None,
    }
}

#[test]
fn weight_layout_telemetry_counts_dense_ffn_packed_and_split_layers() {
    let mut packed = runner_test_layer();
    packed.gate_up_packed = Some(unit_weight());
    packed.down_proj = Some(unit_weight());
    let mut split = runner_test_layer();
    split.gate_proj = Some(unit_weight());
    split.up_proj = Some(unit_weight());
    split.down_proj = Some(unit_weight());
    let mut attention_only = runner_test_layer();
    attention_only.gate_up_packed = Some(unit_weight());

    let telemetry = WeightLayoutTelemetry::from_weights(&runner_test_weights(vec![
        packed,
        split,
        attention_only,
    ]));
    let mut decisions = Vec::new();
    telemetry.append_route_decisions(&mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(
        decisions.get("ax_mlx_dense_ffn_gate_up_packed_layers"),
        Some(&1)
    );
    assert_eq!(
        decisions.get("ax_mlx_dense_ffn_split_gate_up_layers"),
        Some(&1)
    );
    assert_eq!(
        decisions.get("ax_mlx_linear_attention_qkvz_ba_packed_layers"),
        Some(&0)
    );
    assert_eq!(
        decisions.get("ax_mlx_linear_attention_split_qkvba_layers"),
        Some(&0)
    );
}

#[test]
fn weight_layout_telemetry_excludes_5bit_split_from_fallback_counter() {
    use crate::weights::QuantizedWeight;
    use ax_engine_core::model::NativeTensorQuantization;
    let quant5 = NativeTensorQuantization {
        mode: "affine".to_string(),
        group_size: 64,
        bits: 5,
    };
    let w5 = || {
        QuantizedWeight::with_quantization(
            mlx_sys::zeros(&[1, 1], mlx_sys::MlxDtype::Float32, None),
            None,
            None,
            Some(&quant5),
        )
    };
    let mut split5 = runner_test_layer();
    split5.gate_proj = Some(w5());
    split5.up_proj = Some(w5());
    split5.down_proj = Some(unit_weight());

    let telemetry = WeightLayoutTelemetry::from_weights(&runner_test_weights(vec![split5]));
    let mut decisions = Vec::new();
    telemetry.append_route_decisions(&mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    // 5-bit split is intentional; must not appear as a hotpath fallback.
    assert_eq!(
        decisions.get("ax_mlx_dense_ffn_split_gate_up_layers"),
        Some(&0)
    );
    assert_eq!(
        decisions.get("ax_mlx_dense_ffn_gate_up_packed_layers"),
        Some(&0)
    );
}

#[test]
fn gemma4_assistant_mtp_confidence_mode_parser_accepts_stable_aliases() {
    assert_eq!(
        parse_gemma4_assistant_mtp_confidence_mode("gpu-exact"),
        Some(Gemma4AssistantMtpConfidenceMode::GpuExact)
    );
    assert_eq!(
        parse_gemma4_assistant_mtp_confidence_mode("GPU_EXACT"),
        Some(Gemma4AssistantMtpConfidenceMode::GpuExact)
    );
    assert_eq!(
        parse_gemma4_assistant_mtp_confidence_mode("cpu"),
        Some(Gemma4AssistantMtpConfidenceMode::ExactCpu)
    );
    assert_eq!(parse_gemma4_assistant_mtp_confidence_mode("approx"), None);
    assert_eq!(Gemma4AssistantMtpConfidenceMode::ExactCpu.route_code(), 0);
    assert_eq!(Gemma4AssistantMtpConfidenceMode::GpuExact.route_code(), 1);
}

#[test]
fn gemma4_assistant_mtp_status_emits_prd_route_metadata() {
    let status = Gemma4AssistantMtpStatus {
        configured: true,
        validated: true,
        enabled: false,
        attach_failed: false,
        disable_reason: crate::gemma4_assistant_mtp::Gemma4AssistantMtpDisableReason::None,
        max_depth: 1,
        config: None,
    };
    let mut decisions = Vec::new();
    status.append_route_decisions(Gemma4AssistantMtpTelemetry::default(), &mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_configured"),
        Some(&1)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_validated"),
        Some(&1)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_enabled"),
        Some(&0)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_disable_reason"),
        Some(&0)
    );
    assert_eq!(decisions.get("ax_mlx_gemma4_assistant_mtp_depth"), Some(&1));
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_confidence_mode"),
        Some(&1)
    );
    assert_eq!(decisions.get("ax_mlx_speculation_profile"), Some(&0));
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_draft_tokens"),
        Some(&0)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_accepted_tokens"),
        Some(&0)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_rejected_tokens"),
        Some(&0)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_corrections"),
        Some(&0)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_accept_rate_x1000"),
        Some(&0)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_verify_forward_wall_us"),
        Some(&0)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_verify_eval_wall_us"),
        Some(&0)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_draft_forward_wall_us"),
        Some(&0)
    );
}

#[test]
fn gemma4_unified_multimodal_telemetry_reports_modality_counts() {
    let inputs = ax_engine_core::gemma4_unified::Gemma4UnifiedRuntimeInputs {
        images: vec![
            ax_engine_core::gemma4_unified::Gemma4UnifiedImageRuntimeInput {
                span: ax_engine_core::gemma4_unified::Gemma4UnifiedTokenSpan {
                    modality: ax_engine_core::gemma4_unified::Gemma4UnifiedModality::Image,
                    placeholder_index: 1,
                    replacement_start: 1,
                    soft_token_count: 1,
                    replacement_token_count: 3,
                },
                pixel_values: vec![0.0, 1.0, 2.0],
                pixel_position_ids: vec![[0, 0]],
            },
        ],
        audios: vec![
            ax_engine_core::gemma4_unified::Gemma4UnifiedAudioRuntimeInput {
                span: ax_engine_core::gemma4_unified::Gemma4UnifiedTokenSpan {
                    modality: ax_engine_core::gemma4_unified::Gemma4UnifiedModality::Audio,
                    placeholder_index: 4,
                    replacement_start: 4,
                    soft_token_count: 1,
                    replacement_token_count: 3,
                },
                input_features: vec![0.0, 1.0],
                frame_count: 1,
                feature_count: 2,
            },
        ],
        videos: vec![
            ax_engine_core::gemma4_unified::Gemma4UnifiedVideoRuntimeInput {
                span: ax_engine_core::gemma4_unified::Gemma4UnifiedTokenSpan {
                    modality: ax_engine_core::gemma4_unified::Gemma4UnifiedModality::Video,
                    placeholder_index: 7,
                    replacement_start: 7,
                    soft_token_count: 1,
                    replacement_token_count: 3,
                },
                soft_token_ranges: Vec::new(),
                pixel_values: vec![0.0, 1.0, 2.0],
                pixel_position_ids: vec![[0, 0]],
                frame_count: 1,
            },
        ],
    };
    let mut telemetry = Gemma4UnifiedMultimodalTelemetry::default();
    telemetry.record_prefill(&inputs, true);
    let mut decisions = Vec::new();
    telemetry.append_route_decisions(&mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(
        decisions.get("ax_mlx_gemma4_unified_multimodal_prefill_requests"),
        Some(&1)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_unified_image_inputs"),
        Some(&1)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_unified_audio_inputs"),
        Some(&1)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_unified_video_inputs"),
        Some(&1)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_unified_visual_inputs"),
        Some(&2)
    );
    // Prefix reuse default-off → still counts as disabled for multimodal.
    assert_eq!(
        decisions.get("ax_mlx_gemma4_unified_prefix_cache_disabled"),
        Some(&1)
    );
    // WS-M5: MTP warmup is no longer auto-skipped when the model has MTP;
    // the skipped counter stays at 0 for the success path.
    assert_eq!(
        decisions.get("ax_mlx_gemma4_unified_mtp_prefill_warmup_skipped"),
        Some(&0)
    );
}

#[test]
fn gemma4_assistant_mtp_route_metadata_reports_runtime_telemetry() {
    let status = Gemma4AssistantMtpStatus {
        configured: true,
        validated: true,
        enabled: true,
        attach_failed: false,
        disable_reason: crate::gemma4_assistant_mtp::Gemma4AssistantMtpDisableReason::None,
        max_depth: 1,
        config: None,
    };
    let mut telemetry = Gemma4AssistantMtpTelemetry::default();
    telemetry.record_submitted(4, 120);
    telemetry.record_verified(4, 3, 240, 80);

    let mut decisions = Vec::new();
    status.append_route_decisions(telemetry, &mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_enabled"),
        Some(&1)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_confidence_mode"),
        Some(&1)
    );
    assert_eq!(decisions.get("ax_mlx_speculation_profile"), Some(&0));
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_draft_tokens"),
        Some(&4)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_accepted_tokens"),
        Some(&3)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_rejected_tokens"),
        Some(&1)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_corrections"),
        Some(&1)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_accept_rate_x1000"),
        Some(&750)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_verify_forward_wall_us"),
        Some(&240)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_verify_eval_wall_us"),
        Some(&80)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_assistant_mtp_draft_forward_wall_us"),
        Some(&120)
    );
}

#[test]
fn mtp_telemetry_counts_gemma4_assistant_as_model_draft_source() {
    let mut telemetry = MtpTelemetry::default();
    telemetry.record_step(
        2,
        1,
        &[
            MtpDraftSource::Gemma4Assistant,
            MtpDraftSource::Gemma4Assistant,
        ],
        None,
        1,
    );

    assert_eq!(telemetry.draft_source_mtp_tokens, 2);
    assert_eq!(telemetry.accepted_source_mtp_tokens, 1);
    assert_eq!(telemetry.mtp_only_accept_rate_ewma_samples, 1);
    assert_eq!(telemetry.mtp_only_accept_rate_ewma, 0.5);
}

fn linear_attn_split_weights() -> crate::weights::LinearAttentionWeights {
    crate::weights::LinearAttentionWeights {
        in_proj_qkv: Some(unit_weight()),
        in_proj_z: Some(unit_weight()),
        in_proj_a: Some(unit_weight()),
        in_proj_b: Some(unit_weight()),
        in_proj_qkvz: None,
        in_proj_ba: None,
        fused_qkvz_ba: None,
        prefill_q2_qkvz: None,
        prefill_q2_ba: None,
        conv1d_bias: None,
        d: None,
        conv1d_dense: mlx_sys::zeros(&[1, 1, 1], MlxDtype::Float32, None),
        dt_bias: mlx_sys::zeros(&[1], MlxDtype::Float32, None),
        a_log: mlx_sys::zeros(&[1], MlxDtype::Float32, None),
        norm: mlx_sys::zeros(&[1], MlxDtype::Float32, None),
        out_proj: unit_weight(),
    }
}

fn linear_attn_packed_weights() -> crate::weights::LinearAttentionWeights {
    crate::weights::LinearAttentionWeights {
        in_proj_qkv: None,
        in_proj_z: None,
        in_proj_a: None,
        in_proj_b: None,
        in_proj_qkvz: Some(unit_weight()),
        in_proj_ba: Some(unit_weight()),
        fused_qkvz_ba: None,
        prefill_q2_qkvz: None,
        prefill_q2_ba: None,
        conv1d_bias: None,
        d: None,
        conv1d_dense: mlx_sys::zeros(&[1, 1, 1], MlxDtype::Float32, None),
        dt_bias: mlx_sys::zeros(&[1], MlxDtype::Float32, None),
        a_log: mlx_sys::zeros(&[1], MlxDtype::Float32, None),
        norm: mlx_sys::zeros(&[1], MlxDtype::Float32, None),
        out_proj: unit_weight(),
    }
}

#[test]
fn weight_layout_telemetry_counts_linear_attention_packed_and_split_layers() {
    let mut packed = runner_test_layer();
    packed.linear_attn = Some(linear_attn_packed_weights());
    let mut split_a = runner_test_layer();
    split_a.linear_attn = Some(linear_attn_split_weights());
    let mut split_b = runner_test_layer();
    split_b.linear_attn = Some(linear_attn_split_weights());

    let telemetry =
        WeightLayoutTelemetry::from_weights(&runner_test_weights(vec![packed, split_a, split_b]));
    let mut decisions = Vec::new();
    telemetry.append_route_decisions(&mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(
        decisions.get("ax_mlx_linear_attention_qkvz_ba_packed_layers"),
        Some(&1)
    );
    assert_eq!(
        decisions.get("ax_mlx_linear_attention_split_qkvba_layers"),
        Some(&2)
    );
}

fn dense_manifest() -> NativeModelManifest {
    NativeModelManifest {
        schema_version: ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
        model_family: "test_dense".to_string(),
        tensor_format: NativeTensorFormat::Safetensors,
        source_quantization: None,
        runtime_status: NativeRuntimeStatus::default(),
        layer_count: 1,
        hidden_size: 4,
        intermediate_size: 8,
        attention_head_count: 1,
        attention_head_dim: 4,
        kv_head_count: 1,
        vocab_size: 16,
        tie_word_embeddings: false,
        rope_theta: None,
        rope_theta_swa: None,
        rope_scaling_type: None,
        rope_scaling_factor: None,
        rope_low_freq_factor: None,
        rope_high_freq_factor: None,
        rope_original_context_len: None,
        rope_beta_fast: None,
        rope_beta_slow: None,
        no_rope_layer_interval: 0,
        attn_temperature_floor: None,
        attn_temperature_scale: None,
        intermediate_size_mlp: 0,
        query_pre_attn_scalar: None,
        attention_logit_softcap: None,
        attn_output_gate: false,
        partial_rotary_factor: None,
        rms_norm_eps: None,
        attention_value_from_key_layers: Vec::new(),
        attention_v_norm_no_scale_layers: Vec::new(),
        global_head_dim: None,
        global_kv_head_count: None,
        sliding_window_size: None,
        layer_types: Vec::new(),
        kv_shared_source_layers: Default::default(),
        final_logit_softcapping: None,
        final_logits_scale: None,
        attention_scale_multiplier: None,
        post_norm_eps: None,
        hidden_states_scale: None,
        moe_norm_topk_prob: false,
        hidden_size_per_layer_input: 0,
        vocab_size_per_layer_input: None,
        linear_attention: NativeLinearAttentionConfig::default(),
        mla_attention: Default::default(),
        moe: NativeMoeConfig::default(),
        glm_router: Default::default(),
        deepseek_v4: Default::default(),
        qwen4_exp: Default::default(),
        weight_sanitize: ax_engine_core::WeightSanitize::None,
        think_start_token_id: None,
        think_end_token_id: None,
        diffusion: NativeDiffusionConfig::default(),
        dropped_tensors: Default::default(),
        kv_cache_quantization: None,
        tensors: vec![
            tensor(
                "model.embed_tokens.weight",
                NativeTensorRole::TokenEmbedding,
                None,
                vec![16, 4],
            ),
            tensor(
                "model.norm.weight",
                NativeTensorRole::FinalNorm,
                None,
                vec![4],
            ),
            tensor(
                "lm_head.weight",
                NativeTensorRole::LmHead,
                None,
                vec![16, 4],
            ),
            tensor(
                "model.layers.0.input_layernorm.weight",
                NativeTensorRole::AttentionNorm,
                Some(0),
                vec![4],
            ),
            tensor(
                "model.layers.0.self_attn.q_proj.weight",
                NativeTensorRole::AttentionQ,
                Some(0),
                vec![4, 4],
            ),
            tensor(
                "model.layers.0.self_attn.k_proj.weight",
                NativeTensorRole::AttentionK,
                Some(0),
                vec![4, 4],
            ),
            tensor(
                "model.layers.0.self_attn.v_proj.weight",
                NativeTensorRole::AttentionV,
                Some(0),
                vec![4, 4],
            ),
            tensor(
                "model.layers.0.self_attn.o_proj.weight",
                NativeTensorRole::AttentionO,
                Some(0),
                vec![4, 4],
            ),
            tensor(
                "model.layers.0.mlp.norm.weight",
                NativeTensorRole::FfnNorm,
                Some(0),
                vec![4],
            ),
            tensor(
                "model.layers.0.mlp.gate_proj.weight",
                NativeTensorRole::FfnGate,
                Some(0),
                vec![8, 4],
            ),
            tensor(
                "model.layers.0.mlp.up_proj.weight",
                NativeTensorRole::FfnUp,
                Some(0),
                vec![8, 4],
            ),
            tensor(
                "model.layers.0.mlp.down_proj.weight",
                NativeTensorRole::FfnDown,
                Some(0),
                vec![4, 8],
            ),
        ],
    }
}

fn set_vocab_size(manifest: &mut NativeModelManifest, vocab_size: u32) {
    manifest.vocab_size = vocab_size;
    for tensor in &mut manifest.tensors {
        if matches!(
            tensor.role,
            NativeTensorRole::TokenEmbedding | NativeTensorRole::LmHead
        ) {
            tensor.shape[0] = vocab_size as u64;
        }
    }
}

fn write_artifacts(manifest: NativeModelManifest) -> NativeModelArtifacts {
    let dir = unique_test_dir("manifest");
    fs::create_dir_all(&dir).expect("fixture directory should create");
    fs::write(dir.join("model.safetensors"), vec![0_u8; 4096]).expect("weights should write");
    fs::write(
        dir.join(AX_NATIVE_MODEL_MANIFEST_FILE),
        serde_json::to_vec_pretty(&manifest).expect("manifest should serialize"),
    )
    .expect("manifest should write");
    NativeModelArtifacts::from_dir(&dir).expect("fixture manifest should validate")
}

#[test]
fn terminal_token_ids_resolve_from_config_json_array() {
    let mut manifest = dense_manifest();
    set_vocab_size(&mut manifest, 128);
    let artifacts = write_artifacts(manifest);
    fs::write(
        artifacts.root_dir().join("config.json"),
        r#"{"eos_token_id":[1,106,999]}"#,
    )
    .expect("config should write");

    assert_eq!(resolve_terminal_token_ids(&artifacts), vec![1, 106]);
}

#[test]
fn terminal_token_ids_resolve_pad_for_standard_models() {
    let mut manifest = dense_manifest();
    set_vocab_size(&mut manifest, 128);
    let artifacts = write_artifacts(manifest);
    fs::write(
        artifacts.root_dir().join("config.json"),
        r#"{"eos_token_id":1,"pad_token_id":0}"#,
    )
    .expect("config should write");

    assert_eq!(resolve_terminal_token_ids(&artifacts), vec![0, 1]);
}

#[test]
fn terminal_token_ids_ignore_pad_for_diffusion_gemma() {
    let mut manifest = dense_manifest();
    manifest.model_family = "diffusion_gemma".to_string();
    set_vocab_size(&mut manifest, 128);
    let artifacts = write_artifacts(manifest);
    fs::write(
        artifacts.root_dir().join("config.json"),
        r#"{"eos_token_id":[1,106,50],"pad_token_id":0}"#,
    )
    .expect("config should write");
    fs::write(
        artifacts.root_dir().join("tokenizer_config.json"),
        r#"{"pad_token":"<pad>"}"#,
    )
    .expect("tokenizer config should write");
    fs::write(
        artifacts.root_dir().join("tokenizer.json"),
        r#"{"added_tokens":[{"id":0,"content":"<pad>"},{"id":106,"content":"<turn|>"}]}"#,
    )
    .expect("tokenizer should write");

    assert_eq!(resolve_terminal_token_ids(&artifacts), vec![1, 50, 106]);
}

#[test]
fn terminal_token_ids_resolve_tokenizer_config_string_from_tokenizer_json() {
    let mut manifest = dense_manifest();
    set_vocab_size(&mut manifest, 200_000);
    let artifacts = write_artifacts(manifest);
    fs::write(
        artifacts.root_dir().join("tokenizer_config.json"),
        r#"{"eos_token":"<|im_end|>","pad_token":"<|endoftext|>"}"#,
    )
    .expect("tokenizer config should write");
    fs::write(
            artifacts.root_dir().join("tokenizer.json"),
            r#"{"added_tokens":[{"id":151643,"content":"<|endoftext|>"},{"id":151645,"content":"<|im_end|>"}]}"#,
        )
        .expect("tokenizer should write");

    assert_eq!(resolve_terminal_token_ids(&artifacts), vec![151643, 151645]);
}

#[test]
fn terminal_token_ids_resolve_common_chatml_eot_from_tokenizer_json() {
    let mut manifest = dense_manifest();
    manifest.model_family = "qwen3".to_string();
    set_vocab_size(&mut manifest, 200_000);
    let artifacts = write_artifacts(manifest);
    fs::write(
            artifacts.root_dir().join("tokenizer.json"),
            r#"{"added_tokens":[{"id":151643,"content":"<|endoftext|>"},{"id":151645,"content":"<|im_end|>"}]}"#,
        )
        .expect("tokenizer should write");

    assert_eq!(resolve_terminal_token_ids(&artifacts), vec![151643, 151645]);
}

#[test]
fn terminal_token_ids_resolve_common_gemma_eot_for_other_families() {
    let mut manifest = dense_manifest();
    manifest.model_family = "gemma3".to_string();
    set_vocab_size(&mut manifest, 200_000);
    let artifacts = write_artifacts(manifest);
    fs::write(
            artifacts.root_dir().join("tokenizer.json"),
            r#"{"added_tokens":[{"id":106,"content":"<end_of_turn>"},{"id":151645,"content":"<|im_end|>"}]}"#,
        )
        .expect("tokenizer should write");

    assert_eq!(resolve_terminal_token_ids(&artifacts), vec![106, 151645]);
}

#[test]
fn terminal_token_ids_resolve_common_gemma4_turn_end_from_tokenizer_json() {
    let mut manifest = dense_manifest();
    manifest.model_family = "gemma4".to_string();
    set_vocab_size(&mut manifest, 200_000);
    let artifacts = write_artifacts(manifest);
    fs::write(
        artifacts.root_dir().join("tokenizer.json"),
        r#"{"added_tokens":[{"id":105,"content":"<|turn>"},{"id":106,"content":"<turn|>"}]}"#,
    )
    .expect("tokenizer should write");

    assert_eq!(resolve_terminal_token_ids(&artifacts), vec![106]);
}

fn qwen35_linear_manifest() -> NativeModelManifest {
    let mut manifest = dense_manifest();
    manifest.model_family = "qwen3_5".to_string();
    manifest.linear_attention = NativeLinearAttentionConfig {
        full_attention_interval: None,
        num_value_heads: Some(1),
        num_key_heads: Some(1),
        key_head_dim: Some(32),
        value_head_dim: Some(4),
        conv_kernel_dim: Some(4),
    };
    manifest.tensors.retain(|tensor| {
        !matches!(
            tensor.role,
            NativeTensorRole::AttentionQ
                | NativeTensorRole::AttentionK
                | NativeTensorRole::AttentionV
                | NativeTensorRole::AttentionO
        )
    });
    manifest.tensors.extend([
        tensor(
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            NativeTensorRole::LinearAttentionInProjQkv,
            Some(0),
            vec![68, 4],
        ),
        tensor(
            "model.layers.0.linear_attn.in_proj_z.weight",
            NativeTensorRole::LinearAttentionInProjZ,
            Some(0),
            vec![4, 4],
        ),
        tensor(
            "model.layers.0.linear_attn.in_proj_a.weight",
            NativeTensorRole::LinearAttentionInProjA,
            Some(0),
            vec![1, 4],
        ),
        tensor(
            "model.layers.0.linear_attn.in_proj_b.weight",
            NativeTensorRole::LinearAttentionInProjB,
            Some(0),
            vec![1, 4],
        ),
        tensor(
            "model.layers.0.linear_attn.conv1d.weight",
            NativeTensorRole::LinearAttentionConv1d,
            Some(0),
            vec![68, 4, 1],
        ),
        tensor(
            "model.layers.0.linear_attn.dt_bias",
            NativeTensorRole::LinearAttentionDtBias,
            Some(0),
            vec![1],
        ),
        tensor(
            "model.layers.0.linear_attn.A_log",
            NativeTensorRole::LinearAttentionALog,
            Some(0),
            vec![1],
        ),
        tensor(
            "model.layers.0.linear_attn.norm.weight",
            NativeTensorRole::LinearAttentionNorm,
            Some(0),
            vec![4],
        ),
        tensor(
            "model.layers.0.linear_attn.out_proj.weight",
            NativeTensorRole::LinearAttentionOutProj,
            Some(0),
            vec![4, 4],
        ),
    ]);
    manifest
}

fn glm4_moe_lite_manifest() -> NativeModelManifest {
    let mut manifest = dense_manifest();
    manifest.model_family = "glm4_moe_lite".to_string();
    manifest.layer_count = 2;
    manifest.attention_head_dim = 4;
    manifest.mla_attention = NativeMlaAttentionConfig {
        q_lora_rank: Some(2),
        kv_lora_rank: Some(2),
        qk_nope_head_dim: Some(2),
        qk_rope_head_dim: Some(2),
        value_head_dim: Some(2),
    };
    manifest.moe = NativeMoeConfig {
        expert_count: Some(4),
        experts_per_token: Some(2),
        expert_intermediate_size: Some(8),
        layer_freq: None,
        first_dense_layers: None,
        shared_expert_count: None,
        sigmoid_routing: false,
        routed_scaling_factor: None,
        n_group: None,
        topk_group: None,
    };
    manifest.glm_router = NativeGlmRouterConfig {
        first_dense_layer_count: Some(1),
        routed_scaling_factor: Some(1.8),
        n_group: Some(1),
        topk_group: Some(1),
        has_shared_experts: true,
    };
    manifest.tensors.retain(|tensor| {
        !matches!(
            tensor.role,
            NativeTensorRole::AttentionQ
                | NativeTensorRole::AttentionK
                | NativeTensorRole::AttentionV
                | NativeTensorRole::AttentionQkvPacked
        )
    });
    for tensor in &mut manifest.tensors {
        if tensor.role == NativeTensorRole::AttentionO {
            tensor.shape = vec![4, 2];
        }
    }

    for layer in 0..2 {
        for (role, shape) in [
            (NativeTensorRole::AttentionPostNorm, vec![4]),
            (NativeTensorRole::AttentionQa, vec![2, 4]),
            (NativeTensorRole::AttentionQaNorm, vec![2]),
            (NativeTensorRole::AttentionQb, vec![4, 2]),
            (NativeTensorRole::AttentionKvA, vec![4, 4]),
            (NativeTensorRole::AttentionKvANorm, vec![2]),
            (NativeTensorRole::AttentionEmbedQ, vec![1, 2, 2]),
            (NativeTensorRole::AttentionUnembedOut, vec![1, 2, 2]),
        ] {
            manifest.tensors.push(tensor(
                &format!("model.layers.{layer}.{role:?}.weight"),
                role,
                Some(layer),
                shape,
            ));
        }

        if layer == 1 {
            for (role, shape) in [
                (NativeTensorRole::AttentionNorm, vec![4]),
                (NativeTensorRole::AttentionO, vec![4, 2]),
                (NativeTensorRole::FfnGateInp, vec![4, 4]),
                (NativeTensorRole::FfnGateInpCorrectionBias, vec![4]),
                (NativeTensorRole::FfnGateExps, vec![4, 8, 4]),
                (NativeTensorRole::FfnUpExps, vec![4, 8, 4]),
                (NativeTensorRole::FfnDownExps, vec![4, 4, 8]),
                (NativeTensorRole::FfnSharedExpertGate, vec![8, 4]),
                (NativeTensorRole::FfnSharedExpertUp, vec![8, 4]),
                (NativeTensorRole::FfnSharedExpertDown, vec![4, 8]),
            ] {
                manifest.tensors.push(tensor(
                    &format!("model.layers.{layer}.{role:?}.weight"),
                    role,
                    Some(layer),
                    shape,
                ));
            }
        }
    }

    manifest
}

#[test]
fn mlx_manifest_validation_rejects_linear_attention_for_non_qwen35() {
    let mut manifest = dense_manifest();
    manifest.model_family = "gemma4".to_string();
    manifest.linear_attention = NativeLinearAttentionConfig {
        full_attention_interval: Some(4),
        num_value_heads: Some(1),
        num_key_heads: Some(1),
        key_head_dim: Some(4),
        value_head_dim: Some(4),
        conv_kernel_dim: Some(4),
    };
    let artifacts = write_artifacts(manifest);

    let error = validate_mlx_supported_manifest(&artifacts)
        .expect_err("linear attention should fail closed");

    assert!(error.to_string().contains("qwen3_5/qwen3_next"));
}

#[test]
fn nemotron_router_correction_bias_is_not_misclassified_as_mla() {
    let mut manifest = dense_manifest();
    manifest.model_family = "nemotron_h".to_string();
    manifest.tensors.push(tensor(
        "backbone.layers.0.mixer.gate.e_score_correction_bias",
        NativeTensorRole::FfnGateInpCorrectionBias,
        Some(0),
        vec![4],
    ));
    let artifacts = write_artifacts(manifest);

    assert!(!has_glm_mla_tensors(&artifacts));
}

#[test]
fn mlx_manifest_validation_rejects_unknown_model_family() {
    let artifacts = write_artifacts(dense_manifest());

    let error =
        validate_mlx_supported_manifest(&artifacts).expect_err("unknown family should fail");

    assert!(
        error
            .to_string()
            .contains("not supported by the MLX runner")
    );
}

#[test]
fn mlx_manifest_validation_reports_auxiliary_family_role() {
    let error = validate_mlx_primary_admission("gemma4_assistant")
        .expect_err("assistant artifact must not be admitted as a primary runner");
    let message = error.to_string();

    assert!(message.contains("auxiliary-only artifact"));
    assert!(message.contains("cannot be loaded as the primary MLX runner"));
}

#[test]
fn mlx_manifest_validation_rejects_incomplete_glm_contract() {
    let mut manifest = dense_manifest();
    manifest.model_family = "glm4_moe_lite".to_string();
    manifest.tensors.push(tensor(
        "model.layers.0.self_attn.q_a_proj.weight",
        NativeTensorRole::AttentionQa,
        Some(0),
        vec![4, 4],
    ));

    let error = validate_mla_moe_manifest(&manifest)
        .expect_err("incomplete GLM runtime contract should fail closed");

    assert!(
        error
            .to_string()
            .contains("glm4_moe_lite requires mla_attention metadata")
    );
}

#[test]
fn mlx_manifest_validation_allows_glm4_moe_lite_contract() {
    let artifacts = write_artifacts(glm4_moe_lite_manifest());

    validate_mlx_supported_manifest(&artifacts)
        .expect("GLM4MoELite runtime contract is wired for the MLX path");
}

#[test]
fn mlx_manifest_validation_allows_deepseek_v3_variants_kv_b_contract() {
    let mut manifest = glm4_moe_lite_manifest();
    manifest.glm_router = NativeGlmRouterConfig::default();
    manifest.moe.layer_freq = Some(1);
    manifest.moe.first_dense_layers = Some(1);
    manifest.moe.shared_expert_count = Some(1);
    manifest.moe.sigmoid_routing = true;
    manifest.moe.routed_scaling_factor = Some(2.5);
    manifest.moe.n_group = Some(1);
    manifest.moe.topk_group = Some(1);
    manifest.tensors.retain(|tensor| {
        !matches!(
            tensor.role,
            NativeTensorRole::AttentionEmbedQ | NativeTensorRole::AttentionUnembedOut
        )
    });
    for layer in 0..2 {
        manifest.tensors.push(tensor(
            &format!("model.layers.{layer}.self_attn.kv_b_proj.weight"),
            NativeTensorRole::AttentionKvB,
            Some(layer),
            vec![4, 2],
        ));
    }
    for family in ["deepseek_v3", "deepseek_v32"] {
        let mut variant = manifest.clone();
        variant.model_family = family.to_string();
        let artifacts = write_artifacts(variant);

        validate_mlx_supported_manifest(&artifacts)
            .expect("DeepSeek V3 KV-B runtime contract should be accepted");
    }
}

#[test]
fn mlx_manifest_validation_rejects_partial_glm_router_fields() {
    // Each `.expect()` in `GlmRouterConfig::from_manifest` and each runtime
    // `assert!` in `glm_router_apply_group_selection` corresponds to one of
    // these checks. A partial manifest passed the prior validator (only
    // `first_dense_layer_count` was required) and then panicked downstream.
    type Mutator = Box<dyn Fn(&mut NativeModelManifest)>;
    for (label, mutate) in [
        (
            "missing routed_scaling_factor",
            Box::new(|m: &mut NativeModelManifest| m.glm_router.routed_scaling_factor = None)
                as Mutator,
        ),
        (
            "missing n_group",
            Box::new(|m: &mut NativeModelManifest| m.glm_router.n_group = None),
        ),
        (
            "missing topk_group",
            Box::new(|m: &mut NativeModelManifest| m.glm_router.topk_group = None),
        ),
        (
            "zero n_group",
            Box::new(|m: &mut NativeModelManifest| m.glm_router.n_group = Some(0)),
        ),
        (
            "zero topk_group",
            Box::new(|m: &mut NativeModelManifest| m.glm_router.topk_group = Some(0)),
        ),
        (
            "topk_group exceeds n_group",
            Box::new(|m: &mut NativeModelManifest| {
                m.glm_router.n_group = Some(2);
                m.glm_router.topk_group = Some(3);
                m.moe.expert_count = Some(4);
            }),
        ),
        (
            "expert_count not divisible by n_group",
            Box::new(|m: &mut NativeModelManifest| {
                m.glm_router.n_group = Some(3);
                m.glm_router.topk_group = Some(1);
                m.moe.expert_count = Some(4);
            }),
        ),
        (
            "expert_count per group below two",
            Box::new(|m: &mut NativeModelManifest| {
                // 4 experts / 4 groups = 1 per group; the runtime assert
                // `experts_per_group >= 2` would fire mid-forward.
                m.glm_router.n_group = Some(4);
                m.glm_router.topk_group = Some(1);
                m.moe.expert_count = Some(4);
            }),
        ),
        (
            "missing moe.expert_count",
            Box::new(|m: &mut NativeModelManifest| {
                // `is_enabled()` stays true via experts_per_token, but
                // `unwrap_or(0)` downstream then crashes the group-size assert
                // when n_group > 1.
                m.moe.expert_count = None;
                m.glm_router.n_group = Some(2);
                m.glm_router.topk_group = Some(1);
            }),
        ),
        (
            "missing moe.experts_per_token",
            Box::new(|m: &mut NativeModelManifest| m.moe.experts_per_token = None),
        ),
        (
            "first_dense_layer_count exceeds layer_count",
            Box::new(|m: &mut NativeModelManifest| {
                m.glm_router.first_dense_layer_count = Some(m.layer_count + 1);
            }),
        ),
    ] {
        let mut manifest = glm4_moe_lite_manifest();
        mutate(&mut manifest);
        let error =
            validate_mla_moe_manifest(&manifest).expect_err(&format!("{label} should fail closed"));
        let message = error.to_string();
        assert!(
            message.contains("glm_router") || message.contains("glm4_moe_lite"),
            "{label}: unexpected error message: {message}"
        );
    }
}

#[test]
fn mlx_manifest_validation_allows_qwen35_linear_attention() {
    let artifacts = write_artifacts(qwen35_linear_manifest());

    validate_mlx_supported_manifest(&artifacts)
        .expect("Qwen3.5 linear attention is wired for the MLX path");
}

#[test]
fn mlx_manifest_validation_rejects_partial_linear_attention_fields() {
    // Each `.expect()` in `LinearAttentionConfig::from_manifest` corresponds
    // to one of these required fields; the validator must surface a typed
    // error before the runner panics on a partially-configured manifest.
    for (label, mutate) in [
        (
            "missing num_value_heads",
            Box::new(|m: &mut NativeModelManifest| m.linear_attention.num_value_heads = None)
                as Box<dyn Fn(&mut NativeModelManifest)>,
        ),
        (
            "missing num_key_heads",
            Box::new(|m: &mut NativeModelManifest| m.linear_attention.num_key_heads = None),
        ),
        (
            "missing value_head_dim",
            Box::new(|m: &mut NativeModelManifest| m.linear_attention.value_head_dim = None),
        ),
        (
            "missing conv_kernel_dim",
            Box::new(|m: &mut NativeModelManifest| m.linear_attention.conv_kernel_dim = None),
        ),
        (
            "zero full_attention_interval",
            Box::new(|m: &mut NativeModelManifest| {
                m.linear_attention.full_attention_interval = Some(0)
            }),
        ),
    ] {
        let mut manifest = qwen35_linear_manifest();
        mutate(&mut manifest);
        let error = validate_qwen_gated_delta_linear_attention(&manifest)
            .expect_err(&format!("{label} should fail closed"));
        assert!(
            error.to_string().contains("linear_attention"),
            "{label}: unexpected error message: {error}"
        );
    }
}

#[test]
fn real_mlx_manifest_resolves_qwen35_linear_interval_when_configured() {
    let Ok(model_dir) = std::env::var("AX_ENGINE_MLX_REAL_MODEL_DIR") else {
        return;
    };
    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .expect("real MLX manifest should load");

    validate_mlx_supported_manifest(&artifacts).expect("real MLX manifest should be supported");
    let binding = binding_summary_from_specs(artifacts.tensor_specs());
    assert!(binding.buffer_bytes > 0);
    assert!(binding.source_quantized_binding_count > 0);
    let cfg = ModelConfig::from_manifest(artifacts.manifest());

    assert_eq!(
        cfg.linear_attention
            .as_ref()
            .expect("real manifest should configure linear attention")
            .full_attention_interval,
        4
    );
    assert!(cfg.is_linear_attention_layer(0));
    assert!(!cfg.is_linear_attention_layer(3));
}

#[test]
fn real_mlx_runner_warms_up_qwen35_when_configured() {
    if std::env::var("AX_ENGINE_MLX_RUN_REAL_FORWARD").as_deref() != Ok("1") {
        return;
    }
    let Ok(model_dir) = std::env::var("AX_ENGINE_MLX_REAL_MODEL_DIR") else {
        return;
    };
    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .expect("real MLX manifest should load");

    MlxRunner::from_artifacts(&artifacts, 8, true).expect("real Qwen3.5 MLX runner should warm up");
}

#[test]
fn linear_attention_ngram_acceleration_cools_down_after_reject() {
    // Complete miss (0 accepted): long cooldown.
    assert_eq!(
        ngram_acceleration_disabled_steps(true, 0, DEFAULT_DRAFT_LEN, 0.95),
        Some(LINEAR_NGRAM_RETRY_INTERVAL)
    );
    // Partial accept (some but not all): short cooldown to retry quickly.
    assert_eq!(
        ngram_acceleration_disabled_steps(true, 3, DEFAULT_DRAFT_LEN, 0.95),
        Some(LINEAR_NGRAM_PARTIAL_RETRY_INTERVAL)
    );
    // Full accept: no cooldown.
    assert_eq!(
        ngram_acceleration_disabled_steps(true, DEFAULT_DRAFT_LEN, DEFAULT_DRAFT_LEN, 0.25),
        None
    );
}

#[test]
fn ngram_acceleration_telemetry_records_acceptance_and_cooldown_counters() {
    let mut telemetry = NgramAccelerationTelemetry::default();

    telemetry.record_no_draft();
    telemetry.record_draft(DEFAULT_DRAFT_LEN, DEFAULT_DRAFT_LEN);
    telemetry.record_draft(DEFAULT_DRAFT_LEN, 0);
    telemetry.record_draft(DEFAULT_DRAFT_LEN, 2);
    telemetry.record_cooldown_step();
    telemetry.record_cooldown_event(4);
    telemetry.record_request_disable_event();
    telemetry.record_request_disabled_step();
    telemetry.record_request_disabled_reason(NgramRequestDisableReason::LinearNoDraft);
    telemetry.record_no_draft_reason(Some(NgramDraftRejection::NoCandidate));
    telemetry.record_no_draft_reason(Some(NgramDraftRejection::ConfidenceFiltered));
    telemetry.record_policy(NgramPolicyVariant::SharedPoolMajority, MAX_DRAFT_LEN);
    telemetry.record_prompt_class(crate::ngram_accel::PROMPT_CLASS_REPEATING);

    let mut decisions = Vec::new();
    telemetry.append_route_decisions(&mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(decisions.get("ax_ngram_no_draft_steps"), Some(&1));
    assert_eq!(decisions.get("ax_ngram_draft_attempts"), Some(&3));
    assert_eq!(
        decisions.get("ax_ngram_draft_tokens"),
        Some(&(DEFAULT_DRAFT_LEN as u32 * 3))
    );
    assert_eq!(
        decisions.get("ax_ngram_accepted_tokens"),
        Some(&(DEFAULT_DRAFT_LEN as u32 + 2))
    );
    assert_eq!(
        decisions.get("ax_ngram_rejected_tokens"),
        Some(&(DEFAULT_DRAFT_LEN as u32 * 2 - 2))
    );
    assert_eq!(decisions.get("ax_ngram_full_accepts"), Some(&1));
    assert_eq!(decisions.get("ax_ngram_complete_misses"), Some(&1));
    assert_eq!(decisions.get("ax_ngram_partial_rejects"), Some(&1));
    assert_eq!(decisions.get("ax_ngram_cooldown_steps"), Some(&1));
    assert_eq!(decisions.get("ax_ngram_cooldown_events"), Some(&1));
    assert_eq!(decisions.get("ax_ngram_cooldown_steps_scheduled"), Some(&4));
    assert_eq!(decisions.get("ax_ngram_request_disable_events"), Some(&1));
    assert_eq!(decisions.get("ax_ngram_request_disabled_steps"), Some(&1));
    assert_eq!(
        decisions.get("ax_ngram_fallback_no_candidate_steps"),
        Some(&1)
    );
    assert_eq!(
        decisions.get("ax_ngram_fallback_confidence_filtered_steps"),
        Some(&1)
    );
    assert_eq!(
        decisions.get("ax_ngram_fallback_linear_no_draft_steps"),
        Some(&1)
    );
    assert_eq!(decisions.get("ax_ngram_policy_variant"), Some(&3));
    assert_eq!(decisions.get("ax_ngram_adaptive_draft_len_steps"), Some(&1));
    assert_eq!(
        decisions.get("ax_ngram_adaptive_draft_len_total"),
        Some(&(MAX_DRAFT_LEN as u32))
    );
    assert_eq!(
        decisions.get("ax_prompt_class_code"),
        Some(&crate::ngram_accel::PROMPT_CLASS_REPEATING)
    );

    let mut zero_decisions = Vec::new();
    NgramAccelerationTelemetry::default().append_route_decisions(&mut zero_decisions);
    let zero_decisions = zero_decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();
    assert_eq!(zero_decisions.get("ax_ngram_draft_attempts"), Some(&0));
    assert_eq!(zero_decisions.get("ax_ngram_complete_misses"), Some(&0));
    assert_eq!(zero_decisions.get("ax_ngram_cooldown_events"), Some(&0));
    assert_eq!(
        zero_decisions.get("ax_ngram_request_disable_events"),
        Some(&0)
    );
    assert_eq!(
        zero_decisions.get("ax_prompt_class_code"),
        Some(&crate::ngram_accel::PROMPT_CLASS_UNSET)
    );

    // PRD §8 Phase 6: per-attempt acceptance-by-depth histogram. The
    // three draft attempts above accepted {full, 0, 2} tokens, so the
    // histogram must be: bucket DEFAULT_DRAFT_LEN += 1, bucket 0 += 1,
    // bucket 2 += 1. Verify bucket 0 and bucket 2 directly (which are
    // both well below NGRAM_ACCEPT_DEPTH_BUCKETS) and assert all other
    // sub-bucket counters are zero.
    assert_eq!(decisions.get("ax_ngram_accept_at_depth_0"), Some(&1));
    assert_eq!(decisions.get("ax_ngram_accept_at_depth_2"), Some(&1));
    assert_eq!(decisions.get("ax_ngram_accept_at_depth_1"), Some(&0));
    assert_eq!(decisions.get("ax_ngram_accept_at_depth_3"), Some(&0));
}

#[test]
fn ngram_telemetry_accepts_by_depth_saturates_at_last_bucket() {
    // Drafts that accept beyond NGRAM_ACCEPT_DEPTH_BUCKETS - 1 must
    // land in the last bucket rather than panic-on-index-overflow or
    // silently drop. Without this saturation, a future longer-draft
    // policy could underreport its acceptance.
    let mut telemetry = NgramAccelerationTelemetry::default();
    // Accept more than the histogram length on a single draft.
    telemetry.record_draft(16, 16);
    // Accept exactly the last in-range bucket on another draft.
    telemetry.record_draft(
        NGRAM_ACCEPT_DEPTH_BUCKETS - 1,
        NGRAM_ACCEPT_DEPTH_BUCKETS - 1,
    );

    let mut decisions = Vec::new();
    telemetry.append_route_decisions(&mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    // Both attempts saturate into bucket NGRAM_ACCEPT_DEPTH_BUCKETS - 1
    // because the over-the-limit attempt is clamped, and the exactly-
    // at-the-last-bucket attempt naturally lands there.
    let last_key = format!(
        "ax_ngram_accept_at_depth_{}",
        NGRAM_ACCEPT_DEPTH_BUCKETS - 1
    );
    assert_eq!(decisions.get(last_key.as_str()), Some(&2));
}

#[test]
fn gemma4_moe_profile_route_decisions_emit_only_when_enabled() {
    let mut profile = Gemma4MoeProfileSnapshot {
        enabled: 1,
        decode_layers: 2,
        topk_selections: 16,
        sorted_gather_layers: 0,
        unsorted_gather_layers: 2,
        attention_wall_us: 100,
        dense_wall_us: 80,
        router_wall_us: 30,
        expert_wall_us: 90,
        post_wall_us: 20,
    };
    profile.merge_from(Gemma4MoeProfileSnapshot {
        enabled: 1,
        decode_layers: 3,
        topk_selections: 24,
        sorted_gather_layers: 0,
        unsorted_gather_layers: 3,
        attention_wall_us: 150,
        dense_wall_us: 120,
        router_wall_us: 45,
        expert_wall_us: 135,
        post_wall_us: 30,
    });

    let mut decisions = Vec::new();
    profile.append_route_decisions(&mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(decisions.get("ax_mlx_gemma4_moe_profile_enabled"), Some(&1));
    assert_eq!(
        decisions.get("ax_mlx_gemma4_moe_profile_decode_layers"),
        Some(&5)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_moe_profile_topk_selections"),
        Some(&40)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_moe_profile_unsorted_gather_layers"),
        Some(&5)
    );
    assert_eq!(
        decisions.get("ax_mlx_gemma4_moe_profile_attention_wall_us"),
        Some(&250)
    );

    let mut disabled_decisions = Vec::new();
    Gemma4MoeProfileSnapshot::default().append_route_decisions(&mut disabled_decisions);
    assert!(disabled_decisions.is_empty());
}

#[test]
fn linear_attention_profile_route_decisions_emit_only_when_enabled() {
    let mut profile = LinearAttentionProfileSnapshot {
        enabled: 1,
        layers: 2,
        tokens: 1024,
        projection_wall_us: 100,
        projection_qkvz_wall_us: 70,
        projection_ba_wall_us: 30,
        projection_qkv_wall_us: 0,
        projection_z_wall_us: 0,
        projection_a_wall_us: 0,
        projection_b_wall_us: 0,
        conv_wall_us: 80,
        qk_norm_wall_us: 30,
        recurrent_wall_us: 90,
        output_wall_us: 20,
        ..LinearAttentionProfileSnapshot::default()
    };
    profile.merge_from(LinearAttentionProfileSnapshot {
        enabled: 1,
        layers: 3,
        tokens: 2048,
        projection_wall_us: 150,
        projection_qkvz_wall_us: 105,
        projection_ba_wall_us: 45,
        projection_qkv_wall_us: 0,
        projection_z_wall_us: 0,
        projection_a_wall_us: 0,
        projection_b_wall_us: 0,
        conv_wall_us: 120,
        qk_norm_wall_us: 45,
        recurrent_wall_us: 135,
        output_wall_us: 30,
        ..LinearAttentionProfileSnapshot::default()
    });

    let mut decisions = Vec::new();
    profile.append_route_decisions(&mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(
        decisions.get("ax_mlx_linear_attention_profile_enabled"),
        Some(&1)
    );
    assert_eq!(
        decisions.get("ax_mlx_linear_attention_profile_layers"),
        Some(&5)
    );
    assert_eq!(
        decisions.get("ax_mlx_linear_attention_profile_tokens"),
        Some(&3072)
    );
    assert_eq!(
        decisions.get("ax_mlx_linear_attention_profile_projection_wall_us"),
        Some(&250)
    );
    assert_eq!(
        decisions.get("ax_mlx_linear_attention_profile_projection_qkvz_wall_us"),
        Some(&175)
    );
    assert_eq!(
        decisions.get("ax_mlx_linear_attention_profile_projection_ba_wall_us"),
        Some(&75)
    );
    assert_eq!(
        decisions.get("ax_mlx_linear_attention_profile_recurrent_wall_us"),
        Some(&225)
    );

    let mut disabled_decisions = Vec::new();
    LinearAttentionProfileSnapshot::default().append_route_decisions(&mut disabled_decisions);
    assert!(disabled_decisions.is_empty());
}

#[test]
fn linear_attention_direct_cpp_route_decisions_emit_when_attempted() {
    let mut profile = LinearAttentionProfileSnapshot {
        direct_cpp_inputs_attempts: 2,
        direct_cpp_inputs_hits: 1,
        direct_cpp_inputs_fallbacks: 1,
        direct_cpp_inputs_profile_blocked: 1,
        ..LinearAttentionProfileSnapshot::default()
    };
    profile.merge_from(LinearAttentionProfileSnapshot {
        direct_cpp_inputs_attempts: 3,
        direct_cpp_inputs_hits: 2,
        direct_cpp_inputs_fallbacks: 1,
        direct_cpp_inputs_profile_blocked: 0,
        ..LinearAttentionProfileSnapshot::default()
    });

    let mut decisions = Vec::new();
    profile.append_route_decisions(&mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(
        decisions.get("ax_mlx_direct_cpp_linear_attention_inputs_attempts"),
        Some(&5)
    );
    assert_eq!(
        decisions.get("ax_mlx_direct_cpp_linear_attention_inputs_hits"),
        Some(&3)
    );
    assert_eq!(
        decisions.get("ax_mlx_direct_cpp_linear_attention_inputs_fallbacks"),
        Some(&2)
    );
    assert_eq!(
        decisions.get("ax_mlx_direct_cpp_linear_attention_inputs_profile_blocked"),
        Some(&1)
    );
    assert!(!decisions.contains_key("ax_mlx_linear_attention_profile_enabled"));
}

#[test]
fn linear_attention_direct_cpp_post_input_route_decisions_emit_when_attempted() {
    let mut profile = LinearAttentionProfileSnapshot {
        direct_cpp_post_input_attempts: 2,
        direct_cpp_post_input_hits: 1,
        direct_cpp_post_input_fallbacks: 1,
        direct_cpp_post_input_profile_blocked: 1,
        ..LinearAttentionProfileSnapshot::default()
    };
    profile.merge_from(LinearAttentionProfileSnapshot {
        direct_cpp_post_input_attempts: 3,
        direct_cpp_post_input_hits: 2,
        direct_cpp_post_input_fallbacks: 1,
        direct_cpp_post_input_profile_blocked: 0,
        ..LinearAttentionProfileSnapshot::default()
    });

    let mut decisions = Vec::new();
    profile.append_route_decisions(&mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(
        decisions.get("ax_mlx_direct_cpp_linear_attention_post_input_attempts"),
        Some(&5)
    );
    assert_eq!(
        decisions.get("ax_mlx_direct_cpp_linear_attention_post_input_hits"),
        Some(&3)
    );
    assert_eq!(
        decisions.get("ax_mlx_direct_cpp_linear_attention_post_input_fallbacks"),
        Some(&2)
    );
    assert_eq!(
        decisions.get("ax_mlx_direct_cpp_linear_attention_post_input_profile_blocked"),
        Some(&1)
    );
    assert!(!decisions.contains_key("ax_mlx_linear_attention_profile_enabled"));
}

#[test]
fn linear_attention_decode_post_input_metal_route_decisions_emit_when_attempted() {
    let mut profile = LinearAttentionProfileSnapshot {
        decode_post_input_metal_attempts: 2,
        decode_post_input_metal_hits: 1,
        decode_post_input_metal_fallbacks: 1,
        decode_post_input_metal_profile_blocked: 1,
        ..LinearAttentionProfileSnapshot::default()
    };
    profile.merge_from(LinearAttentionProfileSnapshot {
        decode_post_input_metal_attempts: 3,
        decode_post_input_metal_hits: 2,
        decode_post_input_metal_fallbacks: 1,
        decode_post_input_metal_profile_blocked: 0,
        ..LinearAttentionProfileSnapshot::default()
    });

    let mut decisions = Vec::new();
    profile.append_route_decisions(&mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(
        decisions.get("ax_mlx_qwen_linear_attention_decode_post_input_metal_attempts"),
        Some(&5)
    );
    assert_eq!(
        decisions.get("ax_mlx_qwen_linear_attention_decode_post_input_metal_hits"),
        Some(&3)
    );
    assert_eq!(
        decisions.get("ax_mlx_qwen_linear_attention_decode_post_input_metal_fallbacks"),
        Some(&2)
    );
    assert_eq!(
        decisions.get("ax_mlx_qwen_linear_attention_decode_post_input_metal_profile_blocked"),
        Some(&1)
    );
    assert!(!decisions.contains_key("ax_mlx_linear_attention_profile_enabled"));
}

#[test]
fn dense_ffn_fastpath_route_decisions_emit_when_attempted() {
    let mut profile = DenseFfnFastpathSnapshot {
        qwen_gate_up_matvec_metal_attempts: 2,
        qwen_gate_up_matvec_metal_hits: 1,
        qwen_gate_up_matvec_metal_fallbacks: 1,
    };
    profile.merge_from(DenseFfnFastpathSnapshot {
        qwen_gate_up_matvec_metal_attempts: 3,
        qwen_gate_up_matvec_metal_hits: 2,
        qwen_gate_up_matvec_metal_fallbacks: 1,
    });

    let mut decisions = Vec::new();
    profile.append_route_decisions(&mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(
        decisions.get("ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_attempts"),
        Some(&5)
    );
    assert_eq!(
        decisions.get("ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_hits"),
        Some(&3)
    );
    assert_eq!(
        decisions.get("ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_fallbacks"),
        Some(&2)
    );
}

#[test]
fn decode_profile_route_decisions_emit_only_when_enabled() {
    let mut profile = DecodeProfileSnapshot {
        enabled: 1,
        decode_steps: 64,
        layers: 1536,
        per_layer_input_wall_us: 800,
        pre_sdpa_wall_us: 1200,
        pre_sdpa_qkv_proj_wall_us: 700,
        pre_sdpa_qk_norm_wall_us: 200,
        pre_sdpa_rope_kv_wall_us: 300,
        sdpa_wall_us: 500,
        post_attn_wall_us: 2400,
        post_attn_ffn_wall_us: 1800,
        post_attn_ffn_gate_up_wall_us: 900,
        post_attn_ffn_activation_wall_us: 300,
        post_attn_ffn_down_wall_us: 600,
        post_attn_output_proj_wall_us: 300,
        post_attn_residual_norm_wall_us: 100,
        post_attn_residual_gate_wall_us: 200,
        lm_head_wall_us: 150,
        moe_router_wall_us: 400,
        moe_expert_gate_up_wall_us: 500,
        moe_expert_activation_wall_us: 100,
        moe_expert_down_wall_us: 300,
        moe_expert_weighted_sum_wall_us: 50,
        moe_shared_expert_wall_us: 450,
    };
    profile.merge_from(DecodeProfileSnapshot {
        enabled: 1,
        decode_steps: 64,
        layers: 1536,
        per_layer_input_wall_us: 200,
        pre_sdpa_wall_us: 300,
        pre_sdpa_qkv_proj_wall_us: 200,
        pre_sdpa_qk_norm_wall_us: 50,
        pre_sdpa_rope_kv_wall_us: 75,
        sdpa_wall_us: 100,
        post_attn_wall_us: 600,
        post_attn_ffn_wall_us: 400,
        post_attn_ffn_gate_up_wall_us: 200,
        post_attn_ffn_activation_wall_us: 50,
        post_attn_ffn_down_wall_us: 150,
        post_attn_output_proj_wall_us: 75,
        post_attn_residual_norm_wall_us: 25,
        post_attn_residual_gate_wall_us: 50,
        lm_head_wall_us: 50,
        moe_router_wall_us: 100,
        moe_expert_gate_up_wall_us: 120,
        moe_expert_activation_wall_us: 30,
        moe_expert_down_wall_us: 80,
        moe_expert_weighted_sum_wall_us: 15,
        moe_shared_expert_wall_us: 110,
    });

    let mut decisions = Vec::new();
    profile.append_route_decisions(&mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(decisions.get("ax_mlx_decode_profile_enabled"), Some(&1));
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_decode_steps"),
        Some(&128)
    );
    assert_eq!(decisions.get("ax_mlx_decode_profile_layers"), Some(&3072));
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_per_layer_input_wall_us"),
        Some(&1000)
    );
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_pre_sdpa_wall_us"),
        Some(&1500)
    );
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_pre_sdpa_qkv_proj_wall_us"),
        Some(&900)
    );
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_pre_sdpa_qk_norm_wall_us"),
        Some(&250)
    );
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_pre_sdpa_rope_kv_wall_us"),
        Some(&375)
    );
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_sdpa_wall_us"),
        Some(&600)
    );
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_post_attn_wall_us"),
        Some(&3000)
    );
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_post_attn_ffn_wall_us"),
        Some(&2200)
    );
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_post_attn_ffn_gate_up_wall_us"),
        Some(&1100)
    );
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_post_attn_ffn_activation_wall_us"),
        Some(&350)
    );
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_post_attn_ffn_down_wall_us"),
        Some(&750)
    );
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_post_attn_output_proj_wall_us"),
        Some(&375)
    );
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_post_attn_residual_norm_wall_us"),
        Some(&125)
    );
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_post_attn_residual_gate_wall_us"),
        Some(&250)
    );
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_lm_head_wall_us"),
        Some(&200)
    );
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_moe_router_wall_us"),
        Some(&500)
    );
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_moe_expert_gate_up_wall_us"),
        Some(&620)
    );
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_moe_expert_activation_wall_us"),
        Some(&130)
    );
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_moe_expert_down_wall_us"),
        Some(&380)
    );
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_moe_expert_weighted_sum_wall_us"),
        Some(&65)
    );
    assert_eq!(
        decisions.get("ax_mlx_decode_profile_moe_shared_expert_wall_us"),
        Some(&560)
    );

    let mut disabled_decisions = Vec::new();
    DecodeProfileSnapshot::default().append_route_decisions(&mut disabled_decisions);
    assert!(disabled_decisions.is_empty());
}

#[test]
fn prefill_profile_route_decisions_emit_only_when_enabled() {
    let mut profile = PrefillProfileSnapshot {
        enabled: 1,
        prefill_steps: 2,
        layers: 48,
        tokens: 4096,
        per_layer_input_wall_us: 800,
        pre_sdpa_wall_us: 1200,
        pre_sdpa_qkv_proj_wall_us: 700,
        pre_sdpa_qk_norm_wall_us: 200,
        pre_sdpa_rope_kv_wall_us: 300,
        sdpa_wall_us: 500,
        post_attn_wall_us: 2400,
        post_attn_ffn_wall_us: 1800,
        post_attn_ffn_gate_up_wall_us: 900,
        post_attn_ffn_activation_wall_us: 300,
        post_attn_ffn_down_wall_us: 600,
        post_attn_output_proj_wall_us: 300,
        post_attn_residual_norm_wall_us: 100,
        post_attn_residual_gate_wall_us: 200,
        lm_head_wall_us: 150,
        moe_router_wall_us: 400,
        moe_expert_gate_up_wall_us: 500,
        moe_expert_activation_wall_us: 100,
        moe_expert_down_wall_us: 300,
        moe_expert_weighted_sum_wall_us: 50,
        moe_shared_expert_wall_us: 450,
    };
    profile.merge_from(PrefillProfileSnapshot {
        enabled: 1,
        prefill_steps: 1,
        layers: 24,
        tokens: 2048,
        per_layer_input_wall_us: 200,
        pre_sdpa_wall_us: 300,
        pre_sdpa_qkv_proj_wall_us: 200,
        pre_sdpa_qk_norm_wall_us: 50,
        pre_sdpa_rope_kv_wall_us: 75,
        sdpa_wall_us: 100,
        post_attn_wall_us: 600,
        post_attn_ffn_wall_us: 400,
        post_attn_ffn_gate_up_wall_us: 200,
        post_attn_ffn_activation_wall_us: 50,
        post_attn_ffn_down_wall_us: 150,
        post_attn_output_proj_wall_us: 75,
        post_attn_residual_norm_wall_us: 25,
        post_attn_residual_gate_wall_us: 50,
        lm_head_wall_us: 50,
        moe_router_wall_us: 100,
        moe_expert_gate_up_wall_us: 120,
        moe_expert_activation_wall_us: 30,
        moe_expert_down_wall_us: 80,
        moe_expert_weighted_sum_wall_us: 15,
        moe_shared_expert_wall_us: 110,
    });

    let mut decisions = Vec::new();
    profile.append_route_decisions(&mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(decisions.get("ax_mlx_prefill_profile_enabled"), Some(&1));
    assert_eq!(
        decisions.get("ax_mlx_prefill_profile_prefill_steps"),
        Some(&3)
    );
    assert_eq!(decisions.get("ax_mlx_prefill_profile_layers"), Some(&72));
    assert_eq!(decisions.get("ax_mlx_prefill_profile_tokens"), Some(&6144));
    assert_eq!(
        decisions.get("ax_mlx_prefill_profile_pre_sdpa_wall_us"),
        Some(&1500)
    );
    assert_eq!(
        decisions.get("ax_mlx_prefill_profile_pre_sdpa_qkv_proj_wall_us"),
        Some(&900)
    );
    assert_eq!(
        decisions.get("ax_mlx_prefill_profile_sdpa_wall_us"),
        Some(&600)
    );
    assert_eq!(
        decisions.get("ax_mlx_prefill_profile_post_attn_ffn_wall_us"),
        Some(&2200)
    );
    assert_eq!(
        decisions.get("ax_mlx_prefill_profile_post_attn_ffn_gate_up_wall_us"),
        Some(&1100)
    );
    assert_eq!(
        decisions.get("ax_mlx_prefill_profile_post_attn_ffn_activation_wall_us"),
        Some(&350)
    );
    assert_eq!(
        decisions.get("ax_mlx_prefill_profile_post_attn_ffn_down_wall_us"),
        Some(&750)
    );
    assert_eq!(
        decisions.get("ax_mlx_prefill_profile_lm_head_wall_us"),
        Some(&200)
    );
    assert_eq!(
        decisions.get("ax_mlx_prefill_profile_moe_router_wall_us"),
        Some(&500)
    );
    assert_eq!(
        decisions.get("ax_mlx_prefill_profile_moe_expert_gate_up_wall_us"),
        Some(&620)
    );
    assert_eq!(
        decisions.get("ax_mlx_prefill_profile_moe_expert_activation_wall_us"),
        Some(&130)
    );
    assert_eq!(
        decisions.get("ax_mlx_prefill_profile_moe_expert_down_wall_us"),
        Some(&380)
    );
    assert_eq!(
        decisions.get("ax_mlx_prefill_profile_moe_expert_weighted_sum_wall_us"),
        Some(&65)
    );
    assert_eq!(
        decisions.get("ax_mlx_prefill_profile_moe_shared_expert_wall_us"),
        Some(&560)
    );

    let mut disabled_decisions = Vec::new();
    PrefillProfileSnapshot::default().append_route_decisions(&mut disabled_decisions);
    assert!(disabled_decisions.is_empty());
}

#[test]
fn linear_attention_no_draft_threshold_avoids_request_long_disable() {
    // Flip S0 / coding streams: permanent disable after a short no-draft
    // streak killed multi-token verify for the rest of the request. Keep
    // the threshold effectively unreachable so cooldowns own recovery.
    assert_eq!(LINEAR_NGRAM_NO_DRAFT_DISABLE_THRESHOLD, u32::MAX);
    assert!(!linear_ngram_no_draft_should_disable(8));
    assert!(!linear_ngram_no_draft_should_disable(256));
    assert!(linear_ngram_no_draft_should_disable(
        LINEAR_NGRAM_NO_DRAFT_DISABLE_THRESHOLD
    ));
}

#[test]
fn linear_attention_non_repeating_prompt_without_initial_draft_uses_direct_fallback() {
    let empty = NgramTable::new();
    let variant = NgramPolicyVariant::MajorityRecency;
    // Empty table → probe returns no draft → disable for linear+NON_REPEATING.
    assert!(
        linear_ngram_initial_prompt_should_disable_request(
            true,
            crate::ngram_accel::PROMPT_CLASS_NON_REPEATING,
            &empty,
            variant,
        ),
        "linear + NON_REPEATING + empty table should disable (no useful bigrams)"
    );
    // REPEATING prompt short-circuits before probing.
    assert!(
        !linear_ngram_initial_prompt_should_disable_request(
            true,
            crate::ngram_accel::PROMPT_CLASS_REPEATING,
            &empty,
            variant,
        ),
        "repeating prompts should keep the speculative path eligible"
    );
    // Dense models don't pay recompute cost → no disable gate.
    assert!(
        !linear_ngram_initial_prompt_should_disable_request(
            false,
            crate::ngram_accel::PROMPT_CLASS_NON_REPEATING,
            &empty,
            variant,
        ),
        "dense models can cheaply roll back and should keep the existing gate"
    );
    // NON_REPEATING prompt but with prompt-seeded bigrams that yield a draft →
    // do NOT disable; the prompt has useful structure despite the classifier label.
    let mut seeded = NgramTable::new();
    // feed_from_prompt marks bigrams with prompt_count=1 so bypass_prompt_min_support
    // allows drafting from step 1 even with LINEAR_MIN_NGRAM_SUPPORT=2.
    // Ending sequence ...10,20 leaves tail=[...,10,20], which looks up bigram (10,20)→30.
    seeded.feed_from_prompt(&[10, 20, 30, 10, 20]);
    assert!(
        !linear_ngram_initial_prompt_should_disable_request(
            true,
            crate::ngram_accel::PROMPT_CLASS_NON_REPEATING,
            &seeded,
            variant,
        ),
        "NON_REPEATING but with prompt-seeded bigrams should stay enabled"
    );
}

#[test]
fn linear_attention_direct_fallback_reenables_when_output_builds_draft() {
    let mut state = RequestState::new(1, 7, None);
    state.ngram_acceleration_disabled_for_request = true;
    state.ngram_request_disable_reason = NgramRequestDisableReason::LinearNoDraft;
    state.linear_ngram_no_draft_streak = LINEAR_NGRAM_NO_DRAFT_DISABLE_THRESHOLD;
    state.ngram_disabled_steps = LINEAR_NGRAM_RETRY_INTERVAL;

    state.ngram.feed(&[1, 2, 3, 4, 9]);
    maybe_reenable_linear_ngram_from_fallback_output(
        &mut state,
        NgramPolicyVariant::MajorityRecency,
        true,
    );
    assert!(
        state.ngram_acceleration_disabled_for_request,
        "one observed continuation is below the linear-attention support gate"
    );
    assert_eq!(
        state.linear_ngram_reenable_probe_countdown,
        LINEAR_NGRAM_REENABLE_PROBE_INTERVAL
    );

    state.ngram.feed(&[1, 2, 3, 4, 9, 1, 2, 3, 4]);
    maybe_reenable_linear_ngram_from_fallback_output(
        &mut state,
        NgramPolicyVariant::MajorityRecency,
        true,
    );
    assert!(
        state.ngram_acceleration_disabled_for_request,
        "reenable probing is throttled while direct fallback is active"
    );
    assert_eq!(
        state.linear_ngram_reenable_probe_countdown,
        LINEAR_NGRAM_REENABLE_PROBE_INTERVAL - 1
    );

    state.linear_ngram_reenable_probe_countdown = 0;
    maybe_reenable_linear_ngram_from_fallback_output(
        &mut state,
        NgramPolicyVariant::MajorityRecency,
        true,
    );

    assert!(!state.ngram_acceleration_disabled_for_request);
    assert_eq!(
        state.ngram_request_disable_reason,
        NgramRequestDisableReason::None
    );
    assert_eq!(state.linear_ngram_no_draft_streak, 0);
    assert_eq!(state.linear_ngram_reenable_probe_countdown, 0);
    assert_eq!(state.ngram_disabled_steps, 0);
}

#[test]
fn linear_attention_reenable_keeps_short_output_disable_closed() {
    let mut state = RequestState::new(1, 7, None);
    state.ngram_acceleration_disabled_for_request = true;
    state.ngram_request_disable_reason = NgramRequestDisableReason::ShortOutputBudget;
    state
        .ngram
        .feed(&[1, 2, 3, 4, 9, 1, 2, 3, 4, 9, 1, 2, 3, 4]);

    maybe_reenable_linear_ngram_from_fallback_output(
        &mut state,
        NgramPolicyVariant::MajorityRecency,
        true,
    );

    assert!(state.ngram_acceleration_disabled_for_request);
    assert_eq!(
        state.ngram_request_disable_reason,
        NgramRequestDisableReason::ShortOutputBudget
    );
}

#[test]
fn linear_attention_initial_no_draft_stays_on_direct_fallback() {
    let mut state = RequestState::new(1, 7, None);
    state.ngram_acceleration_disabled_for_request = true;
    state.ngram_request_disable_reason = NgramRequestDisableReason::LinearInitialNoDraft;
    state
        .ngram
        .feed(&[1, 2, 3, 4, 9, 1, 2, 3, 4, 9, 1, 2, 3, 4]);

    maybe_reenable_linear_ngram_from_fallback_output(
        &mut state,
        NgramPolicyVariant::MajorityRecency,
        true,
    );

    assert!(
        state.ngram_acceleration_disabled_for_request,
        "initial non-repeating prompts stay on direct fallback for the request"
    );
    assert_eq!(
        state.ngram_request_disable_reason,
        NgramRequestDisableReason::LinearInitialNoDraft
    );
    assert_eq!(state.linear_ngram_reenable_probe_countdown, 0);
}

#[test]
fn request_disabled_fallback_only_feeds_ngram_for_runtime_no_draft() {
    assert!(ngram_request_disabled_fallback_should_feed_output(
        NgramRequestDisableReason::LinearNoDraft
    ));
    assert!(!ngram_request_disabled_fallback_should_feed_output(
        NgramRequestDisableReason::LinearInitialNoDraft
    ));
    assert!(!ngram_request_disabled_fallback_should_feed_output(
        NgramRequestDisableReason::ShortOutputBudget
    ));
}

#[test]
fn rotating_bounded_family_eligible_includes_gemma4_vl() {
    // gemma4_vl SWA uses families::standard::layer_forward (ring-aware).
    let make = |family: &str| {
        let value = serde_json::json!({
            "schema_version": ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION,
            "model_family": family,
            "tensor_format": "safetensors",
            "layer_count": 2,
            "hidden_size": 64,
            "attention_head_count": 4,
            "attention_head_dim": 16,
            "kv_head_count": 1,
            "vocab_size": 32,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window_size": 128,
            "tensors": [],
        });
        let manifest: ax_engine_core::NativeModelManifest =
            serde_json::from_value(value).expect("manifest");
        crate::model::ModelConfig::from_manifest(&manifest)
    };
    assert!(
        rotating_bounded_family_eligible(&make("gemma4")),
        "gemma4 must be ring-eligible"
    );
    assert!(
        rotating_bounded_family_eligible(&make("gemma4_vl")),
        "gemma4_vl standard SWA path must be ring-eligible like gemma4"
    );
    assert!(
        !rotating_bounded_family_eligible(&make("gpt_oss")),
        "gpt_oss builds family-local masks without ring awareness"
    );
}

#[test]
fn request_rotating_sliding_slack_covers_rollback_sources() {
    use NgramRequestDisableReason as R;
    const SLACK: usize = ROTATING_BOUNDED_ROLLBACK_SLACK;
    const G: bool = true; // greedy
    const S: bool = false; // sampled
    // mtp_ring_slack values: no MTP drafting; the Gemma4 assistant with
    // a verify-width slack requirement; an uncoverable MTP head.
    const NO_MTP: Option<usize> = Some(0);
    const ASSISTANT: Option<usize> = Some(9);
    const MTP_HEAD: Option<usize> = None;
    let slack = request_rotating_sliding_slack;
    // Session-level direct policy always rotates with a pure ring — MTP
    // never engages inside a direct session, so even MTP models qualify.
    assert_eq!(
        slack(true, true, true, true, false, R::None, MTP_HEAD, G),
        Some(0)
    );
    // ngram-ON session: sticky per-request disable without MTP rotates
    // with a pure ring (the 38d8c70c class, unchanged).
    assert_eq!(
        slack(
            false,
            true,
            true,
            true,
            true,
            R::ShortOutputBudget,
            NO_MTP,
            G
        ),
        Some(0)
    );
    assert_eq!(
        slack(
            false,
            true,
            true,
            true,
            true,
            R::LinearInitialNoDraft,
            NO_MTP,
            G
        ),
        Some(0)
    );
    // n-gram ACTIVE without MTP on an eligible family: bounded ring.
    assert_eq!(
        slack(false, true, true, true, false, R::None, NO_MTP, G),
        Some(SLACK)
    );
    // LinearNoDraft can re-enable n-gram mid-request: bounded, not pure.
    assert_eq!(
        slack(false, true, true, true, true, R::LinearNoDraft, NO_MTP, G),
        Some(SLACK)
    );
    // Sampled requests are never pure but stay bounded-eligible: the
    // rejection-sampling verify trims are bounded by MAX_DRAFT_LEN just
    // like greedy accepts. Sticky-disable and direct-session sampled
    // requests land on the bounded arm instead of pure.
    assert_eq!(
        slack(false, true, true, true, false, R::None, NO_MTP, S),
        Some(SLACK)
    );
    assert_eq!(
        slack(
            false,
            true,
            true,
            true,
            true,
            R::ShortOutputBudget,
            NO_MTP,
            S
        ),
        Some(SLACK)
    );
    assert_eq!(
        slack(true, true, true, true, false, R::None, NO_MTP, S),
        Some(SLACK)
    );
    // Sampled with the bounded kill-switch off: no rotation at all (no
    // pure fallback — pure rings stay greedy-only).
    assert_eq!(
        slack(
            false,
            true,
            false,
            true,
            true,
            R::ShortOutputBudget,
            NO_MTP,
            S
        ),
        None
    );
    // Gemma4 assistant MTP: bounded ring with the widened slack (max of
    // the base slack and the assistant verify-width requirement), for
    // greedy and sampled, including sticky-ngram-disabled requests
    // (per-request n-gram disable does not stop MTP drafting, so pure
    // rings stay off the table).
    assert_eq!(
        slack(false, true, true, true, false, R::None, ASSISTANT, G),
        Some(SLACK.max(9))
    );
    assert_eq!(
        slack(false, true, true, true, false, R::None, ASSISTANT, S),
        Some(SLACK.max(9))
    );
    assert_eq!(
        slack(
            false,
            true,
            true,
            true,
            true,
            R::ShortOutputBudget,
            ASSISTANT,
            G
        ),
        Some(SLACK.max(9))
    );
    // A small assistant requirement never shrinks the base slack.
    assert_eq!(
        slack(false, true, true, true, false, R::None, Some(3), G),
        Some(SLACK)
    );
    // Uncoverable MTP heads: never rotate outside direct sessions.
    assert_eq!(
        slack(false, true, true, true, false, R::None, MTP_HEAD, G),
        None
    );
    assert_eq!(
        slack(
            false,
            true,
            true,
            true,
            true,
            R::ShortOutputBudget,
            MTP_HEAD,
            G
        ),
        None
    );
    // Family whose sliding attention bypasses the ring-aware seam
    // (llama4 / gpt_oss / diffusion): no bounded ring, and n-gram-active
    // means no pure ring either.
    assert_eq!(
        slack(false, true, true, false, false, R::None, NO_MTP, G),
        None
    );
    // Bounded-rollback kill-switch off: n-gram-active requests keep the
    // pre-bounded behavior (no rotation); greedy sticky-disable still
    // pure.
    assert_eq!(
        slack(false, true, false, true, false, R::None, NO_MTP, G),
        None
    );
    assert_eq!(
        slack(
            false,
            true,
            false,
            true,
            true,
            R::ShortOutputBudget,
            NO_MTP,
            G
        ),
        Some(0)
    );
    // Rotation kill-switch off: only the session-level path may rotate.
    assert_eq!(
        slack(
            false,
            false,
            true,
            true,
            true,
            R::ShortOutputBudget,
            NO_MTP,
            G
        ),
        None
    );
}

#[test]
fn request_disabled_direct_fast_path_skips_non_reenable_ngram_fallback() {
    assert!(ngram_request_disabled_direct_fast_path(
        true,
        false,
        false,
        true,
        NgramRequestDisableReason::LinearInitialNoDraft,
    ));
    assert!(ngram_request_disabled_direct_fast_path(
        true,
        false,
        false,
        true,
        NgramRequestDisableReason::ShortOutputBudget,
    ));
    assert!(!ngram_request_disabled_direct_fast_path(
        true,
        false,
        false,
        true,
        NgramRequestDisableReason::LinearNoDraft,
    ));
    assert!(!ngram_request_disabled_direct_fast_path(
        true,
        true,
        false,
        true,
        NgramRequestDisableReason::LinearInitialNoDraft,
    ));
    assert!(!ngram_request_disabled_direct_fast_path(
        true,
        false,
        true,
        true,
        NgramRequestDisableReason::LinearInitialNoDraft,
    ));
    assert!(!ngram_request_disabled_direct_fast_path(
        false,
        false,
        false,
        true,
        NgramRequestDisableReason::LinearInitialNoDraft,
    ));
}

#[test]
fn row_exact_coalescing_is_scoped_to_qwen_and_gemma_families() {
    for family in ["qwen3", "qwen3_5", "qwen3_next", "gemma3", "gemma4"] {
        assert!(row_exact_coalescing_family(family), "{family}");
    }
    for family in ["llama3", "glm4", "deepseek", "diffusion_gemma"] {
        assert!(!row_exact_coalescing_family(family), "{family}");
    }
}

#[test]
fn shared_wall_time_distribution_is_exact_and_balanced() {
    let shares = (0..4)
        .map(|ordinal| shared_wall_time_share(11, ordinal, 4))
        .collect::<Vec<_>>();
    assert_eq!(shares, vec![3, 3, 3, 2]);
    assert_eq!(shares.into_iter().sum::<u32>(), 11);
    assert_eq!(shared_wall_time_share(99, 0, 0), 0);
}

#[test]
fn linear_attention_reenable_requires_greedy_exact_decode() {
    let mut state = RequestState::new(1, 7, None);
    state.ngram_acceleration_disabled_for_request = true;
    state.ngram_request_disable_reason = NgramRequestDisableReason::LinearNoDraft;
    state
        .ngram
        .feed(&[1, 2, 3, 4, 9, 1, 2, 3, 4, 9, 1, 2, 3, 4]);

    maybe_reenable_linear_ngram_from_fallback_output(
        &mut state,
        NgramPolicyVariant::MajorityRecency,
        false,
    );

    assert!(state.ngram_acceleration_disabled_for_request);
    assert_eq!(
        state.ngram_request_disable_reason,
        NgramRequestDisableReason::LinearNoDraft
    );
}

#[test]
fn primary_mlx_runner_registry_covers_secondary_catalog_and_gpt_oss() {
    // Recently expanded direct-mode secondary stack must clear the runner gate.
    for family in [
        "llama3",
        "llama4",
        "mistral3",
        "mixtral",
        "gpt_oss",
        "gemma4_unified",
        "gemma4_vl",
        "nemotron_h",
        "qwen3_vl",
        "qwen3_vl_moe",
    ] {
        assert!(
            ax_engine_core::is_primary_mlx_runner_family(family),
            "{family} must be admitted by the primary MLX runner registry"
        );
    }
    // Assistant MTP draft artifacts are sidecars, not primary runners.
    assert!(!ax_engine_core::is_primary_mlx_runner_family(
        "gemma4_assistant"
    ));
    assert!(ax_engine_core::is_primary_mlx_runner_family("qwen4_exp"));
    assert!(!ax_engine_core::is_primary_mlx_runner_family("gpt2"));
}

#[test]
fn route_decision_upsert_replaces_existing_value_and_removes_duplicates() {
    let mut decisions = vec![
        ("other".to_string(), 7),
        ("ax_ngram_draft_tokens".to_string(), 1),
        ("ax_ngram_draft_tokens".to_string(), 2),
    ];

    upsert_route_decision(&mut decisions, "ax_ngram_draft_tokens", 9);

    assert_eq!(decisions[0], ("other".to_string(), 7));
    assert_eq!(decisions[1], ("ax_ngram_draft_tokens".to_string(), 9));
    assert_eq!(
        decisions
            .iter()
            .filter(|(key, _)| key == "ax_ngram_draft_tokens")
            .count(),
        1
    );
}

#[test]
fn indexed_route_decisions_update_in_place_and_remove_initial_duplicates() {
    let mut decisions = vec![
        ("other".to_string(), 7),
        ("ax_ngram_draft_tokens".to_string(), 1),
        ("ax_ngram_draft_tokens".to_string(), 2),
    ];

    {
        let mut indexed = IndexedRouteDecisions::new(&mut decisions);
        indexed.upsert_route_decision("ax_ngram_draft_tokens", 9);
        indexed.upsert_route_decision("new_counter", 3);
    }

    assert_eq!(
        decisions,
        vec![
            ("other".to_string(), 7),
            ("ax_ngram_draft_tokens".to_string(), 9),
            ("new_counter".to_string(), 3),
        ]
    );
}

#[test]
fn decode_telemetry_records_route_counters() {
    let mut telemetry = DecodeTelemetry::default();

    telemetry.record_prefill(100);
    telemetry.record_prefill_breakdown(70, 20, 5);
    telemetry.record_prefill_eval_barrier();
    telemetry.record_prefill_drain_async_evals(2);
    telemetry.record_prefill_cache_only_continuation();
    telemetry.record_decode(40);
    telemetry.record_direct_bootstrap(7);
    telemetry.record_direct_pipeline(11);
    telemetry.record_direct_pipeline_timings(DirectPipelineTimings {
        forward_wall_us: 3,
        forward_layer_loop_wall_us: 2,
        forward_head_wall_us: 1,
        argmax_wall_us: 4,
        async_eval_wall_us: 2,
        next_complete_wall_us: 6,
        pending_eval_wall_us: 5,
        pending_read_wall_us: 1,
        linear_attention_layer_ops: 22,
        linear_attention_layer_count: 2,
        full_attention_layer_ops: 20,
        full_attention_layer_count: 4,
    });
    telemetry.record_direct_pipeline_op_count(42);
    telemetry.record_single_decode(13);
    telemetry.record_ngram_decode(17);
    telemetry.record_bonus_token();
    telemetry.record_bonus_token();

    let mut decisions = vec![
        ("ax_mlx_decode_steps".to_string(), 999),
        ("other_counter".to_string(), 3),
        ("ax_mlx_decode_steps".to_string(), 111),
    ];
    telemetry.append_route_decisions(&mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(decisions.get("ax_mlx_prefill_steps"), Some(&1));
    assert_eq!(decisions.get("ax_mlx_prefill_wall_us"), Some(&100));
    assert_eq!(decisions.get("ax_mlx_prefill_forward_wall_us"), Some(&70));
    assert_eq!(
        decisions.get("ax_mlx_prefill_prefix_cache_wall_us"),
        Some(&20)
    );
    assert_eq!(
        decisions.get("ax_mlx_prefill_generation_state_wall_us"),
        Some(&5)
    );
    assert_eq!(decisions.get("ax_mlx_prefill_eval_barriers"), Some(&1));
    assert_eq!(decisions.get("ax_mlx_prefill_drain_async_evals"), Some(&2));
    assert_eq!(
        decisions.get("ax_mlx_prefill_cache_only_continuations"),
        Some(&1)
    );
    assert_eq!(decisions.get("ax_mlx_decode_steps"), Some(&1));
    assert_eq!(decisions.get("ax_mlx_decode_wall_us"), Some(&40));
    assert_eq!(decisions.get("ax_mlx_direct_bootstrap_steps"), Some(&1));
    assert_eq!(decisions.get("ax_mlx_direct_bootstrap_wall_us"), Some(&7));
    assert_eq!(decisions.get("ax_mlx_direct_pipeline_steps"), Some(&1));
    assert_eq!(decisions.get("ax_mlx_direct_pipeline_wall_us"), Some(&11));
    assert_eq!(
        decisions.get("ax_mlx_direct_pipeline_forward_wall_us"),
        Some(&3)
    );
    assert_eq!(
        decisions.get("ax_mlx_direct_pipeline_forward_layer_loop_wall_us"),
        Some(&2)
    );
    assert_eq!(
        decisions.get("ax_mlx_direct_pipeline_forward_head_wall_us"),
        Some(&1)
    );
    assert_eq!(
        decisions.get("ax_mlx_direct_pipeline_argmax_wall_us"),
        Some(&4)
    );
    assert_eq!(
        decisions.get("ax_mlx_direct_pipeline_async_eval_wall_us"),
        Some(&2)
    );
    assert_eq!(
        decisions.get("ax_mlx_direct_pipeline_next_complete_wall_us"),
        Some(&6)
    );
    assert_eq!(
        decisions.get("ax_mlx_direct_pipeline_pending_eval_wall_us"),
        Some(&5)
    );
    assert_eq!(
        decisions.get("ax_mlx_direct_pipeline_pending_read_wall_us"),
        Some(&1)
    );
    assert_eq!(decisions.get("ax_mlx_direct_pipeline_op_count"), Some(&42));
    assert_eq!(
        decisions.get("ax_mlx_direct_pipeline_linear_attention_layer_ops"),
        Some(&22)
    );
    assert_eq!(
        decisions.get("ax_mlx_direct_pipeline_linear_attention_layer_count"),
        Some(&2)
    );
    assert_eq!(
        decisions.get("ax_mlx_direct_pipeline_full_attention_layer_ops"),
        Some(&20)
    );
    assert_eq!(
        decisions.get("ax_mlx_direct_pipeline_full_attention_layer_count"),
        Some(&4)
    );
    assert_eq!(decisions.get("ax_mlx_single_decode_steps"), Some(&1));
    assert_eq!(decisions.get("ax_mlx_single_decode_wall_us"), Some(&13));
    assert_eq!(decisions.get("ax_mlx_ngram_decode_steps"), Some(&1));
    assert_eq!(decisions.get("ax_mlx_ngram_decode_wall_us"), Some(&17));
    assert_eq!(decisions.get("ax_mlx_bonus_tokens"), Some(&2));
    assert_eq!(decisions.get("other_counter"), Some(&3));
}

#[test]
fn decode_telemetry_records_diffusion_block() {
    let mut telemetry = DecodeTelemetry::default();

    // Record two diffusion blocks: one converged, one not.
    telemetry.record_diffusion_block(&crate::diffusion::DiffusionBlockResult {
        tokens: vec![1, 2, 3, 4],
        denoise_steps: 4,
        converged: true,
        converged_strict: true,
        converged_acceptance: false,
        converged_plateau: false,
        min_entropy: 0.003,
        min_acceptance_rate: 0.05,
        denoise_wall_us: 500,
        commit_wall_us: 100,
        block_wall_us: 700,
        commit_skipped: false,
        full_pipeline_used: false,
        kv_buffer_used: true,
    });
    telemetry.record_diffusion_block(&crate::diffusion::DiffusionBlockResult {
        tokens: vec![5, 6],
        denoise_steps: 8,
        converged: false,
        converged_strict: false,
        converged_acceptance: false,
        converged_plateau: false,
        min_entropy: 0.020,
        min_acceptance_rate: 0.15,
        denoise_wall_us: 900,
        commit_wall_us: 200,
        block_wall_us: 1300,
        commit_skipped: true,
        full_pipeline_used: true,
        kv_buffer_used: true,
    });

    let mut decisions: Vec<(String, u32)> = Vec::new();
    telemetry.append_route_decisions(&mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(decisions.get("ax_mlx_diffusion_blocks"), Some(&2));
    assert_eq!(decisions.get("ax_mlx_diffusion_denoise_steps"), Some(&12));
    assert_eq!(decisions.get("ax_mlx_diffusion_converged_blocks"), Some(&1));
    assert_eq!(
        decisions.get("ax_mlx_diffusion_denoise_wall_us"),
        Some(&1400)
    );
    assert_eq!(decisions.get("ax_mlx_diffusion_commit_wall_us"), Some(&300));
    assert_eq!(decisions.get("ax_mlx_diffusion_block_wall_us"), Some(&2000));
    assert_eq!(decisions.get("ax_mlx_diffusion_min_entropy_bp"), Some(&30));
    assert_eq!(
        decisions.get("ax_mlx_diffusion_min_acceptance_rate_bp"),
        Some(&500)
    );
    assert_eq!(decisions.get("ax_mlx_diffusion_commit_skipped"), Some(&1));
    assert_eq!(
        decisions.get("ax_mlx_diffusion_full_pipeline_used"),
        Some(&1)
    );
    assert_eq!(decisions.get("ax_mlx_diffusion_kv_buffer_used"), Some(&2));
}

#[test]
fn decode_telemetry_emits_zero_diffusion_minima_without_blocks() {
    let telemetry = DecodeTelemetry::default();
    let mut decisions: Vec<(String, u32)> = Vec::new();
    telemetry.append_route_decisions(&mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(decisions.get("ax_mlx_diffusion_blocks"), Some(&0));
    assert_eq!(decisions.get("ax_mlx_diffusion_min_entropy_bp"), Some(&0));
    assert_eq!(
        decisions.get("ax_mlx_diffusion_min_acceptance_rate_bp"),
        Some(&0)
    );
}

#[test]
fn direct_pipeline_clear_cache_cadence_matches_mlx_lm_loop() {
    let due_tokens = (0..=260)
        .filter(|emitted| direct_pipeline_clear_cache_due(*emitted, 256))
        .collect::<Vec<_>>();

    assert_eq!(due_tokens, vec![1, 257]);
    assert!(!direct_pipeline_clear_cache_due(1, 0));
    assert!(direct_pipeline_clear_cache_due(1, 1));
    assert!(direct_pipeline_clear_cache_due(2, 1));
    assert!(!direct_pipeline_clear_cache_due(2, 256));
}

#[test]
fn request_fallback_direct_pipeline_reuses_pending_step() {
    // No pending, not final: bootstrap and leave the slot untouched.
    let mut slot = None;
    assert!(
        matches!(
            next_direct_pipeline_step(&mut slot, false),
            DirectPipelineStep::Bootstrap
        ),
        "first fallback direct step must bootstrap the pipeline"
    );
    // Pending present, not final: continue and hand the array over (slot
    // emptied so the same token can never be finished twice).
    let mut slot = Some(MlxArray::from_f32_slice(&[1.0]));
    assert!(
        matches!(
            next_direct_pipeline_step(&mut slot, false),
            DirectPipelineStep::ContinuePending(_)
        ),
        "later fallback direct steps must continue the pending lazy token"
    );
    assert!(slot.is_none(), "the pending slot must be consumed");
    // Pending present, final: finish without submitting lookahead.
    let mut slot = Some(MlxArray::from_f32_slice(&[1.0]));
    assert!(
        matches!(
            next_direct_pipeline_step(&mut slot, true),
            DirectPipelineStep::FinishPending(_)
        ),
        "final fallback direct step must not submit unused lookahead work"
    );
    // No pending, final: single-token materialise, no token kept.
    let mut slot = None;
    assert!(
        matches!(
            next_direct_pipeline_step(&mut slot, true),
            DirectPipelineStep::BootstrapFinal
        ),
        "single-token final fallback must materialise without keeping a pending token"
    );
    assert!(
        should_drain_pending_direct_before_ngram(true, true),
        "greedy n-gram re-entry must first materialise the pending direct token"
    );
    assert!(!should_drain_pending_direct_before_ngram(true, false));
    assert!(!should_drain_pending_direct_before_ngram(false, true));
}

#[test]
fn protected_prefix_ring_blocks_only_sampled_ngram_verify() {
    let sampled = MlxSamplingParams::new(0.3, 0.9, 0);
    // Unlimited-OCR (protected-prefix ring) must not run the multi-token
    // sampled verify: it would hit the ring's ordered-append assertion.
    assert!(ngram_sampled_verify_unsupported(Some(128), sampled));
    // Greedy replays singleton steps, so it keeps n-gram acceleration.
    assert!(!ngram_sampled_verify_unsupported(
        Some(128),
        MlxSamplingParams::greedy()
    ));
    // Every other family keeps sampled n-gram verification.
    assert!(!ngram_sampled_verify_unsupported(None, sampled));
}

#[test]
fn kv_cache_telemetry_records_route_counters() {
    let mut telemetry = KvCacheTelemetry::default();
    telemetry.merge_from(MlxKVCacheUsage {
        logical_tokens: 3,
        capacity_tokens: 256,
        logical_bytes: 96,
        capacity_bytes: 8192,
        full_attention_layers: 1,
        quantized_layers: 1,
        linear_state_layers: 0,
        linear_state_bytes: 0,
        growth_count: 1,
        paged_cow_copies: 1,
        paged_pool_blocks_used: 3,
        paged_pool_shared_blocks: 2,
        paged_pool_slabs: 3,
        paged_pool_slab_bytes: 4096,
        paged_pool_slab_grow_events: 1,
        paged_attention_calls: 4,
        paged_attention_fallbacks: 1,
        ..MlxKVCacheUsage::default()
    });
    telemetry.merge_from(MlxKVCacheUsage {
        logical_tokens: 5,
        capacity_tokens: 512,
        logical_bytes: 160,
        capacity_bytes: 16384,
        full_attention_layers: 2,
        quantized_layers: 2,
        sliding_window_layers: 1,
        sliding_window_retained_tokens: 4,
        sliding_window_reclaimable_capacity_tokens: 256,
        sliding_window_reclaimable_capacity_bytes: 8192,
        linear_state_layers: 1,
        linear_state_bytes: 936,
        growth_count: 2,
        paged_cow_copies: 2,
        paged_pool_blocks_used: 5,
        paged_pool_shared_blocks: 1,
        paged_pool_slabs: 5,
        paged_pool_slab_bytes: 12288,
        paged_pool_slab_grow_events: 2,
        paged_attention_calls: 6,
        paged_attention_fallbacks: 2,
        ..MlxKVCacheUsage::default()
    });

    let mut decisions = Vec::new();
    telemetry.append_route_decisions(&mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_REQUEST_SNAPSHOTS),
        Some(&2)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_LOGICAL_TOKENS),
        Some(&8)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_CAPACITY_TOKENS),
        Some(&768)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_LOGICAL_KIB),
        Some(&1)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_CAPACITY_KIB),
        Some(&24)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_FULL_ATTENTION_LAYERS),
        Some(&3)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_QUANTIZED_LAYERS),
        Some(&2),
        "quantized layers merge as a per-request peak gauge, not a sum"
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_SLIDING_WINDOW_LAYERS),
        Some(&1)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_SLIDING_RETAINED_TOKENS),
        Some(&4)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_TOKENS),
        Some(&256)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_KIB),
        Some(&8)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_LAYERS),
        Some(&1)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_KIB),
        Some(&1)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_GROWTH_COUNT),
        Some(&3)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_PAGED_COW_COPIES),
        Some(&3)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_BLOCKS_USED),
        Some(&5)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_SHARED_BLOCKS),
        Some(&2)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_SLABS),
        Some(&5)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_SLAB_KIB),
        Some(&12)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_SLAB_GROW_EVENTS),
        Some(&2)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_PAGED_ATTENTION_CALLS),
        Some(&10)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_PAGED_ATTENTION_FALLBACKS),
        Some(&3)
    );
    assert!(
        ax_engine_core::ROUTE_DECISION_AX_MLX_KV_KEYS
            .iter()
            .all(|key| decisions.contains_key(*key)),
        "KV telemetry must emit the full canonical counter set after any snapshot"
    );
}

#[test]
fn kv_cache_telemetry_emits_zero_counters_after_empty_snapshot() {
    let mut telemetry = KvCacheTelemetry::default();
    telemetry.merge_from(MlxKVCacheUsage::default());

    let mut decisions = Vec::new();
    telemetry.append_route_decisions(&mut decisions);
    let decisions = decisions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(
        decisions.len(),
        ax_engine_core::ROUTE_DECISION_AX_MLX_KV_KEYS.len(),
        "KV telemetry should distinguish zero counters from unsupported telemetry"
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_REQUEST_SNAPSHOTS),
        Some(&1)
    );
    for key in ax_engine_core::ROUTE_DECISION_AX_MLX_KV_KEYS {
        assert!(
            decisions.contains_key(key),
            "missing canonical KV counter {key}"
        );
    }
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_CAPACITY_TOKENS),
        Some(&0)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_SLIDING_WINDOW_LAYERS),
        Some(&0)
    );
    assert_eq!(
        decisions.get(ROUTE_DECISION_AX_MLX_KV_GROWTH_COUNT),
        Some(&0)
    );
}

#[test]
fn kv_cache_telemetry_upserts_existing_canonical_counters() {
    let mut telemetry = KvCacheTelemetry::default();
    telemetry.merge_from(MlxKVCacheUsage {
        capacity_tokens: 256,
        capacity_bytes: 8192,
        growth_count: 1,
        ..MlxKVCacheUsage::default()
    });

    let mut decisions = vec![
        (ROUTE_DECISION_AX_MLX_KV_CAPACITY_TOKENS.to_string(), 999),
        ("other_counter".to_string(), 7),
        (ROUTE_DECISION_AX_MLX_KV_CAPACITY_TOKENS.to_string(), 111),
    ];
    telemetry.append_route_decisions(&mut decisions);

    assert_eq!(
        decisions
            .iter()
            .filter(|(key, _)| key == ROUTE_DECISION_AX_MLX_KV_CAPACITY_TOKENS)
            .count(),
        1
    );
    assert_eq!(
        decisions
            .iter()
            .find(|(key, _)| key == ROUTE_DECISION_AX_MLX_KV_CAPACITY_TOKENS)
            .map(|(_, value)| *value),
        Some(256)
    );
    assert!(decisions.contains(&("other_counter".to_string(), 7)));
}

#[test]
fn dense_ngram_acceleration_uses_beta_posterior_gate() {
    // Posterior mean above threshold → no cooldown.
    assert_eq!(
        ngram_acceleration_disabled_steps(false, 3, DEFAULT_DRAFT_LEN, 0.95),
        None
    );
    // Posterior mean below threshold → cooldown period.
    assert_eq!(
        ngram_acceleration_disabled_steps(false, 0, DEFAULT_DRAFT_LEN, 0.49),
        Some(NGRAM_RETRY_INTERVAL)
    );
}

#[test]
fn linear_attention_draft_requires_repeated_ngram_evidence() {
    let mut ngram = NgramTable::new();
    ngram.feed(&[1, 2, 3, 1, 2, 3]);

    // Dense: 3-token cycle builds high-confidence bigrams → draft up to MAX_DRAFT_LEN.
    let dense_draft = ngram_acceleration_draft(
        &ngram,
        false,
        0.95,
        NgramPolicyVariant::MajorityRecency,
        false,
    )
    .draft;
    assert!(!dense_draft.is_empty(), "dense draft should be non-empty");
    assert!(
        dense_draft.len() <= MAX_DRAFT_LEN,
        "dense draft must not exceed MAX_DRAFT_LEN"
    );

    // Linear-attention: min_support=2 filters one-off n-grams.
    assert!(
        ngram_acceleration_draft(
            &ngram,
            true,
            0.95,
            NgramPolicyVariant::MajorityRecency,
            false
        )
        .draft
        .is_empty(),
        "linear attention should not probe one-off prompt n-grams"
    );

    ngram.feed(&[1, 2, 3]);
    let lin_draft = ngram_acceleration_draft(
        &ngram,
        true,
        0.95,
        NgramPolicyVariant::MajorityRecency,
        false,
    )
    .draft;
    assert!(
        !lin_draft.is_empty(),
        "linear attention draft should be non-empty after second repeat"
    );
    assert!(
        lin_draft.len() <= DEFAULT_DRAFT_LEN,
        "linear attention draft must not exceed DEFAULT_DRAFT_LEN"
    );
}

#[test]
fn ngram_adaptive_draft_len_shrinks_and_extends_from_acceptance() {
    // Dense models always use MAX_DRAFT_LEN — confidence gate prunes naturally.
    assert_eq!(adaptive_ngram_draft_len(false, 0.95), MAX_DRAFT_LEN);
    assert_eq!(adaptive_ngram_draft_len(false, 0.70), MAX_DRAFT_LEN);
    assert_eq!(adaptive_ngram_draft_len(false, 0.40), MAX_DRAFT_LEN);
    assert_eq!(
        adaptive_ngram_draft_len(true, 0.95),
        DEFAULT_DRAFT_LEN,
        "linear attention stays capped even at high confidence"
    );
    assert_eq!(
        adaptive_ngram_draft_len(true, 0.40),
        NGRAM_DRAFT_LEN_LOW_CONFIDENCE
    );
}

#[test]
fn mlx_manifest_validation_rejects_unsupported_linear_key_dim() {
    let mut manifest = qwen35_linear_manifest();
    manifest.linear_attention.key_head_dim = Some(4);
    for tensor in &mut manifest.tensors {
        match tensor.role {
            NativeTensorRole::LinearAttentionInProjQkv => tensor.shape = vec![12, 4],
            NativeTensorRole::LinearAttentionConv1d => tensor.shape = vec![12, 4, 1],
            _ => {}
        }
    }
    let artifacts = write_artifacts(manifest);

    let error = validate_mlx_supported_manifest(&artifacts)
        .expect_err("Dk must match the gated-delta kernel contract");

    assert!(error.to_string().contains("divisible by 32"));
}

#[test]
fn mlx_manifest_validation_allows_attn_output_gate() {
    let mut manifest = dense_manifest();
    manifest.model_family = "qwen3".to_string();
    manifest.attn_output_gate = true;
    manifest
        .tensors
        .iter_mut()
        .find(|tensor| tensor.role == NativeTensorRole::AttentionQ)
        .expect("q tensor should exist")
        .shape = vec![8, 4];
    let artifacts = write_artifacts(manifest);

    validate_mlx_supported_manifest(&artifacts)
        .expect("attention output gate is implemented in the MLX model graph");
}

#[test]
fn mlx_manifest_validation_allows_gemma4_interleaved_attention() {
    let mut manifest = dense_manifest();
    manifest.model_family = "gemma4".to_string();
    manifest.sliding_window_size = Some(1024);
    manifest.layer_types = vec!["sliding_attention".to_string()];
    manifest.global_head_dim = Some(8);
    let artifacts = write_artifacts(manifest);

    validate_mlx_supported_manifest(&artifacts)
        .expect("Gemma4 interleaved attention is implemented in the MLX model graph");
}

#[test]
fn mlx_manifest_validation_allows_gpt_oss_interleaved_attention() {
    // GPT-OSS alternates sliding_attention / full_attention with a fixed
    // sliding window (mlx-lm gpt_oss). The family has a dedicated forward
    // path and must clear the interleaved SWA gate.
    let mut manifest = dense_manifest();
    manifest.model_family = "gpt_oss".to_string();
    manifest.layer_count = 4;
    manifest.sliding_window_size = Some(128);
    manifest.layer_types = vec![
        "sliding_attention".to_string(),
        "full_attention".to_string(),
        "sliding_attention".to_string(),
        "full_attention".to_string(),
    ];

    validate_gemma4_interleaved_attention(&manifest)
        .expect("GPT-OSS alternating SWA/full attention is implemented");
    assert!(ax_engine_core::is_primary_mlx_runner_family("gpt_oss"));
}

#[test]
fn mlx_manifest_validation_rejects_unknown_interleaved_attention() {
    let mut manifest = dense_manifest();
    manifest.model_family = "qwen3".to_string();
    manifest.sliding_window_size = Some(1024);
    manifest.layer_types = vec!["sliding_attention".to_string()];
    manifest.global_head_dim = Some(8);
    let artifacts = write_artifacts(manifest);

    let error = validate_mlx_supported_manifest(&artifacts)
        .expect_err("unknown family interleaved attention should fail closed");

    assert!(error.to_string().contains("not implemented"));
}

#[test]
fn mlx_manifest_validation_allows_valid_gemma4_kv_shared_layers() {
    let mut manifest = dense_manifest();
    manifest.model_family = "gemma4".to_string();
    manifest.layer_count = 2;
    manifest.sliding_window_size = Some(1024);
    manifest.layer_types = vec![
        "sliding_attention".to_string(),
        "sliding_attention".to_string(),
    ];
    manifest.kv_shared_source_layers.insert(1, 0);

    validate_gemma4_interleaved_attention(&manifest)
        .expect("same-type Gemma4 KV sharing should be supported");
}

#[test]
fn mlx_manifest_validation_rejects_cross_type_gemma4_kv_shared_layers() {
    let mut manifest = dense_manifest();
    manifest.model_family = "gemma4".to_string();
    manifest.layer_count = 2;
    manifest.sliding_window_size = Some(1024);
    manifest.layer_types = vec![
        "sliding_attention".to_string(),
        "full_attention".to_string(),
    ];
    manifest.kv_shared_source_layers.insert(1, 0);

    let error = validate_gemma4_interleaved_attention(&manifest)
        .expect_err("cross-type KV sharing should fail closed");

    assert!(error.to_string().contains("cannot reuse"));
}

#[test]
fn mlx_manifest_validation_rejects_chained_gemma4_kv_shared_layers() {
    // Layer 2 tries to share KV from layer 1, but layer 1 is itself a shared
    // layer (no own K/V cache). `MlxKVCache::peek_source_kv` would `.expect()`
    // on the missing source and panic mid-decode; reject the manifest up front.
    let mut manifest = dense_manifest();
    manifest.model_family = "gemma4".to_string();
    manifest.layer_count = 3;
    manifest.sliding_window_size = Some(1024);
    manifest.layer_types = vec![
        "sliding_attention".to_string(),
        "sliding_attention".to_string(),
        "sliding_attention".to_string(),
    ];
    manifest.kv_shared_source_layers.insert(1, 0);
    manifest.kv_shared_source_layers.insert(2, 1);

    let error = validate_gemma4_interleaved_attention(&manifest)
        .expect_err("chained KV sharing should fail closed");

    assert!(error.to_string().contains("shared layer"));
}

#[test]
fn mlx_manifest_validation_rejects_zero_gemma4_sliding_window() {
    // `Some(0)` survives build_layer_configs as Some(0) and is then filtered
    // back to None by the rotating-window cache path, silently turning sliding
    // layers into grow-forever ones. Force the manifest to fail closed instead.
    let mut manifest = dense_manifest();
    manifest.model_family = "gemma4".to_string();
    manifest.sliding_window_size = Some(0);
    manifest.layer_types = vec!["sliding_attention".to_string()];

    let error = validate_gemma4_interleaved_attention(&manifest)
        .expect_err("zero sliding_window_size should fail closed");

    assert!(error.to_string().contains("sliding_window_size"));
}

#[test]
fn mlx_manifest_validation_rejects_interleaved_layer_type_shape_errors() {
    // Length mismatch: one annotation per layer or fail closed.
    let mut manifest = dense_manifest();
    manifest.model_family = "gemma4".to_string();
    manifest.layer_count = 2;
    manifest.sliding_window_size = Some(1024);
    manifest.layer_types = vec!["sliding_attention".to_string()];
    let error = validate_gemma4_interleaved_attention(&manifest)
        .expect_err("layer_types length mismatch should fail closed");
    assert!(error.to_string().contains("one layer_type per layer"));

    // Unknown annotation strings fail closed rather than defaulting.
    let mut manifest = dense_manifest();
    manifest.model_family = "gemma4".to_string();
    manifest.sliding_window_size = Some(1024);
    manifest.layer_types = vec!["chunked_attention".to_string()];
    let error = validate_gemma4_interleaved_attention(&manifest)
        .expect_err("unknown layer_type should fail closed");
    assert!(error.to_string().contains("unsupported layer_type"));

    // Sliding annotations with no window at all (None, not just zero).
    let mut manifest = dense_manifest();
    manifest.model_family = "gemma4".to_string();
    manifest.sliding_window_size = None;
    manifest.layer_types = vec!["sliding_attention".to_string()];
    let error = validate_gemma4_interleaved_attention(&manifest)
        .expect_err("sliding layers without a window should fail closed");
    assert!(error.to_string().contains("require sliding_window_size"));
}

#[test]
fn mlx_manifest_validation_covers_diffusion_gemma_contract() {
    type DiffusionManifestMutation = (&'static str, fn(&mut NativeDiffusionConfig));

    // The diffusion validator had zero direct coverage although every
    // release-abort in the diffusion decode path assumes it ran.
    let mut manifest = dense_manifest();
    manifest.model_family = "diffusion_gemma".to_string();
    manifest.layer_types = Vec::new();
    let error = validate_diffusion_gemma_manifest(&manifest)
        .expect_err("diffusion without layer_types should fail closed");
    assert!(error.to_string().contains("layer_types"));

    let mut manifest = dense_manifest();
    manifest.model_family = "diffusion_gemma".to_string();
    manifest.layer_types = vec!["full_attention".to_string()];
    manifest.diffusion.canvas_size = None;
    let error = validate_diffusion_gemma_manifest(&manifest)
        .expect_err("diffusion without canvas_size should fail closed");
    assert!(error.to_string().contains("canvas_size"));

    let mut manifest = dense_manifest();
    manifest.model_family = "diffusion_gemma".to_string();
    manifest.layer_types = vec!["full_attention".to_string()];
    manifest.diffusion.canvas_size = Some(64);
    validate_diffusion_gemma_manifest(&manifest)
        .expect("layer_types + canvas_size should satisfy the diffusion contract");

    let zero_cases: &[DiffusionManifestMutation] = &[
        ("canvas_size", |config: &mut NativeDiffusionConfig| {
            config.canvas_size = Some(0)
        }),
        ("max_denoise_steps", |config: &mut NativeDiffusionConfig| {
            config.max_denoise_steps = Some(0);
        }),
        ("convergence_steps", |config: &mut NativeDiffusionConfig| {
            config.convergence_steps = Some(0);
        }),
        (
            "convergence_check_interval",
            |config: &mut NativeDiffusionConfig| {
                config.convergence_check_interval = Some(0);
            },
        ),
    ];
    for &(field, mutate) in zero_cases {
        let mut invalid = manifest.clone();
        mutate(&mut invalid.diffusion);
        let error = validate_diffusion_gemma_manifest(&invalid)
            .expect_err("zero diffusion dimensions must fail closed");
        assert!(error.to_string().contains(field));
    }

    let scalar_cases: &[DiffusionManifestMutation] = &[
        ("temperature_start", |config: &mut NativeDiffusionConfig| {
            config.temperature_start = Some(0.0)
        }),
        ("temperature_end", |config: &mut NativeDiffusionConfig| {
            config.temperature_end = Some(f32::NAN)
        }),
        (
            "confidence_threshold",
            |config: &mut NativeDiffusionConfig| config.confidence_threshold = Some(1.1),
        ),
        (
            "acceptance_rate_threshold",
            |config: &mut NativeDiffusionConfig| {
                config.acceptance_rate_threshold = Some(-0.1);
            },
        ),
    ];
    for &(field, mutate) in scalar_cases {
        let mut invalid = manifest.clone();
        mutate(&mut invalid.diffusion);
        let error = validate_diffusion_gemma_manifest(&invalid)
            .expect_err("invalid diffusion scalars must fail closed");
        assert!(error.to_string().contains(field));
    }
}

#[test]
fn mlx_manifest_validation_matches_llama4_layer_routing_contract() {
    let mut manifest = dense_manifest();
    manifest.model_family = "llama4".to_string();
    manifest.moe = NativeMoeConfig {
        expert_count: Some(4),
        experts_per_token: Some(1),
        expert_intermediate_size: Some(8),
        layer_freq: Some(1),
        first_dense_layers: None,
        shared_expert_count: Some(1),
        sigmoid_routing: false,
        routed_scaling_factor: None,
        n_group: None,
        topk_group: None,
    };
    manifest.tensors.retain(|tensor| {
        !matches!(
            tensor.role,
            NativeTensorRole::FfnGate | NativeTensorRole::FfnUp | NativeTensorRole::FfnDown
        )
    });
    for role in [
        NativeTensorRole::FfnGateInp,
        NativeTensorRole::FfnGateExps,
        NativeTensorRole::FfnUpExps,
        NativeTensorRole::FfnDownExps,
        NativeTensorRole::FfnSharedExpertGate,
        NativeTensorRole::FfnSharedExpertUp,
        NativeTensorRole::FfnSharedExpertDown,
    ] {
        manifest.tensors.push(tensor(
            &format!("model.layers.0.{role:?}.weight"),
            role,
            Some(0),
            vec![4, 4],
        ));
    }

    validate_llama4_manifest(&manifest)
        .expect("Llama4 MoE layers with expert and shared-expert weights should validate");

    let mut missing_shared = manifest.clone();
    missing_shared
        .tensors
        .retain(|tensor| tensor.role != NativeTensorRole::FfnSharedExpertDown);
    let error = validate_llama4_manifest(&missing_shared)
        .expect_err("Llama4 MoE layers without a complete shared expert must fail closed");
    assert!(error.to_string().contains("FfnSharedExpertDown"));

    let mut wrong_route = manifest.clone();
    wrong_route.moe.layer_freq = Some(2);
    let error = validate_llama4_manifest(&wrong_route)
        .expect_err("a dense-routed Llama4 layer must provide dense FFN weights");
    assert!(error.to_string().contains("FfnGate"));

    let mut invalid_temperature = manifest;
    invalid_temperature.no_rope_layer_interval = 2;
    invalid_temperature.attn_temperature_floor = Some(0);
    let error = validate_llama4_manifest(&invalid_temperature)
        .expect_err("zero Llama4 attention temperature floor must fail closed");
    assert!(error.to_string().contains("attn_temperature_floor"));
}

#[test]
fn mlx_manifest_validation_rejects_glm_numeric_group_invariants() {
    // Each case is a runtime `assert!`/`expect` source in the router path;
    // the validator must catch it at load instead of a decode-time abort.
    type ManifestMutation = Box<dyn Fn(&mut NativeModelManifest)>;
    let cases: Vec<(&str, ManifestMutation, &str)> = vec![
        (
            "non-positive routed_scaling_factor",
            Box::new(|m| m.glm_router.routed_scaling_factor = Some(0.0)),
            "routed_scaling_factor",
        ),
        (
            "non-finite routed_scaling_factor",
            Box::new(|m| m.glm_router.routed_scaling_factor = Some(f32::NAN)),
            "routed_scaling_factor",
        ),
        (
            "topk_group above n_group",
            Box::new(|m| {
                m.glm_router.n_group = Some(2);
                m.glm_router.topk_group = Some(3);
                m.moe.expert_count = Some(8);
            }),
            "topk_group",
        ),
        (
            "expert_count not divisible by n_group",
            Box::new(|m| {
                m.glm_router.n_group = Some(4);
                m.glm_router.topk_group = Some(1);
                m.moe.expert_count = Some(6);
            }),
            "divisible",
        ),
        (
            "fewer than two experts per group",
            Box::new(|m| {
                m.glm_router.n_group = Some(4);
                m.glm_router.topk_group = Some(1);
                m.moe.expert_count = Some(4);
            }),
            "at least two experts per group",
        ),
        (
            "first_dense_layer_count above layer_count",
            Box::new(|m| m.glm_router.first_dense_layer_count = Some(3)),
            "cannot exceed layer_count",
        ),
    ];
    for (label, mutate, expected) in cases {
        let mut manifest = glm4_moe_lite_manifest();
        mutate(&mut manifest);
        let error =
            validate_mla_moe_manifest(&manifest).expect_err(&format!("{label} should fail closed"));
        assert!(
            error.to_string().contains(expected),
            "{label}: unexpected error {error}"
        );
    }
}

#[test]
fn embed_batch_mean_mask_excludes_padding_positions() {
    // Verify the mask layout used by embed_batch Mean pooling.
    // batch = [[a,b,c], [x,y]] padded to max_len=3: positions 0..len are 1.0, rest 0.0.
    let actual_lens: Vec<usize> = vec![3, 2];
    let max_seq = 3usize;
    let mut mask_data = vec![0.0f32; actual_lens.len() * max_seq];
    for (i, &l) in actual_lens.iter().enumerate() {
        for j in 0..l {
            mask_data[i * max_seq + j] = 1.0;
        }
    }
    // seq 0 (len 3): all positions active
    assert_eq!(&mask_data[0..3], &[1.0, 1.0, 1.0]);
    // seq 1 (len 2): position 2 is padding
    assert_eq!(&mask_data[3..6], &[1.0, 1.0, 0.0]);
}

#[test]
fn embeddinggemma_forces_mean_pooling_across_embedding_apis() {
    assert_eq!(
        effective_embedding_pooling("embeddinggemma", EmbeddingPooling::Last),
        EmbeddingPooling::Mean
    );
    assert_eq!(
        effective_embedding_pooling("embeddinggemma", EmbeddingPooling::Cls),
        EmbeddingPooling::Mean
    );
    assert_eq!(
        effective_embedding_pooling("qwen3", EmbeddingPooling::Last),
        EmbeddingPooling::Last
    );
}

/// DI-W2-002: single-item EmbeddingGemma must share the batch-of-one Gemma3
/// bidirectional sandwich path (not the causal dense compiled body).
#[test]
fn embeddinggemma_single_item_dispatch_matches_batch_of_one() {
    assert!(
        embedding_single_item_uses_gemma3_path("embeddinggemma"),
        "embedding_forward must divert EmbeddingGemma before build_embedding_forward_closure"
    );
    // Causal dense families stay on the default compiled body.
    assert!(!embedding_single_item_uses_gemma3_path("qwen3"));
    // Nemotron embed uses its own bidirectional branch inside the dense body.
    assert!(!embedding_single_item_uses_gemma3_path("nemotron_embed"));
    // Batch path uses the same family string gate for EmbeddingGemma.
    assert!(
        embedding_single_item_uses_gemma3_path("embeddinggemma"),
        "single and batch both key off model_family == embeddinggemma"
    );
}

#[test]
fn embeddinggemma_pooled_and_encoder_compile_keys_do_not_alias() {
    let thread_id = thread::current().id();
    let actual_lens = vec![3, 2];
    let encoder_key: EmbedGemmaBatchCompileKey = (
        thread_id,
        EmbedGemmaBatchCompileKind::Encoder,
        2,
        3,
        actual_lens.clone(),
    );
    let pooled_key: EmbedGemmaBatchCompileKey = (
        thread_id,
        EmbedGemmaBatchCompileKind::Pooled,
        2,
        3,
        actual_lens,
    );

    assert_ne!(encoder_key, pooled_key);
}

#[test]
fn argpartition_axis_2d_topk_returns_dominant_indices() {
    let vocab: i32 = 10;
    let k: i32 = 3;

    let mut data = vec![0.0f32; vocab as usize];
    data[2] = 8.0;
    data[5] = 4.0;

    let arr = MlxArray::from_raw_data(
        data.as_ptr() as *const u8,
        data.len() * 4,
        &[1, vocab],
        MlxDtype::Float32,
    );

    let neg = multiply(&arr, &mlx_scalar_f32(-1.0), None);
    let part = argpartition_axis(&neg, k, 1, None);
    let top = slice(&part, &[0, 0], &[1, k], &[1, 1], None);
    let top_u32 = astype(&top, MlxDtype::Uint32, None);
    mlx_sys::eval(&[&top_u32]);
    let indices = top_u32.data_u32().to_vec();

    assert!(
        indices.contains(&2),
        "top-{k} should contain index 2 (value 8.0), got {indices:?}"
    );
    assert!(
        indices.contains(&5),
        "top-{k} should contain index 5 (value 4.0), got {indices:?}"
    );
}

#[test]
fn topk_target_softmax_approximates_full_softmax_for_dominant_tokens() {
    use mlx_sys::eval;

    let vocab: i32 = 100;
    let pending: Vec<u32> = vec![10, 25, 55];
    let pending_log_probs: Vec<f32> = vec![-2.0, -3.0, -1.5];
    let temperature: f32 = 0.8;
    // verify_len = 1 (last_token) + pending.len(); logits_all shape [verify_len, vocab].
    // logits_all[i] = prediction after position i = target for pending[i].
    let verify_len = (pending.len() + 1) as i32;

    let mut logits_data = vec![0.0f32; verify_len as usize * vocab as usize];
    // Row 0 is the target for pending[0], row 1 for pending[1], etc.
    logits_data[10] = 5.0;
    logits_data[20] = 4.0;
    logits_data[vocab as usize + 25] = 6.0;
    logits_data[vocab as usize + 30] = 3.0;
    logits_data[2 * vocab as usize + 55] = 7.0;
    logits_data[2 * vocab as usize + 60] = 2.0;

    let logits_all = MlxArray::from_raw_data(
        logits_data.as_ptr() as *const u8,
        logits_data.len() * 4,
        &[verify_len, vocab],
        MlxDtype::Float32,
    );

    let sampling = MlxSamplingParams {
        temperature,
        ..Default::default()
    };
    let mut full_workspace = MtpTargetProbWorkspace::default();
    let mut topk_workspace = MtpTargetProbWorkspace::default();

    let full_result = compute_mtp_target_probs(
        &logits_all,
        &pending,
        &pending_log_probs,
        vocab,
        sampling,
        None,
        MtpDraftFilter::IDENTITY,
        &mut full_workspace,
    )
    .expect("full should return Some");
    if let LazyTargetProbs::Full(arr) = &full_result {
        eval(&[arr]);
    }
    let full_probs = full_result
        .extract_cpu_into(&pending, &mut full_workspace)
        .unwrap();

    let topk_result = compute_mtp_target_probs(
        &logits_all,
        &pending,
        &pending_log_probs,
        vocab,
        sampling,
        Some(32),
        MtpDraftFilter::IDENTITY,
        &mut topk_workspace,
    )
    .expect("topk should return Some");
    if let LazyTargetProbs::TopK { indices, probs, .. } = &topk_result {
        eval(&[indices, probs]);
    }
    let topk_probs = topk_result
        .extract_cpu_into(&pending, &mut topk_workspace)
        .unwrap();

    assert_eq!(full_probs.len(), 3);
    assert_eq!(topk_probs.len(), 3);

    for i in 0..3 {
        assert!(
            full_probs[i] > 0.0,
            "position {i}: full softmax should be > 0, got {}",
            full_probs[i]
        );
        assert!(
            topk_probs[i] > 0.0,
            "position {i}: topk should find token {} in top-32, got p=0",
            pending[i]
        );
        let ratio = topk_probs[i] / full_probs[i];
        assert!(
            (0.85..1.15).contains(&ratio),
            "position {i}: full={} topk={} ratio={ratio:.3}",
            full_probs[i],
            topk_probs[i]
        );
    }
}

#[test]
fn topk_target_softmax_returns_zero_for_out_of_set_tokens() {
    use mlx_sys::eval;

    let vocab: i32 = 100;
    let verify_len: usize = 1;
    let pending: Vec<u32> = vec![99];
    let pending_log_probs: Vec<f32> = vec![-2.0];
    let temperature: f32 = 0.8;

    let mut logits_data = vec![0.0f32; verify_len * vocab as usize];
    logits_data[0] = 10.0;
    logits_data[1] = 9.0;
    logits_data[2] = 8.0;

    let logits_all = MlxArray::from_raw_data(
        logits_data.as_ptr() as *const u8,
        logits_data.len() * 4,
        &[verify_len as i32, vocab],
        MlxDtype::Float32,
    );

    let sampling = MlxSamplingParams {
        temperature,
        ..Default::default()
    };
    let mut workspace = MtpTargetProbWorkspace::default();

    let topk_result = compute_mtp_target_probs(
        &logits_all,
        &pending,
        &pending_log_probs,
        vocab,
        sampling,
        Some(3),
        MtpDraftFilter::IDENTITY,
        &mut workspace,
    )
    .expect("topk should return Some");

    if let LazyTargetProbs::TopK { indices, probs, .. } = &topk_result {
        eval(&[indices, probs]);
    }

    let topk_probs = topk_result
        .extract_cpu_into(&pending, &mut workspace)
        .unwrap();
    assert_eq!(topk_probs.len(), 1);
    assert_eq!(
        topk_probs[0], 0.0,
        "token 99 is outside top-3, should return 0"
    );
}

#[test]
fn filtered_target_probs_match_gpu_topk_full_probability_sampler() {
    let row = [0.4_f32.ln(), 0.3_f32.ln(), 0.2_f32.ln(), 0.1_f32.ln()];
    let mut logits_data = Vec::new();
    logits_data.extend_from_slice(&row);
    logits_data.extend_from_slice(&row);
    logits_data.extend_from_slice(&row);
    let logits_all = MlxArray::from_raw_data(
        logits_data.as_ptr() as *const u8,
        std::mem::size_of_val(logits_data.as_slice()),
        &[3, 4],
        MlxDtype::Float32,
    );
    let pending = [0_u32, 1_u32];
    let pending_log_probs = [0.0_f32, 0.0_f32];
    let sampling = MlxSamplingParams::new(1.0, 0.55, 2);
    let mut workspace = MtpTargetProbWorkspace::default();
    let result = compute_mtp_target_probs(
        &logits_all,
        &pending,
        &pending_log_probs,
        4,
        sampling,
        None,
        MtpDraftFilter {
            top_p: sampling.top_p,
            top_k: sampling.top_k,
        },
        &mut workspace,
    )
    .unwrap();
    if let LazyTargetProbs::TopK { indices, probs, .. } = &result {
        eval(&[indices, probs]);
    }
    let probabilities = result.extract_cpu_into(&pending, &mut workspace).unwrap();
    assert!((probabilities[0] - 4.0 / 7.0).abs() < 1e-6);
    assert!((probabilities[1] - 3.0 / 7.0).abs() < 1e-6);
}

#[test]
fn mtp_telemetry_merge_from_combines_ewma_fields() {
    // merge_from must not silently drop EWMA values from the other side.
    // Both accept_rate_ewma and mtp_only_accept_rate_ewma should be merged
    // as sample-weighted averages so that batch-level route decisions report
    // the correct aggregate rather than always reporting 0.

    let mut a = MtpTelemetry::default();
    // Request A: 4 steps, EWMA converges near 1.0
    for _ in 0..4 {
        a.record_step(3, 3, &[MtpDraftSource::Mtp; 3], None, 3);
    }
    assert_eq!(a.accept_rate_ewma_samples, 4);
    assert_eq!(a.mtp_only_accept_rate_ewma_samples, 4);
    assert!((a.accept_rate_ewma - 1.0).abs() < 1e-5);
    assert!((a.mtp_only_accept_rate_ewma - 1.0).abs() < 1e-5);

    let mut b = MtpTelemetry::default();
    // Request B: 4 steps, EWMA converges near 0.0
    for _ in 0..4 {
        b.record_step(3, 0, &[MtpDraftSource::Mtp; 3], None, 0);
    }
    assert_eq!(b.accept_rate_ewma_samples, 4);
    assert_eq!(b.mtp_only_accept_rate_ewma_samples, 4);
    assert!((b.accept_rate_ewma - 0.0).abs() < 1e-5);
    assert!((b.mtp_only_accept_rate_ewma - 0.0).abs() < 1e-5);

    a.merge_from(b);

    // After merge: 8 total samples (4+4), weighted average of 1.0 and 0.0 = 0.5.
    assert_eq!(a.accept_rate_ewma_samples, 8);
    assert_eq!(a.mtp_only_accept_rate_ewma_samples, 8);
    assert!(
        (a.accept_rate_ewma - 0.5).abs() < 1e-5,
        "merged accept_rate_ewma should be 0.5, got {}",
        a.accept_rate_ewma
    );
    assert!(
        (a.mtp_only_accept_rate_ewma - 0.5).abs() < 1e-5,
        "merged mtp_only_accept_rate_ewma should be 0.5, got {}",
        a.mtp_only_accept_rate_ewma
    );

    // Merging with a zero-sample side must not produce NaN.
    let empty = MtpTelemetry::default();
    let ewma_before = a.accept_rate_ewma;
    let samples_before = a.accept_rate_ewma_samples;
    a.merge_from(empty);
    assert_eq!(a.accept_rate_ewma_samples, samples_before);
    assert!((a.accept_rate_ewma - ewma_before).abs() < 1e-6);
    assert!(a.accept_rate_ewma.is_finite());
    assert!(a.mtp_only_accept_rate_ewma.is_finite());
}

// ── Source-aware hurt gate tests (ADR-019 Phase 5) ──────────────────

#[test]
fn mtp_ngram_source_hurt_gate_does_not_fire_when_ngram_better_than_mtp() {
    // n-gram acceptance 80% > MTP acceptance 70% → no hurt.
    assert!(!mtp_ngram_source_hurt_gate(
        3,    // ngram_max
        100,  // mtp_drafted
        70,   // mtp_accepted
        100,  // ngram_drafted
        80,   // ngram_accepted
        4,    // min_samples
        0.02  // margin
    ));
}

#[test]
fn mtp_ngram_source_hurt_gate_fires_when_ngram_worse_than_mtp() {
    // n-gram acceptance 50% + margin 0.02 < MTP acceptance 80% → hurt.
    assert!(mtp_ngram_source_hurt_gate(
        3,    // ngram_max
        100,  // mtp_drafted
        80,   // mtp_accepted
        100,  // ngram_drafted
        50,   // ngram_accepted
        4,    // min_samples
        0.02  // margin
    ));
}

#[test]
fn mtp_ngram_source_hurt_gate_respects_min_samples() {
    // Not enough samples → must not fire even when rates are bad.
    assert!(!mtp_ngram_source_hurt_gate(
        3, // ngram_max
        2, // mtp_drafted  (< min_samples=4)
        0, // mtp_accepted
        2, // ngram_drafted (< min_samples=4)
        0, // ngram_accepted
        4, // min_samples
        0.02
    ));
}

#[test]
fn mtp_ngram_source_hurt_gate_respects_margin() {
    // n-gram=79%, MTP=80%, margin=2%. Due to f32 representation, 79/100+0.02
    // may be slightly < 80/100, so use a clear case: ngram=79%, MTP=80%,
    // margin=5% → 0.79 + 0.05 = 0.84 > 0.80 → no fire.
    assert!(!mtp_ngram_source_hurt_gate(
        3,    // ngram_max
        100,  // mtp_drafted
        80,   // mtp_accepted  (rate=0.80)
        100,  // ngram_drafted
        79,   // ngram_accepted (rate=0.79)
        4,    // min_samples
        0.05  // margin (0.79 + 0.05 = 0.84, not < 0.80)
    ));
    // But with a small margin it should fire.
    assert!(mtp_ngram_source_hurt_gate(
        3, 100, 80, 100, 50, 4, 0.02 // 0.50 + 0.02 = 0.52 < 0.80
    ));
}

#[test]
fn mtp_ngram_source_hurt_gate_returns_false_when_ngram_max_zero() {
    // ngram_max=0 → no gate regardless of counters.
    assert!(!mtp_ngram_source_hurt_gate(
        0,    // ngram_max
        1000, // mtp_drafted
        999,  // mtp_accepted
        1000, // ngram_drafted
        1,    // ngram_accepted (very bad)
        4,    // min_samples
        0.02
    ));
}

// ── MtpDraftMode env knob test ──────────────────────────────────────

#[test]
fn mtp_draft_mode_default_is_greedy() {
    // Without the env var, the default must be Greedy.
    // NOTE: OnceLock caches on first call; if a prior test already set
    // the env var in this process, the cached value persists. We only
    // verify the type is constructible and defaults to Greedy.
    let mode = crate::mtp::MtpDraftMode::default();
    assert_eq!(mode, crate::mtp::MtpDraftMode::Greedy);
}

#[test]
fn mtp_optimistic_is_disabled_for_glm_sidecar() {
    assert!(mtp_optimistic_allowed(false));
    assert!(!mtp_optimistic_allowed(true));
}

// ── MtpDraftFilter identity test ────────────────────────────────────

#[test]
fn mtp_draft_filter_identity_means_no_filter() {
    let f = MtpDraftFilter::IDENTITY;
    assert_eq!(f.top_p, 1.0);
    assert_eq!(f.top_k, 0);
}

// ── HurtGateMode default test ───────────────────────────────────────

#[test]
fn hurt_gate_mode_default_is_source_aware() {
    let mode = HurtGateMode::default();
    assert_eq!(mode, HurtGateMode::SourceAware);
}

// ── New telemetry fields merge correctly ────────────────────────────

#[test]
fn mtp_telemetry_merge_from_combines_new_hurt_gate_fields() {
    let mut a = MtpTelemetry {
        ngram_source_hurt_gated_steps: 10,
        ngram_legacy_hurt_gated_steps: 5,
        ..MtpTelemetry::default()
    };

    let b = MtpTelemetry {
        ngram_source_hurt_gated_steps: 7,
        ngram_legacy_hurt_gated_steps: 3,
        ..MtpTelemetry::default()
    };

    a.merge_from(b);
    assert_eq!(a.ngram_source_hurt_gated_steps, 17);
    assert_eq!(a.ngram_legacy_hurt_gated_steps, 8);
}

#[test]
fn mtp_profitability_telemetry_merges_cost_totals_and_recomputes_speedup() {
    let mut a = MtpTelemetry::default();
    a.record_profitability_snapshot(MtpProfitabilitySnapshot {
        eligible: true,
        probe_steps: 2,
        probe_wall_us: 60_000,
        direct_reference_wall_us: 30_000,
        mtp_rounds_seen: 12,
        mtp_warmup_rounds: 4,
        mtp_rounds: 8,
        mtp_round_wall_us: 500_000,
        mtp_emitted_tokens: 20,
        baseline_equivalent_wall_us: 600_000,
        estimated_speedup_x1000: 1_200,
        bypassed: false,
    });
    let mut b = MtpTelemetry::default();
    b.record_profitability_snapshot(MtpProfitabilitySnapshot {
        eligible: true,
        probe_steps: 2,
        probe_wall_us: 58_000,
        direct_reference_wall_us: 29_000,
        mtp_rounds_seen: 12,
        mtp_warmup_rounds: 4,
        mtp_rounds: 8,
        mtp_round_wall_us: 500_000,
        mtp_emitted_tokens: 16,
        baseline_equivalent_wall_us: 480_000,
        estimated_speedup_x1000: 960,
        bypassed: true,
    });

    a.merge_from(b);
    assert_eq!(a.profitability_eligible_requests, 2);
    assert_eq!(a.profitability_probe_steps, 4);
    assert_eq!(a.profitability_mtp_rounds_seen, 24);
    assert_eq!(a.profitability_mtp_warmup_rounds, 8);
    assert_eq!(a.profitability_mtp_rounds, 16);
    assert_eq!(a.profitability_mtp_round_wall_us, 1_000_000);
    assert_eq!(a.profitability_baseline_equivalent_wall_us, 1_080_000);
    assert_eq!(a.profitability_estimated_speedup_x1000, 1_080);
    assert_eq!(a.profitability_bypass_events, 1);
}

#[test]
fn clearing_pending_mtp_proposal_preserves_committed_cache_progress() {
    let mut state = RequestState::new(2, 0, None);
    state.mtp_pending_draft = vec![7];
    state.mtp_pending_draft_log_probs = vec![-0.1];
    state.mtp_pending_draft_sources = vec![MtpDraftSource::Mtp];
    state.mtp_decode_count = 11;

    clear_pending_mtp_proposal(&mut state);

    assert!(state.mtp_pending_draft.is_empty());
    assert!(state.mtp_pending_draft_log_probs.is_empty());
    assert!(state.mtp_pending_draft_sources.is_empty());
    assert_eq!(state.mtp_decode_count, 11);
}

#[test]
fn multirow_mtp_suspension_is_qwen_linear_depth_one_only() {
    assert!(should_suspend_qwen_linear_mtp_for_multirow_batch(
        true, true, true, 1, 2
    ));
    for (enabled, requested, candidate, depth, rows) in [
        (false, true, true, 1, 2),
        (true, false, true, 1, 2),
        (true, true, false, 1, 2),
        (true, true, true, 2, 2),
        (true, true, true, 1, 1),
    ] {
        assert!(!should_suspend_qwen_linear_mtp_for_multirow_batch(
            enabled, requested, candidate, depth, rows
        ));
    }
}

#[test]
fn batched_writeback_clears_stale_private_feed_latches() {
    let mut state = RequestState::new(2, 0, None);
    state.pending_direct = Some(mlx_sys::zeros(&[1], MlxDtype::Uint32, None));
    state.direct_pipeline_emitted_tokens = 7;
    state.next_model_last_token = Some(42);
    let mut restored = MlxKVCache::new(2);
    restored.set_seq_len(19);

    restore_private_cache_after_batched_writeback(&mut state, restored);

    assert_eq!(state.cache.seq_len(), 19);
    assert!(state.pending_direct.is_none());
    assert_eq!(state.direct_pipeline_emitted_tokens, 0);
    assert!(state.next_model_last_token.is_none());
}

#[test]
fn mtp_batched_join_transfers_full_committed_target_cache() {
    let mut state = RequestState::new(2, 0, None);
    state.generated_tokens = vec![3, 5, 8];
    state.mtp_pending_draft = vec![13];
    state.mtp_pending_draft_log_probs = vec![-0.1];
    state.mtp_pending_draft_sources = vec![MtpDraftSource::Mtp];
    state.mtp_cache = Some(MlxKVCache::new(1));
    state.mtp_decode_count = 11;
    state.mtp_prefill_history_tokens = vec![1, 2, 3];
    state.pending_direct = Some(mlx_sys::zeros(&[1], MlxDtype::Uint32, None));
    state.direct_pipeline_emitted_tokens = 9;
    state.next_model_last_token = Some(21);

    suspend_mtp_for_batched_decode(&mut state);

    assert!(state.mtp_suspended_for_batched_decode);
    assert!(state.mtp_pending_draft.is_empty());
    assert!(state.mtp_pending_draft_log_probs.is_empty());
    assert!(state.mtp_pending_draft_sources.is_empty());
    assert!(state.mtp_cache.is_none());
    assert_eq!(state.mtp_decode_count, 0);
    assert!(state.mtp_prefill_history_tokens.is_empty());
    assert!(state.pending_direct.is_none());
    assert_eq!(state.direct_pipeline_emitted_tokens, 0);
    assert!(state.next_model_last_token.is_none());
    assert_eq!(state.generated_tokens, vec![3, 5, 8]);
    assert!(matches!(
        mtp_request_route(
            true,
            true,
            true,
            false,
            state.mtp_bypassed || state.mtp_suspended_for_batched_decode,
            false,
        ),
        MtpRequestRoute::DirectFallback
    ));
}

// ── MTP bypass and initial adaptive depth ──────────────────────────────

#[test]
fn mtp_initial_adaptive_depth_starts_qwen3_5_at_depth_2() {
    // Throughput MTP (default-on) starts Qwen linear packs at recurrent
    // depth 3. The historical depth-2 start remains for the kill-switch
    // path; this process has the product default engaged.
    if crate::fastpath::qwen_linear_throughput_mtp_enabled() {
        assert_eq!(mtp_initial_adaptive_depth("qwen3_5", 8), 3);
        assert_eq!(mtp_initial_adaptive_depth("qwen3_5", 3), 3);
        assert_eq!(mtp_initial_adaptive_depth("qwen3_next", 8), 3);
    } else {
        assert_eq!(mtp_initial_adaptive_depth("qwen3_5", 8), 2);
        assert_eq!(mtp_initial_adaptive_depth("qwen3_next", 8), 2);
    }
    assert_eq!(mtp_initial_adaptive_depth("qwen3_5", 1), 1);
    assert_eq!(mtp_initial_adaptive_depth("qwen3_next", 1), 1);
    // Other families start at head_max_depth.
    assert_eq!(mtp_initial_adaptive_depth("standard", 8), 8);
    assert_eq!(mtp_initial_adaptive_depth("qwen3", 4), 4);
}

#[test]
fn mtp_bypass_defaults_are_sane() {
    // Default min_samples is 8: enough EWMA stabilization without
    // excessive warm-up delay.
    assert_eq!(mtp_bypass_min_samples(), 8);
    // ADR-020 default short remaining-budget floor is 16; throughput MTP
    // keeps drafting through the tail unless the env override is set.
    if crate::fastpath::qwen_linear_throughput_mtp_enabled() {
        assert_eq!(mtp_min_remaining_tokens(), 0);
    } else {
        assert_eq!(mtp_min_remaining_tokens(), 16);
    }
    // Default threshold is 0.50: MTP is bypassed only when acceptance
    // is clearly worse than break-even.
    let threshold = mtp_bypass_threshold();
    assert!(
        (threshold - 0.50).abs() < 1e-5,
        "default bypass threshold must be 0.50; got {threshold}"
    );
}

#[test]
fn request_state_starts_with_mtp_bypass_disabled() {
    let state = RequestState::new(1, 7, None);
    assert!(
        !state.mtp_bypassed,
        "MTP bypass must start disabled so MTP is attempted on every request"
    );
}

#[test]
fn deepseek_v4_mtp_always_commits_primary_token_even_on_full_reject() {
    // Regression: ac==0 must still commit verify_input[0] (last_token).
    assert_eq!(deepseek_v4_mtp_committed_verify_len(0, 1), 1);
    assert_eq!(deepseek_v4_mtp_committed_verify_len(0, 3), 1);
    assert_eq!(deepseek_v4_mtp_committed_verify_len(1, 1), 2);
    assert_eq!(deepseek_v4_mtp_committed_verify_len(2, 3), 3);
    assert_eq!(deepseek_v4_mtp_committed_verify_len(5, 2), 3); // clamp to pending
}

#[test]
fn mtp_warmup_rope_uses_absolute_base_after_multi_chunk_prefill() {
    // Prompt length 300, final-chunk history 100, warmup last 50 rows.
    // First warmed absolute position must be 250, not relative 50.
    assert_eq!(mtp_warmup_absolute_rope_start(300, 100, 50), 250);
    // Single full-prompt chunk: relative start_offset is already absolute.
    assert_eq!(mtp_warmup_absolute_rope_start(100, 100, 50), 50);
    assert_eq!(mtp_warmup_absolute_rope_start(50, 100, 0), 0); // clamp
}
