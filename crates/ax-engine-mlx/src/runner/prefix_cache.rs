use super::*;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct MlxPrefixCacheKey {
    pub(crate) model_id: String,
    pub(crate) route_policy: String,
    pub(crate) layer_layout: String,
    pub(crate) block_size_tokens: u32,
    pub(crate) token_count: u32,
    pub(crate) token_hash: u64,
}

#[derive(Clone)]
pub(crate) struct MlxPrefixSnapshot {
    pub(crate) kv_cache_payload: Arc<[u8]>,
    pub(crate) tokens: Vec<u32>,
    pub(crate) token_count: usize,
    pub(crate) bytes: u64,
    pub(crate) greedy_prefill_output_token: Option<u32>,
}

impl MlxPrefixSnapshot {
    /// Build a snapshot around an already-shared payload buffer. The disk
    /// mirror path clones the same `Arc` into its background write job, so
    /// one serialization feeds both tiers without a byte copy.
    pub(crate) fn from_shared_payload(
        payload: Arc<[u8]>,
        tokens: Vec<u32>,
        token_count: usize,
        greedy_prefill_output_token: Option<u32>,
    ) -> Self {
        // Charge the token vector against the byte budget too: a 32K-token
        // prefix carries 128 KiB of tokens per entry, which the payload-only
        // accounting silently exempted from `max_bytes`.
        let bytes = (payload.len() as u64)
            .saturating_add((tokens.len() as u64).saturating_mul(size_of::<u32>() as u64));
        Self {
            kv_cache_payload: payload,
            tokens,
            token_count,
            bytes,
            greedy_prefill_output_token,
        }
    }

    pub(crate) fn rehydrate_cache(
        &self,
    ) -> Result<MlxKVCache, crate::kv_cache::MlxKVCacheSerializeError> {
        MlxKVCache::try_deserialize_from_bytes(&self.kv_cache_payload)
    }
}

pub(crate) struct MlxPrefixCacheEntry {
    pub(crate) snapshot: Arc<MlxPrefixSnapshot>,
    pub(crate) touch_tick: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MlxPrefixCachePolicy {
    pub(crate) max_bytes: u64,
    pub(crate) max_entries: usize,
}

impl MlxPrefixCachePolicy {
    pub(crate) fn from_env() -> Self {
        Self {
            max_bytes: std::env::var("AX_MLX_PREFIX_CACHE_MAX_BYTES")
                .ok()
                .and_then(|raw| raw.parse::<u64>().ok())
                .unwrap_or(DEFAULT_PREFIX_CACHE_MAX_BYTES),
            max_entries: std::env::var("AX_MLX_PREFIX_CACHE_MAX_ENTRIES")
                .ok()
                .and_then(|raw| raw.parse::<usize>().ok())
                .unwrap_or(DEFAULT_PREFIX_CACHE_MAX_ENTRIES),
        }
    }

    pub(crate) fn enabled(self) -> bool {
        self.max_bytes > 0 && self.max_entries > 0
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MlxPrefixCacheStats {
    pub(crate) entries: u32,
    pub(crate) bytes: u64,
}

#[derive(Default)]
pub(crate) struct MlxPrefixCache {
    pub(crate) policy: MlxPrefixCachePolicy,
    pub(crate) entries: HashMap<MlxPrefixCacheKey, MlxPrefixCacheEntry>,
    pub(crate) lru: VecDeque<(MlxPrefixCacheKey, u64)>,
    pub(crate) bytes: u64,
    pub(crate) next_touch_tick: u64,
}

impl MlxPrefixCache {
    pub(crate) fn new(policy: MlxPrefixCachePolicy) -> Self {
        Self {
            policy,
            ..Self::default()
        }
    }

    pub(crate) fn get(
        &mut self,
        key: &MlxPrefixCacheKey,
        requested_tokens: &[u32],
    ) -> Option<Arc<MlxPrefixSnapshot>> {
        if !self.policy.enabled() {
            return None;
        }
        let touch_tick = self.allocate_touch_tick();
        let snapshot = {
            let entry = self.entries.get_mut(key)?;
            if entry.snapshot.tokens.as_slice() != requested_tokens {
                return None;
            }
            entry.touch_tick = touch_tick;
            Arc::clone(&entry.snapshot)
        };
        self.lru.push_back((key.clone(), touch_tick));
        self.compact_stale_lru_if_needed();
        Some(snapshot)
    }

    /// Non-mutating exact membership check. Used by the iterative-chat probe
    /// in `restore_reused_prefix_state` to ask "do we already have a snapshot
    /// for these tokens?" without changing LRU touch ordering. The actual hit
    /// (which does touch the entry) goes through `get`.
    pub(crate) fn contains_exact_tokens(&self, key: &MlxPrefixCacheKey, tokens: &[u32]) -> bool {
        self.policy.enabled()
            && self
                .entries
                .get(key)
                .is_some_and(|entry| entry.snapshot.tokens == tokens)
    }

    /// True when an entry for `key` already holds exactly `tokens` and would
    /// lose nothing if a fresh store were skipped: either the new snapshot
    /// carries no prefill-output token, or the stored one already has one.
    /// Used by the snapshot store to avoid re-cloning and re-serializing the
    /// KV cache for prefixes that are already resident (warm same-prompt
    /// traffic would otherwise pay the full serialize cost every prefill).
    pub(crate) fn contains_superseding_snapshot(
        &self,
        key: &MlxPrefixCacheKey,
        tokens: &[u32],
        prefill_output_token: Option<u32>,
    ) -> bool {
        self.policy.enabled()
            && self.entries.get(key).is_some_and(|entry| {
                entry.snapshot.tokens == tokens
                    && (prefill_output_token.is_none()
                        || entry.snapshot.greedy_prefill_output_token.is_some())
            })
    }

    pub(crate) fn insert(
        &mut self,
        key: MlxPrefixCacheKey,
        snapshot: MlxPrefixSnapshot,
    ) -> MlxPrefixCacheInsertOutcome {
        if !self.policy.enabled() || snapshot.token_count == 0 {
            return MlxPrefixCacheInsertOutcome::default();
        }

        if let Some(previous) = self.entries.remove(&key) {
            self.bytes = self.bytes.saturating_sub(previous.snapshot.bytes);
        }

        let touch_tick = self.allocate_touch_tick();
        self.bytes = self.bytes.saturating_add(snapshot.bytes);
        self.entries.insert(
            key.clone(),
            MlxPrefixCacheEntry {
                snapshot: Arc::new(snapshot),
                touch_tick,
            },
        );
        self.lru.push_back((key.clone(), touch_tick));
        self.compact_stale_lru_if_needed();

        let evictions = self.evict_until_within_policy();
        MlxPrefixCacheInsertOutcome {
            stored: self.entries.contains_key(&key),
            evictions,
        }
    }

    pub(crate) fn stats(&self) -> MlxPrefixCacheStats {
        MlxPrefixCacheStats {
            entries: saturating_u32(self.entries.len()),
            bytes: self.bytes,
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.policy.enabled()
    }

    pub(crate) fn allocate_touch_tick(&mut self) -> u64 {
        let tick = self.next_touch_tick;
        self.next_touch_tick = self.next_touch_tick.wrapping_add(1);
        tick
    }

    pub(crate) fn stale_lru_compaction_limit(&self) -> usize {
        // Scale with the LIVE entry count, not the policy ceiling: with
        // `max_entries` set very large (byte budget doing the real
        // limiting), a policy-derived limit never triggers and the journal
        // grows by one record per get/insert forever. 4× live entries
        // keeps compaction amortized O(1) per operation; the +64 floor
        // avoids thrashing tiny caches.
        self.entries.len().saturating_mul(4).saturating_add(64)
    }

    pub(crate) fn compact_stale_lru_if_needed(&mut self) {
        if self.lru.len() <= self.stale_lru_compaction_limit() {
            return;
        }

        let entries = &self.entries;
        self.lru.retain(|(key, touch_tick)| {
            entries
                .get(key)
                .is_some_and(|entry| entry.touch_tick == *touch_tick)
        });
    }

    pub(crate) fn evict_until_within_policy(&mut self) -> u32 {
        let mut evictions = 0u32;
        while self.bytes > self.policy.max_bytes || self.entries.len() > self.policy.max_entries {
            let Some((key, touch_tick)) = self.lru.pop_front() else {
                break;
            };
            let Some(entry) = self.entries.get(&key) else {
                continue;
            };
            if entry.touch_tick != touch_tick {
                continue;
            }
            if let Some(removed) = self.entries.remove(&key) {
                self.bytes = self.bytes.saturating_sub(removed.snapshot.bytes);
                evictions = evictions.saturating_add(1);
            }
        }
        evictions
    }
}

impl Default for MlxPrefixCachePolicy {
    fn default() -> Self {
        Self {
            max_bytes: DEFAULT_PREFIX_CACHE_MAX_BYTES,
            max_entries: DEFAULT_PREFIX_CACHE_MAX_ENTRIES,
        }
    }
}

/// One queued L2 store: the canonical key plus the same shared payload the
/// L1 snapshot holds.
pub(crate) struct DiskPrefixWriteJob {
    pub(crate) key_bytes: Vec<u8>,
    pub(crate) payload: Arc<[u8]>,
    pub(crate) prefill_output_token: Option<u32>,
}

/// How many stores may wait for the background disk writer before new
/// stores are dropped. Payloads can be hundreds of MiB, so the queue
/// bounds memory, not throughput; a dropped store is safe (disk is
/// strictly additive) and logged.
const DISK_WRITE_QUEUE_DEPTH: usize = 2;

/// Background writer for L2 disk prefix-cache inserts.
///
/// `DiskPrefixCache::insert` runs a full file write + `sync_all`
/// (`F_FULLFSYNC` on macOS) + directory eviction walk under the
/// cross-process lock — previously all inline on the prefill store path.
/// This worker moves that off the request path: `enqueue` is a bounded
/// `try_send` that never blocks, and a full queue drops the newest store
/// with a warning instead of stalling prefill.
///
/// Worker-side eviction counts accumulate in an atomic that the next
/// telemetry-recording request drains, so aggregate metrics stay accurate
/// even though attribution shifts to a later request.
pub(crate) struct DiskPrefixCacheWriter {
    sender: Option<std::sync::mpsc::SyncSender<DiskPrefixWriteJob>>,
    handle: Option<std::thread::JoinHandle<()>>,
    pending_evictions: Arc<std::sync::atomic::AtomicU32>,
}

impl DiskPrefixCacheWriter {
    pub(crate) fn spawn(disk: Arc<crate::disk_prefix_cache::DiskPrefixCache>) -> Option<Self> {
        let (sender, receiver) =
            std::sync::mpsc::sync_channel::<DiskPrefixWriteJob>(DISK_WRITE_QUEUE_DEPTH);
        let pending_evictions = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let worker_evictions = Arc::clone(&pending_evictions);
        let handle = std::thread::Builder::new()
            .name("ax-mlx-disk-prefix-writer".into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    match disk.insert_parts(
                        &job.key_bytes,
                        &job.payload,
                        job.prefill_output_token,
                    ) {
                        Ok(outcome) => {
                            worker_evictions
                                .fetch_add(outcome.evictions, std::sync::atomic::Ordering::Relaxed);
                        }
                        Err(e) => {
                            tracing::warn!(
                                target: "ax_engine_mlx::prefix_cache",
                                error = %e,
                                "background disk prefix-cache insert failed; entry skipped",
                            );
                        }
                    }
                }
            });
        match handle {
            Ok(handle) => Some(Self {
                sender: Some(sender),
                handle: Some(handle),
                pending_evictions,
            }),
            Err(e) => {
                tracing::warn!(
                    target: "ax_engine_mlx::prefix_cache",
                    error = %e,
                    "failed to spawn disk prefix-cache writer; disk stores will run inline",
                );
                None
            }
        }
    }

    /// Queue a store without blocking. Returns `false` (with a warning)
    /// when the queue is full or the worker has exited; the caller treats
    /// that as a skipped store.
    pub(crate) fn enqueue(&self, job: DiskPrefixWriteJob) -> bool {
        let Some(sender) = self.sender.as_ref() else {
            return false;
        };
        match sender.try_send(job) {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!(
                    target: "ax_engine_mlx::prefix_cache",
                    error = %e,
                    "disk prefix-cache write queue rejected a store; entry skipped",
                );
                false
            }
        }
    }

    /// Evictions performed by the worker since the last drain.
    pub(crate) fn drain_evictions(&self) -> u32 {
        self.pending_evictions
            .swap(0, std::sync::atomic::Ordering::Relaxed)
    }
}

impl Drop for DiskPrefixCacheWriter {
    fn drop(&mut self) {
        // Close the channel so the worker drains queued jobs and exits,
        // then join so in-flight writes complete before the cache
        // directory owner goes away.
        drop(self.sender.take());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Cloneable owner for MLX prefix-cache state.
///
/// `MlxRunner` keeps request KV state private, but callers that create a fresh
/// runner per request can pass the same store into each runner so block-aligned
/// prompt snapshots survive across requests.
#[derive(Clone)]
pub struct MlxPrefixCacheStore {
    pub(crate) prefix_cache: Arc<Mutex<MlxPrefixCache>>,
    pub(crate) disk_prefix_cache: Option<Arc<crate::disk_prefix_cache::DiskPrefixCache>>,
    pub(crate) disk_prefix_writer: Option<Arc<DiskPrefixCacheWriter>>,
}

impl fmt::Debug for MlxPrefixCacheStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MlxPrefixCacheStore")
            .field(
                "disk_prefix_cache_enabled",
                &self.disk_prefix_cache.is_some(),
            )
            .finish_non_exhaustive()
    }
}

impl MlxPrefixCacheStore {
    /// Build a prefix-cache store from the documented environment policy.
    pub fn from_env() -> Self {
        let disk_prefix_cache = open_disk_prefix_cache_from_env().map(Arc::new);
        let disk_prefix_writer = disk_prefix_cache
            .as_ref()
            .and_then(|disk| DiskPrefixCacheWriter::spawn(Arc::clone(disk)))
            .map(Arc::new);
        Self {
            prefix_cache: Arc::new(Mutex::new(MlxPrefixCache::new(
                MlxPrefixCachePolicy::from_env(),
            ))),
            disk_prefix_cache,
            disk_prefix_writer,
        }
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        Arc<Mutex<MlxPrefixCache>>,
        Option<Arc<crate::disk_prefix_cache::DiskPrefixCache>>,
        Option<Arc<DiskPrefixCacheWriter>>,
    ) {
        (
            self.prefix_cache,
            self.disk_prefix_cache,
            self.disk_prefix_writer,
        )
    }

    #[cfg(test)]
    pub(crate) fn memory_only_for_tests(policy: MlxPrefixCachePolicy) -> Self {
        Self {
            prefix_cache: Arc::new(Mutex::new(MlxPrefixCache::new(policy))),
            disk_prefix_cache: None,
            disk_prefix_writer: None,
        }
    }
}

pub(crate) fn open_disk_prefix_cache_from_env() -> Option<crate::disk_prefix_cache::DiskPrefixCache>
{
    // F3 M2 — Open the L2 disk prefix cache when an operator has
    // opted in via AX_MLX_PREFIX_CACHE_DIR (and not disabled the
    // disk path via the kill switch). Open failures are non-fatal:
    // we log and fall back to L1-only, since the disk layer is
    // strictly additive on top of the existing in-memory cache.
    match (
        crate::fastpath::prefix_cache_dir(),
        crate::fastpath::prefix_cache_disk_disabled(),
    ) {
        (Some(dir), false) => {
            let policy = crate::disk_prefix_cache::DiskPrefixCachePolicy::from_env();
            if !policy.enabled() {
                tracing::info!(
                    target: "ax_engine_mlx::prefix_cache",
                    "disk prefix cache disabled via zero byte/entry budget",
                );
                return None;
            }
            match crate::disk_prefix_cache::DiskPrefixCache::with_policy(&dir, policy) {
                Ok(c) => Some(c),
                Err(e) => {
                    tracing::warn!(
                        target: "ax_engine_mlx::prefix_cache",
                        error = %e,
                        dir = %dir.display(),
                        "failed to open disk prefix cache; falling back to in-memory only",
                    );
                    None
                }
            }
        }
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MlxPrefixCacheInsertOutcome {
    pub(crate) stored: bool,
    pub(crate) evictions: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MlxPrefixCacheTelemetry {
    pub(crate) hits: u32,
    pub(crate) misses: u32,
    pub(crate) blocked: u32,
    pub(crate) blocked_policy_disabled: u32,
    pub(crate) blocked_unsupported_layout: u32,
    pub(crate) blocked_trim_failure: u32,
    pub(crate) stores: u32,
    pub(crate) evictions: u32,
    pub(crate) reused_tokens: u32,
    pub(crate) warmup_tokens: u32,
    pub(crate) entries: u32,
    pub(crate) bytes: u64,
    // F3 M2 — L2 disk cache counters. These count operations performed
    // by the durable file-backed prefix-cache layer, distinct from the
    // in-memory counters above. `disk_*` events fire only when the
    // disk cache is opened (AX_MLX_PREFIX_CACHE_DIR set + not
    // disabled).
    pub(crate) disk_hits: u32,
    pub(crate) disk_misses: u32,
    pub(crate) disk_inserts: u32,
    pub(crate) disk_insert_bytes: u64,
    pub(crate) disk_evictions: u32,
}

impl MlxPrefixCacheTelemetry {
    pub(crate) fn record_stats(&mut self, stats: MlxPrefixCacheStats) {
        self.entries = stats.entries;
        self.bytes = stats.bytes;
    }

    pub(crate) fn merge_from(&mut self, other: Self) {
        self.hits = self.hits.saturating_add(other.hits);
        self.misses = self.misses.saturating_add(other.misses);
        self.blocked = self.blocked.saturating_add(other.blocked);
        self.blocked_policy_disabled = self
            .blocked_policy_disabled
            .saturating_add(other.blocked_policy_disabled);
        self.blocked_unsupported_layout = self
            .blocked_unsupported_layout
            .saturating_add(other.blocked_unsupported_layout);
        self.blocked_trim_failure = self
            .blocked_trim_failure
            .saturating_add(other.blocked_trim_failure);
        self.stores = self.stores.saturating_add(other.stores);
        self.evictions = self.evictions.saturating_add(other.evictions);
        self.reused_tokens = self.reused_tokens.saturating_add(other.reused_tokens);
        self.warmup_tokens = self.warmup_tokens.saturating_add(other.warmup_tokens);
        self.entries = self.entries.max(other.entries);
        self.bytes = self.bytes.max(other.bytes);
        self.disk_hits = self.disk_hits.saturating_add(other.disk_hits);
        self.disk_misses = self.disk_misses.saturating_add(other.disk_misses);
        self.disk_inserts = self.disk_inserts.saturating_add(other.disk_inserts);
        self.disk_insert_bytes = self
            .disk_insert_bytes
            .saturating_add(other.disk_insert_bytes);
        self.disk_evictions = self.disk_evictions.saturating_add(other.disk_evictions);
    }

    pub(crate) fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
        if *self == Self::default() {
            return;
        }

        let entries = [
            (ROUTE_DECISION_AX_MLX_PREFIX_CACHE_HITS, self.hits),
            (ROUTE_DECISION_AX_MLX_PREFIX_CACHE_MISSES, self.misses),
            (ROUTE_DECISION_AX_MLX_PREFIX_CACHE_BLOCKED, self.blocked),
            (
                ROUTE_DECISION_AX_MLX_PREFIX_CACHE_BLOCKED_POLICY_DISABLED,
                self.blocked_policy_disabled,
            ),
            (
                ROUTE_DECISION_AX_MLX_PREFIX_CACHE_BLOCKED_UNSUPPORTED_LAYOUT,
                self.blocked_unsupported_layout,
            ),
            (
                ROUTE_DECISION_AX_MLX_PREFIX_CACHE_BLOCKED_TRIM_FAILURE,
                self.blocked_trim_failure,
            ),
            (ROUTE_DECISION_AX_MLX_PREFIX_CACHE_STORES, self.stores),
            (ROUTE_DECISION_AX_MLX_PREFIX_CACHE_EVICTIONS, self.evictions),
            (
                ROUTE_DECISION_AX_MLX_PREFIX_CACHE_REUSED_TOKENS,
                self.reused_tokens,
            ),
            (
                ROUTE_DECISION_AX_MLX_PREFIX_CACHE_WARMUP_TOKENS,
                self.warmup_tokens,
            ),
            (ROUTE_DECISION_AX_MLX_PREFIX_CACHE_ENTRIES, self.entries),
            (
                ROUTE_DECISION_AX_MLX_PREFIX_CACHE_BYTES_KIB,
                kib_ceil(self.bytes),
            ),
            (ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_HITS, self.disk_hits),
            (
                ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_MISSES,
                self.disk_misses,
            ),
            (
                ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_INSERTS,
                self.disk_inserts,
            ),
            (
                ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_INSERT_BYTES_KIB,
                kib_ceil(self.disk_insert_bytes),
            ),
            (
                ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_EVICTIONS,
                self.disk_evictions,
            ),
        ];

        for (key, value) in entries {
            decisions.upsert_route_decision(key, value);
        }
    }

    pub(crate) fn record_blocked_policy_disabled(&mut self) {
        self.blocked = self.blocked.saturating_add(1);
        self.blocked_policy_disabled = self.blocked_policy_disabled.saturating_add(1);
    }

    pub(crate) fn record_blocked_unsupported_layout(&mut self) {
        self.blocked = self.blocked.saturating_add(1);
        self.blocked_unsupported_layout = self.blocked_unsupported_layout.saturating_add(1);
    }

    pub(crate) fn record_blocked_trim_failure(&mut self) {
        self.blocked = self.blocked.saturating_add(1);
        self.blocked_trim_failure = self.blocked_trim_failure.saturating_add(1);
    }

    pub(crate) fn record_disk_hit(&mut self) {
        self.disk_hits = self.disk_hits.saturating_add(1);
    }

    pub(crate) fn record_disk_miss(&mut self) {
        self.disk_misses = self.disk_misses.saturating_add(1);
    }

    pub(crate) fn record_disk_insert(&mut self, bytes: u64, evictions: u32) {
        self.disk_inserts = self.disk_inserts.saturating_add(1);
        self.disk_insert_bytes = self.disk_insert_bytes.saturating_add(bytes);
        self.disk_evictions = self.disk_evictions.saturating_add(evictions);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disk_writer_persists_queued_stores_before_shutdown() {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "ax-engine-disk-writer-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let disk =
            Arc::new(crate::disk_prefix_cache::DiskPrefixCache::open(&dir).expect("open disk"));
        let key_bytes = crate::disk_prefix_cache::canonical_key_bytes(
            "writer-model",
            "writer-policy",
            "writer-layout",
            16,
            4,
            0xfeed,
            &[5, 6, 7, 8],
        );
        let payload: Arc<[u8]> = b"writer-payload".to_vec().into();

        let writer = DiskPrefixCacheWriter::spawn(Arc::clone(&disk)).expect("spawn writer");
        assert!(writer.enqueue(DiskPrefixWriteJob {
            key_bytes: key_bytes.clone(),
            payload: Arc::clone(&payload),
            prefill_output_token: Some(321),
        }));
        // Drop joins the worker after it drains the queue, so the entry
        // must be durable and readable afterwards.
        drop(writer);

        let entry = disk
            .get(&key_bytes)
            .expect("get")
            .expect("queued store must be durable after writer shutdown");
        assert_eq!(entry.payload, payload.as_ref());
        assert_eq!(entry.prefill_output_token, Some(321));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
