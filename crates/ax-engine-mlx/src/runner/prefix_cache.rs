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
    pub(crate) fn from_serialized_cache(
        payload: Vec<u8>,
        tokens: Vec<u32>,
        token_count: usize,
        greedy_prefill_output_token: Option<u32>,
    ) -> Self {
        let bytes = payload.len() as u64;
        Self {
            kv_cache_payload: Arc::from(payload.into_boxed_slice()),
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
        self.entries
            .len()
            .max(self.policy.max_entries)
            .max(1)
            .saturating_mul(4)
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

/// Cloneable owner for MLX prefix-cache state.
///
/// `MlxRunner` keeps request KV state private, but callers that create a fresh
/// runner per request can pass the same store into each runner so block-aligned
/// prompt snapshots survive across requests.
#[derive(Clone)]
pub struct MlxPrefixCacheStore {
    pub(crate) prefix_cache: Arc<Mutex<MlxPrefixCache>>,
    pub(crate) disk_prefix_cache: Option<Arc<crate::disk_prefix_cache::DiskPrefixCache>>,
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
        Self {
            prefix_cache: Arc::new(Mutex::new(MlxPrefixCache::new(
                MlxPrefixCachePolicy::from_env(),
            ))),
            disk_prefix_cache: open_disk_prefix_cache_from_env().map(Arc::new),
        }
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        Arc<Mutex<MlxPrefixCache>>,
        Option<Arc<crate::disk_prefix_cache::DiskPrefixCache>>,
    ) {
        (self.prefix_cache, self.disk_prefix_cache)
    }

    #[cfg(test)]
    pub(crate) fn memory_only_for_tests(policy: MlxPrefixCachePolicy) -> Self {
        Self {
            prefix_cache: Arc::new(Mutex::new(MlxPrefixCache::new(policy))),
            disk_prefix_cache: None,
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
