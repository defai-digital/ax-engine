use std::collections::{BTreeMap, BTreeSet};

use crate::ids::{BlockId, CacheGroupId, RequestId};
use thiserror::Error;

const KV_LOW_FREE_BLOCKS_DIVISOR: u32 = 5;

#[inline]
fn kv_diag_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("AX_KV_DIAG").is_ok())
}

macro_rules! kv_diag {
    ($($arg:tt)*) => {
        if kv_diag_enabled() {
            eprintln!("[KV_DIAG] {}", format_args!($($arg)*));
        }
    };
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockTable {
    pub cache_group_id: CacheGroupId,
    pub block_ids: Vec<BlockId>,
    pub logical_token_count: u32,
    pub full_block_count: u32,
    pub partial_block_tokens: u16,
}

impl BlockTable {
    pub fn empty(cache_group_id: CacheGroupId) -> Self {
        Self {
            cache_group_id,
            block_ids: Vec::new(),
            logical_token_count: 0,
            full_block_count: 0,
            partial_block_tokens: 0,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockTableView {
    pub cache_group_id: CacheGroupId,
    pub block_ids: Vec<BlockId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixLookupResult {
    pub matched_blocks: Vec<BlockId>,
    pub matched_token_count: u32,
    pub cache_group_id: CacheGroupId,
    pub hit: bool,
    pub retained_cache_hit: bool,
    cached_block_keys: Vec<CachedBlockKey>,
}

impl PrefixLookupResult {
    pub fn miss(cache_group_id: CacheGroupId) -> Self {
        Self {
            matched_blocks: Vec::new(),
            matched_token_count: 0,
            cache_group_id,
            hit: false,
            retained_cache_hit: false,
            cached_block_keys: Vec::new(),
        }
    }

    pub fn uses_retained_cache(&self) -> bool {
        self.retained_cache_hit
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CachedBlockKey {
    cache_group_id: CacheGroupId,
    block_hash: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CachedBlockEntry {
    block_id: BlockId,
    block_tokens: Vec<u32>,
    parent_block_hash: Option<u64>,
    last_touch_tick: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AppendMode {
    ReuseCurrentPartialBlock,
    AllocateNewBlocks,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationStatus {
    Allocated,
    InsufficientCapacity,
    Deferred,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AllocationPlan {
    pub request_id: RequestId,
    pub new_block_ids: Vec<BlockId>,
    pub append_mode: AppendMode,
    pub allocation_status: AllocationStatus,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FreeResult {
    pub request_id: RequestId,
    pub released_blocks: Vec<BlockId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvManagerConfig {
    pub cache_group_id: CacheGroupId,
    pub block_size_tokens: u32,
    pub total_blocks: u32,
}

impl KvManagerConfig {
    pub fn new(cache_group_id: CacheGroupId, block_size_tokens: u32, total_blocks: u32) -> Self {
        assert!(
            block_size_tokens > 0 && block_size_tokens <= u16::MAX as u32 && total_blocks > 0,
            "block_size_tokens must be in 1..={} and total_blocks must be > 0",
            u16::MAX
        );
        Self {
            cache_group_id,
            block_size_tokens,
            total_blocks,
        }
    }
}

#[derive(Debug)]
pub struct KvManager {
    config: KvManagerConfig,
    free_block_ids: Vec<BlockId>,
    block_tables: BTreeMap<RequestId, BlockTable>,
    prompt_tokens: BTreeMap<RequestId, Vec<u32>>,
    block_ref_counts: BTreeMap<BlockId, u32>,
    live_prefix_requests_by_first_block: BTreeMap<CachedBlockKey, BTreeSet<RequestId>>,
    cached_blocks: BTreeMap<CachedBlockKey, CachedBlockEntry>,
    cached_children_by_parent: BTreeMap<CachedBlockKey, BTreeSet<CachedBlockKey>>,
    next_cache_tick: u64,
    recent_evictions: u32,
}

impl KvManager {
    pub fn new(config: KvManagerConfig) -> Self {
        let free_block_ids = (0..config.total_blocks).rev().map(BlockId).collect();
        Self {
            config,
            free_block_ids,
            block_tables: BTreeMap::new(),
            prompt_tokens: BTreeMap::new(),
            block_ref_counts: BTreeMap::new(),
            live_prefix_requests_by_first_block: BTreeMap::new(),
            cached_blocks: BTreeMap::new(),
            cached_children_by_parent: BTreeMap::new(),
            next_cache_tick: 0,
            recent_evictions: 0,
        }
    }

    pub fn config(&self) -> KvManagerConfig {
        self.config
    }

    pub fn register_request(
        &mut self,
        request_id: RequestId,
        prompt_tokens: Vec<u32>,
    ) -> Result<(), KvManagerError> {
        if self.block_tables.contains_key(&request_id) {
            return Err(KvManagerError::DuplicateRequest(request_id));
        }

        self.block_tables
            .insert(request_id, BlockTable::empty(self.config.cache_group_id));
        self.prompt_tokens.insert(request_id, prompt_tokens);
        Ok(())
    }

    pub fn lookup_prefix(
        &self,
        request_id: RequestId,
        prompt_tokens: &[u32],
    ) -> Result<PrefixLookupResult, KvManagerError> {
        let target_table = self
            .block_tables
            .get(&request_id)
            .ok_or(KvManagerError::UnknownRequest(request_id))?;
        kv_diag!(
            "lookup_prefix(request={}): prompt_tokens.len={} target.logical_token_count={} cached_blocks_total={}",
            request_id.0,
            prompt_tokens.len(),
            target_table.logical_token_count,
            self.cached_blocks.len()
        );
        if target_table.logical_token_count > 0 {
            kv_diag!("lookup_prefix: EARLY MISS (target already has logical tokens)");
            return Ok(PrefixLookupResult::miss(self.config.cache_group_id));
        }
        if !self.prompt_tokens.contains_key(&request_id) {
            return Err(KvManagerError::UnknownRequest(request_id));
        }

        let live_match = self.lookup_live_prefix(request_id, prompt_tokens)?;
        let cached_match = self.lookup_cached_prefix(prompt_tokens)?;
        kv_diag!(
            "lookup_prefix result: live_matched_tokens={} cached_matched_tokens={}",
            live_match.matched_token_count,
            cached_match.matched_token_count
        );

        if cached_match.matched_token_count > live_match.matched_token_count {
            Ok(cached_match)
        } else {
            Ok(live_match)
        }
    }

    fn lookup_live_prefix(
        &self,
        request_id: RequestId,
        prompt_tokens: &[u32],
    ) -> Result<PrefixLookupResult, KvManagerError> {
        let target_full_block_count =
            (prompt_tokens.len() as u32).div_euclid(self.config.block_size_tokens) as usize;
        if target_full_block_count == 0 {
            return Ok(PrefixLookupResult::miss(self.config.cache_group_id));
        }
        let target_block_keys =
            self.full_block_key_sequence(prompt_tokens, target_full_block_count)?;
        let mut best_match = PrefixLookupResult::miss(self.config.cache_group_id);
        let Some(candidate_request_ids) = self
            .live_prefix_requests_by_first_block
            .get(&target_block_keys[0])
        else {
            return Ok(best_match);
        };

        for candidate_request_id in candidate_request_ids {
            if *candidate_request_id == request_id {
                continue;
            }

            let Some(candidate_prompt_tokens) = self.prompt_tokens.get(candidate_request_id) else {
                continue;
            };
            let Some(candidate_table) = self.block_tables.get(candidate_request_id) else {
                continue;
            };
            if candidate_table.full_block_count == 0 {
                continue;
            }

            let candidate_prompt_full_block_count =
                prompt_full_block_count(candidate_prompt_tokens, self.config.block_size_tokens);
            let candidate_block_keys = self.full_block_key_sequence(
                candidate_prompt_tokens,
                (candidate_table.full_block_count as usize).min(candidate_prompt_full_block_count),
            )?;
            let matched_block_count =
                common_prefix_block_count(&candidate_block_keys, &target_block_keys);
            let matched_token_count = matched_block_count as u32 * self.config.block_size_tokens;
            if matched_token_count == 0 || matched_token_count <= best_match.matched_token_count {
                continue;
            }

            best_match = PrefixLookupResult {
                matched_blocks: candidate_table.block_ids[..matched_block_count].to_vec(),
                matched_token_count,
                cache_group_id: self.config.cache_group_id,
                hit: true,
                retained_cache_hit: false,
                cached_block_keys: Vec::new(),
            };
        }

        Ok(best_match)
    }

    pub fn share_prefix(
        &mut self,
        request_id: RequestId,
        lookup: &PrefixLookupResult,
    ) -> Result<(), KvManagerError> {
        self.validate_prefix_share(request_id, lookup)?;

        for block_id in &lookup.matched_blocks {
            let ref_count = self.block_ref_counts.get_mut(block_id).ok_or(
                KvManagerError::InvariantViolation("shared prefix block missing refcount"),
            )?;
            *ref_count += 1;
        }
        for cache_key in &lookup.cached_block_keys {
            let touch_tick = self.allocate_touch_tick();
            let entry =
                self.cached_blocks
                    .get_mut(cache_key)
                    .ok_or(KvManagerError::InvariantViolation(
                        "prefix lookup cache key missing during share",
                    ))?;
            entry.last_touch_tick = touch_tick;
        }

        let matched_block_count = lookup.matched_blocks.len() as u32;
        let table = self
            .block_tables
            .get_mut(&request_id)
            .ok_or(KvManagerError::UnknownRequest(request_id))?;
        table.block_ids = lookup.matched_blocks.clone();
        table.logical_token_count = lookup.matched_token_count;
        table.full_block_count = matched_block_count;
        table.partial_block_tokens = 0;
        self.insert_live_prefix_index(request_id)?;
        Ok(())
    }

    pub fn rollback_prefix_share(
        &mut self,
        request_id: RequestId,
        lookup: &PrefixLookupResult,
    ) -> Result<(), KvManagerError> {
        let table = self
            .block_tables
            .get(&request_id)
            .ok_or(KvManagerError::UnknownRequest(request_id))?;
        if table.block_ids != lookup.matched_blocks
            || table.logical_token_count != lookup.matched_token_count
            || table.partial_block_tokens != 0
        {
            return Err(KvManagerError::InvariantViolation(
                "prefix share rollback target no longer matches lookup",
            ));
        }

        self.remove_live_prefix_index(request_id)?;

        for block_id in lookup.matched_blocks.iter().rev() {
            let ref_count = self.block_ref_counts.get_mut(block_id).ok_or(
                KvManagerError::InvariantViolation("shared prefix block missing refcount"),
            )?;
            if *ref_count == 1 {
                return Err(KvManagerError::InvariantViolation(
                    "shared prefix rollback would release sole block owner",
                ));
            }
            *ref_count -= 1;
        }

        let table = self
            .block_tables
            .get_mut(&request_id)
            .ok_or(KvManagerError::UnknownRequest(request_id))?;
        table.block_ids.clear();
        table.logical_token_count = 0;
        table.full_block_count = 0;
        table.partial_block_tokens = 0;
        Ok(())
    }

    pub fn validate_prefix_share(
        &self,
        request_id: RequestId,
        lookup: &PrefixLookupResult,
    ) -> Result<(), KvManagerError> {
        let matched_block_count = lookup.matched_blocks.len() as u32;
        let expected_token_count = matched_block_count * self.config.block_size_tokens;
        if lookup.matched_token_count != expected_token_count {
            return Err(KvManagerError::InvariantViolation(
                "shared prefix must align to full blocks",
            ));
        }

        let table = self
            .block_tables
            .get(&request_id)
            .ok_or(KvManagerError::UnknownRequest(request_id))?;
        if table.logical_token_count > 0 || !table.block_ids.is_empty() {
            return Err(KvManagerError::InvariantViolation(
                "prefix sharing requires an empty target block table",
            ));
        }

        for block_id in &lookup.matched_blocks {
            if !self.block_ref_counts.contains_key(block_id) {
                return Err(KvManagerError::InvariantViolation(
                    "shared prefix block missing refcount",
                ));
            }
        }
        for cache_key in &lookup.cached_block_keys {
            if !self.cached_blocks.contains_key(cache_key) {
                return Err(KvManagerError::InvariantViolation(
                    "prefix lookup cache key missing during share",
                ));
            }
        }
        Ok(())
    }

    pub fn can_allocate(
        &self,
        request_id: RequestId,
        scheduled_tokens: u32,
    ) -> Result<bool, KvManagerError> {
        let table = self
            .block_tables
            .get(&request_id)
            .ok_or(KvManagerError::UnknownRequest(request_id))?;
        Ok(self.required_new_blocks(table, scheduled_tokens) <= self.free_block_ids.len() as u32)
    }

    pub fn allocate(
        &mut self,
        request_id: RequestId,
        scheduled_tokens: u32,
    ) -> Result<AllocationPlan, KvManagerError> {
        if scheduled_tokens == 0 {
            return Ok(AllocationPlan {
                request_id,
                new_block_ids: Vec::new(),
                append_mode: AppendMode::ReuseCurrentPartialBlock,
                allocation_status: AllocationStatus::Allocated,
            });
        }

        let can_allocate = self.can_allocate(request_id, scheduled_tokens)?;
        if !can_allocate {
            self.evict_cached_prefixes_until(
                self.required_new_blocks_for_request(request_id, scheduled_tokens)?,
            )?;
        }

        let can_allocate = self.can_allocate(request_id, scheduled_tokens)?;
        if !can_allocate {
            return Ok(AllocationPlan {
                request_id,
                new_block_ids: Vec::new(),
                append_mode: AppendMode::AllocateNewBlocks,
                allocation_status: AllocationStatus::InsufficientCapacity,
            });
        }

        let required_new_blocks = {
            let table = self
                .block_tables
                .get(&request_id)
                .ok_or(KvManagerError::UnknownRequest(request_id))?;
            self.required_new_blocks(table, scheduled_tokens) as usize
        };

        let mut new_block_ids = Vec::with_capacity(required_new_blocks);
        for _ in 0..required_new_blocks {
            let block_id = self
                .free_block_ids
                .pop()
                .ok_or(KvManagerError::InvariantViolation("free list underflow"))?;
            if self.block_ref_counts.insert(block_id, 1).is_some() {
                return Err(KvManagerError::InvariantViolation(
                    "allocated block already had refcount state",
                ));
            }
            new_block_ids.push(block_id);
        }

        self.remove_live_prefix_index(request_id)?;
        {
            let table = self
                .block_tables
                .get_mut(&request_id)
                .ok_or(KvManagerError::UnknownRequest(request_id))?;
            table.block_ids.extend(new_block_ids.iter().copied());
            table.logical_token_count = table
                .logical_token_count
                .checked_add(scheduled_tokens)
                .ok_or(KvManagerError::InvariantViolation(
                    "logical_token_count overflow",
                ))?;
            table.full_block_count = table.logical_token_count / self.config.block_size_tokens;
            table.partial_block_tokens =
                (table.logical_token_count % self.config.block_size_tokens) as u16;
        }
        self.insert_live_prefix_index(request_id)?;

        Ok(AllocationPlan {
            request_id,
            append_mode: if new_block_ids.is_empty() {
                AppendMode::ReuseCurrentPartialBlock
            } else {
                AppendMode::AllocateNewBlocks
            },
            new_block_ids,
            allocation_status: AllocationStatus::Allocated,
        })
    }

    pub fn free(&mut self, request_id: RequestId) -> Result<FreeResult, KvManagerError> {
        kv_diag!(
            "free(request={}) called",
            request_id.0
        );
        let table = self
            .block_tables
            .get(&request_id)
            .cloned()
            .ok_or(KvManagerError::UnknownRequest(request_id))?;
        let prompt_tokens = self
            .prompt_tokens
            .get(&request_id)
            .cloned()
            .ok_or(KvManagerError::UnknownRequest(request_id))?;
        kv_diag!(
            "free(request={}): table.full_block_count={} table.logical_token_count={} prompt_tokens.len={}",
            request_id.0,
            table.full_block_count,
            table.logical_token_count,
            prompt_tokens.len()
        );
        self.promote_prompt_prefix_to_cache(&table, &prompt_tokens)?;
        self.remove_live_prefix_index(request_id)?;
        let table = self
            .block_tables
            .remove(&request_id)
            .ok_or(KvManagerError::UnknownRequest(request_id))?;
        let _prompt_tokens = self
            .prompt_tokens
            .remove(&request_id)
            .ok_or(KvManagerError::UnknownRequest(request_id))?;
        let mut released_blocks = Vec::new();

        for block_id in table.block_ids.iter().rev().copied() {
            let Some(ref_count) = self.block_ref_counts.get_mut(&block_id) else {
                return Err(KvManagerError::InvariantViolation(
                    "block refcount missing during free",
                ));
            };
            if *ref_count == 1 {
                self.block_ref_counts.remove(&block_id);
                self.free_block_ids.push(block_id);
                released_blocks.push(block_id);
            } else {
                *ref_count -= 1;
            }
        }

        Ok(FreeResult {
            request_id,
            released_blocks,
        })
    }

    pub fn take_recent_evictions(&mut self) -> u32 {
        std::mem::take(&mut self.recent_evictions)
    }

    pub fn block_table(&self, request_id: RequestId) -> Result<BlockTableView, KvManagerError> {
        let table = self
            .block_tables
            .get(&request_id)
            .ok_or(KvManagerError::UnknownRequest(request_id))?;
        Ok(BlockTableView {
            cache_group_id: table.cache_group_id,
            block_ids: table.block_ids.clone(),
        })
    }

    pub fn block_table_snapshot(
        &self,
        request_id: RequestId,
    ) -> Result<BlockTable, KvManagerError> {
        self.block_tables
            .get(&request_id)
            .cloned()
            .ok_or(KvManagerError::UnknownRequest(request_id))
    }

    pub fn used_block_count(&self) -> u32 {
        self.config.total_blocks - self.available_block_count()
    }

    pub fn block_count_for(&self, request_id: RequestId) -> u32 {
        self.block_tables
            .get(&request_id)
            .map(|table| table.block_ids.len() as u32)
            .unwrap_or(0)
    }

    pub fn available_block_count(&self) -> u32 {
        self.free_block_ids.len() as u32
    }

    pub fn memory_pressure(&self) -> Option<String> {
        let free_blocks = self.available_block_count();
        if free_blocks == 0 {
            if self.select_cached_block_eviction_candidate().is_some() {
                Some("kv_exhausted_reclaimable_cache".into())
            } else {
                Some("kv_exhausted".into())
            }
        } else if u64::from(free_blocks) * u64::from(KV_LOW_FREE_BLOCKS_DIVISOR)
            <= u64::from(self.config.total_blocks)
        {
            Some(format!(
                "kv_low_free_blocks:{}/{}",
                free_blocks, self.config.total_blocks
            ))
        } else {
            None
        }
    }

    fn required_new_blocks(&self, table: &BlockTable, scheduled_tokens: u32) -> u32 {
        if scheduled_tokens == 0 {
            return 0;
        }

        let reusable_partial_capacity = if table.partial_block_tokens == 0 {
            0
        } else {
            self.config.block_size_tokens - u32::from(table.partial_block_tokens)
        };
        let tokens_needing_new_blocks = scheduled_tokens.saturating_sub(reusable_partial_capacity);

        tokens_needing_new_blocks.div_ceil(self.config.block_size_tokens)
    }

    fn required_new_blocks_for_request(
        &self,
        request_id: RequestId,
        scheduled_tokens: u32,
    ) -> Result<u32, KvManagerError> {
        let table = self
            .block_tables
            .get(&request_id)
            .ok_or(KvManagerError::UnknownRequest(request_id))?;
        Ok(self.required_new_blocks(table, scheduled_tokens))
    }

    fn promote_prompt_prefix_to_cache(
        &mut self,
        table: &BlockTable,
        prompt_tokens: &[u32],
    ) -> Result<(), KvManagerError> {
        let prompt_full_token_count = (prompt_tokens.len() as u32)
            .div_euclid(self.config.block_size_tokens)
            * self.config.block_size_tokens;
        let materialized_full_token_count = table.full_block_count * self.config.block_size_tokens;
        let cached_token_count = prompt_full_token_count.min(materialized_full_token_count);
        kv_diag!(
            "promote_prompt_prefix_to_cache: prompt_full_tokens={} materialized_full_tokens={} cached_tokens={} block_size={}",
            prompt_full_token_count,
            materialized_full_token_count,
            cached_token_count,
            self.config.block_size_tokens
        );
        if cached_token_count == 0 {
            kv_diag!("promote: EARLY EXIT (cached_token_count == 0)");
            return Ok(());
        }

        let cached_block_count =
            cached_token_count.div_euclid(self.config.block_size_tokens) as usize;
        let block_keys = self.full_block_keys(prompt_tokens, cached_block_count)?;

        let block_size_tokens = self.config.block_size_tokens as usize;
        let mut newly_inserted = 0u32;
        let mut touched_existing = 0u32;
        for (block_index, ((cache_key, parent_block_hash), block_id)) in block_keys
            .into_iter()
            .zip(table.block_ids.iter().copied())
            .enumerate()
        {
            let start = block_index * block_size_tokens;
            let end = start + block_size_tokens;
            let block_tokens = &prompt_tokens[start..end];
            let touch_tick = self.allocate_touch_tick();
            if let Some(entry) = self.cached_blocks.get_mut(&cache_key) {
                if entry.block_tokens.as_slice() == block_tokens {
                    entry.last_touch_tick = touch_tick;
                    touched_existing += 1;
                } else {
                    break;
                }
                continue;
            }

            let ref_count = self.block_ref_counts.get_mut(&block_id).ok_or(
                KvManagerError::InvariantViolation("cache promotion block missing refcount"),
            )?;
            *ref_count += 1;

            self.insert_cached_block(
                cache_key,
                CachedBlockEntry {
                    block_id,
                    block_tokens: block_tokens.to_vec(),
                    parent_block_hash,
                    last_touch_tick: touch_tick,
                },
            );
            newly_inserted += 1;
        }
        kv_diag!(
            "promote: COMPLETE newly_inserted={} touched_existing={} total_cached_blocks_now={}",
            newly_inserted,
            touched_existing,
            self.cached_blocks.len()
        );
        Ok(())
    }

    fn evict_cached_prefixes_until(
        &mut self,
        required_free_blocks: u32,
    ) -> Result<(), KvManagerError> {
        while self.available_block_count() < required_free_blocks {
            let Some(cache_key) = self.select_cached_block_eviction_candidate() else {
                break;
            };
            self.evict_cached_block(&cache_key)?;
        }
        Ok(())
    }

    fn select_cached_block_eviction_candidate(&self) -> Option<CachedBlockKey> {
        let mut oldest_leaf_releasable = None;
        let mut oldest_releasable = None;
        let mut oldest_leaf = None;
        let mut oldest_any = None;

        for (cache_key, entry) in &self.cached_blocks {
            let has_cached_descendant = self
                .cached_children_by_parent
                .get(cache_key)
                .is_some_and(|children| !children.is_empty());
            let releases_block = self.cached_entry_releases_block(entry);

            remember_oldest_cached_block(&mut oldest_any, *cache_key, entry.last_touch_tick);
            if !has_cached_descendant {
                remember_oldest_cached_block(&mut oldest_leaf, *cache_key, entry.last_touch_tick);
            }
            if releases_block {
                remember_oldest_cached_block(
                    &mut oldest_releasable,
                    *cache_key,
                    entry.last_touch_tick,
                );
            }
            if !has_cached_descendant && releases_block {
                remember_oldest_cached_block(
                    &mut oldest_leaf_releasable,
                    *cache_key,
                    entry.last_touch_tick,
                );
            }
        }

        oldest_leaf_releasable
            .or(oldest_releasable)
            .or(oldest_leaf)
            .or(oldest_any)
            .map(|(cache_key, _)| cache_key)
    }

    fn cached_entry_releases_block(&self, entry: &CachedBlockEntry) -> bool {
        self.block_ref_counts
            .get(&entry.block_id)
            .is_some_and(|ref_count| *ref_count == 1)
    }

    fn live_prefix_index_key(
        &self,
        table: &BlockTable,
        prompt_tokens: &[u32],
    ) -> Result<Option<CachedBlockKey>, KvManagerError> {
        if table.full_block_count == 0
            || prompt_full_block_count(prompt_tokens, self.config.block_size_tokens) == 0
        {
            return Ok(None);
        }

        Ok(self
            .full_block_key_sequence(prompt_tokens, 1)?
            .first()
            .copied())
    }

    fn insert_live_prefix_index(&mut self, request_id: RequestId) -> Result<(), KvManagerError> {
        let table = self
            .block_tables
            .get(&request_id)
            .ok_or(KvManagerError::UnknownRequest(request_id))?;
        let prompt_tokens = self
            .prompt_tokens
            .get(&request_id)
            .ok_or(KvManagerError::UnknownRequest(request_id))?;
        if let Some(first_block_key) = self.live_prefix_index_key(table, prompt_tokens)? {
            self.live_prefix_requests_by_first_block
                .entry(first_block_key)
                .or_default()
                .insert(request_id);
        }
        Ok(())
    }

    fn remove_live_prefix_index(&mut self, request_id: RequestId) -> Result<(), KvManagerError> {
        let table = self
            .block_tables
            .get(&request_id)
            .ok_or(KvManagerError::UnknownRequest(request_id))?;
        let prompt_tokens = self
            .prompt_tokens
            .get(&request_id)
            .ok_or(KvManagerError::UnknownRequest(request_id))?;
        if let Some(first_block_key) = self.live_prefix_index_key(table, prompt_tokens)? {
            if let Some(requests) = self
                .live_prefix_requests_by_first_block
                .get_mut(&first_block_key)
            {
                requests.remove(&request_id);
                if requests.is_empty() {
                    self.live_prefix_requests_by_first_block
                        .remove(&first_block_key);
                }
            }
        }
        Ok(())
    }

    fn evict_cached_block(&mut self, cache_key: &CachedBlockKey) -> Result<(), KvManagerError> {
        if !self.cached_blocks.contains_key(cache_key) {
            return Err(KvManagerError::InvariantViolation(
                "cache eviction target missing from prefix cache",
            ));
        }

        let mut pending = vec![*cache_key];
        let mut eviction_order = Vec::new();
        while let Some(current) = pending.pop() {
            eviction_order.push(current);
            if let Some(children) = self.cached_children_by_parent.get(&current) {
                pending.extend(children.iter().copied());
            }
        }

        for key in eviction_order {
            if let Some(entry) = self.remove_cached_block(&key) {
                self.release_cached_block_entry(entry)?;
                self.recent_evictions = self.recent_evictions.saturating_add(1);
            }
        }

        Ok(())
    }

    fn insert_cached_block(&mut self, cache_key: CachedBlockKey, entry: CachedBlockEntry) {
        debug_assert!(
            !self.cached_blocks.contains_key(&cache_key),
            "cached block insertion must not replace an existing entry"
        );
        if let Some(parent_key) =
            parent_cache_key(cache_key.cache_group_id, entry.parent_block_hash)
        {
            self.cached_children_by_parent
                .entry(parent_key)
                .or_default()
                .insert(cache_key);
        }
        self.cached_blocks.insert(cache_key, entry);
    }

    fn remove_cached_block(&mut self, cache_key: &CachedBlockKey) -> Option<CachedBlockEntry> {
        let entry = self.cached_blocks.remove(cache_key)?;
        if let Some(parent_key) =
            parent_cache_key(cache_key.cache_group_id, entry.parent_block_hash)
        {
            if let Some(children) = self.cached_children_by_parent.get_mut(&parent_key) {
                children.remove(cache_key);
                if children.is_empty() {
                    self.cached_children_by_parent.remove(&parent_key);
                }
            }
        }
        Some(entry)
    }

    fn release_cached_block_entry(
        &mut self,
        entry: CachedBlockEntry,
    ) -> Result<(), KvManagerError> {
        let Some(ref_count) = self.block_ref_counts.get_mut(&entry.block_id) else {
            return Err(KvManagerError::InvariantViolation(
                "block refcount missing during cache eviction",
            ));
        };
        if *ref_count == 1 {
            self.block_ref_counts.remove(&entry.block_id);
            self.free_block_ids.push(entry.block_id);
        } else {
            *ref_count -= 1;
        }
        Ok(())
    }

    fn allocate_touch_tick(&mut self) -> u64 {
        let tick = self.next_cache_tick;
        self.next_cache_tick = self.next_cache_tick.wrapping_add(1);
        tick
    }

    fn lookup_cached_prefix(
        &self,
        prompt_tokens: &[u32],
    ) -> Result<PrefixLookupResult, KvManagerError> {
        let full_block_count =
            (prompt_tokens.len() as u32).div_euclid(self.config.block_size_tokens) as usize;
        if full_block_count == 0 {
            return Ok(PrefixLookupResult::miss(self.config.cache_group_id));
        }

        let mut matched_blocks = Vec::new();
        let mut cached_block_keys = Vec::new();

        let block_size_tokens = self.config.block_size_tokens as usize;
        for (block_index, cache_key) in self
            .full_block_key_sequence(prompt_tokens, full_block_count)?
            .into_iter()
            .enumerate()
        {
            let Some(entry) = self.cached_blocks.get(&cache_key) else {
                break;
            };
            let start = block_index * block_size_tokens;
            let end = start + block_size_tokens;
            if entry.block_tokens.as_slice() != &prompt_tokens[start..end] {
                break;
            }
            matched_blocks.push(entry.block_id);
            cached_block_keys.push(cache_key);
        }

        if matched_blocks.is_empty() {
            return Ok(PrefixLookupResult::miss(self.config.cache_group_id));
        }

        Ok(PrefixLookupResult {
            matched_token_count: matched_blocks.len() as u32 * self.config.block_size_tokens,
            matched_blocks,
            cache_group_id: self.config.cache_group_id,
            hit: true,
            retained_cache_hit: true,
            cached_block_keys,
        })
    }

    fn full_block_keys(
        &self,
        prompt_tokens: &[u32],
        full_block_count: usize,
    ) -> Result<Vec<(CachedBlockKey, Option<u64>)>, KvManagerError> {
        let available_full_blocks =
            (prompt_tokens.len() as u32).div_euclid(self.config.block_size_tokens) as usize;
        if full_block_count > available_full_blocks {
            return Err(KvManagerError::InvariantViolation(
                "requested block hashes exceed available full prompt blocks",
            ));
        }

        let mut parent_block_hash = None;
        let mut block_keys = Vec::with_capacity(full_block_count);
        let block_size_tokens = self.config.block_size_tokens as usize;
        for block_index in 0..full_block_count {
            let start = block_index * block_size_tokens;
            let end = start + block_size_tokens;
            let block_hash = hash_prefix_block(parent_block_hash, &prompt_tokens[start..end]);
            block_keys.push((
                CachedBlockKey {
                    cache_group_id: self.config.cache_group_id,
                    block_hash,
                },
                parent_block_hash,
            ));
            parent_block_hash = Some(block_hash);
        }

        Ok(block_keys)
    }

    fn full_block_key_sequence(
        &self,
        prompt_tokens: &[u32],
        full_block_count: usize,
    ) -> Result<Vec<CachedBlockKey>, KvManagerError> {
        Ok(self
            .full_block_keys(prompt_tokens, full_block_count)?
            .into_iter()
            .map(|(cache_key, _)| cache_key)
            .collect())
    }
}

fn common_prefix_block_count(left: &[CachedBlockKey], right: &[CachedBlockKey]) -> usize {
    left.iter()
        .zip(right.iter())
        .take_while(|(left, right)| left == right)
        .count()
}

fn parent_cache_key(
    cache_group_id: CacheGroupId,
    parent_block_hash: Option<u64>,
) -> Option<CachedBlockKey> {
    parent_block_hash.map(|block_hash| CachedBlockKey {
        cache_group_id,
        block_hash,
    })
}

fn remember_oldest_cached_block(
    oldest: &mut Option<(CachedBlockKey, u64)>,
    cache_key: CachedBlockKey,
    last_touch_tick: u64,
) {
    if oldest.is_none_or(|(_, oldest_tick)| last_touch_tick < oldest_tick) {
        *oldest = Some((cache_key, last_touch_tick));
    }
}

fn prompt_full_block_count(prompt_tokens: &[u32], block_size_tokens: u32) -> usize {
    (prompt_tokens.len() as u32).div_euclid(block_size_tokens) as usize
}

fn hash_prefix_block(parent_block_hash: Option<u64>, block_tokens: &[u32]) -> u64 {
    let mut hash = parent_block_hash.unwrap_or(14_695_981_039_346_656_037u64);
    hash ^= block_tokens.len() as u64;
    hash = hash.wrapping_mul(1_099_511_628_211u64);
    for token in block_tokens {
        hash ^= u64::from(*token);
        hash = hash.wrapping_mul(1_099_511_628_211u64);
    }
    hash
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum KvManagerError {
    #[error("duplicate KV registration: {0:?}")]
    DuplicateRequest(RequestId),
    #[error("unknown KV request: {0:?}")]
    UnknownRequest(RequestId),
    #[error("KV invariant violation: {0}")]
    InvariantViolation(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager(total_blocks: u32, block_size_tokens: u32) -> KvManager {
        KvManager::new(KvManagerConfig::new(
            CacheGroupId(3),
            block_size_tokens,
            total_blocks,
        ))
    }

    #[test]
    fn allocates_append_only_block_tables() {
        let mut manager = make_manager(8, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4, 5, 6])
            .unwrap();

        let first = manager.allocate(RequestId(1), 3).unwrap();
        let second = manager.allocate(RequestId(1), 3).unwrap();
        let table = manager.block_tables.get(&RequestId(1)).unwrap();

        assert_eq!(first.new_block_ids, vec![BlockId(0)]);
        assert_eq!(second.new_block_ids, vec![BlockId(1)]);
        assert_eq!(table.block_ids, vec![BlockId(0), BlockId(1)]);
        assert_eq!(table.logical_token_count, 6);
        assert_eq!(table.full_block_count, 1);
        assert_eq!(table.partial_block_tokens, 2);
    }

    #[test]
    fn reuses_current_partial_block_before_allocating_more() {
        let mut manager = make_manager(8, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4])
            .unwrap();

        manager.allocate(RequestId(1), 3).unwrap();
        let reuse = manager.allocate(RequestId(1), 1).unwrap();

        assert_eq!(reuse.append_mode, AppendMode::ReuseCurrentPartialBlock);
        assert!(reuse.new_block_ids.is_empty());
        assert_eq!(manager.used_block_count(), 1);
    }

    #[test]
    fn block_count_for_reports_request_block_ownership() {
        let mut manager = make_manager(8, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();

        assert_eq!(manager.block_count_for(RequestId(1)), 0);
        assert_eq!(manager.block_count_for(RequestId(99)), 0);

        manager.allocate(RequestId(1), 8).unwrap();
        assert_eq!(manager.block_count_for(RequestId(1)), 2);

        manager.free(RequestId(1)).unwrap();
        assert_eq!(manager.block_count_for(RequestId(1)), 0);
    }

    #[test]
    fn reports_insufficient_capacity_without_mutation() {
        let mut manager = make_manager(1, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4])
            .unwrap();
        manager
            .register_request(RequestId(2), vec![9, 9, 9, 9])
            .unwrap();

        manager.allocate(RequestId(1), 4).unwrap();
        let denied = manager.allocate(RequestId(2), 1).unwrap();

        assert_eq!(
            denied.allocation_status,
            AllocationStatus::InsufficientCapacity
        );
        assert_eq!(
            manager.block_table(RequestId(2)).unwrap().block_ids,
            Vec::<BlockId>::new()
        );
    }

    #[test]
    fn frees_blocks_in_reverse_request_order() {
        let mut manager = make_manager(8, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4, 5, 6, 7, 8, 9])
            .unwrap();
        manager.allocate(RequestId(1), 9).unwrap();

        let free = manager.free(RequestId(1)).unwrap();

        assert_eq!(free.released_blocks, vec![BlockId(2)]);
        assert_eq!(manager.available_block_count(), 6);
        assert_eq!(manager.used_block_count(), 2);
    }

    #[test]
    fn prefix_lookup_matches_full_block_prefix_from_live_request() {
        let mut manager = make_manager(8, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4, 5, 6, 7, 8, 9])
            .unwrap();
        manager
            .register_request(RequestId(5), vec![1, 2, 3, 4, 5, 6, 7, 8, 99])
            .unwrap();
        manager.allocate(RequestId(1), 9).unwrap();

        let lookup = manager
            .lookup_prefix(RequestId(5), &[1, 2, 3, 4, 5, 6, 7, 8, 99])
            .unwrap();

        assert_eq!(lookup.matched_blocks, vec![BlockId(0), BlockId(1)]);
        assert_eq!(lookup.matched_token_count, 8);
        assert!(lookup.hit);
    }

    #[test]
    fn live_prefix_lookup_uses_first_block_index() {
        let mut manager = make_manager(8, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        manager
            .register_request(RequestId(2), vec![9, 10, 11, 12, 13, 14, 15, 16])
            .unwrap();
        manager
            .register_request(RequestId(3), vec![1, 2, 3, 4, 5, 6, 7, 8, 99])
            .unwrap();
        manager.allocate(RequestId(1), 8).unwrap();
        manager.allocate(RequestId(2), 8).unwrap();

        let shared_first_key = manager.full_block_key_sequence(&[1, 2, 3, 4], 1).unwrap()[0];
        let unrelated_first_key = manager
            .full_block_key_sequence(&[9, 10, 11, 12], 1)
            .unwrap()[0];

        assert_eq!(
            manager
                .live_prefix_requests_by_first_block
                .get(&shared_first_key)
                .cloned(),
            Some(std::collections::BTreeSet::from([RequestId(1)]))
        );
        assert_eq!(
            manager
                .live_prefix_requests_by_first_block
                .get(&unrelated_first_key)
                .cloned(),
            Some(std::collections::BTreeSet::from([RequestId(2)]))
        );

        let lookup = manager
            .lookup_prefix(RequestId(3), &[1, 2, 3, 4, 5, 6, 7, 8, 99])
            .unwrap();

        assert_eq!(lookup.matched_blocks, vec![BlockId(0), BlockId(1)]);
        assert_eq!(lookup.matched_token_count, 8);
        assert!(lookup.hit);
    }

    #[test]
    fn free_removes_request_from_live_prefix_index() {
        let mut manager = make_manager(4, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4])
            .unwrap();
        manager.allocate(RequestId(1), 4).unwrap();

        let first_key = manager.full_block_key_sequence(&[1, 2, 3, 4], 1).unwrap()[0];
        assert_eq!(
            manager
                .live_prefix_requests_by_first_block
                .get(&first_key)
                .cloned(),
            Some(std::collections::BTreeSet::from([RequestId(1)]))
        );

        manager.free(RequestId(1)).unwrap();

        assert!(
            !manager
                .live_prefix_requests_by_first_block
                .contains_key(&first_key)
        );
    }

    #[test]
    fn rollback_prefix_share_removes_target_from_live_prefix_index() {
        let mut manager = make_manager(8, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        manager
            .register_request(RequestId(2), vec![1, 2, 3, 4, 5, 6, 7, 8, 99])
            .unwrap();
        manager.allocate(RequestId(1), 8).unwrap();

        let lookup = manager
            .lookup_prefix(RequestId(2), &[1, 2, 3, 4, 5, 6, 7, 8, 99])
            .unwrap();
        manager.share_prefix(RequestId(2), &lookup).unwrap();

        let first_key = manager.full_block_key_sequence(&[1, 2, 3, 4], 1).unwrap()[0];
        assert_eq!(
            manager
                .live_prefix_requests_by_first_block
                .get(&first_key)
                .cloned(),
            Some(std::collections::BTreeSet::from([
                RequestId(1),
                RequestId(2)
            ]))
        );

        manager
            .rollback_prefix_share(RequestId(2), &lookup)
            .unwrap();

        assert_eq!(
            manager
                .live_prefix_requests_by_first_block
                .get(&first_key)
                .cloned(),
            Some(std::collections::BTreeSet::from([RequestId(1)]))
        );
    }

    #[test]
    fn shared_prefix_reuse_defers_free_until_last_owner_releases_blocks() {
        let mut manager = make_manager(8, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        manager
            .register_request(RequestId(2), vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            .unwrap();
        manager.allocate(RequestId(1), 8).unwrap();

        let lookup = manager
            .lookup_prefix(RequestId(2), &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            .unwrap();
        manager.share_prefix(RequestId(2), &lookup).unwrap();

        assert_eq!(manager.used_block_count(), 2);
        assert_eq!(
            manager.block_table(RequestId(2)).unwrap().block_ids,
            vec![BlockId(0), BlockId(1)]
        );

        let first_free = manager.free(RequestId(1)).unwrap();
        assert!(first_free.released_blocks.is_empty());
        assert_eq!(manager.used_block_count(), 2);

        let second_free = manager.free(RequestId(2)).unwrap();
        assert!(second_free.released_blocks.is_empty());
        assert_eq!(manager.available_block_count(), 6);
        assert_eq!(manager.used_block_count(), 2);
    }

    #[test]
    fn shared_full_prefix_reuses_partial_tail_block_before_allocating_more() {
        let mut manager = make_manager(8, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        manager
            .register_request(RequestId(2), vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            .unwrap();
        manager.allocate(RequestId(1), 8).unwrap();

        let lookup = manager
            .lookup_prefix(RequestId(2), &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            .unwrap();
        manager.share_prefix(RequestId(2), &lookup).unwrap();

        let tail = manager.allocate(RequestId(2), 2).unwrap();
        let reuse = manager.allocate(RequestId(2), 1).unwrap();
        let table = manager.block_tables.get(&RequestId(2)).unwrap();

        assert_eq!(tail.append_mode, AppendMode::AllocateNewBlocks);
        assert_eq!(tail.new_block_ids, vec![BlockId(2)]);
        assert_eq!(reuse.append_mode, AppendMode::ReuseCurrentPartialBlock);
        assert!(reuse.new_block_ids.is_empty());
        assert_eq!(table.block_ids, vec![BlockId(0), BlockId(1), BlockId(2)]);
        assert_eq!(table.logical_token_count, 11);
        assert_eq!(table.full_block_count, 2);
        assert_eq!(table.partial_block_tokens, 3);
        assert_eq!(manager.used_block_count(), 3);
    }

    #[test]
    fn cached_prefix_survives_source_free_and_matches_future_request() {
        let mut manager = make_manager(8, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        manager
            .register_request(RequestId(2), vec![1, 2, 3, 4, 5, 6, 7, 8, 99, 100])
            .unwrap();
        manager.allocate(RequestId(1), 8).unwrap();

        let source_free = manager.free(RequestId(1)).unwrap();
        assert!(source_free.released_blocks.is_empty());

        let lookup = manager
            .lookup_prefix(RequestId(2), &[1, 2, 3, 4, 5, 6, 7, 8, 99, 100])
            .unwrap();
        assert_eq!(lookup.matched_blocks, vec![BlockId(0), BlockId(1)]);
        assert_eq!(lookup.matched_token_count, 8);
        assert!(lookup.uses_retained_cache());
        assert_eq!(lookup.cached_block_keys.len(), 2);
    }

    #[test]
    fn share_prefix_missing_cached_key_does_not_partially_increment_refcounts() {
        let mut manager = make_manager(8, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        manager
            .register_request(RequestId(2), vec![1, 2, 3, 4, 5, 6, 7, 8, 99])
            .unwrap();
        manager.allocate(RequestId(1), 8).unwrap();
        manager.free(RequestId(1)).unwrap();

        let lookup = manager
            .lookup_prefix(RequestId(2), &[1, 2, 3, 4, 5, 6, 7, 8, 99])
            .unwrap();
        let missing_key = lookup.cached_block_keys[0];
        manager.cached_blocks.remove(&missing_key);
        let before_ref_counts = manager.block_ref_counts.clone();

        let error = manager.share_prefix(RequestId(2), &lookup).unwrap_err();

        assert_eq!(
            error,
            KvManagerError::InvariantViolation("prefix lookup cache key missing during share")
        );
        assert_eq!(manager.block_ref_counts, before_ref_counts);
        assert_eq!(
            manager.block_table(RequestId(2)).unwrap().block_ids,
            Vec::<BlockId>::new()
        );
    }

    #[test]
    fn rollback_prefix_share_releases_shared_refs_and_clears_target_table() {
        let mut manager = make_manager(3, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        manager
            .register_request(RequestId(2), vec![1, 2, 3, 4, 5, 6, 7, 8, 9])
            .unwrap();
        manager.allocate(RequestId(1), 8).unwrap();

        let lookup = manager
            .lookup_prefix(RequestId(2), &[1, 2, 3, 4, 5, 6, 7, 8, 9])
            .unwrap();
        manager.share_prefix(RequestId(2), &lookup).unwrap();

        manager
            .rollback_prefix_share(RequestId(2), &lookup)
            .unwrap();

        assert_eq!(
            manager.block_table(RequestId(2)).unwrap().block_ids,
            Vec::<BlockId>::new()
        );
        assert_eq!(manager.available_block_count(), 1);
        let free = manager.free(RequestId(1)).unwrap();
        assert!(free.released_blocks.is_empty());
        assert_eq!(manager.available_block_count(), 1);

        manager
            .register_request(RequestId(3), vec![99, 98, 97, 96, 95])
            .unwrap();
        let allocation = manager.allocate(RequestId(3), 5).unwrap();
        assert_eq!(allocation.allocation_status, AllocationStatus::Allocated);
    }

    #[test]
    fn evicts_cached_prefix_when_new_allocation_needs_capacity() {
        let mut manager = make_manager(2, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4])
            .unwrap();
        manager
            .register_request(RequestId(2), vec![9, 10, 11, 12, 13])
            .unwrap();
        manager.allocate(RequestId(1), 4).unwrap();
        manager.free(RequestId(1)).unwrap();

        let allocation = manager.allocate(RequestId(2), 5).unwrap();

        assert_eq!(allocation.allocation_status, AllocationStatus::Allocated);
        assert_eq!(allocation.new_block_ids, vec![BlockId(0), BlockId(1)]);
        assert_eq!(manager.available_block_count(), 0);
    }

    #[test]
    fn deduplicates_cached_blocks_by_content_hash() {
        let mut manager = make_manager(4, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4])
            .unwrap();
        manager
            .register_request(RequestId(2), vec![1, 2, 3, 4])
            .unwrap();
        manager.allocate(RequestId(1), 4).unwrap();
        manager.allocate(RequestId(2), 4).unwrap();

        manager.free(RequestId(1)).unwrap();
        let second_free = manager.free(RequestId(2)).unwrap();

        assert_eq!(second_free.released_blocks, vec![BlockId(1)]);
        assert_eq!(manager.used_block_count(), 1);
        assert_eq!(manager.available_block_count(), 3);
    }

    #[test]
    fn cached_prefix_lookup_rejects_hash_collision_token_mismatch() {
        let mut manager = make_manager(4, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4])
            .unwrap();
        manager
            .register_request(RequestId(2), vec![1, 2, 3, 4])
            .unwrap();
        manager.allocate(RequestId(1), 4).unwrap();
        manager.free(RequestId(1)).unwrap();

        let cache_key = manager.full_block_key_sequence(&[1, 2, 3, 4], 1).unwrap()[0];
        manager
            .cached_blocks
            .get_mut(&cache_key)
            .expect("cached block")
            .block_tokens = vec![9, 9, 9, 9];

        let lookup = manager.lookup_prefix(RequestId(2), &[1, 2, 3, 4]).unwrap();

        assert!(!lookup.hit);
        assert_eq!(lookup.matched_blocks, Vec::<BlockId>::new());
    }

    #[test]
    fn cache_promotion_stops_after_hash_collision_token_mismatch() {
        let mut manager = make_manager(6, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        manager.allocate(RequestId(1), 8).unwrap();
        manager.free(RequestId(1)).unwrap();

        let keys = manager
            .full_block_key_sequence(&[1, 2, 3, 4, 5, 6, 7, 8], 2)
            .unwrap();
        manager
            .cached_blocks
            .get_mut(&keys[0])
            .expect("cached root block")
            .block_tokens = vec![9, 9, 9, 9];
        let child_touch_tick = manager
            .cached_blocks
            .get(&keys[1])
            .expect("cached child block")
            .last_touch_tick;

        manager
            .register_request(RequestId(2), vec![1, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        manager.allocate(RequestId(2), 8).unwrap();
        manager.free(RequestId(2)).unwrap();

        assert_eq!(
            manager
                .cached_blocks
                .get(&keys[1])
                .expect("cached child block")
                .last_touch_tick,
            child_touch_tick,
            "descendant cache entries must not be refreshed after a parent hash collision"
        );
    }

    #[test]
    fn evicting_cached_root_block_also_evicts_descendants() {
        let mut manager = make_manager(3, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        manager.allocate(RequestId(1), 8).unwrap();
        manager.free(RequestId(1)).unwrap();

        let root_key = manager
            .full_block_key_sequence(&[1, 2, 3, 4, 5, 6, 7, 8], 1)
            .unwrap()[0];
        manager.evict_cached_block(&root_key).unwrap();

        assert_eq!(manager.available_block_count(), 3);
        assert_eq!(manager.used_block_count(), 0);
        assert!(manager.cached_children_by_parent.is_empty());
    }

    #[test]
    fn allocation_eviction_preserves_shorter_cached_prefix_when_leaf_is_enough() {
        let mut manager = make_manager(2, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        manager
            .register_request(RequestId(2), vec![9, 10, 11, 12])
            .unwrap();
        manager
            .register_request(RequestId(3), vec![1, 2, 3, 4, 99])
            .unwrap();
        manager.allocate(RequestId(1), 8).unwrap();
        manager.free(RequestId(1)).unwrap();

        let allocation = manager.allocate(RequestId(2), 4).unwrap();

        assert_eq!(allocation.allocation_status, AllocationStatus::Allocated);
        assert_eq!(manager.take_recent_evictions(), 1);
        assert!(
            !manager
                .cached_children_by_parent
                .contains_key(&manager.full_block_key_sequence(&[1, 2, 3, 4], 1).unwrap()[0])
        );
        let lookup = manager
            .lookup_prefix(RequestId(3), &[1, 2, 3, 4, 99])
            .unwrap();
        assert_eq!(lookup.matched_token_count, 4);
        assert!(lookup.uses_retained_cache());
    }

    #[test]
    fn allocation_eviction_prefers_releasable_cached_leaf_over_live_shared_cache_entry() {
        let mut manager = make_manager(3, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4])
            .unwrap();
        manager
            .register_request(RequestId(2), vec![1, 2, 3, 4, 99])
            .unwrap();
        manager
            .register_request(RequestId(3), vec![9, 10, 11, 12])
            .unwrap();
        manager
            .register_request(RequestId(4), vec![20, 21, 22, 23])
            .unwrap();
        manager
            .register_request(RequestId(5), vec![1, 2, 3, 4, 88])
            .unwrap();

        manager.allocate(RequestId(1), 4).unwrap();
        manager.free(RequestId(1)).unwrap();

        let lookup = manager
            .lookup_prefix(RequestId(2), &[1, 2, 3, 4, 99])
            .unwrap();
        let retained_root_key = manager.full_block_key_sequence(&[1, 2, 3, 4], 1).unwrap()[0];
        manager.share_prefix(RequestId(2), &lookup).unwrap();
        manager.allocate(RequestId(2), 1).unwrap();

        manager.allocate(RequestId(3), 4).unwrap();
        manager.free(RequestId(3)).unwrap();

        assert_eq!(manager.available_block_count(), 0);

        let allocation = manager.allocate(RequestId(4), 4).unwrap();

        assert_eq!(allocation.allocation_status, AllocationStatus::Allocated);
        assert_eq!(manager.take_recent_evictions(), 1);
        assert!(manager.cached_blocks.contains_key(&retained_root_key));
        let retained_lookup = manager
            .lookup_prefix(RequestId(5), &[1, 2, 3, 4, 88])
            .unwrap();
        assert_eq!(retained_lookup.matched_token_count, 4);
    }

    #[test]
    fn rejects_duplicate_request_registration() {
        let mut manager = make_manager(4, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4])
            .unwrap();

        let error = manager
            .register_request(RequestId(1), vec![5, 6, 7, 8])
            .unwrap_err();

        assert_eq!(error, KvManagerError::DuplicateRequest(RequestId(1)));
    }

    #[test]
    fn prefix_lookup_returns_miss_when_prompt_is_shorter_than_one_block() {
        let mut manager = make_manager(4, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4])
            .unwrap();
        manager.register_request(RequestId(2), vec![1, 2]).unwrap();
        manager.allocate(RequestId(1), 4).unwrap();

        let lookup = manager.lookup_prefix(RequestId(2), &[1, 2]).unwrap();

        assert!(!lookup.hit);
        assert_eq!(lookup.matched_token_count, 0);
    }

    #[test]
    fn memory_pressure_reports_exhausted_when_no_free_blocks() {
        let mut manager = make_manager(1, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4])
            .unwrap();
        manager.allocate(RequestId(1), 4).unwrap();

        assert_eq!(manager.memory_pressure(), Some("kv_exhausted".into()));
    }

    #[test]
    fn memory_pressure_reports_reclaimable_cache_when_no_free_blocks_can_evict() {
        let mut manager = make_manager(1, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4])
            .unwrap();
        manager.allocate(RequestId(1), 4).unwrap();
        manager.free(RequestId(1)).unwrap();

        assert_eq!(
            manager.memory_pressure(),
            Some("kv_exhausted_reclaimable_cache".into())
        );
    }

    #[test]
    fn memory_pressure_reports_low_when_below_twenty_percent() {
        let mut manager = make_manager(10, 4);
        manager.register_request(RequestId(1), vec![1; 32]).unwrap();
        manager.allocate(RequestId(1), 32).unwrap();

        assert!(
            manager
                .memory_pressure()
                .unwrap()
                .starts_with("kv_low_free_blocks:")
        );
    }

    #[test]
    fn memory_pressure_returns_none_when_plenty_of_blocks_available() {
        let manager = make_manager(10, 4);

        assert_eq!(manager.memory_pressure(), None);
    }

    #[test]
    fn allocate_zero_tokens_returns_allocated_without_new_blocks() {
        let mut manager = make_manager(4, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4])
            .unwrap();

        let plan = manager.allocate(RequestId(1), 0).unwrap();

        assert_eq!(plan.allocation_status, AllocationStatus::Allocated);
        assert!(plan.new_block_ids.is_empty());
    }

    #[test]
    fn multiple_sequential_evictions_free_oldest_cached_blocks_first() {
        let mut manager = make_manager(2, 4);
        manager
            .register_request(RequestId(1), vec![1, 2, 3, 4])
            .unwrap();
        manager
            .register_request(RequestId(2), vec![5, 6, 7, 8])
            .unwrap();
        manager
            .register_request(RequestId(3), vec![9, 10, 11, 12])
            .unwrap();
        manager.allocate(RequestId(1), 4).unwrap();
        manager.free(RequestId(1)).unwrap();
        manager.allocate(RequestId(2), 4).unwrap();
        manager.free(RequestId(2)).unwrap();

        assert_eq!(manager.available_block_count(), 0);

        let plan = manager.allocate(RequestId(3), 4).unwrap();

        assert_eq!(plan.allocation_status, AllocationStatus::Allocated);
        assert_eq!(plan.new_block_ids.len(), 1);
        assert_eq!(manager.take_recent_evictions(), 1);
    }
}
