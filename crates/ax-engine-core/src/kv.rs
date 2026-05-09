use std::collections::BTreeMap;

use crate::ids::{BlockId, CacheGroupId, RequestId};
use thiserror::Error;

const KV_LOW_FREE_BLOCKS_DIVISOR: u32 = 5;

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
    cached_blocks: BTreeMap<CachedBlockKey, CachedBlockEntry>,
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
            cached_blocks: BTreeMap::new(),
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
        if target_table.logical_token_count > 0 {
            return Ok(PrefixLookupResult::miss(self.config.cache_group_id));
        }
        if !self.prompt_tokens.contains_key(&request_id) {
            return Err(KvManagerError::UnknownRequest(request_id));
        }

        let live_match = self.lookup_live_prefix(request_id, prompt_tokens)?;
        let cached_match = self.lookup_cached_prefix(prompt_tokens)?;

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
        for (candidate_request_id, candidate_prompt_tokens) in &self.prompt_tokens {
            if *candidate_request_id == request_id {
                continue;
            }

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
        self.promote_prompt_prefix_to_cache(&table, &prompt_tokens)?;
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

    pub fn available_block_count(&self) -> u32 {
        self.free_block_ids.len() as u32
    }

    pub fn memory_pressure(&self) -> Option<String> {
        let free_blocks = self.available_block_count();
        if free_blocks == 0 {
            Some("kv_exhausted".into())
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
        if cached_token_count == 0 {
            return Ok(());
        }

        let cached_block_count =
            cached_token_count.div_euclid(self.config.block_size_tokens) as usize;
        let block_keys = self.full_block_keys(prompt_tokens, cached_block_count)?;

        let block_size_tokens = self.config.block_size_tokens as usize;
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
                } else {
                    break;
                }
                continue;
            }

            let ref_count = self.block_ref_counts.get_mut(&block_id).ok_or(
                KvManagerError::InvariantViolation("cache promotion block missing refcount"),
            )?;
            *ref_count += 1;

            self.cached_blocks.insert(
                cache_key,
                CachedBlockEntry {
                    block_id,
                    block_tokens: block_tokens.to_vec(),
                    parent_block_hash,
                    last_touch_tick: touch_tick,
                },
            );
        }
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
        let mut has_cached_descendant = BTreeMap::new();
        for entry in self.cached_blocks.values() {
            if let Some(parent_block_hash) = entry.parent_block_hash {
                has_cached_descendant.insert(parent_block_hash, true);
            }
        }

        self.oldest_cached_block_matching(|cache_key, entry| {
            !has_cached_descendant.contains_key(&cache_key.block_hash)
                && self.cached_entry_releases_block(entry)
        })
        .or_else(|| {
            self.oldest_cached_block_matching(|_, entry| self.cached_entry_releases_block(entry))
        })
        .or_else(|| {
            self.oldest_cached_block_matching(|cache_key, _| {
                !has_cached_descendant.contains_key(&cache_key.block_hash)
            })
        })
        .or_else(|| self.oldest_cached_block_matching(|_, _| true))
    }

    fn oldest_cached_block_matching(
        &self,
        predicate: impl Fn(&CachedBlockKey, &CachedBlockEntry) -> bool,
    ) -> Option<CachedBlockKey> {
        self.cached_blocks
            .iter()
            .filter(|(cache_key, entry)| predicate(cache_key, entry))
            .min_by_key(|(_, entry)| entry.last_touch_tick)
            .map(|(cache_key, _)| *cache_key)
    }

    fn cached_entry_releases_block(&self, entry: &CachedBlockEntry) -> bool {
        self.block_ref_counts
            .get(&entry.block_id)
            .is_some_and(|ref_count| *ref_count == 1)
    }

    fn evict_cached_block(&mut self, cache_key: &CachedBlockKey) -> Result<(), KvManagerError> {
        if !self.cached_blocks.contains_key(cache_key) {
            return Err(KvManagerError::InvariantViolation(
                "cache eviction target missing from prefix cache",
            ));
        }

        let mut descendants_by_parent: BTreeMap<u64, Vec<CachedBlockKey>> = BTreeMap::new();
        for (key, entry) in &self.cached_blocks {
            if let Some(parent_block_hash) = entry.parent_block_hash {
                descendants_by_parent
                    .entry(parent_block_hash)
                    .or_default()
                    .push(*key);
            }
        }

        let mut pending = vec![*cache_key];
        let mut eviction_order = Vec::new();
        while let Some(current) = pending.pop() {
            eviction_order.push(current);
            if let Some(children) = descendants_by_parent.remove(&current.block_hash) {
                pending.extend(children);
            }
        }

        for key in eviction_order {
            if let Some(entry) = self.cached_blocks.remove(&key) {
                self.release_cached_block_entry(entry)?;
                self.recent_evictions = self.recent_evictions.saturating_add(1);
            }
        }

        Ok(())
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
