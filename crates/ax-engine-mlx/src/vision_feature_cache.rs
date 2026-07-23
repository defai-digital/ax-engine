//! Bounded vision-feature cache (WS-M2 / R-M2).
//!
//! Keys are content digests from [`ax_engine_core::media_digest`] (bytes + budget +
//! model fingerprint). Values hold opaque post-connector embedding blobs so the
//! runner can skip tower recompute on hits. The cache is process-local and never
//! persisted.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use ax_engine_core::media_digest;

/// Default entry cap (spec: 64).
pub const DEFAULT_ENTRY_CAP: usize = 64;

/// Default byte budget when env is unset (~256 MiB).
pub const DEFAULT_BYTE_BUDGET: u64 = 256 * 1024 * 1024;

pub const ENV_VISION_FEATURE_CACHE_MB: &str = "AX_MLX_VISION_FEATURE_CACHE_MB";
pub const ENV_VISION_FEATURE_CACHE: &str = "AX_MLX_VISION_FEATURE_CACHE";

#[derive(Clone, Debug)]
pub struct FeatureEntry {
    /// Serialized post-connector embedding (dtype + shape owned by caller).
    pub bytes: Vec<u8>,
    pub soft_token_count: u32,
}

pub struct VisionFeatureCache {
    entries: HashMap<String, FeatureEntry>,
    order: Vec<String>,
    entry_cap: usize,
    byte_budget: u64,
    bytes_used: u64,
    hits: AtomicU64,
    misses: AtomicU64,
    evicts: AtomicU64,
}

impl VisionFeatureCache {
    pub fn new(entry_cap: usize, byte_budget: u64) -> Self {
        Self {
            entries: HashMap::new(),
            order: Vec::new(),
            entry_cap: entry_cap.max(1),
            byte_budget: byte_budget.max(1),
            bytes_used: 0,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evicts: AtomicU64::new(0),
        }
    }

    pub fn from_env() -> Option<Self> {
        if !env_cache_enabled() {
            return None;
        }
        let mb = std::env::var(ENV_VISION_FEATURE_CACHE_MB)
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(256);
        Some(Self::new(DEFAULT_ENTRY_CAP, mb.saturating_mul(1024 * 1024)))
    }

    pub fn key(encoded_bytes: &[u8], soft_token_budget: u32, model_fingerprint: &str) -> String {
        media_digest(encoded_bytes, soft_token_budget, model_fingerprint)
    }

    pub fn get(&mut self, key: &str) -> Option<FeatureEntry> {
        if let Some(entry) = self.entries.get(key).cloned() {
            self.touch(key);
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    pub fn insert(&mut self, key: String, entry: FeatureEntry) {
        let entry_bytes = entry.bytes.len() as u64;
        if entry_bytes > self.byte_budget {
            // Single entry larger than budget: do not store.
            return;
        }
        if let Some(old) = self.entries.remove(&key) {
            self.bytes_used = self.bytes_used.saturating_sub(old.bytes.len() as u64);
            self.order.retain(|k| k != &key);
        }
        while self.entries.len() >= self.entry_cap
            || self.bytes_used.saturating_add(entry_bytes) > self.byte_budget
        {
            if !self.evict_one() {
                break;
            }
        }
        if self.entries.len() >= self.entry_cap
            || self.bytes_used.saturating_add(entry_bytes) > self.byte_budget
        {
            return;
        }
        self.bytes_used = self.bytes_used.saturating_add(entry_bytes);
        self.order.push(key.clone());
        self.entries.insert(key, entry);
    }

    fn touch(&mut self, key: &str) {
        self.order.retain(|k| k != key);
        self.order.push(key.to_string());
    }

    fn evict_one(&mut self) -> bool {
        let Some(old_key) = self.order.first().cloned() else {
            return false;
        };
        self.order.remove(0);
        if let Some(old) = self.entries.remove(&old_key) {
            self.bytes_used = self.bytes_used.saturating_sub(old.bytes.len() as u64);
            self.evicts.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    pub fn stats(&self) -> (u64, u64, u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.evicts.load(Ordering::Relaxed),
            self.bytes_used,
        )
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Process-global cache guarded by a mutex (per-process, never persisted).
pub fn global_vision_feature_cache() -> &'static Mutex<Option<VisionFeatureCache>> {
    static CACHE: std::sync::OnceLock<Mutex<Option<VisionFeatureCache>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(VisionFeatureCache::from_env()))
}

fn env_cache_enabled() -> bool {
    match std::env::var(ENV_VISION_FEATURE_CACHE) {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off" || v == "no")
        }
        Err(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_miss_and_budget_domain_separation() {
        let mut cache = VisionFeatureCache::new(4, 1024 * 1024);
        let k1 = VisionFeatureCache::key(b"img", 280, "fp");
        let k2 = VisionFeatureCache::key(b"img", 560, "fp");
        assert_ne!(k1, k2);
        cache.insert(
            k1.clone(),
            FeatureEntry {
                bytes: vec![1, 2, 3],
                soft_token_count: 280,
            },
        );
        assert!(cache.get(&k1).is_some());
        assert!(cache.get(&k2).is_none());
        let (hits, misses, _, _) = cache.stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
    }

    #[test]
    fn lru_eviction() {
        let mut cache = VisionFeatureCache::new(2, 1024 * 1024);
        cache.insert(
            "a".into(),
            FeatureEntry {
                bytes: vec![1],
                soft_token_count: 1,
            },
        );
        cache.insert(
            "b".into(),
            FeatureEntry {
                bytes: vec![2],
                soft_token_count: 1,
            },
        );
        // Touch a so b is oldest? insert order a,b — touch a, then insert c evicts b?
        // order after insert a,b: [a,b]. touch a -> [b,a]. insert c -> evict b.
        let _ = cache.get("a");
        cache.insert(
            "c".into(),
            FeatureEntry {
                bytes: vec![3],
                soft_token_count: 1,
            },
        );
        assert!(cache.get("a").is_some());
        assert!(cache.get("c").is_some());
        assert!(cache.get("b").is_none());
    }
}
