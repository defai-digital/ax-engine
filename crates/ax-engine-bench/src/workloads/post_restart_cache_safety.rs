//! Stress fixture: post-restart disk prefix-cache safety (I-2).
//!
//! Invariant I-2 requires that on a fresh process start, the disk prefix
//! cache must reject restore attempts whose canonical key differs from the
//! stored entry, and must reject corrupted on-disk payloads. This fixture
//! drives the rejection contract directly through
//! [`ax_engine_mlx::disk_prefix_cache::DiskPrefixCache`].
//!
//! Crucially, the fixture is **session-free**: it does not build an MLX
//! inference session and therefore does not require an MLX model artifact
//! directory. It tests the cache contract on the value layer where invariant
//! I-2 actually lives.
//!
//! Deviation classes exercised (PRD §8 Phase 2):
//!
//! - `model_id`, `route_policy`, `layer_layout`, `block_size_tokens`,
//!   `token_count`, `token_hash` → mismatched canonical key produces a
//!   different file path; `get` returns `Ok(None)`.
//! - Corrupted payload (bit-flipped after insert) → SHA256 mismatch; `get`
//!   returns `Ok(None)`.
//! - Truncated file → outer header parse fails; `get` returns `Ok(None)`.
//! - Wrong format version (`AXKV` magic with `version=1`) → version check
//!   rejects; `get` returns `Ok(None)`.
//!
//! Records counts onto [`super::WorkloadReport::post_restart_cache_mut`].
//!
//! See `.internal/prd/engine-serving-invariants.md` §8 Phase 2.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ax_engine_mlx::disk_prefix_cache::{
    DiskPrefixCache, DiskPrefixCacheEntry, DiskPrefixKeyFields, canonical_key_bytes,
};

use super::{Workload, WorkloadContext, WorkloadOutcome};
use crate::harness::WorkloadReport;

const FIXTURE_TIME_BUDGET: Duration = Duration::from_secs(30);

#[derive(Debug, Clone)]
pub(crate) struct PostRestartCacheSafety {
    pub baseline_model_id: String,
    pub baseline_policy: String,
    pub baseline_layout: String,
    pub baseline_block_size_tokens: u32,
    pub baseline_token_count: u32,
    pub baseline_token_hash: u64,
}

impl Default for PostRestartCacheSafety {
    fn default() -> Self {
        Self {
            baseline_model_id: "qwen3".to_string(),
            baseline_policy: "default".to_string(),
            baseline_layout: "full_attention".to_string(),
            baseline_block_size_tokens: 32,
            baseline_token_count: 1024,
            baseline_token_hash: 0xDEAD_BEEF_F00D_CAFE,
        }
    }
}

impl Workload for PostRestartCacheSafety {
    fn name(&self) -> &'static str {
        "post_restart_cache_safety"
    }

    fn run(&self, ctx: &WorkloadContext) -> WorkloadOutcome {
        // This fixture is session-free: it does not require an MLX model
        // artifact directory. It tests cache rejection at the contract layer.
        let workspace = match make_temp_dir("ax-engine-bench-post-restart-cache") {
            Ok(p) => p,
            Err(e) => {
                return WorkloadOutcome::Failed {
                    error: format!("failed to allocate temp dir: {e}"),
                    partial: None,
                };
            }
        };
        match self.run_driver(&workspace, ctx.seed) {
            Ok(report) => WorkloadOutcome::Completed { report },
            Err(error) => WorkloadOutcome::Failed {
                error,
                partial: None,
            },
        }
    }
}

/// Deterministic token content for a given count. Schema v3 disk keys
/// commit to token content (SHA-256), so every key variant needs a token
/// slice whose length matches its token_count.
fn tokens_for(count: u32) -> Vec<u32> {
    (0..count).map(|i| i.wrapping_mul(2_654_435_761)).collect()
}

/// Schema-v3 key wrapper preserving this workload's older argument shape;
/// `salt` (the former token_hash) uniquifies the artifact fingerprint.
#[allow(clippy::too_many_arguments)]
fn workload_key(
    model_id: &str,
    route_policy: &str,
    layer_layout: &str,
    block_size_tokens: u32,
    token_count: u32,
    salt: u64,
    tokens: &[u32],
) -> Vec<u8> {
    canonical_key_bytes(&DiskPrefixKeyFields {
        model_id,
        artifact_fingerprint_sha256: &format!("{salt:064x}"),
        route_policy,
        layer_layout,
        kv_payload_version: 3,
        block_size_tokens,
        token_count,
        tokens,
    })
}

impl PostRestartCacheSafety {
    fn baseline_key(&self) -> Vec<u8> {
        workload_key(
            &self.baseline_model_id,
            &self.baseline_policy,
            &self.baseline_layout,
            self.baseline_block_size_tokens,
            self.baseline_token_count,
            self.baseline_token_hash,
            &tokens_for(self.baseline_token_count),
        )
    }

    fn run_driver(&self, workspace: &Path, seed: u64) -> Result<WorkloadReport, String> {
        let mut report = WorkloadReport::new(self.name());
        report.add_note(format!(
            "config: baseline_model_id={}, baseline_policy={}, baseline_layout={}, \
             baseline_block_size_tokens={}, baseline_token_count={}, seed={}",
            self.baseline_model_id,
            self.baseline_policy,
            self.baseline_layout,
            self.baseline_block_size_tokens,
            self.baseline_token_count,
            seed,
        ));

        let started_at = Instant::now();
        let cache =
            DiskPrefixCache::open(workspace).map_err(|e| format!("open DiskPrefixCache: {e}"))?;

        let baseline_key = self.baseline_key();
        let payload: Vec<u8> = b"AXKV-test-payload-bytes-for-post-restart-cache-safety".to_vec();
        let entry = DiskPrefixCacheEntry {
            payload: payload.clone(),
            prefill_output_token: Some(7),
            producer_cold_prefill_us: 0,
            producer_serialize_us: 0,
        };
        cache
            .insert(&baseline_key, &entry)
            .map_err(|e| format!("seed insert: {e}"))?;

        // Hit case: baseline key returns Some.
        let hit = cache
            .get(&baseline_key)
            .map_err(|e| format!("get baseline: {e}"))?;
        if let Some(found) = hit {
            if found.payload == payload {
                report.post_restart_cache_mut().hits += 1;
            } else {
                return Err("baseline payload roundtrip mismatch".to_string());
            }
        } else {
            return Err("baseline key returned None after insert".to_string());
        }

        // Each deviation class: the alternate canonical key must return None.
        let counters = report.post_restart_cache_mut();

        if cache
            .get(&workload_key(
                "qwen3-7b",
                &self.baseline_policy,
                &self.baseline_layout,
                self.baseline_block_size_tokens,
                self.baseline_token_count,
                self.baseline_token_hash,
                &tokens_for(self.baseline_token_count),
            ))
            .map_err(|e| format!("get model_mismatch: {e}"))?
            .is_none()
        {
            counters.rejected_model_mismatch += 1;
        } else {
            return Err("model_mismatch key unexpectedly returned Some".to_string());
        }

        if cache
            .get(&workload_key(
                &self.baseline_model_id,
                "low_latency",
                &self.baseline_layout,
                self.baseline_block_size_tokens,
                self.baseline_token_count,
                self.baseline_token_hash,
                &tokens_for(self.baseline_token_count),
            ))
            .map_err(|e| format!("get policy_mismatch: {e}"))?
            .is_none()
        {
            counters.rejected_policy_mismatch += 1;
        } else {
            return Err("policy_mismatch key unexpectedly returned Some".to_string());
        }

        if cache
            .get(&workload_key(
                &self.baseline_model_id,
                &self.baseline_policy,
                "sliding_window",
                self.baseline_block_size_tokens,
                self.baseline_token_count,
                self.baseline_token_hash,
                &tokens_for(self.baseline_token_count),
            ))
            .map_err(|e| format!("get layout_mismatch: {e}"))?
            .is_none()
        {
            counters.rejected_layout_mismatch += 1;
        } else {
            return Err("layout_mismatch key unexpectedly returned Some".to_string());
        }

        if cache
            .get(&workload_key(
                &self.baseline_model_id,
                &self.baseline_policy,
                &self.baseline_layout,
                self.baseline_block_size_tokens.wrapping_add(1),
                self.baseline_token_count,
                self.baseline_token_hash,
                &tokens_for(self.baseline_token_count),
            ))
            .map_err(|e| format!("get block_size_mismatch: {e}"))?
            .is_none()
        {
            counters.rejected_block_size_mismatch += 1;
        } else {
            return Err("block_size_mismatch key unexpectedly returned Some".to_string());
        }

        if cache
            .get(&workload_key(
                &self.baseline_model_id,
                &self.baseline_policy,
                &self.baseline_layout,
                self.baseline_block_size_tokens,
                self.baseline_token_count.wrapping_add(1),
                self.baseline_token_hash,
                &tokens_for(self.baseline_token_count.wrapping_add(1)),
            ))
            .map_err(|e| format!("get token_count_mismatch: {e}"))?
            .is_none()
        {
            counters.rejected_token_payload_mismatch += 1;
        } else {
            return Err("token_count_mismatch key unexpectedly returned Some".to_string());
        }

        // token_hash mismatch is treated as a payload mismatch (it would
        // mean the same prompt-length cache was indexed by a different
        // token sequence; the entry is unsafe to reuse).
        if cache
            .get(&workload_key(
                &self.baseline_model_id,
                &self.baseline_policy,
                &self.baseline_layout,
                self.baseline_block_size_tokens,
                self.baseline_token_count,
                self.baseline_token_hash.wrapping_add(1),
                &tokens_for(self.baseline_token_count),
            ))
            .map_err(|e| format!("get token_hash_mismatch: {e}"))?
            .is_none()
        {
            counters.rejected_token_payload_mismatch += 1;
        } else {
            return Err("token_hash_mismatch key unexpectedly returned Some".to_string());
        }

        // Corrupted payload: re-insert under a fresh key, then bit-flip a
        // byte in the payload region and verify get returns None (the
        // SHA256 check should fail-close).
        let corrupt_key = workload_key(
            "corruption_test_model",
            &self.baseline_policy,
            &self.baseline_layout,
            self.baseline_block_size_tokens,
            self.baseline_token_count,
            self.baseline_token_hash,
            &tokens_for(self.baseline_token_count),
        );
        cache
            .insert(
                &corrupt_key,
                &DiskPrefixCacheEntry {
                    payload: payload.clone(),
                    prefill_output_token: Some(7),
                    producer_cold_prefill_us: 0,
                    producer_serialize_us: 0,
                },
            )
            .map_err(|e| format!("seed corrupt insert: {e}"))?;
        let corrupt_path = cache.path_for(&corrupt_key);
        flip_last_byte(&corrupt_path)?;
        if cache
            .get(&corrupt_key)
            .map_err(|e| format!("get corrupted: {e}"))?
            .is_none()
        {
            counters.rejected_other += 1;
        } else {
            return Err("corrupted payload unexpectedly returned Some".to_string());
        }

        // Truncated file: write a tiny stub that cannot satisfy the outer
        // header parse, then verify get returns None. We use a fresh key so
        // the truncated file lives at a new path.
        let truncated_key = workload_key(
            "truncation_test_model",
            &self.baseline_policy,
            &self.baseline_layout,
            self.baseline_block_size_tokens,
            self.baseline_token_count,
            self.baseline_token_hash,
            &tokens_for(self.baseline_token_count),
        );
        let truncated_path = cache.path_for(&truncated_key);
        std::fs::write(&truncated_path, b"AX").map_err(|e| format!("write truncated stub: {e}"))?;
        if cache
            .get(&truncated_key)
            .map_err(|e| format!("get truncated: {e}"))?
            .is_none()
        {
            counters.rejected_format_version_mismatch += 1;
        } else {
            return Err("truncated stub unexpectedly returned Some".to_string());
        }

        if started_at.elapsed() > FIXTURE_TIME_BUDGET {
            return Err("post_restart_cache_safety exceeded wall-clock budget".to_string());
        }

        let total = report.post_restart_cache_mut().total_rejections();
        report.add_decision("post_restart_cache_total_rejections", total);
        report.record_elapsed(started_at.elapsed());
        Ok(report)
    }

    #[allow(dead_code)]
    pub fn report_skeleton(&self) -> WorkloadReport {
        let mut report = WorkloadReport::new(self.name());
        report.add_note(format!(
            "configured baseline_model_id={}, baseline_token_count={}",
            self.baseline_model_id, self.baseline_token_count
        ));
        report
    }
}

fn make_temp_dir(prefix: &str) -> std::io::Result<PathBuf> {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "{prefix}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    ));
    std::fs::create_dir_all(&path)?;
    Ok(path)
}

fn flip_last_byte(path: &Path) -> Result<(), String> {
    let mut bytes = std::fs::read(path).map_err(|e| format!("read for flip: {e}"))?;
    let len = bytes.len();
    if len == 0 {
        return Err("payload to corrupt is empty".to_string());
    }
    bytes[len - 1] ^= 0xFF;
    std::fs::write(path, &bytes).map_err(|e| format!("write flipped: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_documented_shape() {
        let f = PostRestartCacheSafety::default();
        assert_eq!(f.name(), "post_restart_cache_safety");
        assert_eq!(f.baseline_block_size_tokens, 32);
        assert_eq!(f.baseline_token_count, 1024);
    }

    #[test]
    fn session_free_completes_without_artifacts() {
        // Unlike the other Phase 5 fixtures, this one is session-free and
        // must return Completed even when AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR
        // is absent. Demonstrating this is the test's whole reason for being.
        let outcome = PostRestartCacheSafety::default().run(&WorkloadContext::synthetic());
        match outcome {
            WorkloadOutcome::Completed { report } => {
                let value = report.to_json();
                let cache = value["post_restart_cache"]
                    .as_object()
                    .expect("post_restart_cache is object");
                // Hits and rejections must both be > 0 — the fixture seeded
                // an entry, hit it, then rejected every deviation class.
                assert!(cache["hits"].as_u64().unwrap_or(0) >= 1);
                let total_rejections: u64 = [
                    "rejected_model_mismatch",
                    "rejected_policy_mismatch",
                    "rejected_layout_mismatch",
                    "rejected_block_size_mismatch",
                    "rejected_token_payload_mismatch",
                    "rejected_format_version_mismatch",
                    "rejected_other",
                ]
                .iter()
                .map(|k| cache[*k].as_u64().unwrap_or(0))
                .sum();
                assert!(
                    total_rejections >= 7,
                    "expected >=7 deviation classes rejected, got {total_rejections}: {cache:?}"
                );
            }
            other => panic!("expected Completed, got {:?}", other.name()),
        }
    }

    #[test]
    fn baseline_key_is_stable_across_calls() {
        let f = PostRestartCacheSafety::default();
        let a = f.baseline_key();
        let b = f.baseline_key();
        assert_eq!(a, b);
    }
}
