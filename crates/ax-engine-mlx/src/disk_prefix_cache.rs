//! F3 — disk-backed prefix cache over [`MlxKVCache::serialize_to_bytes`].
//!
//! This module owns the on-disk framing for the future durable
//! prefix-cache layer scoped by
//! `.internal/planning/MLX-DISK-PREFIX-CACHE-PRD-2026-05-14.md`. The
//! current implementation owns file-level get / insert primitives,
//! wire framing around the kv_cache payload, runner-side L2 lookup,
//! and best-effort post-insert eviction. Cross-process locking and
//! full cross-restart promotion validation remain deferred follow-ups
//! per the PRD's milestone breakdown.
//!
//! What is implemented here:
//!   - On-disk file layout: outer magic + version + payload SHA256 +
//!     payload length + payload bytes. The payload is exactly what
//!     `MlxKVCache::serialize_to_bytes` produces.
//!   - Atomic-rename writes (write to a temp file, fsync, rename) so
//!     a torn write cannot leave a half-finished file visible to
//!     readers.
//!   - Read-side integrity check via SHA256 of the payload region. A
//!     mismatched payload fails closed (returns `Ok(None)`) — the
//!     reader treats it like a cache miss.
//!   - SHA256-of-key-canonicalisation as the filename, taking the
//!     same `MlxPrefixCacheKey` fields used by the in-memory cache
//!     so future M2 wire-up only needs to hand the existing key
//!     down. The key bytes are also written into the file header for
//!     hash-collision detection on load.
//!
//! What is explicitly out of scope for this layer:
//!   - Multi-process concurrency / flock (M3). Atomic rename gives
//!     readers a consistent file view; concurrent writers may
//!     overwrite each other's files (harmless when content matches
//!     the same key, and key collisions across writers are
//!     vanishingly unlikely under SHA256).
//!   - Network or mmap I/O (PRD §5.3 explicitly defers mmap to a
//!     follow-up).

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Outer file magic. Distinct from the kv_cache payload's `AXKB` magic
/// so an unframed payload cannot be mistaken for a complete file.
const FILE_MAGIC: &[u8; 4] = b"AXKV";
/// File-format version. Bumped to 2 in F3 M4 to carry the
/// `greedy_prefill_output_token` alongside the kv_cache payload —
/// without it, L2 restores diverged at decode step 0 vs cold prefill
/// for single-block prefixes (see
/// `MLX-F3-DISK-PREFIX-CACHE-M4-FINDINGS-2026-05-14.md` §3).
/// Files written under version 1 are rejected as a miss.
const FILE_VERSION: u32 = 2;
/// Outer header byte count: magic(4) + version(4) + payload_sha256(32) +
/// payload_length(8) + key_len(4) + prefill_token_slot(4) = 56 bytes. The
/// trailing 4-byte slot is u32::MAX when there is no prefill token to
/// carry (e.g. partial-prefix snapshots).
const FIXED_HEADER_LEN: usize = 56;
/// Sentinel for "no prefill output token captured" in the header slot.
const PREFILL_TOKEN_NONE: u32 = u32::MAX;
/// File extension for stored entries.
const ENTRY_EXTENSION: &str = "axkv";

/// Default per-process disk-cache size budget when no env override is
/// set. Matches the value documented in
/// `MLX-DISK-PREFIX-CACHE-PRD-2026-05-14.md` §3.5.
const DEFAULT_DISK_CACHE_MAX_BYTES: u64 = 8 * 1024 * 1024 * 1024; // 8 GiB

/// Default per-process disk-cache entry budget when no env override is
/// set. PRD §3.5.
const DEFAULT_DISK_CACHE_MAX_ENTRIES: usize = 1024;

/// Eviction policy for the L2 disk prefix cache. Both budgets are
/// enforced after every successful insert; whichever fires first
/// drives eviction. Policies are immutable for the cache's lifetime.
#[derive(Clone, Copy, Debug)]
pub struct DiskPrefixCachePolicy {
    /// Maximum aggregate bytes across all entries before eviction
    /// trims the oldest files.
    pub max_bytes: u64,
    /// Maximum entry count before eviction trims the oldest files.
    pub max_entries: usize,
}

impl Default for DiskPrefixCachePolicy {
    fn default() -> Self {
        Self {
            max_bytes: DEFAULT_DISK_CACHE_MAX_BYTES,
            max_entries: DEFAULT_DISK_CACHE_MAX_ENTRIES,
        }
    }
}

impl DiskPrefixCachePolicy {
    /// Build a policy from env, falling back to the documented
    /// defaults when an entry is unset, blank, or unparseable. Reads
    /// `AX_MLX_PREFIX_CACHE_DISK_MAX_BYTES` and
    /// `AX_MLX_PREFIX_CACHE_DISK_MAX_ENTRIES`.
    pub fn from_env() -> Self {
        let max_bytes = std::env::var("AX_MLX_PREFIX_CACHE_DISK_MAX_BYTES")
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_DISK_CACHE_MAX_BYTES);
        let max_entries = std::env::var("AX_MLX_PREFIX_CACHE_DISK_MAX_ENTRIES")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_DISK_CACHE_MAX_ENTRIES);
        Self {
            max_bytes,
            max_entries,
        }
    }
}

/// Errors from [`DiskPrefixCache::insert`] and similar mutating calls.
/// Read-side miss / corruption is signalled as `Ok(None)` instead of
/// an error — readers should treat a corrupt file like a cache miss.
#[derive(Debug)]
pub enum DiskPrefixCacheError {
    /// IO failure on the underlying filesystem.
    Io(std::io::Error),
    /// Cache directory could not be created or accessed.
    BadDirectory(PathBuf),
}

impl From<std::io::Error> for DiskPrefixCacheError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl std::fmt::Display for DiskPrefixCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "disk-cache IO error: {e}"),
            Self::BadDirectory(p) => write!(f, "disk-cache directory invalid: {}", p.display()),
        }
    }
}

impl std::error::Error for DiskPrefixCacheError {}

/// Canonical byte representation of the disk-cache key. The exact
/// shape matches the in-memory `MlxPrefixCacheKey` used by the runner
/// so M2 can plumb the key through without re-serialising it.
///
/// `model_id` / `route_policy` / `layer_layout` are length-prefixed
/// UTF-8 strings so the canonical bytes have no ambiguity around
/// boundary tokens.
pub fn canonical_key_bytes(
    model_id: &str,
    route_policy: &str,
    layer_layout: &str,
    block_size_tokens: u32,
    token_count: u32,
    token_hash: u64,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        4 + 4 + 4 + 8 + 2 + model_id.len() + 2 + route_policy.len() + 2 + layer_layout.len(),
    );
    out.extend_from_slice(&1u32.to_le_bytes()); // schema version of the key encoding
    out.extend_from_slice(&block_size_tokens.to_le_bytes());
    out.extend_from_slice(&token_count.to_le_bytes());
    out.extend_from_slice(&token_hash.to_le_bytes());
    push_lp_string(&mut out, model_id);
    push_lp_string(&mut out, route_policy);
    push_lp_string(&mut out, layer_layout);
    out
}

fn push_lp_string(out: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    let len = u16::try_from(bytes.len()).expect("key string too long");
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(bytes);
}

fn key_sha256_hex(key_bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(key_bytes);
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{:02x}", b)).collect()
}

fn payload_sha256(payload: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(payload);
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// Minimal disk-backed prefix-cache file store.
///
/// Each entry lives in a separate file named after the SHA256 of its
/// canonical key bytes, with extension `.axkv`. Reads parse the outer
/// header, validate the embedded key bytes match the requested key
/// (to defeat hash collisions), validate the payload checksum, and
/// then return the payload bytes for the caller to feed to
/// `MlxKVCache::try_deserialize_from_bytes`.
///
/// F3 M3 — the cache enforces byte and entry budgets on `insert`. When
/// either budget is exceeded after a write, `evict_until_within_policy`
/// walks the directory, sorts files by mtime ascending, and removes
/// the oldest until both budgets are satisfied again. Eviction is
/// best-effort: filesystem errors during stat / remove are logged but
/// do not propagate, and the operator can always rotate the directory
/// manually if eviction stalls.
///
/// Concurrency note: multi-process eviction is not yet protected by a
/// `flock` (PRD §5.1). Two processes evicting concurrently may delete
/// the same files (harmless — `fs::remove_file` on a missing path is
/// gracefully handled) or evict slightly past the budget. The disk
/// cache is correctness-safe under concurrent reads + concurrent
/// writes regardless, because every entry write is atomic-renamed.
pub struct DiskPrefixCache {
    dir: PathBuf,
    policy: DiskPrefixCachePolicy,
}

/// Outcome of [`DiskPrefixCache::insert`], including how many entries
/// were evicted after the write. Callers can plumb the eviction count
/// into telemetry to observe cache pressure.
#[derive(Clone, Copy, Debug, Default)]
pub struct DiskPrefixCacheInsertOutcome {
    /// Number of files removed by post-insert eviction.
    pub evictions: u32,
}

/// One disk-cache entry: the serialized kv_cache payload plus the
/// optional greedy prefill output token that the producing prefill
/// captured. Carrying the token through cross-restart prevents L2
/// hits from diverging at decode step 0 vs cold prefill — see the
/// M4 findings doc.
#[derive(Clone, Debug)]
pub struct DiskPrefixCacheEntry {
    /// `MlxKVCache::serialize_to_bytes` payload.
    pub payload: Vec<u8>,
    /// Greedy prefill output token from the producing prefill, if the
    /// snapshot covers the full prompt. `None` for partial-prefix
    /// snapshots that did not capture a decode-step-0 token.
    pub prefill_output_token: Option<u32>,
}

impl DiskPrefixCache {
    /// Open a disk cache rooted at `dir` with the default policy.
    /// Equivalent to [`DiskPrefixCache::with_policy`] with
    /// `DiskPrefixCachePolicy::default()`. The directory is created
    /// if it does not exist.
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self, DiskPrefixCacheError> {
        Self::with_policy(dir, DiskPrefixCachePolicy::default())
    }

    /// Open a disk cache rooted at `dir` with an explicit policy.
    /// Failures to create or stat the directory are returned as
    /// `BadDirectory`.
    pub fn with_policy(
        dir: impl Into<PathBuf>,
        policy: DiskPrefixCachePolicy,
    ) -> Result<Self, DiskPrefixCacheError> {
        let dir = dir.into();
        if let Err(e) = fs::create_dir_all(&dir) {
            return Err(match e.kind() {
                std::io::ErrorKind::PermissionDenied => DiskPrefixCacheError::BadDirectory(dir),
                _ => e.into(),
            });
        }
        let cache = Self { dir, policy };
        // Apply the configured budgets to whatever already exists in
        // the directory. Operators who shrink AX_MLX_PREFIX_CACHE_DISK_*
        // budgets between runs should see the trim happen at startup
        // rather than only after the next insert. Eviction is
        // best-effort and never propagates errors here.
        let _ = cache.evict_until_within_policy();
        Ok(cache)
    }

    /// Active eviction policy for this cache instance.
    pub fn policy(&self) -> &DiskPrefixCachePolicy {
        &self.policy
    }

    /// Cache directory for inspection by callers / tests.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Filename for a given canonical key.
    pub fn path_for(&self, key_bytes: &[u8]) -> PathBuf {
        self.dir
            .join(format!("{}.{ENTRY_EXTENSION}", key_sha256_hex(key_bytes)))
    }

    /// Cheap existence check: returns `true` when a file for the given
    /// canonical key bytes is present on disk, regardless of whether the
    /// file's header / payload is valid. Used by the L2-aware prefix
    /// probe so the runner can walk block-aligned prefixes without
    /// paying the full read + SHA256-verify cost at every candidate.
    /// The eventual `get` call still validates content, so a stale or
    /// corrupted file flagged by `contains` simply falls back to a
    /// disk_miss + cold-prefill warmup.
    pub fn contains(&self, key_bytes: &[u8]) -> bool {
        self.path_for(key_bytes).is_file()
    }

    /// Look up the entry for `key_bytes`. Returns:
    /// - `Ok(Some(entry))` on a clean hit (with the prefill token
    ///   from the producing prefill, if any);
    /// - `Ok(None)` on miss, hash collision (key bytes differ),
    ///   version mismatch, or on-disk corruption (the caller treats
    ///   this like a miss);
    /// - `Err(_)` only for unrecoverable IO failures (e.g. permission
    ///   denied on a file we know exists).
    pub fn get(
        &self,
        key_bytes: &[u8],
    ) -> Result<Option<DiskPrefixCacheEntry>, DiskPrefixCacheError> {
        let path = self.path_for(key_bytes);
        let raw = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        Ok(parse_file(&raw, key_bytes))
    }

    /// Write `payload` for `key_bytes`. Uses an atomic rename so
    /// readers cannot observe a partial write. The `payload` is
    /// expected to be the output of `MlxKVCache::serialize_to_bytes`,
    /// but the disk cache is content-agnostic and does not parse it.
    ///
    /// After a successful rename, the cache enforces both byte and
    /// entry budgets via `evict_until_within_policy`. The returned
    /// outcome reports the number of evictions so callers can plumb
    /// it into telemetry.
    pub fn insert(
        &self,
        key_bytes: &[u8],
        entry: &DiskPrefixCacheEntry,
    ) -> Result<DiskPrefixCacheInsertOutcome, DiskPrefixCacheError> {
        let final_path = self.path_for(key_bytes);
        let tmp_path = self.dir.join(format!(
            "{}.tmp.{}",
            key_sha256_hex(key_bytes),
            std::process::id()
        ));

        let payload = entry.payload.as_slice();
        let payload_hash = payload_sha256(payload);
        let payload_len = payload.len() as u64;
        let key_len = u32::try_from(key_bytes.len()).expect("key too long");
        let prefill_slot = entry.prefill_output_token.unwrap_or(PREFILL_TOKEN_NONE);

        let mut buf = Vec::with_capacity(FIXED_HEADER_LEN + key_bytes.len() + payload.len());
        buf.extend_from_slice(FILE_MAGIC);
        buf.extend_from_slice(&FILE_VERSION.to_le_bytes());
        buf.extend_from_slice(&payload_hash);
        buf.extend_from_slice(&payload_len.to_le_bytes());
        buf.extend_from_slice(&key_len.to_le_bytes());
        buf.extend_from_slice(&prefill_slot.to_le_bytes());
        buf.extend_from_slice(key_bytes);
        buf.extend_from_slice(payload);

        // RAII guard cleans up the temp file if any step between
        // create and rename fails (out-of-space, permission flip,
        // unexpected unlink). Without this, errored inserts leak
        // `.tmp.<pid>` files that the `.axkv`-only eviction sweep
        // would never reclaim.
        let mut guard = TempFileGuard::new(&tmp_path);
        {
            let mut f = fs::File::create(&tmp_path)?;
            f.write_all(&buf)?;
            f.sync_all()?;
        }
        fs::rename(&tmp_path, &final_path)?;
        guard.disarm();

        let evictions = self.evict_until_within_policy();
        Ok(DiskPrefixCacheInsertOutcome { evictions })
    }

    /// Walk the cache directory and remove `.axkv` entries until both
    /// the byte and entry budgets are satisfied. Files are removed
    /// oldest-mtime first. Errors stat'ing or removing individual
    /// files are absorbed (best-effort eviction); the returned count
    /// reflects how many were successfully removed.
    ///
    /// PRD §6 names this as the "after every insert" callback; calling
    /// it independently (e.g. on demand during a low-traffic window)
    /// is also safe.
    pub fn evict_until_within_policy(&self) -> u32 {
        let mut entries = match self.list_entries() {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!(
                    target: "ax_engine_mlx::prefix_cache",
                    error = %err,
                    dir = %self.dir.display(),
                    "disk prefix-cache directory walk failed during eviction; skipping",
                );
                return 0;
            }
        };

        entries.sort_by_key(|entry| entry.mtime);
        let mut total_bytes: u64 = entries.iter().map(|e| e.size).sum();
        let mut total_entries = entries.len();
        let mut evictions: u32 = 0;
        let mut idx = 0;
        while idx < entries.len()
            && (total_bytes > self.policy.max_bytes || total_entries > self.policy.max_entries)
        {
            let entry = &entries[idx];
            match fs::remove_file(&entry.path) {
                Ok(()) => {
                    total_bytes = total_bytes.saturating_sub(entry.size);
                    total_entries = total_entries.saturating_sub(1);
                    evictions = evictions.saturating_add(1);
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    // Another writer may have evicted this entry
                    // concurrently. Treat as already-gone.
                    total_bytes = total_bytes.saturating_sub(entry.size);
                    total_entries = total_entries.saturating_sub(1);
                }
                Err(e) => {
                    tracing::warn!(
                        target: "ax_engine_mlx::prefix_cache",
                        error = %e,
                        path = %entry.path.display(),
                        "disk prefix-cache failed to remove entry during eviction; skipping",
                    );
                }
            }
            idx += 1;
        }
        evictions
    }

    fn list_entries(&self) -> Result<Vec<EntryStat>, DiskPrefixCacheError> {
        let mut out = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            let is_axkv = path.extension().is_some_and(|ext| ext == ENTRY_EXTENSION);
            if !is_axkv {
                continue;
            }
            let metadata = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };
            let size = metadata.len();
            let mtime = metadata.modified().unwrap_or(std::time::UNIX_EPOCH);
            out.push(EntryStat { path, size, mtime });
        }
        Ok(out)
    }
}

struct EntryStat {
    path: PathBuf,
    size: u64,
    mtime: std::time::SystemTime,
}

/// RAII guard that removes a temp file on drop unless explicitly
/// disarmed. Used by [`DiskPrefixCache::insert`] so a failed write or
/// rename does not leave a `.tmp.<pid>` file behind.
struct TempFileGuard {
    path: Option<PathBuf>,
}

impl TempFileGuard {
    fn new(path: &Path) -> Self {
        Self {
            path: Some(path.to_path_buf()),
        }
    }

    fn disarm(&mut self) {
        self.path = None;
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = fs::remove_file(path);
        }
    }
}

fn parse_file(raw: &[u8], expected_key: &[u8]) -> Option<DiskPrefixCacheEntry> {
    if raw.len() < FIXED_HEADER_LEN {
        return None;
    }
    if &raw[0..4] != FILE_MAGIC {
        return None;
    }
    let version = u32::from_le_bytes(raw[4..8].try_into().ok()?);
    if version != FILE_VERSION {
        return None;
    }
    let payload_hash: [u8; 32] = raw[8..40].try_into().ok()?;
    let payload_len = u64::from_le_bytes(raw[40..48].try_into().ok()?) as usize;
    let key_len = u32::from_le_bytes(raw[48..52].try_into().ok()?) as usize;
    let prefill_slot = u32::from_le_bytes(raw[52..56].try_into().ok()?);
    let key_start = FIXED_HEADER_LEN;
    let key_end = key_start.checked_add(key_len)?;
    let payload_start = key_end;
    let payload_end = payload_start.checked_add(payload_len)?;
    if payload_end > raw.len() {
        return None;
    }
    if &raw[key_start..key_end] != expected_key {
        return None;
    }
    let payload = &raw[payload_start..payload_end];
    if payload_sha256(payload) != payload_hash {
        return None;
    }
    let prefill_output_token = if prefill_slot == PREFILL_TOKEN_NONE {
        None
    } else {
        Some(prefill_slot)
    };
    Some(DiskPrefixCacheEntry {
        payload: payload.to_vec(),
        prefill_output_token,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn unique_tempdir(label: &str) -> PathBuf {
        let mut dir = env::temp_dir();
        dir.push(format!(
            "ax-engine-disk-cache-test-{}-{}-{}",
            label,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        dir
    }

    /// Build a no-prefill-token entry from a raw payload. Tests that
    /// only care about the kv_cache payload don't need to think about
    /// the prefill-token slot.
    fn payload_only(bytes: &[u8]) -> DiskPrefixCacheEntry {
        DiskPrefixCacheEntry {
            payload: bytes.to_vec(),
            prefill_output_token: None,
        }
    }

    #[test]
    fn insert_then_get_roundtrip() {
        let dir = unique_tempdir("roundtrip");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key_bytes =
            canonical_key_bytes("model-a", "policy-a", "layout-a", 16, 1024, 0xdead_beef);
        let payload = b"PAYLOAD-FOR-CACHE".to_vec();
        cache
            .insert(&key_bytes, &payload_only(&payload))
            .expect("insert");
        let got = cache.get(&key_bytes).expect("get").expect("hit");
        assert_eq!(got.payload, payload);
        assert_eq!(
            got.prefill_output_token, None,
            "no prefill token written → reads back as None",
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn roundtrip_preserves_prefill_output_token() {
        // Regression for the M4-discovered cross-restart correctness
        // bug: when the producing prefill captured a greedy prefill
        // output token, the on-disk format must carry it so the L2
        // restore path can avoid recomputing at decode step 0.
        let dir = unique_tempdir("prefill-tok-roundtrip");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key_bytes = canonical_key_bytes("m", "p", "l", 16, 1024, 0xfeed_d00d);
        let entry = DiskPrefixCacheEntry {
            payload: b"payload".to_vec(),
            prefill_output_token: Some(987_654),
        };
        cache.insert(&key_bytes, &entry).expect("insert");
        let got = cache.get(&key_bytes).expect("get").expect("hit");
        assert_eq!(got.payload, entry.payload);
        assert_eq!(got.prefill_output_token, Some(987_654));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn get_miss_returns_none() {
        let dir = unique_tempdir("miss");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key_bytes =
            canonical_key_bytes("model-b", "policy-b", "layout-b", 16, 1024, 0xfeed_face);
        assert!(cache.get(&key_bytes).expect("get").is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn key_mismatch_returns_miss() {
        let dir = unique_tempdir("collision");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key_a = canonical_key_bytes("model-c", "policy-c", "layout-c", 16, 1024, 1);
        let key_b = canonical_key_bytes("model-c", "policy-c", "layout-c", 16, 1024, 2);
        cache
            .insert(&key_a, &payload_only(b"payload-a"))
            .expect("insert");
        // Different key, but same filename slot would only happen on a
        // SHA256 collision (vanishingly improbable). Simulate by writing
        // key_a's content under key_b's filename: we manually swap so the
        // parser must reject the on-disk content as a hash collision.
        let path_a = cache.path_for(&key_a);
        let path_b = cache.path_for(&key_b);
        fs::rename(&path_a, &path_b).expect("rename");
        let result = cache.get(&key_b).expect("get");
        assert!(
            result.is_none(),
            "key mismatch should be reported as a miss"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_payload_returns_miss() {
        let dir = unique_tempdir("corrupt");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key_bytes = canonical_key_bytes("model-d", "policy-d", "layout-d", 16, 1024, 9);
        cache
            .insert(&key_bytes, &payload_only(b"payload-correct"))
            .expect("insert");
        // Flip a byte in the payload region (last byte of file).
        let path = cache.path_for(&key_bytes);
        let mut raw = fs::read(&path).expect("read");
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        fs::write(&path, raw).expect("write corrupted");
        let result = cache.get(&key_bytes).expect("get");
        assert!(result.is_none(), "corrupted payload should miss");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn contains_returns_true_after_insert() {
        let dir = unique_tempdir("contains-hit");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key_bytes =
            canonical_key_bytes("model-c1", "policy-c1", "layout-c1", 16, 1024, 0xc04e_7415);
        assert!(!cache.contains(&key_bytes), "before insert: must not exist");
        cache
            .insert(&key_bytes, &payload_only(b"payload-x"))
            .expect("insert");
        assert!(cache.contains(&key_bytes), "after insert: must exist");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn contains_does_not_validate_payload_integrity() {
        // The probe path relies on `contains` being O(1)-cheap — it does
        // NOT read the file or check the SHA256. A subsequent `get` is
        // what surfaces a corrupt file as a cache miss. This test locks
        // that contract: after deliberately corrupting a file, contains
        // still returns true, while get returns None.
        let dir = unique_tempdir("contains-corrupt");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key_bytes =
            canonical_key_bytes("model-c2", "policy-c2", "layout-c2", 16, 1024, 0xc0c0_dead);
        cache
            .insert(&key_bytes, &payload_only(b"payload-valid"))
            .expect("insert");
        // Flip the last byte (payload region) to corrupt the SHA256.
        let path = cache.path_for(&key_bytes);
        let mut raw = fs::read(&path).expect("read");
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        fs::write(&path, raw).expect("write corrupted");

        assert!(
            cache.contains(&key_bytes),
            "contains is existence-only; must remain true post-corruption"
        );
        assert!(
            cache.get(&key_bytes).expect("get").is_none(),
            "get must surface corruption as a miss"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn eviction_drops_oldest_when_entry_budget_exceeded() {
        let dir = unique_tempdir("evict-entries");
        // Allow only 2 entries — third insert must evict the oldest.
        let policy = DiskPrefixCachePolicy {
            max_bytes: u64::MAX,
            max_entries: 2,
        };
        let cache = DiskPrefixCache::with_policy(&dir, policy).expect("open");

        let key_a = canonical_key_bytes("m", "p", "l", 16, 1024, 0xa1);
        let key_b = canonical_key_bytes("m", "p", "l", 16, 1024, 0xb2);
        let key_c = canonical_key_bytes("m", "p", "l", 16, 1024, 0xc3);

        // Insert in order a, b, c with sleeps so mtimes are strictly
        // increasing (1-second filesystem resolution is the worst-case
        // we need to defeat).
        cache
            .insert(&key_a, &payload_only(b"payload-a"))
            .expect("insert a");
        std::thread::sleep(std::time::Duration::from_millis(1100));
        cache
            .insert(&key_b, &payload_only(b"payload-b"))
            .expect("insert b");
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let outcome = cache
            .insert(&key_c, &payload_only(b"payload-c"))
            .expect("insert c");

        assert_eq!(
            outcome.evictions, 1,
            "third insert must evict exactly one entry"
        );
        assert!(!cache.contains(&key_a), "oldest entry (a) must be evicted");
        assert!(cache.contains(&key_b), "b must survive");
        assert!(cache.contains(&key_c), "c must survive");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn eviction_drops_oldest_when_byte_budget_exceeded() {
        let dir = unique_tempdir("evict-bytes");
        // Make the budget tighter than two payloads but bigger than
        // one. The actual per-file size is the payload + header
        // (~56 bytes), so we pick a budget that fits exactly one
        // entry comfortably and not two.
        let payload_size: usize = 4096;
        let per_file = (FIXED_HEADER_LEN + 32 + payload_size) as u64;
        let policy = DiskPrefixCachePolicy {
            max_bytes: per_file + (per_file / 4), // ~1.25 × single file
            max_entries: usize::MAX,
        };
        let cache = DiskPrefixCache::with_policy(&dir, policy).expect("open");

        let key_a = canonical_key_bytes("m", "p", "l", 16, 1024, 0xa1);
        let key_b = canonical_key_bytes("m", "p", "l", 16, 1024, 0xb2);
        let payload = vec![0u8; payload_size];

        cache
            .insert(&key_a, &payload_only(&payload))
            .expect("insert a");
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let outcome = cache
            .insert(&key_b, &payload_only(&payload))
            .expect("insert b");

        assert_eq!(
            outcome.evictions, 1,
            "byte-budget overflow must evict exactly one entry"
        );
        assert!(!cache.contains(&key_a), "oldest entry must be evicted");
        assert!(cache.contains(&key_b), "newest entry must survive");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn eviction_skips_non_axkv_files() {
        // Operator-placed junk in the directory must not crash the
        // walk and must not be counted toward the eviction budget.
        let dir = unique_tempdir("evict-junk");
        let policy = DiskPrefixCachePolicy {
            max_bytes: u64::MAX,
            max_entries: 1,
        };
        let cache = DiskPrefixCache::with_policy(&dir, policy).expect("open");
        fs::write(dir.join("NOTES.md"), b"hello").expect("write junk");

        let key_a = canonical_key_bytes("m", "p", "l", 16, 1024, 0xaa);
        cache
            .insert(&key_a, &payload_only(b"payload-a"))
            .expect("insert a");

        // Junk file must remain untouched.
        assert!(dir.join("NOTES.md").is_file());
        assert!(cache.contains(&key_a));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn with_policy_evicts_existing_entries_to_budget_on_open() {
        // Pre-populate three entries under the permissive default
        // policy, then reopen with a tight `max_entries=1` budget.
        // The new instance must trim back to one entry without
        // requiring a fresh insert. Mtimes must be staggered so the
        // initial-sweep ordering is deterministic.
        let dir = unique_tempdir("reopen-evict");
        {
            let cache = DiskPrefixCache::open(&dir).expect("open default");
            let key_a = canonical_key_bytes("m", "p", "l", 16, 1024, 0xa1);
            let key_b = canonical_key_bytes("m", "p", "l", 16, 1024, 0xb2);
            let key_c = canonical_key_bytes("m", "p", "l", 16, 1024, 0xc3);
            cache
                .insert(&key_a, &payload_only(b"payload-a"))
                .expect("insert a");
            std::thread::sleep(std::time::Duration::from_millis(1100));
            cache
                .insert(&key_b, &payload_only(b"payload-b"))
                .expect("insert b");
            std::thread::sleep(std::time::Duration::from_millis(1100));
            cache
                .insert(&key_c, &payload_only(b"payload-c"))
                .expect("insert c");
        }

        let tight = DiskPrefixCachePolicy {
            max_bytes: u64::MAX,
            max_entries: 1,
        };
        let cache = DiskPrefixCache::with_policy(&dir, tight).expect("reopen");

        let remaining: Vec<_> = fs::read_dir(&dir)
            .expect("read_dir")
            .filter_map(Result::ok)
            .filter(|e| {
                e.path()
                    .extension()
                    .is_some_and(|ext| ext == ENTRY_EXTENSION)
            })
            .collect();
        assert_eq!(
            remaining.len(),
            1,
            "with_policy must trim existing entries down to the policy on open",
        );

        // The newest of the three entries (c) must survive.
        let key_c = canonical_key_bytes("m", "p", "l", 16, 1024, 0xc3);
        assert!(cache.contains(&key_c), "newest entry must survive");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn temp_file_guard_removes_tmp_on_drop() {
        // The guard's job is to clean up a partially written
        // `.tmp.<pid>` file if insert fails between create and
        // rename. We simulate that by manually creating a tmp-shaped
        // file, dropping the guard without disarming, and checking
        // it was reaped. This proves the guard's drop path runs
        // independently of whether insert itself errored.
        let dir = unique_tempdir("tmp-guard");
        fs::create_dir_all(&dir).expect("mkdir");
        let tmp = dir.join("orphan.tmp.999");
        fs::write(&tmp, b"fake-partial-write").expect("seed tmp");
        assert!(tmp.is_file(), "seed");

        {
            let _guard = TempFileGuard::new(&tmp);
            // Drop without disarming.
        }
        assert!(!tmp.exists(), "temp guard must remove the tmp file on drop");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn temp_file_guard_disarmed_does_not_remove() {
        let dir = unique_tempdir("tmp-guard-disarm");
        fs::create_dir_all(&dir).expect("mkdir");
        let tmp = dir.join("survivor.tmp.999");
        fs::write(&tmp, b"keep-me").expect("seed tmp");

        {
            let mut guard = TempFileGuard::new(&tmp);
            guard.disarm();
        }
        assert!(tmp.is_file(), "disarmed guard must not remove the tmp file",);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn atomic_rename_temp_files_cleaned() {
        // The .tmp.* file should not survive a successful insert.
        let dir = unique_tempdir("atomic");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key_bytes = canonical_key_bytes("model-e", "policy-e", "layout-e", 16, 1024, 42);
        cache
            .insert(&key_bytes, &payload_only(b"payload-e"))
            .expect("insert");
        let entries: Vec<_> = fs::read_dir(&dir)
            .expect("read_dir")
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(entries.is_empty(), "no .tmp.* file should remain");
        let _ = fs::remove_dir_all(&dir);
    }
}
