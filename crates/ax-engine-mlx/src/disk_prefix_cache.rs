//! F3 M1 — minimal file-I/O wrapper over [`MlxKVCache::serialize_to_bytes`].
//!
//! This module owns the on-disk framing for the future durable
//! prefix-cache layer scoped by
//! `.internal/planning/MLX-DISK-PREFIX-CACHE-PRD-2026-05-14.md`. M1 ships
//! only the file-level get / insert primitives plus the wire framing
//! around the kv_cache payload; eviction (M3), the runner-side L2
//! lookup wire-up (M2), and the cross-restart integration validation
//! (M4) are explicitly deferred to follow-up sessions per the PRD's
//! milestone breakdown.
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
//! What is explicitly out of scope for M1:
//!   - Eviction (M3). The on-disk directory grows unbounded under M1
//!     usage; this is acceptable because M1 has no runner wire-up, so
//!     only operator-driven test runs touch the directory.
//!   - Multi-process concurrency / flock (M3). Atomic rename gives
//!     readers a consistent file view; concurrent writers may
//!     overwrite each other's files (harmless when content matches
//!     the same key, and key collisions across writers are
//!     vanishingly unlikely under SHA256).
//!   - Telemetry counters (M2 will add `disk_hits` / `disk_misses`
//!     into `MlxPrefixCacheTelemetry`).
//!   - Network or mmap I/O (PRD §5.3 explicitly defers mmap to a
//!     follow-up).

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Outer file magic. Distinct from the kv_cache payload's `AXKB` magic
/// so an unframed payload cannot be mistaken for a complete file.
const FILE_MAGIC: &[u8; 4] = b"AXKV";
const FILE_VERSION: u32 = 1;
/// Outer header byte count: magic(4) + version(4) + payload_sha256(32)
/// + payload_length(8) + key_len(4) + reserved(4) = 56 bytes.
const FIXED_HEADER_LEN: usize = 56;

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
pub struct DiskPrefixCache {
    dir: PathBuf,
}

impl DiskPrefixCache {
    /// Open a disk cache rooted at `dir`. The directory is created if
    /// it does not exist; failures to create or stat are returned as
    /// `BadDirectory`.
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self, DiskPrefixCacheError> {
        let dir = dir.into();
        if let Err(e) = fs::create_dir_all(&dir) {
            return Err(match e.kind() {
                std::io::ErrorKind::PermissionDenied => DiskPrefixCacheError::BadDirectory(dir),
                _ => e.into(),
            });
        }
        Ok(Self { dir })
    }

    /// Cache directory for inspection by callers / tests.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Filename for a given canonical key.
    pub fn path_for(&self, key_bytes: &[u8]) -> PathBuf {
        self.dir.join(format!("{}.axkv", key_sha256_hex(key_bytes)))
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

    /// Look up the payload for `key_bytes`. Returns:
    /// - `Ok(Some(payload))` on a clean hit;
    /// - `Ok(None)` on miss, hash collision (key bytes differ), or
    ///   on-disk corruption (the caller treats this like a miss);
    /// - `Err(_)` only for unrecoverable IO failures (e.g. permission
    ///   denied on a file we know exists).
    pub fn get(&self, key_bytes: &[u8]) -> Result<Option<Vec<u8>>, DiskPrefixCacheError> {
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
    pub fn insert(&self, key_bytes: &[u8], payload: &[u8]) -> Result<(), DiskPrefixCacheError> {
        let final_path = self.path_for(key_bytes);
        let tmp_path = self.dir.join(format!(
            "{}.tmp.{}",
            key_sha256_hex(key_bytes),
            std::process::id()
        ));

        let payload_hash = payload_sha256(payload);
        let payload_len = payload.len() as u64;
        let key_len = u32::try_from(key_bytes.len()).expect("key too long");

        let mut buf = Vec::with_capacity(FIXED_HEADER_LEN + key_bytes.len() + payload.len());
        buf.extend_from_slice(FILE_MAGIC);
        buf.extend_from_slice(&FILE_VERSION.to_le_bytes());
        buf.extend_from_slice(&payload_hash);
        buf.extend_from_slice(&payload_len.to_le_bytes());
        buf.extend_from_slice(&key_len.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
        buf.extend_from_slice(key_bytes);
        buf.extend_from_slice(payload);

        {
            let mut f = fs::File::create(&tmp_path)?;
            f.write_all(&buf)?;
            f.sync_all()?;
        }
        fs::rename(&tmp_path, &final_path)?;
        Ok(())
    }
}

fn parse_file(raw: &[u8], expected_key: &[u8]) -> Option<Vec<u8>> {
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
    // raw[52..56] reserved
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
    Some(payload.to_vec())
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

    #[test]
    fn insert_then_get_roundtrip() {
        let dir = unique_tempdir("roundtrip");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key_bytes =
            canonical_key_bytes("model-a", "policy-a", "layout-a", 16, 1024, 0xdead_beef);
        let payload = b"PAYLOAD-FOR-CACHE".to_vec();
        cache.insert(&key_bytes, &payload).expect("insert");
        let got = cache.get(&key_bytes).expect("get").expect("hit");
        assert_eq!(got, payload);
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
        cache.insert(&key_a, b"payload-a").expect("insert");
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
            .insert(&key_bytes, b"payload-correct")
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
        cache.insert(&key_bytes, b"payload-x").expect("insert");
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
        cache.insert(&key_bytes, b"payload-valid").expect("insert");
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
    fn atomic_rename_temp_files_cleaned() {
        // The .tmp.* file should not survive a successful insert.
        let dir = unique_tempdir("atomic");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key_bytes = canonical_key_bytes("model-e", "policy-e", "layout-e", 16, 1024, 42);
        cache.insert(&key_bytes, b"payload-e").expect("insert");
        let entries: Vec<_> = fs::read_dir(&dir)
            .expect("read_dir")
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(entries.is_empty(), "no .tmp.* file should remain");
        let _ = fs::remove_dir_all(&dir);
    }
}
