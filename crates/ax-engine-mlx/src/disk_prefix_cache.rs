//! F3 — disk-backed prefix cache over `MlxKVCache::serialize_to_bytes`.
//!
//! This module owns the on-disk framing for the future durable
//! prefix-cache layer scoped by
//! `.internal/prd/MLX-DISK-PREFIX-CACHE-PRD-2026-05-14.md`. The
//! current implementation owns file-level get / insert primitives,
//! wire framing around the kv_cache payload, runner-side L2 lookup,
//! and best-effort post-insert eviction. Mutating operations take a
//! directory-level advisory lock so concurrent processes serialize
//! `insert -> atomic rename -> eviction` work while readers remain
//! lock-free.
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
//!   - Cross-process advisory locking around mutating operations via
//!     a sentinel lock file in the cache directory.
//!
//! What is explicitly out of scope for this layer:
//!   - Network or mmap I/O (PRD §5.3 explicitly defers mmap to a
//!     follow-up).

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use fs2::FileExt;
use sha2::{Digest, Sha256};

/// Outer file magic. Distinct from the kv_cache payload's `AXKB` magic
/// so an unframed payload cannot be mistaken for a complete file.
const FILE_MAGIC: &[u8; 4] = b"AXKV";
/// File-format version (spec §7.1). v3 adds a header-length field, a
/// flags word, an entry checksum covering the canonical key + semantic
/// header fields + payload (v2 covered the payload only), and producer
/// cost metadata (cold-prefill and serialization microseconds) used as
/// admission / load-versus-compute hints. Files written under versions
/// 1–2 are rejected as misses and lazily unlinked.
const FILE_VERSION: u32 = 3;
/// Fixed v3 header byte count: magic(4) + version(4) + header_len(4) +
/// flags(4) + key_len(4) + payload_len(8) + entry_sha256(32) +
/// prefill_token_slot(4) + producer_cold_prefill_us(8) +
/// producer_serialize_us(8) = 80 bytes. The prefill slot is u32::MAX when
/// there is no prefill token to carry (e.g. partial-prefix snapshots).
const FIXED_HEADER_LEN: usize = 80;
/// Domain prefix for the v3 entry checksum.
const ENTRY_CHECKSUM_DOMAIN: &[u8] = b"ax.mlx.axkv.entry.v3\0";
/// Sentinel for "no prefill output token captured" in the header slot.
const PREFILL_TOKEN_NONE: u32 = u32::MAX;
/// File extension for stored entries.
const ENTRY_EXTENSION: &str = "axkv";
/// Sentinel file used for the directory-level cross-process lock.
/// The file itself is not a cache entry and is ignored by eviction.
const LOCK_FILE_NAME: &str = ".axkv.lock";
/// Longest a mutating operation waits for the directory lock before
/// giving up. Disk is strictly additive, so a timed-out operation is
/// reported as an IO error the caller degrades on (skip the store /
/// sweep) instead of stalling the request path behind a wedged holder.
const LOCK_ACQUIRE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);
/// Poll interval while waiting for the directory lock.
const LOCK_RETRY_INTERVAL: std::time::Duration = std::time::Duration::from_millis(25);

/// Default per-process disk-cache size budget when no env override is
/// set. Matches the value documented in
/// `MLX-DISK-PREFIX-CACHE-PRD-2026-05-14.md` §3.5.
const DEFAULT_DISK_CACHE_MAX_BYTES: u64 = 8 * 1024 * 1024 * 1024; // 8 GiB

/// Default per-process disk-cache entry budget when no env override is
/// set. PRD §3.5.
const DEFAULT_DISK_CACHE_MAX_ENTRIES: usize = 1024;
/// Default adaptive-admission floor on prefix length (spec §4;
/// provisional until K1 cost evidence).
const DEFAULT_DISK_MIN_PREFIX_TOKENS: u32 = 2048;
/// Default minimum predicted one-hit saving after lifecycle costs (spec
/// §4; provisional until K1 cost evidence).
const DEFAULT_DISK_MIN_SAVINGS_US: u64 = 10_000;
/// Default streaming read/checksum staging chunk (spec §4).
const DEFAULT_DISK_IO_CHUNK_BYTES: usize = 4 * 1024 * 1024;
/// Default background write queue depth (spec §4).
const DEFAULT_DISK_WRITE_QUEUE_DEPTH: usize = 2;
/// Default best-effort writer drain budget at shutdown (spec §4).
const DEFAULT_DISK_SHUTDOWN_DRAIN_MS: u64 = 5000;

/// L2 store admission mode (spec §5). `Adaptive` admits only entries
/// whose predicted lifecycle value is positive; `Always` is a diagnostic
/// mode that still obeys size/layout/correctness limits.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DiskAdmissionMode {
    Disabled,
    #[default]
    Adaptive,
    Always,
}

/// Closed admission decision reasons with stable telemetry codes.
/// Do not reorder: codes are part of the route-decision contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiskAdmissionReason {
    AdmittedAlways,
    AdmittedPositiveValue,
    Disabled,
    UnsupportedLayout,
    PrefixTooShort,
    EntryTooLarge,
    NoCostModel,
    PredictedNoSavings,
    CapacityPressure,
    ArtifactIdentityUnavailable,
}

impl DiskAdmissionReason {
    /// Stable numeric telemetry code (never a Rust discriminant).
    pub fn code(self) -> u32 {
        match self {
            Self::AdmittedAlways => 1,
            Self::AdmittedPositiveValue => 2,
            Self::Disabled => 3,
            Self::UnsupportedLayout => 4,
            Self::PrefixTooShort => 5,
            Self::EntryTooLarge => 6,
            Self::NoCostModel => 7,
            Self::PredictedNoSavings => 8,
            Self::CapacityPressure => 9,
            Self::ArtifactIdentityUnavailable => 10,
        }
    }

    pub fn admitted(self) -> bool {
        matches!(self, Self::AdmittedAlways | Self::AdmittedPositiveValue)
    }
}

/// Inputs and derived quantities of one admission decision (spec §5).
#[derive(Clone, Copy, Debug, Default)]
pub struct DiskCacheCostEstimate {
    pub prefix_tokens: u32,
    pub estimated_entry_bytes: u64,
    pub cold_prefill_us: u64,
    pub restore_us: u64,
    pub write_us: u64,
    pub expected_hits: u32,
}

/// Eviction policy for the L2 disk prefix cache. Byte/entry budgets are
/// enforced after every successful insert; whichever fires first drives
/// eviction. Policies are immutable for the cache's lifetime.
#[derive(Clone, Copy, Debug)]
pub struct DiskPrefixCachePolicy {
    /// Maximum aggregate bytes across all entries before eviction
    /// trims the oldest files.
    pub max_bytes: u64,
    /// Maximum entry count before eviction trims the oldest files.
    pub max_entries: usize,
    /// Store admission mode.
    pub admission: DiskAdmissionMode,
    /// Adaptive-admission hard floor on prefix length in tokens.
    pub min_prefix_tokens: u32,
    /// Reject a single entry larger than this before writing.
    pub max_entry_bytes: u64,
    /// Minimum predicted one-hit saving after lifecycle costs (µs).
    pub min_savings_us: u64,
    /// Streaming read/checksum staging chunk size.
    pub io_chunk_bytes: usize,
    /// Background write queue depth (snapshot-sized jobs).
    pub write_queue_depth: usize,
    /// Best-effort writer drain budget at shutdown.
    pub shutdown_drain_ms: u64,
}

impl Default for DiskPrefixCachePolicy {
    fn default() -> Self {
        Self {
            max_bytes: DEFAULT_DISK_CACHE_MAX_BYTES,
            max_entries: DEFAULT_DISK_CACHE_MAX_ENTRIES,
            admission: DiskAdmissionMode::Adaptive,
            min_prefix_tokens: DEFAULT_DISK_MIN_PREFIX_TOKENS,
            max_entry_bytes: default_max_entry_bytes(DEFAULT_DISK_CACHE_MAX_BYTES),
            min_savings_us: DEFAULT_DISK_MIN_SAVINGS_US,
            io_chunk_bytes: DEFAULT_DISK_IO_CHUNK_BYTES,
            write_queue_depth: DEFAULT_DISK_WRITE_QUEUE_DEPTH,
            shutdown_drain_ms: DEFAULT_DISK_SHUTDOWN_DRAIN_MS,
        }
    }
}

/// `min(512 MiB, max_bytes / 4)` per spec §4.
fn default_max_entry_bytes(max_bytes: u64) -> u64 {
    (512 * 1024 * 1024).min(max_bytes / 4)
}

impl DiskPrefixCachePolicy {
    /// Build a policy from env, falling back to the documented
    /// defaults when an entry is unset, blank, or unparseable. Reads
    /// `AX_MLX_PREFIX_CACHE_DISK_MAX_BYTES` and
    /// `AX_MLX_PREFIX_CACHE_DISK_MAX_ENTRIES`.
    ///
    /// An explicit `0` is preserved (not treated as unset): it means
    /// "disabled", matching the in-memory tier's env semantics, and is
    /// honored by [`Self::enabled`] at the open site.
    pub fn from_env() -> Self {
        fn parsed<T: std::str::FromStr>(name: &str) -> Option<T> {
            let raw = std::env::var(name).ok()?;
            match raw.trim().parse::<T>() {
                Ok(value) => Some(value),
                Err(_) => {
                    tracing::warn!(
                        target: "ax_engine_mlx::prefix_cache",
                        name,
                        "invalid disk prefix-cache env value; using default",
                    );
                    None
                }
            }
        }

        let max_bytes = parsed::<u64>("AX_MLX_PREFIX_CACHE_DISK_MAX_BYTES")
            .unwrap_or(DEFAULT_DISK_CACHE_MAX_BYTES);
        let admission = match std::env::var("AX_MLX_PREFIX_CACHE_DISK_ADMISSION")
            .ok()
            .as_deref()
            .map(str::trim)
        {
            None | Some("") => DiskAdmissionMode::Adaptive,
            Some("adaptive") => DiskAdmissionMode::Adaptive,
            Some("always") => DiskAdmissionMode::Always,
            Some("disabled") => DiskAdmissionMode::Disabled,
            Some(_) => {
                tracing::warn!(
                    target: "ax_engine_mlx::prefix_cache",
                    "invalid AX_MLX_PREFIX_CACHE_DISK_ADMISSION; using adaptive",
                );
                DiskAdmissionMode::Adaptive
            }
        };
        Self {
            max_bytes,
            max_entries: parsed::<usize>("AX_MLX_PREFIX_CACHE_DISK_MAX_ENTRIES")
                .unwrap_or(DEFAULT_DISK_CACHE_MAX_ENTRIES),
            admission,
            min_prefix_tokens: parsed::<u32>("AX_MLX_PREFIX_CACHE_DISK_MIN_PREFIX_TOKENS")
                .unwrap_or(DEFAULT_DISK_MIN_PREFIX_TOKENS),
            max_entry_bytes: parsed::<u64>("AX_MLX_PREFIX_CACHE_DISK_MAX_ENTRY_BYTES")
                .unwrap_or_else(|| default_max_entry_bytes(max_bytes)),
            min_savings_us: parsed::<u64>("AX_MLX_PREFIX_CACHE_DISK_MIN_SAVINGS_US")
                .unwrap_or(DEFAULT_DISK_MIN_SAVINGS_US),
            io_chunk_bytes: parsed::<usize>("AX_MLX_PREFIX_CACHE_DISK_IO_CHUNK_BYTES")
                .unwrap_or(DEFAULT_DISK_IO_CHUNK_BYTES)
                .clamp(64 * 1024, 16 * 1024 * 1024),
            write_queue_depth: parsed::<usize>("AX_MLX_PREFIX_CACHE_DISK_WRITE_QUEUE_DEPTH")
                .unwrap_or(DEFAULT_DISK_WRITE_QUEUE_DEPTH)
                .clamp(1, 8),
            shutdown_drain_ms: parsed::<u64>("AX_MLX_PREFIX_CACHE_DISK_SHUTDOWN_DRAIN_MS")
                .unwrap_or(DEFAULT_DISK_SHUTDOWN_DRAIN_MS),
        }
    }

    /// Whether this policy admits any entry at all. A zero byte or
    /// entry budget disables the disk tier, mirroring the in-memory
    /// `MlxPrefixCachePolicy::enabled` semantics.
    pub fn enabled(&self) -> bool {
        self.max_bytes > 0 && self.max_entries > 0
    }

    /// Evaluate store admission for one candidate entry (spec §5.1).
    ///
    /// `adaptive` admits only when the predicted one-hit saving exceeds
    /// the lifecycle cost by `min_savings_us`:
    ///
    /// ```text
    /// gross_savings_us = expected_hits * sat_sub(cold_prefill_us, restore_us)
    /// admit when gross_savings_us >= write_us + min_savings_us
    /// ```
    ///
    /// `incremental_serialize_us` is zero in the snapshot era because L1
    /// already required the exact shared payload, so it does not appear in
    /// the lifecycle sum. Hard size/length floors apply before the value
    /// model; `always` bypasses only the value model, never the size caps.
    pub fn evaluate_admission(
        &self,
        prefix_tokens: u32,
        entry_bytes: u64,
        cold_prefill_us: Option<u64>,
        throughput: Option<DiskThroughputSnapshot>,
    ) -> (DiskAdmissionReason, DiskCacheCostEstimate) {
        let mut estimate = DiskCacheCostEstimate {
            prefix_tokens,
            estimated_entry_bytes: entry_bytes,
            expected_hits: 1,
            ..DiskCacheCostEstimate::default()
        };
        if matches!(self.admission, DiskAdmissionMode::Disabled) {
            return (DiskAdmissionReason::Disabled, estimate);
        }
        if entry_bytes > self.max_entry_bytes || entry_bytes > self.max_bytes {
            return (DiskAdmissionReason::EntryTooLarge, estimate);
        }
        if matches!(self.admission, DiskAdmissionMode::Always) {
            return (DiskAdmissionReason::AdmittedAlways, estimate);
        }
        if prefix_tokens < self.min_prefix_tokens {
            return (DiskAdmissionReason::PrefixTooShort, estimate);
        }
        let (Some(cold_prefill_us), Some(throughput)) =
            (cold_prefill_us.filter(|us| *us > 0), throughput)
        else {
            return (DiskAdmissionReason::NoCostModel, estimate);
        };
        estimate.cold_prefill_us = cold_prefill_us;
        estimate.restore_us = throughput.restore_us(entry_bytes);
        estimate.write_us = throughput.write_us(entry_bytes);
        let gross_savings_us = u64::from(estimate.expected_hits)
            .saturating_mul(cold_prefill_us.saturating_sub(estimate.restore_us));
        if gross_savings_us >= estimate.write_us.saturating_add(self.min_savings_us) {
            (DiskAdmissionReason::AdmittedPositiveValue, estimate)
        } else {
            (DiskAdmissionReason::PredictedNoSavings, estimate)
        }
    }
}

/// Process-local storage throughput estimate used by adaptive admission.
/// Seeded by an open-time calibration probe and refined by an EWMA over
/// observed stores/restores. A performance hint only — never part of
/// correctness.
#[derive(Clone, Copy, Debug)]
pub struct DiskThroughputSnapshot {
    /// Sequential write + fsync throughput (bytes per microsecond).
    pub write_bytes_per_us: f64,
    /// Sequential read + checksum + deserialize throughput
    /// (bytes per microsecond).
    pub restore_bytes_per_us: f64,
}

impl DiskThroughputSnapshot {
    pub fn restore_us(&self, bytes: u64) -> u64 {
        estimate_us(bytes, self.restore_bytes_per_us)
    }

    pub fn write_us(&self, bytes: u64) -> u64 {
        estimate_us(bytes, self.write_bytes_per_us)
    }
}

fn estimate_us(bytes: u64, bytes_per_us: f64) -> u64 {
    if bytes_per_us <= f64::EPSILON {
        return u64::MAX;
    }
    let us = bytes as f64 / bytes_per_us;
    if us >= u64::MAX as f64 { u64::MAX } else { us as u64 }
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

/// Inputs to the canonical disk-cache key (schema v3, spec §6).
///
/// Compared with v2, the durable identity now commits to a
/// **content-derived artifact fingerprint** (an in-place checkpoint
/// replacement changes the fingerprint, so stale entries miss instead of
/// restoring wrong-model KV) and the **KV payload wire version** (a
/// payload format bump cleanly invalidates every older entry). The
/// process-local 64-bit FNV `token_hash` is no longer part of the durable
/// identity; token content is committed by SHA-256.
pub struct DiskPrefixKeyFields<'a> {
    pub model_id: &'a str,
    pub artifact_fingerprint_sha256: &'a str,
    pub route_policy: &'a str,
    pub layer_layout: &'a str,
    pub kv_payload_version: u32,
    pub block_size_tokens: u32,
    pub token_count: u32,
    pub tokens: &'a [u32],
}

/// Versioned domain prefix for canonical key bytes.
const CANONICAL_KEY_DOMAIN: &str = "ax.mlx.disk_prefix_key.v3";

/// Canonical byte representation of the disk-cache key (schema v3).
///
/// All strings are length-prefixed UTF-8 so the canonical bytes have no
/// ambiguity around boundary tokens; the token content is committed via
/// SHA-256 over the little-endian token bytes. The embedded-key comparison
/// on read turns any hash collision into a miss instead of a wrong-KV hit.
pub fn canonical_key_bytes(fields: &DiskPrefixKeyFields<'_>) -> Vec<u8> {
    debug_assert_eq!(
        fields.tokens.len(),
        fields.token_count as usize,
        "canonical key token slice must match token_count"
    );
    let mut out = Vec::with_capacity(
        4 + CANONICAL_KEY_DOMAIN.len()
            + 4
            + fields.model_id.len()
            + 4
            + fields.artifact_fingerprint_sha256.len()
            + 4
            + fields.route_policy.len()
            + 4
            + fields.layer_layout.len()
            + 4
            + 4
            + 4
            + 32,
    );
    push_lp_string(&mut out, CANONICAL_KEY_DOMAIN);
    push_lp_string(&mut out, fields.model_id);
    push_lp_string(&mut out, fields.artifact_fingerprint_sha256);
    push_lp_string(&mut out, fields.route_policy);
    push_lp_string(&mut out, fields.layer_layout);
    out.extend_from_slice(&fields.kv_payload_version.to_le_bytes());
    out.extend_from_slice(&fields.block_size_tokens.to_le_bytes());
    out.extend_from_slice(&fields.token_count.to_le_bytes());
    out.extend_from_slice(&token_content_sha256(fields.tokens));
    out
}

fn token_content_sha256(tokens: &[u32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn push_lp_string(out: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    // u32 length prefix: key strings embed operator-controlled paths
    // (`layer_layout` folds in the model artifacts root), which a u16
    // prefix would panic on for pathological inputs.
    let len = u32::try_from(bytes.len()).expect("key string too long");
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(bytes);
}

fn key_sha256_hex(key_bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(key_bytes);
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{:02x}", b)).collect()
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
/// Concurrency note: mutating operations take an exclusive advisory
/// lock on a directory-level sentinel file. Readers stay lock-free and
/// rely on atomic rename + payload checksum validation for a
/// consistent view.
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
    /// Producing request's measured cold-prefill wall time. Admission /
    /// load-versus-compute hint only; zero when unknown.
    pub producer_cold_prefill_us: u64,
    /// Producing request's snapshot serialization wall time. Lifecycle
    /// accounting only; zero when unknown.
    pub producer_serialize_us: u64,
}

/// Stage timings and byte counts from one validated streaming read.
#[derive(Clone, Copy, Debug, Default)]
pub struct DiskReadStageTimings {
    /// Wall time spent in filesystem reads.
    pub read_wall_us: u64,
    /// Wall time spent updating/verifying the entry checksum.
    pub checksum_wall_us: u64,
    /// Total bytes read from the entry file.
    pub bytes_read: u64,
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
        let existed = dir.is_dir();
        if let Err(e) = fs::create_dir_all(&dir) {
            return Err(match e.kind() {
                std::io::ErrorKind::PermissionDenied => DiskPrefixCacheError::BadDirectory(dir),
                _ => e.into(),
            });
        }
        // Cache entries are prompt-derived activations: owner-only by
        // default (spec §7.3). A fresh root is tightened to 0700; an
        // existing broader root gets one warning (a future strict mode may
        // reject it).
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(meta) = fs::metadata(&dir) {
                let mode = meta.permissions().mode() & 0o777;
                if !existed {
                    let _ = fs::set_permissions(&dir, fs::Permissions::from_mode(0o700));
                } else if mode & 0o077 != 0 {
                    tracing::warn!(
                        target: "ax_engine_mlx::prefix_cache",
                        dir = %dir.display(),
                        mode = format!("{mode:o}"),
                        "disk prefix-cache directory is broader than owner-only; \
                         entries contain prompt-derived activations",
                    );
                }
            }
        }
        #[cfg(not(unix))]
        let _ = existed;
        let cache = Self { dir, policy };
        // Reclaim `.tmp.<pid>` files leaked by crashed processes before
        // applying budgets: the `.axkv`-only eviction sweep never matches
        // them, so without this they accumulate without bound. Safe under
        // the exclusive lock — inserts hold the same lock across their
        // create→rename window, so any temp file visible here belongs to a
        // process that died mid-write.
        let _ = cache.sweep_stale_temp_files();
        // Apply the configured budgets to whatever already exists in
        // the directory. Operators who shrink AX_MLX_PREFIX_CACHE_DISK_*
        // budgets between runs should see the trim happen at startup
        // rather than only after the next insert. Eviction is
        // best-effort and never propagates errors here.
        let _ = cache.evict_until_within_policy();
        Ok(cache)
    }

    fn sweep_stale_temp_files(&self) -> Result<u32, DiskPrefixCacheError> {
        let _lock = self.lock_exclusive()?;
        let mut removed = 0u32;
        for entry in fs::read_dir(&self.dir)? {
            let Ok(entry) = entry else { continue };
            let path = entry.path();
            let is_temp = path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.contains(".tmp."));
            if is_temp && fs::remove_file(&path).is_ok() {
                removed += 1;
            }
        }
        if removed > 0 {
            tracing::info!(
                target: "ax_engine_mlx::prefix_cache",
                dir = %self.dir.display(),
                removed,
                "removed stale disk prefix-cache temp files from a previous process",
            );
        }
        Ok(removed)
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
        Ok(self.get_timed(key_bytes)?.map(|(entry, _)| entry))
    }

    /// Validated lookup with per-stage timings for restore telemetry.
    pub fn get_timed(
        &self,
        key_bytes: &[u8],
    ) -> Result<Option<(DiskPrefixCacheEntry, DiskReadStageTimings)>, DiskPrefixCacheError> {
        let path = self.path_for(key_bytes);
        match self.read_entry_streaming(&path, key_bytes) {
            Ok(Some(hit)) => {
                // Best-effort recency bump so mtime-ordered eviction
                // approximates LRU instead of FIFO-by-write-time: without
                // this, a hot entry written early is evicted before cold
                // entries written later.
                let _ = fs::OpenOptions::new()
                    .append(true)
                    .open(&path)
                    .and_then(|file| file.set_modified(std::time::SystemTime::now()));
                Ok(Some(hit))
            }
            Ok(None) => Ok(None),
            Err(ReadEntryError::NotFound) => Ok(None),
            Err(ReadEntryError::Invalid) => {
                self.remove_unparseable_entry(&path, key_bytes);
                Ok(None)
            }
            Err(ReadEntryError::Io(e)) => Err(e.into()),
        }
    }

    /// Stream one entry through a bounded chunk buffer: fixed header first
    /// (validated before any payload allocation), then the embedded key,
    /// then the payload read in `io_chunk_bytes` slices directly into its
    /// final buffer while the entry checksum updates. Never reads the whole
    /// file into a transient buffer and never copies the payload again
    /// (spec §9).
    fn read_entry_streaming(
        &self,
        path: &Path,
        expected_key: &[u8],
    ) -> Result<Option<(DiskPrefixCacheEntry, DiskReadStageTimings)>, ReadEntryError> {
        use std::io::Read;

        // Reject symlinks and non-regular files (spec §7.3).
        let meta = match fs::symlink_metadata(path) {
            Ok(meta) => meta,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(ReadEntryError::NotFound);
            }
            Err(e) => return Err(ReadEntryError::Io(e)),
        };
        if !meta.file_type().is_file() {
            return Err(ReadEntryError::Invalid);
        }

        let mut file = match fs::File::open(path) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(ReadEntryError::NotFound);
            }
            Err(e) => return Err(ReadEntryError::Io(e)),
        };

        let mut timings = DiskReadStageTimings::default();
        let read_started = std::time::Instant::now();
        let mut header = [0u8; FIXED_HEADER_LEN];
        if file.read_exact(&mut header).is_err() {
            return Err(ReadEntryError::Invalid);
        }
        timings.read_wall_us = elapsed_us(read_started);
        timings.bytes_read = FIXED_HEADER_LEN as u64;

        if &header[0..4] != FILE_MAGIC {
            return Err(ReadEntryError::Invalid);
        }
        let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
        if version != FILE_VERSION {
            return Err(ReadEntryError::Invalid);
        }
        let header_len = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
        let flags = u32::from_le_bytes(header[12..16].try_into().unwrap());
        // No flags are defined; a nonzero word is from the future and
        // cannot be interpreted safely. Header extension is only legal
        // behind a defined flag, so header_len must match exactly here.
        if flags != 0 || header_len != FIXED_HEADER_LEN {
            return Err(ReadEntryError::Invalid);
        }
        let key_len = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;
        let payload_len = u64::from_le_bytes(header[20..28].try_into().unwrap());
        let expected_hash: [u8; 32] = header[28..60].try_into().unwrap();
        let prefill_slot = u32::from_le_bytes(header[60..64].try_into().unwrap());
        let producer_cold_prefill_us = u64::from_le_bytes(header[64..72].try_into().unwrap());
        let producer_serialize_us = u64::from_le_bytes(header[72..80].try_into().unwrap());

        // Overflow-safe bounds against policy limits before allocation.
        if key_len != expected_key.len() {
            return Err(ReadEntryError::Invalid);
        }
        let total_len = (FIXED_HEADER_LEN as u64)
            .checked_add(key_len as u64)
            .and_then(|n| n.checked_add(payload_len))
            .ok_or(ReadEntryError::Invalid)?;
        if total_len != meta.len()
            || payload_len > self.policy.max_entry_bytes.max(self.policy.max_bytes)
            || payload_len > usize::MAX as u64
        {
            return Err(ReadEntryError::Invalid);
        }

        let read_started = std::time::Instant::now();
        let mut embedded_key = vec![0u8; key_len];
        if file.read_exact(&mut embedded_key).is_err() {
            return Err(ReadEntryError::Invalid);
        }
        timings.read_wall_us = timings.read_wall_us.saturating_add(elapsed_us(read_started));
        timings.bytes_read = timings.bytes_read.saturating_add(key_len as u64);
        if embedded_key != expected_key {
            return Err(ReadEntryError::Invalid);
        }

        let mut hasher = Self::entry_sha256_hasher(
            prefill_slot,
            producer_cold_prefill_us,
            producer_serialize_us,
            expected_key,
        );
        let mut payload = vec![0u8; payload_len as usize];
        let chunk = self.policy.io_chunk_bytes.max(1);
        let mut filled = 0usize;
        while filled < payload.len() {
            let end = (filled + chunk).min(payload.len());
            let read_started = std::time::Instant::now();
            if file.read_exact(&mut payload[filled..end]).is_err() {
                return Err(ReadEntryError::Invalid);
            }
            timings.read_wall_us = timings.read_wall_us.saturating_add(elapsed_us(read_started));
            let checksum_started = std::time::Instant::now();
            hasher.update(&payload[filled..end]);
            timings.checksum_wall_us = timings
                .checksum_wall_us
                .saturating_add(elapsed_us(checksum_started));
            timings.bytes_read = timings.bytes_read.saturating_add((end - filled) as u64);
            filled = end;
        }
        let checksum_started = std::time::Instant::now();
        let digest = hasher.finalize();
        timings.checksum_wall_us = timings
            .checksum_wall_us
            .saturating_add(elapsed_us(checksum_started));
        if digest.as_slice() != expected_hash {
            return Err(ReadEntryError::Invalid);
        }

        let prefill_output_token = if prefill_slot == PREFILL_TOKEN_NONE {
            None
        } else {
            Some(prefill_slot)
        };
        Ok(Some((
            DiskPrefixCacheEntry {
                payload,
                prefill_output_token,
                producer_cold_prefill_us,
                producer_serialize_us,
            },
            timings,
        )))
    }

    /// Best-effort removal of a file that failed header / checksum / key
    /// validation. Re-validates under the exclusive lock before unlinking
    /// so a concurrent insert that just replaced the file with a healthy
    /// entry is never deleted. Without this cleanup, corrupt or
    /// stale-version files consume budget until mtime eviction reaches
    /// them, and keys that are never re-produced linger forever.
    fn remove_unparseable_entry(&self, path: &Path, key_bytes: &[u8]) {
        let Ok(_lock) = self.lock_exclusive() else {
            return;
        };
        let still_invalid = matches!(
            self.read_entry_streaming(path, key_bytes),
            Err(ReadEntryError::Invalid)
        );
        if still_invalid && fs::remove_file(path).is_ok() {
            tracing::warn!(
                target: "ax_engine_mlx::prefix_cache",
                path = %path.display(),
                "removed unparseable disk prefix-cache entry",
            );
        }
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
        self.insert_parts(
            key_bytes,
            &entry.payload,
            entry.prefill_output_token,
            entry.producer_cold_prefill_us,
            entry.producer_serialize_us,
        )
    }

    /// Borrowing variant of [`Self::insert`]: the payload can come from any
    /// shared buffer (the runner's background writer passes the same
    /// `Arc<[u8]>` the in-memory snapshot holds) without materializing a
    /// `DiskPrefixCacheEntry`-owned copy first.
    pub fn insert_parts(
        &self,
        key_bytes: &[u8],
        payload: &[u8],
        prefill_output_token: Option<u32>,
        producer_cold_prefill_us: u64,
        producer_serialize_us: u64,
    ) -> Result<DiskPrefixCacheInsertOutcome, DiskPrefixCacheError> {
        // A single entry larger than its per-entry bound or the whole byte
        // budget can never be durable: post-insert eviction would remove
        // every older entry and then the entry itself, while telemetry
        // reported a successful insert. Skip the write up front instead.
        let file_len = (FIXED_HEADER_LEN + key_bytes.len() + payload.len()) as u64;
        if file_len > self.policy.max_bytes || file_len > self.policy.max_entry_bytes {
            tracing::warn!(
                target: "ax_engine_mlx::prefix_cache",
                file_len,
                max_bytes = self.policy.max_bytes,
                max_entry_bytes = self.policy.max_entry_bytes,
                "disk prefix-cache entry exceeds the byte budget; skipping store",
            );
            return Ok(DiskPrefixCacheInsertOutcome::default());
        }
        let _lock = self.lock_exclusive()?;
        let final_path = self.path_for(key_bytes);
        let tmp_path = self.dir.join(format!(
            "{}.tmp.{}",
            key_sha256_hex(key_bytes),
            std::process::id()
        ));

        let payload_len = payload.len() as u64;
        let key_len = u32::try_from(key_bytes.len()).expect("key too long");
        let prefill_slot = prefill_output_token.unwrap_or(PREFILL_TOKEN_NONE);
        let entry_hash = Self::entry_checksum_oneshot(
            prefill_slot,
            producer_cold_prefill_us,
            producer_serialize_us,
            key_bytes,
            payload,
        );

        // Header + key are small; the payload is written from the caller's
        // (shared) buffer directly — never concatenate another full
        // file-sized copy (spec §7.3).
        let mut head = Vec::with_capacity(FIXED_HEADER_LEN + key_bytes.len());
        head.extend_from_slice(FILE_MAGIC);
        head.extend_from_slice(&FILE_VERSION.to_le_bytes());
        head.extend_from_slice(&(FIXED_HEADER_LEN as u32).to_le_bytes());
        head.extend_from_slice(&0u32.to_le_bytes()); // flags: none defined
        head.extend_from_slice(&key_len.to_le_bytes());
        head.extend_from_slice(&payload_len.to_le_bytes());
        head.extend_from_slice(&entry_hash);
        head.extend_from_slice(&prefill_slot.to_le_bytes());
        head.extend_from_slice(&producer_cold_prefill_us.to_le_bytes());
        head.extend_from_slice(&producer_serialize_us.to_le_bytes());
        head.extend_from_slice(key_bytes);

        // RAII guard cleans up the temp file if any step between
        // create and rename fails (out-of-space, permission flip,
        // unexpected unlink). Without this, errored inserts leak
        // `.tmp.<pid>` files that the `.axkv`-only eviction sweep
        // would never reclaim.
        let mut guard = TempFileGuard::new(&tmp_path);
        {
            let mut open = fs::OpenOptions::new();
            open.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                open.mode(0o600);
            }
            let mut f = open.open(&tmp_path)?;
            f.write_all(&head)?;
            f.write_all(payload)?;
            f.sync_all()?;
        }
        fs::rename(&tmp_path, &final_path)?;
        guard.disarm();
        // Sync the directory so the rename itself is durable before the
        // store is reported committed (spec §7.3).
        if let Ok(dir) = fs::File::open(&self.dir) {
            let _ = dir.sync_all();
        }

        let evictions = self.evict_until_within_policy_unlocked();
        Ok(DiskPrefixCacheInsertOutcome { evictions })
    }

    /// One-shot v3 entry checksum over an in-memory payload (write side).
    #[allow(clippy::too_many_arguments)]
    fn entry_checksum_oneshot(
        prefill_slot: u32,
        producer_cold_prefill_us: u64,
        producer_serialize_us: u64,
        key_bytes: &[u8],
        payload: &[u8],
    ) -> [u8; 32] {
        let mut hasher = Self::entry_sha256_hasher(
            prefill_slot,
            producer_cold_prefill_us,
            producer_serialize_us,
            key_bytes,
        );
        hasher.update(payload);
        let digest = hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        out
    }

    /// v3 entry checksum: domain + semantic header fields + canonical key
    /// + payload (the checksum field itself is excluded).
    fn entry_sha256_hasher(
        prefill_slot: u32,
        producer_cold_prefill_us: u64,
        producer_serialize_us: u64,
        key_bytes: &[u8],
    ) -> Sha256 {
        let mut hasher = Sha256::new();
        hasher.update(ENTRY_CHECKSUM_DOMAIN);
        hasher.update(0u32.to_le_bytes()); // flags
        hasher.update(prefill_slot.to_le_bytes());
        hasher.update(producer_cold_prefill_us.to_le_bytes());
        hasher.update(producer_serialize_us.to_le_bytes());
        hasher.update(key_bytes);
        hasher
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
        let Ok(_lock) = self.lock_exclusive() else {
            tracing::warn!(
                target: "ax_engine_mlx::prefix_cache",
                dir = %self.dir.display(),
                "disk prefix-cache failed to acquire eviction lock; skipping eviction",
            );
            return 0;
        };
        self.evict_until_within_policy_unlocked()
    }

    fn evict_until_within_policy_unlocked(&self) -> u32 {
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

    fn lock_exclusive(&self) -> Result<DiskPrefixCacheLock, DiskPrefixCacheError> {
        let path = self.dir.join(LOCK_FILE_NAME);
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        // Bounded wait instead of a blocking flock: a wedged process
        // holding the lock must not stall runner startup (the open-time
        // temp sweep) or the prefill store path indefinitely. Disk is
        // strictly additive, so timing out and skipping the operation is
        // always safe.
        let deadline = std::time::Instant::now() + LOCK_ACQUIRE_TIMEOUT;
        loop {
            match file.try_lock_exclusive() {
                Ok(()) => return Ok(DiskPrefixCacheLock { file }),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if std::time::Instant::now() >= deadline {
                        return Err(DiskPrefixCacheError::Io(std::io::Error::new(
                            std::io::ErrorKind::TimedOut,
                            "disk prefix-cache lock acquisition timed out",
                        )));
                    }
                    std::thread::sleep(LOCK_RETRY_INTERVAL);
                }
                Err(e) => return Err(e.into()),
            }
        }
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

struct DiskPrefixCacheLock {
    file: fs::File,
}

impl Drop for DiskPrefixCacheLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
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

/// Internal read outcome: `NotFound` is an ordinary miss, `Invalid` is a
/// validation failure eligible for cleanup, `Io` is an environment error.
enum ReadEntryError {
    NotFound,
    Invalid,
    Io(std::io::Error),
}

fn elapsed_us(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

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
    /// the prefill-token slot or producer cost metadata.
    fn payload_only(bytes: &[u8]) -> DiskPrefixCacheEntry {
        DiskPrefixCacheEntry {
            payload: bytes.to_vec(),
            prefill_output_token: None,
            producer_cold_prefill_us: 0,
            producer_serialize_us: 0,
        }
    }

    /// Schema-v3 key helper preserving the older tests' shape: `salt`
    /// uniquifies the artifact fingerprint the way the v2 token_hash
    /// argument used to uniquify the key.
    fn test_key(
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

    #[test]
    fn insert_then_get_roundtrip() {
        let dir = unique_tempdir("roundtrip");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key_bytes =
            test_key("model-a", "policy-a", "layout-a", 16, 4, 0xdead_beef, &[11, 22, 33, 44]);
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
    fn open_sweeps_stale_temp_files_but_keeps_entries() {
        let dir = unique_tempdir("stale-tmp-sweep");
        // First open seeds the directory with one live entry.
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key_bytes =
            test_key("model-a", "policy-a", "layout-a", 16, 4, 0xdead_beef, &[11, 22, 33, 44]);
        cache
            .insert(&key_bytes, &payload_only(b"PAYLOAD"))
            .expect("insert");
        // Simulate a crash mid-insert from a previous process.
        let stale_tmp = dir.join(format!("{}.tmp.99999", key_sha256_hex(b"other-key")));
        fs::write(&stale_tmp, b"partial").expect("write stale tmp");
        drop(cache);

        let cache = DiskPrefixCache::open(&dir).expect("reopen");
        assert!(!stale_tmp.exists(), "stale temp file must be swept at open");
        let got = cache.get(&key_bytes).expect("get").expect("hit");
        assert_eq!(got.payload, b"PAYLOAD".to_vec());
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
        let key_bytes = test_key("m", "p", "l", 16, 4, 0xfeed_d00d, &[11, 22, 33, 44]);
        let entry = DiskPrefixCacheEntry {
            payload: b"payload".to_vec(),
            prefill_output_token: Some(987_654),
            producer_cold_prefill_us: 111,
            producer_serialize_us: 22,
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
            test_key("model-b", "policy-b", "layout-b", 16, 4, 0xfeed_face, &[11, 22, 33, 44]);
        assert!(cache.get(&key_bytes).expect("get").is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn key_mismatch_returns_miss() {
        let dir = unique_tempdir("collision");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key_a = test_key("model-c", "policy-c", "layout-c", 16, 4, 1, &[11, 22, 33, 44]);
        let key_b = test_key("model-c", "policy-c", "layout-c", 16, 4, 2, &[11, 22, 33, 44]);
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
        let key_bytes = test_key("model-d", "policy-d", "layout-d", 16, 4, 9, &[11, 22, 33, 44]);
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
            test_key("model-c1", "policy-c1", "layout-c1", 16, 4, 0xc04e_7415, &[11, 22, 33, 44]);
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
            test_key("model-c2", "policy-c2", "layout-c2", 16, 4, 0xc0c0_dead, &[11, 22, 33, 44]);
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
..DiskPrefixCachePolicy::default()
};
        let cache = DiskPrefixCache::with_policy(&dir, policy).expect("open");

        let key_a = test_key("m", "p", "l", 16, 4, 0xa1, &[11, 22, 33, 44]);
        let key_b = test_key("m", "p", "l", 16, 4, 0xb2, &[11, 22, 33, 44]);
        let key_c = test_key("m", "p", "l", 16, 4, 0xc3, &[11, 22, 33, 44]);

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
..DiskPrefixCachePolicy::default()
};
        let cache = DiskPrefixCache::with_policy(&dir, policy).expect("open");

        let key_a = test_key("m", "p", "l", 16, 4, 0xa1, &[11, 22, 33, 44]);
        let key_b = test_key("m", "p", "l", 16, 4, 0xb2, &[11, 22, 33, 44]);
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
..DiskPrefixCachePolicy::default()
};
        let cache = DiskPrefixCache::with_policy(&dir, policy).expect("open");
        fs::write(dir.join("NOTES.md"), b"hello").expect("write junk");

        let key_a = test_key("m", "p", "l", 16, 4, 0xaa, &[11, 22, 33, 44]);
        cache
            .insert(&key_a, &payload_only(b"payload-a"))
            .expect("insert a");

        // Junk file must remain untouched.
        assert!(dir.join("NOTES.md").is_file());
        assert!(cache.contains(&key_a));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn lock_file_is_ignored_by_eviction_budget() {
        // Opening the cache creates the sentinel lock file. It must
        // never count as a stored entry, otherwise max_entries=1 would
        // evict the only real cache entry immediately.
        let dir = unique_tempdir("lock-file-budget");
        let policy = DiskPrefixCachePolicy {
            max_bytes: u64::MAX,
            max_entries: 1,
..DiskPrefixCachePolicy::default()
};
        let cache = DiskPrefixCache::with_policy(&dir, policy).expect("open");
        let key = test_key("m", "p", "l", 16, 4, 0x10cc, &[11, 22, 33, 44]);

        let outcome = cache
            .insert(&key, &payload_only(b"payload-a"))
            .expect("insert");

        assert_eq!(outcome.evictions, 0, "lock file must not force eviction");
        assert!(cache.contains(&key), "real entry must survive");
        assert!(dir.join(LOCK_FILE_NAME).is_file(), "lock sentinel exists");

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
            let key_a = test_key("m", "p", "l", 16, 4, 0xa1, &[11, 22, 33, 44]);
            let key_b = test_key("m", "p", "l", 16, 4, 0xb2, &[11, 22, 33, 44]);
            let key_c = test_key("m", "p", "l", 16, 4, 0xc3, &[11, 22, 33, 44]);
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
..DiskPrefixCachePolicy::default()
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
        let key_c = test_key("m", "p", "l", 16, 4, 0xc3, &[11, 22, 33, 44]);
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
    fn concurrent_inserts_and_eviction_preserve_cache_consistency() {
        // M3B advisory-lock stress proxy. Eight threads share an
        // `Arc<DiskPrefixCache>` with a tight `max_entries=8` budget
        // and hammer it with 25 inserts each (200 total) plus parallel
        // explicit eviction sweeps. The lock serializes
        // `insert -> rename -> eviction` per thread; without it,
        // concurrent evictors would race against in-flight renames and
        // produce orphan `.tmp.<pid>` files, dropped entries, or
        // double-removals. After all threads join we assert:
        //   - no `.tmp.*` orphan remains (rename either succeeded
        //     atomically or the TempFileGuard reaped the temp);
        //   - the on-disk entry count stays within the policy budget;
        //   - every surviving `.axkv` file parses cleanly when looked
        //     up by its canonical key — i.e. no torn writes.
        use std::sync::Arc;
        use std::thread;

        let dir = unique_tempdir("multi-thread-stress");
        let policy = DiskPrefixCachePolicy {
            max_bytes: u64::MAX,
            max_entries: 8,
..DiskPrefixCachePolicy::default()
};
        let cache = Arc::new(DiskPrefixCache::with_policy(&dir, policy).expect("open"));

        let threads_per_role = 4;
        let inserts_per_thread = 25;
        let mut handles = Vec::with_capacity(threads_per_role * 2);

        // Inserter threads: each writes `inserts_per_thread` entries
        // with unique keys, returning the key bytes it produced.
        for worker_id in 0..threads_per_role {
            let cache = Arc::clone(&cache);
            handles.push(thread::spawn(move || -> Vec<Vec<u8>> {
                let mut written = Vec::with_capacity(inserts_per_thread);
                for op in 0..inserts_per_thread {
                    let token_hash = ((worker_id as u64) << 32) | op as u64;
                    let key = test_key("m", "p", "l", 16, 4, token_hash, &[11, 22, 33, 44]);
                    let payload = format!("payload-w{worker_id}-op{op}").into_bytes();
                    cache
                        .insert(
                            &key,
                            &DiskPrefixCacheEntry {
                                payload,
                                prefill_output_token: Some(op as u32),
                                producer_cold_prefill_us: 0,
                                producer_serialize_us: 0,
                            },
                        )
                        .expect("insert");
                    written.push(key);
                }
                written
            }));
        }

        // Evictor threads: hammer explicit eviction sweeps in parallel
        // with the inserts. These compete for the same lock, so they
        // must not panic or leave the directory in an inconsistent
        // state regardless of the interleaving with inserts.
        for _ in 0..threads_per_role {
            let cache = Arc::clone(&cache);
            handles.push(thread::spawn(move || -> Vec<Vec<u8>> {
                for _ in 0..inserts_per_thread {
                    let _ = cache.evict_until_within_policy();
                    thread::sleep(std::time::Duration::from_micros(50));
                }
                Vec::new()
            }));
        }

        // Collect every key we know was at-least-once written.
        let all_written: Vec<Vec<u8>> = handles
            .into_iter()
            .flat_map(|h| h.join().expect("worker panicked"))
            .collect();
        assert_eq!(
            all_written.len(),
            threads_per_role * inserts_per_thread,
            "every insert must have completed without error",
        );

        // No `.tmp.*` orphans left behind.
        let tmp_orphans: Vec<_> = fs::read_dir(&dir)
            .expect("read_dir")
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(
            tmp_orphans.is_empty(),
            "no .tmp.* files should remain after concurrent stress; found {tmp_orphans:?}",
        );

        // Entry count within budget. The advisory lock serializes
        // insert+evict so the budget must hold strictly, not "give or
        // take one in flight".
        let surviving_axkv: Vec<_> = fs::read_dir(&dir)
            .expect("read_dir")
            .filter_map(Result::ok)
            .filter(|e| {
                e.path()
                    .extension()
                    .is_some_and(|ext| ext == ENTRY_EXTENSION)
            })
            .collect();
        assert!(
            surviving_axkv.len() <= 8,
            "policy budget breached: {} entries > max_entries=8",
            surviving_axkv.len(),
        );

        // Every surviving file must parse cleanly when looked up
        // through the public `get` API. We can't predict which keys
        // survived eviction, but for each `axkv` file on disk we can
        // recover the canonical key it claims to hold by scanning all
        // produced keys.
        let mut clean_hits = 0;
        for key in &all_written {
            if cache.contains(key)
                && cache
                    .get(key)
                    .expect("get")
                    .filter(|entry| !entry.payload.is_empty())
                    .is_some()
            {
                clean_hits += 1;
            }
        }
        assert_eq!(
            clean_hits,
            surviving_axkv.len(),
            "every surviving .axkv must parse cleanly via get() — torn writes detected",
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn atomic_rename_temp_files_cleaned() {
        // The .tmp.* file should not survive a successful insert.
        let dir = unique_tempdir("atomic");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key_bytes = test_key("model-e", "policy-e", "layout-e", 16, 4, 42, &[11, 22, 33, 44]);
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

    // ── SSD memory optimisation: env-var configuration & budget tests ──

    struct EnvVarGuard {
        saved: Vec<(&'static str, Option<String>)>,
    }

    impl EnvVarGuard {
        fn capture(keys: &[&'static str]) -> Self {
            Self {
                saved: keys.iter().map(|&key| (key, env::var(key).ok())).collect(),
            }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            for (key, value) in self.saved.drain(..) {
                // SAFETY: env-mutating tests hold ENV_LOCK for the whole scope,
                // and this guard restores the original value before releasing it.
                unsafe {
                    match value {
                        Some(v) => env::set_var(key, v),
                        None => env::remove_var(key),
                    }
                }
            }
        }
    }

    #[test]
    fn policy_from_env_respects_max_bytes_override() {
        let key = "AX_MLX_PREFIX_CACHE_DISK_MAX_BYTES";
        let _env_lock = ENV_LOCK.lock().expect("env lock");
        let _env_guard = EnvVarGuard::capture(&[key]);
        unsafe {
            env::set_var(key, "4096");
        }
        let policy = DiskPrefixCachePolicy::from_env();
        assert_eq!(policy.max_bytes, 4096, "env override must be respected");
    }

    #[test]
    fn policy_from_env_respects_max_entries_override() {
        let key = "AX_MLX_PREFIX_CACHE_DISK_MAX_ENTRIES";
        let _env_lock = ENV_LOCK.lock().expect("env lock");
        let _env_guard = EnvVarGuard::capture(&[key]);
        unsafe {
            env::set_var(key, "42");
        }
        let policy = DiskPrefixCachePolicy::from_env();
        assert_eq!(policy.max_entries, 42, "env override must be respected");
    }

    #[test]
    fn policy_from_env_respects_combined_overrides() {
        let key_bytes = "AX_MLX_PREFIX_CACHE_DISK_MAX_BYTES";
        let key_entries = "AX_MLX_PREFIX_CACHE_DISK_MAX_ENTRIES";
        let _env_lock = ENV_LOCK.lock().expect("env lock");
        let _env_guard = EnvVarGuard::capture(&[key_bytes, key_entries]);
        unsafe {
            env::set_var(key_bytes, "8192");
            env::set_var(key_entries, "7");
        }
        let policy = DiskPrefixCachePolicy::from_env();
        assert_eq!(policy.max_bytes, 8192);
        assert_eq!(policy.max_entries, 7);
    }

    #[test]
    fn policy_from_env_falls_back_to_defaults() {
        let key_bytes = "AX_MLX_PREFIX_CACHE_DISK_MAX_BYTES";
        let key_entries = "AX_MLX_PREFIX_CACHE_DISK_MAX_ENTRIES";
        let _env_lock = ENV_LOCK.lock().expect("env lock");
        let _env_guard = EnvVarGuard::capture(&[key_bytes, key_entries]);
        unsafe {
            env::remove_var(key_bytes);
            env::remove_var(key_entries);
        }
        let policy = DiskPrefixCachePolicy::from_env();
        assert_eq!(
            policy.max_bytes, DEFAULT_DISK_CACHE_MAX_BYTES,
            "default max_bytes must be 8 GiB"
        );
        assert_eq!(
            policy.max_entries, DEFAULT_DISK_CACHE_MAX_ENTRIES,
            "default max_entries must be 1024"
        );
    }

    #[test]
    fn policy_from_env_ignores_malformed_values() {
        let key_bytes = "AX_MLX_PREFIX_CACHE_DISK_MAX_BYTES";
        let key_entries = "AX_MLX_PREFIX_CACHE_DISK_MAX_ENTRIES";
        let _env_lock = ENV_LOCK.lock().expect("env lock");
        let _env_guard = EnvVarGuard::capture(&[key_bytes, key_entries]);
        unsafe {
            env::set_var(key_bytes, "not-a-number");
            env::set_var(key_entries, "xyz");
        }
        let policy = DiskPrefixCachePolicy::from_env();
        assert_eq!(
            policy.max_bytes, DEFAULT_DISK_CACHE_MAX_BYTES,
            "malformed max_bytes must fall back to default"
        );
        assert_eq!(
            policy.max_entries, DEFAULT_DISK_CACHE_MAX_ENTRIES,
            "malformed max_entries must fall back to default"
        );
    }

    #[test]
    fn policy_from_env_zero_means_disabled() {
        // `0` must mean "disable the disk tier" — matching the in-memory
        // tier's env semantics — not silently fall back to the 8 GiB
        // default, which is the opposite of what an operator setting 0
        // intends.
        let key_bytes = "AX_MLX_PREFIX_CACHE_DISK_MAX_BYTES";
        let key_entries = "AX_MLX_PREFIX_CACHE_DISK_MAX_ENTRIES";
        let _env_lock = ENV_LOCK.lock().expect("env lock");
        let _env_guard = EnvVarGuard::capture(&[key_bytes, key_entries]);
        unsafe {
            env::set_var(key_bytes, "0");
            env::set_var(key_entries, "0");
        }
        let policy = DiskPrefixCachePolicy::from_env();
        assert_eq!(policy.max_bytes, 0, "zero max_bytes must be preserved");
        assert_eq!(policy.max_entries, 0, "zero max_entries must be preserved");
        assert!(!policy.enabled(), "zero budgets disable the disk tier");

        unsafe {
            env::remove_var(key_bytes);
            env::remove_var(key_entries);
        }
        assert!(
            DiskPrefixCachePolicy::from_env().enabled(),
            "default policy is enabled"
        );
    }

    #[test]
    fn cross_session_persistence_writes_and_reopens() {
        // Simulate two independent sessions sharing the same cache directory.
        // Session 1 writes; session 2 opens the same dir and reads.
        let dir = unique_tempdir("cross-session");
        let key_bytes = test_key("model-x", "policy-x", "layout-x", 16, 4, 0xcafe, &[11, 22, 33, 44]);
        let payload = b"session-1-payload".to_vec();
        let entry = DiskPrefixCacheEntry {
            payload: payload.clone(),
            prefill_output_token: Some(42),
            producer_cold_prefill_us: 0,
            producer_serialize_us: 0,
        };

        // Session 1: open, write, drop.
        {
            let cache1 = DiskPrefixCache::open(&dir).expect("session 1 open");
            cache1.insert(&key_bytes, &entry).expect("session 1 insert");
        }

        // Session 2: reopen, read.
        {
            let cache2 = DiskPrefixCache::open(&dir).expect("session 2 open");
            let got = cache2.get(&key_bytes).expect("session 2 get").expect("hit");
            assert_eq!(got.payload, payload);
            assert_eq!(got.prefill_output_token, Some(42));
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn disk_cache_stores_and_measures_large_payload() {
        // Verify that a realistically-sized KV cache payload (simulated)
        // is stored and measured correctly for SSD capacity accounting.
        let dir = unique_tempdir("large-payload");
        let policy = DiskPrefixCachePolicy {
            max_bytes: 1024 * 1024, // 1 MiB budget
            max_entries: usize::MAX,
..DiskPrefixCachePolicy::default()
};
        let cache = DiskPrefixCache::with_policy(&dir, policy).expect("open");

        // 256 KiB payload — should fit within 1 MiB budget
        let payload_256k = vec![0xAB_u8; 256 * 1024];
        let key1 = test_key("m", "p", "l", 16, 4, 1, &[11, 22, 33, 44]);
        cache
            .insert(&key1, &payload_only(&payload_256k))
            .expect("insert 256k");

        // Second 256 KiB — still within budget
        let key2 = test_key("m", "p", "l", 16, 4, 2, &[11, 22, 33, 44]);
        let outcome = cache
            .insert(&key2, &payload_only(&payload_256k))
            .expect("insert second 256k");
        assert_eq!(outcome.evictions, 0, "two 256k payloads must fit in 1 MiB");

        // Verify both are readable
        assert!(cache.get(&key1).expect("get").is_some());
        assert!(cache.get(&key2).expect("get").is_some());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn disk_cache_evicts_when_payload_exceeds_byte_budget() {
        // With a tight budget, inserting a large payload forces eviction.
        let dir = unique_tempdir("evict-large");
        let budget = 4096u64; // 4 KiB
        let policy = DiskPrefixCachePolicy {
            max_bytes: budget,
            max_entries: usize::MAX,
..DiskPrefixCachePolicy::default()
};
        let cache = DiskPrefixCache::with_policy(&dir, policy).expect("open");

        let key1 = test_key("m", "p", "l", 16, 4, 0xaa, &[11, 22, 33, 44]);
        cache
            .insert(&key1, &payload_only(&vec![0u8; 2048]))
            .expect("insert first");
        std::thread::sleep(std::time::Duration::from_millis(1100));

        let key2 = test_key("m", "p", "l", 16, 4, 0xbb, &[11, 22, 33, 44]);
        let outcome = cache
            .insert(&key2, &payload_only(&vec![0u8; 2048]))
            .expect("insert second");
        // Two 2048-byte payloads + header overhead > 4096 budget
        assert!(outcome.evictions >= 1, "must evict at least one entry");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn with_policy_trims_existing_entries_on_reopen() {
        // Pre-populate 5 entries, then reopen with max_entries=2.
        // Only 2 newest should survive.
        let dir = unique_tempdir("reopen-trim");
        {
            let cache = DiskPrefixCache::open(&dir).expect("open default");
            for i in 0..5u64 {
                let key = test_key("m", "p", "l", 16, 4, i, &[11, 22, 33, 44]);
                cache
                    .insert(&key, &payload_only(format!("payload-{i}").as_bytes()))
                    .expect("insert");
                std::thread::sleep(std::time::Duration::from_millis(1050));
            }
        }

        let tight = DiskPrefixCachePolicy {
            max_bytes: u64::MAX,
            max_entries: 2,
..DiskPrefixCachePolicy::default()
};
        let cache = DiskPrefixCache::with_policy(&dir, tight).expect("reopen tight");

        // Entries 0-2 must be evicted; 3-4 must survive.
        for i in 0..3 {
            let key = test_key("m", "p", "l", 16, 4, i, &[11, 22, 33, 44]);
            assert!(!cache.contains(&key), "entry {i} must be evicted");
        }
        for i in 3..5 {
            let key = test_key("m", "p", "l", 16, 4, i, &[11, 22, 33, 44]);
            assert!(cache.contains(&key), "entry {i} must survive");
        }

        let _ = fs::remove_dir_all(&dir);
    }
    #[test]
    fn token_content_mismatch_returns_miss() {
        // Simulated 64-bit FNV token_hash collision: two different prompts
        // with the SAME token_hash and token_count. Schema v2 commits to
        // the token content via SHA-256, so the embedded-key comparison
        // must reject the swapped file instead of restoring the wrong KV.
        let dir = unique_tempdir("fnv-collision");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key_x = test_key("m", "p", "l", 16, 4, 7, &[1, 2, 3, 4]);
        let key_y = test_key("m", "p", "l", 16, 4, 7, &[9, 9, 9, 9]);
        assert_ne!(
            key_x, key_y,
            "same token_hash but different tokens must produce different keys"
        );
        cache
            .insert(&key_x, &payload_only(b"payload-x"))
            .expect("insert");
        fs::rename(cache.path_for(&key_x), cache.path_for(&key_y)).expect("rename");
        assert!(
            cache.get(&key_y).expect("get").is_none(),
            "token-content mismatch must miss, never restore the wrong KV"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_entry_is_unlinked_on_get() {
        // A file that fails validation must be reclaimed, not left
        // consuming budget until mtime eviction reaches it.
        let dir = unique_tempdir("corrupt-unlink");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key = test_key("m", "p", "l", 16, 4, 0xbad, &[11, 22, 33, 44]);
        cache
            .insert(&key, &payload_only(b"payload"))
            .expect("insert");
        let path = cache.path_for(&key);
        let mut raw = fs::read(&path).expect("read");
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        fs::write(&path, raw).expect("corrupt");

        assert!(cache.get(&key).expect("get").is_none(), "corruption misses");
        assert!(
            !path.exists(),
            "corrupt entry must be unlinked by the failed get"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn get_refreshes_mtime_so_eviction_approximates_lru() {
        // Hot entries must survive eviction over cold-but-newer ones.
        let dir = unique_tempdir("lru-touch");
        let policy = DiskPrefixCachePolicy {
            max_bytes: u64::MAX,
            max_entries: 2,
..DiskPrefixCachePolicy::default()
};
        let cache = DiskPrefixCache::with_policy(&dir, policy).expect("open");
        let key_a = test_key("m", "p", "l", 16, 4, 0xa, &[11, 22, 33, 44]);
        let key_b = test_key("m", "p", "l", 16, 4, 0xb, &[11, 22, 33, 44]);
        let key_c = test_key("m", "p", "l", 16, 4, 0xc, &[11, 22, 33, 44]);

        cache.insert(&key_a, &payload_only(b"a")).expect("insert a");
        std::thread::sleep(std::time::Duration::from_millis(1100));
        cache.insert(&key_b, &payload_only(b"b")).expect("insert b");
        std::thread::sleep(std::time::Duration::from_millis(1100));
        // Touch a: it is now more recently used than b.
        assert!(cache.get(&key_a).expect("get").is_some());
        std::thread::sleep(std::time::Duration::from_millis(1100));
        cache.insert(&key_c, &payload_only(b"c")).expect("insert c");

        assert!(cache.contains(&key_a), "hot entry must survive");
        assert!(!cache.contains(&key_b), "cold entry must be evicted");
        assert!(cache.contains(&key_c), "new entry must survive");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn oversized_payload_is_skipped_not_self_evicting() {
        let dir = unique_tempdir("oversized");
        let policy = DiskPrefixCachePolicy {
            max_bytes: 1024,
            max_entries: usize::MAX,
..DiskPrefixCachePolicy::default()
};
        let cache = DiskPrefixCache::with_policy(&dir, policy).expect("open");
        let key_small = test_key("m", "p", "l", 16, 4, 1, &[11, 22, 33, 44]);
        cache
            .insert(&key_small, &payload_only(b"small"))
            .expect("insert small");

        let key_big = test_key("m", "p", "l", 16, 4, 2, &[11, 22, 33, 44]);
        let outcome = cache
            .insert(&key_big, &payload_only(&vec![0u8; 4096]))
            .expect("oversized insert returns Ok");
        assert_eq!(outcome.evictions, 0, "skipped store must not evict");
        assert!(
            !cache.contains(&key_big),
            "oversized entry must not be written"
        );
        assert!(
            cache.contains(&key_small),
            "existing entries must not be sacrificed for an undurable store"
        );
        let _ = fs::remove_dir_all(&dir);
    }
    #[test]
    fn canonical_key_v3_changes_for_every_identity_field() {
        let base = DiskPrefixKeyFields {
            model_id: "m",
            artifact_fingerprint_sha256: "aa",
            route_policy: "p",
            layer_layout: "l",
            kv_payload_version: 3,
            block_size_tokens: 16,
            token_count: 4,
            tokens: &[1, 2, 3, 4],
        };
        let reference = canonical_key_bytes(&base);
        let variants = [
            canonical_key_bytes(&DiskPrefixKeyFields {
                model_id: "m2",
                ..base
            }),
            canonical_key_bytes(&DiskPrefixKeyFields {
                artifact_fingerprint_sha256: "bb",
                ..base
            }),
            canonical_key_bytes(&DiskPrefixKeyFields {
                route_policy: "p2",
                ..base
            }),
            canonical_key_bytes(&DiskPrefixKeyFields {
                layer_layout: "l2",
                ..base
            }),
            canonical_key_bytes(&DiskPrefixKeyFields {
                kv_payload_version: 4,
                ..base
            }),
            canonical_key_bytes(&DiskPrefixKeyFields {
                block_size_tokens: 32,
                ..base
            }),
            canonical_key_bytes(&DiskPrefixKeyFields {
                tokens: &[1, 2, 3, 5],
                ..base
            }),
        ];
        for (idx, variant) in variants.iter().enumerate() {
            assert_ne!(
                &reference, variant,
                "identity field {idx} must change the canonical key"
            );
        }
        // Length-prefixed strings: shifting a boundary must not alias.
        let shifted = canonical_key_bytes(&DiskPrefixKeyFields {
            model_id: "mp",
            route_policy: "",
            ..base
        });
        assert_ne!(reference, shifted, "string boundaries must be unambiguous");
    }

    #[test]
    fn v3_entry_roundtrips_producer_metadata() {
        let dir = unique_tempdir("producer-meta");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key = test_key("m", "p", "l", 16, 4, 0x11, &[11, 22, 33, 44]);
        let entry = DiskPrefixCacheEntry {
            payload: b"payload-meta".to_vec(),
            prefill_output_token: Some(7),
            producer_cold_prefill_us: 123_456,
            producer_serialize_us: 7_890,
        };
        cache.insert(&key, &entry).expect("insert");
        let got = cache.get(&key).expect("get").expect("hit");
        assert_eq!(got.payload, entry.payload);
        assert_eq!(got.prefill_output_token, Some(7));
        assert_eq!(got.producer_cold_prefill_us, 123_456);
        assert_eq!(got.producer_serialize_us, 7_890);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn v3_read_rejects_unknown_flags_and_trailing_bytes() {
        let dir = unique_tempdir("flags-trailing");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key = test_key("m", "p", "l", 16, 4, 0x22, &[11, 22, 33, 44]);
        cache
            .insert(&key, &payload_only(b"payload"))
            .expect("insert");
        let path = cache.path_for(&key);
        let pristine = fs::read(&path).expect("read");

        // Unknown flag bit set (offset 12..16).
        let mut flagged = pristine.clone();
        flagged[12] |= 0x01;
        fs::write(&path, &flagged).expect("write");
        assert!(
            cache.get(&key).expect("get").is_none(),
            "unknown flags are from the future and must fail closed"
        );

        // Trailing bytes beyond the declared sections.
        let mut trailing = pristine.clone();
        trailing.push(0);
        fs::write(&path, &trailing).expect("write");
        assert!(
            cache.get(&key).expect("get").is_none(),
            "trailing bytes outside declared sections are invalid"
        );

        // Header corruption on the checksum must miss.
        let mut bad_hash = pristine.clone();
        bad_hash[28] ^= 0xFF;
        fs::write(&path, &bad_hash).expect("write");
        assert!(cache.get(&key).expect("get").is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn v3_read_rejects_symlinked_entry() {
        let dir = unique_tempdir("symlink");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key_real = test_key("m", "p", "l", 16, 4, 0x33, &[11, 22, 33, 44]);
        let key_link = test_key("m", "p", "l", 16, 4, 0x44, &[11, 22, 33, 44]);
        cache
            .insert(&key_real, &payload_only(b"payload"))
            .expect("insert");
        std::os::unix::fs::symlink(cache.path_for(&key_real), cache.path_for(&key_link))
            .expect("symlink");
        assert!(
            cache.get(&key_link).expect("get").is_none(),
            "symlinked entries must be rejected"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn stale_v2_format_entry_is_a_miss_and_unlinked() {
        let dir = unique_tempdir("stale-v2");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key = test_key("m", "p", "l", 16, 4, 0x55, &[11, 22, 33, 44]);
        // Hand-craft a v2-shaped file at the key's path.
        let mut raw = Vec::new();
        raw.extend_from_slice(FILE_MAGIC);
        raw.extend_from_slice(&2u32.to_le_bytes());
        raw.extend_from_slice(&[0u8; 48]); // v2 header remainder
        fs::write(cache.path_for(&key), &raw).expect("write v2");
        assert!(
            cache.get(&key).expect("get").is_none(),
            "older format versions are clean misses"
        );
        assert!(
            !cache.path_for(&key).exists(),
            "stale-version entries are reclaimed lazily"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn fresh_root_and_entries_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = unique_tempdir("perms");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        assert_eq!(
            fs::metadata(&dir).expect("dir meta").permissions().mode() & 0o777,
            0o700,
            "fresh cache root must be owner-only"
        );
        let key = test_key("m", "p", "l", 16, 4, 0x66, &[11, 22, 33, 44]);
        cache
            .insert(&key, &payload_only(b"payload"))
            .expect("insert");
        assert_eq!(
            fs::metadata(cache.path_for(&key))
                .expect("entry meta")
                .permissions()
                .mode()
                & 0o777,
            0o600,
            "entries must be owner-only"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn admission_emits_every_closed_reason() {
        let policy = DiskPrefixCachePolicy {
            min_prefix_tokens: 2048,
            max_entry_bytes: 1024 * 1024,
            min_savings_us: 10_000,
            ..DiskPrefixCachePolicy::default()
        };
        let throughput = DiskThroughputSnapshot {
            write_bytes_per_us: 100.0,   // 100 B/µs
            restore_bytes_per_us: 200.0, // 200 B/µs
        };

        let disabled = DiskPrefixCachePolicy {
            admission: DiskAdmissionMode::Disabled,
            ..policy
        };
        assert_eq!(
            disabled.evaluate_admission(4096, 1024, Some(1), Some(throughput)).0,
            DiskAdmissionReason::Disabled
        );

        // Oversize rejects even in `always` (diagnostic mode obeys caps).
        let always = DiskPrefixCachePolicy {
            admission: DiskAdmissionMode::Always,
            ..policy
        };
        assert_eq!(
            always
                .evaluate_admission(4096, 2 * 1024 * 1024, None, None)
                .0,
            DiskAdmissionReason::EntryTooLarge
        );
        assert_eq!(
            always.evaluate_admission(1, 1024, None, None).0,
            DiskAdmissionReason::AdmittedAlways,
            "always bypasses the value model and the prefix floor"
        );

        assert_eq!(
            policy.evaluate_admission(100, 1024, Some(1), Some(throughput)).0,
            DiskAdmissionReason::PrefixTooShort
        );
        assert_eq!(
            policy.evaluate_admission(4096, 1024, None, Some(throughput)).0,
            DiskAdmissionReason::NoCostModel
        );
        assert_eq!(
            policy.evaluate_admission(4096, 1024, Some(1_000_000), None).0,
            DiskAdmissionReason::NoCostModel
        );

        // 1 MiB entry: restore ≈ 5.2 ms, write ≈ 10.5 ms. A 10 s cold
        // prefill clears write + min_savings comfortably.
        let (reason, estimate) = policy.evaluate_admission(
            4096,
            1024 * 1024,
            Some(10_000_000),
            Some(throughput),
        );
        assert_eq!(reason, DiskAdmissionReason::AdmittedPositiveValue);
        assert!(estimate.restore_us > 0 && estimate.write_us > 0);

        // A 6 ms cold prefill loses to lifecycle cost.
        assert_eq!(
            policy
                .evaluate_admission(4096, 1024 * 1024, Some(6_000), Some(throughput))
                .0,
            DiskAdmissionReason::PredictedNoSavings
        );

        // Saturation: absurd byte counts must not panic or admit.
        assert_eq!(
            policy
                .evaluate_admission(
                    u32::MAX,
                    u64::MAX,
                    Some(u64::MAX),
                    Some(DiskThroughputSnapshot {
                        write_bytes_per_us: f64::MIN_POSITIVE,
                        restore_bytes_per_us: f64::MIN_POSITIVE,
                    }),
                )
                .0,
            DiskAdmissionReason::EntryTooLarge
        );

        // Every reason has a distinct stable code.
        let reasons = [
            DiskAdmissionReason::AdmittedAlways,
            DiskAdmissionReason::AdmittedPositiveValue,
            DiskAdmissionReason::Disabled,
            DiskAdmissionReason::UnsupportedLayout,
            DiskAdmissionReason::PrefixTooShort,
            DiskAdmissionReason::EntryTooLarge,
            DiskAdmissionReason::NoCostModel,
            DiskAdmissionReason::PredictedNoSavings,
            DiskAdmissionReason::CapacityPressure,
            DiskAdmissionReason::ArtifactIdentityUnavailable,
        ];
        let mut codes: Vec<u32> = reasons.iter().map(|r| r.code()).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), reasons.len(), "codes must be distinct");
    }

    #[test]
    fn get_timed_reports_stage_timings_and_bytes() {
        let dir = unique_tempdir("stage-timings");
        let cache = DiskPrefixCache::open(&dir).expect("open");
        let key = test_key("m", "p", "l", 16, 4, 0x77, &[11, 22, 33, 44]);
        let payload = vec![0x5Au8; 128 * 1024];
        cache.insert(&key, &payload_only(&payload)).expect("insert");
        let (entry, timings) = cache.get_timed(&key).expect("get").expect("hit");
        assert_eq!(entry.payload, payload);
        assert!(
            timings.bytes_read >= payload.len() as u64,
            "bytes_read covers header + key + payload"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
