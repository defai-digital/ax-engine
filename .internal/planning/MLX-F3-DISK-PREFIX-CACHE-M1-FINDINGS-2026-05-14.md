# F3 M1 — Disk Prefix Cache Serialization Foundation (2026-05-14)

PRD: `MLX-DISK-PREFIX-CACHE-PRD-2026-05-14.md`
Parent PRD: `DS4-LEARNINGS-FOLLOWUP-PRD-2026-05-14.md` §6

## 1. Status: **M1 LANDED**. M2/M3/M4/M5 deferred to follow-up session(s).

The session that opened the parent PRD intentionally scoped F3 as
~1 engineer-week of work over five milestones. This artifact documents
that **M1 (serialization correctness — the PRD's largest risk
concentration) is now shipped with 11 passing unit tests**, and that
the remaining milestones are correctly held back rather than rushed.

That trade-off is the right one for best-practice delivery: a solid
foundation now, with the runner wire-up (M2), eviction + concurrency
(M3), and integration validation (M4) handled in dedicated future
sessions that can give each milestone the focus its risk surface
warrants.

## 2. What landed (M1)

### 2.1 `MlxKVCache::serialize_to_bytes` and `try_deserialize_from_bytes`

Two new methods on `MlxKVCache` in
`crates/ax-engine-mlx/src/kv_cache.rs`. The serialised payload uses
the wire format from PRD §3.3 / §4:

- Header: `AXKB` magic + `version=1` + `seq_len` + `growth_count` +
  `layer_count` + reserved.
- Per layer: kind byte (Empty / FA / MLA / Linear) + reserved + a
  payload appropriate to the kind.
- Per tensor: dtype tag + ndim + reserved + shape (zero-padded to
  ndim=4) + byte_count + raw bytes.

Round-trip tests in `kv_cache::tests`:

```
serialize_empty_cache_roundtrips ............................ ok
serialize_fa_cache_roundtrips_values ........................ ok
serialize_mla_cache_roundtrips_values ....................... ok
deserialize_rejects_bad_magic ............................... ok
deserialize_rejects_unsupported_version ..................... ok
deserialize_rejects_truncated_payload ....................... ok
```

A new `MlxKVCacheSerializeError` enum gives callers structured error
information for the four common rejection modes (bad magic, version
mismatch, truncation, unknown dtype / layer kind). Implements
`std::error::Error` for trait-object propagation.

TurboQuant shadow storage is intentionally **not** in the wire format
(PRD §4.5). On reload it rebuilds from base KV via the existing
runtime sync path — costs a few extra decode steps once on first use,
versus widening the on-disk format with a TurboQuant-specific section.

### 2.2 `DiskPrefixCache` file-I/O wrapper

New module `crates/ax-engine-mlx/src/disk_prefix_cache.rs` with the
thin file-level wrapper around the kv_cache payload. Outer file
framing (separate magic `AXKV` so a stray payload cannot be
mistaken for a complete file) includes payload SHA256 for integrity
and the canonical key bytes for hash-collision detection.

API:

```rust
pub fn canonical_key_bytes(
    model_id: &str, route_policy: &str, layer_layout: &str,
    block_size_tokens: u32, token_count: u32, token_hash: u64,
) -> Vec<u8>;

pub struct DiskPrefixCache { /* dir: PathBuf */ }
impl DiskPrefixCache {
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self, DiskPrefixCacheError>;
    pub fn get(&self, key_bytes: &[u8]) -> Result<Option<Vec<u8>>, DiskPrefixCacheError>;
    pub fn insert(&self, key_bytes: &[u8], payload: &[u8]) -> Result<(), DiskPrefixCacheError>;
}
```

Writes use atomic rename (write to `*.tmp.PID` in the same directory,
fsync, then rename) so readers cannot observe a partial write.
Reads validate the outer magic, version, payload SHA256, and key
bytes match — any mismatch surfaces as `Ok(None)` (cache miss), not
as an error.

Unit tests in `disk_prefix_cache::tests`:

```
insert_then_get_roundtrip ................................... ok
get_miss_returns_none ....................................... ok
key_mismatch_returns_miss ................................... ok
corrupt_payload_returns_miss ................................ ok
atomic_rename_temp_files_cleaned ............................ ok
```

The `key_mismatch_returns_miss` and `corrupt_payload_returns_miss`
tests are particularly important: they prove the integrity gates in
`parse_file` work, which is the PRD §G4 fail-closed-on-integrity
requirement.

### 2.3 New dependency

Added `sha2.workspace = true` to `crates/ax-engine-mlx/Cargo.toml`.
Other crates (`ax-engine-core`, `ax-engine-sdk`) already depend on
sha2, so the workspace surface is unchanged.

## 3. What is explicitly **not** in M1

| Milestone | Status | Reason for deferral |
|---|---|---|
| M2 — wire L2 into runner's `restore_reused_prefix_state` and `store_prompt_prefix_snapshots`; add `disk_*` telemetry counters | Deferred | Touches the prefix-cache hot path in `runner.rs`; needs careful integration with the L1 / probe-fallback path added in `27ca5904` and the MLA chunk-alignment safety in `ade74c2f`. Worth a focused session. |
| M3 — LRU eviction (byte + entry budget) and multi-process flock for eviction races | Deferred | The PRD §5 / §6 design is non-trivial; correctness requires multi-process stress tests that need real workloads. |
| M4 — cross-restart `verify_prefix_reuse_equivalence.py` runs on FA / SWA / MLA / Linear; four-process stress test | Deferred | Depends on M2 and M3. |
| M5 — docs updates to `docs/KV-CACHE.md`, `docs/PERFORMANCE.md` | Deferred | Pairs naturally with M2's user-visible env-flag surface. |

This deferral pattern matches PRD §11's own milestone breakdown:
"Total: roughly one engineer-week. Largest risk concentration: M1
(serialization correctness) and M3 (concurrency)." M1 ships now, M3
sits with its co-listed risk concentration until it has a dedicated
slice.

## 4. What M2 will need from M1

When the next session opens M2, the M1 surface is ready to consume:

- `MlxPrefixCache` in `runner.rs` already produces an `MlxPrefixCacheKey`
  per request. M2 takes the same six fields (`model_id`,
  `route_policy`, `layer_layout`, `block_size_tokens`, `token_count`,
  `token_hash`) and calls `disk_prefix_cache::canonical_key_bytes(...)`
  to get the disk-cache key.
- On L1 miss after the existing probe fallback (commit `27ca5904`),
  M2 calls `DiskPrefixCache::get(key_bytes)`. On hit, deserialize via
  `MlxKVCache::try_deserialize_from_bytes(payload)` and proceed
  through the existing restore path.
- On L1 store (in `store_prompt_prefix_snapshots`), M2 additionally
  calls `MlxKVCache::serialize_to_bytes()` and
  `DiskPrefixCache::insert(...)`. The serialization is cheap relative
  to the existing prefill cost, so it does not need to be on a
  background thread for the first cut.
- Telemetry: add `disk_hits`, `disk_misses`, `disk_inserts`,
  `disk_bytes_kib` counters in `MlxPrefixCacheTelemetry` alongside
  the existing in-memory counters. Bench harnesses pick them up
  automatically once added to `TELEMETRY_KEYS` in the Python
  scripts.
- Configuration: `AX_MLX_PREFIX_CACHE_DIR` env var enables L2; unset
  leaves L2 disabled (default off, matching PRD §3.5).

## 5. What M2 must keep correct

M1 was tested with synthetic in-memory caches. M2 must hold these
production invariants:

- **MLA snapshots only round-trip safely at `prefill_chunk = 16`**
  (or whatever satisfies the chunk-alignment property from commit
  `ade74c2f`). The disk format is bit-faithful, but a snapshot
  produced under chunk=512 still has the same shape-dependent fp
  drift on reload, by construction. The M2 wire-up must refuse to
  store a snapshot when the producing path was not chunk-aligned.
- **SWA snapshots after window rotation must not be stored** (PRD
  §4.2). The in-memory cache already refuses to store an alignment-
  restricted snapshot when `full_block_tokens != available_tokens`;
  the M2 path inherits this — disk storage is gated on the same
  predicate.
- **TurboQuant shadow rebuilds lazily on reload.** First decode step
  after a disk hit will trigger the existing `sync_turboquant_shadow_storage`
  pathway; telemetry should be quiet about this.

## 6. Files

| Path | Status |
|---|---|
| `crates/ax-engine-mlx/src/kv_cache.rs` | `serialize_to_bytes` + `try_deserialize_from_bytes` + 6 tests |
| `crates/ax-engine-mlx/src/disk_prefix_cache.rs` | New module: canonical_key_bytes + DiskPrefixCache + 5 tests |
| `crates/ax-engine-mlx/src/lib.rs` | + `pub mod disk_prefix_cache` |
| `crates/ax-engine-mlx/Cargo.toml` | + `sha2.workspace = true` |

## 7. Closure conditions for this M1 artifact (not the full PRD)

- ✅ Round-trip tests pass for empty / FA / MLA caches.
- ✅ Rejection tests pass for bad magic / unsupported version /
  truncation.
- ✅ File-level integrity gates pass (corrupt payload, key mismatch).
- ✅ `cargo clippy -p ax-engine-mlx --all-targets --all-features -- -D warnings`
  is clean.
- ✅ `cargo test -p ax-engine-mlx` is green for the new test suite.

The full F3 PRD remains **open** with M2 / M3 / M4 / M5 still to
land. This artifact updates the parent PRD's §6 row from "spec done"
to "M1 landed; M2-M5 open".

---

**Status:** M1 closed. Parent PRD updated. Recommend the next session
on F3 picks up M2 (runner wire-up) — it consumes the M1 surface most
directly and unlocks the first user-visible disk-cache hits.
