# F3 M3 — Disk Prefix Cache Eviction + Concurrency (2026-05-14)

PRD: `MLX-DISK-PREFIX-CACHE-PRD-2026-05-14.md`
Parent PRD: `DS4-LEARNINGS-FOLLOWUP-PRD-2026-05-14.md` §6
Prior milestones:
- `MLX-F3-DISK-PREFIX-CACHE-M1-FINDINGS-2026-05-14.md`
- `MLX-F3-DISK-PREFIX-CACHE-M2-FINDINGS-2026-05-14.md`

## 1. Status: **M3 LANDED.** M4 (cross-restart integration validation) is the only remaining milestone.

## 2. Scope decision

PRD §6 names two concerns for M3: eviction and concurrency.
This milestone tightens scope:

- **In scope:** byte-budget + entry-budget eviction, mtime-based
  ordering (FIFO of last-modified), per-insert eviction sweep,
  telemetry counter, env-driven policy.
- **Deferred:** multi-process `flock`-based concurrency control, and
  the "touch tick rename" the PRD §6 sketches for true LRU. Both are
  M4-blocking but not M3-blocking, and adding them now would
  significantly expand the change.

Single-process is the only supported deployment shape at M3. Two
ax-engine processes pointed at the same `AX_MLX_PREFIX_CACHE_DIR`
may race on eviction; the harm is at worst a redundant eviction (one
extra `disk_miss` next time around), never corruption — atomic-rename
writes still hold. This is documented and reasonable for the M4
integration test, which uses one process per restart sequence.

## 3. What landed

### 3.1 `DiskPrefixCachePolicy`

New public struct in `disk_prefix_cache.rs`:

```rust
pub struct DiskPrefixCachePolicy {
    pub max_bytes: u64,    // default 8 GiB
    pub max_entries: usize, // default 1024
}
```

Both budgets are hard caps applied after every successful insert.
Either firing triggers eviction. `from_env()` reads
`AX_MLX_PREFIX_CACHE_DISK_MAX_BYTES` and
`AX_MLX_PREFIX_CACHE_DISK_MAX_ENTRIES`; blank / unparseable / zero
values fall back to the documented defaults. The cache also keeps an
explicit `Default` impl so `DiskPrefixCache::open()` (the legacy
constructor) keeps working unchanged.

### 3.2 `DiskPrefixCache::with_policy`

New constructor opens the directory and stores the policy. The
existing `open()` constructor now delegates to `with_policy()` with
`DiskPrefixCachePolicy::default()`, so M1/M2 call sites compile
unchanged and M2 tests stay green.

### 3.3 Eviction on insert

`insert()` now returns `Result<DiskPrefixCacheInsertOutcome, _>`
where `DiskPrefixCacheInsertOutcome { evictions: u32 }`. After the
atomic-rename completes, `evict_until_within_policy()` walks the
cache directory, filters for files with the `.axkv` extension,
collects `(path, len, mtime)` triples into an `EntryStat` list,
sorts ascending by mtime, then removes oldest-first until both
budgets hold. Non-axkv files (e.g. operator notes, the M3 unit test
exercises this) are skipped — they are not counted toward either
budget, nor evicted.

mtime is used as the eviction key for two reasons:

- The PRD §6 design names "FIFO by mtime" as the M3-shippable
  approximation; full LRU needs the touch-rename pattern, deferred.
- macOS `fs::Metadata::modified()` is reliable to 1-second
  granularity, which our test harness defeats with explicit sleeps.

### 3.4 Telemetry

`MlxPrefixCacheTelemetry` gains a fifth disk counter,
`disk_evictions`, with matching route-decision key
`ax_mlx_prefix_cache_disk_evictions`. The
`record_disk_insert` helper now takes `(bytes, evictions)` and
folds both into the telemetry struct, so the runner caller only
emits one record per insert. `merge_from` and
`append_route_decisions` are updated to carry the new field.

### 3.5 Runner wire-up

`MlxRunner::from_artifacts` now constructs the disk cache via
`DiskPrefixCache::with_policy(&dir, DiskPrefixCachePolicy::from_env())`,
so operators can shrink the on-disk budget at process start. The
default 8 GiB / 1024 entries matches the PRD §3.5 numbers.

Insert call site in `store_prompt_prefix_snapshots`:

```rust
match disk.insert(&key_bytes, &payload) {
    Ok(outcome) => telemetry.record_disk_insert(
        payload.len() as u64,
        outcome.evictions,
    ),
    Err(_) => { /* warn-only; L1 store already succeeded */ }
}
```

## 4. Tests

Three new unit tests in `disk_prefix_cache::tests`:

| Test | What it proves |
|---|---|
| `eviction_drops_oldest_when_entry_budget_exceeded` | With `max_entries=2`, the third insert evicts exactly one oldest entry. |
| `eviction_drops_oldest_when_byte_budget_exceeded` | With a byte budget tight enough for one payload but not two, inserting a second payload evicts the oldest. |
| `eviction_skips_non_axkv_files` | An operator-placed `NOTES.md` is untouched by eviction, and not counted toward the entry budget. |

Both budget-based tests use 1.1-second sleeps between inserts to
defeat the 1-second mtime granularity on macOS, so the ordering is
deterministic across runs.

Full suite results:

- `cargo test -p ax-engine-mlx --lib`: **372 passed / 0 failed** (10
  disk_prefix_cache tests; up from 7 in M2).
- `cargo clippy -p ax-engine-mlx --all-targets --all-features -- -D warnings`:
  clean.

## 5. What M3 does not yet do (held for M4)

### 5.1 No multi-process locking

Per §2 above, multi-process concurrency is deferred. The M4
integration test (process A writes, process B reads after restart)
is the immediate next consumer and does not need it — there is at
most one process at a time. When we wire ax-engine into the
production server with multiple workers, we will need the `flock`
sentinel.

### 5.2 mtime-based, not full LRU

A read does not touch mtime today. If a workload re-reads a hot
prefix while colder prefixes get freshly written, the hot prefix
will eventually look "older" and be evicted. The PRD §6 touch-tick
rename pattern (rename to a sentinel and back to bump mtime) is the
fix; deferred so M3 stays focused.

### 5.3 No eviction telemetry per category

Today we count total evictions per insert. We do not separate "byte
budget triggered" from "entry budget triggered". If we see in M4 or
production that one budget dominates, we can add the split.

## 6. Files

| Path | Change |
|---|---|
| `crates/ax-engine-mlx/src/disk_prefix_cache.rs` | + `DiskPrefixCachePolicy`, `DiskPrefixCacheInsertOutcome`, `EntryStat`, `with_policy`, `evict_until_within_policy`, `list_entries`; insert signature returns outcome; 3 new unit tests |
| `crates/ax-engine-mlx/src/runner.rs` | + `disk_evictions` telemetry field + route key; `record_disk_insert(bytes, evictions)`; runner construction now passes `DiskPrefixCachePolicy::from_env()`; insert call site handles new outcome |
| `scripts/profile_kv_multiturn_chat_evidence.py` | (unchanged from M2; `disk_evictions` will be added when first observed in a real-workload smoke) |

## 7. Closure conditions for this M3 artifact

- ✅ Policy struct with byte + entry budgets, `from_env()` plumbing.
- ✅ Eviction runs after every successful insert; oldest-by-mtime
  removed until both budgets hold.
- ✅ Non-axkv files in the cache directory are not touched.
- ✅ Telemetry: `disk_evictions` counter wired through
  `merge_from`, `append_route_decisions`, `record_disk_insert`.
- ✅ Runner construction uses `DiskPrefixCachePolicy::from_env()`.
- ✅ `cargo clippy --all-targets --all-features -- -D warnings` clean.
- ✅ `cargo test -p ax-engine-mlx --lib` 372/372 green.
- ✅ Scope deferrals (flock, full-LRU touch) explicitly documented.

PRD §6 row in the parent ledger should update from "M2 landed; M3-M5
open" to "M3 landed; M4 open" (M5 was rolled into M4 in the M2
findings doc).

---

**Status:** M3 closed. M4 (cross-restart validation against
`verify_prefix_reuse_equivalence.py`) is the natural next milestone.
