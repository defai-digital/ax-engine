# F3 M3B — Disk Prefix Cache Cross-Process Locking (2026-05-14)

PRD: `MLX-DISK-PREFIX-CACHE-PRD-2026-05-14.md`
Prior findings:
- `MLX-F3-DISK-PREFIX-CACHE-M3-FINDINGS-2026-05-14.md`
- `MLX-F3-DISK-PREFIX-CACHE-M4-FINDINGS-2026-05-14.md`

## 1. Status: **M3B LANDED.** The runtime lock primitive is now implemented.

This slice closes the M3 runtime gap that was intentionally deferred
in the original M3 findings: mutating disk-prefix-cache operations
now serialize across processes through a directory-level advisory
lock.

## 2. Runtime contract

- `DiskPrefixCache::insert` takes an exclusive lock on
  `.axkv.lock`, writes the temp file, fsyncs, atomically renames, and
  runs eviction while still holding that lock.
- `DiskPrefixCache::evict_until_within_policy` takes the same lock
  before walking and deleting entries.
- `DiskPrefixCache::get` and `contains` remain lock-free. Readers rely
  on atomic rename plus payload SHA256 / embedded-key validation, so
  a partially written or corrupt entry still fails closed as a miss.
- The lock file is not a cache entry and is ignored by the `.axkv`
  eviction sweep.

This is intentionally a local-filesystem advisory lock. It does not
make a distributed-cache promise for NFS/SMB semantics.

## 3. Tests

- `cargo test -p ax-engine-mlx --lib disk_prefix_cache --quiet`
  passed: 15 / 15 disk-prefix-cache tests.
- New regression: `lock_file_is_ignored_by_eviction_budget` ensures a
  tight `max_entries=1` policy does not evict the only real entry just
  because the sentinel lock file exists.

## 4. Remaining validation

This slice landed the primitive. Follow-up stress evidence now exists
in `MLX-F3-DISK-PREFIX-CACHE-M3C-STRESS-FINDINGS-2026-05-14.md`.
Remaining blockers before F3 closure:

- full AX-serving soak / promotion decision;
- M5 runtime/docs promotion review.
