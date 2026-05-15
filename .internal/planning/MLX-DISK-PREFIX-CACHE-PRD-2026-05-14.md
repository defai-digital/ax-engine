# Disk-Durable Prefix Cache PRD (F3)

Status: Open — design only; implementation not yet scheduled.
Date: 2026-05-14
Owner: AX Engine
Parent PRD: DS4-LEARNINGS-FOLLOWUP-PRD-2026-05-14.md §6 (F3)
Depends on: KV-SCHEDULER-W1-EVIDENCE-REPORT.md, MLX-PHASE-C-MULTITURN-BASELINE-FINDINGS-2026-05-14.md
Reference: DS4 `ds4_server.c` disk-KV section (subtree under `.internal/reference/ds4/`)

## 1. Summary

The in-memory `MlxPrefixCache` in `crates/ax-engine-mlx/src/runner.rs`
delivers the warm-repeat (`28,494×` TTFT on GLM post-fix) and
warm-extend (`7.3×` TTFT on GLM post-fix) wins, but only within one
process lifetime. Server / agent / eval workloads that re-issue the
same system prompt across restarts or across replicas leave that win
on the floor.

This PRD scopes a **durable on-disk prefix cache** that sits beneath
the existing in-memory cache, mirroring DS4's `ds4_server.c` disk-KV
approach but adapted to AX's snapshot model and per-architecture
cache layouts. The goal is **process-restart survival** and
**cross-replica sharing** of the prefix-cache wins, without changing
the in-memory hot-path semantics or telemetry surfaces.

## 2. Goals & Non-Goals

### 2.1 Goals

G1. A second-tier prefix cache that survives process restart. The
runner-side in-memory cache is unchanged; misses there can fall
through to a disk lookup before triggering a full cold prefill.

G2. Cross-replica sharing within one filesystem. Two AX processes
pointing at the same cache directory see each other's stores. No
distributed-cache requirement.

G3. Bounded disk usage with predictable eviction. Configurable
maximum bytes and maximum entries, matching the
`AX_MLX_PREFIX_CACHE_MAX_*` env conventions.

G4. Fail-closed on integrity errors. A truncated or mismatched
payload must miss, not silently corrupt KV state. `verify_prefix_reuse_equivalence.py`
must remain green with the disk layer in front.

G5. Per-architecture coverage: FA, SWA, Linear, MLA. The MLA
chunk-alignment safety established in commit `ade74c2f` must hold
across the disk layer.

### 2.2 Non-Goals

- **Distributed cache / network protocol.** Filesystem-local only;
  shared filesystem (NFS, SMB) is acceptable but the design does
  not assume it.
- **GPU-direct disk load.** Restore loads through the existing
  `MlxArray::from_raw_data` path; bypassing CPU staging is out of
  scope.
- **Compression.** First version stores raw KV tensor bytes.
  Compression is a follow-up if disk usage becomes a constraint.
- **Cross-version compatibility.** A version field is included so
  loads of older formats can be rejected cleanly; backward-compat
  shims are not in scope.

## 3. Architecture

### 3.1 Two-tier structure

```
Request → MlxRunner::restore_reused_prefix_state
              │
              ├── existing in-memory MlxPrefixCache (L1)
              │     hit → restore + return
              │     miss ↓
              │
              └── NEW DiskPrefixCache (L2)
                    hit → load from disk → populate L1 → restore + return
                    miss → continue to full prefill
```

L1 stays exactly as it is today (including the probe added in commit
`27ca5904`). L2 wraps the same key contract.

### 3.2 Key

The on-disk key reuses `MlxPrefixCacheKey` byte-for-byte. Filename =
SHA256 (not SHA1, for collision safety) of the canonical
serialization:

```
fields, in order:
  u32  schema_version   = 1
  u32  block_size_tokens
  u32  token_count
  u64  token_hash
  utf8 model_id          (length-prefixed u16 + bytes)
  utf8 route_policy      (length-prefixed u16 + bytes)
  utf8 layer_layout      (length-prefixed u16 + bytes)
```

The full key (not just its hash) is also written into the file
header so a load can re-validate against the in-memory key. This
catches hash collisions and routing-policy drift.

### 3.3 Payload format

```
file = HEADER + PAYLOAD + FOOTER

HEADER (fixed 64 bytes + variable):
  4   magic                 b"AXKV"
  u32 schema_version        = 1
  u64 payload_byte_count    (offset where PAYLOAD ends)
  u32 architecture_tag      (FA=1, SWA=2, Linear=3, MLA=4)
  u32 layer_count
  u64 created_at_unix_ms
  u64 last_used_unix_ms     (updated by writers via atomic rename)
  u32 hit_count             (informational; not load-blocking)
  u32 reserved              (alignment to 8)
  utf8 key_bytes_len + key_bytes  (re-validation; cf §3.2)

PAYLOAD (variable, architecture-dependent — see §4):
  one or more layer blocks; each block has its own per-layer header
  with shape, dtype, dim metadata, then raw tensor bytes.

FOOTER (32 bytes):
  u64 payload_sha256_8       (first 8 bytes of SHA256(PAYLOAD))
  u64 full_sha256_8          (first 8 bytes of SHA256(HEADER + PAYLOAD))
  u64 reserved
  u32 reserved
  u32 magic_footer           = b"\0AXK"
```

Per-tensor encoding inside PAYLOAD uses a fixed 32-byte tensor
header:

```
TENSOR_HEADER:
  u8   dtype_tag             (matches MlxDtype variant index)
  u8   ndim                  (1..=4)
  u8   reserved[6]
  u32  shape[4]              (zero-padded for ndim < 4)
  u64  byte_count
  u32  layer_idx
  u32  tensor_tag            (per-architecture; see §4)
```

### 3.4 Lookup / store API

A new module `disk_prefix_cache.rs` under `crates/ax-engine-mlx/src/`
exposes:

```rust
pub struct DiskPrefixCache {
    dir: PathBuf,
    policy: DiskPrefixCachePolicy,
}

pub struct DiskPrefixCachePolicy {
    pub max_bytes: u64,
    pub max_entries: u32,
    pub enabled: bool,
}

impl DiskPrefixCache {
    pub fn new(dir: PathBuf, policy: DiskPrefixCachePolicy) -> Result<Self, Error>;
    pub fn get(&self, key: &MlxPrefixCacheKey) -> Option<MlxPrefixSnapshot>;
    pub fn insert(&self, key: &MlxPrefixCacheKey, snapshot: &MlxPrefixSnapshot) -> InsertOutcome;
    pub fn evict_until_within_policy(&self) -> u32;
    pub fn stats(&self) -> DiskPrefixCacheStats;
}
```

`MlxPrefixSnapshot` already wraps a clonable `MlxKVCache`; the disk
cache (de)serializes via the per-architecture functions in §4.

### 3.5 Configuration surface

New env flags following the `AX_MLX_PREFIX_CACHE_*` convention:

- `AX_MLX_PREFIX_CACHE_DIR` — path. Disk cache disabled if unset.
- `AX_MLX_PREFIX_CACHE_DISK_MAX_BYTES` — default `8 * 1024^3` (8 GiB).
- `AX_MLX_PREFIX_CACHE_DISK_MAX_ENTRIES` — default `1024`.
- `AX_MLX_PREFIX_CACHE_DISK_DISABLED` — kill switch.

Wired through `fastpath.rs` with usize / string accessors following
the pattern established in commit `e54cb8fb`.

## 4. Per-architecture serialization

`MlxKVCache` has four shape variants. Each maps to a tensor_tag
subspace.

### 4.1 FA / standard (`architecture_tag = 1`)

Per layer:
- K: `[1, n_kv_heads, seq_len, head_dim]` — tensor_tag = `0`
- V: same shape — tensor_tag = `1`

Trim to `seq_len` before serialization; the on-disk tensor is
exactly the logical KV, no padding. This avoids storing the
overallocated capacity and reduces disk footprint.

Restore: allocate cache with `capacity = chunk_ceiling(seq_len)`,
slice_update the loaded data into position `[0..seq_len]`.

### 4.2 SWA (`architecture_tag = 2`)

Same as FA, plus the rotating-window state. Critical invariant:
**only store a snapshot when no SWA layer has rotated** (i.e.,
`seq_len <= sliding_window` on every windowed layer). Add a check
at store time and skip the disk write if rotation has happened —
the on-disk format simply does not represent a rotated cache, by
design. Mirror the in-memory store's `alignment_restricted` bail.

Per layer:
- K: `[1, n_kv_heads, seq_len, head_dim]` — tensor_tag = `0`
- V: same — tensor_tag = `1`
- (rotating window: implicit; recovered from `cfg` on restore)

### 4.3 Linear (`architecture_tag = 3`)

Per layer:
- conv_state: `[1, conv_kernel - 1, conv_dim]` — tensor_tag = `2`
- recurrent_state: `[1, value_heads, value_dim, key_dim]` — tensor_tag = `3`

Both optional; an empty Linear layer (e.g., the layers that are
full-attention in a hybrid model) writes a tensor_tag = `255`
sentinel record with `byte_count = 0`. This keeps per-layer indexing
positional.

### 4.4 MLA (`architecture_tag = 4`)

Per layer:
- kv_latent: `[1, 1, seq_len, kv_lora_rank]` — tensor_tag = `4`
- k_pe: `[1, 1, seq_len, qk_rope_head_dim]` — tensor_tag = `5`

The MLA chunk-alignment safety (commit `ade74c2f`) is enforced at
*restore* time: after load, the runner sees `cache.seq_len = base_len`
which is guaranteed `block_size_tokens`-aligned because the in-memory
store path enforced that invariant before writing to disk. The
`prefill_chunk = MLA_DEFAULT_PREFILL_CHUNK` default carries over.

### 4.5 TurboQuant shadow

Not serialized in v1. If TurboQuant is active, the disk hit restores
the base KV but the shadow storage rebuilds via the existing
runtime-sync path. The hit telemetry surfaces this as a partial
restore.

## 5. Concurrency

### 5.1 Writers

`DiskPrefixCache::insert` writes to a temporary file in the same
directory, computes the FOOTER hashes, fsync, then atomically
`rename` to the canonical filename `<sha256_hex>.axkv`. This makes
partial writes invisible to readers.

Multiple writers racing on the same key: the loser's rename
overwrites the winner's, which is harmless (same content). No
explicit lock needed.

Multiple writers racing on eviction: each writer that exceeds policy
runs `evict_until_within_policy` after its rename; the eviction
takes a directory-level file lock (POSIX `flock` on a sentinel
`AXKV_LOCK` file). Eviction without the lock skips and logs; the
next writer retries.

### 5.2 Readers

`get` opens the candidate file read-only, parses the HEADER,
verifies the key bytes match exactly, then loads PAYLOAD. Footer
SHA256 prefix is checked at load time; mismatch logs and returns
`None` (the corrupted file is left in place — eviction will pick it
up).

Concurrent readers and writers: a reader that opens a file mid-
rename will see either the old content (succeeds) or fails the open
(returns None). No torn reads because rename is atomic.

### 5.3 mmap

**Not used in v1.** DS4's `ds4_server.c` deliberately avoids mmap
to keep VM mapping count bounded against an already-large weight
mmap. AX's MLX weight loader is the analogous large mmap consumer
(`weights.rs:177`, gated by `AX_MMAP_WEIGHTS`). Using mmap for the
disk cache would multiply the per-process mapping count by the
entry count, risking `ENOMEM` for high-cache-count workloads on
constrained machines.

If a profile shows that read I/O is the dominant restore cost, we
revisit. Until then, plain `read()` into a `Vec<u8>`, then
`MlxArray::from_raw_data` (which takes a `*const u8`).

### 5.4 Cross-host filesystems

Out of scope. Document that the cache assumes local filesystem
semantics (atomic rename, advisory locks). Behavior on NFS / SMB
is undefined; users wanting that need a custom layer.

## 6. Eviction

Two policies, both enforced after every successful insert:

1. **Byte budget.** If `total_bytes > max_bytes`, evict LRU until
   under budget. LRU determined by `last_used_unix_ms` in the file
   header.
2. **Entry budget.** If `entry_count > max_entries`, evict LRU until
   under budget.

A reader that hits an entry updates `last_used_unix_ms` via an
atomic rename of a single-field sidecar `<sha256_hex>.axkv.touch`
file. This avoids re-writing the full payload on every hit.

Tie-break: file mtime (filesystem-supplied, monotonic-ish).

## 7. Integration

### 7.1 Touch points

- `crates/ax-engine-mlx/src/disk_prefix_cache.rs` — new module, ~800 LoC est.
- `crates/ax-engine-mlx/src/runner.rs:restore_reused_prefix_state` — wire L2 lookup after L1 miss.
- `crates/ax-engine-mlx/src/runner.rs:store_prompt_prefix_snapshots` — wire L2 insert when L1 stores.
- `crates/ax-engine-mlx/src/fastpath.rs` — env accessors.
- `crates/mlx-sys/src/array.rs` — possibly add a `data_bytes()` slice accessor if the existing `data_raw()` is not enough (it returns a raw pointer; we need a typed length-bounded view for serialization).

### 7.2 Telemetry

Two new counters in the existing `MlxPrefixCacheTelemetry`:

- `disk_hits: u32`
- `disk_misses: u32`
- `disk_inserts: u32`
- `disk_evictions: u32`
- `disk_bytes_kib: u64`

Surfaced via the same `append_route_decisions` path the in-memory
cache uses. Bench harnesses pick these up automatically once added
to the `TELEMETRY_KEYS` list in the relevant Python scripts.

### 7.3 Public docs

`docs/KV-CACHE.md` and `docs/PERFORMANCE.md` get a paragraph each
describing the disk layer behavior, `AX_MLX_PREFIX_CACHE_DIR` config,
and the perf budget.

## 8. Test plan

### 8.1 Unit (Rust, `cargo test`)

- Round-trip a synthetic FA snapshot through serialize → deserialize,
  assert tensor bit-equality.
- Same for SWA (with and without rotation; rotated should refuse to
  store).
- Same for Linear (with empty layers).
- Same for MLA.
- Atomic-rename simulation: drop a partial file mid-write, verify
  `get` returns None and `evict` picks it up.
- Eviction: insert N entries with known sizes, verify LRU under
  byte budget and entry budget.
- Concurrency: spawn N threads inserting same key; final state has
  exactly one entry.

### 8.2 Integration (Python, existing harnesses)

- `verify_prefix_reuse_equivalence.py --mode warm_repeat`: must
  PASS 5/5 with the disk layer enabled. Two-process variant: process
  A serves the prompt and exits; process B serves the same prompt;
  process B should see `ax_mlx_prefix_cache_disk_hits = 1`.
- `verify_prefix_reuse_equivalence.py --mode warm_extend`: same, on
  GLM-4.7-Flash (regression-check the MLA fix).
- `profile_kv_long_context_evidence.py`: cold → process restart →
  warm. The cross-restart warm must hit the disk cache and reach
  near-zero TTFT.

### 8.3 Stress

- Fill cache to bytes limit (e.g. 8 GiB) with diverse prompts. Verify
  eviction stays bounded and `get` continues to return correct data.
- Spawn 4 concurrent AX processes serving overlapping prompts.
  Verify no corruption, no torn reads, eviction races resolve.

## 9. Telemetry & success criteria

**Quantitative gates for accepting the implementation:**

- New `disk_hits` counter is monotonic across process restarts in
  the cross-restart integration test.
- `verify_prefix_reuse_equivalence.py` token-exact PASS 5/5 on FA,
  SWA, MLA in both warm_repeat and warm_extend modes with the disk
  layer enabled.
- Cross-restart warm: TTFT within 1.5× of in-process warm_repeat
  TTFT (disk load adds some cost; this gate caps it).
- Cache files do not exceed `max_bytes` policy by more than 5% in
  steady state.
- Concurrent four-process stress: zero `disk_corruption_load_failures`
  over a 1-hour run.

## 10. Risks

R1. **Per-architecture serialization correctness.** Four cache
shapes, two of which (SWA, MLA) have safety constraints. Mitigated
by §8.1 unit tests + §8.2 integration tests + reusing the in-memory
store's existing safety predicates.

R2. **Disk IO cost on restore could erode the warm-extend win.** A
2,048-token GLM MLA snapshot is roughly 32 MiB on disk; a sequential
read at typical Mac NVMe speeds (~3 GB/s) takes ~10 ms. Compared to
warm_repeat's ~0.17 ms TTFT, this is a 60× degradation per restore.
Acceptable for the cross-restart case (vs the alternative of a
full cold prefill) but means the disk layer cannot replace the
in-memory layer for in-process repeated hits.

R3. **mmap-vs-read decision** may need to be revisited under
constraint. Currently rules out mmap; if NVMe read becomes the
bottleneck, add an mmap variant under an env flag with explicit
documentation of the VM-mapping cost.

R4. **Schema evolution.** A v2 format must be cleanly rejected by
v1 readers and vice versa. The `schema_version` field carries this;
loaders must validate it and treat mismatches as misses (not
errors).

R5. **Atomic rename semantics on shared filesystems.** Out of scope
in §5.4 but flagged because users will try it. Documentation must
state the local-filesystem assumption.

## 11. Milestones

M1. **Serialization library + unit tests** (2–3 days). **Landed.**
  - `disk_prefix_cache.rs` write / read paths.
  - All four architecture variants round-trip.

M2. **Integration with runner** (1–2 days). **Landed.**
  - L2 wire-up in `restore_reused_prefix_state` and
    `store_prompt_prefix_snapshots`.
  - Telemetry counters land.

M3. **Eviction + locking** (1–2 days). **Partially landed.**
  - Best-effort post-insert eviction under byte/entry budgets is
    implemented and emits `disk_evictions`.
  - Cross-process locking and concurrency stress tests remain open.

M4. **Integration validation** (1 day). **Partially landed.**
  - Gemma 4 E2B cross-restart disk-hit validation passes via
    `scripts/verify_disk_prefix_cache_cross_restart.py`.
  - Broader architecture-tier coverage and four-process stress remain open.

M5. **Docs + PRD closure** (half day).
  - `docs/KV-CACHE.md` and `docs/PERFORMANCE.md` updates.
  - Findings artifact recording final perf numbers.

Total: roughly one engineer-week. Largest risk concentration: M1
(serialization correctness) and M3 (concurrency).

## 12. Closure conditions

This PRD closes when:

1. All §9 quantitative gates PASS, **or**
2. A clear NO-GO finding is recorded (e.g., M1 reveals an MLX
   serialization limitation that defeats the disk-load path) and
   filed alongside this PRD as a `*-FINDINGS-*.md` artifact.

The parent PRD (DS4-LEARNINGS-FOLLOWUP-PRD-2026-05-14.md) updates
its §6 row to "landed" or "NO-GO" based on which closure path
this PRD takes.

---

**Status:** Open — implementation not yet scheduled. Estimated ~1
engineer-week if F4 (MLA bisect tooling) has confirmed the chunk-
alignment workaround holds across the cases we serialize.
