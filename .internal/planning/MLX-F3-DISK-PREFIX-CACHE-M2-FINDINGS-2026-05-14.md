# F3 M2 — Runner Wire-up & Disk Telemetry (2026-05-14)

PRD: `MLX-DISK-PREFIX-CACHE-PRD-2026-05-14.md`
Parent PRD: `DS4-LEARNINGS-FOLLOWUP-PRD-2026-05-14.md` §6
Prior milestone: `MLX-F3-DISK-PREFIX-CACHE-M1-FINDINGS-2026-05-14.md`

## 1. Status: **M2 LANDED.** M3 (eviction + concurrency) and M4
(integration validation) remain open.

## 2. What landed

### 2.1 Fastpath env accessors

Two new accessors in `crates/ax-engine-mlx/src/fastpath.rs`:

- `prefix_cache_dir() -> Option<PathBuf>` reads
  `AX_MLX_PREFIX_CACHE_DIR`. Empty / unset returns `None`, which keeps
  the L2 layer disabled. Cached via `OnceLock` like other env paths.
- `prefix_cache_disk_disabled()` is the dedicated kill switch
  (`AX_MLX_PREFIX_CACHE_DISK_DISABLED=1`). Operators can disable L2
  without unsetting the cache directory, useful for isolating
  regressions to the L1-only path.

### 2.2 Telemetry counters

Extended `MlxPrefixCacheTelemetry` with four new fields and matching
route-decision keys (`ax_mlx_prefix_cache_disk_*`):

- `disk_hits` — increments when L2 returns a valid payload and the
  restore path proceeds.
- `disk_misses` — increments when L2 returns `None` (no entry,
  on-disk corruption, key collision, deserialise failure) **and** the
  pre-conditions to even check L2 were met (L1 missed, MLA gate not
  engaged). Pure L1 hits do not increment disk_misses; that would
  conflate the two layers.
- `disk_inserts` — increments per successful disk write.
- `disk_insert_bytes` — total bytes written, reported via
  `ax_mlx_prefix_cache_disk_insert_bytes_kib`.

All four counters are wired through `merge_from`,
`append_route_decisions`, and the appropriate `record_*` helpers.

### 2.3 MlxRunner field

`MlxRunner` gains an optional `disk_prefix_cache: Option<DiskPrefixCache>`.
`from_artifacts` opens it when (a) `AX_MLX_PREFIX_CACHE_DIR` resolves
to a non-empty path and (b) the disk-disabled kill switch is off. Open
failures are non-fatal: the runner logs at `WARN` and falls back to
the in-memory L1 only.

### 2.4 L2 restore path

In `restore_reused_prefix_state`, after the L1 cache.get returns
`None` and **before** the L1 miss counter increments, the runner
attempts the L2 lookup. Conditions:

- L1 missed for the canonical reused-tokens key,
- The MLA chunk-alignment safety gate is **not** engaged (a disk-
  borne snapshot has identical fp characteristics to its in-memory
  source; the gate must still apply),
- The disk cache is open.

On hit, the runner deserialises the payload via
`MlxKVCache::try_deserialize_from_bytes` and restores into
`state.cache` exactly like the L1 hit path, just with
`state.cached_prefill_output_token = None` (the on-disk format does
not carry that field — runner recomputes on the first decode step).
Deserialise failures, filesystem errors, and missing entries all
surface as `disk_misses` and let the existing cold-prefill warmup
take over — fail-closed per PRD §G4.

### 2.5 L2 store path

In `store_prompt_prefix_snapshots`, when L1 successfully inserts a
snapshot, the runner additionally serialises the trimmed
`MlxKVCache` and writes it to disk **only for the largest valid
prefix** (`prefix_len == full_block_tokens`). This keeps the cache
directory bounded — without eviction (M3), persisting every per-
block intermediate would write `O(N/block_size)` files per cold
prefill, exploding to multiple gigabytes per request. The largest
snapshot is also the most useful for future hits because shorter
prefixes always derive from it.

Disk-write failures do not back out the L1 insert. L1 alone is still
useful, and disk is strictly additive on top.

## 3. End-to-end smoke (single-process)

`Gemma 4 E2B 4-bit`, 3-turn multi-turn chat with
`AX_MLX_PREFIX_CACHE_DIR=/tmp/ax-disk-cache-m2-smoke`:

| Turn | L1 hits | L1 misses | L1 stores | disk_hits | disk_misses | disk_inserts | disk KiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 (cold) | 1 (warmup) | 0 | 1 | 0 | 0 | **1** | **36,866** |
| 2 | 1 (probe) | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 1 (probe) | 0 | 0 | 0 | 0 | 0 | 0 |

On-disk directory after the run:

```
38728fee...d21b66.axkv  36 MiB   (turn 1, 2048-token snapshot)
a2b79950...88ec47.axkv  4.5 MiB  (harness warmup, 64-token snapshot)
```

Observations:

- The L2 write path fires end-to-end on the cold turn (`disk_inserts=1,
  disk_insert_bytes=36 MiB`). The on-disk file content was produced
  by the M1 `MlxKVCache::serialize_to_bytes` we round-tripped against
  earlier.
- The harness warmup also persisted a snapshot, accounting for the
  second `.axkv` file. This is not surfaced in the JSON `turns`
  array because the warmup runs before `run_turn` starts; the disk
  artifact correctly shows it.
- `disk_hits = 0` across all turns because the in-process L1 hits
  first (via the existing probe fallback added in commit `27ca5904`).
  The disk-hit path is exercised across process restarts (M4), not
  within one process.
- `growth_ratio = 1.023` confirms iterative-chat TTFT is flat,
  matching the existing Phase C result. The L2 layer does not
  perturb L1's behaviour.

Smoke artifact:
`benchmarks/results/kv-long-context/gemma4-f3m2-smoke-2026-05-14.json`.

## 4. What M2 does not yet do (held for M3 / M4)

### 4.1 Eviction (M3)

The disk directory grows unbounded. The PRD §6 names byte and entry
budgets plus LRU; M2 stops short because:

- Implementing eviction needs cross-process file locking
  (`flock` on a sentinel) to avoid evictors racing.
- Eviction policy interacts with the touch-tick file rename pattern
  the PRD §6 design proposes, which is non-trivial.

Operators using M2 today should rotate the cache directory manually
(e.g. via cron) or constrain workloads to a known prompt set until
M3 lands.

### 4.2 Cross-process / cross-restart validation (M4)

The single-process smoke above exercises the L2 **store** path. The
L2 **read** path is exercised by code review and the M1 round-trip
tests, but not yet end-to-end against `verify_prefix_reuse_equivalence.py`.
The M4 protocol per PRD §8.2:

```
Process A: serve prompt P, exit
Process B: serve prompt P (or P + suffix) — disk_hits should fire
```

is gated on M3 (eviction) so the directory does not balloon during
the multi-process test runs.

### 4.3 Only the largest prefix is persisted

PRD §4 names per-block-aligned prefix storage. M2 ships only
"largest-prefix-only" disk writes. This is a known trade-off
(documented above). When M3 adds eviction, persisting every
per-block prefix becomes viable.

## 5. Files

| Path | Change |
|---|---|
| `crates/ax-engine-mlx/src/fastpath.rs` | + `prefix_cache_dir`, + `prefix_cache_disk_disabled` |
| `crates/ax-engine-mlx/src/runner.rs` | + 4 telemetry fields and route-decision keys; + L2 lookup branch in `restore_reused_prefix_state`; + L2 store branch in `store_prompt_prefix_snapshots`; + `disk_prefix_cache` field on `MlxRunner`; init in `from_artifacts` |
| `scripts/profile_kv_multiturn_chat_evidence.py` | + 4 new TELEMETRY_KEYS so disk counters appear in bench JSON |
| `benchmarks/results/kv-long-context/gemma4-f3m2-smoke-2026-05-14.json` | end-to-end smoke evidence |

## 6. Closure conditions for this M2 artifact

- ✅ Telemetry counters land and emit via `append_route_decisions`.
- ✅ L2 lookup path wires after L1 miss, before L1 miss-counter
  increment; honours MLA chunk-alignment gate.
- ✅ L2 store path wires after L1 successful insert, persisting the
  largest-prefix snapshot.
- ✅ `cargo build`, `cargo clippy --all-targets --all-features -- -D warnings`,
  `cargo test -p ax-engine-mlx --lib` (367/367) all green.
- ✅ End-to-end smoke against Gemma 4 E2B confirms `disk_inserts=1`
  and a 36 MiB `.axkv` file persisted to the configured directory.

PRD §6 row in the parent ledger should update from "M1 landed; M2-M5
open" to "M2 landed; M3-M5 open".

---

**Status:** M2 closed. M3 (eviction + concurrency) is the natural
next milestone. M4 (cross-restart validation) is gated on M3.
