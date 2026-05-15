# KV Cache and Memory System

AX Engine uses a two-layer KV cache architecture. The layers have separate
responsibilities and communicate through scheduling signals, not shared state.

## Architecture Overview

```
ax-engine-core (scheduling / logical)
┌──────────────────────────────────────────────────────────┐
│  KvManager                                               │
│  ├── BlockTable per request (logical block IDs)          │
│  ├── block_ref_counts (cross-request prefix sharing)     │
│  └── cached_blocks (evictable prefix cache)              │
│        FNV-1a-chained hash keys, parent-linked tree      │
└──────────────────────────────────────────────────────────┘
             ↓  prefix hit signals + reused_block_count
ax-engine-mlx (GPU / physical)
┌──────────────────────────────────────────────────────────┐
│  MLX Runner                                              │
│  ├── prefix_cache: Mutex<LruCache<hash, CacheSnapshot>>  │
│  │     MlxKVCache snapshot per (model, block_size, len)  │
│  └── per-request: Mutex<HashMap<RequestId, RequestState>>│
│        └── MlxKVCache (GPU backing buffers)              │
│            ├── LayerKV    (standard / Gemma4 sliding)    │
│            ├── GlmMlaLayerCache  (latent + rope_key)     │
│            ├── LinearLayerState  (conv + recurrent)      │
│            └── TurboQuantShadowLayerStorage  [experiment]│
└──────────────────────────────────────────────────────────┘
```

`KvManager` never holds a GPU pointer. `MlxKVCache` never calls the scheduler.
The engine orchestrates them through the step loop in `engine.rs`.

---

## KvManager: Logical Block Ledger

`KvManager` (`crates/ax-engine-core/src/kv.rs`) tracks which tokens belong to
which logical blocks and which blocks are shared or evictable.

### Block lifecycle

```
alloc_blocks(request_id, n)
  → assigns logical block IDs in BlockTable

apply_prefix_reuse(request_id, reused_block_count)
  → increments ref_count on each shared block

free(request_id)
  → promote_prompt_prefix_to_cache()   ← must run BEFORE removing block table
  → dec_ref_count on each block
  → remove BlockTable entry
```

The promotion step runs inside `free()` before the block table is removed.
Order is critical: `lookup_prefix` walks the live block table to check for
concurrent sharing; running promotion after removal would miss active refs.

The same `free()` primitive is used by the scheduler's preempt-and-recompute
path (see `docs/SCHEDULER.md` § Preempt-and-recompute). Preemption frees a
newer in-flight prefill request to make room for an older stalled allocation;
because `free()` promotes the freed prompt prefix into the retained cache,
the preempted request hits its own cached prefix when it is re-admitted, so
recompute is largely refcount bookkeeping rather than re-running the prefill.

`KvManager::block_count_for(request_id)` is a small read-only accessor used
by the scheduler's preemption selector to identify candidates that actually
hold KV blocks. It returns 0 for unknown requests rather than erroring.

### Prefix hash chain

Block hashes use an FNV-1a-like chain:

```rust
hash = (hash ^ parent_block_hash) * FNV_PRIME;
for token in block_tokens {
    hash = (hash ^ token as u64) * FNV_PRIME;
}
hash ^= len as u64;  // redundant when block_size is fixed
```

Each block's hash incorporates its parent's hash, forming a linked key tree.
A child block can only be a cache hit if every ancestor also matches. This
prevents incorrect prefix sharing when two prompts diverge mid-sequence.

Hash keys are lookup accelerators, not the only correctness check. Retained
cache entries also store the full token payload for each cached block; lookup
and promotion verify that payload before treating a hash match as reusable. If
a parent hash collision is detected, promotion stops and descendant entries are
not refreshed.

### Prefix lookup order

`lookup_prefix` checks two sources in order:

1. **Live prefix**: another active request's block table shares the same
   leading tokens. Ref-count is incremented; no GPU work needed.
2. **Cached prefix**: a block retained in `cached_blocks` after its request
   completed. A cache hit signals the MLX runner to restore GPU state from its
   own snapshot cache.

### Eviction policy

`select_cached_block_eviction_candidate` uses a four-tier O(N) scan:

1. Leaf block (no cached descendants) with no live block sharing its physical
   buffer — evict this first.
2. Leaf block with live physical sharing — still safe to evict logical entry.
3. Non-leaf block with no sharing.
4. Any non-leaf block.

The `has_cached_descendant` map is rebuilt on every call. Under high eviction
pressure this is an O(N) pass per eviction event; acceptable at current
single-user batch sizes but worth revisiting if the retained cache grows large.

Evicting a non-leaf block cascades: all descendants are also evicted, because
their hash keys embed ancestor hashes.

---

## MlxKVCache: Per-Request GPU Buffers

`MlxKVCache` (`crates/ax-engine-mlx/src/kv_cache.rs`) holds the actual Metal
buffers for a single request's key-value state.

### Buffer growth strategy

Buffers are pre-allocated in 256-token chunks (`KV_CHUNK_TOKENS = 256`):

```
capacity = ceil(seq_len / 256) * 256
```

`append()` compares `write_end` against the current capacity. When growth is
needed:

```rust
let new_buf = zeros([1, n_kv_heads, new_capacity, head_dim]);
let new_buf = slice_update(new_buf, &old_buf, [0,0,0,0]);
```

Without chunking, every token append would call `slice_update` and extend the
MLX lazy graph by one node. Over a 2048-token prefill that produces an
O(N²) computation graph. Chunked pre-allocation reduces graph depth to
O(N / 256) and amortizes Metal buffer allocation overhead.

### Lazy graph materialization

MLX operations are lazy: they return node handles in a computation graph.
The graph is only executed when `eval()` is called.

After each decode step, `collect_eval_refs()` gathers references to all
stateful arrays (K and V buffers, GLM latent and rope-key buffers, linear
conv and recurrent state arrays). The runner evaluates them alongside the
output token in a single `eval()` call:

```rust
let mut eval_refs = cache.collect_eval_refs();
eval_refs.push(&output_token);
mlx_eval(&eval_refs);
```

This flattens the graph after every step. Without it, deferred nodes
accumulate and the graph depth grows linearly with sequence length, causing
an O(N²) evaluation cost over long sequences.

When `temperature > 0`, logits are added to the same `eval()` call so
temperature sampling and KV materialization share one GPU dispatch.

### Sequence length management

`trim_to(prefix_len)` resets `seq_len` without touching GPU buffers:

```rust
self.seq_len = prefix_len.min(self.seq_len);
```

This is O(1) and enables speculative decode rollback: if n-gram speculation
is rejected, the cache is trimmed back to the accepted position and decode
resumes from there.

**Limitation**: `trim_to` does not reset `linear_layers`. Linear attention
recurrent state (Qwen3.5 GatedDeltaNet) is sequential and cannot be rolled
back. The speculative decode retry interval is adjusted for linear-attention
requests to compensate.

---

## Prefix Caching: Two-Cache Coordination

Two caches must agree for a prefix hit to save GPU work.

### KvManager cache (scheduling signal)

`KvManager.cached_blocks` tracks which prefix hashes were retained after
request completion. When `lookup_prefix` finds a hit, it returns a
`reused_block_count` signal that engine.rs passes to the scheduler and
then to the MLX runner via `ApplyPrefixReuseItem`.

### MLX runner prefix_cache (GPU state snapshot)

The runner maintains its own `Mutex<LruCache<PrefixCacheKey, CacheSnapshot>>`.
When a request completes, the runner saves a snapshot of its `MlxKVCache`.

On a prefix hit, `restore_reused_prefix_state` checks this cache:

```rust
if let Some(snapshot) = self.prefix_cache.lock().unwrap().get(&key) {
    state.cache = snapshot.cache.clone();  // CoW: shares Metal buffer
    state.prompt_prefix_tokens = reused_tokens.to_vec();
    return;
}
// Miss: full re-prefill of the shared prefix tokens
self.warm_reused_prefix_without_cache(state, ...);
```

### MLX copy-on-write semantics

`MlxArray::clone()` is reference-counted, not a buffer copy. Both the
snapshot and the new request's cache share the same Metal buffer until one
of them calls `slice_update` (which triggers divergence). A prefix cache hit
therefore costs only a ref-count increment for the shared prefix portion.

When the new request's decode phase appends beyond the shared prefix,
`slice_update` copies and diverges — subsequent writes are isolated.

### What happens on a KvManager hit but runner miss

If `KvManager` signals a prefix hit but the runner's `prefix_cache` has no
snapshot (LRU eviction, first-time warm-up, or restart), `restore_reused_prefix_state`
calls `warm_reused_prefix_without_cache`: a full re-prefill of just the
shared prefix tokens. The request still skips sampling for those tokens
(KvManager has already allocated them), but no GPU state shortcut is taken.

MLX runner route metadata keeps that distinction visible:

| Key | Meaning |
|---|---|
| `ax_mlx_prefix_cache_hits` | Physical prefix snapshot restored |
| `ax_mlx_prefix_cache_misses` | Cache was eligible but no snapshot was present |
| `ax_mlx_prefix_cache_blocked` | Aggregate count for blocked snapshot restore/store |
| `ax_mlx_prefix_cache_blocked_policy_disabled` | Prefix cache policy disabled by size/count limits |
| `ax_mlx_prefix_cache_blocked_unsupported_layout` | Model cache layout cannot be snapshot-restored safely |
| `ax_mlx_prefix_cache_blocked_trim_failure` | Snapshot trim failed while storing a prefix |
| `ax_mlx_prefix_cache_warmup_tokens` | Reused prefix tokens re-prefilled by the runner |
| `ax_mlx_prefix_cache_reused_tokens` | Tokens restored from a physical snapshot |
| `ax_mlx_prefix_cache_stores` | Prefix snapshots stored after request progress |
| `ax_mlx_prefix_cache_evictions` | Prefix snapshots evicted by the runner LRU policy |
| `ax_mlx_prefix_cache_entries` | Current runner prefix snapshot entries |
| `ax_mlx_prefix_cache_bytes_kib` | Current runner prefix snapshot footprint |

## Disk-Durable Prefix Cache (Opt-In)

The MLX runner can add a second, file-backed prefix-cache tier below the
in-memory `prefix_cache`. This is disabled by default. When enabled, the runner
stores serialized `MlxKVCache` snapshots as `.axkv` files so a process restart
or another local process can restore a validated prompt prefix without repeating
the full cold prefill.

The disk layer is additive:

1. The scheduler still discovers logical prefix reuse through `KvManager`.
2. The runner first checks the in-memory snapshot cache.
3. If L1 misses and `AX_MLX_PREFIX_CACHE_DIR` is set, the runner probes the
   `.axkv` store.
4. A disk hit restores the KV snapshot, repopulates L1, and records disk-cache
   telemetry.
5. A miss, corrupt entry, version mismatch, unsupported layout, or disabled disk
   layer falls back to the existing L1-only or warmup path.

### Configuration

| Environment variable | Default | Meaning |
|---|---:|---|
| `AX_MLX_PREFIX_CACHE_DIR` | unset | Enables the disk tier and selects the local cache directory. Empty or unset disables the disk tier. |
| `AX_MLX_PREFIX_CACHE_DISK_DISABLED` | unset | Kill switch. When truthy, disables the disk tier even if `AX_MLX_PREFIX_CACHE_DIR` is set. |
| `AX_MLX_PREFIX_CACHE_DISK_MAX_BYTES` | 8 GiB | Post-insert byte budget. Invalid, blank, or zero values fall back to the default. |
| `AX_MLX_PREFIX_CACHE_DISK_MAX_ENTRIES` | 1024 | Post-insert entry budget. Invalid, blank, or zero values fall back to the default. |

The cache assumes a local filesystem. Writers use a directory-level advisory
lock on `.axkv.lock`; readers remain lock-free and rely on atomic rename plus
payload checksum validation. Network filesystems may have different advisory
lock and rename semantics, so do not treat this as a distributed cache.

### File and Integrity Contract

Each entry is named by the SHA256 of the canonical prefix-cache key and uses the
`.axkv` extension. The file stores:

- `AXKV` magic and a format version;
- payload SHA256;
- serialized canonical key bytes for collision detection;
- optional greedy prefill output token for full-prompt restores;
- serialized `MlxKVCache` payload.

Readers fail closed. If the file is truncated, has a stale version, mismatched
key bytes, or a checksum mismatch, the lookup is treated as a miss. The runner
must never continue with a partially restored KV state.

### Disk Telemetry

When the disk tier is active, route metadata can include:

| Key | Meaning |
|---|---|
| `ax_mlx_prefix_cache_disk_hits` | Disk entry restored successfully. |
| `ax_mlx_prefix_cache_disk_misses` | Disk tier was eligible but no valid entry was restored. |
| `ax_mlx_prefix_cache_disk_inserts` | Disk entries written after a prompt-prefix snapshot store. |
| `ax_mlx_prefix_cache_disk_insert_bytes_kib` | Serialized disk-entry bytes written, rounded to KiB. |
| `ax_mlx_prefix_cache_disk_evictions` | Files removed by post-insert eviction. |

These counters are separate from the in-memory `ax_mlx_prefix_cache_hits` /
`misses` counters. A request can miss L1 and hit L2; that is the expected
cross-restart path.

### Validation Evidence

Current checked-in disk-prefix-cache evidence:

| Artifact | Coverage | Result |
|---|---|---|
| `benchmarks/results/disk-prefix-cache-cross-restart/gemma4-e2b-2026-05-14.json` | Gemma 4 E2B, standard FA + sliding window | PASS, 2/2 token-exact, 2 phase-B disk hits |
| `benchmarks/results/disk-prefix-cache-cross-restart/qwen35-9b-2026-05-14.json` | Qwen3.5-9B, hybrid MLA + linear attention | PASS, 2/2 token-exact, 2 phase-B disk hits |
| `benchmarks/results/disk-prefix-cache-cross-restart/glm47-flash-2026-05-14.json` | GLM-4.7-Flash, pure MLA | PASS, 2/2 token-exact, 2 phase-B disk hits |
| `benchmarks/results/disk-prefix-cache-stress/2026-05-14-m3b-stress.json` | 4 worker processes over overlapping keys plus `max_entries=2` eviction pressure | PASS, zero corruption load failures, zero read misses, 3 evictions |

Reusable commands:

```bash
PYTHONPATH=python python3 scripts/verify_disk_prefix_cache_cross_restart.py \
  --model-id gemma-4-e2b-it-4bit \
  --mlx-artifacts-dir .internal/models/gemma-4-e2b-it-4bit \
  --output benchmarks/results/disk-prefix-cache-cross-restart/gemma4-e2b-2026-05-14.json

cargo run -p ax-engine-mlx --bin disk-prefix-cache-stress -- \
  --output benchmarks/results/disk-prefix-cache-stress/2026-05-14-m3b-stress.json
```

The stress artifact validates the cache primitive, not a long-running
multi-user AX server soak. Keep production-serving claims behind a separate
serving artifact with request latency, queueing, memory pressure, and model
route telemetry.

### Snapshot path support matrix

`restore_reused_prefix_state` and `store_prompt_prefix_snapshots` gate the
physical snapshot path by architecture and request mode. Coverage as of
ADR 0018 Strategies 1–3 + the slice 8 MLA fix:

| Architecture        | Models                  | warm_repeat (Decode)        | warm_extend (Prefill)                      | store constraint           |
|---------------------|-------------------------|-----------------------------|--------------------------------------------|----------------------------|
| Standard FA         | (none in current tier)  | snapshot path               | snapshot path                              | every block boundary       |
| Linear attention    | Qwen3.5, Qwen3.6        | snapshot path               | snapshot path                              | full prefix, block-aligned |
| Sliding window      | Gemma 4 E2B             | snapshot path               | snapshot path                              | full prefix, block-aligned |
| MLA                 | GLM-4.7-Flash           | snapshot path               | **blocked → full recompute fallback**      | full prefix, block-aligned |

Two store-side constraints apply to all non-FA architectures:

- **Full prefix only.** Intermediate-block snapshots are unsound because
  `MlxKVCache::trim_to` cannot roll back per-layer recurrent state
  (linear), nor a rotated sliding window (sliding-window). Stores fall
  back to `record_blocked_trim_failure` when these are attempted.
- **Block-aligned prompts only.** When `available_tokens %
  block_size_tokens != 0`, the store is skipped (also as
  `blocked_trim_failure`) because `trim_to(full_block_tokens)` would
  otherwise leave `(seq_len, recurrent_state)` inconsistent.

The MLA + Prefill block is a different constraint: the snapshot CAN be
stored and the lookup CAN match, but the runner refuses the restore
because `chunked_prefill` extending from `seq_len > 0` produces fp drift
across chunk boundaries on MLA's compressed-latent forward. The
`full_prefill_recompute_tokens_for_warmup_fallback` path bit-exactly
re-prefills the matched prefix as new work. warm_repeat (Decode-mode,
zero suffix tokens) is unaffected on MLA.

### Reading the telemetry — operator cheat sheet

For a single request, the (hits, misses, warmup_tokens,
blocked_*) tuple identifies which code path served the warm reuse:

| Observed                                                                 | Path taken                                                | Action |
|--------------------------------------------------------------------------|-----------------------------------------------------------|--------|
| `hits ≥ 1, misses = 0, warmup_tokens = 0, blocked = 0`                   | Snapshot restored (best case)                             | none — working as designed |
| `hits = 0, misses ≥ 1, warmup_tokens > 0, blocked = 0`                   | Snapshot was eligible but absent (LRU evicted or first encounter); warmup path replayed | normal in cold starts; investigate if persistent on hot prompts (cache may be undersized) |
| `hits = 0, blocked ≥ 1, blocked_unsupported_layout ≥ 1, warmup_tokens > 0` | Architecture-restricted: MLA + Prefill, or model not yet supported. cee4227e's full-recompute path took over | expected on MLA `warm_extend`; otherwise look for an architecture-specific snapshot strategy |
| `hits = 0, blocked ≥ 1, blocked_policy_disabled ≥ 1`                     | Prefix cache disabled by size/count policy                | check policy config; either intentional or undersized cache |
| `hits = 0, blocked ≥ 1, blocked_trim_failure ≥ 1, stores = 0`             | Linear / sliding / MLA + prompt not block-aligned; store skipped | normal — block-aligned prompts are the only safe path here |
| `hits = 0, misses = 0, warmup_tokens = 0, blocked = 0`                   | KvManager did not signal a logical prefix hit — engine had no matched prefix to forward to the runner | check `prefix_reused_blocks` (engine-level) — likely first encounter, or prompt is shorter than `block_size_tokens` (16) |

`prefix_cache_path` in route metadata mirrors the same outcome with a
single string: `retained_prompt_prefix_cache` for snapshot hits,
`metadata_lookup` for everything else. Most operators only need the
string; the cheat sheet is for triaging unexpected `metadata_lookup`
results when a hit was expected.

### Validated hot-prefix evidence

The current positive physical-reuse artifact is the Qwen3.5 9B warm-repeat
equivalence run:

- Artifact:
  `benchmarks/results/mlx-inference/2026-05-13-hot-prefix-w2/equivalence-gate/warm_repeat/qwen3-5-9b-2026-05-13.json`
- Schema: `ax.prefix_reuse_equivalence.v1`
- Model: `Qwen3.5-9B-MLX-4bit`
- Result: PASS, 5/5 prompts matched token-exactly.
- Warm physical snapshot hits: 5.
- Warm physical reused tokens: 176.
- Warmup tokens on the claimed hit path: 0.
- Warm misses: 0.

This supports a positive product claim that AX Engine restores physical MLX
prefix snapshots on the Qwen warm-repeat path. It does not claim a long-prompt
TTFT win by itself: the equivalence corpus is intentionally short and
correctness-focused. Longer hot-prefix TTFT claims need a separate latency
artifact with the same physical-hit counters.

### Correctness regression gate

Any change that loosens `prefix_cache_supported()`, modifies
`store_prompt_prefix_snapshots`, or alters `restore_reused_prefix_state`
must keep this gate green:

```
AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR=<model> \
  bash scripts/check-prefix-reuse-equivalence.sh
```

The wrapper builds a fresh `maturin develop` venv and runs
`scripts/verify_prefix_reuse_equivalence.py` in BOTH `--mode warm_repeat`
AND `--mode warm_extend` with `--pad-to-block-size 16`. Both must
produce 5/5 token-exact matches between a cold baseline and the warm
path. Exit 0 on full PASS, exit 3 on any divergence.

The gate is wired into `.github/workflows/ci.yml` as
`Run prefix-reuse equivalence regression gate`, gated on the same MLX
artifacts availability step as the other model-dependent smoke checks.
Rotating the mounted model across Qwen / Gemma / GLM rotates the
covered architecture (linear / sliding-window / MLA) automatically.

---

## Model-Specific Cache Variants

### Standard attention (most models)

`LayerKV`: full buffer `[1, n_kv_heads, capacity, head_dim]` with
`last_k_view` / `last_v_view` cached after each append to avoid duplicate
slice kernel dispatch on the same decode step.

### Gemma4 rotating sliding window

`append_rotating_retained_window` writes circularly into a window-sized buffer.
On the first decode call that exceeds the window size, the buffer is shrunk to
`[1, n_kv_heads, window_size, head_dim]` and subsequent writes wrap around.

`trim_to` refuses rollback when `rotating_window` is active — circular state
cannot be trivially restored to an earlier position.

Gemma4 also has KV-shared layers (layers 24–41) that reuse the previous
layer's K and V. `peek_source_kv()` returns `last_k_view` / `last_v_view`
from the source layer's `LayerKV`, avoiding a duplicate slice kernel dispatch
for each shared layer (~0.5 ms/step at 42 shared layers).

### GLM MLA (compressed cross-attention)

`GlmMlaLayerCache` stores:
- `kv_latent`: `[1, 1, T, latent_dim]`
- `k_pe`: `[1, 1, T, rope_dim]`

Standard MHA stores `[1, n_kv_heads, T, head_dim]` per layer. For a typical
128-head, 128-dim model with latent_dim=512 and rope_dim=64, MLA achieves
roughly 14× compression in KV memory.

`append_glm_mla` grows both buffers with the same 256-token chunk strategy.

### Linear attention (Qwen3.5 / Qwen3-Next GatedDeltaNet)

`LinearLayerState` holds `conv_state` and `recurrent_state` for each linear
layer. These are sequential: each token's output feeds the next step's input.

`collect_eval_refs` includes these arrays so they are materialized every step.
`trim_to` deliberately does not reset them — there is no correct way to restore
an intermediate recurrent state from a token-index boundary alone.

---

## TurboQuant Compression (Experimental)

`TurboQuantShadowLayerStorage` holds a CPU-side compressed copy of cold tokens
for a subset of layers. Cold tokens are those beyond a recency window;
hot tokens remain in GPU memory uncompressed.

Supported presets:
- `K8V4`: 8-bit keys, 4-bit values (default research target)
- `K4V4`: 4-bit keys, 4-bit values (aggressive)
- `K3V4Research`: 3-bit keys, 4-bit values (research only)

Sync is triggered every `KV_CHUNK_TOKENS` cold token advances via
`sync_turboquant_shadow_storage`, which performs a blocking GPU→CPU download
followed by CPU-side quantization. This is a synchronous operation that stalls
the step loop for the duration of the transfer.

TurboQuant is gated behind `TurboQuantProductionRequirements` and is not yet
wired into the default inference path. A fused decode kernel that reads
compressed cold K/V in-place on GPU (avoiding the CPU roundtrip) is the
primary prerequisite for production use.

---

## Memory Pressure and Scheduling Interaction

`KvManager` surfaces memory pressure in two ways:

- `memory_pressure()` reports `kv_low_free_blocks:<free>/<total>` when free
  blocks drop below the low-water mark.
- `memory_pressure()` reports `kv_exhausted_reclaimable_cache` when no free
  blocks remain but retained prefix cache can be evicted.
- `memory_pressure()` reports `kv_exhausted` when no free blocks remain and no
  retained cache can be reclaimed.
- `allocate()` returns `InsufficientCapacity` if a concrete scheduled item still
  cannot obtain the blocks it needs.

`Scheduler.plan()` reads `SchedulerInput.memory_pressure`:

- `kv_low_free_blocks:*` preserves decode progress and caps new prefill to
  `MEMORY_PRESSURE_MAX_PREFILL_TOKENS_PER_STEP` token per step.
- `kv_exhausted_reclaimable_cache` preserves decode progress and caps new
  prefill to one token per step so concrete allocation can trigger retained-cache
  eviction.
- `kv_exhausted` preserves decode progress but defers new prefill work.

This is a front-line throttle, not a complete admission guarantee: a decode or
capped prefill item can still fail concrete KV allocation if the current block
table shape needs more blocks than are available.

When `allocate()` returns `InsufficientCapacity`, the engine in `engine.rs`:

1. Rolls back any prefix sharing that was applied this step.
2. Rolls back any prefix reuse.
3. Marks all newly blocked requests as `BlockedOnMemory`.
4. Calls `scheduler.plan()` a third time, excluding blocked requests.

The practical effect is that the engine may call `scheduler.plan()` up to
three times per step under memory pressure: initial plan → prefix-reuse
re-plan → fallback re-plan after KV allocation failure. The fallback remains
important because scheduler pressure policy is conservative and does not inspect
per-request block-table allocation details.

---

## Key Invariants

- `KvManager.free()` calls `promote_prompt_prefix_to_cache()` before removing
  the block table entry. Reversing this order breaks prefix lookup.
- `MlxKVCache.trim_to()` does not reset `linear_layers`. Callers must not
  expect recurrent state to match any earlier checkpoint.
- `collect_eval_refs()` + `eval()` must run after every decode step.
  Skipping this allows the MLX lazy graph to grow unboundedly.
- Evicting a cached block in `KvManager` cascade-evicts all descendants.
  Never evict a block without checking `select_cached_block_eviction_candidate`.
- Retained-prefix hash matches must be token-payload validated before reuse.
- The MLX runner `prefix_cache` and `KvManager.cached_blocks` are independent.
  A hit in one does not guarantee a hit in the other.
- Treat `ax_mlx_prefix_cache_misses` differently from
  `ax_mlx_prefix_cache_blocked_*`: misses are eligible physical-cache lookups
  without a snapshot, while blocked counters mean the runner could not safely
  use the snapshot cache path.
