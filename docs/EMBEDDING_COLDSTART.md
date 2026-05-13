# Embedding cold-start: AX_MMAP_WEIGHTS measurement guide

`AX_MMAP_WEIGHTS=1` selects a Rust-side memory-mapped safetensors
loader for the MLX runner. The warm-cache numbers in the README's
"Cold start (serverless / scale-to-zero)" section are useful but
understate the win because the OS page cache supplies most of the
bytes on the second-and-later runs.

This document is the procedure for measuring **true cold start** —
the first load after a fresh boot or after the OS page cache has been
purged. Capture true-cold numbers before deciding to flip
`AX_MMAP_WEIGHTS` to default-on.

## What the loaders actually do

| Path | Pipeline |
|---|---|
| Default (C loader) | `open()` → MLX builds a userspace `read()` plan → reads file into MLX-owned heap buffers → eval |
| `AX_MMAP_WEIGHTS=1` | `open()` + `mmap()` of the safetensors file → Rust parses JSON header → `mlx_array_new_data` copies each tensor from the mapped region into MLX-owned buffer → eval |

Both end with the same MLX-owned in-memory state. The mmap path skips
the userspace heap buffer + `read()` round-trip; the OS supplies pages
directly to the destination via the unified-memory page cache.

## Procedure for true-cold measurement

1. **Build the cold-start bench example.**

   ```bash
   cargo build -p ax-engine-bench --example cold_start_bench --release
   ```

2. **Drop the OS page cache for the model file.**

   macOS:
   ```bash
   sudo purge
   ```
   `sudo purge` drops the entire disk cache. It takes ~1–3 seconds and
   does not affect anything other than I/O latency for the next read.

   Linux (if porting): `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`.

3. **Run the bench with the C loader.**

   ```bash
   AX_MMAP_WEIGHTS=0 ./target/release/examples/cold_start_bench \
       --model-dir .internal/models/qwen3-embedding-8b-4bit-dwq
   ```

   Record `session_new_ms`.

4. **`sudo purge` again.** Critical — otherwise the second loader sees
   the pages already cached.

5. **Run the bench with the mmap loader.**

   ```bash
   AX_MMAP_WEIGHTS=1 ./target/release/examples/cold_start_bench \
       --model-dir .internal/models/qwen3-embedding-8b-4bit-dwq
   ```

   Record `session_new_ms`.

6. **Repeat at least three times alternating C / mmap with `sudo purge`
   between every run.** Cold-disk reads are sensitive to PCIe / NVMe
   queuing; a single run is not enough to claim a winner.

## Expected outcome and decision rule

Two measurement runs across two days produced **conflicting warm-cache
numbers**, which is why the env var is still opt-in:

2026-05-12 run (`benchmarks/results/embedding/2026-05-12-r4-mmap-loader/`):
the mmap loader was faster (`-11% / -30% / -41%` on `0.6B / 4B / 8B`).

2026-05-13 run (`benchmarks/results/embedding/2026-05-13-readme/`):
the mmap loader was **slower** (`+32% / +78% / +106%` on the same three
models).

The discrepancy is most likely measurement-order dependent: when the C
loader runs first and primes the OS page cache, the mmap loader inherits
the hot cache and the absolute work it adds (Rust-side JSON header parse
+ per-tensor `mlx_array_new_data` copy) is what's measured. When the
mmap loader runs first, MLX's parallel-`read()` C loader has to compete
with cold disk and looks worse.

The fair comparison is **true-cold for both paths** — drop the page
cache between every run. On Apple Silicon NVMe the C loader's
`ParallelFileReader` is well tuned; the mmap loader's theoretical win
(no userspace heap buffer for the safetensors file) only shows up when
the kernel can issue read-ahead in parallel with MLX consuming the
already-mapped pages.

Cold-disk absolute times will be larger for both paths — expect 1–3
seconds for the 8B model on M-series NVMe.

**Default-on criteria** (when to flip `AX_MMAP_WEIGHTS=1` to the
runtime default):

1. ✅ True-cold mmap loader is at least 20% faster than C loader on the
   8B model.
2. ✅ True-cold mmap loader is not slower than C loader on any model.
3. ✅ Bit-exact embedding output preserved (the 2026-05-12 warm-cache
   measurement already confirmed this; re-verify on cold).
4. ✅ Tested on at least two macOS versions (Sonoma 14.x, Sequoia 15.x).
5. ✅ Server-mode smoke (set env var before `ax-engine-server` start;
   issue `/v1/embeddings` request; compare output to default loader).
   The runner code path is the same as the bench example's, but the
   server lifecycle (axum + tokio + microbatcher) is what users
   actually hit — verify the env var propagates through it.

Until all five criteria are met, keep `AX_MMAP_WEIGHTS` opt-in and
document it under "Cold start (serverless / scale-to-zero)" in the
README. The current setup intentionally leaves the C loader as the
default to avoid an underspecified behaviour change in a Mac-first
runtime that hasn't been stress-tested on every macOS minor version
yet.

## Server-mode smoke

```bash
# Terminal A — server with mmap loader
AX_MMAP_WEIGHTS=1 ./target/release/ax-engine-server \
    --mlx --mlx-model-artifacts-dir .internal/models/qwen3-embedding-0.6b-8bit \
    --port 8083

# Terminal B — single embedding request
curl -sS -X POST http://127.0.0.1:8083/v1/embeddings \
    -H 'Content-Type: application/json' \
    -d '{"input":[1,2,3,4,5,6,7,8,9,10]}' \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d["data"][0]["embedding"]), d["data"][0]["embedding"][:5])'
```

Expect the same vector dimension and the same first-5 floats as the
default-loader run. If the numbers differ, the mmap loader is
mis-reading bytes for that model's specific dtype mix — file a bug
with the model dir + safetensors header dump.
