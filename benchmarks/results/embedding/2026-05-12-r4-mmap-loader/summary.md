# R4 mmap-backed safetensors loader — cold-start measurements

Captured on 2026-05-12 (warm OS page cache, 3 runs per loader per
model, median reported).

## Throughput / latency

`AX_MMAP_WEIGHTS=1` selects the mmap path; default is the upstream C
loader. The mmap path opens the safetensors file with `memmap2`,
parses the JSON header in Rust, and constructs `MlxArray`s from
mapped slices via `mlx_array_new_data`. The C loader builds an
in-memory buffer and calls `mlx_load_safetensors`.

| Model | Safetensors size | C loader | mmap (R4) | Saved | % |
|---|---:|---:|---:|---:|---:|
| Qwen3-Embedding 0.6B 8-bit | 633 MB | 143 ms | **127 ms** | -16 ms | -11% |
| Qwen3-Embedding 4B 4-bit | 2.2 GB | 197 ms | **139 ms** | -58 ms | -30% |
| Qwen3-Embedding 8B 4-bit DWQ | 4.3 GB | 261 ms | **155 ms** | -106 ms | -41% |

Bigger models gain more, in absolute and relative terms: the per-call
overhead of the C path's file-read scales with the number of tensors
and the cumulative bytes, while the mmap path's parse + slice
construction is constant per tensor and the page-cache hit is free.

## Correctness

Embedding output is **bit-exact** with the C loader for the same input
prompts, last-token pooling, l2-normalized. Verified on 4B 4-bit:

```
C-loader    L2 norm:  1.000002
mmap-loader L2 norm:  1.000002
max |C - mmap| diff:  0.00e+00
```

## Implementation notes

- The MLX C entry actually used is `mlx_array_new_data` (which copies
  the buffer up front), not the borrowed `mlx_array_new_data_managed`.
  Empirically the managed-borrow variant returned NaN-shaped output for
  f16 / quantized tensors despite the source bytes being correct in the
  mmap region. Falling back to copy-on-create is still a win because
  the mmap path skips the user-space buffer allocation and `read()`
  pipeline the C loader does; the OS page cache supplies the bytes
  directly.
- The mmap is held alive across the entire `load_safetensors_mmap`
  call, then dropped at function return — by that point MLX owns its
  own buffer for every tensor.
- The `from_managed_data` safe wrapper and its destructor are exported
  for future callers (e.g. when a borrowed-data path becomes viable
  via a different MLX entry).

## Reproducing

```bash
cargo build -p ax-engine-bench --example cold_start_bench --release
AX_MMAP_WEIGHTS=0 ./target/release/examples/cold_start_bench \
    --model-dir .internal/models/qwen3-embedding-8b-4bit-dwq
AX_MMAP_WEIGHTS=1 ./target/release/examples/cold_start_bench \
    --model-dir .internal/models/qwen3-embedding-8b-4bit-dwq
```

True-cold-start numbers (page cache purged via `sudo purge` between
runs) will be larger than the warm measurements above. The relative
win of the mmap path should be at least as large on cold disk because
the C loader's userspace `read()` does the heavy lifting, while the
mmap path lets the OS read on demand.
