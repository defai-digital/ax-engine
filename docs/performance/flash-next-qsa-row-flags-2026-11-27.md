# Flash Next QSA: the prefill row-flag scan in parallel

**Date:** 2026-11-27 · **Hosts:** Mac Studio M2 Ultra 192 GB and Mac mini
M4 Pro 64 GB, pinned MLX 0.32.4 · **Scope:** kernel timing only; the
MacBook Pro M5 Max was not available, so no first-text figure is claimed.
· **Raw output:**
[`benchmarks/results/flash-next-qsa-row-flags-20261127/`](../../benchmarks/results/flash-next-qsa-row-flags-20261127/README.md)

## What it is

At prefill, Flash Next's sparse attention layers (12 of 48) score every
complete four-token block of the history for every query row and keep the
best 512. The device selection
([`qwen4_exp_qsa.rs`](../../crates/ax-engine-mlx/src/qwen4_exp_qsa.rs))
builds one sort key per row and block in a Metal kernel, and the host reads
back one number: whether any row needs the host sort instead. A row needs
it when one of its causal scores is non-finite (the host sort reports that
as an error) or when it has fewer than 512 finite causal blocks.

That check was a loop in the same kernel: the thread of block 0 of each
row read every causal score of its row, one after another. At 13,023
tokens a row has up to 3,256 blocks, so every chunk waited on 2,048
threads each reading 3,256 scores serially.

## What changed

- **A separate row-flag kernel.** One simdgroup per row reads the causal
  scores 32 at a time (coalesced) and combines them with `simd_max`. The
  rule is the same: with no non-finite causal score, every causal block is
  finite, so "fewer than 512 finite causal blocks" is "fewer than 512
  causal blocks".
- **Short rows are decided on the host.** The host already knows each
  row's causal block count. A chunk with a row below 512 always went to the
  host sort; it now goes there without launching the kernels and without
  the read-back.
- The block-key kernel lost its flags output and is renamed
  `ax_qsa_prefill_block_key_v4`; the new kernel is
  `ax_qsa_prefill_row_flag_v1`. Both are compiled at model load, as
  before.

The sort keys, the row flags and therefore the selected blocks are
unchanged. No switch was added: the result is bit-identical, and
`AX_MLX_FLASH_NEXT_QSA_PREFILL_DEVICE_SELECT=0` still restores the host
sort.

## Kernel timing

2,048 query rows, random scores, row `q` with `offset + (q + 1) / 4`
causal blocks (the shape of a 2,048-token chunk), 30 evaluations each.
Times include about 0.2 ms of fixed cost per evaluation.

| Blocks per row | M2 Ultra before | M2 Ultra after | M4 Pro before | M4 Pro after |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.56 ms | 0.28 ms | 0.87 ms | 0.30 ms |
| 2,112 | 0.89 ms | 0.31 ms | 0.93 ms | 0.41 ms |
| 3,256 | 1.30 ms | 0.38 ms | 1.24 ms | 0.58 ms |

"Before" is the old block-key kernel, "after" the block-key kernel
without the scan plus a copy of the row-flag kernel. Keys and flags were compared bit for bit in
every case, including NaN, NaN payloads, ±inf, subnormals, non-finite
scores past the causal limit, and rows short of 512 blocks.

Over the 12 sparse layers this saves about 6 to 11 ms per 2,048-token chunk
on these hosts at 2,000 to 3,256 blocks. The M5 Max profile in the
[tensor-unit note](flash-next-qsa-tensor-unit-2026-11-26.md) puts block
scoring at 8 to 17 ms per chunk at 13,023 tokens, so the scan was likely a
large part of it there too, but that has to be measured on the M5 Max.

## Tests

- `prefill_row_flags_match_the_serial_scan`: the new kernel against a host
  copy of the old serial rule over five shapes (3 to 300 blocks, batch 1
  and 2, block counts below, at and above one simdgroup).
- `prefill_selection_sends_short_rows_to_the_host_without_a_read_back`:
  a chunk with a short row uses the host sort and adds no read-back.
- The existing prefill selection tests against the host sort pass
  unchanged (`cargo test -p ax-engine-mlx --lib qwen4_exp_qsa`, 73 tests).
- The whole `ax-engine-mlx` suite passes on the M2 Ultra and the M4 Pro
  (2,357 tests each).

## Not done

- First text by input length on the MacBook Pro M5 Max, with rest before
  each arm, against the previous build and oMLX.
- The partial sort that picks 512 of up to 3,256 blocks
  (`argpartition`) is unchanged and is now the larger part of the
  selection at long context.
