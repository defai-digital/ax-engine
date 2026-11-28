# Flash Next QSA prefill row flags: kernel timing

**Date:** 2026-11-27 · **Hosts:** Mac Studio M2 Ultra 192 GB, Mac mini M4
Pro 64 GB · **MLX:** 0.32.4

Note: [the row-flag scan in parallel](../../../docs/performance/flash-next-qsa-row-flags-2026-11-27.md).

## Files

- `old_block_key_body.metal`, `block_key_header.metal`: the block-key
  kernel as of commit `2c97bdd07`, with the serial row-flag scan.
- `bench.py`: builds that kernel, a copy without the scan, and the new
  row-flag kernel; compares keys and flags bit for bit on random scores and
  on scores with NaN, -inf, a subnormal and a non-finite future block; and
  times 30 evaluations per shape. Usage: `python bench.py <this directory>`
  with MLX 0.32.4 and NumPy.
- `m2-ultra.txt`, `m4-pro.txt`: the output on each host.
