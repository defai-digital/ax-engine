# Qwen 3.8 Flash Next peer snapshot — MacBook Pro M5 Max 128 GB, 2026-10-23

Pack: `AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP` at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` (`model_type=qwen4_exp`), one directory on
the internal SSD for every runtime. Host: MacBook Pro, Apple M5 Max, 128 GB, macOS
27.0.1 (the ADR-049 Flash Next standard host). Contract: the flappy suite (four cases,
264-432 prompt tokens), greedy, 256 generated tokens, two warmups and five measured
repetitions per case (20 samples), lanes run one after another
(`scripts/bench_flash_next_peer.py`, defaults). Single-host snapshot; it qualifies no MTP
gate and changes no default.

| Runtime | Version | Result | Decode tok/s (median) | Prefill tok/s (median) |
| --- | --- | --- | ---: | ---: |
| AX Engine, direct (MTP off, product default) | 7.8.0 `1b639eec` | measured | **51.41** | 852.6 |
| AX Engine, MTP required (depth 1) | 7.8.0 `1b639eec` | measured | 47.45 | 798.9 |
| OMLX (VLM engine, MTP depth 3, and again with MTP off) | 0.7.2 (MLX 0.32.4, mlx-vlm 0.7.8) | failed: Metal out of memory at the first prefill | — | — |
| MTPLX | 2.12.2 (MLX 0.32.2) | unsupported: load rejects the pack's PLE n-gram shard tensors | — | — |
| mlx-lm | 0.32.2 (latest) | unsupported: no `qwen4_exp` loader | — | — |

- AX prefill is the runner-internal cold-prefill figure (prefix cache disabled). The AX
  MTP lane drafted 2,550 tokens at the first MTP position across the 20 repetitions and
  accepted 2,535 (99.4%); deeper positions drafted nothing. MTP stays slower than direct on this pack
  (0.92x), as in the 2026-10-05 release-gate matrix, so it is not a speedup claim.
- OMLX 0.7.2 ran out of Metal memory (`kIOGPUCommandBufferCallbackErrorOutOfMemory`) on
  the first prefill in both attempts and served no tokens. The same lane served this pack
  on the 192 GiB M2 Ultra host on 2026-10-08 (33.87 tok/s median, MTP depth 3). OMLX read
  a same-byte symlink view whose copied index lists the `mtp.safetensors` sidecar (see the
  harness); the pack was not modified.
- MTPLX 2.12.2 still rejects the pack at load (384 PLE n-gram shard tensors are not
  parameters of its `qwen4_exp` module), and mlx-lm 0.32.2 has no `qwen4_exp` model, so
  no same-pack peer number exists on this host.
- AX was built with the Homebrew Rust 1.99.1 on this host (no rustup installed); the
  repository pins 1.97.1.

Files: `ax/` (AX MTP lane, mlx-lm lane record, plan), `ax-direct.json` and
`ax-direct-prompts/` (AX direct lane), `mtplx/`, `omlx/`, `omlx-nomtp/` (logs and lane
records), `summary.json` (all lanes).
