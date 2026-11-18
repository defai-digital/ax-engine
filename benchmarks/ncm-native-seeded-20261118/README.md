# NCM Seeded V1 functional controls

The pinned F32 TinyStories artifact is
`janmusic/NCM-TinyStories-89M-Upstream-MLX` at
`2d5f137713fe047abb6fcffe780aad419486f3dd`. The build used Rust 1.99.1,
MLX 0.32.4 and the Cargo dev profile. The runtime changes were built
on top of `cd3fbea1b91bc0da2800598c7d7110cdc55e1bc6` with runtime and test
patch SHA-256 `03d2e9d4cecf1b98403535f297c44fd9704050ac1f3f5e11d15984623fd99379`.
The host receipts contain the hashes of the executed binaries and MLX library.

The two real-weight Rust SDK tests and the synthetic keyed sampling test
passed on each host. The same seed replayed exactly after other requests
and a partially executed request was canceled. The retained greedy trace
matched all 128 raw BPE positions and the text. Seven negative CLI option
cases were rejected before artifact loading.

The CLI also reproduced seed 17 across process invocations, distinguished
seeds 17 and 29, and completed with the maximum unsigned 64-bit seed.
Seeds 17 and 29 produced different outputs on the M5 Max and M4 Pro
despite identical binary and MLX library hashes. The reproducibility
contract is per build, model and hardware. The results cover functional
behavior of the experimental typed SDK and CLI.

- [MacBook Pro M5 Max 128 GB receipt](m5-functional.json) and its
  [raw greedy and seeded outputs](m5-exact-outputs.json).
- [Mac mini M4 Pro 64 GB receipt](m4-functional.json).
- [Mac Studio M2 Ultra 192 GB receipt](m2-functional.json).

Run the [documented functional controls](../../docs/NCM-EXPERIMENTAL.md#reproduce-the-functional-controls)
against a materialized copy of the pinned artifact. Receipt BPE hashes use
SHA-256 of compact JSON integer arrays.
