# ax-engine-microbench

Low-level benchmark and probe binaries live here instead of in the production
MLX runtime crate.

This crate owns source-level measurement tools for kernel experiments,
correctness probes, promotion evidence, and stress checks. It may depend on
`ax-engine-mlx`; `ax-engine-mlx` must not depend on this crate.

## Binaries

```bash
cargo run -p ax-engine-microbench --release --bin dequant-dtype-probe
cargo run -p ax-engine-microbench --release --bin kernel-chain-batching-probe
cargo run -p ax-engine-microbench --release --bin rmsnorm-fused-probe
cargo run -p ax-engine-microbench --release --bin residual-rmsnorm-fused-probe
cargo run -p ax-engine-microbench --release --bin mla-warm-extend-drift-probe -- --help
cargo run -p ax-engine-microbench --release --bin disk-prefix-cache-stress -- --help
```

## Validation

```bash
cargo test -p ax-engine-microbench
cargo check -p ax-engine-microbench
```

Future probes should emit machine-readable JSON artifacts and should keep
correctness gates separate from timing measurements.
