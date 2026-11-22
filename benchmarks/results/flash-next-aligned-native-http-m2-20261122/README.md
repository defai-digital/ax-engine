# Flash Next aligned reader native HTTP controls on M2 Ultra

Mac Studio M2 Ultra 192 GB with an external SSD. The Rust 1.99.1
`release-server` binary from `31e2b6e1f` uses MLX and MLX Metal 0.32.4.
The runner checks every publisher file against the frozen MXFP4 pack
inventory before launching the four servers and after the last exits.

The comparison uses the same binary and seven controls per arm, through
the native HTTP entry point. The arms run in default mapped, default
aligned, Required aligned, then Required mapped order. They cover the
first and repeated text requests, an extension, JSON, SSE, a request
after SSE, and one-token completion. Within each policy, the JSON
choices, stop reasons, token accounting, and verifier counter changes
match exactly. Each Required arm records 107 verifier steps. Each
default arm records zero.

The startup logs confirm the requested reader with 15 resident shards
and 1,942 selected tensors in every arm. Both host pressure samples for
each arm report normal. The receipt includes observed readiness from
this functional run. File cache and workload were uncontrolled, so use
the primary-tier crossover for a startup or serving performance decision.

Replay the curated observations from this directory.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | diff -u comparison.json -
```
