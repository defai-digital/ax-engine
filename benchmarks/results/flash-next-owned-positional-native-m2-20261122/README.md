# Flash Next native positional-reader smoke on M2 Ultra

Mac Studio M2 Ultra 192 GB, external SSD, macOS 27.2.1, Rust 1.99.1
and MLX 0.32.4. Both controls ran `8729b80c2` on the same pinned
installed MXFP4 pack in order, with guarded mmap first and opt-in
owned-storage positional reads second. The harness independently
verified all 49 publisher members before the two launches. It
exercises one resident request with MTP disabled and the same exact
512-, 2,048- and 8,192-token prompts as the earlier native smoke.

| Payload reader | HTTP ready | Resident tensor phase | MTP sidecar phase |
| --- | ---: | ---: | ---: |
| Mapped byte copy | 34.313 s | 10.173 s | 1.448 s |
| Positional read | 94.965 s | 43.258 s | 36.318 s |

The raw traces show the same 15 shards and 1,942 resident tensors. Both
arms reproduce all 96 earlier greedy token IDs with zero verifier
steps and zero route errors. All nine startup pressure samples are
normal. Peak sampled process RSS is 69.07 GiB mapped and 68.98 GiB
positional.

The whole-model timings on the reused snapshot differ from the
[independent single-shard measurements](../flash-next-owned-positional-shards-m2-20261122/README.md).
Initial file residency is uncontrolled here. The sidecar uses the same
route in both runs, yet its phase grows by 34.870 seconds. The observed
61-second readiness gap therefore contains large variation outside the
selected trunk reader. The result establishes route and token identity
for this smoke. The selected-page state and warmed serving performance
remain subject to the counterbalanced M5 comparison.

Replay [comparison.json](comparison.json) from the
[sealed evidence](evidence.tar.xz) and earlier pinned native receipt.

```bash
python3 benchmarks/results/flash-next-owned-positional-native-m2-20261122/replay.py
```
