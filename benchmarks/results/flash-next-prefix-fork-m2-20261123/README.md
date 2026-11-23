# Flash Next serialized prefix fork on M2 Ultra

A Mac Studio M2 Ultra 192 GB ran the pinned resident MXFP4 pack on an
external SSD with macOS 27.2.1, Rust 1.99.1, and MLX 0.32.4. The signed
test uses the production serialization and model-rebind helpers. The
source lands in `434f1cd1e`. TensorFold's saved prompt fork controls
informed the independent cold and warm branch checks.

The probe rehydrates the same 2,048-token seed before each distinct
64-token suffix. It compares both suffix states with cold runs and
drops the first branch after four outputs while a separate peer is
decoding. The replacement matches its cold run's 16 greedy IDs and
serialized decode state. The peer's 21 emitted IDs and final serialized
state match its independent execution. The saved payload retains its
original digest. The probe dispatches 6,984 native singleton GDN kernels.
All 23 pressure samples are normal.

The test process passed. The original launcher exited 1 when it parsed
the result marker at the beginning of a line that also contained the Rust
test name. The recovered receipt parses the retained test log and then
rechecks the same signed executable and the publisher inventory. The
archive retains the original failure and the recovery status. This
exercises the two restored branches inside a model test. The native
serving route and its performance have separate controls.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
