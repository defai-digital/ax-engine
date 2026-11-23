# Flash Next document-prefix cache controls on M4 Pro

A Mac mini M4 Pro 64 GB with macOS 27.0 ran eight focused library tests
from a signed release-server test executable built with Rust 1.99.1
and MLX and MLX Metal 0.32.4. The tested source is the guarded
document-prefix experiment plus the runner controls.

The new tests prove a learned checkpoint survives eviction of both
full-prompt entries, concurrent learners publish one snapshot, and
terminal cleanup discards a cancelled learner's unpublished state.
The remaining tests cover token identity, loaded-model ownership,
reclamation, retention policy and switch classification.

This is a supported-tier software check. These cases use synthetic
runner state. The primary real-pack serving comparison is tracked
separately.

```sh
shasum -a 256 -c SHA256SUMS
xz -dc source-with-controls.patch.xz | shasum -a 256
```
