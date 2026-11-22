# Flash Next shared SDK MTP preflight

The functional control ran on a Mac Studio M2 Ultra with 192 GB and macOS
27.2.1, using Rust 1.99.1 and MLX 0.32.4. Both preflight arms verified all
49 publisher files for Flash Next AXQ MXFP4 at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` before starting the SDK test.

| Explicit Required MTP plus sidecar bypass | SDK error | Shared weights after error | Elapsed |
|---|---|---|---:|
| Instrumented baseline at `fd85d821f` | `MlxMtpRequiredButUnavailable` | Loaded | 14.811957 s |
| Fix at `e91bf4afb` | `MlxMtpRequiredButUnavailable` | Empty | 0.057128 s |

Each arm started with an empty shared cell. The timer covers the stateful
session build after SDK context creation. The baseline instrumentation
only captures the shared cell state, elapsed time, and resulting error. A
separate test at the fixed source verified that two concurrent native
runners share one weight allocation containing the attached drafter.
Both runners generated the same eight token IDs. All recorded pressure
samples were normal.

These observations cover the explicit shared SDK conflict and the
drafter-present reuse control. Product defaults and M5 acceptance retain
their existing evidence requirements.

Run `shasum -a 256 -c SHA256SUMS` and `python3 replay.py` from this
directory to check the archived receipts and raw test output. Pass
`--source` with a worktree at `e91bf4afb` to check the fixed source
hashes recorded with the test binaries.
