# Flash Next late-arrival interleave on M5 Max

A MacBook Pro M5 Max 128 GB with an internal SSD ran the pinned
Flash Next MXFP4 pack at revision
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. The same
experimental Rust 1.99.1 and MLX 0.32.4 executable is bound in
all four arms. Two HTTP requests share resident capacity with MTP
disabled. The 8,192-token request is released as the existing 512-token
stream emits token 84 of its 96-token output. The 16-layer experiment
uses the same physical ends once a logical split starts.

| Arm | Short final 12 token gaps, max | Long TTFT from arrival | CPU idle before / after |
| --- | ---: | ---: | ---: |
| Physical first | 725.90 ms | 10.239 s | 94.57% / 91.30% |
| Layers first | 269.56 ms | 9.875 s | 91.45% / 85.79% |
| Layers second | 261.44 ms | 9.682 s | 90.78% / 86.85% |
| Physical second | 762.71 ms | 10.597 s | 90.10% / 94.35% |

Both layer arms miss the predeclared 90% post-run CPU-idle floor.
The observed 62.87% and 65.72% shorter maximum short gaps remain
diagnostic timing observations. The actual overlap, exact prompt
boundaries, and all 768 output IDs across four arms pass. All
567 observed pressure samples were normal.

Each physical-only arm starts three split logical items and a fourth
ordinary item. Each layer arm starts one split logical item and
three ordinary items, recording eight genuine layer suspensions while
the short request produces output. Every split keeps widths
512, 512, 512, 511 and 1. After the short stream ends, no new split is
started. The workloads and reference IDs stayed fixed. The observed
number of cuts while a peer was active changed under the policy.

The comparison identifies a case for a bounded near-completion
dispatch experiment. Layer checkpointing retains its experimental
status. Direct-default and release qualification remain separate.

## Replay

Run `python3 -B replay.py` to validate the exact trigger, prompt IDs,
greedy outputs, split widths, layer checkpoints, timing eligibility,
and measured values from the archive. The driver, pack inventory and
scrubbed portable capture are included. The executable is identified
by SHA-256 and the source receipt. The same build is also used in the
[sustained mixed-load replay](../flash-next-prefill-layer-sustained-m5-20261122/README.md).
