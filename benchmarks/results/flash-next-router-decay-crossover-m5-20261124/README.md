# Flash Next router and decay projection crossover

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1 and MLX 0.32.4.
The signed Rust 1.99.1 `release-server` test binary runs the pinned
MXFP4 pack at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` on the same
8,192-token prompt, with 64 greedy output tokens. Every arm uses a
fresh resident process, one request, MTP off and stock QSA.

The test-only router switch tiles the affine 8-bit group-32 router
projection on eligible wide BF16 Shared spans. The separate decay
switch uses the existing GDN projection tiler. The switches preserve
the physical prefill partition. The narrow stock and switch controls,
two stock wide repeats, and decay-only arm repeat their own archived
IDs, logits and state witnesses bitwise. The switches engage zero
projections at 512 rows, and engage 192 router or 144 decay
projections per eligible wide request.

| Wide arm | Output positions differing from narrow, of 64 | First differing layer state | Diagnostic first token |
| --- | ---: | ---: | ---: |
| Stock, first | 61 | 0 | 9.995 s |
| Router only | 61 | 0 | 9.934 s |
| Decay only | 60 | 1 | 9.605 s |
| Both | 62 | 2 | 9.996 s |
| Stock, last | 61 | 0 | 9.686 s |

The combined arm recovers the layer-zero and layer-one serialized
states at every retained cut. Its separate trace matches the
non-instrumented arm. All eleven sampled layer-zero router and MLP
positions match the narrow arm. All captured layer-one inputs, all
2,047 GDN output rows and the final recurrent state match bitwise.
The first two completion IDs match the narrow arm.

All 481 memory pressure samples are normal and stay within the 96 GiB
footprint guard. Both narrow controls use about 74.12 GiB of peak MLX
memory. All uninstrumented wide arms use about 84.02 GiB. Diagnostic timing
includes the scheduler and excludes model loading and offline replay.
The stock narrow first-token observations are 8.998 and 9.505 seconds
with the switches disabled and enabled, respectively. The measured
scope supports the local arithmetic controls and leaves the 512-row
release path as the validated default.

`python3 -B replay.py` verifies the signed inputs, archived controls,
per-arm metrics, state and output hashes and regenerates
`comparison.json`.
