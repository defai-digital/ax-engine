# Flash Next native SSE split-boundary negative control

The pinned Flash Next MXFP4 pack
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` ran on a MacBook Pro
M5 Max with 128 GB unified memory, internal SSD, macOS 27.2.1
and MLX 0.32.4. The [build record](build.json) binds the
`0f597f13c` native server, Rust 1.99.1 and the executable hash.
The runner checked all 49 pinned pack files at both ends.

With MTP and the prefix cache disabled, the experiment matched
all 28 output arrays against the equivalent serial request and
completed six disconnect and replacement trials. Each output
array contains 96 token IDs. Four experimental requests exposed
the physical widths 512, 512, 512, 511 and 1. Each shared a
512-token step with a peer that had already emitted its first
output. Every resource sample reported normal pressure.
Among the four nonserial observations per arm, the maximum
inter-token gap was 1.745 s with splitting disabled and 0.535 s
with it enabled. These are bounded workload observations.

The original route audit rejected the receipt because progress
frames omitted the request-keyed exclusive logical cut. The
physical audit confirms the independent observations and marks
the logical cut unverified. The 2,049-, 2,050- and 8,192-token
arms did not start under this build. The strict SSE audit
requires a subsequent run using the corrected SDK.

## Replay the audits

Decompress `result.json.gz` and pass it to either verifier
with `--server-sha256` from `build.json`. The
[original verifier](verify_original.py) exits with an assertion
on the missing cut. The [physical verifier](verify_physical.py)
reproduces [the scoped summary](summary.json).
