# Flash Next KV reclamation rehearsal on M5 Max

The pinned MXFP4 pack at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`
completed the endurance preflight on MacBook Pro M5 Max 128 GB with
macOS 27.2.1 and internal SSD. The `b36b7b287` release-server build used
Rust 1.99.1 and MLX 0.32.4. The run used resident experts under Auto,
one active request, MTP disabled, a 2,048-token scheduler budget and a
pool of 2,048 × 16-token KV blocks.

Readiness and all 26 requests returned HTTP 200. Twenty warmups and four
fill requests reduced free blocks to 255. A fresh 4,280-token prompt
needed 268 blocks, forcing cache reclamation. The long and medium probes
completed with 21 and 88 cumulative prefix-cache eviction events,
respectively. The cache retained 2,039 and 2,040 blocks after the probes,
with no remaining active requests, queued work, stream events or KV block
tables. The server stopped normally.

The rehearsal took 102.75 seconds after readiness. The runner reported
`preflight_passed` with `measurement_started: false`. Measured endurance
and long-run resource trends remain open.

The [receipt](preflight.json) preserves the original capacity anchors,
request outcomes and selected lifecycle counters. It also records the
server, pack and runner hashes and the checksums of the unfiltered
local artifacts.
