# Flash Next MXFP4 serving envelope, 2026-11-21

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1. Both builds use
Rust 1.99.1 and MLX 0.32.4. The candidate uses the `release-server`
profile. The source baseline uses `release`. The model inventory is
bound to the published MXFP4 MTP revision
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`.

| Control | Observation |
| --- | --- |
| [Old ordinary launch](ordinary-launch-before.json) | 16,384 advertised tokens, eight priced live sequences and no configured admission cap |
| [Candidate ordinary launch](startup-envelope.json) | 32,768 advertised tokens, one priced live sequence, resident experts, and zero MTP verifier steps |
| Context boundary | HTTP 200 for 32,767 prompt tokens and one output token. A request for the same prompt and two output tokens receives HTTP 400 with `context_length_exceeded`. |
| Admission and cleanup | The second request receives HTTP 429 while the first stream is active. After disconnect, all five lifecycle gauges drain, and an identical request reproduces the same text, finish reason, and token counts. |
| Explicit overrides | The command line sets `--max-concurrent-requests=0 --total-blocks=1536` with `AX_ENGINE_MAX_CONCURRENT_REQUESTS=1`. The server advertises 24,576 tokens and the overlapping short completion receives HTTP 200. |
| [Mac mini M4 Pro 64 GB control](qwen27b-m4-control.json) | The same candidate binary serves the pinned 27B MXFP4 pack with 16,384 advertised tokens. The Flash Next profile is absent. |
| [Model replacement](replacement-control.json) | A failed load restores the Flash Next session and its 32,768-token budget. A successful replacement with 27B uses 16,384 tokens. The reverse replacement is refused by the memory preflight at an estimated 141.9 GiB. |

## Replacement with disk-served table accounting

The follow-up `release-server` build
(`0a7c6049a5af88ed1ebdfce02c648416349b2b1eed301381cb6eff5b310e6af5`)
ran on the same M5 Max, OS, Rust and MLX stack. The pinned Flash Next pack and
the pinned 27B MXFP4 pack (`b2c5354f779e430d0c1733143db848a72b71c16e`)
were installed on the internal SSD. The return path used a Hub-shaped
snapshot with 49 symlinked files backed by the same pinned Flash Next pack.
Its native manifest was produced by the successful loader.

- [Two-way replacement](replacement-after.json) changes the budget from
  32,768 to 16,384 and back to 32,768. Returning to Flash Next took 48.0 s.
  After rollback and after returning, an overlapping short request received
  HTTP 429. The 27B generation accepted the overlapping short request.
  The recovered Flash Next completion has identical text, finish reason,
  and token counts.
- [Discount disabled](discount-disabled.json) retains the 141.9 GiB
  projected estimate, rejects the replacement with HTTP 422, and continues
  serving the existing 27B session at 16,384 tokens.

## Lifecycle replay on the replacement build

The [same-build lifecycle receipt](lifecycle-current-binary.json) records
all 21 actions on the M5 across disabled, default and required MTP.
Every action drained all five counters and each process shut down
cleanly. The two direct arms finished with zero verifier steps, and
the required arm recorded 112. This replay used the frozen 4,096-block
lifecycle setup. The ordinary 32K admission and disconnect controls
above use the product envelope. CPU idle was 94.77% at start and 84.90%
at finish. The published timing gate requires 90% at both ends.

The [quiet-host repeat](lifecycle-quiet-retry.json) completed all 21
actions on the same build, pack and pool. CPU idle was 93.64% at
start and 93.73% at finish. All five gauges drained after each
action, and all three server processes exited cleanly. Response
texts, finish reasons and token counts match the earlier replay.
Readiness took 67.710, 57.576 and 57.580 seconds for disabled, default
and required. The two direct arms had no verifier steps and required
recorded 112. This passes the frozen lifecycle and CPU-idle checks
for this receipt. The current reference-relative timing matrix remains
a separate release gate.

The boundary probe sends pre-tokenized synthetic input with a one-token output
budget. Readiness and request timings are diagnostic observations. The
[signed v7.9.0 direct-default record](../qualification/2026-10-24-flash-next-mxfp4-m5max-release-v7.9.0/README.md)
remains the release qualification record. The Flash Next certification record
is Candidate and MTP Tier 2 remains pending.
