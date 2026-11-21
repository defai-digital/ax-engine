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

The boundary probe sends pre-tokenized synthetic input with a one-token output
budget. Readiness and request timings are diagnostic observations. The
[signed v7.9.0 direct-default record](../qualification/2026-10-24-flash-next-mxfp4-m5max-release-v7.9.0/README.md)
remains the release qualification record. The Flash Next certification record
is Candidate and MTP Tier 2 remains pending.
