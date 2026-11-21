# Flash Next split boundaries on the native M2 Ultra stream

The [bound server build](build.json) ran the pinned Flash Next MXFP4
pack at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` on a Mac Studio
M2 Ultra with 192 GB, external SSD, macOS 27.2.1 and MLX 0.32.4. The
resident two-slot diagnostic disabled MTP, n-gram acceleration and
prefix-cache reuse. It verified all 49 publisher files before and after
the run.

Two 2,048-token serial controls established the output IDs. Every
one of the nine steady-length responses matched all 96 baseline IDs.
Two paired trials confirmed the request-local logical boundary on
each physical step of 512, 512, 512, 511 and 1 tokens while the
other stream decoded. A five-output decoder finished on the same
engine step as the peer's physical prefill. Its terminal step and
response preserved output identity and did not carry the other
request's boundary.

One client disconnected during physical prefill and another after the
first output. The survivor and replacement in each case matched the
serial controls. All eight lifecycle samples showed zero serving
gauges after drain, and the 119 memory observations had normal
pressure. This evidence covers native wire correctness at the tested
2,048-token shape. M5 holdouts and sustained serving remain separate.

The [result](result.json.gz) contains the detected markers, complete
outputs and resource samples. The [summary](summary.json) binds it to
the executable and records the mixed terminal engine step.

## Replay

Extract [replay tools](replay-tools.tar.gz). Use the exact
`release-server` candidate from the build receipt, the
[publisher inventory](../qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json)
and [pinned prompts](prompts.jsonl). Set the library search path to the
pinned MLX 0.32.4 wheel. Run `run_flash_next_split_wire_m2.py` with
`--binary`, `--flash`, `--prompts`, `--inventory` and a fresh
`--out` directory. The included `test_split_step_sse.py` runs
the parser controls without model weights.
