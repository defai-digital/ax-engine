# Tiel bounded default-session residency on M4 Pro 64 GiB

The bounded session-residency candidate passed the matched server checks below.
This qualifies the scoped memory default, not model accuracy, MTP default
promotion, cross-cache-state token identity or a universal best setting.

This is AX default-server acceptance, not a new MTPLX comparison. The matched
four-host native MTP peer results remain in `../2026-09-20-peer/RESULTS.md`.
MTP remains Auto with its existing candidate promotion gate; no forced MTP,
no n-gram/prefix-cache override and no idle GPU keepalive are used here.

The candidate changes only optional Auto residency for the two audited export
configurations on M4 Pro exactly 64 GiB. It requires a known session KV pool
of 1..16,384 tokens, prefill chunk of 1..2,048, normal OS pressure and at most
512 MiB existing MLX active allocation. Known weight/KV/runtime footprint plus
that active allocation must fit the lower of Metal's recommended working set
and 48 GiB (75% of physical RAM). This leaves at least 16 GiB outside the
model budget; it is not a measurement of free RAM or an allocator guarantee.
Unknown probes/geometry retain the general Auto rule. Required packs and
explicit On/Off are unchanged. Raw weight/runner constructors without an
admitted session budget retain the original full-weights-plus-48-GiB rule.
The existing server add/replace admission is unchanged.

`bench_default.py` uses two reversed arm orders, fresh processes, a short
16-output-token request (one warmup, three measured requests) in Auto, Off and
On; Auto/Off also run a roughly 15k-token prompt with 128 output tokens (one
warmup, two measured requests), then cancel a stream and test recovery.
The same 3-second idle interval is used before every timed request. Repeated
prefixes intentionally exercise the default cache policy. Cold and warm long
requests must therefore be reported separately.

OS pressure, swap and VM counters are sampled each second. A non-normal
pressure level or more than 256 MiB swap growth terminates the test process.
No external memory stress is injected. HTTP TTFT is first visible content or
reasoning delta; completion throughput is authoritative output tokens divided
by request-to-DONE wall time. Do not call that decode throughput: SSE frames
can contain multiple tokens. Raw SSE frames and selected server metrics retain
route, cache and active/peak allocator evidence.

Reproduce on the specified hardware with both pinned model directories:

```bash
rustup run 1.97.1 cargo build --release -p ax-engine-server
python3 bench_default.py --server /path/to/ax-engine-server \
  --model-root /path/to/models --output /path/to/results --port 31947
python3 verify.py
python3 -m pytest test_verify.py
```

Point the dynamic loader at the matching MLX 0.32.2 library if the binary's
embedded library path is unavailable. Run one inference process at a time;
do not mix these HTTP measurements with the native MTP peer timing boundary.

## Recorded results

52 measured requests, 20 explicit warmups, eight cancellations and eight
recovery requests completed across 12 fresh server processes. All processes
exited with code 0. Default server startup warmup remains enabled. This is
single-client performance evidence; separate two-client smoke is functional
coverage, not an unbounded-concurrency or high-pressure certification.

Short workload: 38 prompt tokens, 16 output tokens, six measured samples per
cell. Completion includes HTTP TTFT and stream completion. Empirical p95 uses
linear interpolation over these six samples; it is not a tail-latency SLA.

| Pack | Expert mode | Completion tok/s median | TTFT p50 / p95 (ms) | TTFT range (ms) |
| --- | --- | ---: | ---: | ---: |
| Tiel | Auto | 31.50 | 305.66 / 316.55 | 298.91–319.88 |
| Tiel | Off | 30.99 | 314.24 / 323.37 | 298.39–324.06 |
| Tiel | On | 0.67 | 4982.58 / 5067.50 | 4911.90–5071.60 |
| Cyber-Tiel | Auto | 30.93 | 314.97 / 318.78 | 305.02–319.25 |
| Cyber-Tiel | Off | 31.12 | 311.23 / 317.40 | 306.65–318.08 |
| Cyber-Tiel | On | 0.66 | 5035.91 / 5039.68 | 5023.18–5039.81 |

Auto reaches the full-resident Off control without an operator flag. On is
the same paging route selected by the former general Auto rule on this host;
it is a same-binary control, not a separate old-release measurement. No new
AX-versus-MTPLX multiplier is claimed.

Long workload: 15,139 prompt tokens and 128 output tokens. Four repeated
measured requests per cell, plus two initial warmup requests recorded separately.

| Pack | Expert mode | Initial-request TTFT median (s) | Repeated TTFT median (s) | Repeated completion tok/s |
| --- | --- | ---: | ---: | ---: |
| Tiel | Auto | 18.446 | 18.209 | 6.33 |
| Tiel | Off | 18.393 | 18.043 | 6.38 |
| Cyber-Tiel | Auto | 18.433 | 18.674 | 6.16 |
| Cyber-Tiel | Off | 18.408 | 18.549 | 6.20 |

## Output and cache-state audit

All short outputs and recovery outputs match across modes and blocks. Long
Auto/Off outputs match **at the same request phase** in both blocks. The initial
long output differs from subsequent repeated outputs for each pack, in both
Auto and Off. The first overly strict verifier caught this; the final verifier
matches initial-to-initial and repeated-to-repeated while retaining all raw
outputs. It still rejects any cross-arm or cross-block difference within a phase.

Repeated long requests record one core prefix hit and 15,136 prefix-warmup
tokens; MLX prefix-cache reused-token counters remain zero. The first divergent
logit has not been traced, so this evidence does not establish the numerical
cause or cold/warm token identity. The memory-policy change introduces no
difference relative to the same-phase resident control. MTP active metrics
remain zero throughout; this is not an MTP promotion result.

## Memory and lifecycle

Maximum observed MLX allocator peak: **25.40 GiB**. The admission
baseline was 23.89 GiB; its 48 GiB cap retained substantial margin in these
workloads. All periodic pressure samples stayed at the normal flag (1);
maximum per-process-window swap growth was **0.00 MiB**, below the
256 MiB stop guard. OS counters are host-wide observations, not per-engine
attribution. Background applications were not terminated or system-stressed.

Eight cancellation/recovery sequences completed; all 12 servers shut down
without SIGKILL. Larger-pool and user-alias checks are separate supplemental
artifacts. Wider hardware unwiring, idle keepalive, unlimited concurrency and
high-pressure behavior are not promoted by this acceptance.


## Supplemental checks and provenance

`supplemental.json` records both exports through the user alias's `ornith-35b`
server preset, including two overlapping 128-token requests per pack. Both
Auto decisions retain weights and all processes exit normally. A separate
32,768-token pool (`--total-blocks 2048`) retains Auto paging as intended.
Doctor Auto reports `audited_tiel_default_session_fits_budget`; explicit Off
reports `forced_resident`, with no weight tensors loaded by doctor.

`provenance.json` binds the release server binary, Rust 1.97.1, MLX 0.32.2,
macOS 26.6.2, changed compiled source hashes and all 40 model files. The model
files were rehashed after the campaign against the pinned peer inventory.
The binary was built before the source commit; its compiled Rust files are
byte-identical to the committed files. Evidence files contain no internal
host aliases or local model paths.

Rust 3,763 tests passed; Python 211 passed, 26 skipped (146 subtests passed).
Format, production Clippy, script checks, both qualification dry-runs and
primary-claim checks passed. Full all-target Clippy retains 1,802 existing
error lines, identical in category/count to the preceding baseline, with no
reference to the modified files. The artifact verifier and six mutation tests
pass. No push, release, or replacement of the user's installed CLI is included.

The memory-pressure sysctl returns dispatch flags; normal is 1, distinct from
the kernel's internal enum. This interpretation was checked against
[Apple XNU's sysctl conversion](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_memorystatus_notify.c)
and [dispatch-flag definitions](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/event_private.h).
