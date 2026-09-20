# Tiel and Cyber-Tiel: AX Engine vs MTPLX on four Macs

Campaign date: **2026-09-20 UTC**. This report records the completed native
MTP peer comparison and separately identifies the subsequent M4 default-server
acceptance. It does not present the peer timings as measurements of the later
memory-policy commit.

AX completes the tested coding workloads faster on M5 Max and M3 Ultra.
M2 Ultra results are mixed; M4 Pro is close, with MTPLX sustained slightly
faster overall. AX has lower time to first token (TTFT) on M5; MTPLX has
lower TTFT on the other three hosts under the ordinary AX wiring policy.

| Tested machine | AX completion difference vs MTPLX sustained | Coding-workload interpretation |
| --- | ---: | --- |
| MacBook Pro M5 Max, 128 GiB | +3.1% to +18.0% | AX leads in completion, decode and TTFT. |
| Mac mini M4 Pro, 64 GiB | -2.4% to -0.5% | MTPLX starts sooner and finishes slightly faster; AX decode is slightly faster. |
| Mac Studio M2 Ultra, 192 GiB | -2.1% to +8.6% | AX decode is faster; MTPLX starts sooner; completion is workload-dependent. |
| Mac Studio M3 Ultra, 512 GiB | +2.2% to +4.7% | AX finishes faster; MTPLX starts sooner. |

Ranges describe the four individual coding cells per host, not pooled speedups
or confidence intervals. The mini has **64 GiB physical memory**, not 48 GiB.

## Models, builds and matched conditions

| Pack | Pinned revision |
| --- | --- |
| `AutomatosX/AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP` | `5ab39b24bfd7f65203be9b7823b1840486f58b6d` |
| `AutomatosX/AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP` | `fe05e871ec69ad9ae8eac01fd285555514ac7daf` |

AX uses the optimized Rust 1.97.1 `release-pyext` build on `574d2d19` plus
the Python expert-mode fix `5e07befe`. MTPLX **2.11.3** is pinned to
`7c2205ae4f3b91d7d3852b6ac7510174b4b5d0ae`. Both engines use **MLX 0.32.2**
and identical MLX library bytes. The
[comparison contract](../../benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-20-peer/README.md)
retains full source, binary and model provenance.

- Identical model files and exact input token arrays on each engine.
- Explicit throughput-MTP, maximum draft depth three, greedy sampling, seed
  zero and fixed output count. This is not an out-of-box server comparison.
- Cold request KV, no prefix reuse. Both engines load once per process and
  keep the full model loaded. Loading and tokenization are outside the timer.
- Two fresh-process blocks in reversed engine order. Each workload has two
  warmups and three measured requests per block: six measured samples per cell.
  Medians average the middle two sorted values.
- Three seconds idle before each request. Only one benchmark process runs
  per host; unrelated background processes remain untouched.
- MTPLX sustained and turbo are both reported. Its separate server memory
  setup and GPU keepalive are not exercised by this native API harness.
- AX explicitly disables expert streaming on all four hosts. Its ordinary
  wiring policy automatically unwires the audited exports on M5; M4/M2/M3
  retain wiring. Explicit unwired controls are reported separately below.

The full matrix contains **540 measured requests and 360 warmups**, including
unwired controls. A separate historical M4 Auto diagnostic adds six measured
requests and six warmups; it is not part of the tables below.

| Workload | Tiel prompt tokens | Cyber-Tiel prompt tokens | Output tokens |
| --- | ---: | ---: | ---: |
| `python-lru` | 194 | 219 | 256 |
| `rust-jsonl` | 1039 | 1064 | 256 |
| `random128` (synthetic diagnostic) | 128 | 128 | 128 |

## Completion throughput

All values are median **tokens/s; higher is better**. Completion throughput is
output count divided by native API entry-to-last-callback time, including TTFT.
It excludes model loading, HTTP and network latency. The last column compares
AX with sustained; turbo remains visible because it is not consistently faster.

| Machine | Model | Workload | AX Engine | MTPLX sustained | MTPLX turbo | AX vs sustained |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| M5 Max / 128 GiB | Tiel | python-lru | 194.88 | 177.44 | 178.47 | +9.8% |
| M5 Max / 128 GiB | Tiel | rust-jsonl | 172.08 | 145.82 | 139.93 | +18.0% |
| M5 Max / 128 GiB | Cyber-Tiel | python-lru | 219.43 | 194.15 | 201.52 | +13.0% |
| M5 Max / 128 GiB | Cyber-Tiel | rust-jsonl | 169.14 | 163.99 | 159.01 | +3.1% |
| M4 Pro / 64 GiB | Tiel | python-lru | 90.46 | 92.23 | 93.62 | -1.9% |
| M4 Pro / 64 GiB | Tiel | rust-jsonl | 64.62 | 64.94 | 60.97 | -0.5% |
| M4 Pro / 64 GiB | Cyber-Tiel | python-lru | 94.37 | 96.65 | 94.30 | -2.4% |
| M4 Pro / 64 GiB | Cyber-Tiel | rust-jsonl | 67.71 | 68.92 | 70.17 | -1.8% |
| M2 Ultra / 192 GiB | Tiel | python-lru | 107.28 | 104.96 | 103.12 | +2.2% |
| M2 Ultra / 192 GiB | Tiel | rust-jsonl | 81.24 | 82.98 | 84.90 | -2.1% |
| M2 Ultra / 192 GiB | Cyber-Tiel | python-lru | 117.59 | 108.27 | 113.78 | +8.6% |
| M2 Ultra / 192 GiB | Cyber-Tiel | rust-jsonl | 84.80 | 84.51 | 84.46 | +0.3% |
| M3 Ultra / 512 GiB | Tiel | python-lru | 160.35 | 153.48 | 153.48 | +4.5% |
| M3 Ultra / 512 GiB | Tiel | rust-jsonl | 127.19 | 121.80 | 110.72 | +4.4% |
| M3 Ultra / 512 GiB | Cyber-Tiel | python-lru | 170.65 | 162.99 | 155.93 | +4.7% |
| M3 Ultra / 512 GiB | Cyber-Tiel | rust-jsonl | 134.49 | 131.54 | 132.20 | +2.2% |

Synthetic `random128` results remain in the
[full artifact tables](../../benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-20-peer/RESULTS.md).
Their larger gains in some cells are not generalized to coding workloads.

## Decode and first-token latency

Decode excludes the first callback's tokens and elapsed time. TTFT measures
native API entry to the first committed callback. This table compares AX with
**MTPLX sustained**; turbo phase timings remain in the full artifact tables.

| Machine | Model | Workload | AX decode tok/s | MTPLX decode tok/s | AX TTFT ms | MTPLX TTFT ms |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| M5 Max | Tiel | python-lru | 217.85 | 198.40 | 141.77 | 156.80 |
| M5 Max | Tiel | rust-jsonl | 211.48 | 174.49 | 281.58 | 293.99 |
| M5 Max | Cyber-Tiel | python-lru | 249.01 | 219.84 | 142.08 | 158.41 |
| M5 Max | Cyber-Tiel | rust-jsonl | 206.59 | 201.13 | 279.60 | 293.04 |
| M4 Pro | Tiel | python-lru | 109.12 | 107.04 | 493.77 | 393.47 |
| M4 Pro | Tiel | rust-jsonl | 97.39 | 94.49 | 1344.37 | 1244.43 |
| M4 Pro | Cyber-Tiel | python-lru | 116.29 | 114.25 | 519.00 | 414.17 |
| M4 Pro | Cyber-Tiel | rust-jsonl | 105.51 | 104.34 | 1364.65 | 1271.92 |
| M2 Ultra | Tiel | python-lru | 134.60 | 122.13 | 494.41 | 346.18 |
| M2 Ultra | Tiel | rust-jsonl | 119.42 | 115.25 | 1019.26 | 872.80 |
| M2 Ultra | Cyber-Tiel | python-lru | 152.53 | 125.98 | 505.93 | 341.83 |
| M2 Ultra | Cyber-Tiel | rust-jsonl | 126.67 | 117.52 | 1010.16 | 860.06 |
| M3 Ultra | Tiel | python-lru | 201.30 | 176.46 | 329.74 | 225.93 |
| M3 Ultra | Tiel | rust-jsonl | 179.58 | 154.96 | 593.25 | 457.63 |
| M3 Ultra | Cyber-Tiel | python-lru | 218.80 | 189.95 | 336.72 | 227.31 |
| M3 Ultra | Cyber-Tiel | rust-jsonl | 195.94 | 172.25 | 603.73 | 465.75 |

AX decode is faster in these coding cells, but M4's longer initial wait offsets
that advantage. M2 and M3 also show a TTFT tradeoff. The campaign does **not**
establish comparable isolated prefill tokens/s: TTFT includes prefill, initial
generation and runtime overhead, and the internal prefill timers have different
scopes. Do not relabel TTFT or completion throughput as pure prefill or decode.

## Memory and explicit unwired controls

Both engines retain the model across requests. **Unwired does not mean
unloaded**: it removes the OS residency lock and permits eviction under pressure;
it does not enable expert streaming or move inference to the CPU.

| Native MTP allocator measurement | AX Engine | MTPLX |
| --- | ---: | ---: |
| Request-end active allocation | 21.25 GiB | 19.95 GiB |
| Highest recorded allocator peak, across hosts | 22.41-22.46 GiB | 21.06-21.09 GiB |
| Highest recorded allocator cache | 0.10 GiB | 1.72-1.73 GiB |

MTPLX has a lower active allocation. These figures are not process RSS or
whole-system memory; allocator cache and active allocations are distinct.

Explicit `AX_MLX_WIRED_LIMIT_SCALE=0` controls improve coding completion
throughput relative to each host's ordinary AX wiring arm:

| Host | Completion change | Decode change | Status |
| --- | ---: | ---: | --- |
| M4 Pro | +1.3% to +4.9% | -3.4% to -1.0% | Explicit diagnostic only |
| M2 Ultra | +3.0% to +9.1% | -3.9% to -0.6% | Explicit diagnostic only |
| M3 Ultra | +7.5% to +8.4% | -1.3% to -0.6% | Explicit diagnostic only |

These gains come with lower TTFT and slightly lower decode throughput. They
have not promoted automatic unwiring on M2/M3/M4. M5's scoped automatic
unwiring is already part of its main AX arm. No universal best setting or
high-pressure/endurance qualification follows from these controls.

## Subsequent M4 default-server acceptance

The peer campaign's M4 build selected paging under Auto, so its matched
full-resident comparison explicitly used expert-stream Off. The later
**`c15a2347`** change allows the two audited exports to remain resident under
Auto on **M4 Pro exactly 64 GiB**, subject to session and memory guards.
This is separate AX-only evidence, not a new MTPLX run.

Admission requires a known KV pool of at most 16,384 tokens, prefill chunk of
at most 2,048, normal pressure, at most 512 MiB existing MLX active allocation,
and the estimated footprint plus active allocation fitting the lower of Metal's
recommended budget and 48 GiB. Unknown inputs keep the general Auto policy;
required streaming and explicit On/Off retain precedence. The 48 GiB model cap
here differs from the general policy's added 48 GiB headroom allowance.

The short default-server test uses 38 prompt tokens, 16 output tokens and six
measured samples per cell. MTP active telemetry is zero; default n-gram and
prefix policies remain enabled. Completion includes HTTP TTFT and stream end.

| Model | Auto completion tok/s | Auto TTFT ms | Forced resident Off tok/s | Forced paging On tok/s |
| --- | ---: | ---: | ---: | ---: |
| Tiel | 31.50 | 305.66 | 30.99 | 0.67 |
| Cyber-Tiel | 30.93 | 314.97 | 31.12 | 0.66 |

Auto reaches the resident control without an operator override. On is a
same-binary paging control, not a separately measured old release. These
16-token HTTP results cannot be compared directly with the 256-token native
MTP peer tables.

Across the default acceptance campaign, 52 measured requests, 20 warmups and
eight cancellation/recovery pairs completed; all 12 server processes exited
normally. Maximum MLX allocator peak was 25.40 GiB, periodic pressure stayed
normal and observed swap growth was zero. No external pressure was injected.
The approximately 15k-token prompt checks and raw evidence are in the
[default-server acceptance report](../../benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-20-default/README.md).

Long-request Auto/Off outputs match at the same request/cache phase. Initial
and repeated long outputs differ in both modes; the first divergent logit has
not been traced. This is not evidence of cold/warm token identity or closure
of all stability work. The scoped memory change does not promote MTP defaults,
broader unwiring or GPU keepalive. A fresh MTPLX comparison of this later build
remains unmeasured.

## Interpretation limits and evidence

Six samples per cell do not establish population confidence intervals. Engines
can generate different greedy token sequences and speculative acceptance rates;
this is a matched-workload test, not forced identical output or quality evaluation.
M4/M2/M3 had recorded background activity. M5/M2 used macOS 27.0 and Python
3.14.7; M4/M3 used macOS 26.6.2 and Python 3.14.6. Reversed execution order helps
expose drift but does not eliminate it or isolate hardware from OS differences.
Neither pack gains checkpoint, MTP-default or performance certification here.
These peer measurements do not replace the required `mlx_lm.benchmark` baseline
for broader repo-owned inference-stack claims.

- [Full peer tables, sample spread and host caveats](../../benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-20-peer/RESULTS.md)
- [Pinned contract and reproduction commands](../../benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-20-peer/README.md)
- [Raw peer trials](../../benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-20-peer/trials.json)
- [Artifact verifier](../../benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-20-peer/verify.py)
- [Residency policy and diagnostics](../mtp/tiel-prefill-diagnostics.md)
- [Separate AX-only default-server evidence](../../benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-20-default/README.md)
