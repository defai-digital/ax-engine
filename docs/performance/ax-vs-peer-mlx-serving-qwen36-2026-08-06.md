# Single-client serving: AX Engine vs peer MLX serving engine (Qwen 3.6)

**Date:** 2026-08-06 · **AX Engine** `6.13.1` · **peer MLX serving engine**
`0.4.3` · **Host:** Apple M5 Max 128 GB · macOS 26.5.2 · MLX 0.32.0

This session measures a streaming OpenAI `/v1/chat/completions` client against
one loaded model per process. It is separate from the multi-model S1 campaign,
MTP peer matrices, and offline direct-generation rows. Do not compare absolute
throughput across those session modes.

| Need | Read |
| --- | --- |
| Root README headline | [README Performance](../../README.md#performance) |
| Full public tables index | [Performance Results](../PERFORMANCE-RESULTS.md#session-mode-single-client-serving-ax-vs-peer-mlx) |
| Checked-in evidence | `benchmarks/results/serving/` · 2026-08-06 M5 Max Qwen 3.6 single-client session |
| Reproduction harness | [`bench_single_client_mlx_serving.py`](../../scripts/bench_single_client_mlx_serving.py) |

Public docs use a generic peer label. The checked-in artifact records the peer
binary version and SHA-256, model snapshot identities, commands, process-audit
metadata, host conditions, and exact runner identity. Raw per-process logs
remain on the benchmark worktree because repository policy excludes `*.log`
files.

## Benchmark contract

Four Qwen 3.6 checkpoints × two nominal prompt targets × two engines × three
repetitions produced **48/48 fixed-length measurements with zero errors**.

| Field | Contract |
| --- | --- |
| Models | Qwen3.6 27B and 35B-A3B, each at 4-bit and 6-bit |
| Request | Streaming chat, temperature 0, top-p 1, top-k 0, deterministic seed |
| Shape | Nominal 512- and 2,048-word prompts; 256 completion tokens |
| Isolation | Fresh server per engine/model/repetition; one 32-token warmup |
| Ordering | Engine-first position balanced by model index and repetition |
| Cooldown | 15 seconds after every process |
| Acceptance | Authoritative streamed usage, 256 completion tokens, non-empty output, `[DONE]`, no request/process error |

The deterministic prompt text and seed are identical across engines. Chat
templates add different wrappers, so authoritative prompt counts differ by five
tokens: 724 vs 729 for the shorter target and 2,869 vs 2,874 for the longer
target.

**Metric definitions**

| Metric | Definition |
| --- | --- |
| TTFT | Client request dispatch → first content or reasoning chunk |
| Effective prefill tok/s | Authoritative prompt-token count ÷ client-observed TTFT |
| Decode tok/s | `(completion_tokens - 1) / (last content chunk - first content chunk)` |
| Delta | `(AX / peer - 1) × 100%`; TTFT speedup uses `(peer / AX - 1) × 100%` |

## Decode throughput

Decode is the headline metric for this single-client session.

| Model | Prompt target | AX Engine | Peer | Delta |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | 512 | **34.40** | 32.32 | **+6.4%** |
| Qwen3.6 27B 4-bit | 2,048 | **33.88** | 32.01 | **+5.9%** |
| Qwen3.6 27B 6-bit | 512 | **24.59** | 23.94 | **+2.7%** |
| Qwen3.6 27B 6-bit | 2,048 | **23.97** | 23.35 | **+2.7%** |
| Qwen3.6 35B-A3B 4-bit | 512 | **159.10** | 129.06 | **+23.3%** |
| Qwen3.6 35B-A3B 4-bit | 2,048 | **156.89** | 126.60 | **+23.9%** |
| Qwen3.6 35B-A3B 6-bit | 512 | **128.79** | 106.67 | **+20.7%** |
| Qwen3.6 35B-A3B 6-bit | 2,048 | **126.90** | 105.04 | **+20.8%** |

AX wins **8 of 8** decode cells. The geometric-mean advantage is **12.9%**
across the matrix, **4.4%** for dense 27B, and **22.2%** for 35B-A3B MoE.
Within-cell decode spread is at most 2.7% across the three repetitions.

## Effective prefill and TTFT

Prefill and TTFT are mixed rather than headline wins: AX leads the four MoE
cells while the peer leads the four dense cells.

| Model | Prompt | AX prefill | Peer prefill | Prefill delta | AX TTFT | Peer TTFT | TTFT speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 27B 4-bit | 512 | 739.4 | 833.2 | -11.3% | 0.979 s | 0.876 s | -10.6% |
| 27B 4-bit | 2,048 | 828.3 | 877.5 | -5.6% | 3.456 s | 3.268 s | -5.4% |
| 27B 6-bit | 512 | 655.3 | 773.8 | -15.3% | 1.111 s | 0.942 s | -15.2% |
| 27B 6-bit | 2,048 | 716.2 | 772.7 | -7.3% | 4.006 s | 3.712 s | -7.4% |
| 35B-A3B 4-bit | 512 | 2,771.8 | 2,552.0 | **+8.6%** | 0.261 s | 0.284 s | **+8.8%** |
| 35B-A3B 4-bit | 2,048 | 3,106.5 | 2,857.8 | **+8.7%** | 0.924 s | 1.004 s | **+8.7%** |
| 35B-A3B 6-bit | 512 | 2,559.8 | 2,302.1 | **+11.2%** | 0.283 s | 0.317 s | **+12.0%** |
| 35B-A3B 6-bit | 2,048 | 2,764.7 | 2,655.5 | **+4.1%** | 1.038 s | 1.080 s | **+4.1%** |

Matrix-wide geometric means are **-1.3%** for effective prefill and **-1.1%**
for TTFT speedup. Those near-zero aggregates hide the dense/MoE split and
should not be presented as a general prefill claim.

## Provenance and acceptance

- The run started at a one-minute load average of 0.64 and ended at 1.24; macOS
  recorded no thermal or performance warning.
- AC power was attached throughout.
- The runner was clean at start and is present in source commit
  `85337d14d4bd0d0dacfae23082a650dfe6e7c0eb`; the artifact also records its
  Git blob and SHA-256.
- All 24 managed server processes exited without a forced kill.
- Every measurement ended by the fixed 256-token length boundary and emitted
  visible content or reasoning content.
- Machine serial, hardware UUID, provisioning identifier, and activation-lock
  fields are excluded from host metadata.

## Claim boundaries

- This is one M5 Max 128 GB host, four Qwen checkpoints, two prompt targets,
  and three repetitions. It is not a cross-hardware claim.
- Effective prefill includes client-observed request and first-chunk overhead;
  it is not a raw kernel-prefill benchmark.
- The two prompt targets share a server process within each
  engine/model/repetition. Prefix text beyond the chat wrapper is
  independently generated, and each repetition starts a fresh process.
- This session checks transport completion and fixed-length generation, not
  semantic quality parity. The peer's 6-bit logs retain its shape-derived
  shared-expert quantization warnings for audit.
- MTP packages can be used to serve these model families, but this session
  disables speculative decoding. Use the separate MTP session for speculative
  speedup claims.
