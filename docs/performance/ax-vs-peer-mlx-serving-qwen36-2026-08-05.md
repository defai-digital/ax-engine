# Single-client serving: AX Engine vs peer MLX serving engine (Qwen 3.6)

**Date:** 2026-08-05 · **AX Engine** `6.13.1` · **peer MLX serving engine**
`0.4.3` · **Host:** Apple M3 Max 128 GB · macOS 26.5.2 · MLX 0.32.0

This session measures **what users actually hit**: a streaming OpenAI
`/v1/chat/completions` client against a single loaded model per process.
It is separate from the multi-model S1 campaign, from MTP peer matrices, and
from offline `mlx_lm.benchmark` direct rows. Do not mix hosts or session modes.

| Need | Read |
| --- | --- |
| Headline strip in the root README | [README Performance](../../README.md#performance) |
| Full public tables index | [Performance Results](../PERFORMANCE-RESULTS.md#session-mode-single-client-serving-ax-vs-peer-mlx) |
| Checked-in artifacts | `benchmarks/results/serving/` · 2026-08-05 Qwen 3.6 single-client peer session (`benchmark.json`, `improvement/`, `review/`) |

Public docs intentionally use the generic peer label required by the docs gate;
the peer binary version and artifact tree are enough to audit the session.

## Why this session exists

Users comparing Mac inference engines often open a server and time chat
completions. Offline decode charts and speculative-MTP peer matrices answer
different questions. This session closes the gap: same models, same prompt
targets, same generation length, two production servers, one client metric
definition.

## Overnight matrix (historical, 3 repetitions)

Four Qwen 3.6 checkpoints × two nominal prompt targets × two engines × three
repetitions (48 measurements, zero request errors). Prompt tokens are
fresh/deterministic per target/repetition (not hash-identical across engines:
chat templates differed by ~5 tokens). Generation: 256 tokens, temperature 0,
streaming.

**Metric definitions**

| Metric | Definition |
| --- | --- |
| TTFT | Client request-send → first content or reasoning chunk |
| Effective prefill tok/s | Authoritative prompt-token count ÷ client-observed TTFT |
| Decode tok/s | `(completion_tokens − 1) / (last content chunk − first content chunk)` |
| Δ% | `(AX / peer − 1) × 100%` for throughput; for TTFT, positive means AX is faster (lower latency) |

### Decode throughput (tok/s, higher is better)

| Model | Prompt target | AX Engine | Peer | Δ% |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | 512 | 19.3 | 18.9 | **+2.1%** |
| Qwen3.6 27B 4-bit | 2048 | 16.7 | 18.8 | −11.2% (see fix below) |
| Qwen3.6 27B 6-bit | 512 | **15.1** | 12.9 | **+17.1%** |
| Qwen3.6 27B 6-bit | 2048 | **14.0** | 12.5 | **+12.0%** |
| Qwen3.6 35B-A3B 4-bit | 512 | **99.4** | 83.2 | **+19.5%** |
| Qwen3.6 35B-A3B 4-bit | 2048 | **97.0** | 81.1 | **+19.6%** |
| Qwen3.6 35B-A3B 6-bit | 512 | **80.9** | 69.5 | **+16.4%** |
| Qwen3.6 35B-A3B 6-bit | 2048 | **80.0** | 68.3 | **+17.1%** |

AX won **7 of 8** overnight decode cells. Geometric-mean decode advantage
across the eight cells: **~11.1%**. MoE (35B-A3B) alone: **~18%** decode GM
advantage.

### Effective prefill and TTFT

| Model | Prompt | AX prefill | Peer prefill | Prefill Δ% | AX TTFT (s) | Peer TTFT (s) | TTFT speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 27B 4-bit | 512 | 180.0 | 163.1 | +10.4% | 3.918 | 4.305 | +9.9% |
| 27B 4-bit | 2048 | 183.4 | 169.1 | +8.5% | 15.514 | 16.788 | +8.2% |
| 27B 6-bit | 512 | 175.3 | 167.4 | +4.7% | 3.971 | 4.242 | +6.8% |
| 27B 6-bit | 2048 | 177.3 | 169.5 | +4.6% | 16.047 | 16.816 | +4.8% |
| 35B-A3B 4-bit | 512 | 1120.3 | 1146.6 | −2.3% | 0.635 | 0.621 | −2.3% |
| 35B-A3B 4-bit | 2048 | 1281.4 | 1202.4 | +6.6% | 2.211 | 2.361 | +6.8% |
| 35B-A3B 6-bit | 512 | 1195.2 | 1103.9 | +8.3% | 0.590 | 0.643 | +9.0% |
| 35B-A3B 6-bit | 2048 | 1338.2 | 1189.0 | +12.5% | 2.126 | 2.401 | +12.9% |

Geometric-mean prefill advantage **~6.6%**; TTFT speedup **~6.9%**. The only
prefill/TTFT dip is 35B-A3B 4-bit at p512 (~2.3%, low confidence at n=3).

## 27B 4-bit long-prompt decode fix (post-overnight)

The overnight loss on **Qwen3.6 27B 4-bit @ ~2k prompt tokens** was isolated to
AX's custom dense-FFN Metal matvec path on that exact geometry
(64 layers, 5120 → 17408 FFN). Controlled route A/B (fixed 2,048-token prompt,
direct greedy, 256 gen tokens):

| Arm | Median decode | Spread | Notes |
| --- | ---: | ---: | --- |
| Custom fused FFN (pre-fix default) | 19.33 tok/s | 16.4% | Tailed under Metal load |
| MLX split quantized-matmul FFN | 22.74 tok/s | 2.9% | Byte-identical output IDs |
| **Patched default** (geometry guard) | **22.54 tok/s** | **3.0%** | 2 warmups + 5 measurements |

Versus the overnight peer p2048 median (**18.8 tok/s**), the patched AX cell is
about **+20%**. Versus the overnight AX p2048 median (**16.7 tok/s**), about
**+35%**. The guard is intentionally narrow (that dense 4-bit geometry only);
smaller Qwen3.5-9B keeps the custom path where it still wins.

Raw route A/B and patched JSON live under the same 2026-08-05 single-client
peer session tree (`improvement/ffn-fused-a.json`, `improvement/ffn-mlx-b.json`,
`improvement/patched-default-final.json`).

## Claim boundaries

- **Host is M3 Max**, not the M5 Max used for S1 multi-model and many MTP/direct
  rows. Absolute tok/s must not be compared across those hosts.
- Overnight matrix: **3 repetitions**, fixed engine order (AX first), short
  cooldowns, non-idle host. Effect sizes are directional; the review treats the
  exact −11.2% cell as exploratory until the controlled A/B.
- Chat templates were not token-identical across engines (~5 prompt tokens).
- Patched 22.54 tok/s is decision-grade **AX-controlled** evidence for the FFN
  geometry fix. The +20% vs overnight peer uses a historical serving
  reference, not a same-process interleaved peer re-run.
- Full alternating-order matrix re-run on a clean release binary remains the
  preferred publication seal for every cell; MoE and 27B 6-bit wins are large
  enough that host noise does not reverse the ranking under the recorded
  protocol.

## Recommended user packages

For AutomatosX serve-ready snapshots that match this generation family:

| Goal | Prefer |
| --- | --- |
| Fastest MoE chat decode (serving) | Qwen 3.6 35B-A3B 4-bit or 6-bit MTP packages |
| Dense 27B chat | Qwen 3.6 27B 6-bit (stable overnight win) or 4-bit after the FFN geometry fix |
| Speculative decode (separate session) | Same AutomatosX `*-MTP` packages — see [MTP results](../PERFORMANCE-RESULTS.md#session-mode-mtp-generation) |

Aliases: `ax-engine download --list` and
[Supported Models](../SUPPORTED-MODELS.md).
