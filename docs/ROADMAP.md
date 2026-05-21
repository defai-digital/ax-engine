# Roadmap

AX Engine v4.10.0 is the current serving-oriented runtime baseline. It carries
forward the v4.8.0 Apache License 2.0 release line and includes:

- disk-backed L2 prefix cache
- MLA warm-extend correctness
- TurboQuant compressed-decode telemetry
- per-request MLA prefill chunk decisions for GLM-4.7-Flash cold-prefill
  recovery without losing warm-extend snapshot equivalence inside a single
  session

The README summarizes the current product shape; this page carries active
serving direction and evidence gates.

## Serving Runtime Tracks

The next optimization tracks are:

| Track | Direction |
|---|---|
| KV cache memory layout | Paged or block-aligned KV storage, better per-layer locality, fewer KV copies or transposes, and cache reuse between speculative draft and target verification paths |
| Apple unified memory advantage | Zero-copy weight mapping, memory-mapped quantized weights, direct Metal buffer reuse, fewer temporary tensor materializations, and persistent request buffers to improve cold start, TTFT, and memory pressure |
| MoE expert locality optimization | Expert-weight cache scheduling, token grouping by expert, lower dispatch overhead, likely-expert prefetching, router/dispatch fusion, and top-k routing memory-pattern tuning |
| Speculative decoding software tuning | Adaptive n-gram length, dynamic draft windows, acceptance-rate prediction, fallback thresholds, prompt-pattern-aware speculation, and better cache sharing between draft and verify paths |
| Kernel fusion and quantization path | Fused RMSNorm/matmul, attention projection fusion, fused dequant/matmul, group-wise quantization kernels, Apple AMX/Metal mixed paths, and prepacked weight layouts |

## Evidence Gates

Roadmap items become public support claims only after the matching evidence is
checked in and labeled with the right runtime path.

| Claim area | Expected evidence |
|---|---|
| Repo-owned MLX throughput | MLX inference-stack artifacts with matching `mlx_lm.benchmark` rows, prompt-token provenance, AX decode-policy labels, and route identity |
| Long-context behavior | Separate long-context artifacts for cold prefill, decode at depth, startup, concurrency, and prefix reuse |
| Serving behavior | `ax.serving_benchmark.v1` artifacts with TTFT, TPOT, E2E latency, queue delay, throughput, category summaries, and SLO goodput |
| Delegated compatibility | Explicit `mlx_lm_delegated` or `llama_cpp` route-contract artifacts, not AX-owned MLX throughput claims |

See [`BENCHMARKS.md`](BENCHMARKS.md), [`PERFORMANCE.md`](PERFORMANCE.md), and
[`SERVING-BENCHMARKS.md`](SERVING-BENCHMARKS.md) for the benchmark contracts.
