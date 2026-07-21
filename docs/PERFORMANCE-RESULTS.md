# Performance Results

Canonical public **tables and charts** for AX Engine, grouped by session mode.

| Need | Read |
| --- | --- |
| Claim boundaries and promotion rules | [Performance Docs Map](performance/README.md) |
| How to interpret a row / what it does not prove | [Performance](PERFORMANCE.md) |
| How to reproduce or classify evidence | [Benchmarks](BENCHMARKS.md) |
| Root README peer charts (MTP + direct) | [Root README: Performance](../README.md#performance) |

## On this page

- [Session Mode: MTP Generation](#session-mode-mtp-generation)
- [Session Mode: Direct Generation](#session-mode-direct-generation)
- [Session Mode: Embeddings](#session-mode-embeddings)

Do not compare rows across modes unless the text says they share a same-artifact
denominator.

> [!IMPORTANT]
> **MLX runtime admission: verify the resolved build, not only its version.**
> A Homebrew MLX 0.32.0 bottle and source builds with a deployment target below
> macOS 26.2 can omit the M5-class neural-accelerator GEMM path. On the affected
> build, matrix–matrix throughput drops from ~56–61 to ~15 TFLOP/s (fp16 GEMM
> and quantized matmul alike), cutting prefill throughput and inflating TTFT for
> every MLX-based engine; PyPI's MLX 0.32.0 wheel is not affected by that
> packaging defect. Each comparison must use the same resolved `libmlx` on both
> sides and pass the documented `quantized_matmul` admission check. The
> published v6.9.0 sweep was measured on the known-good local MLX 0.31.2 build.
> The PyPI MLX 0.32.0 wheel (deployment target 26.2) has since passed the same
> admission check at throughput parity with 0.31.2 (56.3 TFLOP/s qmm,
> 2026-07-15) and is the admitted build for new benchmark sessions.

**Benchmarking session baseline (14-Jul-2026):** New AX Engine benchmark rows
use AX Engine `v6.9.0`. Direct-mode peer benchmarking is limited to the existing
local `llama.cpp` and `mlx-lm` versions: `llama.cpp` `b9910` / `ggml` `0.15.3`
for GGUF Metal reference rows and `mlx-lm` `0.31.3` for MLX reference rows.
MTP peer benchmarking uses the current local MTPLX release, `MTPLX 2.0.1`.
The v6.8.2 direct high-water composite is retained below as a collapsed
historical audit, not a current headline matrix. v6.9.0 headline tables and
charts are regenerated only from complete 2-warmup/5-measurement sweeps.

Performance results are grouped by **Session mode**. Read each mode as a
separate benchmark session with its own route, workload shape, and headline
metric; do not compare rows across modes unless the text explicitly says they
share a same-artifact denominator.

| Session mode | What it measures | Headline metric | Keep separate from |
| --- | --- | --- | --- |
| MTP generation | Speculative generation with a draft/MTP package plus target verification | MTP decode tok/s, speedup over same-package direct, accept rate | Direct-mode peer rows and embedding ingest rows |
| Direct generation | Non-speculative autoregressive generation through AX, mlx-lm, or llama.cpp routes | Decode tok/s, prefill tok/s, TTFT | MTP speedup rows; diffusion rows call out their own non-AR metric |
| Embeddings | Encoder-style embedding throughput and ingest scale | Chunks/s, tokens/s, latency at batch/chunk settings | Text generation decode/prefill/TTFT |

### Session Mode: MTP Generation

AX Engine supports two MTP packaging contracts in the repo-owned runtime: Qwen
fused sidecars and Gemma assistant drafters. The
cross-engine peer comparison is Qwen-only — Qwen3.6 27B and Qwen3.6 35B-A3B,
each at 4-bit and 6-bit, MTP-only rows — because MTPLX and lightning-mlx ship
comparable Qwen MTP packages but no comparable Gemma one. Gemma 4
assistant-MTP is published separately below as an AX-only depth result, since no
peer engine ships the same package. Same-package direct baselines may be kept as
AX diagnostics, but they are not headline MTP matrix rows.

#### Supported MTP packages

Use `ax-engine download-mtp <target>` for the packages below. These targets are
the supported repo-owned MTP preparation paths; direct-model aliases are listed
separately in [Getting a Model](#getting-a-model).

| Target | Base model | MTP package |
| --- | --- | --- |
| `gemma-4-12b-4bit` | `mlx-community/gemma-4-12B-it-4bit` | Quick-start Gemma assistant-MTP package with `mlx-community/gemma-4-12B-it-assistant-4bit` |
| `qwen3.6-27b-6bit` | `mlx-community/Qwen3.6-27B-6bit` | Qwen fused sidecar from `Qwen/Qwen3.6-27B` |
| `qwen3.6-35b-a3b` | `mlx-community/Qwen3.6-35B-A3B-6bit` | Qwen fused sidecar from `Qwen/Qwen3.6-35B-A3B` |
| `gemma-4-12b` | `mlx-community/gemma-4-12B-it-6bit` | Gemma assistant-MTP package with `mlx-community/gemma-4-12B-it-assistant-6bit` |
| `gemma-4-26b` | `mlx-community/gemma-4-26b-a4b-it-6bit` | Gemma assistant-MTP package with `google/gemma-4-26b-a4b-it-assistant` |
| `gemma-4-31b` | `mlx-community/gemma-4-31b-it-6bit` | Gemma assistant-MTP package with `google/gemma-4-31b-it-assistant` |

The practical AX Engine benchmark lane is the 6-bit `download-mtp` set. The
`gemma-4-12b-4bit` target is kept as the Quick Start path, and Qwen 4-bit
packages are comparison artifacts for peer MTP engines rather than normal
`download-mtp` targets.

#### Download and serve an MTP package

Install via Homebrew (primary), prepare a target, then run the serve command
printed by the CLI.

```bash
brew install defai-digital/ax-engine/ax-engine
ax-engine download-mtp gemma-4-12b-4bit
```

Other supported targets use the same shape:

```bash
ax-engine download-mtp qwen3.6-27b-6bit
```

The command prints the prepared package path and a matching
`ax-engine serve <prepared-mtp-package> --port 31418` command.

By default, packages are written as synthetic Hugging Face cache snapshots under
the active HF cache root. Use `--output <dir>` when you need an explicit
destination, `--force` to rebuild an existing package, and `--json` for scripts.
See [Supported Models](SUPPORTED-MODELS.md#mtp-downloads) and the
[CLI reference](CLI.md#ax-engine) for aliases and advanced options.

#### Benchmark scope

| Target | Preparation / source | Benchmark mode |
| --- | --- | --- |
| `qwen3.6-27b-4bit` | prepared Qwen fused sidecar | Qwen fused sidecar MTP |
| `qwen3.6-27b-6bit` | `ax-engine download-mtp qwen3.6-27b-6bit` | Qwen fused sidecar MTP |
| `qwen3.6-35b-a3b-4bit` | prepared Qwen fused sidecar | Qwen fused sidecar MTP |
| `qwen3.6-35b-a3b` | `ax-engine download-mtp qwen3.6-35b-a3b` | Qwen fused sidecar MTP |
| `gemma-4-12b` | `ax-engine download-mtp gemma-4-12b` | Gemma assistant-MTP |
| `gemma-4-26b` | `ax-engine download-mtp gemma-4-26b` | Gemma assistant-MTP |
| `gemma-4-31b` | `ax-engine download-mtp gemma-4-31b` | Gemma assistant-MTP |

Rules for current MTP benchmark artifacts:

- Use MTP mode for all promoted rows.
- Report decode tok/s, prefill tok/s, TTFT ms, and MTP accept rate.
- Keep unsupported MTPLX, lightning-mlx, Rapid-MLX, or oMLX lanes visible in
  the plan with their support reason.
- Do not run or promote `mtp-ngram` rows.
- Keep the Qwen3.6 peer matrix free of Qwen3-Coder-Next, 5-bit, 8-bit, FFN-only,
  GGUF variants; Gemma 4 assistant-MTP is published as a separate
  AX-only subsection, not mixed into the cross-engine matrix.
- Direct rows are same-artifact denominators for `AX MTP / AX direct` decode
  acceleration, not a cross-model speed leaderboard.
- Keep promoted peer rows on strict AX MTP verification
  (`AX_MLX_MTP_OPTIMISTIC=0`). Optimistic verify is useful for AX-only
  throughput experiments, but it is not a clean peer-comparison default.

The benchmark prompt suites remain `flappy`, `long_code`, and
`python_modules_long`, with sampled decode (`temperature=0.6`, `top_p=0.95`,
`top_k=20`), `1000` generated tokens, `5` measured repetitions, and recorded
cooldown. Current matrix artifacts live under
`benchmarks/results/mtp-qwen36-matrix/`. Every artifact records the exact model
snapshot or peer model id, MTP package provenance where applicable, route
identity, accept rate, prefill throughput, decode throughput, TTFT, sampler,
prompt suite, repetitions, and cooldown.

Plan without running inference:

```bash
python3 scripts/bench_qwen36_mtp_matrix.py
```

Run supported lanes:

```bash
python3 scripts/bench_qwen36_mtp_matrix.py --execute
```

For production-like AX Engine guidance, use the 6-bit lane. The 4-bit lane is
published to make comparison with other MTP engines easier because many peer
benchmarks use 4-bit models. Historical MTP+n-gram artifacts remain useful for
debugging regressions, but they are not current PERFORMANCE-RESULTS / PERFORMANCE MTP evidence.

#### AX Engine v6.9.0 6-bit exact sampled-MTP acceleration (2026-07-16)

This AX Engine-only matrix compares each prepared 6-bit `download-mtp`
package with MTP disabled and enabled. The enabled route uses
distribution-exact sampled MTP with deterministic-delta proposals and
residual rejection correction; it is not an optimistic speed ceiling or a
cross-engine leaderboard.

All 15 target/suite rows accelerate decode by 1.40x-2.69x.
Every row has 100% MTP step coverage, zero direct-fallback prompts or
steps, and zero n-gram accepted, proposed, submitted, or hit-step
telemetry.

<img src="assets/perf-mtp-6bit-ax-acceleration.svg" alt="AX Engine v6.9.0 6-bit exact sampled-MTP acceleration comparing same-package direct and MTP decode throughput">

| Target | Suite | AX direct decode | AX MTP decode | AX speedup | AX MTP prefill | AX MTP TTFT | AX accept |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen3.6-27b-6bit` | `flappy` | 22.7 tok/s | 61.2 tok/s | 2.69x | 546.0 tok/s | 589 ms | 99.3% |
| `qwen3.6-27b-6bit` | `long_code` | 22.7 tok/s | 51.5 tok/s | 2.26x | 672.4 tok/s | 1067 ms | 98.6% |
| `qwen3.6-27b-6bit` | `python_modules_long` | 22.8 tok/s | 42.7 tok/s | 1.88x | 564.7 tok/s | 615 ms | 96.3% |
| `qwen3.6-35b-a3b` | `flappy` | 100.0 tok/s | 143.6 tok/s | 1.44x | 971.2 tok/s | 334 ms | 99.9% |
| `qwen3.6-35b-a3b` | `long_code` | 99.5 tok/s | 139.6 tok/s | 1.40x | 1759.1 tok/s | 408 ms | 98.5% |
| `qwen3.6-35b-a3b` | `python_modules_long` | 100.0 tok/s | 143.9 tok/s | 1.44x | 1054.9 tok/s | 327 ms | 98.4% |
| `gemma-4-12b` | `flappy` | 38.0 tok/s | 96.8 tok/s | 2.55x | 1113.0 tok/s | 313 ms | 99.9% |
| `gemma-4-12b` | `long_code` | 37.8 tok/s | 95.3 tok/s | 2.52x | 1465.8 tok/s | 558 ms | 100.0% |
| `gemma-4-12b` | `python_modules_long` | 38.1 tok/s | 76.2 tok/s | 2.00x | 1123.0 tok/s | 329 ms | 97.8% |
| `gemma-4-26b` | `flappy` | 88.5 tok/s | 146.9 tok/s | 1.66x | 1285.4 tok/s | 275 ms | 99.9% |
| `gemma-4-26b` | `long_code` | 87.6 tok/s | 144.4 tok/s | 1.65x | 2279.6 tok/s | 359 ms | 100.0% |
| `gemma-4-26b` | `python_modules_long` | 88.9 tok/s | 131.7 tok/s | 1.48x | 1358.5 tok/s | 273 ms | 98.4% |
| `gemma-4-31b` | `flappy` | 17.7 tok/s | 45.5 tok/s | 2.58x | 441.6 tok/s | 788 ms | 99.9% |
| `gemma-4-31b` | `long_code` | 18.0 tok/s | 45.4 tok/s | 2.53x | 572.0 tok/s | 1430 ms | 100.0% |
| `gemma-4-31b` | `python_modules_long` | 18.3 tok/s | 40.2 tok/s | 2.20x | 436.7 tok/s | 822 ms | 98.1% |

Methodology: sampled decode (`temperature=0.6`, `top_p=0.95`,
`top_k=20`), 1,000 generated tokens, 2 warmups, 5 measured repetitions,
and recorded cooldown. Prefill and TTFT are reported as context, not MTP
acceleration claims, because speculative decoding starts after prompt
prefill. Direct and MTP rows use the same package and prompt suite.

Exactness is checked with per-mode seed reproducibility. Summary artifacts:
[`summary.md`](benchmarks/results/speculative/mtp-6bit/2026-07-16-v6.9.0-clean-provenance-exact-retry/summary.md) and
[`summary.json`](benchmarks/results/speculative/mtp-6bit/2026-07-16-v6.9.0-clean-provenance-exact-retry/summary.json).

#### Qwen3.6 MTP peer decode comparison (2026-07-09)

This page keeps only the decode-throughput view for the Qwen3.6 MTP peer
comparison because decode is the closest comparable metric across AX Engine,
MTPLX, and lightning-mlx. The full benchmark page explains why prefill, TTFT,
accept rate, seed policy, model-artifact identity, and output-degeneracy checks
need separate interpretation:
[`mtp/qwen36-peer-comparison.md`](mtp/qwen36-peer-comparison.md).

This is a stitched peer comparison, not one interleaved physical-session
benchmark. AX Engine rows were refreshed on the current code; MTPLX rows were
refreshed with MTPLX 2.0.1; lightning-mlx rows are retained from the prior
peer artifacts and called out as retained rows in the stitched chart source.
The 27B 4-bit rows load the same
`ax-local/Qwen3.6-27B-MTP` sidecar across AX Engine, MTPLX, and lightning-mlx;
the 35B-A3B peer rows remain production-configuration rows with the peer
engines' Youssofal MTPLX-optimized packages. The AX 27B 4-bit row uses strict
MTP verification and passes the output-degeneracy gate; older optimistic
artifacts remain useful only as audit/debug evidence.

<img src="assets/perf-mtp-peer-comparison-apples-to-apples.svg" alt="Qwen3.6 MTP peer comparison production-configuration chart showing decode throughput for AX Engine, MTPLX, and lightning-mlx across 27B and 35B 4-bit and 6-bit rows">

| Target | AX Engine decode | MTPLX decode | lightning-mlx decode | Readout |
| --- | ---: | ---: | ---: | --- |
| Qwen3.6 27B 4-bit | 63.0 tok/s | 58.5 tok/s | 55.7 tok/s | Same AX sidecar across all three engines; AX leads this row |
| Qwen3.6 27B 6-bit | 41.8 tok/s | - | - | No official comparable peer 27B 6-bit MTP artifact |
| Qwen3.6 35B-A3B 4-bit | 172.4 tok/s | 137.9 tok/s | 116.2 tok/s | AX leads this production-config row |
| Qwen3.6 35B-A3B 6-bit | 141.2 tok/s | 119.0 tok/s | 96.3 tok/s | AX leads this production-config row |

**27B effective output work (same sidecar):** On the identical 27B dense
sidecar, active bytes match across engines, so output work tracks the decode
ranking and is safe to show as the bar metric. The active-byte value is the
same for every row, so the chart omits that column.

<img width="100%" src="assets/perf-qwen36-mtp-bandwidth-diagnostic.svg" alt="Qwen3.6 27B MTP effective output work same-sidecar chart for AX Engine, MTPLX, and lightning-mlx">

Read output-work percentages above 100% as MTP output leverage, not impossible
memory bandwidth. For the 27B 4-bit rows, each target verifier pass reads about
16.9 GB of weights, but a successful MTP pass can commit several accepted draft
tokens. AX, for example, runs about 16.5 verifier passes/s and emits about
3.8 output tokens/pass, so the physical target-cycle estimate is about
279 GB/s while the output-scaled diagnostic is about 1065 GB/s. The latter is
useful for explaining committed-token work per second, but it is not a claim
that the GPU exceeded the 577 GB/s physical-memory reference.

35B-A3B is intentionally not charted as an output-work diagnostic because the
peer rows are production-configuration MoE package rows with different
active-byte estimates. AX leads that row on the fair speed metric, decode
tok/s; active bytes and output work are retained in the detailed table only as
audit context.

Full results, charts, artifact links, and fairness limitations:
[`mtp/qwen36-peer-comparison.md`](mtp/qwen36-peer-comparison.md).
Stitched chart source:
[`benchmarks/results/mtp-qwen36-matrix/2026-07-09-peer-comparison-apples-to-apples-refresh/summary.json`](benchmarks/results/mtp-qwen36-matrix/2026-07-09-peer-comparison-apples-to-apples-refresh/summary.json).
Decode and output-work diagnostic source:
[`benchmarks/results/mtp-qwen36-matrix/2026-07-09-peer-comparison-apples-to-apples-refresh/bandwidth_diagnostic.json`](benchmarks/results/mtp-qwen36-matrix/2026-07-09-peer-comparison-apples-to-apples-refresh/bandwidth_diagnostic.json).
For the older AX-only Qwen3.6 table across `flappy`, `long_code`, and
`python_modules_long`, see
[`mtp/qwen36-matrix-refresh.md`](mtp/qwen36-matrix-refresh.md). That
table is useful for prompt-suite regression review, but it is not a separate
root-README headline result.

Rapid-MLX is intentionally not promoted in this table: it starts with the
shared Qwen3.6 artifacts but skips MTP installation for this generation flow, so
including it would measure non-MTP decode. oMLX remains unmeasured because this
repo does not yet have an oMLX Qwen3.6 MTP prompt-suite adapter.

#### Gemma 4 assistant-MTP (depth-2)

Gemma 4 speculates with an **assistant-drafter** package, not a Qwen-style fused
sidecar, so no peer engine ships the same Gemma MTP package — MTPLX and
lightning-mlx have no comparable Gemma assistant-MTP route. The result below is
therefore an AX-only comparison (same-artifact direct decode versus depth-2
assistant drafting), not a cross-engine leaderboard. The assistant is stateless
per decode step and re-reads the target's frozen KV cache each forward.

The current AX-only matrix is Gemma 4 12B 4-bit, measured with 2 warmups and 5
repetitions per mode. It uses the same assistant-MTP package for the direct
baseline and depth-2 assistant route, sampled decode (`temperature=0.6`,
`top_p=0.95`, `top_k=20`), and no n-gram stacking. All three suite rows use the
effective MTP-head verify loop without direct fallback; assistant acceptance is
96.8%-98.4%.

| Suite | Assistant accept | AX direct decode | AX assistant-MTP decode | Speedup | AX MTP prefill | AX MTP TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `flappy` | 98.4% | 41.2 tok/s | 83.9 tok/s | 2.04x | 1,115.3 tok/s | 315 ms |
| `long_code` | 98.4% | 40.7 tok/s | 80.9 tok/s | 1.99x | 1,490.4 tok/s | 551 ms |
| `python_modules_long` | 96.8% | 41.1 tok/s | 67.6 tok/s | 1.65x | 1,162.4 tok/s | 316 ms |

Prefill and TTFT are context only: assistant-MTP starts after prompt prefill.
Depth-2 remains the default assistant configuration; set
`AX_MLX_GEMMA4_ASSISTANT_MTP_MAX_DEPTH=1` to restore single-token drafting.
Raw direct and MTP artifacts, route telemetry, and parity checks are in the
[2026-07-16 Gemma 4 12B 4-bit refresh](benchmarks/results/speculative/gemma4-assistant-mtp/2026-07-16-gemma4-12b-4bit-ax-only-refresh/summary.json).

### Session Mode: Direct Generation

Direct generation disables speculative drafting and measures the base
autoregressive route. Where charts are retained, their titles and notes define
the model-specific metric and comparison boundary; these are not MTP
accept-rate or speedup measurements.

#### Direct evidence status

No current-head, matrix-wide direct peer comparison is published. The recent
prefill work produced useful targeted diagnostics, but the final clean paired
sweep did not complete; those numbers are intentionally excluded from the public results tables
and charts.

| Evidence set | Coverage | Public interpretation |
| --- | --- | --- |
| Current HEAD | Incomplete clean paired sweep | No peer-performance claim |
| 2026-07-12 AX-only validation | Gemma 4 E2B/31B and Qwen 3.6 27B 4-bit; one warmup and three measurements | Diagnostic only; [artifacts](benchmarks/results/inference/mlx-inference/2026-07-12-direct-prefill-improve-validate/sweep_summary.md) |
| 2026-07-11 v6.8.2 composite | Gemma 4 and Qwen 3.6, mixed historical sessions | Archived row-level evidence below; not current-head performance |

#### v6.9.0 AX-only direct snapshot (2026-07-14)

This complete AX-only direct sweep is published as a dated snapshot: AX Engine
`v6.9.0`, MLX `0.31.2`, Apple M5 Max, 2 warmups, 5 measurements, and 128
generated tokens at each prompt depth. It was measured from clean commit
`ed483404`; it is **not** a claim about the later current `main` runtime.
No historical `mlx_lm`, llama.cpp, or other peer value is changed; the peer
series below are retained reference artifacts only.

The restored box-and-whisker charts use this AX snapshot with retained historical
`mlx_lm` and llama.cpp rows. They summarize the 7 peer-compatible Gemma
model/quant rows or 4 Qwen rows across 128 / 512 / 2,048 prompt depths. They
are cross-run distribution diagnostics, not exact per-model deltas or a
same-session peer benchmark; the exact AX values are in the table below.

**Gemma 4:**

<img width="100%" src="assets/perf-gemma4-decode-box-whisker.svg" alt="Gemma 4 direct decode box plot comparing AX Engine v6.9.0 snapshot with retained mlx-lm and llama.cpp reference rows">

<img width="100%" src="assets/perf-gemma4-prefill-box-whisker.svg" alt="Gemma 4 direct prefill box plot comparing AX Engine v6.9.0 snapshot with retained mlx-lm and llama.cpp reference rows">

<img width="100%" src="assets/perf-gemma4-ttft-box-whisker.svg" alt="Gemma 4 direct TTFT box plot comparing AX Engine v6.9.0 snapshot with retained mlx-lm and llama.cpp reference rows">

**Qwen 3.6:**

<img width="100%" src="assets/perf-qwen-decode-box-whisker.svg" alt="Qwen 3.6 direct decode box plot comparing AX Engine v6.9.0 snapshot with retained mlx-lm and llama.cpp reference rows">

<img width="100%" src="assets/perf-qwen-prefill-box-whisker.svg" alt="Qwen 3.6 direct prefill box plot comparing AX Engine v6.9.0 snapshot with retained mlx-lm and llama.cpp reference rows">

<img width="100%" src="assets/perf-qwen-ttft-box-whisker.svg" alt="Qwen 3.6 direct TTFT box plot comparing AX Engine v6.9.0 snapshot with retained mlx-lm and llama.cpp reference rows">

| Model | Quant | Decode 128 | Decode 512 | Decode 2K | Prefill 128 | Prefill 512 | Prefill 2K | TTFT 128 | TTFT 512 | TTFT 2K |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 E2B | 4-bit | 231.8 | 227.6 | 216.9 | 2,259.7 | 7,298.5 | 16,754.1 | 56.6 | 70.2 | 122.2 |
| Gemma 4 E2B | 6-bit | 187.8 | 181.3 | 174.5 | 1,977.1 | 6,620.4 | 15,315.0 | 64.7 | 77.3 | 133.7 |
| Gemma 4 E4B | 4-bit | 145.2 | 142.2 | 138.8 | 1,621.7 | 4,214.2 | 7,442.3 | 78.9 | 121.5 | 275.2 |
| Gemma 4 E4B | 6-bit | 112.1 | 110.3 | 108.1 | 1,398.7 | 3,783.8 | 6,842.4 | 91.5 | 135.3 | 299.3 |
| Gemma 4 26B A4B | 4-bit | 145.4 | 141.4 | 135.2 | 662.5 | 1,947.0 | 3,839.8 | 193.2 | 263.0 | 533.4 |
| Gemma 4 26B A4B | 6-bit | 113.4 | 110.9 | 107.0 | 505.5 | 1,579.5 | 3,333.2 | 253.2 | 324.2 | 614.4 |
| Gemma 4 31B | 4-bit | 29.2 | 28.7 | 27.5 | 312.8 | 596.1 | 734.4 | 409.2 | 858.9 | 2,788.8 |
| Gemma 4 31B | 6-bit | 20.4 | 20.0 | 18.9 | 224.8 | 479.2 | 641.8 | 569.4 | 1,068.4 | 3,191.1 |
| Qwen 3.6 27B | 4-bit | 35.0 | 34.9 | 34.5 | 415.6 | 730.2 | 899.4 | 308.0 | 701.2 | 2,277.2 |
| Qwen 3.6 27B | 6-bit | 25.4 | 25.4 | 25.0 | 294.1 | 595.1 | 800.0 | 435.2 | 860.4 | 2,560.1 |
| Qwen 3.6 35B A3B | 4-bit | 155.8 | 156.5 | 153.7 | 550.8 | 1,612.3 | 3,021.3 | 232.4 | 317.6 | 677.9 |
| Qwen 3.6 35B A3B | 6-bit | 125.9 | 124.0 | 122.6 | 413.7 | 1,326.6 | 2,672.0 | 309.4 | 385.9 | 766.5 |

Decode and prefill values are tok/s; TTFT values are ms. Full raw results:
[`sweep_results.json`](benchmarks/results/inference/ax-direct/2026-07-14-v6.9.0-ax-direct-only/sweep_results.json).

<!-- readme-ax-direct-snapshot: benchmarks/results/inference/ax-direct/2026-07-14-v6.9.0-ax-direct-only/sweep_results.json -->

#### Gemma 4 12B retained v6.8.2 case study

Gemma 4 12B (`model_type: gemma4_unified`) is reported separately from the per-layer-embedding
E2B/E4B and MoE 26B/31B checkpoints because it has a distinct graph, multimodal tensor contract,
and benchmark boundary. **Upstream `mlx_lm` 0.31.3 cannot load it**
(`ValueError: Model type gemma4_unified not supported`), so the direct peer here is
**llama.cpp Metal** on a shape-compatible GGUF.

This retained comparison measures AX Engine v6.8.2 rather than current HEAD.
Its cross-run, shape-compatible peer boundary is useful as a case study, but it
does not establish a current matrix-wide prefill or TTFT lead.

> [!NOTE]
> **AX Engine's repo-owned native MLX route supports Gemma 4 12B text plus inline base64
> image/audio/video chat.** Delegated compatibility routes remain text-first;
> `/v1/generate` accepts the processed `multimodal_inputs.gemma4_unified` tensor contract.

**At a glance:**

- **Direct decode:** AX native MLX reaches **65.2-69.2 tok/s** on the bit-comparable
  4-bit-FFN artifact versus llama.cpp Metal's **56.9-58.7 tok/s** depth-matched range.
- **Context depth:** AX's direct margin is **+21% / +15% / +14%** versus llama.cpp matched-depth decode at 128 / 512 / 2,048 prompt tokens.
- **Assistant-MTP:** On 12B real prompt suites, depth-2 assistant-MTP reaches
  **88.4-94.5 tok/s**, a **1.56-1.68x** speedup over same-artifact direct decode.
- **Why the earlier result flipped:** the upstream MLX snapshot keeps FFN weights at
  8-bit, so it reads about **1.65x** the bytes of the re-quantized 4-bit-FFN artifact.
  Decode is bandwidth-bound; matching quantization closes the gap.

**Direct decode peer comparison:**

AX direct rows use the 4-bit-FFN MLX artifact and random-token prompts. `mlx_lm` is absent
because it has no `gemma4_unified` graph. The llama.cpp rows are shape-compatible external
GGUF references, not prompt-hash-parity MLX rows.

<p>
<strong>Decode rate</strong><br>
<img width="100%"
  src="assets/perf-gemma4-12b-direct-decode-tok-s.svg"
  alt="Gemma 4 12B direct decode throughput: AX MLX vs llama.cpp Metal">
</p>

<p>
<strong>Prefill rate</strong><br>
<img width="100%"
  src="assets/perf-gemma4-12b-direct-prefill-tok-s.svg"
  alt="Gemma 4 12B direct prefill throughput: AX MLX vs llama.cpp Metal">
</p>

<p>
<strong>TTFT</strong><br>
<img width="100%"
  src="assets/perf-gemma4-12b-direct-ttft-ms.svg"
  alt="Gemma 4 12B direct time to first token: AX MLX vs llama.cpp Metal">
</p>

| Prompt tokens | AX decode | llama.cpp decode (depth 0) | llama.cpp decode (matched depth) | AX prefill | llama.cpp prefill | AX TTFT (ms) | llama.cpp TTFT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 69.2 | 57.1 | 56.9 | 1,184 | 1,245 | 108 | 103 |
| 512 | 67.5 | 57.3 | 58.7 | 1,867 | 1,740 | 274 | 294 |
| 2048 | 65.2 | 56.0 | 57.5 | 2,049 | 1,544 | 999 | 1,327 |

Read the two llama.cpp decode columns carefully:

- `depth 0` is plain `llama-bench tg`, decoding from an empty context and representing llama.cpp's best case.
- `matched depth` uses `-d {prompt} -n 128`, so decode happens after the same prompt depth AX has already prefetched.
- AX wins the matched-depth comparison at every prompt size, and prefill also leads at 512 and 2,048 tokens.

The table uses the bit-comparable **4-bit-FFN** AX artifact
(`scripts/requantize_gemma4_12b_ffn_4bit.py`), about 4.5 bpw versus the Q4_K_M GGUF's
about 4.8 bpw. The upstream `mlx-community/gemma-4-12B-it-4bit` snapshot keeps the FFN at
**8-bit** (~10.98 GB) and trails llama.cpp at about 46 tok/s. That is a bytes-read handicap,
not an AX runtime result.

**Memory bandwidth diagnostic:**

This chart is a diagnostic for the Gemma 4 12B quantization story, not a
recommendation to use an 8-bit tier. The "upstream artifact" row is included
only because the public `mlx-community/gemma-4-12B-it-4bit` snapshot keeps FFN
tensors at 8-bit; it explains the older slower AX result. Decode is
memory-bandwidth-bound on Apple Silicon: each token reads the model weights
once, so decode tok/s is set by bytes-read and how close the engine gets to the
memory ceiling. Measured M5 Max GPU peak read bandwidth ≈ 577 GB/s (MLX
reduction over a 6 GB array).

<img src="assets/perf-gemma4-12b-bandwidth.svg" alt="Gemma 4 12B effective decode bandwidth vs theoretical peak for AX and llama.cpp">

| Engine / quantization | Weights/token | Decode tok/s | Effective BW | % of 577 GB/s peak |
| --- | ---: | ---: | ---: | ---: |
| AX upstream artifact — 8-bit FFN diagnostic | 10.98 GB | 45.4 | 498 GB/s | 86% |
| AX re-quantized artifact — 4-bit FFN | 6.74 GB | 67.5 | 455 GB/s | 79% |
| llama.cpp Q4_K_M — decode @ depth 512 | 7.38 GB | 58.7 | 433 GB/s | 75% |
| llama.cpp Q4_K_M — decode @ depth 0 (`tg`) | 7.38 GB | 57.1 | 421 GB/s | 73% |

The bandwidth view is the key explanation: AX is not under-utilizing memory. The re-quantized
AX row sustains **455 GB/s**, in the same band as llama.cpp's **433 GB/s** at matched depth.
The remaining direct-decode difference is bytes read per token: uniform 4-bit group-64 reduces
AX to **6.74 GB/token**, while Q4_K_M reads **7.38 GB/token**. The upstream artifact
has higher bus utilization (86%) but worse speed because its FFN tensors read far more data.

**Methodology and artifacts:**

Direct rows use the 4-bit-FFN artifact, greedy-equivalent sampler, 128 generated tokens,
5 repetitions, 15 s cooldown, and random-token prompts following the `mlx_lm.benchmark`
contract. llama.cpp decode is shown both at depth 0 (`tg`) and at matched context depth
(`-d {prompt}`). Host/runtime for this retained Gemma 4 12B llama.cpp reference:
Apple M5 Max · llama.cpp b9820 / ggml 0.15.3 (Metal, flash-attn) · `mlx_lm`
0.31.3 has no `gemma4_unified` support. The archived direct-mode high-water
composite uses the later b9910 llama.cpp sweep described in the provenance
block below. MTP methodology and artifacts live with
[Session Mode: MTP Generation](#session-mode-mtp-generation).

The llama.cpp peer columns are measured on llama.cpp b9820 / ggml 0.15.3; full per-prompt
llama.cpp data is in the verification artifact
[`gemma-4-12b-it-4bit-b9820-verify.json`](benchmarks/results/inference/llama-cpp-metal/2026-06-27-llama-only-rerun/gemma-4-12b-it-4bit-b9820-verify.json).
The AX rows come from the current direct-only AX artifact below. The llama.cpp rows are retained
from the earlier peer rerun, so these columns are a shape-compatible cross-run comparison, not a
single-session A/B.

Full artifacts:
[`2026-07-04-gemma4-12b-ax-direct-mtp-refresh`](benchmarks/results/inference/mlx-inference/2026-07-04-gemma4-12b-ax-direct-mtp-refresh/gemma-4-12b-it-4bit-direct.json)
(AX direct rerun; chart artifact with retained llama.cpp reference rows in
[`gemma-4-12b-it-4bit-with-llama-reference.json`](benchmarks/results/inference/mlx-inference/2026-07-04-gemma4-12b-ax-direct-mtp-refresh/gemma-4-12b-it-4bit-with-llama-reference.json);
llama.cpp GGUF provenance in
[`llama_cpp_gguf_provenance.json`](benchmarks/results/inference/mlx-inference/2026-06-26-gemma4-12b-4bit-ax-direct-only/llama_cpp_gguf_provenance.json)).
The upstream 8-bit-FFN bandwidth row is backed by
[`2026-06-26-gemma4-12b-upstream-8bit-ffn-ax-direct-only`](benchmarks/results/inference/mlx-inference/2026-06-26-gemma4-12b-upstream-8bit-ffn-ax-direct-only/gemma-4-12b-it-4bit.json).

Gemma 4 12B multimodal benchmark details now live in
[Benchmarks](BENCHMARKS.md#gemma-4-12b-multimodal-benchmark).

Gemma assistant-MTP package layout and cache-location details live in
[Supported Models](SUPPORTED-MODELS.md#mtp-downloads).

#### DiffusionGemma

DiffusionGemma (`mlx-community/diffusiongemma-26B-A4B-it-4bit`, `model_type:
diffusion_gemma`) is an **experimental** repo-owned MLX path. It is a
26B-A4B MoE Gemma 4 checkpoint that generates by **block diffusion** rather than
autoregressive next-token decoding: each visible output comes from a 256-token
canvas that is denoised bidirectionally and then committed with a causal pass.

> [!WARNING]
> DiffusionGemma is **not recommended for production use**. It is published as
> an architecture preview and benchmarking curiosity. The autoregressive paths
> above (Gemma 4, Qwen 3.6, GLM 4.7) are the supported production routes.

> [!IMPORTANT]
> These are **not** the same metric as the autoregressive rows above. For a
> next-token decoder, `decode tok/s` is the steady token-by-token loop and
> `TTFT` is prefill plus one token. DiffusionGemma has neither, so the columns
> below report **first-block decode** (`256 / block wall time`) and **time to
> first committed block** (prefill wall plus the first denoise-and-commit
> block). Do not read them as directly comparable to the Gemma 4 12B or
> Gemma 4 / Qwen 3.6 AR throughput.

The rows are **AX Engine only**. No peer engine loads this architecture in a
released build: `mlx_lm` 0.31.3 rejects `Model type diffusion_gemma not
supported`, and stable `llama.cpp` Metal fails with `unknown model
architecture: 'diffusion-gemma'`. An unmerged llama.cpp draft PR adds the
architecture, but a draft branch is not a stable baseline, so no peer row is
published here.

<p>
<strong>First-block decode rate</strong><br>
<img width="100%"
  src="assets/perf-diffusiongemma-direct-decode-tok-s.svg"
  alt="AX direct DiffusionGemma first-block decode throughput at 128/512/2048 prompt tokens">
</p>

<p>
<strong>Prefill rate</strong><br>
<img width="100%"
  src="assets/perf-diffusiongemma-direct-prefill-tok-s.svg"
  alt="AX direct DiffusionGemma prefill throughput at 128/512/2048 prompt tokens">
</p>

<p>
<strong>Time to first block</strong><br>
<img width="100%"
  src="assets/perf-diffusiongemma-direct-ttft-ms.svg"
  alt="AX direct DiffusionGemma time to first committed block at 128/512/2048 prompt tokens">
</p>

| Prompt tokens | AX first-block decode | AX prefill | AX time to first block | Denoise steps | Committed block |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 158.9 tok/s | 1,151.0 tok/s | 1,723 ms | 12 | 256 tokens |
| 512 | 109.6 tok/s | 2,794.0 tok/s | 2,520 ms | 17 | 256 tokens |
| 2048 | 163.5 tok/s | 3,922.3 tok/s | 2,089 ms | 11 | 256 tokens |

The 2026-07-08 refresh is faster than the prior 2026-07-05 results artifact on
all published DiffusionGemma metrics: first-block decode improved by
**+7.5% / +5.1% / +16.7%** at 128 / 512 / 2048 prompt tokens, time to first
block dropped by **7.0% / 4.8% / 11.4%**, and prefill rose by
**+8.1% / +5.4% / +1.2%**. The main decode win comes from stopping earlier
under the 7.5% adaptive update-rate threshold, reducing denoise work by
1 / 1 / 2 steps.

First-block decode does not scale cleanly with prompt length because the
denoiser is **convergence-gated**: it iterates until the 256-token canvas
stabilises (11-17 steps here on realistic in-distribution prompts), so
throughput tracks how many denoise passes convergence needs, not prompt size.
Random-token prompts never converge, hit the step cap, and measure the failure
mode instead — these rows use prefixes of a coherent technical document
tokenized with the model's own tokenizer.

A block-granularity weight-traffic estimate puts this path at roughly
**21% of the M5 Max ~614 GB/s theoretical bandwidth**, i.e. it is **not**
memory-bandwidth-saturated: the diffusion denoise step is a parallel
whole-canvas matmul, so it is dispatch-, occupancy-, and kernel-mix-bound rather
than weight-streaming-bound. Method, convergence signals, optimization toggles,
and the bandwidth diagnostic live in
[`DIFFUSIONGEMMA.md`](DIFFUSIONGEMMA.md); full artifact:
[`2026-07-08-acceptance-075-first-block/summary.json`](benchmarks/results/inference/diffusion-gemma-direct/2026-07-08-acceptance-075-first-block/summary.json)
(release build, 1 warmup + 5 measured repetitions, 15 s cooldown, medians).

<!-- readme-performance-artifacts: reference=benchmarks/results/inference/mlx-inference/2026-05-26-direct-mode-clean-refresh/; reference=benchmarks/results/inference/mlx-inference/2026-06-26-qwen36-direct-refresh/; reference=benchmarks/results/inference/mlx-inference/2026-06-26-gemma4-6bit-mlx-lm-only/; reference=benchmarks/results/inference/mlx-inference/2026-07-02-gemma4-6bit-direct-refresh/; ax-overlay=benchmarks/results/inference/mlx-inference/2026-07-11-ax-direct-only-v6.8.2-readme/ -->

#### Gemma 4 and Qwen 3.6 historical direct evidence

The July 2026 v6.8.2 comparison is retained only as an audit trail. It is a
cross-run **high-water composite**, not a current-head or same-session matrix:
AX cells may come from different clean runs by metric, while peer cells come
from older reference runs. `llama.cpp Metal*` is shape-compatible context and
does not have prompt-hash parity with the MLX rows.

The exact row-level tables show the result honestly: direct decode generally
leads `mlx_lm`, while prefill and runner-time TTFT are mixed and are often
materially worse, especially for longer prompts. No matrix-wide prefill or
TTFT lead is claimed.

The six restored family box-and-whisker charts above deliberately keep the
same mixed-size/quantization distribution shape as the earlier presentation,
but now label the limitation directly: they compare the v6.9 AX snapshot to
retained cross-run peer rows. Use the exact table for a model-specific reading;
do not infer prompt-hash-parity deltas from a family-level box median.

> **`llama.cpp Metal*` column** — Shape-compatible reference produced by Metal-enabled `llama-bench`. `llama-bench` generates its own internal synthetic prompt tokens and does not consume the harness prompt JSON, so these numbers are **not** prompt-hash parity with the other columns. No percentage delta is shown. MLX bit-widths are mapped to the nearest Unsloth GGUF quant (4→Q4_K_M, 6→Q6_K), with explicit UD-* Unsloth Dynamic rows only when no standard root-level K-quant is published. Source: `benchmarks/manifests/llama_cpp_metal/inventory.json`, `scripts/bench_llama_cpp_metal_sweep.py`.

The AX direct cells use the 2026-07-11 clean sweep in
`benchmarks/results/inference/mlx-inference/2026-07-11-ax-direct-only-v6.8.2-clean/`
from clean commit `9c9db594`, plus the provenance-listed historical overlays
detailed inside the archive, with generation=128 and the 2-warmup,
5-measurement contract.

<details>
<summary>Historical row-level tables and provenance</summary>

The canonical published AX source projection for this refresh is
`benchmarks/results/inference/mlx-inference/2026-07-11-ax-direct-only-v6.8.2-readme/`;
its symlinks point to raw AX-only artifacts and preserve historical sources for
rows not replaced by the 2026-07-11 sweep.

The `mlx_lm` reference rows for the Gemma 4 rows shown below come from `benchmarks/results/inference/mlx-inference/2026-05-26-direct-mode-clean-refresh/`. The refreshed Gemma 4 4-bit AX direct-mode cells come from `benchmarks/results/inference/mlx-inference/2026-07-01-ax-direct-4bit-refresh-clean-r2/`, which reran AX Engine only for Gemma 4 E2B/E4B/26B/31B at 128/512/2048 prompt tokens with 5 repetitions, 1 warmup, and a 15 s cooldown. Those artifacts record benchmark build commit `d4c59ffc` and `git_tracked_dirty: false`. The Gemma 4 26B A4B 4-bit decode cells are raised by the AX-only 2026-07-07 refresh in `benchmarks/results/inference/mlx-inference/2026-07-07-gemma4-26b-4bit-ax-direct-refresh-gen128/`, a clean `ax-engine-server` build at commit `194a235a` with 2 warmups, 5 measured repetitions, generation=128, and a 15 s cooldown; publication high-water merging only publishes the decode medians from that overlay because its prefill and TTFT medians do not beat the earlier 2026-07-01 record. The Gemma 4 26B A4B and 31B 6-bit rows (both the `mlx_lm` and AX cells) come from the same-session paired rerun in `benchmarks/results/inference/mlx-inference/2026-07-02-gemma4-6bit-direct-refresh/`, which ran `mlx_lm.benchmark` and AX Engine back-to-back per model on a clean build at commit `4c0a8358` with the same 5-repetition, 1-warmup, 15 s-cooldown contract; the earlier `mlx_lm`-only 6-bit spot rows in `benchmarks/results/inference/mlx-inference/2026-06-26-gemma4-6bit-mlx-lm-only/` are retained as historical reference. The Gemma 4 E2B/E4B 6-bit AX cells come from the ax-engine-only rerun in `benchmarks/results/inference/mlx-inference/2026-07-05-gemma4-e2b-e4b-6bit-ax-refresh-r2/`, a clean `ax-engine-server` build at commit `6f2e6cd7` with the same 5-repetition, 1-warmup, 15 s-cooldown contract; these rows are AX-only because `mlx-lm` 0.31.3 cannot strict-load either E-series 6-bit checkpoint (see the shared-KV note below), so the E2B 6-bit `mlx_lm` reference cells are retained from the 2026-05-26 refresh. The Qwen 3.6 `mlx_lm` reference rows come from `benchmarks/results/inference/mlx-inference/2026-06-26-qwen36-direct-refresh/`; the published Qwen 3.6 AX cells combine earlier high-water overlays with the AX-only 2026-07-07 refresh in `benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-only-record-refresh-qwen-publishable/`, a clean `ax-engine-server` build at commit `f73f1ac2` with 2 warmups, 5 measured repetitions, and a 15 s cooldown. The 2026-07-07 Qwen overlay contains condition-checked 4/6-bit rerun rows, but publication high-water merging only publishes cells that match or improve the prior record; lower rerun cells keep the earlier faster artifact. The overlay replaces the original 35B-A3B 6-bit continuation row because its recorded load average exceeded the publication limit. Install documentation and package versions can advance beyond these historical benchmark snapshots; each benchmark artifact's `build.commit` records the exact measured build SHA. The `llama.cpp Metal*` column is injected from `benchmarks/manifests/llama_cpp_metal/inventory.json` and the full llama.cpp-only rerun in `benchmarks/results/inference/llama-cpp-metal/2026-07-08-llama-cpp-only-rerun/`, which reran all 12 Gemma 4 + Qwen 3.6 rows (llama.cpp b9910, Metal, flash-attn, `-b/-ub` matched to prompt length, decode measured at matched context depth).

Gemma 4 E4B 6-bit keeps the `mlx_lm` cells blank because `mlx_lm.benchmark` cannot load `mlx-community/gemma-4-e4b-it-6bit` with `mlx-lm` 0.31.3. The checkpoint config declares 42 language layers and `num_kv_shared_layers=18`, so the upstream Gemma4 text model builds K/V projections only for layers 0..23 and treats layers 24..41 as shared-KV layers. The MLX snapshot still contains 126 per-layer K/V tensors for layers 24..41 (`k_norm`, `k_proj`, and `v_proj` quantized weights), causing strict weight loading to fail with `Received 126 parameters not in model`. Source: `benchmarks/results/inference/mlx-inference/2026-06-26-gemma4-6bit-mlx-lm-only/summary.md`. The same strict-load failure now applies to `mlx-community/gemma-4-e2b-it-6bit` on `mlx-lm` 0.31.3 (140 extra tensors for shared-KV layers 15..34), which is why the 2026-07-05 E-series 6-bit refresh is AX-only and the E2B `mlx_lm` cells remain the retained 2026-05-26 measurements.

Setup: generation=128, 5 measured repetitions, 15-second cooldown, AX prefix
cache disabled for cold prefill and TTFT measurement, production-build binaries,
matching prompt SHA checks. AX prefill and TTFT cells are runner-time
measurements for the model work boundary, not end-to-end client-wall latency.
Long-greedy AX prefill rows are runner-time measurements of the cache-state
prefix plus final prompt-token boundary — not full-logits prompt scoring
throughput. Percentages are versus `mlx_lm`.

The 2K `llama.cpp Metal*` prefill rows are long-context, GGUF-runtime-reference rows, produced with llama.cpp b9910 (Metal offload, `-b/-ub` matched to prompt length up to 2048, flash attention enabled). This is our benchmark boundary, not an upstream llama.cpp official bug statement.

Qwen 3.6 direct-mode decode verdict: AX is faster against `mlx_lm` in every refreshed 27B and 35B-A3B 4/6-bit cell. The 35B-A3B margins are large throughout; the dense 27B margins are roughly +5% across prompt lengths in this refresh.

#### Prefill throughput (archived v6.8.2 composite; tok/s)

| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax engine |
| --- | --- | ---: | ---: | ---: | ---: |
| Gemma 4 E2B | 4-bit | 128 | 3,729.7 | 2,338.1 | 2,212.5 (-5.4%) |
|  |  | 512 | 7,095.0 | 7,870.0 | 5,004.3 (-36.4%) |
|  |  | 2048 | 7,136.7 | 18,014.7 | 7,475.7 (-58.5%) |
| Gemma 4 E2B | 6-bit | 128 | 3,612.1 | 1,823.5 | 1,821.2 (-0.1%) |
|  |  | 512 | 7,071.5 | 6,046.6 | 4,505.1 (-25.5%) |
|  |  | 2048 | 7,247.1 | 15,332.1 | 7,186.2 (-53.1%) |
| Gemma 4 E4B | 4-bit | 128 | 2,285.2 | 1,513.2 | **3,405.1 (+125.0%)** |
|  |  | 512 | 4,173.1 | 4,195.5 | **6,934.9 (+65.3%)** |
|  |  | 2048 | 4,197.6 | 7,325.4 | **8,739.7 (+19.3%)** |
| Gemma 4 E4B | 6-bit | 128 | 2,287.0 | — | 1,026.9 |
|  |  | 512 | 4,241.3 | — | 1,907.3 |
|  |  | 2048 | 4,209.0 | — | 2,414.3 |
| Gemma 4 26B A4B | 4-bit | 128 | 1,888.8 | 496.4 | **599.9 (+20.8%)** |
|  |  | 512 | 3,439.4 | 1,621.0 | 1,211.2 (-25.3%) |
|  |  | 2048 | 3,524.3 | 3,300.1 | 1,553.4 (-52.9%) |
| Gemma 4 26B A4B | 6-bit | 128 | 1,688.1 | 574.6 | 515.9 (-10.2%) |
|  |  | 512 | 3,123.4 | 1,729.8 | 1,112.3 (-35.7%) |
|  |  | 2048 | 3,347.8 | 3,411.2 | 1,483.2 (-56.5%) |
| Gemma 4 31B | 4-bit | 128 | 531.0 | 283.1 | 158.6 (-44.0%) |
|  |  | 512 | 667.4 | 619.9 | 204.1 (-67.1%) |
|  |  | 2048 | 579.6 | 733.9 | 210.0 (-71.4%) |
| Gemma 4 31B | 6-bit | 128 | 501.4 | 280.1 | 141.0 (-49.6%) |
|  |  | 512 | 657.8 | 541.7 | 194.2 (-64.2%) |
|  |  | 2048 | 568.6 | 677.4 | 205.4 (-69.7%) |
| Qwen 3.6 27B | 4-bit | 128 | 546.1 | 424.7 | 194.9 (-54.1%) |
|  |  | 512 | 763.6 | 739.0 | 245.9 (-66.7%) |
|  |  | 2048 | 673.4 | 914.9 | 261.4 (-71.4%) |
| Qwen 3.6 27B | 6-bit | 128 | 533.4 | 348.0 | 179.6 (-48.4%) |
|  |  | 512 | 750.4 | 655.1 | 237.3 (-63.8%) |
|  |  | 2048 | 649.9 | 832.1 | 255.7 (-69.3%) |
| Qwen 3.6 35B A3B | 4-bit | 128 | 1,730.5 | 562.4 | 505.3 (-10.1%) |
|  |  | 512 | 3,127.3 | 1,613.6 | 1,173.7 (-27.3%) |
|  |  | 2048 | 3,508.9 | 3,455.1 | 1,695.9 (-50.9%) |
| Qwen 3.6 35B A3B | 6-bit | 128 | 1,602.8 | 431.6 | 396.3 (-8.2%) |
|  |  | 512 | 2,921.1 | 1,394.4 | 1,035.6 (-25.7%) |
|  |  | 2048 | 3,348.1 | 2,494.3 | 1,577.0 (-36.8%) |

#### Decode throughput (archived v6.8.2 composite; tok/s)

| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax direct baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| Gemma 4 E2B | 4-bit | 128 | 161.0 | 214.0 | **220.5 (+3.1%)** |
|  |  | 512 | 161.5 | 210.3 | **210.5 (+0.1%)** |
|  |  | 2048 | 156.2 | 200.9 | **209.3 (+4.2%)** |
| Gemma 4 E2B | 6-bit | 128 | 142.7 | 172.2 | **175.5 (+2.0%)** |
|  |  | 512 | 141.2 | 166.3 | **169.3 (+1.8%)** |
|  |  | 2048 | 137.3 | 162.5 | **168.6 (+3.8%)** |
| Gemma 4 E4B | 4-bit | 128 | 104.5 | 137.1 | **144.1 (+5.1%)** |
|  |  | 512 | 103.4 | 133.6 | **141.1 (+5.6%)** |
|  |  | 2048 | 101.6 | 130.6 | **138.1 (+5.8%)** |
| Gemma 4 E4B | 6-bit | 128 | 87.8 | — | 106.7 |
|  |  | 512 | 87.2 | — | 104.9 |
|  |  | 2048 | 85.4 | — | 102.8 |
| Gemma 4 26B A4B | 4-bit | 128 | 94.3 | 127.9 | **136.3 (+6.5%)** |
|  |  | 512 | 93.2 | 125.0 | **132.9 (+6.3%)** |
|  |  | 2048 | 89.2 | 119.3 | **127.8 (+7.1%)** |
| Gemma 4 26B A4B | 6-bit | 128 | 88.6 | 104.4 | **111.2 (+6.5%)** |
|  |  | 512 | 88.0 | 101.2 | **108.7 (+7.4%)** |
|  |  | 2048 | 86.0 | 96.8 | **104.9 (+8.4%)** |
| Gemma 4 31B | 4-bit | 128 | 25.9 | 28.9 | **29.3 (+1.4%)** |
|  |  | 512 | 26.1 | 28.3 | **28.7 (+1.5%)** |
|  |  | 2048 | 24.8 | 27.0 | **27.3 (+1.0%)** |
| Gemma 4 31B | 6-bit | 128 | 19.8 | 19.5 | **20.4 (+4.7%)** |
|  |  | 512 | 19.6 | 19.1 | **20.1 (+5.3%)** |
|  |  | 2048 | 19.1 | 18.5 | **19.1 (+3.7%)** |
| Qwen 3.6 27B | 4-bit | 128 | 27.6 | 33.2 | **35.0 (+5.4%)** |
|  |  | 512 | 27.6 | 33.1 | **34.9 (+5.4%)** |
|  |  | 2048 | 27.0 | 32.6 | **34.4 (+5.5%)** |
| Qwen 3.6 27B | 6-bit | 128 | 21.8 | 24.3 | **25.5 (+4.9%)** |
|  |  | 512 | 21.7 | 24.3 | **25.4 (+4.8%)** |
|  |  | 2048 | 21.5 | 24.1 | **25.3 (+4.9%)** |
| Qwen 3.6 35B A3B | 4-bit | 128 | 91.5 | 129.7 | **155.0 (+19.5%)** |
|  |  | 512 | 91.4 | 128.3 | **153.0 (+19.3%)** |
|  |  | 2048 | 89.8 | 125.2 | **148.9 (+19.0%)** |
| Qwen 3.6 35B A3B | 6-bit | 128 | 89.6 | 111.3 | **123.3 (+10.7%)** |
|  |  | 512 | 89.0 | 110.4 | **122.2 (+10.7%)** |
|  |  | 2048 | 88.2 | 105.8 | **120.4 (+13.8%)** |

> Qwen 3.6 27B 4-bit at prompt=2,048 originally produced zero decode tokens because 4-bit quantization noise pushed an EOS token to argmax at decode step 0 on the `mlx_lm.benchmark` random-token contract. The benchmark harness now sends `sampling.ignore_eos=true` for AX throughput runs, matching how `mlx_lm.benchmark` measures fixed `gen=N` throughput. Production requests default to `ignore_eos=false`. Source: `benchmarks/results/inference/mlx-inference/2026-05-20-qwen27-4to5-direct-ngram-directcpp-r2/qwen3_6-27b-4bit.json`.

#### Time to first token (archived v6.8.2 composite; ms)

**Lower is better.** `mlx_lm` values are derived from reported prefill throughput. AX values are measured directly from per-step runner timing in the SSE event stream.

| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax engine |
| --- | --- | ---: | ---: | ---: | ---: |
| Gemma 4 E2B | 4-bit | 128 | 34.3 | 54.7 | 57.9 (+5.7%) |
|  |  | 512 | 72.2 | 65.1 | 102.3 (+57.3%) |
|  |  | 2048 | 287.0 | 113.7 | 274.0 (+141.0%) |
| Gemma 4 E2B | 6-bit | 128 | 35.4 | 70.2 | 70.3 (+0.1%) |
|  |  | 512 | 72.4 | 84.7 | 113.6 (+34.2%) |
|  |  | 2048 | 282.6 | 133.6 | 285.0 (+113.4%) |
| Gemma 4 E4B | 4-bit | 128 | 56.0 | 84.6 | **37.6 (-55.6%)** |
|  |  | 512 | 122.7 | 122.0 | **73.8 (-39.5%)** |
|  |  | 2048 | 487.9 | 279.6 | **234.3 (-16.2%)** |
| Gemma 4 E4B | 6-bit | 128 | 56.0 | — | 124.7 |
|  |  | 512 | 120.7 | — | 268.4 |
|  |  | 2048 | 486.6 | — | 848.3 |
| Gemma 4 26B A4B | 4-bit | 128 | 67.8 | 257.8 | **213.4 (-17.2%)** |
|  |  | 512 | 148.9 | 315.8 | 422.7 (+33.8%) |
|  |  | 2048 | 581.1 | 620.6 | 1,318.4 (+112.4%) |
| Gemma 4 26B A4B | 6-bit | 128 | 75.8 | 222.8 | 248.1 (+11.4%) |
|  |  | 512 | 163.9 | 296.0 | 460.3 (+55.5%) |
|  |  | 2048 | 611.7 | 600.4 | 1,380.8 (+130.0%) |
| Gemma 4 31B | 4-bit | 128 | 241.1 | 452.2 | 807.3 (+78.5%) |
|  |  | 512 | 767.1 | 826.0 | 2,508.8 (+203.7%) |
|  |  | 2048 | 3,533.3 | 2,790.6 | 9,753.0 (+249.5%) |
| Gemma 4 31B | 6-bit | 128 | 255.3 | 457.0 | 907.5 (+98.6%) |
|  |  | 512 | 778.4 | 945.1 | 2,636.4 (+178.9%) |
|  |  | 2048 | 3,601.9 | 3,023.2 | 9,972.1 (+229.9%) |
| Qwen 3.6 27B | 4-bit | 128 | 234.4 | 301.4 | 656.8 (+117.9%) |
|  |  | 512 | 670.5 | 692.8 | 2,082.2 (+200.5%) |
|  |  | 2048 | 3,041.2 | 2,238.6 | 7,833.9 (+249.9%) |
| Qwen 3.6 27B | 6-bit | 128 | 240.0 | 367.8 | 712.9 (+93.8%) |
|  |  | 512 | 682.3 | 781.6 | 2,158.0 (+176.1%) |
|  |  | 2048 | 3,151.3 | 2,461.1 | 8,010.9 (+225.5%) |
| Qwen 3.6 35B A3B | 4-bit | 128 | 74.0 | 227.6 | 253.3 (+11.3%) |
|  |  | 512 | 163.7 | 317.3 | 436.2 (+37.5%) |
|  |  | 2048 | 583.7 | 592.7 | 1,207.6 (+103.7%) |
| Qwen 3.6 35B A3B | 6-bit | 128 | 79.9 | 296.6 | 323.0 (+8.9%) |
|  |  | 512 | 175.3 | 367.2 | 494.4 (+34.6%) |
|  |  | 2048 | 611.7 | 821.1 | 1,298.7 (+58.2%) |

</details>

### Session Mode: Embeddings

Embedding sessions use a separate pooling route from text generation. Public
Public results rows focus on sustained **batched** ingest workloads, where callers
embed many chunks and the fixed per-call cost can be amortized. Treat these
rows as embedding throughput and latency evidence, not as direct or MTP decode
evidence.

**How to read the public chart and table (best practice):**

- **Yellow (`mlx-lm`) and green (AX) are both already matrix-batched.** Each
  timed step encodes `B` sequences in one forward and materializes a contiguous
  CPU `float32 [B,H]` matrix. Published batch sizes are **B = 8, 32, 64**.
- **Green is not “AX single-sentence mode.”** A true single-sequence call is
  `B = 1` (or a Python loop of one-id embeds). That path is useful for
  query-latency diagnostics and lives under
  `benchmarks/results/embedding/embedding-fair/`; it is **not** the headline
  ingest claim.
- **Do not add an “AX batch only” series** on the competitive chart. Batching
  is an API shape available to both engines; a third bar that only AX gets would
  misread a usage pattern as an engine-only win.
- Prefer `session.embed_batch_array` / `embed_batch_flat` (or HTTP
  `input: [[ids], …]`) over one call per sentence — that advice applies to
  every backend; details in [`EMBEDDINGS.md`](EMBEDDINGS.md).

Cooled short-query fair runs remain useful for query-serving latency, but they
are not published here as headline throughput because tok/s on short text can
overstate or obscure per-call latency. The current Qwen short-query diagnostic
still shows an AX native-graph latency gap versus `mlx-lm` on some `B = 1`
shapes; treat that as a performance investigation target rather than a
sustained-ingest claim.

#### Qwen3-Embedding ingest scale

The current-main Qwen3 AX-only refresh covers 0.6B / 4B / 8B (2026-07-17). It
is shown beside the retained 2026-07-12 `mlx-lm` medians as a **cross-run
directional view**, not a same-session paired result. The current AX artifact
passes `ax_absolute_trend`; retained-reference differences are intentionally
ineligible for `paired_delta` claims because they come from separate runs. The
percentages below describe direction against the retained reference only, not
a locked engine-to-engine delta.

For larger RAG ingest jobs, use the sustained scale harness instead of
extrapolating from one isolated batch. The scale harness keeps the same
contiguous CPU `float32 [B,H]` output layout but embeds a fixed corpus of 512
chunks per trial, flushed at **batch sizes 8 / 32 / 64**. Both retained and
fresh runs use 2 warmups, 5 measured trials, and a 15-second cooldown. Each
measured pass is a multi-batch ingest run. p95 batch latency is shown because
larger batches increase per-flush latency even when throughput (tok/s) is
steady.

The current runtime enables length-affinity batching and calibrated `max_len`
buckets by default. Because this scale corpus uses uniform fixed-length chunks,
it does not isolate their mixed-length batch benefit.

Fresh AX-only artifact:
`benchmarks/results/embedding/embedding-scale/2026-07-17-ax-only-length-affinity-refresh-qwen/2026-07-17-013116/`.

The chart overlays all **18 batched shapes** (3 models × 2 chunk lengths × 3
batch sizes), grouped 0.6B → 4B → 8B (`mlx-lm` yellow retained reference, AX
green fresh run). Each box summarizes the six chunk/batch shapes for that
model; both series are batched encode, but their cross-run gap is directional.

<img src="assets/perf-embedding-ingest-scale-ax-vs-mlx-lm.svg" alt="Grouped box-and-whisker plot showing retained mlx-lm and fresh AX Engine ingest throughput for Qwen3-Embedding 0.6B, 4B, and 8B at batch sizes 8, 32, and 64">

| Model | Workload | Batch | Batches/trial | Retained mlx-lm tok/s | Fresh AX-only tok/s | Directional vs retained | AX chunks/s | AX p95 batch ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Embedding 0.6B 8-bit | 512 x 256-token chunks | 8 | 64 | 48,901.2 | 48,558.1 | −0.7% | 189.7 | 42.0 |
|  |  | 32 | 16 | 49,988.0 | 49,637.8 | −0.7% | 193.9 | 172.8 |
|  |  | 64 | 8 | 49,878.6 | 49,443.3 | −0.9% | 193.1 | 353.4 |
|  | 512 x 512-token chunks | 8 | 64 | 48,920.9 | 47,493.2 | −2.9% | 92.8 | 89.4 |
|  |  | 32 | 16 | 49,090.2 | 47,838.0 | −2.6% | 93.4 | 361.8 |
|  |  | 64 | 8 | 48,650.8 | 48,100.9 | −1.1% | 93.9 | 711.5 |
| Qwen3-Embedding 4B 4-bit DWQ | 512 x 256-token chunks | 8 | 64 | 6,591.1 | 6,628.2 | +0.6% | 25.9 | 323.0 |
|  |  | 32 | 16 | 6,463.0 | 6,590.4 | +2.0% | 25.7 | 1,304.6 |
|  |  | 64 | 8 | 6,471.9 | 6,579.4 | +1.7% | 25.7 | 2,610.8 |
|  | 512 x 512-token chunks | 8 | 64 | 6,416.5 | 6,429.8 | +0.2% | 12.6 | 656.3 |
|  |  | 32 | 16 | 6,138.0 | 6,265.9 | +2.1% | 12.2 | 2,645.7 |
|  |  | 64 | 8 | 6,307.6 | 6,322.1 | +0.2% | 12.3 | 5,275.2 |
| Qwen3-Embedding 8B 4-bit DWQ | 512 x 256-token chunks | 8 | 64 | 3,379.7 | 3,519.1 | +4.1% | 13.7 | 593.7 |
|  |  | 32 | 16 | 3,266.7 | 3,466.5 | +6.1% | 13.5 | 2,426.8 |
|  |  | 64 | 8 | 3,359.3 | 3,487.8 | +3.8% | 13.6 | 4,842.5 |
|  | 512 x 512-token chunks | 8 | 64 | 3,327.1 | 3,442.9 | +3.5% | 6.7 | 1,213.4 |
|  |  | 32 | 16 | 3,260.3 | 3,376.7 | +3.6% | 6.6 | 4,900.6 |
|  |  | 64 | 8 | 3,333.9 | 3,393.6 | +1.8% | 6.6 | 9,785.6 |

#### EmbeddingGemma ingest scale

The current-main EmbeddingGemma AX-only refresh (2026-07-17) is compared with
the retained 2026-07-02 `mlx-embeddings` medians. It is a cross-run directional
view, not a paired delta. The fresh AX artifact passes the `ax_absolute_trend`
publication gate; do not interpret the percentages as an exact engine-to-engine
claim because retained reference and fresh AX data are separate runs.

EmbeddingGemma uses `mlx-embeddings` as the sustained reference because its
full sentence-transformers route includes mean pooling, the Dense projection
head, and L2 normalization. The chart uses one `EmbeddingGemma 300M` group and
nests `mlx-embeddings` (yellow) plus AX Engine (green) inside it; each engine
box summarizes the six chunk/batch shapes listed below. The two series are
cross-run directional overlays.

<img src="assets/perf-embeddinggemma-ingest-scale-ax-vs-mlx-embeddings.svg" alt="Grouped box-and-whisker plot showing retained mlx-embeddings and fresh AX Engine ingest throughput for EmbeddingGemma 300M workloads">

| Model | Workload | Batch | Batches/trial | Retained `mlx-embeddings` tok/s | Fresh AX-only tok/s | Directional vs retained | AX chunks/s | AX p95 batch ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EmbeddingGemma 300M 8-bit | 512 x 256-token chunks | 8 | 64 | 129,909.0 | 140,125.7 | +7.9% | 547.4 | 14.6 |
|  |  | 32 | 16 | 148,284.8 | 158,028.9 | +6.6% | 617.3 | 58.5 |
|  |  | 64 | 8 | 149,976.1 | 157,849.4 | +5.2% | 616.6 | 123.8 |
|  | 512 x 512-token chunks | 8 | 64 | 127,604.8 | 141,940.9 | +11.2% | 277.2 | 28.5 |
|  |  | 32 | 16 | 140,105.8 | 149,365.5 | +6.6% | 291.7 | 118.2 |
|  |  | 64 | 8 | 132,121.8 | 140,102.5 | +6.0% | 273.6 | 260.2 |

Sources:
Qwen3 0.6B / 4B / 8B reference rows come from the retained paired artifact
`benchmarks/results/embedding/embedding-scale/2026-07-12-qwen-paired-v2/2026-07-12-145710/`;
fresh AX-only rows come from
`benchmarks/results/embedding/embedding-scale/2026-07-17-ax-only-length-affinity-refresh-qwen/2026-07-17-013116/`.
EmbeddingGemma reference rows come from
`benchmarks/results/embedding/embedding-scale/2026-07-02-embeddinggemma-paired-cooldown15-refresh/2026-07-02-175206/`
and fresh AX-only rows from
`benchmarks/results/embedding/embedding-scale/2026-07-17-ax-only-length-affinity-refresh-embeddinggemma/2026-07-17-025900/`.
Both fresh AX artifacts pass `ax_absolute_trend`; retained-reference
differences remain directional because the references are separate runs, not a
paired matrix. All scale runs use Hugging Face snapshot paths, median tok/s,
batch sizes 8/32/64, 512 chunks per trial, l2-normalized output, 2 warmups, 5
measured trials, and a 15-second cooldown between measured passes. Qwen uses AX
last-token pooling; EmbeddingGemma uses AX mean pooling + Dense head.
Single-batch cooled artifacts remain under
`benchmarks/results/embedding/embedding-fair/` for latency diagnostics
(short-query headlined as ms/item), but they are intentionally not published
here as headline throughput. API semantics, pooling modes, micro-batching
behavior, and cooldown profiles are documented in
[`EMBEDDINGS.md`](EMBEDDINGS.md).

Reproduce the sustained Qwen AX-only rows with:

```bash
python scripts/bench_embedding_ingest_scale.py \
  --model qwen3-embedding-0.6b-8bit=/path/to/Qwen3-Embedding-0.6B-8bit/snapshots/<sha> \
  --model qwen3-embedding-4b-4bit-dwq=/path/to/Qwen3-Embedding-4B-4bit-DWQ/snapshots/<sha> \
  --model qwen3-embedding-8b-4bit-dwq=/path/to/Qwen3-Embedding-8B-4bit-DWQ/snapshots/<sha> \
  --batch-sizes 8,32,64 --chunk-tokens 256,512 \
  --total-chunks 512 --warmup 2 --trials 5 --cooldown 15 \
  --ax-only
```

Reproduce the sustained EmbeddingGemma AX rows with:

```bash
python scripts/bench_embedding_ingest_scale.py \
  --model embeddinggemma-300m-8bit=/path/to/embeddinggemma-300m-8bit/snapshots/<sha> \
  --reference mlx_embeddings --pooling mean \
  --batch-sizes 8,32,64 --chunk-tokens 256,512 \
  --total-chunks 512 --warmup 2 --trials 5 --cooldown 15 \
  --ax-only
```
