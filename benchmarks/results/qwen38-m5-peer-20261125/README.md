# Qwen 3.8 MXFP4 HTTP serving on M5 Max

The MacBook Pro M5 Max 128 GB ran each engine against the same immutable
AXQ MXFP4 pack for each model. The 27B snapshot was on network-mounted
storage and the Flash Next snapshot was on the internal SSD. Each server
started in a separate process bound to loopback. The matrix measures
single-request HTTP serving after warmup. The two packs and all engine
builds are listed in [provenance.json](provenance.json).

| Engine | 27B output tok/s | 27B first text p95 | Flash Next output tok/s | Flash Next first text p95 |
| --- | ---: | ---: | ---: | ---: |
| AX Engine | 33.18 | 280 ms | 35.90 | 747 ms |
| oMLX 0.8.2 | 31.30 | 278 ms | **46.57** | 459 ms |
| MTPLX 2.12.3 | 28.61 | 289 ms | 0 | — |
| TensorFold 1.3.4 | 0 | — | 0 | — |
| mlxcel 0.8.1 | 0 | — | 0 | — |
| mlx-lm 0.32.0 | 29.05 | 291 ms | 0 | — |
| ds4 | 0 | — | 0 | — |
| Splash 1.7.0 | **93.79** | 219 ms | 0 | — |

Splash led 27B with DFlash2, reaching 83.96 and 106.24 tok/s across
the two passes. Its second pass reused cached input. Its code response
changed between passes. AX Engine's resident direct path reached 31.53
and 35.02 tok/s.

On Flash Next, oMLX reached 44.80 and 48.48 tok/s across the two passes,
using automatic SSD mapping for PLE. AX Engine reached 35.27 and 36.56
tok/s. Both engines emitted the same visible text for the code and math
prompts and differed on the prose prompt.

### Zero scores

| Engine | 27B | Flash Next |
| --- | --- | --- |
| MTPLX | Completed | Metal admission required 129.1 GiB resident plus 16 GiB headroom. |
| TensorFold | The native 27B loader returned `BadAffine`. | The loader required affine weights. |
| mlxcel | The embedding scales and packed weight shape failed quantization validation. | The loader required expert projection biases absent from the pack. |
| mlx-lm | Completed | The installed model registry lacked `qwen4_exp`. |
| ds4 | The loader rejected the Safetensors shard as invalid GGUF. | The loader rejected the Safetensors shard as invalid GGUF. |
| Splash | Completed | This version required GGUF for Flash Next. |

## Workload and scoring

- Request path is `POST /v1/completions` with SSE, temperature zero and
  a 64-token cap. The [fixed corpus](../../corpora/serving/qwen38_dflash2_paired.jsonl)
  already contains rendered prompts. Both pack tokenizers reproduce its
  exact stored token IDs. The warmup uses `short-answer` (26 input tokens).
  Each measured pass uses `code-shards` (65), `math-trains` (84) and
  `prose-library` (133) in order. There are two passes per engine and pack.
- Output tok/s is total visible completion tokens divided by the sum of
  elapsed request times, including prefill and streaming. Every streamed
  text is retokenized with the pinned pack tokenizer. SSE usage, when
  emitted, is recorded independently. First-text p95 is nearest rank
  over six requests. It measures the first nonempty text delta.
- A whole lane scores **0** if it cannot finish the same six requests.
  The [per-attempt results](results.jsonl) preserve the status and
  diagnostic for every zero. Zero represents no successful score.
- AX Engine uses resident weights, disables MTP and independent n-gram
  speculation, and keeps its ordinary in-process cache behavior. MTPLX
  uses serial target-only AR. oMLX disables its paged SSD prefix cache
  and keeps the batch generator's normal in-process behavior. TensorFold
  disables drafting. Splash uses its published per-model DFlash2 draft
  when its loader can prepare the target. The second pass can reuse
  process-local state. Output text digests expose divergence on the same
  input without claiming token- or quality-equivalence.
- The comparison uses one user, short prompts and one run order. Startup
  and the separate runtime diagnostic are outside the serving score.
  These runs do not measure long context, concurrent users, peak memory
  or model qualification.

The main sweep began at 98.27% host CPU idle and ended at 96.86%. Splash
was retried separately with its documented local draft and default
nonpersistent cache. Each published lane entered and exited at normal
memory pressure. The runner halts the owned server at critical pressure.

The separate [upstream runtime probe](upstream-runtime.txt) ran
`mlx_lm.benchmark` with 128 synthetic input tokens, a 64-token generation
budget and two trials. It produced **34.305 generation tok/s** for 27B.
The Flash Next loader rejected `qwen4_exp`. The HTTP matrix uses the
serving client for all engine scores.

## Replay

Install oMLX, MTPLX and Splash at the versions in the provenance file.
Stage `ax-engine-server` from the recorded AX build, extract the official
[TensorFold](https://github.com/ashhart/TensorFold/releases/tag/v1.3.4)
and [mlxcel](https://github.com/lablup/mlxcel/releases/tag/v0.8.1)
macOS distributions into `tensorfold/` and `mlxcel/`, and build
`ds4-server` from the [pinned source](https://github.com/antirez/ds4/tree/19b3f50180ec2c7324892be1f50bc83555ccb8d5e7)
with `make ds4-server`. Keep the extracted libraries beside their
binaries. In the stage, write the absolute immutable snapshot paths
to `dense.path` and `flash.path`. Put the two Splash draft directories
under `splash-drafts/dense` and `splash-drafts/flash` with their
`config.json` and `model.safetensors`. Use an MLX environment with
`mlx`, `mlx-metal`, `mlx-lm` and `tokenizers` at the recorded versions.
The runner expects the Homebrew executables in `/opt/homebrew/bin`
and the upstream server in `~/.venvs/ax-engine/bin`.

From the repository root, with the benchmark host idle and an empty
stage log directory:

```bash
export AX_BENCH_MLX_LIB=/path/to/site-packages/mlx/lib
"$HOME/.venvs/ax-engine/bin/python" benchmarks/results/qwen38-m5-peer-20261125/run.py \
  --stage /path/to/stage \
  --corpus benchmarks/corpora/serving/qwen38_dflash2_paired.jsonl \
  > replay.jsonl
shasum -a 256 -c benchmarks/results/qwen38-m5-peer-20261125/SHA256SUMS
```

The checksums cover the captured campaign and runner. A replay produces
fresh timing values in `replay.jsonl`.
