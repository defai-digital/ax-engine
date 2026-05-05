# AX Engine

A Mac-first LLM inference core for Apple Silicon M4+, built around one idea:

**for supported transformer families, an AX-owned scheduling and speculative
decode layer above MLX delivers measurably higher effective throughput than
running the MLX reference runtimes directly.**

> Requires macOS on Apple Silicon M4 or newer and Rust 1.85+.

## Why AX Engine

[mlx_lm](https://github.com/ml-explore/mlx-lm) is excellent — it is the
canonical Python MLX inference reference and the benchmark every AX row is
measured against.
[mlx-swift-lm](https://github.com/ml-explore/mlx-swift) brings the same
runtime to Swift with the same model coverage. Both are the right choice when
you need broad HuggingFace model support or direct MLX ecosystem integration.

AX Engine is not trying to replace either.

The gap AX targets is narrow: for **supported model families** on **Apple
Silicon**, an orchestration layer that owns speculative decode, request
scheduling, and KV state management above MLX can extract throughput that the
upstream runtimes leave on the table:

- **n-gram self-speculative decode** reaches up to 2.4x mlx_lm decode
  throughput on high-hit benchmark rows — with no second draft model and no
  model changes
- **AX-owned request lifecycle** provides deterministic, auditable scheduling,
  KV block management, and prefix reuse that upstream Python runtimes do not
  expose as stable contracts
- **workload-contract tooling** (`ax-engine-bench`) validates correctness,
  determinism, route identity, and regression across checked-in manifests, not
  just throughput snapshots

The thesis is not "our MLX tensor ops are faster." MLX compiles and executes the
same compute graph either way. The thesis is that **AX's decode strategy above
MLX** — how tokens are speculated, how requests are scheduled, how KV state is
materialized — produces measurably higher effective throughput on supported
workloads.

## Design

### Execution layer

AX Engine uses MLX directly for all tensor operations via the official `mlx-c`
C API. Matrix multiply, quantized matmul, attention, RMSNorm, and RoPE go
through MLX's own Apple-maintained Metal kernels, which use `simdgroup_matrix`
hardware tile multiply, hardware BF16 acceleration (M3+/M4+), and flash-style
SDPA. AX owns what happens above the compute graph.

What AX builds above MLX:

- **N-gram speculative decode**: a bigram/trigram table built at runtime predicts
  up to 4 draft tokens per step. The target model verifies them in one forward
  pass over `[last_token, D1, …, D_n]`. An EMA accept-rate gate (α=0.1,
  threshold 0.5) disables speculation after a bad sequence and re-enables when
  the table recovers. No second draft model required.
- **Scheduler and KV manager**: request lifecycle, batching, memory-blocked
  recovery, and execution planning live in `ax-engine-core` — deterministic,
  async-free, no framework dependencies.
- **Chunked KV cache**: keys and values grow in pre-allocated backing buffers via
  `slice_update`. Speculative rollback is O(1) — only the sequence-length
  pointer moves. After each decode step, all KV buffers are evaluated with the
  output token to flatten the lazy-eval graph and prevent O(N²) graph depth.
- **Graph compilation**: `mlx_enable_compile()` is called once at startup so
  Metal shader compilation and dispatch tables are reused across steps with the
  same shape — equivalent to `mx.compile()` in mlx_lm.
- **GatedDelta linear attention**: hybrid architectures (Qwen3.5, Qwen3-Next)
  use a custom SIMD-group Metal kernel for the recurrent GatedDelta state update.
  All other ops in the same models (dense attention, FFN, projections) delegate
  to MLX's hardware-optimized paths.

### Memory layer

`mlx_set_wired_limit(recommendedMaxWorkingSetSize)` wires model weights into GPU
memory at startup, preventing Metal from paging them between requests. A
dedicated GPU stream avoids cross-stream synchronization on the shared default
stream.

## Supported Models

| Family | Model | Architecture notes |
|---|---|---|
| Gemma 4 | gemma-4-e2b-it, gemma-4-e4b-it, gemma-4-26b-a4b-it, gemma-4-31b-it | Dense, per-layer embedding, and MoE variants; MLX affine 4/5/6/8-bit weights, sliding-window + full attention, K=V full-attention layers, logit softcapping |
| Qwen 3 | Qwen3-4B | Dense GQA + SwiGLU |
| Qwen 3.5 | Qwen3.5-9B | Linear attention + MoE FFN, attn_output_gate per-head interleaving |
| Qwen 3.6 / Coder Next | Qwen3.6-35B-A3B 4/5/6/8-bit MLX, Qwen3-Coder-Next-4bit | `qwen3_next` architecture: GatedDelta linear attention (3 of every 4 layers) + full attention with per-head sigmoid gate (every 4th layer) + sparse top-k MoE with shared expert |

All models use MLX safetensors format with the AX `model-manifest.json`
descriptor. Each supported architecture has a hand-written forward pass in
`ax-engine-mlx`. Adding a new architecture means implementing the model graph,
not wiring up a generic loader.

## Limitations

- **GatedDelta prefill**: Qwen3.5 and Qwen3-Next linear-attention layers use a
  custom Metal kernel with a serial time loop. For long prompts (512+ tokens),
  this puts AX prefill behind mlx-swift-lm on those models; decode throughput is
  unaffected.
- **Raw HuggingFace weights**: ax-engine loads MLX community (pre-sanitized)
  weights. Raw HF checkpoints for hybrid models need norm-weight `+1.0` and
  conv1d `moveaxis(2,1)` transformations that the converter does not apply.
- **Speculative rows**: effective-throughput measurements, not raw model-kernel
  speedups. n-gram hit rate is prompt/output-pattern dependent.

## Performance

**Apple M5 Max · 128 GB · macOS 26.4.1.** Random-token prompts (mlx_lm seed=0),
batch=1, prefill_step_size=2048, 3 timed trials + 1 warmup. `ax greedy` uses the
same decode policy as `mlx_lm`; `ax speculative` adds n-gram self-speculation and
reports observed effective throughput, not raw model speed.

### Decode throughput (tok/s) — generation=128 tokens, temp=0

| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax engine (greedy) | ax engine (speculative) |
|---|---|---:|---:|---:|---:|---|
| Gemma 4 E2B | 4-bit · group=64 · affine | 128 | 197.5 | 192.4 (−2.6%) | 176.0 (−10.9%) | **467.6** (+136.8%) |
|  |  | 512 | 191.9 | 179.5 (−6.5%) | 170.9 (−11.0%) | **464.8** (+142.2%) |
| Gemma 4 E2B | 5-bit · group=64 · affine‖ | 128 | 182.9 | 174.1 (−4.8%) | 162.2 (−11.3%) | **380.8** (+108.2%) |
|  |  | 512 | 178.1 | 167.0 (−6.2%) | 155.2 (−12.9%) | **377.2** (+111.8%) |
| Gemma 4 E2B | 6-bit · group=64 · affine‖ | 128 | 161.3 | 153.0 (−5.1%) | 145.4 (−9.8%) | **344.0** (+113.3%) |
|  |  | 512 | 154.2 | 147.1 (−4.6%) | 139.6 (−9.5%) | **346.0** (+124.4%) |
| Gemma 4 E2B | 8-bit · group=64 · affine‖ | 128 | 139.4 | 134.9 (−3.2%) | 128.2 (−8.0%) | **369.9** (+165.3%) |
|  |  | 512 | 134.5 | 130.8 (−2.8%) | 123.4 (−8.3%) | **368.2** (+173.7%) |
| Gemma 4 26B A4B | 4-bit · group=64 · affine¶ | 128 | 118.3 | n/a¶ | 115.2 (−2.6%) | **234.7** (+98.4%) |
|  |  | 512 | 113.1 | n/a¶ | 111.3 (−1.6%) | **211.4** (+86.9%) |
| Gemma 4 31B | 4-bit · group=64 · affine | 128 | 26.2 | 24.8 (−5.5%) | 25.5 (−3.0%) | **53.3** (+103.3%) |
|  |  | 512 | 24.9 | 24.7 (−0.9%) | 25.5 (+2.1%) | **50.4** (+102.1%) |
| Qwen 3 4B | 4-bit · group=64 | 128 | 169.6 | 168.7 (−0.6%) | 167.7 (−1.1%) | **311.5** (+83.7%) |
|  |  | 512 | 169.8 | 161.0 (−5.2%) | 158.9 (−6.4%) | **289.5** (+70.4%) |
| Qwen 3.5 9B | 4-bit · group=64 · affine | 128 | 96.5 | 93.7 (−2.9%) | 101.3 (+4.9%) | **179.7** (+86.2%) † |
|  |  | 512 | 101.3 | 91.4 (−9.7%) | 100.6 (−0.7%) | 93.2 (−8.0%) † |
| Qwen 3.6 35B A3B | UD-MLX 4-bit · group=64 · affine§ | 128 | 107.6 | 103.6 (−3.7%) | 108.5 (+0.9%) | **211.0** (+96.2%) † |
|  |  | 512 | 103.3 | 101.4 (−1.9%) | 110.0 (+6.5%) | **207.7** (+101.1%) † |
| Qwen 3.6 35B A3B | MLX 5-bit · group=64 · affine§ | 128 | 116.8 | 110.2 (−5.6%) | 119.8 (+2.5%) | **215.9** (+84.8%) † |
|  |  | 512 | 113.7 | 108.7 (−4.4%) | 120.2 (+5.6%) | **212.5** (+86.8%) † |
| Qwen 3.6 35B A3B | MLX 6-bit · group=64 · affine§ | 128 | 102.9 | 99.1 (−3.6%) | 105.6 (+2.6%) | **197.5** (+91.9%) † |
|  |  | 512 | 101.1 | 98.0 (−3.1%) | 105.4 (+4.3%) | **195.8** (+93.7%) † |
| Qwen 3.6 35B A3B | MLX 8-bit · group=64 · affine§ | 128 | 93.6 | 89.3 (−4.6%) | 103.1 (+10.1%) | **208.9** (+123.1%) † |
|  |  | 512 | 91.4 | 89.1 (−2.6%) | 102.4 (+12.0%) | **205.6** (+124.8%) † |
| Qwen Coder Next | 4-bit · group=64 · affine‡ | 128 | 92.2 | 89.4 (−3.0%) | 95.1 (+3.2%) | **204.7** (+122.1%) † |
|  |  | 512 | 90.4 | 89.2 (−1.3%) | 95.4 (+5.6%) | **200.7** (+122.1%) † |

† Qwen 3.5, Qwen 3.6, and Qwen Coder Next speculative rows use a rollback-safe
branch/recompute path for SSM state. These are effective-throughput
measurements from AX's n-gram speculative policy, not raw model decode speed.
Linear-attention speculation is prompt/output-pattern dependent.

‡ Qwen Coder Next uses MLX affine 4-bit globally, with 8-bit overrides for
router and shared-expert gate tensors.

§ Qwen 3.6 35B A3B includes the Unsloth UD-MLX 4-bit checkpoint plus
MLX-community 5/6/8-bit checkpoints. The 5/6-bit checkpoints are affine 5/6-bit
globally with 8-bit router and shared-expert gate overrides; the 8-bit
checkpoint is affine 8-bit throughout.

¶ Gemma 4 26B A4B is the public Gemma 4 MoE MLX model. Its checkpoint uses
affine 4-bit globally, with 8-bit overrides for dense MLP and router
projections. The local `mlx-swift-lm` reference currently has Gemma4 dense/text
support but no Gemma4 MoE Router/Experts path, so no Swift row is admitted for
that model.

‖ Gemma 4 E2B 5/6/8-bit rows are quantization-support checks against
`mlx_lm` primary baseline plus `mlx_swift_lm` secondary reference rows.

### Prefill throughput (tok/s) — percentages vs mlx_lm

| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax engine |
|---|---|---:|---:|---:|---:|
| Gemma 4 E2B | 4-bit · group=64 · affine | 128 | 2,265.8 | 2,450.4 (+8.1%) | 3,248.7 (+43.4%) |
|  |  | 512 | 7,634.1 | 6,664.3 (−12.7%) | 7,640.2 (+0.1%) |
| Gemma 4 E2B | 5-bit · group=64 · affine‖ | 128 | 2,267.5 | 2,393.9 (+5.6%) | 3,197.3 (+41.0%) |
|  |  | 512 | 8,405.7 | 6,742.6 (−19.8%) | 7,507.0 (−10.7%) |
| Gemma 4 E2B | 6-bit · group=64 · affine‖ | 128 | 2,156.3 | 3,436.8 (+59.4%) | 3,122.3 (+44.8%) |
|  |  | 512 | 7,320.7 | 7,962.3 (+8.8%) | 7,315.7 (−0.1%) |
| Gemma 4 E2B | 8-bit · group=64 · affine‖ | 128 | 1,911.7 | 3,082.0 (+61.2%) | 3,070.4 (+60.6%) |
|  |  | 512 | 6,582.8 | 6,758.1 (+2.7%) | 7,356.1 (+11.7%) |
| Gemma 4 26B A4B | 4-bit · group=64 · affine¶ | 128 | 545.3 | n/a¶ | 691.3 (+26.8%) |
|  |  | 512 | 1,620.7 | n/a¶ | 788.3 (−51.4%) |
| Gemma 4 31B | 4-bit · group=64 · affine | 128 | 336.5 | 641.6 (+90.7%) | 541.8 (+61.0%) |
|  |  | 512 | 563.5 | 760.6 (+35.0%) | 727.2 (+29.1%) |
| Qwen 3 4B | 4-bit · group=64 | 128 | 1,581.1 | 3,627.8 (+129.4%) | 3,077.7 (+94.7%) |
|  |  | 512 | 3,726.0 | 6,173.7 (+65.7%) | 5,428.9 (+45.7%) |
| Qwen 3.5 9B | 4-bit · group=64 · affine | 128 | 1,133.3 | 2,101.1 (+85.4%) | 2,011.7 (+77.5%) |
|  |  | 512 | 2,245.7 | 3,165.8 (+41.0%) | 2,872.4 (+27.9%) |
| Qwen 3.6 35B A3B | UD-MLX 4-bit · group=64 · affine§ | 128 | 531.7 | 963.2 (+81.1%) | 930.3 (+75.0%) |
|  |  | 512 | 1,594.2 | 2,546.5 (+59.7%) | 1,169.3 (−26.7%) |
| Qwen 3.6 35B A3B | MLX 5-bit · group=64 · affine§ | 128 | 474.4 | 861.8 (+81.7%) | 810.2 (+70.8%) |
|  |  | 512 | 1,484.5 | 2,416.7 (+62.8%) | 939.6 (−36.7%) |
| Qwen 3.6 35B A3B | MLX 6-bit · group=64 · affine§ | 128 | 420.0 | 762.4 (+81.5%) | 758.0 (+80.5%) |
|  |  | 512 | 1,377.9 | 2,350.6 (+70.6%) | 872.0 (−36.7%) |
| Qwen 3.6 35B A3B | MLX 8-bit · group=64 · affine§ | 128 | 393.1 | 617.7 (+57.1%) | 812.0 (+106.5%) |
|  |  | 512 | 1,202.2 | 2,305.2 (+91.7%) | 931.1 (−22.5%) |
| Qwen Coder Next | 4-bit · group=64 · affine‡ | 128 | 267.1 | 384.9 (+44.1%) | 698.7 (+161.6%) |
|  |  | 512 | 815.4 | 1,417.0 (+73.8%) | 801.2 (−1.7%) |

### Workload contract (ax-engine-bench scenario, 256-input / 128-output, temp=0)

| Model | Prefill tok/s | Decode tok/s | TTFT | Contract |
|---|---:|---:|---:|:---:|
| Gemma 4 E2B  | 6,984 | 397 | 262 ms | PASS |
| Qwen 3 4B    | 4,743 | 341 | 226 ms | PASS |
| Qwen 3.5 9B  | 2,478 |  88 | 389 ms | PASS |

See `docs/BENCHMARKS.md` for methodology and prompt-provenance requirements.

## Quick Start

```bash
cargo build --workspace --release

# HTTP inference server (MLX mode)
./target/release/ax-engine-server \
  --mlx \
  --mlx-model-artifacts-dir .internal/models/gemma-4-e2b-it-4bit \
  --port 8080

# Python bindings (after maturin develop)
python3 - <<'EOF'
import ax_engine
with ax_engine.Session(model_id='gemma4', mlx=True,
        mlx_model_artifacts_dir='.internal/models/gemma-4-e2b-it-4bit') as s:
    result = s.generate([1, 2, 3], max_output_tokens=32)
    print(result.output_tokens)
EOF

# Primary benchmark: AX vs mlx_lm vs mlx-swift-lm
python3 scripts/bench_mlx_inference_stack.py \
  --model-dir .internal/models/gemma-4-e2b-it-4bit \
  --prompt-tokens 128,512 --generation-tokens 128 \
  --ax-both-modes --repetitions 3 \
  --mlx-swift-lm-command './scripts/mlx-swift-bench/.build/release/mlx-swift-bench \
    --model {model} --prompt-token-ids {prompt_token_ids_path} \
    --generation-tokens {generation_tokens} --trials {trials} \
    --delay {delay} --prefill-step-size {prefill_step_size}' \
  --output benchmarks/results/mlx-inference/2026-05-04/gemma-4-e2b-it-4bit.json

# Secondary workload-contract benchmark
./target/release/ax-engine-bench scenario \
  --manifest benchmarks/manifests/scenario/chat_gemma4_e2b_short.json \
  --output-root benchmarks/results

# Smoke checks
bash scripts/check-server-preview.sh
bash scripts/check-python-preview.sh
```

## Workspace

```
crates/ax-engine-core    Engine state machine, scheduler, KV manager, sampler
crates/ax-engine-mlx     MLX model graph, speculative decode, KV cache, runner
crates/mlx-sys           bindgen FFI over mlx-c; safe MlxArray RAII wrappers
crates/ax-engine-sdk     Session API, backend resolution (MLX or llama.cpp)
crates/ax-engine-server  Axum HTTP/SSE adapter (OpenAI-compatible routes)
crates/ax-engine-bench   Manifest-driven workload-contract CLI
crates/ax-engine-py      PyO3 extension (ABI3, Python 3.10+)
```

Non-MLX inference routes through the delegated `llama.cpp` contract.

## Development

```bash
cargo build --workspace                                           # build all crates
cargo test --quiet                                                # full Rust test suite
cargo clippy --all-targets --all-features -- -D warnings         # lint (CI gate)
cargo fmt                                                         # format
maturin develop                                                   # rebuild Python extension
python -m unittest discover -s python/tests -v                   # Python tests
```

Public documentation is in `docs/`. Canonical benchmark manifests are in
`benchmarks/manifests/`. Private design material is in `.internal/`.

## Contributing

AX Engine welcomes public contributions. See [CONTRIBUTING.md](CONTRIBUTING.md)
for guidelines.

## Community

- Website: [automatosx.com](https://automatosx.com)
- Discord: [Join us](https://discord.com/invite/cTavsMgu)
- Email: enquiry@defai.digital

## License

MIT License. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 [DEFAI Private Limited](https://defai.digital)
