# Scripts

This directory contains repo-owned smoke checks and a small number of diagnostic
helpers.

## Benchmarking Rule

Use `bench_mlx_inference_stack.py` for AX Engine MLX model-inference
comparisons. That script always runs `mlx_lm.benchmark` as the primary baseline
and fails the run if the baseline cannot be produced. The harness mirrors the
`mlx_lm.benchmark` prompt standard by generating `mx.random.seed(0)` random
token IDs from the model vocabulary, writes those token IDs to JSON artifacts,
and reuses them for AX Engine and any admitted `mlx-swift-lm` adapter.

Optional `mlx-swift-lm` rows are secondary-baseline rows only when the command
is built on the reference package's `BenchmarkHelpers` / `MLXLMCommon`
generation APIs, reads the harness-emitted prompt token JSON, and reports
prefill/decode throughput for the same random-token prompt/decode shape. Do
not use an application-server wrapper or unrelated Swift timing script as the
`mlx-swift-lm` baseline.

Use `ax-engine-bench` through the workload-contract `check-bench-*.sh` scripts
for scenario, replay, matrix, baseline, compare, delegated llama.cpp route
checks, and readiness. `check-bench-inference-stack.sh` is the exception in
that family: it validates the MLX inference-stack harness contract without
running `ax-engine-bench` or loading a model.

Use `bench_ax_serving.py` for online serving benchmarks through
`/v1/generate/stream`. That harness measures client-observed TTFT, TPOT,
streaming step intervals, E2E latency, request throughput, output-token
throughput, queue delay, and SLO goodput over a JSONL prompt corpus. It is
serving evidence, not a raw model-runtime throughput baseline.

Do not use ad hoc server timing or llama.cpp route checks as AX-owned MLX
throughput baselines.

## Script Groups

- `lib/common.sh`: shared shell helpers for repo-root discovery, Python binary
  selection, temporary paths, free-port allocation, and PID cleanup.
- `check-scripts.sh`: fast script hygiene gate. It syntax-checks shell scripts,
  compiles Python scripts, and runs the MLX inference-stack contract tests.
- `check-openwebui-e2e.sh` and `openwebui_e2e.py`: optional OpenWebUI
  integration smoke. Set `AX_OPENWEBUI_E2E=1` to run it. The wrapper starts an
  ephemeral OpenWebUI Docker container, points it at an AX OpenAI-compatible
  endpoint, verifies model discovery through OpenWebUI's OpenAI proxy, sends a
  chat completion, and fails on backend disconnect text or obvious corruption
  patterns. It is intentionally outside the default script gate because it
  requires Docker and usually local MLX artifacts; after `AX_OPENWEBUI_E2E=1`
  is set, Docker unavailability is treated as a failure rather than a skip.
- `check_direct_model_compat_smoke.py`: artifact-gated native MLX
  compatibility smoke for direct Qwen3-Coder-Next, Qwen3.6 35B-A3B, and Gemma 4
  support. Provide `AX_ENGINE_QWEN_CODER_NEXT_ARTIFACTS_DIR`,
  `AX_ENGINE_QWEN36_35B_ARTIFACTS_DIR`, and/or
  `AX_ENGINE_GEMMA4_ARTIFACTS_DIR`; the script starts `ax-engine-server`,
  verifies `/v1/models` tool-call metadata, then sends equivalent tool-enabled
  requests through OpenAI `/v1/chat/completions` and Ollama `/api/chat`. With no
  artifacts configured it emits a JSON skip result.
- `check-mlx-telemetry.sh`: targeted Gemma/AX MLX telemetry gate. It runs the
  MLX crate clippy/tests plus the MLX inference-stack and script gates. Use
  `--full-workspace` only when a change may affect shared Rust contracts; that
  mode runs `cargo test --workspace --no-run` first, then crate-by-crate tests
  with timeout and `AX_CARGO_JOBS=1` by default so a stuck full workspace run
  does not leave orphaned compiler processes.
- `brew-release.sh`: local Homebrew release publisher. It packages
  `ax-engine-server` and `ax-engine-bench`; with `--sign-identity`, it
  codesigns and notarizes all packaged binaries before upload. With
  `--minisign`, it signs the release tarball with the shared ax-code signing
  key (`~/signkey/ax-code.sec`), verifies it with `~/signkey/ax-code.pub`, and
  uploads the matching `.minisig` beside the archive. Without
  `--sign-identity`, binaries are intentionally left unsigned.
- `minisign-keygen.sh`: generates the minisign keypair for signing release
  artifacts. Defaults to the shared ax-code key paths (`~/signkey/ax-code.sec`
  / `~/signkey/ax-code.pub`), sets a private `700` directory and `600` secret
  key, and refuses to overwrite an existing keypair unless passed `--force`.
  Supports `--allow-unencrypted-test-key` (short-lived tests only) and
  `--dry-run`.
- `minisign-artifact.sh`: signs one or more release artifacts with minisign.
  The default key is the shared ax-code signing key at `~/signkey/ax-code.sec`
  / `~/signkey/ax-code.pub` (Keychain service `ax-code-minisign`, account
  `ax-code-release`). If the passphrase is already stored in the ax-code
  Keychain entry, the release path needs no password prompt. Override key
  paths via `AX_MINISIGN_SECRET_KEY` / `AX_MINISIGN_PUBLIC_KEY` or Keychain
  lookup via `--keychain-service` / `--keychain-account`. Set
  `AX_MINISIGN_PINNED_PUBLIC_KEY` (or `--pinned-public-key`) to fail closed
  unless the local public key matches the expected release key. Also supports
  `--dry-run`, `--untrusted-comment`, and `--signature-dir`. See
  `docs/MINISIGN.md` for the full operator guide and the
  `scripts/test_minisign_artifact.py` contract tests.
- `publish-github-release.sh`: full local GitHub Release publisher for the
  macOS arm64 CLI assets. It verifies tag/version consistency, requires a clean
  tree by default, runs release gates, builds `ax-engine-server` and
  `ax-engine-bench`, writes a tarball, SHA256 file, and manifest under
  `target/release-artifacts/<tag>/`, signs those artifacts with minisign by
  default, pushes the tag, creates the GitHub release, uploads the assets, and
  verifies the uploaded asset names. Use `--dry-run` to exercise local steps
  without pushing or uploading.
- `download_model.py`: MLX LLM download helper. It delegates acquisition to
  `mlx-lm`, resolves the resulting cache snapshot, validates local model files,
  and generates the AX model manifest when `ax-engine-bench` or Cargo is
  available. It rejects embedding repo IDs; embedding artifacts are manual.
  Use `--dest` only when you want a custom local directory. Use `--json` for
  automation.
- `prepare_mtp_sidecar.py`: generic "download + convert to MTP mode" tool.
  Given `--hf-repo <source>` (a model that ships `mtp.*` tensors) and `--base`
  (a local dir or cached repo id of the serving model), it auto-discovers which
  source shards hold the MTP head from `model.safetensors.index.json`, downloads
  only those, normalizes the tensor layout through a per-architecture registry
  (Qwen3.6 dense + packed-MoE today), and writes the sidecar contract the
  runtime loads: `mtp.safetensors`, `mtplx_runtime.json`, a patched
  `config.json`, and an `ax.mtp_sidecar_provenance.v1` manifest (validated by
  `check_mtp_sidecar_provenance.py`). Optional `--quantize {4,8}` produces a
  mixed sidecar (2-D projections quantized, experts/norms bf16). Output defaults
  to a synthetic `models--ax-local--<base>-MTP/snapshots/v1/` cache entry usable
  by `ax-engine-bench --model-dir` directly.
- `prepare_gemma4_assistant_mtp.py`: packages a Gemma 4 target + assistant pair
  for assistant-MTP speculative decoding. Gemma 4's MTP is a separate small
  "assistant" drafter model (not a fused `mtp.*` sidecar), so this is a distinct
  tool from `prepare_mtp_sidecar.py`. Given `--target` and `--assistant` (local
  dirs or cached repo ids), it assembles a self-contained model dir (target +
  `assistant/` subtree, hardlinked), patches the ax-engine packaging markers on
  the assistant config (`model_type: gemma4_assistant`, `backbone_hidden_size`),
  copies the target tokenizer into the assistant dir for the byte-identity
  check, writes the `ax_gemma4_assistant_mtp.json` contract, and pre-validates
  every rule the runtime (`crates/ax-engine-mlx/src/gemma4_assistant_mtp.rs`)
  enforces — known pair, vocab/hidden match, full KV-sharing, no
  per-layer-input/MoE/double-wide MLP — so failures surface at prep time.
- `prepare_glm_mtp_sidecar.py`: extracts GLM-4.7-Flash's built-in MTP layer
  from the upstream `zai-org/GLM-4.7-Flash` checkpoint and writes the AX runtime
  sidecar files (`glm_mtp.safetensors`, `glm_mtp_runtime.json`, and
  `ax_glm_mtp_manifest.json`) beside a copied MLX 6-bit base snapshot. This is
  not an external draft model; it repackages the GLM `model.layers.47.*` MTP
  tensors into the layout loaded by `crates/ax-engine-mlx/src/weights.rs`.
- `prepare_qwen36_mtp_sidecar.py`: the Qwen3.6-specific predecessor used to
  build the published fair-benchmark sidecars (fixed `--model {27b,35b}` table).
  Kept for provenance reproducibility; prefer `prepare_mtp_sidecar.py` for new
  models.
- `ax-engine-bench generate-manifest <model-dir> --json`: stable
  `ax.generate_manifest.v1` summary for automation callers that need to
  distinguish newly written manifests from already-ready model directories.
- `ax-engine-bench scenario|replay|matrix|compare|matrix-compare|baseline
  ... --json`: stable `ax.benchmark_artifact.v1` summary for callers that need
  the produced `result_dir` without parsing shell text.
- `ax-engine-bench doctor --json`: readiness report plus workflow discovery for
  source-checkout versus installed-tools mode. Automation callers should use its
  command `argv` fields instead of reconstructing Cargo/Homebrew command lines.
  With `--mlx-model-artifacts-dir`, the same JSON includes structured
  `model_artifacts` readiness for config, manifest, safetensors, model type, and
  quantization metadata.
- `bench_mlx_inference_stack.py`: MLX model-inference comparison against
  `mlx_lm.benchmark`. AX rows disable the shared MLX prefix cache by default so
  prefill throughput and TTFT stay cold-prefill measurements across warmup and
  repeated trials; use `--ax-enable-prefix-cache` only for explicit
  prefix-cache experiments. It can pass through
  `--experimental-mlx-kv-compression turboquant-shadow` or
  `turboquant-fused-experimental` for AX rows and records TurboQuant KV
  compression route counters, including shadow-storage sync calls and wall time
  plus fused decode candidate/attempt/success/fallback counters and fallback
  reason labels, when the runtime emits them. It also supports opt-in
  `--ax-decode-profile` rows that expose direct-decode stage counters for
  Gemma per-layer-input, QKV projection, SDPA, post-attention/FFN, and lm-head
  diagnosis; those profile rows insert eval barriers and are not headline
  throughput evidence.
- `bench_qwen36_mtp_fair.py`: Qwen3.6 MTP prompt-suite harness for MTPLX and AX
  Engine. The default `--modes mtp mtp-ngram` path is a fixed-depth comparison.
  Add `--modes tuned` only for tuned best-of rows: MTPLX runs its public
  `mtplx tune --retune` path, while AX sweeps direct, n-gram, MTP, and
  MTP+n-gram policies before running the selected policy as the final row. Keep
  fixed-depth and tuned-best-of summaries separate for headline comparisons.
- `bench_gemma4_assistant_mtp.py`: Gemma 4 assistant-MTP prompt-suite harness
  for AX Engine. It locates or prepares `ax_gemma4_assistant_mtp.json` model
  directories for Gemma 4 12B/26B A4B/31B, runs same-artifact direct,
  assistant-MTP-only, and assistant MTP+n-gram rows, records target affine bit
  composition plus route telemetry, and writes summaries under
  `benchmarks/results/gemma4-assistant-mtp/`. Use `--profiles` for the
  gate/depth/confidence/n-gram ablation matrix and `--resume` to continue a
  long matrix without rerunning completed per-suite artifacts.
- `bench_llama_cpp_metal_sweep.py`: resolves GGUF candidates from
  `benchmarks/manifests/llama_cpp_metal/inventory.json` and runs
  `bench_mlx_inference_stack.py` for the matching README rows. By default it
  captures llama.cpp Metal shape-compatible rows only. With `--full-stack`, it
  captures llama.cpp Metal, `mlx_lm.benchmark`, AX direct, and AX default
  n-gram rows in one artifact per model; with `--update-readme`, it refreshes
  the README performance tables from those artifacts. Full-stack README
  updates fail closed unless every selected row completed; use
  `--allow-partial-readme-update` only for intentionally partial reports.
- `test_bench_mlx_inference_stack.py`: unit tests for the MLX benchmark
  contract, parser, prompt artifact hash checks, and secondary adapter shape.
- `build_long_context_comparison_artifact.py`,
  `check_long_context_comparison_artifact.py`, and
  `render_long_context_comparison_report.py`: build, validate, and render
  `ax.long_context_comparison.v1` artifacts from MLX inference-stack runs. Use
  these when a long-prefill run includes AX, `mlx_lm`, and optional
  `llama.cpp Metal`; the checker enforces AX-vs-`mlx_lm` prompt-hash parity
  while keeping `llama.cpp` as a shape-compatible external GGUF baseline.
- `test_long_context_comparison_artifact.py`: unit tests for the long-context
  comparison artifact builder, checker, and renderer.
- `build_long_context_decode_at_depth_artifact.py`,
  `check_long_context_decode_at_depth_artifact.py`, and
  `render_long_context_decode_at_depth_report.py`: build, validate, and render
  `ax.long_context_decode_at_depth.v1` artifacts. These compare decode
  throughput after a context depth already exists. `llama.cpp Metal` rows are
  admitted only when they carry explicit `llama-bench n_depth` evidence.
  Capture that evidence with `bench_mlx_inference_stack.py
  --llama-cpp-decode-at-depth`, which attaches `decode_at_depth_tok_s` without
  replacing the ordinary shape-compatible `llama.cpp` `pp`/`tg` metrics.
- `test_long_context_decode_at_depth_artifact.py`: unit tests for the
  decode-at-depth artifact builder, checker, and renderer.
- `render_readme_performance_charts.py`: renders the README prefill, decode,
  and TTFT box-and-whisker SVG charts from the completed MLX inference-stack
  artifact directory. By default it reads the current README artifact directory
  line and writes `docs/assets/perf-*-box-whisker.svg`; use `--check` to fail
  if the checked-in charts are stale.
- `render_mtp_flappy_charts.py`: renders the README MTP SVG charts from
  `docs/PERFORMANCE.md#mtp-mode`. The checked-in set includes Speed to Speed
  and Quality summary charts plus separate tok/s and accept-rate charts for
  each bundle pair.
- `bench_ax_serving.py`: online serving benchmark against
  `/v1/generate/stream`. It reads a JSONL prompt corpus, runs closed-loop
  concurrency or open-loop request-rate sweeps, and writes
  `ax.serving_benchmark.v1` artifacts with TTFT, TPOT, E2E, queue-delay,
  throughput, category, route-decision counters, and goodput summaries.
- `build_serving_shared_prefix_corpus.py`: deterministic token-corpus builder
  for serving soaks that must exercise shared long prefixes, especially
  disk-durable prefix-cache promotion evidence.
- `run_disk_prefix_serving_soak.py`: operator-facing runner that creates the
  shared-prefix corpus, runs the online serving benchmark, validates the
  disk-prefix route-decision gate, renders the report, and writes an auditable
  `commands.json` bundle. It assumes the AX server is already running.
- `check_ax_serving_benchmark_artifact.py`: fail-closed validator for
  `ax.serving_benchmark.v1` artifacts. Use `--require-slo`,
  `--min-goodput-ratio`, `--min-input-tokens-p95`, and
  `--require-route-decision-min KEY=MIN` before citing long-prompt or
  runtime-path-specific serving claims.
- `render_ax_serving_benchmark_report.py`: renders validated
  `ax.serving_benchmark.v1` artifacts as Markdown tables for TTFT, client TPOT,
  E2E latency, queue delay, token distributions, goodput, and category
  breakdowns. It accepts the same route-decision gate when the report is for a
  runtime-path-specific promotion claim.
- `test_bench_ax_serving.py`: unit tests for the serving benchmark SSE parser,
  latency accounting, percentile/goodput summary, and artifact writer.
- `test_build_serving_shared_prefix_corpus.py`: unit tests for the deterministic
  shared-prefix serving corpus builder.
- `test_run_disk_prefix_serving_soak.py`: unit tests for the disk-prefix
  serving soak runner and dry-run command bundle.
- `test_ax_serving_benchmark_artifact.py`: unit tests for the serving artifact
  checker, including long-prompt p95 gates and failed-request rejection.
- `test_render_ax_serving_benchmark_report.py`: unit tests for the serving
  artifact Markdown renderer.
- `check_gateddelta_prefill_profile_artifact.py`: fail-closed validator for
  `--gateddelta-prefill-profile` artifacts. It requires Qwen-style
  linear-attention metadata, the 512/2048/8192/32768 prompt matrix, direct AX
  rows, `mlx_lm` primary-reference rows, no n-gram/KV-compression evidence, and
  opt-in `ax_mlx_linear_attention_profile` stage counters. The artifact must
  also carry versioned `gateddelta_prefill_profile.model_preflight` evidence
  from `check_gateddelta_prefill_model.py`.
- `test_gateddelta_prefill_profile_artifact.py`: unit tests for the GatedDelta
  prefill profile artifact validator.
- `render_gateddelta_prefill_profile_report.py`: renders a validated
  GatedDelta prefill profile artifact as a Markdown review table with
  linear-attention stage timings, recurrent share, dominant stage, and next-step
  hints for scan/fusion experiments. The benchmark harness can call it during
  capture with `--gateddelta-prefill-profile-report-output`.
- `render_mlx_forward_profile_report.py`: renders diagnostic
  `AX_MLX_LINEAR_ATTENTION_PROFILE=1` inference-stack artifacts as a Markdown
  review table. It is intentionally broader than the GatedDelta validator and
  is used after prefill-breakdown evidence shows model-forward dominance. Keep
  its timing-barrier output out of README headline throughput tables.
- `render_mlx_decode_profile_report.py`: renders diagnostic
  `AX_MLX_DECODE_PROFILE=1` inference-stack artifacts as a Markdown review
  table with parent decode stages and available substage splits. It tolerates
  older artifacts that predate the finer-grained counters by marking those cells
  as `n/a`.
- `check_gateddelta_prefill_model.py`: fail-closed preflight for real-model
  GatedDelta profile runs. It checks `config.json` plus `model-manifest.json`
  before release-server build and requires a `qwen3_5`/`qwen3_next`
  linear-attention manifest with the gated-delta kernel dimensions configured.
- `check_turboquant_quality_artifact.py`: fail-closed validator for internal
  TurboQuant long-context quality gate artifacts. It checks model identity,
  long-context shape, baseline/candidate provenance, candidate mode
  `turboquant-fused-experimental`, K8/V4 route metadata schema `>= 2`,
  fused_compressed_decode path code `2`, fused decode successes, Metal fused
  decode successes, zero fallbacks, decode quality thresholds, recorded
  throughput ratio, and memory-savings evidence. Decode-throughput promotion is
  checked separately by the readiness report.
- `check_turboquant_microbench_artifact.py`: fail-closed validator for
  standalone fused cold-decode microbenchmark artifacts. It checks K8/V4
  metadata, long-cold-context coverage, `two_stage_scores` quality, memory
  savings, and speedup against the CPU reference plus `dim_parallel` when
  present. Use `--require-dim-parallel` for D3 evidence checks where that
  comparison is mandatory.
- `check_turboquant_prd_completion.py`: fail-closed TurboQuant PRD completion
  report. It joins quality/promotion readiness, fused microbench evidence, and
  short-decode speedup artifacts so the PRD is not marked complete from code
  presence alone.
- `build_turboquant_quality_artifact.py`: compiles a TurboQuant quality artifact
  from MLX inference-stack benchmark output and a quality-metrics JSON file,
  then validates it through the same fail-closed gate. Full-precision shadow
  and legacy CPU oracle rows are rejected as promotion evidence; promoted
  runtime rows must report Metal fused decode successes.
- `build_turboquant_decode_outputs.py`: extracts opt-in AX response
  `output_token_ids` from MLX inference-stack benchmark artifacts into the
  `decode_outputs` vector format consumed by TurboQuant quality metrics. Rerun
  the benchmark with `--capture-output-token-ids` when producing real-model
  quality evidence.
- `run-turboquant-quality-artifact.sh`: real-model TurboQuant promotion-evidence
  runner. It builds the release server, runs full-precision and
  `turboquant-fused-experimental` AX rows with output-token capture, extracts
  decode vectors, builds quality metrics, validates the quality artifact, and
  writes a promotion-readiness report. Its default repetitions match the
  repeated-measurement readiness contract, and its preflight rejects short
  context/generation shapes, zero-cooldown runs, and unsupported head dims
  before loading a model. Use `--dry-run` first to inspect inferred metadata and
  planned commands without loading a model.
- `verify_disk_prefix_cache_cross_restart.py`: F3 M4 disk prefix-cache
  cross-restart validator. It launches a cold phase that writes `.axkv`
  snapshots, then a fresh phase that must hit L2 and reproduce the same output
  token IDs. Use `PYTHONPATH=python` when running against the local extension
  without installing the package.
- `run-mlx-prefill-scaling-artifact.sh`: real-model P1 prefill/TTFT scaling
  runner. It runs the MLX inference-stack benchmark with direct AX rows, writes
  the raw `ax.mlx_inference_stack.v2` artifact, builds
  `ax.mlx_prefill_scaling.v1`, validates the scaling artifact, and renders a
  Markdown review report. Use `--dry-run` first to inspect the planned
  long-context run.
- `run-gateddelta-prefill-profile.sh`: real-model Qwen/GatedDelta prefill
  profile runner. It preflights the model manifest before building the release
  server, runs `--gateddelta-prefill-profile`, writes and validates the raw
  inference-stack artifact, and renders the Markdown stage-profile report. Use
  `--dry-run` first to inspect the planned profile run.
- `run-mlx-p2-latency-artifacts.sh`: real-model P2 startup/concurrency wrapper.
  It builds the release AX server, invokes `run_mlx_p2_latency_artifacts.py`,
  and writes `startup-latency.json`, `concurrent-prefill.json`, and
  `p2-latency.md`. Use `--dry-run` first to inspect the output directory and
  generated command without building or starting the server.
- `build_mlx_prefill_scaling_artifact.py`: converts completed MLX
  inference-stack artifacts into the fail-closed prefill/TTFT scaling artifact
  consumed by `check_mlx_prefill_scaling_artifact.py`.
- `check_mlx_prefill_scaling_artifact.py`: validates long-context prefill/TTFT
  evidence, including `mlx_lm` baseline coverage, direct AX policy labeling,
  shared prompt hashes, TTFT, peak memory, and ratios to baseline.
- `render_mlx_prefill_scaling_report.py`: renders a validated prefill-scaling
  artifact as a Markdown review table with prefill, TTFT, memory, ratios, and
  first-bend marking.
- `check_mlx_prefill_scaling_campaign.py`: validates a multi-model prefill
  scaling campaign by checking per-artifact contracts, required model-family
  coverage, host consistency, maximum context coverage, and first-bend summary.
- `check_mlx_prefill_claim_cycle.py`: aggregate checked-in W0-W4 claim-cycle
  gate. It runs the README public/narrative checker, the canonical P1
  long-context boundary artifact, the canonical P2 concurrent-prefill boundary
  artifact, and the W4 forward-profile diagnostic artifact without loading a
  model.
- `test_mlx_prefill_claim_cycle.py`: unit tests for the aggregate claim-cycle
  gate.
- `check_mlx_startup_latency_artifact.py`: validates P2 cold-vs-warm startup
  artifacts. It requires `process_cold`, `model_warm`, and `benchmark_warm`
  rows for the same prompt hash and direct AX policy, separates server-ready
  and model-load metrics from warm rows, and checks cold/warm ratios against
  the benchmark-warm row.
- `test_mlx_startup_latency_artifact.py`: unit tests for the startup latency
  artifact validator.
- `check_mlx_concurrent_prefill_artifact.py`: validates P2 concurrent-prefill
  artifacts. It requires a single-request baseline plus multi-request rows,
  one prompt hash per request, direct AX policy, per-request TTFT, total wall
  time, queue delay, zero failures, peak memory, overlap classification, and
  ratios to the single-request baseline.
- `test_mlx_concurrent_prefill_artifact.py`: unit tests for the concurrent
  prefill artifact validator.
- `check_mlx_prefix_warmup_artifact.py`: validates physical prefix snapshot
  miss/warmup correctness artifacts. It requires logical prefix reuse, an
  eligible MLX physical snapshot miss, warmup tokens, zero physical snapshot
  hits, zero blocked snapshot paths, and deterministic correctness evidence.
- `test_mlx_prefix_warmup_artifact.py`: unit tests for the prefix warmup
  artifact validator.
- `build_mlx_prefix_warmup_artifact.py`: builds `ax.mlx_prefix_warmup.v1` from
  an `ax-engine-bench` result directory by combining manifest events, route
  prefix counters, trace prefix-reuse items, and correctness/determinism gates.
- `test_build_mlx_prefix_warmup_artifact.py`: unit tests for the prefix warmup
  artifact builder.
- `check_mlx_forward_profile_artifact.py`: validates diagnostic
  `AX_MLX_LINEAR_ATTENTION_PROFILE=1` artifacts and matched split/packed
  projection-pack comparisons. It keeps candidate wins diagnostic-only and
  rejects packed-projection public claim fields. Its stricter
  `--require-pack-candidate-win`, `--min-pack-candidate-wins`, and
  `--min-pack-candidate-win-prompts` modes are for promotion/default-enable
  gates; use `--min-pack-candidate-win-shapes` when generation length matters.
- `test_mlx_forward_profile_artifact.py`: unit tests for the forward-profile
  artifact validator.
- `run_mlx_p2_latency_artifacts.py`: real-model P2 runner for startup and
  concurrent-prefill evidence. It starts the AX MLX server in direct mode,
  writes prompt-token artifacts, captures `ax.mlx_startup_latency.v1` and
  `ax.mlx_concurrent_prefill.v1`, validates both outputs, and writes
  `p2-latency.md` before returning. Use `--dry-run` first to inspect the output
  paths without starting a server.
- `test_run_mlx_p2_latency_artifacts.py`: unit tests for the P2 latency runner
  artifact assembly, ratio calculations, and dry-run CLI.
- `render_mlx_p2_latency_report.py`: renders validated P2 startup and/or
  concurrent-prefill artifacts as a Markdown review report with cold/warm
  ratios, concurrency ratios, queue delay, memory pressure, and overlap
  classification.
- `test_render_mlx_p2_latency_report.py`: unit tests for the P2 latency report
  renderer.
- `build_turboquant_quality_metrics.py`: compares baseline and candidate decode
  output vectors and emits the max/mean absolute error plus minimum cosine
  similarity JSON consumed by the TurboQuant quality artifact builder.
- `test_turboquant_quality_artifact.py`: unit tests for the TurboQuant quality
  artifact validator.
- `test_turboquant_microbench_artifact.py`: unit tests for the TurboQuant fused
  microbenchmark artifact validator.
- `probe_mlx_model_support.py`: support-contract probe for downloaded MLX
  model artifacts. It reads `config.json`, safetensors index metadata, and
  local reference implementations so new architectures fail closed with named
  blockers instead of becoming benchmark-only support claims.
- `test_probe_mlx_model_support.py`: unit tests for GLM/DeepSeek support
  classification and fail-closed partial-reference behavior.
- `check-bench-inference-stack.sh`: lightweight contract check for the MLX
  inference-stack benchmark harness. It does not load a model.
- `check-turboquant-quality-gate.sh`: lightweight CLI pipeline check for
  TurboQuant quality evidence. It builds synthetic quality metrics, compiles a
  quality artifact, validates it, and proves `full_precision_shadow` candidates
  fail promotion.
- `check-turboquant-microbench-gate.sh`: lightweight CLI pipeline check for
  TurboQuant fused-kernel evidence. It validates a synthetic speed-positive
  `two_stage_scores` artifact and proves a CPU-regressed artifact fails.
- `check_turboquant_public_docs.py`: lightweight public-docs contract check for
  the optional TurboQuant switch, telemetry-only shadow boundary, and sync
  timing docs.
- `check_turboquant_promotion_readiness.py`: fail-closed readiness report for
  TurboQuant public-support promotion. It scans local model manifests and
  quality-gate artifacts, then reports whether public docs must remain
  experimental. Passing quality/path evidence alone is not enough for public
  promotion when the decode-throughput performance gate is still blocked.
- `check_turboquant_prd_completion.py --fail-on-blockers`: final TurboQuant PRD
  closeout gate. Use this only after recording model-family quality evidence,
  D3 fused microbench evidence, D4 short-decode speedup evidence, and a passing
  promotion-readiness report.
- `cargo run -p ax-engine-microbench --release --bin turboquant-microbench -- ...`:
  TurboQuant fused cold-decode microbenchmark. It compares the K8/V4 MLX/Metal
  kernels against the CPU reference oracle and writes
  `ax.turboquant_fused_decode_microbench.v1` JSON artifacts. Use `--variants`
  to limit longer sweeps to selected kernel variants and `--hot-tokens` to
  include shared log-sum-exp hot-tail merge evidence.
- `reproduce-mlx-inference-benchmark.sh`: public reproduction wrapper for
  external Apple Silicon benchmark bundles. It records doctor output, command
  logs, prompt artifacts, environment metadata, and raw JSON results.
- `diagnose_server_rss.py`: long-lived RSS diagnostic for MLX and delegated
  llama.cpp server routes. It is not a throughput benchmark.
- `check-bench-*.sh`: smoke checks for `ax-engine-bench` workload-contract
  commands. `check-bench-preview.sh` covers the delegated llama.cpp preset and
  artifact contract, including safe preset metadata, processing/deferred
  request events, KV usage when available, and backend prompt-cache reuse.
- `check-server-preview.sh`, `check-python-preview.sh`: preview transport and
  binding smoke checks.
- `build-metal-kernels.sh`, `check-metal-kernel-contract.sh`: Metal artifact
  build and contract checks.

Run the MLX inference-stack unit tests without loading a model:

```text
bash scripts/check-bench-inference-stack.sh
```

Run the fast script hygiene gate before changing files in this directory:

```text
bash scripts/check-scripts.sh
```

Run the targeted Gemma/AX MLX telemetry gate:

```text
bash scripts/check-mlx-telemetry.sh
```

When a telemetry change touches shared Rust contracts, add the protected full
workspace pass:

```text
bash scripts/check-mlx-telemetry.sh --full-workspace
```
