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

Use `bench_ax_multimodel_serving.py` for timed request/load/unload replay
against one multi-model AX process. It adds per-model output identities,
lifecycle latency, final route-counter contracts, and model-switch evidence.
Use `bench_qwen_gemma_flip_target.py` when the same Qwen 3 + Gemma 4 replay
must run through the raw OpenAI completions streaming API on either AX or
mlxcel. Its target file locks the host, model packages, sampling parameters,
memory budget, protocol, and runtime revision. For mlxcel it also supervises
one process per model so load/unload events retain their lifecycle meaning;
the AX target supervises one fresh multi-model process per artifact so repeated
runs cannot inherit prompt-cache state.
The checked-in mlxcel target pins the official v0.4.2 source revision.
Use `run_qwen_gemma_flip_campaign.py` to alternate both targets across three or
more S0-S3 repetitions with a cooldown, validate every artifact, and prove the
prompt-token and package contract for each paired trial.
Use `summarize_qwen_gemma_flip_campaign.py` to compute the per-scenario medians,
apply the locked gates, and write the explicit `flip` or `not_yet` JSON and
Markdown decision.
Use `certify_row_exact_coalesced_decode.py` to compare the production
Qwen/Gemma row-exact decode route with its independent sequential oracle and,
optionally, the non-coalesced serving baseline.

Use `bench_embedding_fair.py` for published embedding comparisons between
`mlx-lm` and ax-engine, or pass `--ax-only` for README-style AX-only refreshes.
That harness excludes HTTP/cold-start paths and forces the measured engine(s) to
materialize the same contiguous CPU `float32 [B,H]` output matrix, then reports
batch-size and token-length scaling separately. Use
`bench_embedding_ingest_scale.py` for sustained RAG/indexing profiles where a
large deterministic chunk corpus is split into repeated batches and p95 batch
latency matters beside tok/s. Keep `bench_embedding_models.py` for legacy
embedding smoke coverage across single-call, HTTP, and optional Swift paths.
Before treating any embedding throughput row as publishable, run
`verify_embedding_models.py` against the same model artifact. It checks
family-specific numerical correctness against `mlx-lm` for Qwen3 embeddings and
against `mlx-embeddings` for EmbeddingGemma, plus AX direct batch-vs-single
stability. EmbeddingGemma intentionally compares against the reference
single-row path because the upstream mixed-length batch path is not a stable
correctness oracle.

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
- `brew-release.sh`: legacy local Homebrew release preview. Production mutation
  is disabled; only `--dry-run` is accepted. Use `publish-github-release.sh`
  for current releases so Developer ID signing, notarization, Minisign,
  independent draft verification, GitHub publication, PyPI tag promotion, and
  the publisher-dispatched Homebrew update stay on the same release source.
- `minisign-keygen.sh`: generates the minisign keypair for signing release
  artifacts. Defaults to the shared AX key paths (`~/signkey/ax.sec`
  / `~/signkey/ax.pub`), sets a private `700` directory and `600` secret
  key, and refuses to overwrite an existing keypair unless passed `--force`.
  Supports `--allow-unencrypted-test-key` (short-lived tests only) and
  `--dry-run`.
- `minisign-artifact.sh`: signs one or more release artifacts with minisign.
  The default key is the shared AX product signing key at `~/signkey/ax.sec`
  / `~/signkey/ax.pub` (Keychain service `ax-minisign`, account
  `ax-release`). If the passphrase is already stored in the shared AX
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
  tree and successful GitHub CI for the exact release SHA, then creates or
  reuses the SHA-bound artifact from `.github/workflows/release-candidate.yml`.
  That macOS workflow builds the standalone binaries and PyPI wheel once and
  smoke-tests the wheel before publishing immutable candidate manifests.
  The local publisher verifies every binary digest, optionally signs and
  notarizes the three binaries with `--sign-identity`, writes a tarball,
  SHA256 file, and manifest under `target/release-artifacts/<tag>/`, signs those
  artifacts with minisign, pushes the tag, publishes and verifies the GitHub
  assets (including the repository-pinned `ax-minisign.pub`), then dispatches
  the Homebrew formula update. The tag-triggered PyPI
  workflow promotes the matching wheel instead of rebuilding it.
  Notarization can use the local
  `AX_NOTARY_PROFILE` / `--notary-profile` Keychain profile or the same
  App Store Connect API env shape used by ax-code Desktop:
  `APPLE_API_KEY`, `APPLE_API_KEY_B64`, `APPLE_API_KEY_ID`, and
  `APPLE_API_ISSUER`. When `APPLE_API_KEY_B64` is set and `APPLE_API_KEY` is
  not, the script decodes the key into a temporary `.p8` file with `0600`
  permissions before calling `notarytool`. Use `--dry-run` to exercise local
  checks/build/package steps without dispatching workflows, pushing, uploading,
  or submitting to Apple notarization. Use `--full-local-checks` to repeat all
  gates after exact-SHA CI, `--local-build` for an explicit local build, or
  `--skip-brew-dispatch` when intentionally publishing GitHub-only assets.
- `release_candidate.py`: writes and verifies the
  `ax.release_candidate.v1` manifest shared by the local GitHub publisher and
  PyPI promotion workflow. Candidate binaries and wheels are accepted only
  when their tag, full source commit, paths, sizes, and SHA-256 digests match.
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
  prefix-cache experiments. It also supports opt-in
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
- `bench_ax_multimodel_serving.py`: timed JSONL replay for request, model-load,
  and model-unload events against one AX process. It records per-model
  throughput and output hashes, lifecycle latency, stream gaps, and required
  final route counters.
- `certify_row_exact_coalesced_decode.py`: fail-closed Qwen/Gemma decode
  coalescing matrix. It requires the independent forward oracle and row-exact
  route, rejects diagnostic tensor batching, and can report speedup over the
  non-coalesced baseline.
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
- `bench_ax_multimodel_serving.py`: timed multi-model + lifecycle replay
  against a running AX server (`ax.multimodel_serving_benchmark.v1`). Emits
  focus-family tags for the Qwen 3 + Gemma 4 flip schedule, interactive
  stream-gap summaries, and request availability counters.
- `bench_qwen_gemma_flip_target.py`: shared OpenAI SSE runner for AX and the
  pinned official mlxcel peer. Target JSON controls process supervision,
  exact model artifacts, host identity, sampling, and a common aggregate
  memory cap; authoritative streamed usage is required for token accounting.
- `run_qwen_gemma_flip_campaign.py`: repeated S0-S3 campaign runner. It starts
  a fresh process topology for every trial, alternates target order, enforces
  the cooldown and minimum repetition count, validates each artifact, and
  records per-pair comparison-contract verdicts.
- `summarize_qwen_gemma_flip_campaign.py`: fail-closed median aggregator and
  flip-decision renderer for a completed campaign. It requires at least three
  trials per target/scenario and reports every failed gate.
- `check_ax_multimodel_serving_artifact.py`: fail-closed validator for
  multi-model artifacts. Supports `--require-focus-family`, interactive
  stream-gap caps, and HTTP 503 budgets used by the mlxcel flip plan.
- `compare_qwen_gemma_flip.py`: compares candidate vs baseline multi-model
  artifacts with provisional throughput / TTFT / stream-gap / availability
  gates. It fails closed unless scenario hashes, sampling, protocol, host,
  comparison contract, and model-package signatures match. Use
  `--report-only` during Week-1 calibration.
- `test_bench_ax_multimodel_serving.py`,
  `test_bench_qwen_gemma_flip_target.py`,
  `test_run_qwen_gemma_flip_campaign.py`,
  `test_summarize_qwen_gemma_flip_campaign.py`,
  `test_check_ax_multimodel_serving_artifact.py`,
  `test_compare_qwen_gemma_flip.py`: unit tests for the multi-model flip
  harness.
- `render_ax_serving_benchmark_report.py`: renders validated
  `ax.serving_benchmark.v1` artifacts as Markdown tables for TTFT, client TPOT,
  E2E latency, queue delay, token distributions, goodput, and category
  breakdowns. It accepts the same route-decision gate when the report is for a
  runtime-path-specific promotion claim.
- `test_bench_ax_serving.py`: unit tests for the serving benchmark SSE parser,
  latency accounting, percentile/goodput summary, and artifact writer.
- `test_bench_ax_multimodel_serving.py`: unit tests for timed multi-model
  replay, output identities, lifecycle results, and route-counter gates.
- `test_certify_row_exact_coalesced_decode.py`: unit tests for row-exact
  certification environment, route verdicts, timing parsing, and artifact
  writing.
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
- `check_decode_hot_path_kernel_admission.py`: fail-closed validator for
  `.internal/analysis/decode-hot-path-kernels/*/candidate.json` manifests. It
  implements the decode hot-path PRD admission checklist for promoted/default-on
  kernel or graph-fusion routes: profile source, mechanism statement,
  correctness oracle, 2 warmup + 5 measure microbench evidence, real-graph A/B,
  route counters, rollback, kill switch, and promotion threshold. Empty
  candidate roots pass so the checker can live in CI before a candidate exists.
- `test_check_decode_hot_path_kernel_admission.py`: unit tests for the decode
  hot-path admission validator.
- `check_no_turboquant_references.py`: repo-wide guard for the KV-quantization
  runtime path retired by ADR-002 in favor of the durable tiered prefix cache.
  It fails when references to the retired path appear outside historical
  benchmark artifacts, historical design docs, internal notes, and release
  notes.
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
- `check_mlx_prefill_claim_cycle.py`: aggregate W0-W4 claim-cycle gate. It
  always runs the README public/narrative checker. P1 long-context, P2
  concurrent-prefill, and W4 forward-profile boundaries are optional current
  artifacts supplied with `--prefill-scaling-artifact`,
  `--concurrent-prefill-artifact`, and `--forward-profile-artifact`; omitted
  boundaries are reported as skipped instead of falling back to stale
  checked-in results.
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
  paths without starting a server. For explicit warm-prefix comparisons, use
  `--enable-prefix-cache --shared-prefix --capture-output-token-ids`; the
  concurrent runner then records exact response tokens and the checker rejects
  missing/wrong-length IDs or any shared-prefix c1/cN token divergence; it also records relevant paged-KV
  environment flags, native prefix hit/store/eviction counters, fixed-slab
  count/KiB/grow gauges, ready-state RSS, and in-flight process RSS
  high-water. A failed, cancelled, output-less, or unterminated SSE response
  fails the trial instead of being coerced into a numeric sample. Use macOS
  `sample`/`footprint` alongside the artifact when Metal-aware memory is part
  of the claim; RSS alone is insufficient.
- `test_run_mlx_p2_latency_artifacts.py`: unit tests for the P2 latency runner
  artifact assembly, ratio calculations, and dry-run CLI.
- `render_mlx_p2_latency_report.py`: renders validated P2 startup and/or
  concurrent-prefill artifacts as a Markdown review report with cold/warm
  ratios, concurrency ratios, queue delay, memory pressure, and overlap
  classification.
- `test_render_mlx_p2_latency_report.py`: unit tests for the P2 latency report
  renderer.
- `probe_mlx_model_support.py`: support-contract probe for downloaded MLX
  model artifacts. It reads `config.json`, safetensors index metadata, and
  local reference implementations so new architectures fail closed with named
  blockers instead of becoming benchmark-only support claims.
- `test_probe_mlx_model_support.py`: unit tests for GLM/DeepSeek support
  classification and fail-closed partial-reference behavior.
- `check-bench-inference-stack.sh`: lightweight contract check for the MLX
  inference-stack benchmark harness. It does not load a model.
- `reproduce-mlx-inference-benchmark.sh`: public reproduction wrapper for
  external Apple Silicon benchmark bundles. It records doctor output, command
  logs, prompt artifacts, environment metadata, and raw JSON results.
- `diagnose_server_rss.py`: long-lived RSS diagnostic for MLX and delegated
  llama.cpp server routes. It is not a throughput benchmark.
- `run_native_generation_fault_soak.py`: fail-closed native MLX fault lane that
  mixes normal, disconnected, and slow consumers while recording request
  outcomes, RSS, queue/backlog counters, and final lifecycle quiescence.
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
