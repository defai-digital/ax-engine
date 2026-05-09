# Benchmarking System Design

AX Engine keeps benchmarking deliberately split by evidence type. Two separate
tools produce two separate classes of artifacts, and the rules for which numbers
can be cited as which kind of claim follow directly from which tool produced them.

## Two-Tool Split

```
bench_mlx_inference_stack.py       ←  MLX throughput comparison
  Produces: tok/s ratios vs mlx_lm.benchmark baseline
  Answers:  "How fast is AX vs upstream MLX?"

ax-engine-bench                    ←  workload contract CLI
  Produces: workload-contract artifacts (correctness, route, replay)
  Answers:  "Did this workload pass its declared gates?"
```

These tools are intentionally separate. `bench_mlx_inference_stack.py` is the
only tool that can produce repo-owned MLX throughput claims. Every such claim
requires a matching `mlx_lm.benchmark` baseline row for the same random-token
prompt/decode shape. `ax-engine-bench` owns workload contracts: correctness
verification, route identity, prefix reuse, replay determinism, and regression
review. Merging these rows into one unlabeled table would make the evidence
unauditable.

---

## ax-engine-bench CLI

### Subcommands

| Subcommand | Purpose |
|---|---|
| `scenario` | Run a shape-driven workload; emit execution artifacts |
| `replay` | Run a timed-event workload; emit execution artifacts |
| `matrix` | Roll up multiple scenario manifests; emit matrix summary |
| `compare` | Diff two execution artifact dirs; emit comparison artifacts |
| `matrix-compare` | Diff two matrix artifact dirs; emit roll-up comparison |
| `baseline` | Promote a successful artifact dir to a named trusted baseline |
| `autotune` | Bounded knob search over a frozen manifest |
| `doctor` | Inspect local host readiness for repo-owned MLX benchmarking |
| `metal-build` | Compile custom Metal kernels from `metal/phase1-kernels.json` |
| `generate` | One-shot inference (development utility, not for benchmark evidence) |
| `stream` | Streaming inference (development utility, not for benchmark evidence) |

### Exit codes

Exit codes are typed so CI pipelines can distinguish failure categories without
parsing output text:

| Code | `CliError` variant | Meaning |
|---|---|---|
| 0 | — | Success; all gates passed |
| 1 | `Usage` / `Runtime` | Bad flags or infrastructure failure (infra, not workload) |
| 2 | `Contract` | Manifest violated its own declared contract before or during execution |
| 3 | `Correctness` | Correctness or determinism gate failed after execution completed |
| 4 | `Performance` | Performance threshold violated (reserved; not yet enforced at CLI) |

---

## Manifest Format

Each scenario or replay run is driven by a `BenchmarkManifest` JSON file. The
manifest is the source of truth for the workload: it captures the model, runtime
configuration, sampling policy, shape or event sequence, declared checks, and
provenance information. Manifests are checked into `benchmarks/manifests/`.

### Top-level fields

```json
{
  "schema_version": "ax.bench.v1",
  "id": "chat_qwen_short",
  "class": "scenario",            // or "replay"
  "scenario": "dense_prefill_decode_chat",
  "model": { ... },
  "runtime": { ... },
  "sampling": { ... },
  "shape": { ... },               // scenario only
  "events": [ ... ],              // replay only
  "checks": { ... },
  "notes": "optional free text"
}
```

### model

```json
{
  "family": "qwen3_5",
  "revision": "mlx-community/Qwen3-9B-4bit",
  "quant": "4bit",
  "tokenizer_revision": "...",
  "chat_template_revision": "..."
}
```

### runtime

```json
{
  "selected_backend": "mlx",
  "support_tier": "mlx_preview",
  "resolution_policy": "mlx_only",
  "deterministic": true,
  "max_batch_tokens": 2048,
  "kv_total_blocks": 1024,
  "flags": { "prefix_cache": true }
}
```

For delegated backends, `backend_adapter` carries the llama.cpp CLI path or
server URL, and `llama_cpp_preset` captures the parallel slot, batching, and
cache-prompt configuration that must otherwise live in llama.cpp launch flags.
This ensures the delegated controls are auditable from the artifact.

### sampling

```json
{
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": 0,
  "seed": 0
}
```

### shape (scenario only)

```json
{
  "input_tokens_target": 512,
  "output_tokens_target": 128,
  "concurrency": 1
}
```

`input_tokens_target` drives prompt construction; `concurrency` controls how
many requests are submitted before the step loop starts draining.

### events (replay only)

```json
[
  { "t_ms": 0,    "type": "submit", "request_id": "r1", "prompt_ref": "shared_512", "prefix_group": "sys", "body_group": "user_a", "output_tokens_target": 128 },
  { "t_ms": 100,  "type": "submit", "request_id": "r2", "prompt_ref": "shared_512", "prefix_group": "sys", "body_group": "user_b", "output_tokens_target": 128 },
  { "t_ms": 5000, "type": "cancel", "request_id": "r1" }
]
```

`t_ms` is the wall-clock offset from run start. `prefix_group` + `body_group`
control which prompt segments are shared across requests, enabling deterministic
prefix-reuse scenarios. The manifest is replayed in the same event order on
every run, so determinism verification can compare per-request results.

### checks

```json
{
  "expect_deterministic": true,
  "require_prefix_reuse": false,
  "require_no_allocator_churn_failure": false
}
```

| Field | Effect |
|---|---|
| `expect_deterministic` | Triggers a second run; both runs must produce identical per-request output token sequences |
| `require_prefix_reuse` | Correctness gate fails if prefix hit rate is zero |
| `require_no_allocator_churn_failure` | Correctness gate fails if any request reached `Failed` state during allocator churn |

### Matrix manifest

A matrix manifest is a separate file that references multiple scenario manifests:

```json
{
  "schema_version": "ax.bench.v1",
  "id": "mlx_dense_phase7",
  "class": "scenario_matrix",
  "members": [
    { "manifest": "benchmarks/manifests/scenario/chat_qwen_short.json" },
    { "manifest": "benchmarks/manifests/scenario/shared_prefix_long.json", "label": "prefix-long" }
  ]
}
```

`matrix` runs each member manifest and produces a roll-up summary.
`matrix-compare` diffs two matrix artifact dirs at the member level.

---

## Artifact Schema

### Successful run

Every successful `scenario` or `replay` run writes a timestamped directory
under `--output-root` named `{timestamp}-{manifest-id}/`:

```
manifest.json       — copy of the input manifest
environment.json    — run ID, host, timing, runtime identity, gate results
metrics.json        — per-request and aggregate throughput metrics
routes.json         — per-request RouteMetadata (execution plan, KV mode, etc.)
trace.json          — per-step engine trace (scheduler decisions, token counts)
summary.md          — human-readable Markdown summary
```

`environment.json` records the run ID, command, manifest path, started/completed
timestamps, selected backend, support tier, resolution policy, host metadata,
gate pass/fail results, and a `status` field (`passed`, `correctness_failed`,
`determinism_failed`).

`routes.json` captures the `GenerateRouteReport` for each request, including
all `crossover_decisions` telemetry keys. This is the auditable record of which
execution plan, KV mode, and prefix cache path each request used.

### Contract failure

When the manifest contract check fails before or during execution, artifacts
are still written — under `{timestamp}-{manifest-id}-contract-failure/`:

```
manifest.json           — copy of the input manifest
contract_failure.json   — run ID, timing, failure message
summary.md              — human-readable failure summary
```

This ensures every failure is auditable. CI exit code 2 signals contract
failure; the artifact dir path is printed to stdout alongside the failure
message.

### Trusted baseline

`ax-engine-bench baseline --source <dir> --name "Dense Qwen Trusted"` copies a
successful artifact dir into `benchmarks/baselines/` under a stable name and
writes a `trusted_baseline.json` marker. The command fails closed if the
target already exists — overwriting a baseline requires manual deletion first.

`compare` loads the optional `trusted_baseline.json` from the baseline artifact
dir and includes it in the comparison output so regression evidence can be
labeled against a named checkpoint.

---

## Gate System

After execution, `enforce_runtime_gates` checks two gates in sequence. Both
must pass before exit code 0 is returned.

### Correctness gate

Evaluated by `evaluate_correctness`:

1. No request reached `Failed` state.
2. All requests that did not receive a cancel event reached `Finished` state
   (not stuck in a non-terminal state).
3. If `require_prefix_reuse = true`: at least one prefix hit was observed.
4. If a Cancel event was declared: the target request reached `Cancelled` state.
5. If `require_no_allocator_churn_failure = true`: no request reached `Failed`
   during churn.

Failure exits with code 3 (`CliError::Correctness`).

### Determinism gate

Only evaluated when `expect_deterministic = true`. The manifest is executed a
second time; the per-request output token sequences and route digests from both
runs are compared. Any divergence exits with code 3.

---

## Fail-Closed Principles

- Contract failures write artifacts before returning exit code 2. A CI system
  can always retrieve the artifact dir from the printed `result_dir=` line.
- `compare` refuses to compare a contract-failure artifact dir against a
  successful one (`reject_contract_failure_artifact_dir`).
- `baseline` refuses to overwrite an existing trusted baseline.
- `doctor` is authoritative on local readiness; no repo-owned MLX benchmark
  claim is valid without a passing doctor check on the same host.

---

## Tracing

Tracing is disabled by default during benchmark runs because it adds latency
to the step loop. Enable it selectively:

```bash
AX_BENCH_LOG=ax_engine_core=debug ax-engine-bench scenario ...
RUST_LOG=info ax-engine-bench replay ...
```

Both variables are checked; `AX_BENCH_LOG` takes priority. If neither is set,
the subscriber is not initialized at all — no `fmt` overhead.

Do not enable `trace`-level logging for throughput or latency measurements.
Use `info` or `warn` if you need to confirm route identity without affecting
timing.

---

## Evidence Interpretation

| Artifact | Valid claims | Invalid claims |
|---|---|---|
| `bench_mlx_inference_stack.py` with `mlx_lm.benchmark` baseline | Repo-owned MLX tok/s ratios against a named MLX reference | Workload correctness or determinism |
| `ax-engine-bench scenario/replay` | Route identity, correctness, determinism, prefix-reuse provenance, regression against baseline | Direct upstream MLX throughput comparison |
| `ax-engine-bench autotune` | Candidate evidence for bounded manifest knobs | Architecture decisions or cross-runtime ranking |
| `ax-engine-bench compare/matrix-compare` | Regression evidence within the same manifest and runtime family | Cross-runtime ranking or throughput claims |
| llama.cpp delegated artifacts | Delegated route-contract and backend prompt-cache behavior | Repo-owned MLX throughput |
| `mlx_lm_delegated` artifacts | AX surface compatibility with upstream `mlx_lm.server` | Repo-owned MLX throughput or token/KV accounting |

See [`docs/BENCHMARKS.md`](BENCHMARKS.md) for operational commands and
artifact capture procedures for each evidence type.
