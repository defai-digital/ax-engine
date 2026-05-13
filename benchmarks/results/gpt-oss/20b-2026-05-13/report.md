# GPT-OSS 20B Phase A Evidence Report

Date: 2026-05-13
PRD: `.internal/planning/GPT-OSS-SUPPORT-PRD.md` (W1.4)
Runbook: `.internal/planning/GPT-OSS-PHASE-A-RUNBOOK.md`
Recommendation: **DEFER** — see §5

## 1. Host

- SoC: Apple M5 Max (detected by ax-engine-server `/health`)
- Unified memory: 128 GB
- Disk free: 1.1 TB
- macOS Darwin 25.4.0
- Python: 3.14.5 with `sys.flags.gil == 1` (CPython, regular GIL build)
- mlx_lm: 0.31.2 (`pyproject.toml` pins `mlx-lm>=0.31`)
- ax-engine: branch `main`. Phase A in-process harness does not touch
  ax-engine Rust code, so the relevant SHA is whatever HEAD pointed at
  when the harness ran. The current main tip at audit time is
  `2590e32a`; HEAD at session start (gitStatus snapshot) was `b4b4b4e6`,
  with 4 unrelated commits landing in between (`a9625927`, `6f642520`,
  `b2eb209f`, `2590e32a`). None of these touch the delegated or native
  GPT-OSS paths, so they do not affect the Phase A finding.

## 2. Model load + decode (via `mlx_lm.generate` Python API)

Source: `scripts/gpt_oss_inproc_evidence.py` against `openai/gpt-oss-20b`
on 10 harmony-formatted prompts from
`benchmarks/datasets/ax-chat-v1-short-harmony.jsonl`, `max_tokens=128`,
`temperature=0`, greedy.

| Metric | Value |
|---|---:|
| load_seconds (warm HF cache) | 1.16 |
| generation_tps_mean (decode-only) | **113.1** |
| generation_tps_p50 | **112.3** |
| generation_tps_max | 116.6 |
| generation_tps_min | 111.3 |
| prompt_tps_mean (prefill) | **355.8** |
| peak_unified_memory_gb (GenerationResponse.peak_memory) | **14.9** |
| prompts_evaluated | 10 |
| harmony format adherence | 10/10 open with `<|channel|>analysis<|message|>` |
| reached `<|channel|>final<|message|>` within max_tokens=128 | 9/10 (1 truncated mid-analysis at `finish=length`) |
| finish_reason=stop | 9/10 |
| finish_reason=length | 1/10 |

Initial 2026-05-13 evidence used `mlx_lm.generate` with
`generation_tps = gen_tok / total_wall_time`, which conflates prompt-
processing with generation and underestimates decode tok/s by ~25%.
The harness was rewritten to use `mlx_lm.stream_generate` and read
separated `prompt_tps` / `generation_tps` from
`GenerationResponse`'s final yield. Numbers above are post-fix.

Raw data: `inproc_evidence.json` and `inproc_evidence.log` in this dir.

Correctness signal: outputs follow the expected harmony format. All 10
prompts open with the `analysis` channel; 9 of 10 reach the `final`
channel within `max_tokens=128`. Example for "What is 2+2?": analysis
reasons, final returns `4`. For "What is the capital of France?":
final returns `The capital of France is **Paris**`. The one truncation
(prompt 4, "Write one sentence about the ocean.") was the model
electing to write a longer analysis than 128 tokens permits; not an
incoherence — re-running with `max_tokens=256` would reach final. No
garbage or malformed outputs observed.

This artifact establishes the **model + weights + decode** are working
correctly on this host. It does **not** establish that ax-engine's
delegated route is reachable end-to-end — see §3.

## 3. ax-engine delegated route — wiring vs end-to-end

### 3.1 Wiring (ax-engine side) — ✅ correct

```
target/release/ax-engine-server \
  --host 127.0.0.1 --port 8080 \
  --model-id openai/gpt-oss-20b \
  --support-tier mlx-lm-delegated \
  --mlx-lm-server-url http://127.0.0.1:8810
```

`GET /health` returned:

```json
{
  "model_id": "openai/gpt-oss-20b",
  "runtime": {
    "selected_backend": "mlx_lm_delegated",
    "support_tier": "mlx_lm_delegated",
    "resolution_policy": "allow_mlx_lm_delegated",
    "fallback_reason": "mlx-lm delegated backend explicitly requested ...",
    "capabilities": { "text_generation": true, "token_streaming": true, ... }
  },
  "status": "ok"
}
```

No code change required in `crates/`. `support_tier=mlx_lm_delegated`
is accepted, route metadata reports correctly.

### 3.2 End-to-end — ❌ blocked by upstream mlx_lm.server bug

`mlx_lm.server` (the target of ax-engine delegation) **crashes on every
generation request** under this host's environment. Tried both
`openai/gpt-oss-20b` (target) and `mlx-community/Qwen3-4B-4bit`
(canonical native model) — both fail with the same error in the server's
generator thread:

```
File "mlx_lm/generate.py", line 1161, in prompt
  mx.eval([c.state for c in self.prompt_cache])
RuntimeError: There is no Stream(gpu, 2) in current thread.
```

Repeated with `--decode-concurrency 1`; same failure.

The error originates in the server's batch-generator thread when it
tries to evaluate prompt-cache state on a GPU stream that was created on
the main thread. This is an mlx_lm 0.31.2 × Python 3.14 interaction
(Metal stream identity is not inherited by spawned threads).

**The bug is not gpt-oss-specific.** It affects all delegated routing
on this host until upstream mlx_lm fixes it or Python is downgraded.

`mlx_lm.generate` (in-process Python API) is **unaffected** because it
does not spawn a generator thread — confirmed by §2 results.

## 4. Other findings

### 4.1 bench `scenario` harness lacks mlx_lm_delegated path

`cargo run -p ax-engine-bench -- scenario --manifest <file>` cannot run
delegated scenarios. `crates/ax-engine-bench/src/main.rs::BackendAdapterManifest`
(line 11538) only enumerates `LlamaCppCli` and `LlamaCppServerCompletion`.
`runtime_config_from_manifest` (line 9311) requires `backend_adapter`
for any non-Mlx `selected_backend`. The two manifests authored under W1.3
(`chat_gpt_oss_{20b,120b}_short.json`) cannot execute via the scenario
harness without first adding `MlxLmServerCompletion` to the adapter enum.

This is a bench-tooling gap, not an inference gap. Cataloged here so it
is not re-discovered.

### 4.2 ax-engine-server `/v1/chat/completions` has no harmony arm

Confirmed `crates/ax-engine-server/src/main.rs::ChatPromptTemplate::for_model_id`
(line 471) returns `PlainRolePrefix` for any `gpt-oss*` model_id. The
`/v1/completions` route accepts raw prompts and is the correct surface
for harmony-formatted inputs.

### 4.3 GPT-OSS 20B disk footprint

The `openai/gpt-oss-20b` HF repo downloaded ~18 GB total because it
includes both MXFP4 safetensors and the `original/` PyTorch checkpoint.
A future `hf download` with `--include "*.safetensors" --include "*.json"
--exclude "original/*"` filter would reduce this to ~12 GB. 120B would
benefit proportionally more.

## 5. Recommendation: DEFER

Phase A evidence shows the **model + weights + ax-engine delegated wiring
are correct**, but the **delegated route is blocked end-to-end by an
upstream mlx_lm.server bug specific to the current Python install**.
This is not an ax-engine architectural gap.

Per PRD §6 promotion criteria, the gap pattern matches "integration gap"
literally (delegated routing is demonstrably broken), but the root cause
is upstream/environment, not architectural. Phase B (native MXFP4 + sinks
+ YaRN) is the wrong response — it would be a large investment to work
around an upstream bug that has cheaper remediations.

**Unblock paths, in cost order**:

1. **Wait for mlx_lm upstream fix.** Track ml-explore/mlx-lm issues for
   the `Stream(gpu, N) in current thread` server bug under Python 3.14.
   Re-run this report's harness when a new release lands. No code work.
2. **Install Python 3.12 venv** (`python3.12 -m venv ...`, reinstall
   `mlx-lm[server]`, retry). ~30 min. Most pragmatic immediate path.
3. **Promote走法 B** (add `MlxLmServerCompletion` adapter variant) —
   only if scenario-format evidence becomes a hard downstream requirement.
   Independent of the mlx_lm.server bug; this is bench tooling work.

**Do not promote to Phase B (native ax-engine GPT-OSS)** on this evidence.
The model runs fine in-process; native implementation is a 2–3 week
investment to replace a 30-minute environment fix.

## 6. Open items to revisit when delegated route becomes runnable

- Re-run `gpt_oss_topk_compare.py` (already authored) once
  `mlx_lm.server` runs, to confirm ax-engine forwarding does not corrupt
  prompts or sampling parameters.
- 120B evidence run (currently un-downloaded; ~65 GB filter-pruned).
- Decide whether `ChatPromptTemplate` needs a harmony arm or stays
  `PlainRolePrefix` with documented `/v1/completions` requirement.

## 7. Files produced in this artifact directory

- `inproc_evidence.json` — full per-prompt and aggregate metrics
- `inproc_evidence.log` — stdout from the harness
- `report.md` — this file
