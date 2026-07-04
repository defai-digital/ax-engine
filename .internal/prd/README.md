# AX Engine PRDs

This is the consolidated PRD rollup for AX Engine. It lives beside the per-feature PRDs and indexes the current product direction. The per-feature files are kept as the detailed source of truth; this document is the map.

- **Scope:** this directory only. ADRs (`../adr/`) and implementation plans (`../plan/`) are governed separately.
- **Maintenance:** update the index table when a PRD lands, is superseded, or is newly proposed. Use the **status legend** below.
- **Last consolidated:** 2026-07-04. Version baseline: **6.6.1**.

## Status legend

- `active` — current direction, work open or in progress.
- `shipped` — landed in the codebase; kept as reference. Read for context, not as open work.
- `superseded` — a later PRD replaces it; see the *superseded-by* column. Do not action.
- `expired` / `unfit` — timeline passed with no landing, or draft abandoned. (None currently.)

## Current product direction (active epics)

Grouped by theme. Each lists its **authoritative PRD(s)** and the next open milestone.

### 1. Speculative decoding (MTP + n-gram)
Architecture and the UX surface are landed; remaining work is gating/promotion and Gemma-family tuning.

- **Authoritative — architecture:** `PRD-2026-05-31-transactional-speculative-decode.md` (ADR-011)
- **Authoritative — config UX:** `PRD-2026-06-09-speculation-profile-presets.md` (ADR-022, shipped — `--speculation-profile`)
- **Open:** `PRD-2026-06-07-mtp-ngram-utility-gating.md` — code-level gating landed; benchmark promotion report (keep/skip/revert per model) pending.
- **Open (TTFT):** `PRD-2026-06-01-mtp-ttft-optimization.md` — warmup cap + startup shader warm (complementary to phase2).
- **Open (decode defaults):** `PRD-2026-06-01-mtp-decode-optimization.md` — partly folded into phase2; remaining safe-defaults work open.
- **Shipped (Qwen Phase 3):** `PRD-2026-06-18-qwen-mtp-decode-phase3.md` — profile gates, stochastic draft, and auto-optimistic behavior implemented; P4 deferred.

### 2. Quantization (TurboQuant) + offline policy search

- **Authoritative — promotion:** `TURBOQUANT-PROMOTION-PRD.md` — Phases 0/1 done; 2 blocking phases remain (correctness/quality/perf gates).
- **Authoritative — search:** `QUANTUM-OP-PRD.md` — `ax.offline_policy_search.v1` guardrails landed; TurboQuant runtime promotion blocked on 4 milestones.
- **Open:** `PRD-2026-07-04-open-tq-metal-int4-kv-attention.md` — separate Open-TQ-Metal K4/V4 compressed-domain attention track; phase-0 support classification landed.
- **Open:** `PRD-2026-06-01-turboquant-codec-kernel-improvements.md` — code implemented; model/short-decode/promotion evidence + clippy pending.
- **Open:** `PRD-2026-05-27-experimental-low-bit-mlx-quantization.md` — gated experimental 3-bit/2-bit path; direct vs n-gram audit incomplete.

### 3. Decode & prefill speed (direct-mode throughput)

- **Open (MoE):** `PRD-2026-06-19-qwen3-coder-next-moe-decode-prefill-phase2.md` — Qwen3-Coder-Next decode/prefill throughput and MoE dispatch/bandwidth work.
- **Open (MoE graph compile):** `PRD-2026-06-23-graph-level-moe-decode-compilation.md` — graph-level MoE decode compilation for standard attention families (GLM/Qwen3 MoE); ADR-032.
- **Open (kernel strategy):** `PRD-2026-07-03-decode-hot-path-kernel-strategy.md` — evidence-gated decode hot-path kernel admission gate landed; first runtime promotion still candidate-specific; ADR-034.
- **Open (proposed):** `PRD-2026-06-01-decode-speed-optimization.md` — CPU hot-path elimination (37 opportunities).
- **Open (proposed):** `PRD-2026-06-11-direct-mode-bandwidth-speed.md` — prioritize bytes/token reductions over host fusions.
- **Open:** `PRD-2026-06-01-simdgroup-matrix-kernels.md` — simdgroup_matrix GEMV/attention kernels (default-OFF, behind `AX_MLX_GEMV_SIMDGROUP_MATRIX`).
- **Precursor:** `DECODE-SPEED-PRD.md` — sampling-runner CPU overhead (overlaps, not replaced).

### 4. KV / prefix cache

- **Open:** `MLX-DISK-PREFIX-CACHE-PRD-2026-05-14.md` — disk-durable L2 prefix cache; M1–M5 landed; AX-serving soak + final promotion open.

### 5. Serving, scheduling & routes

- **Open:** `PRD-2026-05-26-cache-local-speculative-serving.md` — cache-aware scheduling, prefix reuse, evidence-gated speculation policy.
- **Open:** `LOCAL-SERVE-CLI-PRD.md` — `ax-engine serve`/`download`/`convert-mtplx` shipped (P0); daemon/status/kill pending.

### 6. Model families & multimodal

- **Open:** `PRD-2026-06-08-gemma4-unified-12b-support.md` — native MLX Gemma 4 12B unified; text + multimodal tensor path landed; local runtime smoke blocked (no Metal device in this env).
- **Open (proposed):** `PRD-2026-06-09-gemma4-12b-multimodal-benchmarking.md` — reproducible image/audio/video benchmark matrix.
- **Open (DiffusionGemma Phase 1):** `PRD-2026-06-18-diffusiongemma-denoise-optimization.md` — convergence diagnostics and denoise-loop optimization.
- **Open (DiffusionGemma Phase 2):** `PRD-2026-06-18-diffusiongemma-dispatch-optimization-phase2.md` — commit skipping, KV concat cache, and dispatch reduction.

### 7. SDK / Server / API surface

- **Open:** `PRD-2026-06-10-agentic-api-readiness.md` — API-key auth (M1) done; `/metrics`, logprobs, reasoning extraction, structured output, tool calling (M2–M6) open.
- **Open:** `PRD-2026-05-25-sdk-server-interface-hardening.md` — transport/session/Python/OpenAI-compat hardening (continuous).

### 8. Bench tooling, release & release evidence

- **Open:** `PRD-2026-05-25-benchmark-evidence-tooling.md`, `PRD-2026-05-25-runtime-model-readiness.md`, `PRD-2026-06-16-minisign-release-signing.md`.

## Shipped (landed — reference only)
| Epic | PRD | Note |
|---|---|---|
| MTP config UX | `PRD-2026-06-09-speculation-profile-presets` | `--speculation-profile` |
| MTP n-gram gating | `PRD-2026-06-02-mtp-ngram-lightning-learnings` | hurt/saturation gates |
| MTP impl burst | `PRD-2026-06-01-mtp-fused-lazy-draft-skip-state`, `PRD-2026-06-01-mtp-optimization-phase2` | ADR-012/013 |
| MTP acceptance | `MTP-ACCEPT-RATE-PRD`, `PRD-2026-05-27-mtp-candidate-refinement`, `PRD-2026-05-27-mtp-rapid-mlx-learnings` | ADR-006→010 |
| Gemma4 MTP | `GEMMA4-MTP-PRD`, `PRD-2026-06-09-gemma4-12b-mtp-speedup` | opt-in assistant-MTP default |
| Qwen MTP Phase 3 | `PRD-2026-06-18-qwen-mtp-decode-phase3` | profile gates, stochastic draft, auto-optimistic |
| API baseline | `PRD-2026-06-15-openai-primary-provider-baseline` | OpenAI /v1 primary, Ollama adapter |
| HF tooling | `HF-INTEGRATION-PRD` | `download_model.py`, manifest tooling |
| Interactive downloader | `INTERACTIVE-DOWNLOADER-PRD` | wizard + live progress UI in `python/ax_engine/_cli.py` |

## Superseded
| PRD | Superseded by | Why |
|---|---|---|
| `DS4-LEARNINGS-FOLLOWUP-PRD-2026-05-14.md` | `MLX-DISK-PREFIX-CACHE-PRD-2026-05-14.md` (F3 child) | Phase A/B/C closed; F3 landed as child PRD; F1/F2/F4/F5 not pursued as separate PRDs |

> **Partial supersession (not retired):** `PRD-2026-06-01-mtp-decode-optimization.md` content was largely folded into `PRD-2026-06-01-mtp-optimization-phase2.md`; the former is kept active for its remaining safe-defaults scope.

> **Never written:** a `PRD-2026-06-16-decode-prefill-speed-phase2.md` (linear-attention last-position-only, MoE narrow router softmax, MoE fusion relaxation to prefill) was listed as "approved" in the 2026-06-20 rollup but the file was never created. Its MoE scope is now tracked under `PRD-2026-06-19-qwen3-coder-next-moe-decode-prefill-phase2.md` and `PRD-2026-06-23-graph-level-moe-decode-compilation.md`. The dangling index entry was removed on 2026-06-24.

## Full index (all 42 PRDs)

### Speculative decoding — MTP / n-gram (14)
| File | Date | Status |
|---|---|---|
| `PRD-2026-06-18-qwen-mtp-decode-phase3.md` | 2026-06-18 | shipped |
| `PRD-2026-06-09-gemma4-12b-mtp-speedup.md` | 2026-06-09 | shipped |
| `PRD-2026-05-31-transactional-speculative-decode.md` | 2026-05-31 | **active** (authoritative arch) |
| `PRD-2026-06-09-speculation-profile-presets.md` | 2026-06-09 | shipped (authoritative UX) |
| `PRD-2026-06-07-mtp-ngram-utility-gating.md` | 2026-06-07 | active (promo pending) |
| `PRD-2026-06-02-mtp-ngram-lightning-learnings.md` | 2026-06-02 | shipped |
| `PRD-2026-06-01-mtp-ttft-optimization.md` | 2026-06-01 | active (proposed) |
| `PRD-2026-06-01-mtp-decode-optimization.md` | 2026-06-01 | active (partly folded → phase2) |
| `PRD-2026-06-01-mtp-optimization-phase2.md` | 2026-06-01 | shipped |
| `PRD-2026-06-01-mtp-fused-lazy-draft-skip-state.md` | 2026-06-01 | shipped |
| `PRD-2026-05-27-mtp-rapid-mlx-learnings.md` | 2026-05-27 | shipped |
| `PRD-2026-05-27-mtp-candidate-refinement.md` | 2026-05-27 | shipped |
| `MTP-ACCEPT-RATE-PRD.md` | — | shipped |
| `GEMMA4-MTP-PRD.md` | — | shipped (opt-in) |

### Quantization / TurboQuant (5)
| File | Date | Status |
|---|---|---|
| `PRD-2026-07-04-open-tq-metal-int4-kv-attention.md` | 2026-07-04 | active (draft; phase-0 contract landed) |
| `TURBOQUANT-PROMOTION-PRD.md` | 2026-05-09 | active (blocking phases) |
| `QUANTUM-OP-PRD.md` | 2026-05-14 | active (partially implemented) |
| `PRD-2026-06-01-turboquant-codec-kernel-improvements.md` | 2026-06-01 | active (evidence pending) |
| `PRD-2026-05-27-experimental-low-bit-mlx-quantization.md` | 2026-05-27 | active |

### Decode / prefill / TTFT speed (7)
| File | Date | Status |
|---|---|---|
| `PRD-2026-07-03-decode-hot-path-kernel-strategy.md` | 2026-07-03 | active (admission gate implemented) |
| `PRD-2026-06-23-graph-level-moe-decode-compilation.md` | 2026-06-23 | active (draft) |
| `PRD-2026-06-19-qwen3-coder-next-moe-decode-prefill-phase2.md` | 2026-06-19 | active (draft) |
| `PRD-2026-06-11-direct-mode-bandwidth-speed.md` | 2026-06-11 | active (proposed) |
| `PRD-2026-06-01-decode-speed-optimization.md` | 2026-06-01 | active (proposed) |
| `PRD-2026-06-01-simdgroup-matrix-kernels.md` | 2026-06-01 | active |
| `DECODE-SPEED-PRD.md` | — | active (precursor) |

### KV / prefix cache & serving (3)
| File | Date | Status |
|---|---|---|
| `MLX-DISK-PREFIX-CACHE-PRD-2026-05-14.md` | 2026-05-14 | active (M1–M5 landed) |
| `DS4-LEARNINGS-FOLLOWUP-PRD-2026-05-14.md` | 2026-05-14 | **superseded** → disk-prefix-cache |
| `PRD-2026-05-26-cache-local-speculative-serving.md` | 2026-05-26 | active |

### Model families & multimodal (4)
| File | Date | Status |
|---|---|---|
| `PRD-2026-06-18-diffusiongemma-dispatch-optimization-phase2.md` | 2026-06-18 | active (draft) |
| `PRD-2026-06-18-diffusiongemma-denoise-optimization.md` | 2026-06-18 | active (draft) |
| `PRD-2026-06-08-gemma4-unified-12b-support.md` | 2026-06-08 | active (runtime smoke blocked) |
| `PRD-2026-06-09-gemma4-12b-multimodal-benchmarking.md` | 2026-06-09 | active (proposed) |

### SDK / Server / API (4)
| File | Date | Status |
|---|---|---|
| `PRD-2026-06-10-agentic-api-readiness.md` | 2026-06-10 | active (M1 done) |
| `PRD-2026-06-15-openai-primary-provider-baseline.md` | 2026-06-15 | shipped |
| `PRD-2026-05-25-sdk-server-interface-hardening.md` | 2026-05-25 | active |
| `LOCAL-SERVE-CLI-PRD.md` | — | active (P0 shipped) |

### Tooling, release & readiness (5)
| File | Date | Status |
|---|---|---|
| `PRD-2026-05-25-benchmark-evidence-tooling.md` | 2026-05-25 | active |
| `PRD-2026-05-25-runtime-model-readiness.md` | 2026-05-25 | active |
| `PRD-2026-06-16-minisign-release-signing.md` | 2026-06-16 | active (accepted) |
| `HF-INTEGRATION-PRD.md` | — | shipped |
| `INTERACTIVE-DOWNLOADER-PRD.md` | — | shipped (core) |

## Tally
active: 28 · shipped: 13 · superseded: 1 · expired: 0 · unfit: 0 (42 total)
