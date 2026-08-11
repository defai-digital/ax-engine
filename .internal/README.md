# AX Engine internal design index

This directory contains implementation-facing product and architecture contracts for AX Engine.
Public behavior remains documented under [`../docs`](../docs/) and must not be advertised until the
corresponding release gate has passed.

## Vertical inference improvement (engine × quant × serving)

- [Product requirements](prd/PRD-VERTICAL-INFERENCE-IMPROVEMENT.md)
- [ADR-022: High-value low-risk learning and moat closure](adr/ADR-022-VERTICAL-INFERENCE-HIGH-VALUE-LOW-RISK.md)
- [Technical specification](specs/TECH-SPEC-VERTICAL-INFERENCE-IMPROVEMENT.md)
- [Status ledger](VERTICAL-INFERENCE-IMPROVEMENT-STATUS.md)
- [Implementation plan (zh-TW)](planning/vertical-inference-improvement-plan-2026-08-11.md)
- [English working notes + market citations](reports/vertical-inference-improvement-plan-notes-2026-08-11.md)
- [Reference gap analysis PRD](prd/PRD-REFERENCE-INFERENCE-PERF-LEARNINGS.md)
- [Reference pins (2026-08-11)](reports/reference-pins-2026-08-11.md)

90-day program: close axquant → ax-engine → ax-serving identity/live-pin moat; adopt only
high-value low-risk inference learnings from updated `.internal/reference/*`. Does not break
ADR-002/003; product P0 is live-pin AX Engine (not a new dense kernel).

## Direct-inference model bug hunt package

- [Product requirements](prd/PRD-DIRECT-INFERENCE-MODEL-BUG-HUNT.md)
- [Status ledger](DIRECT-INFERENCE-MODEL-BUG-HUNT-STATUS.md)
- [Implementation plan](planning/direct-inference-model-bug-hunt-implementation-plan.md)
- [Multi-agent prompts (Grok + Codex max reasoning + ax-code GLM 5.2 1M)](planning/bug-hunt-agents/)

Serial, evidence-gated quality program over every **repo-owned direct MLX** family:
inspect for bugs, wrong implementation, wrong MTP mode design, bottlenecks, and dead
code; fix; re-scan; loop until family exit. **Grok CLI** orchestrates; **Codex CLI**
(maximum reasoning) owns deep correctness/MTP fixes; **ax-code CLI**
(`zai-coding-plan/glm-5.2[1m]`) owns wide 1M-context scans. One family at a time;
do not open the next family until the ledger records close/park.

## Agent-aware inference package

- [Product requirements](prd/PRD-AGENT-AWARE-INFERENCE-RUNTIME.md)
- [ADR-001: Agent-aware inference runtime](adr/ADR-001-AGENT-AWARE-INFERENCE-RUNTIME.md)
- [Agent Session Contract v1 technical specification](specs/TECH-SPEC-AGENT-SESSION-RUNTIME-CONTRACT.md)
- [Implementation status and cross-repository handoff](AGENTIC-INFERENCE-STATUS.md)

## Decode dispatch-efficiency package

- [Product requirements](prd/PRD-DECODE-DISPATCH-EFFICIENCY.md)
- [ADR-003: Optimize dispatch-bound decode regions only, evidence-gated](adr/ADR-003-DISPATCH-BOUND-DECODE-OPTIMIZATION.md)
- [Technical specification and phase plan](specs/TECH-SPEC-DECODE-DISPATCH-EFFICIENCY.md)
- [Implementation status ledger](DECODE-DISPATCH-EFFICIENCY-STATUS.md)

Dense decode is measured bandwidth-bound and closed to further single-op fusion; MoE routing,
per-layer compile stability, and decode-loop unification are the active levers. Read ADR-003
before proposing new fused routes or default flips — every promotion is evidence-gated with
greedy-parity as a hard blocker.

## Durable tiered prefix-cache package

- [Product requirements](prd/PRD-DURABLE-TIERED-PREFIX-CACHE.md)
- [ADR-002: Durable tiered prefix cache](adr/ADR-002-DURABLE-TIERED-PREFIX-CACHE.md)
- [Technical specification](specs/TECH-SPEC-DURABLE-TIERED-PREFIX-CACHE.md)
- [Implementation status ledger](DURABLE-PREFIX-CACHE-STATUS.md)

This package prioritizes exact local-disk prefix reuse and native physical KV handling while
retiring the TurboQuant runtime path. Disk remains a cold prefix tier; active attention state must
be hydrated into native memory before decode. Architecture is accepted; performance promotion is
not complete — read the status ledger before changing defaults or public claims.

## Big-model long-lived serving package

- [Product requirements](prd/PRD-BIG-MODEL-LONG-LIVED-SERVING.md)
- [ADR-017: Big-model long-lived serving](adr/ADR-017-BIG-MODEL-LONG-LIVED-SERVING.md)
- [Technical specification](specs/TECH-SPEC-BIG-MODEL-LONG-LIVED-SERVING.md)
- [Multiphase implementation plan](planning/big-model-long-lived-serving-implementation-plan.md)
- [Implementation status ledger](BIG-MODEL-LONG-LIVED-SERVING-STATUS.md)
- Strategy report: [kv-disk-strategy-and-ds4-plan-2026-08-08.zh-TW.md](reports/kv-disk-strategy-and-ds4-plan-2026-08-08.zh-TW.md)

This package programs **stability-first** large-model residency (72h endurance), operator
profiles, and evidence-gated peer parity (e.g. vs specialized V4 engines). It **consumes**
ADR-002 (prefix-only disk; no active SSD attention) and does not replace DTPC wire formats.
Public claims require the status ledger and phase exit artifacts — especially a full 72h pass
and, for peer wording, a locked head-to-head table.

## Session-wide paged-KV sharing package

- [Product requirements](prd/PRD-SESSION-WIDE-PAGED-KV-SHARING.md)
- [ADR-006: Session-wide paged-KV ownership](adr/ADR-006-SESSION-WIDE-PAGED-KV-OWNERSHIP.md)
- [ADR-007: Native block-table attention](adr/ADR-007-NATIVE-BLOCK-TABLE-ATTENTION.md)
- [ADR-008: Content-addressed durable KV pages](adr/ADR-008-CONTENT-ADDRESSED-DURABLE-KV-PAGES.md)
- [Technical specification](specs/TECH-SPEC-SESSION-WIDE-PAGED-KV-SHARING.md)
- [Native block-table attention specification](specs/TECH-SPEC-NATIVE-BLOCK-TABLE-ATTENTION.md)
- [Page-era durable KV specification](specs/TECH-SPEC-PAGE-ERA-DURABLE-KV.md)
- [Implementation status ledger](SESSION-WIDE-PAGED-KV-STATUS.md)

This package advances the existing FA-only private block scaffold to an experimental,
runner-local shared physical pool with refcounted prefix adoption and block-level copy-on-write.
It remains default off. Native block-table decode and content-addressed durable payload pages are
implemented behind additional default-off flags; real-model promotion, MLA/linear/rotating
layouts, continuous batching, and public memory/performance claims remain gated work.

## Nemotron 3 Embed package

- [Product requirements](prd/PRD-NEMOTRON-EMBED.md)
- [ADR-018: Nemotron Embed as EncoderEmbed family](adr/ADR-018-NEMOTRON-EMBED.md)
- [Technical specification](specs/TECH-SPEC-NEMOTRON-EMBED.md)
- [Implementation plan](planning/nemotron-embed-implementation-plan.md)

Compatible encoder-embed path for NVIDIA Nemotron 3 Embed (Ministral bidirectional +
mean pool). Distinct from chat/Omni `nemotron_h`. Default RAG embedder remains
Qwen3-Embedding; real-weight oracle, multi-model allowlist, and public benches are
follow-up.

## Qwen 3.6 linear MTP Tier 2 package

- [Product requirements](prd/PRD-QWEN36-LINEAR-MTP-TIER2.md)
- [ADR-020: Fail-closed default, exact scope independence, workload-scoped Tier 2](adr/ADR-020-QWEN36-LINEAR-MTP-TIER2.md)
- [Technical specification](specs/TECH-SPEC-QWEN36-LINEAR-MTP-TIER2.md)
- [Implementation plan](planning/qwen36-linear-mtp-tier2-implementation-plan.md)
- [Status ledger](QWEN36-LINEAR-MTP-TIER2-STATUS.md)

Closes **runtime** MTP acceleration (exactness + speed) for the Tier 1 certified
`AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP` v3 checkpoint on `df-macbookpro-m5`. Product default
stays direct-fallback until formal dual-profile gates pass and the ledger records a
stock-binary PASS; formal harness uses `AX_MLX_QWEN_LINEAR_MTP_CERTIFICATION_CANDIDATE=1`.

## Nemotron / Qwen3 GenRM package

- [Product requirements](prd/PRD-NEMOTRON-GENRM.md)
- [ADR-019: GenRM as workload on existing causal families](adr/ADR-019-NEMOTRON-GENRM.md)
- [Technical specification](specs/TECH-SPEC-NEMOTRON-GENRM.md)
- [Implementation plan](planning/nemotron-genrm-implementation-plan.md)
- Public operator guide: [`docs/GENRM.md`](../docs/GENRM.md)

Generative reward / principle judges reuse the base graph (MVP: Qwen3). Phase A
lands `principle` chat role + docs; Phase B is numeric Yes/No logprob scoring.
Ultra 550B LatentMoE GenRM is out of scope for AX Apple Silicon.

## Gemma 4 / Qwen 3 focused-deepening package

- [Product requirements](prd/PRD-GEMMA4-QWEN3-FOCUSED-DEEPENING.md)
- [ADR-010: Focused deepening over model-range expansion](adr/ADR-010-GEMMA4-QWEN3-FOCUSED-DEEPENING.md)
- [Technical specification](specs/TECH-SPEC-GEMMA4-QWEN3-FOCUSED-DEEPENING.md)
- [Multi-phase implementation plan](planning/gemma4-qwen3-focused-deepening-implementation-plan.md)
- [Implementation status ledger](GEMMA4-QWEN3-FOCUSED-DEEPENING-STATUS.md)

This package freezes model-range expansion and deepens the two primary family lines
(Gemma 4, Qwen 3): P0 correctness/trust (convert drop accounting, Gemma 4 loop
detection, re-specified prefill correctness), P1 gemma4_unified multimodal depth (video
under atomic budget contract, feature cache, prefix reuse, budget ladder), P2 same-family
VL variants (gemma4_vl, qwen3_vl), P3 batching/kernel ceilings with Decision A bit-exact
certification. All new media paths must meet the AX golden-fixture and fail-closed bar.
Media is data-URI only (no remote fetch, no bare filesystem paths).

**vs mlxcel flip schedule (Qwen 3 + Gemma 4 only):**  
- Plan: [`reports/qwen3-gemma4-mlxcel-flip-plan-2026-07-23.md`](reports/qwen3-gemma4-mlxcel-flip-plan-2026-07-23.md)
  (zh-TW: [`reports/2-3w-milestones-vs-mlxcel-2026-07-23.zh-TW.md`](reports/2-3w-milestones-vs-mlxcel-2026-07-23.zh-TW.md))
- Delivery ledger (W1 harness shipped):  
  [`reports/qwen3-gemma4-mlxcel-flip-delivery-2026-07-23.md`](reports/qwen3-gemma4-mlxcel-flip-delivery-2026-07-23.md)
- Harness: `scripts/bench_ax_multimodel_serving.py`,
  `scripts/check_ax_multimodel_serving_artifact.py`,
  `scripts/compare_qwen_gemma_flip.py`

### Gemma 4 / Qwen 3 focused-deepening work

1. Read `AGENTS.md`, ADR-010, the PRD, tech-spec, plan, and
   [`GEMMA4-QWEN3-FOCUSED-DEEPENING-STATUS.md`](GEMMA4-QWEN3-FOCUSED-DEEPENING-STATUS.md).
2. Prefer symbol anchors over volatile `runner/mod.rs` line numbers.
3. Update the status ledger with commit, tests, and residual hardware gates before handoff.

## CUDA vLLM delegated-provider package

- [Product requirements](prd/PRD-AX-VLLM-CUDA-DELEGATED-PROVIDER.md)
- [ADR-013: AX Engine owns the vLLM CUDA delegated provider](adr/ADR-013-AX-VLLM-CUDA-DELEGATED-PROVIDER.md)
- [Technical specification](specs/TECH-SPEC-AX-VLLM-CUDA-DELEGATED-PROVIDER.md)
- [Multi-phase implementation plan](planning/ax-vllm-cuda-delegated-provider-implementation-plan.md)
- [Current candidate status](evidence/ax-vllm-cuda-2026-07-23/STATUS.zh-TW.md)
- [Release review packet](evidence/ax-vllm-cuda-2026-07-23/CANDIDATE-REVIEW.zh-TW.md)

This package makes AX Engine the single owner of the vLLM provider used by CUDA x86_64 and
Thor/aarch64. The two architectures share one wire/provider contract and use separately certified
runtime profiles. AX OCR retains OCR workflow, quality, model-artifact, and release ownership; its
direct vLLM provider is removed only after dual-route parity, hardware, performance, security, and
rollback gates pass. vLLM is the general OCR/VLM compatibility lane, while TensorRT-LLM and
TensorRT Edge-LLM remain explicit optimized lanes for certified workloads. ADR-013 partially
supersedes the “vLLM reference-only” language in ADR-011 and ADR-012 for this scope.
The current control plane also partitions delegated JSON/SSE keep-alive pools by `Accept` and
uses `TCP_NODELAY`. vLLM, TensorRT-LLM, and TensorRT Edge-LLM share one bounded, fail-closed
OpenAI SSE framing helper, while provider-specific content DTOs, capabilities, identity, and
release evidence remain separate. These changes close code-level transport risks but do not
replace model, OCI, dual-Thor, or 24-hour release gates.

## Mandatory coder start sequence

### Agent-aware inference work

1. Read `/Users/akiralam/code/ax-engine/AGENTS.md` and the agent-aware package documents above.
2. Read the peer AX Serving package at:
   - `/Users/akiralam/code/ax-serving/.internal/prd/PRD-AGENT-AWARE-INFERENCE-FABRIC.md`
   - `/Users/akiralam/code/ax-serving/.internal/adr/ADR-015-AGENT-AWARE-INFERENCE-FABRIC.md`
   - `/Users/akiralam/code/ax-serving/.internal/specs/TECH-SPEC-AGENT-SESSION-FABRIC-CONTRACT.md`
   - `/Users/akiralam/code/ax-serving/.internal/AGENTIC-INFERENCE-STATUS.md`
3. Inspect the peer repository's actual code, current commit, and working tree. The peer status file
   is an index, not proof that code or tests exist.
4. Record both repository commits and relevant working-tree state in
   [`AGENTIC-INFERENCE-STATUS.md`](AGENTIC-INFERENCE-STATUS.md).
5. Implement only the AX Engine ownership defined by ADR-001. Do not edit AX Serving unless that
   repository is also explicitly assigned to the same coder.

Repeat steps 2 through 4 before each cross-repository contract milestone and before declaring the
work complete. If the two technical specifications disagree, stop contract-dependent work, update
both specifications or record the blocker in both status ledgers, and do not invent a private wire
variant.

### Durable prefix-cache work

1. Read `AGENTS.md`, ADR-002, the durable PRD, the durable tech-spec, and
   [`DURABLE-PREFIX-CACHE-STATUS.md`](DURABLE-PREFIX-CACHE-STATUS.md).
2. Read public operator docs in [`docs/KV-CACHE.md`](../docs/KV-CACHE.md) and inspect
   `crates/ax-engine-mlx/src/disk_prefix_cache.rs` plus `runner/prefix_cache.rs`.
3. Update the durable status ledger with commit, residual changes, and any gate results before
   handoff. Do not claim performance promotion without PRD §9.2 artifacts.

### Session-wide paged-KV work

1. Read ADR-006, the paged-KV PRD and technical specification, this package's status ledger,
   ADR-002, and `docs/KV-CACHE.md` before editing pool, prefix-cache, or runner ownership.
2. Preserve the runner-local/native versus cross-runner/serialized boundary. Do not place
   `MlxArray` state into `MlxPrefixCacheStore` or the disk writer.
3. Keep both paged-KV flags default off and retain serialized L1/L2 fallback until the promotion
   gates pass.
4. Update `SESSION-WIDE-PAGED-KV-STATUS.md` with exact tests and unavailable model/hardware gates
   before handoff. Unit correctness is not evidence for a memory or throughput claim.

### CUDA vLLM delegated-provider work

1. Read `AGENTS.md`, ADR-013, its PRD, technical specification, implementation plan, ADR-011, and
   ADR-012 before editing backend resolution, delegated HTTP, OpenAI multimodal routing, or CUDA
   runtime packaging.
2. Inspect both `/Users/akiralam/code/ax-engine` and `/Users/akiralam/code/ax-ocr`, including each
   HEAD and dirty state. The plan baseline is historical evidence, not permission to assume either
   tree is unchanged.
3. Preserve the three ownership layers: AX OCR product/quality, AX Engine API/provider, and the
   isolated `ax-engine-vllm-runtime` worker package. Do not move OCR release or quantization policy
   merely because it is under AX OCR's current `cuda/` directory.
4. Keep AX OCR's direct provider until the plan's parity, real-hardware, performance, security,
   soak, and rollback gates pass; then delete it rather than maintaining a permanent second path.
5. Record exact commands, commits, artifact digests, hardware/software matrix, and residual gates
   using the evidence template in the implementation plan. Unit/mock success is not CUDA or Thor
   release evidence.

## Document precedence

When documents conflict:

1. Accepted ADRs define ownership and irreversible architecture decisions.
2. The PRD defines outcomes and release gates.
3. The technical specification defines the implementation contract.
4. The status ledger records actual evidence and blockers.
5. Public documentation describes released behavior only.


## Thor / Edge-LLM

- ADR: [`adr/ADR-011-AX-TENSORRT-EDGE-LLM-THOR.md`](adr/ADR-011-AX-TENSORRT-EDGE-LLM-THOR.md)
- PRD: [`prd/PRD-AX-TENSORRT-EDGE-LLM-THOR.md`](prd/PRD-AX-TENSORRT-EDGE-LLM-THOR.md)
- Spec: [`specs/TECH-SPEC-AX-TENSORRT-EDGE-LLM-THOR.md`](specs/TECH-SPEC-AX-TENSORRT-EDGE-LLM-THOR.md)
- Plan: [`planning/ax-tensorrt-edge-llm-thor-implementation-plan.md`](planning/ax-tensorrt-edge-llm-thor-implementation-plan.md)
- Status: [`AX-TENSORRT-EDGE-LLM-THOR-STATUS.md`](AX-TENSORRT-EDGE-LLM-THOR-STATUS.md)
- Matrix: [`reports/thor-support-matrix-and-dual-runtime-2026-07-23.md`](reports/thor-support-matrix-and-dual-runtime-2026-07-23.md)
