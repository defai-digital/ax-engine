# Status ledger: Direct-Inference Model Bug Hunt

| Field | Value |
| --- | --- |
| PRD | [PRD-DIRECT-INFERENCE-MODEL-BUG-HUNT.md](prd/PRD-DIRECT-INFERENCE-MODEL-BUG-HUNT.md) |
| Plan | [direct-inference-model-bug-hunt-implementation-plan.md](planning/direct-inference-model-bug-hunt-implementation-plan.md) |
| Agents | [planning/bug-hunt-agents/](planning/bug-hunt-agents/) |
| Last updated | 2026-08-11 |
| Program status | **Complete — closed-code-only program exit** |
| Active family | _(none)_ |
| Orchestrator | Grok CLI |
| Deep reasoner | Codex CLI · `gpt-5.6-sol` · reasoning `max` |
| Wide scanner | ax-code CLI · `zai-coding-plan/glm-5.2[1m]` |

## Legend

| Status | Meaning |
| --- | --- |
| `pending` | Not started |
| `in_progress` | Active serial target |
| `inspect_done` | Dual-agent inspect complete; fixes remaining |
| `closed` | Exit gate passed with weights smoke |
| `closed-code-only` | Code+tests closed; weights smoke deferred |
| `parked` | Stopped with rationale; not claimed fixed |
| `n/a` | Not applicable to this family |

Finding disposition: `open` · `fixed` · `wontfix` · `limit` · `duplicate`.

---

## Program controls

| Control | Value |
| --- | --- |
| Serial rule | Only one `in_progress` family at a time |
| Open next family when | Active family is `closed` / `closed-code-only` / `parked` |
| Report root | `.internal/reports/bug-hunt/` |
| CODEX_MODEL | `gpt-5.6-sol` |
| CODEX_REASONING_EFFORT | `max` |
| AX_CODE_MODEL | `zai-coding-plan/glm-5.2[1m]` |

### CLI version log (update when tools change)

| Date | Tool | Version / note |
| --- | --- | --- |
| 2026-08-11 | grok | local `~/.grok/bin/grok` |
| 2026-08-11 | codex | codex-cli 0.147.0 · model `gpt-5.6-sol` · effort `max` |
| 2026-08-11 | ax-code | local `/opt/homebrew/bin/ax-code` · model `zai-coding-plan/glm-5.2[1m]` |

---

## Wave 0 — Shared substrate

| ID | Surface | Status | Last report | Residual |
| --- | --- | --- | --- | --- |
| W0-registry | architecture_registry + support_tier + convert map | closed | `20260811/W0-shared/` | DI-W0-001/002 fixed |
| W0-runner-mtp | runner MTP policy + mtp.rs + adaptive gate | closed | `20260811/W0-shared/` | fail-closed verified dual-agent |
| W0-convert-manifest | generate-manifest / drop accounting | closed | `20260811/W0-shared/` | gemma4_unified family honesty fixed |
| W0-tooling | smoke/probe/QA harness readiness | closed | `20260811/W0-shared/` | DI-W0-004/005 fixed |

---

## Wave 1 — Primary chat / agent

| Order | Family ID | Manifest family | Status | P0 open | P1 open | Report dir | Exit notes |
| ---: | --- | --- | --- | ---: | ---: | --- | --- |
| 1 | `qwen36-27b` | `qwen3_5` | closed-code-only | 0 | 0 | `20260811/qwen36-27b/` | ADR-020 LIMIT; weights deferred |
| 2 | `qwen36-35b-a3b` | `qwen3_5` MoE | closed-code-only | 0 | 0 | `20260811/qwen36-35b-a3b/` | MoE MTP formal deferred |
| 3 | `qwen35-9b` | `qwen3_5` | closed-code-only | 0 | 0 | `20260811/qwen35-9b/` | 16 GB LIMIT |
| 4 | `qwen3-coder-next` | `qwen3_next` | closed-code-only | 0 | 0 | `20260811/qwen3-coder-next/` | no download-mtp head |
| 5 | `qwen3-dense` | `qwen3` | closed-code-only | 0 | 0 | `20260811/qwen3-dense/` | batched decode cert surface |
| 6 | `gemma4-12b-unified` | `gemma4_unified` | closed-code-only | 0 | 0 | `20260811/gemma4-12b-unified/` | DI-W0-002 fixed |
| 7 | `gemma4-e-series-26-31` | `gemma4` / `gemma4_vl` | closed-code-only | 0 | 0 | `20260811/gemma4-e-series-26-31/` | VL tower fail-closed |
| 8 | `glm47-flash` | `glm4_moe_lite` | closed-code-only | 0 | 0 | `20260811/glm47-flash/` | native MLX default |

---

## Wave 2 — Multimodal / OCR / speech / embed

| Order | Family ID | Manifest family | Status | P0 open | P1 open | Report dir | Exit notes |
| ---: | --- | --- | --- | ---: | ---: | --- | --- |
| 9 | `qwen3-vl` | `qwen3_vl` / `qwen3_vl_moe` | closed-code-only | 0 | 0 | `20260811/qwen3-vl/` | media fail-closed |
| 10 | `minicpmv4_6` | `minicpmv4_6` | closed-code-only | 0 | 0 | `20260811/minicpmv4_6/` | — |
| 11 | `nemotron-omni` | `nemotron_h` | closed-code-only | 0 | 0 | `20260811/nemotron-omni/` | no video |
| 12 | `unlimited-ocr` | `unlimited_ocr` | closed-code-only | 0 | 0 | `20260811/unlimited-ocr/` | — |
| 13 | `whisper-large-v3-turbo` | whisper audio | closed-code-only | 0 | 0 | `20260811/whisper-large-v3-turbo/` | DI-W0-001 fixed |
| 14 | `embeddings-primary` | embeddinggemma / qwen3-embed | closed-code-only | 0 | 0 | `20260811/embeddings-primary/` | — |
| 15 | `nemotron-embed` | `nemotron_embed` | closed-code-only | 0 | 0 | `20260811/nemotron-embed/` | Compatible only |

---

## Wave 3 — Secondary preview

| Order | Family ID | Manifest family | Status | P0 open | P1 open | Report dir | Exit notes |
| ---: | --- | --- | --- | ---: | ---: | --- | --- |
| 16 | `llama3` | `llama3` | closed-code-only | 0 | 0 | `20260811/llama3/` | preview |
| 17 | `llama4-scout` | `llama4` | closed-code-only | 0 | 0 | `20260811/llama4-scout/` | preview |
| 18 | `mistral-family` | `mistral3` | closed-code-only | 0 | 0 | `20260811/mistral-family/` | preview |
| 19 | `gpt-oss` | `gpt_oss` | closed-code-only | 0 | 0 | `20260811/gpt-oss/` | MXFP4 residency |
| 20 | `deepseek-v3` | `deepseek_v3` / `v32` | closed-code-only | 0 | 0 | `20260811/deepseek-v3/` | dense MLA contract |

---

## Wave 4 — Experimental

| Order | Family ID | Manifest family | Status | P0 open | P1 open | Report dir | Exit notes |
| ---: | --- | --- | --- | ---: | ---: | --- | --- |
| 21 | `diffusion-gemma` | `diffusion_gemma` | closed-code-only | 0 | 0 | `20260811/diffusion-gemma/` | honest Experimental LIMIT |
| 22 | `deepseek-v4` | `deepseek_v4` | closed-code-only | 0 | 0 | `20260811/deepseek-v4/` | no support claim inflation |

---

## Active finding log (append-only)

| Date | Family | Finding ID | Class | Sev | Title | Disposition | Fix commit | Agents |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-11 | W0-shared | DI-W0-001 | IMPL | P1 | Whisper registry orphan | fixed | (this branch) | Codex+ax-code |
| 2026-08-11 | W0-shared | DI-W0-002 | IMPL | P1 | gemma4_unified Certified inheritance | fixed | (this branch) | Codex |
| 2026-08-11 | W0-shared | DI-W0-003 | DOC | P2 | parity test handwritten list | fixed | (this branch) | both |
| 2026-08-11 | W0-shared | DI-W0-004 | BUG | P1 | smoke ensure_binary stale | fixed | (this branch) | Codex |
| 2026-08-11 | W0-shared | DI-W0-005 | BUG | P1 | models-dir HF cache fallthrough | fixed | (this branch) | Codex |
| 2026-08-11 | W0-shared | DI-W0-006 | DOC | P3 | tier test Compatible wildcard | parked | — | Codex |
| 2026-08-11 | W0-shared | DI-W0-007 | IMPL | P2 | registry uniqueness | fixed | (this branch) | Codex |
| 2026-08-11 | gemma4-12b-unified | DI-W1-001 | IMPL | P1 | gemma4_unified missing GeGLU/RoPE gates | fixed | (this branch) | Codex |
| 2026-08-11 | qwen3-vl | DI-W2-001 | IMPL | P1 | qwen3_vl_moe empty moe_config | fixed | (this branch) | Codex |
| 2026-08-11 | embeddings-primary | DI-W2-002 | IMPL | P1 | EmbeddingGemma singleton vs batch path | fixed | (this branch) | Codex |
| 2026-08-11 | nemotron-omni | DI-W2-F1a | BUG | P1 | untrusted NCHW pixel buffer OOB | fixed | (this branch) | ax-code |
| 2026-08-11 | minicpmv4_6 | DI-W2-F1b | BUG | P1 | untrusted NHWC pixel buffer OOB | fixed | (this branch) | ax-code |
| 2026-08-11 | qwen3-vl | DI-W2-F1c | BUG | P1 | untrusted patch buffer OOB | fixed | (this branch) | ax-code |

### Finding ID format

`DI-<family-id>-<NNN>` e.g. `DI-qwen36-27b-001`.

---

## Session handoff template

```text
Active family: (none — program complete)
Phase: F6 program exit
Open P0: 0  Open P1: 0 (parked DI-W0-006 only)
Last Codex report: .internal/reports/bug-hunt/20260811/W0-shared/codex-inspect.md
Last ax-code report: .internal/reports/bug-hunt/20260811/W0-shared/axcode-scan.md
Next action: none — closed-code-only program exit
Blocked on: none
```

---

## Change log

| Date | Change |
| --- | --- |
| 2026-08-11 | Ledger initialized with PRD; no family started |
| 2026-08-11 | Bound Codex `gpt-5.6-sol`/max + ax-code `glm-5.2[1m]` |
| 2026-08-11 | Wave 0 dual-agent inspect; fixed DI-W0-001..005,007; parked DI-W0-006 |
| 2026-08-11 | Waves 1–4 closed-code-only (static dual-agent package; weights unavailable) |
| 2026-08-11 | Program status complete |
