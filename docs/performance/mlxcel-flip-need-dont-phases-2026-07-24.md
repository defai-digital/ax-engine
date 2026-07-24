# mlxcel deep-review → need / don't-need phases (2026-07-24)

Source: [`.internal/reports/mlxcel-deep-review-2026-07-24.zh-TW.md`](../../.internal/reports/mlxcel-deep-review-2026-07-24.zh-TW.md).
Locked gates: [`benchmarks/manifests/qwen_gemma_flip_gates.v1.json`](../../benchmarks/manifests/qwen_gemma_flip_gates.v1.json) (thresholds **not** relaxed).
Primary host: `AKMBPM5MAXx` / Apple M5 Max (`applegpu_g17s`).

## Verdict

- mlxcel's measured win on the 2026-07-23 S0–S3 campaign is real; it is **not** "better architecture."
- AX keeps core/sdk/server layering, exactness defaults, and single-process multi-model.
- Wins come from **shim thickness**, **decode/stream path**, and **prefill isolation policy** — all inside existing crates.

## IN SCOPE (implement)

| Phase | Report | Lever | Where |
| --- | --- | --- | --- |
| **B / P0** | Compiled composite hot chains | Split dense SwiGLU FFN compile on Qwen decode; keep packed-path compile; prefer fewer host graph encodes | `crates/ax-engine-mlx` mlp + `mlx-sys` compile |
| **B / P1** | Decode double-buffer + serving path | First-token single-forward TTFT bootstrap then double-buffer catch-up; stream worker burst; SSE backlog decoupling; greedy OpenAI `repetition_penalty=1.0` so direct argmax path engages | runner, generation service, openai requests, session stream progress |
| **C / P3** | TTFT split | Shape-sensitive short-prompt warmup (32/34/64) + direct-pipeline prime; stronger server warmup | runner, app_state/main |
| **D / P2** | Wall-time prefill quantum | Sibling-active multi-prefill quantum sized so one turn's wall stays under the **50 ms** stream-gap SLO (not 1 token) | generation service adaptive prefill isolation |
| **E / P3–P4** | S3 arbiter / batch formation | Profile arbiter hold + row-exact cohort engagement; optional server-mode batched-decode drift decision only if still short | arbiter + batched decode product note |

## OUT OF SCOPE (explicit don't)

- **Retired KV compression path** (mlxcel self-measured slower; AX already removed it under ADR-002).
- **Paged pool default-on** on M5 without new positive evidence (M5 pool was neutral/slower).
- **Monolith restructure** / llama-server flag surface for speed (not a speed lever).
- **Model-breadth chase** (VLM/OCR/audio/TP-PP product surface).
- **core/sdk/server redesign** or abandoning single-process multi-model vs mlxcel multi-process contract.
- **Relaxing** locked thr/TTFT/gap gate thresholds for a paper win.

## Phase exit criteria (locked gates)

Per scenario (median, ≥3 fresh-process reps): thr ≥ 1.15×, p95 TTFT ≤ 0.90×, p95 stream-gap ≤ 0.90× and ≤ 50 ms absolute, zero AX request / 503 / lifecycle errors.

## Implementation status on this branch

1. **Greedy OpenAI default `repetition_penalty`** → `1.0` so MLX direct pipeline engages (was 1.1 for Qwen/Gemma, killing ~25% decode).
2. **Stream path**: larger SSE/event backlog, single-stream engine step burst, lightweight stream progress reports, skip intermediate route maps.
3. **TTFT**: multi-shape prefill warmup + direct pipeline prime; first generated token is single-forward then double-buffer catch-up.
4. **Split dense FFN compile** for Qwen decode (gate/up/down SwiGLU under `AX_MLX_DENSE_FFN_COMPILE`).
5. **Sibling prefill quantum** raised from **1 → 16** tokens under adaptive isolation (M5 dual-model S1: 64-token quanta blew interactive gap to ~810 ms under arbiter hold; 16 tokens cuts turn count 16× vs 1-token without busting the 50 ms SLO envelope). Override with `AX_SERVER_ADAPTIVE_PREFILL_LATENCY_TOKENS`.
6. Flip target keeps `AX_MLX_PACK_LINEAR_ATTENTION_PROJECTIONS=0` where packing regressed the certified S0 path.
7. Linear-attention n-gram no longer permanently disables after short no-draft streaks (cooldowns only).

## Honest residual (2026-07-24 M5 measurements, post matvec+adaptive)

| Metric | Value |
| --- | ---: |
| Pure decode-trace | ~110.7–111.3 tok/s (matvec gate/up+down) |
| AX S0 OpenAI e2e (fresh) | ~108.1 tok/s |
| mlxcel S0 e2e (fresh) | ~94.8 tok/s |
| S0 thr ratio (median) | **~1.141× < 1.15 gate** |
| S0 TTFT ratio | ~0.75× **PASS** |
| S0 gap ratio | ~0.87× **PASS** (≤50 ms abs) |

S0 **thr ≥ 1.15× is still blocked by the pure GPU/BW ceiling** (~111 tok/s pure → ~1.14× e2e). TG-cached multi-row matvec and heavier warmup both regressed and were rejected. S1 gap passes under wall-time adaptive quantum; S1 thr and S2/S3 remain open.
