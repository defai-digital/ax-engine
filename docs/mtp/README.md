# MTP Docs

This is the MTP-specific documentation hub. Use it when the question is about
`ax-engine download-mtp`, MTP benchmark lanes, sidecar or assistant package
validation, or MTP tuning reports.

## Read This First

- Use the 6-bit `download-mtp` packages for practical AX Engine guidance.
- Keep 4-bit rows clearly labeled as comparison evidence for peer MTP engines
  that publish 4-bit results.
- Publish MTP rows in MTP mode only. Do not promote `mtp-ngram` rows in the
  current MTP matrix.
- Current Qwen linear-attention publication rows must explicitly record the
  validated exact-verifier profile (`--ax-qwen-linear-mtp-exact`). The profile
  is an arithmetic/checkpoint contract, not optimistic acceptance.
- Production runners automatically select that contract per loaded Qwen3.5 /
  Qwen3.6 artifact when its MTP depth and dense or affine 4/6/8-bit tensor
  layout are certified. This includes mixed 4/8-bit AXQ packages with group
  sizes 32/64. The selection is runner-scoped, so one resident model cannot
  change another model's arithmetic. Set
  `AX_MLX_QWEN_LINEAR_MTP_EXACT=0` as a kill switch; an ineligible or
  explicitly disabled Qwen linear-MTP package safely uses direct decode
  instead of the slow singleton-replay MTP path.
- Certified depth-one Qwen exact profiles, including the current
  `AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP` package, default the draft confidence gate
  to `0`. (A separate AXQ-4bit sibling for this base is not published: protection
  floors made it redundant with the 6bit pack.) A depth-one sidecar has no
  low-confidence tail to prune, while the generic `0.90` gate still pays for a
  full-vocabulary draft softmax and can discard the only proposal. Explicit
  `AX_MLX_MTP_DRAFT_MIN_CONFIDENCE`, speculation profiles, and the adaptive
  controller retain precedence.
- The policy mechanism covers every model-based drafter, but the numeric
  optimization does not. Qwen multi-depth and GLM heads retain the calibrated
  global/profile/adaptive policy; Gemma 4 assistant MTP retains its independent
  first/deep gates. An ineligible Qwen linear head or an invalid artifact with
  conflicting drafters fails closed to direct decode.
- Direct rows are allowed only as same-package denominators for AX MTP
  acceleration charts, not as cross-model speed evidence.

## Runtime Model Policy

The runner resolves one immutable MTP policy from validated, loaded components.
It does not trust model names or an artifact-provided confidence threshold.
Operator overrides keep the documented precedence, but model-owned defaults can
only come from a runtime-certified policy:

| Route code | Loaded drafter | Confidence policy | Route |
| ---: | --- | --- | --- |
| 0 | None | None | Direct |
| 1 | Qwen dense/recurrent | Calibrated Qwen resolver | MTP |
| 2 | Qwen linear, exact, depth 1 | Model default `0` | MTP |
| 3 | Qwen linear, exact, depth 2-3 | Calibrated Qwen resolver | MTP |
| 4 | Qwen linear without exact capability | None | Direct fallback |
| 5 | GLM | Calibrated GLM resolver; no Qwen override | MTP |
| 6 | Gemma 4 assistant | Independent first/deep calibrated gates | Assistant MTP |
| 7 | Conflicting attached drafters | None | Direct fallback |

Route telemetry exposes this decision through
`ax_mlx_mtp_model_policy`, `ax_mlx_mtp_model_policy_depth`,
`ax_mlx_mtp_model_policy_route_safe`, and
`ax_mlx_mtp_model_policy_active`. A certified model default is separately
reported by `ax_mlx_mtp_model_gate_default_present` and
`ax_mlx_mtp_model_gate_default_x1000`. The existing
`ax_mlx_qwen_linear_mtp_depth_one_gate_zero_model_default` key remains for
backward-compatible Qwen benchmark checks.

Experimental, opt-in knobs (default off, no product surface; measured A/B
evidence lives under
`benchmarks/results/inference/mlx-inference/2026-10-08-m5max-27b-mtp-depth-flags-ab/`):
`AX_MLX_MTP_DEVICE_GREEDY_ACCEPT` moves greedy acceptance on-device, and
`AX_MLX_MTP_COST_MODEL_DEPTH` replaces streak-based draft-depth control with
a cost model that can park speculation for the rest of a request when every
depth loses to direct. Their counters (`ax_mtp_cost_depth_current`,
`ax_mtp_cost_park_events`, per-depth `ax_mtp_cost_t_depth{d}_us` /
`ax_mtp_cost_p_depth{d}_x1000`) appear in route decisions only when enabled.
Flash Next requests that hit a transient think-window or pending-token block
now pause the draft cursor and resume instead of dropping it; the legacy
drop-on-block behavior remains available as
`AX_MLX_FLASH_NEXT_STICKY_FALLBACK=1`, and the lifecycle is visible through
`ax_mlx_flash_next_mtp_cursor_paused_steps` /
`ax_mlx_flash_next_mtp_cursor_resumed` route keys.

Qwen MTP drafts read the vocabulary projection once per draft step. When the
target head is quantized at 6 bits or more (the 8-bit head of the Qwen 3.8
27B MXFP4 pack), AX builds a draft-only 3-bit copy at load
(`AX_MLX_MTP_QUANTIZED_HEAD_DRAFT=0` disables it; `AX_MLX_MTP_DENSE_HEAD_DRAFT_BITS`
sets the width; `AX_MLX_MTP_DRAFT_LM_HEAD_BITS` overrides both). Verification
still uses the target head, so committed tokens are decided as before. On an
M5 Max 128 GB with the 27B MXFP4 pack (17 real prompts, greedy, depth 3) MTP
decode rose from 70.3 to 77.2 tok/s with all 17 outputs token-identical to
MTP with the full-width head, not to direct decode (see the next paragraph);
the
4-bit Qwen 3.6 35B-A3B head is left unchanged (3 bits measured +0.3% there).

Default Qwen linear MTP is not token-identical to direct decode. Since
d9a45f77 (2026-08-26) an explicitly requested Qwen linear MTP session uses the
throughput profile (`AX_MLX_QWEN_LINEAR_THROUGHPUT_MTP`, default on): depth 3,
projected-replay rollback and a relaxed target verifier built from stock MLX
arithmetic. A multi-row verify forward does not reproduce single-row decode
arithmetic, so near-tie tokens can flip. On an M5 Max with 17 real prompts,
greedy, 256 tokens: 27B MXFP4 MTP output differed from direct decode on 5
prompts, and 35B-A3B AXQ-4bit on 9. Forcing the exact profile on the MXFP4
27B (`AX_MLX_QWEN_LINEAR_THROUGHPUT_MTP=0`,
`AX_MLX_QWEN_LINEAR_MTP_CERTIFICATION_CANDIDATE=1`,
`AX_MLX_QWEN_LINEAR_MTP_EXACT=1`) is slower (47 vs 61 tok/s on the streaming
chat route) and still differed on 7 prompts: MXFP4 has no certified
row-exact verifier. n-gram verification matched direct decode on all 17.
Treat throughput MTP as an approximate mode, not an exact path: greedy output
can differ from direct decode. MTP stays explicit opt-in, and no flag here is
named or documented as exact until a row-exact MXFP4 verifier is certified.
When token identity with direct decode matters, use direct decode or n-gram
verification.

Sampled requests with a repetition, presence or frequency penalty or a
`logit_bias` keep Qwen linear MTP (`AX_MLX_MTP_PENALIZED_VERIFY`, default on).
Drafts stay unpenalised; every verify row is sampled on the host from the
target logits processed with the committed history plus the drafts before
it, which is the law direct decode samples from, and the rejection residual
and bonus token use the same rows. On an M5 Max with the 27B MXFP4 pack (17
real prompts, 256 tokens) Qwen's sampled settings with presence penalty 1.5
decode at 47.7 tok/s against 29.3 on the direct route. Greedy penalised
requests and no-repeat n-gram bans still decode directly.

MXFP4 verify windows of 3 to 8 rows use a fused matmul
(`AX_MLX_MTP_VERIFY_QMM_MXFP4`, default on; `=0` restores MLX's own quantized
matmul, `AX_MLX_MTP_VERIFY_QMM_MXFP4_MIN_N` sets an output-width floor). MLX's
quantized matmul pays 2.4 to 5.6 ms per extra row on the 27B FFN chain and
switches kernels at 13 rows; the fused kernel reads and decodes each weight
once and applies it to every row. On an M5 Max the 64-layer FFN chain at eight
rows drops from 46.2 to 29.6 ms and at four rows from 24.7 to 18.8 ms. With the
27B MXFP4 pack, depth-3 MTP, greedy, 256 tokens and 17 real prompts it decodes
1.138x faster than the stock route (median 1.151x, every prompt between 1.046x
and 1.206x; a repeat of the stock arm agreed within 0.2%), 2.32x direct decode
against 2.04x. The sum order differs from MLX's, so outputs change: 11 of 17
prompts match the stock MTP route, and against direct decode 12 of 17 match
with the fused route and 11 of 17 with the stock one, so it is no further from
direct decode than the route it replaces. It applies to dense MXFP4 weights
during a throughput-profile verify forward only; the exact profile, Flash Next
and MoE experts are unchanged.

DFlash2 block drafting is on by default for greedy text requests on the Qwen 3.8
27B target (the pack's recorded source model must be `Qwen/Qwen3.8-27B`: Qwen 3.6
27B has the same shapes but other weights) whenever the Apache-2.0 `z-lab/Qwen3.8-27B-DFlash2`
checkpoint (`config.json` and `model.safetensors`, about 3.6 GB in bf16, held as
MXFP4 in memory) can be found: `AX_MLX_DFLASH2_DIR`, else a `dflash2/` directory
inside the model directory, else the Hugging Face cache (`HF_HUB_CACHE`,
`HF_HOME/hub`, `~/.cache/huggingface/hub`). The checkpoint is not part of the
model pack and is never downloaded by the server; without it nothing changes.
`AX_MLX_DFLASH2=0`, `--mlx-mtp-policy disabled`, `--disable-ngram-acceleration`
and `AX_NO_SPEC` keep it off; `AX_MLX_DFLASH2_BF16=1` keeps the weights in bf16.
A five-layer drafter conditions on the target's hidden states at layers 5, 19, 33,
47 and 61 and proposes seven tokens per round; the target verifies the first few
of them in one forward with the same fused multi-row matmul and the same rollback
as MTP. The verify window (3 to 8 rows) follows the request: drafts that keep
being accepted earn the full window and prose settles at four rows
(`AX_MLX_DFLASH2_WINDOW=N` pins it). On an M5 Max with the 27B MXFP4 pack (17 real
prompts, 256 tokens, greedy) it decodes 1.28x faster than fused depth-3 MTP on the
11 code prompts (1.08x to 1.40x, up to 127 tok/s) and 1.02x on the 6 prose and chat
prompts (0.99x to 1.07x), 90.8 against 79.0 tok/s over the suite. Outputs are
approximate like MTP's: 11 of 17 prompts match direct decode, against 12 of 17 for
MTP. The drafter needs the hidden states of the whole prompt, so a request whose
prefix is restored from the prefix cache decodes on the normal routes; only a
request that prefills its prompt itself uses it. Sampled requests, requests with
logits processors or a grammar, image input and other models use the normal routes.

`AX_MLX_MTP_STEP_TRACE=<path>` is a capture-only diagnostic that appends one
JSON line per verified MTP cycle (verify window, accepted prefix, context
length, draft log-probs when the draft path computed them, timestamp). It adds
no device synchronization. `scripts/simulate_mtp_verify_prefix.py` replays
such traces to price verify-window truncation policies (confidence threshold,
calibrated cost model, oracle) against measured fixed-depth cycle costs before
any runtime change. Pass a fixed-depth trace per verify width with
`--cost-trace`, and a calibration trace captured with a non-zero
`AX_MLX_MTP_DRAFT_MIN_CONFIDENCE` of at least `0.001` (smaller values round to
a zero gate and keep the lazy draft path, which records no log-probs). On an
M5 Max 128 GB with Qwen 3.8 27B MXFP4 and Qwen 3.6 35B-A3B 4-bit (depth 3,
greedy, 17 real prompts), even oracle truncation was worth at most +4.3% and
+3.3% respectively, the synchronous gated draft path that truncation needs
cost about 5% and 8% decode by itself, and real threshold truncation at gates
0.70 / 0.80 lost 14-27%. Full-window verification with a zero gate remains
the default, and simulated gains should be read as upper bounds.

## Where To Go

| Need | Read |
| --- | --- |
| Download or prepare an MTP package | [Supported Models: MTP Downloads](../SUPPORTED-MODELS.md#mtp-downloads), [CLI](../CLI.md#ax-engine) |
| Read headline MTP result tables | [Performance Results: MTP](../PERFORMANCE-RESULTS.md#session-mode-mtp-generation), [Qwen3.6 MTP peer benchmark](qwen36-peer-comparison.md), [Performance: MTP Mode](../PERFORMANCE.md#mtp-mode) |
| Reproduce or review MTP benchmarks | [Benchmarks: MTP Matrix](../BENCHMARKS.md#mtp-matrix), [Benchmark Design](../BENCH-DESIGN.md) |
| Tune the MTP draft confidence gate | [MTP draft gate throughput](draft-gate-throughput.md) |
| Diagnose Tiel prefill latency and review M5 controls | [Tiel MTP prefill timing](tiel-prefill-diagnostics.md) |
| Review Gemma assistant-MTP depth work | [Gemma 4 assistant MTP multi-depth drafting](gemma4-assistant-multi-depth.md) |
| Review Qwen3.6 peer-engine MTP results | [Qwen3.6 MTP peer benchmark](qwen36-peer-comparison.md) |
| Review the 2026-09-21 7.5.3 Qwen / Gemma M5 Max peer campaign | [Qwen / Gemma M5 Max peer campaign](qwen-gemma-peer-m5-2026-09-21.md) |
| Review archived Qwen3.6 AX-only multi-suite MTP results | [Qwen3.6 AX-only multi-suite MTP results](qwen36-matrix-refresh.md) |
| AX Engine native MTP vs Youssofal MTPLX bundle | [AX MTP vs Youssofal MTPLX-Optimized](ax-mtp-vs-youssofal.md) |
| Review tree-draft investigation history | [Tree draft phase A](tree-draft-phase-a.md) |

## Publication Lanes

### Recommended 6-bit Lane

This is the practical AX Engine lane. Prepare packages with:

```text
ax-engine download-mtp qwen3.6-27b-6bit
ax-engine download-mtp qwen3.6-35b-a3b
ax-engine download-mtp gemma-4-12b
ax-engine download-mtp gemma-4-26b
ax-engine download-mtp gemma-4-31b
```

Artifacts should live under `benchmarks/results/speculative/mtp-6bit/` and record the exact
prepared model path, model snapshot, sidecar or assistant provenance, route
identity, sampler, prompt suite, repetitions, cooldown, prefill, decode, TTFT,
and MTP accept rate. Qwen rows must also record
`ax_qwen_linear_mtp_exact=true` and
`ax_qwen_linear_mtp_exact_explicit_enable=true`; the publication runners reject
ambient or missing profile selection.

### 4-bit Comparison Lane

This lane exists to align with peer MTP-engine benchmark publications that use
4-bit models. It is not the recommended AX Engine deployment setting. Keep
artifacts in clearly labeled comparison directories and keep
[Performance Results](../PERFORMANCE-RESULTS.md) / [Performance](../PERFORMANCE.md)
explicit that 6-bit remains the recommended practical lane.

### Out Of Scope

- `mtp-ngram` rows in current MTP publication
- Qwen3-Coder-Next, 5-bit, 8-bit, FFN-only, or GGUF rows in the recommended
  6-bit matrix
- direct rows used as a cross-model speed leaderboard
