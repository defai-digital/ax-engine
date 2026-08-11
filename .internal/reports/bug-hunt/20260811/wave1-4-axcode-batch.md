# ax-code batch scan Waves 1-4

**Scope:** Per-family static scan of all 21 AX Engine model-family forward graphs
in `crates/ax-engine-mlx/src/` (Wave 0 covered the shared MTP / architecture-registry
surface in `axcode-scan.md`).
**Model:** zai-coding-plan/glm-5.2
**Date:** 2026-08-11
**Verdict:** **18/21 families clean (no open P0/P1).** One P1 defect class —
untrusted-shape `from_raw_data` without a buffer-length guard — recurs in 3 VLM
media paths. MTP fail-closed re-confirmed for qwen linear and deepseek v4.

---

## Per-family

- **qwen3** — `model/families/standard.rs` (dense) + `model/families/qwen3_linear.rs` (linear) — no open P0/P1 (mtp fail-closed: qwen linear CONFIRMED, `mtp_model_policy.rs:189` `route_safe()==false` w/o env opt-in).
- **qwen3_5** — `model/families/standard.rs` + `qwen3_linear.rs` (hybrid gated-delta) — no open P0/P1 (mtp fail-closed: qwen linear CONFIRMED).
- **qwen3_next** — `model/families/standard.rs` + `qwen3_linear.rs` (MoE hybrid) — no open P0/P1 (mtp fail-closed: qwen linear CONFIRMED).
- **gemma4** — `model/families/standard.rs` — no open P0/P1.
- **gemma4_unified** — `gemma4_unified.rs` — no open P0/P1 (media spans saturating/`checked_*`; returns `Err` on length mismatch).
- **gemma4_vl** — `gemma4_vl.rs` — no open P0/P1 (buffer-vs-shape validated at `:732`/`:584` — the positive control for the VLM defect class).
- **glm4_moe_lite** — `model/families/glm4_moe_lite.rs` (+ `model/shared/mla.rs`) — no open P0/P1.
- **qwen3_vl** — `qwen3_vl.rs` — **P1 `build_vl_prefill_embeddings` @ `qwen3_vl.rs:235`** (untrusted-shape OOB; see F1).
- **minicpmv4_6** — `minicpm_v.rs` — **P1 `build_vl_prefill_embeddings` @ `minicpm_v.rs:547`** (untrusted-shape OOB; see F1).
- **nemotron_h** — `model/families/nemotron_h.rs` (+ `nemotron_omni.rs` media) — **P1 `build_omni_prefill_embeddings` @ `nemotron_omni.rs:491`** (untrusted-shape OOB on the omni image path; see F1). Nemotron-H core mixers (Mamba-2 / attention / MoE) clean.
- **unlimited_ocr** — `unlimited_ocr.rs` — no open P0/P1 (buffers guarded at `:741`/`:686`; protected-prefix R-SWA fail-closed).
- **whisper** — `whisper.rs` (+ `whisper_mel.rs`, `whisper_tokenizer.rs`) — no open P0/P1 (mel window sized by construction; seek loop clamped).
- **embeddinggemma** — embed forward in `model/mod.rs` (`forward_for_embedding_gemma3_batch`) — no open P0/P1 (embed gather clamped `model/mod.rs:413`; mean-pool bounded by `max_len`).
- **nemotron_embed** — embed forward in `model/mod.rs` (`forward_for_embedding_batch`) — no open P0/P1 (same clamped gather + bounded pool).
- **llama3** — `model/families/standard.rs` — no open P0/P1.
- **llama4** — `model/families/llama4.rs` — no open P0/P1 (iRoPE divisor-zero short-circuit; top-1 routing numerically correct).
- **mistral3** — `model/families/mistral3.rs` — no open P0/P1 (delegates to `standard`).
- **gpt_oss** — `model/families/gpt_oss.rs` — no open P0/P1 (SwitchGLU reshape avoids documented double-top_k panic).
- **deepseek_v3** — `model/families/deepseek_v3.rs` (+ `model/shared/mla.rs`) — no open P0/P1.
- **diffusion_gemma** — `diffusion.rs` — no open P0/P1 (embed clamped; temperature NaN-guards; commit-skip preserves KV).
- **deepseek_v4** — `model/families/deepseek_v4.rs` (+ `model/shared/deepseek_v4_*.rs`, `hyper_connection.rs`) — no open P0/P1 (mtp fail-closed: deepseek v4 CONFIRMED, `mtp_model_policy.rs:189` `route_safe()==false` w/o env opt-in; runtime registry-gated out; `model/mod.rs:260` panic unreachable-by-construction). `moe_router_deepseek_v4` tid2eid gather (`mlp.rs:3879`) lacks the embed id clamp — INFO only, unreachable while V4 runtime is gated out.

---

## Findings

### F1 — Untrusted-shape `from_raw_data` → GPU OOB read (P1, 3 sites, same defect class)

Each site wraps an externally-supplied buffer in `MlxArray::from_raw_data(ptr,
byte_size_of_buffer, &[...claimed shape...])`. The byte-size argument is the real
buffer length, but **MLX drives all gather/matmul/conv index math off the claimed
shape**, and the repo documents that MLX Metal kernels perform **no bounds
checking** (`model/mod.rs:405-411` — the rationale for the embed gather clamp at
`model/mod.rs:413`). A `pub` SDK input struct (re-exported via `ax-engine-core`,
routed through `generate.rs`) carrying a short buffer with inflated dimensions
therefore reads past the allocation during the first vision-projection op.

The repo's own **`gemma4_vl.rs:732`** (`if pixel_values.len() != expected { return Err }`,
mirrored at `:584` for video), **`gemma4_unified.rs:474`**, and **`unlimited_ocr.rs:741`/`:686`**
all validate buffer-vs-shape at the boundary. The three sites below do not.

| # | Family | Site | Trusted shape | Missing guard |
|---|--------|------|---------------|---------------|
| F1a | nemotron_h (omni) | `nemotron_omni.rs:491-496` `build_omni_prefill_embeddings` | `[1,3,height,width]` | `pixel_values.len() == 3*h*w` |
| F1b | minicpmv4_6 | `minicpm_v.rs:547-552` `build_vl_prefill_embeddings` | `[1,height,width,3]` | `pixel_values.len() == h*w*3` |
| F1c | qwen3_vl | `qwen3_vl.rs:235-240` `build_vl_prefill_embeddings` | `[num_patches, patch_dim]` | `patches.len() == num_patches*patch_dim` |

**Downstream validation does not catch it:** the respective `vision.forward` shape
checks compare the *claimed* shape against `grid*patch` geometry (e.g.
`nemotron_omni.rs:192`, `qwen3_vl.rs:892`) — both derived from the same input
struct, so they are trivially self-consistent and never see the real buffer length.

**Suggested fix (same shape for all three):** add a length-vs-shape guard before
the `from_raw_data` call, returning the family's `InvalidGeometry` error on
mismatch — mirror `gemma4_vl.rs:728-738`. Registered as structured findings
`b3955cbbf7a8eaff` / `412c40bfd0339af9` / `4ae50eb6cd2d6bb2`.

**Exploitability note (not a downgrade):** practical reachability depends on
whether a serving path constructs these structs from server-decoded images
(buffer+shape consistent) versus passing caller-controlled values through. Either
way this is a defense-in-depth gap inconsistent with the repo's established
pattern (embed clamp + 3 sibling VLM guards) and is filed at P1/HIGH.

### N1 — MTP fail-closed re-confirmed (NON-FINDING / verified clean)

Re-verified end-to-end against the Wave-0 baseline (`axcode-scan.md` DI-W0-A003):

- **Qwen linear** (`qwen3` / `qwen3_5` / `qwen3_next` linear-attention layers):
  `MtpModelPolicyKind::QwenLinearUncertifiedDirectFallback` is in the
  `route_safe()==false` set (`mtp_model_policy.rs:189-196`); selected by
  `from_loaded` unless env `AX_MLX_QWEN_LINEAR_MTP_CERTIFICATION_CANDIDATE` is
  truthy; runner gates `mtp_requested` on `route_safe()`. Pinned by
  `linear_exact_without_candidate_fails_closed` and
  `default_product_linear_route_is_not_active_without_candidate`.
- **DeepSeek V4**: `DeepseekV4UncertifiedDirectFallback` also in the
  `route_safe()==false` set; selected unless env
  `AX_MLX_DEEPSEEK_V4_MTP_CERTIFICATION_CANDIDATE` is truthy; pinned by
  `deepseek_v4_uncertified_fallback_is_not_route_safe`.

### N2 — INFO (not filed): deepseek_v4 tid2eid gather lacks id clamp

`moe_router_deepseek_v4` (`mlp.rs:3879`) uses raw token ids in a `take(&table,
&ids, ...)` gather without the `embed_tokens_arr` id clamp. Unreachable today
because the V4 runtime is registry-gated out; add the clamp when the V4 graph
goes live.

---

## Completeness

| Dimension | Result | Notes |
|-----------|--------|-------|
| Families covered | 21/21 | All listed families mapped to a primary graph file and reviewed. |
| Review method | bounded source review (5 parallel explore agents over family clusters) + direct verification of every flagged site + positive control (`gemma4_vl.rs:732`) | Every finding and every "CLEAN" verdict is anchored to `file:line`. |
| `cargo check --workspace` | ✅ exit 0 | Clean build. |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | ✅ exit 0 | All residual warnings confined to `ax-engine-sdk/src/session/tests.rs` (test code). **Zero** warnings in any model-family graph under `crates/ax-engine-mlx/src/model/` or the standalone family files. |
| MTP fail-closed | re-confirmed | qwen linear + deepseek v4 (N1). |
| Defect class coverage | one P1 class, 3 sites (F1) | Consistent untrusted-shape `from_raw_data` gap; 3 sibling sites already defend. |
| Out of scope | shared kernels (`shared/mlp.rs` MoE router/gather, `kv_cache.rs`) reviewed for contract only, not exhaustively; scanners are JS/TS-oriented so Rust relied on bounded review + clippy. | No code changed (scan-only); `cargo test` not run because no source was modified — check + clippy + review are the verification for a read-only scan. |

**Overall:** scan complete. 18/21 families clean. The single open P1 class (F1)
has a uniform one-line-per-site fix already proven in `gemma4_vl.rs:732`.
