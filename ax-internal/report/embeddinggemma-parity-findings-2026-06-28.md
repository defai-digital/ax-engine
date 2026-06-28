# EmbeddingGemma Parity Findings

Date: 2026-06-28

## Problem

EmbeddingGemma works operationally through AX Engine, but its embeddings do not yet match the `mlx-embeddings` reference closely enough for a correctness gate.

Observed real-text parity against `mlx-embeddings` single-sample reference:

- `min_ax_single_vs_ref_single`: `0.991011`
- `min_ax_batch_vs_ref_single`: `0.990426`
- `min_ax_single_vs_ax_batch`: `0.999814`

The AX single-vs-batch result is stable, so the current issue is primarily AX-vs-reference numerical parity, not AX internal batch instability.

## Reference Model

Model snapshot used:

`/Volumes/Ext4T/models/hub/models--mlx-community--embeddinggemma-300m-8bit/snapshots/c8316c9e35cf8830541b04d315398c67faa6e497`

Reference implementation:

- Package: `mlx-embeddings`
- Model class: `mlx_embeddings.models.gemma3_text.Model`
- Encoder block source: `mlx_lm.models.gemma3_text`
- Pipeline: token embedding, `sqrt(hidden_size)` scale, bidirectional Gemma3 encoder, final RMSNorm, mean pooling, `dense.0`, `dense.1`, L2 normalization.

## Findings

Qwen embedding models passed the existing verifier at a `0.999` cosine threshold, so the failure is specific to EmbeddingGemma/Gemma3 text embedding behavior.

EmbeddingGemma HTTP/API smoke was operational: output rows and dimensions were correct, and vectors were normalized.

The `mlx-embeddings` reference batch path is not fully batch invariant for variable-length inputs, so the comparison used `mlx-embeddings` single-sample output as the reference for each row.

AX EmbeddingGemma single-vs-batch is stable enough (`0.999814` minimum cosine on the tested real-text batch), which narrows the issue to AX-vs-reference math rather than padding instability inside AX.

## Hypotheses Tested

FFN residual ordering was tested by changing the EmbeddingGemma layer path to apply the FFN residual immediately, matching the reference block order. It did not improve parity; results stayed at `0.991011` / `0.990426`. The change was reverted.

GEGLU and dense FFN fusion were disabled with:

```bash
AX_MLX_GEGLU_MUL_METAL=0
AX_MLX_DENSE_GEGLU_PACKED_METAL=0
AX_MLX_DENSE_QMATMUL_RMS_NORM=0
```

The parity numbers did not change, so these fast paths are not the cause.

Token embedding raw row dequantization was isolated. Direct `take + dequantize` for the first token matched the `mlx-embeddings` quantized embedding row at cosine `1.000000`, so raw embedding weight load and affine quantization metadata are likely correct.

Changing token-id arrays from `uint32` to `int32`, using mode-aware dequantization, and adding `contiguous` before reshape did not improve final AX-vs-reference parity. Those exploratory changes were reverted.

## Current Suspect Area

The remaining likely source is the Gemma3 encoder math path used by AX for EmbeddingGemma:

- Q/K RMSNorm + RoPE path
- bidirectional SDPA with explicit padding mask
- layer-specific sliding/full RoPE selection
- final hidden-state pooling interaction after encoder output

The Dense head is not the first source of drift. Stage checks showed drift exists before the Dense head and is amplified by `dense.0`/`dense.1`.

## Reproduction

Use the real-text parity script pattern from `scripts/bench_embedding_fair.py`:

```bash
.venv/bin/python scripts/bench_embedding_fair.py \
  --model-dir /Volumes/Ext4T/models/hub/models--mlx-community--embeddinggemma-300m-8bit/snapshots/c8316c9e35cf8830541b04d315398c67faa6e497 \
  --model-kind embeddinggemma \
  --pooling mean
```

For targeted parity, compare `make_mlx_embeddings_step(model_dir)` against `make_ax_engine_step(model_dir, pooling="mean", model_id="embeddinggemma")` on fixed real-text tokenized sentences and report minimum cosine for AX single, AX batch, and AX single-vs-batch.

## Recommended Next Step

Add a temporary depth probe for EmbeddingGemma that emits normalized hidden-state cosine after each transformer layer against `mlx-embeddings`/`mlx-lm` for one short prompt. This should identify the first layer where drift appears and decide whether to inspect attention, RoPE, RMSNorm, or residual handling.

Do not update README benchmark/correctness claims for EmbeddingGemma until AX-vs-reference cosine is brought above the same threshold used for Qwen embedding verification.
