# Qwen 3.8 Flash Next

Status: **Incubating** (dedicated graph implemented; public qualification open)

Best-experience SKU: **Mac Studio M5 Ultra, 256 GB**

Last reviewed: **2026-09-15**

Qwen 3.8 Flash Next is a second product SKU, distinct from Qwen 3.8 27B
and Super-class Qwen 3.8 (2.4T). Its HF identity is `model_type=qwen4_exp`:
125B-A6B hybrid Gated-DeltaNet / sparse-attention MoE plus a 51B n-gram table.
MLX remains the tensor and quantized matrix multiplication engine.

## Implemented development path

- Dedicated GDN, QSA, gated residual streams, MoE, PLE and final mixer.
- Bounded disk n-gram row reads; table payloads are excluded from weight load.
- Transactional request state, cache serialization, prefix restore and verified
  speculative replay across the recurrent, attention and n-gram state.
- Affine expert paging with explicit resident/streaming admission.
- A separately attached, default-off MTP candidate with primary verification.
  A sidecar alone does not establish an attached or qualified draft head.

Tiny F32/BF16 oracle checks and bounded real 4-bit development tests passed.
Development evidence on an M2 Ultra 192 GB includes native HTTP/SSE, request
isolation, cancellation recovery, prefix reuse, long prompts, resident/paged
control tokens, and explicit MTP runner/HTTP checks. This is not M5 Ultra
certification, a trained MTP-head oracle, or a published throughput result.
Quantized prefill comparisons must use the same chunk schedule.

## Public admission and remaining gates

Conversion preserves the dedicated metadata and validates tensor geometry.
Default auto-generation/load/serve reject `runtime_status.ready=false` and the
legacy blocker `qwen4_exp_native_trunk_not_implemented`. That compatibility
identifier predates the development graph; the remaining gate is public artifact
qualification. An explicit experimental opt-in admits only the audited affine
4-bit expert layout (4/8-bit projections) through all ordinary tensor, file and
geometry checks. It does not rewrite readiness or imply certification.

There is no download alias or generic Compatible route. Never remap this model
onto `qwen3_5` or Super-class 2.4T. Broader checkpoint/oracle, model quality,
long-context, MTP profitability and target-SKU qualification remain open.
Custom Metal experiments require numerical and measured controls before promotion.

The default local pack remains Qwen 3.8 27B AXQ on Mac mini M5 64 GB:
[Qwen 3.8 27B AXQ certification](qwen3.8-27b-axq.md).

## Operator contract

```bash
python3 scripts/qualify_qwen38_flash_next.py --dry-run
```

The script reports the admission contract without loading weights. A live
`--model-dir` qualification run remains closed pending public qualification.

For development with the audited AXQuant 1.9.0 4-bit Flash Next export, the
native server can generate its metadata manifest and load with explicit opt-in:

```bash
AX_ENGINE_FLASH_NEXT_EXPERIMENTAL=1 ax-engine-server --mlx \
  --mlx-model-artifacts-dir /path/to/flash-next-4bit \
  --host 127.0.0.1 --port 31418
```

The loader rechecks the audited exporter/source identity and convolution layout.
Unknown layouts, other expert bit widths, missing files and additional blockers
remain rejected. Removing the opt-in rejects the manifest again. Expert residency
uses the existing Auto/On/Off policy; forced whole-layer paging has substantial
transfer cost. MTP remains a separate opt-in candidate and is not certified.
See [Testing](../TESTING.md) and [Supported Models](../SUPPORTED-MODELS.md).
