# Qwen 3.8 Flash Next

Status: **Incubating** (no repo-owned graph)

Best-experience SKU: **Mac Studio M5 Ultra, 256 GB**

Last reviewed: **2026-09-14**

Qwen 3.8 Flash Next is a **second product SKU**, not a rename of Qwen 3.8 27B
and not Super-class Qwen 3.8 (2.4T). Upstream identity is HF
`model_type=qwen4_exp`: 125B-A6B hybrid Gated-DeltaNet / sparse-attention MoE
plus a 51B n-gram embedding table.

AX Engine does **not** convert, serve, or benchmark this family yet. Convert
fails closed with `IncubatingQwen38FlashNext`. There is no download alias and
no Compatible generic load path.

The default local pack remains Qwen 3.8 27B AXQ on Mac mini M5 64 GB:
[Qwen 3.8 27B AXQ certification](qwen3.8-27b-axq.md).

## Why it is incubating

- No repo-owned `ax-engine-mlx` trunk for `qwen4_exp`.
- Custom Metal for GDN, sparse attention, hyper-connections, and n-gram
  gather is planned **after** convert+trunk exist. MLX stays the QMM engine.
- Evidence belongs on Mac Studio M5 Ultra 256 GB. Mini M5 64 GB and laptop
  campaign hosts are not this SKU.

## Operator contract

```bash
python3 scripts/qualify_qwen38_flash_next.py --dry-run
```

A live `--model-dir` run is expected to fail closed until Phase 3/4 land.
See [Testing](../TESTING.md) and
[Supported Models](../SUPPORTED-MODELS.md).
