# Qwen 3.8 Flash Next

Status: **Incubating** (no repo-owned graph)

Best-experience SKU: **Mac Studio M5 Ultra, 256 GB**

Last reviewed: **2026-09-15**

Qwen 3.8 Flash Next is a **second product SKU**, not a rename of Qwen 3.8 27B
and not Super-class Qwen 3.8 (2.4T). Upstream identity is HF
`model_type=qwen4_exp`: 125B-A6B hybrid Gated-DeltaNet / sparse-attention MoE
plus a 51B n-gram embedding table.

Convert maps `qwen4_exp` metadata (family label, GDN/MoE dims, n-gram/HC/QSA
contract with `never_eval_ngram_at_load`), gated-residual hyper-connection
and PLE inject tensors, QSA `index_qk_proj` plus q/k layernorms, packed
`mlp.experts.{gate_up,down}_proj` (packed or split gate/up) and `shared_expert`
/ `shared_experts` routers (shared-expert presence follows
`shared_expert_intermediate_size` when `n_shared_experts` is omitted), PLE
`ngram_heads_*` tables, and PLE shards
as `NgramEmbedding` (including `weight_scale`). MTP sidecars are preserved as
`Other`. `load_weights` excludes n-gram shards and scale from the resident
eval set. Auto-generate / load / serve stay fail-closed until the trunk is
implemented. Layer-forward and the generic decode entry reject `qwen4_exp`
before the Qwen 3.5 linear-attention short-circuit.
There is no download alias and no Compatible generic load path. Never remap
onto `qwen3_5` or Super-class 2.4T.

The default local pack remains Qwen 3.8 27B AXQ on Mac mini M4 Pro 64 GB:
[Qwen 3.8 27B AXQ certification](qwen3.8-27b-axq.md).

## Why it is incubating

- No repo-owned `ax-engine-mlx` trunk for `qwen4_exp`.
- Custom Metal for GDN, sparse attention, hyper-connections, and n-gram
  gather is planned **after** convert+trunk exist. MLX stays the QMM engine.
- Evidence belongs on Mac Studio M5 Ultra 256 GB. Mini M4 Pro 64 GB and laptop
  campaign hosts are not this SKU.

## Operator contract

```bash
python3 scripts/qualify_qwen38_flash_next.py --dry-run
```

A live `--model-dir` run is expected to fail closed until Phase 3/4 land.
See [Testing](../TESTING.md) and
[Supported Models](../SUPPORTED-MODELS.md).
