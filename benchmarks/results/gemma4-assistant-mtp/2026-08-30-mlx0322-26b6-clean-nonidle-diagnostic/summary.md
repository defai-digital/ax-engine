# Gemma 4 Assistant MTP Benchmark

Output: `/Volumes/Ext16TR0/ax-engine/evidence/gemma4-26b-6bit-mlx0322-clean-nonidle-diagnostic-0d779e3a-20260830`

| Model | Suite | Profile | Mode | Depth | Decode tok/s | Affine max-bits | 8-bit tensors | Assistant accept | MTP accept | n-gram accept | n-gram hits | Utility gates | Safety tightens |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 26B A4B AXQ 6-bit | long_code | direct_mlx0322_candidate | direct | 2 | 66.8 | 8 | 31 | n/a | n/a | n/a | 0 | 0 | 0 |
| Gemma 4 26B A4B AXQ 6-bit | long_code | assistant_mtp_mlx0322_candidate | mtp | 2 | 90.4 | 8 | 31 | 90.2% | 90.2% | n/a | 0 | 0 | 0 |

## Optimized scenario

| Model | Profile | Mode | Decode tok/s | Direct tok/s | Δ vs direct | Worst suite Δ | Classification |
|---|---|---|---:|---:|---:|---:|---|
| 26b-a4b-6bit | assistant_mtp_mlx0322_candidate | mtp | 90.4 | 66.8 | +35.4% | +35.4% | keep-default |

## Same-artifact survival comparison

Direct-baseline rows share the same target artifact; assistant-MTP+n-gram rows use pure assistant-MTP as their baseline.

| Model | Profile | Mode | Baseline | Decode tok/s | Baseline tok/s | Δ vs baseline | Worst suite Δ | Parity | Drafted | Classification |
|---|---|---|---|---:|---:|---:|---:|:---:|:---:|---|
| 26b-a4b-6bit | assistant_mtp_mlx0322_candidate | mtp | direct_mlx0322_candidate | 90.4 | 66.8 | +35.4% | +35.4% | yes | yes | keep-default |
