# Gemma 4 Assistant MTP Benchmark

Output: `benchmarks/results/gemma4-assistant-mtp/2026-07-04-gemma4-12b-ax-only-direct-mtp-refresh`

| Model | Suite | Profile | Mode | Depth | Decode tok/s | Affine max-bits | 8-bit tensors | Assistant accept | MTP accept | n-gram accept | n-gram hits | Utility gates | Safety tightens |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 12B 4-bit-FFN | flappy | direct | direct | 2 | 58.7 | 4 | 0 | n/a | n/a | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | flappy | assistant_mtp_depth2 | mtp | 2 | 94.5 | 4 | 0 | 98.5% | 98.8% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | long_code | direct | direct | 2 | 56.0 | 4 | 0 | n/a | n/a | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | long_code | assistant_mtp_depth2 | mtp | 2 | 93.7 | 4 | 0 | 99.1% | 99.4% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | python_modules_long | direct | direct | 2 | 56.8 | 4 | 0 | n/a | n/a | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | python_modules_long | assistant_mtp_depth2 | mtp | 2 | 88.4 | 4 | 0 | 97.8% | 98.0% | n/a | 0 | 0 | 0 |

## Optimized scenario

| Model | Profile | Mode | Decode tok/s | Direct tok/s | Δ vs direct | Worst suite Δ | Classification |
|---|---|---|---:|---:|---:|---:|---|
| 12b-4bit-ffn4 | assistant_mtp_depth2 | mtp | 93.7 | 56.8 | +65.0% | +55.6% | keep-default |

## Same-artifact survival comparison

Direct-baseline rows share the same target artifact; assistant-MTP+n-gram rows use pure assistant-MTP as their baseline.

| Model | Profile | Mode | Baseline | Decode tok/s | Baseline tok/s | Δ vs baseline | Worst suite Δ | Parity | Drafted | Classification |
|---|---|---|---|---:|---:|---:|---:|:---:|:---:|---|
| 12b-4bit-ffn4 | assistant_mtp_depth2 | mtp | direct | 93.7 | 56.8 | +65.0% | +55.6% | yes | yes | keep-default |
