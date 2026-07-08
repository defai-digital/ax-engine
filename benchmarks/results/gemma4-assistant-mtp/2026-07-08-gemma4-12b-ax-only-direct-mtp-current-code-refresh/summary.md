# Gemma 4 Assistant MTP Benchmark

Output: `benchmarks/results/gemma4-assistant-mtp/2026-07-08-gemma4-12b-ax-only-direct-mtp-current-code-refresh`

| Model | Suite | Profile | Mode | Depth | Decode tok/s | Affine max-bits | 8-bit tensors | Assistant accept | MTP accept | n-gram accept | n-gram hits | Utility gates | Safety tightens |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 12B 4-bit-FFN | flappy | direct | direct | 2 | 58.9 | 4 | 0 | n/a | n/a | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | flappy | assistant_mtp_depth2 | mtp | 2 | 97.9 | 4 | 0 | 98.4% | 98.7% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | long_code | direct | direct | 2 | 58.1 | 4 | 0 | n/a | n/a | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | long_code | assistant_mtp_depth2 | mtp | 2 | 96.3 | 4 | 0 | 99.1% | 99.2% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | python_modules_long | direct | direct | 2 | 58.9 | 4 | 0 | n/a | n/a | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | python_modules_long | assistant_mtp_depth2 | mtp | 2 | 90.0 | 4 | 0 | 97.0% | 97.2% | n/a | 0 | 0 | 0 |

## Optimized scenario

| Model | Profile | Mode | Decode tok/s | Direct tok/s | Δ vs direct | Worst suite Δ | Classification |
|---|---|---|---:|---:|---:|---:|---|
| 12b-4bit-ffn4 | assistant_mtp_depth2 | mtp | 96.3 | 58.9 | +63.6% | +52.8% | keep-default |

## Same-artifact survival comparison

Direct-baseline rows share the same target artifact; assistant-MTP+n-gram rows use pure assistant-MTP as their baseline.

| Model | Profile | Mode | Baseline | Decode tok/s | Baseline tok/s | Δ vs baseline | Worst suite Δ | Parity | Drafted | Classification |
|---|---|---|---|---:|---:|---:|---:|:---:|:---:|---|
| 12b-4bit-ffn4 | assistant_mtp_depth2 | mtp | direct | 96.3 | 58.9 | +63.6% | +52.8% | yes | yes | keep-default |
