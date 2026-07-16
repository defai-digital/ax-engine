# Gemma 4 Assistant MTP Benchmark

Output: `benchmarks/results/speculative/gemma4-assistant-mtp/2026-07-16-gemma4-12b-4bit-ax-only-refresh`

| Model | Suite | Profile | Mode | Depth | Decode tok/s | Affine max-bits | 8-bit tensors | Assistant accept | MTP accept | n-gram accept | n-gram hits | Utility gates | Safety tightens |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 12B 4-bit | flappy | direct | direct | 2 | 41.2 | 8 | 144 | n/a | n/a | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit | flappy | assistant_mtp_default | mtp | 2 | 83.9 | 8 | 144 | 98.4% | 98.7% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit | long_code | direct | direct | 2 | 40.7 | 8 | 144 | n/a | n/a | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit | long_code | assistant_mtp_default | mtp | 2 | 80.9 | 8 | 144 | 98.4% | 98.7% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit | python_modules_long | direct | direct | 2 | 41.1 | 8 | 144 | n/a | n/a | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit | python_modules_long | assistant_mtp_default | mtp | 2 | 67.6 | 8 | 144 | 96.8% | 96.9% | n/a | 0 | 0 | 0 |

## Optimized scenario

| Model | Profile | Mode | Decode tok/s | Direct tok/s | Δ vs direct | Worst suite Δ | Classification |
|---|---|---|---:|---:|---:|---:|---|
| 12b-4bit | assistant_mtp_default | mtp | 80.9 | 41.1 | +96.8% | +64.5% | keep-default |

## Same-artifact survival comparison

Direct-baseline rows share the same target artifact; assistant-MTP+n-gram rows use pure assistant-MTP as their baseline.

| Model | Profile | Mode | Baseline | Decode tok/s | Baseline tok/s | Δ vs baseline | Worst suite Δ | Parity | Drafted | Classification |
|---|---|---|---|---:|---:|---:|---:|:---:|:---:|---|
| 12b-4bit | assistant_mtp_default | mtp | direct | 80.9 | 41.1 | +96.8% | +64.5% | yes | yes | keep-default |
