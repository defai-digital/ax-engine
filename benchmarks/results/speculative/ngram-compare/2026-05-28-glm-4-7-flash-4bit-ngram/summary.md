# N-gram Opportunity Benchmark

Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--GLM-4.7-Flash-4bit/snapshots/1454cffb1a21737e162f508e5bc70be9def89276`  
Prompts: 7 (3 high, 2 med, 2 low repeat)  
Max tokens: 512  

> **Sampling note:** all paths use temperature=0.6/top_p=0.95/top_k=20 for a fair apples-to-apples comparison.
> Lightning n-gram path samples at each position; a draft token is accepted when the sampled token matches the draft.
> Oracle path is post-hoc temperature-sampled simulation (upper bound for any n-gram implementation at this temperature).

## Verdict

> NO N-GRAM OPPORTUNITY: lightning n-gram accept rate on high-repeat prompts is 0.0% (< 10% threshold). The model's output lacks sufficient repeated token patterns — n-gram cannot help regardless of implementation. Oracle first-hit also confirms: 6.7%.
> ax-engine n-gram speedup on high-repeat: 1.12×
> lightning n-gram speedup on high-repeat (temp=0.6, vs ax direct): 1.32×

## Per-prompt results

| Prompt | Cat | ax direct (tok/s) | ax ngram (tok/s) | ax speedup | ax accept | lightning (tok/s) | lightning accept | oracle hit | oracle cov |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| getter_setter | hig | 63.2 | 74.7 | 1.18× | 55.7% | 83.2 | 0.0% | 6.7% | 13.3% |
| json_array | hig | 63.2 | 67.7 | 1.07× | 52.5% | 83.2 | — | 3.1% | 9.8% |
| sql_inserts | hig | 62.4 | 70.8 | 1.13× | 53.7% | 83.4 | 0.0% | 7.1% | 13.7% |
| markdown_table | med | 62.5 | 76.3 | 1.22× | 57.5% | 83.5 | — | 8.2% | 11.8% |
| csv_data | med | 63.0 | 71.8 | 1.14× | 54.0% | 83.7 | — | 10.6% | 16.5% |
| creative_story | low | 62.9 | 65.8 | 1.04× | 29.8% | 83.6 | — | 5.5% | 9.4% |
| explain_tcp | low | 63.0 | 66.4 | 1.05× | 39.2% | 83.6 | — | 4.7% | 12.5% |

## By category (medians)

| Category | ax direct | ax ngram | ax speedup | ax accept | lightning | lightning speedup | lightning accept | oracle hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| high_repeat | 63.2 | 70.8 | 1.12× | 53.9% | 83.2 | 1.32× | 0.0% | — |
| med_repeat | 62.7 | 74.0 | 1.18× | 55.8% | 83.6 | 1.33× | — | — |
| low_repeat | 63.0 | 66.1 | 1.05× | 34.5% | 83.6 | 1.33× | — | — |

## Column definitions

- **ax direct**: ax-engine with n-gram disabled, baseline throughput (temperature=0.6)
- **ax ngram**: ax-engine with n-gram acceleration enabled (temperature=0.6)
- **ax accept**: fraction of ax-engine draft tokens accepted by the verifier
- **lightning**: mlx_lm model with real n-gram speculative decode (temperature=0.6 acceptance)
- **lightning accept**: fraction of n-gram draft tokens accepted (temperature=0.6; comparable to ax accept)
- **lightning speedup**: lightning n-gram tok/s ÷ ax direct tok/s (different sampling, indicative only)
- **oracle hit**: fraction of output tokens the n-gram drafter would predict correctly
  (post-hoc temperature-sampled simulation; theoretical upper bound for any n-gram implementation at temp=0.6)
- **oracle cov**: fraction of decode steps where the n-gram drafter proposed any draft

**Interpretation:** If lightning accept < 10% on high-repeat prompts, the model itself
has no n-gram opportunity (oracle hit will also be low). If lightning accept is high but
ax accept is low, ax-engine is under-performing on n-gram speculation for this model.
