# N-gram Opportunity Benchmark

Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-6bit/snapshots/6fe8c3cfab2910e5bc3439568f6f89413b4d1dca`  
Prompts: 7 (3 high, 2 med, 2 low repeat)  
Max tokens: 512  

> **Sampling note:** all paths use temperature=0.6/top_p=0.95/top_k=20 for a fair apples-to-apples comparison.
> Lightning n-gram path samples at each position; a draft token is accepted when the sampled token matches the draft.
> Oracle path is post-hoc temperature-sampled simulation (upper bound for any n-gram implementation at this temperature).

## Verdict

> NO N-GRAM OPPORTUNITY: lightning n-gram accept rate on high-repeat prompts is 0.0% (< 10% threshold). The model's output lacks sufficient repeated token patterns — n-gram cannot help regardless of implementation.
> ax-engine n-gram speedup on high-repeat: 1.26×
> lightning n-gram speedup on high-repeat (temp=0.6, vs ax direct): 2.23×

## Per-prompt results

| Prompt | Cat | ax direct (tok/s) | ax ngram (tok/s) | ax speedup | ax accept | lightning (tok/s) | lightning accept | oracle hit | oracle cov |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| getter_setter | hig | 59.9 | 85.4 | 1.43× | 63.0% | 135.1 | — | — | — |
| json_array | hig | 60.9 | 76.0 | 1.25× | 53.6% | 135.2 | 0.0% | — | — |
| sql_inserts | hig | 60.5 | 69.6 | 1.15× | 43.3% | 135.2 | 0.0% | — | — |
| markdown_table | med | 60.4 | 79.6 | 1.32× | 55.6% | 135.0 | 5.3% | — | — |
| csv_data | med | 60.9 | 76.8 | 1.26× | 52.5% | 135.4 | 34.1% | — | — |
| creative_story | low | 59.7 | 69.7 | 1.17× | 47.8% | 135.3 | 0.0% | — | — |
| explain_tcp | low | 60.0 | 66.0 | 1.10× | 36.8% | 135.3 | 0.0% | — | — |

## By category (medians)

| Category | ax direct | ax ngram | ax speedup | ax accept | lightning | lightning speedup | lightning accept | oracle hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| high_repeat | 60.5 | 76.0 | 1.26× | 53.3% | 135.2 | 2.23× | 0.0% | — |
| med_repeat | 60.7 | 78.2 | 1.29× | 54.1% | 135.2 | 2.23× | 19.7% | — |
| low_repeat | 59.9 | 67.9 | 1.13× | 42.3% | 135.3 | 2.26× | 0.0% | — |

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
