# N-gram Opportunity Benchmark

Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-8bit/snapshots/0cc7ae1721072b5bcd2716d161e4a3b5e786a11e`  
Prompts: 7 (3 high, 2 med, 2 low repeat)  
Max tokens: 512  

> **Sampling note:** all paths use temperature=0.6/top_p=0.95/top_k=20 for a fair apples-to-apples comparison.
> Lightning n-gram path samples at each position; a draft token is accepted when the sampled token matches the draft.
> Oracle path is post-hoc temperature-sampled simulation (upper bound for any n-gram implementation at this temperature).

## Verdict

> NO N-GRAM OPPORTUNITY: lightning n-gram accept rate on high-repeat prompts is 8.3% (< 10% threshold). The model's output lacks sufficient repeated token patterns — n-gram cannot help regardless of implementation.
> ax-engine n-gram speedup on high-repeat: 1.26×
> lightning n-gram speedup on high-repeat (temp=0.6, vs ax direct): 2.11×

## Per-prompt results

| Prompt | Cat | ax direct (tok/s) | ax ngram (tok/s) | ax speedup | ax accept | lightning (tok/s) | lightning accept | oracle hit | oracle cov |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| getter_setter | hig | 57.2 | 82.5 | 1.44× | 61.5% | 121.9 | — | — | — |
| json_array | hig | 57.9 | 72.9 | 1.26× | 55.9% | 122.0 | 0.0% | — | — |
| sql_inserts | hig | 57.7 | 65.9 | 1.14× | 41.1% | 122.1 | 16.7% | — | — |
| markdown_table | med | 57.8 | 85.9 | 1.49× | 59.6% | 121.9 | 0.0% | — | — |
| csv_data | med | 58.1 | 68.0 | 1.17× | 45.5% | 125.9 | 35.0% | — | — |
| creative_story | low | 57.0 | 65.1 | 1.14× | 46.4% | 122.0 | 0.0% | — | — |
| explain_tcp | low | 57.3 | 64.3 | 1.12× | 41.2% | 122.0 | 0.0% | — | — |

## By category (medians)

| Category | ax direct | ax ngram | ax speedup | ax accept | lightning | lightning speedup | lightning accept | oracle hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| high_repeat | 57.7 | 72.9 | 1.26× | 52.9% | 122.0 | 2.11× | 8.3% | — |
| med_repeat | 57.9 | 76.9 | 1.33× | 52.5% | 123.9 | 2.14× | 17.5% | — |
| low_repeat | 57.2 | 64.7 | 1.13× | 43.8% | 122.0 | 2.13× | 0.0% | — |

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
