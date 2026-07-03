# N-gram Opportunity Benchmark

Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd`  
Prompts: 7 (3 high, 2 med, 2 low repeat)  
Max tokens: 512  

> **Sampling note:** all paths use temperature=0.6/top_p=0.95/top_k=20 for a fair apples-to-apples comparison.
> Lightning n-gram path samples at each position; a draft token is accepted when the sampled token matches the draft.
> Oracle path is post-hoc temperature-sampled simulation (upper bound for any n-gram implementation at this temperature).

## Verdict

> NO N-GRAM OPPORTUNITY: lightning n-gram accept rate on high-repeat prompts is 0.0% (< 10% threshold). The model's output lacks sufficient repeated token patterns — n-gram cannot help regardless of implementation.
> ax-engine n-gram speedup on high-repeat: 1.28×
> lightning n-gram speedup on high-repeat (temp=0.6, vs ax direct): 2.48×

## Per-prompt results

| Prompt | Cat | ax direct (tok/s) | ax ngram (tok/s) | ax speedup | ax accept | lightning (tok/s) | lightning accept | oracle hit | oracle cov |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| getter_setter | hig | 63.0 | 81.7 | 1.30× | 51.7% | 158.8 | — | — | — |
| json_array | hig | 64.4 | 82.1 | 1.27× | 54.8% | 158.7 | 0.0% | — | — |
| sql_inserts | hig | 64.1 | 85.3 | 1.33× | 54.0% | 159.0 | 0.0% | — | — |
| markdown_table | med | 63.9 | 90.5 | 1.42× | 57.6% | 159.1 | 21.9% | — | — |
| csv_data | med | 64.0 | 79.3 | 1.24× | 49.5% | 159.2 | 17.8% | — | — |
| creative_story | low | 62.1 | 66.2 | 1.07× | 29.0% | 159.1 | 0.0% | — | — |
| explain_tcp | low | 62.4 | 73.1 | 1.17× | 45.0% | 158.9 | 0.0% | — | — |

## By category (medians)

| Category | ax direct | ax ngram | ax speedup | ax accept | lightning | lightning speedup | lightning accept | oracle hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| high_repeat | 64.1 | 82.1 | 1.28× | 53.5% | 158.8 | 2.48× | 0.0% | — |
| med_repeat | 64.0 | 84.9 | 1.33× | 53.6% | 159.2 | 2.49× | 19.9% | — |
| low_repeat | 62.3 | 69.6 | 1.12× | 37.0% | 159.0 | 2.55× | 0.0% | — |

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
