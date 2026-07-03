# N-gram Opportunity Benchmark

Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-5bit/snapshots/7e2d6526209badeacaf09510e86528a107369316`  
Prompts: 7 (3 high, 2 med, 2 low repeat)  
Max tokens: 512  

> **Sampling note:** all paths use temperature=0.6/top_p=0.95/top_k=20 for a fair apples-to-apples comparison.
> Lightning n-gram path samples at each position; a draft token is accepted when the sampled token matches the draft.
> Oracle path is post-hoc temperature-sampled simulation (upper bound for any n-gram implementation at this temperature).

## Verdict

> NO N-GRAM OPPORTUNITY: lightning n-gram accept rate on high-repeat prompts is 8.9% (< 10% threshold). The model's output lacks sufficient repeated token patterns — n-gram cannot help regardless of implementation.
> ax-engine n-gram speedup on high-repeat: 1.15×
> lightning n-gram speedup on high-repeat (temp=0.6, vs ax direct): 2.36×

## Per-prompt results

| Prompt | Cat | ax direct (tok/s) | ax ngram (tok/s) | ax speedup | ax accept | lightning (tok/s) | lightning accept | oracle hit | oracle cov |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| getter_setter | hig | 61.8 | 80.7 | 1.31× | 50.3% | 147.4 | — | — | — |
| json_array | hig | 62.4 | 71.6 | 1.15× | 44.7% | 147.5 | 0.0% | — | — |
| sql_inserts | hig | 62.5 | 71.2 | 1.14× | 43.8% | 147.5 | 17.9% | — | — |
| markdown_table | med | 62.5 | 80.2 | 1.28× | 50.2% | 147.3 | 0.0% | — | — |
| csv_data | med | 62.7 | 76.9 | 1.23× | 48.1% | 154.3 | 48.6% | — | — |
| creative_story | low | 61.6 | 71.3 | 1.16× | 46.7% | 147.5 | 0.0% | — | — |
| explain_tcp | low | 62.1 | 68.9 | 1.11× | 39.1% | 147.5 | 0.0% | — | — |

## By category (medians)

| Category | ax direct | ax ngram | ax speedup | ax accept | lightning | lightning speedup | lightning accept | oracle hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| high_repeat | 62.4 | 71.6 | 1.15× | 46.3% | 147.5 | 2.36× | 8.9% | — |
| med_repeat | 62.6 | 78.5 | 1.26× | 49.2% | 150.8 | 2.41× | 24.3% | — |
| low_repeat | 61.8 | 70.1 | 1.13× | 42.9% | 147.5 | 2.39× | 0.0% | — |

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
