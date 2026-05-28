# N-gram Opportunity Benchmark

Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit/snapshots/cc3b666c01c20395e0dcebd53854504c7d9821f9`  
Prompts: 7 (3 high, 2 med, 2 low repeat)  
Max tokens: 512  

> **Sampling note:** all paths use temperature=0.6/top_p=0.95/top_k=20 for a fair apples-to-apples comparison.
> Lightning n-gram path samples at each position; a draft token is accepted when the sampled token matches the draft.
> Oracle path is post-hoc temperature-sampled simulation (upper bound for any n-gram implementation at this temperature).

## Verdict

> NO N-GRAM OPPORTUNITY: lightning n-gram accept rate on high-repeat prompts is 7.4% (< 10% threshold). The model's output lacks sufficient repeated token patterns — n-gram cannot help regardless of implementation.
> ax-engine n-gram speedup on high-repeat: 1.31×
> lightning n-gram speedup on high-repeat (temp=0.6, vs ax direct): 2.07×

## Per-prompt results

| Prompt | Cat | ax direct (tok/s) | ax ngram (tok/s) | ax speedup | ax accept | lightning (tok/s) | lightning accept | oracle hit | oracle cov |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| getter_setter | hig | 51.5 | 60.4 | 1.17× | 44.9% | 108.9 | — | — | — |
| json_array | hig | 53.0 | 77.0 | 1.45× | 66.1% | 108.4 | 0.0% | — | — |
| sql_inserts | hig | 52.3 | 68.4 | 1.31× | 57.3% | 108.3 | 14.7% | — | — |
| markdown_table | med | 52.0 | 66.4 | 1.28× | 51.6% | 108.1 | 3.1% | — | — |
| csv_data | med | 51.4 | 57.8 | 1.12× | 43.2% | 109.3 | 19.4% | — | — |
| creative_story | low | 50.6 | 54.4 | 1.07× | 42.6% | 108.8 | 8.3% | — | — |
| explain_tcp | low | 51.0 | 54.4 | 1.07× | 30.3% | 108.7 | 0.0% | — | — |

## By category (medians)

| Category | ax direct | ax ngram | ax speedup | ax accept | lightning | lightning speedup | lightning accept | oracle hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| high_repeat | 52.3 | 68.4 | 1.31× | 56.1% | 108.4 | 2.07× | 7.4% | — |
| med_repeat | 51.7 | 62.1 | 1.20× | 47.4% | 108.7 | 2.10× | 11.2% | — |
| low_repeat | 50.8 | 54.4 | 1.07× | 36.5% | 108.8 | 2.14× | 4.2% | — |

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
