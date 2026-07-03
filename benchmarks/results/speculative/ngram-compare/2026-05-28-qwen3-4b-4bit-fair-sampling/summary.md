# N-gram Opportunity Benchmark

Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`  
Prompts: 7 (3 high, 2 med, 2 low repeat)  
Max tokens: 512  

> **Sampling note:** all paths use temperature=0.6/top_p=0.95/top_k=20 for a fair apples-to-apples comparison.
> Lightning n-gram path samples at each position; a draft token is accepted when the sampled token matches the draft.
> Oracle path is post-hoc temperature-sampled simulation (upper bound for any n-gram implementation at this temperature).

## Verdict

> NO N-GRAM OPPORTUNITY: lightning n-gram accept rate on high-repeat prompts is 8.9% (< 10% threshold). The model's output lacks sufficient repeated token patterns — n-gram cannot help regardless of implementation. Oracle first-hit also confirms: 4.7%.
> ax-engine n-gram speedup on high-repeat: 1.06×
> lightning n-gram speedup on high-repeat (temp=0.6, vs ax direct): 2.04×

## Per-prompt results

| Prompt | Cat | ax direct (tok/s) | ax ngram (tok/s) | ax speedup | ax accept | lightning (tok/s) | lightning accept | oracle hit | oracle cov |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| getter_setter | hig | 74.4 | 81.6 | 1.10× | 55.9% | 148.5 | — | 12.2% | 15.7% |
| json_array | hig | 72.6 | 77.2 | 1.06× | 41.8% | 148.6 | 0.0% | 4.7% | 9.0% |
| sql_inserts | hig | 72.9 | 77.0 | 1.06× | 45.2% | 148.1 | 17.7% | 4.7% | 10.6% |
| markdown_table | med | 72.4 | 84.6 | 1.17× | 59.5% | 146.1 | 5.0% | 7.1% | 14.1% |
| csv_data | med | 72.7 | 76.6 | 1.05× | 43.5% | 144.5 | 0.0% | 5.9% | 10.2% |
| creative_story | low | 76.9 | 78.0 | 1.02× | 41.3% | 139.8 | 0.0% | 3.9% | 7.8% |
| explain_tcp | low | 76.0 | 80.2 | 1.06× | 48.9% | 135.1 | 0.0% | 5.5% | 9.4% |

## By category (medians)

| Category | ax direct | ax ngram | ax speedup | ax accept | lightning | lightning speedup | lightning accept | oracle hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| high_repeat | 72.9 | 77.2 | 1.06× | 47.7% | 148.5 | 2.04× | 8.9% | — |
| med_repeat | 72.5 | 80.6 | 1.11× | 51.5% | 145.3 | 2.00× | 2.5% | — |
| low_repeat | 76.4 | 79.1 | 1.04× | 45.1% | 137.4 | 1.80× | 0.0% | — |

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
