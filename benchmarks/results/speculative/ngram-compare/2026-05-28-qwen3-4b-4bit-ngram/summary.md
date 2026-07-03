# N-gram Opportunity Benchmark

Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`  
Prompts: 7 (3 high, 2 med, 2 low repeat)  
Max tokens: 512  

> **Sampling note:** all paths use temperature=0.6/top_p=0.95/top_k=20 for a fair apples-to-apples comparison.
> Lightning n-gram path samples at each position; a draft token is accepted when the sampled token matches the draft.
> Oracle path is post-hoc temperature-sampled simulation (upper bound for any n-gram implementation at this temperature).

## Verdict

> NO N-GRAM OPPORTUNITY: lightning n-gram accept rate on high-repeat prompts is 7.4% (< 10% threshold). The model's output lacks sufficient repeated token patterns — n-gram cannot help regardless of implementation. Oracle first-hit also confirms: 7.8%.
> ax-engine n-gram speedup on high-repeat: 1.07×
> lightning n-gram speedup on high-repeat (temp=0.6, vs ax direct): 1.94×

## Per-prompt results

| Prompt | Cat | ax direct (tok/s) | ax ngram (tok/s) | ax speedup | ax accept | lightning (tok/s) | lightning accept | oracle hit | oracle cov |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| getter_setter | hig | 78.5 | 88.7 | 1.13× | 56.9% | 150.2 | — | 12.5% | 16.5% |
| json_array | hig | 76.6 | 81.9 | 1.07× | 48.3% | 148.0 | 0.0% | 3.9% | 7.8% |
| sql_inserts | hig | 77.1 | 82.3 | 1.07× | 51.2% | 149.8 | 14.7% | 7.8% | 14.1% |
| markdown_table | med | 76.6 | 91.0 | 1.19× | 60.3% | 148.5 | 0.7% | 9.4% | 16.5% |
| csv_data | med | 77.0 | 83.4 | 1.08× | 48.2% | 149.5 | 1.9% | 6.3% | 10.6% |
| creative_story | low | 78.1 | 82.7 | 1.06× | 42.3% | 150.1 | 0.0% | 3.9% | 8.2% |
| explain_tcp | low | 76.8 | 83.1 | 1.08× | 48.7% | 149.9 | 0.0% | 5.1% | 7.8% |

## By category (medians)

| Category | ax direct | ax ngram | ax speedup | ax accept | lightning | lightning speedup | lightning accept | oracle hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| high_repeat | 77.1 | 82.3 | 1.07× | 52.1% | 149.8 | 1.94× | 7.4% | — |
| med_repeat | 76.8 | 87.2 | 1.14× | 54.2% | 149.0 | 1.94× | 1.3% | — |
| low_repeat | 77.4 | 82.9 | 1.07× | 45.5% | 150.0 | 1.94× | 0.0% | — |

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
