# N-gram Opportunity Benchmark

Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`  
Prompts: 7 (3 high, 2 med, 2 low repeat)  
Max tokens: 512  

> **Sampling note:** all paths use temperature=0.6/top_p=0.95/top_k=20 for a fair apples-to-apples comparison.
> Lightning n-gram path samples at each position; a draft token is accepted when the sampled token matches the draft.
> Oracle path is post-hoc temperature-sampled simulation (upper bound for any n-gram implementation at this temperature).

## Verdict

> AX N-GRAM ON PAR WITH LIGHTNING: lightning accept=12.5%, ax accept=52.1% (gap=-39.6%).
> ax-engine n-gram speedup on high-repeat: 1.07×
> lightning n-gram speedup on high-repeat (temp=0.6, vs ax direct): 1.95×

## Per-prompt results

| Prompt | Cat | ax direct (tok/s) | ax ngram (tok/s) | ax speedup | ax accept | lightning (tok/s) | lightning accept | oracle hit | oracle cov |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| getter_setter | hig | 76.9 | 88.8 | 1.15× | 56.9% | 150.0 | — | 12.2% | 16.9% |
| json_array | hig | 76.7 | 82.1 | 1.07× | 48.3% | 149.9 | 0.0% | 5.9% | 8.6% |
| sql_inserts | hig | 77.1 | 82.5 | 1.07× | 51.2% | 150.2 | 25.0% | 4.7% | 9.8% |
| markdown_table | med | 76.6 | 91.1 | 1.19× | 60.3% | 148.6 | 2.5% | 5.9% | 12.2% |
| csv_data | med | 77.0 | 83.5 | 1.08× | 48.2% | 150.0 | 0.0% | 7.1% | 12.9% |
| creative_story | low | 78.1 | 82.9 | 1.06× | 42.3% | 150.3 | 0.0% | 4.3% | 9.0% |
| explain_tcp | low | 76.6 | 83.3 | 1.09× | 48.7% | 150.2 | 0.0% | 4.3% | 7.5% |

## By category (medians)

| Category | ax direct | ax ngram | ax speedup | ax accept | lightning | lightning speedup | lightning accept | oracle hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| high_repeat | 76.9 | 82.5 | 1.07× | 52.1% | 150.0 | 1.95× | 12.5% | — |
| med_repeat | 76.8 | 87.3 | 1.14× | 54.2% | 149.3 | 1.94× | 1.2% | — |
| low_repeat | 77.3 | 83.1 | 1.07× | 45.5% | 150.2 | 1.94× | 0.0% | — |

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
