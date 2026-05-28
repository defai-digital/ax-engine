# N-gram Opportunity Benchmark

Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`  
Prompts: 7 (3 high, 2 med, 2 low repeat)  
Max tokens: 512  

> **Sampling note:** ax-engine paths use temperature=0.6/top_p=0.95/top_k=20 (real-world).
> Lightning n-gram path uses greedy (argmax) acceptance, matching lightning-mlx --mtp-optimistic.
> Oracle path is post-hoc greedy simulation (upper bound for any n-gram implementation).

## Verdict

> AX N-GRAM ON PAR WITH LIGHTNING: lightning accept=35.7%, ax accept=51.9% (gap=-16.2%).
> ax-engine n-gram speedup on high-repeat: 1.08×
> lightning n-gram speedup on high-repeat (greedy, vs ax direct): 2.02×

## Per-prompt results

| Prompt | Cat | ax direct (tok/s) | ax ngram (tok/s) | ax speedup | ax accept | lightning (tok/s) | lightning accept | oracle hit | oracle cov |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| getter_setter | hig | 78.7 | 86.0 | 1.09× | 62.1% | 153.4 | — | 11.8% | 16.1% |
| json_array | hig | 75.8 | 79.6 | 1.05× | 39.5% | 154.3 | — | 7.8% | 12.5% |
| sql_inserts | hig | 76.0 | 82.4 | 1.08× | 54.2% | 151.5 | 35.7% | 5.1% | 9.4% |
| markdown_table | med | 75.7 | 85.8 | 1.13× | 48.4% | 153.6 | 0.0% | 15.3% | 24.3% |
| csv_data | med | 76.0 | 83.3 | 1.10× | 49.6% | 153.3 | 0.0% | 6.3% | 10.6% |
| creative_story | low | 77.5 | 81.0 | 1.04× | 31.8% | 149.4 | 0.0% | 3.5% | 7.1% |
| explain_tcp | low | 76.1 | 78.7 | 1.03× | 42.1% | 153.5 | — | 5.1% | 11.0% |

## By category (medians)

| Category | ax direct | ax ngram | ax speedup | ax accept | lightning | lightning speedup | lightning accept | oracle hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| high_repeat | 76.0 | 82.4 | 1.08× | 51.9% | 153.4 | 2.02× | 35.7% | — |
| med_repeat | 75.8 | 84.6 | 1.12× | 49.0% | 153.4 | 2.02× | 0.0% | — |
| low_repeat | 76.8 | 79.8 | 1.04× | 37.0% | 151.4 | 1.97× | 0.0% | — |

## Column definitions

- **ax direct**: ax-engine with n-gram disabled, baseline throughput (temperature=0.6)
- **ax ngram**: ax-engine with n-gram acceleration enabled (temperature=0.6)
- **ax accept**: fraction of ax-engine draft tokens accepted by the verifier
- **lightning**: mlx_lm model with real n-gram speculative decode (greedy acceptance)
- **lightning accept**: fraction of n-gram draft tokens accepted (greedy; upper-bound for temp=0.6)
- **lightning speedup**: lightning n-gram tok/s ÷ ax direct tok/s (different sampling, indicative only)
- **oracle hit**: fraction of output tokens the n-gram drafter would predict correctly
  (post-hoc greedy simulation; theoretical upper bound for any n-gram implementation)
- **oracle cov**: fraction of decode steps where the n-gram drafter proposed any draft

**Interpretation:** If lightning accept < 10% on high-repeat prompts, the model itself
has no n-gram opportunity (oracle hit will also be low). If lightning accept is high but
ax accept is low, ax-engine is under-performing on n-gram speculation for this model.
