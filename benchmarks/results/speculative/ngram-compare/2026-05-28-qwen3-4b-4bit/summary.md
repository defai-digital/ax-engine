# N-gram Opportunity Benchmark

Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`  
Prompts: 7 (3 high, 2 med, 2 low repeat)  
Max tokens: 512  

> **Sampling note:** ax-engine paths use temperature=0.6/top_p=0.95/top_k=20 (real-world).
> Lightning n-gram path uses greedy (argmax) acceptance, matching lightning-mlx --mtp-optimistic.
> Oracle path is post-hoc greedy simulation (upper bound for any n-gram implementation).

## Verdict

> AX NO DRAFT ATTEMPTS on high-repeat (lightning accept=35.7%): ax-engine made no n-gram draft attempts despite model having potential. Likely cause: only_in_think gate is blocking structured output after </think>.
> ax-engine n-gram speedup on high-repeat: 1.00×
> lightning n-gram speedup on high-repeat (greedy, vs ax direct): 2.09×

## Per-prompt results

| Prompt | Cat | ax direct (tok/s) | ax ngram (tok/s) | ax speedup | ax accept | lightning (tok/s) | lightning accept | oracle hit | oracle cov |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| getter_setter | hig | 75.8 | 73.9 | 0.97× | — | 145.5 | — | 11.8% | 16.1% |
| json_array | hig | 72.8 | 72.2 | 0.99× | — | 152.3 | — | 7.8% | 12.5% |
| sql_inserts | hig | 72.7 | 72.5 | 1.00× | — | 153.1 | 35.7% | 5.1% | 9.4% |
| markdown_table | med | 72.2 | 71.8 | 0.99× | — | 151.0 | 0.0% | 15.3% | 24.3% |
| csv_data | med | 72.3 | 72.2 | 1.00× | — | 152.2 | 0.0% | 6.3% | 10.6% |
| creative_story | low | 73.7 | 74.3 | 1.01× | 20.9% | 152.4 | 0.0% | 3.5% | 7.1% |
| explain_tcp | low | 72.5 | 72.5 | 1.00× | 12.5% | 152.4 | — | 5.1% | 11.0% |

## By category (medians)

| Category | ax direct | ax ngram | ax speedup | ax accept | lightning | lightning speedup | lightning accept | oracle hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| high_repeat | 72.8 | 72.5 | 1.00× | — | 152.3 | 2.09× | 35.7% | — |
| med_repeat | 72.3 | 72.0 | 1.00× | — | 151.6 | 2.10× | 0.0% | — |
| low_repeat | 73.1 | 73.4 | 1.00× | 16.7% | 152.4 | 2.08× | 0.0% | — |

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
