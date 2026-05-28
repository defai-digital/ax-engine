# N-gram Opportunity Benchmark

Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/efbeee6e582ebfd06abc9d65e90839c4b5d2116b`  
Prompts: 7 (3 high, 2 med, 2 low repeat)  
Max tokens: 512  

> **Sampling note:** all paths use temperature=0.6/top_p=0.95/top_k=20 for a fair apples-to-apples comparison.
> Lightning n-gram path samples at each position; a draft token is accepted when the sampled token matches the draft.
> Oracle path is post-hoc temperature-sampled simulation (upper bound for any n-gram implementation at this temperature).

## Verdict

> NO N-GRAM OPPORTUNITY: lightning n-gram accept rate on high-repeat prompts is 1.6% (< 10% threshold). The model's output lacks sufficient repeated token patterns — n-gram cannot help regardless of implementation.

## Per-prompt results

| Prompt | Cat | ax direct (tok/s) | ax ngram (tok/s) | ax speedup | ax accept | lightning (tok/s) | lightning accept | oracle hit | oracle cov |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| getter_setter | hig | — | — | — | — | 102.3 | — | — | — |
| json_array | hig | — | — | — | — | 102.0 | 0.0% | — | — |
| sql_inserts | hig | — | — | — | — | 100.5 | 3.3% | — | — |
| markdown_table | med | — | — | — | — | 99.7 | 2.5% | — | — |
| csv_data | med | — | — | — | — | 102.7 | 35.2% | — | — |
| creative_story | low | — | — | — | — | 101.9 | 0.0% | — | — |
| explain_tcp | low | — | — | — | — | 101.9 | 0.0% | — | — |

## By category (medians)

| Category | ax direct | ax ngram | ax speedup | ax accept | lightning | lightning speedup | lightning accept | oracle hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| high_repeat | — | — | — | — | 102.0 | — | 1.6% | — |
| med_repeat | — | — | — | — | 101.2 | — | 18.9% | — |
| low_repeat | — | — | — | — | 101.9 | — | 0.0% | — |

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
