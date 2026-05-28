# N-gram Opportunity Benchmark

Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-4bit/snapshots/c000ac2c2057d94be3fa931000c31723aac53282`  
Prompts: 7 (3 high, 2 med, 2 low repeat)  
Max tokens: 512  

## Verdict


## Per-prompt results

| Prompt | Cat | ax direct (tok/s) | ax ngram (tok/s) | ax speedup | ax accept | oracle hit | oracle cov |
|---|---|---:|---:|---:|---:|---:|---:|
| getter_setter | hig | — | — | — | — | — | — |
| json_array | hig | — | — | — | — | — | — |
| sql_inserts | hig | — | — | — | — | — | — |
| markdown_table | med | — | — | — | — | — | — |
| csv_data | med | — | — | — | — | — | — |
| creative_story | low | — | — | — | — | — | — |
| explain_tcp | low | — | — | — | — | — | — |

## By category (medians)

| Category | ax direct | ax ngram | speedup | ax accept | oracle hit |
|---|---:|---:|---:|---:|---:|
| high_repeat | — | — | — | — | — |
| med_repeat | — | — | — | — | — |
| low_repeat | — | — | — | — | — |

## Column definitions

- **ax direct**: ax-engine with n-gram disabled, baseline throughput
- **ax ngram**: ax-engine with n-gram acceleration enabled
- **ax accept**: fraction of ax-engine draft tokens accepted by the verifier
- **oracle hit**: fraction of output tokens the lightning-style n-gram drafter
  would predict correctly (post-hoc analysis on baseline output; upper bound for any n-gram implementation)
- **oracle cov**: fraction of decode steps where the n-gram drafter proposed any draft

Oracle hit < 10% on high-repeat → fundamental no-opportunity: the model does not
repeat enough token patterns for n-gram to help, regardless of implementation.
