# Flash Next statistical acceptance (M2)

- Schema: `ax-engine.qwen38.flash-next.statistical-acceptance.v1`; qualification: false; performance claim: false.
- Evidence host: Apple M2 Ultra 192 GB, not the target SKU.
- Reference: `official-chunked`; floor: `official-recurrent`.
- Collect results: `.internal/reports/flash-next-2026-09-15/host-m2/wi2-collect`.
- Acceptance rule: mean KL and top-1 disagreement at or below the official-recurrent floor times 1.25; AX-only high-margin top-1 disagreements at most 1% of aligned positions; coverage parity.

## Overall versus reference

| graph | positions | mean KL | p95 KL | max KL | top-1 disagreement | mean top-5 Jaccard | max-abs |
|---|---|---|---|---|---|---|---|
| official-recurrent | 3260 | 0.0809088 | 0.19698 | 14.3872 | 0.0257669 | 0.733501 | 16.375 |
| mlx-vlm | 3260 | 0.0897196 | 0.171524 | 22.5736 | 0.0269939 | 0.746504 | 21.9062 |
| ax | 3260 | 0.0859684 | 0.193024 | 11.6293 | 0.0285276 | 0.741166 | 16.3125 |

## Acceptance

- Verdict: **PASS** (encoded rule only; this is not a qualification gate).
- AX mean KL 0.0859684 vs limit 0.101136 (floor 0.0809088 x 1.25): ok.
- AX top-1 disagreement 0.0285276 vs limit 0.0322086 (floor 0.0257669 x 1.25): ok.
- AX-only high-margin top-1 disagreements: 22 / 3260 = 0.00674847 (limit 0.01): ok.
- Coverage parity: ok.

## AX-only top-1 disagreements

| prompt | position | reference top-1 | AX top-1 | recurrent top-1 | mlx-vlm top-1 | mlx-vlm agrees | AX margin | AX rank of reference | reference margin | high-margin |
|---|---|---|---|---|---|---|---|---|---|---|
| short-instruction-sort-numbers | 25 | 11 | 15 | 11 | 271 | false | 0 | 3 | 0 | False |
| short-math-average-scores | 20 | 1132 | 440 | 1132 | 1132 | true | 0.375 | 2 | 0.375 | False |
| short-reasoning-cause-effect | 10 | 30 | 310 | 30 | 30 | true | 0.125 | 2 | 0 | False |
| short-reasoning-cause-effect | 22 | 58540 | 271 | 58540 | 58540 | true | 0.3125 | 2 | 0.75 | False |
| short-reading-count-cats | 16 | 271 | 2500 | 271 | 2500 | false | 0.25 | 2 | 1.5625 | True |
| short-reading-count-cats | 29 | 271 | 25 | 271 | 25 | false | 0.125 | 2 | 1.8125 | True |
| short-science-boiling-celsius | 13 | 440 | 303 | 440 | 303 | false | 0 | 1 | 0.875 | False |
| medium-records-1024 | 12 | 198 | 13 | 198 | 13 | false | 0 | 1 | 0.625 | False |
| medium-records-1024 | 125 | 19 | 20 | 19 | 19 | true | 0.25 | 2 | 0.625 | False |
| medium-records-1024 | 129 | 36055 | 10174 | 36055 | 10174 | false | 0 | 1 | 0.875 | False |
| medium-records-1024 | 345 | 12448 | 10458 | 12448 | 12448 | true | 3.25 | 3 | 7.4375 | True |
| medium-records-1024 | 396 | 2450 | 9476 | 2450 | 9476 | false | 1.75 | 5 | 1.125 | True |
| medium-records-1024 | 479 | 869 | 279 | 869 | 869 | true | 1.9375 | 4 | 4.1875 | True |
| medium-records-1024 | 583 | 1455 | 1754 | 1455 | 1455 | true | 0.1875 | 5 | 5.375 | True |
| medium-records-1024 | 633 | 795 | 9476 | 795 | 3766 | false | 9.9375 | 15 | 2.1875 | True |
| medium-records-1024 | 743 | 220 | 3566 | 220 | 1455 | false | 2.625 | 6 | 0.6875 | False |
| medium-records-1024 | 850 | 12448 | 359 | 12448 | 7500 | false | 0.75 | 2 | 2.5625 | True |
| medium-records-1024 | 907 | 1455 | 279 | 1455 | 3150 | false | 0.625 | 2 | 4.5 | True |
| qsa-boundary-v1 | 12 | 198 | 13 | 198 | 13 | false | 0 | 1 | 0.625 | False |
| qsa-boundary-v1 | 125 | 19 | 20 | 19 | 19 | true | 0.25 | 2 | 0.625 | False |
| qsa-boundary-v1 | 129 | 36055 | 10174 | 36055 | 10174 | false | 0 | 1 | 0.875 | False |
| qsa-boundary-v1 | 345 | 12448 | 10458 | 12448 | 12448 | true | 3.25 | 3 | 7.4375 | True |
| qsa-boundary-v1 | 396 | 2450 | 9476 | 2450 | 9476 | false | 1.75 | 5 | 1.125 | True |
| qsa-boundary-v1 | 479 | 869 | 279 | 869 | 869 | true | 1.9375 | 4 | 4.1875 | True |
| qsa-boundary-v1 | 583 | 1455 | 1754 | 1455 | 1455 | true | 0.1875 | 5 | 5.375 | True |
| qsa-boundary-v1 | 633 | 795 | 9476 | 795 | 3766 | false | 9.9375 | 15 | 2.1875 | True |
| qsa-boundary-v1 | 743 | 220 | 3566 | 220 | 1455 | false | 2.625 | 6 | 0.6875 | False |
| qsa-boundary-v1 | 850 | 12448 | 359 | 12448 | 7500 | false | 0.75 | 2 | 2.5625 | True |
| qsa-boundary-v1 | 907 | 1455 | 279 | 1455 | 3150 | false | 0.625 | 2 | 6.625 | True |
| qsa-boundary-v1 | 1013 | 12448 | 2450 | 12448 | 12448 | true | 1.75 | 7 | 4.3125 | True |
| qsa-boundary-v1 | 1179 | 1543 | 357 | 1543 | 359 | false | 1.4375 | 7 | 0.0625 | False |
| qsa-boundary-v1 | 1642 | 1455 | 198 | 1455 | 1455 | true | 0.3125 | 14 | 11.3125 | True |
| qsa-boundary-v1 | 1804 | 12448 | 1543 | 12448 | 12448 | true | 0.25 | 3 | 5.0625 | True |
| qsa-boundary-v1 | 1806 | 1455 | 220 | 1455 | 1455 | true | 0.9375 | 2 | 11.5 | True |
| qsa-boundary-v1 | 1830 | 9476 | 2450 | 9476 | 2450 | false | 1.8125 | 2 | 1.875 | True |
| qsa-boundary-v1 | 2022 | 12448 | 7189 | 12448 | 12448 | true | 1 | 2 | 10.625 | True |

## Limitations

- Eight frozen prompts; this is not a quality, multilingual, coding, or long-context cohort.
- One QSA pruning-boundary trajectory (qsa-boundary-v1) is included; it is a single structural prompt, not a boundary suite.
- The official eager graph ran with CPU request-state fallbacks; grouped_mm and Metal official paths are not claimed.
- Spotlight indexing was active on the evidence host during collection; timings are not performance evidence.
- An encoded-rule PASS is diagnostic only. It is not Mac Studio M5 Ultra qualification and not a performance claim.
- Apple M2 Ultra 192 GB is not the target Flash Next SKU.
