# Flash Next MXFP4 reference evidence

These completed reference runs use the pinned MXFP4 MTP revision
`0b0bf6c1603054df4a8eef0d4bc96bd4672d2c35` on an Apple M2 Ultra with
192 GiB and local external storage. The product target remains MacBook Pro
M5 Max 128 GiB with NAS storage over SMB. Reference results do not qualify
that target or establish AX numerical acceptance, performance or release readiness.

## Numerical reference collection

The official chunked graph, official recurrent graph and unchanged MLX-VLM
graph each completed eight frozen inputs and 3,316 logit positions. The inputs
reuse the earlier affine holdout texts for this different quantization lane;
they are not claimed to be globally unseen. Each graph follows the same
official teacher trajectory with chunk size 128 and four singleton decisions.
All 47 pack files passed post-run verification. All 48 saved array/sidecar
files also passed hash verification before and after CPU scoring.

Compared with the official chunked graph:

| Graph | Positions | Mean KL, nats | Top-1 disagreements |
| --- | ---: | ---: | ---: |
| Official recurrent | 3,316 | 0.02025978695 | 284 / 3,316 |
| MLX-VLM | 3,316 | 0.02128314131 | 268 / 3,316 |

The same frozen statistical comparator produced these values; the published
per-position rows were independently re-aggregated. The original bounds remain
1.25 times the official chunked/recurrent floor for mean KL and top-1
disagreement, plus at most 1% AX-only disagreement where reference margin is
strictly greater than 1.0. AX data is absent, so acceptance is **not evaluated**.
The final decision still requires all four graphs and matching AX/floor coverage.

The official forward code is unchanged. The owned test adapter binds MXFP4
weights and U8 scales; the graph paths share MLX quantization primitives and
do not constitute independent implementations of the MXFP4 quantizer.
Module revisions and hashes are in `summary.json`.

The first CPU analysis attempt failed before scoring because its wrapper used
the wrong saved-array directory. The retry checked all paths against the
completed collection commands and passed. The failed record remains in
`integrity.json.gz`; no model collection was rerun for this path correction.

## Functional reference QA

The separate MLX-VLM reference replay completed all 105 original inputs with
normal stops, valid artifacts and post-run hashes. It passed 101 original
quality checks and failed four. The original verdict remains failed:

- `instruction_alphabet_first`: `A, B, C, D, E`; the comma-only check rejects
  spaces. It accepts either case without spaces, so this is not a case mismatch.
- `science_gravity_earth`: `9`, against the frozen expected `10` and aliases
  `9.8` / `9.81`.
- `knowledge_water_formula`: `H₂O`, while the checker accepts ASCII `H2O`.
- `format_csv_pair`: `pair,1`, while the prompt and checker require `name,1`.

No output normalization, alias changes or retrospective score changes were
applied. These are reference QA results, not completed AX direct/MTP parity.
They also do not replace the required `mlx_lm.benchmark` primary baseline,
whose `qwen4_exp` admission gap remains open.

## Artifact interpretation

`summary.json` identifies the source modules, raw-source hashes, public artifact
hashes and the exact scope of each result. The compressed files preserve all
numeric values, output tokens, texts and verdicts; private location strings
are replaced by stable hash labels. Gzip timestamps are fixed for reproducibility.

- `reference-statistics.json.gz`: full per-position reference comparisons.
- `reference-qa.json.gz`: all 105 reference outputs, tokens and original checks.
- `integrity.json.gz`: terminal collection, scoring and QA integrity records,
  including the failed first analysis attempt.

Elapsed times in these artifacts are retained observations, not an AX/M5
throughput comparison or a cold-storage benchmark. `qualification`,
`numerical_acceptance`, `performance_claim` and `release_ready` remain false.
