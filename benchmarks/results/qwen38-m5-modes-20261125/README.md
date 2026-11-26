# Qwen 3.8 MXFP4 greedy decode and effective prefill on M5 Max

The MacBook Pro M5 Max 128 GB runs the original
[pinned MXFP4 packs and engine builds](../qwen38-m5-peer-20261125/provenance.json).
Each arm opens one loopback HTTP stream at a time. All requests use
temperature zero. The 27B DFlash2 arms use the same local draft
checkpoint. The separate Flash Next routes use its pinned MXFP4 pack.

## Measurement

- Decode uses the same [fixed corpus](../../corpora/serving/qwen38_dflash2_paired.jsonl)
  as the original comparison. After one warmup, two passes use code, math
  and prose prompts with a 64-token output limit. The throughput numerator
  is the retokenized visible output minus the first text chunk's token
  count, and the denominator is elapsed time between the first and last
  nonempty text chunks. The result pools the six streams.
- Effective prefill is the median over two independently paired input
  sizes. Each probe asks for one output token. The rate divides the input
  token count difference by the first-text latency difference. The second
  pair reverses the prompt-length order. Each prompt uses a distinct marker
  at the beginning to limit prefix reuse. The estimate includes any
  input-length-dependent scheduling, tokenization and HTTP effects.
- The prefill probes run after each decode pass. They are outside the decode
  timer. Results use the same tokenizer and accounting across engines.
- A zero indicates that the tested configuration could not complete the
  workload or prove its selected route. Omitted engine/mode combinations
  have no score. The [original matrix](../qwen38-m5-peer-20261125/README.md)
  records its original loader and first-text results.
- AX Engine's explicit direct mode disables MTP and n-gram acceleration.
  Its DFlash2 arm attaches the local drafter and requires live round traces.
  oMLX's DFlash2 arm enables its native batched drafter per model and
  requires an observed DFlash2 load. Splash 1.7.0 uses its own DFlash2
  implementation. MTPLX uses target-only autoregressive generation and
  mlx-lm serves without a speculative model.

[README tables and charts](../../../README.md#performance) ·
[Per-request measurements and failures](results.jsonl) ·
[Build and pack provenance](provenance.json)

## Paired rerun

The rows marked † in the README tables were measured again later on
2026-11-25, after AX Engine gained a one-forward DFlash2 draft tree, an
MX-format Neural Accelerator verify kernel and Flash Next decode and
prefill changes
([what changed](../../../docs/performance/qwen38-m5-tree-verify-and-flash-next-2026-11-25.md)).
AX Engine was built from commit `b04a5f2ef` with default settings. The
peers, packs, drafter, corpus and runner are those of the original session.

- Six arms ran three times each: AX Engine direct and oMLX direct on both
  models, AX Engine DFlash2 and Splash DFlash2 on the 27B. Every AX Engine
  run was followed by its peer's run, with 30 seconds between arms.
- A † row holds the median of the three per-run decode rates and of the
  three per-run effective-prefill estimates. Its `prefill` list holds all
  six probe pairs, and the first-text columns are medians over them.
  [Every run](paired-rerun.jsonl) is recorded with its requests.
- The [original single runs](results-original.jsonl) are kept unchanged.
  They had AX Engine 8.1.0 at 35.35 and 82.83 tok/s (27B direct and
  DFlash2) and 50.94 tok/s (Flash Next), Splash at 108.95 and oMLX at 59.01.
- The original oMLX Flash Next effective prefill (2,284) came from one cold
  long probe (2.13 s in the first pair, 0.71 s in the second). The rerun
  measures 3,939 to 3,989.
- AX Engine's Flash Next effective prefill is lower than in the original
  run (2,217 against 3,202) although both probes are faster: first text
  fell from 0.69 and 0.64 s to 0.34 and 0.24 s at 127 and 128 input
  tokens, and from 1.09 to 0.91 s at 1,492. The estimate is the slope
  between the two probes, so it falls when the short one improves more.
- Rows without † are single runs from the original session and are not
  paired with the reruns.

Reading the reruns: the 27B DFlash2 decode lead over Splash (2.0% at the
median) holds for this 64-token workload; on 256-token outputs AX Engine
measures about 9% behind. The Flash Next decode difference (0.5%) is
smaller than the difference between sessions and is a tie.

### Flash Next rows after the prefill-schedule change

The two Flash Next † rows were measured once more, later the same day,
after AX Engine stopped ending a prefill forward to capture a prefix that
no request restores
([what changed](../../../docs/performance/flash-next-prefill-schedule-2026-11-25.md)).
AX Engine was built from commit `c3ce06f75` with default settings; its row
carries that version in the README table. The method is that of the paired
rerun: three AX Engine runs, each followed by an oMLX run, 30 seconds
between arms. The six per-run records are the rows of
[paired-rerun.jsonl](paired-rerun.jsonl) with `"rerun": 2`; the earlier
Flash Next runs stay in the same file without that field.

- Decode: 59.29, 59.64 and 59.77 against 59.35, 59.34 and 59.42 tok/s. A
  tie, as before.
- First text at 127 and 128 input tokens: 0.21 and 0.24 s (0.34 and 0.24
  before) against 0.38 and 0.35 s for oMLX. At 1,492: 0.83 s (0.91
  before) against 0.71 s.
- AX Engine's effective prefill is 2,300 (2,217 before) against 3,902.
  Both of its probes got faster, the short one by more.
- The 27B rows were not rerun: the change does not touch that model. One
  27B DFlash2 pair in the same session measured 111.42 against 109.29
  tok/s.

### Flash Next rows after the compiled prefill stretches

The two Flash Next † rows were measured a third time, later the same day,
after AX Engine began to run its compiled elementwise stretches in prefill
forwards
([what changed](../../../docs/performance/flash-next-prefill-fused-stretches-2026-11-25.md)).
AX Engine was built from commit `071e5477a` with default settings; its row
carries that version in the README table. The method is unchanged: three
AX Engine runs, each followed by an oMLX run, 30 seconds between arms. The
six per-run records are the rows of
[paired-rerun.jsonl](paired-rerun.jsonl) with `"rerun": 3`; the rows of the
previous section stay in the same file with `"rerun": 2`.

- Decode: 59.05, 59.36 and 59.71 against 59.12, 59.39 and 59.68 tok/s. A
  tie, as before.
- First text at 1,492 input tokens: 0.70 to 0.72 s (median 0.705; 0.83
  before) against 0.70 to 0.71 s (median 0.706) for oMLX. At 127 and 128:
  0.21 and 0.24 s against 0.38 and 0.35 s.
- AX Engine's effective prefill is 2,838 (2,300 before) against 4,017. The
  estimate divides by the difference of the two first-text latencies, so
  oMLX's slower short probe raises its figure: the two engines' long
  probes are level.
- The 27B rows were not rerun: the change does not touch that model. One
  27B DFlash2 pair in the same session measured 110.67 against 109.29
  tok/s, with effective prefill at 966 against 1,282.

### 27B rows at the same commit

The four 27B † rows were then measured again at commit `071e5477a`, so
that every † row of both tables comes from one AX Engine build. Three
rounds, each of AX Engine direct, oMLX direct, AX Engine DFlash2 and
Splash DFlash2 in that order, 30 seconds between arms. The twelve per-run
records are the rows of [paired-rerun.jsonl](paired-rerun.jsonl) with
`"rerun": 4`; the first rerun's 27B rows stay in the same file without
that field.

- Direct greedy: 35.37, 35.20 and 35.30 against 34.49, 34.44 and 34.50
  tok/s for oMLX; first text at 1,492 input tokens 1.63 against 1.51 s.
- DFlash2: 111.27, 111.00 and 110.84 against 109.30, 109.12 and 109.16
  tok/s for Splash; first text at 1,492 input tokens 1.69 against 1.22 s,
  effective prefill 936 against 1,286.
- Against the first rerun the decode rates moved by less than 0.3%. The
  DFlash2 arms' long probe is 0.03 to 0.05 s slower for both engines in
  this session (AX Engine 1.64 to 1.69 s, Splash 1.19 to 1.22 s).
- The 256-token comparison quoted above was not repeated.

### Every † row again, with rest before each arm

All six † rows were then measured once more with AX Engine built from
commit `c688907a6` (default settings), which attends long Flash Next
prompts with one masked dense call per sparse-attention layer. The earlier
reruns left 30 seconds between arms. Here the host rested 150 seconds
before every arm: on this laptop a scan started right after another one
read 0.83 s where a rested one reads 0.70 s.

Three rounds per model. Flash Next: AX Engine, then oMLX. 27B: AX Engine
direct, oMLX direct, AX Engine DFlash2, Splash DFlash2. The eighteen
per-run records are the rows of [paired-rerun.jsonl](paired-rerun.jsonl)
with `"rerun": 5` and `"rest_seconds_before_arm": 150`; the README tables
and charts show their medians.

- Flash Next decode: 59.50, 59.38 and 59.56 against 59.39, 59.51 and 59.25
  tok/s for oMLX. Still a tie.
- Flash Next first text at 1,492 input tokens: 0.70 to 0.72 s (median
  0.705) against 0.71 s (median 0.708); at 127 and 128: 0.22 against
  0.36 s. The masked dense call does not run at these lengths.
- 27B direct greedy: 35.33, 35.19 and 35.31 against 34.56, 34.48 and 34.59
  tok/s for oMLX; first text at 1,492 input tokens 1.63 against 1.51 s.
- 27B DFlash2: 111.09, 111.66 and 111.51 against 109.08, 109.42 and 109.25
  tok/s for Splash; first text at 1,492 input tokens 1.64 against 1.19 s,
  effective prefill 967 against 1,330.
- Against the 30-second reruns the decode rates and the direct first-text
  latencies moved by less than 0.5%. The two DFlash2 long probes read 0.03
  to 0.05 s faster, for both engines alike (effective prefill 936 to 967
  and 1,286 to 1,330).

### 27B rows after the greedy prefill schedule change

The four 27B † rows were measured again after AX Engine stopped splitting a
greedy prompt at its last 16-token block boundary and began running MLX's
gated-delta prefill recurrence on M5 and later
([what changed](../../../docs/performance/qwen38-27b-prefill-schedule-2026-11-25.md)).
Three campaigns of three rounds each, in the order AX Engine direct, oMLX
direct, AX Engine DFlash2, Splash DFlash2, with 150 seconds of rest before
every arm. The Flash Next rows were not rerun in these campaigns.

`"rerun": 8` is AX Engine at commit `df8c86ecf` with default settings.

- Direct greedy: 35.46, 35.45 and 35.46 against 34.68, 34.67 and 34.67
  tok/s for oMLX. First text at 127 and 128 input tokens 0.17 to 0.18 s
  (median 0.171) against 0.22 s. At 1,492 input tokens the six AX Engine
  probes read 1.497 to 1.501 s and the six oMLX probes 1.507 to 1.513 s
  (medians 1.501 and 1.509). It was 0.23 and 1.63 s in the previous rerun.
- DFlash2: 114.02, 114.11 and 114.09 against 108.98, 108.81 and 108.91
  tok/s for Splash. First text at 127 and 128 input tokens 0.18 against
  0.17 s; at 1,492 input tokens 1.50 against 1.19 s (medians 1.504 and
  1.193); effective prefill 1,022 against 1,331.
- One of the three direct replies equals oMLX's (the second). The third
  equals it with the sequential recurrence (rerun 7) and not here; the
  first differs in every configuration. The DFlash2 replies are those of
  the rerun before the schedule change.

`"rerun": 7` is commit `0e642449e`, which ran the sequential recurrence:
what `AX_MLX_QWEN_GATED_DELTA_PREFILL_MLX=0` selects now. It isolates the
schedule change.

- Direct greedy: 35.39, 35.39 and 35.29 against 34.56, 34.55 and 34.64
  tok/s; first text at 127 and 128 input tokens 0.18 against 0.22 s and at
  1,492 input tokens 1.57 against 1.52 s (medians 1.566 and 1.517).
- DFlash2: 116.45, 116.26 and 116.65 against 108.96, 108.91 and 108.68
  tok/s; first text at 1,492 input tokens 1.58 against 1.19 s. The third
  prompt's reply equals the direct path's in this configuration and
  decodes at 89 to 91 tok/s, against 85 to 86 with the reply the MLX
  recurrence leads to; the DFlash2 medians of reruns 7 and 8 differ by
  that reply, not by a kernel.
- Against the rerun before the schedule change, the second DFlash2 prompt
  decodes at 134 to 136 tok/s where it decoded at 126 to 127, with the same
  reply, in reruns 6, 7 and 8 alike. The decode path was not changed; the
  cause was not traced further.

`"rerun": 6` is commit `414607619`, the same configuration as rerun 8 at an
earlier commit: direct 35.34, 35.22 and 35.37 tok/s with first text 1.51 s
at 1,492 input tokens; DFlash2 113.93, 113.79 and 113.80 tok/s.

### Every † row after the shared tokenizer

All six † rows were measured again on 2026-11-26 with AX Engine built from
commit `02ab4a08a` (default settings). Against rerun 8 that build keeps the
wired weights GPU-resident while idle and shares one tokenizer between
requests instead of copying it for each
([what changed](../../../docs/performance/server-tokenizer-shared-2026-11-26.md)).
Three rounds per model in the order of rerun 5, with 150 seconds of rest
before every arm. The records are the rows of
[paired-rerun.jsonl](paired-rerun.jsonl) with `"rerun": 9`; the README
tables and charts show their medians.

- Flash Next decode: 59.56, 59.77 and 59.74 against 59.58, 59.72 and 59.60
  tok/s for oMLX. Still a tie.
- Flash Next first text at 1,492 input tokens: 0.68 to 0.71 s (median
  0.692) against 0.70 s (0.698 to 0.702); at 127 and 128: 0.20 to 0.23
  against 0.34 to 0.38 s.
- 27B direct greedy: 35.46, 35.47 and 35.46 against 34.68, 34.68 and 34.69
  tok/s for oMLX. First text at 127 and 128 input tokens 0.15 to 0.17 s
  against 0.22 s. At 1,492 input tokens the six AX Engine probes read 1.487
  to 1.491 s and the six oMLX probes 1.510 s (1.5096 to 1.5104).
- 27B DFlash2: 113.40, 113.40 and 113.52 against 108.95, 108.93 and 108.77
  tok/s for Splash. First text at 127 and 128 input tokens 0.15 to 0.17
  against 0.17 s; at 1,492 input tokens 1.49 to 1.50 against 1.19 to 1.32 s
  (medians 1.494 and 1.200).

### Flash Next first text by input length

[first-text-by-length.jsonl](first-text-by-length.jsonl) holds a scan of
cold first text over twelve input lengths from 36 to 13,023 tokens.
AX Engine was built from commit `8108ac93c` with default settings, which
run the routed expert projections of a prefill in
[64-row tiles](../../../docs/performance/flash-next-moe-gather-tiles-2026-11-26.md)
(rerun 11; only this file was measured again). Earlier scans are in the
file's history and in the notes on
[the tensor-unit sparse attention](../../../docs/performance/flash-next-qsa-tensor-unit-2026-11-26.md)
(commit `7a8767085`, rerun 10) and on
[the masked dense call](../../../docs/performance/flash-next-qsa-masked-dense-2026-11-25.md)
(commits `02ab4a08a` and `c688907a6`).

- Each server ran alone and the three ran one after another: AX Engine,
  oMLX, AX Engine (`run_order` 1 to 3), with 150 seconds of rest before
  each (`rest_seconds_before_run`).
- A server takes three warmup requests, then three completions per length
  with `max_tokens` 1 and `temperature` 0. The prompts of one length
  differ in their first line, so no request can reuse a stored prefix.
  `first_text_seconds` lists the three latencies.
- AX Engine ran with MTP and n-gram acceleration disabled, oMLX with its
  prompt cache off; both served one request at a time.
- The README table and chart show, per engine and length, the mean over
  that engine's runs of the run median.

This scan and the two probes of the paired rows use different request
wrappers, so their 127 and 1,492-token figures are close but not the same
measurement.

### First text after the server sat idle

[first-text-after-idle.jsonl](first-text-after-idle.jsonl) holds the first
text of one request after 0, 3, 10, 30 and 60 seconds without any request,
for AX Engine with its residency refresh on and off and for oMLX
([what changed](../../../docs/performance/idle-residency-refresh-2026-11-26.md)).

- One server at a time, one run per configuration, three warmup requests,
  then seven completions with `max_tokens` 1 and `temperature` 0 whose
  prompts differ in their first line. `idle_seconds_before_request` is the
  sleep before each of the seven; `first_text_ms` lists them in order.
- The M5 Max rows use the 1,492-token prompt of the by-length scan, the
  M4 Pro rows its 413-token prompt. Servers followed one another after 15
  to 20 seconds, not 150: the comparison is between the requests of one
  server, and the back-to-back requests of all servers of a model agree.
- AX Engine was built from the source of commit `b39f5dc1`, which adds the
  refresh.

## Replay

Stage the binaries and packs using the
[original matrix instructions](../qwen38-m5-peer-20261125/README.md#replay).
Use fresh scratch directories and the admitted 27B
[local DFlash2 checkpoint](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4).
Run the two models one after the other and use the fixed corpus at the
stage's `qwen38_dflash2_paired.jsonl` path.

```bash
export AX_BENCH_MLX_LIB=/path/to/site-packages/mlx/lib
campaign=benchmarks/results/qwen38-m5-modes-20261125
for model in dense flash; do
  python "$campaign/run.py" --model "$model" \
    --stage /path/to/stage --draft /path/to/27b-dflash2 \
    --baseline-runner benchmarks/results/qwen38-m5-peer-20261125/run.py
done
python "$campaign/render_charts.py" --check
```

`render_charts.py` needs matplotlib 3.11.2 to reproduce the SVG files byte
for byte. It draws with the shared [chart style](chart_style.py) and the
engine colours of [`scripts/chart_palette.py`](../../../scripts/chart_palette.py).
