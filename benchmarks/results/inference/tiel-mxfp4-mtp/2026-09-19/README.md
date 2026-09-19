# Tiel MTP depth diagnostic on M5 Max

This is a bounded experiment for the optional
[`AX_MLX_MTP_CONSERVATIVE_DEPTH`](../../../../../docs/mtp/tiel-adaptive-depth.md)
controller. It does not qualify MTP-S/P/D, change automatic MTP admission, or
establish a general coding speedup. Both packs ran on a MacBook Pro with Apple
M5 Max and 128 GiB memory. No model execution used an M3 Mac Studio.

## Method and evidence

`trials.json` retains all measured trial timings, output token IDs, MTP counters,
prompt token artifacts, model revisions and build digests. It is a projection of
the original harness artifacts: internal paths and unrelated telemetry are
omitted, and original artifact SHA-256 values are retained. `verify.py` checks
the workload contract and recomputes the table from individual trials.

- One optimized `release-server` binary, built with Rust 1.97.1 and MLX 0.32.2,
  is used for both option states. `build.source_files_sha256` binds the tested
  source files, and `build.server_sha256` binds the binary copied to the M5.
  The isolated host directory has no Git checkout; this independent manifest
  supplies identity instead of the harness's unresolved checkout/RPATH fields.
- `scripts/bench_mlx_inference_stack.py` runs one server at a time. Each cell
  has two warmups, five measured trials, a one-second cooldown, disabled prefix
  caching, greedy generation and no n-gram stacking. MTP is explicitly
  `required`. Off and on cells run sequentially; they are not randomized.
- Random-token input is p128/g128 using the exact prompt IDs exported by
  `mlx_lm.benchmark`. The primary reference section contains three-trial
  MLX-LM, native direct and native MTP controls on baseline commit `0c94d610`.
  Those controls precede the final A/B and are not simultaneous measurements.
- Two real coding inputs use each model's official chat template with thinking
  disabled and a fixed 256-token output budget. Prompt lengths differ between
  packs. The workload file and rendered token IDs are retained. MLX-LM's
  benchmark cannot consume this external prompt suite, so it is only a
  primary reference for the random-token diagnostic.
- Reported decode throughput uses the harness's post-prefill interval and
  fixed output count, including the first token, matching its MLX-LM convention.
  This is distinct from whole-request throughput. Prefill, TTFT, client wall
  time and memory readings are also retained. AX trial `peak_memory_gb` is
  server RSS sampled after streaming, as identified by `memory_source`; it is
  not a sampled peak or directly comparable to MLX-LM's allocator peak.

## Results

Run the evidence checker to display the measured medians and activation counts:

```bash
python3 benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-19/verify.py
# Also verify the current source matches the measured build:
python3 benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-19/verify.py --check-source
```

| Model | Input / output tokens | Option off (tok/s) | Option on (tok/s) | Change | Conservative decisions (5 trials) |
| --- | --- | ---: | ---: | ---: | ---: |
| Cyber-Tiel | random, 128 / 128 | 203.50 | 205.14 | +0.81% | 0 |
| Cyber-Tiel | python-lru, 219 / 256 | 245.35 | 245.86 | +0.21% | 0 |
| Cyber-Tiel | rust-jsonl, 1064 / 256 | 208.58 | 209.42 | +0.40% | 0 |
| Tiel | random, 128 / 128 | 133.34 | 141.00 | +5.74% | 200 |
| Tiel | python-lru, 194 / 256 | 216.53 | 215.26 | -0.58% | 0 |
| Tiel | rust-jsonl, 1039 / 256 | 215.52 | 214.12 | -0.65% | 0 |

Only the Tiel random-input diagnostic activates the new controller. Its first
trial submits 139 drafts instead of 174, with 55 versus 57 accepted tokens.
The other cells make zero conservative decisions and show changes below 1%;
these measurements do not establish a coding-workload improvement.

The controller changes verifier window sizes. Tiel's random-input output IDs
differ between option states; the other measured cells retain identical output
IDs. Floating-point near ties can differ between graph shapes, so the random
cell is a throughput diagnostic, not same-output quality or correctness
certification. Existing greedy verifier acceptance, rollback and weights remain
unchanged. All measured cells show native draft and accepted tokens, greedy
correctness mode, and no optimistic steps, n-gram proposals or direct fallback.

## Diagnosis and rejected controls

The previous norm-metadata fix is already in baseline `0c94d610`; it is not a
new improvement here. A development build is also not a valid optimized speed
baseline. After those corrections, native MTP already improves the two coding
workloads relative to direct execution. The remaining measured weakness is
low-yield drafting: the existing throughput controller stays at depth two after
a miss and returns to three after any accepted prefix.

Fixed depth one raised the initial Tiel random diagnostic from 133.29 to 141.5
tok/s, but reduced its two coding cells from 216.83/215.32 to 183.3/173.6 tok/s.
An unconditional conservative controller similarly hurt coding by roughly
2-7%. An explicit Q4 draft-head override did not help: the existing dense BF16
target-head path already prepares a Q4 draft head.

An eight-observation first-position gate improved Tiel random by 7.0%, but
regressed Cyber random by 1.66%. Adding a second-position gate with the same
eight-observation minimum still admitted Cyber during its early low-yield
prefix. Those candidates were rejected. The measured final controller waits
for 32 actual observations at each of the first two positions and requires
acceptance below 75% and 50%, respectively. These empirical thresholds remain
opt-in; cumulative acceptance can react slowly to workload changes, and
shallower drafting changes which later positions are sampled.

Retained control names use `conservative` for the unconditional candidate,
`final` for the eight-sample first-position candidate, `yield` for its
eight-sample two-position successor, and `mature` for the final 32-sample
controller. Intermediate build digests and the negative-control trial
projections are included so those results remain distinguishable.

## Reproduce

Download the exact revisions in `trials.json`, build with
`rustup run 1.97.1 cargo build --profile release-server -p ax-engine-server`,
and run on the same hardware class. The harness expects the executable at
`target/release/ax-engine-server`: place the `release-server` artifact there
and ensure its matching MLX libraries resolve on the test host. Check the
recorded source digests before building. For each model directory and each
option state:

```bash
AX_MLX_MTP_CONSERVATIVE_DEPTH=0 python3 scripts/bench_mlx_inference_stack.py \
  --model "$MODEL_DIR" --model-dir "$MODEL_DIR" --model-repo-id "$MODEL_REPO_ID" \
  --no-build-ax-engine --ax-ngram-accel --ax-mtp-policy required \
  --ax-mtp-disable-ngram-stacking --capture-output-token-ids \
  --prompt-tokens 128 --generation-tokens 128 \
  --warmup-repetitions 2 --repetitions 5 --cooldown 1 --output random-off.json
```

Repeat with the environment value `1` and a distinct output path. For coding,
replace `--prompt-tokens 128 --generation-tokens 128` with
`--prompt-source real --real-prompt-suite benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-19/workloads.jsonl --no-thinking --generation-tokens 256 --skip-mlx-lm`.
Use `--ax-compare-policies` in place of `--ax-ngram-accel` to add native direct
control rows; the harness explicitly uses `disabled` MTP policy for those rows.
