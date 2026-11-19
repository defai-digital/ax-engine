# Qwen 3.8 27B MXFP4 MTP timing controls

## Short-answer negative control

The M5 Max short-answer control used the pinned [MXFP4 snapshot](../2026-11-18-qwen38-27b-mxfp4-inventory/),
greedy decoding, and a generation budget of 32 with native EOS handling
enabled. Direct and required MTP both emitted the same two token IDs in all
five measured repetitions per route. Each route ran two warmups. The bench
disabled prefix reuse, DFlash2 and n-gram stacking.

| Route | Token-weighted decode (tok/s) | Median trial decode (tok/s) | Median client total (ms) |
| --- | ---: | ---: | ---: |
| Direct | 34.109 | 34.105 | 200.0 |
| Required MTP | 16.675 | 16.803 | 233.9 |
| MTP / direct | 0.489× | 0.493× | 1.170× |

The [timing receipt](m5-short-answer-timing.json) retains emitted IDs,
runner timings, client timing, native route counters, and SHA-256 digests
for the source inputs. Token weighting divides the full generated-token
count by the sum of the reported runner decode intervals. The trial median
uses the five reported decode throughputs for each route. Telemetry counts
speculative work and the output IDs describe the two visible tokens.

## Agent-coding timing

The same M5 Max and runtime measured 15 [coding prompts](agent-coding.jsonl)
with two warmups and five measured trials per route and prompt. Each trial
emitted all 512 tokens. The [summary](m5-agent-coding-timing.json) binds the
build and pack. The [samples](m5-agent-coding-trials.jsonl) retain all 150
measured intervals and the exact input and output sequences.

| Metric | Direct | Required MTP | MTP / direct |
| --- | ---: | ---: | ---: |
| Generated tokens | 38,400 | 38,400 | |
| Token-weighted decode (tok/s) | 32.198 | 56.757 | 1.763× |
| Ratio of median trial decode rates | | | 1.752× |
| Median paired per-prompt speed ratio | | | 1.753× |

The two routes emitted identical sequences on nine of 15 prompts. Each
route repeated its sequence exactly in all five trials per prompt. The
run exceeded the 1.20× weighted and 1.10× median throughput thresholds.
The benchmark's stability check failed on 15 of 30 prompt-route rows,
with 12 high-variance rows and three tail regressions. The three tail
regressions were on the required-MTP route.

The long-form general profile and paired logit replay are pending.
MTP-P and MTP-D remain open.

## Reproduction

Run from an AX Engine checkout at `21a0c4a9e` with the Python dependencies
listed in the timing receipt. Set `MODEL_DIR` to the pinned snapshot and
`SUITE` to the absolute path of [short-answer.jsonl](short-answer.jsonl).
The harness labels the optimized build `release`. The recorded server hash
matches the `release-server` build.

```bash
rustup run 1.99.1 cargo build --locked -p ax-engine-server --profile release-server

PYTHONPATH=scripts python -c '
import os
from pathlib import Path
import bench_mlx_inference_stack as bench

for key in tuple(os.environ):
    if key.startswith(("AX_", "MLX_")):
        del os.environ[key]
os.environ["AX_MLX_DFLASH2"] = "0"
bench.AX_ENGINE_SERVER = Path("target/release-server/ax-engine-server").resolve()
bench.main()
' --model qwen3.8-27b \
  --model-repo-id AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP \
  --model-dir "$MODEL_DIR" --prompt-source real --real-prompt-suite "$SUITE" \
  --no-thinking --generation-tokens 32 --seed 0 --prefill-step-size 2048 \
  --warmup-repetitions 2 --repetitions 5 --cooldown 0 --skip-mlx-lm \
  --no-build-ax-engine --ax-compare-policies --capture-output-token-ids \
  --ax-mtp-policy required --ax-mtp-disable-ngram-stacking \
  --mlx-numerics exact --axengine-port 31441 \
  --ax-sampling '{"ignore_eos":false}' --output short-answer-ab.json
```

For the coding profile, use `agent-coding.jsonl`, set `--generation-tokens
512`, and omit `--ax-sampling` to use the harness's default fixed-budget
behavior. Write the result to `agent-coding-ab.json`.
