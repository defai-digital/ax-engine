# Qwen 3.8 27B AXQ 6-bit: peer campaign on the qualification SKU

Mac mini M4 Pro, 64 GB (Mac16,11), macOS 26.6.2: the selected qualification
SKU. Pack: `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP` @
`3e290738e96972307c6aeb9934ab170ca0eae1c1`, the same directory offered to
every runtime. AX Engine is the installed bundled wheel built from clean
`ad999f3f` (wheel SHA-256 `2bc3f5e1...9784409`), which includes the
target-head and low-precision SwiGLU corrections.

Contract: repository `flappy` suite, four cases (prompt lengths 264-432
tokens), 256 generated tokens, greedy, chat template with thinking disabled,
two warmups, five measured repetitions, three-second cooldown, prefix cache
off, n-gram stacking off. Decode and prefill are the **median of 20 measured
runs**.

## Measured

| Runtime | Version (latest checked 2026-09-17) | Decode | Prefill |
| --- | --- | ---: | ---: |
| AX Engine | 7.4.0 product-path MTP (recurrent depth 3) | **31.05 tok/s** | **120.3 tok/s** |
| MTPLX | **2.11.3** (PyPI Latest), sustained profile, depth 3 | 28.16 tok/s | 114.0 tok/s |
| OMLX | **0.6.4** (GitHub Latest release), `import_mtplx_sidecar` then Lightning MTP depth 1 | 15.02 tok/s | - (generate-wall tok/s) |
| mlx-lm | **0.31.3** (PyPI Latest), direct AR on the same prompts | 12.78 tok/s | - (decode-only lane) |

AX prefill is `prefill_tok_s`; MTPLX prefill is `prompt_tokens /
prompt_eval_time_s`. mlx-lm decode is its own `generation_tps` (first generated
token excluded). OMLX reports generate-wall tok/s from the harness trials.

Raw: `ax_engine.json` (recorded run), `ax_engine-first-run.json`, `mtplx.json`,
`omlx.json`, `mlx_lm.json`, `summary.json`.

## Conditions

- The AX lane was run twice. The first run overlapped package installation on
  the host (1-minute load 5.9); the recorded run repeated the lane afterwards
  at load 2.0. The two agree within 0.4% on decode (30.95 vs 31.05) and 0.4%
  on prefill (119.8 vs 120.3).
- Two system daemons (`audiomxd`, `configd`) held about 1.2 cores of CPU for
  the whole campaign. Every lane ran under that condition; it was not present
  on the 2026-09-15 M5 Max campaign host.
- The AX run passes the harness stability and MTP-correctness publication
  gates (`stable_enough` on every row, `mtp_head_only_effective` claim status,
  recorded clean build commit).
- OMLX cannot load the raw Hub tree. The import used
  `omlx.oq.import_mtplx_sidecar` on a writable symlink snapshot with the
  campaign `mtplx_runtime.json` and `axquant_omlx_compat.json` carried over
  from the 2026-09-15 import; the Hub revision's minimal `mtplx_runtime.json`
  has no `arch_id` and fails OMLX's contract check.
- MTPLX 2.11.3 is the PyPI release current on 2026-09-17; the M5 campaign
  measured 2.11.2.

## Reproduction

From the repository checkout on the SKU, with the pinned pack at `$PACK` and
the installed wheel's `ax-engine-server` linked at
`target/release/ax-engine-server`:

```sh
python scripts/bench_mlx_inference_stack.py \
  --model-repo-id AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP --model-dir "$PACK" \
  --prompt-source real --real-prompt-suite benchmarks/prompts/mtp-suites/flappy.jsonl \
  --no-thinking --generation-tokens 256 --repetitions 5 --warmup-repetitions 2 \
  --cooldown 3 --prefill-step-size 2048 --skip-mlx-lm --no-build-ax-engine \
  --ax-ngram-accel --ax-mtp-disable-ngram-stacking --ax-qwen-linear-mtp-exact \
  --output ax_engine.json

python scripts/bench_mtplx_prompt_suites.py --model "$PACK" --suite flappy \
  --prompts benchmarks/prompts/mtp-suites/flappy.jsonl --output mtplx.json \
  --profile sustained --depth 3 --temperature 0 --top-p 1 --top-k 0 \
  --max-tokens 256 --repetitions 5 --warmup-repetitions 2 --cooldown 3 \
  --ignore-eos --disable-thinking --allow-unverified-model

python scripts/bench_omlx_prompt_suites.py --model "$IMPORTED_SNAPSHOT" \
  --model-repo-id AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP --suite flappy \
  --prompts benchmarks/prompts/mtp-suites/flappy.jsonl --output omlx.json \
  --max-tokens 256 --repetitions 5 --warmup-repetitions 2 --cooldown 3 \
  --temperature 0 --top-p 1 --top-k 0 --seed 0

python scripts/bench_mlx_lm_prompt_suites.py --model "$PACK" \
  --prompts benchmarks/prompts/mtp-suites/flappy.jsonl --output mlx_lm.json \
  --max-tokens 256 --repetitions 5 --warmup-repetitions 2 --cooldown 3
```

The 20-run AX medians are taken from the per-repetition harness log lines;
`ax_engine.json` keeps per-case median/min/max at full precision.
