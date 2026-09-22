#!/bin/zsh
# A/B: head (8064e97b) vs patched (bf16/q4 invariant kernel) on the v7.5.3 peer-campaign lane contract.
set -uo pipefail
cd /Users/akiralam/code/ax-engine
export PATH=/opt/homebrew/bin:/usr/bin:/bin
OUT=/Users/akiralam/bench-ab-20260922
mkdir -p $OUT
PACK=/Users/akiralam/.cache/huggingface/hub/models--AutomatosX--AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP/snapshots/3e290738e96972307c6aeb9934ab170ca0eae1c1
ax_lane() {
  local suite=$1 arm=$2
  cp /tmp/ax-engine-server.$arm target/release/ax-engine-server
  .venv/bin/python3 scripts/bench_mlx_inference_stack.py --model-dir "$PACK" --model-repo-id AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP --prompt-source real --real-prompt-suite benchmarks/prompts/mtp-suites/${suite}.jsonl --generation-tokens 256 --repetitions 5 --warmup-repetitions 2 --cooldown 3 --ax-ngram-accel --ax-mtp-disable-ngram-stacking --ax-qwen-linear-mtp-exact --skip-mlx-lm --no-build-ax-engine --no-thinking --output $OUT/${suite}_${arm}.json
  echo "${suite}_${arm} rc=$?"
}
echo "AB_START $(date -u +%FT%TZ)"
ax_lane flappy head
ax_lane flappy patched
ax_lane long_code patched
ax_lane long_code head
echo "AB_DONE $(date -u +%FT%TZ)"
