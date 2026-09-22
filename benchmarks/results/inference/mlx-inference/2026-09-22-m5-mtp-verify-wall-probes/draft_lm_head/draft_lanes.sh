#!/bin/zsh
set -uo pipefail
cd /Users/akiralam/code/ax-engine
export PATH=/opt/homebrew/bin:/usr/bin:/bin
OUT=/Users/akiralam/bench-ab-20260922/profile10
PACK=/Users/akiralam/.cache/huggingface/hub/models--AutomatosX--AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP/snapshots/3e290738e96972307c6aeb9934ab170ca0eae1c1
lane() {
  local name=$1; shift
  .venv/bin/python3 scripts/bench_mlx_inference_stack.py --model-dir "$PACK" --model-repo-id AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP --prompt-source real --real-prompt-suite benchmarks/prompts/mtp-suites/flappy.jsonl --generation-tokens 256 --repetitions 3 --warmup-repetitions 1 --cooldown 3 --ax-ngram-accel --ax-mtp-disable-ngram-stacking --ax-qwen-linear-mtp-exact --skip-mlx-lm --no-build-ax-engine --no-thinking --output $OUT/${name}.json "$@"
  echo "${name} rc=$?"
}
until grep -q STOCK_DONE /tmp/stock_lanes.log 2>/dev/null; do sleep 20; done
sleep 5
echo "DRAFT_START $(date -u +%FT%TZ)"
AX_MLX_MTP_DRAFT_LM_HEAD_BITS=3 AX_MLX_MTP_DRAFT_LM_HEAD_GROUP_SIZE=64 lane flappy_draft_head_q3g64
AX_MLX_MTP_DRAFT_LM_HEAD_BITS=4 AX_MLX_MTP_DRAFT_LM_HEAD_GROUP_SIZE=128 lane flappy_draft_head_q4g128
echo "DRAFT_DONE $(date -u +%FT%TZ)"
