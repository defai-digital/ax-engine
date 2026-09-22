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
echo "OPT_START $(date -u +%FT%TZ)"
until ! pgrep -q -f killswitch_lanes.sh; do sleep 15; done
lane flappy_pack_gate_up --ax-pack-dense-ffn-gate-up
AX_MLX_LINEAR_ATTENTION_WHOLE_LAYER_METAL=1 lane flappy_whole_layer_metal
lane flappy_head3
echo "OPT_DONE $(date -u +%FT%TZ)"
