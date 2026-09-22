#!/bin/zsh
set -uo pipefail
cd /Users/akiralam/code/ax-engine
export PATH=/opt/homebrew/bin:/usr/bin:/bin
OUT=/Users/akiralam/bench-ab-20260922/profile10
mkdir -p $OUT
PACK=/Users/akiralam/.cache/huggingface/hub/models--AutomatosX--AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP/snapshots/3e290738e96972307c6aeb9934ab170ca0eae1c1
BASE=$(shasum -a 256 /tmp/ax-engine-server.c5d22c4a | cut -c1-16)
until [ "$(shasum -a 256 target/release/ax-engine-server | cut -c1-16)" != "$BASE" ] && ! pgrep -q -f "cargo build"; do sleep 20; done
sleep 5
lane() {
  local name=$1; shift
  .venv/bin/python3 scripts/bench_mlx_inference_stack.py --model-dir "$PACK" --model-repo-id AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP --prompt-source real --real-prompt-suite benchmarks/prompts/mtp-suites/flappy.jsonl --generation-tokens 256 --warmup-repetitions 1 --cooldown 3 --ax-ngram-accel --ax-mtp-disable-ngram-stacking --ax-qwen-linear-mtp-exact --skip-mlx-lm --no-build-ax-engine --no-thinking --output $OUT/${name}.json "$@"
  echo "${name} rc=$?"
}
echo "PROFILE_START $(date -u +%FT%TZ)"
lane flappy_head --repetitions 3
lane flappy_decode_profile --repetitions 2 --ax-decode-profile
lane flappy_la_profile --repetitions 2 --ax-linear-attention-profile
echo "PROFILE_DONE $(date -u +%FT%TZ)"
