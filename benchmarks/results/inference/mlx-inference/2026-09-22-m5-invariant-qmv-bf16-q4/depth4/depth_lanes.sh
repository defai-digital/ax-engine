#!/bin/zsh
set -uo pipefail
cd /Users/akiralam/code/ax-engine
export PATH=/opt/homebrew/bin:/usr/bin:/bin
OUT=/Users/akiralam/bench-ab-20260922/final_depth2
mkdir -p $OUT
PACK=/Users/akiralam/.cache/huggingface/hub/models--AutomatosX--AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP/snapshots/3e290738e96972307c6aeb9934ab170ca0eae1c1
until grep -q FINAL_DONE /tmp/final_lanes.log 2>/dev/null; do sleep 20; done
git checkout -q -- . && git apply /tmp/final_depth2.patch && cargo build --release --bin ax-engine-server 2>&1 | tail -1 && cp target/release/ax-engine-server /tmp/ax-engine-server.final_depth2
lane() {
  local suite=$1 depth=$2
  AX_MLX_QWEN_LINEAR_THROUGHPUT_MTP_DEPTH=$depth .venv/bin/python3 scripts/bench_mlx_inference_stack.py --model-dir "$PACK" --model-repo-id AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP --prompt-source real --real-prompt-suite benchmarks/prompts/mtp-suites/${suite}.jsonl --generation-tokens 256 --repetitions 5 --warmup-repetitions 2 --cooldown 3 --ax-ngram-accel --ax-mtp-disable-ngram-stacking --ax-qwen-linear-mtp-exact --skip-mlx-lm --no-build-ax-engine --no-thinking --output $OUT/${suite}_depth${depth}.json
  echo "${suite}_depth${depth} rc=$?"
}
echo "FINAL2_START $(date -u +%FT%TZ)"
lane flappy 4
lane long_code 4
lane python_modules_long 4
echo "FINAL2_DONE $(date -u +%FT%TZ)"
