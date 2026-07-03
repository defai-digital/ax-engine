#!/usr/bin/env bash
# Same-session A/B: pre-fix (332c42e5) vs with-fix (main) binaries, measured
# back-to-back so thermal state is shared. Isolates the effect of the hot-path
# error-capture change with no cross-session confound.
set -uo pipefail
cd /Users/akiralam/code/ax-engine

OUT=benchmarks/results/mlx-inference/2026-06-24-ab-samesession
mkdir -p "$OUT/logs"
HUB=/Volumes/Ext4T/models/hub
BIN=target/release/ax-engine-server

run() {  # run MODEL VARIANT ROUND
  local m="$1" variant="$2" round="$3"
  local snap; snap=$(ls -d "$HUB/models--mlx-community--$m/snapshots/"*/ 2>/dev/null | head -1)
  cp "/tmp/ax-$variant" "$BIN"
  python3 scripts/bench_mlx_inference_stack.py \
    --model-repo-id "mlx-community/$m" --model-dir "$snap" \
    --prompt-tokens 128,512,2048 --generation-tokens 128 \
    --repetitions 4 --cooldown 8 \
    --ax-direct --skip-mlx-lm --no-build-ax-engine \
    --output "$OUT/${m}__${variant}__r${round}.json" > "$OUT/logs/${m}__${variant}__r${round}.log" 2>&1
  pkill -f ax-engine-server 2>/dev/null; sleep 5
}

echo "=== settle thermals before A/B  $(date +%H:%M:%S) ==="
sleep 150

# MoE signal models — 2 interleaved rounds each (prefix, withfix, prefix, withfix)
for m in gemma-4-26b-a4b-it-4bit Qwen3.6-27B-4bit; do
  for r in 1 2; do
    echo "===== $m prefix  r$r  $(date +%H:%M:%S) ====="; run "$m" prefix  "$r"
    echo "===== $m withfix r$r  $(date +%H:%M:%S) ====="; run "$m" withfix "$r"
  done
done

# fast control — one round
m=gemma-4-e2b-it-4bit
echo "===== $m prefix  r1  $(date +%H:%M:%S) ====="; run "$m" prefix 1
echo "===== $m withfix r1  $(date +%H:%M:%S) ====="; run "$m" withfix 1

cp /tmp/ax-withfix "$BIN"   # restore canonical fixed binary
echo "===== A/B DONE $(date) ====="
