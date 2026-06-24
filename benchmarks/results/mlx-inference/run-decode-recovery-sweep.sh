#!/usr/bin/env bash
# Full decode/prefill recovery sweep on the fixed (current main) release binary.
# Portable per-model watchdog (macOS has no `timeout`): kills a stuck run and any
# orphaned ax-engine-server so the blocked 35b can't stall the sweep.
set -uo pipefail
cd /Users/akiralam/code/ax-engine

OUT=benchmarks/results/mlx-inference/2026-06-24-full-sweep-fixed
mkdir -p "$OUT/logs"
HUB=/Volumes/Ext4T/models/hub

run_with_timeout() {  # run_with_timeout SECS cmd...
  local secs="$1"; shift
  "$@" &
  local pid=$!
  (
    local waited=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep 10; waited=$((waited+10))
      if [ "$waited" -ge "$secs" ]; then
        kill -TERM "$pid" 2>/dev/null; sleep 5; kill -KILL "$pid" 2>/dev/null
        pkill -f ax-engine-server 2>/dev/null
        break
      fi
    done
  ) &
  local watcher=$!
  wait "$pid"; local rc=$?
  kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null
  return $rc
}

models=(
  gemma-4-e2b-it-4bit
  gemma-4-e2b-it-6bit
  gemma-4-e4b-it-4bit
  gemma-4-26b-a4b-it-4bit
  gemma-4-31b-it-4bit
  Qwen3.6-27B-4bit
  Qwen3.6-27B-6bit
  Qwen3.6-35B-A3B-4bit
)

for m in "${models[@]}"; do
  snap=$(ls -d "$HUB/models--mlx-community--$m/snapshots/"*/ 2>/dev/null | head -1)
  echo "===== $m  start=$(date +%H:%M:%S) ====="
  if [ -z "$snap" ]; then echo "  MISSING snapshot, skip"; continue; fi
  run_with_timeout 1800 python3 scripts/bench_mlx_inference_stack.py \
    --model-repo-id "mlx-community/$m" --model-dir "$snap" \
    --prompt-tokens 128,512,2048 --generation-tokens 128 \
    --repetitions 5 --cooldown 15 \
    --ax-direct --skip-mlx-lm --no-build-ax-engine \
    --output "$OUT/$m.json" > "$OUT/logs/$m.log" 2>&1
  rc=$?
  pkill -f ax-engine-server 2>/dev/null; sleep 2
  if   [ $rc -eq 137 ] || [ $rc -eq 143 ]; then echo "  KILLED by watchdog (stuck?) — logs/$m.log"
  elif [ $rc -ne 0 ];   then echo "  FAILED rc=$rc — logs/$m.log"
  else echo "  ok  end=$(date +%H:%M:%S)"; fi
done
echo "===== SWEEP DONE $(date) ====="
