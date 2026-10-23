#!/bin/zsh
# Fair peer campaign v2 (Qwen3.6-35B-A3B AXQ 6-bit MTP). One pass = the listed lanes in order.
# usage: fair_campaign.sh <pass-label> <suites,comma> <lanes,comma>
#   lanes: ax_tp ax_exact ax_direct ax_d1 ax_d2 mtplx_d1 mtplx_d2 mtplx_d3 omlx_d1 omlx_d2 omlx_d3 mlx_lm
export PATH=/opt/homebrew/bin:/usr/sbin:/usr/bin:/bin:/sbin
B=$HOME/build/peer-bench-20261022; SRC=$HOME/build/ax-win-20261022/src
OUT=$B/out2; mkdir -p $OUT
STOCK=$HOME/build/packs/ax-qwen36-35b-a3b-axq-6bit-mtp-6a4c2207
OPACK=$STOCK-omlx-axq
REPO=AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP
PASS=$1; SUITES=(${(s:,:)2}); LANES=(${(s:,:)3})
AXPY=$SRC/.venv/bin/python
MTPPY=/opt/homebrew/var/mtplx/venv-2.12.2/bin/python
OMLXPY=/opt/homebrew/Cellar/omlx/0.7.2/libexec/bin/python3.11
MLXLMPY=$B/venv-mlx-lm/bin/python
PORT=31520

cpu_idle() { top -l 2 -s 1 -n 0 2>/dev/null | grep "CPU usage" | tail -1 | sed -E 's/.* ([0-9.]+)% idle.*/\1/'; }
gate() { # wait until the CPU has been >= 94% idle on three consecutive samples (cap 10 min)
  local good=0 waited=0
  while (( waited < 600 )); do
    idle=$(cpu_idle)
    if (( ${idle%.*} >= 94 )); then (( good++ )); else good=0; fi
    (( good >= 3 )) && break
    sleep 2; (( waited += 4 ))
  done
  echo "GATE idle=${idle}% waited=${waited}s good=${good} $(pmset -g batt | head -1 | sed -E 's/.*drawing from//') $(date -u +%FT%TZ)"
}

for suite in $SUITES; do
  P=$SRC/benchmarks/prompts/mtp-suites/$suite.jsonl
  for lane in $LANES; do
    gate
    O=$OUT/q36_35b_${suite}_${lane}_${PASS}.json
    echo "LANE_START $PASS $suite $lane $(date -u +%FT%TZ)"
    case $lane in
      ax_tp|ax_exact|ax_direct|ax_d1|ax_d2)
        flags=(--ax-ngram-accel --ax-mtp-policy required --ax-mtp-disable-ngram-stacking); envs=(AX_NOP=1)
        [[ $lane == ax_exact ]] && flags=(--ax-ngram-accel --ax-mtp-disable-ngram-stacking --ax-qwen-linear-mtp-exact)
        [[ $lane == ax_direct ]] && flags=(--ax-direct)
        [[ $lane == ax_d1 ]] && envs=(AX_MLX_MTP_FIXED_DRAFT_DEPTH=1)
        [[ $lane == ax_d2 ]] && envs=(AX_MLX_MTP_FIXED_DRAFT_DEPTH=2)
        (cd $SRC && env ${envs[@]} $AXPY scripts/bench_mlx_inference_stack.py --model-dir $STOCK --model-repo-id $REPO --prompt-source real --real-prompt-suite $P --generation-tokens 256 --repetitions 5 --warmup-repetitions 2 --cooldown 3 ${flags[@]} --skip-mlx-lm --no-build-ax-engine --no-thinking --axengine-port $PORT --output $O) > ${O%.json}.log 2>&1 ;;
      mtplx_d*)
        d=${lane#mtplx_d}
        (cd $SRC && $MTPPY scripts/bench_mtplx_prompt_suites.py --model $STOCK --suite $suite --prompts $P --profile sustained --depth $d --max-tokens 256 --repetitions 5 --warmup-repetitions 2 --cooldown 3 --temperature 0 --top-p 1.0 --top-k 0 --ignore-eos --allow-unverified-model --disable-thinking --output $O) > ${O%.json}.log 2>&1 ;;
      omlx_d*)
        d=${lane#omlx_d}
        (cd $SRC && $OMLXPY $B/bench_omlx_prefill.py --model $OPACK --model-repo-id $REPO --suite $suite --prompts $P --host-label apple-m5-max-128gb --max-tokens 256 --repetitions 5 --warmup-repetitions 2 --cooldown 3 --temperature 0 --top-p 1.0 --top-k 0 --seed 0 --mtp-draft-tokens $d --output $O) > ${O%.json}.log 2>&1 ;;
      mlx_lm)
        (cd $SRC && $MLXLMPY $B/bench_mlx_lm_prefill.py --pack $STOCK --prompts $P --max-tokens 256 --warmup 2 --reps 5 --cooldown 3 --output $O) > ${O%.json}.log 2>&1 ;;
    esac
    echo "LANE_DONE $PASS $suite $lane rc=$? $(date -u +%FT%TZ)"
    pkill -f "$SRC/target/release/ax-engine-server" 2>/dev/null; sleep 2
  done
done
echo "PASS_DONE $PASS $(date -u +%FT%TZ)"
