#!/bin/zsh
export PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
B=~/build; O=$B/retest; mkdir -p $O; P=$O/progress.txt; echo "restart-B $(date)" >> $P
DEV=$B/ax-qual-20261023; REL=$B/ax-rel-790; PY=$DEV/.venv/bin/python
SIX=$B/packs/ax-qwen38-flash-next-6bit-3c1dd547; PACK=$B/packs/ax-qwen38-flash-next-mxfp4-ff2a2848
# ---- B. release 7.9.0 on MXFP4 ----
cd $REL; RB=$REL/target/release
E=$O/rel790-gates; INV=$REL/benchmarks/results/qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json
CACHE=$B/qual-hf-cache-790; rm -rf $CACHE; mkdir -p $CACHE $E
echo "B1 delivery $(date)" >> $P
n=0
while [ $n -lt 6 ]; do
  n=$((n+1)); prev=(); [ -d $E/delivery ] && { mv $E/delivery $E/delivery-attempt-$n; prev=(--previous-attempt $E/delivery-attempt-$n); }
  $PY scripts/run_flash_next_delivery.py --ax-engine $RB/ax-engine --python $PY --helper $REL/scripts/download_model.py --bench-bin $RB/ax-engine-bench --cache-root $CACHE --inventory $INV --server-bin $RB/ax-engine-server --output $E/delivery $prev --transport-env AX_ENGINE_HF_MAX_WORKERS=2 --transport-env HF_HUB_DOWNLOAD_TIMEOUT=60 > $O/rel790-delivery-$n.log 2>&1
  grep -q '"completed": true' $E/delivery/result.json 2>/dev/null && break
done
echo "B2 qa $(date)" >> $P; $PY scripts/run_flash_next_qa_cohort.py --model-dir $PACK --inventory $INV --server-bin $RB/ax-engine-server --output $E/qa --modes disabled default required > $O/rel790-qa.log 2>&1
echo "B3 lifecycle $(date)" >> $P; $PY scripts/run_flash_next_lifecycle.py --model-dir $PACK --inventory $INV --server-bin $RB/ax-engine-server --output $E/lifecycle --modes disabled default required > $O/rel790-lifecycle.log 2>&1
echo "B4 matrix $(date)" >> $P; $PY scripts/run_flash_next_generate_matrix.py --model-dir $PACK --inventory $INV --server-bin $RB/ax-engine-server --output $E/generate-matrix --modes disabled default > $O/rel790-matrix.log 2>&1
echo "B5 reference $(date)" >> $P; $B/flash-next-official-20260923/launch-reference-env/bin/python scripts/run_flash_next_reference.py --model-dir $PACK --inventory $INV --ax-matrix $E/generate-matrix/result.json --output $E/reference > $O/rel790-reference.log 2>&1
echo "B6 verdict $(date)" >> $P; $PY scripts/check_flash_next_release_gates.py --evidence-dir $E --output $E/release-verdict.json > $O/rel790-verdict.txt 2>&1
rm -rf $CACHE
echo "B7 long-context $(date)" >> $P
for arm in 1 2; do sudo -n purge; sleep 3; FN_AB_TOTAL_BLOCKS=4096 FN_AB_MAX_TOKENS=128 /usr/bin/python3 /tmp/fn_ab.py $RB/ax-engine-server rel790lc$arm $O/rel790-longctx-$arm.json $B/fn-ab/prompts/p1.txt $B/fn-ab/prompts/long1.txt $B/fn-ab/prompts/long32.txt > $O/rel790-longctx-$arm.out 2>&1; done
echo "B8 flappy $(date)" >> $P
$PY scripts/bench_flash_next_peer.py --model-dir $PACK --output-dir $O/rel790-flappy --python $PY --lanes ax > $O/rel790-flappy.out 2>&1
$PY scripts/bench_mlx_inference_stack.py --model-repo-id AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP --model-dir $PACK --prompt-source real --real-prompt-suite benchmarks/prompts/mtp-suites/flappy.jsonl --no-thinking --generation-tokens 256 --repetitions 5 --warmup-repetitions 2 --cooldown 3 --prefill-step-size 2048 --skip-mlx-lm --no-build-ax-engine --ax-direct --output $O/rel790-flappy-direct.json > $O/rel790-flappy-direct.out 2>&1
echo ALLDONE >> $P
