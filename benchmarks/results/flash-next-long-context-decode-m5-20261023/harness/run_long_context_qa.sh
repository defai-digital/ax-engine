#!/bin/zsh
export PATH=/opt/homebrew/bin:/usr/sbin:/usr/bin:/bin:/sbin
cd ~/build/fn-ab
BIN=$HOME/build/fn-ab/ax-engine-server-t7
Q=/usr/bin/python3
run() { # label scenario envs... -- files
  :
}
for arm in off def; do
  if [ $arm = off ]; then export AX_FLASH_NEXT_KV_SEGMENTS=0; else unset AX_FLASH_NEXT_KV_SEGMENTS; fi
  echo "== mtp canonical $arm"; QA_MTP_POLICY=required QA_TOTAL_BLOCKS=4096 $Q /tmp/fn_qa2.py $BIN t8-mtp-$arm texts prompts/p3.txt prompts/long1.txt prompts/long32.txt > qa-t8-mtp-$arm.json 2> qa-t8-mtp-$arm.err
  echo "== mtp batched $arm"; QA_MTP_POLICY=required AX_FLASH_NEXT_MTP_VERIFIER=batched QA_TOTAL_BLOCKS=4096 $Q /tmp/fn_qa2.py $BIN t8-mtpb-$arm texts prompts/p3.txt prompts/long1.txt prompts/long32.txt > qa-t8-mtpb-$arm.json 2> qa-t8-mtpb-$arm.err
  echo "== cancel $arm"; QA_TOTAL_BLOCKS=4096 $Q /tmp/fn_qa2.py $BIN t8-cancel-$arm cancel prompts/p1.txt prompts/long32.txt > qa-t8-cancel-$arm.json 2> qa-t8-cancel-$arm.err
done
unset AX_FLASH_NEXT_KV_SEGMENTS
echo "== soak 9k default"; QA_SOAK_N=16 QA_TOTAL_BLOCKS=4096 $Q /tmp/fn_qa2.py $BIN t8-soak soak prompts/long1.txt > qa-t8-soak.json 2> qa-t8-soak.err
echo ALLDONE > t8_done.txt
