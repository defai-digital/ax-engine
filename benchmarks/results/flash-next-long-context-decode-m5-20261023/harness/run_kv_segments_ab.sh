#!/bin/zsh
export PATH=/opt/homebrew/bin:/usr/sbin:/usr/bin:/bin:/sbin
cd ~/build/fn-ab
for round in 1 2; do
  for arm in off def; do
    if [ $arm = off ]; then export AX_FLASH_NEXT_KV_SEGMENTS=0; else unset AX_FLASH_NEXT_KV_SEGMENTS; fi
    sudo -n purge; sleep 3
    FN_AB_TOTAL_BLOCKS=4096 FN_AB_MAX_TOKENS=128 /usr/bin/python3 /tmp/fn_ab.py $PWD/ax-engine-server-t7 t7$arm$round out-t7-$arm-$round.json prompts/p1.txt prompts/long1.txt prompts/long32.txt > t7-$arm-$round.out 2>&1
  done
done
echo ALLDONE > t7_done.txt
