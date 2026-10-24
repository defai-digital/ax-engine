#!/bin/zsh
export PATH=/opt/homebrew/bin:/usr/sbin:/usr/bin:/bin:/sbin
cd ~/build/fn-ab
for round in 1 2; do
  for arm in off on; do
    if [ $arm = on ]; then export AX_MLX_FLASH_NEXT_GDN_METAL=1; else unset AX_MLX_FLASH_NEXT_GDN_METAL; fi
    sudo -n purge; sleep 3
    FN_AB_TOTAL_BLOCKS=4096 FN_AB_MAX_TOKENS=128 /usr/bin/python3 /tmp/fn_ab.py $PWD/ax-engine-server-t7 gm$arm$round out-gm-$arm-$round.json prompts/p1.txt prompts/p3.txt prompts/long1.txt > gm-$arm-$round.out 2>&1
  done
done
echo ALLDONE > gm_done.txt
