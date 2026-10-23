#!/bin/zsh
export PATH=/opt/homebrew/bin:/usr/sbin:/usr/bin:/bin:/sbin
cd ~/build/fn-ab
for round in 1 2; do
  for arm in prev new; do
    bin=ax-engine-server-t11; [ $arm = prev ] && bin=ax-engine-server-t10
    sudo -n purge; sleep 3
    FN_AB_TOTAL_BLOCKS=4096 FN_AB_MAX_TOKENS=128 /usr/bin/python3 /tmp/fn_ab.py $PWD/$bin t11$arm$round out-t11-$arm-$round.json prompts/p1.txt prompts/long1.txt prompts/long32.txt > t11-$arm-$round.out 2>&1
  done
done
echo ALLDONE > t11_done.txt
