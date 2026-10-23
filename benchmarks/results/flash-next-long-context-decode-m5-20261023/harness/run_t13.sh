#!/bin/zsh
export PATH=/opt/homebrew/bin:/usr/sbin:/usr/bin:/bin:/sbin
cd ~/build/fn-ab
for round in 1 2 3; do
  for arm in prev new; do
    bin=ax-engine-server-t13; [ $arm = prev ] && bin=ax-engine-server-t12
    sudo -n purge; sleep 3
    FN_AB_TOTAL_BLOCKS=4096 FN_AB_MAX_TOKENS=128 /usr/bin/python3 /tmp/fn_ab.py $PWD/$bin t13$arm$round out-t13-$arm-$round.json prompts/p1.txt prompts/p3.txt prompts/long1.txt > t13-$arm-$round.out 2>&1
  done
done
echo ALLDONE > t13_done.txt
