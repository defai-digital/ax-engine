#!/bin/zsh
export PATH=/opt/homebrew/bin:/usr/sbin:/usr/bin:/bin:/sbin
cd ~/build/fn-ab
OLD=hc_norm,hc_low,hc_mix,hc_gate,hc_write,gdn_beta,gdn_post,gdn_conv_act,gdn_decay_pre,gdn_decay_post,gdn_l2_sum,gdn_q_scale,gdn_k_scale,moe_swiglu,moe_combine,qsa_rope_lo,qsa_rope_hi,qsa_gate,moe_route_tail
for round in 1 2 3; do
  for arm in old new; do
    if [ $arm = old ]; then export AX_FLASH_NEXT_FUSE=$OLD; else unset AX_FLASH_NEXT_FUSE; fi
    sudo -n purge; sleep 3
    FN_AB_TOTAL_BLOCKS=4096 FN_AB_MAX_TOKENS=128 /usr/bin/python3 /tmp/fn_ab.py $PWD/ax-engine-server-t13 t13b$arm$round out-t13b-$arm-$round.json prompts/p1.txt prompts/p3.txt > t13b-$arm-$round.out 2>&1
  done
done
echo ALLDONE > t13b_done.txt
