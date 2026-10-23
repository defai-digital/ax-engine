#!/bin/zsh
export PATH=/opt/homebrew/bin:/usr/sbin:/usr/bin:/bin:/sbin
cd ~/build/fn-ab
OLDFUSE=hc_norm,hc_low,hc_mix,hc_gate,hc_write,gdn_beta,gdn_post,gdn_conv_act,gdn_decay_pre,gdn_decay_post,gdn_l2_sum,gdn_q_scale,gdn_k_scale,moe_swiglu,moe_combine
for round in 1 2; do
  for arm in prev new newoff; do
    unset AX_FLASH_NEXT_FUSE AX_FLASH_NEXT_POOLED_CACHE AX_MLX_FLASH_NEXT_GDN_METAL
    bin=ax-engine-server-t9
    case $arm in
      prev) bin=ax-engine-server-t7;;
      newoff) export AX_FLASH_NEXT_FUSE=$OLDFUSE AX_FLASH_NEXT_POOLED_CACHE=0 AX_MLX_FLASH_NEXT_GDN_METAL=0;;
    esac
    sudo -n purge; sleep 3
    FN_AB_TOTAL_BLOCKS=4096 FN_AB_MAX_TOKENS=128 /usr/bin/python3 /tmp/fn_ab.py $PWD/$bin t9$arm$round out-t9-$arm-$round.json prompts/p1.txt prompts/long1.txt prompts/long32.txt > t9-$arm-$round.out 2>&1
  done
done
echo ALLDONE > t9_done.txt
