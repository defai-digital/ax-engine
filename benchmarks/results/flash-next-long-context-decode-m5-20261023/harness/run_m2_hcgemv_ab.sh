#!/bin/zsh
export PATH=/opt/homebrew/bin:/usr/sbin:/usr/bin:/bin:/sbin
cd ~/build/fn-ab-m2
DEFAULT=hc_norm,hc_low,hc_mix,hc_gate,hc_write,gdn_beta,gdn_post,gdn_conv_act,gdn_decay_pre,gdn_decay_post,gdn_l2_sum,gdn_q_scale,gdn_k_scale,moe_swiglu,moe_combine,qsa_rope_lo,qsa_rope_hi,qsa_gate,moe_route_tail,gdn_native_decay,gdn_native_qk
mkdir -p ~/build/fn-ab
for round in 1 2; do
  for arm in def nohc; do
    unset AX_FLASH_NEXT_FUSE
    if [ $arm = nohc ]; then export AX_FLASH_NEXT_FUSE=$DEFAULT; fi
    FN_AB_TOTAL_BLOCKS=4096 FN_AB_MAX_TOKENS=128 /usr/bin/python3 fn_ab_m2.py $HOME/build/ax-fn-m2-target/release/ax-engine-server m2$arm$round out-m2-$arm-$round.json prompts/p1.txt prompts/p3.txt prompts/long1.txt > m2-$arm-$round.out 2>&1
  done
done
echo ALLDONE > m2_done.txt
