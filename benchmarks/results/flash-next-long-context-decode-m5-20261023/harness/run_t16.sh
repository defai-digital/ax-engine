#!/bin/zsh
# Same-binary A/B of the GEMV-with-epilogue hyper-connection read (hc_native_low_gate).
# base = shipping default mask; hc = default mask plus hc_native_low_gate.
export PATH=/opt/homebrew/bin:/usr/sbin:/usr/bin:/bin:/sbin
cd ~/build/fn-ab
DEFAULT=hc_norm,hc_low,hc_mix,hc_gate,hc_write,gdn_beta,gdn_post,gdn_conv_act,gdn_decay_pre,gdn_decay_post,gdn_l2_sum,gdn_q_scale,gdn_k_scale,moe_swiglu,moe_combine,qsa_rope_lo,qsa_rope_hi,qsa_gate,moe_route_tail,gdn_native_decay,gdn_native_qk
for round in 1 2 3; do
  for arm in base hc; do
    unset AX_FLASH_NEXT_FUSE
    if [ $arm = hc ]; then export AX_FLASH_NEXT_FUSE=$DEFAULT,hc_native_low_gate; fi
    sudo -n purge; sleep 3
    FN_AB_TOTAL_BLOCKS=4096 FN_AB_MAX_TOKENS=128 /usr/bin/python3 /tmp/fn_ab.py $PWD/ax-engine-server-t16 t16$arm$round out-t16-$arm-$round.json prompts/p1.txt prompts/p3.txt prompts/long1.txt > t16-$arm-$round.out 2>&1
  done
done
echo ALLDONE > t16_done.txt
