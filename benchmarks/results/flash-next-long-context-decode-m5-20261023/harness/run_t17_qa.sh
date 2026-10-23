#!/bin/zsh
export PATH=/opt/homebrew/bin:/usr/sbin:/usr/bin:/bin:/sbin
cd ~/build/fn-ab
BIN=$HOME/build/fn-ab/ax-engine-server-t17
Q=/usr/bin/python3
OLDFUSE=hc_norm,hc_low,hc_mix,hc_gate,hc_write,gdn_beta,gdn_post,gdn_conv_act,gdn_decay_pre,gdn_decay_post,gdn_l2_sum,gdn_q_scale,gdn_k_scale,moe_swiglu,moe_combine
for arm in off def; do
  unset AX_FLASH_NEXT_KV_SEGMENTS AX_FLASH_NEXT_POOLED_CACHE AX_MLX_FLASH_NEXT_GDN_METAL AX_FLASH_NEXT_FUSE AX_FLASH_NEXT_ROPE_CACHE
  if [ $arm = off ]; then export AX_FLASH_NEXT_KV_SEGMENTS=0 AX_FLASH_NEXT_POOLED_CACHE=0 AX_MLX_FLASH_NEXT_GDN_METAL=0 AX_FLASH_NEXT_FUSE=$OLDFUSE AX_FLASH_NEXT_ROPE_CACHE=0; fi
  echo "== direct texts $arm"; QA_TOTAL_BLOCKS=4096 $Q /tmp/fn_qa2.py $BIN t17-direct-$arm texts prompts/p1.txt prompts/p3.txt prompts/p5.txt prompts/s1.txt prompts/s2.txt prompts/s3.txt prompts/long1.txt prompts/long32.txt > qa-t17-direct-$arm.json 2> qa-t17-direct-$arm.err
  echo "== mtp canonical $arm"; QA_MTP_POLICY=required QA_TOTAL_BLOCKS=4096 $Q /tmp/fn_qa2.py $BIN t17-mtp-$arm texts prompts/p3.txt prompts/long1.txt prompts/long32.txt > qa-t17-mtp-$arm.json 2> qa-t17-mtp-$arm.err
  echo "== mtp batched $arm"; QA_MTP_POLICY=required AX_FLASH_NEXT_MTP_VERIFIER=batched QA_TOTAL_BLOCKS=4096 $Q /tmp/fn_qa2.py $BIN t17-mtpb-$arm texts prompts/p3.txt prompts/long1.txt prompts/long32.txt > qa-t17-mtpb-$arm.json 2> qa-t17-mtpb-$arm.err
  echo "== cancel $arm"; QA_TOTAL_BLOCKS=4096 $Q /tmp/fn_qa2.py $BIN t17-cancel-$arm cancel prompts/p1.txt prompts/long32.txt > qa-t17-cancel-$arm.json 2> qa-t17-cancel-$arm.err
done
unset AX_FLASH_NEXT_KV_SEGMENTS AX_FLASH_NEXT_POOLED_CACHE AX_MLX_FLASH_NEXT_GDN_METAL AX_FLASH_NEXT_FUSE AX_FLASH_NEXT_ROPE_CACHE
echo "== soak 9k default"; QA_SOAK_N=16 QA_TOTAL_BLOCKS=4096 $Q /tmp/fn_qa2.py $BIN t17-soak soak prompts/long1.txt > qa-t17-soak.json 2> qa-t17-soak.err
echo "== soak 2k default"; QA_SOAK_N=30 $Q /tmp/fn_qa2.py $BIN t17-soak2k soak prompts/s2.txt > qa-t17-soak2k.json 2> qa-t17-soak2k.err
echo STREAM; $Q stream_smoke.py $BIN t17def > t17_stream.out 2>&1
echo IMG; $Q img_smoke.py $BIN t17def > t17_img.out 2>&1
echo ALLDONE > t17_done.txt
