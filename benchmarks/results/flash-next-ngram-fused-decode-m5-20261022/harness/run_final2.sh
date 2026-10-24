#!/bin/zsh
export PATH=/opt/homebrew/bin:/usr/sbin:/usr/bin:/bin:/sbin
cd ~/build/fn-ab
NEW=$HOME/build/fn-ab/ax-engine-server-fuse2
BASE=$HOME/build/fn-baseline/ax-engine-server
run() { label=$1 bin=$2; shift 2
  sudo -n purge; sleep 3
  env "$@" python3 /tmp/fn_ab.py $bin ${label}-s out-${label}-s.json prompts/p1.txt prompts/p3.txt prompts/p5.txt prompts/p6.txt | cut -c1-200
  sudo -n purge; sleep 3
  env "$@" python3 /tmp/fn_ab.py $bin ${label}-l out-${label}-l.json prompts/long1.txt | cut -c1-200; }
run g_base $BASE X=1
run g_io $NEW AX_FLASH_NEXT_FUSE=0
run g_full $NEW X=1
echo ALL_DONE
