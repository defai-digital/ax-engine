#!/bin/bash
set -euo pipefail
umask 077
cd "$(dirname "$0")"
shasum -a 256 -c inputs-m4.sha256
[[ $(/usr/sbin/sysctl -n machdep.cpu.brand_string hw.memsize) = "$(printf 'Apple M4 Pro\n68719476736')" ]]
[[ $(/usr/sbin/sysctl -n kern.memorystatus_vm_pressure_level) = 1 ]]
for pool in 0 1; do
  env DYLD_LIBRARY_PATH="$PWD" AX_FLASH_NEXT_NGRAM_READ_POOL="$pool" AX_FLASH_NEXT_NGRAM_READ_THREADS=32 \
    ./ax-ple-reader-test ngram_table:: --test-threads=2 > "pool-$pool.txt" 2>&1
  grep -F '20 passed; 0 failed' "pool-$pool.txt"
done
[[ $(/usr/sbin/sysctl -n kern.memorystatus_vm_pressure_level) = 1 ]]
