#!/bin/bash
set -eu
cd "$(dirname "$0")"
umask 077
pack=$1
scratch=$2
test "$(/usr/sbin/sysctl -n hw.memsize)" = 206158430208
test "$(/usr/sbin/sysctl -n kern.memorystatus_vm_pressure_level)" = 1
shasum -a 256 -c inputs.sha256
set +e
/usr/bin/caffeinate -i /usr/bin/env -i HOME="$HOME" PATH="/usr/bin:/bin:/usr/sbin:/sbin" "$PYTHON" -B -u run_probe.py "$PWD/direct_weights" "$PWD/selection.json" "$PWD/inventory.json" "$pack" "$scratch" "$PWD/result.json" --purpose functional
code=$?
printf '%s\n' "$code" > status.txt
rmdir "$scratch" 2>/dev/null
exit "$code"
