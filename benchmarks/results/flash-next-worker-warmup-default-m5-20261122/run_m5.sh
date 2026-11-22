#!/bin/bash
set -euo pipefail
umask 077
cd "$(dirname "$0")"
previous="$1"
cache="$2"
echo waiting-for-owned-server-prefill-tail
deadline=$((SECONDS + 57600))
until test -s "$previous/status.txt"; do
    kill -0 "$(cat "$previous/owner.pid")" 2>/dev/null && (( SECONDS < deadline )) || exit 2
    sleep 15
done
shasum -a 256 -c inputs.sha256
[[ $(/usr/sbin/sysctl -n hw.model hw.memsize) = "$(printf 'Mac17,6\n137438953472')" ]]
if /bin/ps -axo comm= | /usr/bin/grep -E '/(ax-engine-server[^/]*|direct_weights|ax-ple-reader-test)$'; then exit 1; fi
python="$HOME/.venvs/ax-engine/bin/python"
lib=$("$python" -B -c 'from importlib.metadata import version, distribution; assert version("mlx") == version("mlx-metal") == "0.32.4"; print(distribution("mlx").locate_file("mlx/lib"))')
expected=$("$python" -B -c 'import json; print(json.load(open("build.json"))["libmlx_sha256"])')
test "$(shasum -a 256 "$lib/libmlx.dylib" | awk '{print $1}')" = "$expected"
test "$(stat -f %d "$cache")" = "$(stat -f %d .)"
exec /usr/bin/caffeinate -dimsu /usr/bin/env -i HOME="$HOME" PATH="/usr/bin:/bin:/usr/sbin:/sbin" \
  DYLD_LIBRARY_PATH="$lib" "$python" -B -u run_m5.py "$cache"
