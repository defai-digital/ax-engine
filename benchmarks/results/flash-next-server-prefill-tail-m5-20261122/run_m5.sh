#!/bin/bash
set -euo pipefail
umask 077
cd "$(dirname "$0")"
previous="$1"
shift
echo waiting-for-owned-shared-load-retry
deadline=$((SECONDS + 57600))
until test -s "$previous/status.txt"; do
    kill -0 "$(cat "$previous/owner.pid")" 2>/dev/null && (( SECONDS < deadline )) || exit 2
    sleep 15
done
export PYTHONDONTWRITEBYTECODE=1
test "$(/usr/sbin/sysctl -n hw.model hw.memsize)" = "$(printf 'Mac17,6\n137438953472')"
if /bin/ps -axo comm= | /usr/bin/grep -E '/(ax-engine-server[^/]*|ax-flash-shared-load-test|ax-ple-reader-test|direct_weights)$'; then
    exit 1
fi
/usr/bin/shasum -a 256 -c inputs.sha256
cache="$1"
pack="$cache/models--AutomatosX--AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP/snapshots/ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
python="$HOME/.venvs/ax-engine/bin/python"
mlx=$("$python" -c 'import importlib.metadata as m; assert m.version("mlx") == m.version("mlx-metal") == "0.32.4"; print(m.distribution("mlx").locate_file("mlx/lib"))')
expected=$("$python" -B -c 'import json; print(json.load(open("build.json"))["libmlx_sha256"])')
test "$(shasum -a 256 "$mlx/libmlx.dylib" | awk '{print $1}')" = "$expected"
exec /usr/bin/caffeinate -dimsu /usr/bin/env -i HOME="$HOME" \
  PATH="$HOME/.venvs/ax-engine/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
  DYLD_LIBRARY_PATH="$mlx" PYTHONDONTWRITEBYTECODE=1 \
  "$python" -B -u run_server_tail.py --binary "$PWD/ax-engine-server-tail" \
  --pack "$pack" --link-root "$cache" --out "$PWD/observed"
