#!/bin/bash
set -euo pipefail
umask 077
cd "$(dirname "$0")"
shasum -a 256 -c inputs-m2.sha256
test "$(/usr/sbin/sysctl -n hw.memsize)" = 206158430208
if /bin/ps -axo comm= | /usr/bin/grep -E '/(ax-engine-server[^/]*|direct_weights|ax-ple-reader-test|ax-flash-shared-load-test)$'; then exit 1; fi
python="$HOME/code/ax-engine/.venv/bin/python"
lib=$("$python" -B -c 'from importlib.metadata import version, distribution; assert version("mlx") == version("mlx-metal") == "0.32.4"; print(distribution("mlx").locate_file("mlx/lib"))')
expected=$("$python" -B -c 'import json; print(json.load(open("build.json"))["libmlx_sha256"])')
test "$(shasum -a 256 "$lib/libmlx.dylib" | awk '{print $1}')" = "$expected"
pack=/Volumes/Ext16TR0/models/.flash-strict-convert-c3dcu5dx/mxfp4
exec /usr/bin/caffeinate -i /usr/bin/env -i HOME="$HOME" PATH="/usr/bin:/bin:/usr/sbin:/sbin" \
  DYLD_LIBRARY_PATH="$lib" "$python" -B -u run_required.py --server "$PWD/ax-engine-server-startup" \
  --build build.json --inventory inventory.json --pack "$pack" --output observed
