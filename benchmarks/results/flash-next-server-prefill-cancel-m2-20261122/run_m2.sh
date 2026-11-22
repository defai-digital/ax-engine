#!/bin/bash
set -euo pipefail
umask 077
cd "$(dirname "$0")"
shasum -a 256 -c stage.sha256
test "$(/usr/sbin/sysctl -n hw.memsize)" = 206158430208
if /bin/ps -axo comm= | /usr/bin/grep -E '/(ax-engine-server[^/]*|ax-ple-reader-test|direct_weights|ax-flash-shared-load-test)$'; then exit 1; fi
python="$HOME/code/ax-engine/.venv/bin/python"
mlx=$("$python" -B -c 'import importlib.metadata as m; assert m.version("mlx") == m.version("mlx-metal") == "0.32.4"; print(m.distribution("mlx").locate_file("mlx/lib"))')
expected=$("$python" -B -c 'import json; print(json.load(open("build.json"))["libmlx_sha256"])')
test "$(shasum -a 256 "$mlx/libmlx.dylib" | awk '{print $1}')" = "$expected"
pack=/Volumes/Ext16TR0/models/.flash-strict-convert-c3dcu5dx/mxfp4
exec /usr/bin/caffeinate -i /usr/bin/env -i HOME="$HOME" PATH="/usr/bin:/bin:/usr/sbin:/sbin" \
  DYLD_LIBRARY_PATH="$mlx" "$python" -B -u run_cancel.py \
  --binary "$PWD/ax-engine-server-tail" --pack "$pack" --out "$PWD/observed"
