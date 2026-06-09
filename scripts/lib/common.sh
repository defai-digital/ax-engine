#!/usr/bin/env bash

_AX_CALLER="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
AX_SCRIPT_DIR="$(cd "$(dirname "$_AX_CALLER")" && pwd)"
AX_REPO_ROOT="$(cd "$AX_SCRIPT_DIR/.." && pwd)"
AX_PYTHON_BIN="${PYTHON_BIN:-python3}"

ax_tmp_dir() {
    local prefix="$1"
    mktemp -d "${TMPDIR:-/tmp}/${prefix}.XXXXXX"
}

ax_tmp_file() {
    local prefix="$1"
    local suffix="${2:-}"
    local tmp_root="${TMPDIR:-/tmp}"
    if [[ -z "$suffix" ]]; then
        mktemp "$tmp_root/${prefix}.XXXXXX"
        return
    fi
    "$AX_PYTHON_BIN" - "$tmp_root" "$prefix" "$suffix" <<'PY'
import os
import sys
import tempfile

tmp_root, prefix, suffix = sys.argv[1:4]
fd, path = tempfile.mkstemp(prefix=f"{prefix}.", suffix=suffix, dir=tmp_root)
os.close(fd)
print(path)
PY
}

ax_rm_rf() {
    local path
    for path in "$@"; do
        if [[ -n "${path:-}" ]]; then
            rm -rf "$path"
        fi
    done
}

ax_kill_pid() {
    local pid="${1:-}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

ax_allocate_port() {
    "$AX_PYTHON_BIN" - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}
