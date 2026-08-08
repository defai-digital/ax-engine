#!/usr/bin/env bash

# Launch the reusable AXQ endurance runner so it survives terminal, SSH, and
# monitoring-client disconnects. Runtime evidence remains owned by the remote
# Mac; a host reboot or power loss intentionally ends (and does not resume) the
# same-process test contract.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_axq_endurance.py"
PYTHON_BIN="${AX_ENDURANCE_PYTHON:-python3}"
CAFFEINATE_BIN="${AX_ENDURANCE_CAFFEINATE:-/usr/bin/caffeinate}"

usage() {
    echo "Usage: $0 --server PATH --model-dir DIR --output-dir DIR [runner options]" >&2
}

if [[ "$#" -eq 0 ]]; then
    usage
    exit 2
fi

output_dir=""
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
    argument="${arguments[$index]}"
    case "$argument" in
        --output-dir)
            if ((index + 1 >= ${#arguments[@]})); then
                echo "--output-dir requires a value" >&2
                exit 2
            fi
            output_dir="${arguments[$((index + 1))]}"
            ;;
        --output-dir=*)
            output_dir="${argument#--output-dir=}"
            ;;
    esac
done

if [[ -z "$output_dir" ]]; then
    echo "--output-dir is required so detached launch evidence has a stable name" >&2
    usage
    exit 2
fi
if [[ ! -x "$CAFFEINATE_BIN" ]]; then
    echo "caffeinate is unavailable or not executable: $CAFFEINATE_BIN" >&2
    exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python is unavailable: $PYTHON_BIN" >&2
    exit 1
fi
if [[ -e "$output_dir" && ! -d "$output_dir" ]]; then
    echo "output path exists and is not a directory: $output_dir" >&2
    exit 1
fi
if [[ -d "$output_dir" ]] && [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "output directory is not empty: $output_dir" >&2
    exit 1
fi

mkdir -p "$(dirname "$output_dir")"
launcher_log="${output_dir}.launcher.log"
launcher_pid_file="${output_dir}.launcher.pid"
if [[ -e "$launcher_log" || -e "$launcher_pid_file" ]]; then
    echo "detached launcher evidence already exists for: $output_dir" >&2
    exit 1
fi

nohup "$CAFFEINATE_BIN" -dimsu "$PYTHON_BIN" "$RUNNER" "${arguments[@]}" \
    </dev/null >"$launcher_log" 2>&1 &
launcher_pid=$!
printf '%s\n' "$launcher_pid" >"$launcher_pid_file"

sleep 2
if ! kill -0 "$launcher_pid" 2>/dev/null; then
    echo "detached endurance launcher exited during startup; see $launcher_log" >&2
    tail -40 "$launcher_log" >&2 || true
    exit 1
fi

echo "Detached AXQ endurance launch is alive."
echo "Launcher PID: $launcher_pid"
echo "Runner output: $output_dir"
echo "Launcher log: $launcher_log"
echo "PID receipt: $launcher_pid_file"
