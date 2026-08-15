#!/usr/bin/env bash
# Run Qwen 3.8 27B AXQ MTP + direct benches on a remote Apple Silicon host.
#
# The published pack is AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP (27B, not 26B).
# Default host is df-macbookpro-m4. The script exits 75 when the host is
# unreachable or already running a bench so a wrapper can retry later.
set -euo pipefail

HOST="${AX_BENCH_HOST:-df-macbookpro-m4}"
REPO_DIR="${AX_ENGINE_REMOTE_REPO:-/Users/akiralam/code/ax-engine}"
RUN_DATE="${AX_BENCH_RUN_DATE:-$(date +%F)}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=30)

busy_exit() {
  echo "[busy] $*" >&2
  exit 75
}

if ! ssh "${SSH_OPTS[@]}" "$HOST" "true" >/dev/null 2>&1; then
  busy_exit "cannot reach $HOST"
}

remote() {
  ssh "${SSH_OPTS[@]}" "$HOST" "$@"
}

host_busy() {
  remote 'bash -s' <<'EOF'
set -euo pipefail
load="$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')"
if awk -v load="${load:-0}" 'BEGIN { exit !(load+0 > 2.0) }'; then
  echo "load_average=${load}"
  exit 0
fi
if pgrep -fl 'bench_mtp_6bit_ax_refresh|bench_ax_only_sweep|bench_mlx_inference_stack|ax-engine-server' >/dev/null 2>&1; then
  pgrep -fl 'bench_mtp_6bit_ax_refresh|bench_ax_only_sweep|bench_mlx_inference_stack|ax-engine-server' | head
  exit 0
fi
exit 1
EOF
}

if host_busy; then
  busy_exit "$HOST already has a bench or load above 2.0"
fi

echo "[start] $HOST Qwen 3.8 27B AXQ MTP + direct ($RUN_DATE)"
remote "bash -lc $(printf '%q' "$(cat <<EOF
set -euo pipefail
cd '$REPO_DIR'
git pull --ff-only || true
if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
python3 -m pip install -q -U huggingface_hub
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP",
    revision="a5a0b700ea7c5c529c66ca3005b79425ab2f7ea6",
)
snapshot_download(
    "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP",
    revision="7e865596cb32bd41b29c7a25c5b66b9c3ea25e5e",
)
print("downloaded")
PY
cargo build --release -p ax-engine-server
python3 scripts/bench_mtp_6bit_ax_refresh.py \\
  --targets qwen3.8-27b-axq-6bit \\
  --output-dir benchmarks/results/speculative/mtp-6bit/${RUN_DATE}-qwen38-27b-axq-6bit-m4 \\
  --no-build-ax-engine
python3 scripts/bench_ax_only_sweep.py \\
  --output-root benchmarks/results/inference/ax-direct/${RUN_DATE}-qwen38-27b-axq-m4 \\
  --ax-direct-only \\
  --rows-filter qwen3_8-27b-axq-6bit qwen3_8-27b-axq-4bit
echo DONE
EOF
)")"

echo "[done] benches finished on $HOST"
exit 0
