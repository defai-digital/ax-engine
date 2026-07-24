#!/usr/bin/env bash
# Smoke TensorRT Edge-LLM OpenAI server on a Thor host, then optionally probe
# AX Engine delegated backend against that URL.
#
# Usage:
#   bash scripts/edge_llm_thor_smoke.sh                # uses df-thor-01
#   THOR_HOST=df-thor-02 bash scripts/edge_llm_thor_smoke.sh
#   EDGE_LLM_URL=http://127.0.0.1:8000 bash scripts/edge_llm_thor_smoke.sh --ax-only
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
THOR_HOST="${THOR_HOST:-df-thor-01}"
EDGE_PORT="${EDGE_PORT:-8000}"
EDGE_MODEL="${EDGE_MODEL:-Qwen/Qwen3-0.6B}"
AX_MODEL_ID="${AX_MODEL_ID:-qwen3}"
REMOTE_DIR="${REMOTE_DIR:-$HOME/TensorRT-Edge-LLM}"
MODE="${1:-full}"

log() { printf '[edge-llm-smoke] %s\n' "$*"; }

remote() {
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$THOR_HOST" "$@"
}

if [[ "$MODE" != "--ax-only" ]]; then
  log "Probing $THOR_HOST"
  remote "uname -m && cat /etc/nv_tegra_release 2>/dev/null | head -1"

  log "Ensuring TensorRT-Edge-LLM checkout at $REMOTE_DIR"
  remote "set -e
    if [[ ! -d '$REMOTE_DIR/.git' ]]; then
      git clone --depth 1 https://github.com/NVIDIA/TensorRT-Edge-LLM.git '$REMOTE_DIR'
    else
      git -C '$REMOTE_DIR' fetch --depth 1 origin HEAD && git -C '$REMOTE_DIR' reset --hard FETCH_HEAD
    fi
    cd '$REMOTE_DIR'
    python3 -m pip install --user -q -r requirements.txt 2>/dev/null || true
    python3 -m pip install --user -q -r requirements-server.txt 2>/dev/null || true
    python3 -m pip install --user -q -e . 2>/dev/null || python3 -m pip install --user -q .
    python3 -c 'import tensorrt_edgellm,sys; print(\"tensorrt_edgellm\", getattr(tensorrt_edgellm,\"__version__\", \"unknown\"))' || \
      python3 -c 'print(\"import path check\"); import experimental.server' || true
  "

  log "Starting experimental OpenAI server on port $EDGE_PORT (background)"
  remote "set -e
    cd '$REMOTE_DIR'
    pkill -f 'experimental.server' 2>/dev/null || true
    export PATH=/usr/local/cuda-13.2/bin:$PATH
    export LD_LIBRARY_PATH=${REMOTE_DIR}/build-thor:/usr/local/cuda-13.2/lib64:/usr/local/cuda-13.2/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}
    export EDGELLM_PLUGIN_PATH=${REMOTE_DIR}/build-thor/libNvInfer_edgellm_plugin.so
    export BUILD_DIR=${REMOTE_DIR}/build-thor
    nohup env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" EDGELLM_PLUGIN_PATH="$EDGELLM_PLUGIN_PATH" BUILD_DIR="$BUILD_DIR" python3 -m experimental.server --model '$EDGE_MODEL' --port $EDGE_PORT \
      > /tmp/edge-llm-server.log 2>&1 &
    echo \$! > /tmp/edge-llm-server.pid
    for i in \$(seq 1 60); do
      if curl -sf http://127.0.0.1:$EDGE_PORT/v1/models >/dev/null 2>&1 \
         || curl -sf http://127.0.0.1:$EDGE_PORT/health >/dev/null 2>&1; then
        echo ready
        exit 0
      fi
      sleep 2
    done
    echo 'server failed to become ready; last log:' >&2
    tail -n 80 /tmp/edge-llm-server.log >&2 || true
    exit 1
  "

  EDGE_LLM_URL="http://$(remote 'hostname -I' | awk '{print $1}'):${EDGE_PORT}"
  log "Remote Edge-LLM URL candidate: $EDGE_LLM_URL"
else
  EDGE_LLM_URL="${EDGE_LLM_URL:?set EDGE_LLM_URL for --ax-only}"
fi

log "Direct chat smoke against Edge-LLM (via SSH localhost)"
remote "curl -sf http://127.0.0.1:$EDGE_PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{\"model\":\"$EDGE_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in one word.\"}],\"max_tokens\":16}' \
  | head -c 800; echo"

if command -v cargo >/dev/null 2>&1 && [[ -x "$ROOT/target/debug/ax-engine-server" || -x "$ROOT/target/release/ax-engine-server" ]]; then
  BIN="$ROOT/target/release/ax-engine-server"
  [[ -x "$BIN" ]] || BIN="$ROOT/target/debug/ax-engine-server"
  AX_PORT="${AX_PORT:-31419}"
  log "Starting local AX against Edge-LLM URL (if reachable from this host)"
  # Prefer SSH tunnel if remote-only
  if ! curl -sf --connect-timeout 2 "http://127.0.0.1:$EDGE_PORT/v1/models" >/dev/null 2>&1; then
    log "Opening SSH tunnel localhost:$EDGE_PORT -> $THOR_HOST:$EDGE_PORT"
    ssh -f -N -L "$EDGE_PORT:127.0.0.1:$EDGE_PORT" "$THOR_HOST" || true
    LOCAL_EDGE="http://127.0.0.1:$EDGE_PORT"
  else
    LOCAL_EDGE="http://127.0.0.1:$EDGE_PORT"
  fi
  pkill -f "ax-engine-server.*$AX_PORT" 2>/dev/null || true
  "$BIN" \
    --host 127.0.0.1 \
    --port "$AX_PORT" \
    --model-id "$AX_MODEL_ID" \
    --support-tier tensor-rt-edge-llm \
    --edge-llm-server-url "$LOCAL_EDGE" \
    > /tmp/ax-edge-llm-server.log 2>&1 &
  echo $! > /tmp/ax-edge-llm-server.pid
  sleep 2
  log "AX chat smoke"
  curl -sf "http://127.0.0.1:$AX_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$AX_MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in one word.\"}],\"max_tokens\":16}" \
    | head -c 800
  echo
  log "AX smoke done (pid $(cat /tmp/ax-edge-llm-server.pid))"
else
  log "Skipping local AX binary smoke (build ax-engine-server first)"
  log "Manual: ax-engine-server --support-tier tensor-rt-edge-llm --edge-llm-server-url <url> --model-id $AX_MODEL_ID"
fi

log "OK"
