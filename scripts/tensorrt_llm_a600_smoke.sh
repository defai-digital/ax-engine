#!/usr/bin/env bash
# Smoke TensorRT-LLM OpenAI server + AX L2 backend (desktop/datacenter).
#
# Prerequisites on the GPU host (e.g. Thunder tnr-0 / A6000):
#   - venv with tensorrt_llm + trtllm-serve
#   - LD_LIBRARY_PATH includes CUDA 13 + TensorRT wheel libs
#   - trtllm-serve listening (default http://127.0.0.1:8000)
#
# Local (Mac) example with SSH tunnel:
#   ssh -N -L 18000:127.0.0.1:8000 tnr-0
#   TRTLLM_URL=http://127.0.0.1:18000 bash scripts/tensorrt_llm_a600_smoke.sh
#
# Required machine-readable runtime identity (fail-closed before worker I/O):
#   TENSORRT_LLM_UPSTREAM_VERSION   exact package/version string (e.g. 1.2.1)
#   TENSORRT_LLM_EXECUTION_BACKEND  execution path (e.g. pytorch)
# Optional:
#   TENSORRT_LLM_UPSTREAM_MODEL_ID  defaults to MODEL_ID
#   TENSORRT_LLM_RUNTIME_PROFILE    e.g. cuda-linux-x86_64-a6000-sm86
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRTLLM_URL="${TRTLLM_URL:-http://127.0.0.1:8000}"
AX_HOST="${AX_HOST:-127.0.0.1}"
AX_PORT="${AX_PORT:-31428}"
MODEL_ID="${MODEL_ID:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
AX_BIN="${AX_BIN:-./target/debug/ax-engine-server}"
UPSTREAM_MODEL_ID="${TENSORRT_LLM_UPSTREAM_MODEL_ID:-$MODEL_ID}"
UPSTREAM_VERSION="${TENSORRT_LLM_UPSTREAM_VERSION:-}"
EXECUTION_BACKEND="${TENSORRT_LLM_EXECUTION_BACKEND:-}"
RUNTIME_PROFILE="${TENSORRT_LLM_RUNTIME_PROFILE:-cuda-linux-x86_64-a6000-sm86}"

if [[ -z "$UPSTREAM_VERSION" || -z "$EXECUTION_BACKEND" ]]; then
  echo "error: set TENSORRT_LLM_UPSTREAM_VERSION and TENSORRT_LLM_EXECUTION_BACKEND" >&2
  echo "       (AX fails closed without machine-readable TensorRT-LLM identity)" >&2
  exit 2
fi

echo "==> probe TRT-LLM at ${TRTLLM_URL}"
curl -sf "${TRTLLM_URL}/v1/models" | head -c 500
echo

if [[ ! -x "$AX_BIN" ]]; then
  echo "==> build ax-engine-server"
  cargo build -p ax-engine-server --quiet
fi

echo "==> negative: missing runtime identity must fail closed before listen"
set +e
"$AX_BIN" \
  --support-tier tensor-rt-llm \
  --tensorrt-llm-server-url "$TRTLLM_URL" \
  --model-id "$MODEL_ID" \
  --host "$AX_HOST" \
  --port 0 \
  > /tmp/ax-trtllm-negative.log 2>&1
neg_rc=$?
set -e
if [[ "$neg_rc" -eq 0 ]]; then
  echo "error: expected non-zero exit when TensorRT-LLM identity is missing" >&2
  exit 1
fi
if ! grep -q "tensorrt-llm-upstream-version\|machine-readable runtime identity" /tmp/ax-trtllm-negative.log; then
  echo "error: negative path did not report missing identity; log:" >&2
  tail -50 /tmp/ax-trtllm-negative.log >&2
  exit 1
fi
echo "negative fail-closed ok (exit=${neg_rc})"

echo "==> start AX L2 (TensorRtLlm) on ${AX_HOST}:${AX_PORT}"
"$AX_BIN" \
  --support-tier tensor-rt-llm \
  --tensorrt-llm-server-url "$TRTLLM_URL" \
  --tensorrt-llm-upstream-model-id "$UPSTREAM_MODEL_ID" \
  --tensorrt-llm-upstream-version "$UPSTREAM_VERSION" \
  --tensorrt-llm-execution-backend "$EXECUTION_BACKEND" \
  --tensorrt-llm-runtime-profile "$RUNTIME_PROFILE" \
  --model-id "$MODEL_ID" \
  --host "$AX_HOST" \
  --port "$AX_PORT" \
  > /tmp/ax-trtllm-smoke.log 2>&1 &
AX_PID=$!
cleanup() {
  kill "$AX_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 30); do
  if curl -sf "http://${AX_HOST}:${AX_PORT}/health" >/tmp/ax-trtllm-health.json 2>/dev/null; then
    break
  fi
  sleep 0.5
done
if ! curl -sf "http://${AX_HOST}:${AX_PORT}/health" >/tmp/ax-trtllm-health.json; then
  echo "AX health failed; log:"
  tail -50 /tmp/ax-trtllm-smoke.log
  exit 1
fi

python3 - <<'PY'
import json, sys
h = json.load(open("/tmp/ax-trtllm-health.json"))
rt = h.get("runtime") or {}
assert rt.get("selected_backend") == "tensor_rt_llm", rt
assert rt.get("support_tier") == "tensor_rt_llm", rt
print("health backend ok:", rt.get("selected_backend"))
PY

echo "==> chat completions via AX"
curl -sf "http://${AX_HOST}:${AX_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in one word.\"}],\"max_tokens\":16,\"temperature\":0}" \
  | tee /tmp/ax-trtllm-chat.json
echo

python3 - <<'PY'
import json
r = json.load(open("/tmp/ax-trtllm-chat.json"))
content = r["choices"][0]["message"]["content"]
assert content and isinstance(content, str), r
print("chat content:", repr(content))
print("SMOKE OK")
PY
