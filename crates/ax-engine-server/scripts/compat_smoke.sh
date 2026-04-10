#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8080}"
MODEL_ID="${2:-}"
SLOT_FILENAME="${AX_ENGINE_SLOT_FILENAME:-compat-slot.json}"

if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is required" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is required" >&2
  exit 1
fi

AUTH_ARGS=()
if [[ -n "${AX_ENGINE_API_KEY:-}" ]]; then
  AUTH_ARGS+=(-H "Authorization: Bearer ${AX_ENGINE_API_KEY}")
fi

request_json() {
  local method="$1"
  local path="$2"
  local payload="${3:-}"

  if [[ -n "${payload}" ]]; then
    if (( ${#AUTH_ARGS[@]} > 0 )); then
      curl -sS "${AUTH_ARGS[@]}" \
        -H "Content-Type: application/json" \
        -X "${method}" \
        "${BASE_URL}${path}" \
        -d "${payload}"
    else
      curl -sS \
        -H "Content-Type: application/json" \
        -X "${method}" \
        "${BASE_URL}${path}" \
        -d "${payload}"
    fi
  else
    if (( ${#AUTH_ARGS[@]} > 0 )); then
      curl -sS "${AUTH_ARGS[@]}" \
        -X "${method}" \
        "${BASE_URL}${path}"
    else
      curl -sS \
        -X "${method}" \
        "${BASE_URL}${path}"
    fi
  fi
}

assert_json() {
  local json="$1"
  local code="$2"
  python3 - "$json" "$code" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
code = sys.argv[2]

if code == "health":
    assert payload["status"] == "ok"
elif code == "models":
    assert payload["object"] == "list"
    assert payload["data"]
elif code == "slots":
    assert isinstance(payload, list)
    assert payload and payload[0]["id"] == 0
elif code == "tokenize":
    assert "tokens" in payload and payload["tokens"]
elif code == "apply-template":
    assert "prompt" in payload and payload["prompt"]
elif code == "chat":
    assert payload["object"] == "chat.completion"
    assert payload["choices"]
elif code == "chat_repeat":
    assert payload["object"] == "chat.completion"
    assert payload["choices"]
    assert "timings" in payload
    assert payload["timings"]["cache_n"] > 0
elif code == "responses":
    assert payload["object"] == "response"
    assert "output_text" in payload
elif code == "completion":
    assert "content" in payload
elif code == "slot_save":
    assert payload["filename"]
    assert payload["n_saved"] >= 0
    assert payload["n_written"] > 0
elif code == "slot_restore":
    assert payload["filename"]
    assert payload["n_restored"] >= 0
    assert payload["n_read"] > 0
else:
    raise AssertionError(f"unknown assertion code: {code}")
PY
}

echo "== health =="
HEALTH_JSON="$(request_json GET /health)"
echo "${HEALTH_JSON}"
assert_json "${HEALTH_JSON}" health

echo "== models =="
MODELS_JSON="$(request_json GET /v1/models)"
echo "${MODELS_JSON}"
assert_json "${MODELS_JSON}" models

if [[ -z "${MODEL_ID}" ]]; then
  MODEL_ID="$(python3 - "${MODELS_JSON}" <<'PY'
import json
import sys
payload = json.loads(sys.argv[1])
print(payload["data"][0]["id"])
PY
)"
fi

echo "== slots =="
SLOTS_JSON="$(request_json GET /slots)"
echo "${SLOTS_JSON}"
assert_json "${SLOTS_JSON}" slots

echo "== tokenize =="
TOKENIZE_JSON="$(request_json POST /tokenize '{"content":"Hello <|im_start|>user\nHi<|im_end|>","add_special":false,"parse_special":true,"with_pieces":true}')"
echo "${TOKENIZE_JSON}"
assert_json "${TOKENIZE_JSON}" tokenize

echo "== apply-template =="
APPLY_TEMPLATE_JSON="$(request_json POST /apply-template '{"messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"Say hello."}]}')"
echo "${APPLY_TEMPLATE_JSON}"
assert_json "${APPLY_TEMPLATE_JSON}" apply-template

echo "== chat completions =="
CHAT_JSON="$(request_json POST /v1/chat/completions "$(cat <<JSON
{"model":"${MODEL_ID}","messages":[{"role":"system","content":"Answer in six words max."},{"role":"user","content":"Say hello to AX."}],"max_tokens":8,"temperature":0.0,"top_k":1,"seed":7}
JSON
)")"
echo "${CHAT_JSON}"
assert_json "${CHAT_JSON}" chat

echo "== cached chat completions =="
CHAT_CACHED_JSON="$(request_json POST /v1/chat/completions "$(cat <<JSON
{"model":"${MODEL_ID}","messages":[{"role":"system","content":"Answer in six words max."},{"role":"user","content":"Say hello to AX."}],"max_tokens":8,"temperature":0.0,"top_k":1,"seed":7}
JSON
)")"
echo "${CHAT_CACHED_JSON}"
assert_json "${CHAT_CACHED_JSON}" chat_repeat
python3 - "${CHAT_JSON}" "${CHAT_CACHED_JSON}" <<'PY'
import json
import sys

first = json.loads(sys.argv[1])
second = json.loads(sys.argv[2])
assert first["choices"][0]["message"]["content"] == second["choices"][0]["message"]["content"]
PY

echo "== slot save =="
SAVE_JSON="$(request_json POST /slots/0?action=save "$(cat <<JSON
{"filename":"${SLOT_FILENAME}"}
JSON
)")"
echo "${SAVE_JSON}"
assert_json "${SAVE_JSON}" slot_save

echo "== slot erase =="
ERASE_JSON="$(request_json POST /slots/0?action=erase)"
echo "${ERASE_JSON}"

echo "== slot restore =="
RESTORE_JSON="$(request_json POST /slots/0?action=restore "$(cat <<JSON
{"filename":"${SLOT_FILENAME}"}
JSON
)")"
echo "${RESTORE_JSON}"
assert_json "${RESTORE_JSON}" slot_restore

echo "== post-restore cached chat completions =="
CHAT_RESTORED_JSON="$(request_json POST /v1/chat/completions "$(cat <<JSON
{"model":"${MODEL_ID}","messages":[{"role":"system","content":"Answer in six words max."},{"role":"user","content":"Say hello to AX."}],"max_tokens":8,"temperature":0.0,"top_k":1,"seed":7}
JSON
)")"
echo "${CHAT_RESTORED_JSON}"
assert_json "${CHAT_RESTORED_JSON}" chat_repeat
python3 - "${CHAT_JSON}" "${CHAT_RESTORED_JSON}" <<'PY'
import json
import sys

first = json.loads(sys.argv[1])
restored = json.loads(sys.argv[2])
assert first["choices"][0]["message"]["content"] == restored["choices"][0]["message"]["content"]
PY

echo "== responses =="
RESPONSES_JSON="$(request_json POST /v1/responses "$(cat <<JSON
{"model":"${MODEL_ID}","instructions":"Answer in six words max.","input":"Say hello to AX.","max_output_tokens":8,"temperature":0.0,"top_k":1,"seed":7}
JSON
)")"
echo "${RESPONSES_JSON}"
assert_json "${RESPONSES_JSON}" responses

echo "== legacy completion =="
COMPLETION_JSON="$(request_json POST /completion "$(cat <<JSON
{"model":"${MODEL_ID}","prompt":"Hello AX","n_predict":8,"temperature":0.0,"top_k":1,"seed":7}
JSON
)")"
echo "${COMPLETION_JSON}"
assert_json "${COMPLETION_JSON}" completion

echo "compat smoke passed"
