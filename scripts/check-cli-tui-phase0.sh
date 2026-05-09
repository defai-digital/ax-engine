#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"
TMP_DIR="$(ax_tmp_dir ax-cli-tui-phase0)"
SERVER_PORT="$(ax_allocate_port)"
SERVER_LOG="$TMP_DIR/server.log"
SERVER_PID=""

cleanup() {
    ax_kill_pid "$SERVER_PID"
    ax_rm_rf "$TMP_DIR"
}

trap cleanup EXIT

cd "$ROOT_DIR"

MODEL_DIR="$TMP_DIR/model"
mkdir -p "$MODEL_DIR"
cat >"$MODEL_DIR/config.json" <<'JSON'
{
  "model_type": "qwen3",
  "quantization": {
    "mode": "affine",
    "group_size": 64,
    "bits": 4
  }
}
JSON
printf '{}\n' >"$MODEL_DIR/model-manifest.json"
printf 'placeholder\n' >"$MODEL_DIR/model.safetensors"

"$PYTHON_BIN" scripts/download_model.py phase0/local-ready --dest "$MODEL_DIR" --json \
  >"$TMP_DIR/download.json"
cargo run --quiet -p ax-engine-bench -- generate-manifest "$MODEL_DIR" --json \
  >"$TMP_DIR/generate-manifest.json"
cargo run --quiet -p ax-engine-bench -- doctor --json --mlx-model-artifacts-dir "$MODEL_DIR" \
  >"$TMP_DIR/doctor.json"

"$PYTHON_BIN" - <<'PY' "$ROOT_DIR/benchmarks/manifests/scenario/chat_qwen_short.json" "$TMP_DIR/contract-failure-manifest.json"
from __future__ import annotations

import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
dest = Path(sys.argv[2])
manifest = json.loads(source.read_text(encoding="utf-8"))
manifest["shape"]["concurrency"] = 0
dest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

set +e
cargo run --quiet -p ax-engine-bench -- scenario \
  --manifest "$TMP_DIR/contract-failure-manifest.json" \
  --output-root "$TMP_DIR/bench-results" \
  --json >"$TMP_DIR/benchmark-artifact.json" 2>"$TMP_DIR/benchmark-artifact.err"
BENCH_STATUS="$?"
set -e
if [[ "$BENCH_STATUS" -ne 2 ]]; then
    cat "$TMP_DIR/benchmark-artifact.err" >&2
    echo "expected contract-failure benchmark to exit with status 2, got $BENCH_STATUS" >&2
    exit 1
fi

cargo run --quiet -p ax-engine-server -- \
  --host 127.0.0.1 \
  --port "$SERVER_PORT" \
  --llama-server-url http://127.0.0.1:9 \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"

"$PYTHON_BIN" - <<'PY' "$TMP_DIR" "$SERVER_PORT"
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


tmp = Path(sys.argv[1])
port = int(sys.argv[2])


def load(name: str) -> dict:
    return json.loads((tmp / name).read_text(encoding="utf-8"))


download = load("download.json")
assert download["schema_version"] == "ax.download_model.v1", download
assert download["status"] == "ready", download
assert download["manifest_present"] is True, download
assert download["config_present"] is True, download
assert download["safetensors_count"] == 1, download
assert download["server_command"][0] == "ax-engine-server", download

generated = load("generate-manifest.json")
assert generated["schema_version"] == "ax.generate_manifest.v1", generated
assert generated["status"] == "already_exists", generated
assert generated["manifest_present"] is True, generated

doctor = load("doctor.json")
assert doctor["schema_version"] == "ax.engine_bench.doctor.v1", doctor
assert doctor["workflow"]["mode"] == "source_checkout", doctor
assert doctor["workflow"]["doctor"]["argv"][-1] == "--json", doctor
assert doctor["model_artifacts"]["status"] == "ready", doctor
assert doctor["model_artifacts"]["config_present"] is True, doctor
assert doctor["model_artifacts"]["manifest_present"] is True, doctor
assert doctor["model_artifacts"]["safetensors_present"] is True, doctor
assert doctor["model_artifacts"]["model_type"] == "qwen3", doctor
assert doctor["model_artifacts"]["quantization"]["bits"] == 4, doctor
assert doctor["model_artifacts"]["issues"] == [], doctor

benchmark = load("benchmark-artifact.json")
assert benchmark["schema_version"] == "ax.benchmark_artifact.v1", benchmark
assert benchmark["command"] == "scenario", benchmark
assert benchmark["status"] == "contract_failure", benchmark
assert Path(benchmark["result_dir"]).is_dir(), benchmark
assert (Path(benchmark["result_dir"]) / "contract_failure.json").is_file(), benchmark


def fetch_json(path: str) -> dict:
    url = f"http://127.0.0.1:{port}{path}"
    deadline = time.time() + 30
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=1) as response:  # noqa: S310 - local smoke check
                return json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(0.2)
    raise AssertionError(f"server endpoint {path} did not become ready: {last_error}")


health = fetch_json("/health")
assert health["status"] == "ok", health
assert health["service"] == "ax-engine-server", health
assert isinstance(health["runtime"], dict), health

runtime = fetch_json("/v1/runtime")
assert runtime["service"] == "ax-engine-server", runtime
assert runtime["runtime"]["selected_backend"] == "llama_cpp", runtime

models = fetch_json("/v1/models")
assert models["object"] == "list", models
assert models["data"], models
assert models["data"][0]["object"] == "model", models
assert isinstance(models["data"][0]["runtime"], dict), models
PY

echo "CLI TUI Phase 0 contracts verified."
