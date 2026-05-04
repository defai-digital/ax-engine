#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"
UPSTREAM_LOG_FILE="$(ax_tmp_file ax-engine-bench-upstream-check .log)"
UPSTREAM_REQUEST_LOG_FILE="$(ax_tmp_file ax-engine-bench-upstream-requests .jsonl)"
TMP_DIR="$(ax_tmp_dir ax-engine-bench-check)"

UPSTREAM_PORT="$(ax_allocate_port)"
UPSTREAM_PID=""

cleanup() {
    ax_kill_pid "$UPSTREAM_PID"
    ax_rm_rf "$TMP_DIR" "$UPSTREAM_LOG_FILE" "$UPSTREAM_REQUEST_LOG_FILE"
}

trap cleanup EXIT

cd "$ROOT_DIR"

AX_ENGINE_LLAMA_CPP_UPSTREAM_PORT="$UPSTREAM_PORT" \
AX_ENGINE_LLAMA_CPP_UPSTREAM_REQUEST_LOG_FILE="$UPSTREAM_REQUEST_LOG_FILE" \
"$PYTHON_BIN" - <<'PY' >"$UPSTREAM_LOG_FILE" 2>&1 &
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PORT = int(os.environ["AX_ENGINE_LLAMA_CPP_UPSTREAM_PORT"])
REQUEST_LOG_FILE = os.environ["AX_ENGINE_LLAMA_CPP_UPSTREAM_REQUEST_LOG_FILE"]


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/completion":
            self.send_error(404)
            return

        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        prompt = payload.get("prompt")

        with open(REQUEST_LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload) + "\n")

        if payload.get("stream"):
            prompt_len = len(prompt) if isinstance(prompt, list) else 0
            if payload.get("n_predict") == 8:
                cache_tokens = 24 if prompt_len == 320 else 0
                body = (
                    "data: "
                    + json.dumps(
                        {
                            "content": "",
                            "tokens": [],
                            "stop": False,
                            "prompt_progress": {
                                "total": prompt_len,
                                "cache": cache_tokens,
                                "processed": max(prompt_len - cache_tokens, 0),
                                "time_ms": 1.0,
                            },
                        },
                        separators=(",", ":"),
                    )
                    + "\n\n"
                    "data: "
                    + json.dumps(
                        {
                            "content": "llama reuse",
                            "tokens": [131],
                            "stop": True,
                            "stop_type": "limit",
                        },
                        separators=(",", ":"),
                    )
                    + "\n\n"
                    "data: [DONE]\n\n"
                ).encode("utf-8")
            elif payload.get("n_predict") == 2:
                body = (
                    'data: {"content":"llama replay","tokens":[111],"stop":false}\n\n'
                    'data: {"content":" done","tokens":[112],"stop":true,"stop_type":"limit"}\n\n'
                    "data: [DONE]\n\n"
                ).encode("utf-8")
            elif prompt_len == 128:
                request_index = getattr(self.server, "shared_prefix_request_index", 0)
                self.server.shared_prefix_request_index = request_index + 1
                cache_tokens = 64 if request_index % 2 == 1 else 0
                body = (
                    "data: "
                    + json.dumps(
                        {
                            "content": "",
                            "tokens": [],
                            "stop": False,
                            "prompt_progress": {
                                "total": prompt_len,
                                "cache": cache_tokens,
                                "processed": max(prompt_len - cache_tokens, 0),
                                "time_ms": 1.0,
                            },
                        },
                        separators=(",", ":"),
                    )
                    + "\n\n"
                    "data: "
                    + json.dumps(
                        {
                            "content": "llama shared",
                            "tokens": [151, 152, 153, 154],
                            "stop": True,
                            "stop_type": "limit",
                        },
                        separators=(",", ":"),
                    )
                    + "\n\n"
                    "data: [DONE]\n\n"
                ).encode("utf-8")
            else:
                body = (
                    'data: {"content":"llama","tokens":[91,92],"stop":false}\n\n'
                    'data: {"content":" scenario","tokens":[93,94],"stop":true,"stop_type":"limit"}\n\n'
                    "data: [DONE]\n\n"
                ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(400)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
server.serve_forever()
PY
UPSTREAM_PID="$!"

AX_BENCH_TMP_DIR="$TMP_DIR" \
AX_BENCH_LLAMA_CPP_SERVER_URL="http://127.0.0.1:${UPSTREAM_PORT}" \
AX_BENCH_UPSTREAM_REQUEST_LOG_FILE="$UPSTREAM_REQUEST_LOG_FILE" \
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

from collections import Counter
import json
import os
import subprocess
from pathlib import Path


root = Path(os.environ["AX_BENCH_TMP_DIR"])
server_url = os.environ["AX_BENCH_LLAMA_CPP_SERVER_URL"]
request_log_file = Path(os.environ["AX_BENCH_UPSTREAM_REQUEST_LOG_FILE"])
repo = Path.cwd()

scenario_template = repo / "benchmarks/manifests/scenario/llama_cpp_chat_qwen_short.json"
shared_prefix_template = repo / "benchmarks/manifests/scenario/llama_cpp_shared_prefix_qwen_short.json"
replay_cancel_template = repo / "benchmarks/manifests/replay/llama_cpp_submit_cancel_dual.json"
replay_reuse_template = repo / "benchmarks/manifests/replay/llama_cpp_prompt_cache_reuse_dual.json"
scenario_manifest = root / "llama_cpp_chat_qwen_short.json"
shared_prefix_manifest = root / "llama_cpp_shared_prefix_qwen_short.json"
replay_cancel_manifest = root / "llama_cpp_submit_cancel_dual.json"
replay_reuse_manifest = root / "llama_cpp_prompt_cache_reuse_dual.json"
missing_adapter_manifest = root / "llama_cpp_missing_adapter.json"
scenario_output = root / "scenario-results"
shared_prefix_output = root / "shared-prefix-results"
replay_cancel_output = root / "replay-cancel-results"
replay_reuse_output = root / "replay-reuse-results"
compare_output = root / "compare-results"
failure_output = root / "failure-results"

for template_path, target_path in [
    (scenario_template, scenario_manifest),
    (shared_prefix_template, shared_prefix_manifest),
    (replay_cancel_template, replay_cancel_manifest),
    (replay_reuse_template, replay_reuse_manifest),
]:
    payload = json.loads(template_path.read_text())
    payload["runtime"]["backend_adapter"]["server_url"] = server_url
    target_path.write_text(json.dumps(payload, indent=2) + "\n")

missing_adapter_payload = json.loads(scenario_template.read_text())
missing_adapter_payload["runtime"].pop("backend_adapter", None)
missing_adapter_manifest.write_text(json.dumps(missing_adapter_payload, indent=2) + "\n")

for manifest_path, output_root, command in [
    (scenario_manifest, scenario_output, "scenario"),
    (shared_prefix_manifest, shared_prefix_output, "scenario"),
    (replay_cancel_manifest, replay_cancel_output, "replay"),
    (replay_reuse_manifest, replay_reuse_output, "replay"),
]:
    output_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cargo",
            "run",
            "-p",
            "ax-engine-bench",
            "--",
            command,
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
        ],
        check=True,
        cwd=repo,
    )


def load_single_run(output_root: Path) -> tuple[Path, dict, dict, dict]:
    runs = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(runs) == 1, f"expected exactly one run directory under {output_root}, found {runs}"
    run_dir = runs[0]
    environment = json.loads((run_dir / "environment.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    routes = json.loads((run_dir / "routes.json").read_text())
    trace = json.loads((run_dir / "trace.json").read_text())
    return run_dir, environment, metrics, routes | {"trace": trace}


scenario_run_dir, scenario_environment, scenario_metrics, scenario_artifacts = load_single_run(scenario_output)
assert scenario_environment["software"]["tool_mode"] == "llama_cpp_stepwise_runtime"
assert scenario_environment["runtime"]["backend_adapter"]["kind"] == "llama_cpp_server_completion"
assert scenario_environment["route"]["prefix_cache_evidence"] == "none_observed"
assert scenario_environment["route"]["prefix_reuse_provenance"] == "none_observed"
assert scenario_metrics["correctness"]["passed"] is True
assert scenario_metrics["determinism"]["passed"] is True
assert scenario_metrics["replay_status"] == "not_applicable"
assert scenario_metrics["step_count"] == 2
assert scenario_artifacts["route"]["execution_plan"] == "llama_cpp.server_completion_stream"
assert len(scenario_artifacts["trace"]["steps"]) == 2

shared_prefix_run_dir, shared_prefix_environment, shared_prefix_metrics, shared_prefix_artifacts = load_single_run(shared_prefix_output)
assert shared_prefix_environment["software"]["tool_mode"] == "llama_cpp_stepwise_runtime"
assert shared_prefix_environment["route"]["prefix_cache_path"] == "delegated_prompt_cache"
assert shared_prefix_environment["route"]["prefix_cache_evidence"] == "backend_reported_cached_prompt_tokens"
assert shared_prefix_environment["route"]["prefix_reuse_provenance"] == "delegated_backend_prompt_cache"
assert shared_prefix_environment["route"]["backend_reported_cached_prompt_tokens"] == 64
assert shared_prefix_metrics["correctness"]["passed"] is True
assert shared_prefix_metrics["determinism"]["passed"] is True
assert shared_prefix_metrics["metrics"]["prefix_hit_rate"] > 0.0
assert shared_prefix_metrics["step_count"] == 2
assert shared_prefix_artifacts["route"]["execution_plan"] == "llama_cpp.server_completion_stream"
assert shared_prefix_artifacts["route"]["prefix_cache_path"] == "delegated_prompt_cache"
assert shared_prefix_artifacts["route"]["prefix_cache_evidence"] == "backend_reported_cached_prompt_tokens"
assert shared_prefix_artifacts["route"]["prefix_reuse_provenance"] == "delegated_backend_prompt_cache"
assert shared_prefix_artifacts["route"]["backend_reported_cached_prompt_tokens"] == 64
assert len(shared_prefix_artifacts["trace"]["steps"]) == 2
shared_prefix_summary = (shared_prefix_run_dir / "summary.md").read_text()
assert "delegated_backend_prompt_cache" in shared_prefix_summary
assert "backend_reported_cached_prompt_tokens: `64`" in shared_prefix_summary

replay_run_dir, replay_environment, replay_metrics, replay_artifacts = load_single_run(replay_cancel_output)
assert replay_environment["software"]["tool_mode"] == "llama_cpp_stepwise_runtime"
assert replay_environment["route"]["prefix_cache_evidence"] == "none_observed"
assert replay_environment["route"]["prefix_reuse_provenance"] == "none_observed"
assert replay_metrics["correctness"]["passed"] is True
assert replay_metrics["determinism"]["passed"] is True
assert replay_metrics["replay_status"] == "pass"
assert replay_metrics["churn_status"] == "pass"
assert replay_metrics["step_count"] == 2
assert replay_artifacts["route"]["execution_plan"] == "llama_cpp.server_completion_stream"
assert len(replay_artifacts["trace"]["steps"]) == 2

summary = (replay_run_dir / "summary.md").read_text()
assert "llama_cpp_stepwise_runtime" in summary
assert "llama_cpp_server_completion" in summary

replay_reuse_run_dir, replay_reuse_environment, replay_reuse_metrics, replay_reuse_artifacts = load_single_run(replay_reuse_output)
assert replay_reuse_environment["software"]["tool_mode"] == "llama_cpp_stepwise_runtime"
assert replay_reuse_environment["route"]["prefix_cache_path"] == "delegated_prompt_cache"
assert replay_reuse_environment["route"]["prefix_cache_evidence"] == "backend_reported_cached_prompt_tokens"
assert replay_reuse_environment["route"]["prefix_reuse_provenance"] == "delegated_backend_prompt_cache"
assert replay_reuse_metrics["correctness"]["passed"] is True
assert replay_reuse_metrics["determinism"]["passed"] is True
assert replay_reuse_metrics["replay_status"] == "not_applicable"
assert replay_reuse_metrics["churn_status"] == "pass"
assert replay_reuse_metrics["metrics"]["prefix_hit_rate"] > 0.0
assert replay_reuse_metrics["step_count"] >= 2
assert replay_reuse_artifacts["route"]["execution_plan"] == "llama_cpp.server_completion_stream"
assert replay_reuse_artifacts["route"]["prefix_cache_path"] == "delegated_prompt_cache"
assert replay_reuse_artifacts["route"]["prefix_cache_evidence"] == "backend_reported_cached_prompt_tokens"
assert replay_reuse_artifacts["route"]["prefix_reuse_provenance"] == "delegated_backend_prompt_cache"
assert len(replay_reuse_artifacts["trace"]["steps"]) >= 2

summary = (replay_reuse_run_dir / "summary.md").read_text()
assert "llama_cpp_stepwise_runtime" in summary
assert "delegated_backend_prompt_cache" in summary

compare_output.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        "cargo",
        "run",
        "-p",
        "ax-engine-bench",
        "--",
        "compare",
        "--baseline",
        str(replay_reuse_run_dir),
        "--candidate",
        str(replay_reuse_run_dir),
        "--output-root",
        str(compare_output),
    ],
    check=True,
    cwd=repo,
)

compare_runs = [path for path in compare_output.iterdir() if path.is_dir()]
assert len(compare_runs) == 1, f"expected exactly one compare directory under {compare_output}, found {compare_runs}"
compare_run_dir = compare_runs[0]
comparison_summary = (compare_run_dir / "comparison.md").read_text()
regression = json.loads((compare_run_dir / "regression.json").read_text())
assert "mode: `llama_cpp_stepwise_compare`" in comparison_summary
assert "tool_mode: `llama_cpp_stepwise_runtime`" in comparison_summary
assert "selected_backend: `llama_cpp`" in comparison_summary
assert "prefix_cache_path: `delegated_prompt_cache`" in comparison_summary
assert "prefix_cache_evidence: `backend_reported_cached_prompt_tokens`" in comparison_summary
assert "prefix_reuse_provenance: `delegated_backend_prompt_cache`" in comparison_summary
assert regression["runtime"]["tool_mode"] == "llama_cpp_stepwise_runtime"
assert regression["runtime"]["selected_backend"] == "llama_cpp"
assert regression["summary"]["result"] == "llama_cpp_stepwise_compare"
assert regression["summary"]["prefix_cache_path"] == "delegated_prompt_cache"
assert regression["summary"]["prefix_cache_evidence"] == "backend_reported_cached_prompt_tokens"
assert regression["summary"]["prefix_reuse_provenance"] == "delegated_backend_prompt_cache"
assert regression["summary"]["backend_reported_cached_prompt_tokens"] == 24
assert regression["contract"]["prefix_cache_path"]["baseline"] == "delegated_prompt_cache"
assert regression["contract"]["prefix_cache_evidence"]["baseline"] == "backend_reported_cached_prompt_tokens"
assert regression["contract"]["prefix_reuse_provenance"]["baseline"] == "delegated_backend_prompt_cache"
assert regression["contract"]["backend_reported_cached_prompt_tokens"]["baseline"] == 24

failure_output.mkdir(parents=True, exist_ok=True)
failure_result = subprocess.run(
    [
        "cargo",
        "run",
        "-p",
        "ax-engine-bench",
        "--",
        "scenario",
        "--manifest",
        str(missing_adapter_manifest),
        "--output-root",
        str(failure_output),
    ],
    check=False,
    capture_output=True,
    text=True,
    cwd=repo,
)
assert failure_result.returncode == 2, failure_result
assert "status=contract_failure" in failure_result.stdout
assert "artifact_dir=" in failure_result.stderr
failure_runs = [path for path in failure_output.iterdir() if path.is_dir()]
assert len(failure_runs) == 1, f"expected exactly one failure directory under {failure_output}, found {failure_runs}"
failure_run_dir = failure_runs[0]
failure = json.loads((failure_run_dir / "contract_failure.json").read_text())
failure_summary = (failure_run_dir / "summary.md").read_text()
assert failure["status"] == "contract_failure"
assert failure["tool_mode"] == "llama_cpp_blocking_runtime"
assert failure["failure"]["code"] == "llama_backend_adapter_required"
assert "llama_backend_adapter_required" in failure_summary
assert not (failure_run_dir / "metrics.json").exists()

requests = [
    json.loads(line)
    for line in request_log_file.read_text().splitlines()
    if line.strip()
]
assert len(requests) == 14, f"expected 14 upstream requests, found {len(requests)}"
assert all(request["stream"] is True for request in requests)
assert all(request["return_tokens"] is True for request in requests)
assert all(request["return_progress"] is True for request in requests)
assert all(isinstance(request["prompt"], list) for request in requests)
scenario_prompts = [
    request["prompt"] for request in requests
    if request["n_predict"] == 4 and len(request["prompt"]) == 32
]
shared_prefix_prompts = [
    request["prompt"] for request in requests
    if request["n_predict"] == 4 and len(request["prompt"]) == 128
]
replay_cancel_prompts = [request["prompt"] for request in requests if request["n_predict"] == 2]
replay_reuse_prompts = [request["prompt"] for request in requests if request["n_predict"] == 8]
assert len(scenario_prompts) == 2
assert len(shared_prefix_prompts) == 4
assert len(replay_cancel_prompts) == 4
assert len(replay_reuse_prompts) == 4
assert all(prompt == scenario_prompts[0] for prompt in scenario_prompts)
shared_prefix_counts = Counter(tuple(prompt) for prompt in shared_prefix_prompts)
assert sorted(shared_prefix_counts.values()) == [2, 2]
shared_prefix_variants = sorted(shared_prefix_counts)
assert len(shared_prefix_variants[0]) == 128
assert len(shared_prefix_variants[1]) == 128
assert shared_prefix_variants[0][:64] == shared_prefix_variants[1][:64]
assert shared_prefix_variants[0][64:] != shared_prefix_variants[1][64:]
replay_cancel_counts = Counter(tuple(prompt) for prompt in replay_cancel_prompts)
replay_reuse_counts = Counter(tuple(prompt) for prompt in replay_reuse_prompts)
assert sorted(replay_cancel_counts.values()) == [2, 2]
assert sorted(replay_reuse_counts.values()) == [2, 2]
replay_reuse_variants = sorted(replay_reuse_counts, key=len)
assert len(replay_reuse_variants[0]) == 256
assert len(replay_reuse_variants[1]) == 320
assert replay_reuse_variants[0][:64] == replay_reuse_variants[1][:64]
assert replay_reuse_variants[0][64:] != replay_reuse_variants[1][64:]
PY
