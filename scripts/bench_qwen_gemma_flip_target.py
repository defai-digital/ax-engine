#!/usr/bin/env python3
"""Replay Qwen/Gemma flip scenarios against AX or managed mlxcel processes.

The benchmark uses the raw OpenAI completions streaming surface for both
runtimes. A target JSON maps the logical model ids in the shared S0-S3
manifests to one AX endpoint or to one managed mlxcel process per model. The
result keeps the existing ``ax.multimodel_serving_benchmark.v1`` schema so the
same checker and flip comparator can validate both sides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bench_ax_multimodel_serving as multimodel
import bench_ax_serving as serving

TARGET_SCHEMA_VERSION = "ax.qwen_gemma_flip_target.v1"
OPENAI_ENDPOINT = "/v1/completions"


@dataclass(frozen=True)
class ModelTarget:
    model_id: str
    served_model: str
    base_url: str
    model_path: Path | None
    port: int | None
    args: tuple[str, ...]
    env: dict[str, str]
    memory_cap_bytes: int | None


@dataclass(frozen=True)
class TargetSpec:
    path: Path
    name: str
    runtime: str
    runtime_revision: str
    official_source: str | None
    managed_processes: bool
    managed_single_process: bool
    primary_model_id: str | None
    binary: str | None
    host: str
    health_path: str
    startup_timeout_s: float
    shutdown_timeout_s: float
    common_args: tuple[str, ...]
    env: dict[str, str]
    models: dict[str, ModelTarget]
    comparison_contract: dict[str, Any]


def _non_empty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{field} must be a non-empty string")
    return value


def _string_list(value: Any, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SystemExit(f"{field} must be an array of strings")
    return tuple(value)


def _string_map(value: Any, *, field: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise SystemExit(f"{field} must be an object of string values")
    return dict(value)


def _resolve_env_or_value(
    raw: dict[str, Any],
    *,
    value_key: str,
    env_key: str,
    field: str,
    required: bool,
) -> str | None:
    value = raw.get(value_key)
    env_name = raw.get(env_key)
    if value is not None:
        return _non_empty_string(value, field=field)
    if env_name is not None:
        env_name = _non_empty_string(env_name, field=f"{field} environment name")
        resolved = os.environ.get(env_name)
        if resolved:
            return resolved
        if required:
            raise SystemExit(f"{field} requires environment variable {env_name}")
    if required:
        raise SystemExit(f"{field} requires {value_key} or {env_key}")
    return None


def load_target(path: Path) -> TargetSpec:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read target {path}: {error}") from error
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: target must be a JSON object")
    if raw.get("schema_version") != TARGET_SCHEMA_VERSION:
        raise SystemExit(f"{path}: schema_version must be {TARGET_SCHEMA_VERSION!r}")

    name = _non_empty_string(raw.get("name"), field="name")
    runtime = _non_empty_string(raw.get("runtime"), field="runtime")
    if runtime not in {"ax-engine", "mlxcel"}:
        raise SystemExit("runtime must be 'ax-engine' or 'mlxcel'")
    runtime_revision = _resolve_env_or_value(
        raw,
        value_key="runtime_revision",
        env_key="runtime_revision_env",
        field="runtime_revision",
        required=True,
    )
    assert runtime_revision is not None
    official_source = raw.get("official_source")
    if official_source is not None:
        official_source = _non_empty_string(official_source, field="official_source")

    managed = raw.get("managed_processes", False)
    if not isinstance(managed, bool):
        raise SystemExit("managed_processes must be a boolean")
    managed_single = raw.get("managed_single_process", False)
    if not isinstance(managed_single, bool):
        raise SystemExit("managed_single_process must be a boolean")
    if managed and managed_single:
        raise SystemExit("only one process-management mode may be enabled")
    binary = _resolve_env_or_value(
        raw,
        value_key="binary",
        env_key="binary_env",
        field="binary",
        required=managed or managed_single,
    )
    if binary is not None:
        resolved_binary = shutil.which(binary)
        if resolved_binary is None and Path(binary).is_file():
            resolved_binary = str(Path(binary).resolve())
        if resolved_binary is None:
            raise SystemExit(f"target binary does not exist or is not on PATH: {binary}")
        binary = resolved_binary

    host = _non_empty_string(raw.get("host", "127.0.0.1"), field="host")
    health_path = _non_empty_string(raw.get("health_path", "/health"), field="health_path")
    if not health_path.startswith("/"):
        raise SystemExit("health_path must start with '/'")
    startup_timeout_s = float(raw.get("startup_timeout_s", 600.0))
    shutdown_timeout_s = float(raw.get("shutdown_timeout_s", 30.0))
    if startup_timeout_s <= 0 or shutdown_timeout_s <= 0:
        raise SystemExit("startup_timeout_s and shutdown_timeout_s must be positive")

    model_rows = raw.get("models")
    if not isinstance(model_rows, dict) or not model_rows:
        raise SystemExit("models must be a non-empty object")
    models: dict[str, ModelTarget] = {}
    for model_id, model_raw in model_rows.items():
        if not isinstance(model_id, str) or not model_id:
            raise SystemExit("models keys must be non-empty strings")
        if not isinstance(model_raw, dict):
            raise SystemExit(f"models.{model_id} must be an object")
        served_model = _non_empty_string(
            model_raw.get("served_model", model_id),
            field=f"models.{model_id}.served_model",
        )
        port_raw = model_raw.get("port")
        port: int | None = None
        if port_raw is not None:
            if not isinstance(port_raw, int) or not 0 < port_raw < 65536:
                raise SystemExit(f"models.{model_id}.port must be in 1..65535")
            port = port_raw
        base_url = model_raw.get("base_url")
        if base_url is None and port is not None:
            base_url = f"http://{host}:{port}"
        base_url = _non_empty_string(base_url, field=f"models.{model_id}.base_url").rstrip("/")
        model_path_value = _resolve_env_or_value(
            model_raw,
            value_key="model_path",
            env_key="model_path_env",
            field=f"models.{model_id}.model_path",
            required=managed or managed_single,
        )
        model_path = Path(model_path_value).resolve() if model_path_value else None
        if model_path is not None and not model_path.is_dir():
            raise SystemExit(f"model directory does not exist: {model_path}")
        memory_cap = model_raw.get("memory_cap_bytes")
        if memory_cap is not None and (not isinstance(memory_cap, int) or memory_cap <= 0):
            raise SystemExit(f"models.{model_id}.memory_cap_bytes must be positive")
        models[model_id] = ModelTarget(
            model_id=model_id,
            served_model=served_model,
            base_url=base_url,
            model_path=model_path,
            port=port,
            args=_string_list(model_raw.get("args"), field=f"models.{model_id}.args"),
            env=_string_map(model_raw.get("env"), field=f"models.{model_id}.env"),
            memory_cap_bytes=memory_cap,
        )

    if managed and any(model.port is None for model in models.values()):
        raise SystemExit("every managed model requires a port")

    primary_model_id = raw.get("primary_model_id")
    if primary_model_id is not None:
        primary_model_id = _non_empty_string(primary_model_id, field="primary_model_id")
    if managed_single:
        if runtime != "ax-engine":
            raise SystemExit("managed_single_process is supported only for ax-engine")
        if primary_model_id not in models:
            raise SystemExit("managed_single_process requires primary_model_id in models")
        if len({model.base_url for model in models.values()}) != 1:
            raise SystemExit("managed_single_process models must share one base_url")

    target_env = _string_map(raw.get("env"), field="env")
    if not managed and not managed_single:
        mismatched_env = [
            key for key, value in sorted(target_env.items()) if os.environ.get(key) != value
        ]
        if mismatched_env:
            raise SystemExit(
                "unmanaged target requires matching parent/server environment values for: "
                + ", ".join(mismatched_env)
            )

    contract = raw.get("comparison_contract", {})
    if not isinstance(contract, dict):
        raise SystemExit("comparison_contract must be an object")
    total_memory_cap = contract.get("total_memory_cap_bytes")
    if not isinstance(total_memory_cap, int) or total_memory_cap <= 0:
        raise SystemExit("comparison_contract.total_memory_cap_bytes must be positive")
    model_caps = [model.memory_cap_bytes for model in models.values()]
    if any(cap is None for cap in model_caps):
        raise SystemExit("every model requires memory_cap_bytes")
    resolved_caps = [int(cap) for cap in model_caps if cap is not None]
    if managed and sum(resolved_caps) != total_memory_cap:
        raise SystemExit(
            "managed model memory caps must sum to comparison_contract.total_memory_cap_bytes"
        )
    if not managed and any(cap != total_memory_cap for cap in resolved_caps):
        raise SystemExit(
            "unmanaged single-process model caps must equal "
            "comparison_contract.total_memory_cap_bytes"
        )
    return TargetSpec(
        path=path.resolve(),
        name=name,
        runtime=runtime,
        runtime_revision=runtime_revision,
        official_source=official_source,
        managed_processes=managed,
        managed_single_process=managed_single,
        primary_model_id=primary_model_id,
        binary=binary,
        host=host,
        health_path=health_path,
        startup_timeout_s=startup_timeout_s,
        shutdown_timeout_s=shutdown_timeout_s,
        common_args=_string_list(raw.get("common_args"), field="common_args"),
        env=target_env,
        models=models,
        comparison_contract=contract,
    )


def model_package_identity(model_path: Path | None) -> dict[str, Any] | None:
    if model_path is None:
        return None
    identity_files = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "weights_manifest.json",
    )
    file_hashes: dict[str, str] = {}
    for name in identity_files:
        candidate = model_path / name
        if candidate.is_file():
            file_hashes[name] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    weights = sorted(model_path.glob("*.safetensors"))
    weight_layout = [
        {"name": weight.name, "size_bytes": weight.stat().st_size} for weight in weights
    ]
    encoded_layout = json.dumps(weight_layout, separators=(",", ":"), sort_keys=True).encode()
    return {
        "path": str(model_path),
        "identity_file_sha256": file_hashes,
        "safetensors_files": len(weights),
        "safetensors_bytes": sum(row["size_bytes"] for row in weight_layout),
        "safetensors_layout_sha256": hashlib.sha256(encoded_layout).hexdigest(),
    }


class ProcessSupervisor:
    def __init__(self, target: TargetSpec, log_dir: Path) -> None:
        self.target = target
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._logs: dict[str, Any] = {}
        self._attempts: dict[str, int] = {}
        self._audit: list[dict[str, Any]] = []

    def command_for(self, model: ModelTarget) -> list[str]:
        if self.target.binary is None or model.model_path is None or model.port is None:
            raise RuntimeError(f"managed process configuration is incomplete for {model.model_id}")
        return [
            self.target.binary,
            "-m",
            str(model.model_path),
            "--alias",
            model.served_model,
            "--host",
            self.target.host,
            "--port",
            str(model.port),
            *self.target.common_args,
            *model.args,
        ]

    def start(self, model_id: str) -> tuple[bool, dict[str, Any]]:
        model = self.target.models.get(model_id)
        if model is None:
            return False, {"error": f"unknown target model {model_id}"}
        with self._lock:
            existing = self._processes.get(model_id)
            if existing is not None and existing.poll() is None:
                return False, {"error": f"{model_id} process is already running"}
            attempt = self._attempts.get(model_id, 0) + 1
            self._attempts[model_id] = attempt
            command = self.command_for(model)
            log_path = self.log_dir / f"{model_id}-start-{attempt}.log"
            log_handle = log_path.open("wb")
            environment = os.environ.copy()
            environment.update(self.target.env)
            environment.update(model.env)
            started = time.perf_counter()
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            self._processes[model_id] = process
            self._logs[model_id] = log_handle
            record = {
                "model_id": model_id,
                "attempt": attempt,
                "pid": process.pid,
                "command": command,
                "log_path": str(log_path),
                "start_latency_ms": None,
                "ready": False,
                "exit_code": None,
                "forced_kill": False,
            }
            self._audit.append(record)

        ready, error = self._wait_ready(model, process)
        latency_ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            record["start_latency_ms"] = latency_ms
            record["ready"] = ready
            if process.poll() is not None:
                record["exit_code"] = process.returncode
        if not ready:
            self.stop(model_id)
            return False, {
                "error": error,
                "pid": process.pid,
                "log_path": str(log_path),
                "latency_ms": latency_ms,
            }
        return True, {
            "pid": process.pid,
            "log_path": str(log_path),
            "latency_ms": latency_ms,
        }

    def _wait_ready(
        self, model: ModelTarget, process: subprocess.Popen[bytes]
    ) -> tuple[bool, str | None]:
        deadline = time.monotonic() + self.target.startup_timeout_s
        url = f"{model.base_url}{self.target.health_path}"
        last_error: str | None = None
        while time.monotonic() < deadline:
            exit_code = process.poll()
            if exit_code is not None:
                return False, f"process exited before readiness with code {exit_code}"
            try:
                request = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(request, timeout=2.0) as response:
                    if 200 <= response.status < 300:
                        return True, None
                    last_error = f"health status {response.status}"
            except (OSError, urllib.error.URLError) as error:
                last_error = str(error)
            time.sleep(0.25)
        return False, f"health check timed out: {last_error or 'no response'}"

    def stop(self, model_id: str) -> tuple[bool, dict[str, Any]]:
        with self._lock:
            process = self._processes.pop(model_id, None)
            log_handle = self._logs.pop(model_id, None)
            record = next(
                (
                    item
                    for item in reversed(self._audit)
                    if item["model_id"] == model_id and item["exit_code"] is None
                ),
                None,
            )
        if process is None:
            return False, {"error": f"{model_id} process is not running"}
        started = time.perf_counter()
        forced = False
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self.target.shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                forced = True
                process.kill()
                process.wait(timeout=self.target.shutdown_timeout_s)
        if log_handle is not None:
            log_handle.close()
        latency_ms = (time.perf_counter() - started) * 1000.0
        if record is not None:
            record["exit_code"] = process.returncode
            record["forced_kill"] = forced
            record["stop_latency_ms"] = latency_ms
        return True, {
            "exit_code": process.returncode,
            "forced_kill": forced,
            "latency_ms": latency_ms,
        }

    def stop_all(self) -> None:
        with self._lock:
            model_ids = list(self._processes)
        for model_id in model_ids:
            self.stop(model_id)

    def audit(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._audit]


class SingleProcessSupervisor:
    def __init__(self, target: TargetSpec, log_dir: Path) -> None:
        self.target = target
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._process: subprocess.Popen[bytes] | None = None
        self._log: Any = None
        self._audit: dict[str, Any] | None = None

    def command(self) -> list[str]:
        primary_id = self.target.primary_model_id
        if self.target.binary is None or primary_id is None:
            raise RuntimeError("managed single-process configuration is incomplete")
        model = self.target.models[primary_id]
        if model.model_path is None:
            raise RuntimeError(f"managed model path is missing for {primary_id}")
        return [
            self.target.binary,
            "--model-id",
            model.model_id,
            "--mlx",
            "--mlx-model-artifacts-dir",
            str(model.model_path),
            *self.target.common_args,
            *model.args,
        ]

    def start(self) -> tuple[bool, dict[str, Any]]:
        if self._process is not None and self._process.poll() is None:
            return False, {"error": "single managed process is already running"}
        primary_id = self.target.primary_model_id
        assert primary_id is not None
        primary = self.target.models[primary_id]
        command = self.command()
        log_path = self.log_dir / "ax-single-process.log"
        self._log = log_path.open("wb")
        environment = os.environ.copy()
        environment.update(self.target.env)
        environment.update(primary.env)
        started = time.perf_counter()
        self._process = subprocess.Popen(
            command,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        self._audit = {
            "model_id": primary_id,
            "pid": self._process.pid,
            "command": command,
            "log_path": str(log_path),
            "start_latency_ms": None,
            "ready": False,
            "exit_code": None,
            "forced_kill": False,
            "initial_model_loads": [],
        }
        ready, error = self._wait_ready(primary)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._audit["start_latency_ms"] = latency_ms
        self._audit["ready"] = ready
        if self._process.poll() is not None:
            self._audit["exit_code"] = self._process.returncode
        if not ready:
            pid = self._process.pid
            self.stop()
            return False, {
                "error": error,
                "pid": pid,
                "log_path": str(log_path),
                "latency_ms": latency_ms,
            }
        return True, {
            "pid": self._process.pid,
            "log_path": str(log_path),
            "latency_ms": latency_ms,
        }

    def _wait_ready(self, model: ModelTarget) -> tuple[bool, str | None]:
        assert self._process is not None
        deadline = time.monotonic() + self.target.startup_timeout_s
        url = f"{model.base_url}{self.target.health_path}"
        last_error: str | None = None
        while time.monotonic() < deadline:
            exit_code = self._process.poll()
            if exit_code is not None:
                return False, f"process exited before readiness with code {exit_code}"
            try:
                request = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(request, timeout=2.0) as response:
                    if 200 <= response.status < 300:
                        return True, None
                    last_error = f"health status {response.status}"
            except (OSError, urllib.error.URLError) as error:
                last_error = str(error)
            time.sleep(0.25)
        return False, f"health check timed out: {last_error or 'no response'}"

    def load_model(self, model_id: str) -> tuple[bool, dict[str, Any]]:
        if self._process is None or self._process.poll() is not None:
            return False, {"error": "single managed process is not running"}
        model = self.target.models.get(model_id)
        if model is None or model.model_path is None:
            return False, {"error": f"managed model configuration is missing for {model_id}"}
        started = time.perf_counter()
        try:
            status, response = multimodel.post_json(
                f"{model.base_url}/v1/model/load",
                {
                    "model_id": model.model_id,
                    "model_path": str(model.model_path),
                    "load_mode": "add",
                    "load_policy": "availability_first",
                    "make_default": False,
                },
                self.target.startup_timeout_s,
            )
            error = None
        except Exception as caught:  # noqa: BLE001 - preserve setup failure in audit.
            status, response, error = None, None, str(caught)
        latency_ms = (time.perf_counter() - started) * 1000.0
        ok = status is not None and 200 <= status < 300 and error is None
        record = {
            "model_id": model_id,
            "status": status,
            "ok": ok,
            "error": error,
            "latency_ms": latency_ms,
            "response": response,
        }
        assert self._audit is not None
        self._audit["initial_model_loads"].append(record)
        return ok, record

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        started = time.perf_counter()
        forced = False
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self.target.shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                forced = True
                process.kill()
                process.wait(timeout=self.target.shutdown_timeout_s)
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._audit is not None:
            self._audit["exit_code"] = process.returncode
            self._audit["forced_kill"] = forced
            self._audit["stop_latency_ms"] = (time.perf_counter() - started) * 1000.0

    def audit(self) -> list[dict[str, Any]]:
        return [dict(self._audit)] if self._audit is not None else []


def openai_payload(
    prompt: serving.PromptItem,
    *,
    served_model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    if prompt.input_text is None:
        raise RuntimeError(
            f"prompt {prompt.id} has no input_text; fair OpenAI replay requires text"
        )
    return {
        "model": served_model,
        "prompt": prompt.input_text,
        "max_tokens": prompt.max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def observe_openai_stream(
    events: Iterable[tuple[str | None, Any, float]],
    *,
    prompt: serving.PromptItem,
    scheduled_at_s: float,
    started_at_s: float,
    completed_at_s: float,
) -> dict[str, Any]:
    status = 200
    error: str | None = None
    done = False
    finish_reason: str | None = None
    first_token_s: float | None = None
    chunk_times: list[float] = []
    text_parts: list[str] = []
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    event_count = 0

    for event_name, payload, elapsed_s in events:
        event_count += 1
        if event_name == "__http_status__" and isinstance(payload, dict):
            status = int(payload.get("status", status))
            if "error" in payload:
                error = str(payload["error"])
            continue
        if isinstance(payload, dict) and payload.get("done") is True:
            done = True
            continue
        if not isinstance(payload, dict):
            continue
        if "error" in payload:
            error = json.dumps(payload["error"], sort_keys=True)
        usage = payload.get("usage")
        if isinstance(usage, dict):
            if isinstance(usage.get("prompt_tokens"), int):
                prompt_tokens = int(usage["prompt_tokens"])
            if isinstance(usage.get("completion_tokens"), int):
                completion_tokens = int(usage["completion_tokens"])
        choices = payload.get("choices")
        if not isinstance(choices, list):
            continue
        emitted = False
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            reason = choice.get("finish_reason")
            if isinstance(reason, str):
                finish_reason = reason
            delta = choice.get("delta")
            pieces: list[str] = []
            if isinstance(delta, dict):
                for key in ("reasoning_content", "content"):
                    value = delta.get(key)
                    if isinstance(value, str) and value:
                        pieces.append(value)
            text = choice.get("text")
            if isinstance(text, str) and text:
                pieces.append(text)
            if pieces:
                emitted = True
                text_parts.extend(pieces)
        if emitted:
            chunk_times.append(elapsed_s)
            if first_token_s is None:
                first_token_s = elapsed_s

    if error is None and not done:
        error = "stream ended without [DONE]"
    if error is None and completion_tokens is None:
        error = "stream ended without authoritative completion_tokens usage"
    e2e_s = max(completed_at_s - started_at_s, 0.0)
    ttft_ms = first_token_s * 1000.0 if first_token_s is not None else None
    tpot_ms = None
    if first_token_s is not None and completion_tokens is not None and completion_tokens > 1:
        tpot_ms = max(
            (e2e_s - first_token_s) * 1000.0 / (completion_tokens - 1),
            0.0,
        )
    intervals_ms = [
        (later - earlier) * 1000.0
        for earlier, later in zip(chunk_times, chunk_times[1:], strict=False)
        if later >= earlier
    ]
    return {
        "prompt_id": prompt.id,
        "category": prompt.category,
        "phase": "measured",
        "status": status,
        "ok": 200 <= status < 300 and error is None,
        "error": error,
        "finish_reason": finish_reason,
        "scheduled_at_s": scheduled_at_s,
        "started_at_s": started_at_s,
        "queue_delay_ms": max((started_at_s - scheduled_at_s) * 1000.0, 0.0),
        "e2e_latency_ms": e2e_s * 1000.0,
        "ttft_ms": ttft_ms,
        "client_tpot_ms": tpot_ms,
        "stream_step_interval_ms": intervals_ms,
        "input_tokens": prompt_tokens,
        "max_output_tokens": prompt.max_output_tokens,
        "output_tokens": completion_tokens,
        "output_identity": serving.output_identity([], text_parts),
        "output_chunks": len(chunk_times),
        "events": event_count,
        "route_decisions": {},
        "metadata": prompt.metadata,
    }


def run_openai_request(
    *,
    target: TargetSpec,
    prompt: serving.PromptItem,
    model_id: str,
    input_kind: str,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    scheduled_offset_s: float,
    benchmark_started: float,
    timeout: float,
    stream_func=serving.http_sse_events,
    **_: Any,
) -> dict[str, Any]:
    scheduled_at_s = scheduled_offset_s
    target_time = benchmark_started + scheduled_offset_s
    delay = target_time - time.perf_counter()
    if delay > 0:
        time.sleep(delay)
    started = time.perf_counter()
    relative_started = started - benchmark_started
    model = target.models.get(model_id)
    try:
        if input_kind not in {"auto", "text"}:
            raise RuntimeError("OpenAI flip replay supports text input only")
        if model is None:
            raise RuntimeError(f"target has no mapping for model {model_id}")
        payload = openai_payload(
            prompt,
            served_model=model.served_model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )
        url = f"{model.base_url}{OPENAI_ENDPOINT}"
        events = list(stream_func(url, payload, timeout))
        completed = time.perf_counter()
        return observe_openai_stream(
            events,
            prompt=prompt,
            scheduled_at_s=scheduled_at_s,
            started_at_s=relative_started,
            completed_at_s=completed - benchmark_started,
        )
    except Exception as error:  # noqa: BLE001 - preserve failure in the artifact.
        completed = time.perf_counter()
        return {
            "prompt_id": prompt.id,
            "category": prompt.category,
            "phase": "measured",
            "status": None,
            "ok": False,
            "error": str(error),
            "scheduled_at_s": scheduled_at_s,
            "started_at_s": relative_started,
            "queue_delay_ms": max((relative_started - scheduled_at_s) * 1000.0, 0.0),
            "e2e_latency_ms": (completed - started) * 1000.0,
            "ttft_ms": None,
            "client_tpot_ms": None,
            "stream_step_interval_ms": [],
            "input_tokens": prompt.input_tokens_count,
            "max_output_tokens": prompt.max_output_tokens,
            "output_tokens": None,
            "output_identity": None,
            "output_chunks": 0,
            "events": 0,
            "route_decisions": {},
            "metadata": prompt.metadata,
        }


def run_managed_control_event(
    event: multimodel.ScenarioEvent,
    *,
    supervisor: ProcessSupervisor,
    benchmark_started: float,
    **_: Any,
) -> dict[str, Any]:
    target_time = benchmark_started + event.at_s
    delay = target_time - time.perf_counter()
    if delay > 0:
        time.sleep(delay)
    started = time.perf_counter()
    if event.kind == "unload":
        ok, response = supervisor.stop(event.model_id)
    elif event.kind == "load":
        ok, response = supervisor.start(event.model_id)
    else:
        ok, response = False, {"error": f"unsupported control event {event.kind}"}
    completed = time.perf_counter()
    return {
        "event_id": event.id,
        "kind": event.kind,
        "model_id": event.model_id,
        "category": event.category,
        "scheduled_at_s": event.at_s,
        "started_at_s": started - benchmark_started,
        "latency_ms": (completed - started) * 1000.0,
        "status": 200 if ok else 500,
        "ok": ok,
        "error": None if ok else response.get("error", "process control failed"),
        "response": response,
    }


def referenced_models(events: list[multimodel.ScenarioEvent]) -> list[str]:
    return sorted({event.model_id for event in events})


def target_artifact_metadata(
    target: TargetSpec,
    *,
    scenario_model_ids: list[str],
    process_audit: list[dict[str, Any]],
) -> dict[str, Any]:
    endpoints = {
        model_id: {
            "base_url": model.base_url,
            "served_model": model.served_model,
            "memory_cap_bytes": model.memory_cap_bytes,
        }
        for model_id, model in sorted(target.models.items())
    }
    package_identities = {
        model_id: model_package_identity(model.model_path)
        for model_id, model in sorted(target.models.items())
        if model_id in scenario_model_ids
    }
    if target.managed_processes:
        process_count = len({str(item["model_id"]) for item in process_audit})
    else:
        process_count = len({target.models[model_id].base_url for model_id in scenario_model_ids})
    return {
        "name": target.name,
        "base_url": target.name,
        "models": scenario_model_ids,
        "configured_models": sorted(target.models),
        "runtime": target.runtime,
        "runtime_revision": target.runtime_revision,
        "official_source": target.official_source,
        "managed_processes": target.managed_processes,
        "managed_single_process": target.managed_single_process,
        "process_count": process_count,
        "target_config_path": str(target.path),
        "target_config_sha256": hashlib.sha256(target.path.read_bytes()).hexdigest(),
        "endpoints": endpoints,
        "model_packages": package_identities,
        "comparison_contract": target.comparison_contract,
        "memory_cap_enforcement": {
            "mode": (
                "managed_process_environment"
                if target.managed_processes or target.managed_single_process
                else "required_parent_environment"
            ),
            "environment_keys": sorted(
                {key for model in target.models.values() for key in model.env} | set(target.env)
            ),
        },
        "process_audit": process_audit,
        "host": {
            "node": platform.node(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--workers", type=multimodel.positive_int, default=16)
    parser.add_argument("--timeout", type=multimodel.optional_positive_float, default=600.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slo-ttft-ms", type=multimodel.optional_positive_float)
    parser.add_argument("--slo-tpot-ms", type=multimodel.optional_positive_float)
    parser.add_argument("--slo-e2e-ms", type=multimodel.optional_positive_float)
    parser.add_argument("--log-dir", type=Path, default=Path("benchmarks/results/logs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="write failures into the artifact without returning a failing exit code",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    target = load_target(args.target)
    events = multimodel.load_scenario(args.scenario)
    scenario_model_ids = referenced_models(events)
    missing = sorted(set(scenario_model_ids) - set(target.models))
    if missing:
        raise SystemExit(f"target is missing scenario models: {missing}")

    supervisor: ProcessSupervisor | None = None
    single_supervisor: SingleProcessSupervisor | None = None
    if target.managed_processes:
        supervisor = ProcessSupervisor(target, args.log_dir)
        for model_id in scenario_model_ids:
            ok, response = supervisor.start(model_id)
            if not ok:
                supervisor.stop_all()
                raise SystemExit(f"failed to start {model_id}: {response.get('error', response)}")
    elif target.managed_single_process:
        single_supervisor = SingleProcessSupervisor(target, args.log_dir)
        ok, response = single_supervisor.start()
        if not ok:
            raise SystemExit(
                f"failed to start managed AX server: {response.get('error', response)}"
            )
        for model_id in scenario_model_ids:
            if model_id == target.primary_model_id:
                continue
            ok, response = single_supervisor.load_model(model_id)
            if not ok:
                single_supervisor.stop()
                raise SystemExit(f"failed to preload {model_id}: {response.get('error', response)}")

    benchmark_args = argparse.Namespace(
        scenario=args.scenario,
        base_url=target.name,
        workers=args.workers,
        input_kind="text",
        timeout=args.timeout,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        slo_ttft_ms=args.slo_ttft_ms,
        slo_tpot_ms=args.slo_tpot_ms,
        slo_e2e_ms=args.slo_e2e_ms,
        require_route_counter=[],
    )

    def request_runner(**kwargs: Any) -> dict[str, Any]:
        return run_openai_request(target=target, **kwargs)

    if supervisor is not None:

        def control_runner(event: multimodel.ScenarioEvent, **kwargs: Any) -> dict[str, Any]:
            return run_managed_control_event(event, supervisor=supervisor, **kwargs)
    else:
        control_runner = multimodel.run_control_event

    try:
        artifact = multimodel.run_benchmark(
            benchmark_args,
            request_runner=request_runner,
            control_runner=control_runner,
        )
    finally:
        if supervisor is not None:
            supervisor.stop_all()
        if single_supervisor is not None:
            single_supervisor.stop()

    if supervisor is not None:
        process_audit = supervisor.audit()
    elif single_supervisor is not None:
        process_audit = single_supervisor.audit()
    else:
        process_audit = []
    artifact["methodology"]["request_endpoint"] = OPENAI_ENDPOINT
    artifact["methodology"]["protocol"] = "openai_completions_sse"
    artifact["target"] = target_artifact_metadata(
        target,
        scenario_model_ids=scenario_model_ids,
        process_audit=process_audit,
    )
    artifact["sampling"]["input_kind"] = "text"
    text = json.dumps(artifact, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")

    passed = (
        artifact["availability"]["request_error_rate"] == 0.0
        and artifact["lifecycle"]["error_events"] == 0
    )
    return 0 if passed or args.report_only else 2


if __name__ == "__main__":
    raise SystemExit(main())
