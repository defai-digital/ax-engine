#!/usr/bin/env python3
"""Benchmark direct backend HTTP inference against AX server-backed bypass.

The benchmark assumes the delegated backend server is already running:

- `llama.cpp`: compares direct `/completion` with AX `/v1/generate`
- `mlx-lm` OpenAI-compatible server: compares direct `/v1/completions` with
  AX `/v1/generate`

It keeps one persistent HTTP connection per side, alternates run order, enforces
a cooldown between every request, removes the fastest and slowest run from each
side, and reports the mean of the remaining runs.
"""

from __future__ import annotations

import argparse
import http.client
import json
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = (
    "Write a concise Python function that checks whether a number is prime, "
    "explain the time complexity, and include one short example."
)
DEFAULT_REPETITIONS = 5
DEFAULT_COOLDOWN_SECONDS = 20.0
DEFAULT_HTTP_TIMEOUT_SECONDS = 300.0
OUTLIERS_REMOVED_PER_SIDE = 1


@dataclass(frozen=True)
class RunRecord:
    run_index: int
    order_in_round: int
    elapsed_sec: float
    status_summary: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    ax_rss_mb: float | None = None


@dataclass(frozen=True)
class RemovedRun:
    run_index: int
    elapsed_sec: float
    kind: str


class HttpJsonClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        parsed = urlparse(normalize_base_url(base_url))
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"unsupported URL scheme for {base_url!r}")
        if not parsed.hostname:
            raise ValueError(f"URL must include a host: {base_url!r}")

        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port
        self._base_path = parsed.path.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._connection: http.client.HTTPConnection | http.client.HTTPSConnection | None = None

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def get_json(self, path: str) -> dict[str, Any]:
        return self._request_json("GET", path, None)

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request_json("POST", path, payload)

    def _connect(self) -> http.client.HTTPConnection | http.client.HTTPSConnection:
        if self._connection is None:
            connection_cls = (
                http.client.HTTPSConnection if self._scheme == "https" else http.client.HTTPConnection
            )
            self._connection = connection_cls(
                self._host,
                self._port,
                timeout=self._timeout_seconds,
            )
        return self._connection

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
        headers = {"accept": "application/json"}
        if body is not None:
            headers["content-type"] = "application/json"

        request_path = f"{self._base_path}{path}"
        for attempt in range(2):
            try:
                connection = self._connect()
                connection.request(method, request_path, body=body, headers=headers)
                response = connection.getresponse()
                response_body = response.read()
                if response.status < 200 or response.status >= 300:
                    raise RuntimeError(
                        f"{method} {request_path} returned HTTP {response.status}: "
                        f"{response_body.decode(errors='replace')}"
                    )
                if not response_body:
                    return {}
                return json.loads(response_body.decode())
            except (
                BrokenPipeError,
                ConnectionResetError,
                http.client.CannotSendRequest,
                http.client.RemoteDisconnected,
                http.client.ResponseNotReady,
            ):
                self.close()
                if attempt == 1:
                    raise

        raise RuntimeError(f"{method} {request_path} failed unexpectedly")


class RunningAxServer:
    def __init__(
        self,
        binary: Path,
        port: int,
        args: list[str],
        timeout_seconds: float,
    ) -> None:
        self.binary = binary
        self.port = port
        self.args = args
        self.timeout_seconds = timeout_seconds
        self.log_file = tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix="ax-engine-server-bypass-bench-",
            suffix=".log",
            delete=False,
        )
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> dict[str, Any]:
        command = [
            str(self.binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            *self.args,
        ]
        self.process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
        )
        return wait_for_health(self.base_url, self.timeout_seconds)

    def stop(self) -> None:
        if self.process is None:
            self.log_file.close()
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5.0)
        self.process = None
        self.log_file.close()


def normalize_base_url(value: str) -> str:
    return value.rstrip("/")


def allocate_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_release_server() -> None:
    subprocess.run(
        ["cargo", "build", "-p", "ax-engine-server", "--release"],
        cwd=REPO_ROOT,
        check=True,
    )


def ps_rss_mb(pid: int) -> float | None:
    try:
        output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
    except subprocess.SubprocessError:
        return None
    if not output:
        return None
    return int(output) / 1024.0


def wait_for_health(base_url: str, timeout_seconds: float) -> dict[str, Any]:
    client = HttpJsonClient(base_url, timeout_seconds)
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    try:
        while time.monotonic() < deadline:
            try:
                return client.get_json("/health")
            except (OSError, RuntimeError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(0.1)
        raise RuntimeError(f"AX server did not become healthy in time: {last_error}")
    finally:
        client.close()


def maybe_cooldown(label: str, cooldown_seconds: float, last_request_ended_at: float | None) -> None:
    if cooldown_seconds <= 0 or last_request_ended_at is None:
        return
    elapsed_since_last = time.perf_counter() - last_request_ended_at
    remaining = cooldown_seconds - elapsed_since_last
    if remaining <= 0:
        return
    print(f"[cooldown] waiting {remaining:.1f}s before {label}", file=sys.stderr, flush=True)
    time.sleep(remaining)


def sampling_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "deterministic": True,
    }


def direct_backend_payload(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.backend == "llama_cpp":
        return (
            "/completion",
            {
                "prompt": args.prompt,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repeat_penalty": args.repetition_penalty,
                "seed": args.seed,
                "n_predict": args.max_tokens,
                "stream": False,
                "return_tokens": True,
                "return_progress": False,
            },
        )

    return (
        "/v1/completions",
        {
            "model": args.model_id,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "stream": False,
        },
    )


def ax_generate_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model_id,
        "input_text": args.prompt,
        "max_output_tokens": args.max_tokens,
        "sampling": sampling_payload(args),
    }


def summarize_direct_response(backend: str, response: dict[str, Any]) -> tuple[str, int | None, int | None]:
    if backend == "llama_cpp":
        output_tokens = response.get("tokens")
        prompt_tokens = response.get("tokens_evaluated")
        return (
            "finished" if response.get("stop", True) else "running",
            prompt_tokens if isinstance(prompt_tokens, int) else None,
            len(output_tokens) if isinstance(output_tokens, list) else None,
        )

    choices = response.get("choices")
    usage = response.get("usage") or {}
    finish_reason = None
    if isinstance(choices, list) and choices:
        finish_reason = choices[0].get("finish_reason")
    return (
        str(finish_reason or "finished"),
        usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None,
        usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None,
    )


def summarize_ax_response(response: dict[str, Any]) -> tuple[str, int | None, int | None]:
    status = str(response.get("status", "unknown"))
    prompt_count = response.get("prompt_token_count")
    output_count = response.get("output_token_count")
    output_tokens = response.get("output_tokens")
    if output_count is None and isinstance(output_tokens, list):
        output_count = len(output_tokens)
    return (
        status,
        prompt_count if isinstance(prompt_count, int) else None,
        output_count if isinstance(output_count, int) else None,
    )


def timed_post(
    client: HttpJsonClient,
    path: str,
    payload: dict[str, Any],
    summarize: Any,
) -> tuple[float, str, int | None, int | None]:
    started_at = time.perf_counter()
    response = client.post_json(path, payload)
    elapsed_sec = time.perf_counter() - started_at
    status_summary, prompt_tokens, output_tokens = summarize(response)
    return elapsed_sec, status_summary, prompt_tokens, output_tokens


def summarize_numeric(values: list[float]) -> dict[str, float]:
    summary = {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }
    if len(values) > 1:
        summary["stdev"] = statistics.stdev(values)
    return summary


def trim_runs(runs: list[RunRecord]) -> tuple[list[RunRecord], list[RemovedRun]]:
    indexed = list(enumerate(runs))
    sorted_runs = sorted(indexed, key=lambda item: (item[1].elapsed_sec, item[0]))
    removed_indexes: set[int] = set()
    removed: list[RemovedRun] = []

    for index, run in sorted_runs[:OUTLIERS_REMOVED_PER_SIDE]:
        removed_indexes.add(index)
        removed.append(RemovedRun(run.run_index, run.elapsed_sec, "low"))
    for index, run in sorted_runs[-OUTLIERS_REMOVED_PER_SIDE:]:
        removed_indexes.add(index)
        removed.append(RemovedRun(run.run_index, run.elapsed_sec, "high"))

    kept = [run for index, run in enumerate(runs) if index not in removed_indexes]
    return kept, removed


def finalize_side(command_label: str, runs: list[RunRecord]) -> dict[str, Any]:
    kept, removed = trim_runs(runs)
    return {
        "label": command_label,
        "runs": [asdict(run) for run in runs],
        "filtered_runs": [asdict(run) for run in kept],
        "removed_runs": [asdict(run) for run in removed],
        "outlier_policy": {
            "kind": "trim_min_max_elapsed",
            "removed_per_side": OUTLIERS_REMOVED_PER_SIDE,
        },
        "raw_elapsed_sec": summarize_numeric([run.elapsed_sec for run in runs]),
        "elapsed_sec": summarize_numeric([run.elapsed_sec for run in kept]),
    }


def ax_server_args(args: argparse.Namespace) -> tuple[list[str], str]:
    common = ["--model-id", args.model_id]
    if args.backend == "llama_cpp":
        return [*common, "--compat-server-url", args.upstream_url], "shipping_default_llama_cpp"

    if args.llama_fallback_url:
        return (
            [
                *common,
                "--mlx",
                "--compat-server-url",
                args.upstream_url,
                "--llama-fallback-server-url",
                args.llama_fallback_url,
            ],
            "shipping_mlx_with_llama_fallback",
        )

    return (
        [
            *common,
            "--support-tier",
            "compatibility",
            "--compat-backend",
            "mlx",
            "--compat-server-url",
            args.upstream_url,
        ],
        "explicit_mlx_compatibility_no_fallback",
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.repetitions < 5:
        raise ValueError("--repetitions must be at least 5 so min/max outlier trimming is valid")

    binary = args.server_binary
    if not binary.is_absolute():
        binary = REPO_ROOT / binary

    if args.build:
        build_release_server()
    if not binary.exists():
        raise FileNotFoundError(f"AX server binary not found: {binary}")

    direct_path, direct_payload = direct_backend_payload(args)
    ax_payload = ax_generate_payload(args)
    ax_args, ax_mode = ax_server_args(args)
    ax_server = RunningAxServer(
        binary=binary,
        port=allocate_port(),
        args=ax_args,
        timeout_seconds=args.http_timeout_seconds,
    )

    direct_client = HttpJsonClient(args.upstream_url, args.http_timeout_seconds)
    ax_client: HttpJsonClient | None = None
    last_request_ended_at: float | None = None

    try:
        health = ax_server.start()
        ax_client = HttpJsonClient(ax_server.base_url, args.http_timeout_seconds)

        warmups = [
            ("direct warmup", direct_client, direct_path, direct_payload, lambda response: summarize_direct_response(args.backend, response)),
            ("ax warmup", ax_client, "/v1/generate", ax_payload, summarize_ax_response),
        ]
        for label, client, path, payload, summarize in warmups:
            maybe_cooldown(label, args.cooldown_seconds, last_request_ended_at)
            timed_post(client, path, payload, summarize)
            last_request_ended_at = time.perf_counter()

        direct_runs: list[RunRecord] = []
        ax_runs: list[RunRecord] = []

        for round_index in range(args.repetitions):
            run_index = round_index + 1
            if round_index % 2 == 0:
                schedule = [
                    ("direct", direct_client, direct_path, direct_payload, direct_runs, lambda response: summarize_direct_response(args.backend, response)),
                    ("ax", ax_client, "/v1/generate", ax_payload, ax_runs, summarize_ax_response),
                ]
            else:
                schedule = [
                    ("ax", ax_client, "/v1/generate", ax_payload, ax_runs, summarize_ax_response),
                    ("direct", direct_client, direct_path, direct_payload, direct_runs, lambda response: summarize_direct_response(args.backend, response)),
                ]

            for order_in_round, (label, client, path, payload, target, summarize) in enumerate(schedule, start=1):
                maybe_cooldown(
                    f"{label} run {run_index}",
                    args.cooldown_seconds,
                    last_request_ended_at,
                )
                elapsed_sec, status_summary, prompt_tokens, output_tokens = timed_post(
                    client,
                    path,
                    payload,
                    summarize,
                )
                last_request_ended_at = time.perf_counter()
                target.append(
                    RunRecord(
                        run_index=run_index,
                        order_in_round=order_in_round,
                        elapsed_sec=elapsed_sec,
                        status_summary=status_summary,
                        prompt_tokens=prompt_tokens,
                        output_tokens=output_tokens,
                        ax_rss_mb=(
                            None
                            if label != "ax" or ax_server.process is None
                            else ps_rss_mb(ax_server.process.pid)
                        ),
                    )
                )

        direct = finalize_side("direct_backend_http", direct_runs)
        ax = finalize_side("ax_engine_server_bypass", ax_runs)
        direct_mean = direct["elapsed_sec"]["mean"]
        ax_mean = ax["elapsed_sec"]["mean"]
        ax_rss_samples = [
            run.ax_rss_mb for run in ax_runs if run.ax_rss_mb is not None
        ]

        return {
            "schema_version": "ax.server-bypass-benchmark.v1",
            "backend": args.backend,
            "model_id": args.model_id,
            "ax_mode": ax_mode,
            "upstream_url": normalize_base_url(args.upstream_url),
            "llama_fallback_url": (
                None if args.llama_fallback_url is None else normalize_base_url(args.llama_fallback_url)
            ),
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "repetitions": args.repetitions,
            "cooldown_seconds": args.cooldown_seconds,
            "cooldown_policy": "minimum idle time between every direct or AX HTTP request",
            "http_connection_policy": "one persistent client connection per measured side",
            "ax_server": {
                "binary": str(binary),
                "base_url": ax_server.base_url,
                "health": health,
                "log_file": ax_server.log_file.name,
                "rss_mb": (
                    None
                    if not ax_rss_samples
                    else {
                        "start": ax_rss_samples[0],
                        "end": ax_rss_samples[-1],
                        "peak": max(ax_rss_samples),
                        "delta": ax_rss_samples[-1] - ax_rss_samples[0],
                    }
                ),
            },
            "direct": direct,
            "ax": ax,
            "delta": {
                "ax_vs_direct_sec": ax_mean - direct_mean,
                "ax_vs_direct_percent": ((ax_mean / direct_mean) - 1.0) * 100.0,
            },
        }
    finally:
        direct_client.close()
        if ax_client is not None:
            ax_client.close()
        ax_server.stop()


def print_table(result: dict[str, Any]) -> None:
    rss = result["ax_server"]["rss_mb"]
    rss_delta = "n/a" if rss is None else f"{rss['delta']:.2f}"
    print(
        "| Backend | AX Mode | Direct Mean s | AX Mean s | Delta s | Delta % | AX RSS Delta MB |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    print(
        f"| {result['backend']} | {result['ax_mode']} | "
        f"{result['direct']['elapsed_sec']['mean']:.4f} | "
        f"{result['ax']['elapsed_sec']['mean']:.4f} | "
        f"{result['delta']['ax_vs_direct_sec']:.4f} | "
        f"{result['delta']['ax_vs_direct_percent']:.2f}% | "
        f"{rss_delta} |"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark direct backend server inference against AX server-backed bypass."
    )
    parser.add_argument("--backend", choices=("llama_cpp", "mlx"), required=True)
    parser.add_argument(
        "--upstream-url",
        required=True,
        help="Running backend base URL, e.g. http://127.0.0.1:8081",
    )
    parser.add_argument(
        "--llama-fallback-url",
        help="Required to benchmark the shipping --mlx path with llama.cpp fallback configured.",
    )
    parser.add_argument("--model-id", default="qwen3_dense")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=DEFAULT_COOLDOWN_SECONDS,
        help="Minimum idle time between every direct or AX HTTP request.",
    )
    parser.add_argument(
        "--http-timeout-seconds",
        type=float,
        default=DEFAULT_HTTP_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--server-binary",
        type=Path,
        default=Path("target/release/ax-engine-server"),
    )
    parser.add_argument(
        "--no-build",
        dest="build",
        action="store_false",
        help="Skip cargo build -p ax-engine-server --release before benchmarking.",
    )
    parser.set_defaults(build=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only the JSON result to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_benchmark(args)

    rendered = json.dumps(result, indent=2)
    if args.output_json:
        args.output_json.write_text(rendered + "\n", encoding="utf-8")

    if args.json_only:
        print(rendered)
    else:
        print_table(result)
        if args.output_json:
            print(f"\nJSON report: {args.output_json}")
        print(f"AX server log: {result['ax_server']['log_file']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(130)
