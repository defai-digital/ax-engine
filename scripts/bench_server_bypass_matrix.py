#!/usr/bin/env python3
"""Run a managed Qwen/Gemma server-bypass benchmark matrix.

This script owns backend server lifecycle, then delegates each measured
comparison to `scripts/bench_server_bypass.py`.
"""

from __future__ import annotations

import argparse
import http.client
import json
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = (
    "Write a concise Python function that checks whether a number is prime, "
    "explain the time complexity, and include one short example."
)


@dataclass(frozen=True)
class ModelPair:
    key: str
    label: str
    model_id: str
    gguf_path: Path
    mlx_model_id: str


MODEL_PAIRS = {
    "qwen3_5_9b": ModelPair(
        key="qwen3_5_9b",
        label="Qwen3.5-9B 4-bit",
        model_id="qwen3_5_9b_q4",
        gguf_path=REPO_ROOT / ".internal/models/Qwen3.5-9B-Q4_K_M.gguf",
        mlx_model_id="mlx-community/Qwen3.5-9B-MLX-4bit",
    ),
    "gemma4_26b_a4b": ModelPair(
        key="gemma4_26b_a4b",
        label="Gemma 4 26B A4B 4-bit",
        model_id="gemma4_26b_a4b",
        gguf_path=REPO_ROOT / ".internal/models/google_gemma-4-26B-A4B-it-Q4_K_M.gguf",
        mlx_model_id="mlx-community/gemma-4-26b-a4b-it-4bit",
    ),
}


class JsonHttpClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        parsed = urlparse(base_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"unsupported URL scheme: {base_url!r}")
        if not parsed.hostname:
            raise ValueError(f"URL must include a host: {base_url!r}")
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port
        self._base_path = parsed.path.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        connection_cls = (
            http.client.HTTPSConnection if self._scheme == "https" else http.client.HTTPConnection
        )
        connection = connection_cls(self._host, self._port, timeout=self._timeout_seconds)
        try:
            connection.request(
                "POST",
                f"{self._base_path}{path}",
                body=body,
                headers={"content-type": "application/json", "accept": "application/json"},
            )
            response = connection.getresponse()
            response_body = response.read()
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(
                    f"HTTP {response.status}: {response_body.decode(errors='replace')}"
                )
            return json.loads(response_body.decode("utf-8"))
        finally:
            connection.close()


class ManagedProcess:
    def __init__(self, name: str, command: list[str], log_dir: Path) -> None:
        self.name = name
        self.command = command
        self.log_path = log_dir / f"{safe_component(name)}.log"
        self._log_file: Any | None = None
        self.process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        self._log_file = self.log_path.open("w+b")
        self.process = subprocess.Popen(
            self.command,
            cwd=REPO_ROOT,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )

    def stop(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
            self.process = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


def safe_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in value)


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


def llama_probe_payload(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    return (
        "/completion",
        {
            "prompt": "health probe",
            "n_predict": 1,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repeat_penalty": args.repetition_penalty,
            "seed": args.seed,
            "stream": False,
            "return_tokens": True,
            "return_progress": False,
        },
    )


def mlx_probe_payload(pair: ModelPair, args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    return (
        "/v1/completions",
        {
            "model": pair.model_id,
            "prompt": "health probe",
            "max_tokens": 1,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "stream": False,
        },
    )


def wait_for_backend_ready(
    name: str,
    base_url: str,
    probe_path: str,
    probe_payload: dict[str, Any],
    timeout_seconds: float,
) -> None:
    client = JsonHttpClient(base_url, timeout_seconds=30.0)
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            client.post_json(probe_path, probe_payload)
            return
        except (OSError, RuntimeError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(2.0)
    raise RuntimeError(f"{name} did not become ready in time: {last_error}")


def llama_server_command(pair: ModelPair, port: int, args: argparse.Namespace) -> list[str]:
    return [
        args.llama_server_bin,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        str(pair.gguf_path),
        "--ctx-size",
        str(args.ctx_size),
        *args.llama_server_extra_arg,
    ]


def mlx_server_command(pair: ModelPair, port: int, args: argparse.Namespace) -> list[str]:
    return [
        *shlex.split(args.mlx_server_command),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        pair.mlx_model_id,
        "--max-tokens",
        str(max(args.max_tokens, 1)),
        "--log-level",
        args.mlx_log_level,
        *args.mlx_server_extra_arg,
    ]


def bench_command(
    backend: str,
    pair: ModelPair,
    upstream_url: str,
    result_json: Path,
    args: argparse.Namespace,
    llama_fallback_url: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(args.bench_script),
        "--backend",
        backend,
        "--upstream-url",
        upstream_url,
        "--model-id",
        pair.model_id,
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--repetition-penalty",
        str(args.repetition_penalty),
        "--seed",
        str(args.seed),
        "--repetitions",
        str(args.repetitions),
        "--cooldown-seconds",
        str(args.cooldown_seconds),
        "--http-timeout-seconds",
        str(args.http_timeout_seconds),
        "--server-binary",
        str(args.server_binary),
        "--no-build",
        "--output-json",
        str(result_json),
        "--json-only",
    ]
    if llama_fallback_url is not None:
        command.extend(["--llama-fallback-url", llama_fallback_url])
    return command


def run_bench_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "benchmark command failed\n"
            f"command: {shlex.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def selected_backends(value: str) -> list[str]:
    if value == "both":
        return ["llama_cpp", "mlx"]
    return [value]


def print_plan(args: argparse.Namespace) -> None:
    for key in args.pairs:
        pair = MODEL_PAIRS[key]
        print(f"{pair.label}")
        if "llama_cpp" in selected_backends(args.backends):
            print(f"  llama.cpp model: {pair.gguf_path}")
        if "mlx" in selected_backends(args.backends):
            print(f"  mlx model: {pair.mlx_model_id}")
            print(f"  mlx shipping fallback: {args.mlx_with_llama_fallback}")


def run_llama_case(pair: ModelPair, result_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not pair.gguf_path.is_file():
        raise FileNotFoundError(f"GGUF model not found: {pair.gguf_path}")

    port = allocate_port()
    base_url = f"http://127.0.0.1:{port}"
    backend = ManagedProcess(
        f"{pair.key}-llama-server",
        llama_server_command(pair, port, args),
        result_dir,
    )
    result_path = result_dir / f"{pair.key}-llama_cpp.json"
    probe_path, probe_payload = llama_probe_payload(args)

    try:
        backend.start()
        wait_for_backend_ready(
            backend.name,
            base_url,
            probe_path,
            probe_payload,
            args.backend_startup_timeout_seconds,
        )
        result = run_bench_command(
            bench_command("llama_cpp", pair, base_url, result_path, args)
        )
        result["pair_key"] = pair.key
        result["pair_label"] = pair.label
        result["backend_server_log"] = str(backend.log_path)
        return result
    finally:
        backend.stop()


def run_mlx_case(pair: ModelPair, result_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    mlx_port = allocate_port()
    mlx_url = f"http://127.0.0.1:{mlx_port}"
    mlx_backend = ManagedProcess(
        f"{pair.key}-mlx-server",
        mlx_server_command(pair, mlx_port, args),
        result_dir,
    )
    fallback_backend: ManagedProcess | None = None
    fallback_url: str | None = None
    result_path = result_dir / f"{pair.key}-mlx.json"
    probe_path, probe_payload = mlx_probe_payload(pair, args)

    try:
        mlx_backend.start()
        wait_for_backend_ready(
            mlx_backend.name,
            mlx_url,
            probe_path,
            probe_payload,
            args.backend_startup_timeout_seconds,
        )

        if args.mlx_with_llama_fallback:
            if not pair.gguf_path.is_file():
                raise FileNotFoundError(f"GGUF fallback model not found: {pair.gguf_path}")
            fallback_port = allocate_port()
            fallback_url = f"http://127.0.0.1:{fallback_port}"
            fallback_backend = ManagedProcess(
                f"{pair.key}-mlx-llama-fallback",
                llama_server_command(pair, fallback_port, args),
                result_dir,
            )
            fallback_backend.start()
            fallback_probe_path, fallback_probe_payload = llama_probe_payload(args)
            wait_for_backend_ready(
                fallback_backend.name,
                fallback_url,
                fallback_probe_path,
                fallback_probe_payload,
                args.backend_startup_timeout_seconds,
            )

        result = run_bench_command(
            bench_command("mlx", pair, mlx_url, result_path, args, fallback_url)
        )
        result["pair_key"] = pair.key
        result["pair_label"] = pair.label
        result["backend_server_log"] = str(mlx_backend.log_path)
        if fallback_backend is not None:
            result["fallback_server_log"] = str(fallback_backend.log_path)
        return result
    finally:
        if fallback_backend is not None:
            fallback_backend.stop()
        mlx_backend.stop()


def result_row(result: dict[str, Any]) -> dict[str, Any]:
    rss = result["ax_server"]["rss_mb"]
    return {
        "pair": result.get("pair_label", result["model_id"]),
        "pair_key": result.get("pair_key"),
        "backend": result["backend"],
        "ax_mode": result["ax_mode"],
        "direct_mean_sec": result["direct"]["elapsed_sec"]["mean"],
        "ax_mean_sec": result["ax"]["elapsed_sec"]["mean"],
        "delta_sec": result["delta"]["ax_vs_direct_sec"],
        "delta_percent": result["delta"]["ax_vs_direct_percent"],
        "ax_rss_delta_mb": None if rss is None else rss["delta"],
    }


def print_summary_table(rows: list[dict[str, Any]]) -> None:
    print(
        "| Pair | Backend | AX Mode | Direct Mean s | AX Mean s | Delta s | Delta % | AX RSS Delta MB |"
    )
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        rss_delta = (
            "n/a" if row["ax_rss_delta_mb"] is None else f"{row['ax_rss_delta_mb']:.2f}"
        )
        print(
            f"| {row['pair']} | {row['backend']} | {row['ax_mode']} | "
            f"{row['direct_mean_sec']:.4f} | {row['ax_mean_sec']:.4f} | "
            f"{row['delta_sec']:.4f} | {row['delta_percent']:.2f}% | {rss_delta} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run managed Qwen/Gemma direct-server vs AX-bypass benchmark matrix."
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=list(MODEL_PAIRS),
        choices=sorted(MODEL_PAIRS),
    )
    parser.add_argument(
        "--backends",
        choices=("llama_cpp", "mlx", "both"),
        default="both",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--cooldown-seconds", type=float, default=20.0)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--http-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--backend-startup-timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("/tmp/ax-server-bypass-matrix"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional aggregate JSON output path. Defaults to result-dir/matrix.json.",
    )
    parser.add_argument(
        "--bench-script",
        type=Path,
        default=REPO_ROOT / "scripts/bench_server_bypass.py",
    )
    parser.add_argument(
        "--server-binary",
        type=Path,
        default=REPO_ROOT / "target/release/ax-engine-server",
    )
    parser.add_argument("--llama-server-bin", default="llama-server")
    parser.add_argument(
        "--llama-server-extra-arg",
        action="append",
        default=[],
        help="Extra argument passed to llama-server. Repeat for multiple args.",
    )
    parser.add_argument(
        "--mlx-server-command",
        default="python3 -m mlx_lm server",
        help="Command prefix used to start the MLX server.",
    )
    parser.add_argument(
        "--mlx-server-extra-arg",
        action="append",
        default=[],
        help="Extra argument passed to the MLX server. Repeat for multiple args.",
    )
    parser.add_argument("--mlx-log-level", default="WARNING")
    parser.add_argument(
        "--mlx-with-llama-fallback",
        action="store_true",
        help="Configure AX --mlx mode with a live llama.cpp fallback server.",
    )
    parser.add_argument(
        "--no-build",
        dest="build",
        action="store_false",
        help="Skip cargo build -p ax-engine-server --release before the matrix.",
    )
    parser.set_defaults(build=True)
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the planned matrix and exit without launching servers.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.result_dir.is_absolute():
        args.result_dir = REPO_ROOT / args.result_dir
    if args.output_json is not None and not args.output_json.is_absolute():
        args.output_json = REPO_ROOT / args.output_json
    if not args.bench_script.is_absolute():
        args.bench_script = REPO_ROOT / args.bench_script
    if not args.server_binary.is_absolute():
        args.server_binary = REPO_ROOT / args.server_binary

    if args.print_plan:
        print_plan(args)
        return 0

    args.result_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or args.result_dir / "matrix.json"

    if args.build:
        build_release_server()
    if not args.server_binary.exists():
        raise FileNotFoundError(f"AX server binary not found: {args.server_binary}")
    if not args.bench_script.exists():
        raise FileNotFoundError(f"benchmark script not found: {args.bench_script}")

    results: list[dict[str, Any]] = []
    for key in args.pairs:
        pair = MODEL_PAIRS[key]
        for backend in selected_backends(args.backends):
            print(f"[matrix] {pair.label} / {backend}", file=sys.stderr, flush=True)
            if backend == "llama_cpp":
                results.append(run_llama_case(pair, args.result_dir, args))
            else:
                results.append(run_mlx_case(pair, args.result_dir, args))

    rows = [result_row(result) for result in results]
    aggregate = {
        "schema_version": "ax.server-bypass-matrix.v1",
        "pairs": args.pairs,
        "backends": selected_backends(args.backends),
        "repetitions": args.repetitions,
        "cooldown_seconds": args.cooldown_seconds,
        "result_dir": str(args.result_dir),
        "rows": rows,
        "results": results,
    }
    output_json.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print_summary_table(rows)
    print(f"\nJSON report: {output_json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(130)
