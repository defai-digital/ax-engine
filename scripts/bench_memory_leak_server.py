#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


RETENTION_BOUNDARY_REQUESTS = 1024
HEALTH_TIMEOUT_S = 30.0
HTTP_TIMEOUT_S = 30.0


@dataclass
class SamplePoint:
    requests_completed: int
    rss_mb: float
    elapsed_s: float


@dataclass
class ModeResult:
    mode: str
    description: str
    selected_backend: str
    support_tier: str
    resolution_policy: str
    native_runner: str | None
    requests: int
    warmup_requests: int
    prompt_tokens: int
    max_output_tokens: int
    rss_start_mb: float
    rss_end_mb: float
    rss_peak_mb: float
    rss_delta_mb: float
    rss_at_retention_boundary_mb: float | None
    post_retention_delta_mb: float | None
    tail_slope_mb_per_100_requests: float | None
    verdict: str
    samples: list[SamplePoint]
    log_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a long-lived server RSS benchmark to check for request-lifecycle leaks."
    )
    parser.add_argument(
        "--mode",
        choices=("mlx", "native", "llama", "both"),
        default="both",
        help="Which server mode to benchmark.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=2200,
        help="Measured requests per mode after warmup.",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=25,
        help="Warmup requests per mode before RSS sampling starts.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=100,
        help="Collect one RSS sample every N measured requests.",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=512,
        help="Synthetic input token count per request.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8,
        help="max_output_tokens for each request.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/tmp/ax-engine-memory-leak-benchmark.json"),
        help="Where to write the benchmark JSON report.",
    )
    parser.add_argument(
        "--server-binary",
        type=Path,
        default=Path("target/debug/ax-engine-server"),
        help="Path to the ax-engine-server binary.",
    )
    parser.add_argument(
        "--mlx-model-artifacts-dir",
        type=Path,
        default=(
            Path(os.environ["AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR"])
            if os.environ.get("AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR")
            else None
        ),
        help="MLX model artifacts directory for --mode mlx.",
    )
    return parser.parse_args()


def allocate_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_server_binary(repo_root: Path) -> None:
    subprocess.run(
        ["cargo", "build", "-p", "ax-engine-server"],
        cwd=repo_root,
        check=True,
    )


def ps_rss_mb(pid: int) -> float:
    output = subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(pid)],
        text=True,
    ).strip()
    rss_kb = int(output or "0")
    return rss_kb / 1024.0


def request_json(base_url: str, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_health(base_url: str) -> dict[str, Any]:
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return request_json(base_url, "GET", "/health")
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"server did not become healthy in time: {last_error}")


def prompt_tokens(count: int) -> list[int]:
    # Keep token ids small and stable; the synthetic benchmark only needs payload size.
    return [(index % 255) + 1 for index in range(count)]


def make_generate_payload(
    model_id: str, prompt_token_count: int, max_output_tokens: int
) -> dict[str, Any]:
    return {
        "model": model_id,
        "input_tokens": prompt_tokens(prompt_token_count),
        "max_output_tokens": max_output_tokens,
    }


def sample_label(requests_completed: int) -> str:
    if requests_completed == 0:
        return "start"
    return str(requests_completed)


def tail_slope_mb_per_100_requests(samples: list[SamplePoint], retention_boundary: int) -> float | None:
    tail = [sample for sample in samples if sample.requests_completed >= retention_boundary]
    if len(tail) < 2:
        return None

    first = tail[0]
    last = tail[-1]
    request_span = last.requests_completed - first.requests_completed
    if request_span <= 0:
        return None
    return (last.rss_mb - first.rss_mb) / (request_span / 100.0)


def classify_verdict(post_retention_delta_mb: float | None, tail_slope: float | None) -> str:
    if post_retention_delta_mb is None or tail_slope is None:
        return "insufficient_tail_data"
    if post_retention_delta_mb <= 2.0 and tail_slope <= 0.25:
        return "no_continuing_leak_observed"
    if post_retention_delta_mb >= 6.0 and tail_slope >= 0.75:
        return "continuing_growth_suspicious"
    return "inconclusive_allocator_or_small_growth"


class FakeLlamaCppCompletionHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/completion":
            self.send_error(404)
            return

        content_length = int(self.headers.get("content-length", "0"))
        raw_body = self.rfile.read(content_length)
        payload = json.loads(raw_body.decode("utf-8"))
        prompt = payload.get("prompt", [])
        n_predict = int(payload.get("n_predict", 0))

        if isinstance(prompt, list):
            prompt_evaluated = len(prompt)
        elif isinstance(prompt, str):
            prompt_evaluated = len(prompt.split())
        else:
            prompt_evaluated = 0

        response = {
            "content": " ".join(f"tok{i}" for i in range(n_predict)),
            "tokens": [1000 + i for i in range(n_predict)],
            "stop": True,
            "stop_type": "limit",
            "tokens_cached": 0,
            "tokens_evaluated": prompt_evaluated,
        }
        body = json.dumps(response, separators=(",", ":")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


class FakeLlamaCppUpstream:
    def __init__(self) -> None:
        self.port = allocate_port()
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), FakeLlamaCppCompletionHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)


class RunningServer:
    def __init__(self, repo_root: Path, binary: Path, args: list[str], description: str) -> None:
        self.repo_root = repo_root
        self.binary = binary
        self.args = args
        self.description = description
        self.port = allocate_port()
        self.log_file = tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f"ax-engine-memory-{description.replace(' ', '-')}-",
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
            cwd=self.repo_root,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
        )
        return wait_for_health(self.base_url)

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5.0)
        self.process = None
        self.log_file.close()


def find_sample(samples: list[SamplePoint], requests_completed: int) -> SamplePoint | None:
    for sample in samples:
        if sample.requests_completed == requests_completed:
            return sample
    return None


def run_mode(
    repo_root: Path,
    binary: Path,
    mode: str,
    requests_total: int,
    warmup_requests: int,
    sample_every: int,
    prompt_token_count: int,
    max_output_tokens: int,
    mlx_model_artifacts_dir: Path | None,
) -> ModeResult:
    if mode in {"mlx", "native"}:
        if mlx_model_artifacts_dir is None:
            raise ValueError(
                "MLX mode requires --mlx-model-artifacts-dir or AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR"
            )
        upstream = None
        model_id = "qwen3_dense"
        server = RunningServer(
            repo_root,
            binary,
            [
                "--model-id",
                model_id,
                "--mlx",
                "--mlx-model-artifacts-dir",
                str(mlx_model_artifacts_dir),
            ],
            "mlx",
        )
        description = "MLX mode"
    elif mode == "llama":
        upstream = FakeLlamaCppUpstream()
        upstream.start()
        model_id = "qwen3_dense"
        server = RunningServer(
            repo_root,
            binary,
            ["--llama-server-url", upstream.base_url],
            "llama_cpp",
        )
        description = "default llama.cpp bypass"
    else:
        raise ValueError(f"unsupported mode: {mode}")

    try:
        health = server.start()
        runtime = health["runtime"]
        assert server.process is not None

        payload = make_generate_payload(model_id, prompt_token_count, max_output_tokens)
        for _ in range(warmup_requests):
            response = request_json(server.base_url, "POST", "/v1/generate", payload)
            if response.get("status") != "finished":
                raise RuntimeError(f"warmup request did not finish: {response}")

        start_rss = ps_rss_mb(server.process.pid)
        started_at = time.perf_counter()
        samples = [SamplePoint(requests_completed=0, rss_mb=start_rss, elapsed_s=0.0)]

        for request_index in range(1, requests_total + 1):
            response = request_json(server.base_url, "POST", "/v1/generate", payload)
            if response.get("status") != "finished":
                raise RuntimeError(
                    f"request {request_index} did not finish in {mode} benchmark: {response}"
                )

            should_sample = (
                request_index % sample_every == 0
                or request_index == requests_total
                or request_index == RETENTION_BOUNDARY_REQUESTS
            )
            if should_sample:
                samples.append(
                    SamplePoint(
                        requests_completed=request_index,
                        rss_mb=ps_rss_mb(server.process.pid),
                        elapsed_s=time.perf_counter() - started_at,
                    )
                )

        retention_sample = find_sample(samples, RETENTION_BOUNDARY_REQUESTS)
        end_sample = samples[-1]
        peak_rss = max(sample.rss_mb for sample in samples)
        post_retention_delta = None
        if retention_sample is not None:
            post_retention_delta = end_sample.rss_mb - retention_sample.rss_mb

        tail_slope = tail_slope_mb_per_100_requests(samples, RETENTION_BOUNDARY_REQUESTS)
        verdict = classify_verdict(post_retention_delta, tail_slope)

        return ModeResult(
            mode=mode,
            description=description,
            selected_backend=runtime["selected_backend"],
            support_tier=runtime["support_tier"],
            resolution_policy=runtime["resolution_policy"],
            mlx_runner=runtime.get("mlx_runtime", {}).get("runner"),
            requests=requests_total,
            warmup_requests=warmup_requests,
            prompt_tokens=prompt_token_count,
            max_output_tokens=max_output_tokens,
            rss_start_mb=start_rss,
            rss_end_mb=end_sample.rss_mb,
            rss_peak_mb=peak_rss,
            rss_delta_mb=end_sample.rss_mb - start_rss,
            rss_at_retention_boundary_mb=(
                None if retention_sample is None else retention_sample.rss_mb
            ),
            post_retention_delta_mb=post_retention_delta,
            tail_slope_mb_per_100_requests=tail_slope,
            verdict=verdict,
            samples=samples,
            log_file=server.log_file.name,
        )
    finally:
        server.stop()
        if mode == "llama" and upstream is not None:
            upstream.stop()


def print_result_table(results: list[ModeResult]) -> None:
    print(
        "| Mode | Backend | Start RSS MB | RSS @1024 | End RSS MB | Peak RSS MB | Total Delta MB | Post-1024 Delta MB | Tail Slope MB / 100 req | Verdict |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for result in results:
        rss_at_boundary = (
            "n/a"
            if result.rss_at_retention_boundary_mb is None
            else f"{result.rss_at_retention_boundary_mb:.2f}"
        )
        post_boundary = (
            "n/a"
            if result.post_retention_delta_mb is None
            else f"{result.post_retention_delta_mb:.2f}"
        )
        tail_slope = (
            "n/a"
            if result.tail_slope_mb_per_100_requests is None
            else f"{result.tail_slope_mb_per_100_requests:.2f}"
        )
        print(
            f"| {result.mode} | {result.selected_backend} | {result.rss_start_mb:.2f} | "
            f"{rss_at_boundary} | {result.rss_end_mb:.2f} | {result.rss_peak_mb:.2f} | "
            f"{result.rss_delta_mb:.2f} | {post_boundary} | {tail_slope} | {result.verdict} |"
        )


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)
    build_server_binary(repo_root)

    binary = args.server_binary
    if not binary.is_absolute():
        binary = repo_root / binary
    if not binary.exists():
        raise FileNotFoundError(f"server binary not found: {binary}")

    modes = [args.mode] if args.mode != "both" else ["mlx", "llama"]
    results = [
        run_mode(
            repo_root=repo_root,
            binary=binary,
            mode=mode,
            requests_total=args.requests,
            warmup_requests=args.warmup_requests,
            sample_every=args.sample_every,
            prompt_token_count=args.prompt_tokens,
            max_output_tokens=args.max_output_tokens,
            mlx_model_artifacts_dir=args.mlx_model_artifacts_dir,
        )
        for mode in modes
    ]

    payload = {
        "requests": args.requests,
        "warmup_requests": args.warmup_requests,
        "sample_every": args.sample_every,
        "prompt_tokens": args.prompt_tokens,
        "max_output_tokens": args.max_output_tokens,
        "retention_boundary_requests": RETENTION_BOUNDARY_REQUESTS,
        "results": [
            {
                **{
                    key: value
                    for key, value in asdict(result).items()
                    if key != "samples"
                },
                "samples": [asdict(sample) for sample in result.samples],
            }
            for result in results
        ],
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print_result_table(results)
    print(f"\nJSON report: {args.output_json}")
    for result in results:
        print(f"{result.mode} log: {result.log_file}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(130)
