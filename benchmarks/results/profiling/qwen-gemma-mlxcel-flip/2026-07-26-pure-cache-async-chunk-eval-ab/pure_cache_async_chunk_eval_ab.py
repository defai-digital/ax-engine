#!/usr/bin/env python3
"""Pure Gemma under cache_eval: async intermediate chunk KV submit (Path A).

Residual: multi-process keep_base uses CACHE_ONLY_CHUNK_EVAL=1 (~27 blocking
eval barriers on 13.8k / c512). Those serialise host graph build behind GPU.
AX_MLX_CACHE_ONLY_CHUNK_ASYNC_EVAL=1 submits intermediate chunks with
async_eval so host can build N+1 while GPU runs N; final chunk still blocks.

Keep bar: pure ratio ≤0.96 vs base (thr 1.15 physics under multi-process).
Host: mbp-m5 / M5 Max.
"""
from __future__ import annotations

import json
import os
import pathlib
import statistics
import subprocess
import time
import urllib.request

PROMPT = "<bos>" + (
    "The audit record contains alpha beta gamma delta epsilon zeta eta theta "
    "and must be retained exactly. " * 768
)
OUT = pathlib.Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
MODEL = os.environ["AX_BENCH_GEMMA_MODEL_DIR"]
BIN = os.environ["AX_SERVER_BIN"]
MLX = os.environ.get(
    "MLX_LIB_DIR",
    "/Users/akiralam/code/ax-engine/.venv/lib/python3.14/site-packages/mlx/lib",
)
PORT = int(os.environ.get("AX_PURE_PORT", "31593"))
REPS = 3
KEEP_IF = 0.96

BASE_ENV = {
    "AX_MLX_CACHE_ONLY_CHUNK_EVAL": "1",
    "AX_MLX_CACHE_ONLY_CHUNK_ASYNC_EVAL": "0",
    "AX_NO_SPEC": "1",
    "AX_MLX_DENSE_FFN_COMPILE": "0",
    "AX_MLX_DENSE_FFN_COMPILE_PREFILL": "1",
    "AX_MLX_GEGLU_MUL_METAL": "1",
    "AX_MLX_PACK_DENSE_FFN_GATE_UP": "1",
    "AX_MLX_SKIP_DECODE_ROUTE_TELEMETRY": "1",
    "AX_SERVER_LONG_PREFILL_WARM": "0",
    "AX_SERVER_EXEC_ARBITER_MAX_CONCURRENT": "1",
    "AX_MLX_COMPILED_DUAL_GATE_UP": "0",
    "AX_MLX_COMPILED_QGELU_PREFILL_SHAPED": "0",
    "AX_MLX_DUAL_QMM_GEGLU": "0",
    "AX_MLX_ASYNC_DUAL_GATE_UP": "0",
    "AX_MLX_DUAL_AFFINE_QMM": "0",
    "AX_MLX_DUAL_STREAM_GATE_UP": "0",
    "AX_MLX_GEMMA_DUAL_GATE_UP_METAL": "0",
    "AX_MLX_MEMORY_LIMIT": "51539607552",
    "DYLD_LIBRARY_PATH": MLX,
    "MLX_LIB_DIR": MLX,
}

VARIANTS = [
    ("base", {"AX_MLX_CACHE_ONLY_CHUNK_ASYNC_EVAL": "0"}),
    ("async_chunk", {"AX_MLX_CACHE_ONLY_CHUNK_ASYNC_EVAL": "1"}),
]


def kill() -> None:
    subprocess.run(
        ["bash", "-lc", "pgrep -x ax-engine-server | xargs kill 2>/dev/null || true"],
        check=False,
    )
    time.sleep(2)


def once(port: int) -> tuple[float, dict]:
    body = {
        "model": "gemma-4-12b-it",
        "prompt": PROMPT,
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)
    return (time.perf_counter() - t0) * 1000, data


def wait(port: int, proc: subprocess.Popen, log: pathlib.Path, timeout: int = 400) -> None:
    for _ in range(timeout):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited; {log.read_text()[-2000:]}")
            time.sleep(1)
    raise TimeoutError(f"health timeout; {log.read_text()[-2000:]}")


def main() -> None:
    results = []
    for name, overrides in VARIANTS:
        times = []
        texts = []
        for rep in range(1, REPS + 1):
            kill()
            env = os.environ.copy()
            env.update(BASE_ENV)
            env.update(overrides)
            log = OUT / f"ax-{name}_r{rep}.log"
            cmd = [
                BIN,
                "--model-id",
                "gemma-4-12b-it",
                "--mlx",
                "--mlx-model-artifacts-dir",
                MODEL,
                "--host",
                "127.0.0.1",
                "--port",
                str(PORT),
                "--max-concurrent-requests",
                "1",
                "--prefill-chunk",
                "512",
            ]
            with log.open("wb") as lf:
                proc = subprocess.Popen(
                    cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True
                )
            try:
                wait(PORT, proc, log)
                # warm
                once(PORT)
                time.sleep(2)
                # cold-ish after clear: re-request after short pause
                kill()
                time.sleep(3)
                with log.open("ab") as lf:
                    proc = subprocess.Popen(
                        cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True
                    )
                wait(PORT, proc, log)
                ms, data = once(PORT)
                text = ""
                try:
                    text = data["choices"][0].get("text") or ""
                except Exception:
                    text = str(data)[:80]
                times.append(ms)
                texts.append(text)
                print(f"{name} r{rep}: {ms:.1f} ms text={text[:40]!r}", flush=True)
            finally:
                kill()
            time.sleep(5)
        med = statistics.median(times)
        results.append(
            {
                "variant": name,
                "times_ms": times,
                "median_ms": med,
                "texts": texts,
            }
        )

    base = next(r for r in results if r["variant"] == "base")
    for r in results:
        r["ratio_vs_base"] = r["median_ms"] / base["median_ms"]
        r["keep"] = r["ratio_vs_base"] <= KEEP_IF and r["variant"] != "base"

    decision = "keep_async_chunk" if any(
        r["variant"] == "async_chunk" and r["ratio_vs_base"] <= KEEP_IF for r in results
    ) else "reject_keep_base"

    out = {
        "schema": "ax.pure_ab.v1",
        "host": "mbp-m5",
        "keep_if": KEEP_IF,
        "decision": decision,
        "results": results,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print("decision:", decision)


if __name__ == "__main__":
    main()
