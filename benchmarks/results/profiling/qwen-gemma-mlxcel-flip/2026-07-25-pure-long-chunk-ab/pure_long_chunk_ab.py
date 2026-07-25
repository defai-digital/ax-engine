#!/usr/bin/env python3
"""Pure Gemma 13.8k A/B: long-prompt prefill chunk 512 vs 768 vs 1024."""
import json, os, time, urllib.request, subprocess, pathlib, statistics

PROMPT = "<bos>" + (
    "The audit record contains alpha beta gamma delta epsilon zeta eta theta "
    "and must be retained exactly. " * 768
)
OUT = pathlib.Path(os.path.expanduser(
    "~/code/ax-engine-mlxcel-flip-work/benchmarks/results/profiling/"
    "qwen-gemma-mlxcel-flip/2026-07-25-pure-long-chunk-ab"
))
OUT.mkdir(parents=True, exist_ok=True)
MODEL = os.environ["AX_BENCH_GEMMA_MODEL_DIR"]
BIN = os.environ.get(
    "AX_SERVER_BIN",
    os.path.expanduser("~/code/ax-engine-mlxcel-flip-work/target/release-server/ax-engine-server"),
)
MLX_LIB = "/opt/homebrew/lib/python3.14/site-packages/mlx/lib"
PORT = 31494
REPS = 3
CHUNKS = [512, 768, 1024]


def kill():
    subprocess.run(
        ["bash", "-lc", "pgrep -x ax-engine-server | xargs kill 2>/dev/null || true"],
        check=False,
    )
    time.sleep(2)


def once(port):
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


def wait(port, proc, log, timeout=300):
    for _ in range(timeout):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(pathlib.Path(log).read_text(errors="ignore")[-3000:])
            time.sleep(1)
    proc.kill()
    raise RuntimeError("unhealthy")


def run_variant(name, chunk):
    kill()
    env = os.environ.copy()
    env.update(
        {
            "AX_MLX_MEMORY_LIMIT": "103079215104",
            "AX_NO_SPEC": "1",
            "AX_MLX_DENSE_FFN_COMPILE": "0",
            "AX_MLX_DENSE_FFN_COMPILE_PREFILL": "1",
            "AX_MLX_GEGLU_MUL_METAL": "1",
            "AX_MLX_DENSE_GEGLU_PACKED_METAL": "1",
            "AX_SERVER_LONG_PREFILL_WARM": "0",
            "AX_SERVER_EXEC_ARBITER_MAX_CONCURRENT": "1",
            "AX_MLX_SKIP_DECODE_ROUTE_TELEMETRY": "1",
            "AX_MLX_PREFILL_CLEAR_CACHE_PER_CHUNK": "0",
            "AX_MLX_CACHE_ONLY_CHUNK_EVAL": "0",
            "AX_MLX_COMPILED_QGELU_PREFILL_SHAPED": "0",
            "AX_MLX_COMPILED_DUAL_GATE_UP": "0",
            "AX_MLX_GEMMA_DUAL_GATE_UP_METAL": "0",
            "AX_MLX_LONG_PROMPT_PREFILL_CHUNK": str(chunk),
            "DYLD_LIBRARY_PATH": MLX_LIB,
            "MLX_LIB_DIR": MLX_LIB,
        }
    )
    log = str(OUT / f"ax-{name}.log")
    cmd = [
        BIN, "--model-id", "gemma-4-12b-it", "--mlx",
        "--mlx-model-artifacts-dir", MODEL,
        "--port", str(PORT), "--max-concurrent-requests", "2",
        "--prefill-chunk", str(chunk),
    ]
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    wait(PORT, proc, log)
    cold, _ = once(PORT)
    warm, d2 = once(PORT)
    usage = d2.get("usage") if isinstance(d2, dict) else None
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
    return {"name": name, "chunk": chunk, "cold_ms": cold, "warm_ms": warm, "usage": usage}


def main():
    results = []
    for chunk in CHUNKS:
        for i in range(1, REPS + 1):
            r = run_variant(f"c{chunk}_r{i}", chunk)
            results.append(r)
            print(f"c{chunk}_r{i}: cold={r['cold_ms']:.1f} warm={r['warm_ms']:.1f}", flush=True)
    by_chunk = {}
    for chunk in CHUNKS:
        colds = [r["cold_ms"] for r in results if r["chunk"] == chunk]
        by_chunk[str(chunk)] = {
            "cold": colds,
            "median": statistics.median(colds),
            "mean": statistics.mean(colds),
        }
    base = by_chunk["512"]["median"]
    summary = {
        "by_chunk": by_chunk,
        "base_chunk": 512,
        "base_median": base,
        "ratios_vs_512": {
            c: by_chunk[c]["median"] / base if base else None for c in by_chunk
        },
        "best_chunk": min(by_chunk, key=lambda c: by_chunk[c]["median"]),
        "keep_if_ratio_lt": 0.925,
        "results": results,
    }
    best = summary["best_chunk"]
    best_ratio = summary["ratios_vs_512"][best]
    summary["decision"] = (
        f"keep_chunk_{best}" if best != "512" and best_ratio < 0.925 else "keep_512"
    )
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "results"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
