#!/usr/bin/env python3
"""Pure Gemma under multi-process baseline (cache_eval ON): compose residual A/B."""
import json, os, time, urllib.request, subprocess, pathlib, statistics

PROMPT = "<bos>" + (
    "The audit record contains alpha beta gamma delta epsilon zeta eta theta "
    "and must be retained exactly. " * 768
)
OUT = pathlib.Path(os.path.expanduser(
    "~/code/ax-engine-mlxcel-flip-work/benchmarks/results/profiling/"
    "qwen-gemma-mlxcel-flip/2026-07-25-pure-compose-cache-eval-ab"
))
OUT.mkdir(parents=True, exist_ok=True)
MODEL = os.environ["AX_BENCH_GEMMA_MODEL_DIR"]
BIN = os.environ["AX_SERVER_BIN"]
MLX = "/opt/homebrew/lib/python3.14/site-packages/mlx/lib"
PORT = 31501
REPS = 3

VARIANTS = [
    ("base", {
        "AX_MLX_CACHE_ONLY_CHUNK_EVAL": "1",
        "AX_MLX_PREFILL_CLEAR_CACHE_PER_CHUNK": "0",
        "AX_MLX_DENSE_QMATMUL_RMS_NORM": "0",
        "AX_MLX_ROTATING_SLIDING_PREFILL": "1",
    }),
    ("norot", {
        "AX_MLX_CACHE_ONLY_CHUNK_EVAL": "1",
        "AX_MLX_PREFILL_CLEAR_CACHE_PER_CHUNK": "0",
        "AX_MLX_DENSE_QMATMUL_RMS_NORM": "0",
        "AX_MLX_ROTATING_SLIDING_PREFILL": "0",
    }),
    ("qmmrms", {
        "AX_MLX_CACHE_ONLY_CHUNK_EVAL": "1",
        "AX_MLX_PREFILL_CLEAR_CACHE_PER_CHUNK": "0",
        "AX_MLX_DENSE_QMATMUL_RMS_NORM": "1",
        "AX_MLX_ROTATING_SLIDING_PREFILL": "1",
    }),
    ("both", {
        "AX_MLX_CACHE_ONLY_CHUNK_EVAL": "1",
        "AX_MLX_PREFILL_CLEAR_CACHE_PER_CHUNK": "0",
        "AX_MLX_DENSE_QMATMUL_RMS_NORM": "1",
        "AX_MLX_ROTATING_SLIDING_PREFILL": "0",
    }),
]


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


def run_variant(name, extra):
    kill()
    env = os.environ.copy()
    env.update(
        {
            "AX_MLX_MEMORY_LIMIT": "51539607552",
            "AX_NO_SPEC": "1",
            "AX_MLX_DENSE_FFN_COMPILE": "0",
            "AX_MLX_DENSE_FFN_COMPILE_PREFILL": "1",
            "AX_MLX_GEGLU_MUL_METAL": "1",
            "AX_MLX_DENSE_GEGLU_PACKED_METAL": "1",
            "AX_SERVER_LONG_PREFILL_WARM": "0",
            "AX_SERVER_EXEC_ARBITER_MAX_CONCURRENT": "1",
            "AX_MLX_SKIP_DECODE_ROUTE_TELEMETRY": "1",
            "AX_MLX_GEMMA_DUAL_GATE_UP_METAL": "0",
            "DYLD_LIBRARY_PATH": MLX,
            "MLX_LIB_DIR": MLX,
        }
    )
    env.update(extra)
    log = str(OUT / f"ax-{name}.log")
    cmd = [
        BIN, "--model-id", "gemma-4-12b-it", "--mlx",
        "--mlx-model-artifacts-dir", MODEL,
        "--port", str(PORT), "--max-concurrent-requests", "2",
        "--prefill-chunk", "512",
    ]
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    wait(PORT, proc, log)
    cold, _ = once(PORT)
    warm, d2 = once(PORT)
    text = ""
    try:
        ch = d2.get("choices") or []
        if ch:
            text = (ch[0].get("text") or "")[:40]
    except Exception:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
    return {"name": name, "cold_ms": cold, "warm_ms": warm, "usage": d2.get("usage"), "text": text}


def main():
    results = []
    # Interleave variants for thermal fairness
    for i in range(1, REPS + 1):
        for vname, env in VARIANTS:
            r = run_variant(f"{vname}_r{i}", env)
            results.append(r)
            print(
                f"{r['name']}: cold={r['cold_ms']:.1f} warm={r['warm_ms']:.1f} text={r['text']!r}",
                flush=True,
            )
        time.sleep(8)
    by = {}
    for vname, _ in VARIANTS:
        colds = [r["cold_ms"] for r in results if r["name"].startswith(vname + "_")]
        by[vname] = {
            "cold": colds,
            "median": statistics.median(colds),
            "mean": statistics.mean(colds),
        }
    base = by["base"]["median"]
    summary = {
        "by_variant": by,
        "base_median": base,
        "ratios_vs_base": {k: by[k]["median"] / base if base else None for k in by},
        "keep_if_ratio_lt": 0.96,
        "best": min(by, key=lambda k: by[k]["median"]),
        "results": results,
    }
    best = summary["best"]
    summary["decision"] = (
        f"keep_{best}"
        if best != "base" and summary["ratios_vs_base"][best] < 0.96
        else "keep_base"
    )
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "results"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
