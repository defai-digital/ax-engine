#!/usr/bin/env python3
"""Pure Gemma under cache_eval: A/B prefill-chunk 256 vs 512 (multi-process interleave residual)."""
import json, os, time, urllib.request, subprocess, pathlib, statistics

PROMPT = "<bos>" + (
    "The audit record contains alpha beta gamma delta epsilon zeta eta theta "
    "and must be retained exactly. " * 768
)
OUT = pathlib.Path(os.path.expanduser(
    "~/code/ax-engine-mlxcel-flip-work/benchmarks/results/profiling/"
    "qwen-gemma-mlxcel-flip/2026-07-25-pure-chunk256-cache-eval-ab"
))
OUT.mkdir(parents=True, exist_ok=True)
MODEL = os.environ["AX_BENCH_GEMMA_MODEL_DIR"]
BIN = os.environ["AX_SERVER_BIN"]
MLX = "/opt/homebrew/lib/python3.14/site-packages/mlx/lib"
PORT = 31531
REPS = 3
KEEP_IF = 1.02  # allow mild pure regress if concurrent tax win possible; still measure pure tax

BASE_ENV = {
    "AX_MLX_CACHE_ONLY_CHUNK_EVAL": "1",
    "AX_MLX_PREFILL_CLEAR_CACHE_PER_CHUNK": "0",
    "AX_NO_SPEC": "1",
    "AX_MLX_DENSE_FFN_COMPILE": "0",
    "AX_MLX_DENSE_FFN_COMPILE_PREFILL": "1",
    "AX_MLX_GEGLU_MUL_METAL": "1",
    "AX_MLX_DENSE_GEGLU_PACKED_METAL": "1",
    "AX_MLX_SKIP_DECODE_ROUTE_TELEMETRY": "1",
    "AX_SERVER_LONG_PREFILL_WARM": "0",
    "AX_SERVER_EXEC_ARBITER_MAX_CONCURRENT": "1",
    "AX_MLX_MEMORY_LIMIT": "51539607552",
    "DYLD_LIBRARY_PATH": MLX,
    "MLX_LIB_DIR": MLX,
}

VARIANTS = [
    ("c512", 512),
    ("c256", 256),
    ("c384", 384),
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
                raise RuntimeError(f"server exited early; log={log.read_text()[-2000:]}")
            time.sleep(1)
    raise TimeoutError("health timeout")


def run_variant(name, chunk, rep):
    kill()
    log = OUT / f"ax-{name}_r{rep}.log"
    env = os.environ.copy()
    env.update(BASE_ENV)
    cmd = [
        BIN, "--model-id", "gemma-4-12b-it", "--mlx",
        "--mlx-model-artifacts-dir", MODEL,
        "--port", str(PORT), "--max-concurrent-requests", "2",
        "--prefill-chunk", str(chunk),
    ]
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    try:
        wait(PORT, proc, log)
        cold_ms, cold_data = once(PORT)
        warm_ms, _ = once(PORT)
        text = ""
        try:
            text = (cold_data.get("choices") or [{}])[0].get("text") or ""
        except Exception:
            pass
        print(f"{name}_r{rep}: cold={cold_ms:.1f} warm={warm_ms:.1f} text={text[:20]!r}", flush=True)
        return {"name": f"{name}_r{rep}", "cold_ms": cold_ms, "warm_ms": warm_ms, "chunk": chunk, "text": text[:80], "usage": cold_data.get("usage")}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        kill()


def main():
    results = []
    by = {n: [] for n, _ in VARIANTS}
    for rep in range(1, REPS + 1):
        for name, chunk in VARIANTS:
            row = run_variant(name, chunk, rep)
            results.append(row)
            by[name].append(row["cold_ms"])
            time.sleep(10)
    medians = {n: statistics.median(v) for n, v in by.items()}
    base = medians["c512"]
    ratios = {n: medians[n] / base for n in medians}
    out = {
        "by_variant": {n: {"cold": by[n], "median": medians[n], "mean": statistics.mean(by[n])} for n in by},
        "base_median_c512": base,
        "ratios_vs_c512": ratios,
        "results": results,
        "decision": "keep_512" if all(ratios[n] >= 0.98 for n in ("c256","c384")) or min(ratios.values()) >= 0.98 else "consider_smaller_for_mp",
        "note": "smaller chunks only useful for multi-process if pure tax << concurrent tax win",
    }
    # refine decision: pure cut good if ratio < 1.0; pure regress budget for MP is ratio < 1.05
    best = min(ratios, key=ratios.get)
    if ratios[best] <= 0.96:
        out["decision"] = f"keep_{best}_pure"
    elif ratios.get("c256", 99) <= 1.05 and ratios.get("c384", 99) <= 1.05:
        out["decision"] = "mild_regress_ok_try_mp_s1"
    else:
        out["decision"] = "keep_512"
    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("base_median_c512","ratios_vs_c512","decision")}, indent=2))


if __name__ == "__main__":
    main()
