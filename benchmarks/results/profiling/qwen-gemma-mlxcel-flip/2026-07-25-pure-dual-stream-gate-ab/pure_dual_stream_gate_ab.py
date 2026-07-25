#!/usr/bin/env python3
"""Pure Gemma under cache_eval: A/B dual_stream vs dual_qmm vs base for gate_up."""
import json, os, time, urllib.request, subprocess, pathlib, statistics

PROMPT = "<bos>" + (
    "The audit record contains alpha beta gamma delta epsilon zeta eta theta "
    "and must be retained exactly. " * 768
)
OUT = pathlib.Path(os.path.expanduser(
    "~/code/ax-engine-mlxcel-flip-work/benchmarks/results/profiling/"
    "qwen-gemma-mlxcel-flip/2026-07-25-pure-dual-stream-gate-ab"
))
OUT.mkdir(parents=True, exist_ok=True)
MODEL = os.environ["AX_BENCH_GEMMA_MODEL_DIR"]
BIN = os.environ["AX_SERVER_BIN"]
MLX = "/opt/homebrew/lib/python3.14/site-packages/mlx/lib"
PORT = 31581
REPS = 3
KEEP_IF = 0.96

BASE = {
    "AX_MLX_CACHE_ONLY_CHUNK_EVAL": "1",
    "AX_NO_SPEC": "1",
    "AX_MLX_DENSE_FFN_COMPILE": "0",
    "AX_MLX_DENSE_FFN_COMPILE_PREFILL": "1",
    "AX_MLX_GEGLU_MUL_METAL": "1",
    "AX_MLX_DENSE_GEGLU_PACKED_METAL": "1",
    "AX_MLX_SKIP_DECODE_ROUTE_TELEMETRY": "1",
    "AX_SERVER_LONG_PREFILL_WARM": "0",
    "AX_SERVER_EXEC_ARBITER_MAX_CONCURRENT": "1",
    "AX_MLX_COMPILED_DUAL_GATE_UP": "0",
    "AX_MLX_COMPILED_QGELU_PREFILL_SHAPED": "0",
    "AX_MLX_DUAL_QMM_GEGLU": "0",
    "AX_MLX_ASYNC_DUAL_GATE_UP": "0",
    "AX_MLX_MEMORY_LIMIT": "51539607552",
    "DYLD_LIBRARY_PATH": MLX,
    "MLX_LIB_DIR": MLX,
}
VARIANTS = [
    ("base", {
        "AX_MLX_DUAL_AFFINE_QMM": "0",
        "AX_MLX_DUAL_STREAM_GATE_UP": "0",
    }),
    ("dual_qmm", {
        "AX_MLX_DUAL_AFFINE_QMM": "1",
        "AX_MLX_DUAL_STREAM_GATE_UP": "0",
    }),
    ("dual_stream", {
        "AX_MLX_DUAL_AFFINE_QMM": "0",
        "AX_MLX_DUAL_STREAM_GATE_UP": "1",
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
                raise RuntimeError(f"server exited; {log.read_text()[-2500:]}")
            time.sleep(1)
    raise TimeoutError("health")


def run(name, extra, rep):
    kill()
    log = OUT / f"ax-{name}_r{rep}.log"
    env = os.environ.copy()
    env.update(BASE)
    env.update(extra)
    cmd = [
        BIN,
        "--model-id",
        "gemma-4-12b-it",
        "--mlx",
        "--mlx-model-artifacts-dir",
        MODEL,
        "--port",
        str(PORT),
        "--max-concurrent-requests",
        "2",
        "--prefill-chunk",
        "512",
    ]
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    try:
        wait(PORT, proc, log)
        cold, _ = once(PORT)
        warm, d = once(PORT)
        text = (d.get("choices") or [{}])[0].get("text") or ""
        print(
            f"{name}_r{rep}: cold={cold:.1f} warm={warm:.1f} text={text[:20]!r}",
            flush=True,
        )
        return {
            "name": f"{name}_r{rep}",
            "cold_ms": cold,
            "warm_ms": warm,
            "text": text[:40],
            "usage": d.get("usage"),
        }
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
        for name, extra in VARIANTS:
            r = run(name, extra, rep)
            results.append(r)
            by[name].append(r["cold_ms"])
            time.sleep(12)
    med = {n: statistics.median(v) for n, v in by.items()}
    base = med["base"]
    ratios = {n: med[n] / base for n in med}
    best = min(ratios, key=ratios.get)
    decision = (
        "keep_" + best if best != "base" and ratios[best] <= KEEP_IF else "keep_base"
    )
    out = {
        "by_variant": {n: {"cold": by[n], "median": med[n]} for n in by},
        "base_median": base,
        "ratios_vs_base": ratios,
        "best": best,
        "decision": decision,
        "keep_if_ratio_lt": KEEP_IF,
        "results": results,
        "residual": "dual-stream gate/up qmm on M5 Max",
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            {k: out[k] for k in ("base_median", "ratios_vs_base", "best", "decision")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
