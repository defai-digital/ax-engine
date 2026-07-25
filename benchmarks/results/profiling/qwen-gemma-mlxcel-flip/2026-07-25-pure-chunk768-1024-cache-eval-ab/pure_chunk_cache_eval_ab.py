#!/usr/bin/env python3
"""Pure Gemma under cache_eval: A/B prefill-chunk 512 vs 768 vs 1024.

Residual: AX_MLX_CACHE_ONLY_CHUNK_EVAL forces KV materialize every chunk
(#672). Eval barrier count scales with ceil(tokens/chunk). Larger chunks
may cut pure wall more under cache_eval than without (prior non-cache_eval
c768/c1024 ~0.98; under cache_eval only smaller chunks were A/B'd and
regressed). Thr-physics keep bar 0.96 under multi-process keep_base.
"""
import json, os, time, urllib.request, subprocess, pathlib, statistics

PROMPT = "<bos>" + (
    "The audit record contains alpha beta gamma delta epsilon zeta eta theta "
    "and must be retained exactly. " * 768
)
OUT = pathlib.Path(os.path.expanduser(
    "~/code/ax-engine-mlxcel-flip-work/benchmarks/results/profiling/"
    "qwen-gemma-mlxcel-flip/2026-07-25-pure-chunk768-1024-cache-eval-ab"
))
OUT.mkdir(parents=True, exist_ok=True)
MODEL = os.environ["AX_BENCH_GEMMA_MODEL_DIR"]
BIN = os.environ["AX_SERVER_BIN"]
MLX = "/opt/homebrew/lib/python3.14/site-packages/mlx/lib"
PORT = 31581
REPS = 3
KEEP_IF = 0.96

BASE_ENV = {
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
    "AX_MLX_DUAL_AFFINE_QMM": "0",
    "AX_MLX_DUAL_STREAM_GATE_UP": "0",
    "AX_MLX_MEMORY_LIMIT": "51539607552",
    "DYLD_LIBRARY_PATH": MLX,
    "MLX_LIB_DIR": MLX,
}
VARIANTS = [
    ("c512", 512),
    ("c768", 768),
    ("c1024", 1024),
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


def wait(port, proc, log, timeout=400):
    for _ in range(timeout):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited; {log.read_text()[-2000:]}")
            time.sleep(1)
    raise TimeoutError(f"health timeout; {log.read_text()[-2000:]}")


results = []
cold = {name: [] for name, _ in VARIANTS}
warm = {name: [] for name, _ in VARIANTS}

for rep in range(1, REPS + 1):
    for name, chunk in VARIANTS:
        kill()
        env = os.environ.copy()
        env.update(BASE_ENV)
        log = OUT / f"ax-{name}_r{rep}.log"
        cmd = [
            BIN,
            "--model-id", "gemma-4-12b-it",
            "--mlx",
            "--mlx-model-artifacts-dir", MODEL,
            "--port", str(PORT),
            "--max-concurrent-requests", "2",
            "--prefill-chunk", str(chunk),
        ]
        with open(log, "w") as lf:
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        try:
            wait(PORT, proc, log)
            cold_ms, _ = once(PORT)
            warm_ms, data = once(PORT)
            text = ""
            try:
                text = data["choices"][0].get("text", "")
            except Exception:
                pass
            usage = data.get("usage", {})
            row = {
                "name": f"{name}_r{rep}",
                "chunk": chunk,
                "cold_ms": cold_ms,
                "warm_ms": warm_ms,
                "usage": usage,
                "text": text[:80],
            }
            results.append(row)
            cold[name].append(cold_ms)
            warm[name].append(warm_ms)
            print(json.dumps(row), flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.kill()
            kill()
        time.sleep(8)

med = {k: statistics.median(v) for k, v in cold.items() if v}
base = med["c512"]
ratios = {k: med[k] / base for k in med}
decision = "reject_keep_c512"
best = min(ratios, key=ratios.get)
if best != "c512" and ratios[best] <= KEEP_IF:
    decision = f"keep_{best}"
out = {
    "cold": cold,
    "warm": warm,
    "medians_cold": med,
    "ratios_vs_c512": ratios,
    "keep_if_ratio_lt": KEEP_IF,
    "decision": decision,
    "residual": "cache_eval barrier count scales with chunk count (#672)",
    "results": results,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: out[k] for k in out if k != "results"}, indent=2))
