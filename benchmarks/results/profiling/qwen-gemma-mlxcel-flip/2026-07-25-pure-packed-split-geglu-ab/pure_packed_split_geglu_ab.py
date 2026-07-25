#!/usr/bin/env python3
"""Pure Gemma under cache_eval: hybrid GEMM dual-gate residual.

Residual review (mlxcel gemma4.rs + mlx_cxx_bridge.cpp):
- Multi-token bits=8 MLP is op-at-a-time dual steel qmm (#680) — no secret
  dual-gate kernel.
- AX already packs gate/up rows into one QuantizedWeight at load when bits match
  (concat steel qmm = single X-load GEMM for dual projection).
- Long Gemma prefill defaults to *split* two qmms (`AX_MLX_GEMMA4_SPLIT_PREFILL_FFN`
  default ON) after prior packed A/Bs lost.
- Prior pure-packed used packed_geglu_metal on the packed output. Hypothesis:
  single steel packed qmm is the GEMM win, but packed_geglu_metal regresses.
  Hybrid = force packed qmm (SPLIT_PREFILL=0) + kill packed GEGLU metal so path
  slices gate/up and uses production split Metal GEGLU.

Candidate env:
  AX_MLX_GEMMA4_SPLIT_PREFILL_FFN=0
  AX_MLX_DENSE_GEGLU_PACKED_METAL=0
Base: multi-process keep_base with CACHE_ONLY_CHUNK_EVAL=1, split prefill default.

Keep bar: pure ratio ≤0.96 vs base (thr 1.15 physics under multi-process).
"""
import json, os, time, urllib.request, subprocess, pathlib, statistics

PROMPT = "<bos>" + (
    "The audit record contains alpha beta gamma delta epsilon zeta eta theta "
    "and must be retained exactly. " * 768
)
OUT = pathlib.Path(os.path.expanduser(
    "~/code/ax-engine-mlxcel-flip-work/benchmarks/results/profiling/"
    "qwen-gemma-mlxcel-flip/2026-07-25-pure-packed-split-geglu-ab"
))
OUT.mkdir(parents=True, exist_ok=True)
MODEL = os.environ["AX_BENCH_GEMMA_MODEL_DIR"]
BIN = os.environ["AX_SERVER_BIN"]
MLX = "/opt/homebrew/lib/python3.14/site-packages/mlx/lib"
PORT = 31591
REPS = 3
KEEP_IF = 0.96

BASE_ENV = {
    "AX_MLX_CACHE_ONLY_CHUNK_EVAL": "1",
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
    # production multi-process keep_base: split prefill, packed GEGLU metal on
    ("base", {
        "AX_MLX_GEMMA4_SPLIT_PREFILL_FFN": "1",
        "AX_MLX_DENSE_GEGLU_PACKED_METAL": "1",
    }),
    # hybrid GEMM: one packed steel qmm + split Metal GEGLU
    ("hybrid", {
        "AX_MLX_GEMMA4_SPLIT_PREFILL_FFN": "0",
        "AX_MLX_DENSE_GEGLU_PACKED_METAL": "0",
    }),
    # packed qmm + packed GEGLU metal (prior reject class; control)
    ("packed_metal", {
        "AX_MLX_GEMMA4_SPLIT_PREFILL_FFN": "0",
        "AX_MLX_DENSE_GEGLU_PACKED_METAL": "1",
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
    for name, extra in VARIANTS:
        kill()
        env = os.environ.copy()
        env.update(BASE_ENV)
        env.update(extra)
        log = OUT / f"ax-{name}_r{rep}.log"
        cmd = [
            BIN,
            "--model-id", "gemma-4-12b-it",
            "--mlx",
            "--mlx-model-artifacts-dir", MODEL,
            "--port", str(PORT),
            "--max-concurrent-requests", "2",
            "--prefill-chunk", "512",
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
            row = {
                "name": f"{name}_r{rep}",
                "cold_ms": cold_ms,
                "warm_ms": warm_ms,
                "usage": data.get("usage"),
                "text": text[:80],
                "env": extra,
            }
            results.append(row)
            cold[name].append(cold_ms)
            warm[name].append(warm_ms)
            print(json.dumps({k: row[k] for k in row if k != "usage"}), flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.kill()
            kill()
        time.sleep(8)

med = {k: statistics.median(v) for k, v in cold.items() if v}
base = med["base"]
ratios = {k: med[k] / base for k in med}
decision = "reject_keep_base"
if ratios.get("hybrid", 99) <= KEEP_IF:
    decision = "keep_hybrid"
out = {
    "cold": cold,
    "warm": warm,
    "medians_cold": med,
    "ratios_vs_base": ratios,
    "keep_if_ratio_lt": KEEP_IF,
    "decision": decision,
    "residual": (
        "GEMM-class dual-gate hybrid: one packed steel qmm (concat gate/up rows) "
        "+ split Metal GEGLU; mlxcel multi-token bits=8 remains dual steel qmm (#680)"
    ),
    "results": results,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: out[k] for k in out if k != "results"}, indent=2))
