#!/usr/bin/env python3
"""Pure Gemma under thr-b8-like cache_eval: pack ON vs split (PACK=0)."""
from __future__ import annotations
import json, os, pathlib, statistics, subprocess, time, urllib.request

PROMPT = "<bos>" + (
    "The audit record contains alpha beta gamma delta epsilon zeta eta theta "
    "and must be retained exactly. " * 768
)
OUT = pathlib.Path(__file__).resolve().parent
MODEL = os.environ["AX_BENCH_GEMMA_MODEL_DIR"]
BIN = os.environ["AX_SERVER_BIN"]
MLX = os.environ.get("MLX_LIB_DIR", "/Users/akiralam/code/ax-engine/.venv/lib/python3.14/site-packages/mlx/lib")
PORT = int(os.environ.get("AX_PURE_PORT", "31594"))
REPS = 3
KEEP_IF = 0.96

BASE_ENV = {
    "AX_MLX_CACHE_ONLY_CHUNK_EVAL": "1",
    "AX_MLX_CACHE_ONLY_CHUNK_ASYNC_EVAL": "1",
    "AX_NO_SPEC": "1",
    "AX_MLX_DENSE_FFN_COMPILE": "0",
    "AX_MLX_DENSE_FFN_COMPILE_PREFILL": "1",
    "AX_MLX_GEGLU_MUL_METAL": "1",
    "AX_MLX_DENSE_GEGLU_PACKED_METAL": "1",
    "AX_MLX_SKIP_DECODE_ROUTE_TELEMETRY": "1",
    "AX_SERVER_LONG_PREFILL_WARM": "0",
    "AX_SERVER_EXEC_ARBITER_MAX_CONCURRENT": "1",
    "AX_MLX_PIPELINE_GRANULARITY": "layer",
    "AX_MLX_PIPELINE_EVAL_GRANULARITY": "block:8",
    "AX_MLX_MEMORY_LIMIT": "51539607552",
    "DYLD_LIBRARY_PATH": MLX,
    "MLX_LIB_DIR": MLX,
}
VARIANTS = [
    ("pack_on", {"AX_MLX_PACK_DENSE_FFN_GATE_UP": "1"}),
    ("split", {"AX_MLX_PACK_DENSE_FFN_GATE_UP": "0"}),
]

def kill():
    subprocess.run(["bash", "-lc", "pgrep -x ax-engine-server | xargs kill 2>/dev/null || true"], check=False)
    time.sleep(2)

def once(port):
    body = {"model": "gemma-4-12b-it", "prompt": PROMPT, "max_tokens": 1, "temperature": 0, "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = json.loads(resp.read().decode())
    ms = (time.perf_counter() - t0) * 1000.0
    text = payload["choices"][0].get("text") or ""
    return ms, text

def run_variant(name, overrides):
    kill()
    env = os.environ.copy(); env.update(BASE_ENV); env.update(overrides)
    log = OUT / f"ax-{name}.log"
    cmd = [BIN, "--model-id", "gemma-4-12b-it", "--mlx", "--mlx-model-artifacts-dir", MODEL, "--host", "127.0.0.1", "--port", str(PORT), "--max-concurrent-requests", "1", "--prefill-chunk", "512"]
    with log.open("w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    try:
        for _ in range(180):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError(f"{name}: server not ready")
        try:
            once(PORT)
        except Exception:
            pass
        times, texts = [], []
        for i in range(REPS):
            ms, text = once(PORT)
            times.append(ms); texts.append(text)
            print(f"{name} r{i+1}: {ms:.1f} ms text={text!r}", flush=True)
            time.sleep(2)
        return {"variant": name, "times_ms": times, "median_ms": statistics.median(times), "texts": texts}
    finally:
        proc.terminate()
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill()
        kill()

def main():
    results = [run_variant(n, o) for n, o in VARIANTS]
    base = results[0]["median_ms"]
    for r in results:
        r["ratio_vs_pack_on"] = r["median_ms"] / base
        r["keep"] = r["variant"] != "pack_on" and r["ratio_vs_pack_on"] <= KEEP_IF
    split_ratio = results[1]["ratio_vs_pack_on"]
    if split_ratio <= KEEP_IF:
        decision = "keep_split"
    elif split_ratio < 0.995:
        decision = "prefer_split_not_bar"
    else:
        decision = "reject_keep_pack"
    out = {"schema": "ax.pure_ab.v1", "host": "mbp-m5", "keep_if": KEEP_IF, "decision": decision, "results": results}
    (OUT / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
