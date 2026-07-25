#!/usr/bin/env python3
"""Pure Gemma under cache_eval keep_base: A/B AX_MLX_PIPELINE_GRANULARITY."""
import json, os, time, urllib.request, subprocess, pathlib, statistics

PROMPT = "<bos>" + (
    "The audit record contains alpha beta gamma delta epsilon zeta eta theta "
    "and must be retained exactly. " * 768
)
OUT = pathlib.Path(os.path.expanduser(
    "~/code/ax-engine-mlxcel-flip-work/benchmarks/results/profiling/"
    "qwen-gemma-mlxcel-flip/2026-07-25-pure-pipeline-granularity-ab"
))
OUT.mkdir(parents=True, exist_ok=True)
MODEL = os.environ["AX_BENCH_GEMMA_MODEL_DIR"]
BIN = os.environ["AX_SERVER_BIN"]
MLX = "/opt/homebrew/lib/python3.14/site-packages/mlx/lib"
PORT = 31511
REPS = 3
KEEP_IF = 0.96

# Shared multi-process Gemma baseline knobs (keep_base from compose A/B).
BASE_ENV = {
    "AX_MLX_CACHE_ONLY_CHUNK_EVAL": "1",
    "AX_MLX_PREFILL_CLEAR_CACHE_PER_CHUNK": "0",
    "AX_MLX_DENSE_QMATMUL_RMS_NORM": "0",
    "AX_MLX_ROTATING_SLIDING_PREFILL": "1",
    "AX_MLX_PACK_LINEAR_ATTENTION_PROJECTIONS": "0",
    "AX_MLX_SKIP_DECODE_ROUTE_TELEMETRY": "1",
    "AX_MLX_GEGLU_MUL_METAL": "1",
    "AX_MLX_DENSE_GEGLU_PACKED_METAL": "1",
    "AX_NO_SPEC": "1",
}

VARIANTS = [
    ("base", {"AX_MLX_PIPELINE_GRANULARITY": "off"}),
    ("block4", {"AX_MLX_PIPELINE_GRANULARITY": "block:4"}),
    ("block2", {"AX_MLX_PIPELINE_GRANULARITY": "block:2"}),
    ("layer", {"AX_MLX_PIPELINE_GRANULARITY": "layer"}),
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
    raise TimeoutError(f"health timeout; log={log.read_text()[-2000:]}")


def run_variant(name, extra_env, rep):
    kill()
    log = OUT / f"ax-{name}_r{rep}.log"
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update(
        {
            "AX_MLX_MEMORY_LIMIT": "51539607552",
            "AX_MLX_DENSE_FFN_COMPILE": "0",
            "AX_MLX_DENSE_FFN_COMPILE_PREFILL": "1",
            "AX_SERVER_LONG_PREFILL_WARM": "0",
            "AX_SERVER_EXEC_ARBITER_MAX_CONCURRENT": "1",
            "AX_MLX_GEMMA_DUAL_GATE_UP_METAL": "0",
            "DYLD_LIBRARY_PATH": MLX,
            "MLX_LIB_DIR": MLX,
        }
    )
    env.update(extra_env)
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
        cold_ms, cold_data = once(PORT)
        warm_ms, warm_data = once(PORT)
        usage = cold_data.get("usage") or {}
        text = ""
        try:
            text = cold_data["choices"][0].get("text") or ""
        except Exception:
            pass
        row = {
            "name": f"{name}_r{rep}",
            "cold_ms": cold_ms,
            "warm_ms": warm_ms,
            "usage": usage,
            "text": text[:80],
            "env": extra_env,
        }
        print(
            f"{name}_r{rep}: cold={cold_ms:.1f} warm={warm_ms:.1f} text={text[:20]!r}",
            flush=True,
        )
        return row
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
    # Interleave reps: for each rep, run all variants (thermal fair).
    for rep in range(1, REPS + 1):
        for name, extra in VARIANTS:
            row = run_variant(name, extra, rep)
            results.append(row)
            by[name].append(row["cold_ms"])
            time.sleep(8)

    medians = {n: statistics.median(v) for n, v in by.items()}
    base = medians["base"]
    ratios = {n: (medians[n] / base if base else None) for n in medians}
    best = min(ratios, key=ratios.get)
    decision = "keep_" + best if ratios[best] <= KEEP_IF and best != "base" else "keep_base"
    if best != "base" and ratios[best] > KEEP_IF:
        decision = "reject_keep_off"

    out = {
        "by_variant": {
            n: {
                "cold": by[n],
                "median": medians[n],
                "mean": statistics.mean(by[n]),
            }
            for n in by
        },
        "base_median": base,
        "ratios_vs_base": ratios,
        "keep_if_ratio_lt": KEEP_IF,
        "best": best,
        "results": results,
        "decision": decision,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    (OUT / "run.log").write_text(
        "\n".join(
            f"{r['name']}: cold={r['cold_ms']:.1f} warm={r['warm_ms']:.1f} text={r['text']!r}"
            for r in results
        )
        + f"\n\n{json.dumps({k: out[k] for k in ('base_median','ratios_vs_base','best','decision')}, indent=2)}\n"
    )
    print(json.dumps({k: out[k] for k in ("base_median", "ratios_vs_base", "best", "decision")}, indent=2))


if __name__ == "__main__":
    main()
