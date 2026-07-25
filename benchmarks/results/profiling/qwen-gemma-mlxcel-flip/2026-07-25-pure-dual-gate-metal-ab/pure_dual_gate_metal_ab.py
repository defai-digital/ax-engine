#!/usr/bin/env python3
"""Pure-wall A/B: AX_MLX_GEMMA_DUAL_GATE_UP_METAL ON vs OFF on mbp-m5."""
import json, time, urllib.request, os, subprocess, pathlib

prompt = "<bos>" + (
    "The audit record contains alpha beta gamma delta epsilon zeta eta theta "
    "and must be retained exactly. " * 768
)

OUT_DIR = pathlib.Path(
    os.path.expanduser(
        "~/code/ax-engine-mlxcel-flip-work/benchmarks/results/profiling/"
        "qwen-gemma-mlxcel-flip/2026-07-25-pure-dual-gate-metal-ab"
    )
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def once(port=31499):
    body = {
        "model": "gemma-4-12b-it",
        "prompt": prompt,
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
    return (time.perf_counter() - t0) * 1000, data.get("usage")


def kill():
    subprocess.run(
        ["bash", "-lc", "pgrep -x ax-engine-server | xargs kill 2>/dev/null || true"],
        check=False,
    )
    time.sleep(2)


def run_variant(name, env_extra, port=31499):
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
            "AX_MLX_COMPILED_QGELU_PREFILL_SHAPED": "0",
            "AX_MLX_COMPILED_DUAL_GATE_UP": "0",
            "DYLD_LIBRARY_PATH": "/opt/homebrew/lib/python3.14/site-packages/mlx/lib",
            "MLX_LIB_DIR": "/opt/homebrew/lib/python3.14/site-packages/mlx/lib",
        }
    )
    env.update(env_extra)
    model = os.environ["AX_BENCH_GEMMA_MODEL_DIR"]
    binp = os.environ["AX_SERVER_BIN"]
    log = str(OUT_DIR / f"ax-pure-dualmetal-{name}.log")
    cmd = [
        binp,
        "--model-id",
        "gemma-4-12b-it",
        "--mlx",
        "--mlx-model-artifacts-dir",
        model,
        "--port",
        str(port),
        "--max-concurrent-requests",
        "2",
        "--prefill-chunk",
        "512",
    ]
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            break
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(f"{name} died; see {log}")
            time.sleep(1)
    else:
        proc.kill()
        raise RuntimeError(f"{name} unhealthy; see {log}")
    cold, _ = once(port)
    warm, u = once(port)
    proc.terminate()
    try:
        proc.wait(timeout=45)
    except Exception:
        proc.kill()
    print(f"{name}: cold_ms={cold:.1f} warm_ms={warm:.1f} usage={u}", flush=True)
    return {"name": name, "cold_ms": cold, "warm_ms": warm, "usage": u}


results = []
# Alternating order to reduce thermal bias.
results.append(run_variant("metal_off", {"AX_MLX_GEMMA_DUAL_GATE_UP_METAL": "0"}))
results.append(run_variant("metal_on", {"AX_MLX_GEMMA_DUAL_GATE_UP_METAL": "1"}))
results.append(run_variant("metal_off_r2", {"AX_MLX_GEMMA_DUAL_GATE_UP_METAL": "0"}))
results.append(run_variant("metal_on_r2", {"AX_MLX_GEMMA_DUAL_GATE_UP_METAL": "1"}))

out = OUT_DIR / "results.json"
json.dump(results, open(out, "w"), indent=2)
off = [r["cold_ms"] for r in results if "off" in r["name"]]
on = [r["cold_ms"] for r in results if "on" in r["name"] and "off" not in r["name"]]
off_mean = sum(off) / len(off)
on_mean = sum(on) / len(on)
ratio = on_mean / off_mean
print(f"OFF cold mean={off_mean:.1f} {off}", flush=True)
print(f"ON  cold mean={on_mean:.1f} {on}", flush=True)
print(
    f"ratio_on_over_off={ratio:.4f} (want <0.925 for ~7.5% cut; target wall ≲7.94s)",
    flush=True,
)
print("wrote", out, flush=True)
summary = {
    "off_cold_ms": off,
    "on_cold_ms": on,
    "off_mean": off_mean,
    "on_mean": on_mean,
    "ratio_on_over_off": ratio,
    "keep_if_ratio_lt": 0.925,
    "decision": "keep" if ratio < 0.925 else "default_off",
}
json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2), flush=True)
