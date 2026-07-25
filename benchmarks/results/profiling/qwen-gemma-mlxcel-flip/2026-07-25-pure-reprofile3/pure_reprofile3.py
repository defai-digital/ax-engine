#!/usr/bin/env python3
import json, os, time, urllib.request, subprocess, pathlib, statistics

PROMPT = "<bos>" + (
    "The audit record contains alpha beta gamma delta epsilon zeta eta theta "
    "and must be retained exactly. " * 768
)
OUT = pathlib.Path(os.path.expanduser(
    "~/code/ax-engine-mlxcel-flip-work/benchmarks/results/profiling/"
    "qwen-gemma-mlxcel-flip/2026-07-25-pure-reprofile3"
))
OUT.mkdir(parents=True, exist_ok=True)
MODEL = os.environ["AX_BENCH_GEMMA_MODEL_DIR"]
BIN = os.environ.get(
    "AX_SERVER_BIN",
    os.path.expanduser("~/code/ax-engine-mlxcel-flip-work/target/release-server/ax-engine-server"),
)
MLX_LIB = "/opt/homebrew/lib/python3.14/site-packages/mlx/lib"
PORT = 31498


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


def run(name, extra, dump=None):
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
            "AX_MLX_GEMMA_DUAL_GATE_UP_METAL": "0",
            "AX_MLX_COMPILED_QK_NORM_ROPE": "0",
            "AX_MLX_GEMMA_DIRECT_CPP_QK_NORM_ROPE": "0",
            "DYLD_LIBRARY_PATH": MLX_LIB,
            "MLX_LIB_DIR": MLX_LIB,
        }
    )
    env.update(extra)
    if dump:
        env["AX_MLX_PREFILL_PROFILE"] = "1"
        env["AX_MLX_PREFILL_PROFILE_DUMP"] = dump
    log = str(OUT / f"ax-{name}.log")
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
    wait(PORT, proc, log)
    cold, _ = once(PORT)
    warm, d2 = once(PORT)
    usage = d2.get("usage") if isinstance(d2, dict) else None
    profile = {}
    if dump and pathlib.Path(dump).exists():
        try:
            profile = json.loads(pathlib.Path(dump).read_text())
        except Exception as e:
            profile = {"err": str(e)}
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
    return {"name": name, "cold_ms": cold, "warm_ms": warm, "usage": usage, "profile": profile}


def main():
    results = []
    for i in (1, 2):
        r = run(f"base_r{i}", {})
        print(
            f"base_r{i}: cold={r['cold_ms']:.1f} warm={r['warm_ms']:.1f} usage={r['usage']}",
            flush=True,
        )
        results.append(r)
    dump = str(OUT / "prefill_profile_dump.json")
    r = run("profile", {}, dump=dump)
    print(f"profile: cold={r['cold_ms']:.1f}", flush=True)
    print("PROFILE", json.dumps(r["profile"], indent=2), flush=True)
    results.append(r)
    base = [x["cold_ms"] for x in results if x["name"].startswith("base_")]
    summary = {
        "base_cold": base,
        "base_mean": statistics.mean(base),
        "profile": results[-1]["profile"],
        "results": results,
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    print("BASE mean=", summary["base_mean"], base, flush=True)


if __name__ == "__main__":
    main()
