#!/usr/bin/env python3
"""Fresh pure Gemma wall + prefill profile dump on mbp-m5; optional mlxcel pure."""
import json, os, time, urllib.request, subprocess, pathlib, re

PROMPT = "<bos>" + (
    "The audit record contains alpha beta gamma delta epsilon zeta eta theta "
    "and must be retained exactly. " * 768
)
OUT = pathlib.Path(os.path.expanduser(
    "~/code/ax-engine-mlxcel-flip-work/benchmarks/results/profiling/"
    "qwen-gemma-mlxcel-flip/2026-07-25-pure-reprofile"
))
OUT.mkdir(parents=True, exist_ok=True)
MODEL = os.environ["AX_BENCH_GEMMA_MODEL_DIR"]
BIN = os.environ["AX_SERVER_BIN"]
MLX_LIB = "/opt/homebrew/lib/python3.14/site-packages/mlx/lib"


def kill_ax():
    subprocess.run(["bash", "-lc", "pgrep -x ax-engine-server | xargs kill 2>/dev/null || true"], check=False)
    time.sleep(2)


def once(port, model="gemma-4-12b-it"):
    body = {"model": model, "prompt": PROMPT, "max_tokens": 1, "temperature": 0, "stream": False}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)
    return (time.perf_counter() - t0) * 1000, data


def wait_health(port, proc, log, timeout=300):
    for _ in range(timeout):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(f"died {log}")
            time.sleep(1)
    proc.kill()
    raise RuntimeError(f"unhealthy {log}")


def run_ax(name, env_extra, port=31499):
    kill_ax()
    env = os.environ.copy()
    env.update({
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
        "DYLD_LIBRARY_PATH": MLX_LIB,
        "MLX_LIB_DIR": MLX_LIB,
    })
    env.update(env_extra)
    log = str(OUT / f"ax-{name}.log")
    cmd = [BIN, "--model-id", "gemma-4-12b-it", "--mlx",
           "--mlx-model-artifacts-dir", MODEL, "--port", str(port),
           "--max-concurrent-requests", "2", "--prefill-chunk", "512"]
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    wait_health(port, proc, log)
    cold, d1 = once(port)
    warm, d2 = once(port)
    usage = d2.get("usage") if isinstance(d2, dict) else None
    # scrape profile keys from response route metadata if present
    profile = {}
    try:
        # some paths embed profile in choices meta; also check server log
        pass
    except Exception:
        pass
    # parse log for prefill_profile lines
    text = pathlib.Path(log).read_text(errors="ignore")
    for m in re.finditer(r"ax_mlx_prefill_profile_(\w+)=([0-9.]+)", text):
        profile[m.group(1)] = float(m.group(2))
    # also try PREFILL_PROFILE_DUMP json path if set
    dump = env.get("AX_MLX_PREFILL_PROFILE_DUMP")
    if dump and pathlib.Path(dump).exists():
        try:
            profile["dump"] = json.loads(pathlib.Path(dump).read_text())
        except Exception as e:
            profile["dump_err"] = str(e)
    proc.terminate()
    try:
        proc.wait(timeout=45)
    except Exception:
        proc.kill()
    row = {"name": name, "cold_ms": cold, "warm_ms": warm, "usage": usage, "profile": profile}
    print(json.dumps({k: row[k] for k in ("name", "cold_ms", "warm_ms", "usage")}), flush=True)
    return row


results = []
# baseline pure wall (2 reps)
results.append(run_ax("base_r1", {}))
results.append(run_ax("base_r2", {}))
# profiled pure (force stage evals — for residual allocation, not fair wall)
dump_path = str(OUT / "prefill_profile_dump.json")
results.append(run_ax("profile_r1", {
    "AX_MLX_PREFILL_PROFILE": "1",
    "AX_MLX_PREFILL_PROFILE_DUMP": dump_path,
}))

outp = OUT / "results.json"
json.dump(results, open(outp, "w"), indent=2)
bases = [r["cold_ms"] for r in results if r["name"].startswith("base")]
print(f"BASE cold mean={sum(bases)/len(bases):.1f} {bases}", flush=True)
print("wrote", outp, flush=True)
if pathlib.Path(dump_path).exists():
    print("profile dump:", pathlib.Path(dump_path).read_text()[:2000], flush=True)
elif results[-1].get("profile"):
    print("profile from log:", json.dumps(results[-1]["profile"], indent=2)[:2000], flush=True)
else:
    print("no profile keys found; check log", flush=True)
    print((OUT / "ax-profile_r1.log").read_text(errors="ignore")[-2000:], flush=True)
