#!/usr/bin/env python3
"""Probe: two single-model AX servers (mlxcel topology) for S1 thr/gap."""
import json, os, time, urllib.request, subprocess, pathlib, threading, statistics

OUT = pathlib.Path(os.path.expanduser(
    "~/code/ax-engine-mlxcel-flip-work/benchmarks/results/profiling/"
    "qwen-gemma-mlxcel-flip/2026-07-25-ax-multiprocess-s1-probe"
))
OUT.mkdir(parents=True, exist_ok=True)
BIN = os.environ["AX_SERVER_BIN"]
GEMMA = os.environ["AX_BENCH_GEMMA_MODEL_DIR"]
QWEN = os.environ["AX_BENCH_QWEN_MODEL_DIR"]
MLX = "/opt/homebrew/lib/python3.14/site-packages/mlx/lib"
REPS = 3

# S1-like prompts
QWEN_PROMPT = "Write a detailed technical analysis of concurrent inference scheduling on Apple Silicon GPUs. " * 2
GEMMA_PROMPT = "<bos>" + (
    "The audit record contains alpha beta gamma delta epsilon zeta eta theta "
    "and must be retained exactly. " * 768
)


def kill_all():
    subprocess.run(["bash", "-lc", "pgrep -x ax-engine-server | xargs kill 2>/dev/null || true"], check=False)
    time.sleep(2)


def base_env():
    e = os.environ.copy()
    e.update({
        "AX_MLX_MEMORY_LIMIT": "51539607552",  # 48GB per process like mlxcel
        "AX_NO_SPEC": "1",
        "AX_MLX_DENSE_FFN_COMPILE": "0",
        "AX_MLX_DENSE_FFN_COMPILE_PREFILL": "1",
        "AX_MLX_GEGLU_MUL_METAL": "1",
        "AX_MLX_DENSE_GEGLU_PACKED_METAL": "1",
        "AX_MLX_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL": "1",
        "AX_MLX_BATCHED_DECODE": "1",
        "AX_MLX_SKIP_DECODE_ROUTE_TELEMETRY": "1",
        "DYLD_LIBRARY_PATH": MLX,
        "MLX_LIB_DIR": MLX,
    })
    return e


def start_server(model_id, model_dir, port, log_path):
    env = base_env()
    cmd = [
        BIN, "--model-id", model_id, "--mlx",
        "--mlx-model-artifacts-dir", model_dir,
        "--port", str(port),
        "--max-concurrent-requests", "4",
        "--prefill-chunk", "512",
    ]
    lf = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return proc, lf
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(pathlib.Path(log_path).read_text(errors="ignore")[-3000:])
            time.sleep(1)
    proc.kill()
    raise RuntimeError(f"unhealthy {model_id}")


def stream_completion(port, model, prompt, max_tokens):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    gaps = []
    last_tok_t = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            ch = (obj.get("choices") or [{}])[0]
            text = ch.get("text") or ""
            if text:
                now = time.perf_counter()
                if ttft is None:
                    ttft = (now - t0) * 1000
                if last_tok_t is not None:
                    gaps.append((now - last_tok_t) * 1000)
                last_tok_t = now
                tokens += 1
    wall = (time.perf_counter() - t0) * 1000
    return {
        "tokens": tokens,
        "wall_ms": wall,
        "ttft_ms": ttft,
        "gap_p95_ms": sorted(gaps)[int(0.95 * (len(gaps) - 1))] if gaps else None,
        "tok_s": tokens / (wall / 1000) if wall > 0 else 0,
    }


def run_rep(rep):
    kill_all()
    q_log = OUT / f"qwen_r{rep}.log"
    g_log = OUT / f"gemma_r{rep}.log"
    q_proc, q_lf = start_server("qwen3.5-9b", QWEN, 31811, q_log)
    g_proc, g_lf = start_server("gemma-4-12b-it", GEMMA, 31812, g_log)
    results = {}
    errors = []

    def run_qwen():
        try:
            results["qwen"] = stream_completion(31811, "qwen3.5-9b", QWEN_PROMPT, 192)
        except Exception as e:
            errors.append(f"qwen:{e}")

    def run_gemma():
        try:
            results["gemma"] = stream_completion(31812, "gemma-4-12b-it", GEMMA_PROMPT, 1)
        except Exception as e:
            errors.append(f"gemma:{e}")

    t0 = time.perf_counter()
    tq = threading.Thread(target=run_qwen)
    tg = threading.Thread(target=run_gemma)
    # start qwen slightly first like S1 interactive
    tq.start()
    time.sleep(0.05)
    tg.start()
    tq.join()
    tg.join()
    scenario_wall = (time.perf_counter() - t0) * 1000
    total_tok = sum(r.get("tokens", 0) for r in results.values())
    thr = total_tok / (scenario_wall / 1000) if scenario_wall > 0 else 0
    for p, lf in ((q_proc, q_lf), (g_proc, g_lf)):
        p.terminate()
        try:
            p.wait(timeout=30)
        except Exception:
            p.kill()
        lf.close()
    return {
        "rep": rep,
        "scenario_wall_ms": scenario_wall,
        "total_tokens": total_tok,
        "throughput_tok_s": thr,
        "interactive_gap_p95_ms": results.get("qwen", {}).get("gap_p95_ms"),
        "qwen": results.get("qwen"),
        "gemma": results.get("gemma"),
        "errors": errors,
    }


def main():
    rows = []
    for i in range(1, REPS + 1):
        row = run_rep(i)
        rows.append(row)
        print(json.dumps({k: row[k] for k in row if k not in ("qwen", "gemma")}, indent=2), flush=True)
        time.sleep(30)
    thr = [r["throughput_tok_s"] for r in rows]
    gap = [r["interactive_gap_p95_ms"] for r in rows if r["interactive_gap_p95_ms"] is not None]
    summary = {
        "topology": "ax_two_process_single_model_each",
        "reps": rows,
        "thr_median": statistics.median(thr),
        "thr_mean": statistics.mean(thr),
        "gap_p95_median": statistics.median(gap) if gap else None,
        "note": "Probe only; not flip product target (managed_single_process).",
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "reps"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
