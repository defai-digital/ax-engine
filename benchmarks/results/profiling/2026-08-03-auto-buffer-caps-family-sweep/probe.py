#!/usr/bin/env python3
"""Minimal MTP-prefill A/B probe replicating the matrix contract.

Launches ax-engine-server, sends warmups + measured reps of one fixed prompt,
and reports per-rep prefill runner-time tok/s from the SSE stream (same source
as server_sse_runner_time_us in the benchmark artifacts).
"""
import argparse
import http.client
import json
import os
import subprocess
import sys
import time


def iter_sse(lines):
    event, data = "", []
    for raw in lines:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        line = raw.rstrip("\r\n")
        if line == "":
            if data:
                try:
                    yield event, json.loads("\n".join(data))
                except json.JSONDecodeError:
                    pass
            event, data = "", []
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            v = line[5:]
            data.append(v[1:] if v.startswith(" ") else v)
    if data:
        try:
            yield event, json.loads("\n".join(data))
        except json.JSONDecodeError:
            pass


def one_run(port, tokens, gen_tokens, sampler, seed=1234, text=None):
    body = {"max_output_tokens": gen_tokens,
            "sampling": {**sampler, "ignore_eos": True, "seed": seed}}
    if text is not None:
        body["input_text"] = text
    else:
        body["input_tokens"] = tokens
    payload = json.dumps(body).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    started = time.perf_counter()
    prefill_us = 0
    output_tokens = 0
    first_wall = None
    try:
        conn.request("POST", "/v1/generate/stream", body=payload,
                     headers={"Content-Type": "application/json",
                              "Accept": "text/event-stream"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {resp.read(300)!r}")
        seen_prefill = False
        for event, obj in iter_sse(resp):
            if event != "step":
                continue
            step = obj.get("step", {})
            runner_us = int(step.get("runner_time_us") or 0)
            out_len = obj.get("request", {}).get("output_len")
            if out_len is not None:
                output_tokens = int(out_len)
                if output_tokens > 0 and first_wall is None:
                    first_wall = time.perf_counter() - started
            if not seen_prefill:
                prefill_us += runner_us
                seen_prefill = True
    finally:
        conn.close()
    total = time.perf_counter() - started
    return {
        "prefill_us": prefill_us,
        "prefill_tok_s": (len(tokens) / (prefill_us / 1e6)) if (prefill_us and tokens) else None,
        "ttft_ms": (first_wall or 0) * 1000,
        "output_tokens": output_tokens,
        "total_s": total,
    }


def _fmt(r):
    pre = (f"prefill {r['prefill_tok_s']:.1f} tok/s" if r["prefill_tok_s"]
           else f"prefill {r['prefill_us']/1000:.0f} ms")
    return (f"{pre} ttft {r['ttft_ms']:.0f} ms gen {r['output_tokens']} "
            f"decode~{r['output_tokens']/max(r['total_s']-r['ttft_ms']/1000,0.01):.1f} tok/s")


def wait_ready(port, proc, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited rc={proc.returncode}")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/health")
            r = conn.getresponse()
            r.read()
            conn.close()
            if r.status < 500:
                return
        except OSError:
            pass
        time.sleep(2)
    raise RuntimeError("server did not become ready")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-binary", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt-tokens-json",
                    help="json file with token_ids list")
    ap.add_argument("--prompt-text-file",
                    help="plain text file used as input_text instead")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--gen-tokens", type=int, default=1000)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--label", default="run")
    ap.add_argument("--extra-env", action="append", default=[],
                    help="KEY=VAL, repeatable")
    args = ap.parse_args()

    tokens = []
    text = None
    if args.prompt_text_file:
        with open(args.prompt_text_file) as f:
            text = f.read()
    elif args.prompt_tokens_json:
        with open(args.prompt_tokens_json) as f:
            tokens = json.load(f)["token_ids"]
    else:
        ap.error("one of --prompt-tokens-json or --prompt-text-file is required")

    env = {**os.environ, "AX_MLX_NATIVE_CONFIRM": "1",
           "AX_ENGINE_PREFIX_REUSE_DISABLED": "1",
           "AX_MLX_PREFIX_CACHE_MAX_BYTES": "0",
           "AX_MLX_PREFIX_CACHE_MAX_ENTRIES": "0",
           "AX_MLX_PREFIX_CACHE_DISK_DISABLED": "1"}
    for kv in args.extra_env:
        k, _, v = kv.partition("=")
        env[k] = v

    cmd = [args.server_binary, "--mlx", "--mlx-model-artifacts-dir", args.model_dir,
           "--model-id", "probe", "--port", str(args.port)]
    print(f"[probe] launch: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, env=env)
    import threading
    def drain():
        while proc.stderr.readline():
            pass
    threading.Thread(target=drain, daemon=True).start()
    sampler = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    try:
        t0 = time.time()
        wait_ready(args.port, proc)
        print(f"[probe] ready in {time.time()-t0:.1f}s", file=sys.stderr)
        for i in range(args.warmups):
            r = one_run(args.port, tokens, args.gen_tokens, sampler, text=text)
            print(f"[{args.label}] warmup{i+1}: {_fmt(r)}", flush=True)
        results = []
        for i in range(args.reps):
            r = one_run(args.port, tokens, args.gen_tokens, sampler, text=text)
            results.append(r)
            print(f"[{args.label}] rep{i+1}: {_fmt(r)}", flush=True)
        vals = [r["prefill_us"] for r in results]
        print(f"[{args.label}] SUMMARY prefill_us: "
              f"rep1={vals[0]} plateau_mean={sum(vals[1:])/max(len(vals)-1,1):.0f} "
              f"mean={sum(vals)/len(vals):.0f}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
