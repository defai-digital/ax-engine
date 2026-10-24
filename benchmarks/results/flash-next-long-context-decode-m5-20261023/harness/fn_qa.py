#!/usr/bin/env python3
"""Flash Next server-level QA for the fused stretches.

usage: fn_qa.py <server-bin> <label> <scenario> [prompt files...]
scenarios:
  texts   one session, every prompt once (greedy 128 tokens); record text SHA-256, TTFT, decode tok/s
  soak    N back-to-back 256-token requests of the first prompt; record decode tok/s and server RSS
  cancel  start the last prompt, drop the connection after 2 s (mid-prefill), then run the first prompt
Environment (AX_FLASH_NEXT_FUSE, AX_MLX_COMPILE_CACHE_REFRESH_THRESHOLD, ...) is inherited by the server.
Output: JSON on stdout (one line) plus per-request lines on stderr.
"""
import hashlib, json, os, signal, subprocess, sys, time, urllib.request, http.client

bin_, label, scenario = sys.argv[1:4]
files = sys.argv[4:]
pack = os.path.expanduser("~/build/packs/ax-qwen38-flash-next-mxfp4-ff2a2848")
port = 31550
logdir = os.path.expanduser("~/build/fn-ab")
log = open(f"{logdir}/server-qa-{label}.log", "w")
srv = subprocess.Popen([bin_, "--mlx", "--mlx-model-artifacts-dir", pack, "--model-id", "fn", "--port", str(port),
                        "--mlx-mtp-policy", os.environ.get("QA_MTP_POLICY", "disabled"), "--disable-ngram-acceleration", "--total-blocks", os.environ.get("QA_TOTAL_BLOCKS", "2048"),
                        "--prefill-chunk", "2048", "--max-batch-tokens", "2048"],
                       stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, RUST_LOG="info"))


def rss_mb():
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(srv.pid)], capture_output=True, text=True).stdout.strip()
    return int(out) / 1024 if out else None


def body(text, max_tokens):
    return json.dumps({"model": "fn", "messages": [{"role": "user", "content": text}], "temperature": 0,
                       "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()


def ask(text, max_tokens=128):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body(text, max_tokens),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = last = None; usage = None; parts = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                now = time.perf_counter()
                first = first or now
                last = now
                parts.append(ch[0]["delta"]["content"])
    text_out = "".join(parts)
    n = (usage or {}).get("completion_tokens", len(parts))
    dec = (n - 1) / (last - first) if first and last and last > first and n > 1 else None
    return dict(ttft=round(first - t0, 3) if first else None, tokens=n,
                prompt_tokens=(usage or {}).get("prompt_tokens"),
                decode_tok_s=round(dec, 2) if dec else None,
                sha=hashlib.sha256(text_out.encode()).hexdigest()[:16])


def cancel_midway(text, after_s=2.0):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    conn.request("POST", "/v1/chat/completions", body=body(text, 128), headers={"Content-Type": "application/json"})
    time.sleep(after_s)
    conn.close()


result = dict(label=label, scenario=scenario, rows=[])
try:
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            break
        except Exception:
            time.sleep(2)
    first_hello = ask("Say hello.", 8)  # cold weights + first compile of the decode shapes
    result["first_hello"] = first_hello
    result["rss_start_mb"] = rss_mb()
    prompts = [open(f).read() for f in files]
    if scenario == "texts":
        for f, p in zip(files, prompts):
            r = ask(p, 128); r["file"] = os.path.basename(f); result["rows"].append(r)
            print(json.dumps(r), file=sys.stderr, flush=True)
    elif scenario == "soak":
        for i in range(int(os.environ.get("QA_SOAK_N", "30"))):
            r = ask(prompts[0], 256); r["i"] = i; r["rss_mb"] = rss_mb(); result["rows"].append(r)
            print(json.dumps(r), file=sys.stderr, flush=True)
    elif scenario == "cancel":
        ref = ask(prompts[0], 128); result["reference"] = ref
        cancel_midway(prompts[-1], 2.0)
        time.sleep(1)
        after = ask(prompts[0], 128); result["after_cancel"] = after
        again = ask(prompts[-1], 64); result["long_after_cancel"] = again
        result["rows"] = [ref, after]
    result["rss_end_mb"] = rss_mb()
    print(json.dumps(result))
finally:
    srv.send_signal(signal.SIGTERM)
    try:
        srv.wait(30)
    except Exception:
        srv.kill()
