#!/usr/bin/env python3
"""Flash Next A/B: start a server, send novel prompts, record TTFT and decode rate.

usage: fn_ab.py <server-bin> <label> <out.json> <prompt-file>...
Each prompt file is a text file used verbatim as the user message. The page cache
is not purged between prompts, so every prompt must be novel for cold n-gram rows.
"""
import hashlib, json, os, signal, subprocess, sys, time, urllib.request

bin_, label, out = sys.argv[1:4]
files = sys.argv[4:]
pack = os.path.expanduser("~/build/packs/ax-qwen38-flash-next-mxfp4-ff2a2848")
port = 31540
env = dict(os.environ, RUST_LOG="info")
log = open(os.path.expanduser(f"~/build/fn-ab/server-{label}.log"), "w")
srv = subprocess.Popen([bin_, "--mlx", "--mlx-model-artifacts-dir", pack, "--model-id", "fn", "--port", str(port),
                        "--mlx-mtp-policy", "disabled", "--disable-ngram-acceleration", "--total-blocks", os.environ.get("FN_AB_TOTAL_BLOCKS", "2048"),
                        "--prefill-chunk", "2048", "--max-batch-tokens", "2048"],
                       stdout=log, stderr=subprocess.STDOUT, env=env)
try:
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); break
        except Exception:
            time.sleep(2)
    def ask(text, max_tokens):
        body = json.dumps({"model": "fn", "messages": [{"role": "user", "content": text}], "temperature": 0,
                           "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True},
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter(); first = None; last = None; chunks = 0; usage = None; text_out = []
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"): continue
                d = json.loads(line[5:])
                if d.get("usage"): usage = d["usage"]
                ch = d.get("choices") or []
                if ch and ch[0].get("delta", {}).get("content"):
                    now = time.perf_counter()
                    if first is None: first = now
                    last = now; chunks += 1; text_out.append(ch[0]["delta"]["content"])
        return dict(ttft=first - t0, chunks=chunks, decode_s=(last - first) if first else 0, usage=usage,
                    text="".join(text_out))
    ask("Say hello.", 8)  # weights + kernels warm
    pre = os.environ.get("FN_AB_PRE_REQUEST_CMD")
    if pre:  # e.g. warm part of the page cache AFTER the server has loaded its weights
        print("pre-request:", subprocess.run(pre, shell=True, capture_output=True, text=True).stdout.strip(), flush=True)
    rows = []
    for f in files:
        text = open(f).read()
        r = ask(text, int(os.environ.get("FN_AB_MAX_TOKENS", "128")))
        n = (r["usage"] or {}).get("completion_tokens", r["chunks"]); p = (r["usage"] or {}).get("prompt_tokens")
        dec = (n - 1) / r["decode_s"] if r["decode_s"] > 0 and n > 1 else None
        rows.append(dict(file=os.path.basename(f), prompt_tokens=p, completion_tokens=n, ttft_s=round(r["ttft"], 3),
                         prefill_tok_s=round(p / r["ttft"], 1) if p else None,
                         decode_tok_s=round(dec, 2) if dec else None, text_head=r["text"][:80], text_sha=hashlib.sha256(r["text"].encode()).hexdigest()[:16]))
        print(label, json.dumps(rows[-1]), flush=True)
    json.dump(dict(label=label, bin=bin_, rows=rows), open(out, "w"), indent=1)
finally:
    srv.send_signal(signal.SIGTERM)
    try: srv.wait(30)
    except Exception: srv.kill()
