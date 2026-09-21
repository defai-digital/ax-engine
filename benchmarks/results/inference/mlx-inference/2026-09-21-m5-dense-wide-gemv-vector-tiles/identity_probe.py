"""Greedy output-token identity probe: baseline vs patched ax-engine-server on the same prompts."""
import glob, http.client, json, os, signal, subprocess, sys, time
OUT = os.path.expanduser("~/bench-ab-20260921")
PACK = "/Users/akiralam/.cache/huggingface/hub/models--AutomatosX--AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP/snapshots/3e290738e96972307c6aeb9934ab170ca0eae1c1"
PORT = 31499
prompts = sorted(glob.glob(f"{OUT}/flappy_baseline-prompts/*.json") + glob.glob(f"{OUT}/long_code_baseline-prompts/*.json"))
assert len(prompts) == 8, prompts
env = {**os.environ, "AX_MLX_NATIVE_CONFIRM": "1", "AX_ENGINE_PREFIX_REUSE_DISABLED": "1",
       "AX_MLX_PREFIX_CACHE_MAX_BYTES": "0", "AX_MLX_PREFIX_CACHE_MAX_ENTRIES": "0",
       "AX_MLX_QWEN_LINEAR_MTP_EXACT": "1", "AX_MLX_QWEN_LINEAR_MTP_CERTIFICATION_CANDIDATE": "1"}
results = {}
for arm in ("baseline", "patched"):
    binary = f"{OUT}/ax-engine-server.{arm}"
    cmd = [binary, "--mlx", "--mlx-model-artifacts-dir", PACK, "--model-id", PACK, "--port", str(PORT),
           "--mlx-mtp-disable-ngram-stacking", "--prefill-chunk", "2048", "--max-batch-tokens", "2048"]
    log = open(f"{OUT}/identity_{arm}_server.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        time.sleep(3)
        try:
            c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=3); c.request("GET", "/health"); r = c.getresponse()
            if r.status == 200: break
        except Exception: pass
    else:
        proc.kill(); sys.exit(f"{arm}: server never became healthy")
    arm_out = {}
    for path in prompts:
        toks = json.load(open(path))["token_ids"]
        payload = json.dumps({"input_tokens": toks, "max_output_tokens": 256,
                              "sampling": {"ignore_eos": True, "seed": 0, "temperature": 0.0}}).encode()
        for attempt in range(2):
            c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=600)
            c.request("POST", "/v1/generate", body=payload, headers={"Content-Type": "application/json"})
            r = c.getresponse(); body = json.loads(r.read())
            assert r.status == 200, body
            out = body.get("output_tokens") or body.get("tokens") or body
            arm_out.setdefault(os.path.basename(path), []).append(out)
    results[arm] = arm_out
    proc.send_signal(signal.SIGTERM)
    try: proc.wait(60)
    except subprocess.TimeoutExpired: proc.kill()
    time.sleep(3)
json.dump(results, open(f"{OUT}/identity_outputs.json", "w"))
ok = True
for name in results["baseline"]:
    b = results["baseline"][name]; p = results["patched"][name]
    same_arm = b[0] == b[1] and p[0] == p[1]
    cross = b[0] == p[0]
    n = len(b[0]) if isinstance(b[0], list) else -1
    print(f"{name}: tokens={n} repeat_stable={same_arm} baseline==patched={cross}")
    ok = ok and same_arm and cross
print("IDENTITY", "PASS" if ok else "FAIL")
