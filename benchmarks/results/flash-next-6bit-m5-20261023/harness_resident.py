import json,time,urllib.request,subprocess,os,sys
srv=subprocess.Popen([os.path.expanduser("~/build/ax-qual-20261023/target/release/ax-engine-server"),"--mlx","--mlx-model-artifacts-dir",os.path.expanduser("~/build/packs/ax-qwen38-flash-next-6bit"),"--model-id","fn6","--port","31561","--mlx-mtp-policy","disabled","--max-concurrent-requests","1","--total-blocks","4096","--stream-experts","off"],stdout=open(os.path.expanduser("~/build/six-m5-resident.log"),"w"),stderr=subprocess.STDOUT,env=dict(os.environ,RUST_LOG="info"))
t=time.time()
while True:
    try: urllib.request.urlopen("http://127.0.0.1:31561/health",timeout=2); break
    except Exception:
        if srv.poll() is not None: print("SERVER DIED", srv.returncode); sys.exit(1)
        time.sleep(3)
print("ready_s", round(time.time()-t,1), flush=True)
def rss(): return int(subprocess.check_output(["ps","-o","rss=","-p",str(srv.pid)]).decode())/1048576
def swap(): return subprocess.check_output(["sysctl","-n","vm.swapusage"]).decode().strip()
def ask(text,n):
    body=json.dumps({"model":"fn6","messages":[{"role":"user","content":text}],"temperature":0,"max_tokens":n,"stream":True,"stream_options":{"include_usage":True},"chat_template_kwargs":{"enable_thinking":False}}).encode()
    req=urllib.request.Request("http://127.0.0.1:31561/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    t0=time.perf_counter();first=None;last=None;out=[];usage=None
    with urllib.request.urlopen(req,timeout=3600) as r:
        for raw in r:
            l=raw.decode().strip()
            if not l.startswith("data:") or l.endswith("[DONE]"): continue
            d=json.loads(l[5:])
            if d.get("usage"): usage=d["usage"]
            ch=d.get("choices") or []
            if ch and ch[0].get("delta",{}).get("content"):
                now=time.perf_counter(); first=first or now; last=now; out.append(ch[0]["delta"]["content"])
    n=(usage or {}).get("completion_tokens",len(out))
    return dict(ttft=round(first-t0,2), decode=round((n-1)/(last-first),2) if last and last>first else None, prompt=(usage or {}).get("prompt_tokens"), tokens=n, text="".join(out)[:120])
P=os.path.expanduser("~/build/fn-ab/prompts/")
try:
    for q,n in [("What is the capital of France? Answer in one word.",16),("Write a Python function that returns the n-th Fibonacci number iteratively.",64),("Write a Python function that returns the n-th Fibonacci number iteratively.",64)]:
        r=ask(q,n); r["rss_gib"]=round(rss(),1); r["swap"]=swap(); print(json.dumps(r),flush=True); open(os.path.expanduser("~/build/six-m5-resident-rows.jsonl"),"a").write(json.dumps(r)+"\n")
finally:
    srv.terminate(); srv.wait(30)
