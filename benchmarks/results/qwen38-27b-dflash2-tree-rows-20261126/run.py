#!/usr/bin/env python3
"""27B DFlash2 decode rate by output cap. usage: tree816.py <port> <model> <corpus.jsonl> <caps csv> <tokenizer.json>
One JSON line per (prompt, cap): completion tokens, seconds from first to last text, finish reason, text hash."""
import hashlib, http.client, json, sys, time
from tokenizers import Tokenizer
port, model, corpus, caps = int(sys.argv[1]), sys.argv[2], sys.argv[3], [int(c) for c in sys.argv[4].split(",")]
tok = Tokenizer.from_file(sys.argv[5])
prompts = [json.loads(l) for l in open(corpus)]
def stream(text, cap):
    body = json.dumps({"model": model, "prompt": text, "max_tokens": cap, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}})
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    start = time.monotonic(); first = last = None; usage = finish = None; parts = []
    conn.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    for raw in conn.getresponse():
        if not raw.startswith(b"data:"): continue
        data = raw[5:].strip()
        if data == b"[DONE]": break
        event = json.loads(data)
        usage = (event.get("usage") or {}).get("completion_tokens", usage)
        for choice in event.get("choices", []):
            if choice.get("text"):
                last = time.monotonic(); first = first or last; parts.append(choice["text"])
            finish = choice.get("finish_reason") or finish
    conn.close()
    text_out = "".join(parts)
    first_tokens = len(tok.encode(parts[0], add_special_tokens=False).ids) if parts else 0
    return {"tokens": usage, "first_chunk_tokens": first_tokens, "ttft": round(first - start, 4),
            "decode_seconds": round(last - first, 4), "finish": finish,
            "sha": hashlib.sha256(text_out.encode()).hexdigest()[:12]}
stream(prompts[0]["input_text"], 32)
for cap in caps:
    for p in prompts:
        row = stream(p["input_text"], cap)
        print(json.dumps({"id": p["id"], "cap": cap, **row}), flush=True)
