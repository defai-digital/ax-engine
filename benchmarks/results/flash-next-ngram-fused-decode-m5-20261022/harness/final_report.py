import json
rows={}
for lab in ("g_base","g_io","g_full"):
    for kind in ("s","l"):
        d=json.load(open(f"~/build/fn-ab/out-{lab}-{kind}.json"))
        for r in d["rows"]:
            rows.setdefault(r["file"],{})[lab]=r
print("prompt tokens | TTFT base / io / full | decode base / io / full | sha equal")
for f,v in rows.items():
    b,i,u=v["g_base"],v["g_io"],v["g_full"]
    print(f"{f:9s} {b['prompt_tokens']:5d} | {b['ttft_s']:6.2f} {i['ttft_s']:6.2f} {u['ttft_s']:6.2f} | {b['decode_tok_s']:5.1f} {i['decode_tok_s']:5.1f} {u['decode_tok_s']:5.1f} | {b['text_sha']==i['text_sha']==u['text_sha']}")
