#!/usr/bin/env python3
"""Reduce the raw lane outputs of this campaign to slim per-lane records.

usage: reduce_lanes.py <raw-dir> <out-dir> <tokenizer-dir>

Each raw lane (`q36_35b_<suite>_<lane>_<pass>.json`) becomes
`<out-dir>/<suite>_<lane>_<pass>.json` holding, per prompt: the prompt token count and
the per-repetition decode, prefill (engine-reported and client time-to-first-token),
time-to-first-token and end-to-end rates, the draft acceptance and a SHA-256 of the
generated text. Uses the lane loaders of `summarize_fair.py` (same directory).
"""
import hashlib
import importlib.util
import json
import pathlib
import re
import sys

here = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("summarize_fair", here / "summarize_fair.py")
source = (here / "summarize_fair.py").read_text().replace("\nmain()\n", "\n")  # import without running
module = importlib.util.module_from_spec(spec)
exec(compile(source, str(here / "summarize_fair.py"), "exec"), module.__dict__)

raw, out, tokdir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(tokdir)
out.mkdir(parents=True, exist_ok=True)
pattern = re.compile(r"^q36_35b_(flappy|long_code)_(.+)_([ABC])\.json$")
count = 0
for path in sorted(raw.iterdir()):
    match = pattern.match(path.name)
    if not match:
        continue
    suite, lane, label = match.groups()
    cases = module.loader(lane)(json.load(open(path)), tok)
    record = {"suite": suite, "lane": lane, "pass": label, "prompts": {}}
    for cid, case in cases.items():
        text = case.get("text")
        record["prompts"][cid] = {
            "prompt_tokens": case["prompt_tokens"],
            "decode_tok_s": case["decode"],
            "prefill_tok_s_engine": case["prefill_engine"],
            "prefill_tok_s_client_ttft": case["prefill_client"],
            "ttft_ms": case["ttft_ms"],
            "end_to_end_tok_s": case["e2e"],
            "draft_acceptance": case["accept"],
            "text_sha256": hashlib.sha256(text.strip().encode()).hexdigest() if text else None,
        }
    (out / f"{suite}_{lane}_{label}.json").write_text(json.dumps(record, indent=1) + "\n")
    count += 1
print(f"wrote {count} lane records to {out}")
