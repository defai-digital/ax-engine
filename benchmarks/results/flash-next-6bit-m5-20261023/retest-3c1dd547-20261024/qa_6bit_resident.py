#!/usr/bin/env python3
"""Frozen 105-item Flash Next QA cohort on the affine 6-bit pack with experts resident.

Reuses scripts/run_flash_next_qa_cohort.py (items, request bodies, grader) and only
adds `--stream-experts off` to the owned server, because Auto streams this pack's
experts from SSD on a 128 GB host. Not a qualification run (no inventory binding).
usage: qa_6bit_resident.py <repo> <server-bin> <pack> <output>
"""
import json, sys
from pathlib import Path
repo, server, pack, out = (Path(a) for a in sys.argv[1:5])
sys.path.insert(0, str(repo / "scripts"))
import check_flash_next_native_support as native
import run_flash_next_qa_cohort as qa
base_command = native.server_command
def resident_command(server, root, mode, port):
    command = base_command(server, root, mode, port)
    if "--stream-experts" in command:
        command[command.index("--stream-experts") + 1] = "off"
    else:
        command[-4:-4] = ["--stream-experts", "off"]
    return command
native.server_command = resident_command
items = qa.load_items(qa.FROZEN / "full-qa-items.json")
out.mkdir(parents=True, exist_ok=False)
runs = {mode: qa.run_mode(server, pack, out, mode, items, False) for mode in ("disabled", "default")}
grades = {mode: qa.grade(items, {r["id"]: r["text"] for r in run["rows"]}) for mode, run in runs.items()}
summary = qa.summarize(items, runs, grades)
json.dump({"pack": pack.name, "stream_experts": "off", "summary": summary, "grades": grades},
          open(out / "result.json", "w"), indent=2, default=str)
print(json.dumps(summary, indent=1, default=str)[:3000])
