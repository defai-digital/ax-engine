"""Replay the three frozen cross-runtime prompts with the signed native server."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

bundle, pack, out = map(Path, sys.argv[1:])
sys.path.insert(0, str(bundle))
import run_qwen38_reference_serving as control
from run_flash_next_prefill_split_probe import MODEL, RESERVE_OPTIONS, generate
from run_selective_mixed import wait_drained

for name in list(os.environ):
    if name.startswith(("AX_", "MLX_")):
        del os.environ[name]
prefix = subprocess.check_output(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True).strip()
assert prefix == "1"
assert not any(
    Path(s.strip()).name == "ax-engine-server"
    for s in subprocess.check_output(["ps", "-axo", "comm="], text=True).splitlines()
)
build = json.loads((bundle / "build.json").read_bytes())
binary = bundle / "ax-engine-server"
assert build["server_sha256"] == sha256(binary.read_bytes()).hexdigest()
assert build["codesign_verified"]
assert sha256((pack/"model-manifest.json").read_bytes()).hexdigest() == "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"
subprocess.run(["codesign", "--verify", binary], check=True)
corpus = [json.loads(line) for line in (bundle/"mixed-corpus.jsonl").read_bytes().splitlines()]
oracle = json.loads((bundle/"prior-output.json").read_bytes())
wanted = [next(x for x in corpus if len(x["input_tokens"]) == n) for n in (512,2048,8192)]
options = ("--disable-ngram-acceleration", "--total-blocks", "2048", "--max-batch-tokens",
           "2049", *RESERVE_OPTIONS)
env = {"AX_MLX_FLASH_NEXT_ALIGNED_BUFFERED": "0", "AX_MLX_FLASH_NEXT_DIRECT_COALESCING": "0",
       "AX_MLX_FLASH_NEXT_PREFILL_SPLIT": "0", "RUST_LOG": "info,ax_engine_mlx::flash_next_startup=debug"}
args = SimpleNamespace(binary=binary, out=out)
rows = []
with control.server(args, "native", MODEL, pack, 1, env=env, options=options,
                    stream_experts="auto", guard=True) as (base, pid):
    for row in wanted:
        a = generate(base, row | {"max_output_tokens": 16})
        b = generate(base, row | {"max_output_tokens": 16})
        assert a["output_tokens"] == b["output_tokens"]
        rows.append({"id": row["id"], "input_tokens": len(row["input_tokens"]),
                     "prompt_sha256": sha256(json.dumps(row["input_tokens"], separators=(",", ":")).encode()).hexdigest(),
                     "m5_ids": oracle[row["id"]][:16], "m2_ids": a["output_tokens"],
                     "prefill": [r["cursor"] for r in a["prefill_progress"]]})
        print(row["id"], len(row["input_tokens"]), a["output_tokens"] == oracle[row["id"]][:16], flush=True)
    counters = wait_drained(base)
log = (out/"native.log").read_text()
assert "decision=resident" in log and "reader=Mapped" in log
samples = [json.loads(line) for line in (out/"native-samples.jsonl").read_text().splitlines()]
assert samples and all(row["pressure"] == 1 for row in samples)
result = {"build": build, "rows": rows, "drained": counters, "pressure_count": len(samples),
          "max_footprint_bytes": max(row["footprint_bytes"] for row in samples)}
(out/"native-result.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
