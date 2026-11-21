#!/usr/bin/env python3
"""Adapt a single Flash Next sidecar to the selected weight runner."""

import json
from pathlib import Path
import subprocess
import sys

work = json.loads(Path(sys.argv[1]).read_text())
entry, = work["files"]
assert len(entry["names"]) == len(set(entry["names"])) == 31
binary = Path(__file__).with_name("full_sidecar")
result = json.loads(subprocess.check_output([binary, entry["path"], sys.argv[2]]))
observation, = result["files"]
assert observation["tensors"] == 31
assert observation["selected_bytes"] == 5214301696
assert observation["sha256"] == "3c110ca89b3e22236d2eb5d36095f760f3b89b7d8a20a41e8b2ecdcc11d48871"
print(json.dumps(result))
