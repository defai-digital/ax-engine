"""Replay the fixed destination allocation comparison using the original inputs."""
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
base = runpy.run_path(str(ROOT.parent / "replay.py"))
result = ROOT / "result.json"
raw, _ = base["hashed_json"](result)
build, _ = base["hashed_json"](ROOT / "build.json")
base["need"]((ROOT / "status.txt").read_text() == "0\n" and build["mlx"] == build["mlx-metal"] == "0.32.4"
             and build["binary_sha256"] == raw["binary_sha256"] and build["driver_sha256"] == raw["driver_sha256"],
             "Fixed allocation run failed its terminal evidence binding")
print(json.dumps(base["replay"](result, base["INVENTORY"], ROOT.parent / "resident-shards.json",
                                base["STARTUP_ARCHIVE"], True), indent=2, allow_nan=False))
