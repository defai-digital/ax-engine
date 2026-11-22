"""Replay the M5 persistent PLE reader experiment with the shared verifier."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERIFIER = ROOT.parent / "flash-next-ple-reader-dispatch-m5-20261122" / "replay.py"
PRIOR = ROOT.parent / "flash-next-ple-reader-pool-m2-20261122"

if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("m5_reader_dispatch", VERIFIER)
    if spec is None or spec.loader is None:
        raise ValueError("Missing M5 reader verifier")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    print(json.dumps(runner.replay(ROOT, PRIOR), indent=2, allow_nan=False))
