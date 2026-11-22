"""Verify the M5 replay of the bounded Flash Next gate/up probe."""
from hashlib import sha256
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "flash-next-packed-expert-probe-m2-20261122"
RAW_SHA256 = "685eba6535c667e4c5bbf8f0f8ac049799f51fac725324afa5c476928bfc4292"


if __name__ == "__main__":
    common = runpy.run_path(str(BASE / "replay.py"))
    require = common["require"]
    require((ROOT / "status.txt").read_bytes() == b"0\n"
            and (ROOT / "pressure-after.txt").read_bytes() == b"1\n", "Incomplete or pressured M5 run")
    pre = json.loads((ROOT / "quiet.json").read_bytes())
    post = json.loads((ROOT / "idle-after.json").read_bytes())
    require(90 <= pre["cpu_idle_percent"] <= 100 and 90 <= post <= 100, "Host idle floor missed")
    raw = (ROOT / "result.json").read_bytes()
    require(sha256(raw).hexdigest() == RAW_SHA256, "M5 component receipt changed")
    print(json.dumps(common["replay"](json.loads(raw)), indent=2, sort_keys=True, allow_nan=False))
