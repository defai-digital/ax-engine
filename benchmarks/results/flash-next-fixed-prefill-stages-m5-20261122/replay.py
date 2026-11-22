"""Bind the fixed-geometry M5 stage attribution to the verified M2 probe."""
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
M2 = ROOT.parent / "flash-next-fixed-prefill-stages-m2-20261122"
sys.path.insert(0, str(M2))
from summarize import audit  # noqa: E402

def require(ok, message):
    if not ok:
        raise ValueError(message)

def replay():
    inputs = {name: checksum for checksum, name in
              (line.split() for line in (ROOT / "inputs.sha256").read_text().splitlines())}
    m2 = json.loads((M2 / "run.json").read_bytes())
    for name in ("profile-probe", "prompt-8192.json"):
        require(inputs[name] == m2["inputs_sha256"][name], f"Changed {name}")
    require((ROOT / "status.txt").read_bytes() == b"0\n", "Profile runner failed")
    idle = [json.loads((ROOT / "quiet.json").read_bytes())["cpu_idle_percent"],
            json.loads((ROOT / "idle-after.json").read_bytes())]
    require(all(math.isfinite(value) and 0 <= value <= 100 for value in idle), "Invalid CPU idle")
    raw = (ROOT / "attribution.json").read_bytes()
    profile = json.loads(raw)
    timing = audit(profile)
    attrs = profile["traced_stages"]
    families = {"moe_seconds": sum(attrs[name] for name in ("moe_compute", "moe_routing")),
                "qsa_seconds": sum(value for name, value in attrs.items() if name.startswith("qsa")),
                "gdn_seconds": sum(attrs[name] for name in ("gdn", "gdn_chunked"))}
    return {"scope": "Fixed-width model-only M5 stage attribution", "qualification": False,
            "raw_sha256": sha256(raw).hexdigest(), "timing_cpu_idle_percent": idle,
            "timing_cpu_idle_floor_met": all(90 <= value <= 100 for value in idle),
            "stage_family_seconds": families, **timing}

if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
