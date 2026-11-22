"""Sample the existing M5 delivery owner without changing the release inputs."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from time import monotonic, sleep

STAGE = Path(sys.argv[1])
CPU = re.compile(r"([0-9]+(?:\.[0-9]+)?)% idle")
MODES = ("disabled", "required", "default")


def run(command):
    return subprocess.run(command, capture_output=True, text=True, check=False, timeout=12)


def state():
    delivery = STAGE / "delivery"
    support = delivery / "native-support"
    record = delivery / "result.json"
    completed = None
    if record.is_file():
        try:
            completed = json.loads(record.read_text())["completed"]
        except (ValueError, KeyError):
            pass
    modes = {}
    for mode in MODES:
        if (support / f"server-{mode}.log").is_file():
            cases = [p for p in support.glob(f"{mode}-*.json") if p.is_file()]
            modes[mode] = {"written_cases": len(cases)}
    return {
        "delivery_completed": completed,
        "native_controls_recorded": (support / "result.json").is_file(),
        "modes": modes,
    }


def main():
    pid = (STAGE / "owner.pid").read_text().strip()
    begin = monotonic()
    while monotonic() - begin < 2700:
        if run(["/bin/ps", "-p", pid, "-o", "stat="]).returncode:
            print(json.dumps({"event": "owner_exited"}), flush=True)
            return
        before = state()
        at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        top = run(["/usr/bin/top", "-l", "2", "-s", "3", "-n", "0", "-stats", "pid"])
        values = CPU.findall(top.stdout) if top.returncode == 0 else []
        pressure = run(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"]).stdout.strip()
        after = state()
        print(json.dumps({
            "event": "sample", "utc": at, "seconds_since_observer_start": round(monotonic() - begin, 3),
            "cpu_idle_percent": float(values[-1]) if values else None,
            "memory_pressure_level": int(pressure) if pressure.isdecimal() else None,
            "before": before, "after": after,
        }, allow_nan=False), flush=True)
        if after["delivery_completed"] is True:
            return
        sleep(12)
    print(json.dumps({"event": "observer_deadline"}), flush=True)


if __name__ == "__main__":
    main()
