"""Observe Darwin host CPU counters alongside an unchanged reference runner."""
from argparse import ArgumentParser, REMAINDER
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

CPU_LOAD_INFO = 3
CPU_STATES = ("user", "system", "idle", "nice")
UINT32_MASK = (1 << 32) - 1


def clock_ticks():
    library = ctypes.CDLL(None)
    library.mach_host_self.argtypes = ()
    library.mach_host_self.restype = ctypes.c_uint
    library.host_statistics.argtypes = (ctypes.c_uint, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_uint),
                                        ctypes.POINTER(ctypes.c_uint))
    library.host_statistics.restype = ctypes.c_int
    host = library.mach_host_self()

    def sample():
        ticks = (ctypes.c_uint * len(CPU_STATES))()
        count = ctypes.c_uint(len(ticks))
        error = library.host_statistics(host, CPU_LOAD_INFO, ticks, ctypes.byref(count))
        if error or count.value != len(ticks):
            raise RuntimeError(f"Unexpected HOST_CPU_LOAD_INFO result ({error}, {count.value})")
        return dict(zip(CPU_STATES, ticks, strict=True))

    return sample


def tick_delta(before, after):
    return {state: (after[state] - before[state]) & UINT32_MASK for state in CPU_STATES}


def observer(result, output, command):
    phase, previous_mtime = "prelaunch", None

    def read_phase():
        nonlocal phase, previous_mtime
        try:
            changed = result.stat().st_mtime_ns
        except FileNotFoundError:
            return phase
        if changed != previous_mtime:
            phase = json.loads(result.read_bytes())["phase"]
            previous_mtime = changed
        return phase

    sample = clock_ticks()
    with output.open("x") as lines, subprocess.Popen(command) as process:
        before = sample()
        prior_at, prior_phase = time.monotonic(), read_phase()
        while True:
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                code = None
            after = sample()
            now, phase = time.monotonic(), read_phase()
            delta = tick_delta(before, after)
            total = sum(delta.values())
            ratio = 100 * delta["idle"] / total if total else None
            record = {"utc": datetime.now(timezone.utc).isoformat(),
                      "monotonic_seconds": now,
                      "interval_seconds": now - prior_at,
                      "previous_phase": prior_phase, "phase": phase,
                      "ticks": after, "deltas": delta, "host_idle_percent": ratio}
            print(json.dumps(record, allow_nan=False), file=lines, flush=True)
            if code is not None:
                return code
            before, prior_at, prior_phase = after, now, phase


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("the original reference command is required")
    raise SystemExit(observer(args.result, args.output, command))
