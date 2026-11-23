#!/usr/bin/env python3
"""Record Darwin host, server and client CPU counters around a benchmark script."""

from argparse import ArgumentParser, REMAINDER
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import math
import os
from pathlib import Path
import runpy
import sys
from threading import Event
import time


CPU_STATES = ("user", "system", "idle", "nice")
HOST_CPU_LOAD_INFO = 3
RUSAGE_INFO_V0 = 0
UINT32_MASK = (1 << 32) - 1


class ProcessUsage(ctypes.Structure):
    # rusage_info_v0 in the macOS SDK's sys/resource.h.
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "user", "system", "package_wakeups", "interrupt_wakeups", "pageins", "wired",
            "resident", "footprint", "start", "exit",
        )
    ]


def native_reader(server_pid):
    mach = ctypes.CDLL(None)
    mach.mach_host_self.argtypes = ()
    mach.mach_host_self.restype = ctypes.c_uint
    mach.host_statistics.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_uint))
    mach.host_statistics.restype = ctypes.c_int
    mach.mach_timebase_info.argtypes = (ctypes.c_void_p,)
    mach.mach_timebase_info.restype = ctypes.c_int
    timebase = (ctypes.c_uint32 * 2)()
    if mach.mach_timebase_info(ctypes.byref(timebase)) != 0 or not all(timebase):
        raise RuntimeError("Darwin returned an invalid Mach timebase")
    numerator, denominator = timebase
    host = mach.mach_host_self()

    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    usage = libproc.proc_pid_rusage
    usage.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ProcessUsage))
    usage.restype = ctypes.c_int
    server_started = None

    def read():
        nonlocal server_started
        began = time.perf_counter_ns()
        ticks = (ctypes.c_uint32 * len(CPU_STATES))()
        count = ctypes.c_uint(len(ticks))
        status = mach.host_statistics(host, HOST_CPU_LOAD_INFO, ctypes.byref(ticks), ctypes.byref(count))
        if status != 0 or count.value != len(ticks):
            raise RuntimeError("Darwin HOST_CPU_LOAD_INFO failed")
        process = ProcessUsage()
        if usage(server_pid, RUSAGE_INFO_V0, ctypes.byref(process)) != 0:
            raise OSError(ctypes.get_errno(), "Darwin could not sample the serving process")
        if not process.start or process.exit or (server_started is not None and process.start != server_started):
            raise RuntimeError("The serving process changed during the observation")
        server_started = process.start
        client_ns = time.process_time_ns()
        return {
            "perf_counter_ns": began,
            "read_duration_ns": time.perf_counter_ns() - began,
            "host_ticks": dict(zip(CPU_STATES, ticks, strict=True)),
            "server_cpu_ns": (process.user + process.system) * numerator // denominator,
            "client_cpu_ns": client_ns,
        }

    return read


def record(args):
    if sys.platform != "darwin":
        raise ValueError("Serving CPU recording requires macOS")
    if args.server_pid <= 0 or args.server_pid == os.getpid():
        raise ValueError("The serving process must be separate from the client")
    if not math.isfinite(args.interval) or args.interval < 0.25:
        raise ValueError("The sample interval must be at least 0.25 seconds")
    script, *script_args = args.script
    script = Path(script).resolve(strict=True)
    if not script.is_file():
        raise ValueError("The benchmark entry must be a Python file")
    read = native_reader(args.server_pid)
    stop = Event()

    with args.output.open("x") as output:
        def write(row):
            output.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            output.flush()

        def sample_until_done():
            while not stop.wait(args.interval):
                write({"kind": "sample", **read()})

        write({"kind": "metadata", "version": 1, "host_ticks_per_second": os.sysconf("SC_CLK_TCK"),
               "interval_seconds": args.interval})
        write({"kind": "start", **read()})
        with ThreadPoolExecutor(max_workers=1) as pool:
            worker = pool.submit(sample_until_done)
            try:
                sys.path.insert(0, str(script.parent))
                sys.argv = [str(script), *script_args]
                runpy.run_path(str(script), run_name="__main__")
            except SystemExit as exit_status:
                if exit_status.code not in (None, 0):
                    raise
            finally:
                stop.set()
            worker.result()
        write({"kind": "end", **read()})


def summarize(trace):
    metadata, *rows = [json.loads(line) for line in trace.read_text().splitlines()]
    if metadata["kind"] != "metadata" or metadata["version"] != 1:
        raise ValueError("Unknown serving CPU trace format")
    tick_hz = metadata["host_ticks_per_second"]
    if type(tick_hz) is not int or tick_hz <= 0:
        raise ValueError("Invalid host timebase")
    expected = ["start", *(["sample"] * (len(rows) - 2)), "end"]
    if [row["kind"] for row in rows] != expected:
        raise ValueError("The CPU trace did not cover a completed client run")

    for row in rows:
        ticks = row["host_ticks"]
        if set(ticks) != set(CPU_STATES) or any(
            type(value) is not int or not 0 <= value <= UINT32_MASK for value in ticks.values()
        ):
            raise ValueError("Invalid Darwin host counters")
        fields = ("perf_counter_ns", "read_duration_ns", "client_cpu_ns", "server_cpu_ns")
        if any(type(row[field]) is not int or row[field] < 0 for field in fields):
            raise ValueError("Invalid process or wall-clock counters")

    intervals = []
    for old, new in zip(rows, rows[1:]):
        old_ticks, new_ticks = (row["host_ticks"] for row in (old, new))
        deltas = {name: (new_ticks[name] - old_ticks[name]) & UINT32_MASK for name in CPU_STATES}
        client = new["client_cpu_ns"] - old["client_cpu_ns"]
        server = new["server_cpu_ns"] - old["server_cpu_ns"]
        elapsed = new["perf_counter_ns"] - old["perf_counter_ns"]
        if elapsed <= 0 or client < 0 or server < 0:
            raise ValueError("Non-monotonic CPU observation")
        intervals.append({"wall_ns": elapsed, "host_total_ticks": sum(deltas.values()),
                          "host_idle_ticks": deltas["idle"], "client_cpu_ns": client, "server_cpu_ns": server})

    totals = {key: sum(row[key] for row in intervals) for key in intervals[0]}
    if not totals["host_total_ticks"]:
        raise ValueError("No host ticks during client measurement")
    host_busy = (totals["host_total_ticks"] - totals["host_idle_ticks"]) / tick_hz
    server_cpu = totals["server_cpu_ns"] / 1e9
    client_cpu = totals["client_cpu_ns"] / 1e9
    return {
        "client_window_seconds": totals["wall_ns"] / 1e9,
        "host_idle_percent": 100 * totals["host_idle_ticks"] / totals["host_total_ticks"],
        "host_busy_core_seconds": host_busy,
        "server_cpu_seconds": server_cpu,
        "client_cpu_seconds": client_cpu,
        "unassigned_busy_core_seconds": host_busy - server_cpu - client_cpu,
        "sample_count": len(rows),
        "max_read_duration_ns": max(row["read_duration_ns"] for row in rows),
        "intervals": intervals,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    run = modes.add_parser("record", help="Record CPU counters around a Python benchmark")
    run.add_argument("--server-pid", type=int, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--interval", type=float, default=5.0)
    run.add_argument("script", nargs=REMAINDER)
    replay = modes.add_parser("summarize", help="Recompute CPU time over a completed client trace")
    replay.add_argument("trace", type=Path)
    args = parser.parse_args()
    if args.mode == "summarize":
        print(json.dumps(summarize(args.trace), indent=2, allow_nan=False))
    else:
        if args.script[:1] == ["--"]:
            args.script = args.script[1:]
        if not args.script:
            parser.error("record needs a Python benchmark script")
        record(args)


if __name__ == "__main__":
    main()
