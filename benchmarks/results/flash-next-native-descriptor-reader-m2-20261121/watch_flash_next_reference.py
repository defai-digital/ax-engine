"""Supervise the pinned Flash Next reference and record system pressure."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time


POLL_SECONDS = 15
MAX_NO_PROGRESS_SECONDS = 20 * 60
CRITICAL_PRESSURE = (4, 8)


def vm_page_counters(text):
    fields = {
        "Pageins": "pageins", "Pageouts": "pageouts",
        "Swapins": "swapins", "Swapouts": "swapouts",
    }
    counts = {}
    for label, name in fields.items():
        match = re.search(rf"^{label}:\s+([\d,]+)\.", text, re.M)
        if match:
            counts[name] = int(match[1].replace(",", ""))
    return counts


def largest_other_rss(reference_pid):
    # Retain only a size. Process names and identifiers are not evidence fields.
    rows = subprocess.run(["/bin/ps", "-axo", "pid=,rss="], capture_output=True, text=True)
    if rows.returncode:
        return None
    sizes = [int(rss) * 1024 for pid, rss in (row.split() for row in rows.stdout.splitlines())
             if int(pid) != reference_pid]
    return max(sizes, default=None)


def sample_host(reference_pid=None, *, diagnose_host=False):
    pressure, swap = subprocess.check_output(
        ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level", "vm.swapusage"],
        text=True,
    ).splitlines()
    sample = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pressure": int(pressure),
        "swap": swap,
    }
    vm = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True)
    page_size = re.search(r"page size of (\d+) bytes", vm.stdout)
    compressor_pages = re.search(r"^Pages occupied by compressor:\s+(\d+)", vm.stdout, re.M)
    if vm.returncode == 0 and page_size and compressor_pages:
        sample["compressor_bytes"] = int(page_size[1]) * int(compressor_pages[1])
    if diagnose_host:
        if vm.returncode == 0:
            sample["vm_page_counters"] = vm_page_counters(vm.stdout)
        sample["largest_other_process_rss_bytes"] = largest_other_rss(reference_pid)
    if reference_pid is not None:
        child = subprocess.run(
            ["/bin/ps", "-p", str(reference_pid), "-o", "rss=,pcpu="],
            capture_output=True, text=True,
        )
        fields = child.stdout.split()
        if child.returncode == 0 and len(fields) == 2:
            sample["reference_rss_bytes"] = int(fields[0]) * 1024
            sample["reference_cpu_percent"] = float(fields[1])
    return sample


def generation(path):
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


def wait_for_reference(process, receipt, samples, *, stop_after=MAX_NO_PROGRESS_SECONDS,
                       diagnose_host=False):
    last_change = time.monotonic()
    previous = generation(receipt)
    while process.poll() is None:
        current = generation(receipt)
        now = time.monotonic()
        if current != previous:
            last_change = now
            previous = current
        sample = sample_host(process.pid, diagnose_host=diagnose_host)
        elapsed = round(now - last_change, 1)
        sample["seconds_since_progress"] = elapsed
        print(json.dumps(sample), file=samples, flush=True)
        if sample["pressure"] in CRITICAL_PRESSURE:
            return "critical system memory pressure"
        if elapsed >= stop_after:
            return "reference stopped writing progress"
        try:
            process.wait(timeout=POLL_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("python", "runner", "model-dir", "inventory", "ax-matrix", "out", "monitor"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--diagnose-host", action="store_true",
                        help="sample host VM counters and the largest other process RSS")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.monitor.mkdir(parents=True, exist_ok=False)

    command = [
        str(args.python), "-u", str(args.runner), "--model-dir", str(args.model_dir),
        "--inventory", str(args.inventory), "--ax-matrix", str(args.ax_matrix),
        "--output", str(args.out),
    ]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("AX_", "MLX_"))
           and k not in ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    receipt = args.out / "result.json"
    process = None
    with (args.monitor / "reference.log").open("w") as log, \
            (args.monitor / "samples.jsonl").open("w") as samples:
        before = sample_host(diagnose_host=args.diagnose_host)
        print(json.dumps(before), file=samples, flush=True)
        if before["pressure"] in CRITICAL_PRESSURE:
            stop_reason = "critical system memory pressure before launch"
        else:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
            try:
                stop_reason = wait_for_reference(process, receipt, samples, diagnose_host=args.diagnose_host)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=30)
            print(json.dumps(sample_host(diagnose_host=args.diagnose_host)), file=samples, flush=True)
    result = json.loads(receipt.read_text()) if receipt.exists() else {}
    exit_code = process.returncode if process is not None else None
    status = {
        "completed": result.get("completed") is True and exit_code == 0 and stop_reason is None,
        "exit_code": exit_code,
        "stop_reason": stop_reason,
        "phase": result.get("phase"),
        "qa_rows": len(result.get("qa", {}).get("rows", [])),
        "matrix_rows": len(result.get("matrix", {}).get("cells", [])),
    }
    (args.monitor / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status))
    return 0 if status["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
