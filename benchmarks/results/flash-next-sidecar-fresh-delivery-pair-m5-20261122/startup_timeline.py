"""Capture and replay startup intervals from the verified Flash Next delivery receipts."""

from argparse import ArgumentParser
from datetime import datetime
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
import tarfile

MODES = ("disabled", "required", "default")
ARMS = ("mapped", "prefetch")
MARKERS = {
    "memory_plan": "memory plan",
    "prefetch_complete": "Flash Next MTP sidecar prefetch completed bytes=",
    "runner_admission": "MTP admission at runner construction",
    "listener": "preview listening bind_address=",
}
UTC = re.compile(r"^(\d{4}-\d\d-\d\dT[0-9:.]+Z)\s")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def need(valid, reason):
    if not valid:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_bytes())


def extract(raw):
    found = {}
    for line in ANSI.sub("", raw.decode()).splitlines():
        for event, marker in MARKERS.items():
            if marker not in line:
                continue
            stamp = UTC.match(line)
            need(stamp is not None and event not in found, f"Ambiguous startup log event {event}")
            found[event] = stamp.group(1)
    return {"original_log_sha256": sha256(raw).hexdigest(), "timestamps": found}


def capture(raw, comparison, source):
    result = {}
    for arm in comparison["arms"]:
        result[arm] = {}
        for mode in MODES:
            name = f"native-support/server-{mode}.log"
            record = extract((raw / arm / name).read_bytes())
            need(record["original_log_sha256"] == source[arm][name]["original"],
                 f"Source log is not bound to the frozen {arm}/{mode} receipt")
            result[arm][mode] = record
    return result


def measure(events, readiness, prefetch):
    names = ("memory_plan", "prefetch_complete", "runner_admission", "listener") if prefetch else (
        "memory_plan", "runner_admission", "listener")
    need(tuple(events)==names and type(readiness) in (int,float) and isfinite(readiness) and readiness>0,
         "Startup log route or readiness receipt changed")
    stamps = {name:datetime.fromisoformat(events[name]) for name in names}
    need(all(left<right for left,right in zip(stamps.values(),list(stamps.values())[1:])),
         "Startup markers are out of order")
    part = (stamps["runner_admission"]-stamps["memory_plan"]).total_seconds()
    serving = (stamps["listener"]-stamps["runner_admission"]).total_seconds()
    outside = readiness-part-serving
    need(outside>=-0.001, "Observed listener interval exceeds process-to-models readiness")
    result = {"http_ready_seconds":readiness, "memory_plan_to_runner_admission_seconds":round(part,3),
              "runner_admission_to_listener_seconds":round(serving,3),
              "outside_logged_intervals_seconds":round(outside,3)}
    if prefetch:
        until = (stamps["prefetch_complete"]-stamps["memory_plan"]).total_seconds()
        result["memory_plan_to_prefetch_completion_seconds"]=round(until,3)
    return result


def compare(captured, comparison, provenance):
    receipt = comparison["arms"]
    need(tuple(captured)==tuple(receipt) and tuple(captured) in (ARMS,ARMS[:1]),
         "Startup timing metadata uses the wrong delivery arms")
    results = {}
    for arm in captured:
        need(tuple(captured[arm])==MODES, "Incomplete native startup phases")
        results[arm]={}
        for mode, item in captured[arm].items():
            name=f"native-support/server-{mode}.log"
            need(item["original_log_sha256"]==provenance[arm][name]["original"],
                 f"Startup marker source digest differs for {arm}/{mode}")
            official=receipt[arm]["delivery_readiness"]["delivered"][mode]
            results[arm][mode]=measure(item["timestamps"],official,arm=="prefetch")
    result={"scope":"info-level log intervals and the measured native readiness receipt",
            "runtime_commit":comparison["runtime_commit"],"arms":results}
    if comparison["timing_eligible"] and tuple(results)==ARMS:
        result["prefetch_minus_mapped_seconds"]={
            mode:{name:round(results["prefetch"][mode][name]-baseline,3)
                  for name,baseline in results["mapped"][mode].items()}
            for mode in MODES}
        need(all(result["prefetch_minus_mapped_seconds"][mode]["http_ready_seconds"] ==
                 comparison["prefetch_minus_mapped_ready_seconds"][mode] for mode in MODES),
             "Observed total startup deltas differ")
    return result


if __name__ == "__main__":
    cli=ArgumentParser(description=__doc__)
    cli.add_argument("--comparison",type=Path,required=True)
    cli.add_argument("--events",type=Path,required=True)
    group=cli.add_mutually_exclusive_group(required=True)
    group.add_argument("--archive",type=Path)
    group.add_argument("--raw",type=Path)
    cli.add_argument("--provenance",type=Path)
    args=cli.parse_args()
    comparison=read(args.comparison)
    if args.raw:
        provenance=read(args.provenance)
        events=capture(args.raw,comparison,provenance)
        with args.events.open("x") as output:
            json.dump(events,output,indent=2,allow_nan=False)
            output.write("\n")
    else:
        with tarfile.open(args.archive) as package:
            provenance=json.load(package.extractfile("observed/source-sha256.json"))
        events=read(args.events)
    print(json.dumps(compare(events,comparison,provenance),indent=2,allow_nan=False))
