"""Replay matched M5 Flash Next mixed arrivals and scheduler traces."""
from hashlib import sha256
import importlib.util
import json
from math import isfinite
from pathlib import Path
from statistics import median
import tarfile

ROOT = Path(__file__).resolve().parent
EARLY = ROOT.parent / "flash-next-server-prefill-early-m2-20261122"
LATE = ROOT.parent / "flash-next-server-prefill-tail-fixed-m5-20261122"
CORPUS = ROOT.parent / "flash-next-mxfp4-mixed-serving-m5-20261121" / "evidence.tar.xz"
ARMS = ("physical-1", "held-1", "held-2", "physical-2")
TRIALS = 12
NEW_FILES = {"run_sustained_tail.py", "mixed-corpus.jsonl", "await_tail.sh", "run_tail_serialized.sh"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_archive(path):
    files = {}
    with tarfile.open(path) as archive:
        for member in archive:
            name = Path(member.name)
            require(member.isfile() and not name.is_absolute() and ".." not in name.parts
                    and member.name not in files, "Unexpected archive member")
            with archive.extractfile(member) as stream:
                files[member.name] = stream.read()
    return files


def sha(data):
    return sha256(data).hexdigest()


def load_early():
    spec = importlib.util.spec_from_file_location("sustained_tail_workload", EARLY / "replay.py")
    require(spec and spec.loader, "Missing sealed workload verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hashes(data):
    rows = [line.split() for line in data.decode().splitlines()]
    entries = {name: digest for digest, name in rows}
    require(len(entries) == len(rows), "Ambiguous input manifest")
    return entries


def verify_arms(data, receipt, driver, cases, expected):
    require(tuple(receipt["arms"]) == ARMS and receipt["observed_workload"] == [
        {"short": short.id, "long": long.id, "at": at} for short, long, at in cases
    ] and len(cases) == TRIALS, "The workload or treatment order changed")
    arms = {}
    for label in ARMS:
        enabled = label.startswith("held-")
        arm = receipt["arms"][label]
        idle = arm["cpu_idle_percent"]
        require(len(idle) == 2 and all(isfinite(x) and 0 <= x <= 100 for x in idle), "Invalid CPU samples")
        samples = [json.loads(line) for line in data[f"{label}-samples.jsonl"].splitlines()]
        require(samples and {row["pressure"] for row in samples} == {1}
                and len(samples) == arm["normal_pressure_samples"], "Pressure receipt changed")
        trials = []
        for idx, ((short, long, at), recorded) in enumerate(zip(cases, arm["trials"], strict=True), 1):
            raw_bytes = data[f"{label}/burst-{idx:02d}.json"]
            raw = json.loads(raw_bytes)
            checks = driver.verify(raw, (short, long), expected, at, enabled)
            require(raw["drained"] == dict.fromkeys(driver.mixed.COUNTERS, 0)
                    and all(raw[key] == value for key, value in checks.items())
                    and all(isfinite(checks[name]) and checks[name] > 0
                            for name in ("short_max_gap_ms", "long_trigger_ttft_ms"))
                    and recorded == checks | {"sha256": sha(raw_bytes), "trigger": at},
                    "Measured burst no longer agrees with its raw trace and drain")
            observations = raw["observations"]
            elapsed = max(r["started_at_s"] + r["e2e_latency_ms"] / 1000 for r in observations)
            elapsed -= observations[0]["started_at_s"]
            require(isfinite(elapsed) and elapsed > 0, "Invalid burst window")
            trials.append(checks | {"trigger": at, "short": short.id, "long": long.id, "seconds": elapsed})
        late = [t for t in trials if t["trigger"] >= driver.TRIGGER]
        held = sum(t["hold_steps"] > 0 for t in late)
        gap = median(t["short_max_gap_ms"] for t in late)
        ttft = median(t["long_trigger_ttft_ms"] for t in late)
        require(arm["matching_output_ids"] == 192 * len(trials)
                and len(late) == 8 and (held > 0 if enabled else held == 0)
                and arm["late_with_holds"] == held and arm["late_median_short_max_gap_ms"] == gap
                and arm["late_median_long_trigger_ttft_ms"] == ttft, "Late cohort measurements changed")
        arms[label] = {
            "timing_eligible": min(idle) >= 90, "cpu_idle_percent": idle,
            "normal_pressure_samples": len(samples), "matching_output_ids": arm["matching_output_ids"],
            "late_with_holds": held, "late_with_shared_first_prefill": sum(t["first_prefill_overlaps"] for t in late),
            "late_median_short_max_gap_ms": gap, "late_median_long_trigger_ttft_ms": ttft,
            "late_output_tokens_per_second": 192 * len(late) / sum(t["seconds"] for t in late),
            "trials": trials,
        }
    return arms


def replay():
    data = read_archive(ROOT / "evidence.tar.xz")
    prior = read_archive(LATE / "evidence.tar.xz")
    hashes_before = hashes(prior["inputs.sha256"])
    hashes_now = hashes(data["sustained-inputs.sha256"])
    require(hashes_now.keys() - hashes_before.keys() == NEW_FILES
            and hashes_before.keys() <= hashes_now.keys()
            and all(hashes_before[k] == hashes_now[k] for k in hashes_before if k != "run_m5.sh")
            and all(sha(data[f]) == hashes_now[f] for f in (NEW_FILES - {"mixed-corpus.jsonl"}) | {"run_m5.sh"})
            and data["build.json"] == prior["build.json"], "Measured runner or runtime provenance changed")
    receipt = json.loads(data["receipt.json"])
    baseline = json.loads(prior["receipt.json"])
    build = json.loads(data["build.json"])
    capacity = json.loads(data["dispatch-capacity.json"])["capacity"]
    require(receipt["status"] == "complete" and data["status.txt"] == b"0\n"
            and receipt["qualification"] is False
            and receipt["scope"] == "M5 repeated early and late mixed-length arrivals"
            and receipt["build"] == baseline["build"] == build
            and receipt["revision"] == baseline["pack_revision"]
            and all(receipt[k] == baseline[k] for k in ("manifest", "inventory_sha256"))
            and capacity["pressure"] == 1 and capacity["eligible_for_attempt"] is False
            and capacity["reason"] == "background-rss-exceeds-signed-margin",
            "The result did not bind the earlier native run and dispatch guard")
    with tarfile.open(CORPUS) as archive:
        corpus = archive.extractfile("mixed-corpus.jsonl").read()
    require(sha(corpus) == hashes_now["mixed-corpus.jsonl"], "The frozen offered workload changed")
    early = load_early()
    with early.inputs(EARLY) as (_, folder):
        (folder / "run_sustained_tail.py").write_bytes(data["run_sustained_tail.py"])
        (folder / "mixed-corpus.jsonl").write_bytes(corpus)
        import run_sustained_tail as driver

        cases, expected = driver.inputs()
        arms = verify_arms(data, receipt, driver, cases, expected)
    pairs = {}
    for n in (1, 2):
        control, held = (arms[f"{label}-{n}"] for label in ("physical", "held"))
        if control["timing_eligible"] and held["timing_eligible"]:
            pairs[str(n)] = {k: [control[k], held[k]] for k in (
                "late_median_short_max_gap_ms", "late_median_long_trigger_ttft_ms", "late_output_tokens_per_second"
            )}
    return {"scope": receipt["scope"], "qualification": False, "runtime_commit": build["runtime_commit"],
            "matching_output_ids": sum(a["matching_output_ids"] for a in arms.values()),
            "arms": arms, "timing_eligible_late_pairs": pairs}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
