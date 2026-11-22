"""Replay the M5 Flash Next HTTP control that stopped on memory pressure."""

from collections import Counter
from gzip import decompress
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EARLY = ROOT.parent / "flash-next-server-prefill-early-m2-20261122"


def sums(path):
    return dict(line.split()[::-1] for line in path.read_text().splitlines())


def replay():
    spec = importlib.util.spec_from_file_location("early_arrival", EARLY / "replay.py")
    if spec is None or spec.loader is None:
        raise ValueError("Missing sealed driver verifier")
    early = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(early)
    need, read, digest = early.require, early.read, early.digest
    prior = sums(EARLY / "stage.sha256")
    current = sums(ROOT / "inputs.sha256")
    need(current.pop("run_m5.sh") == digest(ROOT / "run_m5.sh")
         and all(prior[name] == sha for name, sha in current.items())
         and set(prior) - set(current) == {"run_m2.sh", "run_early.py", "oracle.json", "oracle-receipt.json"},
         "The M5 schedule used different sealed prompts, workload or launcher")
    with early.inputs(EARLY) as (driver, _):
        cases, build, inventory, oracle, _, _ = driver.inputs()
        receipt = read(ROOT / "receipt.json")
        need(read(ROOT / "build.json") == build == receipt["build"]
             and digest(ROOT / "build.json") == current["build.json"]
             and (ROOT / "status.txt").read_text() == "1\n"
             and receipt["scope"] == "M5 ordered serving comparison"
             and receipt["qualification"] is False and receipt["arms"] == {} and receipt["status"] == "failed"
             and receipt["pack_revision"] == inventory["revision"]
             and receipt["inventory_sha256"] == oracle["inventory_sha256"]
             and receipt["manifest"] == oracle["manifest"],
             "Unexpected server, pack or terminal arm")
        import run_server_tail as tail

        raw = json.loads(decompress((ROOT / "physical-1.json.gz").read_bytes()))
        measured = tail.verify(raw, False)
        _, _, expected = tail.layer.files()["ending-decoder"]
        need(all(row["ok"] and row["prompt_token_ids"] == case.input_tokens
                 and row["output_token_ids"] == expected[case.id]
                 for row, case in zip(raw["observations"], cases, strict=True))
             and all(raw[key] == value for key, value in measured.items())
             and raw["drained"] == dict.fromkeys(tail.mixed.COUNTERS, 0),
             "The physical baseline changed output or drain")
        rows = [json.loads(line) for line in decompress((ROOT / "physical-1-samples.jsonl.gz").read_bytes()).splitlines()]
        histogram = Counter(row["pressure"] for row in rows)
        need(set(histogram) == {1, 2}, "The trace lacks the sampled memory warning")
        return {
            "scope": receipt["scope"], "qualification": False, "timing_eligible": False,
            "terminal_reason": "memory pressure exceeded normal in physical-1",
            "evaluated_only": "physical-1",
            "normal_pressure_samples": histogram[1], "warning_pressure_samples": histogram[2],
            "matching_output_ids": sum(len(row["output_token_ids"]) for row in raw["observations"]),
            "prefill_overlapped": measured["prefill_overlapped"],
            "peak_process_footprint_bytes": max(row["footprint_bytes"] for row in rows if row["footprint_bytes"]),
        }


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
