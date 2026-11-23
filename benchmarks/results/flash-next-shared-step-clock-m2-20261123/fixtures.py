"""Rebuild the bounded native shared-step inputs from the sealed natural-load run."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile


SELECTION = "Median-ranked and maximum matched output gap per physical width"


def member(archive, path):
    source = archive.extractfile(path)
    assert source is not None
    with source:
        return json.load(source)


def extract_arm(archive, comparison, order):
    arm = f"mixed-{order}"
    path = f"observed/{arm}/{arm}"
    workload = member(archive, path + ".json")
    traces = {row["request_id"]: row for row in member(archive, path + "-steps.json")}
    rows = {row["prompt_id"]: row for row in workload["observations"] if row["phase"] == "measured"}
    for short_id, long_id, priority_cut in comparison[order]["matched_priority_turns"]:
        short, long = rows[short_id], rows[long_id]
        short_steps = traces[short["request_id"]]["decode_step_ids"]
        long_prefill = traces[long["request_id"]]["prefill"]
        by_step = {step["step_id"]: step for step in long_prefill}
        assert len(short_steps) == len(short["output_token_ids"]) == 96
        for index, gap in enumerate(short["stream_step_interval_ms"], start=1):
            end_step = short_steps[index]
            cut = by_step.get(end_step)
            if end_step != short_steps[index - 1] + 1 or not cut or cut["width"] not in (511, 512):
                continue
            assert cut["scheduled"] == 2
            prefix = [step["width"] for step in long_prefill if step["step_id"] < end_step]
            assert cut["cursor"] - cut["width"] == sum(prefix)
            yield {
                "long_completed_widths": prefix, "long_id": long_id,
                "long_prefill_cursor_before_step": sum(prefix), "long_prompt_ids": long["prompt_token_ids"],
                "observed_logical_end": cut["split_end"], "observed_mixed_prefill_width": cut["width"],
                "original_short_priority_long_cursor": priority_cut,
                "short_emitted_ids_before_step": short["output_token_ids"][:index], "short_id": short_id,
                "short_prompt_ids": short["prompt_token_ids"],
                "short_token_at_shared_step": short["output_token_ids"][index],
                "source_arm": arm, "source_output_gap_ms": gap, "source_step": end_step,
            }


def fixtures(source):
    source = source.resolve()
    comparison = (source / "comparison.json").read_bytes()
    assert subprocess.check_output([sys.executable, "-B", source / "replay.py"]) == comparison
    recorded = json.loads(comparison)
    bundle = source / "evidence.tar.xz"
    with tarfile.open(bundle) as archive:
        candidates = [row for order in ("first", "second")
                      for row in extract_arm(archive, recorded["comparison"], order)]
    maxima = {}
    for row in candidates:
        key = row["source_arm"], row["short_id"]
        if key not in maxima or row["source_output_gap_ms"] > maxima[key]["source_output_gap_ms"]:
            maxima[key] = row
    assert len(maxima) == sum(len(row["matched_priority_turns"]) for row in recorded["comparison"].values())
    cases = []
    for width in (511, 512):
        options = sorted((row for row in maxima.values() if row["observed_mixed_prefill_width"] == width),
                         key=lambda row: row["source_output_gap_ms"])
        cases.extend((options[len(options) // 2], options[-1]))
    return {"cases": cases, "selection": SELECTION, "server_sha256": recorded["server_sha256"],
            "source_archive_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest()}


if __name__ == "__main__":
    print(json.dumps(fixtures(Path(sys.argv[1])), indent=2, sort_keys=True, allow_nan=False))
