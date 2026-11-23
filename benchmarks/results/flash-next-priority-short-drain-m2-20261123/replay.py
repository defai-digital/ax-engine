"""Recheck Flash Next priority-short decode completion and cancellation on M2."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path


FLAGS = ("AX_MLX_FLASH_NEXT_SHORT_PREFILL_FIRST", "AX_MLX_FLASH_NEXT_DRAIN_PRIORITY_SHORT")
HOLD_MARKER = "ax_scheduler_resident_short_decode_hold"


def read(path):
    raw = path.read_bytes()
    return gzip.decompress(raw) if path.suffix == ".gz" else raw


def route(row):
    return (row.get("route") or {}).get("crossover_decisions", {})


def check(receipt, pressure_by_arm):
    assert receipt["qualification"] is False
    assert receipt["pack_revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
    assert receipt["manifest"]["model-manifest.json"] == (
        "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"
    )
    long_id, short_id, replacement_id = receipt["prompt_ids"]
    long, short, replacement = (receipt["solo"][i] for i in receipt["prompt_ids"])
    assert all(len(ids) == 96 for ids in (long, short, replacement))
    assert len({long_id, short_id, replacement_id}) == 3
    mixed, drain = receipt["arms"]
    assert mixed["drain"] is False and drain["drain"] is True
    for label, arm in (("mixed", mixed), ("drain", drain)):
        assert arm["flags"][FLAGS[0]] == "1" and arm["flags"][FLAGS[1]] == str(int(arm["drain"]))
        grant = arm["grant"]
        assert grant["step"]["scheduled_tokens"] == 512 and grant["step"]["scheduled_requests"] == 1
        assert grant["long"]["processed_prompt_tokens"] == 2048
        assert grant["short"]["output_tokens"] == short[:1]
        assert [row["output_tokens"] for row in arm["completed"]] == [long, short]
        assert len(arm["drained"]) == 5 and not any(arm["drained"].values())
        pressure = pressure_by_arm[label]
        assert len(pressure) == arm["pressure_samples"] and pressure
        assert all(p["pressure"] == 1 for p in pressure)
        assert max(p["footprint_bytes"] for p in pressure) == arm["max_footprint_bytes"] <= 96 << 30
    first = mixed["first_long_step"]
    assert first["step"]["scheduled_requests"] == 2 and first["step"]["scheduled_tokens"] == 513
    assert first["long"]["processed_prompt_tokens"] == 2560
    assert first["short"]["output_tokens"] == short[:2] and not route(first["short"]).get(HOLD_MARKER)
    assert not mixed["held_decode"] and mixed["recovery"] is None

    frames = drain["held_decode"]
    assert len(frames) == 95
    for index, frame in enumerate(frames, 2):
        assert frame["long_cursor"] == 2048 and frame["short_length"] == index and frame["marker"]
        assert frame["step"]["scheduled_requests"] == frame["step"]["scheduled_tokens"] == 1
    steps = [frame["step"]["step_id"] for frame in frames]
    assert all(left < right for left, right in zip(steps, steps[1:]))
    assert frames[-1]["short_state"] == "finished"
    resume = drain["first_long_step"]
    assert resume["step"]["step_id"] > frames[-1]["step"]["step_id"]
    assert resume["step"]["scheduled_requests"] == 1 and resume["step"]["scheduled_tokens"] == 2048
    assert resume["long"]["processed_prompt_tokens"] == 4096
    recovery = drain["recovery"]
    assert recovery["cancelled"]["state"] == "cancelled"
    assert recovery["cancelled"]["output_tokens"] == short[:2]
    assert len(recovery["held"]) == 1 and recovery["held"][0]["long_cursor"] == 2048
    assert recovery["resume"]["scheduled_requests"] == 1 and recovery["resume"]["scheduled_tokens"] == 2048
    assert recovery["resumed_long"]["processed_prompt_tokens"] == 4096
    assert recovery["replacement_at_resume"]["processed_prompt_tokens"] == 0
    assert [row["output_tokens"] for row in recovery["completed"]] == [long, replacement]
    assert len(recovery["drained"]) == 5 and not any(recovery["drained"].values())
    return {
        "qualification": False,
        "matched_completed_output_ids": 576,
        "normal_pressure_samples": sum(arm["pressure_samples"] for arm in (mixed, drain)),
        "peak_footprint_bytes": max(arm["max_footprint_bytes"] for arm in (mixed, drain)),
        "held_short_decode_steps": 95,
        "held_long_cursor": 2048,
        "resumed_long_cursor": 4096,
        "cancel_and_replace_passed": True,
    }


def load(root):
    manifest = {name: digest for digest, name in map(str.split, (root/"inputs.sha256").read_text().splitlines())}
    build = json.loads((root/"build.json").read_bytes())
    assert build["codesign_verified"] and manifest["ax-engine-server"] == build["server_sha256"]
    assert sha256((root/"build.json").read_bytes()).hexdigest() == manifest["build.json"]
    assert sha256(read(root/"source.patch.gz")).hexdigest() == manifest["source.patch"] == build["patch_sha256"]
    assert sha256(read(root/"check_priority_drain.py.gz")).hexdigest() == manifest["check_priority_drain.py"]
    inventory = (root/"mxfp4-inventory.json").read_bytes()
    assert sha256(inventory).hexdigest() == manifest["mxfp4-inventory.json"]
    pack = json.loads(inventory)
    assert pack["revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3" and len(pack["members"]) == 49
    assert (root/"status.txt").read_text().strip() == "0"
    receipt = json.loads(read(root/"receipt.json.gz"))
    assert receipt["build"] == build
    return check(receipt, {
        arm: [json.loads(row) for row in read(root/f"{arm}-samples.jsonl.gz").splitlines()]
        for arm in ("mixed", "drain")
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path, default=Path(__file__).resolve().parent, nargs="?")
    args = cli.parse_args()
    print(json.dumps(load(args.root), indent=2, sort_keys=True, allow_nan=False))
