#!/usr/bin/env python3
"""Verify the Flash Next decay projection and elementwise stage witnesses."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
import struct
import tarfile

HERE = Path(__file__).resolve().parent
RECURRENCE = HERE.parent / "flash-next-gdn-recurrence-drift-m5-20261123"
STAGES = (
    "projection", "biased_projection", "negative_abs_exp", "log1p",
    "softplus", "log_decay", "decay",
)
POSITIONS = [493, 494, 495, 1039, 1040, 1041]


def projection_input(archive, helper, width, cuts):
    root = f"observed/{width}"
    read, member = helper["read"], helper["member"]
    receipt = read(archive, f"{root}/receipt.json")
    assert receipt["flags"]["AX_FLASH_NEXT_DUMP_LAYER"] == "0"
    assert receipt["flags"]["AX_FLASH_NEXT_DUMP_STAGES"] == "attention_hc_read"
    files = receipt["projection_input_probes"]
    assert len(files) == len(POSITIONS) * 2
    output = {}
    for file, digest in files.items():
        data = member(archive, f"{root}/operator/{file}")
        assert sha256(data).hexdigest() == digest
        if not file.endswith(".json"):
            continue
        entry = json.loads(data)
        position = entry["position"]
        start, length = entry["chunk_start"], entry["chunk_len"]
        assert [start, start + length] in cuts
        assert start + entry["local_index"] == position < start + length
        assert entry["stage"] == "attention_hc_read" and entry["layer"] == 0
        assert position not in output
        output[position] = files[file.removesuffix(".json") + ".f32le"]
    assert sorted(output) == POSITIONS
    return output


def row_bits(helper, item):
    record = item["rows"]
    helper["validate"](record)
    assert len(record["shape"]) == 3
    witnesses = item["witnesses"]
    assert [row["position"] for row in witnesses] == POSITIONS
    rows = {}
    for row in witnesses:
        position, bits = row["position"], row["bits"]
        assert len(bits) == record["shape"][-1]
        assert sha256(struct.pack(f"<{len(bits)}I", *bits)).hexdigest() == record["rows"][position]
        rows[position] = bits
    return rows


def changed_heads(left, right):
    def f32(bits):
        return struct.unpack("<f", struct.pack("<I", bits))[0]

    return [
        {"head": head, "narrow_bits": a, "wide_bits": b, "abs_delta": abs(f32(a) - f32(b))}
        for head, (a, b) in enumerate(zip(left, right, strict=True)) if a != b
    ]


def replay(archive, earlier, helper):
    read, member = helper["read"], helper["member"]
    assert sha256((RECURRENCE / "evidence.tar.xz").read_bytes()).hexdigest() == (
        member(archive, "prior-recurrence.sha256").decode().strip()
    )
    with tarfile.open(helper["GEOMETRY"] / "evidence.tar.xz") as geometry, tarfile.open(
        helper["OPERATOR"] / "evidence.tar.xz"
    ) as operator:
        control = helper["run"](archive, geometry, operator)

    both, heads, projection_inputs = [], [], []
    for width in (512, 2048):
        root = f"observed/{width}/result"
        recurrence = read(archive, f"{root}/recurrence.json")
        stages = recurrence.pop("decay_stages")
        assert recurrence == read(earlier, f"{root}/recurrence.json")
        assert sorted(stages) == sorted(STAGES)
        assert stages["decay"]["rows"] == recurrence["input_rows"]["decay"]
        both.append(stages)
        heads.append({name: row_bits(helper, value) for name, value in stages.items()})
        cuts = read(archive, f"{root}/result.json")["cuts"]
        projection_inputs.append(projection_input(archive, helper, width, cuts))

    stages = {}
    for name in STAGES:
        narrow, wide = (control[name]["rows"] for control in both)
        changes = {
            str(position): diff for position in POSITIONS
            if (diff := changed_heads(heads[0][name][position], heads[1][name][position]))
        }
        stages[name] = helper["compare"](narrow, wide) | {
            "different_positions": [
                i for i, (a, b) in enumerate(zip(narrow["rows"], wide["rows"], strict=True))
                if a != b
            ],
            "changed_heads": changes,
        }

    return {
        "qualification": False,
        "source_base": control["source_base"],
        "macos": control["macos"],
        "matched_http_token_ids": control["matched_http_token_ids"],
        "matched_recurrence_receipts": 2,
        "normal_pressure_samples": control["normal_pressure_samples"],
        "peak_sampled_footprint_bytes": control["peak_sampled_footprint_bytes"],
        "different_projection_input_positions": [
            pos for pos in POSITIONS if projection_inputs[0][pos] != projection_inputs[1][pos]
        ],
        "earliest_stage_with_drift": next(name for name in STAGES if stages[name]["different_rows"]),
        "decay_stages": stages,
    }


if __name__ == "__main__":
    helper = runpy.run_path(str(RECURRENCE / "replay.py"))
    with tarfile.open(HERE / "evidence.tar.xz") as current, tarfile.open(
        RECURRENCE / "evidence.tar.xz"
    ) as earlier:
        print(json.dumps(replay(current, earlier, helper), indent=2, sort_keys=True))
