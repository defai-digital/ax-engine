#!/usr/bin/env python3
"""Verify the sampled Flash Next first-layer prefill operator comparison."""

import json
from pathlib import Path
import runpy
import struct
import tarfile

HERE = Path(__file__).resolve().parent
GEOMETRY_DIR = HERE.parent / "flash-next-prefill-geometry-divergence-m5-20261123"
STAGES = (
    "embedding", "ple_operator", "attention_hc_read",
    "gdn_query_input", "gdn_key_input", "gdn_value_input", "gdn_log_decay", "gdn_beta",
    "gdn_core_output", "gdn_gated_output", "gdn",
    "qsa_normalized_query", "qsa_normalized_key", "qsa_attention_before_gate", "qsa",
    "attention_hc_write", "mlp_hc_read", "mlp_hc_write",
)
POSITIONS = [0, 1, 510, 511, 512, 513, 1023, 1024, 1535, 1536, 2046, 2047]


def member(archive, name):
    item = archive.extractfile(name)
    assert item is not None
    return item.read()


def read(archive, name):
    return json.loads(member(archive, name))


def logit_and_state_match(trace, initial, width):
    arm = f"observed/{width}"
    name = f"{arm}/result/result.json"
    current, earlier = (read(source, name) for source in (trace, initial))
    assert current == earlier
    for stage in ("position-8192.logits.f32", "position-8193.logits.f32"):
        name = f"{arm}/result/{stage}"
        assert member(trace, name) == member(initial, name)
    return current["cuts"]


def samples(archive, width, cuts):
    prefix = f"observed/{width}/operator/"
    records = {}
    for filename in archive.getnames():
        if not (filename.startswith(prefix) and filename.endswith(".json")):
            continue
        row = read(archive, filename)
        stage, pos = row["stage"], row["position"]
        assert stage in STAGES and pos in POSITIONS
        assert row["layer"] in (0, None)
        left, right = row["chunk_start"], row["chunk_start"] + row["chunk_len"]
        assert [left, right] in cuts and left + row["local_index"] == pos < right
        assert (stage, pos) not in records
        values = member(archive, filename.removesuffix(".json") + ".f32le")
        count = 1
        for dimension in row["shape"]:
            count *= dimension
        assert len(values) == 4 * count
        records[(stage, pos)] = values, row["shape"]
    assert records
    return records


def compare(trace, initial):
    result = runpy.run_path(str(GEOMETRY_DIR / "replay.py"))
    overall = result["replay"](trace, result["PRIOR"])
    assert member(trace, "build.json") == member(initial, "build.json")
    controls = {}
    for width in (512, 2048):
        cuts = logit_and_state_match(trace, initial, width)
        run = read(trace, f"observed/{width}/receipt.json")
        assert run["flags"]["AX_FLASH_NEXT_DUMP_LAYER"] == "0"
        assert list(map(int, run["flags"]["AX_FLASH_NEXT_DUMP_POSITIONS"].split(","))) == POSITIONS
        controls[width] = samples(trace, width, cuts)
    narrow, wide = controls[512], controls[2048]
    assert narrow.keys() == wide.keys()
    summaries = []
    for name in STAGES:
        pairs = []
        for position in POSITIONS:
            if (name, position) not in narrow:
                continue
            base, candidate = narrow[name, position], wide[name, position]
            assert base[1] == candidate[1]
            a, b = (struct.unpack(f"<{len(values)//4}f", values) for values, _ in (base, candidate))
            pairs.append({"position": position, "exact_bits": base[0] == candidate[0],
                          "max_abs_delta": max(abs(x - y) for x, y in zip(a, b, strict=True))})
        if pairs:
            summaries.append({"stage": name, "rows": pairs})
    assert summaries[0]["stage"] == "embedding"
    assert set(pair["position"] for pair in summaries[0]["rows"]) == set(POSITIONS)
    first_diff = next(item["stage"] for item in summaries
                      if any(not row["exact_bits"] for row in item["rows"]))
    return {"qualification": False, "earliest_recorded_stage_with_drift": first_diff,
            "normal_pressure_samples": overall["normal_pressure_samples"],
            "verified_http_token_ids": overall["matched_http_token_ids"], "operator_summaries": summaries}


if __name__ == "__main__":
    with tarfile.open(HERE / "evidence.tar.xz") as trace, tarfile.open(
        GEOMETRY_DIR / "evidence.tar.xz"
    ) as initial:
        print(json.dumps(compare(trace, initial), indent=2, sort_keys=True))
