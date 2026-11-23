"""Reconcile the same-host peer observations with the AX M5 and M2 controls."""

from hashlib import sha256
import json
from pathlib import Path
import tarfile


HERE = Path(__file__).resolve().parent
NATURAL = HERE.parent / "flash-next-current-natural-load-m5-20261123" / "evidence.tar.xz"
REFERENCE_REVISIONS = {
    "omlx": "138d7617d39653b39a5cbc379b1b5907a4481d67",
    "bounded": "6102cb4ad1a5b3cc38d8dc7e6cbe2aca395596cb",
}


def rows(name):
    return [json.loads(line) for line in (HERE / name).read_bytes().splitlines()]


def shared_prefix(left, right):
    assert len(left) == len(right) == 16
    return next((index for index, (a, b) in enumerate(zip(left, right)) if a != b), 16)


def compare():
    native = json.loads((HERE / "native-result.json").read_bytes())
    with tarfile.open(NATURAL) as archive:
        corpus = {
            row["id"]: row
            for line in archive.extractfile("bundle/mixed-corpus.jsonl")
            if (row := json.loads(line))
        }
        oracle = json.load(archive.extractfile("bundle/prior-output.json"))
        assert native["build"] == json.load(archive.extractfile("build.json"))
        inventory = archive.extractfile("bundle/mxfp4-inventory.json").read()
    pack = json.loads((HERE / "pack-verified.json").read_bytes())
    assert pack["inventory_sha256"] == sha256(inventory).hexdigest()
    assert pack["pack_revision"] == json.loads(inventory)["revision"]
    assert pack["verified_members"] == len(json.loads(inventory)["members"]) == 49
    native_rows = {row["id"]: row for row in native["rows"]}
    assert sorted(row["input_tokens"] for row in native_rows.values()) == [512, 2048, 8192]
    for key, row in native_rows.items():
        prompt = corpus[key]["input_tokens"]
        assert row["prompt_sha256"] == sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
        assert row["m5_ids"] == oracle[key][:16] and row["input_tokens"] == len(prompt)
        assert row["prefill"][-1] == len(prompt)
    samples = rows("native-pressure.jsonl")
    assert len(samples) == native["pressure_count"] > 0 and all(row["pressure"] == 1 for row in samples)
    assert max(row["footprint_bytes"] for row in samples) == native["max_footprint_bytes"]
    assert len(native["drained"]) == 5 and not any(native["drained"].values())
    observations = []
    for mode, expected_revision in REFERENCE_REVISIONS.items():
        ready, *cases, complete = rows(f"{mode}.jsonl")
        assert ready == {
            "stage": "ready", "runner": mode, "mlx": "0.32.4", "ref_revision": expected_revision
        }
        assert complete["stage"] == "complete" and complete["pressure"] == 1
        expected = [(512, 1024), (2048, 1024), (8192, 1024)]
        if mode == "omlx":
            expected.append((8192, 2048))
            repeat_ready, *repeated, end = rows("omlx-repeat.jsonl")
            assert repeat_ready == ready and repeated == cases
            assert end["stage"] == "complete" and end["pressure"] == 1
        assert [(case["length"], case["chunk"]) for case in cases] == expected
        for case in cases:
            ax = native_rows[case["id"]]
            assert case["stage"] == "case" and case["length"] == ax["input_tokens"]
            against_m2 = shared_prefix(case["ids"], ax["m2_ids"])
            assert against_m2 == case["shared_prefix"]
            diff = case["first_split_from_ax_m2"]
            if diff is not None:
                assert diff["index"] == against_m2 and diff["ax_m2_id"] == ax["m2_ids"][against_m2]
                assert diff["peer_id"] == case["ids"][against_m2]
                assert diff["peer_minus_ax_m2"] >= 0
                assert diff["ax_m2_rank"] >= diff["all_better_than_ax_m2"] + 1
            else:
                assert against_m2 == 16
            observations.append({
                "peer": mode, "input_tokens": case["length"], "peer_prefill_chunk": case["chunk"],
                "shared_with_ax_m2": against_m2,
                "shared_with_ax_m5": shared_prefix(case["ids"], ax["m5_ids"]),
                "first_split_from_ax_m2": diff,
            })
    return {
        "native_m2_m5_shared_prefix": {
            row["input_tokens"]: shared_prefix(row["m2_ids"], row["m5_ids"])
            for row in native_rows.values()
        },
        "normal_native_pressure_samples": len(samples),
        "observations": observations,
        "qualification": False,
    }


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2, sort_keys=True, allow_nan=False))
