"""Regrade the fresh production native QA rows and bind them to delivery."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
DELIVERY = ROOT.parent / "flash-next-mxfp4-current-delivery-m5-20261122"

sys.path.insert(0, str(ROOT.parents[2] / "scripts"))
import check_flash_next_release_gates as gate  # noqa: E402
import run_flash_next_qa_cohort as qa  # noqa: E402

MODES = ("disabled", "default")


def payload(archive, name):
    item = archive.extractfile(name)
    if item is None:
        raise ValueError(f"Missing QA evidence {name}")
    return item.read()


def replay(path=ROOT / "evidence.tar.xz"):
    expected = json.loads((DELIVERY / "comparison.json").read_bytes())
    with tarfile.open(DELIVERY / "evidence.tar.xz") as supplied:
        interfaces = json.loads(payload(supplied, "build.json"))
        delivered_raw = payload(supplied, "native-support.json")
        manifest = {name: value for value, name in
                    (row.split() for row in payload(supplied, "inputs.sha256").decode().splitlines())}
    assert sha256(delivered_raw).hexdigest() == expected["native_support_sha256"]
    delivered = json.loads(delivered_raw)
    for modifier in (Path(qa.__file__), qa.FROZEN / "full-qa-items.json",
                     *(qa.FROZEN / key for key in ("checkers.py", "closed_checks.py", "qa_checks.py", "prompt_def.py"))):
        path_token = modifier.relative_to(qa.ROOT).as_posix()
        assert sha256(modifier.read_bytes()).hexdigest() == manifest[path_token]
    items = qa.load_items(qa.FROZEN / "full-qa-items.json")

    with tarfile.open(path) as archive:
        raw = payload(archive, "result.json")
        result = json.loads(raw)
        contract = json.loads(payload(archive, "contract.json"))
        assert all(result[name] == value for name, value in contract.items())
        assert result["items_sha256"] == manifest[str((qa.FROZEN / "full-qa-items.json").relative_to(qa.ROOT))]
        assert result["server_sha256"] == delivered["server_sha256"] == interfaces["binaries_sha256"]["ax-engine-server"]
        for key in ("inventory_sha256", "revision", "hardware"):
            assert result[key] == delivered[key]
        assert result["items"] == len(items) == 105 and result["modes"] == list(MODES)
        assert result["ordinary_launch"] is True and result["release_ready"] is False
        assert result["total_blocks"] * result["block_size_tokens"] == qa.STANDARD_CONTEXT_TOKENS
        assert len(archive.getmembers()) == 2 + len(MODES) * len(items)
        runs = {}
        grades = {}
        for mode in MODES:
            values = []
            for i, item in enumerate(items):
                row = json.loads(payload(archive, f'{mode}/{i:03d}-{item["id"]}.json'))
                assert row["index"] == i and row["id"] == item["id"]
                assert row["budget"] == qa.budget_for(item) and row["finish_reason"] == "stop"
                values.append(row)
            original = result["summary"]["modes"][mode]
            runs[mode] = {"mode": mode, "rows": values, "ready_seconds": original["ready_seconds"],
                          "peak_rss_kib": original["peak_rss_kib"],
                          "advertised_model": original["advertised_model"]}
            grades[mode] = qa.grade(items, {row["id"]: row["text"] for row in values})
    assert grades == result["grades"]
    summary = qa.summarize(items, runs, grades)
    assert summary == result["summary"]
    assert summary["text_identity_vs_disabled"] == {
        "default": {"matching_text_pairs": 105, "differing_text_ids": []}}
    assert all(mode["verified_steps"] == mode["route_errors"] == 0 for mode in summary["modes"].values())
    thresholds_raw = gate.THRESHOLDS.read_bytes()
    assert sha256(thresholds_raw).hexdigest() == expected["thresholds_sha256"]
    thresholds = json.loads(thresholds_raw)
    floor = thresholds["timing_host"]["min_cpu_idle_percent"]
    idle = [result["host_quiet"]["cpu_idle_percent"], result["host_idle_after"]]
    assert all(floor <= value <= 100 for value in idle)
    return {"scope": "Native frozen-cohort QA and replayable memory and lookup gates",
            "release_ready": False, "matching_native_pairs": 105, "native_requests": summary["requests"],
            "timing_cpu_idle_percent": idle, "server_sha256": result["server_sha256"],
            "receipt_sha256": sha256(raw).hexdigest(), "modes": summary["modes"],
            "memory": gate.gate_memory(result, thresholds)[0],
            "long_lookup": gate.gate_long_lookup(result, thresholds)[0]}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
