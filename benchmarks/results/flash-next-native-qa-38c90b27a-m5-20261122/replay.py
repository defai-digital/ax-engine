"""Regrade the current M5 Flash Next native QA cohort against the signed outputs."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent
SCRIPTS = ROOT.parents[2] / "scripts"
DELIVERY = RESULTS / "flash-next-fresh-delivery-38c90b27a-m5-20261122" / "evidence.tar.xz"
SIGNED = RESULTS / "flash-next-native-resident-refresh-m5-20261122" / "evidence.tar.xz"
sys.path.insert(0, str(SCRIPTS))
import check_flash_next_release_gates as gates
import run_flash_next_qa_cohort as qa


def require(condition, message):
    if not condition:
        raise ValueError(message)


def member(archive, name):
    row = archive.getmember(name)
    require(row.isfile(), f"QA evidence member is not a regular file: {name}")
    with archive.extractfile(row) as stream:
        return stream.read()


def fingerprint(row):
    usage = tuple(row["usage"][key] for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
    data = [row["index"], row["id"], row["text"], row["finish_reason"], usage]
    return sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def replay(artifact=ROOT / "evidence.tar.xz"):
    with tarfile.open(DELIVERY) as published, tarfile.open(SIGNED) as signed:
        manifest = {name: checksum for checksum, name in (line.split() for line in
                    member(published, "inputs.sha256").decode().splitlines())}
        support_raw = member(published, "native-support.json")
        support = json.loads(support_raw)
        delivered = json.loads(member(published, "delivery.json"))
        require(not gates.delivery_control_problems(delivered, support_raw), "Delivery and support evidence disagree")
        oracle_raw = member(signed, "driver/signed-native.json")
        require(sha256(oracle_raw).hexdigest() == manifest["signed-native.json"], "The frozen signed QA oracle changed")
        oracle = json.loads(oracle_raw)["qa"]
    files = (Path(qa.__file__), *(qa.FROZEN / name for name in
             ("full-qa-items.json", "checkers.py", "closed_checks.py", "prompt_def.py")))
    for path in files:
        require(sha256(path.read_bytes()).hexdigest() == manifest[path.relative_to(qa.ROOT).as_posix()],
                f"A sealed grader input changed: {path.name}")
    items = qa.load_items(gates.QA_ITEMS)
    modes = gates.AX_ARMS
    runs, responses = {}, {}
    with tarfile.open(artifact) as archive:
        raw = member(archive, "result.json")
        result, contract = json.loads(raw), json.loads(member(archive, "contract.json"))
        require(all(result[name] == value for name, value in contract.items())
                and all(result[name] == support[name] for name in ("hardware", "inventory_sha256", "server_sha256", "revision"))
                and result["server_sha256"] == manifest["ax-engine-server"], "QA build or same-pack delivery identity changed")
        require(result["qualification"] is False and result["release_ready"] is False
                and result["modes"] == list(modes) and result["ordinary_launch"] is True
                and result["total_blocks"] * result["block_size_tokens"] == qa.STANDARD_CONTEXT_TOKENS,
                "QA did not exercise the native product envelope")
        expected = {"result.json", "contract.json"}
        for mode in modes:
            names = [f'{mode}/{index:03d}-{item["id"]}.json' for index, item in enumerate(items)]
            expected.update(names)
            values = [json.loads(member(archive, name)) for name in names]
            want = oracle[mode]
            require({row["id"]: fingerprint(row) for row in values} == want,
                    f"The {mode} responses changed from signed text, usage or stop state")
            metadata = result["summary"]["modes"][mode]
            advertised = {"id": qa.MODEL_ID if mode == "disabled" else qa.ORDINARY_MODEL_ID,
                          "context_length": qa.STANDARD_CONTEXT_TOKENS}
            require(metadata["advertised_model"] == advertised, f"The {mode} advertised profile changed")
            runs[mode] = {"rows": values, **{field: metadata[field] for field in
                          ("peak_rss_kib", "ready_seconds", "advertised_model")}}
            responses[mode] = values
        names = archive.getnames()
        require(len(names) == len(expected) and set(names) == expected, "The native QA cohort is incomplete or contains extras")
    require(not gates.native_qa_problems(result, items, responses), "Frozen native answer graders disagreed")
    summary = qa.summarize(items, runs, result["grades"])
    require(summary == result["summary"] and summary["requests"] == 210
            and all(row["verified_steps"] == row["route_errors"] == 0 for row in summary["modes"].values()),
            "The native QA summary or policy state changed")
    th = json.loads(gates.THRESHOLDS.read_bytes())
    idle = [result["host_quiet"]["cpu_idle_percent"], result["host_idle_after"]]
    return {"scope": "Current-build native frozen-cohort QA", "release_ready": False,
            "server_sha256": result["server_sha256"], "qa_receipt_sha256": sha256(raw).hexdigest(),
            "signed_oracle_sha256": sha256(oracle_raw).hexdigest(), "matching_signed_responses": summary["requests"],
            "modes": summary["modes"], "long_lookup": gates.gate_long_lookup(result, th)[0],
            "memory": gates.gate_memory(result, th)[0], "timing_cpu_idle_percent": idle,
            "timing_eligible": all(th["timing_host"]["min_cpu_idle_percent"] <= value <= 100 for value in idle)}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
