"""Complete the signed fixed-geometry crossover with its separately sealed fourth arm."""

from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent
FIRST = RESULTS / "flash-next-current-buffered-reader-first-m5-20261123"
REFERENCE = RESULTS / "flash-next-reference-heartbeat-pressure-m5-20261123"
ADDED = "mapped-second"
NEW_MEMBERS = {
    "capture.json", f"arms/{ADDED}/receipt.json",
    f"arms/{ADDED}/{ADDED}.log", f"arms/{ADDED}/{ADDED}-samples.jsonl",
}


def stage(archive, destination, expected):
    members = archive.getmembers()
    assert len(members) == len(expected) and {item.name for item in members} == expected
    for member in members:
        assert member.isfile()
        path = destination / member.name
        path.parent.mkdir(parents=True, exist_ok=True)
        with archive.extractfile(member) as source:
            path.write_bytes(source.read())


def replay():
    prior = run_path(str(FIRST / "replay.py"))
    ref = run_path(str(REFERENCE / "replay.py"))["replay"]()
    assert json.loads(json.dumps(ref)) == json.loads((REFERENCE / "comparison.json").read_bytes())
    assert ref["ordinary_reference_started"] is False and ref["stop_reason"] == "critical system memory pressure"
    names = {*prior["INPUT_FILES"], "capture.json", "inputs.sha256"}
    for arm in prior["OBSERVED"]:
        names.update(f"arms/{arm}/{name}" for name in ("receipt.json", f"{arm}.log", f"{arm}-samples.jsonl"))
    with (TemporaryDirectory(prefix="flash-next-completed-reader-crossover-") as temp,
          tarfile.open(FIRST / "evidence.tar.xz") as initial,
          tarfile.open(ROOT / "evidence.tar.xz") as resumed):
        target = Path(temp)
        stage(initial, target, names)
        original = prior["compare"](target, RESULTS)
        original["supported_m4"] = prior["verify_m4"](FIRST, original)
        observed = (FIRST / "comparison.json").read_bytes()
        assert json.loads(json.dumps(original)) == json.loads(observed)
        assert original["missing_arm"] == ADDED
        checksums = {name: sha256((target / name).read_bytes()).hexdigest()
                     for name in names if name.startswith("arms/")}
        assert len(checksums) == 9
        reference_inputs = prior["read"](target / "capture.json")["prior_reference_admission"]
        stage(resumed, target, NEW_MEMBERS)
        assert all(sha256((target / name).read_bytes()).hexdigest() == digest
                   for name, digest in checksums.items())
        owner = prior["read"](target / "capture.json")
        build = prior["read"](target / "build.json")
        hardware = prior["read"](RESULTS / prior["SAME_HOST"] / "lifecycle.json")["hardware"]
        expected_execution = {"initial_status": 1, "resumed_after_independent_reference": True,
                              "reference_status": 1, "resume_status": 0, "preserved_raw_files": 9}
        assert owner["execution"] == expected_execution and owner["run_status"] == 0
        assert owner["codesign_verified"] and owner["server_sha256"] == build["server_sha256"]
        assert owner["soc"] == hardware["soc"] and owner["form_factor"] == hardware["form_factor"]
        assert owner["memory_gib"] == 128 and owner["os"] == hardware["os"]
        assert owner["final_pressure"] == 1 and tuple(owner["order"]) == prior["ARMS"]
        assert owner["prior_reference_admission"] == {**reference_inputs, ADDED: {"previous_reference_terminal": True}}
        inputs = dict(line.split()[::-1] for line in (target / "inputs.sha256").read_text().splitlines())
        expected = prior["older_outputs"](RESULTS)
        sys.path.insert(0, temp)
        try:
            tools = run_path(str(target / "run_cache_policy.py"))
            rows, summaries = {}, {}
            for arm in prior["ARMS"]:
                folder = target / "arms" / arm
                rows[arm], summaries[arm] = prior["replay_one"](
                    folder, prior["read"](folder / "receipt.json"), arm, tools, build, inputs, hardware, expected,
                )
            facts = tools["compare"](rows, prior["read"](target / "model-manifest.json")["vocab_size"], "default")
        finally:
            sys.path.pop(0)
        return {**facts, "source_base": build["source_base"], "source_patch_sha256": build["patch_sha256"],
                "candidate_server_sha256": build["server_sha256"], "on_host_measurements": summaries,
                "execution": expected_execution, "release_ready": False}


if __name__ == "__main__":
    print(json.dumps(replay(), sort_keys=True, indent=2, allow_nan=False))
