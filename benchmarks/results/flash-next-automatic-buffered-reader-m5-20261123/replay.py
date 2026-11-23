"""Verify the guarded Flash Next reader crossover with the signed AX oracle."""

from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent
PRIOR = RESULTS / "flash-next-current-buffered-reader-first-m5-20261123"
REFERENCE = RESULTS / "flash-next-reference-heartbeat-pressure-m5-20261123"
CANDIDATE_COMMIT = "d2220c26f3cfccf2583c583c6b264e22b9459f88"


def stage(archive, destination, expected):
    members = archive.getmembers()
    assert set(member.name for member in members) == expected and len(members) == len(expected)
    for member in members:
        assert member.isfile()
        path = destination / member.name
        path.parent.mkdir(parents=True, exist_ok=True)
        with archive.extractfile(member) as source:
            path.write_bytes(source.read())


def replay():
    earlier = run_path(str(PRIOR / "replay.py"))
    reference = run_path(str(REFERENCE / "replay.py"))["replay"]()
    assert reference["ordinary_reference_started"] is False
    assert reference["stop_reason"] == "critical system memory pressure"
    arms, source_names = earlier["ARMS"], earlier["INPUT_FILES"]
    names = set(source_names) | {"capture.json", "inputs.sha256", "source.json"}
    for arm in arms:
        names.update(f"arms/{arm}/{name}" for name in ("receipt.json", f"{arm}.log", f"{arm}-samples.jsonl"))
    with (TemporaryDirectory(prefix="flash-next-automatic-reader-crossover-") as temp,
          tarfile.open(ROOT / "evidence.tar.xz") as archive):
        target = Path(temp)
        stage(archive, target, names)
        read = earlier["read"]
        source = read(target / "source.json")
        build = read(target / "build.json")
        owner = read(target / "capture.json")
        hardware = read(RESULTS / earlier["SAME_HOST"] / "lifecycle.json")["hardware"]
        assert source["candidate_commit"] == CANDIDATE_COMMIT
        assert source["source_base"] == build["source_base"]
        assert source["source_patch_sha256"] == build["patch_sha256"] == earlier["digest"](target / "source.patch")
        assert source["server_sha256"] == owner["server_sha256"] == build["server_sha256"]
        assert source["independent_reference_complete"] is False
        assert source["independent_reference_receipt_sha256"] == earlier["digest"](REFERENCE / "comparison.json")
        assert source["arm_environment"] == {"mapped": {"AX_MLX_FLASH_NEXT_ALIGNED_BUFFERED": "0"},
                                             "buffered": {}}
        inputs_list = [line.split() for line in (target / "inputs.sha256").read_text().splitlines()]
        inputs = {name: digest for digest, name in inputs_list}
        assert len(inputs_list) == len(inputs) and set(inputs) == source_names | earlier["INPUT_BINARIES"]
        assert all(earlier["digest"](target / name) == inputs[name] for name in source_names)
        assert inputs["ax-engine-server"] == build["server_sha256"] and build["codesign_verified"] is True
        assert inputs["mxfp4-inventory.json"] == build["inventory_sha256"]
        assert inputs["model-manifest.json"] == earlier["DERIVED_MANIFEST"]
        inventory, manifest = (read(target / name) for name in ("mxfp4-inventory.json", "model-manifest.json"))
        assert inventory["revision"] == earlier["PUBLISHER_PIN"] and len(inventory["members"]) == 49
        assert build["profile"] == "release-server" and build["rustc"].startswith("rustc 1.99.1 ")
        assert build["cargo"].startswith("cargo 1.99.1 ") and build["mlx"] == build["mlx_metal"] == "0.32.4"
        assert owner["execution"] == {"initial_status": 0, "resumed_after_independent_reference": False}
        assert tuple(owner["order"]) == arms and owner["run_status"] == 0 and owner["final_pressure"] == 1
        assert owner["codesign_verified"] is True
        assert owner["form_factor"] == hardware["form_factor"] and owner["soc"] == hardware["soc"]
        assert owner["memory_gib"] == 128 and owner["os"] == hardware["os"]
        assert owner["prior_reference_admission"] == {
            arm: {"previous_reference_terminal": True} for arm in arms
        }
        assert sha256((target / "run_cache_policy.py").read_bytes()).hexdigest() == inputs["run_cache_policy.py"]
        expected = earlier["older_outputs"](RESULTS)
        sys.path.insert(0, temp)
        try:
            tools = run_path(str(target / "run_cache_policy.py"))
            assert tuple(label for label, _, _ in tools["ARMS"]) == arms
            assert all(length == 512 for _, _, length in tools["ARMS"])
            rows, on_host = {}, {}
            for arm in arms:
                folder = target / "arms" / arm
                rows[arm], on_host[arm] = earlier["replay_one"](
                    folder, read(folder / "receipt.json"), arm, tools, build, inputs, hardware, expected,
                )
            result = tools["compare"](rows, manifest["vocab_size"], "default")
        finally:
            sys.path.pop(0)
        return {**result, "candidate_commit": source["candidate_commit"],
                "candidate_server_sha256": build["server_sha256"], "source_base": build["source_base"],
                "source_patch_sha256": build["patch_sha256"], "on_host_measurements": on_host,
                "execution": owner["execution"], "release_ready": False}


if __name__ == "__main__":
    print(json.dumps(replay(), sort_keys=True, indent=2, allow_nan=False))
