"""Replay the signed candidate's two-request controls against the sealed oracle."""

from hashlib import sha256
import json
from pathlib import Path
import re
from runpy import run_path
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "flash-next-natural-document-prefix-http-m2-20261123"
PRIOR_SHA256 = "db1563861203784640453460a450c75e4dc43ad8f5e52b93bbcdadf64a872b58"
ARMS = {"ordinary": "Mapped", "explicit": "AlignedBuffered"}
OBSERVED = ("receipt.json", "partial.json", "natural.log", "natural-samples.jsonl")
CANDIDATE = ("build.json", "source.patch", "inputs.sha256", "policy_driver.py",
             "model-manifest.json", "status.txt", "owner.log", "hardware.json")


def read(data):
    return json.loads(data)


def digest(data):
    return sha256(data).hexdigest()


def replay():
    prior = PRIOR / "evidence.tar.xz"
    assert digest(prior.read_bytes()) == PRIOR_SHA256
    shared = run_path(str(PRIOR / "replay.py"))
    member = shared["read_member"]
    with (TemporaryDirectory(prefix="flash-next-reader-http-replay-") as temp,
          tarfile.open(prior) as baseline, tarfile.open(shared["CORPUS"]) as corpus,
          tarfile.open(ROOT / "evidence.tar.xz") as sample):
        expected = set(CANDIDATE) | {f"{mode}/{name}" for mode in ARMS for name in OBSERVED}
        assert set(sample.getnames()) == expected and len(sample.getnames()) == len(expected)
        run = {name: member(sample, name) for name in CANDIDATE}
        old_outputs = [
            read(member(baseline, f"{mode}/observed/receipt.json"))["outputs"] for mode in ("cold", "learned")
        ]
        assert old_outputs[0] == old_outputs[1]
        files = {name.removeprefix("inputs/"): member(baseline, name)
                 for name in baseline.getnames() if name.startswith("inputs/")}
        files["run_natural_fork.py"] = member(baseline, "learned/run_natural_fork.py")
        files["mixed-corpus.jsonl"] = member(corpus, "mixed-corpus.jsonl")
        input_rows = [line.split() for line in run["inputs.sha256"].decode().splitlines()]
        sealed = {name: checksum for checksum, name in input_rows}
        bound = {**files, **{name: run[name] for name in
                             ("build.json", "source.patch", "policy_driver.py", "model-manifest.json")}}
        assert len(input_rows) == len(sealed) and set(sealed) == bound.keys() | {"ax-engine-server"}
        assert all(digest(data) == sealed[name] for name, data in bound.items())
        build = read(run["build.json"])
        assert build["server_sha256"] == sealed["ax-engine-server"] and build["codesign_verified"] is True
        assert build["patch_sha256"] == sealed["source.patch"] and build["profile"] == "release-server"
        assert build["rustc"].startswith("rustc 1.99.1 ") and build["mlx"] == build["mlx_metal"] == "0.32.4"
        assert run["status.txt"].strip() == b"0"
        hardware = read(run["hardware.json"])
        assert hardware == {"form_factor": "Mac Studio", "soc": "Apple M2 Ultra",
                            "memory_bytes": 206158430208, "os": "27.2.1", "storage": "internal startup SSD",
                            "mlx": "0.32.4", "mlx_metal": "0.32.4"}
        for name, data in files.items():
            (Path(temp) / name).write_bytes(data)
        sys.path.insert(0, temp)
        try:
            from run_fixed_extension import CASES, MANIFEST, check_case
            from run_natural_fork import LEARNED_HITS, ORDER
            report = {}
            for mode, reader in ARMS.items():
                raw = {name: member(sample, f"{mode}/{name}") for name in OBSERVED}
                assert not any(marker in data for data in raw.values()
                               for marker in (b"/Users/", b"/private/tmp/", b"127.0.0.1"))
                receipt = read(raw["receipt.json"])
                assert receipt["build"] == build and receipt["manifest"] == MANIFEST
                assert receipt["qualification"] is False and receipt["learning_enabled"] is True
                assert receipt["explicit_seed_requests"] == 0 and tuple(receipt["order"]) == ORDER
                assert receipt["cases"] == read(raw["partial.json"])
                outputs = {name: check_case(receipt["cases"][name], name in LEARNED_HITS, stop, tail)
                           for name, _, _, stop, _, tail in CASES}
                assert outputs == receipt["outputs"] == old_outputs[0]
                samples = [read(row) for row in raw["natural-samples.jsonl"].splitlines()]
                assert len(samples) == receipt["normal_pressure_samples"] > 0
                assert all(row["pressure"] == 1 for row in samples)
                log = re.sub(r"\x1b\[[0-9;]*m", "", raw["natural.log"].decode())
                assert re.findall(r"Flash Next resident reader[^\n]* reader=(\w+)", log) == [reader]
                count = sum(len(ids) for row in outputs.values() for ids in row.values())
                assert count == 1921
                report[mode] = {"reader": reader, "exact_output_token_ids": count,
                                "normal_pressure_samples": len(samples),
                                "physical_restore_cases": sorted(LEARNED_HITS)}
        finally:
            sys.path.pop(0)
    return {"scope": "Two-request HTTP function on Mac Studio M2 Ultra 192 GB",
            "server_sha256": build["server_sha256"], "patch_sha256": build["patch_sha256"],
            "release_qualification": False, "arms": report}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
