"""Replay the Flash Next worker startup warmup functional comparison."""

from gzip import decompress
from hashlib import file_digest, sha256
import json
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
DEPENDENCIES = ROOT.parent / "flash-next-ple-serving-m2-20261122" / "dependencies.tar.xz"
ARM_LABELS = ("default-1", "default-2")
HARNESS = {
    "check_flash_next_native_support.py",
    "qualify_qwen38_flash_next.py",
    "run_flash_next_generate_matrix.py",
}


def need(valid, message):
    if not valid:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def unpack(path, directory, expected):
    with tarfile.open(path) as tar:
        names = tar.getnames()
        need(set(names) == expected and len(names) == len(expected), "An archive member changed")
        for member in tar:
            need(member.isfile() and Path(member.name).name == member.name,
                 "Unexpected archive member path")
            (directory / member.name).write_bytes(tar.extractfile(member).read())


def replay(root, dependencies):
    build = read(root / "build.json")
    hashes = {name: checksum for checksum, name in
              (line.split() for line in (root / "inputs-m2.sha256").read_text().splitlines())}
    need(build["profile"] == "release-server" and build["runtime_commit"].startswith("decb30fd5")
         and hashes["build.json"] == digest(root / "build.json")
         and hashes["ax-engine-server-startup"] == build["server_sha256"]
         and hashes["run_warmup.py"] == build["driver_sha256"] == digest(root / "run_warmup.py"),
         "The measured build or driver changed")
    m4 = read(root / "m4.json")
    m4_raw = decompress((root / "m4-startup-tests.txt.gz").read_bytes())
    m4_output = m4_raw.decode()
    need(m4["qualification"] is False and m4["runtime_commit"] == build["runtime_commit"]
         and m4["libmlx_sha256"] == build["libmlx_sha256"]
         and m4["test_output_sha256"] == sha256(m4_raw).hexdigest()
         and m4["passed"] == 4 and m4["failed"] == 0
         and "Apple M4 Pro\n68719476736\n27.0\n" in m4_output
         and "test result: ok. 4 passed; 0 failed; 0 ignored;" in m4_output,
         "The supported-tier startup software record changed")
    with TemporaryDirectory() as folder:
        scratch = Path(folder)
        unpack(dependencies, scratch, HARNESS)
        need(all(digest(scratch / name) == hashes[name] for name in HARNESS),
             "The benchmark helper source changed")
        sys.path.insert(0, str(scratch))
        import run_warmup as runner
        expected_files = {"result.json", "status.txt",
                          *(f"{name}.json" for name in ARM_LABELS),
                          *(f"server-{name}.log" for name in ARM_LABELS)}
        unpack(root / "evidence.tar.xz", scratch, expected_files)
        result = read(scratch / "result.json")
        need((scratch / "status.txt").read_text() == "0\n" and result["scope"] == "M2 functional warmup"
             and result["qualification"] is False and result["build"] == build
             and result["publisher_revision"] == runner.REVISION
             and result["inventory_sha256"] == runner.INVENTORY == hashes["inventory.json"]
             and result["manifest"] == runner.MANIFEST,
             "The run did not finish against the pinned ready-state contract")
        arms = result["arms"]
        need([(a["name"], a["mode"], a["single"]) for a in arms]
             == list(zip(ARM_LABELS, ("default",) * 2, (False, True), strict=True)),
             "The warmup treatment or run order changed")
        stable, evidence = [], {}
        for row in arms:
            arm = row["name"]
            raw = scratch / f"{arm}.json"
            log = scratch / f"server-{arm}.log"
            need(digest(raw) == row["raw_sha256"] and digest(log) == row["log_sha256"],
                 "The measured stream or warmup log changed")
            need(row["host_idle_percent"] == [None, None] and row["timing_eligible"] is False
                 and row["normal_pressure_samples"] >= 2, "Unexpected functional host conditions")
            observed = read(raw)
            identity, scores = runner.report_rows(observed, "default")
            phases = runner.warmup_phases(log.read_text(), row["single"])
            need(all(row[key] == value for key, value in (phases | scores).items())
                 and row["ready_seconds"] == observed["ready_seconds"],
                 "Startup or streaming data differs from the recorded values")
            stable.append(identity)
            evidence[arm] = {
                "matched_responses": len(identity["outputs"]),
                "generated_ids": sum(len(x[1]) for x in identity["outputs"]),
                "normal_pressure_samples": row["normal_pressure_samples"],
            } | phases
        need(stable[0] == stable[1], "Warmup changed input, generated tokens or stop state")
        return {"scope": result["scope"], "qualification": False, "arms": evidence,
                "m4_startup_software_tests": m4["passed"]}


if __name__ == "__main__":
    print(json.dumps(replay(ROOT, DEPENDENCIES), indent=2, allow_nan=False))
