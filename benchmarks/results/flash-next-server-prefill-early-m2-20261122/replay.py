"""Replay the Flash Next early-arrival HTTP functional check."""

from contextlib import contextmanager
from hashlib import file_digest
import json
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
LATE_M2 = ROOT.parent / "flash-next-server-prefill-tail-m2-20261122"
DRIVERS = ROOT.parent / "flash-next-late-prefill-deferral-m5-20261122" / "evidence.tar.xz"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def unpack(archive_path, destination, members):
    with tarfile.open(archive_path) as archive:
        for original, renamed in members.items():
            member = archive.getmember(original)
            path = destination / renamed
            require(member.isfile() and not path.exists(), "Unexpected archive member")
            with archive.extractfile(member) as source:
                path.write_bytes(source.read())


@contextmanager
def inputs(root):
    hashes = dict(
        (name, checksum)
        for checksum, name in (line.split() for line in (root / "stage.sha256").read_text().splitlines())
    )
    with TemporaryDirectory() as directory:
        files = Path(directory)
        prior = LATE_M2 / "evidence.tar.xz"
        unpack(prior, files, {"solo.json": "oracle.json", "receipt.json": "oracle-receipt.json"})
        for name in ("build.json", "run_server_tail.py"):
            (files / name).write_bytes((LATE_M2 / name).read_bytes())
        for name in ("run_early.py", "run_m2.sh"):
            (files / name).write_bytes((root / name).read_bytes())
        build = read(files / "build.json")
        require(build["runtime_commit"].startswith("2b905c012")
                and build["server_sha256"] == hashes.pop("ax-engine-server-tail")
                and build["driver_sha256"] == hashes["run_server_tail.py"],
                "The archived run does not bind the corrected native binary")
        unpack(DRIVERS, files, {name: name for name in hashes if not (files / name).is_file()})
        require(all(digest(files / name) == value for name, value in hashes.items()),
                "A sealed prompt, driver, or oracle differs from the live inputs")
        sys.path.insert(0, str(files))
        try:
            import run_early as driver
            yield driver, files
        finally:
            sys.path.remove(str(files))


def replay(root):
    with inputs(root) as (driver, files):
        cases, build, inventory, oracle_receipt, oracle, _ = driver.inputs()
        trial_files = {str(at): f"trigger-{at}.json" for at in driver.EARLY_TRIGGERS}
        samples_file = f"{driver.SERVER}-samples.jsonl"
        names = ("receipt.json", "status.txt", samples_file, *trial_files.values())
        unpack(root / "evidence.tar.xz", files, {name: name for name in names})
        receipt = read(files / "receipt.json")
        require((files / "status.txt").read_text() == "0\n"
                and receipt["scope"] == "M2 Ultra 192 GB native HTTP functional test"
                and receipt["qualification"] is False
                and receipt["build"] == build
                and receipt["manifest"] == oracle_receipt["manifest"]
                and receipt["pack_revision"] == inventory["revision"]
                and receipt["inventory_sha256"] == oracle_receipt["inventory_sha256"]
                and receipt["oracle_sha256"] == oracle_receipt["solo"]["data_sha256"]
                and set(receipt["trials"]) == set(trial_files),
                "The run did not complete against the sealed binary, pack and oracle")
        samples = files / samples_file
        pressure = [json.loads(line)["pressure"] for line in samples.read_text().splitlines()]
        require(digest(samples) == receipt["pressure_sha256"]
                and pressure and set(pressure) == {1} and len(pressure) == receipt["pressure_samples"],
                "M2 memory pressure or sample provenance changed")
        trials = {}
        for trigger, name in trial_files.items():
            path = files / name
            require(digest(path) == receipt["trials"][trigger], "The native trial changed")
            raw = read(path)
            verified = driver.verify(raw, int(trigger), cases, oracle)
            require(all(raw[key] == value for key, value in verified.items())
                    and raw["drained"] == dict.fromkeys(driver.mixed.COUNTERS, 0),
                    "The replay disagrees with the recorded schedule or drain")
            trials[trigger] = verified | {
                "held_decode_steps": len(raw["traces"][0]["held"]),
                "matching_generated_ids": sum(len(row["output_token_ids"]) for row in raw["observations"]),
            }
        return {
            "scope": receipt["scope"],
            "qualification": False,
            "runtime_commit": build["runtime_commit"],
            "pack_revision": inventory["revision"],
            "normal_pressure_samples": len(pressure),
            "trials": trials,
        }


if __name__ == "__main__":
    print(json.dumps(replay(ROOT), indent=2, allow_nan=False))
