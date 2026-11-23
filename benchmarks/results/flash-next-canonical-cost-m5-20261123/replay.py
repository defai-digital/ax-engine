"""Replay the signed Flash Next canonical cost comparison."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
FRESH = ROOT.parent / "flash-next-probe-followup-fresh-m5-20261123" / "evidence.tar.xz"
CORPORA = {
    "agent-coding": "benchmarks/prompts/mtp-suites/python_modules_long.jsonl",
    "general-long": "benchmarks/prompts/mtp-suites/qwen38_general_long.jsonl",
    "short-answer": ("benchmarks/results/qualification/2026-11-18-qwen38-27b-mxfp4-mtp-performance/"
                     "short-answer.jsonl"),
}
SOURCE_SCRIPTS = ("check_flash_next_native_support.py", "run_flash_next_generate_matrix.py",
                  "qualify_qwen38_flash_next.py")
DRIVER = {"run_cost.py", "compare_cost.py", "verify_routes.py", "run_cost_m5.sh", "contract.json",
          "contract-reverse.json"}
RAW = {"result.json", "build.json", "inputs.sha256", "status.txt", "rows-disabled.jsonl",
       "rows-required.jsonl", "server-disabled.log", "server-required.log"}


def read(archive, name):
    item = archive.getmember(name)
    if not item.isfile():
        raise ValueError(f"Expected a regular archive member: {name}")
    with archive.extractfile(item) as stream:
        return stream.read()


def replay(path):
    with (tarfile.open(path) as archived, tarfile.open(FRESH) as fresh,
          TemporaryDirectory(prefix="flash-next-cost-replay-") as directory):
        raw_build = read(archived, "raw/build.json")
        build = json.loads(raw_build)
        roster = {f"driver/{name}" for name in DRIVER} | {f"raw/{name}" for name in RAW}
        if (set(archived.getnames()) != roster or len(archived.getnames()) != len(roster)
                or read(archived, "raw/status.txt") != b"0\n"
                or raw_build != read(fresh, "raw/build.json") or build["codesign_verified"] is not True):
            raise ValueError("The paired run is incomplete or uses a different signed build")
        root = Path(directory)
        driver, raw = root / "driver", root / "raw"
        driver.mkdir()
        raw.mkdir()
        driver_files = {name: read(archived, f"driver/{name}") for name in DRIVER}
        driver_files.update({name: (REPO / "scripts" / name).read_bytes() for name in SOURCE_SCRIPTS})
        driver_files.update({f"panels/{name}.jsonl": (REPO / file).read_bytes() for name, file in CORPORA.items()})
        driver_files.update({"inventory.json": read(fresh, "driver/inventory.json"), "build.json": raw_build})
        manifest = {name: digest for digest, name in map(str.split, read(archived, "raw/inputs.sha256").decode().splitlines())}
        if (set(manifest) != (driver_files.keys() - {"compare_cost.py"}) | {"ax-engine-server"}
                or manifest["ax-engine-server"] != build["server_sha256"]
                or any(sha256(data).hexdigest() != manifest[name] for name, data in driver_files.items()
                       if name != "compare_cost.py")):
            raise ValueError("The staged runtime, inventory, driver or corpus changed")
        (driver / "panels").mkdir()
        for name, body in driver_files.items():
            (driver / name).write_bytes(body)
        for name in RAW:
            (raw / name).write_bytes(read(archived, f"raw/{name}"))
        run = json.loads((raw / "result.json").read_bytes())
        if run["inventory_sha256"] != manifest["inventory.json"]:
            raise ValueError("The recorded run changed its pack inventory")
        return subprocess.check_output([sys.executable, "-B", driver / "compare_cost.py", raw], cwd=root)


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    sys.stdout.buffer.write(replay(cli.parse_args().archive))
