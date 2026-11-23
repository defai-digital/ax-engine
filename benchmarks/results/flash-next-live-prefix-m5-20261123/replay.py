"""Replay the guarded two-slot exact-prefix crossover from the original observations."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
M2 = ROOT.parent / "flash-next-openai-stream-disconnect-m2-20261122" / "evidence.tar.xz"
CORPUS_ARCHIVE = ROOT.parent / "flash-next-mxfp4-mixed-serving-m5-20261121" / "evidence.tar.xz"
CORPUS = "mixed-corpus.jsonl"
BIN = "ax-engine-server-resident-mixed"
DRIVERS = (
    "bench_ax_serving.py", "candidate-build.json", "capture_mixed_steps.py",
    "check_flash_next_native_support.py", "gauge-fix.patch", "mxfp4-inventory.json",
    "qualify_qwen38_flash_next.py", "run_flash_next_lifecycle.py",
    "run_flash_next_prefill_split_probe.py", "run_live_prefix.py", "run_live_prefix_m5.sh",
    "run_qwen38_reference_serving.py", "run_selective_mixed.py", "run_warm_prefix.py",
    "split_step_route.py",
)
MODES = ("cold-1", "warm-1", "warm-2", "cold-2")
RAW = ("receipt.json", *(f"{mode}{suffix}" for mode in MODES for suffix in (".log", "-samples.jsonl")))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(archive, name):
    item = archive.getmember(name)
    require(item.isfile(), f"Expected a regular archive member: {name}")
    with archive.extractfile(item) as handle:
        return handle.read()


def checksums(raw):
    lines = raw.decode().splitlines()
    found = {name: digest for digest, name in map(str.split, lines)}
    require(len(found) == len(lines), "Duplicate input checksum")
    return found


def replay(path):
    with (tarfile.open(path) as archive, tarfile.open(M2) as guarded,
          tarfile.open(CORPUS_ARCHIVE) as corpus, TemporaryDirectory(prefix="flash-next-live-prefix-") as directory):
        extra = {"inputs.sha256", "status.txt", "replay_live_prefix.py"}
        expected = set(DRIVERS) | extra | {f"observed/{name}" for name in RAW}
        names = archive.getnames()
        require(set(names) == expected and len(names) == len(expected)
                and read(archive, "status.txt") == b"0\n", "Trial evidence is incomplete")
        installed = {name: read(archive, name) for name in names}
        installed[CORPUS] = read(corpus, CORPUS)
        manifest = checksums(installed["inputs.sha256"])
        inherited = checksums(read(guarded, "inputs.sha256"))
        require(set(manifest) == set(DRIVERS) | {CORPUS, BIN}
                and all(sha256(installed[key]).hexdigest() == digest for key, digest in manifest.items() if key != BIN)
                and all(manifest[key] == inherited[key] for key in manifest.keys() & inherited.keys())
                and installed["candidate-build.json"] == read(guarded, "candidate-build.json")
                and manifest[BIN] == json.loads(installed["candidate-build.json"])["server_sha256"],
                "The driver, corpus, runtime identity, or pack inventory changed")
        stage = Path(directory)
        (stage / "observed").mkdir()
        for name, body in installed.items():
            (stage / name).write_bytes(body)
        return subprocess.check_output([sys.executable, "-B", stage / "replay_live_prefix.py", stage / "observed"])


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    sys.stdout.buffer.write(replay(parser.parse_args().archive))
