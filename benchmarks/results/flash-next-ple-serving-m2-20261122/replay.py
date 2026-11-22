"""Replay Flash Next native HTTP identity for the scoped and pooled workers."""

from argparse import ArgumentParser
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
LENGTHS = (512,)
DEPENDENCIES = {
    "check_flash_next_native_support.py",
    "qualify_qwen38_flash_next.py",
    "run_flash_next_generate_matrix.py",
}


def need(valid, message):
    if not valid:
        raise ValueError(message)


def replay(root, source):
    import run_serving as serving

    receipt = json.loads((root / "result.json").read_bytes())
    build = json.loads((root / "build.json").read_bytes())
    need(receipt["scope"] == "M2 functional serving" and receipt["qualification"] is False
         and receipt["hardware"]["soc"] == "Apple M2 Ultra"
         and receipt["hardware"]["memory_bytes"] == 192 * 2**30,
         "The result does not describe the observed functional tier")
    need(receipt["build"] == build
         and serving.digest(ROOT / "count_workers.c") == build["observer_source_sha256"]
         and serving.digest(ROOT / "run_serving.py") == build["driver_sha256"]
         and serving.digest(source / "run_flash_next_generate_matrix.py") == build["matrix_sha256"],
         "The recorded source changed")
    need(receipt["pack_revision"] == serving.REVISION
         and receipt["manifest"] == {"model-manifest.json": serving.MANIFEST}
         and receipt["inventory_sha256"] == serving.INVENTORY,
         "The pinned pack changed")
    arms = receipt["arms"]
    need([(a["label"], a["pooled"], a["pool_workers"]) for a in arms]
         == [("arm-1", False, 0), ("arm-2", True, serving.POOL_THREADS)],
         "The scoped and pooled worker counts changed")
    identities = []
    for arm in arms:
        need(arm["context_tokens"] == serving.CONTEXT and arm["pressure_samples"] > 0
             and arm["timing_eligible"] is False and arm["cpu_idle_percent"] == [None, None],
             "Unexpected context or timing eligibility")
        observed = gzip.decompress((root / (arm["label"] + ".json.gz")).read_bytes())
        need(sha256(observed).hexdigest() == arm["raw_sha256"], "Native observation bytes changed")
        raw = json.loads(observed)
        need(raw["ready_seconds"] == arm["ready_s"] and serving.metrics(raw, LENGTHS) == arm["cells"],
             "The receipt disagrees with the native stream metrics")
        identities.append(serving.validate(raw, serving.matrix.PROMPTS[:1], LENGTHS))
    need(identities[0] == identities[1] and len(identities[0]["outputs"]) == 6
         and receipt["matching_responses_per_arm"] == 6,
         "Prompt digests, generated IDs or stop statuses changed")
    return {
        "scope": receipt["scope"], "qualification": False, "context_tokens": serving.CONTEXT,
        "pool_workers_by_arm": [arm["pool_workers"] for arm in arms],
        "matching_responses_per_arm": len(identities[0]["outputs"]),
        "matching_output_ids_per_arm": sum(len(row[1]) for row in identities[0]["outputs"]),
        "pressure_samples_by_arm": [arm["pressure_samples"] for arm in arms],
        "timing_eligible": False,
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT)
    args = parser.parse_args()
    with TemporaryDirectory() as directory, tarfile.open(ROOT / "dependencies.tar.xz") as archive:
        source = Path(directory)
        members = archive.getmembers()
        need(len(members) == len(DEPENDENCIES) and set(archive.getnames()) == DEPENDENCIES,
             "Unexpected native harness dependencies")
        for member in members:
            need(member.isfile(), "Unexpected native harness source path")
            (source / member.name).write_bytes(archive.extractfile(member).read())
        sys.path.insert(0, str(source))
        sys.stdout.write(json.dumps(replay(args.evidence, source), indent=2) + "\n")
