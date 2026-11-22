"""Replay the Flash Next resident PLE worker comparison on M5 Max."""

from gzip import decompress
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
M2 = ROOT.parent / "flash-next-ple-serving-m2-20261122"
SHA_FILES = {"check_flash_next_native_support.py", "qualify_qwen38_flash_next.py",
             "run_flash_next_generate_matrix.py"}


def need(valid, message):
    if not valid:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def sha(data):
    return sha256(data).hexdigest()


def replay(root, serving):
    observed = read(root / "result.json")
    build = read(root / "build.json")
    hashes = dict(line.split()[::-1] for line in (root / "inputs-m5.sha256").read_text().splitlines())
    need((root / "status.txt").read_text() == "0\n" and observed["scope"] == "M5 serving comparison"
         and observed["qualification"] is False and observed["build"] == build == read(M2 / "build.json")
         and observed["hardware"] ==
             {"soc": "Apple M5 Max", "memory_bytes": 128 * 2**30, "os": "27.2.1"},
         "The serving result or executable identity changed")
    need(hashes["build.json"] == sha((root / "build.json").read_bytes())
         and hashes["ax-engine-server-ple-pool"] == build["server_sha256"]
         and hashes["count-workers"] == build["observer_sha256"]
         and hashes["run_serving.py"] == build["driver_sha256"] == sha((M2 / "run_serving.py").read_bytes())
         and hashes["run_flash_next_generate_matrix.py"] == build["matrix_sha256"]
         and sha((M2 / "count_workers.c").read_bytes()) == build["observer_source_sha256"],
         "The native comparison or its worker observer changed")
    need(observed["inventory_sha256"] == serving.INVENTORY == hashes["inventory.json"]
         and observed["pack_revision"] == serving.REVISION
         and observed["manifest"] == {"model-manifest.json": serving.MANIFEST},
         "The publisher snapshot binding changed")
    expected = [("initial", False)] + [
        (f"arm-{index}", pooled) for index, pooled in enumerate(serving.PRIMARY_ARMS, 1)
    ]
    arms = observed["arms"]
    need([(arm["label"], arm["pooled"]) for arm in arms] == expected,
         "The conditioning and counterbalanced orders changed")
    identities = []
    lengths = serving.matrix.LENGTHS
    for arm in arms:
        pooled = arm["pooled"]
        idle = arm["cpu_idle_percent"]
        need(arm["context_tokens"] == serving.CONTEXT and arm["pressure_samples"] >= 2
             and type(pooled) is bool and arm["pool_workers"] == (serving.POOL_THREADS if pooled else 0)
             and len(idle) == 2 and all(type(x) in (int, float) and 0 <= x <= 100 for x in idle)
             and arm["timing_eligible"] == all(90 <= x for x in idle),
             "Unexpected context, worker count or host conditions")
        data = decompress((root / f"{arm['label']}.json.gz").read_bytes())
        need(sha(data) == arm["raw_sha256"], "Measured HTTP stream data changed")
        raw = json.loads(data)
        identities.append(serving.validate(raw, serving.matrix.PROMPTS, lengths))
        need(arm["ready_s"] == raw["ready_seconds"] and arm["cells"] == serving.metrics(raw, lengths),
             "Reported readiness or HTTP timing does not replay")
    identity = identities[0]
    need(all(row == identity for row in identities) and
         len(identity["outputs"]) == observed["matching_responses_per_arm"],
         "Prompt, output or stop identity changed")

    comparisons = []
    for control, treatment in ((arms[1], arms[2]), (arms[4], arms[3])):
        eligible = control["timing_eligible"] and treatment["timing_eligible"]
        lengths = {}
        if eligible:
            for length, metrics in control["cells"].items():
                other = treatment["cells"][length]
                lengths[length] = {
                    "ttft_ratio": other["ttft_median_s"] / metrics["ttft_median_s"],
                    "decode_ratio": other["decode_median_tok_s"] / metrics["decode_median_tok_s"],
                }
        comparisons.append({
            "control": control["label"], "pooled": treatment["label"],
            "timing_eligible": eligible,
            "lengths": lengths,
        })
    return {
        "scope": observed["scope"], "qualification": False,
        "readiness_by_arm_s": {arm["label"]: arm["ready_s"] for arm in arms},
        "cpu_idle_percent": {arm["label"]: arm["cpu_idle_percent"] for arm in arms},
        "pressure_samples": sum(arm["pressure_samples"] for arm in arms),
        "matching_responses_per_arm": len(identity["outputs"]),
        "matching_output_ids_per_arm": sum(len(row[1]) for row in identity["outputs"]),
        "pairs": comparisons,
    }


if __name__ == "__main__":
    checksums = dict(line.split()[::-1] for line in (ROOT / "inputs-m5.sha256").read_text().splitlines())
    with TemporaryDirectory() as directory:
        source = Path(directory)
        with tarfile.open(M2 / "dependencies.tar.xz") as archive:
            need(len(archive.getnames()) == len(SHA_FILES) and set(archive.getnames()) == SHA_FILES,
                 "Unexpected sealed dependency roster")
            for member in archive:
                need(member.isfile(), "Invalid dependency type")
                data = archive.extractfile(member).read()
                need(sha(data) == checksums[member.name], "A sealed dependency changed")
                (source / member.name).write_bytes(data)
        sys.path[:0] = [str(source), str(M2)]
        import run_serving
        print(json.dumps(replay(ROOT, run_serving), indent=2, allow_nan=False))
