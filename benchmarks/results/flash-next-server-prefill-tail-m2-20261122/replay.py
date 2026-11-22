"""Replay the Flash Next server-side decode-tail HTTP functional check."""

from hashlib import file_digest
import json
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "flash-next-late-prefill-deferral-m5-20261122" / "evidence.tar.xz"
MANIFEST = "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"


def need(valid, message):
    if not valid:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def unpack(path, root, named):
    with tarfile.open(path) as archive:
        for member in archive:
            name = member.name
            if name not in named:
                continue
            dest = root / name
            need(member.isfile() and Path(name).name == name and not dest.exists(),
                 "Unexpected archive member")
            dest.write_bytes(archive.extractfile(member).read())
    need(all((root / name).is_file() for name in named), "Evidence or driver archive is incomplete")


def check_samples(root, label, count):
    samples = [json.loads(line) for line in (root / f"{label}-samples.jsonl").read_text().splitlines()]
    need(samples and len(samples) == count and all(s["pressure"] == 1 for s in samples),
         f"{label}: pressure or sample count changed")


def replay(root, prior):
    build = read(root / "build.json")
    hashes = {name: checksum for checksum, name in
              (line.split() for line in (root / "inputs-m2.sha256").read_text().splitlines())}
    need(hashes.pop("build.json") == digest(root / "build.json")
         and hashes.pop("ax-engine-server-tail") == build["server_sha256"]
         and hashes.pop("run_server_tail.py") == build["driver_sha256"] == digest(root / "run_server_tail.py")
         and build["runtime_commit"].startswith("2b905c012") and build["profile"] == "release-server",
         "The replay did not bind the corrected executable and harness")
    hashes.pop("run_m2.sh")
    with TemporaryDirectory() as directory:
        files = Path(directory)
        unpack(prior, files, set(hashes))
        need(all(digest(files / name) == value for name, value in hashes.items()),
             "The original prompts or helper source changed")
        (files / "run_server_tail.py").write_bytes((root / "run_server_tail.py").read_bytes())
        sys.path.insert(0, str(files))
        import run_server_tail as runner
        labels = [name for name, _ in runner.ARMS[:2]]
        raw = {
            "solo.json", "solo-samples.jsonl", "receipt.json", "status.txt",
            *(f"{label}{suffix}" for label in labels for suffix in (".json", "-samples.jsonl")),
        }
        unpack(root / "evidence.tar.xz", files, raw)
        observed = read(files / "receipt.json")
        need((files / "status.txt").read_text() == "0\n" and observed["status"] == "complete"
             and observed["scope"] == "M2 functional" and observed["qualification"] is False
             and observed["build"] == build and observed["inventory_sha256"] == hashes["mxfp4-inventory.json"]
             and observed["manifest"] == {"model-manifest.json": MANIFEST}
             and observed["pack_revision"] == read(files / "mxfp4-inventory.json")["revision"]
             and list(observed["arms"]) == labels,
             "The run did not complete on the sealed code and pack")
        _, cases, frozen = runner.layer.files()["ending-decoder"]
        solo = files / "solo.json"
        need(observed["solo"]["data_sha256"] == digest(solo), "Single-slot oracle changed")
        check_samples(files, "solo", observed["solo"]["sample_count"])
        expected = read(solo)
        need(len(expected) == len(cases) == 2, "Incomplete single-slot oracle")
        for row, case in zip(expected, cases, strict=True):
            need(row["ok"] and row["prompt_token_ids"] == case.input_tokens
                 and len(row["output_token_ids"]) == case.max_output_tokens == 96,
                 "Invalid single-slot response")
        result = {}
        for label, enabled in runner.ARMS[:2]:
            meta = observed["arms"][label]
            need(meta["scheduler_tail_deferral"] is enabled
                 and meta["cpu_idle_before_percent"] is meta["cpu_idle_after_percent"] is None,
                 "Unexpected functional arm conditions")
            path = files / f"{label}.json"
            need(meta["data_sha256"] == digest(path), "Measured stream data changed")
            observation = read(path)
            comparison = runner.verify(observation, enabled)
            need(all(observation[k] == value for k, value in comparison.items())
                 and observation["drained"] == dict.fromkeys(runner.mixed.COUNTERS, 0),
                 "Replay disagrees with the measured scheduler or shutdown")
            for row, solo_row in zip(observation["observations"], expected, strict=True):
                need(row["ok"] and all(row[k] == solo_row[k] for k in ("prompt_token_ids", "output_token_ids")),
                     "The concurrent response differs from the single-slot oracle")
            check_samples(files, label, meta["sample_count"])
            result[label] = {
                "matching_generated_ids": sum(len(row["output_token_ids"]) for row in expected),
                "normal_pressure_samples": meta["sample_count"],
                "held_decode_steps": comparison["held_decode_steps"],
                "prefill_overlapped": comparison["prefill_overlapped"],
            }
        return {"scope": observed["scope"], "qualification": False,
                "runtime_commit": build["runtime_commit"], "arms": result,
                "differs_from_m5_archive_by_prompt": {
                    str(len(case.input_tokens)): row["output_token_ids"] != frozen[case.id]
                    for case, row in zip(cases, expected, strict=True)
                }}


if __name__ == "__main__":
    print(json.dumps(replay(ROOT, PRIOR), indent=2, allow_nan=False))
