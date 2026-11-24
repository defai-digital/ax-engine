"""Verify primary-tier QSA scratch accounting against decoded outputs."""

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
MANIFEST_SHA256 = "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"


def read(path):
    return json.loads(path.read_bytes())


def read_rows(data):
    return [json.loads(line) for line in data.splitlines()]


def verify_arm(root, label, prompt_sha):
    folder = root / label
    assert (folder / "status.txt").read_text().strip() == "0"
    receipt = read(folder / "receipt.json")
    assert receipt["input_sha256"] == prompt_sha
    assert receipt["pack_sha256"] == {"model-manifest.json": MANIFEST_SHA256}
    assert receipt["model"] == {"id": "qwen3.8-flash-next", "context_length": 32768}
    assert receipt["mode"] == "stock-first" and receipt["qualification"] is False
    assert receipt["build"]["codesign_verified"] is True
    pressure = read_rows((folder / "server-samples.jsonl").read_bytes())
    assert len(pressure) == receipt["pressure_samples"] > 0
    assert all(row["pressure"] == 1 for row in pressure)
    assert max(row["footprint_bytes"] for row in pressure) == receipt["peak_footprint_bytes"] <= 96 << 30
    assert max(row["rss_bytes"] for row in pressure) == receipt["peak_rss_bytes"]
    raw = (folder / "requests.jsonl").read_bytes()
    assert sha256(raw).hexdigest() == receipt["requests_sha256"]
    rows = read_rows(raw)
    assert len(rows) == 5 and [row["warmup"] for row in rows] == [True, False, False, False, False]
    assert [row["prompt_length"] for row in rows[1:]] == [512, 2048, 8192, 8192]
    for row in rows:
        assert row["finish_reason"] == "max_output_tokens" and len(row["output_tokens"]) == 64
        assert len(row["resources_after"]) == 5 and not any(row["resources_after"].values())
        assert row["metrics_delta"]["ax_engine_mlx_prefill_forward_wall_us_total"] > 0
        for name, value in row["metrics_delta"].items():
            if "_mtp_" in name or name.endswith("paired_gather_tiles_total"):
                assert value == 0
    startup, reader = (folder / "startup.txt").read_text().splitlines()
    fields = dict(re.findall(r"(?<![\w-])([a-z-]+)=([^ ]+)", startup.split("memory plan: ", 1)[1]))
    assert fields["decision"] == "resident"
    assert fields["live-sequences"] == "1" and fields["live-sequence-cap"] == "uncapped"
    assert fields["ceiling"] == "115.20GiB" and fields["wired-shortfall"] == "0B"
    assert fields["ngram-table"].endswith("(not-resident)")
    assert reader.endswith("reader=AlignedBuffered")
    return receipt, rows, fields


def verify(root):
    entries = {name: checksum for checksum, name in map(
        str.split, (root / "inputs.sha256").read_text().splitlines()
    )}
    bundle = root / "bundle"
    assert {p.name for p in bundle.iterdir()} == entries.keys() - {"ax-engine-server"}
    for file in bundle.iterdir():
        assert sha256(file.read_bytes()).hexdigest() == entries[file.name], file.name
    build = read(bundle / "build.json")
    assert build["binary_sha256"] == entries["ax-engine-server"]
    assert build["patch_sha256"] == entries["source.patch"]
    assert len(read(bundle / "mxfp4-inventory.json")["members"]) == 49
    before, old_rows, old_plan = verify_arm(root, "before", entries["prompts.json"])
    after, new_rows, new_plan = verify_arm(root, "after", entries["prompts.json"])
    assert after["build"] == build
    assert before["flags"] == after["flags"]
    prior = HERE.parent / "flash-next-paired-qsa-http-m5-20261123" / "comparison.json"
    assert read(prior)["binary_sha256"] == before["build"]["binary_sha256"]
    assert before["build"]["libmlx_sha256"] == after["build"]["libmlx_sha256"]
    changes = {key: [old_plan[key], new_plan[key]] for key in old_plan if old_plan[key] != new_plan[key]}
    assert changes == {
        "prefill-scratch": ["1.69GiB", "12.02GiB"],
        "working-set": ["81.52GiB", "91.85GiB"],
    }
    for previous, current in zip(old_rows, new_rows, strict=True):
        for name in ("prompt", "prompt_sha256", "prompt_length", "output_tokens", "warmup"):
            assert previous[name] == current[name], name
    return {
        "qualification": False,
        "matched_output_ids": sum(len(row["output_tokens"]) for row in new_rows if not row["warmup"]),
        "reserved_changes": changes, "expert_placement": "resident",
        "reader": "AlignedBuffered", "context_tokens": 32768, "live_sequences": 1,
        "control": {key: before[key] for key in ("build", "ready_seconds", "pressure_samples", "peak_footprint_bytes")},
        "corrected": {key: after[key] for key in ("build", "ready_seconds", "pressure_samples", "peak_footprint_bytes")},
    }


def replay():
    with (TemporaryDirectory(prefix="flash-next-reservation-m5-") as tmp,
          tarfile.open(HERE / "evidence.tar.xz") as archive):
        names = archive.getnames()
        assert len(names) == len(set(names))
        for item in archive:
            path = PurePosixPath(item.name)
            assert item.isfile() and not path.is_absolute() and ".." not in path.parts
            assert path.name != "ax-engine-server"
        archive.extractall(tmp, filter="data")
        return verify(Path(tmp))


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
