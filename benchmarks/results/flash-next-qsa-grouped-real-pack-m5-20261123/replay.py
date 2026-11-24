"""Verify the signed full-model M5 grouped QSA evaluation crossover."""

from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
from statistics import median
import tarfile
from tempfile import TemporaryDirectory


ARMS = (0, 2, 2, 0)
GUARD = 96 << 30
EXECUTABLE = "ax-flash-grouped-qsa-test"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify_inputs(root):
    checks = {filename: sha for sha, filename in map(
        str.split, (root / "inputs.sha256").read_text().splitlines()
    )}
    bundle = root / "bundle"
    assert {f.name for f in bundle.iterdir()} == checks.keys() - {EXECUTABLE}
    assert all(digest(file) == checks[file.name] for file in bundle.iterdir())
    build = read(root / "build.json")
    assert build == read(bundle / "build.json")
    assert checks[EXECUTABLE] == build["binary_sha256"]
    assert checks["source.patch"] == build["patch_sha256"]
    assert checks["prompt.json"] == build["prompt_sha256"]
    assert build["profile"] == "release-server"
    assert build["mlx"] == "0.32.4" and build["codesign_verified"] is True
    assert build["rustc"].startswith("rustc 1.99.1 ") and build["cargo"].startswith("cargo 1.99.1 ")
    assert len(read(bundle / "mxfp4-inventory.json")["members"]) == 49
    assert len(read(bundle / "prompt.json")) == 8192
    return build


def verify(root):
    assert (root / "status.txt").read_text().strip() == "0"
    build = verify_inputs(root)
    observed = root / "observed"
    archive = read(observed / "receipt.json")
    assert archive["build"] == build and archive["qualification"] is False
    assert archive["soc"] == "Apple M5 Max" and archive["memory_bytes"] == 128 << 30
    assert archive["storage"]["internal"] and archive["storage"]["solid_state"]
    assert archive["manifest"] == {"model-manifest.json": build["manifest_sha256"]}
    assert archive["physical_cap_bytes"] == GUARD
    findings = []
    result_identities = set()
    for index, group in enumerate(ARMS):
        name = f"{index:02d}-width2048-group{group}"
        folder = observed / name
        receipt = read(folder / "receipt.json")
        assert receipt["arm"] == name and receipt["evaluation_group"] == group and receipt["width"] == 2048
        assert receipt["exit_code"] == 0 and receipt["passed"] and not receipt["guard_triggered"]
        assert receipt["all_pressure_normal"] and not receipt["timeout"]
        rows = [json.loads(line) for line in (folder / "native-samples.jsonl").read_text().splitlines()]
        assert len(rows) == receipt["pressure_samples"] > 0
        assert all(row["pressure"] == 1 for row in rows)
        assert max(row["footprint_bytes"] for row in rows) == receipt["peak_physical_bytes"] <= GUARD
        assert receipt["result_sha256"] == digest(folder / "result.json")
        output = read(folder / "result.json")
        assert output["evaluation_group"] == group and output["qualification"] is False
        assert output["width"] == 2048 and output["peak_bytes"] > 0
        assert math.isfinite(output["seconds"]) and output["seconds"] > 0
        assert len(receipt["cpu_idle_percent"]) == 2
        assert all(0 <= value <= 100 for value in receipt["cpu_idle_percent"])
        tiles, fences = output["completed_tiles"], output["evaluation_fences"]
        if group == 2:
            assert tiles > fences > 0 and tiles <= 2 * fences
        else:
            assert (tiles, fences) == (0, 0)
        cursor = 0
        for (start, width, elapsed), want in zip(output["chunks"], [2047, 1] * 4, strict=True):
            assert start == cursor and width == want and elapsed > 0
            cursor += width
        assert cursor == 8192 and sum(chunk[2] for chunk in output["chunks"]) <= output["seconds"]
        result_identities.add((output["logit_count"], output["state_bytes"],
                               output["logits_sha256"], output["state_sha256"]))
        findings.append(receipt | {"output": output})
    assert archive["execution_arm_order"] == [receipt["arm"] for receipt in findings]
    assert len(result_identities) == 1 and archive["identical_state_and_logits"] is True
    original = [row for row in findings if row["evaluation_group"] == 0]
    grouped = [row for row in findings if row["evaluation_group"] == 2]
    pairs = []
    for before, after in zip(original, grouped, strict=True):
        pairs.append({
            "grouped_minus_lazy_peak_mlx_bytes": after["output"]["peak_bytes"] - before["output"]["peak_bytes"],
            "grouped_minus_lazy_peak_physical_bytes": after["peak_physical_bytes"] - before["peak_physical_bytes"],
            "grouped_over_lazy_forward": after["output"]["seconds"] / before["output"]["seconds"],
            "endpoints_90_percent_idle": all(
                value >= 90 for row in (before, after) for value in row["cpu_idle_percent"]
            ),
        })
    return {
        "qualification": False,
        "hardware": {key: archive[key] for key in ("soc", "memory_bytes", "macos", "storage")},
        "build": build, "arms": findings, "paired_comparisons": pairs,
        "all_state_and_logits_bitwise_equal": True,
        "grouped_over_lazy_forward_median":
            median(row["output"]["seconds"] for row in grouped) /
            median(row["output"]["seconds"] for row in original),
    }


def replay():
    with (TemporaryDirectory(prefix="flash-next-grouped-qsa-m5-") as tmp,
          tarfile.open(Path(__file__).with_name("evidence.tar.xz")) as archive):
        names = archive.getnames()
        assert len(names) == len(set(names))
        for item in archive:
            name = PurePosixPath(item.name)
            assert item.isfile() and not name.is_absolute() and ".." not in name.parts
            assert name.name != EXECUTABLE
        archive.extractall(tmp, filter="data")
        return verify(Path(tmp))


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
