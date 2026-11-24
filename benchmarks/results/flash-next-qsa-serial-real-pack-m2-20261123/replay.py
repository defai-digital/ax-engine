"""Verify the fresh-process Flash Next QSA allocation diagnostic."""

from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import signal
from statistics import median
import tarfile
from tempfile import TemporaryDirectory

CAP = 96 << 30
ARMS = ((512, False), (512, True), (2048, True), (2048, False))
TEST = "flash_next_serial_qsa_tile_eval_exactness_and_memory"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def samples(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows and all(type(row["footprint_bytes"]) is int and row["footprint_bytes"] >= 0 for row in rows)
    return {
        "count": len(rows),
        "max_physical_bytes": max(row["footprint_bytes"] for row in rows),
        "end_physical_bytes": rows[-1]["footprint_bytes"],
        "max_before_last": max((row["footprint_bytes"] for row in rows[:-1]), default=0),
        "all_normal": all(row["pressure"] == 1 for row in rows),
        "swap_unchanged": len({row["swap"] for row in rows}) == 1,
        "pressure_guard": any(row["pressure"] in (4, 8) for row in rows),
    }


def verify_inputs(root, *, executable):
    expected = {name: checksum for checksum, name in map(
        str.split, (root / "inputs.sha256").read_text().splitlines()
    )}
    bundle = root / "bundle"
    assert set(expected) - {executable} == {item.name for item in bundle.iterdir()}
    for name, checksum in expected.items():
        if name != executable:
            assert digest(bundle / name) == checksum, name
    build = read(root / "build.json")
    assert read(bundle / "build.json") == build
    assert build["binary_sha256"] == expected[executable]
    assert build["patch_sha256"] == expected["source.patch"]
    assert build["prompt_sha256"] == expected["prompt.json"]
    assert build["mlx"] == "0.32.4" and build["profile"] == "release-server"
    assert build["rustc"].startswith("rustc 1.99.1 ")
    assert build["cargo"].startswith("cargo 1.99.1 ")
    assert len(read(bundle / "prompt.json")) == 8192
    assert len(read(bundle / "mxfp4-inventory.json")["members"]) == 49
    return build


def verify_arm(path, width, serial):
    receipt = read(path / "receipt.json")
    assert receipt["width"] == width and receipt["serial_eval"] is serial
    assert receipt["arm"] == path.name
    measured = samples(path / "native-samples.jsonl")
    assert measured["count"] == receipt["pressure_samples"]
    assert measured["max_physical_bytes"] == receipt["peak_physical_bytes"]
    assert measured["all_normal"] == receipt["all_pressure_normal"]
    guarded = (measured["max_physical_bytes"] > CAP or measured["pressure_guard"])
    assert receipt["guard_triggered"] == guarded
    assert receipt["passed"] == (receipt["exit_code"] == 0 and measured["all_normal"] and not guarded)
    assert len(receipt["cpu_idle_percent"]) == 2
    assert all(math.isfinite(value) and 0 <= value <= 100 for value in receipt["cpu_idle_percent"])
    log = (path / "native.log").read_text()
    if guarded:
        assert receipt["exit_code"] == -signal.SIGTERM and not receipt["timeout"]
        assert receipt["result_sha256"] is None and not (path / "result.json").exists()
    result = None
    if receipt["passed"]:
        assert receipt["result_sha256"] == digest(path / "result.json")
        assert TEST in log and "1 passed; 0 failed" in log
        result = read(path / "result.json")
        assert result["qualification"] is False and result["width"] == width and result["serial_eval"] is serial
        assert (result["completed_tiles"] > 0) is serial
        for key in ("state_sha256", "logits_sha256"):
            assert len(bytes.fromhex(result[key])) == 32
        assert result["state_bytes"] > 0 and result["logit_count"] > 0
        widths = [512, 512, 512, 511, 1] * 4 if width == 512 else [2047, 1] * 4
        offset = 0
        assert math.isfinite(result["seconds"]) and result["seconds"] > 0
        for (start, size, seconds), expected in zip(result["chunks"], widths, strict=True):
            assert start == offset and size == expected
            assert math.isfinite(seconds) and seconds > 0
            offset += size
        assert offset == 8192 and sum(row[2] for row in result["chunks"]) <= result["seconds"]
    return receipt | {"samples": measured, "result": result}


def verify(root):
    original = root / "combined"
    repeat = root / "isolated"
    binary = "ax-flash-serial-qsa-test"
    old_build = verify_inputs(original, executable=binary)
    new_build = verify_inputs(repeat, executable=binary)
    assert new_build["codesign_verified"] is True
    for key in ("source_base", "manifest_sha256", "libmlx_sha256", "prompt_sha256"):
        assert old_build[key] == new_build[key]
    for name in ("prompt.json", "mxfp4-inventory.json"):
        assert digest(original / "bundle" / name) == digest(repeat / "bundle" / name)
    assert (original / "status.txt").read_text().strip() == "1"
    prior = samples(original / "observed/native-samples.jsonl")
    assert prior["max_before_last"] <= CAP < prior["end_physical_bytes"] == prior["max_physical_bytes"]
    assert prior["all_normal"] and prior["swap_unchanged"]
    final = read(repeat / "observed/receipt.json")
    assert final["qualification"] is False and final["build"] == new_build
    assert final["soc"] == "Apple M2 Ultra" and final["memory_bytes"] == 192 << 30
    assert final["storage"]["solid_state"]
    assert final["physical_cap_bytes"] == CAP
    assert final["manifest"] == {"model-manifest.json": new_build["manifest_sha256"]}
    arm_results = {}
    by_width = {}
    for width, serial in ARMS:
        name = f"{width}-{'serial' if serial else 'lazy'}"
        arm_results[name] = verify_arm(repeat / "observed" / name, width, serial)
    assert final["execution_arm_order"] == list(arm_results)
    for width in (512, 2048):
        lazy = arm_results[f"{width}-lazy"]
        serial = arm_results[f"{width}-serial"]
        exact = None
        ratios = None
        if lazy["passed"] and serial["passed"]:
            stock, experimental = lazy["result"], serial["result"]
            exact = all(stock[k] == experimental[k]
                        for k in ("state_sha256", "logits_sha256", "state_bytes", "logit_count"))
            ratios = {
                "serial_over_lazy_forward_time": experimental["seconds"] / stock["seconds"],
                "serial_minus_lazy_peak_mlx_bytes": experimental["peak_bytes"] - stock["peak_bytes"],
                "serial_minus_lazy_peak_physical_bytes": serial["peak_physical_bytes"] - lazy["peak_physical_bytes"],
            }
        assert final["identical_state_and_logits"][str(width)] is exact
        by_width[str(width)] = {"identical_state_and_logits": exact, "differences": ratios}
    assert (repeat / "status.txt").read_text().strip() == (
        "0" if all(item["identical_state_and_logits"] is not False for item in by_width.values()) else "1"
    )
    return {
        "qualification": False, "measurement_order": final["execution_arm_order"],
        "macos": final["macos"], "soc": final["soc"], "memory_bytes": final["memory_bytes"],
        "storage": final["storage"], "physical_cap_bytes": CAP,
        "initial_combined_guard": prior, "initial_build": old_build, "isolated_build": new_build,
        "arms": arm_results, "comparisons": by_width,
    }


def replay():
    with (TemporaryDirectory(prefix="flash-next-serial-qsa-m2-") as scratch,
          tarfile.open(Path(__file__).with_name("evidence.tar.xz")) as archive):
        names = archive.getnames()
        assert len(set(names)) == len(names)
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert path.name != "ax-flash-serial-qsa-test"
        archive.extractall(scratch, filter="data")
        return verify(Path(scratch))


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
