"""Verify the repeated paired QSA forward diagnostic and host CPU trace."""
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from statistics import median
import tarfile
from tempfile import TemporaryDirectory


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify(root):
    bundle = root / "bundle"
    entries = {
        filename: digest
        for digest, filename in map(str.split, (root / "inputs.sha256").read_text().splitlines())
    }
    assert {path.name for path in bundle.iterdir()} == entries.keys() - {"ax-flash-paired-qsa-test"}
    for file in bundle.iterdir():
        assert sha256(file.read_bytes()).hexdigest() == entries[file.name], file.name
    assert (root / "build.json").read_bytes() == (bundle / "build.json").read_bytes()
    assert (root / "status.txt").read_text().strip() == "0"
    observed = root / "observed"
    receipt = read(observed / "receipt.json")
    build = read(root / "build.json")
    assert receipt["build"] == build and receipt["qualification"] is False
    assert entries["ax-flash-paired-qsa-test"] == build["binary_sha256"]
    assert entries["source.patch"] == build["patch_sha256"]
    assert entries["prompt.json"] == build["prompt_sha256"]
    assert receipt["manifest"] == {"model-manifest.json": build["manifest_sha256"]}
    assert receipt["storage"]["internal"] and receipt["storage"]["solid_state"]
    assert len(read(bundle / "mxfp4-inventory.json")["members"]) == 49
    assert len(read(bundle / "prompt.json")) == 8192
    assert receipt["soc"] == "Apple M5 Max" and receipt["memory_bytes"] == 128 << 30
    for name, key in (("result.json", "result_sha256"), ("host-cpu.jsonl", "host_cpu_sha256")):
        assert sha256((observed / name).read_bytes()).hexdigest() == receipt[key]
    result = read(observed / "result.json")
    assert result["final_state_and_logits_exact"] and result["qualification"] is False
    arms = result["arms"]
    assert [r["paired"] for r in arms] == [True, False, False, True]
    assert [r["paired_dispatches"] for r in arms] == [432, 0, 0, 432]
    for arm in arms:
        widths = [512, 512, 512, 511, 1] * 4
        offset = 0
        for (start, width, elapsed), expected in zip(arm["chunks"], widths, strict=True):
            assert start == offset and width == expected and elapsed > 0
            offset += width
        assert offset == 8192 and sum(chunk[2] for chunk in arm["chunks"]) <= arm["seconds"]
    pressure = rows(observed / "native-samples.jsonl")
    assert len(pressure) == receipt["pressure_samples"] > 0
    assert all(row["pressure"] == 1 and row["footprint_bytes"] <= 96 << 30 for row in pressure)
    trace = rows(observed / "host-cpu.jsonl")
    assert [row["kind"] for row in trace] == ["start", *(["sample"] * (len(trace) - 2)), "end"]
    assert all(b["monotonic_ns"] > a["monotonic_ns"] for a, b in zip(trace, trace[1:]))
    delta = {key: sum((b["ticks"][key] - a["ticks"][key]) % (1 << 32)
                      for a, b in zip(trace, trace[1:])) for key in ("user", "system", "idle", "nice")}
    endpoints = receipt["cpu_idle_percent"]
    assert len(endpoints) == 2 and all(0 <= p <= 100 for p in endpoints)
    return {
        "qualification": False, "timing_endpoint_eligible": all(p >= 90 for p in endpoints),
        "cpu_idle_endpoints": endpoints, "host_tick_totals_process_window": delta,
        "host_idle_percent_process_window": 100 * delta["idle"] / sum(delta.values()),
        "paired_over_stock_median": median([arms[0]["seconds"], arms[3]["seconds"]]) /
                                   median([arms[1]["seconds"], arms[2]["seconds"]]),
        "paired_over_stock_order": [arms[0]["seconds"] / arms[1]["seconds"],
                                    arms[3]["seconds"] / arms[2]["seconds"]],
        "arms": arms, "pressure_samples": len(pressure), "build": build,
    }


def replay():
    source = Path(__file__).with_name("evidence.tar.xz")
    with (TemporaryDirectory(prefix="flash-next-paired-quiet-") as scratch,
          tarfile.open(source) as archive):
        names = archive.getnames()
        assert len(names) == len(set(names))
        for entry in archive:
            name = PurePosixPath(entry.name)
            assert entry.isfile() and not name.is_absolute() and ".." not in name.parts
        archive.extractall(scratch, filter="data")
        return verify(Path(scratch))


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
