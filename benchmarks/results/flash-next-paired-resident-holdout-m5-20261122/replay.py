"""Replay the M5 resident reader holdout against the same binary and M2 shard inputs."""

from gzip import decompress
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "flash-next-paired-resident-reads-m5-20261122"
READER = ROOT.parent / "flash-next-paired-resident-reads-m2-20261122"
HOLDOUT = ROOT.parent / "flash-next-resident-startup-attribution-m5-20261122"


def replay():
    spec = importlib.util.spec_from_file_location("paired_reader", BASE / "replay.py")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    with TemporaryDirectory() as folder:
        inputs = Path(folder)
        for file in ("build.json", "direct_weights.cpp", "run_probe.py"):
            (inputs / file).write_bytes((READER / file).read_bytes())
        (inputs / "result-m2.json").write_bytes((HOLDOUT / "holdout-m2.json").read_bytes())
        (inputs / "selection.json").write_bytes(decompress((HOLDOUT / "holdout-selection.json.gz").read_bytes()))
        result = reader.replay(ROOT, inputs)
    shards = result["shards"]
    expected = {f"model-{index:05d}-of-00025.safetensors" for index in (17, 18)}
    if shards.keys() != expected or sum(s["matched_tensors_per_arm"] for s in shards.values()) != 316:
        raise ValueError("The startup-sensitive holdout shard or tensor count changed")
    for shard in shards.values():
        phases = shard["median_seconds"]
        shard["paired_change_percent"] = {
            state: 100 * (modes["pread-pair"] / modes["pread"] - 1) for state, modes in phases.items()
        }
    return result


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
