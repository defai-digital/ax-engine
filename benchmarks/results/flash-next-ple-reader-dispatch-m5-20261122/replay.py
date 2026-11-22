"""Bind and compare the M5 native PLE reader dispatch diagnostic."""

from gzip import decompress
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from statistics import median, quantiles

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "flash-next-ple-reader-dispatch-m2-20261122"


def require(valid, message):
    if not valid:
        raise ValueError(message)


def replay(root=ROOT, prior_root=PRIOR):
    spec = importlib.util.spec_from_file_location("m2_reader_dispatch", prior_root / "replay.py")
    require(spec is not None and spec.loader is not None, "Missing M2 driver")
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    anchor = prior.replay()

    record = json.loads(decompress((root / "observed-m5.json.gz").read_bytes()))
    build = record["build"]
    inputs = dict(line.split()[::-1] for line in (root / "inputs-m5.sha256").read_text().splitlines())
    require(record["completed"] is True and record["qualification"] is False
            and (root / "status.txt").read_text() == "0\n"
            and record["scope"] == "m5-diagnostic" and record["soc"] == "Apple M5 Max"
            and record["memory_bytes"] == 128 * 2**30 and record["os"] == "27.2.1"
            and record["pack_revision"] == prior.REVISION and record["inventory_sha256"] == prior.INVENTORY
            and record["manifest"] == {"model-manifest.json": prior.MANIFEST}
            and build == anchor["build"],
            "Unexpected hardware, binary, pack or terminal result")
    require(inputs["inventory.json"] == record["inventory_sha256"]
            and inputs["ax-ple-reader-test"] == build["binary_sha256"]
            and inputs["run_probe.py"] == build["driver_sha256"]
            and inputs["workload.json"] == build["workload_sha256"]
            and sha256((prior_root / "run_probe.py").read_bytes()).hexdigest() == build["driver_sha256"],
            "M5 inputs differ from the sealed executable and workload")
    cpu = [record["pre_cpu_idle_percent"], record["post_cpu_idle_percent"]]
    require(all(90 <= value <= 100 for value in cpu), "The M5 timing window missed the idle floor")
    observed = record["arms"]
    reader_keys = ("reader_threads",)
    if "reader_pool_active" in anchor["observed"][0]:
        reader_keys += ("reader_pool_active",)
        require(all(type(arm["reader_pool_active"]) is bool for arm in observed),
                "M5 pool activation must be explicit")
    require([tuple(arm[k] for k in reader_keys) for arm in observed]
            == [tuple(arm[k] for k in reader_keys) for arm in anchor["observed"]],
            "The M5 reader order changed")
    roster = [(shape, phase) for shape in prior.SHAPES for phase in prior.PHASES]
    results = []
    for arm in observed:
        require(arm["unique_table_shards"] == 128 and arm["embedding_width"] == 160
                and arm["pressure_before"] == arm["pressure_after"] == 1
                and arm["workload_sha256"] == build["workload_sha256"]
                and [(stage["shape"], stage["phase"]) for stage in arm["stages"]] == roster,
                "Incorrect M5 table, stage order, workload or pressure")
        timings = {}
        for stage in arm["stages"]:
            shape = stage["shape"]
            reference = anchor["matching_raw_payloads"][shape]
            samples = stage["elapsed_ns"]
            require(len(samples) == prior.SHAPES[shape][0]
                    and all(type(ns) is int and ns > 0 for ns in samples)
                    and stage["rows_requested"] == reference["requested_rows"]
                    and stage["bytes_read"] == reference["bytes_read"]
                    and stage["payload_sha256"] == reference["sha256"],
                    f"M5 PLE staging or measurements changed for {shape}")
            timings[f"{shape}/{stage['phase']}"] = {
                "samples": len(samples),
                "p50_ms": round(median(samples) / 1e6, 6),
                "p95_ms": round(quantiles(samples, n=100, method="inclusive")[94] / 1e6, 6),
            }
        results.append({key: arm[key] for key in reader_keys} | {"stages": timings})
    return {"scope": "native PLE staging diagnostic on M5 Max", "qualification": False,
            "same_binary_as_m2": True, "cpu_idle_percent": cpu,
            "matching_raw_payloads": anchor["matching_raw_payloads"], "observed": results}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
