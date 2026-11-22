"""Check native PLE staging byte identity and summarize reader timings."""

from gzip import decompress
from hashlib import sha256
import json
from pathlib import Path
from statistics import median, quantiles

ROOT = Path(__file__).resolve().parent
SHAPES = {"decode": (288, 4608), "prefill": (4, 32768)}
PHASES = ("initial", "repeated")
REVISION = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
MANIFEST = "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"
INVENTORY = "97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6"


def require(valid, message):
    if not valid:
        raise ValueError(message)


def replay():
    result = json.loads(decompress((ROOT / "observed-m2.json.gz").read_bytes()))
    source = decompress((ROOT / "probe.patch.gz").read_bytes())
    workload = decompress((ROOT / "workload.json.gz").read_bytes())
    build = result["build"]
    require(result["completed"] is True and result["qualification"] is False
            and result["scope"] == "functional" and result["soc"] == "Apple M2 Ultra"
            and result["memory_bytes"] == 192 * 2**30
            and result["pack_revision"] == REVISION and result["inventory_sha256"] == INVENTORY
            and result["manifest"] == {"model-manifest.json": MANIFEST},
            "Pinned M2 functional control did not complete")
    require(build["mlx"] == build["mlx_metal"] == "0.32.4"
            and build["rustc"].startswith("rustc 1.99.1 ") and build["cargo"].startswith("cargo 1.99.1 ")
            and build["probe_patch_sha256"] == sha256(source).hexdigest()
            and build["driver_sha256"] == sha256((ROOT / "run_probe.py").read_bytes()).hexdigest()
            and build["workload_sha256"] == sha256(workload).hexdigest(),
            "Native probe or workload provenance changed")
    observed = result["arms"]
    require([arm["reader_threads"] for arm in observed] == [1, 32, 32, 1],
            "Reader execution order changed")
    require(all(arm["unique_table_shards"] == 128 and arm["embedding_width"] == 160
                and arm["workload_sha256"] == build["workload_sha256"]
                and arm["pressure_before"] == arm["pressure_after"] == 1 for arm in observed),
            "Unexpected table, workload or pressure")
    roster = [(shape, phase) for shape in SHAPES for phase in PHASES]
    anchors = {}
    summary = []
    for arm in observed:
        require([(stage["shape"], stage["phase"]) for stage in arm["stages"]] == roster,
                "Stage roster changed")
        metrics = {}
        for stage in arm["stages"]:
            shape, phase = stage["shape"], stage["phase"]
            samples = stage["elapsed_ns"]
            expected_samples, expected_rows = SHAPES[shape]
            require(len(samples) == expected_samples
                    and all(type(sample) is int and sample > 0 for sample in samples)
                    and stage["rows_requested"] == expected_rows
                    and 0 < stage["bytes_read"] <= expected_rows * 180,
                    "Missing or invalid native stage measurements")
            signature = (stage["payload_sha256"], stage["rows_requested"], stage["bytes_read"])
            require(anchors.setdefault(shape, signature) == signature,
                    f"Native PLE payload or read volume differs for {shape}")
            metrics[f"{shape}/{phase}"] = {
                "samples": len(samples),
                "p50_ms": round(median(samples) / 1e6, 6),
                "p95_ms": round(quantiles(samples, n=100, method="inclusive")[94] / 1e6, 6),
            }
        summary.append({"reader_threads": arm["reader_threads"], "stages": metrics})
    return {"scope": "native PLE staging functional control on M2 Ultra",
            "qualification": False, "build": build,
            "matching_raw_payloads": {shape: {"sha256": sig[0], "requested_rows": sig[1],
                                               "bytes_read": sig[2]} for shape, sig in anchors.items()},
            "observed": summary}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
