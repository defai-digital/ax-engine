"""Verify the caller-resumed Flash Next forward against the prior M5 probe."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import runpy
from statistics import median


ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "flash-next-prefill-layer-peer-m5-20261122"


def main():
    controls = runpy.run_path(str(PRIOR / "replay.py"))
    require, seconds = controls["require"], controls["seconds"]
    prior = json.loads((PRIOR / "receipt.json").read_bytes())
    prior_bytes = (PRIOR / "result.json").read_bytes()
    receipt = json.loads((ROOT / "receipt.json").read_bytes())
    raw = (ROOT / "result.json").read_bytes()
    result = json.loads(raw)
    require(sha256(prior_bytes).hexdigest() == prior["result_sha256"]
            and sha256(raw).hexdigest() == receipt["result_sha256"], "Result bytes changed")
    for field in ("hardware", "pack_revision", "native_manifest", "inventory_sha256", "prompt_sha256", "test"):
        require(receipt[field] == prior[field], f"Changed benchmark input: {field}")
    require(receipt["hardware"]["soc"] == "Apple M5 Max" and receipt["hardware"]["memory_bytes"] == 128 << 30
            and receipt["status"] == "complete" and receipt["test_exit"] == 0
            and (ROOT / "status.txt").read_bytes() == b"0\n"
            and receipt["qualification"] is result["qualification"] is False
            and receipt["pressure_samples"] and all(level == 1 for level in receipt["pressure_samples"]),
            "Incomplete primary hardware run")
    build = receipt["build"]
    for field in ("rustc", "cargo", "mlx", "mlx_metal", "profile"):
        require(build[field] == prior["build"][field], f"Changed test environment: {field}")
    checksums = {name: digest for digest, name in
                 (line.split() for line in (ROOT / "inputs.sha256").read_text().splitlines())}
    require(checksums["layer-peer-probe"] == build["executable_sha256"] != prior["build"]["executable_sha256"]
            and checksums["prompt-8192.json"] == receipt["prompt_sha256"]
            and checksums["inventory.json"] == receipt["inventory_sha256"], "Test executable or inputs unbound")
    require(result["schema"] == "ax-engine.flash-next.prefill-transaction.v1"
            and result["both_states_and_logits_exact"] is True
            and result["aborted_forward_preserves_original_state"] is True,
            "Missing completed or aborted transaction proof")
    prior_result = json.loads(prior_bytes)
    for field in ("peer_outputs", "peer_prompt_tokens", "long_widths", "peer_state_sha256", "long_state_sha256"):
        require(result[field] == prior_result[field], f"Previous M5 control differs: {field}")
    cuts = result["cuts"]
    require([(row["offset"], row["width"], row["layers"]) for row in cuts] == controls["CUTS"]
            and all(seconds(row["advance_seconds"]) and seconds(row["peer_seconds"]) for row in cuts),
            "An advance changed its geometry or did not complete work")
    for field in ("solo_chunks", "interleaved_chunks"):
        actual = result[field]
        baseline = prior_result[field]
        require(len(actual) == len(baseline)
                and all(a[:2] == b[:2] and seconds(a[2]) for a, b in zip(actual, baseline, strict=True)),
                "Physical forward geometry changed")
    idle = [receipt[f"{phase}_idle_percent"] for phase in ("before", "after")]
    require(all(type(value) in (int, float) and isfinite(value) and 0 <= value <= 100 for value in idle),
            "Invalid idle samples")
    print(json.dumps({"qualification": False, "layer_advances": len(cuts), "exact_prompt_and_peer_states": True,
                      "aborted_forward_preserves_original_state": True, "cpu_idle_percent": idle,
                      "median_advance_ms": 1000 * median(row["advance_seconds"] for row in cuts),
                      "max_advance_ms": 1000 * max(row["advance_seconds"] for row in cuts),
                      "median_peer_decode_ms": 1000 * median(row["peer_seconds"] for row in cuts)},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
