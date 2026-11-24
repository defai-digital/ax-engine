#!/usr/bin/env python3
"""Verify one signed Flash Next grouped QSA fault and rollback control."""

from hashlib import sha256
import json
from pathlib import Path
import re
import tarfile


def replay(archive):
    def get(name):
        entry = archive.extractfile(name)
        assert entry is not None
        return entry.read()

    def read(name):
        return json.loads(get(name))

    def digest(name):
        return sha256(get(name)).hexdigest()

    build = read("build.json")
    assert build["profile"] == "release-server" and build["codesign_verified"]
    assert "1.99.1" in build["rustc"] and "1.99.1" in build["cargo"]
    assert build["mlx"] == build["mlx_metal"] == "0.32.4"
    assert digest("source.patch") == build["patch_sha256"]
    sealed = dict(line.split(maxsplit=1)[::-1]
                  for line in get("inputs.sha256").decode().splitlines())
    for name, expected in sealed.items():
        if name != "ax-flash-grouped-qsa-fault-test":
            assert digest(name) == expected
    assert sealed["ax-flash-grouped-qsa-fault-test"] == build["binary_sha256"]
    assert get("status.txt").strip() == b"0"
    assert len(read("mxfp4-inventory.json")["members"]) == 49
    assert len(read("prompt.json")) == 4095
    soc, memory, macos, internal, ssd = get("host.txt").decode().splitlines()
    assert soc == "Apple M5 Max" and int(memory) == 128 * 2**30
    assert internal == ssd == "true"
    receipt = read("observed/receipt.json")
    assert receipt["build"] == build and receipt["qualification"] is False
    assert "grouped_eval_compile_failure_preserves_committed_prefix" in receipt["test"]
    assert receipt["flags"] == {
        "AX_MLX_FLASH_NEXT_QSA_GROUPED_EVAL": "1", "AX_MLX_FLASH_NEXT_QSA_PAIRED_GATHER": "0",
        "AX_MLX_FLASH_NEXT_QSA_INDEXED_PREFILL": "0", "AX_STREAM_EXPERTS": "off",
    }
    assert receipt["manifest"]["model-manifest.json"] == build["manifest_sha256"]
    assert receipt["prompt_sha256"] == digest("prompt.json")
    assert receipt["fault_sha256"] == digest("observed/fault.json")
    assert receipt["samples_sha256"] == digest("observed/native-samples.jsonl")
    proof = read("observed/log-proof.json")
    assert proof["original_sha256"] == receipt["original_test_log_sha256"]
    assert proof["test_result"] == "1 passed; 0 failed"
    assert proof["qsa_stage"] == "grouped prefill attention"
    assert proof["undefined_metal_symbol"] == "ax_grouped_qsa_missing_symbol"
    result = read("observed/fault.json")
    assert result["qualification"] is False
    assert result["failure"] == "second tile in the second evaluation pair"
    assert result["prefix_position_before"] == result["prefix_position_after_failure"] == 2048
    assert result["position_after_retry"] == 4095
    for first, second in (
        ("prefix_sha256_before", "prefix_sha256_after_failure"),
        ("control_state_sha256", "retry_state_sha256"),
        ("control_logits_sha256", "retry_logits_sha256"),
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", result[first])
        assert result[first] == result[second]
    samples = [json.loads(line) for line in get("observed/native-samples.jsonl").splitlines()]
    assert len(samples) == receipt["sample_count"] and samples
    assert all(s["pressure"] == 1 and s["footprint_bytes"] <= 96 * 2**30 for s in samples)
    assert max(s["footprint_bytes"] for s in samples) == receipt["sampled_footprint_peak_bytes"]
    return {"macos": macos, "source_base": build["source_base"], "fault_hook_test_only": True,
            "normal_pressure_samples": len(samples),
            "sampled_footprint_peak_bytes": receipt["sampled_footprint_peak_bytes"], **result}


if __name__ == "__main__":
    with tarfile.open(Path(__file__).with_name("evidence.tar.xz")) as archive:
        print(json.dumps(replay(archive), indent=2, sort_keys=True))
