"""Replay the bounded-reference M5 diagnostics without evaluating release readiness."""

import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
THRESHOLDS = ROOT.parents[2] / "scripts" / "flash_next_release_thresholds.json"
SIGNED = ROOT.parent / "qualification/2026-10-24-flash-next-mxfp4-m5max-release-v7.9.0/reference/result.json"
NATIVE = ROOT.parent / "flash-next-mxfp4-native-matrix-m5-20261121/quiet-result.json.gz"
SCHEMA = "ax-engine.flash-next.affine8-ple-diagnostic.v1"
LENGTHS = (512, 2048, 8192)


def encoded(path):
    raw = path.read_bytes()
    return gzip.decompress(raw) if path.suffix == ".gz" else raw


def digest(raw):
    return sha256(raw).hexdigest()


def bind_run(name, manifest_name):
    prefix = "smoke-" if name == "smoke" else ""
    manifest = json.loads((ROOT / manifest_name).read_bytes())
    source = encoded(ROOT / f"{prefix}run_flash_next_affine8_reference.py.gz")
    assert digest(source) == manifest["scripts/run_flash_next_affine8_reference.py"]
    raw = encoded(ROOT / f"{prefix}result.json.gz")
    result = json.loads(raw)
    assert result["schema"] == SCHEMA and result["mode"] == "bounded"
    assert result["completed"] is True and result["phase"] == "complete"
    assert result["qualification"] is False and result["release_ready"] is False
    assert result["backend"] == "metal"
    assert result["inventory_sha256"] == manifest[
        "benchmarks/results/qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json"
    ]
    raw_samples = encoded(ROOT / f"{prefix}memory-samples.jsonl.gz")
    samples = [json.loads(line) for line in raw_samples.splitlines()]
    assert samples and {sample["pressure"] for sample in samples} == {1}
    return result, manifest, {
        "result_sha256": digest(raw),
        "memory_samples_sha256": digest(raw_samples),
        "samplings_at_normal_pressure": len(samples),
        "runner_sha256": digest(source),
    }


def replay():
    raw_signed, raw_native = encoded(SIGNED), encoded(NATIVE)
    signed, native = json.loads(raw_signed), json.loads(raw_native)
    smoke, _, smoke_provenance = bind_run("smoke", "smoke-source.json")
    bounded, sources, provenance = bind_run("full", "source.json")
    assert digest(raw_signed) == sources["signed-m5-reference.json"]
    assert digest(raw_native) == sources["ax-matrix.json"]
    assert smoke["hardware"] == bounded["hardware"]
    assert bounded["hardware"]["os"] == native["hardware"]["os"]
    assert bounded["hardware"]["soc"] == signed["hardware"]["soc"] == native["hardware"]["soc"]
    assert bounded["hardware"]["memory_bytes"] == signed["hardware"]["memory_bytes"] == native["hardware"]["memory_bytes"]
    for arm in (bounded, smoke):
        assert arm["pack_revision"] == signed["revision"]
        assert arm["inventory_sha256"] == signed["inventory_sha256"] == native["inventory_sha256"]
        assert arm["versions"] == signed["versions"]
        assert arm["reference"]["commit"] == signed["reference"]["commit"]
        assert arm["language_sha256"] == signed["language_sha256"]
        assert arm["runtime_files_bound"] == signed["runtime_files_bound"]
    for suffix, key in [
        ("full-qa-items.json", "items_sha256"), ("full-qa-inputs.json", "inputs_sha256"),
        ("closed_checks.py", "closed_checks_sha256"), ("qa_checks.py", "qa_checks_sha256"),
        ("checkers.py", "checkers_sha256"), ("prompt_def.py", "prompt_def_sha256"),
    ]:
        name = f"benchmarks/results/qualification/2026-09-19-flash-next-mxfp4-installed-qa/{suffix}"
        assert sources[name] == bounded["workload_hashes"][key]
    assert bounded["workload_hashes"]["ax_matrix_sha256"] == sources["ax-matrix.json"]
    qa, prior = bounded["qa"], signed["qa"]
    assert len(qa["rows"]) == len(prior["rows"]) == 105
    assert len({r["id"] for r in qa["rows"]}) == 105
    for candidate, baseline in zip(qa["rows"], prior["rows"], strict=True):
        for key in (
            "id", "prompt_tokens", "generated_ids", "text", "normal_stop", "terminal_eos_id",
            "generated_tokens",
        ):
            assert candidate[key] == baseline[key], (candidate["id"], key)
    assert qa["grades"] == prior["grades"] and qa["failures"] == prior["failures"]
    matrix, prior_matrix = bounded["matrix"], signed["matrix"]
    assert matrix["prompt_id_hashes"] == prior_matrix["prompt_id_hashes"]
    assert len(matrix["cells"]) == len(prior_matrix["cells"]) == 15
    for candidate, baseline in zip(matrix["cells"], prior_matrix["cells"], strict=True):
        for key in (
            "length", "run", "warmup", "generated_ids", "normal_stop", "terminal_eos_id",
            "generated_tokens",
        ):
            assert candidate[key] == baseline[key], (candidate["length"], candidate["run"], key)
        assert len(candidate["generated_ids"]) == 128
    assert [a["prompt_tokens"] for a in smoke["cases"]] == list(LENGTHS)
    for a in smoke["cases"]:
        length = a["prompt_tokens"]
        row = next(b for b in prior_matrix["cells"] if b["length"] == length and b["run"] == 0)
        assert a["prompt_ids_sha256"] == prior_matrix["prompt_id_hashes"][f"cell-{length}-0"]
        assert len(a["generated_ids"]) == 32
        assert a["generated_ids"] == row["generated_ids"][:32]
    sys.path.insert(0, str(THRESHOLDS.parent))
    import check_flash_next_release_gates as evaluator

    thresholds = json.loads(THRESHOLDS.read_text())
    metrics = evaluator.gate_reference_relative(native, bounded, thresholds)[0]
    assert len(metrics["cells"]) == 6 and metrics["passed"] is True
    return {
        "diagnostic_only": True,
        "signed_reference_sha256": digest(raw_signed),
        "prior_native_matrix_sha256": digest(raw_native),
        "prior_native_server_sha256": native["server_sha256"],
        "threshold_profile_sha256": digest(THRESHOLDS.read_bytes()),
        "smoke": smoke_provenance,
        "full": provenance,
        "matched_smoke_output_ids": 96,
        "matched_qa_rows_and_grades": 105,
        "qa_passing": 105 - len(qa["failures"]),
        "matched_fixed_matrix_cells": 15,
        "prior_native_comparison_cells": metrics["cells"],
    }


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True))
