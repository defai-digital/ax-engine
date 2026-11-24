"""Replay the limited Flash Next ordinary reference startup diagnosis."""

from hashlib import sha256
from itertools import zip_longest
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIGNED = HERE.parent / "qualification/2026-10-24-flash-next-mxfp4-m5max-release-v7.9.0/reference/result.json"
IDENTITY = ("id", "prompt_tokens", "text", "generated_ids", "terminal_eos_id", "normal_stop")


def replay():
    raw = HERE / "raw"
    signed_bytes = SIGNED.read_bytes()
    assert sha256(signed_bytes).hexdigest() == "2eeeb8ab3b5f329726550b675ec61d57edeab6558a582582d75287983d4a19b7"
    signed = json.loads(signed_bytes)
    probe = json.loads((raw / "result.json").read_bytes())
    control = json.loads((raw / "original-control.json").read_bytes())
    inputs = json.loads((raw / "run-inputs.json").read_bytes())
    adapter = (raw / "original-control-adapter.py").read_bytes()
    events = [json.loads(line) for line in (raw / "startup-memory.jsonl").read_bytes().splitlines()]
    assert inputs["original_reference_m2_adapter"] == sha256(adapter).hexdigest()
    assert inputs["pinned_pack_inventory"] == probe["inventory_sha256"]
    assert probe["diagnostic_only"] and probe["completed"]
    assert probe["qualification"] is probe["release_ready"] is False
    assert probe["phases"] == ["qa"] and "matrix" not in probe
    assert probe["hardware"]["soc"] == "Apple M2 Ultra"
    assert probe["hardware"]["memory_bytes"] == 192 * 2**30
    assert probe["hardware"]["os"] == "27.2.1"
    assert signed["hardware"]["os"] == "27.0.1"
    assert probe["hardware"]["storage"]["solid_state"]
    for key in ("repo_id", "revision", "reference", "prefill_chunk", "memory_limit_bytes",
                "cache_limit_bytes", "temperature", "qa_budgets"):
        assert probe[key] == control[key] == signed[key], key
    assert control["completed"] and control["qualification"] is control["release_ready"] is False
    assert control["phases"] == ["qa"] and "matrix" not in control
    for key in ("hardware", "inventory_sha256", "runtime_files_bound", "weights_materialized"):
        assert probe[key] == control[key], key
    assert len(probe["qa"]["rows"]) == probe["qa_items"] == 4
    assert probe["weights_materialized"] == signed["weights_materialized"] == 2326

    stamps = [event["monotonic_ns"] for event in events]
    assert all(a < b for a, b in zip(stamps, stamps[1:]))
    assert all(event["system_pressure_level"] == 1 for event in events)
    assert all("stop_reason" not in event for event in events)
    assert [event["phase"] for event in events[:3]] == [
        "runtime-initialized", "lazy-load-start", "lazy-load-complete"]
    weights = [event for event in events if event["phase"] == "weights-batch-complete"]
    assert [event["arrays_materialized"] for event in weights] == [*range(32, 2326, 32), 2326]
    ready = next(event for event in events if event["phase"] == "weights-ready")
    assert ready["arrays_materialized"] == 2326
    controls = []
    for row, stock, oracle in zip(
        probe["qa"]["rows"], control["qa"]["rows"], signed["qa"]["rows"][:4], strict=True,
    ):
        case = row["id"]
        grade = probe["qa"]["grades"][case]
        assert grade["passed"] and stock["id"] == oracle["id"] == case
        divergence = next((i for i, pair in enumerate(
            zip_longest(row["generated_ids"], oracle["generated_ids"]),
        ) if pair[0] != pair[1]), None)
        sequence = [event for event in events if event.get("case_id") == case]
        prompt_rows = row["prompt_tokens"] - 1
        assert [event["phase"] for event in sequence] == [
            "prefill-start", "prefill-chunk-complete", "first-token-start",
            "first-token-ready", "case-complete"]
        assert sequence[1]["prefill_tokens"] == prompt_rows
        assert sequence[-1]["generated_tokens"] == len(row["generated_ids"])
        controls.append({
            "id": case,
            "matched_original_m2": all(row[key] == stock[key] for key in IDENTITY)
            and grade == control["qa"]["grades"][case],
            "matches_signed": all(row[key] == oracle[key] for key in IDENTITY)
            and grade == signed["qa"]["grades"][case],
            "first_signed_token_difference_index": divergence,
            "generated_ids": len(row["generated_ids"]),
        })

    first = next(event for event in events if event["phase"] == "first-token-ready")
    return {
        "diagnostic_only": True,
        "diagnostic_macos": probe["hardware"]["os"],
        "signed_reference_macos": signed["hardware"]["os"],
        "cases": controls,
        "matching_original_m2_cases": sum(row["matched_original_m2"] for row in controls),
        "matching_signed_cases": sum(row["matches_signed"] for row in controls),
        "weights_materialized": ready["arrays_materialized"],
        "weight_batches": len(weights),
        "normal_pressure_samples": len(events),
        "mlx_active_after_lazy_load_bytes": events[2]["mlx_active_bytes"],
        "mlx_active_weights_ready_bytes": ready["mlx_active_bytes"],
        "mlx_active_first_token_bytes": first["mlx_active_bytes"],
        "mlx_peak_weights_ready_bytes": ready["mlx_peak_bytes"],
        "mlx_peak_first_token_bytes": first["mlx_peak_bytes"],
        "mlx_peak_four_cases_bytes": probe["peak_mlx_memory_bytes"],
        "process_peak_rss_bytes": probe["peak_process_rss_bytes"],
    }


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True))
