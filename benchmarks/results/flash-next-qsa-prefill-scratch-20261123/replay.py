#!/usr/bin/env python3
"""Check QSA allocation observations and replay the 128 GiB plan comparison."""

from hashlib import sha256
import json
import lzma
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
GIB = 1 << 30
MIB = 1 << 20


def selected_attention_reservation(row):
    q = row["queries"]
    heads = row["query_heads"]
    kv_heads = row["kv_heads"]
    dim = row["head_dim"]
    selected = row["selected"]
    assert (heads, kv_heads, dim, selected, row["history"]) == (24, 2, 256, 2051, 8192)
    copies = 4 if row["transpose_gather"] else 2
    partitions = max(128, row["mlx_sdpa_blocks"])
    buffers = {
        "gathered_kv": q * selected * kv_heads * dim * 2 * copies,
        "indices_and_masks": q * selected * (4 * (4 + 2 * kv_heads) + 1 + heads),
        "query_and_output": q * heads * dim * 2 * 2,
        "sdpa_partials": q * heads * partitions * (dim * 2 + 8),
    }
    return sum(buffers.values())


def allocation_observations():
    cells = {}
    for cell in ("128", "256", "511", "512", "1024", "2048", "override", "transpose"):
        row = json.loads((HERE / f"observed-{cell}.json").read_text())
        assert row["all_attention_outputs_exact"] is True
        assert row["one_tile_byte_cap"] == GIB
        assert row["mlx_sdpa_blocks"] == (160 if cell == "override" else 0)
        assert row["transpose_gather"] == (cell == "transpose")
        arms = row["modes"]
        assert [arm["mode"] for arm in arms] == ["lazy"] * 4 + ["serial-eval", "gather-only"]
        assert [arm["sub_batch_queries"] for arm in arms] == [255, 64, 128, 192, 255, 255]
        digests = {arm["output_sha256"] for arm in arms if arm["mode"] != "gather-only"}
        assert len(digests) == 1 and re.fullmatch(r"[a-f0-9]{64}", next(iter(digests)))
        reserve = selected_attention_reservation(row)
        stock_peak = max(arms[0]["peaks_bytes"])
        all_peaks = [max(arm["peaks_bytes"]) for arm in arms]
        assert row["old_gather_charge_bytes"] < stock_peak
        assert reserve >= max(all_peaks), cell
        cells[cell] = {
            "queries": row["queries"],
            "stock_peak_bytes": stock_peak,
            "all_tiles_observed_peak_bytes": max(all_peaks),
            "serial_peak_bytes": all_peaks[4],
            "gather_only_peak_bytes": all_peaks[-1],
            "previous_gather_estimate_bytes": row["old_gather_charge_bytes"],
            "corrected_attention_estimate_bytes": reserve,
            "separate_indexer_reserve_bytes": 256 * MIB,
            "bitwise_equal_within_cell": True,
        }
    return cells


def read_plans(path):
    fields_to_keep = ("prefill-scratch", "working-set", "ceiling", "live-sequence-cap", "decision")
    plans = {}
    for line in path.read_text().splitlines():
        if ": memory plan: " not in line:
            continue
        label, values = line.split(": memory plan: ", 1)
        fields = dict(re.findall(r"(?<![\w-])([a-z-]+)=([^ )]+)", values))
        plans[label] = {name: fields[name] for name in fields_to_keep}
    return plans


def native_control():
    after = json.loads((HERE / "native-receipt.json").read_text())
    build = json.loads((HERE / "build.json").read_text())
    assert after["build"] == build and build["codesign_verified"] is True
    assert build["patch_sha256"] == sha256(lzma.decompress((HERE / "runtime.patch.xz").read_bytes())).hexdigest()
    assert after["model"]["context_length"] == 32768
    assert (after["model"]["id"], after["mode"], after["qualification"]) == (
        "qwen3.8-flash-next", "stock-first", False)
    streams = []
    for arm in ("before", "after"):
        raw = (HERE / f"native-{arm}-requests.jsonl").read_bytes()
        if arm == "after":
            assert sha256(raw).hexdigest() == after["requests_sha256"]
        rows = [json.loads(line) for line in raw.splitlines()]
        assert [r["prompt_length"] for r in rows] == [512, 2048, 8192, 8192]
        for row in rows:
            assert len(row["output_tokens"]) == 64
            assert row["finish_reason"] == "max_output_tokens"
            assert len(row["resources_after"]) == 5 and not any(row["resources_after"].values())
            for key in ("ax_engine_mlx_flash_next_qsa_paired_gather_tiles_total",
                        "ax_engine_flash_next_mtp_verified_steps_total"):
                assert row["metrics_delta"][key] == 0
        streams.append(rows)
    matched = 0
    for a, b in zip(*streams, strict=True):
        assert (a["prompt"], a["prompt_sha256"]) == (b["prompt"], b["prompt_sha256"])
        assert a["output_tokens"] == b["output_tokens"]
        matched += len(a["output_tokens"])
    samples = [json.loads(line) for line in (HERE / "native-after-pressure.jsonl").read_text().splitlines()]
    assert len(samples) == after["pressure_samples"] and all(row["pressure"] == 1 for row in samples)
    assert max(row["footprint_bytes"] for row in samples) == after["peak_footprint_bytes"] <= 96 * GIB
    assert after["peak_rss_bytes"] == max(row["rss_bytes"] for row in samples)
    startup = read_plans(HERE / "native-memory.txt")
    assert all(plan["decision"] == "resident" for plan in startup.values())
    assert startup["before"]["prefill-scratch"] == "1.69GiB"
    assert startup["after"]["prefill-scratch"] == "12.02GiB"
    return {
        "matched_output_ids": matched, "one_slot_context_tokens": after["model"]["context_length"],
        "after_normal_pressure_samples": len(samples), "after_peak_footprint_bytes": after["peak_footprint_bytes"],
        "after_peak_rss_bytes": after["peak_rss_bytes"], "observed": startup,
    }


def main():
    allocations = allocation_observations()
    before = read_plans(HERE / "projected-before.txt")
    after = read_plans(HERE / "projected-after.txt")
    assert len(before) == len(after) == 3 and before.keys() == after.keys()
    for name, old in before.items():
        new = after[name]
        assert old["decision"] == "resident"
        assert new["decision"] == ("resident" if "MXFP4" in name else "stream")
        assert old["ceiling"] == new["ceiling"] == "115.20GiB"
        assert old["prefill-scratch"] == "1.69GiB" and new["prefill-scratch"] == "12.02GiB"
    assert before["Flash Next 6bit (default cohort)"]["live-sequence-cap"] == "1"
    assert after["Flash Next MXFP4 (default cohort)"]["live-sequence-cap"] == "uncapped"
    print(json.dumps({
        "qualification": False,
        "m2_native": native_control(),
        "allocations": allocations,
        "predicted_128_gib_placement": {"before": before, "after": after},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
