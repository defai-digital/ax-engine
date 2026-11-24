"""Compare signed Flash Next grouped-QSA server arms."""
from hashlib import sha256
import json
from pathlib import Path


ORDERS = (("stock-first", "grouped-first"), ("stock-second", "grouped-second"),
          ("stock-fallback", "grouped-fallback"))
SWITCH = "AX_MLX_FLASH_NEXT_QSA_GROUPED_EVAL"
GIB = 2**30


def read_json(path):
    return json.loads(path.read_bytes())


def read_lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def ids_sha(ids):
    return sha256(json.dumps(ids).encode()).hexdigest()


def load_arms(root):
    inventory = read_json(root / "mxfp4-inventory.json")
    assert len(inventory["members"]) == 49
    build = read_json(root / "build.json")
    assert build["codesign_verified"] and build["profile"] == "release-server"
    assert build["mlx"] == build["mlx_metal"] == "0.32.4"
    assert "1.99.1" in build["rustc"] and "1.99.1" in build["cargo"]
    assert digest(root / "source.patch") == build["patch_sha256"]
    sealed = dict(line.split(maxsplit=1)[::-1]
                  for line in (root / "inputs.sha256").read_text().splitlines())
    for name, expected in sealed.items():
        if name != "ax-engine-server":
            assert digest(root / name) == expected, name
    assert sealed["ax-engine-server"] == build["binary_sha256"]

    prompts = read_json(root / "prompts.json")
    assert {name: len(ids) for name, ids in prompts.items()} == {
        "short": 512, "long": 8192, "maximum": 32704,
    }
    arms = {}
    for arm in (arm for order in ORDERS for arm in order):
        record = root / "observed" / arm
        fallback = arm.endswith("fallback")
        grouped = arm.startswith("grouped")
        receipt = read_json(record / "receipt.json")
        assert receipt["mode"] == arm and receipt["build"] == build
        assert receipt["qualification"] is False
        assert receipt["pack_sha256"]["model-manifest.json"] == build["manifest_sha256"]
        assert receipt["input_sha256"] == digest(root / "prompts.json")
        assert receipt["requests_sha256"] == digest(record / "requests.jsonl")
        chunk = 512 if fallback else 2048
        assert receipt["chunk"] == chunk
        assert receipt["flags"]["AX_MLX_LONG_PROMPT_PREFILL_CHUNK"] == str(chunk)
        assert receipt["flags"]["AX_MLX_FLASH_NEXT_QSA_PAIRED_GATHER"] == "0"
        assert receipt["flags"][SWITCH] == str(int(grouped))
        model = receipt["model"]
        assert model == dict(id="qwen3.8-flash-next", context_length=32768)
        startup = receipt["startup_memory"]
        placement = [{"id": model["id"], "expert_streaming_active": False}]
        assert startup["models"] == placement
        assert startup["memory_plan"]["wired_shortfall_bytes"] == 0
        rows = read_lines(record / "requests.jsonl")
        expected = ["long"] if fallback else ["short", "long", "maximum"]
        if arm.endswith("second"):
            expected.reverse()
        if not fallback:
            expected += ["disconnect", "recovery"]
        assert [row["action"] for row in rows] == expected, arm
        by_action = {row["action"]: row for row in rows}
        for action, row in by_action.items():
            assert not any(row["resources_after"].values()), (arm, action)
            markers = row["grouped_forwards"]
            assert bool(markers) == (grouped and not fallback and action != "short")
            assert all(width in (2047, 2048) and width < history <= 32768 and tiles > 1
                       for width, history, tiles in markers)
            if action == "disconnect":
                assert row["active_while_open"] == 1
                prefix = row["output_prefix_ids"]
                assert prefix and by_action["long"]["output_ids"][:len(prefix)] == prefix
                continue
            prompt = prompts["long" if action == "recovery" else action]
            assert row["prompt_length"] == len(prompt)
            assert row["prompt_sha256"] == ids_sha(prompt)
            assert len(row["output_ids"]) == 64 and row["finish_reason"] == "max_output_tokens"
            timing = row["client_timing"]
            assert timing["ttft_seconds"] > 0 and timing["total_seconds"] >= timing["ttft_seconds"]
            assert not any(row["route_error_and_mtp_delta"].values())
            memory = row["memory"]
            assert memory["models"] == placement and memory["memory_plan"] == startup["memory_plan"]
            assert memory["lifetime_max_phys_footprint_bytes"] <= 96 * GIB
            if action == "recovery":
                assert row["output_ids"] == by_action["long"]["output_ids"]
        assert [entry for row in rows for entry in row["grouped_forwards"]] == (
            read_json(record / "grouped-markers.json"))
        samples = read_lines(record / "server-samples.jsonl")
        assert len(samples) == receipt["pressure_samples"] and samples
        assert all(sample["pressure"] == 1 and sample["footprint_bytes"] <= 96 * GIB
                   for sample in samples)
        assert max(sample["footprint_bytes"] for sample in samples) == receipt[
            "peak_sampled_footprint_bytes"]
        arms[arm] = receipt, by_action
    return build, arms


def compare(root):
    host = (root / "host.txt").read_text().splitlines()
    assert host[:2] == ["Apple M5 Max", str(128 * GIB)]
    assert (root / "storage.txt").read_text().splitlines() == ["true", "true"]
    build, arms = load_arms(root)
    paired_non_recovery_ids = 0
    results = []
    for stock, grouped in ORDERS:
        receipt_a, rows_a = arms[stock]
        receipt_b, rows_b = arms[grouped]
        assert receipt_a["startup_memory"]["memory_plan"] == receipt_b["startup_memory"]["memory_plan"]
        work = {}
        for label, row_a in rows_a.items():
            if label == "disconnect":
                continue
            row_b = rows_b[label]
            assert row_a["output_ids"] == row_b["output_ids"], (stock, grouped, label)
            if label != "recovery":
                paired_non_recovery_ids += len(row_a["output_ids"])
            work[label] = {
                "output_ids_sha256": ids_sha(row_a["output_ids"]),
                "stock_ttft_seconds": row_a["client_timing"]["ttft_seconds"],
                "grouped_ttft_seconds": row_b["client_timing"]["ttft_seconds"],
                "stock_idle_percent": row_a["request_host_idle_percent"],
                "grouped_idle_percent": row_b["request_host_idle_percent"],
                "grouped_forwards": len(row_b["grouped_forwards"]),
                "grouped_tiles": sum(mark[2] for mark in row_b["grouped_forwards"]),
            }
        def peak(receipt, rows):
            return max(receipt["startup_memory"]["mlx"]["peak_bytes"],
                       *(row["memory"]["mlx"]["peak_bytes"] for row in rows.values()
                         if "memory" in row))
        results.append({
            "stock": stock, "grouped": grouped, "chunk": receipt_a["chunk"], "workloads": work,
            "stock_peak_mlx_bytes": peak(receipt_a, rows_a),
            "grouped_peak_mlx_bytes": peak(receipt_b, rows_b),
            "stock_sampled_footprint_bytes": receipt_a["peak_sampled_footprint_bytes"],
            "grouped_sampled_footprint_bytes": receipt_b["peak_sampled_footprint_bytes"],
            "samples": receipt_a["pressure_samples"] + receipt_b["pressure_samples"],
        })
    m4 = root / "m4"
    host_m4 = (m4 / "host.txt").read_text().splitlines()
    assert host_m4[:2] == ["Apple M4 Pro", str(64 * GIB)] and host_m4[3:] == ["1", "1"]
    for line in (m4 / "source.sha256").read_text().splitlines():
        expected, name = line.split()
        assert digest(m4 / name) == expected
    synthetic = []
    for row in read_json(m4 / "result.json"):
        assert row["exact_bf16"] and row["selected"] == 2051
        measurements = [sample for group in row["measurements"].values() for sample in group]
        assert len(measurements) == 12
        assert len({sample["output_sha256"] for sample in measurements}) == 1
        for group, samples in row["measurements"].items():
            assert max(sample["peak_bytes"] for sample in samples) == row["peak_bytes_by_group"][group]
        synthetic.append({key: row[key] for key in ("queries", "history", "peak_bytes_by_group")})
    assert [(row["queries"], row["history"]) for row in synthetic] == [
        (512, 8192), (2048, 8192), (512, 32768), (2048, 32768),
    ]
    wide_ids = arms["stock-first"][1]["long"]["output_ids"]
    standard_ids = arms["stock-fallback"][1]["long"]["output_ids"]
    geometry_differences = [index for index, (wide, ordinary) in enumerate(
        zip(wide_ids, standard_ids, strict=True)) if wide != ordinary]
    return {"qualification": False, "source_base": build["source_base"],
            "m5_macos": host[2], "m4_macos": host_m4[2], "m4_exact_component": synthetic,
            "long_512_vs_2048_output_ids_match": not geometry_differences,
            "long_512_vs_2048_differing_output_indices": geometry_differences,
            "runtime_patch_sha256": build["patch_sha256"], "orders": results,
            "matched_output_ids_excluding_recovery": paired_non_recovery_ids}



if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    import tarfile

    with TemporaryDirectory() as temp, tarfile.open(
        Path(__file__).with_name("evidence.tar.xz")
    ) as archive:
        archive.extractall(temp, filter="data")
        print(json.dumps(compare(Path(temp)), indent=2, sort_keys=True))
