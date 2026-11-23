"""Replay exact native prompt and output arrays in the guarded-reader M5 matrix."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
FRESH_SHA = "9e5eaff70d2d365c3f49ba7c1809b121aec5d447ccda2bfed14f826e4ed8230b"
PRIOR_SHA = "9f5961ecd33d2314da3329f34659bd36e39584e16d4582ba8570c83ed5f2c7b6"
MODES = ("disabled", "default")
LAUNCHES = ("warm", *(f"{mode}-{index}" for mode in MODES for index in (1, 2)))
MEMBERS = {"result.json", "raw-runs.json", *(f"server-{label}.log" for label in LAUNCHES)}


def member(archive, name):
    entry = archive.getmember(name)
    assert entry.isfile(), name
    with archive.extractfile(entry) as stream:
        return stream.read()


def sealed(bench, name, digest):
    archive = bench / name / "evidence.tar.xz"
    assert sha256(archive.read_bytes()).hexdigest() == digest
    return tarfile.open(archive)


def trajectories(run):
    return [row["output_tokens"] for row in (*run["cells"], *run["trajectories"])]


def replay(repo, archive):
    bench = repo / "benchmarks/results"
    sys.path.insert(0, str(repo / "scripts"))
    import check_flash_next_release_gates as gates

    with (tarfile.open(archive) as current,
          sealed(bench, "flash-next-guarded-reader-fresh-m5-20261123", FRESH_SHA) as fresh,
          sealed(bench, "flash-next-probe-followup-matrix-repeat-m5-20261123", PRIOR_SHA) as old):
        names = current.getnames()
        assert set(names) == MEMBERS and len(names) == len(MEMBERS)
        data = {name: member(current, name) for name in names}
        delivery = json.loads(member(fresh, "delivery/result.json"))
        manifest = {name: digest for digest, name in map(str.split, member(fresh, "inputs.sha256").decode().splitlines())}
        prior = json.loads(member(old, "raw/observed/generate-matrix/result.json"))
    assert all(token not in value for value in data.values() for token in (b"/Users/", b"/private/tmp/", b"127.0.0.1"))
    for name in ("run_flash_next_generate_matrix.py", "check_flash_next_release_gates.py", "flash_next_release_thresholds.json"):
        path = f"scripts/{name}"
        assert sha256((repo / path).read_bytes()).hexdigest() == manifest[path]
    result, repeats = (json.loads(data[name], object_pairs_hook=gates.unique_json_members)
                       for name in ("result.json", "raw-runs.json"))
    assert all(result[name] == delivery[name] for name in ("hardware", "revision", "inventory_sha256", "server_sha256"))
    assert result["server_sha256"] == manifest["ax-engine-server"]
    assert result["ordinary_launch"] is True and result["total_blocks"] == 2048
    assert result["qualification"] is False and result["release_ready"] is False
    assert result["modes"] == list(MODES) and result["repeats_per_arm"] == 2
    assert result["lengths"] == list(gates.LENGTHS) and result["output_tokens"] == gates.OUTPUT_TOKENS
    assert result["warmups"] == gates.WARMUPS and result["measured"] == gates.MEASURED
    assert result["trajectory_prompts"] == gates.TRAJECTORIES
    for log_path in MEMBERS - {"result.json", "raw-runs.json"}:
        log = re.sub(r"\x1b\[[0-9;]*m", "", data[log_path].decode())
        assert "decision=resident" in log and "context_tokens=32768 effective_sequence_cap=Some(1)" in log, log_path
    assert not gates.matrix_repeat_problems(result, repeats)
    metrics, exact_responses, differences = {}, 0, []
    for mode in MODES:
        selected = result["runs"][mode]
        earlier = prior["runs"][mode]
        assert selected["prompt_id_hashes"] == earlier["prompt_id_hashes"]
        older_outputs = trajectories(earlier)
        for index, run in enumerate(repeats[mode], 1):
            assert run["model"]["context_length"] == 32768
            assert run["prompt_id_hashes"] == earlier["prompt_id_hashes"]
            for row, (current, before) in enumerate(zip(trajectories(run), older_outputs, strict=True)):
                if current == before:
                    exact_responses += 1
                else:
                    differences.append({"mode": mode, "repeat": index, "workload_index": row})
        per_length = {}
        for length in gates.LENGTHS:
            now = gates.cell_stats([row for row in selected["cells"] if row["length"] == length], streamed=True)
            then = gates.cell_stats([row for row in earlier["cells"] if row["length"] == length], streamed=True)
            assert now["full_length"] == now["measured"] == gates.MEASURED
            assert now["decode_tok_s"] is not None and now["ttft_s"] is not None
            per_length[str(length)] = {"ttft_seconds": now["ttft_s"], "decode_tokens_per_second": now["decode_tok_s"],
                                       "ttft_ratio_to_prior_native": now["ttft_s"] / then["ttft_s"],
                                       "decode_ratio_to_prior_native": now["decode_tok_s"] / then["decode_tok_s"]}
        metrics[mode] = per_length
    assert exact_responses + len(differences) == 124
    idle = [result["host_quiet"]["cpu_idle_percent"], result["host_idle_after"]]
    thresholds = json.loads(gates.THRESHOLDS.read_bytes())
    readiness, = gates.gate_readiness(delivery, [(mode, result["runs"][mode]["ready_seconds"]) for mode in MODES], thresholds)
    return {"scope": "Guarded-reader M5 native generation matrix", "release_ready": False,
            "server_sha256": result["server_sha256"], "matrix_sha256": sha256(data["result.json"]).hexdigest(),
            "raw_repeats_sha256": sha256(data["raw-runs.json"]).hexdigest(), "matching_prior_native_arrays": exact_responses,
            "prior_native_differences": differences, "cells": metrics, "repeat_token_identity": result["repeat_token_identity"],
            "timing_cpu_idle_percent": idle, "timing_eligible": all(thresholds["timing_host"]["min_cpu_idle_percent"] <= value <= 100 for value in idle),
            "readiness": readiness}


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--repo", type=Path, default=ROOT.parents[2])
    cli.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    args = cli.parse_args()
    print(json.dumps(replay(args.repo.resolve(), args.archive), sort_keys=True, indent=2, allow_nan=False))
