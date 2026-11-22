"""Replay the current native Flash Next matrix from raw process records."""
from gzip import decompress
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
DELIVERY = ROOT.parent / "flash-next-mxfp4-current-delivery-m5-20261122"
SCRIPTS = ROOT.parents[2] / "scripts"

sys.path.insert(0, str(SCRIPTS))
import check_flash_next_release_gates as gate  # noqa: E402
import run_flash_next_generate_matrix as runner  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode(raw):
    return json.loads(raw, object_pairs_hook=gate.unique_json_members)


def observe(result, repeats):
    modes = tuple(gate.AX_ARMS)
    require(result["modes"] == list(modes) and result["repeats_per_arm"] == gate.AX_REPEATS,
            "Unexpected native arm roster")
    require(result["ax_server_env"] == {} and result["qualification"] is False
            and result["release_ready"] is False and result["mtp_verifier"] == "canonical",
            "The native matrix must use the frozen serving options")
    require(tuple(result["lengths"]) == runner.LENGTHS and result["warmups"] == runner.WARMUPS
            and result["measured"] == runner.MEASURED and result["output_tokens"] == runner.OUTPUT_TOKENS
            and result["workload_seed"] == runner.WORKLOAD_SEED
            and result["trajectory_prompts"] == len(runner.PROMPTS),
            "Unexpected native workload contract")
    require(not gate.matrix_repeat_problems(result, repeats),
            "Native process repeat verification failed")
    identity = runner.repeat_token_identity(repeats)
    require(identity == result["repeat_token_identity"], "Repeat summary disagrees with raw outputs")
    require(runner.repeat_ttft_medians(repeats) == result["repeat_ttft_medians"],
            "Repeat TTFT summary disagrees with raw outputs")
    selected = result["runs"]
    summary = runner.summarize(selected)
    require(summary == result["summary"] and summary["route_errors"] == 0,
            "Native arm aggregate is inconsistent or has route errors")

    run_count = runner.WARMUPS + runner.MEASURED
    expected_cell_keys = [(length, index, index < runner.WARMUPS)
                          for length in runner.LENGTHS for index in range(run_count)]
    for run in [result["warm_pass"], *(row for items in repeats.values() for row in items)]:
        hashes = run["prompt_id_hashes"]
        vocab = run["vocab_size"]
        require([(row["length"], row["run"], row["warmup"]) for row in run["cells"]]
                == expected_cell_keys, "Unexpected fixed-cell order")
        expected = {f"cell-{length}-{index}": runner.ids_sha(runner.workload_ids(vocab, length, index))
                    for length, index, _ in expected_cell_keys}
        prompts = () if run is result["warm_pass"] else runner.PROMPTS
        require([row["prompt"] for row in run["trajectories"]] == list(prompts),
                "Unexpected trajectory prompt set")
        trajectories = {f"traj-{i:02d}" for i in range(len(prompts))}
        require(hashes.keys() == expected.keys() | trajectories
                and all(hashes[key] == value for key, value in expected.items())
                and len({hashes[key] for key in trajectories}) == len(prompts)
                and all(gate.is_sha256(hashes[key]) for key in trajectories),
                "Unexpected prompt digests")
        require(all(row["route_errors"] == row["verified_steps"] == 0
                    for row in (*run["cells"], *run["trajectories"])),
                "A raw row used MTP or recorded an error")
    first, second = (selected[mode] for mode in modes)
    require(first["prompt_id_hashes"] == second["prompt_id_hashes"], "Mode prompt digests differ")
    cross_identity = runner.repeat_token_identity({"cross": [first, second]})["cross"]
    return identity, cross_identity, summary


def replay():
    manifest = json.loads((DELIVERY / "comparison.json").read_text())
    with tarfile.open(DELIVERY / "evidence.tar.xz") as known:
        files = dict(line.split()[::-1] for line in known.extractfile("inputs.sha256").read().decode().splitlines())
        raw_delivered = known.extractfile("native-support.json").read()
    require(sha256(raw_delivered).hexdigest() == manifest["native_support_sha256"],
            "Fresh delivery record mismatch")
    delivered = decode(raw_delivered)
    for name in ("run_flash_next_generate_matrix.py", "check_flash_next_release_gates.py",
                 "flash_next_release_thresholds.json"):
        require(sha256((SCRIPTS / name).read_bytes()).hexdigest() == files[f"scripts/{name}"],
                f"Staged code checksum mismatch for {name}")
    thresholds = decode(gate.THRESHOLDS.read_bytes())
    result_raw = decompress((ROOT / "result.json.gz").read_bytes())
    repeats_raw = decompress((ROOT / "raw-runs.json.gz").read_bytes())
    result, repeats = map(decode, (result_raw, repeats_raw))
    require(all(result[key] == delivered[key] for key in ("server_sha256", "inventory_sha256", "hardware", "revision")),
            "Native matrix differs from the fresh delivery build, pack or host")
    identity, cross, summary = observe(result, repeats)
    floor = thresholds["timing_host"]["min_cpu_idle_percent"]
    idle = [result["host_quiet"]["cpu_idle_percent"], result["host_idle_after"]]
    return {"scope": "Current native direct matrix only", "release_ready": False,
            "result_sha256": sha256(result_raw).hexdigest(),
            "raw_runs_sha256": sha256(repeats_raw).hexdigest(),
            "server_sha256": result["server_sha256"],
            "timing_cpu_idle_percent": idle,
            "timing_cpu_idle_gate": all(type(i) in (int, float) and floor <= i <= 100 for i in idle),
            "repeat_identity": identity, "cross_mode_identity": cross,
            "fixed_workloads": summary["cells"], "route_errors": summary["route_errors"]}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
